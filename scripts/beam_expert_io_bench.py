#!/usr/bin/env python3
"""Measure the SSD read bandwidth that bounds Beam-1.0-501B on Apple Silicon.

Why this number decides everything
----------------------------------
Beam-1.0-501B-A23B activates 8 of 256 routed experts in each of its 49 routed
layers, so a single decoded token needs 392 expert records. In the only
Apple-Silicon pack that exists
(OpensourceWTF/Beam-1.0-501B-A23B-NVFP4-q8-mlx-stream) a record is
22,118,412 B of NVFP4 payload on a 22,122,496 B (4 KiB-aligned) stride, so a
token touches 8.67 GB of expert weights. No Mac below ~512 GB can hold all
12,544 records (277.5 GB), so the misses come off the SSD and **SSD bandwidth,
not compute, sets the token rate**.

That makes one hardware number the coefficient of the whole feasibility
curve: what read bandwidth does an Apple NVMe actually deliver at THIS record
size, at the queue depths a streaming engine can issue? Published figures are
sequential-read marketing numbers, which is the wrong access pattern.

What this measures
------------------
A file of `--size-gb` is written once, then read back in 22,122,496 B records
at randomly chosen aligned offsets, with `--threads` concurrent `pread`s (the
shape dmlx's expert io_pool and mlx-stream's expert bank both use). Page cache
is evicted between passes (`--purge`, needs sudo) so the numbers are the
device's, not the cache's.

Reports GB/s per queue depth, and converts it into the quantity a reader
cares about: ms per token and tok/s for a given cache hit rate.

Usage:
    python3 scripts/beam_expert_io_bench.py --size-gb 24 --purge
"""
from __future__ import annotations

import argparse
import os
import random
import statistics as st
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# The pack's real geometry (expert-manifest.json, mlx-stream-nvfp4-expert-bank-v1).
RECORD_PAYLOAD = 22_118_412
RECORD_STRIDE = 22_122_496
ROUTED_LAYERS = 49
TOPK = 8
RECORDS_PER_TOKEN = ROUTED_LAYERS * TOPK          # 392
BYTES_PER_TOKEN = RECORDS_PER_TOKEN * RECORD_STRIDE


def make_file(path: Path, size_gb: float) -> int:
    want = int(size_gb * 1e9) // RECORD_STRIDE * RECORD_STRIDE
    if path.exists() and path.stat().st_size >= want:
        return path.stat().st_size // RECORD_STRIDE
    print(f"  writing {want/1e9:.1f} GB test file (one time) ...", flush=True)
    chunk = os.urandom(RECORD_STRIDE)
    with path.open("wb") as f:
        written = 0
        while written < want:
            f.write(chunk)
            written += len(chunk)
        f.flush()
        os.fsync(f.fileno())
    return want // RECORD_STRIDE


def purge() -> bool:
    """Evict the page cache so reads hit the device."""
    r = subprocess.run(["sudo", "-n", "purge"], capture_output=True)
    return r.returncode == 0


def read_pass(path: Path, n_records: int, threads: int, reads: int,
              rng: random.Random) -> float:
    """Return GB/s for `reads` random aligned record reads at `threads` depth."""
    offsets = [rng.randrange(n_records) * RECORD_STRIDE for _ in range(reads)]
    fds = [os.open(path, os.O_RDONLY) for _ in range(threads)]
    try:
        def work(i_off):
            i, off = i_off
            os.pread(fds[i % threads], RECORD_PAYLOAD, off)
        t0 = time.perf_counter()
        if threads == 1:
            for item in enumerate(offsets):
                work(item)
        else:
            with ThreadPoolExecutor(max_workers=threads) as ex:
                list(ex.map(work, enumerate(offsets), chunksize=1))
        dt = time.perf_counter() - t0
    finally:
        for fd in fds:
            os.close(fd)
    return reads * RECORD_PAYLOAD / dt / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/tmp/beam_io_bench.bin")
    ap.add_argument("--size-gb", type=float, default=24.0,
                    help="test file size; must exceed RAM-resident cache to "
                         "avoid measuring the page cache")
    ap.add_argument("--reads", type=int, default=240)
    ap.add_argument("--threads", default="1,2,4,8,16,32")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--purge", action="store_true",
                    help="sudo purge between passes (recommended)")
    args = ap.parse_args()

    path = Path(args.path)
    print(f"record: {RECORD_PAYLOAD:,} B payload on a {RECORD_STRIDE:,} B stride")
    print(f"token:  {RECORDS_PER_TOKEN} records = {BYTES_PER_TOKEN/1e9:.2f} GB "
          f"(8 of 256 experts x {ROUTED_LAYERS} routed layers)\n")
    n = make_file(path, args.size_gb)
    if n < 16:
        print("test file too small", file=sys.stderr)
        return 2
    print(f"  file holds {n:,} records ({n*RECORD_STRIDE/1e9:.1f} GB)\n")

    if args.purge and not purge():
        print("  ! sudo purge unavailable — numbers may include page cache\n")

    rng = random.Random(1234)
    depths = [int(x) for x in args.threads.split(",") if x.strip()]
    best: dict[int, float] = {}
    print(f"  {'depth':>6} {'GB/s (median of %d)' % args.reps:>22}")
    for d in depths:
        runs = []
        for _ in range(args.reps):
            if args.purge:
                purge()
            runs.append(read_pass(path, n, d, args.reads, rng))
        best[d] = st.median(runs)
        print(f"  {d:>6} {best[d]:>22.2f}")

    peak_d = max(best, key=best.get)
    bw = best[peak_d]
    print(f"\n  peak {bw:.2f} GB/s at queue depth {peak_d}\n")
    print("  What that means for Beam-1.0-501B decode (expert streaming only,\n"
          "  i.e. an upper bound on tok/s — compute and the trunk are extra):\n")
    print(f"  {'hit rate':>9} {'SSD GB/token':>13} {'ms/token':>10} {'tok/s':>8}")
    for hit in (0.284, 0.381, 0.539, 0.661, 0.848, 0.964):
        gb = BYTES_PER_TOKEN * (1 - hit) / 1e9
        ms = gb / bw * 1000
        print(f"  {hit*100:>8.1f}% {gb:>13.2f} {ms:>10.0f} {1000/ms if ms else 0:>8.2f}")
    print("\n  (hit rates are the measured expert-locality curve from the pack's\n"
          "   own expert-usage-v1.json, for 48/64/96/128/192/256 GB of RAM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
