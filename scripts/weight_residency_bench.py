#!/usr/bin/env python3
"""Does weight RESIDENCY explain the plain-decode gap? No. Measured 2026-11-03.

The question
------------
Plain decode has to stream ~14.42 GB of weights per token. mlx-lm does it at
~222-234 GB/s and dmlx at ~206, a 11-14% gap on the SAME checkpoint through
the SAME MLX qmm kernels. Every op-level explanation has been ablated to zero:
dispatch count, dependency depth, qmm call count, the GDN tail, attention
non-qmm work, CPU graph build. The one structural difference left was how the
weights come to be resident:

  dmlx  : `TensorIndex.loadTensor` preads each tensor into a heap buffer and
          calls `mlx_array_new_data`, which COPIES into a fresh MLX
          allocation -> ~2180 independent Metal buffers.
  mlx-lm: `mx.load` on safetensors, which MLX implements as an mmap with
          zero-copy views -> a few contiguous file-backed mappings.

The answer
----------
Copied is not slower. It is marginally FASTER. Six real MLP projections from
the checkpoint, 20 reps each, min-of-reps, 50.1 MB per tensor:

  tensor                      mmap GB/s   copied GB/s
  layer0 down_proj                 80.4         180.1   <- cold mmap pages
  layer0 gate_proj                154.1         153.2
  layer0 up_proj                  163.7         156.8
  layer1 down_proj                151.7         140.5
  layer1 gate_proj                159.4         156.0
  layer1 up_proj                  155.2         155.3
  TOTAL                           135.3         156.1   (+15.4% for copied)

Discounting the first tensor touched, which pays mmap page faults, the two are
a tie at 140-164 GB/s. So the per-tensor copy is NOT the gap, and the
"load into one arena or mmap zero-copy" plan is dead. Resident anonymous
memory is, if anything, slightly better than file-backed pages here.

This also retires src/bench_weight_layout.zig, which reported 0.1 GB/s for
every layout — a broken bench, not a finding. Its shape is fine in isolation
(N=557056 runs 6.74 ms at 237.9 GB/s through plain MLX), so the ~1860x
discrepancy was in the Zig harness, and the conclusion it invited would have
been drawn from garbage either way.

A separate observation worth keeping: a single 50 MB tensor only reaches
~155 GB/s, while one 1.6 GB tensor reaches 238 GB/s on the same kernel. Small
per-call sizes do not saturate the memory system. That is a per-launch effect
and it applies to mlx-lm equally, since it reads the same 50 MB tensors, so it
does not explain the gap either — but it does mean any absolute GB/s figure
has to be quoted with the tensor size it was measured at.

Usage:
    python3 scripts/weight_residency_bench.py [--layers 2] [--reps 20]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

MODEL = os.path.expanduser("~/models/Qwen3.8-27B-4bit")


def copy_to_mlx(mx, np, a):
    """A new MLX-owned buffer holding the same bytes, as our loader produces.

    bfloat16 has no numpy equivalent, so it round-trips through uint16 and is
    reinterpreted — same bytes, same shape, a fresh allocation.
    """
    if a.dtype == mx.bfloat16:
        raw = np.frombuffer(memoryview(a.view(mx.uint16)), dtype=np.uint16).copy()
        return mx.array(raw).view(mx.bfloat16).reshape(a.shape)
    return mx.array(np.array(a))


def bench(mx, wq, s, b, k, reps):
    x = mx.random.normal((1, k)).astype(mx.bfloat16)
    mx.eval(x)

    def f():
        return mx.quantized_matmul(x, wq, s, b, transpose=True,
                                   group_size=64, bits=4)

    for _ in range(3):          # warm: first call compiles and faults pages in
        mx.eval(f())
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(f())
        best = min(best, time.perf_counter() - t)
    return best, wq.nbytes + s.nbytes + b.nbytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    import mlx.core as mx
    import numpy as np

    index = os.path.join(MODEL, "model.safetensors.index.json")
    if not os.path.exists(index):
        print(f"checkpoint index not found: {index}", file=sys.stderr)
        return 2
    wmap = json.load(open(index))["weight_map"]
    keys = sorted(k for k in wmap
                  if k.endswith(("gate_proj.weight", "up_proj.weight",
                                 "down_proj.weight")))[: args.layers * 3]

    totals = {"mmap": [0.0, 0], "copied": [0.0, 0]}
    print(f"  {'tensor':<30} {'MB':>7} {'mmap':>9} {'copied':>9}  GB/s")
    for key in keys:
        shard = mx.load(os.path.join(MODEL, wmap[key]))
        wq = shard[key]
        s = shard[key.replace(".weight", ".scales")]
        b = shard[key.replace(".weight", ".biases")]
        k = s.shape[1] * 64

        t_mmap, nbytes = bench(mx, wq, s, b, k, args.reps)
        wq2, s2, b2 = (copy_to_mlx(mx, np, wq), copy_to_mlx(mx, np, s),
                       copy_to_mlx(mx, np, b))
        mx.eval(wq2, s2, b2)
        t_copy, _ = bench(mx, wq2, s2, b2, k, args.reps)

        totals["mmap"][0] += t_mmap
        totals["mmap"][1] += nbytes
        totals["copied"][0] += t_copy
        totals["copied"][1] += nbytes
        label = key.split("layers.")[1][:28]
        print(f"  {label:<30} {nbytes/1e6:7.1f} {nbytes/t_mmap/1e9:9.1f} "
              f"{nbytes/t_copy/1e9:9.1f}")
        del shard, wq, s, b, wq2, s2, b2
        mx.clear_cache()

    bw = {k: by / t for k, (t, by) in totals.items()}
    for k in ("mmap", "copied"):
        print(f"  {k:>8} total: {bw[k]/1e9:.1f} GB/s")
    print(f"\n  copied vs mmap: {100*(bw['copied']/bw['mmap']-1):+.1f}%  "
          "(positive = the per-tensor copy is not the problem)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
