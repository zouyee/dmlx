#!/usr/bin/env python3
"""Compare Qwen3.8 oracle segments: python .npy dumps vs Zig harness .bin dumps.

The .bin format written by src/test_qwen38_oracle_dump.zig:
    [ndim: i32][dims: i32 x ndim][f32 row-major payload]

Usage:
    python3 scripts/compare_qwen38_oracle.py ORACLE_DIR ZIG_DIR [--threshold T]

Compares every segment present in both directories (embed, layer*_out,
hidden_final, logits). Prints per-segment cos/max_abs_err and exits non-zero
when any comparison fails: exit 2 on shape mismatch, exit 1 on cos < threshold.
Default threshold 0.9999 (f32 residual trajectory vs bf16 oracle tolerance).
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def read_bin(path):
    with open(path, "rb") as f:
        header = f.read(4)
        if len(header) < 4:
            sys.exit(f"{path}: truncated header")
        (ndim,) = struct.unpack("<i", header)
        if not 1 <= ndim <= 8:
            sys.exit(f"{path}: implausible ndim {ndim}")
        dims_raw = f.read(4 * ndim)
        if len(dims_raw) < 4 * ndim:
            sys.exit(f"{path}: truncated shape")
        shape = struct.unpack(f"<{ndim}i", dims_raw)
        count = int(np.prod(shape))
        payload = f.read()
        if len(payload) != 4 * count:
            sys.exit(
                f"{path}: payload {len(payload)} bytes != {4 * count} for shape {shape}"
            )
        return np.frombuffer(payload, dtype="<f4").reshape(shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle_dir", type=Path)
    parser.add_argument("zig_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.9999)
    args = parser.parse_args()

    failures = []
    compared = 0
    for npy in sorted(args.oracle_dir.glob("*.npy")):
        name = npy.stem
        bin_path = args.zig_dir / f"{name}.bin"
        if not bin_path.exists():
            print(f"SKIP {name}: no {bin_path}")
            continue
        a = np.load(npy).astype(np.float64)
        b = read_bin(bin_path).astype(np.float64)
        compared += 1
        if a.shape != b.shape:
            print(f"FAIL {name}: shape mismatch {a.shape} vs {b.shape}")
            failures.append(name)
            continue
        av, bv = a.ravel(), b.ravel()
        cos = float(av @ bv / (np.linalg.norm(av) * np.linalg.norm(bv)))
        max_abs = float(np.max(np.abs(av - bv)))
        status = "PASS" if cos >= args.threshold else "FAIL"
        print(f"{status} {name}: cos={cos:.8f} max_abs_err={max_abs:.6e} shape={a.shape}")
        if cos < args.threshold:
            failures.append(name)

    if compared == 0:
        sys.exit("no segments compared")
    if failures:
        sys.exit(f"FAILED segments: {', '.join(failures)} (threshold {args.threshold})")
    print(f"all {compared} segments pass (cos >= {args.threshold})")


if __name__ == "__main__":
    main()
