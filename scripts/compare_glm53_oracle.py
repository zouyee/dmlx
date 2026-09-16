#!/usr/bin/env python3
"""Compare a numpy oracle fixture (.npy) against a Zig harness dump (.bin).

The .bin format written by src/test_glm53_oracle_dump.zig:
    [ndim: i32][dims: i32 x ndim][f32 row-major payload]

Usage:
    python3 scripts/compare_glm53_oracle.py A.npy B.bin [--threshold T]

Prints shape/cos/max_abs_err and exits non-zero when cos < threshold.
Default threshold 0.999 (bf16 compute-vs-f32 tolerance; pass --threshold
0.9999 for FP32_KEEP segments, 0.998 only when bf16 quantization noise is
explicitly expected).
"""

import argparse
import struct
import sys

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
    parser.add_argument("npy")
    parser.add_argument("bin")
    parser.add_argument("--threshold", type=float, default=0.999)
    args = parser.parse_args()

    a = np.load(args.npy).astype(np.float64)
    b = read_bin(args.bin).astype(np.float64)

    print(f"A {args.npy}: shape={a.shape}")
    print(f"B {args.bin}: shape={b.shape}")
    if a.shape != b.shape:
        print("FAIL: shape mismatch")
        sys.exit(1)

    av, bv = a.ravel(), b.ravel()
    cos = float(av @ bv / (np.linalg.norm(av) * np.linalg.norm(bv)))
    max_abs = float(np.max(np.abs(av - bv)))
    print(f"cos={cos:.8f} max_abs_err={max_abs:.6e} threshold={args.threshold}")
    if cos >= args.threshold:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
