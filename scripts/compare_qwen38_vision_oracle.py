#!/usr/bin/env python3
"""Compare two Qwen3.8 VISION oracle dump directories (.npy segments).

Companion to scripts/qwen38_vision_oracle_dump.py and
src/test_qwen38_vision_oracle.zig (which enforces the actual gates against
the Zig implementation). This script A/Bs two dump dirs — e.g. fused vs
QWEN38_NAIVE_GDN=1, or a re-dump after an mlx-vlm upgrade — printing
per-segment cos/max_abs_err and exiting non-zero when any shared segment
falls below the threshold.

Usage:
    /Users/zouyee/venvs/dmlx-omlx/bin/python scripts/compare_qwen38_vision_oracle.py \
        artifacts/qwen38_vision_oracle artifacts/qwen38_vision_oracle_naive \
        [--threshold 0.999]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def compare(a: np.ndarray, b: np.ndarray):
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    cos = float(av @ bv / (np.linalg.norm(av) * np.linalg.norm(bv)))
    return cos, float(np.max(np.abs(av - bv)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir_a", type=Path)
    parser.add_argument("dir_b", type=Path)
    parser.add_argument("--threshold", type=float, default=0.999)
    args = parser.parse_args()

    failures = []
    compared = 0
    for npy in sorted(args.dir_a.glob("*.npy")):
        other = args.dir_b / npy.name
        if not other.exists():
            print(f"SKIP {npy.stem}: no {other}")
            continue
        a = np.load(npy)
        b = np.load(other)
        compared += 1
        if a.shape != b.shape:
            print(f"FAIL {npy.stem}: shape {a.shape} vs {b.shape}")
            failures.append(npy.stem)
            continue
        if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
            exact = bool(np.array_equal(a, b))
            status = "PASS" if exact else "FAIL"
            print(f"{status} {npy.stem}: exact={exact} shape={a.shape}")
            if not exact:
                failures.append(npy.stem)
            continue
        cos, max_abs = compare(a, b)
        status = "PASS" if cos >= args.threshold else "FAIL"
        print(f"{status} {npy.stem}: cos={cos:.8f} max_abs_err={max_abs:.6e} shape={a.shape}")
        if cos < args.threshold:
            failures.append(npy.stem)

    for name in ("meta.json",):
        ma = args.dir_a / name
        mb = args.dir_b / name
        if ma.exists() and mb.exists():
            ja = json.loads(ma.read_text())
            jb = json.loads(mb.read_text())
            same = ja.get("input_ids") == jb.get("input_ids")
            print(f"{'PASS' if same else 'FAIL'} meta.input_ids identical={same}")
            if not same:
                failures.append("meta.input_ids")

    if compared == 0:
        sys.exit("no segments compared")
    if failures:
        sys.exit(f"FAILED segments: {', '.join(failures)} (threshold {args.threshold})")
    print(f"all {compared} segments pass (cos >= {args.threshold})")


if __name__ == "__main__":
    main()
