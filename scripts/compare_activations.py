#!/usr/bin/env python3
"""Compare native dmlx per-layer activations against MLX reference dumps.

Usage:
    python3 scripts/compare_activations.py /tmp/dsv4_native_dump /tmp/dsv4_mlx_dump
"""
import os
import re
import sys
import numpy as np


def load_bin(path):
    return np.fromfile(path, dtype=np.float32)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 compare_activations.py <native_dir> <mlx_dir>")
        sys.exit(1)

    native_dir, mlx_dir = sys.argv[1], sys.argv[2]
    pattern = re.compile(r"L(\d+)_(in|out)_pos0\.bin$")

    # Collect matching pairs.
    pairs = []
    for name in sorted(os.listdir(native_dir)):
        m = pattern.match(name)
        if not m:
            continue
        layer, kind = int(m.group(1)), m.group(2)
        native_path = os.path.join(native_dir, name)
        mlx_path = os.path.join(mlx_dir, name)
        if not os.path.exists(mlx_path):
            continue
        pairs.append((layer, kind, native_path, mlx_path))

    if not pairs:
        print("No matching L<layer>_{in,out}_pos0.bin pairs found.")
        sys.exit(1)

    # Sort by layer then kind (in before out).
    pairs.sort(key=lambda x: (x[0], 0 if x[1] == "in" else 1))

    threshold = 1e-3
    diverged = []
    rows = []
    for layer, kind, npa, npx in pairs:
        a = load_bin(npa)
        x = load_bin(npx)
        if a.shape != x.shape:
            rows.append((layer, kind, str(a.shape), str(x.shape), "shape-mismatch", "shape-mismatch"))
            diverged.append((layer, kind, float("inf")))
            continue
        diff = np.abs(a - x)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        rows.append((layer, kind, str(a.shape), f"{max_diff:.6e}", f"{mean_diff:.6e}"))
        if max_diff > threshold:
            diverged.append((layer, kind, max_diff))

    # Print table.
    print(f"{'Layer':>5} {'Kind':>4} {'Shape':>16} {'Max abs diff':>14} {'Mean abs diff':>14}")
    print("-" * 60)
    for row in rows:
        print(f"{row[0]:>5} {row[1]:>4} {row[2]:>16} {row[3]:>14} {row[4]:>14}")

    print()
    if diverged:
        first_layer, first_kind, first_max = diverged[0]
        print(f"First diverging activation: L{first_layer:02d}_{first_kind}_pos0.bin  max_diff={first_max:.6e}")
        print(f"Threshold: {threshold}")
        print("First 5 diverging layers/files:")
        for layer, kind, max_diff in diverged[:5]:
            print(f"  L{layer:02d}_{kind}_pos0.bin  max_diff={max_diff:.6e}")
    else:
        print(f"No divergence above threshold {threshold}.")


if __name__ == "__main__":
    main()
