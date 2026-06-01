#!/usr/bin/env python3
"""Compare two DSV4 activation dump directories layer-by-layer.

Each directory is produced by running the server with DSV4_DUMP_DIR set
(see src/models/activation_dump.zig). Files are float32 .npy:
  layer_00.npy ... layer_42.npy, final_norm.npy, logits.npy

Typical use (Phase 1 oracle):
  DSV4_DUMP_DIR=/tmp/mlx_ref   bash scripts/dsv4_smoke.sh
  DSV4_DUMP_DIR=/tmp/metal_out METAL_MOE=1 bash scripts/dsv4_smoke.sh
  python3 scripts/compare_metal_mlx.py /tmp/mlx_ref /tmp/metal_out

Reports per-file max abs diff, mean abs diff, relative L2, and flags the
first file exceeding the threshold (where divergence begins).

Exit code: 0 if all within threshold, 1 otherwise.
"""
import sys
import os
import argparse

try:
    import numpy as np
except ImportError:
    sys.exit("numpy required: pip install numpy")


def load(d, name):
    path = os.path.join(d, name)
    if not os.path.exists(path):
        return None
    return np.load(path)


def ordered_names(ref_dir):
    """Return dump file names in forward order: layers, final_norm, logits."""
    files = set(f for f in os.listdir(ref_dir) if f.endswith(".npy"))
    layers = sorted(f for f in files if f.startswith("layer_"))
    tail = [f for f in ("final_norm.npy", "logits.npy") if f in files]
    return layers + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir", help="reference dump dir (MLX oracle)")
    ap.add_argument("cmp_dir", help="dump dir to compare (e.g. metal)")
    ap.add_argument("--threshold", type=float, default=1e-2,
                    help="relative L2 threshold to flag divergence (default 1e-2)")
    args = ap.parse_args()

    names = ordered_names(args.ref_dir)
    if not names:
        sys.exit(f"no .npy files in {args.ref_dir}")

    print(f"{'file':<16} {'shape':<18} {'max_abs':>11} {'mean_abs':>11} {'rel_L2':>10}  status")
    print("-" * 80)

    first_bad = None
    any_bad = False
    for name in names:
        a = load(args.ref_dir, name)
        b = load(args.cmp_dir, name)
        if a is None or b is None:
            print(f"{name:<16} {'MISSING':<18} "
                  f"({'ref' if a is None else 'cmp'} dir lacks this file)")
            any_bad = True
            continue
        if a.shape != b.shape:
            print(f"{name:<16} SHAPE MISMATCH ref={a.shape} cmp={b.shape}")
            any_bad = True
            if first_bad is None:
                first_bad = name
            continue

        af = a.astype(np.float64).ravel()
        bf = b.astype(np.float64).ravel()
        diff = np.abs(af - bf)
        max_abs = float(diff.max()) if diff.size else 0.0
        mean_abs = float(diff.mean()) if diff.size else 0.0
        denom = np.linalg.norm(af)
        rel_l2 = float(np.linalg.norm(af - bf) / denom) if denom > 0 else 0.0

        bad = rel_l2 > args.threshold
        status = "OK" if not bad else "DIVERGE"
        if bad:
            any_bad = True
            if first_bad is None:
                first_bad = name
        print(f"{name:<16} {str(a.shape):<18} {max_abs:>11.3e} "
              f"{mean_abs:>11.3e} {rel_l2:>10.3e}  {status}")

    print("-" * 80)
    if first_bad is not None:
        print(f"⚠ first divergence at: {first_bad} (rel_L2 > {args.threshold:g})")
    if any_bad:
        print("RESULT: DIVERGENCE DETECTED")
        return 1
    print(f"RESULT: all within rel_L2 threshold {args.threshold:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
