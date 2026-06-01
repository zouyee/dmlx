#!/usr/bin/env python3
"""S1 go/no-go: verify the Metal affine-4bit dequant formula against MLX.

Replicates the exact arithmetic of the `dequant_matvec_affine` Metal kernel
(src/models/moe_kernel.metal) in numpy, and compares against mlx.core.dequantize
on a real DeepSeek-V4 attention weight (layer 0 wq_a, affine 4bit, group_size=64).

If max-abs diff is ~0, the kernel formula + nibble packing order are correct and
the full-metal attention port (Phase 3-5 / S1) is GO.

Usage: python3 scripts/verify_affine_dequant.py
"""
import sys, json, struct, glob
import numpy as np

MODEL = "/Users/zouyee/models/DeepSeek-V4-Flash-4bit"
GROUP_SIZE = 64
BITS = 4


def find_tensor(name):
    """Return (dtype, shape, data_offset, file) for a tensor across shards."""
    for f in sorted(glob.glob(MODEL + "/model-*-of-00033.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            base = 8 + n
            if name in hdr:
                v = hdr[name]
                return v["dtype"], v["shape"], v["data_offsets"], base, f
    return None


def load_raw(name):
    info = find_tensor(name)
    if info is None:
        sys.exit(f"tensor not found: {name}")
    dtype, shape, (o0, o1), base, f = info
    with open(f, "rb") as fh:
        fh.seek(base + o0)
        raw = fh.read(o1 - o0)
    return dtype, shape, raw


def to_np(dtype, shape, raw):
    if dtype == "U32":
        arr = np.frombuffer(raw, dtype="<u4")
    elif dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        arr = (u16 << 16).view(np.float32)
    elif dtype == "F32":
        arr = np.frombuffer(raw, dtype="<f4")
    else:
        sys.exit(f"unhandled dtype {dtype}")
    return arr.reshape(shape)


def kernel_dequant(packed, scales, biases):
    """Exact replica of dequant_matvec_affine's weight reconstruction.
    packed: [out, in/8] uint32 ; scales/biases: [out, in/gs] f32.
    Returns dense w: [out, in] f32 with w = scale_g*nibble + bias_g.
    """
    out_dim, packed_cols = packed.shape
    in_dim = packed_cols * 8
    num_groups = in_dim // GROUP_SIZE
    w = np.empty((out_dim, in_dim), dtype=np.float32)
    for r in range(out_dim):
        for g in range(num_groups):
            scale = scales[r, g]
            bias = biases[r, g]
            for p in range(GROUP_SIZE // 8):
                pw = packed[r, g * (GROUP_SIZE // 8) + p]
                xbase = g * GROUP_SIZE + p * 8
                for i in range(8):
                    nib = (pw >> (i * 4)) & 0xF
                    w[r, xbase + i] = scale * nib + bias
    return w


def main():
    name = "model.layers.0.attn.wq_a"
    dt_w, sh_w, raw_w = load_raw(name + ".weight")
    dt_s, sh_s, raw_s = load_raw(name + ".scales")
    dt_b, sh_b, raw_b = load_raw(name + ".biases")
    print(f"weight {dt_w}{sh_w}  scales {dt_s}{sh_s}  biases {dt_b}{sh_b}")

    packed = to_np(dt_w, sh_w, raw_w)
    scales = to_np(dt_s, sh_s, raw_s)
    biases = to_np(dt_b, sh_b, raw_b)

    # Validate against MLX dequantize on a small slice (full is 1024x4096; slice rows).
    ROWS = 32
    try:
        import mlx.core as mx
        w_q = mx.array(packed[:ROWS])
        w_s = mx.array(scales[:ROWS])
        w_b = mx.array(biases[:ROWS])
        mlx_w = np.array(mx.dequantize(w_q, w_s, w_b, group_size=GROUP_SIZE, bits=BITS).astype(mx.float32))
    except Exception as e:
        sys.exit(f"MLX dequantize failed: {e}")

    my_w = kernel_dequant(packed[:ROWS], scales[:ROWS], biases[:ROWS])

    diff = np.abs(mlx_w - my_w)
    print(f"slice rows={ROWS}  mlx_w{mlx_w.shape}  my_w{my_w.shape}")
    print(f"max_abs_diff={diff.max():.3e}  mean_abs_diff={diff.mean():.3e}")
    print(f"sample mlx={mlx_w[0,:4]}  mine={my_w[0,:4]}")
    if diff.max() < 1e-3:
        print("RESULT: GO — affine dequant formula matches MLX")
        return 0
    print("RESULT: NO-GO — formula/packing mismatch")
    return 1


if __name__ == "__main__":
    sys.exit(main())
