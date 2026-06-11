#!/usr/bin/env python3
"""Repack MoE expert weights from original int8 model into affine 4-bit format.

Reads original int8 + E8M0 weights directly from safetensors (no mlx needed),
dequantizes to f32, then re-quantizes to affine 4-bit (group_size=64).

Output: ~/models/DeepSeek-V4-Flash/packed_experts_affine/
Per-expert: ~14.2 MB, SMELT N=51 fits in 48GB.
"""

import sys, os, json, struct, time, math
from pathlib import Path
import numpy as np

GROUP_SIZE = 64  # affine 4-bit group size

COMPONENTS = [
    ("gate_proj", "w1"),   # [2048, 2048] int8 + [2048, 128] F8_E8M0
    ("up_proj", "w3"),     # [2048, 2048] int8 + [2048, 128] F8_E8M0
    ("down_proj", "w2"),   # [4096, 1024] int8 + [4096, 64] F8_E8M0
]

GATE_UP_WEIGHT_BYTES = 2048 * 512 * 4   # uint32 packed 4-bit
GATE_UP_SCALE_BYTES  = 2048 * 64 * 2    # bf16
DOWN_WEIGHT_BYTES    = 4096 * 256 * 4    # uint32 packed 4-bit
DOWN_SCALE_BYTES     = 4096 * 32 * 2     # bf16


def load_safetensors_raw(path, key):
    """Load tensor raw bytes from safetensors file."""
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
    if key not in header:
        raise KeyError(f"Tensor {key} not found in {path}")
    info = header[key]
    start, end = info['data_offsets']
    with open(path, 'rb') as f:
        f.seek(8 + header_len + start)
        return f.read(end - start), info['dtype'], info['shape']


def dequantize_int8_to_f32(raw_bytes, dtype, shape):
    """Dequantize: int8 weight * exp2(F8_E8M0_scale - 127) → f32."""
    assert dtype == 'I8', f"Expected I8, got {dtype}"
    rows, cols = shape  # e.g., (2048, 2048) or (4096, 1024)
    weight = np.frombuffer(raw_bytes, dtype=np.int8).reshape(shape).astype(np.float32)
    return weight


def dequantize_scale_f8_e8m0(raw_bytes, dtype, shape):
    """Dequantize F8_E8M0 scale to f32: 2^(uint8 - 127)."""
    assert dtype == 'F8_E8M0', f"Expected F8_E8M0, got {dtype}"
    rows, cols = shape  # e.g., (2048, 128) or (4096, 64)
    scale = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(shape).astype(np.float32)
    # Broadcast: each scale applies to 16 elements → repeat 16 times along column axis
    scale_bc = np.repeat(scale, 16, axis=1)
    return np.exp2(scale_bc - 127.0)


def quantize_affine_4bit(f32_weights):
    """Quantize f32 array to affine 4-bit: nibble * scale + bias per group of 64.
    Returns (packed_uint8, scales_bf16, biases_bf16). Vectorized for speed.
    """
    out_dim, in_dim = f32_weights.shape
    num_groups = in_dim // GROUP_SIZE

    # Reshape to [out_dim, num_groups, 64]
    x = f32_weights.reshape(out_dim, num_groups, GROUP_SIZE)

    # Compute per-group min and max (vectorized)
    xmin = x.min(axis=2)  # [out_dim, num_groups]
    xmax = x.max(axis=2)  # [out_dim, num_groups]

    # Compute scales and biases
    scales = (xmax - xmin) / 15.0
    scales = np.maximum(scales, 1e-12)  # avoid division by zero
    biases = xmin

    # Quantize: q = round((x - bias) / scale), clamp to [0, 15]
    q = np.clip(np.round((x - biases[:, :, np.newaxis]) / scales[:, :, np.newaxis]), 0, 15).astype(np.uint8)

    # Pack 2 nibbles per byte: low nibble at bit 0, high nibble at bit 4
    packed = np.zeros((out_dim, in_dim // 2), dtype=np.uint8)
    q_flat = q.reshape(out_dim, -1)  # flatten last two dims
    packed = (q_flat[:, 0::2] | (q_flat[:, 1::2] << 4))

    return packed, scales.astype(np.float32), biases.astype(np.float32)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else input_dir / "packed_experts_affine"

    # Load index
    with open(input_dir / "model.safetensors.index.json") as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Detect MoE layers
    moe_layers = set()
    for key in weight_map:
        if "ffn.experts." in key:
            parts = key.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        moe_layers.add(int(parts[i + 1]))
                    except ValueError:
                        pass
    moe_layers = sorted(moe_layers)
    print(f"MoE layers: {len(moe_layers)} ({moe_layers[0]}..{moe_layers[-1]})")

    # Count experts from first MoE layer
    n_experts = len([k for k in weight_map if f"layers.{moe_layers[0]}.ffn.experts." in k and ".weight" in k and ".w1." in k])
    print(f"Experts per layer: {n_experts}")

    # Cache shard data
    shard_files = sorted(set(weight_map.values()))
    print(f"Shard files: {len(shard_files)}")

    # Manifest
    manifest = {
        "version": 2,
        "format": "affine_4bit",
        "group_size": GROUP_SIZE,
        "model_dir": str(input_dir),
        "n_experts": n_experts,
        "layers": {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for layer_idx in moe_layers:
        t_layer = time.time()
        print(f"  [{layer_idx}] ", end="", flush=True)

        out_path = output_dir / f"layer_{layer_idx:02d}.bin"
        expert_size = GATE_UP_WEIGHT_BYTES * 2 + DOWN_WEIGHT_BYTES + \
                       GATE_UP_SCALE_BYTES * 4 + DOWN_SCALE_BYTES * 2

        with open(out_path, "wb") as outf:
            for eid in range(n_experts):
                for comp_name, comp_key in COMPONENTS:
                    w_key = f"layers.{layer_idx}.ffn.experts.{eid}.{comp_key}.weight"
                    s_key = f"layers.{layer_idx}.ffn.experts.{eid}.{comp_key}.scale"

                    shard = weight_map[w_key]
                    w_raw, w_dtype, w_shape = load_safetensors_raw(input_dir / shard, w_key)
                    s_raw, s_dtype, s_shape = load_safetensors_raw(input_dir / shard, s_key)

                    # Dequantize int8 + E8M0 → f32
                    w_f32 = dequantize_int8_to_f32(w_raw, w_dtype, w_shape)
                    s_f32 = dequantize_scale_f8_e8m0(s_raw, s_dtype, s_shape)
                    f32 = w_f32 * s_f32

                    # Quantize to affine 4-bit
                    packed, scales, biases = quantize_affine_4bit(f32)

                    # Write
                    outf.write(packed.tobytes())
                    outf.write(scales.astype(np.float16).tobytes())
                    outf.write(biases.astype(np.float16).tobytes())

        elapsed = time.time() - t_layer
        size_mb = expert_size * n_experts / (1024 * 1024)
        print(f"{size_mb:.1f}MB ({elapsed:.1f}s)")

        manifest["layers"][str(layer_idx)] = {
            "expert_size": expert_size,
            "n_experts": n_experts,
            "file": out_path.name,
        }

    # Manifest
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - t0
    total_size = sum(m["expert_size"] * m["n_experts"] for m in manifest["layers"].values())
    print(f"\nDone in {total_time:.0f}s, total size: {total_size / (1024**3):.1f} GB")


if __name__ == "__main__":
    main()