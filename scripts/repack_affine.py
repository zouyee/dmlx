#!/usr/bin/env python3
"""Repack MoE expert weights from MXFP4 model into affine 4-bit format.

Reads MXFP4 uint32 weights and uint8 scales directly from safetensors,
dequantizes to f32, then re-quantizes to affine 4-bit (group_size=64).

Output: ~/models/DeepSeek-V4-Flash/packed_experts_affine/
Per-expert: ~14.2 MB, SMELT N=45 fits in 48GB.
"""

import sys, os, json, struct, time, math
from pathlib import Path
import numpy as np

GROUP_SIZE = 64

# MXFP4 model expert tensor layout
# weight: [n_experts, 2048, 512] uint32, scale: [n_experts, 2048, 128] uint8
# For each expert: down_proj weight: [n_experts, 4096, 256] uint32, scale: [n_experts, 4096, 64] uint8

WEIGHT_BYTES = 2048 * 512 * 4   # uint32 packed 4-bit
SCALE_BYTES  = 2048 * 128 * 2   # bf16

def load_safetensors_raw(path, key):
    """Load tensor raw bytes from safetensors file."""
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
    if key not in header:
        raise KeyError(f"Tensor {key} not found in {path}")
    info = header[key]
    start = info['data_offsets'][0]
    end = info['data_offsets'][1]
    with open(path, 'rb') as f:
        f.seek(8 + header_len + start)
        return f.read(end - start), info['dtype'], info['shape']

def dequantize_uint32_to_f32(raw_bytes, dtype, expert_shape):
    """Dequantize MXFP4 uint32 weight to f32."""
    assert dtype in ('U32', 'I32')
    n_experts, rows, cols = expert_shape  # (256, 2048, 512)
    # Dequantize the full batch: (256, 2048, 512) -> (256, 2048, 4096)
    weight = np.frombuffer(raw_bytes, dtype=np.uint32).reshape(n_experts, rows, cols)
    lut = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
    nib_low = np.zeros((n_experts, rows, cols * 8), dtype=np.float32)
    nib_high = np.zeros((n_experts, rows, cols * 8), dtype=np.float32)
    nib_low[:, :, 0::2] = lut[weight & 0xF]
    nib_high[:, :, 1::2] = lut[weight >> 4]
    return nib_low + nib_high

def dequantize_scale_f8_e8m0(raw_bytes, dtype, expert_shape):
    """Dequantize F8_E8M0 scale to f32: 2^(uint8 - 127)."""
    assert dtype == 'F8_E8M0', f"Expected F8_E8M0, got {dtype}"
    return np.exp2(np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 127.0)

def quantize_affine_4bit(f32_weights):
    """Quantize f32 array to affine 4-bit: nibble * scale + bias per group of 64.
    Uses symmetric quantization. Returns packed_uint8, scales_bf16, biases_bf16."""
    out_dim, in_dim = f32_weights.shape  # (2048, 4096) or (4096, 2048)
    num_groups = in_dim // GROUP_SIZE

    # Reshape to [out_dim, num_groups, 64]
    x = f32_weights.reshape(out_dim, num_groups, GROUP_SIZE)
    xmax = np.maximum(np.abs(x.min(axis=2)), np.abs(x.max(axis=2)))
    scales = xmax / 7.0
    scales = np.maximum(scales, 1e-12)
    q = np.clip(np.round(x / scales[:, :, np.newaxis]), -7, 7).astype(np.int8)
    q = q + 8

    packed = np.zeros((out_dim, in_dim // 2), dtype=np.uint8)
    for r in range(out_dim):
        for c in range(0, num_groups, 2):
            packed[r, c // 2] = q[r, c] | (q[r, c + 1] << 4)
    return packed, scales.astype(np.float32), np.zeros_like(scales)

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else input_dir / "packed_experts_affine"

    index_file = input_dir / "model.safetensors.index.json"
    with open(index_file) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    moe_layers = set()
    for key in weight_map:
        if "switch_mlp" in key:
            parts = key.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        moe_layers.add(int(parts[i + 1]))
                    except ValueError:
                        pass
    moe_layers = sorted(moe_layers)
    print(f"MoE layers: {len(moe_layers)} ({moe_layers[0]}..{moe_layers[-1]})")

    shard_files = sorted(set(weight_map.values()))
    print(f"Shard files: {len(shard_files)}")

    manifest = {
        "version": 3,
        "format": "affine_4bit",
        "group_size": GROUP_SIZE,
        "model_dir": str(input_dir),
        "n_experts": 256,
        "layers": {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for layer_idx in moe_layers:
        t_layer = time.time()
        print(f"  [{layer_idx}] ", end="", flush=True)

        out_path = output_dir / f"layer_{layer_idx:02d}.bin"
        expert_size = 3 * 2048 * 512 * 4 + 3 * 2048 * 128 * 2 + 4096 * 256 * 4 + 4096 * 64 * 2

        with open(out_path, "wb") as outf:
            for eid in range(256):
                for comp_name, comp_key, weight_rows, scale_cols in [
                    ("gate_proj", "w1", 2048, 512),
                    ("up_proj", "w3", 2048, 512),
                    ("down_proj", "w2", 4096, 256),
                ]:
                    w_key = f"model.layers.{layer_idx}.ffn.switch_mlp.{comp_name}.weight"
                    s_key = f"model.layers.{layer_idx}.ffn.switch_mlp.{comp_name}.scales"

                    shard = weight_map[w_key]
                    w_raw, w_dtype, w_shape = load_safetensors_raw(input_dir / shard, w_key)
                    s_raw, s_dtype, s_shape = load_safetensors_raw(input_dir / shard, s_key)

                    # Dequantize uint32 to f32
                    w_f32 = dequantize_uint32_to_f32(w_raw, w_dtype, w_shape)
                    s_f32 = dequantize_scale_f8_e8m0(s_raw, s_dtype, s_shape)
                    f32 = w_f32 * s_f32

                    # Quantize to affine 4-bit
                    packed, scales, biases = quantize_affine_4bit(f32)

                    outf.write(packed.tobytes())
                    outf.write(scales.tobytes())
                    outf.write(biases.tobytes())

        elapsed = time.time() - t_layer
        size_mb = expert_size * 256 / (1024 * 1024)
        print(f"{size_mb:.1f}MB ({elapsed:.1f}s)")

        manifest["layers"][str(layer_idx)] = {
            "expert_size": expert_size,
            "n_experts": 256,
            "file": out_path.name,
        }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - t0
    total_size = sum(m["expert_size"] * m["n_experts"] for m in manifest["layers"].values())
    print(f"\nDone in {total_time:.0f}s, total size: {total_size / (1024**3):.1f} GB")


if __name__ == "__main__":
    main()