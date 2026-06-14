#!/usr/bin/env python3
"""Convert MXFP4 MoE expert weights to LUT-based 4-bit format with larger group size.

The native engine uses LUT-based kernels that apply the EXACT same formula as MXFP4:
  value = LUT[nibble] * exp2(scale - 127)

The only difference from MXFP4 is group_size=64 instead of 32. This is achieved by:
  1. Keeping original MXFP4 nibbles (no re-quantization)
  2. Merging 2 adjacent MXFP4 scale groups into 1 (taking the max)

Output format per expert (12.4 MB):
  gate_W: [2048, 512] uint32  = 4,194,304 bytes  (same as MXFP4)
  gate_s: [2048, 64] uint8    =   131,072 bytes  (merged from 128 groups of 32)
  up_W:   [2048, 512] uint32  = 4,194,304 bytes
  up_s:   [2048, 64] uint8    =   131,072 bytes
  down_W: [4096, 256] uint32  = 4,194,304 bytes
  down_s: [4096, 32] uint8    =   131,072 bytes
  Total: 12,976,128 bytes
"""

import sys, json, struct, time
from pathlib import Path
import numpy as np

GROUP_SIZE = 64

COMPONENTS = [
    ("gate_proj",   2048, 4096, 512, 128),  # 128 MXFP4 scale groups
    ("up_proj",     2048, 4096, 512, 128),
    ("down_proj",   4096, 2048, 256, 64),   # 64 MXFP4 scale groups
]

def load_safetensors_raw(path, key):
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
    info = header[key]
    start = info['data_offsets'][0]
    end = info['data_offsets'][1]
    with open(path, 'rb') as f:
        f.seek(8 + header_len + start)
        return f.read(end - start), info['dtype'], info['shape']

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

    manifest = {
        "version": 4,
        "format": "affine_4bit_lut",
        "group_size": GROUP_SIZE,
        "model_dir": str(input_dir),
        "n_experts": 256,
        "layers": {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for layer_idx in moe_layers:
        t_layer = time.time()
        print(f"  [{layer_idx}] Loading...", end="", flush=True)

        component_data = {}
        for comp_name, out_dim, in_dim, packed_cols, num_groups_mxfp4 in COMPONENTS:
            w_key = f"model.layers.{layer_idx}.ffn.switch_mlp.{comp_name}.weight"
            s_key = f"model.layers.{layer_idx}.ffn.switch_mlp.{comp_name}.scales"

            shard = weight_map[w_key]
            w_raw, w_dtype, w_shape = load_safetensors_raw(input_dir / shard, w_key)
            s_raw, s_dtype, s_shape = load_safetensors_raw(input_dir / shard, s_key)

            # Original MXFP4 nibbles (keep as-is)
            weight = np.frombuffer(w_raw, dtype=np.uint32).reshape(256, out_dim, packed_cols)
            scale_uint8 = np.frombuffer(s_raw, dtype=np.uint8).reshape(256, out_dim, num_groups_mxfp4)

            # Merge 2 adjacent groups: max(scale_g, scale_{g+1})
            scales_merged = np.maximum(
                scale_uint8[:, :, 0::2],
                scale_uint8[:, :, 1::2]
            )  # [256, out_dim, num_groups_mxfp4//2]

            component_data[comp_name] = (
                [weight[eid].tobytes() for eid in range(256)],
                [scales_merged[eid].tobytes() for eid in range(256)]
            )
            print(f" {comp_name}", end="", flush=True)

        out_path = output_dir / f"layer_{layer_idx:02d}.bin"
        with open(out_path, "wb") as outf:
            for eid in range(256):
                for comp_name, _, _, _, _ in COMPONENTS:
                    packed_list, scales_list = component_data[comp_name]
                    outf.write(packed_list[eid])
                    outf.write(scales_list[eid])

        actual = out_path.stat().st_size
        expected = 256 * 12976128
        status = "OK" if actual == expected else f"SIZE MISMATCH ({actual} != {expected})"
        elapsed = time.time() - t_layer
        print(f" → {actual / (1024*1024):.0f}MB ({elapsed:.1f}s) {status}")

        if actual != expected:
            sys.exit(1)

        manifest["layers"][str(layer_idx)] = {
            "expert_size": 12976128,
            "n_experts": 256,
            "file": out_path.name,
        }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - t0
    total_gb = sum(m["expert_size"] * m["n_experts"] for m in manifest["layers"].values()) / (1024**3)
    print(f"\nDone in {total_time:.0f}s, total size: {total_gb:.1f} GB")

if __name__ == "__main__":
    main()