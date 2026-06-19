#!/usr/bin/env python3
"""Repack MXFP4 MoE experts to proper affine 4-bit (bf16 scale+bias, gs=64).

Correct flow: MXFP4 dequant → float32 → affine 4-bit requantize (per 64-element group).
No max-merge hack. Output compatible with fused_gate_up_swiglu_v2_affine kernel.

Usage:
  python3 scripts/repack_affine_v2.py ~/models/DeepSeek-V4-Flash-4bit [output_dir]
"""
import sys, json, struct, time, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

AFFINE_GS   = 64    # output group size
MXFP4_GS    = 32    # input group size
EXPERT_BATCH = 32   # experts per processing batch (memory control)
EXPERT_SIZE  = 256 * 3 * (4194304 + 262144 + 262144)  // 256  # 13,762,560 per expert

NIBBLE_TO_FLOAT = np.array(
    [0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
    dtype=np.float32)

COMPONENTS = [
    ("gate_proj", 2048, 4096),
    ("up_proj",   2048, 4096),
    ("down_proj", 4096, 2048),
]

def load_raw(path, key):
    with open(path, 'rb') as f:
        hlen = struct.unpack('<Q', f.read(8))[0]
        hdr  = json.loads(f.read(hlen))
    info = hdr[key]
    with open(path, 'rb') as f:
        f.seek(8 + hlen + info['data_offsets'][0])
        return f.read(info['data_offsets'][1] - info['data_offsets'][0])

def to_bf16(arr_f32):
    u = arr_f32.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

def process_component_batch(weight_b, scale_b, out_dim, in_dim):
    """Convert experts one at a time (32MB peak vs 1GB for batch).
    weight_b:  [B, out_dim, in_dim//8] uint32
    scale_b:   [B, out_dim, in_dim//MXFP4_GS] uint8
    Returns: (list_of_packed_bytes, list_of_scale_bf16_bytes, list_of_bias_bf16_bytes)
    """
    B = weight_b.shape[0]
    n_aff_grp = in_dim // AFFINE_GS
    shifts    = (np.arange(8, dtype=np.uint32) * 4)   # [8], reused across experts

    packed_list = []
    scale_list  = []
    bias_list   = []

    for i in range(B):
        w = weight_b[i]    # [out_dim, in_dim//8] uint32
        s = scale_b[i]     # [out_dim, in_dim//MXFP4_GS] uint8

        # Dequant MXFP4: [out_dim, in_dim//8, 8] → [out_dim, in_dim] float32
        nidx = (w[:, :, np.newaxis] >> shifts) & 0xF          # [out_dim, 512, 8] uint32
        vals = NIBBLE_TO_FLOAT[nidx].reshape(out_dim, in_dim)  # [out_dim, in_dim] float32

        # Apply E8M0 scale per MXFP4 group
        sf = np.exp2(s.astype(np.float32) - 127.0)            # [out_dim, n_mxfp4_grp]
        vals *= np.repeat(sf, MXFP4_GS, axis=1)               # broadcast

        # Affine requantize per gs=64 group
        grouped = vals.reshape(out_dim, n_aff_grp, AFFINE_GS)
        g_min   = grouped.min(axis=2)                          # [out_dim, n_aff_grp]
        g_max   = grouped.max(axis=2)
        g_rng   = g_max - g_min
        g_scale = np.where(g_rng > 1e-8, g_rng / 15.0, np.ones_like(g_rng))

        nq = np.clip(
            np.round((grouped - g_min[:, :, None]) / g_scale[:, :, None]),
            0, 15
        ).astype(np.uint8).reshape(out_dim, in_dim)            # [out_dim, in_dim]

        # Pack (vectorized): [out_dim, in_dim//8, 8] → [out_dim, in_dim//8] uint32
        packed = (
            nq.reshape(out_dim, in_dim // 8, 8).astype(np.uint32)
            << (np.arange(8, dtype=np.uint32) * 4)
        ).sum(axis=2).astype(np.uint32)

        # bf16 encode scale/bias
        def to_bf16(a):
            u = a.astype(np.float32).view(np.uint32)
            return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

        packed_list.append(packed.tobytes())
        scale_list.append(to_bf16(g_scale).tobytes())
        bias_list.append(to_bf16(g_min).tobytes())

    return packed_list, scale_list, bias_list

def process_layer(args):
    layer_idx, input_dir_str, output_dir_str = args
    t0 = time.time()
    input_dir  = Path(input_dir_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    import re
    with open(input_dir / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    # For each component: collect per-expert bytes
    comp_data = {}  # comp_name -> list of (packed, scale_bf16, bias_bf16) per expert
    for comp_name, out_dim, in_dim in COMPONENTS:
        w_key = f"model.layers.{layer_idx}.ffn.switch_mlp.{comp_name}.weight"
        s_key = f"model.layers.{layer_idx}.ffn.switch_mlp.{comp_name}.scales"
        shard  = weight_map[w_key]
        weight   = np.frombuffer(load_raw(input_dir / shard, w_key),
                                 dtype=np.uint32).reshape(256, out_dim, in_dim // 8)
        scale_u8 = np.frombuffer(load_raw(input_dir / shard, s_key),
                                 dtype=np.uint8).reshape(256, out_dim, in_dim // MXFP4_GS)

        packed_list, scale_list, bias_list = [], [], []
        p_b, s_b, b_b = process_component_batch(
            weight, scale_u8, out_dim, in_dim)
        packed_list = p_b
        scale_list  = s_b
        bias_list   = b_b

        comp_data[comp_name] = (packed_list, scale_list, bias_list)

    # Write expert-major: for each expert, write gate(W+S+B), up(W+S+B), down(W+S+B)
    out_path = output_dir / f"layer_{layer_idx:02d}.bin"
    with open(out_path, "wb") as f:
        for eid in range(256):
            for comp_name, _, _ in COMPONENTS:
                pk, sc, bi = comp_data[comp_name]
                f.write(pk[eid]); f.write(sc[eid]); f.write(bi[eid])

    actual   = out_path.stat().st_size
    expected = 256 * EXPERT_SIZE
    elapsed  = time.time() - t0
    status   = "OK" if actual == expected else f"SIZE_ERR({actual}!={expected})"
    return layer_idx, elapsed, status

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)

    input_dir  = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(Path(input_dir) / "packed_experts_affine_v2")

    import re
    with open(Path(input_dir) / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    moe_layers = sorted({
        int(m.group(1))
        for k in weight_map
        if "switch_mlp" in k
        for m in [re.search(r'layers\.(\d+)\.', k)] if m
    })
    print(f"MoE layers: {len(moe_layers)} ({moe_layers[0]}..{moe_layers[-1]})")

    import os
    nworkers = min(4, os.cpu_count() or 2)  # ThreadPoolExecutor — numpy releases GIL
    print(f"Workers: {nworkers}  Output: {output_dir}")

    args = [(li, input_dir, output_dir) for li in moe_layers]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=nworkers) as executor:
        futures = {executor.submit(process_layer, a): a[0] for a in args}
        for fut in as_completed(futures):
            layer_idx_done, elapsed, status = fut.result()
            print(f"  Layer {layer_idx_done:02d}: {elapsed:.1f}s  {status}", flush=True)

    total = time.time() - t0
    print(f"\nDone in {total:.0f}s ({total/60:.1f}min)")
    print(f"Output: {output_dir}")

    manifest = {
        "version": 5,
        "format": "affine_4bit_gs64_bf16",
        "group_size": AFFINE_GS,
        "expert_size": EXPERT_SIZE,
        "layout": "per expert: gate(W uint32, S bf16, B bf16), up(...), down(...)",
    }
    with open(Path(output_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    main()
