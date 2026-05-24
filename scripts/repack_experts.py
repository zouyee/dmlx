#!/usr/bin/env python3
"""Repack MoE expert weights from safetensors into per-layer contiguous binary files.

This enables parallel pread() loading instead of mmap page faults.
Each expert's gate/up/down weights + scales are packed into a contiguous blob.
Per-layer file stores all 256 experts sequentially for O(1) offset calculation.

Usage:
    python3 scripts/repack_experts.py <model_dir> [--output-dir <dir>]

Example:
    python3 scripts/repack_experts.py ~/models/DeepSeek-V4-Flash-4bit

Output:
    <model_dir>/packed_experts/layer_00.bin  (256 experts × EXPERT_SIZE bytes)
    <model_dir>/packed_experts/layer_01.bin
    ...
    <model_dir>/packed_experts/manifest.json  (metadata for loader)
"""
import sys
import os
import json
import struct
import time
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors package required. Install with: pip install safetensors")
    sys.exit(1)

# DeepSeek V4 Flash 4-bit (MXFP4) expert layout per expert:
#   gate_proj.weight: [2048, 512] uint32 = 4,194,304 bytes
#   gate_proj.scales: [2048, 128] uint8  =   262,144 bytes
#   up_proj.weight:   [2048, 512] uint32 = 4,194,304 bytes
#   up_proj.scales:   [2048, 128] uint8  =   262,144 bytes
#   down_proj.weight: [4096, 256] uint32 = 4,194,304 bytes
#   down_proj.scales: [4096, 64]  uint8  =   262,144 bytes
#   Total per expert: 13,369,728 bytes (~12.75 MB)

COMPONENTS = [
    "ffn.switch_mlp.gate_proj.weight",
    "ffn.switch_mlp.gate_proj.scales",
    "ffn.switch_mlp.up_proj.weight",
    "ffn.switch_mlp.up_proj.scales",
    "ffn.switch_mlp.down_proj.weight",
    "ffn.switch_mlp.down_proj.scales",
]


def find_safetensor_files(model_dir: Path) -> list[Path]:
    """Find all safetensor shard files in order."""
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        files = sorted(set(index["weight_map"].values()))
        return [model_dir / f for f in files]
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"No safetensors files found in {model_dir}")


def detect_moe_layers(weight_map: dict) -> list[int]:
    """Detect which layers have MoE (switch_mlp) weights."""
    layers = set()
    for key in weight_map:
        if "ffn.switch_mlp.gate_proj.weight" in key:
            parts = key.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layers.add(int(parts[i + 1]))
                    except ValueError:
                        pass
    return sorted(layers)


def get_expert_size_for_tensor(handle, tensor_name: str, n_experts: int) -> int:
    """Get per-expert byte size for a fused tensor [n_experts, ...]."""
    tensor = handle.get_tensor(tensor_name)
    total_bytes = tensor.nbytes
    return total_bytes // n_experts


def repack_layer(
    layer_idx: int,
    handles: dict,  # filename -> safe_open handle
    weight_map: dict,  # tensor_name -> filename
    output_dir: Path,
    n_experts: int = 256,
) -> dict:
    """Repack one layer's experts into a contiguous binary file.

    Returns metadata dict with component sizes.
    """
    prefix = f"model.layers.{layer_idx}."

    # First pass: determine per-expert sizes for each component
    component_sizes = {}
    for comp in COMPONENTS:
        tensor_name = prefix + comp
        if tensor_name not in weight_map:
            print(f"  WARNING: {tensor_name} not found, skipping component")
            component_sizes[comp] = 0
            continue
        filename = weight_map[tensor_name]
        handle = handles[filename]
        size = get_expert_size_for_tensor(handle, tensor_name, n_experts)
        component_sizes[comp] = size

    expert_size = sum(component_sizes.values())
    if expert_size == 0:
        return {}

    # Write binary file
    out_path = output_dir / f"layer_{layer_idx:02d}.bin"
    with open(out_path, "wb") as f:
        for expert_id in range(n_experts):
            for comp in COMPONENTS:
                tensor_name = prefix + comp
                if component_sizes[comp] == 0:
                    continue
                filename = weight_map[tensor_name]
                handle = handles[filename]
                tensor = handle.get_tensor(tensor_name)
                # tensor shape: [n_experts, ...] — slice expert_id along axis 0
                expert_data = tensor[expert_id].tobytes()
                assert len(expert_data) == component_sizes[comp], (
                    f"Size mismatch for {tensor_name}[{expert_id}]: "
                    f"got {len(expert_data)}, expected {component_sizes[comp]}"
                )
                f.write(expert_data)

    return {
        "expert_size": expert_size,
        "component_sizes": component_sizes,
        "n_experts": n_experts,
        "file": out_path.name,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    output_dir = None
    if "--output-dir" in sys.argv:
        idx = sys.argv.index("--output-dir")
        output_dir = Path(sys.argv[idx + 1])
    else:
        output_dir = model_dir / "packed_experts"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model dir: {model_dir}")
    print(f"Output dir: {output_dir}")

    # Find safetensor files
    st_files = find_safetensor_files(model_dir)
    print(f"Found {len(st_files)} safetensor shard(s)")

    # Build weight map
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        # Single file — all tensors in one file
        weight_map = {}
        with safe_open(st_files[0], framework="numpy") as f:
            for key in f.keys():
                weight_map[key] = st_files[0].name

    # Detect MoE layers
    moe_layers = detect_moe_layers(weight_map)
    print(f"MoE layers: {len(moe_layers)} ({moe_layers[0]}..{moe_layers[-1]})")

    # Open all shard files
    handles = {}
    for st_file in st_files:
        handles[st_file.name] = safe_open(st_file, framework="numpy")

    # Repack each layer
    manifest = {
        "version": 1,
        "model_dir": str(model_dir),
        "n_experts": 256,
        "layers": {},
    }

    t0 = time.time()
    for i, layer_idx in enumerate(moe_layers):
        t_layer = time.time()
        print(f"  [{i+1}/{len(moe_layers)}] Layer {layer_idx}...", end="", flush=True)
        meta = repack_layer(layer_idx, handles, weight_map, output_dir)
        if meta:
            manifest["layers"][str(layer_idx)] = meta
            elapsed = time.time() - t_layer
            size_gb = meta["expert_size"] * meta["n_experts"] / (1024**3)
            print(f" {size_gb:.2f} GB ({elapsed:.1f}s)")
        else:
            print(" SKIPPED (no data)")

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - t0
    total_size = sum(
        m["expert_size"] * m["n_experts"]
        for m in manifest["layers"].values()
    )
    print(f"\nDone in {total_time:.0f}s")
    print(f"Total size: {total_size / (1024**3):.1f} GB")
    print(f"Manifest: {manifest_path}")
    print(f"\nUsage: dmlx serve --expert-format packed --model {model_dir}")


if __name__ == "__main__":
    main()
