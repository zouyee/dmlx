#!/usr/bin/env python3
"""Repack DSpark MTP expert weights (INT8 + E8M0 scale) into per-layer binary files.

Each MTP layer has 256 experts with gate(w1), down(w2), up(w3) projections.
Packs them into the same pread-friendly layout as the target model's packed_experts.

Layout per expert (6 slots):
  w1.weight  [2048, 2048] int8  =  4,194,304 bytes (gate)
  w1.scale   [2048, 128]  uint8 =    262,144 bytes
  w3.weight  [2048, 2048] int8  =  4,194,304 bytes (up)
  w3.scale   [2048, 128]  uint8 =    262,144 bytes
  w2.weight  [4096, 1024] int8  =  4,194,304 bytes (down)
  w2.scale   [4096, 64]   uint8 =    262,144 bytes
  Total per expert: 13,369,344 bytes (~12.75 MB)

Note: Order is gate/up/down (w1/w3/w2) to match target's gate_proj/up_proj/down_proj order.

Usage:
    python3 scripts/repack_mtp_experts.py <dspark_shard_dir> [--output-dir <dir>]

Example:
    python3 scripts/repack_mtp_experts.py ~/models/DeepSeek-V4-Flash-DSpark

Output:
    <output_dir>/mtp_layer_00.bin  (256 experts)
    <output_dir>/mtp_layer_01.bin
    <output_dir>/mtp_layer_02.bin
    <output_dir>/mtp_manifest.json
"""
import sys
import os
import json
import time
import struct
import numpy as np
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors package required. Install with: pip install safetensors")
    sys.exit(1)


N_MTP_LAYERS = 3
N_EXPERTS = 256

# Per-expert component order: gate(w1), up(w3), down(w2) — matches target layout
COMPONENTS = [
    ("w1", "weight"),  # gate
    ("w1", "scale"),
    ("w3", "weight"),  # up
    ("w3", "scale"),
    ("w2", "weight"),  # down
    ("w2", "scale"),
]


def read_raw_tensor(shard_path: str, header: dict, key: str) -> bytes:
    """Read a tensor as raw bytes using safetensors header offsets.
    Needed for dtypes numpy doesn't support (F8_E8M0, BF16, F8_E4M3)."""
    info = header[key]
    start, end = info['data_offsets']
    # Header size is stored in first 8 bytes of file
    with open(shard_path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        data_offset = 8 + header_size
        f.seek(data_offset + start)
        return f.read(end - start)


def load_shard_header(shard_path: str) -> dict:
    """Load safetensors header (tensor metadata) without loading tensor data."""
    with open(shard_path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header_bytes = f.read(header_size)
    header = json.loads(header_bytes)
    header.pop('__metadata__', None)
    return header


def find_mtp_shards(model_dir: Path) -> dict:
    """Find which shards contain MTP weights using the index file."""
    index_file = model_dir / "model.safetensors.index.json"
    if not index_file.exists():
        # Try the meta directory
        meta_dir = model_dir.parent / (model_dir.name + "-meta")
        index_file = meta_dir / "model.safetensors.index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"No model.safetensors.index.json found")

    with open(index_file) as f:
        idx = json.load(f)

    # Find shards containing mtp.* weights
    mtp_shards = set()
    for key, shard in idx["weight_map"].items():
        if key.startswith("mtp."):
            mtp_shards.add(shard)

    # Map shard filename to full path
    shard_paths = {}
    for shard in sorted(mtp_shards):
        path = model_dir / shard
        if path.exists():
            shard_paths[shard] = path
        else:
            print(f"WARNING: shard {shard} not found at {path}")

    return shard_paths


def repack_mtp_layer(
    mtp_layer_idx: int,
    handles: dict,
    shard_headers: dict,
    shard_paths: dict,
    output_dir: Path,
) -> dict:
    """Repack one MTP layer's 256 experts into a contiguous binary file.

    Uses get_tensor() for INT8 weights (numpy supports int8) and
    raw file reads for E8M0 scales (numpy doesn't support float8_e8m0fnu).
    """
    prefix = f"mtp.{mtp_layer_idx}.ffn.experts."

    # Verify first expert exists
    test_key = f"{prefix}0.w1.weight"
    found = False
    for fname, handle in handles.items():
        if test_key in handle.keys():
            found = True
            break
    if not found:
        print(f"  ERROR: {test_key} not found in any shard")
        return {}

    # Determine component sizes from expert 0 using header metadata
    component_sizes = {}
    for proj, attr in COMPONENTS:
        key = f"{prefix}0.{proj}.{attr}"
        for fname, header in shard_headers.items():
            if key in header:
                info = header[key]
                start, end = info['data_offsets']
                component_sizes[(proj, attr)] = end - start
                break
        else:
            print(f"  WARNING: {key} not found in any header")
            component_sizes[(proj, attr)] = 0

    expert_size = sum(component_sizes.values())
    print(f"    Expert size: {expert_size:,} bytes ({expert_size/1024/1024:.2f} MB)")

    # Write binary file
    out_path = output_dir / f"mtp_layer_{mtp_layer_idx:02d}.bin"
    written = 0
    with open(out_path, "wb") as f:
        for expert_id in range(N_EXPERTS):
            for proj, attr in COMPONENTS:
                key = f"{prefix}{expert_id}.{proj}.{attr}"
                size = component_sizes[(proj, attr)]
                if size == 0:
                    continue

                data = None
                if attr == "weight":
                    # INT8 — numpy can read this directly
                    for fname, handle in handles.items():
                        if key in handle.keys():
                            tensor = handle.get_tensor(key)
                            data = tensor.tobytes()
                            break
                else:
                    # E8M0 scale — must read as raw bytes
                    for fname, header in shard_headers.items():
                        if key in header:
                            data = read_raw_tensor(str(shard_paths[fname]), header, key)
                            break

                if data is None:
                    print(f"  ERROR: {key} not found")
                    f.write(b'\x00' * size)
                    written += size
                    continue

                assert len(data) == size, f"Size mismatch: {key} got {len(data)} expected {size}"
                f.write(data)
                written += size

            if (expert_id + 1) % 64 == 0:
                print(f"    packed {expert_id + 1}/{N_EXPERTS} experts...", flush=True)

    assert written == expert_size * N_EXPERTS, f"Written {written} != expected {expert_size * N_EXPERTS}"
    return {
        "expert_size": expert_size,
        "n_experts": N_EXPERTS,
        "file": out_path.name,
        "component_sizes": {f"{p}.{a}": s for (p, a), s in component_sizes.items()},
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
        output_dir = model_dir / "packed_mtp_experts"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"DSpark model dir: {model_dir}")
    print(f"Output dir: {output_dir}")

    # Find MTP shards
    shard_paths = find_mtp_shards(model_dir)
    if not shard_paths:
        print("ERROR: No MTP shards found")
        sys.exit(1)
    print(f"Found {len(shard_paths)} MTP shard(s): {sorted(shard_paths.keys())}")

    # Open shard files (numpy framework to avoid torch dependency)
    handles = {}
    shard_headers = {}
    for fname, path in shard_paths.items():
        print(f"  Opening {fname}...")
        handles[fname] = safe_open(str(path), framework="numpy")
        shard_headers[fname] = load_shard_header(str(path))

    # Repack each MTP layer
    manifest = {
        "version": 1,
        "format": "int8_e8m0",
        "model_dir": str(model_dir),
        "n_mtp_layers": N_MTP_LAYERS,
        "n_experts": N_EXPERTS,
        "layers": {},
    }

    t0 = time.time()
    for layer_idx in range(N_MTP_LAYERS):
        print(f"\n  [{layer_idx + 1}/{N_MTP_LAYERS}] MTP Layer {layer_idx}...")
        t_layer = time.time()
        meta = repack_mtp_layer(layer_idx, handles, shard_headers, shard_paths, output_dir)
        if meta:
            manifest["layers"][str(layer_idx)] = meta
            elapsed = time.time() - t_layer
            size_gb = meta["expert_size"] * meta["n_experts"] / (1024**3)
            print(f"    Done: {size_gb:.2f} GB ({elapsed:.1f}s)")
        else:
            print(f"    SKIPPED")

    # Write manifest
    manifest_path = output_dir / "mtp_manifest.json"
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


if __name__ == "__main__":
    main()
