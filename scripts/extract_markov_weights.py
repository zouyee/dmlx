#!/usr/bin/env python3
"""Extract DSpark Markov Head weights from DeepSeek-V4-Flash-DSpark safetensors.

Downloads only the specific safetensors shard(s) containing the Markov Head
weights, then saves them as raw f32 binary files for the dmlx native engine.

Usage:
    # From HuggingFace (downloads only needed shards, ~500MB):
    python3 scripts/extract_markov_weights.py \
        --model deepseek-ai/DeepSeek-V4-Flash-DSpark \
        --output ~/models/DeepSeek-V4-Flash-4bit/dspark/

    # From local path:
    python3 scripts/extract_markov_weights.py \
        --model /path/to/DeepSeek-V4-Flash-DSpark \
        --output ~/models/DeepSeek-V4-Flash-4bit/dspark/

Output files:
    markov_w1.bin  - [vocab_size, markov_rank] f32, row-major (Embedding weight)
    markov_w2.bin  - [vocab_size, markov_rank] f32, row-major (Linear weight, transposed)
    dspark_config.json - metadata (vocab_size, markov_rank, block_size)
"""

import argparse
import json
import os
import sys
import struct
from pathlib import Path

import numpy as np

# Weight key patterns in the DSpark checkpoint:
# model.mtp.{layer_idx}.markov_head.markov_w1.weight  -> [vocab_size, markov_rank]
# model.mtp.{layer_idx}.markov_head.markov_w2.weight  -> [vocab_size, markov_rank]
# (markov_w2 is nn.Linear(markov_rank, vocab_size, bias=False) so .weight is [vocab_size, markov_rank])
MARKOV_W1_PATTERN = "markov_head.markov_w1.weight"
MARKOV_W2_PATTERN = "markov_head.markov_w2.weight"


def load_from_huggingface(model_id: str, cache_dir: str = None):
    """Load only Markov Head weights from HuggingFace, downloading minimal shards."""
    try:
        from huggingface_hub import hf_hub_download, HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    try:
        from safetensors import safe_open
    except ImportError:
        print("ERROR: safetensors not installed. Run: pip install safetensors", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching model index from {model_id}...")
    # Download the index file to find which shard has Markov weights
    index_path = hf_hub_download(
        model_id, "model.safetensors.index.json", cache_dir=cache_dir
    )
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Find shards containing Markov Head weights
    markov_shards = set()
    w1_key = None
    w2_key = None
    for key, shard in weight_map.items():
        if MARKOV_W1_PATTERN in key:
            markov_shards.add(shard)
            w1_key = key
        elif MARKOV_W2_PATTERN in key:
            markov_shards.add(shard)
            w2_key = key

    if not w1_key or not w2_key:
        print(f"ERROR: Could not find Markov Head weights in {model_id}", file=sys.stderr)
        print(f"  Searched for patterns: '{MARKOV_W1_PATTERN}', '{MARKOV_W2_PATTERN}'", file=sys.stderr)
        # Try listing all mtp-related keys
        mtp_keys = [k for k in weight_map if "mtp" in k.lower()]
        if mtp_keys:
            print(f"  Found mtp-related keys: {mtp_keys[:10]}", file=sys.stderr)
        sys.exit(1)

    print(f"  Found markov_w1: {w1_key} in {weight_map[w1_key]}")
    print(f"  Found markov_w2: {w2_key} in {weight_map[w2_key]}")
    print(f"  Need to download {len(markov_shards)} shard(s): {sorted(markov_shards)}")

    # Download needed shards
    w1_tensor = None
    w2_tensor = None
    for shard_name in sorted(markov_shards):
        print(f"  Downloading {shard_name}...")
        shard_path = hf_hub_download(model_id, shard_name, cache_dir=cache_dir)
        with safe_open(shard_path, framework="numpy") as f:
            if w1_key in f.keys():
                w1_tensor = f.get_tensor(w1_key)
                print(f"    Loaded {w1_key}: shape={w1_tensor.shape}, dtype={w1_tensor.dtype}")
            if w2_key in f.keys():
                w2_tensor = f.get_tensor(w2_key)
                print(f"    Loaded {w2_key}: shape={w2_tensor.shape}, dtype={w2_tensor.dtype}")

    # Also download config.json
    config_path = hf_hub_download(model_id, "config.json", cache_dir=cache_dir)
    with open(config_path) as f:
        config = json.load(f)

    return w1_tensor, w2_tensor, config


def load_from_local(model_path: str):
    """Load Markov Head weights from a local model directory."""
    try:
        from safetensors import safe_open
    except ImportError:
        print("ERROR: safetensors not installed. Run: pip install safetensors", file=sys.stderr)
        sys.exit(1)

    model_path = Path(model_path)
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        print(f"ERROR: {index_file} not found", file=sys.stderr)
        sys.exit(1)

    with open(index_file) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    w1_key = None
    w2_key = None
    for key in weight_map:
        if MARKOV_W1_PATTERN in key:
            w1_key = key
        elif MARKOV_W2_PATTERN in key:
            w2_key = key

    if not w1_key or not w2_key:
        print(f"ERROR: Could not find Markov Head weights in {model_path}", file=sys.stderr)
        sys.exit(1)

    w1_tensor = None
    w2_tensor = None
    for key, shard_name in [(w1_key, weight_map[w1_key]), (w2_key, weight_map[w2_key])]:
        shard_path = model_path / shard_name
        with safe_open(str(shard_path), framework="numpy") as f:
            tensor = f.get_tensor(key)
            if key == w1_key:
                w1_tensor = tensor
            else:
                w2_tensor = tensor
            print(f"  Loaded {key}: shape={tensor.shape}, dtype={tensor.dtype}")

    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    return w1_tensor, w2_tensor, config


def save_weights(w1: np.ndarray, w2: np.ndarray, config: dict, output_dir: str):
    """Save Markov Head weights as raw f32 binary files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vocab_size = config.get("vocab_size", 129280)
    markov_rank = config.get("dspark_markov_rank", 256)
    block_size = config.get("dspark_block_size", 5)

    # Convert to f32 if needed (checkpoint is bf16 or fp8)
    w1_f32 = w1.astype(np.float32)
    w2_f32 = w2.astype(np.float32)

    # Validate shapes
    assert w1_f32.shape == (vocab_size, markov_rank), \
        f"markov_w1 shape mismatch: got {w1_f32.shape}, expected ({vocab_size}, {markov_rank})"
    assert w2_f32.shape == (vocab_size, markov_rank), \
        f"markov_w2 shape mismatch: got {w2_f32.shape}, expected ({vocab_size}, {markov_rank})"

    # Save as raw f32 binary (row-major, directly mmap-able from Zig)
    w1_path = output_path / "markov_w1.bin"
    w2_path = output_path / "markov_w2.bin"

    w1_f32.tofile(str(w1_path))
    w2_f32.tofile(str(w2_path))

    # Save metadata
    dspark_config = {
        "vocab_size": vocab_size,
        "markov_rank": markov_rank,
        "block_size": block_size,
        "w1_shape": list(w1_f32.shape),
        "w2_shape": list(w2_f32.shape),
        "dtype": "float32",
        "layout": "row_major",
        "description": "DSpark Markov Head weights extracted from DeepSeek-V4-Flash-DSpark",
        "usage": (
            "markov_w1[token_id] gives a [markov_rank] embedding vector. "
            "matmul(markov_w1[token_id], markov_w2.T) gives [vocab_size] logit bias."
        ),
    }
    config_path = output_path / "dspark_config.json"
    with open(config_path, "w") as f:
        json.dump(dspark_config, f, indent=2)

    w1_size_mb = w1_f32.nbytes / (1024 * 1024)
    w2_size_mb = w2_f32.nbytes / (1024 * 1024)
    print(f"\nSaved to {output_path}/:")
    print(f"  markov_w1.bin  [{vocab_size} x {markov_rank}] f32  ({w1_size_mb:.1f} MB)")
    print(f"  markov_w2.bin  [{vocab_size} x {markov_rank}] f32  ({w2_size_mb:.1f} MB)")
    print(f"  dspark_config.json")
    print(f"  Total: {w1_size_mb + w2_size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Extract DSpark Markov Head weights for dmlx native engine"
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model ID (e.g. deepseek-ai/DeepSeek-V4-Flash-DSpark) or local path"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for binary weight files"
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="HuggingFace cache directory (optional)"
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if model_path.is_dir() and (model_path / "config.json").exists():
        print(f"Loading from local path: {args.model}")
        w1, w2, config = load_from_local(args.model)
    else:
        print(f"Loading from HuggingFace: {args.model}")
        w1, w2, config = load_from_huggingface(args.model, args.cache_dir)

    if w1 is None or w2 is None:
        print("ERROR: Failed to load weights", file=sys.stderr)
        sys.exit(1)

    save_weights(w1, w2, config, args.output)
    print("\nDone! To use with dmlx:")
    print(f"  ./zig-out/bin/dmlx serve --model ~/models/DeepSeek-V4-Flash-4bit \\")
    print(f"    --expert-packed-dir ~/models/DeepSeek-V4-Flash-4bit/packed_experts \\")
    print(f"    --dspark {args.output} \\")
    print(f"    --native --port 8930")


if __name__ == "__main__":
    main()
