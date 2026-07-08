#!/usr/bin/env python3
"""Extract DSpark MTP non-expert weights from safetensors into binary files.

Extracts:
  - Markov Head: markov_w1.bin, markov_w2.bin (BF16 → f32)
  - Per-layer attention weights (FP8 → raw bytes)
  - Per-layer norms (BF16 → f32)
  - Per-layer HC params (F32 → raw)
  - Per-layer gate weights (BF16 → f32, F32 bias → raw)
  - main_proj (FP8, mtp.0 only)
  - confidence_head (BF16 → f32, mtp.2 only)
  - dspark_config.json

Usage:
    python3 scripts/extract_dspark_weights.py <dspark_shard_dir> [--output-dir <dir>]

Example:
    python3 scripts/extract_dspark_weights.py ~/models/DeepSeek-V4-Flash-DSpark
"""
import sys
import os
import json
import struct
import time
import numpy as np
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors required. pip install safetensors")
    sys.exit(1)


def load_shard_header(shard_path: str) -> dict:
    """Load safetensors header metadata."""
    with open(shard_path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header_bytes = f.read(header_size)
    header = json.loads(header_bytes)
    header.pop('__metadata__', None)
    return header


def read_raw_tensor(shard_path: str, header: dict, key: str) -> bytes:
    """Read tensor as raw bytes (for unsupported dtypes like BF16, F8_E4M3, F8_E8M0)."""
    info = header[key]
    start, end = info['data_offsets']
    with open(shard_path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        f.seek(8 + header_size + start)
        return f.read(end - start)


def bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    """Convert BF16 raw bytes to float32 numpy array."""
    bf16 = np.frombuffer(raw, dtype=np.uint16)
    # BF16 → F32: left-shift 16 bits into float32
    f32_bits = bf16.astype(np.uint32) << 16
    return f32_bits.view(np.float32)


def fp8_e4m3_bytes_to_f32(raw: bytes) -> np.ndarray:
    """Convert FP8 E4M3 raw bytes to float32 numpy array.
    FP8 E4M3: 1 sign + 4 exponent + 3 mantissa, bias=7."""
    u8 = np.frombuffer(raw, dtype=np.uint8)
    sign = (u8 >> 7).astype(np.int32)
    exp = ((u8 >> 3) & 0xF).astype(np.int32)
    man = (u8 & 0x7).astype(np.int32)

    result = np.zeros(len(u8), dtype=np.float32)
    # Normal: (-1)^sign * 2^(exp-7) * (1 + man/8)
    normal_mask = exp > 0
    result[normal_mask] = ((-1.0) ** sign[normal_mask]) * \
        (2.0 ** (exp[normal_mask] - 7)) * (1.0 + man[normal_mask] / 8.0)
    # Subnormal: (-1)^sign * 2^(-6) * (man/8)
    subnormal_mask = (exp == 0) & (man > 0)
    result[subnormal_mask] = ((-1.0) ** sign[subnormal_mask]) * \
        (2.0 ** (-6)) * (man[subnormal_mask] / 8.0)
    return result


def find_mtp_shards(model_dir: Path) -> dict:
    """Find shards containing MTP weights."""
    # Try index in model_dir first, then meta dir
    for search_dir in [model_dir, model_dir.parent / (model_dir.name + "-meta")]:
        index_file = search_dir / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file) as f:
                idx = json.load(f)
            mtp_shards = set()
            for key, shard in idx["weight_map"].items():
                if key.startswith("mtp."):
                    mtp_shards.add(shard)
            return {s: model_dir / s for s in sorted(mtp_shards) if (model_dir / s).exists()}
    raise FileNotFoundError("No model.safetensors.index.json found")


def extract_tensor(key: str, handles: dict, headers: dict, shard_paths: dict, dtype_hint: str = None):
    """Extract a tensor, handling various dtypes.
    Returns (numpy_f32_array, original_dtype, shape) or (raw_bytes, dtype, shape) for FP8."""
    # Find which shard has this key
    for fname, header in headers.items():
        if key in header:
            info = header[key]
            shape = info['shape']
            dtype = info['dtype']
            raw = read_raw_tensor(str(shard_paths[fname]), header, key)

            if dtype == 'F32':
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
                return arr, dtype, shape
            elif dtype == 'BF16':
                arr = bf16_bytes_to_f32(raw).reshape(shape)
                return arr, dtype, shape
            elif dtype == 'I8':
                arr = np.frombuffer(raw, dtype=np.int8).reshape(shape)
                return arr, dtype, shape
            elif dtype in ('F8_E4M3', 'F8_E8M0'):
                # Return raw bytes + metadata (native engine will handle decoding)
                return raw, dtype, shape
            else:
                print(f"  WARNING: unsupported dtype {dtype} for {key}, returning raw")
                return raw, dtype, shape
    return None, None, None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    output_dir = model_dir / "dspark_weights" if "--output-dir" not in sys.argv \
        else Path(sys.argv[sys.argv.index("--output-dir") + 1])

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"DSpark model dir: {model_dir}")
    print(f"Output dir: {output_dir}")

    shard_paths = find_mtp_shards(model_dir)
    print(f"Found {len(shard_paths)} MTP shard(s)")

    handles = {}
    headers = {}
    for fname, path in shard_paths.items():
        print(f"  Loading {fname}...")
        handles[fname] = safe_open(str(path), framework="numpy")
        headers[fname] = load_shard_header(str(path))

    manifest = {"version": 1, "files": {}}
    t0 = time.time()

    # --- Markov Head (mtp.2 only) ---
    print("\n=== Markov Head ===")
    for wname in ["markov_w1", "markov_w2"]:
        key = f"mtp.2.markov_head.{wname}.weight"
        arr, dtype, shape = extract_tensor(key, handles, headers, shard_paths)
        if arr is not None:
            out_path = output_dir / f"{wname}.bin"
            arr.astype(np.float32).tofile(str(out_path))
            print(f"  {key}: {shape} {dtype} → {out_path.name} ({arr.nbytes/1e6:.1f}MB f32)")
            manifest["files"][wname] = {"shape": shape, "dtype": "f32", "file": out_path.name,
                                        "original_dtype": dtype, "bytes": arr.astype(np.float32).nbytes}

    # --- Confidence Head (mtp.2 only) ---
    print("\n=== Confidence Head ===")
    key = "mtp.2.confidence_head.proj.weight"
    arr, dtype, shape = extract_tensor(key, handles, headers, shard_paths)
    if arr is not None:
        out_path = output_dir / "confidence_head.bin"
        arr.astype(np.float32).tofile(str(out_path))
        print(f"  {key}: {shape} {dtype} → {out_path.name} ({arr.astype(np.float32).nbytes}B)")
        manifest["files"]["confidence_head"] = {"shape": shape, "dtype": "f32", "file": out_path.name}

    # --- Per-layer weights ---
    for layer_idx in range(3):
        print(f"\n=== MTP Layer {layer_idx} ===")
        layer_dir = output_dir / f"layer_{layer_idx:02d}"
        layer_dir.mkdir(exist_ok=True)
        layer_files = {}

        # Attention weights (FP8 → stored as raw bytes for native engine)
        attn_keys = [
            "attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b",
        ]
        for attn_key in attn_keys:
            for suffix in ["weight", "scale"]:
                full_key = f"mtp.{layer_idx}.{attn_key}.{suffix}"
                data, dtype, shape = extract_tensor(full_key, handles, headers, shard_paths)
                if data is not None:
                    fname = f"{attn_key.replace('.', '_')}_{suffix}.bin"
                    out_path = layer_dir / fname
                    if isinstance(data, np.ndarray):
                        data.tofile(str(out_path))
                        nbytes = data.nbytes
                    else:
                        with open(out_path, 'wb') as f:
                            f.write(data)
                        nbytes = len(data)
                    layer_files[fname] = {"shape": shape, "dtype": dtype, "bytes": nbytes}
                    print(f"  {full_key}: {shape} {dtype} → {fname} ({nbytes/1024:.0f}KB)")

        # Attention sink + norms (F32/BF16 → f32)
        for norm_key in ["attn.attn_sink", "attn.q_norm.weight", "attn.kv_norm.weight",
                         "attn_norm.weight", "ffn_norm.weight"]:
            full_key = f"mtp.{layer_idx}.{norm_key}"
            arr, dtype, shape = extract_tensor(full_key, handles, headers, shard_paths)
            if arr is not None:
                fname = f"{norm_key.replace('.', '_')}.bin"
                out_path = layer_dir / fname
                arr.astype(np.float32).tofile(str(out_path))
                layer_files[fname] = {"shape": shape, "dtype": "f32", "bytes": arr.astype(np.float32).nbytes}
                print(f"  {full_key}: {shape} {dtype} → {fname}")

        # Gate weight (BF16 → f32) + bias (F32)
        for gate_key, out_name in [("ffn.gate.weight", "gate_weight.bin"),
                                    ("ffn.gate.bias", "gate_bias.bin")]:
            full_key = f"mtp.{layer_idx}.{gate_key}"
            arr, dtype, shape = extract_tensor(full_key, handles, headers, shard_paths)
            if arr is not None:
                out_path = layer_dir / out_name
                arr.astype(np.float32).tofile(str(out_path))
                layer_files[out_name] = {"shape": shape, "dtype": "f32", "bytes": arr.astype(np.float32).nbytes}
                print(f"  {full_key}: {shape} {dtype} → {out_name}")

        # HC params (F32, already float32)
        for hc_key in ["hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                       "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"]:
            full_key = f"mtp.{layer_idx}.{hc_key}"
            arr, dtype, shape = extract_tensor(full_key, handles, headers, shard_paths)
            if arr is not None:
                fname = f"{hc_key}.bin"
                out_path = layer_dir / fname
                arr.astype(np.float32).tofile(str(out_path))
                layer_files[fname] = {"shape": shape, "dtype": "f32", "bytes": arr.astype(np.float32).nbytes}

        # mtp.0 special: main_proj (FP8)
        if layer_idx == 0:
            for suffix in ["weight", "scale"]:
                full_key = f"mtp.0.main_proj.{suffix}"
                data, dtype, shape = extract_tensor(full_key, handles, headers, shard_paths)
                if data is not None:
                    fname = f"main_proj_{suffix}.bin"
                    out_path = layer_dir / fname
                    if isinstance(data, np.ndarray):
                        data.tofile(str(out_path))
                        nbytes = data.nbytes
                    else:
                        with open(out_path, 'wb') as f:
                            f.write(data)
                        nbytes = len(data)
                    layer_files[fname] = {"shape": shape, "dtype": dtype, "bytes": nbytes}
                    print(f"  {full_key}: {shape} {dtype} → {fname} ({nbytes/1024:.0f}KB)")

            # main_norm
            full_key = "mtp.0.main_norm.weight"
            arr, dtype, shape = extract_tensor(full_key, handles, headers, shard_paths)
            if arr is not None:
                fname = "main_norm_weight.bin"
                out_path = layer_dir / fname
                arr.astype(np.float32).tofile(str(out_path))
                layer_files[fname] = {"shape": shape, "dtype": "f32", "bytes": arr.astype(np.float32).nbytes}
                print(f"  {full_key}: {shape} {dtype} → {fname}")

        # mtp.2 special: hc_head + norm
        if layer_idx == 2:
            for hc_key in ["hc_head_fn", "hc_head_base", "hc_head_scale", "norm.weight"]:
                full_key = f"mtp.2.{hc_key}"
                arr, dtype, shape = extract_tensor(full_key, handles, headers, shard_paths)
                if arr is not None:
                    fname = f"{hc_key.replace('.', '_')}.bin"
                    out_path = layer_dir / fname
                    arr.astype(np.float32).tofile(str(out_path))
                    layer_files[fname] = {"shape": shape, "dtype": "f32", "bytes": arr.astype(np.float32).nbytes}
                    print(f"  {full_key}: {shape} {dtype} → {fname}")

        manifest["files"][f"layer_{layer_idx}"] = layer_files

    # --- Write DSpark config ---
    dspark_config = {
        "n_mtp_layers": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [40, 41, 42],
        "dspark_markov_rank": 256,
        "vocab_size": 129280,
        "dim": 4096,
        "moe_inter_dim": 2048,
        "n_routed_experts": 256,
        "n_activated_experts": 6,
        "head_dim": 512,
        "rope_head_dim": 64,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "window_size": 128,
        "hc_mult": 4,
    }
    config_path = output_dir / "dspark_config.json"
    with open(config_path, 'w') as f:
        json.dump(dspark_config, f, indent=2)
    print(f"\n  Config: {config_path}")

    # --- Write manifest ---
    manifest_path = output_dir / "dspark_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Manifest: {manifest_path}")
    print(f"Config: {config_path}")


if __name__ == "__main__":
    main()
