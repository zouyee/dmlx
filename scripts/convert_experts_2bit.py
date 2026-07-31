#!/usr/bin/env python3
"""Convert DeepSeek-V4-Flash routed expert weights to custom 2-bit packed format.

Source format (verified against inference/convert.py + kernel.py shipped with the
model): routed experts are stored as FP4 E2M1 (NOT FP8 E4M3 as originally assumed):
  - layers.{L}.ffn.experts.{E}.w1.weight : int8 [2048, 2048] = 2x E2M1 per byte,
        low nibble = element 2i, high nibble = element 2i+1  -> logical [2048, 4096]
  - layers.{L}.ffn.experts.{E}.w1.scale  : float8_e8m0 [2048, 128]
        (1 scale per 32 elements along K, scale = 2^(byte-127))
  - w3 same as w1. w2: int8 [4096, 1024] -> [4096, 2048], scale [4096, 64].
  E2M1 LUT: [0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6]

Target: RTN affine int2, group_size=64 per row:
  bias = bf16(min(group)), scale stored as E8M0 byte (2^(b-127)),
  q = round((w - bias_deq) / scale_deq) clamped to [0,3], packed 16 values/u32
  (value i at bits 2i..2i+1).

Output per expert (expert_size = 7,471,104 bytes), component order:
  gate_w (2097152) | gate_s (131072) | gate_b (262144) |
  up_w   (2097152) | up_s   (131072) | up_b   (262144) |
  down_w (2097152) | down_s (131072) | down_b (262144)
Per layer: 256 experts concatenated -> packed_experts_2bit/layer_{L:02d}.bin

Usage:
    python3 scripts/convert_experts_2bit.py [--layers 0-42] [--workers 8] [--self-check-only]

NOTE vs original spec: spec assumed FP8 E4M3 source and gate_w=1048576 bytes; both
were arithmetically impossible (int2 [2048,4096] = 2097152 bytes). Corrected sizes
are used here; manifest records actual values.
"""
import argparse
import json
import os
import struct
import sys
import time
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SRC_DIR = Path("/Users/zouyee/models/deepseek-ai/DeepSeek-V4-Flash")
OUT_DIR = Path("/Users/zouyee/models/DeepSeek-V4-Flash-4bit/packed_experts_2bit")

N_LAYERS = 43
N_EXPERTS = 256
GS = 64  # int2 group size
FP4_GS = 32  # source e2m1 scale group size

# (name, out_dim, in_dim)
MATRICES = [("w1", 2048, 4096), ("w3", 2048, 4096), ("w2", 4096, 2048)]
# blob component names in layout order
LAYOUT = ["gate_w", "gate_s", "gate_b",
          "up_w", "up_s", "up_b",
          "down_w", "down_s", "down_b"]

def comp_sizes():
    d = {}
    for mat, prefix in (("w1", "gate"), ("w3", "up"), ("w2", "down")):
        out_d, in_d = dict((m, (o, i)) for m, o, i in MATRICES)[mat]
        ng = in_d // GS
        d[prefix + "_w"] = out_d * in_d // 4      # int2: 4 values/byte
        d[prefix + "_s"] = out_d * ng             # 1 E8M0 byte/group
        d[prefix + "_b"] = out_d * ng * 2         # bf16/group
    return d

COMP_SIZES = comp_sizes()
EXPERT_SIZE = sum(COMP_SIZES[k] for k in LAYOUT)
assert EXPERT_SIZE == 3 * (2048 * 4096 // 4) + 3 * 131072 + 3 * 262144 == 7_471_104

FP4_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                   dtype=np.float32)
SHIFTS = (2 * np.arange(16, dtype=np.uint32))

# ---------------------------------------------------------------- index / I/O

_INDEX = None  # name -> (shard_path, abs_offset, nbytes)

def build_index():
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with open(SRC_DIR / "model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]
    hdr_cache = {}
    idx = {}
    for name, shard in wm.items():
        if ".experts." not in name:
            continue
        if shard not in hdr_cache:
            with open(SRC_DIR / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr_cache[shard] = (8 + n, json.loads(f.read(n)))
        base, hdr = hdr_cache[shard]
        o0, o1 = hdr[name]["data_offsets"]
        idx[name] = (str(SRC_DIR / shard), base + o0, o1 - o0)
    _INDEX = idx
    return idx

_FD = {}

def pread(path, off, nbytes):
    fd = _FD.get(path)
    if fd is None:
        fd = os.open(path, os.O_RDONLY)
        _FD[path] = fd
    return os.pread(fd, nbytes, off)

def read_raw(name):
    path, off, nbytes = build_index()[name]
    return pread(path, off, nbytes)

# ---------------------------------------------------------------- dequant

def dequant_fp4(name, out_d, in_d):
    """Return f32 [out_d, in_d] dequantized from packed E2M1 + E8M0 scales."""
    w = np.frombuffer(read_raw(name + ".weight"), dtype=np.uint8).reshape(out_d, in_d // 2)
    s = np.frombuffer(read_raw(name + ".scale"), dtype=np.uint8).reshape(out_d, in_d // FP4_GS)
    vals = np.empty((out_d, in_d), dtype=np.float32)
    vals[:, 0::2] = FP4_LUT[w & 0xF]
    vals[:, 1::2] = FP4_LUT[w >> 4]
    sf = np.exp2((s.astype(np.int32) - 127).astype(np.float32))  # 2^(b-127)
    vals = vals.reshape(out_d, in_d // FP4_GS, FP4_GS) * sf[:, :, None]
    return vals.reshape(out_d, in_d)

# ---------------------------------------------------------------- int2 quant

def f32_to_bf16_bytes(x):
    """float32 array -> bf16 bytes (round-to-nearest-even), little-endian."""
    u = x.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype("<u2").tobytes()

def bf16_roundtrip(x):
    u = x.astype(np.float32).view(np.uint32)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return u.view(np.float32)

def quant_int2(w):
    """w: f32 [out,in] -> (packed uint8 [out*in//4], scale bytes, bias bf16 bytes)."""
    out_d, in_d = w.shape
    ng = in_d // GS
    g = w.reshape(out_d, ng, GS)
    bias = g.min(axis=2)
    scale_f = (g.max(axis=2) - bias) / 3.0
    # E8M0 scale byte, self-consistent dequant scale
    with np.errstate(divide="ignore"):
        sb = np.rint(np.log2(scale_f)).astype(np.int32) + 127
    sb = np.clip(sb, 0, 255).astype(np.uint8)
    ds = np.exp2((sb.astype(np.int32) - 127).astype(np.float32))
    bias_deq = bf16_roundtrip(bias)  # self-consistent with stored bf16 bias
    q = np.rint((g - bias_deq[:, :, None]) / ds[:, :, None])
    q = np.clip(q, 0, 3).astype(np.uint8).reshape(out_d * ng * GS // 16, 16)
    packed = (q.astype(np.uint32) << SHIFTS).sum(axis=1).astype("<u4")
    return packed.tobytes(), sb.tobytes(), f32_to_bf16_bytes(bias)

def dequant_int2(packed_b, s_b, b_b, out_d, in_d):
    """Rebuild f32 [out,in] from blob components (for self-check)."""
    ng = in_d // GS
    p = np.frombuffer(packed_b, dtype="<u4").reshape(-1, 1)
    q = ((p >> SHIFTS) & 3).astype(np.float32).reshape(out_d, ng, GS)
    s = np.frombuffer(s_b, dtype=np.uint8).astype(np.int32)
    ds = np.exp2((s - 127).astype(np.float32)).reshape(out_d, ng)
    b16 = np.frombuffer(b_b, dtype="<u2").astype(np.uint32) << 16
    bias = b16.view(np.float32).reshape(out_d, ng)
    return (q * ds[:, :, None] + bias[:, :, None]).reshape(out_d, in_d)

# ---------------------------------------------------------------- layer convert

def convert_layer(layer):
    t0 = time.time()
    idx = build_index()
    tmp = OUT_DIR / f".layer_{layer:02d}.bin.tmp"
    final = OUT_DIR / f"layer_{layer:02d}.bin"
    if final.exists():
        return layer, -1.0, "skip (exists)"
    with open(tmp, "wb") as out:
        for e in range(N_EXPERTS):
            blob = []
            for mat, out_d, in_d in MATRICES:
                name = f"layers.{layer}.ffn.experts.{e}.{mat}"
                if name + ".weight" not in idx:
                    raise KeyError(name)
                w = dequant_fp4(name, out_d, in_d)
                blob.extend(quant_int2(w))
            out.write(b"".join(blob))
    os.rename(tmp, final)
    dt = time.time() - t0
    return layer, dt, f"{dt:.1f}s"

# ---------------------------------------------------------------- self-check

def self_check(n_samples=3, seed=0, layers=None):
    if layers is None:
        layers = [l for l in range(N_LAYERS) if (OUT_DIR / f"layer_{l:02d}.bin").exists()]
    if not layers:
        print("No converted layers found for self-check")
        return False
    rng = np.random.default_rng(seed)
    print("\n=== Self-check: reconstruct from blob vs source dequant ===")
    ok = True
    for i in range(n_samples):
        layer = layers[int(rng.integers(0, len(layers)))]
        expert = int(rng.integers(0, N_EXPERTS))
        mi = int(rng.integers(0, 3))
        mat, out_d, in_d = MATRICES[mi]
        prefix = ["gate", "up", "down"][mi]
        src = dequant_fp4(f"layers.{layer}.ffn.experts.{expert}.{mat}", out_d, in_d)
        # locate components in blob
        offs = {}
        o = 0
        for k in LAYOUT:
            offs[k] = o
            o += COMP_SIZES[k]
        base = expert * EXPERT_SIZE
        path = str(OUT_DIR / f"layer_{layer:02d}.bin")
        pw = pread(path, base + offs[prefix + "_w"], COMP_SIZES[prefix + "_w"])
        ps = pread(path, base + offs[prefix + "_s"], COMP_SIZES[prefix + "_s"])
        pb = pread(path, base + offs[prefix + "_b"], COMP_SIZES[prefix + "_b"])
        recon = dequant_int2(pw, ps, pb, out_d, in_d)
        sig = float((src.astype(np.float64) ** 2).sum())
        noise = float(((src - recon).astype(np.float64) ** 2).sum())
        snr = 10.0 * np.log10(sig / max(noise, 1e-30))
        status = "PASS" if snr > 15.0 else "FAIL"
        ok &= snr > 15.0
        print(f"  [{status}] layer={layer:2d} expert={expert:3d} {mat}: SNR = {snr:.2f} dB")
    return ok

# ---------------------------------------------------------------- main

def parse_layers(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=f"0-{N_LAYERS - 1}")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--self-check-only", action="store_true")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)

    if not args.self_check_only:
        need = EXPERT_SIZE * N_EXPERTS * len(layers)
        free = shutil.disk_usage(OUT_DIR).free
        existing = sum((OUT_DIR / f"layer_{l:02d}.bin").stat().st_size
                       for l in layers if (OUT_DIR / f"layer_{l:02d}.bin").exists())
        if free < need - existing:
            print(f"ERROR: need ~{(need - existing) / 2**30:.1f} GiB, only {free / 2**30:.1f} GiB free")
            sys.exit(1)
        print(f"Converting layers {layers[0]}..{layers[-1]} ({len(layers)} layers), "
              f"expert_size={EXPERT_SIZE}, ~{need / 2**30:.1f} GiB total, workers={args.workers}")
        t0 = time.time()
        with Pool(args.workers) as pool:
            for layer, dt, msg in pool.imap_unordered(convert_layer, layers):
                el = time.time() - t0
                print(f"layer {layer:2d} done: {msg}  (elapsed {el / 60:.1f} min)", flush=True)
        print(f"Conversion finished in {(time.time() - t0) / 60:.1f} min")

    manifest = {
        "version": 2,
        "expert_format": "int2_gs64",
        "source": "deepseek-ai/DeepSeek-V4-Flash FP4 (E2M1+E8M0 packed as int8)",
        "expert_size": EXPERT_SIZE,
        "n_experts": N_EXPERTS,
        "layout": LAYOUT,
        "component_sizes": COMP_SIZES,
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest.json written (expert_size={EXPERT_SIZE})")

    if not self_check(args.samples):
        print("SELF-CHECK FAILED")
        sys.exit(2)
    print("Self-check passed (all SNR > 15 dB)")

if __name__ == "__main__":
    main()
