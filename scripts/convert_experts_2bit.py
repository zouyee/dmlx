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
    ap.add_argument("--mixed-v1-fixture", type=Path,
                    help="write deterministic tensor-level mixed2 v1 fixture pair")
    ap.add_argument("--mixed-v1-self-check", action="store_true",
                    help="build the mixed2 v1 fixture twice and compare bytes")
    args = ap.parse_args()

    if args.mixed_v1_fixture is not None or args.mixed_v1_self_check:
        first_b, first_j = build_mixed2_small_fixture()
        second_b, second_j = build_mixed2_small_fixture()
        if first_b != second_b or mixed2_canonical_json(first_j) != mixed2_canonical_json(second_j):
            raise SystemExit("mixed2 determinism: generation mismatch")
        if args.mixed_v1_fixture is not None:
            target = args.mixed_v1_fixture
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(first_b)
            target.with_suffix(".json").write_bytes(mixed2_canonical_json(first_j) + b"\n")
            print(f"mixed2 fixture written: {target}")
        print("mixed2 fixture self-check passed")
        return

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



# ---------------------------------------------------------------- mixed2 v1 fixture/reference primitives
# This intentionally implements tensor-level fixture generation only.  It does
# not read model files and does not claim to produce a runtime-qualified model.
import hashlib
import math

MIXED2_MAGIC = b"DMLXMIX2V1" + b"\0" * 6
MIXED2_HEADER_SIZE = 512
MIXED2_ENTRY_SIZE = 84
MIXED2_ALIGNMENT = 64
MIXED2_VERSION = (1, 0)
MIXED2_ENDIAN_TAG = 0x01020304
MIXED2_ENCODING_Q4 = 1
MIXED2_ENCODING_INT2 = 2
MIXED2_SCALE_NONE = 0
MIXED2_SCALE_E8M0 = 1
MIXED2_BIAS_NONE = 0
MIXED2_BIAS_BF16 = 1
MIXED2_Q4_RECORD_SIZE = 34
MIXED2_INT2_RECORD_SIZE = 19
MIXED2_CORE_DOMAIN = b"DMLX-MIX2-CORE-V1\0"
MIXED2_MANIFEST_DOMAIN = b"DMLX-MIX2-MANIFEST-V1\0"
MIXED2_LAYER_DOMAIN = b"DMLX-MIX2-LAYER-V1\0"
MIXED2_ZERO_HASH = "0" * 64
assert len(MIXED2_MAGIC) == 16
assert MIXED2_HEADER_SIZE == 512 and MIXED2_ENTRY_SIZE == 84
assert struct.calcsize("<HBBBBBBIIIIIIHHQQ32s") == MIXED2_ENTRY_SIZE


def mixed2_sha256(data):
    return hashlib.sha256(data).digest()


def mixed2_canonical_json(value):
    """RFC canonical JSON bytes (fixture objects contain no floats)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def mixed2_hash_hex(domain, value):
    return hashlib.sha256(domain + mixed2_canonical_json(value)).hexdigest()


def mixed2_bf16_encode(value):
    if not math.isfinite(value):
        raise ValueError("finite: non-finite BF16 input")
    try:
        packed = struct.pack("<f", float(value))
    except OverflowError as exc:
        raise ValueError("bf16-range: value is outside finite float32 range") from exc
    bits = struct.unpack("<I", packed)[0]
    if bits & 0x7F800000 == 0x7F800000:
        raise ValueError("bf16-range: finite input rounded to non-finite float32")
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return struct.pack("<H", rounded & 0xFFFF)


def mixed2_bf16_decode(raw):
    if len(raw) != 2:
        raise ValueError("bounds: BF16 requires two bytes")
    return struct.unpack("<f", struct.pack("<I", struct.unpack("<H", raw)[0] << 16))[0]


def mixed2_decode_q4_record(record):
    if len(record) != MIXED2_Q4_RECORD_SIZE:
        raise ValueError("bounds: Q4 record must be 34 bytes")
    lut = (0.0, .5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
           -0.0, -.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
    out = []
    for base in (0, 17):
        scale = math.ldexp(1.0, record[base + 16] - 127)
        for packed in record[base:base + 16]:
            out.append(lut[packed & 15] * scale)
            out.append(lut[packed >> 4] * scale)
    if not all(math.isfinite(v) for v in out):
        raise ValueError("finite: non-finite Q4 decode")
    return out


def mixed2_q4_record(block0, scale0, block1, scale1):
    if len(block0) != 16 or len(block1) != 16:
        raise ValueError("bounds: Q4 sub-block must be 16 bytes")
    if not (0 <= scale0 <= 255 and 0 <= scale1 <= 255):
        raise ValueError("enum: E8M0 scale byte")
    record = bytes(block0) + bytes((scale0,)) + bytes(block1) + bytes((scale1,))
    assert len(record) == MIXED2_Q4_RECORD_SIZE
    return record


def mixed2_encode_int2(values):
    """Encode one affine uint2/E8M0/BF16 group with deterministic RFC RNE."""
    if len(values) != 64:
        raise ValueError("shape: int2 group must contain 64 values")
    vals = [float(v) for v in values]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("finite: non-finite int2 source")
    lo, hi = min(vals), max(vals)
    if hi == lo:
        scale_byte = 127
        q = [0] * 64
    else:
        value_range = hi - lo
        if not math.isfinite(value_range) or value_range <= 0.0:
            raise ValueError("scale-range: non-finite or nonpositive finite-source range")
        raw_scale = value_range / 3.0
        if not math.isfinite(raw_scale) or raw_scale <= 0.0:
            raise ValueError("scale-range: invalid scale intermediate")
        exponent = round(math.log2(raw_scale))
        if exponent < -127 or exponent > 128:
            raise ValueError("scale-range: E8M0 exponent out of range")
        scale_byte = exponent + 127
        bias_r = mixed2_bf16_decode(mixed2_bf16_encode(lo))
        scale = math.ldexp(1.0, exponent)
        q = [max(0, min(3, round((v - bias_r) / scale))) for v in vals]
    bias = mixed2_bf16_encode(lo)
    packed = bytearray(16)
    for i, value in enumerate(q):
        packed[i // 4] |= value << (2 * (i % 4))
    record = bytes(packed) + bytes((scale_byte,)) + bias
    assert len(record) == MIXED2_INT2_RECORD_SIZE
    return record


def mixed2_decode_int2_record(record):
    if len(record) != MIXED2_INT2_RECORD_SIZE:
        raise ValueError("bounds: int2 record must be 19 bytes")
    bias = mixed2_bf16_decode(record[17:19])
    scale = math.ldexp(1.0, record[16] - 127)
    out = [bias + ((record[i // 4] >> (2 * (i % 4))) & 3) * scale for i in range(64)]
    if not all(math.isfinite(v) for v in out):
        raise ValueError("finite: non-finite int2 decode")
    return out


MIXED2_FIXTURE_BINARY_FIELDS = (
    "file_size", "table_offset", "table_length", "payload_offset", "payload_length",
    "header_sha256", "table_sha256", "payload_sha256", "layer_binding_sha256", "file_sha256",
)


def mixed2_fixture_core_projection(sidecar):
    """Return the reduced-fixture core projection, not the RFC production projection.

    The production projection covers a complete 43-layer manifest and its accounting and
    qualification objects.  This fixture-only projection instead zeros every generated
    field in the reduced sidecar's single ``binary`` object while retaining its immutable
    shape, allocation, source, and expected-value contract.
    """
    core = json.loads(mixed2_canonical_json(sidecar))
    core["binary"] = {
        key: (0 if key in ("file_size", "table_offset", "table_length", "payload_offset", "payload_length")
              else MIXED2_ZERO_HASH)
        for key in MIXED2_FIXTURE_BINARY_FIELDS
    }
    core["manifest_core_sha256"] = MIXED2_ZERO_HASH
    core["manifest_binding_sha256"] = MIXED2_ZERO_HASH
    return core


def mixed2_fixture_bindings(sidecar):
    """Return the fixture-sidecar core projection/hash and final sidecar binding."""
    core = mixed2_fixture_core_projection(sidecar)
    core_hash = mixed2_hash_hex(MIXED2_CORE_DOMAIN, core)
    final = json.loads(mixed2_canonical_json(sidecar))
    final["manifest_core_sha256"] = core_hash
    final["manifest_binding_sha256"] = MIXED2_ZERO_HASH
    binding = mixed2_hash_hex(MIXED2_MANIFEST_DOMAIN, final)
    return core, core_hash, binding


def _mixed2_align(value, alignment=MIXED2_ALIGNMENT):
    return (value + alignment - 1) // alignment * alignment


def mixed2_fixture_number(value):
    """Use JSON integers for integral decoded values as required by the fixture contract."""
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def mixed2_fixture_decoded(records, decoder):
    return [[mixed2_fixture_number(value) for value in decoder(record)] for record in records]


def build_mixed2_small_fixture():
    """Build a deterministic reduced fixture; it is not a production 43-layer manifest."""
    q4a = bytes.fromhex("1032547698badcfe0123456789abcdef")
    q4b = bytes.fromhex("fedcba98765432100123456789abcdef")
    q4c = bytes.fromhex("00112233445566778899aabbccddeeff")
    q4d = bytes.fromhex("ffeeddccbbaa99887766554433221100")
    q4_records = [mixed2_q4_record(q4a, 127, q4b, 128),
                  mixed2_q4_record(q4c, 126, q4d, 129)]
    int_sources = [[float((i % 4) - 2) for i in range(64)], [3.25] * 64]
    int_records = [mixed2_encode_int2(v) for v in int_sources]

    # Immutable fixture-sidecar subset skeleton.  Its fixture-only core projection
    # zeros every binary generated field, so the core can be computed before any
    # header/table/payload bytes without introducing a hash cycle.
    manifest = {
        "schema": "dmlx.mixed2.fixture.v1",
        "contract_scope": "reduced-single-expert-gate-tensor-fixture-only",
        "runtime_qualified": False,
        "profile": "reduced-tensor-production-wire",
        "format_major": 1,
        "format_minor": 0,
        "tensor_shape": [1, 256],
        "allocation_entries": [
            {"encoding": "q4", "row": 0, "group_start": 0, "group_count": 2},
            {"encoding": "int2", "row": 0, "group_start": 2, "group_count": 2},
        ],
        "source_q4_records_hex": [record.hex() for record in q4_records],
        "expected_decoded": {
            "q4": mixed2_fixture_decoded(q4_records, mixed2_decode_q4_record),
            "int2": mixed2_fixture_decoded(int_records, mixed2_decode_int2_record),
        },
        "binary": {
            key: (0 if key in ("file_size", "table_offset", "table_length", "payload_offset", "payload_length")
                  else MIXED2_ZERO_HASH)
            for key in MIXED2_FIXTURE_BINARY_FIELDS
        },
        "manifest_core_sha256": MIXED2_ZERO_HASH,
        "manifest_binding_sha256": MIXED2_ZERO_HASH,
    }
    _, core_hash, _ = mixed2_fixture_bindings(manifest)
    manifest["manifest_core_sha256"] = core_hash

    payload_parts = [b"".join(q4_records), b"".join(int_records)]
    table_count = 2
    table_offset = MIXED2_HEADER_SIZE
    table_length = table_count * MIXED2_ENTRY_SIZE
    payload_offset = _mixed2_align(table_offset + table_length)
    first_off = payload_offset
    second_off = _mixed2_align(first_off + len(payload_parts[0]))
    file_length = second_off + len(payload_parts[1])
    payload = bytearray(file_length - payload_offset)
    payload[:len(payload_parts[0])] = payload_parts[0]
    payload[second_off - payload_offset:second_off - payload_offset + len(payload_parts[1])] = payload_parts[1]
    entries = [
        struct.pack("<HBBBBBBIIIIIIHHQQ32s", 0, 0, MIXED2_ENCODING_Q4, 4,
                    MIXED2_SCALE_E8M0, MIXED2_BIAS_NONE, 0, 1, 256, 0, 1, 0, 2,
                    64, 32, first_off, len(payload_parts[0]), mixed2_sha256(payload_parts[0])),
        struct.pack("<HBBBBBBIIIIIIHHQQ32s", 0, 0, MIXED2_ENCODING_INT2, 2,
                    MIXED2_SCALE_E8M0, MIXED2_BIAS_BF16, 0, 1, 256, 0, 1, 2, 2,
                    64, 0, second_off, len(payload_parts[1]), mixed2_sha256(payload_parts[1])),
    ]
    table = b"".join(entries)
    table_hash, payload_hash = mixed2_sha256(table), mixed2_sha256(payload)
    identity = [mixed2_sha256(("mixed2-fixture-identity-%d" % i).encode()) for i in range(8)]
    identity.append(bytes.fromhex(core_hash))
    header = bytearray(MIXED2_HEADER_SIZE)
    struct.pack_into("<16sHHII I HHHH I QQQQQ II I", header, 0,
                     MIXED2_MAGIC, 1, 0, MIXED2_ENDIAN_TAG, 512, 84, 0, 3, 256, 6,
                     table_count, table_offset, table_length, payload_offset,
                     len(payload), file_length, 64, 0, 0)
    header[96:384] = b"".join(identity)
    header[384:416], header[416:448] = table_hash, payload_hash
    layer_input = bytes(header) + table + bytes(payload)
    layer_hash = mixed2_sha256(MIXED2_LAYER_DOMAIN + layer_input)
    header[448:480] = layer_hash
    binary = bytes(header) + table + b"\0" * (payload_offset - table_offset - table_length) + bytes(payload)
    assert len(binary) == file_length

    manifest["binary"] = {
        "file_size": len(binary), "table_offset": table_offset, "table_length": table_length,
        "payload_offset": payload_offset, "payload_length": len(payload),
        "header_sha256": mixed2_sha256(bytes(header)).hex(), "table_sha256": table_hash.hex(),
        "payload_sha256": payload_hash.hex(), "layer_binding_sha256": layer_hash.hex(),
        "file_sha256": mixed2_sha256(binary).hex(),
    }
    _, recomputed_core, binding = mixed2_fixture_bindings(manifest)
    assert recomputed_core == core_hash
    manifest["manifest_binding_sha256"] = binding
    return binary, manifest


if __name__ == "__main__":
    main()
