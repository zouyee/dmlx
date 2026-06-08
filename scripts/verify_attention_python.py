#!/usr/bin/env python3
"""
Verify mla_attention_decode_bf16 correctness in Python.
Given MLX golden L0_attn_normed (all 9 tokens) and layer 0 weights,
compute attention for each token using the same algorithm as mla_attention_decode_bf16.
Compare with MLX golden L0_attn_out.

This script is entirely standalone—no native server needed.
"""
import numpy as np
import json, struct, os, sys

MODEL = "/Users/zouyee/models/DeepSeek-V4-Flash-4bit"
REF   = "/tmp/mlx_ref_new"

# ── constants ────────────────────────────────────────────────────────────────
DIM         = 4096
N_HEADS     = 64
HEAD_DIM    = 512
Q_LORA_RANK = 1024
KV_LORA_RANK= 512
QK_ROPE_DIM = 64
QK_NOPE_DIM = 448
O_GROUPS    = 8
O_LORA_RANK = 1024
ATTN_GS     = 64
EPS         = 1e-6
PI          = 3.141592653589793

# ── helpers ──────────────────────────────────────────────────────────────────
def bf16(x):
    """Truncate f32 to bf16 (take upper 16 bits)."""
    u = x.view(np.uint32) & np.uint32(0xFFFF0000)
    return u.view(np.float32)

def rms_norm(x, w):
    x = x.astype(np.float32)
    inv = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + EPS)
    return (x * inv * w).astype(np.float32)

def yarn_cos_sin(pos):
    base = 10000.; factor = 16.; beta_fast = 32.; beta_slow = 1.
    orig_max = 65536.
    half = QK_ROPE_DIM // 2
    cd_fast = QK_ROPE_DIM * np.log(orig_max / (beta_fast * 2 * PI)) / (2 * np.log(base))
    cd_slow = QK_ROPE_DIM * np.log(orig_max / (beta_slow * 2 * PI)) / (2 * np.log(base))
    low  = int(max(0, np.floor(cd_fast)))
    high = int(min(QK_ROPE_DIM - 1, np.ceil(cd_slow)))
    freqs = np.zeros(half, dtype=np.float32)
    for i in range(half):
        freq = 1. / (base ** (2. * i / QK_ROPE_DIM))
        if low == high:
            ramp = 0. if i <= low else 1.
        else:
            ramp = float(np.clip((i - low) / (high - low), 0, 1))
        smooth = 1. - ramp
        freqs[i] = freq / factor * (1 - smooth) + freq * smooth
    c = np.cos(pos * freqs).astype(np.float32)
    s = np.sin(pos * freqs).astype(np.float32)
    return c, s

def rope_apply(q, cos, sin):
    """Apply YaRN tail RoPE to q [n_heads, HEAD_DIM]."""
    half = QK_ROPE_DIM // 2
    q = q.copy()
    for h in range(N_HEADS):
        for i in range(half):
            j0 = QK_NOPE_DIM + i
            j1 = QK_NOPE_DIM + half + i
            x0 = float(q[h, j0]); x1 = float(q[h, j1])
            c  = float(cos[i]);    s  = float(sin[i])
            q[h, j0] = np.float32(x0 * c - x1 * s)
            q[h, j1] = np.float32(x0 * s + x1 * c)
    return q

def rope_apply_inverse(q, cos, sin):
    """Inverse RoPE (used on output)."""
    half = QK_ROPE_DIM // 2
    q = q.copy()
    for h in range(N_HEADS):
        for i in range(half):
            j0 = QK_NOPE_DIM + i
            j1 = QK_NOPE_DIM + half + i
            x0 = float(q[h, j0]); x1 = float(q[h, j1])
            c  = float(cos[i]);    s  = float(sin[i])
            # Inverse: cos,-sin
            q[h, j0] = np.float32( x0 * c + x1 * s)
            q[h, j1] = np.float32(-x0 * s + x1 * c)
    return q

def rope_apply_1head(kv, cos, sin):
    """Apply RoPE to single KV head [HEAD_DIM]."""
    half = QK_ROPE_DIM // 2
    kv = kv.copy()
    for i in range(half):
        j0 = QK_NOPE_DIM + i
        j1 = QK_NOPE_DIM + half + i
        x0 = float(kv[j0]); x1 = float(kv[j1])
        c  = float(cos[i]); s  = float(sin[i])
        kv[j0] = np.float32(x0 * c - x1 * s)
        kv[j1] = np.float32(x0 * s + x1 * c)
    return kv

# ── load safetensors ──────────────────────────────────────────────────────────
def sf_load_bf16_as_f32(path, key, meta):
    b0, b1 = meta['data_offsets']
    with open(path, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        f.seek(8 + hdr_len + b0)
        raw = f.read(b1 - b0)
    dtype = meta['dtype']
    shape = meta['shape']
    if dtype == 'BF16':
        arr = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        arr = (arr << 16).view(np.float32).copy()
    elif dtype == 'F32':
        arr = np.frombuffer(raw, dtype=np.float32).copy()
    elif dtype in ('U32', 'I32'):
        arr = np.frombuffer(raw, dtype=np.uint32).copy()
    else:
        raise ValueError(f'Unsupported dtype {dtype}')
    return arr.reshape(shape)

def load_sf_meta(path):
    with open(path, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len
    return hdr, data_start, path

# ── affine matvec ─────────────────────────────────────────────────────────────
def affine_matvec(packed, scales, biases, x, out_dim, in_dim, gs=ATTN_GS):
    """y = W_affine @ x  (4-bit quantized)"""
    n_groups  = in_dim // gs
    ppg       = gs // 8          # packed uint32s per group
    packed_cols = in_dim // 8
    result = np.zeros(out_dim, dtype=np.float64)
    for r in range(out_dim):
        wp = packed[r]          # shape [in_dim//8]
        sc = scales[r]          # shape [n_groups]
        bi = biases[r]          # shape [n_groups]
        for g in range(n_groups):
            sv = float(sc[g]); bv = float(bi[g])
            xi = x[g*gs : (g+1)*gs].astype(np.float64)
            for p in range(ppg):
                pw = int(wp[g*ppg + p])
                for k in range(8):
                    nib = (pw >> (k*4)) & 0xF
                    result[r] += (sv * nib + bv) * xi[p*8 + k]
    return result.astype(np.float32)

def affine_matvec_fast(packed, scales, biases, x, out_dim, in_dim, gs=ATTN_GS):
    """Vectorised version — still exact same arithmetic."""
    n_groups = in_dim // gs
    ppg      = gs // 8
    # Unpack 4-bit nibbles for all rows at once
    # packed: [out_dim, in_dim//8] uint32
    # We expand to [out_dim, in_dim] nibble values
    packed_flat = packed.astype(np.uint64)  # [out_dim, in_dim//8]
    nibbles = np.zeros((out_dim, in_dim), dtype=np.float32)
    for k in range(8):
        nibbles[:, k::8] = ((packed_flat >> (k*4)) & 0xF).astype(np.float32)
    # scales & biases: [out_dim, n_groups]
    # Broadcast to [out_dim, in_dim]
    sc_exp = np.repeat(scales.astype(np.float32), gs, axis=1)   # [out_dim, in_dim]
    bi_exp = np.repeat(biases.astype(np.float32), gs, axis=1)
    W = sc_exp * nibbles + bi_exp           # [out_dim, in_dim] f32
    return (W @ x.astype(np.float32))      # [out_dim]

# ── dense matvec ──────────────────────────────────────────────────────────────
def dense_matvec(W, x):
    """y = W @ x, W is [out, in] f32, x is [in] f32."""
    return (W @ x.astype(np.float32))

# ── SDPA decode (1 query, n_kv keys) ─────────────────────────────────────────
def sdpa_decode(q_heads, kv_cache, sink, scale):
    """
    q_heads : [N_HEADS, HEAD_DIM] f32  — query (RoPE applied)
    kv_cache: [n_kv, HEAD_DIM]   f32  — KV cache
    sink    : [N_HEADS]          f32
    scale   : float
    Returns : [N_HEADS, HEAD_DIM] f32  — attended value
    """
    n_kv = kv_cache.shape[0]
    out  = np.zeros((N_HEADS, HEAD_DIM), dtype=np.float32)

    for h in range(N_HEADS):
        qh = q_heads[h].astype(np.float64)
        m  = -np.inf; s = 0.0
        acc = np.zeros(HEAD_DIM, dtype=np.float64)

        for k in range(n_kv):
            kv_k = kv_cache[k].astype(np.float64)
            dot = np.dot(qh, kv_k) * scale
            m_new = max(m, dot)
            corr  = 0.0 if m == -np.inf else np.exp(m - m_new)
            p     = np.exp(dot - m_new)
            acc   = acc * corr + p * kv_k
            s     = s * corr + p
            m     = m_new

        # Sink
        sk = float(sink[h])
        m_new = max(m, sk)
        corr  = 0.0 if m == -np.inf else np.exp(m - m_new)
        vs    = np.exp(sk - m_new)
        acc  *= corr
        s     = s * corr + vs
        m     = m_new

        out[h] = (acc / s).astype(np.float32)

    return out

# ── main attention function ───────────────────────────────────────────────────
def mla_attention_decode_bf16_python(normed_x, weights, kv_cache_so_far, pos):
    """
    normed_x: [DIM] f32  — attention-normed input (bf16 truncated)
    weights:  dict of loaded weight tensors
    kv_cache_so_far: list of [HEAD_DIM] f32 — KV cache up to pos-1
    pos: current position
    Returns: ([DIM] f32 attn_out, updated kv_cache list)
    """
    cos, sin = yarn_cos_sin(pos)
    x = bf16(normed_x)  # ensure bf16 precision on input

    # Q chain: wq_a → q_norm → wq_b → per-head norm → RoPE
    q_a = affine_matvec_fast(weights['wq_a.w'], weights['wq_a.s'], weights['wq_a.b'],
                              x, Q_LORA_RANK, DIM)
    q_a = bf16(q_a)
    q_res = bf16(rms_norm(q_a, weights['q_norm']))
    q = affine_matvec_fast(weights['wq_b.w'], weights['wq_b.s'], weights['wq_b.b'],
                            q_res, N_HEADS * HEAD_DIM, Q_LORA_RANK)
    q = bf16(q.reshape(N_HEADS, HEAD_DIM))
    # per-head norm (no weight, just divide by rms)
    q_n = bf16(rms_norm(q.reshape(-1), np.ones(N_HEADS * HEAD_DIM)).reshape(N_HEADS, HEAD_DIM))
    # Hmm—per-head norm should be per-head, not global
    q_n2 = np.zeros_like(q)
    for h in range(N_HEADS):
        rms = np.sqrt(np.mean(q[h]**2) + EPS)
        q_n2[h] = bf16(q[h] / rms)
    q_rope = bf16(rope_apply(q_n2, cos, sin))

    # KV chain: wkv → kv_norm → RoPE
    kv = affine_matvec_fast(weights['wkv.w'], weights['wkv.s'], weights['wkv.b'],
                             x, KV_LORA_RANK, DIM)
    kv = bf16(kv)
    kv_n = bf16(rms_norm(kv, weights['kv_norm']))
    kv_rope = bf16(rope_apply_1head(kv_n, cos, sin))

    # Update KV cache
    kv_cache = kv_cache_so_far + [kv_rope]
    kv_mat = np.stack(kv_cache, axis=0)  # [n_kv, HEAD_DIM]

    # SDPA
    scale = 1.0 / np.sqrt(HEAD_DIM)
    attn = sdpa_decode(q_rope, kv_mat, weights['sink'], scale)  # [N_HEADS, HEAD_DIM]

    # Inverse RoPE
    attn = bf16(rope_apply_inverse(attn, cos, sin))

    # wo_a (grouped, dense) → concat → wo_b
    heads_per_group = N_HEADS // O_GROUPS
    group_feat = heads_per_group * HEAD_DIM
    concat = np.zeros(O_GROUPS * O_LORA_RANK, dtype=np.float32)
    wo_a = weights['wo_a']  # [O_GROUPS, O_LORA_RANK, group_feat] f32
    for g in range(O_GROUPS):
        gv = attn[g*heads_per_group : (g+1)*heads_per_group, :].ravel()  # [group_feat] bf16
        concat[g*O_LORA_RANK : (g+1)*O_LORA_RANK] = bf16(wo_a[g] @ gv.astype(np.float32))

    out = affine_matvec_fast(weights['wo_b.w'], weights['wo_b.s'], weights['wo_b.b'],
                              concat, DIM, O_GROUPS * O_LORA_RANK)
    return out, kv_cache

# ── load weights ──────────────────────────────────────────────────────────────
print("Loading layer 0 attention weights…", flush=True)

idx_path = f"{MODEL}/model.safetensors.index.json"
with open(idx_path) as f:
    idx = json.load(f)
wmap = idx["weight_map"]

# All keys live in the first shard
shard = wmap.get("model.layers.0.attn.wq_a.weight", "model-00001-of-00033.safetensors")
shard_path = f"{MODEL}/{shard}"
hdr, data_off, _ = load_sf_meta(shard_path)

def L(key):
    m = hdr.get(f"model.layers.0.attn.{key}")
    if m is None:
        raise KeyError(f"model.layers.0.attn.{key} not in shard")
    return sf_load_bf16_as_f32(shard_path, key, m)

# Also check if wo_a is in a different shard
wo_a_shard_key = "model.layers.0.attn.wo_a.weight"
wo_a_shard = wmap.get(wo_a_shard_key, shard)
wo_a_path  = f"{MODEL}/{wo_a_shard}"
wo_a_hdr, wo_a_off, _ = load_sf_meta(wo_a_path)

def La(key):
    m = wo_a_hdr.get(f"model.layers.0.attn.{key}")
    if m is None:
        raise KeyError(f"model.layers.0.attn.{key} not in wo_a shard")
    return sf_load_bf16_as_f32(wo_a_path, key, m)

weights = {}
print("  wq_a…", flush=True)
weights['wq_a.w'] = L('wq_a.weight')    # [1024, 512]  uint32 4bit
weights['wq_a.s'] = L('wq_a.scales')    # [1024, 64]   bf16→f32
weights['wq_a.b'] = L('wq_a.biases')    # [1024, 64]   bf16→f32
print("  q_norm…", flush=True)
weights['q_norm']  = L('q_norm.weight') # [1024]        bf16→f32
print("  wq_b…", flush=True)
weights['wq_b.w'] = L('wq_b.weight')    # [32768, 128]  uint32 4bit
weights['wq_b.s'] = L('wq_b.scales')    # [32768, 16]
weights['wq_b.b'] = L('wq_b.biases')    # [32768, 16]
print("  wkv…", flush=True)
# wkv might have different name (wkv_a_proj_with_mqa or wkv)
try:
    weights['wkv.w'] = L('wkv.weight')
    weights['wkv.s'] = L('wkv.scales')
    weights['wkv.b'] = L('wkv.biases')
except KeyError:
    weights['wkv.w'] = L('wkv_a_proj_with_mqa.weight')
    weights['wkv.s'] = L('wkv_a_proj_with_mqa.scales')
    weights['wkv.b'] = L('wkv_a_proj_with_mqa.biases')
print("  kv_norm…", flush=True)
weights['kv_norm'] = L('kv_norm.weight')  # [512]
print("  sink…", flush=True)
weights['sink']    = L('attn_sink')       # [64] f32
print("  wo_a…", flush=True)
# wo_a is dense f32: [O_GROUPS * O_LORA_RANK, group_feat] where group_feat = heads_per_group * HEAD_DIM = 4096
# stored as [8192, 512] uint32 packed 4bit but our loader dequantizes it
# Actually wo_a is stored differently—let me check shape
wo_a_packed = La('wo_a.weight')   # shape [8192, 512] uint32
wo_a_scales = La('wo_a.scales')   # [8192, 64]
wo_a_biases = La('wo_a.biases')   # [8192, 64]
print(f"  wo_a shape: {wo_a_packed.shape} (packed), scales: {wo_a_scales.shape}")
# Dequantize wo_a fully
# out_dim = 8192 = O_GROUPS * O_LORA_RANK
# in_dim  = 4096 = group_feat (but stored in 512 uint32s = 512*8=4096 nibbles)
wo_a_indim = wo_a_packed.shape[1] * 8  # 4096
wo_a_outdim = wo_a_packed.shape[0]     # 8192
print(f"  wo_a dequantizing [{wo_a_outdim}, {wo_a_indim}]…", flush=True)
# This is per-group dequantization; vectorise over nibbles
nibbles = np.zeros((wo_a_outdim, wo_a_indim), dtype=np.float32)
for k in range(8):
    nibbles[:, k::8] = ((wo_a_packed.astype(np.uint32) >> (k*4)) & 0xF).astype(np.float32)
sc_exp = np.repeat(wo_a_scales.astype(np.float32), 64, axis=1)  # [8192, 4096]
bi_exp = np.repeat(wo_a_biases.astype(np.float32), 64, axis=1)
wo_a_dense = sc_exp * nibbles + bi_exp  # [8192, 4096] f32
# Reshape: [O_GROUPS, O_LORA_RANK, group_feat]
weights['wo_a'] = wo_a_dense.reshape(O_GROUPS, O_LORA_RANK, -1)
print("  wo_b…", flush=True)
weights['wo_b.w'] = La('wo_b.weight')   # [4096, 1024] uint32
weights['wo_b.s'] = La('wo_b.scales')   # [4096, 128]
weights['wo_b.b'] = La('wo_b.biases')   # [4096, 128]

print("All weights loaded.", flush=True)

# ── load MLX golden ───────────────────────────────────────────────────────────
attn_normed = np.load(f"{REF}/L0_attn_normed.npy")  # [1, 9, 4096]
attn_out_ref = np.load(f"{REF}/L0_attn_out.npy")    # [1, 9, 4096]
n_tokens = attn_normed.shape[1]
print(f"\nGolden: {n_tokens} tokens", flush=True)

# ── run Python attention for each token ──────────────────────────────────────
kv_cache = []
print(f"\n{'tok':>3} {'rel_L2':>8} {'cosine':>8} {'nat_norm':>9} {'mlx_norm':>9}")
print("-" * 50)
for t in range(n_tokens):
    x_in = attn_normed[0, t, :].astype(np.float32)
    ref_out = attn_out_ref[0, t, :].astype(np.float32)

    native_out, kv_cache = mla_attention_decode_bf16_python(x_in, weights, kv_cache, pos=t)

    diff = native_out - ref_out
    rel_l2 = np.linalg.norm(diff) / (np.linalg.norm(ref_out) + 1e-12)
    cos = np.dot(native_out, ref_out) / (np.linalg.norm(native_out) * np.linalg.norm(ref_out) + 1e-12)
    print(f"{t:>3} {rel_l2:>8.4f} {cos:>8.4f} {np.linalg.norm(native_out):>9.2f} {np.linalg.norm(ref_out):>9.2f}")

print("\nDone.")
