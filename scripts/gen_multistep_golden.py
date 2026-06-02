#!/usr/bin/env python3
"""Generate multi-step MLA attention golden for KV-cache correctness test.

Simulates N_PREFILL token prefill (each token processed sequentially, building
the KV cache), then one decode step at pos=N_PREFILL.

Dumps to /tmp/attn_ms_golden/:
  hidden_*.bin        [4096] f32 input for each step (random)
  kv_cache.bin        [N_PREFILL+1, 512] f32  — full KV cache after all steps
  golden.bin          [4096] f32  — attn output at decode step
  meta.txt            N_PREFILL, decode_pos

The test harness (mla_attention_multistep_test.m) calls mla_attention_decode
once per step and checks the final output matches golden.bin.
"""
import sys, os, json, struct, glob
import numpy as np

MODEL = "/Users/zouyee/models/DeepSeek-V4-Flash-4bit"
OUT = "/tmp/attn_ms_golden"
N_PREFILL = int(sys.argv[1]) if len(sys.argv) > 1 else 8
GS = 64
EPS = 1e-6
H = 64
HEAD_DIM = 512
ROPE_DIM = 64
NOPE = HEAD_DIM - ROPE_DIM
N_GROUPS = 8
HPG = H // N_GROUPS
O_LORA = 1024
BASE = 10000.0
FACTOR = 16.0
BETA_FAST = 32.0
BETA_SLOW = 1.0
ORIG_MAX = 65536
SCALE = 1.0 / np.sqrt(HEAD_DIM)


def find(name):
    for f in sorted(glob.glob(MODEL + "/model-*-of-00033.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n)); base = 8 + n
            if name in hdr:
                v = hdr[name]; o0, o1 = v["data_offsets"]
                fh.seek(base + o0); raw = fh.read(o1 - o0)
                return v["dtype"], v["shape"], raw
    sys.exit("not found " + name)


def npf(dtype, shape, raw):
    if dtype == "U32": a = np.frombuffer(raw, "<u4")
    elif dtype == "BF16":
        u = np.frombuffer(raw, "<u2").astype(np.uint32); a = (u << 16).view(np.float32)
    elif dtype == "F32": a = np.frombuffer(raw, "<f4")
    else: sys.exit("dtype " + dtype)
    return a.reshape(shape)


def load(name): dt, sh, raw = find(name); return npf(dt, sh, raw)


def dense(name):
    packed = load(name + ".weight"); scales = load(name + ".scales"); biases = load(name + ".biases")
    out_dim, pcols = packed.shape; in_dim = pcols * 8
    nib = np.empty((out_dim, in_dim), np.float32)
    for i in range(8): nib[:, i::8] = (packed >> (i * 4)) & 0xF
    sc = np.repeat(scales, GS, 1); bi = np.repeat(biases, GS, 1)
    return (sc * nib + bi).astype(np.float32)


def rmsnorm(x, w, eps):
    var = np.mean(x.astype(np.float64) ** 2, -1, keepdims=True)
    xn = (x / np.sqrt(var + eps)).astype(np.float32)
    return (xn * w).astype(np.float32) if w is not None else xn


def yarn(pos):
    half = ROPE_DIM // 2
    def cd(nr): return ROPE_DIM * np.log(ORIG_MAX / (nr * 2 * np.pi)) / (2 * np.log(BASE))
    low = max(0, int(np.floor(cd(BETA_FAST)))); high = min(ROPE_DIM - 1, int(np.ceil(cd(BETA_SLOW))))
    cos = np.empty(half, np.float32); sin = np.empty(half, np.float32)
    for i in range(half):
        freq = 1.0 / (BASE ** (2.0 * i / ROPE_DIM))
        ramp = 0.0 if low == high and i <= low else (1.0 if low == high else min(1.0, max(0.0, (i - low) / (high - low))))
        smooth = 1.0 - ramp
        freq = freq / FACTOR * (1.0 - smooth) + freq * smooth
        cos[i] = np.cos(pos * freq); sin[i] = np.sin(pos * freq)
    return cos, sin


def rope_tail(vec_heads, cos, sin, inverse=False):
    out = vec_heads.copy(); half = ROPE_DIM // 2
    s = -sin if inverse else sin
    for h in range(vec_heads.shape[0]):
        for i in range(half):
            j0 = NOPE + 2 * i; j1 = NOPE + 2 * i + 1
            x0 = vec_heads[h, j0]; x1 = vec_heads[h, j1]
            out[h, j0] = x0 * cos[i] - x1 * s[i]
            out[h, j1] = x0 * s[i] + x1 * cos[i]
    return out


def attn_step(wq_a, q_norm, wq_b, wkv, kv_norm, wo_a, wo_b, attn_sink,
              x, pos, kv_cache):
    """One MLA decode step. Updates kv_cache[pos] and returns attn output."""
    cos, sin = yarn(pos)
    # Q chain
    q_a = wq_a @ x
    q_res = rmsnorm(q_a, q_norm, EPS)
    q = (wq_b @ q_res).reshape(H, HEAD_DIM)
    q = rmsnorm(q, None, EPS)
    q = rope_tail(q, cos, sin)
    # KV chain
    kv = rmsnorm(wkv @ x, kv_norm, EPS).reshape(1, HEAD_DIM)
    kv = rope_tail(kv, cos, sin)[0]
    kv_cache[pos] = kv
    # SDPA over all cached KV (pos+1 entries)
    n_cached = pos + 1
    out_heads = np.empty((H, HEAD_DIM), np.float32)
    for h in range(H):
        scores = np.array([float(np.dot(q[h].astype(np.float64),
                                        kv_cache[k].astype(np.float64))) * SCALE
                           for k in range(n_cached)])
        sink_s = float(attn_sink[h])
        m = max(scores.max(), sink_s)
        exp_scores = np.exp(scores - m)
        exp_sink = np.exp(sink_s - m)
        denom = exp_scores.sum() + exp_sink
        w = exp_scores / denom
        out_heads[h] = (w[:, None] * kv_cache[:n_cached]).sum(0).astype(np.float32)
    # inverse RoPE
    out_heads = rope_tail(out_heads, cos, sin, inverse=True)
    # wo_a grouped -> concat -> wo_b
    parts = []
    for g in range(N_GROUPS):
        gv = np.concatenate([out_heads[g * HPG + hh] for hh in range(HPG)])
        parts.append(wo_a[g * O_LORA:(g + 1) * O_LORA] @ gv)
    return (wo_b @ np.concatenate(parts)).astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    P = "model.layers.0.attn."

    wq_a = dense(P + "wq_a"); q_norm = load(P + "q_norm.weight").astype(np.float32)
    wq_b = dense(P + "wq_b"); wkv = dense(P + "wkv"); kv_norm = load(P + "kv_norm.weight").astype(np.float32)
    wo_a = dense(P + "wo_a"); wo_b = dense(P + "wo_b")
    attn_sink = load(P + "attn_sink").astype(np.float32)

    rng = np.random.default_rng(123)
    # Generate N_PREFILL+1 random hidden inputs
    hiddens = [(rng.standard_normal(4096) * 0.1).astype(np.float32)
               for _ in range(N_PREFILL + 1)]

    MAX_SEQ = 4096
    kv_cache = np.zeros((MAX_SEQ, HEAD_DIM), np.float32)

    # Prefill: process each token sequentially
    for i in range(N_PREFILL):
        attn_step(wq_a, q_norm, wq_b, wkv, kv_norm, wo_a, wo_b, attn_sink,
                  hiddens[i], i, kv_cache)

    # Decode step at pos = N_PREFILL
    decode_pos = N_PREFILL
    golden = attn_step(wq_a, q_norm, wq_b, wkv, kv_norm, wo_a, wo_b, attn_sink,
                       hiddens[decode_pos], decode_pos, kv_cache)

    # Save
    for i, h in enumerate(hiddens):
        h.tofile(os.path.join(OUT, f"hidden_{i:02d}.bin"))
    kv_cache[:N_PREFILL + 1].astype(np.float32).tofile(os.path.join(OUT, "kv_cache.bin"))
    golden.tofile(os.path.join(OUT, "golden.bin"))
    with open(os.path.join(OUT, "meta.txt"), "w") as f:
        f.write(f"N_PREFILL={N_PREFILL}\ndecode_pos={decode_pos}\n")
    print(f"Multi-step golden: N_PREFILL={N_PREFILL}, decode_pos={decode_pos}")
    print(f"  golden[:4]={golden[:4]}  norm={np.linalg.norm(golden):.4f}")
    print(f"  KV cache at pos 0: norm={np.linalg.norm(kv_cache[0]):.4f}")
    print(f"  KV cache at pos {decode_pos}: norm={np.linalg.norm(kv_cache[decode_pos]):.4f}")


if __name__ == "__main__":
    main()
