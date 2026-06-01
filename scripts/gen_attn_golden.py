#!/usr/bin/env python3
"""Generate a layer-0 MLA attention golden + raw weights for the C harness.

Dumps to /tmp/attn_golden/:
  meta.txt              dims
  hidden.bin            [4096] f32 input (post input-RMSNorm, attention input)
  wq_a.{packed,sc,bi}   affine 4bit raw (u32 packed, f32 scales, f32 biases)
  q_norm.bin            [1024] f32
  wq_b.{packed,sc,bi}
  wkv.{packed,sc,bi}
  kv_norm.bin           [512] f32
  wo_a.{packed,sc,bi}   [8,1024,4096] grouped (packed [8,1024,512])
  wo_b.{packed,sc,bi}
  attn_sink.bin         [64] f32
  cos.bin sin.bin       [32] f32  (YaRN tail RoPE for pos)
  golden.bin            [4096] f32 attention output (decode, single token)

The golden is computed with the S1-S5 validated numpy primitives. The C
host attention function (mla_attention_decode) must reproduce golden.bin.

Usage: python3 scripts/gen_attn_golden.py [pos]
"""
import sys, os, json, struct, glob
import numpy as np

MODEL = "/Users/zouyee/models/DeepSeek-V4-Flash-4bit"
OUT = "/tmp/attn_golden"
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


def dump_quant(d, name):
    packed = load(name + ".weight"); scales = load(name + ".scales").astype(np.float32); biases = load(name + ".biases").astype(np.float32)
    short = name.split(".")[-1]
    packed.astype("<u4").tofile(os.path.join(d, short + ".packed"))
    scales.tofile(os.path.join(d, short + ".sc"))
    biases.tofile(os.path.join(d, short + ".bi"))


def main():
    pos = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    os.makedirs(OUT, exist_ok=True)
    P = "model.layers.0.attn."

    wq_a = dense(P + "wq_a"); q_norm = load(P + "q_norm.weight").astype(np.float32)
    wq_b = dense(P + "wq_b"); wkv = dense(P + "wkv"); kv_norm = load(P + "kv_norm.weight").astype(np.float32)
    wo_a = dense(P + "wo_a"); wo_b = dense(P + "wo_b")
    attn_sink = load(P + "attn_sink").astype(np.float32)

    rng = np.random.default_rng(42)
    hidden = (rng.standard_normal(4096) * 0.1).astype(np.float32)  # attention input (post input-norm)
    cos, sin = yarn(pos)

    # Q chain
    q_a = wq_a @ hidden
    q_res = rmsnorm(q_a, q_norm, EPS)
    q = (wq_b @ q_res).reshape(H, HEAD_DIM)
    q = rmsnorm(q, None, EPS)
    q = rope_tail(q, cos, sin)
    # KV chain (single head)
    kv = rmsnorm(wkv @ hidden, kv_norm, EPS).reshape(1, HEAD_DIM)
    kv = rope_tail(kv, cos, sin)[0]  # [512]
    # SDPA decode: 1 query token, 1 cached key (itself), MQA broadcast + sink
    out_heads = np.empty((H, HEAD_DIM), np.float32)
    for h in range(H):
        score = float(np.dot(q[h].astype(np.float64), kv.astype(np.float64))) * SCALE
        m = max(score, float(attn_sink[h]))
        denom = np.exp(score - m) + np.exp(attn_sink[h] - m)
        w = np.exp(score - m) / denom
        out_heads[h] = (w * kv).astype(np.float32)
    # inverse RoPE
    out_heads = rope_tail(out_heads, cos, sin, inverse=True)
    # grouped wo_a -> concat -> wo_b
    parts = []
    for g in range(N_GROUPS):
        gv = np.concatenate([out_heads[g * HPG + hh] for hh in range(HPG)])  # 4096
        parts.append(wo_a[g * O_LORA:(g + 1) * O_LORA] @ gv)
    attn_out = (wo_b @ np.concatenate(parts)).astype(np.float32)

    # dump
    hidden.tofile(os.path.join(OUT, "hidden.bin"))
    for nm in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"]:
        dump_quant(OUT, P + nm)
    q_norm.tofile(os.path.join(OUT, "q_norm.bin"))
    kv_norm.tofile(os.path.join(OUT, "kv_norm.bin"))
    attn_sink.tofile(os.path.join(OUT, "attn_sink.bin"))
    cos.tofile(os.path.join(OUT, "cos.bin")); sin.tofile(os.path.join(OUT, "sin.bin"))
    attn_out.tofile(os.path.join(OUT, "golden.bin"))
    with open(os.path.join(OUT, "meta.txt"), "w") as f:
        f.write(f"pos={pos}\n")
    print(f"golden written to {OUT}; attn_out[:4]={attn_out[:4]} norm={np.linalg.norm(attn_out):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
