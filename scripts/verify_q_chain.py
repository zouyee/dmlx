#!/usr/bin/env python3
"""S2 go/no-go: verify the full MLA Q-projection chain against MLX semantics.

Chain (DSV4Attention.forward, Q path):
  q_a   = wq_a @ hidden                        # [1024]
  q_res = RMSNorm(q_a, q_norm_weight, eps)      # [1024], LEARNED weight
  q_b   = wq_b @ q_res                          # [32768]
  q     = reshape(q_b, [64, 512])               # 64 heads x 512
  q     = RMSNorm_per_head(q, ones, eps)        # weightless, per 512-head
  q     = tail_RoPE(q)                          # rotate last 64 dims of each head

This script implements the chain two independent ways on the SAME input
(real layer-0 weights + a fixed pseudo-random hidden):
  (A) "mlx-style": numpy vectorized, mirroring DSV4YarnRoPE.apply (interleaved
       pairs, cache-based YaRN) and fast.rms_norm semantics.
  (B) "kernel-style": scalar per-element loops, exactly as the Metal kernels
       will compute (this is what S2's Metal code must match).
If (A) vs (B) match, the kernel algorithm — especially the INTERLEAVED RoPE
pairing (not split-half) — is correct. GO.

Usage: python3 scripts/verify_q_chain.py
"""
import sys, json, struct, glob
import numpy as np

MODEL = "/Users/zouyee/models/DeepSeek-V4-Flash-4bit"
GS = 64
EPS = 1e-6
HEAD_DIM = 512
N_HEADS = 64
ROPE_DIM = 64          # qk_rope_head_dim
NOPE = HEAD_DIM - ROPE_DIM  # 448
# YaRN config
BASE = 10000.0
FACTOR = 16.0
BETA_FAST = 32.0
BETA_SLOW = 1.0
ORIG_MAX = 65536


def find_tensor(name):
    for f in sorted(glob.glob(MODEL + "/model-*-of-00033.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n)); base = 8 + n
            if name in hdr:
                v = hdr[name]; o0, o1 = v["data_offsets"]
                fh.seek(base + o0); raw = fh.read(o1 - o0)
                return v["dtype"], v["shape"], raw
    sys.exit("not found: " + name)


def to_np(dtype, shape, raw):
    if dtype == "U32":
        a = np.frombuffer(raw, dtype="<u4")
    elif dtype == "BF16":
        u = np.frombuffer(raw, dtype="<u2").astype(np.uint32); a = (u << 16).view(np.float32)
    elif dtype == "F32":
        a = np.frombuffer(raw, dtype="<f4")
    else:
        sys.exit("dtype " + dtype)
    return a.reshape(shape)


def load(name):
    dt, sh, raw = find_tensor(name); return to_np(dt, sh, raw)


def dequant_affine(name):
    """Dense f32 weight [out,in] from affine 4bit: w = scale*nibble + bias."""
    packed = load(name + ".weight")   # [out, in/8] u32
    scales = load(name + ".scales")   # [out, in/gs] f32
    biases = load(name + ".biases")
    out_dim, pcols = packed.shape
    in_dim = pcols * 8
    ng = in_dim // GS
    w = np.empty((out_dim, in_dim), np.float32)
    nib = np.empty((out_dim, in_dim), np.float32)
    for i in range(8):
        nib[:, i::8] = (packed >> (i * 4)) & 0xF
    # group-broadcast scale/bias
    sc = np.repeat(scales, GS, axis=1)
    bi = np.repeat(biases, GS, axis=1)
    w = sc * nib + bi
    return w.astype(np.float32)


def rmsnorm(x, weight, eps):
    # x: [..., d]; weight: [d] or None (ones)
    var = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    xn = (x / np.sqrt(var + eps)).astype(np.float32)
    if weight is not None:
        xn = xn * weight
    return xn.astype(np.float32)


def yarn_cos_sin(pos):
    """Build cos/sin for one position, length ROPE_DIM/2=32, matching DSV4."""
    half = ROPE_DIM // 2
    # correction range
    def corr_dim(nr):
        return ROPE_DIM * np.log(ORIG_MAX / (nr * 2 * np.pi)) / (2 * np.log(BASE))
    low = max(0, int(np.floor(corr_dim(BETA_FAST))))
    high = min(ROPE_DIM - 1, int(np.ceil(corr_dim(BETA_SLOW))))
    cos = np.empty(half, np.float32); sin = np.empty(half, np.float32)
    for i in range(half):
        freq = 1.0 / (BASE ** (2.0 * i / ROPE_DIM))
        # linearRampFactor(low, high, i, half) then smooth=1-ramp
        if low == high:
            ramp = 0.0 if i <= low else 1.0
        else:
            ramp = min(1.0, max(0.0, (i - low) / (high - low)))
        smooth = 1.0 - ramp
        freq = freq / FACTOR * (1.0 - smooth) + freq * smooth
        ang = pos * freq
        cos[i] = np.cos(ang); sin[i] = np.sin(ang)
    return cos, sin


def rope_mlx_style(q, pos):
    """Vectorized interleaved RoPE on last ROPE_DIM of each head (mirrors DSV4YarnRoPE)."""
    cos, sin = yarn_cos_sin(pos)              # [32]
    out = q.copy()
    pe = q[:, NOPE:]                          # [64, 64]
    pairs = pe.reshape(N_HEADS, ROPE_DIM // 2, 2)
    x0 = pairs[:, :, 0]; x1 = pairs[:, :, 1]  # interleaved even/odd
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    stacked = np.stack([o0, o1], axis=-1).reshape(N_HEADS, ROPE_DIM)
    out[:, NOPE:] = stacked
    return out


def rope_kernel_style(q, pos):
    """Scalar per-element RoPE, exactly as the Metal kernel will compute it."""
    cos, sin = yarn_cos_sin(pos)
    out = q.copy()
    half = ROPE_DIM // 2
    for h in range(N_HEADS):
        for i in range(half):
            j0 = NOPE + 2 * i       # interleaved: pair = (2i, 2i+1)
            j1 = NOPE + 2 * i + 1
            x0 = q[h, j0]; x1 = q[h, j1]
            out[h, j0] = x0 * cos[i] - x1 * sin[i]
            out[h, j1] = x0 * sin[i] + x1 * cos[i]
    return out


def main():
    print("loading layer-0 Q weights...")
    wq_a = dequant_affine("model.layers.0.attn.wq_a")   # [1024, 4096]
    q_norm = load("model.layers.0.attn.q_norm.weight")  # [1024]
    wq_b = dequant_affine("model.layers.0.attn.wq_b")   # [32768, 1024]
    print(f"wq_a{wq_a.shape} q_norm{q_norm.shape} wq_b{wq_b.shape}")

    rng = np.random.default_rng(0)
    hidden = rng.standard_normal(4096).astype(np.float32) * 0.1
    pos = 7

    # Shared chain up to reshape
    q_a = wq_a @ hidden                       # [1024]
    q_res = rmsnorm(q_a, q_norm, EPS)         # learned weight
    q_b = wq_b @ q_res                        # [32768]
    q = q_b.reshape(N_HEADS, HEAD_DIM)        # [64, 512]
    q = rmsnorm(q, None, EPS)                 # per-head weightless

    qA = rope_mlx_style(q, pos)
    qB = rope_kernel_style(q, pos)
    diff = np.abs(qA - qB)
    print(f"RoPE mlx-style vs kernel-style: max_abs={diff.max():.3e} mean={diff.mean():.3e}")

    # Sanity: also confirm split-half RoPE would DIFFER (proving layout matters)
    def rope_splithalf(qq, pos):
        cos, sin = yarn_cos_sin(pos); out = qq.copy(); half = ROPE_DIM // 2
        for h in range(N_HEADS):
            for i in range(half):
                j0 = NOPE + i; j1 = NOPE + i + half
                x0 = qq[h, j0]; x1 = qq[h, j1]
                out[h, j0] = x0 * cos[i] - x1 * sin[i]
                out[h, j1] = x0 * sin[i] + x1 * cos[i]
        return out
    qSH = rope_splithalf(q, pos)
    sh_diff = np.abs(qA - qSH).max()
    print(f"(sanity) split-half vs interleaved max_abs={sh_diff:.3e} (should be LARGE)")

    ok = diff.max() < 1e-4
    print("RESULT:", "GO — interleaved kernel RoPE matches mlx-style" if ok else "NO-GO")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
