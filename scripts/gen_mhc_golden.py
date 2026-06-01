#!/usr/bin/env python3
"""Generate an mHC pre/post golden for the C CPU mHC implementation.

Uses layer-0 attn_hc weights + a random residual [HC, DIM]. Computes
mhc_pre (sublayer input, post_mix, sinkhorn comb) and mhc_post, mirroring
deepseek_v4.zig (validated in verify_mhc.py). Dumps to /tmp/mhc_golden/.

Usage: python3 scripts/gen_mhc_golden.py
"""
import os, json, struct, glob, sys
import numpy as np

MODEL = "/Users/zouyee/models/DeepSeek-V4-Flash-4bit"
OUT = "/tmp/mhc_golden"
HC = 4
DIM = 4096
MHC_H = HC * DIM
EPS = 1e-6
ITERS = 20
POST_MULT = 2.0


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


def load(name):
    dt, sh, raw = find(name)
    a = np.frombuffer(raw, "<f4") if dt == "F32" else None
    return a.reshape(sh)


def pre_norm_fn(fn, residual):
    res = residual.reshape(MHC_H).astype(np.float64)
    norm = 1.0 / np.sqrt(np.mean(res ** 2) + EPS)
    return ((fn.astype(np.float64) @ res) * norm).astype(np.float32)  # [24]


def sinkhorn(comb):
    e = np.exp(comb - comb.max(-1, keepdims=True))
    cur = (e / e.sum(-1, keepdims=True)).astype(np.float64) + EPS
    cur = cur / (cur.sum(-2, keepdims=True) + EPS)
    for _ in range(ITERS - 1):
        cur = cur / (cur.sum(-1, keepdims=True) + EPS)
        cur = cur / (cur.sum(-2, keepdims=True) + EPS)
    return cur


def main():
    os.makedirs(OUT, exist_ok=True)
    fn = load("model.layers.0.attn_hc.fn")     # [24,16384]
    base = load("model.layers.0.attn_hc.base") # [24]
    scale = load("model.layers.0.attn_hc.scale") # [3]

    rng = np.random.default_rng(123)
    residual = (rng.standard_normal((HC, DIM)) * 0.1).astype(np.float32)
    x = (rng.standard_normal(DIM) * 0.1).astype(np.float32)  # sublayer output

    mixes = pre_norm_fn(fn, residual)  # [24]
    pre = 1.0 / (1.0 + np.exp(-(mixes[:HC] * scale[0] + base[:HC]))) + EPS
    post = (1.0 / (1.0 + np.exp(-(mixes[HC:2*HC] * scale[1] + base[HC:2*HC])))) * POST_MULT
    comb = (mixes[2*HC:] * scale[2] + base[2*HC:]).reshape(HC, HC)
    comb_n = sinkhorn(comb).astype(np.float32)

    # sublayer input = sum_m pre[m]*residual[m]
    sub_in = (pre[:, None] * residual).sum(0).astype(np.float32)

    # post: out[m] = post[m]*x + sum_k comb_n[k][m]*residual[k]
    out_res = np.empty((HC, DIM), np.float32)
    for m in range(HC):
        acc = post[m] * x
        for k in range(HC):
            acc = acc + comb_n[k, m] * residual[k]
        out_res[m] = acc

    residual.tofile(os.path.join(OUT, "residual.bin"))
    x.tofile(os.path.join(OUT, "x.bin"))
    fn.astype(np.float32).tofile(os.path.join(OUT, "fn.bin"))
    base.astype(np.float32).tofile(os.path.join(OUT, "base.bin"))
    scale.astype(np.float32).tofile(os.path.join(OUT, "scale.bin"))
    sub_in.tofile(os.path.join(OUT, "sub_in.bin"))
    post.astype(np.float32).tofile(os.path.join(OUT, "post.bin"))
    comb_n.tofile(os.path.join(OUT, "comb.bin"))
    out_res.tofile(os.path.join(OUT, "out_res.bin"))
    print(f"mhc golden -> {OUT}; sub_in[:3]={sub_in[:3]} post={post} comb[0]={comb_n[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
