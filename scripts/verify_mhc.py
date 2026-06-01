#!/usr/bin/env python3
"""S6 go/no-go: verify mHC (Manifold-Constrained Hyper-Connections) math.

mHC wraps each sublayer (attention, MoE) with a learned multi-residual mix:
  pre:  mixes = preNormFn(residual, hc_fn) ; split -> pre_mix, post_mix, comb
        comb_norm = sinkhorn(comb)
        sublayer_input = sum_m(pre_mix[...,m] * residual[...,m,:])   # [B,S,H]
  post: out[...,m,:] = post_mix[...,m]*x + sum_k comb_norm[...,k,m]*residual[...,k,:]

This replicates the two NON-OBVIOUS pieces against MLX primitives:
  (1) sinkhornNormalize: softmax(-1) -> +eps -> col-norm -> [row-norm,col-norm]x(R-1)
  (2) mhcPost: term1 = x[...,None,:]*post_mix ; term2 = comb^T @ residual

numpy replication vs an MLX-ops replication of the same formulas; match => GO.

Usage: python3 scripts/verify_mhc.py
"""
import sys
import numpy as np

try:
    import mlx.core as mx
except Exception as e:
    sys.exit("mlx required: " + str(e))

HC = 4          # hc_mult
H = 16          # tiny hidden for the test
EPS = 1e-6
ITERS = 20


def sinkhorn_numpy(x):
    # x: [B,S,HC,HC] comb logits
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    sm = (e / e.sum(axis=-1, keepdims=True)).astype(np.float64)
    cur = sm + EPS
    # initial col-norm (axis=-2)
    cur = cur / (cur.sum(axis=-2, keepdims=True) + EPS)
    for _ in range(ITERS - 1):
        cur = cur / (cur.sum(axis=-1, keepdims=True) + EPS)   # row
        cur = cur / (cur.sum(axis=-2, keepdims=True) + EPS)   # col
    return cur


def sinkhorn_mlx(x):
    xm = mx.array(x)
    sm = mx.softmax(xm, axis=-1, precise=True)
    cur = sm + EPS
    cur = cur / (mx.sum(cur, axis=-2, keepdims=True) + EPS)
    for _ in range(ITERS - 1):
        cur = cur / (mx.sum(cur, axis=-1, keepdims=True) + EPS)
        cur = cur / (mx.sum(cur, axis=-2, keepdims=True) + EPS)
    return np.array(cur)


def post_numpy(x, residual, post_mix, comb):
    # x:[B,S,H] residual:[B,S,HC,H] post_mix:[B,S,HC,1] comb:[B,S,HC,HC]
    B, S, _ = x.shape
    term1 = x[:, :, None, :] * post_mix          # [B,S,HC,H]
    bs = B * S
    comb_2d = comb.reshape(bs, HC, HC)
    comb_t = np.transpose(comb_2d, (0, 2, 1))    # swapaxes(-1,-2)
    res_2d = residual.reshape(bs, HC, H)
    term2 = (comb_t @ res_2d).reshape(B, S, HC, H)
    return (term1 + term2)


def post_mlx(x, residual, post_mix, comb):
    B, S, _ = x.shape
    xm = mx.array(x); rm = mx.array(residual); pm = mx.array(post_mix); cm = mx.array(comb)
    term1 = mx.expand_dims(xm, 2) * pm
    bs = B * S
    comb_t = mx.swapaxes(cm.reshape(bs, HC, HC), -1, -2)
    term2 = (comb_t @ rm.reshape(bs, HC, H)).reshape(B, S, HC, H)
    return np.array(term1 + term2)


def main():
    rng = np.random.default_rng(0)
    comb = (rng.standard_normal((1, 1, HC, HC)) * 0.5).astype(np.float32)
    sk_n = sinkhorn_numpy(comb)
    sk_m = sinkhorn_mlx(comb)
    d1 = np.abs(sk_n - sk_m).max()
    print(f"sinkhorn: max_abs_diff={d1:.3e}")

    x = rng.standard_normal((1, 1, H)).astype(np.float32)
    residual = rng.standard_normal((1, 1, HC, H)).astype(np.float32)
    post_mix = rng.standard_normal((1, 1, HC, 1)).astype(np.float32)
    p_n = post_numpy(x, residual, post_mix, sk_n.astype(np.float32))
    p_m = post_mlx(x, residual, post_mix, sk_n.astype(np.float32))
    d2 = np.abs(p_n - p_m).max()
    print(f"mhcPost: max_abs_diff={d2:.3e}")

    # sanity: doubly-stochastic-ish — sinkhorn rows AND cols near 1 after iters
    rowsum = sk_n.sum(-1).ravel(); colsum = sk_n.sum(-2).ravel()
    print(f"(sanity) sinkhorn rowsum~[{rowsum.min():.3f},{rowsum.max():.3f}] colsum~[{colsum.min():.3f},{colsum.max():.3f}]")

    ok = d1 < 1e-5 and d2 < 1e-5
    print("RESULT:", "GO — mHC sinkhorn+post match MLX" if ok else "NO-GO")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
