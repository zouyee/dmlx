#!/usr/bin/env python3
"""S4 go/no-go: verify SDPA + attention-sink against MLX fast SDPA.

Replicates the exact MLX sink math (mlx/fast.cpp fallback):
  scores = (q @ k^T) * scale            # [H, Lq, Lk]
  (apply causal mask)
  scores = concat([sink_h, scores], -1) # prepend per-head sink logit
  w = softmax(scores, -1)
  w = w[..., 1:]                         # drop sink column
  out = w @ v

Two implementations on the SAME random q/k/v + sinks:
  (A) MLX mx.fast.scaled_dot_product_attention(..., sinks=...)
  (B) "kernel-style" online softmax with dsv4_attend_sink semantics
      (running max/sum, fold sink into denominator at the end) — this is
      what the Metal SDPA kernel (ported from dsv4_misc.metal) computes.

MQA: 1 KV head broadcast to N query heads. Match => GO.

Usage: python3 scripts/verify_sdpa_sink.py
"""
import sys
import numpy as np

try:
    import mlx.core as mx
except Exception as e:
    sys.exit("mlx required: " + str(e))

H = 64          # query heads
HEAD_DIM = 512
SCALE = 1.0 / np.sqrt(HEAD_DIM)


def mlx_sdpa(q, k, v, sinks, scale):
    # q: [1,H,Lq,D]  k/v: [1,1,Lk,D]  sinks: [H]
    qн = mx.array(q); kн = mx.array(k); vн = mx.array(v); sн = mx.array(sinks)
    out = mx.fast.scaled_dot_product_attention(
        qн, kн, vн, scale=scale, mask="causal", sinks=sн)
    return np.array(out)


def kernel_sdpa(q, k, v, sinks, scale):
    """Online-softmax per (head, query) with sink folded into denominator.
    q: [H,Lq,D]  k/v: [Lk,D] (single KV head, broadcast)  sinks: [H]."""
    Hh, Lq, D = q.shape
    Lk = k.shape[0]
    out = np.zeros((Hh, Lq, D), np.float32)
    for h in range(Hh):
        for qi in range(Lq):
            qvec = q[h, qi].astype(np.float64)
            # causal: query position qi attends keys [0..qi] (Lq==Lk aligned at end)
            # here Lq==Lk so key range is [0..qi]
            m = -np.inf
            s = 0.0
            acc = np.zeros(D, np.float64)
            for ki in range(qi + 1):
                score = np.dot(qvec, k[ki].astype(np.float64)) * scale
                new_m = max(m, score)
                corr = np.exp(m - new_m) if m != -np.inf else 0.0
                p = np.exp(score - new_m)
                s = s * corr + p
                acc = acc * corr + p * v[ki].astype(np.float64)
                m = new_m
            # fold sink: sink logit competes in softmax denominator (dsv4_attend_sink)
            sink = float(sinks[h])
            new_m = max(m, sink)
            corr = np.exp(m - new_m)
            s = s * corr + np.exp(sink - new_m)
            acc = acc * corr
            out[h, qi] = (acc / s).astype(np.float32)
    return out


def main():
    rng = np.random.default_rng(0)
    Lq = Lk = 5
    q = (rng.standard_normal((1, H, Lq, HEAD_DIM)) * 0.1).astype(np.float32)
    k = (rng.standard_normal((1, 1, Lk, HEAD_DIM)) * 0.1).astype(np.float32)
    v = (rng.standard_normal((1, 1, Lk, HEAD_DIM)) * 0.1).astype(np.float32)
    sinks = (rng.standard_normal(H) * 0.5).astype(np.float32)

    a = mlx_sdpa(q, k, v, sinks, SCALE)[0]          # [H,Lq,D]
    # broadcast single KV head to kernel impl
    b = kernel_sdpa(q[0], k[0, 0], v[0, 0], sinks, SCALE)
    diff = np.abs(a - b)
    print(f"shapes mlx{a.shape} kernel{b.shape}")
    print(f"SDPA+sink: max_abs={diff.max():.3e} mean={diff.mean():.3e}")

    # sanity: without sink the result should DIFFER (proving sink matters)
    no_sink = kernel_sdpa(q[0], k[0, 0], v[0, 0], np.full(H, -1e9, np.float32), SCALE)
    print(f"(sanity) sink vs no-sink max_abs={np.abs(b-no_sink).max():.3e} (should be > 0)")

    ok = diff.max() < 2e-3
    print("RESULT:", "GO — SDPA+sink kernel math matches MLX" if ok else "NO-GO")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
