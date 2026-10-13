#!/usr/bin/env python3
"""Reference for the Qwen3.8 GDN single-step unit test (small config).

Mirrors the Zig test construction exactly:
deterministic weight fill (i*37+11)%13-6)*0.02, numpy-only gated delta rule.
Prints expected output and SSM state for hardcoding into the Zig test.
"""
import numpy as np

HIDDEN, HK, HV, D, K = 8, 2, 6, 4, 4
QKV = 2 * HK * D + HV * D  # 40
RATIO = HV // HK


def fill(n):
    i = np.arange(n)
    return (((i * 37 + 11) % 13 - 6) * 0.02).astype(np.float64)


def W(out_f, in_f):
    return fill(out_f * in_f).reshape(out_f, in_f)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def rmsnorm(x, eps, weight=None):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if weight is None else y * weight


in_proj_qkv = W(QKV, HIDDEN)
conv_w = fill(QKV * K).reshape(QKV, K)
in_proj_a = W(HV, HIDDEN)
in_proj_b = W(HV, HIDDEN)
in_proj_z = W(HV * D, HIDDEN)
A_log = np.array([-0.5, -0.6, -0.7, -0.8, -0.9, -1.0])
dt_bias = fill(HV)
norm_w = np.array([1.1, 0.9, 1.05, 0.95])
out_proj = W(HIDDEN, HV * D)

i = np.arange(HIDDEN)
x = (((i * 13 + 5) % 11 - 5) * 0.1).astype(np.float64)

qkv_lin = x @ in_proj_qkv.T
window = np.concatenate([np.zeros((K - 1, QKV)), qkv_lin[None, :]], axis=0)
conv_out = sum(conv_w[:, j] * window[j] for j in range(K))
qkv = silu(conv_out)

q = qkv[: HK * D].reshape(HK, D)
k = qkv[HK * D : 2 * HK * D].reshape(HK, D)
v = qkv[2 * HK * D :].reshape(HV, D)

q = rmsnorm(q, 1e-6) * D ** -1.0
k = rmsnorm(k, 1e-6) * D ** -0.5

a = x @ in_proj_a.T
b = x @ in_proj_b.T
z = x @ in_proj_z.T

g = np.exp(-np.exp(A_log) * softplus(a + dt_bias))
beta = sigmoid(b)

qb = np.repeat(q, RATIO, axis=0)
kb = np.repeat(k, RATIO, axis=0)

S = np.zeros((HV, D, D))  # [HV, Dv, Dk]
S = S * g[:, None, None]
kv = (S * kb[:, None, :]).sum(-1)
d = (v - kv) * beta[:, None]
S = S + d[:, :, None] * kb[:, None, :]
y = (S * qb[:, None, :]).sum(-1)

yn = rmsnorm(y, 1e-6, norm_w) * silu(z.reshape(HV, D))
out = yn.reshape(-1) @ out_proj.T

print("out:", " ".join(f"{v:.6f}" for v in out))
print("state:", " ".join(f"{v:.6f}" for v in S.reshape(-1)))
