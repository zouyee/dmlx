#!/usr/bin/env python3
"""Reference for the GlmKDA single-step unit test (small config).

Mirrors src/models/glm53_kda.zig test construction exactly:
deterministic weight fill (i*37+11)%13-6)*0.02, numpy-only gated delta rule.
Prints expected output and SSM state for hardcoding into the Zig test.
"""
import numpy as np

HIDDEN, H, D, K = 8, 2, 4, 4
P = H * D


def fill(n):
    i = np.arange(n)
    return (((i * 37 + 11) % 13 - 6) * 0.02).astype(np.float64)


def W(out_f, in_f):
    return fill(out_f * in_f).reshape(out_f, in_f)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rmsnorm(x, eps, weight=None):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if weight is None else y * weight


qkv_proj = W(3 * P, HIDDEN)
qkv_conv = fill(3 * P * K).reshape(3 * P, K)
fbg_a_proj = W(2 * D + H, HIDDEN)
f_b_proj = W(P, D)
g_b_proj = W(P, D)
A_log = np.array([-0.5, -0.75])
dt_bias = fill(P)
o_norm_w = np.array([1.1, 0.9, 1.05, 0.95])
o_proj = W(HIDDEN, P)

i = np.arange(HIDDEN)
x = (((i * 13 + 5) % 11 - 5) * 0.1).astype(np.float64)

qkv_lin = x @ qkv_proj.T
conv_input = np.concatenate([np.zeros((K - 1, 3 * P)), qkv_lin[None, :]], axis=0)
conv_out = sum(qkv_conv[:, j] * conv_input[j] for j in range(K))
qkv = conv_out * sigmoid(conv_out)  # silu

q, k, v = np.split(qkv, 3)
scale = D ** -0.5
eps = 1e-6 / D
q = (scale**2) * rmsnorm(q.reshape(H, D), eps)
k = scale * rmsnorm(k.reshape(H, D), eps)
v = v.reshape(H, D)

fbg = x @ fbg_a_proj.T
f_a, b, g_a = fbg[:D], fbg[D : D + H], fbg[D + H :]
a = (f_a @ f_b_proj.T).reshape(H, D)
gate = (g_a @ g_b_proj.T).reshape(H, D)

beta = sigmoid(b)
g = np.exp(-5.0 * sigmoid(np.exp(A_log)[:, None] * (a + dt_bias.reshape(H, D))))

S = np.zeros((H, D, D))  # [H, Dv, Dk]
S = S * g[:, None, :]
kv_mem = (S * k[:, None, :]).sum(-1)
delta = (v - kv_mem) * beta[:, None]
S = S + k[:, None, :] * delta[..., None]
y = (S * q[:, None, :]).sum(-1)

y = rmsnorm(y, 1e-5, o_norm_w) * sigmoid(gate)
out = y.reshape(1, P) @ o_proj.T

np.set_printoptions(precision=6, suppress=True)
print("expected_out =", ", ".join(f"{v:.6f}" for v in out[0]))
print("expected_state =")
for row in S.reshape(-1, 4):
    print("    " + ", ".join(f"{v:.6f}" for v in row) + ",")
