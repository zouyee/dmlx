#!/usr/bin/env python3
"""Kernel-level bench: stock causal SDPA vs omlx _wide_causal_sdpa split-KV
at the dmlx verify shape (H=24, HKV=4, D=256, L=8, bf16), long KV.

Run: ~/venvs/dmlx-cmp/bin/python3 scripts/bench_verify_sdpa.py [kv_len]
"""
import sys, time, types
import mlx.core as mx

# Stub the package context so the patch file's relative import resolves.
pkg = types.ModuleType("omlx_patches_stub")
pkg.__path__ = ["/Users/zouyee/work/code/omlx/omlx/patches"]
sys.modules["omlx_patches_stub"] = pkg
sys.modules["omlx_patches_stub.qwen35_verify_qmm"] = types.SimpleNamespace(
    is_row_exact_armed=lambda: False)
pkg.__name__ = "omlx_patches_stub"
import importlib.util
spec = importlib.util.spec_from_file_location(
    "omlx_patches_stub.qwen35_verify_sdpa_split",
    "/Users/zouyee/work/code/omlx/omlx/patches/qwen35_verify_sdpa_split.py")
vs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vs)

H, HKV, D, L = 24, 4, 256, 8
T = int(sys.argv[1]) if len(sys.argv) > 1 else 16400
scale = 1.0 / (D ** 0.5)

q = mx.random.normal((1, H, L, D)).astype(mx.bfloat16)
k = mx.random.normal((1, HKV, T, D)).astype(mx.bfloat16)
v = mx.random.normal((1, HKV, T, D)).astype(mx.bfloat16)
mx.eval(q, k, v)


def stock():
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")


def wide():
    return vs._wide_causal_sdpa(q, k, v, scale)


# correctness spot-check
a, b = stock(), wide()
mx.eval(a, b)
diff = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
print(f"max abs diff stock vs wide: {diff:.4f}")

for name, fn in [("stock", stock), ("wide", wide)]:
    for _ in range(3):
        mx.eval(fn())
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    print(f"{name:6s} kv={T} L={L}: median {ts[len(ts)//2]:.2f} ms  min {ts[0]:.2f}  max {ts[-1]:.2f}")
