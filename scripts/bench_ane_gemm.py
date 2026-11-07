#!/usr/bin/env python3
"""D-spike: CoreML/ANE FP16 GEMM microbenchmark at dmlx prefill shapes.

Builds one static matmul model per (M,K,N) shape from bench_prefill_v8.zig,
times predict() on CPU_AND_NEURAL_ENGINE, reports ms and effective TF/s.
FP16 weights are random; compute time is data-independent.
"""
import time
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types as miltypes

M = 2048
SHAPES = [  # (name, out N, in K, layers per pass)
    ("gdn_qkv ", 10240, 5120, 48),
    ("gdn_z   ", 6144, 5120, 48),
    ("gdn_out ", 5120, 6144, 48),
    ("attn_q  ", 12288, 5120, 16),
    ("attn_o  ", 5120, 6144, 16),
    ("mlp_gate", 17408, 5120, 64),
    ("mlp_down", 5120, 17408, 64),
]
REPS = 20

def build_model(K, N):
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((N, K)) * 0.02).astype(np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(M, K), dtype=miltypes.fp16)])
    def prog(x):
        wc = mb.const(val=w)
        y = mb.matmul(x=x, y=wc, transpose_y=True)
        return y

    return ct.convert(
        prog,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )

print(f"CoreML FP16 matmul, M={M}, ANE-targeted")
total_ms = 0.0
total_flops = 0
total_weighted_ms = 0.0
for name, n, k, layers in SHAPES:
    model = build_model(k, n)
    x = np.zeros((M, k), dtype=np.float16)
    model.predict({"x": x})  # warmup (compiles + first dispatch)
    model.predict({"x": x})
    t0 = time.perf_counter()
    for _ in range(REPS):
        model.predict({"x": x})
    ms = (time.perf_counter() - t0) / REPS * 1000
    fl = 2 * M * k * n
    tfs = fl / (ms / 1000) / 1e12
    print(f"{name} M={M} K={k:>6} N={n:>6}: {ms:8.3f} ms  {tfs:6.2f} TF/s  (x{layers} layers)")
    total_weighted_ms += ms * layers
    total_flops += fl * layers
print(f"weighted full-pass: {total_weighted_ms:.1f} ms, {total_flops/1e12:.1f} TFLOP, "
      f"effective {total_flops/(total_weighted_ms/1000)/1e12:.2f} TF/s")
