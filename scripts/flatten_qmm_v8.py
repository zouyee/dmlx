#!/usr/bin/env python3
"""Flatten the MLX steel GEMM header closure + omlx's vendored
affine_quantized.h into one self-contained Metal header for dmlx's fast-metal
kernel port of the affine_q4_qmm_t variant-8 tile (bm=64, bk=32, bn=64).

Include root: /opt/homebrew/include (mlx-c install). `mlx/...`-quoted includes
are expanded recursively (pragma-once dedup); angle includes are kept.
The main kernels/utils.h is NOT expanded: mlx-c's fast metal kernel already
prepends it to the compilation unit.
"""
import re
import sys

ROOT = "/opt/homebrew/include"
OUT = "src/models/qwen38_qmm_v8_steel.metal"

ENTRY = [
    "mlx/backend/metal/kernels/steel/gemm/gemm.h",
    # omlx's vendored copy (verbatim from the omlx repo):
    "/Users/zouyee/work/code/omlx/omlx/custom_kernels/quant_matmul/csrc/affine_quantized.h",
]

SKIP = {"mlx/backend/metal/kernels/utils.h"}  # auto-prepended by mlx-c

inc_re = re.compile(r'^\s*#include\s+"([^"]+)"')

seen = set()
out = []


def expand(path: str) -> None:
    if path in seen or path in SKIP:
        return
    seen.add(path)
    full = path if path.startswith("/") else f"{ROOT}/{path}"
    with open(full) as f:
        lines = f.readlines()
    out.append(f"// ---- begin {path} ----\n")
    for ln in lines:
        m = inc_re.match(ln)
        if m:
            expand(m.group(1))
            continue
        if ln.strip() == "#pragma once":
            continue
        out.append(ln)
    out.append(f"// ---- end {path} ----\n")


for e in ENTRY:
    expand(e)

# qmm_t_impl takes `const constant int&` dims, which only bind to constant
# address-space values (buffer constants). Our fast-metal kernel bakes the
# dims as literals in the generated body, so switch the impl's four int
# params to by-value (source-compatible with the other, unused callers).
text = "".join(out)
old_sig = """    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& K_eff,"""
new_sig = """    threadgroup T* Xs,
    threadgroup T* Ws,
    int K,
    int N,
    int M,
    int K_eff,"""
assert text.count(old_sig) == 1, text.count(old_sig)
text = text.replace(old_sig, new_sig)
out = [text]

# The kernel entry points: v8 tile (64,32,64), q4, gs=64, bf16 + f16.
out.append(
    r"""
// ---- dmlx entry: affine_q4_qmm_t variant 8 (bm=64, bk=32, bn=64) ----
// threadgroup storage is declared by the caller (Metal only allows
// threadgroup variables at kernel-function scope).
template <typename T>
void dmlx_qmm_v8_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    int K,
    int N,
    int M,
    uint3 tid,
    uint lid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int BM = 64;
  constexpr int BK = 32;
  constexpr int BN = 64;

  qmm_t_impl<T, 64, 4, true, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}
"""
)

with open(OUT, "w") as f:
    f.writelines(out)
print(f"wrote {OUT}: {sum(len(l) for l in out)} bytes, {len(out)} lines")
