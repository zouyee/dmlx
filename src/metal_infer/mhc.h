// mHC (Manifold-Constrained Hyper-Connections) — CPU host implementation (S6/S7c).
// mHC compute is tiny (HC=4: 4x4 comb, 24-wide mixes) so it stays on CPU.
// Validated against MLX in scripts/verify_mhc.py + the layer golden.
#pragma once
#include "engine.h"

// mHC weights for one sublayer wrapper (attn_hc or ffn_hc). All f32.
typedef struct {
    const float *fn;     // [mhc_mult*(mhc_mult+2), mhc_mult*H] = [24, 16384]
    const float *base;   // [24]
    const float *scale;  // [3]  (pre, post, comb scales)
} MhcWeights;

// F16-precision variant (ds4-style): fn weights are f16, everything else f32.
typedef struct {
    const uint16_t *fn;  // [24, 16384] f16 weights
    const float *base;   // [24]
    const float *scale;  // [3]
} MhcWeightsF16;

// hc.pre: from residual streams [HC, H], compute the sublayer input [H] and the
// post/comb mixes to be applied after the sublayer.
//   residual : [MHC_MULT, DIM]
//   out_input: [DIM]                 (sublayer input = sum_m pre_mix[m]*residual[m])
//   out_post : [MHC_MULT]            (post-layer mix)
//   out_comb : [MHC_MULT*MHC_MULT]   (sinkhorn-normalized comb, row-major [k][m])
void mhc_pre(const MhcWeights *w, const float *residual,
             float *out_input, float *out_post, float *out_comb);

// F16-precision variant (ds4-style): fn weights are f16, matvec uses f16
// operands with f32 accumulation. Everything else stays f32.
void mhc_pre_f16(const MhcWeightsF16 *w, const float *residual,
                 float *out_input, float *out_post, float *out_comb);

// Same as mhc_pre but also returns pre_mix[MHC_MULT] so the caller can
// re-run the blend step on GPU (for bf16 precision matching MLX).
void mhc_pre_with_premix(const MhcWeights *w, const float *residual,
                          float *out_input, float *out_post, float *out_comb,
                          float *out_premix);

// hc.post: combine sublayer output x[DIM] with residual streams using post/comb.
//   out_residual[m,:] = post[m]*x + sum_k comb[k][m]*residual[k,:]
//   x        : [DIM]
//   residual : [MHC_MULT, DIM]
//   post     : [MHC_MULT]
//   comb     : [MHC_MULT*MHC_MULT]
//   out_residual : [MHC_MULT, DIM]
void mhc_post(const float *x, const float *residual,
              const float *post, const float *comb, float *out_residual);

// Compress final residual [MHC_MULT, DIM] -> [DIM] for the final norm + lm head.
// (HyperHead). For now uses the simple learned-mix compression.
void mhc_head_compress(const MhcWeights *w, const float *residual, float *out);

// HyperHead compression using standalone hc_head.{fn,base,scale} weights.
void hyper_head_compress(const float *fn, const float *base, const float *scale,
                         const float *residual, float *out);
