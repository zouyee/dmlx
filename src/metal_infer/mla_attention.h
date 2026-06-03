// MLA attention host orchestration (S7b).
// Chains the validated S2-S5 kernels into one decode-step attention pass,
// entirely in Metal (no MLX round-trip). Designed to be testable in isolation
// (scripts/metal_kernel_test.m) before wiring into engine.c.
#pragma once
#include <Metal/Metal.h>
#include "engine.h"

// Pipelines needed by MLA attention (subset of the engine's pipelines).
typedef struct {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> dequant_matvec_affine;
    id<MTLComputePipelineState> rms_norm_rows;
    id<MTLComputePipelineState> rope_tail_interleaved;
    id<MTLComputePipelineState> mla_sdpa_decode;
    id<MTLComputePipelineState> matvec_f32; // dense matvec for wo_a
    // bf16-output variants for Q chain (matches MLX bf16 intermediate precision)
    id<MTLComputePipelineState> dequant_matvec_affine_bf16;
    id<MTLComputePipelineState> rms_norm_rows_bf16;
    id<MTLComputePipelineState> bf16_to_f32;
    // Full bf16 chain: bfloat input + bfloat output kernels
    id<MTLComputePipelineState> f32_to_bf16;
    id<MTLComputePipelineState> dequant_matvec_affine_bf16in_f32out;
    id<MTLComputePipelineState> dequant_matvec_affine_bf16in_bf16out;
    id<MTLComputePipelineState> rms_norm_rows_bf16in_bf16out;
    id<MTLComputePipelineState> rope_tail_bf16;        // in-place bfloat RoPE
    id<MTLComputePipelineState> matvec_f32_bf16in;     // dense matmul with bfloat input
} MlaPipes;

// Compute one decode-step MLA attention.
//   pipes    : compiled pipelines
//   aw       : layer attention weights (quantized pointers + norms + sink)
//   x        : [DIM] attention input (already input-RMSNorm'd), f32
//   kv_cache : [MAX_SEQ_LEN, KV_LORA_RANK] f32, this layer's cache
//   cache_len: number of valid cached KV rows (>=1; includes current token)
//   pos      : current sequence position (for RoPE)
//   out      : [DIM] attention output, f32 (caller-allocated)
// Returns 0 on success.
int mla_attention_decode(MlaPipes *pipes, const AttnWeights *aw,
                         const float *x, float *kv_cache, int cache_len,
                         int pos, float *out);
