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
    id<MTLComputePipelineState> mla_sdpa_decode_f16; // KV cache f16 (ds4 path)
    id<MTLComputePipelineState> matvec_f32; // dense matvec for wo_a
    // F16 precision chain (ds4-style)
    id<MTLComputePipelineState> dequant_matvec_affine_f16out;
    id<MTLComputePipelineState> rms_norm_rows_f16out;
    id<MTLComputePipelineState> dequant_matvec_affine_f16in_f16out;
    id<MTLComputePipelineState> rms_norm_rows_f16in_f16out;
    id<MTLComputePipelineState> rope_tail_interleaved_f16;
    id<MTLComputePipelineState> matvec_f32_f16in;
    id<MTLComputePipelineState> mla_sdpa_decode_f16in_f16out;
    // BF16 precision chain (S8b)
    id<MTLComputePipelineState> dequant_matvec_affine_bf16out;
    id<MTLComputePipelineState> rms_norm_rows_bf16out;
    id<MTLComputePipelineState> dequant_matvec_affine_bf16in_bf16out;
    id<MTLComputePipelineState> rms_norm_rows_bf16in_bf16out;
    id<MTLComputePipelineState> rope_tail_interleaved_bf16;
    id<MTLComputePipelineState> matvec_f32_bf16in;
    id<MTLComputePipelineState> mla_sdpa_decode_bfloat;
    id<MTLComputePipelineState> dequant_matvec_affine_bf16in_f32out;
    // Prefill batch SDPA (Path B: matches MLX simdgroup reduction order)
    id<MTLComputePipelineState> mla_sdpa_prefill_bfloat;
    // KV cache bf16→f16 conversion (used in CB1+CB2 merge to avoid CPU round-trip)
    id<MTLComputePipelineState> bf16_to_f16_row;
} MlaPipes;

// Compute one decode-step MLA attention.
//   pipes    : compiled pipelines
//   aw       : layer attention weights (quantized pointers + norms + sink)
//   x        : [DIM] attention input (already input-RMSNorm'd), f32
//   kv_cache : [MAX_SEQ_LEN, KV_LORA_RANK] f16, this layer's cache
//   cache_len: number of valid cached KV rows (>=1; includes current token)
//   pos      : current sequence position (for RoPE)
//   out      : [DIM] attention output, f32 (caller-allocated)
// Returns 0 on success.
int mla_attention_decode(MlaPipes *pipes, const AttnWeights *aw,
                         const float *x, uint16_t *kv_cache, int cache_len,
                         int pos, float *out);

// F16-KV variant (ds4-style): Q chain is f32, KV cache is f32 (was f16, now upgraded for accuracy),
// SDPA uses mla_sdpa_decode_f16 (Q f32 · KV f32), output is f32.
int mla_attention_decode_f16kv(MlaPipes *pipes, const AttnWeights *aw,
                               const float *x, float *kv_cache, int cache_len,
                               int pos, float *out);

// BF16 variant: x is bfloat16, kv_cache is bfloat16, out is float.
// kv_cache_gpu_buf: optional id<MTLBuffer> wrapping kv_cache in Shared mode.
// When non-NULL, the function uses GPU blit to update the KV cache and merges
// CB1+CB2 into one command buffer (saves 1 GPU sync = ~8ms/layer).
// When NULL, falls back to CPU KV copy (safe for any kv_cache pointer).
int mla_attention_decode_bf16(MlaPipes *pipes, const AttnWeights *aw,
                              const uint16_t *x, uint16_t *kv_cache, int cache_len,
                              int pos, float *out, void *kv_cache_gpu_buf);

// Mixed attention: uint16_t raw KV (SWA window) + f32 comp_kv (selected blocks).
int mla_attention_decode_mixed(MlaPipes *pipes, const AttnWeights *aw,
                               const float *x, uint16_t *raw_kv_cache, int raw_cache_len,
                               int pos, const float *comp_kv, int n_comp,
                               const bool *comp_allowed, float *out);

// Batch prefill attention for N tokens (bf16 end-to-end).
// x_batch   : [n_tokens, DIM] bfloat16 (pre-computed attention-norm'd inputs)
// kv_cache  : [MAX_SEQ_LEN, KV_LORA_RANK] bf16, this layer's cache (written in place)
// start_pos : sequence position of x_batch[0]
// out_batch : [n_tokens, DIM] float (attention outputs, caller-allocated)
// Returns 0 on success.
int mla_attention_prefill_bfloat(MlaPipes *pipes, const AttnWeights *aw,
                                  const uint16_t *x_batch, int n_tokens,
                                  uint16_t *kv_cache, int start_pos,
                                  float *out_batch);
