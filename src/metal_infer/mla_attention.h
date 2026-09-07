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
    id<MTLComputePipelineState> dequant_matvec_affine_v2; // coalesced ds4 pattern
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
    id<MTLComputePipelineState> dequant_matvec_affine_bf16in_bf16out_v2;  // SIMD-parallel
    id<MTLComputePipelineState> rms_norm_rows_bf16in_bf16out;
    id<MTLComputePipelineState> rope_tail_interleaved_bf16;
    id<MTLComputePipelineState> matvec_f32_bf16in;
    id<MTLComputePipelineState> matvec_q8_0_f32;    // ds4 Q8_0 matvec for wo_a
    id<MTLComputePipelineState> mla_sdpa_decode_bfloat;
    id<MTLComputePipelineState> dequant_matvec_affine_bf16in_f32out;
    id<MTLComputePipelineState> dequant_matvec_affine_bf16in_f32out_grp8; // fused wo_a
    // Prefill batch SDPA (legacy full-sequence path).
    id<MTLComputePipelineState> mla_sdpa_prefill_bfloat;
    // Wave-2 chunk-local causal SDPA: BF16 Q/output with full-cache f16 KV.
    id<MTLComputePipelineState> mla_sdpa_batch_f16_kv;
    // KV cache bf16→f16 conversion (used in CB1+CB2 merge to avoid CPU round-trip)
    id<MTLComputePipelineState> bf16_to_f16_row;
    // Exact scalar-compatible mHC-post chain reused by production-v2 batches.
    id<MTLComputePipelineState> f32_to_bf16_vec;
    id<MTLComputePipelineState> mhc_post_bfloat;
    id<MTLComputePipelineState> bf16_to_f32_vec;
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
// x_gpu_buf: optional id<MTLBuffer> [DIM] bf16. When x is NULL and x_gpu_buf is
// non-NULL, the function reads the attention input directly from GPU memory
// (eliminates CPU→GPU upload of normed input).
// external_cb1: optional id<MTLCommandBuffer>. When non-NULL, the function encodes
// Q/KV/SDPA into this existing command buffer instead of creating its own CB1,
// and does NOT commit or wait (caller is responsible). Requires kv_cache_gpu_buf.
// out_gpu_buf: optional id<MTLBuffer> [DIM] f32. When non-NULL and external_cb1 is set,
// wo_b output is written to this GPU buffer instead of being copied to CPU `out`.
// Caller must read from this buffer after committing external_cb1.
int mla_attention_select_bf16_identity(
    bool has_persistent_kv, bool has_external_cb, bool stage_time,
    DmlxMlaPathIdentity *out_identity);

int mla_attention_decode_bf16(MlaPipes *pipes, const AttnWeights *aw,
                              const uint16_t *x, uint16_t *kv_cache, int cache_len,
                              int pos, float *out, void *kv_cache_gpu_buf, void *x_gpu_buf,
                              void *external_cb1, void *out_gpu_buf,
                              DmlxMlaPathIdentity *out_identity,
                              MoEInferEngine *stage_engine);

// Mixed attention: uint16_t raw KV (SWA window) + f32 comp_kv (selected blocks).
int mla_attention_decode_mixed(MlaPipes *pipes, const AttnWeights *aw,
                               const float *x, uint16_t *raw_kv_cache, int raw_cache_len,
                               int pos, const float *comp_kv, int n_comp,
                               const bool *comp_allowed, float *out);

// Wave-2 explicit batch boundary. Task 2.1 defines and validates the contract;
// Tasks 2.2-2.5 implement its stages. The existing scalar and legacy prefill
// entry points remain available and unchanged during rollout.
typedef struct {
    const void *hidden;              // [batch_tokens, DIM]
    void *attention_output;          // [batch_tokens, DIM]
    uint32_t batch_tokens;
    uint32_t layer;
    const uint32_t *positions;       // absolute [batch_tokens]
    NativeKVStagingView kv;          // explicit rows owned by this layer/generation
    NativeKVTransaction *kv_transaction;
    NativeBatchWorkspace *workspace; // owned bounded arena, reset per chunk
} MLAPrefillBatchArgs;

typedef struct {
    uint64_t q_ns;
    uint64_t kv_rope_ns;
    uint64_t sdpa_ns;
    uint64_t output_ns;
    uint64_t workspace_high_water_bytes;
    uint32_t dispatches;
    // FNV-1a trace of complete-v2 per-row logical stages. Zero for standalone
    // stages; complete v2 records compressor, indexer, checkpoint, then mHC.
    uint32_t operator_order_signature;
} MLAAttentionMetrics;

// Task 2.5 explicit recurrent boundary. The caller supplies the scalar mHC-pre
// products for every row; the v2 entrypoint consumes them only after packed
// output projection and applies scalar-order compressor -> indexer -> BF16
// attention capture -> mHC-post semantics. residual_rows is updated in place
// only when the complete layer succeeds. attention_capture is materialized at
// the pre-mHC checkpoint under rollback backup; the callback is published only
// after success so an injected failure cannot leak a partial checkpoint.
// inject_failure_after_row is UINT32_MAX in normal operation and is exposed
// solely for atomic rollback tests.
typedef void (*MLAAttentionCaptureFn)(
    void *context,
    uint32_t layer,
    uint32_t position,
    const uint16_t *attention_bf16,
    uint32_t element_count);

typedef struct {
    float *residual_rows;          // [capacity_tokens, MHC_MULT, DIM], f32 BF16-valued
    const float *post_rows;        // [capacity_tokens, MHC_MULT]
    const float *comb_rows;        // [capacity_tokens, MHC_MULT, MHC_MULT]
    uint16_t *attention_capture;   // optional [capacity_tokens, DIM]
    // Optional per-row scalar-visible state metadata. When supplied, both
    // arrays must contain capacity_tokens entries; complete-v2 fills pre before
    // recurrent mutation and post after compressor/indexer/KV visibility.
    MoeInferStateSnapshot *pre_state_snapshots;
    MoeInferStateSnapshot *post_state_snapshots;
    MLAAttentionCaptureFn capture;
    void *capture_context;
    uint32_t capacity_tokens;
    uint32_t completed_tokens;
    uint32_t inject_failure_after_row;
    uint32_t reserved;
    uintptr_t self_cookie;
} MLARecurrentBatchState;

void mla_recurrent_batch_state_init(
    MLARecurrentBatchState *state,
    float *residual_rows,
    const float *post_rows,
    const float *comb_rows,
    uint16_t *attention_capture,
    uint32_t capacity_tokens);

// Exact bytes consumed by mla_attention_prefill_batch_v2 for the current
// transaction/layer/state. This includes all four stage arenas, q/sdpa handoff
// matrices, bounded CPU recurrent scratch, staged-KV rollback bytes, BF16
// output/capture rollback bytes, and deep recurrent/mHC snapshots.
int mla_attention_prefill_batch_workspace_bytes(
    const MoEInferEngine *engine,
    const MLAPrefillBatchArgs *args,
    const MLARecurrentBatchState *state,
    uint64_t *out_bytes);

// Public stage estimators share the exact checked arithmetic used by complete
// v2. region_end exposes the same checked add+alignment primitive for boundary
// validation without requiring an impractically large allocation.
int mla_attention_kv_stage_workspace_bytes(
    uint32_t rows, uint64_t *out_bytes);
int mla_attention_sdpa_stage_workspace_bytes(
    uint32_t rows, uint32_t committed, uint64_t *out_bytes);
int mla_attention_workspace_region_end(
    uint64_t offset, uint64_t size_bytes, uint64_t *out_end);

// Task 2.2 Q stage. Inputs and outputs are contiguous bfloat16 rows.
// q_output is pre-RoPE [batch_tokens, N_HEADS, HEAD_DIM]. The four GPU
// dispatches are independent of batch_tokens: wq_a, q_norm, wq_b, head_norm.
// Scratch capacity is charged to the owned bounded workspace before encoding.
int mla_attention_q_batch(
    MlaPipes *pipes,
    const AttnWeights *aw,
    const uint16_t *hidden,
    uint16_t *q_output,
    uint32_t batch_tokens,
    NativeBatchWorkspace *workspace,
    MLAAttentionMetrics *metrics);

// Task 2.3 KV/RoPE stage. Uses explicit absolute positions and writes only to
// the transaction-owned staging view; committed KV metadata is not changed.
int mla_attention_kv_rope_batch(
    MoEInferEngine *engine,
    MlaPipes *pipes,
    const AttnWeights *aw,
    const MLAPrefillBatchArgs *args,
    MLAAttentionMetrics *metrics);

// Task 2.4 SDPA stage. q_input and sdpa_output are contiguous BF16
// [batch_tokens, N_HEADS, HEAD_DIM] matrices before Q RoPE and before inverse
// output RoPE respectively. The stage reads f16 KV rows [0, start+B) from the
// authenticated engine view, keeps committed/staged KV and q_input immutable,
// and encodes exactly Q RoPE + chunk-local causal SDPA in one command buffer.
int mla_attention_sdpa_batch(
    MoEInferEngine *engine,
    MlaPipes *pipes,
    const AttnWeights *aw,
    const MLAPrefillBatchArgs *args,
    const uint16_t *q_input,
    uint16_t *sdpa_output,
    MLAAttentionMetrics *metrics);

// Task 2.5 bounded output projection. sdpa_input is pre-inverse-RoPE BF16
// [B,N_HEADS,HEAD_DIM], attention_output is BF16 [B,DIM]. The default and
// required path is packed affine-4bit grouped wo_a followed by packed wo_b.
int mla_attention_output_batch(
    MlaPipes *pipes,
    const AttnWeights *aw,
    const uint32_t *positions,
    const uint16_t *sdpa_input,
    uint16_t *attention_output,
    uint32_t batch_tokens,
    NativeBatchWorkspace *workspace,
    MLAAttentionMetrics *metrics);

// Complete Wave-2 v2 layer/chunk boundary. This composes Tasks 2.2-2.5,
// snapshots staged KV plus pointer-backed compressor/indexer and mHC state,
// and marks the KV layer staged only after every row succeeds. During a VERIFY
// transaction each successful recurrent row is also appended to the Task 5.3
// journal; outer commit can therefore expose A=0..B without retaining rejected
// compressor/indexer mutations. The caller owns the all-layer commit/abort.
int mla_attention_prefill_batch_v2(
    MoEInferEngine *engine,
    MlaPipes *pipes,
    const AttnWeights *aw,
    const MLAPrefillBatchArgs *args,
    MLARecurrentBatchState *state,
    MLAAttentionMetrics *metrics);

// Compatibility declaration retained for the staged rollout. It intentionally
// does not become a hidden scalar fallback; callers should use the explicit v2
// boundary above when recurrent state is available.
int mla_attention_prefill_batch(
    MlaPipes *pipes,
    const AttnWeights *aw,
    const MLAPrefillBatchArgs *args,
    MLAAttentionMetrics *metrics);

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
