// DSpark Speculative Decoding Engine
//
// DSpark is a block-wise speculative decoding framework (paper: "DSpark: Confidence-
// Scheduled Speculative Decoding with Semi-Autoregressive Generation").
//
// Key insight: DSpark ≠ MTP. MTP (Multi-Token Prediction) is a training objective
// from DeepSeek-V3 that predicts the next-next token with 1 extra layer.
// DSpark is a full speculative decoding framework with:
//   - A 3-layer draft backbone (parallel, generates block_size=5 tokens at once)
//   - A Markov Head (sequential correction to fix "suffix decay")
//   - A Confidence Head (per-position acceptance probability estimation)
//
// The DSpark draft backbone happens to reuse V4's architecture (MLA + MoE + mHC),
// and its weights are stored under the `mtp.*` namespace in safetensors. This is
// an implementation detail of DeepSeek's checkpoint format, not a semantic choice.
//
// Inference flow (per decode step):
//   1. Target forward (43 layers) → main_hidden = mean(hidden[40,41,42])
//   2. DSpark forward (3 layers):
//      a. main_proj(concat(main_hidden)) → main_x
//      b. embed([anchor, noise×4]) → draft initial hidden
//      c. 3× DSparkBlock(draft_hidden, main_x)
//      d. hc_head + norm + lm_head → base draft logits [5, vocab]
//   3. Markov Head: sequential correction logits[k] += B(x_{k-1})
//   4. Sample 5 draft tokens
//   5. Target batch-verify → accept first k correct + 1 bonus
//
// Module boundaries:
//   dspark_engine.h    — this file: types, config, API
//   dspark_engine.c    — forward orchestration + Markov/Confidence heads
//   dspark_attention.c — DSparkAttention (differs from target Attention)
//
// Integration with target engine (MoEInferEngine):
//   - Shares Metal device/queue (no duplicate GPU context)
//   - Shares embed/lm_head weights (DSpark reuses target's embedding + LM head)
//   - Shares expert I/O pool (DSpark experts also use packed binary pread)
//   - Does NOT share KV cache (DSpark has its own tiny window-only cache)
//
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// DSpark Configuration
// ============================================================================

#define DSPARK_N_LAYERS       3      // draft backbone layers
#define DSPARK_BLOCK_SIZE     5      // tokens drafted per step
#define DSPARK_MARKOV_RANK    256    // low-rank for Markov transition matrix
#define DSPARK_WINDOW_SIZE    128    // DSparkAttention sliding window
#define DSPARK_NOISE_TOKEN_ID 128799 // padding token for draft positions 1-4

// Target layers whose hidden states feed into DSpark
#define DSPARK_TARGET_LAYER_IDS_COUNT 3
static const int DSPARK_TARGET_LAYER_IDS[3] = {40, 41, 42};

// Expert format: MXFP4 (E2M1 4-bit packed + E8M0 scales, group_size=32)
// Same format as target model experts! Storage is uint8 (2 nibbles per byte).
#define DSPARK_EXPERT_BLOCK_SIZE 32       // quantization group size
#define DSPARK_EXPERT_SIZE       13369344 // bytes per expert
#define DSPARK_EXPERT_IN_DIM     4096     // expert input dim = DIM (gate/up input)
#define DSPARK_EXPERT_INTER_DIM  2048     // gate/up output dim (moe_inter_dim)
#define DSPARK_EXPERT_OUT_DIM    4096     // down output dim = DIM
// Expert binary layout per expert (contiguous, 4-bit packed):
//   w1.weight [INTER_DIM, IN_DIM/2]  uint8  = [2048, 2048] = 4194304 bytes
//   w1.scale  [INTER_DIM, IN_DIM/32] E8M0   = [2048, 128]  = 262144 bytes
//   w3.weight [INTER_DIM, IN_DIM/2]  uint8  = [2048, 2048] = 4194304 bytes
//   w3.scale  [INTER_DIM, IN_DIM/32] E8M0   = [2048, 128]  = 262144 bytes
//   w2.weight [OUT_DIM, INTER_DIM/2] uint8  = [4096, 1024] = 4194304 bytes
//   w2.scale  [OUT_DIM, INTER_DIM/32] E8M0  = [4096, 64]   = 262144 bytes
//   Total = 3 * (4194304 + 262144) = 13369344 ✓

// Reuse target engine constants where identical
// DIM=4096, INTERMEDIATE=2048, N_EXPERTS=256, N_ACTIVE=6, HEAD_DIM=512,
// N_HEADS=64, Q_LORA_RANK=1024, KV_LORA_RANK=512, O_GROUPS=8, O_LORA_RANK=1024,
// MHC_MULT=4 — all from engine.h

// ============================================================================
// DSpark Weight Structures
// ============================================================================

// FP8 quantized weight (E4M3 weight + E8M0 block scale, block_size=128)
// Used for DSpark attention projections (same format as target's attention weights in FP8 model)
typedef struct {
    const uint8_t *weight;    // [out_dim, in_dim] uint8 (FP8 E4M3 raw bytes)
    const uint8_t *scale;     // [ceil(out_dim/128), ceil(in_dim/128)] uint8 (E8M0)
    int out_dim;
    int in_dim;
    int block_size;           // 128 for attention FP8
} DSparkFP8Weight;

// Per-layer DSparkAttention weights
// Same structure as target MLA, but no compressor/indexer (dense window only)
typedef struct {
    DSparkFP8Weight wq_a;     // [Q_LORA_RANK, DIM] = [1024, 4096]
    const float *q_norm;      // [Q_LORA_RANK] = [1024]
    DSparkFP8Weight wq_b;     // [N_HEADS*HEAD_DIM, Q_LORA_RANK] = [32768, 1024]
    DSparkFP8Weight wkv;      // [KV_LORA_RANK, DIM] = [512, 4096]
    const float *kv_norm;     // [KV_LORA_RANK] = [512]
    DSparkFP8Weight wo_a;     // [N_HEADS*HEAD_DIM/O_GROUPS * O_GROUPS, DIM] = [8192, 4096]
    DSparkFP8Weight wo_b;     // [DIM, O_GROUPS*O_LORA_RANK] = [4096, 8192]
    const float *attn_sink;   // [N_HEADS] = [64]
    // Pre-dequantized f32 weights for fast cblas_sgemv dispatch
    float *wq_a_f32;          // [Q_LORA_RANK * DIM] = [1024 * 4096]
    float *wq_b_f32;          // [N_HEADS*HEAD_DIM * Q_LORA_RANK] = [32768 * 1024]
    float *wkv_f32;           // [KV_LORA_RANK * DIM] = [512 * 4096]
    float *wo_a_f32;          // [O_GROUPS*O_LORA_RANK * (N_HEADS/O_GROUPS)*HEAD_DIM] = [8192 * 4096]
    float *wo_b_f32;          // [DIM * O_GROUPS*O_LORA_RANK] = [4096 * 8192]
} DSparkAttnWeights;

// Per-layer DSpark block weights (complete layer)
typedef struct {
    DSparkAttnWeights attn;
    const float *attn_norm;   // [DIM] pre-attention RMSNorm
    const float *ffn_norm;    // [DIM] pre-FFN RMSNorm
    const float *gate_weight; // [N_EXPERTS, DIM] routing gate (f32, converted from BF16)
    const float *gate_bias;   // [N_EXPERTS] score correction bias (f32)
    // mHC weights (same as target)
    const float *hc_attn_fn;    // [24, MHC_MULT*DIM]
    const float *hc_attn_base;  // [24]
    const float *hc_attn_scale; // [3]
    const float *hc_ffn_fn;     // [24, MHC_MULT*DIM]
    const float *hc_ffn_base;   // [24]
    const float *hc_ffn_scale;  // [3]
} DSparkLayerWeights;

// DSpark-specific weights (beyond the 3 backbone layers)
typedef struct {
    // main_proj: projects target hidden states into DSpark input space
    // input = concat(target_hidden[40], target_hidden[41], target_hidden[42]) ∈ R^{3*DIM}
    // output = main_proj(input) ∈ R^{DIM}
    DSparkFP8Weight main_proj;   // [DIM, 3*DIM] = [4096, 12288]
    float *main_proj_f32;        // pre-dequantized [DIM * 3*DIM] f32
    const float *main_norm;      // [DIM] RMSNorm after main_proj

    // Markov Head (only on last DSpark layer output)
    // B(x_{k-1}, v) = dot(W1[x_{k-1}], W2[v]) — low-rank transition bias
    const float *markov_w1;      // [vocab_size, DSPARK_MARKOV_RANK] f32
    const float *markov_w2;      // [vocab_size, DSPARK_MARKOV_RANK] f32

    // Confidence Head: sigmoid(proj @ [hidden; markov_embed])
    const float *confidence_proj; // [1, DIM + DSPARK_MARKOV_RANK] = [1, 4352] f32

    // Final output projection (after last DSpark layer)
    const float *hc_head_fn;     // [MHC_MULT, MHC_MULT*DIM] = [4, 16384]
    const float *hc_head_base;   // [MHC_MULT] = [4]
    const float *hc_head_scale;  // [1]
    const float *final_norm;     // [DIM] RMSNorm before lm_head
} DSparkHeadWeights;

// ============================================================================
// DSpark KV Cache (per DSparkAttention layer)
// ============================================================================

// DSparkAttention uses a different KV strategy than target:
//   - main_kv: sliding window of target model's KV (copied from target's forward)
//   - draft_kv: block_size entries from draft tokens (ephemeral, reset each step)
// Total KV seen by SDPA = [window_entries + block_size, KV_LORA_RANK]
typedef struct {
    uint16_t *main_kv;    // [DSPARK_WINDOW_SIZE, KV_LORA_RANK] f16 — from target
    uint16_t *draft_kv;   // [DSPARK_BLOCK_SIZE, KV_LORA_RANK] f16 — from draft tokens
    int main_len;         // current entries in main window (0..DSPARK_WINDOW_SIZE)
    int draft_len;        // current draft entries (0..DSPARK_BLOCK_SIZE)
} DSparkKVCache;

// ============================================================================
// DSpark Engine State
// ============================================================================

typedef struct {
    // --- Metal context (shared with target engine, NOT owned) ---
    void *device;     // id<MTLDevice> — borrowed from MoEInferEngine
    void *queue;      // id<MTLCommandQueue> — borrowed

    // --- DSpark-specific Metal pipelines ---
    void *pipe_dequant_int8_e8m0;        // dequant_matvec_int8_e8m0
    void *pipe_fused_gate_up_int8_e8m0;  // fused_gate_up_swiglu_int8_e8m0
    // Reuse target engine pipelines for FP8 attention matmuls:
    // pipe_dequant_matvec_affine (for FP8→f32 with E8M0 scale)
    // pipe_rms_norm_rows, pipe_rope_tail, pipe_mla_sdpa_f16in_f16out, etc.

    // --- Weights ---
    DSparkLayerWeights layers[DSPARK_N_LAYERS];
    DSparkHeadWeights head;

    // --- Expert I/O (INT8 packed binary, same pool as target) ---
    int expert_fd[DSPARK_N_LAYERS];       // per-layer packed expert file descriptors
    uint8_t *expert_buf[6];               // reuse target's 2MB-aligned buffers
    uint8_t *expert_mmap[DSPARK_N_LAYERS]; // mmap'd full expert files (preloaded)
    size_t expert_mmap_size;               // size of each mmap'd file

    // --- KV Cache (one per DSpark layer) ---
    DSparkKVCache kv_cache[DSPARK_N_LAYERS];

    // --- Scratch buffers ---
    float *buf_main_hidden;       // [3 * DIM] target hidden concat buffer
    float *buf_main_x;            // [DIM] after main_proj + norm
    float *buf_draft_hidden;      // [DSPARK_BLOCK_SIZE * MHC_MULT * DIM] draft residuals
    float *buf_draft_logits;      // [DSPARK_BLOCK_SIZE * vocab_size] output logits
    float *buf_confidence;        // [DSPARK_BLOCK_SIZE] confidence scores

    // --- Config ---
    int block_size;               // default 5
    int markov_rank;              // default 256
    int vocab_size;               // 129280 (from target)
    int noise_token_id;           // 128799

    // --- Reference to target engine (for shared resources) ---
    void *target_engine;          // MoEInferEngine* — for embed, lm_head, expert I/O pool

    bool initialized;
} DSparkEngine;

// ============================================================================
// DSpark API
// ============================================================================

// --- Lifecycle ---

// Initialize DSpark engine. Borrows Metal context and shared weights from target.
// dspark_weight_dir: path to extracted DSpark weights (dspark_weights/ directory)
// packed_expert_dir: path to packed MTP experts (packed_mtp_experts/ directory)
// target: pointer to initialized MoEInferEngine (for device/queue/embed/lm_head)
// Returns NULL on failure.
DSparkEngine *dspark_init(
    const char *dspark_weight_dir,
    const char *packed_expert_dir,
    void *target_engine  // MoEInferEngine*
);

// Free all DSpark resources (does NOT free target engine).
void dspark_deinit(DSparkEngine *eng);

// --- Forward pass ---

// DSpark draft forward: produces block_size draft logits from target hidden states.
//
// Inputs:
//   main_hidden: [3 * DIM] f32 — concatenated mean-pooled hidden from target layers 40/41/42
//   anchor_token_id: the last token produced by target (serves as draft position 0 input)
//   start_pos: current sequence position in target (for RoPE in DSparkAttention)
//
// Outputs:
//   draft_logits: [block_size, vocab_size] f32 — base logits from DSpark backbone
//   confidence:   [block_size] f32 — per-position confidence (0-1), or NULL to skip
//
// Returns number of draft positions computed (normally = block_size = 5).
int dspark_forward(
    DSparkEngine *eng,
    const float *main_hidden,
    int anchor_token_id,
    int start_pos,
    float *draft_logits,
    float *confidence
);

// --- Markov Head ---

// Apply Markov Head sequential correction to draft logits and sample tokens.
// This is the "semi-autoregressive" part: each position's logits are corrected
// based on the previous position's sampled token.
//
// Inputs:
//   draft_logits: [block_size, vocab_size] — from dspark_forward()
//   anchor_token_id: token preceding the first draft position
//
// Outputs:
//   corrected_logits: [block_size, vocab_size] — logits after Markov bias (may alias draft_logits)
//   draft_tokens: [block_size] — sampled token IDs (greedy argmax)
//
// Returns number of tokens proposed (= block_size).
int dspark_markov_sample(
    DSparkEngine *eng,
    float *draft_logits,
    int anchor_token_id,
    float *corrected_logits,
    uint32_t *draft_tokens
);

// --- KV Cache management ---

// Update DSpark's main_kv window from target's latest KV cache entry.
// Called after each target decode step to keep DSpark's window in sync.
// target_kv_entry: [KV_LORA_RANK] f16 — the KV cache row target just wrote at `pos`.
void dspark_update_main_kv(DSparkEngine *eng, const uint16_t *target_kv_entry, int pos);

// Reset DSpark state (call at start of each new sequence).
void dspark_reset(DSparkEngine *eng);

// --- Target integration helpers ---

// Extract and accumulate target hidden state for DSpark consumption.
// Called from target engine's forward loop after layers 40, 41, 42 complete.
// hidden: [MHC_MULT, DIM] f32 — current layer's residual (before next layer)
// layer_idx: which target layer just completed (40, 41, or 42)
// Internally does: mean over MHC_MULT dimension → store in buf_main_hidden[offset..].
void dspark_accumulate_target_hidden(DSparkEngine *eng, const float *hidden, int layer_idx);

#ifdef __cplusplus
}
#endif
