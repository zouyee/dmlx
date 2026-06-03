// Metal inference engine — flash-moe style per-token pipeline.
// Handles MoE layers via 3-command-buffer pipelining with deferred CMD3.
// Attention and RMSNorm use CPU (Accelerate framework for matmuls).
//
// Architecture (per layer):
//   CMD1: attention projections (q/k/v proj matvecs)
//   CPU:  RoPE, KV cache update, SDPA
//   CMD2: o_proj + residual_add + rms_norm + routing + shared expert
//   CPU:  softmax + topK → expert indices
//   I/O:  parallel pread K=6 experts
//   CMD3: expert forward (gate/up/SwiGLU/down) + moe_combine + rms_norm [DEFERRED]
//
// Key data structures and flow adapted from flash-moe infer.m.
#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Configuration
// ============================================================================

#define DIM 4096           // hidden dimension
#define INTERMEDIATE 2048  // gate/up intermediate (MoE expert)
#define N_LAYERS 43        // transformer layers
#define N_EXPERTS 256      // routed experts per layer
#define N_ACTIVE 6         // K = 6 active experts per token

// --- MLA (Multi-head Latent Attention) dimensions — DeepSeek-V4-Flash ---
// Verified from safetensors headers (see dsv4-first-class-support-plan.md §Phase 3-5).
#define N_HEADS 64         // attention heads
#define HEAD_DIM 512       // per-head dim = QK_NOPE_DIM + QK_ROPE_DIM
#define QK_ROPE_DIM 64     // rotary portion (tail), YaRN partial RoPE
#define QK_NOPE_DIM 448    // non-rotary portion (head front)
#define Q_LORA_RANK 1024   // wq_a output / wq_b input
#define KV_LORA_RANK 512   // wkv output (MQA: 1 KV head, broadcast to N_HEADS)
#define O_GROUPS 8         // grouped output LoRA (wo_a)
#define O_LORA_RANK 1024   // per-group wo_a output
#define ATTN_GROUP_SIZE 64 // affine quant group size for attention weights
#define MOE_GROUP_SIZE 32  // mxfp4 quant group size for expert weights
#define MHC_MULT 4         // mHC HyperConnection multiplier
#define MAX_SEQ_LEN 4096   // max sequence length

// Quantized weight (affine 4-bit, gs=64): packed u32 + bf16->f32 scales & biases.
// w = scale_g * nibble + bias_g.  Pointers are owned by the model (eval'd, not freed).
typedef struct {
    const uint32_t *packed;  // [out_dim, in_dim/8]
    const float    *scales;  // [out_dim, in_dim/group_size] (converted bf16->f32)
    const float    *biases;  // [out_dim, in_dim/group_size]
    int out_dim;
    int in_dim;
    int group_size;
} QuantWeight;

// Shared expert weights (affine 4bit, gs=64). One per layer.
typedef struct {
    QuantWeight gate;  // [2048, 4096]
    QuantWeight up;    // [2048, 4096]
    QuantWeight down;  // [4096, 2048]
} SharedExpert;

// Per-layer MLA attention weights (quantized, on-the-fly dequant in Metal).
typedef struct {
    QuantWeight wq_a;        // [1024, 4096]
    const float *q_norm;     // [1024] RMSNorm weight
    QuantWeight wq_b;        // [32768, 1024]  (= N_HEADS*HEAD_DIM, Q_LORA_RANK)
    QuantWeight wkv;         // [512, 4096]
    const float *kv_norm;    // [512] RMSNorm weight
    // wo_a is DENSE f32 (loader dequantizes it): [O_GROUPS, O_LORA_RANK, group_feat]
    // flattened, group_feat = (N_HEADS/O_GROUPS)*HEAD_DIM = 4096.
    const float *wo_a_dense; // [O_GROUPS * O_LORA_RANK * 4096]
    QuantWeight wo_b;        // [4096, 8192]  (DIM, O_GROUPS*O_LORA_RANK)
    const float *attn_sink;  // [64] per-head sink logits
} AttnWeights;


// Expert packed binary layout (see repack_experts.py)
#define EXPERT_SIZE 13369344  // bytes per expert
#define GATE_W_OFF  0
#define GATE_S_OFF  4194304
#define UP_W_OFF    4456448
#define UP_S_OFF    8650752
#define DOWN_W_OFF  8912896
#define DOWN_S_OFF  13107200

// ============================================================================
// Layer config — which layers use full attention vs linear attention
// ============================================================================

typedef struct {
    bool is_full_attn;   // true = standard attention, false = linear (GatedDeltaNet)
    int input_norm_idx;  // index into input_norm_weights array
    int attn_norm_idx;   // index into attn_norm_weights array
} LayerConfig;

// ============================================================================
// Weight pointers (set by moe_infer_set_weights from MLX)
// ============================================================================

typedef struct {
    const float *embed;             // [vocab, DIM]
    int vocab_size;
    const float *lm_head;           // [vocab, DIM]
    const float *final_norm;        // [DIM]
    const float *input_norms[N_LAYERS];  // [DIM] attn_norm (pre-attention)
    const float *attn_norms[N_LAYERS];   // [DIM] ffn_norm (pre-MoE)
    AttnWeights  attn[N_LAYERS];         // MLA attention weights (quantized)
    const float *gate_proj[N_LAYERS];    // [N_EXPERTS, DIM] router weight
    int expert_fd[N_LAYERS];
    bool weights_set;
} WeightFile;

// ============================================================================
// KV Cache — MLA stores compressed KV-latent (1 KV head, KV_LORA_RANK wide)
// ============================================================================

typedef struct {
    float *kv;  // [MAX_SEQ_LEN, KV_LORA_RANK] post-norm, post-RoPE KV latent
    int len;
} KVCache;

// ============================================================================
// Temporal expert prediction
// ============================================================================

typedef struct {
    int experts[N_LAYERS][N_ACTIVE];  // previous token's experts per layer
    bool valid;                        // true after first token
    int hits;
    int misses;
} ExpertPredictor;

// ============================================================================
// Deferred CMD3 state
// ============================================================================

typedef struct {
    bool active;            // whether deferred work is pending
    bool gpu_combined;      // GPU already did combine+residual+norm
    void *cmd_experts;      // id<MTLCommandBuffer> — submitted but NOT waited
    int actual_K;
    float h_mid[DIM];       // saved for CPU-side combine (fallback)
    float *hidden;          // pointer to output buffer
    int layer_idx;
    float expert_weights[N_ACTIVE];
    int valid[N_ACTIVE];
} DeferredExpertState;

// ============================================================================
// Metal inference engine
// ============================================================================

typedef struct {
    // Metal objects
    void *device;     // id<MTLDevice>
    void *queue;      // id<MTLCommandQueue>

    // Pipeline states (id<MTLComputePipelineState>)
    void *pipe_gate_up_swiglu;
    void *pipe_dequant_matvec;
    void *pipe_moe_combine;
    void *pipe_rms_norm_sum_sq;
    void *pipe_rms_norm_apply;
    void *pipe_matvec;
    // S7: MLA attention pipelines
    void *pipe_dequant_matvec_affine;
    void *pipe_rms_norm_rows;
    void *pipe_rope_tail;
    void *pipe_mla_sdpa;
    // bf16-output variants for Q chain alignment with MLX bf16 precision
    void *pipe_dequant_matvec_affine_bf16;
    void *pipe_rms_norm_rows_bf16;
    void *pipe_bf16_to_f32;
    // mhc_pre_gpu: full mhc_pre on GPU with bfloat out_input (matches MLX .astype(bf16))
    void *pipe_mhc_pre_gpu;
    // Full bf16 chain kernels (bfloat input + bfloat output throughout)
    void *pipe_f32_to_bf16;
    void *pipe_dequant_matvec_affine_bf16in_f32out;
    void *pipe_dequant_matvec_affine_bf16in_bf16out;
    void *pipe_rms_norm_rows_bf16in_bf16out;
    void *pipe_rope_tail_bf16;
    void *pipe_matvec_f32_bf16in;

    // Buffers (id<MTLBuffer>)
    void *buf_hidden;            // [DIM] current hidden state
    void *buf_h_mid;             // [DIM] after attention + residual
    void *buf_normed;            // [DIM] after RMSNorm
    void *buf_attn_out;          // [DIM] o_proj output
    void *buf_routing_scores;    // [N_EXPERTS] gate logits
    void *buf_expert_mid[6];     // [INTERMEDIATE] gate+up output per expert
    void *buf_expert_out[6];     // [DIM] down output per expert
    void *buf_shared_gate;       // [INTERMEDIATE] shared expert gate
    void *buf_shared_up;         // [INTERMEDIATE] shared expert up
    void *buf_shared_down;       // [DIM] shared expert down
    void *buf_norm_sum_sq;       // [1] sum of squares for RMS
    void *buf_input_norm;        // [DIM] next layer input norm weight (GPU)

    // Expert I/O
    int packed_fd[N_LAYERS];     // per-layer packed expert file descriptors
    uint8_t *expert_buf[6];      // 2MB-aligned expert data buffers
    uint8_t *expert_buf_pred[6]; // second buffer set for prediction prefetch
    void *io_pool;               // persistent I/O thread pool

    // Deferred CMD3
    DeferredExpertState deferred;

    // Prediction
    ExpertPredictor predictor;

    // Weights (set via moe_infer_set_weights)
    const float *embed;
    int vocab_size;
    const float *lm_head;
    const float *final_norm;
    const float *input_norms[N_LAYERS];  // attn_norm (pre-attention)
    const float *attn_norms[N_LAYERS];   // ffn_norm (pre-MoE)
    AttnWeights  attn[N_LAYERS];         // MLA attention weights (quantized)
    SharedExpert shared[N_LAYERS];       // shared expert (affine 4bit, gs=64)
    const float *gate_proj[N_LAYERS];    // [N_EXPERTS, DIM] router weight
    const float *gate_bias[N_LAYERS];    // [N_EXPERTS] e_score_correction_bias (NULL if absent)
    const int64_t *tid2eid[N_LAYERS];    // [vocab_size, N_ACTIVE] hash routing table (NULL = score-based)
    // mHC weights per layer (f32): fn [24,16384], base [24], scale [3]
    const float *attn_hc_fn[N_LAYERS];
    const float *attn_hc_base[N_LAYERS];
    const float *attn_hc_scale[N_LAYERS];
    const float *ffn_hc_fn[N_LAYERS];
    const float *ffn_hc_base[N_LAYERS];
    const float *ffn_hc_scale[N_LAYERS];
    int expert_fd[N_LAYERS];

    // KV cache
    KVCache kv_cache[N_LAYERS];

    // Layer config
    LayerConfig layers[N_LAYERS];

    // State
    int current_pos;       // current sequence position
    int current_token_id;  // current input token ID (needed for hash routing in layers 0-2)
    bool initialized;
} MoEInferEngine;

// ============================================================================
// API
// ============================================================================

// Initialize engine: Metal setup, open expert files, allocate buffers.
// Returns engine pointer on success, NULL on error.
MoEInferEngine *moe_infer_init(const char *packed_dir,
                                const char *kernel_src, unsigned long kernel_src_len);

// Set global backbone weights (embed, lm_head, final_norm, per-layer norms, gate).
// Pointers must remain valid for the engine's lifetime (eval'd, not freed).
void moe_infer_set_weights(MoEInferEngine *engine,
    const float *embed, int vocab_size,
    const float *lm_head,
    const float *final_norm,
    const float **input_norms,     // [N_LAYERS] -> [DIM] (attn_norm)
    const float **attn_norms,      // [N_LAYERS] -> [DIM] (ffn_norm)
    const float **gate_projs,      // [N_LAYERS] -> [N_EXPERTS, DIM]
    const float **gate_biases);    // [N_LAYERS] -> [N_EXPERTS] (or NULL per layer)

// Set one layer's MLA attention weights (quantized; on-the-fly dequant in Metal).
// Called once per layer after moe_infer_set_weights. The AttnWeights pointers
// (packed u32, f32 scales/biases, norms, sink) must remain valid for the
// engine's lifetime.
void moe_infer_set_layer_attn(MoEInferEngine *engine, int layer, AttnWeights attn);

// Set one layer's shared expert weights.
void moe_infer_set_layer_shared(MoEInferEngine *engine, int layer, SharedExpert se);

// Reset KV cache (call at the start of each new sequence/request).
void moe_infer_reset_kv(MoEInferEngine *engine);

// Set one layer's mHC weights (f32 pointers, kept alive by caller).
void moe_infer_set_layer_hc(MoEInferEngine *engine, int layer,
    const float *attn_fn, const float *attn_base, const float *attn_scale,
    const float *ffn_fn, const float *ffn_base, const float *ffn_scale);

// Set hash routing table for a layer (layers 0-2 in this model).
// tid2eid: [vocab_size, N_ACTIVE] int64 — for each token, the N_ACTIVE expert IDs.
void moe_infer_set_layer_tid2eid(MoEInferEngine *engine, int layer,
                                  const int64_t *tid2eid);

// Set the current token ID (call before each forwardLayer for hash routing).
void moe_infer_set_token_id(MoEInferEngine *engine, int token_id);

// Process one layer: RMSNorm → MLA attention → routing → MoE → output.
// hidden: [DIM] input, overwritten with output on return.
// Returns 0 on success.
int moe_infer_forward_layer(MoEInferEngine *engine, int layer, float *hidden, int pos);

// Forward pass for ALL layers. hidden: [DIM] input and output.
int moe_infer_forward(MoEInferEngine *engine, float *hidden, int pos);

// Cleanup
void moe_infer_deinit(MoEInferEngine *engine);

#ifdef __cplusplus
}
#endif
