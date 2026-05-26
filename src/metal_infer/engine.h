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
#define INTERMEDIATE 2048  // gate/up intermediate
#define N_LAYERS 43        // MoE layers (layer 0..42)
#define N_EXPERTS 256      // experts per layer
#define N_ACTIVE 6         // K = 6 active experts per token
#define N_HEADS 32         // attention heads (full-attn layers)
#define HEAD_DIM 128       // head dimension = DIM / N_HEADS
#define KV_HEADS 8         // KV heads (MLA compressed)
#define KV_LORA_RANK 512   // KV compression rank
#define MAX_SEQ_LEN 4096   // max sequence length

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
// Weight pointers (loaded from safetensors / packed experts)
// ============================================================================

typedef struct {
    // Backbone weights (CPU memory, mmap'd)
    float *embed;                   // [vocab, DIM]
    float *lm_head;                 // [vocab, DIM]
    float *final_norm;              // [DIM]

    // Per-layer attention weights
    float *q_proj[N_LAYERS];        // [DIM, DIM]
    float *k_proj[N_LAYERS];        // [DIM, KV_LORA_RANK]
    float *v_proj[N_LAYERS];        // [DIM, KV_LORA_RANK]
    float *o_proj[N_LAYERS];        // [DIM, DIM]
    float *q_norm[N_LAYERS];        // [HEAD_DIM]
    float *k_norm[N_LAYERS];        // [HEAD_DIM]

    // Per-layer norm weights
    float *input_norm[N_LAYERS];    // [DIM]
    float *attn_norm[N_LAYERS];     // [DIM]

    // Per-layer shared expert
    float *shared_gate[N_LAYERS];   // [INTERMEDIATE, DIM]
    float *shared_up[N_LAYERS];     // [INTERMEDIATE, DIM]
    float *shared_down[N_LAYERS];   // [DIM, INTERMEDIATE]
    float *shared_gate_s[N_LAYERS]; // [INTERMEDIATE, DIM/group]
    float *shared_up_s[N_LAYERS];   // [INTERMEDIATE, DIM/group]
    float *shared_down_s[N_LAYERS]; // [DIM, INTERMEDIATE/group]

    // Expert files — fd per layer (packed .bin files)
    int expert_fd[N_LAYERS];
} WeightFile;

// ============================================================================
// KV Cache
// ============================================================================

typedef struct {
    float *k;   // [MAX_SEQ_LEN, KV_LORA_RANK]
    float *v;   // [MAX_SEQ_LEN, KV_LORA_RANK]
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
    void *pipe_residual_add;
    void *pipe_matvec;           // generic float matvec for attention proj

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

    // Weights
    WeightFile wf;

    // KV cache
    KVCache kv_cache[N_LAYERS];

    // Layer config
    LayerConfig layers[N_LAYERS];

    // State
    int current_pos;       // current sequence position
    bool initialized;
} MoEInferEngine;

// ============================================================================
// API
// ============================================================================

// Initialize engine: load model weights, setup Metal, open expert files.
// Returns 0 on success, -1 on error.
int moe_infer_init(MoEInferEngine *engine, const char *model_path,
                   const char *packed_dir,
                   const char *kernel_src, unsigned long kernel_src_len);

// Forward pass for one token. Writes next-token logits into `logits` [vocab_size].
// Returns 0 on success, -1 on error.
int moe_infer_forward(MoEInferEngine *engine, int token, float *logits);

// Cleanup
void moe_infer_deinit(MoEInferEngine *engine);

#ifdef __cplusplus
}
#endif
