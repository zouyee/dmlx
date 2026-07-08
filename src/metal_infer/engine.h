// Metal inference engine — flash-moe style per-token pipeline.
// Handles MoE layers via 3-command-buffer pipelining with deferred CMD3.
// Attention and RMSNorm use CPU (Accelerate framework for matmuls).
//
// Architecture (per layer):
//   CMD1: attention projections (q/k/v proj matvecs)
#include <stddef.h>
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

// Compressor configuration constants
#define COMP_HEAD_DIM    512   // main compressor output head_dim = HEAD_DIM
#define IDX_HEAD_DIM     128   // indexer compressor output head_dim
#define CSA_OUT_DIM     1024   // ratio=4 CSA: out_dim = head_dim * 2
#define HCA_OUT_DIM      512   // ratio=128 HCA: out_dim = head_dim
#define MAX_COMP_BLOCKS 8192   // max compressed blocks per layer per sequence
#define SWA_WINDOW       128   // sliding window size (sliding_window in config)

// Per-layer streaming compressor runtime state.
typedef struct {
    // Main compressor rolling state [2*ratio, out_dim]
    float *state_kv;    // [2*ratio, out_dim]  (ratio=4: [8,1024], ratio=128: [128,512])
    float *state_score; // same shape
    uint32_t ratio;     // compress_ratio for this layer (0 = not initialized)
    uint32_t out_dim;   // 1024 for CSA, 512 for HCA
    // Main compressor output blocks
    float *comp_kv;     // [MAX_COMP_BLOCKS, COMP_HEAD_DIM] f32
    uint32_t n_comp;    // current number of emitted comp blocks
    // Indexer's own compressor state (only ratio=4 layers)
    float *idx_state_kv;    // [8, 256] (ratio=4, out_dim=128*2=256)
    float *idx_state_score; // same
    float *idx_comp_kv;     // [MAX_COMP_BLOCKS, IDX_HEAD_DIM]
    uint32_t n_idx_comp;
} CompressorState;

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

// Q8_0 quantization block (GGUF format, used by ds4 kernel_mul_mv_q8_0_f32).
// 32 elements quantized to int8 with a shared float scale.
// d = max(|x_i|) / 127.0, qs[i] = round(x_i / d) clamped to [-127, 127].
typedef struct {
    float d;       // block scale
    int8_t qs[32]; // quantized values
} Q8_0Block;

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


// Expert packed binary layout (MXFP4, group_size=32)
// Uses formula: LUT[nibble] * exp2(scale - 127)
// Scales stored as uint8 (E8M0), weights as uint32
// Layout per expert: [GATE_W: 4MB][GATE_S: 256KB][UP_W: 4MB][UP_S: 256KB][DOWN_W: 4MB][DOWN_S: 256KB]
// Bias planes (GATE_B/UP_B/DOWN_B) are reserved but not present in the actual files.
#define EXPERT_SIZE 13369344  // bytes per expert (~12.75 MB)
#define GATE_W_OFF  0
#define GATE_S_OFF  4194304
#define UP_W_OFF    4456448
#define UP_S_OFF    8650752
#define DOWN_W_OFF  8912896
#define DOWN_S_OFF  13107200

// Affine v2 layout: proper dequant→requantize, bf16 scale+bias, gs=64
// [gate: W(4MB) S(256KB) B(256KB)] [up: ...] [down: W(4MB) S(256KB) B(256KB)]
#define AFFINE_EXPERT_SIZE  14155776  // 3 * (4194304 + 262144 + 262144)
#define AFFINE_GATE_W_OFF   0
#define AFFINE_GATE_S_OFF   4194304
#define AFFINE_GATE_B_OFF   4456448
#define AFFINE_UP_W_OFF     4718592
#define AFFINE_UP_S_OFF     8912896
#define AFFINE_UP_B_OFF     9175040
#define AFFINE_DOWN_W_OFF   9437184
#define AFFINE_DOWN_S_OFF   13631488
#define AFFINE_DOWN_B_OFF   13893632

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
    uint16_t *kv;   // [MAX_SEQ_LEN, KV_LORA_RANK] post-norm, post-RoPE KV latent
                    // Points to [kv_gpu_buf contents] when kv_gpu_buf is set.
                    // Format: half-precision (f16, NOT bf16) for SDPA compatibility.
    void *kv_gpu_buf; // id<MTLBuffer> (Shared mode) wrapping kv — GPU-accessible.
                      // NULL until first decode call allocates it.
                      // Enables blit-based KV update (eliminates CPU round-trip between CB1 and CB2).
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
    void *pipe_gate_up_swiglu_v2;      // ds4 no-x_shared coalesced pattern (MXFP4, gs=32)
    void *pipe_gate_up_swiglu_v4;      // MLX-style: 64 threads, 0 TGMEM, 4 rows/SIMD (new)
    void *pipe_gather_gate_up_v4;      // MLX-style gather version
    void *pipe_gate_up_swiglu_v2_affine; // affine 4-bit dequant (bf16 scales+biases, gs=64) — experimental
    void *pipe_dequant_matvec_4bit_affine; // affine 4-bit down_proj
    void *pipe_dequant_matvec;
    void *pipe_moe_combine;
    // Fused 6-expert kernels: process K=6 experts in one dispatch
    void *pipe_fused_6expert_gate_up;   // fused_6expert_gate_up_swiglu
    void *pipe_fused_6expert_down;      // fused_6expert_down
    // Gather MoE kernels: gatherQmm equivalent — K experts from full N_EXPERTS buffer
    void *pipe_gather_gate_up;          // gather_gate_up_swiglu
    void *pipe_gather_down;             // gather_down
    void *pipe_rms_norm_sum_sq;
    void *pipe_rms_norm_apply;
    void *pipe_matvec;
    // S7: MLA attention pipelines (f32 baseline)
    void *pipe_dequant_matvec_affine;
    void *pipe_dequant_matvec_affine_v2;   // coalesced ds4 pattern (NR0=2, NSG=4)
    void *pipe_rms_norm_rows;
    void *pipe_rope_tail;
    void *pipe_mla_sdpa;
    void *pipe_mla_sdpa_f16;     // KV cache f16 precision (ds4 path)
    // S8: Full f16 precision chain (ds4-style end-to-end)
    void *pipe_dequant_matvec_affine_f16out;
    void *pipe_rms_norm_rows_f16out;
    void *pipe_dequant_matvec_affine_f16in_f16out;
    void *pipe_rms_norm_rows_f16in_f16out;
    void *pipe_rope_tail_f16;
    void *pipe_matvec_f32_f16in;
    void *pipe_mla_sdpa_f16in_f16out;
    void *pipe_mhc_pre_f16;
    void *pipe_mhc_post_f16;
    // MoE f16 precision chain
    void *pipe_gate_up_swiglu_f32in_f16out;
    void *pipe_dequant_matvec_4bit_f16in_f32out;

    // BF16 precision chain (S8b — MLX alignment)
    void *pipe_dequant_matvec_affine_bf16out;
    void *pipe_rms_norm_rows_bf16out;
    void *pipe_dequant_matvec_affine_bf16in_bf16out;
    void *pipe_dequant_matvec_affine_bf16in_bf16out_v2;  // SIMD-parallel v2
    void *pipe_rms_norm_rows_bf16in_bf16out;
    void *pipe_rope_tail_bf16;
    void *pipe_matvec_f32_bf16in;
    void *pipe_matvec_f32_bf16in_simd;  // SIMD-parallel routing gate matmul
    void *pipe_matvec_q8_0_f32;        // ds4 Q8_0 matvec for wo_a
    void *pipe_mla_sdpa_bfloat;
    void *pipe_mhc_pre_bfloat;
    void *pipe_mhc_post_bfloat;
    void *pipe_mhc_post_ffn_expand4; // ds4 kernel_dsv4_hc_expand4: single f32 dispatch mhc_post_ffn
    void *pipe_mhc_pre_split_weighted_sum_norm; // ds4 fused mhc_pre + weighted sum + RMSNorm
    void *pipe_f32_to_bf16;
    void *pipe_bf16_to_f32;
    void *pipe_dequant_matvec_affine_bf16in_f32out;
    void *pipe_mla_sdpa_prefill_bfloat; // batch prefill SDPA (Path B)
    void *pipe_bf16_to_f16_row;         // KV cache bf16→f16 conversion
    void *pipe_limited_swiglu;          // in-place SwiGLU for shared expert
    void *pipe_f32_to_bf16_vec;         // residual f32→bf16 on GPU
    void *pipe_bf16_to_f32_vec;         // residual bf16→f32 writeback
    void *pipe_moe_route_gpu;           // GPU top-K routing (sqrtsoftplus+bitonic+normalize)
    // GPU-resident residual buffer: [MHC_MULT * DIM] f32, Shared mode.
    // Path B: eliminate all CPU↔GPU residual transfers (5 memcpy/layer, 3 GPU syncs).
    // In steady state (Step 2+), residual never leaves GPU between layers.
    void *buf_residual_gpu;              // [MHC_MULT*DIM] f32 — GPU-resident residual
    // GPU routing result buffers (written by CMD2, read by CPU after CB1 wait)
    void *buf_routing_scores_f32;        // [N_EXPERTS] f32 — sqrtsoftplus scores
    void *buf_routing_selected;          // [N_ACTIVE] int32 — top-6 indices
    void *buf_routing_weights_gpu;       // [N_ACTIVE] f32  — normalized weights

    // Buffers (id<MTLBuffer>)
    void *buf_hidden;            // [DIM] current hidden state
    void *buf_h_mid;             // [DIM] after attention + residual
    void *buf_normed;            // [DIM] after RMSNorm
    void *buf_attn_out;          // [DIM] o_proj output
    void *buf_routing_scores;    // [N_EXPERTS] gate logits
    void *buf_expert_mid[6];         // [INTERMEDIATE] gate+up output per expert
    void *buf_expert_out[6];         // [DIM] down output per expert
    void *buf_expert_contiguous;     // [N_ACTIVE*DIM] contiguous expert outputs for combine
    // Gather mode output buffers: [K × dim] contiguous (vs K separate buffers)
    void *buf_gather_mid;            // [N_ACTIVE × INTERMEDIATE] f32 — gather gate+up output
    void *buf_gather_out;            // [N_ACTIVE × DIM] f32 — gather down output
    void *buf_gather_expert_ids;     // [N_ACTIVE] uint32 — current expert IDs for gather dispatch
    void *buf_gpu_route_selected;    // [N_ACTIVE] int32  — GPU routing: top-K expert IDs
    void *buf_gpu_route_weights;     // [N_ACTIVE] f32    — GPU routing: normalized weights
    void *buf_cached_flags;          // [N_EXPERTS] uint8 — 1=cached in SMELT, 0=not
    void *buf_shared_gate;           // [INTERMEDIATE] shared expert gate
    void *buf_shared_up;             // [INTERMEDIATE] shared expert up
    void *buf_shared_down;           // [DIM] shared expert down
    void *buf_norm_sum_sq;       // [1] sum of squares for RMS
    void *buf_input_norm;        // [DIM] next layer input norm weight (GPU)

    // BF16 buffers (S8b — MLX alignment)
    void *buf_residual_bf16;     // [MHC_MULT, DIM] current residual (bf16)
    void *buf_attn_input_bf16;   // [DIM] attn sublayer input (bf16)
    void *buf_ffn_input_bf16;    // [DIM] ffn sublayer input (bf16)
    void *buf_ffn_normed_bf16;   // [DIM] ffn normed for gate (bf16)
    void *buf_attn_out_bf16;     // [DIM] attention output (bf16)
    void *buf_ffn_out_bf16;      // [DIM] ffn output (bf16)

    // Persistent scratch buffers for mhc_pre / mhc_post (reused every forward_layer call)
    // Eliminates repeated large allocations that caused OOM at scale.
    void *buf_mhc_res_in;        // [MHC_MULT*DIM] f32  — residual input to mhc_pre (attn)
    void *buf_mhc_attn_in;       // [DIM] f32           — mhc_pre attn output
    void *buf_mhc_attn_norm_bf16;// [DIM] u16           — attn normed (bf16)
    void *buf_mhc_post_weights;  // [MHC_MULT] f32      — post weights from mhc_pre
    void *buf_mhc_comb_weights;  // [MHC_MULT*MHC_MULT] f32 — comb weights from mhc_pre
    void *buf_mhc_res_out;       // [MHC_MULT*DIM] u16  — mhc_post output (bf16)
    void *buf_mhc_ffn_in;        // [DIM] f32           — mhc_pre ffn output
    void *buf_mhc_ffn_norm_bf16; // [DIM] u16           — ffn normed (bf16)
    void *buf_mhc_ffn_res_in;    // [MHC_MULT*DIM] f32  — residual copy for ffn mhc_pre
    // Extra dedicated scratch buffers for mhc_post (avoids buffer aliasing bugs)
    void *buf_mhc_attn_out_bf16; // [DIM] u16           — attn_out converted to bf16 for mhc_post
    void *buf_mhc_res_bf16_in;   // [MHC_MULT*DIM] u16  — residual in bf16 for mhc_post input
    void *buf_mhc_post_res_out;  // [MHC_MULT*DIM] u16  — mhc_post result (bf16)
    void *buf_mhc_ffn_post_out;  // [MHC_MULT*DIM] u16  — mhc_post result for FFN sublayer (bf16)
    void *buf_ffn_out_f32;        // [DIM] f32              — ffn output in f32 for mhc_post_ffn_expand4

    // Persistent GPU buffers for per-layer mHC weights (uploaded once at set_layer_hc)
    // Avoids 1.5MB newBufferWithBytes per forward_layer call (was main OOM source)
    void *buf_attn_hc_fn[N_LAYERS];    // id<MTLBuffer> [MIX3*MHC_MULT*DIM] f32
    void *buf_attn_hc_base[N_LAYERS];  // id<MTLBuffer> [MIX3] f32
    void *buf_attn_hc_scale[N_LAYERS]; // id<MTLBuffer> [3] f32
    void *buf_ffn_hc_fn[N_LAYERS];     // id<MTLBuffer> [MIX3*MHC_MULT*DIM] f32
    void *buf_ffn_hc_base[N_LAYERS];   // id<MTLBuffer> [MIX3] f32
    void *buf_ffn_hc_scale[N_LAYERS];  // id<MTLBuffer> [3] f32
    // Persistent per-layer norm weight GPU buffers (uploaded once at set_weights)
    void *buf_input_norm_gpu[N_LAYERS]; // id<MTLBuffer> [DIM] f32  (attn pre-norm)
    void *buf_attn_norm_gpu[N_LAYERS];  // id<MTLBuffer> [DIM] f32  (ffn pre-norm)
    void *buf_gate_proj_gpu[N_LAYERS];  // id<MTLBuffer> [N_EXPERTS*DIM] f32  (router weights)

    // Expert I/O
    int packed_fd[N_LAYERS];     // per-layer packed expert file descriptors
    uint8_t *expert_buf[6];      // 2MB-aligned expert data buffers
    uint8_t *expert_buf_pred[6]; // second buffer set for prediction prefetch
    void *io_pool;               // persistent I/O thread pool

    // Expert memory cache — avoids SSD reads for frequently-used experts.
    // Layout: expert_mem_cache[layer][expert_id] = pointer into expert_mem_pool,
    // or NULL if not cached.
    // Pool is allocated once at startup, sized for expert_cache_n_experts experts per layer.
    uint8_t **expert_mem_cache[N_LAYERS];  // [N_LAYERS][N_EXPERTS] -> ptr or NULL
    uint8_t *expert_mem_pool[N_LAYERS];    // flat pool per layer (expert_cache_n_experts × EXPERT_SIZE)
    int expert_cache_n_experts;            // how many experts cached per layer (0=disabled)

    // Persistent GPU MTLBuffer wrappers for SMELT-cached experts.
    // Created once after SMELT warmup; reused every forward call.
    // expert_gpu_buf[layer][eid][slot]: slot 0=gate_W, 1=gate_S, 2=up_W, 3=up_S, 4=down_W, 5=down_S
    // NULL if expert not cached.
    void *expert_gpu_buf[N_LAYERS][N_EXPERTS][9];  // 6 for MXFP4, 9 for affine_v2

    // Gather MoE: per-layer NoCopy Metal buffer over the SMELT RAM pool.
    // The gather kernels address experts via pool_pos (slot index within the pool)
    // rather than raw expert_id, using the smelt_pool_pos remapping table.
    // This allows gather mode with any smelt_n (not just full 256).
    // buf_gather_gate_W[layer] == buf_gather_gate_s[layer] == ... == buf_gather_down_s[layer]
    // — they all point to the same single NoCopy buffer over expert_mem_pool[layer].
    // The gather kernel uses: offset = pool_pos * EXPERT_SIZE_U32 + component_offset + row*cols
    // NULL if gather mode is not active for that layer.
    void *buf_gather_gate_W[N_LAYERS];   // NoCopy view of entire SMELT pool per layer
    void *buf_gather_gate_s[N_LAYERS];   // alias of buf_gather_gate_W[layer]
    void *buf_gather_up_W[N_LAYERS];     // alias
    void *buf_gather_up_s[N_LAYERS];     // alias
    void *buf_gather_down_W[N_LAYERS];   // alias
    void *buf_gather_down_s[N_LAYERS];   // alias
    bool gather_mode;                    // true = use gather kernels instead of per-expert
    bool use_affine_experts;             // true = experts are in affine_v2 format (bf16 scale+bias, gs=64)
    size_t active_expert_size;           // EXPERT_SIZE or AFFINE_EXPERT_SIZE based on format

    // Remapping table: expert_id → pool_slot (position in SMELT pool).
    // Set during smelt_finish_warmup: smelt_pool_pos[layer][eid] = slot (0..n-1), or -1 if uncached.
    // Used in moe_forward_layer gather path to convert expert_ids → pool positions.
    int smelt_pool_pos[N_LAYERS][N_EXPERTS];

    // SMELT-style hot-expert preloading.
    // Phase 1 (warmup): count routing selections per expert per layer.
    // Phase 2 (post-warmup): preload top-N most-used experts, apply routing bias
    //          to steer future routing away from uncached experts.
    uint32_t routing_counts[N_LAYERS][N_EXPERTS]; // selection frequency (accumulated)
    int smelt_warmup_tokens;      // how many decode tokens to collect stats over (0=off)
    int smelt_n_per_layer;        // how many experts to cache per layer after warmup
    int smelt_tokens_seen;        // decode tokens processed so far (for warmup countdown)
    bool smelt_warmup_done;       // true once warmup is complete and cache is populated
    bool smelt_enabled;           // true if SMELT is active
    bool smelt_in_decode_phase;   // true after prefill completes (set by moe_infer_set_decode_phase)
    const char *smelt_stats_path; // path to routing stats file (for periodic auto-save on OOM protection)
    float smelt_penalty;          // routing score penalty for uncached experts (default -1e9)

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
    void *gate_bias_gpu[N_LAYERS];      // pre-allocated MTLBuffer wrapping gate_bias (avoids per-call newBuffer)
    const int64_t *tid2eid[N_LAYERS];    // [vocab_size, N_ACTIVE] hash routing table (NULL = score-based)
    // mHC weights per layer (f32): fn [24,16384], base [24], scale [3]
    const float *attn_hc_fn[N_LAYERS];
    const float *attn_hc_base[N_LAYERS];
    const float *attn_hc_scale[N_LAYERS];
    const float *ffn_hc_fn[N_LAYERS];
    const float *ffn_hc_base[N_LAYERS];
    const float *ffn_hc_scale[N_LAYERS];
    // ds4-style f16 copies of fn weights (allocated in set_layer_hc, freed in deinit)
    const uint16_t *attn_hc_fn_f16[N_LAYERS];
    const uint16_t *ffn_hc_fn_f16[N_LAYERS];
    // bf16 copies for MLX alignment
    const uint16_t *attn_hc_fn_bf16[N_LAYERS];
    const uint16_t *ffn_hc_fn_bf16[N_LAYERS];
    int expert_fd[N_LAYERS];

    // KV cache
    KVCache kv_cache[N_LAYERS];

    // Compressor/Indexer weights and state (set by moe_infer_set_layer_compressor)
    uint32_t compress_ratio[N_LAYERS];   // 0=none, 4=CSA, 128=HCA
    QuantWeight comp_wkv[N_LAYERS];
    QuantWeight comp_wgate[N_LAYERS];
    const float *comp_ape[N_LAYERS];     // [compress_ratio, out_dim] f32
    const float *comp_norm[N_LAYERS];    // [COMP_HEAD_DIM] f32
    // Indexer weights (only ratio=4 layers)
    QuantWeight idx_wq_b[N_LAYERS];
    QuantWeight idx_weights_proj[N_LAYERS];
    QuantWeight idx_comp_wkv[N_LAYERS];
    QuantWeight idx_comp_wgate[N_LAYERS];
    const float *idx_comp_ape[N_LAYERS]; // [4, 256] f32
    const float *idx_comp_norm[N_LAYERS];// [IDX_HEAD_DIM] f32
    // Runtime compressor state
    CompressorState comp_state[N_LAYERS];

    // Layer config
    LayerConfig layers[N_LAYERS];

    // State
    int current_pos;       // current sequence position
    int current_token_id;  // current input token ID (needed for hash routing in layers 0-2)
    bool initialized;

    // DSpark speculative decoding engine (optional, set by caller after init)
    void *dspark_engine;   // DSparkEngine* — NULL if DSpark not active
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

// Rollback KV cache to a specific position (for speculative decoding).
// After a batch verification where some draft tokens are rejected, roll back
// the KV cache to keep only entries at positions [0, valid_len).
// This truncates kv_cache[l].len and comp_state[l].n_comp/n_idx_comp appropriately.
void moe_infer_rollback_kv(MoEInferEngine *engine, int valid_len);

// Preload N experts per layer into a memory cache (eliminates SSD reads on cache hits).
// expert_cache_mb: total MB to allocate for cache. Pass 0 to preload ALL experts (~3.43 GB).
// Returns number of experts cached per layer.
int moe_infer_preload_experts(MoEInferEngine *engine, int expert_cache_mb);

// Initialize SMELT hot-expert preloading.
// warmup_tokens: number of decode tokens to collect routing stats over (e.g. 20).
// n_per_layer: how many experts to cache per layer after warmup (e.g. 51 for 20%).
// penalty: routing score subtracted from uncached experts during inference (e.g. 1e9).
// Call this INSTEAD of moe_infer_preload_experts when using SMELT.
void moe_infer_smelt_init(MoEInferEngine *engine, int warmup_tokens, int n_per_layer, float penalty);

// Save/load routing_counts to/from disk for SMELT hot-expert persistence.
// path: file path (e.g. "/path/to/model/.smelt_routing_stats.bin")
// load returns 1 if successfully loaded, 0 on first-run/error.
// Call load BEFORE smelt_init so smelt_finish_warmup uses the correct expert order.
void moe_infer_smelt_save_stats(MoEInferEngine *engine, const char *path);
int  moe_infer_smelt_load_stats(MoEInferEngine *engine, const char *path);
void moe_infer_smelt_set_penalty(MoEInferEngine *engine, float penalty);
void moe_infer_smelt_set_stats_path(MoEInferEngine *engine, const char *path);

// Signal that prefill is complete and decode phase begins.
// SMELT token counting (for warmup) only runs after this is called, ensuring
// routing_counts reflect actual decode routing rather than prefill (which uses
// hash routing for layers 0-2 and has all-zero counts).
void moe_infer_smelt_set_decode_phase(MoEInferEngine *engine);

// Called after warmup tokens are processed. Reads routing_counts, preloads top-N
// experts per layer, sets smelt_warmup_done=true.
// Returns number of experts cached per layer (0 on failure).
int moe_infer_smelt_finish_warmup(MoEInferEngine *engine);

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

// Batch forward pass for N tokens (proper transformer order: layer-first, then tokens).
// hidden_batch: [n_tokens, MHC_MULT*DIM] — each token's residual is updated in-place.
// After return, KV caches for all layers contain all n_tokens entries.
// token_ids: optional [n_tokens] token IDs for per-token hash routing (NULL = use current_token_id).
int moe_infer_forward_batch(MoEInferEngine *engine, float *hidden_batch, int n_tokens, int start_pos, const int *token_ids);

// Embed token_id into hidden state.
// hidden_out: [MHC_MULT * DIM] — all MHC streams set to embed[token_id].
void moe_infer_embed(MoEInferEngine *engine, int token_id, float *hidden_out);

// Compress mHC residual [MHC_MULT, DIM] → [DIM] using simple mean.
void moe_infer_compress_hc(MoEInferEngine *engine, const float *residual, float *out);

// Apply final RMSNorm + lm_head matmul.
// hidden: [DIM] (after moe_infer_compress_hc)
// logits_out: [vocab_size] (caller allocates)
// Returns 0 on success.
int moe_infer_get_logits(MoEInferEngine *engine, const float *hidden, float *logits_out);

// Initialize gather mode using the SMELT preloaded expert pool.
// Requires smelt_warmup_done=true and all N_EXPERTS cached for all layers.
// Creates per-layer NoCopy Metal buffer views over the SMELT RAM pool.
// Returns 1 on success, 0 on failure.
int moe_infer_init_gather_mode(MoEInferEngine *engine);

// Cleanup
void moe_infer_deinit(MoEInferEngine *engine);

// Set one layer's compressor weights. Call after moe_infer_set_layer_attn.
void moe_infer_set_layer_compressor(MoEInferEngine *engine, int layer,
    uint32_t compress_ratio,
    QuantWeight comp_wkv, QuantWeight comp_wgate,
    const float *comp_ape, const float *comp_norm);

// Set one layer's indexer weights (only for compress_ratio==4 layers).
void moe_infer_set_layer_indexer(MoEInferEngine *engine, int layer,
    QuantWeight idx_wq_b, QuantWeight idx_weights_proj,
    QuantWeight idx_comp_wkv, QuantWeight idx_comp_wgate,
    const float *idx_comp_ape, const float *idx_comp_norm);

// Run one token through the compressor. Emits a comp_kv block every `ratio` tokens.
void moe_infer_compressor_step(MoEInferEngine *engine, int layer, int pos,
                                const float *attn_normed);

// (Private helper exposed for testing) Run indexer to get allowed comp block mask.
// Returns false if no selection needed (n_comp <= index_topk).
bool moe_infer_indexer_step(MoEInferEngine *engine, int layer, int pos,
                             const float *attn_normed, const float *q_a_out,
                             bool *allowed_out);

#ifdef __cplusplus
}
#endif
