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
#include <stdatomic.h>

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
    QuantWeight wo_a;        // [8192, 4096] packed affine gs=64 (native format; preferred path)
    QuantWeight wo_b;        // [4096, 8192]  (DIM, O_GROUPS*O_LORA_RANK)
    const float *attn_sink;  // [64] per-head sink logits
} AttnWeights;


// Expert packed binary layout (MXFP4, group_size=32)
// Uses formula: LUT[nibble] * exp2(scale - 127)
// Scales stored as uint8 (E8M0), weights as uint32
// Layout per expert: [GATE_W: 4MB][GATE_S: 256KB][UP_W: 4MB][UP_S: 256KB][DOWN_W: 4MB][DOWN_S: 256KB]
// Bias planes (GATE_B/UP_B/DOWN_B) are reserved but not present in the actual files.
#define EXPERT_SIZE 13369344  // bytes per expert (~12.75 MB)
#define EXPERT_SIZE_2B 7471104  // bytes per expert, int2_gs64 format (~7.1 MB)
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
// Opt-in scalar/batch checkpoint capture
// ============================================================================

// Capture is dormant when callback == NULL (the default). Values and expert IDs
// are borrowed only for the duration of the callback; a test harness must copy
// them before returning. No capture buffer is allocated by the engine.
typedef enum {
    MOE_INFER_CAPTURE_SCALAR = 0,
    MOE_INFER_CAPTURE_BATCH = 1,
} MoeInferCapturePath;

typedef enum {
    MOE_INFER_CAPTURE_LAYER_INPUT = 0,
    MOE_INFER_CAPTURE_ATTENTION_OUTPUT,
    MOE_INFER_CAPTURE_ROUTER,
    MOE_INFER_CAPTURE_MOE_OUTPUT,
    MOE_INFER_CAPTURE_LAYER_OUTPUT,
} MoeInferCaptureCheckpoint;

typedef enum {
    MOE_INFER_CAPTURE_DTYPE_F32 = 0,
    MOE_INFER_CAPTURE_DTYPE_BF16,
} MoeInferCaptureDType;

typedef struct {
    int kv_length;
    uint32_t compressor_length;
    uint32_t indexer_length;
    uint64_t kv_state_digest;
    uint64_t compressor_state_digest;
    uint64_t indexer_state_digest;
} MoeInferStateSnapshot;

typedef struct {
    MoeInferCapturePath path;
    MoeInferCaptureCheckpoint checkpoint;
    MoeInferCaptureDType dtype;
    int layer;
    int position;            // requested absolute sequence position
    int kv_length;           // actual layer KV length at this checkpoint
    uint32_t compressor_length;
    uint32_t indexer_length;
    uint64_t kv_state_digest;
    uint64_t compressor_state_digest;
    uint64_t indexer_state_digest;
    const float *values;
    size_t value_count;
    const int *expert_ids;   // non-NULL only for ROUTER
    size_t expert_count;
} MoeInferDebugCaptureRecord;

typedef void (*MoeInferDebugCaptureFn)(
    void *context,
    const MoeInferDebugCaptureRecord *record);

// ============================================================================
// Opt-in request/layer telemetry (schema v1)
// ============================================================================

#define NATIVE_TELEMETRY_SCHEMA_VERSION 1
#define NATIVE_TELEMETRY_VALID_EXPERT_IO       UINT64_C(1)
#define NATIVE_TELEMETRY_VALID_MOE_GEOMETRY    UINT64_C(2)
#define NATIVE_TELEMETRY_VALID_WORKSPACE       UINT64_C(4)
#define NATIVE_TELEMETRY_VALID_ATTENTION_GPU   UINT64_C(8)

typedef enum NativeTelemetryPhase {
    NATIVE_TELEMETRY_PHASE_PREFILL = 0,
    NATIVE_TELEMETRY_PHASE_DECODE = 1,
    NATIVE_TELEMETRY_PHASE_VERIFY = 2,
    NATIVE_TELEMETRY_PHASE_COUNT = 3,
} NativeTelemetryPhase;

typedef struct NativeLayerMetrics {
    uint32_t layer;
    uint32_t phase;
    uint64_t calls;
    uint64_t unique_experts;
    uint64_t hot_hits;
    uint64_t prefetch_hits;
    uint64_t batch_pool_hits;
    uint64_t cold_misses;
    uint64_t pread_calls;
    uint64_t bytes_requested;
    uint64_t bytes_read;
    uint64_t short_reads;
    uint64_t io_wait_ns;
    uint64_t bytes_avoided;
    uint64_t cache_admissions;
    uint64_t cache_evictions;
    uint64_t moe_dispatches;
    uint64_t moe_command_buffers;
    uint64_t threadgroups;
    uint64_t scheduled_lanes;
    uint64_t active_lanes;
    uint32_t max_threads_per_threadgroup;
    uint32_t reserved;
} NativeLayerMetrics;

typedef struct NativeRequestMetrics {
    uint32_t schema_version;
    uint32_t enabled;
    uint32_t active_phase;
    uint32_t reserved;
    uint64_t valid_fields;
    NativeLayerMetrics layers[NATIVE_TELEMETRY_PHASE_COUNT][N_LAYERS];
} NativeRequestMetrics;

// Exact-Q4 latency events use an independent, frozen ABI version. They are not
// part of NATIVE_TELEMETRY_SCHEMA_VERSION and are not emitted until Task 0.2.
#define DMLX_EXACT_Q4_LATENCY_EVENT_ABI_VERSION 1u

typedef enum DmlxExactQ4Stage {
    DMLX_EXACT_Q4_STAGE_ATTENTION_ENCODE = 0,
    DMLX_EXACT_Q4_STAGE_ATTENTION_GPU_EXECUTION = 1,
    DMLX_EXACT_Q4_STAGE_ATTENTION_GPU_WAIT = 2,
    DMLX_EXACT_Q4_STAGE_ROUTING = 3,
    DMLX_EXACT_Q4_STAGE_IO_QUEUE = 4,
    DMLX_EXACT_Q4_STAGE_IO_PREAD = 5,
    DMLX_EXACT_Q4_STAGE_IO_EXPOSED_WAIT = 6,
    DMLX_EXACT_Q4_STAGE_MOE_ENCODE = 7,
    DMLX_EXACT_Q4_STAGE_MOE_GPU_EXECUTION = 8,
    DMLX_EXACT_Q4_STAGE_MOE_GPU_WAIT = 9,
    DMLX_EXACT_Q4_STAGE_CACHE_ADMISSION = 10,
    DMLX_EXACT_Q4_STAGE_HOST_ALLOC_WRAPPER = 11,
    DMLX_EXACT_Q4_STAGE_LOGITS_SAMPLE = 12,
    DMLX_EXACT_Q4_STAGE_OTHER_ACCOUNTED = 13,
    DMLX_EXACT_Q4_STAGE_COUNT = 14,
} DmlxExactQ4Stage;

#define DMLX_EXACT_Q4_EVENT_PARENT_ONLY       UINT32_C(1)
#define DMLX_EXACT_Q4_EVENT_GPU_TIME_KNOWN    UINT32_C(2)
#define DMLX_EXACT_Q4_EVENT_GPU_TIME_UNKNOWN  UINT32_C(4)
#define DMLX_EXACT_Q4_EVENT_JOIN_BLOCKED      UINT32_C(8)
#define DMLX_EXACT_Q4_EVENT_CPU_ACTIVE        UINT32_C(16)
#define DMLX_EXACT_Q4_EVENT_RAW_DIAGNOSTIC    UINT32_C(32)
#define DMLX_EXACT_Q4_EVENT_VALID_FLAGS       UINT32_C(63)

typedef enum DmlxExactQ4ReadSource {
    DMLX_EXACT_Q4_READ_FIXED_HOT = 0,
    DMLX_EXACT_Q4_READ_PREFETCH = 1,
    DMLX_EXACT_Q4_READ_BATCH_POOL = 2,
    DMLX_EXACT_Q4_READ_PREAD = 3,
    DMLX_EXACT_Q4_READ_SOURCE_COUNT = 4,
} DmlxExactQ4ReadSource;

#define DMLX_EXACT_Q4_READ_COMPLETED UINT32_C(1)
#define DMLX_EXACT_Q4_READ_VALID_FLAGS UINT32_C(1)

#define DMLX_EXACT_Q4_MLA_PATH_ABI_VERSION UINT32_C(1)
typedef enum DmlxMlaPath {
    DMLX_MLA_PATH_PERSISTENT_EXTERNAL = 0,
    DMLX_MLA_PATH_PERSISTENT_INTERNAL = 1,
    DMLX_MLA_PATH_PERSISTENT_INTERNAL_SPLIT = 2,
    DMLX_MLA_PATH_FALLBACK_TEMP_KV = 3,
    DMLX_MLA_PATH_COUNT = 4,
} DmlxMlaPath;
#define DMLX_MLA_PATH_PERSISTENT_KV UINT8_C(1)
#define DMLX_MLA_PATH_EXTERNAL_CB_REQUESTED UINT8_C(2)
#define DMLX_MLA_PATH_EXTERNAL_CB_USED UINT8_C(4)
#define DMLX_MLA_PATH_FALLBACK UINT8_C(8)
#define DMLX_MLA_PATH_STAGE_TIME_SPLIT UINT8_C(16)
#define DMLX_MLA_PATH_VALID_FLAGS UINT8_C(31)

typedef struct DmlxMlaPathIdentity {
    uint8_t path;
    uint8_t flags;
    uint16_t reserved;
} DmlxMlaPathIdentity;

int mla_attention_select_bf16_identity(
    bool has_persistent_kv, bool has_external_cb, bool stage_time,
    DmlxMlaPathIdentity *out_identity);

typedef struct DmlxMlaPathEvent {
    uint32_t request_ordinal;
    uint32_t token_ordinal;
    uint32_t event_ordinal;
    uint16_t layer;
    uint8_t path;
    uint8_t flags;
} DmlxMlaPathEvent;

typedef struct DmlxExactQ4StageEvent {
    uint32_t request_ordinal;
    uint32_t token_ordinal;
    uint32_t event_ordinal;
    uint16_t layer;
    uint16_t stage;
    uint32_t flags;
    uint32_t calls;
    uint64_t begin_ns;
    uint64_t end_ns;
    uint64_t bytes;
} DmlxExactQ4StageEvent;

typedef struct DmlxExactQ4ExpertReadEvent {
    uint32_t request_ordinal;
    uint32_t token_ordinal;
    uint32_t event_ordinal;
    uint16_t layer;
    uint16_t expert;
    uint8_t route_rank;
    uint8_t source;
    uint16_t reserved;
    int32_t error_code;
    uint32_t flags;
    uint64_t submit_ns;
    uint64_t worker_start_ns;
    uint64_t complete_ns;
    uint64_t requested_bytes;
    uint64_t read_bytes;
} DmlxExactQ4ExpertReadEvent;

#if defined(__cplusplus)
#define DMLX_EXACT_Q4_STATIC_ASSERT static_assert
#else
#define DMLX_EXACT_Q4_STATIC_ASSERT _Static_assert
#endif
DMLX_EXACT_Q4_STATIC_ASSERT(sizeof(DmlxExactQ4StageEvent) == 48, "DmlxExactQ4StageEvent ABI size");
DMLX_EXACT_Q4_STATIC_ASSERT(offsetof(DmlxExactQ4StageEvent, flags) == 16, "DmlxExactQ4StageEvent flags offset");
DMLX_EXACT_Q4_STATIC_ASSERT(offsetof(DmlxExactQ4StageEvent, begin_ns) == 24, "DmlxExactQ4StageEvent timing offset");
DMLX_EXACT_Q4_STATIC_ASSERT(offsetof(DmlxExactQ4StageEvent, bytes) == 40, "DmlxExactQ4StageEvent bytes offset");
DMLX_EXACT_Q4_STATIC_ASSERT(sizeof(DmlxExactQ4ExpertReadEvent) == 72, "DmlxExactQ4ExpertReadEvent ABI size");
DMLX_EXACT_Q4_STATIC_ASSERT(offsetof(DmlxExactQ4ExpertReadEvent, error_code) == 20, "DmlxExactQ4ExpertReadEvent error offset");
DMLX_EXACT_Q4_STATIC_ASSERT(offsetof(DmlxExactQ4ExpertReadEvent, submit_ns) == 32, "DmlxExactQ4ExpertReadEvent timing offset");
DMLX_EXACT_Q4_STATIC_ASSERT(offsetof(DmlxExactQ4ExpertReadEvent, requested_bytes) == 56, "DmlxExactQ4ExpertReadEvent bytes offset");
DMLX_EXACT_Q4_STATIC_ASSERT(sizeof(DmlxMlaPathIdentity) == 4, "DmlxMlaPathIdentity ABI size");
DMLX_EXACT_Q4_STATIC_ASSERT(sizeof(DmlxMlaPathEvent) == 16, "DmlxMlaPathEvent ABI size");
DMLX_EXACT_Q4_STATIC_ASSERT(offsetof(DmlxMlaPathEvent, layer) == 12, "DmlxMlaPathEvent layer offset");
#undef DMLX_EXACT_Q4_STATIC_ASSERT

// A scalar layer publishes its timing stages and MLA identity as one bounded
// unit. The fixed capacity covers the production stage set plus blocked joins.
#define DMLX_EXACT_Q4_LAYER_STAGE_CAPACITY UINT32_C(9)
typedef struct {
    bool active;
    bool failed;
    bool path_set;
    uint8_t reserved;
    uint16_t layer;
    uint16_t reserved2;
    uint32_t stage_count;
    DmlxExactQ4StageEvent stages[DMLX_EXACT_Q4_LAYER_STAGE_CAPACITY];
    DmlxMlaPathEvent path;
} DmlxExactQ4PendingLayer;

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
    DmlxExactQ4PendingLayer exact_q4_pending;
} DeferredExpertState;

// Bounded exact-Q4 producer lifecycle. Event layouts above remain the frozen
// Task 0.1 ABI; this state and API are independently configured and default off.
#define DMLX_EXACT_Q4_MAX_STAGE_CAPACITY UINT32_C(1048576)
#define DMLX_EXACT_Q4_MAX_READ_CAPACITY UINT32_C(1048576)

typedef enum DmlxExactQ4ProducerStatus {
    DMLX_EXACT_Q4_PRODUCER_OK = 0,
    DMLX_EXACT_Q4_PRODUCER_DISABLED = 1,
    DMLX_EXACT_Q4_PRODUCER_BUSY = 2,
    DMLX_EXACT_Q4_PRODUCER_INVALID_ARGUMENT = -1,
    DMLX_EXACT_Q4_PRODUCER_CAPACITY_EXCEEDED = -2,
    DMLX_EXACT_Q4_PRODUCER_INVALID_STATE = -3,
} DmlxExactQ4ProducerStatus;

typedef struct DmlxExactQ4LatencySnapshot {
    uint32_t abi_version;
    uint32_t valid;
    uint32_t enabled;
    uint32_t finalized;
    uint32_t request_ordinal;
    uint32_t token_ordinal;
    uint32_t stage_capacity;
    uint32_t read_capacity;
    uint32_t stage_count;
    uint32_t read_count;
    uint32_t dropped_stages;
    uint32_t dropped_reads;
    uint32_t incomplete;
    uint32_t active_read_publishers;
    DmlxExactQ4StageEvent *stages;
    DmlxExactQ4ExpertReadEvent *reads;
} DmlxExactQ4LatencySnapshot;

typedef struct DmlxExactQ4MlaPathSnapshot {
    uint32_t abi_version;
    uint32_t valid;
    uint32_t enabled;
    uint32_t finalized;
    uint32_t request_ordinal;
    uint32_t token_ordinal;
    uint32_t capacity;
    uint32_t count;
    uint32_t dropped;
    uint32_t reserved;
    DmlxMlaPathEvent *paths;
} DmlxExactQ4MlaPathSnapshot;

#if defined(__cplusplus)
#define DMLX_EXACT_Q4_STATIC_ASSERT static_assert
#else
#define DMLX_EXACT_Q4_STATIC_ASSERT _Static_assert
#endif
DMLX_EXACT_Q4_STATIC_ASSERT(sizeof(DmlxExactQ4LatencySnapshot) == 72,
    "DmlxExactQ4LatencySnapshot ABI size");
DMLX_EXACT_Q4_STATIC_ASSERT(offsetof(DmlxExactQ4LatencySnapshot, stages) == 56,
    "DmlxExactQ4LatencySnapshot stage pointer offset");
DMLX_EXACT_Q4_STATIC_ASSERT(sizeof(DmlxExactQ4MlaPathSnapshot) == 48,
    "DmlxExactQ4MlaPathSnapshot ABI size");
DMLX_EXACT_Q4_STATIC_ASSERT(offsetof(DmlxExactQ4MlaPathSnapshot, paths) == 40,
    "DmlxExactQ4MlaPathSnapshot paths pointer offset");
#undef DMLX_EXACT_Q4_STATIC_ASSERT

// Raw completed-command-buffer timestamp probe. This ABI is independent from
// exact-Q4 latency events: it captures source clock domains for offline proof
// and must not publish GPU_TIME_KNOWN until that proof is accepted.
#define DMLX_GPU_TIMESTAMP_PROBE_ABI_VERSION UINT32_C(1)
#define DMLX_GPU_TIMESTAMP_PROBE_CAPACITY UINT32_C(4096)

typedef enum DmlxGpuCommandBufferKind {
    DMLX_GPU_CB_ATTENTION = 1,
    DMLX_GPU_CB_ROUTED_MOE = 2,
} DmlxGpuCommandBufferKind;

typedef enum DmlxGpuTimestampInvalidReason {
    DMLX_GPU_TIMESTAMP_VALID = 0,
    DMLX_GPU_TIMESTAMP_SELECTOR_UNAVAILABLE = 1,
    DMLX_GPU_TIMESTAMP_NONFINITE = 2,
    DMLX_GPU_TIMESTAMP_ZERO = 3,
    DMLX_GPU_TIMESTAMP_REVERSED = 4,
    DMLX_GPU_TIMESTAMP_HOST_ORDER = 5,
} DmlxGpuTimestampInvalidReason;

typedef enum DmlxGpuTimestampProbeStatus {
    DMLX_GPU_TIMESTAMP_PROBE_OK = 0,
    DMLX_GPU_TIMESTAMP_PROBE_DISABLED = 1,
    DMLX_GPU_TIMESTAMP_PROBE_INVALID_ARGUMENT = -1,
    DMLX_GPU_TIMESTAMP_PROBE_ALLOCATION_FAILED = -2,
    DMLX_GPU_TIMESTAMP_PROBE_CAPACITY_EXCEEDED = -3,
} DmlxGpuTimestampProbeStatus;

typedef struct DmlxGpuTimestampSample {
    uint64_t sequence;
    uint64_t request_id;
    uint32_t token_index;
    uint16_t layer;
    uint16_t kind;
    uint64_t pre_commit_ns;
    uint64_t post_commit_ns;
    uint64_t post_wait_ns;
    double gpu_start_s;
    double gpu_end_s;
    uint32_t valid;
    uint32_t reason;
} DmlxGpuTimestampSample;

#if defined(__cplusplus)
#define DMLX_GPU_TIMESTAMP_STATIC_ASSERT static_assert
#else
#define DMLX_GPU_TIMESTAMP_STATIC_ASSERT _Static_assert
#endif
DMLX_GPU_TIMESTAMP_STATIC_ASSERT(sizeof(DmlxGpuTimestampSample) == 72,
    "DmlxGpuTimestampSample ABI size");
DMLX_GPU_TIMESTAMP_STATIC_ASSERT(offsetof(DmlxGpuTimestampSample, pre_commit_ns) == 24,
    "DmlxGpuTimestampSample host timing offset");
DMLX_GPU_TIMESTAMP_STATIC_ASSERT(offsetof(DmlxGpuTimestampSample, gpu_start_s) == 48,
    "DmlxGpuTimestampSample GPU timing offset");
DMLX_GPU_TIMESTAMP_STATIC_ASSERT(offsetof(DmlxGpuTimestampSample, valid) == 64,
    "DmlxGpuTimestampSample validity offset");
#undef DMLX_GPU_TIMESTAMP_STATIC_ASSERT

#define DMLX_ATTENTION_STAGE_COUNT 14u
#define DMLX_ATTENTION_STAGE_SAMPLE_COUNT 28u

typedef enum DmlxAttentionStage {
    DMLX_ATTENTION_STAGE_MHC_PRE = 0,
    DMLX_ATTENTION_STAGE_WQ_A = 1,
    DMLX_ATTENTION_STAGE_Q_RMS = 2,
    DMLX_ATTENTION_STAGE_WQ_B = 3,
    DMLX_ATTENTION_STAGE_Q_HEAD_RMS = 4,
    DMLX_ATTENTION_STAGE_Q_ROPE = 5,
    DMLX_ATTENTION_STAGE_WKV = 6,
    DMLX_ATTENTION_STAGE_KV_RMS = 7,
    DMLX_ATTENTION_STAGE_KV_ROPE = 8,
    DMLX_ATTENTION_STAGE_KV_UPDATE = 9,
    DMLX_ATTENTION_STAGE_SDPA = 10,
    DMLX_ATTENTION_STAGE_INV_Q_ROPE = 11,
    DMLX_ATTENTION_STAGE_WO_A = 12,
    DMLX_ATTENTION_STAGE_WO_B = 13,
} DmlxAttentionStage;

typedef enum DmlxAttentionStageStatus {
    DMLX_ATTENTION_STAGE_AVAILABLE = 0,
    DMLX_ATTENTION_STAGE_UNSUPPORTED = 1,
    DMLX_ATTENTION_STAGE_CONFIGURE_FAILED = 2,
    DMLX_ATTENTION_STAGE_INVALID_SAMPLE = 3,
    DMLX_ATTENTION_STAGE_TOPOLOGY_MISMATCH = 4,
} DmlxAttentionStageStatus;

typedef struct DmlxAttentionStageSnapshot {
    uint32_t status;
    uint32_t enabled;
    uint64_t duration_ns_sum[DMLX_ATTENTION_STAGE_COUNT];
    uint64_t valid_count[DMLX_ATTENTION_STAGE_COUNT];
} DmlxAttentionStageSnapshot;

// Independent bounded CPU decode timeline (Task 1.1). Status is published last.
#define DMLX_DECODE_TIMELINE_ABI_VERSION UINT32_C(2)
#define DMLX_DECODE_TIMELINE_MAX_CAPACITY UINT32_C(1048576)
#define DMLX_DECODE_TIMELINE_NO_LAYER UINT16_MAX

typedef enum DmlxDecodeTimelineStage {
    DMLX_DECODE_TIMELINE_TOKEN_FORWARD = 0,
    DMLX_DECODE_TIMELINE_ATTENTION_CB = 1,
    DMLX_DECODE_TIMELINE_DSA_CPU = 2,
    DMLX_DECODE_TIMELINE_CB2CMD2_CB = 3,
    DMLX_DECODE_TIMELINE_IO_EXPOSED_WAIT = 4,
    DMLX_DECODE_TIMELINE_SHARED_EXPERT_CB = 5,
    DMLX_DECODE_TIMELINE_ROUTED_MOE_CB = 6,
    DMLX_DECODE_TIMELINE_CB3_CB = 7,
    DMLX_DECODE_TIMELINE_LOGITS_CB = 8,
    DMLX_DECODE_TIMELINE_TOKEN_SAMPLE = 9,
    DMLX_DECODE_TIMELINE_STAGE_COUNT = 10,
} DmlxDecodeTimelineStage;

typedef enum DmlxDecodeTimelineEventStatus {
    DMLX_DECODE_TIMELINE_EVENT_RESERVED = 0,
    DMLX_DECODE_TIMELINE_EVENT_COMPLETE = 1,
    DMLX_DECODE_TIMELINE_EVENT_INCOMPLETE = 2,
} DmlxDecodeTimelineEventStatus;

typedef enum DmlxDecodeTimelineStatus {
    DMLX_DECODE_TIMELINE_OK = 0,
    DMLX_DECODE_TIMELINE_DISABLED = 1,
    DMLX_DECODE_TIMELINE_BUSY = 2,
    DMLX_DECODE_TIMELINE_INVALID_ARGUMENT = -1,
    DMLX_DECODE_TIMELINE_CAPACITY_EXCEEDED = -2,
    DMLX_DECODE_TIMELINE_INVALID_STATE = -3,
} DmlxDecodeTimelineStatus;

typedef struct DmlxDecodeTimelineEvent {
    uint32_t token;
    uint16_t layer;
    uint8_t stage;
    uint8_t status;
    uint64_t begin_ns;
    uint64_t encode_done_ns;
    uint64_t pre_commit_ns;
    uint64_t post_commit_ns;
    uint64_t post_wait_ns;
    uint64_t end_ns;
} DmlxDecodeTimelineEvent;

typedef struct DmlxDecodeTimelineSnapshot {
    uint32_t abi_version;
    uint32_t valid;
    uint32_t enabled;
    uint32_t finalized;
    uint32_t capacity;
    uint32_t count;
    uint32_t dropped;
    uint32_t incomplete;
    uint32_t active_publishers;
    uint32_t reserved;
    DmlxDecodeTimelineEvent *events;
} DmlxDecodeTimelineSnapshot;

#if defined(__cplusplus)
#define DMLX_DECODE_TIMELINE_STATIC_ASSERT static_assert
#else
#define DMLX_DECODE_TIMELINE_STATIC_ASSERT _Static_assert
#endif
DMLX_DECODE_TIMELINE_STATIC_ASSERT(sizeof(DmlxDecodeTimelineEvent) == 56,
    "DmlxDecodeTimelineEvent ABI size");
DMLX_DECODE_TIMELINE_STATIC_ASSERT(offsetof(DmlxDecodeTimelineEvent, begin_ns) == 8,
    "DmlxDecodeTimelineEvent timing offset");
DMLX_DECODE_TIMELINE_STATIC_ASSERT(sizeof(DmlxDecodeTimelineSnapshot) == 48,
    "DmlxDecodeTimelineSnapshot ABI size");
#undef DMLX_DECODE_TIMELINE_STATIC_ASSERT

struct DmlxExactQ4ProducerState;
struct DmlxGpuTimestampProbeState;
struct DmlxAttentionStageProbeState;
struct DmlxDecodeTimelineState;
typedef void (*DmlxDecodeTimelinePublisherHookFn)(void *context);

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
    // Gather6 MoE kernels: pointer-array variant for non-contiguous expert blobs
    void *pipe_gather6_gate_up;         // gather6_gate_up_swiglu
    void *pipe_gather6_down;            // gather6_down
    void *pipe_gather6_gate_up_2b;      // gather6_gate_up_swiglu_2b (int2_gs64)
    void *pipe_gather6_down_2b;         // gather6_down_2b (int2_gs64)
    // Persistent whole-blob NoCopy wrappers for the fixed pread/prefetch buffers,
    // so gather6 reuses stable GPU page mappings across tokens instead of
    // remapping 6×13.4MB fresh every layer.
    void *blob_wrap_pread[6];
    void *blob_wrap_pred[6];
    void *blob_wrap_pred2[6];
    // Backing pointer captured when each wrapper was created.  If the pointer
    // changes (e.g. SMELT slot evicted and reused for a different expert) the
    // old NoCopy wrapper is stale; we release it and create a fresh one.
    void *blob_wrap_pread_ptr[6];
    void *blob_wrap_pred_ptr[6];
    void *blob_wrap_pred2_ptr[6];
    // Cross-token expert I/O dedup pool for moe_infer_forward_batch (prefill).
    // Within one layer, each unique expert is pread once into a pool slot and
    // shared by all tokens that routed to it (60 reads → ~28 for N=10).
    // Lazily allocated on first batch forward; decode-only paths pay nothing.
    uint8_t *batch_pool;         // contiguous slot memory (2MB-aligned)
    void   **batch_wrap;         // [capacity] persistent NoCopy MTLBuffer wrappers
    int     *batch_eids;         // [capacity] slot → expert_id (this layer)
    uint8_t *batch_pins;         // [capacity] explicit resolver use-lifetime pins
    int      batch_count;        // slots used this layer (reset per layer)
    int      batch_hits;         // cross-token pool hits this layer (stats)
    int      batch_capacity;     // total slots (env NATIVE_BATCH_POOL_SLOTS, default 48)
    size_t   batch_slot_size;    // = active_expert_size at alloc time
    uintptr_t unique_resolver_owner; // one active pinned resolution per engine
    void *pipe_int8_gate_up_swiglu;     // fused_gate_up_swiglu_int8_e8m0 (DSpark MTP)
    void *pipe_int8_dequant_matvec;     // dequant_matvec_int8_e8m0 (DSpark MTP)
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
    void *pipe_dequant_matvec_affine_bf16in_f32out_grp8;  // fused wo_a (8 groups, 1 dispatch)
    void *pipe_mla_sdpa_prefill_bfloat; // legacy full-sequence prefill SDPA
    void *pipe_mla_sdpa_batch_f16_kv;   // Wave-2 chunk-local BF16 Q / f16 KV SDPA
    void *pipe_bf16_to_f16_row;         // KV cache bf16→f16 conversion
    void *pipe_limited_swiglu;          // in-place SwiGLU for shared expert
    void *pipe_f32_to_bf16_vec;         // residual f32→bf16 on GPU
    void *pipe_bf16_to_f32_vec;         // residual bf16→f32 writeback
    void *pipe_moe_route_gpu;           // GPU top-K routing (sqrtsoftplus+bitonic+normalize)
    // Wave-4 opt-in primitives consumed by forwardBatch only when both expert
    // bucket flags are enabled in the immutable startup batch config.
    void *pipe_bucket_gather_f32;        // bucket token ids: [B,H] -> [M,H]
    void *pipe_bucket_mxfp4_gate_up;     // one expert: [M,H] -> [M,I] SwiGLU
    void *pipe_bucket_mxfp4_down;        // one expert: [M,I] -> [M,H]
    void *pipe_bucket6_mxfp4_gate_up;    // up to six experts: stable rows -> [M,I]
    void *pipe_bucket6_mxfp4_down;       // up to six experts: [M,I] -> [M,H]
    void *pipe_bucket_combine_f32;       // bucket rows + shared rows -> [B,H]
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
    void *buf_gpu_route_selected;    // [MAX_SEQ_LEN × N_ACTIVE] int32 — GPU route IDs
    void *buf_gpu_route_weights;     // [MAX_SEQ_LEN × N_ACTIVE] f32 — normalized weights
    void *buf_batch_routing_scores;  // [MAX_SEQ_LEN × N_EXPERTS] f32 — persistent batch input
    void *buf_batch_route_bias;      // [N_EXPERTS] f32 — persistent batch bias scratch
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

    // Persistent shared-expert weight buffers (created lazily, reused every
    // token): [gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b].
    // Previously 9 MTLBuffer objects were created PER LAYER PER TOKEN.
    void *buf_shared_w[N_LAYERS][9];

    // Engine-lifetime buffers hoisted from per-layer-per-token allocations.
    // buf_weights[l]: 24-byte routing-weight vector for pipe_moe_combine atIndex:1.
    // buf_zero_resid[l]: 16KB zero tensor for pipe_moe_combine atIndex:2.
    // Both are lazily allocated on first use (same pattern as buf_shared_w) and
    // released in moe_infer_deinit. Contents are written by the CPU routing path
    // (weights) or left as all-zeros (zero_resid) and never read back by the CPU,
    // so reusing the same allocation across tokens is numerically identical to
    // creating a fresh one — the combine kernel reads but never writes them back.
    void *buf_weights[N_LAYERS];
    void *buf_zero_resid[N_LAYERS];

    // Expert I/O
    int packed_fd[N_LAYERS];     // per-layer packed expert file descriptors
    uint8_t *expert_buf[6];      // 2MB-aligned expert data buffers
    uint8_t *expert_buf_pred[6]; // prefetch buffer set 0 (layer parity: even layers)
    uint8_t *expert_buf_pred2[6];// prefetch buffer set 1 (odd layers)
    void *io_pool;               // persistent I/O thread pool (critical path)
    void *prefetch_pool;         // second I/O thread pool (async prediction prefetch)

    // DSpark prefill/verify: per-position mean-pooled hidden for layers
    // 40/41/42 ([3, n, DIM]), engine-owned, grown on demand.
    float *dspark_hidden3;
    int    dspark_hidden3_n;
    int    dspark_hidden3_cap;
    // Predictive cross-layer expert prefetch state.
    // After layer L's routing is known, async-prefetch layer L+1's predicted
    // experts (previous token's routing at L+1) into pred set (L+1)%2 while
    // layer L's GPU work + layer L+1's attention chain execute (~11ms window).
    int  prefetch_layer;         // target layer of in-flight prefetch (-1 = none)
    int  prefetch_ids[6];        // prefetched expert ids aligned to buffer slots
    int  prefetch_task[6];       // io_pool task index per buffer slot (for wait_task)
    int  prefetch_count;         // number of valid slots in prefetch_ids
    int  prefetch_generation;    // posted I/O generation, independent of logical consume
    bool prefetch_active;        // a prefetch can be consumed for prefetch_layer
    int  in_batch;               // !=0 inside moe_infer_forward_batch (prefetch off)
    int  prefetch_disabled;      // 0=auto, 1=off (env NATIVE_PREFETCH=0 or adaptive)

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
    int  expert_format_2b;               // 1 = packed dir is int2_gs64 (2-bit) format
    size_t active_expert_size;           // EXPERT_SIZE or AFFINE_EXPERT_SIZE based on format

    // Remapping table: expert_id → pool_slot (position in SMELT pool).
    // Set during smelt_finish_warmup: smelt_pool_pos[layer][eid] = slot (0..n-1), or -1 if uncached.
    // Used in moe_forward_layer gather path to convert expert_ids → pool positions.
    int smelt_pool_pos[N_LAYERS][N_EXPERTS];

    // Experimental bounded per-layer LRU. Metadata is fixed-size and the
    // backing expert pools retain the exact fixed-hot geometry. Hash-routed
    // layers remain immutable/full-resident.
    bool experimental_cache_enabled;
    uint64_t cache_lru_clock;
    uint64_t cache_lru_recency[N_LAYERS][N_EXPERTS];
    uint64_t cache_lru_admissions;
    uint64_t cache_lru_evictions;

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
    bool decode_phase;            // true after prefill completes (unconditional; gates predictive prefetch)
    const char *smelt_stats_path; // path to routing stats file (for periodic auto-save on OOM protection)
    float smelt_penalty;          // routing score penalty for uncached experts (default -1e9)

    // Deferred CMD3
    DeferredExpertState deferred;

    // Prediction
    ExpertPredictor predictor;

    // SMELT hit-rate counters (decode phase): hits = selections served from
    // the resident SMELT pool; printed every 500 layers' worth of routing.
    uint64_t smelt_hits;
    uint64_t smelt_total;

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

    // KV cache and transaction ownership. Only one staged KV transaction may
    // own an engine at a time; generation prevents stale/cross-engine commit.
    KVCache kv_cache[N_LAYERS];
    uint64_t kv_transaction_generation;
    uint64_t kv_active_transaction_generation;
    uintptr_t kv_active_transaction_owner;

    // Compressor/Indexer weights and state (set by moe_infer_set_layer_compressor)
    uint32_t compress_ratio[N_LAYERS];   // 0=none, 4=CSA, 128=HCA
    QuantWeight comp_wkv[N_LAYERS];
    QuantWeight comp_wgate[N_LAYERS];
    // Pre-allocated GPU buffers for compressor weights (avoid CPU matvec)
    void *comp_wkv_gpu_w[N_LAYERS];   // packed weights
    void *comp_wkv_gpu_s[N_LAYERS];   // scales
    void *comp_wkv_gpu_b[N_LAYERS];   // biases
    void *comp_wgate_gpu_w[N_LAYERS];
    void *comp_wgate_gpu_s[N_LAYERS];
    void *comp_wgate_gpu_b[N_LAYERS];
    void *comp_out_gpu[N_LAYERS];     // [max(CSA_OUT_DIM, HCA_OUT_DIM)] f32 — kv output
    void *comp_out2_gpu[N_LAYERS];    // [max(CSA_OUT_DIM, HCA_OUT_DIM)] f32 — gate output
    const float *comp_ape[N_LAYERS];     // [compress_ratio, out_dim] f32
    const float *comp_norm[N_LAYERS];    // [COMP_HEAD_DIM] f32
    // Indexer weights (only ratio=4 layers)
    QuantWeight idx_wq_b[N_LAYERS];
    QuantWeight idx_weights_proj[N_LAYERS];
    QuantWeight idx_comp_wkv[N_LAYERS];
    QuantWeight idx_comp_wgate[N_LAYERS];
    // Pre-allocated GPU buffers for indexer compressor weights (same pattern
    // as comp_wkv_gpu_* above — eliminates CPU matvec in the decode hot path)
    void *idx_comp_wkv_gpu_w[N_LAYERS];
    void *idx_comp_wkv_gpu_s[N_LAYERS];
    void *idx_comp_wkv_gpu_b[N_LAYERS];
    void *idx_comp_wgate_gpu_w[N_LAYERS];
    void *idx_comp_wgate_gpu_s[N_LAYERS];
    void *idx_comp_wgate_gpu_b[N_LAYERS];
    void *idx_comp_out_gpu[N_LAYERS];   // [256] f32 — kv output
    void *idx_comp_out2_gpu[N_LAYERS];  // [256] f32 — gate output
    const float *idx_comp_ape[N_LAYERS]; // [4, 256] f32
    const float *idx_comp_norm[N_LAYERS];// [IDX_HEAD_DIM] f32
    // Runtime compressor state
    CompressorState comp_state[N_LAYERS];

    // Layer config
    LayerConfig layers[N_LAYERS];

    // Immutable startup batch configuration. The opaque workspace is allocated
    // once when both Wave-4 expert flags are enabled and reused for every layer.
    uint32_t batch_config_set;
    uint32_t batch_enable_prefill;
    uint32_t batch_enable_mla;
    uint32_t batch_enable_expert_buckets;
    uint32_t batch_enable_batched_expert_kernels;
    uint32_t batch_enable_dspark_verify;
    uint32_t batch_max_prefill_chunk;
    uint32_t batch_max_verify_chunk;
    uint64_t batch_workspace_budget_bytes;
    void *forward_batch_workspace; // NativeBatchWorkspace*
    uint64_t batch_scalar_layer_calls;
    uint64_t batch_batched_moe_layers;
    uint64_t batch_workspace_high_water_bytes;

    // State
    int current_pos;       // current sequence position
    int current_token_id;  // current input token ID (needed for hash routing in layers 0-2)
    bool initialized;

    // Debug-only checkpoint capture. NULL by default: no allocation and only a
    // predictable branch at each semantic checkpoint.
    MoeInferDebugCaptureFn debug_capture;
    void *debug_capture_context;

    // Exact-Q4 latency storage is configured independently. `exact_q4` is
    // NULL while disabled, making producer helpers a single null branch;
    // `exact_q4_storage` retains configured bounded storage for reuse.
    struct DmlxExactQ4ProducerState *exact_q4_storage;
    struct DmlxExactQ4ProducerState *exact_q4;

    // Independent bounded decode timeline. The zero-initialized gate protects
    // storage ownership and enabled publisher acquisition. NULL remains the
    // disabled hot path; the hook is used only by deterministic focused tests.
    struct DmlxDecodeTimelineState *decode_timeline_storage;
    _Atomic(struct DmlxDecodeTimelineState *) decode_timeline;
    _Atomic uint32_t decode_timeline_gate;
    DmlxDecodeTimelinePublisherHookFn decode_timeline_publisher_hook;
    void *decode_timeline_publisher_hook_context;

    // Raw Metal timestamp probe is independently configured and default off.
    // `gpu_timestamp_probe` is NULL on the disabled hot path; storage is
    // allocated only by explicit enable and retained for reset/reuse.
    struct DmlxGpuTimestampProbeState *gpu_timestamp_probe_storage;
    struct DmlxGpuTimestampProbeState *gpu_timestamp_probe;

    // Production-topology attention compute-stage timestamps. Both pointers are
    // NULL by default; configure allocates one retained 28-sample buffer and
    // fourteen fixed descriptors, reused serially for all scalar layers.
    struct DmlxAttentionStageProbeState *attention_stage_probe_storage;
    struct DmlxAttentionStageProbeState *attention_stage_probe;

    // Request telemetry is fixed-size and opt-in. When disabled, hot paths pay
    // one predictable branch and do not read clocks or allocate memory.
    bool telemetry_enabled;
    uint32_t telemetry_gpu_timestamp_status;
    uint32_t telemetry_attention_stage_status;
    NativeTelemetryPhase telemetry_phase;
    NativeRequestMetrics telemetry;
    uint64_t telemetry_seen[NATIVE_TELEMETRY_PHASE_COUNT][N_LAYERS][4];

    // Privacy-safe routing tracing is independently opt-in. NULL is the
    // default and makes the route hot path a single predictable pointer check.
    // The writer owns one fixed-capacity buffer allocated only at configure.
    struct RoutingTraceWriter *routing_trace;
    NativeTelemetryPhase routing_trace_phase;

    // DSpark speculative decoding engine (optional, set by caller after init)
    void *dspark_engine;   // DSparkEngine* — NULL if DSpark not active
    bool dspark_accumulate_enabled;  // Only accumulate hidden during real decode, not verification

    // Quantized native embed/lm_head (4-bit affine, gs=64). embed_q is
    // dequantized per-row on read (bit-exact); lm_head_q drives the GPU
    // logits matvec. Set by moe_infer_set_head_quant; active when embed /
    // lm_head (f32) are NULL.
    QuantWeight embed_q;
    QuantWeight lm_head_q;
    void *lmh_w;      // packed weights (copied once)
    void *lmh_sc;     // scales
    void *lmh_bi;     // biases
    void *lmh_x;      // [DIM] f32 input
    void *lmh_out;    // [vocab] f32 logits output
} MoEInferEngine;

// ============================================================================
// API
// ============================================================================

// Wave-2 batch controls are explicit and default to all-zero/off. The legacy
// scalar API remains unchanged. uint32_t booleans keep the C/Zig ABI stable.
typedef enum NativeBatchPhase {
    NATIVE_BATCH_PHASE_PREFILL = 0,
    NATIVE_BATCH_PHASE_VERIFY = 1,
    NATIVE_BATCH_PHASE_DECODE = 2,
} NativeBatchPhase;

typedef enum NativeBatchStatus {
    NATIVE_BATCH_OK = 0,
    NATIVE_BATCH_INVALID_ARGUMENT = -1,
    NATIVE_BATCH_CAPACITY_EXCEEDED = -2,
    NATIVE_BATCH_INVALID_POSITION = -3,
    NATIVE_BATCH_INVALID_STATE = -4,
    NATIVE_BATCH_BUDGET_EXCEEDED = -5,
} NativeBatchStatus;

// Capture the exact scalar-visible recurrent state at an explicit logical KV
// prefix. The physical cache may already contain later transaction-staged rows;
// only [0, logical_kv_length) contributes to the digest.
int moe_infer_capture_state_snapshot(
    const MoEInferEngine *engine,
    int layer,
    int logical_kv_length,
    MoeInferStateSnapshot *snapshot);

typedef struct NativeBatchConfig {
    uint32_t enable_batch_prefill;
    uint32_t enable_batch_mla;
    uint32_t enable_expert_buckets;
    uint32_t enable_batched_expert_kernels;
    uint32_t enable_dspark_batch_verify;
    uint32_t max_prefill_chunk;
    uint32_t max_verify_chunk;
    uint32_t reserved;
    uint64_t workspace_budget_bytes;
} NativeBatchConfig;

#define NATIVE_BATCH_WORKSPACE_ALIGNMENT 256u

typedef void (*NativeBatchWorkspaceMetalReleaseFn)(void *metal_buffer);

typedef struct NativeBatchWorkspace {
    uint32_t capacity_tokens;
    // Non-zero arena lifetime epoch. Reset advances it before storage may be
    // reused, permanently poisoning the workspace rather than wrapping.
    uint32_t arena_epoch;
    uint64_t budget_bytes;
    uint64_t allocated_bytes;
    uint64_t used_bytes;
    uint64_t high_water_bytes;
    void *storage;
    uintptr_t self_cookie;
    // Optional persistent Metal no-copy view. The owning parent workspace has
    // a release callback; stack subworkspaces borrow the same opaque view and
    // select their arena with metal_buffer_offset.
    void *metal_buffer;
    uint64_t metal_buffer_offset;
    uint64_t metal_buffer_creations;
    NativeBatchWorkspaceMetalReleaseFn metal_buffer_release;
} NativeBatchWorkspace;

// Wave-3 token-major routing record. Task 3.2 populates exactly N_ACTIVE
// records per token; Task 3.3 stably scatters the same records into buckets.
// Fixed-width fields keep the C/Zig ABI and GPU upload layout deterministic.
typedef struct NativeActivationPair {
    uint32_t token_id;
    uint32_t expert_id;
    uint32_t route_rank;
    float weight;
} NativeActivationPair;

// Task 3.1 typed view into one bounded NativeBatchWorkspace reservation. Bind
// once per chunk and reset the index arrays between layers; no member owns
// memory and no per-layer allocation is permitted. inverse_mapping maps each
// token-major pair index to its eventual bucketed-pair index.
typedef struct NativeExpertBucketWorkspace {
    NativeActivationPair *activation_pairs; // [pair_capacity], token-major
    NativeActivationPair *bucketed_pairs;   // [pair_capacity], expert-major
    uint32_t *counts;                       // [N_EXPERTS]
    uint32_t *offsets;                      // [N_EXPERTS + 1]
    uint32_t *unique_expert_ids;            // [N_EXPERTS], ascending prefix
    uint32_t *inverse_mapping;              // [pair_capacity]
    NativeBatchWorkspace *owner_workspace;
    uintptr_t self_cookie;
    uint64_t arena_offset_bytes;
    uint64_t arena_size_bytes;
    uint32_t batch_tokens;
    uint32_t pair_capacity;
    uint32_t unique_capacity;
    uint32_t unique_count;
    // Captured owner arena_epoch; rejects a view after parent reset/reuse.
    uint32_t arena_epoch;
} NativeExpertBucketWorkspace;

typedef enum NativeExpertResolutionSource {
    NATIVE_EXPERT_RESOLUTION_HOT = 1,
    NATIVE_EXPERT_RESOLUTION_BATCH_POOL = 2,
    NATIVE_EXPERT_RESOLUTION_COLD = 3,
} NativeExpertResolutionSource;

typedef struct NativeResolvedExpert {
    uint32_t expert_id;
    uint32_t source;
    uint32_t slot_index;
    uint32_t reserved;
    uint8_t *data;
} NativeResolvedExpert;

typedef struct NativeExpertResolutionMetrics {
    uint32_t layer;
    uint32_t activation_pairs;
    uint32_t unique_experts;
    uint32_t hot_hits;
    uint32_t batch_pool_hits;
    uint32_t cold_reads;
    uint32_t pread_calls;
    uint32_t short_reads;
    uint32_t error_expert;
    uint32_t reserved;
    uint64_t bytes_requested;
    uint64_t bytes_read;
    uint64_t bytes_avoided;
    uint64_t io_wait_ns;
} NativeExpertResolutionMetrics;

typedef int64_t (*NativeExpertPreadFn)(
    void *context,
    int fd,
    void *buffer,
    uint64_t size_bytes,
    uint64_t offset_bytes);

typedef struct NativeUniqueExpertResolution {
    NativeResolvedExpert experts[N_EXPERTS];
    NativeExpertResolutionMetrics metrics;
    uintptr_t owner_engine;
    uintptr_t self_cookie;
    uint32_t count;
    uint32_t active;
} NativeUniqueExpertResolution;

typedef struct NativeKVStagingView {
    void *kv_data;
    void *kv_gpu_buffer;
    uint64_t generation;
    uint32_t layer;
    uint32_t start_position;
    uint32_t row_count;
    uint32_t row_stride_bytes;
    uint32_t reserved;
} NativeKVStagingView;

// Task 5.3 extends VERIFY transactions with an opaque recurrent-state journal.
// The journal snapshots compressor/indexer state before any row is staged and
// records bounded per-row deltas, so commit can expose exactly an accepted
// prefix without retaining B+1 full recurrent snapshots. Request cursor,
// hidden, sampler/RNG, and external-emission state remain owned by the Zig
// request transaction at the NativeEngine boundary.
typedef struct NativeKVTransaction {
    uint64_t generation;
    uintptr_t owner_engine;
    uintptr_t self_cookie;
    uint64_t completed_layer_mask;
    int original_kv_lengths[N_LAYERS];
    void *verify_state;
    uint32_t phase;
    uint32_t start_position;
    uint32_t staged_tokens;
    uint32_t accepted_tokens;
    uint32_t active;
    uint32_t reserved;
} NativeKVTransaction;

int moe_infer_batch_config_validate(const NativeBatchConfig *config);
// Install the validated startup config exactly once. The enabled Wave-4 path
// owns one bounded persistent workspace; all-zero expert flags allocate none.
int moe_infer_set_batch_config(
    MoEInferEngine *engine,
    const NativeBatchConfig *config);
void moe_infer_get_batch_path_counters(
    const MoEInferEngine *engine,
    uint64_t *out_scalar_layer_calls,
    uint64_t *out_batched_moe_layers);
int moe_infer_batch_workspace_init(
    NativeBatchWorkspace *workspace,
    uint32_t capacity_tokens,
    uint64_t allocated_bytes,
    uint64_t budget_bytes);
void moe_infer_batch_workspace_reset(NativeBatchWorkspace *workspace);
void *moe_infer_batch_workspace_reserve(
    NativeBatchWorkspace *workspace,
    uint64_t size_bytes,
    uint64_t alignment);
void moe_infer_batch_workspace_deinit(NativeBatchWorkspace *workspace);
int moe_infer_expert_bucket_workspace_bytes(
    uint32_t batch_tokens,
    uint64_t *out_bytes);
int moe_infer_expert_bucket_workspace_bind(
    NativeBatchWorkspace *workspace,
    uint32_t batch_tokens,
    NativeExpertBucketWorkspace *out_view);
int moe_infer_expert_bucket_workspace_reset(
    NativeExpertBucketWorkspace *view);
// Stable CPU counting sort over the token-major activation pairs. On success,
// bucketed_pairs is expert-major, unique_expert_ids is ascending, and
// inverse_mapping[source_pair] identifies the corresponding bucket row.
// Malformed pairs fail before any bucket metadata or payload is modified.
int moe_infer_expert_bucket_build(
    NativeExpertBucketWorkspace *view);
// Task 4.1 explicit primitive; forwardBatch does not call it before Task 4.5.
// The function stages hidden_rows as aligned GPU-visible [B,DIM] shared-expert
// input and gathers one expert's stable bucket as aligned [M,DIM]. Output
// pointers are published only after a successful, bounds-checked dispatch and
// remain valid until the owner workspace is reset or deinitialized.
int moe_infer_bucket_input_gather_f32(
    MoEInferEngine *engine,
    const float *hidden_rows,
    const NativeExpertBucketWorkspace *buckets,
    uint32_t expert_id,
    float **out_shared_rows,
    float **out_bucket_rows,
    uint32_t *out_bucket_count);
// CPU oracle for Task 4.4. It reconstructs token-major route-rank order from
// inverse_mapping, accumulates all six routed rows, then adds shared output.
// All metadata and inputs are validated before output publication.
int moe_infer_expert_bucket_combine_reference_f32(
    const NativeExpertBucketWorkspace *buckets,
    const float *bucket_outputs,
    const float *shared_outputs,
    uint32_t hidden_dim,
    float *combined_outputs);
// Explicit Task 4.4 GPU primitives. forwardBatch consumes them only in 4.5.
int moe_infer_shared_expert_batch_f32(
    MoEInferEngine *engine,
    int layer,
    const float *shared_inputs,
    uint32_t batch_tokens,
    NativeBatchWorkspace *workspace,
    float **out_shared_outputs);
int moe_infer_bucket_combine_f32(
    MoEInferEngine *engine,
    const NativeExpertBucketWorkspace *buckets,
    const float *bucket_outputs,
    const float *shared_outputs,
    uint32_t hidden_dim,
    float **out_combined_outputs);
// Allocate the bounded persistent per-chunk pool outside the layer hot loop.
// Existing storage is reused; an incompatible or undersized live pool fails.
int moe_infer_batch_pool_prepare(
    MoEInferEngine *engine,
    uint32_t capacity);
void moe_infer_unique_expert_resolution_init(
    NativeUniqueExpertResolution *resolution);
// Resolve the ascending unique IDs emitted by the bucket builder exactly once.
// reader may be NULL for system pread; tests inject deterministic short reads.
int moe_infer_unique_expert_resolve(
    MoEInferEngine *engine,
    uint32_t layer,
    const NativeExpertBucketWorkspace *buckets,
    NativeUniqueExpertResolution *resolution,
    NativeExpertPreadFn reader,
    void *reader_context);
int moe_infer_unique_expert_release(
    MoEInferEngine *engine,
    NativeUniqueExpertResolution *resolution);
// Deterministic score-based routing. The GPU function dispatches one 256-thread
// group per token into persistent engine scratch, then publishes exactly B*6
// token-major activation pairs only after validating every route. The reference
// function uses scalar CPU routing with identical bias, SMELT and tie semantics.
int moe_infer_expert_bucket_route_batch(
    MoEInferEngine *engine,
    const float *logits,
    const float *bias,
    const uint8_t *cached,
    float smelt_penalty,
    uint32_t has_bias,
    uint32_t has_smelt,
    NativeExpertBucketWorkspace *view);
int moe_infer_expert_bucket_route_batch_reference(
    const float *logits,
    const float *bias,
    const uint8_t *cached,
    float smelt_penalty,
    uint32_t has_bias,
    uint32_t has_smelt,
    NativeExpertBucketWorkspace *view);
void moe_infer_kv_transaction_init(NativeKVTransaction *transaction);
int moe_infer_kv_transaction_begin(
    MoEInferEngine *engine,
    const NativeBatchConfig *config,
    NativeBatchPhase phase,
    const uint32_t *positions,
    uint32_t batch_tokens,
    const NativeBatchWorkspace *workspace,
    NativeKVTransaction *transaction);
int moe_infer_kv_transaction_layer_view(
    MoEInferEngine *engine,
    NativeKVTransaction *transaction,
    uint32_t layer,
    NativeKVStagingView *out_view);
int moe_infer_kv_transaction_validate_layer_view(
    const MoEInferEngine *engine,
    const NativeKVTransaction *transaction,
    uint32_t layer,
    const NativeKVStagingView *view);
int moe_infer_kv_transaction_mark_layer_staged(
    MoEInferEngine *engine,
    NativeKVTransaction *transaction,
    uint32_t layer,
    uint32_t staged_tokens);
// Record the compressor/indexer delta produced by one completed VERIFY row.
// Rows must be recorded in order before the layer is marked staged. PREFILL and
// DECODE transactions reject this API rather than silently changing semantics.
int moe_infer_kv_transaction_record_recurrent_row(
    MoEInferEngine *engine,
    NativeKVTransaction *transaction,
    uint32_t layer,
    uint32_t row);
int moe_infer_kv_transaction_commit(
    MoEInferEngine *engine,
    NativeKVTransaction *transaction,
    uint32_t accepted_tokens);
int moe_infer_kv_transaction_abort(
    MoEInferEngine *engine,
    NativeKVTransaction *transaction);

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
// Enable the bounded observed-hot-seeded per-layer LRU experiment. Must be
// configured before SMELT warmup completes; false preserves fixed-hot behavior.
int moe_infer_set_experimental_cache(MoEInferEngine *engine, bool enabled);

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

// Install or remove the opt-in checkpoint callback. Call only while inference
// is idle. Passing callback=NULL disables capture and clears context. The
// existing scalar and batch forward APIs are unchanged.
void moe_infer_set_debug_capture(
    MoEInferEngine *engine,
    MoeInferDebugCaptureFn callback,
    void *context);

int moe_infer_decode_timeline_configure(MoEInferEngine *engine, uint32_t capacity);
int moe_infer_decode_timeline_enable(MoEInferEngine *engine, bool enabled);
bool moe_infer_decode_timeline_recording_enabled(const MoEInferEngine *engine);
uint64_t moe_infer_decode_timeline_now_ns(void);
int moe_infer_decode_timeline_begin_request(MoEInferEngine *engine);
int moe_infer_decode_timeline_begin_token(MoEInferEngine *engine, uint32_t token);
int moe_infer_decode_timeline_append(
    MoEInferEngine *engine, const DmlxDecodeTimelineEvent *event);
int moe_infer_decode_timeline_finalize(MoEInferEngine *engine);
int moe_infer_decode_timeline_snapshot(
    const MoEInferEngine *engine, DmlxDecodeTimelineSnapshot *snapshot,
    DmlxDecodeTimelineEvent *events, uint32_t capacity);
int moe_infer_decode_timeline_reset(MoEInferEngine *engine);
int moe_infer_decode_timeline_deconfigure(MoEInferEngine *engine);

// Exact-Q4 producer storage and lifecycle. The request thread serializes
// configure/enable/begin/append/reserve/finalize/snapshot/reset/deconfigure;
// snapshot must not overlap those lifecycle calls. Configure allocates bounded
// storage once and reset reuses it. Each successful read reservation must reach
// exactly one completion or abort terminal state. Completion/abort atomically
// claims the slot, release-publishes its terminal state, and releases its owner.
// Deinit aborts claimable slots and waits for any in-progress publication before
// freeing storage.
int moe_infer_exact_q4_configure(
    MoEInferEngine *engine, uint32_t stage_capacity, uint32_t read_capacity);
int moe_infer_exact_q4_enable(MoEInferEngine *engine, bool enabled);
bool moe_infer_exact_q4_recording_enabled(const MoEInferEngine *engine);
int moe_infer_exact_q4_begin_request(
    MoEInferEngine *engine, uint32_t request_ordinal);
int moe_infer_exact_q4_begin_token(
    MoEInferEngine *engine, uint32_t token_ordinal);
int moe_infer_exact_q4_append_stage(
    MoEInferEngine *engine, const DmlxExactQ4StageEvent *event,
    uint32_t *out_event_ordinal);
int moe_infer_exact_q4_commit_layer(
    MoEInferEngine *engine, const DmlxExactQ4StageEvent *stages,
    uint32_t stage_count, const DmlxMlaPathEvent *path);
int moe_infer_exact_q4_reserve_read(
    MoEInferEngine *engine, const DmlxExactQ4ExpertReadEvent *event,
    uint32_t *out_event_ordinal);
int moe_infer_exact_q4_publish_read_worker_start(
    MoEInferEngine *engine, uint32_t event_ordinal, uint64_t worker_start_ns);
int moe_infer_exact_q4_publish_read_completion(
    MoEInferEngine *engine, uint32_t event_ordinal, uint64_t complete_ns,
    uint64_t read_bytes, int32_t error_code);
int moe_infer_exact_q4_abort_read(
    MoEInferEngine *engine, uint32_t event_ordinal);
int moe_infer_exact_q4_abort_request(MoEInferEngine *engine);
int moe_infer_exact_q4_drain_and_abort_request(MoEInferEngine *engine);
int moe_infer_exact_q4_finalize(MoEInferEngine *engine);
int moe_infer_exact_q4_snapshot(
    const MoEInferEngine *engine, DmlxExactQ4LatencySnapshot *snapshot,
    DmlxExactQ4StageEvent *stages, uint32_t stage_capacity,
    DmlxExactQ4ExpertReadEvent *reads, uint32_t read_capacity);
int moe_infer_exact_q4_snapshot_mla_paths(
    const MoEInferEngine *engine, DmlxExactQ4MlaPathSnapshot *snapshot,
    DmlxMlaPathEvent *paths, uint32_t path_capacity);
int moe_infer_exact_q4_reset(MoEInferEngine *engine);
int moe_infer_exact_q4_deconfigure(MoEInferEngine *engine);

#ifdef DMLX_EXACT_Q4_PRODUCER_TESTING
// Runs deterministic coverage of the production scalar layer transaction.
int moe_infer_exact_q4_test_run_layer_transaction_suite(void);
#endif

#ifdef DMLX_EXACT_Q4_IO_TESTING
// Runs deterministic, no-model coverage of scalar expert-source and I/O owner
// lifecycles. Returns zero on success or the failing source line.
int moe_infer_exact_q4_test_run_io_suite(void);
#endif

#ifdef DMLX_EXACT_Q4_PRODUCER_TESTING
typedef uint64_t (*DmlxExactQ4TestClockFn)(void *context);
void moe_infer_exact_q4_test_set_clock(
    MoEInferEngine *engine, DmlxExactQ4TestClockFn clock_fn, void *context);
uint64_t moe_infer_exact_q4_test_clock_now(MoEInferEngine *engine);
void moe_infer_exact_q4_test_seed_drops(
    MoEInferEngine *engine, uint32_t dropped_stages, uint32_t dropped_reads);
void moe_infer_exact_q4_test_seed_read_state(
    MoEInferEngine *engine, uint32_t event_ordinal, uint32_t state);
const void *moe_infer_exact_q4_test_storage(const MoEInferEngine *engine);
size_t moe_infer_exact_q4_test_allocation_bytes(const MoEInferEngine *engine);
bool moe_infer_exact_q4_test_no_inflight_owners(MoEInferEngine *engine);
#endif

// Default-off raw Metal timestamp probe. Enabling allocates one fixed
// 4096-record store; append never allocates or overwrites. All probe APIs,
// including append, snapshot, reset, disable and deconfigure, are serialized
// by the same engine/request owner thread that first enables the probe; calls
// from another thread fail (or return an empty snapshot). They must not overlap
// engine teardown. Snapshot is non-destructive and returns records in sequence
// order. Passing out=NULL queries the retained count. If capacity is short, the
// return value is the copied count and total_seen remains the authoritative
// attempted-append count.
int moe_infer_gpu_timestamp_probe_set_enabled(
    MoEInferEngine *engine, bool enabled);
bool moe_infer_gpu_timestamp_probe_recording_enabled(
    const MoEInferEngine *engine);
int moe_infer_gpu_timestamp_probe_reset(MoEInferEngine *engine);
size_t moe_infer_gpu_timestamp_probe_snapshot(
    const MoEInferEngine *engine, DmlxGpuTimestampSample *out, size_t capacity,
    uint64_t *total_seen, uint64_t *overflow_count);
int moe_infer_gpu_timestamp_probe_deconfigure(MoEInferEngine *engine);

#ifdef DMLX_EXACT_Q4_PRODUCER_TESTING
int moe_infer_gpu_timestamp_probe_test_append(
    MoEInferEngine *engine, const DmlxGpuTimestampSample *sample);
int moe_infer_gpu_timestamp_probe_test_record_raw(
    MoEInferEngine *engine, uint16_t kind, uint16_t layer,
    uint32_t fallback_token_index, uint64_t pre_commit_ns,
    uint64_t post_commit_ns, uint64_t post_wait_ns,
    bool selectors_available, double gpu_start_s, double gpu_end_s);
void moe_infer_gpu_timestamp_probe_test_corrupt_accounting(
    MoEInferEngine *engine, uint64_t total_seen, uint64_t overflow_count);
void moe_infer_gpu_timestamp_probe_test_corrupt_sample(
    MoEInferEngine *engine, uint32_t index, uint16_t kind, uint32_t reason);
bool moe_infer_gpu_timestamp_test_round_scaled_ns(
    long double scaled_ns, uint64_t *rounded_ns);
#endif

// Telemetry control and snapshot APIs. The feature defaults off and must be
// enabled explicitly before reset/set_phase/snapshot/emit calls have effect.
uint64_t moe_infer_monotonic_ns(void);
void moe_infer_telemetry_enable(MoEInferEngine *engine, bool enabled);
int moe_infer_attention_stage_configure(MoEInferEngine *engine, bool enabled);
void moe_infer_attention_stage_reset(MoEInferEngine *engine);
bool moe_infer_attention_stage_recording_enabled(const MoEInferEngine *engine);
void *moe_infer_attention_stage_encoder(
    MoEInferEngine *engine, void *command_buffer, DmlxAttentionStage stage);
void moe_infer_attention_stage_resolve(
    MoEInferEngine *engine, const DmlxMlaPathIdentity *identity);
void moe_infer_attention_stage_snapshot(
    const MoEInferEngine *engine, DmlxAttentionStageSnapshot *snapshot);
void moe_infer_attention_stage_deconfigure(MoEInferEngine *engine);
#ifdef DMLX_EXACT_Q4_PRODUCER_TESTING
int moe_infer_attention_stage_test_enable(MoEInferEngine *engine);
int moe_infer_attention_stage_test_aggregate(
    MoEInferEngine *engine, const uint64_t *samples, size_t count,
    bool topology_valid);
void moe_infer_attention_stage_test_seed_sum(
    MoEInferEngine *engine, uint32_t stage, uint64_t sum, uint64_t count);
#endif
void moe_infer_telemetry_reset(MoEInferEngine *engine);
void moe_infer_telemetry_set_phase(MoEInferEngine *engine, NativeTelemetryPhase phase);
void moe_infer_telemetry_snapshot(
    const MoEInferEngine *engine,
    NativeRequestMetrics *out_metrics);
void moe_infer_telemetry_emit_json(
    const MoEInferEngine *engine,
    uint64_t request_start_ns,
    uint64_t first_token_ns,
    uint64_t decode_start_ns,
    uint64_t request_end_ns,
    uint32_t prompt_tokens,
    uint32_t generated_tokens,
    const uint64_t *token_timestamps_ns,
    size_t recorded_token_count,
    uint64_t dspark_candidate_count,
    uint64_t dspark_accepted_count,
    uint64_t dspark_verification_batches,
    uint64_t dspark_verification_ns);

// Privacy-safe routing trace control. Configure allocates one bounded writer
// and opens a 0600 JSONL file. Passing no configuration leaves routing_trace
// NULL, so disabled routing has no per-token allocation or clock read.
int moe_infer_routing_trace_configure(
    MoEInferEngine *engine,
    const char *path,
    size_t record_capacity,
    uint64_t model_fingerprint,
    uint32_t cache_capacity);
int moe_infer_routing_trace_request_begin(MoEInferEngine *engine);
void moe_infer_routing_trace_set_phase(
    MoEInferEngine *engine,
    NativeTelemetryPhase phase);
int moe_infer_routing_trace_request_end(MoEInferEngine *engine);

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
// Legacy layer-major batch entry retained for Task 5.3/5.4 verification rollout.
int moe_infer_forward_batch(MoEInferEngine *engine, float *hidden_batch, int n_tokens, int start_pos, const int *token_ids);

// Task 5.2 explicit prefill-only v2 entry. This requires all four prefill/MLA/
// expert flags, executes one authenticated all-layer KV transaction, calls the
// complete batched MLA v2 boundary once per layer, and never retries after
// mutation. On failure the caller must reset the request before reuse.
int moe_infer_forward_prefill_batch_v2(
    MoEInferEngine *engine,
    float *hidden_batch,
    int n_tokens,
    int start_pos,
    const int *token_ids);

// Task 5.4 transactional verify-v2 entry. On success the caller-owned VERIFY
// transaction remains active; the caller must commit exactly anchor+accepted
// rows or abort before any token is externally published.
int moe_infer_forward_verify_batch_v2(
    MoEInferEngine *engine,
    float *hidden_batch,
    int n_tokens,
    int start_pos,
    const int *token_ids,
    NativeKVTransaction *transaction);

// Embed token_id into hidden state.
// hidden_out: [MHC_MULT * DIM] — all MHC streams set to embed[token_id].
void moe_infer_embed(MoEInferEngine *engine, int token_id, float *hidden_out);
// Register quantized embed/lm_head (native 4-bit affine). When lm_head_q is
// set and lm_head is NULL, logits go through the GPU dequant matvec.
void moe_infer_set_head_quant(MoEInferEngine *eng, QuantWeight embed_q, QuantWeight lm_head_q);

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

// DSpark INT8 MoE GPU forward (called from dspark_engine.c)
void dspark_moe_forward_gpu(
    MoEInferEngine *eng,
    const float *normed_input,
    uint8_t *expert_ptrs[6],
    int *expert_ids,
    float *expert_weights,
    int K,
    float *moe_out
);

// Enable/disable DSpark hidden state accumulation (disable during verification)
void moe_infer_set_dspark_accumulate(MoEInferEngine *eng, bool enabled);
// Commit accepted draft positions after a verify batch (backfill main_kv +
// correct buf_main_hidden for the next draft). start_pos = anchor position.
// Returns NativeBatchStatus; mutation begins only after commit preflight.
int moe_infer_dspark_commit(MoEInferEngine *eng, int n_accepted, int start_pos);

// DSpark GPU-accelerated shared expert forward (f32 weights)
void dspark_shared_expert_gpu(
    MoEInferEngine *eng,
    const float *input,
    const float *w1_f32,
    const float *w3_f32,
    const float *w2_f32,
    float *output
);

// DSpark GPU f32 matvec (generic, for attention projections)
void dspark_matvec_f32_gpu(
    MoEInferEngine *eng,
    const float *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim
);

// DSpark batched MoE: all positions in ONE command buffer (eliminates 15× sync)
void dspark_moe_forward_batched_gpu(
    MoEInferEngine *eng,
    int n_pos,
    const float *normed_inputs,
    uint8_t **all_expert_ptrs,
    float *all_expert_weights,
    float *all_moe_out
);

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

// Bounded sequential variants used by the explicit Wave-2 chunk boundary.
// They force recurrent projections through the CPU implementation so no
// per-token Metal dispatch or engine-owned mutable GPU scratch escapes the
// caller-supplied workspace. Persistent compressor/indexer state remains
// engine-owned and is transactionally snapshotted by the caller.
int moe_infer_compressor_step_bounded(
    MoEInferEngine *engine, int layer, int pos, const float *attn_normed,
    float *scratch, size_t scratch_floats);
int moe_infer_indexer_step_bounded(
    MoEInferEngine *engine, int layer, int pos, const float *attn_normed,
    bool *allowed_out, float *scratch, size_t scratch_floats);

// (Private helper exposed for testing) Run indexer to get allowed comp block mask.
// Returns false if no selection needed (n_comp <= index_topk).
bool moe_infer_indexer_step(MoEInferEngine *engine, int layer, int pos,
                             const float *attn_normed, const float *q_a_out,
                             bool *allowed_out);

#ifdef __cplusplus
}
#endif
