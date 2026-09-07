// Zig bindings for the Metal inference engine.
// Bridges MLX model loading to the C engine, providing kernel source via @embedFile.
const std = @import("std");

const moe_metal_source = @embedFile("../models/moe_kernel.metal");

// Opaque engine handle
pub const Engine = opaque {};

extern fn moe_infer_init(
    packed_dir: [*c]const u8,
    kernel_src: [*c]const u8,
    kernel_src_len: usize,
) ?*Engine;

extern fn moe_infer_set_weights(
    engine: *Engine,
    embed: [*c]const f32,
    vocab_size: c_int,
    lm_head: [*c]const f32,
    final_norm: [*c]const f32,
    input_norms: [*c][*c]const f32,
    attn_norms: [*c][*c]const f32,
    gate_projs: [*c][*c]const f32,
    gate_biases: [*c][*c]const f32,
) void;

extern fn moe_infer_set_head_quant(engine: *Engine, embed_q: CQuantWeight, lm_head_q: CQuantWeight) void;

extern fn moe_infer_forward(engine: *Engine, hidden: [*c]f32, pos: c_int) c_int;
extern fn moe_infer_forward_layer(engine: *Engine, layer: c_int, hidden: [*c]f32, pos: c_int) c_int;
extern fn moe_infer_forward_batch(engine: *Engine, hidden_batch: [*c]f32, n_tokens: c_int, start_pos: c_int, token_ids: [*c]const c_int) c_int;
extern fn moe_infer_forward_prefill_batch_v2(engine: *Engine, hidden_batch: [*c]f32, n_tokens: c_int, start_pos: c_int, token_ids: [*c]const c_int) c_int;

pub const DebugCaptureRecord = extern struct {
    path: c_int,
    checkpoint: c_int,
    dtype: c_int,
    layer: c_int,
    position: c_int,
    kv_length: c_int,
    compressor_length: u32,
    indexer_length: u32,
    kv_state_digest: u64,
    compressor_state_digest: u64,
    indexer_state_digest: u64,
    values: [*c]const f32,
    value_count: usize,
    expert_ids: [*c]const c_int,
    expert_count: usize,
};
pub const DebugCaptureCallback = *const fn (
    context: ?*anyopaque,
    record: [*c]const DebugCaptureRecord,
) callconv(.c) void;
extern fn moe_infer_set_debug_capture(
    engine: *Engine,
    callback: ?DebugCaptureCallback,
    context: ?*anyopaque,
) void;

pub const TELEMETRY_PHASE_COUNT = 3;
pub const TELEMETRY_LAYER_COUNT = 43;
pub const ROUTING_EXPERT_COUNT = 256;
pub const ROUTING_ACTIVE_COUNT = 6;
pub const TelemetryPhase = enum(c_int) {
    prefill = 0,
    decode = 1,
    verify = 2,
};

pub const NativeLayerMetrics = extern struct {
    layer: u32,
    phase: u32,
    calls: u64,
    unique_experts: u64,
    hot_hits: u64,
    prefetch_hits: u64,
    batch_pool_hits: u64,
    cold_misses: u64,
    pread_calls: u64,
    bytes_requested: u64,
    bytes_read: u64,
    short_reads: u64,
    io_wait_ns: u64,
    bytes_avoided: u64,
    cache_admissions: u64,
    cache_evictions: u64,
    moe_dispatches: u64,
    moe_command_buffers: u64,
    threadgroups: u64,
    scheduled_lanes: u64,
    active_lanes: u64,
    max_threads_per_threadgroup: u32,
    reserved: u32,
};

pub const NativeRequestMetrics = extern struct {
    schema_version: u32,
    enabled: u32,
    active_phase: u32,
    reserved: u32,
    valid_fields: u64,
    layers: [TELEMETRY_PHASE_COUNT][TELEMETRY_LAYER_COUNT]NativeLayerMetrics,
};

pub const DECODE_TIMELINE_ABI_VERSION: u32 = 2;
pub const DECODE_TIMELINE_MAX_CAPACITY: u32 = 1_048_576;
pub const DECODE_TIMELINE_NO_LAYER: u16 = std.math.maxInt(u16);
pub const DecodeTimelineStage = enum(u8) {
    token_forward = 0,
    attention_cb = 1,
    dsa_cpu = 2,
    cb2cmd2_cb = 3,
    io_exposed_wait = 4,
    shared_expert_cb = 5,
    routed_moe_cb = 6,
    cb3_cb = 7,
    logits_cb = 8,
    token_sample = 9,
};
pub const DecodeTimelineEvent = extern struct {
    token: u32 = 0,
    layer: u16 = DECODE_TIMELINE_NO_LAYER,
    stage: u8,
    status: u8 = 1,
    begin_ns: u64,
    encode_done_ns: u64,
    pre_commit_ns: u64,
    post_commit_ns: u64,
    post_wait_ns: u64,
    end_ns: u64,
};
pub const DecodeTimelineSnapshot = extern struct {
    abi_version: u32 = 0,
    valid: u32 = 0,
    enabled: u32 = 0,
    finalized: u32 = 0,
    capacity: u32 = 0,
    count: u32 = 0,
    dropped: u32 = 0,
    incomplete: u32 = 0,
    active_publishers: u32 = 0,
    reserved: u32 = 0,
    events: [*c]DecodeTimelineEvent = null,
};

extern fn moe_infer_decode_timeline_configure(engine: *Engine, capacity: u32) c_int;
extern fn moe_infer_decode_timeline_enable(engine: *Engine, enabled: bool) c_int;
extern fn moe_infer_decode_timeline_recording_enabled(engine: *const Engine) bool;
extern fn moe_infer_decode_timeline_now_ns() u64;
extern fn moe_infer_decode_timeline_begin_request(engine: *Engine) c_int;
extern fn moe_infer_decode_timeline_begin_token(engine: *Engine, token: u32) c_int;
extern fn moe_infer_decode_timeline_append(engine: *Engine, event: *const DecodeTimelineEvent) c_int;
extern fn moe_infer_decode_timeline_finalize(engine: *Engine) c_int;
extern fn moe_infer_decode_timeline_snapshot(engine: *const Engine, snapshot: *DecodeTimelineSnapshot, events: [*c]DecodeTimelineEvent, capacity: u32) c_int;
extern fn moe_infer_decode_timeline_reset(engine: *Engine) c_int;
extern fn moe_infer_decode_timeline_deconfigure(engine: *Engine) c_int;

pub const EXACT_Q4_LATENCY_EVENT_ABI_VERSION: u32 = 1;
pub const ExactQ4Stage = enum(u16) {
    attention_encode = 0,
    attention_gpu_execution = 1,
    attention_gpu_wait = 2,
    routing = 3,
    io_queue = 4,
    io_pread = 5,
    io_exposed_wait = 6,
    moe_encode = 7,
    moe_gpu_execution = 8,
    moe_gpu_wait = 9,
    cache_admission = 10,
    host_alloc_wrapper = 11,
    logits_sample = 12,
    other_accounted = 13,
};
pub const EXACT_Q4_EVENT_PARENT_ONLY: u32 = 1;
pub const EXACT_Q4_EVENT_GPU_TIME_KNOWN: u32 = 2;
pub const EXACT_Q4_EVENT_GPU_TIME_UNKNOWN: u32 = 4;
pub const EXACT_Q4_EVENT_JOIN_BLOCKED: u32 = 8;
pub const EXACT_Q4_EVENT_CPU_ACTIVE: u32 = 16;
pub const EXACT_Q4_EVENT_RAW_DIAGNOSTIC: u32 = 32;
pub const EXACT_Q4_EVENT_VALID_FLAGS: u32 = 63;

pub const ExactQ4ReadSource = enum(u8) {
    fixed_hot = 0,
    prefetch = 1,
    batch_pool = 2,
    pread = 3,
};
pub const EXACT_Q4_READ_COMPLETED: u32 = 1;
pub const EXACT_Q4_READ_VALID_FLAGS: u32 = 1;
pub const EXACT_Q4_MLA_PATH_ABI_VERSION: u32 = 1;
pub const ExactQ4MlaPath = enum(u8) {
    persistent_external = 0,
    persistent_internal = 1,
    persistent_internal_split = 2,
    fallback_temp_kv = 3,
};
pub const EXACT_Q4_MLA_PERSISTENT_KV: u8 = 1;
pub const EXACT_Q4_MLA_EXTERNAL_CB_REQUESTED: u8 = 2;
pub const EXACT_Q4_MLA_EXTERNAL_CB_USED: u8 = 4;
pub const EXACT_Q4_MLA_FALLBACK: u8 = 8;
pub const EXACT_Q4_MLA_STAGE_TIME_SPLIT: u8 = 16;

pub const MlaPathIdentity = extern struct {
    path: u8 = 0,
    flags: u8 = 0,
    reserved: u16 = 0,
};

pub const MlaPathEvent = extern struct {
    request_ordinal: u32 = 0,
    token_ordinal: u32 = 0,
    event_ordinal: u32 = 0,
    layer: u16 = 0,
    path: u8 = 0,
    flags: u8 = 0,
};

pub const ExactQ4StageEvent = extern struct {
    request_ordinal: u32,
    token_ordinal: u32,
    event_ordinal: u32,
    layer: u16,
    stage: u16,
    flags: u32,
    calls: u32,
    begin_ns: u64,
    end_ns: u64,
    bytes: u64,
};

pub const ExactQ4ExpertReadEvent = extern struct {
    request_ordinal: u32,
    token_ordinal: u32,
    event_ordinal: u32,
    layer: u16,
    expert: u16,
    route_rank: u8,
    source: u8,
    reserved: u16,
    error_code: i32,
    flags: u32,
    submit_ns: u64,
    worker_start_ns: u64,
    complete_ns: u64,
    requested_bytes: u64,
    read_bytes: u64,
};

pub const EXACT_Q4_MAX_STAGE_CAPACITY: u32 = 1_048_576;
pub const EXACT_Q4_MAX_READ_CAPACITY: u32 = 1_048_576;
pub const ExactQ4LatencySnapshot = extern struct {
    abi_version: u32 = 0,
    valid: u32 = 0,
    enabled: u32 = 0,
    finalized: u32 = 0,
    request_ordinal: u32 = 0,
    token_ordinal: u32 = 0,
    stage_capacity: u32 = 0,
    read_capacity: u32 = 0,
    stage_count: u32 = 0,
    read_count: u32 = 0,
    dropped_stages: u32 = 0,
    dropped_reads: u32 = 0,
    incomplete: u32 = 0,
    active_read_publishers: u32 = 0,
    stages: [*c]ExactQ4StageEvent = null,
    reads: [*c]ExactQ4ExpertReadEvent = null,
};

pub const ExactQ4MlaPathSnapshot = extern struct {
    abi_version: u32 = 0,
    valid: u32 = 0,
    enabled: u32 = 0,
    finalized: u32 = 0,
    request_ordinal: u32 = 0,
    token_ordinal: u32 = 0,
    capacity: u32 = 0,
    count: u32 = 0,
    dropped: u32 = 0,
    reserved: u32 = 0,
    paths: [*c]MlaPathEvent = null,
};

extern fn moe_infer_exact_q4_configure(engine: *Engine, stage_capacity: u32, read_capacity: u32) c_int;
extern fn moe_infer_exact_q4_enable(engine: *Engine, enabled: bool) c_int;
extern fn moe_infer_exact_q4_recording_enabled(engine: *const Engine) bool;
extern fn moe_infer_exact_q4_begin_request(engine: *Engine, request_ordinal: u32) c_int;
extern fn moe_infer_exact_q4_begin_token(engine: *Engine, token_ordinal: u32) c_int;
extern fn moe_infer_exact_q4_append_stage(engine: *Engine, event: *const ExactQ4StageEvent, out_event_ordinal: *u32) c_int;
extern fn moe_infer_exact_q4_commit_layer(engine: *Engine, stages: [*c]const ExactQ4StageEvent, stage_count: u32, path: *const MlaPathEvent) c_int;
extern fn moe_infer_exact_q4_reserve_read(engine: *Engine, event: *const ExactQ4ExpertReadEvent, out_event_ordinal: *u32) c_int;
extern fn moe_infer_exact_q4_publish_read_worker_start(engine: *Engine, event_ordinal: u32, worker_start_ns: u64) c_int;
extern fn moe_infer_exact_q4_publish_read_completion(engine: *Engine, event_ordinal: u32, complete_ns: u64, read_bytes: u64, error_code: i32) c_int;
extern fn moe_infer_exact_q4_abort_read(engine: *Engine, event_ordinal: u32) c_int;
extern fn moe_infer_exact_q4_abort_request(engine: *Engine) c_int;
extern fn moe_infer_exact_q4_drain_and_abort_request(engine: *Engine) c_int;
extern fn moe_infer_exact_q4_finalize(engine: *Engine) c_int;
extern fn moe_infer_exact_q4_snapshot(engine: *const Engine, snapshot: *ExactQ4LatencySnapshot, stages: [*c]ExactQ4StageEvent, stage_capacity: u32, reads: [*c]ExactQ4ExpertReadEvent, read_capacity: u32) c_int;
extern fn moe_infer_exact_q4_snapshot_mla_paths(engine: *const Engine, snapshot: *ExactQ4MlaPathSnapshot, paths: [*c]MlaPathEvent, path_capacity: u32) c_int;
extern fn moe_infer_exact_q4_reset(engine: *Engine) c_int;
extern fn moe_infer_exact_q4_deconfigure(engine: *Engine) c_int;

extern fn moe_infer_monotonic_ns() u64;
extern fn moe_infer_telemetry_enable(engine: *Engine, enabled: bool) void;
extern fn moe_infer_attention_stage_configure(engine: *Engine, enabled: bool) c_int;
extern fn moe_infer_telemetry_reset(engine: *Engine) void;
extern fn moe_infer_telemetry_set_phase(engine: *Engine, phase: c_int) void;
extern fn moe_infer_telemetry_snapshot(
    engine: *const Engine,
    out_metrics: *NativeRequestMetrics,
) void;
extern fn moe_infer_telemetry_emit_json(
    engine: *const Engine,
    request_start_ns: u64,
    first_token_ns: u64,
    decode_start_ns: u64,
    request_end_ns: u64,
    prompt_tokens: u32,
    generated_tokens: u32,
    token_timestamps_ns: [*c]const u64,
    recorded_token_count: usize,
    dspark_candidate_count: u64,
    dspark_accepted_count: u64,
    dspark_verification_batches: u64,
    dspark_verification_ns: u64,
) void;

extern fn moe_infer_routing_trace_configure(
    engine: *Engine,
    path: [*c]const u8,
    record_capacity: usize,
    model_fingerprint: u64,
    cache_capacity: u32,
) c_int;
extern fn moe_infer_routing_trace_request_begin(engine: *Engine) c_int;
extern fn moe_infer_routing_trace_set_phase(engine: *Engine, phase: c_int) void;
extern fn moe_infer_routing_trace_request_end(engine: *Engine) c_int;

extern fn moe_infer_deinit(engine: *Engine) void;

// C-compatible structs (must match engine.h layout exactly).
pub const CQuantWeight = extern struct {
    packed_ptr: [*c]const u32,
    scales: [*c]const f32,
    biases: [*c]const f32,
    out_dim: c_int,
    in_dim: c_int,
    group_size: c_int,
};
pub const CAttnWeights = extern struct {
    wq_a: CQuantWeight,
    q_norm: [*c]const f32,
    wq_b: CQuantWeight,
    wkv: CQuantWeight,
    kv_norm: [*c]const f32,
    wo_a_dense: [*c]const f32,
    wo_a: CQuantWeight,
    wo_b: CQuantWeight,
    attn_sink: [*c]const f32,
};
extern fn moe_infer_set_layer_attn(engine: *Engine, layer: c_int, attn: CAttnWeights) void;
extern fn moe_infer_set_layer_hc(
    engine: *Engine,
    layer: c_int,
    attn_fn: [*c]const f32,
    attn_base: [*c]const f32,
    attn_scale: [*c]const f32,
    ffn_fn: [*c]const f32,
    ffn_base: [*c]const f32,
    ffn_scale: [*c]const f32,
) void;

const CSharedExpert = extern struct {
    gate: CQuantWeight,
    up: CQuantWeight,
    down: CQuantWeight,
};
extern fn moe_infer_set_layer_shared(engine: *Engine, layer: c_int, se: CSharedExpert) void;

pub const BatchPhase = enum(c_int) {
    prefill = 0,
    verify = 1,
    decode = 2,
};

pub const NativeBatchConfig = extern struct {
    enable_batch_prefill: u32 = 0,
    enable_batch_mla: u32 = 0,
    enable_expert_buckets: u32 = 0,
    enable_batched_expert_kernels: u32 = 0,
    enable_dspark_batch_verify: u32 = 0,
    max_prefill_chunk: u32 = 8,
    max_verify_chunk: u32 = 5,
    reserved: u32 = 0,
    workspace_budget_bytes: u64 = 256 * 1024 * 1024,
};

// Task 5.1 read-only runtime snapshot derived from the same immutable config
// installed into the opaque C engine. The C implementation allocates exactly
// 8 MiB + 512 KiB/token and min(256, max_prefill*6) exact-Q4 pool slots.
pub const BATCH_HARD_TOKEN_LIMIT: u32 = 8;
pub const BATCH_ACTIVE_EXPERTS: u32 = 6;
pub const BATCH_ROUTED_EXPERTS: u32 = 256;
pub const BATCH_EXACT_Q4_EXPERT_BYTES: u64 = 13_369_344;
pub const BATCH_WORKSPACE_FIXED_BYTES: u64 = 8 * 1024 * 1024;
pub const BATCH_WORKSPACE_PER_TOKEN_BYTES: u64 = 512 * 1024;

pub const BatchRuntimeCapacity = struct {
    workspace_capacity_tokens: u32,
    supported_capacity_tokens: u32,
    workspace_allocated_bytes: u64,
    workspace_budget_bytes: u64,
    batch_pool_slots: u32,
    expert_slot_bytes: u64,
};

pub fn batchRuntimeCapacity(config: *const NativeBatchConfig) BatchRuntimeCapacity {
    const expert_path = config.enable_expert_buckets == 1 and
        config.enable_batched_expert_kernels == 1;
    const workspace_bytes = if (expert_path)
        BATCH_WORKSPACE_FIXED_BYTES +
            @as(u64, config.max_prefill_chunk) * BATCH_WORKSPACE_PER_TOKEN_BYTES
    else
        0;
    const pool_slots = if (expert_path)
        @min(BATCH_ROUTED_EXPERTS, config.max_prefill_chunk * BATCH_ACTIVE_EXPERTS)
    else
        0;
    return .{
        .workspace_capacity_tokens = if (expert_path) config.max_prefill_chunk else 1,
        .supported_capacity_tokens = if (expert_path)
            @min(config.max_prefill_chunk, BATCH_HARD_TOKEN_LIMIT)
        else
            1,
        .workspace_allocated_bytes = workspace_bytes,
        .workspace_budget_bytes = config.workspace_budget_bytes,
        .batch_pool_slots = pool_slots,
        .expert_slot_bytes = BATCH_EXACT_Q4_EXPERT_BYTES,
    };
}

pub const NativeBatchWorkspaceMetalReleaseFn = *const fn (
    metal_buffer: ?*anyopaque,
) callconv(.c) void;

pub const NativeBatchWorkspace = extern struct {
    capacity_tokens: u32 = 0,
    arena_epoch: u32 = 0,
    budget_bytes: u64 = 0,
    allocated_bytes: u64 = 0,
    used_bytes: u64 = 0,
    high_water_bytes: u64 = 0,
    storage: ?*anyopaque = null,
    self_cookie: usize = 0,
    metal_buffer: ?*anyopaque = null,
    metal_buffer_offset: u64 = 0,
    metal_buffer_creations: u64 = 0,
    metal_buffer_release: ?NativeBatchWorkspaceMetalReleaseFn = null,
};

pub const NativeActivationPair = extern struct {
    token_id: u32 = 0,
    expert_id: u32 = 0,
    route_rank: u32 = 0,
    weight: f32 = 0,
};

pub const NativeExpertBucketWorkspace = extern struct {
    activation_pairs: [*c]NativeActivationPair = null,
    bucketed_pairs: [*c]NativeActivationPair = null,
    counts: [*c]u32 = null,
    offsets: [*c]u32 = null,
    unique_expert_ids: [*c]u32 = null,
    inverse_mapping: [*c]u32 = null,
    owner_workspace: ?*NativeBatchWorkspace = null,
    self_cookie: usize = 0,
    arena_offset_bytes: u64 = 0,
    arena_size_bytes: u64 = 0,
    batch_tokens: u32 = 0,
    pair_capacity: u32 = 0,
    unique_capacity: u32 = 0,
    unique_count: u32 = 0,
    arena_epoch: u32 = 0,
};

pub const NativeKVStagingView = extern struct {
    kv_data: ?*anyopaque = null,
    kv_gpu_buffer: ?*anyopaque = null,
    generation: u64 = 0,
    layer: u32 = 0,
    start_position: u32 = 0,
    row_count: u32 = 0,
    row_stride_bytes: u32 = 0,
    reserved: u32 = 0,
};

pub const NativeKVTransaction = extern struct {
    generation: u64 = 0,
    owner_engine: usize = 0,
    self_cookie: usize = 0,
    completed_layer_mask: u64 = 0,
    original_kv_lengths: [TELEMETRY_LAYER_COUNT]c_int = [_]c_int{0} ** TELEMETRY_LAYER_COUNT,
    verify_state: ?*anyopaque = null,
    phase: u32 = 0,
    start_position: u32 = 0,
    staged_tokens: u32 = 0,
    accepted_tokens: u32 = 0,
    active: u32 = 0,
    reserved: u32 = 0,
};

pub const MlaPipes = opaque {};

pub const MLAPrefillBatchArgs = extern struct {
    hidden: ?*const anyopaque = null,
    attention_output: ?*anyopaque = null,
    batch_tokens: u32 = 0,
    layer: u32 = 0,
    positions: [*c]const u32 = null,
    kv: NativeKVStagingView = .{},
    kv_transaction: ?*NativeKVTransaction = null,
    workspace: ?*NativeBatchWorkspace = null,
};

pub const MLAAttentionMetrics = extern struct {
    q_ns: u64 = 0,
    kv_rope_ns: u64 = 0,
    sdpa_ns: u64 = 0,
    output_ns: u64 = 0,
    workspace_high_water_bytes: u64 = 0,
    dispatches: u32 = 0,
    operator_order_signature: u32 = 0,
};

pub const MLAAttentionCaptureFn = *const fn (
    context: ?*anyopaque,
    layer: u32,
    position: u32,
    attention_bf16: [*c]const u16,
    element_count: u32,
) callconv(.c) void;

pub const MoeInferStateSnapshot = extern struct {
    kv_length: c_int = 0,
    compressor_length: u32 = 0,
    indexer_length: u32 = 0,
    kv_state_digest: u64 = 0,
    compressor_state_digest: u64 = 0,
    indexer_state_digest: u64 = 0,
};

pub const MLARecurrentBatchState = extern struct {
    residual_rows: [*c]f32 = null,
    post_rows: [*c]const f32 = null,
    comb_rows: [*c]const f32 = null,
    attention_capture: [*c]u16 = null,
    pre_state_snapshots: [*c]MoeInferStateSnapshot = null,
    post_state_snapshots: [*c]MoeInferStateSnapshot = null,
    capture: ?MLAAttentionCaptureFn = null,
    capture_context: ?*anyopaque = null,
    capacity_tokens: u32 = 0,
    completed_tokens: u32 = 0,
    inject_failure_after_row: u32 = std.math.maxInt(u32),
    reserved: u32 = 0,
    self_cookie: usize = 0,
};

comptime {
    if (@sizeOf(DecodeTimelineEvent) != 56) @compileError("DecodeTimelineEvent C ABI drift");
    if (@offsetOf(DecodeTimelineEvent, "begin_ns") != 8) @compileError("DecodeTimelineEvent timing ABI drift");
    if (@sizeOf(DecodeTimelineSnapshot) != 48) @compileError("DecodeTimelineSnapshot C ABI drift");
    if (@offsetOf(DecodeTimelineSnapshot, "events") != 40) @compileError("DecodeTimelineSnapshot events ABI drift");
    if (@sizeOf(NativeLayerMetrics) != 168) @compileError("NativeLayerMetrics C ABI drift");
    if (@offsetOf(NativeLayerMetrics, "cache_admissions") != 104) @compileError("NativeLayerMetrics cache ABI drift");
    if (@offsetOf(NativeLayerMetrics, "moe_dispatches") != 120) @compileError("NativeLayerMetrics dispatch ABI drift");
    if (@sizeOf(NativeRequestMetrics) != 21696) @compileError("NativeRequestMetrics C ABI drift");
    if (@sizeOf(ExactQ4StageEvent) != 48) @compileError("ExactQ4StageEvent C ABI drift");
    if (@offsetOf(ExactQ4StageEvent, "flags") != 16) @compileError("ExactQ4StageEvent flags ABI drift");
    if (@offsetOf(ExactQ4StageEvent, "begin_ns") != 24) @compileError("ExactQ4StageEvent timing ABI drift");
    if (@offsetOf(ExactQ4StageEvent, "bytes") != 40) @compileError("ExactQ4StageEvent bytes ABI drift");
    if (@sizeOf(ExactQ4ExpertReadEvent) != 72) @compileError("ExactQ4ExpertReadEvent C ABI drift");
    if (@offsetOf(ExactQ4ExpertReadEvent, "error_code") != 20) @compileError("ExactQ4ExpertReadEvent error ABI drift");
    if (@offsetOf(ExactQ4ExpertReadEvent, "submit_ns") != 32) @compileError("ExactQ4ExpertReadEvent timing ABI drift");
    if (@offsetOf(ExactQ4ExpertReadEvent, "requested_bytes") != 56) @compileError("ExactQ4ExpertReadEvent bytes ABI drift");
    if (@sizeOf(MlaPathIdentity) != 4) @compileError("MlaPathIdentity C ABI drift");
    if (@sizeOf(MlaPathEvent) != 16) @compileError("MlaPathEvent C ABI drift");
    if (@offsetOf(MlaPathEvent, "layer") != 12) @compileError("MlaPathEvent layer ABI drift");
    if (@sizeOf(ExactQ4LatencySnapshot) != 72) @compileError("ExactQ4LatencySnapshot C ABI drift");
    if (@offsetOf(ExactQ4LatencySnapshot, "stages") != 56) @compileError("ExactQ4LatencySnapshot stage pointer ABI drift");
    if (@sizeOf(ExactQ4MlaPathSnapshot) != 48) @compileError("ExactQ4MlaPathSnapshot C ABI drift");
    if (@offsetOf(ExactQ4MlaPathSnapshot, "paths") != 40) @compileError("ExactQ4MlaPathSnapshot paths ABI drift");
    if (@sizeOf(NativeBatchConfig) != 40) @compileError("NativeBatchConfig C ABI drift");
    if (@sizeOf(NativeBatchWorkspace) != 88) @compileError("NativeBatchWorkspace C ABI drift");
    if (@sizeOf(NativeActivationPair) != 16) @compileError("NativeActivationPair C ABI drift");
    if (@sizeOf(NativeExpertBucketWorkspace) != 104) @compileError("NativeExpertBucketWorkspace C ABI drift");
    if (@sizeOf(NativeKVStagingView) != 48) @compileError("NativeKVStagingView C ABI drift");
    if (@sizeOf(NativeKVTransaction) != 240) @compileError("NativeKVTransaction C ABI drift");
    if (@sizeOf(MLAPrefillBatchArgs) != 96) @compileError("MLAPrefillBatchArgs C ABI drift");
    if (@sizeOf(MLAAttentionMetrics) != 48) @compileError("MLAAttentionMetrics C ABI drift");
    if (@sizeOf(MoeInferStateSnapshot) != 40) @compileError("MoeInferStateSnapshot C ABI drift");
    if (@sizeOf(MLARecurrentBatchState) != 88) @compileError("MLARecurrentBatchState C ABI drift");
}

extern fn moe_infer_batch_config_validate(config: *const NativeBatchConfig) c_int;
extern fn moe_infer_set_batch_config(
    engine: *Engine,
    config: *const NativeBatchConfig,
) c_int;
extern fn moe_infer_get_batch_path_counters(
    engine: *const Engine,
    out_scalar_layer_calls: *u64,
    out_batched_moe_layers: *u64,
) void;
extern fn moe_infer_batch_workspace_init(
    workspace: *NativeBatchWorkspace,
    capacity_tokens: u32,
    allocated_bytes: u64,
    budget_bytes: u64,
) c_int;
extern fn moe_infer_batch_workspace_reset(workspace: *NativeBatchWorkspace) void;
extern fn moe_infer_batch_workspace_reserve(
    workspace: *NativeBatchWorkspace,
    size_bytes: u64,
    alignment: u64,
) ?*anyopaque;
extern fn moe_infer_batch_workspace_deinit(workspace: *NativeBatchWorkspace) void;
extern fn moe_infer_expert_bucket_workspace_bytes(
    batch_tokens: u32,
    out_bytes: *u64,
) c_int;
extern fn moe_infer_expert_bucket_workspace_bind(
    workspace: *NativeBatchWorkspace,
    batch_tokens: u32,
    out_view: *NativeExpertBucketWorkspace,
) c_int;
extern fn moe_infer_expert_bucket_workspace_reset(
    view: *NativeExpertBucketWorkspace,
) c_int;
extern fn moe_infer_expert_bucket_route_batch(
    engine: *Engine,
    logits: [*c]const f32,
    bias: [*c]const f32,
    cached: [*c]const u8,
    smelt_penalty: f32,
    has_bias: u32,
    has_smelt: u32,
    view: *NativeExpertBucketWorkspace,
) c_int;
extern fn moe_infer_expert_bucket_route_batch_reference(
    logits: [*c]const f32,
    bias: [*c]const f32,
    cached: [*c]const u8,
    smelt_penalty: f32,
    has_bias: u32,
    has_smelt: u32,
    view: *NativeExpertBucketWorkspace,
) c_int;
extern fn moe_infer_kv_transaction_init(transaction: *NativeKVTransaction) void;
extern fn moe_infer_kv_transaction_begin(
    engine: *Engine,
    config: *const NativeBatchConfig,
    phase: c_int,
    positions: [*c]const u32,
    batch_tokens: u32,
    workspace: *const NativeBatchWorkspace,
    transaction: *NativeKVTransaction,
) c_int;
extern fn moe_infer_kv_transaction_layer_view(
    engine: *Engine,
    transaction: *NativeKVTransaction,
    layer: u32,
    out_view: *NativeKVStagingView,
) c_int;
extern fn moe_infer_kv_transaction_mark_layer_staged(
    engine: *Engine,
    transaction: *NativeKVTransaction,
    layer: u32,
    staged_tokens: u32,
) c_int;
extern fn moe_infer_kv_transaction_record_recurrent_row(
    engine: *Engine,
    transaction: *NativeKVTransaction,
    layer: u32,
    row: u32,
) c_int;
extern fn moe_infer_kv_transaction_commit(
    engine: *Engine,
    transaction: *NativeKVTransaction,
    accepted_tokens: u32,
) c_int;
extern fn moe_infer_kv_transaction_abort(
    engine: *Engine,
    transaction: *NativeKVTransaction,
) c_int;
extern fn moe_infer_forward_verify_batch_v2(
    engine: *Engine,
    hidden_batch: [*c]f32,
    n_tokens: c_int,
    start_pos: c_int,
    token_ids: [*c]const c_int,
    transaction: *NativeKVTransaction,
) c_int;

pub extern fn mla_recurrent_batch_state_init(
    state: *MLARecurrentBatchState,
    residual_rows: [*c]f32,
    post_rows: [*c]const f32,
    comb_rows: [*c]const f32,
    attention_capture: [*c]u16,
    capacity_tokens: u32,
) void;
pub extern fn mla_attention_prefill_batch_workspace_bytes(
    engine: *const Engine,
    args: *const MLAPrefillBatchArgs,
    state: *const MLARecurrentBatchState,
    out_bytes: *u64,
) c_int;
pub extern fn mla_attention_kv_stage_workspace_bytes(
    rows: u32,
    out_bytes: *u64,
) c_int;
pub extern fn mla_attention_sdpa_stage_workspace_bytes(
    rows: u32,
    committed: u32,
    out_bytes: *u64,
) c_int;
pub extern fn mla_attention_workspace_region_end(
    offset: u64,
    size_bytes: u64,
    out_end: *u64,
) c_int;
pub extern fn mla_attention_output_batch(
    pipes: *MlaPipes,
    weights: *const CAttnWeights,
    positions: [*c]const u32,
    sdpa_input: [*c]const u16,
    attention_output: [*c]u16,
    batch_tokens: u32,
    workspace: *NativeBatchWorkspace,
    metrics: ?*MLAAttentionMetrics,
) c_int;
pub extern fn mla_attention_prefill_batch_v2(
    engine: *Engine,
    pipes: *MlaPipes,
    weights: *const CAttnWeights,
    args: *const MLAPrefillBatchArgs,
    state: *MLARecurrentBatchState,
    metrics: ?*MLAAttentionMetrics,
) c_int;

pub fn validateBatchConfig(config: *const NativeBatchConfig) !void {
    if (moe_infer_batch_config_validate(config) != 0) return error.InvalidBatchConfig;
}

pub fn setBatchConfig(
    engine: *Engine,
    config: *const NativeBatchConfig,
) !void {
    const status = moe_infer_set_batch_config(engine, config);
    switch (status) {
        0 => {},
        -2 => return error.BatchCapacityExceeded,
        -4 => return error.InvalidBatchState,
        -5 => return error.BatchWorkspaceBudgetExceeded,
        else => return error.InvalidBatchConfig,
    }
}

pub const BatchPathCounters = struct {
    scalar_layer_calls: u64,
    batched_moe_layers: u64,
};

pub fn batchPathCounters(engine: *const Engine) BatchPathCounters {
    var counters = BatchPathCounters{
        .scalar_layer_calls = 0,
        .batched_moe_layers = 0,
    };
    moe_infer_get_batch_path_counters(
        engine,
        &counters.scalar_layer_calls,
        &counters.batched_moe_layers,
    );
    return counters;
}

pub fn initBatchWorkspace(
    workspace: *NativeBatchWorkspace,
    capacity_tokens: u32,
    allocated_bytes: u64,
    budget_bytes: u64,
) !void {
    const status = moe_infer_batch_workspace_init(
        workspace,
        capacity_tokens,
        allocated_bytes,
        budget_bytes,
    );
    if (status == -5) return error.BatchWorkspaceBudgetExceeded;
    if (status != 0) return error.InvalidBatchWorkspace;
}

pub fn resetBatchWorkspace(workspace: *NativeBatchWorkspace) void {
    moe_infer_batch_workspace_reset(workspace);
}

pub fn reserveBatchWorkspace(
    workspace: *NativeBatchWorkspace,
    size_bytes: u64,
    alignment: u64,
) !*anyopaque {
    return moe_infer_batch_workspace_reserve(workspace, size_bytes, alignment) orelse
        error.BatchWorkspaceCapacityExceeded;
}

pub fn deinitBatchWorkspace(workspace: *NativeBatchWorkspace) void {
    moe_infer_batch_workspace_deinit(workspace);
}

pub fn expertBucketWorkspaceBytes(batch_tokens: u32) !u64 {
    var bytes: u64 = 0;
    const status = moe_infer_expert_bucket_workspace_bytes(batch_tokens, &bytes);
    if (status == -2) return error.BatchWorkspaceCapacityExceeded;
    if (status != 0) return error.InvalidBatchWorkspace;
    return bytes;
}

pub fn bindExpertBucketWorkspace(
    workspace: *NativeBatchWorkspace,
    batch_tokens: u32,
    view: *NativeExpertBucketWorkspace,
) !void {
    const status = moe_infer_expert_bucket_workspace_bind(
        workspace,
        batch_tokens,
        view,
    );
    if (status == -2) return error.BatchWorkspaceCapacityExceeded;
    if (status != 0) return error.InvalidBatchWorkspace;
}

pub fn resetExpertBucketWorkspace(view: *NativeExpertBucketWorkspace) !void {
    if (moe_infer_expert_bucket_workspace_reset(view) != 0)
        return error.InvalidExpertBucketWorkspace;
}

pub fn routeExpertBucketBatch(
    engine: *Engine,
    logits: []const f32,
    bias: ?[]const f32,
    cached: ?[]const u8,
    smelt_penalty: f32,
    view: *NativeExpertBucketWorkspace,
) !void {
    if (logits.len != @as(usize, view.batch_tokens) * ROUTING_EXPERT_COUNT or
        (bias != null and bias.?.len != ROUTING_EXPERT_COUNT) or
        (cached != null and cached.?.len != ROUTING_EXPERT_COUNT))
    {
        return error.InvalidBatchRoutes;
    }
    const status = moe_infer_expert_bucket_route_batch(
        engine,
        logits.ptr,
        if (bias) |values| values.ptr else null,
        if (cached) |values| values.ptr else null,
        smelt_penalty,
        @intFromBool(bias != null),
        @intFromBool(cached != null),
        view,
    );
    if (status != 0) return error.InvalidBatchRoutes;
}

pub fn routeExpertBucketBatchReference(
    logits: []const f32,
    bias: ?[]const f32,
    cached: ?[]const u8,
    smelt_penalty: f32,
    view: *NativeExpertBucketWorkspace,
) !void {
    if (logits.len != @as(usize, view.batch_tokens) * ROUTING_EXPERT_COUNT or
        (bias != null and bias.?.len != ROUTING_EXPERT_COUNT) or
        (cached != null and cached.?.len != ROUTING_EXPERT_COUNT))
    {
        return error.InvalidBatchRoutes;
    }
    const status = moe_infer_expert_bucket_route_batch_reference(
        logits.ptr,
        if (bias) |values| values.ptr else null,
        if (cached) |values| values.ptr else null,
        smelt_penalty,
        @intFromBool(bias != null),
        @intFromBool(cached != null),
        view,
    );
    if (status != 0) return error.InvalidBatchRoutes;
}

pub fn initKvTransaction(transaction: *NativeKVTransaction) void {
    moe_infer_kv_transaction_init(transaction);
}

pub fn beginKvTransaction(
    engine: *Engine,
    config: *const NativeBatchConfig,
    phase: BatchPhase,
    positions: []const u32,
    workspace: *const NativeBatchWorkspace,
    transaction: *NativeKVTransaction,
) !void {
    const status = moe_infer_kv_transaction_begin(
        engine,
        config,
        @intFromEnum(phase),
        positions.ptr,
        @intCast(positions.len),
        workspace,
        transaction,
    );
    switch (status) {
        0 => {},
        -2 => return error.BatchCapacityExceeded,
        -3 => return error.InvalidBatchPositions,
        -4 => return error.InvalidKvState,
        -5 => return error.BatchWorkspaceBudgetExceeded,
        else => return error.InvalidBatchTransaction,
    }
}

pub fn kvTransactionLayerView(
    engine: *Engine,
    transaction: *NativeKVTransaction,
    layer: u32,
) !NativeKVStagingView {
    var view = NativeKVStagingView{};
    if (moe_infer_kv_transaction_layer_view(engine, transaction, layer, &view) != 0)
        return error.InvalidKvStagingView;
    return view;
}

pub fn markKvLayerStaged(
    engine: *Engine,
    transaction: *NativeKVTransaction,
    layer: u32,
    staged_tokens: u32,
) !void {
    if (moe_infer_kv_transaction_mark_layer_staged(
        engine,
        transaction,
        layer,
        staged_tokens,
    ) != 0) return error.KvLayerStageFailed;
}

pub fn recordKvRecurrentRow(
    engine: *Engine,
    transaction: *NativeKVTransaction,
    layer: u32,
    row: u32,
) !void {
    if (moe_infer_kv_transaction_record_recurrent_row(
        engine,
        transaction,
        layer,
        row,
    ) != 0) return error.KvRecurrentStageFailed;
}

pub fn commitKvTransaction(
    engine: *Engine,
    transaction: *NativeKVTransaction,
    accepted_tokens: u32,
) !void {
    if (moe_infer_kv_transaction_commit(engine, transaction, accepted_tokens) != 0)
        return error.KvTransactionCommitFailed;
}

pub fn abortKvTransaction(engine: *Engine, transaction: *NativeKVTransaction) !void {
    if (moe_infer_kv_transaction_abort(engine, transaction) != 0)
        return error.KvTransactionAbortFailed;
}

extern fn moe_infer_reset_kv(engine: *Engine) void;
extern fn moe_infer_rollback_kv(engine: *Engine, valid_len: c_int) void;
extern fn moe_infer_set_layer_tid2eid(engine: *Engine, layer: c_int, tid2eid: [*c]const i64) void;
extern fn moe_infer_set_token_id(engine: *Engine, token_id: c_int) void;
extern fn moe_infer_embed(engine: *Engine, token_id: c_int, hidden_out: [*c]f32) void;
extern fn moe_infer_compress_hc(engine: *Engine, residual: [*c]const f32, out: [*c]f32) void;
extern fn hyper_head_compress(attn_fn: [*c]const f32, attn_base: [*c]const f32, attn_scale: [*c]const f32, residual: [*c]const f32, out: [*c]f32) void;
extern fn moe_infer_get_logits(engine: *Engine, hidden: [*c]const f32, logits_out: [*c]f32) c_int;
extern fn moe_infer_set_layer_compressor(engine: *Engine, layer: c_int, compress_ratio: u32, comp_wkv: CQuantWeight, comp_wgate: CQuantWeight, comp_ape: ?[*]const f32, comp_norm: ?[*]const f32) void;
extern fn moe_infer_set_layer_indexer(engine: *Engine, layer: c_int, idx_wq_b: CQuantWeight, idx_weights_proj: CQuantWeight, idx_comp_wkv: CQuantWeight, idx_comp_wgate: CQuantWeight, idx_comp_ape: ?[*]const f32, idx_comp_norm: ?[*]const f32) void;

extern fn moe_infer_preload_experts(engine: *Engine, expert_cache_mb: c_int) c_int;
extern fn moe_infer_smelt_init(engine: *Engine, warmup_tokens: c_int, n_per_layer: c_int, penalty: f32) void;
extern fn moe_infer_smelt_finish_warmup(engine: *Engine) c_int;
extern fn moe_infer_smelt_preload_async(engine: *Engine) void;
extern fn moe_infer_smelt_set_decode_phase(engine: *Engine) void;
extern fn moe_infer_init_gather_mode(engine: *Engine) c_int;
extern fn moe_infer_smelt_save_stats(engine: *Engine, path: [*c]const u8) void;
extern fn moe_infer_smelt_load_stats(engine: *Engine, path: [*c]const u8) c_int;
extern fn moe_infer_smelt_set_penalty(engine: *Engine, penalty: f32) void;
extern fn moe_infer_smelt_set_stats_path(engine: *Engine, path: [*c]const u8) void;
extern fn moe_infer_set_experimental_cache(engine: *Engine, enabled: bool) c_int;

pub fn resetKv(engine: *Engine) void {
    moe_infer_reset_kv(engine);
}

pub fn rollbackKv(engine: *Engine, valid_len: i32) void {
    moe_infer_rollback_kv(engine, @intCast(valid_len));
}

pub fn preloadExperts(engine: *Engine, expert_cache_mb: i32) i32 {
    return moe_infer_preload_experts(engine, @intCast(expert_cache_mb));
}

pub fn smeltInit(engine: *Engine, warmup_tokens: i32, n_per_layer: i32, penalty: f32) void {
    moe_infer_smelt_init(engine, @intCast(warmup_tokens), @intCast(n_per_layer), penalty);
}

pub fn smeltFinishWarmup(engine: *Engine) i32 {
    return moe_infer_smelt_finish_warmup(engine);
}

pub fn smeltPreloadAsync(engine: *Engine) void {
    moe_infer_smelt_preload_async(engine);
}

pub fn smeltSetDecodePhase(engine: *Engine) void {
    moe_infer_smelt_set_decode_phase(engine);
}

pub fn smeltSaveStats(engine: *Engine, path: [*c]const u8) void {
    moe_infer_smelt_save_stats(engine, path);
}

pub fn smeltLoadStats(engine: *Engine, path: [*c]const u8) i32 {
    return moe_infer_smelt_load_stats(engine, path);
}

pub fn smeltSetPenalty(engine: *Engine, penalty: f32) void {
    moe_infer_smelt_set_penalty(engine, penalty);
}

pub fn smeltSetStatsPath(engine: *Engine, path: [*c]const u8) void {
    moe_infer_smelt_set_stats_path(engine, path);
}

pub fn setExperimentalCache(engine: *Engine, enabled: bool) !void {
    if (moe_infer_set_experimental_cache(engine, enabled) != 0)
        return error.InvalidExperimentalCacheConfiguration;
}

pub fn initGatherMode(engine: *Engine) i32 {
    return moe_infer_init_gather_mode(engine);
}
pub fn setTokenId(engine: *Engine, token_id: i32) void {
    moe_infer_set_token_id(engine, @intCast(token_id));
}
pub fn embed(engine: *Engine, token_id: i32, hidden_out: [*c]f32) void {
    moe_infer_embed(engine, @intCast(token_id), hidden_out);
}
pub fn compressHc(engine: *Engine, residual: [*c]const f32, out: [*c]f32) void {
    moe_infer_compress_hc(engine, residual, out);
}
pub fn hyperHeadCompress(attn_fn: [*c]const f32, attn_base: [*c]const f32, attn_scale: [*c]const f32, residual: [*c]const f32, out: [*c]f32) void {
    hyper_head_compress(attn_fn, attn_base, attn_scale, residual, out);
}
pub fn getLogits(engine: *Engine, hidden: [*c]const f32, logits_out: [*c]f32) !void {
    const rc = moe_infer_get_logits(engine, hidden, logits_out);
    if (rc != 0) return error.GetLogitsFailed;
}

pub fn init(packed_dir: []const u8) !*Engine {
    const engine = moe_infer_init(packed_dir.ptr, moe_metal_source, moe_metal_source.len);
    if (engine == null) return error.InitFailed;
    return engine.?;
}

pub fn forward(engine: *Engine, hidden: []f32, pos: u32) !void {
    const rc = moe_infer_forward(engine, hidden.ptr, @intCast(pos));
    if (rc != 0) return error.ForwardFailed;
}

pub fn forwardBatch(engine: *Engine, hidden_batch: []f32, n_tokens: usize, start_pos: u32, token_ids: []const i32) !void {
    const rc = moe_infer_forward_batch(engine, hidden_batch.ptr, @intCast(n_tokens), @intCast(start_pos), token_ids.ptr);
    if (rc != 0) return error.ForwardFailed;
}

/// Task 5.2 guarded request-flow boundary. Planning and host allocation must
/// finish before this call; once C begins, failure is propagated and is never
/// retried against partially mutated KV/recurrent state.
pub fn forwardPlannedBatch(
    engine: *Engine,
    hidden_batch: []f32,
    n_tokens: usize,
    start_pos: u32,
    token_ids: []const i32,
) !void {
    const hidden_values_per_token = 4 * 4096;
    if (n_tokens == 0 or n_tokens > BATCH_HARD_TOKEN_LIMIT or
        token_ids.len != n_tokens or
        hidden_batch.len != n_tokens * hidden_values_per_token)
    {
        return error.InvalidPlannedBatch;
    }
    const status = moe_infer_forward_prefill_batch_v2(
        engine,
        hidden_batch.ptr,
        @intCast(n_tokens),
        @intCast(start_pos),
        token_ids.ptr,
    );
    switch (status) {
        0 => {},
        -2 => return error.BatchCapacityExceeded,
        -3 => return error.InvalidBatchPositions,
        -4 => return error.InvalidBatchState,
        -5 => return error.BatchWorkspaceBudgetExceeded,
        else => return error.ForwardPlannedBatchFailed,
    }
}

/// Task 5.4 verify-v2 entry. Planning and allocation must complete before this
/// call. Success leaves `transaction` active so the caller can commit the
/// anchor plus the accepted draft prefix after sampling target logits.
pub fn forwardVerifyBatch(
    engine: *Engine,
    hidden_batch: []f32,
    n_tokens: usize,
    start_pos: u32,
    token_ids: []const i32,
    transaction: *NativeKVTransaction,
) !void {
    const hidden_values_per_token = 4 * 4096;
    if (n_tokens < 2 or n_tokens > BATCH_HARD_TOKEN_LIMIT or
        token_ids.len != n_tokens or
        hidden_batch.len != n_tokens * hidden_values_per_token)
    {
        return error.InvalidVerifyBatch;
    }
    const status = moe_infer_forward_verify_batch_v2(
        engine,
        hidden_batch.ptr,
        @intCast(n_tokens),
        @intCast(start_pos),
        token_ids.ptr,
        transaction,
    );
    switch (status) {
        0 => if (transaction.active != 1) return error.InvalidVerifyBatchState,
        -2 => return error.BatchCapacityExceeded,
        -3 => return error.InvalidBatchPositions,
        -4 => return error.InvalidVerifyBatchState,
        -5 => return error.BatchWorkspaceBudgetExceeded,
        else => return error.ForwardVerifyBatchFailed,
    }
}

pub fn commitVerifyBatch(
    engine: *Engine,
    transaction: *NativeKVTransaction,
    accepted_rows: usize,
) !void {
    const status = moe_infer_kv_transaction_commit(engine, transaction, @intCast(accepted_rows));
    switch (status) {
        0 => {},
        -1 => return error.InvalidVerifyCommit,
        -4 => return error.InvalidVerifyBatchState,
        else => return error.VerifyCommitFailed,
    }
}

pub fn abortVerifyBatch(
    engine: *Engine,
    transaction: *NativeKVTransaction,
) !void {
    const status = moe_infer_kv_transaction_abort(engine, transaction);
    switch (status) {
        0 => {},
        -4 => return error.InvalidVerifyBatchState,
        else => return error.VerifyAbortFailed,
    }
}

pub fn forwardLayer(engine: *Engine, layer: usize, hidden: []f32, pos: usize) !void {
    const rc = moe_infer_forward_layer(engine, @intCast(layer), hidden.ptr, @intCast(pos));
    if (rc != 0) return error.ForwardFailed;
}

pub fn setDebugCapture(
    engine: *Engine,
    callback: ?DebugCaptureCallback,
    context: ?*anyopaque,
) void {
    moe_infer_set_debug_capture(engine, callback, context);
}

pub fn monotonicNs() u64 {
    return moe_infer_monotonic_ns();
}

pub fn decodeTimelineNowNs() u64 {
    return moe_infer_decode_timeline_now_ns();
}
fn decodeTimelineStatus(status: c_int) !void {
    switch (status) {
        0 => {},
        1 => return error.DecodeTimelineDisabled,
        2 => return error.DecodeTimelineBusy,
        -2 => return error.DecodeTimelineCapacityExceeded,
        -3 => return error.InvalidDecodeTimelineState,
        else => return error.InvalidDecodeTimelineArgument,
    }
}
pub fn decodeTimelineConfigure(engine: *Engine, capacity: u32) !void {
    try decodeTimelineStatus(moe_infer_decode_timeline_configure(engine, capacity));
}
pub fn decodeTimelineEnable(engine: *Engine, enabled: bool) !void {
    try decodeTimelineStatus(moe_infer_decode_timeline_enable(engine, enabled));
}
pub fn decodeTimelineRecordingEnabled(engine: *const Engine) bool {
    return moe_infer_decode_timeline_recording_enabled(engine);
}
pub fn decodeTimelineBeginRequest(engine: *Engine) !void {
    try decodeTimelineStatus(moe_infer_decode_timeline_begin_request(engine));
}
pub fn decodeTimelineBeginToken(engine: *Engine, token: u32) !void {
    try decodeTimelineStatus(moe_infer_decode_timeline_begin_token(engine, token));
}
pub fn decodeTimelineAppend(engine: *Engine, event: *const DecodeTimelineEvent) !void {
    try decodeTimelineStatus(moe_infer_decode_timeline_append(engine, event));
}
pub fn decodeTimelineFinalize(engine: *Engine) !void {
    try decodeTimelineStatus(moe_infer_decode_timeline_finalize(engine));
}
pub fn decodeTimelineSnapshot(engine: *const Engine, events: []DecodeTimelineEvent) !DecodeTimelineSnapshot {
    if (events.len > std.math.maxInt(u32)) return error.DecodeTimelineCapacityExceeded;
    var snapshot = DecodeTimelineSnapshot{};
    try decodeTimelineStatus(moe_infer_decode_timeline_snapshot(engine, &snapshot, if (events.len > 0) events.ptr else null, @intCast(events.len)));
    return snapshot;
}
pub fn decodeTimelineReset(engine: *Engine) !void {
    try decodeTimelineStatus(moe_infer_decode_timeline_reset(engine));
}
pub fn decodeTimelineDeconfigure(engine: *Engine) !void {
    try decodeTimelineStatus(moe_infer_decode_timeline_deconfigure(engine));
}

fn exactQ4Status(status: c_int) !void {
    switch (status) {
        0 => {},
        1 => return error.ExactQ4Disabled,
        2 => return error.ExactQ4Busy,
        -2 => return error.ExactQ4CapacityExceeded,
        -3 => return error.InvalidExactQ4State,
        else => return error.InvalidExactQ4Argument,
    }
}

pub fn exactQ4Configure(engine: *Engine, stage_capacity: u32, read_capacity: u32) !void {
    try exactQ4Status(moe_infer_exact_q4_configure(engine, stage_capacity, read_capacity));
}

pub fn exactQ4Enable(engine: *Engine, enabled: bool) !void {
    try exactQ4Status(moe_infer_exact_q4_enable(engine, enabled));
}

pub fn exactQ4RecordingEnabled(engine: *const Engine) bool {
    return moe_infer_exact_q4_recording_enabled(engine);
}

pub fn exactQ4BeginRequest(engine: *Engine, request_ordinal: u32) !void {
    try exactQ4Status(moe_infer_exact_q4_begin_request(engine, request_ordinal));
}

pub fn exactQ4BeginToken(engine: *Engine, token_ordinal: u32) !void {
    try exactQ4Status(moe_infer_exact_q4_begin_token(engine, token_ordinal));
}

pub fn exactQ4AppendStage(engine: *Engine, event: *const ExactQ4StageEvent) !u32 {
    var ordinal: u32 = 0;
    try exactQ4Status(moe_infer_exact_q4_append_stage(engine, event, &ordinal));
    return ordinal;
}

pub fn exactQ4CommitLayer(engine: *Engine, stages: []const ExactQ4StageEvent, path: *const MlaPathEvent) !void {
    if (stages.len > std.math.maxInt(u32))
        return error.ExactQ4CapacityExceeded;
    try exactQ4Status(moe_infer_exact_q4_commit_layer(engine, if (stages.len > 0) stages.ptr else null, @intCast(stages.len), path));
}

pub fn exactQ4ReserveRead(engine: *Engine, event: *const ExactQ4ExpertReadEvent) !u32 {
    var ordinal: u32 = 0;
    try exactQ4Status(moe_infer_exact_q4_reserve_read(engine, event, &ordinal));
    return ordinal;
}

pub fn exactQ4PublishReadWorkerStart(engine: *Engine, event_ordinal: u32, worker_start_ns: u64) !void {
    try exactQ4Status(moe_infer_exact_q4_publish_read_worker_start(engine, event_ordinal, worker_start_ns));
}

pub fn exactQ4PublishReadCompletion(engine: *Engine, event_ordinal: u32, complete_ns: u64, read_bytes: u64, error_code: i32) !void {
    try exactQ4Status(moe_infer_exact_q4_publish_read_completion(engine, event_ordinal, complete_ns, read_bytes, error_code));
}

pub fn exactQ4AbortRead(engine: *Engine, event_ordinal: u32) !void {
    try exactQ4Status(moe_infer_exact_q4_abort_read(engine, event_ordinal));
}

pub fn exactQ4AbortRequest(engine: *Engine) !void {
    try exactQ4Status(moe_infer_exact_q4_abort_request(engine));
}

pub fn exactQ4DrainAndAbortRequest(engine: *Engine) !void {
    try exactQ4Status(moe_infer_exact_q4_drain_and_abort_request(engine));
}

pub fn exactQ4Finalize(engine: *Engine) !void {
    try exactQ4Status(moe_infer_exact_q4_finalize(engine));
}

pub fn exactQ4Snapshot(
    engine: *const Engine,
    stages: []ExactQ4StageEvent,
    reads: []ExactQ4ExpertReadEvent,
) !ExactQ4LatencySnapshot {
    if (stages.len > std.math.maxInt(u32) or
        reads.len > std.math.maxInt(u32)) return error.ExactQ4CapacityExceeded;
    var snapshot = ExactQ4LatencySnapshot{};
    try exactQ4Status(moe_infer_exact_q4_snapshot(engine, &snapshot, if (stages.len > 0) stages.ptr else null, @intCast(stages.len), if (reads.len > 0) reads.ptr else null, @intCast(reads.len)));
    return snapshot;
}

pub fn exactQ4SnapshotMlaPaths(
    engine: *const Engine,
    paths: []MlaPathEvent,
) !ExactQ4MlaPathSnapshot {
    if (paths.len > std.math.maxInt(u32))
        return error.ExactQ4CapacityExceeded;
    var snapshot = ExactQ4MlaPathSnapshot{};
    try exactQ4Status(moe_infer_exact_q4_snapshot_mla_paths(engine, &snapshot, if (paths.len > 0) paths.ptr else null, @intCast(paths.len)));
    return snapshot;
}

pub fn exactQ4Reset(engine: *Engine) !void {
    try exactQ4Status(moe_infer_exact_q4_reset(engine));
}

pub fn exactQ4Deconfigure(engine: *Engine) !void {
    try exactQ4Status(moe_infer_exact_q4_deconfigure(engine));
}

pub fn telemetryEnable(engine: *Engine, enabled: bool) void {
    moe_infer_telemetry_enable(engine, enabled);
}

pub fn attentionStageConfigure(engine: *Engine, enabled: bool) i32 {
    return moe_infer_attention_stage_configure(engine, enabled);
}

pub fn telemetryReset(engine: *Engine) void {
    moe_infer_telemetry_reset(engine);
}

pub fn telemetrySetPhase(engine: *Engine, phase: TelemetryPhase) void {
    moe_infer_telemetry_set_phase(engine, @intFromEnum(phase));
}

pub fn telemetrySnapshot(engine: *Engine) NativeRequestMetrics {
    var metrics: NativeRequestMetrics = undefined;
    moe_infer_telemetry_snapshot(engine, &metrics);
    return metrics;
}

pub fn telemetryEmitJson(
    engine: *Engine,
    request_start_ns: u64,
    first_token_ns: u64,
    decode_start_ns: u64,
    request_end_ns: u64,
    prompt_tokens: u32,
    generated_tokens: u32,
    token_timestamps_ns: []const u64,
    dspark_candidate_count: u64,
    dspark_accepted_count: u64,
    dspark_verification_batches: u64,
    dspark_verification_ns: u64,
) void {
    moe_infer_telemetry_emit_json(
        engine,
        request_start_ns,
        first_token_ns,
        decode_start_ns,
        request_end_ns,
        prompt_tokens,
        generated_tokens,
        if (token_timestamps_ns.len > 0) token_timestamps_ns.ptr else null,
        token_timestamps_ns.len,
        dspark_candidate_count,
        dspark_accepted_count,
        dspark_verification_batches,
        dspark_verification_ns,
    );
}

pub fn routingTraceConfigure(
    engine: *Engine,
    path: [*:0]const u8,
    record_capacity: usize,
    model_fingerprint: u64,
    cache_capacity: u32,
) !void {
    if (moe_infer_routing_trace_configure(
        engine,
        path,
        record_capacity,
        model_fingerprint,
        cache_capacity,
    ) != 0) return error.RoutingTraceConfigureFailed;
}

pub fn routingTraceRequestBegin(engine: *Engine) !void {
    if (moe_infer_routing_trace_request_begin(engine) != 0)
        return error.RoutingTraceRequestBeginFailed;
}

pub fn routingTraceSetPhase(engine: *Engine, phase: TelemetryPhase) void {
    moe_infer_routing_trace_set_phase(engine, @intFromEnum(phase));
}

pub fn routingTraceRequestEnd(engine: *Engine) bool {
    return moe_infer_routing_trace_request_end(engine) == 0;
}

pub fn setWeights(engine: *Engine, w: anytype) void {
    var in_arr: [64][*c]const f32 = [_][*c]const f32{null} ** 64;
    var an_arr: [64][*c]const f32 = [_][*c]const f32{null} ** 64;
    var gp_arr: [64][*c]const f32 = [_][*c]const f32{null} ** 64;
    var gb_arr: [64][*c]const f32 = [_][*c]const f32{null} ** 64;
    for (0..@intCast(w.n_layers)) |i| {
        in_arr[i] = w.input_norms[i].ptr;
        an_arr[i] = w.attn_norms[i].ptr;
        gp_arr[i] = w.gate_projs[i].ptr;
        if (w.gate_biases[i]) |b| gb_arr[i] = b.ptr;
    }
    const dense_head = w.embed.len > 0;
    const vocab_size: i32 = if (dense_head) @intCast(w.embed.len / 4096) else if (@hasField(@TypeOf(w), "embed_q")) w.embed_q.out_dim else 0;
    moe_infer_set_weights(engine, if (dense_head) w.embed.ptr else null, vocab_size, if (dense_head) w.lm_head.ptr else null, w.final_norm.ptr, &in_arr, &an_arr, &gp_arr, &gb_arr);
    if (!dense_head and @hasField(@TypeOf(w), "embed_q")) {
        moe_infer_set_head_quant(engine, cqw(w.embed_q), cqw(w.lm_head_q));
    }

    // Per-layer MLA attention + mHC weights.
    for (0..@intCast(w.n_layers)) |i| {
        const ap = w.attn[i] orelse continue;
        var ca: CAttnWeights = undefined;
        ca.wq_a = cqw(ap.wq_a);
        ca.q_norm = ap.q_norm.ptr;
        ca.wq_b = cqw(ap.wq_b);
        ca.wkv = cqw(ap.wkv);
        ca.kv_norm = ap.kv_norm.ptr;
        ca.wo_a_dense = ap.wo_a_dense.ptr;
        // native_loader keeps wo_a packed (GPU dequants in-kernel); the older
        // DSV4Model loader only has the f32 dense form — leave wo_a empty so
        // the C side takes its f32 fallback.
        if (@hasField(@TypeOf(ap), "wo_a")) {
            ca.wo_a = cqw(ap.wo_a);
        } else {
            ca.wo_a = .{ .packed_ptr = null, .scales = null, .biases = null, .out_dim = 0, .in_dim = 0, .group_size = 64 };
        }
        ca.wo_b = cqw(ap.wo_b);
        ca.attn_sink = ap.attn_sink.ptr;
        moe_infer_set_layer_attn(engine, @intCast(i), ca);
        moe_infer_set_layer_hc(engine, @intCast(i), w.attn_hc_fn[i].ptr, w.attn_hc_base[i].ptr, w.attn_hc_scale[i].ptr, w.ffn_hc_fn[i].ptr, w.ffn_hc_base[i].ptr, w.ffn_hc_scale[i].ptr);
        if (w.tid2eid[i]) |t| moe_infer_set_layer_tid2eid(engine, @intCast(i), t.ptr);
        if (w.shared_gate[i].out_dim > 0) {
            const cse = CSharedExpert{
                .gate = cqw(w.shared_gate[i]),
                .up = cqw(w.shared_up[i]),
                .down = cqw(w.shared_down[i]),
            };
            moe_infer_set_layer_shared(engine, @intCast(i), cse);
        }
    }
}

fn cqw(q: anytype) CQuantWeight {
    return .{
        .packed_ptr = q.packed_ptr.ptr,
        .scales = q.scales.ptr,
        .biases = q.biases.ptr,
        .out_dim = q.out_dim,
        .in_dim = q.in_dim,
        .group_size = q.group_size,
    };
}

pub fn deinit(engine: *Engine) void {
    moe_infer_deinit(engine);
}

pub fn setLayerCompressor(engine: *Engine, layer: usize, compress_ratio: u32, comp_wkv: CQuantWeight, comp_wgate: CQuantWeight, comp_ape: ?[*]const f32, comp_norm: ?[*]const f32) void {
    moe_infer_set_layer_compressor(engine, @intCast(layer), compress_ratio, comp_wkv, comp_wgate, comp_ape, comp_norm);
}

pub fn setLayerIndexer(engine: *Engine, layer: usize, idx_wq_b: CQuantWeight, idx_weights_proj: CQuantWeight, idx_comp_wkv: CQuantWeight, idx_comp_wgate: CQuantWeight, idx_comp_ape: ?[*]const f32, idx_comp_norm: ?[*]const f32) void {
    moe_infer_set_layer_indexer(engine, @intCast(layer), idx_wq_b, idx_weights_proj, idx_comp_wkv, idx_comp_wgate, idx_comp_ape, idx_comp_norm);
}

pub fn toCQuantWeight(q: anytype) CQuantWeight {
    return .{
        .packed_ptr = q.packed_ptr.ptr,
        .scales = q.scales.ptr,
        .biases = if (q.biases.len > 0) q.biases.ptr else null,
        .out_dim = q.out_dim,
        .in_dim = q.in_dim,
        .group_size = q.group_size,
    };
}

// ============================================================================
// DSpark Engine Bindings
// ============================================================================

pub const DSparkEngine = opaque {};

extern fn dspark_init(
    dspark_weight_dir: [*c]const u8,
    packed_expert_dir: [*c]const u8,
    target_engine: *Engine,
) ?*DSparkEngine;

extern fn dspark_deinit(eng: *DSparkEngine) void;
extern fn dspark_reset(eng: *DSparkEngine) void;

extern fn dspark_forward(
    eng: *DSparkEngine,
    main_hidden: ?[*]const f32,
    anchor_token_id: c_int,
    start_pos: c_int,
    draft_logits: [*]f32,
    confidence: ?[*]f32,
) c_int;

extern fn dspark_markov_sample(
    eng: *DSparkEngine,
    draft_logits: [*]f32,
    anchor_token_id: c_int,
    corrected_logits: [*]f32,
    draft_tokens: [*]u32,
) c_int;

extern fn dspark_update_main_kv(
    eng: *DSparkEngine,
    target_kv_entry: [*c]const u16,
    pos: c_int,
) void;

// Set the dspark_engine pointer on the target engine (enables hidden state extraction)
extern fn moe_infer_set_dspark_engine(engine: *Engine, dspark: ?*DSparkEngine) void;

// --- Zig-friendly wrappers ---

pub fn dsparkInit(dspark_weight_dir: []const u8, packed_expert_dir: []const u8, target: *Engine) ?*DSparkEngine {
    return dspark_init(dspark_weight_dir.ptr, packed_expert_dir.ptr, target);
}

pub fn dsparkDeinit(eng: *DSparkEngine) void {
    dspark_deinit(eng);
}

pub fn dsparkReset(eng: *DSparkEngine) void {
    dspark_reset(eng);
}

pub fn dsparkForward(eng: *DSparkEngine, main_hidden: ?[]const f32, anchor_token_id: i32, start_pos: i32, draft_logits: []f32, confidence: ?[]f32) i32 {
    const mh = if (main_hidden) |h| h.ptr else null;
    const conf_ptr = if (confidence) |cf| cf.ptr else null;
    return dspark_forward(eng, mh, @intCast(anchor_token_id), @intCast(start_pos), draft_logits.ptr, conf_ptr);
}

pub fn dsparkMarkovSample(eng: *DSparkEngine, draft_logits: []f32, anchor_token_id: i32, corrected_logits: []f32, draft_tokens: []u32) i32 {
    return dspark_markov_sample(eng, draft_logits.ptr, @intCast(anchor_token_id), corrected_logits.ptr, draft_tokens.ptr);
}

pub fn dsparkUpdateMainKv(eng: *DSparkEngine, target_kv_entry: [*c]const u16, pos: i32) void {
    dspark_update_main_kv(eng, target_kv_entry, @intCast(pos));
}

pub fn setDSparkEngine(engine: *Engine, dspark: ?*DSparkEngine) void {
    moe_infer_set_dspark_engine(engine, dspark);
}

extern fn moe_infer_set_dspark_accumulate(engine: *Engine, enabled: bool) void;
pub fn setDSparkAccumulate(engine: *Engine, enabled: bool) void {
    moe_infer_set_dspark_accumulate(engine, enabled);
}

extern fn moe_infer_dspark_commit(engine: *Engine, n_accepted: i32, start_pos: i32) c_int;
pub fn dsparkCommit(engine: *Engine, n_accepted: i32, start_pos: i32) !void {
    const status = moe_infer_dspark_commit(engine, n_accepted, start_pos);
    switch (status) {
        0 => {},
        -1 => return error.InvalidDSparkCommit,
        -2 => return error.DSparkCommitAllocationFailed,
        -4 => return error.InvalidDSparkState,
        else => return error.DSparkCommitFailed,
    }
}
