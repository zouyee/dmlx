// Zig bindings for the Metal inference engine.
// Bridges MLX model loading to the C engine, providing kernel source via @embedFile.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;

const Array = array_mod.Array;

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

extern fn moe_infer_forward(engine: *Engine, hidden: [*c]f32, pos: c_int) c_int;
extern fn moe_infer_forward_layer(engine: *Engine, layer: c_int, hidden: [*c]f32, pos: c_int) c_int;
extern fn moe_infer_forward_batch(engine: *Engine, hidden_batch: [*c]f32, n_tokens: c_int, start_pos: c_int, token_ids: [*c]const c_int) c_int;
extern fn moe_infer_deinit(engine: *Engine) void;

// C-compatible structs (must match engine.h layout exactly).
const CQuantWeight = extern struct {
    packed_ptr: [*c]const u32,
    scales: [*c]const f32,
    biases: [*c]const f32,
    out_dim: c_int,
    in_dim: c_int,
    group_size: c_int,
};
const CAttnWeights = extern struct {
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

pub fn forwardLayer(engine: *Engine, layer: usize, hidden: []f32, pos: usize) !void {
    const rc = moe_infer_forward_layer(engine, @intCast(layer), hidden.ptr, @intCast(pos));
    if (rc != 0) return error.ForwardFailed;
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
    moe_infer_set_weights(engine, w.embed.ptr, @intCast(w.embed.len / 4096), w.lm_head.ptr, w.final_norm.ptr, &in_arr, &an_arr, &gp_arr, &gb_arr);

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
