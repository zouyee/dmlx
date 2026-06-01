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
) void;

extern fn moe_infer_forward(engine: *Engine, hidden: [*c]f32, pos: c_int) c_int;
extern fn moe_infer_forward_layer(engine: *Engine, layer: c_int, hidden: [*c]f32, pos: c_int) c_int;
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

pub fn init(packed_dir: []const u8) !*Engine {
    const engine = moe_infer_init(packed_dir.ptr, moe_metal_source, moe_metal_source.len);
    if (engine == null) return error.InitFailed;
    return engine.?;
}

pub fn forward(engine: *Engine, hidden: []f32, pos: u32) !void {
    const rc = moe_infer_forward(engine, hidden.ptr, @intCast(pos));
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
    for (0..@intCast(w.n_layers)) |i| {
        in_arr[i] = w.input_norms[i].ptr;
        an_arr[i] = w.attn_norms[i].ptr;
        gp_arr[i] = w.gate_projs[i].ptr;
    }
    moe_infer_set_weights(engine, w.embed.ptr, @intCast(w.embed.len / 4096), w.lm_head.ptr, w.final_norm.ptr, &in_arr, &an_arr, &gp_arr);

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
        ca.wo_b = cqw(ap.wo_b);
        ca.attn_sink = ap.attn_sink.ptr;
        moe_infer_set_layer_attn(engine, @intCast(i), ca);
        moe_infer_set_layer_hc(engine, @intCast(i), w.attn_hc_fn[i].ptr, w.attn_hc_base[i].ptr, w.attn_hc_scale[i].ptr, w.ffn_hc_fn[i].ptr, w.ffn_hc_base[i].ptr, w.ffn_hc_scale[i].ptr);
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
