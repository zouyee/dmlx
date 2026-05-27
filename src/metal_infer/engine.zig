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
    q_proj_w: [*c][*c]const f32,
    k_proj_w: [*c][*c]const f32,
    v_proj_w: [*c][*c]const f32,
    o_proj_w: [*c][*c]const f32,
    q_norms: [*c][*c]const f32,
    k_norms: [*c][*c]const f32,
    gate_projs: [*c][*c]const f32,
) void;

extern fn moe_infer_forward(engine: *Engine, hidden: [*c]f32, pos: c_int) c_int;
extern fn moe_infer_forward_layer(engine: *Engine, layer: c_int, hidden: [*c]f32, pos: c_int) c_int;
extern fn moe_infer_deinit(engine: *Engine) void;

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
    // q/k/v/o proj + q/k norms — placeholder NULL (attention not wired yet)
    var qp_arr: [64][*c]const f32 = [_][*c]const f32{null} ** 64;
    var kp_arr: [64][*c]const f32 = [_][*c]const f32{null} ** 64;
    var vp_arr: [64][*c]const f32 = [_][*c]const f32{null} ** 64;
    var op_arr: [64][*c]const f32 = [_][*c]const f32{null} ** 64;
    var qn_arr: [64][*c]const f32 = [_][*c]const f32{null} ** 64;
    var kn_arr: [64][*c]const f32 = [_][*c]const f32{null} ** 64;
    moe_infer_set_weights(engine, w.embed.ptr, @intCast(w.embed.len / 4096), w.lm_head.ptr, w.final_norm.ptr, &in_arr, &an_arr, &qp_arr, &kp_arr, &vp_arr, &op_arr, &qn_arr, &kn_arr, &gp_arr);
}

pub fn deinit(engine: *Engine) void {
    moe_infer_deinit(engine);
}
