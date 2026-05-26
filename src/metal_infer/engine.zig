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

extern fn moe_infer_forward(engine: *Engine, hidden: [*c]f32, pos: c_int) c_int;
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

pub fn deinit(engine: *Engine) void {
    moe_infer_deinit(engine);
}
