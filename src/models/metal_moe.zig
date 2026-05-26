/// Metal MoE forward executor — flash-moe FMA kernel patterns.
/// Calls C wrapper (moe_metal_wrapper.c) for runtime Metal shader compilation.
const std = @import("std");

const moe_metal_source = @embedFile("moe_kernel.metal");
var g_enabled: bool = false;

pub fn setEnabled(on: bool) void {
    g_enabled = on;
}
pub fn isEnabled() bool {
    return g_enabled;
}

// C functions: pass source as parameter, avoid @export issues
extern fn moe_metal_init_from_source(src: [*c]const u8, len: usize) c_int;
extern fn moe_metal_forward(
    expert_data: ?*const anyopaque,
    hidden: [*c]const f32,
    scores: [*c]const f32,
    output: [*c]f32,
    K: c_int,
    hidden_dim: c_int,
    intermediate_dim: c_int,
    group_size: c_int,
) c_int;

pub fn init() bool {
    return moe_metal_init_from_source(moe_metal_source, moe_metal_source.len) == 0;
}

/// MoE forward for one layer. K experts, DeepSeek V4 shapes.
/// expert_data: packed expert buffer (K × 13,369,344 bytes)
/// hidden: [4096] f32 input
/// scores: [K] f32 router weights
/// output: [4096] f32 (caller allocates)
pub fn forward(
    expert_ptrs: []const [*]const u8,
    hidden: []const f32,
    scores: []const f32,
    output: []f32,
) !void {
    if (expert_ptrs.len == 0 or hidden.len != 4096 or output.len != 4096) return error.InvalidArgs;
    const K: c_int = @intCast(scores.len);
    const rc = moe_metal_forward(
        @ptrCast(expert_ptrs.ptr),
        hidden.ptr,
        scores.ptr,
        output.ptr,
        K,
        4096,
        2048,
        32,
    );
    if (rc != 0) return error.MetalForwardFailed;
}
