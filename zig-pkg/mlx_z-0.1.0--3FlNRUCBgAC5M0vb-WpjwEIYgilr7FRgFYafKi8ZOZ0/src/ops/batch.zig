/// Batch / structured matrix operations (einsum, block_masked_mm, segmented_mm).
/// These wrap mlx-c operations that involve structured, batched,
/// or sparse matrix multiplication beyond basic matmul.
const std = @import("std");
const c = @import("../c.zig");
const ops_mod = @import("../ops.zig");
const array_mod = @import("../array.zig");
const device_mod = @import("../device.zig");

const Array = ops_mod.Array;
const Stream = device_mod.Stream;
const EagerContext = ops_mod.EagerContext;

/// Einstein summation on operands.
pub fn einsum(_ctx: EagerContext, subscripts: []const u8, arrays: []const Array) !Array {
    const sub_z = try _ctx.allocator.dupeZ(u8, subscripts);
    defer _ctx.allocator.free(sub_z);

    const raw = try _ctx.allocator.alloc(c.c.mlx_array, arrays.len);
    defer _ctx.allocator.free(raw);
    for (arrays, 0..) |arr, i| raw[i] = arr.inner;

    const vec = c.c.mlx_vector_array_new_data(raw.ptr, raw.len);
    defer _ = c.c.mlx_vector_array_free(vec);

    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_einsum(&res, sub_z.ptr, vec, _ctx.stream.inner));
    return Array.fromHandle(res);
}

/// Block-sparse matrix multiplication (used in block-sparse attention).
pub fn blockMaskedMm(
    _ctx: EagerContext,
    a: Array,
    b: Array,
    block_size: i32,
    mask_out: ?Array,
    mask_lhs: ?Array,
    mask_rhs: ?Array,
) !Array {
    const out_ptr = if (mask_out) |m| m.inner else null;
    const lhs_ptr = if (mask_lhs) |m| m.inner else null;
    const rhs_ptr = if (mask_rhs) |m| m.inner else null;

    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_block_masked_mm(&res, a.inner, b.inner, block_size, out_ptr, lhs_ptr, rhs_ptr, _ctx.stream.inner));
    return Array.fromHandle(res);
}

/// Segmented matrix multiplication (segments indicated by array).
pub fn segmentedMm(_ctx: EagerContext, a: Array, b: Array, segments: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_segmented_mm(&res, a.inner, b.inner, segments.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

/// Hadamard transform of an array.
pub fn hadamardTransform(_ctx: EagerContext, a: Array, scale: ?f32) !Array {
    var res = c.c.mlx_array_new();
    const opt_scale: c.c.mlx_optional_float = if (scale) |s| .{ .has_value = true, .value = s } else .{ .has_value = false, .value = 0 };
    try c.check(c.c.mlx_hadamard_transform(&res, a.inner, opt_scale, _ctx.stream.inner));
    return Array.fromHandle(res);
}
