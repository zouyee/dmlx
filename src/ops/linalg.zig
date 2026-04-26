const std = @import("std");
const c = @import("../c.zig");
const ops_mod = @import("../ops.zig");
const array_mod = @import("../array.zig");
const device_mod = @import("../device.zig");
const dtype_mod = @import("../dtype.zig");

const Array = ops_mod.Array;
const Stream = device_mod.Stream;
const Dtype = dtype_mod.Dtype;
const EagerContext = ops_mod.EagerContext;

pub fn cholesky(_ctx: EagerContext, a: Array, upper: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_linalg_cholesky(&res, a.inner, upper, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn choleskyInv(_ctx: EagerContext, a: Array, upper: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_linalg_cholesky_inv(&res, a.inner, upper, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn cross(_ctx: EagerContext, a: Array, b: Array, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_linalg_cross(&res, a.inner, b.inner, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn eig(_ctx: EagerContext, a: Array) !struct { w: Array, v: Array } {
    var w = c.c.mlx_array_new();
    var v = c.c.mlx_array_new();
    try c.check(c.c.mlx_linalg_eig(&w, &v, a.inner, _ctx.stream.inner));
    return .{ .w = Array.fromHandle(w), .v = Array.fromHandle(v) };
}
pub fn eigh(_ctx: EagerContext, a: Array, uplo: []const u8) !struct { w: Array, v: Array } {
    var w = c.c.mlx_array_new();
    var v = c.c.mlx_array_new();
    try c.check(c.c.mlx_linalg_eigh(&w, &v, a.inner, uplo.ptr, _ctx.stream.inner));
    return .{ .w = Array.fromHandle(w), .v = Array.fromHandle(v) };
}
pub fn eigvals(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_linalg_eigvals(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn eigvalsh(_ctx: EagerContext, a: Array, uplo: []const u8) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_linalg_eigvalsh(&res, a.inner, uplo.ptr, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn inv(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_linalg_inv(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn lu(_ctx: EagerContext, a: Array) ![]Array {
    var vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(vec);
    try c.check(c.c.mlx_linalg_lu(&vec, a.inner, _ctx.stream.inner));
    const n = c.c.mlx_vector_array_size(vec);
    var list = try std.ArrayList(Array).initCapacity(std.heap.page_allocator, n);
    for (0..n) |i| {
        var arr = c.c.mlx_array_new();
        try c.check(c.c.mlx_vector_array_get(&arr, vec, i));
        list.appendAssumeCapacity(Array.fromHandle(arr));
    }
    return list.toOwnedSlice(std.heap.page_allocator);
}
pub fn luFactor(_ctx: EagerContext, a: Array) !struct { lu: Array, pivots: Array } {
    var lu_arr = c.c.mlx_array_new();
    var pivots = c.c.mlx_array_new();
    try c.check(c.c.mlx_linalg_lu_factor(&lu_arr, &pivots, a.inner, _ctx.stream.inner));
    return .{ .lu = Array.fromHandle(lu_arr), .pivots = Array.fromHandle(pivots) };
}
pub fn norm(_ctx: EagerContext, a: Array, ord: f64, axis: ?[]const i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    const axis_ptr = if (axis) |ax| ax.ptr else null;
    const axis_len = if (axis) |ax| ax.len else 0;
    try c.check(c.c.mlx_linalg_norm(&res, a.inner, ord, axis_ptr, axis_len, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn normMatrix(_ctx: EagerContext, a: Array, ord: []const u8, axis: ?[]const i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    const axis_ptr = if (axis) |ax| ax.ptr else null;
    const axis_len = if (axis) |ax| ax.len else 0;
    try c.check(c.c.mlx_linalg_norm_matrix(&res, a.inner, ord.ptr, axis_ptr, axis_len, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn normL2(_ctx: EagerContext, a: Array, axis: ?[]const i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    const axis_ptr = if (axis) |ax| ax.ptr else null;
    const axis_len = if (axis) |ax| ax.len else 0;
    try c.check(c.c.mlx_linalg_norm_l2(&res, a.inner, axis_ptr, axis_len, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn pinv(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_linalg_pinv(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn qr(_ctx: EagerContext, a: Array) !struct { q: Array, r: Array } {
    var q = c.c.mlx_array_new();
    var r = c.c.mlx_array_new();
    try c.check(c.c.mlx_linalg_qr(&q, &r, a.inner, _ctx.stream.inner));
    return .{ .q = Array.fromHandle(q), .r = Array.fromHandle(r) };
}
pub fn solve(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_linalg_solve(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn solveTriangular(_ctx: EagerContext, a: Array, b: Array, upper: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_linalg_solve_triangular(&res, a.inner, b.inner, upper, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn svd(_ctx: EagerContext, a: Array, compute_uv: bool) ![]Array {
    var vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(vec);
    try c.check(c.c.mlx_linalg_svd(&vec, a.inner, compute_uv, _ctx.stream.inner));
    const n = c.c.mlx_vector_array_size(vec);
    var list = try std.ArrayList(Array).initCapacity(std.heap.page_allocator, n);
    for (0..n) |i| {
        var arr = c.c.mlx_array_new();
        try c.check(c.c.mlx_vector_array_get(&arr, vec, i));
        list.appendAssumeCapacity(Array.fromHandle(arr));
    }
    return list.toOwnedSlice(std.heap.page_allocator);
}
pub fn triInv(_ctx: EagerContext, a: Array, upper: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_linalg_tri_inv(&res, a.inner, upper, _ctx.stream.inner));
    return Array.fromHandle(res);
}
