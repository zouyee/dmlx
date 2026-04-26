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

pub fn key(key_seed: u64) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_key(&res, key_seed));
    return Array.fromHandle(res);
}
pub fn split(_ctx: EagerContext, k: Array) !struct { a: Array, b: Array } {
    var a = c.c.mlx_array_new();
    var b = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_split(&a, &b, k.inner, _ctx.stream.inner));
    return .{ .a = Array.fromHandle(a), .b = Array.fromHandle(b) };
}
pub fn splitNum(_ctx: EagerContext, k: Array, num: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_split_num(&res, k.inner, num, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn seed(s: u64) !void {
    try c.check(c.c.mlx_random_seed(s));
}
pub fn normal(_ctx: EagerContext, shape_: []const i32, dt: Dtype, loc: f32, scale: f32, k: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const key_val = if (k) |v| v.inner else c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(&res, shape_.ptr, shape_.len, @intCast(@intFromEnum(dt)), loc, scale, key_val, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn normalBroadcast(_ctx: EagerContext, shape_: []const i32, dt: Dtype, loc: ?Array, scale: ?Array, k: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const loc_ptr = if (loc) |v| v.inner else null;
    const scale_ptr = if (scale) |v| v.inner else null;
    const key_val = if (k) |v| v.inner else c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal_broadcast(&res, shape_.ptr, shape_.len, @intCast(@intFromEnum(dt)), loc_ptr, scale_ptr, key_val, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn uniform(_ctx: EagerContext, low: Array, high: Array, shape_: []const i32, dt: Dtype, k: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const key_val = if (k) |v| v.inner else c.c.mlx_array_new();
    try c.check(c.c.mlx_random_uniform(&res, low.inner, high.inner, shape_.ptr, shape_.len, @intCast(@intFromEnum(dt)), key_val, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn bernoulli(_ctx: EagerContext, p: Array, shape_: []const i32, k: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const key_val = if (k) |v| v.inner else c.c.mlx_array_new();
    try c.check(c.c.mlx_random_bernoulli(&res, p.inner, shape_.ptr, shape_.len, key_val, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn bits(_ctx: EagerContext, shape_: []const i32, width: i32, k: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const key_val = if (k) |v| v.inner else c.c.mlx_array_new();
    try c.check(c.c.mlx_random_bits(&res, shape_.ptr, shape_.len, width, key_val, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn categorical(_ctx: EagerContext, logits: Array, axis: i32, k: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const key_val = if (k) |v| v.inner else c.c.mlx_array_new();
    try c.check(c.c.mlx_random_categorical(&res, logits.inner, axis, key_val, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn categoricalNumSamples(_ctx: EagerContext, logits: Array, axis: i32, num_samples: i32, k: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const key_val = if (k) |v| v.inner else c.c.mlx_array_new();
    try c.check(c.c.mlx_random_categorical_num_samples(&res, logits.inner, axis, num_samples, key_val, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn categoricalShape(_ctx: EagerContext, logits: Array, axis: i32, shape_: []const i32, k: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const key_val = if (k) |v| v.inner else c.c.mlx_array_new();
    try c.check(c.c.mlx_random_categorical_shape(&res, logits.inner, axis, shape_.ptr, shape_.len, key_val, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn gumbel(_ctx: EagerContext, shape_: []const i32, dt: Dtype, k: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const key_val = if (k) |v| v.inner else c.c.mlx_array_new();
    try c.check(c.c.mlx_random_gumbel(&res, shape_.ptr, shape_.len, @intCast(@intFromEnum(dt)), key_val, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn laplace(_ctx: EagerContext, shape_: []const i32, dt: Dtype, loc: f32, scale: f32, k: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const key_val = if (k) |v| v.inner else c.c.mlx_array_new();
    try c.check(c.c.mlx_random_laplace(&res, shape_.ptr, shape_.len, @intCast(@intFromEnum(dt)), loc, scale, key_val, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn multivariateNormal(_ctx: EagerContext, mean: Array, cov: Array, shape_: []const i32, dt: Dtype, k: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const key_val = if (k) |v| v.inner else c.c.mlx_array_new();
    try c.check(c.c.mlx_random_multivariate_normal(&res, mean.inner, cov.inner, shape_.ptr, shape_.len, @intCast(@intFromEnum(dt)), key_val, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn randint(_ctx: EagerContext, low: Array, high: Array, shape_: []const i32, dt: Dtype, k: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const key_val = if (k) |v| v.inner else c.c.mlx_array_new();
    try c.check(c.c.mlx_random_randint(&res, low.inner, high.inner, shape_.ptr, shape_.len, @intCast(@intFromEnum(dt)), key_val, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn permutation(_ctx: EagerContext, x: Array, axis: i32, k: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const key_val = if (k) |v| v.inner else c.c.mlx_array_new();
    try c.check(c.c.mlx_random_permutation(&res, x.inner, axis, key_val, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn permutationArange(_ctx: EagerContext, x: i32, k: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const key_val = if (k) |v| v.inner else c.c.mlx_array_new();
    try c.check(c.c.mlx_random_permutation_arange(&res, x, key_val, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn truncatedNormal(_ctx: EagerContext, lower: Array, upper: Array, shape_: []const i32, dt: Dtype, k: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const key_val = if (k) |v| v.inner else c.c.mlx_array_new();
    try c.check(c.c.mlx_random_truncated_normal(&res, lower.inner, upper.inner, shape_.ptr, shape_.len, @intCast(@intFromEnum(dt)), key_val, _ctx.stream.inner));
    return Array.fromHandle(res);
}
