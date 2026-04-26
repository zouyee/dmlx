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

pub fn zeros(_ctx: EagerContext, shape_: []const i32, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_zeros(&res, shape_.ptr, shape_.len, @intCast(@intFromEnum(dt)), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn ones(_ctx: EagerContext, shape_: []const i32, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&res, shape_.ptr, shape_.len, @intCast(@intFromEnum(dt)), _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn zerosLike(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_zeros_like(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn onesLike(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones_like(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn full(_ctx: EagerContext, shape_: []const i32, vals: Array, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_full(&res, shape_.ptr, shape_.len, vals.inner, @intCast(@intFromEnum(dt)), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn fullLike(_ctx: EagerContext, a: Array, vals: Array, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_full_like(&res, a.inner, vals.inner, @intCast(@intFromEnum(dt)), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn eye(_ctx: EagerContext, n: i32, m: i32, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_eye(&res, n, m, 0, @intCast(@intFromEnum(dt)), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn arange(_ctx: EagerContext, start: f64, stop: f64, step: f64, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_arange(&res, start, stop, step, @intCast(@intFromEnum(dt)), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn linspace(_ctx: EagerContext, start: f64, stop: f64, num: i32, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_linspace(&res, start, stop, num, @intCast(@intFromEnum(dt)), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn identity(_ctx: EagerContext, n: i32, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_identity(&res, n, @intCast(@intFromEnum(dt)), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn tri(_ctx: EagerContext, n: i32, m: i32, k: i32, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_tri(&res, n, m, k, @intCast(@intFromEnum(dt)), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn bartlett(_ctx: EagerContext, M: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_bartlett(&res, M, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn blackman(_ctx: EagerContext, M: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_blackman(&res, M, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn hamming(_ctx: EagerContext, M: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_hamming(&res, M, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn hanning(_ctx: EagerContext, M: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_hanning(&res, M, _ctx.stream.inner));
    return Array.fromHandle(res);
}
