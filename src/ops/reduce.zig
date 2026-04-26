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

pub fn sum(_ctx: EagerContext, a: Array, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_sum(&res, a.inner, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn sumAxis(_ctx: EagerContext, a: Array, axis: i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_sum_axis(&res, a.inner, axis, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn sumAxes(_ctx: EagerContext, a: Array, axes: []const i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_sum_axes(&res, a.inner, axes.ptr, axes.len, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn mean(_ctx: EagerContext, a: Array, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_mean(&res, a.inner, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn meanAxis(_ctx: EagerContext, a: Array, axis: i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_mean_axis(&res, a.inner, axis, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn meanAxes(_ctx: EagerContext, a: Array, axes: []const i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_mean_axes(&res, a.inner, axes.ptr, axes.len, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn max(_ctx: EagerContext, a: Array, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_max(&res, a.inner, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn maxAxis(_ctx: EagerContext, a: Array, axis: i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_max_axis(&res, a.inner, axis, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn maxAxes(_ctx: EagerContext, a: Array, axes: []const i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_max_axes(&res, a.inner, axes.ptr, axes.len, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn min(_ctx: EagerContext, a: Array, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_min(&res, a.inner, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn minAxis(_ctx: EagerContext, a: Array, axis: i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_min_axis(&res, a.inner, axis, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn minAxes(_ctx: EagerContext, a: Array, axes: []const i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_min_axes(&res, a.inner, axes.ptr, axes.len, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn prod(_ctx: EagerContext, a: Array, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_prod(&res, a.inner, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn prodAxis(_ctx: EagerContext, a: Array, axis: i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_prod_axis(&res, a.inner, axis, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn prodAxes(_ctx: EagerContext, a: Array, axes: []const i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_prod_axes(&res, a.inner, axes.ptr, axes.len, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn argmax(_ctx: EagerContext, a: Array, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_argmax(&res, a.inner, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn argmaxAxis(_ctx: EagerContext, a: Array, axis: i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_argmax_axis(&res, a.inner, axis, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn argmin(_ctx: EagerContext, a: Array, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_argmin(&res, a.inner, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn argminAxis(_ctx: EagerContext, a: Array, axis: i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_argmin_axis(&res, a.inner, axis, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn logsumexp(_ctx: EagerContext, a: Array, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_logsumexp(&res, a.inner, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn logsumexpAxis(_ctx: EagerContext, a: Array, axis: i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_logsumexp_axis(&res, a.inner, axis, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn logsumexpAxes(_ctx: EagerContext, a: Array, axes: []const i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_logsumexp_axes(&res, a.inner, axes.ptr, axes.len, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn std_(_ctx: EagerContext, a: Array, keepdims: bool, ddof: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_std(&res, a.inner, keepdims, ddof, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn stdAxis(_ctx: EagerContext, a: Array, axis: i32, keepdims: bool, ddof: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_std_axis(&res, a.inner, axis, keepdims, ddof, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn stdAxes(_ctx: EagerContext, a: Array, axes: []const i32, keepdims: bool, ddof: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_std_axes(&res, a.inner, axes.ptr, axes.len, keepdims, ddof, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn var_(_ctx: EagerContext, a: Array, keepdims: bool, ddof: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_var(&res, a.inner, keepdims, ddof, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn varAxis(_ctx: EagerContext, a: Array, axis: i32, keepdims: bool, ddof: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_var_axis(&res, a.inner, axis, keepdims, ddof, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn varAxes(_ctx: EagerContext, a: Array, axes: []const i32, keepdims: bool, ddof: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_var_axes(&res, a.inner, axes.ptr, axes.len, keepdims, ddof, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn cumsum(_ctx: EagerContext, a: Array, axis: i32, reverse: bool, inclusive: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_cumsum(&res, a.inner, axis, reverse, inclusive, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn cumprod(_ctx: EagerContext, a: Array, axis: i32, reverse: bool, inclusive: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_cumprod(&res, a.inner, axis, reverse, inclusive, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn cummax(_ctx: EagerContext, a: Array, axis: i32, reverse: bool, inclusive: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_cummax(&res, a.inner, axis, reverse, inclusive, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn cummin(_ctx: EagerContext, a: Array, axis: i32, reverse: bool, inclusive: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_cummin(&res, a.inner, axis, reverse, inclusive, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn logcumsumexp(_ctx: EagerContext, a: Array, axis: i32, reverse: bool, inclusive: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_logcumsumexp(&res, a.inner, axis, reverse, inclusive, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn median(_ctx: EagerContext, a: Array, axes: []const i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_median(&res, a.inner, axes.ptr, axes.len, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn topk(_ctx: EagerContext, a: Array, k: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_topk(&res, a.inner, k, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn topkAxis(_ctx: EagerContext, a: Array, k: i32, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_topk_axis(&res, a.inner, k, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn trace(_ctx: EagerContext, a: Array, offset: i32, axis1: i32, axis2: i32, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_trace(&res, a.inner, offset, axis1, axis2, @intCast(@intFromEnum(dt)), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn numberOfElements(_ctx: EagerContext, a: Array, axes: []const i32, inverted: bool, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_number_of_elements(&res, a.inner, axes.ptr, axes.len, inverted, @intCast(@intFromEnum(dt)), _ctx.stream.inner));
    return Array.fromHandle(res);
}
