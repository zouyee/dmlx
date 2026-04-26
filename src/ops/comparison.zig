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

pub fn equal(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_equal(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn notEqual(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_not_equal(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn greater(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_greater(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn greaterEqual(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_greater_equal(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn less(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_less(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn lessEqual(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_less_equal(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn logicalAnd(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_logical_and(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn logicalOr(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_logical_or(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn logicalNot(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_logical_not(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn bitwiseAnd(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_bitwise_and(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn bitwiseOr(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_bitwise_or(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn bitwiseXor(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_bitwise_xor(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn bitwiseInvert(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_bitwise_invert(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn leftShift(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_left_shift(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn rightShift(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_right_shift(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn allClose(_ctx: EagerContext, a: Array, b: Array, rtol: f64, atol: f64, equal_nan: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_allclose(&res, a.inner, b.inner, rtol, atol, equal_nan, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn isClose(_ctx: EagerContext, a: Array, b: Array, rtol: f64, atol: f64, equal_nan: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_isclose(&res, a.inner, b.inner, rtol, atol, equal_nan, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn arrayEqual(_ctx: EagerContext, a: Array, b: Array, equal_nan: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_array_equal(&res, a.inner, b.inner, equal_nan, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn all(_ctx: EagerContext, a: Array, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_all(&res, a.inner, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn allAxis(_ctx: EagerContext, a: Array, axis: i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_all_axis(&res, a.inner, axis, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn allAxes(_ctx: EagerContext, a: Array, axes: []const i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_all_axes(&res, a.inner, axes.ptr, axes.len, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn any(_ctx: EagerContext, a: Array, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_any(&res, a.inner, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn anyAxis(_ctx: EagerContext, a: Array, axis: i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_any_axis(&res, a.inner, axis, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn anyAxes(_ctx: EagerContext, a: Array, axes: []const i32, keepdims: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_any_axes(&res, a.inner, axes.ptr, axes.len, keepdims, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn isfinite(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_isfinite(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn isinf(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_isinf(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn isnan(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_isnan(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn isneginf(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_isneginf(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn isposinf(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_isposinf(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}
