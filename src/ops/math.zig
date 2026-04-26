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

pub fn sign(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_sign(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn square(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_square(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn reciprocal(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_reciprocal(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn floor(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_floor(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn ceil(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_ceil(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn round(_ctx: EagerContext, a: Array, decimals: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_round(&res, a.inner, decimals, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn clip(_ctx: EagerContext, a: Array, a_min: ?Array, a_max: ?Array) !Array {
    var res = c.c.mlx_array_new();
    const min_ptr = if (a_min) |m| m.inner else null;
    const max_ptr = if (a_max) |m| m.inner else null;
    try c.check(c.c.mlx_clip(&res, a.inner, min_ptr, max_ptr, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn conjugate(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_conjugate(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn imag(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_imag(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn real(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_real(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn degrees(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_degrees(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn radians(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_radians(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn log10(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_log10(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn log1p(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_log1p(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn log2(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_log2(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn expm1(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_expm1(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn rsqrt(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_rsqrt(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn tan(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_tan(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn sinh(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_sinh(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn cosh(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_cosh(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn arcsin(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_arcsin(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn arccos(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_arccos(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn arctan(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_arctan(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn arctan2(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_arctan2(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn arcsinh(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_arcsinh(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn arccosh(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_arccosh(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn arctanh(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_arctanh(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn erf(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_erf(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn erfinv(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_erfinv(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn logaddexp(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_logaddexp(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn maximum(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_maximum(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn minimum(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_minimum(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn floorDivide(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_floor_divide(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn remainder(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_remainder(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn power(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_power(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn nanToNum(_ctx: EagerContext, a: Array, nan: f32, posinf: ?f32, neginf: ?f32) !Array {
    var res = c.c.mlx_array_new();
    const opt_posinf: c.c.mlx_optional_float = .{ .has_value = posinf != null, .value = if (posinf) |v| v else 0 };
    const opt_neginf: c.c.mlx_optional_float = .{ .has_value = neginf != null, .value = if (neginf) |v| v else 0 };
    try c.check(c.c.mlx_nan_to_num(&res, a.inner, nan, opt_posinf, opt_neginf, _ctx.stream.inner));
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

pub fn inner(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_inner(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn outer(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_outer(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn kron(_ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_kron(&res, a.inner, b.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}
