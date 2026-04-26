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

pub fn fft(_ctx: EagerContext, a: Array, n: i32, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_fft(&res, a.inner, n, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn ifft(_ctx: EagerContext, a: Array, n: i32, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_ifft(&res, a.inner, n, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn fft2(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_fft2(&res, a.inner, n.ptr, n.len, axes.ptr, axes.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn ifft2(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_ifft2(&res, a.inner, n.ptr, n.len, axes.ptr, axes.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn fftn(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_fftn(&res, a.inner, n.ptr, n.len, axes.ptr, axes.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn ifftn(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_ifftn(&res, a.inner, n.ptr, n.len, axes.ptr, axes.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn rfft(_ctx: EagerContext, a: Array, n: i32, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_rfft(&res, a.inner, n, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn irfft(_ctx: EagerContext, a: Array, n: i32, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_irfft(&res, a.inner, n, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn rfft2(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_rfft2(&res, a.inner, n.ptr, n.len, axes.ptr, axes.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn irfft2(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_irfft2(&res, a.inner, n.ptr, n.len, axes.ptr, axes.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn rfftn(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_rfftn(&res, a.inner, n.ptr, n.len, axes.ptr, axes.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn irfftn(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_irfftn(&res, a.inner, n.ptr, n.len, axes.ptr, axes.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn fftshift(_ctx: EagerContext, a: Array, axes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_fftshift(&res, a.inner, axes.ptr, axes.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn ifftshift(_ctx: EagerContext, a: Array, axes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_ifftshift(&res, a.inner, axes.ptr, axes.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
