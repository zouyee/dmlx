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

/// FFT normalization mode (maps to mlx_fft_norm enum).
pub const Norm = enum(c_uint) {
    backward = 0, // MLX_FFT_NORM_BACKWARD (default, like numpy)
    ortho = 1, // MLX_FFT_NORM_ORTHO
    forward = 2, // MLX_FFT_NORM_FORWARD
};

pub fn fft(_ctx: EagerContext, a: Array, n: i32, axis: i32) !Array {
    return fftNorm(_ctx, a, n, axis, .backward);
}
pub fn fftNorm(_ctx: EagerContext, a: Array, n: i32, axis: i32, norm: Norm) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_fft(&res, a.inner, n, axis, @intFromEnum(norm), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn ifft(_ctx: EagerContext, a: Array, n: i32, axis: i32) !Array {
    return ifftNorm(_ctx, a, n, axis, .backward);
}
pub fn ifftNorm(_ctx: EagerContext, a: Array, n: i32, axis: i32, norm: Norm) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_ifft(&res, a.inner, n, axis, @intFromEnum(norm), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn fft2(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32) !Array {
    return fft2Norm(_ctx, a, n, axes, .backward);
}
pub fn fft2Norm(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32, norm: Norm) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_fft2(&res, a.inner, n.ptr, n.len, axes.ptr, axes.len, @intFromEnum(norm), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn ifft2(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32) !Array {
    return ifft2Norm(_ctx, a, n, axes, .backward);
}
pub fn ifft2Norm(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32, norm: Norm) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_ifft2(&res, a.inner, n.ptr, n.len, axes.ptr, axes.len, @intFromEnum(norm), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn fftn(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32) !Array {
    return fftnNorm(_ctx, a, n, axes, .backward);
}
pub fn fftnNorm(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32, norm: Norm) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_fftn(&res, a.inner, n.ptr, n.len, axes.ptr, axes.len, @intFromEnum(norm), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn ifftn(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32) !Array {
    return ifftnNorm(_ctx, a, n, axes, .backward);
}
pub fn ifftnNorm(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32, norm: Norm) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_ifftn(&res, a.inner, n.ptr, n.len, axes.ptr, axes.len, @intFromEnum(norm), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn rfft(_ctx: EagerContext, a: Array, n: i32, axis: i32) !Array {
    return rfftNorm(_ctx, a, n, axis, .backward);
}
pub fn rfftNorm(_ctx: EagerContext, a: Array, n: i32, axis: i32, norm: Norm) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_rfft(&res, a.inner, n, axis, @intFromEnum(norm), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn irfft(_ctx: EagerContext, a: Array, n: i32, axis: i32) !Array {
    return irfftNorm(_ctx, a, n, axis, .backward);
}
pub fn irfftNorm(_ctx: EagerContext, a: Array, n: i32, axis: i32, norm: Norm) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_irfft(&res, a.inner, n, axis, @intFromEnum(norm), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn rfft2(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32) !Array {
    return rfft2Norm(_ctx, a, n, axes, .backward);
}
pub fn rfft2Norm(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32, norm: Norm) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_rfft2(&res, a.inner, n.ptr, n.len, axes.ptr, axes.len, @intFromEnum(norm), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn irfft2(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32) !Array {
    return irfft2Norm(_ctx, a, n, axes, .backward);
}
pub fn irfft2Norm(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32, norm: Norm) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_irfft2(&res, a.inner, n.ptr, n.len, axes.ptr, axes.len, @intFromEnum(norm), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn rfftn(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32) !Array {
    return rfftnNorm(_ctx, a, n, axes, .backward);
}
pub fn rfftnNorm(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32, norm: Norm) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_rfftn(&res, a.inner, n.ptr, n.len, axes.ptr, axes.len, @intFromEnum(norm), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn irfftn(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32) !Array {
    return irfftnNorm(_ctx, a, n, axes, .backward);
}
pub fn irfftnNorm(_ctx: EagerContext, a: Array, n: []const i32, axes: []const i32, norm: Norm) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_fft_irfftn(&res, a.inner, n.ptr, n.len, axes.ptr, axes.len, @intFromEnum(norm), _ctx.stream.inner));
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
