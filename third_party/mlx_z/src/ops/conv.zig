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

pub fn conv1d(_ctx: EagerContext, input: Array, weight: Array, stride: i32, padding: i32, dilation: i32, groups: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_conv1d(&res, input.inner, weight.inner, stride, padding, dilation, groups, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn conv2d(_ctx: EagerContext, input: Array, weight: Array, stride_0: i32, stride_1: i32, padding_0: i32, padding_1: i32, dilation_0: i32, dilation_1: i32, groups: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_conv2d(&res, input.inner, weight.inner, stride_0, stride_1, padding_0, padding_1, dilation_0, dilation_1, groups, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn conv3d(_ctx: EagerContext, input: Array, weight: Array, stride_0: i32, stride_1: i32, stride_2: i32, padding_0: i32, padding_1: i32, padding_2: i32, dilation_0: i32, dilation_1: i32, dilation_2: i32, groups: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_conv3d(&res, input.inner, weight.inner, stride_0, stride_1, stride_2, padding_0, padding_1, padding_2, dilation_0, dilation_1, dilation_2, groups, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn convGeneral(_ctx: EagerContext, input: Array, weight: Array, stride: []const i32, padding_lo: []const i32, padding_hi: []const i32, kernel_dilation: []const i32, input_dilation: []const i32, groups: i32, flip: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_conv_general(&res, input.inner, weight.inner, stride.ptr, stride.len, padding_lo.ptr, padding_lo.len, padding_hi.ptr, padding_hi.len, kernel_dilation.ptr, kernel_dilation.len, input_dilation.ptr, input_dilation.len, groups, flip, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn convTranspose1d(_ctx: EagerContext, input: Array, weight: Array, stride: i32, padding: i32, dilation: i32, output_padding: i32, groups: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_conv_transpose1d(&res, input.inner, weight.inner, stride, padding, dilation, output_padding, groups, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn convTranspose2d(_ctx: EagerContext, input: Array, weight: Array, stride_0: i32, stride_1: i32, padding_0: i32, padding_1: i32, dilation_0: i32, dilation_1: i32, output_padding_0: i32, output_padding_1: i32, groups: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_conv_transpose2d(&res, input.inner, weight.inner, stride_0, stride_1, padding_0, padding_1, dilation_0, dilation_1, output_padding_0, output_padding_1, groups, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn convTranspose3d(_ctx: EagerContext, input: Array, weight: Array, stride_0: i32, stride_1: i32, stride_2: i32, padding_0: i32, padding_1: i32, padding_2: i32, dilation_0: i32, dilation_1: i32, dilation_2: i32, output_padding_0: i32, output_padding_1: i32, output_padding_2: i32, groups: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_conv_transpose3d(&res, input.inner, weight.inner, stride_0, stride_1, stride_2, padding_0, padding_1, padding_2, dilation_0, dilation_1, dilation_2, output_padding_0, output_padding_1, output_padding_2, groups, _ctx.stream.inner));
    return Array.fromHandle(res);
}
