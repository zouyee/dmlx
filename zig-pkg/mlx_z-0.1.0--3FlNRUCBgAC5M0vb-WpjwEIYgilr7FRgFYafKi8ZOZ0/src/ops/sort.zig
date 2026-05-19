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

pub fn sort(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_sort(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn sortAxis(_ctx: EagerContext, a: Array, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_sort_axis(&res, a.inner, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn argsort(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_argsort(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn argsortAxis(_ctx: EagerContext, a: Array, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_argsort_axis(&res, a.inner, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn partition(_ctx: EagerContext, a: Array, kth: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_partition(&res, a.inner, kth, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn partitionAxis(_ctx: EagerContext, a: Array, kth: i32, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_partition_axis(&res, a.inner, kth, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn argpartition(_ctx: EagerContext, a: Array, kth: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_argpartition(&res, a.inner, kth, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn argpartitionAxis(_ctx: EagerContext, a: Array, kth: i32, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_argpartition_axis(&res, a.inner, kth, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
