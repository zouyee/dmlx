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

pub fn reshape(_ctx: EagerContext, a: Array, shape_: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_reshape(&res, a.inner, shape_.ptr, shape_.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn transpose(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_transpose(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn transposeAxes(_ctx: EagerContext, a: Array, axes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_transpose_axes(&res, a.inner, axes.ptr, axes.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn swapaxes(_ctx: EagerContext, a: Array, axis1: i32, axis2: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_swapaxes(&res, a.inner, axis1, axis2, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn moveaxis(_ctx: EagerContext, a: Array, source: i32, destination: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_moveaxis(&res, a.inner, source, destination, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn squeeze(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_squeeze(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn squeezeAxis(_ctx: EagerContext, a: Array, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_squeeze_axis(&res, a.inner, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn squeezeAxes(_ctx: EagerContext, a: Array, axes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_squeeze_axes(&res, a.inner, axes.ptr, axes.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn expandDims(_ctx: EagerContext, a: Array, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_expand_dims(&res, a.inner, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn expandDimsAxes(_ctx: EagerContext, a: Array, axes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_expand_dims_axes(&res, a.inner, axes.ptr, axes.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn flatten(_ctx: EagerContext, a: Array, start_axis: i32, end_axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_flatten(&res, a.inner, start_axis, end_axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn unflatten(_ctx: EagerContext, a: Array, axis: i32, shape_: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_unflatten(&res, a.inner, axis, shape_.ptr, shape_.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn broadcastTo(_ctx: EagerContext, a: Array, shape_: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_broadcast_to(&res, a.inner, shape_.ptr, shape_.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn contiguous(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_contiguous(&res, a.inner, false, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn contiguousEx(_ctx: EagerContext, a: Array, allow_col_major: bool) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_contiguous(&res, a.inner, allow_col_major, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn asStrided(_ctx: EagerContext, a: Array, shape_: []const i32, strides_: []const i64, offset: usize) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_as_strided(&res, a.inner, shape_.ptr, shape_.len, strides_.ptr, strides_.len, offset, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn slice(_ctx: EagerContext, a: Array, start: []const i32, stop: []const i32, strides_: []const i32) !Array {
    var res = c.c.mlx_array_new();
    if (strides_.len == 0) {
        const ones = try _ctx.allocator.alloc(i32, start.len);
        defer _ctx.allocator.free(ones);
        @memset(ones, 1);
        try c.check(c.c.mlx_slice(&res, a.inner, start.ptr, start.len, stop.ptr, stop.len, ones.ptr, ones.len, _ctx.stream.inner));
    } else {
        try c.check(c.c.mlx_slice(&res, a.inner, start.ptr, start.len, stop.ptr, stop.len, strides_.ptr, strides_.len, _ctx.stream.inner));
    }
    return Array.fromHandle(res);
}
pub fn sliceUpdate(_ctx: EagerContext, src: Array, update: Array, start: []const i32, stop: []const i32, strides_: []const i32) !Array {
    var res = c.c.mlx_array_new();
    if (strides_.len == 0) {
        const ones = try _ctx.allocator.alloc(i32, start.len);
        defer _ctx.allocator.free(ones);
        @memset(ones, 1);
        try c.check(c.c.mlx_slice_update(&res, src.inner, update.inner, start.ptr, start.len, stop.ptr, stop.len, ones.ptr, ones.len, _ctx.stream.inner));
    } else {
        try c.check(c.c.mlx_slice_update(&res, src.inner, update.inner, start.ptr, start.len, stop.ptr, stop.len, strides_.ptr, strides_.len, _ctx.stream.inner));
    }
    return Array.fromHandle(res);
}
pub fn take(_ctx: EagerContext, a: Array, indices: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_take(&res, a.inner, indices.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn takeAxis(_ctx: EagerContext, a: Array, indices: Array, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_take_axis(&res, a.inner, indices.inner, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn takeAlongAxis(_ctx: EagerContext, a: Array, indices: Array, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_take_along_axis(&res, a.inner, indices.inner, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn roll(_ctx: EagerContext, a: Array, shift: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_roll(&res, a.inner, shift.ptr, shift.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn rollAxis(_ctx: EagerContext, a: Array, shift: []const i32, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_roll_axis(&res, a.inner, shift.ptr, shift.len, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn rollAxes(_ctx: EagerContext, a: Array, shift: []const i32, axes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_roll_axes(&res, a.inner, shift.ptr, shift.len, axes.ptr, axes.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn tile(_ctx: EagerContext, a: Array, reps: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_tile(&res, a.inner, reps.ptr, reps.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn repeat(_ctx: EagerContext, a: Array, repeats: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_repeat(&res, a.inner, repeats, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn repeatAxis(_ctx: EagerContext, a: Array, repeats: i32, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_repeat_axis(&res, a.inner, repeats, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn view(_ctx: EagerContext, a: Array, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_view(&res, a.inner, @intCast(@intFromEnum(dt)), _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn astype(_ctx: EagerContext, a: Array, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_astype(&res, a.inner, @intCast(@intFromEnum(dt)), _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn atleast1d(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_atleast_1d(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn atleast2d(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_atleast_2d(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn atleast3d(_ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_atleast_3d(&res, a.inner, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn tril(_ctx: EagerContext, a: Array, k: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_tril(&res, a.inner, k, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn triu(_ctx: EagerContext, a: Array, k: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_triu(&res, a.inner, k, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn diag(_ctx: EagerContext, a: Array, k: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_diag(&res, a.inner, k, _ctx.stream.inner));
    return Array.fromHandle(res);
}
pub fn diagonal(_ctx: EagerContext, a: Array, offset: i32, axis1: i32, axis2: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_diagonal(&res, a.inner, offset, axis1, axis2, _ctx.stream.inner));
    return Array.fromHandle(res);
}

// === Concatenate & Split ===

/// Concatenate arrays along the given axis.
pub fn concatenateAxis(_ctx: EagerContext, arrays: []const Array, axis: i32) !Array {
    const raw = try _ctx.allocator.alloc(c.c.mlx_array, arrays.len);
    defer _ctx.allocator.free(raw);
    for (arrays, 0..) |arr, i| raw[i] = arr.inner;

    const vec = c.c.mlx_vector_array_new_data(raw.ptr, raw.len);
    defer _ = c.c.mlx_vector_array_free(vec);

    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_concatenate_axis(&res, vec, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}

/// Concatenate arrays along axis 0.
pub fn concatenate(_ctx: EagerContext, arrays: []const Array) !Array {
    const raw = try _ctx.allocator.alloc(c.c.mlx_array, arrays.len);
    defer _ctx.allocator.free(raw);
    for (arrays, 0..) |arr, i| raw[i] = arr.inner;

    const vec = c.c.mlx_vector_array_new_data(raw.ptr, raw.len);
    defer _ = c.c.mlx_vector_array_free(vec);

    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_concatenate(&res, vec, _ctx.stream.inner));
    return Array.fromHandle(res);
}

/// Split array into `num_splits` equal parts along axis.
pub fn split(_ctx: EagerContext, a: Array, num_splits: i32, axis: i32) ![]Array {
    var vec: c.c.mlx_vector_array = .{ .ctx = null };
    try c.check(c.c.mlx_split(&vec, a.inner, num_splits, axis, _ctx.stream.inner));
    defer _ = c.c.mlx_vector_array_free(vec);

    const n = c.c.mlx_vector_array_size(vec);
    const result = try _ctx.allocator.alloc(Array, n);
    errdefer _ctx.allocator.free(result);

    for (0..n) |i| {
        var arr: c.c.mlx_array = undefined;
        try c.check(c.c.mlx_vector_array_get(&arr, vec, i));
        result[i] = Array.fromHandle(arr);
    }
    return result;
}

/// Split array at given indices along axis.
pub fn splitAt(_ctx: EagerContext, a: Array, indices: []const i32, axis: i32) ![]Array {
    var vec: c.c.mlx_vector_array = .{ .ctx = null };
    try c.check(c.c.mlx_split_sections(&vec, a.inner, indices.ptr, indices.len, axis, _ctx.stream.inner));
    defer _ = c.c.mlx_vector_array_free(vec);

    const n = c.c.mlx_vector_array_size(vec);
    const result = try _ctx.allocator.alloc(Array, n);
    errdefer _ctx.allocator.free(result);

    for (0..n) |i| {
        var arr: c.c.mlx_array = undefined;
        try c.check(c.c.mlx_vector_array_get(&arr, vec, i));
        result[i] = Array.fromHandle(arr);
    }
    return result;
}

/// Gather elements from array along axis using indices.
pub fn gather(_ctx: EagerContext, a: Array, indices: Array, axis: i32, slice_sizes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_gather_single(&res, a.inner, indices.inner, axis, slice_sizes.ptr, slice_sizes.len, _ctx.stream.inner));
    return Array.fromHandle(res);
}

/// Scatter-add updates into array at indices along axis.
pub fn scatterAdd(_ctx: EagerContext, a: Array, indices: Array, updates: Array, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_scatter_add_single(&res, a.inner, indices.inner, updates.inner, axis, _ctx.stream.inner));
    return Array.fromHandle(res);
}
