/// Evaluation and async evaluation for MLX arrays.
const std = @import("std");
const c = @import("c.zig");
const array_mod = @import("array.zig");

const Array = array_mod.Array;

inline fn defaultStream() c.c.mlx_stream {
    return c.c.mlx_default_cpu_stream_new();
}

fn toVectorArray(allocator: std.mem.Allocator, arrs: []const Array) !c.c.mlx_vector_array {
    var list = try allocator.alloc(c.c.mlx_array, arrs.len);
    defer allocator.free(list);
    for (arrs, 0..) |arr, i| {
        list[i] = arr.inner;
    }
    return c.c.mlx_vector_array_new_data(list.ptr, list.len);
}

/// Synchronously evaluate an array.
pub fn eval(arr: Array) !void {
    const vec = c.c.mlx_vector_array_new_value(arr.inner);
    defer _ = c.c.mlx_vector_array_free(vec);
    try c.check(c.c.mlx_eval(vec));
}

/// Asynchronously evaluate a set of arrays.
pub fn asyncEval(allocator: std.mem.Allocator, arrs: []const Array) !void {
    const vec = try toVectorArray(allocator, arrs);
    defer _ = c.c.mlx_vector_array_free(vec);
    try c.check(c.c.mlx_async_eval(vec));
}
