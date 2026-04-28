/// Core operations backed by mlx-c.
const std = @import("std");
const c = @import("c.zig");
const array_mod = @import("array.zig");
const device_mod = @import("device.zig");

pub const Array = array_mod.Array;
const Stream = device_mod.Stream;
const Dtype = @import("dtype.zig").Dtype;

/// Execution context for eager operations.
pub const EagerContext = struct {
    allocator: std.mem.Allocator,
    stream: Stream,

    pub fn init(allocator: std.mem.Allocator) EagerContext {
        return .{
            .allocator = allocator,
            .stream = .{ .inner = c.c.mlx_default_cpu_stream_new() },
        };
    }

    pub fn initWithStream(allocator: std.mem.Allocator, stream: Stream) EagerContext {
        return .{ .allocator = allocator, .stream = stream };
    }

    /// Release the mlx_stream held by this context.
    /// Safe to call even if the stream is a default/global stream.
    pub fn deinit(self: EagerContext) void {
        _ = c.c.mlx_stream_free(self.stream.inner);
    }
};

// === Element-wise binary ops ===

pub fn add(ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_add(&res, a.inner, b.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn subtract(ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_subtract(&res, a.inner, b.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn multiply(ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_multiply(&res, a.inner, b.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn divide(ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_divide(&res, a.inner, b.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn matmul(ctx: EagerContext, a: Array, b: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_matmul(&res, a.inner, b.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn addmm(ctx: EagerContext, c_: Array, a: Array, b: Array, alpha: f32, beta: f32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_addmm(&res, c_.inner, a.inner, b.inner, alpha, beta, ctx.stream.inner));
    return Array.fromHandle(res);
}

// === Element-wise unary ops ===

pub fn abs(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_abs(&res, a.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn exp(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_exp(&res, a.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn log(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_log(&res, a.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn sqrt(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_sqrt(&res, a.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn sin(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_sin(&res, a.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn cos(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_cos(&res, a.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn tanh(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_tanh(&res, a.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn relu(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    const zero = try Array.zeros(ctx.allocator, &[_]i32{}, .float32);
    defer zero.deinit();
    try c.check(c.c.mlx_maximum(&res, a.inner, zero.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn sigmoid(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_sigmoid(&res, a.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn softmax(ctx: EagerContext, a: Array, axes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    if (axes.len == 0) {
        try c.check(c.c.mlx_softmax(&res, a.inner, false, ctx.stream.inner));
    } else if (axes.len == 1) {
        try c.check(c.c.mlx_softmax_axis(&res, a.inner, axes[0], false, ctx.stream.inner));
    } else {
        try c.check(c.c.mlx_softmax_axes(&res, a.inner, axes.ptr, axes.len, false, ctx.stream.inner));
    }
    return Array.fromHandle(res);
}

pub fn negative(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_negative(&res, a.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

// === Reductions ===

pub fn sum(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_sum(&res, a.inner, false, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn mean(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_mean(&res, a.inner, false, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn max(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_max(&res, a.inner, false, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn min(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_min(&res, a.inner, false, ctx.stream.inner));
    return Array.fromHandle(res);
}

// === Shape ops ===

pub fn reshape(ctx: EagerContext, a: Array, shape_: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_reshape(&res, a.inner, shape_.ptr, shape_.len, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn transpose(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_transpose(&res, a.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn transposeAxes(ctx: EagerContext, a: Array, axes: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_transpose_axes(&res, a.inner, axes.ptr, axes.len, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn expandDims(ctx: EagerContext, a: Array, axis: i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_expand_dims(&res, a.inner, axis, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn squeeze(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_squeeze(&res, a.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn tile(ctx: EagerContext, a: Array, reps: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_tile(&res, a.inner, reps.ptr, reps.len, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn slice(ctx: EagerContext, a: Array, start: []const i32, stop: []const i32, strides_: []const i32) !Array {
    var res = c.c.mlx_array_new();
    if (strides_.len == 0) {
        const ones = try ctx.allocator.alloc(i32, start.len);
        defer ctx.allocator.free(ones);
        @memset(ones, 1);
        try c.check(c.c.mlx_slice(&res, a.inner, start.ptr, start.len, stop.ptr, stop.len, ones.ptr, ones.len, ctx.stream.inner));
    } else {
        try c.check(c.c.mlx_slice(&res, a.inner, start.ptr, start.len, stop.ptr, stop.len, strides_.ptr, strides_.len, ctx.stream.inner));
    }
    return Array.fromHandle(res);
}

pub fn broadcastTo(ctx: EagerContext, a: Array, shape_: []const i32) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_broadcast_to(&res, a.inner, shape_.ptr, shape_.len, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn copy(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_copy(&res, a.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn astype(ctx: EagerContext, a: Array, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_astype(&res, a.inner, @intCast(@intFromEnum(dt)), ctx.stream.inner));
    return Array.fromHandle(res);
}

// === Creation ===

pub fn arange(ctx: EagerContext, start: f64, stop: f64, step: f64, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_arange(&res, start, stop, step, @intCast(@intFromEnum(dt)), ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn eye(ctx: EagerContext, n: i32, m: i32, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_eye(&res, n, m, 0, @intCast(@intFromEnum(dt)), ctx.stream.inner));
    return Array.fromHandle(res);
}

pub fn full(ctx: EagerContext, shape_: []const i32, val: Array, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_full(&res, shape_.ptr, shape_.len, val.inner, @intCast(@intFromEnum(dt)), ctx.stream.inner));
    return Array.fromHandle(res);
}

// === Conditional ops ===

/// Select elements from `x` where condition is true, otherwise from `y`.
pub fn where(ctx: EagerContext, condition: Array, x: Array, y: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_where(&res, condition.inner, x.inner, y.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

// === Scalar helpers ===

pub fn scalar(ctx: EagerContext, val: f32, dt: Dtype) !Array {
    return switch (dt) {
        .float32 => Array.scalar(ctx.allocator, f32, val),
        .float64 => Array.scalar(ctx.allocator, f64, @floatCast(val)),
        .float16 => Array.scalar(ctx.allocator, f16, @floatCast(val)),
        .bfloat16 => Array.scalar(ctx.allocator, f32, val), // bfloat16 scalar not directly supported; create as f32 and cast
        .int32 => Array.scalar(ctx.allocator, i32, @intFromFloat(val)),
        .int64 => Array.scalar(ctx.allocator, i64, @intFromFloat(val)),
        .uint32 => Array.scalar(ctx.allocator, u32, @intFromFloat(val)),
        .uint64 => Array.scalar(ctx.allocator, u64, @intFromFloat(val)),
        else => Array.scalar(ctx.allocator, f32, val),
    };
}

pub fn scalarF32(ctx: EagerContext, val: f32) !Array {
    return Array.scalar(ctx.allocator, f32, val);
}

pub fn scalarI32(ctx: EagerContext, val: i32) !Array {
    return Array.scalar(ctx.allocator, i32, val);
}

// === FP8 Conversion ===

/// Convert an array to FP8 (E4M3) format.
pub fn toFp8(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_to_fp8(&res, a.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

/// Convert an FP8 array back to the specified dtype.
pub fn fromFp8(ctx: EagerContext, a: Array, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_from_fp8(&res, a.inner, @intCast(@intFromEnum(dt)), ctx.stream.inner));
    return Array.fromHandle(res);
}

/// Gather matrix multiplication: computes a subset of rows/cols from a matmul.
/// Uses mlx_gather_mm for efficient sparse expert dispatch.
/// a: [M, K], b: [N, K] (if transpose) or [K, N]
/// lhs_indices: optional row indices for a
/// rhs_indices: optional row indices for b (when b is [n_experts, out, in])
/// sorted_indices: true if rhs_indices are sorted by expert id for better memory locality
pub fn gatherMm(ctx: EagerContext, a: Array, b: Array, lhs_indices: ?Array, rhs_indices: ?Array, sorted_indices: bool) !Array {
    var res = c.c.mlx_array_new();
    const lhs_ptr = if (lhs_indices) |i| i.inner else c.c.mlx_array_empty;
    const rhs_ptr = if (rhs_indices) |i| i.inner else c.c.mlx_array_empty;
    try c.check(c.c.mlx_gather_mm(&res, a.inner, b.inner, lhs_ptr, rhs_ptr, sorted_indices, ctx.stream.inner));
    return Array.fromHandle(res);
}
