/// Array wrapper around mlx-c's mlx_array.
/// Provides Zig-idiomatic API while using official MLX for computation.
const std = @import("std");
const c = @import("c.zig");
const dtype_mod = @import("dtype.zig");
const device_mod = @import("device.zig");

pub const Dtype = dtype_mod.Dtype;
pub const Stream = device_mod.Stream;

pub const ShapeElem = i32;

/// High-level array handle backed by mlx-c.
pub const Array = struct {
    inner: c.c.mlx_array,

    // === Construction ===

    /// Create from raw mlx_array (takes ownership).
    pub fn fromHandle(handle: c.c.mlx_array) Array {
        return .{ .inner = handle };
    }

    /// Create an array from a data slice with explicit shape.
    pub fn fromData(allocator: std.mem.Allocator, comptime T: type, data: []const T, shape_: []const ShapeElem) !Array {
        _ = allocator;
        const dt = dtype_mod.dtypeOf(T);
        const arr = c.c.mlx_array_new_data(
            data.ptr,
            shape_.ptr,
            @intCast(shape_.len),
            @intCast(@intFromEnum(dt)),
        );
        // mlx_array_new_data copies data; we don't need to keep it alive
        return fromHandle(arr);
    }

    /// Create a 1-D array from a slice.
    pub fn fromSlice(allocator: std.mem.Allocator, comptime T: type, data: []const T) !Array {
        const shape_ = [_]ShapeElem{@intCast(data.len)};
        return fromData(allocator, T, data, &shape_);
    }

    /// Create a scalar array.
    pub fn scalar(allocator: std.mem.Allocator, comptime T: type, val: T) !Array {
        return fromData(allocator, T, &[_]T{val}, &[_]ShapeElem{});
    }

    /// Create an array of zeros.
    pub fn zeros(allocator: std.mem.Allocator, shape_: []const ShapeElem, dt: Dtype) !Array {
        _ = allocator;
        var arr = c.c.mlx_array_new();
        const stream = c.c.mlx_default_cpu_stream_new();
        defer _ = c.c.mlx_stream_free(stream);
        try c.check(c.c.mlx_zeros(&arr, shape_.ptr, shape_.len, @intCast(@intFromEnum(dt)), stream));
        return fromHandle(arr);
    }

    /// Create an array of ones.
    pub fn ones(allocator: std.mem.Allocator, shape_: []const ShapeElem, dt: Dtype) !Array {
        _ = allocator;
        var arr = c.c.mlx_array_new();
        const stream = c.c.mlx_default_cpu_stream_new();
        defer _ = c.c.mlx_stream_free(stream);
        try c.check(c.c.mlx_ones(&arr, shape_.ptr, shape_.len, @intCast(@intFromEnum(dt)), stream));
        return fromHandle(arr);
    }

    // === Lifecycle ===

    pub fn deinit(self: Array) void {
        _ = c.c.mlx_array_free(self.inner);
    }

    /// Evaluate the array (handles cross-device scheduling).
    pub fn eval(self: Array) !void {
        // Use mlx_eval (vector version) which handles cross-device scheduling
        // (e.g., Load primitives on CPU, compute on GPU).
        // mlx_array_eval only evaluates on the array's stream which may fail
        // for Load primitives on GPU stream.
        const vec = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(vec);
        try c.check(c.c.mlx_vector_array_append_data(vec, &self.inner, 1));
        try c.check(c.c.mlx_eval(vec));
    }

    // === Properties ===

    pub fn shape(self: Array) []const ShapeElem {
        const nd = c.c.mlx_array_ndim(self.inner);
        const ptr = c.c.mlx_array_shape(self.inner);
        return ptr[0..nd];
    }

    pub fn strides(self: Array) []const i64 {
        const nd = c.c.mlx_array_ndim(self.inner);
        const ptr = c.c.mlx_array_strides(self.inner);
        // mlx-c returns size_t*; reinterpret as i64 on 64-bit platforms
        const cast_ptr: [*]const i64 = @ptrCast(@alignCast(ptr));
        return cast_ptr[0..nd];
    }

    pub fn ndim(self: Array) usize {
        return c.c.mlx_array_ndim(self.inner);
    }

    pub fn size(self: Array) usize {
        return c.c.mlx_array_size(self.inner);
    }

    pub fn itemsize(self: Array) usize {
        return c.c.mlx_dtype_size(c.c.mlx_array_dtype(self.inner));
    }

    pub fn nbytes(self: Array) usize {
        return self.size() * self.itemsize();
    }

    pub fn dtype(self: Array) Dtype {
        return @enumFromInt(c.c.mlx_array_dtype(self.inner));
    }

    pub fn isScalar(self: Array) bool {
        return self.ndim() == 0;
    }

    // === Data Access ===

    /// Get typed pointer. Array must be evaluated.
    pub fn dataPtr(self: Array, comptime T: type) ![*]const T {
        try self.eval();
        const dt = dtype_mod.dtypeOf(T);
        return switch (dt) {
            .float32 => @ptrCast(@alignCast(c.c.mlx_array_data_float32(self.inner))),
            .float64 => @ptrCast(@alignCast(c.c.mlx_array_data_float64(self.inner))),
            .int32 => @ptrCast(@alignCast(c.c.mlx_array_data_int32(self.inner))),
            .int64 => @ptrCast(@alignCast(c.c.mlx_array_data_int64(self.inner))),
            .uint32 => @ptrCast(@alignCast(c.c.mlx_array_data_uint32(self.inner))),
            .uint64 => @ptrCast(@alignCast(c.c.mlx_array_data_uint64(self.inner))),
            .bool_ => @ptrCast(@alignCast(c.c.mlx_array_data_bool(self.inner))),
            else => error.UnsupportedDtype,
        };
    }

    /// Get typed slice. Array must be evaluated.
    pub fn dataSlice(self: Array, comptime T: type) ![]const T {
        const ptr = try self.dataPtr(T);
        return ptr[0..self.size()];
    }

    /// Get mutable typed slice. Array must be evaluated.
    pub fn dataSliceMut(self: Array, comptime T: type) ![]T {
        const ptr = try self.dataPtr(T);
        return @constCast(ptr)[0..self.size()];
    }

    /// Get scalar value from a 0-dim array.
    pub fn item(self: Array, comptime T: type) !T {
        if (!self.isScalar()) return error.NotScalar;
        const ptr = try self.dataPtr(T);
        return ptr[0];
    }

    // === Formatting ===

    pub fn format(self: Array, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        var str: c.c.mlx_string = undefined;
        _ = c.c.mlx_array_tostring(&str, self.inner);
        defer _ = c.c.mlx_string_free(str);
        const cstr = c.c.mlx_string_data(str);
        try writer.writeAll(std.mem.span(cstr));
    }
};

// Convenience exports matching old API
pub fn zeros(allocator: std.mem.Allocator, shape_: []const ShapeElem, dt: Dtype) !Array {
    return Array.zeros(allocator, shape_, dt);
}

pub fn ones(allocator: std.mem.Allocator, shape_: []const ShapeElem, dt: Dtype) !Array {
    return Array.ones(allocator, shape_, dt);
}
