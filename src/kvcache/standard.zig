/// StandardKVCache — Pre-allocated fixed-length KV cache.
///
/// Strategy: allocate a buffer of shape [B, H, max_seq, D] at initialization.
/// On update: write new tokens via mlx_slice_update at the current offset.
/// On fetch: return a sliced view [0:offset] via mlx_slice.
///
/// Pros: minimal allocation during inference, simple, works with any attention.
/// Cons: memory grows with max_seq_len (not dynamically shrinkable).
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const iface = @import("interface.zig");

const Array = array_mod.Array;
const KVSlice = iface.KVSlice;
const KVCacheStrategy = iface.KVCacheStrategy;
const LayerConfig = iface.LayerConfig;

/// Concrete implementation of a standard pre-allocated KV cache.
pub const StandardKVCache = struct {
    allocator: std.mem.Allocator,

    // Pre-allocated buffers: [batch, num_kv_heads, max_seq, head_dim]
    keys: Array,
    values: Array,

    // Current sequence length (number of tokens cached so far).
    offset: usize,

    // Shape parameters (stored for filter/slice operations).
    batch_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,

    // VTable singleton — all instances share the same function pointers.
    pub const vtable: iface.VTable = .{
        .updateAndFetch = updateAndFetchImpl,
        .currentLen = currentLenImpl,
        .reset = resetImpl,
        .filter = filterImpl,
        .rollback = rollbackImpl,
        .extend = extendImpl,
        .getState = getStateImpl,
        .deinit = deinitImpl,
    };

    /// Create a new StandardKVCache with pre-allocated buffers.
    pub fn init(
        allocator: std.mem.Allocator,
        config: LayerConfig,
        stream: c.c.mlx_stream,
    ) !StandardKVCache {
        // Build shape arrays for mlx_zeros
        const shape = &[_]i32{
            @intCast(config.batch_size),
            @intCast(config.num_kv_heads),
            @intCast(config.max_seq_len),
            @intCast(config.head_dim),
        };

        var keys_arr = c.c.mlx_array_new();
        var values_arr = c.c.mlx_array_new();

        try c.check(c.c.mlx_zeros(
            &keys_arr,
            shape.ptr,
            shape.len,
            @intCast(@intFromEnum(config.dtype)),
            stream,
        ));
        try c.check(c.c.mlx_zeros(
            &values_arr,
            shape.ptr,
            shape.len,
            @intCast(@intFromEnum(config.dtype)),
            stream,
        ));

        return .{
            .allocator = allocator,
            .keys = Array.fromHandle(keys_arr),
            .values = Array.fromHandle(values_arr),
            .offset = 0,
            .batch_size = config.batch_size,
            .num_kv_heads = config.num_kv_heads,
            .head_dim = config.head_dim,
            .max_seq_len = config.max_seq_len,
        };
    }

    /// Cast to runtime-polymorphic KVCacheStrategy handle.
    pub fn asStrategy(self: *StandardKVCache) KVCacheStrategy {
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    // ------------------------------------------------------------------
    // VTable implementations
    // ------------------------------------------------------------------

    fn updateAndFetchImpl(
        ctx: *anyopaque,
        keys: Array,
        values: Array,
        stream: c.c.mlx_stream,
    ) anyerror!KVSlice {
        const self: *StandardKVCache = @ptrCast(@alignCast(ctx));
        const seq_len = @as(usize, @intCast(keys.shape()[2])); // [B, H, S, D]
        const new_offset = self.offset + seq_len;

        if (new_offset > self.max_seq_len) {
            return error.CacheOverflow;
        }

        // Write new keys into the pre-allocated buffer at [offset:offset+seq_len]
        try sliceUpdateKV(
            &self.keys,
            keys,
            self.offset,
            new_offset,
            stream,
        );
        try sliceUpdateKV(
            &self.values,
            values,
            self.offset,
            new_offset,
            stream,
        );

        self.offset = new_offset;

        // Return sliced view [0:offset]
        const fetched_keys = try sliceFetch(self.keys, self.offset, stream);
        const fetched_values = try sliceFetch(self.values, self.offset, stream);

        return .{
            .keys = fetched_keys,
            .values = fetched_values,
        };
    }

    fn currentLenImpl(ctx: *anyopaque) usize {
        const self: *StandardKVCache = @ptrCast(@alignCast(ctx));
        return self.offset;
    }

    fn resetImpl(ctx: *anyopaque) void {
        const self: *StandardKVCache = @ptrCast(@alignCast(ctx));
        self.offset = 0;
    }

    fn rollbackImpl(ctx: *anyopaque, to_len: usize) void {
        const self: *StandardKVCache = @ptrCast(@alignCast(ctx));
        self.offset = to_len;
    }

    fn filterImpl(
        ctx: *anyopaque,
        indices: []const usize,
        allocator: std.mem.Allocator,
    ) anyerror!void {
        const self: *StandardKVCache = @ptrCast(@alignCast(ctx));
        _ = allocator;

        if (indices.len == 0) {
            self.offset = 0;
            return;
        }

        // Build indices array for mlx_take_axis
        var idx_buf = try self.allocator.alloc(i32, indices.len);
        defer self.allocator.free(idx_buf);
        for (indices, 0..) |idx, i| {
            idx_buf[i] = @intCast(idx);
        }

        const idx_arr = c.c.mlx_array_new_data(
            idx_buf.ptr,
            &[_]i32{@intCast(indices.len)},
            1,
            c.c.MLX_INT32,
        );
        defer _ = c.c.mlx_array_free(idx_arr);

        // take along batch axis (axis 0)
        var new_keys = c.c.mlx_array_new();
        var new_values = c.c.mlx_array_new();
        const trim_stream = c.c.mlx_default_cpu_stream_new();
        defer _ = c.c.mlx_stream_free(trim_stream);
        try c.check(c.c.mlx_take_axis(&new_keys, self.keys.inner, idx_arr, 0, trim_stream));
        try c.check(c.c.mlx_take_axis(&new_values, self.values.inner, idx_arr, 0, trim_stream));

        self.keys.deinit();
        self.values.deinit();
        self.keys = Array.fromHandle(new_keys);
        self.values = Array.fromHandle(new_values);
        self.batch_size = indices.len;
    }

    fn getStateImpl(ctx: *anyopaque) ?iface.CacheState {
        const self: *StandardKVCache = @ptrCast(@alignCast(ctx));
        return .{ .keys = self.keys, .values = self.values, .offset = self.offset };
    }

    fn extendImpl(
        ctx: *anyopaque,
        sources: []KVCacheStrategy,
        allocator: std.mem.Allocator,
    ) anyerror!void {
        _ = allocator;
        const self: *StandardKVCache = @ptrCast(@alignCast(ctx));
        if (sources.len == 0) return;

        // Compute max cached length among self and all sources.
        var max_len: usize = self.offset;
        for (sources) |src| {
            max_len = @max(max_len, src.currentLen());
        }

        const stream = c.c.mlx_default_cpu_stream_new();
        defer _ = c.c.mlx_stream_free(stream);

        const k_vec = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(k_vec);
        const v_vec = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(v_vec);

        // Only include self if it holds actual cached data.
        const include_self = self.offset > 0;

        if (include_self) {
            try c.check(c.c.mlx_vector_array_append_data(k_vec, &self.keys.inner, 1));
            try c.check(c.c.mlx_vector_array_append_data(v_vec, &self.values.inner, 1));
        }

        for (sources) |src| {
            if (src.vtable.getState) |getStateFn| {
                const state = getStateFn(src.ptr);
                if (state) |s| {
                    try c.check(c.c.mlx_vector_array_append_data(k_vec, &s.keys.inner, 1));
                    try c.check(c.c.mlx_vector_array_append_data(v_vec, &s.values.inner, 1));
                } else {
                    return error.InvalidSourceCache;
                }
            } else {
                return error.InvalidSourceCache;
            }
        }

        var new_keys = c.c.mlx_array_new();
        var new_values = c.c.mlx_array_new();
        try c.check(c.c.mlx_concatenate_axis(&new_keys, k_vec, 0, stream));
        try c.check(c.c.mlx_concatenate_axis(&new_values, v_vec, 0, stream));

        self.keys.deinit();
        self.values.deinit();
        self.keys = Array.fromHandle(new_keys);
        self.values = Array.fromHandle(new_values);
        self.batch_size = if (include_self) self.batch_size + sources.len else sources.len;
        self.offset = max_len;
    }

    fn deinitImpl(ctx: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *StandardKVCache = @ptrCast(@alignCast(ctx));
        self.keys.deinit();
        self.values.deinit();
        allocator.destroy(self);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// In-place update: buffer[..., offset:end_offset, :] = new_kv
    fn sliceUpdateKV(
        buffer: *Array,
        new_kv: Array,
        offset: usize,
        end_offset: usize,
        stream: c.c.mlx_stream,
    ) !void {
        // slice_update expects: src, update, start, stop, strides
        const start = &[_]i32{ 0, 0, @intCast(offset), 0 };
        const stop = &[_]i32{ std.math.maxInt(i32), std.math.maxInt(i32), @intCast(end_offset), std.math.maxInt(i32) };
        const strides = &[_]i32{ 1, 1, 1, 1 };

        var res = c.c.mlx_array_new();
        try c.check(c.c.mlx_slice_update(
            &res,
            buffer.inner,
            new_kv.inner,
            start.ptr,
            start.len,
            stop.ptr,
            stop.len,
            strides.ptr,
            strides.len,
            stream,
        ));

        // mlx_slice_update returns a new array; replace the old one.
        buffer.deinit();
        buffer.* = Array.fromHandle(res);
    }

    /// Return a sliced view: buffer[..., 0:offset, :]
    fn sliceFetch(buffer: Array, offset: usize, stream: c.c.mlx_stream) !Array {
        const start = &[_]i32{ 0, 0, 0, 0 };
        const stop = &[_]i32{ std.math.maxInt(i32), std.math.maxInt(i32), @intCast(offset), std.math.maxInt(i32) };
        const strides = &[_]i32{ 1, 1, 1, 1 };

        var res = c.c.mlx_array_new();
        try c.check(c.c.mlx_slice(
            &res,
            buffer.inner,
            start.ptr,
            start.len,
            stop.ptr,
            stop.len,
            strides.ptr,
            strides.len,
            stream,
        ));
        return Array.fromHandle(res);
    }
};

/// Factory function conforming to StrategyFactory signature.
pub fn createStandard(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    stream: c.c.mlx_stream,
) !KVCacheStrategy {
    const cache = try allocator.create(StandardKVCache);
    errdefer allocator.destroy(cache);
    cache.* = try StandardKVCache.init(allocator, config, stream);
    return cache.asStrategy();
}
