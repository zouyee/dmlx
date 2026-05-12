/// RotatingKVCache — Sliding-window KV cache with fixed capacity.
///
/// Strategy: maintain a circular buffer of fixed size `window_size`.
/// When the buffer is full, new tokens overwrite the oldest ones.
/// The attention layer sees a contiguous view of the most recent `window_size` tokens.
///
/// This is the strategy used by models with limited context windows
/// (e.g., original LLaMA 2 with 4K window, Mistral 7B with 32K sliding window).
///
/// Pros: bounded memory regardless of sequence length.
/// Cons: tokens outside the window are lost (no access to distant context).
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const iface = @import("interface.zig");

const Array = array_mod.Array;
const KVSlice = iface.KVSlice;
const KVCacheStrategy = iface.KVCacheStrategy;
const LayerConfig = iface.LayerConfig;

pub const RotatingKVCache = struct {
    allocator: std.mem.Allocator,

    // Circular buffer: [batch, num_kv_heads, window_size, head_dim]
    keys: Array,
    values: Array,

    // Number of tokens seen so far (may exceed window_size).
    total_tokens: usize,

    // Current write position in the circular buffer.
    cursor: usize,

    // Shape parameters.
    batch_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    window_size: usize,

    pub const vtable: iface.VTable = .{
        .updateAndFetch = updateAndFetchImpl,
        .currentLen = currentLenImpl,
        .reset = resetImpl,
        .filter = filterImpl,
        .rollback = rollbackImpl,
        .extend = null,
        .deinit = deinitImpl,
    };

    /// Create a rotating cache with the given window size.
    /// If window_size == 0, uses config.max_seq_len (full context, no rotation).
    pub fn init(
        allocator: std.mem.Allocator,
        config: LayerConfig,
        window_size: usize,
        stream: c.c.mlx_stream,
    ) !RotatingKVCache {
        const cap = if (window_size == 0) config.max_seq_len else window_size;
        if (cap > config.max_seq_len) {
            return error.WindowExceedsMaxSeq;
        }

        const shape = &[_]i32{
            @intCast(config.batch_size),
            @intCast(config.num_kv_heads),
            @intCast(cap),
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
            .total_tokens = 0,
            .cursor = 0,
            .batch_size = config.batch_size,
            .num_kv_heads = config.num_kv_heads,
            .head_dim = config.head_dim,
            .window_size = cap,
        };
    }

    pub fn asStrategy(self: *RotatingKVCache) KVCacheStrategy {
        return .{ .ptr = self, .vtable = &vtable };
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
        const self: *RotatingKVCache = @ptrCast(@alignCast(ctx));
        const seq_len = @as(usize, @intCast(keys.shape()[2]));

        // Write tokens, wrapping around the circular buffer.
        var written: usize = 0;
        while (written < seq_len) : (written += 1) {
            // Extract single token from keys/values
            const token_k = try extractSingleToken(keys, written, stream);
            const token_v = try extractSingleToken(values, written, stream);

            try writeSingleToken(self, token_k, token_v, stream);

            self.cursor = (self.cursor + 1) % self.window_size;
            self.total_tokens += 1;
        }

        // Return the contiguous view of cached tokens.
        // If total_tokens <= window_size: simple slice [0:total_tokens].
        // If total_tokens > window_size: need to rotate the buffer to make it contiguous.
        const active_len = @min(self.total_tokens, self.window_size);

        if (self.total_tokens <= self.window_size) {
            const k = try sliceFetch(self.keys, active_len, stream);
            const v = try sliceFetch(self.values, active_len, stream);
            return .{ .keys = k, .values = v };
        } else {
            // Buffer is full and wrapped: rotate to make contiguous.
            // Newest tokens start at cursor, oldest at cursor-1.
            const k = try rotateAndFetch(self.keys, self.cursor, active_len, stream);
            const v = try rotateAndFetch(self.values, self.cursor, active_len, stream);
            return .{ .keys = k, .values = v };
        }
    }

    fn currentLenImpl(ctx: *anyopaque) usize {
        const self: *RotatingKVCache = @ptrCast(@alignCast(ctx));
        return @min(self.total_tokens, self.window_size);
    }

    fn resetImpl(ctx: *anyopaque) void {
        const self: *RotatingKVCache = @ptrCast(@alignCast(ctx));
        self.total_tokens = 0;
        self.cursor = 0;
    }

    fn rollbackImpl(ctx: *anyopaque, to_len: usize) void {
        const self: *RotatingKVCache = @ptrCast(@alignCast(ctx));
        self.total_tokens = to_len;
        if (to_len <= self.window_size) {
            self.cursor = to_len;
        } else {
            self.cursor = to_len % self.window_size;
        }
    }

    fn filterImpl(
        ctx: *anyopaque,
        indices: []const usize,
        allocator: std.mem.Allocator,
    ) !void {
        const self: *RotatingKVCache = @ptrCast(@alignCast(ctx));
        _ = allocator;

        if (indices.len == 0) {
            self.total_tokens = 0;
            self.cursor = 0;
            return;
        }

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

    fn deinitImpl(ctx: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *RotatingKVCache = @ptrCast(@alignCast(ctx));
        self.keys.deinit();
        self.values.deinit();
        allocator.destroy(self);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn extractSingleToken(batch_kv: Array, token_idx: usize, stream: c.c.mlx_stream) !Array {
        const start = &[_]i32{ 0, 0, @intCast(token_idx), 0 };
        const stop = &[_]i32{ std.math.maxInt(i32), std.math.maxInt(i32), @intCast(token_idx + 1), std.math.maxInt(i32) };
        const strides = &[_]i32{ 1, 1, 1, 1 };

        var res = c.c.mlx_array_new();
        try c.check(c.c.mlx_slice(
            &res,
            batch_kv.inner,
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

    fn writeSingleToken(
        self: *RotatingKVCache,
        token_k: Array,
        token_v: Array,
        stream: c.c.mlx_stream,
    ) !void {
        const pos = self.cursor;
        const start = &[_]i32{ 0, 0, @intCast(pos), 0 };
        const stop = &[_]i32{ std.math.maxInt(i32), std.math.maxInt(i32), @intCast(pos + 1), std.math.maxInt(i32) };
        const strides = &[_]i32{ 1, 1, 1, 1 };

        var new_keys = c.c.mlx_array_new();
        var new_values = c.c.mlx_array_new();

        try c.check(c.c.mlx_slice_update(
            &new_keys,
            self.keys.inner,
            token_k.inner,
            start.ptr,
            start.len,
            stop.ptr,
            stop.len,
            strides.ptr,
            strides.len,
            stream,
        ));
        try c.check(c.c.mlx_slice_update(
            &new_values,
            self.values.inner,
            token_v.inner,
            start.ptr,
            start.len,
            stop.ptr,
            stop.len,
            strides.ptr,
            strides.len,
            stream,
        ));

        self.keys.deinit();
        self.values.deinit();
        self.keys = Array.fromHandle(new_keys);
        self.values = Array.fromHandle(new_values);
    }

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

    /// Rotate circular buffer so that newest tokens are at the end.
    /// Returns a contiguous array of shape [B, H, active_len, D].
    fn rotateAndFetch(
        buffer: Array,
        cursor: usize,
        active_len: usize,
        stream: c.c.mlx_stream,
    ) !Array {
        _ = active_len;
        // Split buffer into [cursor:end] and [0:cursor], then concatenate.
        // For simplicity, use two slices and mlx_concatenate.
        // Note: cursor points to where the *next* write would go, so oldest token is at cursor.

        const tail_start = &[_]i32{ 0, 0, @intCast(cursor), 0 };
        const tail_stop = &[_]i32{ std.math.maxInt(i32), std.math.maxInt(i32), std.math.maxInt(i32), std.math.maxInt(i32) };
        const head_stop = &[_]i32{ std.math.maxInt(i32), std.math.maxInt(i32), @intCast(cursor), std.math.maxInt(i32) };
        const strides = &[_]i32{ 1, 1, 1, 1 };

        var tail = c.c.mlx_array_new();
        var head = c.c.mlx_array_new();

        try c.check(c.c.mlx_slice(&tail, buffer.inner, tail_start.ptr, tail_start.len, tail_stop.ptr, tail_stop.len, strides.ptr, strides.len, stream));
        try c.check(c.c.mlx_slice(&head, buffer.inner, &[_]i32{ 0, 0, 0, 0 }, 4, head_stop.ptr, head_stop.len, strides.ptr, strides.len, stream));

        // Build vector_array for concatenate
        const vec = c.c.mlx_vector_array_new_data(&[_]c.c.mlx_array{ tail, head }, 2);
        defer _ = c.c.mlx_vector_array_free(vec);

        var res = c.c.mlx_array_new();
        try c.check(c.c.mlx_concatenate_axis(&res, vec, 2, stream));

        _ = c.c.mlx_array_free(tail);
        _ = c.c.mlx_array_free(head);

        return Array.fromHandle(res);
    }
};

/// Factory for RotatingKVCache with a default window size.
/// Use `createRotatingWithWindow` if you need a specific window.
pub fn createRotating(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    stream: c.c.mlx_stream,
) !KVCacheStrategy {
    // Default: half of max_seq_len as window (common heuristic).
    const window = config.max_seq_len / 2;
    const cache = try allocator.create(RotatingKVCache);
    errdefer allocator.destroy(cache);
    cache.* = try RotatingKVCache.init(allocator, config, window, stream);
    return cache.asStrategy();
}

/// Factory with explicit window size.
pub fn createRotatingWithWindow(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    window_size: usize,
    stream: c.c.mlx_stream,
) !KVCacheStrategy {
    const cache = try allocator.create(RotatingKVCache);
    errdefer allocator.destroy(cache);
    cache.* = try RotatingKVCache.init(allocator, config, window_size, stream);
    return cache.asStrategy();
}
