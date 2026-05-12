/// BatchKVCache — Dynamic batch-size KV cache for continuous batching.
///
/// Reference: mlx-lm BatchKVCache (simplified).
///
/// Strategy: maintain contiguous [batch, heads, capacity, dim] arrays.
/// All batch entries share the same offset (left-justified, no left_padding).
/// This simplified version assumes uniform offsets across the batch.
///
/// Usage:
///   1. `BatchKVCache.merge(caches)` — create from per-request caches
///   2. `updateAndFetch` — append tokens for all batch entries
///   3. `filter(indices)` — shrink batch by keeping selected entries
///   4. `extend(sources)` — grow batch by merging new caches
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const iface = @import("interface.zig");

const Array = array_mod.Array;
const KVSlice = iface.KVSlice;
const KVCacheStrategy = iface.KVCacheStrategy;
const LayerConfig = iface.LayerConfig;

pub const BatchKVCache = struct {
    allocator: std.mem.Allocator,

    // [batch, num_kv_heads, capacity, head_dim]
    keys: ?Array,
    values: ?Array,

    // Per-batch cached lengths (assumed uniform in this simplified version)
    offsets: []usize,

    // Current max cached length across all batch entries
    max_offset: usize,

    // Pre-allocated capacity (grows by step)
    capacity: usize,
    step: usize,

    num_kv_heads: usize,
    head_dim: usize,
    dtype: @import("mlx").dtype.Dtype,

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

    pub fn init(allocator: std.mem.Allocator, config: LayerConfig) BatchKVCache {
        return .{
            .allocator = allocator,
            .keys = null,
            .values = null,
            .offsets = &[_]usize{},
            .max_offset = 0,
            .capacity = 0,
            .step = 256,
            .num_kv_heads = config.num_kv_heads,
            .head_dim = config.head_dim,
            .dtype = config.dtype,
        };
    }

    pub fn asStrategy(self: *BatchKVCache) KVCacheStrategy {
        return .{ .ptr = self, .vtable = &vtable };
    }

    /// Create a BatchKVCache by merging multiple independent caches.
    /// Each source cache must support `getState` (e.g. StandardKVCache).
    pub fn merge(
        allocator: std.mem.Allocator,
        caches: []KVCacheStrategy,
        stream: c.c.mlx_stream,
    ) !BatchKVCache {
        if (caches.len == 0) return error.EmptyMerge;

        var max_len: usize = 0;
        var num_kv_heads: usize = 0;
        var head_dim: usize = 0;
        var dtype: @import("mlx").dtype.Dtype = .float32;

        // Collect states and validate
        var states = try allocator.alloc(?iface.CacheState, caches.len);
        defer allocator.free(states);

        for (caches, 0..) |cache, i| {
            if (cache.vtable.getState) |getStateFn| {
                states[i] = getStateFn(cache.ptr);
            } else {
                return error.InvalidSourceCache;
            }
            if (states[i]) |s| {
                max_len = @max(max_len, s.offset);
                const shape = s.keys.shape();
                num_kv_heads = @max(num_kv_heads, @as(usize, @intCast(shape[1])));
                head_dim = @max(head_dim, @as(usize, @intCast(shape[3])));
                // Use dtype from first valid cache
                if (i == 0) dtype = .float32; // TODO: extract actual dtype
            }
        }

        if (max_len == 0) {
            return error.EmptyMerge;
        }

        const batch = caches.len;
        const k_shape = &[_]i32{
            @intCast(batch),
            @intCast(num_kv_heads),
            @intCast(max_len),
            @intCast(head_dim),
        };

        var keys_arr = c.c.mlx_array_new();
        var values_arr = c.c.mlx_array_new();
        try c.check(c.c.mlx_zeros(&keys_arr, k_shape.ptr, k_shape.len, @intCast(@intFromEnum(dtype)), stream));
        try c.check(c.c.mlx_zeros(&values_arr, k_shape.ptr, k_shape.len, @intCast(@intFromEnum(dtype)), stream));

        // Copy each cache's data into the merged buffer
        for (caches, 0..) |_, i| {
            if (states[i]) |s| {
                const offset = s.offset;
                const src_shape = s.keys.shape();
                const src_heads = @as(usize, @intCast(src_shape[1]));
                const src_dim = @as(usize, @intCast(src_shape[3]));
                const pad_left = max_len - offset;

                // Copy keys: keys[i:i+1, 0:src_heads, pad_left:pad_left+offset, 0:src_dim]
                const k_start = &[_]i32{ @intCast(i), 0, @intCast(pad_left), 0 };
                const k_stop = &[_]i32{ @intCast(i + 1), @intCast(src_heads), @intCast(pad_left + offset), @intCast(src_dim) };
                const k_strides = &[_]i32{ 1, 1, 1, 1 };

                var k_slice = c.c.mlx_array_new();
                try c.check(c.c.mlx_slice(
                    &k_slice,
                    keys_arr,
                    k_start.ptr,
                    k_start.len,
                    k_stop.ptr,
                    k_stop.len,
                    k_strides.ptr,
                    k_strides.len,
                    stream,
                ));
                defer _ = c.c.mlx_array_free(k_slice);

                const src_k_start = &[_]i32{ 0, 0, 0, 0 };
                const src_k_stop = &[_]i32{ 1, @intCast(src_heads), @intCast(offset), @intCast(src_dim) };
                try c.check(c.c.mlx_slice_update(
                    &keys_arr,
                    k_slice,
                    s.keys.inner,
                    src_k_start.ptr,
                    src_k_start.len,
                    src_k_stop.ptr,
                    src_k_stop.len,
                    k_strides.ptr,
                    k_strides.len,
                    stream,
                ));

                // Copy values
                var v_slice = c.c.mlx_array_new();
                try c.check(c.c.mlx_slice(
                    &v_slice,
                    values_arr,
                    k_start.ptr,
                    k_start.len,
                    k_stop.ptr,
                    k_stop.len,
                    k_strides.ptr,
                    k_strides.len,
                    stream,
                ));
                defer _ = c.c.mlx_array_free(v_slice);

                try c.check(c.c.mlx_slice_update(
                    &values_arr,
                    v_slice,
                    s.values.inner,
                    src_k_start.ptr,
                    src_k_start.len,
                    src_k_stop.ptr,
                    src_k_stop.len,
                    k_strides.ptr,
                    k_strides.len,
                    stream,
                ));
            }
        }

        var offsets = try allocator.alloc(usize, batch);
        for (caches, 0..) |cache, i| {
            offsets[i] = cache.currentLen();
        }

        return .{
            .allocator = allocator,
            .keys = Array.fromHandle(keys_arr),
            .values = Array.fromHandle(values_arr),
            .offsets = offsets,
            .max_offset = max_len,
            .capacity = max_len,
            .step = 256,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .dtype = dtype,
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
        const self: *BatchKVCache = @ptrCast(@alignCast(ctx));
        const shape = keys.shape();
        const batch = @as(usize, @intCast(shape[0]));
        const seq_len = @as(usize, @intCast(shape[2]));

        if (batch != self.offsets.len) {
            return error.BatchSizeMismatch;
        }

        const new_max = self.max_offset + seq_len;
        if (self.keys == null or new_max > self.capacity) {
            try self.growCapacity(new_max, batch, stream);
        }

        const k = self.keys.?;
        const v = self.values.?;

        // Check if all offsets are uniform
        var uniform = true;
        const first_offset = if (self.offsets.len > 0) self.offsets[0] else 0;
        for (self.offsets) |o| {
            if (o != first_offset) {
                uniform = false;
                break;
            }
        }

        const strides = &[_]i32{ 1, 1, 1, 1 };

        if (uniform) {
            // Fast path: all entries have same offset — single batch slice_update
            const start = &[_]i32{ 0, 0, @intCast(first_offset), 0 };
            const stop = &[_]i32{ @intCast(batch), std.math.maxInt(i32), @intCast(first_offset + seq_len), std.math.maxInt(i32) };

            var new_keys = c.c.mlx_array_new();
            try c.check(c.c.mlx_slice_update(
                &new_keys,
                k.inner,
                keys.inner,
                start.ptr,
                start.len,
                stop.ptr,
                stop.len,
                strides.ptr,
                strides.len,
                stream,
            ));
            k.deinit();
            self.keys = Array.fromHandle(new_keys);

            var new_values = c.c.mlx_array_new();
            try c.check(c.c.mlx_slice_update(
                &new_values,
                v.inner,
                values.inner,
                start.ptr,
                start.len,
                stop.ptr,
                stop.len,
                strides.ptr,
                strides.len,
                stream,
            ));
            v.deinit();
            self.values = Array.fromHandle(new_values);
        } else {
            // Slow path: per-batch-entry offset — loop over batch entries
            var cur_k = k;
            var cur_v = v;
            for (0..batch) |i| {
                const offset = self.offsets[i];
                const start = &[_]i32{ @intCast(i), 0, @intCast(offset), 0 };
                const stop = &[_]i32{ @intCast(i + 1), std.math.maxInt(i32), @intCast(offset + seq_len), std.math.maxInt(i32) };

                var new_k = c.c.mlx_array_new();
                try c.check(c.c.mlx_slice_update(
                    &new_k,
                    cur_k.inner,
                    keys.inner,
                    start.ptr,
                    start.len,
                    stop.ptr,
                    stop.len,
                    strides.ptr,
                    strides.len,
                    stream,
                ));
                cur_k.deinit();
                cur_k = Array.fromHandle(new_k);

                var new_v = c.c.mlx_array_new();
                try c.check(c.c.mlx_slice_update(
                    &new_v,
                    cur_v.inner,
                    values.inner,
                    start.ptr,
                    start.len,
                    stop.ptr,
                    stop.len,
                    strides.ptr,
                    strides.len,
                    stream,
                ));
                cur_v.deinit();
                cur_v = Array.fromHandle(new_v);
            }
            self.keys = cur_k;
            self.values = cur_v;
        }

        for (self.offsets) |*o| o.* += seq_len;
        self.max_offset += seq_len;

        // Return sliced view [0:max_offset]
        const f_start = &[_]i32{ 0, 0, 0, 0 };
        const f_stop = &[_]i32{ @intCast(batch), std.math.maxInt(i32), @intCast(self.max_offset), std.math.maxInt(i32) };
        const f_strides = &[_]i32{ 1, 1, 1, 1 };

        var k_res = c.c.mlx_array_new();
        try c.check(c.c.mlx_slice(
            &k_res,
            self.keys.?.inner,
            f_start.ptr,
            f_start.len,
            f_stop.ptr,
            f_stop.len,
            f_strides.ptr,
            f_strides.len,
            stream,
        ));
        var v_res = c.c.mlx_array_new();
        try c.check(c.c.mlx_slice(
            &v_res,
            self.values.?.inner,
            f_start.ptr,
            f_start.len,
            f_stop.ptr,
            f_stop.len,
            f_strides.ptr,
            f_strides.len,
            stream,
        ));

        return .{ .keys = Array.fromHandle(k_res), .values = Array.fromHandle(v_res) };
    }

    fn currentLenImpl(ctx: *anyopaque) usize {
        const self: *BatchKVCache = @ptrCast(@alignCast(ctx));
        return self.max_offset;
    }

    fn getStateImpl(ctx: *anyopaque) ?iface.CacheState {
        const self: *BatchKVCache = @ptrCast(@alignCast(ctx));
        if (self.keys == null or self.values == null) return null;
        return .{
            .keys = self.keys.?,
            .values = self.values.?,
            .offset = self.max_offset,
        };
    }

    fn resetImpl(ctx: *anyopaque) void {
        const self: *BatchKVCache = @ptrCast(@alignCast(ctx));
        if (self.keys) |k| {
            k.deinit();
            self.keys = null;
        }
        if (self.values) |v| {
            v.deinit();
            self.values = null;
        }
        self.allocator.free(self.offsets);
        self.offsets = &[_]usize{};
        self.max_offset = 0;
        self.capacity = 0;
    }

    fn rollbackImpl(ctx: *anyopaque, to_len: usize) void {
        const self: *BatchKVCache = @ptrCast(@alignCast(ctx));
        for (self.offsets) |*o| o.* = @min(o.*, to_len);
        self.max_offset = 0;
        for (self.offsets) |o| {
            if (o > self.max_offset) {
                self.max_offset = o;
            }
        }
    }

    fn filterImpl(
        ctx: *anyopaque,
        indices: []const usize,
        allocator: std.mem.Allocator,
    ) anyerror!void {
        _ = allocator;
        const self: *BatchKVCache = @ptrCast(@alignCast(ctx));

        if (indices.len == 0) {
            if (self.keys) |k| {
                k.deinit();
                self.keys = null;
            }
            if (self.values) |v| {
                v.deinit();
                self.values = null;
            }
            self.allocator.free(self.offsets);
            self.offsets = &[_]usize{};
            self.max_offset = 0;
            self.capacity = 0;
            return;
        }

        var idx_buf = try self.allocator.alloc(i32, indices.len);
        defer self.allocator.free(idx_buf);
        for (indices, 0..) |idx, i| idx_buf[i] = @intCast(idx);

        const idx_arr = c.c.mlx_array_new_data(idx_buf.ptr, &[_]i32{@intCast(indices.len)}, 1, c.c.MLX_INT32);
        defer _ = c.c.mlx_array_free(idx_arr);

        const stream = c.c.mlx_default_cpu_stream_new();
        defer _ = c.c.mlx_stream_free(stream);

        if (self.keys) |keys| {
            var new_keys = c.c.mlx_array_new();
            try c.check(c.c.mlx_take_axis(&new_keys, keys.inner, idx_arr, 0, stream));
            keys.deinit();
            self.keys = Array.fromHandle(new_keys);
        }
        if (self.values) |values| {
            var new_values = c.c.mlx_array_new();
            try c.check(c.c.mlx_take_axis(&new_values, values.inner, idx_arr, 0, stream));
            values.deinit();
            self.values = Array.fromHandle(new_values);
        }

        var new_offsets = try self.allocator.alloc(usize, indices.len);
        for (indices, 0..) |idx, i| new_offsets[i] = self.offsets[idx];
        self.allocator.free(self.offsets);
        self.offsets = new_offsets;

        self.max_offset = 0;
        for (self.offsets) |o| {
            if (o > self.max_offset) {
                self.max_offset = o;
            }
        }
    }

    fn extendImpl(
        ctx: *anyopaque,
        sources: []KVCacheStrategy,
        allocator: std.mem.Allocator,
    ) anyerror!void {
        _ = allocator;
        const self: *BatchKVCache = @ptrCast(@alignCast(ctx));
        if (sources.len == 0) return;

        const stream = c.c.mlx_default_cpu_stream_new();
        defer _ = c.c.mlx_stream_free(stream);

        // Compute max offset
        var max_len: usize = self.max_offset;
        for (sources) |src| {
            max_len = @max(max_len, src.currentLen());
        }

        // Build vector arrays for concatenation along batch axis
        const k_vec = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(k_vec);
        const v_vec = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(v_vec);

        var self_keys_deinited = false;
        var self_values_deinited = false;

        // Append self (if exists)
        if (self.keys) |keys| {
            if (self.max_offset < max_len) {
                const pad_right = max_len - self.max_offset;
                const pad_arr = c.c.mlx_array_new_float32(0.0);
                defer _ = c.c.mlx_array_free(pad_arr);
                const axes = &[_]i32{2};
                const low = &[_]i32{0};
                const high = &[_]i32{@intCast(pad_right)};
                var padded = c.c.mlx_array_new();
                try c.check(c.c.mlx_pad(&padded, keys.inner, axes.ptr, 1, low.ptr, 1, high.ptr, 1, pad_arr, "constant", stream));
                keys.deinit();
                self_keys_deinited = true;
                try c.check(c.c.mlx_vector_array_append_data(k_vec, &padded, 1));
                _ = c.c.mlx_array_free(padded);
            } else {
                try c.check(c.c.mlx_vector_array_append_data(k_vec, &keys.inner, 1));
            }
            if (self.values) |values| {
                if (self.max_offset < max_len) {
                    const pad_right = max_len - self.max_offset;
                    const pad_arr = c.c.mlx_array_new_float32(0.0);
                    defer _ = c.c.mlx_array_free(pad_arr);
                    const axes = &[_]i32{2};
                    const low = &[_]i32{0};
                    const high = &[_]i32{@intCast(pad_right)};
                    var padded = c.c.mlx_array_new();
                    try c.check(c.c.mlx_pad(&padded, values.inner, axes.ptr, 1, low.ptr, 1, high.ptr, 1, pad_arr, "constant", stream));
                    values.deinit();
                    self_values_deinited = true;
                    try c.check(c.c.mlx_vector_array_append_data(v_vec, &padded, 1));
                    _ = c.c.mlx_array_free(padded);
                } else {
                    try c.check(c.c.mlx_vector_array_append_data(v_vec, &values.inner, 1));
                }
            }
        }

        // Append each source
        for (sources, 0..) |src, src_idx| {
            if (src.vtable.getState) |getStateFn| {
                const state = getStateFn(src.ptr);
                if (state) |s| {
                    const src_len = s.offset;
                    if (src_len < max_len) {
                        const pad_right = max_len - src_len;
                        const pad_arr = c.c.mlx_array_new_float32(0.0);
                        defer _ = c.c.mlx_array_free(pad_arr);
                        const axes = &[_]i32{2};
                        const low = &[_]i32{0};
                        const high = &[_]i32{@intCast(pad_right)};
                        var padded_k = c.c.mlx_array_new();
                        try c.check(c.c.mlx_pad(&padded_k, s.keys.inner, axes.ptr, 1, low.ptr, 1, high.ptr, 1, pad_arr, "constant", stream));
                        try c.check(c.c.mlx_vector_array_append_data(k_vec, &padded_k, 1));
                        _ = c.c.mlx_array_free(padded_k);

                        var padded_v = c.c.mlx_array_new();
                        try c.check(c.c.mlx_pad(&padded_v, s.values.inner, axes.ptr, 1, low.ptr, 1, high.ptr, 1, pad_arr, "constant", stream));
                        try c.check(c.c.mlx_vector_array_append_data(v_vec, &padded_v, 1));
                        _ = c.c.mlx_array_free(padded_v);
                    } else {
                        try c.check(c.c.mlx_vector_array_append_data(k_vec, &s.keys.inner, 1));
                        try c.check(c.c.mlx_vector_array_append_data(v_vec, &s.values.inner, 1));
                    }
                } else {
                    std.log.err("[BatchKVCache.extend] source {d}: getState returned null", .{src_idx});
                    return error.InvalidSourceCache;
                }
            } else {
                std.log.err("[BatchKVCache.extend] source {d}: vtable.getState is null", .{src_idx});
                return error.InvalidSourceCache;
            }
        }

        var new_keys = c.c.mlx_array_new();
        try c.check(c.c.mlx_concatenate_axis(&new_keys, k_vec, 0, stream));
        var new_values = c.c.mlx_array_new();
        try c.check(c.c.mlx_concatenate_axis(&new_values, v_vec, 0, stream));

        // Deinit old arrays if not already deinit (non-padded path)
        if (!self_keys_deinited and self.keys != null) {
            self.keys.?.deinit();
        }
        if (!self_values_deinited and self.values != null) {
            self.values.?.deinit();
        }
        self.keys = Array.fromHandle(new_keys);
        self.values = Array.fromHandle(new_values);

        // Update offsets
        var new_offsets_len = if (self.offsets.len > 0) self.offsets.len else 0;
        new_offsets_len += sources.len;
        var new_offsets = try self.allocator.alloc(usize, new_offsets_len);

        var offset_idx: usize = 0;
        if (self.offsets.len > 0) {
            for (self.offsets) |o| {
                new_offsets[offset_idx] = o;
                offset_idx += 1;
            }
            self.allocator.free(self.offsets);
        }
        for (sources) |src| {
            new_offsets[offset_idx] = src.currentLen();
            offset_idx += 1;
        }
        self.offsets = new_offsets;
        self.max_offset = max_len;
        self.capacity = max_len;
    }

    fn deinitImpl(ctx: *anyopaque, allocator: std.mem.Allocator) void {
        _ = allocator;
        const self: *BatchKVCache = @ptrCast(@alignCast(ctx));
        if (self.keys) |k| k.deinit();
        if (self.values) |v| v.deinit();
        if (self.offsets.len > 0) self.allocator.free(self.offsets);
        self.allocator.destroy(self);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn growCapacity(self: *BatchKVCache, min_len: usize, batch: usize, stream: c.c.mlx_stream) !void {
        if (self.keys == null) {
            const new_cap = (min_len + self.step - 1) / self.step * self.step;
            const shape = &[_]i32{
                @intCast(batch),
                @intCast(self.num_kv_heads),
                @intCast(new_cap),
                @intCast(self.head_dim),
            };
            var k = c.c.mlx_array_new();
            var v = c.c.mlx_array_new();
            try c.check(c.c.mlx_zeros(&k, shape.ptr, shape.len, @intCast(@intFromEnum(self.dtype)), stream));
            try c.check(c.c.mlx_zeros(&v, shape.ptr, shape.len, @intCast(@intFromEnum(self.dtype)), stream));
            self.keys = Array.fromHandle(k);
            self.values = Array.fromHandle(v);
            self.capacity = new_cap;
            return;
        }

        if (min_len <= self.capacity) return;

        const new_cap = (min_len + self.step - 1) / self.step * self.step;
        const ext_len = new_cap - self.capacity;
        const ext_shape = &[_]i32{
            @intCast(batch),
            @intCast(self.num_kv_heads),
            @intCast(ext_len),
            @intCast(self.head_dim),
        };

        var k_ext = c.c.mlx_array_new();
        var v_ext = c.c.mlx_array_new();
        try c.check(c.c.mlx_zeros(&k_ext, ext_shape.ptr, ext_shape.len, @intCast(@intFromEnum(self.dtype)), stream));
        try c.check(c.c.mlx_zeros(&v_ext, ext_shape.ptr, ext_shape.len, @intCast(@intFromEnum(self.dtype)), stream));

        const k_vec = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(k_vec);
        try c.check(c.c.mlx_vector_array_append_data(k_vec, &self.keys.?.inner, 1));
        try c.check(c.c.mlx_vector_array_append_data(k_vec, &k_ext, 1));
        var new_keys = c.c.mlx_array_new();
        try c.check(c.c.mlx_concatenate_axis(&new_keys, k_vec, 2, stream));
        self.keys.?.deinit();
        self.keys = Array.fromHandle(new_keys);
        _ = c.c.mlx_array_free(k_ext);

        const v_vec = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(v_vec);
        try c.check(c.c.mlx_vector_array_append_data(v_vec, &self.values.?.inner, 1));
        try c.check(c.c.mlx_vector_array_append_data(v_vec, &v_ext, 1));
        var new_values = c.c.mlx_array_new();
        try c.check(c.c.mlx_concatenate_axis(&new_values, v_vec, 2, stream));
        self.values.?.deinit();
        self.values = Array.fromHandle(new_values);
        _ = c.c.mlx_array_free(v_ext);

        self.capacity = new_cap;
    }
};

/// Factory function conforming to StrategyFactory signature.
pub fn createBatch(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    stream: c.c.mlx_stream,
) !KVCacheStrategy {
    _ = stream;
    const cache = try allocator.create(BatchKVCache);
    errdefer allocator.destroy(cache);
    cache.* = BatchKVCache.init(allocator, config);
    return cache.asStrategy();
}

// ============================================================
// Unit Tests
// ============================================================

test "BatchKVCache.merge creates batch from two caches" {
    const allocator = std.testing.allocator;
    c.initErrorHandler();

    const stream = c.c.mlx_default_cpu_stream_new();
    defer _ = c.c.mlx_stream_free(stream);

    // Create two StandardKVCache instances
    const config1 = iface.LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 4,
        .head_dim = 64,
        .max_seq_len = 128,
        .dtype = .float32,
    };

    var cache1 = try @import("standard.zig").StandardKVCache.init(allocator, config1, stream);
    defer cache1.deinit(allocator);

    var cache2 = try @import("standard.zig").StandardKVCache.init(allocator, config1, stream);
    defer cache2.deinit(allocator);

    // Append some data to cache1
    const shape = &[_]i32{ 1, 4, 3, 64 };
    var keys1 = c.c.mlx_array_new();
    var values1 = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&keys1, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&values1, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    _ = try cache1.asStrategy().updateAndFetch(Array.fromHandle(keys1), Array.fromHandle(values1), stream);

    // Append some data to cache2
    var keys2 = c.c.mlx_array_new();
    var values2 = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&keys2, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&values2, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    _ = try cache2.asStrategy().updateAndFetch(Array.fromHandle(keys2), Array.fromHandle(values2), stream);

    // Merge caches
    const caches = [_]KVCacheStrategy{ cache1.asStrategy(), cache2.asStrategy() };
    const batch = try BatchKVCache.merge(allocator, &caches, stream);
    defer {
        if (batch.keys) |k| k.deinit();
        if (batch.values) |v| v.deinit();
        if (batch.offsets.len > 0) allocator.free(batch.offsets);
    }

    // Verify batch size = 2
    try std.testing.expectEqual(@as(usize, 2), batch.offsets.len);
    try std.testing.expect(batch.keys != null);
    try std.testing.expect(batch.values != null);

    // Verify shape is [2, 4, 3, 64]
    const k_shape = batch.keys.?.shape();
    try std.testing.expectEqual(@as(usize, 2), @as(usize, @intCast(k_shape[0])));
    try std.testing.expectEqual(@as(usize, 4), @as(usize, @intCast(k_shape[1])));
    try std.testing.expectEqual(@as(usize, 3), @as(usize, @intCast(k_shape[2])));
    try std.testing.expectEqual(@as(usize, 64), @as(usize, @intCast(k_shape[3])));
}

test "BatchKVCache.updateAndFetch appends tokens" {
    const allocator = std.testing.allocator;
    c.initErrorHandler();

    const stream = c.c.mlx_default_cpu_stream_new();
    defer _ = c.c.mlx_stream_free(stream);

    // Create BatchKVCache via merge
    const config = iface.LayerConfig{
        .batch_size = 1,
        .num_heads = 2,
        .num_kv_heads = 2,
        .head_dim = 32,
        .max_seq_len = 64,
        .dtype = .float32,
    };

    var cache1 = try @import("standard.zig").StandardKVCache.init(allocator, config, stream);
    defer cache1.deinit(allocator);

    // Initial append
    const shape = &[_]i32{ 1, 2, 2, 32 };
    var keys = c.c.mlx_array_new();
    var values = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&keys, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&values, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    _ = try cache1.asStrategy().updateAndFetch(Array.fromHandle(keys), Array.fromHandle(values), stream);

    const caches = [_]KVCacheStrategy{cache1.asStrategy()};
    const batch = try BatchKVCache.merge(allocator, &caches, stream);
    defer {
        if (batch.keys) |k| k.deinit();
        if (batch.values) |v| v.deinit();
        if (batch.offsets.len > 0) allocator.free(batch.offsets);
    }

    try std.testing.expectEqual(@as(usize, 2), batch.max_offset);

    // Update via vtable
    var batch_ptr = try allocator.create(BatchKVCache);
    batch_ptr.* = batch;
    const strategy = batch_ptr.asStrategy();

    const new_shape = &[_]i32{ 1, 2, 3, 32 };
    var new_keys = c.c.mlx_array_new();
    var new_values = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&new_keys, new_shape.ptr, new_shape.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&new_values, new_shape.ptr, new_shape.len, c.c.MLX_FLOAT32, stream));

    const result = try strategy.updateAndFetch(Array.fromHandle(new_keys), Array.fromHandle(new_values), stream);
    defer {
        result.keys.deinit();
        result.values.deinit();
    }

    // Verify offset updated
    try std.testing.expectEqual(@as(usize, 5), batch_ptr.max_offset);

    // Verify returned shape [1, 2, 5, 32]
    const result_shape = result.keys.shape();
    try std.testing.expectEqual(@as(usize, 1), @as(usize, @intCast(result_shape[0])));
    try std.testing.expectEqual(@as(usize, 2), @as(usize, @intCast(result_shape[1])));
    try std.testing.expectEqual(@as(usize, 5), @as(usize, @intCast(result_shape[2])));
    try std.testing.expectEqual(@as(usize, 32), @as(usize, @intCast(result_shape[3])));
}

test "BatchKVCache.filter shrinks batch to selected indices" {
    const allocator = std.testing.allocator;
    c.initErrorHandler();

    const stream = c.c.mlx_default_cpu_stream_new();
    defer _ = c.c.mlx_stream_free(stream);

    // Create three StandardKVCache instances
    const config = iface.LayerConfig{
        .batch_size = 1,
        .num_heads = 2,
        .num_kv_heads = 2,
        .head_dim = 16,
        .max_seq_len = 32,
        .dtype = .float32,
    };

    var cache1 = try @import("standard.zig").StandardKVCache.init(allocator, config, stream);
    defer cache1.deinit(allocator);
    var cache2 = try @import("standard.zig").StandardKVCache.init(allocator, config, stream);
    defer cache2.deinit(allocator);
    var cache3 = try @import("standard.zig").StandardKVCache.init(allocator, config, stream);
    defer cache3.deinit(allocator);

    // Append data to each
    const shape = &[_]i32{ 1, 2, 2, 16 };
    for ([_]*@import("standard.zig").StandardKVCache{ &cache1, &cache2, &cache3 }) |cache| {
        var keys = c.c.mlx_array_new();
        var values = c.c.mlx_array_new();
        try check(c.c.mlx_ones(&keys, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
        try check(c.c.mlx_ones(&values, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
        _ = try cache.asStrategy().updateAndFetch(Array.fromHandle(keys), Array.fromHandle(values), stream);
    }

    // Merge into batch of 3
    const caches = [_]KVCacheStrategy{ cache1.asStrategy(), cache2.asStrategy(), cache3.asStrategy() };
    const batch = try BatchKVCache.merge(allocator, &caches, stream);
    defer {
        if (batch.keys) |k| k.deinit();
        if (batch.values) |v| v.deinit();
        if (batch.offsets.len > 0) allocator.free(batch.offsets);
    }

    try std.testing.expectEqual(@as(usize, 3), batch.offsets.len);

    // Filter to keep indices [0, 2]
    var batch_ptr = try allocator.create(BatchKVCache);
    batch_ptr.* = batch;
    const strategy = batch_ptr.asStrategy();

    const indices = [_]usize{ 0, 2 };
    try strategy.filter(&indices, allocator);

    // Verify batch size = 2
    try std.testing.expectEqual(@as(usize, 2), batch_ptr.offsets.len);

    // Verify shape is [2, 2, 2, 16]
    const k_shape = batch_ptr.keys.?.shape();
    try std.testing.expectEqual(@as(usize, 2), @as(usize, @intCast(k_shape[0])));
}

test "BatchKVCache.extend grows batch by appending caches" {
    const allocator = std.testing.allocator;
    c.initErrorHandler();

    const stream = c.c.mlx_default_cpu_stream_new();
    defer _ = c.c.mlx_stream_free(stream);

    // Create initial batch with 1 cache
    const config = iface.LayerConfig{
        .batch_size = 1,
        .num_heads = 2,
        .num_kv_heads = 2,
        .head_dim = 16,
        .max_seq_len = 32,
        .dtype = .float32,
    };

    var cache1 = try @import("standard.zig").StandardKVCache.init(allocator, config, stream);
    defer cache1.deinit(allocator);

    const shape = &[_]i32{ 1, 2, 2, 16 };
    var keys1 = c.c.mlx_array_new();
    var values1 = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&keys1, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&values1, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    _ = try cache1.asStrategy().updateAndFetch(Array.fromHandle(keys1), Array.fromHandle(values1), stream);

    const caches = [_]KVCacheStrategy{cache1.asStrategy()};
    const batch = try BatchKVCache.merge(allocator, &caches, stream);
    defer {
        if (batch.keys) |k| k.deinit();
        if (batch.values) |v| v.deinit();
        if (batch.offsets.len > 0) allocator.free(batch.offsets);
    }

    try std.testing.expectEqual(@as(usize, 1), batch.offsets.len);

    // Create two more caches to extend with
    var cache2 = try @import("standard.zig").StandardKVCache.init(allocator, config, stream);
    defer cache2.deinit(allocator);
    var cache3 = try @import("standard.zig").StandardKVCache.init(allocator, config, stream);
    defer cache3.deinit(allocator);

    for ([_]*@import("standard.zig").StandardKVCache{ &cache2, &cache3 }) |cache| {
        var keys = c.c.mlx_array_new();
        var values = c.c.mlx_array_new();
        try c.check(c.c.mlx_ones(&keys, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
        try c.check(c.c.mlx_ones(&values, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
        _ = try cache.asStrategy().updateAndFetch(Array.fromHandle(keys), Array.fromHandle(values), stream);
    }

    // Extend batch
    var batch_ptr = try allocator.create(BatchKVCache);
    batch_ptr.* = batch;
    const strategy = batch_ptr.asStrategy();

    const sources = [_]KVCacheStrategy{ cache2.asStrategy(), cache3.asStrategy() };
    try strategy.extend(&sources, allocator);

    // Verify batch size = 3
    try std.testing.expectEqual(@as(usize, 3), batch_ptr.offsets.len);

    // Verify shape is [3, 2, 2, 16]
    const k_shape = batch_ptr.keys.?.shape();
    try std.testing.expectEqual(@as(usize, 3), @as(usize, @intCast(k_shape[0])));
}

test "BatchKVCache.reset clears all data" {
    const allocator = std.testing.allocator;
    c.initErrorHandler();

    const stream = c.c.mlx_default_cpu_stream_new();
    defer _ = c.c.mlx_stream_free(stream);

    const config = iface.LayerConfig{
        .batch_size = 1,
        .num_heads = 2,
        .num_kv_heads = 2,
        .head_dim = 16,
        .max_seq_len = 32,
        .dtype = .float32,
    };

    var cache1 = try @import("standard.zig").StandardKVCache.init(allocator, config, stream);
    defer cache1.deinit(allocator);

    const shape = &[_]i32{ 1, 2, 2, 16 };
    var keys = c.c.mlx_array_new();
    var values = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&keys, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&values, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    _ = try cache1.asStrategy().updateAndFetch(Array.fromHandle(keys), Array.fromHandle(values), stream);

    const caches = [_]KVCacheStrategy{cache1.asStrategy()};
    const batch = try BatchKVCache.merge(allocator, &caches, stream);

    try std.testing.expect(batch.keys != null);

    // Reset
    var batch_ptr = try allocator.create(BatchKVCache);
    batch_ptr.* = batch;
    const strategy = batch_ptr.asStrategy();
    strategy.reset();

    // Verify cleared
    try std.testing.expect(batch_ptr.keys == null);
    try std.testing.expect(batch_ptr.values == null);
    try std.testing.expectEqual(@as(usize, 0), batch_ptr.offsets.len);
    try std.testing.expectEqual(@as(usize, 0), batch_ptr.max_offset);

    allocator.destroy(batch_ptr);
}

fn check(result: c.c.mlx_error) !void {
    return c.check(result);
}
