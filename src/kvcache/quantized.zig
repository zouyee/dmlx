/// QuantizedKVCache — KV cache with configurable quantization precision.
///
/// Supports kv_bits = 4, 8, or 16:
///   - 4-bit: ~4x memory reduction vs float16, slight quality loss
///   - 8-bit: ~2x memory reduction vs float16, minimal quality loss
///   - 16-bit: passthrough mode — no quantization, delegates to StandardKVCache behavior
///
/// Uses MLX's mlx_quantize/mlx_dequantize with configurable group_size.
///
/// Requirements: R11.1, R11.2, R11.3, R11.4
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const iface = @import("interface.zig");

const Array = array_mod.Array;
const KVSlice = iface.KVSlice;
const KVCacheStrategy = iface.KVCacheStrategy;
const LayerConfig = iface.LayerConfig;

/// Configuration for KV cache quantization.
pub const QuantConfig = struct {
    kv_bits: u8 = 16, // 4, 8, or 16
    group_size: i32 = 64,

    pub fn validate(self: QuantConfig) !void {
        switch (self.kv_bits) {
            4, 8, 16 => {},
            else => return error.InvalidQuantBits,
        }
        if (self.group_size <= 0) return error.InvalidGroupSize;
    }
};

/// Quantized KV cache storing (packed, scales, biases) tuples for K and V.
pub const QuantizedKVCache = struct {
    allocator: std.mem.Allocator,

    // When kv_bits < 16: quantized storage
    keys: ?QuantizedTuple,
    values: ?QuantizedTuple,

    // When kv_bits == 16: raw float storage (passthrough)
    raw_keys: ?Array,
    raw_values: ?Array,

    // Current sequence length.
    offset: usize,

    // Expansion step size (allocate in chunks).
    step: usize,

    // Quantization parameters.
    group_size: i32,
    bits: i32,
    kv_bits: u8,

    // Shape.
    batch_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,

    // Elements per uint32 (for packed data).
    el_per_int: i32,

    pub const QuantizedTuple = struct {
        packed_data: Array,
        scales: Array,
        biases: Array,
    };

    pub const vtable: iface.VTable = .{
        .updateAndFetch = updateAndFetchImpl,
        .currentLen = currentLenImpl,
        .reset = resetImpl,
        .filter = filterImpl,
        .rollback = rollbackImpl,
        .deinit = deinitImpl,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        config: LayerConfig,
        group_size: i32,
        bits: i32,
        stream: c.c.mlx_stream,
    ) !QuantizedKVCache {
        _ = stream;
        const kv_bits: u8 = @intCast(bits);
        const qconfig = QuantConfig{ .kv_bits = kv_bits, .group_size = group_size };
        try qconfig.validate();

        const el_per_int: i32 = if (bits < 16) @divTrunc(8 * @as(i32, @sizeOf(u32)), bits) else 1;
        return .{
            .allocator = allocator,
            .keys = null,
            .values = null,
            .raw_keys = null,
            .raw_values = null,
            .offset = 0,
            .step = 256,
            .group_size = group_size,
            .bits = bits,
            .kv_bits = kv_bits,
            .batch_size = config.batch_size,
            .num_kv_heads = config.num_kv_heads,
            .head_dim = config.head_dim,
            .max_seq_len = config.max_seq_len,
            .el_per_int = el_per_int,
        };
    }

    pub fn asStrategy(self: *QuantizedKVCache) KVCacheStrategy {
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
        const self: *QuantizedKVCache = @ptrCast(@alignCast(ctx));
        const seq_len = @as(usize, @intCast(keys.shape()[2]));
        const prev = self.offset;
        const new_offset = prev + seq_len;

        if (new_offset > self.max_seq_len) {
            return error.CacheOverflow;
        }

        if (self.kv_bits == 16) {
            return self.updateAndFetchPassthrough(keys, values, seq_len, prev, new_offset, stream);
        } else {
            return self.updateAndFetchQuantized(keys, values, seq_len, prev, new_offset, stream);
        }
    }

    /// 16-bit passthrough: store raw keys/values without quantization.
    fn updateAndFetchPassthrough(
        self: *QuantizedKVCache,
        keys: Array,
        values: Array,
        seq_len: usize,
        prev: usize,
        new_offset: usize,
        stream: c.c.mlx_stream,
    ) !KVSlice {
        _ = seq_len;
        // Ensure raw buffers exist.
        if (self.raw_keys == null) {
            const shape = &[_]i32{
                @intCast(self.batch_size),
                @intCast(self.num_kv_heads),
                @intCast(self.max_seq_len),
                @intCast(self.head_dim),
            };
            var k_arr = c.c.mlx_array_new();
            var v_arr = c.c.mlx_array_new();
            try c.check(c.c.mlx_zeros(&k_arr, shape.ptr, shape.len, c.c.MLX_FLOAT16, stream));
            try c.check(c.c.mlx_zeros(&v_arr, shape.ptr, shape.len, c.c.MLX_FLOAT16, stream));
            self.raw_keys = Array.fromHandle(k_arr);
            self.raw_values = Array.fromHandle(v_arr);
        }

        // Write new keys/values at [prev:new_offset].
        try sliceUpdateKV(&self.raw_keys.?, keys, prev, new_offset, stream);
        try sliceUpdateKV(&self.raw_values.?, values, prev, new_offset, stream);

        self.offset = new_offset;

        // Return sliced view [0:offset].
        const fetched_keys = try sliceFetch(self.raw_keys.?, self.offset, stream);
        const fetched_values = try sliceFetch(self.raw_values.?, self.offset, stream);

        return .{ .keys = fetched_keys, .values = fetched_values };
    }

    /// Quantized path: quantize before storage, dequantize before attention.
    fn updateAndFetchQuantized(
        self: *QuantizedKVCache,
        keys: Array,
        values: Array,
        seq_len: usize,
        prev: usize,
        new_offset: usize,
        stream: c.c.mlx_stream,
    ) !KVSlice {
        _ = seq_len;
        // Expand buffer if needed.
        try ensureCapacity(self, new_offset, keys, values, stream);

        // Quantize incoming keys/values.
        const q_keys = try quantizeArray(keys, self.group_size, self.bits, stream);
        const q_values = try quantizeArray(values, self.group_size, self.bits, stream);
        defer {
            q_keys.packed_data.deinit();
            q_keys.scales.deinit();
            q_keys.biases.deinit();
            q_values.packed_data.deinit();
            q_values.scales.deinit();
            q_values.biases.deinit();
        }

        // Write quantized data into cache at [prev:new_offset].
        try writeQuantizedSlice(&self.keys.?, q_keys, prev, new_offset, stream);
        try writeQuantizedSlice(&self.values.?, q_values, prev, new_offset, stream);

        self.offset = new_offset;

        // Dequantize and return full cache [0:offset].
        const d_keys = try dequantizeTuple(self.keys.?, self.offset, self.group_size, self.bits, stream);
        const d_values = try dequantizeTuple(self.values.?, self.offset, self.group_size, self.bits, stream);

        return .{ .keys = d_keys, .values = d_values };
    }

    fn currentLenImpl(ctx: *anyopaque) usize {
        const self: *QuantizedKVCache = @ptrCast(@alignCast(ctx));
        return self.offset;
    }

    fn resetImpl(ctx: *anyopaque) void {
        const self: *QuantizedKVCache = @ptrCast(@alignCast(ctx));
        self.offset = 0;
    }

    fn rollbackImpl(ctx: *anyopaque, to_len: usize) void {
        const self: *QuantizedKVCache = @ptrCast(@alignCast(ctx));
        self.offset = to_len;
    }

    fn filterImpl(
        ctx: *anyopaque,
        indices: []const usize,
        allocator: std.mem.Allocator,
    ) !void {
        const self: *QuantizedKVCache = @ptrCast(@alignCast(ctx));
        _ = allocator;

        if (indices.len == 0) {
            self.offset = 0;
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

        if (self.kv_bits == 16) {
            // Filter raw buffers.
            if (self.raw_keys) |rk| {
                var new_k = c.c.mlx_array_new();
                var new_v = c.c.mlx_array_new();
                const stream = c.c.mlx_default_cpu_stream_new();
                defer _ = c.c.mlx_stream_free(stream);
                try c.check(c.c.mlx_take_axis(&new_k, rk.inner, idx_arr, 0, stream));
                try c.check(c.c.mlx_take_axis(&new_v, self.raw_values.?.inner, idx_arr, 0, stream));
                self.raw_keys.?.deinit();
                self.raw_values.?.deinit();
                self.raw_keys = Array.fromHandle(new_k);
                self.raw_values = Array.fromHandle(new_v);
            }
        } else {
            // Filter quantized tuples.
            if (self.keys != null) {
                self.keys = try filterTuple(self.keys.?, idx_arr);
                self.values = try filterTuple(self.values.?, idx_arr);
            }
        }
        self.batch_size = indices.len;
    }

    fn deinitImpl(ctx: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *QuantizedKVCache = @ptrCast(@alignCast(ctx));
        if (self.keys) |k| {
            k.packed_data.deinit();
            k.scales.deinit();
            k.biases.deinit();
        }
        if (self.values) |v| {
            v.packed_data.deinit();
            v.scales.deinit();
            v.biases.deinit();
        }
        if (self.raw_keys) |rk| rk.deinit();
        if (self.raw_values) |rv| rv.deinit();
        allocator.destroy(self);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn ensureCapacity(
        self: *QuantizedKVCache,
        needed: usize,
        keys: Array,
        values: Array,
        stream: c.c.mlx_stream,
    ) !void {
        const k_dim = keys.shape()[3];
        const v_dim = values.shape()[3];

        if (self.keys == null or needed > @as(usize, @intCast(self.keys.?.packed_data.shape()[2]))) {
            const new_steps = ((needed + self.step - 1) / self.step) * self.step;
            const shape = &[_]i32{
                @intCast(self.batch_size),
                @intCast(self.num_kv_heads),
                @intCast(new_steps),
            };

            if (self.keys == null) {
                self.keys = try initQuantizedBuffer(shape, k_dim, self.el_per_int, self.group_size, stream);
                self.values = try initQuantizedBuffer(shape, v_dim, self.el_per_int, self.group_size, stream);
            } else {
                self.keys = try expandQuantizedBuffer(self.keys.?, shape, self.el_per_int, self.group_size, stream);
                self.values = try expandQuantizedBuffer(self.values.?, shape, self.el_per_int, self.group_size, stream);
            }
        }
    }

    fn initQuantizedBuffer(
        shape: []const i32,
        dim: i32,
        el_per_int: i32,
        group_size: i32,
        stream: c.c.mlx_stream,
    ) !QuantizedTuple {
        const packed_shape = &[_]i32{ shape[0], shape[1], shape[2], @divTrunc(dim, el_per_int) };
        const scale_shape = &[_]i32{ shape[0], shape[1], shape[2], @divTrunc(dim, group_size) };

        var packed_arr = c.c.mlx_array_new();
        var scales = c.c.mlx_array_new();
        var biases = c.c.mlx_array_new();

        try c.check(c.c.mlx_zeros(&packed_arr, packed_shape.ptr, 4, c.c.MLX_UINT32, stream));
        try c.check(c.c.mlx_zeros(&scales, scale_shape.ptr, 4, c.c.MLX_FLOAT16, stream));
        try c.check(c.c.mlx_zeros(&biases, scale_shape.ptr, 4, c.c.MLX_FLOAT16, stream));

        return .{
            .packed_data = Array.fromHandle(packed_arr),
            .scales = Array.fromHandle(scales),
            .biases = Array.fromHandle(biases),
        };
    }

    fn expandQuantizedBuffer(
        tuple: QuantizedTuple,
        new_shape: []const i32,
        el_per_int: i32,
        group_size: i32,
        stream: c.c.mlx_stream,
    ) !QuantizedTuple {
        // Free old tuple and create new larger buffers.
        tuple.packed_data.deinit();
        tuple.scales.deinit();
        tuple.biases.deinit();
        const dim = new_shape[2] * el_per_int;
        return initQuantizedBuffer(new_shape, dim, el_per_int, group_size, stream);
    }

    fn quantizeArray(
        arr: Array,
        group_size: i32,
        bits: i32,
        stream: c.c.mlx_stream,
    ) !QuantizedTuple {
        var vec = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(vec);

        const opt_group: c.c.mlx_optional_int = .{ .value = group_size, .has_value = true };
        const opt_bits: c.c.mlx_optional_int = .{ .value = bits, .has_value = true };

        try c.check(c.c.mlx_quantize(
            &vec,
            arr.inner,
            opt_group,
            opt_bits,
            "affine",
            .{ .ctx = null }, // no global_scale
            stream,
        ));

        // mlx_quantize returns a vector of 3 arrays: [packed, scales, biases]
        var packed_arr = c.c.mlx_array_new();
        var scales_arr = c.c.mlx_array_new();
        var biases_arr = c.c.mlx_array_new();

        try c.check(c.c.mlx_vector_array_get(&packed_arr, vec, 0));
        try c.check(c.c.mlx_vector_array_get(&scales_arr, vec, 1));
        try c.check(c.c.mlx_vector_array_get(&biases_arr, vec, 2));

        return .{
            .packed_data = Array.fromHandle(packed_arr),
            .scales = Array.fromHandle(scales_arr),
            .biases = Array.fromHandle(biases_arr),
        };
    }

    fn writeQuantizedSlice(
        dst: *QuantizedTuple,
        src: QuantizedTuple,
        start: usize,
        end: usize,
        stream: c.c.mlx_stream,
    ) !void {
        const s = &[_]i32{ 0, 0, @intCast(start), 0 };
        const e = &[_]i32{ std.math.maxInt(i32), std.math.maxInt(i32), @intCast(end), std.math.maxInt(i32) };
        const st = &[_]i32{ 1, 1, 1, 1 };

        var new_packed = c.c.mlx_array_new();
        var new_scales = c.c.mlx_array_new();
        var new_biases = c.c.mlx_array_new();

        try c.check(c.c.mlx_slice_update(&new_packed, dst.packed_data.inner, src.packed_data.inner, s, 4, e, 4, st, 4, stream));
        try c.check(c.c.mlx_slice_update(&new_scales, dst.scales.inner, src.scales.inner, s, 4, e, 4, st, 4, stream));
        try c.check(c.c.mlx_slice_update(&new_biases, dst.biases.inner, src.biases.inner, s, 4, e, 4, st, 4, stream));

        dst.packed_data.deinit();
        dst.scales.deinit();
        dst.biases.deinit();
        dst.packed_data = Array.fromHandle(new_packed);
        dst.scales = Array.fromHandle(new_scales);
        dst.biases = Array.fromHandle(new_biases);
    }

    fn dequantizeTuple(tuple: QuantizedTuple, seq_len: usize, group_size: i32, bits: i32, stream: c.c.mlx_stream) !Array {
        // Slice to [0:seq_len] then dequantize.
        const start = &[_]i32{ 0, 0, 0, 0 };
        const stop = &[_]i32{ std.math.maxInt(i32), std.math.maxInt(i32), @intCast(seq_len), std.math.maxInt(i32) };
        const strides = &[_]i32{ 1, 1, 1, 1 };

        var packed_sliced = c.c.mlx_array_new();
        var scales_sliced = c.c.mlx_array_new();
        var biases_sliced = c.c.mlx_array_new();

        try c.check(c.c.mlx_slice(&packed_sliced, tuple.packed_data.inner, start, 4, stop, 4, strides, 4, stream));
        try c.check(c.c.mlx_slice(&scales_sliced, tuple.scales.inner, start, 4, stop, 4, strides, 4, stream));
        try c.check(c.c.mlx_slice(&biases_sliced, tuple.biases.inner, start, 4, stop, 4, strides, 4, stream));

        const opt_group: c.c.mlx_optional_int = .{ .value = group_size, .has_value = true };
        const opt_bits: c.c.mlx_optional_int = .{ .value = bits, .has_value = true };
        const no_dtype: c.c.mlx_optional_dtype = .{ .value = c.c.MLX_FLOAT16, .has_value = false };

        var res = c.c.mlx_array_new();
        try c.check(c.c.mlx_dequantize(
            &res,
            packed_sliced,
            scales_sliced,
            biases_sliced,
            opt_group,
            opt_bits,
            "affine",
            .{ .ctx = null }, // no global_scale
            no_dtype,
            stream,
        ));

        _ = c.c.mlx_array_free(packed_sliced);
        _ = c.c.mlx_array_free(scales_sliced);
        _ = c.c.mlx_array_free(biases_sliced);

        return Array.fromHandle(res);
    }

    fn filterTuple(tuple: QuantizedTuple, idx_arr: c.c.mlx_array) !QuantizedTuple {
        var new_packed = c.c.mlx_array_new();
        var new_scales = c.c.mlx_array_new();
        var new_biases = c.c.mlx_array_new();

        const stream = c.c.mlx_default_cpu_stream_new();
        defer _ = c.c.mlx_stream_free(stream);
        try c.check(c.c.mlx_take_axis(&new_packed, tuple.packed_data.inner, idx_arr, 0, stream));
        try c.check(c.c.mlx_take_axis(&new_scales, tuple.scales.inner, idx_arr, 0, stream));
        try c.check(c.c.mlx_take_axis(&new_biases, tuple.biases.inner, idx_arr, 0, stream));

        tuple.packed_data.deinit();
        tuple.scales.deinit();
        tuple.biases.deinit();

        return .{
            .packed_data = Array.fromHandle(new_packed),
            .scales = Array.fromHandle(new_scales),
            .biases = Array.fromHandle(new_biases),
        };
    }

    /// In-place update: buffer[..., offset:end_offset, :] = new_kv
    fn sliceUpdateKV(
        buffer: *Array,
        new_kv: Array,
        offset: usize,
        end_offset: usize,
        stream: c.c.mlx_stream,
    ) !void {
        const s = &[_]i32{ 0, 0, @intCast(offset), 0 };
        const e = &[_]i32{ std.math.maxInt(i32), std.math.maxInt(i32), @intCast(end_offset), std.math.maxInt(i32) };
        const st = &[_]i32{ 1, 1, 1, 1 };

        var res = c.c.mlx_array_new();
        try c.check(c.c.mlx_slice_update(
            &res,
            buffer.inner,
            new_kv.inner,
            s,
            4,
            e,
            4,
            st,
            4,
            stream,
        ));

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
            start,
            4,
            stop,
            4,
            strides,
            4,
            stream,
        ));
        return Array.fromHandle(res);
    }
};

// ------------------------------------------------------------------
// Factory functions
// ------------------------------------------------------------------

/// Create a quantized KV cache with custom kv_bits and group_size.
pub fn createQuantized(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    kv_bits: u8,
    group_size: i32,
    stream: c.c.mlx_stream,
) !KVCacheStrategy {
    const cache = try allocator.create(QuantizedKVCache);
    errdefer allocator.destroy(cache);
    cache.* = try QuantizedKVCache.init(allocator, config, group_size, @intCast(kv_bits), stream);
    return cache.asStrategy();
}

/// Factory for 8-bit quantized KV cache (group_size=64).
pub fn createQuantized8Bit(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    stream: c.c.mlx_stream,
) !KVCacheStrategy {
    return createQuantized(allocator, config, 8, 64, stream);
}

/// Factory for 4-bit quantized KV cache (group_size=64).
pub fn createQuantized4Bit(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    stream: c.c.mlx_stream,
) !KVCacheStrategy {
    return createQuantized(allocator, config, 4, 64, stream);
}

/// Factory for 16-bit passthrough KV cache (no quantization).
pub fn createQuantized16Bit(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    stream: c.c.mlx_stream,
) !KVCacheStrategy {
    return createQuantized(allocator, config, 16, 64, stream);
}
