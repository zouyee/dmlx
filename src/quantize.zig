/// Weight quantization infrastructure for model weights.
///
/// Separate from `kvcache/quantized.zig` which handles KV cache quantization.
/// This module provides:
///   - `quantize()` / `dequantize()`: affine and MXFP4 quantization via `mlx_quantize`
///   - `quantizedMatmul()`: fused dequantize + matmul kernel
///   - `qqmm()`: double-quantized matmul (both operands quantized)
///   - `gatherQmm()`: quantized gather matmul for batched/indexed inference
///   - `toFp8()` / `fromFp8()`: FP8 (E4M3) conversion
///   - `QuantizedWeight`: struct holding packed data, scales, biases, config
///   - GPTQ weight loading support with on-the-fly dequantization
///
/// Quantization modes (passed as `mode` string to mlx-c):
///   - `"affine"` (default): uniform affine quantization with per-group scale+bias
///   - `"mxfp4"`: Microscaling FP4 (E2M1), group_size=32, 8-bit shared scale per block
///
/// Requirements: R18.1, R18.2, R18.3
const std = @import("std");
const c = @import("c.zig");
const array_mod = @import("array.zig");
const ops = @import("ops.zig");
const Dtype = @import("dtype.zig").Dtype;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

/// Quantization mode.
pub const QuantMode = enum {
    /// Uniform affine quantization with per-group scale and bias.
    affine,
    /// Microscaling FP4 (E2M1): 4-bit float, group_size=32, 8-bit shared scale.
    mxfp4,
    /// NVIDIA FP4 (E2M1): 4-bit float, group_size=16, used by DeepSeek V4 expert weights.
    nvfp4,
    /// Microscaling FP8 (E4M3/E5M2): 8-bit float, group_size=32, 8-bit shared scale.
    mxfp8,

    /// Return the C string expected by mlx-c.
    pub fn toCStr(self: QuantMode) [*:0]const u8 {
        return switch (self) {
            .affine => "affine",
            .mxfp4 => "mxfp4",
            .nvfp4 => "nvfp4",
            .mxfp8 => "mxfp8",
        };
    }
};

/// Configuration for weight quantization.
pub const QuantConfig = struct {
    bits: u8 = 4, // 4 or 8
    group_size: i32 = 64,
    mode: QuantMode = .affine,

    pub fn validate(self: QuantConfig) !void {
        switch (self.mode) {
            .affine => {
                switch (self.bits) {
                    4, 8 => {},
                    else => return error.InvalidQuantBits,
                }
                if (self.group_size <= 0) return error.InvalidGroupSize;
            },
            .mxfp4 => {
                if (self.bits != 4) return error.InvalidQuantBits;
                if (self.group_size != 32) return error.InvalidGroupSize;
            },
            .nvfp4 => {
                if (self.bits != 4) return error.InvalidQuantBits;
                if (self.group_size != 16) return error.InvalidGroupSize;
            },
            .mxfp8 => {
                if (self.bits != 8) return error.InvalidQuantBits;
                if (self.group_size != 32) return error.InvalidGroupSize;
            },
        }
    }

    /// Convenience constructor for MXFP4.
    pub fn mxfp4() QuantConfig {
        return .{ .bits = 4, .group_size = 32, .mode = .mxfp4 };
    }

    /// Convenience constructor for NVIDIA FP4 (DeepSeek V4 expert weights).
    pub fn nvfp4() QuantConfig {
        return .{ .bits = 4, .group_size = 16, .mode = .nvfp4 };
    }

    /// Convenience constructor for MXFP8.
    pub fn mxfp8() QuantConfig {
        return .{ .bits = 8, .group_size = 32, .mode = .mxfp8 };
    }
};

/// A quantized model weight storing packed data, scales, biases, and metadata.
pub const QuantizedWeight = struct {
    data: Array,
    scales: Array,
    biases: Array,
    config: QuantConfig,
    original_shape: []const i32,

    /// Free all owned arrays and the shape allocation.
    pub fn deinit(self: QuantizedWeight, allocator: std.mem.Allocator) void {
        self.data.deinit();
        self.scales.deinit();
        self.biases.deinit();
        allocator.free(self.original_shape);
    }
};

/// Quantize a weight tensor to the specified bit-width and mode.
///
/// Uses `mlx_quantize` with the configured quantization scheme.
/// The input weight should be a 2D tensor [out_features, in_features].
/// Returns a `QuantizedWeight` with packed data, per-group scales and biases.
///
/// For MXFP4 mode, biases may be empty (MXFP4 uses global_scale instead of per-group bias).
///
/// Requirements: R18.1
pub fn quantize(ctx: EagerContext, weight: Array, config: QuantConfig) !QuantizedWeight {
    try config.validate();

    const allocator = ctx.allocator;
    const stream = ctx.stream.inner;

    // Save original shape before quantization.
    const orig_shape = weight.shape();
    const shape_copy = try allocator.alloc(i32, orig_shape.len);
    @memcpy(shape_copy, orig_shape);

    // Call mlx_quantize: returns vector of [packed_data, scales, biases].
    var vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(vec);

    const opt_group: c.c.mlx_optional_int = .{ .value = config.group_size, .has_value = true };
    const opt_bits: c.c.mlx_optional_int = .{ .value = @as(i32, @intCast(config.bits)), .has_value = true };
    const null_array: c.c.mlx_array = .{ .ctx = null };

    try c.check(c.c.mlx_quantize(
        &vec,
        weight.inner,
        opt_group,
        opt_bits,
        config.mode.toCStr(),
        null_array,
        stream,
    ));

    const vec_size = c.c.mlx_vector_array_size(vec);

    var packed_arr = c.c.mlx_array_new();
    var scales_arr = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&packed_arr, vec, 0));
    try c.check(c.c.mlx_vector_array_get(&scales_arr, vec, 1));

    // MXFP4 may return only 2 elements (no biases); affine returns 3.
    var biases_arr = c.c.mlx_array_new();
    if (vec_size >= 3) {
        try c.check(c.c.mlx_vector_array_get(&biases_arr, vec, 2));
    }

    return .{
        .data = Array.fromHandle(packed_arr),
        .scales = Array.fromHandle(scales_arr),
        .biases = Array.fromHandle(biases_arr),
        .config = config,
        .original_shape = shape_copy,
    };
}

/// Dequantize a quantized weight back to full precision.
///
/// Uses `mlx_dequantize` with the configured mode to restore the tensor.
/// The output dtype defaults to float16 (not specified → mlx chooses).
///
/// Requirements: R18.2
pub fn dequantize(ctx: EagerContext, qw: QuantizedWeight) !Array {
    const stream = ctx.stream.inner;

    const opt_group: c.c.mlx_optional_int = .{ .value = qw.config.group_size, .has_value = true };
    const opt_bits: c.c.mlx_optional_int = .{ .value = @as(i32, @intCast(qw.config.bits)), .has_value = true };
    const no_dtype: c.c.mlx_optional_dtype = .{ .value = c.c.MLX_FLOAT16, .has_value = false };
    const null_array: c.c.mlx_array = .{ .ctx = null };

    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_dequantize(
        &res,
        qw.data.inner,
        qw.scales.inner,
        qw.biases.inner,
        opt_group,
        opt_bits,
        qw.config.mode.toCStr(),
        null_array,
        no_dtype,
        stream,
    ));

    return Array.fromHandle(res);
}

/// Dequantize to a specific output dtype.
pub fn dequantizeAs(ctx: EagerContext, qw: QuantizedWeight, dt: Dtype) !Array {
    const stream = ctx.stream.inner;

    const opt_group: c.c.mlx_optional_int = .{ .value = qw.config.group_size, .has_value = true };
    const opt_bits: c.c.mlx_optional_int = .{ .value = @as(i32, @intCast(qw.config.bits)), .has_value = true };
    const opt_dtype: c.c.mlx_optional_dtype = .{ .value = @intCast(@intFromEnum(dt)), .has_value = true };
    const null_array: c.c.mlx_array = .{ .ctx = null };

    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_dequantize(
        &res,
        qw.data.inner,
        qw.scales.inner,
        qw.biases.inner,
        opt_group,
        opt_bits,
        qw.config.mode.toCStr(),
        null_array,
        opt_dtype,
        stream,
    ));

    return Array.fromHandle(res);
}

/// Load a pre-quantized weight from GPTQ-format components.
///
/// GPTQ stores weights as (qweight, scales, qzeros) which map to
/// MLX's (packed_data, scales, biases) representation.
/// This creates a `QuantizedWeight` from pre-quantized arrays without
/// re-quantizing, enabling on-the-fly dequantization in forward passes.
///
/// Requirements: R18.3
pub fn loadPreQuantized(
    allocator: std.mem.Allocator,
    packed_data: Array,
    scales: Array,
    biases: Array,
    config: QuantConfig,
    original_shape: []const i32,
) !QuantizedWeight {
    try config.validate();

    const shape_copy = try allocator.alloc(i32, original_shape.len);
    @memcpy(shape_copy, original_shape);

    return .{
        .data = packed_data,
        .scales = scales,
        .biases = biases,
        .config = config,
        .original_shape = shape_copy,
    };
}

/// Fused quantized matrix multiplication: x @ dequantize(qw).
///
/// Uses `mlx_quantized_matmul` which dequantizes and multiplies in a single
/// fused kernel — faster and more memory-efficient than separate dequantize+matmul.
///
/// Requirements: R18.3
pub fn quantizedMatmul(ctx: EagerContext, x: Array, qw: QuantizedWeight, transpose_w: bool) !Array {
    const stream = ctx.stream.inner;

    const opt_group: c.c.mlx_optional_int = .{ .value = qw.config.group_size, .has_value = true };
    const opt_bits: c.c.mlx_optional_int = .{ .value = @as(i32, @intCast(qw.config.bits)), .has_value = true };

    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_quantized_matmul(
        &res,
        x.inner,
        qw.data.inner,
        qw.scales.inner,
        qw.biases.inner,
        transpose_w,
        opt_group,
        opt_bits,
        qw.config.mode.toCStr(),
        stream,
    ));
    return Array.fromHandle(res);
}

/// Double-quantized matrix multiplication (both operands quantized).
///
/// Uses `mlx_qqmm` for cases where both x and w are quantized (e.g., FP4 activations × FP4 weights).
pub fn qqmm(
    ctx: EagerContext,
    x: Array,
    w: Array,
    w_scales: ?Array,
    config: QuantConfig,
    global_scale_x: ?Array,
    global_scale_w: ?Array,
) !Array {
    const stream = ctx.stream.inner;

    const opt_group: c.c.mlx_optional_int = .{ .value = config.group_size, .has_value = true };
    const opt_bits: c.c.mlx_optional_int = .{ .value = @as(i32, @intCast(config.bits)), .has_value = true };
    const null_array: c.c.mlx_array = .{ .ctx = null };

    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_qqmm(
        &res,
        x.inner,
        w.inner,
        if (w_scales) |s| s.inner else null_array,
        opt_group,
        opt_bits,
        config.mode.toCStr(),
        if (global_scale_x) |s| s.inner else null_array,
        if (global_scale_w) |s| s.inner else null_array,
        stream,
    ));
    return Array.fromHandle(res);
}

/// Quantized gather matrix multiplication for batched/indexed inference.
///
/// Uses `mlx_gather_qmm` for efficient batched quantized matmul with index selection.
pub fn gatherQmm(
    ctx: EagerContext,
    x: Array,
    w: Array,
    scales: Array,
    biases: ?Array,
    lhs_indices: ?Array,
    rhs_indices: ?Array,
    transpose_w: bool,
    config: QuantConfig,
    sorted_indices: bool,
) !Array {
    const stream = ctx.stream.inner;

    const opt_group: c.c.mlx_optional_int = .{ .value = config.group_size, .has_value = true };
    const opt_bits: c.c.mlx_optional_int = .{ .value = @as(i32, @intCast(config.bits)), .has_value = true };
    const null_array: c.c.mlx_array = .{ .ctx = null };

    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_gather_qmm(
        &res,
        x.inner,
        w.inner,
        scales.inner,
        if (biases) |b| b.inner else null_array,
        if (lhs_indices) |i| i.inner else null_array,
        if (rhs_indices) |i| i.inner else null_array,
        transpose_w,
        opt_group,
        opt_bits,
        config.mode.toCStr(),
        sorted_indices,
        stream,
    ));
    return Array.fromHandle(res);
}

// === FP8 Conversion ===

/// Convert an array to FP8 (E4M3) format.
///
/// FP8 is not a first-class mlx_dtype — it's accessed through dedicated conversion functions.
/// The result is stored internally as FP8 but appears as uint8 in the dtype enum.
pub fn toFp8(ctx: EagerContext, a: Array) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_to_fp8(&res, a.inner, ctx.stream.inner));
    return Array.fromHandle(res);
}

/// Convert an FP8 array back to the specified dtype (e.g., float16, bfloat16).
pub fn fromFp8(ctx: EagerContext, a: Array, dt: Dtype) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_from_fp8(&res, a.inner, @intCast(@intFromEnum(dt)), ctx.stream.inner));
    return Array.fromHandle(res);
}

/// Perform a dequantized matrix multiplication: x @ dequantize(qw).
///
/// This is the fallback path that dequantizes first, then multiplies.
/// Prefer `quantizedMatmul` which uses a fused kernel.
///
/// Requirements: R18.3
pub fn dequantizedMatmul(ctx: EagerContext, x: Array, qw: QuantizedWeight) !Array {
    const deq = try dequantize(ctx, qw);
    defer deq.deinit();

    // Transpose weight for matmul: x [batch, in] @ W^T [in, out] = [batch, out]
    var transposed = c.c.mlx_array_new();
    try c.check(c.c.mlx_transpose(&transposed, deq.inner, ctx.stream.inner));
    defer _ = c.c.mlx_array_free(transposed);

    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_matmul(&res, x.inner, transposed, ctx.stream.inner));
    return Array.fromHandle(res);
}

// ============================================================
// Unit Tests
// ============================================================

test "QuantConfig: valid 4-bit config" {
    const cfg = QuantConfig{ .bits = 4, .group_size = 64 };
    try cfg.validate();
}

test "QuantConfig: valid 8-bit config" {
    const cfg = QuantConfig{ .bits = 8, .group_size = 32 };
    try cfg.validate();
}

test "QuantConfig: default values" {
    const cfg = QuantConfig{};
    try std.testing.expectEqual(@as(u8, 4), cfg.bits);
    try std.testing.expectEqual(@as(i32, 64), cfg.group_size);
    try std.testing.expectEqual(QuantMode.affine, cfg.mode);
    try cfg.validate();
}

test "QuantConfig: invalid bits rejected" {
    const cfg = QuantConfig{ .bits = 3, .group_size = 64 };
    try std.testing.expectError(error.InvalidQuantBits, cfg.validate());
}

test "QuantConfig: invalid group_size rejected" {
    const cfg = QuantConfig{ .bits = 4, .group_size = 0 };
    try std.testing.expectError(error.InvalidGroupSize, cfg.validate());
}

test "QuantConfig: 16-bit rejected for weight quantization" {
    const cfg = QuantConfig{ .bits = 16, .group_size = 64 };
    try std.testing.expectError(error.InvalidQuantBits, cfg.validate());
}

test "QuantConfig: mxfp4 convenience constructor" {
    const cfg = QuantConfig.mxfp4();
    try std.testing.expectEqual(@as(u8, 4), cfg.bits);
    try std.testing.expectEqual(@as(i32, 32), cfg.group_size);
    try std.testing.expectEqual(QuantMode.mxfp4, cfg.mode);
    try cfg.validate();
}

test "QuantConfig: mxfp4 rejects wrong group_size" {
    const cfg = QuantConfig{ .bits = 4, .group_size = 64, .mode = .mxfp4 };
    try std.testing.expectError(error.InvalidGroupSize, cfg.validate());
}

test "QuantConfig: mxfp4 rejects wrong bits" {
    const cfg = QuantConfig{ .bits = 8, .group_size = 32, .mode = .mxfp4 };
    try std.testing.expectError(error.InvalidQuantBits, cfg.validate());
}

test "QuantMode: toCStr returns correct strings" {
    try std.testing.expectEqualStrings("affine", std.mem.span(QuantMode.affine.toCStr()));
    try std.testing.expectEqualStrings("mxfp4", std.mem.span(QuantMode.mxfp4.toCStr()));
    try std.testing.expectEqualStrings("nvfp4", std.mem.span(QuantMode.nvfp4.toCStr()));
    try std.testing.expectEqualStrings("mxfp8", std.mem.span(QuantMode.mxfp8.toCStr()));
}

test "QuantConfig: nvfp4 convenience constructor" {
    const cfg = QuantConfig.nvfp4();
    try std.testing.expectEqual(@as(u8, 4), cfg.bits);
    try std.testing.expectEqual(@as(i32, 16), cfg.group_size);
    try std.testing.expectEqual(QuantMode.nvfp4, cfg.mode);
    try cfg.validate();
}

test "QuantConfig: mxfp8 convenience constructor" {
    const cfg = QuantConfig.mxfp8();
    try std.testing.expectEqual(@as(u8, 8), cfg.bits);
    try std.testing.expectEqual(@as(i32, 32), cfg.group_size);
    try std.testing.expectEqual(QuantMode.mxfp8, cfg.mode);
    try cfg.validate();
}

test "QuantConfig: nvfp4 rejects wrong group_size" {
    const cfg = QuantConfig{ .bits = 4, .group_size = 32, .mode = .nvfp4 };
    try std.testing.expectError(error.InvalidGroupSize, cfg.validate());
}

test "QuantConfig: mxfp8 rejects wrong bits" {
    const cfg = QuantConfig{ .bits = 4, .group_size = 32, .mode = .mxfp8 };
    try std.testing.expectError(error.InvalidQuantBits, cfg.validate());
}

test "quantize and dequantize round-trip — 8-bit" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Create a 2D weight tensor [64, 64] with random float16 data.
    const stream = ctx.stream.inner;
    const shape = &[_]i32{ 64, 64 };
    var weight_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(
        &weight_raw,
        shape.ptr,
        shape.len,
        c.c.MLX_FLOAT16,
        0.0,
        1.0,
        .{ .ctx = null },
        stream,
    ));
    const weight = Array.fromHandle(weight_raw);
    defer weight.deinit();

    const config = QuantConfig{ .bits = 8, .group_size = 64 };
    const qw = try quantize(ctx, weight, config);
    defer qw.deinit(allocator);

    // Verify quantized weight has expected config.
    try std.testing.expectEqual(@as(u8, 8), qw.config.bits);
    try std.testing.expectEqual(@as(i32, 64), qw.config.group_size);
    try std.testing.expectEqual(@as(usize, 2), qw.original_shape.len);
    try std.testing.expectEqual(@as(i32, 64), qw.original_shape[0]);
    try std.testing.expectEqual(@as(i32, 64), qw.original_shape[1]);

    // Dequantize and check shape matches original.
    const restored = try dequantize(ctx, qw);
    defer restored.deinit();

    const restored_shape = restored.shape();
    try std.testing.expectEqual(@as(usize, 2), restored_shape.len);
    try std.testing.expectEqual(@as(i32, 64), restored_shape[0]);
    try std.testing.expectEqual(@as(i32, 64), restored_shape[1]);
}

test "quantize and dequantize round-trip — 4-bit" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const stream = ctx.stream.inner;
    const shape = &[_]i32{ 128, 64 };
    var weight_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(
        &weight_raw,
        shape.ptr,
        shape.len,
        c.c.MLX_FLOAT16,
        0.0,
        1.0,
        .{ .ctx = null },
        stream,
    ));
    const weight = Array.fromHandle(weight_raw);
    defer weight.deinit();

    const config = QuantConfig{ .bits = 4, .group_size = 64 };
    const qw = try quantize(ctx, weight, config);
    defer qw.deinit(allocator);

    try std.testing.expectEqual(@as(u8, 4), qw.config.bits);
    try std.testing.expectEqual(@as(i32, 128), qw.original_shape[0]);
    try std.testing.expectEqual(@as(i32, 64), qw.original_shape[1]);

    const restored = try dequantize(ctx, qw);
    defer restored.deinit();

    const restored_shape = restored.shape();
    try std.testing.expectEqual(@as(i32, 128), restored_shape[0]);
    try std.testing.expectEqual(@as(i32, 64), restored_shape[1]);
}

test "loadPreQuantized creates valid QuantizedWeight" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();

    // Create mock pre-quantized components.
    const packed_shape = &[_]i32{ 64, 8 }; // 64 rows, 64/8=8 packed cols for 4-bit
    const scale_shape = &[_]i32{ 64, 1 }; // 64 rows, 64/64=1 group

    var packed_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_zeros(&packed_raw, packed_shape.ptr, packed_shape.len, c.c.MLX_UINT32, stream));
    var scales_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&scales_raw, scale_shape.ptr, scale_shape.len, c.c.MLX_FLOAT16, stream));
    var biases_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_zeros(&biases_raw, scale_shape.ptr, scale_shape.len, c.c.MLX_FLOAT16, stream));

    const orig_shape = &[_]i32{ 64, 64 };
    const config = QuantConfig{ .bits = 4, .group_size = 64 };

    const qw = try loadPreQuantized(
        allocator,
        Array.fromHandle(packed_raw),
        Array.fromHandle(scales_raw),
        Array.fromHandle(biases_raw),
        config,
        orig_shape,
    );
    defer qw.deinit(allocator);

    try std.testing.expectEqual(@as(u8, 4), qw.config.bits);
    try std.testing.expectEqual(@as(i32, 64), qw.config.group_size);
    try std.testing.expectEqual(@as(usize, 2), qw.original_shape.len);
    try std.testing.expectEqual(@as(i32, 64), qw.original_shape[0]);
}

test "dequantizedMatmul produces correct output shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = ctx.stream.inner;

    // Create weight [64, 64] and quantize it.
    const w_shape = &[_]i32{ 64, 64 };
    var weight_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(
        &weight_raw,
        w_shape.ptr,
        w_shape.len,
        c.c.MLX_FLOAT16,
        0.0,
        1.0,
        .{ .ctx = null },
        stream,
    ));
    const weight = Array.fromHandle(weight_raw);
    defer weight.deinit();

    const config = QuantConfig{ .bits = 8, .group_size = 64 };
    const qw = try quantize(ctx, weight, config);
    defer qw.deinit(allocator);

    // Create input [4, 64].
    const x_shape = &[_]i32{ 4, 64 };
    var x_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(
        &x_raw,
        x_shape.ptr,
        x_shape.len,
        c.c.MLX_FLOAT16,
        0.0,
        1.0,
        .{ .ctx = null },
        stream,
    ));
    const x = Array.fromHandle(x_raw);
    defer x.deinit();

    // dequantizedMatmul: x [4, 64] @ W^T [64, 64] = [4, 64]
    const result = try dequantizedMatmul(ctx, x, qw);
    defer result.deinit();

    const result_shape = result.shape();
    try std.testing.expectEqual(@as(usize, 2), result_shape.len);
    try std.testing.expectEqual(@as(i32, 4), result_shape[0]);
    try std.testing.expectEqual(@as(i32, 64), result_shape[1]);
}

test "toFp8 and fromFp8 round-trip" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = ctx.stream.inner;

    // Create a small float32 tensor.
    const shape = &[_]i32{ 4, 8 };
    var arr_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(
        &arr_raw,
        shape.ptr,
        shape.len,
        c.c.MLX_FLOAT32,
        0.0,
        1.0,
        .{ .ctx = null },
        stream,
    ));
    const arr = Array.fromHandle(arr_raw);
    defer arr.deinit();

    // Convert to FP8.
    const fp8 = try toFp8(ctx, arr);
    defer fp8.deinit();

    // Convert back to float32.
    const restored = try fromFp8(ctx, fp8, .float32);
    defer restored.deinit();

    const restored_shape = restored.shape();
    try std.testing.expectEqual(@as(usize, 2), restored_shape.len);
    try std.testing.expectEqual(@as(i32, 4), restored_shape[0]);
    try std.testing.expectEqual(@as(i32, 8), restored_shape[1]);
}

test "quantizedMatmul produces correct output shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = ctx.stream.inner;

    // Create weight [64, 64] and quantize it.
    const w_shape = &[_]i32{ 64, 64 };
    var weight_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(
        &weight_raw,
        w_shape.ptr,
        w_shape.len,
        c.c.MLX_FLOAT16,
        0.0,
        1.0,
        .{ .ctx = null },
        stream,
    ));
    const weight = Array.fromHandle(weight_raw);
    defer weight.deinit();

    const config = QuantConfig{ .bits = 8, .group_size = 64 };
    const qw = try quantize(ctx, weight, config);
    defer qw.deinit(allocator);

    // Create input [4, 64].
    const x_shape = &[_]i32{ 4, 64 };
    var x_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(
        &x_raw,
        x_shape.ptr,
        x_shape.len,
        c.c.MLX_FLOAT16,
        0.0,
        1.0,
        .{ .ctx = null },
        stream,
    ));
    const x = Array.fromHandle(x_raw);
    defer x.deinit();

    // Fused quantizedMatmul: x [4, 64] @ W^T [64, 64] = [4, 64]
    const result = try quantizedMatmul(ctx, x, qw, true);
    defer result.deinit();

    const result_shape = result.shape();
    try std.testing.expectEqual(@as(usize, 2), result_shape.len);
    try std.testing.expectEqual(@as(i32, 4), result_shape[0]);
    try std.testing.expectEqual(@as(i32, 64), result_shape[1]);
}
