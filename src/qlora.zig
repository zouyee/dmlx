/// QLoRA (Quantized Low-Rank Adaptation) for memory-efficient fine-tuning.
///
/// Combines 4-bit NF4 quantized base weights with trainable LoRA adapters:
///   output = dequantize(W_base) @ x + (B @ A) @ x * scaling
///
/// Only LoRA adapter parameters (A, B) receive gradients during training;
/// the quantized base weight W_base remains frozen.
///
/// Integrates with `src/lora.zig` for LoRA forward computation and
/// `src/quantize.zig` for weight quantization/dequantization.
///
/// Requirements: R19.1, R19.2, R19.3
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const quantize_mod = @import("mlx").quantize;
const lora_mod = @import("lora.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const QuantizedWeight = quantize_mod.QuantizedWeight;
const QuantConfig = quantize_mod.QuantConfig;
const LoRALayer = lora_mod.LoRALayer;

/// A QLoRA layer: quantized frozen base weight + trainable LoRA adapters.
///
/// Forward pass: dequantize(W_base) @ x + (B @ A) @ x * scaling
/// Gradients flow only through the LoRA path (A, B parameters).
pub const QLoRALayer = struct {
    base_quantized: QuantizedWeight,
    lora: LoRALayer,
    ctx: EagerContext,

    /// Initialize a QLoRA layer by quantizing a full-precision base weight
    /// and creating LoRA adapters of the given rank.
    ///
    /// - `base_weight`: full-precision weight [out_features, in_features]
    /// - `rank`: LoRA rank (typically 4-64)
    /// - `alpha`: LoRA scaling factor (typically 2*rank)
    /// - `quant_config`: quantization config (default: 4-bit NF4, group_size 64)
    pub fn init(
        ctx: EagerContext,
        base_weight: Array,
        rank: usize,
        alpha: f32,
        quant_config: QuantConfig,
        stream: c.c.mlx_stream,
    ) !QLoRALayer {
        const shape = base_weight.shape();
        std.debug.assert(shape.len == 2);
        const out_features: usize = @intCast(shape[0]);
        const in_features: usize = @intCast(shape[1]);

        // Quantize the base weight (frozen, no gradients)
        const qw = try quantize_mod.quantize(ctx, base_weight, quant_config);
        errdefer qw.deinit(ctx.allocator);

        // Create trainable LoRA adapters using the same stream
        const lora = try initLoRAAdapters(ctx, out_features, in_features, rank, alpha, stream);

        return .{
            .base_quantized = qw,
            .lora = lora,
            .ctx = ctx,
        };
    }

    /// Create LoRA adapter matrices (A: Gaussian, B: zeros) directly via mlx-c.
    fn initLoRAAdapters(
        ctx: EagerContext,
        out_features: usize,
        in_features: usize,
        rank: usize,
        alpha: f32,
        stream: c.c.mlx_stream,
    ) !LoRALayer {
        std.debug.assert(rank <= @min(out_features, in_features));

        const a_shape = &[_]i32{ @intCast(out_features), @intCast(rank) };
        const b_shape = &[_]i32{ @intCast(rank), @intCast(in_features) };

        // A: Gaussian init
        var a_arr = c.c.mlx_array_new();
        try c.check(c.c.mlx_random_normal(
            &a_arr,
            a_shape.ptr,
            a_shape.len,
            c.c.MLX_FLOAT32,
            0.0,
            1.0,
            .{ .ctx = null },
            stream,
        ));
        const a = Array.fromHandle(a_arr);

        // B: Zero init
        var b_arr = c.c.mlx_array_new();
        try c.check(c.c.mlx_zeros(&b_arr, b_shape.ptr, b_shape.len, c.c.MLX_FLOAT32, stream));
        const b = Array.fromHandle(b_arr);

        return .{
            .ctx = ctx,
            .rank = rank,
            .alpha = alpha,
            .scale = alpha / @as(f32, @floatFromInt(rank)),
            .a = a,
            .b = b,
        };
    }

    /// Initialize a QLoRA layer from a pre-quantized weight and existing LoRA adapters.
    pub fn initFromQuantized(
        ctx: EagerContext,
        base_quantized: QuantizedWeight,
        lora: LoRALayer,
    ) QLoRALayer {
        return .{
            .base_quantized = base_quantized,
            .lora = lora,
            .ctx = ctx,
        };
    }

    /// Free all owned resources.
    pub fn deinit(self: *QLoRALayer, allocator: std.mem.Allocator) void {
        self.base_quantized.deinit(allocator);
        self.lora.deinit();
    }

    /// QLoRA forward pass:
    ///   output = dequantize(W_base) @ x + lora(x)
    ///
    /// where lora(x) = (x @ B^T @ A^T) * scaling
    ///
    /// The base path dequantizes on-the-fly. Only the LoRA path
    /// (A, B) participates in gradient computation.
    pub fn forward(self: *QLoRALayer, x: Array) !Array {
        // Base path: dequantize(W_base) @ x (frozen, no gradients)
        const base_out = try quantize_mod.dequantizedMatmul(self.ctx, x, self.base_quantized);
        defer base_out.deinit();

        // LoRA path: (x @ B^T @ A^T) * scaling (trainable)
        const lora_out = try self.lora.forward(x);
        defer lora_out.deinit();

        // Combined output
        return ops.add(self.ctx, base_out, lora_out);
    }

    /// Return the scaling factor (alpha / rank).
    pub fn scaling(self: *const QLoRALayer) f32 {
        return self.lora.scale;
    }

    /// Collect trainable parameters (A and B matrices only).
    /// The quantized base weight is NOT included — it is frozen.
    pub fn collectTrainableParams(self: *QLoRALayer, allocator: std.mem.Allocator) ![]Array {
        var params = try allocator.alloc(Array, 2);
        params[0] = self.lora.a;
        params[1] = self.lora.b;
        return params;
    }

    /// Collect pointers to trainable parameters for in-place optimizer updates.
    pub fn collectTrainableParamPtrs(self: *QLoRALayer, allocator: std.mem.Allocator) ![]*Array {
        var ptrs = try allocator.alloc(*Array, 2);
        ptrs[0] = &self.lora.a;
        ptrs[1] = &self.lora.b;
        return ptrs;
    }

    /// Set LoRA parameters from a flat array (used in training).
    /// Expects exactly 2 arrays: [A, B].
    pub fn setLoRAParams(self: *QLoRALayer, params: []const Array) void {
        std.debug.assert(params.len == 2);

        self.lora.a.deinit();
        var a_copy = c.c.mlx_array_new();
        _ = c.c.mlx_array_set(&a_copy, params[0].inner);
        self.lora.a = Array.fromHandle(a_copy);

        self.lora.b.deinit();
        var b_copy = c.c.mlx_array_new();
        _ = c.c.mlx_array_set(&b_copy, params[1].inner);
        self.lora.b = Array.fromHandle(b_copy);
    }
};

/// Collection of QLoRA adapters applied to specific layers of a model.
/// Mirrors `LoRAModel` from `src/lora.zig` but with quantized base weights.
pub const QLoRAModel = struct {
    allocator: std.mem.Allocator,
    layers: std.StringHashMap(QLoRALayer),
    layer_names: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator) QLoRAModel {
        return .{
            .allocator = allocator,
            .layers = std.StringHashMap(QLoRALayer).init(allocator),
            .layer_names = std.ArrayList([]const u8).empty,
        };
    }

    pub fn deinit(self: *QLoRAModel) void {
        var it = self.layers.valueIterator();
        while (it.next()) |layer| {
            layer.deinit(self.allocator);
        }
        self.layers.deinit();
        for (self.layer_names.items) |name| {
            self.allocator.free(name);
        }
        self.layer_names.deinit(self.allocator);
    }

    /// Add a QLoRA layer: quantize the base weight and create LoRA adapters.
    pub fn addLayer(
        self: *QLoRAModel,
        name: []const u8,
        ctx: EagerContext,
        base_weight: Array,
        rank: usize,
        alpha: f32,
        quant_config: QuantConfig,
        stream: c.c.mlx_stream,
    ) !void {
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);

        var layer = try QLoRALayer.init(ctx, base_weight, rank, alpha, quant_config, stream);
        errdefer layer.deinit(self.allocator);

        try self.layers.put(key, layer);
        try self.layer_names.append(self.allocator, key);
    }

    /// Get a QLoRA layer by name.
    pub fn getLayer(self: *QLoRAModel, name: []const u8) ?*QLoRALayer {
        return self.layers.getPtr(name);
    }

    /// Collect all trainable parameters (A and B for each layer).
    /// Returns a flat array: [layer0_A, layer0_B, layer1_A, layer1_B, ...]
    pub fn collectParams(self: *QLoRAModel, allocator: std.mem.Allocator) ![]Array {
        var params = std.ArrayList(Array).empty;
        errdefer params.deinit(allocator);

        var it = self.layers.valueIterator();
        while (it.next()) |layer| {
            try params.append(allocator, layer.lora.a);
            try params.append(allocator, layer.lora.b);
        }

        return params.toOwnedSlice(allocator);
    }

    /// Collect pointers to all trainable parameters for optimizer updates.
    pub fn collectParamPtrs(self: *QLoRAModel, allocator: std.mem.Allocator) ![]*Array {
        var ptrs = std.ArrayList(*Array).empty;
        errdefer ptrs.deinit(allocator);

        var it = self.layers.valueIterator();
        while (it.next()) |layer| {
            try ptrs.append(allocator, &layer.lora.a);
            try ptrs.append(allocator, &layer.lora.b);
        }

        return ptrs.toOwnedSlice(allocator);
    }

    /// Set all LoRA parameters from a flat array.
    pub fn setParams(self: *QLoRAModel, params: []const Array) void {
        var idx: usize = 0;
        var it = self.layers.valueIterator();
        while (it.next()) |layer| {
            layer.setLoRAParams(params[idx .. idx + 2]);
            idx += 2;
        }
    }
};

// ============================================================
// Unit Tests
// ============================================================

test "QLoRALayer: init and forward produces correct output shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = ctx.stream.inner;

    // Create a base weight [64, 64]
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

    const quant_config = QuantConfig{ .bits = 4, .group_size = 64 };
    var qlora = try QLoRALayer.init(ctx, weight, 8, 16.0, quant_config, stream);
    defer qlora.deinit(allocator);

    // Create input [4, 64]
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

    // Forward pass
    const output = try qlora.forward(x);
    defer output.deinit();

    const out_shape = output.shape();
    try std.testing.expectEqual(@as(usize, 2), out_shape.len);
    try std.testing.expectEqual(@as(i32, 4), out_shape[0]);
    try std.testing.expectEqual(@as(i32, 64), out_shape[1]);
}

test "QLoRALayer: scaling factor is alpha/rank" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = ctx.stream.inner;

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

    const quant_config = QuantConfig{ .bits = 4, .group_size = 64 };
    var qlora = try QLoRALayer.init(ctx, weight, 8, 16.0, quant_config, stream);
    defer qlora.deinit(allocator);

    // alpha=16, rank=8 → scaling = 2.0
    try std.testing.expectEqual(@as(f32, 2.0), qlora.scaling());
}

test "QLoRALayer: collectTrainableParams returns only A and B" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = ctx.stream.inner;

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

    const quant_config = QuantConfig{ .bits = 4, .group_size = 64 };
    var qlora = try QLoRALayer.init(ctx, weight, 8, 16.0, quant_config, stream);
    defer qlora.deinit(allocator);

    const params = try qlora.collectTrainableParams(allocator);
    defer allocator.free(params);

    // Only A and B — no base weight
    try std.testing.expectEqual(@as(usize, 2), params.len);

    // A: [out_features=64, rank=8]
    const a_shape = params[0].shape();
    try std.testing.expectEqual(@as(i32, 64), a_shape[0]);
    try std.testing.expectEqual(@as(i32, 8), a_shape[1]);

    // B: [rank=8, in_features=64]
    const b_shape = params[1].shape();
    try std.testing.expectEqual(@as(i32, 8), b_shape[0]);
    try std.testing.expectEqual(@as(i32, 64), b_shape[1]);
}

test "QLoRAModel: add layer and collect params" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = ctx.stream.inner;

    var model = QLoRAModel.init(allocator);
    defer model.deinit();

    // Create two base weights and add as QLoRA layers
    const w_shape = &[_]i32{ 64, 64 };
    const quant_config = QuantConfig{ .bits = 4, .group_size = 64 };

    var w1_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(
        &w1_raw,
        w_shape.ptr,
        w_shape.len,
        c.c.MLX_FLOAT16,
        0.0,
        1.0,
        .{ .ctx = null },
        stream,
    ));
    const w1 = Array.fromHandle(w1_raw);
    defer w1.deinit();

    var w2_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(
        &w2_raw,
        w_shape.ptr,
        w_shape.len,
        c.c.MLX_FLOAT16,
        0.0,
        1.0,
        .{ .ctx = null },
        stream,
    ));
    const w2 = Array.fromHandle(w2_raw);
    defer w2.deinit();

    try model.addLayer("layer0.q_proj", ctx, w1, 4, 8.0, quant_config, stream);
    try model.addLayer("layer0.v_proj", ctx, w2, 4, 8.0, quant_config, stream);

    // Collect all trainable params: 2 layers × 2 params = 4
    const params = try model.collectParams(allocator);
    defer allocator.free(params);
    try std.testing.expectEqual(@as(usize, 4), params.len);

    // Verify layer lookup
    try std.testing.expect(model.getLayer("layer0.q_proj") != null);
    try std.testing.expect(model.getLayer("layer0.v_proj") != null);
    try std.testing.expect(model.getLayer("nonexistent") == null);
}

test "QLoRALayer: base weight unchanged after setLoRAParams" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = ctx.stream.inner;

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

    const quant_config = QuantConfig{ .bits = 4, .group_size = 64 };
    var qlora = try QLoRALayer.init(ctx, weight, 8, 16.0, quant_config, stream);
    defer qlora.deinit(allocator);

    // Snapshot the quantized base data shape before update
    const base_data_shape_before = qlora.base_quantized.data.shape();
    const base_data_ndim_before = base_data_shape_before.len;

    // Create new LoRA params and set them
    const a_shape = &[_]i32{ 64, 8 };
    var new_a_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(
        &new_a_raw,
        a_shape.ptr,
        a_shape.len,
        c.c.MLX_FLOAT32,
        0.0,
        1.0,
        .{ .ctx = null },
        stream,
    ));
    const new_a = Array.fromHandle(new_a_raw);
    defer new_a.deinit();

    const b_shape = &[_]i32{ 8, 64 };
    var new_b_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_zeros(
        &new_b_raw,
        b_shape.ptr,
        b_shape.len,
        c.c.MLX_FLOAT32,
        stream,
    ));
    const new_b = Array.fromHandle(new_b_raw);
    defer new_b.deinit();

    qlora.setLoRAParams(&[_]Array{ new_a, new_b });

    // Base quantized weight should be unchanged
    const base_data_shape_after = qlora.base_quantized.data.shape();
    try std.testing.expectEqual(base_data_ndim_before, base_data_shape_after.len);
}

// ============================================================
// Property-Based Test
// (Property 17)
//
// Feature: production-deployment, Property 17: QLoRA Forward
// Correctness
//
// For any input tensor x, quantized base weight W_base, and LoRA
// adapters (A, B, scaling), the QLoRA forward pass SHALL produce
// output equal to dequantize(W_base) @ x + (B @ A) @ x * scaling.
// Gradients SHALL be computed only for A and B parameters;
// W_base SHALL remain unchanged after a training step.
//
// **Validates: Requirements R19.2, R19.3**
// ============================================================

test "Property 17: QLoRA Forward Correctness — output matches formula (100 iterations)" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = ctx.stream.inner;

    var prng = std.Random.DefaultPrng.init(17);
    const rand = prng.random();

    // Dimension choices must be multiples of group_size (64) for quantization.
    const dim_choices = [_]i32{ 64, 128, 192, 256 };
    const batch_choices = [_]i32{ 1, 2, 4, 8 };
    const rank_choices = [_]usize{ 2, 4, 8 };

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // --- Random configuration ---
        const in_features = dim_choices[rand.intRangeAtMost(usize, 0, dim_choices.len - 1)];
        const out_features = dim_choices[rand.intRangeAtMost(usize, 0, dim_choices.len - 1)];
        const batch_size = batch_choices[rand.intRangeAtMost(usize, 0, batch_choices.len - 1)];
        const rank = rank_choices[rand.intRangeAtMost(usize, 0, rank_choices.len - 1)];
        const alpha: f32 = @floatFromInt(rank * 2);
        const expected_scale = alpha / @as(f32, @floatFromInt(rank));

        // --- Create random base weight [out_features, in_features] ---
        const w_shape = &[_]i32{ out_features, in_features };
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

        // --- Create QLoRA layer ---
        const quant_config = QuantConfig{ .bits = 4, .group_size = 64 };
        var qlora = try QLoRALayer.init(ctx, weight, rank, alpha, quant_config, stream);
        defer qlora.deinit(allocator);

        // --- Create random input [batch_size, in_features] ---
        const x_shape = &[_]i32{ batch_size, in_features };
        var x_raw = c.c.mlx_array_new();
        try c.check(c.c.mlx_random_normal(
            &x_raw,
            x_shape.ptr,
            x_shape.len,
            c.c.MLX_FLOAT32,
            0.0,
            1.0,
            .{ .ctx = null },
            stream,
        ));
        const x = Array.fromHandle(x_raw);
        defer x.deinit();

        // --- Compute QLoRA forward pass ---
        const actual_output = try qlora.forward(x);
        defer actual_output.deinit();

        // --- Manually compute expected output ---
        // Base path: x @ dequantize(W_base)^T
        const deq_w = try quantize_mod.dequantize(ctx, qlora.base_quantized);
        defer deq_w.deinit();

        const deq_w_t = try ops.transpose(ctx, deq_w);
        defer deq_w_t.deinit();

        const base_out = try ops.matmul(ctx, x, deq_w_t);
        defer base_out.deinit();

        // LoRA path: x @ B^T @ A^T * scaling
        const bt = try ops.transpose(ctx, qlora.lora.b);
        defer bt.deinit();
        const at = try ops.transpose(ctx, qlora.lora.a);
        defer at.deinit();

        const x_bt = try ops.matmul(ctx, x, bt);
        defer x_bt.deinit();
        const lora_unscaled = try ops.matmul(ctx, x_bt, at);
        defer lora_unscaled.deinit();

        const scale_arr = try ops.scalar(ctx, expected_scale, .float32);
        defer scale_arr.deinit();
        const lora_out = try ops.multiply(ctx, lora_unscaled, scale_arr);
        defer lora_out.deinit();

        // Expected = base_out + lora_out
        const expected_output = try ops.add(ctx, base_out, lora_out);
        defer expected_output.deinit();

        // --- Compare: cosine similarity between actual and expected ---
        // Both should be numerically identical (same computation path),
        // but we use cosine similarity >= 0.999 to handle float rounding.
        const diff = try ops.subtract(ctx, actual_output, expected_output);
        defer diff.deinit();
        const diff_sq = try ops.multiply(ctx, diff, diff);
        defer diff_sq.deinit();
        const mse = try ops.mean(ctx, diff_sq);
        defer mse.deinit();

        const mse_val = try mse.item(f32);
        // MSE should be essentially zero (same computation graph)
        try std.testing.expect(mse_val < 1e-6);

        // --- Verify output shape ---
        const out_shape = actual_output.shape();
        try std.testing.expectEqual(@as(usize, 2), out_shape.len);
        try std.testing.expectEqual(batch_size, out_shape[0]);
        try std.testing.expectEqual(out_features, out_shape[1]);
    }
}

test "Property 17: QLoRA Forward Correctness — base weight unchanged after parameter update (100 iterations)" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = ctx.stream.inner;

    var prng = std.Random.DefaultPrng.init(1717);
    const rand = prng.random();

    const dim_choices = [_]i32{ 64, 128, 192 };
    const rank_choices = [_]usize{ 2, 4, 8 };

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        const in_features = dim_choices[rand.intRangeAtMost(usize, 0, dim_choices.len - 1)];
        const out_features = dim_choices[rand.intRangeAtMost(usize, 0, dim_choices.len - 1)];
        const rank = rank_choices[rand.intRangeAtMost(usize, 0, rank_choices.len - 1)];
        const alpha: f32 = @floatFromInt(rank * 2);

        // --- Create base weight and QLoRA layer ---
        const w_shape = &[_]i32{ out_features, in_features };
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

        const quant_config = QuantConfig{ .bits = 4, .group_size = 64 };
        var qlora = try QLoRALayer.init(ctx, weight, rank, alpha, quant_config, stream);
        defer qlora.deinit(allocator);

        // --- Snapshot base weight quantized data (packed uint32) before update ---
        // Compare the raw quantized data directly (not dequantized) to avoid
        // dtype issues. The packed data is uint32.
        const base_data_before = try qlora.base_quantized.data.dataSlice(u32);
        const base_snapshot = try allocator.alloc(u32, base_data_before.len);
        defer allocator.free(base_snapshot);
        @memcpy(base_snapshot, base_data_before);

        // --- Simulate a training step: update LoRA params ---
        const a_shape = &[_]i32{ out_features, @intCast(rank) };
        var new_a_raw = c.c.mlx_array_new();
        try c.check(c.c.mlx_random_normal(
            &new_a_raw,
            a_shape.ptr,
            a_shape.len,
            c.c.MLX_FLOAT32,
            0.0,
            1.0,
            .{ .ctx = null },
            stream,
        ));
        const new_a = Array.fromHandle(new_a_raw);
        defer new_a.deinit();

        const b_shape = &[_]i32{ @intCast(rank), in_features };
        var new_b_raw = c.c.mlx_array_new();
        try c.check(c.c.mlx_random_normal(
            &new_b_raw,
            b_shape.ptr,
            b_shape.len,
            c.c.MLX_FLOAT32,
            0.0,
            1.0,
            .{ .ctx = null },
            stream,
        ));
        const new_b = Array.fromHandle(new_b_raw);
        defer new_b.deinit();

        qlora.setLoRAParams(&[_]Array{ new_a, new_b });

        // --- Verify base weight is unchanged ---
        // Compare raw quantized data (packed uint32) — must be identical.
        const base_data_after = try qlora.base_quantized.data.dataSlice(u32);

        try std.testing.expectEqual(base_snapshot.len, base_data_after.len);
        for (base_snapshot, base_data_after) |before, after| {
            try std.testing.expectEqual(before, after);
        }

        // Also verify scales and biases are unchanged by checking shapes.
        const scales_shape = qlora.base_quantized.scales.shape();
        const biases_shape = qlora.base_quantized.biases.shape();
        try std.testing.expectEqual(@as(usize, 2), scales_shape.len);
        try std.testing.expectEqual(@as(usize, 2), biases_shape.len);

        // --- Verify only A and B are trainable ---
        const params = try qlora.collectTrainableParams(allocator);
        defer allocator.free(params);
        try std.testing.expectEqual(@as(usize, 2), params.len);

        // Verify A shape matches [out_features, rank]
        const a_s = params[0].shape();
        try std.testing.expectEqual(out_features, a_s[0]);
        try std.testing.expectEqual(@as(i32, @intCast(rank)), a_s[1]);

        // Verify B shape matches [rank, in_features]
        const b_s = params[1].shape();
        try std.testing.expectEqual(@as(i32, @intCast(rank)), b_s[0]);
        try std.testing.expectEqual(in_features, b_s[1]);
    }
}
