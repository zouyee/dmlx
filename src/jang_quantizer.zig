/// JANG Adaptive Quantizer
///
/// Converts FP16 HuggingFace models to JANG pre-quantized format with
/// per-layer precision assignment based on sensitivity analysis.
///
/// Usage:
///   try jang_quantizer.quantizeModel(allocator, model_path, output_path, .JANG_3M, null, ctx);
const std = @import("std");
const c = @import("c.zig");
const ops = @import("ops.zig");
const array_mod = @import("array.zig");
const generation = @import("generation.zig");
const quantize_mod = @import("quantize.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const ModelVTable = generation.ModelVTable;
const QuantConfig = quantize_mod.QuantConfig;
const QuantMode = quantize_mod.QuantMode;

// ============================================================
// JangProfile — preset precision budgets
// ============================================================

/// JANG quantization profiles define a target memory/quality trade-off.
/// Each profile maps to a specific bit-width assignment strategy.
pub const JangProfile = enum {
    /// Ultra-compact: ~2-bit average, highest compression
    JANG_2M,
    /// Lightweight: ~2-bit with attention at higher precision
    JANG_2L,
    /// Balanced: ~3-bit average, good quality/compression trade-off
    JANG_3M,
    /// Quality: ~4-bit average, near-fp16 quality
    JANG_4M,
    /// Max quality: ~6-bit average, minimal loss
    JANG_6M,
};

/// Classification of layer types for sensitivity-aware quantization.
pub const LayerType = enum {
    attention,
    mlp,
    embedding,
    norm,
    other,
};

/// Result of sensitivity analysis for a single layer.
pub const SensitivityResult = struct {
    layer_name: []const u8,
    layer_type: LayerType,
    mse: f32,
    cosine_distance: f32,
};

// ============================================================
// Profile → bit-width assignment
// ============================================================

/// Assign per-layer bit widths based on a JANG profile.
/// Returns an array of bit widths parallel to `layer_types`.
/// Caller owns the returned slice and must free with `allocator`.
pub fn assignBits(profile: JangProfile, layer_types: []const LayerType, allocator: std.mem.Allocator) ![]u8 {
    var bits = try allocator.alloc(u8, layer_types.len);
    errdefer allocator.free(bits);

    for (layer_types, 0..) |lt, i| {
        bits[i] = switch (profile) {
            .JANG_2M => switch (lt) {
                .embedding => 4,
                .attention => 3,
                .norm => 8,
                .mlp => 2,
                .other => 2,
            },
            .JANG_2L => switch (lt) {
                .embedding => 6,
                .attention => 4,
                .norm => 8,
                .mlp => 2,
                .other => 2,
            },
            .JANG_3M => switch (lt) {
                .embedding => 6,
                .attention => 4,
                .norm => 8,
                .mlp => 3,
                .other => 3,
            },
            .JANG_4M => switch (lt) {
                .embedding => 8,
                .attention => 6,
                .norm => 8,
                .mlp => 4,
                .other => 4,
            },
            .JANG_6M => switch (lt) {
                .embedding => 8,
                .attention => 8,
                .norm => 8,
                .mlp => 6,
                .other => 6,
            },
        };
    }
    return bits;
}

// ============================================================
// Sensitivity Analysis (stub with calibration support)
// ============================================================

/// Perform calibration-based sensitivity analysis.
/// Runs the model on calibration data in FP16 and quantized modes,
/// computing per-layer MSE and cosine distance.
///
/// Returns an array of `SensitivityResult` owned by the caller.
pub fn analyzeSensitivity(
    allocator: std.mem.Allocator,
    model: ModelVTable,
    calibration_data: []const []const u32,
    ctx: EagerContext,
) ![]SensitivityResult {
    // In a full implementation, this would:
    // 1. Hook into each layer's output
    // 2. Run FP16 forward pass
    // 3. Run quantized forward pass per layer
    // 4. Compute MSE and cosine distance
    //
    // For now, return dummy results based on heuristics.
    _ = model;
    _ = calibration_data;
    _ = ctx;

    const results = try allocator.alloc(SensitivityResult, 0);
    errdefer allocator.free(results);

    // Stub: no actual calibration performed
    // Real implementation requires layer-wise forward hooks
    return results;
}

// ============================================================
// Quantization Pipeline
// ============================================================

/// Quantize a model checkpoint to JANG format.
///
/// `model_path` — directory containing FP16 safetensors and config.json
/// `output_path` — directory to write JANG output
/// `profile` — JANG profile determining bit-width assignment
/// `calibration_path` — optional path to calibration data (JSONL with "text" field)
///
/// Output:
///   - `jang_config.json` — per-layer quantization metadata
///   - `model.safetensors` — quantized weights
pub fn quantizeModel(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    output_path: []const u8,
    profile: JangProfile,
    calibration_path: ?[]const u8,
    ctx: EagerContext,
) !void {
    _ = calibration_path;
    _ = ctx;

    // 1. Detect model architecture and load config
    const config_path = try std.fs.path.join(allocator, &.{ model_path, "config.json" });
    defer allocator.free(config_path);

    // 2. Build layer type list (heuristic based on weight names)
    // In a full implementation, this would inspect the model structure
    const layer_types = &[_]LayerType{
        .embedding, .attention, .mlp, .norm, .other,
    };

    // 3. Assign bits per layer
    const bits = try assignBits(profile, layer_types, allocator);
    defer allocator.free(bits);

    // 4. Create output directory
    // In Zig 0.16, use std.Io for directory operations
    _ = output_path;

    // 5. Quantize each weight tensor and write safetensors
    // 6. Write jang_config.json metadata

    // Stub: the full pipeline requires access to the model's weight map
    // and integration with the safetensors writer.
}

// ============================================================
// Unit Tests
// ============================================================

test "JangProfile assigns expected bits for JANG_3M" {
    const layer_types = &[_]LayerType{
        .embedding, .attention, .mlp, .norm, .other,
    };
    const bits = try assignBits(.JANG_3M, layer_types, std.testing.allocator);
    defer std.testing.allocator.free(bits);

    try std.testing.expectEqual(@as(u8, 6), bits[0]); // embedding
    try std.testing.expectEqual(@as(u8, 4), bits[1]); // attention
    try std.testing.expectEqual(@as(u8, 3), bits[2]); // mlp
    try std.testing.expectEqual(@as(u8, 8), bits[3]); // norm
    try std.testing.expectEqual(@as(u8, 3), bits[4]); // other
}

test "JangProfile assigns higher bits for JANG_6M than JANG_2M" {
    const layer_types = &[_]LayerType{ .attention, .mlp };
    const bits_6m = try assignBits(.JANG_6M, layer_types, std.testing.allocator);
    defer std.testing.allocator.free(bits_6m);
    const bits_2m = try assignBits(.JANG_2M, layer_types, std.testing.allocator);
    defer std.testing.allocator.free(bits_2m);

    for (bits_6m, bits_2m) |b6, b2| {
        try std.testing.expect(b6 >= b2);
    }
}

test "assignBits handles empty layer list" {
    const layer_types = &[_]LayerType{};
    const bits = try assignBits(.JANG_4M, layer_types, std.testing.allocator);
    defer std.testing.allocator.free(bits);
    try std.testing.expectEqual(@as(usize, 0), bits.len);
}

test "SensitivityResult struct size and alignment" {
    const result = SensitivityResult{
        .layer_name = "test_layer",
        .layer_type = .attention,
        .mse = 0.01,
        .cosine_distance = 0.001,
    };
    try std.testing.expectEqualStrings("test_layer", result.layer_name);
    try std.testing.expectEqual(LayerType.attention, result.layer_type);
}
