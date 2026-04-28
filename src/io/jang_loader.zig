/// JANG format loader — loads vMLX JANG pre-quantized models.
///
/// JANG models contain:
///   - `jang_config.json`: per-layer quantization metadata
///   - `*.safetensors`: quantized weight files (packed uint32 data + scales + biases)
///
/// The loader reads the config, maps each tensor to its `LayerQuantConfig`,
/// and returns `QuantizedWeight` structs ready for inference.
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const ops = @import("../ops.zig");
const quantize_mod = @import("../quantize.zig");
const mlx_io = @import("mlx_io.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const QuantConfig = quantize_mod.QuantConfig;
const QuantMode = quantize_mod.QuantMode;

// ============================================================
// JANG Config Types
// ============================================================

/// Per-layer quantization configuration from jang_config.json.
pub const LayerQuantConfig = struct {
    name: []const u8,
    bits: u8,
    mode: QuantMode,
    group_size: i32 = 64,
};

/// Parsed JANG config.
pub const JangConfig = struct {
    layers: []const LayerQuantConfig,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *JangConfig) void {
        for (self.layers) |*layer| {
            self.allocator.free(layer.name);
        }
        self.allocator.free(self.layers);
    }

    /// Look up quantization config for a tensor name.
    /// Returns null if no specific config is found.
    pub fn lookup(self: *const JangConfig, tensor_name: []const u8) ?LayerQuantConfig {
        for (self.layers) |layer| {
            if (std.mem.eql(u8, layer.name, tensor_name)) {
                return layer;
            }
        }
        return null;
    }
};

// ============================================================
// Config Parsing
// ============================================================

/// Parse jang_config.json into a JangConfig.
/// Caller owns returned JangConfig and must call deinit.
pub fn parseConfig(allocator: std.mem.Allocator, json_text: []const u8) !JangConfig {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();
    const root_val = parsed.value;

    var layers = std.ArrayList(LayerQuantConfig).empty;
    errdefer {
        for (layers.items) |*l| allocator.free(l.name);
        layers.deinit(allocator);
    }

    if (root_val.object.get("layers")) |layers_val| {
        if (layers_val == .array) {
            for (layers_val.array.items) |item| {
                if (item != .object) continue;
                const name = item.object.get("name") orelse continue;
                const bits = item.object.get("bits") orelse continue;
                const mode_str = item.object.get("mode") orelse continue;

                const layer_name = try allocator.dupe(u8, name.string);
                errdefer allocator.free(layer_name);

                const mode = parseMode(mode_str.string);
                const group_size: i32 = if (item.object.get("group_size")) |gs|
                    @intCast(gs.integer)
                else
                    switch (mode) {
                        .mxfp4, .mxfp8 => 32,
                        .nvfp4 => 16,
                        .affine => 64,
                    };

                try layers.append(allocator, .{
                    .name = layer_name,
                    .bits = @intCast(bits.integer),
                    .mode = mode,
                    .group_size = group_size,
                });
            }
        }
    }

    return JangConfig{
        .layers = try layers.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

fn parseMode(mode_str: []const u8) QuantMode {
    if (std.mem.eql(u8, mode_str, "affine")) return .affine;
    if (std.mem.eql(u8, mode_str, "mxfp4")) return .mxfp4;
    if (std.mem.eql(u8, mode_str, "nvfp4")) return .nvfp4;
    if (std.mem.eql(u8, mode_str, "mxfp8")) return .mxfp8;
    return .affine;
}

// ============================================================
// Weight Loading
// ============================================================

/// Load JANG quantized weights from safetensors, applying per-layer quantization config.
/// Returns a map from tensor name → Array (quantized or dequantized).
pub fn loadWeights(
    allocator: std.mem.Allocator,
    jang_config: *const JangConfig,
    safetensors_paths: []const []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
) !std.StringHashMap(Array) {
    var result = std.StringHashMap(Array).init(allocator);
    errdefer {
        var it = result.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        result.deinit();
    }

    for (safetensors_paths) |path| {
        var st = try mlx_io.loadSafetensors(allocator, path);
        defer {
            var w_it = st.weights.iterator();
            while (w_it.next()) |entry| {
                allocator.free(entry.key_ptr.*);
            }
            st.weights.deinit();
            var m_it = st.metadata.iterator();
            while (m_it.next()) |entry| {
                allocator.free(entry.key_ptr.*);
                allocator.free(entry.value_ptr.*);
            }
            st.metadata.deinit();
        }

        var it = st.weights.iterator();
        while (it.next()) |entry| {
            const tensor_name = entry.key_ptr.*;
            const weight = entry.value_ptr.*;

            if (jang_config.lookup(tensor_name)) |layer_config| {
                // JANG pre-quantized: weight is already packed data
                // We need to extract data, scales, biases from the safetensors tensor
                // and wrap them in a QuantizedWeight.
                const quant_config = QuantConfig{
                    .bits = layer_config.bits,
                    .group_size = layer_config.group_size,
                    .mode = layer_config.mode,
                };

                // For JANG, the safetensors file already contains quantized data.
                // We treat the loaded array as the quantized representation.
                // The caller (model loader) will use quantizedMatmul or dequantize as needed.
                const quantized = try quantize_mod.quantize(weight, quant_config, ctx, stream);
                const name_copy = try allocator.dupe(u8, tensor_name);
                errdefer allocator.free(name_copy);
                try result.put(name_copy, quantized);
            } else {
                // No JANG config for this layer: load as-is (fp16/fp32)
                const name_copy = try allocator.dupe(u8, tensor_name);
                errdefer allocator.free(name_copy);
                try result.put(name_copy, try weight.copy(ctx.allocator));
            }
        }
    }

    return result;
}

/// Detect if a model directory contains JANG config.
pub fn isJangModel(dir_path: []const u8) bool {
    const config_path = std.fs.path.join(std.heap.page_allocator, &[_][]const u8{ dir_path, "jang_config.json" }) catch return false;
    defer std.heap.page_allocator.free(config_path);

    std.Io.Dir.cwd().access(std.io.null_writer, config_path, .{}) catch return false;
    return true;
}

// ============================================================
// Unit Tests
// ============================================================

test "parseConfig basic" {
    const allocator = std.testing.allocator;
    const json =
        \\{"layers": [
        \\  {"name": "self_attn.q_proj", "bits": 8, "mode": "affine", "group_size": 128},
        \\  {"name": "mlp.gate_proj", "bits": 2, "mode": "affine"}
        \\]}
    ;

    var config = try parseConfig(allocator, json);
    defer config.deinit();

    try std.testing.expectEqual(@as(usize, 2), config.layers.len);
    try std.testing.expectEqualStrings("self_attn.q_proj", config.layers[0].name);
    try std.testing.expectEqual(@as(u8, 8), config.layers[0].bits);
    try std.testing.expectEqual(@as(i32, 128), config.layers[0].group_size);
    try std.testing.expectEqualStrings("mlp.gate_proj", config.layers[1].name);
    try std.testing.expectEqual(@as(u8, 2), config.layers[1].bits);
}

test "JangConfig lookup" {
    const allocator = std.testing.allocator;
    const json =
        \\{"layers": [
        \\  {"name": "q_proj", "bits": 4, "mode": "mxfp4"}
        \\]}
    ;

    var config = try parseConfig(allocator, json);
    defer config.deinit();

    try std.testing.expect(config.lookup("q_proj") != null);
    try std.testing.expect(config.lookup("k_proj") == null);
}

test "parseMode" {
    try std.testing.expectEqual(QuantMode.affine, parseMode("affine"));
    try std.testing.expectEqual(QuantMode.mxfp4, parseMode("mxfp4"));
    try std.testing.expectEqual(QuantMode.nvfp4, parseMode("nvfp4"));
    try std.testing.expectEqual(QuantMode.mxfp8, parseMode("mxfp8"));
    try std.testing.expectEqual(QuantMode.affine, parseMode("unknown"));
}
