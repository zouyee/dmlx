//! Qwen3.8 (qwen3_5) config.json parsing. Language-model hyperparameters live
//! under `text_config`; quantization defaults under top-level `quantization`.
const std = @import("std");

pub const LoadError = error{
    InvalidConfig,
    UnsupportedArchitecture,
};

pub const Qwen38Config = struct {
    allocator: std.mem.Allocator,
    hidden_size: u32 = 5120,
    num_hidden_layers: u32 = 64,
    vocab_size: u32 = 248320,
    intermediate_size: u32 = 17408,
    num_attention_heads: u32 = 24,
    num_key_value_heads: u32 = 4,
    head_dim: u32 = 256,
    partial_rotary_factor: f32 = 0.25,
    rope_theta: f32 = 1e7,
    rms_norm_eps: f32 = 1e-6,
    attn_output_gate: bool = true,
    linear_num_key_heads: u32 = 16,
    linear_num_value_heads: u32 = 48,
    linear_key_head_dim: u32 = 128,
    linear_value_head_dim: u32 = 128,
    linear_conv_kernel_dim: u32 = 4,
    eos_token_ids: []u32 = &.{},
    full_attn_layers: []u32 = &.{},
    quant_bits_default: u8 = 4,
    quant_group_size: u32 = 64,

    pub fn deinit(self: *Qwen38Config) void {
        if (self.eos_token_ids.len > 0) self.allocator.free(self.eos_token_ids);
        if (self.full_attn_layers.len > 0) self.allocator.free(self.full_attn_layers);
    }

    pub fn isFullAttention(self: *const Qwen38Config, layer: u32) bool {
        for (self.full_attn_layers) |l| {
            if (l == layer) return true;
        }
        return false;
    }

    /// Rotary dims = head_dim * partial_rotary_factor (64 for the real config).
    pub fn rotaryDims(self: *const Qwen38Config) usize {
        return @intFromFloat(@as(f32, @floatFromInt(self.head_dim)) * self.partial_rotary_factor);
    }

    /// kv-group ratio: value heads per key head (3 for the real config).
    pub fn kvGroupRatio(self: *const Qwen38Config) u32 {
        return self.linear_num_value_heads / self.linear_num_key_heads;
    }
};

const getInt = struct {
    fn call(o: std.json.ObjectMap, key: []const u8) ?i64 {
        const val = o.get(key) orelse return null;
        return switch (val) {
            .integer => |n| n,
            .float => |f| @intFromFloat(f),
            else => null,
        };
    }
}.call;

const getFloat = struct {
    fn call(o: std.json.ObjectMap, key: []const u8) ?f64 {
        const val = o.get(key) orelse return null;
        return switch (val) {
            .float => |f| f,
            .integer => |n| @floatFromInt(n),
            else => null,
        };
    }
}.call;

const getBool = struct {
    fn call(o: std.json.ObjectMap, key: []const u8) ?bool {
        const val = o.get(key) orelse return null;
        return switch (val) {
            .bool => |b| b,
            else => null,
        };
    }
}.call;

fn nestedInt(text: ?std.json.ObjectMap, top: std.json.ObjectMap, key: []const u8) ?i64 {
    if (text) |t| {
        if (getInt(t, key)) |v| return v;
    }
    return getInt(top, key);
}

fn nestedFloat(text: ?std.json.ObjectMap, top: std.json.ObjectMap, key: []const u8) ?f64 {
    if (text) |t| {
        if (getFloat(t, key)) |v| return v;
    }
    return getFloat(top, key);
}

fn toU32(v: ?i64, default: u32) u32 {
    const n = v orelse return default;
    return std.math.cast(u32, n) orelse default;
}

fn toF32(v: ?f64, default: f32) f32 {
    const f = v orelse return default;
    return @floatCast(f);
}

fn appendEosIds(allocator: std.mem.Allocator, list: *std.ArrayList(u32), val: std.json.Value) !void {
    switch (val) {
        .integer => |n| try appendEosId(allocator, list, n),
        .array => |arr| {
            for (arr.items) |item| {
                const n: ?i64 = switch (item) {
                    .integer => |v| v,
                    .float => |f| @intFromFloat(f),
                    else => null,
                };
                if (n) |v| try appendEosId(allocator, list, v);
            }
        },
        else => {},
    }
}

fn appendEosId(allocator: std.mem.Allocator, list: *std.ArrayList(u32), n: i64) !void {
    const id = std.math.cast(u32, n) orelse return;
    for (list.items) |existing| {
        if (existing == id) return;
    }
    try list.append(allocator, id);
}

/// Parse Qwen3.8 config.json into Qwen38Config.
/// Caller must call `cfg.deinit()` to free owned slices.
pub fn parseQwen38Config(allocator: std.mem.Allocator, json_text: []const u8) !Qwen38Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return LoadError.InvalidConfig;
    const obj = root.object;

    if (obj.get("model_type")) |mt| {
        if (mt == .string and !std.mem.startsWith(u8, mt.string, "qwen3_5")) {
            return LoadError.UnsupportedArchitecture;
        }
    }

    const text: ?std.json.ObjectMap = blk: {
        const tv = obj.get("text_config") orelse break :blk null;
        if (tv != .object) break :blk null;
        break :blk tv.object;
    };

    var cfg = Qwen38Config{ .allocator = allocator };
    errdefer cfg.deinit();

    cfg.hidden_size = toU32(nestedInt(text, obj, "hidden_size"), cfg.hidden_size);
    cfg.num_hidden_layers = toU32(nestedInt(text, obj, "num_hidden_layers"), cfg.num_hidden_layers);
    cfg.vocab_size = toU32(nestedInt(text, obj, "vocab_size"), cfg.vocab_size);
    cfg.intermediate_size = toU32(nestedInt(text, obj, "intermediate_size"), cfg.intermediate_size);
    cfg.num_attention_heads = toU32(nestedInt(text, obj, "num_attention_heads"), cfg.num_attention_heads);
    cfg.num_key_value_heads = toU32(nestedInt(text, obj, "num_key_value_heads"), cfg.num_key_value_heads);
    cfg.head_dim = toU32(nestedInt(text, obj, "head_dim"), cfg.head_dim);
    cfg.partial_rotary_factor = toF32(nestedFloat(text, obj, "partial_rotary_factor"), cfg.partial_rotary_factor);
    cfg.rms_norm_eps = toF32(nestedFloat(text, obj, "rms_norm_eps"), cfg.rms_norm_eps);
    cfg.attn_output_gate = blk: {
        if (text) |t| {
            if (getBool(t, "attn_output_gate")) |b| break :blk b;
        }
        break :blk cfg.attn_output_gate;
    };
    cfg.linear_num_key_heads = toU32(nestedInt(text, obj, "linear_num_key_heads"), cfg.linear_num_key_heads);
    cfg.linear_num_value_heads = toU32(nestedInt(text, obj, "linear_num_value_heads"), cfg.linear_num_value_heads);
    cfg.linear_key_head_dim = toU32(nestedInt(text, obj, "linear_key_head_dim"), cfg.linear_key_head_dim);
    cfg.linear_value_head_dim = toU32(nestedInt(text, obj, "linear_value_head_dim"), cfg.linear_value_head_dim);
    cfg.linear_conv_kernel_dim = toU32(nestedInt(text, obj, "linear_conv_kernel_dim"), cfg.linear_conv_kernel_dim);

    // rope_theta lives under text_config.rope_parameters.rope_theta.
    if (text) |t| {
        if (t.get("rope_parameters")) |rp| {
            if (rp == .object) {
                cfg.rope_theta = toF32(getFloat(rp.object, "rope_theta"), cfg.rope_theta);
            }
        }
    }

    // eos_token_id: union of text_config and top-level values (each an integer
    // or array), deduplicated, text_config entries first.
    var eos_list = std.ArrayList(u32).empty;
    defer eos_list.deinit(allocator);
    if (text) |t| {
        if (t.get("eos_token_id")) |v| try appendEosIds(allocator, &eos_list, v);
    }
    if (obj.get("eos_token_id")) |v| try appendEosIds(allocator, &eos_list, v);
    if (eos_list.items.len > 0) cfg.eos_token_ids = try eos_list.toOwnedSlice(allocator);

    // layer_types: collect indices of "full_attention" entries.
    if (text) |t| {
        if (t.get("layer_types")) |lt| {
            if (lt == .array) {
                var list = std.ArrayList(u32).empty;
                defer list.deinit(allocator);
                for (lt.array.items, 0..) |item, i| {
                    if (item != .string) return LoadError.InvalidConfig;
                    if (std.mem.eql(u8, item.string, "full_attention")) {
                        try list.append(allocator, @intCast(i));
                    } else if (!std.mem.eql(u8, item.string, "linear_attention")) {
                        std.log.err("qwen38: unknown layer_types[{d}] = \"{s}\"", .{ i, item.string });
                        return LoadError.InvalidConfig;
                    }
                }
                if (list.items.len > 0) cfg.full_attn_layers = try list.toOwnedSlice(allocator);
            }
        }
    }

    // Top-level quantization defaults.
    if (obj.get("quantization")) |q_val| {
        if (q_val == .object) {
            const q = q_val.object;
            cfg.quant_bits_default = @intCast(@min(255, @max(0, getInt(q, "bits") orelse cfg.quant_bits_default)));
            cfg.quant_group_size = toU32(getInt(q, "group_size"), cfg.quant_group_size);
        }
    }

    return cfg;
}

test "parseQwen38Config: text_config nesting + layer_types" {
    const allocator = std.testing.allocator;
    const json_text =
        \\{"model_type":"qwen3_5","eos_token_id":[248046,248044],
        \\  "quantization":{"group_size":64,"bits":4,"mode":"affine"},
        \\  "text_config":{
        \\    "hidden_size":5120,"num_hidden_layers":64,"vocab_size":248320,
        \\    "intermediate_size":17408,"head_dim":256,
        \\    "num_attention_heads":24,"num_key_value_heads":4,
        \\    "partial_rotary_factor":0.25,"rms_norm_eps":1e-6,
        \\    "linear_num_key_heads":16,"linear_num_value_heads":48,
        \\    "linear_key_head_dim":128,"linear_value_head_dim":128,
        \\    "linear_conv_kernel_dim":4,
        \\    "attn_output_gate":true,
        \\    "eos_token_id":248044,
        \\    "rope_parameters":{"rope_theta":10000000},
        \\    "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"]
        \\  }}
    ;
    var cfg = try parseQwen38Config(allocator, json_text);
    defer cfg.deinit();
    try std.testing.expectEqual(@as(u32, 64), cfg.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 5120), cfg.hidden_size);
    try std.testing.expectEqual(@as(u32, 248320), cfg.vocab_size);
    try std.testing.expectEqual(@as(u32, 16), cfg.linear_num_key_heads);
    try std.testing.expectEqual(@as(u32, 48), cfg.linear_num_value_heads);
    try std.testing.expectEqual(@as(u32, 128), cfg.linear_key_head_dim);
    try std.testing.expectEqual(@as(u32, 128), cfg.linear_value_head_dim);
    try std.testing.expectEqual(@as(u32, 24), cfg.num_attention_heads);
    try std.testing.expectEqual(@as(u32, 4), cfg.num_key_value_heads);
    try std.testing.expectEqual(@as(u32, 256), cfg.head_dim);
    try std.testing.expectEqual(@as(usize, 64), cfg.rotaryDims());
    try std.testing.expectEqual(@as(f32, 1e7), cfg.rope_theta);
    try std.testing.expectEqual(@as(usize, 2), cfg.eos_token_ids.len);
    try std.testing.expectEqual(@as(u32, 248044), cfg.eos_token_ids[0]);
    try std.testing.expectEqual(@as(u32, 248046), cfg.eos_token_ids[1]);
    try std.testing.expect(cfg.isFullAttention(3));
    try std.testing.expect(!cfg.isFullAttention(0));
    try std.testing.expect(!cfg.isFullAttention(64)); // out of range is safe
    try std.testing.expectEqual(@as(u8, 4), cfg.quant_bits_default);
    try std.testing.expectEqual(@as(u32, 64), cfg.quant_group_size);
}

test "parseQwen38Config: missing fields fall back to real-config defaults" {
    const allocator = std.testing.allocator;
    var cfg = try parseQwen38Config(allocator, "{\"model_type\":\"qwen3_5\",\"text_config\":{\"num_hidden_layers\":64}}");
    defer cfg.deinit();
    try std.testing.expectEqual(@as(u32, 5120), cfg.hidden_size);
    try std.testing.expectEqual(@as(u32, 248320), cfg.vocab_size);
    try std.testing.expectEqual(@as(u32, 17408), cfg.intermediate_size);
    try std.testing.expectEqual(@as(usize, 0), cfg.eos_token_ids.len);
    try std.testing.expect(!cfg.isFullAttention(0));
}

test "parseQwen38Config: top-level integer eos_token_id" {
    const allocator = std.testing.allocator;
    var cfg = try parseQwen38Config(allocator, "{\"model_type\":\"qwen3_5\",\"eos_token_id\":248046}");
    defer cfg.deinit();
    try std.testing.expectEqual(@as(usize, 1), cfg.eos_token_ids.len);
    try std.testing.expectEqual(@as(u32, 248046), cfg.eos_token_ids[0]);
}

test "parseQwen38Config: rejects non-qwen3_5 model_type" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(
        LoadError.UnsupportedArchitecture,
        parseQwen38Config(allocator, "{\"model_type\":\"llama\"}"),
    );
}
