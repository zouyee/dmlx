//! Qwen3.8 (qwen3_5) config.json parsing. Language-model hyperparameters live
//! under `text_config`; quantization defaults under top-level `quantization`.
const std = @import("std");

pub const LoadError = error{
    InvalidConfig,
    UnsupportedArchitecture,
};

/// Vision-tower hyperparameters from top-level `vision_config`. Field names
/// match what the Task-3 `Qwen38VitConfig` consumes; mapping from the JSON
/// keys: depth->depth, hidden->hidden_size, heads->num_heads,
/// intermediate->intermediate_size, patch->patch_size,
/// temporal_patch->temporal_patch_size, merge->spatial_merge_size,
/// out_hidden->out_hidden_size, num_pos->num_position_embeddings,
/// in_ch->in_channels.
pub const Qwen38VisionConfig = struct {
    depth: u32 = 27,
    hidden: u32 = 1152,
    heads: u32 = 16,
    intermediate: u32 = 4304,
    patch: u32 = 16,
    temporal_patch: u32 = 2,
    merge: u32 = 2,
    out_hidden: u32 = 5120,
    num_pos: u32 = 2304,
    in_ch: u32 = 3,
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
    vision: Qwen38VisionConfig = .{},
    image_token_id: u32 = 248056,
    video_token_id: u32 = 248057,
    vision_start_token_id: u32 = 248053,
    vision_end_token_id: u32 = 248054,
    mrope_section: [3]u32 = .{ 11, 11, 10 },
    mrope_interleaved: bool = true,

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

    // rope_theta + MRoPE layout live under text_config.rope_parameters.
    if (text) |t| {
        if (t.get("rope_parameters")) |rp| {
            if (rp == .object) {
                const rpo = rp.object;
                cfg.rope_theta = toF32(getFloat(rpo, "rope_theta"), cfg.rope_theta);
                if (getBool(rpo, "mrope_interleaved")) |b| cfg.mrope_interleaved = b;
                if (rpo.get("mrope_section")) |ms| {
                    if (ms != .array) return LoadError.InvalidConfig;
                    if (ms.array.items.len != 3) return LoadError.InvalidConfig;
                    var section: [3]u32 = undefined;
                    for (ms.array.items, 0..) |item, i| {
                        const n: i64 = switch (item) {
                            .integer => |v| v,
                            .float => |f| @intFromFloat(f),
                            else => return LoadError.InvalidConfig,
                        };
                        section[i] = std.math.cast(u32, n) orelse return LoadError.InvalidConfig;
                    }
                    cfg.mrope_section = section;
                }
            }
        }
    }

    // Vision tower config (top-level `vision_config`).
    if (obj.get("vision_config")) |vc| {
        if (vc == .object) {
            const v = vc.object;
            cfg.vision.depth = toU32(getInt(v, "depth"), cfg.vision.depth);
            cfg.vision.hidden = toU32(getInt(v, "hidden_size"), cfg.vision.hidden);
            cfg.vision.heads = toU32(getInt(v, "num_heads"), cfg.vision.heads);
            cfg.vision.intermediate = toU32(getInt(v, "intermediate_size"), cfg.vision.intermediate);
            cfg.vision.patch = toU32(getInt(v, "patch_size"), cfg.vision.patch);
            cfg.vision.temporal_patch = toU32(getInt(v, "temporal_patch_size"), cfg.vision.temporal_patch);
            cfg.vision.merge = toU32(getInt(v, "spatial_merge_size"), cfg.vision.merge);
            cfg.vision.out_hidden = toU32(getInt(v, "out_hidden_size"), cfg.vision.out_hidden);
            cfg.vision.num_pos = toU32(getInt(v, "num_position_embeddings"), cfg.vision.num_pos);
            cfg.vision.in_ch = toU32(getInt(v, "in_channels"), cfg.vision.in_ch);
        }
    }

    // Special token ids for image/video spans.
    cfg.image_token_id = toU32(getInt(obj, "image_token_id"), cfg.image_token_id);
    cfg.video_token_id = toU32(getInt(obj, "video_token_id"), cfg.video_token_id);
    cfg.vision_start_token_id = toU32(getInt(obj, "vision_start_token_id"), cfg.vision_start_token_id);
    cfg.vision_end_token_id = toU32(getInt(obj, "vision_end_token_id"), cfg.vision_end_token_id);

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

test "parseQwen38Config: vision_config + mrope rope_parameters" {
    const allocator = std.testing.allocator;
    const json_text =
        \\{"model_type":"qwen3_5",
        \\  "image_token_id":248056,"video_token_id":248057,
        \\  "vision_start_token_id":248053,"vision_end_token_id":248054,
        \\  "vision_config":{
        \\    "deepstack_visual_indexes":[],"depth":27,
        \\    "hidden_act":"gelu_pytorch_tanh","hidden_size":1152,
        \\    "in_channels":3,"initializer_range":0.02,
        \\    "intermediate_size":4304,"model_type":"qwen3_5",
        \\    "num_heads":16,"num_position_embeddings":2304,
        \\    "out_hidden_size":5120,"patch_size":16,
        \\    "spatial_merge_size":2,"temporal_patch_size":2
        \\  },
        \\  "text_config":{
        \\    "rope_parameters":{
        \\      "mrope_interleaved":true,"mrope_section":[11,11,10],
        \\      "partial_rotary_factor":0.25,"rope_theta":10000000,
        \\      "rope_type":"default"
        \\    }
        \\  }}
    ;
    var cfg = try parseQwen38Config(allocator, json_text);
    defer cfg.deinit();
    try std.testing.expectEqual(@as(u32, 27), cfg.vision.depth);
    try std.testing.expectEqual(@as(u32, 1152), cfg.vision.hidden);
    try std.testing.expectEqual(@as(u32, 16), cfg.vision.heads);
    try std.testing.expectEqual(@as(u32, 4304), cfg.vision.intermediate);
    try std.testing.expectEqual(@as(u32, 16), cfg.vision.patch);
    try std.testing.expectEqual(@as(u32, 2), cfg.vision.temporal_patch);
    try std.testing.expectEqual(@as(u32, 2), cfg.vision.merge);
    try std.testing.expectEqual(@as(u32, 5120), cfg.vision.out_hidden);
    try std.testing.expectEqual(@as(u32, 2304), cfg.vision.num_pos);
    try std.testing.expectEqual(@as(u32, 3), cfg.vision.in_ch);
    try std.testing.expectEqualSlices(u32, &.{ 11, 11, 10 }, &cfg.mrope_section);
    try std.testing.expect(cfg.mrope_interleaved);
    try std.testing.expectEqual(@as(u32, 248056), cfg.image_token_id);
    try std.testing.expectEqual(@as(u32, 248057), cfg.video_token_id);
    try std.testing.expectEqual(@as(u32, 248053), cfg.vision_start_token_id);
    try std.testing.expectEqual(@as(u32, 248054), cfg.vision_end_token_id);
    try std.testing.expectEqual(@as(f32, 1e7), cfg.rope_theta);
}

test "parseQwen38Config: vision_config + mrope parse non-default values" {
    const allocator = std.testing.allocator;
    const json_text =
        \\{"model_type":"qwen3_5",
        \\  "image_token_id":999,"video_token_id":998,
        \\  "vision_start_token_id":997,"vision_end_token_id":996,
        \\  "vision_config":{
        \\    "depth":12,"hidden_size":64,"num_heads":4,
        \\    "intermediate_size":128,"patch_size":14,
        \\    "temporal_patch_size":1,"spatial_merge_size":4,
        \\    "out_hidden_size":256,"num_position_embeddings":100,
        \\    "in_channels":1
        \\  },
        \\  "text_config":{
        \\    "rope_parameters":{
        \\      "mrope_interleaved":false,"mrope_section":[8,8,16],
        \\      "rope_theta":500000
        \\    }
        \\  }}
    ;
    var cfg = try parseQwen38Config(allocator, json_text);
    defer cfg.deinit();
    try std.testing.expectEqual(@as(u32, 12), cfg.vision.depth);
    try std.testing.expectEqual(@as(u32, 64), cfg.vision.hidden);
    try std.testing.expectEqual(@as(u32, 4), cfg.vision.heads);
    try std.testing.expectEqual(@as(u32, 128), cfg.vision.intermediate);
    try std.testing.expectEqual(@as(u32, 14), cfg.vision.patch);
    try std.testing.expectEqual(@as(u32, 1), cfg.vision.temporal_patch);
    try std.testing.expectEqual(@as(u32, 4), cfg.vision.merge);
    try std.testing.expectEqual(@as(u32, 256), cfg.vision.out_hidden);
    try std.testing.expectEqual(@as(u32, 100), cfg.vision.num_pos);
    try std.testing.expectEqual(@as(u32, 1), cfg.vision.in_ch);
    try std.testing.expectEqualSlices(u32, &.{ 8, 8, 16 }, &cfg.mrope_section);
    try std.testing.expect(!cfg.mrope_interleaved);
    try std.testing.expectEqual(@as(u32, 999), cfg.image_token_id);
    try std.testing.expectEqual(@as(u32, 998), cfg.video_token_id);
    try std.testing.expectEqual(@as(u32, 997), cfg.vision_start_token_id);
    try std.testing.expectEqual(@as(u32, 996), cfg.vision_end_token_id);
    try std.testing.expectEqual(@as(f32, 5e5), cfg.rope_theta);
}

test "parseQwen38Config: wrong-length mrope_section is rejected" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(
        LoadError.InvalidConfig,
        parseQwen38Config(allocator, "{\"model_type\":\"qwen3_5\",\"text_config\":{\"rope_parameters\":{\"mrope_section\":[11,11]}}}"),
    );
    try std.testing.expectError(
        LoadError.InvalidConfig,
        parseQwen38Config(allocator, "{\"model_type\":\"qwen3_5\",\"text_config\":{\"rope_parameters\":{\"mrope_section\":\"11,11,10\"}}}"),
    );
}

test "parseQwen38Config: vision defaults when vision_config absent" {
    const allocator = std.testing.allocator;
    var cfg = try parseQwen38Config(allocator, "{\"model_type\":\"qwen3_5\",\"text_config\":{}}");
    defer cfg.deinit();
    try std.testing.expectEqual(@as(u32, 27), cfg.vision.depth);
    try std.testing.expectEqual(@as(u32, 1152), cfg.vision.hidden);
    try std.testing.expectEqualSlices(u32, &.{ 11, 11, 10 }, &cfg.mrope_section);
    try std.testing.expect(cfg.mrope_interleaved);
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
