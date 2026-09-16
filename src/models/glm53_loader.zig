//! GLM-5.3 config.json parsing (loader step 1).
//!
//! GLM-5.3 nests the language-model hyperparameters under `text_config` and
//! the linear-attention (KDA) block under `text_config.linear_attn_config`.
//! Missing keys fall back to the top level, then to GLM-5.3-Flash defaults
//! verified against the real config.json.

const std = @import("std");

pub const LoadError = error{
    InvalidConfig,
    UnsupportedArchitecture,
};

pub const Glm53Config = struct {
    allocator: std.mem.Allocator,
    hidden_size: u32 = 4096,
    num_hidden_layers: u32 = 45,
    vocab_size: u32 = 154880,
    first_k_dense_replace: u32 = 3,
    hc_mult: u32 = 4,
    hc_sinkhorn_iters: u32 = 20,
    hc_eps: f32 = 1e-6,
    kv_lora_rank: u32 = 512,
    q_lora_rank: u32 = 1536,
    num_attention_heads: u32 = 64,
    qk_nope_head_dim: u32 = 256,
    v_head_dim: u32 = 256,
    intermediate_size: u32 = 12288,
    moe_intermediate_size: u32 = 2048,
    n_routed_experts: u32 = 288,
    num_experts_per_tok: u32 = 8,
    n_shared_experts: u32 = 1,
    norm_topk_prob: bool = true,
    routed_scaling_factor: f32 = 2.5,
    swiglu_limit: f32 = 10.0,
    rms_norm_eps: f32 = 1e-5,
    index_n_heads: u32 = 32,
    index_head_dim: u32 = 128,
    index_topk: u32 = 2048,
    index_kpool: u32 = 4,
    index_kpool_always_select_tail: bool = true,
    index_kpool_compress: bool = true,
    linear_num_heads: u32 = 64,
    linear_head_dim: u32 = 128,
    linear_conv_kernel: u32 = 4,
    linear_gate_lower_bound: f32 = -5.0,
    eos_token_ids: []u32 = &.{},
    full_attn_layers: []u32 = &.{},
    quant_bits_default: u8 = 4,
    quant_group_size: u32 = 64,

    pub fn deinit(self: *Glm53Config) void {
        if (self.eos_token_ids.len > 0) self.allocator.free(self.eos_token_ids);
        if (self.full_attn_layers.len > 0) self.allocator.free(self.full_attn_layers);
    }

    pub fn isFullAttention(self: *const Glm53Config, layer: u32) bool {
        for (self.full_attn_layers) |l| {
            if (l == layer) return true;
        }
        return false;
    }

    pub fn isMoE(self: *const Glm53Config, layer: u32) bool {
        return layer >= self.first_k_dense_replace;
    }
};

const getInt = struct {
    fn call(o: std.json.ObjectMap, key: []const u8) ?i64 {
        const val = o.get(key) orelse return null;
        switch (val) {
            .integer => |n| return n,
            .float => |f| return @intFromFloat(f),
            else => return null,
        }
    }
}.call;

const getFloat = struct {
    fn call(o: std.json.ObjectMap, key: []const u8) ?f64 {
        const val = o.get(key) orelse return null;
        switch (val) {
            .float => |f| return f,
            .integer => |n| return @floatFromInt(n),
            else => return null,
        }
    }
}.call;

const getBool = struct {
    fn call(o: std.json.ObjectMap, key: []const u8) ?bool {
        const val = o.get(key) orelse return null;
        switch (val) {
            .bool => |b| return b,
            else => return null,
        }
    }
}.call;

/// Look up a key in `text_config` first, falling back to the top-level object.
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

fn nestedBool(text: ?std.json.ObjectMap, top: std.json.ObjectMap, key: []const u8) ?bool {
    if (text) |t| {
        if (getBool(t, key)) |v| return v;
    }
    return getBool(top, key);
}

fn toU32(v: ?i64, default: u32) u32 {
    const n = v orelse return default;
    if (n < 0) return default;
    return @intCast(n);
}

fn toF32(v: ?f64, default: f32) f32 {
    const f = v orelse return default;
    return @floatCast(f);
}

/// Parse GLM-5.3 config.json into Glm53Config.
/// Caller must call `cfg.deinit()` to free owned slices.
pub fn parseGlm53Config(allocator: std.mem.Allocator, json_text: []const u8) !Glm53Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return LoadError.InvalidConfig;
    const obj = root.object;

    // Model type check: accept glm5_next family, or absent model_type.
    if (obj.get("model_type")) |mt| {
        if (mt == .string and !std.mem.startsWith(u8, mt.string, "glm5")) {
            return LoadError.UnsupportedArchitecture;
        }
    }

    const text: ?std.json.ObjectMap = blk: {
        const tv = obj.get("text_config") orelse break :blk null;
        if (tv != .object) break :blk null;
        break :blk tv.object;
    };

    var cfg = Glm53Config{ .allocator = allocator };
    errdefer cfg.deinit();

    cfg.hidden_size = toU32(nestedInt(text, obj, "hidden_size"), cfg.hidden_size);
    cfg.num_hidden_layers = toU32(nestedInt(text, obj, "num_hidden_layers"), cfg.num_hidden_layers);
    cfg.vocab_size = toU32(nestedInt(text, obj, "vocab_size"), cfg.vocab_size);
    cfg.first_k_dense_replace = toU32(nestedInt(text, obj, "first_k_dense_replace"), cfg.first_k_dense_replace);
    cfg.hc_mult = toU32(nestedInt(text, obj, "hc_mult"), cfg.hc_mult);
    cfg.hc_sinkhorn_iters = toU32(nestedInt(text, obj, "hc_sinkhorn_iters"), cfg.hc_sinkhorn_iters);
    cfg.hc_eps = toF32(nestedFloat(text, obj, "hc_eps"), cfg.hc_eps);
    cfg.kv_lora_rank = toU32(nestedInt(text, obj, "kv_lora_rank"), cfg.kv_lora_rank);
    cfg.q_lora_rank = toU32(nestedInt(text, obj, "q_lora_rank"), cfg.q_lora_rank);
    cfg.num_attention_heads = toU32(nestedInt(text, obj, "num_attention_heads"), cfg.num_attention_heads);
    cfg.qk_nope_head_dim = toU32(nestedInt(text, obj, "qk_nope_head_dim"), cfg.qk_nope_head_dim);
    cfg.v_head_dim = toU32(nestedInt(text, obj, "v_head_dim"), cfg.v_head_dim);
    cfg.intermediate_size = toU32(nestedInt(text, obj, "intermediate_size"), cfg.intermediate_size);
    cfg.moe_intermediate_size = toU32(nestedInt(text, obj, "moe_intermediate_size"), cfg.moe_intermediate_size);
    cfg.n_routed_experts = toU32(nestedInt(text, obj, "n_routed_experts"), cfg.n_routed_experts);
    cfg.num_experts_per_tok = toU32(nestedInt(text, obj, "num_experts_per_tok"), cfg.num_experts_per_tok);
    cfg.n_shared_experts = toU32(nestedInt(text, obj, "n_shared_experts"), cfg.n_shared_experts);
    cfg.norm_topk_prob = nestedBool(text, obj, "norm_topk_prob") orelse cfg.norm_topk_prob;
    cfg.routed_scaling_factor = toF32(nestedFloat(text, obj, "routed_scaling_factor"), cfg.routed_scaling_factor);
    cfg.swiglu_limit = toF32(nestedFloat(text, obj, "swiglu_limit"), cfg.swiglu_limit);
    cfg.rms_norm_eps = toF32(nestedFloat(text, obj, "rms_norm_eps"), cfg.rms_norm_eps);
    cfg.index_n_heads = toU32(nestedInt(text, obj, "index_n_heads"), cfg.index_n_heads);
    cfg.index_head_dim = toU32(nestedInt(text, obj, "index_head_dim"), cfg.index_head_dim);
    cfg.index_topk = toU32(nestedInt(text, obj, "index_topk"), cfg.index_topk);
    cfg.index_kpool = toU32(nestedInt(text, obj, "index_kpool"), cfg.index_kpool);
    cfg.index_kpool_always_select_tail = nestedBool(text, obj, "index_kpool_always_select_tail") orelse cfg.index_kpool_always_select_tail;
    cfg.index_kpool_compress = nestedBool(text, obj, "index_kpool_compress") orelse cfg.index_kpool_compress;

    // eos_token_id: integer or array of integers.
    if (text) |t| {
        if (t.get("eos_token_id")) |eos_val| {
            switch (eos_val) {
                .integer => |n| {
                    if (n >= 0) {
                        const ids = try allocator.alloc(u32, 1);
                        ids[0] = @intCast(n);
                        cfg.eos_token_ids = ids;
                    }
                },
                .array => |arr| {
                    var list = std.ArrayList(u32).empty;
                    defer list.deinit(allocator);
                    for (arr.items) |item| {
                        const n: ?i64 = switch (item) {
                            .integer => |v| v,
                            .float => |f| @intFromFloat(f),
                            else => null,
                        };
                        if (n) |v| {
                            if (v >= 0) try list.append(allocator, @intCast(v));
                        }
                    }
                    if (list.items.len > 0) {
                        cfg.eos_token_ids = try list.toOwnedSlice(allocator);
                    }
                },
                else => {},
            }
        }
    }

    // linear_attn_config sub-object: linear_* fields and full_attn_layers.
    if (text) |t| {
        if (t.get("linear_attn_config")) |la_val| {
            if (la_val == .object) {
                const la = la_val.object;
                cfg.linear_num_heads = toU32(getInt(la, "num_heads"), cfg.linear_num_heads);
                cfg.linear_head_dim = toU32(getInt(la, "head_dim"), cfg.linear_head_dim);
                cfg.linear_conv_kernel = toU32(getInt(la, "short_conv_kernel_size"), cfg.linear_conv_kernel);
                cfg.linear_gate_lower_bound = toF32(getFloat(la, "gate_lower_bound"), cfg.linear_gate_lower_bound);

                if (la.get("full_attn_layers")) |fa_val| {
                    if (fa_val == .array) {
                        var list = std.ArrayList(u32).empty;
                        defer list.deinit(allocator);
                        for (fa_val.array.items) |item| {
                            const n: ?i64 = switch (item) {
                                .integer => |v| v,
                                .float => |f| @intFromFloat(f),
                                else => null,
                            };
                            if (n) |v| {
                                if (v >= 0) try list.append(allocator, @intCast(v));
                            }
                        }
                        if (list.items.len > 0) {
                            cfg.full_attn_layers = try list.toOwnedSlice(allocator);
                        }
                    }
                }
            }
        }
    }

    // Top-level quantization defaults (per-tensor overrides ignored here).
    if (obj.get("quantization")) |q_val| {
        if (q_val == .object) {
            const q = q_val.object;
            cfg.quant_bits_default = @intCast(@min(255, @max(0, getInt(q, "bits") orelse cfg.quant_bits_default)));
            cfg.quant_group_size = toU32(getInt(q, "group_size"), cfg.quant_group_size);
        }
    }

    return cfg;
}

test "parseGlm53Config: text_config 嵌套 + linear_attn_config" {
    const allocator = std.testing.allocator;
    const json_text =
        \\{"model_type":"glm5_next","text_config":{
        \\  "hidden_size":4096,"num_hidden_layers":45,"vocab_size":154880,
        \\  "first_k_dense_replace":3,"hc_mult":4,"hc_sinkhorn_iters":20,
        \\  "kv_lora_rank":512,"intermediate_size":12288,"moe_intermediate_size":2048,
        \\  "n_routed_experts":288,"num_experts_per_tok":8,"n_shared_experts":1,
        \\  "norm_topk_prob":true,"routed_scaling_factor":2.5,"swiglu_limit":10.0,
        \\  "index_n_heads":32,"index_head_dim":128,"index_topk":2048,"index_kpool":4,
        \\  "index_kpool_always_select_tail":true,"index_kpool_compress":true,
        \\  "rms_norm_eps":1e-5,"eos_token_id":[154820,154827,154829],
        \\  "linear_attn_config":{"num_heads":64,"head_dim":128,
        \\    "short_conv_kernel_size":4,"gate_lower_bound":-5.0,
        \\    "full_attn_layers":[3,7,11]}
        \\}}
    ;
    var cfg = try parseGlm53Config(allocator, json_text);
    defer cfg.deinit();
    try std.testing.expectEqual(@as(u32, 45), cfg.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 288), cfg.n_routed_experts);
    try std.testing.expectEqual(@as(u32, 64), cfg.linear_num_heads);
    try std.testing.expect(cfg.isFullAttention(3));
    try std.testing.expect(!cfg.isFullAttention(4));
    try std.testing.expectEqual(@as(usize, 3), cfg.eos_token_ids.len);
    // Real-config values applied via defaults (absent from fixture).
    try std.testing.expectEqual(@as(u32, 1536), cfg.q_lora_rank);
    try std.testing.expectEqual(@as(u32, 64), cfg.num_attention_heads);
    try std.testing.expectEqual(@as(u32, 256), cfg.qk_nope_head_dim);
    try std.testing.expectEqual(@as(u32, 256), cfg.v_head_dim);
    try std.testing.expectApproxEqAbs(@as(f32, 1e-5), cfg.rms_norm_eps, 1e-9);
    try std.testing.expect(cfg.isMoE(3));
    try std.testing.expect(!cfg.isMoE(2));
}

test "parseGlm53Config: 缺省字段回退默认 + 顶层回退" {
    const allocator = std.testing.allocator;
    const json_text =
        \\{"model_type":"glm5_next","hidden_size":4096,"num_hidden_layers":45}
    ;
    var cfg = try parseGlm53Config(allocator, json_text);
    defer cfg.deinit();
    try std.testing.expectEqual(@as(u32, 45), cfg.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 154880), cfg.vocab_size);
    try std.testing.expectEqual(@as(usize, 0), cfg.eos_token_ids.len);
    try std.testing.expect(!cfg.isFullAttention(0));
    try std.testing.expectEqual(@as(u8, 4), cfg.quant_bits_default);
}

test "parseGlm53Config: 非 glm5 model_type 拒绝" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(
        LoadError.UnsupportedArchitecture,
        parseGlm53Config(allocator, "{\"model_type\":\"llama\"}"),
    );
}
