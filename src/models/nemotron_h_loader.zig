/// Nemotron-H model loader from HuggingFace Safetensors format.
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const ops = @import("../ops.zig");
const nn = @import("../ops/nn.zig");
const io = @import("../io/mlx_io.zig");
const nemotron_h = @import("nemotron_h.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const NemotronHConfig = nemotron_h.NemotronHConfig;
const NemotronHModel = nemotron_h.NemotronHModel;
const NemotronHLayer = nemotron_h.NemotronHLayer;
const LayerType = nemotron_h.LayerType;

// ============================================================
// Config parsing
// ============================================================

pub fn parseNemotronHConfig(allocator: std.mem.Allocator, json_text: []const u8) !NemotronHConfig {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    const root = parsed.value;
    const obj = root.object;

    const vocab_size: usize = @intCast(getInt(obj, "vocab_size") orelse 32000);
    const hidden_size: usize = @intCast(getInt(obj, "hidden_size") orelse 4096);
    const num_layers: usize = @intCast(getInt(obj, "num_hidden_layers") orelse 32);
    const num_attention_heads: usize = @intCast(getInt(obj, "num_attention_heads") orelse 32);
    const num_key_value_heads: usize = blk: {
        if (getInt(obj, "num_key_value_heads")) |n| {
            break :blk @intCast(n);
        }
        break :blk num_attention_heads;
    };
    const ssm_dim: usize = @intCast(getInt(obj, "ssm_dim") orelse 128);
    const conv_kernel_size: usize = @intCast(getInt(obj, "conv_kernel_size") orelse 4);
    const ssm_state_dim: usize = @intCast(getInt(obj, "ssm_state_dim") orelse 128);
    const intermediate_size: usize = @intCast(getInt(obj, "intermediate_size") orelse 11008);
    const rms_norm_eps: f32 = @floatCast(getFloat(obj, "rms_norm_eps") orelse 1e-5);
    const rope_theta: f32 = @floatCast(getFloat(obj, "rope_theta") orelse 10000.0);
    const max_position_embeddings: usize = @intCast(getInt(obj, "max_position_embeddings") orelse 4096);

    var ssm_pattern: []const u8 = "MA";
    if (obj.get("ssm_pattern")) |pattern_val| {
        if (pattern_val == .string) {
            ssm_pattern = try allocator.dupe(u8, pattern_val.string);
        }
    }

    return NemotronHConfig{
        .vocab_size = vocab_size,
        .hidden_size = hidden_size,
        .num_layers = num_layers,
        .num_attention_heads = num_attention_heads,
        .num_key_value_heads = num_key_value_heads,
        .ssm_dim = ssm_dim,
        .conv_kernel_size = conv_kernel_size,
        .ssm_state_dim = ssm_state_dim,
        .intermediate_size = intermediate_size,
        .rms_norm_eps = rms_norm_eps,
        .rope_theta = rope_theta,
        .max_position_embeddings = max_position_embeddings,
        .ssm_pattern = ssm_pattern,
    };
}

// ============================================================
// Weight loading
// ============================================================

pub fn loadNemotronHModel(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
) !NemotronHModel {
    const dir = std.fs.path.dirname(model_path) orelse ".";
    const config_path = try std.fs.path.join(allocator, &.{ dir, "config.json" });
    defer allocator.free(config_path);

    const config_text = try std.fs.cwd().readFileAlloc(allocator, config_path, 1024 * 1024);
    defer allocator.free(config_text);

    const config = try parseNemotronHConfig(allocator, config_text);

    var st = try io.loadSafetensors(allocator, model_path);
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

    var weights = std.StringHashMap(Array).init(allocator);
    errdefer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    var hf_it = st.weights.iterator();
    while (hf_it.next()) |entry| {
        const hf_name = entry.key_ptr.*;
        const mapped = try mapNemotronHWeightName(hf_name, allocator) orelse {
            std.log.warn("Unmapped weight: {s}", .{hf_name});
            continue;
        };
        const weight = entry.value_ptr.*;
        const f32_weight = if (weight.dtype() == .bfloat16)
            try ops.astype(ctx, weight, .float32)
        else
            weight;
        try weights.put(mapped, f32_weight);
    }

    return try buildNemotronHModel(allocator, &config, &weights, ctx, stream);
}

pub fn buildNemotronHModel(
    allocator: std.mem.Allocator,
    config: *const NemotronHConfig,
    weights: *std.StringHashMap(Array),
    ctx: EagerContext,
    stream: c.c.mlx_stream,
) !NemotronHModel {
    _ = stream;
    const hidden_size = config.hidden_size;
    const vocab_size = config.vocab_size;
    const num_layers = config.num_layers;
    const num_heads = config.num_attention_heads;
    const head_dim = config.getHeadDim();

    // Embedding
    const embed_weight = weights.get("embed_tokens.weight") orelse return error.MissingEmbedWeight;
    if (weights.fetchRemove("embed_tokens.weight")) |kv| allocator.free(kv.key);
    const embed = nn.Embedding{
        .ctx = ctx,
        .num_embeddings = vocab_size,
        .embedding_dim = hidden_size,
        .weight = embed_weight,
    };

    // Norm
    const norm_weight = weights.get("norm.weight") orelse return error.MissingNormWeight;
    if (weights.fetchRemove("norm.weight")) |kv| allocator.free(kv.key);
    var norm = try nn.RMSNorm.init(ctx, hidden_size, config.rms_norm_eps);
    norm.weight.deinit();
    norm.weight = norm_weight;

    // LM head
    const lm_head = weights.get("lm_head.weight") orelse return error.MissingLMHeadWeight;
    if (weights.fetchRemove("lm_head.weight")) |kv| allocator.free(kv.key);

    // Layers
    var layers = std.ArrayList(NemotronHLayer).empty;
    errdefer {
        for (layers.items) |*layer| {
            layer.deinit(allocator);
        }
        layers.deinit(allocator);
    }

    for (0..num_layers) |i| {
        const layer_type = config.getLayerType(i);
        const layer = try buildLayer(allocator, config, weights, i, layer_type, ctx, hidden_size, num_heads, head_dim);
        try layers.append(allocator, layer);
    }

    return NemotronHModel{
        .allocator = allocator,
        .ctx = ctx,
        .config = config.*,
        .embed_tokens = embed,
        .layers = layers,
        .norm = norm,
        .lm_head = lm_head,
    };
}

fn buildLayer(
    allocator: std.mem.Allocator,
    config: *const NemotronHConfig,
    weights: *std.StringHashMap(Array),
    layer_idx: usize,
    layer_type: LayerType,
    ctx: EagerContext,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
) !NemotronHLayer {
    _ = num_heads;
    const prefix = try std.fmt.allocPrint(allocator, "layers.{d}.", .{layer_idx});
    defer allocator.free(prefix);

    const input_norm_name = try std.fmt.allocPrint(allocator, "{s}input_norm.weight", .{prefix});
    defer allocator.free(input_norm_name);
    const input_norm_weight = weights.get(input_norm_name) orelse return error.MissingWeight;
    if (weights.fetchRemove(input_norm_name)) |kv| allocator.free(kv.key);
    var input_norm = try nn.RMSNorm.init(ctx, hidden_size, config.rms_norm_eps);
    input_norm.weight.deinit();
    input_norm.weight = input_norm_weight;

    const post_norm_name = try std.fmt.allocPrint(allocator, "{s}post_norm.weight", .{prefix});
    defer allocator.free(post_norm_name);
    const post_norm_weight = weights.get(post_norm_name) orelse return error.MissingWeight;
    if (weights.fetchRemove(post_norm_name)) |kv| allocator.free(kv.key);
    var post_norm = try nn.RMSNorm.init(ctx, hidden_size, config.rms_norm_eps);
    post_norm.weight.deinit();
    post_norm.weight = post_norm_weight;

    switch (layer_type) {
        .mamba => {
            const ssm_prefix = try std.fmt.allocPrint(allocator, "{s}ssm.", .{prefix});
            defer allocator.free(ssm_prefix);

            const in_proj_name = try std.fmt.allocPrint(allocator, "{s}in_proj", .{ssm_prefix});
            defer allocator.free(in_proj_name);
            const in_proj = weights.get(in_proj_name) orelse return error.MissingWeight;
            if (weights.fetchRemove(in_proj_name)) |kv| allocator.free(kv.key);

            const conv_name = try std.fmt.allocPrint(allocator, "{s}conv", .{ssm_prefix});
            defer allocator.free(conv_name);
            const conv_w = weights.get(conv_name) orelse return error.MissingWeight;
            if (weights.fetchRemove(conv_name)) |kv| allocator.free(kv.key);

            const out_proj_name = try std.fmt.allocPrint(allocator, "{s}out_proj", .{ssm_prefix});
            defer allocator.free(out_proj_name);
            const out_proj = weights.get(out_proj_name) orelse return error.MissingWeight;
            if (weights.fetchRemove(out_proj_name)) |kv| allocator.free(kv.key);

            const dt_proj_name = try std.fmt.allocPrint(allocator, "{s}dt_proj", .{ssm_prefix});
            defer allocator.free(dt_proj_name);
            const dt_proj = weights.get(dt_proj_name) orelse return error.MissingWeight;
            if (weights.fetchRemove(dt_proj_name)) |kv| allocator.free(kv.key);

            const a_name = try std.fmt.allocPrint(allocator, "{s}A", .{ssm_prefix});
            defer allocator.free(a_name);
            const A = weights.get(a_name) orelse return error.MissingWeight;
            if (weights.fetchRemove(a_name)) |kv| allocator.free(kv.key);

            const d_name = try std.fmt.allocPrint(allocator, "{s}D", .{ssm_prefix});
            defer allocator.free(d_name);
            const D = weights.get(d_name) orelse return error.MissingWeight;
            if (weights.fetchRemove(d_name)) |kv| allocator.free(kv.key);

            const block = nemotron_h.MambaBlock{
                .ctx = ctx,
                .config = config,
                .in_proj = in_proj,
                .conv_weight = conv_w,
                .out_proj = out_proj,
                .dt_proj = dt_proj,
                .A = A,
                .D = D,
            };

            return NemotronHLayer{ .mamba = .{
                .ctx = ctx,
                .input_norm = input_norm,
                .block = block,
                .post_norm = post_norm,
            } };
        },
        .gated_delta => {
            const ssm_prefix = try std.fmt.allocPrint(allocator, "{s}ssm.", .{prefix});
            defer allocator.free(ssm_prefix);

            const gate_proj_name = try std.fmt.allocPrint(allocator, "{s}gate_proj", .{ssm_prefix});
            defer allocator.free(gate_proj_name);
            const gate_proj = weights.get(gate_proj_name) orelse return error.MissingWeight;
            if (weights.fetchRemove(gate_proj_name)) |kv| allocator.free(kv.key);

            const x_proj_name = try std.fmt.allocPrint(allocator, "{s}x_proj", .{ssm_prefix});
            defer allocator.free(x_proj_name);
            const x_proj = weights.get(x_proj_name) orelse return error.MissingWeight;
            if (weights.fetchRemove(x_proj_name)) |kv| allocator.free(kv.key);

            const delta_proj_name = try std.fmt.allocPrint(allocator, "{s}delta_proj", .{ssm_prefix});
            defer allocator.free(delta_proj_name);
            const delta_proj = weights.get(delta_proj_name) orelse return error.MissingWeight;
            if (weights.fetchRemove(delta_proj_name)) |kv| allocator.free(kv.key);

            const out_proj_name = try std.fmt.allocPrint(allocator, "{s}out_proj", .{ssm_prefix});
            defer allocator.free(out_proj_name);
            const out_proj = weights.get(out_proj_name) orelse return error.MissingWeight;
            if (weights.fetchRemove(out_proj_name)) |kv| allocator.free(kv.key);

            const block = nemotron_h.GatedDeltaNetBlock{
                .ctx = ctx,
                .config = config,
                .gate_proj = gate_proj,
                .x_proj = x_proj,
                .delta_proj = delta_proj,
                .out_proj = out_proj,
            };

            return NemotronHLayer{ .gated_delta = .{
                .ctx = ctx,
                .input_norm = input_norm,
                .block = block,
                .post_norm = post_norm,
            } };
        },
        .attention => {
            const attn_prefix = try std.fmt.allocPrint(allocator, "{s}self_attn.", .{prefix});
            defer allocator.free(attn_prefix);

            const wq_name = try std.fmt.allocPrint(allocator, "{s}wq", .{attn_prefix});
            defer allocator.free(wq_name);
            const wq = weights.get(wq_name) orelse return error.MissingWeight;
            if (weights.fetchRemove(wq_name)) |kv| allocator.free(kv.key);

            const wk_name = try std.fmt.allocPrint(allocator, "{s}wk", .{attn_prefix});
            defer allocator.free(wk_name);
            const wk = weights.get(wk_name) orelse return error.MissingWeight;
            if (weights.fetchRemove(wk_name)) |kv| allocator.free(kv.key);

            const wv_name = try std.fmt.allocPrint(allocator, "{s}wv", .{attn_prefix});
            defer allocator.free(wv_name);
            const wv = weights.get(wv_name) orelse return error.MissingWeight;
            if (weights.fetchRemove(wv_name)) |kv| allocator.free(kv.key);

            const wo_name = try std.fmt.allocPrint(allocator, "{s}wo", .{attn_prefix});
            defer allocator.free(wo_name);
            const wo = weights.get(wo_name) orelse return error.MissingWeight;
            if (weights.fetchRemove(wo_name)) |kv| allocator.free(kv.key);

            const rope = try nn.RoPE.init(ctx, head_dim, config.max_position_embeddings, config.rope_theta);
            const attn = nemotron_h.NemotronHAttention{
                .ctx = ctx,
                .config = config,
                .wq = wq,
                .wk = wk,
                .wv = wv,
                .wo = wo,
                .rope = rope,
            };

            return NemotronHLayer{ .attention = .{
                .ctx = ctx,
                .input_norm = input_norm,
                .attn = attn,
                .post_norm = post_norm,
            } };
        },
    }
}

fn getInt(obj: std.json.ObjectMap, key: []const u8) ?i64 {
    const val = obj.get(key) orelse return null;
    switch (val) {
        .integer => |n| return n,
        .float => |f| return @intFromFloat(f),
        else => return null,
    }
}

fn getFloat(obj: std.json.ObjectMap, key: []const u8) ?f64 {
    const val = obj.get(key) orelse return null;
    switch (val) {
        .float => |f| return f,
        .integer => |n| return @floatFromInt(n),
        else => return null,
    }
}

pub fn mapNemotronHWeightName(hf_name: []const u8, allocator: std.mem.Allocator) !?[]const u8 {
    if (std.mem.eql(u8, hf_name, "model.embed_tokens.weight")) {
        return try allocator.dupe(u8, "embed_tokens.weight");
    }
    if (std.mem.eql(u8, hf_name, "model.norm.weight")) {
        return try allocator.dupe(u8, "norm.weight");
    }
    if (std.mem.eql(u8, hf_name, "lm_head.weight")) {
        return try allocator.dupe(u8, "lm_head.weight");
    }

    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, hf_name, prefix)) return null;

    const after_prefix = hf_name[prefix.len..];
    const dot_idx = std.mem.indexOf(u8, after_prefix, ".") orelse return null;
    const layer_idx = after_prefix[0..dot_idx];
    const rest = after_prefix[dot_idx + 1 ..];

    const mapped = mapLayerComponent(rest) orelse return null;
    return try std.fmt.allocPrint(allocator, "layers.{s}.{s}", .{ layer_idx, mapped });
}

fn mapLayerComponent(component: []const u8) ?[]const u8 {
    const map = std.StaticStringMap([]const u8).initComptime(.{
        .{ "ssm.in_proj.weight", "ssm.in_proj" },
        .{ "ssm.conv.weight", "ssm.conv" },
        .{ "ssm.out_proj.weight", "ssm.out_proj" },
        .{ "ssm.dt_proj.weight", "ssm.dt_proj" },
        .{ "ssm.A", "ssm.A" },
        .{ "ssm.D", "ssm.D" },
        .{ "ssm.gate_proj.weight", "ssm.gate_proj" },
        .{ "ssm.x_proj.weight", "ssm.x_proj" },
        .{ "ssm.delta_proj.weight", "ssm.delta_proj" },
        .{ "self_attn.q_proj.weight", "self_attn.wq" },
        .{ "self_attn.k_proj.weight", "self_attn.wk" },
        .{ "self_attn.v_proj.weight", "self_attn.wv" },
        .{ "self_attn.o_proj.weight", "self_attn.wo" },
        .{ "input_norm.weight", "input_norm.weight" },
        .{ "post_norm.weight", "post_norm.weight" },
    });
    return map.get(component);
}

// ============================================================
// Tests
// ============================================================

test "parseNemotronHConfig" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "vocab_size": 32000,
        \\  "hidden_size": 4096,
        \\  "num_hidden_layers": 32,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 8,
        \\  "ssm_dim": 128,
        \\  "conv_kernel_size": 4,
        \\  "ssm_state_dim": 128,
        \\  "intermediate_size": 11008,
        \\  "rms_norm_eps": 1e-5,
        \\  "rope_theta": 10000.0,
        \\  "max_position_embeddings": 4096,
        \\  "ssm_pattern": "MGA"
        \\}
    ;

    const config = try parseNemotronHConfig(allocator, json);
    defer allocator.free(config.ssm_pattern);

    try std.testing.expectEqual(@as(usize, 32000), config.vocab_size);
    try std.testing.expectEqual(@as(usize, 4096), config.hidden_size);
    try std.testing.expectEqual(@as(usize, 32), config.num_layers);
    try std.testing.expectEqual(@as(usize, 128), config.ssm_dim);
    try std.testing.expectEqual(@as(usize, 4), config.conv_kernel_size);
    try std.testing.expect(std.mem.eql(u8, "MGA", config.ssm_pattern));
    try std.testing.expectEqual(LayerType.mamba, config.getLayerType(0));
    try std.testing.expectEqual(LayerType.gated_delta, config.getLayerType(1));
    try std.testing.expectEqual(LayerType.attention, config.getLayerType(2));
}

test "buildNemotronHModel with random weights" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = NemotronHConfig{
        .vocab_size = 16,
        .hidden_size = 8,
        .num_layers = 2,
        .num_attention_heads = 2,
        .num_key_value_heads = 2,
        .ssm_dim = 4,
        .conv_kernel_size = 3,
        .ssm_state_dim = 4,
        .intermediate_size = 16,
        .rms_norm_eps = 1e-5,
        .ssm_pattern = "MA",
    };

    var weights = std.StringHashMap(Array).init(allocator);
    defer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    const hidden: i32 = @intCast(config.hidden_size);
    const vocab: i32 = @intCast(config.vocab_size);
    const ssm: i32 = @intCast(config.ssm_dim);
    const heads: i32 = @intCast(config.num_attention_heads);
    const head_dim = @divTrunc(hidden, heads);

    try weights.put(try allocator.dupe(u8, "embed_tokens.weight"), try array_mod.zeros(allocator, &[_]i32{ vocab, hidden }, .float32));
    try weights.put(try allocator.dupe(u8, "norm.weight"), try array_mod.ones(allocator, &[_]i32{hidden}, .float32));
    try weights.put(try allocator.dupe(u8, "lm_head.weight"), try array_mod.zeros(allocator, &[_]i32{ vocab, hidden }, .float32));

    // Layer 0: Mamba
    {
        const p = "layers.0.";
        try weights.put(try allocator.dupe(u8, p ++ "input_norm.weight"), try array_mod.ones(allocator, &[_]i32{hidden}, .float32));
        try weights.put(try allocator.dupe(u8, p ++ "ssm.in_proj"), try array_mod.zeros(allocator, &[_]i32{ 2 * ssm, hidden }, .float32));
        try weights.put(try allocator.dupe(u8, p ++ "ssm.conv"), try array_mod.zeros(allocator, &[_]i32{ ssm, @intCast(config.conv_kernel_size), 1 }, .float32));
        try weights.put(try allocator.dupe(u8, p ++ "ssm.out_proj"), try array_mod.zeros(allocator, &[_]i32{ hidden, ssm }, .float32));
        try weights.put(try allocator.dupe(u8, p ++ "ssm.dt_proj"), try array_mod.zeros(allocator, &[_]i32{ ssm, ssm }, .float32));
        try weights.put(try allocator.dupe(u8, p ++ "ssm.A"), try array_mod.zeros(allocator, &[_]i32{ssm}, .float32));
        try weights.put(try allocator.dupe(u8, p ++ "ssm.D"), try array_mod.zeros(allocator, &[_]i32{ssm}, .float32));
        try weights.put(try allocator.dupe(u8, p ++ "post_norm.weight"), try array_mod.ones(allocator, &[_]i32{hidden}, .float32));
    }

    // Layer 1: Attention
    {
        const p = "layers.1.";
        try weights.put(try allocator.dupe(u8, p ++ "input_norm.weight"), try array_mod.ones(allocator, &[_]i32{hidden}, .float32));
        try weights.put(try allocator.dupe(u8, p ++ "self_attn.wq"), try array_mod.zeros(allocator, &[_]i32{ heads * head_dim, hidden }, .float32));
        try weights.put(try allocator.dupe(u8, p ++ "self_attn.wk"), try array_mod.zeros(allocator, &[_]i32{ heads * head_dim, hidden }, .float32));
        try weights.put(try allocator.dupe(u8, p ++ "self_attn.wv"), try array_mod.zeros(allocator, &[_]i32{ heads * head_dim, hidden }, .float32));
        try weights.put(try allocator.dupe(u8, p ++ "self_attn.wo"), try array_mod.zeros(allocator, &[_]i32{ hidden, heads * head_dim }, .float32));
        try weights.put(try allocator.dupe(u8, p ++ "post_norm.weight"), try array_mod.ones(allocator, &[_]i32{hidden}, .float32));
    }

    var model = try buildNemotronHModel(allocator, &config, &weights, ctx, stream);
    defer model.deinit();

    const input_ids = try Array.fromData(allocator, u32, &[_]u32{ 1, 2, 3 }, &[_]i32{ 1, 3 });
    defer input_ids.deinit();

    const logits = try model.forward(input_ids, null, null);
    defer logits.deinit();

    const shape = logits.shape();
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 3), shape[1]);
    try std.testing.expectEqual(@as(i32, 16), shape[2]);
}
