/// MiniMax weight loader.
///
/// Supports loading from safetensors with HF-format weight names.
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const ops = @import("../ops.zig");
const nn = @import("../ops/nn.zig");
const io = @import("../io/mlx_io.zig");
const minimax = @import("minimax.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const MiniMaxConfig = minimax.MiniMaxConfig;
const MiniMaxModel = minimax.MiniMaxModel;

/// Error set for loading.
pub const LoadError = error{
    MissingIndexJson,
    MissingShardFile,
    MissingWeight,
    InvalidConfig,
    UnsupportedArchitecture,
};

/// Parse MiniMax config.json into MiniMaxConfig.
pub fn parseMiniMaxConfig(allocator: std.mem.Allocator, json_text: []const u8) !MiniMaxConfig {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    const root = parsed.value;
    const obj = root.object;

    // Architecture check
    const archs = obj.get("architectures") orelse return LoadError.InvalidConfig;
    const arch = archs.array.items[0].string;
    if (!std.mem.startsWith(u8, arch, "MiniMax")) {
        return LoadError.UnsupportedArchitecture;
    }

    // Helper to extract integer
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

    // Helper to extract float
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

    return MiniMaxConfig{
        .vocab_size = @intCast(getInt(obj, "vocab_size") orelse 32000),
        .hidden_size = @intCast(getInt(obj, "hidden_size") orelse 4096),
        .num_hidden_layers = @intCast(getInt(obj, "num_hidden_layers") orelse 32),
        .num_attention_heads = @intCast(getInt(obj, "num_attention_heads") orelse 32),
        .num_key_value_heads = @intCast(getInt(obj, "num_key_value_heads") orelse 8),
        .head_dim = @intCast(getInt(obj, "head_dim") orelse 128),
        .intermediate_size = @intCast(getInt(obj, "intermediate_size") orelse 14336),
        .num_experts = @intCast(getInt(obj, "num_experts") orelse 8),
        .num_experts_per_tok = @intCast(getInt(obj, "num_experts_per_tok") orelse 2),
        .num_shared_experts = @intCast(getInt(obj, "num_shared_experts") orelse 1),
        .rms_norm_eps = @floatCast(getFloat(obj, "rms_norm_eps") orelse 1e-5),
        .rope_theta = @floatCast(getFloat(obj, "rope_theta") orelse 10000.0),
        .max_position_embeddings = @intCast(getInt(obj, "max_position_embeddings") orelse 4096),
    };
}

/// Load config from a directory path.
pub fn loadMiniMaxConfig(allocator: std.mem.Allocator, dir_path: []const u8) !MiniMaxConfig {
    const config_path = try std.fs.path.join(allocator, &.{ dir_path, "config.json" });
    defer allocator.free(config_path);

    const content = try std.fs.cwd().readFileAlloc(allocator, config_path, 1024 * 1024);
    defer allocator.free(content);

    return parseMiniMaxConfig(allocator, content);
}

/// Load weights from a model directory.
/// Automatically detects sharded vs single-file models.
pub fn loadMiniMaxWeights(
    allocator: std.mem.Allocator,
    dir_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
) !std.StringHashMap(Array) {
    _ = stream;

    // Try sharded loading first
    const index_path = try std.fs.path.join(allocator, &.{ dir_path, "model.safetensors.index.json" });
    defer allocator.free(index_path);

    const has_index = blk: {
        const file = std.fs.cwd().openFile(index_path, .{}) catch break :blk false;
        file.close();
        break :blk true;
    };

    if (has_index) {
        return try loadShardedWeights(allocator, dir_path, index_path, ctx);
    }

    // Fall back to single file
    const model_path = try std.fs.path.join(allocator, &.{ dir_path, "model.safetensors" });
    defer allocator.free(model_path);

    var st = try io.loadSafetensors(allocator, model_path);
    defer st.deinit(allocator);

    // Transfer ownership of weights into a new HashMap
    var weights = std.StringHashMap(Array).init(allocator);
    errdefer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    var it = st.weights.iterator();
    while (it.next()) |entry| {
        const key = try allocator.dupe(u8, entry.key_ptr.*);
        const weight = entry.value_ptr.*;
        const f32_weight = if (weight.dtype() == .bfloat16)
            try ops.astype(ctx, weight, .float32)
        else
            weight;
        try weights.put(key, f32_weight);
    }

    return weights;
}

/// Load sharded safetensors using index.json.
fn loadShardedWeights(
    allocator: std.mem.Allocator,
    dir_path: []const u8,
    index_path: []const u8,
    ctx: EagerContext,
) !std.StringHashMap(Array) {
    // Read index.json
    const index_content = try std.fs.cwd().readFileAlloc(allocator, index_path, 50 * 1024 * 1024);
    defer allocator.free(index_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, index_content, .{});
    defer parsed.deinit();

    const weight_map = parsed.value.object.get("weight_map") orelse return LoadError.MissingIndexJson;
    const wm_obj = weight_map.object;

    // Collect unique shard filenames
    var shard_set = std.StringHashMap(void).init(allocator);
    defer shard_set.deinit();

    var wm_it = wm_obj.iterator();
    while (wm_it.next()) |entry| {
        const shard_file = entry.value_ptr.*.string;
        if (!shard_set.contains(shard_file)) {
            const key = try allocator.dupe(u8, shard_file);
            try shard_set.put(key, {});
        }
    }

    // Result weights map
    var weights = std.StringHashMap(Array).init(allocator);
    errdefer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    // Load each shard and merge
    var shard_it = shard_set.keyIterator();
    while (shard_it.next()) |shard_file_ptr| {
        const shard_file = shard_file_ptr.*;
        const shard_path = try std.fs.path.join(allocator, &.{ dir_path, shard_file });
        defer allocator.free(shard_path);

        std.log.info("Loading shard: {s}", .{shard_file});

        var st = try io.loadSafetensors(allocator, shard_path);
        defer st.deinit(allocator);

        var w_it = st.weights.iterator();
        while (w_it.next()) |entry| {
            const key = try allocator.dupe(u8, entry.key_ptr.*);
            const weight = entry.value_ptr.*;
            const f32_weight = if (weight.dtype() == .bfloat16)
                try ops.astype(ctx, weight, .float32)
            else
                weight;
            try weights.put(key, f32_weight);
        }
    }

    // Free shard_set keys
    var ss_it = shard_set.keyIterator();
    while (ss_it.next()) |k| allocator.free(k.*);

    return weights;
}

/// Build MiniMaxModel from loaded weights.
pub fn buildMiniMaxModel(
    allocator: std.mem.Allocator,
    config: *const MiniMaxConfig,
    weights: *std.StringHashMap(Array),
    ctx: EagerContext,
    stream: c.c.mlx_stream,
) !MiniMaxModel {
    _ = stream;
    const num_layers = config.num_hidden_layers;
    const hidden_size = config.hidden_size;
    const vocab_size = config.vocab_size;
    const head_dim = config.head_dim;
    const num_experts = config.num_experts;
    const intermediate_size = config.intermediate_size;

    // === Embedding ===
    const embed_weight = weights.get("model.embed_tokens.weight") orelse return LoadError.MissingWeight;
    const embed = nn.Embedding{
        .ctx = ctx,
        .num_embeddings = vocab_size,
        .embedding_dim = hidden_size,
        .weight = embed_weight,
    };
    if (weights.fetchRemove("model.embed_tokens.weight")) |kv| allocator.free(kv.key);

    // === Final norm ===
    const norm_weight = weights.get("model.norm.weight") orelse return LoadError.MissingWeight;
    var norm = try nn.RMSNorm.init(ctx, hidden_size, config.rms_norm_eps);
    norm.weight.deinit();
    norm.weight = norm_weight;
    if (weights.fetchRemove("model.norm.weight")) |kv| allocator.free(kv.key);

    // === LM Head ===
    const lm_head_weight = weights.get("lm_head.weight") orelse weights.get("model.embed_tokens.weight") orelse return LoadError.MissingWeight;
    const lm_head = lm_head_weight;
    if (weights.fetchRemove("lm_head.weight")) |kv| allocator.free(kv.key);

    // === Layers ===
    const layers = try allocator.alloc(minimax.MiniMaxTransformerBlock, num_layers);
    errdefer allocator.free(layers);

    for (0..num_layers) |i| {
        const idx_fmt = try std.fmt.allocPrint(allocator, "model.layers.{d}.", .{i});
        defer allocator.free(idx_fmt);

        // --- Attention weights ---
        const wq_name = try std.fmt.allocPrint(allocator, "{s}self_attn.q_proj.weight", .{idx_fmt});
        defer allocator.free(wq_name);
        const wq = weights.get(wq_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(wq_name)) |kv| allocator.free(kv.key);

        const wk_name = try std.fmt.allocPrint(allocator, "{s}self_attn.k_proj.weight", .{idx_fmt});
        defer allocator.free(wk_name);
        const wk = weights.get(wk_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(wk_name)) |kv| allocator.free(kv.key);

        const wv_name = try std.fmt.allocPrint(allocator, "{s}self_attn.v_proj.weight", .{idx_fmt});
        defer allocator.free(wv_name);
        const wv = weights.get(wv_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(wv_name)) |kv| allocator.free(kv.key);

        const wo_name = try std.fmt.allocPrint(allocator, "{s}self_attn.o_proj.weight", .{idx_fmt});
        defer allocator.free(wo_name);
        const wo = weights.get(wo_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(wo_name)) |kv| allocator.free(kv.key);

        // RoPE
        const rope = try minimax.MiniMaxRoPE.init(ctx, head_dim, config.max_position_embeddings, config.rope_theta);

        const attention = minimax.MiniMaxAttention{
            .ctx = ctx,
            .config = config,
            .wq = wq,
            .wk = wk,
            .wv = wv,
            .wo = wo,
            .rope = rope,
        };

        // --- MoE weights ---
        const gate_weight_name = try std.fmt.allocPrint(allocator, "{s}mlp.gate.weight", .{idx_fmt});
        defer allocator.free(gate_weight_name);
        const gate_weight = weights.get(gate_weight_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(gate_weight_name)) |kv| allocator.free(kv.key);

        const gate = minimax.MiniMaxGate{
            .ctx = ctx,
            .weight = gate_weight,
            .topk = config.num_experts_per_tok,
            .num_experts = num_experts,
        };

        // Shared expert
        const shared_gate_name = try std.fmt.allocPrint(allocator, "{s}mlp.shared_expert.gate_proj.weight", .{idx_fmt});
        defer allocator.free(shared_gate_name);
        const shared_gate = weights.get(shared_gate_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(shared_gate_name)) |kv| allocator.free(kv.key);

        const shared_up_name = try std.fmt.allocPrint(allocator, "{s}mlp.shared_expert.up_proj.weight", .{idx_fmt});
        defer allocator.free(shared_up_name);
        const shared_up = weights.get(shared_up_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(shared_up_name)) |kv| allocator.free(kv.key);

        const shared_down_name = try std.fmt.allocPrint(allocator, "{s}mlp.shared_expert.down_proj.weight", .{idx_fmt});
        defer allocator.free(shared_down_name);
        const shared_down = weights.get(shared_down_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(shared_down_name)) |kv| allocator.free(kv.key);

        const shared_expert = minimax.MiniMaxExpert{
            .ctx = ctx,
            .gate_proj = shared_gate,
            .up_proj = shared_up,
            .down_proj = shared_down,
        };

        // Routed experts
        const experts = try allocator.alloc(minimax.MiniMaxExpert, num_experts);
        errdefer allocator.free(experts);

        for (0..num_experts) |e| {
            const egate_name = try std.fmt.allocPrint(allocator, "{s}mlp.experts.{d}.gate_proj.weight", .{ idx_fmt, e });
            defer allocator.free(egate_name);
            const egate = weights.get(egate_name);
            if (weights.fetchRemove(egate_name)) |kv| allocator.free(kv.key);

            const eup_name = try std.fmt.allocPrint(allocator, "{s}mlp.experts.{d}.up_proj.weight", .{ idx_fmt, e });
            defer allocator.free(eup_name);
            const eup = weights.get(eup_name);
            if (weights.fetchRemove(eup_name)) |kv| allocator.free(kv.key);

            const edown_name = try std.fmt.allocPrint(allocator, "{s}mlp.experts.{d}.down_proj.weight", .{ idx_fmt, e });
            defer allocator.free(edown_name);
            const edown = weights.get(edown_name);
            if (weights.fetchRemove(edown_name)) |kv| allocator.free(kv.key);

            const gate_actual = egate orelse try array_mod.zeros(allocator, &[_]i32{ @intCast(intermediate_size), @intCast(hidden_size) }, .float32);
            const up_actual = eup orelse try array_mod.zeros(allocator, &[_]i32{ @intCast(intermediate_size), @intCast(hidden_size) }, .float32);
            const down_actual = edown orelse try array_mod.zeros(allocator, &[_]i32{ @intCast(hidden_size), @intCast(intermediate_size) }, .float32);

            experts[e] = minimax.MiniMaxExpert{
                .ctx = ctx,
                .gate_proj = gate_actual,
                .up_proj = up_actual,
                .down_proj = down_actual,
            };
        }

        const moe = minimax.MiniMaxMoE{
            .ctx = ctx,
            .gate = gate,
            .experts = experts,
            .shared_expert = shared_expert,
            .num_experts = num_experts,
            .topk = config.num_experts_per_tok,
        };

        // --- Layer norms ---
        const attn_norm_name = try std.fmt.allocPrint(allocator, "{s}input_layernorm.weight", .{idx_fmt});
        defer allocator.free(attn_norm_name);
        const attn_norm_w = weights.get(attn_norm_name) orelse return LoadError.MissingWeight;
        var attn_norm = try nn.RMSNorm.init(ctx, hidden_size, config.rms_norm_eps);
        attn_norm.weight.deinit();
        attn_norm.weight = attn_norm_w;
        if (weights.fetchRemove(attn_norm_name)) |kv| allocator.free(kv.key);

        const ffn_norm_name = try std.fmt.allocPrint(allocator, "{s}post_attention_layernorm.weight", .{idx_fmt});
        defer allocator.free(ffn_norm_name);
        const ffn_norm_w = weights.get(ffn_norm_name) orelse return LoadError.MissingWeight;
        var ffn_norm = try nn.RMSNorm.init(ctx, hidden_size, config.rms_norm_eps);
        ffn_norm.weight.deinit();
        ffn_norm.weight = ffn_norm_w;
        if (weights.fetchRemove(ffn_norm_name)) |kv| allocator.free(kv.key);

        layers[i] = minimax.MiniMaxTransformerBlock{
            .ctx = ctx,
            .config = config,
            .layer_idx = i,
            .input_layernorm = attn_norm,
            .attention = attention,
            .post_attention_layernorm = ffn_norm,
            .moe = moe,
        };
    }

    // Check for unmapped weights
    var remaining = weights.iterator();
    while (remaining.next()) |entry| {
        if (std.mem.endsWith(u8, entry.key_ptr.*, ".scale")) continue;
        std.log.warn("Unused weight: {s}", .{entry.key_ptr.*});
    }

    return MiniMaxModel{
        .allocator = allocator,
        .ctx = ctx,
        .config = config.*,
        .embed_tokens = embed,
        .layers = layers,
        .norm = norm,
        .lm_head = lm_head,
    };
}

/// Load MiniMaxModel from a directory path (config + weights).
pub fn loadMiniMaxModel(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
) !MiniMaxModel {
    const config = try loadMiniMaxConfig(allocator, model_path);
    var weights = try loadMiniMaxWeights(allocator, model_path, ctx, stream);
    defer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    return try buildMiniMaxModel(allocator, &config, &weights, ctx, stream);
}

// ============================================================
// Tests
// ============================================================

test "parseMiniMaxConfig basic" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;

    const json =
        \\{
        \\  "architectures": ["MiniMaxForCausalLM"],
        \\  "vocab_size": 32000,
        \\  "hidden_size": 4096,
        \\  "num_hidden_layers": 32,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 8,
        \\  "head_dim": 128,
        \\  "intermediate_size": 14336,
        \\  "num_experts": 8,
        \\  "num_experts_per_tok": 2,
        \\  "num_shared_experts": 1,
        \\  "rms_norm_eps": 1e-5,
        \\  "rope_theta": 10000.0,
        \\  "max_position_embeddings": 4096
        \\}
    ;

    const config = try parseMiniMaxConfig(allocator, json);
    try std.testing.expectEqual(@as(usize, 32000), config.vocab_size);
    try std.testing.expectEqual(@as(usize, 4096), config.hidden_size);
    try std.testing.expectEqual(@as(usize, 32), config.num_hidden_layers);
    try std.testing.expectEqual(@as(usize, 32), config.num_attention_heads);
    try std.testing.expectEqual(@as(usize, 8), config.num_key_value_heads);
    try std.testing.expectEqual(@as(usize, 128), config.head_dim);
    try std.testing.expectEqual(@as(usize, 14336), config.intermediate_size);
    try std.testing.expectEqual(@as(usize, 8), config.num_experts);
    try std.testing.expectEqual(@as(usize, 2), config.num_experts_per_tok);
    try std.testing.expectEqual(@as(usize, 1), config.num_shared_experts);
    try std.testing.expectApproxEqAbs(@as(f32, 1e-5), config.rms_norm_eps, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10000.0), config.rope_theta, 1e-1);
    try std.testing.expectEqual(@as(usize, 4096), config.max_position_embeddings);
}

test "buildMiniMaxModel dummy" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const config = MiniMaxConfig{
        .vocab_size = 8,
        .hidden_size = 16,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 4,
        .intermediate_size = 32,
        .num_experts = 2,
        .num_experts_per_tok = 1,
        .num_shared_experts = 1,
        .max_position_embeddings = 32,
    };

    var weights = std.StringHashMap(Array).init(allocator);
    defer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
            allocator.free(entry.key_ptr.*);
        }
        weights.deinit();
    }

    const W = struct {
        fn put(w: *std.StringHashMap(Array), alloc: std.mem.Allocator, name: []const u8, shape: []const i32) !void {
            const key = try alloc.dupe(u8, name);
            const arr = try array_mod.zeros(alloc, shape, .float32);
            try w.put(key, arr);
        }
    };

    try W.put(&weights, allocator, "model.embed_tokens.weight", &[_]i32{ 8, 16 });
    try W.put(&weights, allocator, "model.norm.weight", &[_]i32{16});
    try W.put(&weights, allocator, "lm_head.weight", &[_]i32{ 8, 16 });

    for (0..2) |i| {
        const prefix = try std.fmt.allocPrint(allocator, "model.layers.{d}.", .{i});
        defer allocator.free(prefix);

        const q_name = try std.fmt.allocPrint(allocator, "{s}self_attn.q_proj.weight", .{prefix});
        defer allocator.free(q_name);
        try W.put(&weights, allocator, q_name, &[_]i32{ 16, 16 });

        const k_name = try std.fmt.allocPrint(allocator, "{s}self_attn.k_proj.weight", .{prefix});
        defer allocator.free(k_name);
        try W.put(&weights, allocator, k_name, &[_]i32{ 8, 16 });

        const v_name = try std.fmt.allocPrint(allocator, "{s}self_attn.v_proj.weight", .{prefix});
        defer allocator.free(v_name);
        try W.put(&weights, allocator, v_name, &[_]i32{ 8, 16 });

        const o_name = try std.fmt.allocPrint(allocator, "{s}self_attn.o_proj.weight", .{prefix});
        defer allocator.free(o_name);
        try W.put(&weights, allocator, o_name, &[_]i32{ 16, 16 });

        const gate_name = try std.fmt.allocPrint(allocator, "{s}mlp.gate.weight", .{prefix});
        defer allocator.free(gate_name);
        try W.put(&weights, allocator, gate_name, &[_]i32{ 2, 16 });

        const sg_name = try std.fmt.allocPrint(allocator, "{s}mlp.shared_expert.gate_proj.weight", .{prefix});
        defer allocator.free(sg_name);
        try W.put(&weights, allocator, sg_name, &[_]i32{ 32, 16 });

        const su_name = try std.fmt.allocPrint(allocator, "{s}mlp.shared_expert.up_proj.weight", .{prefix});
        defer allocator.free(su_name);
        try W.put(&weights, allocator, su_name, &[_]i32{ 32, 16 });

        const sd_name = try std.fmt.allocPrint(allocator, "{s}mlp.shared_expert.down_proj.weight", .{prefix});
        defer allocator.free(sd_name);
        try W.put(&weights, allocator, sd_name, &[_]i32{ 16, 32 });

        for (0..2) |e| {
            const eg_name = try std.fmt.allocPrint(allocator, "{s}mlp.experts.{d}.gate_proj.weight", .{ prefix, e });
            defer allocator.free(eg_name);
            try W.put(&weights, allocator, eg_name, &[_]i32{ 32, 16 });

            const eu_name = try std.fmt.allocPrint(allocator, "{s}mlp.experts.{d}.up_proj.weight", .{ prefix, e });
            defer allocator.free(eu_name);
            try W.put(&weights, allocator, eu_name, &[_]i32{ 32, 16 });

            const ed_name = try std.fmt.allocPrint(allocator, "{s}mlp.experts.{d}.down_proj.weight", .{ prefix, e });
            defer allocator.free(ed_name);
            try W.put(&weights, allocator, ed_name, &[_]i32{ 16, 32 });
        }

        const in_name = try std.fmt.allocPrint(allocator, "{s}input_layernorm.weight", .{prefix});
        defer allocator.free(in_name);
        try W.put(&weights, allocator, in_name, &[_]i32{16});

        const post_name = try std.fmt.allocPrint(allocator, "{s}post_attention_layernorm.weight", .{prefix});
        defer allocator.free(post_name);
        try W.put(&weights, allocator, post_name, &[_]i32{16});
    }

    var model = try buildMiniMaxModel(allocator, &config, &weights, ctx, ctx.stream.inner);
    defer model.deinit();

    const input_ids = try Array.fromData(allocator, u32, &[_]u32{ 1, 2 }, &[_]i32{ 1, 2 });
    defer input_ids.deinit();

    const logits = try model.forward(input_ids, null, null, 0, ctx.stream.inner);
    defer logits.deinit();

    const ls = logits.shape();
    try std.testing.expectEqual(@as(i32, 1), ls[0]);
    try std.testing.expectEqual(@as(i32, 2), ls[1]);
    try std.testing.expectEqual(@as(i32, 8), ls[2]);
}
