/// DeepSeek-V4-Flash weight loader.
///
/// Supports:
/// - Single-file model.safetensors
/// - Sharded model-00001-of-NNNNN.safetensors with index.json
/// - Direct DeepSeek weight naming (no HF mapping needed)
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const ops = @import("../ops.zig");
const nn = @import("../ops/nn.zig");
const io = @import("../io/mlx_io.zig");
const deepseek_v4 = @import("deepseek_v4.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const DSV4Config = deepseek_v4.DSV4Config;
const DSV4Model = deepseek_v4.DSV4Model;

/// Error set for loading.
pub const LoadError = error{
    MissingIndexJson,
    MissingShardFile,
    MissingWeight,
    InvalidConfig,
    UnsupportedArchitecture,
};

/// Parse DeepSeek-V4 config.json into DSV4Config.
pub fn parseDSV4Config(allocator: std.mem.Allocator, json_text: []const u8) !DSV4Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    const root = parsed.value;
    const obj = root.object;

    // Architecture check
    const archs = obj.get("architectures") orelse return LoadError.InvalidConfig;
    const arch = archs.array.items[0].string;
    if (!std.mem.startsWith(u8, arch, "DeepseekV4")) {
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

    // Parse compress_ratios array
    var compress_ratios = std.ArrayList(usize).empty;
    errdefer compress_ratios.deinit(allocator);
    if (obj.get("compress_ratios")) |ratios_val| {
        for (ratios_val.array.items) |item| {
            const ratio: usize = switch (item) {
                .integer => |n| @intCast(n),
                .float => |f| @intFromFloat(f),
                else => 0,
            };
            try compress_ratios.append(allocator, ratio);
        }
    }

    // Parse rope_scaling
    var rope_scaling: ?DSV4Config.YarnRoPEConfig = null;
    if (obj.get("rope_scaling")) |rs| {
        const rs_obj = rs.object;
        rope_scaling = DSV4Config.YarnRoPEConfig{
            .factor = @floatCast(getFloat(rs_obj, "factor") orelse 16.0),
            .original_max_position_embeddings = @intCast(getInt(rs_obj, "original_max_position_embeddings") orelse 65536),
            .beta_fast = @floatCast(getFloat(rs_obj, "beta_fast") orelse 32.0),
            .beta_slow = @floatCast(getFloat(rs_obj, "beta_slow") orelse 1.0),
        };
    }

    // Parse scoring_func
    const scoring_func: DSV4Config.ScoringFunc = blk: {
        if (obj.get("scoring_func")) |sf| {
            const sf_str = sf.string;
            if (std.mem.eql(u8, sf_str, "softmax")) break :blk .softmax;
            if (std.mem.eql(u8, sf_str, "sigmoid")) break :blk .sigmoid;
            if (std.mem.eql(u8, sf_str, "sqrtsoftplus")) break :blk .sqrtsoftplus;
        }
        break :blk .sqrtsoftplus;
    };

    return DSV4Config{
        .vocab_size = @intCast(getInt(obj, "vocab_size") orelse 129280),
        .hidden_size = @intCast(getInt(obj, "hidden_size") orelse 4096),
        .num_hidden_layers = @intCast(getInt(obj, "num_hidden_layers") orelse 43),
        .num_attention_heads = @intCast(getInt(obj, "num_attention_heads") orelse 64),
        .head_dim = @intCast(getInt(obj, "head_dim") orelse 512),
        .num_key_value_heads = @intCast(getInt(obj, "num_key_value_heads") orelse 1),
        .q_lora_rank = @intCast(getInt(obj, "q_lora_rank") orelse 1024),
        .o_lora_rank = @intCast(getInt(obj, "o_lora_rank") orelse 1024),
        .qk_rope_head_dim = @intCast(getInt(obj, "qk_rope_head_dim") orelse 64),
        .max_position_embeddings = @intCast(getInt(obj, "max_position_embeddings") orelse 1048576),
        .n_routed_experts = @intCast(getInt(obj, "n_routed_experts") orelse 256),
        .n_shared_experts = @intCast(getInt(obj, "n_shared_experts") orelse 1),
        .num_experts_per_tok = @intCast(getInt(obj, "num_experts_per_tok") orelse 6),
        .moe_intermediate_size = @intCast(getInt(obj, "moe_intermediate_size") orelse 2048),
        .routed_scaling_factor = @floatCast(getFloat(obj, "routed_scaling_factor") orelse 1.5),
        .scoring_func = scoring_func,
        .norm_topk_prob = blk: {
            if (obj.get("norm_topk_prob")) |v| {
                switch (v) {
                    .bool => |b| break :blk b,
                    else => break :blk true,
                }
            }
            break :blk true;
        },
        .sliding_window = @intCast(getInt(obj, "sliding_window") orelse 128),
        .num_hash_layers = @intCast(getInt(obj, "num_hash_layers") orelse 3),
        .index_n_heads = @intCast(getInt(obj, "index_n_heads") orelse 64),
        .index_head_dim = @intCast(getInt(obj, "index_head_dim") orelse 128),
        .index_topk = @intCast(getInt(obj, "index_topk") orelse 512),
        .hc_mult = @intCast(getInt(obj, "hc_mult") orelse 4),
        .hc_sinkhorn_iters = @intCast(getInt(obj, "hc_sinkhorn_iters") orelse 20),
        .hc_eps = @floatCast(getFloat(obj, "hc_eps") orelse 1e-6),
        .rms_norm_eps = @floatCast(getFloat(obj, "rms_norm_eps") orelse 1e-6),
        .rope_theta = @floatCast(getFloat(obj, "rope_theta") orelse 10000.0),
        .compress_rope_theta = @floatCast(getFloat(obj, "compress_rope_theta") orelse 160000.0),
        .rope_scaling = rope_scaling,
        .compress_ratios = try compress_ratios.toOwnedSlice(allocator),
        .swiglu_limit = @floatCast(getFloat(obj, "swiglu_limit") orelse 10.0),
        // Parse global quantization defaults from "quantization" or "quantization_config"
        .quantize_default_bits = blk: {
            const qc = obj.get("quantization") orelse obj.get("quantization_config") orelse break :blk 0;
            if (qc != .object) break :blk 0;
            break :blk @intCast(getInt(qc.object, "bits") orelse 0);
        },
        .quantize_default_group_size = blk: {
            const qc = obj.get("quantization") orelse obj.get("quantization_config") orelse break :blk 64;
            if (qc != .object) break :blk 64;
            break :blk @intCast(getInt(qc.object, "group_size") orelse 64);
        },
        .quantize_default_mode = blk: {
            const qc = obj.get("quantization") orelse obj.get("quantization_config") orelse break :blk "affine";
            if (qc != .object) break :blk "affine";
            if (qc.object.get("mode")) |m| {
                if (m == .string) break :blk m.string;
            }
            break :blk "affine";
        },
    };
}

/// Load config from a directory path.
pub fn loadDSV4Config(allocator: std.mem.Allocator, io_ctx: std.Io, dir_path: []const u8) !DSV4Config {
    const config_path = try std.fs.path.join(allocator, &.{ dir_path, "config.json" });
    defer allocator.free(config_path);

    const content = try std.Io.Dir.cwd().readFileAlloc(io_ctx, config_path, allocator, .limited(1024 * 1024));
    defer allocator.free(content);

    return parseDSV4Config(allocator, content);
}

/// Load weights from a model directory.
/// Automatically detects sharded vs single-file models.
pub fn loadWeightsFromDirectory(
    allocator: std.mem.Allocator,
    io_ctx: std.Io,
    dir_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
) !std.StringHashMap(Array) {
    // Try sharded loading first
    const index_path = try std.fs.path.join(allocator, &.{ dir_path, "model.safetensors.index.json" });
    defer allocator.free(index_path);

    const has_index = blk: {
        const dir = std.Io.Dir.cwd();
        const file = dir.openFile(io_ctx, index_path, .{}) catch break :blk false;
        file.close(io_ctx);
        break :blk true;
    };

    if (has_index) {
        return try loadShardedWeights(allocator, io_ctx, dir_path, index_path, ctx, stream);
    }

    // Fall back to single file
    const model_path = try std.fs.path.join(allocator, &.{ dir_path, "model.safetensors" });
    defer allocator.free(model_path);

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
        // Convert BF16 to F32 for CPU eager layers
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
    io_ctx: std.Io,
    dir_path: []const u8,
    index_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
) !std.StringHashMap(Array) {
    _ = stream;
    // Read index.json
    const index_content = try std.Io.Dir.cwd().readFileAlloc(io_ctx, index_path, allocator, .limited(50 * 1024 * 1024));
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

/// Build DSV4Model from loaded weights.
pub fn buildDSV4Model(
    allocator: std.mem.Allocator,
    config: *const DSV4Config,
    weights: *std.StringHashMap(Array),
    ctx: EagerContext,
    stream: c.c.mlx_stream,
) !DSV4Model {
    _ = stream;
    const num_layers = config.num_hidden_layers;
    const hidden_size = config.hidden_size;
    const vocab_size = config.vocab_size;
    const head_dim = config.head_dim;
    const rope_dim = config.qk_rope_head_dim;
    const n_routed_experts = config.n_routed_experts;
    const moe_inter_dim = config.moe_intermediate_size;

    // === Embedding ===
    const embed_weight = weights.get("embed.weight") orelse return LoadError.MissingWeight;
    const embed = nn.Embedding{
        .ctx = ctx,
        .num_embeddings = vocab_size,
        .embedding_dim = hidden_size,
        .weight = embed_weight,
    };
    if (weights.fetchRemove("embed.weight")) |kv| allocator.free(kv.key);

    // === Output norms ===
    const norm_weight = weights.get("norm.weight") orelse return LoadError.MissingWeight;
    var norm = try nn.RMSNorm.init(ctx, hidden_size, config.rms_norm_eps);
    norm.weight.deinit();
    norm.weight = norm_weight;
    if (weights.fetchRemove("norm.weight")) |kv| allocator.free(kv.key);

    var head_norm: ?nn.RMSNorm = null;
    if (weights.get("head_norm.weight")) |hn| {
        head_norm = try nn.RMSNorm.init(ctx, hidden_size, config.rms_norm_eps);
        head_norm.?.weight.deinit();
        head_norm.?.weight = hn;
        if (weights.fetchRemove("head_norm.weight")) |kv| allocator.free(kv.key);
    }

    // === LM Head ===
    const lm_head_weight = weights.get("head.weight") orelse weights.get("lm_head.weight") orelse return LoadError.MissingWeight;
    const lm_head = lm_head_weight;
    if (weights.fetchRemove("head.weight")) |kv| allocator.free(kv.key);
    if (weights.fetchRemove("lm_head.weight")) |kv| allocator.free(kv.key);

    // === Layers ===
    const layers = try allocator.alloc(deepseek_v4.DSV4TransformerBlock, num_layers);
    errdefer allocator.free(layers);

    for (0..num_layers) |i| {
        const idx_fmt = try std.fmt.allocPrint(allocator, "layers.{d}.", .{i});
        defer allocator.free(idx_fmt);

        // --- Attention weights ---
        const wq_a_name = try std.fmt.allocPrint(allocator, "{s}attn.wq_a.weight", .{idx_fmt});
        defer allocator.free(wq_a_name);
        const wq_a = weights.get(wq_a_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(wq_a_name)) |kv| allocator.free(kv.key);

        const wq_b_name = try std.fmt.allocPrint(allocator, "{s}attn.wq_b.weight", .{idx_fmt});
        defer allocator.free(wq_b_name);
        const wq_b = weights.get(wq_b_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(wq_b_name)) |kv| allocator.free(kv.key);

        const wkv_name = try std.fmt.allocPrint(allocator, "{s}attn.wkv.weight", .{idx_fmt});
        defer allocator.free(wkv_name);
        const wkv = weights.get(wkv_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(wkv_name)) |kv| allocator.free(kv.key);

        const wo_a_name = try std.fmt.allocPrint(allocator, "{s}attn.wo_a.weight", .{idx_fmt});
        defer allocator.free(wo_a_name);
        const wo_a = weights.get(wo_a_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(wo_a_name)) |kv| allocator.free(kv.key);

        const wo_b_name = try std.fmt.allocPrint(allocator, "{s}attn.wo_b.weight", .{idx_fmt});
        defer allocator.free(wo_b_name);
        const wo_b = weights.get(wo_b_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(wo_b_name)) |kv| allocator.free(kv.key);

        // Attention norms
        const q_norm_name = try std.fmt.allocPrint(allocator, "{s}attn.q_norm.weight", .{idx_fmt});
        defer allocator.free(q_norm_name);
        const q_norm_w = weights.get(q_norm_name) orelse return LoadError.MissingWeight;
        var q_norm = try nn.RMSNorm.init(ctx, config.q_lora_rank, config.rms_norm_eps);
        q_norm.weight.deinit();
        q_norm.weight = q_norm_w;
        if (weights.fetchRemove(q_norm_name)) |kv| allocator.free(kv.key);

        const kv_norm_name = try std.fmt.allocPrint(allocator, "{s}attn.kv_norm.weight", .{idx_fmt});
        defer allocator.free(kv_norm_name);
        const kv_norm_w = weights.get(kv_norm_name) orelse return LoadError.MissingWeight;
        var kv_norm = try nn.RMSNorm.init(ctx, head_dim, config.rms_norm_eps);
        kv_norm.weight.deinit();
        kv_norm.weight = kv_norm_w;
        if (weights.fetchRemove(kv_norm_name)) |kv| allocator.free(kv.key);

        // Attn sink: learnable logits for attention softmax denominator
        var sink_logits: ?Array = null;
        const attn_sink_name = try std.fmt.allocPrint(allocator, "{s}attn.attn_sink", .{idx_fmt});
        defer allocator.free(attn_sink_name);
        if (weights.get(attn_sink_name)) |sl| {
            sink_logits = sl;
            if (weights.fetchRemove(attn_sink_name)) |kv| allocator.free(kv.key);
        }

        // RoPE
        const compress_ratio = if (i < config.compress_ratios.len) config.compress_ratios[i] else 0;
        const rope_config = config.rope_scaling orelse DSV4Config.YarnRoPEConfig{};
        const rope = try deepseek_v4.DSV4YarnRoPE.init(ctx, rope_dim, config.max_position_embeddings, config.rope_theta, rope_config);

        // Compressor weights (optional, only for CSA/HCA layers with compress_ratio > 1)
        var compress_gate_weight: ?Array = null;
        var compress_pos_bias: ?Array = null;
        if (compress_ratio > 1) {
            const cgw_name = try std.fmt.allocPrint(allocator, "{s}attn.compress_gate_weight", .{idx_fmt});
            defer allocator.free(cgw_name);
            if (weights.get(cgw_name)) |cgw| {
                compress_gate_weight = cgw;
                if (weights.fetchRemove(cgw_name)) |kv| allocator.free(kv.key);
            }

            const cpb_name = try std.fmt.allocPrint(allocator, "{s}attn.compress_pos_bias", .{idx_fmt});
            defer allocator.free(cpb_name);
            if (weights.get(cpb_name)) |cpb| {
                compress_pos_bias = cpb;
                if (weights.fetchRemove(cpb_name)) |kv| allocator.free(kv.key);
            }
        }

        // Lightning Indexer weights (optional, only for CSA layers with compress_ratio > 1)
        var indexer: ?deepseek_v4.LightningIndexer = null;
        if (compress_ratio > 1) {
            const wq_idx_name = try std.fmt.allocPrint(allocator, "{s}attn.indexer.wq.weight", .{idx_fmt});
            defer allocator.free(wq_idx_name);
            const wk_idx_name = try std.fmt.allocPrint(allocator, "{s}attn.indexer.wk.weight", .{idx_fmt});
            defer allocator.free(wk_idx_name);

            const wq_idx = weights.get(wq_idx_name);
            const wk_idx = weights.get(wk_idx_name);

            if (wq_idx != null and wk_idx != null) {
                if (weights.fetchRemove(wq_idx_name)) |kv| allocator.free(kv.key);
                if (weights.fetchRemove(wk_idx_name)) |kv| allocator.free(kv.key);
                indexer = deepseek_v4.LightningIndexer{
                    .ctx = ctx,
                    .wq_index = wq_idx.?,
                    .wk_index = wk_idx.?,
                    .index_n_heads = config.index_n_heads,
                    .index_head_dim = config.index_head_dim,
                    .index_topk = config.index_topk,
                };
            }
        }

        // --- Attention struct ---
        const attention = deepseek_v4.DSV4Attention{
            .ctx = ctx,
            .config = config,
            .layer_idx = i,
            .wq_a = wq_a,
            .wq_b = wq_b,
            .q_norm = q_norm,
            .wkv = wkv,
            .kv_norm = kv_norm,
            .kv_b = null, // V4 doesn't use kv_b
            .wo_a = wo_a,
            .wo_b = wo_b,
            .rope = rope,
            .compress_ratio = compress_ratio,
            .compress_gate_weight = compress_gate_weight,
            .compress_pos_bias = compress_pos_bias,
            .indexer = indexer,
            .sink_logits = sink_logits,
        };

        // --- MoE weights ---
        // Gate
        const gate_weight_name = try std.fmt.allocPrint(allocator, "{s}ffn.gate.weight", .{idx_fmt});
        defer allocator.free(gate_weight_name);
        const gate_weight = weights.get(gate_weight_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(gate_weight_name)) |kv| allocator.free(kv.key);

        var gate_bias: ?Array = null;
        const gate_bias_name = try std.fmt.allocPrint(allocator, "{s}ffn.gate.bias", .{idx_fmt});
        defer allocator.free(gate_bias_name);
        if (weights.get(gate_bias_name)) |gb| {
            gate_bias = gb;
            if (weights.fetchRemove(gate_bias_name)) |kv| allocator.free(kv.key);
        }

        var tid2eid: ?Array = null;
        const tid2eid_name = try std.fmt.allocPrint(allocator, "{s}ffn.gate.tid2eid", .{idx_fmt});
        defer allocator.free(tid2eid_name);
        if (weights.get(tid2eid_name)) |te| {
            tid2eid = te;
            if (weights.fetchRemove(tid2eid_name)) |kv| allocator.free(kv.key);
        }

        const is_hash = i < config.num_hash_layers;
        const gate = deepseek_v4.DSV4Gate{
            .ctx = ctx,
            .weight = gate_weight,
            .bias = gate_bias,
            .tid2eid = tid2eid,
            .topk = config.num_experts_per_tok,
            .n_routed_experts = n_routed_experts,
            .route_scale = config.routed_scaling_factor,
            .scoring_func = config.scoring_func,
            .is_hash = is_hash,
        };

        // Shared expert
        const shared_w1_name = try std.fmt.allocPrint(allocator, "{s}ffn.shared_experts.w1.weight", .{idx_fmt});
        defer allocator.free(shared_w1_name);
        const shared_w1 = weights.get(shared_w1_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(shared_w1_name)) |kv| allocator.free(kv.key);

        const shared_w3_name = try std.fmt.allocPrint(allocator, "{s}ffn.shared_experts.w3.weight", .{idx_fmt});
        defer allocator.free(shared_w3_name);
        const shared_w3 = weights.get(shared_w3_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(shared_w3_name)) |kv| allocator.free(kv.key);

        const shared_w2_name = try std.fmt.allocPrint(allocator, "{s}ffn.shared_experts.w2.weight", .{idx_fmt});
        defer allocator.free(shared_w2_name);
        const shared_w2 = weights.get(shared_w2_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(shared_w2_name)) |kv| allocator.free(kv.key);

        const shared_expert = deepseek_v4.DSV4Expert{
            .ctx = ctx,
            .w1 = shared_w1,
            .w2 = shared_w2,
            .w3 = shared_w3,
            .swiglu_limit = 0, // shared expert has no limit
        };

        // Routed experts
        const experts = try allocator.alloc(deepseek_v4.DSV4Expert, n_routed_experts);
        errdefer allocator.free(experts);

        for (0..n_routed_experts) |e| {
            const ew1_name = try std.fmt.allocPrint(allocator, "{s}ffn.experts.{d}.w1.weight", .{ idx_fmt, e });
            defer allocator.free(ew1_name);
            const ew1 = weights.get(ew1_name); // may be missing for some experts
            if (weights.fetchRemove(ew1_name)) |kv| allocator.free(kv.key);

            const ew3_name = try std.fmt.allocPrint(allocator, "{s}ffn.experts.{d}.w3.weight", .{ idx_fmt, e });
            defer allocator.free(ew3_name);
            const ew3 = weights.get(ew3_name);
            if (weights.fetchRemove(ew3_name)) |kv| allocator.free(kv.key);

            const ew2_name = try std.fmt.allocPrint(allocator, "{s}ffn.experts.{d}.w2.weight", .{ idx_fmt, e });
            defer allocator.free(ew2_name);
            const ew2 = weights.get(ew2_name);
            if (weights.fetchRemove(ew2_name)) |kv| allocator.free(kv.key);

            // If expert weights are missing, create zero arrays as placeholders
            const w1_actual = ew1 orelse try array_mod.zeros(allocator, &[_]i32{ @intCast(moe_inter_dim), @intCast(hidden_size) }, .float32);
            const w3_actual = ew3 orelse try array_mod.zeros(allocator, &[_]i32{ @intCast(moe_inter_dim), @intCast(hidden_size) }, .float32);
            const w2_actual = ew2 orelse try array_mod.zeros(allocator, &[_]i32{ @intCast(hidden_size), @intCast(moe_inter_dim) }, .float32);

            experts[e] = deepseek_v4.DSV4Expert{
                .ctx = ctx,
                .w1 = w1_actual,
                .w2 = w2_actual,
                .w3 = w3_actual,
                .swiglu_limit = config.swiglu_limit,
            };
        }

        const moe = deepseek_v4.DSV4MoE{
            .ctx = ctx,
            .gate = gate,
            .experts = experts,
            .shared_expert = shared_expert,
            .n_routed_experts = n_routed_experts,
            .n_activated_experts = config.num_experts_per_tok,
        };

        // --- Layer norms ---
        const attn_norm_name = try std.fmt.allocPrint(allocator, "{s}attn_norm.weight", .{idx_fmt});
        defer allocator.free(attn_norm_name);
        const attn_norm_w = weights.get(attn_norm_name) orelse return LoadError.MissingWeight;
        var attn_norm = try nn.RMSNorm.init(ctx, hidden_size, config.rms_norm_eps);
        attn_norm.weight.deinit();
        attn_norm.weight = attn_norm_w;
        if (weights.fetchRemove(attn_norm_name)) |kv| allocator.free(kv.key);

        const ffn_norm_name = try std.fmt.allocPrint(allocator, "{s}ffn_norm.weight", .{idx_fmt});
        defer allocator.free(ffn_norm_name);
        const ffn_norm_w = weights.get(ffn_norm_name) orelse return LoadError.MissingWeight;
        var ffn_norm = try nn.RMSNorm.init(ctx, hidden_size, config.rms_norm_eps);
        ffn_norm.weight.deinit();
        ffn_norm.weight = ffn_norm_w;
        if (weights.fetchRemove(ffn_norm_name)) |kv| allocator.free(kv.key);

        // --- mHC parameters (optional, loaded but may not be used in simplified version) ---
        const hc_attn_fn_name = try std.fmt.allocPrint(allocator, "{s}hc_attn_fn", .{idx_fmt});
        defer allocator.free(hc_attn_fn_name);
        const hc_attn_fn = weights.get(hc_attn_fn_name);
        if (weights.fetchRemove(hc_attn_fn_name)) |kv| allocator.free(kv.key);

        const hc_attn_base_name = try std.fmt.allocPrint(allocator, "{s}hc_attn_base", .{idx_fmt});
        defer allocator.free(hc_attn_base_name);
        const hc_attn_base = weights.get(hc_attn_base_name);
        if (weights.fetchRemove(hc_attn_base_name)) |kv| allocator.free(kv.key);

        const hc_attn_scale_name = try std.fmt.allocPrint(allocator, "{s}hc_attn_scale", .{idx_fmt});
        defer allocator.free(hc_attn_scale_name);
        const hc_attn_scale = weights.get(hc_attn_scale_name);
        if (weights.fetchRemove(hc_attn_scale_name)) |kv| allocator.free(kv.key);

        const hc_ffn_fn_name = try std.fmt.allocPrint(allocator, "{s}hc_ffn_fn", .{idx_fmt});
        defer allocator.free(hc_ffn_fn_name);
        const hc_ffn_fn = weights.get(hc_ffn_fn_name);
        if (weights.fetchRemove(hc_ffn_fn_name)) |kv| allocator.free(kv.key);

        const hc_ffn_base_name = try std.fmt.allocPrint(allocator, "{s}hc_ffn_base", .{idx_fmt});
        defer allocator.free(hc_ffn_base_name);
        const hc_ffn_base = weights.get(hc_ffn_base_name);
        if (weights.fetchRemove(hc_ffn_base_name)) |kv| allocator.free(kv.key);

        const hc_ffn_scale_name = try std.fmt.allocPrint(allocator, "{s}hc_ffn_scale", .{idx_fmt});
        defer allocator.free(hc_ffn_scale_name);
        const hc_ffn_scale = weights.get(hc_ffn_scale_name);
        if (weights.fetchRemove(hc_ffn_scale_name)) |kv| allocator.free(kv.key);

        // Use mHC if weights present, otherwise use dummy
        const dummy_arr = try array_mod.zeros(allocator, &[_]i32{1}, .float32);
        const hc_attn = deepseek_v4.DSV4HyperConn{
            .hc_fn = hc_attn_fn orelse dummy_arr,
            .hc_base = hc_attn_base orelse try array_mod.zeros(allocator, &[_]i32{1}, .float32),
            .hc_scale = hc_attn_scale orelse try array_mod.zeros(allocator, &[_]i32{1}, .float32),
            .hc_mult = config.hc_mult,
            .hc_sinkhorn_iters = config.hc_sinkhorn_iters,
            .hc_eps = config.hc_eps,
        };
        const hc_ffn = deepseek_v4.DSV4HyperConn{
            .hc_fn = hc_ffn_fn orelse try array_mod.zeros(allocator, &[_]i32{1}, .float32),
            .hc_base = hc_ffn_base orelse try array_mod.zeros(allocator, &[_]i32{1}, .float32),
            .hc_scale = hc_ffn_scale orelse try array_mod.zeros(allocator, &[_]i32{1}, .float32),
            .hc_mult = config.hc_mult,
            .hc_sinkhorn_iters = config.hc_sinkhorn_iters,
            .hc_eps = config.hc_eps,
        };

        layers[i] = deepseek_v4.DSV4TransformerBlock{
            .ctx = ctx,
            .config = config,
            .layer_idx = i,
            .attn_norm = attn_norm,
            .ffn_norm = ffn_norm,
            .attn = attention,
            .ffn = moe,
            .hc_attn = hc_attn,
            .hc_ffn = hc_ffn,
        };
    }

    // Check for unmapped weights
    var remaining = weights.iterator();
    while (remaining.next()) |entry| {
        // Skip scale tensors (quantization metadata)
        if (std.mem.endsWith(u8, entry.key_ptr.*, ".scale")) continue;
        std.log.warn("Unused weight: {s}", .{entry.key_ptr.*});
    }

    return deepseek_v4.DSV4Model{
        .allocator = allocator,
        .ctx = ctx,
        .config = config.*,
        .embed_tokens = embed,
        .layers = layers,
        .norm = norm,
        .head_norm = head_norm,
        .lm_head = lm_head,
    };
}
