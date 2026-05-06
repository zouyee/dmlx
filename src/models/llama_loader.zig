/// LlamaModel weight loader from HuggingFace Safetensors format.
///
/// Usage:
///   var model = try LlamaModelLoader.load(allocator, &config, "path/to/model.safetensors", stream);
///   defer model.deinit();
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const nn = @import("mlx").nn;
const io = @import("mlx").io;
const hf_config = @import("../hf_config.zig");
const llama = @import("llama.zig");
const quantize_mod = @import("mlx").quantize;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const LlamaConfig = llama.LlamaConfig;
const LlamaModel = llama.LlamaModel;
const LlamaAttention = llama.LlamaAttention;
const LlamaMLP = llama.LlamaMLP;
const LlamaTransformerBlock = llama.LlamaTransformerBlock;
const QuantizedWeight = quantize_mod.QuantizedWeight;
const QuantConfig = quantize_mod.QuantConfig;

/// Load a LlamaModel from a HF Safetensors file (or multiple shard files).
pub fn loadFromSafetensors(
    allocator: std.mem.Allocator,
    config: *const LlamaConfig,
    path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
) !LlamaModel {
    return loadFromSafetensorsPaths(allocator, config, &[_][]const u8{path}, ctx, stream);
}

/// Load a LlamaModel from multiple safetensors shard files.
pub fn loadFromSafetensorsPaths(
    allocator: std.mem.Allocator,
    config: *const LlamaConfig,
    paths: []const []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
) !LlamaModel {
    // Build internal name -> Array map from all shards
    var weights = std.StringHashMap(Array).init(allocator);
    defer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        weights.deinit();
    }

    for (paths) |path| {
        var st = try io.loadSafetensors(allocator, path);
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

        var hf_it = st.weights.iterator();
        while (hf_it.next()) |entry| {
            const hf_name = entry.key_ptr.*;
            const mapped = try hf_config.mapWeightName(hf_name, allocator) orelse {
                std.log.warn("Unmapped weight: {s}", .{hf_name});
                continue;
            };
            // Convert BF16 weights to float32 for CPU eager layers (Embedding, RMSNorm)
            const weight = entry.value_ptr.*;
            const f32_weight = if (weight.dtype() == .bfloat16)
                try ops.astype(ctx, weight, .float32)
            else
                weight;
            try weights.put(mapped, f32_weight);
        }
    }

    // Construct model (with quantized weight detection)
    const eagle_path = try std.fs.path.join(allocator, &.{ std.fs.path.dirname(paths[0]) orelse ".", "eagle_draft_head.safetensors" });
    defer allocator.free(eagle_path);
    return try buildModel(allocator, config, &weights, ctx, stream, eagle_path);
}

/// Construct LlamaModel from a map of internal-name -> Array.
/// `eagle_path` is an optional path to EAGLE draft head safetensors.
pub fn buildModel(
    allocator: std.mem.Allocator,
    config: *const LlamaConfig,
    weights: *std.StringHashMap(Array),
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    eagle_path: ?[]const u8,
) !LlamaModel {
    const num_layers = config.num_hidden_layers;
    const hidden_size = config.hidden_size;
    const intermediate_size = config.intermediate_size;
    const vocab_size = config.vocab_size;
    const num_heads = config.num_attention_heads;
    const num_kv_heads = config.num_key_value_heads;
    const head_dim = config.getHeadDim();

    // === Embedding (with optional quantized weight dequantization) ===
    // Some models (e.g., Qwen3-4bit) quantize embed_tokens. Since Embedding.forward
    // uses mlx_take which needs full-precision weights, we dequantize at load time.
    const embed_quant = try tryGetQuantizedWeight(allocator, weights, "embed_tokens.weight", config);
    var embed_weight: Array = undefined;
    if (embed_quant) |qw| {
        // Dequantize embedding weight at load time
        embed_weight = try quantize_mod.dequantize(ctx, qw);
        var mutable_qw = qw;
        mutable_qw.deinit(allocator);
    } else {
        embed_weight = weights.get("embed_tokens.weight") orelse return error.MissingEmbedWeight;
        if (weights.fetchRemove("embed_tokens.weight")) |kv| allocator.free(kv.key);
    }
    const embed = nn.Embedding{
        .ctx = ctx,
        .num_embeddings = vocab_size,
        .embedding_dim = hidden_size,
        .weight = embed_weight,
    };

    // === Output norm ===
    const norm_weight = weights.get("norm.weight") orelse return error.MissingNormWeight;
    var norm = try nn.RMSNorm.init(ctx, hidden_size, config.rms_norm_eps);
    norm.weight.deinit();
    norm.weight = norm_weight;
    if (weights.fetchRemove("norm.weight")) |kv| allocator.free(kv.key);

    // === LM Head (with quantized weight detection) ===
    const lm_head_quant = try tryGetQuantizedWeight(allocator, weights, "lm_head.weight", config);
    var lm_head_tied = false;
    const lm_head = blk: {
        if (lm_head_quant != null) {
            break :blk lm_head_quant.?.data;
        }
        if (weights.get("lm_head.weight")) |w| {
            if (weights.fetchRemove("lm_head.weight")) |kv| allocator.free(kv.key);
            break :blk w;
        }
        // Some models tie embeddings with lm_head (common in quantized models)
        std.log.info("lm_head.weight not found, tying to embed_tokens.weight", .{});
        lm_head_tied = true;
        break :blk embed.weight;
    };

    // === Layers ===
    const layers = try allocator.alloc(LlamaTransformerBlock, num_layers);
    errdefer allocator.free(layers);

    for (0..num_layers) |i| {
        layers[i] = try buildLayer(
            allocator,
            config,
            weights,
            i,
            ctx,
            stream,
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
        );
    }

    // Check for unmapped weights
    var remaining = weights.iterator();
    while (remaining.next()) |entry| {
        std.log.warn("Unused weight: {s}", .{entry.key_ptr.*});
    }

    // === EAGLE Draft Head (optional) ===
    var eagle_drafter: ?@import("../speculative.zig").EagleDrafter = null;
    if (eagle_path) |ep| {
        var eagle_loaded = false;
        var eagle_st = blk: {
            const st = io.loadSafetensors(allocator, ep) catch |err| {
                std.log.warn("EAGLE draft head not found at {s}: {}, skipping", .{ ep, err });
                break :blk null;
            };
            eagle_loaded = true;
            break :blk st;
        };
        if (eagle_loaded) {
            defer {
                var w_it = eagle_st.?.weights.iterator();
                while (w_it.next()) |entry| {
                    allocator.free(entry.key_ptr.*);
                    entry.value_ptr.*.deinit();
                }
                eagle_st.?.weights.deinit();
                var m_it = eagle_st.?.metadata.iterator();
                while (m_it.next()) |entry| {
                    allocator.free(entry.key_ptr.*);
                    allocator.free(entry.value_ptr.*);
                }
                eagle_st.?.metadata.deinit();
            }

            var ed = @import("../speculative.zig").EagleDrafter.init(hidden_size, vocab_size);
            if (eagle_st.?.weights.get("weight")) |w| {
                const bias = eagle_st.?.weights.get("bias");
                // Copy arrays so the drafter owns them independently
                var w_copy = c.c.mlx_array_new();
                _ = c.c.mlx_array_set(&w_copy, w.inner);
                const bias_copy = if (bias) |b| blk: {
                    var bc = c.c.mlx_array_new();
                    _ = c.c.mlx_array_set(&bc, b.inner);
                    break :blk Array.fromHandle(bc);
                } else null;
                ed.loadWeights(Array.fromHandle(w_copy), bias_copy);
                eagle_drafter = ed;
            }
        }
    }

    return LlamaModel{
        .allocator = allocator,
        .ctx = ctx,
        .config = config.*,
        .embed_tokens = embed,
        .layers = layers,
        .norm = norm,
        .lm_head = lm_head,
        .lm_head_quant = lm_head_quant,
        .lm_head_tied = lm_head_tied,
        .lora = null,
        .eagle_drafter = eagle_drafter,
    };
}

/// Try to assemble a QuantizedWeight from base_name.scales and base_name.biases in the weights map.
/// Returns null if the weight is not quantized (no .scales/.biases found).
fn tryGetQuantizedWeight(
    allocator: std.mem.Allocator,
    weights: *std.StringHashMap(Array),
    base_name: []const u8,
    config: *const LlamaConfig,
) !?QuantizedWeight {
    const scales_name = try std.fmt.allocPrint(allocator, "{s}.scales", .{base_name});
    defer allocator.free(scales_name);
    const biases_name = try std.fmt.allocPrint(allocator, "{s}.biases", .{base_name});
    defer allocator.free(biases_name);

    const scales = weights.get(scales_name) orelse return null;
    const biases = weights.get(biases_name) orelse return null;
    const data = weights.get(base_name) orelse return null;

    // Remove from map (ownership transferred)
    if (weights.fetchRemove(scales_name)) |kv| allocator.free(kv.key);
    if (weights.fetchRemove(biases_name)) |kv| allocator.free(kv.key);
    if (weights.fetchRemove(base_name)) |kv| allocator.free(kv.key);

    const qconfig = QuantConfig{
        .bits = if (config.quantize_bits > 0) config.quantize_bits else 4,
        .group_size = config.quantize_group_size,
    };

    // Infer original shape from scales: scales shape is [out_features, in_features/group_size]
    // and data shape is [out_features, in_features * bits / 32]
    // Original shape: [out_features, data_cols * 32 / bits]
    const data_shape = data.shape();
    const out_features = data_shape[0];
    const packed_cols = data_shape[1];
    const in_features = @divExact(packed_cols * 32, @as(i32, @intCast(qconfig.bits)));
    const orig_shape = &[_]i32{ out_features, in_features };

    return try quantize_mod.loadPreQuantized(
        allocator,
        data,
        scales,
        biases,
        qconfig,
        orig_shape,
    );
}

fn buildLayer(
    allocator: std.mem.Allocator,
    config: *const LlamaConfig,
    weights: *std.StringHashMap(Array),
    layer_idx: usize,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) !LlamaTransformerBlock {
    _ = stream;
    _ = intermediate_size;
    _ = num_heads;
    _ = num_kv_heads;
    const idx_fmt = try std.fmt.allocPrint(allocator, "layers.{d}.", .{layer_idx});
    defer allocator.free(idx_fmt);

    // === Attention weights (with quantized weight detection) ===
    const wq_name = try std.fmt.allocPrint(allocator, "{s}attention.wq", .{idx_fmt});
    defer allocator.free(wq_name);
    const wq_quant = try tryGetQuantizedWeight(allocator, weights, wq_name, config);
    var wq: Array = undefined;
    if (wq_quant == null) {
        wq = weights.get(wq_name) orelse return error.MissingWeight;
        if (weights.fetchRemove(wq_name)) |kv| allocator.free(kv.key);
    } else {
        // For quantized weights, store a dummy array (data is in QuantizedWeight)
        wq = wq_quant.?.data;
    }

    const wk_name = try std.fmt.allocPrint(allocator, "{s}attention.wk", .{idx_fmt});
    defer allocator.free(wk_name);
    const wk_quant = try tryGetQuantizedWeight(allocator, weights, wk_name, config);
    var wk: Array = undefined;
    if (wk_quant == null) {
        wk = weights.get(wk_name) orelse return error.MissingWeight;
        if (weights.fetchRemove(wk_name)) |kv| allocator.free(kv.key);
    } else {
        wk = wk_quant.?.data;
    }

    const wv_name = try std.fmt.allocPrint(allocator, "{s}attention.wv", .{idx_fmt});
    defer allocator.free(wv_name);
    const wv_quant = try tryGetQuantizedWeight(allocator, weights, wv_name, config);
    var wv: Array = undefined;
    if (wv_quant == null) {
        wv = weights.get(wv_name) orelse return error.MissingWeight;
        if (weights.fetchRemove(wv_name)) |kv| allocator.free(kv.key);
    } else {
        wv = wv_quant.?.data;
    }

    const wo_name = try std.fmt.allocPrint(allocator, "{s}attention.wo", .{idx_fmt});
    defer allocator.free(wo_name);
    const wo_quant = try tryGetQuantizedWeight(allocator, weights, wo_name, config);
    var wo: Array = undefined;
    if (wo_quant == null) {
        wo = weights.get(wo_name) orelse return error.MissingWeight;
        if (weights.fetchRemove(wo_name)) |kv| allocator.free(kv.key);
    } else {
        wo = wo_quant.?.data;
    }

    // Optional Q/K norm (Qwen2)
    var q_norm: ?nn.RMSNorm = null;
    var k_norm: ?nn.RMSNorm = null;

    const q_norm_name = try std.fmt.allocPrint(allocator, "{s}attention.q_norm", .{idx_fmt});
    defer allocator.free(q_norm_name);
    if (weights.get(q_norm_name)) |qn| {
        q_norm = try nn.RMSNorm.init(ctx, head_dim, config.rms_norm_eps);
        q_norm.?.weight.deinit();
        q_norm.?.weight = qn;
        if (weights.fetchRemove(q_norm_name)) |kv| allocator.free(kv.key);
    }

    const k_norm_name = try std.fmt.allocPrint(allocator, "{s}attention.k_norm", .{idx_fmt});
    defer allocator.free(k_norm_name);
    if (weights.get(k_norm_name)) |kn| {
        k_norm = try nn.RMSNorm.init(ctx, head_dim, config.rms_norm_eps);
        k_norm.?.weight.deinit();
        k_norm.?.weight = kn;
        if (weights.fetchRemove(k_norm_name)) |kv| allocator.free(kv.key);
    }

    // === Attention bias (Qwen2) ===
    const bias_names = [_]struct { field: []const u8, suffix: []const u8 }{
        .{ .field = "wq_bias", .suffix = "attention.wq_bias" },
        .{ .field = "wk_bias", .suffix = "attention.wk_bias" },
        .{ .field = "wv_bias", .suffix = "attention.wv_bias" },
        .{ .field = "wo_bias", .suffix = "attention.wo_bias" },
    };
    var wq_bias: ?Array = null;
    var wk_bias: ?Array = null;
    var wv_bias: ?Array = null;
    var wo_bias: ?Array = null;
    inline for (bias_names, 0..) |bn, bi| {
        const bias_key = try std.fmt.allocPrint(allocator, "{s}{s}", .{ idx_fmt, bn.suffix });
        defer allocator.free(bias_key);
        if (weights.get(bias_key)) |b| {
            switch (bi) {
                0 => wq_bias = b,
                1 => wk_bias = b,
                2 => wv_bias = b,
                3 => wo_bias = b,
                else => {},
            }
            if (weights.fetchRemove(bias_key)) |kv| allocator.free(kv.key);
        }
    }

    // === Attention ===
    const rope = try nn.RoPE.init(ctx, head_dim, config.max_position_embeddings, config.rope_theta);
    const attention = LlamaAttention{
        .ctx = ctx,
        .config = config,
        .wq = wq,
        .wk = wk,
        .wv = wv,
        .wo = wo,
        .wq_quant = wq_quant,
        .wk_quant = wk_quant,
        .wv_quant = wv_quant,
        .wo_quant = wo_quant,
        .wq_bias = wq_bias,
        .wk_bias = wk_bias,
        .wv_bias = wv_bias,
        .wo_bias = wo_bias,
        .q_norm = q_norm,
        .k_norm = k_norm,
        .rope = rope,
    };

    // === MLP weights (with quantized weight detection) ===
    const gate_name = try std.fmt.allocPrint(allocator, "{s}mlp.gate_proj", .{idx_fmt});
    defer allocator.free(gate_name);
    const gate_quant = try tryGetQuantizedWeight(allocator, weights, gate_name, config);
    var gate_proj: Array = undefined;
    if (gate_quant == null) {
        gate_proj = weights.get(gate_name) orelse return error.MissingWeight;
        if (weights.fetchRemove(gate_name)) |kv| allocator.free(kv.key);
    } else {
        gate_proj = gate_quant.?.data;
    }

    const up_name = try std.fmt.allocPrint(allocator, "{s}mlp.up_proj", .{idx_fmt});
    defer allocator.free(up_name);
    const up_quant = try tryGetQuantizedWeight(allocator, weights, up_name, config);
    var up_proj: Array = undefined;
    if (up_quant == null) {
        up_proj = weights.get(up_name) orelse return error.MissingWeight;
        if (weights.fetchRemove(up_name)) |kv| allocator.free(kv.key);
    } else {
        up_proj = up_quant.?.data;
    }

    const down_name = try std.fmt.allocPrint(allocator, "{s}mlp.down_proj", .{idx_fmt});
    defer allocator.free(down_name);
    const down_quant = try tryGetQuantizedWeight(allocator, weights, down_name, config);
    var down_proj: Array = undefined;
    if (down_quant == null) {
        down_proj = weights.get(down_name) orelse return error.MissingWeight;
        if (weights.fetchRemove(down_name)) |kv| allocator.free(kv.key);
    } else {
        down_proj = down_quant.?.data;
    }

    const mlp = LlamaMLP{
        .ctx = ctx,
        .config = config,
        .gate_proj = gate_proj,
        .up_proj = up_proj,
        .down_proj = down_proj,
        .gate_proj_quant = gate_quant,
        .up_proj_quant = up_quant,
        .down_proj_quant = down_quant,
    };

    // === Layer norms ===
    const input_ln_name = try std.fmt.allocPrint(allocator, "{s}input_layernorm", .{idx_fmt});
    defer allocator.free(input_ln_name);
    const input_ln_w = weights.get(input_ln_name) orelse return error.MissingWeight;
    if (weights.fetchRemove(input_ln_name)) |kv| allocator.free(kv.key);

    var input_layernorm = try nn.RMSNorm.init(ctx, hidden_size, config.rms_norm_eps);
    input_layernorm.weight.deinit();
    input_layernorm.weight = input_ln_w;

    const post_ln_name = try std.fmt.allocPrint(allocator, "{s}post_attention_layernorm", .{idx_fmt});
    defer allocator.free(post_ln_name);
    const post_ln_w = weights.get(post_ln_name) orelse return error.MissingWeight;
    if (weights.fetchRemove(post_ln_name)) |kv| allocator.free(kv.key);

    var post_attention_layernorm = try nn.RMSNorm.init(ctx, hidden_size, config.rms_norm_eps);
    post_attention_layernorm.weight.deinit();
    post_attention_layernorm.weight = post_ln_w;

    return LlamaTransformerBlock{
        .ctx = ctx,
        .config = config,
        .layer_idx = layer_idx,
        .input_layernorm = input_layernorm,
        .attention = attention,
        .post_attention_layernorm = post_attention_layernorm,
        .mlp = mlp,
    };
}
