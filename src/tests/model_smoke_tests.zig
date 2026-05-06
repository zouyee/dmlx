/// Minimal end-to-end smoke tests for LLaMA and DeepSeek V4 model code paths.
/// Uses tiny random-weight models (no real weights needed).
const std = @import("std");
const mlx = @import("../root.zig");
const c = @import("../c.zig");

const Array = mlx.Array;
const EagerContext = mlx.EagerContext;
const array_mod = @import("../array.zig");

// ============================================================
// Helper: create random weights map for a tiny LLaMA model
// ============================================================

fn createLlamaWeights(
    allocator: std.mem.Allocator,
    config: *const mlx.models.LlamaConfig,
    ctx: EagerContext,
) !std.StringHashMap(Array) {
    var weights = std.StringHashMap(Array).init(allocator);
    errdefer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    const hidden: i32 = @intCast(config.hidden_size);
    const vocab: i32 = @intCast(config.vocab_size);
    const inter: i32 = @intCast(config.intermediate_size);
    const num_heads: i32 = @intCast(config.num_attention_heads);
    const num_kv_heads: i32 = @intCast(config.num_key_value_heads);
    const head_dim = @divTrunc(hidden, num_heads);

    var rng = std.Random.DefaultPrng.init(42);
    const key_arr = try mlx.random.key(rng.random().int(u64));
    defer key_arr.deinit();

    // embed_tokens.weight [vocab_size, hidden_size]
    try weights.put(
        try allocator.dupe(u8, "embed_tokens.weight"),
        try mlx.random.normal(ctx, &[_]i32{ vocab, hidden }, .float32, 0.0, 0.02, key_arr),
    );

    // norm.weight [hidden_size] — ones for stable norm
    try weights.put(
        try allocator.dupe(u8, "norm.weight"),
        try array_mod.ones(allocator, &[_]i32{hidden}, .float32),
    );

    // lm_head.weight [vocab_size, hidden_size]
    try weights.put(
        try allocator.dupe(u8, "lm_head.weight"),
        try mlx.random.normal(ctx, &[_]i32{ vocab, hidden }, .float32, 0.0, 0.02, key_arr),
    );

    for (0..config.num_hidden_layers) |layer_idx| {
        const prefix = try std.fmt.allocPrint(allocator, "layers.{d}.", .{layer_idx});
        defer allocator.free(prefix);

        // input_layernorm — ones
        try weights.put(
            try std.fmt.allocPrint(allocator, "{s}input_layernorm", .{prefix}),
            try array_mod.ones(allocator, &[_]i32{hidden}, .float32),
        );

        // attention weights
        try weights.put(
            try std.fmt.allocPrint(allocator, "{s}attention.wq", .{prefix}),
            try mlx.random.normal(ctx, &[_]i32{ num_heads * head_dim, hidden }, .float32, 0.0, 0.02, key_arr),
        );
        try weights.put(
            try std.fmt.allocPrint(allocator, "{s}attention.wk", .{prefix}),
            try mlx.random.normal(ctx, &[_]i32{ num_kv_heads * head_dim, hidden }, .float32, 0.0, 0.02, key_arr),
        );
        try weights.put(
            try std.fmt.allocPrint(allocator, "{s}attention.wv", .{prefix}),
            try mlx.random.normal(ctx, &[_]i32{ num_kv_heads * head_dim, hidden }, .float32, 0.0, 0.02, key_arr),
        );
        try weights.put(
            try std.fmt.allocPrint(allocator, "{s}attention.wo", .{prefix}),
            try mlx.random.normal(ctx, &[_]i32{ hidden, hidden }, .float32, 0.0, 0.02, key_arr),
        );

        // post_attention_layernorm — ones
        try weights.put(
            try std.fmt.allocPrint(allocator, "{s}post_attention_layernorm", .{prefix}),
            try array_mod.ones(allocator, &[_]i32{hidden}, .float32),
        );

        // MLP weights
        try weights.put(
            try std.fmt.allocPrint(allocator, "{s}mlp.gate_proj", .{prefix}),
            try mlx.random.normal(ctx, &[_]i32{ inter, hidden }, .float32, 0.0, 0.02, key_arr),
        );
        try weights.put(
            try std.fmt.allocPrint(allocator, "{s}mlp.up_proj", .{prefix}),
            try mlx.random.normal(ctx, &[_]i32{ inter, hidden }, .float32, 0.0, 0.02, key_arr),
        );
        try weights.put(
            try std.fmt.allocPrint(allocator, "{s}mlp.down_proj", .{prefix}),
            try mlx.random.normal(ctx, &[_]i32{ hidden, inter }, .float32, 0.0, 0.02, key_arr),
        );
    }

    return weights;
}

// ============================================================
// Test 1: LLaMA forward pass smoke test
// ============================================================

test "llama smoke test - forward pass with tiny random model" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = mlx.models.LlamaConfig{
        .vocab_size = 32,
        .hidden_size = 16,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .num_key_value_heads = 2,
        .intermediate_size = 32,
        .rms_norm_eps = 1e-5,
        .rope_theta = 10000.0,
        .max_position_embeddings = 64,
    };

    var weights = try createLlamaWeights(allocator, &config, ctx);
    defer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    var model = try mlx.model_loader.buildModel(allocator, &config, &weights, ctx, stream, null);
    defer model.deinit();

    // Forward pass: batch=1, seq_len=3, tokens [1, 2, 3]
    const input_ids = try Array.fromData(allocator, u32, &[_]u32{ 1, 2, 3 }, &[_]i32{ 1, 3 });
    defer input_ids.deinit();

    const logits = try model.forward(input_ids, null, null);
    defer logits.deinit();

    // Verify output shape: [1, 3, 32] (batch=1, seq=3, vocab=32)
    const shape = logits.shape();
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 3), shape[1]);
    try std.testing.expectEqual(@as(i32, 32), shape[2]);

    // Verify output values are finite (not NaN or Inf)
    const data = try logits.dataSlice(f32);
    for (data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }
}

// ============================================================
// Test 2: LLaMA generate smoke test
// ============================================================

test "llama smoke test - generate tokens with tiny random model" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = mlx.models.LlamaConfig{
        .vocab_size = 32,
        .hidden_size = 16,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .num_key_value_heads = 2,
        .intermediate_size = 32,
        .rms_norm_eps = 1e-5,
        .rope_theta = 10000.0,
        .max_position_embeddings = 64,
    };

    var weights = try createLlamaWeights(allocator, &config, ctx);
    defer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    var model = try mlx.model_loader.buildModel(allocator, &config, &weights, ctx, stream, null);
    defer model.deinit();

    // Prompt: [1, 2, 3], generate 5 new tokens
    const prompt = try Array.fromData(allocator, u32, &[_]u32{ 1, 2, 3 }, &[_]i32{ 1, 3 });
    defer prompt.deinit();

    var sampler_config = mlx.sampling.SamplerConfig.init(42);
    sampler_config.temperature = 0.1;
    sampler_config.top_k = 10;
    sampler_config.top_p = 1.0;

    // Generate without KV cache (simpler path)
    const tokens = try model.generate(prompt, 5, &sampler_config, &[_]mlx.kvcache.KVCacheStrategy{}, .{});
    defer allocator.free(tokens);

    // Should return prompt (3) + generated (5) = 8 tokens
    try std.testing.expectEqual(@as(usize, 8), tokens.len);

    // All generated tokens must be in range [0, vocab_size)
    for (tokens) |tok| {
        try std.testing.expect(tok < config.vocab_size);
    }
}

// ============================================================
// Test 3: DeepSeek V4 forward pass smoke test
// ============================================================

test "deepseek v4 smoke test - forward pass with tiny random model" {
    const allocator = std.testing.allocator;
    const deepseek_v4 = @import("../models/deepseek_v4.zig");
    const loader = @import("../models/deepseek_v4_loader.zig");
    const ctx = EagerContext.init(allocator);

    const config = deepseek_v4.DSV4Config{
        .vocab_size = 8,
        .hidden_size = 16,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .head_dim = 8,
        .num_key_value_heads = 1,
        .q_lora_rank = 4,
        .o_lora_rank = 4,
        .qk_rope_head_dim = 4,
        .max_position_embeddings = 128,
        .n_routed_experts = 2,
        .n_shared_experts = 1,
        .num_experts_per_tok = 2,
        .moe_intermediate_size = 8,
        .compress_ratios = &[_]usize{0},
        .use_mhc = false,
        .hc_mult = 1,
        .hc_sinkhorn_iters = 5,
        .hc_eps = 1e-6,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
        .sliding_window = 8,
        .num_hash_layers = 0,
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

    try W.put(&weights, allocator, "embed.weight", &[_]i32{ 8, 16 });
    try W.put(&weights, allocator, "norm.weight", &[_]i32{16});
    try W.put(&weights, allocator, "lm_head.weight", &[_]i32{ 8, 16 });

    try W.put(&weights, allocator, "layers.0.attn.wq_a.weight", &[_]i32{ 4, 16 });
    try W.put(&weights, allocator, "layers.0.attn.wq_b.weight", &[_]i32{ 16, 4 });
    try W.put(&weights, allocator, "layers.0.attn.wkv.weight", &[_]i32{ 8, 16 });
    try W.put(&weights, allocator, "layers.0.attn.wo_a.weight", &[_]i32{ 4, 16 });
    try W.put(&weights, allocator, "layers.0.attn.wo_b.weight", &[_]i32{ 16, 4 });
    try W.put(&weights, allocator, "layers.0.attn.q_norm.weight", &[_]i32{4});
    try W.put(&weights, allocator, "layers.0.attn.kv_norm.weight", &[_]i32{8});

    try W.put(&weights, allocator, "layers.0.ffn.gate.weight", &[_]i32{ 2, 16 });
    try W.put(&weights, allocator, "layers.0.ffn.shared_experts.w1.weight", &[_]i32{ 8, 16 });
    try W.put(&weights, allocator, "layers.0.ffn.shared_experts.w2.weight", &[_]i32{ 16, 8 });
    try W.put(&weights, allocator, "layers.0.ffn.shared_experts.w3.weight", &[_]i32{ 8, 16 });
    try W.put(&weights, allocator, "layers.0.ffn.experts.0.w1.weight", &[_]i32{ 8, 16 });
    try W.put(&weights, allocator, "layers.0.ffn.experts.0.w2.weight", &[_]i32{ 16, 8 });
    try W.put(&weights, allocator, "layers.0.ffn.experts.0.w3.weight", &[_]i32{ 8, 16 });
    try W.put(&weights, allocator, "layers.0.ffn.experts.1.w1.weight", &[_]i32{ 8, 16 });
    try W.put(&weights, allocator, "layers.0.ffn.experts.1.w2.weight", &[_]i32{ 16, 8 });
    try W.put(&weights, allocator, "layers.0.ffn.experts.1.w3.weight", &[_]i32{ 8, 16 });

    try W.put(&weights, allocator, "layers.0.attn_norm.weight", &[_]i32{16});
    try W.put(&weights, allocator, "layers.0.ffn_norm.weight", &[_]i32{16});

    var model = try loader.buildDSV4Model(allocator, &config, &weights, ctx, ctx.stream.inner, .{});
    defer model.deinit();

    // Forward pass: batch=1, seq_len=3, tokens [1, 2, 3]
    const input_ids = try Array.fromData(allocator, u32, &[_]u32{ 1, 2, 3 }, &[_]i32{ 1, 3 });
    defer input_ids.deinit();

    const logits = try model.forward(input_ids, null, null, 0, ctx.stream.inner);
    defer logits.deinit();

    // Verify output shape: [1, 3, 8] (batch=1, seq=3, vocab=8)
    const shape = logits.shape();
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 3), shape[1]);
    try std.testing.expectEqual(@as(i32, 8), shape[2]);

    // Verify output values are finite
    const data = try logits.dataSlice(f32);
    for (data) |val| {
        try std.testing.expect(!std.math.isNan(val));
        try std.testing.expect(!std.math.isInf(val));
    }
}
