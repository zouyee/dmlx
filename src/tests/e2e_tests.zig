const std = @import("std");
const mlx = @import("../root.zig");
const c = @import("mlx").c;

const Array = mlx.Array;
const EagerContext = mlx.EagerContext;

fn createRandomWeights(
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

    const hidden_size: i32 = @intCast(config.hidden_size);
    const vocab_size: i32 = @intCast(config.vocab_size);
    const intermediate_size: i32 = @intCast(config.intermediate_size);
    const num_heads: i32 = @intCast(config.num_attention_heads);
    const num_kv_heads: i32 = @intCast(config.num_key_value_heads);
    const head_dim = @divTrunc(hidden_size, num_heads);

    var rng = std.Random.DefaultPrng.init(42);
    const key_arr = try mlx.random.key(rng.random().int(u64));
    defer key_arr.deinit();

    // embed_tokens.weight [vocab_size, hidden_size]
    const embed_name = try allocator.dupe(u8, "embed_tokens.weight");
    try weights.put(embed_name, try mlx.random.normal(ctx, &[_]i32{ vocab_size, hidden_size }, .float32, 0.0, 0.02, key_arr));

    // norm.weight [hidden_size]
    const norm_name = try allocator.dupe(u8, "norm.weight");
    try weights.put(norm_name, try mlx.random.normal(ctx, &[_]i32{hidden_size}, .float32, 1.0, 0.0, key_arr));

    // lm_head.weight [vocab_size, hidden_size]
    const lm_head_name = try allocator.dupe(u8, "lm_head.weight");
    try weights.put(lm_head_name, try mlx.random.normal(ctx, &[_]i32{ vocab_size, hidden_size }, .float32, 0.0, 0.02, key_arr));

    for (0..config.num_hidden_layers) |layer_idx| {
        const prefix = try std.fmt.allocPrint(allocator, "layers.{d}.", .{layer_idx});
        defer allocator.free(prefix);

        // input_layernorm.weight
        const iln_name = try std.fmt.allocPrint(allocator, "{s}input_layernorm", .{prefix});
        try weights.put(iln_name, try mlx.random.normal(ctx, &[_]i32{hidden_size}, .float32, 1.0, 0.0, key_arr));

        // attention weights
        const wq_name = try std.fmt.allocPrint(allocator, "{s}attention.wq", .{prefix});
        try weights.put(wq_name, try mlx.random.normal(ctx, &[_]i32{ hidden_size, hidden_size }, .float32, 0.0, 0.02, key_arr));

        const wk_name = try std.fmt.allocPrint(allocator, "{s}attention.wk", .{prefix});
        try weights.put(wk_name, try mlx.random.normal(ctx, &[_]i32{ num_kv_heads * head_dim, hidden_size }, .float32, 0.0, 0.02, key_arr));

        const wv_name = try std.fmt.allocPrint(allocator, "{s}attention.wv", .{prefix});
        try weights.put(wv_name, try mlx.random.normal(ctx, &[_]i32{ num_kv_heads * head_dim, hidden_size }, .float32, 0.0, 0.02, key_arr));

        const wo_name = try std.fmt.allocPrint(allocator, "{s}attention.wo", .{prefix});
        try weights.put(wo_name, try mlx.random.normal(ctx, &[_]i32{ hidden_size, hidden_size }, .float32, 0.0, 0.02, key_arr));

        // post_attention_layernorm.weight
        const paln_name = try std.fmt.allocPrint(allocator, "{s}post_attention_layernorm", .{prefix});
        try weights.put(paln_name, try mlx.random.normal(ctx, &[_]i32{hidden_size}, .float32, 1.0, 0.0, key_arr));

        // mlp weights
        const gate_name = try std.fmt.allocPrint(allocator, "{s}mlp.gate_proj", .{prefix});
        try weights.put(gate_name, try mlx.random.normal(ctx, &[_]i32{ intermediate_size, hidden_size }, .float32, 0.0, 0.02, key_arr));

        const up_name = try std.fmt.allocPrint(allocator, "{s}mlp.up_proj", .{prefix});
        try weights.put(up_name, try mlx.random.normal(ctx, &[_]i32{ intermediate_size, hidden_size }, .float32, 0.0, 0.02, key_arr));

        const down_name = try std.fmt.allocPrint(allocator, "{s}mlp.down_proj", .{prefix});
        try weights.put(down_name, try mlx.random.normal(ctx, &[_]i32{ hidden_size, intermediate_size }, .float32, 0.0, 0.02, key_arr));
    }

    return weights;
}

test "tiny random model forward" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = mlx.models.LlamaConfig{
        .vocab_size = 128,
        .hidden_size = 32,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 4,
        .intermediate_size = 64,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
        .max_position_embeddings = 128,
    };

    var weights = try createRandomWeights(allocator, &config, ctx);
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

    // Forward pass with batch=1, seq_len=4
    const input_ids_data = [_]u32{ 1, 2, 3, 4 };
    const input_ids = try Array.fromData(allocator, u32, &input_ids_data, &[_]i32{ 1, 4 });
    defer input_ids.deinit();

    const logits = try model.forward(input_ids, null, null);
    defer logits.deinit();

    const shape = logits.shape();
    try std.testing.expectEqual(@as(i32, 1), shape[0]); // batch
    try std.testing.expectEqual(@as(i32, 4), shape[1]); // seq_len
    try std.testing.expectEqual(@as(i32, 128), shape[2]); // vocab_size
}

test "tiny random model generate" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = mlx.models.LlamaConfig{
        .vocab_size = 128,
        .hidden_size = 32,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 4,
        .intermediate_size = 64,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
        .max_position_embeddings = 128,
    };

    var weights = try createRandomWeights(allocator, &config, ctx);
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

    const layer_config = mlx.kvcache.LayerConfig{
        .batch_size = 1,
        .num_heads = config.num_attention_heads,
        .num_kv_heads = config.num_key_value_heads,
        .head_dim = config.hidden_size / config.num_attention_heads,
        .max_seq_len = config.max_position_embeddings,
        .dtype = .float32,
    };

    const prompt_data = [_]u32{ 1, 2, 3 };
    const prompt = try Array.fromData(allocator, u32, &prompt_data, &[_]i32{ 1, 3 });
    defer prompt.deinit();

    var sampler_config = mlx.sampling.SamplerConfig.init(42);
    sampler_config.temperature = 0.1;
    sampler_config.top_k = 10;
    sampler_config.top_p = 1.0;

    // Test generate without KV cache first
    const tokens_no_cache = try model.generate(prompt, 1, &sampler_config, &[_]mlx.kvcache.KVCacheStrategy{}, .{});
    defer allocator.free(tokens_no_cache);
    try std.testing.expectEqual(@as(usize, 4), tokens_no_cache.len);

    // Test generate with KV cache
    var caches = try allocator.alloc(mlx.kvcache.KVCacheStrategy, config.num_hidden_layers);
    defer {
        for (caches) |cache| cache.deinit(allocator);
        allocator.free(caches);
    }

    for (0..config.num_hidden_layers) |i| {
        caches[i] = try mlx.kvcache.createStandard(allocator, layer_config, stream);
    }

    const tokens = try model.generate(prompt, 1, &sampler_config, caches, .{});
    defer allocator.free(tokens);
    try std.testing.expectEqual(@as(usize, 4), tokens.len);
}

test "tiny random model forward with GQA" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = mlx.models.LlamaConfig{
        .vocab_size = 128,
        .hidden_size = 32,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .intermediate_size = 64,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
        .max_position_embeddings = 128,
    };

    var weights = try createRandomWeights(allocator, &config, ctx);
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

    const input_ids_data = [_]u32{ 1, 2, 3, 4 };
    const input_ids = try Array.fromData(allocator, u32, &input_ids_data, &[_]i32{ 1, 4 });
    defer input_ids.deinit();

    const logits = try model.forward(input_ids, null, null);
    defer logits.deinit();

    const shape = logits.shape();
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 4), shape[1]);
    try std.testing.expectEqual(@as(i32, 128), shape[2]);
}

test "tiny random model generate with GQA" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = mlx.models.LlamaConfig{
        .vocab_size = 128,
        .hidden_size = 32,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .intermediate_size = 64,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
        .max_position_embeddings = 128,
    };

    var weights = try createRandomWeights(allocator, &config, ctx);
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

    const layer_config = mlx.kvcache.LayerConfig{
        .batch_size = 1,
        .num_heads = config.num_attention_heads,
        .num_kv_heads = config.num_key_value_heads,
        .head_dim = config.hidden_size / config.num_attention_heads,
        .max_seq_len = config.max_position_embeddings,
        .dtype = .float32,
    };

    const prompt_data = [_]u32{ 1, 2, 3 };
    const prompt = try Array.fromData(allocator, u32, &prompt_data, &[_]i32{ 1, 3 });
    defer prompt.deinit();

    var sampler_config = mlx.sampling.SamplerConfig.init(42);
    sampler_config.temperature = 0.1;
    sampler_config.top_k = 10;
    sampler_config.top_p = 1.0;

    var caches = try allocator.alloc(mlx.kvcache.KVCacheStrategy, config.num_hidden_layers);
    defer {
        for (caches) |cache| cache.deinit(allocator);
        allocator.free(caches);
    }

    for (0..config.num_hidden_layers) |i| {
        caches[i] = try mlx.kvcache.createStandard(allocator, layer_config, stream);
    }

    const tokens = try model.generate(prompt, 1, &sampler_config, caches, .{});
    defer allocator.free(tokens);
    try std.testing.expectEqual(@as(usize, 4), tokens.len);
}
