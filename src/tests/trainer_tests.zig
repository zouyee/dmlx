const std = @import("std");
const mlx = @import("../root.zig");
const c = @import("../c.zig");

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

    const embed_name = try allocator.dupe(u8, "embed_tokens.weight");
    try weights.put(embed_name, try mlx.random.normal(ctx, &[_]i32{ vocab_size, hidden_size }, .float32, 0.0, 0.02, key_arr));

    const norm_name = try allocator.dupe(u8, "norm.weight");
    try weights.put(norm_name, try mlx.random.normal(ctx, &[_]i32{hidden_size}, .float32, 1.0, 0.0, key_arr));

    const lm_head_name = try allocator.dupe(u8, "lm_head.weight");
    try weights.put(lm_head_name, try mlx.random.normal(ctx, &[_]i32{ vocab_size, hidden_size }, .float32, 0.0, 0.02, key_arr));

    for (0..config.num_hidden_layers) |layer_idx| {
        const prefix = try std.fmt.allocPrint(allocator, "layers.{d}.", .{layer_idx});
        defer allocator.free(prefix);

        const iln_name = try std.fmt.allocPrint(allocator, "{s}input_layernorm", .{prefix});
        try weights.put(iln_name, try mlx.random.normal(ctx, &[_]i32{hidden_size}, .float32, 1.0, 0.0, key_arr));

        const wq_name = try std.fmt.allocPrint(allocator, "{s}attention.wq", .{prefix});
        try weights.put(wq_name, try mlx.random.normal(ctx, &[_]i32{ hidden_size, hidden_size }, .float32, 0.0, 0.02, key_arr));

        const wk_name = try std.fmt.allocPrint(allocator, "{s}attention.wk", .{prefix});
        try weights.put(wk_name, try mlx.random.normal(ctx, &[_]i32{ num_kv_heads * head_dim, hidden_size }, .float32, 0.0, 0.02, key_arr));

        const wv_name = try std.fmt.allocPrint(allocator, "{s}attention.wv", .{prefix});
        try weights.put(wv_name, try mlx.random.normal(ctx, &[_]i32{ num_kv_heads * head_dim, hidden_size }, .float32, 0.0, 0.02, key_arr));

        const wo_name = try std.fmt.allocPrint(allocator, "{s}attention.wo", .{prefix});
        try weights.put(wo_name, try mlx.random.normal(ctx, &[_]i32{ hidden_size, hidden_size }, .float32, 0.0, 0.02, key_arr));

        const paln_name = try std.fmt.allocPrint(allocator, "{s}post_attention_layernorm", .{prefix});
        try weights.put(paln_name, try mlx.random.normal(ctx, &[_]i32{hidden_size}, .float32, 1.0, 0.0, key_arr));

        const gate_name = try std.fmt.allocPrint(allocator, "{s}mlp.gate_proj", .{prefix});
        try weights.put(gate_name, try mlx.random.normal(ctx, &[_]i32{ intermediate_size, hidden_size }, .float32, 0.0, 0.02, key_arr));

        const up_name = try std.fmt.allocPrint(allocator, "{s}mlp.up_proj", .{prefix});
        try weights.put(up_name, try mlx.random.normal(ctx, &[_]i32{ intermediate_size, hidden_size }, .float32, 0.0, 0.02, key_arr));

        const down_name = try std.fmt.allocPrint(allocator, "{s}mlp.down_proj", .{prefix});
        try weights.put(down_name, try mlx.random.normal(ctx, &[_]i32{ hidden_size, intermediate_size }, .float32, 0.0, 0.02, key_arr));
    }

    return weights;
}

test "SFTTrainer trainStep on tiny model" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = mlx.models.LlamaConfig{
        .vocab_size = 64,
        .hidden_size = 16,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .num_key_value_heads = 2,
        .intermediate_size = 32,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
        .max_position_embeddings = 32,
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

    var optimizer = try mlx.optim.AdamW.initFromStruct(allocator, &model, 1e-3, 0.9, 0.999, 1e-8, 0.0, stream);
    defer optimizer.deinit();

    // Create trainer
    const trainer_config = mlx.trainer.TrainerConfig{
        .max_seq_len = config.max_position_embeddings,
        .lr_schedule = .{ .constant = .{ .lr = 1e-3 } },
    };
    var trainer = try mlx.trainer.SFTTrainer.init(allocator, &model, &optimizer, trainer_config, ctx, stream, null);
    defer trainer.deinit();

    // Create a simple training example: input [1, 2, 3], labels [1, 2, 3]
    const input_ids = try allocator.dupe(u32, &[_]u32{ 1, 2, 3 });
    defer allocator.free(input_ids);
    const labels = try allocator.dupe(i32, &[_]i32{ 1, 2, 3 });
    defer allocator.free(labels);

    const example = mlx.trainer.TrainingExample{
        .input_ids = input_ids,
        .labels = labels,
    };

    const loss = try trainer.trainStep(example);

    // Loss should be a finite positive number
    try std.testing.expect(loss > 0);
    try std.testing.expect(!std.math.isNan(loss));
}

test "SFTTrainer trainEpoch on tiny model" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = mlx.models.LlamaConfig{
        .vocab_size = 64,
        .hidden_size = 16,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .num_key_value_heads = 2,
        .intermediate_size = 32,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
        .max_position_embeddings = 32,
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

    var optimizer = try mlx.optim.AdamW.initFromStruct(allocator, &model, 1e-3, 0.9, 0.999, 1e-8, 0.0, stream);
    defer optimizer.deinit();

    const trainer_config = mlx.trainer.TrainerConfig{
        .max_seq_len = config.max_position_embeddings,
        .lr_schedule = .{ .constant = .{ .lr = 1e-3 } },
    };
    var trainer = try mlx.trainer.SFTTrainer.init(allocator, &model, &optimizer, trainer_config, ctx, stream, null);
    defer trainer.deinit();

    // Create a tiny dataset with 2 examples
    const ex1_input = try allocator.dupe(u32, &[_]u32{ 1, 2, 3 });
    const ex1_labels = try allocator.dupe(i32, &[_]i32{ 1, 2, 3 });
    const ex2_input = try allocator.dupe(u32, &[_]u32{ 4, 5, 6 });
    const ex2_labels = try allocator.dupe(i32, &[_]i32{ 4, 5, 6 });

    const examples = try allocator.alloc(mlx.trainer.TrainingExample, 2);
    examples[0] = .{ .input_ids = ex1_input, .labels = ex1_labels };
    examples[1] = .{ .input_ids = ex2_input, .labels = ex2_labels };

    const dataset = mlx.trainer.Dataset{ .examples = examples };
    defer dataset.deinit(allocator);

    try trainer.trainEpoch(dataset, 1);
}
