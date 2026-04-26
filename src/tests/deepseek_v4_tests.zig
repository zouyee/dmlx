const std = @import("std");
const deepseek_v4 = @import("../models/deepseek_v4.zig");
const ops = @import("../ops.zig");
const array_mod = @import("../array.zig");
const device_mod = @import("../device.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

test "ops.slice on 1D array" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const data = &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const arr = try Array.fromData(allocator, f32, data, &[_]i32{ 8 });
    defer arr.deinit();

    const sliced = try ops.slice(ctx, arr, &[_]i32{ 2 }, &[_]i32{ 6 }, &[_]i32{ 1 });
    defer sliced.deinit();
    
    try std.testing.expectEqual(@as(i32, 4), sliced.shape()[0]);
}

test "ops.slice on 3D array" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const data = &[_]f32{
        1,  2,  3,  4,  5,  6,  7,  8,
        9,  10, 11, 12, 13, 14, 15, 16,
    };
    const kv = try Array.fromData(allocator, f32, data, &[_]i32{ 1, 8, 2 });
    defer kv.deinit();

    const suffix = try ops.slice(ctx, kv, &[_]i32{ 0, 4, 0 }, &[_]i32{ 1, 8, 2 }, &[_]i32{ 1, 1, 1 });
    defer suffix.deinit();
    
    try std.testing.expectEqual(@as(i32, 1), suffix.shape()[0]);
    try std.testing.expectEqual(@as(i32, 4), suffix.shape()[1]);
    try std.testing.expectEqual(@as(i32, 2), suffix.shape()[2]);
}

test "compressKV passthrough when ratio <= 1" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // [B=1, S=4, D=2]
    const data = &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const kv = try Array.fromData(allocator, f32, data, &[_]i32{ 1, 4, 2 });
    defer kv.deinit();

    const result = try deepseek_v4.compressKV(ctx, kv, 1, 2, null, null, ctx.stream.inner);
    defer result.deinit();

    const shape = result.shape();
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 4), shape[1]);
    try std.testing.expectEqual(@as(i32, 2), shape[2]);
}

test "compressKV mean pools prefix and keeps window" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // [B=1, S=8, D=2]
    const data = &[_]f32{
        1,  2,  3,  4,  5,  6,  7,  8,
        9,  10, 11, 12, 13, 14, 15, 16,
    };
    const kv = try Array.fromData(allocator, f32, data, &[_]i32{ 1, 8, 2 });
    defer kv.deinit();

    const result = try deepseek_v4.compressKV(ctx, kv, 2, 4, null, null, ctx.stream.inner);
    defer result.deinit();

    const shape = result.shape();
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 6), shape[1]);
    try std.testing.expectEqual(@as(i32, 2), shape[2]);
}

test "compressKV handles remainder" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const data = &[_]f32{
        1,  2,  3,  4,  5,  6,  7,  8,
        9,  10, 11, 12, 13, 14,
    };
    const kv = try Array.fromData(allocator, f32, data, &[_]i32{ 1, 7, 2 });
    defer kv.deinit();

    const result = try deepseek_v4.compressKV(ctx, kv, 2, 4, null, null, ctx.stream.inner);
    defer result.deinit();

    const shape = result.shape();
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 6), shape[1]);
    try std.testing.expectEqual(@as(i32, 2), shape[2]);
}

test "compressKV no-op when seq_len <= window" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // [B=1, S=4, D=2]
    const data = &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const kv = try Array.fromData(allocator, f32, data, &[_]i32{ 1, 4, 2 });
    defer kv.deinit();

    // ratio=4, window=8 -> seq_len (4) <= window (8), passthrough
    const result = try deepseek_v4.compressKV(ctx, kv, 4, 8, null, null, ctx.stream.inner);
    defer result.deinit();

    const shape = result.shape();
    try std.testing.expectEqual(@as(i32, 4), shape[1]);
}

test "compressKV softmax-gated pooling with learned weights" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // [B=1, S=8, D=2], compress_ratio=2, window=4
    // prefix_len=4, num_groups=2, remainder=0
    const data = &[_]f32{
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16,
    };
    const kv = try Array.fromData(allocator, f32, data, &[_]i32{ 1, 8, 2 });
    defer kv.deinit();

    // gate_weight: [compress_ratio=2, D=2]
    const gw_data = &[_]f32{ 1, 0, 0, 1 };
    const gate_weight = try Array.fromData(allocator, f32, gw_data, &[_]i32{ 2, 2 });
    defer gate_weight.deinit();

    // pos_bias: [compress_ratio=2]
    const pb_data = &[_]f32{ 0, 0 };
    const pos_bias = try Array.fromData(allocator, f32, pb_data, &[_]i32{2});
    defer pos_bias.deinit();

    const result = try deepseek_v4.compressKV(ctx, kv, 2, 4, gate_weight, pos_bias, ctx.stream.inner);
    defer result.deinit();

    const shape = result.shape();
    // 2 compressed groups + 4 window = 6
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 6), shape[1]);
    try std.testing.expectEqual(@as(i32, 2), shape[2]);
}

test "compressKV softmax-gated pooling with remainder" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // [B=1, S=7, D=2], compress_ratio=2, window=4
    // prefix_len=3, num_groups=1, remainder=1
    const data = &[_]f32{
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14,
    };
    const kv = try Array.fromData(allocator, f32, data, &[_]i32{ 1, 7, 2 });
    defer kv.deinit();

    // gate_weight: [compress_ratio=2, D=2]
    const gw_data = &[_]f32{ 1, 0, 0, 1 };
    const gate_weight = try Array.fromData(allocator, f32, gw_data, &[_]i32{ 2, 2 });
    defer gate_weight.deinit();

    // pos_bias: [compress_ratio=2]
    const pb_data = &[_]f32{ 0, 0 };
    const pos_bias = try Array.fromData(allocator, f32, pb_data, &[_]i32{2});
    defer pos_bias.deinit();

    const result = try deepseek_v4.compressKV(ctx, kv, 2, 4, gate_weight, pos_bias, ctx.stream.inner);
    defer result.deinit();

    const shape = result.shape();
    // 1 compressed group + 1 remainder + 4 window = 6
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 6), shape[1]);
    try std.testing.expectEqual(@as(i32, 2), shape[2]);
}

test "expandToMHC expands to correct shape" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const data = &[_]f32{ 1, 2, 3, 4, 5, 6 };
    const hidden = try Array.fromData(allocator, f32, data, &[_]i32{ 1, 2, 3 });
    defer hidden.deinit();

    const expanded = try deepseek_v4.expandToMHC(ctx, hidden, 4, ctx.stream.inner);
    defer expanded.deinit();

    const shape = expanded.shape();
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 2), shape[1]);
    try std.testing.expectEqual(@as(i32, 4), shape[2]);
    try std.testing.expectEqual(@as(i32, 3), shape[3]);
}

test "sinkhornNormalize produces doubly stochastic" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // [B=1, S=1, M=4, M=4] with random-ish values
    const data = &[_]f32{
        1, 2, 3, 4,
        4, 3, 2, 1,
        1, 1, 1, 1,
        2, 2, 2, 2,
    };
    const x = try Array.fromData(allocator, f32, data, &[_]i32{ 1, 1, 4, 4 });
    defer x.deinit();

    const result = try deepseek_v4.sinkhornNormalize(ctx, x, 10, 1e-6, ctx.stream.inner);
    defer result.deinit();

    const shape = result.shape();
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 1), shape[1]);
    try std.testing.expectEqual(@as(i32, 4), shape[2]);
    try std.testing.expectEqual(@as(i32, 4), shape[3]);
}

test "mhcPreApplyMix computes weighted sum" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // residual [B=1, S=1, mhc=2, H=2]
    const res_data = &[_]f32{ 1, 2, 3, 4 };
    const residual = try Array.fromData(allocator, f32, res_data, &[_]i32{ 1, 1, 2, 2 });
    defer residual.deinit();

    // mix [B=1, S=1, mhc=2, 1]
    const mix_data = &[_]f32{ 0.5, 0.5 };
    const mix = try Array.fromData(allocator, f32, mix_data, &[_]i32{ 1, 1, 2, 1 });
    defer mix.deinit();

    const result = try deepseek_v4.mhcPreApplyMix(ctx, residual, mix, ctx.stream.inner);
    defer result.deinit();

    const shape = result.shape();
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 1), shape[1]);
    try std.testing.expectEqual(@as(i32, 2), shape[2]); // H=2
}

test "mhcPost combines output with residual" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // x [B=1, S=1, H=2]
    const x_data = &[_]f32{ 1, 2 };
    const x = try Array.fromData(allocator, f32, x_data, &[_]i32{ 1, 1, 2 });
    defer x.deinit();

    // residual [B=1, S=1, mhc=2, H=2]
    const res_data = &[_]f32{ 1, 2, 3, 4 };
    const residual = try Array.fromData(allocator, f32, res_data, &[_]i32{ 1, 1, 2, 2 });
    defer residual.deinit();

    // post_mix [B=1, S=1, mhc=2, 1]
    const post_data = &[_]f32{ 1, 1 };
    const post_mix = try Array.fromData(allocator, f32, post_data, &[_]i32{ 1, 1, 2, 1 });
    defer post_mix.deinit();

    // comb_mix [B=1, S=1, mhc=2, mhc=2]
    const comb_data = &[_]f32{ 0.5, 0.5, 0.5, 0.5 };
    const comb_mix = try Array.fromData(allocator, f32, comb_data, &[_]i32{ 1, 1, 2, 2 });
    defer comb_mix.deinit();

    const result = try deepseek_v4.mhcPost(ctx, x, residual, post_mix, comb_mix, ctx.stream.inner);
    defer result.deinit();

    const shape = result.shape();
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 1), shape[1]);
    try std.testing.expectEqual(@as(i32, 2), shape[2]);
    try std.testing.expectEqual(@as(i32, 2), shape[3]);
}

test "mhcPreNormFn computes mixes" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const res_data = &[_]f32{ 1, 2, 3, 4 };
    const residual = try Array.fromData(allocator, f32, res_data, &[_]i32{ 1, 1, 2, 2 });
    defer residual.deinit();

    const fn_data = &[_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1,
    };
    const fn_arr = try Array.fromData(allocator, f32, fn_data, &[_]i32{ 8, 4 });
    defer fn_arr.deinit();

    const mixes = try deepseek_v4.mhcPreNormFn(ctx, residual, fn_arr, null, 1e-6, ctx.stream.inner);
    defer mixes.deinit();

    const shape = mixes.shape();
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 1), shape[1]);
    try std.testing.expectEqual(@as(i32, 8), shape[2]);
}

test "mhcPreSplitMixes splits correctly" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const mix_data = &[_]f32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const mixes = try Array.fromData(allocator, f32, mix_data, &[_]i32{ 1, 1, 8 });
    defer mixes.deinit();

    const scale = try Array.fromData(allocator, f32, &[_]f32{ 1, 1, 1 }, &[_]i32{3});
    defer scale.deinit();
    const base = try Array.fromData(allocator, f32, &[_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 }, &[_]i32{8});
    defer base.deinit();

    const pre_mix, const post_mix, const comb_mix = try deepseek_v4.mhcPreSplitMixes(ctx, mixes, scale, base, 2, 1.0, 1e-6, ctx.stream.inner);
    defer pre_mix.deinit();
    defer post_mix.deinit();
    defer comb_mix.deinit();

    try std.testing.expectEqual(@as(i32, 2), pre_mix.shape()[2]);
    try std.testing.expectEqual(@as(i32, 1), pre_mix.shape()[3]);
    try std.testing.expectEqual(@as(i32, 2), post_mix.shape()[2]);
    try std.testing.expectEqual(@as(i32, 2), comb_mix.shape()[2]);
    try std.testing.expectEqual(@as(i32, 2), comb_mix.shape()[3]);
}

test "DSV4HyperConn pre/post roundtrip" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const res_data = &[_]f32{ 1, 2, 3, 4 };
    const residual = try Array.fromData(allocator, f32, res_data, &[_]i32{ 1, 1, 2, 2 });
    defer residual.deinit();

    const fn_data = &[_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1,
    };
    const fn_arr = try Array.fromData(allocator, f32, fn_data, &[_]i32{ 8, 4 });
    defer fn_arr.deinit();

    const base = try Array.fromData(allocator, f32, &[_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 }, &[_]i32{8});
    defer base.deinit();
    const scale = try Array.fromData(allocator, f32, &[_]f32{ 1, 1, 1 }, &[_]i32{3});
    defer scale.deinit();

    var hc = deepseek_v4.DSV4HyperConn{
        .hc_fn = fn_arr,
        .hc_base = base,
        .hc_scale = scale,
        .hc_mult = 2,
        .hc_sinkhorn_iters = 5,
        .hc_eps = 1e-6,
    };

    const layer_input, const post_mix, const comb_mix = try hc.pre(ctx, residual, ctx.stream.inner);
    defer layer_input.deinit();
    defer post_mix.deinit();
    defer comb_mix.deinit();

    const shape = layer_input.shape();
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 1), shape[1]);
    try std.testing.expectEqual(@as(i32, 2), shape[2]);

    const x_data = &[_]f32{ 1, 2 };
    const x = try Array.fromData(allocator, f32, x_data, &[_]i32{ 1, 1, 2 });
    defer x.deinit();

    const result = try hc.post(ctx, x, residual, post_mix, comb_mix, ctx.stream.inner);
    defer result.deinit();

    const out_shape = result.shape();
    try std.testing.expectEqual(@as(i32, 1), out_shape[0]);
    try std.testing.expectEqual(@as(i32, 1), out_shape[1]);
    try std.testing.expectEqual(@as(i32, 2), out_shape[2]);
    try std.testing.expectEqual(@as(i32, 2), out_shape[3]);
}

test "DSV4Model dummy build and forward+generate" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const loader = @import("../models/deepseek_v4_loader.zig");
    const kvcache_mod = @import("../kvcache.zig");
    const standard = @import("../kvcache/standard.zig");
    const sampling = @import("../sampling.zig");

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

    var model = try loader.buildDSV4Model(allocator, &config, &weights, ctx, ctx.stream.inner);
    defer model.deinit();

    const input_ids = try Array.fromData(allocator, u32, &[_]u32{ 1, 2, 3 }, &[_]i32{ 1, 3 });
    defer input_ids.deinit();

    const logits = try model.forward(input_ids, null, null, 0, ctx.stream.inner);
    defer logits.deinit();

    const ls = logits.shape();
    try std.testing.expectEqual(@as(i32, 1), ls[0]);
    try std.testing.expectEqual(@as(i32, 3), ls[1]);
    try std.testing.expectEqual(@as(i32, 8), ls[2]);

    var cache0 = try standard.StandardKVCache.init(allocator, .{
        .batch_size = 1,
        .num_heads = 2,
        .num_kv_heads = 1,
        .head_dim = 8,
        .max_seq_len = 32,
        .dtype = .float32,
    }, ctx.stream.inner);
    defer {
        cache0.keys.deinit();
        cache0.values.deinit();
    }

    var caches = [_]kvcache_mod.KVCacheStrategy{cache0.asStrategy()};
    var sampler = sampling.SamplerConfig.init(42);

    const generated = try model.generate(&[_]u32{ 1, 2 }, 3, &sampler, &caches, ctx.stream.inner);
    defer allocator.free(generated);

    try std.testing.expectEqual(@as(usize, 3), generated.len);
}


test "DSV4Model with MHC enabled" {
    const loader = @import("../models/deepseek_v4_loader.zig");
    const allocator = std.testing.allocator;
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
        .use_mhc = true,
        .hc_mult = 2,
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

    // MHC weights: hc_mult=2 => mhc_mult3 = 2*(2+2) = 8, mhc_mult*H = 32
    try W.put(&weights, allocator, "layers.0.hc_attn_fn", &[_]i32{ 8, 32 });
    try W.put(&weights, allocator, "layers.0.hc_attn_base", &[_]i32{8});
    try W.put(&weights, allocator, "layers.0.hc_attn_scale", &[_]i32{3});
    try W.put(&weights, allocator, "layers.0.hc_ffn_fn", &[_]i32{ 8, 32 });
    try W.put(&weights, allocator, "layers.0.hc_ffn_base", &[_]i32{8});
    try W.put(&weights, allocator, "layers.0.hc_ffn_scale", &[_]i32{3});

    var model = try loader.buildDSV4Model(allocator, &config, &weights, ctx, ctx.stream.inner);
    defer model.deinit();

    const input_ids = try Array.fromData(allocator, u32, &[_]u32{ 1, 2, 3 }, &[_]i32{ 1, 3 });
    defer input_ids.deinit();

    const logits = try model.forward(input_ids, null, null, 0, ctx.stream.inner);
    defer logits.deinit();

    const ls = logits.shape();
    try std.testing.expectEqual(@as(i32, 1), ls[0]);
    try std.testing.expectEqual(@as(i32, 3), ls[1]);
    try std.testing.expectEqual(@as(i32, 8), ls[2]);
}
