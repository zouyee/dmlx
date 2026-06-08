/// Quick functional test for NativeEngineLoader.
/// Run with: zig build test-native-loader
const std = @import("std");
const loader_mod = @import("loader.zig");
const sf = @import("safetensors.zig");

test "config parse" {
    const model_path = "/Users/zouyee/models/DeepSeek-V4-Flash-4bit";
    var l = try loader_mod.NativeEngineLoader.init(std.testing.allocator, model_path);
    defer l.deinit();

    const cfg = try l.loadConfig();
    try std.testing.expectEqual(@as(u32, 43), cfg.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 129280), cfg.vocab_size);
    try std.testing.expectEqual(@as(u32, 256), cfg.n_routed_experts);
    std.log.info("Config ok: layers={d} vocab={d} experts={d}", .{
        cfg.num_hidden_layers, cfg.vocab_size, cfg.n_routed_experts,
    });
}

test "tensor index build and BF16 load" {
    const model_path = "/Users/zouyee/models/DeepSeek-V4-Flash-4bit";

    var idx = try sf.buildIndex(std.testing.allocator, model_path);
    defer idx.deinit();

    // Should have many tensors
    try std.testing.expect(idx.entries.count() > 100);
    std.log.info("Index: {d} tensors", .{idx.entries.count()});

    // Check key tensors exist
    const required = [_][]const u8{
        "model.layers.0.attn_norm.weight",
        "model.layers.0.attn.wq_a.weight",
        "model.layers.0.attn.wq_a.scales",
        "model.layers.0.attn.wq_a.biases",
        "model.layers.0.attn.attn_sink",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.layers.0.attn_hc.fn",
        "model.layers.0.ffn.gate.weight",
        "model.layers.0.ffn.shared_experts.gate_proj.weight",
        "model.layers.0.ffn.switch_mlp.gate_proj.weight",
    };
    for (required) |name| {
        const present = idx.get(name) != null;
        if (!present) std.log.err("MISSING tensor: {s}", .{name});
        try std.testing.expect(present);
    }

    // Load BF16 → F32 attn_norm layer 0
    const norm = try idx.loadBF16AsF32("model.layers.0.attn_norm.weight", std.testing.allocator);
    defer std.testing.allocator.free(norm);
    try std.testing.expectEqual(@as(usize, 4096), norm.len);
    // Values should be finite and reasonable for RMSNorm weights (near 1.0)
    for (norm[0..10]) |v| {
        try std.testing.expect(v > 0.0 and v < 10.0);
    }
    std.log.info("attn_norm[0..5]: {d:.4} {d:.4} {d:.4} {d:.4} {d:.4}", .{
        norm[0], norm[1], norm[2], norm[3], norm[4],
    });

    // Load U32 packed weight
    const wqa = try idx.loadU32("model.layers.0.attn.wq_a.weight", std.testing.allocator);
    defer std.testing.allocator.free(wqa);
    try std.testing.expectEqual(@as(usize, 1024 * 512), wqa.len);
    std.log.info("wq_a.weight[0]={d} (packed U32)", .{wqa[0]});
}
