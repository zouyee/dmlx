/// Benchmark: load all weights and report time + memory.
/// Run with: zig build run-load-bench
const std = @import("std");
const loader_mod = @import("loader.zig");

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const model_path = "/Users/zouyee/models/DeepSeek-V4-Flash-4bit";

    var l = try loader_mod.NativeEngineLoader.init(allocator, model_path);
    defer l.deinit();

    const cfg = try l.loadConfig();
    std.debug.print("Config: {d} layers, vocab {d}, {d} experts\n", .{
        cfg.num_hidden_layers, cfg.vocab_size, cfg.n_routed_experts,
    });

    // 60s timeout
    std.debug.print("Loading weights (timeout=60s)...\n", .{});
    var store = try l.loadWeights(cfg, 60_000);
    defer store.deinit();

    std.debug.print("Load complete.\n", .{});
    std.debug.print("  embed[0..3]: {d:.4} {d:.4} {d:.4}\n", .{
        store.embed_f32[0], store.embed_f32[1], store.embed_f32[2],
    });
    std.debug.print("  lm_head[0..3]: {d:.4} {d:.4} {d:.4}\n", .{
        store.lm_head_f32[0], store.lm_head_f32[1], store.lm_head_f32[2],
    });
    std.debug.print("  final_norm[0..3]: {d:.4} {d:.4} {d:.4}\n", .{
        store.weights.final_norm[0], store.weights.final_norm[1], store.weights.final_norm[2],
    });
}
