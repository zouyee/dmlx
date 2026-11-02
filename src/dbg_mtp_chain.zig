/// Chained-draft probe: mirrors /tmp/mtp_chain_ref.py (mlx_vlm's qwen3_5
/// MTPModule — the exact head math omlx serves) step for step. Both sides
/// start from the same synthetic hidden vector (/tmp/probe_hidden.bin, numpy
/// default_rng(42).standard_normal(5120)) and token 12345, chain 6 steps on
/// the head's own post-norm output + argmax token, and print per-step top-6
/// logits + hidden checksum. Divergence isolates a head-math defect from
/// content drift (cross-engine greedy streams diverge too early for
/// position-aligned draft comparison).
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const reduce_mod = @import("mlx").reduce;
const safetensors_reader = @import("mlx").safetensors_reader;
const qwen38_loader = @import("models/qwen38_loader.zig");
const qwen38 = @import("models/qwen38.zig");
const qwen38_mtp = @import("models/qwen38_mtp.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

fn top6(ctx: EagerContext, logits: Array) !struct { ids: [6]usize, vals: [6]f32 } {
    const l32 = try ops.astype(ctx, logits, .float32);
    defer l32.deinit();
    try l32.eval();
    const data = try l32.dataSlice(f32);
    var top: [6]usize = .{ 0, 0, 0, 0, 0, 0 };
    for (data, 0..) |v, i| {
        var j: usize = 6;
        while (j > 0 and v > data[top[j - 1]]) : (j -= 1) {}
        if (j < 6) {
            var m: usize = 5;
            while (m > j) : (m -= 1) top[m] = top[m - 1];
            top[j] = i;
        }
    }
    var vals: [6]f32 = undefined;
    for (top, 0..) |ti, i| vals[i] = data[ti];
    return .{ .ids = top, .vals = vals };
}

pub fn main(init: std.process.Init) !void {
    c.initErrorHandler();
    const allocator = init.arena.allocator();
    const io = init.io;

    const model_dir = "/Users/zouyee/models/Qwen3.8-27B-4bit";
    const cfg_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, allocator, .limited(1024 * 1024));
    var cfg = try qwen38_loader.parseQwen38Config(allocator, cfg_text);
    defer cfg.deinit();

    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    var idx = try safetensors_reader.loadOrBuildIndex(allocator, model_dir);
    defer idx.deinit();
    var weights = std.StringHashMap(Array).init(allocator);
    var it = idx.entries.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (!std.mem.startsWith(u8, name, "language_model.")) continue;
        try weights.put(try allocator.dupe(u8, name), try idx.loadTensor(name));
    }
    var model = try qwen38.Qwen38ServingModel.init(allocator, ctx, &cfg, weights);
    model.ws = qwen38.WeightSource.fromMap(&model.weights, ctx, &model.ws_backing);

    var mtp = try qwen38_mtp.Qwen38Mtp.load(allocator, ctx, "/Users/zouyee/models/Qwen3.8-27B-MTP-4bit", .{
        .hidden_size = cfg.hidden_size,
        .num_heads = cfg.num_attention_heads,
        .num_kv_heads = cfg.num_key_value_heads,
        .head_dim = cfg.head_dim,
        .rotary_dims = cfg.rotaryDims(),
        .intermediate_size = cfg.intermediate_size,
        .rope_theta = cfg.rope_theta,
    });
    defer mtp.deinit();
    const lm = model.lm_head.?;
    const lm_head = qwen38_mtp.MtpLinear{ .w = lm.w, .scales = lm.scales, .biases = lm.biases };

    const hidden_bytes = try std.Io.Dir.cwd().readFileAlloc(io, "/tmp/probe_hidden.bin", allocator, .limited(1 << 20));
    const n_f32: usize = @intCast(cfg.hidden_size);
    const hidden_vals = try allocator.alloc(f32, n_f32);
    @memcpy(std.mem.sliceAsBytes(hidden_vals), hidden_bytes[0 .. n_f32 * 4]);
    const hidden_f32 = try Array.fromData(allocator, f32, hidden_vals, &[_]i32{ 1, @intCast(n_f32) });
    defer hidden_f32.deinit();

    var hidden = try ops.astype(ctx, hidden_f32, .bfloat16);
    defer hidden.deinit(); // outer scope: holds the CURRENT value at exit
    var token: u32 = 12345;
    for (0..6) |step| {
        const out = try mtp.forward(allocator, hidden, token, model.embed_table_bf16.?, lm_head);
        defer out.logits.deinit();
        const t6 = try top6(ctx, out.logits);
        {
            const h32 = try ops.astype(ctx, out.hidden_out, .float32);
            defer h32.deinit();
            try h32.eval();
            var chk: f64 = 0;
            for (try h32.dataSlice(f32)) |x| chk += x;
            std.debug.print("step{d} top6=", .{step});
            for (t6.ids, t6.vals) |ti, tv| std.debug.print(" ({d}, {d:.4})", .{ ti, tv });
            std.debug.print(" hchk={d:.6}\n", .{chk});
        }
        token = @intCast(t6.ids[0]);
        // Chain: the head's post-norm output becomes the next step's input.
        const old = hidden;
        hidden = out.hidden_out;
        old.deinit();
    }
}
