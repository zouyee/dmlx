/// Scratch: compare dmlx Qwen38Mtp.forward vs mlx-vlm drafter on a fixed
/// (hidden, token) probe input. Not committed.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const safetensors_reader = @import("mlx").safetensors_reader;
const qwen38_loader = @import("models/qwen38_loader.zig");
const qwen38 = @import("models/qwen38.zig");
const qwen38_mtp = @import("models/qwen38_mtp.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

var rng_state: u64 = 42;
fn nextGaussian() f32 {
    // deterministic LCG + Box-Muller-ish (matches nothing — only needs to be
    // shared with the python probe via the same seed formula)
    rng_state = rng_state *% 6364136223846793005 +% 1442695040888963407;
    const u: f32 = @as(f32, @floatFromInt(rng_state >> 33)) / @as(f32, @floatFromInt(@as(u64, 1) << 31));
    return (u - 0.5) * 2.0;
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

    // Same probe input as the python side: numpy default_rng(42) standard_normal
    // — reproduced here by reading the same values from a binary the python
    // probe writes? No — instead generate in Zig with an identical formula is
    // impossible; read the hidden vector from /tmp/probe_hidden.bin (written
    // by the python probe runner before this run).
    const hidden_bytes = try std.Io.Dir.cwd().readFileAlloc(io, "/tmp/probe_hidden.bin", allocator, .limited(1 << 20));
    const n_f32: usize = @intCast(cfg.hidden_size);
    const hidden_vals = try allocator.alloc(f32, n_f32);
    const src: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, hidden_bytes));
    @memcpy(hidden_vals, src[0..n_f32]);
    const hidden_f32 = try Array.fromData(allocator, f32, hidden_vals, &[_]i32{ 1, @intCast(n_f32) });
    defer hidden_f32.deinit();
    const hidden = try ops.astype(ctx, hidden_f32, .bfloat16);
    defer hidden.deinit();

    const out = try mtp.forward(allocator, hidden, 12345, model.embed_table_bf16.?, lm_head);
    defer out.logits.deinit();
    defer out.hidden_out.deinit();
    try out.logits.eval();
    try out.hidden_out.eval();

    const l32 = try ops.astype(ctx, out.logits, .float32);
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
    std.debug.print("DMLX drafter top6:", .{});
    for (top) |ti| std.debug.print(" ({d}, {d:.4})", .{ ti, data[ti] });
    std.debug.print("\n", .{});
    const h32 = try ops.astype(ctx, out.hidden_out, .float32);
    defer h32.deinit();
    try h32.eval();
    var chk: f64 = 0;
    for (try h32.dataSlice(f32)) |x| chk += x;
    std.debug.print("DMLX drafter hidden chk: {d}\n", .{chk});
}
