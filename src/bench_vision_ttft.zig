/// Scratch: stage-level wall-clock breakdown of the Qwen3.8 vision TTFT.
/// Usage: zig build run-bench-vision-ttft -Doptimize=ReleaseFast -- <image>...
/// Prints per image: preprocess ms (incl eval), tower ms (incl eval),
/// vision prefill ms (logits eval), and a same-length text-only prefill
/// reference so tower+merge cost can be separated from LM prefill.
const std = @import("std");
const mlx = @import("mlx");
const c = mlx.c;
const Array = mlx.Array;
const ops = mlx.ops;
const EagerContext = ops.EagerContext;
const qwen38_loader = @import("models/qwen38_loader.zig");
const qwen38 = @import("models/qwen38.zig");
const vit_mod = @import("vision/qwen38_vit.zig");
const preprocess = @import("vision/qwen38_preprocess.zig");
const safetensors_reader = mlx.safetensors_reader;

const MODEL_DIR = "/Users/zouyee/models/Qwen3.8-27B-4bit";

fn loadPrefixedWeights(
    allocator: std.mem.Allocator,
    index: *safetensors_reader.TensorIndex,
    prefix: []const u8,
) !std.StringHashMap(Array) {
    var map = std.StringHashMap(Array).init(allocator);
    errdefer map.deinit();
    var it = index.entries.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (!std.mem.startsWith(u8, name, prefix)) continue;
        try map.put(try allocator.dupe(u8, name), try index.loadTensor(name));
    }
    return map;
}

fn msSince(io: std.Io, start: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(std.Io.Timestamp.durationTo(start, std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e6;
}

pub fn main(init: std.process.Init) !void {
    c.initErrorHandler();
    const aa = init.arena.allocator();
    const io = init.io;

    const argv = try init.minimal.args.toSlice(aa);
    if (argv.len < 2) return error.Usage;
    const images = argv[1..];

    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(aa, .{ .inner = stream });

    const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, MODEL_DIR ++ "/config.json", aa, .limited(1 << 20));
    var cfg = try qwen38_loader.parseQwen38Config(aa, cfg_text);
    var index = try safetensors_reader.loadOrBuildIndex(aa, MODEL_DIR);

    const weights = try loadPrefixedWeights(aa, &index, "language_model.");
    var model = try qwen38.Qwen38ServingModel.init(aa, ctx, &cfg, weights);
    model.ws = qwen38.WeightSource.fromMap(&model.weights, ctx, &model.ws_backing);

    var vision_map = try loadPrefixedWeights(aa, &index, "vision_tower.");
    const vcfg = vit_mod.Qwen38VitConfig{
        .depth = cfg.vision.depth,
        .hidden = @intCast(cfg.vision.hidden),
        .heads = @intCast(cfg.vision.heads),
        .intermediate = @intCast(cfg.vision.intermediate),
        .patch = @intCast(cfg.vision.patch),
        .temporal_patch = @intCast(cfg.vision.temporal_patch),
        .merge = @intCast(cfg.vision.merge),
        .out_hidden = @intCast(cfg.vision.out_hidden),
        .num_pos = @intCast(cfg.vision.num_pos),
        .in_ch = @intCast(cfg.vision.in_ch),
    };
    var vit = try vit_mod.Qwen38Vit.init(aa, ctx, vcfg, &vision_map);

    // Warmup: one vision pass to trigger all JIT compiles.
    {
        var pp = try preprocess.preprocessImageFile(aa, ctx, images[0]);
        defer pp.deinit();
        const feats = try vit.forward(pp.pixels, pp.grid_thw[0], pp.grid_thw[1], pp.grid_thw[2]);
        defer feats.deinit();
        try feats.eval();
        model.reset() catch {};
    }

    for (0..2) |round| {
        if (round > 0) std.debug.print("--- warm round ---\n", .{});
        for (images) |path| {
            var mark = std.Io.Timestamp.now(io, .awake);

            var pp = try preprocess.preprocessImageFile(aa, ctx, path);
            defer pp.deinit();
            try pp.pixels.eval();
            const t_pre = msSince(io, mark);
            mark = std.Io.Timestamp.now(io, .awake);

            const feats = try vit.forward(pp.pixels, pp.grid_thw[0], pp.grid_thw[1], pp.grid_thw[2]);
            defer feats.deinit();
            try feats.eval();
            const t_tower = msSince(io, mark);
            mark = std.Io.Timestamp.now(io, .awake);

            // Prompt: <vs> pads <ve> + a fixed text tail, mirroring a chat prompt.
            const n_text: usize = 40;
            const L = pp.n_image_pad + 2 + n_text;
            const ids = try aa.alloc(u32, L);
            ids[0] = cfg.vision_start_token_id;
            for (0..pp.n_image_pad) |i| ids[1 + i] = cfg.image_token_id;
            ids[1 + pp.n_image_pad] = cfg.vision_end_token_id;
            for (0..n_text) |i| ids[2 + pp.n_image_pad + i] = 1000 + @as(u32, @intCast(i % 50));

            const blocks = [_][3]i32{.{ pp.grid_thw[0], @divExact(pp.grid_thw[1], 2), @divExact(pp.grid_thw[2], 2) }};
            const logits = try model.forwardTokensWithVision(ids, .{ .features = feats, .blocks = &blocks });
            defer logits.deinit();
            try logits.eval();
            const t_prefill = msSince(io, mark);
            mark = std.Io.Timestamp.now(io, .awake);

            // Text-only reference at the same L.
            try model.reset();
            const tids = try aa.alloc(u32, L);
            for (tids, 0..) |*x, i| x.* = 1000 + @as(u32, @intCast(i % 50));
            const tlogits = try model.forwardTokensLast(tids);
            defer tlogits.deinit();
            try tlogits.eval();
            const t_text = msSince(io, mark);

            std.debug.print("{s}: grid={d}x{d} pads={d} L={d} | preprocess {d:.1} ms, tower {d:.1} ms, vision-prefill {d:.1} ms, text-prefill {d:.1} ms\n", .{
                path, pp.grid_thw[1], pp.grid_thw[2], pp.n_image_pad, L, t_pre, t_tower, t_prefill, t_text,
            });
        }
    }
}
