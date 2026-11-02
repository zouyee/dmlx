/// Scratch: end-to-end MTP correctness self-check (greedy, N=2 drafts, a few
/// rounds) + self-consistency re-score. Not committed.
///
/// Checks:
///  1. Run the real MTP flow (draft chain via Qwen38Mtp, verify via
///     forwardVerifyImpl, greedy acceptance, commit) for R rounds.
///  2. Reset the model and re-prefill prompt+emitted in ONE batched pass;
///     every emitted token must equal the argmax of the model's logits at
///     its position. Any mismatch = acceptance/state bug (vs benign bf16
///     near-tie flips between batched and decode-step scoring).
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

fn argmaxSlice(data: []const f32) u32 {
    var best: usize = 0;
    for (data[1..], 1..) |v, i| {
        if (v > data[best]) best = i;
    }
    return @intCast(best);
}

fn argmax(ctx: EagerContext, logits: Array) !u32 {
    const f32l = try ops.astype(ctx, logits, .float32);
    defer f32l.deinit();
    try f32l.eval();
    return argmaxSlice(try f32l.dataSlice(f32));
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

    const prompt = [_]u32{ 760, 6511, 314, 9338, 369 };

    // Full MTP-flow determinism probe: real drafter + verify + greedy accept,
    // 6 rounds. Print emitted tokens; run twice externally and compare.
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

    try model.reset();
    mtp.cache.deinit();
    mtp.cache = .{};
    var pf = try qwen38.forwardVerifyImpl(&model, &prompt);
    const L = pf.logits.shape()[1];
    const vocab: i32 = pf.logits.shape()[2];
    const pl_last = try ops.slice(ctx, pf.logits, &[_]i32{ 0, L - 1, 0 }, &[_]i32{ 1, L, vocab }, &[_]i32{ 1, 1, 1 });
    defer pl_last.deinit();
    const pl_flat = try ops.reshape(ctx, pl_last, &[_]i32{vocab});
    defer pl_flat.deinit();
    var t_next = try argmax(ctx, pl_flat);
    const pl_hidden = try ops.slice(ctx, pf.hidden_pre_norm_all, &[_]i32{ 0, L - 1, 0 }, &[_]i32{ 1, L, @intCast(cfg.hidden_size) }, &[_]i32{ 1, 1, 1 });
    defer pl_hidden.deinit();
    var hidden = try ops.reshape(ctx, pl_hidden, &[_]i32{ 1, @intCast(cfg.hidden_size) });
    qwen38.commitVerify(&model, &pf, @intCast(L));
    var j: usize = 0;
    while (j + 1 < prompt.len) : (j += 1) {
        const hj = try ops.slice(ctx, pf.hidden_pre_norm_all, &[_]i32{ 0, @intCast(j), 0 }, &[_]i32{ 1, @as(i32, @intCast(j)) + 1, @intCast(cfg.hidden_size) }, &[_]i32{ 1, 1, 1 });
        defer hj.deinit();
        const hj_flat = try ops.reshape(ctx, hj, &[_]i32{ 1, @intCast(cfg.hidden_size) });
        defer hj_flat.deinit();
        var d = try mtp.forward(allocator, hj_flat, prompt[j + 1], model.embed_table_bf16.?, lm_head);
        d.logits.deinit();
        d.hidden_out.deinit();
    }
    pf.deinit(allocator);

    var emitted = std.ArrayList(u32).empty;
    try emitted.append(allocator, t_next);
    const N: usize = 2;
    for (0..6) |_| {
        var drafts = std.ArrayList(u32).empty;
        var h_cur = hidden;
        var tok = t_next;
        for (0..N) |_| {
            var d = try mtp.forward(allocator, h_cur, tok, model.embed_table_bf16.?, lm_head);
            if (drafts.items.len > 0) h_cur.deinit();
            h_cur = d.hidden_out;
            const d_tok = try argmax(ctx, d.logits);
            d.logits.deinit();
            try drafts.append(allocator, d_tok);
            tok = d_tok;
        }
        var vids = std.ArrayList(u32).empty;
        try vids.append(allocator, t_next);
        try vids.appendSlice(allocator, drafts.items);
        var v = try qwen38.forwardVerifyImpl(&model, vids.items);
        var k: usize = 0;
        var t_prime: u32 = 0;
        for (0..N) |i| {
            const col = try ops.slice(ctx, v.logits, &[_]i32{ 0, @intCast(i), 0 }, &[_]i32{ 1, @as(i32, @intCast(i)) + 1, vocab }, &[_]i32{ 1, 1, 1 });
            defer col.deinit();
            const flat = try ops.reshape(ctx, col, &[_]i32{vocab});
            defer flat.deinit();
            const target = try argmax(ctx, flat);
            if (target != drafts.items[i]) {
                t_prime = target;
                break;
            }
            k += 1;
        }
        if (k == N) {
            const col = try ops.slice(ctx, v.logits, &[_]i32{ 0, @intCast(N), 0 }, &[_]i32{ 1, @as(i32, N) + 1, vocab }, &[_]i32{ 1, 1, 1 });
            defer col.deinit();
            const flat = try ops.reshape(ctx, col, &[_]i32{vocab});
            defer flat.deinit();
            t_prime = try argmax(ctx, flat);
            var d = try mtp.forward(allocator, h_cur, drafts.items[N - 1], model.embed_table_bf16.?, lm_head);
            d.logits.deinit();
            d.hidden_out.deinit();
        }
        h_cur.deinit();
        qwen38.commitVerify(&model, &v, k + 1);
        const hrow = try ops.slice(ctx, v.hidden_pre_norm_all, &[_]i32{ 0, @intCast(k), 0 }, &[_]i32{ 1, @as(i32, @intCast(k)) + 1, @intCast(cfg.hidden_size) }, &[_]i32{ 1, 1, 1 });
        defer hrow.deinit();
        const hflat = try ops.reshape(ctx, hrow, &[_]i32{ 1, @intCast(cfg.hidden_size) });
        defer hflat.deinit();
        const hidden_next = try ops.copy(ctx, hflat);
        v.deinit(allocator);
        for (drafts.items[0..k]) |d| try emitted.append(allocator, d);
        try emitted.append(allocator, t_prime);
        hidden.deinit();
        hidden = hidden_next;
        t_next = t_prime;
    }
    hidden.deinit();
    std.debug.print("emitted:", .{});
    for (emitted.items) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});
}
