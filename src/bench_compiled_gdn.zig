// Scratch micro-benchmark: compiled GDN step apply cost at real model shapes.
// Not committed.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const random = @import("mlx").random;
const eval_mod = @import("mlx").eval;
const qwen38_gdn = @import("models/qwen38_gdn.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

fn elapsedMs(start: u64) f64 {
    const ns = (std.c.mach_absolute_time() - start) * 125 / 3; // Apple Silicon timebase
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn randW(ctx: EagerContext, shape: []const i32) !Array {
    return random.normal(ctx, shape, .float32, 0.0, 0.02, null);
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    c.initErrorHandler();
    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    // Real Qwen3.8 GDN dims.
    const hidden: i32 = 5120;
    const HK: i32 = 16;
    const HV: i32 = 48;
    const D: i32 = 128;
    const K: i32 = 4;
    const QKV: i32 = (2 * HK + HV) * D;

    const x = try randW(ctx, &[_]i32{ 1, hidden });
    defer x.deinit();
    const conv_state = try randW(ctx, &[_]i32{ K - 1, QKV });
    defer conv_state.deinit();
    const ssm_state = try randW(ctx, &[_]i32{ HV, D, D });
    defer ssm_state.deinit();
    const qkv_w = try randW(ctx, &[_]i32{ QKV, hidden });
    defer qkv_w.deinit();
    const conv1d_t = try randW(ctx, &[_]i32{ K, QKV });
    defer conv1d_t.deinit();
    const a_w = try randW(ctx, &[_]i32{ HV, hidden });
    defer a_w.deinit();
    const b_w = try randW(ctx, &[_]i32{ HV, hidden });
    defer b_w.deinit();
    const z_w = try randW(ctx, &[_]i32{ HV * D, hidden });
    defer z_w.deinit();
    const A_log = try randW(ctx, &[_]i32{HV});
    defer A_log.deinit();
    const dt_bias = try randW(ctx, &[_]i32{HV});
    defer dt_bias.deinit();
    const norm_w = try randW(ctx, &[_]i32{D});
    defer norm_w.deinit();
    const out_w = try randW(ctx, &[_]i32{ hidden, HV * D });
    defer out_w.deinit();
    const inv_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    const nw_buf = try allocator.alloc(f32, @intCast(D));
    defer allocator.free(nw_buf);
    @memset(nw_buf, inv_scale * inv_scale);
    const q_norm_w = try Array.fromData(allocator, f32, nw_buf, &[_]i32{D});
    defer q_norm_w.deinit();
    @memset(nw_buf, inv_scale);
    const k_norm_w = try Array.fromData(allocator, f32, nw_buf, &[_]i32{D});
    defer k_norm_w.deinit();
    var idx_buf: [HV]i32 = undefined;
    for (&idx_buf, 0..) |*v, h| v.* = @intCast(h / 3);
    const qk_idx = try Array.fromData(allocator, i32, &idx_buf, &[_]i32{HV});
    defer qk_idx.deinit();

    // Eval all inputs so they are materialized like resident weights.
    try eval_mod.asyncEval(allocator, &[_]Array{ x, conv_state, ssm_state, qkv_w, conv1d_t, a_w, b_w, z_w, A_log, dt_bias, norm_w, out_w, q_norm_w, k_norm_w, qk_idx });
    try x.eval();

    const cc = try qwen38_gdn.compiledStep(allocator, false);
    defer cc.deinit();

    var cur_conv = try ops.copy(ctx, conv_state);
    defer cur_conv.deinit();
    var cur_ssm = try ops.copy(ctx, ssm_state);
    defer cur_ssm.deinit();

    var timer = std.c.mach_absolute_time();

    // Warm (trace).
    {
        const inputs = [_]Array{ x, cur_conv, cur_ssm, qkv_w, conv1d_t, a_w, b_w, z_w, A_log, dt_bias, norm_w, out_w, q_norm_w, k_norm_w, qk_idx };
        const outs = try cc.apply(&inputs, allocator);
        for (outs) |o| try o.eval();
        outs[0].deinit();
        cur_conv.deinit();
        cur_conv = outs[1];
        cur_ssm.deinit();
        cur_ssm = outs[2];
        allocator.free(outs);
    }
    std.debug.print("warm+trace: {d:.2} ms\n", .{elapsedMs(timer)});
    timer = std.c.mach_absolute_time();

    const N = 50;
    var i: usize = 0;
    while (i < N) : (i += 1) {
        const inputs = [_]Array{ x, cur_conv, cur_ssm, qkv_w, conv1d_t, a_w, b_w, z_w, A_log, dt_bias, norm_w, out_w, q_norm_w, k_norm_w, qk_idx };
        const outs = try cc.apply(&inputs, allocator);
        // mimic serving: async states, sync out
        try eval_mod.asyncEval(allocator, &[_]Array{ outs[1], outs[2] });
        try outs[0].eval();
        outs[0].deinit();
        cur_conv.deinit();
        cur_conv = outs[1];
        cur_ssm.deinit();
        cur_ssm = outs[2];
        allocator.free(outs);
    }
    const per_apply = elapsedMs(timer) / @as(f64, N);
    std.debug.print("compiled apply+eval: {d:.3} ms/step ({d} steps)\n", .{ per_apply, N });

    // Eager comparison.
    var g = qwen38_gdn.Qwen38GDN{
        .ctx = ctx,
        .hidden_size = hidden,
        .num_key_heads = HK,
        .num_value_heads = HV,
        .head_dim = D,
        .conv_kernel = K,
        .in_proj_qkv = qkv_w,
        .conv1d = qkv_w, // unused by forward
        .in_proj_a = a_w,
        .in_proj_b = b_w,
        .in_proj_z = z_w,
        .A_log = A_log,
        .dt_bias = dt_bias,
        .norm_w = norm_w,
        .out_proj = out_w,
        .conv1d_t = conv1d_t,
        .q_norm_w = q_norm_w,
        .k_norm_w = k_norm_w,
        .qk_idx = qk_idx,
    };
    var cache = try qwen38_gdn.Qwen38GdnCache.init(allocator, K, QKV, HV, D);
    defer cache.deinit();
    i = 0;
    timer = std.c.mach_absolute_time();
    while (i < N) : (i += 1) {
        const out = try g.forward(x, &cache);
        try eval_mod.asyncEval(allocator, &[_]Array{ cache.conv_state, cache.ssm_state });
        try out.eval();
        out.deinit();
    }
    std.debug.print("eager forward+eval: {d:.3} ms/step\n", .{elapsedMs(timer) / @as(f64, N)});

    // CPU-only apply cost: no eval until the very end.
    i = 0;
    timer = std.c.mach_absolute_time();
    var pending = std.ArrayList(Array).empty;
    defer {
        for (pending.items) |o| o.deinit();
        pending.deinit(allocator);
    }
    while (i < N) : (i += 1) {
        const inputs = [_]Array{ x, cur_conv, cur_ssm, qkv_w, conv1d_t, a_w, b_w, z_w, A_log, dt_bias, norm_w, out_w, q_norm_w, k_norm_w, qk_idx };
        const outs = try cc.apply(&inputs, allocator);
        try pending.append(allocator, outs[0]);
        outs[1].deinit();
        outs[2].deinit();
        allocator.free(outs);
    }
    std.debug.print("compiled apply only (no eval): {d:.3} ms/step\n", .{elapsedMs(timer) / @as(f64, N)});
    // eager CPU-only comparison: build the same step graph without eval
    i = 0;
    timer = std.c.mach_absolute_time();
    while (i < N) : (i += 1) {
        const out = try g.forward(x, &cache);
        try pending.append(allocator, out);
    }
    std.debug.print("eager forward only (no eval): {d:.3} ms/step\n", .{elapsedMs(timer) / @as(f64, N)});
}
