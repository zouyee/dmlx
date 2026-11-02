// Scratch micro-benchmark: compiled GDN decode step with REAL 4-bit quantized
// weights, batched into a single command buffer to get true GPU time/step.
// Not committed.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const random = @import("mlx").random;
const eval_mod = @import("mlx").eval;
const quantize_mod = @import("mlx").quantize;
const qwen38_gdn = @import("models/qwen38_gdn.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

fn elapsedMs(start: u64) f64 {
    const ns = (std.c.mach_absolute_time() - start) * 125 / 3;
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

const QW = struct { w: Array, s: Array, b: Array };

fn qweight(ctx: EagerContext, out: i32, in: i32) !QW {
    const w = try random.normal(ctx, &[_]i32{ out, in }, .bfloat16, 0.0, 0.02, null);
    defer w.deinit();
    const qw = try quantize_mod.quantize(ctx, w, .{ .bits = 4, .group_size = 64, .mode = .affine });
    return .{ .w = qw.data, .s = qw.scales, .b = qw.biases };
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    c.initErrorHandler();
    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    const hidden: i32 = 5120;
    const HK: i32 = 16;
    const HV: i32 = 48;
    const D: i32 = 128;
    const K: i32 = 4;
    const QKV: i32 = (2 * HK + HV) * D; // 10240

    const x = try random.normal(ctx, &[_]i32{ 1, hidden }, .bfloat16, 0.0, 0.02, null);
    defer x.deinit();
    const conv_state = try random.normal(ctx, &[_]i32{ K - 1, QKV }, .float32, 0.0, 0.02, null);
    defer conv_state.deinit();
    const ssm_state = try random.normal(ctx, &[_]i32{ HV, D, D }, .float32, 0.0, 0.02, null);
    defer ssm_state.deinit();
    const qkv = try qweight(ctx, QKV, hidden);
    defer qkv.w.deinit();
    defer qkv.s.deinit();
    defer qkv.b.deinit();
    const conv1d_t = try random.normal(ctx, &[_]i32{ K, QKV }, .bfloat16, 0.0, 0.02, null);
    defer conv1d_t.deinit();
    const a = try qweight(ctx, HV, hidden);
    defer a.w.deinit();
    defer a.s.deinit();
    defer a.b.deinit();
    const b = try qweight(ctx, HV, hidden);
    defer b.w.deinit();
    defer b.s.deinit();
    defer b.b.deinit();
    const z = try qweight(ctx, HV * D, hidden);
    defer z.w.deinit();
    defer z.s.deinit();
    defer z.b.deinit();
    const A_log = try random.normal(ctx, &[_]i32{HV}, .float32, 0.0, 0.02, null);
    defer A_log.deinit();
    const dt_bias = try random.normal(ctx, &[_]i32{HV}, .float32, 0.0, 0.02, null);
    defer dt_bias.deinit();
    const norm_w = try random.normal(ctx, &[_]i32{D}, .float32, 0.0, 0.02, null);
    defer norm_w.deinit();
    const o = try qweight(ctx, hidden, HV * D);
    defer o.w.deinit();
    defer o.s.deinit();
    defer o.b.deinit();
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

    const inputs = [_]Array{ x, conv_state, ssm_state, qkv.w, qkv.s, qkv.b, conv1d_t, a.w, a.s, a.b, b.w, b.s, b.b, z.w, z.s, z.b, A_log, dt_bias, norm_w, o.w, o.s, o.b, q_norm_w, k_norm_w, qk_idx };
    try eval_mod.asyncEval(allocator, &inputs);
    try x.eval();

    const cc = try qwen38_gdn.compiledStep(allocator, true);
    defer cc.deinit();

    // Warm (trace + JIT).
    {
        const outs = try cc.apply(&inputs, allocator);
        for (outs) |oo| try oo.eval();
        for (outs) |oo| oo.deinit();
        allocator.free(outs);
    }

    // Batched: N independent steps, single eval -> true GPU time/step.
    const N = 64;
    var pending = std.ArrayList(Array).empty;
    var timer = std.c.mach_absolute_time();
    for (0..N) |_| {
        const outs = try cc.apply(&inputs, allocator);
        for (outs) |oo| try pending.append(allocator, oo);
        allocator.free(outs);
    }
    const submit_ms = elapsedMs(timer);
    timer = std.c.mach_absolute_time();
    try eval_mod.asyncEval(allocator, pending.items);
    try pending.items[pending.items.len - 1].eval();
    const gpu_ms = elapsedMs(timer);
    for (pending.items) |oo| oo.deinit();
    pending.deinit(allocator);

    std.debug.print("quantized compiled GDN step: submit {d:.4} ms/step | GPU {d:.4} ms/step (N={d})\n", .{ submit_ms / N, gpu_ms / N, N });
    std.debug.print("  -> x48 layers: {d:.2} ms/token GPU (of which qmm bench says ~10.5 ms)\n", .{gpu_ms / N * 48});
}
