// Scratch: how much of prefill is spent on lm_head rows nobody reads?
//
// `qwen38.forwardTokensCore` applies the final norm and lm_head to the FULL
// hidden state [1, L, hidden], producing [1, L, 248320] logits, but only the
// last row is used for sampling. (HANDOFF already noted that those logits
// dominate peak memory at L=4096 — this measures the TIME.) The MTP hidden
// capture is already sliced to the last position; the logits path is not.
//
// Measured prefill throughput (scripts/compare_engines.py): dmlx 109.6 tok/s
// vs omlx 140.8, mlx-vlm 133.0, mlx-lm 130.0 — dmlx is last on this axis, and
// TTFT at a 2k prompt is 18.9 s vs omlx's 15.5 s. If the wasted lm_head rows
// are a meaningful share, slicing before the head is a pure win.
//
// Not committed. Run: zig build run-bench-prefill-head -Doptimize=ReleaseFast
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const random = @import("mlx").random;
const quantize_mod = @import("mlx").quantize;
const eval_mod = @import("mlx").eval;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

fn elapsedMs(start: u64) f64 {
    const ns = (std.c.mach_absolute_time() - start) * 125 / 3;
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn timeQmm(allocator: std.mem.Allocator, ctx: EagerContext, x: Array, qw: anytype, reps: usize) !f64 {
    for (0..2) |_| {
        const y = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
        defer y.deinit();
        try y.eval();
    }
    var pending = std.ArrayList(Array).empty;
    defer pending.deinit(allocator);
    defer for (pending.items) |y| y.deinit();
    for (0..reps) |_| {
        const y = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
        try pending.append(allocator, y);
    }
    const t = std.c.mach_absolute_time();
    try eval_mod.asyncEval(allocator, pending.items);
    try pending.items[reps - 1].eval();
    return elapsedMs(t) / @as(f64, @floatFromInt(reps));
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    c.initErrorHandler();
    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    const K: i32 = 5120;
    const vocab: i32 = 248320;
    const ms = [_]i32{ 1, 512, 2048 };

    const w = try random.normal(ctx, &[_]i32{ vocab, K }, .bfloat16, 0.0, 0.02, null);
    defer w.deinit();
    const qw = try quantize_mod.quantize(ctx, w, .{ .bits = 4, .group_size = 64, .mode = .affine });
    defer qw.data.deinit();
    defer qw.scales.deinit();
    defer qw.biases.deinit();

    std.debug.print("lm_head [{d},{d}] 4b/gs64 vs row count (prefill shapes)\n\n", .{ vocab, K });
    std.debug.print("{s: >6}{s: >12}{s: >14}{s: >16}\n", .{ "M", "ms", "ms/row", "vs M=1" });
    var base: f64 = 0;
    for (ms) |m| {
        const x = try random.normal(ctx, &[_]i32{ m, K }, .bfloat16, 0.0, 0.02, null);
        defer x.deinit();
        try x.eval();
        const reps: usize = if (m >= 2048) 3 else if (m >= 512) 10 else 50;
        const t = try timeQmm(allocator, ctx, x, qw, reps);
        if (m == 1) base = t;
        std.debug.print("{d: >6}{d: >12.2}{d: >14.4}{d: >16.1}\n", .{
            m, t, t / @as(f64, @floatFromInt(m)), t / base,
        });
    }

    std.debug.print("\nWaste if only the last row is needed (M - 1 rows discarded):\n", .{});
    std.debug.print("{s: >6}{s: >14}{s: >14}{s: >16}\n", .{ "M", "full ms", "needed ms", "wasted ms" });
    for (ms) |m| {
        if (m == 1) continue;
        const x = try random.normal(ctx, &[_]i32{ m, K }, .bfloat16, 0.0, 0.02, null);
        defer x.deinit();
        try x.eval();
        const x1 = try random.normal(ctx, &[_]i32{ 1, K }, .bfloat16, 0.0, 0.02, null);
        defer x1.deinit();
        try x1.eval();
        const reps: usize = if (m >= 2048) 3 else 10;
        const full = try timeQmm(allocator, ctx, x, qw, reps);
        const one = try timeQmm(allocator, ctx, x1, qw, 50);
        std.debug.print("{d: >6}{d: >14.2}{d: >14.2}{d: >16.2}\n", .{ m, full, one, full - one });
    }
    std.debug.print("\nMeasured TTFT@2048: dmlx 18.885 s, omlx 15.456 s, mlx-vlm 15.734 s.\n", .{});
}
