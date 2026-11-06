//! Faithful microbench of the MTP coarse draft head, linked against the
//! SAME mlx-c as the server (the python venv's mlx gave numbers 2x off the
//! in-context ones — do not trust it for this path).
//!
//! Per draft step the chain is: coarse 2-bit qmm [1,5120]x[248320,5120]
//! -> argpartition top-64 -> gather 64 rows of the 4-bit head -> small qmm
//! -> argmax. Prints per-stage ms so the fusion work knows what to kill.
//!
//! Run: zig build run-bench-coarse-head -Doptimize=ReleaseFast

const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const random = @import("mlx").random;
const quantize_mod = @import("mlx").quantize;
const sort_mod = @import("mlx").sort;
const reduce_mod = @import("mlx").reduce;
const shape_mod = @import("mlx").shape;
const eval_mod = @import("mlx").eval;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

fn elapsedMs(start: u64) f64 {
    const ns = (std.c.mach_absolute_time() - start) * 125 / 3;
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    c.initErrorHandler();
    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    const vocab: i32 = 248320;
    const hidden: i32 = 5120;
    const cand: i32 = 64;
    const reps = 20;

    // Coarse head: 2-bit gs128 (as built by buildCoarseHead).
    const w = try random.normal(ctx, &[_]i32{ vocab, hidden }, .bfloat16, 0.0, 0.02, null);
    defer w.deinit();
    try w.eval();
    const coarse = try quantize_mod.quantize(ctx, w, .{ .group_size = 128, .bits = 2 });
    defer coarse.deinit(allocator);
    const exact = try quantize_mod.quantize(ctx, w, .{ .group_size = 64, .bits = 4 });
    defer exact.deinit(allocator);
    const x = try random.normal(ctx, &[_]i32{ 1, 1, hidden }, .bfloat16, 0.0, 0.02, null);
    defer x.deinit();
    try x.eval();

    var pending = std.ArrayList(Array).empty;
    defer pending.deinit(allocator);
    defer for (pending.items) |y| y.deinit();

    // (a) coarse qmm alone
    for (0..3) |_| {
        const y = try quantize_mod.quantizedMatmul(ctx, x, coarse, true);
        defer y.deinit();
        try y.eval();
    }
    var t = std.c.mach_absolute_time();
    for (0..reps) |_| {
        try pending.append(allocator, try quantize_mod.quantizedMatmul(ctx, x, coarse, true));
    }
    try eval_mod.asyncEval(allocator, pending.items);
    try pending.items[pending.items.len - 1].eval();
    const qmm_ms = elapsedMs(t) / reps;
    pending.clearRetainingCapacity();

    // (b) argpartition top-64 over the coarse logits
    const approx = try quantize_mod.quantizedMatmul(ctx, x, coarse, true);
    defer approx.deinit();
    try approx.eval();
    const approx2 = try ops.reshape(ctx, approx, &[_]i32{ 1, vocab });
    defer approx2.deinit();
    for (0..3) |_| {
        const p = try sort_mod.argpartitionAxis(ctx, approx2, vocab - cand, 1);
        defer p.deinit();
        try p.eval();
    }
    t = std.c.mach_absolute_time();
    for (0..reps) |_| {
        try pending.append(allocator, try sort_mod.argpartitionAxis(ctx, approx2, vocab - cand, 1));
    }
    try eval_mod.asyncEval(allocator, pending.items);
    try pending.items[pending.items.len - 1].eval();
    const part_ms = elapsedMs(t) / reps;
    pending.clearRetainingCapacity();

    // (c) rescore: gather 64 rows of the 4-bit head + small qmm + argmax
    const part = try sort_mod.argpartitionAxis(ctx, approx2, vocab - cand, 1);
    defer part.deinit();
    const ids2 = try ops.slice(ctx, part, &[_]i32{ 0, vocab - cand }, &[_]i32{ 1, vocab }, &[_]i32{ 1, 1 });
    defer ids2.deinit();
    const ids = try ops.reshape(ctx, ids2, &[_]i32{cand});
    defer ids.deinit();
    try ids.eval();
    const rescore = struct {
        fn go(cx: EagerContext, id_arr: Array, qw: quantize_mod.QuantizedWeight, xx: Array) !Array {
            const rows = try shape_mod.takeAxis(cx, qw.data, id_arr, 0);
            defer rows.deinit();
            const rs = try shape_mod.takeAxis(cx, qw.scales, id_arr, 0);
            defer rs.deinit();
            const rb = try shape_mod.takeAxis(cx, qw.biases, id_arr, 0);
            defer rb.deinit();
            const sub = quantize_mod.QuantizedWeight{ .data = rows, .scales = rs, .biases = rb, .config = qw.config, .original_shape = &.{} };
            const sc = try quantize_mod.quantizedMatmul(cx, xx, sub, true);
            defer sc.deinit();
            return reduce_mod.argmaxAxis(cx, sc, 2, false);
        }
    };
    for (0..3) |_| {
        const y = try rescore.go(ctx, ids, exact, x);
        defer y.deinit();
        try y.eval();
    }
    t = std.c.mach_absolute_time();
    for (0..reps) |_| {
        try pending.append(allocator, try rescore.go(ctx, ids, exact, x));
    }
    try eval_mod.asyncEval(allocator, pending.items);
    try pending.items[pending.items.len - 1].eval();
    const rescore_ms = elapsedMs(t) / reps;
    pending.clearRetainingCapacity();

    const bw = @as(f64, @floatFromInt(@as(u64, @intCast(vocab)) * @as(u64, @intCast(hidden)))) * 0.25 / qmm_ms / 1e6;
    std.debug.print("coarse head per draft step (vocab={d}, cand={d}):\n", .{ vocab, cand });
    std.debug.print("  coarse qmm 2-bit: {d:.3} ms  ({d:.0} GB/s)\n", .{ qmm_ms, bw });
    std.debug.print("  argpartition t64: {d:.3} ms\n", .{part_ms});
    std.debug.print("  rescore+argmax  : {d:.3} ms\n", .{rescore_ms});
    std.debug.print("  TOTAL           : {d:.3} ms/step  ({d:.1} ms/round at depth 7)\n", .{ qmm_ms + part_ms + rescore_ms, 7 * (qmm_ms + part_ms + rescore_ms) });
}
