// Scratch micro-benchmark: non-qmm decode kernels — SDPA at decode shape,
// rmsNorm cluster, residual add, embed take. Batched single-eval timing.
// Not committed.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const random = @import("mlx").random;
const fast_mod = @import("mlx").fast;
const eval_mod = @import("mlx").eval;
const shape_mod = @import("mlx").shape;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

fn elapsedMs(start: u64) f64 {
    const ns = (std.c.mach_absolute_time() - start) * 125 / 3;
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn bench(label: []const u8, allocator: std.mem.Allocator, comptime N: usize, outs: []Array) f64 {
    const t = std.c.mach_absolute_time();
    eval_mod.asyncEval(allocator, outs) catch unreachable;
    outs[N - 1].eval() catch unreachable;
    const per = elapsedMs(t) / @as(f64, N);
    std.debug.print("{s: <44} {d: >8.4} ms/op\n", .{ label, per });
    return per;
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    c.initErrorHandler();
    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    const N = 128;
    var pending = std.ArrayList(Array).empty;
    defer {
        for (pending.items) |a| a.deinit();
        pending.deinit(allocator);
    }

    // --- SDPA decode shape: q [1,24,1,256], k/v [1,4,T,256], GQA ---
    for ([_]i32{ 64, 512, 2048 }) |T| {
        pending.clearRetainingCapacity();
        const q = try random.normal(ctx, &[_]i32{ 1, 24, 1, 256 }, .bfloat16, 0.0, 0.02, null);
        const k = try random.normal(ctx, &[_]i32{ 1, 4, T, 256 }, .bfloat16, 0.0, 0.02, null);
        const v = try random.normal(ctx, &[_]i32{ 1, 4, T, 256 }, .bfloat16, 0.0, 0.02, null);
        try q.eval();
        try k.eval();
        try v.eval();
        for (0..N) |_| {
            const o = try fast_mod.scaledDotProductAttention(ctx, q, k, v, 1.0 / 16.0, "causal", null, null);
            try pending.append(allocator, o);
        }
        var buf: [64]u8 = undefined;
        const label = try std.fmt.bufPrint(&buf, "sdpa T={d}", .{T});
        _ = bench(label, allocator, N, pending.items);
        for (pending.items) |a| a.deinit();
        q.deinit();
        k.deinit();
        v.deinit();
    }

    // --- rmsNorm cluster as in the harness: astype f32 -> rmsNorm -> astype bf16 ---
    {
        pending.clearRetainingCapacity();
        const h = try random.normal(ctx, &[_]i32{ 1, 1, 5120 }, .bfloat16, 0.0, 0.02, null);
        defer h.deinit();
        const w = try random.normal(ctx, &[_]i32{5120}, .float32, 0.9, 0.02, null);
        defer w.deinit();
        try h.eval();
        try w.eval();
        for (0..N) |_| {
            const hf = try ops.astype(ctx, h, .float32);
            defer hf.deinit();
            const nf = try fast_mod.rmsNorm(ctx, hf, w, 1e-6);
            defer nf.deinit();
            const out = try ops.astype(ctx, nf, .bfloat16);
            try pending.append(allocator, out);
        }
        _ = bench("rmsnorm cluster (astype+norm+astype)", allocator, N, pending.items);
        for (pending.items) |a| a.deinit();
    }

    // --- residual add [1,1,5120] bf16 ---
    {
        pending.clearRetainingCapacity();
        const a = try random.normal(ctx, &[_]i32{ 1, 1, 5120 }, .bfloat16, 0.0, 0.02, null);
        defer a.deinit();
        try a.eval();
        for (0..N) |_| {
            const o = try ops.add(ctx, a, a);
            try pending.append(allocator, o);
        }
        _ = bench("residual add [1,1,5120] bf16", allocator, N, pending.items);
        for (pending.items) |a2| a2.deinit();
    }

    // --- embed take: [248320,5120] bf16 table, 1 row ---
    {
        pending.clearRetainingCapacity();
        const table = try random.normal(ctx, &[_]i32{ 248320, 5120 }, .bfloat16, 0.0, 0.02, null);
        defer table.deinit();
        try table.eval();
        const id_arr = try Array.fromData(allocator, i32, &[_]i32{42}, &[_]i32{1});
        defer id_arr.deinit();
        for (0..N) |_| {
            const rows = try shape_mod.takeAxis(ctx, table, id_arr, 0);
            try pending.append(allocator, rows);
        }
        _ = bench("embed take 1 row from [248320,5120]", allocator, N, pending.items);
        for (pending.items) |a2| a2.deinit();
    }

    // --- GDN conv sliceUpdate-ish: ssm state write [48,128,128] f32 copy ---
    {
        pending.clearRetainingCapacity();
        const s = try random.normal(ctx, &[_]i32{ 48, 128, 128 }, .float32, 0.0, 0.02, null);
        defer s.deinit();
        try s.eval();
        for (0..N) |_| {
            const o = try ops.copy(ctx, s);
            try pending.append(allocator, o);
        }
        _ = bench("copy ssm_state [48,128,128] f32 (6.3MB r+w)", allocator, N, pending.items);
        for (pending.items) |a2| a2.deinit();
    }

    // --- add+rmsNorm cluster as in forwardTokens: add bf16 -> astype f32 ->
    // rmsNorm -> astype bf16 (two arrays out) vs the fused custom kernel ---
    const qwen38 = @import("models/qwen38.zig");
    {
        pending.clearRetainingCapacity();
        const h = try random.normal(ctx, &[_]i32{ 1, 1, 5120 }, .bfloat16, 0.0, 0.02, null);
        defer h.deinit();
        const b = try random.normal(ctx, &[_]i32{ 1, 1, 5120 }, .bfloat16, 0.0, 0.02, null);
        defer b.deinit();
        const w = try random.normal(ctx, &[_]i32{5120}, .float32, 0.9, 0.02, null);
        defer w.deinit();
        try h.eval();
        try b.eval();
        try w.eval();
        for (0..N) |_| {
            const r = try qwen38.addRmsNormEager(ctx, h, b, w, 1e-6);
            try pending.append(allocator, r.h);
            try pending.append(allocator, r.normed);
        }
        _ = bench("add+rmsnorm eager chain (4 ops, 2 out)", allocator, 2 * N, pending.items);
        for (pending.items) |a2| a2.deinit();
    }
    {
        pending.clearRetainingCapacity();
        const h = try random.normal(ctx, &[_]i32{ 1, 1, 5120 }, .bfloat16, 0.0, 0.02, null);
        defer h.deinit();
        const b = try random.normal(ctx, &[_]i32{ 1, 1, 5120 }, .bfloat16, 0.0, 0.02, null);
        defer b.deinit();
        const w = try random.normal(ctx, &[_]i32{5120}, .float32, 0.9, 0.02, null);
        defer w.deinit();
        try h.eval();
        try b.eval();
        try w.eval();
        for (0..N) |_| {
            const r = (try qwen38.addRmsNormFused(ctx, h, b, w, 1e-6)) orelse {
                std.debug.print("fused addrms unavailable\n", .{});
                return;
            };
            try pending.append(allocator, r.h);
            try pending.append(allocator, r.normed);
        }
        _ = bench("add+rmsnorm fused kernel (1 op, 2 out)", allocator, 2 * N, pending.items);
        for (pending.items) |a2| a2.deinit();
    }
}
