// Scratch: is MLX's split-K the wrong qmm path for PREFILL shapes?
//
// `QuantizedMatmul::eval_gpu` (mlx/backend/metal/quantized.cpp) routes every
// `M >= get_qmv_batch_limit` case to `qmm_splitk` as long as the weights are
// transposed and the batch is 1:
//
//     if (M >= vector_limit) {
//         int B = out.size() / M / N;
//         if (transpose_ && B == 1) { qmm_splitk(...); return; }
//         qmm(...); return;
//     }
//
// Split-K exists to manufacture parallelism when M is small by splitting the
// K reduction. At prefill shapes (M = 512..4096, N = 17408) the M x N tile
// grid already saturates the GPU, so the extra partial-sum reduction is pure
// overhead — and the plain `qmm` path is only reachable when B > 1.
//
// That makes for a free experiment: the same arithmetic reshaped to
// [B, M/B, K] takes the `qmm` path instead. If it is faster, dmlx gets the
// speedup by reshaping its prefill activations — no kernel to write.
//
// Measured context: qmm is 85.6% of dmlx's prefill wall; matching omlx's
// 134.5 tok/s needs ~15% of the qmm term removed.
//
// Not committed. Run: zig build run-bench-prefill-splitk -Doptimize=ReleaseFast
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

const Shape = struct { name: []const u8, out: i32, in: i32, per_pass: u32 };

const shapes = [_]Shape{
    .{ .name = "mlp_gate [17408,5120]", .out = 17408, .in = 5120, .per_pass = 64 },
    .{ .name = "mlp_down [5120,17408]", .out = 5120, .in = 17408, .per_pass = 64 },
    .{ .name = "gdn_qkv  [10240,5120]", .out = 10240, .in = 5120, .per_pass = 48 },
    .{ .name = "attn_q   [12288,5120]", .out = 12288, .in = 5120, .per_pass = 16 },
};

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    c.initErrorHandler();
    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    const total_m: i32 = 2048;
    // B = 1 is the current path (split-K). B > 1 reaches plain `qmm`.
    const batches = [_]i32{ 1, 2, 4, 8 };

    std.debug.print("Prefill qmm: split-K (B=1) vs plain qmm (B>1), same arithmetic\n", .{});
    std.debug.print("total rows M={d}, reshaped to [B, M/B, K]\n\n", .{total_m});
    std.debug.print("{s: <24}", .{"shape"});
    for (batches) |b| {
        var buf: [16]u8 = undefined;
        std.debug.print("{s: >11}", .{std.fmt.bufPrint(&buf, "B={d}", .{b}) catch "?"});
    }
    std.debug.print("{s: >10}\n", .{"best x"});

    var pass_b1: f64 = 0;
    var pass_best: f64 = 0;
    for (shapes) |s| {
        const w = try random.normal(ctx, &[_]i32{ s.out, s.in }, .bfloat16, 0.0, 0.02, null);
        defer w.deinit();
        const qw = try quantize_mod.quantize(ctx, w, .{ .bits = 4, .group_size = 64, .mode = .affine });
        defer qw.data.deinit();
        defer qw.scales.deinit();
        defer qw.biases.deinit();

        std.debug.print("{s: <24}", .{s.name});
        var t1: f64 = 0;
        var best: f64 = std.math.floatMax(f64);
        for (batches) |b| {
            const rows = @divExact(total_m, b);
            const x = if (b == 1)
                try random.normal(ctx, &[_]i32{ rows, s.in }, .bfloat16, 0.0, 0.02, null)
            else
                try random.normal(ctx, &[_]i32{ b, rows, s.in }, .bfloat16, 0.0, 0.02, null);
            defer x.deinit();
            try x.eval();
            const t = try timeQmm(allocator, ctx, x, qw, 3);
            if (b == 1) t1 = t;
            best = @min(best, t);
            std.debug.print("{d: >11.2}", .{t});
        }
        std.debug.print("{d: >10.2}\n", .{t1 / best});
        pass_b1 += t1 * @as(f64, @floatFromInt(s.per_pass));
        pass_best += best * @as(f64, @floatFromInt(s.per_pass));
    }

    std.debug.print("\nOver one pass, these four shapes only (they are ~82% of prefill qmm):\n", .{});
    std.debug.print("  split-K (current): {d:.0} ms\n", .{pass_b1});
    std.debug.print("  best batching    : {d:.0} ms  ({d:.2}x)\n", .{ pass_best, pass_b1 / pass_best });
    const saved = pass_b1 - pass_best;
    std.debug.print("  saved            : {d:.0} ms of a 17479 ms prefill wall -> {d:.1}%\n", .{ saved, 100.0 * saved / 17479.0 });
    const new_wall = 17479.0 - saved;
    std.debug.print("  prefill would go  117.4 -> {d:.1} tok/s (omlx 134.5, mlx-vlm 133.1)\n", .{2056.0 / (new_wall / 1000.0)});
}
