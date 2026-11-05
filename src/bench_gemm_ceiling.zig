// Phase 0 / A0: what is the hardware GEMM ceiling at prefill shapes?
//
// dmlx's prefill 4-bit qmm runs at 6.59 TFLOP/s (mlp_gate, M=2048). To match
// omlx we need ~15% more. Before writing a kernel, find out whether that 15%
// exists at all: measure a DENSE bf16 matmul at the same shape, which has no
// dequantization work and therefore bounds what any 4-bit kernel can reach.
//
//   dense ~= qmm        -> we are already at the hardware limit; workstream A
//                          is dead and no kernel can deliver the 15%.
//   dense >> qmm        -> the delta is dequant overhead, i.e. real headroom.
//
// Gate for continuing workstream A: dense / qmm >= 1.25 (15% target plus
// implementation slack).
//
// Also reports 8-bit qmm as a second reference point: it moves 2x the weight
// bytes of 4-bit for the same FLOPs, so comparing 4-bit/8-bit/dense separates
// "bandwidth-bound" from "dequant-ALU-bound".
//
// Not committed. Run: zig build run-bench-gemm-ceiling -Doptimize=ReleaseFast
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

/// Time `reps` ops submitted into ONE command buffer (matches how the real
/// forward queues work, and avoids per-call barrier inflation).
fn timeBatched(allocator: std.mem.Allocator, ctx: EagerContext, comptime F: type, f: F, reps: usize) !f64 {
    for (0..2) |_| {
        const y = try f.run(ctx);
        defer y.deinit();
        try y.eval();
    }
    var pending = std.ArrayList(Array).empty;
    defer pending.deinit(allocator);
    defer for (pending.items) |y| y.deinit();
    for (0..reps) |_| {
        const y = try f.run(ctx);
        try pending.append(allocator, y);
    }
    const t = std.c.mach_absolute_time();
    try eval_mod.asyncEval(allocator, pending.items);
    try pending.items[reps - 1].eval();
    return elapsedMs(t) / @as(f64, @floatFromInt(reps));
}

const Shape = struct { name: []const u8, out: i32, in: i32 };

const shapes = [_]Shape{
    .{ .name = "mlp_gate [17408,5120]", .out = 17408, .in = 5120 },
    .{ .name = "mlp_down [5120,17408]", .out = 5120, .in = 17408 },
    .{ .name = "gdn_qkv  [10240,5120]", .out = 10240, .in = 5120 },
};

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    c.initErrorHandler();
    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    const ms = [_]i32{ 512, 2048 };

    std.debug.print("A0: hardware GEMM ceiling at prefill shapes (bf16 activations)\n", .{});
    std.debug.print("TFLOP/s = 2*M*N*K / time\n\n", .{});
    std.debug.print("{s: <24}{s: >6}{s: >11}{s: >11}{s: >11}{s: >11}{s: >11}\n", .{
        "shape", "M", "q4 ms", "q8 ms", "dense ms", "q4 TF/s", "dense TF/s",
    });

    var gate_ratio: f64 = 0;
    for (shapes) |s| {
        const w = try random.normal(ctx, &[_]i32{ s.out, s.in }, .bfloat16, 0.0, 0.02, null);
        defer w.deinit();
        const q4 = try quantize_mod.quantize(ctx, w, .{ .bits = 4, .group_size = 64, .mode = .affine });
        defer q4.data.deinit();
        defer q4.scales.deinit();
        defer q4.biases.deinit();
        const q8 = try quantize_mod.quantize(ctx, w, .{ .bits = 8, .group_size = 64, .mode = .affine });
        defer q8.data.deinit();
        defer q8.scales.deinit();
        defer q8.biases.deinit();
        // Dense needs w^T for x @ w^T; transpose once, outside the timing.
        const wt = try ops.transpose(ctx, w);
        defer wt.deinit();
        try wt.eval();

        for (ms) |m| {
            const x = try random.normal(ctx, &[_]i32{ m, s.in }, .bfloat16, 0.0, 0.02, null);
            defer x.deinit();
            try x.eval();
            const reps: usize = if (m >= 2048) 3 else 8;

            const Q4 = struct {
                x: Array,
                qw: quantize_mod.QuantizedWeight,
                fn run(self: @This(), cx: EagerContext) !Array {
                    return quantize_mod.quantizedMatmul(cx, self.x, self.qw, true);
                }
            };
            const Q8 = Q4;
            const Dense = struct {
                x: Array,
                wt: Array,
                fn run(self: @This(), cx: EagerContext) !Array {
                    return ops.matmul(cx, self.x, self.wt);
                }
            };

            const t4 = try timeBatched(allocator, ctx, Q4, .{ .x = x, .qw = q4 }, reps);
            const t8 = try timeBatched(allocator, ctx, Q8, .{ .x = x, .qw = q8 }, reps);
            const td = try timeBatched(allocator, ctx, Dense, .{ .x = x, .wt = wt }, reps);

            const flop = 2.0 * @as(f64, @floatFromInt(m)) *
                @as(f64, @floatFromInt(s.out)) * @as(f64, @floatFromInt(s.in));
            const tf4 = flop / (t4 / 1e3) / 1e12;
            const tfd = flop / (td / 1e3) / 1e12;
            std.debug.print("{s: <24}{d: >6}{d: >11.2}{d: >11.2}{d: >11.2}{d: >11.2}{d: >11.2}\n", .{
                s.name, m, t4, t8, td, tf4, tfd,
            });
            if (std.mem.startsWith(u8, s.name, "mlp_gate") and m == 2048) gate_ratio = t4 / td;
        }
    }

    std.debug.print("\nGate for workstream A: dense/q4 speed ratio >= 1.25\n", .{});
    std.debug.print("  mlp_gate @ M=2048: q4 is {d:.2}x the dense time", .{gate_ratio});
    if (gate_ratio >= 1.25) {
        std.debug.print("  -> PASS, dequant overhead is real headroom\n", .{});
    } else {
        std.debug.print("  -> FAIL, 4-bit qmm is already near the dense ceiling;\n" ++
            "     no prefill GEMM kernel can deliver 15%. Kill workstream A.\n", .{});
    }

    // Streaming variant: the loop above reuses ONE weight tensor, so the GPU
    // may keep it partially cached across reps. Real prefill streams 13.5 GB
    // of weights exactly once. Cycle through N distinct weight copies so every
    // rep reads cold weights; the gap to the reused rate is the floor error.
    std.debug.print("\nStreaming q4 (cold weights each rep), M=2048:\n", .{});
    const n_streams = 8;
    for (shapes) |s| {
        const m: i32 = 2048;
        const x = try random.normal(ctx, &[_]i32{ m, s.in }, .bfloat16, 0.0, 0.02, null);
        defer x.deinit();
        try x.eval();
        var qws: [n_streams]quantize_mod.QuantizedWeight = undefined;
        for (0..n_streams) |i| {
            const w = try random.normal(ctx, &[_]i32{ s.out, s.in }, .bfloat16, 0.0, 0.02, null);
            defer w.deinit();
            qws[i] = try quantize_mod.quantize(ctx, w, .{ .bits = 4, .group_size = 64, .mode = .affine });
        }
        defer for (0..n_streams) |i| {
            qws[i].data.deinit();
            qws[i].scales.deinit();
            qws[i].biases.deinit();
        };
        // warm each copy once (compile + first-touch), then time one cold pass each
        for (0..n_streams) |i| {
            const y = try quantize_mod.quantizedMatmul(ctx, x, qws[i], true);
            defer y.deinit();
            try y.eval();
        }
        var pending = std.ArrayList(Array).empty;
        defer pending.deinit(allocator);
        defer for (pending.items) |y| y.deinit();
        for (0..n_streams) |i| {
            const y = try quantize_mod.quantizedMatmul(ctx, x, qws[i], true);
            try pending.append(allocator, y);
        }
        const t = std.c.mach_absolute_time();
        try eval_mod.asyncEval(allocator, pending.items);
        try pending.items[n_streams - 1].eval();
        const per_ms = elapsedMs(t) / n_streams;
        const flop = 2.0 * 2048.0 * @as(f64, @floatFromInt(s.out)) * @as(f64, @floatFromInt(s.in));
        std.debug.print("  {s: <24} {d:.2} ms  {d:.2} TF/s\n", .{ s.name, per_ms, flop / (per_ms / 1e3) / 1e12 });
    }
}
