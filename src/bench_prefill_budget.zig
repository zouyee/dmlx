// Scratch: where does prefill time actually go at L=512 / 2048?
//
// Measured (scripts/compare_engines.py, after the lm_head fix): dmlx prefill
// is 117.4 tok/s vs omlx 134.5 and mlx-vlm 133.1 — last by 12-13%, which is
// the only reason dmlx loses TTFT above ~360 tokens despite having by far the
// lowest fixed overhead. A 2056-token prefill measures ~17.5 s wall.
//
// Before porting a prefill-shape GEMM kernel, establish how much of that wall
// is quantized matmul at all. The verify work taught the lesson twice: the
// bottleneck was NOT where the first plausible story put it (the "80 ms of
// graph build" was a blocking async_eval, and the "21 ms/row intrinsic" was
// stock-MLX-only). So measure first.
//
// Not committed. Run: zig build run-bench-prefill-budget -Doptimize=ReleaseFast
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

const Shape = struct {
    name: []const u8,
    out: i32,
    in: i32,
    per_pass: u32, // calls per forward pass
};

// Real Qwen3.8-27B shapes: 64 layers = 48 GDN + 16 full attention.
// lm_head is deliberately absent: prefill now applies it to one row only
// (see qwen38.forwardTokensCore `last_logits_only`).
const shapes = [_]Shape{
    .{ .name = "gdn_qkv  [10240,5120]", .out = 10240, .in = 5120, .per_pass = 48 },
    .{ .name = "gdn_z    [6144,5120]", .out = 6144, .in = 5120, .per_pass = 48 },
    .{ .name = "gdn_out  [5120,6144]", .out = 5120, .in = 6144, .per_pass = 48 },
    .{ .name = "gdn_a/b  [48,5120]", .out = 48, .in = 5120, .per_pass = 96 },
    .{ .name = "attn_q   [12288,5120]", .out = 12288, .in = 5120, .per_pass = 16 },
    .{ .name = "attn_kv  [1024,5120]", .out = 1024, .in = 5120, .per_pass = 32 },
    .{ .name = "attn_o   [5120,6144]", .out = 5120, .in = 6144, .per_pass = 16 },
    .{ .name = "mlp_gate [17408,5120]", .out = 17408, .in = 5120, .per_pass = 64 },
    .{ .name = "mlp_up   [17408,5120]", .out = 17408, .in = 5120, .per_pass = 64 },
    .{ .name = "mlp_down [5120,17408]", .out = 5120, .in = 17408, .per_pass = 64 },
};

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

    const ms = [_]i32{ 512, 2048 };
    // Measured prefill wall for ONE forward at these prompt lengths
    // (compare_engines.py, dmlx, after the lm_head fix).
    const wall_s = [_]f64{ 4.394, 17.479 };

    std.debug.print("Prefill qmm budget — Qwen3.8-27B, 4b/gs64, bf16 activations\n\n", .{});
    for (ms, 0..) |m, mi| {
        const x = try random.normal(ctx, &[_]i32{ m, 5120 }, .bfloat16, 0.0, 0.02, null);
        defer x.deinit();
        try x.eval();

        std.debug.print("M = {d}  (measured prefill wall {d:.3} s)\n", .{ m, wall_s[mi] });
        std.debug.print("   {s: <24}{s: >10}{s: >12}{s: >10}\n", .{ "shape", "ms/call", "ms/pass", "GB/s" });
        var total: f64 = 0;
        for (shapes) |s| {
            const w = try random.normal(ctx, &[_]i32{ s.out, s.in }, .bfloat16, 0.0, 0.02, null);
            defer w.deinit();
            const qw = try quantize_mod.quantize(ctx, w, .{ .bits = 4, .group_size = 64, .mode = .affine });
            defer qw.data.deinit();
            defer qw.scales.deinit();
            defer qw.biases.deinit();
            const xi = if (s.in == 5120) x else blk: {
                const x2 = try random.normal(ctx, &[_]i32{ m, s.in }, .bfloat16, 0.0, 0.02, null);
                try x2.eval();
                break :blk x2;
            };
            defer if (s.in != 5120) xi.deinit();
            const reps: usize = if (m >= 2048) 3 else 8;
            const t = try timeQmm(allocator, ctx, xi, qw, reps);
            const per_pass = t * @as(f64, @floatFromInt(s.per_pass));
            total += per_pass;
            // Effective bandwidth: weights are re-read once per call.
            const elems = @as(f64, @floatFromInt(s.out)) * @as(f64, @floatFromInt(s.in));
            const gb = (elems * 0.5 + elems / 64.0 * 8.0) / 1e9;
            std.debug.print("   {s: <24}{d: >10.3}{d: >12.1}{d: >10.1}\n", .{ s.name, t, per_pass, gb / (t / 1e3) });
        }
        const wall_ms = wall_s[mi] * 1000.0;
        std.debug.print("   {s: <24}{s: >10}{d: >12.1}\n", .{ "TOTAL qmm", "", total });
        std.debug.print("   => qmm is {d:.1}% of the {d:.0} ms prefill wall; non-qmm {d:.1} ms\n\n", .{
            100.0 * total / wall_ms, wall_ms, wall_ms - total,
        });
        // What a kernel port could buy, if it matched omlx's throughput.
        const target = 134.5; // omlx tok/s
        const need_ms = @as(f64, @floatFromInt(m)) / target * 1000.0;
        std.debug.print("   to reach omlx's {d:.1} tok/s this pass must take {d:.0} ms (now {d:.0});\n", .{ target, need_ms, wall_ms });
        std.debug.print("   that needs {d:.0} ms removed, i.e. {d:.0}% of the qmm term.\n\n", .{
            wall_ms - need_ms, 100.0 * (wall_ms - need_ms) / total,
        });
    }
}
