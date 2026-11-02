// Scratch micro-benchmark: does the 4-bit quantized matmul cost scale with
// the row count M? MTP verify runs the whole 27B stack at M = L = N+1 rows
// instead of the M=1 decode shape, and the measured per-round verify cost is
// superlinear in L (77.0 / 84.6 / 102.7 / 126.5 ms at L = 2/3/4/5). If
// `quantizedMatmul` is weight-bandwidth-bound and M-invariant for small M,
// verify should be nearly flat in L and the superlinearity lives elsewhere
// (GDN recurrence / s_all / attention). This bench isolates the qmm term.
//
// Not committed. Run: zig build run-bench-verify-m -Doptimize=ReleaseFast
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
    const ns = (std.c.mach_absolute_time() - start) * 125 / 3; // Apple Silicon timebase
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

const Shape = struct {
    name: []const u8,
    out: i32,
    in: i32,
    per_token: u32, // how many of these run per forward pass
};

// Real Qwen3.8-27B shapes: 64 layers = 48 GDN + 16 full attention.
const shapes = [_]Shape{
    .{ .name = "gdn_qkv  [10240,5120]", .out = 10240, .in = 5120, .per_token = 48 },
    .{ .name = "gdn_z    [6144,5120]", .out = 6144, .in = 5120, .per_token = 48 },
    .{ .name = "gdn_out  [5120,6144]", .out = 5120, .in = 6144, .per_token = 48 },
    .{ .name = "attn_q   [12288,5120]", .out = 12288, .in = 5120, .per_token = 16 },
    .{ .name = "attn_k/v [1024,5120]", .out = 1024, .in = 5120, .per_token = 32 },
    .{ .name = "attn_o   [5120,6144]", .out = 5120, .in = 6144, .per_token = 16 },
    .{ .name = "mlp_gate [17408,5120]", .out = 17408, .in = 5120, .per_token = 64 },
    .{ .name = "mlp_up   [17408,5120]", .out = 17408, .in = 5120, .per_token = 64 },
    .{ .name = "mlp_down [5120,17408]", .out = 5120, .in = 17408, .per_token = 64 },
    .{ .name = "lm_head  [248320,5120]", .out = 248320, .in = 5120, .per_token = 1 },
};

// MLX dispatches `M >= get_qmv_batch_limit(K, N)` to the split-K GEMM and
// everything below it to the batched qmv (mlx/backend/metal/quantized.cpp
// `QuantizedMatmul::eval_gpu`). For K=5120 (> 4096) the limit is 6..12
// depending on arch gen, so verify at L=2..5 is always on the qmv path.
// Sweep across the boundary to see whether the GEMM path is cheaper — if it
// is, padding the verify rows up to the limit is a free win.
const MS = [_]i32{ 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 24, 32 };

/// True kernel time: submit N qmms, then one eval so every kernel lands in a
/// single command buffer (mirrors the real forward graph) — not N barriers.
fn kernelMs(
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    x: Array,
    qw: anytype,
    n: usize,
) !f64 {
    var pending = std.ArrayList(Array).empty;
    defer pending.deinit(allocator);
    defer for (pending.items) |y| y.deinit();
    for (0..n) |_| {
        const y = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
        try pending.append(allocator, y);
    }
    const t = std.c.mach_absolute_time();
    try eval_mod.asyncEval(allocator, pending.items);
    try pending.items[n - 1].eval();
    return elapsedMs(t) / @as(f64, @floatFromInt(n));
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    c.initErrorHandler();
    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    std.debug.print("4-bit quantizedMatmul (gs=64, affine) vs row count M — Qwen3.8-27B shapes\n", .{});
    std.debug.print("per-shape kernel ms (single command buffer, 50 reps)\n\n", .{});
    std.debug.print("{s: <24}", .{"shape"});
    for (MS) |m| std.debug.print("{s: >5}{d: <4}", .{ "M=", m });
    std.debug.print("{s: >10}\n", .{"M4/M1"});

    var total_per_m = [_]f64{0} ** MS.len;

    for (shapes) |s| {
        const w = try random.normal(ctx, &[_]i32{ s.out, s.in }, .bfloat16, 0.0, 0.02, null);
        defer w.deinit();
        const qw = try quantize_mod.quantize(ctx, w, .{ .bits = 4, .group_size = 64, .mode = .affine });
        defer qw.data.deinit();
        defer qw.scales.deinit();
        defer qw.biases.deinit();

        std.debug.print("{s: <24}", .{s.name});
        var per_m = [_]f64{0} ** MS.len;
        for (MS, 0..) |m, mi| {
            const x = try random.normal(ctx, &[_]i32{ m, s.in }, .bfloat16, 0.0, 0.02, null);
            defer x.deinit();
            try x.eval();
            // Warm (also JIT-compiles the kernel for this shape).
            for (0..5) |_| {
                const y = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
                defer y.deinit();
                try y.eval();
            }
            const ms = try kernelMs(allocator, ctx, x, qw, 50);
            per_m[mi] = ms;
            total_per_m[mi] += ms * @as(f64, @floatFromInt(s.per_token));
            std.debug.print("{d: >9.4}", .{ms});
        }
        std.debug.print("{d: >10.2}\n", .{per_m[3] / per_m[0]});
    }

    std.debug.print("\nSummed over one full 64-layer forward pass (all shapes x per_token):\n", .{});
    std.debug.print("{s: <10}{s: >12}{s: >14}{s: >14}\n", .{ "M", "qmm ms", "vs M=1", "marginal ms" });
    for (MS, 0..) |m, mi| {
        const marginal = if (mi == 0) 0.0 else total_per_m[mi] - total_per_m[mi - 1];
        std.debug.print("{d: <10}{d: >12.2}{d: >14.2}{d: >14.2}\n", .{ m, total_per_m[mi], total_per_m[mi] / total_per_m[0], marginal });
    }
    std.debug.print("\nCompare: measured MTP verify ms/round = 77.0 (L=2) 84.6 (L=3) 102.7 (L=4) 126.5 (L=5)\n", .{});
    std.debug.print("and an M=1 decode step (MTP off) = 72.5 ms total wall.\n", .{});
}
