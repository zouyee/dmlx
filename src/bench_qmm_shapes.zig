// Scratch micro-benchmark: quantized-matmul GPU time + effective GB/s at the
// real Qwen3.8-27B decode shapes. Not committed.
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
    per_token: u32, // how many of these run per decode token
};

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    c.initErrorHandler();
    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    const shapes = [_]Shape{
        .{ .name = "gdn_qkv  [10240,5120]  x48", .out = 10240, .in = 5120, .per_token = 48 },
        .{ .name = "gdn_z    [6144,5120]   x48", .out = 6144, .in = 5120, .per_token = 48 },
        .{ .name = "gdn_a/b  [48,5120]     x96", .out = 48, .in = 5120, .per_token = 96 },
        .{ .name = "gdn_out  [5120,6144]   x48", .out = 5120, .in = 6144, .per_token = 48 },
        .{ .name = "attn_q   [12288,5120]  x16", .out = 12288, .in = 5120, .per_token = 16 },
        .{ .name = "attn_k/v [1024,5120]   x32", .out = 1024, .in = 5120, .per_token = 32 },
        .{ .name = "attn_o   [5120,6144]   x16", .out = 5120, .in = 6144, .per_token = 16 },
        .{ .name = "mlp_gate [17408,5120]  x64", .out = 17408, .in = 5120, .per_token = 64 },
        .{ .name = "mlp_up   [17408,5120]  x64", .out = 17408, .in = 5120, .per_token = 64 },
        .{ .name = "mlp_down [5120,17408]  x64", .out = 5120, .in = 17408, .per_token = 64 },
        .{ .name = "lm_head  [248320,5120] x1", .out = 248320, .in = 5120, .per_token = 1 },
    };

    // Null-barrier calibration: eval a pre-materialized scalar.
    const tiny = try ops.scalarF32(ctx, 1.0);
    defer tiny.deinit();
    try tiny.eval();
    var t = std.c.mach_absolute_time();
    const NB = 200;
    for (0..NB) |_| try tiny.eval();
    const null_barrier_ms = elapsedMs(t) / NB;
    std.debug.print("null barrier (eval of materialized scalar): {d:.4} ms\n", .{null_barrier_ms});

    std.debug.print("\n{s: <28} {s: >9} {s: >9} {s: >9} {s: >10} {s: >10} {s: >10}\n", .{ "shape", "barrierm", "kernelms", "GB/s", "ms/token", "GB/token", "% 71ms" });
    var total_ms_token: f64 = 0;
    for (shapes) |s| {
        const w = try random.normal(ctx, &[_]i32{ s.out, s.in }, .bfloat16, 0.0, 0.02, null);
        defer w.deinit();
        const qw = try quantize_mod.quantize(ctx, w, .{ .bits = 4, .group_size = 64, .mode = .affine });
        defer qw.data.deinit();
        defer qw.scales.deinit();
        defer qw.biases.deinit();
        const x = try random.normal(ctx, &[_]i32{ 1, s.in }, .bfloat16, 0.0, 0.02, null);
        defer x.deinit();
        try x.eval();

        // warm (also JIT)
        for (0..5) |_| {
            const y = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
            defer y.deinit();
            try y.eval();
        }
        const N = 100;
        t = std.c.mach_absolute_time();
        for (0..N) |_| {
            const y = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
            defer y.deinit();
            try y.eval();
        }
        const per_barriered = elapsedMs(t) / N;
        // True kernel time: submit N qmms, then ONE eval of all outputs so all
        // kernels land in a single command buffer (like the real decode graph).
        var pending = std.ArrayList(Array).empty;
        for (0..N) |_| {
            const y = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
            try pending.append(allocator, y);
        }
        t = std.c.mach_absolute_time();
        try eval_mod.asyncEval(allocator, pending.items);
        try pending.items[N - 1].eval();
        const per_kernel = elapsedMs(t) / N;
        for (pending.items) |y| y.deinit();
        pending.deinit(allocator);
        // bytes: packed weights (in*out/2) + scales+biases (out*in/64 groups * 2 * 4B f32) + x + out
        const w_bytes = @as(f64, @floatFromInt(s.out)) * @as(f64, @floatFromInt(s.in)) * 0.5;
        const sb_bytes = @as(f64, @floatFromInt(s.out)) * @as(f64, @floatFromInt(s.in)) / 64.0 * 8.0;
        const io_bytes = @as(f64, @floatFromInt(s.in + s.out)) * 2.0;
        const gb = (w_bytes + sb_bytes + io_bytes) / 1e9;
        const gbps_kernel = gb / (per_kernel / 1e3);
        const ms_token = per_kernel * @as(f64, @floatFromInt(s.per_token));
        total_ms_token += ms_token;
        std.debug.print("{s: <28} {d: >9.4} {d: >9.4} {d: >9.1} {d: >10.3} {d: >10.4} {d: >10.1}\n", .{ s.name, per_barriered, per_kernel, gbps_kernel, ms_token, gb * @as(f64, @floatFromInt(s.per_token)), 100.0 * ms_token / 71.0 });
    }
    std.debug.print("\nqmm total per token: {d:.2} ms ({d:.1}% of 71ms wall)\n", .{ total_ms_token, 100.0 * total_ms_token / 71.0 });

    // lm_head split: vocab 248320 may use a different kernel for very tall out.
    // Also measure a mid token-count batch x to see kernel switching (M=8 like DECODE_BLOCK).
    const w8 = try random.normal(ctx, &[_]i32{ 17408, 5120 }, .bfloat16, 0.0, 0.02, null);
    defer w8.deinit();
    const qw8 = try quantize_mod.quantize(ctx, w8, .{ .bits = 4, .group_size = 64, .mode = .affine });
    defer qw8.data.deinit();
    defer qw8.scales.deinit();
    defer qw8.biases.deinit();
    inline for (.{ 1, 8 }) |m| {
        const xm = try random.normal(ctx, &[_]i32{ m, 5120 }, .bfloat16, 0.0, 0.02, null);
        defer xm.deinit();
        try xm.eval();
        for (0..5) |_| {
            const y = try quantize_mod.quantizedMatmul(ctx, xm, qw8, true);
            defer y.deinit();
            try y.eval();
        }
        t = std.c.mach_absolute_time();
        for (0..100) |_| {
            const y = try quantize_mod.quantizedMatmul(ctx, xm, qw8, true);
            defer y.deinit();
            try y.eval();
        }
        std.debug.print("mlp_gate M={d}: {d:.4} ms\n", .{ m, elapsedMs(t) / 100 - null_barrier_ms });
    }
}
