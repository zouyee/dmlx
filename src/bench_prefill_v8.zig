// Scratch: correctness + speed of the v8-tile prefill qmm vs stock
// quantizedMatmul at the real M=2048 projection shapes.
// Run: zig build run-bench-prefill-v8 -Doptimize=ReleaseFast
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const random = @import("mlx").random;
const quantize_mod = @import("mlx").quantize;
const qmm_v8 = @import("models/qwen38_qmm_v8.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

fn elapsedMs(start: u64) f64 {
    const ns = (std.c.mach_absolute_time() - start) * 125 / 3;
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

const Shape = struct { name: []const u8, out: i32, in: i32, per_pass: u32 };
const shapes = [_]Shape{
    .{ .name = "gdn_qkv ", .out = 10240, .in = 5120, .per_pass = 48 },
    .{ .name = "gdn_z   ", .out = 6144, .in = 5120, .per_pass = 48 },
    .{ .name = "gdn_out ", .out = 5120, .in = 6144, .per_pass = 48 },
    .{ .name = "attn_q  ", .out = 12288, .in = 5120, .per_pass = 16 },
    .{ .name = "attn_o  ", .out = 5120, .in = 6144, .per_pass = 16 },
    .{ .name = "mlp_gate", .out = 17408, .in = 5120, .per_pass = 64 },
    .{ .name = "mlp_down", .out = 5120, .in = 17408, .per_pass = 64 },
};

const M: i32 = 2048;
const REPS = 5;

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    c.initErrorHandler();
    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    var stock_total: f64 = 0;
    var v8_total: f64 = 0;
    for (shapes) |s| {
        const x = try random.normal(ctx, &[_]i32{ M, s.in }, .bfloat16, 0.0, 0.02, null);
        defer x.deinit();
        const w = try random.normal(ctx, &[_]i32{ s.out, s.in }, .bfloat16, 0.0, 0.02, null);
        defer w.deinit();
        const qw = try quantize_mod.quantize(ctx, w, .{ .bits = 4, .group_size = 64, .mode = .affine });
        defer qw.data.deinit();
        defer qw.scales.deinit();
        defer qw.biases.deinit();
        try x.eval();
        try qw.data.eval();
        try qw.scales.eval();
        try qw.biases.eval();

        // correctness
        const ref = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
        defer ref.deinit();
        const deq = try quantize_mod.dequantize(ctx, qw);
        defer deq.deinit();
        const deq_t = try ops.transpose(ctx, deq);
        defer deq_t.deinit();
        const ref2 = try ops.matmul(ctx, x, deq_t);
        defer ref2.deinit();
        const got = (try qmm_v8.apply(ctx, x, qw.data, qw.scales, qw.biases)) orelse {
            std.debug.print("{s}: v8 kernel unavailable\n", .{s.name});
            continue;
        };
        defer got.deinit();
        const diff = try ops.subtract(ctx, ref, got);
        defer diff.deinit();
        const adiff0 = try ops.abs(ctx, diff);
        defer adiff0.deinit();
        const adiff = try ops.astype(ctx, adiff0, .float32);
        defer adiff.deinit();
        const mx = try ops.max(ctx, adiff);
        defer mx.deinit();
        try mx.eval();
        const max_diff = try mx.item(f32);
        // locate the max |v8-stock| diff: max over each 512-row band
        if (s.out == 10240) {
            var band: i32 = 0;
            while (band < M) : (band += 512) {
                const bd = try ops.slice(ctx, adiff, &[_]i32{ band, 0 }, &[_]i32{ band + 512, 10240 }, &[_]i32{ 1, 1 });
                defer bd.deinit();
                const bdm = try ops.max(ctx, bd);
                defer bdm.deinit();
                try bdm.eval();
                std.debug.print("  band {d:5}-{d:5}: {e}\n", .{ band, band + 512, try bdm.item(f32) });
            }
        }
        const diff2 = try ops.subtract(ctx, ref2, got);
        defer diff2.deinit();
        const adiff2a = try ops.abs(ctx, diff2);
        defer adiff2a.deinit();
        const adiff2 = try ops.astype(ctx, adiff2a, .float32);
        defer adiff2.deinit();
        const mx2 = try ops.max(ctx, adiff2);
        defer mx2.deinit();
        try mx2.eval();
        const max_diff2 = try mx2.item(f32);
        // stock vs manual reference
        const diff3 = try ops.subtract(ctx, ref2, ref);
        defer diff3.deinit();
        const adiff3a = try ops.abs(ctx, diff3);
        defer adiff3a.deinit();
        const adiff3 = try ops.astype(ctx, adiff3a, .float32);
        defer adiff3.deinit();
        const mx3 = try ops.max(ctx, adiff3);
        defer mx3.deinit();
        try mx3.eval();
        const max_diff3 = try mx3.item(f32);
        if (s.out == 10240) {
            inline for (.{ .{ "ref", ref }, .{ "got", got }, .{ "ref2", ref2 } }) |pair| {
                const sl = try ops.slice(ctx, pair[1], &[_]i32{ 0, 0 }, &[_]i32{ 1, 4 }, &[_]i32{ 1, 1 });
                defer sl.deinit();
                const slf = try ops.astype(ctx, sl, .float32);
                defer slf.deinit();
                try slf.eval();
                const vals = try slf.dataSlice(f32);
                std.debug.print("  {s}[0,0:4]: {d:.6} {d:.6} {d:.6} {d:.6}\n", .{ pair[0], vals[0], vals[1], vals[2], vals[3] });
            }
        }

        // speed
        var t = std.c.mach_absolute_time();
        for (0..REPS) |_| {
            const y = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
            defer y.deinit();
            try y.eval();
        }
        const t_stock = elapsedMs(t) / REPS;
        t = std.c.mach_absolute_time();
        for (0..REPS) |_| {
            const y = (try qmm_v8.apply(ctx, x, qw.data, qw.scales, qw.biases)).?;
            defer y.deinit();
            try y.eval();
        }
        const t_v8 = elapsedMs(t) / REPS;
        stock_total += t_stock * @as(f64, @floatFromInt(s.per_pass));
        v8_total += t_v8 * @as(f64, @floatFromInt(s.per_pass));
        std.debug.print("{s} [{d:6},{d:6}]  stock {d:8.3} ms  v8 {d:8.3} ms  ratio {d:5.3}x  |v8-stock| {e}  |v8-deq| {e}  |stock-deq| {e}\n", .{ s.name, s.out, s.in, t_stock, t_v8, t_stock / t_v8, max_diff, max_diff2, max_diff3 });
    }
    std.debug.print("TOTAL per pass: stock {d:.1} ms  v8 {d:.1} ms  (saved {d:.1} ms)\n", .{ stock_total, v8_total, stock_total - v8_total });
}
