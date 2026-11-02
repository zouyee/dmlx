// Scratch: pipelined GDN prefill recurrence vs the serial fused kernel —
// numerics (tolerance, not bit-exact: the 128-wide dots use a different
// pairwise tree) and speed at L = 13 / 1901.
// Run: zig build run-bench-gdn-pipe -Doptimize=ReleaseFast
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const random = @import("mlx").random;
const gdn = @import("models/qwen38_gdn.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

fn elapsedMs(start: u64) f64 {
    const ns = (std.c.mach_absolute_time() - start) * 125 / 3;
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

const HV: i32 = 48;
const HK: i32 = 16;
const D: i32 = 128;

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    c.initErrorHandler();
    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    for ([_]i32{ 1, 11, 12, 13, 24, 1901 }) |L| {
        // unbroadcast [L,HK,D] masters; broadcast [L,HV,D] via qk_idx
        const qh = try random.normal(ctx, &[_]i32{ L, HK, D }, .float32, 0.0, 0.09, null);
        defer qh.deinit();
        const kh = try random.normal(ctx, &[_]i32{ L, HK, D }, .float32, 0.0, 0.09, null);
        defer kh.deinit();
        var idx_buf: [HV]i32 = undefined;
        for (&idx_buf, 0..) |*iv, h| iv.* = @intCast(h / 3);
        const qk_idx = try Array.fromData(allocator, i32, &idx_buf, &[_]i32{HV});
        defer qk_idx.deinit();
        const q = try @import("mlx").shape.takeAxis(ctx, qh, qk_idx, 1);
        defer q.deinit();
        const k = try @import("mlx").shape.takeAxis(ctx, kh, qk_idx, 1);
        defer k.deinit();
        const v = try random.normal(ctx, &[_]i32{ L, HV, D }, .float32, 0.0, 0.5, null);
        defer v.deinit();
        // g in (0,1): exp of negative
        const gr = try random.normal(ctx, &[_]i32{ L, HV }, .float32, 2.2, 0.3, null); // sigmoid ~ 0.9
        defer gr.deinit();
        const g = try ops.sigmoid(ctx, gr);
        defer g.deinit();
        const beta = try ops.sigmoid(ctx, gr);
        defer beta.deinit();
        const s0 = try Array.zeros(allocator, &[_]i32{ HV, D, D }, .float32);
        defer s0.deinit();

        const serial = (try gdn.testDeltaRuleSerial(ctx, s0, k, v, beta, g, q, HV, D)).?;
        defer serial.s_new.deinit();
        defer serial.y.deinit();
        const qb = try ops.astype(ctx, qh, .bfloat16);
        defer qb.deinit();
        const kb = try ops.astype(ctx, kh, .bfloat16);
        defer kb.deinit();
        const vb = try ops.astype(ctx, v, .bfloat16);
        defer vb.deinit();
        const pipe_bf = (try gdn.testDeltaRulePipelined(ctx, s0, kb, vb, beta, g, qb, HK, HV, D)).?;
        defer pipe_bf.s_new.deinit();
        defer pipe_bf.y.deinit();
        const dyb = try ops.subtract(ctx, serial.y, try ops.astype(ctx, pipe_bf.y, .float32));
        defer dyb.deinit();
        const ayb = try ops.abs(ctx, dyb);
        defer ayb.deinit();
        const myb = try ops.max(ctx, ayb);
        defer myb.deinit();
        try myb.eval();
        std.debug.print("L={d:5} bf16-in max|dy|={e:.3}\n", .{ L, try myb.item(f32) });
        const pipe = (try gdn.testDeltaRulePipelined(ctx, s0, kh, v, beta, g, qh, HK, HV, D)) orelse {
            std.debug.print("L={d}: pipelined unavailable\n", .{L});
            continue;
        };
        defer pipe.s_new.deinit();
        defer pipe.y.deinit();

        const dy = try ops.subtract(ctx, serial.y, pipe.y);
        defer dy.deinit();
        const ay = try ops.abs(ctx, dy);
        defer ay.deinit();
        const my = try ops.max(ctx, ay);
        defer my.deinit();
        try my.eval();
        const ds = try ops.subtract(ctx, serial.s_new, pipe.s_new);
        defer ds.deinit();
        const as_ = try ops.abs(ctx, ds);
        defer as_.deinit();
        const ms = try ops.max(ctx, as_);
        defer ms.deinit();
        try ms.eval();
        std.debug.print("L={d:5}  max|dy|={e:.3}  max|ds|={e:.3}", .{ L, try my.item(f32), try ms.item(f32) });

        if (L == 1901) {
            var t = std.c.mach_absolute_time();
            for (0..5) |_| {
                const r = (try gdn.testDeltaRuleSerial(ctx, s0, k, v, beta, g, q, HV, D)).?;
                defer r.s_new.deinit();
                defer r.y.deinit();
                try r.y.eval();
            }
            const t_serial = elapsedMs(t) / 5;
            t = std.c.mach_absolute_time();
            for (0..5) |_| {
                const r = (try gdn.testDeltaRulePipelined(ctx, s0, k, v, beta, g, q, HK, HV, D)).?;
                defer r.s_new.deinit();
                defer r.y.deinit();
                try r.y.eval();
            }
            const t_pipe = elapsedMs(t) / 5;
            std.debug.print("  serial {d:.2} ms  pipe {d:.2} ms  ({d:.1}x)", .{ t_serial, t_pipe, t_serial / t_pipe });
        }
        std.debug.print("\n", .{});
    }
}
