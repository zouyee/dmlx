const std = @import("std");
const qwen38 = @import("models/qwen38.zig");
const mlx = @import("mlx");
const array_mod = mlx.array;
const ops = mlx.ops;
const Array = array_mod.Array;

fn sliceF32(ctx: ops.EagerContext, a: Array) ![]const f32 {
    const f = try ops.astype(ctx, a, .float32);
    try f.eval();
    return f.dataSlice(f32) catch unreachable;
}

pub fn main() !void {
    mlx.c.initErrorHandler();
    const allocator = std.heap.smp_allocator;
    const ctx = ops.EagerContext.init(allocator);
    for ([_]i32{ 5120, 64 }) |n| {
        const hbuf = try allocator.alloc(f32, @intCast(n));
        defer allocator.free(hbuf);
        const bbuf = try allocator.alloc(f32, @intCast(n));
        defer allocator.free(bbuf);
        const wbuf = try allocator.alloc(f32, @intCast(n));
        defer allocator.free(wbuf);
        for (hbuf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 37 + 11, 23) - 11)) * 0.05;
        for (bbuf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 19 + 7, 17) - 8)) * 0.03;
        for (wbuf, 0..) |*v, i| v.* = 0.5 + @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 3, 11))) * 0.1;

        const h_f32 = try Array.fromData(allocator, f32, hbuf, &[_]i32{ 1, 1, n });
        defer h_f32.deinit();
        const b_f32 = try Array.fromData(allocator, f32, bbuf, &[_]i32{ 1, 1, n });
        defer b_f32.deinit();
        const h = try ops.astype(ctx, h_f32, .bfloat16);
        defer h.deinit();
        const branch = try ops.astype(ctx, b_f32, .bfloat16);
        defer branch.deinit();
        const w = try Array.fromData(allocator, f32, wbuf, &[_]i32{n});
        defer w.deinit();

        const eager = try qwen38.addRmsNormEager(ctx, h, branch, w, 1e-6);
        defer eager.h.deinit();
        defer eager.normed.deinit();
        const fused = (try qwen38.addRmsNormFused(ctx, h, branch, w, 1e-6)) orelse {
            std.debug.print("n={d}: kernel null\n", .{n});
            continue;
        };
        defer fused.h.deinit();
        defer fused.normed.deinit();

        const fh = try ops.astype(ctx, fused.h, .float32);
        defer fh.deinit();
        const fno = try ops.astype(ctx, fused.normed, .float32);
        defer fno.deinit();
        const eh = try ops.astype(ctx, eager.h, .float32);
        defer eh.deinit();
        const eno = try ops.astype(ctx, eager.normed, .float32);
        defer eno.deinit();
        try fh.eval();
        try fno.eval();
        try eh.eval();
        try eno.eval();
        const got_h = try fh.dataSlice(f32);
        const got_n = try fno.dataSlice(f32);
        const exp_h = try eh.dataSlice(f32);
        const exp_n = try eno.dataSlice(f32);

        var bad_h: usize = 0;
        var bad_n: usize = 0;
        var max_dn: f32 = 0;
        for (exp_h, got_h) |e, o| {
            if (e != o) bad_h += 1;
        }
        for (exp_n, got_n) |e, o| {
            const d = @abs(e - o);
            if (d > max_dn) max_dn = d;
            if (d > 2e-2) bad_n += 1;
        }
        std.debug.print("n={d}: bad_h={d} bad_n={d} max_dn={d} got_h0={d} exp_h0={d} got_n0={d} exp_n0={d}\n", .{ n, bad_h, bad_n, max_dn, got_h[0], exp_h[0], got_n[0], exp_n[0] });
    }
}
