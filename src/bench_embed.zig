// Scratch: embed gather cost — takeAxis on [248320, 5120] bf16, 1901 rows.
// Run: zig build run-bench-embed -Doptimize=ReleaseFast
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;

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

    const table = try Array.zeros(allocator, &[_]i32{ 248320, 5120 }, .bfloat16);
    defer table.deinit();
    try table.eval();

    var idx_buf: [1901]i32 = undefined;
    for (&idx_buf, 0..) |*v, i| v.* = @intCast(i * 97 % 248320);
    const idx = try Array.fromData(allocator, i32, &idx_buf, &[_]i32{1901});
    defer idx.deinit();

    // first call (includes JIT)
    var t = std.c.mach_absolute_time();
    {
        const rows = try shape_mod.takeAxis(ctx, table, idx, 0);
        defer rows.deinit();
        try rows.eval();
    }
    std.debug.print("first: {d:.2} ms\n", .{elapsedMs(t)});
    for (0..5) |i| {
        t = std.c.mach_absolute_time();
        const rows = try shape_mod.takeAxis(ctx, table, idx, 0);
        defer rows.deinit();
        try rows.eval();
        std.debug.print("warm {d}: {d:.3} ms\n", .{ i, elapsedMs(t) });
    }
}
