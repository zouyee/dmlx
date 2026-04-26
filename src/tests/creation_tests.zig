const std = @import("std");
const mlx = @import("../root.zig");

test "creation ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const ctx = mlx.EagerContext.init(alloc);

    const z = try mlx.creation.zeros(ctx, &[_]i32{ 2, 3 }, .float32);
    defer z.deinit();
    try std.testing.expectEqual(z.size(), 6);

    const o = try mlx.creation.ones(ctx, &[_]i32{ 3, 3 }, .float32);
    defer o.deinit();
    try std.testing.expectEqual(o.size(), 9);

    const e = try mlx.creation.eye(ctx, 4, 4, .float32);
    defer e.deinit();
    try std.testing.expectEqual(e.shape()[0], 4);
    try std.testing.expectEqual(e.shape()[1], 4);

    const ar = try mlx.creation.arange(ctx, 0, 10, 1, .float32);
    defer ar.deinit();
    try std.testing.expectEqual(ar.size(), 10);

    const ls = try mlx.creation.linspace(ctx, 0, 1, 5, .float32);
    defer ls.deinit();
    try std.testing.expectEqual(ls.size(), 5);
}
