const std = @import("std");
const mlx = @import("../root.zig");

test "random ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const ctx = mlx.EagerContext.init(alloc);

    const k = try mlx.random.key(42);
    defer k.deinit();

    const n = try mlx.random.normal(ctx, &[_]i32{ 2, 2 }, .float32, 0.0, 1.0, k);
    defer n.deinit();
    try std.testing.expectEqual(n.ndim(), 2);

    const u = try mlx.random.uniform(ctx, k, k, &[_]i32{ 2, 2 }, .float32, k);
    defer u.deinit();
    try std.testing.expectEqual(u.ndim(), 2);
}
