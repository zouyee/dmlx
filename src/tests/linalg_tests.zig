const std = @import("std");
const mlx = @import("../root.zig");

test "linalg ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 4, 7, 2, 6 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 2 });
    defer a.deinit();

    const ctx = mlx.EagerContext.init(alloc);

    const inv_a = try mlx.linalg.inv(ctx, a);
    defer inv_a.deinit();
    try std.testing.expectEqual(inv_a.ndim(), 2);

    const qr = try mlx.linalg.qr(ctx, a);
    defer qr.q.deinit();
    defer qr.r.deinit();
    try std.testing.expectEqual(qr.q.ndim(), 2);
    try std.testing.expectEqual(qr.r.ndim(), 2);
}
