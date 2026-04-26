const std = @import("std");
const mlx = @import("../root.zig");

test "sort ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 3, 1, 2 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{3});
    defer a.deinit();

    const ctx = mlx.EagerContext.init(alloc);

    const s = try mlx.sort.sort(ctx, a);
    defer s.deinit();
    try std.testing.expectEqual(s.ndim(), 1);

    const as = try mlx.sort.argsort(ctx, a);
    defer as.deinit();
    try std.testing.expectEqual(as.ndim(), 1);

    const pt = try mlx.sort.partition(ctx, a, 1);
    defer pt.deinit();
    try std.testing.expectEqual(pt.ndim(), 1);
}
