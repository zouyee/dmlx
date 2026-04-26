const std = @import("std");
const mlx = @import("../root.zig");

test "comparison ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 2 });
    defer a.deinit();
    const b = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 2 });
    defer b.deinit();

    const ctx = mlx.EagerContext.init(alloc);

    const eq = try mlx.comparison.equal(ctx, a, b);
    defer eq.deinit();
    try std.testing.expectEqual(eq.ndim(), 2);

    const gt = try mlx.comparison.greater(ctx, a, b);
    defer gt.deinit();
    try std.testing.expectEqual(gt.ndim(), 2);

    const all_r = try mlx.comparison.all(ctx, a, false);
    defer all_r.deinit();
    try std.testing.expectEqual(all_r.ndim(), 0);

    const any_r = try mlx.comparison.any(ctx, a, false);
    defer any_r.deinit();
    try std.testing.expectEqual(any_r.ndim(), 0);

    const allclose_r = try mlx.comparison.allClose(ctx, a, b, 1e-5, 1e-8, false);
    defer allclose_r.deinit();
    try std.testing.expectEqual(allclose_r.ndim(), 0);
}
