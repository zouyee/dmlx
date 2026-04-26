const std = @import("std");
const mlx = @import("../root.zig");

test "reduce ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 3 });
    defer a.deinit();

    const ctx = mlx.EagerContext.init(alloc);

    const s = try mlx.reduce.sumAxis(ctx, a, 0, false);
    defer s.deinit();
    try std.testing.expectEqual(s.ndim(), 1);

    const m = try mlx.reduce.meanAxis(ctx, a, 1, false);
    defer m.deinit();
    try std.testing.expectEqual(m.ndim(), 1);

    const am = try mlx.reduce.argmaxAxis(ctx, a, 0, false);
    defer am.deinit();

    const cs = try mlx.reduce.cumsum(ctx, a, 0, false, false);
    defer cs.deinit();
    try std.testing.expectEqual(cs.ndim(), 2);

    const tk = try mlx.reduce.topk(ctx, a, 2);
    defer tk.deinit();
}
