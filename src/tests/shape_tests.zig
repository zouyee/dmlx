const std = @import("std");
const mlx = @import("../root.zig");

test "shape ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 3 });
    defer a.deinit();

    const ctx = mlx.EagerContext.init(alloc);

    const r = try mlx.shape.reshape(ctx, a, &[_]i32{ 3, 2 });
    defer r.deinit();
    try std.testing.expectEqual(r.shape()[0], 3);
    try std.testing.expectEqual(r.shape()[1], 2);

    const t = try mlx.shape.transpose(ctx, a);
    defer t.deinit();
    try std.testing.expectEqual(t.shape()[0], 3);
    try std.testing.expectEqual(t.shape()[1], 2);

    const s = try mlx.shape.squeeze(ctx, a);
    defer s.deinit();
    try std.testing.expectEqual(s.ndim(), 2);

    const idx = try mlx.Array.fromData(alloc, i32, &[_]i32{0}, &[_]i32{1});
    defer idx.deinit();
    const tk = try mlx.shape.take(ctx, a, idx);
    defer tk.deinit();
}
