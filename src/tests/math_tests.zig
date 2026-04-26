const std = @import("std");
const mlx = @import("../root.zig");

test "math ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 2 });
    defer a.deinit();
    const b = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 2 });
    defer b.deinit();

    const ctx = mlx.EagerContext.init(alloc);

    const s = try mlx.math.sign(ctx, a);
    defer s.deinit();
    try std.testing.expectEqual(s.ndim(), 2);

    const f = try mlx.math.floor(ctx, a);
    defer f.deinit();
    try std.testing.expectEqual(f.ndim(), 2);

    const r = try mlx.math.round(ctx, a, 0);
    defer r.deinit();
    try std.testing.expectEqual(r.ndim(), 2);

    const mx = try mlx.math.maximum(ctx, a, b);
    defer mx.deinit();
    try std.testing.expectEqual(mx.ndim(), 2);

    const p = try mlx.math.power(ctx, a, b);
    defer p.deinit();
    try std.testing.expectEqual(p.ndim(), 2);
}
