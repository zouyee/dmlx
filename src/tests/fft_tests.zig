const std = @import("std");
const mlx = @import("../root.zig");

test "fft ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{4});
    defer a.deinit();

    const ctx = mlx.EagerContext.init(alloc);

    const f = try mlx.fft.fft(ctx, a, 4, -1);
    defer f.deinit();
    try std.testing.expectEqual(f.ndim(), 1);

    const rf = try mlx.fft.rfft(ctx, a, 4, -1);
    defer rf.deinit();
    try std.testing.expectEqual(rf.ndim(), 1);
}
