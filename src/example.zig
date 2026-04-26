const std = @import("std");
const mlx = @import("root.zig");

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const b_data = [_]f32{ 5, 6, 7, 8 };
    const a = try mlx.Array.fromData(allocator, f32, &a_data, &[_]i32{ 2, 2 });
    defer a.deinit();
    const b = try mlx.Array.fromData(allocator, f32, &b_data, &[_]i32{ 2, 2 });
    defer b.deinit();

    const ctx = mlx.EagerContext.init(allocator);
    const c = try mlx.ops.matmul(ctx, a, b);
    defer c.deinit();

    std.debug.print("A @ B = {any}\n", .{try c.dataSlice(f32)});

    const logits = try mlx.Array.fromSlice(allocator, f32, &[_]f32{ 2.0, 1.0, 0.1 });
    defer logits.deinit();
    const probs = try mlx.ops.softmax(ctx, logits, &[_]i32{});
    defer probs.deinit();
    std.debug.print("softmax = {any}\n", .{try probs.dataSlice(f32)});
}
