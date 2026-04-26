const std = @import("std");
const mlx = @import("../root.zig");

test "array creation and properties" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const data = [_]f32{ 1, 2, 3, 4 };
    const arr = try mlx.Array.fromData(alloc, f32, &data, &[_]i32{ 2, 2 });
    defer arr.deinit();

    try std.testing.expectEqual(arr.ndim(), 2);
    try std.testing.expectEqual(arr.size(), 4);
    try std.testing.expectEqual(arr.shape()[0], 2);
    try std.testing.expectEqual(arr.shape()[1], 2);
    try std.testing.expectEqual(arr.dtype(), .float32);
}

test "matmul" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const b_data = [_]f32{ 5, 6, 7, 8 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 2 });
    defer a.deinit();
    const b = try mlx.Array.fromData(alloc, f32, &b_data, &[_]i32{ 2, 2 });
    defer b.deinit();

    const ctx = mlx.EagerContext.init(alloc);
    const c = try mlx.ops.matmul(ctx, a, b);
    defer c.deinit();

    const result = try c.dataSlice(f32);
    try std.testing.expectEqual(result[0], 19.0);
    try std.testing.expectEqual(result[1], 22.0);
    try std.testing.expectEqual(result[2], 43.0);
    try std.testing.expectEqual(result[3], 50.0);
}

test "element-wise ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 2 });
    defer a.deinit();
    const b = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 2 });
    defer b.deinit();

    const ctx = mlx.EagerContext.init(alloc);
    const sum_arr = try mlx.ops.add(ctx, a, b);
    defer sum_arr.deinit();
    const sum_data = try sum_arr.dataSlice(f32);
    try std.testing.expectEqual(sum_data[0], 2.0);
    try std.testing.expectEqual(sum_data[1], 4.0);
}

test "softmax" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const logits = try mlx.Array.fromSlice(alloc, f32, &[_]f32{ 2.0, 1.0, 0.1 });
    defer logits.deinit();

    const ctx = mlx.EagerContext.init(alloc);
    const probs = try mlx.ops.softmax(ctx, logits, &[_]i32{});
    defer probs.deinit();

    const p = try probs.dataSlice(f32);
    try std.testing.expectApproxEqAbs(p[0], 0.659, 0.01);
    try std.testing.expectApproxEqAbs(p[1], 0.242, 0.01);
    try std.testing.expectApproxEqAbs(p[2], 0.098, 0.01);
}

test "zeros and ones" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const z = try mlx.Array.zeros(alloc, &[_]i32{ 2, 3 }, .float32);
    defer z.deinit();
    try std.testing.expectEqual(z.size(), 6);
    const z_data = try z.dataSlice(f32);
    for (z_data) |v| try std.testing.expectEqual(v, 0.0);

    const o = try mlx.Array.ones(alloc, &[_]i32{ 3, 3 }, .float32);
    defer o.deinit();
    const o_data = try o.dataSlice(f32);
    for (o_data) |v| try std.testing.expectEqual(v, 1.0);
}
