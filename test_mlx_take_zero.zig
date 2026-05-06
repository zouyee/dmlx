const std = @import("std");
const Array = @import("src/array.zig").Array;
const ops = @import("src/ops.zig");
const c = @import("src/c.zig");

const EagerContext = ops.EagerContext;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    c.initErrorHandler();
    const ctx = EagerContext.init(allocator);

    std.debug.print("\n=== Testing mlx_take with zeros ===\n\n", .{});

    // Create a simple remap array: [10, 20, 30, 40, 50]
    const remap_data = [_]i32{ 10, 20, 30, 40, 50 };
    const remap = try Array.fromData(allocator, i32, &remap_data, &[_]i32{5});
    defer remap.deinit();
    try remap.eval();

    std.debug.print("Remap array: {any}\n", .{remap_data});

    // Test 1: indices with zeros: [0, 1, 0, 2, 0, 3]
    const indices1_data = [_]u32{ 0, 1, 0, 2, 0, 3 };
    const indices1 = try Array.fromData(allocator, u32, &indices1_data, &[_]i32{6});
    defer indices1.deinit();
    try indices1.eval();

    std.debug.print("\nTest 1: indices = {any}\n", .{indices1_data});

    var remapped1_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_take(&remapped1_raw, remap.inner, indices1.inner, ctx.stream.inner));
    const remapped1 = Array.fromHandle(remapped1_raw);
    defer remapped1.deinit();
    try remapped1.eval();

    const result1 = try remapped1.dataSlice(i32);
    std.debug.print("Expected: {{ 10, 20, 10, 30, 10, 40 }}\n", .{});
    std.debug.print("Got:      {any}\n", .{result1});

    // Test 2: indices without zeros: [1, 2, 3, 4, 1, 2]
    const indices2_data = [_]u32{ 1, 2, 3, 4, 1, 2 };
    const indices2 = try Array.fromData(allocator, u32, &indices2_data, &[_]i32{6});
    defer indices2.deinit();
    try indices2.eval();

    std.debug.print("\nTest 2: indices = {any}\n", .{indices2_data});

    var remapped2_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_take(&remapped2_raw, remap.inner, indices2.inner, ctx.stream.inner));
    const remapped2 = Array.fromHandle(remapped2_raw);
    defer remapped2.deinit();
    try remapped2.eval();

    const result2 = try remapped2.dataSlice(i32);
    std.debug.print("Expected: {{ 20, 30, 40, 50, 20, 30 }}\n", .{});
    std.debug.print("Got:      {any}\n", .{result2});

    std.debug.print("\n=== Test complete ===\n", .{});
}
