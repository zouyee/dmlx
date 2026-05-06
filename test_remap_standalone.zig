const std = @import("std");
const c = @import("src/c.zig");
const array_mod = @import("src/array.zig");
const ops = @import("src/ops.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    c.initErrorHandler();
    const ctx = EagerContext.init(allocator);
    
    std.debug.print("\n=== Testing mlx_take remap behavior ===\n\n", .{});
    
    // Test 1: 2D indices (actual scenario)
    std.debug.print("Test 1: 2D indices [1, 6]\n", .{});
    try testRemap2D(allocator, ctx);
    
    // Test 2: 1D indices (control)
    std.debug.print("\nTest 2: 1D indices [6] (control)\n", .{});
    try testRemap1D(allocator, ctx);
}

fn testRemap2D(allocator: std.mem.Allocator, ctx: EagerContext) !void {
    // Build remap array
    var remap_data = try allocator.alloc(i32, 256);
    defer allocator.free(remap_data);
    @memset(remap_data, 0);
    remap_data[0] = 0;
    remap_data[22] = 1;
    remap_data[31] = 2;
    remap_data[87] = 3;
    
    const remap = try Array.fromData(allocator, i32, remap_data, &[_]i32{256});
    defer remap.deinit();
    
    // Indices as 2D array [1, 6]
    const indices_data = [_]u32{ 87, 0, 31, 0, 22, 0 };
    const indices = try Array.fromData(allocator, u32, &indices_data, &[_]i32{1, 6});
    defer indices.deinit();
    
    std.debug.print("  indices shape: {any}\n", .{indices.shape()});
    std.debug.print("  indices data:  {any}\n", .{indices_data});
    
    // Perform mlx_take
    var remapped_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_take(&remapped_raw, remap.inner, indices.inner, ctx.stream.inner));
    const remapped = Array.fromHandle(remapped_raw);
    defer remapped.deinit();
    try remapped.eval();
    
    const remapped_u32 = try ops.astype(ctx, remapped, .uint32);
    defer remapped_u32.deinit();
    try remapped_u32.eval();
    
    const result = try remapped_u32.dataSlice(u32);
    
    std.debug.print("  remapped shape: {any}\n", .{remapped_u32.shape()});
    std.debug.print("  remapped data:  {any}\n", .{result[0..6]});
    std.debug.print("  expected:       [3, 0, 2, 0, 1, 0]\n", .{});
    
    // Verify
    std.debug.print("\n  Verification:\n", .{});
    var all_correct = true;
    for (indices_data, 0..) |idx, i| {
        const expected = remap_data[idx];
        const actual = @as(i32, @intCast(result[i]));
        const match = expected == actual;
        if (!match) all_correct = false;
        const symbol = if (match) "✓" else "✗";
        std.debug.print("    [{d}]: indices={d} → remap[{d}]={d}, got {d} {s}\n", 
            .{ i, idx, idx, expected, actual, symbol });
    }
    
    if (all_correct) {
        std.debug.print("\n  ✅ Test PASSED\n", .{});
    } else {
        std.debug.print("\n  ❌ Test FAILED\n", .{});
    }
}

fn testRemap1D(allocator: std.mem.Allocator, ctx: EagerContext) !void {
    // Build remap array
    var remap_data = try allocator.alloc(i32, 256);
    defer allocator.free(remap_data);
    @memset(remap_data, 0);
    remap_data[0] = 0;
    remap_data[22] = 1;
    remap_data[31] = 2;
    remap_data[87] = 3;
    
    const remap = try Array.fromData(allocator, i32, remap_data, &[_]i32{256});
    defer remap.deinit();
    
    // Indices as 1D array [6]
    const indices_data = [_]u32{ 87, 0, 31, 0, 22, 0 };
    const indices = try Array.fromData(allocator, u32, &indices_data, &[_]i32{6});
    defer indices.deinit();
    
    std.debug.print("  indices shape: {any}\n", .{indices.shape()});
    std.debug.print("  indices data:  {any}\n", .{indices_data});
    
    // Perform mlx_take
    var remapped_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_take(&remapped_raw, remap.inner, indices.inner, ctx.stream.inner));
    const remapped = Array.fromHandle(remapped_raw);
    defer remapped.deinit();
    try remapped.eval();
    
    const remapped_u32 = try ops.astype(ctx, remapped, .uint32);
    defer remapped_u32.deinit();
    try remapped_u32.eval();
    
    const result = try remapped_u32.dataSlice(u32);
    
    std.debug.print("  remapped shape: {any}\n", .{remapped_u32.shape()});
    std.debug.print("  remapped data:  {any}\n", .{result[0..6]});
    std.debug.print("  expected:       [3, 0, 2, 0, 1, 0]\n", .{});
    
    // Verify
    std.debug.print("\n  Verification:\n", .{});
    var all_correct = true;
    for (indices_data, 0..) |idx, i| {
        const expected = remap_data[idx];
        const actual = @as(i32, @intCast(result[i]));
        const match = expected == actual;
        if (!match) all_correct = false;
        const symbol = if (match) "✓" else "✗";
        std.debug.print("    [{d}]: indices={d} → remap[{d}]={d}, got {d} {s}\n", 
            .{ i, idx, idx, expected, actual, symbol });
    }
    
    if (all_correct) {
        std.debug.print("\n  ✅ Test PASSED\n", .{});
    } else {
        std.debug.print("\n  ❌ Test FAILED\n", .{});
    }
}
