const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const ops = @import("../ops.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

test "mlx_take remap with 2D indices - exact scenario from logs" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Simulate the exact scenario from debug logs:
    // Router selected: [87, 0, 31, 0, 22, 0]
    // Unique experts (sorted): [0, 22, 31, 87]
    // Remap: remap[0]=0, remap[22]=1, remap[31]=2, remap[87]=3

    // Build remap array (256 elements, most are 0 as fallback)
    var remap_data = try allocator.alloc(i32, 256);
    defer allocator.free(remap_data);
    @memset(remap_data, 0); // Default to 0
    remap_data[0] = 0;
    remap_data[22] = 1;
    remap_data[31] = 2;
    remap_data[87] = 3;

    const remap = try Array.fromData(allocator, i32, remap_data, &[_]i32{256});
    defer remap.deinit();

    // Indices as 2D array [1, 6] - matching the actual shape from logs
    const indices_data = [_]u32{ 87, 0, 31, 0, 22, 0 };
    const indices = try Array.fromData(allocator, u32, &indices_data, &[_]i32{ 1, 6 });
    defer indices.deinit();

    // Perform mlx_take (same as in streamingForward)
    var remapped_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_take(&remapped_raw, remap.inner, indices.inner, ctx.stream.inner));
    const remapped = Array.fromHandle(remapped_raw);
    defer remapped.deinit();
    try remapped.eval();

    // Convert to u32 for easier comparison (same as in streamingForward)
    const remapped_u32 = try ops.astype(ctx, remapped, .uint32);
    defer remapped_u32.deinit();
    try remapped_u32.eval();

    // Check results using dataSlice (same method as in streamingForward)
    const result = try remapped_u32.dataSlice(u32);

    std.debug.print("\nTest results:\n", .{});
    std.debug.print("indices:  {any}\n", .{indices_data});
    std.debug.print("remapped: {any}\n", .{result[0..6]});
    std.debug.print("expected: [3, 0, 2, 0, 1, 0]\n", .{});

    // Verify each element
    std.debug.print("\nPer-element verification:\n", .{});
    for (indices_data, 0..) |idx, i| {
        const expected = remap_data[idx];
        const actual = @as(i32, @intCast(result[i]));
        const match = if (expected == actual) "✓" else "✗";
        std.debug.print("  [{d}]: indices={d} → remap[{d}]={d}, got {d} {s}\n", .{ i, idx, idx, expected, actual, match });
    }

    // Assert expected values
    try std.testing.expectEqual(@as(u32, 3), result[0]); // remap[87] = 3
    try std.testing.expectEqual(@as(u32, 0), result[1]); // remap[0] = 0
    try std.testing.expectEqual(@as(u32, 2), result[2]); // remap[31] = 2
    try std.testing.expectEqual(@as(u32, 0), result[3]); // remap[0] = 0
    try std.testing.expectEqual(@as(u32, 1), result[4]); // remap[22] = 1
    try std.testing.expectEqual(@as(u32, 0), result[5]); // remap[0] = 0
}

test "mlx_take remap with 1D indices - control test" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Same remap as above
    var remap_data = try allocator.alloc(i32, 256);
    defer allocator.free(remap_data);
    @memset(remap_data, 0);
    remap_data[0] = 0;
    remap_data[22] = 1;
    remap_data[31] = 2;
    remap_data[87] = 3;

    const remap = try Array.fromData(allocator, i32, remap_data, &[_]i32{256});
    defer remap.deinit();

    // Indices as 1D array [6] - flattened version
    const indices_data = [_]u32{ 87, 0, 31, 0, 22, 0 };
    const indices = try Array.fromData(allocator, u32, &indices_data, &[_]i32{6});
    defer indices.deinit();

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

    std.debug.print("\n1D Control test results:\n", .{});
    std.debug.print("indices:  {any}\n", .{indices_data});
    std.debug.print("remapped: {any}\n", .{result[0..6]});

    // Assert expected values
    try std.testing.expectEqual(@as(u32, 3), result[0]);
    try std.testing.expectEqual(@as(u32, 0), result[1]);
    try std.testing.expectEqual(@as(u32, 2), result[2]);
    try std.testing.expectEqual(@as(u32, 0), result[3]);
    try std.testing.expectEqual(@as(u32, 1), result[4]);
    try std.testing.expectEqual(@as(u32, 0), result[5]);
}

test "remap construction correctness (P1): remap[e] == index of e in unique_ids" {
    // Property P1 from design.md:
    // For all expert IDs `e` in `unique_ids`, `remap[e]` equals the index of `e` in `unique_ids`.
    // **Validates: Requirements R2**
    const allocator = std.testing.allocator;

    const n_experts: usize = 256;

    // Simulate unique_ids as sorted, deduplicated expert selections
    const unique_ids = [_]u32{ 0, 5, 22, 31, 87, 100, 200, 255 };

    // Build remap the same way streamingForward does
    var remap_data = try allocator.alloc(i32, n_experts);
    defer allocator.free(remap_data);
    @memset(remap_data, 0); // Default to 0
    for (unique_ids, 0..) |eid, i| {
        remap_data[eid] = @intCast(i);
    }

    // Verify P1: for every expert in unique_ids, remap[e] == its index
    for (unique_ids, 0..) |eid, i| {
        const expected: i32 = @intCast(i);
        try std.testing.expectEqual(expected, remap_data[eid]);
    }

    // Also verify that non-selected experts map to 0 (the default)
    // Pick a few IDs that are NOT in unique_ids
    const non_selected = [_]u32{ 1, 2, 10, 50, 99, 150, 254 };
    for (non_selected) |eid| {
        try std.testing.expectEqual(@as(i32, 0), remap_data[eid]);
    }
}

test "flatten-remap round-trip: manual remap matches expected output" {
    // This test verifies the manual remap approach used in streamingForward:
    // 1. Flatten 2D indices to 1D
    // 2. For each index, look up remap[index] manually
    // 3. Verify the result matches expected values
    // **Validates: Requirements R2, R4**
    const allocator = std.testing.allocator;

    const n_experts: usize = 256;

    // Scenario: router selects experts [87, 0, 31, 0, 22, 0] as 2D [1, 6]
    const indices_2d = [_]u32{ 87, 0, 31, 0, 22, 0 };
    const unique_ids = [_]u32{ 0, 22, 31, 87 }; // sorted

    // Build remap
    var remap_data = try allocator.alloc(i32, n_experts);
    defer allocator.free(remap_data);
    @memset(remap_data, 0);
    for (unique_ids, 0..) |eid, i| {
        remap_data[eid] = @intCast(i);
    }

    // Manual remap (same logic as streamingForward's manual remap)
    var remapped = try allocator.alloc(i32, indices_2d.len);
    defer allocator.free(remapped);
    for (indices_2d, 0..) |idx, i| {
        remapped[i] = remap_data[idx];
    }

    // Expected: [3, 0, 2, 0, 1, 0]
    // remap[87]=3, remap[0]=0, remap[31]=2, remap[0]=0, remap[22]=1, remap[0]=0
    const expected = [_]i32{ 3, 0, 2, 0, 1, 0 };
    for (expected, 0..) |exp, i| {
        try std.testing.expectEqual(exp, remapped[i]);
    }

    // Also verify via MLX Array round-trip to ensure fromData + dataSlice is consistent
    const ctx = EagerContext.init(allocator);

    const remap_arr = try Array.fromData(allocator, i32, remap_data, &[_]i32{@intCast(n_experts)});
    defer remap_arr.deinit();
    try remap_arr.eval();

    const remap_readback = try remap_arr.dataSlice(i32);

    // Verify the MLX array round-trip preserves remap values
    for (unique_ids, 0..) |eid, i| {
        const exp: i32 = @intCast(i);
        try std.testing.expectEqual(exp, remap_readback[eid]);
    }

    // Now do the manual remap using the MLX-backed remap data (same as streamingForward)
    var remapped_mlx = try allocator.alloc(i32, indices_2d.len);
    defer allocator.free(remapped_mlx);
    for (indices_2d, 0..) |idx, i| {
        remapped_mlx[i] = remap_readback[idx];
    }

    // Build an MLX array from the manual remap result
    const remapped_arr = try Array.fromData(allocator, i32, remapped_mlx, &[_]i32{@intCast(indices_2d.len)});
    defer remapped_arr.deinit();
    try remapped_arr.eval();

    const remapped_u32 = try ops.astype(ctx, remapped_arr, .uint32);
    defer remapped_u32.deinit();
    try remapped_u32.eval();

    const result = try remapped_u32.dataSlice(u32);

    // Final verification: MLX array contains the correct remapped values
    try std.testing.expectEqual(@as(u32, 3), result[0]); // remap[87] = 3
    try std.testing.expectEqual(@as(u32, 0), result[1]); // remap[0] = 0
    try std.testing.expectEqual(@as(u32, 2), result[2]); // remap[31] = 2
    try std.testing.expectEqual(@as(u32, 0), result[3]); // remap[0] = 0
    try std.testing.expectEqual(@as(u32, 1), result[4]); // remap[22] = 1
    try std.testing.expectEqual(@as(u32, 0), result[5]); // remap[0] = 0
}
