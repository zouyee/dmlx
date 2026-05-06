const std = @import("std");
const mlx = @import("../root.zig");
const c = @import("../c.zig");

const Array = mlx.Array;

test "save and load safetensors roundtrip" {
    const allocator = std.testing.allocator;

    const full_path = "/tmp/dmlx_test_roundtrip.safetensors";

    // Prepare weights
    var weights = std.StringHashMap(Array).init(allocator);
    defer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const arr = c.c.mlx_array_new_data(&data, &[_]c_int{ 2, 2 }, 2, c.c.MLX_FLOAT32);
    const owned_key = try allocator.dupe(u8, "weight");
    try weights.put(owned_key, Array.fromHandle(arr));

    // Prepare metadata
    var metadata = std.StringHashMap([]const u8).init(allocator);
    defer {
        var it = metadata.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        metadata.deinit();
    }
    const meta_key = try allocator.dupe(u8, "format");
    const meta_val = try allocator.dupe(u8, "test");
    try metadata.put(meta_key, meta_val);

    // Save
    try mlx.io.saveSafetensors(allocator, full_path, weights, metadata);

    // Load
    var loaded = try mlx.io.loadSafetensors(allocator, full_path);
    defer loaded.deinit(allocator);

    // Verify weights
    try std.testing.expectEqual(@as(usize, 1), loaded.weights.count());
    const loaded_arr = loaded.weights.get("weight").?;
    const shape = loaded_arr.shape();
    try std.testing.expectEqual(@as(i32, 2), shape[0]);
    try std.testing.expectEqual(@as(i32, 2), shape[1]);

    const loaded_data = try loaded_arr.dataPtr(f32);
    try std.testing.expectEqual(@as(f32, 1.0), loaded_data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), loaded_data[1]);
    try std.testing.expectEqual(@as(f32, 3.0), loaded_data[2]);
    try std.testing.expectEqual(@as(f32, 4.0), loaded_data[3]);

    // Verify metadata
    try std.testing.expectEqual(@as(usize, 1), loaded.metadata.count());
    try std.testing.expectEqualStrings("test", loaded.metadata.get("format").?);
}

test "loadSafetensors empty metadata" {
    const allocator = std.testing.allocator;

    const full_path = "/tmp/dmlx_test_empty.safetensors";

    var weights = std.StringHashMap(Array).init(allocator);
    defer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    const data = [_]f32{42.0};
    const arr = c.c.mlx_array_new_data(&data, &[_]c_int{1}, 1, c.c.MLX_FLOAT32);
    const owned_key = try allocator.dupe(u8, "scalar");
    try weights.put(owned_key, Array.fromHandle(arr));

    var metadata = std.StringHashMap([]const u8).init(allocator);
    defer {
        var it = metadata.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        metadata.deinit();
    }

    try mlx.io.saveSafetensors(allocator, full_path, weights, metadata);

    var loaded = try mlx.io.loadSafetensors(allocator, full_path);
    defer loaded.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), loaded.weights.count());
    try std.testing.expectEqual(@as(usize, 0), loaded.metadata.count());
}
