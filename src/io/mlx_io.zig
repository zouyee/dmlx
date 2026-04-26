/// I/O operations backed by mlx-c (safetensors, gguf, npy via mlx_load/mlx_save).
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");

const Array = array_mod.Array;

inline fn defaultStream() c.c.mlx_stream {
    return c.c.mlx_default_cpu_stream_new();
}

pub const SafetensorsResult = struct {
    weights: std.StringHashMap(Array),
    metadata: std.StringHashMap([]const u8),

    pub fn deinit(self: *SafetensorsResult, allocator: std.mem.Allocator) void {
        var w_it = self.weights.iterator();
        while (w_it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        self.weights.deinit();
        var m_it = self.metadata.iterator();
        while (m_it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }
};

/// Load a safetensors file into weight and metadata maps.
pub fn loadSafetensors(allocator: std.mem.Allocator, path: []const u8) !SafetensorsResult {
    const file_z = try allocator.dupeZ(u8, path);
    defer allocator.free(file_z);

    var weights_map = c.c.mlx_map_string_to_array_new();
    defer _ = c.c.mlx_map_string_to_array_free(weights_map);
    var metadata_map = c.c.mlx_map_string_to_string_new();
    defer _ = c.c.mlx_map_string_to_string_free(metadata_map);

    try c.check(c.c.mlx_load_safetensors(&weights_map, &metadata_map, file_z.ptr, defaultStream()));

    var weights = std.StringHashMap(Array).init(allocator);
    var metadata = std.StringHashMap([]const u8).init(allocator);

    // Iterate weights
    const w_it = c.c.mlx_map_string_to_array_iterator_new(weights_map);
    defer _ = c.c.mlx_map_string_to_array_iterator_free(w_it);
    while (true) {
        var key_ptr: [*c]const u8 = null;
        var val = c.c.mlx_array{ .ctx = null };
        const rc = c.c.mlx_map_string_to_array_iterator_next(&key_ptr, &val, w_it);
        if (rc != 0 or key_ptr == null) break;
        const key = try allocator.dupe(u8, std.mem.span(key_ptr));
        var copied = c.c.mlx_array_new();
        try c.check(c.c.mlx_array_set(&copied, val));
        try weights.put(key, Array.fromHandle(copied));
    }

    // Iterate metadata
    const m_it = c.c.mlx_map_string_to_string_iterator_new(metadata_map);
    defer _ = c.c.mlx_map_string_to_string_iterator_free(m_it);
    while (true) {
        var key_ptr: [*c]const u8 = null;
        var val_ptr: [*c]const u8 = null;
        const rc = c.c.mlx_map_string_to_string_iterator_next(&key_ptr, &val_ptr, m_it);
        if (rc != 0 or key_ptr == null) break;
        const key = try allocator.dupe(u8, std.mem.span(key_ptr));
        const val = try allocator.dupe(u8, std.mem.span(val_ptr));
        try metadata.put(key, val);
    }

    return .{ .weights = weights, .metadata = metadata };
}

/// Save weights and metadata to a safetensors file.
pub fn saveSafetensors(
    allocator: std.mem.Allocator,
    path: []const u8,
    weights: std.StringHashMap(Array),
    metadata: std.StringHashMap([]const u8),
) !void {
    const file_z = try allocator.dupeZ(u8, path);
    defer allocator.free(file_z);

    const weights_map = c.c.mlx_map_string_to_array_new();
    defer _ = c.c.mlx_map_string_to_array_free(weights_map);
    const metadata_map = c.c.mlx_map_string_to_string_new();
    defer _ = c.c.mlx_map_string_to_string_free(metadata_map);

    var w_it = weights.iterator();
    while (w_it.next()) |entry| {
        const key_z = try allocator.dupeZ(u8, entry.key_ptr.*);
        defer allocator.free(key_z);
        try c.check(c.c.mlx_map_string_to_array_insert(weights_map, key_z.ptr, entry.value_ptr.*.inner));
    }

    var m_it = metadata.iterator();
    while (m_it.next()) |entry| {
        const key_z = try allocator.dupeZ(u8, entry.key_ptr.*);
        defer allocator.free(key_z);
        const val_z = try allocator.dupeZ(u8, entry.value_ptr.*);
        defer allocator.free(val_z);
        try c.check(c.c.mlx_map_string_to_string_insert(metadata_map, key_z.ptr, val_z.ptr));
    }

    try c.check(c.c.mlx_save_safetensors(file_z.ptr, weights_map, metadata_map));
}

/// Load a single array from file (supports GGUF and other mlx formats).
pub fn load(allocator: std.mem.Allocator, path: []const u8) !Array {
    const file_z = try allocator.dupeZ(u8, path);
    defer allocator.free(file_z);
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_load(&res, file_z.ptr, defaultStream()));
    return Array.fromHandle(res);
}

/// Save a single array to file.
pub fn save(allocator: std.mem.Allocator, path: []const u8, arr: Array) !void {
    const file_z = try allocator.dupeZ(u8, path);
    defer allocator.free(file_z);
    try c.check(c.c.mlx_save(file_z.ptr, arr.inner));
}
