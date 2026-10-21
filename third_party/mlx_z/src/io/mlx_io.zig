/// I/O operations backed by mlx-c (safetensors, gguf, npy via mlx_load/mlx_save).
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");

const Array = array_mod.Array;

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

    const stream = c.c.mlx_default_cpu_stream_new();
    // NOTE: stream is intentionally NOT freed here. MLX load operations are lazy —
    // the returned arrays reference this stream for deferred evaluation. Freeing
    // the stream before eval would cause data load failures. The stream will be
    // freed when the process exits or when MLX's internal refcount drops to zero.
    try c.check(c.c.mlx_load_safetensors(&weights_map, &metadata_map, file_z.ptr, stream));

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
    const stream = c.c.mlx_default_cpu_stream_new();
    // NOTE: stream intentionally not freed — MLX load is lazy, array references stream.
    try c.check(c.c.mlx_load(&res, file_z.ptr, stream));
    return Array.fromHandle(res);
}

/// Save a single array to file.
pub fn save(allocator: std.mem.Allocator, path: []const u8, arr: Array) !void {
    const file_z = try allocator.dupeZ(u8, path);
    defer allocator.free(file_z);
    try c.check(c.c.mlx_save(file_z.ptr, arr.inner));
}

// === GGUF I/O ===

/// GGUF container object.
pub const Gguf = struct {
    inner: c.c.mlx_io_gguf,

    pub fn new() Gguf {
        return .{ .inner = c.c.mlx_io_gguf_new() };
    }

    pub fn deinit(self: Gguf) void {
        _ = c.c.mlx_io_gguf_free(self.inner);
    }

    /// Load a GGUF file and return a parsed GGUF object.
    pub fn load(allocator: std.mem.Allocator, path: []const u8) !Gguf {
        const file_z = try allocator.dupeZ(u8, path);
        defer allocator.free(file_z);
        var gguf = c.c.mlx_io_gguf_new();
        const stream = c.c.mlx_default_cpu_stream_new();
        try c.check(c.c.mlx_load_gguf(&gguf, file_z.ptr, stream));
        return .{ .inner = gguf };
    }

    /// Save a GGUF object to file.
    pub fn save(self: Gguf, allocator: std.mem.Allocator, path: []const u8) !void {
        const file_z = try allocator.dupeZ(u8, path);
        defer allocator.free(file_z);
        try c.check(c.c.mlx_save_gguf(file_z.ptr, self.inner));
    }

    /// Get all tensor keys in the GGUF file.
    /// Note: full key extraction requires mlx_vector_string iteration APIs
    /// not yet exposed in the C header. Use hasMetadataString/hasMetadataArray
    /// for probing, or call getArray directly with known keys.
    pub fn getKeys(self: Gguf) !void {
        var keys_vec: c.c.mlx_vector_string = undefined;
        try c.check(c.c.mlx_io_gguf_get_keys(&keys_vec, self.inner));
        // Free immediately since we can't iterate vector_string elements yet.
        _ = c.c.mlx_vector_string_free(keys_vec);
    }

    /// Get a tensor array by key.
    pub fn getArray(self: Gguf, key: []const u8) !Array {
        var arr = c.c.mlx_array_new();
        try c.check(c.c.mlx_io_gguf_get_array(&arr, self.inner, key.ptr));
        return Array.fromHandle(arr);
    }

    /// Set a tensor array by key.
    pub fn setArray(self: Gguf, key: []const u8, arr: Array) !void {
        try c.check(c.c.mlx_io_gguf_set_array(self.inner, key.ptr, arr.inner));
    }

    /// Get a metadata string.
    pub fn getMetadataString(self: Gguf, key: []const u8) ![]const u8 {
        var str: c.c.mlx_string = undefined;
        try c.check(c.c.mlx_io_gguf_get_metadata_string(&str, self.inner, key.ptr));
        return c.c.mlx_string_data(str);
    }

    /// Set a metadata string.
    pub fn setMetadataString(self: Gguf, key: []const u8, value: []const u8) !void {
        try c.check(c.c.mlx_io_gguf_set_metadata_string(self.inner, key.ptr, value.ptr));
    }

    /// Check if a metadata array exists.
    pub fn hasMetadataArray(self: Gguf, key: []const u8) !bool {
        var flag: bool = false;
        try c.check(c.c.mlx_io_gguf_has_metadata_array(&flag, self.inner, key.ptr));
        return flag;
    }

    /// Check if a metadata string exists.
    pub fn hasMetadataString(self: Gguf, key: []const u8) !bool {
        var flag: bool = false;
        try c.check(c.c.mlx_io_gguf_has_metadata_string(&flag, self.inner, key.ptr));
        return flag;
    }

    /// Get metadata array by key.
    pub fn getMetadataArray(self: Gguf, key: []const u8) !Array {
        var arr = c.c.mlx_array_new();
        try c.check(c.c.mlx_io_gguf_get_metadata_array(&arr, self.inner, key.ptr));
        return Array.fromHandle(arr);
    }

    /// Set metadata array by key.
    pub fn setMetadataArray(self: Gguf, key: []const u8, arr: Array) !void {
        try c.check(c.c.mlx_io_gguf_set_metadata_array(self.inner, key.ptr, arr.inner));
    }
};

/// Load a GGUF file directly (shortcut for Gguf.load).
pub fn loadGguf(allocator: std.mem.Allocator, path: []const u8) !Gguf {
    return Gguf.load(allocator, path);
}

/// Save a GGUF file directly (shortcut).
pub fn saveGguf(gguf: Gguf, allocator: std.mem.Allocator, path: []const u8) !void {
    return gguf.save(allocator, path);
}
