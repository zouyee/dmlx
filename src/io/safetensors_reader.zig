/// Safetensors random-access reader.
///
/// Parses safetensors file headers to build a tensor index, then loads
/// individual tensors by file offset using pread — without loading the
/// entire file into memory.
///
/// Format: [8-byte LE u64 header_len] [JSON header] [tensor data...]
/// Header JSON: { "tensor_name": { "dtype": "BF16", "shape": [M, N], "data_offsets": [start, end] }, ... }
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");

const Array = array_mod.Array;

/// Metadata for a single tensor in a safetensors file.
pub const TensorInfo = struct {
    dtype_str: []const u8, // "F32", "BF16", "I32", "U8", "U32", etc.
    shape: []const i64,
    data_offset_start: u64,
    data_offset_end: u64,
    shard_path: []const u8,
};

/// Index of all tensors across all shards.
pub const TensorIndex = struct {
    allocator: std.mem.Allocator,
    entries: std.StringHashMap(TensorInfo),

    pub fn init(allocator: std.mem.Allocator) TensorIndex {
        return .{
            .allocator = allocator,
            .entries = std.StringHashMap(TensorInfo).init(allocator),
        };
    }

    pub fn deinit(self: *TensorIndex) void {
        var it = self.entries.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*.dtype_str);
            self.allocator.free(entry.value_ptr.*.shape);
            self.allocator.free(entry.value_ptr.*.shard_path);
        }
        self.entries.deinit();
    }

    /// Parse one safetensors file header and add its tensors to the index.
    pub fn addShard(self: *TensorIndex, shard_path: []const u8) !void {
        const posix = @cImport(@cInclude("fcntl.h"));
        const unistd = @cImport(@cInclude("unistd.h"));

        const path_z = try self.allocator.dupeZ(u8, shard_path);
        defer self.allocator.free(path_z);
        const fd = posix.open(path_z.ptr, posix.O_RDONLY);
        if (fd < 0) return error.FileNotFound;
        defer _ = unistd.close(fd);

        // Read 8-byte header length (little-endian u64)
        var header_len_buf: [8]u8 = undefined;
        const r1 = unistd.pread(fd, &header_len_buf, 8, 0);
        if (r1 < 8) return error.InvalidSafetensors;
        const header_len = std.mem.readInt(u64, &header_len_buf, .little);
        if (header_len > 100 * 1024 * 1024) return error.HeaderTooLarge;

        // Read header JSON
        const header_buf = try self.allocator.alloc(u8, @intCast(header_len));
        defer self.allocator.free(header_buf);
        const r2 = unistd.pread(fd, header_buf.ptr, @intCast(header_len), 8);
        if (r2 < @as(isize, @intCast(header_len))) return error.InvalidSafetensors;

        const data_base: u64 = 8 + header_len;

        // Parse JSON
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, header_buf, .{});
        defer parsed.deinit();

        const obj = parsed.value.object;
        var obj_it = obj.iterator();
        while (obj_it.next()) |entry| {
            const tensor_name = entry.key_ptr.*;
            const tensor_val = entry.value_ptr.*;

            // Skip __metadata__
            if (std.mem.eql(u8, tensor_name, "__metadata__")) continue;
            if (tensor_val != .object) continue;

            const tobj = tensor_val.object;
            const dtype_val = tobj.get("dtype") orelse continue;
            const shape_val = tobj.get("shape") orelse continue;
            const offsets_val = tobj.get("data_offsets") orelse continue;

            if (dtype_val != .string) continue;
            if (shape_val != .array) continue;
            if (offsets_val != .array) continue;

            const offsets = offsets_val.array.items;
            if (offsets.len != 2) continue;

            const start: u64 = switch (offsets[0]) {
                .integer => |n| @intCast(n),
                .float => |f| @intFromFloat(f),
                else => continue,
            };
            const end: u64 = switch (offsets[1]) {
                .integer => |n| @intCast(n),
                .float => |f| @intFromFloat(f),
                else => continue,
            };

            // Parse shape
            var shape_list = try self.allocator.alloc(i64, shape_val.array.items.len);
            for (shape_val.array.items, 0..) |s, idx| {
                shape_list[idx] = switch (s) {
                    .integer => |n| n,
                    .float => |f| @intFromFloat(f),
                    else => 0,
                };
            }

            const key = try self.allocator.dupe(u8, tensor_name);
            const info = TensorInfo{
                .dtype_str = try self.allocator.dupe(u8, dtype_val.string),
                .shape = shape_list,
                .data_offset_start = data_base + start,
                .data_offset_end = data_base + end,
                .shard_path = try self.allocator.dupe(u8, shard_path),
            };
            try self.entries.put(key, info);
        }
    }

    /// Load a single tensor by name from the index.
    pub fn loadTensor(self: *TensorIndex, name: []const u8) !Array {
        const info = self.entries.get(name) orelse return error.TensorNotFound;
        const data_len: usize = @intCast(info.data_offset_end - info.data_offset_start);

        const posix = @cImport(@cInclude("fcntl.h"));
        const unistd = @cImport(@cInclude("unistd.h"));

        const path_z = try self.allocator.dupeZ(u8, info.shard_path);
        defer self.allocator.free(path_z);
        const fd = posix.open(path_z.ptr, posix.O_RDONLY);
        if (fd < 0) return error.FileNotFound;
        defer _ = unistd.close(fd);

        const buf = try self.allocator.alloc(u8, data_len);
        defer self.allocator.free(buf);

        const read_len = unistd.pread(fd, buf.ptr, data_len, @intCast(info.data_offset_start));
        if (read_len < @as(isize, @intCast(data_len))) return error.IncompleteRead;

        // Map dtype string to MLX dtype
        const mlx_dtype = dtypeFromString(info.dtype_str) orelse return error.UnsupportedDtype;

        // Build shape as i32 array
        var shape_i32 = try self.allocator.alloc(i32, info.shape.len);
        defer self.allocator.free(shape_i32);
        for (info.shape, 0..) |s, idx| {
            shape_i32[idx] = @intCast(s);
        }

        // Create MLX array from raw data
        const arr = c.c.mlx_array_new_data(
            buf.ptr,
            shape_i32.ptr,
            @intCast(shape_i32.len),
            mlx_dtype,
        );
        return Array.fromHandle(arr);
    }

    /// Check if a tensor exists in the index.
    pub fn contains(self: *TensorIndex, name: []const u8) bool {
        return self.entries.contains(name);
    }
};

/// Map safetensors dtype string to mlx-c dtype enum.
pub fn dtypeFromString(s: []const u8) ?c.c.mlx_dtype {
    if (std.mem.eql(u8, s, "F32")) return c.c.MLX_FLOAT32;
    if (std.mem.eql(u8, s, "F16")) return c.c.MLX_FLOAT16;
    if (std.mem.eql(u8, s, "BF16")) return c.c.MLX_BFLOAT16;
    if (std.mem.eql(u8, s, "I8")) return c.c.MLX_INT8;
    if (std.mem.eql(u8, s, "I16")) return c.c.MLX_INT16;
    if (std.mem.eql(u8, s, "I32")) return c.c.MLX_INT32;
    if (std.mem.eql(u8, s, "I64")) return c.c.MLX_INT64;
    if (std.mem.eql(u8, s, "U8")) return c.c.MLX_UINT8;
    if (std.mem.eql(u8, s, "U16")) return c.c.MLX_UINT16;
    if (std.mem.eql(u8, s, "U32")) return c.c.MLX_UINT32;
    if (std.mem.eql(u8, s, "U64")) return c.c.MLX_UINT64;
    if (std.mem.eql(u8, s, "BOOL")) return c.c.MLX_BOOL;
    return null;
}

/// Build a TensorIndex from a model directory with sharded safetensors.
/// Uses model.safetensors.index.json to find shard files.
pub fn buildIndexFromDirectory(allocator: std.mem.Allocator, dir_path: []const u8) !TensorIndex {
    var index = TensorIndex.init(allocator);
    errdefer index.deinit();

    const posix = @cImport({
        @cInclude("unistd.h");
        @cInclude("fcntl.h");
        @cInclude("sys/stat.h");
    });

    const index_path = try std.fs.path.join(allocator, &.{ dir_path, "model.safetensors.index.json" });
    defer allocator.free(index_path);
    const index_path_z = try allocator.dupeZ(u8, index_path);
    defer allocator.free(index_path_z);

    if (posix.access(index_path_z.ptr, posix.F_OK) != 0) {
        const single_path = try std.fs.path.join(allocator, &.{ dir_path, "model.safetensors" });
        defer allocator.free(single_path);
        try index.addShard(single_path);
        return index;
    }

    // Read index.json
    const fd = posix.open(index_path_z.ptr, posix.O_RDONLY);
    if (fd < 0) return error.FileNotFound;
    defer _ = posix.close(fd);

    var stat_buf: posix.struct_stat = undefined;
    if (posix.fstat(fd, &stat_buf) != 0) return error.StatFailed;
    const file_size: usize = @intCast(stat_buf.st_size);

    const content = try allocator.alloc(u8, file_size);
    defer allocator.free(content);
    _ = posix.read(fd, content.ptr, file_size);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();

    const weight_map = parsed.value.object.get("weight_map") orelse return error.InvalidIndex;

    // Collect unique shard files
    var shard_set = std.StringHashMap(void).init(allocator);
    defer {
        var it = shard_set.keyIterator();
        while (it.next()) |k| allocator.free(k.*);
        shard_set.deinit();
    }

    var wm_it = weight_map.object.iterator();
    while (wm_it.next()) |entry| {
        const shard = entry.value_ptr.*.string;
        if (!shard_set.contains(shard)) {
            try shard_set.put(try allocator.dupe(u8, shard), {});
        }
    }

    // Parse each shard header (only reads first few KB per file)
    var sit = shard_set.keyIterator();
    while (sit.next()) |shard_ptr| {
        const shard_file = shard_ptr.*;
        const shard_path = try std.fs.path.join(allocator, &.{ dir_path, shard_file });
        defer allocator.free(shard_path);
        std.log.info("Indexing shard: {s}", .{shard_file});
        try index.addShard(shard_path);
    }

    return index;
}


/// Lazy weight provider — loads tensors on-demand from a TensorIndex.
/// Provides the same `get(name)` interface as StringHashMap(Array) but
/// only loads a tensor from disk when first accessed.
/// Applies HF→internal name mapping transparently.
pub const LazyWeightProvider = struct {
    index: *TensorIndex,
    cache: std.StringHashMap(Array),
    allocator: std.mem.Allocator,
    name_mapper: ?*const fn (std.mem.Allocator, []const u8) anyerror!?[]const u8,
    /// Reverse map: internal name → HF name (for looking up in index)
    reverse_map: std.StringHashMap([]const u8),

    pub fn init(allocator: std.mem.Allocator, index: *TensorIndex, name_mapper: ?*const fn (std.mem.Allocator, []const u8) anyerror!?[]const u8) LazyWeightProvider {
        return .{
            .index = index,
            .cache = std.StringHashMap(Array).init(allocator),
            .allocator = allocator,
            .name_mapper = name_mapper,
            .reverse_map = std.StringHashMap([]const u8).init(allocator),
        };
    }

    /// Build reverse name mapping: for each HF name in the index, compute the
    /// internal name and store the mapping.
    pub fn buildReverseMap(self: *LazyWeightProvider) !void {
        var it = self.index.entries.iterator();
        while (it.next()) |entry| {
            const hf_name = entry.key_ptr.*;
            if (self.name_mapper) |mapper| {
                const mapped = mapper(self.allocator, hf_name) catch continue;
                if (mapped) |internal_name| {
                    try self.reverse_map.put(internal_name, try self.allocator.dupe(u8, hf_name));
                } else {
                    try self.reverse_map.put(try self.allocator.dupe(u8, hf_name), try self.allocator.dupe(u8, hf_name));
                }
            } else {
                try self.reverse_map.put(try self.allocator.dupe(u8, hf_name), try self.allocator.dupe(u8, hf_name));
            }
        }
    }

    /// Get a tensor by internal name. Loads from disk on first access.
    pub fn get(self: *LazyWeightProvider, internal_name: []const u8) ?Array {
        // Check cache first
        if (self.cache.get(internal_name)) |arr| return arr;

        // Find HF name via reverse map
        const hf_name = self.reverse_map.get(internal_name) orelse return null;

        // Load from disk
        const tensor = self.index.loadTensor(hf_name) catch return null;

        // Cache it
        const key = self.allocator.dupe(u8, internal_name) catch return null;
        self.cache.put(key, tensor) catch {
            tensor.deinit();
            self.allocator.free(key);
            return null;
        };
        return tensor;
    }

    /// Remove a key from the cache (equivalent to HashMap.fetchRemove for key cleanup).
    pub fn fetchRemove(self: *LazyWeightProvider, name: []const u8) ?struct { key: []const u8, value: Array } {
        if (self.cache.fetchRemove(name)) |kv| {
            return .{ .key = kv.key, .value = kv.value };
        }
        return null;
    }

    /// Check if a tensor exists (in index, not necessarily loaded).
    pub fn contains(self: *LazyWeightProvider, name: []const u8) bool {
        return self.cache.contains(name) or self.reverse_map.contains(name);
    }

    pub fn deinit(self: *LazyWeightProvider) void {
        var it = self.cache.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        self.cache.deinit();
        var rit = self.reverse_map.iterator();
        while (rit.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.reverse_map.deinit();
    }
};
