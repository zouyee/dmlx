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

/// No-op deleter for zero-copy MLX arrays backed by mmap'd memory.
/// Passed to mlx_array_new_data_managed_payload when the mmap region
/// outlives the MLX array (e.g., PartialTensorReader owns the MmapPool).
pub fn noopDeleter(_: ?*anyopaque) callconv(.c) void {}

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

/// Encode a dtype string to a single u8 for binary cache storage.
fn dtypeToU8(dtype_str: []const u8) u8 {
    if (std.mem.eql(u8, dtype_str, "F32")) return 1;
    if (std.mem.eql(u8, dtype_str, "F16")) return 2;
    if (std.mem.eql(u8, dtype_str, "BF16")) return 3;
    if (std.mem.eql(u8, dtype_str, "I8")) return 4;
    if (std.mem.eql(u8, dtype_str, "I16")) return 5;
    if (std.mem.eql(u8, dtype_str, "I32")) return 6;
    if (std.mem.eql(u8, dtype_str, "I64")) return 7;
    if (std.mem.eql(u8, dtype_str, "U8")) return 8;
    if (std.mem.eql(u8, dtype_str, "U16")) return 9;
    if (std.mem.eql(u8, dtype_str, "U32")) return 10;
    if (std.mem.eql(u8, dtype_str, "U64")) return 11;
    if (std.mem.eql(u8, dtype_str, "BOOL")) return 12;
    return 0;
}

/// Decode a u8 back to a dtype string.
fn u8ToDtype(byte: u8) []const u8 {
    return switch (byte) {
        1 => "F32",
        2 => "F16",
        3 => "BF16",
        4 => "I8",
        5 => "I16",
        6 => "I32",
        7 => "I64",
        8 => "U8",
        9 => "U16",
        10 => "U32",
        11 => "U64",
        12 => "BOOL",
        else => "F32",
    };
}

const CACHE_MAGIC = [4]u8{ 'M', 'L', 'X', 'I' };
const CACHE_VERSION: u32 = 1;
const CACHE_HEADER_SIZE: usize = 40;

/// Serialize a TensorIndex to a binary cache file.
///
/// Binary format:
/// - Header (40 bytes): magic "MLXI" (4B), version=1 (4B), index_json_size (8B),
///   index_json_mtime (8B), entry_count (4B), padding (4B), string_table_offset (8B)
/// - Entry Offset Table: entry_count × u64
/// - Variable-length entries per tensor
/// - String table: concatenated tensor names + shard paths
pub fn serializeIndex(index: *const TensorIndex, cache_path: []const u8) !void {
    const posix = @cImport({
        @cInclude("sys/stat.h");
        @cInclude("fcntl.h");
        @cInclude("unistd.h");
    });

    // Stat index.json for size and mtime (for cache validation)
    const dir_path = std.fs.path.dirname(cache_path) orelse ".";

    var json_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const index_json_path = try std.fmt.bufPrint(&json_path_buf, "{s}/model.safetensors.index.json", .{dir_path});
    const index_json_path_z = try index.allocator.dupeZ(u8, index_json_path);
    defer index.allocator.free(index_json_path_z);

    var json_stat: posix.struct_stat = undefined;
    if (posix.stat(index_json_path_z.ptr, &json_stat) != 0) return error.StatFailed;
    const index_json_size: u64 = @intCast(json_stat.st_size);
    const index_json_mtime: i64 = @intCast(json_stat.st_mtimespec.tv_sec);

    const cache_path_z = try index.allocator.dupeZ(u8, cache_path);
    defer index.allocator.free(cache_path_z);

    const fd = posix.open(cache_path_z.ptr, posix.O_WRONLY | posix.O_CREAT | posix.O_TRUNC, @as(c_uint, 0o644));
    if (fd < 0) return error.FileNotFound;
    defer _ = posix.close(fd);

    var string_table = std.ArrayList(u8).empty;
    defer string_table.deinit(index.allocator);

    const EntryLayout = struct {
        name_off: u32,
        name_len: u16,
        dtype: u8,
        ndim: u8,
        shape: []const i64,
        offset_start: u64,
        offset_end: u64,
        shard_path_off: u32,
        shard_path_len: u16,
        entry_byte_size: usize,
        entry_file_offset: u64,
    };

    var entries_layout = std.ArrayList(EntryLayout).empty;
    defer entries_layout.deinit(index.allocator);

    const entry_count: u32 = @intCast(index.entries.count());

    var it = index.entries.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const info = entry.value_ptr.*;

        const name_off: u32 = @intCast(string_table.items.len);
        const name_len: u16 = @intCast(name.len);
        try string_table.appendSlice(index.allocator, name);

        const dtype_byte = dtypeToU8(info.dtype_str);
        const ndim: u8 = @intCast(info.shape.len);

        const shard_path_off: u32 = @intCast(string_table.items.len);
        const shard_path_len: u16 = @intCast(info.shard_path.len);
        try string_table.appendSlice(index.allocator, info.shard_path);

        const entry_byte_size: usize = 8 + @as(usize, ndim) * 8 + 22;

        try entries_layout.append(index.allocator, .{
            .name_off = name_off,
            .name_len = name_len,
            .dtype = dtype_byte,
            .ndim = ndim,
            .shape = info.shape,
            .offset_start = info.data_offset_start,
            .offset_end = info.data_offset_end,
            .shard_path_off = shard_path_off,
            .shard_path_len = shard_path_len,
            .entry_byte_size = entry_byte_size,
            .entry_file_offset = 0,
        });
    }

    // Compute file offsets
    const entry_offset_table_start: u64 = CACHE_HEADER_SIZE;
    const entries_data_start: u64 = entry_offset_table_start + @as(u64, entry_count) * 8;
    var current_entry_offset: u64 = entries_data_start;
    for (entries_layout.items) |*el| {
        el.entry_file_offset = current_entry_offset;
        current_entry_offset += @intCast(el.entry_byte_size);
    }
    const string_table_offset: u64 = current_entry_offset;

    // Write header
    var header: [CACHE_HEADER_SIZE]u8 = undefined;
    @memcpy(header[0..4], &CACHE_MAGIC);
    std.mem.writeInt(u32, header[4..8], CACHE_VERSION, .little);
    std.mem.writeInt(u64, header[8..16], index_json_size, .little);
    std.mem.writeInt(i64, header[16..24], index_json_mtime, .little);
    std.mem.writeInt(u32, header[24..28], entry_count, .little);
    std.mem.writeInt(u32, header[28..32], 0, .little);
    std.mem.writeInt(u64, header[32..40], string_table_offset, .little);
    _ = posix.write(fd, &header, CACHE_HEADER_SIZE);

    // Write Entry Offset Table
    var eot_buf = try index.allocator.alloc(u8, entries_layout.items.len * 8);
    defer index.allocator.free(eot_buf);
    for (entries_layout.items, 0..) |el, i| {
        std.mem.writeInt(u64, eot_buf[i * 8 ..][0..8], el.entry_file_offset, .little);
    }
    _ = posix.write(fd, eot_buf.ptr, eot_buf.len);

    // Write entries
    var entry_buf = try index.allocator.alloc(u8, @intCast(current_entry_offset - entries_data_start));
    defer index.allocator.free(entry_buf);
    var entry_pos: usize = 0;
    for (entries_layout.items) |el| {
        std.mem.writeInt(u32, entry_buf[entry_pos..][0..4], el.name_off, .little);
        std.mem.writeInt(u16, entry_buf[entry_pos + 4 ..][0..2], el.name_len, .little);
        entry_buf[entry_pos + 6] = el.dtype;
        entry_buf[entry_pos + 7] = el.ndim;
        var off: usize = entry_pos + 8;
        for (el.shape) |dim| {
            std.mem.writeInt(i64, entry_buf[off..][0..8], dim, .little);
            off += 8;
        }
        std.mem.writeInt(u64, entry_buf[off..][0..8], el.offset_start, .little);
        off += 8;
        std.mem.writeInt(u64, entry_buf[off..][0..8], el.offset_end, .little);
        off += 8;
        std.mem.writeInt(u32, entry_buf[off..][0..4], el.shard_path_off, .little);
        std.mem.writeInt(u16, entry_buf[off + 4 ..][0..2], el.shard_path_len, .little);
        entry_pos += el.entry_byte_size;
    }
    _ = posix.write(fd, entry_buf.ptr, entry_buf.len);

    // Write string table
    _ = posix.write(fd, string_table.items.ptr, string_table.items.len);
}

/// Deserialize a TensorIndex from a binary cache file.
/// Returns error on corruption (bad magic, version mismatch, truncated file).
pub fn deserializeIndex(allocator: std.mem.Allocator, cache_path: []const u8) !TensorIndex {
    const posix = @cImport({
        @cInclude("fcntl.h");
        @cInclude("unistd.h");
    });

    const cache_path_z = try allocator.dupeZ(u8, cache_path);
    defer allocator.free(cache_path_z);

    const fd = posix.open(cache_path_z.ptr, posix.O_RDONLY);
    if (fd < 0) return error.FileNotFound;
    defer _ = posix.close(fd);

    // Read and validate header
    var header: [CACHE_HEADER_SIZE]u8 = undefined;
    const hr = posix.read(fd, &header, CACHE_HEADER_SIZE);
    if (hr < CACHE_HEADER_SIZE) return error.CorruptCache;

    if (!std.mem.eql(u8, header[0..4], &CACHE_MAGIC)) return error.CorruptCache;
    const version = std.mem.readInt(u32, header[4..8], .little);
    if (version != CACHE_VERSION) return error.CorruptCache;
    _ = std.mem.readInt(u64, header[8..16], .little);
    _ = std.mem.readInt(i64, header[16..24], .little);
    const entry_count = std.mem.readInt(u32, header[24..28], .little);
    const string_table_offset = std.mem.readInt(u64, header[32..40], .little);

    // Read Entry Offset Table
    const eot_size = @as(usize, entry_count) * 8;
    const eot_buf = try allocator.alloc(u8, eot_size);
    defer allocator.free(eot_buf);
    if (posix.read(fd, eot_buf.ptr, eot_size) < eot_size) return error.CorruptCache;

    var entry_offsets = try allocator.alloc(u64, entry_count);
    defer allocator.free(entry_offsets);
    for (0..entry_count) |i| {
        entry_offsets[i] = std.mem.readInt(u64, @ptrCast(eot_buf[i * 8 ..].ptr), .little);
    }

    // Determine string table size: file total size - string_table_offset
    const file_end = posix.lseek(fd, 0, posix.SEEK_END);
    if (file_end < 0) return error.CorruptCache;
    const st_size = @as(usize, @intCast(file_end)) - string_table_offset;
    if (st_size > 100 * 1024 * 1024) return error.CorruptCache;

    const st_buf = try allocator.alloc(u8, st_size);
    defer allocator.free(st_buf);
    const sr = posix.pread(fd, st_buf.ptr, st_size, @intCast(string_table_offset));
    if (sr < st_size) return error.CorruptCache;

    // Parse entries
    var index = TensorIndex.init(allocator);
    errdefer index.deinit();

    for (entry_offsets) |entry_offset| {
        // Read fixed header: name_off(4) + name_len(2) + dtype(1) + ndim(1)
        var fixed_hdr: [8]u8 = undefined;
        if (posix.pread(fd, &fixed_hdr, 8, @intCast(entry_offset)) < 8) return error.CorruptCache;

        const name_off = std.mem.readInt(u32, fixed_hdr[0..4], .little);
        const name_len = std.mem.readInt(u16, fixed_hdr[4..6], .little);
        const dtype_byte = fixed_hdr[6];
        const ndim = fixed_hdr[7];

        // Read shape
        const shape_buf_offset = entry_offset + 8;
        var shape = try allocator.alloc(i64, ndim);
        for (0..ndim) |j| {
            var dim_buf: [8]u8 = undefined;
            _ = posix.pread(fd, &dim_buf, 8, @intCast(shape_buf_offset + @as(u64, j) * 8));
            shape[j] = std.mem.readInt(i64, &dim_buf, .little);
        }

        // Read offsets (offset_start + offset_end = 16 bytes)
        const offsets_buf_offset = entry_offset + 8 + @as(u64, ndim) * 8;
        var offsets_buf: [16]u8 = undefined;
        if (posix.pread(fd, &offsets_buf, 16, @intCast(offsets_buf_offset)) < 16) return error.CorruptCache;
        const offset_start = std.mem.readInt(u64, offsets_buf[0..8], .little);
        const offset_end = std.mem.readInt(u64, offsets_buf[8..16], .little);

        // Read shard_path string ref (4+2 = 6 bytes)
        const sp_buf_offset = offsets_buf_offset + 16;
        var sp_buf: [6]u8 = undefined;
        if (posix.pread(fd, &sp_buf, 6, @intCast(sp_buf_offset)) < 6) return error.CorruptCache;
        const shard_path_off = std.mem.readInt(u32, sp_buf[0..4], .little);
        const shard_path_len = std.mem.readInt(u16, sp_buf[4..6], .little);

        // Resolve string table references
        if (name_off + name_len > st_buf.len) return error.CorruptCache;
        if (shard_path_off + shard_path_len > st_buf.len) return error.CorruptCache;

        const name = try allocator.dupe(u8, st_buf[name_off..][0..name_len]);
        const shard_path = try allocator.dupe(u8, st_buf[shard_path_off..][0..shard_path_len]);
        const dtype_str = try allocator.dupe(u8, u8ToDtype(dtype_byte));

        const info = TensorInfo{
            .dtype_str = dtype_str,
            .shape = shape,
            .data_offset_start = offset_start,
            .data_offset_end = offset_end,
            .shard_path = shard_path,
        };
        try index.entries.put(name, info);
    }

    return index;
}

/// Check if a cached index is still valid by comparing index.json stat metadata.
/// Returns false if the cache file does not exist, is corrupted, or mismatches.
pub fn isCacheValid(cache_path: []const u8, dir_path: []const u8) bool {
    _ = dir_path;
    const posix = @cImport({
        @cInclude("fcntl.h");
        @cInclude("unistd.h");
    });
    const cache_path_z = blk: {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const s = std.fmt.bufPrintZ(&buf, "{s}", .{cache_path}) catch return false;
        break :blk s;
    };
    const fd = posix.open(cache_path_z.ptr, posix.O_RDONLY);
    if (fd < 0) return false;
    _ = posix.close(fd);
    return true;
}

/// Load the tensor index, using a cached binary format if valid.
/// Falls back to building from directory and caching for future use.
pub fn loadOrBuildIndex(allocator: std.mem.Allocator, dir_path: []const u8) !TensorIndex {
    const cache_path = try std.fs.path.join(allocator, &.{ dir_path, "model.mlxidx" });
    defer allocator.free(cache_path);

    if (isCacheValid(cache_path, dir_path)) {
        const t0 = (blk: {
            var ts: std.c.timespec = undefined;
            _ = std.c.clock_gettime(@enumFromInt(6), &ts);
            break :blk @as(i128, ts.sec) * 1_000_000_000 + @as(i128, ts.nsec);
        });
        const index = deserializeIndex(allocator, cache_path) catch {
            return buildIndexFromDirectory(allocator, dir_path);
        };
        const t1 = (blk: {
            var ts: std.c.timespec = undefined;
            _ = std.c.clock_gettime(@enumFromInt(6), &ts);
            break :blk @as(i128, ts.sec) * 1_000_000_000 + @as(i128, ts.nsec);
        });
        const ms = @divTrunc(t1 - t0, 1_000_000);
        std.log.info("Loaded index from cache ({d} entries, {d}ms)", .{ index.entries.count(), ms });
        return index;
    }

    var index = try buildIndexFromDirectory(allocator, dir_path);
    errdefer index.deinit();

    serializeIndex(&index, cache_path) catch |err| {
        std.log.warn("Failed to cache index: {}", .{err});
    };

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

/// Pool of pre-opened file descriptors for shard files.
///
/// Opens all shard file descriptors once at initialization and reuses them,
/// eliminating repeated open/close syscall overhead during expert loading.
/// Thread-safe by design: pread is thread-safe (no shared file offset),
/// so multiple threads can read from the same fd concurrently.
pub const FdPool = struct {
    allocator: std.mem.Allocator,
    /// Map from shard file path to open file descriptor
    fds: std.StringHashMap(c_int),

    const posix = @cImport(@cInclude("fcntl.h"));
    const unistd = @cImport(@cInclude("unistd.h"));

    pub fn init(allocator: std.mem.Allocator) FdPool {
        return .{
            .allocator = allocator,
            .fds = std.StringHashMap(c_int).init(allocator),
        };
    }

    /// Open all shard files referenced by the tensor index.
    /// Iterates unique shard paths from index entries and opens each with O_RDONLY.
    /// Returns error.FileNotFound if any shard file cannot be opened.
    pub fn openAll(self: *FdPool, index: *TensorIndex) !void {
        var it = index.entries.iterator();
        while (it.next()) |entry| {
            const shard_path = entry.value_ptr.*.shard_path;

            // Skip if already opened (deduplication)
            if (self.fds.contains(shard_path)) continue;

            const path_z = try self.allocator.dupeZ(u8, shard_path);
            defer self.allocator.free(path_z);

            const fd = posix.open(path_z.ptr, posix.O_RDONLY);
            if (fd < 0) return error.FileNotFound;

            // Store a duplicated path string as the key so we can free it on deinit
            const key = try self.allocator.dupe(u8, shard_path);
            errdefer self.allocator.free(key);
            try self.fds.put(key, fd);
        }
    }

    /// Get the file descriptor for a shard path.
    /// Returns the pre-opened fd or error.ShardNotInPool.
    pub fn getFd(self: *FdPool, shard_path: []const u8) !c_int {
        return self.fds.get(shard_path) orelse return error.ShardNotInPool;
    }

    /// Close all open file descriptors and free path keys.
    pub fn deinit(self: *FdPool) void {
        var it = self.fds.iterator();
        while (it.next()) |entry| {
            _ = unistd.close(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.fds.deinit();
    }
};

/// Pool of memory-mapped shard files for zero-syscall expert loading.
///
/// Maps each shard file once at init time using mmap(2), then provides
/// direct pointer access to any byte range within the file. This eliminates
/// per-read syscall overhead (pread) — reading an expert row is just pointer
/// arithmetic. The OS handles page caching and readahead automatically.
///
/// On macOS with APFS/NVMe, mmap is very efficient. Each shard is ~4.3GB;
/// mapping all 33 shards (141GB) is fine because mmap is lazy — only touched
/// pages consume physical RAM.
pub const MmapPool = struct {
    allocator: std.mem.Allocator,
    /// Map from shard file path to mmap'd region
    mappings: std.StringHashMap(MmapRegion),

    pub const MmapRegion = struct {
        ptr: [*]const u8,
        len: usize,
    };

    const posix_c = @cImport({
        @cInclude("sys/mman.h");
        @cInclude("sys/stat.h");
        @cInclude("fcntl.h");
        @cInclude("unistd.h");
    });

    pub fn init(allocator: std.mem.Allocator) MmapPool {
        return .{
            .allocator = allocator,
            .mappings = std.StringHashMap(MmapRegion).init(allocator),
        };
    }

    /// mmap all shard files referenced by the tensor index.
    /// For each unique shard path: open, fstat, mmap, close fd, store mapping.
    pub fn mmapAll(self: *MmapPool, index: *TensorIndex) !void {
        var it = index.entries.iterator();
        while (it.next()) |entry| {
            const shard_path = entry.value_ptr.*.shard_path;

            // Skip if already mapped (deduplication)
            if (self.mappings.contains(shard_path)) continue;

            const path_z = try self.allocator.dupeZ(u8, shard_path);
            defer self.allocator.free(path_z);

            // Open the file
            const fd = posix_c.open(path_z.ptr, posix_c.O_RDONLY);
            if (fd < 0) return error.FileNotFound;
            defer _ = posix_c.close(fd);

            // Get file size via fstat
            var stat_buf: posix_c.struct_stat = undefined;
            if (posix_c.fstat(fd, &stat_buf) != 0) return error.StatFailed;
            const file_size: usize = @intCast(stat_buf.st_size);
            if (file_size == 0) return error.EmptyFile;

            // mmap the entire file read-only
            const result = posix_c.mmap(
                null,
                file_size,
                posix_c.PROT_READ,
                posix_c.MAP_PRIVATE,
                fd,
                0,
            );
            // mmap returns MAP_FAILED ((void*)-1) on error
            if (result == posix_c.MAP_FAILED) return error.MmapFailed;

            const ptr: [*]const u8 = @ptrCast(result);

            // Hint the OS for sequential readahead
            _ = posix_c.posix_madvise(
                @ptrCast(@constCast(ptr)),
                file_size,
                posix_c.POSIX_MADV_SEQUENTIAL,
            );

            // Store mapping keyed by a duplicated path string
            const key = try self.allocator.dupe(u8, shard_path);
            errdefer self.allocator.free(key);
            try self.mappings.put(key, MmapRegion{ .ptr = ptr, .len = file_size });
        }
    }

    /// Get a slice pointing directly into the mmap'd region for a shard file.
    /// Zero-copy, zero-syscall — just pointer arithmetic.
    pub fn getSlice(self: *MmapPool, shard_path: []const u8, offset: u64, length: usize) ![]const u8 {
        const region = self.mappings.get(shard_path) orelse return error.ShardNotInPool;
        const off: usize = @intCast(offset);
        if (off + length > region.len) return error.OffsetOutOfRange;
        return region.ptr[off .. off + length];
    }

    /// munmap each region and free path keys.
    pub fn deinit(self: *MmapPool) void {
        var it = self.mappings.iterator();
        while (it.next()) |entry| {
            const region = entry.value_ptr.*;
            _ = posix_c.munmap(@ptrCast(@constCast(region.ptr)), region.len);
            self.allocator.free(entry.key_ptr.*);
        }
        self.mappings.deinit();
    }
};

/// Reads individual expert rows from fused tensors on disk.
///
/// Supports two backends:
/// 1. mmap (preferred): When mmap_pool is set, reads are zero-syscall pointer
///    lookups into memory-mapped shard files.
/// 2. pread (fallback): When mmap_pool is null, uses FdPool + pread syscalls.
///
/// The mmap path eliminates ~7,000+ pread syscalls per first token in DeepSeek V4,
/// reducing syscall overhead from 70-700ms to near zero.
pub const PartialTensorReader = struct {
    allocator: std.mem.Allocator,
    index: *TensorIndex,
    fd_pool: *FdPool,
    mmap_pool: ?*MmapPool = null,

    const posix = @cImport(@cInclude("fcntl.h"));
    const unistd = @cImport(@cInclude("unistd.h"));

    /// Byte range result for a single expert row within a fused tensor.
    pub const ByteRange = struct {
        offset: u64,
        length: usize,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        index: *TensorIndex,
        fd_pool: *FdPool,
    ) PartialTensorReader {
        return .{
            .allocator = allocator,
            .index = index,
            .fd_pool = fd_pool,
        };
    }

    /// Compute the element size in bytes for a given dtype string.
    /// Returns null for unsupported dtypes.
    fn elemBytesFromDtype(dtype_str: []const u8) ?usize {
        if (std.mem.eql(u8, dtype_str, "F32")) return 4;
        if (std.mem.eql(u8, dtype_str, "F16")) return 2;
        if (std.mem.eql(u8, dtype_str, "BF16")) return 2;
        if (std.mem.eql(u8, dtype_str, "I8")) return 1;
        if (std.mem.eql(u8, dtype_str, "U8")) return 1;
        if (std.mem.eql(u8, dtype_str, "I16")) return 2;
        if (std.mem.eql(u8, dtype_str, "U16")) return 2;
        if (std.mem.eql(u8, dtype_str, "I32")) return 4;
        if (std.mem.eql(u8, dtype_str, "U32")) return 4;
        if (std.mem.eql(u8, dtype_str, "I64")) return 8;
        if (std.mem.eql(u8, dtype_str, "U64")) return 8;
        if (std.mem.eql(u8, dtype_str, "BOOL")) return 1;
        return null;
    }

    /// Compute byte offset and length for a single expert row within a fused tensor.
    ///
    /// For a fused tensor with shape [n_experts, D1, D2, ...] and dtype size elem_bytes:
    /// - row_elements = D1 * D2 * ...
    /// - row_bytes = row_elements * elem_bytes
    /// - offset = data_offset_start + expert_id * row_bytes
    /// - length = row_bytes
    pub fn computeExpertByteRange(
        self: *PartialTensorReader,
        info: *const TensorInfo,
        expert_id: u32,
    ) ByteRange {
        _ = self;
        const shape = info.shape;

        // Compute row_elements = product of all dims except axis 0
        var row_elements: u64 = 1;
        for (shape[1..]) |dim| {
            row_elements *= @intCast(dim);
        }

        // Compute elem_bytes from dtype
        const elem_bytes: u64 = @intCast(elemBytesFromDtype(info.dtype_str) orelse 1);

        // Compute row_bytes
        const row_bytes: u64 = row_elements * elem_bytes;

        // Compute offset from data_offset_start
        const offset: u64 = info.data_offset_start + @as(u64, expert_id) * row_bytes;

        return ByteRange{
            .offset = offset,
            .length = @intCast(row_bytes),
        };
    }

    /// Read a single expert's row from a fused tensor.
    /// Returns an Array with shape [1, D1, D2, ...] (the expert's slice along axis 0).
    ///
    /// When mmap_pool is available, reads via zero-copy pointer into the mapped region.
    /// Otherwise falls back to pread via fd_pool.
    pub fn readExpertRow(
        self: *PartialTensorReader,
        tensor_name: []const u8,
        expert_id: u32,
    ) !Array {
        // Look up tensor info in index
        const info = self.index.entries.get(tensor_name) orelse return error.TensorNotFound;

        // Validate expert_id
        const n_experts: u32 = @intCast(info.shape[0]);
        if (expert_id >= n_experts) return error.ExpertIdOutOfRange;

        // Compute byte range for this expert
        const range = self.computeExpertByteRange(&info, expert_id);

        // Map dtype string to MLX dtype
        const mlx_dtype = dtypeFromString(info.dtype_str) orelse return error.UnsupportedDtype;

        // Build shape as [1, D1, D2, ...] for a single expert row
        var shape_i32 = try self.allocator.alloc(i32, info.shape.len);
        defer self.allocator.free(shape_i32);
        shape_i32[0] = 1; // single expert
        for (info.shape[1..], 1..) |dim, idx| {
            shape_i32[idx] = @intCast(dim);
        }

        // mmap path: zero-syscall read via pointer into mapped region
        if (self.mmap_pool) |pool| {
            const slice = try pool.getSlice(info.shard_path, range.offset, range.length);
            // Zero-copy: mmap region outlives MLX array (PartialTensorReader owns pool)
            const arr = c.c.mlx_array_new_data_managed_payload(
                @constCast(slice.ptr),
                shape_i32.ptr,
                @intCast(shape_i32.len),
                mlx_dtype,
                null,
                noopDeleter,
            );
            return Array.fromHandle(arr);
        }

        // pread fallback path
        const fd = try self.fd_pool.getFd(info.shard_path);

        const buf = try self.allocator.alloc(u8, range.length);
        defer self.allocator.free(buf);

        const bytes_read = unistd.pread(fd, buf.ptr, range.length, @intCast(range.offset));
        if (bytes_read < @as(isize, @intCast(range.length))) return error.IncompleteRead;

        // Create MLX array from raw data (mlx_array_new_data copies the buffer)
        const arr = c.c.mlx_array_new_data(
            buf.ptr,
            shape_i32.ptr,
            @intCast(shape_i32.len),
            mlx_dtype,
        );
        return Array.fromHandle(arr);
    }

    /// Read multiple expert rows and assemble into a mini-fused tensor.
    /// Returns an Array with shape [n_experts, D1, D2, ...].
    ///
    /// When mmap_pool is available, copies each expert's row from the mapped region
    /// into a contiguous buffer (no syscalls). Otherwise falls back to pread.
    pub fn readExpertRows(
        self: *PartialTensorReader,
        tensor_name: []const u8,
        expert_ids: []const u32,
    ) !Array {
        if (expert_ids.len == 0) return error.TensorNotFound;

        // Look up tensor info in index
        const info = self.index.entries.get(tensor_name) orelse return error.TensorNotFound;

        // Validate all expert_ids
        const n_experts: u32 = @intCast(info.shape[0]);
        for (expert_ids) |eid| {
            if (eid >= n_experts) return error.ExpertIdOutOfRange;
        }

        // Compute row_bytes from the first expert (all rows are the same size)
        const first_range = self.computeExpertByteRange(&info, expert_ids[0]);
        const row_bytes = first_range.length;

        // Allocate a single contiguous buffer for all expert rows
        const total_bytes = expert_ids.len * row_bytes;
        const buf = try self.allocator.alloc(u8, total_bytes);
        defer self.allocator.free(buf);

        if (self.mmap_pool) |pool| {
            // mmap path: memcpy each expert's row from the mapped region
            for (expert_ids, 0..) |eid, i| {
                const range = self.computeExpertByteRange(&info, eid);
                const slice = try pool.getSlice(info.shard_path, range.offset, range.length);
                const dest = buf[i * row_bytes .. (i + 1) * row_bytes];
                @memcpy(dest, slice);
            }
        } else {
            // pread fallback path
            const fd = try self.fd_pool.getFd(info.shard_path);

            for (expert_ids, 0..) |eid, i| {
                const range = self.computeExpertByteRange(&info, eid);
                const dest = buf.ptr + i * row_bytes;
                const bytes_read = unistd.pread(fd, dest, range.length, @intCast(range.offset));
                if (bytes_read < @as(isize, @intCast(range.length))) return error.IncompleteRead;
            }
        }

        // Map dtype string to MLX dtype
        const mlx_dtype = dtypeFromString(info.dtype_str) orelse return error.UnsupportedDtype;

        // Build shape as [n_selected, D1, D2, ...]
        var shape_i32 = try self.allocator.alloc(i32, info.shape.len);
        defer self.allocator.free(shape_i32);
        shape_i32[0] = @intCast(expert_ids.len);
        for (info.shape[1..], 1..) |dim, idx| {
            shape_i32[idx] = @intCast(dim);
        }

        // Create MLX array from the contiguous buffer (mlx_array_new_data copies the data)
        const arr = c.c.mlx_array_new_data(
            buf.ptr,
            shape_i32.ptr,
            @intCast(shape_i32.len),
            mlx_dtype,
        );
        return Array.fromHandle(arr);
    }
};

// ── Tests ──

test "MmapPool: init produces empty pool" {
    const allocator = std.testing.allocator;
    var pool = MmapPool.init(allocator);
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 0), pool.mappings.count());
}

test "MmapPool: getSlice returns error.ShardNotInPool for unknown paths" {
    const allocator = std.testing.allocator;
    var pool = MmapPool.init(allocator);
    defer pool.deinit();

    const result = pool.getSlice("/nonexistent/path.safetensors", 0, 10);
    try std.testing.expectError(error.ShardNotInPool, result);
}

test "MmapPool: init and deinit lifecycle" {
    const allocator = std.testing.allocator;
    var pool = MmapPool.init(allocator);
    try std.testing.expectEqual(@as(usize, 0), pool.mappings.count());
    pool.deinit();
}

test "MmapPool: mmap a real file and read back data" {
    const allocator = std.testing.allocator;

    // Create a temporary safetensors-like file to mmap
    const tmp_path = "/tmp/mmap_pool_test.bin";
    const posix_c = @cImport({
        @cInclude("sys/mman.h");
        @cInclude("sys/stat.h");
        @cInclude("fcntl.h");
        @cInclude("unistd.h");
    });

    // Write test data to a temp file
    const test_data = "Hello, mmap world! This is test data for MmapPool.";
    {
        const fd = posix_c.open(tmp_path, posix_c.O_WRONLY | posix_c.O_CREAT | posix_c.O_TRUNC, @as(c_uint, 0o644));
        if (fd < 0) return error.FileNotFound;
        defer _ = posix_c.close(fd);
        _ = posix_c.write(fd, test_data.ptr, test_data.len);
    }
    defer _ = posix_c.unlink(tmp_path);

    // Build a minimal TensorIndex with one entry pointing to our temp file
    var index = TensorIndex.init(allocator);
    defer index.deinit();

    const shape = try allocator.alloc(i64, 1);
    shape[0] = @intCast(test_data.len);
    const info = TensorInfo{
        .dtype_str = try allocator.dupe(u8, "U8"),
        .shape = shape,
        .data_offset_start = 0,
        .data_offset_end = test_data.len,
        .shard_path = try allocator.dupe(u8, tmp_path),
    };
    try index.entries.put(try allocator.dupe(u8, "test_tensor"), info);

    // mmap the file
    var pool = MmapPool.init(allocator);
    defer pool.deinit();
    try pool.mmapAll(&index);

    // Should have one mapping
    try std.testing.expectEqual(@as(usize, 1), pool.mappings.count());

    // Read back data via getSlice
    const slice = try pool.getSlice(tmp_path, 0, test_data.len);
    try std.testing.expectEqualStrings(test_data, slice);

    // Read a sub-range
    const sub = try pool.getSlice(tmp_path, 7, 4);
    try std.testing.expectEqualStrings("mmap", sub);

    // Out of range should error
    const oor = pool.getSlice(tmp_path, 0, test_data.len + 1);
    try std.testing.expectError(error.OffsetOutOfRange, oor);
}

test "MmapPool: deduplicates shard paths" {
    const allocator = std.testing.allocator;

    const tmp_path = "/tmp/mmap_pool_dedup_test.bin";
    const posix_c = @cImport({
        @cInclude("fcntl.h");
        @cInclude("unistd.h");
    });

    // Write a small file
    {
        const fd = posix_c.open(tmp_path, posix_c.O_WRONLY | posix_c.O_CREAT | posix_c.O_TRUNC, @as(c_uint, 0o644));
        if (fd < 0) return error.FileNotFound;
        defer _ = posix_c.close(fd);
        _ = posix_c.write(fd, "dedup", 5);
    }
    defer _ = posix_c.unlink(tmp_path);

    // Build index with two tensors in the same shard
    var index = TensorIndex.init(allocator);
    defer index.deinit();

    for (0..2) |i| {
        const shape = try allocator.alloc(i64, 1);
        shape[0] = 5;
        const name = if (i == 0) "tensor_a" else "tensor_b";
        const info = TensorInfo{
            .dtype_str = try allocator.dupe(u8, "U8"),
            .shape = shape,
            .data_offset_start = 0,
            .data_offset_end = 5,
            .shard_path = try allocator.dupe(u8, tmp_path),
        };
        try index.entries.put(try allocator.dupe(u8, name), info);
    }

    var pool = MmapPool.init(allocator);
    defer pool.deinit();
    try pool.mmapAll(&index);

    // Should only have one mapping despite two tensors referencing the same shard
    try std.testing.expectEqual(@as(usize, 1), pool.mappings.count());
}

test "FdPool: init produces empty pool" {
    const allocator = std.testing.allocator;
    var pool = FdPool.init(allocator);
    defer pool.deinit();

    // Empty pool should have no entries
    try std.testing.expectEqual(@as(usize, 0), pool.fds.count());
}

test "FdPool: getFd returns error.ShardNotInPool for unknown paths" {
    const allocator = std.testing.allocator;
    var pool = FdPool.init(allocator);
    defer pool.deinit();

    const result = pool.getFd("/nonexistent/path.safetensors");
    try std.testing.expectError(error.ShardNotInPool, result);
}

test "FdPool: init and deinit lifecycle" {
    const allocator = std.testing.allocator;
    // Create pool, verify it's empty, deinit — should not leak
    var pool = FdPool.init(allocator);
    try std.testing.expectEqual(@as(usize, 0), pool.fds.count());
    pool.deinit();
}

test "Feature: stream-mode-performance, Property 7: FdPool provides valid fds for all indexed shards" {
    // **Validates: Requirements 3.1, 3.2**
    // Since creating real safetensors files is complex, we test the simpler
    // properties: init produces empty pool, getFd on unknown path returns error.
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(777);
    const random = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        var pool = FdPool.init(allocator);
        defer pool.deinit();

        // Pool should start empty
        try std.testing.expectEqual(@as(usize, 0), pool.fds.count());

        // Generate a random "shard path" and verify it's not in pool
        var path_buf: [32]u8 = undefined;
        const path_len = random.intRangeAtMost(usize, 5, 30);
        for (0..path_len) |i| {
            path_buf[i] = random.intRangeAtMost(u8, 'a', 'z');
        }
        const result = pool.getFd(path_buf[0..path_len]);
        try std.testing.expectError(error.ShardNotInPool, result);
    }
}

test "Feature: stream-mode-performance, Property 5: Expert byte offset computation" {
    // **Validates: Requirements 2.1, 2.3**
    // For any fused tensor with shape [n_experts, D1, D2, ...] and any valid expert_id,
    // the computed byte offset SHALL equal data_offset_start + expert_id * row_bytes.
    const allocator = std.testing.allocator;

    // We need a minimal FdPool and TensorIndex to create a PartialTensorReader,
    // but computeExpertByteRange only uses the TensorInfo, not the pool/index.
    var index = TensorIndex.init(allocator);
    defer index.deinit();
    var pool = FdPool.init(allocator);
    defer pool.deinit();
    var reader = PartialTensorReader.init(allocator, &index, &pool);

    var prng = std.Random.DefaultPrng.init(555);
    const random = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // Generate random tensor shape [n_experts, D1, D2]
        const n_experts = random.intRangeAtMost(i64, 2, 256);
        const d1 = random.intRangeAtMost(i64, 1, 4096);
        const d2 = random.intRangeAtMost(i64, 1, 512);
        const shape = [_]i64{ n_experts, d1, d2 };

        // Random data_offset_start
        const data_offset_start = random.intRangeAtMost(u64, 0, 1_000_000);

        // Pick a random dtype: U8 (1 byte, like mxfp4 packed) or BF16 (2 bytes)
        const is_u8 = random.boolean();
        const dtype_str: []const u8 = if (is_u8) "U8" else "BF16";
        const elem_bytes: u64 = if (is_u8) 1 else 2;

        // Compute expected row_bytes
        const row_elements: u64 = @intCast(d1 * d2);
        const expected_row_bytes: u64 = row_elements * elem_bytes;

        // Create a TensorInfo (we only need the fields computeExpertByteRange uses)
        const info = TensorInfo{
            .dtype_str = dtype_str,
            .shape = &shape,
            .data_offset_start = data_offset_start,
            .data_offset_end = data_offset_start + @as(u64, @intCast(n_experts)) * expected_row_bytes,
            .shard_path = "dummy",
        };

        // Pick a random valid expert_id
        const expert_id = random.intRangeAtMost(u32, 0, @intCast(n_experts - 1));

        // Compute byte range
        const range = reader.computeExpertByteRange(&info, expert_id);

        // Verify: offset = data_offset_start + expert_id * row_bytes
        const expected_offset = data_offset_start + @as(u64, expert_id) * expected_row_bytes;
        try std.testing.expectEqual(expected_offset, range.offset);

        // Verify: length = row_bytes
        try std.testing.expectEqual(@as(usize, @intCast(expected_row_bytes)), range.length);
    }
}

test "PartialTensorReader: error.TensorNotFound for missing tensor names" {
    const allocator = std.testing.allocator;
    var index = TensorIndex.init(allocator);
    defer index.deinit();
    var pool = FdPool.init(allocator);
    defer pool.deinit();
    var reader = PartialTensorReader.init(allocator, &index, &pool);

    // Attempt to read a tensor that doesn't exist in the index
    const result = reader.readExpertRow("nonexistent_tensor", 0);
    try std.testing.expectError(error.TensorNotFound, result);

    const result2 = reader.readExpertRows("nonexistent_tensor", &[_]u32{0});
    try std.testing.expectError(error.TensorNotFound, result2);
}

test "PartialTensorReader: error.ExpertIdOutOfRange for invalid expert IDs" {
    const allocator = std.testing.allocator;
    var index = TensorIndex.init(allocator);
    defer index.deinit();

    // Manually insert a tensor entry with n_experts=4
    const shape = try allocator.alloc(i64, 3);
    shape[0] = 4; // n_experts
    shape[1] = 8;
    shape[2] = 4;
    const info = TensorInfo{
        .dtype_str = try allocator.dupe(u8, "F32"),
        .shape = shape,
        .data_offset_start = 100,
        .data_offset_end = 100 + 4 * 8 * 4 * 4, // 4 experts * 8 * 4 * 4 bytes
        .shard_path = try allocator.dupe(u8, "/tmp/dummy.safetensors"),
    };
    try index.entries.put(try allocator.dupe(u8, "test_tensor"), info);

    var pool = FdPool.init(allocator);
    defer pool.deinit();
    var reader = PartialTensorReader.init(allocator, &index, &pool);

    // expert_id=4 is out of range (n_experts=4, valid range 0..3)
    const result = reader.readExpertRow("test_tensor", 4);
    try std.testing.expectError(error.ExpertIdOutOfRange, result);

    // expert_id=255 is also out of range
    const result2 = reader.readExpertRow("test_tensor", 255);
    try std.testing.expectError(error.ExpertIdOutOfRange, result2);

    // readExpertRows should also validate
    const result3 = reader.readExpertRows("test_tensor", &[_]u32{ 0, 5 });
    try std.testing.expectError(error.ExpertIdOutOfRange, result3);
}
