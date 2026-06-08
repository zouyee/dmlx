/// MLX-free safetensors reader.
///
/// Parses safetensors shards and provides raw tensor loading via pread.
/// No MLX or external dependencies — only std and POSIX pread.
///
/// Format: [8-byte LE u64 header_len] [JSON header] [tensor data...]
/// Header JSON: { "name": { "dtype": "BF16", "shape": [M, N], "data_offsets": [start, end] }, ... }
const std = @import("std");

pub const DType = enum {
    F32,
    BF16,
    U32,
    U8,
    I64,
    U16,
    I32,
    F16,

    pub fn fromStr(s: []const u8) ?DType {
        if (std.mem.eql(u8, s, "F32")) return .F32;
        if (std.mem.eql(u8, s, "BF16")) return .BF16;
        if (std.mem.eql(u8, s, "U32")) return .U32;
        if (std.mem.eql(u8, s, "U8")) return .U8;
        if (std.mem.eql(u8, s, "I64")) return .I64;
        if (std.mem.eql(u8, s, "U16")) return .U16;
        if (std.mem.eql(u8, s, "I32")) return .I32;
        if (std.mem.eql(u8, s, "F16")) return .F16;
        return null;
    }

    pub fn byteSize(self: DType) u8 {
        return switch (self) {
            .U8 => 1,
            .F16, .BF16, .U16 => 2,
            .F32, .U32, .I32 => 4,
            .I64 => 8,
        };
    }
};

pub const TensorInfo = struct {
    dtype: DType,
    shape: []u64,
    shard_path: []u8,
    /// Absolute file offset (from start of file, including 8-byte header prefix)
    data_offset: u64,
    data_len: u64,
    /// Number of elements (product of shape)
    n_elems: u64,
};

/// Index mapping tensor names to their location across all shards.
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
            self.allocator.free(entry.value_ptr.*.shape);
            self.allocator.free(entry.value_ptr.*.shard_path);
        }
        self.entries.deinit();
    }

    /// Parse one safetensors shard header and add tensors to index.
    pub fn addShard(self: *TensorIndex, shard_path: []const u8) !void {
        const shard_path_z = try self.allocator.dupeZ(u8, shard_path);
        defer self.allocator.free(shard_path_z);

        const fd = std.c.open(shard_path_z.ptr, .{});
        if (fd < 0) return error.FileNotFound;
        defer _ = std.c.close(fd);

        // Read 8-byte little-endian header length
        var header_len_buf: [8]u8 = undefined;
        const r1 = std.c.pread(fd, &header_len_buf, 8, 0);
        if (r1 < 8) return error.InvalidSafetensors;
        const header_len = std.mem.readInt(u64, &header_len_buf, .little);
        if (header_len > 256 * 1024 * 1024) return error.HeaderTooLarge;

        // Read header JSON
        const header_buf = try self.allocator.alloc(u8, @intCast(header_len));
        defer self.allocator.free(header_buf);
        const r2 = std.c.pread(fd, header_buf.ptr, @intCast(header_len), 8);
        if (r2 < @as(isize, @intCast(header_len))) return error.InvalidSafetensors;

        const data_base: u64 = 8 + header_len;

        // Parse JSON
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, header_buf, .{});
        defer parsed.deinit();

        const obj = parsed.value.object;
        var it = obj.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (std.mem.eql(u8, name, "__metadata__")) continue;

            const info_obj = entry.value_ptr.*.object;

            // dtype
            const dtype_str = info_obj.get("dtype") orelse continue;
            const dtype = DType.fromStr(dtype_str.string) orelse continue;

            // shape
            const shape_arr = (info_obj.get("shape") orelse continue).array;
            var shape = try self.allocator.alloc(u64, shape_arr.items.len);
            var n_elems: u64 = 1;
            for (shape_arr.items, 0..) |dim, i| {
                shape[i] = @intCast(dim.integer);
                n_elems *= shape[i];
            }

            // data_offsets
            const offsets = (info_obj.get("data_offsets") orelse {
                self.allocator.free(shape);
                continue;
            }).array;
            if (offsets.items.len != 2) {
                self.allocator.free(shape);
                continue;
            }
            const rel_start: u64 = @intCast(offsets.items[0].integer);
            const rel_end: u64 = @intCast(offsets.items[1].integer);
            const data_len = rel_end - rel_start;

            const key = try self.allocator.dupe(u8, name);
            const shard_copy = try self.allocator.dupe(u8, shard_path);
            try self.entries.put(key, .{
                .dtype = dtype,
                .shape = shape,
                .shard_path = shard_copy,
                .data_offset = data_base + rel_start,
                .data_len = data_len,
                .n_elems = n_elems,
            });
        }
    }

    pub fn get(self: *const TensorIndex, name: []const u8) ?TensorInfo {
        return self.entries.get(name);
    }

    /// Load raw bytes of tensor into caller-owned buffer.
    pub fn loadRaw(self: *const TensorIndex, name: []const u8, allocator: std.mem.Allocator) ![]u8 {
        const info = self.entries.get(name) orelse return error.TensorNotFound;
        const buf = try allocator.alloc(u8, @intCast(info.data_len));
        errdefer allocator.free(buf);
        try preadFull(info.shard_path, buf, info.data_offset);
        return buf;
    }

    /// Load tensor as F32. If dtype is BF16, converts to F32. If already F32, copies.
    pub fn loadF32(self: *const TensorIndex, name: []const u8, allocator: std.mem.Allocator) ![]f32 {
        const info = self.entries.get(name) orelse return error.TensorNotFound;
        switch (info.dtype) {
            .F32 => {
                const buf = try self.loadRaw(name, allocator);
                // Reinterpret as f32 slice (same memory, just the type)
                const f32buf = try allocator.alloc(f32, info.n_elems);
                @memcpy(std.mem.sliceAsBytes(f32buf), buf);
                allocator.free(buf);
                return f32buf;
            },
            .BF16 => {
                const raw = try self.loadRaw(name, allocator);
                defer allocator.free(raw);
                const out = try allocator.alloc(f32, info.n_elems);
                bf16SliceToF32(@alignCast(std.mem.bytesAsSlice(u16, raw)), out);
                return out;
            },
            else => return error.UnsupportedDTypeForF32,
        }
    }

    /// Load tensor as []u32. Tensor must be U32 dtype.
    pub fn loadU32(self: *const TensorIndex, name: []const u8, allocator: std.mem.Allocator) ![]u32 {
        const info = self.entries.get(name) orelse return error.TensorNotFound;
        if (info.dtype != .U32) return error.WrongDType;
        const raw = try self.loadRaw(name, allocator);
        const out = try allocator.alloc(u32, info.n_elems);
        @memcpy(std.mem.sliceAsBytes(out), raw);
        allocator.free(raw);
        return out;
    }

    /// Load tensor as []u8. Tensor must be U8 dtype.
    pub fn loadU8(self: *const TensorIndex, name: []const u8, allocator: std.mem.Allocator) ![]u8 {
        const info = self.entries.get(name) orelse return error.TensorNotFound;
        if (info.dtype != .U8) return error.WrongDType;
        return self.loadRaw(name, allocator);
    }

    /// Load tensor as []i64 (copies into 8-byte aligned allocation).
    pub fn loadI64(self: *const TensorIndex, name: []const u8, allocator: std.mem.Allocator) ![]i64 {
        const info = self.entries.get(name) orelse return error.TensorNotFound;
        if (info.dtype != .I64) return error.WrongDType;
        const raw = try self.loadRaw(name, allocator);
        defer allocator.free(raw);
        // Allocate with alignment guarantee for i64
        const out = try allocator.alloc(i64, info.n_elems);
        @memcpy(std.mem.sliceAsBytes(out), raw);
        return out;
    }

    /// Load BF16 tensor and convert scales/biases to F32.
    /// Alias for loadF32 — just for clarity at call sites.
    pub fn loadBF16AsF32(self: *const TensorIndex, name: []const u8, allocator: std.mem.Allocator) ![]f32 {
        return self.loadF32(name, allocator);
    }
};

/// BF16 (brain float, upper 16 bits of f32) → f32 conversion.
pub fn bf16SliceToF32(src: []const u16, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    for (src, dst) |h, *d| {
        const u: u32 = @as(u32, h) << 16;
        d.* = @bitCast(u);
    }
}

/// pread the entire `len` bytes from `path` at `offset` into `buf`.
fn preadFull(shard_path: []const u8, buf: []u8, offset: u64) !void {
    const path_z = try std.heap.c_allocator.dupeZ(u8, shard_path);
    defer std.heap.c_allocator.free(path_z);

    const fd = std.c.open(path_z.ptr, .{});
    if (fd < 0) return error.FileOpenFailed;
    defer _ = std.c.close(fd);

    var total_read: usize = 0;
    while (total_read < buf.len) {
        const n = std.c.pread(fd, buf.ptr + total_read, buf.len - total_read, @intCast(offset + total_read));
        if (n < 0) return error.PreadFailed;
        if (n == 0) return error.UnexpectedEof;
        total_read += @intCast(n);
    }
}

/// Read entire file into caller-owned buffer.
fn readFileAlloc(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    const fd = std.c.open(path_z.ptr, .{});
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);

    var stat: std.c.Stat = undefined;
    if (std.c.fstat(fd, &stat) != 0) return error.StatFailed;
    const file_size: usize = @intCast(stat.size);

    const buf = try allocator.alloc(u8, file_size);
    errdefer allocator.free(buf);
    var total: usize = 0;
    while (total < file_size) {
        const n = std.c.pread(fd, buf.ptr + total, file_size - total, @intCast(total));
        if (n <= 0) return error.ReadFailed;
        total += @intCast(n);
    }
    return buf;
}

/// Build a TensorIndex from a model directory.
/// Reads model.safetensors.index.json (sharded) or model.safetensors (single).
pub fn buildIndex(allocator: std.mem.Allocator, model_dir: []const u8) !TensorIndex {
    var idx = TensorIndex.init(allocator);
    errdefer idx.deinit();

    // Try sharded index first
    const index_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors.index.json" });
    defer allocator.free(index_path);

    const index_path_z = try allocator.dupeZ(u8, index_path);
    defer allocator.free(index_path_z);
    const ifd = std.c.open(index_path_z.ptr, .{});
    const has_index = ifd >= 0;
    if (has_index) _ = std.c.close(ifd);

    if (has_index) {
        const content = try readFileAlloc(allocator, index_path);
        defer allocator.free(content);

        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
        defer parsed.deinit();

        const weight_map = parsed.value.object.get("weight_map") orelse return error.NoWeightMap;
        var seen_shards = std.StringHashMap(void).init(allocator);
        defer seen_shards.deinit();

        var it = weight_map.object.iterator();
        while (it.next()) |entry| {
            const shard_name = entry.value_ptr.*.string;
            if (seen_shards.contains(shard_name)) continue;
            try seen_shards.put(shard_name, {});

            const shard_path = try std.fs.path.join(allocator, &.{ model_dir, shard_name });
            defer allocator.free(shard_path);
            try idx.addShard(shard_path);
        }
        return idx;
    }

    // Fall back to single-file model.safetensors
    const single_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(single_path);
    try idx.addShard(single_path);
    return idx;
}
