/// NumPy .npy file format support.
/// Implements reading and writing of the NumPy .npy binary format.
/// Note: .npz (ZIP archive of multiple .npy files) is not yet supported.
const std = @import("std");
const array_mod = @import("../array.zig");
const dtype_mod = @import("../dtype.zig");

pub const Array = array_mod.Array;

/// Magic bytes for .npy format
const NPY_MAGIC = "NUMPY";
const NPY_VERSION_MAJOR: u8 = 1;
const NPY_VERSION_MINOR: u8 = 0;

/// NPY header structure
const NpyHeader = struct {
    dtype: []const u8,
    fortran_order: bool,
    shape: []const usize,
};

/// Write a .npy format header for the given dtype, shape, and memory order.
fn writeNpyHeader(allocator: std.mem.Allocator, dtype: dtype_mod.Dtype, shape: []const ShapeElem, fortran_order: bool) ![]u8 {
    var header = std.ArrayList(u8).empty;

    // Magic bytes
    try header.appendSlice(NPY_MAGIC);
    try header.append(NPY_VERSION_MAJOR);
    try header.append(NPY_VERSION_MINOR);

    // Header length (2 bytes, little-endian) - will fill this later
    const header_len_pos = header.items.len;
    try header.append(0);
    try header.append(0);

    // Write dictionary
    try header.appendSlice("{'descr': '");

    // Dtype descriptor
    const dtype_str = switch (dtype.val) {
        .float32 => "<f4",
        .float64 => "<f8",
        .int32 => "<i4",
        .int64 => "<i8",
        .uint32 => "<u4",
        .uint64 => "<u8",
        .bool => "?",
        else => "<f4", // default to float32
    };
    try header.appendSlice(dtype_str);
    try header.appendSlice("', 'fortran_order': ");
    try header.appendSlice(if (fortran_order) "True" else "False");
    try header.appendSlice(", 'shape': (");

    // Shape
    for (shape, 0..shape.len) |dim, i| {
        try header.appendSlice(if (i == 0) "" else ", ");
        try std.fmt.formatInt(dim, 10, .lower, .{}, header.writer());
    }
    try header.appendSlice("), }");

    // Pad with spaces to make total header length a multiple of 16
    const header_len = header.items.len;
    const padded_len = ((header_len + 1 + 15) / 16) * 16;
    while (header.items.len < padded_len) {
        try header.append(' ');
    }
    try header.append('\n');

    // Fill in header length
    const total_header_len = header.items.len;
    std.mem.writeInt(u16, header.items[header_len_pos .. header_len_pos + 2], @intCast(total_header_len - 10), .little);

    return header.toOwnedSlice(allocator);
}

const ShapeElem = array_mod.ShapeElem;

/// Save array to .npy file
pub fn save(ctx: *std.mem.Allocator, io: std.Io, path: []const u8, arr: Array) !void {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.createFile(io, path, .{});
    defer file.close(io);

    const dtype = arr.dtype();
    const shape = arr.shape();

    // Write header
    const header = try writeNpyHeader(ctx.*, dtype, shape, false);
    defer ctx.*.free(header);
    try file.writeStreamingAll(io, header);

    // Write data
    switch (dtype.val) {
        .float32 => {
            const data = arr.dataSlice(f32);
            try file.writeStreamingAll(io, std.mem.sliceAsBytes(data));
        },
        .float64 => {
            const data = arr.dataSlice(f64);
            try file.writeStreamingAll(io, std.mem.sliceAsBytes(data));
        },
        .int32 => {
            const data = arr.dataSlice(i32);
            try file.writeStreamingAll(io, std.mem.sliceAsBytes(data));
        },
        .int64 => {
            const data = arr.dataSlice(i64);
            try file.writeStreamingAll(io, std.mem.sliceAsBytes(data));
        },
        .uint32 => {
            const data = arr.dataSlice(u32);
            try file.writeStreamingAll(io, std.mem.sliceAsBytes(data));
        },
        .uint64 => {
            const data = arr.dataSlice(u64);
            try file.writeStreamingAll(io, std.mem.sliceAsBytes(data));
        },
        else => return error.UnsupportedDtype,
    }
}

/// Load array from .npy file
pub fn load(ctx: *std.mem.Allocator, io: std.Io, path: []const u8, allocator: std.mem.Allocator) !Array {
    const contents = try std.Io.Dir.cwd().readFileAlloc(io, path, ctx.*, .unlimited);
    defer ctx.*.free(contents);

    // Verify magic
    if (contents.len < 10 or !std.mem.eql(u8, contents[0..6], NPY_MAGIC)) {
        return error.InvalidNpyFormat;
    }

    const major_version = contents[6];
    const minor_version = contents[7];
    _ = major_version;
    _ = minor_version;

    const header_len = std.mem.readInt(u16, contents[8..10], .little);

    // Parse header to get dtype and shape
    const header_str = contents[10..header_len];

    // Find dtype
    const descr_start = std.mem.indexOf(u8, header_str, "{'descr': '") orelse return error.InvalidNpyFormat;
    const dtype_start = descr_start + 12;
    const dtype_end = std.mem.indexOf(u8, header_str[dtype_start..], "'") orelse return error.InvalidNpyFormat;
    const dtype_str = header_str[dtype_start .. dtype_start + dtype_end];

    const dtype = switch (dtype_str[1]) {
        'f' => switch (dtype_str[2]) {
            '4' => dtype_mod.float32,
            '8' => dtype_mod.float64,
            else => return error.UnsupportedDtype,
        },
        'i' => switch (dtype_str[2]) {
            '4' => dtype_mod.int32,
            '8' => dtype_mod.int64,
            else => return error.UnsupportedDtype,
        },
        'u' => switch (dtype_str[2]) {
            '4' => dtype_mod.uint32,
            '8' => dtype_mod.uint64,
            else => return error.UnsupportedDtype,
        },
        '?' => dtype_mod.bool,
        else => dtype_mod.float32,
    };

    // Find shape
    const shape_start = std.mem.indexOf(u8, header_str, "'shape': (") orelse return error.InvalidNpyFormat;
    var shape_end = std.mem.indexOf(u8, header_str[shape_start..], ")") orelse return error.InvalidNpyFormat;
    shape_end += shape_start;

    var shape = std.ArrayList(usize).empty;
    defer shape.deinit();

    var pos = shape_start + 10;
    while (pos < shape_end) {
        // Skip spaces and commas
        while (pos < shape_end and (header_str[pos] == ' ' or header_str[pos] == ',')) pos += 1;
        if (pos >= shape_end) break;

        var num_end = pos;
        while (num_end < shape_end and header_str[num_end] >= '0' and header_str[num_end] <= '9') num_end += 1;
        if (num_end > pos) {
            const num_str = header_str[pos..num_end];
            const num = try std.fmt.parseInt(usize, num_str, 10);
            try shape.append(num);
        }
        pos = num_end;
    }

    // Read data
    const data_start = header_len;

    switch (dtype.val) {
        .float32 => {
            var arr = try array_mod.zeros(allocator, shape.items, dtype);
            const arr_data = arr.dataSlice(f32);
            @memcpy(std.mem.sliceAsBytes(arr_data), contents[data_start..]);
            return arr;
        },
        .float64 => {
            var arr = try array_mod.zeros(allocator, shape.items, dtype);
            const arr_data = arr.dataSlice(f64);
            @memcpy(std.mem.sliceAsBytes(arr_data), contents[data_start..]);
            return arr;
        },
        .int32 => {
            var arr = try array_mod.zeros(allocator, shape.items, dtype);
            const arr_data = arr.dataSlice(i32);
            @memcpy(std.mem.sliceAsBytes(arr_data), contents[data_start..]);
            return arr;
        },
        .int64 => {
            var arr = try array_mod.zeros(allocator, shape.items, dtype);
            const arr_data = arr.dataSlice(i64);
            @memcpy(std.mem.sliceAsBytes(arr_data), contents[data_start..]);
            return arr;
        },
        .uint32 => {
            var arr = try array_mod.zeros(allocator, shape.items, dtype);
            const arr_data = arr.dataSlice(u32);
            @memcpy(std.mem.sliceAsBytes(arr_data), contents[data_start..]);
            return arr;
        },
        .uint64 => {
            var arr = try array_mod.zeros(allocator, shape.items, dtype);
            const arr_data = arr.dataSlice(u64);
            @memcpy(std.mem.sliceAsBytes(arr_data), contents[data_start..]);
            return arr;
        },
        .bool => {
            var arr = try array_mod.zeros(allocator, shape.items, dtype);
            const arr_data = arr.dataSlice(bool);
            @memcpy(std.mem.sliceAsBytes(arr_data), contents[data_start..]);
            return arr;
        },
        else => return error.UnsupportedDtype,
    }
}
