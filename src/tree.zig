/// Parameter Tree System — tree_map, tree_flatten, tree_unflatten
///
/// Central abstraction for managing nested model parameters.
/// Enables:
///   - Optimizers to traverse all trainable params
///   - LoRA to identify and replace target layers
///   - Checkpoint save/load with structure preservation
///   - Model surgery (quantization, pruning, merging)
///
/// Design: Zig comptime introspection over struct fields.
/// Any struct with Array fields or nested param structs can be flattened.
const std = @import("std");
const array_mod = @import("array.zig");
const c = @import("c.zig");

const Array = array_mod.Array;

/// A flattened parameter entry with dotted path key.
pub const TreeEntry = struct {
    key: []const u8, // e.g., "layers.0.attention.wq.weight"
    value: Array,
};

/// Recursively flatten a struct into a list of (key, Array) entries.
/// Only fields of type `Array` are collected as leaves.
/// Nested structs are traversed with dotted keys.
pub fn treeFlatten(
    allocator: std.mem.Allocator,
    prefix: []const u8,
    value: anytype,
    entries: *std.ArrayList(TreeEntry),
) !void {
    const T = @TypeOf(value);
    const type_info = @typeInfo(T);

    switch (type_info) {
        .@"struct" => |s| {
            inline for (s.fields) |field| {
                const field_value = @field(value, field.name);
                const FieldType = @TypeOf(field_value);

                // Build key: "prefix.field_name"
                const key = if (prefix.len == 0)
                    try allocator.dupe(u8, field.name)
                else
                    try std.fmt.allocPrint(allocator, "{s}.{s}", .{ prefix, field.name });
                errdefer allocator.free(key);

                if (FieldType == Array) {
                    try entries.append(allocator, .{ .key = key, .value = field_value });
                } else if (@typeInfo(FieldType) == .@"struct") {
                    try treeFlatten(allocator, key, field_value, entries);
                    allocator.free(key); // key consumed in recursion
                } else {
                    allocator.free(key); // not a param, discard
                }
            }
        },
        .@"pointer" => |ptr| {
            if (ptr.size == .one and @typeInfo(ptr.child) == .@"struct") {
                try treeFlatten(allocator, prefix, value.*, entries);
            }
        },
        else => {},
    }
}

/// Convenience wrapper: flatten a struct starting from "" prefix.
pub fn flattenStruct(
    allocator: std.mem.Allocator,
    root: anytype,
) ![]TreeEntry {
    var entries = std.ArrayList(TreeEntry).empty;
    errdefer {
        for (entries.items) |e| {
            allocator.free(e.key);
        }
        entries.deinit(allocator);
    }
    try treeFlatten(allocator, "", root, &entries);
    return entries.toOwnedSlice(allocator);
}

/// Free entries returned by flattenStruct.
pub fn freeEntries(allocator: std.mem.Allocator, entries: []TreeEntry) void {
    for (entries) |e| {
        allocator.free(e.key);
    }
    allocator.free(entries);
}

/// Apply a function to every Array in a struct, producing a new struct.
/// The function signature: fn(Array) !Array
pub fn treeMap(
    comptime T: type,
    allocator: std.mem.Allocator,
    tree: T,
    map_fn: *const fn (Array, std.mem.Allocator) error{MlxError}!Array,
) !T {
    var result: T = undefined;
    const type_info = @typeInfo(T);

    switch (type_info) {
        .@"struct" => |s| {
            inline for (s.fields) |field| {
                const field_value = @field(tree, field.name);
                const FieldType = @TypeOf(field_value);

                if (FieldType == Array) {
                    @field(result, field.name) = try map_fn(field_value, allocator);
                } else if (@typeInfo(FieldType) == .@"struct") {
                    @field(result, field.name) = try treeMap(FieldType, allocator, field_value, map_fn);
                } else {
                    @field(result, field.name) = field_value;
                }
            }
        },
        else => @compileError("treeMap only works on structs"),
    }

    return result;
}

/// Apply a function to every Array in a struct **in-place**.
pub fn treeMapInPlace(
    comptime T: type,
    tree: *T,
    map_fn: *const fn (*Array) error{MlxError}!void,
) !void {
    const type_info = @typeInfo(T);

    switch (type_info) {
        .@"struct" => |s| {
            inline for (s.fields) |field| {
                const FieldType = @TypeOf(@field(tree.*, field.name));

                if (FieldType == Array) {
                    try map_fn(&@field(tree.*, field.name));
                } else if (@typeInfo(FieldType) == .@"struct") {
                    try treeMapInPlace(FieldType, &@field(tree.*, field.name), map_fn);
                }
            }
        },
        else => @compileError("treeMapInPlace only works on structs"),
    }
}

/// Set Arrays in a struct from a flat slice, maintaining the same order as treeToArrays.
pub fn treeSetArrays(tree: anytype, arrays: []const Array, idx: *usize) void {
    const T = @TypeOf(tree);
    const type_info = @typeInfo(T);

    switch (type_info) {
        .@"struct" => |s| {
            inline for (s.fields) |field| {
                const field_ptr = &@field(tree, field.name);
                const FieldType = @TypeOf(field_ptr.*);

                if (FieldType == Array) {
                    field_ptr.deinit();
                    var copy = c.c.mlx_array_new();
                    _ = c.c.mlx_array_set(&copy, arrays[idx.*].inner);
                    field_ptr.* = Array.fromHandle(copy);
                    idx.* += 1;
                } else if (@typeInfo(FieldType) == .@"struct") {
                    treeSetArrays(field_ptr, arrays, idx);
                } else if (@typeInfo(FieldType) == .@"pointer") {
                    const ptr = @typeInfo(FieldType).pointer;
                    if (ptr.size == .slice and @typeInfo(ptr.child) == .@"struct") {
                        for (field_ptr.*) |*item| {
                            treeSetArrays(item, arrays, idx);
                        }
                    } else if (ptr.size == .one and @typeInfo(ptr.child) == .@"struct") {
                        treeSetArrays(field_ptr.*, arrays, idx);
                    }
                }
            }
        },
        .@"pointer" => |ptr| {
            if (ptr.size == .one and @typeInfo(ptr.child) == .@"struct") {
                const S = @typeInfo(ptr.child).@"struct";
                inline for (S.fields) |field| {
                    const field_ptr = &@field(tree, field.name);
                    const FieldType = @TypeOf(field_ptr.*);

                    if (FieldType == Array) {
                        field_ptr.deinit();
                        var copy = c.c.mlx_array_new();
                        _ = c.c.mlx_array_set(&copy, arrays[idx.*].inner);
                        field_ptr.* = Array.fromHandle(copy);
                        idx.* += 1;
                    } else if (@typeInfo(FieldType) == .@"struct") {
                        treeSetArrays(field_ptr, arrays, idx);
                    } else if (@typeInfo(FieldType) == .@"pointer") {
                        const fptr = @typeInfo(FieldType).pointer;
                        if (fptr.size == .one and @typeInfo(fptr.child) == .@"struct") {
                            treeSetArrays(field_ptr.*, arrays, idx);
                        } else if (fptr.size == .slice and @typeInfo(fptr.child) == .@"struct") {
                            for (field_ptr.*) |*item| {
                                treeSetArrays(item, arrays, idx);
                            }
                        }
                    }
                }
            } else if (ptr.size == .slice and @typeInfo(ptr.child) == .@"struct") {
                for (tree) |*item| {
                    treeSetArrays(item, arrays, idx);
                }
            }
        },
        else => {},
    }
}

/// Count total number of Array leaves in a struct.
pub fn treeSize(tree: anytype) usize {
    const T = @TypeOf(tree);
    const type_info = @typeInfo(T);
    var count: usize = 0;

    switch (type_info) {
        .@"struct" => |s| {
            inline for (s.fields) |field| {
                const field_value = @field(tree, field.name);
                const FieldType = @TypeOf(field_value);

                if (FieldType == Array) {
                    count += 1;
                } else if (@typeInfo(FieldType) == .@"struct") {
                    count += treeSize(field_value);
                }
            }
        },
        else => {},
    }

    return count;
}

/// Collect all Arrays from a struct into a flat slice (no keys).
pub fn treeToArrays(
    allocator: std.mem.Allocator,
    tree: anytype,
) ![]Array {
    var arrays = std.ArrayList(Array).empty;
    errdefer arrays.deinit(allocator);

    const T = @TypeOf(tree);
    const type_info = @typeInfo(T);

    switch (type_info) {
        .@"struct" => |s| {
            inline for (s.fields) |field| {
                const field_value = @field(tree, field.name);
                const FieldType = @TypeOf(field_value);

                if (FieldType == Array) {
                    try arrays.append(allocator, field_value);
                } else if (@typeInfo(FieldType) == .@"struct") {
                    const nested = try treeToArrays(allocator, field_value);
                    defer allocator.free(nested);
                    try arrays.appendSlice(allocator, nested);
                } else if (@typeInfo(FieldType) == .@"pointer") {
                    const ptr = @typeInfo(FieldType).pointer;
                    if (ptr.size == .slice and @typeInfo(ptr.child) == .@"struct") {
                        for (field_value) |item| {
                            const nested = try treeToArrays(allocator, item);
                            defer allocator.free(nested);
                            try arrays.appendSlice(allocator, nested);
                        }
                    } else if (ptr.size == .one and @typeInfo(ptr.child) == .@"struct") {
                        const nested = try treeToArrays(allocator, field_value);
                        defer allocator.free(nested);
                        try arrays.appendSlice(allocator, nested);
                    }
                }
            }
        },
        else => {},
    }

    return arrays.toOwnedSlice(allocator);
}

/// Collect pointers to all Array fields in a struct.
/// Enables optimizers to update model parameters in-place.
pub fn treeToArrayPtrs(
    allocator: std.mem.Allocator,
    tree: anytype,
    arrays: *std.ArrayList(*Array),
) !void {
    const T = @TypeOf(tree);
    const type_info = @typeInfo(T);

    switch (type_info) {
        .@"struct" => |s| {
            inline for (s.fields) |field| {
                const field_raw = &@field(tree, field.name);
                const FieldType = @TypeOf(field_raw.*);

                if (FieldType == Array) {
                    const field_ptr: *Array = @constCast(field_raw);
                    try arrays.append(allocator, field_ptr);
                } else if (@typeInfo(FieldType) == .@"struct") {
                    try treeToArrayPtrs(allocator, field_raw, arrays);
                }
            }
        },
        .@"pointer" => |ptr| {
            if (ptr.size == .one and @typeInfo(ptr.child) == .@"struct") {
                const S = @typeInfo(ptr.child).@"struct";
                inline for (S.fields) |field| {
                    const field_ptr = &@field(tree, field.name);
                    const FieldType = @TypeOf(field_ptr.*);

                    if (FieldType == Array) {
                        try arrays.append(allocator, @constCast(field_ptr));
                    } else if (@typeInfo(FieldType) == .@"struct") {
                        try treeToArrayPtrs(allocator, field_ptr, arrays);
                    } else if (@typeInfo(FieldType) == .@"pointer") {
                        const fptr = @typeInfo(FieldType).pointer;
                        if (fptr.size == .one and @typeInfo(fptr.child) == .@"struct") {
                            try treeToArrayPtrs(allocator, field_ptr.*, arrays);
                        } else if (fptr.size == .slice and @typeInfo(fptr.child) == .@"struct") {
                            for (field_ptr.*) |*item| {
                                try treeToArrayPtrs(allocator, item, arrays);
                            }
                        }
                    }
                }
            } else if (ptr.size == .slice and @typeInfo(ptr.child) == .@"struct") {
                for (tree) |*item| {
                    try treeToArrayPtrs(allocator, item, arrays);
                }
            }
        },
        else => {},
    }
}
