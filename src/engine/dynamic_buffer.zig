/// DynamicBuffer — growable byte buffer with a size limit.
///
/// Used for HTTP request body reading. Replaces the fixed `var buf: [65536]u8`
/// pattern so that large requests (>64KB) are handled safely.
const std = @import("std");

pub const DynamicBuffer = struct {
    data: std.ArrayList(u8),
    max_size: usize,

    pub const Error = error{ PayloadTooLarge, OutOfMemory };

    /// Create a new empty DynamicBuffer with the given size limit.
    pub fn init(max_size: usize) DynamicBuffer {
        return .{
            .data = std.ArrayList(u8).empty,
            .max_size = max_size,
        };
    }

    /// Release all allocated memory.
    pub fn deinit(self: *DynamicBuffer, allocator: std.mem.Allocator) void {
        self.data.deinit(allocator);
    }

    /// Append bytes to the buffer. Returns `error.PayloadTooLarge` if the
    /// new total would exceed `max_size`.
    pub fn append(self: *DynamicBuffer, allocator: std.mem.Allocator, bytes: []const u8) Error!void {
        const new_len = self.data.items.len + bytes.len;
        if (new_len > self.max_size) {
            return error.PayloadTooLarge;
        }
        try self.data.appendSlice(allocator, bytes);
    }

    /// Return the accumulated bytes.
    pub fn items(self: *const DynamicBuffer) []u8 {
        return self.data.items;
    }

    /// Current number of bytes in the buffer.
    pub fn len(self: *const DynamicBuffer) usize {
        return self.data.items.len;
    }

    /// Ensure at least `additional` bytes of free capacity are available.
    pub fn ensureUnusedCapacity(self: *DynamicBuffer, allocator: std.mem.Allocator, additional: usize) Error!void {
        if (self.data.items.len + additional > self.max_size) {
            return error.PayloadTooLarge;
        }
        try self.data.ensureUnusedCapacity(allocator, additional);
    }
};
