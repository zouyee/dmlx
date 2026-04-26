const std = @import("std");
const array_mod = @import("array.zig");

const Array = array_mod.Array;

/// ScopedArrayArena — Track intermediate Arrays in a forward pass and free them in batch.
///
/// Usage:
///   var arena = ScopedArrayArena.init(ctx.allocator);
///   defer arena.deinit(); // frees all tracked Arrays
///   const a = try arena.track(try ops.matmul(ctx, x, w));
///   const b = try arena.track(try ops.add(ctx, a, bias));
///   // ... b is returned, a and intermediates are freed by defer
pub const ScopedArrayArena = struct {
    allocator: std.mem.Allocator,
    arrays: std.ArrayList(Array),

    pub fn init(allocator: std.mem.Allocator) ScopedArrayArena {
        return .{
            .allocator = allocator,
            .arrays = std.ArrayList(Array).empty,
        };
    }

    pub fn deinit(self: *ScopedArrayArena) void {
        for (self.arrays.items) |arr| {
            arr.deinit();
        }
        self.arrays.deinit(self.allocator);
    }



    /// Track an Array for later bulk release. Returns the same Array for chaining.
    pub fn track(self: *ScopedArrayArena, array: Array) !Array {
        try self.arrays.append(self.allocator, array);
        return array;
    }

    /// Track an Array only if the arena is non-null.
    pub fn trackOptional(a: ?*ScopedArrayArena, array: Array) !Array {
        if (a) |arena| {
            try arena.track(array);
        }
        return array;
    }
};
