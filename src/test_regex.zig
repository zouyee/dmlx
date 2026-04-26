const std = @import("std");
const Regex = @import("regex").Regex;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var re = try Regex.compile(allocator, "\\p{N}{1,3}");
    defer re.deinit();
    std.debug.print("Compiled OK\n", .{});
}
