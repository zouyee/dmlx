const std = @import("std");
const c = @import("../c.zig");
const nn = @import("../ops/nn.zig");
const ops = @import("../ops.zig");

test "check random exists" {
    c.initErrorHandler();
    const ctx = ops.EagerContext.init(std.testing.allocator);
    defer ctx.deinit();
    var linear = try nn.Linear.init(ctx, 4, 8, true);
    defer {
        linear.weight.deinit();
        if (linear.bias) |b| b.deinit();
    }
}
