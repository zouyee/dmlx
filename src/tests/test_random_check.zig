const std = @import("std");
const c = @import("mlx").c;
const nn = @import("mlx").nn;
const ops = @import("mlx").ops;

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
