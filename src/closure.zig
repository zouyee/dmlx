/// Closure wrapper for mlx-c closures, enabling Zig functions to be used with transforms.
const std = @import("std");
const c = @import("c.zig");
const array_mod = @import("array.zig");

const Array = array_mod.Array;

const ClosurePayload = struct {
    zig_fn: *const fn (inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array,
    allocator: std.mem.Allocator,
};

fn closureCallback(
    res: [*c]c.c.mlx_vector_array,
    inputs: c.c.mlx_vector_array,
    payload: ?*anyopaque,
) callconv(.c) c_int {
    const p: *ClosurePayload = @ptrCast(@alignCast(payload.?));

    const n_inputs = c.c.mlx_vector_array_size(inputs);
    var in_slice = p.allocator.alloc(Array, n_inputs) catch return 1;
    defer {
        for (in_slice) |arr| {
            _ = c.c.mlx_array_free(arr.inner);
        }
        p.allocator.free(in_slice);
    }
    for (0..n_inputs) |i| {
        var arr = c.c.mlx_array_new();
        if (c.c.mlx_vector_array_get(&arr, inputs, i) != 0) return 1;
        in_slice[i] = Array.fromHandle(arr);
    }

    const out = p.zig_fn(in_slice, p.allocator) catch return 1;
    defer {
        for (out) |arr| {
            _ = c.c.mlx_array_free(arr.inner);
        }
        p.allocator.free(out);
    }

    var out_arrs = p.allocator.alloc(c.c.mlx_array, out.len) catch return 1;
    defer p.allocator.free(out_arrs);
    for (out, 0..) |arr, i| {
        out_arrs[i] = c.c.mlx_array_new();
        if (c.c.mlx_array_set(&out_arrs[i], arr.inner) != 0) return 1;
    }

    res.* = c.c.mlx_vector_array_new_data(out_arrs.ptr, out_arrs.len);
    return 0;
}

fn closureDtor(payload: ?*anyopaque) callconv(.c) void {
    const p: *ClosurePayload = @ptrCast(@alignCast(payload.?));
    p.allocator.destroy(p);
}

pub const Closure = struct {
    inner: c.c.mlx_closure,

    pub fn init(
        zig_fn: *const fn (inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array,
        allocator: std.mem.Allocator,
    ) !Closure {
        const payload = try allocator.create(ClosurePayload);
        payload.* = .{
            .zig_fn = zig_fn,
            .allocator = allocator,
        };
        const closure = c.c.mlx_closure_new_func_payload(closureCallback, payload, closureDtor);
        return .{ .inner = closure };
    }

    pub fn deinit(self: Closure) void {
        _ = c.c.mlx_closure_free(self.inner);
    }

    pub fn apply(self: Closure, inputs: []const Array, allocator: std.mem.Allocator) ![]Array {
        var in_arrs = try allocator.alloc(c.c.mlx_array, inputs.len);
        defer allocator.free(in_arrs);
        for (inputs, 0..) |arr, i| {
            in_arrs[i] = arr.inner;
        }
        const in_vec = c.c.mlx_vector_array_new_data(in_arrs.ptr, in_arrs.len);
        defer _ = c.c.mlx_vector_array_free(in_vec);

        var out_vec: c.c.mlx_vector_array = .{ .ctx = null };
        try c.check(c.c.mlx_closure_apply(&out_vec, self.inner, in_vec));

        const n = c.c.mlx_vector_array_size(out_vec);
        var result = try allocator.alloc(Array, n);
        errdefer {
            for (result) |arr| {
                _ = c.c.mlx_array_free(arr.inner);
            }
            allocator.free(result);
        }
        for (0..n) |i| {
            var arr = c.c.mlx_array_new();
            try c.check(c.c.mlx_vector_array_get(&arr, out_vec, i));
            result[i] = Array.fromHandle(arr);
        }
        _ = c.c.mlx_vector_array_free(out_vec);
        return result;
    }
};
