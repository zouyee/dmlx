/// Automatic differentiation via mlx-c transforms.
const std = @import("std");
const c = @import("c.zig");
const array_mod = @import("array.zig");
const closure_mod = @import("closure.zig");

const Array = array_mod.Array;
pub const Closure = closure_mod.Closure;

inline fn defaultStream() c.c.mlx_stream {
    return c.c.mlx_default_cpu_stream_new();
}

fn toVectorArray(allocator: std.mem.Allocator, arrs: []const Array) !c.c.mlx_vector_array {
    var list = try allocator.alloc(c.c.mlx_array, arrs.len);
    defer allocator.free(list);
    for (arrs, 0..) |arr, i| {
        list[i] = arr.inner;
    }
    return c.c.mlx_vector_array_new_data(list.ptr, list.len);
}

pub const ValueAndGradClosure = struct {
    inner: c.c.mlx_closure_value_and_grad,

    pub fn deinit(self: ValueAndGradClosure) void {
        _ = c.c.mlx_closure_value_and_grad_free(self.inner);
    }

    pub fn apply(
        self: ValueAndGradClosure,
        inputs: []const Array,
        allocator: std.mem.Allocator,
    ) !struct { value: []Array, grad: []Array } {
        const in_vec = try toVectorArray(allocator, inputs);
        defer _ = c.c.mlx_vector_array_free(in_vec);

        var value_vec: c.c.mlx_vector_array = .{ .ctx = null };
        var grad_vec: c.c.mlx_vector_array = .{ .ctx = null };
        try c.check(c.c.mlx_closure_value_and_grad_apply(&value_vec, &grad_vec, self.inner, in_vec));

        const n_val = c.c.mlx_vector_array_size(value_vec);
        var value = try allocator.alloc(Array, n_val);
        errdefer {
            for (value) |arr| {
                _ = c.c.mlx_array_free(arr.inner);
            }
            allocator.free(value);
        }
        for (0..n_val) |i| {
            var arr = c.c.mlx_array_new();
            try c.check(c.c.mlx_vector_array_get(&arr, value_vec, i));
            value[i] = Array.fromHandle(arr);
        }
        _ = c.c.mlx_vector_array_free(value_vec);

        const n_grad = c.c.mlx_vector_array_size(grad_vec);
        var grad = try allocator.alloc(Array, n_grad);
        errdefer {
            for (grad) |arr| {
                _ = c.c.mlx_array_free(arr.inner);
            }
            allocator.free(grad);
        }
        for (0..n_grad) |i| {
            var arr = c.c.mlx_array_new();
            try c.check(c.c.mlx_vector_array_get(&arr, grad_vec, i));
            grad[i] = Array.fromHandle(arr);
        }
        _ = c.c.mlx_vector_array_free(grad_vec);

        return .{ .value = value, .grad = grad };
    }
};

/// Create a value-and-grad closure from a forward closure.
pub fn valueAndGrad(
    closure: Closure,
    argnums: []const i32,
) !ValueAndGradClosure {
    var vg: c.c.mlx_closure_value_and_grad = .{ .ctx = null };
    try c.check(c.c.mlx_value_and_grad(&vg, closure.inner, argnums.ptr, argnums.len));
    return .{ .inner = vg };
}

/// Compute vector-Jacobian product.
pub fn vjp(
    closure: Closure,
    primals: []const Array,
    cotangents: []const Array,
    allocator: std.mem.Allocator,
) !struct { outputs: []Array, grads: []Array } {
    const primal_vec = try toVectorArray(allocator, primals);
    defer _ = c.c.mlx_vector_array_free(primal_vec);
    const cotangent_vec = try toVectorArray(allocator, cotangents);
    defer _ = c.c.mlx_vector_array_free(cotangent_vec);

    var out_vec: c.c.mlx_vector_array = .{ .ctx = null };
    var grad_vec: c.c.mlx_vector_array = .{ .ctx = null };
    try c.check(c.c.mlx_vjp(&out_vec, &grad_vec, closure.inner, primal_vec, cotangent_vec));

    const n_out = c.c.mlx_vector_array_size(out_vec);
    var outputs = try allocator.alloc(Array, n_out);
    errdefer {
        for (outputs) |arr| {
            _ = c.c.mlx_array_free(arr.inner);
        }
        allocator.free(outputs);
    }
    for (0..n_out) |i| {
        var arr = c.c.mlx_array_new();
        try c.check(c.c.mlx_vector_array_get(&arr, out_vec, i));
        outputs[i] = Array.fromHandle(arr);
    }
    _ = c.c.mlx_vector_array_free(out_vec);

    const n_grad = c.c.mlx_vector_array_size(grad_vec);
    var grads = try allocator.alloc(Array, n_grad);
    errdefer {
        for (grads) |arr| {
            _ = c.c.mlx_array_free(arr.inner);
        }
        allocator.free(grads);
    }
    for (0..n_grad) |i| {
        var arr = c.c.mlx_array_new();
        try c.check(c.c.mlx_vector_array_get(&arr, grad_vec, i));
        grads[i] = Array.fromHandle(arr);
    }
    _ = c.c.mlx_vector_array_free(grad_vec);

    return .{ .outputs = outputs, .grads = grads };
}

/// Compute Jacobian-vector product.
pub fn jvp(
    closure: Closure,
    primals: []const Array,
    tangents: []const Array,
    allocator: std.mem.Allocator,
) !struct { outputs: []Array, tangents_out: []Array } {
    const primal_vec = try toVectorArray(allocator, primals);
    defer _ = c.c.mlx_vector_array_free(primal_vec);
    const tangent_vec = try toVectorArray(allocator, tangents);
    defer _ = c.c.mlx_vector_array_free(tangent_vec);

    var out_vec: c.c.mlx_vector_array = .{ .ctx = null };
    var tan_vec: c.c.mlx_vector_array = .{ .ctx = null };
    try c.check(c.c.mlx_jvp(&out_vec, &tan_vec, closure.inner, primal_vec, tangent_vec));

    const n_out = c.c.mlx_vector_array_size(out_vec);
    var outputs = try allocator.alloc(Array, n_out);
    errdefer {
        for (outputs) |arr| {
            _ = c.c.mlx_array_free(arr.inner);
        }
        allocator.free(outputs);
    }
    for (0..n_out) |i| {
        var arr = c.c.mlx_array_new();
        try c.check(c.c.mlx_vector_array_get(&arr, out_vec, i));
        outputs[i] = Array.fromHandle(arr);
    }
    _ = c.c.mlx_vector_array_free(out_vec);

    const n_tan = c.c.mlx_vector_array_size(tan_vec);
    var tangents_out = try allocator.alloc(Array, n_tan);
    errdefer {
        for (tangents_out) |arr| {
            _ = c.c.mlx_array_free(arr.inner);
        }
        allocator.free(tangents_out);
    }
    for (0..n_tan) |i| {
        var arr = c.c.mlx_array_new();
        try c.check(c.c.mlx_vector_array_get(&arr, tan_vec, i));
        tangents_out[i] = Array.fromHandle(arr);
    }
    _ = c.c.mlx_vector_array_free(tan_vec);

    return .{ .outputs = outputs, .tangents_out = tangents_out };
}
