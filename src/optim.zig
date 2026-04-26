/// Optimizers for training.
///
/// Currently implements AdamW — the standard optimizer for LLM fine-tuning.
/// Design follows the pattern of PyTorch/MLX optimizers:
///   - Initialize from a parameter tree
///   - `step(gradients)` updates parameters in-place
///   - State (momentum, variance) kept internally
const std = @import("std");
const array_mod = @import("array.zig");
const tree_mod = @import("tree.zig");

const Array = array_mod.Array;
const TreeEntry = tree_mod.TreeEntry;

/// AdamW optimizer state for a single parameter.
const ParamState = struct {
    m: Array,
    v: Array,
};

/// AdamW optimizer.
/// Reference: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
pub const AdamW = struct {
    allocator: std.mem.Allocator,

    params: []*Array,
    states: []ParamState,

    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,

    step_count: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        params: []*Array,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        stream: c.c.mlx_stream,
    ) !AdamW {
        const states = try allocator.alloc(ParamState, params.len);
        errdefer allocator.free(states);

        for (params, 0..) |param_ptr, i| {
            const shape = param_ptr.shape();
            var m_arr = c.c.mlx_array_new();
            var v_arr = c.c.mlx_array_new();
            try c.check(c.c.mlx_zeros(&m_arr, shape.ptr, @intCast(shape.len), c.c.MLX_FLOAT32, stream));
            try c.check(c.c.mlx_zeros(&v_arr, shape.ptr, @intCast(shape.len), c.c.MLX_FLOAT32, stream));

            states[i] = .{
                .m = Array.fromHandle(m_arr),
                .v = Array.fromHandle(v_arr),
            };
        }

        return .{
            .allocator = allocator,
            .params = params,
            .states = states,
            .lr = lr,
            .beta1 = beta1,
            .beta2 = beta2,
            .eps = eps,
            .weight_decay = weight_decay,
            .step_count = 0,
        };
    }

    pub fn initFromStruct(
        allocator: std.mem.Allocator,
        model: anytype,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        stream: c.c.mlx_stream,
    ) !AdamW {
        var ptrs = std.ArrayList(*Array).empty;
        defer ptrs.deinit(allocator);
        try tree_mod.treeToArrayPtrs(allocator, model, &ptrs);

        const params_copy = try allocator.dupe(*Array, ptrs.items);
        errdefer allocator.free(params_copy);

        return try init(allocator, params_copy, lr, beta1, beta2, eps, weight_decay, stream);
    }

    pub fn deinit(self: *AdamW) void {
        for (self.states) |state| {
            state.m.deinit();
            state.v.deinit();
        }
        self.allocator.free(self.states);
        self.allocator.free(self.params);
    }

    pub fn step(self: *AdamW, grads: []const Array, stream: c.c.mlx_stream) !void {
        std.debug.assert(grads.len == self.params.len);
        self.step_count += 1;

        // FUSION INTEGRATION POINT (R8.2): The per-parameter update loop below creates
        // ~15 intermediate Arrays per parameter. This can be replaced with
        // `compiledAdamWStep` from `src/ops/fused.zig`, which fuses the entire
        // m/v update + bias correction + step + weight decay into a single compiled
        // kernel launch via mlx_compile. Usage (per parameter):
        //   const fused = try fused_ops.compiledAdamWStep(self.allocator);
        //   defer fused.deinit();
        //   const result = try fused.call(&.{
        //       param, grad, state.m, state.v,
        //       sc_lr, sc_beta1, sc_beta2, sc_eps, sc_wd, sc_bias1, sc_bias2,
        //   }, self.allocator);
        //   param.* = result[0]; state.m = result[1]; state.v = result[2];
        //
        const t = @as(f32, @floatFromInt(self.step_count));
        const bias_correction1 = 1.0 - std.math.pow(f32, self.beta1, t);
        const bias_correction2 = 1.0 - std.math.pow(f32, self.beta2, t);

        const sc_lr = c.c.mlx_array_new_float32(self.lr);
        defer _ = c.c.mlx_array_free(sc_lr);
        const sc_eps = c.c.mlx_array_new_float32(self.eps);
        defer _ = c.c.mlx_array_free(sc_eps);
        const sc_beta1 = c.c.mlx_array_new_float32(self.beta1);
        defer _ = c.c.mlx_array_free(sc_beta1);
        const sc_omb1 = c.c.mlx_array_new_float32(1.0 - self.beta1);
        defer _ = c.c.mlx_array_free(sc_omb1);
        const sc_beta2 = c.c.mlx_array_new_float32(self.beta2);
        defer _ = c.c.mlx_array_free(sc_beta2);
        const sc_omb2 = c.c.mlx_array_new_float32(1.0 - self.beta2);
        defer _ = c.c.mlx_array_free(sc_omb2);
        const sc_lr_wd = c.c.mlx_array_new_float32(self.lr * self.weight_decay);
        defer _ = c.c.mlx_array_free(sc_lr_wd);
        const sc_bias1 = c.c.mlx_array_new_float32(bias_correction1);
        defer _ = c.c.mlx_array_free(sc_bias1);
        const sc_bias2 = c.c.mlx_array_new_float32(bias_correction2);
        defer _ = c.c.mlx_array_free(sc_bias2);

        for (self.params, self.states, grads) |param_ptr, *state, grad| {
            // m = beta1 * m + (1 - beta1) * grad
            var tmp1: c.c.mlx_array = .{ .ctx = null };
            try c.check(c.c.mlx_multiply(&tmp1, state.m.inner, sc_beta1, stream));
            defer _ = c.c.mlx_array_free(tmp1);
            var tmp2: c.c.mlx_array = .{ .ctx = null };
            try c.check(c.c.mlx_multiply(&tmp2, grad.inner, sc_omb1, stream));
            defer _ = c.c.mlx_array_free(tmp2);
            var new_m: c.c.mlx_array = .{ .ctx = null };
            try c.check(c.c.mlx_add(&new_m, tmp1, tmp2, stream));
            state.m.deinit();
            state.m = Array.fromHandle(new_m);

            // v = beta2 * v + (1 - beta2) * grad^2
            var grad_sq: c.c.mlx_array = .{ .ctx = null };
            try c.check(c.c.mlx_multiply(&grad_sq, grad.inner, grad.inner, stream));
            defer _ = c.c.mlx_array_free(grad_sq);
            var tmp3: c.c.mlx_array = .{ .ctx = null };
            try c.check(c.c.mlx_multiply(&tmp3, state.v.inner, sc_beta2, stream));
            defer _ = c.c.mlx_array_free(tmp3);
            var tmp4: c.c.mlx_array = .{ .ctx = null };
            try c.check(c.c.mlx_multiply(&tmp4, grad_sq, sc_omb2, stream));
            defer _ = c.c.mlx_array_free(tmp4);
            var new_v: c.c.mlx_array = .{ .ctx = null };
            try c.check(c.c.mlx_add(&new_v, tmp3, tmp4, stream));
            state.v.deinit();
            state.v = Array.fromHandle(new_v);

            // m_hat = m / bias_correction1
            var m_hat: c.c.mlx_array = .{ .ctx = null };
            try c.check(c.c.mlx_divide(&m_hat, state.m.inner, sc_bias1, stream));
            defer _ = c.c.mlx_array_free(m_hat);

            // v_hat = v / bias_correction2
            var v_hat: c.c.mlx_array = .{ .ctx = null };
            try c.check(c.c.mlx_divide(&v_hat, state.v.inner, sc_bias2, stream));
            defer _ = c.c.mlx_array_free(v_hat);

            // sqrt(v_hat) + eps
            var sqrt_v: c.c.mlx_array = .{ .ctx = null };
            try c.check(c.c.mlx_sqrt(&sqrt_v, v_hat, stream));
            defer _ = c.c.mlx_array_free(sqrt_v);
            var denom: c.c.mlx_array = .{ .ctx = null };
            try c.check(c.c.mlx_add(&denom, sqrt_v, sc_eps, stream));
            defer _ = c.c.mlx_array_free(denom);

            // step = lr * m_hat / denom
            var step_tmp: c.c.mlx_array = .{ .ctx = null };
            try c.check(c.c.mlx_divide(&step_tmp, m_hat, denom, stream));
            defer _ = c.c.mlx_array_free(step_tmp);
            var scaled_step: c.c.mlx_array = .{ .ctx = null };
            try c.check(c.c.mlx_multiply(&scaled_step, step_tmp, sc_lr, stream));
            defer _ = c.c.mlx_array_free(scaled_step);

            // weight_decay_term = lr * wd * param
            var wd_term: c.c.mlx_array = .{ .ctx = null };
            try c.check(c.c.mlx_multiply(&wd_term, param_ptr.inner, sc_lr_wd, stream));
            defer _ = c.c.mlx_array_free(wd_term);

            // param = param - scaled_step - wd_term
            var tmp5: c.c.mlx_array = .{ .ctx = null };
            try c.check(c.c.mlx_subtract(&tmp5, param_ptr.inner, scaled_step, stream));
            defer _ = c.c.mlx_array_free(tmp5);
            var final_param: c.c.mlx_array = .{ .ctx = null };
            try c.check(c.c.mlx_subtract(&final_param, tmp5, wd_term, stream));

            param_ptr.deinit();
            param_ptr.* = Array.fromHandle(final_param);
        }
    }
};

const c = @import("c.zig");
