/// Activation functions backed by mlx-c operations.
/// All functions operate through the MLX computation graph (GPU/CPU accelerated).
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const dtype_mod = @import("../dtype.zig");
const ops_mod = @import("../ops.zig");
const math_mod = @import("math.zig");
const comparison_mod = @import("comparison.zig");

pub const Array = array_mod.Array;
pub const Dtype = dtype_mod.Dtype;
const ShapeElem = array_mod.ShapeElem;
const EagerContext = ops_mod.EagerContext;

// ============================================================
// GELU (Gaussian Error Linear Unit) — via mlx-c ops
// ============================================================

pub fn gelu(ctx: EagerContext, input: Array) !Array {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const x_sq = try ops_mod.multiply(ctx, input, input);
    defer x_sq.deinit();
    const x_cu = try ops_mod.multiply(ctx, x_sq, input);
    defer x_cu.deinit();

    const coeff = try ops_mod.scalarF32(ctx, 0.044715);
    defer coeff.deinit();
    const inner_term = try ops_mod.multiply(ctx, coeff, x_cu);
    defer inner_term.deinit();
    const inner_sum = try ops_mod.add(ctx, input, inner_term);
    defer inner_sum.deinit();

    const sqrt_2_pi = try ops_mod.scalarF32(ctx, @sqrt(2.0 / std.math.pi));
    defer sqrt_2_pi.deinit();
    const tanh_arg = try ops_mod.multiply(ctx, sqrt_2_pi, inner_sum);
    defer tanh_arg.deinit();
    const tanh_val = try ops_mod.tanh(ctx, tanh_arg);
    defer tanh_val.deinit();

    const one = try ops_mod.scalarF32(ctx, 1.0);
    defer one.deinit();
    const one_plus_tanh = try ops_mod.add(ctx, one, tanh_val);
    defer one_plus_tanh.deinit();

    const half = try ops_mod.scalarF32(ctx, 0.5);
    defer half.deinit();
    const half_x = try ops_mod.multiply(ctx, half, input);
    defer half_x.deinit();

    return ops_mod.multiply(ctx, half_x, one_plus_tanh);
}

// Alias
pub const geluApprox = gelu;

// ============================================================
// SiLU / Swish — via mlx-c ops
// ============================================================

pub fn silu(ctx: EagerContext, input: Array) !Array {
    // SiLU(x) = x * sigmoid(x)
    const sig = try ops_mod.sigmoid(ctx, input);
    defer sig.deinit();
    return ops_mod.multiply(ctx, input, sig);
}

pub const swish = silu;

// ============================================================
// Leaky ReLU — via mlx-c ops
// ============================================================

pub fn leakyRelu(ctx: EagerContext, input: Array, negative_slope: f32) !Array {
    const zero = try ops_mod.scalarF32(ctx, 0.0);
    defer zero.deinit();
    const cond = try comparison_mod.greaterEqual(ctx, input, zero);
    defer cond.deinit();
    const slope = try ops_mod.scalarF32(ctx, negative_slope);
    defer slope.deinit();
    const neg_part = try ops_mod.multiply(ctx, slope, input);
    defer neg_part.deinit();
    return ops_mod.where(ctx, cond, input, neg_part);
}

// ============================================================
// Softplus — via mlx-c ops
// ============================================================

pub fn softplus(ctx: EagerContext, input: Array) !Array {
    // softplus(x) = log(1 + exp(x)), with x > 20 → x for stability
    const threshold = try ops_mod.scalarF32(ctx, 20.0);
    defer threshold.deinit();
    const cond = try comparison_mod.greater(ctx, input, threshold);
    defer cond.deinit();
    const exp_x = try ops_mod.exp(ctx, input);
    defer exp_x.deinit();
    const one = try ops_mod.scalarF32(ctx, 1.0);
    defer one.deinit();
    const one_plus_exp = try ops_mod.add(ctx, one, exp_x);
    defer one_plus_exp.deinit();
    const log_val = try ops_mod.log(ctx, one_plus_exp);
    defer log_val.deinit();
    return ops_mod.where(ctx, cond, input, log_val);
}

// ============================================================
// Mish — via mlx-c ops
// ============================================================

pub fn mish(ctx: EagerContext, input: Array) !Array {
    // mish(x) = x * tanh(softplus(x))
    const sp = try softplus(ctx, input);
    defer sp.deinit();
    const tanh_sp = try ops_mod.tanh(ctx, sp);
    defer tanh_sp.deinit();
    return ops_mod.multiply(ctx, input, tanh_sp);
}

// ============================================================
// CELU — via mlx-c ops
// ============================================================

pub fn celu(ctx: EagerContext, input: Array, alpha: f32) !Array {
    const zero = try ops_mod.scalarF32(ctx, 0.0);
    defer zero.deinit();
    const pos = try math_mod.maximum(ctx, input, zero);
    defer pos.deinit();
    const alpha_s = try ops_mod.scalarF32(ctx, alpha);
    defer alpha_s.deinit();
    const x_over_a = try ops_mod.divide(ctx, input, alpha_s);
    defer x_over_a.deinit();
    const exp_val = try ops_mod.exp(ctx, x_over_a);
    defer exp_val.deinit();
    const one = try ops_mod.scalarF32(ctx, 1.0);
    defer one.deinit();
    const exp_m1 = try ops_mod.subtract(ctx, exp_val, one);
    defer exp_m1.deinit();
    const neg_raw = try ops_mod.multiply(ctx, alpha_s, exp_m1);
    defer neg_raw.deinit();
    const neg = try math_mod.minimum(ctx, neg_raw, zero);
    defer neg.deinit();
    return ops_mod.add(ctx, pos, neg);
}

// ============================================================
// Tanh Shrink — via mlx-c ops
// ============================================================

pub fn tanhshrink(ctx: EagerContext, input: Array) !Array {
    const tanh_x = try ops_mod.tanh(ctx, input);
    defer tanh_x.deinit();
    return ops_mod.subtract(ctx, input, tanh_x);
}

// ============================================================
// ELU — via mlx-c ops
// ============================================================

pub fn elu(ctx: EagerContext, input: Array, alpha: f32) !Array {
    const zero = try ops_mod.scalarF32(ctx, 0.0);
    defer zero.deinit();
    const cond = try comparison_mod.greaterEqual(ctx, input, zero);
    defer cond.deinit();
    const exp_x = try ops_mod.exp(ctx, input);
    defer exp_x.deinit();
    const one = try ops_mod.scalarF32(ctx, 1.0);
    defer one.deinit();
    const exp_m1 = try ops_mod.subtract(ctx, exp_x, one);
    defer exp_m1.deinit();
    const alpha_s = try ops_mod.scalarF32(ctx, alpha);
    defer alpha_s.deinit();
    const neg_part = try ops_mod.multiply(ctx, alpha_s, exp_m1);
    defer neg_part.deinit();
    return ops_mod.where(ctx, cond, input, neg_part);
}

// ============================================================
// Hard Swish — via mlx-c ops
// ============================================================

pub fn hardswish(ctx: EagerContext, input: Array) !Array {
    const three = try ops_mod.scalarF32(ctx, 3.0);
    defer three.deinit();
    const six = try ops_mod.scalarF32(ctx, 6.0);
    defer six.deinit();
    const zero = try ops_mod.scalarF32(ctx, 0.0);
    defer zero.deinit();
    const x_plus_3 = try ops_mod.add(ctx, input, three);
    defer x_plus_3.deinit();
    const clamped = try math_mod.clip(ctx, x_plus_3, zero, six);
    defer clamped.deinit();
    const x_times_c = try ops_mod.multiply(ctx, input, clamped);
    defer x_times_c.deinit();
    return ops_mod.divide(ctx, x_times_c, six);
}

// ============================================================
// Hard Tanh — via mlx-c ops
// ============================================================

pub fn hardtanh(ctx: EagerContext, input: Array) !Array {
    const neg_one = try ops_mod.scalarF32(ctx, -1.0);
    defer neg_one.deinit();
    const one = try ops_mod.scalarF32(ctx, 1.0);
    defer one.deinit();
    return math_mod.clip(ctx, input, neg_one, one);
}

// ============================================================
// Log Sigmoid — via mlx-c ops
// ============================================================

pub fn logSigmoid(ctx: EagerContext, input: Array) !Array {
    // log(sigmoid(x)) = -softplus(-x)
    const neg_x = try ops_mod.negative(ctx, input);
    defer neg_x.deinit();
    const sp = try softplus(ctx, neg_x);
    defer sp.deinit();
    return ops_mod.negative(ctx, sp);
}

// ============================================================
// Log Softmax — via mlx-c ops
// ============================================================

pub fn logSoftmax(ctx: EagerContext, input: Array) !Array {
    const reduce_mod = @import("reduce.zig");
    const lse = try reduce_mod.logsumexpAxis(ctx, input, -1, true);
    defer lse.deinit();
    return ops_mod.subtract(ctx, input, lse);
}

// ============================================================
// Softmin — via mlx-c ops
// ============================================================

pub fn softmin(ctx: EagerContext, input: Array) !Array {
    const neg_x = try ops_mod.negative(ctx, input);
    defer neg_x.deinit();
    return ops_mod.softmax(ctx, neg_x, &[_]i32{-1});
}

// ============================================================
// SELU — via mlx-c ops
// ============================================================

pub fn selu(ctx: EagerContext, input: Array) !Array {
    const alpha_val: f32 = 1.6732632423543772848170429916717;
    const scale_val: f32 = 1.0507009873554804934193349852946;
    const elu_result = try elu(ctx, input, alpha_val);
    defer elu_result.deinit();
    const scale_s = try ops_mod.scalarF32(ctx, scale_val);
    defer scale_s.deinit();
    return ops_mod.multiply(ctx, scale_s, elu_result);
}

// ============================================================
// Hard Shrink — via mlx-c ops
// ============================================================

pub fn hardshrink(ctx: EagerContext, input: Array, lambd: f32) !Array {
    const zero = try ops_mod.scalarF32(ctx, 0.0);
    defer zero.deinit();
    const abs_x = try ops_mod.abs(ctx, input);
    defer abs_x.deinit();
    const threshold = try ops_mod.scalarF32(ctx, lambd);
    defer threshold.deinit();
    const cond = try comparison_mod.greater(ctx, abs_x, threshold);
    defer cond.deinit();
    return ops_mod.where(ctx, cond, input, zero);
}

// ============================================================
// Soft Shrink — via mlx-c ops
// ============================================================

pub fn softshrink(ctx: EagerContext, input: Array, lambd: f32) !Array {
    const zero = try ops_mod.scalarF32(ctx, 0.0);
    defer zero.deinit();
    const l = try ops_mod.scalarF32(ctx, lambd);
    defer l.deinit();
    const neg_l = try ops_mod.scalarF32(ctx, -lambd);
    defer neg_l.deinit();
    const cond_pos = try comparison_mod.greater(ctx, input, l);
    defer cond_pos.deinit();
    const cond_neg = try comparison_mod.less(ctx, input, neg_l);
    defer cond_neg.deinit();
    const pos_val = try ops_mod.subtract(ctx, input, l);
    defer pos_val.deinit();
    const neg_val = try ops_mod.add(ctx, input, l);
    defer neg_val.deinit();
    const partial = try ops_mod.where(ctx, cond_neg, neg_val, zero);
    defer partial.deinit();
    return ops_mod.where(ctx, cond_pos, pos_val, partial);
}

// ============================================================
// Step — via mlx-c ops
// ============================================================

pub fn step(ctx: EagerContext, input: Array) !Array {
    const zero = try ops_mod.scalarF32(ctx, 0.0);
    defer zero.deinit();
    const one = try ops_mod.scalarF32(ctx, 1.0);
    defer one.deinit();
    const cond = try comparison_mod.greaterEqual(ctx, input, zero);
    defer cond.deinit();
    return ops_mod.where(ctx, cond, one, zero);
}

// ============================================================
// ReLU6 — via mlx-c ops
// ============================================================

pub fn relu6(ctx: EagerContext, input: Array) !Array {
    const zero = try ops_mod.scalarF32(ctx, 0.0);
    defer zero.deinit();
    const six = try ops_mod.scalarF32(ctx, 6.0);
    defer six.deinit();
    return math_mod.clip(ctx, input, zero, six);
}

// ============================================================
// GLU (Gated Linear Unit) — via mlx-c ops
// ============================================================

pub fn glu(ctx: EagerContext, input: Array) !Array {
    const shape = input.shape();
    const ndim = shape.len;
    const last_dim: i32 = shape[ndim - 1];
    const half_dim = @divExact(last_dim, 2);

    var start_a = try ctx.allocator.alloc(i32, ndim);
    defer ctx.allocator.free(start_a);
    var stop_a = try ctx.allocator.alloc(i32, ndim);
    defer ctx.allocator.free(stop_a);
    var start_b = try ctx.allocator.alloc(i32, ndim);
    defer ctx.allocator.free(start_b);
    var stop_b = try ctx.allocator.alloc(i32, ndim);
    defer ctx.allocator.free(stop_b);

    for (0..ndim) |i| {
        start_a[i] = 0;
        stop_a[i] = shape[i];
        start_b[i] = 0;
        stop_b[i] = shape[i];
    }
    stop_a[ndim - 1] = half_dim;
    start_b[ndim - 1] = half_dim;

    const val_half = try ops_mod.slice(ctx, input, start_a, stop_a, &[_]i32{});
    defer val_half.deinit();
    const gate_half = try ops_mod.slice(ctx, input, start_b, stop_b, &[_]i32{});
    defer gate_half.deinit();
    const gate_sig = try ops_mod.sigmoid(ctx, gate_half);
    defer gate_sig.deinit();
    return ops_mod.multiply(ctx, gate_sig, val_half);
}
