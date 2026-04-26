/// Fused (compiled) composite operations via mlx_compile.
///
/// Uses `src/compile.zig` to compile multi-step operator graphs into
/// single fused GPU kernel launches, reducing intermediate Array
/// allocations and kernel launch overhead.
///
/// - `compiledSwiGLU`: gate_proj + silu + up_proj + down_proj as one fused op
/// - `compiledAdamWStep`: AdamW optimizer step (~15 intermediates) fused
///
/// Requirements: R8.1, R8.2, R8.3
const std = @import("std");
const c = @import("../c.zig");
const ops_mod = @import("../ops.zig");
const array_mod = @import("../array.zig");
const compile_mod = @import("../compile.zig");
const closure_mod = @import("../closure.zig");
const activations = @import("activations.zig");

const Array = array_mod.Array;
const EagerContext = ops_mod.EagerContext;
const Closure = closure_mod.Closure;

// ============================================================
// SwiGLU MLP — unfused reference + compiled fused version
// ============================================================

/// Unfused SwiGLU MLP forward pass.
///
/// Computes: down_proj @ (silu(gate_proj @ x) * (up_proj @ x))
///
/// Inputs (as closure args):
///   [0] x           — input tensor [batch, seq_len, hidden_dim]
///   [1] gate_weight — gate projection weight [intermediate_dim, hidden_dim]
///   [2] up_weight   — up projection weight   [intermediate_dim, hidden_dim]
///   [3] down_weight — down projection weight  [hidden_dim, intermediate_dim]
///
/// Returns: [output] — [batch, seq_len, hidden_dim]
pub fn swigluForward(inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array {
    if (inputs.len < 4) return error.MlxError;

    const x = inputs[0];
    const gate_weight = inputs[1];
    const up_weight = inputs[2];
    const down_weight = inputs[3];

    const stream = c.c.mlx_default_cpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    // gate = x @ gate_weight^T
    const gate_weight_t = ops_mod.transpose(ctx, gate_weight) catch return error.MlxError;
    defer gate_weight_t.deinit();
    const gate = ops_mod.matmul(ctx, x, gate_weight_t) catch return error.MlxError;
    defer gate.deinit();

    // gate_activated = silu(gate)
    const gate_activated = activations.silu(ctx, gate) catch return error.MlxError;
    defer gate_activated.deinit();

    // up = x @ up_weight^T
    const up_weight_t = ops_mod.transpose(ctx, up_weight) catch return error.MlxError;
    defer up_weight_t.deinit();
    const up = ops_mod.matmul(ctx, x, up_weight_t) catch return error.MlxError;
    defer up.deinit();

    // combined = gate_activated * up
    const combined = ops_mod.multiply(ctx, gate_activated, up) catch return error.MlxError;
    defer combined.deinit();

    // output = combined @ down_weight^T
    const down_weight_t = ops_mod.transpose(ctx, down_weight) catch return error.MlxError;
    defer down_weight_t.deinit();
    const output = ops_mod.matmul(ctx, combined, down_weight_t) catch return error.MlxError;

    const result = allocator.alloc(Array, 1) catch return error.MlxError;
    result[0] = output;
    return result;
}

/// Compile the SwiGLU MLP into a fused operation via mlx_compile.
///
/// Returns a Closure that can be called with [x, gate_weight, up_weight, down_weight].
/// The compiled closure fuses all intermediate operations into fewer kernel launches.
pub fn compiledSwiGLU(allocator: std.mem.Allocator) !Closure {
    const base_closure = try Closure.init(&swigluForward, allocator);
    defer base_closure.deinit();
    return compile_mod.compile(base_closure, false);
}

/// Execute the unfused SwiGLU MLP (for comparison / fallback).
pub fn unfusedSwiGLU(ctx: EagerContext, x: Array, gate_weight: Array, up_weight: Array, down_weight: Array) !Array {
    // gate = x @ gate_weight^T
    const gate_weight_t = try ops_mod.transpose(ctx, gate_weight);
    defer gate_weight_t.deinit();
    const gate = try ops_mod.matmul(ctx, x, gate_weight_t);
    defer gate.deinit();

    // gate_activated = silu(gate)
    const gate_activated = try activations.silu(ctx, gate);
    defer gate_activated.deinit();

    // up = x @ up_weight^T
    const up_weight_t = try ops_mod.transpose(ctx, up_weight);
    defer up_weight_t.deinit();
    const up = try ops_mod.matmul(ctx, x, up_weight_t);
    defer up.deinit();

    // combined = gate_activated * up
    const combined = try ops_mod.multiply(ctx, gate_activated, up);
    defer combined.deinit();

    // output = combined @ down_weight^T
    const down_weight_t = try ops_mod.transpose(ctx, down_weight);
    defer down_weight_t.deinit();
    return ops_mod.matmul(ctx, combined, down_weight_t);
}

// ============================================================
// AdamW Step — unfused reference + compiled fused version
// ============================================================

/// Unfused AdamW optimizer step for a single parameter.
///
/// Inputs (as closure args):
///   [0] param       — current parameter
///   [1] grad        — gradient for this parameter
///   [2] m           — first moment estimate
///   [3] v           — second moment estimate
///   [4] lr          — learning rate (scalar)
///   [5] beta1       — first moment decay (scalar)
///   [6] beta2       — second moment decay (scalar)
///   [7] eps         — epsilon for numerical stability (scalar)
///   [8] weight_decay — weight decay coefficient (scalar)
///   [9] bias_corr1  — bias correction factor 1 (scalar)
///   [10] bias_corr2 — bias correction factor 2 (scalar)
///
/// Returns: [new_param, new_m, new_v]
pub fn adamwStepForward(inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array {
    if (inputs.len < 11) return error.MlxError;

    const param = inputs[0];
    const grad = inputs[1];
    const m = inputs[2];
    const v = inputs[3];
    const lr = inputs[4];
    const beta1 = inputs[5];
    const beta2 = inputs[6];
    const eps = inputs[7];
    const wd = inputs[8];
    const bias_corr1 = inputs[9];
    const bias_corr2 = inputs[10];

    const stream = c.c.mlx_default_cpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    // one_minus_beta1 = 1 - beta1
    const one = ops_mod.scalarF32(ctx, 1.0) catch return error.MlxError;
    defer one.deinit();
    const one_minus_beta1 = ops_mod.subtract(ctx, one, beta1) catch return error.MlxError;
    defer one_minus_beta1.deinit();

    // one_minus_beta2 = 1 - beta2
    const one_minus_beta2 = ops_mod.subtract(ctx, one, beta2) catch return error.MlxError;
    defer one_minus_beta2.deinit();

    // new_m = beta1 * m + (1 - beta1) * grad
    const m_scaled = ops_mod.multiply(ctx, beta1, m) catch return error.MlxError;
    defer m_scaled.deinit();
    const g_scaled = ops_mod.multiply(ctx, one_minus_beta1, grad) catch return error.MlxError;
    defer g_scaled.deinit();
    const new_m = ops_mod.add(ctx, m_scaled, g_scaled) catch return error.MlxError;

    // new_v = beta2 * v + (1 - beta2) * grad^2
    const grad_sq = ops_mod.multiply(ctx, grad, grad) catch return error.MlxError;
    defer grad_sq.deinit();
    const v_scaled = ops_mod.multiply(ctx, beta2, v) catch return error.MlxError;
    defer v_scaled.deinit();
    const g2_scaled = ops_mod.multiply(ctx, one_minus_beta2, grad_sq) catch return error.MlxError;
    defer g2_scaled.deinit();
    const new_v = ops_mod.add(ctx, v_scaled, g2_scaled) catch return error.MlxError;

    // m_hat = new_m / bias_corr1
    const m_hat = ops_mod.divide(ctx, new_m, bias_corr1) catch return error.MlxError;
    defer m_hat.deinit();

    // v_hat = new_v / bias_corr2
    const v_hat = ops_mod.divide(ctx, new_v, bias_corr2) catch return error.MlxError;
    defer v_hat.deinit();

    // denom = sqrt(v_hat) + eps
    const sqrt_v = ops_mod.sqrt(ctx, v_hat) catch return error.MlxError;
    defer sqrt_v.deinit();
    const denom = ops_mod.add(ctx, sqrt_v, eps) catch return error.MlxError;
    defer denom.deinit();

    // step = lr * m_hat / denom
    const step_val = ops_mod.divide(ctx, m_hat, denom) catch return error.MlxError;
    defer step_val.deinit();
    const scaled_step = ops_mod.multiply(ctx, lr, step_val) catch return error.MlxError;
    defer scaled_step.deinit();

    // wd_term = lr * wd * param
    const lr_wd = ops_mod.multiply(ctx, lr, wd) catch return error.MlxError;
    defer lr_wd.deinit();
    const wd_term = ops_mod.multiply(ctx, lr_wd, param) catch return error.MlxError;
    defer wd_term.deinit();

    // new_param = param - scaled_step - wd_term
    const tmp = ops_mod.subtract(ctx, param, scaled_step) catch return error.MlxError;
    defer tmp.deinit();
    const new_param = ops_mod.subtract(ctx, tmp, wd_term) catch return error.MlxError;

    const result = allocator.alloc(Array, 3) catch return error.MlxError;
    result[0] = new_param;
    result[1] = new_m;
    result[2] = new_v;
    return result;
}

/// Compile the AdamW optimizer step into a fused operation via mlx_compile.
///
/// Returns a Closure that can be called with
/// [param, grad, m, v, lr, beta1, beta2, eps, weight_decay, bias_corr1, bias_corr2].
/// Returns [new_param, new_m, new_v].
pub fn compiledAdamWStep(allocator: std.mem.Allocator) !Closure {
    const base_closure = try Closure.init(&adamwStepForward, allocator);
    defer base_closure.deinit();
    return compile_mod.compile(base_closure, false);
}

/// Execute the unfused AdamW step for a single parameter (for comparison / fallback).
///
/// Returns: .{ new_param, new_m, new_v }
pub fn unfusedAdamWStep(
    ctx: EagerContext,
    param: Array,
    grad: Array,
    m: Array,
    v: Array,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps_val: f32,
    weight_decay: f32,
    bias_correction1: f32,
    bias_correction2: f32,
) !struct { param: Array, m: Array, v: Array } {
    // new_m = beta1 * m + (1 - beta1) * grad
    const sc_beta1 = try ops_mod.scalarF32(ctx, beta1);
    defer sc_beta1.deinit();
    const sc_omb1 = try ops_mod.scalarF32(ctx, 1.0 - beta1);
    defer sc_omb1.deinit();
    const m_scaled = try ops_mod.multiply(ctx, sc_beta1, m);
    defer m_scaled.deinit();
    const g_scaled = try ops_mod.multiply(ctx, sc_omb1, grad);
    defer g_scaled.deinit();
    const new_m = try ops_mod.add(ctx, m_scaled, g_scaled);

    // new_v = beta2 * v + (1 - beta2) * grad^2
    const sc_beta2 = try ops_mod.scalarF32(ctx, beta2);
    defer sc_beta2.deinit();
    const sc_omb2 = try ops_mod.scalarF32(ctx, 1.0 - beta2);
    defer sc_omb2.deinit();
    const grad_sq = try ops_mod.multiply(ctx, grad, grad);
    defer grad_sq.deinit();
    const v_scaled = try ops_mod.multiply(ctx, sc_beta2, v);
    defer v_scaled.deinit();
    const g2_scaled = try ops_mod.multiply(ctx, sc_omb2, grad_sq);
    defer g2_scaled.deinit();
    const new_v = try ops_mod.add(ctx, v_scaled, g2_scaled);

    // m_hat = new_m / bias_correction1
    const sc_bc1 = try ops_mod.scalarF32(ctx, bias_correction1);
    defer sc_bc1.deinit();
    const m_hat = try ops_mod.divide(ctx, new_m, sc_bc1);
    defer m_hat.deinit();

    // v_hat = new_v / bias_correction2
    const sc_bc2 = try ops_mod.scalarF32(ctx, bias_correction2);
    defer sc_bc2.deinit();
    const v_hat = try ops_mod.divide(ctx, new_v, sc_bc2);
    defer v_hat.deinit();

    // denom = sqrt(v_hat) + eps
    const sqrt_v = try ops_mod.sqrt(ctx, v_hat);
    defer sqrt_v.deinit();
    const sc_eps = try ops_mod.scalarF32(ctx, eps_val);
    defer sc_eps.deinit();
    const denom = try ops_mod.add(ctx, sqrt_v, sc_eps);
    defer denom.deinit();

    // step = lr * m_hat / denom
    const step_val = try ops_mod.divide(ctx, m_hat, denom);
    defer step_val.deinit();
    const sc_lr = try ops_mod.scalarF32(ctx, lr);
    defer sc_lr.deinit();
    const scaled_step = try ops_mod.multiply(ctx, sc_lr, step_val);
    defer scaled_step.deinit();

    // wd_term = lr * weight_decay * param
    const sc_lr_wd = try ops_mod.scalarF32(ctx, lr * weight_decay);
    defer sc_lr_wd.deinit();
    const wd_term = try ops_mod.multiply(ctx, sc_lr_wd, param);
    defer wd_term.deinit();

    // new_param = param - scaled_step - wd_term
    const tmp = try ops_mod.subtract(ctx, param, scaled_step);
    defer tmp.deinit();
    const new_param = try ops_mod.subtract(ctx, tmp, wd_term);

    return .{ .param = new_param, .m = new_m, .v = new_v };
}
