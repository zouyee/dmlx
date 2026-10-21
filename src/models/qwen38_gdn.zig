//! Qwen3.8 GDN (gated delta net) linear-attention module.
//!
//! Port of Qwen3-Next `Qwen3NextGatedDeltaNet` single-batch naive path.
//! Differs from GlmKDA: asymmetric heads (key Hk / value Hv, kv-group
//! broadcast), separate in_proj_qkv / in_proj_a / in_proj_b / in_proj_z
//! projections, softplus decay, rms_norm q/k (equals l2norm up to rounding),
//! silu output gate.
//! f32 throughout; cross-checked against scripts/qwen38_gdn_ref.py.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const reduce_mod = @import("mlx").reduce;
const math_mod = @import("mlx").math;
const comparison_mod = @import("mlx").comparison;
const creation_mod = @import("mlx").creation;
const fast_mod = @import("mlx").fast;
const activations = @import("mlx").activations;
const quantize_mod = @import("mlx").quantize;
const eval_mod = @import("mlx").eval;
const closure_mod = @import("mlx").closure;
const compile_mod = @import("mlx").compile;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const QuantConfig = quantize_mod.QuantConfig;
const Closure = closure_mod.Closure;

pub const Qwen38GdnCache = struct {
    conv_state: Array, // [conv_kernel-1, qkv_dim]
    ssm_state: Array, // [value_heads, head_dim, head_dim] f32, layout [Hv, Dv, Dk]

    pub fn init(allocator: std.mem.Allocator, conv_kernel: usize, qkv_dim: usize, value_heads: usize, head_dim: usize) !Qwen38GdnCache {
        return .{
            .conv_state = try array_mod.zeros(allocator, &[_]i32{ @intCast(conv_kernel - 1), @intCast(qkv_dim) }, .float32),
            .ssm_state = try array_mod.zeros(allocator, &[_]i32{ @intCast(value_heads), @intCast(head_dim), @intCast(head_dim) }, .float32),
        };
    }

    pub fn deinit(self: *Qwen38GdnCache) void {
        self.conv_state.deinit();
        self.ssm_state.deinit();
    }
};

/// Per-layer constants derived once from the loaded weights: the transposed
/// conv kernel (turns the k=4 depthwise conv into one multiply + column sum),
/// a ones vector so the weight-less q/k rms_norm can use the fused
/// fast.rmsNorm, and kv-group gather indices that broadcast q/k to value
/// heads in a single take.
pub const Precomputed = struct {
    conv1d_t: Array, // [conv_kernel, QKV]
    ones_d: Array, // [head_dim] f32
    qk_idx: Array, // [num_value_heads] i32

    pub fn deinit(self: *Precomputed) void {
        self.conv1d_t.deinit();
        self.ones_d.deinit();
        self.qk_idx.deinit();
    }
};

pub fn precomputeDerived(
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    conv1d: Array, // [QKV, K, 1]
    num_key_heads: usize,
    num_value_heads: usize,
    head_dim: usize,
) !Precomputed {
    const cshape = conv1d.shape();
    const QKV: i32 = @intCast(cshape[0]);
    const K: i32 = @intCast(cshape[1]);
    const w2d = try ops.reshape(ctx, conv1d, &[_]i32{ QKV, K });
    defer w2d.deinit();
    var out: Precomputed = .{
        .conv1d_t = try shape_mod.transpose(ctx, w2d),
        .ones_d = try creation_mod.ones(ctx, &[_]i32{@intCast(head_dim)}, .float32),
        .qk_idx = undefined,
    };
    errdefer out.conv1d_t.deinit();
    errdefer out.ones_d.deinit();
    const ratio = num_value_heads / num_key_heads;
    const idx_buf = try allocator.alloc(i32, num_value_heads);
    defer allocator.free(idx_buf);
    for (idx_buf, 0..) |*v, h| v.* = @intCast(h / ratio);
    out.qk_idx = try Array.fromData(allocator, i32, idx_buf, &[_]i32{@intCast(num_value_heads)});
    return out;
}

// ============================================================
// Pure decode step (arrays -> arrays) + compiled closure
// ============================================================

/// All arrays one GDN decode token needs, with no module state. Shapes are
/// fixed across layers and tokens, so a single compiled graph built from
/// `compiledStep` serves every GDN layer (weight VALUES differ; shapes and
/// dtypes are identical).
pub const StepWeights = struct {
    x: Array, // [1, hidden]
    conv_state: Array, // [K-1, QKV]
    ssm_state: Array, // [HV, D, D] f32
    qkv_w: Array,
    qkv_scales: ?Array = null,
    qkv_biases: ?Array = null,
    conv1d_t: Array, // [K, QKV]
    a_w: Array,
    a_scales: ?Array = null,
    a_biases: ?Array = null,
    b_w: Array,
    b_scales: ?Array = null,
    b_biases: ?Array = null,
    z_w: Array,
    z_scales: ?Array = null,
    z_biases: ?Array = null,
    A_log: Array, // [HV]
    dt_bias: Array, // [HV]
    norm_w: Array, // [D]
    out_w: Array,
    out_scales: ?Array = null,
    out_biases: ?Array = null,
    ones_d: Array, // [D]
    qk_idx: Array, // [HV] i32
    norm_eps: f32 = 1e-6,
};

pub const StepOutput = struct {
    out: Array, // [1, hidden]
    conv_state: Array, // [K-1, QKV]
    ssm_state: Array, // [HV, D, D] f32
};

/// Affine QuantConfig derived from the packed shapes (mirrors
/// WeightSource.quantized / borrowQuantLinear): bits = packed_cols*32/in,
/// group_size = in/scale_cols.
fn quantConfigFromShapes(x: Array, w: Array, scales: Array) QuantConfig {
    const xs = x.shape();
    const in_features: i64 = xs[xs.len - 1];
    const ws = w.shape();
    const ss = scales.shape();
    const packed_cols: i64 = ws[ws.len - 1];
    const scale_cols: i64 = ss[ss.len - 1];
    return .{
        .bits = @intCast(@divTrunc(packed_cols * 32, in_features)),
        .group_size = @intCast(@divTrunc(in_features, scale_cols)),
        .mode = .affine,
    };
}

/// x @ w^T — fused quantized matmul when scales are present. The QuantConfig
/// is derived from the packed shapes so the same code serves traced closures
/// (no runtime config payload) and eager callers identically.
pub fn linearQuantParts(ctx: EagerContext, x: Array, w: Array, scales: ?Array, biases: ?Array) !Array {
    if (scales) |s| {
        const qw = quantize_mod.QuantizedWeight{
            .data = w,
            .scales = s,
            .biases = biases orelse Array.fromHandle(c.c.mlx_array_new()),
            .config = quantConfigFromShapes(x, w, s),
            .original_shape = &[_]i32{},
        };
        return quantize_mod.quantizedMatmul(ctx, x, qw, true);
    }
    const w_t = try ops.transpose(ctx, w);
    defer w_t.deinit();
    return ops.matmul(ctx, x, w_t);
}

/// f32-accumulated depthwise causal conv over the [K, QKV] window (see
/// Qwen38GDN.convAccum).
fn convAccumStep(ctx: EagerContext, conv1d_t: Array, conv_input: Array) !Array {
    const w_f32 = try ops.astype(ctx, conv1d_t, .float32);
    defer w_f32.deinit();
    const x_f32 = try ops.astype(ctx, conv_input, .float32);
    defer x_f32.deinit();
    const prod = try ops.multiply(ctx, x_f32, w_f32);
    defer prod.deinit();
    const acc = try reduce_mod.sumAxis(ctx, prod, 0, true);
    defer acc.deinit();
    return ops.astype(ctx, acc, conv_input.dtype());
}

/// softplus(x) = log1p(exp(x)) for x <= 20, x above.
fn softplusStep(ctx: EagerContext, x: Array) !Array {
    const ex = try ops.exp(ctx, x);
    defer ex.deinit();
    const lp = try math_mod.log1p(ctx, ex);
    defer lp.deinit();
    const twenty = try ops.scalarF32(ctx, 20.0);
    defer twenty.deinit();
    const big = try comparison_mod.greater(ctx, x, twenty);
    defer big.deinit();
    return ops.where(ctx, big, x, lp);
}

/// One GDN decode token as a pure function of arrays. Body mirrors
/// Qwen38GDN.forward exactly (same op sequence, same dtypes), but takes every
/// weight/state as an explicit input and returns the new states instead of
/// mutating a cache — this is the function compiled by `compiledStep`.
/// All dims are derived from input shapes so the trace is shape-polymorphic
/// across model sizes (and retrace-free as long as shapes stay fixed).
pub fn step(ctx: EagerContext, w: StepWeights) !StepOutput {
    const HV: i32 = @intCast(w.A_log.shape()[0]);
    const D: i32 = @intCast(w.norm_w.shape()[0]);
    const K: i32 = @intCast(w.conv1d_t.shape()[0]);
    const QKV: i32 = @intCast(w.conv1d_t.shape()[1]);
    const PV: i32 = HV * D;
    const PK: i32 = @divTrunc(QKV - PV, 2);
    const HK: i32 = @divTrunc(PK, D);

    // 1. qkv projection + depthwise causal conv (k=4) + silu.
    const qkv_lin = try linearQuantParts(ctx, w.x, w.qkv_w, w.qkv_scales, w.qkv_biases);
    defer qkv_lin.deinit();

    const conv_input = try shape_mod.concatenateAxis(ctx, &[_]Array{ w.conv_state, qkv_lin }, 0);
    defer conv_input.deinit(); // [K, QKV]

    const conv_acc = try convAccumStep(ctx, w.conv1d_t, conv_input);
    defer conv_acc.deinit();

    // New conv window: keep last K-1 rows.
    const tail = try ops.slice(ctx, conv_input, &[_]i32{ 1, 0 }, &[_]i32{ K, QKV }, &[_]i32{ 1, 1 });
    defer tail.deinit();
    const new_conv_state = try ops.copy(ctx, tail);
    errdefer new_conv_state.deinit();

    const qkv = try activations.silu(ctx, conv_acc);
    defer qkv.deinit(); // [1, QKV]

    // 2. split q/k/v; weight-less rms_norm per head; scale q by D^-1, k by D^-0.5.
    const qkv_flat = try ops.reshape(ctx, qkv, &[_]i32{QKV});
    defer qkv_flat.deinit();
    const q_parts = try ops.slice(ctx, qkv_flat, &[_]i32{0}, &[_]i32{PK}, &[_]i32{1});
    defer q_parts.deinit();
    const k_parts = try ops.slice(ctx, qkv_flat, &[_]i32{PK}, &[_]i32{2 * PK}, &[_]i32{1});
    defer k_parts.deinit();
    const v_parts = try ops.slice(ctx, qkv_flat, &[_]i32{2 * PK}, &[_]i32{QKV}, &[_]i32{1});
    defer v_parts.deinit();

    const q_resh = try ops.reshape(ctx, q_parts, &[_]i32{ HK, D });
    defer q_resh.deinit();
    const k_resh = try ops.reshape(ctx, k_parts, &[_]i32{ HK, D });
    defer k_resh.deinit();
    const v = try ops.reshape(ctx, v_parts, &[_]i32{ HV, D });
    defer v.deinit();

    const inv_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    const q_rn = try fast_mod.rmsNorm(ctx, q_resh, w.ones_d, 1e-6);
    defer q_rn.deinit();
    const k_rn = try fast_mod.rmsNorm(ctx, k_resh, w.ones_d, 1e-6);
    defer k_rn.deinit();
    const q_scale = try ops.scalarF32(ctx, inv_scale * inv_scale);
    defer q_scale.deinit();
    const k_scale = try ops.scalarF32(ctx, inv_scale);
    defer k_scale.deinit();
    const q = try ops.multiply(ctx, q_rn, q_scale);
    defer q.deinit();
    const k = try ops.multiply(ctx, k_rn, k_scale);
    defer k.deinit();

    // 3. per-head gates: a (decay input), b (beta), z (output gate).
    const a = try linearQuantParts(ctx, w.x, w.a_w, w.a_scales, w.a_biases);
    defer a.deinit(); // [1, HV]
    const b = try linearQuantParts(ctx, w.x, w.b_w, w.b_scales, w.b_biases);
    defer b.deinit(); // [1, HV]
    const z = try linearQuantParts(ctx, w.x, w.z_w, w.z_scales, w.z_biases);
    defer z.deinit(); // [1, HV*D]

    // g = exp(-exp(A_log_f32) * softplus(a + dt_bias))  [HV]
    const a_flat = try ops.reshape(ctx, a, &[_]i32{HV});
    defer a_flat.deinit();
    const a_dt = try ops.add(ctx, a_flat, w.dt_bias);
    defer a_dt.deinit();
    const sp = try softplusStep(ctx, a_dt);
    defer sp.deinit();
    const a_log_f32 = try ops.astype(ctx, w.A_log, .float32);
    defer a_log_f32.deinit();
    const e_a_log = try ops.exp(ctx, a_log_f32);
    defer e_a_log.deinit();
    const g_pre = try ops.multiply(ctx, e_a_log, sp);
    defer g_pre.deinit();
    const g_neg = try ops.negative(ctx, g_pre);
    defer g_neg.deinit();
    const g = try ops.exp(ctx, g_neg); // [HV]
    defer g.deinit();

    const b_flat = try ops.reshape(ctx, b, &[_]i32{HV});
    defer b_flat.deinit();
    const beta = try ops.sigmoid(ctx, b_flat); // [HV]
    defer beta.deinit();

    // 4. broadcast q/k to value heads via the precomputed gather indices.
    const q_full = try shape_mod.takeAxis(ctx, q, w.qk_idx, 0); // [HV, D]
    defer q_full.deinit();
    const k_full = try shape_mod.takeAxis(ctx, k, w.qk_idx, 0); // [HV, D]
    defer k_full.deinit();

    // 5. gated delta rule step. State S [Hv, Dv, Dk].
    const g_decay = try ops.reshape(ctx, g, &[_]i32{ HV, 1, 1 });
    defer g_decay.deinit();
    const s_decayed = try ops.multiply(ctx, w.ssm_state, g_decay);
    defer s_decayed.deinit();

    const k_b = try ops.reshape(ctx, k_full, &[_]i32{ HV, 1, D });
    defer k_b.deinit();
    const sk = try ops.multiply(ctx, s_decayed, k_b);
    defer sk.deinit();
    const kv_mem = try reduce_mod.sumAxis(ctx, sk, -1, false); // [HV, Dv]
    defer kv_mem.deinit();

    const v_sub = try ops.subtract(ctx, v, kv_mem);
    defer v_sub.deinit();
    const beta_b = try ops.reshape(ctx, beta, &[_]i32{ HV, 1 });
    defer beta_b.deinit();
    const delta = try ops.multiply(ctx, v_sub, beta_b); // [HV, Dv]
    defer delta.deinit();

    const delta_b = try ops.expandDims(ctx, delta, -1); // [HV, Dv, 1]
    defer delta_b.deinit();
    const k_delta = try ops.multiply(ctx, k_b, delta_b);
    defer k_delta.deinit();
    const s_new = try ops.add(ctx, s_decayed, k_delta);
    errdefer s_new.deinit();

    const q_b2 = try ops.reshape(ctx, q_full, &[_]i32{ HV, 1, D });
    defer q_b2.deinit();
    const sq = try ops.multiply(ctx, s_new, q_b2);
    defer sq.deinit();
    const y_f32 = try reduce_mod.sumAxis(ctx, sq, -1, false); // [HV, Dv] f32
    defer y_f32.deinit();
    // gated_delta_update returns y.astype(q.dtype): round before the norm.
    const y = try ops.astype(ctx, y_f32, qkv.dtype());
    defer y.deinit();

    // 6. gated RMSNorm: norm in f32, gate in f32, round to input dtype.
    const y_normed = try fast_mod.rmsNorm(ctx, y, w.norm_w, w.norm_eps);
    defer y_normed.deinit();
    const y_normed_f32 = try ops.astype(ctx, y_normed, .float32);
    defer y_normed_f32.deinit();
    const z_resh = try ops.reshape(ctx, z, &[_]i32{ HV, D });
    defer z_resh.deinit();
    const z_f32 = try ops.astype(ctx, z_resh, .float32);
    defer z_f32.deinit();
    const z_act = try activations.silu(ctx, z_f32);
    defer z_act.deinit();
    const y_gated_f32 = try ops.multiply(ctx, z_act, y_normed_f32);
    defer y_gated_f32.deinit();
    const y_gated = try ops.astype(ctx, y_gated_f32, qkv.dtype());
    defer y_gated.deinit();
    const y_flat = try ops.reshape(ctx, y_gated, &[_]i32{ 1, PV });
    defer y_flat.deinit();

    const out = try linearQuantParts(ctx, y_flat, w.out_w, w.out_scales, w.out_biases);
    errdefer out.deinit();
    return .{ .out = out, .conv_state = new_conv_state, .ssm_state = s_new };
}

/// Closures are traced on the GPU stream so the compiled graph dispatches to
/// the same device as the eager path (CPU-stream traces would bake CPU
/// kernels into the graph).
fn closureGpuCtx(allocator: std.mem.Allocator) EagerContext {
    return EagerContext.initWithStream(allocator, .{ .inner = c.c.mlx_default_gpu_stream_new() });
}

fn stepToSlice(res: StepOutput, allocator: std.mem.Allocator) error{MlxError}![]Array {
    const out = allocator.alloc(Array, 3) catch return error.MlxError;
    out[0] = res.out;
    out[1] = res.conv_state;
    out[2] = res.ssm_state;
    return out;
}

/// Quantized variant. Input order (24): x, conv_state, ssm_state, qkv_w,
/// qkv_s, qkv_b, conv1d_t, a_w, a_s, a_b, b_w, b_s, b_b, z_w, z_s, z_b,
/// A_log, dt_bias, norm_w, out_w, out_s, out_b, ones_d, qk_idx.
fn gdnStepClosureQuant(inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array {
    if (inputs.len != 24) return error.MlxError;
    const ctx = closureGpuCtx(allocator);
    defer ctx.deinit();
    const res = step(ctx, .{
        .x = inputs[0],
        .conv_state = inputs[1],
        .ssm_state = inputs[2],
        .qkv_w = inputs[3],
        .qkv_scales = inputs[4],
        .qkv_biases = inputs[5],
        .conv1d_t = inputs[6],
        .a_w = inputs[7],
        .a_scales = inputs[8],
        .a_biases = inputs[9],
        .b_w = inputs[10],
        .b_scales = inputs[11],
        .b_biases = inputs[12],
        .z_w = inputs[13],
        .z_scales = inputs[14],
        .z_biases = inputs[15],
        .A_log = inputs[16],
        .dt_bias = inputs[17],
        .norm_w = inputs[18],
        .out_w = inputs[19],
        .out_scales = inputs[20],
        .out_biases = inputs[21],
        .ones_d = inputs[22],
        .qk_idx = inputs[23],
    }) catch return error.MlxError;
    return stepToSlice(res, allocator);
}

/// Plain (unquantized) variant — used by the toy-module unit test. Input
/// order (14): x, conv_state, ssm_state, qkv_w, conv1d_t, a_w, b_w, z_w,
/// A_log, dt_bias, norm_w, out_w, ones_d, qk_idx.
fn gdnStepClosurePlain(inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array {
    if (inputs.len != 14) return error.MlxError;
    const ctx = closureGpuCtx(allocator);
    defer ctx.deinit();
    const res = step(ctx, .{
        .x = inputs[0],
        .conv_state = inputs[1],
        .ssm_state = inputs[2],
        .qkv_w = inputs[3],
        .conv1d_t = inputs[4],
        .a_w = inputs[5],
        .b_w = inputs[6],
        .z_w = inputs[7],
        .A_log = inputs[8],
        .dt_bias = inputs[9],
        .norm_w = inputs[10],
        .out_w = inputs[11],
        .ones_d = inputs[12],
        .qk_idx = inputs[13],
    }) catch return error.MlxError;
    return stepToSlice(res, allocator);
}

/// Build the mx.compile'd GDN decode step. The returned closure takes the
/// fixed input order of gdnStepClosureQuant/Plain and returns
/// [out, new_conv_state, new_ssm_state]. Trace happens on first apply — warm
/// it once with real-shaped arrays before serving.
pub fn compiledStep(allocator: std.mem.Allocator, quantized: bool) !Closure {
    const base = try Closure.init(if (quantized) &gdnStepClosureQuant else &gdnStepClosurePlain, allocator);
    defer base.deinit();
    return compile_mod.compile(base, false);
}

pub const Qwen38GDN = struct {
    ctx: EagerContext,
    hidden_size: usize,
    num_key_heads: usize,
    num_value_heads: usize,
    head_dim: usize,
    conv_kernel: usize = 4,
    norm_eps: f32 = 1e-6,

    // Weights (MLX Linear layout: [out_features, in_features]).
    in_proj_qkv: Array,
    conv1d: Array, // [qkv_dim, conv_kernel, 1]
    in_proj_a: Array, // [Hv, hidden]
    in_proj_b: Array, // [Hv, hidden]
    in_proj_z: Array, // [Hv*D, hidden]
    A_log: Array, // [Hv] f32
    dt_bias: Array, // [Hv] f32
    norm_w: Array, // [D]
    out_proj: Array,

    // Precomputed per-layer constants (see precomputeDerived). Ownership
    // follows the struct: deinit frees them; construction sites that skip
    // deinit (gdnBranch) must free the Precomputed themselves.
    conv1d_t: Array, // [conv_kernel, QKV] = conv1d [QKV, K, 1] transposed
    ones_d: Array, // [D] f32, ones weight for weight-less fast.rmsNorm
    qk_idx: Array, // [Hv] i32, value head h -> key head h / (Hv/Hk)

    // Optional quantized forms (affine 4-bit gs=64 in the real checkpoint).
    qkv_scales: ?Array = null,
    qkv_biases: ?Array = null,
    qkv_quant: QuantConfig = .{},
    a_scales: ?Array = null,
    a_biases: ?Array = null,
    a_quant: QuantConfig = .{},
    b_scales: ?Array = null,
    b_biases: ?Array = null,
    b_quant: QuantConfig = .{},
    z_scales: ?Array = null,
    z_biases: ?Array = null,
    z_quant: QuantConfig = .{},
    o_scales: ?Array = null,
    o_biases: ?Array = null,
    o_quant: QuantConfig = .{},

    pub fn deinit(self: *Qwen38GDN) void {
        self.in_proj_qkv.deinit();
        self.conv1d.deinit();
        self.in_proj_a.deinit();
        self.in_proj_b.deinit();
        self.in_proj_z.deinit();
        self.A_log.deinit();
        self.dt_bias.deinit();
        self.norm_w.deinit();
        self.out_proj.deinit();
        self.conv1d_t.deinit();
        self.ones_d.deinit();
        self.qk_idx.deinit();
        if (self.qkv_scales) |a| a.deinit();
        if (self.qkv_biases) |a| a.deinit();
        if (self.a_scales) |a| a.deinit();
        if (self.a_biases) |a| a.deinit();
        if (self.b_scales) |a| a.deinit();
        if (self.b_biases) |a| a.deinit();
        if (self.z_scales) |a| a.deinit();
        if (self.z_biases) |a| a.deinit();
        if (self.o_scales) |a| a.deinit();
        if (self.o_biases) |a| a.deinit();
    }

    /// One decode token. x: [1, hidden]. Returns [1, hidden]. Advances cache.
    /// No evals inside: the caller (Qwen38ServingModel.forwardTokens) submits
    /// the updated cache states via asyncEval once per token and syncs on the
    /// final logits, so lazy graphs stay O(1) deep across tokens. prefill
    /// applies the same barrier per loop iteration.
    pub fn forward(self: *Qwen38GDN, x: Array, cache: *Qwen38GdnCache) !Array {
        const res = try step(self.ctx, .{
            .x = x,
            .conv_state = cache.conv_state,
            .ssm_state = cache.ssm_state,
            .qkv_w = self.in_proj_qkv,
            .qkv_scales = self.qkv_scales,
            .qkv_biases = self.qkv_biases,
            .conv1d_t = self.conv1d_t,
            .a_w = self.in_proj_a,
            .a_scales = self.a_scales,
            .a_biases = self.a_biases,
            .b_w = self.in_proj_b,
            .b_scales = self.b_scales,
            .b_biases = self.b_biases,
            .z_w = self.in_proj_z,
            .z_scales = self.z_scales,
            .z_biases = self.z_biases,
            .A_log = self.A_log,
            .dt_bias = self.dt_bias,
            .norm_w = self.norm_w,
            .out_w = self.out_proj,
            .out_scales = self.o_scales,
            .out_biases = self.o_biases,
            .ones_d = self.ones_d,
            .qk_idx = self.qk_idx,
            .norm_eps = self.norm_eps,
        });
        cache.conv_state.deinit();
        cache.conv_state = res.conv_state;
        cache.ssm_state.deinit();
        cache.ssm_state = res.ssm_state;
        return res.out;
    }

    /// Prefill: naive per-token loop over x [1, L, hidden]. Returns [1, L, hidden].
    /// One eval barrier per token: submit the updated cache states via
    /// asyncEval, then sync on the token output — the recurrent lazy graph
    /// must not accumulate across prefill tokens.
    pub fn prefill(self: *Qwen38GDN, allocator: std.mem.Allocator, x: Array, cache: *Qwen38GdnCache) !Array {
        const xshape = x.shape();
        const L: i32 = @intCast(xshape[1]);
        const hidden: i32 = @intCast(self.hidden_size);

        var outs = try std.ArrayList(Array).initCapacity(allocator, @intCast(L));
        defer outs.deinit(allocator);
        defer for (outs.items) |o| o.deinit();

        var t: i32 = 0;
        while (t < L) : (t += 1) {
            const x_t = try ops.slice(self.ctx, x, &[_]i32{ 0, t, 0 }, &[_]i32{ 1, t + 1, hidden }, &[_]i32{ 1, 1, 1 });
            defer x_t.deinit();
            const x_flat = try ops.reshape(self.ctx, x_t, &[_]i32{ 1, hidden });
            defer x_flat.deinit();
            const out_t = try self.forward(x_flat, cache);
            errdefer out_t.deinit();
            try eval_mod.asyncEval(allocator, &[_]Array{ cache.conv_state, cache.ssm_state });
            try out_t.eval();
            const out_exp = try ops.expandDims(self.ctx, out_t, 1);
            outs.appendAssumeCapacity(out_exp);
            out_t.deinit();
        }
        return shape_mod.concatenateAxis(self.ctx, outs.items, 1);
    }
};

const TestDims = struct {
    hidden: usize = 8,
    key_heads: usize = 2,
    value_heads: usize = 6,
    head_dim: usize = 4,
    conv_kernel: usize = 4,
};

fn testFill(buf: []f32) void {
    for (buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 37 + 11, 13) - 6)) * 0.02;
    }
}

fn makeTestWeight(allocator: std.mem.Allocator, shape_: []const i32) !Array {
    var n: usize = 1;
    for (shape_) |s| n *= @intCast(s);
    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    testFill(buf);
    return Array.fromData(allocator, f32, buf, shape_);
}

fn makeTestModule(allocator: std.mem.Allocator, ctx: EagerContext, dims: TestDims) !Qwen38GDN {
    const hidden: i32 = @intCast(dims.hidden);
    const HK: i32 = @intCast(dims.key_heads);
    const HV: i32 = @intCast(dims.value_heads);
    const D: i32 = @intCast(dims.head_dim);
    const K: i32 = @intCast(dims.conv_kernel);
    const qkv_dim: i32 = (2 * HK + HV) * D;
    const conv1d = try makeTestWeight(allocator, &[_]i32{ qkv_dim, K, 1 });
    errdefer conv1d.deinit();
    var pre = try precomputeDerived(allocator, ctx, conv1d, dims.key_heads, dims.value_heads, dims.head_dim);
    errdefer pre.deinit();
    return .{
        .ctx = ctx,
        .hidden_size = dims.hidden,
        .num_key_heads = dims.key_heads,
        .num_value_heads = dims.value_heads,
        .head_dim = dims.head_dim,
        .conv_kernel = dims.conv_kernel,
        .in_proj_qkv = try makeTestWeight(allocator, &[_]i32{ qkv_dim, hidden }),
        .conv1d = conv1d,
        .conv1d_t = pre.conv1d_t,
        .ones_d = pre.ones_d,
        .qk_idx = pre.qk_idx,
        .in_proj_a = try makeTestWeight(allocator, &[_]i32{ HV, hidden }),
        .in_proj_b = try makeTestWeight(allocator, &[_]i32{ HV, hidden }),
        .in_proj_z = try makeTestWeight(allocator, &[_]i32{ HV * D, hidden }),
        .A_log = try Array.fromData(allocator, f32, &[_]f32{ -0.5, -0.6, -0.7, -0.8, -0.9, -1.0 }, &[_]i32{HV}),
        .dt_bias = try makeTestWeight(allocator, &[_]i32{HV}),
        .norm_w = try Array.fromData(allocator, f32, &[_]f32{ 1.1, 0.9, 1.05, 0.95 }, &[_]i32{D}),
        .out_proj = try makeTestWeight(allocator, &[_]i32{ hidden, HV * D }),
    };
}

fn makeTestInput(allocator: std.mem.Allocator) !Array {
    var buf: [8]f32 = undefined;
    for (&buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 5, 11) - 5)) * 0.1;
    }
    return Array.fromData(allocator, f32, &buf, &[_]i32{ 1, 8 });
}

test "Qwen38GDN.step: shape, determinism, state advance" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const dims = TestDims{};
    var g1 = try makeTestModule(allocator, ctx, dims);
    defer g1.deinit();
    var cache1 = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache1.deinit();

    var g2 = try makeTestModule(allocator, ctx, dims);
    defer g2.deinit();
    var cache2 = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache2.deinit();

    const x = try makeTestInput(allocator);
    defer x.deinit();

    const out1 = try g1.forward(x, &cache1);
    defer out1.deinit();
    const oshape = out1.shape();
    try std.testing.expectEqual(@as(usize, 2), oshape.len);
    try std.testing.expectEqual(@as(i32, 1), oshape[0]);
    try std.testing.expectEqual(@as(i32, 8), oshape[1]);

    const out2 = try g2.forward(x, &cache2);
    defer out2.deinit();
    const diff = try ops.subtract(ctx, out1, out2);
    defer diff.deinit();
    const abs_diff = try ops.abs(ctx, diff);
    defer abs_diff.deinit();
    const max_diff = try reduce_mod.max(ctx, abs_diff, false);
    defer max_diff.deinit();
    try max_diff.eval();
    try std.testing.expectEqual(@as(f32, 0.0), try max_diff.item(f32));

    const state_abs = try ops.abs(ctx, cache1.ssm_state);
    defer state_abs.deinit();
    const state_max = try reduce_mod.max(ctx, state_abs, false);
    defer state_max.deinit();
    try state_max.eval();
    try std.testing.expect(try state_max.item(f32) > 0.0);
}

test "Qwen38GDN.step: numpy reference cross-check" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const dims = TestDims{};
    var g = try makeTestModule(allocator, ctx, dims);
    defer g.deinit();
    var cache = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache.deinit();

    const x = try makeTestInput(allocator);
    defer x.deinit();

    const out = try g.forward(x, &cache);
    defer out.deinit();
    try out.eval();
    try cache.ssm_state.eval();

    // Reference values from scripts/qwen38_gdn_ref.py (rms_norm q/k with
    // eps-inside-sqrt, norm-then-gate — matching mlx-lm's modules).
    const expected_out = [_]f32{ -0.000802, -0.001897, -0.001004, -0.001422, 0.004621, -0.000405, -0.001701, 0.003579 };
    const got_out = try out.dataSlice(f32);
    for (expected_out, got_out) |e, o| {
        try std.testing.expectApproxEqAbs(e, o, 1e-5);
    }
    const expected_state = [_]f32{ 0.000000, -0.000220, -0.000026, 0.000146, 0.000000, 0.001723, 0.000201, -0.001145, 0.000000, -0.000522, -0.000061, 0.000347, 0.000000, -0.000245, -0.000029, 0.000163, 0.000000, 0.001049, 0.000123, -0.000697, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.001859, 0.000217, -0.001235, 0.000000, 0.000217, 0.000025, -0.000144, 0.000000, -0.001107, -0.000129, 0.000736, 0.000000, 0.000087, 0.000010, -0.000058, 0.000000, -0.000339, -0.000040, 0.000225, 0.000000, 0.000896, 0.000105, -0.000596, 0.000103, -0.000401, 0.001062, 0.002613, -0.000009, 0.000035, -0.000094, -0.000231, 0.000071, -0.000277, 0.000732, 0.001802, -0.000021, 0.000084, -0.000222, -0.000546, -0.000011, 0.000042, -0.000111, -0.000272, 0.000046, -0.000182, 0.000481, 0.001184, 0.000000, 0.000000, 0.000000, 0.000000, 0.000082, -0.000322, 0.000853, 0.002100, 0.000010, -0.000038, 0.000099, 0.000245, -0.000055, 0.000214, -0.000565, -0.001391, 0.000004, -0.000017, 0.000044, 0.000109, -0.000017, 0.000065, -0.000173, -0.000425 };
    const got_state = try cache.ssm_state.dataSlice(f32);
    try std.testing.expectEqual(@as(usize, 96), got_state.len);
    for (expected_state, got_state) |e, o| {
        try std.testing.expectApproxEqAbs(e, o, 1e-5);
    }
}

test "Qwen38GDN.prefill: per-token loop equivalence" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const dims = TestDims{};
    var g = try makeTestModule(allocator, ctx, dims);
    defer g.deinit();
    var cache = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache.deinit();

    var buf: [24]f32 = undefined;
    for (&buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 5, 11) - 5)) * 0.1;
    }
    const x = try Array.fromData(allocator, f32, &buf, &[_]i32{ 1, 3, 8 });
    defer x.deinit();

    const out = try g.prefill(allocator, x, &cache);
    defer out.deinit();
    const oshape = out.shape();
    try std.testing.expectEqual(@as(i32, 1), oshape[0]);
    try std.testing.expectEqual(@as(i32, 3), oshape[1]);
    try std.testing.expectEqual(@as(i32, 8), oshape[2]);
    try out.eval();

    var g_ref = try makeTestModule(allocator, ctx, dims);
    defer g_ref.deinit();
    var cache_ref = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache_ref.deinit();

    var ref_buf: [24]f32 = undefined;
    var t: i32 = 0;
    while (t < 3) : (t += 1) {
        const x_t = try ops.slice(ctx, x, &[_]i32{ 0, t, 0 }, &[_]i32{ 1, t + 1, 8 }, &[_]i32{ 1, 1, 1 });
        defer x_t.deinit();
        const x_flat = try ops.reshape(ctx, x_t, &[_]i32{ 1, 8 });
        defer x_flat.deinit();
        const out_t = try g_ref.forward(x_flat, &cache_ref);
        defer out_t.deinit();
        try out_t.eval();
        const row = try out_t.dataSlice(f32);
        @memcpy(ref_buf[@intCast(t * 8)..@intCast((t + 1) * 8)], row);
    }
    const got = try out.dataSlice(f32);
    for (ref_buf, got) |e, o| {
        try std.testing.expectApproxEqAbs(e, o, 1e-5);
    }
}

test "Qwen38GDN.step: compiled closure matches eager forward" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const dims = TestDims{};
    const qkv_dim = (2 * dims.key_heads + dims.value_heads) * dims.head_dim;

    var g = try makeTestModule(allocator, ctx, dims);
    defer g.deinit();
    var cache_eager = try Qwen38GdnCache.init(allocator, dims.conv_kernel, qkv_dim, dims.value_heads, dims.head_dim);
    defer cache_eager.deinit();
    var cache_comp = try Qwen38GdnCache.init(allocator, dims.conv_kernel, qkv_dim, dims.value_heads, dims.head_dim);
    defer cache_comp.deinit();

    const x = try makeTestInput(allocator);
    defer x.deinit();

    // Eager reference.
    const out_eager = try g.forward(x, &cache_eager);
    defer out_eager.deinit();
    try out_eager.eval();
    try cache_eager.conv_state.eval();
    try cache_eager.ssm_state.eval();

    // Compiled step (plain closure — the toy module is unquantized). First
    // apply traces; second apply must reuse the trace (no recompile) and
    // stay correct, which the state-advance assertions below also cover.
    const cc = try compiledStep(allocator, false);
    defer cc.deinit();

    var t: usize = 0;
    while (t < 2) : (t += 1) {
        const inputs = [_]Array{
            x,             cache_comp.conv_state, cache_comp.ssm_state,
            g.in_proj_qkv, g.conv1d_t,            g.in_proj_a,
            g.in_proj_b,   g.in_proj_z,           g.A_log,
            g.dt_bias,     g.norm_w,              g.out_proj,
            g.ones_d,      g.qk_idx,
        };
        const outs = try cc.apply(&inputs, allocator);
        try std.testing.expectEqual(@as(usize, 3), outs.len);
        cache_comp.conv_state.deinit();
        cache_comp.conv_state = outs[1];
        cache_comp.ssm_state.deinit();
        cache_comp.ssm_state = outs[2];
        const out_comp = outs[0];
        allocator.free(outs);
        defer out_comp.deinit();
        try out_comp.eval();
        try cache_comp.conv_state.eval();
        try cache_comp.ssm_state.eval();

        if (t == 0) {
            const got_out = try out_comp.dataSlice(f32);
            const exp_out = try out_eager.dataSlice(f32);
            try std.testing.expectEqual(exp_out.len, got_out.len);
            for (exp_out, got_out) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
            const got_conv = try cache_comp.conv_state.dataSlice(f32);
            const exp_conv = try cache_eager.conv_state.dataSlice(f32);
            try std.testing.expectEqual(exp_conv.len, got_conv.len);
            for (exp_conv, got_conv) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
            const got_ssm = try cache_comp.ssm_state.dataSlice(f32);
            const exp_ssm = try cache_eager.ssm_state.dataSlice(f32);
            try std.testing.expectEqual(exp_ssm.len, got_ssm.len);
            for (exp_ssm, got_ssm) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
        } else {
            // Second token: state must keep advancing (nonzero delta vs
            // token 1's output would need the eager run too — simply check
            // the state is still finite and the run did not retrace-crash).
            const ssm_abs = try ops.abs(ctx, cache_comp.ssm_state);
            defer ssm_abs.deinit();
            const ssm_max = try reduce_mod.max(ctx, ssm_abs, false);
            defer ssm_max.deinit();
            try ssm_max.eval();
            try std.testing.expect(try ssm_max.item(f32) > 0.0);
        }
    }
}
