//! Qwen3.8 GDN (gated delta net) linear-attention module.
//!
//! Port of Qwen3-Next `Qwen3NextGatedDeltaNet` single-batch naive path.
//! Differs from GlmKDA: asymmetric heads (key Hk / value Hv, kv-group
//! broadcast), separate in_proj_qkv / in_proj_a / in_proj_b / in_proj_z
//! projections, softplus decay, l2norm q/k, silu output gate.
//! f32 throughout; cross-checked against scripts/qwen38_gdn_ref.py.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const reduce_mod = @import("mlx").reduce;
const math_mod = @import("mlx").math;
const fast_mod = @import("mlx").fast;
const activations = @import("mlx").activations;
const quantize_mod = @import("mlx").quantize;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const QuantConfig = quantize_mod.QuantConfig;

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

pub const Qwen38GDN = struct {
    ctx: EagerContext,
    hidden_size: usize,
    num_key_heads: usize,
    num_value_heads: usize,
    head_dim: usize,
    conv_kernel: usize = 4,
    norm_eps: f32 = 1e-6,
    l2_eps: f32 = 1e-6,

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

    fn keyProjDim(self: *const Qwen38GDN) usize {
        return self.num_key_heads * self.head_dim;
    }

    fn valueProjDim(self: *const Qwen38GDN) usize {
        return self.num_value_heads * self.head_dim;
    }

    /// x @ w^T, quantizedMatmul when scales are present.
    fn linear(self: *const Qwen38GDN, x: Array, w: Array, scales: ?Array, biases: ?Array, qcfg: QuantConfig) !Array {
        if (scales) |s| {
            const qw = quantize_mod.QuantizedWeight{
                .data = w,
                .scales = s,
                .biases = biases orelse Array.fromHandle(c.c.mlx_array_new()),
                .config = qcfg,
                .original_shape = &[_]i32{},
            };
            return quantize_mod.quantizedMatmul(self.ctx, x, qw, true);
        }
        const w_t = try ops.transpose(self.ctx, w);
        defer w_t.deinit();
        return ops.matmul(self.ctx, x, w_t);
    }

    /// Weight-less l2norm over the last dim: x / max(||x||_2, eps).
    fn l2Norm(self: *const Qwen38GDN, x: Array) !Array {
        const sq = try math_mod.square(self.ctx, x);
        defer sq.deinit();
        const ss = try reduce_mod.sumAxis(self.ctx, sq, -1, true);
        defer ss.deinit();
        const norm = try ops.sqrt(self.ctx, ss);
        defer norm.deinit();
        const eps_arr = try ops.scalarF32(self.ctx, self.l2_eps);
        defer eps_arr.deinit();
        const clipped = try math_mod.maximum(self.ctx, norm, eps_arr);
        defer clipped.deinit();
        return ops.divide(self.ctx, x, clipped);
    }

    /// softplus(x) = log1p(exp(-|x|)) + max(x, 0), numerically stable.
    fn softplus(self: *const Qwen38GDN, x: Array) !Array {
        const ctx = self.ctx;
        const abs_x = try ops.abs(ctx, x);
        defer abs_x.deinit();
        const neg_abs = try ops.negative(ctx, abs_x);
        defer neg_abs.deinit();
        const ex = try ops.exp(ctx, neg_abs);
        defer ex.deinit();
        const one = try ops.scalarF32(ctx, 1.0);
        defer one.deinit();
        const p1 = try ops.add(ctx, ex, one);
        defer p1.deinit();
        const logp = try ops.log(ctx, p1);
        defer logp.deinit();
        const zero = try ops.scalarF32(ctx, 0.0);
        defer zero.deinit();
        const relu = try math_mod.maximum(ctx, x, zero);
        defer relu.deinit();
        return ops.add(ctx, logp, relu);
    }

    /// One decode token. x: [1, hidden]. Returns [1, hidden]. Advances cache.
    /// Caller MUST eval the returned output each step (decode path does) so
    /// lazy graphs do not accumulate across tokens.
    pub fn forward(self: *Qwen38GDN, x: Array, cache: *Qwen38GdnCache) !Array {
        const ctx = self.ctx;
        const HK: i32 = @intCast(self.num_key_heads);
        const HV: i32 = @intCast(self.num_value_heads);
        const D: i32 = @intCast(self.head_dim);
        const PK: i32 = @intCast(self.keyProjDim());
        const PV: i32 = @intCast(self.valueProjDim());
        const QKV: i32 = 2 * PK + PV;
        const K: i32 = @intCast(self.conv_kernel);

        // 1. qkv projection + depthwise causal conv (k=4) + silu.
        const qkv_lin = try self.linear(x, self.in_proj_qkv, self.qkv_scales, self.qkv_biases, self.qkv_quant);
        defer qkv_lin.deinit();

        const conv_input = try shape_mod.concatenateAxis(ctx, &[_]Array{ cache.conv_state, qkv_lin }, 0);
        defer conv_input.deinit(); // [K, QKV]

        var conv_acc = blk: {
            const x_0 = try ops.slice(ctx, conv_input, &[_]i32{ 0, 0 }, &[_]i32{ 1, QKV }, &[_]i32{ 1, 1 });
            defer x_0.deinit();
            const w_0_s = try ops.slice(ctx, self.conv1d, &[_]i32{ 0, 0, 0 }, &[_]i32{ QKV, 1, 1 }, &[_]i32{ 1, 1, 1 });
            defer w_0_s.deinit();
            const w_0 = try ops.reshape(ctx, w_0_s, &[_]i32{QKV});
            defer w_0.deinit();
            break :blk try ops.multiply(ctx, x_0, w_0);
        };
        {
            var j: i32 = 1;
            while (j < K) : (j += 1) {
                const x_j = try ops.slice(ctx, conv_input, &[_]i32{ j, 0 }, &[_]i32{ j + 1, QKV }, &[_]i32{ 1, 1 });
                defer x_j.deinit();
                const w_j_s = try ops.slice(ctx, self.conv1d, &[_]i32{ 0, j, 0 }, &[_]i32{ QKV, j + 1, 1 }, &[_]i32{ 1, 1, 1 });
                defer w_j_s.deinit();
                const w_j = try ops.reshape(ctx, w_j_s, &[_]i32{QKV});
                defer w_j.deinit();
                const term = try ops.multiply(ctx, x_j, w_j);
                defer term.deinit();
                const acc_new = try ops.add(ctx, conv_acc, term);
                conv_acc.deinit();
                conv_acc = acc_new;
            }
        }
        defer conv_acc.deinit();

        // Advance conv window: keep last K-1 rows.
        const tail = try ops.slice(ctx, conv_input, &[_]i32{ 1, 0 }, &[_]i32{ K, QKV }, &[_]i32{ 1, 1 });
        defer tail.deinit();
        const new_conv_state = try ops.copy(ctx, tail);
        cache.conv_state.deinit();
        cache.conv_state = new_conv_state;
        try cache.conv_state.eval();

        const qkv = try activations.silu(ctx, conv_acc);
        defer qkv.deinit(); // [1, QKV]

        // 2. split q/k/v; l2norm q,k per head; scale q by 1/sqrt(D).
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

        const q_norm = try self.l2Norm(q_resh);
        defer q_norm.deinit();
        const k_norm = try self.l2Norm(k_resh);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const scale_arr = try ops.scalarF32(ctx, scale);
        defer scale_arr.deinit();
        const q = try ops.multiply(ctx, q_norm, scale_arr);
        defer q.deinit();
        const k = k_norm; // rename for symmetry
        defer k.deinit();

        // 3. per-head gates: a (decay input), b (beta), z (output gate).
        const a = try self.linear(x, self.in_proj_a, self.a_scales, self.a_biases, self.a_quant);
        defer a.deinit(); // [1, HV]
        const b = try self.linear(x, self.in_proj_b, self.b_scales, self.b_biases, self.b_quant);
        defer b.deinit(); // [1, HV]
        const z = try self.linear(x, self.in_proj_z, self.z_scales, self.z_biases, self.z_quant);
        defer z.deinit(); // [1, HV*D]

        // g = exp(-exp(A_log) * softplus(a + dt_bias))  [HV]
        const a_flat = try ops.reshape(ctx, a, &[_]i32{HV});
        defer a_flat.deinit();
        const a_dt = try ops.add(ctx, a_flat, self.dt_bias);
        defer a_dt.deinit();
        const sp = try self.softplus(a_dt);
        defer sp.deinit();
        const e_a_log = try ops.exp(ctx, self.A_log);
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

        // 4. broadcast q/k to value heads (kv-group ratio = HV/HK).
        const ratio: i32 = @intCast(self.num_value_heads / self.num_key_heads);
        const q4 = try ops.reshape(ctx, q, &[_]i32{ HK, 1, D });
        defer q4.deinit();
        const q_b = try shape_mod.broadcastTo(ctx, q4, &[_]i32{ HK, ratio, D });
        defer q_b.deinit();
        const q_full = try ops.reshape(ctx, q_b, &[_]i32{ HV, D });
        defer q_full.deinit();
        const k4 = try ops.reshape(ctx, k, &[_]i32{ HK, 1, D });
        defer k4.deinit();
        const k_b0 = try shape_mod.broadcastTo(ctx, k4, &[_]i32{ HK, ratio, D });
        defer k_b0.deinit();
        const k_full = try ops.reshape(ctx, k_b0, &[_]i32{ HV, D });
        defer k_full.deinit();

        // 5. gated delta rule step. State S [Hv, Dv, Dk].
        const g_decay = try ops.reshape(ctx, g, &[_]i32{ HV, 1, 1 });
        defer g_decay.deinit();
        const s_decayed = try ops.multiply(ctx, cache.ssm_state, g_decay);
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
        cache.ssm_state.deinit();
        cache.ssm_state = s_new;
        // Force materialization each step (recurrent lazy graph stays O(1) deep).
        try cache.ssm_state.eval();

        const q_b2 = try ops.reshape(ctx, q_full, &[_]i32{ HV, 1, D });
        defer q_b2.deinit();
        const sq = try ops.multiply(ctx, cache.ssm_state, q_b2);
        defer sq.deinit();
        const y = try reduce_mod.sumAxis(ctx, sq, -1, false); // [HV, Dv]
        defer y.deinit();

        // 6. gated RMSNorm: rmsnorm(y * silu(z)) * norm_w -> out_proj.
        const z_resh = try ops.reshape(ctx, z, &[_]i32{ HV, D });
        defer z_resh.deinit();
        const z_act = try activations.silu(ctx, z_resh);
        defer z_act.deinit();
        const y_gated = try ops.multiply(ctx, y, z_act);
        defer y_gated.deinit();
        const y_normed = try fast_mod.rmsNorm(ctx, y_gated, self.norm_w, self.norm_eps);
        defer y_normed.deinit();
        const y_flat = try ops.reshape(ctx, y_normed, &[_]i32{ 1, PV });
        defer y_flat.deinit();

        return self.linear(y_flat, self.out_proj, self.o_scales, self.o_biases, self.o_quant);
    }

    /// Prefill: naive per-token loop over x [1, L, hidden]. Returns [1, L, hidden].
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
    return .{
        .ctx = ctx,
        .hidden_size = dims.hidden,
        .num_key_heads = dims.key_heads,
        .num_value_heads = dims.value_heads,
        .head_dim = dims.head_dim,
        .conv_kernel = dims.conv_kernel,
        .in_proj_qkv = try makeTestWeight(allocator, &[_]i32{ qkv_dim, hidden }),
        .conv1d = try makeTestWeight(allocator, &[_]i32{ qkv_dim, K, 1 }),
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

    // Reference values from scripts/qwen38_gdn_ref.py (Task 2 Step 3 output).
    const expected_out = [_]f32{ -0.000908, -0.002161, -0.001194, -0.001672, 0.005465, -0.000532, -0.002023, 0.004240 };
    const got_out = try out.dataSlice(f32);
    for (expected_out, got_out) |e, o| {
        try std.testing.expectApproxEqAbs(e, o, 1e-5);
    }
    const expected_state = [_]f32{ 0.000000, -0.000234, -0.000027, 0.000156, 0.000000, 0.001830, 0.000214, -0.001216, 0.000000, -0.000555, -0.000065, 0.000369, 0.000000, -0.000260, -0.000030, 0.000173, 0.000000, 0.001114, 0.000130, -0.000740, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.001975, 0.000231, -0.001312, 0.000000, 0.000231, 0.000027, -0.000153, 0.000000, -0.001176, -0.000138, 0.000781, 0.000000, 0.000092, 0.000011, -0.000061, 0.000000, -0.000360, -0.000042, 0.000239, 0.000000, 0.000952, 0.000111, -0.000633, 0.000107, -0.000419, 0.001109, 0.002728, -0.000009, 0.000037, -0.000098, -0.000241, 0.000074, -0.000289, 0.000764, 0.001881, -0.000022, 0.000088, -0.000232, -0.000570, -0.000011, 0.000044, -0.000116, -0.000284, 0.000049, -0.000190, 0.000503, 0.001236, 0.000000, 0.000000, 0.000000, 0.000000, 0.000086, -0.000337, 0.000891, 0.002192, 0.000010, -0.000039, 0.000104, 0.000256, -0.000057, 0.000223, -0.000590, -0.001452, 0.000004, -0.000017, 0.000046, 0.000114, -0.000017, 0.000068, -0.000181, -0.000444 };
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
