/// GLM-5.3 KDA (Kimi Delta Attention) linear-attention module.
///
/// Faithful port of mlx-vlm `Glm5NextLinearAttention` +
/// `gated_delta.py` naive ops path (`gated_delta_ops`), single batch.
///
/// Per decode step:
///   qkv = silu(depthwise_conv1d(qkv_proj(x), k=4))        // windowed conv cache
///   q = scale^2 * rmsnorm(q), k = scale * rmsnorm(k)      // eps = 1e-6 / head_dim
///   g = exp(lower_bound * sigmoid(exp(A_log) * (a + dt_bias)))  // per-head decay
///   beta = sigmoid(b)
///   S = S * g; kv = (S * k).sum(dk); d = (v - kv) * beta
///   S = S + k ⊗ d; y = (S * q).sum(dk)
///   out = o_proj(o_norm(y) * sigmoid(gate))
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

// ============================================================
// Cache
// ============================================================

pub const GlmKdaCache = struct {
    conv_state: Array, // [conv_kernel-1, 3*proj_dim]
    ssm_state: Array, // [heads, head_dim, head_dim] f32, layout [H, Dv, Dk]

    pub fn init(allocator: std.mem.Allocator, conv_kernel: usize, proj_dim3: usize, heads: usize, head_dim: usize) !GlmKdaCache {
        return .{
            .conv_state = try array_mod.zeros(allocator, &[_]i32{ @intCast(conv_kernel - 1), @intCast(proj_dim3) }, .float32),
            .ssm_state = try array_mod.zeros(allocator, &[_]i32{ @intCast(heads), @intCast(head_dim), @intCast(head_dim) }, .float32),
        };
    }

    pub fn deinit(self: *GlmKdaCache) void {
        self.conv_state.deinit();
        self.ssm_state.deinit();
    }
};

// ============================================================
// Module
// ============================================================

pub const GlmKDA = struct {
    ctx: EagerContext,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    conv_kernel: usize = 4,
    lower_bound: f32 = -5.0,
    o_norm_eps: f32 = 1e-5,

    // Weights (MLX Linear layout: [out_features, in_features]).
    qkv_proj: Array,
    qkv_conv: Array, // [3*proj_dim, conv_kernel, 1]
    fbg_a_proj: Array,
    f_b_proj: Array,
    g_b_proj: Array,
    A_log: Array, // [num_heads] f32
    dt_bias: Array, // [num_heads * head_dim] f32
    o_norm: Array, // [head_dim]
    o_proj: Array,

    // Optional quantized forms (affine bits/gs from dmlx_manifest).
    qkv_proj_scales: ?Array = null,
    qkv_proj_biases: ?Array = null,
    qkv_quant: QuantConfig = .{},
    fbg_a_proj_scales: ?Array = null,
    fbg_a_proj_biases: ?Array = null,
    fbg_a_quant: QuantConfig = .{},
    o_proj_scales: ?Array = null,
    o_proj_biases: ?Array = null,
    o_quant: QuantConfig = .{},

    pub fn deinit(self: *GlmKDA) void {
        self.qkv_proj.deinit();
        self.qkv_conv.deinit();
        self.fbg_a_proj.deinit();
        self.f_b_proj.deinit();
        self.g_b_proj.deinit();
        self.A_log.deinit();
        self.dt_bias.deinit();
        self.o_norm.deinit();
        self.o_proj.deinit();
        if (self.qkv_proj_scales) |a| a.deinit();
        if (self.qkv_proj_biases) |a| a.deinit();
        if (self.fbg_a_proj_scales) |a| a.deinit();
        if (self.fbg_a_proj_biases) |a| a.deinit();
        if (self.o_proj_scales) |a| a.deinit();
        if (self.o_proj_biases) |a| a.deinit();
    }

    fn projDim(self: *const GlmKDA) usize {
        return self.num_heads * self.head_dim;
    }

    /// x @ w^T, quantizedMatmul when scales are present (DSV4 expert pattern).
    fn linear(self: *const GlmKDA, x: Array, w: Array, scales: ?Array, biases: ?Array, qcfg: QuantConfig) !Array {
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

    /// Weight-less RMSNorm over the last dim: x * rsqrt(mean(x^2) + eps).
    fn rmsNormNoWeight(self: *const GlmKDA, x: Array, eps: f32) !Array {
        const sq = try math_mod.square(self.ctx, x);
        defer sq.deinit();
        const ms = try reduce_mod.meanAxis(self.ctx, sq, -1, true);
        defer ms.deinit();
        const eps_arr = try ops.scalarF32(self.ctx, eps);
        defer eps_arr.deinit();
        const denom = try ops.add(self.ctx, ms, eps_arr);
        defer denom.deinit();
        const rs = try math_mod.rsqrt(self.ctx, denom);
        defer rs.deinit();
        return ops.multiply(self.ctx, x, rs);
    }

    /// One decode token. x: [1, hidden]. Returns [1, hidden]. Advances cache.
    pub fn forward(self: *GlmKDA, x: Array, cache: *GlmKdaCache) !Array {
        const H: i32 = @intCast(self.num_heads);
        const D: i32 = @intCast(self.head_dim);
        const P: i32 = @intCast(self.projDim());
        const K: i32 = @intCast(self.conv_kernel);
        const ctx = self.ctx;

        // 1. qkv projection + depthwise causal conv (k=4) + silu.
        const qkv_lin = try self.linear(x, self.qkv_proj, self.qkv_proj_scales, self.qkv_proj_biases, self.qkv_quant);
        defer qkv_lin.deinit();

        const conv_input = try shape_mod.concatenateAxis(ctx, &[_]Array{ cache.conv_state, qkv_lin }, 0);
        defer conv_input.deinit(); // [K, 3P]

        // Hand-rolled sliding-window multiply-accumulate (k is tiny).
        const x_0 = try ops.slice(ctx, conv_input, &[_]i32{ 0, 0 }, &[_]i32{ 1, 3 * P }, &[_]i32{ 1, 1 });
        defer x_0.deinit();
        const w_0_s = try ops.slice(ctx, self.qkv_conv, &[_]i32{ 0, 0, 0 }, &[_]i32{ 3 * P, 1, 1 }, &[_]i32{ 1, 1, 1 });
        defer w_0_s.deinit();
        const w_0 = try ops.reshape(ctx, w_0_s, &[_]i32{3 * P});
        defer w_0.deinit();
        var conv_acc = try ops.multiply(ctx, x_0, w_0);
        {
            var j: i32 = 1;
            while (j < K) : (j += 1) {
                const x_j = try ops.slice(ctx, conv_input, &[_]i32{ j, 0 }, &[_]i32{ j + 1, 3 * P }, &[_]i32{ 1, 1 });
                defer x_j.deinit();
                const w_j_s = try ops.slice(ctx, self.qkv_conv, &[_]i32{ 0, j, 0 }, &[_]i32{ 3 * P, j + 1, 1 }, &[_]i32{ 1, 1, 1 });
                defer w_j_s.deinit();
                const w_j = try ops.reshape(ctx, w_j_s, &[_]i32{3 * P});
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
        const tail = try ops.slice(ctx, conv_input, &[_]i32{ 1, 0 }, &[_]i32{ K, 3 * P }, &[_]i32{ 1, 1 });
        defer tail.deinit();
        const new_conv_state = try ops.copy(ctx, tail);
        cache.conv_state.deinit();
        cache.conv_state = new_conv_state;
        try cache.conv_state.eval();

        const qkv = try activations.silu(ctx, conv_acc);
        defer qkv.deinit(); // [1, 3P]

        // 2. split q/k/v -> [H, D]; scale + weight-less rmsnorm, eps = 1e-6 / D.
        const qkv_flat = try ops.reshape(ctx, qkv, &[_]i32{3 * P});
        defer qkv_flat.deinit();
        const q_parts = try ops.slice(ctx, qkv_flat, &[_]i32{0}, &[_]i32{P}, &[_]i32{1});
        defer q_parts.deinit();
        const k_parts = try ops.slice(ctx, qkv_flat, &[_]i32{P}, &[_]i32{2 * P}, &[_]i32{1});
        defer k_parts.deinit();
        const v_parts = try ops.slice(ctx, qkv_flat, &[_]i32{2 * P}, &[_]i32{3 * P}, &[_]i32{1});
        defer v_parts.deinit();
        const eps: f32 = 1e-6 / @as(f32, @floatFromInt(self.head_dim));
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));

        const q_resh = try ops.reshape(ctx, q_parts, &[_]i32{ H, D });
        defer q_resh.deinit();
        const k_resh = try ops.reshape(ctx, k_parts, &[_]i32{ H, D });
        defer k_resh.deinit();
        const v = try ops.reshape(ctx, v_parts, &[_]i32{ H, D });
        defer v.deinit();

        const q_norm = try self.rmsNormNoWeight(q_resh, eps);
        defer q_norm.deinit();
        const k_norm = try self.rmsNormNoWeight(k_resh, eps);
        defer k_norm.deinit();
        const q = try ops.multiply(ctx, q_norm, try ops.scalarF32(ctx, scale * scale));
        defer q.deinit();
        const k = try ops.multiply(ctx, k_norm, try ops.scalarF32(ctx, scale));
        defer k.deinit();

        // 3. fbg split: f_a [D], b_raw [H], g_a [D].
        const fbg = try self.linear(x, self.fbg_a_proj, self.fbg_a_proj_scales, self.fbg_a_proj_biases, self.fbg_a_quant);
        defer fbg.deinit();
        const fbg_flat = try ops.reshape(ctx, fbg, &[_]i32{2 * D + H});
        defer fbg_flat.deinit();
        const f_a = try ops.slice(ctx, fbg_flat, &[_]i32{0}, &[_]i32{D}, &[_]i32{1});
        defer f_a.deinit();
        const b_raw = try ops.slice(ctx, fbg_flat, &[_]i32{D}, &[_]i32{D + H}, &[_]i32{1});
        defer b_raw.deinit();
        const g_a = try ops.slice(ctx, fbg_flat, &[_]i32{D + H}, &[_]i32{2 * D + H}, &[_]i32{1});
        defer g_a.deinit();

        const a_flat = try self.linear(f_a, self.f_b_proj, null, null, .{});
        defer a_flat.deinit();
        const a = try ops.reshape(ctx, a_flat, &[_]i32{ H, D });
        defer a.deinit();

        const gate_flat = try self.linear(g_a, self.g_b_proj, null, null, .{});
        defer gate_flat.deinit();
        const gate = try ops.reshape(ctx, gate_flat, &[_]i32{ H, D });
        defer gate.deinit();

        // 4. Gated delta rule step (gated_delta_ops naive path).
        //    g = exp(lower_bound * sigmoid(exp(A_log) * (a + dt_bias))), beta = sigmoid(b).
        const dt_bias_h = try ops.reshape(ctx, self.dt_bias, &[_]i32{ H, D });
        defer dt_bias_h.deinit();
        const a_dt = try ops.add(ctx, a, dt_bias_h);
        defer a_dt.deinit();
        const e_a_log = try ops.exp(ctx, self.A_log);
        defer e_a_log.deinit();
        const e_a_log_h = try ops.reshape(ctx, e_a_log, &[_]i32{ H, 1 });
        defer e_a_log_h.deinit();
        const g_pre = try ops.multiply(ctx, e_a_log_h, a_dt);
        defer g_pre.deinit();
        const g_sig = try ops.sigmoid(ctx, g_pre);
        defer g_sig.deinit();
        const lb = try ops.scalarF32(ctx, self.lower_bound);
        defer lb.deinit();
        const g_log = try ops.multiply(ctx, g_sig, lb);
        defer g_log.deinit();
        const g = try ops.exp(ctx, g_log); // [H, D] vectorized gate
        defer g.deinit();

        const beta = try ops.sigmoid(ctx, b_raw); // [H]
        defer beta.deinit();

        // State decay (vectorized gating: decay = g[..., None, :] over Dk).
        const g_decay = try ops.expandDims(ctx, g, 1); // [H, 1, Dk]
        defer g_decay.deinit();
        const s_decayed = try ops.multiply(ctx, cache.ssm_state, g_decay);
        defer s_decayed.deinit();

        const k_b = try ops.reshape(ctx, k, &[_]i32{ H, 1, D });
        defer k_b.deinit();
        const sk = try ops.multiply(ctx, s_decayed, k_b);
        defer sk.deinit();
        const kv_mem = try reduce_mod.sumAxis(ctx, sk, -1, false); // [H, Dv]
        defer kv_mem.deinit();

        const v_sub = try ops.subtract(ctx, v, kv_mem);
        defer v_sub.deinit();
        const beta_b = try ops.reshape(ctx, beta, &[_]i32{ H, 1 });
        defer beta_b.deinit();
        const delta = try ops.multiply(ctx, v_sub, beta_b); // [H, Dv]
        defer delta.deinit();

        const delta_b = try ops.expandDims(ctx, delta, -1); // [H, Dv, 1]
        defer delta_b.deinit();
        const k_delta = try ops.multiply(ctx, k_b, delta_b);
        defer k_delta.deinit();
        const s_new = try ops.add(ctx, s_decayed, k_delta);
        cache.ssm_state.deinit();
        cache.ssm_state = s_new;

        const q_b = try ops.reshape(ctx, q, &[_]i32{ H, 1, D });
        defer q_b.deinit();
        const sq = try ops.multiply(ctx, cache.ssm_state, q_b);
        defer sq.deinit();
        const y = try reduce_mod.sumAxis(ctx, sq, -1, false); // [H, Dv]
        defer y.deinit();

        // 5. o_norm(y) * sigmoid(gate) -> o_proj.
        const y_normed = try fast_mod.rmsNorm(ctx, y, self.o_norm, self.o_norm_eps);
        defer y_normed.deinit();
        const gate_sig = try ops.sigmoid(ctx, gate);
        defer gate_sig.deinit();
        const y_gated = try ops.multiply(ctx, y_normed, gate_sig);
        defer y_gated.deinit();
        const y_flat = try ops.reshape(ctx, y_gated, &[_]i32{ 1, P });
        defer y_flat.deinit();

        return self.linear(y_flat, self.o_proj, self.o_proj_scales, self.o_proj_biases, self.o_quant);
    }

    /// Prefill: naive per-token loop over x [1, L, hidden]. Returns [1, L, hidden].
    pub fn prefill(self: *GlmKDA, allocator: std.mem.Allocator, x: Array, cache: *GlmKdaCache) !Array {
        const xshape = x.shape();
        const L: i32 = @intCast(xshape[1]);
        const hidden: i32 = @intCast(self.hidden_size);

        var outs = try std.ArrayList(Array).initCapacity(allocator, @intCast(L));
        defer outs.deinit(allocator);
        errdefer for (outs.items) |o| o.deinit();

        var t: i32 = 0;
        while (t < L) : (t += 1) {
            const x_t = try ops.slice(self.ctx, x, &[_]i32{ 0, t, 0 }, &[_]i32{ 1, t + 1, hidden }, &[_]i32{ 1, 1, 1 });
            defer x_t.deinit();
            const x_flat = try ops.reshape(self.ctx, x_t, &[_]i32{ 1, hidden });
            defer x_flat.deinit();
            const out_t = try self.forward(x_flat, cache);
            errdefer out_t.deinit();
            const out_exp = try ops.expandDims(self.ctx, out_t, 1);
            out_t.deinit();
            try outs.append(allocator, out_exp);
        }
        defer for (outs.items) |o| o.deinit();
        return shape_mod.concatenateAxis(self.ctx, outs.items, 1);
    }
};

// ============================================================
// Tests
// ============================================================

const TestDims = struct {
    hidden: usize = 8,
    heads: usize = 2,
    head_dim: usize = 4,
    conv_kernel: usize = 4,
};

// Deterministic weight fill shared with scripts/glm53_kda_ref.py.
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

fn makeTestModule(allocator: std.mem.Allocator, ctx: EagerContext, dims: TestDims) !GlmKDA {
    const hidden: i32 = @intCast(dims.hidden);
    const H: i32 = @intCast(dims.heads);
    const D: i32 = @intCast(dims.head_dim);
    const P: i32 = H * D;
    const K: i32 = @intCast(dims.conv_kernel);
    return .{
        .ctx = ctx,
        .hidden_size = dims.hidden,
        .num_heads = dims.heads,
        .head_dim = dims.head_dim,
        .conv_kernel = dims.conv_kernel,
        .qkv_proj = try makeTestWeight(allocator, &[_]i32{ 3 * P, hidden }),
        .qkv_conv = try makeTestWeight(allocator, &[_]i32{ 3 * P, K, 1 }),
        .fbg_a_proj = try makeTestWeight(allocator, &[_]i32{ 2 * D + H, hidden }),
        .f_b_proj = try makeTestWeight(allocator, &[_]i32{ P, D }),
        .g_b_proj = try makeTestWeight(allocator, &[_]i32{ P, D }),
        .A_log = try Array.fromData(allocator, f32, &[_]f32{ -0.5, -0.75 }, &[_]i32{H}),
        .dt_bias = try makeTestWeight(allocator, &[_]i32{P}),
        .o_norm = try Array.fromData(allocator, f32, &[_]f32{ 1.1, 0.9, 1.05, 0.95 }, &[_]i32{D}),
        .o_proj = try makeTestWeight(allocator, &[_]i32{ hidden, P }),
    };
}

fn makeTestInput(allocator: std.mem.Allocator) !Array {
    var buf: [8]f32 = undefined;
    for (&buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 5, 11) - 5)) * 0.1;
    }
    return Array.fromData(allocator, f32, &buf, &[_]i32{ 1, 8 });
}

test "GlmKDA.step: 输出形状与状态推进" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const dims = TestDims{};

    var kda1 = try makeTestModule(allocator, ctx, dims);
    defer kda1.deinit();
    var cache1 = try GlmKdaCache.init(allocator, dims.conv_kernel, 3 * dims.heads * dims.head_dim, dims.heads, dims.head_dim);
    defer cache1.deinit();

    var kda2 = try makeTestModule(allocator, ctx, dims);
    defer kda2.deinit();
    var cache2 = try GlmKdaCache.init(allocator, dims.conv_kernel, 3 * dims.heads * dims.head_dim, dims.heads, dims.head_dim);
    defer cache2.deinit();

    const x = try makeTestInput(allocator);
    defer x.deinit();

    // 1. Output shape == [1, hidden].
    const out1 = try kda1.forward(x, &cache1);
    defer out1.deinit();
    const oshape = out1.shape();
    try std.testing.expectEqual(@as(usize, 2), oshape.len);
    try std.testing.expectEqual(@as(i32, 1), oshape[0]);
    try std.testing.expectEqual(@as(i32, 8), oshape[1]);

    // 2. Determinism: same initial state + same input -> same output.
    const out2 = try kda2.forward(x, &cache2);
    defer out2.deinit();
    const diff = try ops.subtract(ctx, out1, out2);
    defer diff.deinit();
    const abs_diff = try ops.abs(ctx, diff);
    defer abs_diff.deinit();
    const max_diff = try reduce_mod.max(ctx, abs_diff, false);
    defer max_diff.deinit();
    try max_diff.eval();
    try std.testing.expectEqual(@as(f32, 0.0), try max_diff.item(f32));

    // 3. SSM state was modified (non-zero after step).
    const state_abs = try ops.abs(ctx, cache1.ssm_state);
    defer state_abs.deinit();
    const state_max = try reduce_mod.max(ctx, state_abs, false);
    defer state_max.deinit();
    try state_max.eval();
    try std.testing.expect(try state_max.item(f32) > 0.0);
}

test "GlmKDA.step: numpy 参考交叉验证" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const dims = TestDims{};
    var kda = try makeTestModule(allocator, ctx, dims);
    defer kda.deinit();
    var cache = try GlmKdaCache.init(allocator, dims.conv_kernel, 3 * dims.heads * dims.head_dim, dims.heads, dims.head_dim);
    defer cache.deinit();

    const x = try makeTestInput(allocator);
    defer x.deinit();

    const out = try kda.forward(x, &cache);
    defer out.deinit();
    try out.eval();
    try cache.ssm_state.eval();

    // Reference values from scripts/glm53_kda_ref.py (numpy, same weights/inputs).
    const expected_out = [_]f32{ 0.011439, 0.000536, 0.001318, -0.015662, -0.009254, 0.004171, 0.007550, -0.012028 };
    const got_out = try out.dataSlice(f32);
    for (expected_out, got_out) |e, g| {
        try std.testing.expectApproxEqAbs(e, g, 1e-5);
    }

    const expected_state = [_]f32{
        0.000000, -0.000228, -0.000027, 0.000151, 0.000000, 0.001781, 0.000208, -0.001183, 0.000000, -0.000540, -0.000063, 0.000359, 0.000000, -0.000253, -0.000030, 0.000168,
        0.000048, -0.000187, 0.000496,  0.001219, 0.000000, 0.000000, 0.000000, 0.000000,  0.000085, -0.000332, 0.000878,  0.002161, 0.000010, -0.000039, 0.000103,  0.000253,
    };
    const got_state = try cache.ssm_state.dataSlice(f32);
    try std.testing.expectEqual(@as(usize, 32), got_state.len);
    for (expected_state, got_state) |e, g| {
        try std.testing.expectApproxEqAbs(e, g, 1e-5);
    }
}

test "GlmKDA.prefill: 逐 token 循环等价" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const dims = TestDims{};
    var kda = try makeTestModule(allocator, ctx, dims);
    defer kda.deinit();
    var cache = try GlmKdaCache.init(allocator, dims.conv_kernel, 3 * dims.heads * dims.head_dim, dims.heads, dims.head_dim);
    defer cache.deinit();

    // Build a [1, 3, 8] input from three token rows.
    var buf: [24]f32 = undefined;
    for (&buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 5, 11) - 5)) * 0.1;
    }
    const x = try Array.fromData(allocator, f32, &buf, &[_]i32{ 1, 3, 8 });
    defer x.deinit();

    const out = try kda.prefill(allocator, x, &cache);
    defer out.deinit();
    const oshape = out.shape();
    try std.testing.expectEqual(@as(i32, 1), oshape[0]);
    try std.testing.expectEqual(@as(i32, 3), oshape[1]);
    try std.testing.expectEqual(@as(i32, 8), oshape[2]);
}
