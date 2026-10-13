//! Qwen3.8 full-attention layer: per-head q/k RMSNorm, partial RoPE
//! (dims = head_dim * partial_rotary_factor), GQA, sigmoid output gate
//! (fused into q_proj, interleaved per head), model-internal KV cache.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const reduce_mod = @import("mlx").reduce;
const fast_mod = @import("mlx").fast;
const quantize_mod = @import("mlx").quantize;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const QuantConfig = quantize_mod.QuantConfig;

pub const Qwen38AttnCache = struct {
    keys: ?Array = null, // [1, n_kv, L, head_dim]
    values: ?Array = null, // [1, n_kv, L, head_dim]

    pub fn deinit(self: *Qwen38AttnCache) void {
        if (self.keys) |*k| k.deinit();
        if (self.values) |*v| v.deinit();
        self.keys = null;
        self.values = null;
    }

    pub fn len(self: *const Qwen38AttnCache) i32 {
        const k = self.keys orelse return 0;
        return k.shape()[2];
    }
};

test "Qwen38Attn.step: shapes, cache growth, determinism" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Toy dims: hidden=16, q heads=2, kv heads=1, head_dim=8, rotary dims=2.
    var attn = try Qwen38Attn.makeTestModule(allocator, ctx);
    defer attn.deinit();
    var cache = Qwen38AttnCache{};
    defer cache.deinit();

    var buf: [16]f32 = undefined;
    for (&buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 5, 11) - 5)) * 0.1;
    }
    const x = try Array.fromData(allocator, f32, &buf, &[_]i32{ 1, 16 });
    defer x.deinit();

    const out1 = try attn.forward(x, &cache);
    defer out1.deinit();
    try out1.eval();
    const oshape = out1.shape();
    try std.testing.expectEqual(@as(i32, 1), oshape[0]);
    try std.testing.expectEqual(@as(i32, 16), oshape[1]);
    try std.testing.expectEqual(@as(i32, 1), cache.len());

    const out2 = try attn.forward(x, &cache);
    defer out2.deinit();
    try out2.eval();
    try std.testing.expectEqual(@as(i32, 2), cache.len());

    // Determinism: fresh module + cache, same input, first-step output matches.
    var attn2 = try Qwen38Attn.makeTestModule(allocator, ctx);
    defer attn2.deinit();
    var cache2 = Qwen38AttnCache{};
    defer cache2.deinit();
    const out3 = try attn2.forward(x, &cache2);
    defer out3.deinit();
    const diff = try ops.subtract(ctx, out1, out3);
    defer diff.deinit();
    const abs_diff = try ops.abs(ctx, diff);
    defer abs_diff.deinit();
    const max_diff = try reduce_mod.max(ctx, abs_diff, false);
    defer max_diff.deinit();
    try max_diff.eval();
    try std.testing.expectEqual(@as(f32, 0.0), try max_diff.item(f32));
}

pub const Qwen38Attn = struct {
    ctx: EagerContext,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dims: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    output_gate: bool = true,

    // q_proj packs per-head [q head_dim | gate head_dim]: out = H * 2 * D.
    q_proj: Array,
    k_proj: Array,
    v_proj: Array,
    o_proj: Array,
    q_norm: Array, // [head_dim]
    k_norm: Array, // [head_dim]

    q_scales: ?Array = null,
    q_biases: ?Array = null,
    q_quant: QuantConfig = .{},
    k_scales: ?Array = null,
    k_biases: ?Array = null,
    k_quant: QuantConfig = .{},
    v_scales: ?Array = null,
    v_biases: ?Array = null,
    v_quant: QuantConfig = .{},
    o_scales: ?Array = null,
    o_biases: ?Array = null,
    o_quant: QuantConfig = .{},

    pub fn deinit(self: *Qwen38Attn) void {
        self.q_proj.deinit();
        self.k_proj.deinit();
        self.v_proj.deinit();
        self.o_proj.deinit();
        self.q_norm.deinit();
        self.k_norm.deinit();
        if (self.q_scales) |a| a.deinit();
        if (self.q_biases) |a| a.deinit();
        if (self.k_scales) |a| a.deinit();
        if (self.k_biases) |a| a.deinit();
        if (self.v_scales) |a| a.deinit();
        if (self.v_biases) |a| a.deinit();
        if (self.o_scales) |a| a.deinit();
        if (self.o_biases) |a| a.deinit();
    }

    /// x @ w^T, quantizedMatmul when scales are present.
    fn linear(self: *const Qwen38Attn, x: Array, w: Array, scales: ?Array, biases: ?Array, qcfg: QuantConfig) !Array {
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

    /// x: [1, hidden] single token OR [1, L, hidden] prefill.
    /// Returns the same leading shape with hidden last dim. Advances cache.
    pub fn forward(self: *Qwen38Attn, x: Array, cache: *Qwen38AttnCache) !Array {
        const ctx = self.ctx;
        const H: i32 = @intCast(self.num_heads);
        const HKV: i32 = @intCast(self.num_kv_heads);
        const D: i32 = @intCast(self.head_dim);
        const xshape = x.shape();
        const prefill = xshape.len == 3;
        const L: i32 = if (prefill) xshape[1] else 1;
        const offset: i32 = cache.len();

        // q path: [.., H*2*D] -> [.., H, 2*D] -> chunk into q / gate.
        const q_lin = try self.linear(x, self.q_proj, self.q_scales, self.q_biases, self.q_quant);
        defer q_lin.deinit();
        const q_resh = if (prefill)
            try ops.reshape(ctx, q_lin, &[_]i32{ 1, L, H, 2 * D })
        else
            try ops.reshape(ctx, q_lin, &[_]i32{ 1, H, 2 * D });
        defer q_resh.deinit();
        const q_raw = if (prefill)
            try ops.slice(ctx, q_resh, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, L, H, D }, &[_]i32{ 1, 1, 1, 1 })
        else
            try ops.slice(ctx, q_resh, &[_]i32{ 0, 0, 0 }, &[_]i32{ 1, H, D }, &[_]i32{ 1, 1, 1 });
        defer q_raw.deinit();
        const gate = if (self.output_gate) blk: {
            const g = if (prefill)
                try ops.slice(ctx, q_resh, &[_]i32{ 0, 0, 0, D }, &[_]i32{ 1, L, H, 2 * D }, &[_]i32{ 1, 1, 1, 1 })
            else
                try ops.slice(ctx, q_resh, &[_]i32{ 0, 0, D }, &[_]i32{ 1, H, 2 * D }, &[_]i32{ 1, 1, 1 });
            break :blk g;
        } else null;
        defer if (gate) |g| g.deinit();

        const k_lin = try self.linear(x, self.k_proj, self.k_scales, self.k_biases, self.k_quant);
        defer k_lin.deinit();
        const v_lin = try self.linear(x, self.v_proj, self.v_scales, self.v_biases, self.v_quant);
        defer v_lin.deinit();
        const k_resh = if (prefill)
            try ops.reshape(ctx, k_lin, &[_]i32{ 1, L, HKV, D })
        else
            try ops.reshape(ctx, k_lin, &[_]i32{ 1, HKV, D });
        defer k_resh.deinit();
        const v_resh = if (prefill)
            try ops.reshape(ctx, v_lin, &[_]i32{ 1, L, HKV, D })
        else
            try ops.reshape(ctx, v_lin, &[_]i32{ 1, HKV, D });
        defer v_resh.deinit();

        // Per-head RMSNorm, then partial RoPE (dims = rotary_dims).
        const q_n = try fast_mod.rmsNorm(ctx, q_raw, self.q_norm, self.rms_norm_eps);
        defer q_n.deinit();
        const k_n = try fast_mod.rmsNorm(ctx, k_resh, self.k_norm, self.rms_norm_eps);
        defer k_n.deinit();
        const rd: i32 = @intCast(self.rotary_dims);
        const q_r = try fast_mod.rope(ctx, q_n, rd, false, self.rope_theta, 1.0, offset, null);
        defer q_r.deinit();
        const k_r = try fast_mod.rope(ctx, k_n, rd, false, self.rope_theta, 1.0, offset, null);
        defer k_r.deinit();

        // Transpose to [1, heads, L, D] and append to cache.
        const q_t = if (prefill)
            try ops.transposeAxes(ctx, q_r, &[_]i32{ 0, 2, 1, 3 })
        else
            try ops.expandDims(ctx, q_r, 2); // [1, H, 1, D]
        defer q_t.deinit();
        const k_t = if (prefill)
            try ops.transposeAxes(ctx, k_r, &[_]i32{ 0, 2, 1, 3 })
        else
            try ops.expandDims(ctx, k_r, 2);
        defer k_t.deinit();
        const v_t = if (prefill)
            try ops.transposeAxes(ctx, v_resh, &[_]i32{ 0, 2, 1, 3 })
        else
            try ops.expandDims(ctx, v_resh, 2);
        defer v_t.deinit();

        // Build new cache arrays first; only on full success swap them in, so
        // any failure path leaves the old cache untouched (no leak, no
        // dangling pointer, no double-free).
        const k_all = if (cache.keys) |ck|
            try shape_mod.concatenateAxis(ctx, &[_]Array{ ck, k_t }, 2)
        else
            try ops.copy(ctx, k_t);
        errdefer k_all.deinit();
        const v_all = if (cache.values) |cv|
            try shape_mod.concatenateAxis(ctx, &[_]Array{ cv, v_t }, 2)
        else
            try ops.copy(ctx, v_t);
        errdefer v_all.deinit();
        if (cache.keys) |*ck| ck.deinit();
        if (cache.values) |*cv| cv.deinit();
        cache.keys = k_all;
        cache.values = v_all;
        try cache.keys.?.eval();
        try cache.values.?.eval();

        // GQA: repeat kv heads to q head count.
        const kv_len: i32 = cache.len();
        const rep: i32 = @intCast(self.num_heads / self.num_kv_heads);
        const k_rep = if (rep > 1) blk: {
            const e = try ops.expandDims(ctx, cache.keys.?, 2); // [1, HKV, 1, L, D]
            defer e.deinit();
            const b = try shape_mod.broadcastTo(ctx, e, &[_]i32{ 1, HKV, rep, kv_len, D });
            defer b.deinit();
            break :blk try ops.reshape(ctx, b, &[_]i32{ 1, H, kv_len, D });
        } else try ops.copy(ctx, cache.keys.?);
        defer k_rep.deinit();
        const v_rep = if (rep > 1) blk: {
            const e = try ops.expandDims(ctx, cache.values.?, 2);
            defer e.deinit();
            const b = try shape_mod.broadcastTo(ctx, e, &[_]i32{ 1, HKV, rep, kv_len, D });
            defer b.deinit();
            break :blk try ops.reshape(ctx, b, &[_]i32{ 1, H, kv_len, D });
        } else try ops.copy(ctx, cache.values.?);
        defer v_rep.deinit();

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const attn = try fast_mod.scaledDotProductAttention(
            ctx,
            q_t,
            k_rep,
            v_rep,
            scale,
            if (prefill) "causal" else "",
            null,
            null,
        );
        defer attn.deinit(); // [1, H, L, D]

        // Back to [1, L, H*D] (or [1, H*D]).
        const attn_t = if (prefill)
            try ops.transposeAxes(ctx, attn, &[_]i32{ 0, 2, 1, 3 })
        else
            try ops.squeeze(ctx, attn);
        defer attn_t.deinit();
        const attn_flat = if (prefill)
            try ops.reshape(ctx, attn_t, &[_]i32{ 1, L, H * D })
        else
            try ops.reshape(ctx, attn_t, &[_]i32{ 1, H * D });
        defer attn_flat.deinit();

        // Sigmoid output gate.
        const gated = if (gate) |g| blk: {
            const g_flat = if (prefill)
                try ops.reshape(ctx, g, &[_]i32{ 1, L, H * D })
            else
                try ops.reshape(ctx, g, &[_]i32{ 1, H * D });
            defer g_flat.deinit();
            const sig = try ops.sigmoid(ctx, g_flat);
            defer sig.deinit();
            break :blk try ops.multiply(ctx, attn_flat, sig);
        } else try ops.copy(ctx, attn_flat);
        defer gated.deinit();

        return self.linear(gated, self.o_proj, self.o_scales, self.o_biases, self.o_quant);
    }

    /// Test-only constructor: toy dims hidden=16, H=2, HKV=1, D=8, rotary=2.
    pub fn makeTestModule(allocator: std.mem.Allocator, ctx: EagerContext) !Qwen38Attn {
        const fill = struct {
            fn call(buf: []f32) void {
                for (buf, 0..) |*v, i| {
                    v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 37 + 11, 13) - 6)) * 0.02;
                }
            }
        }.call;
        const mk = struct {
            fn call(alloc: std.mem.Allocator, shape_: []const i32, fillFn: *const fn ([]f32) void) !Array {
                var n: usize = 1;
                for (shape_) |s| n *= @intCast(s);
                const buf = try alloc.alloc(f32, n);
                defer alloc.free(buf);
                fillFn(buf);
                return Array.fromData(alloc, f32, buf, shape_);
            }
        }.call;
        const norm_buf = try allocator.alloc(f32, 8);
        defer allocator.free(norm_buf);
        for (norm_buf, 0..) |*v, i| v.* = 1.0 + 0.01 * @as(f32, @floatFromInt(i));
        return .{
            .ctx = ctx,
            .hidden_size = 16,
            .num_heads = 2,
            .num_kv_heads = 1,
            .head_dim = 8,
            .rotary_dims = 2,
            .rope_theta = 10000.0,
            .rms_norm_eps = 1e-6,
            .q_proj = try mk(allocator, &[_]i32{ 2 * 2 * 8, 16 }, fill),
            .k_proj = try mk(allocator, &[_]i32{ 1 * 8, 16 }, fill),
            .v_proj = try mk(allocator, &[_]i32{ 1 * 8, 16 }, fill),
            .o_proj = try mk(allocator, &[_]i32{ 16, 2 * 8 }, fill),
            .q_norm = try Array.fromData(allocator, f32, norm_buf, &[_]i32{8}),
            .k_norm = try Array.fromData(allocator, f32, norm_buf, &[_]i32{8}),
        };
    }
};

test "Qwen38Attn.prefill: [1,3,hidden] forward, cache growth, decode after prefill" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    var attn = try Qwen38Attn.makeTestModule(allocator, ctx);
    defer attn.deinit();
    var cache = Qwen38AttnCache{};
    defer cache.deinit();

    var buf: [48]f32 = undefined;
    for (&buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 5, 11) - 5)) * 0.1;
    }
    const x = try Array.fromData(allocator, f32, &buf, &[_]i32{ 1, 3, 16 });
    defer x.deinit();

    const out = try attn.forward(x, &cache);
    defer out.deinit();
    try out.eval();
    const oshape = out.shape();
    try std.testing.expectEqual(@as(i32, 1), oshape[0]);
    try std.testing.expectEqual(@as(i32, 3), oshape[1]);
    try std.testing.expectEqual(@as(i32, 16), oshape[2]);
    try std.testing.expectEqual(@as(i32, 3), cache.len());

    // Decode one more token after prefill.
    var tok: [16]f32 = undefined;
    for (&tok, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 7 + 3, 11) - 5)) * 0.1;
    }
    const xt = try Array.fromData(allocator, f32, &tok, &[_]i32{ 1, 16 });
    defer xt.deinit();
    const out_t = try attn.forward(xt, &cache);
    defer out_t.deinit();
    try out_t.eval();
    const tshape = out_t.shape();
    try std.testing.expectEqual(@as(i32, 1), tshape[0]);
    try std.testing.expectEqual(@as(i32, 16), tshape[1]);
    try std.testing.expectEqual(@as(i32, 4), cache.len());
}
