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
const creation = @import("mlx").creation;

const closure_mod = @import("mlx").closure;
const compile_mod = @import("mlx").compile;
const qwen38_gdn = @import("qwen38_gdn.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const Closure = closure_mod.Closure;
const QuantConfig = quantize_mod.QuantConfig;

// ============================================================
// Decode pre-cache segment (projections → q/k norm → rope → gate sigmoid)
// ============================================================

/// Trace-time constants baked into the compiled pre-cache closure. The
/// serving model only enables the compiled path when the config matches.
pub const COMPILED_ROTARY_DIMS: i32 = 64;
pub const COMPILED_ROPE_THETA: f32 = 1e7;
pub const COMPILED_RMS_EPS: f32 = 1e-6;

/// All arrays one attention decode token needs up to the KV-cache read.
/// Shapes are fixed across layers and tokens (the rope offset is an Array
/// input, not a trace constant), so one compiled graph serves every layer.
pub const PreCacheWeights = struct {
    x: Array, // [1, hidden]
    q_w: Array,
    q_scales: ?Array = null,
    q_biases: ?Array = null,
    k_w: Array,
    k_scales: ?Array = null,
    k_biases: ?Array = null,
    v_w: Array,
    v_scales: ?Array = null,
    v_biases: ?Array = null,
    q_norm: Array, // [D]
    k_norm: Array, // [D]
    offset: Array, // scalar i32 (ropeDynamic)
};

pub const PreCacheOut = struct {
    q_t: Array, // [1, H, 1, D] roped
    k_t: Array, // [1, HKV, 1, D] roped
    v_t: Array, // [1, HKV, 1, D]
    gate_sig: ?Array, // [1, H*D] sigmoid(output gate)

    pub fn deinit(self: *PreCacheOut) void {
        self.q_t.deinit();
        self.k_t.deinit();
        self.v_t.deinit();
        if (self.gate_sig) |*g| g.deinit();
    }
};

/// Decode-only pre-cache segment as a pure function of arrays. Body mirrors
/// the decode branch of Qwen38Attn.forward exactly (same op sequence, same
/// dtypes), but takes every weight as an explicit input and reads the rope
/// offset via ropeDynamic so a compiled graph is offset-invariant. Head dims
/// are derived from the weight shapes; rotary_dims/theta/eps are supplied by
/// the caller (trace-time constants for the compiled closure).
pub fn preCacheStep(
    ctx: EagerContext,
    w: PreCacheWeights,
    rotary_dims: i32,
    rope_theta: f32,
    eps: f32,
    output_gate: bool,
) !PreCacheOut {
    const D: i32 = @intCast(w.q_norm.shape()[0]);
    const H: i32 = @intCast(@divTrunc(w.q_w.shape()[0], 2 * D));
    const HKV: i32 = @intCast(@divTrunc(w.k_w.shape()[0], D));

    const q_lin = try qwen38_gdn.linearQuantParts(ctx, w.x, w.q_w, w.q_scales, w.q_biases);
    defer q_lin.deinit();
    const q_resh = try ops.reshape(ctx, q_lin, &[_]i32{ 1, H, 2 * D });
    defer q_resh.deinit();
    const q_raw = try ops.slice(ctx, q_resh, &[_]i32{ 0, 0, 0 }, &[_]i32{ 1, H, D }, &[_]i32{ 1, 1, 1 });
    defer q_raw.deinit();

    const k_lin = try qwen38_gdn.linearQuantParts(ctx, w.x, w.k_w, w.k_scales, w.k_biases);
    defer k_lin.deinit();
    const k_resh = try ops.reshape(ctx, k_lin, &[_]i32{ 1, HKV, D });
    defer k_resh.deinit();
    const v_lin = try qwen38_gdn.linearQuantParts(ctx, w.x, w.v_w, w.v_scales, w.v_biases);
    defer v_lin.deinit();
    const v_resh = try ops.reshape(ctx, v_lin, &[_]i32{ 1, HKV, D });
    defer v_resh.deinit();

    const q_n = try fast_mod.rmsNorm(ctx, q_raw, w.q_norm, eps);
    defer q_n.deinit();
    const k_n = try fast_mod.rmsNorm(ctx, k_resh, w.k_norm, eps);
    defer k_n.deinit();

    const q_t0 = try ops.expandDims(ctx, q_n, 2); // [1, H, 1, D]
    defer q_t0.deinit();
    const k_t0 = try ops.expandDims(ctx, k_n, 2);
    defer k_t0.deinit();
    const v_t = try ops.expandDims(ctx, v_resh, 2);
    errdefer v_t.deinit();

    const q_t = try fast_mod.ropeDynamic(ctx, q_t0, rotary_dims, false, rope_theta, 1.0, w.offset, null);
    errdefer q_t.deinit();
    const k_t = try fast_mod.ropeDynamic(ctx, k_t0, rotary_dims, false, rope_theta, 1.0, w.offset, null);
    errdefer k_t.deinit();

    var gate_sig: ?Array = null;
    errdefer if (gate_sig) |*g| g.deinit();
    if (output_gate) {
        const gate = try ops.slice(ctx, q_resh, &[_]i32{ 0, 0, D }, &[_]i32{ 1, H, 2 * D }, &[_]i32{ 1, 1, 1 });
        defer gate.deinit();
        const g_flat = try ops.reshape(ctx, gate, &[_]i32{ 1, H * D });
        defer g_flat.deinit();
        gate_sig = try ops.sigmoid(ctx, g_flat);
    }
    return .{ .q_t = q_t, .k_t = k_t, .v_t = v_t, .gate_sig = gate_sig };
}

/// Closures are traced on the GPU stream so the compiled graph dispatches to
/// the same device as the eager path.
fn closureGpuCtx(allocator: std.mem.Allocator) EagerContext {
    return EagerContext.initWithStream(allocator, .{ .inner = c.c.mlx_default_gpu_stream_new() });
}

fn preCacheToSlice(res: PreCacheOut, allocator: std.mem.Allocator) error{MlxError}![]Array {
    const out = allocator.alloc(Array, 4) catch return error.MlxError;
    out[0] = res.q_t;
    out[1] = res.k_t;
    out[2] = res.v_t;
    out[3] = res.gate_sig orelse return error.MlxError;
    return out;
}

/// Quantized real-config variant. Input order (13): x, q_w, q_s, q_b, k_w,
/// k_s, k_b, v_w, v_s, v_b, q_norm, k_norm, offset. Returns
/// [q_t, k_t, v_t, gate_sig].
fn attnPreCacheClosureQuant(inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array {
    if (inputs.len != 13) return error.MlxError;
    const ctx = closureGpuCtx(allocator);
    defer ctx.deinit();
    const res = preCacheStep(ctx, .{
        .x = inputs[0],
        .q_w = inputs[1],
        .q_scales = inputs[2],
        .q_biases = inputs[3],
        .k_w = inputs[4],
        .k_scales = inputs[5],
        .k_biases = inputs[6],
        .v_w = inputs[7],
        .v_scales = inputs[8],
        .v_biases = inputs[9],
        .q_norm = inputs[10],
        .k_norm = inputs[11],
        .offset = inputs[12],
    }, COMPILED_ROTARY_DIMS, COMPILED_ROPE_THETA, COMPILED_RMS_EPS, true) catch return error.MlxError;
    return preCacheToSlice(res, allocator);
}

/// Build the mx.compile'd attention pre-cache segment for the real
/// (quantized, rotary_dims=64, theta=1e7, eps=1e-6, output gate) config.
/// Trace happens on first apply — warm once before serving.
pub fn compiledPreCache(allocator: std.mem.Allocator) !Closure {
    const base = try Closure.init(&attnPreCacheClosureQuant, allocator);
    defer base.deinit();
    return compile_mod.compile(base, false);
}

/// Slab-allocated KV cache: keys/values are [1, n_kv, capacity, head_dim]
/// buffers grown in 256-token slabs (capacity doubles when a grow is needed,
/// so appends are amortized O(1)); `len` tracks the number of valid tokens.
pub const Qwen38AttnCache = struct {
    pub const slab: i32 = 256;

    keys: ?Array = null, // [1, n_kv, capacity, head_dim], len tokens valid
    values: ?Array = null,
    len_: i32 = 0,

    pub fn deinit(self: *Qwen38AttnCache) void {
        if (self.keys) |*k| k.deinit();
        if (self.values) |*v| v.deinit();
        self.keys = null;
        self.values = null;
        self.len_ = 0;
    }

    pub fn len(self: *const Qwen38AttnCache) i32 {
        return self.len_;
    }

    fn capacity(self: *const Qwen38AttnCache) i32 {
        const k = self.keys orelse return 0;
        return @intCast(k.shape()[2]);
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

/// Slab cache write: grow capacity (doubling, 256-token granularity) if
/// needed, then one slice_update per tensor at [len, len+L). sliceUpdate
/// returns a new array; swap it in and free the old buffer. On failure
/// the cache stays consistent because `len` only advances at the end.
fn cacheAppend(ctx: EagerContext, cache: *Qwen38AttnCache, k_t: Array, v_t: Array, L: i32, HKV: i32, D: i32) !void {
    const needed: i32 = cache.len_ + L;
    if (needed > cache.capacity()) {
        var newcap: i32 = @max(cache.capacity() * 2, Qwen38AttnCache.slab);
        while (newcap < needed) newcap += Qwen38AttnCache.slab;
        const grow: i32 = newcap - cache.capacity();
        if (cache.keys == null) {
            const zk = try creation.zeros(ctx, &[_]i32{ 1, HKV, newcap, D }, k_t.dtype());
            errdefer zk.deinit();
            const zv = try creation.zeros(ctx, &[_]i32{ 1, HKV, newcap, D }, v_t.dtype());
            errdefer zv.deinit();
            cache.keys = zk;
            cache.values = zv;
        } else {
            const zk = try creation.zeros(ctx, &[_]i32{ 1, HKV, grow, D }, k_t.dtype());
            defer zk.deinit();
            const zv = try creation.zeros(ctx, &[_]i32{ 1, HKV, grow, D }, v_t.dtype());
            defer zv.deinit();
            const k_grown = try shape_mod.concatenateAxis(ctx, &[_]Array{ cache.keys.?, zk }, 2);
            errdefer k_grown.deinit();
            const v_grown = try shape_mod.concatenateAxis(ctx, &[_]Array{ cache.values.?, zv }, 2);
            errdefer v_grown.deinit();
            cache.keys.?.deinit();
            cache.values.?.deinit();
            cache.keys = k_grown;
            cache.values = v_grown;
        }
    }
    const start = &[_]i32{ 0, 0, cache.len_, 0 };
    const stop = &[_]i32{ 1, HKV, needed, D };
    const ones = &[_]i32{ 1, 1, 1, 1 };
    const k_new = try shape_mod.sliceUpdate(ctx, cache.keys.?, k_t, start, stop, ones);
    cache.keys.?.deinit();
    cache.keys = k_new;
    const v_new = try shape_mod.sliceUpdate(ctx, cache.values.?, v_t, start, stop, ones);
    cache.values.?.deinit();
    cache.values = v_new;
    cache.len_ = needed;
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

        // Decode (L=1): shared pre-cache segment (ropeDynamic — numerically
        // equal to static rope, verified by unit test) + eager remainder.
        if (!prefill) {
            const offset_arr = try ops.scalarI32(ctx, offset);
            defer offset_arr.deinit();
            var pre = try preCacheStep(ctx, .{
                .x = x,
                .q_w = self.q_proj,
                .q_scales = self.q_scales,
                .q_biases = self.q_biases,
                .k_w = self.k_proj,
                .k_scales = self.k_scales,
                .k_biases = self.k_biases,
                .v_w = self.v_proj,
                .v_scales = self.v_scales,
                .v_biases = self.v_biases,
                .q_norm = self.q_norm,
                .k_norm = self.k_norm,
                .offset = offset_arr,
            }, @intCast(self.rotary_dims), self.rope_theta, self.rms_norm_eps, self.output_gate);
            defer pre.deinit();
            return self.postCache(pre, cache);
        }

        // q path: [1, L, H*2*D] -> [1, L, H, 2*D] -> chunk into q / gate.
        const q_lin = try self.linear(x, self.q_proj, self.q_scales, self.q_biases, self.q_quant);
        defer q_lin.deinit();
        const q_resh = try ops.reshape(ctx, q_lin, &[_]i32{ 1, L, H, 2 * D });
        defer q_resh.deinit();
        const q_raw = try ops.slice(ctx, q_resh, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, L, H, D }, &[_]i32{ 1, 1, 1, 1 });
        defer q_raw.deinit();
        const gate = if (self.output_gate) blk: {
            const g = try ops.slice(ctx, q_resh, &[_]i32{ 0, 0, 0, D }, &[_]i32{ 1, L, H, 2 * D }, &[_]i32{ 1, 1, 1, 1 });
            break :blk g;
        } else null;
        defer if (gate) |g| g.deinit();

        const k_lin = try self.linear(x, self.k_proj, self.k_scales, self.k_biases, self.k_quant);
        defer k_lin.deinit();
        const v_lin = try self.linear(x, self.v_proj, self.v_scales, self.v_biases, self.v_quant);
        defer v_lin.deinit();
        const k_resh = try ops.reshape(ctx, k_lin, &[_]i32{ 1, L, HKV, D });
        defer k_resh.deinit();
        const v_resh = try ops.reshape(ctx, v_lin, &[_]i32{ 1, L, HKV, D });
        defer v_resh.deinit();

        // Per-head RMSNorm, then partial RoPE (dims = rotary_dims).
        // mx.fast.rope treats axis -2 as the sequence position, so transpose
        // to [1, heads, L, D] BEFORE rope (oracle order: norm -> transpose ->
        // rope); [1, L, H, D] would rotate over the head axis instead.
        const q_n = try fast_mod.rmsNorm(ctx, q_raw, self.q_norm, self.rms_norm_eps);
        defer q_n.deinit();
        const k_n = try fast_mod.rmsNorm(ctx, k_resh, self.k_norm, self.rms_norm_eps);
        defer k_n.deinit();

        // Transpose to [1, heads, L, D].
        const q_t0 = try ops.transposeAxes(ctx, q_n, &[_]i32{ 0, 2, 1, 3 });
        defer q_t0.deinit();
        const k_t0 = try ops.transposeAxes(ctx, k_n, &[_]i32{ 0, 2, 1, 3 });
        defer k_t0.deinit();
        const v_t = try ops.transposeAxes(ctx, v_resh, &[_]i32{ 0, 2, 1, 3 });
        defer v_t.deinit();

        const rd: i32 = @intCast(self.rotary_dims);
        const q_t = try fast_mod.rope(ctx, q_t0, rd, false, self.rope_theta, 1.0, offset, null);
        defer q_t.deinit();
        const k_t = try fast_mod.rope(ctx, k_t0, rd, false, self.rope_theta, 1.0, offset, null);
        defer k_t.deinit();

        // Slab cache write (see postCache).
        try cacheAppend(ctx, cache, k_t, v_t, L, HKV, D);

        // SDPA reads a [1, HKV, len, D] view of the slab — no copy; MLX
        // scaled_dot_product_attention handles GQA (H % HKV == 0) natively.
        const kv_len: i32 = cache.len();
        const k_view = try ops.slice(ctx, cache.keys.?, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, HKV, kv_len, D }, &[_]i32{ 1, 1, 1, 1 });
        defer k_view.deinit();
        const v_view = try ops.slice(ctx, cache.values.?, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, HKV, kv_len, D }, &[_]i32{ 1, 1, 1, 1 });
        defer v_view.deinit();

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const attn = try fast_mod.scaledDotProductAttention(
            ctx,
            q_t,
            k_view,
            v_view,
            scale,
            "causal",
            null,
            null,
        );
        defer attn.deinit(); // [1, H, L, D]

        // Back to [1, L, H*D].
        const attn_t = try ops.transposeAxes(ctx, attn, &[_]i32{ 0, 2, 1, 3 });
        defer attn_t.deinit();
        const attn_flat = try ops.reshape(ctx, attn_t, &[_]i32{ 1, L, H * D });
        defer attn_flat.deinit();

        // Sigmoid output gate.
        const gated = if (gate) |g| blk: {
            const g_flat = try ops.reshape(ctx, g, &[_]i32{ 1, L, H * D });
            defer g_flat.deinit();
            const sig = try ops.sigmoid(ctx, g_flat);
            defer sig.deinit();
            break :blk try ops.multiply(ctx, attn_flat, sig);
        } else try ops.copy(ctx, attn_flat);
        defer gated.deinit();

        return self.linear(gated, self.o_proj, self.o_scales, self.o_biases, self.o_quant);
    }

    /// Decode-only eager remainder after the pre-cache segment: KV slab
    /// append, SDPA, gate multiply, o_proj. Consumes `pre` (the caller still
    /// owns/deinits it). Returns [1, hidden].
    pub fn postCache(self: *Qwen38Attn, pre: PreCacheOut, cache: *Qwen38AttnCache) !Array {
        const ctx = self.ctx;
        const H: i32 = @intCast(self.num_heads);
        const HKV: i32 = @intCast(self.num_kv_heads);
        const D: i32 = @intCast(self.head_dim);

        try cacheAppend(ctx, cache, pre.k_t, pre.v_t, 1, HKV, D);
        // No eval here: the caller (Qwen38ServingModel.forwardTokens) submits
        // the updated cache states via asyncEval once per token.

        const kv_len: i32 = cache.len();
        const k_view = try ops.slice(ctx, cache.keys.?, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, HKV, kv_len, D }, &[_]i32{ 1, 1, 1, 1 });
        defer k_view.deinit();
        const v_view = try ops.slice(ctx, cache.values.?, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, HKV, kv_len, D }, &[_]i32{ 1, 1, 1, 1 });
        defer v_view.deinit();

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const attn = try fast_mod.scaledDotProductAttention(ctx, pre.q_t, k_view, v_view, scale, "", null, null);
        defer attn.deinit(); // [1, H, 1, D]

        const attn_t = try ops.squeeze(ctx, attn);
        defer attn_t.deinit();
        const attn_flat = try ops.reshape(ctx, attn_t, &[_]i32{ 1, H * D });
        defer attn_flat.deinit();

        // Output gate (sigmoid already applied in the pre-cache segment).
        const gated = if (pre.gate_sig) |g|
            try ops.multiply(ctx, attn_flat, g)
        else
            try ops.copy(ctx, attn_flat);
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

test "Qwen38Attn.slab: 260-token decode crosses slab boundary, matches prefill" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const N = 260; // crosses the 256-token slab boundary
    const HIDDEN = 16;

    // Deterministic per-position inputs.
    var buf: [N * HIDDEN]f32 = undefined;
    for (&buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 5, 11) - 5)) * 0.1;
    }

    // Decode path: one token at a time.
    var attn_d = try Qwen38Attn.makeTestModule(allocator, ctx);
    defer attn_d.deinit();
    var cache_d = Qwen38AttnCache{};
    defer cache_d.deinit();
    var dec_outs: [N]Array = undefined;
    for (0..N) |i| {
        const xt = try Array.fromData(allocator, f32, buf[i * HIDDEN ..][0..HIDDEN], &[_]i32{ 1, HIDDEN });
        defer xt.deinit();
        dec_outs[i] = try attn_d.forward(xt, &cache_d);
        try dec_outs[i].eval();
    }
    defer for (&dec_outs) |*o| o.deinit();
    try std.testing.expectEqual(@as(i32, N), cache_d.len());
    // Slab growth: capacity must be >= N and a multiple of the slab size.
    const cap = cache_d.keys.?.shape()[2];
    try std.testing.expect(cap >= N);
    try std.testing.expectEqual(@as(i32, 0), @mod(cap, Qwen38AttnCache.slab));

    // Prefill path: all N tokens at once on a fresh module.
    var attn_p = try Qwen38Attn.makeTestModule(allocator, ctx);
    defer attn_p.deinit();
    var cache_p = Qwen38AttnCache{};
    defer cache_p.deinit();
    const xp = try Array.fromData(allocator, f32, &buf, &[_]i32{ 1, N, HIDDEN });
    defer xp.deinit();
    const out_p = try attn_p.forward(xp, &cache_p);
    defer out_p.deinit();
    try out_p.eval();
    try std.testing.expectEqual(@as(i32, N), cache_p.len());

    // Each decode step output must match the corresponding prefill row.
    for (0..N) |i| {
        const ii: i32 = @intCast(i);
        const row = try ops.slice(ctx, out_p, &[_]i32{ 0, ii, 0 }, &[_]i32{ 1, ii + 1, HIDDEN }, &[_]i32{ 1, 1, 1 });
        defer row.deinit();
        const row_flat = try ops.reshape(ctx, row, &[_]i32{ 1, HIDDEN });
        defer row_flat.deinit();
        const diff = try ops.subtract(ctx, dec_outs[i], row_flat);
        defer diff.deinit();
        const abs_diff = try ops.abs(ctx, diff);
        defer abs_diff.deinit();
        const max_diff = try reduce_mod.max(ctx, abs_diff, false);
        defer max_diff.deinit();
        try max_diff.eval();
        try std.testing.expect(try max_diff.item(f32) <= 1e-5);
    }
}

test "ropeDynamic: scalar-Array offset matches static rope" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    var buf: [16]f32 = undefined;
    for (&buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 7 + 3, 11) - 5)) * 0.1;
    }
    const x = try Array.fromData(allocator, f32, &buf, &[_]i32{ 1, 2, 1, 8 });
    defer x.deinit();

    const stat = try fast_mod.rope(ctx, x, 2, false, 10000.0, 1.0, 7, null);
    defer stat.deinit();
    const offset_arr = try ops.scalarI32(ctx, 7);
    defer offset_arr.deinit();
    const dyn = try fast_mod.ropeDynamic(ctx, x, 2, false, 10000.0, 1.0, offset_arr, null);
    defer dyn.deinit();

    const diff = try ops.subtract(ctx, stat, dyn);
    defer diff.deinit();
    const abs_diff = try ops.abs(ctx, diff);
    defer abs_diff.deinit();
    const max_diff = try reduce_mod.max(ctx, abs_diff, false);
    defer max_diff.deinit();
    try max_diff.eval();
    try std.testing.expect(try max_diff.item(f32) <= 1e-6);
}

/// Toy-config (plain, rotary=2, theta=1e4, eps=1e-6, output gate) closure for
/// the compiled-segment unit test. Input order (7): x, q_w, k_w, v_w, q_norm,
/// k_norm, offset. Returns [q_t, k_t, v_t, gate_sig].
fn attnPreCacheClosurePlainTest(inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array {
    if (inputs.len != 7) return error.MlxError;
    const ctx = closureGpuCtx(allocator);
    defer ctx.deinit();
    const res = preCacheStep(ctx, .{
        .x = inputs[0],
        .q_w = inputs[1],
        .k_w = inputs[2],
        .v_w = inputs[3],
        .q_norm = inputs[4],
        .k_norm = inputs[5],
        .offset = inputs[6],
    }, 2, 10000.0, 1e-6, true) catch return error.MlxError;
    return preCacheToSlice(res, allocator);
}

test "compiled pre-cache segment matches eager preCacheStep" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    var attn = try Qwen38Attn.makeTestModule(allocator, ctx);
    defer attn.deinit();

    var buf: [16]f32 = undefined;
    for (&buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 5, 11) - 5)) * 0.1;
    }
    const x = try Array.fromData(allocator, f32, &buf, &[_]i32{ 1, 16 });
    defer x.deinit();
    const offset_arr = try ops.scalarI32(ctx, 7);
    defer offset_arr.deinit();

    var eager = try preCacheStep(ctx, .{
        .x = x,
        .q_w = attn.q_proj,
        .k_w = attn.k_proj,
        .v_w = attn.v_proj,
        .q_norm = attn.q_norm,
        .k_norm = attn.k_norm,
        .offset = offset_arr,
    }, 2, 10000.0, 1e-6, true);
    defer eager.deinit();

    const base = try Closure.init(&attnPreCacheClosurePlainTest, allocator);
    defer base.deinit();
    const cc = try compile_mod.compile(base, false);
    defer cc.deinit();
    const outs = try cc.apply(&[_]Array{ x, attn.q_proj, attn.k_proj, attn.v_proj, attn.q_norm, attn.k_norm, offset_arr }, allocator);
    defer allocator.free(outs);
    defer for (outs) |o| o.deinit();
    try std.testing.expectEqual(@as(usize, 4), outs.len);

    const compiled_outs = [_]Array{ outs[0], outs[1], outs[2], outs[3] };
    const eager_outs = [_]Array{ eager.q_t, eager.k_t, eager.v_t, eager.gate_sig.? };
    for (compiled_outs, eager_outs) |co, eo| {
        const diff = try ops.subtract(ctx, co, eo);
        defer diff.deinit();
        const abs_diff = try ops.abs(ctx, diff);
        defer abs_diff.deinit();
        const max_diff = try reduce_mod.max(ctx, abs_diff, false);
        defer max_diff.deinit();
        try max_diff.eval();
        try std.testing.expect(try max_diff.item(f32) <= 1e-5);
    }

    // Offset-invariance: a second apply with a different offset must NOT
    // retrace into a different graph semantics — outputs match eager at 3.
    const offset3 = try ops.scalarI32(ctx, 3);
    defer offset3.deinit();
    var eager3 = try preCacheStep(ctx, .{
        .x = x,
        .q_w = attn.q_proj,
        .k_w = attn.k_proj,
        .v_w = attn.v_proj,
        .q_norm = attn.q_norm,
        .k_norm = attn.k_norm,
        .offset = offset3,
    }, 2, 10000.0, 1e-6, true);
    defer eager3.deinit();
    const outs3 = try cc.apply(&[_]Array{ x, attn.q_proj, attn.k_proj, attn.v_proj, attn.q_norm, attn.k_norm, offset3 }, allocator);
    defer allocator.free(outs3);
    defer for (outs3) |o| o.deinit();
    const d3 = try ops.subtract(ctx, outs3[0], eager3.q_t);
    defer d3.deinit();
    const a3 = try ops.abs(ctx, d3);
    defer a3.deinit();
    const m3 = try reduce_mod.max(ctx, a3, false);
    defer m3.deinit();
    try m3.eval();
    try std.testing.expect(try m3.item(f32) <= 1e-5);
}
