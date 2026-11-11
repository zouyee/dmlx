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
const eval_mod = @import("mlx").eval;
const qwen38_gdn = @import("qwen38_gdn.zig");
const verify_qmm = @import("verify_qmm.zig");
const qmm_v8 = @import("qwen38_qmm_v8.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const Closure = closure_mod.Closure;
const QuantConfig = quantize_mod.QuantConfig;
const dtype_mod = @import("mlx").dtype;

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

test "kvInplaceWrite: bf16 in-place slab append matches expected layout" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    if (kvInplaceDisabled()) {
        // Force-enable for this test (the feature is opt-in by default).
        kv_inplace_on = true;
        kv_inplace_broken = false;
        if (kvInplaceDisabled()) return error.SkipTest;
    }
    // The kernel mutates the slab asynchronously on the GPU stream; reads
    // must be enqueued on the SAME stream to be ordered after the write
    // (a CPU-stream ctx would race and read stale data).
    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    const HKV: i32 = 2;
    const D: i32 = 4;
    const CAP: i32 = 8;
    const slab = try creation.zeros(ctx, &[_]i32{ 1, HKV, CAP, D }, .bfloat16);
    defer slab.deinit();

    // rows1: [1, 2, 3, 4] with distinct values v = hk*100 + l*10 + d
    var buf1: [2 * 3 * 4]f32 = undefined;
    for (0..2) |hk| for (0..3) |l| for (0..4) |d| {
        buf1[hk * 12 + l * 4 + d] = @floatFromInt(hk * 100 + l * 10 + d);
    };
    const r1f = try Array.fromData(allocator, f32, &buf1, &[_]i32{ 1, HKV, 3, D });
    defer r1f.deinit();
    const rows1 = try ops.astype(ctx, r1f, .bfloat16);
    defer rows1.deinit();

    try std.testing.expect(try kvInplaceWrite(ctx, slab, rows1, 2, 3, HKV, D));

    // Interleave a dependent read between two writes to exercise ordering:
    // read back slab[0, 1, 2..5, :] and check rows1's head-1 content landed.
    {
        const seg = try ops.slice(ctx, slab, &[_]i32{ 0, 1, 2, 0 }, &[_]i32{ 1, 2, 5, D }, &[_]i32{ 1, 1, 1, 1 });
        defer seg.deinit();
        const segf = try ops.astype(ctx, seg, .float32);
        defer segf.deinit();
        try segf.eval();
        const sv = try segf.dataSlice(f32);
        // rows1[0, 1, 0..3, :] = [100..103, 110..113, 120..123]
        try std.testing.expectApproxEqAbs(@as(f32, 100), sv[0], 1e-2);
        try std.testing.expectApproxEqAbs(@as(f32, 123), sv[11], 1e-2);
    }

    var buf2: [2 * 2 * 4]f32 = undefined;
    for (0..2) |hk| for (0..2) |l| for (0..4) |d| {
        buf2[hk * 8 + l * 4 + d] = @floatFromInt(1000 + hk * 100 + l * 10 + d);
    };
    const r2f = try Array.fromData(allocator, f32, &buf2, &[_]i32{ 1, HKV, 2, D });
    defer r2f.deinit();
    const rows2 = try ops.astype(ctx, r2f, .bfloat16);
    defer rows2.deinit();

    try std.testing.expect(try kvInplaceWrite(ctx, slab, rows2, 5, 2, HKV, D));

    // Read back exact regions.
    const got1 = try ops.slice(ctx, slab, &[_]i32{ 0, 1, 3, 0 }, &[_]i32{ 1, 2, 4, D }, &[_]i32{ 1, 1, 1, 1 });
    defer got1.deinit();
    const got32 = try ops.astype(ctx, got1, .float32);
    defer got32.deinit();
    try got32.eval();
    // slab[0, 1, 3, :] == rows1[0, 1, 1, :] == [110, 111, 112, 113]
    const g1 = try got32.dataSlice(f32);
    try std.testing.expectApproxEqAbs(@as(f32, 110), g1[0], 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 113), g1[3], 1e-2);

    const got2 = try ops.slice(ctx, slab, &[_]i32{ 0, 0, 5, 0 }, &[_]i32{ 1, 1, 6, D }, &[_]i32{ 1, 1, 1, 1 });
    defer got2.deinit();
    const got2f = try ops.astype(ctx, got2, .float32);
    defer got2f.deinit();
    try got2f.eval();
    const g2 = try got2f.dataSlice(f32);
    try std.testing.expectApproxEqAbs(@as(f32, 1000), g2[0], 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1003), g2[3], 1.0);

    // Untouched region stays zero.
    const gotz = try ops.slice(ctx, slab, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, 1, 2, D }, &[_]i32{ 1, 1, 1, 1 });
    defer gotz.deinit();
    const gotzf = try ops.astype(ctx, gotz, .float32);
    defer gotzf.deinit();
    try gotzf.eval();
    const gz = try gotzf.dataSlice(f32);
    for (gz) |v| try std.testing.expectEqual(@as(f32, 0), v);
}

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

// ============================================================
// In-place KV slab append (OPT-IN, DMLX_KV_INPLACE=1; default OFF)
//
// slice_update is functional: every append copies the whole slab
// (2 x [HKV, cap, D] per layer per token — ~1 GB per token at 16K ctx
// across 16 layers). This kernel writes only the new L rows into the
// existing slab buffer instead. MEASURED NEUTRAL-TO-NEGATIVE at 16K MTP
// (see kvInplaceDisabled): the functional copies overlap compute, so the
// O(cap) write traffic is not on the critical path. The kernel remains
// correct (bit-identical outputs) and is kept as the reference pattern
// for eager in-place mutation: the Metal input arrives as `const device`
// (const_cast is deliberate), and mlx_fast_metal_kernel_apply is LAZY —
// the sentinel output must be asyncEval'd for the write to be enqueued.
// Ordering rests on all custom kernels and MLX evals sharing the default
// GPU stream (fusedStream()).
// ============================================================

const kv_inplace_body =
    \\    const uint3 gid = thread_position_in_grid;
    \\    const uint offset = params[0];
    \\    const uint cap = params[1];
    \\    const uint L = params[2];
    \\    const uint D = params[3];
    \\    device bfloat16_t* slab_w = const_cast<device bfloat16_t*>(slab);
    \\    slab_w[(gid.z * cap + offset + gid.y) * D + gid.x] = rows[(gid.z * L + gid.y) * D + gid.x];
    \\    if (gid.x == 0 and gid.y == 0 and gid.z == 0) sentinel[0] = 1.0f;
;

var kv_inplace_kernel: ?c.c.mlx_fast_metal_kernel = null;
var kv_inplace_broken: bool = false;
var kv_inplace_on: ?bool = null;

/// OPT-IN (DMLX_KV_INPLACE=1, default OFF). Measured neutral-to-slightly-
/// negative at 16K MTP (2026-11-11, scripts/ab_kvinplace_16k.sh: verify
/// 157.1/109.3 vs 141.4/106.7 ms/round, wall 112.6/109.3 vs 110.9/108.8s,
/// output text bit-identical): the slice_update slab copies overlap compute
/// and are not on the critical path — bandwidth math does not translate to
/// wall time here, at any context length (copy and SDPA both scale ~T).
/// Kept as a reference implementation of eager in-place mutation via
/// const_cast + sentinel asyncEval; do not extend without new evidence.
fn kvInplaceDisabled() bool {
    if (kv_inplace_on == null) {
        kv_inplace_on = if (std.c.getenv("DMLX_KV_INPLACE")) |v| v[0] == '1' else false;
    }
    return !kv_inplace_on.?;
}

fn getKvInplaceKernel(allocator: std.mem.Allocator) ?c.c.mlx_fast_metal_kernel {
    if (kv_inplace_broken) return null;
    if (kv_inplace_kernel) |k| return k;
    const name_z = allocator.dupeZ(u8, "dmlx_kv_inplace_append") catch return null;
    defer allocator.free(name_z);
    const src_z = allocator.dupeZ(u8, kv_inplace_body) catch return null;
    defer allocator.free(src_z);
    const header_z = allocator.dupeZ(u8, "") catch return null;
    defer allocator.free(header_z);
    const in_vec = qwen38_gdn.makeStringVec(allocator, &.{ "slab", "rows", "params" }) catch return null;
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = qwen38_gdn.makeStringVec(allocator, &.{"sentinel"}) catch return null;
    defer _ = c.c.mlx_vector_string_free(out_vec);
    const k = c.c.mlx_fast_metal_kernel_new(name_z.ptr, in_vec, out_vec, src_z.ptr, header_z.ptr, true, false);
    kv_inplace_kernel = k;
    return k;
}

var kv_inplace_logged: bool = false;

/// Write rows [1, HKV, L, D] into slab [1, HKV, cap, D] at [.., off..off+L, :]
/// without copying the slab. Returns false on any shape/dtype/layout
/// mismatch (caller falls back to slice_update).
fn kvInplaceWrite(ctx: EagerContext, slab: Array, rows: Array, off: i32, L: i32, HKV: i32, D: i32) !bool {
    if (kvInplaceDisabled() or kv_inplace_broken) return false;
    const dt = dtype_mod.Dtype.bfloat16;
    if (slab.dtype() != dt or rows.dtype() != dt) return false;
    const cap: i32 = @intCast(slab.shape()[2]);
    if (!isContiguous4D(slab, 1, HKV, cap, D)) return false;
    if (!isContiguous4D(rows, 1, HKV, L, D)) return false;
    const kernel = getKvInplaceKernel(ctx.allocator) orelse return false;

    const params = try Array.fromData(ctx.allocator, u32, &[_]u32{ @intCast(off), @intCast(cap), @intCast(L), @intCast(D) }, &[_]i32{4});
    defer params.deinit();

    const cfg = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg);
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(D), @intCast(L), @intCast(HKV)));
    const tg: c_int = @min(D, 128);
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{1}, 1, c.c.MLX_FLOAT32));

    const in_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(in_vec);
    for ([_]Array{ slab, rows, params }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(in_vec, inp.inner));
    }
    var out_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(out_vec);
    c.check(c.c.mlx_fast_metal_kernel_apply(&out_vec, kernel, in_vec, cfg, qwen38_gdn.fusedStream())) catch |err| {
        kv_inplace_broken = true;
        sdpaDecodeWarn(err);
        return false;
    };
    // mlx_fast_metal_kernel_apply is LAZY: the kernel only runs when the
    // outputs are evaluated. Submit the sentinel (non-blocking) so the
    // slab write is enqueued on the GPU stream now — subsequent reads of
    // the slab on the same stream are ordered after it.
    if (c.c.mlx_vector_array_size(out_vec) > 0) {
        var h = c.c.mlx_array_new();
        try c.check(c.c.mlx_vector_array_get(&h, out_vec, 0));
        var sentinel = Array.fromHandle(h);
        defer sentinel.deinit();
        try eval_mod.asyncEval(ctx.allocator, &[_]Array{sentinel});
    }
    if (!kv_inplace_logged) {
        kv_inplace_logged = true;
        std.log.info("[kv-inplace] engaged (in-place slab append, no slice_update copy)", .{});
    }
    return true;
}

/// Slab cache write: grow capacity (doubling, 256-token granularity) if
/// needed, then write the new rows at [len, len+L) — in-place via
/// kvInplaceWrite when possible (O(L), no copy), else one slice_update
/// per tensor (functional, copies the whole slab). On failure
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
    // In-place slab write: O(L) instead of the O(cap) slice_update copy.
    // Rows that are not contiguous get a cheap same-shape copy first
    // (rope/transposed views; at most [HKV, L, D]).
    var k_owned: ?Array = null;
    defer if (k_owned) |*o| o.deinit();
    var v_owned: ?Array = null;
    defer if (v_owned) |*o| o.deinit();
    const k_c = if (isContiguous4D(k_t, 1, HKV, L, D)) k_t else blk: {
        k_owned = try ops.copy(ctx, k_t);
        break :blk k_owned.?;
    };
    const v_c = if (isContiguous4D(v_t, 1, HKV, L, D)) v_t else blk: {
        v_owned = try ops.copy(ctx, v_t);
        break :blk v_owned.?;
    };
    if (try kvInplaceWrite(ctx, cache.keys.?, k_c, cache.len_, L, HKV, D) and
        try kvInplaceWrite(ctx, cache.values.?, v_c, cache.len_, L, HKV, D))
    {
        cache.len_ = needed;
        return;
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

// ============================================================
// Fused decode SDPA (qL=1) — port of omlx sdpa_decode_kernels.h
// `omlx_sdpa_decode` (itself a port of ml-explore/mlx PR #4294)
// ============================================================
//
// One threadgroup per query head (1024 threads = 32 simdgroups); the KV
// sequence is striped across simdgroups in TK=4-key tiles; per thread
// qk_per_thread = D/32 elements; Q·K scores are scalar FMA + simd_sum; tiled
// online softmax rescales the accumulator only when the running max
// increases; fp32 accumulation throughout; widest aligned vectorized loads
// (bf16 → metal::vec<bfloat16_t, 8>). KV must be last-dim-contiguous with
// 16B-aligned rows (checked by the gate). Decode attends to all cached KV,
// so no causal/mask path. B=1 and qL=1 only.
//
// Follow-up note: the 2-pass split-KV variant for long context
// (kv_len >= 1024) is implemented below (sdpaDecode2Pass).

const sdpa_decode_body =
    \\    const uint q_head = threadgroup_position_in_grid.x;
    \\    const uint sg = simdgroup_index_in_threadgroup; // 0..31
    \\    const uint lane = thread_index_in_simdgroup; // 0..31
    \\    const int N = n_in[0];
    \\    const int khs = k_head_stride_in[0];
    \\    const int kss = k_seq_stride_in[0];
    \\    const int vhs = v_head_stride_in[0];
    \\    const int vss = v_seq_stride_in[0];
    \\
    \\    const int kv_head = q_head / {[gqa]}; // GQA: kv head = q head / factor
    \\    const device bfloat16_t* qp = q + q_head * {[d]} + lane * {[qpt]};
    \\    const device bfloat16_t* kp = keys + kv_head * khs + sg * 4 * kss + lane * {[qpt]};
    \\    const device bfloat16_t* vp = values + kv_head * vhs + sg * 4 * vss + lane * {[qpt]};
    \\
    \\    // Read the query (scaled) and zero the output accumulator.
    \\    float qq[{[qpt]}];
    \\    {{
    \\        const thread metal::vec<bfloat16_t, {[qpt]}> qt = *(const device metal::vec<bfloat16_t, {[qpt]}>*)qp;
    \\        for (int j = 0; j < {[qpt]}; ++j) qq[j] = {[scale]s} * float(qt[j]);
    \\    }}
    \\    float o[{[qpt]}];
    \\    for (int j = 0; j < {[qpt]}; ++j) o[j] = 0.0f;
    \\    float max_score = -3.402823466e+38f;
    \\    float sum_exp = 0.0f;
    \\
    \\    // Main loop: this simdgroup's stripe of keys, 4-key tiles.
    \\    int i = sg * 4;
    \\    for (; i + 4 <= N; i += 128) {{
    \\        bfloat16_t ks[4][{[qpt]}];
    \\        for (int t = 0; t < 4; ++t) {{
    \\            const thread metal::vec<bfloat16_t, {[qpt]}> kv = *(const device metal::vec<bfloat16_t, {[qpt]}>*)(kp + t * kss);
    \\            for (int j = 0; j < {[qpt]}; ++j) ks[t][j] = kv[j];
    \\        }}
    \\        float scores[4];
    \\        for (int t = 0; t < 4; ++t) {{
    \\            float s = 0.0f;
    \\            for (int j = 0; j < {[qpt]}; ++j) s += qq[j] * float(ks[t][j]);
    \\            scores[t] = metal::simd_sum(s);
    \\        }}
    \\        const float tile_max = metal::max(metal::max(scores[0], scores[1]), metal::max(scores[2], scores[3]));
    \\        const float new_max = metal::max(max_score, tile_max);
    \\
    \\        bfloat16_t vs[4][{[qpt]}];
    \\        for (int t = 0; t < 4; ++t) {{
    \\            const thread metal::vec<bfloat16_t, {[qpt]}> vv = *(const device metal::vec<bfloat16_t, {[qpt]}>*)(vp + t * vss);
    \\            for (int j = 0; j < {[qpt]}; ++j) vs[t][j] = vv[j];
    \\        }}
    \\
    \\        // Rescale only when the running max actually increased.
    \\        if (new_max > max_score) {{
    \\            const float factor = metal::exp(max_score - new_max);
    \\            sum_exp *= factor;
    \\            for (int j = 0; j < {[qpt]}; ++j) o[j] *= factor;
    \\            max_score = new_max;
    \\        }}
    \\        for (int t = 0; t < 4; ++t) {{
    \\            const float p = metal::exp(scores[t] - max_score);
    \\            sum_exp += p;
    \\            for (int j = 0; j < {[qpt]}; ++j) o[j] += p * float(vs[t][j]);
    \\        }}
    \\        kp += 128 * kss;
    \\        vp += 128 * vss;
    \\    }}
    \\
    \\    // Remaining keys of the last partial tile (< 4 for this simdgroup).
    \\    for (; i < N; ++i) {{
    \\        const thread metal::vec<bfloat16_t, {[qpt]}> kv = *(const device metal::vec<bfloat16_t, {[qpt]}>*)kp;
    \\        float s = 0.0f;
    \\        for (int j = 0; j < {[qpt]}; ++j) s += qq[j] * float(kv[j]);
    \\        s = metal::simd_sum(s);
    \\        const float new_max = metal::max(max_score, s);
    \\        const float factor = metal::exp(max_score - new_max);
    \\        const float exp_score = metal::exp(s - new_max);
    \\        max_score = new_max;
    \\        sum_exp = sum_exp * factor + exp_score;
    \\        const thread metal::vec<bfloat16_t, {[qpt]}> vv = *(const device metal::vec<bfloat16_t, {[qpt]}>*)vp;
    \\        for (int j = 0; j < {[qpt]}; ++j) o[j] = o[j] * factor + exp_score * float(vv[j]);
    \\        kp += kss;
    \\        vp += vss;
    \\    }}
    \\
    \\    // Combine the 32 simdgroup partials: max/sum_exp first, then the
    \\    // output accumulator via a shared-memory transpose.
    \\    threadgroup float outputs[1024];
    \\    threadgroup float max_scores[32];
    \\    threadgroup float sum_exp_scores[32];
    \\    if (lane == 0) {{
    \\        max_scores[sg] = max_score;
    \\        sum_exp_scores[sg] = sum_exp;
    \\    }}
    \\    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    max_score = max_scores[lane];
    \\    const float total_max = metal::simd_max(max_score);
    \\    const float factor = metal::exp(max_score - total_max);
    \\    sum_exp = metal::simd_sum(sum_exp_scores[lane] * factor);
    \\    for (int j = 0; j < {[qpt]}; ++j) {{
    \\        outputs[lane * 32 + sg] = o[j];
    \\        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\        o[j] = metal::simd_sum(outputs[sg * 32 + lane] * factor);
    \\        o[j] = sum_exp == 0.0f ? o[j] : o[j] / sum_exp;
    \\        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    }}
    \\    if (lane == 0) {{
    \\        device bfloat16_t* op = out + q_head * {[d]} + sg * {[qpt]};
    \\        for (int j = 0; j < {[qpt]}; ++j) op[j] = bfloat16_t(o[j]);
    \\    }}
    \\
;

// ============================================================
// 2-pass split-KV decode SDPA for long context (omlx
// omlx_sdpa_decode_2pass_1/2, thresholds from sdpa_decode.cpp)
// ============================================================
//
// Pass 1: grid (HKV, 1, blocks) threadgroups of (32, gqa, 1) threads — all
// GQA q-heads of one KV head share the threadgroup so KV is read once per KV
// head per block; each block covers a contiguous ~256-key chunk. Writes fp32
// partials [H, blocks, D] + per-block maxs/sums [H, blocks]. Pass 2: one
// 1024-thread group per q-head merges blocks with the log-sum-exp reduction
// and writes the final bf16 output. block count is a runtime grid/stride
// parameter (multiples of 32 only — pass 2 reduces 32 blocks per loop), so
// the shader source is stable and never recompiled per kv_len.

const sdpa_decode_2pass1_body =
    \\    const uint kv_head = threadgroup_position_in_grid.x;
    \\    const uint block = threadgroup_position_in_grid.z;
    \\    const uint lane = thread_index_in_simdgroup;
    \\    const uint q_head = {[gqa]} * kv_head + thread_position_in_threadgroup.y;
    \\    const int N = n_in[0];
    \\    const int blocks = blocks_in[0];
    \\    const int khs = k_head_stride_in[0];
    \\    const int kss = k_seq_stride_in[0];
    \\    const int vhs = v_head_stride_in[0];
    \\    const int vss = v_seq_stride_in[0];
    \\    const int chunk = (N + blocks - 1) / blocks;
    \\    const int k_start = block * chunk;
    \\    const int k_end = metal::min(N, k_start + chunk);
    \\
    \\    const device bfloat16_t* qp = q + q_head * {[d]} + lane * {[qpt]};
    \\    const device bfloat16_t* kp = keys + kv_head * khs + k_start * kss + lane * {[qpt]};
    \\    const device bfloat16_t* vp = values + kv_head * vhs + k_start * vss + lane * {[qpt]};
    \\    device float* pp = partials + (q_head * blocks + block) * {[d]} + lane * {[qpt]};
    \\
    \\    float qq[{[qpt]}];
    \\    {{
    \\        const thread metal::vec<bfloat16_t, {[qpt]}> qt = *(const device metal::vec<bfloat16_t, {[qpt]}>*)qp;
    \\        for (int j = 0; j < {[qpt]}; ++j) qq[j] = {[scale]s} * float(qt[j]);
    \\    }}
    \\    float o[{[qpt]}];
    \\    for (int j = 0; j < {[qpt]}; ++j) o[j] = 0.0f;
    \\    float max_score = -3.402823466e+38f;
    \\    float sum_exp = 0.0f;
    \\
    \\    int i = k_start;
    \\    for (; i + 4 <= k_end; i += 4) {{
    \\        bfloat16_t ks[4][{[qpt]}];
    \\        for (int t = 0; t < 4; ++t) {{
    \\            const thread metal::vec<bfloat16_t, {[qpt]}> kv = *(const device metal::vec<bfloat16_t, {[qpt]}>*)(kp + t * kss);
    \\            for (int j = 0; j < {[qpt]}; ++j) ks[t][j] = kv[j];
    \\        }}
    \\        float scores[4];
    \\        for (int t = 0; t < 4; ++t) {{
    \\            float s = 0.0f;
    \\            for (int j = 0; j < {[qpt]}; ++j) s += qq[j] * float(ks[t][j]);
    \\            scores[t] = metal::simd_sum(s);
    \\        }}
    \\        const float tile_max = metal::max(metal::max(scores[0], scores[1]), metal::max(scores[2], scores[3]));
    \\        const float new_max = metal::max(max_score, tile_max);
    \\
    \\        bfloat16_t vs[4][{[qpt]}];
    \\        for (int t = 0; t < 4; ++t) {{
    \\            const thread metal::vec<bfloat16_t, {[qpt]}> vv = *(const device metal::vec<bfloat16_t, {[qpt]}>*)(vp + t * vss);
    \\            for (int j = 0; j < {[qpt]}; ++j) vs[t][j] = vv[j];
    \\        }}
    \\
    \\        if (new_max > max_score) {{
    \\            const float factor = metal::exp(max_score - new_max);
    \\            sum_exp *= factor;
    \\            for (int j = 0; j < {[qpt]}; ++j) o[j] *= factor;
    \\            max_score = new_max;
    \\        }}
    \\        for (int t = 0; t < 4; ++t) {{
    \\            const float p = metal::exp(scores[t] - max_score);
    \\            sum_exp += p;
    \\            for (int j = 0; j < {[qpt]}; ++j) o[j] += p * float(vs[t][j]);
    \\        }}
    \\        kp += 4 * kss;
    \\        vp += 4 * vss;
    \\    }}
    \\
    \\    for (; i < k_end; ++i) {{
    \\        const thread metal::vec<bfloat16_t, {[qpt]}> kv = *(const device metal::vec<bfloat16_t, {[qpt]}>*)kp;
    \\        float s = 0.0f;
    \\        for (int j = 0; j < {[qpt]}; ++j) s += qq[j] * float(kv[j]);
    \\        s = metal::simd_sum(s);
    \\        const float new_max = metal::max(max_score, s);
    \\        const float factor = metal::exp(max_score - new_max);
    \\        const float exp_score = metal::exp(s - new_max);
    \\        max_score = new_max;
    \\        sum_exp = sum_exp * factor + exp_score;
    \\        const thread metal::vec<bfloat16_t, {[qpt]}> vv = *(const device metal::vec<bfloat16_t, {[qpt]}>*)vp;
    \\        for (int j = 0; j < {[qpt]}; ++j) o[j] = o[j] * factor + exp_score * float(vv[j]);
    \\        kp += kss;
    \\        vp += vss;
    \\    }}
    \\
    \\    if (lane == 0) {{
    \\        sums[q_head * blocks + block] = sum_exp;
    \\        maxs[q_head * blocks + block] = max_score;
    \\    }}
    \\    for (int j = 0; j < {[qpt]}; ++j) pp[j] = o[j];
    \\
;

const sdpa_decode_2pass2_body =
    \\    const uint head = threadgroup_position_in_grid.x;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint lane = thread_index_in_simdgroup;
    \\    const int blocks = blocks_in[0];
    \\
    \\    const device float* pp = partials + head * blocks * {[d]} + sg * {[d]} + lane * {[ept]};
    \\    const device float* sp = sums + head * blocks;
    \\    const device float* mp = maxs + head * blocks;
    \\
    \\    // Cross-block max, then sum_exp (LSE merge).
    \\    float max_score = -3.402823466e+38f;
    \\    for (int b = 0; b < blocks / 32; ++b) max_score = metal::max(max_score, mp[lane + 32 * b]);
    \\    max_score = metal::simd_max(max_score);
    \\    float sum_exp = 0.0f;
    \\    for (int b = 0; b < blocks / 32; ++b) sum_exp += metal::exp(mp[lane + 32 * b] - max_score) * sp[lane + 32 * b];
    \\    sum_exp = metal::simd_sum(sum_exp);
    \\
    \\    // Rescale and accumulate each block's partial output.
    \\    float o[{[ept]}];
    \\    for (int j = 0; j < {[ept]}; ++j) o[j] = 0.0f;
    \\    for (int b = 0; b < blocks / 32; ++b) {{
    \\        const float factor = metal::exp(mp[sg] - max_score);
    \\        for (int j = 0; j < {[ept]}; ++j) o[j] += factor * pp[j];
    \\        mp += 32;
    \\        pp += 32 * {[d]};
    \\    }}
    \\
    \\    // Shared-memory transpose reduce across simdgroups (same as 1-pass).
    \\    threadgroup float outputs[1024];
    \\    for (int j = 0; j < {[ept]}; ++j) {{
    \\        outputs[lane * 32 + sg] = o[j];
    \\        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\        o[j] = metal::simd_sum(outputs[sg * 32 + lane]);
    \\        o[j] = sum_exp == 0.0f ? o[j] : o[j] / sum_exp;
    \\        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    }}
    \\    if (lane == 0) {{
    \\        device bfloat16_t* op = out + head * {[d]} + sg * {[ept]};
    \\        for (int j = 0; j < {[ept]}; ++j) op[j] = bfloat16_t(o[j]);
    \\    }}
    \\
;

const SdpaDecodeKey = struct {
    d: i32,
    gqa: i32,
    scale_bits: u32,
};

const SdpaDecodeEntry = struct {
    key: SdpaDecodeKey,
    kernel: c.c.mlx_fast_metal_kernel,
    broken: bool = false,
};

var sdpa_decode_entries: [4]?SdpaDecodeEntry = .{ null, null, null, null };
var sdpa_decode_warned = false;
var sdpa_decode_off: ?bool = null;

/// Debug escape hatch / A/B toggle: DMLX_NO_SDPA_DECODE=1 forces the stock
/// fused SDPA path.
fn sdpaDecodeDisabled() bool {
    if (sdpa_decode_off == null) {
        sdpa_decode_off = if (std.c.getenv("DMLX_NO_SDPA_DECODE")) |v| v[0] != 0 and v[0] != '0' else false;
    }
    return sdpa_decode_off.?;
}

fn sdpaDecodeWarn(err: anyerror) void {
    if (!sdpa_decode_warned) {
        sdpa_decode_warned = true;
        std.log.warn("fused decode SDPA kernel unavailable ({s}); using stock scaled_dot_product_attention", .{@errorName(err)});
    }
}

fn createSdpaDecodeKernel(allocator: std.mem.Allocator, key: SdpaDecodeKey) !c.c.mlx_fast_metal_kernel {
    const name_z = try std.fmt.allocPrintSentinel(allocator, "sdpa_decode_d{d}_g{d}_s{x}", .{ key.d, key.gqa, key.scale_bits }, 0);
    defer allocator.free(name_z);
    const scale_z = try qwen38_gdn.f32MetalLit(allocator, @bitCast(key.scale_bits));
    defer allocator.free(scale_z);
    const source_z = try std.fmt.allocPrintSentinel(allocator, sdpa_decode_body, .{
        .d = key.d,
        .qpt = @divExact(key.d, 32),
        .gqa = key.gqa,
        .scale = scale_z,
    }, 0);
    defer allocator.free(source_z);
    const header_z = try allocator.dupeZ(u8, "");
    defer allocator.free(header_z);

    const in_vec = try qwen38_gdn.makeStringVec(allocator, &.{ "q", "keys", "values", "n_in", "k_head_stride_in", "k_seq_stride_in", "v_head_stride_in", "v_seq_stride_in" });
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = try qwen38_gdn.makeStringVec(allocator, &.{"out"});
    defer _ = c.c.mlx_vector_string_free(out_vec);

    return c.c.mlx_fast_metal_kernel_new(
        name_z.ptr,
        in_vec,
        out_vec,
        source_z.ptr,
        header_z.ptr,
        true, // ensure_row_contiguous (all inputs are contiguous by the gate)
        false, // atomic_outputs
    );
}

fn getSdpaDecodeKernel(allocator: std.mem.Allocator, key: SdpaDecodeKey) ?c.c.mlx_fast_metal_kernel {
    for (&sdpa_decode_entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) return if (e.broken) null else e.kernel;
        }
    }
    const kernel = createSdpaDecodeKernel(allocator, key) catch |err| {
        sdpaDecodeWarn(err);
        return null;
    };
    for (&sdpa_decode_entries) |*slot| {
        if (slot.* == null) {
            slot.* = .{ .key = key, .kernel = kernel };
            return kernel;
        }
    }
    return null;
}

fn markSdpaDecodeBroken(key: SdpaDecodeKey) void {
    for (&sdpa_decode_entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) e.broken = true;
        }
    }
}

const Sdpa2PassEntry = struct {
    key: SdpaDecodeKey,
    kernel1: c.c.mlx_fast_metal_kernel,
    kernel2: c.c.mlx_fast_metal_kernel,
    broken: bool = false,
};

var sdpa_2pass_entries: [4]?Sdpa2PassEntry = .{ null, null, null, null };

/// Successful 2-pass applies (test/observability counter).
pub var sdpa_2pass_applies: u64 = 0;

fn createSdpa2PassEntry(allocator: std.mem.Allocator, key: SdpaDecodeKey) !Sdpa2PassEntry {
    const scale_z = try qwen38_gdn.f32MetalLit(allocator, @bitCast(key.scale_bits));
    defer allocator.free(scale_z);
    const header_z = try allocator.dupeZ(u8, "");
    defer allocator.free(header_z);

    const name1_z = try std.fmt.allocPrintSentinel(allocator, "sdpa_decode_2pass1_d{d}_g{d}_s{x}", .{ key.d, key.gqa, key.scale_bits }, 0);
    defer allocator.free(name1_z);
    const source1_z = try std.fmt.allocPrintSentinel(allocator, sdpa_decode_2pass1_body, .{
        .d = key.d,
        .qpt = @divExact(key.d, 32),
        .gqa = key.gqa,
        .scale = scale_z,
    }, 0);
    defer allocator.free(source1_z);
    const in1_vec = try qwen38_gdn.makeStringVec(allocator, &.{ "q", "keys", "values", "n_in", "blocks_in", "k_head_stride_in", "k_seq_stride_in", "v_head_stride_in", "v_seq_stride_in" });
    defer _ = c.c.mlx_vector_string_free(in1_vec);
    const out1_vec = try qwen38_gdn.makeStringVec(allocator, &.{ "partials", "sums", "maxs" });
    defer _ = c.c.mlx_vector_string_free(out1_vec);
    const k1 = c.c.mlx_fast_metal_kernel_new(name1_z.ptr, in1_vec, out1_vec, source1_z.ptr, header_z.ptr, true, false);
    errdefer _ = c.c.mlx_fast_metal_kernel_free(k1);

    const name2_z = try std.fmt.allocPrintSentinel(allocator, "sdpa_decode_2pass2_d{d}_g{d}_s{x}", .{ key.d, key.gqa, key.scale_bits }, 0);
    defer allocator.free(name2_z);
    const source2_z = try std.fmt.allocPrintSentinel(allocator, sdpa_decode_2pass2_body, .{
        .d = key.d,
        .ept = @divExact(key.d, 32),
    }, 0);
    defer allocator.free(source2_z);
    const in2_vec = try qwen38_gdn.makeStringVec(allocator, &.{ "partials", "sums", "maxs", "blocks_in" });
    defer _ = c.c.mlx_vector_string_free(in2_vec);
    const out2_vec = try qwen38_gdn.makeStringVec(allocator, &.{"out"});
    defer _ = c.c.mlx_vector_string_free(out2_vec);
    const k2 = c.c.mlx_fast_metal_kernel_new(name2_z.ptr, in2_vec, out2_vec, source2_z.ptr, header_z.ptr, true, false);

    return .{ .key = key, .kernel1 = k1, .kernel2 = k2 };
}

fn getSdpa2PassKernels(allocator: std.mem.Allocator, key: SdpaDecodeKey) ?*Sdpa2PassEntry {
    for (&sdpa_2pass_entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) return if (e.broken) null else e;
        }
    }
    const entry = createSdpa2PassEntry(allocator, key) catch |err| {
        sdpaDecodeWarn(err);
        return null;
    };
    for (&sdpa_2pass_entries) |*slot| {
        if (slot.* == null) {
            slot.* = entry;
            return &slot.*.?;
        }
    }
    return null;
}

fn markSdpa2PassBroken(key: SdpaDecodeKey) void {
    for (&sdpa_2pass_entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) e.broken = true;
        }
    }
}

/// 2-pass dispatch threshold (omlx 'd'-class rule): GQA and kv_len >= 1024.
fn useSdpa2Pass(kv_len: i32, h: i32, hkv: i32) bool {
    return h > hkv and kv_len >= 1024;
}

/// Block count (omlx 'd'-class rule, M3 Ultra tuned): ~256-key chunks,
/// multiple of 32, clamped to [32/64, 256].
fn sdpa2PassBlocks(n: i32) i32 {
    const b = @divTrunc(@divTrunc(n + 255, 256) + 31, 32) * 32;
    return @min(256, @max(if (n >= 4096) @as(i32, 64) else 32, b));
}

fn makeI32Scalar(allocator: std.mem.Allocator, v: i32) !Array {
    return Array.fromData(allocator, i32, &[_]i32{v}, &[_]i32{1});
}

/// 2-pass split-KV decode SDPA. Same in/out contract as sdpaDecodeFused.
fn sdpaDecode2Pass(
    ctx: EagerContext,
    q: Array,
    k: Array,
    v: Array,
    scale: f32,
    h: i32,
    hkv: i32,
    d: i32,
    kv_len: i32,
) !?Array {
    const key = SdpaDecodeKey{ .d = d, .gqa = @divExact(h, hkv), .scale_bits = @bitCast(scale) };
    const entry = getSdpa2PassKernels(ctx.allocator, key) orelse return null;
    const gqa: i32 = @divExact(h, hkv);
    const cap: i32 = @intCast(k.shape()[2]);
    const blocks = sdpa2PassBlocks(kv_len);

    const n_arr = try makeI32Scalar(ctx.allocator, kv_len);
    defer n_arr.deinit();
    const blocks_arr = try makeI32Scalar(ctx.allocator, blocks);
    defer blocks_arr.deinit();
    const khs_arr = try makeI32Scalar(ctx.allocator, cap * d);
    defer khs_arr.deinit();
    const kss_arr = try makeI32Scalar(ctx.allocator, d);
    defer kss_arr.deinit();
    const vhs_arr = try makeI32Scalar(ctx.allocator, cap * d);
    defer vhs_arr.deinit();
    const vss_arr = try makeI32Scalar(ctx.allocator, d);
    defer vss_arr.deinit();

    // Pass 1: grid (HKV, 1, blocks) groups of (32, gqa, 1) threads; outputs
    // are fp32 partials [H, blocks, D] and per-block sums/maxs [H, blocks].
    const cfg1 = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg1);
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg1, hkv * 32, gqa, blocks));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg1, 32, gqa, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg1, &[_]i32{ h, blocks, d }, 3, c.c.MLX_FLOAT32));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg1, &[_]i32{ h, blocks }, 2, c.c.MLX_FLOAT32));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg1, &[_]i32{ h, blocks }, 2, c.c.MLX_FLOAT32));

    const in1_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(in1_vec);
    for ([_]Array{ q, k, v, n_arr, blocks_arr, khs_arr, kss_arr, vhs_arr, vss_arr }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(in1_vec, inp.inner));
    }
    var out1_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(out1_vec);
    c.check(c.c.mlx_fast_metal_kernel_apply(&out1_vec, entry.kernel1, in1_vec, cfg1, qwen38_gdn.fusedStream())) catch |err| {
        markSdpa2PassBroken(key);
        sdpaDecodeWarn(err);
        return null;
    };

    var partials_h = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&partials_h, out1_vec, 0));
    var sums_h = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&sums_h, out1_vec, 1));
    var maxs_h = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&maxs_h, out1_vec, 2));
    const partials = Array.fromHandle(partials_h);
    defer partials.deinit();
    const sums = Array.fromHandle(sums_h);
    defer sums.deinit();
    const maxs = Array.fromHandle(maxs_h);
    defer maxs.deinit();

    // Pass 2: one 1024-thread group per q head; LSE merge over blocks.
    const cfg2 = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg2);
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg2, h * 1024, 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg2, 1024, 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg2, &[_]i32{ 1, h, 1, d }, 4, c.c.MLX_BFLOAT16));

    const in2_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(in2_vec);
    for ([_]Array{ partials, sums, maxs, blocks_arr }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(in2_vec, inp.inner));
    }
    var out2_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(out2_vec);
    c.check(c.c.mlx_fast_metal_kernel_apply(&out2_vec, entry.kernel2, in2_vec, cfg2, qwen38_gdn.fusedStream())) catch |err| {
        markSdpa2PassBroken(key);
        sdpaDecodeWarn(err);
        return null;
    };

    var out_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&out_handle, out2_vec, 0));
    sdpa_2pass_applies += 1;
    return Array.fromHandle(out_handle);
}

fn isContiguous4D(a: Array, d0: i32, d1: i32, d2: i32, d3: i32) bool {
    const sh = a.shape();
    if (sh.len != 4) return false;
    if (sh[0] != d0 or sh[1] != d1 or sh[2] != d2 or sh[3] != d3) return false;
    const st = a.strides();
    return st[0] == d1 * d2 * d3 and st[1] == d2 * d3 and st[2] == d3 and st[3] == 1;
}

/// Fused decode SDPA over the KV slab. q: [1, H, 1, D] bf16; k/v: the full
/// contiguous slabs [1, HKV, cap, D] bf16 with kv_len valid rows. Returns
/// [1, H, 1, D] bf16, or null when unsupported (caller falls back to stock
/// SDPA). The kernel indexes strides directly, so slab capacity is fine.
/// Long contexts (kv_len >= 1024 with GQA) take the 2-pass split-KV path.
fn sdpaDecodeFused(
    ctx: EagerContext,
    q: Array,
    k: Array,
    v: Array,
    scale: f32,
    h: i32,
    hkv: i32,
    d: i32,
    kv_len: i32,
) !?Array {
    if (sdpaDecodeDisabled()) return null;
    if (kv_len < 1 or @mod(d, 32) != 0 or d < 32 or @mod(h, hkv) != 0) return null;
    const dt = dtype_mod.Dtype.bfloat16;
    if (q.dtype() != dt or k.dtype() != dt or v.dtype() != dt) return null;
    const cap: i32 = @intCast(k.shape()[2]);
    if (!isContiguous4D(q, 1, h, 1, d)) return null;
    if (!isContiguous4D(k, 1, hkv, cap, d)) return null;
    if (!isContiguous4D(v, 1, hkv, cap, d)) return null;
    // Vectorized bf16 loads need 16B-aligned rows: strides divisible by 8.
    const khs: i32 = cap * d;
    if (@mod(khs, 8) != 0 or @mod(d, 8) != 0) return null;

    if (useSdpa2Pass(kv_len, h, hkv)) {
        if (try sdpaDecode2Pass(ctx, q, k, v, scale, h, hkv, d, kv_len)) |out| return out;
        // Fall through to the 1-pass kernel when the 2-pass pair is broken.
    }

    const key = SdpaDecodeKey{ .d = d, .gqa = @divExact(h, hkv), .scale_bits = @bitCast(scale) };
    const kernel = getSdpaDecodeKernel(ctx.allocator, key) orelse return null;

    const n_arr = try Array.fromData(ctx.allocator, i32, &[_]i32{kv_len}, &[_]i32{1});
    defer n_arr.deinit();
    const khs_arr = try Array.fromData(ctx.allocator, i32, &[_]i32{khs}, &[_]i32{1});
    defer khs_arr.deinit();
    const kss_arr = try Array.fromData(ctx.allocator, i32, &[_]i32{d}, &[_]i32{1});
    defer kss_arr.deinit();
    const vhs_arr = try Array.fromData(ctx.allocator, i32, &[_]i32{cap * d}, &[_]i32{1});
    defer vhs_arr.deinit();
    const vss_arr = try Array.fromData(ctx.allocator, i32, &[_]i32{d}, &[_]i32{1});
    defer vss_arr.deinit();

    const cfg = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg);
    // One threadgroup per q head; 1024 threads = 32 simdgroups.
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, h * 1024, 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ 1, h, 1, d }, 4, c.c.MLX_BFLOAT16));

    const input_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(input_vec);
    for ([_]Array{ q, k, v, n_arr, khs_arr, kss_arr, vhs_arr, vss_arr }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(input_vec, inp.inner));
    }
    var output_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(output_vec);
    c.check(c.c.mlx_fast_metal_kernel_apply(&output_vec, kernel, input_vec, cfg, qwen38_gdn.fusedStream())) catch |err| {
        markSdpaDecodeBroken(key);
        sdpaDecodeWarn(err);
        return null;
    };

    var out_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&out_handle, output_vec, 0));
    return Array.fromHandle(out_handle);
}

// Verify-width split-KV SDPA (L = 2..8 rows) — port of omlx
// patches/qwen35_verify_sdpa_split.py `_wide_causal_sdpa` (Apache-2.0, tile
// design adapted there from Splash paged_attention_tile.h). Hardcoded to the
// Qwen3.8 dense layout: D=256, H=24, HKV=4, 8 padded rows, bf16.
//
// Why: stock scaled_dot_product_attention routes this shape (L*GQA = 48 > the
// vector kernel's 32-row budget) to the composed unfused path, which at 16K
// context costs ~100-110 ms of a ~190 ms verify round (measured with
// DMLX_VERIFY_NO_ATTN_CORE=1, 2026-11-07). Here one threadgroup per (q head,
// key split) holds all 8 rows: 4 simdgroups own 64 head dims each, partial
// scores meet in threadgroup memory once per 32-key block, online softmax
// runs 16 threads per row, and a second launch merges the splits with the
// log-sum-exp pairs. Causal: row r attends keys <= kv_len - L + r (the same
// bottom-right alignment stock "causal" gives this shape).
const sdpa_verify_partial_body =
    \\    constexpr int D = 256;
    \\    constexpr int BK = 32;
    \\    constexpr int SS = 36;   // padded partial-S row stride (floats)
    \\    constexpr int PS = 40;   // padded P row stride (bf16)
    \\    uint tid = thread_position_in_threadgroup.x;
    \\    uint sg = tid / 32;
    \\    uint lane = tid % 32;
    \\    int qh = int(threadgroup_position_in_grid.x);
    \\    int split = int(threadgroup_position_in_grid.y);
    \\    int h = qh / 6;          // GQA factor 6: kv head = q head / 6
    \\    int T = int(params[0]);
    \\    int L = int(params[1]);
    \\    int chunk = int(params[2]);
    \\    float scale = as_type<float>(params[4]);
    \\    int t_begin = split * chunk;
    \\    int t_end = min(t_begin + chunk, T);
    \\    const int khs = k_head_stride_in[0];
    \\    const int kss = k_seq_stride_in[0];
    \\    const int vhs = v_head_stride_in[0];
    \\    const int vss = v_seq_stride_in[0];
    \\
    \\    threadgroup float Sp[4 * 8 * SS];
    \\    threadgroup bfloat16_t P[8 * PS];
    \\    threadgroup float st_m[8];
    \\    threadgroup float st_l[8];
    \\    threadgroup float st_a[8];
    \\
    \\    const device bfloat16_t* qp = q + qh * 8 * D + int(sg) * 64;
    \\    simdgroup_matrix<bfloat16_t, 8, 8> qa[8];
    \\    for (int dk = 0; dk < 8; ++dk)
    \\        simdgroup_load(qa[dk], qp + dk * 8, D);
    \\    const device bfloat16_t* kh = k + h * khs + int(sg) * 64;
    \\    const device bfloat16_t* vh = v + h * vhs + int(sg) * 64;
    \\
    \\    simdgroup_matrix<float, 8, 8> o[8];
    \\    for (int j = 0; j < 8; ++j) o[j] = simdgroup_matrix<float, 8, 8>(0.0f);
    \\    if (tid < 8) {
    \\        st_m[tid] = -INFINITY;
    \\        st_l[tid] = 0.0f;
    \\    }
    \\    short qid = short(lane / 4);
    \\    short fm = (qid & 4) + short((lane / 2) % 4);
    \\
    \\    for (int t0 = t_begin; t0 < t_end; t0 += BK) {
    \\        for (int c = 0; c < 4; ++c) {
    \\            int ts = min(t0 + c * 8, T - 8);
    \\            simdgroup_matrix<float, 8, 8> s = simdgroup_matrix<float, 8, 8>(0.0f);
    \\            for (int dk = 0; dk < 8; ++dk) {
    \\                simdgroup_matrix<bfloat16_t, 8, 8> kb;
    \\                simdgroup_load(kb, kh + ts * kss + dk * 8, kss, ulong2(0, 0), true);
    \\                simdgroup_multiply_accumulate(s, qa[dk], kb, s);
    \\            }
    \\            simdgroup_store(s, Sp + int(sg) * 8 * SS + c * 8, SS);
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        {
    \\            int r = int(tid) / 16;
    \\            int cp = (int(tid) % 16) * 2;
    \\            float sv[2];
    \\            float rmax = -INFINITY;
    \\            for (int e = 0; e < 2; ++e) {
    \\                int col = cp + e;
    \\                int c = col / 8;
    \\                int base = t0 + c * 8;
    \\                int key = min(base, T - 8) + (col - c * 8);
    \\                bool ok = r < L && key >= base && key < t_end && key <= T - L + r;
    \\                float val = Sp[r * SS + col] + Sp[8 * SS + r * SS + col]
    \\                    + Sp[16 * SS + r * SS + col] + Sp[24 * SS + r * SS + col];
    \\                sv[e] = ok ? val * scale : -INFINITY;
    \\                rmax = max(rmax, sv[e]);
    \\            }
    \\            for (int off = 1; off < 16; off <<= 1)
    \\                rmax = max(rmax, simd_shuffle_xor(rmax, ushort(off)));
    \\            float m_old = st_m[r];
    \\            float m_new = max(m_old, rmax);
    \\            float rsum = 0.0f;
    \\            for (int e = 0; e < 2; ++e) {
    \\                float p = m_new == -INFINITY ? 0.0f : exp(sv[e] - m_new);
    \\                rsum += p;
    \\                P[r * PS + cp + e] = bfloat16_t(p);
    \\            }
    \\            for (int off = 1; off < 16; off <<= 1)
    \\                rsum += simd_shuffle_xor(rsum, ushort(off));
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            if ((tid % 16) == 0) {
    \\                float alpha = m_new == -INFINITY ? 1.0f : exp(m_old - m_new);
    \\                st_a[r] = alpha;
    \\                st_l[r] = st_l[r] * alpha + rsum;
    \\                st_m[r] = m_new;
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        float alpha = st_a[fm];
    \\        for (int j = 0; j < 8; ++j) {
    \\            o[j].thread_elements()[0] *= alpha;
    \\            o[j].thread_elements()[1] *= alpha;
    \\        }
    \\        for (int kt = 0; kt < 4; ++kt) {
    \\            simdgroup_matrix<bfloat16_t, 8, 8> pa;
    \\            simdgroup_load(pa, P + kt * 8, PS);
    \\            int ts = min(t0 + kt * 8, T - 8);
    \\            simdgroup_matrix<bfloat16_t, 8, 8> vb[8];
    \\            for (int j = 0; j < 8; ++j)
    \\                simdgroup_load(vb[j], vh + ts * vss + j * 8, vss);
    \\            for (int j = 0; j < 8; ++j)
    \\                simdgroup_multiply_accumulate(o[j], pa, vb[j], o[j]);
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    device float* ob = o_part + ((split * 24 + qh) * 8) * D + int(sg) * 64;
    \\    for (int j = 0; j < 8; ++j)
    \\        simdgroup_store(o[j], ob + j * 8, D);
    \\    if (tid < 8) {
    \\        ml_part[((split * 24 + qh) * 8 + int(tid)) * 2] = st_m[tid];
    \\        ml_part[((split * 24 + qh) * 8 + int(tid)) * 2 + 1] = st_l[tid];
    \\    }
;

const sdpa_verify_combine_body =
    \\    constexpr int D = 256;
    \\    uint d = thread_position_in_threadgroup.x;
    \\    int r = int(threadgroup_position_in_grid.x);
    \\    int qh = int(threadgroup_position_in_grid.y);
    \\    int L = int(params[1]);
    \\    int n_splits = int(params[3]);
    \\    float M = -INFINITY;
    \\    for (int s = 0; s < n_splits; ++s)
    \\        M = max(M, ml_part[((s * 24 + qh) * 8 + r) * 2]);
    \\    float total = 0.0f, acc = 0.0f;
    \\    for (int s = 0; s < n_splits; ++s) {
    \\        int idx = (s * 24 + qh) * 8 + r;
    \\        float m = ml_part[idx * 2];
    \\        if (m == -INFINITY) continue;
    \\        float w = exp(m - M);
    \\        total += w * ml_part[idx * 2 + 1];
    \\        acc += w * o_part[idx * D + d];
    \\    }
    \\    out[(qh * L + r) * D + d] = bfloat16_t(acc / total);
;

const SdpaVerifyEntry = struct {
    kernel1: c.c.mlx_fast_metal_kernel,
    kernel2: c.c.mlx_fast_metal_kernel,
    broken: bool = false,
};

var sdpa_verify_entry: ?SdpaVerifyEntry = null;
var sdpa_verify_off: ?bool = null;
var sdpa_verify_gate_logged: u32 = 0;
pub var sdpa_verify_applies: u64 = 0;

/// One-shot observability: which gate rejected the split kernel (or that it
/// engaged). Bit-coded so each reason logs once per process.
fn sdpaVerifyGateLog(bit: u32, msg: []const u8) void {
    if (sdpa_verify_gate_logged & bit != 0) return;
    sdpa_verify_gate_logged |= bit;
    std.log.info("[verify-split] {s}", .{msg});
}

/// Escape hatch / A/B toggle: DMLX_VERIFY_SDPA_SPLIT=0 forces stock SDPA.
fn sdpaVerifySplitDisabled() bool {
    if (sdpa_verify_off == null) {
        sdpa_verify_off = if (std.c.getenv("DMLX_VERIFY_SDPA_SPLIT")) |v| v[0] == '0' else false;
    }
    return sdpa_verify_off.?;
}

fn getSdpaVerifyKernels(allocator: std.mem.Allocator) ?*SdpaVerifyEntry {
    if (sdpa_verify_entry) |*e| return if (e.broken) null else e;
    const header_z = allocator.dupeZ(u8, "") catch return null;
    defer allocator.free(header_z);

    const name1_z = allocator.dupeZ(u8, "dmlx_verify_sdpa_split_partial") catch return null;
    defer allocator.free(name1_z);
    const src1_z = allocator.dupeZ(u8, sdpa_verify_partial_body) catch return null;
    defer allocator.free(src1_z);
    const in1_vec = qwen38_gdn.makeStringVec(allocator, &.{ "q", "k", "v", "params", "k_head_stride_in", "k_seq_stride_in", "v_head_stride_in", "v_seq_stride_in" }) catch return null;
    defer _ = c.c.mlx_vector_string_free(in1_vec);
    const out1_vec = qwen38_gdn.makeStringVec(allocator, &.{ "o_part", "ml_part" }) catch return null;
    defer _ = c.c.mlx_vector_string_free(out1_vec);
    const k1 = c.c.mlx_fast_metal_kernel_new(name1_z.ptr, in1_vec, out1_vec, src1_z.ptr, header_z.ptr, true, false);
    // mlx returns null-ish handle on compile failure; validated on first apply.

    const name2_z = allocator.dupeZ(u8, "dmlx_verify_sdpa_split_combine") catch return null;
    defer allocator.free(name2_z);
    const src2_z = allocator.dupeZ(u8, sdpa_verify_combine_body) catch return null;
    defer allocator.free(src2_z);
    const in2_vec = qwen38_gdn.makeStringVec(allocator, &.{ "o_part", "ml_part", "params" }) catch return null;
    defer _ = c.c.mlx_vector_string_free(in2_vec);
    const out2_vec = qwen38_gdn.makeStringVec(allocator, &.{"out"}) catch return null;
    defer _ = c.c.mlx_vector_string_free(out2_vec);
    const k2 = c.c.mlx_fast_metal_kernel_new(name2_z.ptr, in2_vec, out2_vec, src2_z.ptr, header_z.ptr, true, false);

    sdpa_verify_entry = .{ .kernel1 = k1, .kernel2 = k2 };
    return &sdpa_verify_entry.?;
}

/// Split-KV causal SDPA for the verify block: q_pad [1, H, 8, D] bf16
/// (L valid rows), k/v the contiguous slabs [1, HKV, cap, D]. Returns
/// [1, H, L, D] bf16, or null when the shape is unsupported / kernel broken.
fn sdpaVerifySplit(
    ctx: EagerContext,
    q_t: Array, // [1, H, L, D] bf16 contiguous
    k: Array,
    v: Array,
    scale: f32,
    h: i32,
    hkv: i32,
    d: i32,
    l: i32,
    kv_len: i32,
) !?Array {
    if (sdpaVerifySplitDisabled()) return null;
    // Hardcoded kernel layout: D=256, GQA=6, H=24, padded to 8 rows.
    if (d != 256 or h != 24 or hkv != 4 or l < 2 or l > 8) {
        sdpaVerifyGateLog(1, "shape gate (d/h/hkv/l)");
        return null;
    }
    const dt = dtype_mod.Dtype.bfloat16;
    if (q_t.dtype() != dt or k.dtype() != dt or v.dtype() != dt) {
        sdpaVerifyGateLog(2, "dtype gate");
        return null;
    }
    const cap: i32 = @intCast(k.shape()[2]);
    if (!isContiguous4D(q_t, 1, h, l, d)) {
        sdpaVerifyGateLog(4, "q not contiguous");
        return null;
    }
    if (!isContiguous4D(k, 1, hkv, cap, d) or !isContiguous4D(v, 1, hkv, cap, d)) {
        sdpaVerifyGateLog(8, "kv slab not contiguous");
        return null;
    }
    const entry = getSdpaVerifyKernels(ctx.allocator) orelse {
        sdpaVerifyGateLog(16, "kernel compile failed");
        return null;
    };
    sdpaVerifyGateLog(128, "engaged");

    // Pad rows L..8 with zeros (garbage rows are masked/discarded).
    var q8 = q_t;
    var q8_owned = false;
    var pad: ?Array = null;
    defer if (pad) |p| p.deinit();
    if (l < 8) {
        pad = try array_mod.zeros(ctx.allocator, &[_]i32{ 1, h, 8 - l, d }, dt);
        var cat_parts = [_]Array{ q_t, pad.? };
        q8 = try shape_mod.concatenateAxis(ctx, &cat_parts, 2);
        q8_owned = true;
    }
    defer if (q8_owned) q8.deinit();

    // Chunk heuristic from omlx (M3 Ultra tuned): ~16 splits per head,
    // 32-aligned, clamped [512, 2048]; very long caches use 512-key splits.
    var chunk: i32 = @min(2048, @max(512, @divTrunc(@divTrunc(kv_len + 15, 16) + 31, 32) * 32));
    if (kv_len > 32768) chunk = 512;
    const n_splits: i32 = @divTrunc(kv_len + chunk - 1, chunk);

    const params = try Array.fromData(ctx.allocator, u32, &[_]u32{
        @intCast(kv_len),
        @intCast(l),
        @intCast(chunk),
        @intCast(n_splits),
        @bitCast(scale),
    }, &[_]i32{5});
    defer params.deinit();
    const khs = try makeI32Scalar(ctx.allocator, cap * d);
    defer khs.deinit();
    const kss = try makeI32Scalar(ctx.allocator, d);
    defer kss.deinit();
    const vhs = try makeI32Scalar(ctx.allocator, cap * d);
    defer vhs.deinit();
    const vss = try makeI32Scalar(ctx.allocator, d);
    defer vss.deinit();

    // Pass 1: one 128-thread group per (q head, key split).
    const cfg1 = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg1);
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg1, h * 128, n_splits, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg1, 128, 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg1, &[_]i32{ n_splits, h, 8, d }, 4, c.c.MLX_FLOAT32));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg1, &[_]i32{ n_splits, h, 8, 2 }, 4, c.c.MLX_FLOAT32));

    const in1_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(in1_vec);
    for ([_]Array{ q8, k, v, params, khs, kss, vhs, vss }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(in1_vec, inp.inner));
    }
    var out1_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(out1_vec);
    c.check(c.c.mlx_fast_metal_kernel_apply(&out1_vec, entry.kernel1, in1_vec, cfg1, qwen38_gdn.fusedStream())) catch |err| {
        sdpa_verify_entry.?.broken = true;
        sdpaDecodeWarn(err);
        return null;
    };

    var o_part_h = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&o_part_h, out1_vec, 0));
    var ml_part_h = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&ml_part_h, out1_vec, 1));
    const o_part = Array.fromHandle(o_part_h);
    defer o_part.deinit();
    const ml_part = Array.fromHandle(ml_part_h);
    defer ml_part.deinit();

    // Pass 2: LSE merge over splits, one 256-thread group per (row, q head).
    const cfg2 = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg2);
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg2, l * 256, h, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg2, 256, 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg2, &[_]i32{ 1, h, l, d }, 4, c.c.MLX_BFLOAT16));

    const in2_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(in2_vec);
    for ([_]Array{ o_part, ml_part, params }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(in2_vec, inp.inner));
    }
    var out2_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(out2_vec);
    c.check(c.c.mlx_fast_metal_kernel_apply(&out2_vec, entry.kernel2, in2_vec, cfg2, qwen38_gdn.fusedStream())) catch |err| {
        sdpa_verify_entry.?.broken = true;
        sdpaDecodeWarn(err);
        return null;
    };

    var out_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&out_handle, out2_vec, 0));
    sdpa_verify_applies += 1;
    return Array.fromHandle(out_handle);
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
            // MTP verify rows (M = 1 + draft depth) are a shape stock qmm
            // handles badly; routed only while armed (see verify_qmm.zig).
            if (try verify_qmm.maybeQmm(self.ctx, x, qw)) |y| return y;
            // Long-prefill rows: variant-8 tile (see qwen38_qmm_v8.zig).
            if (try qmm_v8.maybeApply(self.ctx, x, w, s, biases)) |y| return y;
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
        // While verify-qmm routing is armed, q/k/v share this input and go
        // out in ONE sg8 group launch (close-decode-gap fusion program F1);
        // falls back to three separate calls when ineligible.
        var q_lin: Array = undefined;
        var k_lin: Array = undefined;
        var v_lin: Array = undefined;
        if (self.q_scales != null and self.k_scales != null and self.v_scales != null and
            self.q_biases != null and self.k_biases != null and self.v_biases != null)
        {
            const group_qws = [_]quantize_mod.QuantizedWeight{
                .{ .data = self.q_proj, .scales = self.q_scales.?, .biases = self.q_biases.?, .config = self.q_quant, .original_shape = &[_]i32{} },
                .{ .data = self.k_proj, .scales = self.k_scales.?, .biases = self.k_biases.?, .config = self.k_quant, .original_shape = &[_]i32{} },
                .{ .data = self.v_proj, .scales = self.v_scales.?, .biases = self.v_biases.?, .config = self.v_quant, .original_shape = &[_]i32{} },
            };
            if (try verify_qmm.maybeGroupQmm(ctx, x, &group_qws)) |*gout| {
                var go = gout.*;
                q_lin = go.outs[0];
                k_lin = go.outs[1];
                v_lin = go.outs[2];
                go.count = 0; // ownership moved to the three locals
            } else {
                q_lin = try self.linear(x, self.q_proj, self.q_scales, self.q_biases, self.q_quant);
                k_lin = try self.linear(x, self.k_proj, self.k_scales, self.k_biases, self.k_quant);
                v_lin = try self.linear(x, self.v_proj, self.v_scales, self.v_biases, self.v_quant);
            }
        } else {
            q_lin = try self.linear(x, self.q_proj, self.q_scales, self.q_biases, self.q_quant);
            k_lin = try self.linear(x, self.k_proj, self.k_scales, self.k_biases, self.k_quant);
            v_lin = try self.linear(x, self.v_proj, self.v_scales, self.v_biases, self.v_quant);
        }
        defer q_lin.deinit();
        defer k_lin.deinit();
        defer v_lin.deinit();
        const q_resh = try ops.reshape(ctx, q_lin, &[_]i32{ 1, L, H, 2 * D });
        defer q_resh.deinit();
        const q_raw = try ops.slice(ctx, q_resh, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, L, H, D }, &[_]i32{ 1, 1, 1, 1 });
        defer q_raw.deinit();
        const gate = if (self.output_gate) blk: {
            const g = try ops.slice(ctx, q_resh, &[_]i32{ 0, 0, 0, D }, &[_]i32{ 1, L, H, 2 * D }, &[_]i32{ 1, 1, 1, 1 });
            break :blk g;
        } else null;
        defer if (gate) |g| g.deinit();

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

        // DIAGNOSTIC ONLY (`DMLX_VERIFY_NO_ATTN_CORE=1`): price the
        // multi-row (verify / prefill) attention core — slab append, the
        // causal SDPA over L query rows, the transposes and the output gate
        // — across the 16 full-attention layers.
        //
        // Worth pricing because omlx wrote dedicated kernels for exactly this
        // shape (`patches/qwen35_verify_sdpa_split.py`: blocks of up to 8
        // rows through `_gqa_causal_sdpa` / `_wide_causal_sdpa`), and its
        // docstring states the per-row fallback "is the reason deeper MTP
        // chains stop paying on this family". Their verify forward's non-qmm
        // is ~5 ms against our ~16 ms (plan doc §41).
        //
        // Per §30.6 the stand-in consumes q_t, k_t AND v_t so none of the
        // three projection qmms is pruned, and per §31.2 it keeps values
        // bounded (q_t is already rope'd and norm'd, so the residual stream
        // stays in range). Output is garbage.
        if (std.c.getenv("DMLX_VERIFY_NO_ATTN_CORE") != null and L > 1) {
            const qf = try ops.transposeAxes(ctx, q_t, &[_]i32{ 0, 2, 1, 3 });
            defer qf.deinit();
            const qflat = try ops.reshape(ctx, qf, &[_]i32{ 1, L, H * D });
            defer qflat.deinit();
            const kf = try ops.transposeAxes(ctx, k_t, &[_]i32{ 0, 2, 1, 3 });
            defer kf.deinit();
            const vf = try ops.transposeAxes(ctx, v_t, &[_]i32{ 0, 2, 1, 3 });
            defer vf.deinit();
            const kvsum = try ops.add(ctx, kf, vf);
            defer kvsum.deinit();
            const kvflat = try ops.reshape(ctx, kvsum, &[_]i32{ 1, L, HKV * D });
            defer kvflat.deinit();
            const reps: usize = @intCast(@divExact(H, HKV));
            var parts: [64]Array = undefined;
            for (0..reps) |i| parts[i] = kvflat;
            const wide = try shape_mod.concatenateAxis(ctx, parts[0..reps], 2);
            defer wide.deinit();
            const sub = try ops.add(ctx, qflat, wide);
            defer sub.deinit();
            return self.linear(sub, self.o_proj, self.o_scales, self.o_biases, self.o_quant);
        }

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
        // Verify-width split-KV (FlashDecoding-style): one threadgroup per
        // (q head, key split) holds all rows; stock composed SDPA is the
        // fallback for shapes the kernel does not cover.
        const attn = (try sdpaVerifySplit(ctx, q_t, cache.keys.?, cache.values.?, scale, H, HKV, D, L, kv_len)) orelse
            try fast_mod.scaledDotProductAttention(
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

        // DIAGNOSTIC ONLY (`DMLX_DECODE_NO_ATTN_CORE=1`): price the
        // attention block's non-qmm work (KV slab append + SDPA + output
        // gate) across the 16 full-attention layers, as part of decomposing
        // the ~7.7 ms of per-forward non-qmm left after plan doc §30.
        //
        // Per the §30.6 rule, the stand-in CONSUMES q_t, k_t AND v_t so that
        // none of the three projection qmms becomes dead code in MLX's lazy
        // graph — that pruning is exactly what inflated §25.1 by 9x. Bytes
        // read per token are therefore unchanged; only the append and the
        // SDPA disappear. Output is garbage.
        if (std.c.getenv("DMLX_DECODE_NO_ATTN_CORE") != null) {
            const qf = try ops.reshape(ctx, pre.q_t, &[_]i32{ 1, H * D });
            defer qf.deinit();
            const kf = try ops.reshape(ctx, pre.k_t, &[_]i32{ 1, HKV * D });
            defer kf.deinit();
            const vf = try ops.reshape(ctx, pre.v_t, &[_]i32{ 1, HKV * D });
            defer vf.deinit();
            const kv = try ops.add(ctx, kf, vf);
            defer kv.deinit();
            // Widen [1, HKV*D] to [1, H*D] by repetition (H*D is an exact
            // multiple of HKV*D for grouped-query attention).
            const reps: usize = @intCast(@divExact(H, HKV));
            var parts: [64]Array = undefined;
            for (0..reps) |i| parts[i] = kv;
            const wide = try shape_mod.concatenateAxis(ctx, parts[0..reps], 1);
            defer wide.deinit();
            const sub = try ops.add(ctx, qf, wide);
            defer sub.deinit();
            return self.linear(sub, self.o_proj, self.o_scales, self.o_biases, self.o_quant);
        }

        try cacheAppend(ctx, cache, pre.k_t, pre.v_t, 1, HKV, D);
        // No eval here: the caller (Qwen38ServingModel.forwardTokens) submits
        // the updated cache states via asyncEval once per token.

        const kv_len: i32 = cache.len();
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const attn = (try sdpaDecodeFused(ctx, pre.q_t, cache.keys.?, cache.values.?, scale, H, HKV, D, kv_len)) orelse blk: {
            // Fallback: stock fused SDPA on [1, HKV, len, D] slab views.
            const k_view = try ops.slice(ctx, cache.keys.?, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, HKV, kv_len, D }, &[_]i32{ 1, 1, 1, 1 });
            defer k_view.deinit();
            const v_view = try ops.slice(ctx, cache.values.?, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, HKV, kv_len, D }, &[_]i32{ 1, 1, 1, 1 });
            defer v_view.deinit();
            break :blk try fast_mod.scaledDotProductAttention(ctx, pre.q_t, k_view, v_view, scale, "", null, null);
        };
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

    /// Decode attention core for an externally-computed pre-cache segment
    /// (MTP drafter's compiled step): KV slab append + decode SDPA.
    /// Returns [1, H, 1, D] bf16. Numerically identical to the corresponding
    /// slice of postCache.
    pub fn cacheAppendAndSdpa(self: *Qwen38Attn, cache: *Qwen38AttnCache, q_t: Array, k_t: Array, v_t: Array) !Array {
        const ctx = self.ctx;
        const H: i32 = @intCast(self.num_heads);
        const HKV: i32 = @intCast(self.num_kv_heads);
        const D: i32 = @intCast(self.head_dim);

        try cacheAppend(ctx, cache, k_t, v_t, 1, HKV, D);

        const kv_len: i32 = cache.len();
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        if (try sdpaDecodeFused(ctx, q_t, cache.keys.?, cache.values.?, scale, H, HKV, D, kv_len)) |attn| return attn;
        const k_view = try ops.slice(ctx, cache.keys.?, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, HKV, kv_len, D }, &[_]i32{ 1, 1, 1, 1 });
        defer k_view.deinit();
        const v_view = try ops.slice(ctx, cache.values.?, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, HKV, kv_len, D }, &[_]i32{ 1, 1, 1, 1 });
        defer v_view.deinit();
        return fast_mod.scaledDotProductAttention(ctx, q_t, k_view, v_view, scale, "", null, null);
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

test "Qwen38Attn fused decode SDPA matches stock SDPA" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const H: i32 = 24;
    const HKV: i32 = 4;
    const D: i32 = 256;
    const cap: i32 = 8192; // covers the 2-pass cases; exercises non-trivial head strides
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));

    // Dispatch threshold: 2-pass (split-KV) only for GQA and kv_len >= 1024.
    try std.testing.expect(!useSdpa2Pass(1023, H, HKV));
    try std.testing.expect(useSdpa2Pass(1024, H, HKV));
    try std.testing.expect(useSdpa2Pass(4097, H, HKV));
    try std.testing.expect(!useSdpa2Pass(4097, H, H)); // no GQA → 1-pass
    // Block count: multiples of 32, clamped [32/64, 256].
    try std.testing.expectEqual(@as(i32, 32), sdpa2PassBlocks(1024));
    try std.testing.expectEqual(@as(i32, 32), sdpa2PassBlocks(2048));
    try std.testing.expectEqual(@as(i32, 64), sdpa2PassBlocks(4096));
    try std.testing.expectEqual(@as(i32, 64), sdpa2PassBlocks(8192));
    try std.testing.expectEqual(@as(i32, 256), sdpa2PassBlocks(100000));

    // kv_len <= 513 exercises the 1-pass kernel; >= 1025 the 2-pass path.
    for ([_]i32{ 37, 257, 513, 1025, 2048, 4097 }) |kv_len| {
        const applies_before = sdpa_2pass_applies;
        const n_q: usize = @intCast(H * D);
        const n_kv: usize = @intCast(HKV * cap * D);
        const qbuf = try allocator.alloc(f32, n_q);
        defer allocator.free(qbuf);
        const kvbuf = try allocator.alloc(f32, 2 * n_kv);
        defer allocator.free(kvbuf);
        // Wider dynamic range than testFill so softmax weighting matters.
        for (qbuf, 0..) |*x, i| {
            x.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 29 + 7, 23) - 11)) * 0.1;
        }
        for (kvbuf, 0..) |*x, i| {
            x.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 17 + 3, 19) - 9)) * 0.1;
        }
        const q_f32 = try Array.fromData(allocator, f32, qbuf, &[_]i32{ 1, H, 1, D });
        defer q_f32.deinit();
        const q = try ops.astype(ctx, q_f32, .bfloat16);
        defer q.deinit();
        const k_f32 = try Array.fromData(allocator, f32, kvbuf[0..n_kv], &[_]i32{ 1, HKV, cap, D });
        defer k_f32.deinit();
        const k = try ops.astype(ctx, k_f32, .bfloat16);
        defer k.deinit();
        const v_f32 = try Array.fromData(allocator, f32, kvbuf[n_kv..], &[_]i32{ 1, HKV, cap, D });
        defer v_f32.deinit();
        const v = try ops.astype(ctx, v_f32, .bfloat16);
        defer v.deinit();

        const fused = (try sdpaDecodeFused(ctx, q, k, v, scale, H, HKV, D, kv_len)) orelse
            return error.FusedKernelUnavailable;
        defer fused.deinit();
        try fused.eval();
        // The dispatch must actually take the expected path (no silent
        // fallback within sdpaDecodeFused).
        try std.testing.expectEqual(useSdpa2Pass(kv_len, H, HKV), sdpa_2pass_applies > applies_before);
        const fshape = fused.shape();
        try std.testing.expectEqual(@as(i32, 1), fshape[0]);
        try std.testing.expectEqual(H, fshape[1]);
        try std.testing.expectEqual(@as(i32, 1), fshape[2]);
        try std.testing.expectEqual(D, fshape[3]);

        const k_view = try ops.slice(ctx, k, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, HKV, kv_len, D }, &[_]i32{ 1, 1, 1, 1 });
        defer k_view.deinit();
        const v_view = try ops.slice(ctx, v, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, HKV, kv_len, D }, &[_]i32{ 1, 1, 1, 1 });
        defer v_view.deinit();
        const stock = try fast_mod.scaledDotProductAttention(ctx, q, k_view, v_view, scale, "", null, null);
        defer stock.deinit();
        try stock.eval();

        const fused_f32 = try ops.astype(ctx, fused, .float32);
        defer fused_f32.deinit();
        const stock_f32 = try ops.astype(ctx, stock, .float32);
        defer stock_f32.deinit();
        const got = try fused_f32.dataSlice(f32);
        const exp = try stock_f32.dataSlice(f32);
        try std.testing.expectEqual(exp.len, got.len);
        for (exp, got) |e, o| try std.testing.expectApproxEqAbs(e, o, 5e-3);
    }
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

test "sdpaVerifySplit matches stock causal SDPA (bf16, split-KV)" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const H: i32 = 24;
    const HKV: i32 = 4;
    const D: i32 = 256;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));

    // Pseudo-random but deterministic fill (xorshift64 over the indices).
    const fillBuf = struct {
        fn call(alloc: std.mem.Allocator, n: usize, seed: u64) ![]f32 {
            const buf = try alloc.alloc(f32, n);
            var s = seed;
            for (0..n) |i| {
                s = s *% 6364136223846793005 +% 1442695040888963407;
                buf[i] = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(s >> 33)), 2000) - 1000)) / 1000.0;
            }
            return buf;
        }
    }.call;

    for ([_]i32{ 3, 8 }) |L| {
        for ([_]i32{ 40, 1500 }) |T| {
            const cap = T;
            // q [1, H, L, D] bf16
            const qn: usize = @intCast(H * L * D);
            const qb = try fillBuf(allocator, qn, 11);
            defer allocator.free(qb);
            const qf = try Array.fromData(allocator, f32, qb, &[_]i32{ 1, H, L, D });
            defer qf.deinit();
            const q = try ops.astype(ctx, qf, .bfloat16);
            defer q.deinit();
            // k/v slabs [1, HKV, cap, D] bf16
            const kn: usize = @intCast(HKV * cap * D);
            const kb = try fillBuf(allocator, kn, 22);
            defer allocator.free(kb);
            const vb = try fillBuf(allocator, kn, 33);
            defer allocator.free(vb);
            const kf = try Array.fromData(allocator, f32, kb, &[_]i32{ 1, HKV, cap, D });
            defer kf.deinit();
            const vf = try Array.fromData(allocator, f32, vb, &[_]i32{ 1, HKV, cap, D });
            defer vf.deinit();
            const k = try ops.astype(ctx, kf, .bfloat16);
            defer k.deinit();
            const v = try ops.astype(ctx, vf, .bfloat16);
            defer v.deinit();

            const split = (try sdpaVerifySplit(ctx, q, k, v, scale, H, HKV, D, L, T)) orelse
                return error.TestUnexpectedResult; // kernel must apply to this shape
            defer split.deinit();

            const k_view = try ops.slice(ctx, k, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, HKV, T, D }, &[_]i32{ 1, 1, 1, 1 });
            defer k_view.deinit();
            const v_view = try ops.slice(ctx, v, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ 1, HKV, T, D }, &[_]i32{ 1, 1, 1, 1 });
            defer v_view.deinit();
            const stock = try fast_mod.scaledDotProductAttention(ctx, q, k_view, v_view, scale, "causal", null, null);
            defer stock.deinit();

            const split32 = try ops.astype(ctx, split, .float32);
            defer split32.deinit();
            const stock32 = try ops.astype(ctx, stock, .float32);
            defer stock32.deinit();
            const diff = try ops.subtract(ctx, split32, stock32);
            defer diff.deinit();
            const adiff = try ops.abs(ctx, diff);
            defer adiff.deinit();
            const m = try reduce_mod.max(ctx, adiff, false);
            defer m.deinit();
            try m.eval();
            // Both compute in fp32 and round to bf16 on output; tolerance is
            // a couple of bf16 ulps at O(1) magnitude.
            try std.testing.expect(try m.item(f32) <= 2e-2);
        }
    }
}
