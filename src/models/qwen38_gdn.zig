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
const dtype_mod = @import("mlx").dtype;

// ============================================================
// Fused Metal kernel for the gated delta-rule state update
// ============================================================
//
// One threadgroup per value head, D threads per group; thread t owns state
// row dv=t in registers (float4 chunks). Reads the [HV, D, D] f32 state once
// and writes it once, replacing ~8 elementwise/reduction passes. The kernel
// source body is generated per head-dim D (register array must be a Metal
// compile-time size); mlx auto-generates the function signature from the
// input/output names, so `source` is the body only. Enabled for f32 tensors
// with D % 4 == 0 and D <= 128; anything else falls back to the eager chain.

const fused_kernel_body =
    \\    const uint dv = thread_position_in_grid.x;
    \\    const uint h = thread_position_in_grid.y;
    \\    const device metal::float4* srow_in = (const device metal::float4*)(s_in + (h * {[d]} + dv) * {[d]});
    \\    device metal::float4* srow_out = (device metal::float4*)(s_out + (h * {[d]} + dv) * {[d]});
    \\    const device metal::float4* k4 = (const device metal::float4*)(k + h * {[d]});
    \\    const device metal::float4* q4 = (const device metal::float4*)(q + h * {[d]});
    \\    const float gh = g[h];
    \\    metal::float4 row[{[df4]}];
    \\    float kv = 0.0f;
    \\    for (int i = 0; i < {[df4]}; ++i) {{
    \\        metal::float4 s = srow_in[i] * gh;
    \\        row[i] = s;
    \\        kv += metal::dot(s, k4[i]);
    \\    }}
    \\    const float d = (v[h * {[d]} + dv] - kv) * beta[h];
    \\    float yv = 0.0f;
    \\    for (int i = 0; i < {[df4]}; ++i) {{
    \\        metal::float4 s = row[i] + k4[i] * d;
    \\        srow_out[i] = s;
    \\        yv += metal::dot(s, q4[i]);
    \\    }}
    \\    y[h * {[d]} + dv] = yv;
    \\
;

const FusedKernelEntry = struct {
    d: i32,
    kernel: c.c.mlx_fast_metal_kernel,
    broken: bool = false,
};

// Lazy-init registry. Not synchronized: decode serving is serial
// (vtable_serial) and the test runner is single-threaded.
var fused_entries: [4]?FusedKernelEntry = .{ null, null, null, null };
var fused_gpu_stream: ?c.c.mlx_stream = null;
var fused_warned = false;

/// Default GPU stream, shared by every fused apply. Kernel work must land on
/// a GPU stream even when the caller's EagerContext targets the CPU (unit
/// tests); the default stream handle is process-lifetime and never freed.
fn fusedStream() c.c.mlx_stream {
    if (fused_gpu_stream == null) {
        fused_gpu_stream = c.c.mlx_default_gpu_stream_new();
    }
    return fused_gpu_stream.?;
}

fn fusedWarn(err: anyerror) void {
    if (!fused_warned) {
        fused_warned = true;
        std.log.warn("fused GDN delta-rule kernel unavailable ({s}); using eager op chain", .{@errorName(err)});
    }
}

fn makeStringVec(allocator: std.mem.Allocator, names: []const []const u8) !c.c.mlx_vector_string {
    const vec = c.c.mlx_vector_string_new();
    errdefer _ = c.c.mlx_vector_string_free(vec);
    for (names) |n| {
        const n_z = try allocator.dupeZ(u8, n);
        defer allocator.free(n_z);
        try c.check(c.c.mlx_vector_string_append_value(vec, n_z.ptr));
    }
    return vec;
}

fn createFusedKernel(allocator: std.mem.Allocator, d: i32) !c.c.mlx_fast_metal_kernel {
    const df4 = @divExact(d, 4);
    const name_z = try std.fmt.allocPrintSentinel(allocator, "gdn_delta_d{d}", .{d}, 0);
    defer allocator.free(name_z);
    const source_z = try std.fmt.allocPrintSentinel(allocator, fused_kernel_body, .{ .d = d, .df4 = df4 }, 0);
    defer allocator.free(source_z);
    const header_z = try allocator.dupeZ(u8, "");
    defer allocator.free(header_z);

    const in_vec = try makeStringVec(allocator, &.{ "q", "k", "v", "g", "beta", "s_in" });
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = try makeStringVec(allocator, &.{ "y", "s_out" });
    defer _ = c.c.mlx_vector_string_free(out_vec);

    // The Zig binding in mlx-zig predates the mlx-c 0.7.0 signature (it lacks
    // output_names / ensure_row_contiguous), so call the C API directly.
    return c.c.mlx_fast_metal_kernel_new(
        name_z.ptr,
        in_vec,
        out_vec,
        source_z.ptr,
        header_z.ptr,
        true, // ensure_row_contiguous
        false, // atomic_outputs
    );
}

/// Fetch (or lazily build) the fused kernel for head-dim d. Returns null when
/// the kernel is known-broken so callers fall back to the eager chain.
fn getFusedKernel(allocator: std.mem.Allocator, d: i32) ?c.c.mlx_fast_metal_kernel {
    for (&fused_entries) |*slot| {
        if (slot.*) |*e| {
            if (e.d == d) return if (e.broken) null else e.kernel;
        }
    }
    const kernel = createFusedKernel(allocator, d) catch |err| {
        fusedWarn(err);
        return null;
    };
    for (&fused_entries) |*slot| {
        if (slot.* == null) {
            slot.* = .{ .d = d, .kernel = kernel };
            return kernel;
        }
    }
    // Registry full (more than 4 distinct head dims in one process): leak the
    // handle rather than evict a live kernel; the eager chain stays correct.
    return null;
}

fn markFusedBroken(d: i32) void {
    for (&fused_entries) |*slot| {
        if (slot.*) |*e| {
            if (e.d == d) e.broken = true;
        }
    }
}

const DeltaRuleOut = struct {
    s_new: Array, // [HV, D, D] f32
    y_f32: Array, // [HV, D] f32
};

/// Fused single-kernel delta rule. Returns null when the shape/dtype is
/// unsupported or the kernel is unavailable (caller falls back to eager).
fn deltaRuleFused(
    ctx: EagerContext,
    ssm_state: Array,
    k_full: Array,
    v: Array,
    beta: Array,
    g: Array,
    q_full: Array,
    hv: i32,
    d: i32,
) !?DeltaRuleOut {
    if (@mod(d, 4) != 0 or d < 4 or d > 128) return null;
    const dt_f32 = dtype_mod.Dtype.float32;
    if (ssm_state.dtype() != dt_f32 or q_full.dtype() != dt_f32 or k_full.dtype() != dt_f32 or
        v.dtype() != dt_f32 or g.dtype() != dt_f32 or beta.dtype() != dt_f32) return null;

    const kernel = getFusedKernel(ctx.allocator, d) orelse return null;

    const cfg = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg);
    // Grid is in threads: D threads per head-row owner, HV groups on y.
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, d, hv, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, d, 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ hv, d }, 2, c.c.MLX_FLOAT32));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ hv, d, d }, 3, c.c.MLX_FLOAT32));

    const input_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(input_vec);
    for ([_]Array{ q_full, k_full, v, g, beta, ssm_state }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(input_vec, inp.inner));
    }
    var output_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(output_vec);
    c.check(c.c.mlx_fast_metal_kernel_apply(&output_vec, kernel, input_vec, cfg, fusedStream())) catch |err| {
        markFusedBroken(d);
        fusedWarn(err);
        return null;
    };

    var y_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&y_handle, output_vec, 0));
    errdefer _ = c.c.mlx_array_free(y_handle);
    var s_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&s_handle, output_vec, 1));
    return .{ .s_new = Array.fromHandle(s_handle), .y_f32 = Array.fromHandle(y_handle) };
}

// ============================================================
// Fused Metal kernel v2: the whole post-conv GDN tail
// ============================================================
//
// One threadgroup per value head, D threads per group; thread t owns state
// row dv=t (float4 register chunks) as in the v1 delta-rule kernel. On top of
// the delta rule this kernel folds in: the q/k per-head weight-less rmsNorm
// (scales pre-folded into the per-head factors — item A), the per-head gates
// (softplus decay g, sigmoid beta), and the silu out-gate (bf16 round-trips
// included), producing the out_proj input directly. The q/k norms are
// redundant per-thread serial reductions over D elements (no cross-thread
// sync); the out-gate norm needs the head's full y row, exchanged via one
// threadgroup buffer + barrier.
//
// Dtype round-trips replicate the eager chain exactly: y is rounded to the
// qkv dtype T before the out-gate norm (gated_delta_update returns
// y.astype(q.dtype)), the normed value is rounded to T before AND after the
// norm_w multiply (mlx fast.rms_norm: astype to out_type, then a multiply in
// out_type), beta is rounded to b's dtype, and the final gated value is
// rounded to T. With T=float every rounding is an identity — the real decode
// path is f32 here because the conv state promotes the concat.
//
// Source is text-generated per (D, ratio, PK, qkv dtype, b dtype); registry
// and fallback mirror the v1 kernel.

const tail_kernel_body =
    \\    const uint dv = thread_position_in_grid.x;
    \\    const uint h = thread_position_in_grid.y;
    \\    const uint hk = h / {[ratio]};
    \\    const device {[t]s}* qp = qkv + hk * {[d]};
    \\    const device {[t]s}* kp = qkv + {[pk]} + hk * {[d]};
    \\    const device {[t]s}* vp = qkv + 2 * {[pk]} + h * {[d]};
    \\
    \\    // Per-head weight-less rmsNorm factors for q/k (folded scales):
    \\    // redundant serial reduction per thread — no cross-thread sync.
    \\    float ssq = 0.0f;
    \\    float ssk = 0.0f;
    \\    for (int i = 0; i < {[d]}; ++i) {{
    \\        const float qi = float(qp[i]);
    \\        const float ki = float(kp[i]);
    \\        ssq += qi * qi;
    \\        ssk += ki * ki;
    \\    }}
    \\    const float qfac = {[qscale]s} * metal::rsqrt(ssq / {[d]}.0f + 1e-6f);
    \\    const float kfac = {[kscale]s} * metal::rsqrt(ssk / {[d]}.0f + 1e-6f);
    \\
    \\    // Per-head gates (f32): g = exp(-exp(A_log) * softplus(a + dt_bias)),
    \\    // beta = sigmoid(b) rounded to b's dtype.
    \\    const float ax = float(a[h]) + float(dt_bias[h]);
    \\    const float sp = ax > 20.0f ? ax : metal::log(1.0f + metal::exp(ax));
    \\    const float gh = metal::exp(-metal::exp(float(A_log[h])) * sp);
    \\    const float bh = float({[bt]s}(1.0f / (1.0f + metal::exp(-float(b[h])))));
    \\
    \\    // Delta rule: thread owns state row dv.
    \\    const device metal::float4* srow_in = (const device metal::float4*)(s_in + (h * {[d]} + dv) * {[d]});
    \\    device metal::float4* srow_out = (device metal::float4*)(s_out + (h * {[d]} + dv) * {[d]});
    \\    metal::float4 row[{[df4]}];
    \\    float kv = 0.0f;
    \\    for (int i = 0; i < {[df4]}; ++i) {{
    \\        const metal::float4 s = srow_in[i] * gh;
    \\        row[i] = s;
    \\        const metal::float4 k4 = kfac * metal::float4(float(kp[4 * i]), float(kp[4 * i + 1]), float(kp[4 * i + 2]), float(kp[4 * i + 3]));
    \\        kv += metal::dot(s, k4);
    \\    }}
    \\    const float dl = (float(vp[dv]) - kv) * bh;
    \\    float yv = 0.0f;
    \\    for (int i = 0; i < {[df4]}; ++i) {{
    \\        const metal::float4 s = row[i] + (kfac * dl) * metal::float4(float(kp[4 * i]), float(kp[4 * i + 1]), float(kp[4 * i + 2]), float(kp[4 * i + 3]));
    \\        srow_out[i] = s;
    \\        yv += metal::dot(s, qfac * metal::float4(float(qp[4 * i]), float(qp[4 * i + 1]), float(qp[4 * i + 2]), float(qp[4 * i + 3])));
    \\    }}
    \\
    \\    // Out gate: y rounds to T before the norm; norm in f32; norm_w
    \\    // multiply rounds to T; silu(z) gate in f32; output rounds to T.
    \\    const {[t]s} y_t = {[t]s}(yv);
    \\    threadgroup float y_shared[{[d]}];
    \\    y_shared[dv] = float(y_t);
    \\    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    float ssy = 0.0f;
    \\    for (int i = 0; i < {[d]}; ++i) ssy += y_shared[i] * y_shared[i];
    \\    const float yn = float({[t]s}(y_shared[dv] * metal::rsqrt(ssy / {[d]}.0f + 1e-6f)));
    \\    const float yw = float({[t]s}(yn * float(norm_w[dv])));
    \\    const float zz = float(z[h * {[d]} + dv]);
    \\    const float zact = zz / (1.0f + metal::exp(-zz));
    \\    out[h * {[d]} + dv] = {[t]s}(zact * yw);
    \\
;

const TailKernelKey = struct {
    d: i32,
    ratio: i32,
    pk: i32,
    qkv_bf16: bool,
    b_bf16: bool,
};

const TailKernelEntry = struct {
    key: TailKernelKey,
    kernel: c.c.mlx_fast_metal_kernel,
    broken: bool = false,
};

var tail_entries: [8]?TailKernelEntry = .{ null, null, null, null, null, null, null, null };
var tail_warned = false;

fn tailWarn(err: anyerror) void {
    if (!tail_warned) {
        tail_warned = true;
        std.log.warn("fused GDN tail kernel unavailable ({s}); using eager op chain", .{@errorName(err)});
    }
}

fn f32MetalLit(allocator: std.mem.Allocator, v: f32) ![:0]u8 {
    // Shortest round-trip decimal + f suffix parses back to the exact f32.
    return std.fmt.allocPrintSentinel(allocator, "{d}f", .{v}, 0);
}

fn createTailKernel(allocator: std.mem.Allocator, key: TailKernelKey) !c.c.mlx_fast_metal_kernel {
    const d = key.d;
    const inv_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
    const qscale_z = try f32MetalLit(allocator, inv_scale * inv_scale);
    defer allocator.free(qscale_z);
    const kscale_z = try f32MetalLit(allocator, inv_scale);
    defer allocator.free(kscale_z);
    const t: []const u8 = if (key.qkv_bf16) "bfloat16_t" else "float";
    const bt: []const u8 = if (key.b_bf16) "bfloat16_t" else "float";

    const name_z = try std.fmt.allocPrintSentinel(allocator, "gdn_tail_d{d}_r{d}_p{d}_{c}{c}", .{ d, key.ratio, key.pk, if (key.qkv_bf16) @as(u8, 'b') else 'f', if (key.b_bf16) @as(u8, 'b') else 'f' }, 0);
    defer allocator.free(name_z);
    const source_z = try std.fmt.allocPrintSentinel(allocator, tail_kernel_body, .{
        .d = d,
        .df4 = @divExact(d, 4),
        .ratio = key.ratio,
        .pk = key.pk,
        .qscale = qscale_z,
        .kscale = kscale_z,
        .t = t,
        .bt = bt,
    }, 0);
    defer allocator.free(source_z);
    const header_z = try allocator.dupeZ(u8, "");
    defer allocator.free(header_z);

    const in_vec = try makeStringVec(allocator, &.{ "qkv", "a", "b", "z", "A_log", "dt_bias", "norm_w", "s_in" });
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = try makeStringVec(allocator, &.{ "out", "s_out" });
    defer _ = c.c.mlx_vector_string_free(out_vec);

    return c.c.mlx_fast_metal_kernel_new(
        name_z.ptr,
        in_vec,
        out_vec,
        source_z.ptr,
        header_z.ptr,
        true, // ensure_row_contiguous
        false, // atomic_outputs
    );
}

fn getTailKernel(allocator: std.mem.Allocator, key: TailKernelKey) ?c.c.mlx_fast_metal_kernel {
    for (&tail_entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) return if (e.broken) null else e.kernel;
        }
    }
    const kernel = createTailKernel(allocator, key) catch |err| {
        tailWarn(err);
        return null;
    };
    for (&tail_entries) |*slot| {
        if (slot.* == null) {
            slot.* = .{ .key = key, .kernel = kernel };
            return kernel;
        }
    }
    return null;
}

fn markTailBroken(key: TailKernelKey) void {
    for (&tail_entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) e.broken = true;
        }
    }
}

// ============================================================
// Fused Metal kernel v3: the depthwise causal conv chain + the whole
// post-conv GDN tail in ONE kernel
// ============================================================
//
// v2's post-conv tail plus the conv front-end that used to run as
// concatenate(conv_state, qkv_lin) → multiply → column-sum → silu →
// slice/copy new conv state. Per value-head threadgroup, thread dv owns the
// three channels it needs: q channel hk*D+dv, k channel PK+hk*D+dv (both
// shared into threadgroup memory for the norm/delta-rule passes), and v
// channel 2*PK+h*D+dv (register-local). The conv sum runs j=0..K-1 ascending
// per channel, matching convAccumStep semantics; silu is f32.
//
// The new conv window (window rows 1..K-1 = conv_state rows 1..K-2 plus the
// raw qkv_lin row) is written by the same groups: v channels by the owning
// group, q/k channels only by the first group of each ratio-sharing set
// (h % ratio == 0) so shared channels are written exactly once.
//
// The post-conv tensors are always f32 in the fused path (the conv promotes
// to f32), so the v2 template's qkv-dtype round-trips become identities and
// are dropped; the b-dtype round on beta is kept. Everything downstream of
// the conv (norms, gates, delta rule, out gate) is numerically identical to
// v2 with T=float.

const conv_tail_kernel_body =
    \\    const uint dv = thread_position_in_grid.x;
    \\    const uint h = thread_position_in_grid.y;
    \\    const uint hk = h / {[ratio]};
    \\    const uint qbase = hk * {[d]} + dv;
    \\    const uint kbase = {[pk]} + hk * {[d]} + dv;
    \\    const uint vbase = 2 * {[pk]} + h * {[d]} + dv;
    \\
    \\    // Depthwise causal conv (K taps) + silu, f32. Window rows 0..K-2 come
    \\    // from conv_state, row K-1 is the fresh qkv_lin token.
    \\    threadgroup float qsh[{[d]}];
    \\    threadgroup float ksh[{[d]}];
    \\    float qa = 0.0f;
    \\    float ka = 0.0f;
    \\    float va = 0.0f;
    \\    for (int j = 0; j < {[km1]}; ++j) {{
    \\        const uint off = j * {[qkv]};
    \\        qa += conv_state[off + qbase] * conv1d_t[off + qbase];
    \\        ka += conv_state[off + kbase] * conv1d_t[off + kbase];
    \\        va += conv_state[off + vbase] * conv1d_t[off + vbase];
    \\    }}
    \\    const uint loff = {[km1]} * {[qkv]};
    \\    qa += float(qkv_lin[qbase]) * conv1d_t[loff + qbase];
    \\    ka += float(qkv_lin[kbase]) * conv1d_t[loff + kbase];
    \\    va += float(qkv_lin[vbase]) * conv1d_t[loff + vbase];
    \\    qsh[dv] = qa / (1.0f + metal::exp(-qa));
    \\    ksh[dv] = ka / (1.0f + metal::exp(-ka));
    \\    const float vown = va / (1.0f + metal::exp(-va));
    \\
    \\    // New conv window = window rows 1..K-1 (raw pre-conv values).
    \\    if (h % {[ratio]} == 0) {{
    \\        for (int r = 0; r < {[km2]}; ++r) {{
    \\            conv_out[r * {[qkv]} + qbase] = conv_state[(r + 1) * {[qkv]} + qbase];
    \\            conv_out[r * {[qkv]} + kbase] = conv_state[(r + 1) * {[qkv]} + kbase];
    \\        }}
    \\        conv_out[{[km2]} * {[qkv]} + qbase] = float(qkv_lin[qbase]);
    \\        conv_out[{[km2]} * {[qkv]} + kbase] = float(qkv_lin[kbase]);
    \\    }}
    \\    for (int r = 0; r < {[km2]}; ++r) {{
    \\        conv_out[r * {[qkv]} + vbase] = conv_state[(r + 1) * {[qkv]} + vbase];
    \\    }}
    \\    conv_out[{[km2]} * {[qkv]} + vbase] = float(qkv_lin[vbase]);
    \\
    \\    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
    \\    // Per-head weight-less rmsNorm factors for q/k (folded scales):
    \\    // redundant serial reduction per thread — no cross-thread sync.
    \\    float ssq = 0.0f;
    \\    float ssk = 0.0f;
    \\    for (int i = 0; i < {[d]}; ++i) {{
    \\        ssq += qsh[i] * qsh[i];
    \\        ssk += ksh[i] * ksh[i];
    \\    }}
    \\    const float qfac = {[qscale]s} * metal::rsqrt(ssq / {[d]}.0f + 1e-6f);
    \\    const float kfac = {[kscale]s} * metal::rsqrt(ssk / {[d]}.0f + 1e-6f);
    \\
    \\    // Per-head gates (f32): g = exp(-exp(A_log) * softplus(a + dt_bias)),
    \\    // beta = sigmoid(b) rounded to b's dtype.
    \\    const float ax = float(a[h]) + float(dt_bias[h]);
    \\    const float sp = ax > 20.0f ? ax : metal::log(1.0f + metal::exp(ax));
    \\    const float gh = metal::exp(-metal::exp(float(A_log[h])) * sp);
    \\    const float bh = float({[bt]s}(1.0f / (1.0f + metal::exp(-float(b[h])))));
    \\
    \\    // Delta rule: thread owns state row dv.
    \\    const device metal::float4* srow_in = (const device metal::float4*)(s_in + (h * {[d]} + dv) * {[d]});
    \\    device metal::float4* srow_out = (device metal::float4*)(s_out + (h * {[d]} + dv) * {[d]});
    \\    metal::float4 row[{[df4]}];
    \\    float kv = 0.0f;
    \\    for (int i = 0; i < {[df4]}; ++i) {{
    \\        const metal::float4 s = srow_in[i] * gh;
    \\        row[i] = s;
    \\        const metal::float4 k4 = kfac * metal::float4(ksh[4 * i], ksh[4 * i + 1], ksh[4 * i + 2], ksh[4 * i + 3]);
    \\        kv += metal::dot(s, k4);
    \\    }}
    \\    const float dl = (vown - kv) * bh;
    \\    float yv = 0.0f;
    \\    for (int i = 0; i < {[df4]}; ++i) {{
    \\        const metal::float4 s = row[i] + (kfac * dl) * metal::float4(ksh[4 * i], ksh[4 * i + 1], ksh[4 * i + 2], ksh[4 * i + 3]);
    \\        srow_out[i] = s;
    \\        yv += metal::dot(s, qfac * metal::float4(qsh[4 * i], qsh[4 * i + 1], qsh[4 * i + 2], qsh[4 * i + 3]));
    \\    }}
    \\
    \\    // Out gate (all f32 here — the conv promotes the tail input to f32, so
    \\    // the eager chain's T round-trips are identities): norm the head's y
    \\    // row, weight multiply, silu(z) gate.
    \\    threadgroup float y_shared[{[d]}];
    \\    y_shared[dv] = yv;
    \\    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    float ssy = 0.0f;
    \\    for (int i = 0; i < {[d]}; ++i) ssy += y_shared[i] * y_shared[i];
    \\    const float yn = y_shared[dv] * metal::rsqrt(ssy / {[d]}.0f + 1e-6f);
    \\    const float yw = yn * float(norm_w[dv]);
    \\    const float zz = float(z[h * {[d]} + dv]);
    \\    const float zact = zz / (1.0f + metal::exp(-zz));
    \\    out[h * {[d]} + dv] = zact * yw;
    \\
;

const ConvTailKernelKey = struct {
    d: i32,
    ratio: i32,
    pk: i32,
    k: i32,
    b_bf16: bool,
};

const ConvTailKernelEntry = struct {
    key: ConvTailKernelKey,
    kernel: c.c.mlx_fast_metal_kernel,
    broken: bool = false,
};

var conv_tail_entries: [8]?ConvTailKernelEntry = .{ null, null, null, null, null, null, null, null };
var conv_tail_warned = false;

fn convTailWarn(err: anyerror) void {
    if (!conv_tail_warned) {
        conv_tail_warned = true;
        std.log.warn("fused GDN conv+tail kernel unavailable ({s}); using eager op chain", .{@errorName(err)});
    }
}

fn createConvTailKernel(allocator: std.mem.Allocator, key: ConvTailKernelKey) !c.c.mlx_fast_metal_kernel {
    const d = key.d;
    const inv_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
    const qscale_z = try f32MetalLit(allocator, inv_scale * inv_scale);
    defer allocator.free(qscale_z);
    const kscale_z = try f32MetalLit(allocator, inv_scale);
    defer allocator.free(kscale_z);
    const bt: []const u8 = if (key.b_bf16) "bfloat16_t" else "float";
    const qkv = key.pk * (2 + key.ratio);

    const name_z = try std.fmt.allocPrintSentinel(allocator, "gdn_conv_tail_d{d}_r{d}_p{d}_k{d}_{c}", .{ d, key.ratio, key.pk, key.k, if (key.b_bf16) @as(u8, 'b') else 'f' }, 0);
    defer allocator.free(name_z);
    const source_z = try std.fmt.allocPrintSentinel(allocator, conv_tail_kernel_body, .{
        .d = d,
        .df4 = @divExact(d, 4),
        .ratio = key.ratio,
        .pk = key.pk,
        .qkv = qkv,
        .km1 = key.k - 1,
        .km2 = key.k - 2,
        .qscale = qscale_z,
        .kscale = kscale_z,
        .bt = bt,
    }, 0);
    defer allocator.free(source_z);
    const header_z = try allocator.dupeZ(u8, "");
    defer allocator.free(header_z);

    const in_vec = try makeStringVec(allocator, &.{ "qkv_lin", "conv_state", "conv1d_t", "a", "b", "z", "A_log", "dt_bias", "norm_w", "s_in" });
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = try makeStringVec(allocator, &.{ "out", "s_out", "conv_out" });
    defer _ = c.c.mlx_vector_string_free(out_vec);

    return c.c.mlx_fast_metal_kernel_new(
        name_z.ptr,
        in_vec,
        out_vec,
        source_z.ptr,
        header_z.ptr,
        true, // ensure_row_contiguous
        false, // atomic_outputs
    );
}

fn getConvTailKernel(allocator: std.mem.Allocator, key: ConvTailKernelKey) ?c.c.mlx_fast_metal_kernel {
    for (&conv_tail_entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) return if (e.broken) null else e.kernel;
        }
    }
    const kernel = createConvTailKernel(allocator, key) catch |err| {
        convTailWarn(err);
        return null;
    };
    for (&conv_tail_entries) |*slot| {
        if (slot.* == null) {
            slot.* = .{ .key = key, .kernel = kernel };
            return kernel;
        }
    }
    return null;
}

fn markConvTailBroken(key: ConvTailKernelKey) void {
    for (&conv_tail_entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) e.broken = true;
        }
    }
}

pub const ConvTailOut = struct {
    y: Array, // [1, HV*D] f32, ready for out_proj
    s_new: Array, // [HV, D, D] f32
    conv_state: Array, // [K-1, QKV] f32
};

/// Fused single-kernel conv chain + post-conv tail (conv+silu+new conv state,
/// then norms + gates + delta rule + out gate). Returns null when
/// shapes/dtypes are unsupported or the kernel is unavailable (caller falls
/// back to the eager conv chain + v2/eager tail).
fn gdnConvTailFused(
    ctx: EagerContext,
    qkv_lin: Array, // [1, QKV] raw in_proj_qkv output (f32 or bf16)
    conv_state: Array, // [K-1, QKV] f32
    conv1d_t: Array, // [K, QKV] f32
    a: Array, // [1, HV]
    b: Array, // [1, HV]
    z: Array, // [1, HV*D]
    tw: TailWeights,
    hv: i32,
    hk: i32,
    d: i32,
    k: i32,
) !?ConvTailOut {
    if (@mod(d, 4) != 0 or d < 4 or d > 128) return null;
    if (hk < 1 or @mod(hv, hk) != 0) return null;
    if (k < 2) return null;
    if (@abs(tw.norm_eps - 1e-6) > 1e-12) return null;
    if (!isFloatDtype(qkv_lin.dtype())) return null;
    if (conv_state.dtype() != .float32) return null;
    if (!isFloatDtype(conv1d_t.dtype())) return null; // read as f32 (eager casts too)
    if (tw.ssm_state.dtype() != .float32) return null;
    if (!isFloatDtype(a.dtype()) or !isFloatDtype(b.dtype()) or !isFloatDtype(z.dtype()) or
        !isFloatDtype(tw.A_log.dtype()) or !isFloatDtype(tw.dt_bias.dtype()) or !isFloatDtype(tw.norm_w.dtype())) return null;

    const key = ConvTailKernelKey{
        .d = d,
        .ratio = @divExact(hv, hk),
        .pk = hk * d,
        .k = k,
        .b_bf16 = b.dtype() == .bfloat16,
    };
    const kernel = getConvTailKernel(ctx.allocator, key) orelse return null;
    const qkv_dim: i32 = key.pk * (2 + key.ratio);

    const cfg = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg);
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, d, hv, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, d, 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ 1, hv * d }, 2, c.c.MLX_FLOAT32));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ hv, d, d }, 3, c.c.MLX_FLOAT32));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ k - 1, qkv_dim }, 2, c.c.MLX_FLOAT32));

    const input_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(input_vec);
    for ([_]Array{ qkv_lin, conv_state, conv1d_t, a, b, z, tw.A_log, tw.dt_bias, tw.norm_w, tw.ssm_state }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(input_vec, inp.inner));
    }
    var output_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(output_vec);
    c.check(c.c.mlx_fast_metal_kernel_apply(&output_vec, kernel, input_vec, cfg, fusedStream())) catch |err| {
        markConvTailBroken(key);
        convTailWarn(err);
        return null;
    };

    var y_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&y_handle, output_vec, 0));
    errdefer _ = c.c.mlx_array_free(y_handle);
    var s_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&s_handle, output_vec, 1));
    errdefer _ = c.c.mlx_array_free(s_handle);
    var cs_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&cs_handle, output_vec, 2));
    return .{
        .y = Array.fromHandle(y_handle),
        .s_new = Array.fromHandle(s_handle),
        .conv_state = Array.fromHandle(cs_handle),
    };
}

pub const TailOut = struct {
    y: Array, // [1, HV*D] qkv dtype, ready for out_proj
    s_new: Array, // [HV, D, D] f32
};

/// Everything the post-conv tail needs from StepWeights.
const TailWeights = struct {
    q_norm_w: Array, // [D] f32 (folded q scale)
    k_norm_w: Array, // [D] f32 (folded k scale)
    A_log: Array, // [HV]
    dt_bias: Array, // [HV]
    norm_w: Array, // [D]
    ssm_state: Array, // [HV, D, D] f32
    qk_idx: Array, // [HV] i32
    norm_eps: f32 = 1e-6,
};

fn isFloatDtype(dt: dtype_mod.Dtype) bool {
    return dt == .float32 or dt == .bfloat16;
}

/// Fused single-kernel post-conv tail (norms + gates + delta rule + out
/// gate). Returns null when shapes/dtypes are unsupported or the kernel is
/// unavailable (caller falls back to the eager tail).
fn gdnTailFused(
    ctx: EagerContext,
    qkv: Array, // [1, QKV]
    a: Array, // [1, HV]
    b: Array, // [1, HV]
    z: Array, // [1, HV*D]
    tw: TailWeights,
    hv: i32,
    hk: i32,
    d: i32,
) !?TailOut {
    if (@mod(d, 4) != 0 or d < 4 or d > 128) return null;
    if (hk < 1 or @mod(hv, hk) != 0) return null;
    if (@abs(tw.norm_eps - 1e-6) > 1e-12) return null;
    const qdt = qkv.dtype();
    if (!isFloatDtype(qdt)) return null;
    if (tw.ssm_state.dtype() != .float32) return null;
    if (!isFloatDtype(a.dtype()) or !isFloatDtype(b.dtype()) or !isFloatDtype(z.dtype()) or
        !isFloatDtype(tw.A_log.dtype()) or !isFloatDtype(tw.dt_bias.dtype()) or !isFloatDtype(tw.norm_w.dtype())) return null;

    const key = TailKernelKey{
        .d = d,
        .ratio = @divExact(hv, hk),
        .pk = hk * d,
        .qkv_bf16 = qdt == .bfloat16,
        .b_bf16 = b.dtype() == .bfloat16,
    };
    const kernel = getTailKernel(ctx.allocator, key) orelse return null;

    const cfg = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg);
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, d, hv, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, d, 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ 1, hv * d }, 2, if (key.qkv_bf16) c.c.MLX_BFLOAT16 else c.c.MLX_FLOAT32));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ hv, d, d }, 3, c.c.MLX_FLOAT32));

    const input_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(input_vec);
    for ([_]Array{ qkv, a, b, z, tw.A_log, tw.dt_bias, tw.norm_w, tw.ssm_state }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(input_vec, inp.inner));
    }
    var output_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(output_vec);
    c.check(c.c.mlx_fast_metal_kernel_apply(&output_vec, kernel, input_vec, cfg, fusedStream())) catch |err| {
        markTailBroken(key);
        tailWarn(err);
        return null;
    };

    var y_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&y_handle, output_vec, 0));
    errdefer _ = c.c.mlx_array_free(y_handle);
    var s_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&s_handle, output_vec, 1));
    return .{ .y = Array.fromHandle(y_handle), .s_new = Array.fromHandle(s_handle) };
}

/// Eager op-chain post-conv tail (reference implementation and fallback):
/// split q/k/v, per-head weight-less rmsNorm (folded scales), gates, kv-group
/// broadcast, delta rule (v1 fused kernel when eligible), silu out-gate.
fn gdnTailEager(
    ctx: EagerContext,
    qkv: Array, // [1, QKV]
    a: Array, // [1, HV]
    b: Array, // [1, HV]
    z: Array, // [1, HV*D]
    tw: TailWeights,
    hv: i32,
    hk: i32,
    d: i32,
    qkv_dim: i32,
) !TailOut {
    const qkv_flat = try ops.reshape(ctx, qkv, &[_]i32{qkv_dim});
    defer qkv_flat.deinit();
    const pk = hk * d;
    const q_parts = try ops.slice(ctx, qkv_flat, &[_]i32{0}, &[_]i32{pk}, &[_]i32{1});
    defer q_parts.deinit();
    const k_parts = try ops.slice(ctx, qkv_flat, &[_]i32{pk}, &[_]i32{2 * pk}, &[_]i32{1});
    defer k_parts.deinit();
    const v_parts = try ops.slice(ctx, qkv_flat, &[_]i32{2 * pk}, &[_]i32{qkv_dim}, &[_]i32{1});
    defer v_parts.deinit();

    const q_resh = try ops.reshape(ctx, q_parts, &[_]i32{ hk, d });
    defer q_resh.deinit();
    const k_resh = try ops.reshape(ctx, k_parts, &[_]i32{ hk, d });
    defer k_resh.deinit();
    const v = try ops.reshape(ctx, v_parts, &[_]i32{ hv, d });
    defer v.deinit();

    // Weight-less per-head rmsNorm with the q/k scales folded into the weight.
    const q = try fast_mod.rmsNorm(ctx, q_resh, tw.q_norm_w, 1e-6);
    defer q.deinit();
    const k = try fast_mod.rmsNorm(ctx, k_resh, tw.k_norm_w, 1e-6);
    defer k.deinit();

    // g = exp(-exp(A_log_f32) * softplus(a + dt_bias))  [HV]
    const a_flat = try ops.reshape(ctx, a, &[_]i32{hv});
    defer a_flat.deinit();
    const a_dt = try ops.add(ctx, a_flat, tw.dt_bias);
    defer a_dt.deinit();
    const sp = try softplusStep(ctx, a_dt);
    defer sp.deinit();
    const a_log_f32 = try ops.astype(ctx, tw.A_log, .float32);
    defer a_log_f32.deinit();
    const e_a_log = try ops.exp(ctx, a_log_f32);
    defer e_a_log.deinit();
    const g_pre = try ops.multiply(ctx, e_a_log, sp);
    defer g_pre.deinit();
    const g_neg = try ops.negative(ctx, g_pre);
    defer g_neg.deinit();
    const g = try ops.exp(ctx, g_neg); // [HV]
    defer g.deinit();

    const b_flat = try ops.reshape(ctx, b, &[_]i32{hv});
    defer b_flat.deinit();
    const beta = try ops.sigmoid(ctx, b_flat); // [HV]
    defer beta.deinit();

    // Broadcast q/k to value heads via the precomputed gather indices.
    const q_full = try shape_mod.takeAxis(ctx, q, tw.qk_idx, 0); // [HV, D]
    defer q_full.deinit();
    const k_full = try shape_mod.takeAxis(ctx, k, tw.qk_idx, 0); // [HV, D]
    defer k_full.deinit();

    // Gated delta rule step. State S [Hv, Dv, Dk]. Fused v1 kernel when
    // supported (one state read + one write), eager op chain otherwise.
    const dr = (try deltaRuleFused(ctx, tw.ssm_state, k_full, v, beta, g, q_full, hv, d)) orelse
        (try deltaRuleEager(ctx, tw.ssm_state, k_full, v, beta, g, q_full, hv, d));
    errdefer dr.s_new.deinit();
    defer dr.y_f32.deinit();

    // gated_delta_update returns y.astype(q.dtype): round before the norm.
    const y = try ops.astype(ctx, dr.y_f32, qkv.dtype());
    defer y.deinit();

    // Gated RMSNorm: norm in f32, gate in f32, round to input dtype.
    const y_normed = try fast_mod.rmsNorm(ctx, y, tw.norm_w, tw.norm_eps);
    defer y_normed.deinit();
    const y_normed_f32 = try ops.astype(ctx, y_normed, .float32);
    defer y_normed_f32.deinit();
    const z_resh = try ops.reshape(ctx, z, &[_]i32{ hv, d });
    defer z_resh.deinit();
    const z_f32 = try ops.astype(ctx, z_resh, .float32);
    defer z_f32.deinit();
    const z_act = try activations.silu(ctx, z_f32);
    defer z_act.deinit();
    const y_gated_f32 = try ops.multiply(ctx, z_act, y_normed_f32);
    defer y_gated_f32.deinit();
    const y_gated = try ops.astype(ctx, y_gated_f32, qkv.dtype());
    defer y_gated.deinit();
    const y_flat = try ops.reshape(ctx, y_gated, &[_]i32{ 1, hv * d });
    errdefer y_flat.deinit();
    return .{ .y = y_flat, .s_new = dr.s_new };
}

/// Eager op-chain delta rule (reference implementation and fallback).
fn deltaRuleEager(
    ctx: EagerContext,
    ssm_state: Array,
    k_full: Array,
    v: Array,
    beta: Array,
    g: Array,
    q_full: Array,
    hv: i32,
    d: i32,
) !DeltaRuleOut {
    const g_decay = try ops.reshape(ctx, g, &[_]i32{ hv, 1, 1 });
    defer g_decay.deinit();
    const s_decayed = try ops.multiply(ctx, ssm_state, g_decay);
    defer s_decayed.deinit();

    const k_b = try ops.reshape(ctx, k_full, &[_]i32{ hv, 1, d });
    defer k_b.deinit();
    const sk = try ops.multiply(ctx, s_decayed, k_b);
    defer sk.deinit();
    const kv_mem = try reduce_mod.sumAxis(ctx, sk, -1, false); // [HV, Dv]
    defer kv_mem.deinit();

    const v_sub = try ops.subtract(ctx, v, kv_mem);
    defer v_sub.deinit();
    const beta_b = try ops.reshape(ctx, beta, &[_]i32{ hv, 1 });
    defer beta_b.deinit();
    const delta = try ops.multiply(ctx, v_sub, beta_b); // [HV, Dv]
    defer delta.deinit();

    const delta_b = try ops.expandDims(ctx, delta, -1); // [HV, Dv, 1]
    defer delta_b.deinit();
    const k_delta = try ops.multiply(ctx, k_b, delta_b);
    defer k_delta.deinit();
    const s_new = try ops.add(ctx, s_decayed, k_delta);
    errdefer s_new.deinit();

    const q_b2 = try ops.reshape(ctx, q_full, &[_]i32{ hv, 1, d });
    defer q_b2.deinit();
    const sq = try ops.multiply(ctx, s_new, q_b2);
    defer sq.deinit();
    const y_f32 = try reduce_mod.sumAxis(ctx, sq, -1, false); // [HV, Dv] f32
    return .{ .s_new = s_new, .y_f32 = y_f32 };
}

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
/// the weight-less q/k rms_norm "weights" (ones with the q/k scalar scales
/// folded in — rmsNorm computes x*rsqrt(mean(x²)+eps)*w, so scaling w is
/// exactly the post-norm scalar multiply), and kv-group gather indices that
/// broadcast q/k to value heads in a single take.
pub const Precomputed = struct {
    conv1d_t: Array, // [conv_kernel, QKV]
    q_norm_w: Array, // [head_dim] f32, ones * D^-1 (folded q scale)
    k_norm_w: Array, // [head_dim] f32, ones * D^-0.5 (folded k scale)
    qk_idx: Array, // [num_value_heads] i32

    pub fn deinit(self: *Precomputed) void {
        self.conv1d_t.deinit();
        self.q_norm_w.deinit();
        self.k_norm_w.deinit();
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
    const inv_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const nw_buf = try allocator.alloc(f32, head_dim);
    defer allocator.free(nw_buf);
    @memset(nw_buf, inv_scale * inv_scale);
    const q_norm_w = try Array.fromData(allocator, f32, nw_buf, &[_]i32{@intCast(head_dim)});
    errdefer q_norm_w.deinit();
    @memset(nw_buf, inv_scale);
    const k_norm_w = try Array.fromData(allocator, f32, nw_buf, &[_]i32{@intCast(head_dim)});
    errdefer k_norm_w.deinit();
    var out: Precomputed = .{
        .conv1d_t = try shape_mod.transpose(ctx, w2d),
        .q_norm_w = q_norm_w,
        .k_norm_w = k_norm_w,
        .qk_idx = undefined,
    };
    errdefer out.conv1d_t.deinit();
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
    q_norm_w: Array, // [D] f32 (folded q scale)
    k_norm_w: Array, // [D] f32 (folded k scale)
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
    const HK: i32 = @divTrunc(@divTrunc(QKV - HV * D, 2), D);

    // 1. qkv projection (raw) + gate projections; the conv chain lives in the
    // fused conv+tail kernel below when eligible.
    const qkv_lin = try linearQuantParts(ctx, w.x, w.qkv_w, w.qkv_scales, w.qkv_biases);
    defer qkv_lin.deinit();

    const a = try linearQuantParts(ctx, w.x, w.a_w, w.a_scales, w.a_biases);
    defer a.deinit(); // [1, HV]
    const b = try linearQuantParts(ctx, w.x, w.b_w, w.b_scales, w.b_biases);
    defer b.deinit(); // [1, HV]
    const z = try linearQuantParts(ctx, w.x, w.z_w, w.z_scales, w.z_biases);
    defer z.deinit(); // [1, HV*D]

    // 2. conv chain + post-conv tail: q/k rmsNorm (folded scales) + gates +
    // delta rule + silu out-gate. Single fused Metal kernel when supported;
    // otherwise the eager conv chain feeds the v2 fused tail / eager tail.
    const tw = TailWeights{
        .q_norm_w = w.q_norm_w,
        .k_norm_w = w.k_norm_w,
        .A_log = w.A_log,
        .dt_bias = w.dt_bias,
        .norm_w = w.norm_w,
        .ssm_state = w.ssm_state,
        .qk_idx = w.qk_idx,
        .norm_eps = w.norm_eps,
    };
    const tail_res: ConvTailOut = (try gdnConvTailFused(ctx, qkv_lin, w.conv_state, w.conv1d_t, a, b, z, tw, HV, HK, D, K)) orelse blk: {
        // Eager depthwise causal conv (k=4) + silu.
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

        const t2 = (try gdnTailFused(ctx, qkv, a, b, z, tw, HV, HK, D)) orelse
            (try gdnTailEager(ctx, qkv, a, b, z, tw, HV, HK, D, QKV));
        break :blk ConvTailOut{ .y = t2.y, .s_new = t2.s_new, .conv_state = new_conv_state };
    };
    errdefer tail_res.s_new.deinit();
    errdefer tail_res.conv_state.deinit();
    defer tail_res.y.deinit();
    const s_new = tail_res.s_new;

    const out = try linearQuantParts(ctx, tail_res.y, w.out_w, w.out_scales, w.out_biases);
    errdefer out.deinit();
    return .{ .out = out, .conv_state = tail_res.conv_state, .ssm_state = s_new };
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

/// Quantized variant. Input order (25): x, conv_state, ssm_state, qkv_w,
/// qkv_s, qkv_b, conv1d_t, a_w, a_s, a_b, b_w, b_s, b_b, z_w, z_s, z_b,
/// A_log, dt_bias, norm_w, out_w, out_s, out_b, q_norm_w, k_norm_w, qk_idx.
fn gdnStepClosureQuant(inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array {
    if (inputs.len != 25) return error.MlxError;
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
        .q_norm_w = inputs[22],
        .k_norm_w = inputs[23],
        .qk_idx = inputs[24],
    }) catch return error.MlxError;
    return stepToSlice(res, allocator);
}

/// Plain (unquantized) variant — used by the toy-module unit test. Input
/// order (15): x, conv_state, ssm_state, qkv_w, conv1d_t, a_w, b_w, z_w,
/// A_log, dt_bias, norm_w, out_w, q_norm_w, k_norm_w, qk_idx.
fn gdnStepClosurePlain(inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array {
    if (inputs.len != 15) return error.MlxError;
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
        .q_norm_w = inputs[12],
        .k_norm_w = inputs[13],
        .qk_idx = inputs[14],
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
    q_norm_w: Array, // [D] f32, ones * D^-1 (folded q scale) for fast.rmsNorm
    k_norm_w: Array, // [D] f32, ones * D^-0.5 (folded k scale) for fast.rmsNorm
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
        self.q_norm_w.deinit();
        self.k_norm_w.deinit();
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
            .q_norm_w = self.q_norm_w,
            .k_norm_w = self.k_norm_w,
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
        .q_norm_w = pre.q_norm_w,
        .k_norm_w = pre.k_norm_w,
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

test "Qwen38GDN fused delta-rule kernel matches eager chain" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const Case = struct { hv: i32, d: i32 };
    for ([_]Case{ .{ .hv = 6, .d = 4 }, .{ .hv = 48, .d = 128 } }) |cs| {
        const state = try makeTestWeight(allocator, &[_]i32{ cs.hv, cs.d, cs.d });
        defer state.deinit();
        const q = try makeTestWeight(allocator, &[_]i32{ cs.hv, cs.d });
        defer q.deinit();
        const k = try makeTestWeight(allocator, &[_]i32{ cs.hv, cs.d });
        defer k.deinit();
        const v = try makeTestWeight(allocator, &[_]i32{ cs.hv, cs.d });
        defer v.deinit();

        // Gates in (0, 1) like the real softplus-decay / sigmoid-beta path.
        const n_hv: usize = @intCast(cs.hv);
        var gbuf: [48]f32 = undefined;
        var bbuf: [48]f32 = undefined;
        for (0..n_hv) |i| {
            gbuf[i] = 0.5 + 0.04 * @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 7 + 3, 11)));
            bbuf[i] = 0.3 + 0.05 * @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 5 + 1, 9)));
        }
        const g = try Array.fromData(allocator, f32, gbuf[0..n_hv], &[_]i32{cs.hv});
        defer g.deinit();
        const beta = try Array.fromData(allocator, f32, bbuf[0..n_hv], &[_]i32{cs.hv});
        defer beta.deinit();

        const fused = (try deltaRuleFused(ctx, state, k, v, beta, g, q, cs.hv, cs.d)) orelse
            return error.FusedKernelUnavailable;
        defer fused.s_new.deinit();
        defer fused.y_f32.deinit();
        try fused.s_new.eval();
        try fused.y_f32.eval();

        const eager = try deltaRuleEager(ctx, state, k, v, beta, g, q, cs.hv, cs.d);
        defer eager.s_new.deinit();
        defer eager.y_f32.deinit();
        try eager.s_new.eval();
        try eager.y_f32.eval();

        const got_s = try fused.s_new.dataSlice(f32);
        const exp_s = try eager.s_new.dataSlice(f32);
        try std.testing.expectEqual(exp_s.len, got_s.len);
        for (exp_s, got_s) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
        const got_y = try fused.y_f32.dataSlice(f32);
        const exp_y = try eager.y_f32.dataSlice(f32);
        try std.testing.expectEqual(exp_y.len, got_y.len);
        for (exp_y, got_y) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
    }
}

test "Qwen38GDN fused tail kernel matches eager tail" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const Case = struct { hv: i32, hk: i32, d: i32 };
    for ([_]Case{ .{ .hv = 6, .hk = 2, .d = 4 }, .{ .hv = 48, .hk = 16, .d = 128 } }) |cs| {
        const hv = cs.hv;
        const d = cs.d;
        const qkv_dim: i32 = (2 * cs.hk + hv) * d;

        const qkv = try makeTestWeight(allocator, &[_]i32{ 1, qkv_dim });
        defer qkv.deinit();
        const a = try makeTestWeight(allocator, &[_]i32{ 1, hv });
        defer a.deinit();
        const b = try makeTestWeight(allocator, &[_]i32{ 1, hv });
        defer b.deinit();
        const z = try makeTestWeight(allocator, &[_]i32{ 1, hv * d });
        defer z.deinit();
        const ssm_state = try makeTestWeight(allocator, &[_]i32{ hv, d, d });
        defer ssm_state.deinit();
        const a_log = try makeTestWeight(allocator, &[_]i32{hv});
        defer a_log.deinit();
        const dt_bias = try makeTestWeight(allocator, &[_]i32{hv});
        defer dt_bias.deinit();
        const norm_w = try makeTestWeight(allocator, &[_]i32{d});
        defer norm_w.deinit();

        const inv_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
        const nw_buf = try allocator.alloc(f32, @intCast(d));
        defer allocator.free(nw_buf);
        @memset(nw_buf, inv_scale * inv_scale);
        const q_norm_w = try Array.fromData(allocator, f32, nw_buf, &[_]i32{d});
        defer q_norm_w.deinit();
        @memset(nw_buf, inv_scale);
        const k_norm_w = try Array.fromData(allocator, f32, nw_buf, &[_]i32{d});
        defer k_norm_w.deinit();

        const ratio: usize = @intCast(@divExact(hv, cs.hk));
        const idx_buf = try allocator.alloc(i32, @intCast(hv));
        defer allocator.free(idx_buf);
        for (idx_buf, 0..) |*vv, h| vv.* = @intCast(h / ratio);
        const qk_idx = try Array.fromData(allocator, i32, idx_buf, &[_]i32{hv});
        defer qk_idx.deinit();

        const tw = TailWeights{
            .q_norm_w = q_norm_w,
            .k_norm_w = k_norm_w,
            .A_log = a_log,
            .dt_bias = dt_bias,
            .norm_w = norm_w,
            .ssm_state = ssm_state,
            .qk_idx = qk_idx,
        };

        const fused = (try gdnTailFused(ctx, qkv, a, b, z, tw, hv, cs.hk, d)) orelse
            return error.FusedKernelUnavailable;
        defer fused.y.deinit();
        defer fused.s_new.deinit();
        try fused.y.eval();
        try fused.s_new.eval();

        const eager = try gdnTailEager(ctx, qkv, a, b, z, tw, hv, cs.hk, d, qkv_dim);
        defer eager.y.deinit();
        defer eager.s_new.deinit();
        try eager.y.eval();
        try eager.s_new.eval();

        const got_y = try fused.y.dataSlice(f32);
        const exp_y = try eager.y.dataSlice(f32);
        try std.testing.expectEqual(exp_y.len, got_y.len);
        for (exp_y, got_y) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
        const got_s = try fused.s_new.dataSlice(f32);
        const exp_s = try eager.s_new.dataSlice(f32);
        try std.testing.expectEqual(exp_s.len, got_s.len);
        for (exp_s, got_s) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
    }
}

test "Qwen38GDN fused conv+tail kernel matches eager conv chain + eager tail" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const Case = struct { hv: i32, hk: i32, d: i32, k: i32, qkv_bf16: bool = false, conv_bf16: bool = false };
    for ([_]Case{
        .{ .hv = 6, .hk = 2, .d = 4, .k = 4 },
        .{ .hv = 6, .hk = 2, .d = 4, .k = 4, .qkv_bf16 = true, .conv_bf16 = true }, // real decode dtypes
        .{ .hv = 48, .hk = 16, .d = 128, .k = 4 },
    }) |cs| {
        const hv = cs.hv;
        const d = cs.d;
        const k = cs.k;
        const qkv_dim: i32 = (2 * cs.hk + hv) * d;

        const qkv_lin_f32 = try makeTestWeight(allocator, &[_]i32{ 1, qkv_dim });
        defer qkv_lin_f32.deinit();
        const qkv_lin = if (cs.qkv_bf16) try ops.astype(ctx, qkv_lin_f32, .bfloat16) else qkv_lin_f32;
        defer if (cs.qkv_bf16) qkv_lin.deinit();
        const conv_state = try makeTestWeight(allocator, &[_]i32{ k - 1, qkv_dim });
        defer conv_state.deinit();
        const conv1d_t_f32 = try makeTestWeight(allocator, &[_]i32{ k, qkv_dim });
        defer conv1d_t_f32.deinit();
        const conv1d_t = if (cs.conv_bf16) try ops.astype(ctx, conv1d_t_f32, .bfloat16) else conv1d_t_f32;
        defer if (cs.conv_bf16) conv1d_t.deinit();
        const a = try makeTestWeight(allocator, &[_]i32{ 1, hv });
        defer a.deinit();
        const b = try makeTestWeight(allocator, &[_]i32{ 1, hv });
        defer b.deinit();
        const z = try makeTestWeight(allocator, &[_]i32{ 1, hv * d });
        defer z.deinit();
        const ssm_state = try makeTestWeight(allocator, &[_]i32{ hv, d, d });
        defer ssm_state.deinit();
        const a_log = try makeTestWeight(allocator, &[_]i32{hv});
        defer a_log.deinit();
        const dt_bias = try makeTestWeight(allocator, &[_]i32{hv});
        defer dt_bias.deinit();
        const norm_w = try makeTestWeight(allocator, &[_]i32{d});
        defer norm_w.deinit();

        const inv_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
        const nw_buf = try allocator.alloc(f32, @intCast(d));
        defer allocator.free(nw_buf);
        @memset(nw_buf, inv_scale * inv_scale);
        const q_norm_w = try Array.fromData(allocator, f32, nw_buf, &[_]i32{d});
        defer q_norm_w.deinit();
        @memset(nw_buf, inv_scale);
        const k_norm_w = try Array.fromData(allocator, f32, nw_buf, &[_]i32{d});
        defer k_norm_w.deinit();

        const ratio: usize = @intCast(@divExact(hv, cs.hk));
        const idx_buf = try allocator.alloc(i32, @intCast(hv));
        defer allocator.free(idx_buf);
        for (idx_buf, 0..) |*vv, h| vv.* = @intCast(h / ratio);
        const qk_idx = try Array.fromData(allocator, i32, idx_buf, &[_]i32{hv});
        defer qk_idx.deinit();

        const tw = TailWeights{
            .q_norm_w = q_norm_w,
            .k_norm_w = k_norm_w,
            .A_log = a_log,
            .dt_bias = dt_bias,
            .norm_w = norm_w,
            .ssm_state = ssm_state,
            .qk_idx = qk_idx,
        };

        const fused = (try gdnConvTailFused(ctx, qkv_lin, conv_state, conv1d_t, a, b, z, tw, hv, cs.hk, d, k)) orelse
            return error.FusedKernelUnavailable;
        defer fused.y.deinit();
        defer fused.s_new.deinit();
        defer fused.conv_state.deinit();
        try fused.y.eval();
        try fused.s_new.eval();
        try fused.conv_state.eval();

        // Eager reference: concatenate → convAccum → slice/copy window → silu
        // → eager tail (the exact fallback path in step()).
        const conv_input = try shape_mod.concatenateAxis(ctx, &[_]Array{ conv_state, qkv_lin }, 0);
        defer conv_input.deinit();
        const conv_acc = try convAccumStep(ctx, conv1d_t, conv_input);
        defer conv_acc.deinit();
        const tail = try ops.slice(ctx, conv_input, &[_]i32{ 1, 0 }, &[_]i32{ k, qkv_dim }, &[_]i32{ 1, 1 });
        defer tail.deinit();
        const exp_conv = try ops.copy(ctx, tail);
        defer exp_conv.deinit();
        const qkv_e = try activations.silu(ctx, conv_acc);
        defer qkv_e.deinit();
        const eager = try gdnTailEager(ctx, qkv_e, a, b, z, tw, hv, cs.hk, d, qkv_dim);
        defer eager.y.deinit();
        defer eager.s_new.deinit();
        try eager.y.eval();
        try eager.s_new.eval();
        try exp_conv.eval();

        const got_y = try fused.y.dataSlice(f32);
        const exp_y = try eager.y.dataSlice(f32);
        try std.testing.expectEqual(exp_y.len, got_y.len);
        for (exp_y, got_y) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
        const got_s = try fused.s_new.dataSlice(f32);
        const exp_s = try eager.s_new.dataSlice(f32);
        try std.testing.expectEqual(exp_s.len, got_s.len);
        for (exp_s, got_s) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
        const got_cs = try fused.conv_state.dataSlice(f32);
        const exp_cs = try exp_conv.dataSlice(f32);
        try std.testing.expectEqual(exp_cs.len, got_cs.len);
        for (exp_cs, got_cs) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
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
            g.q_norm_w,    g.k_norm_w,            g.qk_idx,
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
