//! Qwen3.8 GDN (gated delta net) linear-attention module.
//!
//! Port of Qwen3-Next `Qwen3NextGatedDeltaNet` single-batch naive path.
//! Differs from GlmKDA: asymmetric heads (key Hk / value Hv, kv-group
//! broadcast), separate in_proj_qkv / in_proj_a / in_proj_b / in_proj_z
//! projections, softplus decay, rms_norm q/k (equals l2norm up to rounding),
//! silu output gate.
//! f32 throughout; cross-checked against scripts/qwen38_gdn_ref.py.
const std = @import("std");
const verify_qmm = @import("verify_qmm.zig");
const qmm_v8 = @import("qwen38_qmm_v8.zig");
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

// ============================================================
// GDN prefill section profiling (DMLX_GDN_PREFILL_PROF=1)
// ============================================================
// The prefill breakdown (plan doc §17) put GDN's NON-qmm cost at 1448 ms of a
// 17479 ms L=2048 prefill — the largest non-GEMM item and the only one worth
// attacking, since 4-bit qmm already runs at 90% of a dense bf16 matmul (so
// there is no GEMM headroom; see §18.0/A0). This splits GDN's prefill into
// front (qkv qmm + causal conv + gates) / recurrence (delta rule) / tail
// (out-gate norm + out_proj) to find which one holds that 1448 ms.
var gdn_prof: struct {
    // Verify-path counters (DMLX_VERIFY_PROF=1). The MTP round's non-qmm cost
    // is the whole remaining gap to mlx-vlm at matched depth — 19.7 ms/round,
    // of which ~17 ms is the verify forward's non-qmm part (plan doc §21.2).
    // Splitting prefillVerify the same way §17 split prefill is what locates it.
    v_front_ns: u64 = 0,
    v_rec_ns: u64 = 0,
    v_states_ns: u64 = 0,
    v_tail_ns: u64 = 0,
    v_layers: u64 = 0,
    // Host-side build-time counters (DMLX_MTP_DEBUG=1), no barriers.
    v_front_host_ns: u64 = 0,
    v_rec_host_ns: u64 = 0,
    v_tail_host_ns: u64 = 0,
    v_submit_host_ns: u64 = 0,
    v_layers_host: u64 = 0,
    f_proj_ns: u64 = 0,
    f_conv_ns: u64 = 0,
    f_gate_ns: u64 = 0,
    f_bcast_ns: u64 = 0,
    front_ns: u64 = 0,
    rec_ns: u64 = 0,
    tail_ns: u64 = 0,
    layers: u64 = 0,
} = .{};

/// Print and reset the host-side build-time counters (DMLX_MTP_DEBUG=1).
pub fn gdnVerifyHostProfReport() void {
    if (gdn_prof.v_layers_host == 0) return;
    const f = struct {
        fn ms(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    };
    const per = @as(f64, @floatFromInt(gdn_prof.v_layers_host));
    std.debug.print("gdn verify host build/layer: front={d:.3}ms rec={d:.3}ms tail={d:.3}ms submit={d:.3}ms (n_layers={d})\n", .{
        f.ms(gdn_prof.v_front_host_ns) / per,
        f.ms(gdn_prof.v_rec_host_ns) / per,
        f.ms(gdn_prof.v_tail_host_ns) / per,
        f.ms(gdn_prof.v_submit_host_ns) / per,
        gdn_prof.v_layers_host,
    });
    gdn_prof.v_front_host_ns = 0;
    gdn_prof.v_rec_host_ns = 0;
    gdn_prof.v_tail_host_ns = 0;
    gdn_prof.v_submit_host_ns = 0;
    gdn_prof.v_layers_host = 0;
}

var gdn_prof_enabled: ?bool = null;
var verify_prof_enabled: ?bool = null;

fn gdnVerifyProfEnabled() bool {
    if (verify_prof_enabled) |v| return v;
    const v = std.c.getenv("DMLX_VERIFY_PROF") != null;
    verify_prof_enabled = v;
    return v;
}

/// Print and reset the accumulated GDN VERIFY section times.
pub fn gdnVerifyProfReport() void {
    if (gdn_prof.v_layers == 0) return;
    const ms = struct {
        fn f(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.f;
    const total = gdn_prof.v_front_ns + gdn_prof.v_rec_ns + gdn_prof.v_states_ns + gdn_prof.v_tail_ns;
    const pct = struct {
        fn f(x: u64, t: u64) f64 {
            return 100.0 * @as(f64, @floatFromInt(x)) / @as(f64, @floatFromInt(@max(t, 1)));
        }
    }.f;
    std.debug.print("\n=== GDN VERIFY sections over {d} layer-passes (total {d:.1} ms) ===\n", .{ gdn_prof.v_layers, ms(total) });
    std.debug.print("  front  (4 proj qmm + conv + gates)     {d: >8.1} ms  {d: >5.1}%\n", .{ ms(gdn_prof.v_front_ns), pct(gdn_prof.v_front_ns, total) });
    std.debug.print("  recur  (fused retain, writes s_all)    {d: >8.1} ms  {d: >5.1}%\n", .{ ms(gdn_prof.v_rec_ns), pct(gdn_prof.v_rec_ns, total) });
    std.debug.print("  states (L conv-state slices + copies)  {d: >8.1} ms  {d: >5.1}%\n", .{ ms(gdn_prof.v_states_ns), pct(gdn_prof.v_states_ns, total) });
    std.debug.print("  tail   (out-gate norm + out_proj qmm)  {d: >8.1} ms  {d: >5.1}%\n", .{ ms(gdn_prof.v_tail_ns), pct(gdn_prof.v_tail_ns, total) });
    gdn_prof.v_front_ns = 0;
    gdn_prof.v_rec_ns = 0;
    gdn_prof.v_states_ns = 0;
    gdn_prof.v_tail_ns = 0;
    gdn_prof.v_layers = 0;
}

var verify_lazy_conv: ?bool = null;

/// Verify path keeps the whole conv `window` and slices the committed
/// position at commit time, instead of materializing L row copies per layer.
/// `DMLX_NO_VERIFY_LAZY_CONV=1` restores the eager per-position copies.
fn verifyLazyConv() bool {
    if (verify_lazy_conv) |v| return v;
    const v = std.c.getenv("DMLX_NO_VERIFY_LAZY_CONV") == null;
    verify_lazy_conv = v;
    return v;
}

var verify_no_sall: ?bool = null;

/// DIAGNOSTIC ONLY (`DMLX_VERIFY_NO_SALL=1`): run the verify recurrence with
/// the non-retain kernel, which writes just the final state instead of all L.
/// This prices the s_all traffic (L x 3 MB per GDN layer = 432 MB/round at
/// L=3). It commits the WRONG state whenever accepted < L-1, so output is
/// garbage — use it to read ms/round, never for correctness.
fn verifyNoSall() bool {
    if (verify_no_sall) |v| return v;
    const v = std.c.getenv("DMLX_VERIFY_NO_SALL") != null;
    verify_no_sall = v;
    return v;
}

var verify_no_recur: ?bool = null;

/// DIAGNOSTIC ONLY (`DMLX_VERIFY_NO_RECUR=1`): replace the entire verify
/// recurrence with a zero-cost stand-in. Prices the whole delta-rule step
/// (launch + 600 MB of state traffic + ALU) in ONE reading, without the
/// barrier distortion that makes the DMLX_VERIFY_PROF section shares
/// unusable as absolute numbers. Output is garbage.
fn verifyNoRecur() bool {
    if (verify_no_recur) |v| return v;
    const v = std.c.getenv("DMLX_VERIFY_NO_RECUR") != null;
    verify_no_recur = v;
    return v;
}

var verify_snapshot: ?bool = null;

/// Snapshot + replay-on-rejection instead of writing every position's state
/// (`DMLX_NO_VERIFY_SNAPSHOT=1` restores the s_all path). See
/// PrefillVerifyResult.replay.
fn verifySnapshot() bool {
    if (verify_snapshot) |v| return v;
    const v = std.c.getenv("DMLX_NO_VERIFY_SNAPSHOT") == null;
    verify_snapshot = v;
    return v;
}

fn gdnPrefillProfEnabled() bool {
    if (gdn_prof_enabled) |v| return v;
    const v = std.c.getenv("DMLX_GDN_PREFILL_PROF") != null;
    gdn_prof_enabled = v;
    return v;
}

fn nowNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

/// Print and reset the accumulated GDN prefill section times.
pub fn gdnPrefillProfReport() void {
    if (gdn_prof.layers == 0 and gdn_prof.f_proj_ns == 0) return;
    const ms = struct {
        fn f(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.f;
    const total = gdn_prof.front_ns + gdn_prof.rec_ns + gdn_prof.tail_ns;
    std.debug.print("\n=== GDN prefill sections over {d} layer-passes (total {d:.0} ms) ===\n", .{ gdn_prof.layers, ms(total) });
    std.debug.print("  front  (qkv qmm + causal conv + gates) {d: >9.1} ms  {d: >5.1}%\n", .{ ms(gdn_prof.front_ns), 100.0 * @as(f64, @floatFromInt(gdn_prof.front_ns)) / @as(f64, @floatFromInt(@max(total, 1))) });
    if (gdn_prof.f_proj_ns > 0) {
        std.debug.print("    front/ 4 projections (qmm)           {d: >9.1} ms\n", .{ms(gdn_prof.f_proj_ns)});
        std.debug.print("    front/ causal conv + window          {d: >9.1} ms\n", .{ms(gdn_prof.f_conv_ns)});
        std.debug.print("    front/ gates (softplus/exp/sigmoid)  {d: >9.1} ms\n", .{ms(gdn_prof.f_gate_ns)});
        std.debug.print("    front/ q,k head broadcast (gather)    {d: >9.1} ms\n", .{ms(gdn_prof.f_bcast_ns)});
    }
    std.debug.print("  recur  (delta rule, serial over L)     {d: >9.1} ms  {d: >5.1}%\n", .{ ms(gdn_prof.rec_ns), 100.0 * @as(f64, @floatFromInt(gdn_prof.rec_ns)) / @as(f64, @floatFromInt(@max(total, 1))) });
    std.debug.print("  tail   (out-gate norm + out_proj qmm)  {d: >9.1} ms  {d: >5.1}%\n", .{ ms(gdn_prof.tail_ns), 100.0 * @as(f64, @floatFromInt(gdn_prof.tail_ns)) / @as(f64, @floatFromInt(@max(total, 1))) });
    gdn_prof = .{};
}

// Lazy-init registry. Not synchronized: decode serving is serial
// (vtable_serial) and the test runner is single-threaded.
var fused_entries: [4]?FusedKernelEntry = .{ null, null, null, null };
var fused_gpu_stream: ?c.c.mlx_stream = null;
var fused_warned = false;

/// Default GPU stream, shared by every fused apply. Kernel work must land on
/// a GPU stream even when the caller's EagerContext targets the CPU (unit
/// tests); the default stream handle is process-lifetime and never freed.
pub fn fusedStream() c.c.mlx_stream {
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

pub fn makeStringVec(allocator: std.mem.Allocator, names: []const []const u8) !c.c.mlx_vector_string {
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

var decode_wide_rec: ?bool = null;

/// `DMLX_DECODE_WIDE_REC=1`: run the DECODE delta rule on the prefill
/// recurrence kernel's thread mapping (grid (32, D, HV) = 196608 threads,
/// one simdgroup per state row) instead of the decode kernels' one-thread-
/// per-row mapping (grid (D, HV) = 6144 threads).
///
/// Why (plan doc §25): the decode fused conv+tail kernel measured
/// 0.254 ms/layer against a 23 us memory floor — 11x off, because 6144
/// threads cannot cover memory latency on a part with 2560 ALU lanes. The
/// same math with the 32x wider mapping measured FREE in the verify path
/// (§23). Ablating the whole kernel is worth 12.2 ms/token (§25.1).
///
/// Cost: forces the eager conv + eager tail path, so the monolithic
/// conv+tail and tail kernels are bypassed and their elementwise work goes
/// back to separate ops. That trade is what this switch measures.
fn decodeWideRec() bool {
    if (decode_wide_rec) |v| return v;
    const v = std.c.getenv("DMLX_DECODE_WIDE_REC") != null;
    decode_wide_rec = v;
    return v;
}

var decode_eager_tail: ?bool = null;

/// Bypass BOTH monolithic decode kernels (`gdnConvTailFused`, `gdnTailFused`)
/// so the tail runs as eager ops with `deltaRuleFused` as a separate kernel.
/// `DMLX_DECODE_WIDE_REC` implies this (it has to, to reach the delta rule),
/// which is why the two must be measured separately: this switch alone prices
/// the eager penalty, so that `wide = eager+wide - eager` isolates the
/// mapping's real gain (plan doc §25.5).
fn decodeEagerTail() bool {
    if (decode_eager_tail) |v| return v;
    const v = std.c.getenv("DMLX_DECODE_EAGER_TAIL") != null or decodeWideRec();
    decode_eager_tail = v;
    return v;
}

/// Decode-shaped delta rule on the prefill kernel's wide mapping. Inputs are
/// the decode shapes ([HV, D] / [HV]); the L=1 reshapes are free views.
fn deltaRuleWide(
    ctx: EagerContext,
    ssm_state: Array,
    k_full: Array, // [HV, D] f32
    v: Array,
    beta: Array, // [HV] f32
    g: Array,
    q_full: Array,
    hv: i32,
    d: i32,
) !?DeltaRuleOut {
    const k3 = try ops.reshape(ctx, k_full, &[_]i32{ 1, hv, d });
    defer k3.deinit();
    const v3 = try ops.reshape(ctx, v, &[_]i32{ 1, hv, d });
    defer v3.deinit();
    const q3 = try ops.reshape(ctx, q_full, &[_]i32{ 1, hv, d });
    defer q3.deinit();
    const b2 = try ops.reshape(ctx, beta, &[_]i32{ 1, hv });
    defer b2.deinit();
    const g2 = try ops.reshape(ctx, g, &[_]i32{ 1, hv });
    defer g2.deinit();
    const rec = (try deltaRulePrefillFused(ctx, ssm_state, k3, v3, b2, g2, q3, hv, d)) orelse return null;
    errdefer rec.s_new.deinit();
    defer rec.y.deinit();
    const y2 = try ops.reshape(ctx, rec.y, &[_]i32{ hv, d });
    return .{ .s_new = rec.s_new, .y_f32 = y2 };
}

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

    if (decodeWideRec()) {
        if (try deltaRuleWide(ctx, ssm_state, k_full, v, beta, g, q_full, hv, d)) |w| return w;
    }

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

pub fn f32MetalLit(allocator: std.mem.Allocator, v: f32) ![:0]u8 {
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

// ============================================================
// Fused Metal kernel v4 ("wide"): same math as v3, re-mapped so the state
// access coalesces — §25 option B
// ============================================================
//
// v3 (`conv_tail_kernel_body`) launches grid (D, HV) = 6144 threads with one
// thread per state ROW. Two things follow, both bad:
//
//  1. Only HV = 48 threadgroups exist, because the kernel needs the head's
//     whole D-length q/k vector in threadgroup memory. 48 threadgroups is
//     2.4 per core on this M4 Pro (20 cores) and leaves cores idle outright
//     on any Mac with more than 48 (§25.7).
//  2. Thread `dv` reads `s_in[(h*D + dv)*D + i]`, so neighbouring lanes are
//     512 B apart: 32 lanes pull 32 separate 128 B cache lines for 512 B of
//     useful data — an 8x read AND write amplification on the only sizable
//     traffic in the kernel (6 MB/layer of state becomes ~48 MB/layer).
//     That is the bulk of the 12.2 ms/token §25.1 measured by ablation.
//
// v4 keeps the fusion and flips the mapping: grid (32, D, HV) with
// threadgroup (32, 4, 1) = 1536 threadgroups, and lane `dk` owns float4
// index `dk` of the row, so consecutive lanes touch consecutive 16 B and a
// simdgroup issues ONE coalesced 512 B transaction per row. The two
// row-length dot products (kv, readout) become `simd_sum`.
//
// Two consequences handled below:
//
//  - A threadgroup now owns only 4 state rows but still needs the head's
//    full q/k vectors, so phase 1 recomputes the whole head's conv with one
//    channel per thread (threadgroup size is exactly D = 128). The 32x
//    redundancy across a head's threadgroups is free in DRAM terms:
//    `conv_state` and `conv1d_t` are ~123 KB each and stay L2-resident.
//    `conv_out` writes are NOT free, so they are gated to the first
//    row-block (`thread_position_in_grid.y < 4`) and happen exactly once.
//  - The out-gate RMSNorm reduces over the head's whole y row, which no
//    longer lives in one threadgroup. It moves to a second, cheap kernel
//    (`out_gate_kernel_body`, grid (D, HV), reading a 24 KB y buffer).
//
// Requires D == 128 (threadgroup size must equal D so phase 1 covers every
// channel exactly once, and D/4 must equal the 32 lanes). Anything else
// falls back to v3.

const wide_tail_kernel_body =
    \\    const uint lane = thread_position_in_threadgroup.x;   // 0..31, owns float4 `lane`
    \\    const uint dv = thread_position_in_grid.y;            // state row
    \\    const uint h = thread_position_in_grid.z;
    \\    const uint hk = h / {[ratio]};
    \\    // Threadgroup is (32, 4, 1) = 128 threads == D, so this is the channel.
    \\    const uint tid = thread_position_in_threadgroup.x + 32 * thread_position_in_threadgroup.y;
    \\
    \\    threadgroup float qsh[{[d]}];
    \\    threadgroup float ksh[{[d]}];
    \\    threadgroup float vsh[{[d]}];
    \\
    \\    // Phase 1: the whole head's depthwise causal conv + silu, one
    \\    // channel per thread. Ascending j matches convAccumStep.
    \\    {{
    \\        const uint qbase = hk * {[d]} + tid;
    \\        const uint kbase = {[pk]} + hk * {[d]} + tid;
    \\        const uint vbase = 2 * {[pk]} + h * {[d]} + tid;
    \\        float qa = 0.0f;
    \\        float ka = 0.0f;
    \\        float va = 0.0f;
    \\        for (int j = 0; j < {[km1]}; ++j) {{
    \\            const uint off = j * {[qkv]};
    \\            qa += conv_state[off + qbase] * conv1d_t[off + qbase];
    \\            ka += conv_state[off + kbase] * conv1d_t[off + kbase];
    \\            va += conv_state[off + vbase] * conv1d_t[off + vbase];
    \\        }}
    \\        const uint loff = {[km1]} * {[qkv]};
    \\        qa += float(qkv_lin[qbase]) * conv1d_t[loff + qbase];
    \\        ka += float(qkv_lin[kbase]) * conv1d_t[loff + kbase];
    \\        va += float(qkv_lin[vbase]) * conv1d_t[loff + vbase];
    \\        qsh[tid] = qa / (1.0f + metal::exp(-qa));
    \\        ksh[tid] = ka / (1.0f + metal::exp(-ka));
    \\        vsh[tid] = va / (1.0f + metal::exp(-va));
    \\
    \\        // New conv window = window rows 1..K-1 (raw pre-conv values).
    \\        // Written by the FIRST row-block only: every threadgroup of this
    \\        // head computes the same values, and 32x redundant writes would
    \\        // cost real bandwidth (unlike the L2-resident reads above).
    \\        if (thread_position_in_grid.y < 4) {{
    \\            if (h % {[ratio]} == 0) {{
    \\                for (int r = 0; r < {[km2]}; ++r) {{
    \\                    conv_out[r * {[qkv]} + qbase] = conv_state[(r + 1) * {[qkv]} + qbase];
    \\                    conv_out[r * {[qkv]} + kbase] = conv_state[(r + 1) * {[qkv]} + kbase];
    \\                }}
    \\                conv_out[{[km2]} * {[qkv]} + qbase] = float(qkv_lin[qbase]);
    \\                conv_out[{[km2]} * {[qkv]} + kbase] = float(qkv_lin[kbase]);
    \\            }}
    \\            for (int r = 0; r < {[km2]}; ++r) {{
    \\                conv_out[r * {[qkv]} + vbase] = conv_state[(r + 1) * {[qkv]} + vbase];
    \\            }}
    \\            conv_out[{[km2]} * {[qkv]} + vbase] = float(qkv_lin[vbase]);
    \\        }}
    \\    }}
    \\
    \\    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
    \\    // This lane's slice of the head's q/k vectors (column dimension).
    \\    const metal::float4 q4 = metal::float4(qsh[4 * lane], qsh[4 * lane + 1], qsh[4 * lane + 2], qsh[4 * lane + 3]);
    \\    const metal::float4 k4 = metal::float4(ksh[4 * lane], ksh[4 * lane + 1], ksh[4 * lane + 2], ksh[4 * lane + 3]);
    \\
    \\    // Weight-less per-head rmsNorm factors (folded scales). v3 ran a
    \\    // redundant D-iteration serial loop in every thread; here the 32
    \\    // lanes split it and simd_sum combines.
    \\    const float ssq = metal::simd_sum(metal::dot(q4, q4));
    \\    const float ssk = metal::simd_sum(metal::dot(k4, k4));
    \\    const float qfac = {[qscale]s} * metal::rsqrt(ssq / {[d]}.0f + 1e-6f);
    \\    const float kfac = {[kscale]s} * metal::rsqrt(ssk / {[d]}.0f + 1e-6f);
    \\
    \\    // Per-head gates (f32), identical to v3.
    \\    const float ax = float(a[h]) + float(dt_bias[h]);
    \\    const float sp = ax > 20.0f ? ax : metal::log(1.0f + metal::exp(ax));
    \\    const float gh = metal::exp(-metal::exp(float(A_log[h])) * sp);
    \\    const float bh = float({[bt]s}(1.0f / (1.0f + metal::exp(-float(b[h])))));
    \\
    \\    // Delta rule. Lane `lane` owns float4 `lane` of state row dv, so the
    \\    // simdgroup's 32 accesses are one contiguous 512 B span.
    \\    const device metal::float4* srow_in = (const device metal::float4*)(s_in + (h * {[d]} + dv) * {[d]});
    \\    device metal::float4* srow_out = (device metal::float4*)(s_out + (h * {[d]} + dv) * {[d]});
    \\    const metal::float4 kk = kfac * k4;
    \\    metal::float4 s = srow_in[lane] * gh;
    \\    const float kv = metal::simd_sum(metal::dot(s, kk));
    \\    const float dl = (vsh[dv] - kv) * bh;
    \\    s = s + dl * kk;
    \\    srow_out[lane] = s;
    \\    const float yv = metal::simd_sum(metal::dot(s, qfac * q4));
    \\    if (lane == 0) y_out[h * {[d]} + dv] = yv;
    \\
;

// Out gate for the v4 split: norm the head's y row, weight multiply, silu(z)
// gate. Separate kernel because the reduction spans the head's whole row,
// which v4's narrow threadgroups no longer hold. Reads a 24 KB y buffer, so
// the extra launch is the only real cost.
const out_gate_kernel_body =
    \\    const uint dv = thread_position_in_grid.x;
    \\    const uint h = thread_position_in_grid.y;
    \\    threadgroup float ysh[{[d]}];
    \\    ysh[dv] = y_in[h * {[d]} + dv];
    \\    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    float ssy = 0.0f;
    \\    for (int i = 0; i < {[d]}; ++i) ssy += ysh[i] * ysh[i];
    \\    const float yn = ysh[dv] * metal::rsqrt(ssy / {[d]}.0f + 1e-6f);
    \\    const float yw = yn * float(norm_w[dv]);
    \\    const float zz = float(z[h * {[d]} + dv]);
    \\    const float zact = zz / (1.0f + metal::exp(-zz));
    \\    out[h * {[d]} + dv] = zact * yw;
    \\
;

// Dependency-depth probe (DMLX_DECODE_EXTRA_LAUNCH=N).
//
// §29.3 attributed the ~7 ms that v4 did NOT recover to dependent
// kernel-launch latency, on the arithmetic that 96 dependent launches per
// token at ~70-100 us each lands in that range. That was an INFERENCE, and
// the decision it feeds (transposing the SSM state layout so v3's
// single-kernel shape coalesces, which has a large blast radius) is too
// expensive to take on an unverified attribution.
//
// This prices dependency depth directly: insert N numerically-identity
// pass-through kernels into each GDN layer's serial chain and read the slope
// of ms/token against N. Each insertion adds exactly 48 dependent launches
// per token and no meaningful work (one float copy per thread over a 24 KB
// buffer), so slope / 48 is the per-dependent-launch cost.
const passthru_kernel_body =
    \\    const uint i = thread_position_in_grid.x + {[d]} * thread_position_in_grid.y;
    \\    y_out[i] = y_in[i];
    \\
;

var passthru_entries: [4]?struct { d: i32, kernel: c.c.mlx_fast_metal_kernel } = .{ null, null, null, null };
var extra_launch: ?i32 = null;

fn extraLaunchCount() i32 {
    if (extra_launch) |v| return v;
    const v = blk: {
        const s = std.c.getenv("DMLX_DECODE_EXTRA_LAUNCH") orelse break :blk 0;
        break :blk std.fmt.parseInt(i32, std.mem.span(s), 10) catch 0;
    };
    extra_launch = v;
    return v;
}

fn getPassthruKernel(allocator: std.mem.Allocator, d: i32) ?c.c.mlx_fast_metal_kernel {
    for (&passthru_entries) |*slot| {
        if (slot.*) |e| {
            if (e.d == d) return e.kernel;
        }
    }
    const name_z = std.fmt.allocPrintSentinel(allocator, "gdn_passthru_d{d}", .{d}, 0) catch return null;
    defer allocator.free(name_z);
    const source_z = std.fmt.allocPrintSentinel(allocator, passthru_kernel_body, .{ .d = d }, 0) catch return null;
    defer allocator.free(source_z);
    const header_z = allocator.dupeZ(u8, "") catch return null;
    defer allocator.free(header_z);
    const in_vec = makeStringVec(allocator, &.{"y_in"}) catch return null;
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = makeStringVec(allocator, &.{"y_out"}) catch return null;
    defer _ = c.c.mlx_vector_string_free(out_vec);
    const kernel = c.c.mlx_fast_metal_kernel_new(name_z.ptr, in_vec, out_vec, source_z.ptr, header_z.ptr, true, false);
    for (&passthru_entries) |*slot| {
        if (slot.* == null) {
            slot.* = .{ .d = d, .kernel = kernel };
            return kernel;
        }
    }
    return kernel;
}

/// Chain `n` identity kernels onto `y` ([HV, D] f32). Returns a new array.
fn passthruChain(ctx: EagerContext, y: Array, hv: i32, d: i32, n: i32) !Array {
    const kernel = getPassthruKernel(ctx.allocator, d) orelse return error.MlxError;
    var cur = try ops.reshape(ctx, y, &[_]i32{ hv, d }); // owned view
    var i: i32 = 0;
    while (i < n) : (i += 1) {
        const cfg = c.c.mlx_fast_metal_kernel_config_new();
        defer c.c.mlx_fast_metal_kernel_config_free(cfg);
        try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, d, hv, 1));
        try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, d, 1, 1));
        try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ hv, d }, 2, c.c.MLX_FLOAT32));
        const in_v = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(in_v);
        try c.check(c.c.mlx_vector_array_append_value(in_v, cur.inner));
        var out_v = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(out_v);
        try c.check(c.c.mlx_fast_metal_kernel_apply(&out_v, kernel, in_v, cfg, fusedStream()));
        var h = c.c.mlx_array_new();
        try c.check(c.c.mlx_vector_array_get(&h, out_v, 0));
        cur.deinit();
        cur = Array.fromHandle(h);
    }
    return cur;
}

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

// --- v4 "wide" kernel cache (two kernels; see wide_tail_kernel_body) -------

var wide_tail_entries: [8]?ConvTailKernelEntry = .{ null, null, null, null, null, null, null, null };
var out_gate_entries: [4]?struct { d: i32, kernel: c.c.mlx_fast_metal_kernel, broken: bool = false } =
    .{ null, null, null, null };

fn createWideTailKernel(allocator: std.mem.Allocator, key: ConvTailKernelKey) !c.c.mlx_fast_metal_kernel {
    const d = key.d;
    const inv_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
    const qscale_z = try f32MetalLit(allocator, inv_scale * inv_scale);
    defer allocator.free(qscale_z);
    const kscale_z = try f32MetalLit(allocator, inv_scale);
    defer allocator.free(kscale_z);
    const bt: []const u8 = if (key.b_bf16) "bfloat16_t" else "float";
    const qkv = key.pk * (2 + key.ratio);

    const name_z = try std.fmt.allocPrintSentinel(allocator, "gdn_wide_tail_d{d}_r{d}_p{d}_k{d}_{c}", .{ d, key.ratio, key.pk, key.k, if (key.b_bf16) @as(u8, 'b') else 'f' }, 0);
    defer allocator.free(name_z);
    const source_z = try std.fmt.allocPrintSentinel(allocator, wide_tail_kernel_body, .{
        .d = d,
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

    const in_vec = try makeStringVec(allocator, &.{ "qkv_lin", "conv_state", "conv1d_t", "a", "b", "A_log", "dt_bias", "s_in" });
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = try makeStringVec(allocator, &.{ "y_out", "s_out", "conv_out" });
    defer _ = c.c.mlx_vector_string_free(out_vec);

    return c.c.mlx_fast_metal_kernel_new(name_z.ptr, in_vec, out_vec, source_z.ptr, header_z.ptr, true, false);
}

fn createOutGateKernel(allocator: std.mem.Allocator, d: i32) !c.c.mlx_fast_metal_kernel {
    const name_z = try std.fmt.allocPrintSentinel(allocator, "gdn_out_gate_d{d}", .{d}, 0);
    defer allocator.free(name_z);
    const source_z = try std.fmt.allocPrintSentinel(allocator, out_gate_kernel_body, .{ .d = d }, 0);
    defer allocator.free(source_z);
    const header_z = try allocator.dupeZ(u8, "");
    defer allocator.free(header_z);
    const in_vec = try makeStringVec(allocator, &.{ "y_in", "z", "norm_w" });
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = try makeStringVec(allocator, &.{"out"});
    defer _ = c.c.mlx_vector_string_free(out_vec);
    return c.c.mlx_fast_metal_kernel_new(name_z.ptr, in_vec, out_vec, source_z.ptr, header_z.ptr, true, false);
}

fn getWideTailKernel(allocator: std.mem.Allocator, key: ConvTailKernelKey) ?c.c.mlx_fast_metal_kernel {
    for (&wide_tail_entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) return if (e.broken) null else e.kernel;
        }
    }
    const kernel = createWideTailKernel(allocator, key) catch |err| {
        convTailWarn(err);
        return null;
    };
    for (&wide_tail_entries) |*slot| {
        if (slot.* == null) {
            slot.* = .{ .key = key, .kernel = kernel };
            return kernel;
        }
    }
    return null;
}

fn getOutGateKernel(allocator: std.mem.Allocator, d: i32) ?c.c.mlx_fast_metal_kernel {
    for (&out_gate_entries) |*slot| {
        if (slot.*) |*e| {
            if (e.d == d) return if (e.broken) null else e.kernel;
        }
    }
    const kernel = createOutGateKernel(allocator, d) catch |err| {
        convTailWarn(err);
        return null;
    };
    for (&out_gate_entries) |*slot| {
        if (slot.* == null) {
            slot.* = .{ .d = d, .kernel = kernel };
            return kernel;
        }
    }
    return null;
}

fn markWideTailBroken(key: ConvTailKernelKey) void {
    for (&wide_tail_entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) e.broken = true;
        }
    }
}

var decode_wide_tail: ?bool = null;

/// v4 coalesced mapping (§25 option B) is the DEFAULT; set
/// `DMLX_DECODE_NO_WIDE_TAIL=1` to fall back to v3.
///
/// Gates it passed before becoming the default: ms/token 73.98 -> 70.73
/// (median of 3 interleaved passes with cooldown, won all 3), and
/// `scripts/decode_env_ab_text.py` 6/6 byte-identical greedy output. The
/// text diff is the only gate that covers this kernel at all — unit tests
/// run at D=4 where it is ineligible, and the mlx-lm oracle dump is a
/// prefill pass.
fn decodeWideTail() bool {
    if (decode_wide_tail) |v| return v;
    const v = std.c.getenv("DMLX_DECODE_NO_WIDE_TAIL") == null;
    decode_wide_tail = v;
    return v;
}

/// v4 conv+tail: grid (32, D, HV) / threadgroup (32, 4, 1) so the delta
/// rule's state access coalesces, plus a small second kernel for the
/// out gate. Same math as gdnConvTailFused; null when ineligible.
fn gdnConvTailWide(
    ctx: EagerContext,
    qkv_lin: Array,
    conv_state: Array,
    conv1d_t: Array,
    a: Array,
    b: Array,
    z: Array,
    tw: TailWeights,
    hv: i32,
    hk: i32,
    d: i32,
    k: i32,
) !?ConvTailOut {
    // Threadgroup size must equal D (phase 1 is one channel per thread) and
    // D/4 must equal the 32 lanes, so this kernel is D == 128 only.
    if (d != 128) return null;
    if (hk < 1 or @mod(hv, hk) != 0) return null;
    if (k < 2) return null;
    if (@abs(tw.norm_eps - 1e-6) > 1e-12) return null;
    if (!isFloatDtype(qkv_lin.dtype())) return null;
    if (conv_state.dtype() != .float32) return null;
    if (!isFloatDtype(conv1d_t.dtype())) return null;
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
    const kernel = getWideTailKernel(ctx.allocator, key) orelse return null;
    const gate_kernel = getOutGateKernel(ctx.allocator, d) orelse return null;
    const qkv_dim: i32 = key.pk * (2 + key.ratio);

    // --- kernel 1: conv + norms + gates + delta rule -> y, s_out, conv_out
    var y_dev: Array = undefined;
    var s_new: Array = undefined;
    var conv_new: Array = undefined;
    {
        const cfg = c.c.mlx_fast_metal_kernel_config_new();
        defer c.c.mlx_fast_metal_kernel_config_free(cfg);
        try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, 32, d, hv));
        try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 4, 1));
        try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ hv, d }, 2, c.c.MLX_FLOAT32));
        try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ hv, d, d }, 3, c.c.MLX_FLOAT32));
        try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ k - 1, qkv_dim }, 2, c.c.MLX_FLOAT32));

        const in1 = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(in1);
        for ([_]Array{ qkv_lin, conv_state, conv1d_t, a, b, tw.A_log, tw.dt_bias, tw.ssm_state }) |inp| {
            try c.check(c.c.mlx_vector_array_append_value(in1, inp.inner));
        }
        var out1 = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(out1);
        c.check(c.c.mlx_fast_metal_kernel_apply(&out1, kernel, in1, cfg, fusedStream())) catch |err| {
            markWideTailBroken(key);
            convTailWarn(err);
            return null;
        };
        var h0 = c.c.mlx_array_new();
        try c.check(c.c.mlx_vector_array_get(&h0, out1, 0));
        y_dev = Array.fromHandle(h0);
        errdefer y_dev.deinit();
        var h1 = c.c.mlx_array_new();
        try c.check(c.c.mlx_vector_array_get(&h1, out1, 1));
        s_new = Array.fromHandle(h1);
        errdefer s_new.deinit();
        var h2 = c.c.mlx_array_new();
        try c.check(c.c.mlx_vector_array_get(&h2, out1, 2));
        conv_new = Array.fromHandle(h2);
    }
    defer y_dev.deinit();
    errdefer s_new.deinit();
    errdefer conv_new.deinit();

    // Dependency-depth probe: identity kernels inserted into this layer's
    // serial chain. Numerically a no-op; see passthru_kernel_body.
    const n_extra = extraLaunchCount();
    const y_chained: ?Array = if (n_extra > 0)
        try passthruChain(ctx, y_dev, hv, d, n_extra)
    else
        null;
    defer if (y_chained) |yc| yc.deinit();
    const y_for_gate = y_chained orelse y_dev;

    // --- kernel 2: out gate -> [1, HV*D]
    const cfg2 = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg2);
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg2, d, hv, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg2, d, 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg2, &[_]i32{ 1, hv * d }, 2, c.c.MLX_FLOAT32));

    const in2 = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(in2);
    for ([_]Array{ y_for_gate, z, tw.norm_w }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(in2, inp.inner));
    }
    var out2 = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(out2);
    c.check(c.c.mlx_fast_metal_kernel_apply(&out2, gate_kernel, in2, cfg2, fusedStream())) catch |err| {
        markWideTailBroken(key);
        convTailWarn(err);
        return null;
    };
    var hy = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&hy, out2, 0));
    return .{ .y = Array.fromHandle(hy), .s_new = s_new, .conv_state = conv_new };
}

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
    // Wide-recurrence experiment (§25.3 option A) needs the eager conv +
    // eager tail path, because the delta rule is buried inside this
    // monolithic kernel and cannot be re-mapped from outside.
    if (decodeEagerTail()) return null;
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
    // See gdnConvTailFused: the wide-recurrence path needs gdnTailEager,
    // which is the only tail that calls deltaRuleFused as a separate kernel.
    if (decodeEagerTail()) return null;
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

pub const PrefillRecurrenceOut = struct {
    s_new: Array, // [HV, D, D] f32, final state (owned)
    y: Array, // [L, HV, D] f32, per-step outputs stacked (owned)
};

// ============================================================
// Single-kernel sequential delta-rule recurrence for prefill
// ============================================================
//
// Port of mlx-lm gated_delta.py `_make_gated_delta_kernel`: one Metal kernel
// runs the whole L-step recurrence with the state held in registers.
// Layout: grid (32, Dv, Hv) threads, threadgroup (32, 4, 1) — the 32 lanes of
// a simdgroup split Dk (n_per_t = Dk/32 floats per thread), thread dv owns
// state row dv, and a serial for-t loop does decay → kv_mem (simd_sum over
// Dk) → delta → rank-1 update → readout (simd_sum) → y[t]. Our q/k are
// already broadcast to value heads ([L, HV, D]), so the hk mapping in the
// reference drops out. Everything is f32 (prefill's post-conv tail is f32),
// so the reference's InT/StT casts are identities. Eligibility: D % 32 == 0
// (simd split); smaller/odd head dims fall back to the eager per-step loop.

const prefill_rec_kernel_body =
    \\    const uint hv = thread_position_in_grid.z;
    \\    const uint dv = thread_position_in_grid.y;
    \\    const uint dk = thread_position_in_threadgroup.x;
    \\    const int T = T_in[0];
    \\
    \\    // state: [HV, Dv, Dk]; this thread owns row dv, Dk slice dk.
    \\    const device float* i_state = s_in + (hv * {[d]} + dv) * {[d]};
    \\    device float* o_state = s_out + (hv * {[d]} + dv) * {[d]};
    \\    float state[{[npt]}];
    \\    for (int i = 0; i < {[npt]}; ++i) {{
    \\        state[i] = i_state[{[npt]} * dk + i];
    \\    }}
    \\
    \\    // q, k, v, y: [L, HV, D]; g, beta: [L, HV].
    \\    // NB: inputs with fewer elements than MLX's constant-space cutoff are
    \\    // passed in the constant address space, so small-input pointers use
    \\    // auto (g/beta are [L, HV] — tiny for short prompts / small heads).
    \\    const device float* q_ = q + hv * {[d]};
    \\    const device float* k_ = k + hv * {[d]};
    \\    const device float* v_ = v + hv * {[d]};
    \\    device float* y_ = y + hv * {[d]};
    \\    auto g_ = g + hv;
    \\    auto beta_ = beta + hv;
    \\
    \\    for (int t = 0; t < T; ++t) {{
    \\        const float gh = g_[0];
    \\        float kv = 0.0f;
    \\        for (int i = 0; i < {[npt]}; ++i) {{
    \\            const uint s_idx = {[npt]} * dk + i;
    \\            state[i] = state[i] * gh;
    \\            kv += state[i] * k_[s_idx];
    \\        }}
    \\        kv = metal::simd_sum(kv);
    \\        const float delta = (v_[dv] - kv) * beta_[0];
    \\        float out = 0.0f;
    \\        for (int i = 0; i < {[npt]}; ++i) {{
    \\            const uint s_idx = {[npt]} * dk + i;
    \\            state[i] = state[i] + k_[s_idx] * delta;
    \\            out += state[i] * q_[s_idx];
    \\        }}
    \\        out = metal::simd_sum(out);
    \\        if (dk == 0) y_[dv] = out; // lane 0 of the simdgroup owns the write
    \\        q_ += {[hvd]};
    \\        k_ += {[hvd]};
    \\        v_ += {[hvd]};
    \\        y_ += {[hvd]};
    \\        g_ += {[hv]};
    \\        beta_ += {[hv]};
    \\    }}
    \\    for (int i = 0; i < {[npt]}; ++i) {{
    \\        o_state[{[npt]} * dk + i] = state[i];
    \\    }}
    \\
;

// Verify variant: same recurrence, plus a per-position state output s_all
// [L, HV, Dv, Dk] written inside the t-loop after each rank-1 update, so a
// later commit can adopt any accepted position's state without re-running.
// Replaces the per-step eager retain loop (one kernel launch per layer
// instead of L × (12 view ops + 1 step kernel)).
const prefill_rec_retain_kernel_body =
    \\    const uint hv = thread_position_in_grid.z;
    \\    const uint dv = thread_position_in_grid.y;
    \\    const uint dk = thread_position_in_threadgroup.x;
    \\    const int T = T_in[0];
    \\
    \\    // state: [HV, Dv, Dk]; this thread owns row dv, Dk slice dk.
    \\    const device float* i_state = s_in + (hv * {[d]} + dv) * {[d]};
    \\    float state[{[npt]}];
    \\    for (int i = 0; i < {[npt]}; ++i) {{
    \\        state[i] = i_state[{[npt]} * dk + i];
    \\    }}
    \\
    \\    // q, k, v, y: [L, HV, D]; g, beta: [L, HV]; s_all: [L, HV, D, D].
    \\    // NB: inputs with fewer elements than MLX's constant-space cutoff are
    \\    // passed in the constant address space, so input pointers use auto.
    \\    auto q_ = q + hv * {[d]};
    \\    auto k_ = k + hv * {[d]};
    \\    auto v_ = v + hv * {[d]};
    \\    device float* y_ = y + hv * {[d]};
    \\    auto g_ = g + hv;
    \\    auto beta_ = beta + hv;
    \\    device float* s_all_ = s_all + (hv * {[d]} + dv) * {[d]};
    \\
    \\    for (int t = 0; t < T; ++t) {{
    \\        const float gh = g_[0];
    \\        float kv = 0.0f;
    \\        for (int i = 0; i < {[npt]}; ++i) {{
    \\            const uint s_idx = {[npt]} * dk + i;
    \\            state[i] = state[i] * gh;
    \\            kv += state[i] * k_[s_idx];
    \\        }}
    \\        kv = metal::simd_sum(kv);
    \\        const float delta = (v_[dv] - kv) * beta_[0];
    \\        float out = 0.0f;
    \\        for (int i = 0; i < {[npt]}; ++i) {{
    \\            const uint s_idx = {[npt]} * dk + i;
    \\            state[i] = state[i] + k_[s_idx] * delta;
    \\            out += state[i] * q_[s_idx];
    \\        }}
    \\        out = metal::simd_sum(out);
    \\        if (dk == 0) y_[dv] = out; // lane 0 of the simdgroup owns the write
    \\        for (int i = 0; i < {[npt]}; ++i) {{
    \\            s_all_[{[npt]} * dk + i] = state[i];
    \\        }}
    \\        q_ += {[hvd]};
    \\        k_ += {[hvd]};
    \\        v_ += {[hvd]};
    \\        y_ += {[hvd]};
    \\        g_ += {[hv]};
    \\        beta_ += {[hv]};
    \\        s_all_ += {[hvdd]};
    \\    }}
    \\
;

const PrefillRecKernelKey = struct {
    d: i32,
    hv: i32,
    retain: bool = false,
};

// omlx's software-pipelined blocked-sequential GDN prefill recurrence
// (omlx/custom_kernels/attention/gated_delta.py `_pipelined_source(12)`,
// verbatim). At T=1901 on M4 Pro this runs 3.96 ms/layer where our
// per-lane-serial kernel above measures ~27 ms (DMLX_GDN_PREFILL_PROF): the
// exact recurrence is issue/latency-bound and this layout hides the
// cross-lane reduction latency. Same per-step math; only the fp32 summation
// order of the two 128-wide dots differs (documented in the source header).
const pipe_kernel_body = @embedFile("qwen38_gdn_pipe.metal");

const PipeKernelKey = struct { hk: i32, hv: i32, bf16: bool = false };
const PipeKernelEntry = struct {
    key: PipeKernelKey,
    kernel: c.c.mlx_fast_metal_kernel,
    broken: bool = false,
};
var pipe_entries: [4]?PipeKernelEntry = .{ null, null, null, null };
var pipe_warned = false;
var pipe_enabled: ?bool = null;

fn pipeWarn(err: anyerror) void {
    if (!pipe_warned) {
        pipe_warned = true;
        std.log.warn("pipelined GDN prefill kernel unavailable ({s}); using the serial fused kernel", .{@errorName(err)});
    }
}

fn pipeEnabled() bool {
    if (pipe_enabled == null) {
        if (std.c.getenv("DMLX_GDN_PIPE_PREFILL")) |v| {
            pipe_enabled = !(v[0] == '0' and v[1] == 0);
        } else pipe_enabled = true;
    }
    return pipe_enabled.?;
}

var pipe_bf16: ?bool = null;

fn pipeBf16Enabled() bool {
    if (pipe_bf16 == null) {
        if (std.c.getenv("DMLX_GDN_PIPE_BF16")) |v| {
            pipe_bf16 = !(v[0] == '0' and v[1] == 0);
        } else pipe_bf16 = true;
    }
    return pipe_bf16.?;
}

fn createPipeKernel(allocator: std.mem.Allocator, key: PipeKernelKey) !c.c.mlx_fast_metal_kernel {
    const name_z = try std.fmt.allocPrintSentinel(allocator, "gdn_prefill_pipe_h{d}_v{d}{s}", .{ key.hk, key.hv, if (key.bf16) "_bf" else "" }, 0);
    defer allocator.free(name_z);
    const header_z = try allocator.dupeZ(u8, "#include <metal_stdlib>\nusing namespace metal;\n");
    defer allocator.free(header_z);
    const source_z = try allocator.dupeZ(u8, pipe_kernel_body);
    defer allocator.free(source_z);

    const in_vec = try makeStringVec(allocator, &.{ "q", "k", "v", "g", "beta", "state_in", "T_in" });
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = try makeStringVec(allocator, &.{ "y", "state_out" });
    defer _ = c.c.mlx_vector_string_free(out_vec);

    const kernel = c.c.mlx_fast_metal_kernel_new(name_z.ptr, in_vec, out_vec, source_z.ptr, header_z.ptr, true, false);
    return kernel;
}

fn getPipeKernel(allocator: std.mem.Allocator, key: PipeKernelKey) ?c.c.mlx_fast_metal_kernel {
    for (&pipe_entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) return if (e.broken) null else e.kernel;
        }
    }
    const kernel = createPipeKernel(allocator, key) catch |err| {
        pipeWarn(err);
        return null;
    };
    for (&pipe_entries) |*slot| {
        if (slot.* == null) {
            slot.* = .{ .key = key, .kernel = kernel };
            return kernel;
        }
    }
    return null;
}

fn markPipeBroken(key: PipeKernelKey) void {
    for (&pipe_entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) e.broken = true;
        }
    }
}

/// Pipelined whole-sequence delta-rule recurrence (omlx Kernel P). Same
/// in/out contract as deltaRulePrefillFused. Eligible when the head dims fit
/// the kernel's fixed layout: Dk == Dv == 128, Dv % 16 == 0, Hv % Hk == 0,
/// all inputs f32. Null → caller falls back.
fn deltaRulePrefillPipelined(
    ctx: EagerContext,
    ssm_state: Array, // [HV, D, D] f32
    k_full: Array, // [L, HK, D] f32/bf16
    v: Array, // [L, HV, D] f32/bf16
    beta: Array, // [L, HV] f32
    g: Array, // [L, HV] f32
    q_full: Array, // [L, HK, D] f32/bf16
    hk: i32,
    hv: i32,
    d: i32,
) !?PrefillRecurrenceOut {
    if (!pipeEnabled()) return null;
    if (d != 128 or @mod(hv, 16) != 0 or hk < 1 or @mod(hv, hk) != 0) return null;
    const L: i32 = @intCast(k_full.shape()[0]);
    if (L < 1) return null;
    const dt_f32 = dtype_mod.Dtype.float32;
    const in_dt = q_full.dtype();
    if (in_dt != dt_f32 and in_dt != .bfloat16) return null;
    if (k_full.dtype() != in_dt or v.dtype() != in_dt) return null;
    if (ssm_state.dtype() != dt_f32 or g.dtype() != dt_f32 or beta.dtype() != dt_f32) return null;
    const in_bf16 = in_dt == .bfloat16;

    const key = PipeKernelKey{ .hk = hk, .hv = hv, .bf16 = in_bf16 };
    const kernel = getPipeKernel(ctx.allocator, key) orelse return null;

    const t_arr = try Array.fromData(ctx.allocator, i32, &[_]i32{L}, &[_]i32{1});
    defer t_arr.deinit();

    const cfg = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg);
    // grid (128 * (Dv/16), Hv, 1) threads, threadgroup (128, 1, 1).
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, 128 * @divTrunc(d, 16), hv, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ 1, L, hv, d }, 4, if (in_bf16) c.c.MLX_BFLOAT16 else c.c.MLX_FLOAT32));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ 1, hv, d, d }, 4, c.c.MLX_FLOAT32));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", if (in_bf16) c.c.MLX_BFLOAT16 else c.c.MLX_FLOAT32));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dk", d));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dv", d));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Hk", hk));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Hv", hv));

    const input_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(input_vec);
    for ([_]Array{ q_full, k_full, v, g, beta, ssm_state, t_arr }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(input_vec, inp.inner));
    }
    var output_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(output_vec);
    c.check(c.c.mlx_fast_metal_kernel_apply(&output_vec, kernel, input_vec, cfg, fusedStream())) catch |err| {
        markPipeBroken(key);
        pipeWarn(err);
        return null;
    };

    var y_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&y_handle, output_vec, 0));
    errdefer _ = c.c.mlx_array_free(y_handle);
    var s_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&s_handle, output_vec, 1));
    // y comes out [1, L, HV, D]; state [1, HV, D, D] — drop the unit axis.
    const y4 = Array.fromHandle(y_handle);
    defer y4.deinit();
    const y = try ops.reshape(ctx, y4, &[_]i32{ L, hv, d });
    errdefer y.deinit();
    const s4 = Array.fromHandle(s_handle);
    defer s4.deinit();
    const s_new = try ops.reshape(ctx, s4, &[_]i32{ hv, d, d });
    return .{ .s_new = s_new, .y = y };
}

const PrefillRecKernelEntry = struct {
    key: PrefillRecKernelKey,
    kernel: c.c.mlx_fast_metal_kernel,
    broken: bool = false,
};

var prefill_rec_entries: [4]?PrefillRecKernelEntry = .{ null, null, null, null };
var prefill_rec_warned = false;

fn prefillRecWarn(err: anyerror) void {
    if (!prefill_rec_warned) {
        prefill_rec_warned = true;
        std.log.warn("fused GDN prefill recurrence kernel unavailable ({s}); using eager per-step loop", .{@errorName(err)});
    }
}

fn createPrefillRecKernel(allocator: std.mem.Allocator, key: PrefillRecKernelKey) !c.c.mlx_fast_metal_kernel {
    const name_z = try std.fmt.allocPrintSentinel(allocator, "gdn_prefill_rec{s}_d{d}_h{d}", .{ if (key.retain) "_ret" else "", key.d, key.hv }, 0);
    defer allocator.free(name_z);
    const source_z = if (key.retain)
        try std.fmt.allocPrintSentinel(allocator, prefill_rec_retain_kernel_body, .{
            .d = key.d,
            .npt = @divExact(key.d, 32),
            .hv = key.hv,
            .hvd = key.hv * key.d,
            .hvdd = key.hv * key.d * key.d,
        }, 0)
    else
        try std.fmt.allocPrintSentinel(allocator, prefill_rec_kernel_body, .{
            .d = key.d,
            .npt = @divExact(key.d, 32),
            .hv = key.hv,
            .hvd = key.hv * key.d,
        }, 0);
    defer allocator.free(source_z);
    const header_z = try allocator.dupeZ(u8, "");
    defer allocator.free(header_z);

    const in_vec = try makeStringVec(allocator, &.{ "q", "k", "v", "g", "beta", "s_in", "T_in" });
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = try makeStringVec(allocator, if (key.retain) &.{ "y", "s_all" } else &.{ "y", "s_out" });
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

fn getPrefillRecKernel(allocator: std.mem.Allocator, key: PrefillRecKernelKey) ?c.c.mlx_fast_metal_kernel {
    for (&prefill_rec_entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) return if (e.broken) null else e.kernel;
        }
    }
    const kernel = createPrefillRecKernel(allocator, key) catch |err| {
        prefillRecWarn(err);
        return null;
    };
    for (&prefill_rec_entries) |*slot| {
        if (slot.* == null) {
            slot.* = .{ .key = key, .kernel = kernel };
            return kernel;
        }
    }
    return null;
}

fn markPrefillRecBroken(key: PrefillRecKernelKey) void {
    for (&prefill_rec_entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) e.broken = true;
        }
    }
}

/// Single-kernel whole-sequence delta-rule recurrence. Returns null when the
/// shape/dtype is unsupported or the kernel is unavailable (caller falls back
/// to the eager per-step loop).
fn deltaRulePrefillFused(
    ctx: EagerContext,
    ssm_state: Array, // [HV, D, D] f32
    k_full: Array, // [L, HV, D] f32
    v: Array, // [L, HV, D] f32
    beta: Array, // [L, HV] f32
    g: Array, // [L, HV] f32
    q_full: Array, // [L, HV, D] f32
    hv: i32,
    d: i32,
) !?PrefillRecurrenceOut {
    if (@mod(d, 32) != 0 or d < 32) return null;
    const L: i32 = @intCast(k_full.shape()[0]);
    if (L < 1) return null;
    const dt_f32 = dtype_mod.Dtype.float32;
    if (ssm_state.dtype() != dt_f32 or q_full.dtype() != dt_f32 or k_full.dtype() != dt_f32 or
        v.dtype() != dt_f32 or g.dtype() != dt_f32 or beta.dtype() != dt_f32) return null;

    const key = PrefillRecKernelKey{ .d = d, .hv = hv };
    const kernel = getPrefillRecKernel(ctx.allocator, key) orelse return null;

    const t_arr = try Array.fromData(ctx.allocator, i32, &[_]i32{L}, &[_]i32{1});
    defer t_arr.deinit();

    const cfg = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg);
    // mlx-lm layout: grid (32, Dv, Hv) threads, threadgroup (32, 4, 1).
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, 32, d, hv));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 4, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ L, hv, d }, 3, c.c.MLX_FLOAT32));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ hv, d, d }, 3, c.c.MLX_FLOAT32));

    const input_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(input_vec);
    for ([_]Array{ q_full, k_full, v, g, beta, ssm_state, t_arr }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(input_vec, inp.inner));
    }
    var output_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(output_vec);
    c.check(c.c.mlx_fast_metal_kernel_apply(&output_vec, kernel, input_vec, cfg, fusedStream())) catch |err| {
        markPrefillRecBroken(key);
        prefillRecWarn(err);
        return null;
    };

    var y_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&y_handle, output_vec, 0));
    errdefer _ = c.c.mlx_array_free(y_handle);
    var s_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&s_handle, output_vec, 1));
    return .{ .s_new = Array.fromHandle(s_handle), .y = Array.fromHandle(y_handle) };
}

pub const PrefillRetainFusedOut = struct {
    y: Array, // [L, HV, D] f32
    s_all: Array, // [L, HV, D, D] f32 — state after each step (one buffer)
};

/// Verify-mode fused recurrence: single Metal kernel over the whole (tiny)
/// verify block, additionally writing the post-step state of EVERY position
/// into s_all. Same eligibility rules as deltaRulePrefillFused; null → caller
/// falls back to deltaRulePrefillEagerRetain.
fn deltaRulePrefillFusedRetain(
    ctx: EagerContext,
    ssm_state: Array, // [HV, D, D] f32
    k_full: Array, // [L, HV, D] f32
    v: Array, // [L, HV, D] f32
    beta: Array, // [L, HV] f32
    g: Array, // [L, HV] f32
    q_full: Array, // [L, HV, D] f32
    hv: i32,
    d: i32,
) !?PrefillRetainFusedOut {
    if (@mod(d, 32) != 0 or d < 32) return null;
    // DIAGNOSTIC ONLY (`DMLX_VERIFY_EAGER_REC=1`): force the eager per-step
    // loop. The fused kernel path leaks ~0.4 MB per layer per round of MLX
    // active memory (MemProbe, 2026-10-30); this isolates kernel vs eager.
    if (std.c.getenv("DMLX_VERIFY_EAGER_REC") != null) return null;
    const L: i32 = @intCast(k_full.shape()[0]);
    if (L < 1) return null;
    const dt_f32 = dtype_mod.Dtype.float32;
    if (ssm_state.dtype() != dt_f32 or q_full.dtype() != dt_f32 or k_full.dtype() != dt_f32 or
        v.dtype() != dt_f32 or g.dtype() != dt_f32 or beta.dtype() != dt_f32) return null;

    const key = PrefillRecKernelKey{ .d = d, .hv = hv, .retain = true };
    const kernel = getPrefillRecKernel(ctx.allocator, key) orelse return null;

    const t_arr = try Array.fromData(ctx.allocator, i32, &[_]i32{L}, &[_]i32{1});
    defer t_arr.deinit();

    const cfg = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg);
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, 32, d, hv));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 4, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ L, hv, d }, 3, c.c.MLX_FLOAT32));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ L, hv, d, d }, 4, c.c.MLX_FLOAT32));

    const input_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(input_vec);
    for ([_]Array{ q_full, k_full, v, g, beta, ssm_state, t_arr }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(input_vec, inp.inner));
    }
    var output_vec2 = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(output_vec2);
    c.check(c.c.mlx_fast_metal_kernel_apply(&output_vec2, kernel, input_vec, cfg, fusedStream())) catch |err| {
        markPrefillRecBroken(key);
        prefillRecWarn(err);
        return null;
    };

    var y_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&y_handle, output_vec2, 0));
    errdefer _ = c.c.mlx_array_free(y_handle);
    var s_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&s_handle, output_vec2, 1));
    return .{ .y = Array.fromHandle(y_handle), .s_all = Array.fromHandle(s_handle) };
}

/// Prefill recurrence dispatch: single Metal kernel over the whole sequence
/// when eligible (D % 32 == 0, f32), otherwise the eager per-step loop.
fn deltaRulePrefill(
    ctx: EagerContext,
    allocator: std.mem.Allocator,
    ssm_state: Array,
    k_full: Array,
    v: Array,
    beta: Array,
    g: Array,
    q_full: Array,
    hv: i32,
    d: i32,
) !PrefillRecurrenceOut {
    return (try deltaRulePrefillFused(ctx, ssm_state, k_full, v, beta, g, q_full, hv, d)) orelse
        deltaRulePrefillEager(ctx, allocator, ssm_state, k_full, v, beta, g, q_full, hv, d);
}

/// Test/bench hooks for the two whole-sequence recurrence kernels.
pub fn testDeltaRuleSerial(ctx: EagerContext, ssm_state: Array, k_full: Array, v: Array, beta: Array, g: Array, q_full: Array, hv: i32, d: i32) !?PrefillRecurrenceOut {
    return deltaRulePrefillFused(ctx, ssm_state, k_full, v, beta, g, q_full, hv, d);
}

pub fn testDeltaRulePipelined(ctx: EagerContext, ssm_state: Array, k_full: Array, v: Array, beta: Array, g: Array, q_full: Array, hk: i32, hv: i32, d: i32) !?PrefillRecurrenceOut {
    return deltaRulePrefillPipelined(ctx, ssm_state, k_full, v, beta, g, q_full, hk, hv, d);
}

/// Eager per-step delta-rule recurrence over a prefilled sequence (fallback
/// for head dims the single kernel does not support, e.g. toy D < 32).
/// k_full / q_full / v: [L, HV, D] f32; beta / g: [L, HV] f32; ssm_state:
/// [HV, D, D] f32 (borrowed). Per step this runs the exact decode recurrence
/// (deltaRuleFused when eligible, deltaRuleEager otherwise) on [HV, D]
/// slices, then asyncEvals the new state so the recurrent lazy graph and its
/// intermediates stay O(1) per step instead of accumulating over L.
fn deltaRulePrefillEager(
    ctx: EagerContext,
    allocator: std.mem.Allocator,
    ssm_state: Array,
    k_full: Array,
    v: Array,
    beta: Array,
    g: Array,
    q_full: Array,
    hv: i32,
    d: i32,
) !PrefillRecurrenceOut {
    const L: i32 = @intCast(k_full.shape()[0]);
    if (L < 1) return .{ .s_new = try ops.copy(ctx, ssm_state), .y = try array_mod.zeros(allocator, &[_]i32{ 0, hv, d }, .float32) };

    var state: ?Array = null; // owned state from the previous step (null = borrow ssm_state)
    errdefer if (state) |s| s.deinit();
    var ys = std.ArrayList(Array).empty;
    defer ys.deinit(allocator);
    errdefer for (ys.items) |yi| yi.deinit();
    try ys.ensureTotalCapacity(allocator, @intCast(L));

    var t: i32 = 0;
    while (t < L) : (t += 1) {
        const cur = state orelse ssm_state;
        const k_sl = try ops.slice(ctx, k_full, &[_]i32{ t, 0, 0 }, &[_]i32{ t + 1, hv, d }, &[_]i32{ 1, 1, 1 });
        defer k_sl.deinit();
        const k_t = try ops.reshape(ctx, k_sl, &[_]i32{ hv, d });
        defer k_t.deinit();
        const q_sl = try ops.slice(ctx, q_full, &[_]i32{ t, 0, 0 }, &[_]i32{ t + 1, hv, d }, &[_]i32{ 1, 1, 1 });
        defer q_sl.deinit();
        const q_t = try ops.reshape(ctx, q_sl, &[_]i32{ hv, d });
        defer q_t.deinit();
        const v_sl = try ops.slice(ctx, v, &[_]i32{ t, 0, 0 }, &[_]i32{ t + 1, hv, d }, &[_]i32{ 1, 1, 1 });
        defer v_sl.deinit();
        const v_t = try ops.reshape(ctx, v_sl, &[_]i32{ hv, d });
        defer v_t.deinit();
        const beta_sl = try ops.slice(ctx, beta, &[_]i32{ t, 0 }, &[_]i32{ t + 1, hv }, &[_]i32{ 1, 1 });
        defer beta_sl.deinit();
        const beta_t = try ops.reshape(ctx, beta_sl, &[_]i32{hv});
        defer beta_t.deinit();
        const g_sl = try ops.slice(ctx, g, &[_]i32{ t, 0 }, &[_]i32{ t + 1, hv }, &[_]i32{ 1, 1 });
        defer g_sl.deinit();
        const g_t = try ops.reshape(ctx, g_sl, &[_]i32{hv});
        defer g_t.deinit();

        const dr = (try deltaRuleFused(ctx, cur, k_t, v_t, beta_t, g_t, q_t, hv, d)) orelse
            (try deltaRuleEager(ctx, cur, k_t, v_t, beta_t, g_t, q_t, hv, d));
        if (state) |s| s.deinit();
        state = dr.s_new;

        const y_t = try ops.expandDims(ctx, dr.y_f32, 0); // [1, HV, D]
        dr.y_f32.deinit();
        ys.appendAssumeCapacity(y_t);
        // Periodic blocking sync on the state: bounds lazy-graph depth and
        // intermediate memory. Never (async)eval the stacked ys here: a
        // Metal shared event is released only when the evaled array is
        // freed, the ys live until the end-of-prefill concat, and L x
        // n_layer live events exhaust the pool ("Failed to create Metal
        // shared event"). The state arrays are freed as the recurrence
        // advances, so syncing only the state is safe.
        if (@mod(t, 16) == 15 or t == L - 1) try dr.s_new.eval();
    }

    const y = try shape_mod.concatenateAxis(ctx, ys.items, 0); // [L, HV, D] f32
    errdefer y.deinit();
    for (ys.items) |yi| yi.deinit();
    ys.clearRetainingCapacity();
    return .{ .s_new = state.?, .y = y };
}

pub const PrefillRetainOut = struct {
    y: Array, // [L, HV, D] f32
    ssm_states: []Array, // [L] — state after each step (owned)
};

/// Verify-mode per-step recurrence: identical math to deltaRulePrefillEager
/// but retains every step's post-update state for later commit/rollback
/// (verify sequences are tiny, L ≤ ~5, so retaining L × [HV,D,D] f32 is
/// fine). ssm_state borrowed; the initial state is NOT included in the
/// output list (position t = after step t).
fn deltaRulePrefillEagerRetain(
    ctx: EagerContext,
    allocator: std.mem.Allocator,
    ssm_state: Array,
    k_full: Array,
    v: Array,
    beta: Array,
    g: Array,
    q_full: Array,
    hv: i32,
    d: i32,
) !PrefillRetainOut {
    const L: i32 = @intCast(k_full.shape()[0]);
    const states = try allocator.alloc(Array, @intCast(L));
    errdefer allocator.free(states);
    var made: usize = 0;
    errdefer for (states[0..made]) |s| s.deinit();

    var ys = std.ArrayList(Array).empty;
    defer ys.deinit(allocator);
    errdefer for (ys.items) |yi| yi.deinit();
    try ys.ensureTotalCapacity(allocator, @intCast(L));

    var t: i32 = 0;
    while (t < L) : (t += 1) {
        const cur = if (t == 0) ssm_state else states[@intCast(t - 1)];
        const k_sl = try ops.slice(ctx, k_full, &[_]i32{ t, 0, 0 }, &[_]i32{ t + 1, hv, d }, &[_]i32{ 1, 1, 1 });
        defer k_sl.deinit();
        const k_t = try ops.reshape(ctx, k_sl, &[_]i32{ hv, d });
        defer k_t.deinit();
        const q_sl = try ops.slice(ctx, q_full, &[_]i32{ t, 0, 0 }, &[_]i32{ t + 1, hv, d }, &[_]i32{ 1, 1, 1 });
        defer q_sl.deinit();
        const q_t = try ops.reshape(ctx, q_sl, &[_]i32{ hv, d });
        defer q_t.deinit();
        const v_sl = try ops.slice(ctx, v, &[_]i32{ t, 0, 0 }, &[_]i32{ t + 1, hv, d }, &[_]i32{ 1, 1, 1 });
        defer v_sl.deinit();
        const v_t = try ops.reshape(ctx, v_sl, &[_]i32{ hv, d });
        defer v_t.deinit();
        const beta_sl = try ops.slice(ctx, beta, &[_]i32{ t, 0 }, &[_]i32{ t + 1, hv }, &[_]i32{ 1, 1 });
        defer beta_sl.deinit();
        const beta_t = try ops.reshape(ctx, beta_sl, &[_]i32{hv});
        defer beta_t.deinit();
        const g_sl = try ops.slice(ctx, g, &[_]i32{ t, 0 }, &[_]i32{ t + 1, hv }, &[_]i32{ 1, 1 });
        defer g_sl.deinit();
        const g_t = try ops.reshape(ctx, g_sl, &[_]i32{hv});
        defer g_t.deinit();

        const dr = (try deltaRuleFused(ctx, cur, k_t, v_t, beta_t, g_t, q_t, hv, d)) orelse
            (try deltaRuleEager(ctx, cur, k_t, v_t, beta_t, g_t, q_t, hv, d));
        states[@intCast(t)] = dr.s_new; // retained, NOT freed — survives the loop
        made += 1;

        const y_t = try ops.expandDims(ctx, dr.y_f32, 0); // [1, HV, D]
        dr.y_f32.deinit();
        ys.appendAssumeCapacity(y_t);
        // Flush at the last step only (L is tiny); retained states share the
        // chain and are materialized by the caller's asyncEval / final sync.
        if (t == L - 1) try dr.s_new.eval();
    }

    const y = try shape_mod.concatenateAxis(ctx, ys.items, 0); // [L, HV, D] f32
    errdefer y.deinit();
    for (ys.items) |yi| yi.deinit();
    ys.clearRetainingCapacity();
    return .{ .y = y, .ssm_states = states };
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
        // See verify_qmm.zig: armed only around the MTP verify forward.
        if (try verify_qmm.maybeQmm(ctx, x, qw)) |y| return y;
        // Long-prefill rows: omlx's affine_q4_qmm_t variant-8 tile (~1.05x at
        // every projection shape, bit-exact; bench_prefill_v8).
        if (try qmm_v8.maybeApply(ctx, x, w, s, biases)) |y| return y;
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

/// Whole-sequence causal depthwise conv for prefill: out[t] =
/// sum_j window[t+j] * conv1d_t[j] for j in 0..K-1, ascending j (same order
/// as the fused decode conv kernel), f32 throughout. window is the
/// [L+K-1, QKV] concat of conv_state and the raw qkv projection (already
/// promoted to f32); the final astype in convAccumStep is the identity here.
// ============================================================
// Fused whole-sequence causal conv (+ silu)
// ============================================================
//
// `convAccumSeq` below is the eager form: for each of the K taps it slices the
// window, multiplies by the tap's per-channel weight and adds into an
// accumulator. At prefill shapes that is the largest NON-qmm cost in the whole
// forward (plan doc §19.2): at L=2048, QKV=10240 each tap reads an 84 MB
// strided slice and writes 84 MB, the adds read/write another 84 MB each, so
// one layer moves ~1.43 GB and 48 layers move ~68 GB. The caller also builds
// `window` with a `concatenateAxis`, another 84 MB copy per layer, and then
// applies `silu` in a separate pass (another 168 MB).
//
// This kernel does the whole thing in one pass: it reads `conv_state` and the
// projection output SEPARATELY (so no concatenated window is needed), applies
// the K taps, folds in silu, and writes the result once — ~8 GB instead of
// ~68 GB plus the concat and silu passes.
//
// The new conv state does not need the window either: window rows
// [L, L+K-1) are, by construction, the last K-1 rows of the projection
// output, so the caller slices those directly (valid while L >= K-1, which
// prefill always satisfies; shorter inputs fall back to the eager path).
//
// Accumulation order matches the eager form exactly (tap 0 first, then += in
// tap order) — the delta-rule recurrence downstream is sensitive to conv
// rounding, and HANDOFF records an earlier FMA rewrite that passed smoke but
// cost 10.6% and broke Paris, so the order is kept and gated by a unit test
// against the eager path.
/// Each thread produces TPT consecutive output rows for one channel.
///
/// The first version did one output per thread, which needs K strided global
/// reads per output (stride QKV*4 = 40 KB at the real width). Interleaved A/B
/// showed that lost ~2% of prefill slope at L=2048 even though it won 4% at
/// L=512 — consecutive rows were re-read instead of reused. Producing TPT rows
/// per thread loads TPT+K-1 rows once and reuses them from registers:
/// 7 reads per 4 outputs instead of 16.
///
/// Generated fully unrolled with named registers, never thread-local arrays:
/// the Metal compiler spills such arrays unless the trip count is a power of
/// two, which cost 10x in the verify kernels (plan doc §9.5).
fn buildConvSeqSource(allocator: std.mem.Allocator, k: i32, tpt: i32) ![]u8 {
    var s = std.ArrayList(u8).empty;
    errdefer s.deinit(allocator);
    const Emit = struct {
        a: std.mem.Allocator,
        buf: *std.ArrayList(u8),
        fn raw(self: @This(), text: []const u8) !void {
            try self.buf.appendSlice(self.a, text);
        }
        fn fmt(self: @This(), comptime f: []const u8, args: anytype) !void {
            var tmp: [512]u8 = undefined;
            try self.buf.appendSlice(self.a, try std.fmt.bufPrint(&tmp, f, args));
        }
    };
    const w = Emit{ .a = allocator, .buf = &s };

    try w.raw(
        \\    const uint c = thread_position_in_grid.x;
        \\    const int t0 = int(thread_position_in_grid.y) * TPT;
        \\    if (c >= QKV || t0 >= L) { return; }
        \\
    );
    // Tap weights for this channel, reused by every row below.
    var j: i32 = 0;
    while (j < k) : (j += 1) {
        try w.fmt("    const float wv{d} = w[{d} * QKV + c];\n", .{ j, j });
    }
    try w.raw(
        \\
        \\    // Logical window row r: the first KM1 rows come from the carried
        \\    // conv state, the rest from this call's projection output. Rows
        \\    // past the end are zero and their outputs are never stored.
        \\
    );
    var i: i32 = 0;
    while (i < k - 1 + tpt) : (i += 1) {
        try w.fmt(
            \\    const int r{d} = t0 + {d};
            \\    const float xv{d} = (r{d} < KM1) ? cs[r{d} * QKV + c]
            \\        : ((r{d} - KM1) < L ? x[(r{d} - KM1) * QKV + c] : 0.0f);
            \\
        , .{ i, i, i, i, i, i, i });
    }
    try w.raw("\n");
    var row: i32 = 0;
    while (row < tpt) : (row += 1) {
        try w.fmt("    if (t0 + {d} < L) {{\n", .{row});
        // Same accumulation order as the eager taps: tap 0 first, then +=.
        try w.fmt("        float acc = xv{d} * wv0;\n", .{row});
        j = 1;
        while (j < k) : (j += 1) {
            try w.fmt("        acc += xv{d} * wv{d};\n", .{ row + j, j });
        }
        // silu, folded in to avoid a separate full-tensor pass.
        try w.fmt(
            "        y[(t0 + {d}) * QKV + c] = acc / (1.0f + metal::exp(-acc));\n    }}\n",
            .{row},
        );
    }
    return s.toOwnedSlice(allocator);
}

/// Output rows per thread. 4 keeps the reused window rows (TPT+K-1 = 7
/// floats) plus the K tap weights comfortably in registers.
const conv_seq_tpt: i32 = 4;

var conv_seq_kernel: ?c.c.mlx_fast_metal_kernel = null;
var conv_seq_broken: bool = false;

fn getConvSeqKernel(allocator: std.mem.Allocator) ?c.c.mlx_fast_metal_kernel {
    if (conv_seq_broken) return null;
    if (conv_seq_kernel) |k| return k;
    const name_z = allocator.dupeZ(u8, "gdn_conv_seq_silu") catch return null;
    defer allocator.free(name_z);
    const body = buildConvSeqSource(allocator, 4, conv_seq_tpt) catch return null;
    defer allocator.free(body);
    const src_z = allocator.dupeZ(u8, body) catch return null;
    defer allocator.free(src_z);
    const header_z = allocator.dupeZ(u8, "") catch return null;
    defer allocator.free(header_z);
    const in_vec = makeStringVec(allocator, &.{ "x", "cs", "w" }) catch return null;
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = makeStringVec(allocator, &.{"y"}) catch return null;
    defer _ = c.c.mlx_vector_string_free(out_vec);
    const k = c.c.mlx_fast_metal_kernel_new(
        name_z.ptr,
        in_vec,
        out_vec,
        src_z.ptr,
        header_z.ptr,
        true, // ensure_row_contiguous
        false,
    );
    conv_seq_kernel = k;
    return k;
}

/// silu(causal_conv(concat(conv_state, x))) in one kernel. Returns null when
/// ineligible (caller falls back to `convAccumSeq` + separate silu).
var conv_fusion_off: ?bool = null;

fn convFusionDisabled() bool {
    if (conv_fusion_off) |v| return v;
    const v = std.c.getenv("DMLX_NO_CONV_FUSION") != null;
    conv_fusion_off = v;
    return v;
}

fn convAccumSeqFused(
    ctx: EagerContext,
    conv1d_t_f32: Array,
    conv_state: Array,
    x_f32: Array,
    L: i32,
    qkv_dim: i32,
    K: i32,
) !?Array {
    if (convFusionDisabled()) return null;
    if (L < K - 1) return null; // new-state slice below assumes L >= K-1
    if (conv1d_t_f32.dtype() != .float32 or x_f32.dtype() != .float32 or
        conv_state.dtype() != .float32) return null;
    const allocator = ctx.allocator;
    const kernel = getConvSeqKernel(allocator) orelse return null;

    const cfg = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg);
    // One thread per output element, channel-major so the K reads coalesce
    // across the threadgroup.
    const rows_per_thread = conv_seq_tpt;
    const grid_y = @divFloor(L + rows_per_thread - 1, rows_per_thread);
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, qkv_dim, grid_y, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ L, qkv_dim }, 2, c.c.MLX_FLOAT32));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "L", L));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "QKV", qkv_dim));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", K));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KM1", K - 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TPT", rows_per_thread));

    const in_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(in_vec);
    for ([_]Array{ x_f32, conv_state, conv1d_t_f32 }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(in_vec, inp.inner));
    }
    var out_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(out_vec);
    c.check(c.c.mlx_fast_metal_kernel_apply(&out_vec, kernel, in_vec, cfg, fusedStream())) catch |err| {
        conv_seq_broken = true;
        prefillRecWarn(err);
        return null;
    };
    var h = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&h, out_vec, 0));
    return Array.fromHandle(h);
}

/// Bench hook for the fused whole-sequence causal conv (+silu inside).
pub fn testConvAccumSeqFused(ctx: EagerContext, conv1d_t_f32: Array, conv_state: Array, x_f32: Array, L: i32, qkv_dim: i32, K: i32) !?Array {
    return convAccumSeqFused(ctx, conv1d_t_f32, conv_state, x_f32, L, qkv_dim, K);
}

fn convAccumSeq(ctx: EagerContext, conv1d_t: Array, window: Array, L: i32, qkv_dim: i32) !Array {
    const k: i32 = @intCast(conv1d_t.shape()[0]);
    const w_f32 = try ops.astype(ctx, conv1d_t, .float32);
    defer w_f32.deinit();
    const x_f32 = try ops.astype(ctx, window, .float32);
    defer x_f32.deinit();
    var acc: ?Array = null;
    errdefer if (acc) |a| a.deinit();
    var j: i32 = 0;
    while (j < k) : (j += 1) {
        const xj = try ops.slice(ctx, x_f32, &[_]i32{ j, 0 }, &[_]i32{ j + L, qkv_dim }, &[_]i32{ 1, 1 });
        defer xj.deinit();
        const wj = try ops.slice(ctx, w_f32, &[_]i32{ j, 0 }, &[_]i32{ j + 1, qkv_dim }, &[_]i32{ 1, 1 });
        defer wj.deinit();
        const prod = try ops.multiply(ctx, xj, wj); // [L, QKV] * [1, QKV]
        if (acc) |a| {
            acc = try ops.add(ctx, a, prod);
            a.deinit();
            prod.deinit();
        } else {
            acc = prod;
        }
    }
    return acc.?;
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

/// The verify/prefill gate chain as one compiled kernel.
///
/// `prefillFront` computes the gates as 11 separate elementwise dispatches
/// over tiny [L, HV] tensors — add, exp, log1p, scalar, greater, where,
/// multiply, negative, exp, sigmoid, astype. The verify forward reuses
/// prefill's code, so that runs 48 times per verify round, and the decode
/// path by contrast does the whole of its equivalent inside one monolithic
/// kernel (`gdnConvTailWide`).
///
/// omlx fuses exactly this region. `qwen35_gdn_prework` collapses its
/// conv-concat / conv1d / SiLU / split / two ones-weight RMS norms / scalar
/// scales / next-conv-slice chain into a single Metal launch, and its own
/// header prices the unfused version at ~10 dispatches per GDN layer per
/// verify cycle, "the largest remaining verify-cycle cost after the
/// attention split" (48 layers x ~0.29 ms measured on their side).
///
/// This is the cheap half of that idea: the gates are pure elementwise over
/// one shape, so `mlx_compile` fuses them with no hand-written Metal and no
/// new numerics. Op order is preserved exactly, so results are bit-identical
/// to the eager chain. The gathers and RMS norms are left alone — they are
/// not elementwise and would need the real kernel.
fn gateClosureBody(inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array {
    if (inputs.len != 4) return error.MlxError;
    const ctx = closureGpuCtx(allocator);
    defer ctx.deinit();
    const a_flat = inputs[0];
    const dt_bias = inputs[1];
    const e_a_log = inputs[2];
    const b_flat = inputs[3];

    const a_dt = ops.add(ctx, a_flat, dt_bias) catch return error.MlxError;
    defer a_dt.deinit();
    const sp = softplusStep(ctx, a_dt) catch return error.MlxError;
    defer sp.deinit();
    const g_pre = ops.multiply(ctx, e_a_log, sp) catch return error.MlxError;
    defer g_pre.deinit();
    const g_neg = ops.negative(ctx, g_pre) catch return error.MlxError;
    defer g_neg.deinit();
    const g = ops.exp(ctx, g_neg) catch return error.MlxError;
    errdefer g.deinit();

    const beta_raw = ops.sigmoid(ctx, b_flat) catch return error.MlxError;
    defer beta_raw.deinit();
    const beta = ops.astype(ctx, beta_raw, .float32) catch return error.MlxError;
    errdefer beta.deinit();

    const out = allocator.alloc(Array, 2) catch return error.MlxError;
    out[0] = g;
    out[1] = beta;
    return out;
}

var gate_closure: ?Closure = null;
var gate_closure_failed: bool = false;

/// Compiled gate chain, or null when compilation is off or has failed.
fn gateClosure(allocator: std.mem.Allocator) ?Closure {
    // OPT-IN. Collapsing the 11 elementwise gate dispatches into one compiled
    // kernel is bit-identical (decode_env_ab_text 6/6 byte-identical) and
    // measurement-neutral, so it does not earn being on by default:
    //   code  mean +3.1%  median +9.9%  pairwise 55/100 (50 = chance)
    //   math  mean -0.9%  median +0.4%  pairwise 47/100
    // (10 draws/arm, artifacts/fusedgates_code_20261104.json,
    // fusedgates_math_20261104.json. The code median moved more than the
    // distribution did — the two sets straddle each other, which is what a
    // median does on overlapping samples and why the pairwise count is the
    // statistic to read.)
    //
    // The useful part is what this rules OUT. omlx's `qwen35_gdn_prework`
    // header attributes its verify-cycle win to collapsing ~10 dispatches per
    // GDN layer, and the natural reading is that the dispatch COUNT is the
    // cost. On this machine it is not: removing 11 of ~27 non-qmm dispatches
    // per GDN layer, ~530 per verify round, changes nothing measurable. That
    // agrees with the earlier finding that dependent launch cost here is ~0
    // (DMLX_DECODE_EXTRA_LAUNCH, +192 launches = 0 ms).
    //
    // So the verify forward's non-qmm time is real tensor work, not launch
    // overhead: the conv kernel, the two rmsNorms, the two takeAxis gathers
    // and the 7-op tail. A fused prework kernel would have to subsume THOSE
    // to pay, and the gathers and norms are the parts that actually move
    // bytes. Enable with DMLX_GDN_FUSED_GATES=1.
    if (std.c.getenv("DMLX_GDN_FUSED_GATES") == null) return null;
    if (gate_closure) |cl| return cl;
    if (gate_closure_failed) return null;
    const base = Closure.init(&gateClosureBody, allocator) catch {
        gate_closure_failed = true;
        return null;
    };
    defer base.deinit();
    const compiled = compile_mod.compile(base, false) catch {
        gate_closure_failed = true;
        return null;
    };
    gate_closure = compiled;
    return compiled;
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
    // DIAGNOSTIC ONLY (`DMLX_DECODE_NO_GDN_TAIL=1`): zero-cost stand-in for
    // the fused conv+tail kernel — the ONLY non-qmm GPU kernel in a GDN
    // decode layer. All five qmm stay, so bytes/token is unchanged; what
    // disappears is one kernel per layer plus the recurrence's state traffic.
    // This is the §24.5 experiment: if the weight stream's efficiency jumps,
    // the cost is the bubble these kernels punch in the stream, not their own
    // duration. `reshape` to the identical shape is a free view, so the
    // stand-in really costs nothing. Output is garbage.
    if (std.c.getenv("DMLX_DECODE_NO_GDN_TAIL") != null) {
        // The stand-in MUST keep every input projection alive. The first
        // version used only `z`, which left `qkv_lin` unconsumed — and MLX is
        // lazy, so the whole in_proj_qkv qmm ([10240, 5120] 4-bit = 29.5 MB
        // per layer, 1.42 GB per token, ~6.1 ms) was silently pruned along
        // with the tail. That inflated the measured "tail cost" by about 2x
        // and is what §25.1's 12.2 ms actually contained (plan doc §30).
        // Adding the two [1, 6144] rows costs ~24 KB and keeps both qmms.
        const qkv_head = try ops.slice(ctx, qkv_lin, &[_]i32{ 0, 0 }, &[_]i32{ 1, HV * D }, &[_]i32{ 1, 1 });
        defer qkv_head.deinit();
        const z_row = try ops.reshape(ctx, z, &[_]i32{ 1, HV * D });
        defer z_row.deinit();
        const y_sub = try ops.add(ctx, qkv_head, z_row);
        errdefer y_sub.deinit();
        const cs_sub = try ops.reshape(ctx, w.conv_state, w.conv_state.shape());
        errdefer cs_sub.deinit();
        const ss_sub = try ops.reshape(ctx, w.ssm_state, w.ssm_state.shape());
        errdefer ss_sub.deinit();
        const out_sub = try linearQuantParts(ctx, y_sub, w.out_w, w.out_scales, w.out_biases);
        y_sub.deinit();
        return .{ .out = out_sub, .conv_state = cs_sub, .ssm_state = ss_sub };
    }
    const wide: ?ConvTailOut = if (decodeWideTail())
        try gdnConvTailWide(ctx, qkv_lin, w.conv_state, w.conv1d_t, a, b, z, tw, HV, HK, D, K)
    else
        null;
    const tail_res: ConvTailOut = wide orelse (try gdnConvTailFused(ctx, qkv_lin, w.conv_state, w.conv1d_t, a, b, z, tw, HV, HK, D, K)) orelse blk: {
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
    // Hoisted per-round constants (optional; null = compute per call).
    // prefillFront otherwise re-casts conv1d_t to f32 and re-computes
    // exp(A_log) on EVERY prefill/verify call — two ops x 48 layers per round.
    conv1d_t_f32: ?Array = null, // [K, QKV] f32
    e_a_log: ?Array = null, // [HV] f32, exp(astype(A_log, f32))

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
        if (self.conv1d_t_f32) |a| a.deinit();
        if (self.e_a_log) |a| a.deinit();
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
    /// applies the same barrier per recurrence step in the eager fallback
    /// (deltaRulePrefillEager); the fused recurrence kernel needs none.
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

    /// Everything prefill computes before the delta-rule recurrence: batched
    /// projections, whole-sequence causal conv, vectorized q/k norms and
    /// gates. Shared by prefill (fused/eager recurrence, cache swap) and
    /// prefillVerify (per-step recurrence with retained states, no swap).
    const PrefillFront = struct {
        qkv: Array, // [L, QKV] f32, post-conv+silu — dtype round-trip reference
        q_full: Array, // [L, HV, D] f32
        k_full: Array, // [L, HV, D] f32
        q_heads: Array, // [L, HK, D] f32 — unbroadcast; the pipelined recurrence maps hv -> hv/(HV/HK) itself
        k_heads: Array, // [L, HK, D] f32
        v: Array, // [L, HV, D] f32
        beta: Array, // [L, HV] f32
        g: Array, // [L, HV] f32
        z: Array, // [1, L, HV*D]
        window: Array, // [L+K-1, QKV] f32 — per-step conv states are row slices
        new_conv_state: Array, // [K-1, QKV] f32 — state after the last step
        conv_moved: bool = false, // prefill moves new_conv_state into the cache
        window_moved: bool = false, // verify moves `window` into its result
        /// The snapshot+replay verify path moves the recurrence inputs
        /// (q_full/k_full/v/beta/g) into its result so a rejection can replay
        /// the accepted prefix. See PrefillVerifyResult.replay.
        rec_moved: bool = false,

        fn deinit(self: *PrefillFront) void {
            self.qkv.deinit();
            self.q_heads.deinit();
            self.k_heads.deinit();
            if (!self.rec_moved) {
                self.q_full.deinit();
                self.k_full.deinit();
                self.v.deinit();
                self.beta.deinit();
                self.g.deinit();
            }
            self.z.deinit();
            if (!self.window_moved) self.window.deinit();
            if (!self.conv_moved) self.new_conv_state.deinit();
        }
    };

    fn prefillFront(self: *Qwen38GDN, x: Array, cache: *Qwen38GdnCache) !PrefillFront {
        const ctx = self.ctx;
        const L: i32 = @intCast(x.shape()[1]);
        const HV: i32 = @intCast(self.num_value_heads);
        const HK: i32 = @intCast(self.num_key_heads);
        const D: i32 = @intCast(self.head_dim);
        const K: i32 = @intCast(self.conv_kernel);
        const QKV: i32 = (2 * HK + HV) * D;
        const pk = HK * D;

        // 1. Batched projections: [1, L, hidden] -> [1, L, *].
        const fp = gdnPrefillProfEnabled();
        var ft: i128 = if (fp) nowNs() else 0;
        // While the verify-qmm routing is armed, the four in-projections
        // share this input and go out in ONE sg8 group launch (plan doc:
        // close-decode-gap fusion program F1). Falls back to the four
        // separate calls when anything is ineligible.
        var qkv_lin: Array = undefined;
        var a: Array = undefined;
        var b: Array = undefined;
        var z: Array = undefined;
        var grouped = false;
        if (self.qkv_scales != null and self.a_scales != null and
            self.b_scales != null and self.z_scales != null and
            self.qkv_biases != null and self.a_biases != null and
            self.b_biases != null and self.z_biases != null)
        {
            const group_qws = [_]quantize_mod.QuantizedWeight{
                .{ .data = self.in_proj_qkv, .scales = self.qkv_scales.?, .biases = self.qkv_biases.?, .config = quantConfigFromShapes(x, self.in_proj_qkv, self.qkv_scales.?), .original_shape = &[_]i32{} },
                .{ .data = self.in_proj_a, .scales = self.a_scales.?, .biases = self.a_biases.?, .config = quantConfigFromShapes(x, self.in_proj_a, self.a_scales.?), .original_shape = &[_]i32{} },
                .{ .data = self.in_proj_b, .scales = self.b_scales.?, .biases = self.b_biases.?, .config = quantConfigFromShapes(x, self.in_proj_b, self.b_scales.?), .original_shape = &[_]i32{} },
                .{ .data = self.in_proj_z, .scales = self.z_scales.?, .biases = self.z_biases.?, .config = quantConfigFromShapes(x, self.in_proj_z, self.z_scales.?), .original_shape = &[_]i32{} },
            };
            if (try verify_qmm.maybeGroupQmm(ctx, x, &group_qws)) |*gout| {
                var go = gout.*;
                qkv_lin = go.outs[0];
                a = go.outs[1];
                b = go.outs[2];
                z = go.outs[3];
                go.count = 0; // ownership moved to the four locals
                grouped = true;
            }
        }
        if (!grouped) {
            qkv_lin = try linearQuantParts(ctx, x, self.in_proj_qkv, self.qkv_scales, self.qkv_biases);
            a = try linearQuantParts(ctx, x, self.in_proj_a, self.a_scales, self.a_biases);
            b = try linearQuantParts(ctx, x, self.in_proj_b, self.b_scales, self.b_biases);
            z = try linearQuantParts(ctx, x, self.in_proj_z, self.z_scales, self.z_biases);
        }
        defer qkv_lin.deinit(); // [1, L, QKV]
        defer a.deinit(); // [1, L, HV]
        defer b.deinit(); // [1, L, HV]
        errdefer z.deinit(); // [1, L, HV*D]

        if (fp) {
            qkv_lin.eval() catch {};
            a.eval() catch {};
            b.eval() catch {};
            z.eval() catch {};
            gdn_prof.f_proj_ns += @intCast(nowNs() - ft);
            ft = nowNs();
        }

        // 2. Causal depthwise conv over the whole sequence (f32) + silu.
        const qkv_lin2 = try ops.reshape(ctx, qkv_lin, &[_]i32{ L, QKV });
        defer qkv_lin2.deinit();
        // `window` stays for the VERIFY path, which slices per-position conv
        // states out of it. In the prefill path nothing consumes it (the fused
        // kernel reads conv_state and qkv_lin2 separately, and the new conv
        // state is sliced from qkv_lin2), so MLX never materializes the
        // concat — keeping it is free. Do NOT eval it in the profiler, or the
        // dead 84 MB copy comes back.
        const window = try shape_mod.concatenateAxis(ctx, &[_]Array{ cache.conv_state, qkv_lin2 }, 0); // [L+K-1, QKV] f32
        errdefer window.deinit();

        // Fast path: one kernel for the K taps + silu, no concatenated
        // window and no separate silu pass (see `convAccumSeqFused`).
        var used_fused = true;
        const w_f32 = if (self.conv1d_t_f32) |w| w else try ops.astype(ctx, self.conv1d_t, .float32);
        defer if (self.conv1d_t_f32 == null) w_f32.deinit();
        const qkv = blk: {
            if (try convAccumSeqFused(ctx, w_f32, cache.conv_state, qkv_lin2, L, QKV, K)) |fused| {
                break :blk fused;
            }
            used_fused = false;
            const conv_acc = try convAccumSeq(ctx, self.conv1d_t, window, L, QKV);
            defer conv_acc.deinit(); // [L, QKV] f32
            break :blk try activations.silu(ctx, conv_acc);
        };
        errdefer qkv.deinit();
        if (fp) {
            qkv.eval() catch {};
            gdn_prof.f_conv_ns += @intCast(nowNs() - ft);
            ft = nowNs();
        }

        // New conv state: the last K-1 rows of the RAW (pre-silu) window.
        // With L >= K-1 those are exactly qkv_lin2's last K-1 rows, so the
        // fused path never touches `window`.
        const new_conv_state = blk: {
            const src: Array = if (used_fused) qkv_lin2 else window;
            const row0: i32 = if (used_fused) L - (K - 1) else L;
            const row1: i32 = if (used_fused) L else L + K - 1;
            const tail = try ops.slice(ctx, src, &[_]i32{ row0, 0 }, &[_]i32{ row1, QKV }, &[_]i32{ 1, 1 });
            defer tail.deinit();
            break :blk try ops.copy(ctx, tail);
        };
        errdefer new_conv_state.deinit();

        // 3. Split q/k/v and apply the weight-less per-head rmsNorm (folded
        // scales) — the norm runs along the last dim, so [L, H, D] works
        // directly.
        const q_parts = try ops.slice(ctx, qkv, &[_]i32{ 0, 0 }, &[_]i32{ L, pk }, &[_]i32{ 1, 1 });
        defer q_parts.deinit();
        const k_parts = try ops.slice(ctx, qkv, &[_]i32{ 0, pk }, &[_]i32{ L, 2 * pk }, &[_]i32{ 1, 1 });
        defer k_parts.deinit();
        const v_parts = try ops.slice(ctx, qkv, &[_]i32{ 0, 2 * pk }, &[_]i32{ L, QKV }, &[_]i32{ 1, 1 });
        defer v_parts.deinit();

        const q_resh = try ops.reshape(ctx, q_parts, &[_]i32{ L, HK, D });
        defer q_resh.deinit();
        const k_resh = try ops.reshape(ctx, k_parts, &[_]i32{ L, HK, D });
        defer k_resh.deinit();
        const v = try ops.reshape(ctx, v_parts, &[_]i32{ L, HV, D });
        errdefer v.deinit();

        const q = try fast_mod.rmsNorm(ctx, q_resh, self.q_norm_w, 1e-6); // [L, HK, D]
        errdefer q.deinit();
        const k = try fast_mod.rmsNorm(ctx, k_resh, self.k_norm_w, 1e-6);
        errdefer k.deinit();

        // 4. Gates over [L, HV]: g = exp(-exp(A_log_f32) * softplus(a +
        // dt_bias)), beta = sigmoid(b). beta is widened to f32 afterwards —
        // an exact cast — so the per-step recurrence can use the fused
        // kernel (decode rounds beta to b's dtype; widening back is lossless).
        const a_flat = try ops.reshape(ctx, a, &[_]i32{ L, HV });
        defer a_flat.deinit();
        var e_a_log_owned: ?Array = null;
        defer if (e_a_log_owned) |*e| e.deinit();
        const e_a_log = if (self.e_a_log) |e| e else blk: {
            const a_log_f32 = try ops.astype(ctx, self.A_log, .float32);
            defer a_log_f32.deinit();
            e_a_log_owned = try ops.exp(ctx, a_log_f32);
            break :blk e_a_log_owned.?;
        };
        const b_flat = try ops.reshape(ctx, b, &[_]i32{ L, HV });
        defer b_flat.deinit();

        // Fused path: the 11 elementwise gate dispatches as one compiled
        // kernel (see gateClosureBody). Falls back to the eager chain when
        // compilation is unavailable, so this can never break the model.
        var g: Array = undefined;
        var beta: Array = undefined;
        var gates_fused = false;
        if (gateClosure(self.ctx.allocator)) |cl| fused: {
            const ins = [_]Array{ a_flat, self.dt_bias, e_a_log, b_flat };
            const outs = cl.apply(&ins, self.ctx.allocator) catch break :fused;
            defer self.ctx.allocator.free(outs);
            if (outs.len != 2) {
                for (outs) |o| o.deinit();
                break :fused;
            }
            g = outs[0];
            beta = outs[1];
            gates_fused = true;
        }
        if (!gates_fused) {
            const a_dt = try ops.add(ctx, a_flat, self.dt_bias);
            defer a_dt.deinit();
            const sp = try softplusStep(ctx, a_dt);
            defer sp.deinit();
            const g_pre = try ops.multiply(ctx, e_a_log, sp);
            defer g_pre.deinit();
            const g_neg = try ops.negative(ctx, g_pre);
            defer g_neg.deinit();
            g = try ops.exp(ctx, g_neg); // [L, HV] f32
            const beta_raw = try ops.sigmoid(ctx, b_flat); // [L, HV]
            defer beta_raw.deinit();
            beta = ops.astype(ctx, beta_raw, .float32) catch |e| {
                g.deinit();
                return e;
            };
        }
        errdefer g.deinit();
        errdefer beta.deinit();

        if (fp) {
            g.eval() catch {};
            beta.eval() catch {};
            gdn_prof.f_gate_ns += @intCast(nowNs() - ft);
            ft = nowNs();
        }

        // Broadcast q/k to value heads via the precomputed gather indices.
        const q_full = try shape_mod.takeAxis(ctx, q, self.qk_idx, 1); // [L, HV, D]
        errdefer q_full.deinit();
        const k_full = try shape_mod.takeAxis(ctx, k, self.qk_idx, 1);
        errdefer k_full.deinit();
        if (fp) {
            q_full.eval() catch {};
            k_full.eval() catch {};
            gdn_prof.f_bcast_ns += @intCast(nowNs() - ft);
        }

        return .{
            .qkv = qkv,
            .q_full = q_full,
            .k_full = k_full,
            .q_heads = q,
            .k_heads = k,
            .v = v,
            .beta = beta,
            .g = g,
            .z = z,
            .window = window,
            .new_conv_state = new_conv_state,
        };
    }

    /// Out gate + norm over y [L, HV, D], then one out_proj qmm →
    /// [1, L, hidden]. Shared prefill tail.
    fn prefillTail(self: *Qwen38GDN, y_f32: Array, z: Array, qkv_dtype: dtype_mod.Dtype) !Array {
        const ctx = self.ctx;
        const HV: i32 = @intCast(self.num_value_heads);
        const D: i32 = @intCast(self.head_dim);
        const L: i32 = @intCast(y_f32.shape()[0]);

        // gated_delta_update returns y.astype(qkv.dtype): round before norm.
        const y = try ops.astype(ctx, y_f32, qkv_dtype);
        defer y.deinit();
        const y_normed = try fast_mod.rmsNorm(ctx, y, self.norm_w, self.norm_eps);
        defer y_normed.deinit();
        const y_normed_f32 = try ops.astype(ctx, y_normed, .float32);
        defer y_normed_f32.deinit();
        const z_resh = try ops.reshape(ctx, z, &[_]i32{ L, HV, D });
        defer z_resh.deinit();
        const z_f32 = try ops.astype(ctx, z_resh, .float32);
        defer z_f32.deinit();
        const z_act = try activations.silu(ctx, z_f32);
        defer z_act.deinit();
        const y_gated_f32 = try ops.multiply(ctx, z_act, y_normed_f32);
        defer y_gated_f32.deinit();
        const y_gated = try ops.astype(ctx, y_gated_f32, qkv_dtype);
        defer y_gated.deinit();
        const y_flat = try ops.reshape(ctx, y_gated, &[_]i32{ 1, L, HV * D });
        defer y_flat.deinit();
        return linearQuantParts(ctx, y_flat, self.out_proj, self.o_scales, self.o_biases);
    }

    /// Prefill over x [1, L, hidden]. Returns [1, L, hidden]. Advances cache.
    ///
    /// Vectorized twin of the per-token decode path: one batched qmm per
    /// projection over the whole sequence, a whole-sequence causal conv
    /// (convAccumSeq), vectorized gates, and a vectorized out-gate + out_proj.
    /// The delta-rule recurrence runs as one Metal kernel over the whole
    /// sequence when the head dim allows (deltaRulePrefill →
    /// deltaRulePrefillFused), falling back to a per-step eager loop
    /// (deltaRulePrefillEager) otherwise. Op-for-op math and dtype
    /// round-trips match gdnTailEager; everything past the conv is f32
    /// because the conv_state concat promotes, exactly as in decode.
    pub fn prefill(self: *Qwen38GDN, allocator: std.mem.Allocator, x: Array, cache: *Qwen38GdnCache, sync: bool) !Array {
        const ctx = self.ctx;
        // `DMLX_GDN_PREFILL_PROF=1` splits this layer's prefill into its three
        // sections with a barrier after each. Prefill has essentially no
        // CPU/GPU overlap (the barriered whole-pass total measures within 2.5%
        // of the real wall, see the plan doc §17.2), so these numbers add up
        // and can be trusted — unlike the decode breakdown, which is heavily
        // barrier-inflated. Accumulates across layers; dump with
        // `gdnPrefillProfReport`.
        const prof = gdnPrefillProfEnabled();
        var t0: i128 = if (prof) nowNs() else 0;

        var front = try self.prefillFront(x, cache);
        defer front.deinit();
        if (prof) {
            // Force EVERY front output. Evaluating only some of them
            // mis-attributes the rest: `z` is produced by its own qmm
            // (gdn_z, ~937 ms over 48 layers at L=2048) and is consumed in
            // the tail, so leaving it lazy charges that qmm to the tail and
            // made the tail look ~2.5x more expensive than it is.
            eval_mod.asyncEval(allocator, &[_]Array{
                front.qkv,  front.q_full, front.k_full, front.v,
                front.beta, front.g,      front.z,
            }) catch {};
            front.z.eval() catch {};
            gdn_prof.front_ns += @intCast(nowNs() - t0);
            t0 = nowNs();
        }

        // Delta-rule recurrence over the whole sequence: omlx's pipelined
        // kernel first on the UNBROADCAST q/k (Hk=16: a third of the staging
        // traffic of the Hv=48 broadcast form; the kernel maps hv -> hv/(Hv/Hk)
        // itself, matching qk_idx), then the serial fused kernel, then eager.
        // Inputs are rounded to bf16 first (mlx_vlm/omlx feed their recurrence
        // bf16 too; the kernel accumulates in f32, state stays f32): halves
        // the q/k/v staging + y writeback traffic. DMLX_GDN_PIPE_BF16=0 keeps
        // the f32 inputs.
        const hk_i: i32 = @intCast(self.num_key_heads);
        const hv_i: i32 = @intCast(self.num_value_heads);
        const d_i: i32 = @intCast(self.head_dim);
        const rec = blk: {
            if (pipeBf16Enabled()) {
                const qb = try ops.astype(ctx, front.q_heads, .bfloat16);
                defer qb.deinit();
                const kb = try ops.astype(ctx, front.k_heads, .bfloat16);
                defer kb.deinit();
                const vb = try ops.astype(ctx, front.v, .bfloat16);
                defer vb.deinit();
                if (try deltaRulePrefillPipelined(ctx, cache.ssm_state, kb, vb, front.beta, front.g, qb, hk_i, hv_i, d_i)) |r| break :blk r;
            }
            if (try deltaRulePrefillPipelined(ctx, cache.ssm_state, front.k_heads, front.v, front.beta, front.g, front.q_heads, hk_i, hv_i, d_i)) |r| break :blk r;
            break :blk try deltaRulePrefill(ctx, allocator, cache.ssm_state, front.k_full, front.v, front.beta, front.g, front.q_full, hv_i, d_i);
        };
        errdefer rec.s_new.deinit();
        defer rec.y.deinit(); // [L, HV, D] f32
        if (prof) {
            rec.y.eval() catch {};
            gdn_prof.rec_ns += @intCast(nowNs() - t0);
            t0 = nowNs();
        }

        const out = try self.prefillTail(rec.y, front.z, front.qkv.dtype());
        errdefer out.deinit();
        if (prof) {
            out.eval() catch {};
            gdn_prof.tail_ns += @intCast(nowNs() - t0);
            gdn_prof.layers += 1;
        }

        // Single cache swap at the end. `sync` (serving loop: every few
        // layers) bounds peak memory: without any barrier the whole
        // forward's prefill graphs stay live until the final logits sync
        // (~+12 GB at L=4096); syncing every layer costs ~5% wall in lost
        // CPU/GPU overlap.
        if (sync) try out.eval();
        try eval_mod.asyncEval(allocator, &[_]Array{ front.new_conv_state, rec.s_new });
        cache.conv_state.deinit();
        cache.conv_state = front.new_conv_state;
        cache.ssm_state.deinit();
        cache.ssm_state = rec.s_new;
        front.conv_moved = true; // new_conv_state now owned by the cache
        return out;
    }

    /// Per-position states from a verify pass (L ≤ ~5): the conv/ssm state
    /// AFTER step t, for every t. Fused path keeps them in one buffer
    /// (`s_all` [L, HV, D, D]); the eager fallback keeps per-step arrays
    /// (`ssm_states` [L]). Ownership: lists and arrays are owned by the
    /// result; commitVerifyStates extracts one position.
    pub const PrefillVerifyResult = struct {
        out: Array, // [1, L, hidden]
        conv_states: []Array, // [L] of [K-1, QKV]
        ssm_states: []Array, // [L] of [HV, D, D] f32 (eager path; empty when fused)
        s_all: ?Array = null, // [L, HV, D, D] f32 (fused path)
        // Lazy conv-state path (default): instead of materializing L row-slice
        // copies per layer, keep the whole-sequence `window` and slice the ONE
        // committed position out of it. The window is never evaluated on the
        // fused conv path, so the concat that builds it is free too; what this
        // removes is 2*(L-1) small dispatches per GDN layer per round (288 ops
        // at L=3 over 48 layers) — the same "defer to commit" shape the s_all
        // path already uses. Set DMLX_NO_VERIFY_LAZY_CONV=1 for the old path.
        conv_window: ?Array = null, // [L+K-1, QKV] f32
        conv_k: i32 = 0, // conv kernel width K

        // --- snapshot + replay-on-rejection (plan doc §42) ----------------
        // `s_all` holds the post-step state of EVERY position so commit can
        // adopt any of them, which costs L x 3.1 MB x 48 layers of writes
        // unconditionally — 1.19 GB per round at the depth-7 ceiling (~5 ms).
        // omlx instead snapshots the pre-forward state and, only on a
        // rejection, replays the accepted prefix
        // (`qwen35_model.mtp_partial_rollback`: "One small gated-delta kernel
        // per linear layer, paid only on rejections").
        //
        // `prefillVerify` never mutates `cache.ssm_state`, so the snapshot is
        // free — we only need to retain the recurrence INPUTS to replay from
        // it. Those are small: [L, HV, D] f32 x3 + [L, HV] f32 x2, about
        // 590 KB per layer at L=8.
        replay: ?struct {
            s_in: Array, // [HV, D, D] f32 — pre-forward snapshot (borrowed)
            k_full: Array, // [L, HV, D] f32
            v: Array,
            beta: Array, // [L, HV] f32
            g: Array,
            q_full: Array,
            hv: i32,
            d: i32,
        } = null,

        /// Conv state after step `t`, whichever retention path ran. Owned by
        /// the caller.
        pub fn convStateAt(self: *const PrefillVerifyResult, ctx: EagerContext, t: usize) !Array {
            if (self.conv_window) |w| {
                const a: i32 = @intCast(t);
                const qkv: i32 = w.shape()[1];
                const rows = try ops.slice(ctx, w, &[_]i32{ a + 1, 0 }, &[_]i32{ a + self.conv_k, qkv }, &[_]i32{ 1, 1 });
                defer rows.deinit();
                return ops.copy(ctx, rows);
            }
            return ops.copy(ctx, self.conv_states[t]);
        }

        pub fn deinit(self: *PrefillVerifyResult, allocator: std.mem.Allocator) void {
            self.out.deinit();
            if (self.replay) |r| {
                // `s_in` is borrowed from the cache; the rest is owned here.
                r.k_full.deinit();
                r.v.deinit();
                r.beta.deinit();
                r.g.deinit();
                r.q_full.deinit();
                self.replay = null;
            }
            if (self.conv_window) |w| w.deinit();
            self.conv_window = null;
            for (self.conv_states) |s| s.deinit();
            for (self.ssm_states) |s| s.deinit();
            if (self.conv_states.len > 0) allocator.free(self.conv_states);
            if (self.ssm_states.len > 0) allocator.free(self.ssm_states);
            if (self.s_all) |s| s.deinit();
            self.conv_states = &.{};
            self.ssm_states = &.{};
            self.s_all = null;
        }
    };

    /// Verify pass: identical math to prefill but does NOT touch the cache.
    /// The recurrence runs as ONE Metal kernel when eligible
    /// (deltaRulePrefillFusedRetain, per-position states in s_all), falling
    /// back to the per-step eager loop with per-step retained arrays
    /// otherwise. The conv state after step t is window rows [t+1, t+K) of
    /// the precomputed whole-sequence window.
    pub fn prefillVerify(self: *Qwen38GDN, allocator: std.mem.Allocator, x: Array, cache: *Qwen38GdnCache) !PrefillVerifyResult {
        const ctx = self.ctx;
        // `DMLX_VERIFY_PROF=1` barriers each section. Unlike decode, the verify
        // forward has little CPU/GPU overlap, so these add up (same reasoning
        // as §17.2 for prefill). Dump with `gdnVerifyProfReport`.
        const vp = gdnVerifyProfEnabled();
        var vt: i128 = if (vp) nowNs() else 0;
        const hp = std.c.getenv("DMLX_MTP_DEBUG") != null;
        var ht: i128 = if (hp) nowNs() else 0;

        var front = try self.prefillFront(x, cache);
        defer front.deinit();
        if (hp) {
            gdn_prof.v_front_host_ns += @intCast(nowNs() - ht);
            ht = nowNs();
        }
        if (vp) {
            // Force EVERY front output: `z` has its own qmm and is consumed in
            // the tail, so leaving it lazy charges that qmm to the tail (the
            // mis-attribution that made the prefill tail look 2.5x too
            // expensive in §19.2).
            eval_mod.asyncEval(allocator, &[_]Array{
                front.qkv,  front.q_full, front.k_full, front.v,
                front.beta, front.g,      front.z,
            }) catch {};
            front.z.eval() catch {};
            gdn_prof.v_front_ns += @intCast(nowNs() - vt);
            vt = nowNs();
        }

        const L: i32 = @intCast(x.shape()[1]);
        const HV: i32 = @intCast(self.num_value_heads);
        const D: i32 = @intCast(self.head_dim);
        const K: i32 = @intCast(self.conv_kernel);
        const QKV: i32 = @intCast(front.qkv.shape()[1]);

        // Recurrence: fused single-kernel with per-position states, else the
        // eager per-step retain loop (toy/odd head dims).
        var y: Array = undefined;
        var s_all: ?Array = null;
        var ssm_states: []Array = &.{};
        errdefer if (s_all) |s| s.deinit();
        var replay_inputs: @FieldType(PrefillVerifyResult, "replay") = null;
        if (verifySnapshot() and !verifyNoRecur()) {
            // Snapshot path: write only the FINAL state and retain the
            // recurrence inputs; commit replays the accepted prefix when a
            // draft was rejected. Removes (L-1) x 3.1 MB x 48 layers of
            // unconditional s_all writes — 1.04 GB per round at L=8.
            if (try deltaRulePrefillFused(ctx, cache.ssm_state, front.k_full, front.v, front.beta, front.g, front.q_full, HV, D)) |fr| {
                y = fr.y;
                defer fr.s_new.deinit();
                s_all = try ops.reshape(ctx, fr.s_new, &[_]i32{ 1, HV, D, D });
                replay_inputs = .{
                    .s_in = cache.ssm_state, // borrowed: the cache still owns it
                    .k_full = front.k_full,
                    .v = front.v,
                    .beta = front.beta,
                    .g = front.g,
                    .q_full = front.q_full,
                    .hv = HV,
                    .d = D,
                };
                front.rec_moved = true;
            }
        }
        if (replay_inputs != null) {
            // handled above
        } else if (verifyNoRecur()) {
            // Diagnostic: zero-cost stand-in for the WHOLE recurrence — no
            // kernel launch, no state read, no s_all write, no ALU. `v` has
            // the readout's exact shape/dtype ([L, HV, D] f32) and `s_in`
            // reshaped to [1, HV, D, D] is a free view (commit clamps the
            // row index). The round-cost delta is the recurrence's total
            // price, with none of the barrier distortion of §21.3.
            y = try ops.copy(ctx, front.v);
            s_all = try ops.reshape(ctx, cache.ssm_state, &[_]i32{ 1, HV, D, D });
        } else if (verifyNoSall()) {
            // Diagnostic: final state only, shaped [1, HV, D, D] so commit's
            // s_all path still works (it clamps the row index).
            const fr = (try deltaRulePrefillFused(ctx, cache.ssm_state, front.k_full, front.v, front.beta, front.g, front.q_full, HV, D)) orelse
                return error.VerifyNoSallUnavailable;
            defer fr.s_new.deinit();
            y = fr.y;
            s_all = try ops.reshape(ctx, fr.s_new, &[_]i32{ 1, HV, D, D });
        } else if (try deltaRulePrefillFusedRetain(ctx, cache.ssm_state, front.k_full, front.v, front.beta, front.g, front.q_full, HV, D)) |fr| {
            y = fr.y;
            s_all = fr.s_all;
        } else {
            const rec = try deltaRulePrefillEagerRetain(ctx, allocator, cache.ssm_state, front.k_full, front.v, front.beta, front.g, front.q_full, HV, D);
            y = rec.y;
            ssm_states = rec.ssm_states;
        }
        // NOTE: no errdefer for y — the plain defer below already covers the
        // error path; registering both double-freed y (segfault in error
        // unwinding, first seen when a verify-kernel compile failure
        // propagated through here).
        errdefer {
            for (ssm_states) |s| s.deinit();
            if (ssm_states.len > 0) allocator.free(ssm_states);
        }
        defer y.deinit();
        if (vp) {
            y.eval() catch {};
            if (s_all) |sa| sa.eval() catch {};
            gdn_prof.v_rec_ns += @intCast(nowNs() - vt);
            vt = nowNs();
        }
        if (hp) {
            gdn_prof.v_rec_host_ns += @intCast(nowNs() - ht);
            ht = nowNs();
        }

        // Per-step conv states: rows [t+1, t+K) of the window. By default we
        // hand the window itself to the caller and let commitVerifyStates cut
        // the single accepted position out of it.
        var conv_window: ?Array = null;
        var conv_states: []Array = &.{};
        var made: usize = 0;
        errdefer if (conv_window) |w| w.deinit();
        errdefer if (conv_states.len > 0) {
            for (conv_states[0..made]) |s| s.deinit();
            allocator.free(conv_states);
        };
        if (verifyLazyConv()) {
            conv_window = front.window;
            front.window_moved = true;
        } else {
            conv_states = try allocator.alloc(Array, @intCast(L));
            var t: i32 = 0;
            while (t < L) : (t += 1) {
                const rows = try ops.slice(ctx, front.window, &[_]i32{ t + 1, 0 }, &[_]i32{ t + K, QKV }, &[_]i32{ 1, 1 });
                defer rows.deinit();
                conv_states[@intCast(t)] = try ops.copy(ctx, rows);
                made += 1;
            }
        }
        if (vp and conv_states.len > 0) {
            eval_mod.asyncEval(allocator, conv_states) catch {};
            if (made > 0) conv_states[made - 1].eval() catch {};
            gdn_prof.v_states_ns += @intCast(nowNs() - vt);
            vt = nowNs();
        }

        const out = try self.prefillTail(y, front.z, front.qkv.dtype());
        errdefer out.deinit();
        if (hp) {
            gdn_prof.v_tail_host_ns += @intCast(nowNs() - ht);
            ht = nowNs();
        }
        if (vp) {
            out.eval() catch {};
            gdn_prof.v_tail_ns += @intCast(nowNs() - vt);
            gdn_prof.v_layers += 1;
        }
        // Materialize everything the caller may commit: all retained states.
        // `conv_window` is deliberately NOT evaluated — commit slices it.
        //
        // The per-layer asyncEval submits DOUBLE as CPU/GPU overlap: while the
        // host builds layer i+1's graph, the GPU runs layer i. Removing them
        // (measured 2026-11-03) dropped total verify host build 70 -> 2 ms but
        // the serial final eval grew 18 -> 85+ ms and decode fell ~10%
        // (34.9 -> 30.7 tok/s thinking code) — the GPU needs the whole ~85 ms
        // regardless, so overlapping it with the host build is strictly
        // better. DMLX_VERIFY_LAYER_SUBMIT=0 disables for experiments.
        if (std.c.getenv("DMLX_VERIFY_LAYER_SUBMIT") == null or
            !std.mem.eql(u8, std.mem.span(std.c.getenv("DMLX_VERIFY_LAYER_SUBMIT").?), "0"))
        {
            if (conv_states.len > 0) try eval_mod.asyncEval(allocator, conv_states);
            if (s_all) |s| {
                try eval_mod.asyncEval(allocator, &[_]Array{s});
            } else if (ssm_states.len > 0) {
                try eval_mod.asyncEval(allocator, ssm_states);
            }
        }
        if (hp) {
            gdn_prof.v_submit_host_ns += @intCast(nowNs() - ht);
            gdn_prof.v_layers_host += 1;
        }
        return .{
            .out = out,
            .conv_states = conv_states,
            .ssm_states = ssm_states,
            .s_all = s_all,
            .conv_window = conv_window,
            .conv_k = K,
            .replay = replay_inputs,
        };
    }

    /// Commit the post-step-`accepted` states (accepted ∈ 0..L-1) into the
    /// cache, freeing every other retained state. Afterwards `res` holds
    /// only `out` (its lists are emptied).
    pub fn commitVerifyStates(ctx: EagerContext, cache: *Qwen38GdnCache, res: *PrefillVerifyResult, allocator: std.mem.Allocator, accepted: usize) void {
        if (res.conv_window) |w| {
            // Lazy path: cut rows [a+1, a+K) out of the retained window.
            if (res.convStateAt(ctx, accepted)) |copied| {
                cache.conv_state.deinit();
                cache.conv_state = copied;
            } else |_| {
                std.log.err("GDN commitVerifyStates: conv window slice failed; cache state unchanged", .{});
            }
            w.deinit();
            res.conv_window = null;
        } else {
            cache.conv_state.deinit();
            cache.conv_state = res.conv_states[accepted];
            for (res.conv_states, 0..) |s, i| {
                if (i != accepted) s.deinit();
            }
            allocator.free(res.conv_states);
            res.conv_states = &.{};
        }

        if (res.replay) |r| {
            // Snapshot path. `s_all` holds only the state after the LAST
            // step, which is correct exactly when every draft was accepted.
            // Otherwise replay steps 0..accepted from the pre-forward
            // snapshot — one extra recurrence kernel per layer, paid only on
            // a rejection (omlx's `mtp_partial_rollback` does the same).
            const steps: i32 = @intCast(accepted + 1);
            const total: i32 = r.k_full.shape()[0];
            var adopted = false;
            if (steps < total) {
                const k3 = ops.slice(ctx, r.k_full, &[_]i32{ 0, 0, 0 }, &[_]i32{ steps, r.hv, r.d }, &[_]i32{ 1, 1, 1 }) catch null;
                defer if (k3) |x| x.deinit();
                const v3 = if (k3 != null) ops.slice(ctx, r.v, &[_]i32{ 0, 0, 0 }, &[_]i32{ steps, r.hv, r.d }, &[_]i32{ 1, 1, 1 }) catch null else null;
                defer if (v3) |x| x.deinit();
                const q3 = if (v3 != null) ops.slice(ctx, r.q_full, &[_]i32{ 0, 0, 0 }, &[_]i32{ steps, r.hv, r.d }, &[_]i32{ 1, 1, 1 }) catch null else null;
                defer if (q3) |x| x.deinit();
                const b2 = if (q3 != null) ops.slice(ctx, r.beta, &[_]i32{ 0, 0 }, &[_]i32{ steps, r.hv }, &[_]i32{ 1, 1 }) catch null else null;
                defer if (b2) |x| x.deinit();
                const g2 = if (b2 != null) ops.slice(ctx, r.g, &[_]i32{ 0, 0 }, &[_]i32{ steps, r.hv }, &[_]i32{ 1, 1 }) catch null else null;
                defer if (g2) |x| x.deinit();
                if (g2 != null) {
                    if (deltaRulePrefillFused(ctx, r.s_in, k3.?, v3.?, b2.?, g2.?, q3.?, r.hv, r.d) catch null) |rep| {
                        rep.y.deinit();
                        cache.ssm_state.deinit();
                        cache.ssm_state = rep.s_new;
                        adopted = true;
                    }
                }
            } else if (res.s_all) |sa| {
                const flat = ops.reshape(ctx, sa, &[_]i32{ r.hv, r.d, r.d }) catch null;
                if (flat) |f| {
                    defer f.deinit();
                    if (ops.copy(ctx, f)) |copied| {
                        cache.ssm_state.deinit();
                        cache.ssm_state = copied;
                        adopted = true;
                    } else |_| {}
                }
            }
            if (!adopted) std.log.err("GDN commitVerifyStates: snapshot commit failed; cache state unchanged", .{});
            // The replay inputs were MOVED out of `front` (rec_moved) and are
            // owned by `res`; deinit frees them only while res.replay != null.
            // Freeing here was missing -> ~0.4 MB per GDN layer per committed
            // round leaked (MemProbe: +0.68 GB/request, metal::malloc limit
            // crash after ~20 sustained MTP requests, 2026-10-30).
            r.k_full.deinit();
            r.v.deinit();
            r.beta.deinit();
            r.g.deinit();
            r.q_full.deinit();
            res.replay = null;
            if (res.s_all) |sa| {
                sa.deinit();
                res.s_all = null;
            }
            return;
        }

        if (res.s_all) |sa| {
            // Fused path: copy row `accepted` of [L, HV, D, D] into the cache.
            const hv: i32 = sa.shape()[1];
            const d: i32 = sa.shape()[2];
            const a: i32 = @min(@as(i32, @intCast(accepted)), sa.shape()[0] - 1);
            const row = ops.slice(ctx, sa, &[_]i32{ a, 0, 0, 0 }, &[_]i32{ a + 1, hv, d, d }, &[_]i32{ 1, 1, 1, 1 }) catch {
                std.log.err("GDN commitVerifyStates: s_all slice failed; cache state unchanged", .{});
                return;
            };
            defer row.deinit();
            const flat = ops.reshape(ctx, row, &[_]i32{ hv, d, d }) catch {
                std.log.err("GDN commitVerifyStates: s_all reshape failed; cache state unchanged", .{});
                return;
            };
            defer flat.deinit();
            const copied = ops.copy(ctx, flat) catch {
                std.log.err("GDN commitVerifyStates: s_all copy failed; cache state unchanged", .{});
                return;
            };
            cache.ssm_state.deinit();
            cache.ssm_state = copied;
            sa.deinit();
            res.s_all = null;
            return;
        }

        cache.ssm_state.deinit();
        cache.ssm_state = res.ssm_states[accepted];
        for (res.ssm_states, 0..) |s, i| {
            if (i != accepted) s.deinit();
        }
        allocator.free(res.ssm_states);
        res.ssm_states = &.{};
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

test "Qwen38GDN fused verify recurrence kernel matches eager per-step retain" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Real model head dims; verify blocks are tiny (N+1 ≤ 4), plus one
    // larger L to cross the fused kernel's loop.
    const hv: i32 = 48;
    const d: i32 = 128;
    for ([_]i32{ 1, 3, 4, 17 }) |L| {
        const n_lhvd: usize = @intCast(L * hv * d);
        const n_lhv: usize = @intCast(L * hv);
        const n_state: usize = @intCast(hv * d * d);

        const qkv_buf = try allocator.alloc(f32, 3 * n_lhvd);
        defer allocator.free(qkv_buf);
        testFill(qkv_buf);
        const gb_buf = try allocator.alloc(f32, 2 * n_lhv);
        defer allocator.free(gb_buf);
        for (0..n_lhv) |i| {
            gb_buf[i] = 0.5 + 0.04 * @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 7 + 3, 11)));
            gb_buf[n_lhv + i] = 0.3 + 0.05 * @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 5 + 1, 9)));
        }
        const state_buf = try allocator.alloc(f32, n_state);
        defer allocator.free(state_buf);
        testFill(state_buf);

        const q = try Array.fromData(allocator, f32, qkv_buf[0..n_lhvd], &[_]i32{ L, hv, d });
        defer q.deinit();
        const k = try Array.fromData(allocator, f32, qkv_buf[n_lhvd .. 2 * n_lhvd], &[_]i32{ L, hv, d });
        defer k.deinit();
        const v = try Array.fromData(allocator, f32, qkv_buf[2 * n_lhvd ..], &[_]i32{ L, hv, d });
        defer v.deinit();
        const g = try Array.fromData(allocator, f32, gb_buf[0..n_lhv], &[_]i32{ L, hv });
        defer g.deinit();
        const beta = try Array.fromData(allocator, f32, gb_buf[n_lhv..], &[_]i32{ L, hv });
        defer beta.deinit();
        const state = try Array.fromData(allocator, f32, state_buf, &[_]i32{ hv, d, d });
        defer state.deinit();

        const fused = (try deltaRulePrefillFusedRetain(ctx, state, k, v, beta, g, q, hv, d)) orelse
            return error.FusedKernelUnavailable;
        defer fused.y.deinit();
        defer fused.s_all.deinit();
        try fused.y.eval();
        try fused.s_all.eval();

        const eager = try deltaRulePrefillEagerRetain(ctx, allocator, state, k, v, beta, g, q, hv, d);
        defer {
            eager.y.deinit();
            for (eager.ssm_states) |s| s.deinit();
            allocator.free(eager.ssm_states);
        }
        try eager.y.eval();

        const got_y = try fused.y.dataSlice(f32);
        const exp_y = try eager.y.dataSlice(f32);
        try std.testing.expectEqual(exp_y.len, got_y.len);
        for (exp_y, got_y) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);

        // Row t of s_all equals the eager retained state after step t.
        const got_all = try fused.s_all.dataSlice(f32);
        var t: usize = 0;
        while (t < @as(usize, @intCast(L))) : (t += 1) {
            try eager.ssm_states[t].eval();
            const exp_s = try eager.ssm_states[t].dataSlice(f32);
            const got_s = got_all[t * n_state .. (t + 1) * n_state];
            try std.testing.expectEqual(exp_s.len, got_s.len);
            for (exp_s, got_s) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
        }
    }
}

test "Qwen38GDN fused prefill recurrence kernel matches eager per-step loop" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Real model head dims; L values cross the eager loop's sync cadence (16).
    const hv: i32 = 48;
    const d: i32 = 128;
    for ([_]i32{ 37, 64, 128 }) |L| {
        const n_lhvd: usize = @intCast(L * hv * d);
        const n_lhv: usize = @intCast(L * hv);
        const n_state: usize = @intCast(hv * d * d);

        const qkv_buf = try allocator.alloc(f32, 3 * n_lhvd);
        defer allocator.free(qkv_buf);
        testFill(qkv_buf);
        const gb_buf = try allocator.alloc(f32, 2 * n_lhv);
        defer allocator.free(gb_buf);
        // Gates in (0, 1) like the real softplus-decay / sigmoid-beta path.
        for (0..n_lhv) |i| {
            gb_buf[i] = 0.5 + 0.04 * @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 7 + 3, 11)));
            gb_buf[n_lhv + i] = 0.3 + 0.05 * @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 5 + 1, 9)));
        }
        const state_buf = try allocator.alloc(f32, n_state);
        defer allocator.free(state_buf);
        testFill(state_buf);

        const q = try Array.fromData(allocator, f32, qkv_buf[0..n_lhvd], &[_]i32{ L, hv, d });
        defer q.deinit();
        const k = try Array.fromData(allocator, f32, qkv_buf[n_lhvd .. 2 * n_lhvd], &[_]i32{ L, hv, d });
        defer k.deinit();
        const v = try Array.fromData(allocator, f32, qkv_buf[2 * n_lhvd ..], &[_]i32{ L, hv, d });
        defer v.deinit();
        const g = try Array.fromData(allocator, f32, gb_buf[0..n_lhv], &[_]i32{ L, hv });
        defer g.deinit();
        const beta = try Array.fromData(allocator, f32, gb_buf[n_lhv..], &[_]i32{ L, hv });
        defer beta.deinit();
        const state = try Array.fromData(allocator, f32, state_buf, &[_]i32{ hv, d, d });
        defer state.deinit();

        const fused = (try deltaRulePrefillFused(ctx, state, k, v, beta, g, q, hv, d)) orelse
            return error.FusedKernelUnavailable;
        defer fused.s_new.deinit();
        defer fused.y.deinit();
        try fused.s_new.eval();
        try fused.y.eval();

        const eager = try deltaRulePrefillEager(ctx, allocator, state, k, v, beta, g, q, hv, d);
        defer eager.s_new.deinit();
        defer eager.y.deinit();
        try eager.s_new.eval();
        try eager.y.eval();

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

test "Qwen38GDN.prefillVerify: matches prefill + per-position states + commit" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const dims = TestDims{};
    const hidden: i32 = @intCast(dims.hidden);
    const L: i32 = 5;
    const n: usize = @intCast(L * hidden);

    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 5, 11) - 5)) * 0.1;
    }
    const x = try Array.fromData(allocator, f32, buf, &[_]i32{ 1, L, hidden });
    defer x.deinit();

    var g = try makeTestModule(allocator, ctx, dims);
    defer g.deinit();
    var cache = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache.deinit();

    var res = try g.prefillVerify(allocator, x, &cache);
    defer res.deinit(allocator);
    try res.out.eval();
    const ssm_max0 = try reduce_mod.max(ctx, try ops.abs(ctx, cache.ssm_state), false);
    defer ssm_max0.deinit();
    try ssm_max0.eval();
    try std.testing.expectEqual(@as(f32, 0.0), try ssm_max0.item(f32));

    // Output matches the plain prefill path.
    var g_ref = try makeTestModule(allocator, ctx, dims);
    defer g_ref.deinit();
    var cache_ref = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache_ref.deinit();
    const out_ref = try g_ref.prefill(allocator, x, &cache_ref, true);
    defer out_ref.deinit();
    try out_ref.eval();
    {
        const got = try res.out.dataSlice(f32);
        const exp = try out_ref.dataSlice(f32);
        try std.testing.expectEqual(exp.len, got.len);
        for (exp, got) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
    }

    // Each retained position-a state equals the cache state after a fresh
    // prefix prefill of x[0..a+1].
    var a: i32 = 0;
    while (a < L) : (a += 1) {
        var g_a = try makeTestModule(allocator, ctx, dims);
        defer g_a.deinit();
        var cache_a = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
        defer cache_a.deinit();
        const x_pre = try ops.slice(ctx, x, &[_]i32{ 0, 0, 0 }, &[_]i32{ 1, a + 1, hidden }, &[_]i32{ 1, 1, 1 });
        defer x_pre.deinit();
        const out_a = try g_a.prefill(allocator, x_pre, &cache_a, true);
        defer out_a.deinit();
        try out_a.eval();
        try cache_a.ssm_state.eval();
        try cache_a.conv_state.eval();
        try res.ssm_states[@intCast(a)].eval();
        const conv_a = try res.convStateAt(ctx, @intCast(a));
        defer conv_a.deinit();
        try conv_a.eval();
        const got_s = try res.ssm_states[@intCast(a)].dataSlice(f32);
        const exp_s = try cache_a.ssm_state.dataSlice(f32);
        for (exp_s, got_s) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
        const got_c = try conv_a.dataSlice(f32);
        const exp_c = try cache_a.conv_state.dataSlice(f32);
        for (exp_c, got_c) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
    }

    // Commit path: verify, commit position 2 (accept 3 tokens), then decode
    // the next token — must match a fresh prefix-prefill(3) + forward.
    var g2 = try makeTestModule(allocator, ctx, dims);
    defer g2.deinit();
    var cache2 = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache2.deinit();
    var res2 = try g2.prefillVerify(allocator, x, &cache2);
    defer res2.deinit(allocator);
    Qwen38GDN.commitVerifyStates(ctx, &cache2, &res2, allocator, 2);

    const x_next = try ops.slice(ctx, x, &[_]i32{ 0, 3, 0 }, &[_]i32{ 1, 4, hidden }, &[_]i32{ 1, 1, 1 });
    defer x_next.deinit();
    const x_flat = try ops.reshape(ctx, x_next, &[_]i32{ 1, hidden });
    defer x_flat.deinit();

    var g3 = try makeTestModule(allocator, ctx, dims);
    defer g3.deinit();
    var cache3 = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache3.deinit();
    const x_pre3 = try ops.slice(ctx, x, &[_]i32{ 0, 0, 0 }, &[_]i32{ 1, 3, hidden }, &[_]i32{ 1, 1, 1 });
    defer x_pre3.deinit();
    const out_pre3 = try g3.prefill(allocator, x_pre3, &cache3, true);
    defer out_pre3.deinit();
    try out_pre3.eval();

    const out_commit = try g2.forward(x_flat, &cache2);
    defer out_commit.deinit();
    const out_ref3 = try g3.forward(x_flat, &cache3);
    defer out_ref3.deinit();
    try out_commit.eval();
    try out_ref3.eval();
    const got2 = try out_commit.dataSlice(f32);
    const exp2 = try out_ref3.dataSlice(f32);
    for (exp2, got2) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
    // Commit emptied the lists; res2.deinit (defer) frees only `out`.
    try std.testing.expectEqual(@as(usize, 0), res2.conv_states.len);
    try std.testing.expectEqual(@as(usize, 0), res2.ssm_states.len);
}

test "Qwen38GDN.prefillVerify fused: s_all commit matches prefix prefill" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // head_dim 32 → the fused retain kernel is eligible (D % 32 == 0).
    const dims = TestDims{ .hidden = 8, .key_heads = 1, .value_heads = 2, .head_dim = 32, .conv_kernel = 4 };
    const hidden: i32 = @intCast(dims.hidden);
    const L: i32 = 5;

    const buf = try allocator.alloc(f32, @intCast(L * hidden));
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 5, 11) - 5)) * 0.1;
    }
    const x = try Array.fromData(allocator, f32, buf, &[_]i32{ 1, L, hidden });
    defer x.deinit();

    var g = try makeTestModule(allocator, ctx, dims);
    defer g.deinit();
    var cache = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache.deinit();

    var res = try g.prefillVerify(allocator, x, &cache);
    defer res.deinit(allocator);
    // Fused path taken: per-position states live in one [L, HV, D, D] buffer.
    try std.testing.expect(res.s_all != null);
    try std.testing.expectEqual(@as(usize, 0), res.ssm_states.len);
    try res.out.eval();

    // Output matches the plain prefill path.
    var g_ref = try makeTestModule(allocator, ctx, dims);
    defer g_ref.deinit();
    var cache_ref = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache_ref.deinit();
    const out_ref = try g_ref.prefill(allocator, x, &cache_ref, true);
    defer out_ref.deinit();
    try out_ref.eval();
    {
        const got = try res.out.dataSlice(f32);
        const exp = try out_ref.dataSlice(f32);
        try std.testing.expectEqual(exp.len, got.len);
        for (exp, got) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
    }

    // Commit position 2 (accept 3 tokens) then decode token 4 — must match a
    // fresh prefix-prefill(3) + forward.
    Qwen38GDN.commitVerifyStates(ctx, &cache, &res, allocator, 2);
    try std.testing.expect(res.s_all == null);

    const x_next = try ops.slice(ctx, x, &[_]i32{ 0, 3, 0 }, &[_]i32{ 1, 4, hidden }, &[_]i32{ 1, 1, 1 });
    defer x_next.deinit();
    const x_flat = try ops.reshape(ctx, x_next, &[_]i32{ 1, hidden });
    defer x_flat.deinit();

    var g3 = try makeTestModule(allocator, ctx, dims);
    defer g3.deinit();
    var cache3 = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
    defer cache3.deinit();
    const x_pre3 = try ops.slice(ctx, x, &[_]i32{ 0, 0, 0 }, &[_]i32{ 1, 3, hidden }, &[_]i32{ 1, 1, 1 });
    defer x_pre3.deinit();
    const out_pre3 = try g3.prefill(allocator, x_pre3, &cache3, true);
    defer out_pre3.deinit();
    try out_pre3.eval();

    const out_commit = try g.forward(x_flat, &cache);
    defer out_commit.deinit();
    const out_ref3 = try g3.forward(x_flat, &cache3);
    defer out_ref3.deinit();
    try out_commit.eval();
    try out_ref3.eval();
    const got2 = try out_commit.dataSlice(f32);
    const exp2 = try out_ref3.dataSlice(f32);
    for (exp2, got2) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
}

test "Qwen38GDN.prefill: per-token loop equivalence" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const dims = TestDims{};
    const hidden: i32 = @intCast(dims.hidden);

    for ([_]i32{ 3, 64 }) |L| {
        var g = try makeTestModule(allocator, ctx, dims);
        defer g.deinit();
        var cache = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
        defer cache.deinit();

        const n: usize = @intCast(L * hidden);
        const buf = try allocator.alloc(f32, n);
        defer allocator.free(buf);
        for (buf, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 5, 11) - 5)) * 0.1;
        }
        const x = try Array.fromData(allocator, f32, buf, &[_]i32{ 1, L, hidden });
        defer x.deinit();

        const out = try g.prefill(allocator, x, &cache, true);
        defer out.deinit();
        const oshape = out.shape();
        try std.testing.expectEqual(@as(i32, 1), oshape[0]);
        try std.testing.expectEqual(L, oshape[1]);
        try std.testing.expectEqual(hidden, oshape[2]);
        try out.eval();

        var g_ref = try makeTestModule(allocator, ctx, dims);
        defer g_ref.deinit();
        var cache_ref = try Qwen38GdnCache.init(allocator, dims.conv_kernel, (2 * dims.key_heads + dims.value_heads) * dims.head_dim, dims.value_heads, dims.head_dim);
        defer cache_ref.deinit();

        const ref_buf = try allocator.alloc(f32, n);
        defer allocator.free(ref_buf);
        var t: i32 = 0;
        while (t < L) : (t += 1) {
            const x_t = try ops.slice(ctx, x, &[_]i32{ 0, t, 0 }, &[_]i32{ 1, t + 1, hidden }, &[_]i32{ 1, 1, 1 });
            defer x_t.deinit();
            const x_flat = try ops.reshape(ctx, x_t, &[_]i32{ 1, hidden });
            defer x_flat.deinit();
            const out_t = try g_ref.forward(x_flat, &cache_ref);
            defer out_t.deinit();
            try out_t.eval();
            const row = try out_t.dataSlice(f32);
            @memcpy(ref_buf[@intCast(t * hidden)..@intCast((t + 1) * hidden)], row);
        }
        const got = try out.dataSlice(f32);
        for (ref_buf, got) |e, o| {
            try std.testing.expectApproxEqAbs(e, o, 1e-5);
        }
        // Final states must match too (they seed the decode that follows).
        try cache.conv_state.eval();
        try cache_ref.conv_state.eval();
        const got_conv = try cache.conv_state.dataSlice(f32);
        const exp_conv = try cache_ref.conv_state.dataSlice(f32);
        try std.testing.expectEqual(exp_conv.len, got_conv.len);
        for (exp_conv, got_conv) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
        try cache.ssm_state.eval();
        try cache_ref.ssm_state.eval();
        const got_ssm = try cache.ssm_state.dataSlice(f32);
        const exp_ssm = try cache_ref.ssm_state.dataSlice(f32);
        try std.testing.expectEqual(exp_ssm.len, got_ssm.len);
        for (exp_ssm, got_ssm) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-5);
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

test "convAccumSeqFused matches the eager taps + silu" {
    // Gate for the fused conv: the delta-rule recurrence downstream is
    // sensitive to conv rounding, and HANDOFF records an FMA rewrite that
    // passed smoke yet cost 10.6% and broke Paris. So compare the fused
    // kernel against the eager form it replaces, element by element.
    const allocator = std.testing.allocator;
    c.initErrorHandler();
    const ctx = EagerContext.init(allocator);

    const K: i32 = 4;
    const QKV: i32 = 256;
    for ([_]i32{ 3, 8, 37, 64 }) |L| {
        const x = try makeTestWeight(allocator, &[_]i32{ L, QKV });
        defer x.deinit();
        const cs = try makeTestWeight(allocator, &[_]i32{ K - 1, QKV });
        defer cs.deinit();
        const w = try makeTestWeight(allocator, &[_]i32{ K, QKV });
        defer w.deinit();

        const fused = (try convAccumSeqFused(ctx, w, cs, x, L, QKV, K)) orelse {
            // Kernel unavailable on this machine: nothing to compare.
            return error.SkipZigTest;
        };
        defer fused.deinit();

        const window = try shape_mod.concatenateAxis(ctx, &[_]Array{ cs, x }, 0);
        defer window.deinit();
        const acc = try convAccumSeq(ctx, w, window, L, QKV);
        defer acc.deinit();
        const want = try activations.silu(ctx, acc);
        defer want.deinit();

        const d = try ops.subtract(ctx, fused, want);
        defer d.deinit();
        const ad = try ops.abs(ctx, d);
        defer ad.deinit();
        const mx = try ops.max(ctx, ad);
        defer mx.deinit();
        try mx.eval();
        const worst = (try mx.dataSlice(f32))[0];
        if (!(worst <= 1e-5)) {
            std.debug.print("convAccumSeqFused mismatch at L={d}: max abs diff {e}\n", .{ L, worst });
            return error.TestUnexpectedResult;
        }
    }
}

test "prefill fused conv path produces the same new conv state" {
    // The fused path slices the new conv state from the projection output
    // instead of the concatenated window. Those must be the same rows:
    // window[L .. L+K-1) == qkv[L-(K-1) .. L) whenever L >= K-1. An off-by-one
    // here would silently corrupt the next chunk's conv context.
    const allocator = std.testing.allocator;
    c.initErrorHandler();
    const ctx = EagerContext.init(allocator);

    const K: i32 = 4;
    const QKV: i32 = 32;
    const L: i32 = 16;
    const x = try makeTestWeight(allocator, &[_]i32{ L, QKV });
    defer x.deinit();
    const cs = try makeTestWeight(allocator, &[_]i32{ K - 1, QKV });
    defer cs.deinit();

    const window = try shape_mod.concatenateAxis(ctx, &[_]Array{ cs, x }, 0);
    defer window.deinit();
    const from_window = try ops.slice(ctx, window, &[_]i32{ L, 0 }, &[_]i32{ L + K - 1, QKV }, &[_]i32{ 1, 1 });
    defer from_window.deinit();
    const from_x = try ops.slice(ctx, x, &[_]i32{ L - (K - 1), 0 }, &[_]i32{ L, QKV }, &[_]i32{ 1, 1 });
    defer from_x.deinit();

    const d = try ops.subtract(ctx, from_window, from_x);
    defer d.deinit();
    const ad = try ops.abs(ctx, d);
    defer ad.deinit();
    const mx = try ops.max(ctx, ad);
    defer mx.deinit();
    try mx.eval();
    try std.testing.expectEqual(@as(f32, 0.0), (try mx.dataSlice(f32))[0]);
}
