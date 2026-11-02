// SPDX-License-Identifier: Apache-2.0
//
// Verify-shape quantized-matmul kernels for MTP speculative decoding.
//
// Ported from oMLX (omlx/patches/qwen35_verify_qmm.py, Apache-2.0). The
// split-K morphology there is adapted from MTPLX
// (https://github.com/youssofal/mtplx, Copyright 2026 Youssof Altoukhi,
// Apache-2.0) — "Powered by MTPLX by Youssof Altoukhi". The sg8 kernel's
// `128 + q` weight operand bits and per-group input-sum correction are
// adapted from Splash
// (runtime/metal/kernels/decode/linear_q4_sgmatrix.metal, Apache-2.0,
// https://github.com/incoai/splash).
//
// ## Why this exists
//
// Speculative verify multiplies a skinny row batch (M = 1 + draft depth)
// against the model's quantized weights. Stock MLX qmm is tuned for M=1
// (decode) and large M (prefill); in between it dispatches to the batched
// `qmv` kernel, whose cost climbs steeply from M=3
// (`mlx/backend/metal/quantized.cpp` `QuantizedMatmul::eval_gpu` routes
// `M < get_qmv_batch_limit` there, and for K=5120 that limit is 6..12).
// Measured on M4 Pro, qmm summed over one 64-layer Qwen3.8-27B forward:
// 57 / 67 / 84 / 105 / 137 ms at M = 2 / 3 / 4 / 5 / 6. Padding M up into
// MLX's split-K GEMM does not help — that path only engages at M >= 14 and
// is ~3x worse in absolute terms.
//
// ## Morphologies
//
// - split-K: one threadgroup owns 4 output columns; K_PARTS simdgroups each
//   reduce a slice of the K packs; a threadgroup reduction combines them.
//   Used for M = 2..3, where sg8 is not eligible.
// - sg8: 4..8 rows on exact simdgroup-matrix operands. A bf16 whose bits are
//   `0x4300 | q` is EXACTLY 128 + q (0x4300 is 128.0, and the 7-bit mantissa
//   makes 128 * (1 + q/128) = 128 + q for q in 0..15), so a 4-bit code
//   becomes a usable operand with one OR — no dequant and no threadgroup
//   staging. The product yields sum((128 + q) * x) = 128 * sum(x) + sum(q*x),
//   and the per-group fold `s * (acc - 128 * sum(x)) + b * sum(x)` recovers
//   sum((q*s + b) * x), i.e. the affine dequant. Its cost is nearly
//   independent of M (measured flat 3.5 ms for lm_head at M = 4..8 while
//   stock goes 4.95 -> 15.52), which is the whole point.
//
// Measured speedup of the pair over stock, summed over a 64-layer forward:
// 1.03x (M=2), 1.11x (M=3), 1.26x (M=4), 1.38x (M=5), 1.74x (M=6), 2.10x (M=8).
//
// ## Implementation note that is easy to get wrong
//
// Both kernels are CODE-GENERATED with every row/column index a compile-time
// constant and no thread-local arrays. A version using `Vec8 xr[M]` and
// `acc[j * M + r]` with loop variables was bit-correct but 10x SLOWER than
// stock at M = 3/5/6 (fine at M = 2/4): the Metal compiler only kept the
// arrays in registers when the trip count was a power of two and otherwise
// spilled them to thread-local memory. This is why oMLX generates unrolled
// source too. Do not "simplify" these into loops.
//
// ## Numerics
//
// fp32 accumulation in a different K order than stock qmm, so results differ
// at the bf16 tail-ULP level (measured max relative deviation 1e-5..4e-3).
// For intermediate projections the deviation is absorbed by the following
// norms. For `lm_head` it lands at ~0.07 absolute on logits of magnitude
// ~20, which is the same order as the top-2 gap that flips a greedy argmax —
// so lm_head routing is OFF by default and must be enabled explicitly
// (`DMLX_VERIFY_QMM_LM_HEAD=1`) and validated with the self-consistency
// probe. `DMLX_NO_VERIFY_QMM=1` disables the whole module.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const quantize_mod = @import("mlx").quantize;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

/// Routing is armed only around the MTP verify forward, so nothing else in
/// the process (decode, prefill, the drafter) sees these kernels. The serving
/// engine runs verify on a single thread, so a plain global is sufficient;
/// if verify ever moves off-thread this must become thread-local.
var armed: bool = false;
var disabled: ?bool = null;
var route_lm_head: ?bool = null;

/// The default CPU stream singleton, created lazily (mlx-c hands out the
/// same underlying stream for every `mlx_default_cpu_stream_new` call, so
/// `mlx_stream_equal` against it identifies any default-CPU-stream context).
var default_cpu_stream: ?c.c.mlx_stream = null;

fn defaultCpuStream() c.c.mlx_stream {
    if (default_cpu_stream == null) {
        default_cpu_stream = c.c.mlx_default_cpu_stream_new();
    }
    return default_cpu_stream.?;
}

/// Projections narrower than this are left on stock: the kernels win on GPU
/// time at every eligible shape, but the tiny GDN/attention outputs have so
/// little work that the win is in the noise.
const min_route_n_default: i32 = 32;
var min_route_n_cached: ?i32 = null;

/// Threshold below which a projection stays on stock qmm.
///
/// 1024 was chosen on the reasoning that "the tiny GDN/attention outputs have
/// so little work that the win is in the noise". That skipped `in_proj_a` and
/// `in_proj_b` (n = 48), and the routing tally shows **480 stock calls per
/// round at M=5** for them (48 layers x 2 projections x 5 rounds). Sectioned
/// profiling then put +18.9 ms of the +30 ms M=3->M=5 verify marginal inside
/// the GDN front — where those two projections live and where the weight
/// bytes do NOT scale with M, so the growth is qmm's response to row count,
/// exactly what these kernels exist to fix (plan doc §37).
/// Override with `DMLX_VERIFY_QMM_MIN_N`.
const sg8_min_m_default: i32 = 6;
var sg8_min_m_cached: ?i32 = null;

fn sg8MinM() i32 {
    if (sg8_min_m_cached) |v| return v;
    const v = blk: {
        const s = std.c.getenv("DMLX_VERIFY_QMM_SG8_MIN_M") orelse break :blk sg8_min_m_default;
        break :blk std.fmt.parseInt(i32, std.mem.span(s), 10) catch sg8_min_m_default;
    };
    sg8_min_m_cached = v;
    return v;
}

fn minRouteN() i32 {
    if (min_route_n_cached) |v| return v;
    const v = blk: {
        const s = std.c.getenv("DMLX_VERIFY_QMM_MIN_N") orelse break :blk min_route_n_default;
        break :blk std.fmt.parseInt(i32, std.mem.span(s), 10) catch min_route_n_default;
    };
    min_route_n_cached = v;
    return v;
}
/// `lm_head` is the only projection whose output feeds an argmax directly.
const lm_head_n_floor: i32 = 100_000;

fn envTrue(name: [*:0]const u8) bool {
    return std.c.getenv(name) != null;
}

pub fn arm() void {
    if (disabled == null) disabled = envTrue("DMLX_NO_VERIFY_QMM");
    // `lm_head` routing is ON by default. It is sg8-only (see `pick`), so it
    // can only engage at M >= 4 where sg8 is a flat 3.78 ms against stock's
    // 6.59 ms at M=5 — i.e. it flattens cost(n) exactly where the depth
    // decision is made.
    //
    // It was default-off because the ~0.07 logit deviation is the size of an
    // argmax-flipping top-2 gap, gated on `mtp_self_consistency.py` demanding
    // byte-identity with plain greedy. That gate was the wrong criterion:
    // speculative decoding cannot be byte-identical to plain decode (verify
    // runs L rows in one forward; see mtp_fidelity_profile.py), and the
    // BASELINE fails it too at 256 tokens — so judging this change by it
    // produced a wrong rejection (plan doc §36).
    //
    // Re-judged with `scripts/mtp_fidelity_profile.py`, which compares WHERE
    // each config first departs from plain greedy: base 5/6 identical vs
    // variant 5/6, and the single divergence lands at the same character
    // index (235) in both. Fidelity-neutral.
    // `DMLX_NO_VERIFY_QMM_LM_HEAD=1` disables it.
    if (route_lm_head == null) route_lm_head = !envTrue("DMLX_NO_VERIFY_QMM_LM_HEAD");
    armed = !disabled.?;
}

pub fn disarm() void {
    armed = false;
}

pub fn isArmed() bool {
    return armed;
}

// ---------------------------------------------------------------------------
// Shared kernel plumbing
// ---------------------------------------------------------------------------

fn makeStringVec(allocator: std.mem.Allocator, items: []const []const u8) !c.c.mlx_vector_string {
    const vec = c.c.mlx_vector_string_new();
    for (items) |it| {
        const z = try allocator.dupeZ(u8, it);
        defer allocator.free(z);
        _ = c.c.mlx_vector_string_append_value(vec, z.ptr);
    }
    return vec;
}

/// Append-only source builder (ArrayList(u8) has no writer in Zig 0.16).
const Emit = struct {
    a: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    fn raw(self: @This(), text: []const u8) !void {
        try self.buf.appendSlice(self.a, text);
    }
    fn fmt(self: @This(), comptime f: []const u8, args: anytype) !void {
        var tmp: [768]u8 = undefined;
        try self.buf.appendSlice(self.a, try std.fmt.bufPrint(&tmp, f, args));
    }
};

fn newKernel(
    allocator: std.mem.Allocator,
    name: []const u8,
    body: []const u8,
) !c.c.mlx_fast_metal_kernel {
    const name_z = try allocator.dupeZ(u8, name);
    defer allocator.free(name_z);
    const source_z = try allocator.dupeZ(u8, body);
    defer allocator.free(source_z);
    const header_z = try allocator.dupeZ(u8, "");
    defer allocator.free(header_z);
    const in_vec = try makeStringVec(allocator, &.{ "x", "w_q", "scales", "biases" });
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = try makeStringVec(allocator, &.{"y"});
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

// ---------------------------------------------------------------------------
// split-K (M = 2..3)
// ---------------------------------------------------------------------------

fn buildSplitkSource(allocator: std.mem.Allocator, m: i32, bn: i32) ![]u8 {
    var s = std.ArrayList(u8).empty;
    errdefer s.deinit(allocator);
    const w = Emit{ .a = allocator, .buf = &s };

    // BN = output columns per threadgroup. Widening 4 -> 8 halves the x
    // re-read traffic across the N grid (every column tile re-reads the
    // activation rows for its K-share; traffic is proportional to N/BN x M),
    // which is the dominant in-context cost at verify M=3..6 (2026-10-30:
    // verify marginal row 8.1 vs omlx ~3.2 ms, traced to this term).
    try w.raw(
        \\    const uint part = simdgroup_index_in_threadgroup;
        \\    const uint lane = thread_index_in_simdgroup;
        \\    const uint tg_n = threadgroup_position_in_grid.y;
        \\    constexpr int K_by_p = K / 8;
        \\    constexpr int K_by_gs = K / GS;
        \\    const int per_part = K_by_p / K_PARTS;
        \\
    );
    try w.fmt("    const int n0 = int(tg_n) * {d};\n", .{bn});
    try w.raw(
        \\    const int p_start = int(part) * per_part;
        \\    const int p_end = (int(part) == K_PARTS - 1) ? K_by_p : p_start + per_part;
        \\    using Vec8 = vec<T, 8>;
        \\    const device Vec8 *xv = (const device Vec8 *)x;
        \\
    );
    var j: i32 = 0;
    while (j < bn) : (j += 1) {
        var r: i32 = 0;
        while (r < m) : (r += 1) try w.fmt("    float a{d}_{d} = 0.0f;\n", .{ j, r });
    }
    try w.raw(
        \\    for (int pack = p_start + int(lane); pack < p_end; pack += 32) {
        \\        const int k_base = pack * 8;
        \\        const int gi = k_base / GS;
        \\
    );
    var r0: i32 = 0;
    while (r0 < m) : (r0 += 1) {
        try w.fmt("        const Vec8 x{d} = xv[({d} * K + k_base) / 8];\n", .{ r0, r0 });
    }
    j = 0;
    while (j < bn) : (j += 1) {
        try w.fmt(
            \\        {{
            \\            const uint32_t pw = w_q[(n0 + {d}) * K_by_p + pack];
            \\            const float s = float(scales[(n0 + {d}) * K_by_gs + gi]);
            \\            const float b = float(biases[(n0 + {d}) * K_by_gs + gi]);
            \\
        , .{ j, j, j });
        var ki: i32 = 0;
        while (ki < 8) : (ki += 1) {
            try w.fmt("            {{ const float wv = float((pw >> {d}) & 0xFu) * s + b;\n", .{ki * 4});
            var r: i32 = 0;
            while (r < m) : (r += 1) {
                try w.fmt("              a{d}_{d} += float(x{d}[{d}]) * wv;\n", .{ j, r, r, ki });
            }
            try w.raw("            }\n");
        }
        try w.raw("        }\n");
    }
    try w.raw("    }\n");
    j = 0;
    while (j < bn) : (j += 1) {
        var r: i32 = 0;
        while (r < m) : (r += 1) {
            try w.fmt("    a{d}_{d} = metal::simd_sum(a{d}_{d});\n", .{ j, r, j, r });
        }
    }
    try w.fmt(
        \\    threadgroup float partials[K_PARTS * {d}];
        \\    if (lane == 0) {{
        \\
    , .{bn * m});
    j = 0;
    while (j < bn) : (j += 1) {
        var r: i32 = 0;
        while (r < m) : (r += 1) {
            try w.fmt("        partials[int(part) * {d} + {d}] = a{d}_{d};\n", .{ bn * m, j * m + r, j, r });
        }
    }
    try w.fmt(
        \\    }}
        \\    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    if (part == 0) {{
        \\        for (int idx = int(lane); idx < {d}; idx += 32) {{
        \\            float total = 0.0f;
        \\            for (int p = 0; p < K_PARTS; ++p) {{ total += partials[p * {d} + idx]; }}
        \\            const int j = idx / {d};
        \\            const int row = idx - j * {d};
        \\            y[row * N + n0 + j] = T(total);
        \\        }}
        \\    }}
        \\
    , .{ bn * m, bn * m, m, m });
    return s.toOwnedSlice(allocator);
}

const splitk_k_parts: i32 = 2; // measured best or tied at every shape

var splitk_kernels = [_]?c.c.mlx_fast_metal_kernel{null} ** 9;

fn getSplitk(allocator: std.mem.Allocator, m: i32) !c.c.mlx_fast_metal_kernel {
    if (splitk_kernels[@intCast(m)]) |k| return k;
    const body = try buildSplitkSource(allocator, m, 4);
    defer allocator.free(body);
    var nbuf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&nbuf, "dmlx_vk_splitk_q4_m{d}", .{m});
    const k = try newKernel(allocator, name, body);
    splitk_kernels[@intCast(m)] = k;
    return k;
}

// ---------------------------------------------------------------------------
// sg8 (M = 4..8)
// ---------------------------------------------------------------------------

const Sg8Geom = struct { nsg: i32, ct: i32 };

fn sg8Geometry(k: i32, n: i32) Sg8Geom {
    if (n >= 65536) return .{ .nsg = 4, .ct = 4 };
    return .{ .nsg = 8, .ct = if (k >= 16384) 2 else 1 };
}

fn buildSg8Source(allocator: std.mem.Allocator, ct: i32) ![]u8 {
    var s = std.ArrayList(u8).empty;
    errdefer s.deinit(allocator);
    const w = Emit{ .a = allocator, .buf = &s };

    try w.raw(
        \\    const uint tid = thread_position_in_threadgroup.x;
        \\    const uint sg = tid / 32;
        \\    const uint lane = tid % 32;
        \\    // Apple 8x8 simdgroup_matrix thread-element mapping.
        \\    const uint qid = lane >> 2;
        \\    const uint fm = (qid & 4) | ((lane >> 1) & 3);
        \\    const uint fn_ = ((qid & 2) << 1) | ((lane & 1) << 1);
        \\    constexpr int GROUPS = K / GS;
        \\    constexpr int PER = GROUPS / NSG;
        \\    constexpr int WB = GS * 4 / 8;
        \\    constexpr long ROW_BYTES = long(K) * 4 / 8;
        \\    const int g_begin = int(sg) * PER;
        \\    const int nbase = int(threadgroup_position_in_grid.y) * 8 * CT;
        \\    const bool r0 = int(fn_) < M;
        \\    const bool r1 = int(fn_) + 1 < M;
        \\    // Step j of a 32-wide k block takes codes j and j+4 of the lane's
        \\    // eight packed values, so one shift+mask yields both A columns.
        \\    const uint kx = (fm >> 1) * 8 + (fm & 1u) * 4;
        \\    // Lanes past the last row read row M-1 and never store, so the hot
        \\    // loop carries no row branches.
        \\    const device T* x0 = x + min(int(fn_), M - 1) * K + kx;
        \\    const device T* x1 = x + min(int(fn_) + 1, M - 1) * K + kx;
        \\
    );
    var ci: i32 = 0;
    while (ci < ct) : (ci += 1) try w.fmt("    float2 out{d} = float2(0.0f);\n", .{ci});
    try w.raw("    for (int g = g_begin; g < g_begin + PER; ++g) {\n");
    ci = 0;
    while (ci < ct) : (ci += 1) {
        try w.fmt("        simdgroup_matrix<float, 8, 8> acc{d} = simdgroup_matrix<float, 8, 8>(0.0f);\n", .{ci});
    }
    try w.raw(
        \\        float2 xsum = float2(0.0f);
        \\        for (int qd = 0; qd < GS / 32; ++qd) {
        \\            const int kq = g * GS + qd * 32;
        \\            const vec<T, 4> va = *((const device vec<T, 4>*)(x0 + kq));
        \\            const vec<T, 4> vb = *((const device vec<T, 4>*)(x1 + kq));
        \\            xsum += float2(float(va[0]) + float(va[1]) + float(va[2]) + float(va[3]),
        \\                           float(vb[0]) + float(vb[1]) + float(vb[2]) + float(vb[3]));
        \\
    );
    ci = 0;
    while (ci < ct) : (ci += 1) {
        try w.fmt(
            \\            {{
            \\                const long n = long(nbase) + {d} * 8 + long(fm);
            \\                const device uchar* wg = (const device uchar*)w_q
            \\                    + n * ROW_BYTES + long(g) * WB;
            \\                const uint wd = ((const device uint*)(wg + qd * 16))[fn_ >> 1];
            \\
        , .{ci});
        var j: i32 = 0;
        while (j < 4) : (j += 1) {
            try w.fmt(
                \\                {{
                \\                    const uint pr = ((wd >> {d}) & 0x000F000Fu) | 0x43004300u;
                \\                    simdgroup_matrix<T, 8, 8> ma, mb;
                \\                    ma.thread_elements()[0] = as_type<T>(ushort(pr));
                \\                    ma.thread_elements()[1] = as_type<T>(ushort(pr >> 16));
                \\                    mb.thread_elements()[0] = va[{d}];
                \\                    mb.thread_elements()[1] = vb[{d}];
                \\                    simdgroup_multiply_accumulate(acc{d}, ma, mb, acc{d});
                \\                }}
                \\
            , .{ j * 4, j, j, ci, ci });
        }
        try w.raw("            }\n");
    }
    try w.raw(
        \\        }
        \\        // Lanes sharing fn_ differ only in the fm bits of the lane id.
        \\        xsum += float2(simd_shuffle_xor(xsum.x, 2), simd_shuffle_xor(xsum.y, 2));
        \\        xsum += float2(simd_shuffle_xor(xsum.x, 4), simd_shuffle_xor(xsum.y, 4));
        \\        xsum += float2(simd_shuffle_xor(xsum.x, 16), simd_shuffle_xor(xsum.y, 16));
        \\
    );
    ci = 0;
    while (ci < ct) : (ci += 1) {
        try w.fmt(
            \\        {{
            \\            const long n = long(nbase) + {d} * 8 + long(fm);
            \\            const float sc = float(scales[n * GROUPS + g]);
            \\            const float bi = float(biases[n * GROUPS + g]);
            \\            const float2 a2 = float2(acc{d}.thread_elements()[0],
            \\                                     acc{d}.thread_elements()[1]);
            \\            out{d} += sc * a2 + (bi - 128.0f * sc) * xsum;
            \\        }}
            \\
        , .{ ci, ci, ci, ci });
    }
    try w.fmt(
        \\    }}
        \\    threadgroup float2 part[NSG * {d} * 32];
        \\
    , .{ct});
    ci = 0;
    while (ci < ct) : (ci += 1) {
        try w.fmt("    part[({d} * NSG + sg) * 32 + lane] = out{d};\n", .{ ci, ci });
    }
    try w.raw(
        \\    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    if (sg == 0) {
        \\
    );
    ci = 0;
    while (ci < ct) : (ci += 1) {
        try w.fmt(
            \\        {{
            \\            const long n = long(nbase) + {d} * 8 + long(fm);
            \\            float2 t = float2(0.0f);
            \\            for (int i = 0; i < NSG; ++i) {{ t += part[({d} * NSG + i) * 32 + lane]; }}
            \\            if (r0) {{ y[long(fn_) * N + n] = T(t.x); }}
            \\            if (r1) {{ y[(long(fn_) + 1) * N + n] = T(t.y); }}
            \\        }}
            \\
        , .{ ci, ci });
    }
    try w.raw("    }\n");
    return s.toOwnedSlice(allocator);
}

var sg8_kernels = [_]?c.c.mlx_fast_metal_kernel{null} ** 5;

fn getSg8(allocator: std.mem.Allocator, ct: i32) !c.c.mlx_fast_metal_kernel {
    if (sg8_kernels[@intCast(ct)]) |k| return k;
    const body = try buildSg8Source(allocator, ct);
    defer allocator.free(body);
    var nbuf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&nbuf, "dmlx_vk_sg8_q4_ct{d}", .{ct});
    const k = try newKernel(allocator, name, body);
    sg8_kernels[@intCast(ct)] = k;
    return k;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

const Kind = enum { splitk, sg8 };

/// A morphology whose kernel failed to COMPILE is never retried — but per
/// morphology: an sg8 build failure says nothing about split-K.
var compile_broken: [2]bool = .{ false, false };

/// Shapes whose DISPATCH failed, so they fall back to stock without taking
/// anything else down with them.
///
/// This replaces a single process-global `broken` flag. That flag was
/// tolerable while `min_route_n` was 1024 and only a handful of wide shapes
/// routed; lowering it to 32 (§37.3) brought far more shapes into the path,
/// which made one odd shape able to disable routing for the entire process —
/// a fragility introduced by that very change (§37.5). The toy-dimension
/// unit tests already trip it: they run on a CPU stream, where
/// `mlx_fast_metal_kernel_apply` reports "[metal_kernel] Only supports the
/// GPU".
var failed_shapes: [32]struct { kind: Kind, k: i32, n: i32 } = undefined;
var failed_count: usize = 0;

fn shapeFailed(kind: Kind, k: i32, n: i32) bool {
    for (failed_shapes[0..failed_count]) |f| {
        if (f.kind == kind and f.k == k and f.n == n) return true;
    }
    return false;
}

fn markShapeFailed(kind: Kind, k: i32, n: i32) void {
    if (shapeFailed(kind, k, n)) return;
    if (failed_count < failed_shapes.len) {
        failed_shapes[failed_count] = .{ .kind = kind, .k = k, .n = n };
        failed_count += 1;
    } else {
        // Table full: something is systematically wrong, so stop trying.
        compile_broken = .{ true, true };
    }
}

// ---------------------------------------------------------------------------
// Routing tally (DMLX_VERIFY_QMM_TALLY=1)
// ---------------------------------------------------------------------------
// The benchmarked speedups (1.24x at M=4, 1.36x at M=5 summed over a forward)
// are NOT what production delivers: an A/B against DMLX_NO_VERIFY_QMM=1
// measured only -2.8 ms at M=4 and -19.1 ms at M=5, i.e. 13.8 / 8.6 ms of the
// prediction unrealized (plan doc §28). Every projection call site does route
// through maybeQmm, so the loss must be in the eligibility predicates. This
// tallies, per distinct (M, K, N) shape, how many calls took sg8 / split-K /
// fell through to stock, so the misses can be named instead of guessed at.

const TallyRow = struct {
    m: i32 = 0,
    k: i32 = 0,
    n: i32 = 0,
    sg8: u32 = 0,
    splitk: u32 = 0,
    stock: u32 = 0,
};
var tally: [48]TallyRow = @splat(.{});
var tally_enabled: ?bool = null;

fn tallyOn() bool {
    if (tally_enabled) |v| return v;
    const v = envTrue("DMLX_VERIFY_QMM_TALLY");
    tally_enabled = v;
    return v;
}

fn tallyRecord(m: i32, k: i32, n: i32, kind: ?Kind) void {
    for (&tally) |*row| {
        if (row.n == 0) {
            row.* = .{ .m = m, .k = k, .n = n };
        } else if (row.m != m or row.k != k or row.n != n) {
            continue;
        }
        switch (kind orelse {
            row.stock += 1;
            return;
        }) {
            .sg8 => row.sg8 += 1,
            .splitk => row.splitk += 1,
        }
        return;
    }
}

/// Print and clear the routing tally. Rows with a non-zero `stock` count at
/// M >= 4 are the unrealized speedup.
pub fn tallyReport() void {
    if (!tallyOn()) return;
    var any = false;
    for (tally) |row| {
        if (row.n != 0) any = true;
    }
    if (!any) return;
    std.debug.print("\n=== verify_qmm routing tally (M, K, N -> calls) ===\n", .{});
    std.debug.print("  {s: >3} {s: >6} {s: >8} {s: >7} {s: >8} {s: >7}\n", .{ "M", "K", "N", "sg8", "split-K", "stock" });
    for (&tally) |*row| {
        if (row.n == 0) continue;
        std.debug.print("  {d: >3} {d: >6} {d: >8} {d: >7} {d: >8} {d: >7}{s}\n", .{
            row.m,                                                 row.k, row.n, row.sg8, row.splitk, row.stock,
            if (row.stock > 0 and row.m >= 4) "  <- MISS" else "",
        });
        row.* = .{};
    }
}

fn pick(m: i32, k: i32, n: i32, gs: i32, bits: u8) ?Kind {
    if (bits != 4) return null; // 5/8-bit paths not ported yet
    if (gs != 32 and gs != 64 and gs != 128) return null;
    if (n < minRouteN() or @mod(n, 4) != 0) return null;
    if (n >= lm_head_n_floor and !(route_lm_head orelse false)) return null;
    if (@mod(k, 8) != 0) return null;
    // Morphology choice is SHAPE-dependent, not just row-count dependent.
    //
    // This used to return sg8 whenever it was eligible (M >= 4), which is
    // wrong on every projection except `lm_head`. From
    // `bench_verify_kernel` (ms per call, kp=2 vs sg8):
    //
    //   shape                  M=4            M=5            M=6
    //   gdn_qkv [10240,5120]   .1407 / .1677  .1610 / .1688  .1833 / .1691
    //   gdn_z   [6144,5120]    .0853 / .1014  .0988 / .1061  .1124 / .1016
    //   gdn_out [5120,6144]    .0847 / .1041  .0980 / .1013  .1118 / .1046
    //   attn_q  [12288,5120]   .1687 / .2005  .1908 / .2002  .2191 / .2020
    //   attn_o  [5120,6144]    .0837 / .1032  .0958 / .1015  .1170 / .1031
    //   attn_kv [1024,5120]    .0202 / .0217  .0210 / .0218  .0228 / .0218
    //   lm_head [248320,5120] 5.59  / 3.75   6.79  / 3.78   —
    //
    // split-K wins by 3-23% at M=4..5 on everything narrow, and sg8 only
    // takes over at M>=6; `lm_head` is the opposite — sg8 wins by 25-43%
    // because its cost is flat in M while stock/split-K climb. Since the
    // depth controller operates at M=3..5, preferring sg8 there cost ~3.9 ms
    // per round at M=4 and ~1.2 ms at M=5, summed over the ~330 routed
    // projections (plan doc §38).
    // `DMLX_VERIFY_QMM_SG8_MIN_M` overrides the non-lm_head threshold (6);
    // setting it to 4 restores the old sg8-always-wins routing for A/B.
    const sg8_min_m: i32 = if (n >= lm_head_n_floor) 4 else sg8MinM();
    if (m >= sg8_min_m and m <= 8) {
        const g = sg8Geometry(k, n);
        if (@mod(k, gs * g.nsg) == 0 and @mod(n, 8 * g.ct) == 0) return .sg8;
    }
    // `lm_head` is sg8-only. split-K measurably LOSES on this shape — bench
    // `lm_head [248320,5120]` gives stock 3.22/3.82/4.80 vs kp=2
    // 3.22/4.11/5.59 ms at M=2/3/4 — and at the depth the controller
    // actually picks (M=3) routing lm_head through split-K cost 2-4% per
    // round in an A/B (plan doc §28). sg8 needs M>=4, so below that the
    // right answer is stock.
    if (n >= lm_head_n_floor) return null;
    if (m >= 2 and m <= 8 and @mod(@divExact(k, 8), splitk_k_parts) == 0) return .splitk;
    return null;
}

/// Route `x @ dequantize(w)^T` through a verify kernel when armed and
/// eligible. Returns null when the caller should use stock MLX qmm.
///
/// `x` must be [M, K] or [1, M, K] row-contiguous bf16; the output matches
/// the input rank.
pub fn maybeQmm(ctx: EagerContext, x: Array, qw: quantize_mod.QuantizedWeight) !?Array {
    if (!armed or (compile_broken[0] and compile_broken[1])) return null;
    // Metal kernels only run on the GPU. On the default CPU stream (unit
    // tests) stay on stock without attempting — a failed apply logs an MLX
    // error, which the Zig test runner counts as a test failure. NOTE:
    // `mlx_stream_get_device` segfaults in mlx-c 0.7.0 (ABI mismatch), so
    // detect the CPU stream by identity instead of querying its device.
    if (c.c.mlx_stream_equal(ctx.stream.inner, defaultCpuStream())) return null;
    if (x.dtype() != .bfloat16) return null;
    if (qw.config.mode != .affine) return null;

    const xs = x.shape();
    // Accept [M, K] and [1, M, K]; anything else (batched) stays on stock.
    var m: i32 = undefined;
    var k: i32 = undefined;
    var three_d = false;
    if (xs.len == 2) {
        m = @intCast(xs[0]);
        k = @intCast(xs[1]);
    } else if (xs.len == 3 and xs[0] == 1) {
        m = @intCast(xs[1]);
        k = @intCast(xs[2]);
        three_d = true;
    } else return null;

    const ws = qw.data.shape();
    const n: i32 = @intCast(ws[ws.len - 2]);
    const gs = qw.config.group_size;
    const kind_opt = pick(m, k, n, gs, qw.config.bits);
    if (tallyOn()) tallyRecord(m, k, n, kind_opt);
    const kind = kind_opt orelse return null;
    if (compile_broken[@intFromEnum(kind)] or shapeFailed(kind, k, n)) return null;

    const allocator = ctx.allocator;
    // The kernels index x as [M, K]; reshape the 3-D case (cheap view).
    const x2 = if (three_d)
        try ops.reshape(ctx, x, &[_]i32{ m, k })
    else
        x;
    defer if (three_d) x2.deinit();

    const cfg = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg);

    const kernel = switch (kind) {
        .sg8 => blk: {
            const g = sg8Geometry(k, n);
            try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * g.nsg, @divExact(n, 8 * g.ct), 1));
            try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * g.nsg, 1, 1));
            break :blk getSg8(allocator, g.ct) catch {
                compile_broken[@intFromEnum(Kind.sg8)] = true;
                return null;
            };
        },
        .splitk => blk: {
            try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * splitk_k_parts, @divExact(n, 4), 1));
            try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * splitk_k_parts, 1, 1));
            break :blk getSplitk(allocator, m) catch {
                compile_broken[@intFromEnum(Kind.splitk)] = true;
                return null;
            };
        },
    };

    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ m, n }, 2, c.c.MLX_BFLOAT16));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", c.c.MLX_BFLOAT16));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", k));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", n));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "M", m));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", gs));
    switch (kind) {
        .sg8 => {
            const g = sg8Geometry(k, n);
            try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NSG", g.nsg));
            try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "CT", g.ct));
        },
        .splitk => {
            try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K_PARTS", splitk_k_parts));
        },
    }

    const in_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(in_vec);
    for ([_]Array{ x2, qw.data, qw.scales, qw.biases }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(in_vec, inp.inner));
    }
    var out_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(out_vec);
    c.check(c.c.mlx_fast_metal_kernel_apply(&out_vec, kernel, in_vec, cfg, ctx.stream.inner)) catch {
        // A dispatch failure means this shape slipped past `pick` (or the
        // stream is not a GPU stream). Fall back to stock for THIS shape and
        // keep routing everything else.
        markShapeFailed(kind, k, n);
        std.log.warn("verify_qmm: kernel apply failed (M={d} K={d} N={d}); this shape falls back to stock", .{ m, k, n });
        return null;
    };
    var h = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&h, out_vec, 0));
    const y2 = Array.fromHandle(h);
    if (!three_d) return y2;
    defer y2.deinit();
    return try ops.reshape(ctx, y2, &[_]i32{ 1, m, n });
}

test "verify_qmm: source generation is unrolled and array-free" {
    const allocator = std.testing.allocator;
    // The M1 regression: thread-local arrays indexed by a loop variable made
    // the kernel 10x slower at M=3/5/6. Guard the property, not the timing.
    for ([_]i32{ 2, 3, 4, 5, 6, 7, 8 }) |m| {
        const src = try buildSplitkSource(allocator, m, 4);
        defer allocator.free(src);
        try std.testing.expect(std.mem.indexOf(u8, src, "xr[") == null);
        try std.testing.expect(std.mem.indexOf(u8, src, "acc[") == null);
        // One named accumulator per (column, row).
        var count: usize = 0;
        var it = std.mem.splitScalar(u8, src, '\n');
        while (it.next()) |line| {
            if (std.mem.indexOf(u8, line, "float a") != null and
                std.mem.indexOf(u8, line, "= 0.0f;") != null) count += 1;
        }
        try std.testing.expectEqual(@as(usize, @intCast(4 * m)), count);
    }
    for ([_]i32{ 1, 2, 4 }) |ct| {
        const src = try buildSg8Source(allocator, ct);
        defer allocator.free(src);
        try std.testing.expect(std.mem.indexOf(u8, src, "acc[") == null);
        try std.testing.expect(std.mem.indexOf(u8, src, "out[") == null);
        try std.testing.expect(std.mem.indexOf(u8, src, "0x43004300u") != null);
    }
}

test "verify_qmm: dispatch picks the right morphology and respects gates" {
    route_lm_head = false;
    sg8_min_m_cached = sg8_min_m_default;
    min_route_n_cached = min_route_n_default;
    // Morphology is shape-dependent (see `pick`): on ordinary projections
    // split-K wins all the way to M=5 and sg8 only takes over at M>=6.
    // Asserting sg8 from M=4 here is what let the mis-routing ship — it cost
    // 8.8 ms per round at M=4 (plan doc §38).
    try std.testing.expectEqual(Kind.splitk, pick(2, 5120, 10240, 64, 4).?);
    try std.testing.expectEqual(Kind.splitk, pick(3, 5120, 10240, 64, 4).?);
    try std.testing.expectEqual(Kind.splitk, pick(4, 5120, 10240, 64, 4).?);
    try std.testing.expectEqual(Kind.splitk, pick(5, 5120, 10240, 64, 4).?);
    try std.testing.expectEqual(Kind.sg8, pick(6, 5120, 10240, 64, 4).?);
    try std.testing.expectEqual(Kind.sg8, pick(8, 17408, 5120, 64, 4).?);
    // The threshold is overridable, and 4 restores the old (wrong) routing.
    sg8_min_m_cached = 4;
    try std.testing.expectEqual(Kind.sg8, pick(4, 5120, 10240, 64, 4).?);
    sg8_min_m_cached = sg8_min_m_default;
    // lm_head is the ONE shape where sg8 wins from M=4 (its cost is flat in
    // M while stock/split-K climb), and it is sg8-only — split-K loses there.
    try std.testing.expect(pick(4, 5120, 248320, 64, 4) == null); // gated off here
    route_lm_head = true;
    try std.testing.expectEqual(Kind.sg8, pick(4, 5120, 248320, 64, 4).?);
    try std.testing.expect(pick(3, 5120, 248320, 64, 4) == null); // no split-K fallback
    route_lm_head = false;
    // `min_route_n` is 32 (was 1024). The GDN `in_proj_a` / `in_proj_b`
    // projections are n = 48 and were the single largest skipped group —
    // 480 stock calls per round at M=5 — so they MUST route now; that is the
    // whole point of lowering the threshold (plan doc §37).
    min_route_n_cached = min_route_n_default;
    try std.testing.expect(pick(4, 5120, 48, 64, 4) != null);
    try std.testing.expect(pick(2, 5120, 48, 64, 4) != null);
    try std.testing.expect(pick(4, 5120, 512, 64, 4) != null);
    // Still too narrow to be worth a custom kernel, and n must be a multiple
    // of 4 for either morphology's column handling.
    try std.testing.expect(pick(4, 5120, 16, 64, 4) == null);
    try std.testing.expect(pick(4, 5120, 50, 64, 4) == null);
    // The threshold is overridable, and raising it restores the old skip.
    min_route_n_cached = 1024;
    try std.testing.expect(pick(4, 5120, 48, 64, 4) == null);
    min_route_n_cached = min_route_n_default;
    // Non-4-bit, M=1 (decode) and M>8 stay on stock.
    try std.testing.expect(pick(4, 5120, 10240, 64, 8) == null);
    try std.testing.expect(pick(1, 5120, 10240, 64, 4) == null);
    try std.testing.expect(pick(9, 5120, 10240, 64, 4) == null);
}
