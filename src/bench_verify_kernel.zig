// M1 feasibility bench: can a custom verify-shape Metal kernel beat stock MLX
// qmm at M = 2..6 (the MTP verify row counts)?
//
// Background (measured, see docs/analysis/beat-omlx-mlxvlm-plan.md):
// stock MLX dispatches M < get_qmv_batch_limit to the batched `qmv` kernel,
// whose cost climbs steeply from M=3 (59.4/62.0/68.5/84.6/105.8 ms of qmm per
// 64-layer forward at M=1..5). omlx works around this with custom verify
// kernels and reports lm_head 2.6-3.3x at M=3-4. This bench ports the
// simplest of its four morphologies — split-K — and measures the real ratio
// at dmlx's actual shapes before committing to the full port.
//
// split-K morphology (ported from omlx/patches/qwen35_verify_qmm.py
// `_build_ksplit_kernel`, which is itself adapted from MTPLX
// (github.com/youssofal/mtplx, Apache-2.0) — "Powered by MTPLX by Youssof
// Altoukhi"): one threadgroup owns 4 output columns; K_PARTS simdgroups each
// reduce a contiguous slice of the K packs; a threadgroup reduction combines
// the partials. Every row is accumulated against a weight value the moment it
// is unpacked, so the M rows share ONE pass over the weights — which is
// exactly what stock qmv fails to do at M>=3.
//
// Not committed. Run: zig build run-bench-verify-kernel -Doptimize=ReleaseFast
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const random = @import("mlx").random;
const quantize_mod = @import("mlx").quantize;
const eval_mod = @import("mlx").eval;
const dtype_mod = @import("mlx").dtype;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

fn elapsedMs(start: u64) f64 {
    const ns = (std.c.mach_absolute_time() - start) * 125 / 3;
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

// ---------------------------------------------------------------------------
// split-K kernel
// ---------------------------------------------------------------------------
// Template params are supplied in this exact order via
// mlx_fast_metal_kernel_config_add_template_arg_*:
//   T (dtype), K, N, GS, K_PARTS   (M is baked into the generated body)
/// Generate the split-K body with EVERY row/column index a compile-time
/// constant and no thread-local arrays.
///
/// This matters a lot. The first version of this bench used `Vec8 xr[M]` and
/// `acc[j * M + r]` with loop variables. It was bit-correct, but 10x SLOWER
/// than stock at M=3/5/6 while being fine at M=2/4 — the Metal compiler only
/// kept the arrays in registers when M was a power of two, and otherwise
/// spilled them to thread-local memory. That is exactly why omlx generates
/// unrolled source with individually named variables instead of using arrays.
fn buildSplitkSource(allocator: std.mem.Allocator, m: i32) ![]u8 {
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
        \\    const uint part = simdgroup_index_in_threadgroup;
        \\    const uint lane = thread_index_in_simdgroup;
        \\    const uint tg_n = threadgroup_position_in_grid.y;
        \\
        \\    constexpr int K_by_p = K / 8;   // uint32 words per weight row
        \\    constexpr int K_by_gs = K / GS; // quant groups per weight row
        \\    const int per_part = K_by_p / K_PARTS;
        \\    const int n0 = int(tg_n) * 4;
        \\    const int p_start = int(part) * per_part;
        \\    const int p_end = (int(part) == K_PARTS - 1) ? K_by_p : p_start + per_part;
        \\
        \\    using Vec8 = vec<T, 8>;
        \\    const device Vec8 *xv = (const device Vec8 *)x;
        \\
    );
    // Accumulators: one named register per (column, row).
    var j: i32 = 0;
    while (j < 4) : (j += 1) {
        var r: i32 = 0;
        while (r < m) : (r += 1) {
            try w.fmt("    float a{d}_{d} = 0.0f;\n", .{ j, r });
        }
    }
    try w.raw(
        \\
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
    while (j < 4) : (j += 1) {
        try w.fmt(
            \\        {{
            \\            const uint32_t pw = w_q[(n0 + {d}) * K_by_p + pack];
            \\            const float s = float(scales[(n0 + {d}) * K_by_gs + gi]);
            \\            const float b = float(biases[(n0 + {d}) * K_by_gs + gi]);
            \\
        , .{ j, j, j });
        var ki: i32 = 0;
        while (ki < 8) : (ki += 1) {
            try w.fmt(
                "            {{ const float wv = float((pw >> {d}) & 0xFu) * s + b;\n",
                .{ki * 4},
            );
            var r: i32 = 0;
            while (r < m) : (r += 1) {
                try w.fmt("              a{d}_{d} += float(x{d}[{d}]) * wv;\n", .{ j, r, r, ki });
            }
            try w.raw("            }\n");
        }
        try w.raw("        }\n");
    }
    try w.raw("    }\n\n");

    // simd reduction, then a threadgroup reduction across the K parts.
    j = 0;
    while (j < 4) : (j += 1) {
        var r: i32 = 0;
        while (r < m) : (r += 1) {
            try w.fmt("    a{d}_{d} = metal::simd_sum(a{d}_{d});\n", .{ j, r, j, r });
        }
    }
    try w.fmt(
        \\
        \\    threadgroup float partials[K_PARTS * {d}];
        \\    if (lane == 0) {{
        \\
    , .{4 * m});
    j = 0;
    while (j < 4) : (j += 1) {
        var r: i32 = 0;
        while (r < m) : (r += 1) {
            try w.fmt("        partials[int(part) * {d} + {d}] = a{d}_{d};\n", .{ 4 * m, j * m + r, j, r });
        }
    }
    try w.fmt(
        \\    }}
        \\    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\
        \\    if (part == 0 && int(lane) < {d}) {{
        \\        float total = 0.0f;
        \\        for (int p = 0; p < K_PARTS; ++p) {{ total += partials[p * {d} + int(lane)]; }}
        \\        const int j = int(lane) / {d};
        \\        const int row = int(lane) - j * {d};
        \\        y[row * N + n0 + j] = T(total);
        \\    }}
        \\
    , .{ 4 * m, 4 * m, m, m });
    return s.toOwnedSlice(allocator);
}

fn makeStringVec(allocator: std.mem.Allocator, items: []const []const u8) !c.c.mlx_vector_string {
    const vec = c.c.mlx_vector_string_new();
    for (items) |it| {
        const z = try allocator.dupeZ(u8, it);
        defer allocator.free(z);
        _ = c.c.mlx_vector_string_append_value(vec, z.ptr);
    }
    return vec;
}

// ---------------------------------------------------------------------------
// sg8 kernel — 4..8 rows on exact (128 + q) simdgroup-matrix operands.
// ---------------------------------------------------------------------------
// Ported from omlx/patches/qwen35_verify_qmm.py `_sg8_source`, whose
// `128 + q` operand trick and per-group input-sum correction are themselves
// adapted from Splash (runtime/metal/kernels/decode/linear_q4_sgmatrix.metal,
// Apache-2.0, https://github.com/incoai/splash). Apache-2.0.
//
// The trick: a bf16 whose bits are `0x4300 | q` is EXACTLY the value 128 + q
// (0x4300 is 128.0; the 7-bit mantissa makes 128 * (1 + q/128) = 128 + q for
// q in 0..15). So a 4-bit code becomes a usable bf16 operand with one OR —
// no dequant, no threadgroup staging. The matmul then yields
// sum((128 + q) * x) = 128 * sum(x) + sum(q * x), and the per-group fold
// `s * (acc - 128 * sum(x)) + b * sum(x)` recovers sum((q * s + b) * x),
// which is exactly the affine dequant. That is why the kernel needs the
// per-group input sums (`xsum`).
//
// One simdgroup computes C^T = W X^T for CT x 8 columns and all M rows. Lane
// (fm, fn) owns W[nbase + fm] and rows fn, fn+1, which is precisely the
// Apple 8x8 simdgroup_matrix thread-element layout, so MLX's packed layout
// feeds the operand directly.
fn sg8Geometry(k: i32, n: i32) struct { nsg: i32, ct: i32 } {
    if (n >= 65536) return .{ .nsg = 4, .ct = 4 };
    return .{ .nsg = 8, .ct = if (k >= 16384) 2 else 1 };
}

fn sg8Eligible(m: i32, k: i32, n: i32, gs: i32) bool {
    const g = sg8Geometry(k, n);
    return m >= 4 and m <= 8 and
        @mod(k, gs * g.nsg) == 0 and
        @mod(n, 8 * g.ct) == 0 and
        n >= 1024;
}

/// Template params, in order: T, K, N, M, GS, NSG, CT.
/// CT is unrolled in the generated body — see the M1 lesson in
/// docs/analysis/beat-omlx-mlxvlm-plan.md §9.5: thread-local arrays indexed by
/// a loop variable spill to memory unless the trip count is a power of two.
fn buildSg8Source(allocator: std.mem.Allocator, ct: i32) ![]u8 {
    var s = std.ArrayList(u8).empty;
    errdefer s.deinit(allocator);
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
        \\    constexpr int WB = GS * 4 / 8;              // bytes per group row
        \\    constexpr long ROW_BYTES = long(K) * 4 / 8; // bytes per weight row
        \\    const int g_begin = int(sg) * PER;
        \\    const int nbase = int(threadgroup_position_in_grid.y) * 8 * CT;
        \\    const bool r0 = int(fn_) < M;
        \\    const bool r1 = int(fn_) + 1 < M;
        \\    // Step j of a 32-wide k block takes values j and j+4 of the lane's
        \\    // eight packed codes, so one shift+mask yields both A columns.
        \\    const uint kx = (fm >> 1) * 8 + (fm & 1u) * 4;
        \\    // Lanes past the last row read row M-1 and never store, so the hot
        \\    // loop carries no row branches.
        \\    const device T* x0 = x + min(int(fn_), M - 1) * K + kx;
        \\    const device T* x1 = x + min(int(fn_) + 1, M - 1) * K + kx;
        \\
    );
    var c0: i32 = 0;
    while (c0 < ct) : (c0 += 1) {
        try w.fmt("    float2 out{d} = float2(0.0f);\n", .{c0});
    }
    try w.raw(
        \\
        \\    for (int g = g_begin; g < g_begin + PER; ++g) {
        \\
    );
    c0 = 0;
    while (c0 < ct) : (c0 += 1) {
        try w.fmt("        simdgroup_matrix<float, 8, 8> acc{d} = simdgroup_matrix<float, 8, 8>(0.0f);\n", .{c0});
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
    c0 = 0;
    while (c0 < ct) : (c0 += 1) {
        try w.fmt(
            \\            {{
            \\                const long n = long(nbase) + {d} * 8 + long(fm);
            \\                const device uchar* wg = (const device uchar*)w_q
            \\                    + n * ROW_BYTES + long(g) * WB;
            \\                const uint wd = ((const device uint*)(wg + qd * 16))[fn_ >> 1];
            \\
        , .{c0});
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
            , .{ j * 4, j, j, c0, c0 });
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
    c0 = 0;
    while (c0 < ct) : (c0 += 1) {
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
        , .{ c0, c0, c0, c0 });
    }
    try w.fmt(
        \\    }}
        \\
        \\    threadgroup float2 part[NSG * {d} * 32];
        \\
    , .{ct});
    c0 = 0;
    while (c0 < ct) : (c0 += 1) {
        try w.fmt("    part[({d} * NSG + sg) * 32 + lane] = out{d};\n", .{ c0, c0 });
    }
    try w.raw(
        \\    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\
        \\    if (sg == 0) {
        \\
    );
    c0 = 0;
    while (c0 < ct) : (c0 += 1) {
        try w.fmt(
            \\        {{
            \\            const long n = long(nbase) + {d} * 8 + long(fm);
            \\            float2 t = float2(0.0f);
            \\            for (int i = 0; i < NSG; ++i) {{ t += part[({d} * NSG + i) * 32 + lane]; }}
            \\            if (r0) {{ y[long(fn_) * N + n] = T(t.x); }}
            \\            if (r1) {{ y[(long(fn_) + 1) * N + n] = T(t.y); }}
            \\        }}
            \\
        , .{ c0, c0 });
    }
    try w.raw("    }\n");
    return s.toOwnedSlice(allocator);
}

var sg8_kernels = [_]?c.c.mlx_fast_metal_kernel{null} ** 5;

fn getSg8Kernel(allocator: std.mem.Allocator, ct: i32) !c.c.mlx_fast_metal_kernel {
    if (sg8_kernels[@intCast(ct)]) |k| return k;
    const name_z = try std.fmt.allocPrintSentinel(allocator, "dmlx_vk_sg8_q4_ct{d}", .{ct}, 0);
    defer allocator.free(name_z);
    const body = try buildSg8Source(allocator, ct);
    defer allocator.free(body);
    const source_z = try allocator.dupeZ(u8, body);
    defer allocator.free(source_z);
    const header_z = try allocator.dupeZ(u8, "");
    defer allocator.free(header_z);
    const in_vec = try makeStringVec(allocator, &.{ "x", "w_q", "scales", "biases" });
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = try makeStringVec(allocator, &.{"y"});
    defer _ = c.c.mlx_vector_string_free(out_vec);
    const k = c.c.mlx_fast_metal_kernel_new(
        name_z.ptr,
        in_vec,
        out_vec,
        source_z.ptr,
        header_z.ptr,
        true,
        false,
    );
    sg8_kernels[@intCast(ct)] = k;
    return k;
}

fn sg8Qmm(
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    x: Array,
    qw: quantize_mod.QuantizedWeight,
    m: i32,
    k: i32,
    n: i32,
) !Array {
    const g = sg8Geometry(k, n);
    const kernel = try getSg8Kernel(allocator, g.ct);

    const cfg = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg);
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * g.nsg, @divExact(n, 8 * g.ct), 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * g.nsg, 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ m, n }, 2, c.c.MLX_BFLOAT16));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", c.c.MLX_BFLOAT16));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", k));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", n));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "M", m));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", qw.config.group_size));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NSG", g.nsg));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "CT", g.ct));
    if (verbose_once) {
        try c.check(c.c.mlx_fast_metal_kernel_config_set_verbose(cfg, true));
        verbose_once = false;
    }

    const in_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(in_vec);
    for ([_]Array{ x, qw.data, qw.scales, qw.biases }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(in_vec, inp.inner));
    }
    var out_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(out_vec);
    try c.check(c.c.mlx_fast_metal_kernel_apply(&out_vec, kernel, in_vec, cfg, ctx.stream.inner));
    var h = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&h, out_vec, 0));
    return Array.fromHandle(h);
}

/// One kernel per M (the source is unrolled for it).
var splitk_kernels = [_]?c.c.mlx_fast_metal_kernel{null} ** 9;
var verbose_once: bool = false;

fn getSplitkKernel(allocator: std.mem.Allocator, m: i32) !c.c.mlx_fast_metal_kernel {
    if (splitk_kernels[@intCast(m)]) |k| return k;
    const name_z = try std.fmt.allocPrintSentinel(allocator, "dmlx_vk_splitk_q4_m{d}", .{m}, 0);
    defer allocator.free(name_z);
    const body = try buildSplitkSource(allocator, m);
    defer allocator.free(body);
    const source_z = try allocator.dupeZ(u8, body);
    defer allocator.free(source_z);
    const header_z = try allocator.dupeZ(u8, "");
    defer allocator.free(header_z);
    const in_vec = try makeStringVec(allocator, &.{ "x", "w_q", "scales", "biases" });
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = try makeStringVec(allocator, &.{"y"});
    defer _ = c.c.mlx_vector_string_free(out_vec);
    const k = c.c.mlx_fast_metal_kernel_new(
        name_z.ptr,
        in_vec,
        out_vec,
        source_z.ptr,
        header_z.ptr,
        true, // ensure_row_contiguous
        false, // atomic_outputs
    );
    splitk_kernels[@intCast(m)] = k;
    return k;
}

/// (M,K) x (N,K)^T -> (M,N) through the split-K kernel.
fn splitkQmm(
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    x: Array,
    qw: quantize_mod.QuantizedWeight,
    m: i32,
    k: i32,
    n: i32,
    k_parts: i32,
) !Array {
    const kernel = try getSplitkKernel(allocator, m);

    const cfg = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg);
    // One threadgroup per 4 output columns; K_PARTS simdgroups inside it.
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * k_parts, @divExact(n, 4), 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * k_parts, 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]i32{ m, n }, 2, c.c.MLX_BFLOAT16));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", c.c.MLX_BFLOAT16));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", k));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", n));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", qw.config.group_size));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K_PARTS", k_parts));
    if (verbose_once) {
        try c.check(c.c.mlx_fast_metal_kernel_config_set_verbose(cfg, true));
        verbose_once = false;
    }

    const in_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(in_vec);
    for ([_]Array{ x, qw.data, qw.scales, qw.biases }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(in_vec, inp.inner));
    }
    var out_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(out_vec);
    try c.check(c.c.mlx_fast_metal_kernel_apply(&out_vec, kernel, in_vec, cfg, ctx.stream.inner));
    var h = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&h, out_vec, 0));
    return Array.fromHandle(h);
}

const Shape = struct {
    name: []const u8,
    out: i32, // N
    in: i32, // K
    per_token: u32,
};

const shapes = [_]Shape{
    .{ .name = "gdn_qkv  [10240,5120]", .out = 10240, .in = 5120, .per_token = 48 },
    .{ .name = "gdn_z    [6144,5120]", .out = 6144, .in = 5120, .per_token = 48 },
    .{ .name = "gdn_out  [5120,6144]", .out = 5120, .in = 6144, .per_token = 48 },
    .{ .name = "attn_q   [12288,5120]", .out = 12288, .in = 5120, .per_token = 16 },
    .{ .name = "attn_o   [5120,6144]", .out = 5120, .in = 6144, .per_token = 16 },
    .{ .name = "attn_kv  [1024,5120]", .out = 1024, .in = 5120, .per_token = 32 },
    .{ .name = "mlp_gate [17408,5120]", .out = 17408, .in = 5120, .per_token = 64 },
    .{ .name = "mlp_up   [17408,5120]", .out = 17408, .in = 5120, .per_token = 64 },
    .{ .name = "mlp_down [5120,17408]", .out = 5120, .in = 17408, .per_token = 64 },
    .{ .name = "lm_head  [248320,5120]", .out = 248320, .in = 5120, .per_token = 1 },
};

fn timeStock(allocator: std.mem.Allocator, ctx: EagerContext, x: Array, qw: anytype, reps: usize) !f64 {
    var pending = std.ArrayList(Array).empty;
    defer pending.deinit(allocator);
    defer for (pending.items) |y| y.deinit();
    for (0..reps) |_| {
        const y = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
        try pending.append(allocator, y);
    }
    const t = std.c.mach_absolute_time();
    try eval_mod.asyncEval(allocator, pending.items);
    try pending.items[reps - 1].eval();
    return elapsedMs(t) / @as(f64, @floatFromInt(reps));
}

fn timeSplitk(
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    x: Array,
    qw: quantize_mod.QuantizedWeight,
    m: i32,
    k: i32,
    n: i32,
    k_parts: i32,
    reps: usize,
) !f64 {
    var pending = std.ArrayList(Array).empty;
    defer pending.deinit(allocator);
    defer for (pending.items) |y| y.deinit();
    for (0..reps) |_| {
        const y = try splitkQmm(allocator, ctx, x, qw, m, k, n, k_parts);
        try pending.append(allocator, y);
    }
    const t = std.c.mach_absolute_time();
    try eval_mod.asyncEval(allocator, pending.items);
    try pending.items[reps - 1].eval();
    return elapsedMs(t) / @as(f64, @floatFromInt(reps));
}

/// Max abs difference between the two paths, in units of the stock result's
/// max abs value (so it is comparable across shapes).
fn relDiff(ctx: EagerContext, a: Array, b: Array) !f64 {
    // NB: cast to f32 BEFORE reducing/reading. `dataSlice(f32)` on a bf16
    // array reinterprets the bytes and silently returns garbage — that bug
    // made a bit-correct kernel look like it produced nonsense.
    const a32 = try ops.astype(ctx, a, .float32);
    defer a32.deinit();
    const b32 = try ops.astype(ctx, b, .float32);
    defer b32.deinit();
    const d = try ops.subtract(ctx, a32, b32);
    defer d.deinit();
    const ad = try ops.abs(ctx, d);
    defer ad.deinit();
    const mx = try ops.max(ctx, ad);
    defer mx.deinit();
    try mx.eval();
    const aa = try ops.abs(ctx, a32);
    defer aa.deinit();
    const scale = try ops.max(ctx, aa);
    defer scale.deinit();
    try scale.eval();
    const num = (try mx.dataSlice(f32))[0];
    const den = (try scale.dataSlice(f32))[0];
    return if (den > 0) @as(f64, num) / @as(f64, den) else @as(f64, num);
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    c.initErrorHandler();
    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    // Layout probe: confirm the packed-weight dtype/shape the kernel assumes
    // (uint32 words holding 8 nibbles each, row length K/8).
    {
        const w = try random.normal(ctx, &[_]i32{ 256, 512 }, .bfloat16, 0.0, 0.02, null);
        defer w.deinit();
        const qw = try quantize_mod.quantize(ctx, w, .{ .bits = 4, .group_size = 64, .mode = .affine });
        defer qw.data.deinit();
        defer qw.scales.deinit();
        defer qw.biases.deinit();
        std.debug.print("layout probe (w [256,512] 4b gs64):\n", .{});
        std.debug.print("  data   shape={any} dtype={s}\n", .{ qw.data.shape(), @tagName(qw.data.dtype()) });
        std.debug.print("  scales shape={any} dtype={s}\n", .{ qw.scales.shape(), @tagName(qw.scales.dtype()) });
        std.debug.print("  biases shape={any} dtype={s}\n\n", .{ qw.biases.shape(), @tagName(qw.biases.dtype()) });

        // Correctness on a tiny shape, with the generated source dumped.
        const xs = try random.normal(ctx, &[_]i32{ 2, 512 }, .bfloat16, 0.0, 0.02, null);
        defer xs.deinit();
        try xs.eval();
        verbose_once = std.c.getenv("DMLX_VK_VERBOSE") != null;
        const got = try splitkQmm(allocator, ctx, xs, qw, 2, 512, 256, 2);
        defer got.deinit();
        const ref = try quantize_mod.quantizedMatmul(ctx, xs, qw, true);
        defer ref.deinit();
        try got.eval();
        try ref.eval();
        std.debug.print("tiny check: reldiff={e:.3}\n", .{try relDiff(ctx, ref, got)});
        std.debug.print("  ref[0][0..4] = ", .{});
        const rs = try ops.astype(ctx, ref, .float32);
        defer rs.deinit();
        try rs.eval();
        for ((try rs.dataSlice(f32))[0..4]) |v| std.debug.print("{d: >10.4}", .{v});
        std.debug.print("\n  got[0][0..4] = ", .{});
        const gs2 = try ops.astype(ctx, got, .float32);
        defer gs2.deinit();
        try gs2.eval();
        for ((try gs2.dataSlice(f32))[0..4]) |v| std.debug.print("{d: >10.4}", .{v});
        std.debug.print("\n", .{});
        // Localize the mismatch: which (row, col) blocks disagree?
        const rv = try rs.dataSlice(f32);
        const gv = try gs2.dataSlice(f32);
        for ([_]usize{ 0, 1 }) |row| {
            var bad_first: ?usize = null;
            var bad_count: usize = 0;
            for (0..256) |col| {
                const i = row * 256 + col;
                const d = @abs(rv[i] - gv[i]);
                if (d > 1e-3 * @max(@abs(rv[i]), 1e-6)) {
                    if (bad_first == null) bad_first = col;
                    bad_count += 1;
                }
            }
            std.debug.print("  row {d}: {d}/256 cols mismatch", .{ row, bad_count });
            if (bad_first) |bf| {
                std.debug.print(", first at col {d} (ref={d:.5} got={d:.5})", .{
                    bf, rv[row * 256 + bf], gv[row * 256 + bf],
                });
            }
            std.debug.print("\n", .{});
        }
        std.debug.print("\n", .{});
    }

    const ms_list = [_]i32{ 2, 3, 4, 5, 6, 7, 8 };
    const kp_list = [_]i32{ 2, 4 };
    const reps: usize = 40;

    std.debug.print("split-K verify kernel vs stock MLX qmm (4-bit affine gs=64, bf16 x)\n", .{});
    std.debug.print("per-call kernel ms, single command buffer, {d} reps\n\n", .{reps});

    // Per-M totals over a whole 64-layer forward, for the round-cost model.
    var stock_total = [_]f64{0} ** ms_list.len;
    var best_total = [_]f64{0} ** ms_list.len;

    for (shapes) |s| {
        const w = try random.normal(ctx, &[_]i32{ s.out, s.in }, .bfloat16, 0.0, 0.02, null);
        defer w.deinit();
        const qw = try quantize_mod.quantize(ctx, w, .{ .bits = 4, .group_size = 64, .mode = .affine });
        defer qw.data.deinit();
        defer qw.scales.deinit();
        defer qw.biases.deinit();

        std.debug.print("{s}\n", .{s.name});
        std.debug.print("   {s: >3}{s: >10}{s: >10}{s: >10}{s: >10}{s: >9}{s: >11}{s: >11}\n", .{ "M", "stock", "kp=2", "kp=4", "sg8", "best x", "kp diff", "sg8 diff" });

        for (ms_list, 0..) |m, mi| {
            const x = try random.normal(ctx, &[_]i32{ m, s.in }, .bfloat16, 0.0, 0.02, null);
            defer x.deinit();
            try x.eval();

            // Warm both paths (JIT + PSO compile).
            for (0..3) |_| {
                const y = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
                defer y.deinit();
                try y.eval();
            }
            const stock_ms = try timeStock(allocator, ctx, x, qw, reps);

            var kp_ms = [_]f64{ 0, 0 };
            var diff: f64 = 0;
            var ok = true;
            for (kp_list, 0..) |kp, ki| {
                // Eligibility: K packs must split evenly across the simdgroups.
                if (@mod(@divExact(s.in, 8), kp) != 0 or @mod(s.out, 4) != 0 or 4 * m > 32) {
                    ok = false;
                    break;
                }
                for (0..3) |_| {
                    const y = splitkQmm(allocator, ctx, x, qw, m, s.in, s.out, kp) catch {
                        ok = false;
                        break;
                    };
                    defer y.deinit();
                    y.eval() catch {
                        ok = false;
                        break;
                    };
                }
                if (!ok) break;
                kp_ms[ki] = try timeSplitk(allocator, ctx, x, qw, m, s.in, s.out, kp, reps);
                if (ki == 0) {
                    const ref = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
                    defer ref.deinit();
                    const got = try splitkQmm(allocator, ctx, x, qw, m, s.in, s.out, kp);
                    defer got.deinit();
                    diff = try relDiff(ctx, ref, got);
                }
            }
            // sg8: 4..8 rows, its own geometry.
            var sg8_ms: f64 = 0;
            var sg8_diff: f64 = 0;
            var sg8_ok = sg8Eligible(m, s.in, s.out, 64);
            if (sg8_ok) {
                for (0..3) |_| {
                    const y = sg8Qmm(allocator, ctx, x, qw, m, s.in, s.out) catch {
                        sg8_ok = false;
                        break;
                    };
                    defer y.deinit();
                    y.eval() catch {
                        sg8_ok = false;
                        break;
                    };
                }
            }
            if (sg8_ok) {
                var pending = std.ArrayList(Array).empty;
                defer pending.deinit(allocator);
                defer for (pending.items) |y| y.deinit();
                for (0..reps) |_| {
                    const y = try sg8Qmm(allocator, ctx, x, qw, m, s.in, s.out);
                    try pending.append(allocator, y);
                }
                const t0 = std.c.mach_absolute_time();
                try eval_mod.asyncEval(allocator, pending.items);
                try pending.items[reps - 1].eval();
                sg8_ms = elapsedMs(t0) / @as(f64, @floatFromInt(reps));

                const ref = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
                defer ref.deinit();
                const got = try sg8Qmm(allocator, ctx, x, qw, m, s.in, s.out);
                defer got.deinit();
                sg8_diff = try relDiff(ctx, ref, got);
            }

            if (!ok and !sg8_ok) {
                std.debug.print("   {d: >3}{d: >10.4}{s: >10}{s: >10}{s: >10}{s: >9}{s: >11}{s: >11}\n", .{ m, stock_ms, "-", "-", "-", "inelig", "-", "-" });
                stock_total[mi] += stock_ms * @as(f64, @floatFromInt(s.per_token));
                best_total[mi] += stock_ms * @as(f64, @floatFromInt(s.per_token));
                continue;
            }
            // Only count a candidate as "best" if it is also numerically sane.
            var best = stock_ms;
            if (ok) best = @min(best, @min(kp_ms[0], kp_ms[1]));
            if (sg8_ok) best = @min(best, sg8_ms);
            stock_total[mi] += stock_ms * @as(f64, @floatFromInt(s.per_token));
            best_total[mi] += best * @as(f64, @floatFromInt(s.per_token));
            std.debug.print("   {d: >3}{d: >10.4}{d: >10.4}{d: >10.4}{d: >10.4}{d: >9.2}{e: >11.3}{e: >11.3}\n", .{
                m, stock_ms, kp_ms[0], kp_ms[1], sg8_ms, stock_ms / best, diff, sg8_diff,
            });
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("Summed over one 64-layer verify forward (best-of per shape, never worse than stock):\n", .{});
    std.debug.print("{s: <4}{s: >12}{s: >12}{s: >10}\n", .{ "M", "stock ms", "kernel ms", "speedup" });
    for (ms_list, 0..) |m, mi| {
        std.debug.print("{d: <4}{d: >12.2}{d: >12.2}{d: >10.2}\n", .{ m, stock_total[mi], best_total[mi], stock_total[mi] / best_total[mi] });
    }
    // What the round cost becomes, using the measured per-round structure:
    //   verify(L) = qmm(L) + overhead_v, draft(N) = 4.5 * N ms
    // with overhead_v taken from the measured verify totals (84.6 ms at L=3
    // minus 68.5 ms of qmm = 16.1; 102.7 - 84.6 = 18.1 at L=4).
    std.debug.print("\nProjected MTP round (draft 4.5ms/pos, verify overhead ~17ms, measured tokens/round):\n", .{});
    std.debug.print("{s: <4}{s: >10}{s: >12}{s: >12}{s: >11}{s: >11}\n", .{ "N", "tok/round", "now ms", "kernel ms", "now tok/s", "new tok/s" });
    // tokens/round: measured at N=2,3,4; N=5..7 extrapolated from the
    // measured per-depth conditional acceptance (p decays, so these are
    // optimistic-but-bounded and marked as such below).
    const toks = [_]f64{ 2.54, 2.91, 3.28, 3.55, 3.76, 3.92 };
    for ([_]usize{ 2, 3, 4, 5, 6, 7 }, 0..) |n, i| {
        const mi = n - 1; // M = N+1 -> index in ms_list (2,3,4,5,6)
        const draft = 4.5 * @as(f64, @floatFromInt(n));
        const ov = 17.0;
        const now = draft + stock_total[mi] + ov;
        const new = draft + best_total[mi] + ov;
        std.debug.print("{d: <4}{d: >10.2}{d: >12.1}{d: >12.1}{d: >11.1}{d: >11.1}\n", .{
            n, toks[i], now, new, 1000.0 * toks[i] / now, 1000.0 * toks[i] / new,
        });
    }
    std.debug.print("\nReference: mlx-vlm 31.1 tok/s (code, its own basis). Gate: >= 1.3x summed at M=3..5.\n", .{});
    std.debug.print("NOTE: lm_head loses on split-K (omlx uses its `msg` morphology there,\n", .{});
    std.debug.print("reporting 2.6-3.3x at M=3-4) — that shape is still counted at stock cost here.\n", .{});
}
