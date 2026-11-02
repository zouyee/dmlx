// omlx affine_q4_qmm_t variant-8 tile (bm=64, bk=32, bn=64) as an mlx-c fast
// metal kernel, for prefill-shape quantized matmuls (M >= min_m). The kernel
// source is scripts/flatten_qmm_v8.py output: the MLX steel GEMM header
// closure + omlx's vendored affine_quantized.h, flattened (mlx-c's JIT cannot
// include mlx/backend/metal/kernels/steel/* itself).
//
// Isolated at M=2048 on M4 Pro (bench_prefill_v8): uniform ~1.06x over stock
// mx.quantized_matmul across all seven Qwen3.8-27B projection shapes, worth
// ~0.85 s of the 17.5 s prefill wall. Bit-exact vs stock (same qmm_t_impl,
// only the tile shape differs).
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;

const Array = array_mod.Array;

const header_src = @embedFile("qwen38_qmm_v8_steel.metal");

const body_src =
    \\    const uint3 tid = threadgroup_position_in_grid;
    \\    const uint lid = thread_index_in_threadgroup;
    \\    threadgroup {[t]s} Xs[64 * 40];
    \\    threadgroup {[t]s} Ws[64 * 40];
    \\    dmlx_qmm_v8_impl<{[t]s}>(w, scales, biases, x, y, Xs, Ws, {[k]}, {[n]}, {[m]}, tid, lid, lid / 32, lid % 32);
    \\
;

pub const Key = struct {
    m: i32,
    n: i32,
    k: i32,
    bf16: bool,
};

const Entry = struct {
    key: Key,
    kernel: c.c.mlx_fast_metal_kernel,
    broken: bool = false,
};

var entries: [16]?Entry = .{null} ** 16;
var warned = false;

fn warn(err: anyerror) void {
    if (!warned) {
        warned = true;
        std.log.warn("prefill qmm v8 kernel unavailable ({s}); using stock quantized_matmul", .{@errorName(err)});
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

fn createKernel(allocator: std.mem.Allocator, key: Key) !c.c.mlx_fast_metal_kernel {
    const name_z = try std.fmt.allocPrintSentinel(allocator, "qmm_v8_{s}_{d}_{d}_{d}", .{ if (key.bf16) "bf16" else "f16", key.m, key.n, key.k }, 0);
    defer allocator.free(name_z);
    const source_z = try std.fmt.allocPrintSentinel(allocator, body_src, .{
        .t = if (key.bf16) "bfloat16_t" else "float16_t",
        .k = key.k,
        .n = key.n,
        .m = key.m,
    }, 0);
    defer allocator.free(source_z);
    const header_z = try allocator.dupeZ(u8, header_src ++ "\n");
    defer allocator.free(header_z);

    const in_vec = try makeStringVec(allocator, &.{ "w", "scales", "biases", "x" });
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = try makeStringVec(allocator, &.{"y"});
    defer _ = c.c.mlx_vector_string_free(out_vec);

    return c.c.mlx_fast_metal_kernel_new(name_z.ptr, in_vec, out_vec, source_z.ptr, header_z.ptr, true, false);
}

fn getKernel(allocator: std.mem.Allocator, key: Key) ?c.c.mlx_fast_metal_kernel {
    for (&entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) return if (e.broken) null else e.kernel;
        }
    }
    const kernel = createKernel(allocator, key) catch |err| {
        warn(err);
        return null;
    };
    for (&entries) |*slot| {
        if (slot.* == null) {
            slot.* = .{ .key = key, .kernel = kernel };
            return kernel;
        }
    }
    return null;
}

fn markBroken(key: Key) void {
    for (&entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) e.broken = true;
        }
    }
}

/// Variant-8 quantized matmul: y = x @ dequant(w).T at [...,K]x[N,K], q4 gs64.
/// Returns null when the shape/dtype is unsupported or the kernel failed
/// (caller falls back to stock quantizedMatmul). w is the packed [N, K/8]
/// uint32; scales/biases [N, K/64]; x [..., K] bf16/f16 row-contiguous.
pub fn apply(
    ctx: anytype,
    x: Array,
    w: Array,
    scales: Array,
    biases: Array,
) !?Array {
    const nd = x.ndim();
    if (nd < 2) return null;
    const dt = x.dtype();
    const bf16 = dt == .bfloat16;
    if (!bf16 and dt != .float16) return null;
    const xs = x.shape();
    const k: i32 = @intCast(xs[nd - 1]);
    const m: i32 = @intCast(@divTrunc(x.size(), @as(usize, @intCast(k))));
    if (w.ndim() != 2 or w.dtype() != .uint32) return null;
    if (scales.ndim() != 2 or scales.dtype() != dt) return null;
    // An unset/empty biases handle (callers without affine biases) has no
    // valid dtype; check size before touching it.
    if (biases.size() == 0 or biases.dtype() != dt) return null;
    const n: i32 = @intCast(w.shape()[0]);
    if (@mod(k, 64) != 0 or @mod(n, 64) != 0) return null;
    if (scales.shape()[0] != n or scales.shape()[1] != @divTrunc(k, 64)) return null;
    if (biases.shape()[0] != n or biases.shape()[1] != @divTrunc(k, 64)) return null;
    // v8 tile: bm=64, bk=32, bn=64 — K % 32 and N % 64 required (checked above);
    // the impl handles a partial M edge at runtime.
    const key = Key{ .m = m, .n = n, .k = k, .bf16 = bf16 };
    const kernel = getKernel(ctx.allocator, key) orelse return null;

    const cfg = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg);
    const tg_n = @divTrunc(n + 63, 64);
    const tg_m = @divTrunc(m + 63, 64);
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, tg_n * 32, tg_m * 2, 2));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 2, 2));
    var out_shape_buf: [8]i32 = undefined;
    const out_dims = xs.len;
    if (out_dims > out_shape_buf.len) return null;
    for (xs[0 .. nd - 1], 0..) |d, i| out_shape_buf[i] = @intCast(d);
    out_shape_buf[nd - 1] = n;
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape_buf, out_dims, if (bf16) c.c.MLX_BFLOAT16 else c.c.MLX_FLOAT16));

    const input_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(input_vec);
    for ([_]Array{ w, scales, biases, x }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(input_vec, inp.inner));
    }
    var output_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(output_vec);
    c.check(c.c.mlx_fast_metal_kernel_apply(&output_vec, kernel, input_vec, cfg, ctx.stream.inner)) catch |err| {
        markBroken(key);
        warn(err);
        return null;
    };

    var y_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&y_handle, output_vec, 0));
    return Array.fromHandle(y_handle);
}

var route_enabled: ?bool = null;
var route_min_m: i32 = 256;

fn routeEnabled() bool {
    if (route_enabled == null) {
        if (std.c.getenv("DMLX_PREFILL_QMM_V8")) |v| {
            route_enabled = !(v[0] == '0' and v[1] == 0);
        } else route_enabled = true;
        if (std.c.getenv("DMLX_PREFILL_QMM_V8_MIN_M")) |v| {
            route_min_m = std.fmt.parseInt(i32, std.mem.sliceTo(v, 0), 10) catch 256;
        }
    }
    return route_enabled.?;
}

/// Routing gate for the prefill path: v8 wins ~1.04x from M=256 up on every
/// Qwen3.8 projection shape and is flat at M=128 (bench_prefill_v8 + shape
/// probe on M4 Pro, mlx 0.32.4). Below that, and for other bit widths / group
/// sizes, fall through to stock quantizedMatmul.
pub fn maybeApply(ctx: anytype, x: Array, w: Array, scales: Array, biases: ?Array) !?Array {
    if (!routeEnabled()) return null;
    const b = biases orelse return null;
    const nd = x.ndim();
    if (nd < 2) return null;
    const k = xs_last(x);
    if (@divTrunc(@as(i64, @intCast(x.size())), k) < route_min_m) return null;
    return apply(ctx, x, w, scales, b);
}

fn xs_last(x: Array) i64 {
    const xs = x.shape();
    return @intCast(xs[xs.len - 1]);
}
