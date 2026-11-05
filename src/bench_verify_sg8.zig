//! Per-shape bandwidth bench for the PRODUCTION verify_qmm routing
//! (sg8 / split-K as dispatched by `maybeQmm`), at the MTP verify row
//! counts M = 4..8. Prints ms/call and effective GB/s (weight bytes /
//! call time) per shape, stock vs routed, so kernel-geometry iterations
//! have a fast loop that does not need a model load or a server.
//!
//! Run: zig build run-bench-verify-sg8 -Doptimize=ReleaseFast
//! Geometry overrides: DMLX_VERIFY_QMM_SG8_CT / _NSG / _SG8_MIN_M.

const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const random = @import("mlx").random;
const quantize_mod = @import("mlx").quantize;
const eval_mod = @import("mlx").eval;

const verify_qmm = @import("models/verify_qmm.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

fn elapsedMs(start: u64) f64 {
    const ns = (std.c.mach_absolute_time() - start) * 125 / 3;
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

const Shape = struct { name: []const u8, n: i32, k: i32, per_token: i32 };

// The verify-shape set of Qwen3.8-27B (same list as bench_verify_kernel.zig).
const shapes = [_]Shape{
    .{ .name = "gdn_qkv  [10240,5120]", .n = 10240, .k = 5120, .per_token = 48 },
    .{ .name = "gdn_z    [6144,5120]", .n = 6144, .k = 5120, .per_token = 48 },
    .{ .name = "gdn_out  [5120,6144]", .n = 5120, .k = 6144, .per_token = 48 },
    .{ .name = "attn_q   [12288,5120]", .n = 12288, .k = 5120, .per_token = 16 },
    .{ .name = "attn_o   [5120,6144]", .n = 5120, .k = 6144, .per_token = 16 },
    .{ .name = "attn_kv  [1024,5120]", .n = 1024, .k = 5120, .per_token = 32 },
    .{ .name = "mlp_gate [17408,5120]", .n = 17408, .k = 5120, .per_token = 64 },
    .{ .name = "mlp_up   [17408,5120]", .n = 17408, .k = 5120, .per_token = 64 },
    .{ .name = "mlp_down [5120,17408]", .n = 5120, .k = 17408, .per_token = 64 },
    .{ .name = "lm_head  [248320,5120]", .n = 248320, .k = 5120, .per_token = 1 },
};

fn timeCalls(reps: usize, pending: *std.ArrayList(Array), allocator: std.mem.Allocator, call: *const fn (ctx: EagerContext, x: Array, qw: quantize_mod.QuantizedWeight) anyerror!Array, ctx: EagerContext, x: Array, qw: quantize_mod.QuantizedWeight) !f64 {
    pending.clearRetainingCapacity();
    for (0..reps) |_| {
        try pending.append(allocator, try call(ctx, x, qw));
    }
    const t = std.c.mach_absolute_time();
    try eval_mod.asyncEval(allocator, pending.items);
    try pending.items[pending.items.len - 1].eval();
    return elapsedMs(t) / @as(f64, @floatFromInt(reps));
}

fn stockCall(ctx: EagerContext, x: Array, qw: quantize_mod.QuantizedWeight) anyerror!Array {
    return quantize_mod.quantizedMatmul(ctx, x, qw, true);
}

fn routedCall(ctx: EagerContext, x: Array, qw: quantize_mod.QuantizedWeight) anyerror!Array {
    return (try verify_qmm.maybeQmm(ctx, x, qw)) orelse error.NotRouted;
}

fn relDiff(ctx: EagerContext, a: Array, b: Array) !f64 {
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

    verify_qmm.arm();
    if (!verify_qmm.isArmed()) return error.ArmFailed;

    const ms_list = [_]i32{ 4, 6, 8 };
    const reps = 50;
    const cfg = quantize_mod.QuantConfig{ .group_size = 64, .bits = 4 };

    std.debug.print("{s:24} {s:>3} {s:>10} {s:>10} {s:>9} {s:>9} {s:>9}\n", .{ "shape", "M", "stock ms", "routed ms", "stock GB/s", "routed GB/s", "rel diff" });
    var pending = std.ArrayList(Array).empty;
    defer pending.deinit(allocator);
    defer for (pending.items) |y| y.deinit();

    for (shapes) |sh| {
        const w = try random.normal(ctx, &[_]i32{ sh.n, sh.k }, .bfloat16, 0.0, 0.02, null);
        defer w.deinit();
        try w.eval();
        const qw = try quantize_mod.quantize(ctx, w, cfg);
        defer qw.deinit(allocator);
        const weight_bytes = @as(f64, @floatFromInt(@as(u64, @intCast(sh.n)) * @as(u64, @intCast(sh.k)))) * 0.5 * 1.06; // 4-bit + scales/biases overhead ~6%

        for (ms_list) |m| {
            const x = try random.normal(ctx, &[_]i32{ m, sh.k }, .bfloat16, 0.0, 0.02, null);
            defer x.deinit();
            try x.eval();

            // Warm BOTH paths before timing: Metal kernel compilation is
            // lazy (charged at eval), so without a warm call the first
            // timed batch eats a ~40 ms compile, ~0.8 ms/call at reps=50.
            {
                const w1 = try stockCall(ctx, x, qw);
                defer w1.deinit();
                try w1.eval();
            }
            {
                const w2 = routedCall(ctx, x, qw) catch null;
                if (w2) |w_| {
                    defer w_.deinit();
                    try w_.eval();
                }
            }

            const stock_ms = timeCalls(reps, &pending, allocator, stockCall, ctx, x, qw) catch unreachable;
            const routed_ms = timeCalls(reps, &pending, allocator, routedCall, ctx, x, qw) catch {
                std.debug.print("{s:24} {d:>3} {d:>10.4} {s:>10}\n", .{ sh.name, m, stock_ms, "stock" });
                continue;
            };
            // Correctness: one fresh pair, relative max-abs diff.
            const ys = try stockCall(ctx, x, qw);
            defer ys.deinit();
            const yr = try routedCall(ctx, x, qw);
            defer yr.deinit();
            const rd = try relDiff(ctx, ys, yr);

            std.debug.print("{s:24} {d:>3} {d:>10.4} {d:>10.4} {d:>9.0} {d:>9.0} {e:>9.1}\n", .{
                sh.name,                       m,                              stock_ms, routed_ms,
                weight_bytes / stock_ms / 1e6, weight_bytes / routed_ms / 1e6, rd,
            });
        }
    }
    verify_qmm.disarm();
}
