// Scratch micro-benchmark: does decode-shape qmm time scale with weight bit
// width? Decode is pinned at ~97% of the DRAM roofline (14.4 GB of packed
// 4-bit weights per token), so no kernel work can beat mlx-lm/omlx — the only
// structural win is reading FEWER BYTES. This measures the actual payoff
// curve of bits in {3,4,5,6,8} at the real Qwen3.8-27B decode shapes (M=1),
// which sets the ceiling for any mixed-precision plan.
//
// Not committed. Run: zig build run-bench-bits -Doptimize=ReleaseFast
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const random = @import("mlx").random;
const quantize_mod = @import("mlx").quantize;
const eval_mod = @import("mlx").eval;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

fn elapsedMs(start: u64) f64 {
    const ns = (std.c.mach_absolute_time() - start) * 125 / 3;
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

const Shape = struct {
    name: []const u8,
    out: i32,
    in: i32,
    per_token: u32,
};

const shapes = [_]Shape{
    .{ .name = "gdn_qkv", .out = 10240, .in = 5120, .per_token = 48 },
    .{ .name = "gdn_z", .out = 6144, .in = 5120, .per_token = 48 },
    .{ .name = "gdn_out", .out = 5120, .in = 6144, .per_token = 48 },
    .{ .name = "attn_q", .out = 12288, .in = 5120, .per_token = 16 },
    .{ .name = "attn_kv", .out = 1024, .in = 5120, .per_token = 32 },
    .{ .name = "attn_o", .out = 5120, .in = 6144, .per_token = 16 },
    .{ .name = "mlp_gate", .out = 17408, .in = 5120, .per_token = 64 },
    .{ .name = "mlp_up", .out = 17408, .in = 5120, .per_token = 64 },
    .{ .name = "mlp_down", .out = 5120, .in = 17408, .per_token = 64 },
    .{ .name = "lm_head", .out = 248320, .in = 5120, .per_token = 1 },
};

const Config = struct { bits: i32, gs: i32 };
const configs = [_]Config{
    .{ .bits = 2, .gs = 64 },
    .{ .bits = 3, .gs = 128 },
    .{ .bits = 3, .gs = 64 },
    .{ .bits = 4, .gs = 128 },
    .{ .bits = 4, .gs = 64 }, // current production config
    .{ .bits = 5, .gs = 64 },
    .{ .bits = 6, .gs = 64 },
    .{ .bits = 8, .gs = 64 },
};

fn kernelMs(allocator: std.mem.Allocator, ctx: EagerContext, x: Array, qw: anytype, n: usize) !f64 {
    var pending = std.ArrayList(Array).empty;
    defer pending.deinit(allocator);
    defer for (pending.items) |y| y.deinit();
    for (0..n) |_| {
        const y = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
        try pending.append(allocator, y);
    }
    const t = std.c.mach_absolute_time();
    try eval_mod.asyncEval(allocator, pending.items);
    try pending.items[n - 1].eval();
    return elapsedMs(t) / @as(f64, @floatFromInt(n));
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    c.initErrorHandler();
    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    std.debug.print("Decode-shape qmm (M=1) vs weight bit width — Qwen3.8-27B, 64 layers\n", .{});
    std.debug.print("bytes/token counts packed weights + scales(+biases) only\n\n", .{});
    std.debug.print("{s: <10}{s: >10}{s: >12}{s: >12}{s: >12}{s: >11}\n", .{ "config", "qmm ms", "GB/token", "GB/s", "vs 4b/gs64", "decode t/s" });

    var baseline_ms: f64 = 0;
    for (configs) |cfg| {
        var total_ms: f64 = 0;
        var total_gb: f64 = 0;
        for (shapes) |s| {
            const w = try random.normal(ctx, &[_]i32{ s.out, s.in }, .bfloat16, 0.0, 0.02, null);
            defer w.deinit();
            const qw = quantize_mod.quantize(ctx, w, .{
                .bits = @intCast(cfg.bits),
                .group_size = @intCast(cfg.gs),
                .mode = .affine,
            }) catch {
                std.debug.print("{s: <10}  quantize failed (bits={d} gs={d})\n", .{ "", cfg.bits, cfg.gs });
                break;
            };
            defer qw.data.deinit();
            defer qw.scales.deinit();
            defer qw.biases.deinit();
            const x = try random.normal(ctx, &[_]i32{ 1, s.in }, .bfloat16, 0.0, 0.02, null);
            defer x.deinit();
            try x.eval();
            for (0..5) |_| {
                const y = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
                defer y.deinit();
                try y.eval();
            }
            const ms = try kernelMs(allocator, ctx, x, qw, 50);
            total_ms += ms * @as(f64, @floatFromInt(s.per_token));

            const elems = @as(f64, @floatFromInt(s.out)) * @as(f64, @floatFromInt(s.in));
            const w_bytes = elems * @as(f64, @floatFromInt(cfg.bits)) / 8.0;
            // affine: one f32 scale + one f32 bias per group (mlx stores both)
            const groups = elems / @as(f64, @floatFromInt(cfg.gs));
            const sb_bytes = groups * 8.0;
            total_gb += (w_bytes + sb_bytes) / 1e9 * @as(f64, @floatFromInt(s.per_token));
        }
        if (cfg.bits == 4 and cfg.gs == 64) baseline_ms = total_ms;
        var label: [16]u8 = undefined;
        const lbl = try std.fmt.bufPrint(&label, "{d}b/gs{d}", .{ cfg.bits, cfg.gs });
        // Non-qmm per-token overhead measured at ~13 ms (72.5 ms decode step
        // minus 59.4 ms of qmm at 4b/gs64); assume it is bits-invariant.
        const overhead_ms = 13.0;
        std.debug.print("{s: <10}{d: >10.2}{d: >12.3}{d: >12.1}{d: >12.2}{d: >11.1}\n", .{
            lbl,
            total_ms,
            total_gb,
            total_gb / (total_ms / 1e3),
            if (baseline_ms > 0) total_ms / baseline_ms else 0.0,
            1000.0 / (total_ms + overhead_ms),
        });
    }
    std.debug.print("\nReference: dmlx decode 14.0 t/s, mlx-lm 15.6, omlx 15.42.\n", .{});
}
