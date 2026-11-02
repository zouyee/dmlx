// Does the NUMBER of distinct weight buffers change achieved streaming
// bandwidth?
//
// Why this matters (plan doc §31.4 / §33): plain decode must stream 14.42 GB
// of weights per token. mlx-lm does it in ~61.7 ms (234 GB/s) and we need
// ~69.4 ms (208 GB/s) — a 13% difference with the SAME checkpoint and the
// same MLX qmm kernels. Every op-level explanation has been ablated to zero:
// dispatch count, dependency depth, qmm call count, GDN tail, attention
// non-qmm, CPU graph build (§30, §31).
//
// The one structural difference left is HOW the weights are resident:
//
//   dmlx : `TensorIndex.loadTensor` preads each tensor into a heap buffer and
//          calls `mlx_array_new_data`, which COPIES into a fresh MLX
//          allocation -> ~2180 independent Metal buffers.
//   mlx-lm: `mx.load` on a safetensors file, which MLX implements as an mmap
//          with zero-copy array views -> 3 contiguous file-backed mappings.
//
// So this bench holds total bytes and kernel constant and varies only the
// buffer count: stream the same ~1 GB of 4-bit weights as 1 big buffer vs
// N slices-of-one vs N separate allocations, and report GB/s.
//
// Run: zig build run-bench-weight-layout -Doptimize=ReleaseFast
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

/// One decode-shaped projection: [1, K] @ dequant(W)^T with W = [N, K].
const K: i32 = 5120;
const N_ROWS: i32 = 17408; // mlx_gate/up shape in this model
const CHUNKS: i32 = 32; // 32 x ~29.5 MB ~= 944 MB, well past any cache
const REPS: usize = 3;

fn qbytes(rows: i32, cols: i32, gs: i32) f64 {
    // 4-bit packed weight + bf16 scales + bf16 biases
    const w = @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(cols)) / 2.0;
    const groups = @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(@divExact(cols, gs)));
    return w + groups * 2.0 * 2.0;
}

const Variant = enum { one_buffer, slices_of_one, separate_allocs };

fn run(allocator: std.mem.Allocator, ctx: EagerContext, variant: Variant) !f64 {
    const gs: i32 = 64;
    const cfg = quantize_mod.QuantConfig{ .bits = 4, .group_size = gs, .mode = .affine };

    var weights = std.ArrayList(quantize_mod.QuantizedWeight).empty;
    defer {
        for (weights.items) |qw| {
            qw.data.deinit();
            qw.scales.deinit();
            qw.biases.deinit();
        }
        weights.deinit(allocator);
    }

    switch (variant) {
        .separate_allocs => {
            // CHUNKS independent quantize() calls -> CHUNKS distinct buffers,
            // which is the shape of our loader's output.
            var i: i32 = 0;
            while (i < CHUNKS) : (i += 1) {
                const dense = try random.normal(ctx, &[_]i32{ N_ROWS, K }, .bfloat16, 0.0, 1.0, null);
                defer dense.deinit();
                const qw = try quantize_mod.quantize(ctx, dense, cfg);
                try qw.data.eval();
                try qw.scales.eval();
                try qw.biases.eval();
                try weights.append(allocator, qw);
            }
        },
        .one_buffer, .slices_of_one => {
            // ONE allocation holding all CHUNKS worth of rows; the slices are
            // contiguous row ranges, i.e. views into a single buffer.
            const dense = try random.normal(ctx, &[_]i32{ N_ROWS * CHUNKS, K }, .bfloat16, 0.0, 1.0, null);
            defer dense.deinit();
            const big = try quantize_mod.quantize(ctx, dense, cfg);
            try big.data.eval();
            try big.scales.eval();
            try big.biases.eval();
            const wcols: i32 = big.data.shape()[1];
            const scols: i32 = big.scales.shape()[1];
            const parts: i32 = if (variant == .one_buffer) 1 else CHUNKS;
            const rows_per: i32 = @divExact(N_ROWS * CHUNKS, parts);
            var i: i32 = 0;
            while (i < parts) : (i += 1) {
                const r0 = i * rows_per;
                const r1 = r0 + rows_per;
                const w = try ops.slice(ctx, big.data, &[_]i32{ r0, 0 }, &[_]i32{ r1, wcols }, &[_]i32{ 1, 1 });
                const s = try ops.slice(ctx, big.scales, &[_]i32{ r0, 0 }, &[_]i32{ r1, scols }, &[_]i32{ 1, 1 });
                const b = try ops.slice(ctx, big.biases, &[_]i32{ r0, 0 }, &[_]i32{ r1, scols }, &[_]i32{ 1, 1 });
                try weights.append(allocator, .{
                    .data = w,
                    .scales = s,
                    .biases = b,
                    .config = cfg,
                    .original_shape = &[_]i32{},
                });
            }
            big.data.deinit();
            big.scales.deinit();
            big.biases.deinit();
        },
    }

    const x = try random.normal(ctx, &[_]i32{ 1, K }, .bfloat16, 0.0, 1.0, null);
    defer x.deinit();
    try x.eval();

    // Warm.
    for (weights.items) |qw| {
        const y = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
        defer y.deinit();
        try y.eval();
    }

    var best: f64 = std.math.floatMax(f64);
    for (0..REPS) |_| {
        const t0 = std.c.mach_absolute_time();
        // Chain so nothing is pruned and the sequence stays serial, exactly
        // like a real forward's dependency structure.
        var acc: ?Array = null;
        for (weights.items) |qw| {
            const y = try quantize_mod.quantizedMatmul(ctx, x, qw, true);
            if (acc) |a| {
                const n = try ops.add(ctx, a, y);
                a.deinit();
                y.deinit();
                acc = n;
            } else {
                acc = y;
            }
        }
        if (acc) |a| {
            try a.eval();
            a.deinit();
        }
        const ms = elapsedMs(t0);
        if (ms < best) best = ms;
    }
    return best;
}

pub fn main() !void {
    c.initErrorHandler();
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const ctx = EagerContext.init(allocator);

    const total_bytes = qbytes(N_ROWS * CHUNKS, K, 64);
    std.debug.print(
        \\Weight-buffer-layout bandwidth test
        \\  per-call shape [{d}, {d}] 4-bit gs64, {d} calls, {d:.2} GB total
        \\  (same bytes and same kernel in every row; only the number of
        \\   distinct Metal buffers changes)
        \\
    , .{ N_ROWS, K, CHUNKS, total_bytes / 1e9 });

    const rows = [_]struct { v: Variant, label: []const u8 }{
        .{ .v = .one_buffer, .label = "1 buffer, 1 big qmm" },
        .{ .v = .slices_of_one, .label = "1 buffer, 32 slice views" },
        .{ .v = .separate_allocs, .label = "32 separate allocations" },
    };
    std.debug.print("\n  {s: <28} {s: >10} {s: >12}\n", .{ "layout", "ms", "GB/s" });
    for (rows) |r| {
        const ms = try run(allocator, ctx, r.v);
        std.debug.print("  {s: <28} {d: >10.2} {d: >12.1}\n", .{ r.label, ms, total_bytes / ms / 1e6 });
    }
    std.debug.print(
        \\
        \\If "32 separate allocations" is materially slower than "32 slice
        \\views" at identical bytes, our per-tensor `mlx_array_new_data` copy
        \\(~2180 buffers) is the remaining plain-decode gap and the fix is to
        \\load into one arena or mmap zero-copy like mx.load does.
        \\
    , .{});
}
