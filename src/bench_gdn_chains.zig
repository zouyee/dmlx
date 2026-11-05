// Scratch: GDN tail chain and conv chain isolated cost at L=1901.
// Run: zig build run-bench-gdn-chains -Doptimize=ReleaseFast
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const random = @import("mlx").random;
const fast_mod = @import("mlx").fast;
const gdn = @import("models/qwen38_gdn.zig");
const activations = @import("mlx").activations;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

fn elapsedMs(start: u64) f64 {
    const ns = (std.c.mach_absolute_time() - start) * 125 / 3;
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    c.initErrorHandler();
    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    const L: i32 = 1901;
    const HV: i32 = 48;
    const D: i32 = 128;

    const y_f32 = try random.normal(ctx, &[_]i32{ L, HV, D }, .float32, 0.0, 1.0, null);
    defer y_f32.deinit();
    const z = try random.normal(ctx, &[_]i32{ 1, L, HV * D }, .bfloat16, 0.0, 1.0, null);
    defer z.deinit();
    const norm_w = try random.normal(ctx, &[_]i32{D}, .float32, 1.0, 0.1, null);
    defer norm_w.deinit();
    try y_f32.eval();
    try z.eval();
    try norm_w.eval();

    // --- tail chain (mirrors prefillTail) ---
    for (0..2) |_| {
        const y = try ops.astype(ctx, y_f32, .bfloat16);
        defer y.deinit();
        const yn = try fast_mod.rmsNorm(ctx, y, norm_w, 1e-6);
        defer yn.deinit();
        const yn32 = try ops.astype(ctx, yn, .float32);
        defer yn32.deinit();
        const zr = try ops.reshape(ctx, z, &[_]i32{ L, HV, D });
        defer zr.deinit();
        const zf = try ops.astype(ctx, zr, .float32);
        defer zf.deinit();
        const za = try activations.silu(ctx, zf);
        defer za.deinit();
        const g = try ops.multiply(ctx, za, yn32);
        defer g.deinit();
        const gb = try ops.astype(ctx, g, .bfloat16);
        defer gb.deinit();
        try gb.eval();
    }
    var t = std.c.mach_absolute_time();
    const REPS = 5;
    for (0..REPS) |_| {
        const y = try ops.astype(ctx, y_f32, .bfloat16);
        defer y.deinit();
        const yn = try fast_mod.rmsNorm(ctx, y, norm_w, 1e-6);
        defer yn.deinit();
        const yn32 = try ops.astype(ctx, yn, .float32);
        defer yn32.deinit();
        const zr = try ops.reshape(ctx, z, &[_]i32{ L, HV, D });
        defer zr.deinit();
        const zf = try ops.astype(ctx, zr, .float32);
        defer zf.deinit();
        const za = try activations.silu(ctx, zf);
        defer za.deinit();
        const g = try ops.multiply(ctx, za, yn32);
        defer g.deinit();
        const gb = try ops.astype(ctx, g, .bfloat16);
        defer gb.deinit();
        try gb.eval();
    }
    std.debug.print("tail chain: {d:.3} ms/layer  ({d:.0} ms / 48 layers)\n", .{ elapsedMs(t) / REPS, elapsedMs(t) / REPS * 48 });

    // --- bf16 variant: cast once, norm in bf16, silu+mul in bf16 ---
    t = std.c.mach_absolute_time();
    for (0..REPS) |_| {
        const y = try ops.astype(ctx, y_f32, .bfloat16);
        defer y.deinit();
        const yn = try fast_mod.rmsNorm(ctx, y, norm_w, 1e-6);
        defer yn.deinit();
        const zr = try ops.reshape(ctx, z, &[_]i32{ L, HV, D });
        defer zr.deinit();
        const za = try activations.silu(ctx, zr);
        defer za.deinit();
        const g = try ops.multiply(ctx, za, yn);
        defer g.deinit();
        try g.eval();
    }
    std.debug.print("tail chain bf16: {d:.3} ms/layer  ({d:.0} ms / 48 layers)\n", .{ elapsedMs(t) / REPS, elapsedMs(t) / REPS * 48 });

    // --- fused causal conv + silu over [L, 10240] f32 ---
    const QKV: i32 = 10240;
    const K: i32 = 4;
    const x_f32 = try random.normal(ctx, &[_]i32{ L, QKV }, .float32, 0.0, 1.0, null);
    defer x_f32.deinit();
    const conv_state = try random.normal(ctx, &[_]i32{ K - 1, QKV }, .float32, 0.0, 1.0, null);
    defer conv_state.deinit();
    const conv_w = try random.normal(ctx, &[_]i32{ QKV, K }, .float32, 0.0, 0.5, null);
    defer conv_w.deinit();
    try x_f32.eval();
    try conv_state.eval();
    try conv_w.eval();
    for (0..2) |_| {
        const y = (try gdn.testConvAccumSeqFused(ctx, conv_w, conv_state, x_f32, L, QKV, K)).?;
        defer y.deinit();
        try y.eval();
    }
    t = std.c.mach_absolute_time();
    for (0..REPS) |_| {
        const y = (try gdn.testConvAccumSeqFused(ctx, conv_w, conv_state, x_f32, L, QKV, K)).?;
        defer y.deinit();
        try y.eval();
    }
    std.debug.print("conv fused+silu: {d:.3} ms/layer  ({d:.0} ms / 48 layers)\n", .{ elapsedMs(t) / REPS, elapsedMs(t) / REPS * 48 });

    // --- fused prefill tail: out-gate norm + silu(z) gate, one kernel ---
    {
        const y2 = try random.normal(ctx, &[_]i32{ L, HV, D }, .float32, 0.0, 1.0, null);
        defer y2.deinit();
        const z2 = try random.normal(ctx, &[_]i32{ 1, L, HV * D }, .bfloat16, 0.0, 1.0, null);
        defer z2.deinit();
        try y2.eval();
        try z2.eval();
        for (0..2) |_| {
            const o = (try gdn.testPrefillTailFused(ctx, y2, z2, norm_w, 1e-6, L, HV, D)).?;
            defer o.deinit();
            try o.eval();
        }
        t = std.c.mach_absolute_time();
        for (0..REPS) |_| {
            const o = (try gdn.testPrefillTailFused(ctx, y2, z2, norm_w, 1e-6, L, HV, D)).?;
            defer o.deinit();
            try o.eval();
        }
        std.debug.print("prefill tail fused: {d:.3} ms/layer  ({d:.0} ms / 48 layers)\n", .{ elapsedMs(t) / REPS, elapsedMs(t) / REPS * 48 });
    }
}
