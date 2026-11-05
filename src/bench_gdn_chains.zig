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

    // --- synthetic GDN layer, section by section at serving shapes --------
    // Sum of parts vs the serving per-layer cost (DMLX_LAYER_PROF). One
    // layer's weights, L rows, v8 qmm for projections/out_proj/MLP.
    const quantize_mod = @import("mlx").quantize;
    const qmm_v8 = @import("models/qwen38_qmm_v8.zig");
    const HID: i32 = 5120;
    const INTER: i32 = 17408;
    const HK: i32 = 16;
    const QKV2: i32 = (2 * HK + HV) * D;

    const mkQ4 = struct {
        fn f(cx: EagerContext, out: i32, in: i32) !quantize_mod.QuantizedWeight {
            const w = try random.normal(cx, &[_]i32{ out, in }, .bfloat16, 0.0, 0.02, null);
            defer w.deinit();
            return quantize_mod.quantize(cx, w, .{ .bits = 4, .group_size = 64, .mode = .affine });
        }
    }.f;
    const qkv_w = try mkQ4(ctx, QKV2, HID);
    const z_w = try mkQ4(ctx, HV * D, HID);
    const a_w = try mkQ4(ctx, HV, HID);
    const b_w = try mkQ4(ctx, HV, HID);
    const o_w = try mkQ4(ctx, HID, HV * D);
    const gate_w = try mkQ4(ctx, INTER, HID);
    const up_w = try mkQ4(ctx, INTER, HID);
    const down_w = try mkQ4(ctx, HID, INTER);

    const x_in = try random.normal(ctx, &[_]i32{ 1, L, HID }, .bfloat16, 0.0, 1.0, null);
    defer x_in.deinit();
    try x_in.eval();

    const qmmV8 = struct {
        fn f(cx: EagerContext, x: Array, qw: quantize_mod.QuantizedWeight) !Array {
            return (try qmm_v8.maybeApply(cx, x, qw.data, qw.scales, qw.biases)) orelse
                quantize_mod.quantizedMatmul(cx, x, qw, true);
        }
    }.f;

    t = std.c.mach_absolute_time();
    for (0..REPS) |_| {
        const qkv = try qmmV8(ctx, x_in, qkv_w);
        defer qkv.deinit();
        const za = try qmmV8(ctx, x_in, z_w);
        defer za.deinit();
        const aa = try qmmV8(ctx, x_in, a_w);
        defer aa.deinit();
        const ba = try qmmV8(ctx, x_in, b_w);
        defer ba.deinit();
        try ba.eval();
    }
    const ms_proj = elapsedMs(t) / REPS;
    std.debug.print("layer/proj4 (v8): {d:.2} ms\n", .{ms_proj});

    t = std.c.mach_absolute_time();
    for (0..REPS) |_| {
        const gq = try qmmV8(ctx, x_in, gate_w);
        defer gq.deinit();
        const uq = try qmmV8(ctx, x_in, up_w);
        defer uq.deinit();
        const sa = try activations.silu(ctx, gq);
        defer sa.deinit();
        const mu = try ops.multiply(ctx, sa, uq);
        defer mu.deinit();
        const dq = try qmmV8(ctx, mu, down_w);
        defer dq.deinit();
        try dq.eval();
    }
    const ms_mlp = elapsedMs(t) / REPS;
    std.debug.print("layer/mlp (v8 + silu*mul): {d:.2} ms\n", .{ms_mlp});

    // out_proj: f32 input (what the serving tail emits) vs bf16 through v8
    const gate_f32 = try random.normal(ctx, &[_]i32{ 1, L, HV * D }, .float32, 0.0, 1.0, null);
    defer gate_f32.deinit();
    try gate_f32.eval();
    t = std.c.mach_absolute_time();
    for (0..REPS) |_| {
        const oq = try quantize_mod.quantizedMatmul(ctx, gate_f32, o_w, true);
        defer oq.deinit();
        try oq.eval();
    }
    const ms_out_f32 = elapsedMs(t) / REPS;
    std.debug.print("layer/out_proj f32-in stock: {d:.2} ms\n", .{ms_out_f32});

    t = std.c.mach_absolute_time();
    for (0..REPS) |_| {
        const oq = try qmmV8(ctx, try ops.astype(ctx, gate_f32, .bfloat16), o_w);
        defer oq.deinit();
        try oq.eval();
    }
    const ms_o_bf16 = elapsedMs(t) / REPS;
    std.debug.print("layer/out_proj astype+v8:    {d:.2} ms\n", .{ms_o_bf16});

    // recurrence (pipelined kernel, bf16 inputs as in serving)
    const ssm = try array_mod.zeros(allocator, &[_]i32{ HV, D, D }, .float32);
    defer ssm.deinit();
    const kb = try random.normal(ctx, &[_]i32{ L, HK, D }, .bfloat16, 0.0, 1.0, null);
    defer kb.deinit();
    const qb = try random.normal(ctx, &[_]i32{ L, HK, D }, .bfloat16, 0.0, 1.0, null);
    defer qb.deinit();
    const vb = try random.normal(ctx, &[_]i32{ L, HV, D }, .bfloat16, 0.0, 1.0, null);
    defer vb.deinit();
    const beta_b = try random.normal(ctx, &[_]i32{ L, HV }, .float32, 0.5, 0.1, null);
    defer beta_b.deinit();
    const g_b = try random.normal(ctx, &[_]i32{ L, HV }, .float32, -0.1, 0.01, null);
    defer g_b.deinit();
    try kb.eval();
    try qb.eval();
    try vb.eval();
    t = std.c.mach_absolute_time();
    for (0..REPS) |_| {
        const r = (try gdn.testDeltaRulePipelined(ctx, ssm, kb, vb, beta_b, g_b, qb, HK, HV, D)).?;
        defer r.y.deinit();
        defer r.s_new.deinit();
        try r.y.eval();
    }
    const ms_rec = elapsedMs(t) / REPS;
    std.debug.print("layer/recur (pipelined): {d:.2} ms\n", .{ms_rec});

    std.debug.print("SUM proj4+mlp+out(f32)+recur+conv+tail: {d:.2} ms/layer\n", .{ms_proj + ms_mlp + ms_out_f32 + ms_rec + 0.89 + 1.18});
}
