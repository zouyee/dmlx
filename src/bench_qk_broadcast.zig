// Phase 0 / B0 follow-up: how much does materializing the q/k head broadcast
// cost during prefill?
//
// `Qwen38GDN.prefillFront` ends with
//
//     q_full = takeAxis(q, qk_idx, 1)   // [L, HK, D] -> [L, HV, D]
//     k_full = takeAxis(k, qk_idx, 1)
//
// which expands 16 key/query heads to 48 value heads so the delta-rule kernel
// can index heads uniformly. HANDOFF records this as a deliberate
// simplification: "Our q/k are already broadcast to value heads ([L, HV, D]),
// so the hk mapping in the reference drops out" — i.e. mlx-lm's kernel does
// the mapping inside the kernel and dmlx materializes it instead.
//
// At L=2048 each gather writes 2048*48*128*4 = 50 MB, so the pair moves
// ~150 MB per layer and ~7 GB per prefill across 48 GDN layers. The section
// profile (plan doc §18.0/B0) puts GDN's front non-qmm at ~1.2 s of a 17.5 s
// prefill, and these two gathers are the prime suspect.
//
// Reference point: the same expansion done as a contiguous broadcast+reshape
// (what it would cost if the layout allowed it) and the theoretical traffic.
//
// Not committed. Run: zig build run-bench-qk-broadcast -Doptimize=ReleaseFast
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const random = @import("mlx").random;
const eval_mod = @import("mlx").eval;

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

    const HK: i32 = 16;
    const HV: i32 = 48;
    const D: i32 = 128;
    const reps: usize = 10;

    std.debug.print("q/k head broadcast cost: [L,{d},{d}] -> [L,{d},{d}] f32\n\n", .{ HK, D, HV, D });
    std.debug.print("{s: >6}{s: >14}{s: >14}{s: >12}{s: >12}\n", .{ "L", "gather ms", "pair/layer", "x48 layers", "written MB" });

    for ([_]i32{ 512, 2048 }) |L| {
        const q = try random.normal(ctx, &[_]i32{ L, HK, D }, .float32, 0.0, 0.02, null);
        defer q.deinit();
        try q.eval();

        // The production index vector: value head hv reads key head hv / 3.
        const idx_data = try allocator.alloc(i32, @intCast(HV));
        defer allocator.free(idx_data);
        for (0..@intCast(HV)) |i| idx_data[i] = @intCast(i / @as(usize, @intCast(@divExact(HV, HK))));
        const idx = try Array.fromData(allocator, i32, idx_data, &[_]i32{HV});
        defer idx.deinit();
        try idx.eval();

        for (0..2) |_| {
            const y = try shape_mod.takeAxis(ctx, q, idx, 1);
            defer y.deinit();
            try y.eval();
        }
        var pending = std.ArrayList(Array).empty;
        defer pending.deinit(allocator);
        defer for (pending.items) |y| y.deinit();
        for (0..reps) |_| {
            const y = try shape_mod.takeAxis(ctx, q, idx, 1);
            try pending.append(allocator, y);
        }
        const t = std.c.mach_absolute_time();
        try eval_mod.asyncEval(allocator, pending.items);
        try pending.items[reps - 1].eval();
        const ms = elapsedMs(t) / @as(f64, @floatFromInt(reps));

        const written_mb = @as(f64, @floatFromInt(L)) * @as(f64, @floatFromInt(HV)) *
            @as(f64, @floatFromInt(D)) * 4.0 / 1e6;
        std.debug.print("{d: >6}{d: >14.3}{d: >14.3}{d: >12.1}{d: >12.1}\n", .{
            L, ms, ms * 2, ms * 2 * 48, written_mb,
        });
    }

    std.debug.print("\nIf the kernel did the hk mapping itself (as mlx-lm's reference does),\n", .{});
    std.debug.print("this entire term disappears — q/k stay [L, HK, D].\n", .{});
}
