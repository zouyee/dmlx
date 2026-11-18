// SPDX-License-Identifier: Apache-2.0
//
// GPU-side acceptance for sampled (temp>0) speculative decoding.
//
// The CPU path computes min(1, p(d)/q(d)) per draft position with full-vocab
// softmaxes on the host, pulling every verify row across the bus (measured
// 23-31 ms/round at 2k context, 33-135 ms at 8k). This module instead runs
// the same math as a short stock-op pipeline — per-row max/exp/sum reductions
// plus a token-index slice, all on the GPU — and reads back ONE [6L] f32
// array. The host then applies the uniforms and finds the first rejection.
// The engine pulls at most ONE verify row per round afterwards (bonus sample
// or adjusted resample), so a round costs ~2-5 ms of acceptance instead of
// L row round-trips.
//
// Numerics: identical formula to mtpAcceptProb (softmax at temperature T via
// pre-scaling), f32 reductions in a different order — accept probabilities
// match the CPU path to ~1e-6, far below the uniform noise that decides
// acceptance. Any setup failure returns .used=false and the engine runs the
// original CPU loop unchanged.

const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

var disabled: ?bool = null;

fn envTrue(name: [*:0]const u8) bool {
    return std.c.getenv(name) != null;
}

pub fn enabled() bool {
    if (disabled == null) disabled = envTrue("DMLX_NO_ACCEPT_GPU");
    return !disabled.?;
}

pub const AcceptResult = struct {
    /// Number of accepted drafts before the first rejection (== drafts.len
    /// when every draft was accepted).
    k: usize,
    /// false -> caller runs the CPU per-row loop.
    used: bool,
};

pub fn compute(
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    target_rows: []const Array, // [drafts.len + 1] rows [V] f32
    draft_rows: []const Array, // [drafts.len] rows [V] f32
    drafts: []const u32,
    uniforms: []const f32,
    temperature: f32,
) !AcceptResult {
    const l = drafts.len;
    if (l == 0 or target_rows.len != l + 1 or draft_rows.len != l) {
        return .{ .k = 0, .used = false };
    }
    if (uniforms.len != l) return .{ .k = 0, .used = false };
    const v: i32 = target_rows[0].shape()[0];
    if (v <= 0) return .{ .k = 0, .used = false };

    const inv_t = 1.0 / @max(temperature, 1e-6);
    const scale = try ops.scalarF32(ctx, inv_t);
    defer scale.deinit();

    // Stack rows: [L, V] f32, pre-scaled by 1/T (ratio-invariant vs the
    // CPU path's per-softmax divide, but keeps rows directly comparable).
    const t_cat = try shape_mod.concatenateAxis(ctx, target_rows, 0);
    defer t_cat.deinit();
    const t_flat = try ops.reshape(ctx, t_cat, &[_]i32{ @intCast(l + 1), v });
    defer t_flat.deinit();
    const t_scaled = try ops.multiply(ctx, t_flat, scale);
    defer t_scaled.deinit();

    const q_cat = try shape_mod.concatenateAxis(ctx, draft_rows, 0);
    defer q_cat.deinit();
    const q_flat = try ops.reshape(ctx, q_cat, &[_]i32{ @intCast(l), v });
    defer q_flat.deinit();
    const q_scaled = try ops.multiply(ctx, q_flat, scale);
    defer q_scaled.deinit();

    // Per row: six scalars [1] f32 — row max, sum(exp(x-max)) for T and Q,
    // and the two logits at the drafted token index.
    var scalars = std.ArrayList(Array).empty;
    defer {
        for (scalars.items) |*s| s.deinit();
        scalars.deinit(allocator);
    }
    const one = [_]i32{1};
    for (0..l) |i| {
        const ii: i32 = @intCast(i);
        const t_row = try ops.slice(ctx, t_scaled, &[_]i32{ ii, 0 }, &[_]i32{ ii + 1, v }, &[_]i32{ 1, 1 });
        defer t_row.deinit();
        const q_row = try ops.slice(ctx, q_scaled, &[_]i32{ ii, 0 }, &[_]i32{ ii + 1, v }, &[_]i32{ 1, 1 });
        defer q_row.deinit();

        const t_max = try ops.max(ctx, t_row);
        defer t_max.deinit();
        const q_max = try ops.max(ctx, q_row);
        defer q_max.deinit();
        const t_norm = try ops.subtract(ctx, t_row, t_max);
        defer t_norm.deinit();
        const q_norm = try ops.subtract(ctx, q_row, q_max);
        defer q_norm.deinit();
        const t_se = try ops.sum(ctx, try ops.exp(ctx, t_norm));
        defer t_se.deinit();
        const q_se = try ops.sum(ctx, try ops.exp(ctx, q_norm));
        defer q_se.deinit();

        const tok: i32 = @intCast(drafts[i]);
        const t_tok = try ops.slice(ctx, t_row, &[_]i32{ 0, tok }, &[_]i32{ 1, tok + 1 }, &[_]i32{ 1, 1 });
        defer t_tok.deinit();
        const q_tok = try ops.slice(ctx, q_row, &[_]i32{ 0, tok }, &[_]i32{ 1, tok + 1 }, &[_]i32{ 1, 1 });
        defer q_tok.deinit();

        // Reshape every scalar to [1] so they concatenate cleanly.
        for ([_]Array{ t_max, q_max, t_se, q_se, t_tok, q_tok }) |s| {
            const r = try ops.reshape(ctx, s, &one);
            try scalars.append(allocator, r);
        }
    }

    const all = try shape_mod.concatenateAxis(ctx, scalars.items, 0);
    defer all.deinit();
    const data = all.dataSlice(f32) catch return .{ .k = 0, .used = false };
    if (data.len != 6 * l) return .{ .k = 0, .used = false };

    // Host: accept probabilities and first-rejection scan.
    var k: usize = l;
    for (0..l) |i| {
        const t_max = data[i * 6 + 0];
        const q_max = data[i * 6 + 1];
        const t_se = data[i * 6 + 2];
        const q_se = data[i * 6 + 3];
        const t_tok = data[i * 6 + 4];
        const q_tok = data[i * 6 + 5];
        // qd == 0 -> lp-lq = +inf -> a = 1 via the min clamp.
        const lp = t_tok - t_max - @log(t_se);
        const lq = q_tok - q_max - @log(q_se);
        const a = @min(1.0, @exp(lp - lq));
        if (uniforms[i] >= a) {
            k = i;
            break;
        }
    }
    return .{ .k = k, .used = true };
}
