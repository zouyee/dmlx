/// Qwen3.8 decode profiling harness (TEMPORARY — do not commit).
///
/// Loads /Users/zouyee/models/Qwen3.8-27B-4bit exactly like the serving path
/// (model_registry.qwen38Loader: resident weight map + Qwen38ServingModel),
/// prefills a short prompt, then decodes N tokens twice:
///
///   mode "baseline":  uninstrumented forwardTokens loop, wall ms/token
///                     (this is the number to compare against ~90ms).
///   mode "breakdown": harness-local re-implementation of the L=1 decode step
///                     with an eval barrier + wall timer around every section
///                     (embed / per-layer ln1 / gdn / attn / ln2 / mlp /
///                     final norm + lm_head / asyncEval+logits barrier /
///                     sampling). Barrier-per-section serializes CPU graph
///                     build and GPU execution, so the breakdown total
///                     OVERSTATES the baseline; section shares still identify
///                     the hotspot.
///
/// Sub-breakdown of GDN/attn internals requires the temporary
/// instrumentation in qwen38_gdn.zig / qwen38_attn.zig (DMLX_PROF_SUB=1).
///
/// Env-gated apply-level timing: with DMLX_PROF_APPLY=1 the baseline mode
/// additionally runs an instrumented replica of the forwardTokens decode
/// step (same op sequence, compiled closures active, no intermediate
/// barriers) that accumulates per token:
///   - GDN closure apply x48 (input vector build + mlx_closure_apply + state swap)
///   - MLP swiglu closure apply x64
///   - attn pre-cache closure apply x16
///   - attn post-cache eager ops (KV append + SDPA + o_proj) x16
///   - other eager ops (embed, ln/residual clusters, final norm + lm_head)
///   - barrier split: asyncEval cache-state submit (CPU dispatch) vs
///     logits.eval (GPU wait)
///
/// Usage:
///   zig build run-qwen38-profile -Doptimize=ReleaseFast -- \
///       [--model-dir DIR] [--prefill N] [--decode N] [--mode baseline|breakdown|both]
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const fast_mod = @import("mlx").fast;
const activations = @import("mlx").activations;
const eval_mod = @import("mlx").eval;
const memory_mod = @import("mlx").memory;
const safetensors_reader = @import("mlx").safetensors_reader;
const qwen38_loader = @import("models/qwen38_loader.zig");
const qwen38 = @import("models/qwen38.zig");
const qwen38_gdn = @import("models/qwen38_gdn.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

const Lap = struct {
    t: u64,
    fn start() Lap {
        return .{ .t = std.c.mach_absolute_time() };
    }
    fn ticksToNs(ticks: u64) u64 {
        return ticks * 125 / 3; // Apple Silicon timebase
    }
    /// ns since start, and restart.
    fn lap(self: *Lap) u64 {
        const now = std.c.mach_absolute_time();
        const d = ticksToNs(now - self.t);
        self.t = now;
        return d;
    }
    fn reset(self: *Lap) void {
        self.t = std.c.mach_absolute_time();
    }
    fn read(self: *Lap) u64 {
        return ticksToNs(std.c.mach_absolute_time() - self.t);
    }
};

const Accum = struct {
    embed_ns: u64 = 0,
    ln_ns: u64 = 0, // both layernorms + residual adds
    gdn_ns: u64 = 0, // 48 GDN layers (whole branch)
    attn_ns: u64 = 0, // 16 full-attn layers (whole branch)
    mlp_ns: u64 = 0, // 64 MLPs (3 qmm each)
    head_ns: u64 = 0, // final norm + lm_head qmm
    barrier_ns: u64 = 0, // asyncEval(cache states) + logits.eval
    sample_ns: u64 = 0, // argmax readback
    tokens: u64 = 0,

    fn ms(ns: u64) f64 {
        return @as(f64, @floatFromInt(ns)) / 1e6;
    }

    fn report(self: *const Accum, label: []const u8) void {
        const per = @as(f64, @floatFromInt(@max(self.tokens, 1)));
        const section_ns = self.embed_ns + self.ln_ns + self.gdn_ns + self.attn_ns + self.mlp_ns + self.head_ns + self.barrier_ns + self.sample_ns;
        std.debug.print("\n=== {s}: {d} decode tokens, breakdown-total {d:.2} ms/token ===\n", .{ label, self.tokens, ms(section_ns) / per });
        const rows = .{
            .{ "embed lookup", self.embed_ns },
            .{ "layernorms+residual (128/token)", self.ln_ns },
            .{ "GDN layers x48 (5 qmm + conv + state)", self.gdn_ns },
            .{ "full-attn layers x16 (4 qmm + sdpa)", self.attn_ns },
            .{ "MLP x64 (3 qmm)", self.mlp_ns },
            .{ "final norm + lm_head qmm", self.head_ns },
            .{ "asyncEval + logits.eval barrier", self.barrier_ns },
            .{ "sampling (argmax readback)", self.sample_ns },
        };
        inline for (rows) |r| {
            const ns = r[1];
            std.debug.print("  {s: <44} {d: >7.3} ms/tok  {d: >5.1}%\n", .{ r[0], ms(ns) / per, 100.0 * @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(@max(section_ns, 1))) });
        }
        if (self.gdn_ns > 0) {
            std.debug.print("  per-layer: gdn {d:.4} ms | attn {d:.4} ms | mlp {d:.4} ms\n", .{
                ms(self.gdn_ns) / per / 48.0,
                ms(self.attn_ns) / per / 16.0,
                ms(self.mlp_ns) / per / 64.0,
            });
        }
    }
};

/// Replicates Qwen38ServingModel.asyncEvalCacheStates (private) from pub fields.
fn asyncEvalCacheStates(model: *qwen38.Qwen38ServingModel) !void {
    var states = std.ArrayList(Array).empty;
    defer states.deinit(model.allocator);
    for (model.caches) |*lc| {
        switch (lc.*) {
            .gdn => |*gc| {
                try states.append(model.allocator, gc.conv_state);
                try states.append(model.allocator, gc.ssm_state);
            },
            .attn => |*ac| {
                if (ac.keys) |k| try states.append(model.allocator, k);
                if (ac.values) |v| try states.append(model.allocator, v);
            },
        }
    }
    try eval_mod.asyncEval(model.allocator, states.items);
}

/// Instrumented re-implementation of Qwen38ServingModel.forwardTokens for the
/// L=1 decode path. Uses only pub fields of the serving model.
/// Prefill (L > 1) counterpart of `forwardStepProfiled`.
///
/// Added because the prefill gap to omlx/mlx-vlm (117.4 vs 134.5 / 133.1
/// tok/s) could not be attributed from isolated qmm micro-benchmarks:
/// summing those gave "qmm is 85.6% of the prefill wall", but the same kind
/// of isolated-vs-in-context mismatch already bit the verify work twice, so
/// the non-qmm 14.4% had to be measured in place rather than inferred. This
/// mirrors `Qwen38ServingModel.forwardTokensCore` at L > 1, including the
/// GDN prefill path (`g.prefill`, not `g.forward`) and the last-row-only
/// head, with a barrier after each section.
///
/// As with the decode breakdown, barriers serialize work that overlaps in the
/// real forward, so the total OVERSTATES the wall; the shares are what matter.
fn forwardPrefillProfiled(
    model: *qwen38.Qwen38ServingModel,
    ids: []const u32,
    acc: *Accum,
) !Array {
    const allocator = model.allocator;
    const cfg = model.cfg;
    const ctx = model.ctx;
    const hidden: i32 = @intCast(cfg.hidden_size);
    const L: i32 = @intCast(ids.len);
    var t = Lap.start();

    const ids_i32 = try allocator.alloc(i32, ids.len);
    defer allocator.free(ids_i32);
    for (ids, 0..) |v, i| ids_i32[i] = @intCast(v);
    const id_arr = try Array.fromData(allocator, i32, ids_i32, &[_]i32{L});
    defer id_arr.deinit();
    const rows = try shape_mod.takeAxis(ctx, model.embed_table_bf16.?, id_arr, 0);
    defer rows.deinit();
    const emb = try ops.reshape(ctx, rows, &[_]i32{ 1, L, hidden });
    defer emb.deinit();
    var h = try ops.copy(ctx, emb);
    defer h.deinit();
    try h.eval();
    acc.embed_ns += t.lap();

    var gdn_seen: usize = 0;
    for (model.layers, 0..) |*lp, layer| {
        t.reset();
        const normed1 = blk: {
            const h_f32 = try ops.astype(ctx, h, .float32);
            defer h_f32.deinit();
            const n_f32 = try fast_mod.rmsNorm(ctx, h_f32, lp.ln1_f32, cfg.rms_norm_eps);
            defer n_f32.deinit();
            break :blk try ops.astype(ctx, n_f32, .bfloat16);
        };
        defer normed1.deinit();
        try normed1.eval();
        acc.ln_ns += t.lap();

        t.reset();
        const attn_out = switch (lp.module) {
            .gdn => |*g| blk: {
                gdn_seen += 1;
                const cache = &model.caches[layer].gdn;
                break :blk try g.prefill(allocator, normed1, cache, gdn_seen % 8 == 0 or layer + 1 == model.layers.len);
            },
            .attn => |*a| try a.forward(normed1, &model.caches[layer].attn, null),
        };
        defer attn_out.deinit();
        try attn_out.eval();
        switch (lp.module) {
            .gdn => acc.gdn_ns += t.lap(),
            .attn => acc.attn_ns += t.lap(),
        }

        t.reset();
        {
            const next = try ops.add(ctx, h, attn_out);
            h.deinit();
            h = next;
        }
        const normed2 = blk: {
            const h_f32 = try ops.astype(ctx, h, .float32);
            defer h_f32.deinit();
            const n_f32 = try fast_mod.rmsNorm(ctx, h_f32, lp.ln2_f32, cfg.rms_norm_eps);
            defer n_f32.deinit();
            break :blk try ops.astype(ctx, n_f32, .bfloat16);
        };
        defer normed2.deinit();
        try normed2.eval();
        acc.ln_ns += t.lap();

        t.reset();
        const mlp_out = blk: {
            const gate = try lp.mlp_gate.apply(ctx, normed2);
            defer gate.deinit();
            const up = try lp.mlp_up.apply(ctx, normed2);
            defer up.deinit();
            const act = try activations.silu(ctx, gate);
            defer act.deinit();
            const prod = try ops.multiply(ctx, act, up);
            defer prod.deinit();
            break :blk try lp.mlp_down.apply(ctx, prod);
        };
        defer mlp_out.deinit();
        try mlp_out.eval();
        acc.mlp_ns += t.lap();

        t.reset();
        {
            const next = try ops.add(ctx, h, mlp_out);
            h.deinit();
            h = next;
        }
        try h.eval();
        acc.ln_ns += t.lap();
    }

    // Head on the LAST row only, matching production (`last_logits_only`).
    t.reset();
    const last = try ops.slice(ctx, h, &[_]i32{ 0, L - 1, 0 }, &[_]i32{ 1, L, hidden }, &[_]i32{ 1, 1, 1 });
    defer last.deinit();
    const h_f32 = try ops.astype(ctx, last, .float32);
    defer h_f32.deinit();
    const hf = try fast_mod.rmsNorm(ctx, h_f32, model.final_norm_f32.?, cfg.rms_norm_eps);
    defer hf.deinit();
    const hb = try ops.astype(ctx, hf, .bfloat16);
    defer hb.deinit();
    const logits = try model.lm_head.?.apply(ctx, hb);
    try logits.eval();
    acc.head_ns += t.lap();
    return logits;
}

fn forwardStepProfiled(model: *qwen38.Qwen38ServingModel, id: u32, acc: *Accum) !Array {
    const allocator = model.allocator;
    const cfg = model.cfg;
    const ctx = model.ctx;
    const hidden: i32 = @intCast(cfg.hidden_size);
    var t = Lap.start();

    // --- embed ---
    const id_arr = try Array.fromData(allocator, i32, &[_]i32{@intCast(id)}, &[_]i32{1});
    defer id_arr.deinit();
    const rows = try shape_mod.takeAxis(ctx, model.embed_table_bf16.?, id_arr, 0);
    defer rows.deinit();
    const emb = try ops.reshape(ctx, rows, &[_]i32{ 1, 1, hidden });
    defer emb.deinit();
    var h = try ops.copy(ctx, emb);
    defer h.deinit();
    try h.eval();
    acc.embed_ns += t.lap();

    for (model.layers, 0..) |*lp, layer| {
        // --- ln1 ---
        t.reset();
        const normed1 = blk: {
            const h_f32 = try ops.astype(ctx, h, .float32);
            defer h_f32.deinit();
            const n_f32 = try fast_mod.rmsNorm(ctx, h_f32, lp.ln1_f32, cfg.rms_norm_eps);
            defer n_f32.deinit();
            break :blk try ops.astype(ctx, n_f32, .bfloat16);
        };
        defer normed1.deinit();
        try normed1.eval();
        acc.ln_ns += t.lap();

        // --- attention sub-block branch ---
        t.reset();
        const attn_out = switch (lp.module) {
            .gdn => |*g| blk: {
                const cache = &model.caches[layer].gdn;
                const step_in = try ops.reshape(ctx, normed1, &[_]i32{ 1, hidden });
                defer step_in.deinit();
                const step_out = try g.forward(step_in, cache);
                defer step_out.deinit();
                break :blk try ops.reshape(ctx, step_out, &[_]i32{ 1, 1, hidden });
            },
            .attn => |*a| try a.forward(normed1, &model.caches[layer].attn, null),
        };
        defer attn_out.deinit();
        try attn_out.eval();
        switch (lp.module) {
            .gdn => acc.gdn_ns += t.lap(),
            .attn => acc.attn_ns += t.lap(),
        }

        // residual add (folded into ln accounting)
        t.reset();
        {
            const next = try ops.add(ctx, h, attn_out);
            h.deinit();
            h = next;
        }

        // --- ln2 ---
        const normed2 = blk: {
            const h_f32 = try ops.astype(ctx, h, .float32);
            defer h_f32.deinit();
            const n_f32 = try fast_mod.rmsNorm(ctx, h_f32, lp.ln2_f32, cfg.rms_norm_eps);
            defer n_f32.deinit();
            break :blk try ops.astype(ctx, n_f32, .bfloat16);
        };
        defer normed2.deinit();
        try normed2.eval();
        acc.ln_ns += t.lap();

        // --- MLP ---
        t.reset();
        const mlp_out = blk: {
            const gate = try lp.mlp_gate.apply(ctx, normed2);
            defer gate.deinit();
            const up = try lp.mlp_up.apply(ctx, normed2);
            defer up.deinit();
            const act = try activations.silu(ctx, gate);
            defer act.deinit();
            const hh = try ops.multiply(ctx, act, up);
            defer hh.deinit();
            break :blk try lp.mlp_down.apply(ctx, hh);
        };
        defer mlp_out.deinit();
        {
            const next = try ops.add(ctx, h, mlp_out);
            h.deinit();
            h = next;
        }
        try h.eval();
        acc.mlp_ns += t.lap();
    }

    // --- final norm + lm_head ---
    t.reset();
    const h_f32 = try ops.astype(ctx, h, .float32);
    defer h_f32.deinit();
    const hidden_final_f32 = try fast_mod.rmsNorm(ctx, h_f32, model.final_norm_f32.?, cfg.rms_norm_eps);
    defer hidden_final_f32.deinit();
    const hidden_final = try ops.astype(ctx, hidden_final_f32, .bfloat16);
    defer hidden_final.deinit();
    const logits = try model.lm_head.?.apply(ctx, hidden_final);
    try logits.eval();
    acc.head_ns += t.lap();

    // --- barrier (already-evaled logits: measures asyncEval submit cost) ---
    t.reset();
    try asyncEvalCacheStates(model);
    try logits.eval();
    acc.barrier_ns += t.lap();
    return logits;
}

/// Same step as forwardStepProfiled but with NO intermediate barriers:
/// splits wall time into CPU graph-build vs final GPU eval wait. This tells
/// whether the decode loop is CPU-bound (graph construction) or GPU-bound.
fn forwardStepBuildSplit(model: *qwen38.Qwen38ServingModel, id: u32, build_ns: *u64, eval_ns: *u64) !Array {
    const allocator = model.allocator;
    const cfg = model.cfg;
    const ctx = model.ctx;
    const hidden: i32 = @intCast(cfg.hidden_size);
    var t = Lap.start();

    const id_arr = try Array.fromData(allocator, i32, &[_]i32{@intCast(id)}, &[_]i32{1});
    defer id_arr.deinit();
    const rows = try shape_mod.takeAxis(ctx, model.embed_table_bf16.?, id_arr, 0);
    defer rows.deinit();
    const emb = try ops.reshape(ctx, rows, &[_]i32{ 1, 1, hidden });
    defer emb.deinit();
    var h = try ops.copy(ctx, emb);
    defer h.deinit();

    for (model.layers, 0..) |*lp, layer| {
        const normed1 = blk: {
            const h_f32 = try ops.astype(ctx, h, .float32);
            defer h_f32.deinit();
            const n_f32 = try fast_mod.rmsNorm(ctx, h_f32, lp.ln1_f32, cfg.rms_norm_eps);
            defer n_f32.deinit();
            break :blk try ops.astype(ctx, n_f32, .bfloat16);
        };
        defer normed1.deinit();
        const attn_out = switch (lp.module) {
            .gdn => |*g| blk: {
                const cache = &model.caches[layer].gdn;
                const step_in = try ops.reshape(ctx, normed1, &[_]i32{ 1, hidden });
                defer step_in.deinit();
                if (model.gdn_step_compiled) |cc| {
                    const step_out = try model.applyGdnStep(cc, g, cache, step_in);
                    defer step_out.deinit();
                    break :blk try ops.reshape(ctx, step_out, &[_]i32{ 1, 1, hidden });
                }
                const step_out = try g.forward(step_in, cache);
                defer step_out.deinit();
                break :blk try ops.reshape(ctx, step_out, &[_]i32{ 1, 1, hidden });
            },
            .attn => |*a| try a.forward(normed1, &model.caches[layer].attn, null),
        };
        defer attn_out.deinit();
        {
            const next = try ops.add(ctx, h, attn_out);
            h.deinit();
            h = next;
        }
        const normed2 = blk: {
            const h_f32 = try ops.astype(ctx, h, .float32);
            defer h_f32.deinit();
            const n_f32 = try fast_mod.rmsNorm(ctx, h_f32, lp.ln2_f32, cfg.rms_norm_eps);
            defer n_f32.deinit();
            break :blk try ops.astype(ctx, n_f32, .bfloat16);
        };
        defer normed2.deinit();
        const mlp_out = blk: {
            if (model.mlp_swiglu_compiled) |mc| {
                break :blk try qwen38.applyMlpSwiglu(allocator, mc, lp, normed2);
            }
            const gate = try lp.mlp_gate.apply(ctx, normed2);
            defer gate.deinit();
            const up = try lp.mlp_up.apply(ctx, normed2);
            defer up.deinit();
            const act = try activations.silu(ctx, gate);
            defer act.deinit();
            const hh = try ops.multiply(ctx, act, up);
            defer hh.deinit();
            break :blk try lp.mlp_down.apply(ctx, hh);
        };
        defer mlp_out.deinit();
        {
            const next = try ops.add(ctx, h, mlp_out);
            h.deinit();
            h = next;
        }
    }

    const h_f32 = try ops.astype(ctx, h, .float32);
    defer h_f32.deinit();
    const hidden_final_f32 = try fast_mod.rmsNorm(ctx, h_f32, model.final_norm_f32.?, cfg.rms_norm_eps);
    defer hidden_final_f32.deinit();
    const hidden_final = try ops.astype(ctx, hidden_final_f32, .bfloat16);
    defer hidden_final.deinit();
    const logits = try model.lm_head.?.apply(ctx, hidden_final);
    try asyncEvalCacheStates(model);
    build_ns.* += t.lap();
    try logits.eval();
    eval_ns.* += t.lap();
    return logits;
}

/// Per-token apply-level accumulators for the DMLX_PROF_APPLY=1 mode.
const ApplyAccum = struct {
    gdn_ns: u64 = 0, // 48 compiled GDN step applies (vector build + apply + swap)
    gdn_n: u64 = 0,
    mlp_ns: u64 = 0, // 64 compiled SwiGLU applies
    mlp_n: u64 = 0,
    attn_pre_ns: u64 = 0, // 16 compiled attn pre-cache applies
    attn_pre_n: u64 = 0,
    attn_post_ns: u64 = 0, // 16 eager attn post-cache segments (KV append + SDPA + o_proj)
    eager_ns: u64 = 0, // embed + ln/residual clusters + final norm + lm_head
    barrier_ns: u64 = 0, // asyncEval(cache states) submit (CPU dispatch)
    eval_ns: u64 = 0, // logits.eval (GPU wait after submit)
    tokens: u64 = 0,

    fn ms(ns: u64) f64 {
        return @as(f64, @floatFromInt(ns)) / 1e6;
    }

    fn report(self: *const ApplyAccum, wall_ns: u64) void {
        const per = @as(f64, @floatFromInt(@max(self.tokens, 1)));
        const wall = ms(wall_ns) / per;
        const applies = ms(self.gdn_ns + self.mlp_ns + self.attn_pre_ns) / per;
        const eager = ms(self.attn_post_ns + self.eager_ns) / per;
        std.debug.print("\n=== apply-level (DMLX_PROF_APPLY): {d} decode tokens, wall {d:.2} ms/token ===\n", .{ self.tokens, wall });
        const rows = .{
            .{ "GDN closure apply x48", self.gdn_ns, self.gdn_n },
            .{ "MLP closure apply x64", self.mlp_ns, self.mlp_n },
            .{ "attn pre-cache apply x16", self.attn_pre_ns, self.attn_pre_n },
            .{ "attn post-cache eager x16", self.attn_post_ns, @as(u64, 0) },
            .{ "eager (embed/ln/residual/head)", self.eager_ns, @as(u64, 0) },
            .{ "barrier submit (asyncEval states)", self.barrier_ns, @as(u64, 0) },
            .{ "logits.eval (GPU wait)", self.eval_ns, @as(u64, 0) },
        };
        inline for (rows) |r| {
            const ns = r[1];
            std.debug.print("  {s: <38} {d: >7.3} ms/tok  {d: >5.1}% of wall\n", .{ r[0], ms(ns) / per, 100.0 * @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(@max(wall_ns, 1))) });
            if (r[2] > 0) {
                const n = r[2];
                std.debug.print("     -> {d:.4} ms/apply ({d} applies/token)\n", .{ ms(ns) / per / @as(f64, @floatFromInt(@max(n / @max(self.tokens, 1), 1))), n / @max(self.tokens, 1) });
            }
        }
        std.debug.print("  applies total (GDN+MLP+attn-pre):   {d:.3} ms/tok ({d:.1}% of wall)\n", .{ applies, 100.0 * applies / wall });
        std.debug.print("  eager remnants total:               {d:.3} ms/tok ({d:.1}% of wall)\n", .{ eager, 100.0 * eager / wall });
    }
};

/// Instrumented replica of the forwardTokens L=1 decode step (compiled
/// closures active, no intermediate barriers) that times each compiled
/// closure apply against the surrounding eager dispatch. Wall-identical op
/// sequence to forwardStepBuildSplit plus the attn pre-cache segment.
fn forwardStepApplyProf(model: *qwen38.Qwen38ServingModel, id: u32, acc: *ApplyAccum) !Array {
    const allocator = model.allocator;
    const cfg = model.cfg;
    const ctx = model.ctx;
    const hidden: i32 = @intCast(cfg.hidden_size);
    var t = Lap.start();

    const id_arr = try Array.fromData(allocator, i32, &[_]i32{@intCast(id)}, &[_]i32{1});
    defer id_arr.deinit();
    const rows = try shape_mod.takeAxis(ctx, model.embed_table_bf16.?, id_arr, 0);
    defer rows.deinit();
    const emb = try ops.reshape(ctx, rows, &[_]i32{ 1, 1, hidden });
    defer emb.deinit();
    var h = try ops.copy(ctx, emb);
    defer h.deinit();
    acc.eager_ns += t.lap();

    for (model.layers, 0..) |*lp, layer| {
        // --- ln1 (eager) ---
        t.reset();
        const normed1 = blk: {
            const h_f32 = try ops.astype(ctx, h, .float32);
            defer h_f32.deinit();
            const n_f32 = try fast_mod.rmsNorm(ctx, h_f32, lp.ln1_f32, cfg.rms_norm_eps);
            defer n_f32.deinit();
            break :blk try ops.astype(ctx, n_f32, .bfloat16);
        };
        defer normed1.deinit();
        acc.eager_ns += t.lap();

        // --- attention sub-block branch ---
        const attn_out = switch (lp.module) {
            .gdn => |*g| blk: {
                const cache = &model.caches[layer].gdn;
                t.reset();
                const step_in = try ops.reshape(ctx, normed1, &[_]i32{ 1, hidden });
                defer step_in.deinit();
                acc.eager_ns += t.lap();
                if (model.gdn_step_compiled) |cc| {
                    t.reset();
                    const step_out = try model.applyGdnStep(cc, g, cache, step_in);
                    acc.gdn_ns += t.lap();
                    acc.gdn_n += 1;
                    defer step_out.deinit();
                    t.reset();
                    const r = try ops.reshape(ctx, step_out, &[_]i32{ 1, 1, hidden });
                    acc.eager_ns += t.lap();
                    break :blk r;
                }
                t.reset();
                const step_out = try g.forward(step_in, cache);
                acc.eager_ns += t.lap();
                defer step_out.deinit();
                break :blk try ops.reshape(ctx, step_out, &[_]i32{ 1, 1, hidden });
            },
            .attn => |*a| blk: {
                const cache = &model.caches[layer].attn;
                if (model.attn_precache_compiled) |ac| {
                    t.reset();
                    const step_in = try ops.reshape(ctx, normed1, &[_]i32{ 1, hidden });
                    defer step_in.deinit();
                    acc.eager_ns += t.lap();
                    t.reset();
                    var pre = try model.applyAttnPreCache(ac, a, cache, step_in);
                    acc.attn_pre_ns += t.lap();
                    acc.attn_pre_n += 1;
                    defer pre.deinit();
                    t.reset();
                    const out2d = try a.postCache(pre, cache);
                    defer out2d.deinit();
                    const r = try ops.reshape(ctx, out2d, &[_]i32{ 1, 1, hidden });
                    acc.attn_post_ns += t.lap();
                    break :blk r;
                }
                t.reset();
                const r = try a.forward(normed1, cache, null);
                acc.eager_ns += t.lap();
                break :blk r;
            },
        };
        defer attn_out.deinit();

        // --- residual + ln2 (eager) ---
        t.reset();
        {
            const next = try ops.add(ctx, h, attn_out);
            h.deinit();
            h = next;
        }
        const normed2 = blk: {
            const h_f32 = try ops.astype(ctx, h, .float32);
            defer h_f32.deinit();
            const n_f32 = try fast_mod.rmsNorm(ctx, h_f32, lp.ln2_f32, cfg.rms_norm_eps);
            defer n_f32.deinit();
            break :blk try ops.astype(ctx, n_f32, .bfloat16);
        };
        defer normed2.deinit();
        acc.eager_ns += t.lap();

        // --- MLP ---
        if (model.mlp_swiglu_compiled) |mc| {
            t.reset();
            const mlp_out = try qwen38.applyMlpSwiglu(allocator, mc, lp, normed2);
            acc.mlp_ns += t.lap();
            acc.mlp_n += 1;
            defer mlp_out.deinit();
            t.reset();
            const next = try ops.add(ctx, h, mlp_out);
            h.deinit();
            h = next;
            acc.eager_ns += t.lap();
            continue;
        }
        t.reset();
        const mlp_out = blk: {
            const gate = try lp.mlp_gate.apply(ctx, normed2);
            defer gate.deinit();
            const up = try lp.mlp_up.apply(ctx, normed2);
            defer up.deinit();
            const act = try activations.silu(ctx, gate);
            defer act.deinit();
            const hh = try ops.multiply(ctx, act, up);
            defer hh.deinit();
            break :blk try lp.mlp_down.apply(ctx, hh);
        };
        defer mlp_out.deinit();
        {
            const next = try ops.add(ctx, h, mlp_out);
            h.deinit();
            h = next;
        }
        acc.eager_ns += t.lap();
    }

    // --- final norm + lm_head (eager) ---
    t.reset();
    const h_f32 = try ops.astype(ctx, h, .float32);
    defer h_f32.deinit();
    const hidden_final_f32 = try fast_mod.rmsNorm(ctx, h_f32, model.final_norm_f32.?, cfg.rms_norm_eps);
    defer hidden_final_f32.deinit();
    const hidden_final = try ops.astype(ctx, hidden_final_f32, .bfloat16);
    defer hidden_final.deinit();
    const logits = try model.lm_head.?.apply(ctx, hidden_final);
    acc.eager_ns += t.lap();

    // --- barrier ---
    t.reset();
    try asyncEvalCacheStates(model);
    acc.barrier_ns += t.lap();
    t.reset();
    try logits.eval();
    acc.eval_ns += t.lap();
    return logits;
}

fn sampleGreedy(ctx: EagerContext, logits: Array) !u32 {
    const sh = logits.shape();
    const vocab: usize = @intCast(sh[sh.len - 1]);
    const seq: i32 = @intCast(sh[sh.len - 2]);
    const last = try ops.slice(ctx, logits, &[_]i32{ 0, seq - 1, 0 }, &[_]i32{ 1, seq, @intCast(vocab) }, &[_]i32{ 1, 1, 1 });
    defer last.deinit();
    const last_f32 = try ops.astype(ctx, last, .float32);
    defer last_f32.deinit();
    try last_f32.eval();
    const data = try last_f32.dataSlice(f32);
    var best: usize = 0;
    for (data[1..], 1..) |v, i| {
        if (v > data[best]) best = i;
    }
    return @intCast(best);
}

pub fn main(init: std.process.Init) !void {
    c.initErrorHandler();
    const allocator = init.arena.allocator();
    const io = init.io;

    var model_dir: []const u8 = "/Users/zouyee/models/Qwen3.8-27B-4bit";
    var n_prefill: usize = 10;
    var n_decode: usize = 50;
    var mode: []const u8 = "both";

    const argv = try init.minimal.args.toSlice(allocator);
    var ai: usize = 1;
    while (ai < argv.len) : (ai += 1) {
        const arg: []const u8 = argv[ai];
        if (std.mem.eql(u8, arg, "--model-dir")) {
            ai += 1;
            model_dir = if (ai < argv.len) argv[ai] else return error.MissingArgValue;
        } else if (std.mem.eql(u8, arg, "--prefill")) {
            ai += 1;
            n_prefill = try std.fmt.parseInt(usize, argv[ai], 10);
        } else if (std.mem.eql(u8, arg, "--decode")) {
            ai += 1;
            n_decode = try std.fmt.parseInt(usize, argv[ai], 10);
        } else if (std.mem.eql(u8, arg, "--mode")) {
            ai += 1;
            mode = if (ai < argv.len) argv[ai] else return error.MissingArgValue;
        } else return error.UnknownArg;
    }

    // --- load config ---
    const cfg_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, allocator, .limited(1024 * 1024));
    var cfg = try qwen38_loader.parseQwen38Config(allocator, cfg_text);
    defer cfg.deinit();

    // Match the serving path: wired/cache limits at ~85%/~80% of system RAM.
    {
        const sysctl_c = @cImport(@cInclude("sys/sysctl.h"));
        var memsize: u64 = 0;
        var len: usize = @sizeOf(u64);
        var mib = [_]c_int{ sysctl_c.CTL_HW, sysctl_c.HW_MEMSIZE };
        _ = sysctl_c.sysctl(&mib, 2, &memsize, &len, null, 0);
        var old_wired: usize = 0;
        _ = c.c.mlx_set_wired_limit(&old_wired, @intCast(memsize * 85 / 100));
        var old_cache: usize = 0;
        _ = c.c.mlx_set_cache_limit(&old_cache, @intCast(memsize * 4 / 5));
    }

    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });
    defer ctx.deinit();

    // --- resident weights (identical to model_registry.qwen38Loader) ---
    var idx = try safetensors_reader.loadOrBuildIndex(allocator, model_dir);
    defer idx.deinit();

    // Packed-weight byte totals for the roofline check (before embed removal).
    var packed_bytes_total: u64 = 0;
    var packed_bytes_lm_head: u64 = 0;
    var packed_bytes_embed: u64 = 0;
    {
        var it = idx.entries.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (!std.mem.startsWith(u8, name, "language_model.")) continue;
            if (!std.mem.endsWith(u8, name, ".weight")) continue;
            const info = entry.value_ptr.*;
            const bytes: u64 = info.data_offset_end - info.data_offset_start;
            // packed 4-bit: weight bytes + scales + biases accounted roughly by
            // weight bytes * 1.125; use raw weight bytes and note the +12.5%.
            const scale_bytes = bytes / 8; // gs=64 f32 scale+bias ≈ 8B/64 vals = 1/8 of 4-bit payload
            if (std.mem.indexOf(u8, name, "embed_tokens") != null) {
                packed_bytes_embed += bytes + scale_bytes;
            } else if (std.mem.indexOf(u8, name, "lm_head") != null) {
                packed_bytes_lm_head += bytes + scale_bytes;
            } else {
                packed_bytes_total += bytes + scale_bytes;
            }
        }
    }
    std.debug.print("packed weights/token (layers): {d:.3} GB | lm_head: {d:.3} GB | embed(resident bf16): {d:.3} GB packed\n", .{
        @as(f64, @floatFromInt(packed_bytes_total)) / 1e9,
        @as(f64, @floatFromInt(packed_bytes_lm_head)) / 1e9,
        @as(f64, @floatFromInt(packed_bytes_embed)) / 1e9,
    });

    var weights = std.StringHashMap(Array).init(allocator);
    {
        var it = idx.entries.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (!std.mem.startsWith(u8, name, "language_model.")) continue;
            const key_copy = try allocator.dupe(u8, name);
            const arr = try idx.loadTensor(name);
            try weights.put(key_copy, arr);
        }
    }
    std.debug.print("weights resident: {d} tensors\n", .{weights.count()});

    var model = try qwen38.Qwen38ServingModel.init(allocator, ctx, &cfg, weights);
    defer model.deinit();
    model.ws = qwen38.WeightSource.fromMap(&model.weights, ctx, &model.ws_backing);
    {
        const peak = memory_mod.getPeakMemory() catch 0;
        const active = memory_mod.getActiveMemory() catch 0;
        std.debug.print("memory after load: active {d:.2} GB, peak {d:.2} GB\n", .{
            @as(f64, @floatFromInt(active)) / 1e9,
            @as(f64, @floatFromInt(peak)) / 1e9,
        });
    }

    // Prompt ids: "The capital of France is" + filler (content is irrelevant).
    var prompt = std.ArrayList(u32).empty;
    try prompt.appendSlice(allocator, &[_]u32{ 760, 6511, 314, 9338, 369 });
    while (prompt.items.len < n_prefill) try prompt.append(allocator, 9707);

    const run_baseline = std.mem.eql(u8, mode, "baseline") or std.mem.eql(u8, mode, "both");
    const run_breakdown = std.mem.eql(u8, mode, "breakdown") or std.mem.eql(u8, mode, "both");
    const run_buildsplit = std.mem.eql(u8, mode, "buildsplit") or std.mem.eql(u8, mode, "both");
    const run_prefill_bd = std.mem.eql(u8, mode, "prefill") or std.mem.eql(u8, mode, "both");

    // ================= baseline: uninstrumented forwardTokens =================
    if (run_baseline) {
        var logits = try model.forwardTokens(prompt.items); // prefill (resets state)
        var next = try sampleGreedy(ctx, logits);
        logits.deinit();
        // warmup
        for (0..5) |_| {
            logits = try model.forwardTokens(&[_]u32{next});
            next = try sampleGreedy(ctx, logits);
            logits.deinit();
        }
        var wall = Lap.start();
        var i: usize = 0;
        while (i < n_decode) : (i += 1) {
            logits = try model.forwardTokens(&[_]u32{next});
            next = try sampleGreedy(ctx, logits);
            logits.deinit();
        }
        const total_ns = wall.read();
        std.debug.print("\n=== baseline: {d} tokens, {d:.2} ms/token ({d:.1} tok/s) ===\n", .{
            n_decode,
            @as(f64, @floatFromInt(total_ns)) / 1e6 / @as(f64, @floatFromInt(n_decode)),
            1e9 * @as(f64, @floatFromInt(n_decode)) / @as(f64, @floatFromInt(total_ns)),
        });
    }

    // ================= apply-level timing (DMLX_PROF_APPLY=1) =================
    if (std.c.getenv("DMLX_PROF_APPLY")) |v| {
        if (v[0] != 0 and v[0] != '0') {
            var logits = try model.forwardTokens(prompt.items); // prefill (resets state)
            var next = try sampleGreedy(ctx, logits);
            logits.deinit();
            var acc = ApplyAccum{};
            for (0..3) |_| { // warmup (also primes the compiled closures)
                logits = try forwardStepApplyProf(&model, next, &acc);
                next = try sampleGreedy(ctx, logits);
                logits.deinit();
            }
            acc = .{};
            var wall = Lap.start();
            var i: usize = 0;
            while (i < n_decode) : (i += 1) {
                logits = try forwardStepApplyProf(&model, next, &acc);
                acc.tokens += 1;
                next = try sampleGreedy(ctx, logits);
                logits.deinit();
            }
            acc.report(wall.read());
        }
    }

    // ================= breakdown: instrumented step =================
    if (run_breakdown) {
        var logits = try model.forwardTokens(prompt.items); // prefill (resets state)
        var next = try sampleGreedy(ctx, logits);
        logits.deinit();
        var acc = Accum{};
        var t = Lap.start();
        // warmup (also excluded)
        for (0..3) |_| {
            logits = try forwardStepProfiled(&model, next, &acc);
            next = try sampleGreedy(ctx, logits);
            logits.deinit();
        }
        acc = .{};
        var i: usize = 0;
        while (i < n_decode) : (i += 1) {
            logits = try forwardStepProfiled(&model, next, &acc);
            acc.tokens += 1;
            t.reset();
            next = try sampleGreedy(ctx, logits);
            acc.sample_ns += t.lap();
            logits.deinit();
        }
        acc.report("breakdown (barrier per section — overstates wall)");
    }

    // ============ prefill breakdown: where the L>1 pass goes ============
    if (run_prefill_bd) {
        // One cold pass to warm kernels/PSOs, then the measured pass. State
        // is reset by each full prefill, so successive passes are equivalent.
        {
            var warm_acc = Accum{};
            const warm = try forwardPrefillProfiled(&model, prompt.items, &warm_acc);
            warm.deinit();
        }
        var acc = Accum{};
        var wall = Lap.start();
        var logits = try forwardPrefillProfiled(&model, prompt.items, &acc);
        const wall_ns = wall.read();
        logits.deinit();
        acc.tokens = 1;
        acc.report("prefill breakdown (barrier per section — overstates wall)");
        qwen38_gdn.gdnPrefillProfReport();
        std.debug.print("  prompt tokens: {d} | barriered pass {d:.0} ms | unbarriered wall for reference: run compare_engines.py\n", .{
            prompt.items.len, @as(f64, @floatFromInt(wall_ns)) / 1e6,
        });
    }
    // ================= buildsplit: CPU build vs GPU eval =================
    if (run_buildsplit) {
        var logits = try model.forwardTokens(prompt.items); // prefill (resets state)
        var next = try sampleGreedy(ctx, logits);
        logits.deinit();
        var build_ns: u64 = 0;
        var eval_ns: u64 = 0;
        for (0..3) |_| {
            logits = try forwardStepBuildSplit(&model, next, &build_ns, &eval_ns);
            next = try sampleGreedy(ctx, logits);
            logits.deinit();
        }
        build_ns = 0;
        eval_ns = 0;
        var wall = Lap.start();
        var i: usize = 0;
        while (i < n_decode) : (i += 1) {
            logits = try forwardStepBuildSplit(&model, next, &build_ns, &eval_ns);
            next = try sampleGreedy(ctx, logits);
            logits.deinit();
        }
        const total_ns = wall.read();
        const per = @as(f64, @floatFromInt(n_decode));
        std.debug.print("\n=== buildsplit: {d} tokens, total {d:.2} ms/token | CPU graph-build {d:.2} ms ({d:.0}%) | GPU eval-wait {d:.2} ms ({d:.0}%) | sampling+rest {d:.2} ms ===\n", .{
            n_decode,
            @as(f64, @floatFromInt(total_ns)) / 1e6 / per,
            @as(f64, @floatFromInt(build_ns)) / 1e6 / per,
            100.0 * @as(f64, @floatFromInt(build_ns)) / @as(f64, @floatFromInt(total_ns)),
            @as(f64, @floatFromInt(eval_ns)) / 1e6 / per,
            100.0 * @as(f64, @floatFromInt(eval_ns)) / @as(f64, @floatFromInt(total_ns)),
            @as(f64, @floatFromInt(total_ns -| build_ns -| eval_ns)) / 1e6 / per,
        });
    }

    const peak = memory_mod.getPeakMemory() catch 0;
    std.debug.print("peak MLX memory: {d:.2} GB\n", .{@as(f64, @floatFromInt(peak)) / 1e9});
}
