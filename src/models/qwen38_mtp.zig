//! Qwen3.8 MTP (multi-token prediction) head — speculative drafter module.
//!
//! Structure (HF qwen3_5 MTP / omlx mlx_lm_mtp qwen35_model.py):
//!   e = pre_fc_norm_embedding(embed(t_next))
//!   h = pre_fc_norm_hidden(backbone pre-final-norm hidden at t)
//!   fused = fc(concat([e, h], -1))          // Linear 2H -> H
//!   fused = decoder_layer(fused)            // full attention (own KV cache)
//!                                           // + SwiGLU MLP, residual pairs
//!   out = norm(fused)                       // final RMSNorm
//!   logits = shared_lm_head(out)
//!
//! Chaining: the NEXT MTP pass consumes `out` (the POST-final-norm decoder
//! output) as its `hidden_pre_norm` argument — omlx's `mtp_forward(
//! return_hidden=true)` returns `self.mtp(...)` which is the post-norm
//! tensor, and the head re-norms it through pre_fc_norm_hidden on entry.
//!
//! The head shares the backbone's embedding table and lm_head
//! (mtp_use_dedicated_embeddings=false, tie_word_embeddings=false).
//! Task 1 scope: module + drafter weight loading + toy-dims tests. The
//! speculative loop (engine_loop) and backbone verify mode are Tasks 2/3.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const reduce_mod = @import("mlx").reduce;
const fast_mod = @import("mlx").fast;
const activations = @import("mlx").activations;
const quantize_mod = @import("mlx").quantize;
const sort_mod = @import("mlx").sort;
const safetensors_reader = @import("mlx").safetensors_reader;
const glm53 = @import("glm53.zig");

const qwen38_attn = @import("qwen38_attn.zig");
const qwen38_gdn = @import("qwen38_gdn.zig");
const closure_mod = @import("mlx").closure;
const compile_mod = @import("mlx").compile;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const Closure = closure_mod.Closure;
const QuantConfig = quantize_mod.QuantConfig;
const WeightSource = glm53.WeightSource;

/// Architectural dims shared with the backbone (the MTP decoder layer is a
/// full-attention Qwen3.8 layer).
pub const Qwen38MtpDims = struct {
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dims: usize,
    intermediate_size: usize,
    rope_theta: f32,
    rms_norm_eps: f32 = 1e-6,
    output_gate: bool = true,
};

/// A linear projection with optional affine quantization (scales present iff
/// the checkpoint quantizes this tensor; the community MTP checkpoint
/// quantizes everything at 4-bit gs=64, but omlx keeps mtp.fc full precision
/// — handle both). `cfg` is only consumed by Qwen38Attn's internal linear.
pub const MtpLinear = struct {
    w: Array,
    scales: ?Array = null,
    biases: ?Array = null,
    cfg: QuantConfig = .{},

    pub fn apply(self: *const MtpLinear, ctx: EagerContext, x: Array) !Array {
        return qwen38_gdn.linearQuantParts(ctx, x, self.w, self.scales, self.biases);
    }

    fn deinit(self: *const MtpLinear) void {
        self.w.deinit();
        if (self.scales) |s| s.deinit();
        if (self.biases) |b| b.deinit();
    }
};

// ============================================================
// Compiled decode-step segments (launch-fusion program)
// ============================================================
//
// The eager drafter step (forwardBodyRaw + final norm) issues ~30 MLX op
// dispatches per token: gather, 5 RMSNorms at 3 dispatches each (astype /
// rmsNorm / astype), concat, 8 quantized projections, 2 q/k norms, 2 ropes,
// sigmoid, 2 cache sliceUpdates, SDPA, gate multiply, 2 residual adds, silu,
// multiply, copy. GPU time per step is dominated by the coarse-head read
// (~0.4 ms), so the step is launch-bound; mx.compile fuses the elementwise
// chains and collapses the per-op CPU dispatch cost (same precedent as the
// backbone's compiled GDN step / MLP swiglu / attention pre-cache).
//
// Two closures bracket the stateful middle (KV slab append + custom decode
// SDPA stay eager):
//   pre : id, embed, hidden, norms, fc, ln1, qkv weights, q/k_norm, offset
//         -> [fused, q_t, k_t, v_t, gate_sig]
//   post: attn, gate_sig, fused, o, ln2, gate/up/down weights, norm
//         -> [normed, dec(pre-norm)]
// Trace-time constants (rotary dims / theta / eps, output gate) are baked;
// the compiled path is gated on the serving config matching them. Kill
// switch: DMLX_MTP_NO_COMPILE=1. DMLX_MTP_STAGE forces the eager path so the
// stage checkpoints stay observable.

const MtpPreSegOut = struct {
    fused: Array, // [1, hidden] bf16 (residual for the post segment)
    q_t: Array, // [1, H, 1, D] roped
    k_t: Array, // [1, HKV, 1, D] roped
    v_t: Array, // [1, HKV, 1, D]
    gate_sig: Array, // [1, H*D] sigmoid(output gate)
};

/// Pre-attention segment, pure function of arrays. Mirrors the eager op
/// sequence of forwardBodyRaw up to the KV-cache read exactly (same order,
/// same dtypes).
fn mtpPreSeg(
    ctx: EagerContext,
    id_arr: Array, // [1] i32
    embed_table: Array, // [vocab, hidden] bf16
    hidden_in: Array, // [1, hidden] bf16
    w_e: Array, // pre_fc_norm_embedding, [hidden] f32
    w_h: Array, // pre_fc_norm_hidden, [hidden] f32
    fc: MtpLinear,
    ln1: Array, // [hidden] f32
    qw: qwen38_attn.PreCacheWeights, // .x is overridden with n1 below
    rotary_dims: i32,
    rope_theta: f32,
    eps: f32,
) !MtpPreSegOut {
    const row = try shape_mod.takeAxis(ctx, embed_table, id_arr, 0);
    defer row.deinit();
    const e = try rmsNormBf16(ctx, row, w_e, eps);
    defer e.deinit();
    const h = try rmsNormBf16(ctx, hidden_in, w_h, eps);
    defer h.deinit();
    const cat = try shape_mod.concatenateAxis(ctx, &[_]Array{ e, h }, 1);
    defer cat.deinit();
    const fused = try fc.apply(ctx, cat);
    errdefer fused.deinit();
    const n1 = try rmsNormBf16(ctx, fused, ln1, eps);
    defer n1.deinit();

    var w = qw;
    w.x = n1;
    var pre = try qwen38_attn.preCacheStep(ctx, w, rotary_dims, rope_theta, eps, true);
    errdefer pre.deinit();
    return .{
        .fused = fused,
        .q_t = pre.q_t,
        .k_t = pre.k_t,
        .v_t = pre.v_t,
        .gate_sig = pre.gate_sig.?,
    };
}

const MtpPostSegOut = struct {
    normed: Array, // [1, hidden] bf16 (final-normed; feeds lm_head / chaining)
    dec: Array, // [1, hidden] bf16 pre-final-norm (CHAIN_PRENORM experiment)
};

/// Same pair, named at module scope so the eager and compiled step drivers
/// share one return type.
pub const MtpBodyPair = struct {
    normed: Array,
    pre_norm: Array,
};

/// Post-attention segment: gate multiply, o_proj, both residual adds, the
/// SwiGLU MLP and the final norm. Mirrors forwardBodyRaw/forwardBodyIds
/// exactly.
fn mtpPostSeg(
    ctx: EagerContext,
    attn4: Array, // [1, H, 1, D] bf16 (decode SDPA output)
    gate_sig: Array, // [1, H*D] bf16
    fused: Array, // [1, hidden] bf16 residual
    o: MtpLinear,
    ln2: Array, // [hidden] f32
    gate: MtpLinear,
    up: MtpLinear,
    down: MtpLinear,
    norm_w: Array, // [hidden] f32
    eps: f32,
) !MtpPostSegOut {
    const hd: i32 = gate_sig.shape()[1];
    const attn_t = try ops.squeeze(ctx, attn4);
    defer attn_t.deinit();
    const attn_flat = try ops.reshape(ctx, attn_t, &[_]i32{ 1, hd });
    defer attn_flat.deinit();
    const gated = try ops.multiply(ctx, attn_flat, gate_sig);
    defer gated.deinit();
    const ao = try o.apply(ctx, gated);
    defer ao.deinit();
    const h2 = try ops.add(ctx, fused, ao);
    defer h2.deinit();

    const n2 = try rmsNormBf16(ctx, h2, ln2, eps);
    defer n2.deinit();
    const g = try gate.apply(ctx, n2);
    defer g.deinit();
    const u = try up.apply(ctx, n2);
    defer u.deinit();
    const act = try activations.silu(ctx, g);
    defer act.deinit();
    const hh = try ops.multiply(ctx, act, u);
    defer hh.deinit();
    const mlp_out = try down.apply(ctx, hh);
    defer mlp_out.deinit();
    const dec = try ops.add(ctx, h2, mlp_out);
    errdefer dec.deinit();
    const normed = try rmsNormBf16(ctx, dec, norm_w, eps);
    return .{ .normed = normed, .dec = dec };
}

/// Closures are traced on the GPU stream so the compiled graph dispatches to
/// the same device as the eager path (same convention as qwen38_attn).
fn mtpClosureGpuCtx(allocator: std.mem.Allocator) EagerContext {
    return EagerContext.initWithStream(allocator, .{ .inner = c.c.mlx_default_gpu_stream_new() });
}

fn preSegToSlice(res: MtpPreSegOut, allocator: std.mem.Allocator) error{MlxError}![]Array {
    const out = allocator.alloc(Array, 5) catch return error.MlxError;
    out[0] = res.fused;
    out[1] = res.q_t;
    out[2] = res.k_t;
    out[3] = res.v_t;
    out[4] = res.gate_sig;
    return out;
}

fn postSegToSlice(res: MtpPostSegOut, allocator: std.mem.Allocator) error{MlxError}![]Array {
    const out = allocator.alloc(Array, 2) catch return error.MlxError;
    out[0] = res.normed;
    out[1] = res.dec;
    return out;
}

/// Quantized real-config pre segment. Input order (21): id, embed, hidden,
/// w_e, w_h, fc(w,s,b), ln1, q(w,s,b), k(w,s,b), v(w,s,b), q_norm, k_norm,
/// offset.
fn mtpPreClosureQuant(inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array {
    if (inputs.len != 21) return error.MlxError;
    const ctx = mtpClosureGpuCtx(allocator);
    defer ctx.deinit();
    const res = mtpPreSeg(ctx, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], .{
        .w = inputs[5],
        .scales = inputs[6],
        .biases = inputs[7],
    }, inputs[8], .{
        .x = inputs[2], // overridden with n1 inside mtpPreSeg
        .q_w = inputs[9],
        .q_scales = inputs[10],
        .q_biases = inputs[11],
        .k_w = inputs[12],
        .k_scales = inputs[13],
        .k_biases = inputs[14],
        .v_w = inputs[15],
        .v_scales = inputs[16],
        .v_biases = inputs[17],
        .q_norm = inputs[18],
        .k_norm = inputs[19],
        .offset = inputs[20],
    }, qwen38_attn.COMPILED_ROTARY_DIMS, qwen38_attn.COMPILED_ROPE_THETA, qwen38_attn.COMPILED_RMS_EPS) catch return error.MlxError;
    return preSegToSlice(res, allocator);
}

/// Quantized real-config post segment. Input order (17): attn, gate_sig,
/// fused, o(w,s,b), ln2, gate(w,s,b), up(w,s,b), down(w,s,b), norm.
fn mtpPostClosureQuant(inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array {
    if (inputs.len != 17) return error.MlxError;
    const ctx = mtpClosureGpuCtx(allocator);
    defer ctx.deinit();
    const res = mtpPostSeg(ctx, inputs[0], inputs[1], inputs[2], .{
        .w = inputs[3],
        .scales = inputs[4],
        .biases = inputs[5],
    }, inputs[6], .{
        .w = inputs[7],
        .scales = inputs[8],
        .biases = inputs[9],
    }, .{
        .w = inputs[10],
        .scales = inputs[11],
        .biases = inputs[12],
    }, .{
        .w = inputs[13],
        .scales = inputs[14],
        .biases = inputs[15],
    }, inputs[16], qwen38_attn.COMPILED_RMS_EPS) catch return error.MlxError;
    return postSegToSlice(res, allocator);
}

/// Build the mx.compile'd drafter pre-attention segment (real quantized
/// config). Trace happens on first apply.
fn compiledMtpStepPre(allocator: std.mem.Allocator) !Closure {
    const base = try Closure.init(&mtpPreClosureQuant, allocator);
    defer base.deinit();
    return compile_mod.compile(base, false);
}

fn compiledMtpStepPost(allocator: std.mem.Allocator) !Closure {
    const base = try Closure.init(&mtpPostClosureQuant, allocator);
    defer base.deinit();
    return compile_mod.compile(base, false);
}

pub const MtpForwardOut = struct {
    logits: Array, // [1, 1, vocab] f32
    /// Post-final-norm decoder output [1, hidden] bf16 — feed this as
    /// `hidden_pre_norm` to chain the next MTP pass.
    hidden_out: Array,
};

pub const Qwen38Mtp = struct {
    ctx: EagerContext,
    dims: Qwen38MtpDims,

    pre_fc_norm_hidden_f32: Array, // [hidden] f32, evaled
    pre_fc_norm_embedding_f32: Array,
    norm_f32: Array, // final norm
    fc: MtpLinear, // 2H -> H

    ln1_f32: Array, // decoder layer input_layernorm
    ln2_f32: Array, // decoder layer post_attention_layernorm
    attn: qwen38_attn.Qwen38Attn,
    mlp_gate: MtpLinear,
    mlp_up: MtpLinear,
    mlp_down: MtpLinear,

    /// The head's own KV cache (one full-attention layer). Grows by one per
    /// accepted token; rollback is a length truncation (Task 3).
    cache: qwen38_attn.Qwen38AttnCache = .{},

    /// mx.compile'd decode-step segments (see the "Compiled decode-step
    /// segments" section above). null = eager fallback. Traced lazily on the
    /// first compiled apply.
    step_pre_compiled: ?Closure = null,
    step_post_compiled: ?Closure = null,

    // --- coarse draft head (plan doc §28.4 / §32) --------------------------
    // A 3-bit gs128 copy of the shared lm_head, used ONLY to propose draft
    // tokens. Built lazily on first use; `coarse_state` makes the attempt
    // once so a failure costs nothing per step.
    coarse_head: ?MtpLinear = null,
    coarse_state: enum { untried, ready, failed } = .untried,

    /// Prompt pass: run the head over the prompt so the drafter's KV cache
    /// covers the confirmed sequence with correct RoPE offsets. Inputs follow
    /// the head's shifted convention (omlx/HF): head position j fuses
    /// backbone hidden row j with embed(token_ids[j]); pass
    /// `hidden_rows = hidden_pre_norm_all[0 .. L-2]` and
    /// `token_ids = prompt[1..]`. Attention runs prefill-mode causal over the
    /// head's own cache (offset = current cache len). Returns the last
    /// position's post-norm hidden [1, hidden] bf16 (chaining input for the
    /// first draft) — the caller's first draft still uses the BACKBONE
    /// last-position hidden per the omlx convention.
    pub fn prefill(
        self: *Qwen38Mtp,
        allocator: std.mem.Allocator,
        hidden_rows: Array, // [L-1, hidden] bf16
        token_ids: []const u32, // [L-1]
        embed_table_bf16: Array, // shared backbone embed table
    ) !Array {
        const ctx = self.ctx;
        const hidden: i32 = @intCast(self.dims.hidden_size);
        const L: i32 = @intCast(token_ids.len);
        const eps = self.dims.rms_norm_eps;
        if (L < 1) return error.MtpPrefillEmpty;

        const ids_i32 = try allocator.alloc(i32, token_ids.len);
        defer allocator.free(ids_i32);
        for (token_ids, 0..) |t, i| ids_i32[i] = @intCast(t);
        const ids_arr = try Array.fromData(allocator, i32, ids_i32, &[_]i32{L});
        defer ids_arr.deinit();

        // e = normE(embed(tokens)); h = normH(hidden rows) — positionwise.
        const rows = try shape_mod.takeAxis(ctx, embed_table_bf16, ids_arr, 0); // [L-1, hidden] bf16
        defer rows.deinit();
        const e = try rmsNormBf16(ctx, rows, self.pre_fc_norm_embedding_f32, eps);
        defer e.deinit();
        const h = try rmsNormBf16(ctx, hidden_rows, self.pre_fc_norm_hidden_f32, eps);
        defer h.deinit();

        // fused = fc(concat([e, h], -1))  [L-1, hidden]
        const cat = try shape_mod.concatenateAxis(ctx, &[_]Array{ e, h }, 1);
        defer cat.deinit();
        stageChk("cat", ctx, cat);
        const fused = try self.fc.apply(ctx, cat);
        defer fused.deinit();
        stageChk("fused", ctx, fused);

        // Decoder layer over the sequence (prefill-mode causal attention).
        const fused3 = try ops.reshape(ctx, fused, &[_]i32{ 1, L, hidden });
        defer fused3.deinit();
        const n1 = try rmsNormBf16(ctx, fused3, self.ln1_f32, eps);
        defer n1.deinit();
        const attn_out = try self.attn.forward(n1, &self.cache); // [1, L-1, hidden]
        defer attn_out.deinit();
        const h2 = try ops.add(ctx, fused3, attn_out);
        defer h2.deinit();

        const n2 = try rmsNormBf16(ctx, h2, self.ln2_f32, eps);
        defer n2.deinit();
        const gate = try self.mlp_gate.apply(ctx, n2);
        defer gate.deinit();
        const up = try self.mlp_up.apply(ctx, n2);
        defer up.deinit();
        const act = try activations.silu(ctx, gate);
        defer act.deinit();
        const hh = try ops.multiply(ctx, act, up);
        defer hh.deinit();
        const mlp_out = try self.mlp_down.apply(ctx, hh);
        defer mlp_out.deinit();
        stageChk("mlp", ctx, mlp_out);
        const dec_out = try ops.add(ctx, h2, mlp_out);
        defer dec_out.deinit();

        const normed = try rmsNormBf16(ctx, dec_out, self.norm_f32, eps);
        defer normed.deinit();
        const last = try ops.slice(ctx, normed, &[_]i32{ 0, L - 1, 0 }, &[_]i32{ 1, L, hidden }, &[_]i32{ 1, 1, 1 });
        defer last.deinit();
        const flat = try ops.reshape(ctx, last, &[_]i32{ 1, hidden });
        defer flat.deinit();
        return ops.copy(ctx, flat);
    }

    pub fn deinit(self: *Qwen38Mtp) void {
        self.pre_fc_norm_hidden_f32.deinit();
        self.pre_fc_norm_embedding_f32.deinit();
        self.norm_f32.deinit();
        self.fc.deinit();
        self.ln1_f32.deinit();
        self.ln2_f32.deinit();
        self.attn.deinit();
        self.mlp_gate.deinit();
        self.mlp_up.deinit();
        self.mlp_down.deinit();
        if (self.coarse_head) |ch| ch.deinit();
        if (self.step_pre_compiled) |cc| cc.deinit();
        if (self.step_post_compiled) |cc| cc.deinit();
        self.cache.deinit();
    }

    /// NOTE (verified against mlx-vlm's qwen3_5_mtp drafter sanitize): HF
    /// `mtp.*`-prefixed layouts store RMSNorm weights as (scale − 1) deltas
    /// and consumers add 1.0 — but the community drafter checkpoint uses the
    /// FLAT mlx layout, where norms are already scale-encoded and NO shift
    /// applies. Do not "fix" this by adding 1.0 (measured: it drops
    /// acceptance from ~85% to ~63%).
    fn loadNormF32(ctx: EagerContext, ws: WeightSource, name: []const u8) !Array {
        const w = try ws.tensor(name);
        defer w.deinit();
        const w_f32 = try ops.astype(ctx, w, .float32);
        errdefer w_f32.deinit();
        try w_f32.eval();
        return w_f32;
    }

    fn loadLinear(ws: WeightSource, prefix: []const u8, in_features: i32) !MtpLinear {
        var nbuf: [256]u8 = undefined;
        const w_name = try std.fmt.bufPrint(&nbuf, "{s}.weight", .{prefix});
        const w = try ws.tensor(w_name);
        errdefer w.deinit();
        var nbuf2: [256]u8 = undefined;
        const scales = ws.tensor(std.fmt.bufPrint(&nbuf2, "{s}.scales", .{prefix}) catch unreachable) catch null;
        errdefer if (scales) |s| s.deinit();
        var nbuf3: [256]u8 = undefined;
        const biases = if (scales != null)
            (ws.tensor(std.fmt.bufPrint(&nbuf3, "{s}.biases", .{prefix}) catch unreachable) catch null)
        else
            null;
        errdefer if (biases) |b| b.deinit();
        var cfg: QuantConfig = .{};
        if (scales) |s| {
            const wshape = w.shape();
            const sshape = s.shape();
            cfg = .{
                .bits = @intCast(@divTrunc(wshape[wshape.len - 1] * 32, in_features)),
                .group_size = @intCast(@divTrunc(in_features, sshape[sshape.len - 1])),
                .mode = .affine,
            };
        }
        return .{ .w = w, .scales = scales, .biases = biases, .cfg = cfg };
    }

    /// Load from a WeightSource (checkpoint tensor namespace is flat:
    /// `fc.*`, `layers.0.self_attn.*`, `layers.0.mlp.*`, `norm.weight`,
    /// `pre_fc_norm_hidden.weight`, `pre_fc_norm_embedding.weight`).
    pub fn loadFromSource(ctx: EagerContext, ws: WeightSource, dims: Qwen38MtpDims) !Qwen38Mtp {
        const hidden: i32 = @intCast(dims.hidden_size);
        const intermediate: i32 = @intCast(dims.intermediate_size);

        const nh = try loadNormF32(ctx, ws, "pre_fc_norm_hidden.weight");
        errdefer nh.deinit();
        const ne = try loadNormF32(ctx, ws, "pre_fc_norm_embedding.weight");
        errdefer ne.deinit();
        const nf = try loadNormF32(ctx, ws, "norm.weight");
        errdefer nf.deinit();
        const ln1 = try loadNormF32(ctx, ws, "layers.0.input_layernorm.weight");
        errdefer ln1.deinit();
        const ln2 = try loadNormF32(ctx, ws, "layers.0.post_attention_layernorm.weight");
        errdefer ln2.deinit();
        const fc = try loadLinear(ws, "fc", 2 * hidden);
        errdefer fc.deinit();
        const gate = try loadLinear(ws, "layers.0.mlp.gate_proj", hidden);
        errdefer gate.deinit();
        const up = try loadLinear(ws, "layers.0.mlp.up_proj", hidden);
        errdefer up.deinit();
        const down = try loadLinear(ws, "layers.0.mlp.down_proj", intermediate);
        errdefer down.deinit();

        const sa = "layers.0.self_attn";
        const q_l = try loadLinear(ws, sa ++ ".q_proj", hidden);
        errdefer q_l.deinit();
        const k_l = try loadLinear(ws, sa ++ ".k_proj", hidden);
        errdefer k_l.deinit();
        const v_l = try loadLinear(ws, sa ++ ".v_proj", hidden);
        errdefer v_l.deinit();
        const o_l = try loadLinear(ws, sa ++ ".o_proj", @intCast(dims.num_heads * dims.head_dim));
        errdefer o_l.deinit();
        // q_norm/k_norm stay in checkpoint dtype to mirror nn.RMSNorm (f32
        // internal, input-dtype out) — same convention as the backbone.
        const q_norm = try ws.tensor(sa ++ ".q_norm.weight");
        errdefer q_norm.deinit();
        const k_norm = try ws.tensor(sa ++ ".k_norm.weight");
        errdefer k_norm.deinit();

        // Nothing below can fail, so the errdefers above never fire after
        // ownership moves into the struct.
        return .{
            .ctx = ctx,
            .dims = dims,
            .pre_fc_norm_hidden_f32 = nh,
            .pre_fc_norm_embedding_f32 = ne,
            .norm_f32 = nf,
            .fc = fc,
            .ln1_f32 = ln1,
            .ln2_f32 = ln2,
            .attn = .{
                .ctx = ctx,
                .hidden_size = dims.hidden_size,
                .num_heads = dims.num_heads,
                .num_kv_heads = dims.num_kv_heads,
                .head_dim = dims.head_dim,
                .rotary_dims = dims.rotary_dims,
                .rope_theta = dims.rope_theta,
                .rms_norm_eps = dims.rms_norm_eps,
                .output_gate = dims.output_gate,
                .q_proj = q_l.w,
                .k_proj = k_l.w,
                .v_proj = v_l.w,
                .o_proj = o_l.w,
                .q_norm = q_norm,
                .k_norm = k_norm,
                .q_scales = q_l.scales,
                .q_biases = q_l.biases,
                .q_quant = q_l.cfg,
                .k_scales = k_l.scales,
                .k_biases = k_l.biases,
                .k_quant = k_l.cfg,
                .v_scales = v_l.scales,
                .v_biases = v_l.biases,
                .v_quant = v_l.cfg,
                .o_scales = o_l.scales,
                .o_biases = o_l.biases,
                .o_quant = o_l.cfg,
            },
            .mlp_gate = gate,
            .mlp_up = up,
            .mlp_down = down,
        };
    }

    /// The compiled decode step is traced for one fixed config: fully
    /// quantized (4-bit affine) linears, rotary dims/theta/eps matching the
    /// baked closure constants, output gate on. Anything else stays eager.
    fn compiledStepEligible(self: *const Qwen38Mtp) bool {
        if (std.c.getenv("DMLX_MTP_NO_COMPILE") != null) return false;
        const d = self.dims;
        if (!d.output_gate) return false;
        if (d.rotary_dims != qwen38_attn.COMPILED_ROTARY_DIMS) return false;
        if (@abs(d.rope_theta - qwen38_attn.COMPILED_ROPE_THETA) >= 1.0) return false;
        if (@abs(d.rms_norm_eps - qwen38_attn.COMPILED_RMS_EPS) >= 1e-12) return false;
        const lins = [_]MtpLinear{ self.fc, self.mlp_gate, self.mlp_up, self.mlp_down };
        for (lins) |l| {
            if (l.scales == null or l.biases == null) return false;
        }
        const a = &self.attn;
        if (a.q_scales == null or a.q_biases == null or
            a.k_scales == null or a.k_biases == null or
            a.v_scales == null or a.v_biases == null or
            a.o_scales == null or a.o_biases == null) return false;
        return true;
    }

    fn initCompiledStep(self: *Qwen38Mtp, allocator: std.mem.Allocator) void {
        if (!self.compiledStepEligible()) return;
        const pre = compiledMtpStepPre(allocator) catch {
            std.log.warn("[MTP] compiled pre-attention segment build failed — eager drafter step", .{});
            return;
        };
        const post = compiledMtpStepPost(allocator) catch {
            std.log.warn("[MTP] compiled post-attention segment build failed — eager drafter step", .{});
            pre.deinit();
            return;
        };
        self.step_pre_compiled = pre;
        self.step_post_compiled = post;
        std.log.info("[MTP] compiled drafter decode step active", .{});
    }

    /// Load from a drafter checkpoint directory (safetensors + index).
    pub fn load(allocator: std.mem.Allocator, ctx: EagerContext, model_dir: []const u8, dims: Qwen38MtpDims) !Qwen38Mtp {
        var index = try safetensors_reader.loadOrBuildIndex(allocator, model_dir);
        defer index.deinit();
        var mtp = try loadFromSource(ctx, WeightSource.fromIndex(&index), dims);
        errdefer mtp.deinit();
        mtp.initCompiledStep(allocator);
        return mtp;
    }

    /// One MTP decode step. `hidden_pre_norm`: [1, hidden] bf16 (backbone
    /// pre-final-norm hidden for pass 1; this head's own `hidden_out` for
    /// chained passes). Advances the head's own KV cache by one position.
    /// Returns lazy arrays; the caller syncs on logits.
    /// `lm_head`: the backbone's shared lm_head — from a
    /// `glm53_attn.QuantLinear` `ql`, pass
    /// `.{ .w = ql.w, .scales = ql.scales, .biases = ql.biases }`.
    pub fn forward(
        self: *Qwen38Mtp,
        allocator: std.mem.Allocator,
        hidden_pre_norm: Array,
        token_id: u32,
        embed_table_bf16: Array,
        lm_head: MtpLinear,
    ) !MtpForwardOut {
        const id_arr = try Array.fromData(allocator, i32, &[_]i32{@intCast(token_id)}, &[_]i32{1});
        defer id_arr.deinit();
        return self.forwardIds(allocator, hidden_pre_norm, id_arr, embed_table_bf16, lm_head);
    }

    /// As forward(), but the token id arrives as a GPU [1] i32 array — e.g.
    /// the previous step's argmax, keeping a greedy draft chain fully on GPU.
    pub fn forwardIds(
        self: *Qwen38Mtp,
        allocator: std.mem.Allocator,
        hidden_pre_norm: Array,
        id_arr: Array, // [1] i32
        embed_table_bf16: Array,
        lm_head: MtpLinear,
    ) !MtpForwardOut {
        const ctx = self.ctx;
        const hidden_out = try self.forwardBodyIds(allocator, hidden_pre_norm, id_arr, embed_table_bf16);
        errdefer hidden_out.deinit();
        const h3 = try ops.reshape(ctx, hidden_out, &[_]i32{ 1, 1, @as(i32, @intCast(self.dims.hidden_size)) });
        defer h3.deinit();
        const logits_bf16 = try lm_head.apply(ctx, h3);
        defer logits_bf16.deinit();
        const logits = try ops.astype(ctx, logits_bf16, .float32);
        errdefer logits.deinit();
        return .{ .logits = logits, .hidden_out = hidden_out };
    }

    /// forwardIds minus the lm_head: returns the final-normed hidden row
    /// ([1, hidden] bf16) that feeds both lm_head and the next chained step.
    fn forwardBodyRaw(
        self: *Qwen38Mtp,
        allocator: std.mem.Allocator,
        hidden_pre_norm: Array,
        id_arr: Array, // [1] i32
        embed_table_bf16: Array,
    ) !Array {
        _ = allocator; // id array is supplied by the caller
        const ctx = self.ctx;
        const eps = self.dims.rms_norm_eps;

        // e = normE(embed(token)); h = normH(hidden)
        const row = try shape_mod.takeAxis(ctx, embed_table_bf16, id_arr, 0); // [1, hidden] bf16
        defer row.deinit();
        const e = try rmsNormBf16(ctx, row, self.pre_fc_norm_embedding_f32, eps);
        defer e.deinit();
        stageChk("e", ctx, e);
        const h = try rmsNormBf16(ctx, hidden_pre_norm, self.pre_fc_norm_hidden_f32, eps);
        defer h.deinit();
        stageChk("hn", ctx, h);

        // fused = fc(concat([e, h], -1))  [1, hidden]
        const cat = try shape_mod.concatenateAxis(ctx, &[_]Array{ e, h }, 1);
        defer cat.deinit();
        stageChk("cat", ctx, cat);
        const fused = try self.fc.apply(ctx, cat);
        defer fused.deinit();
        stageChk("fused", ctx, fused);

        // Decoder layer: attn block + MLP block, both residual.
        const n1 = try rmsNormBf16(ctx, fused, self.ln1_f32, eps);
        defer n1.deinit();
        stageChk("n1", ctx, n1);
        const attn_out = try self.attn.forward(n1, &self.cache); // [1, hidden]
        defer attn_out.deinit();
        stageChk("attn", ctx, attn_out);
        const h2 = try ops.add(ctx, fused, attn_out);
        defer h2.deinit();

        const n2 = try rmsNormBf16(ctx, h2, self.ln2_f32, eps);
        defer n2.deinit();
        const gate = try self.mlp_gate.apply(ctx, n2);
        defer gate.deinit();
        const up = try self.mlp_up.apply(ctx, n2);
        defer up.deinit();
        const act = try activations.silu(ctx, gate);
        defer act.deinit();
        const hh = try ops.multiply(ctx, act, up);
        defer hh.deinit();
        const mlp_out = try self.mlp_down.apply(ctx, hh);
        defer mlp_out.deinit();
        stageChk("mlp", ctx, mlp_out);
        const dec_out = try ops.add(ctx, h2, mlp_out);
        defer dec_out.deinit();

        // Final norm → hidden_out (this tensor feeds both lm_head and, in
        // chained drafting, the next pass's pre_fc_norm_hidden).
        stageChk("dec", ctx, dec_out);
        return ops.copy(ctx, dec_out);
    }

    /// Body-only step for a CPU-side token id. Used by the commit replay,
    /// which needs nothing but the KV-cache side effect: it discarded the
    /// logits immediately, yet still built the full-vocab lm_head node
    /// (248320 x 5120 at 4 bits = 715 MB, 3.1 ms) on every replayed draft.
    /// Whether MLX's laziness already pruned that is measured, not assumed
    /// (plan doc §35).
    pub fn forwardBodyToken(
        self: *Qwen38Mtp,
        allocator: std.mem.Allocator,
        hidden_pre_norm: Array,
        token_id: u32,
        embed_table_bf16: Array,
    ) !Array {
        const id_arr = try Array.fromData(allocator, i32, &[_]i32{@intCast(token_id)}, &[_]i32{1});
        defer id_arr.deinit();
        return self.forwardBodyIds(allocator, hidden_pre_norm, id_arr, embed_table_bf16);
    }

    /// Compiled decode step: two mx.compile'd segments bracketing the eager
    /// KV-cache append + decode SDPA. Numerically identical to the eager
    /// forwardBodyRaw + final norm (same op sequence/dtypes; elementwise
    /// fusion only). Returns the post-norm hidden and the pre-norm decoder
    /// output, mirroring forwardBodyIdsPair.
    fn forwardBodyCompiled(
        self: *Qwen38Mtp,
        hidden_pre_norm: Array,
        id_arr: Array,
        embed_table_bf16: Array,
    ) !MtpBodyPair {
        const ctx = self.ctx;
        const a = &self.attn;
        const allocator = ctx.allocator;

        const offset_arr = try ops.scalarI32(ctx, self.cache.len());
        defer offset_arr.deinit();
        const pre_inputs = [_]Array{
            id_arr,                         embed_table_bf16,            hidden_pre_norm,
            self.pre_fc_norm_embedding_f32, self.pre_fc_norm_hidden_f32, self.fc.w,
            self.fc.scales.?,               self.fc.biases.?,            self.ln1_f32,
            a.q_proj,                       a.q_scales.?,                a.q_biases.?,
            a.k_proj,                       a.k_scales.?,                a.k_biases.?,
            a.v_proj,                       a.v_scales.?,                a.v_biases.?,
            a.q_norm,                       a.k_norm,                    offset_arr,
        };
        const pre_outs = self.step_pre_compiled.?.apply(&pre_inputs, allocator) catch |err| {
            // Pre-segment failure happens BEFORE the cache append, so the
            // eager fallback below sees an untouched cache.
            std.log.warn("[MTP] compiled pre segment apply failed ({s}); eager step", .{@errorName(err)});
            return self.forwardBodyIdsPairEager(allocator, hidden_pre_norm, id_arr, embed_table_bf16);
        };
        defer allocator.free(pre_outs);
        if (pre_outs.len != 5) {
            for (pre_outs) |o| o.deinit();
            return error.MlxError;
        }
        const fused = pre_outs[0];
        defer fused.deinit();
        const q_t = pre_outs[1];
        defer q_t.deinit();
        const k_t = pre_outs[2];
        defer k_t.deinit();
        const v_t = pre_outs[3];
        defer v_t.deinit();
        const gate_sig = pre_outs[4];
        defer gate_sig.deinit();

        // Stateful middle: slab append (mutates the cache) + decode SDPA.
        const attn = try a.cacheAppendAndSdpa(&self.cache, q_t, k_t, v_t);
        defer attn.deinit();

        const post_inputs = [_]Array{
            attn,                   gate_sig,        fused,
            a.o_proj,               a.o_scales.?,    a.o_biases.?,
            self.ln2_f32,           self.mlp_gate.w, self.mlp_gate.scales.?,
            self.mlp_gate.biases.?, self.mlp_up.w,   self.mlp_up.scales.?,
            self.mlp_up.biases.?,   self.mlp_down.w, self.mlp_down.scales.?,
            self.mlp_down.biases.?, self.norm_f32,
        };
        const post_outs = try self.step_post_compiled.?.apply(&post_inputs, allocator);
        defer allocator.free(post_outs);
        if (post_outs.len != 2) {
            for (post_outs) |o| o.deinit();
            return error.MlxError;
        }
        return .{ .normed = post_outs[0], .pre_norm = post_outs[1] };
    }

    fn compiledStepActive(self: *const Qwen38Mtp) bool {
        // DMLX_MTP_STAGE forces the eager path so the stage checkpoints stay
        // observable.
        return self.step_pre_compiled != null and self.step_post_compiled != null and
            std.c.getenv("DMLX_MTP_STAGE") == null;
    }

    /// Final-normed body output (the head's defined output; feeds lm_head).
    pub fn forwardBodyIds(
        self: *Qwen38Mtp,
        allocator: std.mem.Allocator,
        hidden_pre_norm: Array,
        id_arr: Array,
        embed_table_bf16: Array,
    ) !Array {
        if (self.compiledStepActive()) {
            const pair = try self.forwardBodyCompiled(hidden_pre_norm, id_arr, embed_table_bf16);
            pair.pre_norm.deinit();
            return pair.normed;
        }
        const dec = try self.forwardBodyRaw(allocator, hidden_pre_norm, id_arr, embed_table_bf16);
        defer dec.deinit();
        return rmsNormBf16(self.ctx, dec, self.norm_f32, self.dims.rms_norm_eps);
    }

    /// As forwardBodyIds, but also hands back the PRE-final-norm decoder
    /// output so a caller can chain on it.
    ///
    /// Why this exists: draft 1 consumes the BACKBONE's pre-final-norm hidden
    /// (the convention the head was trained on), while drafts 2..N consume the
    /// HEAD's post-final-norm output. RMSNorm makes the magnitude difference
    /// irrelevant, but post-norm is pre-norm reweighted channel-by-channel by
    /// `norm_f32`, so the two steps feed genuinely different distributions.
    /// Our measured conditional acceptance drops from p1 = 0.86 to
    /// p2 = 0.68 while omlx, on the same drafter weights, measures 0.76 ->
    /// 0.82 -> 0.83 -> 0.94 (plan doc §34). This is the cheapest testable
    /// candidate for that cliff: `DMLX_MTP_CHAIN_PRENORM=1` chains on the
    /// pre-norm output so every step sees the same kind of input.
    fn forwardBodyIdsPair(
        self: *Qwen38Mtp,
        allocator: std.mem.Allocator,
        hidden_pre_norm: Array,
        id_arr: Array,
        embed_table_bf16: Array,
    ) !MtpBodyPair {
        if (self.compiledStepActive())
            return self.forwardBodyCompiled(hidden_pre_norm, id_arr, embed_table_bf16);
        return self.forwardBodyIdsPairEager(allocator, hidden_pre_norm, id_arr, embed_table_bf16);
    }

    fn forwardBodyIdsPairEager(
        self: *Qwen38Mtp,
        allocator: std.mem.Allocator,
        hidden_pre_norm: Array,
        id_arr: Array,
        embed_table_bf16: Array,
    ) !MtpBodyPair {
        const ctx = self.ctx;
        const dec = try self.forwardBodyRaw(allocator, hidden_pre_norm, id_arr, embed_table_bf16);
        errdefer dec.deinit();
        const normed = try rmsNormBf16(ctx, dec, self.norm_f32, self.dims.rms_norm_eps);
        return .{ .normed = normed, .pre_norm = dec };
    }

    /// Rows dequantized/requantized per chunk when building the coarse head.
    /// 16384 x 5120 bf16 = 168 MB of transient dense weight, which bounds
    /// peak memory during the one-time build.
    const coarse_chunk_rows: i32 = 16384;
    /// Candidates rescored against the exact head. 64 is omlx's choice
    /// (`batch_generator._COARSE_CANDIDATES`) and is far above the rank at
    /// which a 3-bit head misplaces the true argmax.
    const coarse_candidates: i32 = 64;
    /// Bit width of the coarse draft head. Raising the depth ceiling to 7
    /// (§39) made the drafter the largest single per-round item — 4.1 ms per
    /// step, up to 7 steps — and 2.24 ms of each step is this head reading
    /// 517 MB at 3 bits. 2 bits reads 358 MB (~1.55 ms), and the quality
    /// argument is unchanged: the proposal only has to put the true token
    /// inside the top-`coarse_candidates`, after which the EXACT head decides.
    /// `DMLX_MTP_COARSE_BITS` overrides it for A/B.
    const coarse_bits_default: u8 = 2;
    const coarse_group: i32 = 128;

    fn coarseBits() u8 {
        const s = std.c.getenv("DMLX_MTP_COARSE_BITS") orelse return coarse_bits_default;
        const v = std.fmt.parseInt(u8, std.mem.span(s), 10) catch return coarse_bits_default;
        return switch (v) {
            2, 3, 4, 5, 6, 8 => v,
            else => coarse_bits_default,
        };
    }
    /// Below this vocabulary the full head is already cheap and the extra
    /// resident copy is not worth it.
    const coarse_min_vocab: i32 = 65536;

    /// Build a 3-bit gs128 copy of `lm_head`.
    ///
    /// Why this is quality-neutral: a draft token is only a GUESS. Acceptance
    /// is decided by comparing it against the TARGET's own argmax in the
    /// verify step, so a coarser proposal can lower the acceptance rate but
    /// cannot change the emitted sequence. The top-`coarse_candidates` rescore
    /// against the exact head's gathered rows removes most of even that loss,
    /// because the true argmax is essentially always inside the coarse top-64
    /// while 64 x hidden of exact weight is negligible traffic.
    ///
    /// Cost: ~517 MB resident for this model (477 MB of 3-bit weight + 40 MB
    /// of scales/biases) against 715 MB read per draft step by the 4-bit gs64
    /// head — 3.10 ms/step down to ~2.24 ms at the measured 231 GB/s.
    fn buildCoarseHead(self: *Qwen38Mtp, allocator: std.mem.Allocator, lm_head: MtpLinear) !MtpLinear {
        const ctx = self.ctx;
        const scales = lm_head.scales orelse return error.HeadNotQuantized;
        const biases = lm_head.biases orelse return error.HeadNotQuantized;
        if (lm_head.cfg.mode != .affine) return error.HeadNotAffine;
        const ws = lm_head.w.shape();
        if (ws.len != 2) return error.HeadRank;
        const rows: i32 = ws[0];
        if (rows < coarse_min_vocab) return error.HeadTooSmall;
        const wcols: i32 = ws[1];
        const scols: i32 = scales.shape()[1];

        var w_parts = std.ArrayList(Array).empty;
        defer w_parts.deinit(allocator);
        var s_parts = std.ArrayList(Array).empty;
        defer s_parts.deinit(allocator);
        var b_parts = std.ArrayList(Array).empty;
        defer b_parts.deinit(allocator);
        errdefer {
            for (w_parts.items) |x| x.deinit();
            for (s_parts.items) |x| x.deinit();
            for (b_parts.items) |x| x.deinit();
        }

        var start: i32 = 0;
        while (start < rows) : (start += coarse_chunk_rows) {
            const end = @min(rows, start + coarse_chunk_rows);
            const wc = try ops.slice(ctx, lm_head.w, &[_]i32{ start, 0 }, &[_]i32{ end, wcols }, &[_]i32{ 1, 1 });
            defer wc.deinit();
            const sc = try ops.slice(ctx, scales, &[_]i32{ start, 0 }, &[_]i32{ end, scols }, &[_]i32{ 1, 1 });
            defer sc.deinit();
            const bc = try ops.slice(ctx, biases, &[_]i32{ start, 0 }, &[_]i32{ end, scols }, &[_]i32{ 1, 1 });
            defer bc.deinit();
            const qw = quantize_mod.QuantizedWeight{
                .data = wc,
                .scales = sc,
                .biases = bc,
                .config = lm_head.cfg,
                .original_shape = &[_]i32{},
            };
            const dense = try quantize_mod.dequantizeAs(ctx, qw, .bfloat16);
            defer dense.deinit();
            const part = try quantize_mod.quantize(ctx, dense, .{
                .bits = coarseBits(),
                .group_size = coarse_group,
                .mode = .affine,
            });
            // Materialize per chunk so the transient dense array is released
            // before the next one is built.
            try part.data.eval();
            try part.scales.eval();
            try part.biases.eval();
            try w_parts.append(allocator, part.data);
            try s_parts.append(allocator, part.scales);
            try b_parts.append(allocator, part.biases);
        }

        const w3 = try shape_mod.concatenateAxis(ctx, w_parts.items, 0);
        errdefer w3.deinit();
        const s3 = try shape_mod.concatenateAxis(ctx, s_parts.items, 0);
        errdefer s3.deinit();
        const b3 = try shape_mod.concatenateAxis(ctx, b_parts.items, 0);
        errdefer b3.deinit();
        try w3.eval();
        try s3.eval();
        try b3.eval();
        for (w_parts.items) |x| x.deinit();
        for (s_parts.items) |x| x.deinit();
        for (b_parts.items) |x| x.deinit();
        w_parts.clearRetainingCapacity();
        s_parts.clearRetainingCapacity();
        b_parts.clearRetainingCapacity();
        return .{
            .w = w3,
            .scales = s3,
            .biases = b3,
            .cfg = .{ .bits = coarseBits(), .group_size = coarse_group, .mode = .affine },
        };
    }

    fn coarseHead(self: *Qwen38Mtp, allocator: std.mem.Allocator, lm_head: MtpLinear) ?*const MtpLinear {
        if (std.c.getenv("DMLX_MTP_NO_COARSE_HEAD") != null) return null;
        switch (self.coarse_state) {
            .ready => return &self.coarse_head.?,
            .failed => return null,
            .untried => {},
        }
        if (self.buildCoarseHead(allocator, lm_head)) |ch| {
            self.coarse_head = ch;
            self.coarse_state = .ready;
            std.log.info("[MTP] coarse draft head built ({d}-bit gs{d})", .{ coarseBits(), coarse_group });
            return &self.coarse_head.?;
        } else |err| {
            self.coarse_state = .failed;
            std.log.warn("[MTP] coarse draft head unavailable ({s}); drafting on the full head", .{@errorName(err)});
            return null;
        }
    }

    /// Draft token from the coarse head + exact top-k rescore, as a GPU [1]
    /// i32 array (chainable, like the full-head path).
    fn coarseDraftToken(self: *Qwen38Mtp, body: Array, lm_head: MtpLinear, coarse: *const MtpLinear) !Array {
        const ctx = self.ctx;
        const hidden: i32 = @intCast(self.dims.hidden_size);
        const h3 = try ops.reshape(ctx, body, &[_]i32{ 1, 1, hidden });
        defer h3.deinit();
        const approx3 = try coarse.apply(ctx, h3);
        defer approx3.deinit();
        const vocab: i32 = approx3.shape()[2];
        if (vocab <= coarse_candidates) return error.VocabTooSmall;
        const approx = try ops.reshape(ctx, approx3, &[_]i32{ 1, vocab });
        defer approx.deinit();

        // Top-K by argpartition: everything at or after kth is >= the kth
        // element, so the last K indices are the K largest (unordered, which
        // is fine — the exact rescore below decides the order). Runs
        // directly on the bf16 head output: bf16 ties only reshuffle the
        // candidate SET boundary, and the exact rescore decides among the
        // candidates, so skipping the f32 promotion is quality-neutral
        // while removing a full-vocab 1 MB write+read per draft step.
        const part = try sort_mod.argpartitionAxis(ctx, approx, vocab - coarse_candidates, 1);
        defer part.deinit();
        const ids2 = try ops.slice(ctx, part, &[_]i32{ 0, vocab - coarse_candidates }, &[_]i32{ 1, vocab }, &[_]i32{ 1, 1 });
        defer ids2.deinit();
        const ids = try ops.reshape(ctx, ids2, &[_]i32{coarse_candidates});
        defer ids.deinit();

        // Exact rescore: gather those rows of the real head (64 x hidden of
        // 4-bit weight, negligible bytes) and run the small matmul.
        const w_rows = try shape_mod.takeAxis(ctx, lm_head.w, ids, 0);
        defer w_rows.deinit();
        const s_rows = try shape_mod.takeAxis(ctx, lm_head.scales.?, ids, 0);
        defer s_rows.deinit();
        const b_rows = try shape_mod.takeAxis(ctx, lm_head.biases.?, ids, 0);
        defer b_rows.deinit();
        const qw = quantize_mod.QuantizedWeight{
            .data = w_rows,
            .scales = s_rows,
            .biases = b_rows,
            .config = lm_head.cfg,
            .original_shape = &[_]i32{},
        };
        const body2 = try ops.reshape(ctx, body, &[_]i32{ 1, hidden });
        defer body2.deinit();
        const exact = try quantize_mod.quantizedMatmul(ctx, body2, qw, true);
        defer exact.deinit();
        const am = try reduce_mod.argmax(ctx, exact, false);
        defer am.deinit();
        const am_i32 = try ops.astype(ctx, am, .int32);
        defer am_i32.deinit();
        const am1 = try ops.reshape(ctx, am_i32, &[_]i32{1});
        defer am1.deinit();
        const picked = try shape_mod.takeAxis(ctx, ids, am1, 0);
        defer picked.deinit();
        return ops.astype(ctx, picked, .int32);
    }

    /// Greedy GPU-chained step: forwardIds + argmax, returning the token id
    /// as an unevaled GPU [1] i32 array (chainable into the next step) and
    /// the chained hidden. The logits are dropped — greedy drafting never
    /// needs them on CPU. GPU argmax tie-breaks may differ from the CPU
    /// first-max scan on exact f32 ties (measure-zero); either is a valid
    /// greedy draft.
    pub fn forwardGreedyGpu(
        self: *Qwen38Mtp,
        allocator: std.mem.Allocator,
        hidden_pre_norm: Array,
        id_arr: Array, // [1] i32
        embed_table_bf16: Array,
        lm_head: MtpLinear,
    ) !MtpGreedyStepOut {
        const ctx = self.ctx;
        if (std.c.getenv("DMLX_MTP_NO_LM_HEAD") != null) {
            // DIAGNOSTIC ONLY: prices the drafter's full-vocab lm_head
            // (5120 x 248320 at 4 bits = 636 MB read per draft step) by
            // replacing it with an argmax over the hidden row. Drafts are
            // garbage, so acceptance collapses — read ms/round only, and
            // only at a FIXED depth.
            const body = try self.forwardBodyIds(allocator, hidden_pre_norm, id_arr, embed_table_bf16);
            errdefer body.deinit();
            const am = try reduce_mod.argmax(ctx, body, false);
            defer am.deinit();
            const am_i32 = try ops.astype(ctx, am, .int32);
            defer am_i32.deinit();
            const tok = try ops.reshape(ctx, am_i32, &[_]i32{1});
            errdefer tok.deinit();
            return .{ .tok_gpu = tok, .hidden_out = body };
        }
        // Coarse draft head: propose from a 3-bit copy of lm_head, then
        // rescore the top-64 against the exact rows. Quality-neutral (the
        // target's verify decides acceptance) and saves ~0.9 ms of the
        // 3.10 ms/step the full 4-bit head costs. See buildCoarseHead.
        if (self.coarseHead(allocator, lm_head)) |coarse| {
            // `DMLX_MTP_CHAIN_PRENORM=1`: chain the next step on the PRE-final
            // -norm output so every draft step sees the same kind of hidden as
            // draft 1 (which gets the backbone's pre-norm row). See
            // forwardBodyIdsPair — this is the p_i-cliff experiment.
            const chain_prenorm = std.c.getenv("DMLX_MTP_CHAIN_PRENORM") != null;
            var body: Array = undefined;
            var chain: Array = undefined;
            if (chain_prenorm) {
                const pair = try self.forwardBodyIdsPair(allocator, hidden_pre_norm, id_arr, embed_table_bf16);
                body = pair.normed;
                chain = pair.pre_norm;
            } else {
                body = try self.forwardBodyIds(allocator, hidden_pre_norm, id_arr, embed_table_bf16);
                chain = body;
            }
            errdefer body.deinit();
            errdefer if (chain_prenorm) chain.deinit();
            if (self.coarseDraftToken(body, lm_head, coarse)) |tok| {
                if (chain_prenorm) body.deinit();
                return .{ .tok_gpu = tok, .hidden_out = chain };
            } else |err| {
                if (chain_prenorm) chain.deinit();
                // One-shot downgrade: never retry per step.
                self.coarse_state = .failed;
                std.log.warn("[MTP] coarse draft token failed ({s}); full head from here", .{@errorName(err)});
                const h3 = try ops.reshape(ctx, body, &[_]i32{ 1, 1, @as(i32, @intCast(self.dims.hidden_size)) });
                defer h3.deinit();
                const lg = try lm_head.apply(ctx, h3);
                defer lg.deinit();
                const lgf = try ops.astype(ctx, lg, .float32);
                defer lgf.deinit();
                const am0 = try reduce_mod.argmax(ctx, lgf, false);
                defer am0.deinit();
                const am0i = try ops.astype(ctx, am0, .int32);
                defer am0i.deinit();
                const tok0 = try ops.reshape(ctx, am0i, &[_]i32{1});
                return .{ .tok_gpu = tok0, .hidden_out = body };
            }
        }
        var out = try self.forwardIds(allocator, hidden_pre_norm, id_arr, embed_table_bf16, lm_head);
        defer out.logits.deinit();
        const am = try reduce_mod.argmax(ctx, out.logits, false); // scalar
        defer am.deinit();
        const am_i32 = try ops.astype(ctx, am, .int32);
        defer am_i32.deinit();
        const tok = try ops.reshape(ctx, am_i32, &[_]i32{1});
        errdefer tok.deinit();
        return .{ .tok_gpu = tok, .hidden_out = out.hidden_out };
    }
};

pub const MtpGreedyStepOut = struct {
    tok_gpu: Array, // [1] i32 (unevaled)
    hidden_out: Array, // [1, hidden] bf16
};

fn rmsNormBf16(ctx: EagerContext, x: Array, w_f32: Array, eps: f32) !Array {
    const x_f32 = try ops.astype(ctx, x, .float32);
    defer x_f32.deinit();
    const n_f32 = try fast_mod.rmsNorm(ctx, x_f32, w_f32, eps);
    defer n_f32.deinit();
    return ops.astype(ctx, n_f32, .bfloat16);
}

fn stageChk(name: []const u8, ctx: EagerContext, t: Array) void {
    if (std.c.getenv("DMLX_MTP_STAGE") == null) return;
    const f = ops.astype(ctx, t, .float32) catch return;
    defer f.deinit();
    const s = reduce_mod.sum(ctx, f, false) catch return;
    defer s.deinit();
    const a = ops.abs(ctx, f) catch return;
    defer a.deinit();
    const m = reduce_mod.max(ctx, a, false) catch return;
    defer m.deinit();
    s.eval() catch return;
    m.eval() catch return;
    std.debug.print("DMLX {s}: chk={d:.4} absmax={d:.4}\n", .{ name, s.item(f32) catch 0, m.item(f32) catch 0 });
}

// ============================================================
// Tests (toy dims)
// ============================================================

fn testFill(buf: []f32) void {
    for (buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 37 + 11, 13) - 6)) * 0.02;
    }
}

fn makeWeight(allocator: std.mem.Allocator, shape_: []const i32) !Array {
    var n: usize = 1;
    for (shape_) |s| n *= @intCast(s);
    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    testFill(buf);
    return Array.fromData(allocator, f32, buf, shape_);
}

fn makeTestMtp(allocator: std.mem.Allocator, ctx: EagerContext) !Qwen38Mtp {
    // Toy dims: hidden=16, heads=2, kv=1, head_dim=8, rotary=2, inter=32,
    // vocab=64. Unquantized linears (scales=null → plain matmul path).
    const hidden: i32 = 16;
    const inter: i32 = 32;
    var attn = try qwen38_attn.Qwen38Attn.makeTestModule(allocator, ctx);
    errdefer attn.deinit();

    const ln1 = try makeWeight(allocator, &[_]i32{hidden});
    errdefer ln1.deinit();
    const ln2 = try makeWeight(allocator, &[_]i32{hidden});
    errdefer ln2.deinit();
    const norm = try makeWeight(allocator, &[_]i32{hidden});
    errdefer norm.deinit();
    const nh = try makeWeight(allocator, &[_]i32{hidden});
    errdefer nh.deinit();
    const ne = try makeWeight(allocator, &[_]i32{hidden});
    errdefer ne.deinit();

    return .{
        .ctx = ctx,
        .dims = .{
            .hidden_size = 16,
            .num_heads = 2,
            .num_kv_heads = 1,
            .head_dim = 8,
            .rotary_dims = 2,
            .intermediate_size = 32,
            .rope_theta = 10000.0,
        },
        .pre_fc_norm_hidden_f32 = nh,
        .pre_fc_norm_embedding_f32 = ne,
        .norm_f32 = norm,
        .fc = .{ .w = try makeWeight(allocator, &[_]i32{ hidden, 2 * hidden }) },
        .ln1_f32 = ln1,
        .ln2_f32 = ln2,
        .attn = attn,
        .mlp_gate = .{ .w = try makeWeight(allocator, &[_]i32{ inter, hidden }) },
        .mlp_up = .{ .w = try makeWeight(allocator, &[_]i32{ inter, hidden }) },
        .mlp_down = .{ .w = try makeWeight(allocator, &[_]i32{ hidden, inter }) },
    };
}

/// Toy-config (plain weights, rotary=2, theta=1e4, eps=1e-6, output gate)
/// pre-segment closure for the compiled-step unit test. Input order (13):
/// id, embed, hidden, w_e, w_h, fc.w, ln1, q_w, k_w, v_w, q_norm, k_norm,
/// offset. Returns [fused, q_t, k_t, v_t, gate_sig].
fn mtpPreClosurePlainTest(inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array {
    if (inputs.len != 13) return error.MlxError;
    const ctx = mtpClosureGpuCtx(allocator);
    defer ctx.deinit();
    const res = mtpPreSeg(ctx, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], .{
        .w = inputs[5],
    }, inputs[6], .{
        .x = inputs[2],
        .q_w = inputs[7],
        .k_w = inputs[8],
        .v_w = inputs[9],
        .q_norm = inputs[10],
        .k_norm = inputs[11],
        .offset = inputs[12],
    }, 2, 10000.0, 1e-6) catch return error.MlxError;
    return preSegToSlice(res, allocator);
}

/// Toy-config post-segment closure. Input order (9): attn, gate_sig, fused,
/// o_w, ln2, gate_w, up_w, down_w, norm. Returns [normed, dec].
fn mtpPostClosurePlainTest(inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array {
    if (inputs.len != 9) return error.MlxError;
    const ctx = mtpClosureGpuCtx(allocator);
    defer ctx.deinit();
    const res = mtpPostSeg(ctx, inputs[0], inputs[1], inputs[2], .{ .w = inputs[3] }, inputs[4], .{ .w = inputs[5] }, .{ .w = inputs[6] }, .{ .w = inputs[7] }, inputs[8], 1e-6) catch return error.MlxError;
    return postSegToSlice(res, allocator);
}

test "Qwen38Mtp: compiled decode-step segments match the eager step" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const hidden: i32 = 16;
    const vocab: i32 = 64;

    var mtp_e = try makeTestMtp(allocator, ctx);
    defer mtp_e.deinit();
    var mtp_c = try makeTestMtp(allocator, ctx);
    defer mtp_c.deinit();

    const emb_f32 = try makeWeight(allocator, &[_]i32{ vocab, hidden });
    defer emb_f32.deinit();
    const emb = try ops.astype(ctx, emb_f32, .bfloat16);
    defer emb.deinit();
    const hidden_in = try makeWeight(allocator, &[_]i32{ 1, hidden });
    defer hidden_in.deinit();
    const id_arr = try Array.fromData(allocator, i32, &[_]i32{5}, &[_]i32{1});
    defer id_arr.deinit();

    // Eager reference (advances mtp_e's cache).
    const eager = try mtp_e.forwardBodyIdsPairEager(allocator, hidden_in, id_arr, emb);
    defer eager.normed.deinit();
    defer eager.pre_norm.deinit();
    try eager.normed.eval();
    try eager.pre_norm.eval();
    try std.testing.expectEqual(@as(i32, 1), mtp_e.cache.len());

    // Compiled segments over mtp_c (toy-config closures).
    const base_pre = try Closure.init(&mtpPreClosurePlainTest, allocator);
    defer base_pre.deinit();
    const cc_pre = try compile_mod.compile(base_pre, false);
    defer cc_pre.deinit();
    const base_post = try Closure.init(&mtpPostClosurePlainTest, allocator);
    defer base_post.deinit();
    const cc_post = try compile_mod.compile(base_post, false);
    defer cc_post.deinit();

    const offset_arr = try ops.scalarI32(ctx, mtp_c.cache.len());
    defer offset_arr.deinit();
    const pre_inputs = [_]Array{
        id_arr,                          emb,                          hidden_in,
        mtp_c.pre_fc_norm_embedding_f32, mtp_c.pre_fc_norm_hidden_f32, mtp_c.fc.w,
        mtp_c.ln1_f32,                   mtp_c.attn.q_proj,            mtp_c.attn.k_proj,
        mtp_c.attn.v_proj,               mtp_c.attn.q_norm,            mtp_c.attn.k_norm,
        offset_arr,
    };
    const pre_outs = try cc_pre.apply(&pre_inputs, allocator);
    defer allocator.free(pre_outs);
    defer for (pre_outs) |o| o.deinit();
    try std.testing.expectEqual(@as(usize, 5), pre_outs.len);

    const attn = try mtp_c.attn.cacheAppendAndSdpa(&mtp_c.cache, pre_outs[1], pre_outs[2], pre_outs[3]);
    defer attn.deinit();
    try std.testing.expectEqual(@as(i32, 1), mtp_c.cache.len());

    const post_inputs = [_]Array{
        attn,              pre_outs[4],      pre_outs[0],
        mtp_c.attn.o_proj, mtp_c.ln2_f32,    mtp_c.mlp_gate.w,
        mtp_c.mlp_up.w,    mtp_c.mlp_down.w, mtp_c.norm_f32,
    };
    const post_outs = try cc_post.apply(&post_inputs, allocator);
    defer allocator.free(post_outs);
    defer for (post_outs) |o| o.deinit();
    try std.testing.expectEqual(@as(usize, 2), post_outs.len);

    // bf16-ULP parity on both the post-norm output and the pre-norm decoder
    // output.
    for ([_]Array{ post_outs[0], post_outs[1] }, [_]Array{ eager.normed, eager.pre_norm }) |got, exp| {
        const gf = try ops.astype(ctx, got, .float32);
        defer gf.deinit();
        const ef = try ops.astype(ctx, exp, .float32);
        defer ef.deinit();
        const diff = try ops.subtract(ctx, gf, ef);
        defer diff.deinit();
        const abs_diff = try ops.abs(ctx, diff);
        defer abs_diff.deinit();
        const max_diff = try reduce_mod.max(ctx, abs_diff, false);
        defer max_diff.deinit();
        try max_diff.eval();
        try std.testing.expect(try max_diff.item(f32) <= 1e-4);
    }

    // Offset-invariance: a second chained step at cache offset 1 must not
    // retrace into different semantics — compare against eager step 2.
    const eager2 = try mtp_e.forwardBodyIdsPairEager(allocator, eager.normed, id_arr, emb);
    defer eager2.normed.deinit();
    defer eager2.pre_norm.deinit();
    try eager2.normed.eval();

    const offset1 = try ops.scalarI32(ctx, mtp_c.cache.len());
    defer offset1.deinit();
    const pre_inputs1 = [_]Array{
        id_arr,                          emb,                          post_outs[0],
        mtp_c.pre_fc_norm_embedding_f32, mtp_c.pre_fc_norm_hidden_f32, mtp_c.fc.w,
        mtp_c.ln1_f32,                   mtp_c.attn.q_proj,            mtp_c.attn.k_proj,
        mtp_c.attn.v_proj,               mtp_c.attn.q_norm,            mtp_c.attn.k_norm,
        offset1,
    };
    const pre_outs1 = try cc_pre.apply(&pre_inputs1, allocator);
    defer allocator.free(pre_outs1);
    defer for (pre_outs1) |o| o.deinit();
    const attn1 = try mtp_c.attn.cacheAppendAndSdpa(&mtp_c.cache, pre_outs1[1], pre_outs1[2], pre_outs1[3]);
    defer attn1.deinit();
    try std.testing.expectEqual(@as(i32, 2), mtp_c.cache.len());
    const post_inputs1 = [_]Array{
        attn1,             pre_outs1[4],     pre_outs1[0],
        mtp_c.attn.o_proj, mtp_c.ln2_f32,    mtp_c.mlp_gate.w,
        mtp_c.mlp_up.w,    mtp_c.mlp_down.w, mtp_c.norm_f32,
    };
    const post_outs1 = try cc_post.apply(&post_inputs1, allocator);
    defer allocator.free(post_outs1);
    defer for (post_outs1) |o| o.deinit();

    const gf = try ops.astype(ctx, post_outs1[0], .float32);
    defer gf.deinit();
    const ef = try ops.astype(ctx, eager2.normed, .float32);
    defer ef.deinit();
    const diff = try ops.subtract(ctx, gf, ef);
    defer diff.deinit();
    const abs_diff = try ops.abs(ctx, diff);
    defer abs_diff.deinit();
    const max_diff = try reduce_mod.max(ctx, abs_diff, false);
    defer max_diff.deinit();
    try max_diff.eval();
    try std.testing.expect(try max_diff.item(f32) <= 1e-4);
}

test "Qwen38Mtp: shape, determinism, cache advance, chaining" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const hidden: i32 = 16;
    const vocab: i32 = 64;

    var mtp = try makeTestMtp(allocator, ctx);
    defer mtp.deinit();
    var mtp2 = try makeTestMtp(allocator, ctx);
    defer mtp2.deinit();

    // Toy embed table [64, 16] bf16 and a plain (unquantized) lm_head.
    const emb_f32 = try makeWeight(allocator, &[_]i32{ vocab, hidden });
    defer emb_f32.deinit();
    const emb = try ops.astype(ctx, emb_f32, .bfloat16);
    defer emb.deinit();
    const lm_head_w = try makeWeight(allocator, &[_]i32{ vocab, hidden });
    defer lm_head_w.deinit();
    const lm_head = MtpLinear{ .w = lm_head_w };

    const hidden_in = try makeWeight(allocator, &[_]i32{ 1, hidden });
    defer hidden_in.deinit();

    const out1 = try mtp.forward(allocator, hidden_in, 5, emb, lm_head);
    defer out1.logits.deinit();
    defer out1.hidden_out.deinit();
    try out1.logits.eval();
    try out1.hidden_out.eval();

    const lshape = out1.logits.shape();
    try std.testing.expectEqual(@as(usize, 3), lshape.len);
    try std.testing.expectEqual(@as(i32, 1), lshape[0]);
    try std.testing.expectEqual(@as(i32, 1), lshape[1]);
    try std.testing.expectEqual(vocab, lshape[2]);
    try std.testing.expectEqual(@as(i32, 1), mtp.cache.len());
    const hshape = out1.hidden_out.shape();
    try std.testing.expectEqual(@as(usize, 2), hshape.len);
    try std.testing.expectEqual(hidden, hshape[1]);

    // Determinism: fresh module, same inputs → identical outputs.
    const out2 = try mtp2.forward(allocator, hidden_in, 5, emb, lm_head);
    defer out2.logits.deinit();
    defer out2.hidden_out.deinit();
    const diff = try ops.subtract(ctx, out1.logits, out2.logits);
    defer diff.deinit();
    const abs_diff = try ops.abs(ctx, diff);
    defer abs_diff.deinit();
    const max_diff = try reduce_mod.max(ctx, abs_diff, false);
    defer max_diff.deinit();
    try max_diff.eval();
    try std.testing.expectEqual(@as(f32, 0.0), try max_diff.item(f32));

    // Chaining: hidden_out feeds the next pass; cache advances.
    const out3 = try mtp.forward(allocator, out1.hidden_out, 7, emb, lm_head);
    defer out3.logits.deinit();
    defer out3.hidden_out.deinit();
    try out3.logits.eval();
    try std.testing.expectEqual(@as(i32, 2), mtp.cache.len());
    const l3 = try out3.logits.dataSlice(f32);
    var finite = true;
    for (l3) |v| finite = finite and std.math.isFinite(v);
    try std.testing.expect(finite);
}
