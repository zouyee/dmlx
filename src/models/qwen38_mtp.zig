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
const safetensors_reader = @import("mlx").safetensors_reader;
const glm53 = @import("glm53.zig");

const qwen38_attn = @import("qwen38_attn.zig");
const qwen38_gdn = @import("qwen38_gdn.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
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
        const fused = try self.fc.apply(ctx, cat);
        defer fused.deinit();

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
        self.cache.deinit();
    }

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

    /// Load from a drafter checkpoint directory (safetensors + index).
    pub fn load(allocator: std.mem.Allocator, ctx: EagerContext, model_dir: []const u8, dims: Qwen38MtpDims) !Qwen38Mtp {
        var index = try safetensors_reader.loadOrBuildIndex(allocator, model_dir);
        defer index.deinit();
        return loadFromSource(ctx, WeightSource.fromIndex(&index), dims);
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
        const ctx = self.ctx;
        const hidden: i32 = @intCast(self.dims.hidden_size);
        const eps = self.dims.rms_norm_eps;

        // e = normE(embed(token)); h = normH(hidden)
        const id_arr = try Array.fromData(allocator, i32, &[_]i32{@intCast(token_id)}, &[_]i32{1});
        defer id_arr.deinit();
        const row = try shape_mod.takeAxis(ctx, embed_table_bf16, id_arr, 0); // [1, hidden] bf16
        defer row.deinit();
        const e = try rmsNormBf16(ctx, row, self.pre_fc_norm_embedding_f32, eps);
        defer e.deinit();
        const h = try rmsNormBf16(ctx, hidden_pre_norm, self.pre_fc_norm_hidden_f32, eps);
        defer h.deinit();

        // fused = fc(concat([e, h], -1))  [1, hidden]
        const cat = try shape_mod.concatenateAxis(ctx, &[_]Array{ e, h }, 1);
        defer cat.deinit();
        const fused = try self.fc.apply(ctx, cat);
        defer fused.deinit();

        // Decoder layer: attn block + MLP block, both residual.
        const n1 = try rmsNormBf16(ctx, fused, self.ln1_f32, eps);
        defer n1.deinit();
        const attn_out = try self.attn.forward(n1, &self.cache); // [1, hidden]
        defer attn_out.deinit();
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
        const dec_out = try ops.add(ctx, h2, mlp_out);
        defer dec_out.deinit();

        // Final norm → hidden_out (this tensor feeds both lm_head and, in
        // chained drafting, the next pass's pre_fc_norm_hidden).
        const hidden_out = try rmsNormBf16(ctx, dec_out, self.norm_f32, eps);
        errdefer hidden_out.deinit();
        const h3 = try ops.reshape(ctx, hidden_out, &[_]i32{ 1, 1, hidden });
        defer h3.deinit();
        const logits_bf16 = try lm_head.apply(ctx, h3);
        defer logits_bf16.deinit();
        const logits = try ops.astype(ctx, logits_bf16, .float32);
        errdefer logits.deinit();
        return .{ .logits = logits, .hidden_out = hidden_out };
    }
};

fn rmsNormBf16(ctx: EagerContext, x: Array, w_f32: Array, eps: f32) !Array {
    const x_f32 = try ops.astype(ctx, x, .float32);
    defer x_f32.deinit();
    const n_f32 = try fast_mod.rmsNorm(ctx, x_f32, w_f32, eps);
    defer n_f32.deinit();
    return ops.astype(ctx, n_f32, .bfloat16);
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
