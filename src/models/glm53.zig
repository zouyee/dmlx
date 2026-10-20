/// GLM-5.3 model assembly: embed → mHC residual stack of 45 decoder layers
/// (KDA linear attention / NoPE MLA full attention × dense MLP / sparse MoE)
/// → final norm → lm_head logits. Mirrors mlx-vlm `Glm5NextTextModel` +
/// `LanguageModel` (reference: /tmp/glm5_next_ref/language.py).
///
/// This module holds the per-layer branch helpers (mHC apply_branch
/// semantics), the MoE block (sigmoid gate + correction bias + top-8 routing,
/// limited-SwiGLU experts, shared expert), and `Glm53Model.forward` which
/// drives the whole stack over a safetensors index with per-layer weight
/// loading (the mixed 4/8-bit checkpoint is ~125GB, far larger than RAM, so
/// every layer's weights are released before the next layer loads; MoE expert
/// rows are pread on demand for the router-selected experts only).
///
/// Oracle alignment status (7-token fixture, tests/fixtures/glm53_oracle):
/// segments through layer 3 verified in Task 7/8 (layer0_kda_out cos
/// 0.99995949, layer3_mla_out cos 0.99998813, layer3_indexer_topk exact).
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const reduce_mod = @import("mlx").reduce;
const math_mod = @import("mlx").math;
const sort_mod = @import("mlx").sort;
const comparison_mod = @import("mlx").comparison;
const fast_mod = @import("mlx").fast;
const activations = @import("mlx").activations;
const quantize_mod = @import("mlx").quantize;
const safetensors_reader = @import("mlx").safetensors_reader;
const glm53_loader = @import("glm53_loader.zig");
const glm53_kda = @import("glm53_kda.zig");
const glm53_attn = @import("glm53_attention.zig");
const dsv4 = @import("deepseek_v4.zig");
const expert_stream = @import("expert_stream.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

// ============================================================
// Checkpoint loading helpers
// ============================================================

pub const QuantParts = struct {
    data: Array,
    scales: Array,
    biases: Array,
    cfg: quantize_mod.QuantConfig,

    pub fn deinit(self: *QuantParts) void {
        self.data.deinit();
        self.scales.deinit();
        self.biases.deinit();
    }

    pub fn linear(self: *const QuantParts) glm53_attn.QuantLinear {
        return .{ .w = self.data, .scales = self.scales, .biases = self.biases, .cfg = self.cfg };
    }
};

pub fn loadTensor(index: *safetensors_reader.TensorIndex, name: []const u8) !Array {
    return index.loadTensor(name) catch |err| {
        std.log.err("tensor not found or unreadable: {s} ({})", .{ name, err });
        return err;
    };
}

/// Tensor lookup abstraction: safetensors-index-backed (Task 9 harness,
/// every call reads fresh from disk) or resident-map-backed (serving: the
/// backbone is loaded once; lookups return cheap COW copies via ops.copy).
/// In both cases the caller owns the returned Array.
pub const WeightSource = struct {
    ctx: *anyopaque,
    tensorFn: *const fn (ctx: *anyopaque, name: []const u8) anyerror!Array,

    pub fn tensor(self: WeightSource, name: []const u8) !Array {
        return self.tensorFn(self.ctx, name);
    }

    /// Affine-quantized Linear triple with bits/group_size derived from the
    /// packed shapes (same derivation as scripts/glm53_oracle_dump.py).
    pub fn quantized(self: WeightSource, prefix: []const u8, in_features: i32) !QuantParts {
        var nbuf: [256]u8 = undefined;
        const data = try self.tensor(fmtName(&nbuf, "{s}.weight", .{prefix}));
        errdefer data.deinit();
        const scales = try self.tensor(fmtName(&nbuf, "{s}.scales", .{prefix}));
        errdefer scales.deinit();
        const biases = try self.tensor(fmtName(&nbuf, "{s}.biases", .{prefix}));
        errdefer biases.deinit();

        const wshape = data.shape();
        const sshape = scales.shape();
        const packed_cols: i64 = wshape[wshape.len - 1];
        const scale_cols: i64 = sshape[sshape.len - 1];
        const bits: u8 = @intCast(@divTrunc(packed_cols * 32, in_features));
        const group_size: i32 = @intCast(@divTrunc(in_features, scale_cols));
        return .{
            .data = data,
            .scales = scales,
            .biases = biases,
            .cfg = .{ .bits = bits, .group_size = group_size, .mode = .affine },
        };
    }

    fn fromIndexFn(ctx: *anyopaque, name: []const u8) anyerror!Array {
        const index: *safetensors_reader.TensorIndex = @ptrCast(@alignCast(ctx));
        return loadTensor(index, name);
    }

    pub fn fromIndex(index: *safetensors_reader.TensorIndex) WeightSource {
        return .{ .ctx = index, .tensorFn = fromIndexFn };
    }

    const MapBacked = struct {
        map: *const std.StringHashMap(Array),
        ectx: EagerContext,
    };

    fn fromMapFn(ctx: *anyopaque, name: []const u8) anyerror!Array {
        const mb: *MapBacked = @ptrCast(@alignCast(ctx));
        const arr = mb.map.get(name) orelse {
            std.log.err("resident weights: tensor missing: {s}", .{name});
            return error.TensorNotFound;
        };
        // COW copy: the map keeps ownership, the caller gets a safe handle.
        return ops.copy(mb.ectx, arr);
    }

    /// `backing` must outlive the WeightSource.
    pub fn fromMap(map: *const std.StringHashMap(Array), ectx: EagerContext, backing: *MapBacked) WeightSource {
        backing.* = .{ .map = map, .ectx = ectx };
        return .{ .ctx = backing, .tensorFn = fromMapFn };
    }
};

/// Weight names are fixed patterns well under 128 bytes (longest:
/// "language_model.model.layers.43.self_attn.indexer.index_kpool_compress_gate"
/// ≈ 75 chars). All name formatting uses stack buffers via this helper — no
/// heap allocation, so the module is safe to call with any allocator
/// (Task 10 streaming/server paths use GPA, not arena).
fn fmtName(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch unreachable;
}

/// Load an affine-quantized Linear triple, deriving bits/group_size from the
/// packed shapes (same derivation as scripts/glm53_oracle_dump.py).
pub fn loadQuantized(index: *safetensors_reader.TensorIndex, prefix: []const u8, in_features: i32) !QuantParts {
    return WeightSource.fromIndex(index).quantized(prefix, in_features);
}

/// Load one expert's quantized projection (weight+scales+biases row slice)
/// via PartialTensorReader (mmap zero-copy when the pool is set, pread
/// fallback otherwise; BF16 handled natively) and derive the affine quant
/// config from the packed shapes.
fn loadExpertQuant(
    reader: *safetensors_reader.PartialTensorReader,
    ctx: EagerContext,
    base: []const u8,
    ids: []const u32,
    in_f: i32,
) !QuantParts {
    var nbuf: [256]u8 = undefined;
    const w = try reader.readExpertRows(fmtName(&nbuf, "{s}.weight", .{base}), ids);
    defer w.deinit();
    const s = try reader.readExpertRows(fmtName(&nbuf, "{s}.scales", .{base}), ids);
    defer s.deinit();
    const b = try reader.readExpertRows(fmtName(&nbuf, "{s}.biases", .{base}), ids);
    defer b.deinit();
    // Squeeze the leading [1, ...] expert dim.
    const w2 = try ops.squeeze(ctx, w);
    errdefer w2.deinit();
    const s2 = try ops.squeeze(ctx, s);
    errdefer s2.deinit();
    const b2 = try ops.squeeze(ctx, b);
    errdefer b2.deinit();
    const wshape = w2.shape();
    const sshape = s2.shape();
    const packed_cols: i64 = wshape[wshape.len - 1];
    const scale_cols: i64 = sshape[sshape.len - 1];
    return QuantParts{
        .data = w2,
        .scales = s2,
        .biases = b2,
        .cfg = .{
            .bits = @intCast(@divTrunc(packed_cols * 32, in_f)),
            .group_size = @intCast(@divTrunc(in_f, scale_cols)),
            .mode = .affine,
        },
    };
}

// ============================================================
// mHC branch helpers (apply_branch semantics)
// ============================================================

/// mHC apply_branch pre-phase: compute mixes from the residual, collapse into
/// the branch input, and RMSNorm it. Returned arrays are owned by the caller.
pub const HcPre = struct {
    normed: Array, // [1, L, hidden]
    post_mix: Array,
    comb_norm: Array,

    pub fn deinit(self: *HcPre) void {
        self.normed.deinit();
        self.post_mix.deinit();
        self.comb_norm.deinit();
    }
};

pub fn hcPre(
    ctx: EagerContext,
    ws: WeightSource,
    cfg: *const glm53_loader.Glm53Config,
    hc_prefix: []const u8,
    ln_name: []const u8,
    residual: Array,
    stream_raw: c.c.mlx_stream,
) !HcPre {
    var nbuf: [256]u8 = undefined;
    const hc_fn = try ws.tensor(fmtName(&nbuf, "{s}.fn", .{hc_prefix}));
    defer hc_fn.deinit();
    const hc_base = try ws.tensor(fmtName(&nbuf, "{s}.base", .{hc_prefix}));
    defer hc_base.deinit();
    const hc_scale = try ws.tensor(fmtName(&nbuf, "{s}.scale", .{hc_prefix}));
    defer hc_scale.deinit();
    const ln_w = try ws.tensor(ln_name);
    defer ln_w.deinit();

    // HC pre: Python computes mixes from the f32 residual and applies the
    // weight-less rms_norm with rms_norm_eps (not hc_eps).
    const res_f32 = try ops.astype(ctx, residual, .float32);
    defer res_f32.deinit();
    const hc_fn_f32 = try ops.astype(ctx, hc_fn, .float32);
    defer hc_fn_f32.deinit();
    const mixes = try dsv4.mhcPreNormFn(ctx, res_f32, hc_fn_f32, null, cfg.rms_norm_eps, stream_raw);
    defer mixes.deinit();
    const splits = try dsv4.mhcPreSplitMixes(ctx, mixes, hc_scale, hc_base, cfg.hc_mult, 2.0, cfg.hc_eps, stream_raw);
    const pre_mix = splits[0];
    const post_mix = splits[1];
    const comb_mix = splits[2];
    defer pre_mix.deinit();
    errdefer post_mix.deinit();
    errdefer comb_mix.deinit();
    const comb_norm = try dsv4.sinkhornNormalize(ctx, comb_mix, cfg.hc_sinkhorn_iters, cfg.hc_eps, stream_raw);
    comb_mix.deinit();
    errdefer comb_norm.deinit();

    const collapsed = try dsv4.mhcPreApplyMix(ctx, residual, pre_mix, stream_raw); // [1, L, hidden]
    defer collapsed.deinit();
    const collapsed_f32 = try ops.astype(ctx, collapsed, .float32);
    defer collapsed_f32.deinit();
    const ln_w_f32 = try ops.astype(ctx, ln_w, .float32);
    defer ln_w_f32.deinit();
    const normed = try fast_mod.rmsNorm(ctx, collapsed_f32, ln_w_f32, cfg.rms_norm_eps);
    return .{ .normed = normed, .post_mix = post_mix, .comb_norm = comb_norm };
}

/// KDA attention branch (attn_hc + input_layernorm + self_attn(KDA) + expand).
/// residual: [1, L, hc_mult, hidden]; returns the updated residual.
pub fn kdaAttnBranch(
    ctx: EagerContext,
    ws: WeightSource,
    allocator: std.mem.Allocator,
    cfg: *const glm53_loader.Glm53Config,
    layer_idx: u32,
    residual: Array,
    stream_raw: c.c.mlx_stream,
    kda_cache: ?*glm53_kda.GlmKdaCache,
) !Array {
    const hidden: i32 = @intCast(cfg.hidden_size);
    var pbuf: [64]u8 = undefined;
    const p = fmtName(&pbuf, "language_model.model.layers.{d}", .{layer_idx});
    var nbuf: [256]u8 = undefined;
    var nbuf2: [256]u8 = undefined;

    var pre = try hcPre(ctx, ws, cfg, fmtName(&nbuf, "{s}.attn_hc", .{p}), fmtName(&nbuf2, "{s}.input_layernorm.weight", .{p}), residual, stream_raw);
    defer pre.deinit();

    // KDA weights (prep'd fused names; forget_gate.* keeps its prefix).
    var sabuf: [96]u8 = undefined;
    const sa = fmtName(&sabuf, "{s}.self_attn", .{p});
    var qkv_q = try ws.quantized(fmtName(&nbuf, "{s}.qkv_proj", .{sa}), hidden);
    defer qkv_q.deinit();
    var fbg_q = try ws.quantized(fmtName(&nbuf, "{s}.fbg_a_proj", .{sa}), hidden);
    defer fbg_q.deinit();
    var o_q = try ws.quantized(fmtName(&nbuf, "{s}.o_proj", .{sa}), @intCast(cfg.linear_num_heads * cfg.linear_head_dim));
    defer o_q.deinit();

    const qkv_conv = try ws.tensor(fmtName(&nbuf, "{s}.qkv_conv.weight", .{sa}));
    defer qkv_conv.deinit();
    const qkv_conv_f32 = try ops.astype(ctx, qkv_conv, .float32);
    defer qkv_conv_f32.deinit();

    // f_b/g_b are consumed unquantized by GlmKDA — dequantize at load.
    var fb_q = try ws.quantized(fmtName(&nbuf, "{s}.forget_gate.f_b_proj", .{sa}), @intCast(cfg.linear_head_dim));
    defer fb_q.deinit();
    const f_b = try quantize_mod.dequantizeAs(ctx, .{ .data = fb_q.data, .scales = fb_q.scales, .biases = fb_q.biases, .config = fb_q.cfg, .original_shape = &[_]i32{} }, .float32);
    defer f_b.deinit();
    var gb_q = try ws.quantized(fmtName(&nbuf, "{s}.g_b_proj", .{sa}), @intCast(cfg.linear_head_dim));
    defer gb_q.deinit();
    const g_b = try quantize_mod.dequantizeAs(ctx, .{ .data = gb_q.data, .scales = gb_q.scales, .biases = gb_q.biases, .config = gb_q.cfg, .original_shape = &[_]i32{} }, .float32);
    defer g_b.deinit();

    const a_log = try ws.tensor(fmtName(&nbuf, "{s}.forget_gate.A_log", .{sa}));
    defer a_log.deinit();
    const dt_bias = try ws.tensor(fmtName(&nbuf, "{s}.forget_gate.dt_bias", .{sa}));
    defer dt_bias.deinit();
    const o_norm_w = try ws.tensor(fmtName(&nbuf, "{s}.o_norm.weight", .{sa}));
    defer o_norm_w.deinit();
    const o_norm_f32 = try ops.astype(ctx, o_norm_w, .float32);
    defer o_norm_f32.deinit();

    var kda = glm53_kda.GlmKDA{
        .ctx = ctx,
        .hidden_size = cfg.hidden_size,
        .num_heads = cfg.linear_num_heads,
        .head_dim = cfg.linear_head_dim,
        .conv_kernel = cfg.linear_conv_kernel,
        .lower_bound = cfg.linear_gate_lower_bound,
        .o_norm_eps = cfg.rms_norm_eps,
        .qkv_proj = qkv_q.data,
        .qkv_conv = qkv_conv_f32,
        .fbg_a_proj = fbg_q.data,
        .f_b_proj = f_b,
        .g_b_proj = g_b,
        .A_log = a_log,
        .dt_bias = dt_bias,
        .o_norm = o_norm_f32,
        .o_proj = o_q.data,
        .qkv_proj_scales = qkv_q.scales,
        .qkv_proj_biases = qkv_q.biases,
        .qkv_quant = qkv_q.cfg,
        .fbg_a_proj_scales = fbg_q.scales,
        .fbg_a_proj_biases = fbg_q.biases,
        .fbg_a_quant = fbg_q.cfg,
        .o_proj_scales = o_q.scales,
        .o_proj_biases = o_q.biases,
        .o_quant = o_q.cfg,
    };
    // Weight Arrays are owned by the defers above — do NOT call kda.deinit().

    var tmp_cache: ?glm53_kda.GlmKdaCache = null;
    defer if (tmp_cache) |*tc| tc.deinit();
    const cache = kda_cache orelse blk: {
        tmp_cache = try glm53_kda.GlmKdaCache.init(allocator, cfg.linear_conv_kernel, 3 * cfg.linear_num_heads * cfg.linear_head_dim, cfg.linear_num_heads, cfg.linear_head_dim);
        break :blk &tmp_cache.?;
    };

    const n = pre.normed.shape()[1];
    const kda_out = if (n > 1)
        try kda.prefill(allocator, pre.normed, cache) // [1, L, hidden] f32; advances cache
    else blk: {
        // Decode step: [1,1,hidden] -> GlmKDA.forward on [1,hidden].
        const step_in = try ops.reshape(ctx, pre.normed, &[_]i32{ 1, @intCast(cfg.hidden_size) });
        defer step_in.deinit();
        const step_out = try kda.forward(step_in, cache);
        defer step_out.deinit();
        break :blk try ops.reshape(ctx, step_out, &[_]i32{ 1, 1, @intCast(cfg.hidden_size) });
    };
    defer kda_out.deinit();
    // HC expand (apply_branch post): post * out + comb^T @ residual.
    // NOTE: the harness deliberately keeps the residual in f32 across layers
    // (the oracle rounds to bf16 after each branch); empirically the f32
    // trajectory stays closer to the kernel-generated fixtures
    // (logits cos 0.99591 f32 vs 0.99504 with bf16 rounding).
    return dsv4.mhcPost(ctx, kda_out, residual, pre.post_mix, pre.comb_norm, stream_raw);
}

/// Dense MLP branch (ffn_hc + post_attention_layernorm + SwiGLU MLP + expand).
pub fn denseFfnBranch(
    ctx: EagerContext,
    ws: WeightSource,
    cfg: *const glm53_loader.Glm53Config,
    layer_idx: u32,
    residual: Array,
    stream_raw: c.c.mlx_stream,
) !Array {
    const hidden: i32 = @intCast(cfg.hidden_size);
    var pbuf: [64]u8 = undefined;
    const p = fmtName(&pbuf, "language_model.model.layers.{d}", .{layer_idx});
    var nbuf: [256]u8 = undefined;
    var nbuf2: [256]u8 = undefined;

    var pre = try hcPre(ctx, ws, cfg, fmtName(&nbuf, "{s}.ffn_hc", .{p}), fmtName(&nbuf2, "{s}.post_attention_layernorm.weight", .{p}), residual, stream_raw);
    defer pre.deinit();

    var gate_up_q = try ws.quantized(fmtName(&nbuf, "{s}.mlp.gate_up_proj", .{p}), hidden);
    defer gate_up_q.deinit();
    var down_q = try ws.quantized(fmtName(&nbuf, "{s}.mlp.down_proj", .{p}), @intCast(cfg.intermediate_size));
    defer down_q.deinit();

    const mlp = glm53_attn.Glm53Mlp{
        .ctx = ctx,
        .gate_up_proj = gate_up_q.linear(),
        .down_proj = down_q.linear(),
        .swiglu_limit = cfg.swiglu_limit,
    };
    const mlp_out = try mlp.forward(pre.normed);
    defer mlp_out.deinit();
    return dsv4.mhcPost(ctx, mlp_out, residual, pre.post_mix, pre.comb_norm, stream_raw);
}

/// NoPE MLA attention branch (attn_hc + input_layernorm + MLA + expand).
/// Returns the updated residual and the indexer topk indices.
pub const MlaBranchResult = struct { residual: Array, topk: Array };

pub fn mlaAttnBranch(
    ctx: EagerContext,
    ws: WeightSource,
    allocator: std.mem.Allocator,
    cfg: *const glm53_loader.Glm53Config,
    layer_idx: u32,
    residual: Array,
    stream_raw: c.c.mlx_stream,
    mla_cache: ?*glm53_attn.Glm53MlaCache,
) !MlaBranchResult {
    const hidden: i32 = @intCast(cfg.hidden_size);
    var pbuf: [64]u8 = undefined;
    const p = fmtName(&pbuf, "language_model.model.layers.{d}", .{layer_idx});
    var nbuf: [256]u8 = undefined;
    var nbuf2: [256]u8 = undefined;
    var sabuf: [96]u8 = undefined;
    const sa = fmtName(&sabuf, "{s}.self_attn", .{p});

    var pre = try hcPre(ctx, ws, cfg, fmtName(&nbuf, "{s}.attn_hc", .{p}), fmtName(&nbuf2, "{s}.input_layernorm.weight", .{p}), residual, stream_raw);
    defer pre.deinit();

    var qkv_a_q = try ws.quantized(fmtName(&nbuf, "{s}.qkv_a_proj", .{sa}), hidden);
    defer qkv_a_q.deinit();
    var q_b_q = try ws.quantized(fmtName(&nbuf, "{s}.q_b_proj", .{sa}), @intCast(cfg.q_lora_rank));
    defer q_b_q.deinit();
    var o_q = try ws.quantized(fmtName(&nbuf, "{s}.o_proj", .{sa}), @intCast(cfg.num_attention_heads * cfg.v_head_dim));
    defer o_q.deinit();

    const q_a_ln = try ws.tensor(fmtName(&nbuf, "{s}.q_a_layernorm.weight", .{sa}));
    defer q_a_ln.deinit();
    const q_a_ln_f32 = try ops.astype(ctx, q_a_ln, .float32);
    defer q_a_ln_f32.deinit();
    const kv_a_ln = try ws.tensor(fmtName(&nbuf, "{s}.kv_a_layernorm.weight", .{sa}));
    defer kv_a_ln.deinit();
    const kv_a_ln_f32 = try ops.astype(ctx, kv_a_ln, .float32);
    defer kv_a_ln_f32.deinit();

    // Per-head MultiLinear weights: dequantize once, keep both orientations.
    var embed_q_q = try ws.quantized(fmtName(&nbuf, "{s}.embed_q", .{sa}), @intCast(cfg.qk_nope_head_dim));
    defer embed_q_q.deinit();
    const embed_q_w = try quantize_mod.dequantizeAs(ctx, .{ .data = embed_q_q.data, .scales = embed_q_q.scales, .biases = embed_q_q.biases, .config = embed_q_q.cfg, .original_shape = &[_]i32{} }, .bfloat16);
    defer embed_q_w.deinit(); // [H, kv_lora, qk_nope]
    const embed_q_t = try ops.transposeAxes(ctx, embed_q_w, &[_]i32{ 0, 2, 1 });
    defer embed_q_t.deinit(); // [H, qk_nope, kv_lora]

    var unembed_q = try ws.quantized(fmtName(&nbuf, "{s}.unembed_out", .{sa}), @intCast(cfg.kv_lora_rank));
    defer unembed_q.deinit();
    const unembed_w = try quantize_mod.dequantizeAs(ctx, .{ .data = unembed_q.data, .scales = unembed_q.scales, .biases = unembed_q.biases, .config = unembed_q.cfg, .original_shape = &[_]i32{} }, .bfloat16);
    defer unembed_w.deinit(); // [H, v_head, kv_lora]
    const unembed_out_t = try ops.transposeAxes(ctx, unembed_w, &[_]i32{ 0, 2, 1 });
    defer unembed_out_t.deinit(); // [H, kv_lora, v_head]

    // Indexer weights.
    var ixbuf: [112]u8 = undefined;
    const ix = fmtName(&ixbuf, "{s}.indexer", .{sa});
    var wq_b_q = try ws.quantized(fmtName(&nbuf, "{s}.wq_b", .{ix}), @intCast(cfg.q_lora_rank));
    defer wq_b_q.deinit();
    var wk_q = try ws.quantized(fmtName(&nbuf, "{s}.wk", .{ix}), hidden);
    defer wk_q.deinit();
    var weights_proj_q = try ws.quantized(fmtName(&nbuf, "{s}.weights_proj", .{ix}), hidden);
    defer weights_proj_q.deinit();

    const k_norm_w = try ws.tensor(fmtName(&nbuf, "{s}.k_norm.weight", .{ix}));
    defer k_norm_w.deinit();
    const k_norm_w_f32 = try ops.astype(ctx, k_norm_w, .float32);
    defer k_norm_w_f32.deinit();
    const k_norm_b = try ws.tensor(fmtName(&nbuf, "{s}.k_norm.bias", .{ix}));
    defer k_norm_b.deinit();
    const k_norm_b_f32 = try ops.astype(ctx, k_norm_b, .float32);
    defer k_norm_b_f32.deinit();

    const ape = try ws.tensor(fmtName(&nbuf, "{s}.index_kpool_compress_ape", .{ix}));
    defer ape.deinit();
    const ape_f32 = try ops.astype(ctx, ape, .float32);
    defer ape_f32.deinit();
    const compress_gate = try ws.tensor(fmtName(&nbuf, "{s}.index_kpool_compress_gate", .{ix}));
    defer compress_gate.deinit();
    const compress_gate_f32 = try ops.astype(ctx, compress_gate, .float32);
    defer compress_gate_f32.deinit();
    const compress_gate_t = try ops.transpose(ctx, compress_gate_f32);
    defer compress_gate_t.deinit(); // [hidden, head_dim]

    const indexer = glm53_attn.Glm53Indexer{
        .ctx = ctx,
        .n_heads = @intCast(cfg.index_n_heads),
        .head_dim = @intCast(cfg.index_head_dim),
        .index_topk = @intCast(cfg.index_topk),
        .kpool = @intCast(cfg.index_kpool),
        .always_select_tail = cfg.index_kpool_always_select_tail,
        .wq_b = wq_b_q.linear(),
        .wk = wk_q.linear(),
        .weights_proj = weights_proj_q.linear(),
        .k_norm_w = k_norm_w_f32,
        .k_norm_b = k_norm_b_f32,
        .ape = ape_f32,
        .compress_gate_t = compress_gate_t,
    };
    const mla = glm53_attn.Glm53Mla{
        .ctx = ctx,
        .num_heads = @intCast(cfg.num_attention_heads),
        .q_lora_rank = @intCast(cfg.q_lora_rank),
        .kv_lora_rank = @intCast(cfg.kv_lora_rank),
        .qk_nope_head_dim = @intCast(cfg.qk_nope_head_dim),
        .v_head_dim = @intCast(cfg.v_head_dim),
        .rms_norm_eps = cfg.rms_norm_eps,
        .qkv_a_proj = qkv_a_q.linear(),
        .q_a_layernorm_w = q_a_ln_f32,
        .q_b_proj = q_b_q.linear(),
        .kv_a_layernorm_w = kv_a_ln_f32,
        .embed_q_t = embed_q_t,
        .embed_q_w = embed_q_w,
        .unembed_out_t = unembed_out_t,
        .o_proj = o_q.linear(),
        .indexer = indexer,
    };

    const n = pre.normed.shape()[1];
    const result = if (n > 1 or mla_cache == null)
        try mla.forward(allocator, pre.normed, mla_cache)
    else
        try mla.forwardStep(allocator, pre.normed, mla_cache.?);
    defer result.out.deinit();
    errdefer result.topk.deinit();
    const out = try dsv4.mhcPost(ctx, result.out, residual, pre.post_mix, pre.comb_norm, stream_raw);
    return .{ .residual = out, .topk = result.topk };
}

// ============================================================
// MoE (sparse layers: sigmoid gate + correction bias, top-8 of 288,
// limited-SwiGLU experts + 1 shared expert)
// ============================================================

/// One routed expert's three projections (gate/up/down), borrowed arrays.
pub const ExpertWeights = struct {
    gate: glm53_attn.QuantLinear,
    up: glm53_attn.QuantLinear,
    down: glm53_attn.QuantLinear,
};

/// Provider vtable: returns the weights of expert `e` (arrays owned by the
/// provider's internal cache, released via releaseAll).
pub const ExpertProvider = struct {
    ctx: *anyopaque,
    loadFn: *const fn (ctx: *anyopaque, e: u32) anyerror!ExpertWeights,
    releaseAllFn: *const fn (ctx: *anyopaque) void,

    pub fn load(self: ExpertProvider, e: u32) !ExpertWeights {
        return self.loadFn(self.ctx, e);
    }
    pub fn releaseAll(self: ExpertProvider) void {
        self.releaseAllFn(self.ctx);
    }
};

pub const Glm53Moe = struct {
    ctx: EagerContext,
    n_experts: i32, // 288
    top_k: i32, // 8
    scaling: f32, // 2.5
    norm_topk_prob: bool,
    swiglu_limit: f32,
    gate_w_t: Array, // [hidden, n_experts] f32 (transposed at load)
    gate_bias: Array, // [n_experts] f32
    shared: glm53_attn.Glm53Mlp,

    /// Router: oracle `_expert_select` with n_group=1 (no group masking).
    /// x_flat: [N, hidden]. Returns evaluated indices [N,K] i32 and weights
    /// [N,K] f32 (normalized, scaled).
    pub const Route = struct { indices: Array, weights: Array };

    pub fn route(self: *const Glm53Moe, x_flat: Array) !Route {
        const ctx = self.ctx;
        const n = x_flat.shape()[0];
        const x_f32 = try ops.astype(ctx, x_flat, .float32);
        defer x_f32.deinit();
        const logits = try ops.matmul(ctx, x_f32, self.gate_w_t);
        defer logits.deinit(); // [N, E]
        const scores = try ops.sigmoid(ctx, logits);
        defer scores.deinit();
        const bias2 = try ops.expandDims(ctx, self.gate_bias, 0);
        defer bias2.deinit();
        const choice = try ops.add(ctx, scores, bias2);
        defer choice.deinit();
        const neg = try ops.negative(ctx, choice);
        defer neg.deinit();
        const part = try sort_mod.argpartitionAxis(ctx, neg, self.top_k - 1, -1);
        defer part.deinit();
        const sel = try ops.slice(ctx, part, &[_]i32{ 0, 0 }, &[_]i32{ n, self.top_k }, &[_]i32{ 1, 1 });
        defer sel.deinit();
        const indices = try ops.astype(ctx, sel, .int32);
        errdefer indices.deinit();

        var weights = try shape_mod.takeAlongAxis(ctx, scores, indices, -1);
        errdefer weights.deinit();
        if (self.top_k > 1 and self.norm_topk_prob) {
            const wsum = try reduce_mod.sumAxis(ctx, weights, -1, true);
            defer wsum.deinit();
            const eps = try ops.scalarF32(ctx, 1e-20);
            defer eps.deinit();
            const denom = try ops.add(ctx, wsum, eps);
            defer denom.deinit();
            const normed = try ops.divide(ctx, weights, denom);
            weights.deinit();
            weights = normed;
        }
        const scale_arr = try ops.scalarF32(ctx, self.scaling);
        defer scale_arr.deinit();
        const scaled = try ops.multiply(ctx, weights, scale_arr);
        weights.deinit();
        weights = scaled;

        try indices.eval();
        try weights.eval();
        return .{ .indices = indices, .weights = weights };
    }

    /// One routed expert: down(silu(min(gate, limit)) * clip(up, ±limit)).
    fn expertForward(self: *const Glm53Moe, x_e: Array, w: ExpertWeights) !Array {
        const ctx = self.ctx;
        const gate = try w.gate.apply(ctx, x_e);
        defer gate.deinit();
        const up = try w.up.apply(ctx, x_e);
        defer up.deinit();
        const limit_pos = try ops.scalarF32(ctx, self.swiglu_limit);
        defer limit_pos.deinit();
        const limit_neg = try ops.scalarF32(ctx, -self.swiglu_limit);
        defer limit_neg.deinit();
        const gate_c = try math_mod.minimum(ctx, gate, limit_pos);
        defer gate_c.deinit();
        const up_lo = try math_mod.maximum(ctx, up, limit_neg);
        defer up_lo.deinit();
        const up_c = try math_mod.minimum(ctx, up_lo, limit_pos);
        defer up_c.deinit();
        const act = try activations.silu(ctx, gate_c);
        defer act.deinit();
        const h = try ops.multiply(ctx, act, up_c);
        defer h.deinit();
        return w.down.apply(ctx, h);
    }

    /// Streaming MoE for serving: routed experts computed by the Task-4
    /// ExpertStreamProvider (batched SwitchGLU, scores applied after), shared
    /// expert added at the end — same math as `forward`, batched reduction.
    pub fn forwardStreaming(
        self: *const Glm53Moe,
        allocator: std.mem.Allocator,
        x: Array,
        layer_idx: usize,
        sp: *expert_stream.ExpertStreamProvider,
    ) !Array {
        const ctx = self.ctx;
        const sh = x.shape();
        const b = sh[0];
        const l = sh[1];
        const hidden = sh[2];
        const n = b * l;

        const shared_out = try self.shared.forward(x);
        defer shared_out.deinit();

        const x_flat = try ops.reshape(ctx, x, &[_]i32{ n, hidden });
        defer x_flat.deinit();

        var rte = try self.route(x_flat);
        defer rte.indices.deinit();
        defer rte.weights.deinit();

        const routed = try sp.streamForward(layer_idx, x_flat, rte.indices, rte.weights);
        defer routed.deinit(); // [N, hidden]
        const routed3 = try ops.reshape(ctx, routed, &[_]i32{ b, l, hidden });
        defer routed3.deinit();
        _ = allocator;
        return ops.add(ctx, routed3, shared_out);
    }

    /// Full MoE: shared expert + sum_k w_k * expert_k(x). Expert weights are
    /// pulled from the provider per selected expert id (provider caches).
    /// x: [B, L, hidden]; returns [B, L, hidden].
    pub fn forward(self: *const Glm53Moe, allocator: std.mem.Allocator, x: Array, provider: ExpertProvider) !Array {
        const ctx = self.ctx;
        const sh = x.shape();
        const b = sh[0];
        const l = sh[1];
        const hidden = sh[2];
        const n = b * l;

        const shared_out = try self.shared.forward(x);
        defer shared_out.deinit();

        const x_flat = try ops.reshape(ctx, x, &[_]i32{ n, hidden });
        defer x_flat.deinit();

        var rte = try self.route(x_flat);
        defer rte.indices.deinit();
        defer rte.weights.deinit();
        const idx_cpu = try rte.indices.dataSlice(i32);
        const w_cpu = try rte.weights.dataSlice(f32);
        const k_usize: usize = @intCast(self.top_k);

        var rows = try allocator.alloc(Array, @intCast(n));
        var made: usize = 0;
        defer {
            for (rows[0..made]) |r| r.deinit();
            allocator.free(rows);
        }

        var t: i32 = 0;
        while (t < n) : (t += 1) {
            const x_t = try ops.slice(ctx, x_flat, &[_]i32{ t, 0 }, &[_]i32{ t + 1, hidden }, &[_]i32{ 1, 1 });
            defer x_t.deinit();
            var acc: ?Array = null;
            defer if (acc) |a| a.deinit();
            for (0..k_usize) |k_i| {
                const e: u32 = @intCast(idx_cpu[@as(usize, @intCast(t)) * k_usize + k_i]);
                const w_val = w_cpu[@as(usize, @intCast(t)) * k_usize + k_i];
                const ew = try provider.load(e);
                const y = try self.expertForward(x_t, ew);
                defer y.deinit();
                const w_arr = try ops.scalarF32(ctx, w_val);
                defer w_arr.deinit();
                const yw = try ops.multiply(ctx, y, w_arr);
                defer yw.deinit();
                if (acc) |a| {
                    const next = try ops.add(ctx, a, yw);
                    a.deinit();
                    acc = next;
                } else {
                    acc = try ops.copy(ctx, yw);
                }
            }
            rows[@intCast(t)] = try ops.copy(ctx, acc.?);
            made += 1;
        }

        const routed = try shape_mod.concatenateAxis(ctx, rows, 0);
        defer routed.deinit(); // [N, hidden]
        const routed3 = try ops.reshape(ctx, routed, &[_]i32{ b, l, hidden });
        defer routed3.deinit();
        return ops.add(ctx, routed3, shared_out);
    }
};

/// Expert weight provider backed by on-demand pread row slices, one cache
/// per MoE layer.
pub const ReaderExpertProvider = struct {
    reader: *safetensors_reader.PartialTensorReader,
    ectx: EagerContext,
    allocator: std.mem.Allocator,
    cfg: *const glm53_loader.Glm53Config,
    switch_prefix: []const u8, // "language_model.model.layers.{i}.mlp.switch_mlp"
    cache: std.AutoHashMap(u32, QuantParts3),

    const QuantParts3 = struct { gate: QuantParts, up: QuantParts, down: QuantParts };

    fn loadFn(opaque_ctx: *anyopaque, e: u32) anyerror!ExpertWeights {
        const self: *ReaderExpertProvider = @ptrCast(@alignCast(opaque_ctx));
        const gop = try self.cache.getOrPut(e);
        if (!gop.found_existing) {
            errdefer self.cache.removeByPtr(gop.key_ptr);
            gop.value_ptr.* = try self.loadOne(e);
        }
        const v = gop.value_ptr;
        return .{ .gate = v.gate.linear(), .up = v.up.linear(), .down = v.down.linear() };
    }

    fn releaseAllFn(opaque_ctx: *anyopaque) void {
        const self: *ReaderExpertProvider = @ptrCast(@alignCast(opaque_ctx));
        var it = self.cache.valueIterator();
        while (it.next()) |v| {
            v.gate.deinit();
            v.up.deinit();
            v.down.deinit();
        }
        self.cache.clearRetainingCapacity();
    }

    fn loadOne(self: *ReaderExpertProvider, e: u32) !QuantParts3 {
        const ids = [_]u32{e};
        const cfg = self.cfg;
        const reader = self.reader;
        const ctx = self.ectx;
        var result: QuantParts3 = undefined;
        var done: usize = 0;
        errdefer {
            // Release whichever projections were fully built.
            if (done > 0) result.gate.deinit();
            if (done > 1) result.up.deinit();
            if (done > 2) result.down.deinit();
        }
        inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, pi| {
            const in_f: i32 = if (pi == 2) @intCast(cfg.moe_intermediate_size) else @intCast(cfg.hidden_size);
            var bb: [160]u8 = undefined;
            const base = fmtName(&bb, "{s}.{s}", .{ self.switch_prefix, proj });
            const qp = loadExpertQuant(reader, ctx, base, &ids, in_f);
            switch (pi) {
                0 => result.gate = try qp,
                1 => result.up = try qp,
                2 => result.down = try qp,
                else => unreachable,
            }
            done += 1;
        }
        return result;
    }
};

// ============================================================
// Model assembly
// ============================================================

/// Segment hook: called with the post-attn-branch residual at dump layers
/// ("layer{i}_kda_out"/"layer{i}_mla_out"/"layer{i}_moe_out") plus
/// "embed"/"hidden_final". The tensor is borrowed (f32 copy not guaranteed;
/// hooks should astype/copy if they keep it).
pub const SegmentHook = struct {
    ctx: *anyopaque,
    func: *const fn (ctx: *anyopaque, name: []const u8, tensor: Array) anyerror!void,
};

pub const Glm53Model = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    cfg: *const glm53_loader.Glm53Config,
    ws: WeightSource,
    /// Expert row reads go through this (mmap zero-copy / pread fallback).
    reader: *safetensors_reader.PartialTensorReader,
    stream: c.c.mlx_stream,

    /// embed_tokens lookup → [1, L, hidden] (pre-expansion, matches the
    /// oracle's `embed` dump point).
    fn embedTokens(self: *const Glm53Model, ids: []const i32) !Array {
        const ctx = self.ctx;
        const hidden: i32 = @intCast(self.cfg.hidden_size);
        var embed_q = try self.ws.quantized("language_model.model.embed_tokens", hidden);
        defer embed_q.deinit();
        const embed_w = try quantize_mod.dequantizeAs(ctx, .{
            .data = embed_q.data,
            .scales = embed_q.scales,
            .biases = embed_q.biases,
            .config = embed_q.cfg,
            .original_shape = &[_]i32{},
        }, .bfloat16);
        defer embed_w.deinit();
        try embed_w.eval();

        const id_arr = try Array.fromData(self.allocator, i32, ids, &[_]i32{@intCast(ids.len)});
        defer id_arr.deinit();
        const rows = try shape_mod.takeAxis(ctx, embed_w, id_arr, 0);
        defer rows.deinit();
        return ops.reshape(ctx, rows, &[_]i32{ 1, @intCast(ids.len), hidden });
    }

    /// hidden [1, L, hc_mult, hidden] → norm(mean(h, axis=2)) → lm_head logits.
    fn finalLogits(self: *const Glm53Model, h: Array) !Array {
        const ctx = self.ctx;
        const hidden: i32 = @intCast(self.cfg.hidden_size);
        const h_f32 = try ops.astype(ctx, h, .float32);
        defer h_f32.deinit();
        const mean_h = try reduce_mod.meanAxis(ctx, h_f32, 2, false);
        defer mean_h.deinit(); // [1, L, hidden]
        const norm_w = try self.ws.tensor("language_model.model.norm.weight");
        defer norm_w.deinit();
        const norm_w_f32 = try ops.astype(ctx, norm_w, .float32);
        defer norm_w_f32.deinit();
        const normed = try fast_mod.rmsNorm(ctx, mean_h, norm_w_f32, self.cfg.rms_norm_eps);
        defer normed.deinit();

        var head_q = try self.ws.quantized("language_model.lm_head", hidden);
        defer head_q.deinit();
        return head_q.linear().apply(ctx, normed);
    }

    pub const ForwardOpts = struct {
        /// Debug subset: when non-null, only listed layers run their branches;
        /// other layers pass the residual through unchanged.
        layer_filter: ?[]const u32 = null,
        /// Early stop: run only layers 0..=stop_after (cheap segment reruns).
        stop_after: ?u32 = null,
        hook: ?SegmentHook = null,
    };

    /// Full eager forward. Returns logits [1, L, vocab] f32.
    pub fn forward(self: *const Glm53Model, ids: []const i32, opts: ForwardOpts) !Array {
        const allocator = self.allocator;
        const cfg = self.cfg;
        var nbuf: [256]u8 = undefined;

        const emb = try self.embedTokens(ids);
        defer emb.deinit();
        if (opts.hook) |hk| try hk.func(hk.ctx, "embed", emb);
        var h = try dsv4.expandToMHC(self.ctx, emb, self.cfg.hc_mult, self.stream);
        defer h.deinit();

        var layer: u32 = 0;
        while (layer < cfg.num_hidden_layers) : (layer += 1) {
            if (opts.stop_after) |sa| {
                if (layer > sa) break;
            }
            if (opts.layer_filter) |lf| {
                var listed = false;
                for (lf) |x| {
                    if (x == layer) listed = true;
                }
                if (!listed) continue;
            }

            if (cfg.isFullAttention(layer)) {
                const res = try mlaAttnBranch(self.ctx, self.ws, allocator, cfg, layer, h, self.stream, null);
                h.deinit();
                h = res.residual;
                defer res.topk.deinit();
                if (opts.hook) |hk| {
                    const name = fmtName(&nbuf, "layer{d}_indexer_topk", .{layer});
                    try hk.func(hk.ctx, name, res.topk);
                }
                if (opts.hook) |hk| {
                    const name = fmtName(&nbuf, "layer{d}_mla_out", .{layer});
                    try hk.func(hk.ctx, name, h);
                }
            } else {
                const next = try kdaAttnBranch(self.ctx, self.ws, allocator, cfg, layer, h, self.stream, null);
                h.deinit();
                h = next;
                if (opts.hook) |hk| {
                    const name = fmtName(&nbuf, "layer{d}_kda_out", .{layer});
                    try hk.func(hk.ctx, name, h);
                }
            }

            if (cfg.isMoE(layer)) {
                const next = try self.moeFfnBranch(layer, h);
                h.deinit();
                h = next;
                if (opts.hook) |hk| {
                    const name = fmtName(&nbuf, "layer{d}_moe_out", .{layer});
                    try hk.func(hk.ctx, name, h);
                }
            } else {
                const next = try denseFfnBranch(self.ctx, self.ws, cfg, layer, h, self.stream);
                h.deinit();
                h = next;
            }
            try h.eval();
        }

        // hidden_final / logits are only meaningful on a full run (no layer
        // filter, no early stop): skip both otherwise so early segment reruns
        // never emit a misleading hidden_final.bin or load lm_head.
        const full_run = opts.layer_filter == null and
            (opts.stop_after == null or opts.stop_after.? >= cfg.num_hidden_layers - 1);
        if (!full_run) {
            const empty = try creation_mod.zeros(self.ctx, &[_]i32{0}, .float32);
            return empty;
        }

        if (opts.hook) |hk| {
            const h_f32 = try ops.astype(self.ctx, h, .float32);
            defer h_f32.deinit();
            const mean_h = try reduce_mod.meanAxis(self.ctx, h_f32, 2, false);
            defer mean_h.deinit();
            const norm_w = try self.ws.tensor("language_model.model.norm.weight");
            defer norm_w.deinit();
            const norm_w_f32 = try ops.astype(self.ctx, norm_w, .float32);
            defer norm_w_f32.deinit();
            const hidden_final = try fast_mod.rmsNorm(self.ctx, mean_h, norm_w_f32, self.cfg.rms_norm_eps);
            defer hidden_final.deinit();
            try hk.func(hk.ctx, "hidden_final", hidden_final);
        }
        return self.finalLogits(h);
    }

    /// Sparse MoE ffn branch (ffn_hc + post_attention_layernorm + MoE).
    fn moeFfnBranch(self: *const Glm53Model, layer_idx: u32, residual: Array) !Array {
        return moeFfnBranchImpl(self.ctx, self.ws, self.reader, null, self.allocator, self.cfg, layer_idx, residual, self.stream);
    }
};

/// Sparse MoE ffn branch (ffn_hc + post_attention_layernorm + MoE).
/// When `sp` is set (serving), routed experts go through the Task-4
/// ExpertStreamProvider; otherwise per-expert rows are pread on demand via
/// ReaderExpertProvider.
fn moeFfnBranchImpl(
    ctx: EagerContext,
    ws: WeightSource,
    reader: *safetensors_reader.PartialTensorReader,
    sp: ?*expert_stream.ExpertStreamProvider,
    allocator: std.mem.Allocator,
    cfg: *const glm53_loader.Glm53Config,
    layer_idx: u32,
    residual: Array,
    stream: c.c.mlx_stream,
) !Array {
    const hidden: i32 = @intCast(cfg.hidden_size);
    var pbuf: [64]u8 = undefined;
    const p = fmtName(&pbuf, "language_model.model.layers.{d}", .{layer_idx});
    var nbuf: [256]u8 = undefined;
    var nbuf2: [256]u8 = undefined;

    var pre = try hcPre(ctx, ws, cfg, fmtName(&nbuf, "{s}.ffn_hc", .{p}), fmtName(&nbuf2, "{s}.post_attention_layernorm.weight", .{p}), residual, stream);
    defer pre.deinit();

    var mpbuf: [96]u8 = undefined;
    const mlp_p = fmtName(&mpbuf, "{s}.mlp", .{p});
    const gate_w = try ws.tensor(fmtName(&nbuf, "{s}.gate.weight", .{mlp_p}));
    defer gate_w.deinit();
    const gate_w_f32 = try ops.astype(ctx, gate_w, .float32);
    defer gate_w_f32.deinit();
    const gate_w_t = try ops.transpose(ctx, gate_w_f32);
    defer gate_w_t.deinit();
    const gate_bias = try ws.tensor(fmtName(&nbuf, "{s}.gate.e_score_correction_bias", .{mlp_p}));
    defer gate_bias.deinit();
    const gate_bias_f32 = try ops.astype(ctx, gate_bias, .float32);
    defer gate_bias_f32.deinit();

    var shared_gate_up_q = try ws.quantized(fmtName(&nbuf, "{s}.shared_experts.gate_up_proj", .{mlp_p}), hidden);
    defer shared_gate_up_q.deinit();
    var shared_down_q = try ws.quantized(fmtName(&nbuf, "{s}.shared_experts.down_proj", .{mlp_p}), @intCast(cfg.moe_intermediate_size * cfg.n_shared_experts));
    defer shared_down_q.deinit();

    const moe = Glm53Moe{
        .ctx = ctx,
        .n_experts = @intCast(cfg.n_routed_experts),
        .top_k = @intCast(cfg.num_experts_per_tok),
        .scaling = cfg.routed_scaling_factor,
        .norm_topk_prob = cfg.norm_topk_prob,
        .swiglu_limit = cfg.swiglu_limit,
        .gate_w_t = gate_w_t,
        .gate_bias = gate_bias_f32,
        .shared = .{
            .ctx = ctx,
            .gate_up_proj = shared_gate_up_q.linear(),
            .down_proj = shared_down_q.linear(),
            .swiglu_limit = cfg.swiglu_limit,
        },
    };

    const moe_out = if (sp) |provider|
        try moe.forwardStreaming(allocator, pre.normed, layer_idx, provider)
    else blk: {
        var spbuf: [160]u8 = undefined;
        var provider = ReaderExpertProvider{
            .reader = reader,
            .ectx = ctx,
            .allocator = allocator,
            .cfg = cfg,
            .switch_prefix = fmtName(&spbuf, "{s}.switch_mlp", .{mlp_p}),
            .cache = std.AutoHashMap(u32, ReaderExpertProvider.QuantParts3).init(allocator),
        };
        defer provider.cache.deinit();
        defer ReaderExpertProvider.releaseAllFn(&provider);
        break :blk try moe.forward(allocator, pre.normed, .{
            .ctx = &provider,
            .loadFn = ReaderExpertProvider.loadFn,
            .releaseAllFn = ReaderExpertProvider.releaseAllFn,
        });
    };
    defer moe_out.deinit();
    // Evaluate before returning: the branch defers release the expert
    // weights/provider cache, and the caller's hook evals the result only
    // after those defers have run — the lazy graph must be materialized
    // while its inputs are still alive.
    const out = try dsv4.mhcPost(ctx, moe_out, residual, pre.post_mix, pre.comb_norm, stream);
    try out.eval();
    return out;
}

// ============================================================
// Serving model: resident backbone + per-layer decode caches
// ============================================================

/// embedTokens on a WeightSource (shared by the harness model and serving).
pub fn embedTokensWs(ctx: EagerContext, ws: WeightSource, allocator: std.mem.Allocator, cfg: *const glm53_loader.Glm53Config, ids: []const i32) !Array {
    const hidden: i32 = @intCast(cfg.hidden_size);
    var embed_q = try ws.quantized("language_model.model.embed_tokens", hidden);
    defer embed_q.deinit();
    const embed_w = try quantize_mod.dequantizeAs(ctx, .{
        .data = embed_q.data,
        .scales = embed_q.scales,
        .biases = embed_q.biases,
        .config = embed_q.cfg,
        .original_shape = &[_]i32{},
    }, .bfloat16);
    defer embed_w.deinit();
    try embed_w.eval();

    const id_arr = try Array.fromData(allocator, i32, ids, &[_]i32{@intCast(ids.len)});
    defer id_arr.deinit();
    const rows = try shape_mod.takeAxis(ctx, embed_w, id_arr, 0);
    defer rows.deinit();
    return ops.reshape(ctx, rows, &[_]i32{ 1, @intCast(ids.len), hidden });
}

/// hidden [1, L, hc_mult, hidden] → norm(mean(h, axis=2)) → lm_head logits.
pub fn finalLogitsWs(ctx: EagerContext, ws: WeightSource, cfg: *const glm53_loader.Glm53Config, h: Array) !Array {
    const hidden: i32 = @intCast(cfg.hidden_size);
    const h_f32 = try ops.astype(ctx, h, .float32);
    defer h_f32.deinit();
    const mean_h = try reduce_mod.meanAxis(ctx, h_f32, 2, false);
    defer mean_h.deinit(); // [1, L, hidden]
    const norm_w = try ws.tensor("language_model.model.norm.weight");
    defer norm_w.deinit();
    const norm_w_f32 = try ops.astype(ctx, norm_w, .float32);
    defer norm_w_f32.deinit();
    const normed = try fast_mod.rmsNorm(ctx, mean_h, norm_w_f32, cfg.rms_norm_eps);
    defer normed.deinit();

    var head_q = try ws.quantized("language_model.lm_head", hidden);
    defer head_q.deinit();
    return head_q.linear().apply(ctx, normed);
}

/// Per-layer decode state.
pub const Glm53LayerCache = union(enum) {
    kda: glm53_kda.GlmKdaCache,
    mla: glm53_attn.Glm53MlaCache,

    pub fn deinit(self: *Glm53LayerCache) void {
        switch (self.*) {
            .kda => |*kc| kc.deinit(),
            .mla => |*mc| mc.deinit(),
        }
    }
};

/// Resident serving model: backbone weights live in a map (loaded once,
/// borrowed via WeightSource.fromMap COW copies); MoE experts stream through
/// the Task-4 ExpertStreamProvider (or per-call pread when smelt is off).
/// Per-layer caches (KDA conv+ssm state / MLA latent + indexer history) are
/// internal — the KVCacheStrategy array from the engine is unused.
///
/// LIMITATION: caches are not reset between requests. forwardTokens treats
/// any L>1 input as a fresh prefill and resets state; the engine's decode
/// calls are always L=1.
pub const Glm53ServingModel = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    cfg: *const glm53_loader.Glm53Config,
    stream: c.c.mlx_stream,
    ws: WeightSource,
    ws_backing: WeightSource.MapBacked,
    weights: std.StringHashMap(Array), // resident backbone, owned
    reader: *safetensors_reader.PartialTensorReader,
    stream_provider: ?*expert_stream.ExpertStreamProvider,
    caches: []Glm53LayerCache,

    pub fn init(
        allocator: std.mem.Allocator,
        ctx: EagerContext,
        cfg: *const glm53_loader.Glm53Config,
        stream: c.c.mlx_stream,
        weights: std.StringHashMap(Array),
        reader: *safetensors_reader.PartialTensorReader,
        stream_provider: ?*expert_stream.ExpertStreamProvider,
    ) !Glm53ServingModel {
        const caches = try allocator.alloc(Glm53LayerCache, cfg.num_hidden_layers);
        errdefer allocator.free(caches);
        for (caches, 0..) |*lc, i| {
            if (cfg.isFullAttention(@intCast(i))) {
                lc.* = .{ .mla = .{} };
            } else {
                lc.* = .{ .kda = try glm53_kda.GlmKdaCache.init(allocator, cfg.linear_conv_kernel, 3 * cfg.linear_num_heads * cfg.linear_head_dim, cfg.linear_num_heads, cfg.linear_head_dim) };
            }
        }
        var self = Glm53ServingModel{
            .allocator = allocator,
            .ctx = ctx,
            .cfg = cfg,
            .stream = stream,
            .ws = undefined,
            .ws_backing = undefined,
            .weights = weights,
            .reader = reader,
            .stream_provider = stream_provider,
            .caches = caches,
        };
        self.ws = WeightSource.fromMap(&self.weights, ctx, &self.ws_backing);
        return self;
    }

    pub fn deinit(self: *Glm53ServingModel) void {
        for (self.caches) |*lc| lc.deinit();
        self.allocator.free(self.caches);
        // weights map entries are owned by the caller (registry adapter).
    }

    /// Reset all per-layer decode state (fresh request).
    pub fn reset(self: *Glm53ServingModel) !void {
        for (self.caches) |*lc| lc.deinit();
        for (self.caches, 0..) |*lc, i| {
            if (self.cfg.isFullAttention(@intCast(i))) {
                lc.* = .{ .mla = .{} };
            } else {
                lc.* = .{ .kda = try glm53_kda.GlmKdaCache.init(self.allocator, self.cfg.linear_conv_kernel, 3 * self.cfg.linear_num_heads * self.cfg.linear_head_dim, self.cfg.linear_num_heads, self.cfg.linear_head_dim) };
            }
        }
    }

    /// ids: [L] u32 token ids → logits [1, L, vocab] f32. L>1 = prefill
    /// (resets state first); L=1 = cached decode step.
    pub fn forwardTokens(self: *Glm53ServingModel, ids: []const u32) !Array {
        const allocator = self.allocator;
        const cfg = self.cfg;

        if (ids.len > 1) try self.reset();

        const ids_i32 = try allocator.alloc(i32, ids.len);
        defer allocator.free(ids_i32);
        for (ids, 0..) |t, i| ids_i32[i] = @intCast(t);

        const emb = try embedTokensWs(self.ctx, self.ws, allocator, cfg, ids_i32);
        defer emb.deinit();
        var h = try dsv4.expandToMHC(self.ctx, emb, cfg.hc_mult, self.stream);
        defer h.deinit();

        var layer: u32 = 0;
        while (layer < cfg.num_hidden_layers) : (layer += 1) {
            if (cfg.isFullAttention(layer)) {
                const res = try mlaAttnBranch(self.ctx, self.ws, allocator, cfg, layer, h, self.stream, &self.caches[layer].mla);
                h.deinit();
                h = res.residual;
                res.topk.deinit();
            } else {
                const next = try kdaAttnBranch(self.ctx, self.ws, allocator, cfg, layer, h, self.stream, &self.caches[layer].kda);
                h.deinit();
                h = next;
            }
            if (cfg.isMoE(layer)) {
                const next = try moeFfnBranchImpl(self.ctx, self.ws, self.reader, self.stream_provider, allocator, cfg, layer, h, self.stream);
                h.deinit();
                h = next;
            } else {
                const next = try denseFfnBranch(self.ctx, self.ws, cfg, layer, h, self.stream);
                h.deinit();
                h = next;
            }
            try h.eval();
        }

        const logits = try finalLogitsWs(self.ctx, self.ws, cfg, h);
        try logits.eval();
        return logits;
    }
};

// ============================================================
// Tests
// ============================================================

const test_c = @import("mlx").c;
const creation_mod = @import("mlx").creation;

fn makeTestQuantLinear(ctx: EagerContext, allocator: std.mem.Allocator, out_f: i32, in_f: i32, seed: i64) !glm53_attn.QuantLinear {
    const n: usize = @intCast(out_f * in_f);
    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 7 + seed, 13) - 6)) * 0.02;
    }
    const w = try Array.fromData(allocator, f32, buf, &[_]i32{ out_f, in_f });
    defer w.deinit();
    const qw = try quantize_mod.quantize(ctx, w, .{ .bits = 8, .group_size = 32, .mode = .affine });
    allocator.free(qw.original_shape);
    return .{ .w = qw.data, .scales = qw.scales, .biases = qw.biases, .cfg = .{ .bits = 8, .group_size = 32, .mode = .affine } };
}

fn makeTestArray(allocator: std.mem.Allocator, shape: []const i32, seed: i64) !Array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 5 + seed, 17) - 8)) * 0.05;
    }
    return Array.fromData(allocator, f32, buf, shape);
}

const TestExpertProvider = struct {
    ctx: EagerContext,
    allocator: std.mem.Allocator,
    cache: std.AutoHashMap(u32, [3]glm53_attn.QuantLinear),

    fn loadFn(opaque_ctx: *anyopaque, e: u32) anyerror!ExpertWeights {
        const self: *TestExpertProvider = @ptrCast(@alignCast(opaque_ctx));
        const gop = try self.cache.getOrPut(e);
        if (!gop.found_existing) {
            gop.value_ptr.* = .{
                try makeTestQuantLinear(self.ctx, self.allocator, 32, 64, @intCast(11 + e)),
                try makeTestQuantLinear(self.ctx, self.allocator, 32, 64, @intCast(101 + e)),
                try makeTestQuantLinear(self.ctx, self.allocator, 64, 32, @intCast(211 + e)),
            };
        }
        const v = gop.value_ptr;
        return .{ .gate = v[0], .up = v[1], .down = v[2] };
    }

    fn releaseAllFn(opaque_ctx: *anyopaque) void {
        const self: *TestExpertProvider = @ptrCast(@alignCast(opaque_ctx));
        var it = self.cache.valueIterator();
        while (it.next()) |v| {
            for (v) |ql| {
                ql.w.deinit();
                ql.scales.deinit();
                ql.biases.deinit();
            }
        }
        self.cache.clearRetainingCapacity();
    }
};

test "Glm53Moe: 路由选择 + 组合形状与权重归一" {
    test_c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const n_experts: i32 = 8;
    const top_k: i32 = 3;
    const hidden: i32 = 64;
    const l: i32 = 4;

    const gate_w_t = try makeTestArray(allocator, &[_]i32{ hidden, n_experts }, 7);
    defer gate_w_t.deinit();
    const gate_bias = try makeTestArray(allocator, &[_]i32{n_experts}, 8);
    defer gate_bias.deinit();

    var shared_gate_up = try makeTestQuantLinear(ctx, allocator, 64, hidden, 9);
    defer shared_gate_up.w.deinit();
    defer shared_gate_up.scales.deinit();
    defer shared_gate_up.biases.deinit();
    var shared_down = try makeTestQuantLinear(ctx, allocator, hidden, 32, 10);
    defer shared_down.w.deinit();
    defer shared_down.scales.deinit();
    defer shared_down.biases.deinit();

    const moe = Glm53Moe{
        .ctx = ctx,
        .n_experts = n_experts,
        .top_k = top_k,
        .scaling = 2.5,
        .norm_topk_prob = true,
        .swiglu_limit = 10.0,
        .gate_w_t = gate_w_t,
        .gate_bias = gate_bias,
        .shared = .{
            .ctx = ctx,
            .gate_up_proj = shared_gate_up,
            .down_proj = shared_down,
            .swiglu_limit = 10.0,
        },
    };

    var provider = TestExpertProvider{
        .ctx = ctx,
        .allocator = allocator,
        .cache = std.AutoHashMap(u32, [3]glm53_attn.QuantLinear).init(allocator),
    };
    defer provider.cache.deinit();
    defer TestExpertProvider.releaseAllFn(&provider);

    const x = try makeTestArray(allocator, &[_]i32{ 1, l, hidden }, 11);
    defer x.deinit();

    const out = try moe.forward(allocator, x, .{
        .ctx = &provider,
        .loadFn = TestExpertProvider.loadFn,
        .releaseAllFn = TestExpertProvider.releaseAllFn,
    });
    defer out.deinit();

    const osh = out.shape();
    try std.testing.expectEqual(@as(usize, 3), osh.len);
    try std.testing.expectEqual(@as(i32, 1), osh[0]);
    try std.testing.expectEqual(l, osh[1]);
    try std.testing.expectEqual(hidden, osh[2]);

    // Re-run routing only and verify: per row, weights sum to the scaling
    // factor (norm_topk_prob), indices are in-range, and the selected set
    // equals the top-k by choice score (recomputed on CPU).
    const x_flat = try ops.reshape(ctx, x, &[_]i32{ l, hidden });
    defer x_flat.deinit();
    var rte = try moe.route(x_flat);
    defer rte.indices.deinit();
    defer rte.weights.deinit();
    const idx = try rte.indices.dataSlice(i32);
    const w = try rte.weights.dataSlice(f32);

    const gw = try gate_w_t.dataSlice(f32);
    const gb = try gate_bias.dataSlice(f32);
    const xs = try x_flat.dataSlice(f32);
    for (0..@intCast(l)) |t| {
        // expected top-k by sigmoid(x.w)+bias
        var choice: [8]f32 = undefined;
        for (0..8) |e| {
            var dot: f64 = 0;
            for (0..@intCast(hidden)) |d| {
                dot += @as(f64, xs[t * @as(usize, @intCast(hidden)) + d]) * @as(f64, gw[d * 8 + e]);
            }
            const sig = 1.0 / (1.0 + @exp(-dot));
            choice[e] = @as(f32, @floatCast(sig)) + gb[e];
        }
        var expect_set = [_]bool{false} ** 8;
        for (0..@intCast(top_k)) |_| {
            var best: ?usize = null;
            for (0..8) |e| {
                if (expect_set[e]) continue;
                if (best == null or choice[e] > choice[best.?]) best = e;
            }
            expect_set[best.?] = true;
        }
        var wsum: f32 = 0;
        for (0..@intCast(top_k)) |k_i| {
            const e = idx[t * @as(usize, @intCast(top_k)) + k_i];
            try std.testing.expect(e >= 0 and e < n_experts);
            try std.testing.expect(expect_set[@intCast(e)]);
            wsum += w[t * @as(usize, @intCast(top_k)) + k_i];
        }
        try std.testing.expectApproxEqAbs(@as(f32, 2.5), wsum, 1e-4);
    }
}
