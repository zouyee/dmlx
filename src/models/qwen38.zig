//! Qwen3.8 (qwen3_5) model assembly: embed → 64 decoder layers
//! (GDN linear attention / partial-RoPE full attention × SwiGLU MLP,
//! plain residual connections) → final norm → lm_head logits.
//!
//! WeightSource / QuantParts are the generic GLM-5.3 helpers re-exported.
//! Branch helpers construct the layer modules (Qwen38GDN / Qwen38Attn) with
//! weights read fresh via the WeightSource per call, exactly like glm53.zig's
//! kdaAttnBranch / denseFfnBranch.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const fast_mod = @import("mlx").fast;
const activations = @import("mlx").activations;
const quantize_mod = @import("mlx").quantize;
const creation_mod = @import("mlx").creation;
const eval_mod = @import("mlx").eval;
const glm53 = @import("glm53.zig");
const glm53_attn = @import("glm53_attention.zig");
const qwen38_loader = @import("qwen38_loader.zig");
const qwen38_gdn = @import("qwen38_gdn.zig");
const qwen38_attn = @import("qwen38_attn.zig");
const closure_mod = @import("mlx").closure;
const compile_mod = @import("mlx").compile;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const Closure = closure_mod.Closure;

pub const WeightSource = glm53.WeightSource;
pub const QuantParts = glm53.QuantParts;

/// Weight names are fixed patterns well under 256 bytes; stack buffers only.
fn fmtName(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch unreachable;
}

// ============================================================
// Embedding / final logits
// ============================================================

/// embedTokens on a WeightSource: quantized embed → bf16 rows → [1, L, hidden].
pub fn embedTokensWs(ctx: EagerContext, ws: WeightSource, allocator: std.mem.Allocator, cfg: *const qwen38_loader.Qwen38Config, ids: []const i32) !Array {
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

/// hidden [1, L, hidden] → rmsNorm(language_model.model.norm) → lm_head logits.
pub fn finalLogitsWs(ctx: EagerContext, ws: WeightSource, cfg: *const qwen38_loader.Qwen38Config, h: Array) !Array {
    const hidden: i32 = @intCast(cfg.hidden_size);
    const h_f32 = try ops.astype(ctx, h, .float32);
    defer h_f32.deinit();
    const norm_w = try ws.tensor("language_model.model.norm.weight");
    defer norm_w.deinit();
    const norm_w_f32 = try ops.astype(ctx, norm_w, .float32);
    defer norm_w_f32.deinit();
    const normed = try fast_mod.rmsNorm(ctx, h_f32, norm_w_f32, cfg.rms_norm_eps);
    defer normed.deinit();

    var head_q = try ws.quantized("language_model.lm_head", hidden);
    defer head_q.deinit();
    return head_q.linear().apply(ctx, normed);
}

// ============================================================
// Vision token rejection (text-only serving path)
// ============================================================

/// Vision token id range from the checkpoint's config/tokenizer:
/// 248053 <|vision_start|>, 248054 <|vision_end|>, 248055 <|vision_pad|>,
/// 248056 <|image_pad|>, 248057 <|video_pad|>.
pub const vision_token_lo: u32 = 248053;
pub const vision_token_hi: u32 = 248057;

pub const VisionError = error{VisionInputNotSupported};

/// Reject prompts containing vision tokens: the text-only path has no vision
/// tower, so feeding these through the text embedding would produce garbage.
pub fn rejectVisionTokens(ids: []const u32) VisionError!void {
    for (ids) |t| {
        if (t >= vision_token_lo and t <= vision_token_hi) {
            // warn under `zig build test` (the runner fails tests on err logs).
            if (@import("builtin").is_test)
                std.log.warn("qwen38: vision input not supported (token id {d})", .{t})
            else
                std.log.err("qwen38: vision input not supported (token id {d} in vision range [{d}, {d}])", .{ t, vision_token_lo, vision_token_hi });
            return VisionError.VisionInputNotSupported;
        }
    }
}

// ============================================================
// Per-layer decode state
// ============================================================

pub const Qwen38LayerCache = union(enum) {
    gdn: qwen38_gdn.Qwen38GdnCache,
    attn: qwen38_attn.Qwen38AttnCache,

    pub fn deinit(self: *Qwen38LayerCache) void {
        switch (self.*) {
            .gdn => |*gc| gc.deinit(),
            .attn => |*ac| ac.deinit(),
        }
    }
};

// ============================================================
// Branch helpers
// ============================================================

fn loadNormWeight(ctx: EagerContext, ws: WeightSource, name: []const u8) !Array {
    const w = try ws.tensor(name);
    defer w.deinit();
    return ops.astype(ctx, w, .float32);
}

/// GDN linear-attention branch: loads layer weights via ws, constructs
/// Qwen38GDN, runs forward/prefill against the layer cache.
/// x: [1, L, hidden] (L>1 prefill) or [1, 1, hidden]; returns same shape.
pub fn gdnBranch(
    ctx: EagerContext,
    ws: WeightSource,
    allocator: std.mem.Allocator,
    cfg: *const qwen38_loader.Qwen38Config,
    layer_idx: u32,
    x: Array,
    gdn_cache: ?*qwen38_gdn.Qwen38GdnCache,
) !Array {
    const hidden: i32 = @intCast(cfg.hidden_size);
    var pbuf: [96]u8 = undefined;
    const la = fmtName(&pbuf, "language_model.model.layers.{d}.linear_attn", .{layer_idx});
    var nbuf: [256]u8 = undefined;

    var qkv_q = try ws.quantized(fmtName(&nbuf, "{s}.in_proj_qkv", .{la}), hidden);
    defer qkv_q.deinit();
    var a_q = try ws.quantized(fmtName(&nbuf, "{s}.in_proj_a", .{la}), hidden);
    defer a_q.deinit();
    var b_q = try ws.quantized(fmtName(&nbuf, "{s}.in_proj_b", .{la}), hidden);
    defer b_q.deinit();
    var z_q = try ws.quantized(fmtName(&nbuf, "{s}.in_proj_z", .{la}), hidden);
    defer z_q.deinit();
    var o_q = try ws.quantized(fmtName(&nbuf, "{s}.out_proj", .{la}), @intCast(cfg.linear_num_value_heads * cfg.linear_value_head_dim));
    defer o_q.deinit();

    // Plain (bf16) GDN tensors are loaded without upcasting: the oracle
    // (mlx-lm) runs the conv, softplus, beta sigmoid and gated-norm in the
    // activation dtype and only casts A_log to f32 inside compute_g.
    const conv1d = try ws.tensor(fmtName(&nbuf, "{s}.conv1d.weight", .{la}));
    defer conv1d.deinit();
    var pre = try qwen38_gdn.precomputeDerived(
        allocator,
        ctx,
        conv1d,
        cfg.linear_num_key_heads,
        cfg.linear_num_value_heads,
        cfg.linear_key_head_dim,
    );
    defer pre.deinit();
    const a_log = try ws.tensor(fmtName(&nbuf, "{s}.A_log", .{la}));
    defer a_log.deinit();
    const dt_bias = try ws.tensor(fmtName(&nbuf, "{s}.dt_bias", .{la}));
    defer dt_bias.deinit();
    const norm_w = try ws.tensor(fmtName(&nbuf, "{s}.norm.weight", .{la}));
    defer norm_w.deinit();

    var gdn = qwen38_gdn.Qwen38GDN{
        .ctx = ctx,
        .hidden_size = cfg.hidden_size,
        .num_key_heads = cfg.linear_num_key_heads,
        .num_value_heads = cfg.linear_num_value_heads,
        .head_dim = cfg.linear_key_head_dim,
        .conv_kernel = cfg.linear_conv_kernel_dim,
        .norm_eps = cfg.rms_norm_eps,
        .in_proj_qkv = qkv_q.data,
        .conv1d = conv1d,
        .conv1d_t = pre.conv1d_t,
        .q_norm_w = pre.q_norm_w,
        .k_norm_w = pre.k_norm_w,
        .qk_idx = pre.qk_idx,
        .in_proj_a = a_q.data,
        .in_proj_b = b_q.data,
        .in_proj_z = z_q.data,
        .A_log = a_log,
        .dt_bias = dt_bias,
        .norm_w = norm_w,
        .out_proj = o_q.data,
        .qkv_scales = qkv_q.scales,
        .qkv_biases = qkv_q.biases,
        .qkv_quant = qkv_q.cfg,
        .a_scales = a_q.scales,
        .a_biases = a_q.biases,
        .a_quant = a_q.cfg,
        .b_scales = b_q.scales,
        .b_biases = b_q.biases,
        .b_quant = b_q.cfg,
        .z_scales = z_q.scales,
        .z_biases = z_q.biases,
        .z_quant = z_q.cfg,
        .o_scales = o_q.scales,
        .o_biases = o_q.biases,
        .o_quant = o_q.cfg,
    };
    // Weight Arrays are owned by the defers above — do NOT call gdn.deinit().

    var tmp_cache: ?qwen38_gdn.Qwen38GdnCache = null;
    defer if (tmp_cache) |*tc| tc.deinit();
    const cache = gdn_cache orelse blk: {
        tmp_cache = try qwen38_gdn.Qwen38GdnCache.init(
            allocator,
            cfg.linear_conv_kernel_dim,
            (2 * cfg.linear_num_key_heads + cfg.linear_num_value_heads) * cfg.linear_key_head_dim,
            cfg.linear_num_value_heads,
            cfg.linear_value_head_dim,
        );
        break :blk &tmp_cache.?;
    };

    const n = x.shape()[1];
    if (n > 1) {
        return gdn.prefill(allocator, x, cache, true);
    }
    const step_in = try ops.reshape(ctx, x, &[_]i32{ 1, hidden });
    defer step_in.deinit();
    const step_out = try gdn.forward(step_in, cache);
    defer step_out.deinit();
    return ops.reshape(ctx, step_out, &[_]i32{ 1, 1, hidden });
}

/// Full-attention branch (partial RoPE + q/k norm + sigmoid output gate).
pub fn attnBranch(
    ctx: EagerContext,
    ws: WeightSource,
    allocator: std.mem.Allocator,
    cfg: *const qwen38_loader.Qwen38Config,
    layer_idx: u32,
    x: Array,
    attn_cache: ?*qwen38_attn.Qwen38AttnCache,
) !Array {
    const hidden: i32 = @intCast(cfg.hidden_size);
    var pbuf: [96]u8 = undefined;
    const sa = fmtName(&pbuf, "language_model.model.layers.{d}.self_attn", .{layer_idx});
    var nbuf: [256]u8 = undefined;

    var q_q = try ws.quantized(fmtName(&nbuf, "{s}.q_proj", .{sa}), hidden);
    defer q_q.deinit();
    var k_q = try ws.quantized(fmtName(&nbuf, "{s}.k_proj", .{sa}), hidden);
    defer k_q.deinit();
    var v_q = try ws.quantized(fmtName(&nbuf, "{s}.v_proj", .{sa}), hidden);
    defer v_q.deinit();
    var o_q = try ws.quantized(fmtName(&nbuf, "{s}.o_proj", .{sa}), @intCast(cfg.num_attention_heads * cfg.head_dim));
    defer o_q.deinit();

    // q_norm/k_norm stay bf16 to mirror nn.RMSNorm (f32 internal, bf16 out).
    const q_norm = try ws.tensor(fmtName(&nbuf, "{s}.q_norm.weight", .{sa}));
    defer q_norm.deinit();
    const k_norm = try ws.tensor(fmtName(&nbuf, "{s}.k_norm.weight", .{sa}));
    defer k_norm.deinit();

    var attn = qwen38_attn.Qwen38Attn{
        .ctx = ctx,
        .hidden_size = cfg.hidden_size,
        .num_heads = cfg.num_attention_heads,
        .num_kv_heads = cfg.num_key_value_heads,
        .head_dim = cfg.head_dim,
        .rotary_dims = cfg.rotaryDims(),
        .rope_theta = cfg.rope_theta,
        .rms_norm_eps = cfg.rms_norm_eps,
        .output_gate = cfg.attn_output_gate,
        .q_proj = q_q.data,
        .k_proj = k_q.data,
        .v_proj = v_q.data,
        .o_proj = o_q.data,
        .q_norm = q_norm,
        .k_norm = k_norm,
        .q_scales = q_q.scales,
        .q_biases = q_q.biases,
        .q_quant = q_q.cfg,
        .k_scales = k_q.scales,
        .k_biases = k_q.biases,
        .k_quant = k_q.cfg,
        .v_scales = v_q.scales,
        .v_biases = v_q.biases,
        .v_quant = v_q.cfg,
        .o_scales = o_q.scales,
        .o_biases = o_q.biases,
        .o_quant = o_q.cfg,
    };
    // Weight Arrays are owned by the defers above — do NOT call attn.deinit().

    var tmp_cache: ?qwen38_attn.Qwen38AttnCache = null;
    defer if (tmp_cache) |*tc| tc.deinit();
    const cache = attn_cache orelse blk: {
        tmp_cache = .{};
        break :blk &tmp_cache.?;
    };
    _ = allocator;

    return attn.forward(x, cache);
}

/// SwiGLU MLP branch: down(silu(gate(x)) * up(x)), all quantized.
pub fn mlpBranch(
    ctx: EagerContext,
    ws: WeightSource,
    cfg: *const qwen38_loader.Qwen38Config,
    layer_idx: u32,
    x: Array,
) !Array {
    const hidden: i32 = @intCast(cfg.hidden_size);
    var pbuf: [96]u8 = undefined;
    const mp = fmtName(&pbuf, "language_model.model.layers.{d}.mlp", .{layer_idx});
    var nbuf: [256]u8 = undefined;

    var gate_q = try ws.quantized(fmtName(&nbuf, "{s}.gate_proj", .{mp}), hidden);
    defer gate_q.deinit();
    var up_q = try ws.quantized(fmtName(&nbuf, "{s}.up_proj", .{mp}), hidden);
    defer up_q.deinit();
    var down_q = try ws.quantized(fmtName(&nbuf, "{s}.down_proj", .{mp}), @intCast(cfg.intermediate_size));
    defer down_q.deinit();

    const gate = try gate_q.linear().apply(ctx, x);
    defer gate.deinit();
    const up = try up_q.linear().apply(ctx, x);
    defer up.deinit();
    const act = try activations.silu(ctx, gate);
    defer act.deinit();
    const h = try ops.multiply(ctx, act, up);
    defer h.deinit();
    return down_q.linear().apply(ctx, h);
}

// ============================================================
// Model assembly (oracle harness)
// ============================================================

/// Segment hook: called with the post-attn-branch residual at every layer
/// ("layer{i}_gdn_out"/"layer{i}_attn_out"/"layer{i}_mlp_out") plus "embed"
/// and "hidden_final". The tensor is borrowed (hooks should astype/copy if
/// they keep it).
pub const SegmentHook = struct {
    ctx: *anyopaque,
    func: *const fn (ctx: *anyopaque, name: []const u8, tensor: Array) anyerror!void,
};

pub const Qwen38Model = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    cfg: *const qwen38_loader.Qwen38Config,
    ws: WeightSource,

    pub const ForwardOpts = struct {
        /// Early stop: run only layers 0..=max_layer (cheap segment reruns).
        max_layer: ?u32 = null,
        hook: ?SegmentHook = null,
    };

    /// Full eager forward. Returns logits [1, L, vocab] f32 (empty [0] array
    /// on early-stopped runs). Plain residual: h = h + attn(rmsnorm(h)) then
    /// h = h + mlp(rmsnorm(h)).
    ///
    /// The residual trajectory mirrors the mlx-lm oracle dtype-for-dtype: h
    /// stays bf16, rmsNorm is computed in f32 and rounded back to bf16 for the
    /// branch input, and branch outputs (bf16 quantized matmuls) accumulate
    /// into h in bf16. This keeps the 64-layer error compounding within the
    /// oracle's own rounding (f32 residuals drift past the 0.9999 gate by
    /// layer ~40).
    /// Keep the dtype/op sequence in sync with `Qwen38ServingModel.forwardTokens`.
    pub fn forward(self: *const Qwen38Model, ids: []const i32, opts: ForwardOpts) !Array {
        const allocator = self.allocator;
        const cfg = self.cfg;
        const ctx = self.ctx;
        var nbuf: [256]u8 = undefined;

        const emb = try embedTokensWs(ctx, self.ws, allocator, cfg, ids);
        defer emb.deinit();
        if (opts.hook) |hk| try hk.func(hk.ctx, "embed", emb);
        var h = try ops.copy(ctx, emb); // bf16
        defer h.deinit();

        var layer: u32 = 0;
        while (layer < cfg.num_hidden_layers) : (layer += 1) {
            if (opts.max_layer) |ml| {
                if (layer > ml) break;
            }
            var pbuf: [96]u8 = undefined;
            const p = fmtName(&pbuf, "language_model.model.layers.{d}", .{layer});

            // Attention sub-block: h = h + attn(rmsnorm(h, input_layernorm)).
            const ln1 = try loadNormWeight(ctx, self.ws, fmtName(&nbuf, "{s}.input_layernorm.weight", .{p}));
            defer ln1.deinit();
            const normed1 = blk: {
                const h_f32 = try ops.astype(ctx, h, .float32);
                defer h_f32.deinit();
                const n_f32 = try fast_mod.rmsNorm(ctx, h_f32, ln1, cfg.rms_norm_eps);
                defer n_f32.deinit();
                break :blk try ops.astype(ctx, n_f32, .bfloat16);
            };
            defer normed1.deinit();
            const attn_out = if (cfg.isFullAttention(layer)) blk: {
                break :blk try attnBranch(ctx, self.ws, allocator, cfg, layer, normed1, null);
            } else blk: {
                break :blk try gdnBranch(ctx, self.ws, allocator, cfg, layer, normed1, null);
            };
            defer attn_out.deinit();
            {
                const next = try ops.add(ctx, h, attn_out);
                h.deinit();
                h = next;
                try h.eval();
            }
            if (opts.hook) |hk| {
                const name = if (cfg.isFullAttention(layer))
                    fmtName(&nbuf, "layer{d}_attn_out", .{layer})
                else
                    fmtName(&nbuf, "layer{d}_gdn_out", .{layer});
                try hk.func(hk.ctx, name, h);
            }

            // MLP sub-block: h = h + mlp(rmsnorm(h, post_attention_layernorm)).
            const ln2 = try loadNormWeight(ctx, self.ws, fmtName(&nbuf, "{s}.post_attention_layernorm.weight", .{p}));
            defer ln2.deinit();
            const normed2 = blk: {
                const h_f32 = try ops.astype(ctx, h, .float32);
                defer h_f32.deinit();
                const n_f32 = try fast_mod.rmsNorm(ctx, h_f32, ln2, cfg.rms_norm_eps);
                defer n_f32.deinit();
                break :blk try ops.astype(ctx, n_f32, .bfloat16);
            };
            defer normed2.deinit();
            const mlp_out = try mlpBranch(ctx, self.ws, cfg, layer, normed2);
            defer mlp_out.deinit();
            {
                const next = try ops.add(ctx, h, mlp_out);
                h.deinit();
                h = next;
                try h.eval();
            }
            if (opts.hook) |hk| {
                const name = fmtName(&nbuf, "layer{d}_mlp_out", .{layer});
                try hk.func(hk.ctx, name, h);
            }
        }

        // hidden_final / logits are only meaningful on a full run.
        const full_run = opts.max_layer == null or opts.max_layer.? >= cfg.num_hidden_layers - 1;
        if (!full_run) {
            return creation_mod.zeros(ctx, &[_]i32{0}, .float32);
        }

        const h_f32 = try ops.astype(ctx, h, .float32);
        defer h_f32.deinit();
        const norm_w = try loadNormWeight(ctx, self.ws, "language_model.model.norm.weight");
        defer norm_w.deinit();
        const hidden_final_f32 = try fast_mod.rmsNorm(ctx, h_f32, norm_w, cfg.rms_norm_eps);
        defer hidden_final_f32.deinit();
        const hidden_final = try ops.astype(ctx, hidden_final_f32, .bfloat16);
        defer hidden_final.deinit();
        if (opts.hook) |hk| try hk.func(hk.ctx, "hidden_final", hidden_final);

        const hidden: i32 = @intCast(cfg.hidden_size);
        var head_q = try self.ws.quantized("language_model.lm_head", hidden);
        defer head_q.deinit();
        return head_q.linear().apply(ctx, hidden_final);
    }
};

// ============================================================
// Serving model: resident weights + per-layer decode caches
// ============================================================

/// Borrow a resident-map Array handle WITHOUT a COW copy: the returned Array
/// aliases the map entry (mlx arrays are reference-counted handles; sharing
/// the handle is safe because the map outlives every borrower here). The
/// caller must NOT deinit the returned Array — the map owns it.
fn borrowTensor(map: *const std.StringHashMap(Array), name: []const u8) !Array {
    return map.get(name) orelse {
        std.log.err("resident weights: tensor missing: {s}", .{name});
        return error.TensorNotFound;
    };
}

/// Borrowed affine-quantized Linear triple (weight/scales/biases aliased from
/// the resident map), with bits/group_size derived from the packed shapes —
/// same derivation as WeightSource.quantized, but no COW copies and no
/// ownership: do NOT deinit the returned Arrays.
fn borrowQuantLinear(map: *const std.StringHashMap(Array), prefix: []const u8, in_features: i32) !glm53_attn.QuantLinear {
    var nbuf: [256]u8 = undefined;
    const data = try borrowTensor(map, fmtName(&nbuf, "{s}.weight", .{prefix}));
    const scales = try borrowTensor(map, fmtName(&nbuf, "{s}.scales", .{prefix}));
    const biases = try borrowTensor(map, fmtName(&nbuf, "{s}.biases", .{prefix}));
    const wshape = data.shape();
    const sshape = scales.shape();
    const packed_cols: i64 = wshape[wshape.len - 1];
    const scale_cols: i64 = sshape[sshape.len - 1];
    return .{
        .w = data,
        .scales = scales,
        .biases = biases,
        .cfg = .{
            .bits = @intCast(@divTrunc(packed_cols * 32, in_features)),
            .group_size = @intCast(@divTrunc(in_features, scale_cols)),
            .mode = .affine,
        },
    };
}

/// A ready-to-use per-layer attention module with weight Arrays borrowed
/// from the resident map (built once at Qwen38ServingModel.init).
///
/// OWNERSHIP: modules do NOT own their weight Arrays — those alias resident
/// map entries. Each module owns only its derived artifacts (GDN: conv1d_t /
/// q_norm_w / k_norm_w / qk_idx; attn: none). deinitShallow frees only the owned derived
/// arrays; the map (freed by the registry adapter after model.deinit) owns
/// the weights. Never call the module's own deinit() on these instances —
/// it would free the shared map handles.
pub const LayerModule = union(enum) {
    gdn: qwen38_gdn.Qwen38GDN,
    attn: qwen38_attn.Qwen38Attn,

    pub fn deinitShallow(self: *LayerModule) void {
        switch (self.*) {
            .gdn => |*g| {
                g.conv1d_t.deinit();
                g.q_norm_w.deinit();
                g.k_norm_w.deinit();
                g.qk_idx.deinit();
            },
            .attn => {},
        }
    }
};

/// Per-layer persistent state: the attention module plus the two layernorm
/// weights (cast to f32 and evaluated once) and the borrowed MLP projections.
pub const LayerPersistent = struct {
    ln1_f32: Array, // owned (f32 cast of input_layernorm.weight, evaled)
    ln2_f32: Array, // owned (f32 cast of post_attention_layernorm.weight, evaled)
    module: LayerModule,
    mlp_gate: glm53_attn.QuantLinear, // borrowed
    mlp_up: glm53_attn.QuantLinear, // borrowed
    mlp_down: glm53_attn.QuantLinear, // borrowed
    /// Fused gate|up projection (one qmm instead of two): row-concat of the
    /// packed weights/scales/biases along dim 0 (gs-row packing makes this
    /// exactly equivalent). Null when the gate/up shapes or quant configs
    /// don't match — the unfused gate/up path is the fallback. Arrays are
    /// OWNED via gate_up_bufs (the concat results, evaluated once at init).
    mlp_gate_up: ?glm53_attn.QuantLinear = null,
    gate_up_bufs: ?[3]Array = null, // owned: w, scales, biases

    pub fn deinitShallow(self: *LayerPersistent) void {
        self.ln1_f32.deinit();
        self.ln2_f32.deinit();
        self.module.deinitShallow();
        if (self.gate_up_bufs) |*bufs| {
            for (bufs) |*b| b.deinit();
        }
    }
};

/// Concatenate gate and up packed quantized weights along the out dim into
/// one fused projection. Packed layout is row-wise ([out, in*bits/32] U32,
/// scales/biases [out, in/gs]), so row-concat is exactly equivalent to
/// running both projections and concatenating the outputs. Returns null when
/// the shapes/dtypes/quant configs don't match (caller keeps the unfused
/// path). All three returned Arrays are owned by the caller.
fn fuseGateUp(ctx: EagerContext, gate: glm53_attn.QuantLinear, up: glm53_attn.QuantLinear) !?glm53_attn.QuantLinear {
    if (gate.w.dtype() != up.w.dtype()) return null;
    if (gate.scales.dtype() != up.scales.dtype() or gate.biases.dtype() != up.biases.dtype()) return null;
    if (!std.mem.eql(i32, gate.w.shape(), up.w.shape())) return null;
    if (!std.mem.eql(i32, gate.scales.shape(), up.scales.shape())) return null;
    if (!std.mem.eql(i32, gate.biases.shape(), up.biases.shape())) return null;
    if (gate.cfg.bits != up.cfg.bits or gate.cfg.group_size != up.cfg.group_size or gate.cfg.mode != up.cfg.mode) return null;

    const w = try shape_mod.concatenateAxis(ctx, &[_]Array{ gate.w, up.w }, 0);
    errdefer w.deinit();
    const s = try shape_mod.concatenateAxis(ctx, &[_]Array{ gate.scales, up.scales }, 0);
    errdefer s.deinit();
    const b = try shape_mod.concatenateAxis(ctx, &[_]Array{ gate.biases, up.biases }, 0);
    errdefer b.deinit();
    // Materialize once at init so per-token graphs reference resident arrays.
    try w.eval();
    try s.eval();
    try b.eval();
    return .{ .w = w, .scales = s, .biases = b, .cfg = gate.cfg };
}

/// Construct one layer's persistent module set from the resident map
/// (mirrors gdnBranch/attnBranch/mlpBranch weight lookup exactly, minus the
/// COW copies; GDN derived constants are precomputed and evaluated here).
fn buildLayerPersistent(
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    cfg: *const qwen38_loader.Qwen38Config,
    map: *const std.StringHashMap(Array),
    layer_idx: u32,
) !LayerPersistent {
    const hidden: i32 = @intCast(cfg.hidden_size);
    var pbuf: [96]u8 = undefined;
    const p = fmtName(&pbuf, "language_model.model.layers.{d}", .{layer_idx});
    var nbuf: [256]u8 = undefined;

    const ln1 = try borrowTensor(map, fmtName(&nbuf, "{s}.input_layernorm.weight", .{p}));
    const ln1_f32 = try ops.astype(ctx, ln1, .float32);
    try ln1_f32.eval();
    const ln2 = try borrowTensor(map, fmtName(&nbuf, "{s}.post_attention_layernorm.weight", .{p}));
    const ln2_f32 = try ops.astype(ctx, ln2, .float32);
    try ln2_f32.eval();

    var out: LayerPersistent = .{
        .ln1_f32 = ln1_f32,
        .ln2_f32 = ln2_f32,
        .module = undefined,
        .mlp_gate = undefined,
        .mlp_up = undefined,
        .mlp_down = undefined,
    };
    errdefer out.ln1_f32.deinit();
    errdefer out.ln2_f32.deinit();

    if (cfg.isFullAttention(layer_idx)) {
        const sa = fmtName(&nbuf, "{s}.self_attn", .{p});
        var qbuf: [256]u8 = undefined;
        const q_q = try borrowQuantLinear(map, fmtName(&qbuf, "{s}.q_proj", .{sa}), hidden);
        const k_q = try borrowQuantLinear(map, fmtName(&qbuf, "{s}.k_proj", .{sa}), hidden);
        const v_q = try borrowQuantLinear(map, fmtName(&qbuf, "{s}.v_proj", .{sa}), hidden);
        const o_q = try borrowQuantLinear(map, fmtName(&qbuf, "{s}.o_proj", .{sa}), @intCast(cfg.num_attention_heads * cfg.head_dim));
        out.module = .{ .attn = .{
            .ctx = ctx,
            .hidden_size = cfg.hidden_size,
            .num_heads = cfg.num_attention_heads,
            .num_kv_heads = cfg.num_key_value_heads,
            .head_dim = cfg.head_dim,
            .rotary_dims = cfg.rotaryDims(),
            .rope_theta = cfg.rope_theta,
            .rms_norm_eps = cfg.rms_norm_eps,
            .output_gate = cfg.attn_output_gate,
            .q_proj = q_q.w,
            .k_proj = k_q.w,
            .v_proj = v_q.w,
            .o_proj = o_q.w,
            .q_norm = try borrowTensor(map, fmtName(&qbuf, "{s}.q_norm.weight", .{sa})),
            .k_norm = try borrowTensor(map, fmtName(&qbuf, "{s}.k_norm.weight", .{sa})),
            .q_scales = q_q.scales,
            .q_biases = q_q.biases,
            .q_quant = q_q.cfg,
            .k_scales = k_q.scales,
            .k_biases = k_q.biases,
            .k_quant = k_q.cfg,
            .v_scales = v_q.scales,
            .v_biases = v_q.biases,
            .v_quant = v_q.cfg,
            .o_scales = o_q.scales,
            .o_biases = o_q.biases,
            .o_quant = o_q.cfg,
        } };
    } else {
        const la = fmtName(&nbuf, "{s}.linear_attn", .{p});
        var qbuf: [256]u8 = undefined;
        const qkv_q = try borrowQuantLinear(map, fmtName(&qbuf, "{s}.in_proj_qkv", .{la}), hidden);
        const a_q = try borrowQuantLinear(map, fmtName(&qbuf, "{s}.in_proj_a", .{la}), hidden);
        const b_q = try borrowQuantLinear(map, fmtName(&qbuf, "{s}.in_proj_b", .{la}), hidden);
        const z_q = try borrowQuantLinear(map, fmtName(&qbuf, "{s}.in_proj_z", .{la}), hidden);
        const o_q = try borrowQuantLinear(map, fmtName(&qbuf, "{s}.out_proj", .{la}), @intCast(cfg.linear_num_value_heads * cfg.linear_value_head_dim));

        const conv1d = try borrowTensor(map, fmtName(&qbuf, "{s}.conv1d.weight", .{la}));
        var pre = try qwen38_gdn.precomputeDerived(
            allocator,
            ctx,
            conv1d,
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
        );
        errdefer pre.deinit();
        // Materialize the derived constants once so per-token graphs
        // reference resident arrays instead of view/transpose chains.
        try pre.conv1d_t.eval();
        try pre.q_norm_w.eval();
        try pre.k_norm_w.eval();

        out.module = .{ .gdn = .{
            .ctx = ctx,
            .hidden_size = cfg.hidden_size,
            .num_key_heads = cfg.linear_num_key_heads,
            .num_value_heads = cfg.linear_num_value_heads,
            .head_dim = cfg.linear_key_head_dim,
            .conv_kernel = cfg.linear_conv_kernel_dim,
            .norm_eps = cfg.rms_norm_eps,
            .in_proj_qkv = qkv_q.w,
            .conv1d = conv1d,
            .conv1d_t = pre.conv1d_t,
            .q_norm_w = pre.q_norm_w,
            .k_norm_w = pre.k_norm_w,
            .qk_idx = pre.qk_idx,
            .in_proj_a = a_q.w,
            .in_proj_b = b_q.w,
            .in_proj_z = z_q.w,
            .A_log = try borrowTensor(map, fmtName(&qbuf, "{s}.A_log", .{la})),
            .dt_bias = try borrowTensor(map, fmtName(&qbuf, "{s}.dt_bias", .{la})),
            .norm_w = try borrowTensor(map, fmtName(&qbuf, "{s}.norm.weight", .{la})),
            .out_proj = o_q.w,
            .qkv_scales = qkv_q.scales,
            .qkv_biases = qkv_q.biases,
            .qkv_quant = qkv_q.cfg,
            .a_scales = a_q.scales,
            .a_biases = a_q.biases,
            .a_quant = a_q.cfg,
            .b_scales = b_q.scales,
            .b_biases = b_q.biases,
            .b_quant = b_q.cfg,
            .z_scales = z_q.scales,
            .z_biases = z_q.biases,
            .z_quant = z_q.cfg,
            .o_scales = o_q.scales,
            .o_biases = o_q.biases,
            .o_quant = o_q.cfg,
        } };
    }
    errdefer out.module.deinitShallow();

    const mp = fmtName(&nbuf, "{s}.mlp", .{p});
    var qbuf2: [256]u8 = undefined;
    out.mlp_gate = try borrowQuantLinear(map, fmtName(&qbuf2, "{s}.gate_proj", .{mp}), hidden);
    out.mlp_up = try borrowQuantLinear(map, fmtName(&qbuf2, "{s}.up_proj", .{mp}), hidden);
    out.mlp_down = try borrowQuantLinear(map, fmtName(&qbuf2, "{s}.down_proj", .{mp}), @intCast(cfg.intermediate_size));
    // Fused gate|up: one wide qmm per token instead of two (halves the launch
    // count of the biggest qmms and streams one contiguous 2× weight read).
    // Best-effort — the unfused gate/up above stays as the fallback.
    if (fuseGateUp(ctx, out.mlp_gate, out.mlp_up) catch null) |fused| {
        out.gate_up_bufs = .{ fused.w, fused.scales, fused.biases };
        out.mlp_gate_up = fused;
    }
    return out;
}

/// SwiGLU MLP on persistent borrowed projections: down(silu(gate(x)) * up(x)).
fn mlpForwardPersistent(ctx: EagerContext, lp: *const LayerPersistent, x: Array) !Array {
    const gate = try lp.mlp_gate.apply(ctx, x);
    defer gate.deinit();
    const up = try lp.mlp_up.apply(ctx, x);
    defer up.deinit();
    const act = try activations.silu(ctx, gate);
    defer act.deinit();
    const h = try ops.multiply(ctx, act, up);
    defer h.deinit();
    return lp.mlp_down.apply(ctx, h);
}

// ============================================================
// Compiled decode steps (mx.compile) — dispatch-bound fix
// ============================================================

/// The compiled closures bake rms_norm eps as a trace-time constant; the
/// serving model only builds them when the config matches.
const COMPILED_RMS_EPS: f32 = 1e-6;

/// Compiled SwiGLU core: down(silu(gate(x)) * up(x)) on an already-normed
/// bf16 x (layernorm + residual stay eager). Input order (10): x, gate_w,
/// gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b.
fn mlpSwigluClosure(inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array {
    if (inputs.len != 10) return error.MlxError;
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = c.c.mlx_default_gpu_stream_new() });
    defer ctx.deinit();
    const x = inputs[0];
    const gate = qwen38_gdn.linearQuantParts(ctx, x, inputs[1], inputs[2], inputs[3]) catch return error.MlxError;
    defer gate.deinit();
    const act = activations.silu(ctx, gate) catch return error.MlxError;
    defer act.deinit();
    const up = qwen38_gdn.linearQuantParts(ctx, x, inputs[4], inputs[5], inputs[6]) catch return error.MlxError;
    defer up.deinit();
    const gated = ops.multiply(ctx, act, up) catch return error.MlxError;
    defer gated.deinit();
    const mlp_out = qwen38_gdn.linearQuantParts(ctx, gated, inputs[7], inputs[8], inputs[9]) catch return error.MlxError;
    const out = allocator.alloc(Array, 1) catch return error.MlxError;
    out[0] = mlp_out;
    return out;
}

/// Build the mx.compile'd SwiGLU core (see mlpSwigluClosure).
pub fn compiledMlpSwiglu(allocator: std.mem.Allocator) !Closure {
    const base = try Closure.init(&mlpSwigluClosure, allocator);
    defer base.deinit();
    return compile_mod.compile(base, false);
}

/// One compiled SwiGLU core on normed x: returns down(silu(gate(x))*up(x))
/// (owned by the caller; x is NOT consumed).
pub fn applyMlpSwiglu(allocator: std.mem.Allocator, mc: Closure, lp: *const LayerPersistent, x: Array) !Array {
    const inputs = [_]Array{
        x,                  lp.mlp_gate.w,
        lp.mlp_gate.scales, lp.mlp_gate.biases,
        lp.mlp_up.w,        lp.mlp_up.scales,
        lp.mlp_up.biases,   lp.mlp_down.w,
        lp.mlp_down.scales, lp.mlp_down.biases,
    };
    const outs = try mc.apply(&inputs, allocator);
    if (outs.len != 1) {
        for (outs) |o| o.deinit();
        allocator.free(outs);
        return error.MlxError;
    }
    const out = outs[0];
    allocator.free(outs);
    return out;
}

// ============================================================
// Fused Metal kernel: silu(gate) * up over one gate|up row
// ============================================================
//
// The fused gate|up qmm produces [.., 2*inter]; the eager SwiGLU tail is
// slice → silu → slice → multiply (4 kernels × 64 layers). This kernel reads
// the gate half and the up half in one pass: out[i] = silu(gu[i]) * gu[i+n].
// silu is computed in f32 and rounded to bf16 before the multiply, mirroring
// the eager silu(bf16)→multiply(bf16) rounding chain (a bf16 product rounds
// identically whether computed in bf16 or f32, since the f32 product is
// exact). Falls back to the eager slice/silu/mul chain when unavailable.

const silu_mul_kernel_body =
    \\    const uint t = thread_position_in_grid.x;
    \\    if (t < {[n]}) {{
    \\        const float g = float(gu[t]);
    \\        const float a = g / (1.0f + metal::exp(-g));
    \\        sm_out[t] = bfloat16_t(float(bfloat16_t(a)) * float(gu[t + {[n]}]));
    \\    }}
    \\
;

const SiluMulEntry = struct {
    n: i32,
    kernel: c.c.mlx_fast_metal_kernel,
    broken: bool = false,
};

var silu_mul_entries: [4]?SiluMulEntry = .{ null, null, null, null };
var silu_mul_stream: ?c.c.mlx_stream = null;
var silu_mul_warned = false;

fn siluMulStream() c.c.mlx_stream {
    if (silu_mul_stream == null) {
        silu_mul_stream = c.c.mlx_default_gpu_stream_new();
    }
    return silu_mul_stream.?;
}

fn siluMulWarn(err: anyerror) void {
    if (!silu_mul_warned) {
        silu_mul_warned = true;
        std.log.warn("fused silu_mul kernel unavailable ({s}); using eager slice/silu/mul", .{@errorName(err)});
    }
}

fn createSiluMulKernel(allocator: std.mem.Allocator, n: i32) !c.c.mlx_fast_metal_kernel {
    const name_z = try std.fmt.allocPrintSentinel(allocator, "silu_mul_n{d}", .{n}, 0);
    defer allocator.free(name_z);
    const source_z = try std.fmt.allocPrintSentinel(allocator, silu_mul_kernel_body, .{ .n = n }, 0);
    defer allocator.free(source_z);
    const header_z = try allocator.dupeZ(u8, "");
    defer allocator.free(header_z);

    const in_vec = try addRmsStringVec(allocator, &.{"gu"});
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = try addRmsStringVec(allocator, &.{"sm_out"});
    defer _ = c.c.mlx_vector_string_free(out_vec);

    return c.c.mlx_fast_metal_kernel_new(
        name_z.ptr,
        in_vec,
        out_vec,
        source_z.ptr,
        header_z.ptr,
        true, // ensure_row_contiguous
        false, // atomic_outputs
    );
}

fn getSiluMulKernel(allocator: std.mem.Allocator, n: i32) ?c.c.mlx_fast_metal_kernel {
    for (&silu_mul_entries) |*slot| {
        if (slot.*) |*e| {
            if (e.n == n) return if (e.broken) null else e.kernel;
        }
    }
    const kernel = createSiluMulKernel(allocator, n) catch |err| {
        siluMulWarn(err);
        return null;
    };
    for (&silu_mul_entries) |*slot| {
        if (slot.* == null) {
            slot.* = .{ .n = n, .kernel = kernel };
            return kernel;
        }
    }
    return null;
}

fn markSiluMulBroken(n: i32) void {
    for (&silu_mul_entries) |*slot| {
        if (slot.*) |*e| {
            if (e.n == n) e.broken = true;
        }
    }
}

/// Eager SwiGLU tail on a fused gate|up row [.., 2*inter]: slice the two
/// halves, silu(gate) * up. Matches mlpBranch/mlpForwardPersistent exactly.
pub fn siluMulEager(ctx: EagerContext, gu: Array) !Array {
    const gs = gu.shape();
    const nd = gs.len;
    if (nd == 0 or nd > 4 or @mod(gs[nd - 1], 2) != 0) return error.MlxError;
    const inter: i32 = @intCast(@divTrunc(gs[nd - 1], 2));
    var start: [4]i32 = [_]i32{0} ** 4;
    var stop: [4]i32 = undefined;
    var strides: [4]i32 = [_]i32{1} ** 4;
    for (gs, 0..) |s, i| stop[i] = @intCast(s);
    stop[nd - 1] = inter;
    const gate = try ops.slice(ctx, gu, start[0..nd], stop[0..nd], strides[0..nd]);
    defer gate.deinit();
    start[nd - 1] = inter;
    stop[nd - 1] = @intCast(gs[nd - 1]);
    const up = try ops.slice(ctx, gu, start[0..nd], stop[0..nd], strides[0..nd]);
    defer up.deinit();
    const act = try activations.silu(ctx, gate);
    defer act.deinit();
    return ops.multiply(ctx, act, up);
}

/// Fused silu(gate)*up on a gate|up row [.., 2*inter]. Returns null for
/// non-bf16 inputs, multi-row (prefill) shapes, or when the kernel is
/// unavailable.
pub fn siluMulFused(ctx: EagerContext, gu: Array) !?Array {
    if (gu.dtype() != .bfloat16) return null;
    const gs = gu.shape();
    const nd = gs.len;
    if (nd == 0 or nd > 4 or @mod(gs[nd - 1], 2) != 0) return null;
    const n: i64 = @divTrunc(gs[nd - 1], 2);
    // Decode only: exactly one gate|up row. Multi-row (prefill) inputs fall
    // back to the eager slice/silu/mul chain — the flat gate/up offset
    // arithmetic in the kernel assumes a single row.
    var rows: i64 = 1;
    for (gs[0 .. nd - 1]) |s| rows *= s;
    if (rows != 1) return null;

    const kernel = getSiluMulKernel(ctx.allocator, @intCast(n)) orelse return null;
    const nt: i32 = @intCast(@min(@as(i64, 1024), std.mem.alignForward(i64, n, 32)));
    const groups = @divTrunc(n + nt - 1, nt);

    const cfg = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg);
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(groups * nt), 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, nt, 1, 1));
    // Output shape mirrors gu with the last dim halved.
    var dims_buf: [4]i32 = undefined;
    for (gs, 0..) |s, i| dims_buf[i] = @intCast(s);
    dims_buf[nd - 1] = @intCast(n);
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &dims_buf, @intCast(nd), c.c.MLX_BFLOAT16));

    const input_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(input_vec);
    try c.check(c.c.mlx_vector_array_append_value(input_vec, gu.inner));
    var output_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(output_vec);
    c.check(c.c.mlx_fast_metal_kernel_apply(&output_vec, kernel, input_vec, cfg, siluMulStream())) catch |err| {
        markSiluMulBroken(@intCast(n));
        siluMulWarn(err);
        return null;
    };

    var out_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&out_handle, output_vec, 0));
    return Array.fromHandle(out_handle);
}

/// silu(gate) * up from a fused gate|up row: fused kernel when eligible,
/// eager slice/silu/mul otherwise.
fn siluMul(ctx: EagerContext, gu: Array) !Array {
    return (try siluMulFused(ctx, gu)) orelse siluMulEager(ctx, gu);
}

/// Compiled fused SwiGLU core: down(silu_mul(gate_up(x))) on an already-normed
/// bf16 x. Input order (7): x, gate_up_w, gate_up_s, gate_up_b, down_w,
/// down_s, down_b. gate_up_w is the row-concatenated packed [gate; up]
/// projection (see fuseGateUp).
fn mlpSwigluFusedClosure(inputs: []const Array, allocator: std.mem.Allocator) error{MlxError}![]Array {
    if (inputs.len != 7) return error.MlxError;
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = c.c.mlx_default_gpu_stream_new() });
    defer ctx.deinit();
    const x = inputs[0];
    const gu = qwen38_gdn.linearQuantParts(ctx, x, inputs[1], inputs[2], inputs[3]) catch return error.MlxError;
    defer gu.deinit();
    const gated = siluMul(ctx, gu) catch return error.MlxError;
    defer gated.deinit();
    const mlp_out = qwen38_gdn.linearQuantParts(ctx, gated, inputs[4], inputs[5], inputs[6]) catch return error.MlxError;
    const out = allocator.alloc(Array, 1) catch return error.MlxError;
    out[0] = mlp_out;
    return out;
}

/// Build the mx.compile'd fused SwiGLU core (see mlpSwigluFusedClosure).
pub fn compiledMlpSwigluFused(allocator: std.mem.Allocator) !Closure {
    const base = try Closure.init(&mlpSwigluFusedClosure, allocator);
    defer base.deinit();
    return compile_mod.compile(base, false);
}

/// One compiled fused SwiGLU apply on normed x (owned by the caller; x is NOT
/// consumed). Requires lp.mlp_gate_up.
pub fn applyMlpSwigluFused(allocator: std.mem.Allocator, mc: Closure, lp: *const LayerPersistent, x: Array) !Array {
    const gu = lp.mlp_gate_up orelse return error.MlxError;
    const inputs = [_]Array{
        x,                  gu.w,          gu.scales,
        gu.biases,          lp.mlp_down.w, lp.mlp_down.scales,
        lp.mlp_down.biases,
    };
    const outs = try mc.apply(&inputs, allocator);
    if (outs.len != 1) {
        for (outs) |o| o.deinit();
        allocator.free(outs);
        return error.MlxError;
    }
    const out = outs[0];
    allocator.free(outs);
    return out;
}

// ============================================================
// Fused Metal kernel: residual add + rmsNorm (bf16 h/branch, f32 weight)
// ============================================================
//
// The decode layer loop runs `h = h + branch_out; normed = rmsNorm(h, w)` as
// an astype→add→astype→rmsNorm→astype chain (4+ kernels × 128/token). This
// kernel does both in one pass over the flat hidden vector: the add is
// computed in f32 and rounded to bf16 (bit-identical to a bf16 add), the
// bf16-rounded value feeds the f32 norm (matching the eager astype), and the
// normed result rounds to bf16. One threadgroup, nt threads, each owning
// npt = ceil(n/nt) elements; block reduction via simd_sum + a 32-slot shared
// buffer. Decode only (total elements == one hidden row); prefill and
// non-bf16 inputs fall back to the eager chain.

const add_rms_kernel_body =
    \\    const uint t = thread_position_in_grid.x;
    \\    float vals[{[npt]}];
    \\    float local = 0.0f;
    \\    for (int i = 0; i < {[npt]}; ++i) {{
    \\        const uint idx = t + i * {[nt]};
    \\        if (idx < {[n]}) {{
    \\            const float s = float(h[idx]) + float(branch[idx]);
    \\            const bfloat16_t r = bfloat16_t(s);
    \\            h_out[idx] = r;
    \\            const float v = float(r);
    \\            vals[i] = v;
    \\            local += v * v;
    \\        }}
    \\    }}
    \\    local = metal::simd_sum(local);
    \\    threadgroup float red[32];
    \\    const uint lane = t % 32;
    \\    const uint wid = t / 32;
    \\    if (lane == 0) red[wid] = local;
    \\    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    if (wid == 0) {{
    \\        const uint nwid = {[nt]} / 32;
    \\        float s2 = (lane < nwid) ? red[lane] : 0.0f;
    \\        s2 = metal::simd_sum(s2);
    \\        if (lane == 0) red[0] = s2;
    \\    }}
    \\    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    const float inv = metal::rsqrt(red[0] / {[n]}.0f + {[eps]s});
    \\    for (int i = 0; i < {[npt]}; ++i) {{
    \\        const uint idx = t + i * {[nt]};
    \\        if (idx < {[n]}) {{
    \\            normed[idx] = bfloat16_t((vals[i] * inv) * float(w[idx]));
    \\        }}
    \\    }}
    \\
;

const AddRmsKey = struct {
    n: i32,
    eps_bits: u32,
};

const AddRmsEntry = struct {
    key: AddRmsKey,
    kernel: c.c.mlx_fast_metal_kernel,
    broken: bool = false,
};

var add_rms_entries: [4]?AddRmsEntry = .{ null, null, null, null };
var add_rms_stream: ?c.c.mlx_stream = null;
var add_rms_warned = false;

fn addRmsStream() c.c.mlx_stream {
    if (add_rms_stream == null) {
        add_rms_stream = c.c.mlx_default_gpu_stream_new();
    }
    return add_rms_stream.?;
}

fn addRmsWarn(err: anyerror) void {
    if (!add_rms_warned) {
        add_rms_warned = true;
        std.log.warn("fused add+rmsNorm kernel unavailable ({s}); using eager op chain", .{@errorName(err)});
    }
}

fn addRmsStringVec(allocator: std.mem.Allocator, names: []const []const u8) !c.c.mlx_vector_string {
    const vec = c.c.mlx_vector_string_new();
    errdefer _ = c.c.mlx_vector_string_free(vec);
    for (names) |n| {
        const n_z = try allocator.dupeZ(u8, n);
        defer allocator.free(n_z);
        try c.check(c.c.mlx_vector_string_append_value(vec, n_z.ptr));
    }
    return vec;
}

fn createAddRmsKernel(allocator: std.mem.Allocator, key: AddRmsKey) !c.c.mlx_fast_metal_kernel {
    const n: usize = @intCast(key.n);
    const nt: usize = if (n >= 1024) 1024 else std.mem.alignForward(usize, n, 32);
    const npt = (n + nt - 1) / nt;
    const eps: f32 = @bitCast(key.eps_bits);
    const eps_z = try std.fmt.allocPrintSentinel(allocator, "{d}f", .{eps}, 0);
    defer allocator.free(eps_z);
    const name_z = try std.fmt.allocPrintSentinel(allocator, "add_rmsnorm_n{d}_{x}", .{ key.n, key.eps_bits }, 0);
    defer allocator.free(name_z);
    const source_z = try std.fmt.allocPrintSentinel(allocator, add_rms_kernel_body, .{
        .n = key.n,
        .nt = nt,
        .npt = npt,
        .eps = eps_z,
    }, 0);
    defer allocator.free(source_z);
    const header_z = try allocator.dupeZ(u8, "");
    defer allocator.free(header_z);

    const in_vec = try addRmsStringVec(allocator, &.{ "h", "branch", "w" });
    defer _ = c.c.mlx_vector_string_free(in_vec);
    const out_vec = try addRmsStringVec(allocator, &.{ "h_out", "normed" });
    defer _ = c.c.mlx_vector_string_free(out_vec);

    return c.c.mlx_fast_metal_kernel_new(
        name_z.ptr,
        in_vec,
        out_vec,
        source_z.ptr,
        header_z.ptr,
        true, // ensure_row_contiguous
        false, // atomic_outputs
    );
}

fn getAddRmsKernel(allocator: std.mem.Allocator, key: AddRmsKey) ?c.c.mlx_fast_metal_kernel {
    for (&add_rms_entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) return if (e.broken) null else e.kernel;
        }
    }
    const kernel = createAddRmsKernel(allocator, key) catch |err| {
        addRmsWarn(err);
        return null;
    };
    for (&add_rms_entries) |*slot| {
        if (slot.* == null) {
            slot.* = .{ .key = key, .kernel = kernel };
            return kernel;
        }
    }
    return null;
}

fn markAddRmsBroken(key: AddRmsKey) void {
    for (&add_rms_entries) |*slot| {
        if (slot.*) |*e| {
            if (std.meta.eql(e.key, key)) e.broken = true;
        }
    }
}

const AddRmsOut = struct {
    h: Array, // bf16, same shape as input h: h + branch
    normed: Array, // bf16, same shape: rmsNorm(h + branch, w)
};

/// Eager reference: bf16 add → astype f32 → rmsNorm → astype bf16 (the exact
/// op sequence this replaces).
pub fn addRmsNormEager(ctx: EagerContext, h: Array, branch: Array, w_f32: Array, eps: f32) !AddRmsOut {
    const next = try ops.add(ctx, h, branch);
    errdefer next.deinit();
    const h_f32 = try ops.astype(ctx, next, .float32);
    defer h_f32.deinit();
    const n_f32 = try fast_mod.rmsNorm(ctx, h_f32, w_f32, eps);
    defer n_f32.deinit();
    const normed = try ops.astype(ctx, n_f32, .bfloat16);
    return .{ .h = next, .normed = normed };
}

/// Fused add+rmsNorm. Returns null for prefill shapes (more than one hidden
/// row), non-bf16 inputs, or when the kernel is unavailable.
pub fn addRmsNormFused(ctx: EagerContext, h: Array, branch: Array, w_f32: Array, eps: f32) !?AddRmsOut {
    if (h.dtype() != .bfloat16 or branch.dtype() != .bfloat16 or w_f32.dtype() != .float32) return null;
    const hs = h.shape();
    const ws = w_f32.shape();
    if (ws.len != 1) return null;
    var n: i64 = 1;
    for (hs) |s| n *= s;
    if (n != ws[0]) return null; // decode only: exactly one hidden row

    const key = AddRmsKey{ .n = @intCast(n), .eps_bits = @bitCast(eps) };
    const kernel = getAddRmsKernel(ctx.allocator, key) orelse return null;
    const n_usize: usize = @intCast(n);
    const nt: i32 = @intCast(if (n_usize >= 1024) @as(usize, 1024) else std.mem.alignForward(usize, n_usize, 32));

    const cfg = c.c.mlx_fast_metal_kernel_config_new();
    defer c.c.mlx_fast_metal_kernel_config_free(cfg);
    try c.check(c.c.mlx_fast_metal_kernel_config_set_grid(cfg, nt, 1, 1));
    try c.check(c.c.mlx_fast_metal_kernel_config_set_thread_group(cfg, nt, 1, 1));
    // Output shape mirrors h (row dims preserved).
    var dims_buf: [4]i32 = undefined;
    const nd: usize = hs.len;
    for (hs, 0..) |s, i| dims_buf[i] = @intCast(s);
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &dims_buf, @intCast(nd), c.c.MLX_BFLOAT16));
    try c.check(c.c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &dims_buf, @intCast(nd), c.c.MLX_BFLOAT16));

    const input_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(input_vec);
    for ([_]Array{ h, branch, w_f32 }) |inp| {
        try c.check(c.c.mlx_vector_array_append_value(input_vec, inp.inner));
    }
    var output_vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(output_vec);
    c.check(c.c.mlx_fast_metal_kernel_apply(&output_vec, kernel, input_vec, cfg, addRmsStream())) catch |err| {
        markAddRmsBroken(key);
        addRmsWarn(err);
        return null;
    };

    var h_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&h_handle, output_vec, 0));
    errdefer _ = c.c.mlx_array_free(h_handle);
    var n_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_vector_array_get(&n_handle, output_vec, 1));
    return .{ .h = Array.fromHandle(h_handle), .normed = Array.fromHandle(n_handle) };
}

/// h + branch → rmsNorm(h + branch, w_f32): fused kernel when eligible,
/// eager chain otherwise.
fn addRmsNorm(ctx: EagerContext, h: Array, branch: Array, w_f32: Array, eps: f32) !AddRmsOut {
    return (try addRmsNormFused(ctx, h, branch, w_f32, eps)) orelse
        addRmsNormEager(ctx, h, branch, w_f32, eps);
}

/// Standalone bf16→f32→rmsNorm→bf16 (no residual add), used for the first
/// layer's input layernorm where there is no preceding branch output.
fn rmsNormOnly(ctx: EagerContext, h: Array, w_f32: Array, eps: f32) !Array {
    const h_f32 = try ops.astype(ctx, h, .float32);
    defer h_f32.deinit();
    const n_f32 = try fast_mod.rmsNorm(ctx, h_f32, w_f32, eps);
    defer n_f32.deinit();
    return ops.astype(ctx, n_f32, .bfloat16);
}

/// Resident serving model: all weights live in a map (loaded once); per-layer
/// modules (GDN/attn), layernorms, MLP projections and lm_head are built ONCE
/// at init with Arrays borrowed from the map — the token loop runs no
/// WeightSource lookups, COW copies, or derived-constant precomputes.
/// Per-layer caches (GDN conv+ssm state / attention KV) are internal — the
/// engine's KVCacheStrategy array is unused.
///
/// LIMITATION: caches are not reset between requests. forwardTokens treats
/// any L>1 input as a fresh prefill and resets state; the engine's decode
/// calls are always L=1.
pub const Qwen38ServingModel = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    cfg: *const qwen38_loader.Qwen38Config,
    ws: WeightSource,
    ws_backing: WeightSource.MapBacked,
    weights: std.StringHashMap(Array), // resident weights, owned by caller
    /// embed_tokens dequantized once at init (bf16 [vocab, hidden], evaled);
    /// null only when the resident map lacks the embed entries (tests).
    embed_table_bf16: ?Array,
    caches: []Qwen38LayerCache,
    /// Persistent per-layer modules + norm weights + borrowed MLP projections.
    /// Empty when the resident map lacks layer weights (tests).
    layers: []LayerPersistent,
    /// Final norm weight (f32, evaled, owned); null when the map lacks it.
    final_norm_f32: ?Array,
    /// lm_head projection (Arrays borrowed from the map); null when absent.
    lm_head: ?glm53_attn.QuantLinear,
    /// mx.compile'd GDN decode step (one graph shared by all GDN layers —
    /// shapes/dtypes are layer-invariant). Null when unavailable (tests,
    /// non-1e-6 eps, or compile/warm failure → eager fallback).
    gdn_step_compiled: ?Closure,
    /// mx.compile'd SwiGLU MLP core (ln + residual stay eager). The
    /// ln+residual-inside-compile variant measured SLOWER than eager
    /// (A/B on the profile harness), so only the qmm/silu/mul core is
    /// compiled.
    mlp_swiglu_compiled: ?Closure,
    /// mx.compile'd fused SwiGLU core over the row-concatenated gate|up
    /// projection (one wide qmm + one silu_mul kernel instead of two qmms +
    /// slice/silu/mul). Preferred over mlp_swiglu_compiled when the layer has
    /// fused gate_up weights.
    mlp_swiglu_fused_compiled: ?Closure,
    /// mx.compile'd attention pre-cache segment (projections → q/k norm →
    /// ropeDynamic → gate sigmoid), one graph shared by all full-attention
    /// layers; KV append + SDPA + o_proj stay eager. Null when unavailable.
    attn_precache_compiled: ?Closure,

    /// Dequantize embed_tokens (packed U32 + scales + biases) to a resident
    /// bf16 [vocab, hidden] table and eval it once, so decode gathers rows
    /// from materialized memory instead of re-dequantizing per token.
    fn buildResidentEmbedTable(ctx: EagerContext, ws: WeightSource, hidden: i32) !Array {
        var embed_q = try ws.quantized("language_model.model.embed_tokens", hidden);
        defer embed_q.deinit();
        const embed_w = try quantize_mod.dequantizeAs(ctx, .{
            .data = embed_q.data,
            .scales = embed_q.scales,
            .biases = embed_q.biases,
            .config = embed_q.cfg,
            .original_shape = &[_]i32{},
        }, .bfloat16);
        errdefer embed_w.deinit();
        try embed_w.eval();
        return embed_w;
    }

    /// Allocate one fresh cache per layer. On failure every cache built so
    /// far is deinited and the slice freed.
    fn initCaches(allocator: std.mem.Allocator, cfg: *const qwen38_loader.Qwen38Config) ![]Qwen38LayerCache {
        const caches = try allocator.alloc(Qwen38LayerCache, cfg.num_hidden_layers);
        errdefer allocator.free(caches);
        var made: usize = 0;
        errdefer for (caches[0..made]) |*lc| lc.deinit();
        for (caches, 0..) |*lc, i| {
            if (cfg.isFullAttention(@intCast(i))) {
                lc.* = .{ .attn = .{} };
            } else {
                lc.* = .{ .gdn = try qwen38_gdn.Qwen38GdnCache.init(
                    allocator,
                    cfg.linear_conv_kernel_dim,
                    (2 * cfg.linear_num_key_heads + cfg.linear_num_value_heads) * cfg.linear_key_head_dim,
                    cfg.linear_num_value_heads,
                    cfg.linear_value_head_dim,
                ) };
            }
            made += 1;
        }
        return caches;
    }

    pub fn init(
        allocator: std.mem.Allocator,
        ctx: EagerContext,
        cfg: *const qwen38_loader.Qwen38Config,
        weights: std.StringHashMap(Array),
    ) !Qwen38ServingModel {
        const caches = try initCaches(allocator, cfg);
        errdefer allocator.free(caches);
        var self = Qwen38ServingModel{
            .allocator = allocator,
            .ctx = ctx,
            .cfg = cfg,
            .ws = undefined,
            .ws_backing = undefined,
            .weights = weights,
            .embed_table_bf16 = null,
            .caches = caches,
            .layers = @constCast(&.{}),
            .final_norm_f32 = null,
            .lm_head = null,
            .gdn_step_compiled = null,
            .mlp_swiglu_compiled = null,
            .mlp_swiglu_fused_compiled = null,
            .attn_precache_compiled = null,
        };
        // NOTE: the WeightSource built here points at this local's fields;
        // the caller MUST reseat after placement:
        //   placed.ws = WeightSource.fromMap(&placed.weights, ctx, &placed.ws_backing);
        self.ws = WeightSource.fromMap(&self.weights, ctx, &self.ws_backing);
        // Dequantize embed_tokens once (was ~3.1 GB dequant+eval per token),
        // then drop the packed entries from the resident map (~0.8 GB).
        if (self.weights.get("language_model.model.embed_tokens.weight") != null) {
            errdefer if (self.embed_table_bf16) |*t| t.deinit();
            self.embed_table_bf16 = try buildResidentEmbedTable(ctx, self.ws, @intCast(cfg.hidden_size));
            inline for (.{ "weight", "scales", "biases" }) |suffix| {
                if (self.weights.fetchRemove("language_model.model.embed_tokens." ++ suffix)) |kv| {
                    self.allocator.free(kv.key);
                    kv.value.deinit();
                }
            }
        }
        // Build the persistent per-layer modules once: weight Arrays are
        // borrowed from the resident map (no COW copies), GDN derived
        // constants precomputed, layernorms cast to f32 and evaluated. Tests
        // with an empty map (zero layers) skip this — the vision-rejection
        // path never touches weights.
        if (cfg.num_hidden_layers > 0 and self.weights.get("language_model.model.layers.0.input_layernorm.weight") != null) {
            const layers = try allocator.alloc(LayerPersistent, cfg.num_hidden_layers);
            var built: usize = 0;
            errdefer {
                for (layers[0..built]) |*lp| lp.deinitShallow();
                allocator.free(layers);
            }
            for (layers, 0..) |*lp, i| {
                lp.* = try buildLayerPersistent(allocator, ctx, cfg, &self.weights, @intCast(i));
                built += 1;
            }
            self.layers = layers;
        }
        // Final norm (f32, owned) + lm_head (borrowed) — hoisted out of the
        // per-token finalLogits path.
        if (self.weights.get("language_model.model.norm.weight")) |nw| {
            const nf = try ops.astype(ctx, nw, .float32);
            errdefer nf.deinit();
            try nf.eval();
            self.final_norm_f32 = nf;
        }
        if (self.weights.get("language_model.lm_head.weight") != null) {
            self.lm_head = try borrowQuantLinear(&self.weights, "language_model.lm_head", @intCast(cfg.hidden_size));
        }
        // Compiled decode steps: one mx.compile graph per sub-block type,
        // shared across layers (shapes/dtypes are layer-invariant). Trace +
        // warm happens here, once; any failure leaves the eager fallback.
        if (self.layers.len > 0 and @abs(cfg.rms_norm_eps - COMPILED_RMS_EPS) < 1e-12 and
            std.c.getenv("DMLX_QWEN38_NO_COMPILE") == null)
        {
            self.initCompiledSteps();
        }
        return self;
    }

    /// Build and warm the compiled GDN/MLP decode closures. Best-effort: a
    /// compile or warm failure disables just that path (eager fallback).
    fn initCompiledSteps(self: *Qwen38ServingModel) void {
        if (qwen38_gdn.compiledStep(self.allocator, true)) |cc| {
            if (self.warmGdnStep(cc)) {
                self.gdn_step_compiled = cc;
                std.debug.print("info: compiled GDN decode step active\n", .{});
            } else {
                std.debug.print("warn: compiled GDN step warm failed — eager fallback\n", .{});
                cc.deinit();
            }
        } else |_| {
            std.debug.print("warn: compiled GDN step build failed — eager fallback\n", .{});
        }
        if (compiledMlpSwiglu(self.allocator)) |mc| {
            if (self.warmMlpSwiglu(mc)) {
                self.mlp_swiglu_compiled = mc;
                std.debug.print("info: compiled MLP swiglu core active\n", .{});
            } else {
                std.debug.print("warn: compiled MLP swiglu warm failed — eager fallback\n", .{});
                mc.deinit();
            }
        } else |_| {
            std.debug.print("warn: compiled MLP swiglu build failed — eager fallback\n", .{});
        }
        // Fused gate|up variant: only when the layer weights fused at init.
        if (self.layers[0].mlp_gate_up != null) {
            if (compiledMlpSwigluFused(self.allocator)) |mc| {
                if (self.warmMlpSwigluFused(mc)) {
                    self.mlp_swiglu_fused_compiled = mc;
                    std.debug.print("info: compiled fused MLP gate_up+swiglu core active\n", .{});
                } else {
                    std.debug.print("warn: compiled fused MLP swiglu warm failed — unfused fallback\n", .{});
                    mc.deinit();
                }
            } else |_| {
                std.debug.print("warn: compiled fused MLP swiglu build failed — unfused fallback\n", .{});
            }
        }
        // Attention pre-cache segment: only for the exact config the closure
        // bakes (quantized, rotary_dims=64, theta=1e7, output gate).
        const cfg = self.cfg;
        if (cfg.attn_output_gate and cfg.rotaryDims() == qwen38_attn.COMPILED_ROTARY_DIMS and
            @abs(cfg.rope_theta - qwen38_attn.COMPILED_ROPE_THETA) < 1.0)
        {
            if (qwen38_attn.compiledPreCache(self.allocator)) |ac| {
                if (self.warmAttnPreCache(ac)) {
                    self.attn_precache_compiled = ac;
                    std.debug.print("info: compiled attention pre-cache segment active\n", .{});
                } else {
                    std.debug.print("warn: compiled attn pre-cache warm failed — eager fallback\n", .{});
                    ac.deinit();
                }
            } else |_| {
                std.debug.print("warn: compiled attn pre-cache build failed — eager fallback\n", .{});
            }
        }
    }

    /// One compiled pre-cache apply: builds the scalar rope offset from the
    /// cache length and returns the segment outputs (owned by the caller).
    /// On failure nothing is consumed — callers fall back to the eager path.
    pub fn applyAttnPreCache(self: *Qwen38ServingModel, ac: Closure, a: *qwen38_attn.Qwen38Attn, cache: *qwen38_attn.Qwen38AttnCache, x: Array) !qwen38_attn.PreCacheOut {
        const offset_arr = try ops.scalarI32(self.ctx, cache.len());
        defer offset_arr.deinit();
        const inputs = [_]Array{
            x,            a.q_proj,     a.q_scales.?, a.q_biases.?,
            a.k_proj,     a.k_scales.?, a.k_biases.?, a.v_proj,
            a.v_scales.?, a.v_biases.?, a.q_norm,     a.k_norm,
            offset_arr,
        };
        const outs = try ac.apply(&inputs, self.allocator);
        if (outs.len != 4) {
            for (outs) |o| o.deinit();
            self.allocator.free(outs);
            return error.MlxError;
        }
        const pre = qwen38_attn.PreCacheOut{
            .q_t = outs[0],
            .k_t = outs[1],
            .v_t = outs[2],
            .gate_sig = outs[3],
        };
        self.allocator.free(outs);
        return pre;
    }

    /// Trace + warm the compiled pre-cache segment on the first full-attention
    /// layer's weights with a zero input; outputs are discarded after eval.
    fn warmAttnPreCache(self: *Qwen38ServingModel, ac: Closure) bool {
        const hidden: i32 = @intCast(self.cfg.hidden_size);
        for (self.layers) |*lp| {
            switch (lp.module) {
                .attn => |*a| {
                    if (a.q_scales == null) return false;
                    var warm_cache: qwen38_attn.Qwen38AttnCache = .{};
                    defer warm_cache.deinit();
                    const x_f32 = creation_mod.zeros(self.ctx, &[_]i32{ 1, hidden }, .float32) catch return false;
                    defer x_f32.deinit();
                    const x = ops.astype(self.ctx, x_f32, .bfloat16) catch return false;
                    defer x.deinit();
                    x.eval() catch return false;
                    var pre = self.applyAttnPreCache(ac, a, &warm_cache, x) catch return false;
                    defer pre.deinit();
                    pre.q_t.eval() catch return false;
                    pre.k_t.eval() catch return false;
                    pre.v_t.eval() catch return false;
                    pre.gate_sig.?.eval() catch return false;
                    // A second apply at a non-zero offset proves the traced
                    // graph is offset-invariant (no per-token retrace).
                    warm_cache.len_ = 17;
                    var pre2 = self.applyAttnPreCache(ac, a, &warm_cache, x) catch return false;
                    defer pre2.deinit();
                    pre2.q_t.eval() catch return false;
                    return true;
                },
                .gdn => {},
            }
        }
        return false;
    }

    /// Trace + warm the compiled SwiGLU core on the first layer's weights.
    fn warmMlpSwiglu(self: *Qwen38ServingModel, mc: Closure) bool {
        const hidden: i32 = @intCast(self.cfg.hidden_size);
        const lp = &self.layers[0];
        const x_f32 = creation_mod.zeros(self.ctx, &[_]i32{ 1, 1, hidden }, .float32) catch return false;
        defer x_f32.deinit();
        const x = ops.astype(self.ctx, x_f32, .bfloat16) catch return false;
        defer x.deinit();
        x.eval() catch return false;
        const out = applyMlpSwiglu(self.allocator, mc, lp, x) catch return false;
        defer out.deinit();
        out.eval() catch return false;
        return true;
    }

    /// Trace + warm the compiled fused SwiGLU core on the first layer's fused
    /// gate|up weights.
    fn warmMlpSwigluFused(self: *Qwen38ServingModel, mc: Closure) bool {
        const hidden: i32 = @intCast(self.cfg.hidden_size);
        const lp = &self.layers[0];
        if (lp.mlp_gate_up == null) return false;
        const x_f32 = creation_mod.zeros(self.ctx, &[_]i32{ 1, 1, hidden }, .float32) catch return false;
        defer x_f32.deinit();
        const x = ops.astype(self.ctx, x_f32, .bfloat16) catch return false;
        defer x.deinit();
        x.eval() catch return false;
        const out = applyMlpSwigluFused(self.allocator, mc, lp, x) catch return false;
        defer out.deinit();
        out.eval() catch return false;
        return true;
    }

    /// One compiled GDN decode step against `cache`: applies the closure and
    /// swaps the new states in. Returns the [1, hidden] output (owned by the
    /// caller). On apply failure the cache is untouched and the error
    /// propagates — callers fall back to the eager path.
    pub fn applyGdnStep(self: *Qwen38ServingModel, cc: Closure, g: *qwen38_gdn.Qwen38GDN, cache: *qwen38_gdn.Qwen38GdnCache, x: Array) !Array {
        const inputs = [_]Array{
            x,             cache.conv_state, cache.ssm_state,
            g.in_proj_qkv, g.qkv_scales.?,   g.qkv_biases.?,
            g.conv1d_t,    g.in_proj_a,      g.a_scales.?,
            g.a_biases.?,  g.in_proj_b,      g.b_scales.?,
            g.b_biases.?,  g.in_proj_z,      g.z_scales.?,
            g.z_biases.?,  g.A_log,          g.dt_bias,
            g.norm_w,      g.out_proj,       g.o_scales.?,
            g.o_biases.?,  g.q_norm_w,       g.k_norm_w,
            g.qk_idx,
        };
        const outs = try cc.apply(&inputs, self.allocator);
        if (outs.len != 3) {
            for (outs) |o| o.deinit();
            self.allocator.free(outs);
            return error.MlxError;
        }
        cache.conv_state.deinit();
        cache.conv_state = outs[1];
        cache.ssm_state.deinit();
        cache.ssm_state = outs[2];
        const out = outs[0];
        self.allocator.free(outs);
        return out;
    }

    /// First apply of a compiled closure triggers the trace (seconds). Runs
    /// the first GDN layer's weights against its zero init state with a zero
    /// input; outputs are discarded (after a full eval, which is what forces
    /// the compile) so the caches stay zeroed.
    fn warmGdnStep(self: *Qwen38ServingModel, cc: Closure) bool {
        const allocator = self.allocator;
        const hidden: i32 = @intCast(self.cfg.hidden_size);
        for (self.layers) |*lp| {
            switch (lp.module) {
                .gdn => |*g| {
                    // Fresh zero cache: applyGdnStep swaps states into the
                    // cache it is given, and the real per-layer caches must
                    // keep their (zero) init state.
                    var warm_cache = qwen38_gdn.Qwen38GdnCache.init(
                        allocator,
                        self.cfg.linear_conv_kernel_dim,
                        (2 * self.cfg.linear_num_key_heads + self.cfg.linear_num_value_heads) * self.cfg.linear_key_head_dim,
                        self.cfg.linear_num_value_heads,
                        self.cfg.linear_value_head_dim,
                    ) catch return false;
                    defer warm_cache.deinit();
                    const x_f32 = creation_mod.zeros(self.ctx, &[_]i32{ 1, hidden }, .float32) catch return false;
                    defer x_f32.deinit();
                    const x = ops.astype(self.ctx, x_f32, .bfloat16) catch return false;
                    defer x.deinit();
                    x.eval() catch return false;
                    const out = self.applyGdnStep(cc, g, &warm_cache, x) catch return false;
                    defer out.deinit();
                    out.eval() catch return false;
                    warm_cache.conv_state.eval() catch return false;
                    warm_cache.ssm_state.eval() catch return false;
                    return true;
                },
                .attn => {},
            }
        }
        return false;
    }

    pub fn deinit(self: *Qwen38ServingModel) void {
        for (self.caches) |*lc| lc.deinit();
        self.allocator.free(self.caches);
        // Shallow: frees only module-owned derived arrays (GDN conv1d_t /
        // q_norm_w / k_norm_w / qk_idx) and the f32 layernorms; borrowed weight Arrays
        // stay with the resident map (freed by the caller after deinit).
        for (self.layers) |*lp| lp.deinitShallow();
        if (self.layers.len > 0) self.allocator.free(self.layers);
        if (self.final_norm_f32) |*a| a.deinit();
        if (self.embed_table_bf16) |*t| t.deinit();
        if (self.gdn_step_compiled) |cc| cc.deinit();
        if (self.mlp_swiglu_compiled) |mc| mc.deinit();
        if (self.mlp_swiglu_fused_compiled) |mc| mc.deinit();
        if (self.attn_precache_compiled) |ac| ac.deinit();
        // remaining weights map entries are owned by the caller (registry adapter).
    }

    /// Reset all per-layer decode state (fresh request). Builds the fresh
    /// caches first and swaps only once every allocation succeeded, so a
    /// failure leaves the old state intact.
    pub fn reset(self: *Qwen38ServingModel) !void {
        const new_caches = try initCaches(self.allocator, self.cfg);
        for (self.caches) |*lc| lc.deinit();
        self.allocator.free(self.caches);
        self.caches = new_caches;
    }

    /// Submit every live per-layer cache state array for async evaluation
    /// (non-blocking). forwardTokens calls this once per token before the
    /// single logits sync, so the next token's lazy graph references
    /// materialized state arrays and graph depth stays O(1) across tokens.
    fn asyncEvalCacheStates(self: *Qwen38ServingModel) !void {
        var states = std.ArrayList(Array).empty;
        defer states.deinit(self.allocator);
        for (self.caches) |*lc| {
            switch (lc.*) {
                .gdn => |*gc| {
                    try states.append(self.allocator, gc.conv_state);
                    try states.append(self.allocator, gc.ssm_state);
                },
                .attn => |*ac| {
                    if (ac.keys) |k| try states.append(self.allocator, k);
                    if (ac.values) |v| try states.append(self.allocator, v);
                },
            }
        }
        try eval_mod.asyncEval(self.allocator, states.items);
    }

    /// ids: [L] u32 token ids → logits [1, L, vocab] f32. L>1 = prefill
    /// (resets state first); L=1 = cached decode step.
    /// Keep the dtype/op sequence in sync with `Qwen38Model.forward` (the
    /// oracle-aligned harness, cos >= 0.9999).
    ///
    /// All weights come from the persistent per-layer modules built at init —
    /// no WeightSource lookups, COW copies, or module construction per token.
    pub fn forwardTokens(self: *Qwen38ServingModel, ids: []const u32) !Array {
        const res = try self.forwardTokensCore(ids, .none);
        return res.logits;
    }

    pub const ForwardHiddenOut = struct {
        logits: Array, // [1, L, vocab]
        hidden: Array, // [1, hidden] bf16, pre-final-norm, LAST position
    };

    pub const ForwardAllHiddenOut = struct {
        logits: Array, // [1, L, vocab]
        hidden_all: Array, // [1, L, hidden] bf16, pre-final-norm, ALL positions
    };

    /// forwardTokens + the pre-final-norm hidden at the last position (the
    /// MTP head's chaining input after a prefill or plain decode step).
    pub fn forwardTokensHidden(self: *Qwen38ServingModel, ids: []const u32) !ForwardHiddenOut {
        const res = try self.forwardTokensCore(ids, .last);
        return .{ .logits = res.logits, .hidden = res.hidden.? };
    }

    /// forwardTokens + pre-final-norm hidden for ALL positions (MTP drafter
    /// prompt pass input; one array, no per-step state retention).
    pub fn forwardTokensAllHidden(self: *Qwen38ServingModel, ids: []const u32) !ForwardAllHiddenOut {
        const res = try self.forwardTokensCore(ids, .all);
        return .{ .logits = res.logits, .hidden_all = res.hidden.? };
    }

    const HiddenCapture = enum { none, last, all };
    const ForwardCoreOut = struct { logits: Array, hidden: ?Array };

    fn forwardTokensCore(self: *Qwen38ServingModel, ids: []const u32, capture: HiddenCapture) !ForwardCoreOut {
        const allocator = self.allocator;
        const cfg = self.cfg;
        const ctx = self.ctx;
        const hidden: i32 = @intCast(cfg.hidden_size);

        try rejectVisionTokens(ids);
        if (ids.len > 1) try self.reset();

        const ids_i32 = try allocator.alloc(i32, ids.len);
        defer allocator.free(ids_i32);
        for (ids, 0..) |t, i| ids_i32[i] = @intCast(t);

        // Gather rows from the resident bf16 table (dequantized once at
        // init) — never re-dequantize the full vocab table per token.
        if (self.embed_table_bf16 == null) {
            self.embed_table_bf16 = try buildResidentEmbedTable(ctx, self.ws, @intCast(cfg.hidden_size));
        }
        const id_arr = try Array.fromData(allocator, i32, ids_i32, &[_]i32{@intCast(ids_i32.len)});
        defer id_arr.deinit();
        const rows = try shape_mod.takeAxis(ctx, self.embed_table_bf16.?, id_arr, 0);
        defer rows.deinit();
        const emb = try ops.reshape(ctx, rows, &[_]i32{ 1, @intCast(ids.len), @intCast(cfg.hidden_size) });
        defer emb.deinit();
        var h = try ops.copy(ctx, emb); // bf16
        defer h.deinit();

        // Fused add+rmsNorm restructure: the norm feeding layer i+1's input
        // layernorm is computed together with layer i's post-MLP residual
        // add (identical dataflow — the eager fallback produces the same
        // values), so `normed1` for the next iteration is carried across.
        var normed1_carry: ?Array = null;
        defer if (normed1_carry) |*nc| nc.deinit();
        var gdn_seen: usize = 0;

        for (self.layers, 0..) |*lp, layer| {
            // Attention sub-block (mirrors Qwen38Model.forward).
            // Prefill sync cadence: sync every 8th GDN layer (and on the
            // final layer) to bound peak memory without paying a pipeline
            // drain per layer.
            gdn_seen += if (lp.module == .gdn) 1 else 0;
            const normed1 = if (normed1_carry) |nc| blk: {
                normed1_carry = null;
                break :blk nc;
            } else try rmsNormOnly(ctx, h, lp.ln1_f32, cfg.rms_norm_eps);
            defer normed1.deinit();
            const attn_out = switch (lp.module) {
                .gdn => |*g| blk: {
                    const cache = &self.caches[layer].gdn;
                    const n = normed1.shape()[1];
                    if (n > 1) break :blk try g.prefill(allocator, normed1, cache, gdn_seen % 8 == 0 or layer + 1 == self.layers.len);
                    const step_in = try ops.reshape(ctx, normed1, &[_]i32{ 1, hidden });
                    defer step_in.deinit();
                    if (self.gdn_step_compiled) |cc| {
                        if (applyGdnStep(self, cc, g, cache, step_in)) |step_out| {
                            defer step_out.deinit();
                            break :blk try ops.reshape(ctx, step_out, &[_]i32{ 1, 1, hidden });
                        } else |_| {} // fall through to the eager path
                    }
                    const step_out = try g.forward(step_in, cache);
                    defer step_out.deinit();
                    break :blk try ops.reshape(ctx, step_out, &[_]i32{ 1, 1, hidden });
                },
                .attn => |*a| blk: {
                    const cache = &self.caches[layer].attn;
                    const n = normed1.shape()[1];
                    if (n > 1) break :blk try a.forward(normed1, cache);
                    if (self.attn_precache_compiled) |ac| {
                        const step_in = try ops.reshape(ctx, normed1, &[_]i32{ 1, hidden });
                        defer step_in.deinit();
                        if (self.applyAttnPreCache(ac, a, cache, step_in)) |pre| {
                            var pre_v = pre;
                            defer pre_v.deinit();
                            const out2d = try a.postCache(pre_v, cache);
                            defer out2d.deinit();
                            break :blk try ops.reshape(ctx, out2d, &[_]i32{ 1, 1, hidden });
                        } else |_| {} // fall through to the eager path
                    }
                    break :blk try a.forward(normed1, cache);
                },
            };
            defer attn_out.deinit();
            // h += attn_out, then rmsNorm(h, ln2) for the MLP branch.
            const r1 = try addRmsNorm(ctx, h, attn_out, lp.ln2_f32, cfg.rms_norm_eps);
            h.deinit();
            h = r1.h;
            const normed2 = r1.normed;
            defer normed2.deinit();
            const mlp_out = blk: {
                if (self.mlp_swiglu_fused_compiled) |mc| {
                    if (lp.mlp_gate_up != null) {
                        if (applyMlpSwigluFused(allocator, mc, lp, normed2)) |mo| {
                            break :blk mo;
                        } else |_| {} // fall through to the unfused path
                    }
                }
                if (self.mlp_swiglu_compiled) |mc| {
                    if (applyMlpSwiglu(allocator, mc, lp, normed2)) |mo| {
                        break :blk mo;
                    } else |_| {} // fall through to the eager path
                }
                break :blk try mlpForwardPersistent(ctx, lp, normed2);
            };
            defer mlp_out.deinit();
            // h += mlp_out, then rmsNorm(h, next layer's ln1) — fused into the
            // residual add. The final layer has no following ln1: plain add.
            if (layer + 1 < self.layers.len) {
                const r2 = try addRmsNorm(ctx, h, mlp_out, self.layers[layer + 1].ln1_f32, cfg.rms_norm_eps);
                h.deinit();
                h = r2.h;
                normed1_carry = r2.normed;
            } else {
                const next = try ops.add(ctx, h, mlp_out);
                h.deinit();
                h = next;
            }
        }

        // Final norm → bf16 → lm_head (mirrors Qwen38Model.forward).
        const hidden_cap: ?Array = switch (capture) {
            .none => null,
            .last => blk: {
                const last = try ops.slice(ctx, h, &[_]i32{ 0, @as(i32, @intCast(ids.len)) - 1, 0 }, &[_]i32{ 1, @intCast(ids.len), @intCast(cfg.hidden_size) }, &[_]i32{ 1, 1, 1 });
                defer last.deinit();
                const flat = try ops.reshape(ctx, last, &[_]i32{ 1, @intCast(cfg.hidden_size) });
                defer flat.deinit();
                break :blk try ops.copy(ctx, flat); // [1, hidden] bf16, pre-norm
            },
            .all => try ops.copy(ctx, h), // [1, L, hidden] bf16, pre-norm
        };
        errdefer if (hidden_cap) |hc| hc.deinit();
        const h_f32 = try ops.astype(ctx, h, .float32);
        defer h_f32.deinit();
        const hidden_final_f32 = try fast_mod.rmsNorm(ctx, h_f32, self.final_norm_f32.?, cfg.rms_norm_eps);
        defer hidden_final_f32.deinit();
        const hidden_final = try ops.astype(ctx, hidden_final_f32, .bfloat16);
        defer hidden_final.deinit();

        const logits = try self.lm_head.?.apply(ctx, hidden_final);
        // Single sync point per token: submit all live cache states for async
        // evaluation (non-blocking), then block only on the logits needed by
        // sampling. Step N+1's graph then references materialized states.
        try self.asyncEvalCacheStates();
        try logits.eval();
        return .{ .logits = logits, .hidden = hidden_cap };
    }
};

/// Verify-pass output (MTP speculative decoding). `gdn[i]` holds the
/// per-position states for GDN layer `gdn_layers[i]`; `attn_old_lens[i]` is
/// the pre-verify `len_` of attention layer `attn_layers[i]` (its slab
/// already contains the L appended rows; rollback truncates `len_`).
pub const Qwen38VerifyOut = struct {
    logits: Array, // [1, L, vocab], lm_head dtype (bf16) — same as forwardTokens
    /// Pre-final-norm hidden states for all L positions [1, L, hidden] bf16.
    /// The MTP head consumes row `accepted-1` (post-norm chaining convention
    /// from Task 1: the head re-norms via pre_fc_norm_hidden).
    hidden_pre_norm_all: Array,
    gdn: []qwen38_gdn.Qwen38GDN.PrefillVerifyResult,
    gdn_layers: []usize,
    attn_old_lens: []i32,
    attn_layers: []usize,

    pub fn deinit(self: *Qwen38VerifyOut, allocator: std.mem.Allocator) void {
        self.logits.deinit();
        self.hidden_pre_norm_all.deinit();
        for (self.gdn) |*r| r.deinit(allocator);
        allocator.free(self.gdn);
        allocator.free(self.gdn_layers);
        allocator.free(self.attn_old_lens);
        allocator.free(self.attn_layers);
    }
};

/// Verify-continuation pass for MTP speculative decoding: identical dataflow
/// to forwardTokens at L>1 but WITHOUT the cache reset — the ids continue
/// the sequence already in the caches. GDN layers run prefillVerify
/// (per-position states retained, caches untouched); attention layers append
/// normally (rollback truncates len_). Returns per-position logits and the
/// pre-final-norm hidden states needed by the MTP head.
pub fn forwardVerifyImpl(self: *Qwen38ServingModel, ids: []const u32) !Qwen38VerifyOut {
    const allocator = self.allocator;
    const cfg = self.cfg;
    const ctx = self.ctx;

    try rejectVisionTokens(ids);
    if (ids.len <= 1) return error.VerifyNeedsSequence;

    const ids_i32 = try allocator.alloc(i32, ids.len);
    defer allocator.free(ids_i32);
    for (ids, 0..) |t, i| ids_i32[i] = @intCast(t);

    if (self.embed_table_bf16 == null) {
        self.embed_table_bf16 = try Qwen38ServingModel.buildResidentEmbedTable(ctx, self.ws, @intCast(cfg.hidden_size));
    }
    const id_arr = try Array.fromData(allocator, i32, ids_i32, &[_]i32{@intCast(ids_i32.len)});
    defer id_arr.deinit();
    const rows = try shape_mod.takeAxis(ctx, self.embed_table_bf16.?, id_arr, 0);
    defer rows.deinit();
    const emb = try ops.reshape(ctx, rows, &[_]i32{ 1, @intCast(ids.len), @intCast(cfg.hidden_size) });
    defer emb.deinit();
    var h = try ops.copy(ctx, emb); // bf16
    errdefer h.deinit();

    var gdn_results = std.ArrayList(qwen38_gdn.Qwen38GDN.PrefillVerifyResult).empty;
    errdefer {
        for (gdn_results.items) |*r| r.deinit(allocator);
        gdn_results.deinit(allocator);
    }
    var gdn_layers = std.ArrayList(usize).empty;
    errdefer gdn_layers.deinit(allocator);
    var attn_old_lens = std.ArrayList(i32).empty;
    errdefer attn_old_lens.deinit(allocator);
    var attn_layers = std.ArrayList(usize).empty;
    errdefer attn_layers.deinit(allocator);

    var normed1_carry: ?Array = null;
    defer if (normed1_carry) |*nc| nc.deinit();

    for (self.layers, 0..) |*lp, layer| {
        const normed1 = if (normed1_carry) |nc| blk: {
            normed1_carry = null;
            break :blk nc;
        } else try rmsNormOnly(ctx, h, lp.ln1_f32, cfg.rms_norm_eps);
        defer normed1.deinit();
        const attn_out = switch (lp.module) {
            .gdn => |*g| blk: {
                const cache = &self.caches[layer].gdn;
                var res = try g.prefillVerify(allocator, normed1, cache);
                errdefer res.deinit(allocator);
                try gdn_results.append(allocator, res);
                try gdn_layers.append(allocator, layer);
                break :blk res.out; // owned by gdn_results now — no defer
            },
            .attn => |*a| blk: {
                const cache = &self.caches[layer].attn;
                try attn_old_lens.append(allocator, cache.len());
                try attn_layers.append(allocator, layer);
                break :blk try a.forward(normed1, cache);
            },
        };
        defer if (lp.module != .gdn) attn_out.deinit();
        // h += attn_out, then rmsNorm(h, ln2) for the MLP branch.
        const r1 = try addRmsNorm(ctx, h, attn_out, lp.ln2_f32, cfg.rms_norm_eps);
        h.deinit();
        h = r1.h;
        const normed2 = r1.normed;
        defer normed2.deinit();
        const mlp_out = blk: {
            if (self.mlp_swiglu_fused_compiled) |mc| {
                if (lp.mlp_gate_up != null) {
                    if (applyMlpSwigluFused(allocator, mc, lp, normed2)) |mo| {
                        break :blk mo;
                    } else |_| {}
                }
            }
            if (self.mlp_swiglu_compiled) |mc| {
                if (applyMlpSwiglu(allocator, mc, lp, normed2)) |mo| {
                    break :blk mo;
                } else |_| {}
            }
            break :blk try mlpForwardPersistent(ctx, lp, normed2);
        };
        defer mlp_out.deinit();
        if (layer + 1 < self.layers.len) {
            const r2 = try addRmsNorm(ctx, h, mlp_out, self.layers[layer + 1].ln1_f32, cfg.rms_norm_eps);
            h.deinit();
            h = r2.h;
            normed1_carry = r2.normed;
        } else {
            const next = try ops.add(ctx, h, mlp_out);
            h.deinit();
            h = next;
        }
    }

    // Capture pre-final-norm hidden for all positions (MTP chaining input).
    const hidden_pre_norm_all = h; // [1, L, hidden] bf16 — ownership moves out
    errdefer hidden_pre_norm_all.deinit();

    const h_f32 = try ops.astype(ctx, h, .float32);
    defer h_f32.deinit();
    const hidden_final_f32 = try fast_mod.rmsNorm(ctx, h_f32, self.final_norm_f32.?, cfg.rms_norm_eps);
    defer hidden_final_f32.deinit();
    const hidden_final = try ops.astype(ctx, hidden_final_f32, .bfloat16);
    defer hidden_final.deinit();
    const logits = try self.lm_head.?.apply(ctx, hidden_final);
    errdefer logits.deinit();
    try logits.eval();

    return .{
        .logits = logits,
        .hidden_pre_norm_all = hidden_pre_norm_all,
        .gdn = try gdn_results.toOwnedSlice(allocator),
        .gdn_layers = try gdn_layers.toOwnedSlice(allocator),
        .attn_old_lens = try attn_old_lens.toOwnedSlice(allocator),
        .attn_layers = try attn_layers.toOwnedSlice(allocator),
    };
}

/// Commit `accepted` ∈ 1..L verify positions: GDN caches take the retained
/// post-step-(accepted-1) states; attention caches keep old+accepted rows.
/// Frees the GDN per-position lists inside `v` (v.deinit still required for
/// logits/hidden/lists-of-results).
pub fn commitVerify(self: *Qwen38ServingModel, v: *Qwen38VerifyOut, accepted: usize) void {
    const a_idx = accepted - 1;
    for (v.gdn, v.gdn_layers) |*r, layer| {
        qwen38_gdn.Qwen38GDN.commitVerifyStates(&self.caches[layer].gdn, r, self.allocator, a_idx);
    }
    for (v.attn_layers, v.attn_old_lens) |layer, old| {
        self.caches[layer].attn.len_ = old + @as(i32, @intCast(accepted));
    }
}

/// Discard the whole verify pass: attention lengths truncated back, GDN
/// caches untouched (their retained states die with v.deinit).
pub fn rollbackVerify(self: *Qwen38ServingModel, v: *Qwen38VerifyOut) void {
    for (v.attn_layers, v.attn_old_lens) |layer, old| {
        self.caches[layer].attn.len_ = old;
    }
}

// ============================================================
// Tests
// ============================================================

test "Qwen38Model: module surface compiles" {
    // WeightSource/QuantParts re-exports and config helpers.
    const cfg = qwen38_loader.Qwen38Config{ .allocator = std.testing.allocator };
    try std.testing.expect(cfg.isFullAttention(3) == false); // empty full_attn_layers
    try std.testing.expectEqual(@as(usize, 64), cfg.rotaryDims());
    _ = WeightSource.fromIndex;
    _ = Qwen38ServingModel.init;
    _ = Qwen38ServingModel.forwardTokens;
}

test "fused add+rmsNorm matches eager chain (decode shapes)" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    for ([_]i32{ 5120, 64 }) |n| {
        const hbuf = try allocator.alloc(f32, @intCast(n));
        defer allocator.free(hbuf);
        const bbuf = try allocator.alloc(f32, @intCast(n));
        defer allocator.free(bbuf);
        const wbuf = try allocator.alloc(f32, @intCast(n));
        defer allocator.free(wbuf);
        for (hbuf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 37 + 11, 23) - 11)) * 0.05;
        for (bbuf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 19 + 7, 17) - 8)) * 0.03;
        for (wbuf, 0..) |*v, i| v.* = 0.5 + @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 3, 11))) * 0.1;

        const h_f32 = try Array.fromData(allocator, f32, hbuf, &[_]i32{ 1, 1, n });
        defer h_f32.deinit();
        const b_f32 = try Array.fromData(allocator, f32, bbuf, &[_]i32{ 1, 1, n });
        defer b_f32.deinit();
        const h = try ops.astype(ctx, h_f32, .bfloat16);
        defer h.deinit();
        const branch = try ops.astype(ctx, b_f32, .bfloat16);
        defer branch.deinit();
        const w = try Array.fromData(allocator, f32, wbuf, &[_]i32{n});
        defer w.deinit();

        const fused = (try addRmsNormFused(ctx, h, branch, w, 1e-6)) orelse
            return error.FusedKernelUnavailable;
        defer fused.h.deinit();
        defer fused.normed.deinit();
        const eager = try addRmsNormEager(ctx, h, branch, w, 1e-6);
        defer eager.h.deinit();
        defer eager.normed.deinit();

        // dataSlice(f32) does not convert dtypes — astype to f32 first.
        const fh = try ops.astype(ctx, fused.h, .float32);
        defer fh.deinit();
        const fno = try ops.astype(ctx, fused.normed, .float32);
        defer fno.deinit();
        const eh = try ops.astype(ctx, eager.h, .float32);
        defer eh.deinit();
        const eno = try ops.astype(ctx, eager.normed, .float32);
        defer eno.deinit();
        try fh.eval();
        try fno.eval();
        try eh.eval();
        try eno.eval();

        const got_h = try fh.dataSlice(f32);
        const exp_h = try eh.dataSlice(f32);
        try std.testing.expectEqual(exp_h.len, got_h.len);
        for (exp_h, got_h) |e, o| try std.testing.expectEqual(e, o); // bf16 add: bit-identical
        const got_n = try fno.dataSlice(f32);
        const exp_n = try eno.dataSlice(f32);
        try std.testing.expectEqual(exp_n.len, got_n.len);
        for (exp_n, got_n) |e, o| try std.testing.expectApproxEqAbs(e, o, 2e-2); // ≤1 bf16 ulp
    }
}

test "fuseGateUp: dequantized fused weight == row-concat of dequantized parts, fused qmm matches" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    const in_dim: i32 = 128;
    const inter: i32 = 16;
    const qcfg = quantize_mod.QuantConfig{ .bits = 4, .group_size = 64, .mode = .affine };

    const gw_buf = try allocator.alloc(f32, @intCast(in_dim * inter));
    defer allocator.free(gw_buf);
    const uw_buf = try allocator.alloc(f32, @intCast(in_dim * inter));
    defer allocator.free(uw_buf);
    for (gw_buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 37 + 11, 23) - 11)) * 0.05;
    for (uw_buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 19 + 7, 17) - 8)) * 0.04;

    const gate_f32 = try Array.fromData(allocator, f32, gw_buf, &[_]i32{ inter, in_dim });
    defer gate_f32.deinit();
    const up_f32 = try Array.fromData(allocator, f32, uw_buf, &[_]i32{ inter, in_dim });
    defer up_f32.deinit();
    var gate_q = try quantize_mod.quantize(ctx, gate_f32, qcfg);
    defer gate_q.deinit(allocator);
    var up_q = try quantize_mod.quantize(ctx, up_f32, qcfg);
    defer up_q.deinit(allocator);

    const gate_ql = glm53_attn.QuantLinear{ .w = gate_q.data, .scales = gate_q.scales, .biases = gate_q.biases, .cfg = qcfg };
    const up_ql = glm53_attn.QuantLinear{ .w = up_q.data, .scales = up_q.scales, .biases = up_q.biases, .cfg = qcfg };
    var fused = (try fuseGateUp(ctx, gate_ql, up_ql)) orelse return error.FusionRejected;
    defer fused.w.deinit();
    defer fused.scales.deinit();
    defer fused.biases.deinit();
    try std.testing.expectEqual(@as(i64, 2 * inter), fused.w.shape()[0]);
    try std.testing.expectEqual(@as(i64, 2 * inter), fused.scales.shape()[0]);
    try std.testing.expectEqual(@as(i64, 2 * inter), fused.biases.shape()[0]);

    // Packed row-concat must dequantize to exactly the row-concat of the
    // separate dequantized weights (bit-identical, per-row affine dequant).
    const fused_deq = try quantize_mod.dequantizeAs(ctx, .{
        .data = fused.w,
        .scales = fused.scales,
        .biases = fused.biases,
        .config = fused.cfg,
        .original_shape = &[_]i32{},
    }, .float32);
    defer fused_deq.deinit();
    const gate_deq = try quantize_mod.dequantizeAs(ctx, .{
        .data = gate_q.data,
        .scales = gate_q.scales,
        .biases = gate_q.biases,
        .config = qcfg,
        .original_shape = &[_]i32{},
    }, .float32);
    defer gate_deq.deinit();
    const up_deq = try quantize_mod.dequantizeAs(ctx, .{
        .data = up_q.data,
        .scales = up_q.scales,
        .biases = up_q.biases,
        .config = qcfg,
        .original_shape = &[_]i32{},
    }, .float32);
    defer up_deq.deinit();
    const sep_deq = try shape_mod.concatenateAxis(ctx, &[_]Array{ gate_deq, up_deq }, 0);
    defer sep_deq.deinit();
    try fused_deq.eval();
    try sep_deq.eval();
    const fd = try fused_deq.dataSlice(f32);
    const sd = try sep_deq.dataSlice(f32);
    try std.testing.expectEqual(sd.len, fd.len);
    for (sd, fd) |e, o| try std.testing.expectEqual(e, o); // bit-identical

    // One fused qmm == the two separate qmms concatenated (decode shape).
    const x_buf = try allocator.alloc(f32, @intCast(in_dim));
    defer allocator.free(x_buf);
    for (x_buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 3, 11) - 5)) * 0.1;
    const x_f32 = try Array.fromData(allocator, f32, x_buf, &[_]i32{ 1, 1, in_dim });
    defer x_f32.deinit();
    const x = try ops.astype(ctx, x_f32, .bfloat16);
    defer x.deinit();

    const fused_out = try fused.apply(ctx, x);
    defer fused_out.deinit();
    const g_out = try gate_ql.apply(ctx, x);
    defer g_out.deinit();
    const u_out = try up_ql.apply(ctx, x);
    defer u_out.deinit();
    const sep_out = try shape_mod.concatenateAxis(ctx, &[_]Array{ g_out, u_out }, 2);
    defer sep_out.deinit();
    const fo = try ops.astype(ctx, fused_out, .float32);
    defer fo.deinit();
    const so = try ops.astype(ctx, sep_out, .float32);
    defer so.deinit();
    try fo.eval();
    try so.eval();
    const fos = try fo.dataSlice(f32);
    const sos = try so.dataSlice(f32);
    try std.testing.expectEqual(sos.len, fos.len);
    for (sos, fos) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-2);
}

test "siluMulFused matches eager slice+silu+mul (decode shapes)" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    for ([_]i32{ 17408, 64 }) |n| {
        const buf = try allocator.alloc(f32, @intCast(2 * n));
        defer allocator.free(buf);
        for (buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 37 + 11, 23) - 11)) * 0.3;

        const gu_f32 = try Array.fromData(allocator, f32, buf, &[_]i32{ 1, 1, 2 * n });
        defer gu_f32.deinit();
        const gu = try ops.astype(ctx, gu_f32, .bfloat16);
        defer gu.deinit();

        const fused = (try siluMulFused(ctx, gu)) orelse
            return error.FusedKernelUnavailable;
        defer fused.deinit();
        try std.testing.expectEqual(@as(i64, n), fused.shape()[2]);
        const eager = try siluMulEager(ctx, gu);
        defer eager.deinit();

        const fo = try ops.astype(ctx, fused, .float32);
        defer fo.deinit();
        const eo = try ops.astype(ctx, eager, .float32);
        defer eo.deinit();
        try fo.eval();
        try eo.eval();
        const fos = try fo.dataSlice(f32);
        const eos = try eo.dataSlice(f32);
        try std.testing.expectEqual(eos.len, fos.len);
        for (eos, fos) |e, o| {
            // ≤1 bf16 ulp (ulp grows with magnitude; 0.03125 at |x|≈8).
            const tol = 0.0625 * @max(@abs(e), 1.0);
            try std.testing.expectApproxEqAbs(e, o, tol);
        }
    }
}

test "compiled fused SwiGLU matches eager unfused MLP" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    const in_dim: i32 = 128;
    const inter: i32 = 64;
    const hidden_down: i32 = 128;
    const qcfg = quantize_mod.QuantConfig{ .bits = 4, .group_size = 64, .mode = .affine };

    const gw_buf = try allocator.alloc(f32, @intCast(in_dim * inter));
    defer allocator.free(gw_buf);
    const uw_buf = try allocator.alloc(f32, @intCast(in_dim * inter));
    defer allocator.free(uw_buf);
    const dw_buf = try allocator.alloc(f32, @intCast(hidden_down * inter));
    defer allocator.free(dw_buf);
    for (gw_buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 37 + 11, 23) - 11)) * 0.05;
    for (uw_buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 19 + 7, 17) - 8)) * 0.04;
    for (dw_buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 23 + 5, 19) - 9)) * 0.03;

    const gate_f32 = try Array.fromData(allocator, f32, gw_buf, &[_]i32{ inter, in_dim });
    defer gate_f32.deinit();
    const up_f32 = try Array.fromData(allocator, f32, uw_buf, &[_]i32{ inter, in_dim });
    defer up_f32.deinit();
    const down_f32 = try Array.fromData(allocator, f32, dw_buf, &[_]i32{ hidden_down, inter });
    defer down_f32.deinit();
    var gate_q = try quantize_mod.quantize(ctx, gate_f32, qcfg);
    defer gate_q.deinit(allocator);
    var up_q = try quantize_mod.quantize(ctx, up_f32, qcfg);
    defer up_q.deinit(allocator);
    var down_q = try quantize_mod.quantize(ctx, down_f32, qcfg);
    defer down_q.deinit(allocator);

    var lp: LayerPersistent = .{
        .ln1_f32 = undefined,
        .ln2_f32 = undefined,
        .module = undefined,
        .mlp_gate = .{ .w = gate_q.data, .scales = gate_q.scales, .biases = gate_q.biases, .cfg = qcfg },
        .mlp_up = .{ .w = up_q.data, .scales = up_q.scales, .biases = up_q.biases, .cfg = qcfg },
        .mlp_down = .{ .w = down_q.data, .scales = down_q.scales, .biases = down_q.biases, .cfg = qcfg },
    };
    if (try fuseGateUp(ctx, lp.mlp_gate, lp.mlp_up)) |fused| {
        lp.gate_up_bufs = .{ fused.w, fused.scales, fused.biases };
        lp.mlp_gate_up = fused;
    } else return error.FusionRejected;
    defer for (&lp.gate_up_bufs.?) |*b| b.deinit();

    const x_buf = try allocator.alloc(f32, @intCast(in_dim));
    defer allocator.free(x_buf);
    for (x_buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 13 + 3, 11) - 5)) * 0.1;
    const x_f32 = try Array.fromData(allocator, f32, x_buf, &[_]i32{ 1, 1, in_dim });
    defer x_f32.deinit();
    const x = try ops.astype(ctx, x_f32, .bfloat16);
    defer x.deinit();
    try x.eval();

    const ref = try mlpForwardPersistent(ctx, &lp, x);
    defer ref.deinit();

    const mc = try compiledMlpSwigluFused(allocator);
    defer mc.deinit();
    const got = try applyMlpSwigluFused(allocator, mc, &lp, x);
    defer got.deinit();

    const ro = try ops.astype(ctx, ref, .float32);
    defer ro.deinit();
    const go = try ops.astype(ctx, got, .float32);
    defer go.deinit();
    try ro.eval();
    try go.eval();
    const ros = try ro.dataSlice(f32);
    const gos = try go.dataSlice(f32);
    try std.testing.expectEqual(ros.len, gos.len);
    var max_diff: f32 = 0;
    for (ros, gos) |e, o| max_diff = @max(max_diff, @abs(e - o));
    for (ros, gos) |e, o| {
        const tol = 0.0625 * @max(@abs(e), 1.0);
        try std.testing.expectApproxEqAbs(e, o, tol);
    }
}

test "rejectVisionTokens: vision range rejected, text ids pass" {
    // Text ids (incl. eos and im_start/im_end) pass through.
    try rejectVisionTokens(&[_]u32{ 248044, 248045, 248046, 100, 0, 248320 - 1 });
    try rejectVisionTokens(&[_]u32{});
    // Every id in the vision range is rejected.
    var t: u32 = vision_token_lo;
    while (t <= vision_token_hi) : (t += 1) {
        try std.testing.expectError(VisionError.VisionInputNotSupported, rejectVisionTokens(&[_]u32{ 97063, t }));
    }
    // Boundary check: ids just outside the range pass.
    try rejectVisionTokens(&[_]u32{ vision_token_lo - 1, vision_token_hi + 1 });
}

test "Qwen38ServingModel.forwardTokens: vision token rejected before any weight access" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    // Zero-layer config + empty weights: any forward past the vision check
    // would fail on the missing embed_tokens weight, so a clean
    // VisionInputNotSupported proves the check fires first. Covers both the
    // prefill (L>1) and decode (L=1) paths.
    var cfg = qwen38_loader.Qwen38Config{ .allocator = allocator, .num_hidden_layers = 0 };
    defer cfg.deinit();
    const weights = std.StringHashMap(Array).init(allocator);
    var model = try Qwen38ServingModel.init(allocator, ctx, &cfg, weights);
    defer model.deinit();

    try std.testing.expectError(
        VisionError.VisionInputNotSupported,
        model.forwardTokens(&[_]u32{248056}), // <|image_pad|>, decode path
    );
    try std.testing.expectError(
        VisionError.VisionInputNotSupported,
        model.forwardTokens(&[_]u32{ 100, 248053, 200 }), // <|vision_start|>, prefill path
    );
}

// ============================================================
// MTP forwardVerify toy-model tests
// ============================================================

fn testFillToy(buf: []f32) void {
    for (buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@mod(@as(i64, @intCast(i)) * 37 + 11, 13) - 6)) * 0.02;
    }
}

fn putToyTensor(allocator: std.mem.Allocator, map: *std.StringHashMap(Array), name: []const u8, shape_: []const i32) !void {
    var n: usize = 1;
    for (shape_) |s| n *= @intCast(s);
    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    testFillToy(buf);
    const arr = try Array.fromData(allocator, f32, buf, shape_);
    errdefer arr.deinit();
    try map.put(try allocator.dupe(u8, name), arr);
}

fn putToyQuant(allocator: std.mem.Allocator, ctx: EagerContext, map: *std.StringHashMap(Array), prefix: []const u8, out_f: i32, in_f: i32) !void {
    var nbuf: [256]u8 = undefined;
    const n: usize = @intCast(out_f * in_f);
    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    testFillToy(buf);
    const w = try Array.fromData(allocator, f32, buf, &[_]i32{ out_f, in_f });
    defer w.deinit();
    const qw = try quantize_mod.quantize(ctx, w, .{ .group_size = 64, .bits = 4, .mode = .affine });
    defer allocator.free(qw.original_shape);
    try map.put(try allocator.dupe(u8, try std.fmt.bufPrint(&nbuf, "{s}.weight", .{prefix})), qw.data);
    try map.put(try allocator.dupe(u8, try std.fmt.bufPrint(&nbuf, "{s}.scales", .{prefix})), qw.scales);
    try map.put(try allocator.dupe(u8, try std.fmt.bufPrint(&nbuf, "{s}.biases", .{prefix})), qw.biases);
}

test "Qwen38ServingModel.forwardVerify: continuation, commit, rollback" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const fa = try allocator.dupe(u32, &[_]u32{1});
    var cfg = qwen38_loader.Qwen38Config{
        .allocator = allocator,
        .hidden_size = 64,
        .num_hidden_layers = 2,
        .vocab_size = 64,
        .intermediate_size = 128,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 32,
        .partial_rotary_factor = 0.25,
        .linear_num_key_heads = 2,
        .linear_num_value_heads = 8,
        .linear_key_head_dim = 8,
        .linear_value_head_dim = 8,
        .full_attn_layers = fa,
    };
    defer cfg.deinit();

    const mkModel = struct {
        fn call(alloc: std.mem.Allocator, c2: EagerContext, c3: *qwen38_loader.Qwen38Config) !Qwen38ServingModel {
            const hidden: i32 = 64;
            const vocab: i32 = 64;
            const inter: i32 = 128;
            const gdn_qkv: i32 = 96;
            const gdn_z: i32 = 64;
            var map = std.StringHashMap(Array).init(alloc);
            errdefer {
                var it = map.iterator();
                while (it.next()) |e| {
                    alloc.free(e.key_ptr.*);
                    e.value_ptr.deinit();
                }
                map.deinit();
            }
            try putToyQuant(alloc, c2, &map, "language_model.model.embed_tokens", vocab, hidden);
            try putToyTensor(alloc, &map, "language_model.model.norm.weight", &[_]i32{hidden});
            try putToyQuant(alloc, c2, &map, "language_model.lm_head", vocab, hidden);
            try putToyTensor(alloc, &map, "language_model.model.layers.0.input_layernorm.weight", &[_]i32{hidden});
            try putToyTensor(alloc, &map, "language_model.model.layers.0.post_attention_layernorm.weight", &[_]i32{hidden});
            const la = "language_model.model.layers.0.linear_attn";
            try putToyQuant(alloc, c2, &map, la ++ ".in_proj_qkv", gdn_qkv, hidden);
            try putToyQuant(alloc, c2, &map, la ++ ".in_proj_a", 8, hidden);
            try putToyQuant(alloc, c2, &map, la ++ ".in_proj_b", 8, hidden);
            try putToyQuant(alloc, c2, &map, la ++ ".in_proj_z", gdn_z, hidden);
            try putToyQuant(alloc, c2, &map, la ++ ".out_proj", hidden, gdn_z);
            try putToyTensor(alloc, &map, la ++ ".conv1d.weight", &[_]i32{ gdn_qkv, 4, 1 });
            try putToyTensor(alloc, &map, la ++ ".A_log", &[_]i32{8});
            try putToyTensor(alloc, &map, la ++ ".dt_bias", &[_]i32{8});
            try putToyTensor(alloc, &map, la ++ ".norm.weight", &[_]i32{8});
            try putToyQuant(alloc, c2, &map, "language_model.model.layers.0.mlp.gate_proj", inter, hidden);
            try putToyQuant(alloc, c2, &map, "language_model.model.layers.0.mlp.up_proj", inter, hidden);
            try putToyQuant(alloc, c2, &map, "language_model.model.layers.0.mlp.down_proj", hidden, inter);
            try putToyTensor(alloc, &map, "language_model.model.layers.1.input_layernorm.weight", &[_]i32{hidden});
            try putToyTensor(alloc, &map, "language_model.model.layers.1.post_attention_layernorm.weight", &[_]i32{hidden});
            const sa = "language_model.model.layers.1.self_attn";
            try putToyQuant(alloc, c2, &map, sa ++ ".q_proj", 128, hidden);
            try putToyQuant(alloc, c2, &map, sa ++ ".k_proj", 32, hidden);
            try putToyQuant(alloc, c2, &map, sa ++ ".v_proj", 32, hidden);
            try putToyQuant(alloc, c2, &map, sa ++ ".o_proj", hidden, 64);
            try putToyTensor(alloc, &map, sa ++ ".q_norm.weight", &[_]i32{32});
            try putToyTensor(alloc, &map, sa ++ ".k_norm.weight", &[_]i32{32});
            try putToyQuant(alloc, c2, &map, "language_model.model.layers.1.mlp.gate_proj", inter, hidden);
            try putToyQuant(alloc, c2, &map, "language_model.model.layers.1.mlp.up_proj", inter, hidden);
            try putToyQuant(alloc, c2, &map, "language_model.model.layers.1.mlp.down_proj", hidden, inter);
            var m = try Qwen38ServingModel.init(alloc, c2, c3, map);
            m.ws = WeightSource.fromMap(&m.weights, c2, &m.ws_backing);
            return m;
        }
    }.call;

    // model: prefill 3, then verify [4,5] as continuation.
    var model = try mkModel(allocator, ctx, &cfg);
    defer {
        model.deinit();
        var it = model.weights.iterator();
        while (it.next()) |e| {
            allocator.free(e.key_ptr.*);
            e.value_ptr.deinit();
        }
        model.weights.deinit();
    }
    // reference: fresh model, one-shot prefill of all 5.
    var ref = try mkModel(allocator, ctx, &cfg);
    defer {
        ref.deinit();
        var it = ref.weights.iterator();
        while (it.next()) |e| {
            allocator.free(e.key_ptr.*);
            e.value_ptr.deinit();
        }
        ref.weights.deinit();
    }

    const l0 = try model.forwardTokens(&[_]u32{ 1, 2, 3 });
    l0.deinit();
    const l_ref = try ref.forwardTokens(&[_]u32{ 1, 2, 3, 4, 5 });
    defer l_ref.deinit();

    var v = try forwardVerifyImpl(&model, &[_]u32{ 4, 5 });
    defer v.deinit(allocator);
    const vshape = v.logits.shape();
    try std.testing.expectEqual(@as(i32, 2), vshape[1]);
    try std.testing.expectEqual(@as(i32, 64), vshape[2]);
    try std.testing.expectEqual(@as(i32, 2), v.hidden_pre_norm_all.shape()[1]);

    // Logits of the verify positions match the fresh full prefill.
    {
        const ref_tail = try ops.slice(ctx, l_ref, &[_]i32{ 0, 3, 0 }, &[_]i32{ 1, 5, 64 }, &[_]i32{ 1, 1, 1 });
        defer ref_tail.deinit();
        const got_f32 = try ops.astype(ctx, v.logits, .float32);
        defer got_f32.deinit();
        const exp_f32 = try ops.astype(ctx, ref_tail, .float32);
        defer exp_f32.deinit();
        const got = try got_f32.dataSlice(f32);
        const exp = try exp_f32.dataSlice(f32);
        for (exp, got) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-4);
    }

    // hidden_pre_norm row i must produce logits row i via final norm + lm_head.
    {
        const row0 = try ops.slice(ctx, v.hidden_pre_norm_all, &[_]i32{ 0, 0, 0 }, &[_]i32{ 1, 1, 64 }, &[_]i32{ 1, 1, 1 });
        defer row0.deinit();
        const r0f = try ops.astype(ctx, row0, .float32);
        defer r0f.deinit();
        const n0 = try fast_mod.rmsNorm(ctx, r0f, model.final_norm_f32.?, cfg.rms_norm_eps);
        defer n0.deinit();
        const n0b = try ops.astype(ctx, n0, .bfloat16);
        defer n0b.deinit();
        const lg0 = try model.lm_head.?.apply(ctx, n0b);
        defer lg0.deinit();
        const lg0f = try ops.astype(ctx, lg0, .float32);
        defer lg0f.deinit();
        const v_col0 = try ops.slice(ctx, v.logits, &[_]i32{ 0, 0, 0 }, &[_]i32{ 1, 1, 64 }, &[_]i32{ 1, 1, 1 });
        defer v_col0.deinit();
        const v0f = try ops.astype(ctx, v_col0, .float32);
        defer v0f.deinit();
        const got = try lg0f.dataSlice(f32);
        const exp = try v0f.dataSlice(f32);
        for (exp, got) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-4);
    }

    // Commit both verify positions, then decode token 6 — must match the
    // reference model (which prefilled all 5) decoding token 6.
    commitVerify(&model, &v, 2);
    const l_a = try model.forwardTokens(&[_]u32{6});
    defer l_a.deinit();
    const l_b = try ref.forwardTokens(&[_]u32{6});
    defer l_b.deinit();
    {
        const a32 = try ops.astype(ctx, l_a, .float32);
        defer a32.deinit();
        const b32 = try ops.astype(ctx, l_b, .float32);
        defer b32.deinit();
        const got = try a32.dataSlice(f32);
        const exp = try b32.dataSlice(f32);
        for (exp, got) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-4);
    }

    // Rollback: a second model prefills 3, verifies [4,5], rolls back, then
    // decodes token 4 — must match a fresh 3-prefill model decoding 4.
    var model_r = try mkModel(allocator, ctx, &cfg);
    defer {
        model_r.deinit();
        var it = model_r.weights.iterator();
        while (it.next()) |e| {
            allocator.free(e.key_ptr.*);
            e.value_ptr.deinit();
        }
        model_r.weights.deinit();
    }
    var ref_r = try mkModel(allocator, ctx, &cfg);
    defer {
        ref_r.deinit();
        var it = ref_r.weights.iterator();
        while (it.next()) |e| {
            allocator.free(e.key_ptr.*);
            e.value_ptr.deinit();
        }
        ref_r.weights.deinit();
    }
    const lr0 = try model_r.forwardTokens(&[_]u32{ 1, 2, 3 });
    lr0.deinit();
    const lr_ref0 = try ref_r.forwardTokens(&[_]u32{ 1, 2, 3 });
    lr_ref0.deinit();
    var v2 = try forwardVerifyImpl(&model_r, &[_]u32{ 4, 5 });
    defer v2.deinit(allocator);
    rollbackVerify(&model_r, &v2);
    const l_r = try model_r.forwardTokens(&[_]u32{4});
    defer l_r.deinit();
    const l_rr = try ref_r.forwardTokens(&[_]u32{4});
    defer l_rr.deinit();
    {
        const a32 = try ops.astype(ctx, l_r, .float32);
        defer a32.deinit();
        const b32 = try ops.astype(ctx, l_rr, .float32);
        defer b32.deinit();
        const got = try a32.dataSlice(f32);
        const exp = try b32.dataSlice(f32);
        for (exp, got) |e, o| try std.testing.expectApproxEqAbs(e, o, 1e-4);
    }
}
