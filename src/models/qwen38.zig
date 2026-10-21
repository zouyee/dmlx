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
        .ones_d = pre.ones_d,
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
        return gdn.prefill(allocator, x, cache);
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
/// ones_d / qk_idx; attn: none). deinitShallow frees only the owned derived
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
                g.ones_d.deinit();
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

    pub fn deinitShallow(self: *LayerPersistent) void {
        self.ln1_f32.deinit();
        self.ln2_f32.deinit();
        self.module.deinitShallow();
    }
};

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
        try pre.ones_d.eval();

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
            .ones_d = pre.ones_d,
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
            g.o_biases.?,  g.ones_d,         g.qk_idx,
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
        // ones_d / qk_idx) and the f32 layernorms; borrowed weight Arrays
        // stay with the resident map (freed by the caller after deinit).
        for (self.layers) |*lp| lp.deinitShallow();
        if (self.layers.len > 0) self.allocator.free(self.layers);
        if (self.final_norm_f32) |*a| a.deinit();
        if (self.embed_table_bf16) |*t| t.deinit();
        if (self.gdn_step_compiled) |cc| cc.deinit();
        if (self.mlp_swiglu_compiled) |mc| mc.deinit();
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

        for (self.layers, 0..) |*lp, layer| {
            // Attention sub-block (mirrors Qwen38Model.forward).
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
                    const cache = &self.caches[layer].gdn;
                    const n = normed1.shape()[1];
                    if (n > 1) break :blk try g.prefill(allocator, normed1, cache);
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
            {
                const next = try ops.add(ctx, h, attn_out);
                h.deinit();
                h = next;
            }

            // MLP sub-block.
            const normed2 = blk: {
                const h_f32 = try ops.astype(ctx, h, .float32);
                defer h_f32.deinit();
                const n_f32 = try fast_mod.rmsNorm(ctx, h_f32, lp.ln2_f32, cfg.rms_norm_eps);
                defer n_f32.deinit();
                break :blk try ops.astype(ctx, n_f32, .bfloat16);
            };
            defer normed2.deinit();
            if (self.mlp_swiglu_compiled) |mc| {
                if (applyMlpSwiglu(allocator, mc, lp, normed2)) |mlp_out| {
                    defer mlp_out.deinit();
                    const next = try ops.add(ctx, h, mlp_out);
                    h.deinit();
                    h = next;
                    continue;
                } else |_| {} // fall through to the eager path
            }
            const mlp_out = try mlpForwardPersistent(ctx, lp, normed2);
            defer mlp_out.deinit();
            {
                const next = try ops.add(ctx, h, mlp_out);
                h.deinit();
                h = next;
            }
        }

        // Final norm → bf16 → lm_head (mirrors Qwen38Model.forward).
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
        return logits;
    }
};

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
