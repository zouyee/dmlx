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
const qwen38_loader = @import("qwen38_loader.zig");
const qwen38_gdn = @import("qwen38_gdn.zig");
const qwen38_attn = @import("qwen38_attn.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

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

/// Resident serving model: all weights live in a map (loaded once, borrowed
/// via WeightSource.fromMap COW copies); dense model — no expert streaming.
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
        return self;
    }

    pub fn deinit(self: *Qwen38ServingModel) void {
        for (self.caches) |*lc| lc.deinit();
        self.allocator.free(self.caches);
        if (self.embed_table_bf16) |*t| t.deinit();
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
    pub fn forwardTokens(self: *Qwen38ServingModel, ids: []const u32) !Array {
        const allocator = self.allocator;
        const cfg = self.cfg;
        const ctx = self.ctx;
        var nbuf: [256]u8 = undefined;

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

        var layer: u32 = 0;
        while (layer < cfg.num_hidden_layers) : (layer += 1) {
            var pbuf: [96]u8 = undefined;
            const p = fmtName(&pbuf, "language_model.model.layers.{d}", .{layer});

            // Attention sub-block (mirrors Qwen38Model.forward).
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
            const attn_out = if (cfg.isFullAttention(layer))
                try attnBranch(ctx, self.ws, allocator, cfg, layer, normed1, &self.caches[layer].attn)
            else
                try gdnBranch(ctx, self.ws, allocator, cfg, layer, normed1, &self.caches[layer].gdn);
            defer attn_out.deinit();
            {
                const next = try ops.add(ctx, h, attn_out);
                h.deinit();
                h = next;
            }

            // MLP sub-block.
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
            }
        }

        // Final norm → bf16 → lm_head (mirrors Qwen38Model.forward).
        const h_f32 = try ops.astype(ctx, h, .float32);
        defer h_f32.deinit();
        const norm_w = try loadNormWeight(ctx, self.ws, "language_model.model.norm.weight");
        defer norm_w.deinit();
        const hidden_final_f32 = try fast_mod.rmsNorm(ctx, h_f32, norm_w, cfg.rms_norm_eps);
        defer hidden_final_f32.deinit();
        const hidden_final = try ops.astype(ctx, hidden_final_f32, .bfloat16);
        defer hidden_final.deinit();

        const hidden: i32 = @intCast(cfg.hidden_size);
        var head_q = try self.ws.quantized("language_model.lm_head", hidden);
        defer head_q.deinit();
        const logits = try head_q.linear().apply(ctx, hidden_final);
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
