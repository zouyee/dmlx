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
// Tests
// ============================================================

test "Qwen38Model: module surface compiles" {
    // WeightSource/QuantParts re-exports and config helpers.
    const cfg = qwen38_loader.Qwen38Config{ .allocator = std.testing.allocator };
    try std.testing.expect(cfg.isFullAttention(3) == false); // empty full_attn_layers
    try std.testing.expectEqual(@as(usize, 64), cfg.rotaryDims());
    _ = WeightSource.fromIndex;
}
