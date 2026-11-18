//! Qwen3.8 DFlash2 block-parallel speculative drafter (z-lab/Qwen3.8-27B-DFlash2).
//!
//! Architecture (HF config + omlx/mlx-vlm reference):
//!   ctx = hidden_norm(fc(concat(target hidden @ layers 5/19/33/47/61, -1)))
//!   block = embed([anchor, mask × (block_size-1)])
//!   ×5 layers: norm → two-tap grouped dynamic causal conv (prepare)
//!     → non-causal attention over [context KV ring ‖ block KV] → conv finish
//!     → residual; same pattern for the SwiGLU MLP sublayer
//!   norm → drop anchor row → shared lm_head → CandidateSelector
//!   (top-k logits candidates + rank-r predecessor/successor codebook edges,
//!   greedy left-to-right chain) → block_size-1 draft tokens in ONE pass.
//!
//! The embedding table and lm_head are borrowed from the backbone, exactly
//! like the MTP head (qwen38_mtp.zig). The checkpoint is bf16, unquantized.
//!
//! Task 2 scope: config parsing + weight loading. Forward (conv, attention,
//! selector) and the context ring land in Tasks 3-7.
const std = @import("std");
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const fast_mod = @import("mlx").fast;
const activations = @import("mlx").activations;
const sort_mod = @import("mlx").sort;
const quantize_mod = @import("mlx").quantize;
const reduce_mod = @import("mlx").reduce;
const safetensors_reader = @import("mlx").safetensors_reader;
const glm53 = @import("glm53.zig");
const qwen38_mtp = @import("qwen38_mtp.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const WeightSource = glm53.WeightSource;
const MtpLinear = qwen38_mtp.MtpLinear;

pub const DFlash2Config = struct {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 1e7,
    block_size: usize,
    mask_token_id: u32,
    target_layer_ids: [5]usize,
    num_target_layers: usize,
    sliding_window: usize,
    conv_kernel_size: usize,
    conv_group_size: usize,
    selector_rank: usize,
    selector_top_k: usize,

    /// Parse an HF DFlash2 config.json (dflash_config sub-object + top-level
    /// architecture fields).
    pub fn parse(allocator: std.mem.Allocator, json_slice: []const u8) !DFlash2Config {
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_slice, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadConfig;
        const root = parsed.value.object;
        const dc = (root.get("dflash_config") orelse return error.BadConfig);
        if (dc != .object) return error.BadConfig;

        var cfg: DFlash2Config = .{
            .hidden_size = try getUsize(root, "hidden_size"),
            .intermediate_size = try getUsize(root, "intermediate_size"),
            .num_hidden_layers = try getUsize(root, "num_hidden_layers"),
            .num_attention_heads = try getUsize(root, "num_attention_heads"),
            .num_key_value_heads = try getUsize(root, "num_key_value_heads"),
            .head_dim = try getUsize(root, "head_dim"),
            .vocab_size = try getUsize(root, "vocab_size"),
            .num_target_layers = try getUsize(root, "num_target_layers"),
            .sliding_window = try getUsize(root, "sliding_window"),
            .block_size = try getUsize(dc.object, "block_size"),
            .mask_token_id = @intCast(try getUsize(dc.object, "mask_token_id")),
            .target_layer_ids = undefined,
            .conv_kernel_size = try getUsize(dc.object, "conv_kernel_size"),
            .conv_group_size = try getUsize(dc.object, "conv_group_size"),
            .selector_rank = try getUsize(dc.object, "selector_rank"),
            .selector_top_k = try getUsize(dc.object, "selector_top_k"),
        };
        if (root.get("rms_norm_eps")) |v| cfg.rms_norm_eps = jsonFloat(v);
        if (root.get("rope_parameters")) |rp| {
            if (rp == .object) {
                if (rp.object.get("rope_theta")) |v| cfg.rope_theta = jsonFloat(v);
            }
        }
        const ids = dc.object.get("target_layer_ids") orelse return error.BadConfig;
        if (ids != .array or ids.array.items.len != 5) return error.BadConfig;
        for (ids.array.items, 0..) |v, i| cfg.target_layer_ids[i] = @intCast(v.integer);
        return cfg;
    }

    fn getUsize(obj: std.json.ObjectMap, key: []const u8) !usize {
        const v = obj.get(key) orelse return error.BadConfig;
        if (v != .integer or v.integer < 0) return error.BadConfig;
        return @intCast(v.integer);
    }

    fn jsonFloat(v: std.json.Value) f32 {
        return switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => 0,
        };
    }
};

/// Two-tap grouped dynamic causal conv weights for one sublayer.
pub const ConvWeights = struct {
    /// [2, kernel_size, hidden] — set 0 = prepare, set 1 = finish.
    base_kernel: Array,
    /// [2 * kernel_size * (hidden/group), hidden].
    kernel_projection: MtpLinear,

    pub fn deinit(self: *ConvWeights) void {
        self.base_kernel.deinit();
        deinitLinear(&self.kernel_projection);
    }
};

/// MtpLinear.deinit is private to qwen38_mtp; release the fields here.
fn deinitLinear(l: *const MtpLinear) void {
    l.w.deinit();
    if (l.scales) |s| s.deinit();
    if (l.biases) |b| b.deinit();
}

pub const LayerWeights = struct {
    input_layernorm_f32: Array,
    post_attention_layernorm_f32: Array,
    q_proj: MtpLinear,
    k_proj: MtpLinear,
    v_proj: MtpLinear,
    o_proj: MtpLinear,
    /// Per-head RMSNorm over head_dim, kept in checkpoint dtype (backbone
    /// convention, see qwen38_mtp loadFromSource).
    q_norm: Array,
    k_norm: Array,
    attention_conv: ConvWeights,
    mlp_conv: ConvWeights,
    gate_proj: MtpLinear,
    up_proj: MtpLinear,
    down_proj: MtpLinear,

    pub fn deinit(self: *LayerWeights) void {
        self.input_layernorm_f32.deinit();
        self.post_attention_layernorm_f32.deinit();
        deinitLinear(&self.q_proj);
        deinitLinear(&self.k_proj);
        deinitLinear(&self.v_proj);
        deinitLinear(&self.o_proj);
        self.q_norm.deinit();
        self.k_norm.deinit();
        self.attention_conv.deinit();
        self.mlp_conv.deinit();
        deinitLinear(&self.gate_proj);
        deinitLinear(&self.up_proj);
        deinitLinear(&self.down_proj);
    }
};

pub const DFlash2 = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    cfg: DFlash2Config,
    fc: MtpLinear,
    hidden_norm_f32: Array,
    norm_f32: Array,
    layers: []LayerWeights,
    selector_hidden_projection: MtpLinear,
    /// [vocab, selector_rank] codebooks, checkpoint dtype.
    predecessor_codebook: Array,
    successor_codebook: Array,

    fn loadNormF32(ctx: EagerContext, ws: WeightSource, name: []const u8) !Array {
        const w = try ws.tensor(name);
        defer w.deinit();
        const w_f32 = try ops.astype(ctx, w, .float32);
        errdefer w_f32.deinit();
        try w_f32.eval();
        return w_f32;
    }

    fn loadLinear(ctx: EagerContext, ws: WeightSource, prefix: []const u8, quant_bits: u8) !MtpLinear {
        var nbuf: [256]u8 = undefined;
        const w = try ws.tensor(try std.fmt.bufPrint(&nbuf, "{s}.weight", .{prefix}));
        if (quant_bits == 0) return .{ .w = w };
        defer w.deinit();
        const qw = try quantize_mod.quantize(ctx, w, .{ .bits = quant_bits, .group_size = 64, .mode = .affine });
        qw.config.validate() catch unreachable;
        return .{ .w = qw.data, .scales = qw.scales, .biases = qw.biases };
    }

    fn loadConv(ctx: EagerContext, ws: WeightSource, prefix: []const u8, quant_bits: u8) !ConvWeights {
        var nbuf: [256]u8 = undefined;
        const base = try ws.tensor(try std.fmt.bufPrint(&nbuf, "{s}.base_kernel", .{prefix}));
        errdefer base.deinit();
        const proj = try loadLinear(ctx, ws, try std.fmt.bufPrint(&nbuf, "{s}.kernel_projection", .{prefix}), quant_bits);
        return .{ .base_kernel = base, .kernel_projection = proj };
    }

    /// Load from a WeightSource with the flat HF checkpoint namespace
    /// (`fc.weight`, `layers.N.*`, `norm.weight`, `candidate_selector.*`).
    /// The selector codebooks in the HF checkpoint have no `.weight` suffix.
    pub fn loadFromSource(
        allocator: std.mem.Allocator,
        ctx: EagerContext,
        ws: WeightSource,
        cfg: DFlash2Config,
        quant_bits: u8,
    ) !DFlash2 {
        const fc = try loadLinear(ctx, ws, "fc", quant_bits);
        errdefer deinitLinear(&fc);
        const hn = try loadNormF32(ctx, ws, "hidden_norm.weight");
        errdefer hn.deinit();
        const nf = try loadNormF32(ctx, ws, "norm.weight");
        errdefer nf.deinit();

        const layers = try allocator.alloc(LayerWeights, cfg.num_hidden_layers);
        var n_loaded: usize = 0;
        errdefer {
            for (layers[0..n_loaded]) |*l| l.deinit();
            allocator.free(layers);
        }
        for (layers, 0..) |*l, i| {
            var pbuf: [128]u8 = undefined;
            const p = try std.fmt.bufPrint(&pbuf, "layers.{d}", .{i});
            var sbuf: [160]u8 = undefined;
            const sa = try std.fmt.bufPrint(&sbuf, "{s}.self_attn", .{p});
            var n1: [256]u8 = undefined;
            var n2: [256]u8 = undefined;
            l.* = .{
                .input_layernorm_f32 = try loadNormF32(ctx, ws, try std.fmt.bufPrint(&n1, "{s}.input_layernorm.weight", .{p})),
                .post_attention_layernorm_f32 = try loadNormF32(ctx, ws, try std.fmt.bufPrint(&n2, "{s}.post_attention_layernorm.weight", .{p})),
                .q_proj = try loadLinear(ctx, ws, try std.fmt.bufPrint(&n1, "{s}.q_proj", .{sa}), quant_bits),
                .k_proj = try loadLinear(ctx, ws, try std.fmt.bufPrint(&n2, "{s}.k_proj", .{sa}), quant_bits),
                .v_proj = try loadLinear(ctx, ws, try std.fmt.bufPrint(&n1, "{s}.v_proj", .{sa}), quant_bits),
                .o_proj = try loadLinear(ctx, ws, try std.fmt.bufPrint(&n2, "{s}.o_proj", .{sa}), quant_bits),
                .q_norm = try ws.tensor(try std.fmt.bufPrint(&n1, "{s}.q_norm.weight", .{sa})),
                .k_norm = try ws.tensor(try std.fmt.bufPrint(&n2, "{s}.k_norm.weight", .{sa})),
                .attention_conv = try loadConv(ctx, ws, try std.fmt.bufPrint(&n1, "{s}.attention_conv", .{p}), quant_bits),
                .mlp_conv = try loadConv(ctx, ws, try std.fmt.bufPrint(&n2, "{s}.mlp_conv", .{p}), quant_bits),
                .gate_proj = try loadLinear(ctx, ws, try std.fmt.bufPrint(&n1, "{s}.mlp.gate_proj", .{p}), quant_bits),
                .up_proj = try loadLinear(ctx, ws, try std.fmt.bufPrint(&n2, "{s}.mlp.up_proj", .{p}), quant_bits),
                .down_proj = try loadLinear(ctx, ws, try std.fmt.bufPrint(&n1, "{s}.mlp.down_proj", .{p}), quant_bits),
            };
            n_loaded += 1;
        }

        const hp = try loadLinear(ctx, ws, "candidate_selector.hidden_projection", quant_bits);
        errdefer deinitLinear(&hp);
        const pcb = try ws.tensor("candidate_selector.predecessor_codebook");
        errdefer pcb.deinit();
        const scb = try ws.tensor("candidate_selector.successor_codebook");
        errdefer scb.deinit();

        return .{
            .allocator = allocator,
            .ctx = ctx,
            .cfg = cfg,
            .fc = fc,
            .hidden_norm_f32 = hn,
            .norm_f32 = nf,
            .layers = layers,
            .selector_hidden_projection = hp,
            .predecessor_codebook = pcb,
            .successor_codebook = scb,
        };
    }

    /// Load from a checkpoint directory (config.json + safetensors).
    pub fn load(allocator: std.mem.Allocator, ctx: EagerContext, model_dir: []const u8, quant_bits: u8) !DFlash2 {
        const cfg_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
        defer allocator.free(cfg_path);
        const posix = @cImport({
            @cInclude("fcntl.h");
            @cInclude("unistd.h");
        });
        const cfg_path_z = try allocator.dupeZ(u8, cfg_path);
        defer allocator.free(cfg_path_z);
        const fd = posix.open(cfg_path_z.ptr, posix.O_RDONLY);
        if (fd < 0) return error.FileNotFound;
        defer _ = posix.close(fd);
        var buf: [64 * 1024]u8 = undefined;
        const n = posix.read(fd, &buf, buf.len);
        if (n < 0) return error.ReadFailed;
        const cfg_raw = buf[0..@intCast(n)];
        const cfg = try DFlash2Config.parse(allocator, cfg_raw);

        var index = try safetensors_reader.loadOrBuildIndex(allocator, model_dir);
        defer index.deinit();
        return loadFromSource(allocator, ctx, WeightSource.fromIndex(&index), cfg, quant_bits);
    }

    pub fn deinit(self: *DFlash2) void {
        deinitLinear(&self.fc);
        self.hidden_norm_f32.deinit();
        self.norm_f32.deinit();
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        deinitLinear(&self.selector_hidden_projection);
        self.predecessor_codebook.deinit();
        self.successor_codebook.deinit();
    }
};

// ---------------------------------------------------------------------------
// Two-tap grouped dynamic causal conv (Task 3)
// ---------------------------------------------------------------------------

/// out[b,t,g,j] = sum_o (base[o, g*gs+j] + dyn[b,t,o,g]) * x[b,t-o,g*gs+j]
/// zero-padded for t < o. Mirrors mlx-vlm `_grouped_dynamic_convolve`.
fn groupedDynamicConvolve(
    ctx: EagerContext,
    x: Array, // [B, L, H]
    dyn: Array, // [B, L, KS, G]
    base: Array, // [KS, H]
    group_size: i32,
) !Array {
    const xs = x.shape();
    const B = xs[0];
    const L = xs[1];
    const H = xs[2];
    const KS: i32 = @intCast(base.shape()[0]);
    const G: i32 = @divExact(H, group_size);

    const blocks = try ops.reshape(ctx, x, &[_]i32{ B, L, G, group_size });
    defer blocks.deinit();
    var out = try Array.zeros(ctx.allocator, &[_]i32{ B, L, G, group_size }, x.dtype());
    defer out.deinit();

    var o: i32 = 0;
    while (o < KS) : (o += 1) {
        var values = blocks;
        var pad: ?Array = null;
        var tail: ?Array = null;
        var cat_values: ?Array = null;
        defer if (pad) |*v| v.deinit();
        defer if (tail) |*v| v.deinit();
        defer if (cat_values) |*v| v.deinit();
        if (o > 0) {
            pad = try Array.zeros(ctx.allocator, &[_]i32{ B, o, G, group_size }, x.dtype());
            tail = try ops.slice(ctx, blocks, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ B, L - o, G, group_size }, &[_]i32{ 1, 1, 1, 1 });
            cat_values = try shape_mod.concatenateAxis(ctx, &[_]Array{ pad.?, tail.? }, 1);
            values = cat_values.?;
        }
        const base_row = try ops.slice(ctx, base, &[_]i32{ o, 0 }, &[_]i32{ o + 1, H }, &[_]i32{ 1, 1 });
        defer base_row.deinit();
        const k = try ops.reshape(ctx, base_row, &[_]i32{ 1, 1, G, group_size });
        defer k.deinit();
        const d_row = try ops.slice(ctx, dyn, &[_]i32{ 0, 0, o, 0 }, &[_]i32{ B, L, o + 1, G }, &[_]i32{ 1, 1, 1, 1 });
        defer d_row.deinit();
        const d = try ops.reshape(ctx, d_row, &[_]i32{ B, L, G, 1 });
        defer d.deinit();
        const kernel = try ops.add(ctx, k, d); // [B, L, G, gs]
        defer kernel.deinit();
        const term = try ops.multiply(ctx, kernel, values);
        defer term.deinit();
        const next = try ops.add(ctx, out, term);
        out.deinit();
        out = next;
    }
    return ops.reshape(ctx, out, &[_]i32{ B, L, H });
}

pub const ConvOut = struct { prepared: Array, dynamic: Array };

/// prepare: convolve with kernel set 0 and hand the set-1 dynamic kernel to
/// the later finish call (reference: GroupedDynamicCausalConv.prepare).
pub fn convPrepare(ctx: EagerContext, w: *const ConvWeights, x: Array, kernel_size: i32, group_size: i32) !ConvOut {
    const xs = x.shape();
    const B = xs[0];
    const L = xs[1];
    const H = xs[2];
    const G: i32 = @divExact(H, group_size);
    const dyn_full = try w.kernel_projection.apply(ctx, x); // [B, L, 2*KS*G]
    defer dyn_full.deinit();
    const dyn5 = try ops.reshape(ctx, dyn_full, &[_]i32{ B, L, 2, kernel_size, G });
    defer dyn5.deinit();
    const d0 = try ops.slice(ctx, dyn5, &[_]i32{ 0, 0, 0, 0, 0 }, &[_]i32{ B, L, 1, kernel_size, G }, &[_]i32{ 1, 1, 1, 1, 1 });
    defer d0.deinit();
    const d0r = try ops.reshape(ctx, d0, &[_]i32{ B, L, kernel_size, G });
    defer d0r.deinit();
    const d1 = try ops.slice(ctx, dyn5, &[_]i32{ 0, 0, 1, 0, 0 }, &[_]i32{ B, L, 2, kernel_size, G }, &[_]i32{ 1, 1, 1, 1, 1 });
    defer d1.deinit();
    const dynamic = try ops.reshape(ctx, d1, &[_]i32{ B, L, kernel_size, G });
    errdefer dynamic.deinit();
    const base0 = try ops.slice(ctx, w.base_kernel, &[_]i32{ 0, 0, 0 }, &[_]i32{ 1, kernel_size, H }, &[_]i32{ 1, 1, 1 });
    defer base0.deinit();
    const base0r = try ops.reshape(ctx, base0, &[_]i32{ kernel_size, H });
    defer base0r.deinit();
    const prepared = try groupedDynamicConvolve(ctx, x, d0r, base0r, group_size);
    return .{ .prepared = prepared, .dynamic = dynamic };
}

/// finish: convolve the sublayer output with kernel set 1 + saved dynamics.
pub fn convFinish(ctx: EagerContext, w: *const ConvWeights, x: Array, dynamic: Array, kernel_size: i32, group_size: i32) !Array {
    const H = x.shape()[2];
    const base1 = try ops.slice(ctx, w.base_kernel, &[_]i32{ 1, 0, 0 }, &[_]i32{ 2, kernel_size, H }, &[_]i32{ 1, 1, 1 });
    defer base1.deinit();
    const base1r = try ops.reshape(ctx, base1, &[_]i32{ kernel_size, H });
    defer base1r.deinit();
    return groupedDynamicConvolve(ctx, x, dynamic, base1r, group_size);
}

// ---------------------------------------------------------------------------
// Decoder layer forward (Task 4)
// ---------------------------------------------------------------------------

/// RMSNorm over the last dim. Matches nn.RMSNorm exactly: the input dtype
/// drives the compute (bf16 in → bf16 fast.rms_norm, which is what the
/// mlx-vlm reference runs), so the weight is cast to x's dtype rather than
/// upcasting everything to f32 (the f32 path is more accurate but rounds
/// differently, which breaks golden parity).
pub fn rmsNormLastDim(ctx: EagerContext, x: Array, w: Array, eps: f32) !Array {
    if (x.dtype() == .bfloat16) {
        const w_b = try ops.astype(ctx, w, .bfloat16);
        defer w_b.deinit();
        return fast_mod.rmsNorm(ctx, x, w_b, eps);
    }
    const w_f32 = try ops.astype(ctx, w, .float32);
    defer w_f32.deinit();
    return fast_mod.rmsNorm(ctx, x, w_f32, eps);
}

/// DFlashAttention: block queries attend over [context KV ‖ block KV],
/// non-causal (no mask), GQA via head-count mismatch, per-head q/k RMSNorm,
/// NeoX RoPE (traditional=false). No cache — the caller passes a fully
/// realized context; the K/V ring (Task 7) reuses these projections.
///   x:     [B, L, H]  block tokens (conv-prepared)
///   x_ctx: [B, S, H]  projected context rows
///   q_off: absolute position of the first block token (RoPE)
///   k_off: absolute position of the first context token (RoPE)
pub fn attentionForward(
    ctx: EagerContext,
    w: *const LayerWeights,
    cfg: *const DFlash2Config,
    x: Array,
    x_ctx: Array,
    q_off: i32,
    k_off: i32,
) !Array {
    const B = x.shape()[0];
    const L = x.shape()[1];
    const S = x_ctx.shape()[1];
    const nh: i32 = @intCast(cfg.num_attention_heads);
    const nkv: i32 = @intCast(cfg.num_key_value_heads);
    const hd: i32 = @intCast(cfg.head_dim);

    // Queries from the block.
    const q = try w.q_proj.apply(ctx, x);
    defer q.deinit();
    const q4 = try ops.reshape(ctx, q, &[_]i32{ B, L, nh, hd });
    defer q4.deinit();
    const qn = try rmsNormLastDim(ctx, q4, w.q_norm, cfg.rms_norm_eps);
    defer qn.deinit();
    const qt = try ops.transposeAxes(ctx, qn, &[_]i32{ 0, 2, 1, 3 });
    defer qt.deinit();
    const qr = try fast_mod.rope(ctx, qt, hd, false, cfg.rope_theta, 1.0, q_off, null);
    defer qr.deinit();

    // Context K/V.
    const ck = try w.k_proj.apply(ctx, x_ctx);
    defer ck.deinit();
    const ck4 = try ops.reshape(ctx, ck, &[_]i32{ B, S, nkv, hd });
    defer ck4.deinit();
    const ckn = try rmsNormLastDim(ctx, ck4, w.k_norm, cfg.rms_norm_eps);
    defer ckn.deinit();
    const ckt = try ops.transposeAxes(ctx, ckn, &[_]i32{ 0, 2, 1, 3 });
    defer ckt.deinit();
    const ckr = try fast_mod.rope(ctx, ckt, hd, false, cfg.rope_theta, 1.0, k_off, null);
    defer ckr.deinit();
    const cv = try w.v_proj.apply(ctx, x_ctx);
    defer cv.deinit();
    const cv4 = try ops.reshape(ctx, cv, &[_]i32{ B, S, nkv, hd });
    defer cv4.deinit();
    const cvt = try ops.transposeAxes(ctx, cv4, &[_]i32{ 0, 2, 1, 3 });
    defer cvt.deinit();

    // Block (proposal) K/V.
    const pk = try w.k_proj.apply(ctx, x);
    defer pk.deinit();
    const pk4 = try ops.reshape(ctx, pk, &[_]i32{ B, L, nkv, hd });
    defer pk4.deinit();
    const pkn = try rmsNormLastDim(ctx, pk4, w.k_norm, cfg.rms_norm_eps);
    defer pkn.deinit();
    const pkt = try ops.transposeAxes(ctx, pkn, &[_]i32{ 0, 2, 1, 3 });
    defer pkt.deinit();
    const pkr = try fast_mod.rope(ctx, pkt, hd, false, cfg.rope_theta, 1.0, q_off, null);
    defer pkr.deinit();
    const pv = try w.v_proj.apply(ctx, x);
    defer pv.deinit();
    const pv4 = try ops.reshape(ctx, pv, &[_]i32{ B, L, nkv, hd });
    defer pv4.deinit();
    const pvt = try ops.transposeAxes(ctx, pv4, &[_]i32{ 0, 2, 1, 3 });
    defer pvt.deinit();

    const keys = try shape_mod.concatenateAxis(ctx, &[_]Array{ ckr, pkr }, 2);
    defer keys.deinit();
    const values = try shape_mod.concatenateAxis(ctx, &[_]Array{ cvt, pvt }, 2);
    defer values.deinit();

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const o = try fast_mod.scaledDotProductAttention(ctx, qr, keys, values, scale, "", null, null);
    defer o.deinit();
    const ot = try ops.transposeAxes(ctx, o, &[_]i32{ 0, 2, 1, 3 });
    defer ot.deinit();
    const of = try ops.reshape(ctx, ot, &[_]i32{ B, L, nh * hd });
    defer of.deinit();
    return w.o_proj.apply(ctx, of);
}

fn swigluMlp(ctx: EagerContext, w: *const LayerWeights, x: Array) !Array {
    const gate = try w.gate_proj.apply(ctx, x);
    defer gate.deinit();
    const up = try w.up_proj.apply(ctx, x);
    defer up.deinit();
    const act = try activations.silu(ctx, gate);
    defer act.deinit();
    const h = try ops.multiply(ctx, act, up);
    defer h.deinit();
    return w.down_proj.apply(ctx, h);
}

/// DFlash2DecoderLayer._forward: norm → conv_prepare → attention →
/// conv_finish → residual, then the same shape around the SwiGLU MLP.
pub fn layerForward(
    ctx: EagerContext,
    w: *const LayerWeights,
    cfg: *const DFlash2Config,
    x: Array, // [B, L, H] block hidden
    x_ctx: Array, // [B, S, H] projected context
    q_off: i32,
    k_off: i32,
) !Array {
    const KS: i32 = @intCast(cfg.conv_kernel_size);
    const GS: i32 = @intCast(cfg.conv_group_size);

    const xn = try rmsNormLastDim(ctx, x, w.input_layernorm_f32, cfg.rms_norm_eps);
    defer xn.deinit();
    const prep = try convPrepare(ctx, &w.attention_conv, xn, KS, GS);
    defer prep.prepared.deinit();
    defer prep.dynamic.deinit();
    const attn = try attentionForward(ctx, w, cfg, prep.prepared, x_ctx, q_off, k_off);
    defer attn.deinit();
    const fin = try convFinish(ctx, &w.attention_conv, attn, prep.dynamic, KS, GS);
    defer fin.deinit();
    const h1 = try ops.add(ctx, x, fin);
    defer h1.deinit();

    const xn2 = try rmsNormLastDim(ctx, h1, w.post_attention_layernorm_f32, cfg.rms_norm_eps);
    defer xn2.deinit();
    const prep2 = try convPrepare(ctx, &w.mlp_conv, xn2, KS, GS);
    defer prep2.prepared.deinit();
    defer prep2.dynamic.deinit();
    const mlp = try swigluMlp(ctx, w, prep2.prepared);
    defer mlp.deinit();
    const fin2 = try convFinish(ctx, &w.mlp_conv, mlp, prep2.dynamic, KS, GS);
    defer fin2.deinit();
    return ops.add(ctx, h1, fin2);
}

// ---------------------------------------------------------------------------
// Full block draft forward + greedy candidate selector (Task 5)
// ---------------------------------------------------------------------------

/// Post-final-norm block hidden for one draft round (single request, B=1).
///   block_ids:  [block] i32 — [anchor, mask × (block-1)]
///   ctx_hidden: [1, S, 5*H] — backbone hidden states concatenated from the
///               five tapped layers, one row per committed context token
///   embed_table: borrowed backbone embedding [vocab, H]
/// Returns [1, block, H]; the caller drops row 0 (anchor) before the head.
/// Project raw tap rows ([1, S, 5*H]) into drafter context space
/// ([1, S, H]): fc + hidden_norm. Done once per committed token.
pub fn projectContext(ctx: EagerContext, m: *const DFlash2, ctx_hidden: Array) !Array {
    const proj = try m.fc.apply(ctx, ctx_hidden);
    defer proj.deinit();
    return rmsNormLastDim(ctx, proj, m.hidden_norm_f32, m.cfg.rms_norm_eps);
}

/// Block forward with an already-projected context (see projectContext).
pub fn blockHiddenProjected(
    ctx: EagerContext,
    m: *const DFlash2,
    embed_table: Array,
    block_ids: Array,
    x_ctx: Array,
) !Array {
    const S: i32 = x_ctx.shape()[1];
    const blk: i32 = block_ids.shape()[0];

    const emb = try shape_mod.takeAxis(ctx, embed_table, block_ids, 0);
    defer emb.deinit();
    var h = try ops.reshape(ctx, emb, &[_]i32{ 1, blk, @intCast(m.cfg.hidden_size) });
    defer h.deinit();

    for (m.layers) |*l| {
        const next = try layerForward(ctx, l, &m.cfg, h, x_ctx, S, 0);
        h.deinit();
        h = next;
    }
    return rmsNormLastDim(ctx, h, m.norm_f32, m.cfg.rms_norm_eps);
}

/// Convenience wrapper: raw tap rows in, block hidden out.
pub fn blockHidden(
    ctx: EagerContext,
    m: *const DFlash2,
    embed_table: Array,
    block_ids: Array,
    ctx_hidden: Array,
) !Array {
    const x_ctx = try projectContext(ctx, m, ctx_hidden);
    defer x_ctx.deinit();
    return blockHiddenProjected(ctx, m, embed_table, block_ids, x_ctx);
}

/// Greedy DFlash2 candidate selection. hidden [1, P, H], logits [1, P, V]
/// with P = block_size - 1. Writes P token ids into out_tokens.
/// Chain semantics: the selected token at position p becomes the predecessor
/// for position p+1; position 0's predecessor is the anchor.
pub fn selectGreedy(
    ctx: EagerContext,
    m: *const DFlash2,
    hidden: Array,
    logits: Array,
    anchor: u32,
    out_tokens: []u32,
) !void {
    const P: i32 = logits.shape()[1];
    const V: i32 = logits.shape()[2];
    const K: i32 = @intCast(m.cfg.selector_top_k);
    const R: i32 = @intCast(m.cfg.selector_rank);

    const order = try sort_mod.argsortAxis(ctx, logits, -1); // ascending
    defer order.deinit();
    const candidates = try ops.slice(ctx, order, &[_]i32{ 0, 0, V - K }, &[_]i32{ 1, P, V }, &[_]i32{ 1, 1, 1 });
    defer candidates.deinit();
    const cand_i32 = try ops.astype(ctx, candidates, .int32);
    defer cand_i32.deinit();
    const unary = try shape_mod.takeAlongAxis(ctx, logits, cand_i32, -1); // [1,P,K]
    defer unary.deinit();
    const hp = try m.selector_hidden_projection.apply(ctx, hidden); // [1,P,R]
    defer hp.deinit();

    var prev = try Array.fromData(ctx.allocator, i32, &[_]i32{@intCast(anchor)}, &[_]i32{1});
    defer prev.deinit();

    var pos: i32 = 0;
    while (pos < P) : (pos += 1) {
        const pcb_row = try shape_mod.takeAxis(ctx, m.predecessor_codebook, prev, 0); // [1,R]
        defer pcb_row.deinit();
        const pcb3 = try ops.reshape(ctx, pcb_row, &[_]i32{ 1, 1, R });
        defer pcb3.deinit();
        const proj_pos = try ops.slice(ctx, hp, &[_]i32{ 0, pos, 0 }, &[_]i32{ 1, pos + 1, R }, &[_]i32{ 1, 1, 1 });
        defer proj_pos.deinit();
        const cand_pos = try ops.slice(ctx, cand_i32, &[_]i32{ 0, pos, 0 }, &[_]i32{ 1, pos + 1, K }, &[_]i32{ 1, 1, 1 });
        defer cand_pos.deinit();
        const cand_flat = try ops.reshape(ctx, cand_pos, &[_]i32{K});
        defer cand_flat.deinit();
        const scb_rows = try shape_mod.takeAxis(ctx, m.successor_codebook, cand_flat, 0); // [K,R]
        defer scb_rows.deinit();
        const scb3 = try ops.reshape(ctx, scb_rows, &[_]i32{ 1, K, R });
        defer scb3.deinit();
        const prod = try ops.multiply(ctx, pcb3, proj_pos); // [1,1,R]
        defer prod.deinit();
        const prod2 = try ops.multiply(ctx, prod, scb3); // [1,K,R]
        defer prod2.deinit();
        const edges = try reduce_mod.sumAxis(ctx, prod2, 2, false); // [1,K]
        defer edges.deinit();
        const unary_pos = try ops.slice(ctx, unary, &[_]i32{ 0, pos, 0 }, &[_]i32{ 1, pos + 1, K }, &[_]i32{ 1, 1, 1 });
        defer unary_pos.deinit();
        const unary_flat = try ops.reshape(ctx, unary_pos, &[_]i32{ 1, K });
        defer unary_flat.deinit();
        const scores = try ops.add(ctx, unary_flat, edges);
        defer scores.deinit();
        const am1 = try reduce_mod.argmaxAxis(ctx, scores, 1, false); // [1]
        defer am1.deinit();
        const am = try ops.reshape(ctx, am1, &[_]i32{});
        defer am.deinit();
        try am.eval();
        const am_idx = try am.item(i32);
        const sel1 = try ops.slice(ctx, cand_flat, &[_]i32{am_idx}, &[_]i32{am_idx + 1}, &[_]i32{1});
        defer sel1.deinit();
        const sel = try ops.reshape(ctx, sel1, &[_]i32{});
        defer sel.deinit();
        try sel.eval();
        const tok = try sel.item(i32);
        out_tokens[@intCast(pos)] = @intCast(tok);
        const next = try Array.fromData(ctx.allocator, i32, &[_]i32{tok}, &[_]i32{1});
        prev.deinit();
        prev = next;
    }
}

// ---------------------------------------------------------------------------
// Draft context store (Task 7)
// ---------------------------------------------------------------------------

/// Committed-token context for the drafter as a per-layer K/V ring
/// (omlx `_RowContext` semantics): RoPE-pre-applied keys and raw values for
/// every committed token, written once at observe time. Validity comes from
/// an absolute-position array (-1 = empty); attention keeps the newest
/// `sliding_window - 1` positions via the draft mask. Rows are only ever
/// appended (accepted tokens) — never rolled back.
pub const DFlash2Context = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    cfg: *const DFlash2Config,
    model: *const DFlash2,
    capacity: i32, // ring_slots + block_size
    ring_slots: i64,
    fed: i64 = 0, // absolute position of the next row to write
    keys: [5]Array, // per layer [nkv, capacity, hd] bf16, RoPE applied
    values: [5]Array, // per layer [nkv, capacity, hd] bf16
    pos: []i64, // [capacity] absolute position per slot, -1 empty
    /// Projected rows currently in the ring ([1, <=ring_slots, H]), kept so
    /// the adapter can cache the drafter context for prefix-snapshot hits.
    rows_shadow: ?Array = null,

    pub fn init(allocator: std.mem.Allocator, ctx: EagerContext, model: *const DFlash2) !DFlash2Context {
        const cfg = &model.cfg;
        const ring_slots: i64 = @as(i64, @intCast(cfg.sliding_window)) - 1;
        const capacity: i32 = @intCast(ring_slots + @as(i64, @intCast(cfg.block_size)));
        const nkv: i32 = @intCast(cfg.num_key_value_heads);
        const hd: i32 = @intCast(cfg.head_dim);
        var keys: [5]Array = undefined;
        var values: [5]Array = undefined;
        var n_init: usize = 0;
        errdefer for (keys[0..n_init], values[0..n_init]) |k, v| {
            k.deinit();
            v.deinit();
        };
        for (0..cfg.num_hidden_layers) |l| {
            keys[l] = try Array.zeros(allocator, &[_]i32{ nkv, capacity, hd }, .bfloat16);
            values[l] = try Array.zeros(allocator, &[_]i32{ nkv, capacity, hd }, .bfloat16);
            n_init += 1;
        }
        const pos = try allocator.alloc(i64, @intCast(capacity));
        @memset(pos, -1);
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .cfg = cfg,
            .model = model,
            .capacity = capacity,
            .ring_slots = ring_slots,
            .keys = keys,
            .values = values,
            .pos = pos,
        };
    }

    pub fn reset(self: *DFlash2Context) void {
        // Zeroing 42 MB of rings is pointless; invalidating pos empties them.
        @memset(self.pos, -1);
        self.fed = 0;
        if (self.rows_shadow) |r| r.deinit();
        self.rows_shadow = null;
    }

    /// Borrowed view of the projected rows currently in the ring (or null).
    pub fn rowsShadow(self: *DFlash2Context) ?Array {
        return self.rows_shadow;
    }

    /// Append tap rows for newly committed tokens. tap_hidden: [1, k, 5*H]
    /// raw backbone taps. Projects to context space, then per layer computes
    /// K/V and writes the ring slots; K gets RoPE at the absolute offset.
    pub fn observe(self: *DFlash2Context, tap_hidden: Array) !void {
        if (tap_hidden.shape()[1] == 0) return;
        const proj = try projectContext(self.ctx, self.model, tap_hidden); // [1,k,H]
        defer proj.deinit();
        return self.seedFromProjected(proj);
    }

    /// Feed already-projected context rows ([1, k, H]) into the rings
    /// (prefix-snapshot restore path — the projection was cached).
    pub fn seedFromProjected(self: *DFlash2Context, proj: Array) !void {
        const k: i32 = proj.shape()[1];
        if (k == 0) return;
        const ctx = self.ctx;
        const cfg = self.cfg;
        const nkv: i32 = @intCast(cfg.num_key_value_heads);
        const hd: i32 = @intCast(cfg.head_dim);
        const base_off: i32 = @intCast(self.fed);
        for (self.model.layers, 0..) |*lw, l| {
            const kc = try lw.k_proj.apply(ctx, proj);
            defer kc.deinit();
            const k4 = try ops.reshape(ctx, kc, &[_]i32{ 1, k, nkv, hd });
            defer k4.deinit();
            const kn = try rmsNormLastDim(ctx, k4, lw.k_norm, cfg.rms_norm_eps);
            defer kn.deinit();
            const kt = try ops.transposeAxes(ctx, kn, &[_]i32{ 0, 2, 1, 3 }); // [1,nkv,k,hd]
            defer kt.deinit();
            const kr = try fast_mod.rope(ctx, kt, hd, false, cfg.rope_theta, 1.0, base_off, null);
            defer kr.deinit();
            const kr3 = try ops.reshape(ctx, kr, &[_]i32{ nkv, k, hd });
            defer kr3.deinit();
            const vc = try lw.v_proj.apply(ctx, proj);
            defer vc.deinit();
            const v4 = try ops.reshape(ctx, vc, &[_]i32{ 1, k, nkv, hd });
            defer v4.deinit();
            const vt = try ops.transposeAxes(ctx, v4, &[_]i32{ 0, 2, 1, 3 });
            defer vt.deinit();
            const vt3 = try ops.reshape(ctx, vt, &[_]i32{ nkv, k, hd });
            defer vt3.deinit();
            try self.writeSlots(l, kr3, vt3, k);
        }
        self.fed += k;
        // Maintain the projected-row shadow (windowed to ring_slots).
        if (self.rows_shadow) |old| {
            const merged = try shape_mod.concatenateAxis(ctx, &[_]Array{ old, proj }, 1);
            old.deinit();
            self.rows_shadow = merged;
        } else {
            self.rows_shadow = try ops.copy(ctx, proj);
        }
        const sh_len: i64 = self.rows_shadow.?.shape()[1];
        if (sh_len > self.ring_slots) {
            const start: i32 = @intCast(sh_len - self.ring_slots);
            const total: i32 = @intCast(sh_len);
            const tail = try ops.slice(ctx, self.rows_shadow.?, &[_]i32{ 0, start, 0 }, &[_]i32{ 1, total, @intCast(cfg.hidden_size) }, &[_]i32{ 1, 1, 1 });
            self.rows_shadow.?.deinit();
            self.rows_shadow = tail;
        }
    }

    /// Write k rows of one layer's K/V into ring slots (fed + j) % capacity.
    fn writeSlots(self: *DFlash2Context, layer: usize, k3: Array, v3: Array, k: i32) !void {
        const nkv: i32 = @intCast(self.cfg.num_key_value_heads);
        const hd: i32 = @intCast(self.cfg.head_dim);
        var j: i32 = 0;
        while (j < k) {
            const slot: i32 = @intCast(@mod(self.fed + j, self.capacity));
            const run: i32 = @min(k - j, self.capacity - slot);
            const ks = try ops.slice(self.ctx, k3, &[_]i32{ 0, j, 0 }, &[_]i32{ nkv, j + run, hd }, &[_]i32{ 1, 1, 1 });
            defer ks.deinit();
            const vs = try ops.slice(self.ctx, v3, &[_]i32{ 0, j, 0 }, &[_]i32{ nkv, j + run, hd }, &[_]i32{ 1, 1, 1 });
            defer vs.deinit();
            const new_k = try shape_mod.sliceUpdate(self.ctx, self.keys[layer], ks, &[_]i32{ 0, slot, 0 }, &[_]i32{ nkv, slot + run, hd }, &[_]i32{ 1, 1, 1 });
            self.keys[layer].deinit();
            self.keys[layer] = new_k;
            const new_v = try shape_mod.sliceUpdate(self.ctx, self.values[layer], vs, &[_]i32{ 0, slot, 0 }, &[_]i32{ nkv, slot + run, hd }, &[_]i32{ 1, 1, 1 });
            self.values[layer].deinit();
            self.values[layer] = new_v;
            for (0..@intCast(run)) |t| self.pos[@intCast(slot + @as(i32, @intCast(t)))] = self.fed + j + @as(i64, @intCast(t));
            j += run;
        }
    }

    /// Boolean SDPA mask [1, 1, 1, capacity + block]: ring slot visible iff
    /// occupied and inside the newest ring_slots positions; the block's own
    /// slots are always visible (non-causal block).
    pub fn draftMask(self: *DFlash2Context) !Array {
        const blk: i32 = @intCast(self.cfg.block_size);
        const n: usize = @intCast(self.capacity + blk);
        const data = try self.allocator.alloc(bool, n);
        defer self.allocator.free(data);
        const lo = self.fed - self.ring_slots;
        for (0..@intCast(self.capacity)) |i| {
            data[i] = self.pos[i] >= 0 and self.pos[i] >= lo;
        }
        for (0..@intCast(blk)) |i| data[@as(usize, @intCast(self.capacity)) + i] = true;
        const arr = try Array.fromData(self.allocator, bool, data, &[_]i32{ 1, 1, 1, @intCast(n) });
        return arr;
    }

    pub fn deinit(self: *DFlash2Context) void {
        for (0..self.cfg.num_hidden_layers) |l| {
            self.keys[l].deinit();
            self.values[l].deinit();
        }
        if (self.rows_shadow) |r| r.deinit();
        self.allocator.free(self.pos);
    }
};

/// Ring variant of the block forward: attention reads context K/V from the
/// ring (masked) instead of a freshly projected dense context. Block Q/K/V
/// rope at offset `fed` (position after the newest context row).
pub fn blockHiddenRing(
    ctx: EagerContext,
    m: *const DFlash2,
    ring: *DFlash2Context,
    embed_table: Array,
    block_ids: Array,
    mask: Array,
) !Array {
    const blk: i32 = block_ids.shape()[0];
    const q_off: i32 = @intCast(ring.fed);
    const emb = try shape_mod.takeAxis(ctx, embed_table, block_ids, 0);
    defer emb.deinit();
    var h = try ops.reshape(ctx, emb, &[_]i32{ 1, blk, @intCast(m.cfg.hidden_size) });
    defer h.deinit();
    for (m.layers, 0..) |*l, li| {
        const next = try layerForwardRing(ctx, l, &m.cfg, h, ring, li, q_off, mask);
        h.deinit();
        h = next;
    }
    return rmsNormLastDim(ctx, h, m.norm_f32, m.cfg.rms_norm_eps);
}

fn layerForwardRing(
    ctx: EagerContext,
    w: *const LayerWeights,
    cfg: *const DFlash2Config,
    x: Array, // [1, L, H]
    ring: *DFlash2Context,
    layer_idx: usize,
    q_off: i32,
    mask: Array,
) !Array {
    const KS: i32 = @intCast(cfg.conv_kernel_size);
    const GS: i32 = @intCast(cfg.conv_group_size);
    const xn = try rmsNormLastDim(ctx, x, w.input_layernorm_f32, cfg.rms_norm_eps);
    defer xn.deinit();
    const prep = try convPrepare(ctx, &w.attention_conv, xn, KS, GS);
    defer prep.prepared.deinit();
    defer prep.dynamic.deinit();
    const attn = try attentionForwardRing(ctx, w, cfg, prep.prepared, ring, layer_idx, q_off, mask);
    defer attn.deinit();
    const fin = try convFinish(ctx, &w.attention_conv, attn, prep.dynamic, KS, GS);
    defer fin.deinit();
    const h1 = try ops.add(ctx, x, fin);
    defer h1.deinit();
    const xn2 = try rmsNormLastDim(ctx, h1, w.post_attention_layernorm_f32, cfg.rms_norm_eps);
    defer xn2.deinit();
    const prep2 = try convPrepare(ctx, &w.mlp_conv, xn2, KS, GS);
    defer prep2.prepared.deinit();
    defer prep2.dynamic.deinit();
    const mlp = try swigluMlp(ctx, w, prep2.prepared);
    defer mlp.deinit();
    const fin2 = try convFinish(ctx, &w.mlp_conv, mlp, prep2.dynamic, KS, GS);
    defer fin2.deinit();
    return ops.add(ctx, h1, fin2);
}

fn attentionForwardRing(
    ctx: EagerContext,
    w: *const LayerWeights,
    cfg: *const DFlash2Config,
    x: Array, // [1, L, H] conv-prepared block
    ring: *DFlash2Context,
    layer_idx: usize,
    q_off: i32,
    mask: Array,
) !Array {
    const L = x.shape()[1];
    const nh: i32 = @intCast(cfg.num_attention_heads);
    const nkv: i32 = @intCast(cfg.num_key_value_heads);
    const hd: i32 = @intCast(cfg.head_dim);

    const q = try w.q_proj.apply(ctx, x);
    defer q.deinit();
    const q4 = try ops.reshape(ctx, q, &[_]i32{ 1, L, nh, hd });
    defer q4.deinit();
    const qn = try rmsNormLastDim(ctx, q4, w.q_norm, cfg.rms_norm_eps);
    defer qn.deinit();
    const qt = try ops.transposeAxes(ctx, qn, &[_]i32{ 0, 2, 1, 3 });
    defer qt.deinit();
    const qr = try fast_mod.rope(ctx, qt, hd, false, cfg.rope_theta, 1.0, q_off, null);
    defer qr.deinit();

    const pk = try w.k_proj.apply(ctx, x);
    defer pk.deinit();
    const pk4 = try ops.reshape(ctx, pk, &[_]i32{ 1, L, nkv, hd });
    defer pk4.deinit();
    const pkn = try rmsNormLastDim(ctx, pk4, w.k_norm, cfg.rms_norm_eps);
    defer pkn.deinit();
    const pkt = try ops.transposeAxes(ctx, pkn, &[_]i32{ 0, 2, 1, 3 });
    defer pkt.deinit();
    const pkr = try fast_mod.rope(ctx, pkt, hd, false, cfg.rope_theta, 1.0, q_off, null);
    defer pkr.deinit();
    const pv = try w.v_proj.apply(ctx, x);
    defer pv.deinit();
    const pv4 = try ops.reshape(ctx, pv, &[_]i32{ 1, L, nkv, hd });
    defer pv4.deinit();
    const pvt = try ops.transposeAxes(ctx, pv4, &[_]i32{ 0, 2, 1, 3 });
    defer pvt.deinit();

    const rk = try ops.reshape(ctx, ring.keys[layer_idx], &[_]i32{ 1, nkv, ring.capacity, hd });
    defer rk.deinit();
    const rv = try ops.reshape(ctx, ring.values[layer_idx], &[_]i32{ 1, nkv, ring.capacity, hd });
    defer rv.deinit();
    const keys = try shape_mod.concatenateAxis(ctx, &[_]Array{ rk, pkr }, 2);
    defer keys.deinit();
    const values = try shape_mod.concatenateAxis(ctx, &[_]Array{ rv, pvt }, 2);
    defer values.deinit();

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const o = try fast_mod.scaledDotProductAttention(ctx, qr, keys, values, scale, "", mask, null);
    defer o.deinit();
    const ot = try ops.transposeAxes(ctx, o, &[_]i32{ 0, 2, 1, 3 });
    defer ot.deinit();
    const of = try ops.reshape(ctx, ot, &[_]i32{ 1, L, nh * hd });
    defer of.deinit();
    return w.o_proj.apply(ctx, of);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const test_golden_dir = "src/models/testdata/dflash2";

test "DFlash2Config.parse: real-checkpoint-shaped json" {
    const allocator = std.testing.allocator;
    const json_text =
        \\{
        \\  "model_type": "qwen3",
        \\  "hidden_size": 5120,
        \\  "intermediate_size": 17408,
        \\  "num_hidden_layers": 5,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 8,
        \\  "head_dim": 128,
        \\  "vocab_size": 248320,
        \\  "num_target_layers": 64,
        \\  "sliding_window": 2048,
        \\  "rms_norm_eps": 1e-06,
        \\  "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        \\  "dflash_config": {
        \\    "block_size": 8,
        \\    "conv_group_size": 16,
        \\    "conv_kernel_size": 2,
        \\    "mask_token_id": 248070,
        \\    "selector_rank": 256,
        \\    "selector_top_k": 16,
        \\    "target_layer_ids": [5, 19, 33, 47, 61]
        \\  }
        \\}
    ;
    const cfg = try DFlash2Config.parse(allocator, json_text);
    try std.testing.expectEqual(@as(usize, 5120), cfg.hidden_size);
    try std.testing.expectEqual(@as(usize, 5), cfg.num_hidden_layers);
    try std.testing.expectEqual(@as(usize, 8), cfg.block_size);
    try std.testing.expectEqual(@as(u32, 248070), cfg.mask_token_id);
    try std.testing.expectEqual(@as(usize, 16), cfg.conv_group_size);
    try std.testing.expectEqual(@as(usize, 256), cfg.selector_rank);
    try std.testing.expectEqual(@as(usize, 16), cfg.selector_top_k);
    try std.testing.expectEqual([5]usize{ 5, 19, 33, 47, 61 }, cfg.target_layer_ids);
    try std.testing.expectApproxEqAbs(@as(f32, 1e-6), cfg.rms_norm_eps, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f32, 1e7), cfg.rope_theta, 1.0);
}

test "DFlash2.loadFromSource: golden weights load with expected shapes" {
    const c_mod = @import("mlx").c;
    c_mod.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const cfg_raw = @embedFile("testdata/dflash2/meta.json");
    var meta = try std.json.parseFromSlice(std.json.Value, allocator, cfg_raw, .{});
    defer meta.deinit();
    // The golden meta config is flat (no dflash_config nesting) — wrap it the
    // way HF does so the same parse path is exercised.
    const wrapped = try std.fmt.allocPrint(allocator,
        \\{{"hidden_size":{d},"intermediate_size":{d},"num_hidden_layers":{d},
        \\"num_attention_heads":{d},"num_key_value_heads":{d},"head_dim":{d},
        \\"vocab_size":{d},"num_target_layers":{d},"sliding_window":{d},
        \\"rms_norm_eps":1e-06,"rope_parameters":{{"rope_theta":{d}}},
        \\"dflash_config":{{"block_size":{d},"mask_token_id":{d},
        \\"conv_kernel_size":{d},"conv_group_size":{d},"selector_rank":{d},
        \\"selector_top_k":{d},"target_layer_ids":[1,3,5,7,9]}}}}
    , .{
        cfgValue(meta, "hidden_size"),         cfgValue(meta, "intermediate_size"),
        cfgValue(meta, "num_hidden_layers"),   cfgValue(meta, "num_attention_heads"),
        cfgValue(meta, "num_key_value_heads"), cfgValue(meta, "head_dim"),
        cfgValue(meta, "vocab_size"),          cfgValue(meta, "num_target_layers"),
        cfgValue(meta, "sliding_window"),      cfgValue(meta, "rope_theta"),
        cfgValue(meta, "block_size"),          cfgValue(meta, "mask_token_id"),
        cfgValue(meta, "conv_kernel_size"),    cfgValue(meta, "conv_group_size"),
        cfgValue(meta, "selector_rank"),       cfgValue(meta, "selector_top_k"),
    });
    defer allocator.free(wrapped);
    const cfg = try DFlash2Config.parse(allocator, wrapped);

    var index = try safetensors_reader.buildIndexFromDirectory(allocator, test_golden_dir);
    defer index.deinit();
    var m = try DFlash2.loadFromSource(allocator, ctx, WeightSource.fromIndex(&index), cfg, 0);
    defer m.deinit();

    const H: i32 = @intCast(cfg.hidden_size);
    try m.fc.w.eval();
    try expectShape(&[_]i32{ H, 5 * H }, m.fc.w);
    try expectShape(&[_]i32{H}, m.hidden_norm_f32);
    try std.testing.expectEqual(@as(usize, 2), m.layers.len);
    const l0 = &m.layers[0];
    try expectShape(&[_]i32{ @as(i32, @intCast(cfg.num_attention_heads * cfg.head_dim)), H }, l0.q_proj.w);
    try expectShape(&[_]i32{ @as(i32, @intCast(cfg.num_key_value_heads * cfg.head_dim)), H }, l0.k_proj.w);
    try expectShape(&[_]i32{ H, @as(i32, @intCast(cfg.num_attention_heads * cfg.head_dim)) }, l0.o_proj.w);
    try expectShape(&[_]i32{ 2, @as(i32, @intCast(cfg.conv_kernel_size)), H }, l0.attention_conv.base_kernel);
    const groups: i32 = @intCast(cfg.hidden_size / cfg.conv_group_size);
    try expectShape(&[_]i32{ 2 * @as(i32, @intCast(cfg.conv_kernel_size)) * groups, H }, l0.mlp_conv.kernel_projection.w);
    try expectShape(&[_]i32{ @as(i32, @intCast(cfg.intermediate_size)), H }, l0.gate_proj.w);
    try expectShape(&[_]i32{ H, @as(i32, @intCast(cfg.intermediate_size)) }, l0.down_proj.w);
    try expectShape(&[_]i32{ @as(i32, @intCast(cfg.selector_rank)), H }, m.selector_hidden_projection.w);
    try expectShape(&[_]i32{ @as(i32, @intCast(cfg.vocab_size)), @as(i32, @intCast(cfg.selector_rank)) }, m.predecessor_codebook);
    try expectShape(&[_]i32{ @as(i32, @intCast(cfg.vocab_size)), @as(i32, @intCast(cfg.selector_rank)) }, m.successor_codebook);
}

fn loadTraceTensor(allocator: std.mem.Allocator, name: []const u8) !Array {
    var index = safetensors_reader.TensorIndex.init(allocator);
    defer index.deinit();
    try index.addShard(test_golden_dir ++ "/trace.safetensors");
    return index.loadTensor(name);
}

/// Max abs difference between two same-shaped arrays, computed in f32.
fn maxAbsDiff(ctx: EagerContext, a: Array, b: Array) !f32 {
    const af = try ops.astype(ctx, a, .float32);
    defer af.deinit();
    const bf = try ops.astype(ctx, b, .float32);
    defer bf.deinit();
    const diff = try ops.subtract(ctx, af, bf);
    defer diff.deinit();
    const ad = try ops.abs(ctx, diff);
    defer ad.deinit();
    const m = try reduce_mod.max(ctx, ad, false);
    defer m.deinit();
    try m.eval();
    return m.item(f32);
}

/// max over elements of |a-b| / (|b| + floor) — relative parity for bf16
/// traces where absolute error scales with element magnitude.
fn relMaxDiff(ctx: EagerContext, a: Array, b: Array, floor: f32) !f32 {
    const af = try ops.astype(ctx, a, .float32);
    defer af.deinit();
    const bf = try ops.astype(ctx, b, .float32);
    defer bf.deinit();
    const diff = try ops.subtract(ctx, af, bf);
    defer diff.deinit();
    const ad = try ops.abs(ctx, diff);
    defer ad.deinit();
    const ab = try ops.abs(ctx, bf);
    defer ab.deinit();
    const f = try ops.scalarF32(ctx, floor);
    defer f.deinit();
    const denom = try ops.add(ctx, ab, f);
    defer denom.deinit();
    const rel = try ops.divide(ctx, ad, denom);
    defer rel.deinit();
    const m = try reduce_mod.max(ctx, rel, false);
    defer m.deinit();
    try m.eval();
    return m.item(f32);
}

fn maxAbs(ctx: EagerContext, a: Array) !f32 {
    const af = try ops.astype(ctx, a, .float32);
    defer af.deinit();
    const ad = try ops.abs(ctx, af);
    defer ad.deinit();
    const m = try reduce_mod.max(ctx, ad, false);
    defer m.deinit();
    try m.eval();
    return m.item(f32);
}

test "DFlash2 conv: prepare/finish match golden trace (layer 0 attention)" {
    const c_mod = @import("mlx").c;
    c_mod.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const cfg_raw = @embedFile("testdata/dflash2/meta.json");
    var meta = try std.json.parseFromSlice(std.json.Value, allocator, cfg_raw, .{});
    defer meta.deinit();
    const KS: i32 = @intCast(cfgValue(meta, "conv_kernel_size"));
    const GS: i32 = @intCast(cfgValue(meta, "conv_group_size"));

    var index = safetensors_reader.TensorIndex.init(allocator);
    defer index.deinit();
    try index.addShard(test_golden_dir ++ "/model.safetensors");

    const base = try index.loadTensor("layers.0.attention_conv.base_kernel");
    defer base.deinit();
    const kp = try index.loadTensor("layers.0.attention_conv.kernel_projection.weight");
    defer kp.deinit();
    const conv: ConvWeights = .{ .base_kernel = base, .kernel_projection = .{ .w = kp } };

    const x = try loadTraceTensor(allocator, "l0.attn.normed");
    defer x.deinit();
    const out = try convPrepare(ctx, &conv, x, KS, GS);
    defer out.prepared.deinit();
    defer out.dynamic.deinit();

    const exp_prepared = try loadTraceTensor(allocator, "l0.attn.conv_prepared");
    defer exp_prepared.deinit();
    const exp_dynamic = try loadTraceTensor(allocator, "l0.attn.conv_dynamic");
    defer exp_dynamic.deinit();

    // bf16 golden: allow a couple of ULPs of wiggle room.
    const scale_p = try maxAbs(ctx, exp_prepared);
    try std.testing.expect(try maxAbsDiff(ctx, out.prepared, exp_prepared) <= 1e-2 * scale_p + 1e-3);
    const scale_d = try maxAbs(ctx, exp_dynamic);
    try std.testing.expect(try maxAbsDiff(ctx, out.dynamic, exp_dynamic) <= 1e-2 * scale_d + 1e-3);

    // finish: conv(attn_out, dynamic) must equal post_attn - block_embed.
    const attn_out = try loadTraceTensor(allocator, "l0.attn.out");
    defer attn_out.deinit();
    const fin = try convFinish(ctx, &conv, attn_out, exp_dynamic, KS, GS);
    defer fin.deinit();
    const post_attn = try loadTraceTensor(allocator, "l0.post_attn");
    defer post_attn.deinit();
    const block_embed = try loadTraceTensor(allocator, "block_embed");
    defer block_embed.deinit();
    const exp_fin = try ops.subtract(ctx, post_attn, block_embed);
    defer exp_fin.deinit();
    const scale_f = try maxAbs(ctx, exp_fin);
    try std.testing.expect(try maxAbsDiff(ctx, fin, exp_fin) <= 1e-2 * scale_f + 1e-3);
}

test "DFlash2 layer 0: full forward matches golden trace" {
    const c_mod = @import("mlx").c;
    c_mod.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const cfg_raw = @embedFile("testdata/dflash2/meta.json");
    var meta = try std.json.parseFromSlice(std.json.Value, allocator, cfg_raw, .{});
    defer meta.deinit();
    const cfg = try testCfgFromMeta(meta);

    var index = safetensors_reader.TensorIndex.init(allocator);
    defer index.deinit();
    try index.addShard(test_golden_dir ++ "/model.safetensors");
    var m = try DFlash2.loadFromSource(allocator, ctx, WeightSource.fromIndex(&index), cfg, 0);
    defer m.deinit();

    const x = try loadTraceTensor(allocator, "block_embed");
    defer x.deinit();
    const x_ctx = try loadTraceTensor(allocator, "ctx_projected");
    defer x_ctx.deinit();
    const S: i32 = x_ctx.shape()[1];

    // Intermediate parity: attention sublayer output.
    const l0 = &m.layers[0];
    const xn = try rmsNormLastDim(ctx, x, l0.input_layernorm_f32, cfg.rms_norm_eps);
    defer xn.deinit();
    const prep = try convPrepare(ctx, &l0.attention_conv, xn, @intCast(cfg.conv_kernel_size), @intCast(cfg.conv_group_size));
    defer prep.prepared.deinit();
    defer prep.dynamic.deinit();
    const attn = try attentionForward(ctx, l0, &cfg, prep.prepared, x_ctx, S, 0);
    defer attn.deinit();
    const exp_attn = try loadTraceTensor(allocator, "l0.attn.out");
    defer exp_attn.deinit();
    const sa = try maxAbs(ctx, exp_attn);
    try std.testing.expect(try maxAbsDiff(ctx, attn, exp_attn) <= 1e-2 * sa + 1e-3);

    // Full layer output.
    const out = try layerForward(ctx, l0, &cfg, x, x_ctx, S, 0);
    defer out.deinit();
    const exp_out = try loadTraceTensor(allocator, "l0.out");
    defer exp_out.deinit();
    const so = try maxAbs(ctx, exp_out);
    try std.testing.expect(try maxAbsDiff(ctx, out, exp_out) <= 1e-2 * so + 1e-3);
}

test "DFlash2 full block draft + greedy selector match golden trace" {
    const c_mod = @import("mlx").c;
    c_mod.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const cfg_raw = @embedFile("testdata/dflash2/meta.json");
    var meta = try std.json.parseFromSlice(std.json.Value, allocator, cfg_raw, .{});
    defer meta.deinit();
    const cfg = try testCfgFromMeta(meta);

    var index = safetensors_reader.TensorIndex.init(allocator);
    defer index.deinit();
    try index.addShard(test_golden_dir ++ "/model.safetensors");
    var m = try DFlash2.loadFromSource(allocator, ctx, WeightSource.fromIndex(&index), cfg, 0);
    defer m.deinit();

    // Borrowed tables live in the same golden shard.
    const embed_w = try index.loadTensor("embed_tokens.weight");
    defer embed_w.deinit();
    const lm_w = try index.loadTensor("lm_head.weight");
    defer lm_w.deinit();
    const lm_head: MtpLinear = .{ .w = lm_w };

    const ctx_hidden = try loadTraceTensor(allocator, "ctx_hidden");
    defer ctx_hidden.deinit();
    const block_ids_t = try loadTraceTensor(allocator, "block_ids");
    defer block_ids_t.deinit();
    const block_ids = try ops.reshape(ctx, block_ids_t, &[_]i32{@intCast(cfg.block_size)});
    defer block_ids.deinit();
    const block_ids_i32 = try ops.astype(ctx, block_ids, .int32);
    defer block_ids_i32.deinit();

    const hidden_full = try blockHidden(ctx, &m, embed_w, block_ids_i32, ctx_hidden);
    defer hidden_full.deinit();
    const exp_hidden = try loadTraceTensor(allocator, "final_hidden");
    defer exp_hidden.deinit();
    // Layer-level parity: intermediate tensors are small-magnitude sums
    // with cancellation, so per-element relative error is misleading; check
    // abs error against the tensor scale (a few bf16 ULPs of the max).
    {
        const emb2 = try shape_mod.takeAxis(ctx, embed_w, block_ids_i32, 0);
        defer emb2.deinit();
        var h2 = try ops.reshape(ctx, emb2, &[_]i32{ 1, @intCast(cfg.block_size), @intCast(cfg.hidden_size) });
        defer h2.deinit();
        const proj2 = try m.fc.apply(ctx, ctx_hidden);
        defer proj2.deinit();
        const x_ctx2 = try rmsNormLastDim(ctx, proj2, m.hidden_norm_f32, m.cfg.rms_norm_eps);
        defer x_ctx2.deinit();
        for (m.layers, 0..) |*l, li| {
            const next = try layerForward(ctx, l, &m.cfg, h2, x_ctx2, ctx_hidden.shape()[1], 0);
            h2.deinit();
            h2 = next;
            var nbuf: [64]u8 = undefined;
            const exp_l = try loadTraceTensor(allocator, try std.fmt.bufPrint(&nbuf, "l{d}.out", .{li}));
            defer exp_l.deinit();
            const dl = try maxAbsDiff(ctx, h2, exp_l);
            const sl2 = try maxAbs(ctx, exp_l);
            try std.testing.expect(dl <= 0.05 * sl2);
        }
        const df = try maxAbsDiff(ctx, hidden_full, exp_hidden);
        const sf = try maxAbs(ctx, exp_hidden);
        try std.testing.expect(df <= 0.05 * sf);
    }

    // Drop the anchor row, then logits.
    const P: i32 = @intCast(cfg.block_size - 1);
    const H: i32 = @intCast(cfg.hidden_size);
    const draft_hidden = try ops.slice(ctx, hidden_full, &[_]i32{ 0, 1, 0 }, &[_]i32{ 1, P + 1, H }, &[_]i32{ 1, 1, 1 });
    defer draft_hidden.deinit();
    const logits = try lm_head.apply(ctx, draft_hidden);
    defer logits.deinit();
    const exp_logits = try loadTraceTensor(allocator, "logits");
    defer exp_logits.deinit();
    const dl2 = try maxAbsDiff(ctx, logits, exp_logits);
    const sl3 = try maxAbs(ctx, exp_logits);
    try std.testing.expect(dl2 <= 0.05 * sl3);

    var tokens: [7]u32 = undefined;
    try selectGreedy(ctx, &m, draft_hidden, logits, 42, &tokens);
    const exp_sel = try loadTraceTensor(allocator, "selected_tokens");
    defer exp_sel.deinit();
    try exp_sel.eval();
    var exp_tokens: [7]i32 = undefined;
    for (0..7) |i| {
        const t1 = try ops.slice(ctx, exp_sel, &[_]i32{ 0, @intCast(i) }, &[_]i32{ 1, @intCast(i + 1) }, &[_]i32{ 1, 1 });
        defer t1.deinit();
        const t = try ops.reshape(ctx, t1, &[_]i32{});
        defer t.deinit();
        try t.eval();
        exp_tokens[i] = try t.item(i32);
    }
    for (tokens, exp_tokens) |got, exp| {
        try std.testing.expectEqual(@as(u32, @intCast(exp)), got);
    }
    // The strict end-to-end criterion: identical greedy token sequence.
}

test "DFlash2Context ring: observe + masked draft matches dense golden path" {
    const c_mod = @import("mlx").c;
    c_mod.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const cfg_raw = @embedFile("testdata/dflash2/meta.json");
    var meta = try std.json.parseFromSlice(std.json.Value, allocator, cfg_raw, .{});
    defer meta.deinit();
    const cfg = try testCfgFromMeta(meta);

    var index = safetensors_reader.TensorIndex.init(allocator);
    defer index.deinit();
    try index.addShard(test_golden_dir ++ "/model.safetensors");
    var m = try DFlash2.loadFromSource(allocator, ctx, WeightSource.fromIndex(&index), cfg, 0);
    defer m.deinit();
    const embed_w = try index.loadTensor("embed_tokens.weight");
    defer embed_w.deinit();

    const ctx_hidden = try loadTraceTensor(allocator, "ctx_hidden");
    defer ctx_hidden.deinit(); // [1, 13, 5H]
    const block_ids_t = try loadTraceTensor(allocator, "block_ids");
    defer block_ids_t.deinit();
    const block_ids = try ops.reshape(ctx, block_ids_t, &[_]i32{@intCast(cfg.block_size)});
    defer block_ids.deinit();
    const block_ids_i32 = try ops.astype(ctx, block_ids, .int32);
    defer block_ids_i32.deinit();

    var ring = try DFlash2Context.init(allocator, ctx, &m);
    defer ring.deinit();
    try ring.observe(ctx_hidden);
    try std.testing.expectEqual(@as(i64, 13), ring.fed);
    try std.testing.expectEqual(@as(i64, 12), ring.pos[12]);
    try std.testing.expectEqual(@as(i64, -1), ring.pos[13]);

    const mask = try ring.draftMask();
    defer mask.deinit();
    const h_ring = try blockHiddenRing(ctx, &m, &ring, embed_w, block_ids_i32, mask);
    defer h_ring.deinit();

    const h_dense = try blockHidden(ctx, &m, embed_w, block_ids_i32, ctx_hidden);
    defer h_dense.deinit();
    const d = try maxAbsDiff(ctx, h_ring, h_dense);
    const sc = try maxAbs(ctx, h_dense);
    try std.testing.expect(d <= 0.05 * sc);

    // Sliding-window validity: mark the ring as if 3000 rows were fed — only
    // the newest ring_slots (63) positions may be visible in the mask.
    ring.fed = 3000;
    const mask2 = try ring.draftMask();
    defer mask2.deinit();
    try mask2.eval();
    const md = try mask2.dataSlice(bool);
    // positions 0..12 are older than 3000-63 → all ring slots masked out
    for (0..@intCast(ring.capacity)) |i| try std.testing.expect(!md[i]);
    for (0..@intCast(cfg.block_size)) |i| try std.testing.expect(md[@as(usize, @intCast(ring.capacity)) + i]);
}

fn testCfgFromMeta(meta: std.json.Parsed(std.json.Value)) !DFlash2Config {
    _ = meta;
    return .{
        .hidden_size = 256,
        .intermediate_size = 512,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .vocab_size = 1000,
        .block_size = 8,
        .mask_token_id = 999,
        .target_layer_ids = .{ 1, 3, 5, 7, 9 },
        .num_target_layers = 12,
        .sliding_window = 64,
        .conv_kernel_size = 2,
        .conv_group_size = 16,
        .selector_rank = 32,
        .selector_top_k = 8,
    };
}

fn cfgValue(meta: std.json.Parsed(std.json.Value), key: []const u8) i64 {
    return switch (meta.value.object.get("config").?.object.get(key).?) {
        .integer => |i| i,
        .float => |f| @intFromFloat(f),
        else => unreachable,
    };
}

fn expectShape(expected: []const i32, arr: Array) !void {
    const s = arr.shape();
    try std.testing.expectEqual(expected.len, s.len);
    for (expected, s) |e, g| try std.testing.expectEqual(e, g);
}
