/// GLM-5.3 oracle alignment harness (Zig side).
///
/// Loads the prep'd checkpoint (~/models/GLM-5.3-Flash-MLX-mixed-4_8bit-dmlx),
/// runs the fixed 7-token fixture through embed + layers 0-2 (KDA attn + dense
/// MLP) + layer-3 NoPE MLA with the mHC apply_branch semantics from mlx-vlm's
/// HyperConnection, and dumps each segment to artifacts/glm53_oracle/{seg}.bin
/// for comparison against the Python oracle fixtures
/// (tests/fixtures/glm53_oracle/*.npy) via scripts/compare_glm53_oracle.py.
///
/// bin format: [ndim: i32][dims: i32 x ndim][f32 row-major payload].
///
/// Usage:
///   zig build run-glm53-oracle-dump -- [--model-dir DIR] [--out-dir DIR] [--seg SEG]
/// SEG ∈ embed | layer0_kda_out | layer3_mla_out | layer3_indexer_topk | all (default all).
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const quantize_mod = @import("mlx").quantize;
const fast_mod = @import("mlx").fast;
const safetensors_reader = @import("mlx").safetensors_reader;
const glm53_loader = @import("models/glm53_loader.zig");
const glm53_kda = @import("models/glm53_kda.zig");
const glm53_attn = @import("models/glm53_attention.zig");
const dsv4 = @import("models/deepseek_v4.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

const token_ids = [_]i32{ 17, 42, 314, 2718, 7, 99, 1234 };

const QuantParts = struct {
    data: Array,
    scales: Array,
    biases: Array,
    cfg: quantize_mod.QuantConfig,

    fn deinit(self: *QuantParts) void {
        self.data.deinit();
        self.scales.deinit();
        self.biases.deinit();
    }

    fn linear(self: *const QuantParts) glm53_attn.QuantLinear {
        return .{ .w = self.data, .scales = self.scales, .biases = self.biases, .cfg = self.cfg };
    }
};

fn loadTensor(index: *safetensors_reader.TensorIndex, name: []const u8) !Array {
    return index.loadTensor(name) catch |err| {
        std.log.err("tensor not found or unreadable: {s} ({})", .{ name, err });
        return err;
    };
}

/// Load an affine-quantized Linear triple, deriving bits/group_size from the
/// packed shapes (same derivation as scripts/glm53_oracle_dump.py).
fn loadQuantized(index: *safetensors_reader.TensorIndex, allocator: std.mem.Allocator, prefix: []const u8, in_features: i32) !QuantParts {
    const w_name = try std.fmt.allocPrint(allocator, "{s}.weight", .{prefix});
    const s_name = try std.fmt.allocPrint(allocator, "{s}.scales", .{prefix});
    const b_name = try std.fmt.allocPrint(allocator, "{s}.biases", .{prefix});

    const data = try loadTensor(index, w_name);
    errdefer data.deinit();
    const scales = try loadTensor(index, s_name);
    errdefer scales.deinit();
    const biases = try loadTensor(index, b_name);
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

fn dumpBin(allocator: std.mem.Allocator, out_dir: []const u8, name: []const u8, arr: Array) !void {
    std.debug.assert(arr.dtype() == .float32);
    try arr.eval();
    const sh = arr.shape();
    std.debug.assert(sh.len <= 8);
    const data = try arr.dataSlice(f32);

    var dir_buf: [1024]u8 = undefined;
    const dir_z = try std.fmt.bufPrintZ(&dir_buf, "{s}", .{out_dir});
    // mkdir -p: create each path component, ignoring EEXIST.
    {
        var i: usize = 0;
        while (i < dir_z.len) : (i += 1) {
            if (dir_z[i] == '/' and i > 0) {
                dir_z[i] = 0;
                _ = std.c.mkdir(dir_z[0..i :0].ptr, 0o755);
                dir_z[i] = '/';
            }
        }
        _ = std.c.mkdir(dir_z.ptr, 0o755);
    }

    const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}.bin", .{ out_dir, name }, 0);
    const fp = std.c.fopen(path.ptr, "wb") orelse return error.OpenFailed;
    defer _ = std.c.fclose(fp);

    var header: [9]i32 = undefined;
    const ndim: i32 = @intCast(sh.len);
    header[0] = ndim;
    for (sh, 0..) |d, i| header[1 + i] = d;
    const header_bytes = std.mem.sliceAsBytes(header[0 .. 1 + sh.len]);
    if (std.c.fwrite(header_bytes.ptr, 1, header_bytes.len, fp) != header_bytes.len) return error.WriteFailed;
    const payload = std.mem.sliceAsBytes(data);
    if (payload.len > 0 and std.c.fwrite(payload.ptr, 1, payload.len, fp) != payload.len) return error.WriteFailed;
    std.log.info("dumped {s}.bin ndim={d} n={d}", .{ name, ndim, data.len });
}

fn loadConfig(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !glm53_loader.Glm53Config {
    const path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024));
    defer allocator.free(text);
    return glm53_loader.parseGlm53Config(allocator, text);
}

/// mHC apply_branch pre-phase: compute mixes from the residual, collapse into
/// the branch input, and RMSNorm it. Returned arrays are owned by the caller.
const HcPre = struct {
    normed: Array, // [1, L, hidden]
    post_mix: Array,
    comb_norm: Array,
};

fn hcPre(
    ctx: EagerContext,
    index: *safetensors_reader.TensorIndex,
    allocator: std.mem.Allocator,
    cfg: *const glm53_loader.Glm53Config,
    hc_prefix: []const u8,
    ln_name: []const u8,
    residual: Array,
    stream_raw: c.c.mlx_stream,
) !HcPre {
    const hc_fn = try loadTensor(index, try std.fmt.allocPrint(allocator, "{s}.fn", .{hc_prefix}));
    defer hc_fn.deinit();
    const hc_base = try loadTensor(index, try std.fmt.allocPrint(allocator, "{s}.base", .{hc_prefix}));
    defer hc_base.deinit();
    const hc_scale = try loadTensor(index, try std.fmt.allocPrint(allocator, "{s}.scale", .{hc_prefix}));
    defer hc_scale.deinit();
    const ln_w = try loadTensor(index, ln_name);
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
fn kdaAttnBranch(
    ctx: EagerContext,
    index: *safetensors_reader.TensorIndex,
    allocator: std.mem.Allocator,
    cfg: *const glm53_loader.Glm53Config,
    layer_idx: u32,
    residual: Array,
    stream_raw: c.c.mlx_stream,
) !Array {
    const hidden: i32 = @intCast(cfg.hidden_size);
    const p = try std.fmt.allocPrint(allocator, "language_model.model.layers.{d}", .{layer_idx});

    const pre = try hcPre(ctx, index, allocator, cfg, try std.fmt.allocPrint(allocator, "{s}.attn_hc", .{p}), try std.fmt.allocPrint(allocator, "{s}.input_layernorm.weight", .{p}), residual, stream_raw);
    defer pre.normed.deinit();
    defer pre.post_mix.deinit();
    defer pre.comb_norm.deinit();

    // KDA weights (prep'd fused names; forget_gate.* keeps its prefix).
    const sa = try std.fmt.allocPrint(allocator, "{s}.self_attn", .{p});
    var qkv_q = try loadQuantized(index, allocator, try std.fmt.allocPrint(allocator, "{s}.qkv_proj", .{sa}), hidden);
    defer qkv_q.deinit();
    var fbg_q = try loadQuantized(index, allocator, try std.fmt.allocPrint(allocator, "{s}.fbg_a_proj", .{sa}), hidden);
    defer fbg_q.deinit();
    var o_q = try loadQuantized(index, allocator, try std.fmt.allocPrint(allocator, "{s}.o_proj", .{sa}), @intCast(cfg.linear_num_heads * cfg.linear_head_dim));
    defer o_q.deinit();

    const qkv_conv = try loadTensor(index, try std.fmt.allocPrint(allocator, "{s}.qkv_conv.weight", .{sa}));
    defer qkv_conv.deinit();
    const qkv_conv_f32 = try ops.astype(ctx, qkv_conv, .float32);
    defer qkv_conv_f32.deinit();

    // f_b/g_b are consumed unquantized by GlmKDA — dequantize at load.
    var fb_q = try loadQuantized(index, allocator, try std.fmt.allocPrint(allocator, "{s}.forget_gate.f_b_proj", .{sa}), @intCast(cfg.linear_head_dim));
    defer fb_q.deinit();
    const f_b = try quantize_mod.dequantizeAs(ctx, .{ .data = fb_q.data, .scales = fb_q.scales, .biases = fb_q.biases, .config = fb_q.cfg, .original_shape = &[_]i32{} }, .float32);
    defer f_b.deinit();
    var gb_q = try loadQuantized(index, allocator, try std.fmt.allocPrint(allocator, "{s}.g_b_proj", .{sa}), @intCast(cfg.linear_head_dim));
    defer gb_q.deinit();
    const g_b = try quantize_mod.dequantizeAs(ctx, .{ .data = gb_q.data, .scales = gb_q.scales, .biases = gb_q.biases, .config = gb_q.cfg, .original_shape = &[_]i32{} }, .float32);
    defer g_b.deinit();

    const a_log = try loadTensor(index, try std.fmt.allocPrint(allocator, "{s}.forget_gate.A_log", .{sa}));
    defer a_log.deinit();
    const dt_bias = try loadTensor(index, try std.fmt.allocPrint(allocator, "{s}.forget_gate.dt_bias", .{sa}));
    defer dt_bias.deinit();
    const o_norm_w = try loadTensor(index, try std.fmt.allocPrint(allocator, "{s}.o_norm.weight", .{sa}));
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

    var cache = try glm53_kda.GlmKdaCache.init(allocator, cfg.linear_conv_kernel, 3 * cfg.linear_num_heads * cfg.linear_head_dim, cfg.linear_num_heads, cfg.linear_head_dim);
    defer cache.deinit();

    const kda_out = try kda.prefill(allocator, pre.normed, &cache); // [1, L, hidden] f32
    defer kda_out.deinit();

    // HC expand (apply_branch post): post * out + comb^T @ residual.
    return dsv4.mhcPost(ctx, kda_out, residual, pre.post_mix, pre.comb_norm, stream_raw);
}

/// Dense MLP branch (ffn_hc + post_attention_layernorm + SwiGLU MLP + expand).
fn denseFfnBranch(
    ctx: EagerContext,
    index: *safetensors_reader.TensorIndex,
    allocator: std.mem.Allocator,
    cfg: *const glm53_loader.Glm53Config,
    layer_idx: u32,
    residual: Array,
    stream_raw: c.c.mlx_stream,
) !Array {
    const hidden: i32 = @intCast(cfg.hidden_size);
    const p = try std.fmt.allocPrint(allocator, "language_model.model.layers.{d}", .{layer_idx});

    const pre = try hcPre(ctx, index, allocator, cfg, try std.fmt.allocPrint(allocator, "{s}.ffn_hc", .{p}), try std.fmt.allocPrint(allocator, "{s}.post_attention_layernorm.weight", .{p}), residual, stream_raw);
    defer pre.normed.deinit();
    defer pre.post_mix.deinit();
    defer pre.comb_norm.deinit();

    var gate_up_q = try loadQuantized(index, allocator, try std.fmt.allocPrint(allocator, "{s}.mlp.gate_up_proj", .{p}), hidden);
    defer gate_up_q.deinit();
    var down_q = try loadQuantized(index, allocator, try std.fmt.allocPrint(allocator, "{s}.mlp.down_proj", .{p}), @intCast(cfg.intermediate_size));
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

/// Layer-3 NoPE MLA attention branch. Returns the updated residual (dumped as
/// layer3_mla_out) and the indexer topk (dumped as layer3_indexer_topk).
const MlaBranchResult = struct { residual: Array, topk: Array };

fn mlaAttnBranch(
    ctx: EagerContext,
    index: *safetensors_reader.TensorIndex,
    allocator: std.mem.Allocator,
    cfg: *const glm53_loader.Glm53Config,
    layer_idx: u32,
    residual: Array,
    stream_raw: c.c.mlx_stream,
) !MlaBranchResult {
    const hidden: i32 = @intCast(cfg.hidden_size);
    const p = try std.fmt.allocPrint(allocator, "language_model.model.layers.{d}", .{layer_idx});
    const sa = try std.fmt.allocPrint(allocator, "{s}.self_attn", .{p});

    const pre = try hcPre(ctx, index, allocator, cfg, try std.fmt.allocPrint(allocator, "{s}.attn_hc", .{p}), try std.fmt.allocPrint(allocator, "{s}.input_layernorm.weight", .{p}), residual, stream_raw);
    defer pre.normed.deinit();
    defer pre.post_mix.deinit();
    defer pre.comb_norm.deinit();

    var qkv_a_q = try loadQuantized(index, allocator, try std.fmt.allocPrint(allocator, "{s}.qkv_a_proj", .{sa}), hidden);
    defer qkv_a_q.deinit();
    var q_b_q = try loadQuantized(index, allocator, try std.fmt.allocPrint(allocator, "{s}.q_b_proj", .{sa}), @intCast(cfg.q_lora_rank));
    defer q_b_q.deinit();
    var o_q = try loadQuantized(index, allocator, try std.fmt.allocPrint(allocator, "{s}.o_proj", .{sa}), @intCast(cfg.num_attention_heads * cfg.v_head_dim));
    defer o_q.deinit();

    const q_a_ln = try loadTensor(index, try std.fmt.allocPrint(allocator, "{s}.q_a_layernorm.weight", .{sa}));
    defer q_a_ln.deinit();
    const q_a_ln_f32 = try ops.astype(ctx, q_a_ln, .float32);
    defer q_a_ln_f32.deinit();
    const kv_a_ln = try loadTensor(index, try std.fmt.allocPrint(allocator, "{s}.kv_a_layernorm.weight", .{sa}));
    defer kv_a_ln.deinit();
    const kv_a_ln_f32 = try ops.astype(ctx, kv_a_ln, .float32);
    defer kv_a_ln_f32.deinit();

    // Per-head MultiLinear weights: dequantize once, keep both orientations.
    var embed_q_q = try loadQuantized(index, allocator, try std.fmt.allocPrint(allocator, "{s}.embed_q", .{sa}), @intCast(cfg.qk_nope_head_dim));
    defer embed_q_q.deinit();
    const embed_q_w = try quantize_mod.dequantizeAs(ctx, .{ .data = embed_q_q.data, .scales = embed_q_q.scales, .biases = embed_q_q.biases, .config = embed_q_q.cfg, .original_shape = &[_]i32{} }, .bfloat16);
    defer embed_q_w.deinit(); // [H, kv_lora, qk_nope]
    const embed_q_t = try ops.transposeAxes(ctx, embed_q_w, &[_]i32{ 0, 2, 1 });
    defer embed_q_t.deinit(); // [H, qk_nope, kv_lora]

    var unembed_q = try loadQuantized(index, allocator, try std.fmt.allocPrint(allocator, "{s}.unembed_out", .{sa}), @intCast(cfg.kv_lora_rank));
    defer unembed_q.deinit();
    const unembed_w = try quantize_mod.dequantizeAs(ctx, .{ .data = unembed_q.data, .scales = unembed_q.scales, .biases = unembed_q.biases, .config = unembed_q.cfg, .original_shape = &[_]i32{} }, .bfloat16);
    defer unembed_w.deinit(); // [H, v_head, kv_lora]
    const unembed_out_t = try ops.transposeAxes(ctx, unembed_w, &[_]i32{ 0, 2, 1 });
    defer unembed_out_t.deinit(); // [H, kv_lora, v_head]

    // Indexer weights.
    const ix = try std.fmt.allocPrint(allocator, "{s}.indexer", .{sa});
    var wq_b_q = try loadQuantized(index, allocator, try std.fmt.allocPrint(allocator, "{s}.wq_b", .{ix}), @intCast(cfg.q_lora_rank));
    defer wq_b_q.deinit();
    var wk_q = try loadQuantized(index, allocator, try std.fmt.allocPrint(allocator, "{s}.wk", .{ix}), hidden);
    defer wk_q.deinit();
    var weights_proj_q = try loadQuantized(index, allocator, try std.fmt.allocPrint(allocator, "{s}.weights_proj", .{ix}), hidden);
    defer weights_proj_q.deinit();

    const k_norm_w = try loadTensor(index, try std.fmt.allocPrint(allocator, "{s}.k_norm.weight", .{ix}));
    defer k_norm_w.deinit();
    const k_norm_w_f32 = try ops.astype(ctx, k_norm_w, .float32);
    defer k_norm_w_f32.deinit();
    const k_norm_b = try loadTensor(index, try std.fmt.allocPrint(allocator, "{s}.k_norm.bias", .{ix}));
    defer k_norm_b.deinit();
    const k_norm_b_f32 = try ops.astype(ctx, k_norm_b, .float32);
    defer k_norm_b_f32.deinit();

    const ape = try loadTensor(index, try std.fmt.allocPrint(allocator, "{s}.index_kpool_compress_ape", .{ix}));
    defer ape.deinit();
    const ape_f32 = try ops.astype(ctx, ape, .float32);
    defer ape_f32.deinit();
    const compress_gate = try loadTensor(index, try std.fmt.allocPrint(allocator, "{s}.index_kpool_compress_gate", .{ix}));
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

    const result = try mla.forward(allocator, pre.normed);
    defer result.out.deinit();
    errdefer result.topk.deinit();

    const out = try dsv4.mhcPost(ctx, result.out, residual, pre.post_mix, pre.comb_norm, stream_raw);
    return .{ .residual = out, .topk = result.topk };
}

pub fn main(init: std.process.Init) !void {
    c.initErrorHandler();
    const allocator = init.arena.allocator();
    const io = init.io;

    var model_dir: []const u8 = "";
    var out_dir: []const u8 = "artifacts/glm53_oracle";
    var seg: []const u8 = "all";

    const argv = try init.minimal.args.toSlice(allocator);
    var ai: usize = 1;
    while (ai < argv.len) : (ai += 1) {
        const arg: []const u8 = argv[ai];
        if (std.mem.eql(u8, arg, "--model-dir")) {
            ai += 1;
            model_dir = if (ai < argv.len) argv[ai] else return error.MissingArgValue;
        } else if (std.mem.eql(u8, arg, "--out-dir")) {
            ai += 1;
            out_dir = if (ai < argv.len) argv[ai] else return error.MissingArgValue;
        } else if (std.mem.eql(u8, arg, "--seg")) {
            ai += 1;
            seg = if (ai < argv.len) argv[ai] else return error.MissingArgValue;
        } else return error.UnknownArg;
    }
    if (model_dir.len == 0) {
        const home_z = std.c.getenv("HOME") orelse return error.NoHome;
        const home = std.mem.span(home_z);
        model_dir = try std.fs.path.join(allocator, &.{ home, "models/GLM-5.3-Flash-MLX-mixed-4_8bit-dmlx" });
    }

    const want_embed = std.mem.eql(u8, seg, "all") or std.mem.eql(u8, seg, "embed");
    const want_kda = std.mem.eql(u8, seg, "all") or std.mem.eql(u8, seg, "layer0_kda_out");
    const want_mla = std.mem.eql(u8, seg, "all") or std.mem.eql(u8, seg, "layer3_mla_out");
    const want_topk = std.mem.eql(u8, seg, "all") or std.mem.eql(u8, seg, "layer3_indexer_topk");
    const want_layer3 = want_mla or want_topk;
    if (!want_embed and !want_kda and !want_layer3) return error.UnknownSegment;

    var cfg = try loadConfig(allocator, io, model_dir);
    defer cfg.deinit();

    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });
    // LIFO: declared before every Array/index defer below, so the stream is
    // released only after all arrays and the weight index are freed.
    defer ctx.deinit();
    const stream_raw = ctx.stream.inner;

    var index = try safetensors_reader.loadOrBuildIndex(allocator, model_dir);
    defer index.deinit();

    const hidden: i32 = @intCast(cfg.hidden_size);
    const L: i32 = @intCast(token_ids.len);
    const hc_mult: usize = cfg.hc_mult;

    // === embed segment: 8bit-quantized embed_tokens lookup ===
    var embed_q = try loadQuantized(&index, allocator, "language_model.model.embed_tokens", hidden);
    defer embed_q.deinit();
    const embed_w = try quantize_mod.dequantizeAs(ctx, .{
        .data = embed_q.data,
        .scales = embed_q.scales,
        .biases = embed_q.biases,
        .config = embed_q.cfg,
        .original_shape = &[_]i32{},
    }, .bfloat16);
    defer embed_w.deinit(); // [vocab, hidden] bf16
    try embed_w.eval();

    const ids = try Array.fromData(allocator, i32, &token_ids, &[_]i32{L});
    defer ids.deinit();
    const rows = try shape_mod.takeAxis(ctx, embed_w, ids, 0); // [L, hidden] bf16
    defer rows.deinit();
    const embed = try ops.reshape(ctx, rows, &[_]i32{ 1, L, hidden });
    defer embed.deinit();

    if (want_embed) {
        const embed_f32 = try ops.astype(ctx, embed, .float32);
        defer embed_f32.deinit();
        try dumpBin(allocator, out_dir, "embed", embed_f32);
    }
    if (!want_kda and !want_layer3) return;

    // residual = repeat(embed, hc_mult) on axis 2 — matches the oracle's
    // mx.repeat(h[:, :, None], hc_mult, axis=2).
    var h = try dsv4.expandToMHC(ctx, embed, hc_mult, stream_raw); // [1, L, 4, hidden] bf16
    defer h.deinit();

    // === layers 0-2: KDA attention branch + dense MLP branch ===
    // Hardcoded topology per the GLM-5.3-Flash config; assert so a config
    // change fails loudly instead of silently running the wrong branch kinds.
    std.debug.assert(!cfg.isFullAttention(0) and !cfg.isFullAttention(1) and !cfg.isFullAttention(2));
    std.debug.assert(cfg.isFullAttention(3));
    std.debug.assert(!cfg.isMoE(0) and !cfg.isMoE(1) and !cfg.isMoE(2));
    for (0..3) |li| {
        const h_next = try kdaAttnBranch(ctx, &index, allocator, &cfg, @intCast(li), h, stream_raw);
        h.deinit();
        h = h_next;
        if (li == 0 and want_kda) {
            const out_f32 = try ops.astype(ctx, h, .float32);
            defer out_f32.deinit();
            try dumpBin(allocator, out_dir, "layer0_kda_out", out_f32);
        }
        if (!want_layer3 and li == 0) return;
        const h_ffn = try denseFfnBranch(ctx, &index, allocator, &cfg, @intCast(li), h, stream_raw);
        h.deinit();
        h = h_ffn;
    }
    if (!want_layer3) return;

    // === layer 3: NoPE MLA + DSA indexer attention branch ===
    const mla_result = try mlaAttnBranch(ctx, &index, allocator, &cfg, 3, h, stream_raw);
    defer mla_result.residual.deinit();
    defer mla_result.topk.deinit();

    if (want_mla) {
        const out_f32 = try ops.astype(ctx, mla_result.residual, .float32);
        defer out_f32.deinit();
        try dumpBin(allocator, out_dir, "layer3_mla_out", out_f32);
    }
    if (want_topk) {
        const topk_f32 = try ops.astype(ctx, mla_result.topk, .float32);
        defer topk_f32.deinit();
        try dumpBin(allocator, out_dir, "layer3_indexer_topk", topk_f32);
    }
}
