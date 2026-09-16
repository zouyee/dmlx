/// GLM-5.3 oracle alignment harness (Zig side).
///
/// Loads the prep'd checkpoint (~/models/GLM-5.3-Flash-MLX-mixed-4_8bit-dmlx),
/// runs the fixed 7-token fixture through embed + layer-0 KDA with the mHC
/// apply_branch semantics from mlx-vlm's HyperConnection, and dumps each
/// segment to artifacts/glm53_oracle/{seg}.bin for comparison against the
/// Python oracle fixtures (tests/fixtures/glm53_oracle/*.npy) via
/// scripts/compare_glm53_oracle.py.
///
/// bin format: [ndim: i32][dims: i32 x ndim][f32 row-major payload].
///
/// Usage:
///   zig build run-glm53-oracle-dump -- [--model-dir DIR] [--out-dir DIR] [--seg SEG]
/// SEG ∈ embed | layer0_kda_out | all (default all).
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
    if (!want_embed and !want_kda) return error.UnknownSegment;

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
    if (!want_kda) return;

    // === layer0 KDA segment ===
    // residual = repeat(embed, hc_mult) on axis 2 — matches the oracle's
    // mx.repeat(h[:, :, None], hc_mult, axis=2).
    const residual = try dsv4.expandToMHC(ctx, embed, hc_mult, stream_raw); // [1, L, 4, hidden] bf16
    defer residual.deinit();

    const p = "language_model.model.layers.0";

    // mHC (attn_hc) weights: fn bf16 [24, 4*hidden], base/scale f32.
    const hc_fn = try loadTensor(&index, try std.fmt.allocPrint(allocator, "{s}.attn_hc.fn", .{p}));
    defer hc_fn.deinit();
    const hc_base = try loadTensor(&index, try std.fmt.allocPrint(allocator, "{s}.attn_hc.base", .{p}));
    defer hc_base.deinit();
    const hc_scale = try loadTensor(&index, try std.fmt.allocPrint(allocator, "{s}.attn_hc.scale", .{p}));
    defer hc_scale.deinit();
    const ln_w = try loadTensor(&index, try std.fmt.allocPrint(allocator, "{s}.input_layernorm.weight", .{p}));
    defer ln_w.deinit();

    // HC pre: Python computes mixes from the f32 residual and applies the
    // weight-less rms_norm with rms_norm_eps (not hc_eps).
    const res_f32 = try ops.astype(ctx, residual, .float32);
    defer res_f32.deinit();
    const hc_fn_f32 = try ops.astype(ctx, hc_fn, .float32);
    defer hc_fn_f32.deinit();
    const mixes = try dsv4.mhcPreNormFn(ctx, res_f32, hc_fn_f32, null, cfg.rms_norm_eps, stream_raw);
    defer mixes.deinit();
    const splits = try dsv4.mhcPreSplitMixes(ctx, mixes, hc_scale, hc_base, hc_mult, 2.0, cfg.hc_eps, stream_raw);
    const pre_mix = splits[0];
    const post_mix = splits[1];
    const comb_mix = splits[2];
    defer pre_mix.deinit();
    defer post_mix.deinit();
    defer comb_mix.deinit();
    const comb_norm = try dsv4.sinkhornNormalize(ctx, comb_mix, cfg.hc_sinkhorn_iters, cfg.hc_eps, stream_raw);
    defer comb_norm.deinit();

    const collapsed = try dsv4.mhcPreApplyMix(ctx, residual, pre_mix, stream_raw); // [1, L, hidden] bf16
    defer collapsed.deinit();
    const collapsed_f32 = try ops.astype(ctx, collapsed, .float32);
    defer collapsed_f32.deinit();
    const ln_w_f32 = try ops.astype(ctx, ln_w, .float32);
    defer ln_w_f32.deinit();
    const normed = try fast_mod.rmsNorm(ctx, collapsed_f32, ln_w_f32, cfg.rms_norm_eps);
    defer normed.deinit();

    // KDA weights (prep'd fused names; forget_gate.* keeps its prefix).
    const sa = try std.fmt.allocPrint(allocator, "{s}.self_attn", .{p});
    var qkv_q = try loadQuantized(&index, allocator, try std.fmt.allocPrint(allocator, "{s}.qkv_proj", .{sa}), hidden);
    defer qkv_q.deinit();
    var fbg_q = try loadQuantized(&index, allocator, try std.fmt.allocPrint(allocator, "{s}.fbg_a_proj", .{sa}), hidden);
    defer fbg_q.deinit();
    var o_q = try loadQuantized(&index, allocator, try std.fmt.allocPrint(allocator, "{s}.o_proj", .{sa}), @intCast(cfg.linear_num_heads * cfg.linear_head_dim));
    defer o_q.deinit();

    const qkv_conv = try loadTensor(&index, try std.fmt.allocPrint(allocator, "{s}.qkv_conv.weight", .{sa}));
    defer qkv_conv.deinit();
    const qkv_conv_f32 = try ops.astype(ctx, qkv_conv, .float32);
    defer qkv_conv_f32.deinit();

    // f_b/g_b are consumed unquantized by GlmKDA — dequantize at load.
    var fb_q = try loadQuantized(&index, allocator, try std.fmt.allocPrint(allocator, "{s}.forget_gate.f_b_proj", .{sa}), @intCast(cfg.linear_head_dim));
    defer fb_q.deinit();
    const f_b = try quantize_mod.dequantizeAs(ctx, .{ .data = fb_q.data, .scales = fb_q.scales, .biases = fb_q.biases, .config = fb_q.cfg, .original_shape = &[_]i32{} }, .float32);
    defer f_b.deinit();
    var gb_q = try loadQuantized(&index, allocator, try std.fmt.allocPrint(allocator, "{s}.g_b_proj", .{sa}), @intCast(cfg.linear_head_dim));
    defer gb_q.deinit();
    const g_b = try quantize_mod.dequantizeAs(ctx, .{ .data = gb_q.data, .scales = gb_q.scales, .biases = gb_q.biases, .config = gb_q.cfg, .original_shape = &[_]i32{} }, .float32);
    defer g_b.deinit();

    const a_log = try loadTensor(&index, try std.fmt.allocPrint(allocator, "{s}.forget_gate.A_log", .{sa}));
    defer a_log.deinit();
    const dt_bias = try loadTensor(&index, try std.fmt.allocPrint(allocator, "{s}.forget_gate.dt_bias", .{sa}));
    defer dt_bias.deinit();
    const o_norm_w = try loadTensor(&index, try std.fmt.allocPrint(allocator, "{s}.o_norm.weight", .{sa}));
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

    const kda_out = try kda.prefill(allocator, normed, &cache); // [1, L, hidden] f32
    defer kda_out.deinit();

    // HC expand (apply_branch post): post * out + comb^T @ residual.
    const out = try dsv4.mhcPost(ctx, kda_out, residual, post_mix, comb_norm, stream_raw);
    defer out.deinit();

    try dumpBin(allocator, out_dir, "layer0_kda_out", out);
}
