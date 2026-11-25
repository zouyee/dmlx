//! Qwen3.8 DSpine block-parallel speculative drafter.
const std = @import("std");
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const fast_mod = @import("mlx").fast;
const math_mod = @import("mlx").math;
const activations = @import("mlx").activations;
const sort_mod = @import("mlx").sort;
const reduce_mod = @import("mlx").reduce;
const quantize_mod = @import("mlx").quantize;
const safetensors_reader = @import("mlx").safetensors_reader;
const glm53 = @import("glm53.zig");
const qwen38_mtp = @import("qwen38_mtp.zig");
const qwen38_dflash2 = @import("qwen38_dflash2.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const WeightSource = glm53.WeightSource;
const MtpLinear = qwen38_mtp.MtpLinear;

pub const DSpineConfig = struct {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    block_size: usize,
    trained_block_size: usize,
    mask_token_id: u32,
    target_layer_ids: [5]usize,
    num_target_layers: usize,
    sliding_window: usize,
    code_size: usize,
    bridge_rank: usize,
    bridge_eps: f32,
    head_topk: usize,

    pub fn parse(allocator: std.mem.Allocator, json_slice: []const u8, served_block: usize) !DSpineConfig {
        if (served_block < 2 or served_block > 8) return error.UnsupportedBlockSize;
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_slice, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadConfig;
        const root = parsed.value.object;
        const dc_value = root.get("dflash_config") orelse return error.BadConfig;
        if (dc_value != .object) return error.BadConfig;
        const dc = dc_value.object;
        if (!jsonBool(dc, "draft_block_causal") or
            !jsonBool(dc, "selected_prefix_feature_writes") or
            jsonBool(dc, "selected_prefix_terminal")) return error.UnsupportedConfig;
        if (!jsonStringEq(dc, "selected_prefix_decode_mode", "serial") or
            !jsonStringEq(dc, "selected_prefix_feature_write_mode", "independent")) return error.UnsupportedConfig;
        if (dc.get("conv_kernel_size") != null or dc.get("conv_group_size") != null) return error.UnsupportedConfig;
        const layers_value = dc.get("selected_prefix_feature_layers") orelse return error.BadConfig;
        if (layers_value != .array or layers_value.array.items.len != 5) return error.UnsupportedConfig;
        for (layers_value.array.items, 1..) |v, i| {
            if (v != .integer or v.integer != i) return error.UnsupportedConfig;
        }
        const bridge_value = dc.get("selected_prefix_code_bridge") orelse return error.BadConfig;
        if (bridge_value != .object) return error.BadConfig;
        const bridge = bridge_value.object;
        const second_value = root.get("code_bridge_second_pass") orelse return error.BadConfig;
        if (second_value != .object) return error.BadConfig;
        const second = second_value.object;
        if (!jsonStringEq(second, "mode", "serial_table")) return error.UnsupportedConfig;

        var cfg: DSpineConfig = .{
            .hidden_size = try getUsize(root, "hidden_size"),
            .intermediate_size = try getUsize(root, "intermediate_size"),
            .num_hidden_layers = try getUsize(root, "num_hidden_layers"),
            .num_attention_heads = try getUsize(root, "num_attention_heads"),
            .num_key_value_heads = try getUsize(root, "num_key_value_heads"),
            .head_dim = try getUsize(root, "head_dim"),
            .vocab_size = try getUsize(root, "vocab_size"),
            .rms_norm_eps = try getFloat(root, "rms_norm_eps"),
            .rope_theta = 0,
            .block_size = served_block,
            .trained_block_size = try getUsize(root, "block_size"),
            .mask_token_id = @intCast(try getUsize(dc, "mask_token_id")),
            .target_layer_ids = undefined,
            .num_target_layers = try getUsize(root, "num_target_layers"),
            .sliding_window = try getUsize(root, "sliding_window"),
            .code_size = try getUsize(bridge, "code_size"),
            .bridge_rank = try getUsize(bridge, "rank"),
            .bridge_eps = try getFloat(bridge, "eps"),
            .head_topk = try getUsize(second, "head_topk"),
        };
        if (cfg.num_hidden_layers != 5 or cfg.head_topk != 16 or cfg.trained_block_size < served_block) return error.UnsupportedConfig;
        if (cfg.hidden_size == 0 or cfg.intermediate_size == 0 or cfg.num_attention_heads == 0 or
            cfg.num_key_value_heads == 0 or cfg.head_dim == 0 or cfg.vocab_size == 0 or
            cfg.sliding_window < 2 or cfg.code_size == 0 or cfg.bridge_rank == 0 or
            cfg.rms_norm_eps <= 0 or cfg.bridge_eps <= 0) return error.BadConfig;
        const rope_value = root.get("rope_parameters") orelse return error.BadConfig;
        if (rope_value != .object) return error.BadConfig;
        cfg.rope_theta = try getFloat(rope_value.object, "rope_theta");
        if (cfg.rope_theta <= 0) return error.BadConfig;
        const taps = dc.get("target_layer_ids") orelse return error.BadConfig;
        if (taps != .array or taps.array.items.len != 5) return error.BadConfig;
        for (taps.array.items, 0..) |v, i| {
            if (v != .integer or v.integer < 0 or v.integer >= cfg.num_target_layers) return error.BadConfig;
            cfg.target_layer_ids[i] = @intCast(v.integer);
        }
        return cfg;
    }

    pub fn parseBlockEnv(text: ?[]const u8) !usize {
        const raw = text orelse return 8;
        const block = std.fmt.parseInt(usize, raw, 10) catch return error.InvalidDSpineBlock;
        if (block < 2 or block > 8) return error.InvalidDSpineBlock;
        return block;
    }

    fn getUsize(obj: std.json.ObjectMap, key: []const u8) !usize {
        const v = obj.get(key) orelse return error.BadConfig;
        if (v != .integer or v.integer < 0) return error.BadConfig;
        return @intCast(v.integer);
    }

    fn getFloat(obj: std.json.ObjectMap, key: []const u8) !f32 {
        const v = obj.get(key) orelse return error.BadConfig;
        return switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => error.BadConfig,
        };
    }

    fn jsonBool(obj: std.json.ObjectMap, key: []const u8) bool {
        const v = obj.get(key) orelse return false;
        return v == .bool and v.bool;
    }

    fn jsonStringEq(obj: std.json.ObjectMap, key: []const u8, expected: []const u8) bool {
        const v = obj.get(key) orelse return false;
        return v == .string and std.mem.eql(u8, v.string, expected);
    }
};

test "DSpineConfig.parse released checkpoint and served block eight" {
    const raw = @embedFile("testdata/dspine/meta.json");
    const cfg = try DSpineConfig.parse(std.testing.allocator, raw, 8);
    try std.testing.expectEqual(@as(usize, 8), cfg.block_size);
    try std.testing.expectEqual(@as(usize, 16), cfg.trained_block_size);
    try std.testing.expectEqual(@as(usize, 16), cfg.head_topk);
    try std.testing.expectEqual([5]usize{ 1, 16, 31, 46, 61 }, cfg.target_layer_ids);
}

test "DSpineConfig rejects unsupported served block" {
    const raw = @embedFile("testdata/dspine/meta.json");
    try std.testing.expectError(error.UnsupportedBlockSize, DSpineConfig.parse(std.testing.allocator, raw, 16));
}

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
    q_norm: Array,
    k_norm: Array,
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
        deinitLinear(&self.gate_proj);
        deinitLinear(&self.up_proj);
        deinitLinear(&self.down_proj);
    }
};

pub const BridgeWeights = struct {
    prev_w: Array,
    gate_w: Array,
    gate_b: Array,
    predecessor_gate_w: Array,
    out_w: Array,
    read_w: Array,
    early_read_w: Array,
    feature_write_w: Array,
    codebook: Array,
    packed_w: Array,

    pub fn deinit(self: *BridgeWeights) void {
        self.prev_w.deinit();
        self.gate_w.deinit();
        self.gate_b.deinit();
        self.predecessor_gate_w.deinit();
        self.out_w.deinit();
        self.read_w.deinit();
        self.early_read_w.deinit();
        self.feature_write_w.deinit();
        self.codebook.deinit();
        self.packed_w.deinit();
    }
};

pub const DSpine = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    cfg: DSpineConfig,
    fc: MtpLinear,
    hidden_norm_f32: Array,
    norm_f32: Array,
    layers: []LayerWeights,
    bridge: BridgeWeights,

    fn loadNormF32(ctx: EagerContext, ws: WeightSource, name: []const u8) !Array {
        const w = try ws.tensor(name);
        defer w.deinit();
        const out = try ops.astype(ctx, w, .float32);
        errdefer out.deinit();
        try out.eval();
        return out;
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

    pub fn loadFromSource(allocator: std.mem.Allocator, ctx: EagerContext, ws: WeightSource, cfg: DSpineConfig, quant_bits: u8) !DSpine {
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
                .gate_proj = try loadLinear(ctx, ws, try std.fmt.bufPrint(&n1, "{s}.mlp.gate_proj", .{p}), quant_bits),
                .up_proj = try loadLinear(ctx, ws, try std.fmt.bufPrint(&n2, "{s}.mlp.up_proj", .{p}), quant_bits),
                .down_proj = try loadLinear(ctx, ws, try std.fmt.bufPrint(&n1, "{s}.mlp.down_proj", .{p}), quant_bits),
            };
            n_loaded += 1;
        }
        const prev_w = try ws.tensor("selected_prefix_bridge.prev.weight");
        errdefer prev_w.deinit();
        const gate_w = try ws.tensor("selected_prefix_bridge.gate.weight");
        errdefer gate_w.deinit();
        const gate_b = try ws.tensor("selected_prefix_bridge.gate.bias");
        errdefer gate_b.deinit();
        const predecessor_gate_w = try ws.tensor("selected_prefix_bridge.predecessor_gate.weight");
        errdefer predecessor_gate_w.deinit();
        const out_w = try ws.tensor("selected_prefix_bridge.out.weight");
        errdefer out_w.deinit();
        const read_w = try ws.tensor("selected_prefix_bridge.read_weight");
        errdefer read_w.deinit();
        const early_read_w = try ws.tensor("selected_prefix_bridge.early_read_weights");
        errdefer early_read_w.deinit();
        const feature_write_w = try ws.tensor("selected_prefix_bridge.feature_write_weights");
        errdefer feature_write_w.deinit();
        const codebook = try ws.tensor("selected_prefix_bridge.codebook");
        errdefer codebook.deinit();
        const packed_w = try shape_mod.concatenateAxis(ctx, &[_]Array{ gate_w, predecessor_gate_w, prev_w }, 0);
        errdefer packed_w.deinit();
        try packed_w.eval();
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .cfg = cfg,
            .fc = fc,
            .hidden_norm_f32 = hn,
            .norm_f32 = nf,
            .layers = layers,
            .bridge = .{
                .prev_w = prev_w,
                .gate_w = gate_w,
                .gate_b = gate_b,
                .predecessor_gate_w = predecessor_gate_w,
                .out_w = out_w,
                .read_w = read_w,
                .early_read_w = early_read_w,
                .feature_write_w = feature_write_w,
                .codebook = codebook,
                .packed_w = packed_w,
            },
        };
    }

    pub fn load(allocator: std.mem.Allocator, ctx: EagerContext, model_dir: []const u8, quant_bits: u8) !DSpine {
        return loadBlock(allocator, ctx, model_dir, quant_bits, 8);
    }

    pub fn loadBlock(allocator: std.mem.Allocator, ctx: EagerContext, model_dir: []const u8, quant_bits: u8, served_block: usize) !DSpine {
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
        const cfg = try DSpineConfig.parse(allocator, buf[0..@intCast(n)], served_block);
        var index = try safetensors_reader.loadOrBuildIndex(allocator, model_dir);
        defer index.deinit();
        return loadFromSource(allocator, ctx, WeightSource.fromIndex(&index), cfg, quant_bits);
    }

    pub fn deinit(self: *DSpine) void {
        deinitLinear(&self.fc);
        self.hidden_norm_f32.deinit();
        self.norm_f32.deinit();
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        self.bridge.deinit();
    }
};

pub fn projectContext(ctx: EagerContext, m: *const DSpine, tap_hidden: Array) !Array {
    const proj = try m.fc.apply(ctx, tap_hidden);
    defer proj.deinit();
    return qwen38_dflash2.rmsNormLastDim(ctx, proj, m.hidden_norm_f32, m.cfg.rms_norm_eps);
}

fn scalarF32(ctx: EagerContext, value: f32) !Array {
    return Array.fromData(ctx.allocator, f32, &[_]f32{value}, &[_]i32{1});
}

fn matmulT(ctx: EagerContext, x: Array, w: Array) !Array {
    const wt = try ops.transpose(ctx, w);
    defer wt.deinit();
    return ops.matmul(ctx, x, wt);
}

pub const BridgeStash = struct {
    raw: Array,
    rms: Array,
    packed_state: Array,

    pub fn deinit(self: *BridgeStash) void {
        self.raw.deinit();
        self.rms.deinit();
        self.packed_state.deinit();
    }
};

pub const BridgeOut = struct {
    normed: Array,
    residual: Array,
    stash: ?BridgeStash = null,

    pub fn deinit(self: *BridgeOut) void {
        self.normed.deinit();
        self.residual.deinit();
        if (self.stash) |*s| s.deinit();
    }
};

pub fn bridgeWrite(ctx: EagerContext, m: *const DSpine, raw: Array, block_ids: Array, layer: usize, next_norm_f32: Array, save_stash: bool) !BridgeOut {
    const cfg = &m.cfg;
    const b = &m.bridge;
    const T = raw.shape()[1];
    const H = raw.shape()[2];
    const C: i32 = @intCast(cfg.code_size);
    const R: i32 = @intCast(cfg.bridge_rank);
    if (raw.dtype() != .bfloat16) return error.BridgeRequiresBf16;
    const f = try ops.astype(ctx, raw, .float32);
    defer f.deinit();
    const sq = try ops.multiply(ctx, f, f);
    defer sq.deinit();
    const mean = try reduce_mod.meanAxis(ctx, sq, -1, true);
    defer mean.deinit();
    const tiny = try scalarF32(ctx, std.math.floatMin(f32));
    defer tiny.deinit();
    const mean_clamped = try math_mod.maximum(ctx, mean, tiny);
    defer mean_clamped.deinit();
    const rms = try ops.sqrt(ctx, mean_clamped);
    defer rms.deinit();
    const eps = try scalarF32(ctx, cfg.bridge_eps);
    defer eps.deinit();
    const rms_eps = try ops.add(ctx, rms, eps);
    defer rms_eps.deinit();
    const norm_f = try ops.divide(ctx, f, rms_eps);
    defer norm_f.deinit();
    const normalized = try ops.astype(ctx, norm_f, .bfloat16);
    defer normalized.deinit();
    const read_w = if (layer == cfg.num_hidden_layers - 1)
        try ops.copy(ctx, b.read_w)
    else
        try ops.slice(ctx, b.early_read_w, &[_]i32{ @intCast(layer), 0, 0 }, &[_]i32{ @intCast(layer + 1), C, H }, &[_]i32{ 1, 1, 1 });
    defer read_w.deinit();
    const read_2d = try ops.reshape(ctx, read_w, &[_]i32{ C, H });
    defer read_2d.deinit();
    const projected_b = try matmulT(ctx, normalized, read_2d);
    defer projected_b.deinit();
    const projected = try ops.astype(ctx, projected_b, .float32);
    defer projected.deinit();
    const psq = try ops.multiply(ctx, projected, projected);
    defer psq.deinit();
    const psum = try reduce_mod.sumAxis(ctx, psq, -1, true);
    defer psum.deinit();
    const pnorm = try ops.sqrt(ctx, psum);
    defer pnorm.deinit();
    const pdenom = try ops.add(ctx, pnorm, eps);
    defer pdenom.deinit();
    const sqrt_c = try scalarF32(ctx, @sqrt(@as(f32, @floatFromInt(cfg.code_size))));
    defer sqrt_c.deinit();
    const projected_scaled = try ops.multiply(ctx, projected, sqrt_c);
    defer projected_scaled.deinit();
    const codes_f = try ops.divide(ctx, projected_scaled, pdenom);
    defer codes_f.deinit();
    const codes_b = try ops.astype(ctx, codes_f, .bfloat16);
    defer codes_b.deinit();
    const anchor = try ops.slice(ctx, block_ids, &[_]i32{0}, &[_]i32{1}, &[_]i32{1});
    defer anchor.deinit();
    const anchor_code = try shape_mod.takeAxis(ctx, b.codebook, anchor, 0);
    defer anchor_code.deinit();
    const anchor_f = try ops.astype(ctx, anchor_code, .float32);
    defer anchor_f.deinit();
    const anchor_scaled = try ops.multiply(ctx, anchor_f, sqrt_c);
    defer anchor_scaled.deinit();
    const anchor_b = try ops.astype(ctx, anchor_scaled, .bfloat16);
    defer anchor_b.deinit();
    const anchor_3 = try ops.reshape(ctx, anchor_b, &[_]i32{ 1, 1, C });
    defer anchor_3.deinit();
    const code_tail = try ops.slice(ctx, codes_b, &[_]i32{ 0, 1, 0 }, &[_]i32{ 1, T, C }, &[_]i32{ 1, 1, 1 });
    defer code_tail.deinit();
    const codes = try shape_mod.concatenateAxis(ctx, &[_]Array{ anchor_3, code_tail }, 1);
    defer codes.deinit();
    const packed_state = try matmulT(ctx, codes, b.packed_w);
    defer packed_state.deinit();
    const packed_f = try ops.astype(ctx, packed_state, .float32);
    defer packed_f.deinit();
    const q = try ops.slice(ctx, packed_f, &[_]i32{ 0, 1, 0 }, &[_]i32{ 1, T, R }, &[_]i32{ 1, 1, 1 });
    defer q.deinit();
    const s = try ops.slice(ctx, packed_f, &[_]i32{ 0, 0, R }, &[_]i32{ 1, T - 1, 2 * R }, &[_]i32{ 1, 1, 1 });
    defer s.deinit();
    const v = try ops.slice(ctx, packed_f, &[_]i32{ 0, 0, 2 * R }, &[_]i32{ 1, T - 1, 3 * R }, &[_]i32{ 1, 1, 1 });
    defer v.deinit();
    const qs = try ops.add(ctx, q, s);
    defer qs.deinit();
    const gate_b32 = try ops.astype(ctx, b.gate_b, .float32);
    defer gate_b32.deinit();
    const qsb = try ops.add(ctx, qs, gate_b32);
    defer qsb.deinit();
    const gate = try ops.sigmoid(ctx, qsb);
    defer gate.deinit();
    const message_tail_f = try ops.multiply(ctx, gate, v);
    defer message_tail_f.deinit();
    const message_tail = try ops.astype(ctx, message_tail_f, .bfloat16);
    defer message_tail.deinit();
    const message_head = try Array.zeros(ctx.allocator, &[_]i32{ 1, 1, R }, .bfloat16);
    defer message_head.deinit();
    const message = try shape_mod.concatenateAxis(ctx, &[_]Array{ message_head, message_tail }, 1);
    defer message.deinit();
    const write_3 = try ops.slice(ctx, b.feature_write_w, &[_]i32{ @intCast(layer), 0, 0 }, &[_]i32{ @intCast(layer + 1), H, R }, &[_]i32{ 1, 1, 1 });
    defer write_3.deinit();
    const write_w = try ops.reshape(ctx, write_3, &[_]i32{ H, R });
    defer write_w.deinit();
    const delta_b = try matmulT(ctx, message, write_w);
    defer delta_b.deinit();
    const delta = try ops.astype(ctx, delta_b, .float32);
    defer delta.deinit();
    const scaled_delta = try ops.multiply(ctx, rms, delta);
    defer scaled_delta.deinit();
    const f_head = try ops.slice(ctx, f, &[_]i32{ 0, 0, 0 }, &[_]i32{ 1, 1, H }, &[_]i32{ 1, 1, 1 });
    defer f_head.deinit();
    const f_tail = try ops.slice(ctx, f, &[_]i32{ 0, 1, 0 }, &[_]i32{ 1, T, H }, &[_]i32{ 1, 1, 1 });
    defer f_tail.deinit();
    const delta_tail = try ops.slice(ctx, scaled_delta, &[_]i32{ 0, 1, 0 }, &[_]i32{ 1, T, H }, &[_]i32{ 1, 1, 1 });
    defer delta_tail.deinit();
    const out_tail = try ops.add(ctx, f_tail, delta_tail);
    defer out_tail.deinit();
    const out_f = try shape_mod.concatenateAxis(ctx, &[_]Array{ f_head, out_tail }, 1);
    defer out_f.deinit();
    const residual = try ops.astype(ctx, out_f, .bfloat16);
    errdefer residual.deinit();
    const normed = try qwen38_dflash2.rmsNormLastDim(ctx, residual, next_norm_f32, cfg.rms_norm_eps);
    errdefer normed.deinit();
    var stash: ?BridgeStash = null;
    if (save_stash) {
        stash = .{
            .raw = try ops.copy(ctx, raw),
            .rms = try ops.copy(ctx, rms),
            .packed_state = try ops.copy(ctx, packed_state),
        };
    }
    return .{ .normed = normed, .residual = residual, .stash = stash };
}

pub const DSpineContext = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    cfg: *const DSpineConfig,
    model: *const DSpine,
    capacity: i32,
    ring_slots: i64,
    fed: i64 = 0,
    keys: [5]Array,
    values: [5]Array,
    pos: []i64,
    rows_shadow: ?Array = null,

    pub fn init(allocator: std.mem.Allocator, ctx: EagerContext, model: *const DSpine) !DSpineContext {
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

    pub fn reset(self: *DSpineContext) void {
        @memset(self.pos, -1);
        self.fed = 0;
        if (self.rows_shadow) |r| r.deinit();
        self.rows_shadow = null;
    }

    pub fn rowsShadow(self: *DSpineContext) ?Array {
        return self.rows_shadow;
    }

    pub fn observe(self: *DSpineContext, tap_hidden: Array) !void {
        if (tap_hidden.shape()[1] == 0) return;
        const proj = try projectContext(self.ctx, self.model, tap_hidden);
        defer proj.deinit();
        return self.seedFromProjectedAt(proj, self.fed);
    }

    pub fn seedFromProjectedAt(self: *DSpineContext, proj: Array, fed_start: i64) !void {
        self.fed = fed_start;
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
            const kn = try qwen38_dflash2.rmsNormLastDim(ctx, k4, lw.k_norm, cfg.rms_norm_eps);
            defer kn.deinit();
            const kt = try ops.transposeAxes(ctx, kn, &[_]i32{ 0, 2, 1, 3 });
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

    fn writeSlots(self: *DSpineContext, layer: usize, k3: Array, v3: Array, k: i32) !void {
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

    pub fn draftMask(self: *DSpineContext) !Array {
        const blk: i32 = @intCast(self.cfg.block_size);
        const width: usize = @intCast(self.capacity + blk);
        const data = try self.allocator.alloc(bool, @as(usize, @intCast(blk)) * width);
        defer self.allocator.free(data);
        const lo = self.fed - self.ring_slots;
        for (0..@intCast(blk)) |qi| {
            for (0..@intCast(self.capacity)) |i| {
                data[qi * width + i] = self.pos[i] >= 0 and self.pos[i] >= lo;
            }
            for (0..@intCast(blk)) |kj| data[qi * width + @as(usize, @intCast(self.capacity)) + kj] = kj <= qi;
        }
        return Array.fromData(self.allocator, bool, data, &[_]i32{ 1, 1, blk, @intCast(width) });
    }

    pub fn deinit(self: *DSpineContext) void {
        for (0..self.cfg.num_hidden_layers) |l| {
            self.keys[l].deinit();
            self.values[l].deinit();
        }
        if (self.rows_shadow) |r| r.deinit();
        self.allocator.free(self.pos);
    }
};

fn attentionForwardRing(ctx: EagerContext, w: *const LayerWeights, cfg: *const DSpineConfig, x: Array, ring: *DSpineContext, layer_idx: usize, q_off: i32, mask: Array) !Array {
    const L = x.shape()[1];
    const nh: i32 = @intCast(cfg.num_attention_heads);
    const nkv: i32 = @intCast(cfg.num_key_value_heads);
    const hd: i32 = @intCast(cfg.head_dim);
    const q = try w.q_proj.apply(ctx, x);
    defer q.deinit();
    const q4 = try ops.reshape(ctx, q, &[_]i32{ 1, L, nh, hd });
    defer q4.deinit();
    const qn = try qwen38_dflash2.rmsNormLastDim(ctx, q4, w.q_norm, cfg.rms_norm_eps);
    defer qn.deinit();
    const qt = try ops.transposeAxes(ctx, qn, &[_]i32{ 0, 2, 1, 3 });
    defer qt.deinit();
    const qr = try fast_mod.rope(ctx, qt, hd, false, cfg.rope_theta, 1.0, q_off, null);
    defer qr.deinit();
    const pk = try w.k_proj.apply(ctx, x);
    defer pk.deinit();
    const pk4 = try ops.reshape(ctx, pk, &[_]i32{ 1, L, nkv, hd });
    defer pk4.deinit();
    const pkn = try qwen38_dflash2.rmsNormLastDim(ctx, pk4, w.k_norm, cfg.rms_norm_eps);
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

fn swigluMlp(ctx: EagerContext, w: *const LayerWeights, x: Array) !Array {
    const gate = try w.gate_proj.apply(ctx, x);
    defer gate.deinit();
    const up = try w.up_proj.apply(ctx, x);
    defer up.deinit();
    const sig = try ops.sigmoid(ctx, gate);
    defer sig.deinit();
    const silu = try ops.multiply(ctx, gate, sig);
    defer silu.deinit();
    const prod = try ops.multiply(ctx, silu, up);
    defer prod.deinit();
    return w.down_proj.apply(ctx, prod);
}

pub const BlockOut = struct {
    hidden: Array,
    stash: BridgeStash,

    pub fn deinit(self: *BlockOut) void {
        self.hidden.deinit();
        self.stash.deinit();
    }
};

pub fn blockHiddenRing(ctx: EagerContext, m: *const DSpine, ring: *DSpineContext, embed_table: Array, block_ids: Array) !BlockOut {
    const blk: i32 = block_ids.shape()[0];
    const H: i32 = @intCast(m.cfg.hidden_size);
    const q_off: i32 = @intCast(ring.fed);
    if (blk != m.cfg.block_size) return error.UnsupportedBlockSize;
    const mask = try ring.draftMask();
    defer mask.deinit();
    const emb = try shape_mod.takeAxis(ctx, embed_table, block_ids, 0);
    defer emb.deinit();
    var residual = try ops.reshape(ctx, emb, &[_]i32{ 1, blk, H });
    errdefer residual.deinit();
    var normed = try qwen38_dflash2.rmsNormLastDim(ctx, residual, m.layers[0].input_layernorm_f32, m.cfg.rms_norm_eps);
    errdefer normed.deinit();
    var stash: ?BridgeStash = null;
    for (m.layers, 0..) |*l, li| {
        const attn = try attentionForwardRing(ctx, l, &m.cfg, normed, ring, li, q_off, mask);
        defer attn.deinit();
        const h1 = try ops.add(ctx, residual, attn);
        defer h1.deinit();
        const xn2 = try qwen38_dflash2.rmsNormLastDim(ctx, h1, l.post_attention_layernorm_f32, m.cfg.rms_norm_eps);
        defer xn2.deinit();
        const mlp = try swigluMlp(ctx, l, xn2);
        defer mlp.deinit();
        const raw = try ops.add(ctx, h1, mlp);
        defer raw.deinit();
        const next_norm = if (li + 1 < m.layers.len) m.layers[li + 1].input_layernorm_f32 else m.norm_f32;
        var bridged = try bridgeWrite(ctx, m, raw, block_ids, li, next_norm, li + 1 == m.layers.len);
        residual.deinit();
        normed.deinit();
        residual = bridged.residual;
        normed = bridged.normed;
        if (bridged.stash) |s| stash = s;
        bridged.stash = null;
    }
    residual.deinit();
    const hidden = try ops.slice(ctx, normed, &[_]i32{ 0, 1, 0 }, &[_]i32{ 1, blk, H }, &[_]i32{ 1, 1, 1 });
    normed.deinit();
    errdefer hidden.deinit();
    return .{ .hidden = hidden, .stash = stash.? };
}

test "DSpine serial host chain and predecessor-conditioned sparse q" {
    const allocator = std.testing.allocator;
    const candidates = [_]u32{ 10, 11, 20, 21 };
    const table = [_]f32{
        0.1, 0.9, 0.7, 0.2,
        0.3, 0.4, 0.8, 0.2,
    };
    const greedy = try greedyChainFromHost(&table, &candidates, 2, 2, allocator);
    defer allocator.free(greedy);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 11, 20 }, greedy);
    const row = try sparseLogitRowFromHost(&table, &candidates, 1, 11, 32, allocator, 2, 2);
    defer allocator.free(row);
    try std.testing.expectEqual(@as(f32, 0.8), row[20]);
    try std.testing.expectEqual(@as(f32, 0.2), row[21]);
    try std.testing.expect(std.math.isNegativeInf(row[10]));
    try std.testing.expectError(error.PredecessorNotCandidate, sparseLogitRowFromHost(&table, &candidates, 1, 12, 32, allocator, 2, 2));
}

pub fn greedyChainFromHost(table: []const f32, candidates: []const u32, positions: usize, k: usize, allocator: std.mem.Allocator) ![]u32 {
    if (positions == 0 or k == 0 or table.len != positions * k * k or candidates.len != positions * k) return error.BadRefinementShape;
    const out = try allocator.alloc(u32, positions);
    errdefer allocator.free(out);
    var sender: usize = 0;
    for (0..positions) |p| {
        var best: usize = 0;
        var best_score = -std.math.inf(f32);
        const row = table[(p * k + sender) * k ..][0..k];
        for (row, 0..) |score, j| {
            if (score > best_score) {
                best_score = score;
                best = j;
            }
        }
        out[p] = candidates[p * k + best];
        sender = best;
    }
    return out;
}

pub fn sparseLogitRowFromHost(table: []const f32, candidates: []const u32, position: usize, previous_token: u32, vocab: usize, allocator: std.mem.Allocator, positions: usize, k: usize) ![]f32 {
    return sparseRowImpl(table, candidates, position, previous_token, vocab, allocator, positions, k);
}

fn sparseRowImpl(table: []const f32, candidates: []const u32, position: usize, previous_token: u32, vocab: usize, allocator: std.mem.Allocator, positions: usize, k: usize) ![]f32 {
    if (position >= positions or candidates.len != positions * k or table.len != positions * k * k) return error.BadRefinementShape;
    var sender: usize = 0;
    if (position > 0) {
        sender = std.mem.indexOfScalar(u32, candidates[(position - 1) * k ..][0..k], previous_token) orelse return error.PredecessorNotCandidate;
    }
    const out = try allocator.alloc(f32, vocab);
    errdefer allocator.free(out);
    @memset(out, -std.math.inf(f32));
    const scores = table[(position * k + sender) * k ..][0..k];
    for (candidates[position * k ..][0..k], scores) |token, score| {
        if (token >= vocab) return error.CandidateOutOfVocabulary;
        out[token] = score;
    }
    return out;
}

pub fn selectCandidates(ctx: EagerContext, logits: Array, k: usize) !Array {
    if (logits.shape().len != 3) return error.BadLogitsShape;
    const P = logits.shape()[1];
    const V = logits.shape()[2];
    if (k == 0 or k > V) return error.BadLogitsShape;
    const neg = try ops.negative(ctx, logits);
    defer neg.deinit();
    const order = try sort_mod.argsortAxis(ctx, neg, -1);
    defer order.deinit();
    const top = try ops.slice(ctx, order, &[_]i32{ 0, 0, 0 }, &[_]i32{ 1, P, @intCast(k) }, &[_]i32{ 1, 1, 1 });
    defer top.deinit();
    const top_i32 = try ops.astype(ctx, top, .int32);
    defer top_i32.deinit();
    const sorted = try sort_mod.sortAxis(ctx, top_i32, -1);
    defer sorted.deinit();
    return ops.reshape(ctx, sorted, &[_]i32{ P, @intCast(k) });
}

fn gatherHeadRows(ctx: EagerContext, lm: MtpLinear, candidates: Array, hidden_size: usize) !Array {
    const P = candidates.shape()[0];
    const K = candidates.shape()[1];
    const flat = try ops.reshape(ctx, candidates, &[_]i32{P * K});
    defer flat.deinit();
    if (lm.scales) |scales| {
        const qw = try shape_mod.takeAxis(ctx, lm.w, flat, 0);
        defer qw.deinit();
        const qs = try shape_mod.takeAxis(ctx, scales, flat, 0);
        defer qs.deinit();
        const qb = try shape_mod.takeAxis(ctx, lm.biases.?, flat, 0);
        defer qb.deinit();
        const deq = try quantize_mod.dequantizeAs(ctx, .{
            .data = qw,
            .scales = qs,
            .biases = qb,
            .config = .{ .bits = 4, .group_size = 64, .mode = .affine },
            .original_shape = &.{},
        }, .bfloat16);
        defer deq.deinit();
        return ops.reshape(ctx, deq, &[_]i32{ P, K, @intCast(hidden_size) });
    }
    const rows = try shape_mod.takeAxis(ctx, lm.w, flat, 0);
    defer rows.deinit();
    const rows_b = try ops.astype(ctx, rows, .bfloat16);
    defer rows_b.deinit();
    return ops.reshape(ctx, rows_b, &[_]i32{ P, K, @intCast(hidden_size) });
}

pub fn refinementTable(ctx: EagerContext, m: *const DSpine, lm: MtpLinear, hidden_rows: Array, stash: BridgeStash, candidates: Array, anchor_gpu: Array) !Array {
    const P = candidates.shape()[0];
    const K = candidates.shape()[1];
    const H: i32 = @intCast(m.cfg.hidden_size);
    const C: i32 = @intCast(m.cfg.code_size);
    const R: i32 = @intCast(m.cfg.bridge_rank);
    if (hidden_rows.shape()[1] != P or K != m.cfg.head_topk) return error.BadRefinementShape;
    const anchor_2 = try ops.reshape(ctx, anchor_gpu, &[_]i32{ 1, 1 });
    defer anchor_2.deinit();
    const anchor_arr = try ops.broadcastTo(ctx, anchor_2, &[_]i32{ 1, K });
    defer anchor_arr.deinit();
    const previous = try ops.slice(ctx, candidates, &[_]i32{ 0, 0 }, &[_]i32{ P - 1, K }, &[_]i32{ 1, 1 });
    defer previous.deinit();
    const senders = try shape_mod.concatenateAxis(ctx, &[_]Array{ anchor_arr, previous }, 0);
    defer senders.deinit();
    const sender_flat = try ops.reshape(ctx, senders, &[_]i32{P * K});
    defer sender_flat.deinit();
    const code = try shape_mod.takeAxis(ctx, m.bridge.codebook, sender_flat, 0);
    defer code.deinit();
    const code_3 = try ops.reshape(ctx, code, &[_]i32{ P, K, C });
    defer code_3.deinit();
    const code_f = try ops.astype(ctx, code_3, .float32);
    defer code_f.deinit();
    const sqrt_c = try scalarF32(ctx, @sqrt(@as(f32, @floatFromInt(m.cfg.code_size))));
    defer sqrt_c.deinit();
    const e = try ops.multiply(ctx, code_f, sqrt_c);
    defer e.deinit();
    const pred_w = try ops.astype(ctx, m.bridge.predecessor_gate_w, .float32);
    defer pred_w.deinit();
    const pred = try matmulT(ctx, e, pred_w);
    defer pred.deinit();
    const gate_q = try ops.slice(ctx, stash.packed_state, &[_]i32{ 0, 1, 0 }, &[_]i32{ 1, P + 1, R }, &[_]i32{ 1, 1, 1 });
    defer gate_q.deinit();
    const gate_q_f = try ops.astype(ctx, gate_q, .float32);
    defer gate_q_f.deinit();
    const gate_q_3 = try ops.reshape(ctx, gate_q_f, &[_]i32{ P, 1, R });
    defer gate_q_3.deinit();
    const q_pred = try ops.add(ctx, gate_q_3, pred);
    defer q_pred.deinit();
    const gate_b = try ops.astype(ctx, m.bridge.gate_b, .float32);
    defer gate_b.deinit();
    const gate_pre = try ops.add(ctx, q_pred, gate_b);
    defer gate_pre.deinit();
    const gate = try ops.sigmoid(ctx, gate_pre);
    defer gate.deinit();
    const prev_w = try ops.astype(ctx, m.bridge.prev_w, .float32);
    defer prev_w.deinit();
    const value = try matmulT(ctx, e, prev_w);
    defer value.deinit();
    const message = try ops.multiply(ctx, gate, value);
    defer message.deinit();
    const write_3 = try ops.slice(ctx, m.bridge.feature_write_w, &[_]i32{ @intCast(m.cfg.num_hidden_layers - 1), 0, 0 }, &[_]i32{ @intCast(m.cfg.num_hidden_layers), H, R }, &[_]i32{ 1, 1, 1 });
    defer write_3.deinit();
    const write_2 = try ops.reshape(ctx, write_3, &[_]i32{ H, R });
    defer write_2.deinit();
    const write_w = try ops.astype(ctx, write_2, .float32);
    defer write_w.deinit();
    const delta = try matmulT(ctx, message, write_w);
    defer delta.deinit();
    const raw = try ops.slice(ctx, stash.raw, &[_]i32{ 0, 1, 0 }, &[_]i32{ 1, P + 1, H }, &[_]i32{ 1, 1, 1 });
    defer raw.deinit();
    const raw_f = try ops.astype(ctx, raw, .float32);
    defer raw_f.deinit();
    const raw_3 = try ops.reshape(ctx, raw_f, &[_]i32{ P, 1, H });
    defer raw_3.deinit();
    const rms = try ops.slice(ctx, stash.rms, &[_]i32{ 0, 1, 0 }, &[_]i32{ 1, P + 1, 1 }, &[_]i32{ 1, 1, 1 });
    defer rms.deinit();
    const rms_3 = try ops.reshape(ctx, rms, &[_]i32{ P, 1, 1 });
    defer rms_3.deinit();
    const scaled = try ops.multiply(ctx, rms_3, delta);
    defer scaled.deinit();
    const refined = try ops.add(ctx, raw_3, scaled);
    defer refined.deinit();
    const rsq = try ops.multiply(ctx, refined, refined);
    defer rsq.deinit();
    const rmean = try reduce_mod.meanAxis(ctx, rsq, -1, true);
    defer rmean.deinit();
    const eps = try scalarF32(ctx, m.cfg.rms_norm_eps);
    defer eps.deinit();
    const rmean_eps = try ops.add(ctx, rmean, eps);
    defer rmean_eps.deinit();
    const rnorm = try ops.sqrt(ctx, rmean_eps);
    defer rnorm.deinit();
    const normed = try ops.divide(ctx, refined, rnorm);
    defer normed.deinit();
    const norm_w = try ops.astype(ctx, m.norm_f32, .float32);
    defer norm_w.deinit();
    const weighted = try ops.multiply(ctx, normed, norm_w);
    defer weighted.deinit();
    const head_rows = try gatherHeadRows(ctx, lm, candidates, m.cfg.hidden_size);
    defer head_rows.deinit();
    const head_f = try ops.astype(ctx, head_rows, .float32);
    defer head_f.deinit();
    const head_t = try ops.transposeAxes(ctx, head_f, &[_]i32{ 0, 2, 1 });
    defer head_t.deinit();
    return ops.matmul(ctx, weighted, head_t);
}

test "DSpineConfig accepts block five" {
    const raw = @embedFile("testdata/dspine/meta.json");
    const cfg = try DSpineConfig.parse(std.testing.allocator, raw, 5);
    try std.testing.expectEqual(@as(usize, 5), cfg.block_size);
    try std.testing.expectEqual(@as(usize, 16), cfg.trained_block_size);
}

test "DSpineConfig rejects blocks outside two through eight" {
    const raw = @embedFile("testdata/dspine/meta.json");
    try std.testing.expectError(error.UnsupportedBlockSize, DSpineConfig.parse(std.testing.allocator, raw, 1));
    try std.testing.expectError(error.UnsupportedBlockSize, DSpineConfig.parse(std.testing.allocator, raw, 9));
}

test "DSpineConfig parses optional block environment" {
    try std.testing.expectEqual(@as(usize, 8), try DSpineConfig.parseBlockEnv(null));
    try std.testing.expectEqual(@as(usize, 5), try DSpineConfig.parseBlockEnv("5"));
    try std.testing.expectError(error.InvalidDSpineBlock, DSpineConfig.parseBlockEnv("abc"));
    try std.testing.expectError(error.InvalidDSpineBlock, DSpineConfig.parseBlockEnv("0"));
    try std.testing.expectError(error.InvalidDSpineBlock, DSpineConfig.parseBlockEnv("9"));
}
