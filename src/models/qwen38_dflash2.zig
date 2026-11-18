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

    fn loadLinear(ws: WeightSource, prefix: []const u8) !MtpLinear {
        var nbuf: [256]u8 = undefined;
        const w = try ws.tensor(try std.fmt.bufPrint(&nbuf, "{s}.weight", .{prefix}));
        return .{ .w = w };
    }

    fn loadConv(ws: WeightSource, prefix: []const u8) !ConvWeights {
        var nbuf: [256]u8 = undefined;
        const base = try ws.tensor(try std.fmt.bufPrint(&nbuf, "{s}.base_kernel", .{prefix}));
        errdefer base.deinit();
        const proj = try loadLinear(ws, try std.fmt.bufPrint(&nbuf, "{s}.kernel_projection", .{prefix}));
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
    ) !DFlash2 {
        const fc = try loadLinear(ws, "fc");
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
                .q_proj = try loadLinear(ws, try std.fmt.bufPrint(&n1, "{s}.q_proj", .{sa})),
                .k_proj = try loadLinear(ws, try std.fmt.bufPrint(&n2, "{s}.k_proj", .{sa})),
                .v_proj = try loadLinear(ws, try std.fmt.bufPrint(&n1, "{s}.v_proj", .{sa})),
                .o_proj = try loadLinear(ws, try std.fmt.bufPrint(&n2, "{s}.o_proj", .{sa})),
                .q_norm = try ws.tensor(try std.fmt.bufPrint(&n1, "{s}.q_norm.weight", .{sa})),
                .k_norm = try ws.tensor(try std.fmt.bufPrint(&n2, "{s}.k_norm.weight", .{sa})),
                .attention_conv = try loadConv(ws, try std.fmt.bufPrint(&n1, "{s}.attention_conv", .{p})),
                .mlp_conv = try loadConv(ws, try std.fmt.bufPrint(&n2, "{s}.mlp_conv", .{p})),
                .gate_proj = try loadLinear(ws, try std.fmt.bufPrint(&n1, "{s}.mlp.gate_proj", .{p})),
                .up_proj = try loadLinear(ws, try std.fmt.bufPrint(&n2, "{s}.mlp.up_proj", .{p})),
                .down_proj = try loadLinear(ws, try std.fmt.bufPrint(&n1, "{s}.mlp.down_proj", .{p})),
            };
            n_loaded += 1;
        }

        const hp = try loadLinear(ws, "candidate_selector.hidden_projection");
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
    pub fn load(allocator: std.mem.Allocator, io: std.Io, ctx: EagerContext, model_dir: []const u8) !DFlash2 {
        const cfg_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
        defer allocator.free(cfg_path);
        const cfg_raw = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, allocator, .limited(64 * 1024));
        defer allocator.free(cfg_raw);
        const cfg = try DFlash2Config.parse(allocator, cfg_raw);

        var index = try safetensors_reader.loadOrBuildIndex(allocator, model_dir);
        defer index.deinit();
        return loadFromSource(allocator, ctx, WeightSource.fromIndex(&index), cfg);
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
    var m = try DFlash2.loadFromSource(allocator, ctx, WeightSource.fromIndex(&index), cfg);
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
