/// Nemotron-H: Hybrid SSM + Attention model architecture.
///
/// Combines Mamba/GatedDeltaNet SSM layers with standard multi-head attention
/// in a hybrid transformer stack.
const std = @import("std");
const c = @import("mlx").c;
const ops = @import("mlx").ops;
const nn = @import("mlx").nn;
const array_mod = @import("mlx").array;
const kvcache = @import("../kvcache.zig");
const array_arena_mod = @import("mlx").array_arena;
const activations = @import("mlx").activations;
const conv = @import("mlx").conv;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const ScopedArrayArena = array_arena_mod.ScopedArrayArena;

// ============================================================
// Configuration
// ============================================================

pub const SSMVariant = enum {
    mamba,
    gated_delta,
};

pub const NemotronHConfig = struct {
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    ssm_dim: usize,
    conv_kernel_size: usize,
    ssm_state_dim: usize,
    intermediate_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32 = 10000.0,
    max_position_embeddings: usize = 4096,
    head_dim: ?usize = null,
    ssm_pattern: []const u8 = "MA",

    pub fn getHeadDim(self: NemotronHConfig) usize {
        return self.head_dim orelse (self.hidden_size / self.num_attention_heads);
    }

    pub fn getLayerType(self: NemotronHConfig, layer_idx: usize) LayerType {
        const pattern_len = self.ssm_pattern.len;
        const ch = self.ssm_pattern[layer_idx % pattern_len];
        return switch (ch) {
            'M', 'm' => .mamba,
            'G', 'g' => .gated_delta,
            else => .attention,
        };
    }
};

// ============================================================
// Layer Type
// ============================================================

pub const LayerType = enum {
    mamba,
    gated_delta,
    attention,
};

// ============================================================
// Mamba SSM Block
// ============================================================

pub const MambaBlock = struct {
    ctx: EagerContext,
    config: *const NemotronHConfig,

    in_proj: Array,
    conv_weight: Array,
    out_proj: Array,
    dt_proj: Array,
    A: Array,
    D: Array,

    pub fn deinit(self: *MambaBlock, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.in_proj.deinit();
        self.conv_weight.deinit();
        self.out_proj.deinit();
        self.dt_proj.deinit();
        self.A.deinit();
        self.D.deinit();
    }

    pub fn forward(self: *MambaBlock, x: Array) !Array {
        const shape = x.shape();
        const batch = @as(usize, @intCast(shape[0]));
        const seq_len = @as(usize, @intCast(shape[1]));
        const ssm_dim = self.config.ssm_dim;

        // 1. Linear projection: x -> [x_ssm, z]
        const in_proj_t = try ops.transpose(self.ctx, self.in_proj);
        defer in_proj_t.deinit();
        const xz = try ops.matmul(self.ctx, x, in_proj_t);
        defer xz.deinit();

        // 2. Split
        const x_ssm = try ops.slice(self.ctx, xz, &[_]i32{ 0, 0, 0 }, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(ssm_dim) }, &[_]i32{ 1, 1, 1 });
        defer x_ssm.deinit();

        const z = try ops.slice(self.ctx, xz, &[_]i32{ 0, 0, @intCast(ssm_dim) }, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(2 * ssm_dim) }, &[_]i32{ 1, 1, 1 });
        defer z.deinit();

        // 3. Causal Conv1d + SiLU
        // MLX conv1d expects: input [N, L, C], weight [C_out, K, C_in/groups]
        const padding = @divTrunc(@as(i32, @intCast(self.config.conv_kernel_size - 1)), 2);
        const x_conv = try conv.conv1d(self.ctx, x_ssm, self.conv_weight, 1, padding, 1, @intCast(ssm_dim));
        defer x_conv.deinit();

        const x_act = try activations.silu(self.ctx, x_conv);
        defer x_act.deinit();

        // 4. Simplified selective scan
        const dt_proj_t = try ops.transpose(self.ctx, self.dt_proj);
        defer dt_proj_t.deinit();
        const dt_raw = try ops.matmul(self.ctx, x_act, dt_proj_t);
        defer dt_raw.deinit();

        const dt_with_A = try ops.add(self.ctx, dt_raw, self.A);
        defer dt_with_A.deinit();
        const dt_with_D = try ops.add(self.ctx, dt_with_A, self.D);
        defer dt_with_D.deinit();
        const dt = try activations.softplus(self.ctx, dt_with_D);
        defer dt.deinit();

        const y = try ops.multiply(self.ctx, x_act, dt);
        defer y.deinit();

        // 5. Gating
        const y_gated = try ops.multiply(self.ctx, y, z);
        defer y_gated.deinit();

        // 6. Output projection
        const out_proj_t = try ops.transpose(self.ctx, self.out_proj);
        defer out_proj_t.deinit();
        return ops.matmul(self.ctx, y_gated, out_proj_t);
    }
};

// ============================================================
// Gated DeltaNet SSM Block
// ============================================================

pub const GatedDeltaNetBlock = struct {
    ctx: EagerContext,
    config: *const NemotronHConfig,

    gate_proj: Array,
    x_proj: Array,
    delta_proj: Array,
    out_proj: Array,

    pub fn deinit(self: *GatedDeltaNetBlock, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.gate_proj.deinit();
        self.x_proj.deinit();
        self.delta_proj.deinit();
        self.out_proj.deinit();
    }

    pub fn forward(self: *GatedDeltaNetBlock, x: Array) !Array {
        const gate_proj_t = try ops.transpose(self.ctx, self.gate_proj);
        defer gate_proj_t.deinit();
        const gate_linear = try ops.matmul(self.ctx, x, gate_proj_t);
        defer gate_linear.deinit();
        const gate = try ops.sigmoid(self.ctx, gate_linear);
        defer gate.deinit();

        const x_proj_t = try ops.transpose(self.ctx, self.x_proj);
        defer x_proj_t.deinit();
        const x_p = try ops.matmul(self.ctx, x, x_proj_t);
        defer x_p.deinit();

        const delta_proj_t = try ops.transpose(self.ctx, self.delta_proj);
        defer delta_proj_t.deinit();
        const delta_linear = try ops.matmul(self.ctx, x, delta_proj_t);
        defer delta_linear.deinit();
        const delta = try activations.silu(self.ctx, delta_linear);
        defer delta.deinit();

        const state = try ops.multiply(self.ctx, x_p, delta);
        defer state.deinit();

        const out_proj_t = try ops.transpose(self.ctx, self.out_proj);
        defer out_proj_t.deinit();
        const out_linear = try ops.matmul(self.ctx, state, out_proj_t);
        defer out_linear.deinit();
        return ops.multiply(self.ctx, gate, out_linear);
    }
};

// ============================================================
// Attention
// ============================================================

pub const NemotronHAttention = struct {
    ctx: EagerContext,
    config: *const NemotronHConfig,

    wq: Array,
    wk: Array,
    wv: Array,
    wo: Array,
    rope: nn.RoPE,

    pub fn deinit(self: *NemotronHAttention, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.wq.deinit();
        self.wk.deinit();
        self.wv.deinit();
        self.wo.deinit();
        self.rope.cos_cache.deinit();
        self.rope.sin_cache.deinit();
    }

    pub fn forward(self: *NemotronHAttention, x: Array, mask: ?Array, cache: ?kvcache.KVCacheStrategy) !Array {
        const shape = x.shape();
        const batch = @as(usize, @intCast(shape[0]));
        const seq_len = @as(usize, @intCast(shape[1]));
        const num_heads = self.config.num_attention_heads;
        const head_dim = self.config.getHeadDim();

        const wq_t = try ops.transpose(self.ctx, self.wq);
        defer wq_t.deinit();
        var q = try ops.matmul(self.ctx, x, wq_t);
        defer q.deinit();

        const wk_t = try ops.transpose(self.ctx, self.wk);
        defer wk_t.deinit();
        var k = try ops.matmul(self.ctx, x, wk_t);
        defer k.deinit();

        const wv_t = try ops.transpose(self.ctx, self.wv);
        defer wv_t.deinit();
        var v = try ops.matmul(self.ctx, x, wv_t);
        defer v.deinit();

        const q_rs = try ops.reshape(self.ctx, q, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_heads), @intCast(head_dim) });
        defer q_rs.deinit();
        const k_rs = try ops.reshape(self.ctx, k, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_heads), @intCast(head_dim) });
        defer k_rs.deinit();
        const v_rs = try ops.reshape(self.ctx, v, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_heads), @intCast(head_dim) });
        defer v_rs.deinit();

        const q_t = try ops.transposeAxes(self.ctx, q_rs, &[_]i32{ 0, 2, 1, 3 });
        defer q_t.deinit();
        const k_t = try ops.transposeAxes(self.ctx, k_rs, &[_]i32{ 0, 2, 1, 3 });
        defer k_t.deinit();
        const v_t = try ops.transposeAxes(self.ctx, v_rs, &[_]i32{ 0, 2, 1, 3 });
        defer v_t.deinit();

        var q_rot = try self.rope.apply(q_t);
        defer q_rot.deinit();
        var k_rot = try self.rope.apply(k_t);
        defer k_rot.deinit();

        var k_final = k_rot;
        var v_final = v_t;
        if (cache) |kv_cache| {
            const kv = try kv_cache.updateAndFetch(k_rot, v_t, self.ctx.stream.inner);
            k_final = kv.keys;
            v_final = kv.values;
        }

        const k_t2 = try ops.transposeAxes(self.ctx, k_final, &[_]i32{ 0, 1, 3, 2 });
        defer k_t2.deinit();

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        var scores = try ops.matmul(self.ctx, q_rot, k_t2);
        defer scores.deinit();
        var scaled_scores = try ops.multiply(self.ctx, scores, try ops.scalarF32(self.ctx, scale));
        defer scaled_scores.deinit();

        var masked_scores = scaled_scores;
        if (mask) |m| {
            const masked = try ops.add(self.ctx, scaled_scores, m);
            masked_scores.deinit();
            masked_scores = masked;
        }

        var attn_weights = try ops.softmax(self.ctx, masked_scores, &[_]i32{-1});
        defer attn_weights.deinit();

        var attn_out = try ops.matmul(self.ctx, attn_weights, v_final);
        defer attn_out.deinit();

        const attn_t = try ops.transposeAxes(self.ctx, attn_out, &[_]i32{ 0, 2, 1, 3 });
        defer attn_t.deinit();
        const attn_flat = try ops.reshape(self.ctx, attn_t, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_heads * head_dim) });
        defer attn_flat.deinit();

        const wo_t = try ops.transpose(self.ctx, self.wo);
        defer wo_t.deinit();
        return ops.matmul(self.ctx, attn_flat, wo_t);
    }
};

// ============================================================
// Layer wrappers
// ============================================================

pub const MambaLayer = struct {
    ctx: EagerContext,
    input_norm: nn.RMSNorm,
    block: MambaBlock,
    post_norm: nn.RMSNorm,

    pub fn deinit(self: *MambaLayer, allocator: std.mem.Allocator) void {
        self.input_norm.weight.deinit();
        self.block.deinit(allocator);
        self.post_norm.weight.deinit();
    }

    pub fn forward(self: *MambaLayer, x: Array, _: ?Array, _: ?kvcache.KVCacheStrategy) !Array {
        const normed = try self.input_norm.forward(x);
        defer normed.deinit();
        const block_out = try self.block.forward(normed);
        defer block_out.deinit();
        const h = try ops.add(self.ctx, x, block_out);
        defer h.deinit();
        return self.post_norm.forward(h);
    }
};

pub const GatedDeltaLayer = struct {
    ctx: EagerContext,
    input_norm: nn.RMSNorm,
    block: GatedDeltaNetBlock,
    post_norm: nn.RMSNorm,

    pub fn deinit(self: *GatedDeltaLayer, allocator: std.mem.Allocator) void {
        self.input_norm.weight.deinit();
        self.block.deinit(allocator);
        self.post_norm.weight.deinit();
    }

    pub fn forward(self: *GatedDeltaLayer, x: Array, _: ?Array, _: ?kvcache.KVCacheStrategy) !Array {
        const normed = try self.input_norm.forward(x);
        defer normed.deinit();
        const block_out = try self.block.forward(normed);
        defer block_out.deinit();
        const h = try ops.add(self.ctx, x, block_out);
        defer h.deinit();
        return self.post_norm.forward(h);
    }
};

pub const AttentionLayer = struct {
    ctx: EagerContext,
    input_norm: nn.RMSNorm,
    attn: NemotronHAttention,
    post_norm: nn.RMSNorm,

    pub fn deinit(self: *AttentionLayer, allocator: std.mem.Allocator) void {
        self.input_norm.weight.deinit();
        self.attn.deinit(allocator);
        self.post_norm.weight.deinit();
    }

    pub fn forward(self: *AttentionLayer, x: Array, mask: ?Array, cache: ?kvcache.KVCacheStrategy) !Array {
        const normed = try self.input_norm.forward(x);
        defer normed.deinit();
        const attn_out = try self.attn.forward(normed, mask, cache);
        defer attn_out.deinit();
        const h = try ops.add(self.ctx, x, attn_out);
        defer h.deinit();
        return self.post_norm.forward(h);
    }
};

// ============================================================
// Nemotron-H Layer (union)
// ============================================================

pub const NemotronHLayer = union(LayerType) {
    mamba: MambaLayer,
    gated_delta: GatedDeltaLayer,
    attention: AttentionLayer,

    pub fn deinit(self: *NemotronHLayer, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .mamba => |*layer| layer.deinit(allocator),
            .gated_delta => |*layer| layer.deinit(allocator),
            .attention => |*layer| layer.deinit(allocator),
        }
    }

    pub fn forward(self: *NemotronHLayer, x: Array, mask: ?Array, cache: ?kvcache.KVCacheStrategy) !Array {
        switch (self.*) {
            .mamba => |*layer| return try layer.forward(x, mask, cache),
            .gated_delta => |*layer| return try layer.forward(x, mask, cache),
            .attention => |*layer| return try layer.forward(x, mask, cache),
        }
    }
};

// ============================================================
// Full Nemotron-H Model
// ============================================================

pub const NemotronHModel = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    config: NemotronHConfig,

    embed_tokens: nn.Embedding,
    layers: std.ArrayList(NemotronHLayer),
    norm: nn.RMSNorm,
    lm_head: Array,

    pub fn deinit(self: *NemotronHModel) void {
        self.embed_tokens.weight.deinit();
        for (self.layers.items) |*layer| {
            layer.deinit(self.allocator);
        }
        self.layers.deinit(self.allocator);
        self.norm.weight.deinit();
        self.lm_head.deinit();
    }

    pub fn forward(self: *NemotronHModel, input_ids: Array, mask: ?Array, caches: ?[]kvcache.KVCacheStrategy) !Array {
        var arena = ScopedArrayArena.init(self.allocator);
        defer arena.deinit();

        var hidden = try arena.track(try self.embed_tokens.forward(input_ids));

        for (self.layers.items, 0..) |*layer, i| {
            const cache = if (caches) |cache_list| (if (i < cache_list.len) cache_list[i] else null) else null;
            hidden = try arena.track(try layer.forward(hidden, mask, cache));
        }

        const normed = try arena.track(try self.norm.forward(hidden));

        const lm_head_t = try arena.track(try ops.transpose(self.ctx, self.lm_head));
        return ops.matmul(self.ctx, normed, lm_head_t);
    }
};

// ============================================================
// Tests
// ============================================================

test "NemotronHConfig getLayerType" {
    const config = NemotronHConfig{
        .vocab_size = 32,
        .hidden_size = 16,
        .num_layers = 4,
        .num_attention_heads = 2,
        .num_key_value_heads = 2,
        .ssm_dim = 8,
        .conv_kernel_size = 3,
        .ssm_state_dim = 8,
        .intermediate_size = 32,
        .rms_norm_eps = 1e-5,
        .ssm_pattern = "MGA",
    };

    try std.testing.expectEqual(LayerType.mamba, config.getLayerType(0));
    try std.testing.expectEqual(LayerType.gated_delta, config.getLayerType(1));
    try std.testing.expectEqual(LayerType.attention, config.getLayerType(2));
    try std.testing.expectEqual(LayerType.mamba, config.getLayerType(3));
}

test "MambaBlock forward shape check" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const config = NemotronHConfig{
        .vocab_size = 8,
        .hidden_size = 8,
        .num_layers = 1,
        .num_attention_heads = 2,
        .num_key_value_heads = 2,
        .ssm_dim = 4,
        .conv_kernel_size = 3,
        .ssm_state_dim = 4,
        .intermediate_size = 16,
        .rms_norm_eps = 1e-5,
    };

    const hidden: i32 = @intCast(config.hidden_size);
    const ssm: i32 = @intCast(config.ssm_dim);

    var block = MambaBlock{
        .ctx = ctx,
        .config = &config,
        .in_proj = try array_mod.zeros(allocator, &[_]i32{ 2 * ssm, hidden }, .float32),
        .conv_weight = try array_mod.zeros(allocator, &[_]i32{ ssm, @intCast(config.conv_kernel_size), 1 }, .float32),
        .out_proj = try array_mod.zeros(allocator, &[_]i32{ hidden, ssm }, .float32),
        .dt_proj = try array_mod.zeros(allocator, &[_]i32{ ssm, ssm }, .float32),
        .A = try array_mod.zeros(allocator, &[_]i32{ssm}, .float32),
        .D = try array_mod.zeros(allocator, &[_]i32{ssm}, .float32),
    };
    defer block.deinit(allocator);

    const x = try array_mod.zeros(allocator, &[_]i32{ 1, 4, hidden }, .float32);
    defer x.deinit();

    const out = try block.forward(x);
    defer out.deinit();

    const shape = out.shape();
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 4), shape[1]);
    try std.testing.expectEqual(hidden, shape[2]);
}

test "NemotronHAttention forward shape check" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const config = NemotronHConfig{
        .vocab_size = 8,
        .hidden_size = 8,
        .num_layers = 1,
        .num_attention_heads = 2,
        .num_key_value_heads = 2,
        .ssm_dim = 4,
        .conv_kernel_size = 3,
        .ssm_state_dim = 4,
        .intermediate_size = 16,
        .rms_norm_eps = 1e-5,
    };

    const hidden: i32 = @intCast(config.hidden_size);
    const heads: i32 = @intCast(config.num_attention_heads);
    const head_dim: i32 = @divTrunc(hidden, heads);

    var attn = NemotronHAttention{
        .ctx = ctx,
        .config = &config,
        .wq = try array_mod.zeros(allocator, &[_]i32{ heads * head_dim, hidden }, .float32),
        .wk = try array_mod.zeros(allocator, &[_]i32{ heads * head_dim, hidden }, .float32),
        .wv = try array_mod.zeros(allocator, &[_]i32{ heads * head_dim, hidden }, .float32),
        .wo = try array_mod.zeros(allocator, &[_]i32{ hidden, heads * head_dim }, .float32),
        .rope = try nn.RoPE.init(ctx, @intCast(head_dim), 64, 10000.0),
    };
    defer attn.deinit(allocator);

    const x = try array_mod.zeros(allocator, &[_]i32{ 1, 4, hidden }, .float32);
    defer x.deinit();

    const out = try attn.forward(x, null, null);
    defer out.deinit();

    const shape = out.shape();
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 4), shape[1]);
    try std.testing.expectEqual(hidden, shape[2]);
}
