/// MiniMax model architecture.
///
/// Simplified MoE transformer with standard GQA attention.
/// Based on DeepSeek V4's MoE framework but without MLA, CSA/HCA, or mHC.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const reduce_mod = @import("mlx").reduce;
const cmp_mod = @import("mlx").comparison;
const nn = @import("mlx").nn;
const kvcache = @import("../kvcache.zig");
const array_arena_mod = @import("mlx").array_arena;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const ScopedArrayArena = array_arena_mod.ScopedArrayArena;

// ============================================================
// Configuration
// ============================================================

pub const MiniMaxConfig = struct {
    vocab_size: usize = 32000,
    hidden_size: usize = 4096,
    num_hidden_layers: usize = 32,
    num_attention_heads: usize = 32,
    num_key_value_heads: usize = 8,
    head_dim: usize = 128,
    intermediate_size: usize = 14336,
    num_experts: usize = 8,
    num_experts_per_tok: usize = 2,
    num_shared_experts: usize = 1,
    rms_norm_eps: f32 = 1e-5,
    rope_theta: f32 = 10000.0,
    max_position_embeddings: usize = 4096,
};

// ============================================================
// RoPE (simplified, supports start_pos for KV cache)
// ============================================================

pub const MiniMaxRoPE = struct {
    ctx: EagerContext,
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
    cos_cache: Array,
    sin_cache: Array,

    pub fn init(ctx: EagerContext, head_dim: usize, max_seq_len: usize, theta: f32) !MiniMaxRoPE {
        std.debug.assert(head_dim % 2 == 0);
        const half_dim = head_dim / 2;

        const cos_cache = try array_mod.zeros(ctx.allocator, &[_]i32{ @intCast(max_seq_len), @intCast(half_dim) }, .float32);
        const sin_cache = try array_mod.zeros(ctx.allocator, &[_]i32{ @intCast(max_seq_len), @intCast(half_dim) }, .float32);

        const cos_data = @constCast(try cos_cache.dataPtr(f32))[0 .. max_seq_len * half_dim];
        const sin_data = @constCast(try sin_cache.dataPtr(f32))[0 .. max_seq_len * half_dim];

        for (0..max_seq_len) |pos| {
            for (0..half_dim) |i| {
                const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;
                const idx = pos * half_dim + i;
                cos_data[idx] = @cos(angle);
                sin_data[idx] = @sin(angle);
            }
        }

        return MiniMaxRoPE{
            .ctx = ctx,
            .head_dim = head_dim,
            .max_seq_len = max_seq_len,
            .theta = theta,
            .cos_cache = cos_cache,
            .sin_cache = sin_cache,
        };
    }

    pub fn deinit(self: *MiniMaxRoPE) void {
        self.cos_cache.deinit();
        self.sin_cache.deinit();
    }

    /// Apply RoPE to input of shape [B, H, S, head_dim].
    pub fn apply(self: *MiniMaxRoPE, input: Array, start_pos: usize) !Array {
        const shape = input.shape();
        const seq_len = @as(usize, @intCast(shape[2]));
        const head_dim = self.head_dim;
        const half_dim = head_dim / 2;

        const out = try array_mod.zeros(self.ctx.allocator, shape, input.dtype());
        const src = try input.dataSliceMut(f32);
        const dst = try out.dataSliceMut(f32);
        const cos_data = try self.cos_cache.dataSliceMut(f32);
        const sin_data = try self.sin_cache.dataSliceMut(f32);

        const batch = @as(usize, @intCast(shape[0]));
        const num_heads = @as(usize, @intCast(shape[1]));

        for (0..batch) |b| {
            for (0..num_heads) |h| {
                for (0..seq_len) |s| {
                    const pos = start_pos + s;
                    if (pos >= self.max_seq_len) continue;
                    const base_idx = ((b * num_heads + h) * seq_len + s) * head_dim;
                    const cache_idx = pos * half_dim;
                    for (0..half_dim) |i| {
                        const x1 = src[base_idx + i];
                        const x2 = src[base_idx + half_dim + i];
                        const c_ = cos_data[cache_idx + i];
                        const s_ = sin_data[cache_idx + i];
                        dst[base_idx + i] = x1 * c_ - x2 * s_;
                        dst[base_idx + half_dim + i] = x1 * s_ + x2 * c_;
                    }
                }
            }
        }

        return out;
    }
};

// ============================================================
// Attention (standard GQA)
// ============================================================

pub const MiniMaxAttention = struct {
    ctx: EagerContext,
    config: *const MiniMaxConfig,

    wq: Array,
    wk: Array,
    wv: Array,
    wo: Array,
    rope: MiniMaxRoPE,

    pub fn deinit(self: *MiniMaxAttention) void {
        self.wq.deinit();
        self.wk.deinit();
        self.wv.deinit();
        self.wo.deinit();
        self.rope.deinit();
    }

    pub fn forward(self: *MiniMaxAttention, x: Array, mask: ?Array, cache: ?kvcache.KVCacheStrategy, start_pos: usize) !Array {
        const shape = x.shape();
        const batch = @as(usize, @intCast(shape[0]));
        const seq_len = @as(usize, @intCast(shape[1]));
        const num_heads = self.config.num_attention_heads;
        const num_kv_heads = self.config.num_key_value_heads;
        const head_dim = self.config.head_dim;

        // Linear projections
        const wq_t = try ops.transpose(self.ctx, self.wq);
        defer wq_t.deinit();
        const q = try ops.matmul(self.ctx, x, wq_t);
        defer q.deinit();

        const wk_t = try ops.transpose(self.ctx, self.wk);
        defer wk_t.deinit();
        const k = try ops.matmul(self.ctx, x, wk_t);
        defer k.deinit();

        const wv_t = try ops.transpose(self.ctx, self.wv);
        defer wv_t.deinit();
        const v = try ops.matmul(self.ctx, x, wv_t);
        defer v.deinit();

        // Reshape to multi-head: [B, S, H, D]
        const q_rs = try ops.reshape(self.ctx, q, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_heads), @intCast(head_dim) });
        const k_rs = try ops.reshape(self.ctx, k, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_kv_heads), @intCast(head_dim) });
        const v_rs = try ops.reshape(self.ctx, v, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_kv_heads), @intCast(head_dim) });
        defer q_rs.deinit();
        defer k_rs.deinit();
        defer v_rs.deinit();

        // Transpose for attention: [B, H, S, D]
        const q_t = try ops.transposeAxes(self.ctx, q_rs, &[_]i32{ 0, 2, 1, 3 });
        const k_t = try ops.transposeAxes(self.ctx, k_rs, &[_]i32{ 0, 2, 1, 3 });
        const v_t = try ops.transposeAxes(self.ctx, v_rs, &[_]i32{ 0, 2, 1, 3 });
        defer q_t.deinit();
        defer k_t.deinit();
        defer v_t.deinit();

        // Apply RoPE
        const q_rot = try self.rope.apply(q_t, start_pos);
        const k_rot = try self.rope.apply(k_t, start_pos);
        defer q_rot.deinit();
        defer k_rot.deinit();

        // KV Cache
        var k_final = k_rot;
        var v_final = v_t;
        if (cache) |kv_cache| {
            const kv = try kv_cache.updateAndFetch(k_rot, v_t, self.ctx.stream.inner);
            k_final = kv.keys;
            v_final = kv.values;
        }
        defer if (cache != null) {
            k_final.deinit();
            v_final.deinit();
        };

        // Broadcast K/V heads for GQA
        const num_kv_groups = num_heads / num_kv_heads;
        var k_bc = k_final;
        var v_bc = v_final;
        if (num_kv_groups > 1) {
            k_bc = try repeatHeads(self.ctx, k_final, num_kv_groups);
            v_bc = try repeatHeads(self.ctx, v_final, num_kv_groups);
        }
        defer if (num_kv_groups > 1) {
            k_bc.deinit();
            v_bc.deinit();
        };

        // Scaled dot-product attention
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const k_t2 = try ops.transposeAxes(self.ctx, k_bc, &[_]i32{ 0, 1, 3, 2 });
        defer k_t2.deinit();

        var scores = try ops.matmul(self.ctx, q_rot, k_t2);
        defer scores.deinit();

        const scale_arr = try ops.scalarF32(self.ctx, scale);
        defer scale_arr.deinit();
        var scaled_scores = try ops.multiply(self.ctx, scores, scale_arr);
        defer scaled_scores.deinit();

        // Apply mask
        var masked_scores = scaled_scores;
        if (mask) |m| {
            const masked = try ops.add(self.ctx, scaled_scores, m);
            masked_scores = masked;
        }

        const attn_weights = try ops.softmax(self.ctx, masked_scores, &[_]i32{-1});
        defer attn_weights.deinit();

        const attn_out = try ops.matmul(self.ctx, attn_weights, v_bc);
        defer attn_out.deinit();

        // Reshape back: [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        const attn_t = try ops.transposeAxes(self.ctx, attn_out, &[_]i32{ 0, 2, 1, 3 });
        defer attn_t.deinit();

        const attn_flat = try ops.reshape(self.ctx, attn_t, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_heads * head_dim) });
        defer attn_flat.deinit();

        // Output projection
        const wo_t = try ops.transpose(self.ctx, self.wo);
        defer wo_t.deinit();
        return ops.matmul(self.ctx, attn_flat, wo_t);
    }
};

fn repeatHeads(ctx: EagerContext, tensor: Array, repeats: usize) !Array {
    const shape = tensor.shape();
    const b = shape[0];
    const h = shape[1];
    const s = shape[2];
    const d = shape[3];

    const expanded = try ops.expandDims(ctx, tensor, 2);
    defer expanded.deinit();

    const tiled = try ops.tile(ctx, expanded, &[_]i32{ 1, 1, @intCast(repeats), 1, 1 });
    defer tiled.deinit();

    const reshaped = try ops.reshape(ctx, tiled, &[_]i32{ b, h * @as(i32, @intCast(repeats)), s, d });
    return ops.copy(ctx, reshaped);
}

// ============================================================
// MoE Expert (SwiGLU)
// ============================================================

pub const MiniMaxExpert = struct {
    ctx: EagerContext,
    gate_proj: Array,
    up_proj: Array,
    down_proj: Array,

    pub fn deinit(self: *MiniMaxExpert) void {
        self.gate_proj.deinit();
        self.up_proj.deinit();
        self.down_proj.deinit();
    }

    pub fn forward(self: *MiniMaxExpert, x: Array, stream: c.c.mlx_stream) !Array {
        _ = stream;
        const gate_t = try ops.transpose(self.ctx, self.gate_proj);
        defer gate_t.deinit();
        const gate_linear = try ops.matmul(self.ctx, x, gate_t);
        defer gate_linear.deinit();

        const gate_sigmoid = try ops.sigmoid(self.ctx, gate_linear);
        defer gate_sigmoid.deinit();
        const gate_act = try ops.multiply(self.ctx, gate_linear, gate_sigmoid);
        defer gate_act.deinit();

        const up_t = try ops.transpose(self.ctx, self.up_proj);
        defer up_t.deinit();
        const up = try ops.matmul(self.ctx, x, up_t);
        defer up.deinit();

        const hidden = try ops.multiply(self.ctx, gate_act, up);
        defer hidden.deinit();

        const down_t = try ops.transpose(self.ctx, self.down_proj);
        defer down_t.deinit();
        return ops.matmul(self.ctx, hidden, down_t);
    }
};

// ============================================================
// MoE Gate
// ============================================================

pub const MiniMaxGate = struct {
    ctx: EagerContext,
    weight: Array,
    topk: usize,
    num_experts: usize,

    pub fn deinit(self: *MiniMaxGate) void {
        self.weight.deinit();
    }

    pub fn forward(self: *MiniMaxGate, hidden_states: Array, stream: c.c.mlx_stream) !struct { Array, Array } {
        const shape = hidden_states.shape();
        const batch = @as(usize, @intCast(shape[0]));
        const seq_len = if (shape.len >= 3) @as(usize, @intCast(shape[1])) else 1;
        const dim = @as(usize, @intCast(shape[shape.len - 1]));

        // Flatten to [B*S, dim]
        const flat = try ops.reshape(self.ctx, hidden_states, &[_]i32{ @intCast(batch * seq_len), @intCast(dim) });
        defer flat.deinit();

        // Scores: flat @ weight^T -> [B*S, num_experts]
        const weight_t = try ops.transpose(self.ctx, self.weight);
        defer weight_t.deinit();
        const logits = try ops.matmul(self.ctx, flat, weight_t);
        defer logits.deinit();

        const scores = try ops.softmax(self.ctx, logits, &[_]i32{-1});
        defer scores.deinit();

        // Top-k selection via argsort
        var sorted_handle = c.c.mlx_array_new();
        try c.check(c.c.mlx_argsort_axis(&sorted_handle, scores.inner, -1, stream));
        const sorted = Array.fromHandle(sorted_handle);
        defer sorted.deinit();

        const start_y = @as(i32, @intCast(self.num_experts - @as(usize, @intCast(self.topk))));
        const stop_y = @as(i32, @intCast(self.num_experts));
        const batch_size = sorted.shape()[0];
        const indices = try ops.slice(self.ctx, sorted, &[_]i32{ 0, start_y }, &[_]i32{ batch_size, stop_y }, &[_]i32{ 1, 1 });
        try indices.eval();

        // Gather weights at selected indices
        var res = c.c.mlx_array_new();
        try c.check(c.c.mlx_take_along_axis(&res, scores.inner, indices.inner, 1, self.ctx.stream.inner));
        const weights = Array.fromHandle(res);

        return .{ weights, indices };
    }
};

// ============================================================
// MoE Layer
// ============================================================

pub const MiniMaxMoE = struct {
    ctx: EagerContext,
    gate: MiniMaxGate,
    experts: []MiniMaxExpert,
    shared_expert: MiniMaxExpert,
    num_experts: usize,
    topk: usize,

    pub fn deinit(self: *MiniMaxMoE) void {
        self.gate.deinit();
        for (self.experts) |*expert| {
            expert.deinit();
        }
        self.ctx.allocator.free(self.experts);
        self.shared_expert.deinit();
    }

    pub fn forward(self: *MiniMaxMoE, hidden_states: Array, stream: c.c.mlx_stream) !Array {
        const shape = hidden_states.shape();
        const batch = @as(usize, @intCast(shape[0]));
        const seq_len = @as(usize, @intCast(shape[1]));
        const dim = @as(usize, @intCast(shape[2]));

        // Flatten to [B*S, dim]
        const flat = try ops.reshape(self.ctx, hidden_states, &[_]i32{ @intCast(batch * seq_len), @intCast(dim) });
        defer flat.deinit();

        // Gate routing
        const weights, const indices = try self.gate.forward(flat, stream);
        defer weights.deinit();
        defer indices.deinit();

        // Initialize output
        const output_shape = [_]i32{ @intCast(batch * seq_len), @intCast(dim) };
        var y = try array_mod.zeros(self.ctx.allocator, &output_shape, .float32);
        errdefer y.deinit();

        // Build expert masks: [N, topk, num_experts]
        const indices_exp = try ops.expandDims(self.ctx, indices, 2);
        defer indices_exp.deinit();

        var expert_range_data = try self.ctx.allocator.alloc(f32, self.num_experts);
        defer self.ctx.allocator.free(expert_range_data);
        for (0..self.num_experts) |i| expert_range_data[i] = @floatFromInt(i);
        const expert_range = try Array.fromData(self.ctx.allocator, f32, expert_range_data, &[_]i32{ 1, 1, @intCast(self.num_experts) });
        defer expert_range.deinit();

        const all_masks = try cmp_mod.equal(self.ctx, indices_exp, expert_range);
        defer all_masks.deinit();

        // Route to each expert
        for (0..self.num_experts) |eid| {
            const expert = &self.experts[eid];

            // Extract mask for this expert
            const mask = try ops.slice(self.ctx, all_masks, &[_]i32{ 0, 0, @intCast(eid) }, &[_]i32{ @intCast(batch * seq_len), @intCast(self.topk), @intCast(eid + 1) }, &[_]i32{});
            defer mask.deinit();

            const has_tokens = try reduce_mod.sumAxes(self.ctx, mask, &[_]i32{ 0, 1, 2 }, false);
            defer has_tokens.deinit();
            const has_tokens_val = try has_tokens.dataPtr(f32);
            if (has_tokens_val[0] == 0.0) continue;

            for (0..self.topk) |k| {
                const k_slice = try ops.slice(self.ctx, mask, &[_]i32{ 0, @intCast(k), 0 }, &[_]i32{ @intCast(batch * seq_len), @intCast(k + 1), 1 }, &[_]i32{});
                defer k_slice.deinit();
                const k_mask = try ops.reshape(self.ctx, k_slice, &[_]i32{@intCast(batch * seq_len)});
                defer k_mask.deinit();

                const mask_f32 = try ops.astype(self.ctx, k_mask, .float32);
                defer mask_f32.deinit();
                const mask_exp = try ops.expandDims(self.ctx, mask_f32, 1);
                defer mask_exp.deinit();

                const masked_input = try ops.multiply(self.ctx, flat, mask_exp);
                defer masked_input.deinit();

                const expert_out = try expert.forward(masked_input, stream);
                defer expert_out.deinit();

                const k_weight = try ops.slice(self.ctx, weights, &[_]i32{ 0, @intCast(k) }, &[_]i32{ @intCast(batch * seq_len), @intCast(k + 1) }, &[_]i32{});
                defer k_weight.deinit();
                const k_weight_flat = try ops.reshape(self.ctx, k_weight, &[_]i32{@intCast(batch * seq_len)});
                defer k_weight_flat.deinit();
                const k_weight_exp = try ops.expandDims(self.ctx, k_weight_flat, 1);
                defer k_weight_exp.deinit();

                const weighted_out = try ops.multiply(self.ctx, expert_out, k_weight_exp);
                defer weighted_out.deinit();

                const new_y = try ops.add(self.ctx, y, weighted_out);
                y.deinit();
                y = new_y;
            }
        }

        // Add shared expert
        const shared_out = try self.shared_expert.forward(flat, stream);
        defer shared_out.deinit();
        const final_out = try ops.add(self.ctx, y, shared_out);
        y.deinit();

        // Reshape back
        return ops.reshape(self.ctx, final_out, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(dim) });
    }
};

// ============================================================
// Transformer Block
// ============================================================

pub const MiniMaxTransformerBlock = struct {
    ctx: EagerContext,
    config: *const MiniMaxConfig,
    layer_idx: usize,

    input_layernorm: nn.RMSNorm,
    attention: MiniMaxAttention,
    post_attention_layernorm: nn.RMSNorm,
    moe: MiniMaxMoE,

    pub fn deinit(self: *MiniMaxTransformerBlock) void {
        self.input_layernorm.weight.deinit();
        self.attention.deinit();
        self.post_attention_layernorm.weight.deinit();
        self.moe.deinit();
    }

    pub fn forward(self: *MiniMaxTransformerBlock, x: Array, mask: ?Array, cache: ?kvcache.KVCacheStrategy, start_pos: usize, stream: c.c.mlx_stream) !Array {
        // Attention with residual
        const normed = try self.input_layernorm.forward(x);
        defer normed.deinit();
        const attn_out = try self.attention.forward(normed, mask, cache, start_pos);
        defer attn_out.deinit();
        const h = try ops.add(self.ctx, x, attn_out);
        defer h.deinit();

        // MoE with residual
        const mlp_normed = try self.post_attention_layernorm.forward(h);
        defer mlp_normed.deinit();
        const mlp_out = try self.moe.forward(mlp_normed, stream);
        defer mlp_out.deinit();

        return ops.add(self.ctx, h, mlp_out);
    }
};

// ============================================================
// Full Model
// ============================================================

pub const MiniMaxModel = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    config: MiniMaxConfig,
    embed_tokens: nn.Embedding,
    layers: []MiniMaxTransformerBlock,
    norm: nn.RMSNorm,
    lm_head: Array,

    pub fn deinit(self: *MiniMaxModel) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
        self.norm.weight.deinit();
        self.lm_head.deinit();
    }

    pub fn forward(self: *MiniMaxModel, input_ids: Array, mask: ?Array, caches: ?[]kvcache.KVCacheStrategy, start_pos: usize, stream: c.c.mlx_stream) !Array {
        var arena = ScopedArrayArena.init(self.allocator);
        defer arena.deinit();

        var hidden = try arena.track(try self.embed_tokens.forward(input_ids));

        for (self.layers, 0..) |*layer, i| {
            const cache = if (caches) |caches_arr| caches_arr[i] else null;
            hidden = try arena.track(try layer.forward(hidden, mask, cache, start_pos, stream));
        }

        const normed = try arena.track(try self.norm.forward(hidden));

        const lm_head_t = try arena.track(try ops.transpose(self.ctx, self.lm_head));
        const logits = try ops.matmul(self.ctx, normed, lm_head_t);
        return logits;
    }
};

// ============================================================
// Tests
// ============================================================

test "MiniMaxRoPE applies rotation" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    var rope = try MiniMaxRoPE.init(ctx, 4, 16, 10000.0);
    defer rope.deinit();

    const data = &[_]f32{ 1, 0, 0, 1, 1, 0, 0, 1 };
    const input = try Array.fromData(allocator, f32, data, &[_]i32{ 1, 1, 2, 4 });
    defer input.deinit();

    const output = try rope.apply(input, 0);
    defer output.deinit();

    try std.testing.expectEqual(@as(i32, 1), output.shape()[0]);
    try std.testing.expectEqual(@as(i32, 1), output.shape()[1]);
    try std.testing.expectEqual(@as(i32, 2), output.shape()[2]);
    try std.testing.expectEqual(@as(i32, 4), output.shape()[3]);
}

test "MiniMaxAttention forward shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const config = MiniMaxConfig{
        .vocab_size = 8,
        .hidden_size = 16,
        .num_hidden_layers = 1,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 4,
        .intermediate_size = 32,
        .num_experts = 2,
        .num_experts_per_tok = 1,
        .max_position_embeddings = 32,
    };

    const wq = try array_mod.zeros(allocator, &[_]i32{ 16, 16 }, .float32);
    const wk = try array_mod.zeros(allocator, &[_]i32{ 8, 16 }, .float32);
    const wv = try array_mod.zeros(allocator, &[_]i32{ 8, 16 }, .float32);
    const wo = try array_mod.zeros(allocator, &[_]i32{ 16, 16 }, .float32);

    const rope = try MiniMaxRoPE.init(ctx, 4, 32, 10000.0);

    var attn = MiniMaxAttention{
        .ctx = ctx,
        .config = &config,
        .wq = wq,
        .wk = wk,
        .wv = wv,
        .wo = wo,
        .rope = rope,
    };
    defer attn.deinit();

    const x = try array_mod.zeros(allocator, &[_]i32{ 1, 2, 16 }, .float32);
    defer x.deinit();

    const out = try attn.forward(x, null, null, 0);
    defer out.deinit();

    try std.testing.expectEqual(@as(i32, 1), out.shape()[0]);
    try std.testing.expectEqual(@as(i32, 2), out.shape()[1]);
    try std.testing.expectEqual(@as(i32, 16), out.shape()[2]);
}

test "MiniMaxModel forward shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const loader = @import("minimax_loader.zig");

    const config = MiniMaxConfig{
        .vocab_size = 8,
        .hidden_size = 16,
        .num_hidden_layers = 1,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 4,
        .intermediate_size = 32,
        .num_experts = 2,
        .num_experts_per_tok = 1,
        .num_shared_experts = 1,
        .max_position_embeddings = 32,
    };

    var weights = std.StringHashMap(Array).init(allocator);
    defer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
            allocator.free(entry.key_ptr.*);
        }
        weights.deinit();
    }

    const W = struct {
        fn put(w: *std.StringHashMap(Array), alloc: std.mem.Allocator, name: []const u8, shape: []const i32) !void {
            const key = try alloc.dupe(u8, name);
            const arr = try array_mod.zeros(alloc, shape, .float32);
            try w.put(key, arr);
        }
    };

    try W.put(&weights, allocator, "model.embed_tokens.weight", &[_]i32{ 8, 16 });
    try W.put(&weights, allocator, "model.norm.weight", &[_]i32{16});
    try W.put(&weights, allocator, "lm_head.weight", &[_]i32{ 8, 16 });

    try W.put(&weights, allocator, "model.layers.0.self_attn.q_proj.weight", &[_]i32{ 16, 16 });
    try W.put(&weights, allocator, "model.layers.0.self_attn.k_proj.weight", &[_]i32{ 8, 16 });
    try W.put(&weights, allocator, "model.layers.0.self_attn.v_proj.weight", &[_]i32{ 8, 16 });
    try W.put(&weights, allocator, "model.layers.0.self_attn.o_proj.weight", &[_]i32{ 16, 16 });

    try W.put(&weights, allocator, "model.layers.0.mlp.gate.weight", &[_]i32{ 2, 16 });
    try W.put(&weights, allocator, "model.layers.0.mlp.shared_expert.gate_proj.weight", &[_]i32{ 32, 16 });
    try W.put(&weights, allocator, "model.layers.0.mlp.shared_expert.up_proj.weight", &[_]i32{ 32, 16 });
    try W.put(&weights, allocator, "model.layers.0.mlp.shared_expert.down_proj.weight", &[_]i32{ 16, 32 });
    try W.put(&weights, allocator, "model.layers.0.mlp.experts.0.gate_proj.weight", &[_]i32{ 32, 16 });
    try W.put(&weights, allocator, "model.layers.0.mlp.experts.0.up_proj.weight", &[_]i32{ 32, 16 });
    try W.put(&weights, allocator, "model.layers.0.mlp.experts.0.down_proj.weight", &[_]i32{ 16, 32 });
    try W.put(&weights, allocator, "model.layers.0.mlp.experts.1.gate_proj.weight", &[_]i32{ 32, 16 });
    try W.put(&weights, allocator, "model.layers.0.mlp.experts.1.up_proj.weight", &[_]i32{ 32, 16 });
    try W.put(&weights, allocator, "model.layers.0.mlp.experts.1.down_proj.weight", &[_]i32{ 16, 32 });

    try W.put(&weights, allocator, "model.layers.0.input_layernorm.weight", &[_]i32{16});
    try W.put(&weights, allocator, "model.layers.0.post_attention_layernorm.weight", &[_]i32{16});

    var model = try loader.buildMiniMaxModel(allocator, &config, &weights, ctx, ctx.stream.inner);
    defer model.deinit();

    const input_ids = try Array.fromData(allocator, u32, &[_]u32{ 1, 2, 3 }, &[_]i32{ 1, 3 });
    defer input_ids.deinit();

    const logits = try model.forward(input_ids, null, null, 0, ctx.stream.inner);
    defer logits.deinit();

    const ls = logits.shape();
    try std.testing.expectEqual(@as(i32, 1), ls[0]);
    try std.testing.expectEqual(@as(i32, 3), ls[1]);
    try std.testing.expectEqual(@as(i32, 8), ls[2]);
}
