/// LLaMA / LLaMA-2 / Mistral / Qwen model architecture.
///
/// Supports:
///   - Grouped Query Attention (GQA)
///   - RoPE positional encoding
///   - RMSNorm
///   - SwiGLU MLP
///   - KV Cache integration
const std = @import("std");
const c = @import("../c.zig");
const ops = @import("../ops.zig");
const nn = @import("../ops/nn.zig");
const array_mod = @import("../array.zig");
const kvcache = @import("../kvcache.zig");
const tree_mod = @import("../tree.zig");
const lora_mod = @import("../lora.zig");
const array_arena_mod = @import("../array_arena.zig");
const quantize_mod = @import("../quantize.zig");
const distributed_mod = @import("../distributed.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const ScopedArrayArena = array_arena_mod.ScopedArrayArena;
const QuantizedWeight = quantize_mod.QuantizedWeight;
const TokenizerStrategy = @import("../tokenizer/interface.zig").TokenizerStrategy;

/// Options for the generate() function.
pub const GenerateOptions = struct {
    /// EOS token ID — generation stops when this token is sampled.
    eos_token_id: ?u32 = null,
    /// Optional tokenizer for streaming token output during generation.
    tokenizer: ?TokenizerStrategy = null,
};

// ============================================================
// Configuration
// ============================================================

pub const LlamaConfig = struct {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    intermediate_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32 = 10000.0,
    max_position_embeddings: usize = 4096,
    /// Explicit head dimension (overrides hidden_size / num_attention_heads when set)
    head_dim: ?usize = null,
    /// Quantization config from mlx-lm (0 means not quantized)
    quantize_bits: u8 = 0,
    quantize_group_size: i32 = 64,
    /// EOS token ID from config.json (null if not specified)
    eos_token_id: ?u32 = null,
    /// RoPE scaling config (Phi-4, LLaMA-3.1 long context)
    rope_scaling_type: ?[]const u8 = null,
    rope_scaling_factor: f32 = 1.0,
    rope_scaling_original_max_position_embeddings: ?usize = null,

    /// Get the effective head dimension.
    pub fn getHeadDim(self: LlamaConfig) usize {
        return self.head_dim orelse (self.hidden_size / self.num_attention_heads);
    }
};

// ============================================================
// Attention (with optional KV Cache)
// ============================================================

pub const LlamaAttention = struct {
    ctx: EagerContext,
    config: *const LlamaConfig,

    // Weights: [out_features, in_features]
    wq: Array,
    wk: Array,
    wv: Array,
    wo: Array,

    // Optional quantized weights (mlx-lm format)
    wq_quant: ?QuantizedWeight = null,
    wk_quant: ?QuantizedWeight = null,
    wv_quant: ?QuantizedWeight = null,
    wo_quant: ?QuantizedWeight = null,

    // Optional attention bias (Qwen2)
    wq_bias: ?Array = null,
    wk_bias: ?Array = null,
    wv_bias: ?Array = null,
    wo_bias: ?Array = null,

    q_norm: ?nn.RMSNorm,
    k_norm: ?nn.RMSNorm,
    rope: nn.RoPE,
    /// Optional distributed group for tensor parallelism.
    distributed_group: ?@import("../distributed.zig").DistributedGroup = null,

    pub fn setDistributedGroup(self: *LlamaAttention, group: ?@import("../distributed.zig").DistributedGroup) void {
        self.distributed_group = group;
    }

    pub fn deinit(self: *LlamaAttention, allocator: std.mem.Allocator) void {
        // Free quantized weights (owns data + scales + biases + original_shape).
        // When quantized, wq/wk/wv/wo point to qw.data — don't double-free.
        if (self.wq_quant) |qw| { var m = qw; m.deinit(allocator); } else self.wq.deinit();
        if (self.wk_quant) |qw| { var m = qw; m.deinit(allocator); } else self.wk.deinit();
        if (self.wv_quant) |qw| { var m = qw; m.deinit(allocator); } else self.wv.deinit();
        if (self.wo_quant) |qw| { var m = qw; m.deinit(allocator); } else self.wo.deinit();
    }

    /// Perform linear projection: x @ W^T + bias, using quantized matmul if available.
    fn linear(self: *LlamaAttention, x: Array, weight: Array, quant: ?QuantizedWeight, bias: ?Array) !Array {
        var result = if (quant) |qw|
            try quantize_mod.quantizedMatmul(self.ctx, x, qw, true)
        else
            try ops.matmul(self.ctx, x, try ops.transpose(self.ctx, weight));
        if (bias) |b| {
            const biased = try ops.add(self.ctx, result, b);
            result.deinit();
            return biased;
        }
        return result;
    }

    pub fn forward(self: *LlamaAttention, x: Array, mask: ?Array, cache: ?kvcache.KVCacheStrategy, lora: ?*lora_mod.LoRAModel, layer_idx: usize) !Array {
        const shape = x.shape();
        const batch = @as(usize, @intCast(shape[0]));
        const seq_len = @as(usize, @intCast(shape[1]));
        const num_heads = self.config.num_attention_heads;
        const num_kv_heads = self.config.num_key_value_heads;
        const head_dim = self.config.getHeadDim();

        // 1. Linear projections: x @ W^T (or quantized matmul)
        var q = try self.linear(x, self.wq, self.wq_quant, self.wq_bias);
        if (lora) |lm| {
            var buf: [64]u8 = undefined;
            const key = try std.fmt.bufPrint(&buf, "layers.{d}.attention.wq", .{layer_idx});
            if (lm.getAdapter(key)) |adapter| {
                const q_new = try adapter.apply(q, x);
                q.deinit();
                q = q_new;
            }
        }
        var k = try self.linear(x, self.wk, self.wk_quant, self.wk_bias);
        if (lora) |lm| {
            var buf: [64]u8 = undefined;
            const key = try std.fmt.bufPrint(&buf, "layers.{d}.attention.wk", .{layer_idx});
            if (lm.getAdapter(key)) |adapter| {
                const k_new = try adapter.apply(k, x);
                k.deinit();
                k = k_new;
            }
        }
        var v = try self.linear(x, self.wv, self.wv_quant, self.wv_bias);
        if (lora) |lm| {
            var buf: [64]u8 = undefined;
            const key = try std.fmt.bufPrint(&buf, "layers.{d}.attention.wv", .{layer_idx});
            if (lm.getAdapter(key)) |adapter| {
                const v_new = try adapter.apply(v, x);
                v.deinit();
                v = v_new;
            }
        }
        defer q.deinit();
        defer k.deinit();
        defer v.deinit();

        // 2. Reshape: [B, S, H*D] -> [B, S, H, D]
        const q_rs = try ops.reshape(self.ctx, q, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_heads), @intCast(head_dim) });
        const k_rs = try ops.reshape(self.ctx, k, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_kv_heads), @intCast(head_dim) });
        const v_rs = try ops.reshape(self.ctx, v, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_kv_heads), @intCast(head_dim) });
        defer q_rs.deinit();
        defer k_rs.deinit();
        defer v_rs.deinit();

        // 3. Transpose for attention: [B, S, H, D] -> [B, H, S, D]
        const q_t = try ops.transposeAxes(self.ctx, q_rs, &[_]i32{ 0, 2, 1, 3 });
        const k_t = try ops.transposeAxes(self.ctx, k_rs, &[_]i32{ 0, 2, 1, 3 });
        const v_t = try ops.transposeAxes(self.ctx, v_rs, &[_]i32{ 0, 2, 1, 3 });
        defer q_t.deinit();
        defer k_t.deinit();
        defer v_t.deinit();

        // 4. Apply RoPE (with Q/K norm order depending on architecture)
        // When q_norm/k_norm are present, apply them BEFORE RoPE (Qwen3/Qwen2 style)
        var q_pre_rope = q_t;
        var k_pre_rope = k_t;
        if (self.q_norm) |*qn| {
            const q_normed = try qn.forward(q_t);
            q_pre_rope = q_normed;
        }
        if (self.k_norm) |*kn| {
            const k_normed = try kn.forward(k_t);
            k_pre_rope = k_normed;
        }
        defer if (self.q_norm != null) q_pre_rope.deinit();
        defer if (self.k_norm != null) k_pre_rope.deinit();

        // RoPE offset = number of tokens already in KV cache (for incremental decoding)
        const rope_offset: i32 = if (cache) |kv_cache| @intCast(kv_cache.currentLen()) else 0;
        var q_rot = try self.rope.applyWithOffset(q_pre_rope, rope_offset);
        var k_rot = try self.rope.applyWithOffset(k_pre_rope, rope_offset);
        defer q_rot.deinit();
        defer k_rot.deinit();

        // 6. KV Cache
        var k_final = k_rot;
        var v_final = v_t;
        if (cache) |kv_cache| {
            const kv = try kv_cache.updateAndFetch(k_rot, v_t, self.ctx.stream.inner);
            k_final = kv.keys;
            v_final = kv.values;
        }

        // 7. Attention scores: Q @ K^T / sqrt(D)
        // [B, H, S, D] @ [B, H_kv, D, S] -> [B, H, S, S]
        // Need to handle GQA: k has fewer heads, broadcast
        const k_t2 = try ops.transposeAxes(self.ctx, k_final, &[_]i32{ 0, 1, 3, 2 });
        defer k_t2.deinit();

        // Broadcast K/V heads for GQA
        var k_bc = k_t2;
        var v_bc = v_final;
        const num_kv_groups = num_heads / num_kv_heads;
        if (num_kv_groups > 1) {
            // Repeat K and V heads: [B, H_kv, S, D] -> [B, H, S, D]
            k_bc = try repeatHeads(self.ctx, k_t2, num_kv_groups);
            v_bc = try repeatHeads(self.ctx, v_final, num_kv_groups);
        }
        // Only deinit k_bc/v_bc if we created new arrays via repeatHeads.
        // When num_kv_groups == 1, k_bc == k_t2 and v_bc == v_final,
        // so their lifetimes are managed by their own defer statements.
        defer if (num_kv_groups > 1) k_bc.deinit();
        defer if (num_kv_groups > 1) v_bc.deinit();

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        // q_rot, k_bc ready for matmul
        var scores = try ops.matmul(self.ctx, q_rot, k_bc);
        defer scores.deinit();

        var scaled_scores = try ops.multiply(self.ctx, scores, try ops.scalar(self.ctx, scale, .float32));

        // 8. Apply causal mask
        var masked_scores = if (mask) |m| blk: {
            const masked = try ops.add(self.ctx, scaled_scores, m);
            scaled_scores.deinit();
            break :blk masked;
        } else scaled_scores;
        defer masked_scores.deinit();

        // 9. Softmax
        var attn_weights = try ops.softmax(self.ctx, masked_scores, &[_]i32{-1});
        defer attn_weights.deinit();

        // 10. Apply attention to values
        var attn_out = try ops.matmul(self.ctx, attn_weights, v_bc);
        defer attn_out.deinit();

        // 11. Reshape back: [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        const attn_t = try ops.transposeAxes(self.ctx, attn_out, &[_]i32{ 0, 2, 1, 3 });
        defer attn_t.deinit();

        const attn_flat = try ops.reshape(self.ctx, attn_t, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_heads * head_dim) });
        defer attn_flat.deinit();

        // 12. Output projection
        var out = try self.linear(attn_flat, self.wo, self.wo_quant, self.wo_bias);
        if (lora) |lm| {
            var buf: [64]u8 = undefined;
            const key = try std.fmt.bufPrint(&buf, "layers.{d}.attention.wo", .{layer_idx});
            if (lm.getAdapter(key)) |adapter| {
                const out_new = try adapter.apply(out, attn_flat);
                out.deinit();
                out = out_new;
            }
        }

        // Tensor parallelism: all-reduce after attention output projection
        if (self.distributed_group) |group| {
            const reduced = try distributed_mod.allSum(out, group, self.ctx.stream.inner);
            out.deinit();
            out = reduced;
        }

        return out;
    }
};

/// Repeat KV heads for GQA broadcasting.
fn repeatHeads(ctx: EagerContext, tensor: Array, repeats: usize) !Array {
    // tensor: [B, H_kv, S, D]
    // Expand dims: [B, H_kv, 1, S, D] then tile -> [B, H_kv, repeats, S, D]
    // Reshape -> [B, H_kv*repeats, S, D]
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
    // Make a deep copy so the returned array does not depend on 'tiled'
    // which gets deinit'd when this function returns.
    return ops.copy(ctx, reshaped);
}

// ============================================================
// MLP (SwiGLU)
// ============================================================

pub const LlamaMLP = struct {
    ctx: EagerContext,
    config: *const LlamaConfig,

    // SwiGLU: gate_proj @ up_proj -> down_proj
    gate_proj: Array, // [intermediate_size, hidden_size]
    up_proj: Array, // [intermediate_size, hidden_size]
    down_proj: Array, // [hidden_size, intermediate_size]

    // Optional quantized weights (mlx-lm format)
    gate_proj_quant: ?QuantizedWeight = null,
    up_proj_quant: ?QuantizedWeight = null,
    down_proj_quant: ?QuantizedWeight = null,
    /// Optional distributed group for tensor parallelism.
    distributed_group: ?distributed_mod.DistributedGroup = null,

    pub fn setDistributedGroup(self: *LlamaMLP, group: ?distributed_mod.DistributedGroup) void {
        self.distributed_group = group;
    }

    pub fn deinit(self: *LlamaMLP, allocator: std.mem.Allocator) void {
        if (self.gate_proj_quant) |qw| { var m = qw; m.deinit(allocator); } else self.gate_proj.deinit();
        if (self.up_proj_quant) |qw| { var m = qw; m.deinit(allocator); } else self.up_proj.deinit();
        if (self.down_proj_quant) |qw| { var m = qw; m.deinit(allocator); } else self.down_proj.deinit();
    }

    /// Perform linear projection: x @ W^T, using quantized matmul if available.
    fn linear(self: *LlamaMLP, x: Array, weight: Array, quant: ?QuantizedWeight) !Array {
        if (quant) |qw| {
            return quantize_mod.quantizedMatmul(self.ctx, x, qw, true);
        }
        return ops.matmul(self.ctx, x, try ops.transpose(self.ctx, weight));
    }

    pub fn forward(self: *LlamaMLP, x: Array, lora: ?*lora_mod.LoRAModel, layer_idx: usize) !Array {
        // FUSION INTEGRATION POINT (R8.1): When no LoRA adapters are active, the manual
        // gate_proj + silu + up_proj + down_proj chain below can be replaced with a single
        // `compiledSwiGLU` call from `src/ops/fused.zig`, which fuses all intermediate ops
        // into fewer GPU kernel launches via mlx_compile. Usage (non-LoRA path only):
        //   const fused = try fused_ops.compiledSwiGLU(self.ctx.allocator);
        //   defer fused.deinit();
        //   const result = try fused.call(&.{ x, self.gate_proj, self.up_proj, self.down_proj }, self.ctx.allocator);
        //   return result[0];
        //
        // gate = SiLU(x @ gate_proj^T)
        var gate_linear = try self.linear(x, self.gate_proj, self.gate_proj_quant);
        if (lora) |lm| {
            var buf: [64]u8 = undefined;
            const key = try std.fmt.bufPrint(&buf, "layers.{d}.mlp.gate_proj", .{layer_idx});
            if (lm.getAdapter(key)) |adapter| {
                const g_new = try adapter.apply(gate_linear, x);
                gate_linear.deinit();
                gate_linear = g_new;
            }
        }
        defer gate_linear.deinit();

        // SiLU = x * sigmoid(x)
        const gate_sigmoid = try ops.sigmoid(self.ctx, gate_linear);
        defer gate_sigmoid.deinit();
        const gate_act = try ops.multiply(self.ctx, gate_linear, gate_sigmoid);
        defer gate_act.deinit();

        // up = x @ up_proj^T
        var up = try self.linear(x, self.up_proj, self.up_proj_quant);
        if (lora) |lm| {
            var buf: [64]u8 = undefined;
            const key = try std.fmt.bufPrint(&buf, "layers.{d}.mlp.up_proj", .{layer_idx});
            if (lm.getAdapter(key)) |adapter| {
                const u_new = try adapter.apply(up, x);
                up.deinit();
                up = u_new;
            }
        }
        defer up.deinit();

        // hidden = gate * up
        const hidden = try ops.multiply(self.ctx, gate_act, up);
        defer hidden.deinit();

        // out = hidden @ down_proj^T
        var out = try self.linear(hidden, self.down_proj, self.down_proj_quant);
        if (lora) |lm| {
            var buf: [64]u8 = undefined;
            const key = try std.fmt.bufPrint(&buf, "layers.{d}.mlp.down_proj", .{layer_idx});
            if (lm.getAdapter(key)) |adapter| {
                const o_new = try adapter.apply(out, hidden);
                out.deinit();
                out = o_new;
            }
        }

        // Tensor parallelism: all-reduce after MLP down projection
        if (self.distributed_group) |group| {
            const reduced = try distributed_mod.allSum(out, group, self.ctx.stream.inner);
            out.deinit();
            out = reduced;
        }

        return out;
    }
};

// ============================================================
// Transformer Block
// ============================================================

pub const LlamaTransformerBlock = struct {
    ctx: EagerContext,
    config: *const LlamaConfig,
    layer_idx: usize,

    input_layernorm: nn.RMSNorm,
    attention: LlamaAttention,
    post_attention_layernorm: nn.RMSNorm,
    mlp: LlamaMLP,

    pub fn deinit(self: *LlamaTransformerBlock, allocator: std.mem.Allocator) void {
        self.attention.deinit(allocator);
        self.mlp.deinit(allocator);
    }

    pub fn forward(self: *LlamaTransformerBlock, x: Array, mask: ?Array, cache: ?kvcache.KVCacheStrategy, lora: ?*lora_mod.LoRAModel) !Array {
        // Pre-attention norm
        const normed = try self.input_layernorm.forward(x);
        defer normed.deinit();

        // Self-attention with residual
        const attn_out = try self.attention.forward(normed, mask, cache, lora, self.layer_idx);
        defer attn_out.deinit();

        const h = try ops.add(self.ctx, x, attn_out);
        defer h.deinit();

        // Pre-MLP norm
        const mlp_normed = try self.post_attention_layernorm.forward(h);
        defer mlp_normed.deinit();

        // MLP with residual
        const mlp_out = try self.mlp.forward(mlp_normed, lora, self.layer_idx);
        defer mlp_out.deinit();

        return ops.add(self.ctx, h, mlp_out);
    }
};

// ============================================================
// Full LLaMA Model
// ============================================================

pub const LlamaModel = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    config: LlamaConfig,

    embed_tokens: nn.Embedding,
    layers: []LlamaTransformerBlock,
    norm: nn.RMSNorm,
    lm_head: Array, // [vocab_size, hidden_size]
    lm_head_quant: ?QuantizedWeight,
    lm_head_tied: bool, // true when lm_head == embed_tokens.weight (don't double-free)
    lora: ?*lora_mod.LoRAModel,
    /// Optional EAGLE draft head for speculative decoding.
    eagle_drafter: ?@import("../speculative.zig").EagleDrafter = null,
    /// Optional distributed group for tensor parallelism.
    distributed_group: ?distributed_mod.DistributedGroup = null,

    pub fn deinit(self: *LlamaModel) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            layer.deinit(self.allocator);
        }
        self.allocator.free(self.layers);
        self.norm.weight.deinit();
        // Only free lm_head if it's not tied to embed_tokens (already freed above)
        if (!self.lm_head_tied) {
            if (self.lm_head_quant) |qw| {
                var m = qw;
                m.deinit(self.allocator);
            } else {
                self.lm_head.deinit();
            }
        }
        if (self.eagle_drafter) |*ed| {
            ed.deinit();
        }
    }

    /// Set distributed group for tensor parallelism on all layers.
    pub fn setDistributedGroup(self: *LlamaModel, group: ?distributed_mod.DistributedGroup) void {
        self.distributed_group = group;
        for (self.layers) |*layer| {
            layer.attention.setDistributedGroup(group);
            layer.mlp.setDistributedGroup(group);
        }
    }

    /// Count total number of Array parameters in the model.
    pub fn paramCount(self: LlamaModel) usize {
        var count: usize = 3; // embed_tokens.weight, norm.weight, lm_head
        for (self.layers) |layer| {
            count += 1; // input_layernorm
            count += 4; // wq, wk, wv, wo
            if (layer.attention.q_norm != null) count += 1;
            if (layer.attention.k_norm != null) count += 1;
            count += 1; // post_attention_layernorm
            count += 3; // gate_proj, up_proj, down_proj
        }
        return count;
    }

    /// Set model parameters from a flat array (used in training).
    /// Releases old arrays before assigning new ones to prevent leaks.
    pub fn setParams(self: *LlamaModel, params: []const Array) void {
        var idx: usize = 0;
        tree_mod.treeSetArrays(self, params, &idx);
    }

    pub fn forward(
        self: *LlamaModel,
        input_ids: Array,
        mask: ?Array,
        caches: ?[]kvcache.KVCacheStrategy,
    ) !Array {
        var arena = ScopedArrayArena.init(self.allocator);
        defer arena.deinit();

        // input_ids: [batch, seq_len]
        var hidden = try arena.track(try self.embed_tokens.forward(input_ids));

        // Create causal attention mask if not provided and seq_len > 1
        // Shape: [seq_len, seq_len] with -inf for future positions, 0 for valid positions
        const seq_len_val = @as(usize, @intCast(input_ids.shape()[1]));
        var causal_mask: ?Array = mask;
        if (mask == null and seq_len_val > 1) {
            var mask_data = try self.allocator.alloc(f32, seq_len_val * seq_len_val);
            defer self.allocator.free(mask_data);
            for (0..seq_len_val) |row| {
                for (0..seq_len_val) |col| {
                    mask_data[row * seq_len_val + col] = if (col <= row) 0.0 else -1.0e9;
                }
            }
            // Shape [1, 1, seq_len, seq_len] for broadcasting with [batch, heads, seq, seq]
            const mask_arr = try Array.fromData(self.allocator, f32, mask_data, &[_]i32{
                1, 1, @intCast(seq_len_val), @intCast(seq_len_val),
            });
            causal_mask = try arena.track(mask_arr);
        }

        for (self.layers, 0..) |*layer, i| {
            const cache = if (caches) |cache_list| (if (i < cache_list.len) cache_list[i] else null) else null;
            hidden = try arena.track(try layer.forward(hidden, causal_mask, cache, self.lora));
        }

        const normed = try arena.track(try self.norm.forward(hidden));

        // logits = normed @ lm_head^T — final output NOT tracked (caller owns it)
        var logits = if (self.lm_head_quant) |qw|
            try quantize_mod.quantizedMatmul(self.ctx, normed, qw, true)
        else blk: {
            const lm_head_t = try arena.track(try ops.transpose(self.ctx, self.lm_head));
            break :blk try ops.matmul(self.ctx, normed, lm_head_t);
        };
        if (self.lora) |lm| {
            if (lm.getAdapter("lm_head")) |adapter| {
                const logits_new = try adapter.apply(logits, normed);
                logits.deinit();
                logits = logits_new;
            }
        }
        return logits;
    }

    /// Forward pass that returns both logits and the last-layer hidden states.
    /// Required for EAGLE speculative decoding.
    pub fn forwardWithHidden(
        self: *LlamaModel,
        input_ids: Array,
        mask: ?Array,
        caches: ?[]kvcache.KVCacheStrategy,
    ) !@import("../generation.zig").ForwardWithHiddenResult {
        var arena = ScopedArrayArena.init(self.allocator);
        defer arena.deinit();

        // input_ids: [batch, seq_len]
        var hidden = try arena.track(try self.embed_tokens.forward(input_ids));

        for (self.layers, 0..) |*layer, i| {
            const cache = if (caches) |cache_list| (if (i < cache_list.len) cache_list[i] else null) else null;
            hidden = try arena.track(try layer.forward(hidden, mask, cache, self.lora));
        }

        const normed = try arena.track(try self.norm.forward(hidden));

        // logits = normed @ lm_head^T
        var logits = if (self.lm_head_quant) |qw|
            try quantize_mod.quantizedMatmul(self.ctx, normed, qw, true)
        else blk: {
            const lm_head_t = try arena.track(try ops.transpose(self.ctx, self.lm_head));
            break :blk try ops.matmul(self.ctx, normed, lm_head_t);
        };
        if (self.lora) |lm| {
            if (lm.getAdapter("lm_head")) |adapter| {
                const logits_new = try adapter.apply(logits, normed);
                logits.deinit();
                logits = logits_new;
            }
        }

        // Copy normed hidden states since arena will be freed
        const hidden_copy = try ops.copy(self.ctx, normed);
        return .{ .logits = logits, .hidden = hidden_copy };
    }

    /// Generate tokens autoregressively.
    /// Supports EOS stop condition and optional streaming via tokenizer callback.
    pub fn generate(
        self: *LlamaModel,
        prompt: Array,
        max_new_tokens: usize,
        sampler_config: *@import("../sampling.zig").SamplerConfig,
        caches: []kvcache.KVCacheStrategy,
        options: GenerateOptions,
    ) ![]u32 {
        const batch = prompt.shape()[0];
        std.debug.assert(batch == 1); // Simplified for single batch

        var tokens = try self.allocator.dupe(u32, try prompt.dataSlice(u32));
        errdefer self.allocator.free(tokens);

        // First pass: prefill with full prompt
        var prefill_done = false;

        for (0..max_new_tokens) |_| {
            var arena = ScopedArrayArena.init(self.allocator);
            defer arena.deinit();

            // After first pass (prefill), only feed the last generated token.
            // KV cache stores all previous keys/values.
            const input_start = if (prefill_done) tokens.len - 1 else 0;
            const input_slice = tokens[input_start..];
            const input_len = input_slice.len;

            const input_arr = c.c.mlx_array_new_data(
                input_slice.ptr,
                &[_]c_int{ 1, @intCast(input_len) },
                2,
                c.c.MLX_UINT32,
            );
            const input = Array.fromHandle(input_arr);
            defer input.deinit();

            // Performance verification: after prefill, input_len should always be 1
            if (prefill_done and input_len > 1) {
                std.log.warn("Performance bug: input_len={d} after prefill (should be 1)", .{input_len});
            }

            // Forward pass
            const logits = try arena.track(try self.forward(input, null, caches));

            // Get last token logits
            const last_logits = try arena.track(try ops.slice(
                self.ctx,
                logits,
                &[_]i32{ 0, @intCast(input_len - 1), 0 },
                &[_]i32{ 1, @intCast(input_len), std.math.maxInt(i32) },
                &[_]i32{ 1, 1, 1 },
            ));

            // Squeeze to [vocab_size]
            const squeezed = try arena.track(try ops.squeeze(self.ctx, last_logits));

            // Convert to float32 for CPU sampling
            const logits_f32 = try arena.track(try ops.astype(self.ctx, squeezed, .float32));

            // DIAGNOSTIC: Print first token's top logits for comparison with Python
            if (!prefill_done) {
                try logits_f32.eval();
                const ldata = try logits_f32.dataSlice(f32);
                var max1_idx: usize = 0;
                var max1_val: f32 = -std.math.inf(f32);
                var max2_idx: usize = 0;
                var max2_val: f32 = -std.math.inf(f32);
                for (ldata, 0..) |v, idx| {
                    if (v > max1_val) {
                        max2_val = max1_val;
                        max2_idx = max1_idx;
                        max1_val = v;
                        max1_idx = idx;
                    } else if (v > max2_val) {
                        max2_val = v;
                        max2_idx = idx;
                    }
                }
                std.log.info("LLAMA DIAG first token: argmax=[{d}]={d:.4}, 2nd=[{d}]={d:.4}", .{
                    max1_idx, max1_val, max2_idx, max2_val,
                });
                // Python ref: token 10234=20.375, 39814=20.0
                std.log.info("LLAMA DIAG Python ref: [10234]=20.375 [39814]=20.0", .{});
                // Also check token 10234 specifically
                if (ldata.len > 10234) {
                    std.log.info("LLAMA DIAG token 10234 ('Why') logit: {d:.4}", .{ldata[10234]});
                }
            }

            // Sample (update context for repetition penalty)
            sampler_config.context_tokens = tokens;
            const next_token = (try sampler_config.sample(logits_f32, self.allocator)).token;

            // Check EOS stop condition
            if (options.eos_token_id) |eos_id| {
                if (next_token == eos_id) break;
            }

            // Append token
            const new_tokens = try self.allocator.realloc(tokens, tokens.len + 1);
            tokens = new_tokens;
            tokens[tokens.len - 1] = next_token;

            // Stream token output if tokenizer is provided
            if (options.tokenizer) |tok| {
                const text = tok.decode(&[_]u32{next_token}, self.allocator) catch null;
                if (text) |t| {
                    defer self.allocator.free(t);
                    std.debug.print("{s}", .{t});
                }
            }

            prefill_done = true;
        }

        // Print newline after streaming output
        if (options.tokenizer != null) {
            std.debug.print("\n", .{});
        }

        return tokens;
    }
};
