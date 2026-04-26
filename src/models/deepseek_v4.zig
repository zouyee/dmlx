/// DeepSeek-V4-Flash model architecture.
///
/// Features:
/// - MLA (Multi-head Latent Attention) with Q-lora and O-lora
/// - MoE (Mixture of Experts) with hash-based and score-based routing
/// - CSA/HCA (Compressed Sparse Attention / Heavily Compressed Attention)
/// - mHC (Manifold-Constrained Hyper-Connections)
/// - YARN RoPE scaling
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const ops = @import("../ops.zig");
const shape_mod = @import("../ops/shape.zig");
const reduce_mod = @import("../ops/reduce.zig");
const cmp_mod = @import("../ops/comparison.zig");
const math_mod = @import("../ops/math.zig");
const dtype_mod = @import("../dtype.zig");
const nn = @import("../ops/nn.zig");
const kvcache = @import("../kvcache.zig");
const lora_mod = @import("../lora.zig");
const fast_mod = @import("../ops/fast.zig");
const array_arena_mod = @import("../array_arena.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const ScopedArrayArena = array_arena_mod.ScopedArrayArena;

/// DeepSeek-V4 configuration.
pub const DSV4Config = struct {
    vocab_size: usize = 129280,
    hidden_size: usize = 4096,
    num_hidden_layers: usize = 43,
    num_attention_heads: usize = 64,
    head_dim: usize = 512,
    num_key_value_heads: usize = 1,
    q_lora_rank: usize = 1024,
    o_lora_rank: usize = 1024,
    qk_rope_head_dim: usize = 64,
    max_position_embeddings: usize = 1048576,
    n_routed_experts: usize = 256,
    n_shared_experts: usize = 1,
    num_experts_per_tok: usize = 6,
    moe_intermediate_size: usize = 2048,
    routed_scaling_factor: f32 = 1.5,
    scoring_func: ScoringFunc = .sqrtsoftplus,
    norm_topk_prob: bool = true,
    sliding_window: usize = 128,
    num_hash_layers: usize = 3,
    index_n_heads: usize = 64,
    index_head_dim: usize = 128,
    index_topk: usize = 512,
    hc_mult: usize = 4,
    hc_sinkhorn_iters: usize = 20,
    hc_eps: f32 = 1e-6,
    use_mhc: bool = true,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 10000.0,
    compress_rope_theta: f32 = 160000.0,
    rope_scaling: ?YarnRoPEConfig = null,
    compress_ratios: []const usize = &.{},
    swiglu_limit: f32 = 10.0,

    /// Per-weight quantization config (from config.json "quantization" field).
    /// Maps HF weight name (e.g., "model.layers.0.ffn.switch_mlp.gate_proj") to its
    /// quantization parameters. When loading quantized weights, the loader looks up
    /// the per-weight config here; falls back to global defaults if not found.
    /// Default global: mode="affine", bits=4, group_size=64.
    quantize_default_bits: u8 = 0,
    quantize_default_group_size: i32 = 64,
    quantize_default_mode: []const u8 = "affine",

    /// Dtype for KV cache storage of non-RoPE dimensions.
    /// DeepSeek V4 stores most KV dimensions in FP8 (E4M3) via mlx_to_fp8/mlx_from_fp8.
    /// RoPE dimensions are always stored in bfloat16 regardless of this setting.
    /// Set to .float32 to disable reduced-precision KV storage (FP8 path is skipped).
    /// The actual dtype value is only used as a flag: anything != .float32 enables FP8.
    kv_storage_dtype: dtype_mod.Dtype = .float16,

    pub const ScoringFunc = enum {
        softmax,
        sigmoid,
        sqrtsoftplus,
    };

    pub const YarnRoPEConfig = struct {
        factor: f32 = 16.0,
        original_max_position_embeddings: usize = 65536,
        beta_fast: f32 = 32.0,
        beta_slow: f32 = 1.0,
    };
};

/// YARN-scaled RoPE positional encoding.
pub const DSV4YarnRoPE = struct {
    ctx: EagerContext,
    dim: usize,
    max_seq_len: usize,
    base: f32,
    scale: f32,
    yarn_config: DSV4Config.YarnRoPEConfig,
    cos_cache: Array,
    sin_cache: Array,

    pub fn deinit(self: *DSV4YarnRoPE) void {
        self.cos_cache.deinit();
        self.sin_cache.deinit();
    }

    pub fn init(ctx: EagerContext, dim: usize, max_seq_len: usize, base: f32, config: DSV4Config.YarnRoPEConfig) !DSV4YarnRoPE {
        std.debug.assert(dim % 2 == 0);
        const half_dim = dim / 2;

        const cos_cache = try array_mod.zeros(ctx.allocator, &[_]i32{ @intCast(max_seq_len), @intCast(half_dim) }, .float32);
        const sin_cache = try array_mod.zeros(ctx.allocator, &[_]i32{ @intCast(max_seq_len), @intCast(half_dim) }, .float32);

        const cos_data = @constCast(try cos_cache.dataPtr(f32))[0 .. max_seq_len * half_dim];
        const sin_data = @constCast(try sin_cache.dataPtr(f32))[0 .. max_seq_len * half_dim];

        // Precompute YARN-scaled frequencies
        const original_seq_len = config.original_max_position_embeddings;
        const factor = config.factor;
        const beta_fast = config.beta_fast;
        const beta_slow = config.beta_slow;

        // Find correction range for YARN interpolation
        const low, const high = findCorrectionRange(beta_fast, beta_slow, dim, base, original_seq_len);

        for (0..max_seq_len) |pos| {
            for (0..half_dim) |i| {
                var freq = 1.0 / std.math.pow(f32, base, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(dim)));

                if (original_seq_len > 0) {
                    const smooth = 1.0 - linearRampFactor(low, high, @intCast(i), half_dim);
                    freq = freq / factor * (1.0 - smooth) + freq * smooth;
                }

                const angle = @as(f32, @floatFromInt(pos)) * freq;
                const idx = pos * half_dim + i;
                cos_data[idx] = @cos(angle);
                sin_data[idx] = @sin(angle);
            }
        }

        return DSV4YarnRoPE{
            .ctx = ctx,
            .dim = dim,
            .max_seq_len = max_seq_len,
            .base = base,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dim))),
            .yarn_config = config,
            .cos_cache = cos_cache,
            .sin_cache = sin_cache,
        };
    }

    /// Apply RoPE in-place to the last `dim` dimensions of input.
    /// Input shape: (..., dim) where dim = self.dim
    pub fn apply(self: *DSV4YarnRoPE, input: Array, start_pos: usize, stream: c.c.mlx_stream) !Array {
        _ = stream;
        const shape = input.shape();
        const seq_len = @as(usize, @intCast(shape[shape.len - 2]));
        const dim = self.dim;
        const half_dim = dim / 2;

        const out = try array_mod.zeros(self.ctx.allocator, shape, input.dtype());
        const src = try input.dataSliceMut(f32);
        const dst = try out.dataSliceMut(f32);
        const cos_data = try self.cos_cache.dataSliceMut(f32);
        const sin_data = try self.sin_cache.dataSliceMut(f32);

        const total_elements = input.size();
        const seq_stride = dim;
        const non_seq_elements = total_elements / (seq_len * dim);

        for (0..non_seq_elements) |n| {
            for (0..seq_len) |s| {
                const pos = start_pos + s;
                if (pos >= self.max_seq_len) continue;
                const base_idx = n * seq_len * seq_stride + s * seq_stride;
                const cache_idx = pos * half_dim;
                for (0..half_dim) |i| {
                    const x1 = src[base_idx + i];
                    const x2 = src[base_idx + half_dim + i];
                    const cos_val = cos_data[cache_idx + i];
                    const sin_val = sin_data[cache_idx + i];
                    dst[base_idx + i] = x1 * cos_val - x2 * sin_val;
                    dst[base_idx + half_dim + i] = x1 * sin_val + x2 * cos_val;
                }
            }
        }

        return out;
    }

    pub fn applyInverse(self: *DSV4YarnRoPE, input: Array, start_pos: usize, stream: c.c.mlx_stream) !Array {
        _ = stream;
        const shape = input.shape();
        const seq_len = @as(usize, @intCast(shape[shape.len - 2]));
        const dim = self.dim;
        const half_dim = dim / 2;

        const out = try array_mod.zeros(self.ctx.allocator, shape, input.dtype());
        const src = try input.dataSliceMut(f32);
        const dst = try out.dataSliceMut(f32);
        const cos_data = try self.cos_cache.dataSliceMut(f32);
        const sin_data = try self.sin_cache.dataSliceMut(f32);

        const total_elements = input.size();
        const seq_stride = dim;
        const non_seq_elements = total_elements / (seq_len * dim);

        for (0..non_seq_elements) |n| {
            for (0..seq_len) |s| {
                const pos = start_pos + s;
                if (pos >= self.max_seq_len) continue;
                const base_idx = n * seq_len * seq_stride + s * seq_stride;
                const cache_idx = pos * half_dim;
                for (0..half_dim) |i| {
                    const x1 = src[base_idx + i];
                    const x2 = src[base_idx + half_dim + i];
                    const cos_val = cos_data[cache_idx + i];
                    const sin_val = sin_data[cache_idx + i];
                    // Inverse rotation: conjugate
                    dst[base_idx + i] = x1 * cos_val + x2 * sin_val;
                    dst[base_idx + half_dim + i] = -x1 * sin_val + x2 * cos_val;
                }
            }
        }

        return out;
    }
};

fn findCorrectionDim(num_rotations: f32, dim: usize, base: f32, max_seq_len: usize) f32 {
    return @as(f32, @floatFromInt(dim)) * @log(@as(f32, @floatFromInt(max_seq_len)) / (num_rotations * 2.0 * std.math.pi)) / (2.0 * @log(base));
}

fn findCorrectionRange(beta_fast: f32, beta_slow: f32, dim: usize, base: f32, max_seq_len: usize) struct { usize, usize } {
    const low = @max(0, @as(i32, @intFromFloat(@floor(findCorrectionDim(beta_fast, dim, base, max_seq_len)))));
    const high = @min(@as(i32, @intCast(dim - 1)), @as(i32, @intFromFloat(@ceil(findCorrectionDim(beta_slow, dim, base, max_seq_len)))));
    return .{ @intCast(low), @intCast(high) };
}

fn linearRampFactor(min: usize, max: usize, idx: usize, total: usize) f32 {
    _ = total;
    const min_f = @as(f32, @floatFromInt(min));
    const max_f = @as(f32, @floatFromInt(max));
    const idx_f = @as(f32, @floatFromInt(idx));
    if (min == max) {
        return if (idx_f <= min_f) 0.0 else 1.0;
    }
    const val = (idx_f - min_f) / (max_f - min_f);
    return std.math.clamp(val, 0.0, 1.0);
}

/// Apply per-feature RMSNorm using mlx_fast_rms_norm (GPU accelerated).
fn applyRMSNorm(ctx: EagerContext, x: Array, eps: f32) !Array {
    const shape = x.shape();
    const last_dim = @as(usize, @intCast(shape[shape.len - 1]));
    const ones = try array_mod.ones(ctx.allocator, &[_]i32{@intCast(last_dim)}, x.dtype());
    defer ones.deinit();
    return fast_mod.rmsNorm(ctx, x, ones, eps);
}

/// Create causal mask with sliding window.
/// Returns mask of shape [B, H, S, S] where upper triangle and positions outside window are -inf.
/// Top-k selection using MLX C API.
fn topkIndices(ctx: EagerContext, input: Array, k: i32, n_experts: usize, stream: c.c.mlx_stream) !Array {
    // argsort in ascending order along last axis
    var sorted_handle = c.c.mlx_array_new();
    try c.check(c.c.mlx_argsort_axis(&sorted_handle, input.inner, -1, stream));
    const sorted = Array.fromHandle(sorted_handle);
    defer sorted.deinit();
    // Slice the last k elements (largest scores)
    const start_y = @as(i32, @intCast(n_experts - @as(usize, @intCast(k))));
    const stop_y = @as(i32, @intCast(n_experts));
    const batch_size = sorted.shape()[0];
    const result = try ops.slice(ctx, sorted, &[_]i32{ 0, start_y }, &[_]i32{ batch_size, stop_y }, &[_]i32{ 1, 1 });
    try result.eval();
    return result;
}

/// Gather scores at given indices.
fn gatherScores(scores: Array, indices: Array, ctx: EagerContext, stream: c.c.mlx_stream) !Array {
    _ = stream;
    // scores: [N, n_experts]
    // indices: [N, topk]
    // output: [N, topk]
    // Use simple gather with take_along_axis
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_take_along_axis(&res, scores.inner, indices.inner, 1, ctx.stream.inner));
    return Array.fromHandle(res);
}

fn createCausalMask(ctx: EagerContext, batch: usize, num_heads: usize, seq_len: usize, window_size: usize, start_pos: usize) !Array {
    const mask_shape = [_]i32{ @intCast(batch), @intCast(num_heads), @intCast(seq_len), @intCast(seq_len) };
    const mask = try array_mod.zeros(ctx.allocator, &mask_shape, .float32);
    const data = @constCast(try mask.dataPtr(f32))[0..@intCast(batch * num_heads * seq_len * seq_len)];
    @memset(data, 0.0);

    const neg_inf = -std.math.inf(f32);
    for (0..batch) |b| {
        for (0..num_heads) |h| {
            for (0..seq_len) |i| {
                const q_pos = start_pos + i;
                for (0..seq_len) |j| {
                    const k_pos = start_pos + j;
                    // Causal: q can only attend to k where k_pos <= q_pos
                    const causal = k_pos > q_pos;
                    // Sliding window: q can only attend to k within window_size
                    const outside_window = q_pos >= window_size and k_pos < q_pos + 1 - window_size;
                    if (causal or outside_window) {
                        const idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                        data[idx] = neg_inf;
                    }
                }
            }
        }
    }

    return mask;
}

/// Single MoE expert: SwiGLU FFN.
pub const DSV4Expert = struct {
    ctx: EagerContext,
    w1: Array, // gate projection
    w2: Array, // down projection
    w3: Array, // up projection
    swiglu_limit: f32,

    pub fn deinit(self: *DSV4Expert) void {
        self.w1.deinit();
        self.w2.deinit();
        self.w3.deinit();
    }

    pub fn forward(self: *DSV4Expert, x: Array, stream: c.c.mlx_stream) !Array {
        _ = stream;
        // SwiGLU: gate = silu(w1(x)) * w3(x) -> w2(gate)
        //
        // FUSION INTEGRATION POINT (R8.1): The manual gate_proj + silu + up_proj + down_proj
        // chain below can be replaced with a single `compiledSwiGLU` call from
        // `src/ops/fused.zig`, which fuses all intermediate ops into fewer GPU kernel
        // launches via mlx_compile. Usage:
        //   const fused = try fused_ops.compiledSwiGLU(self.ctx.allocator);
        //   defer fused.deinit();
        //   const result = try fused.call(&.{ x, self.w1, self.w3, self.w2 }, self.ctx.allocator);
        //   return result[0];
        //
        const w1_t = try ops.transpose(self.ctx, self.w1);
        defer w1_t.deinit();
        const gate_proj = try ops.matmul(self.ctx, x, w1_t);
        defer gate_proj.deinit();
        const w3_t = try ops.transpose(self.ctx, self.w3);
        defer w3_t.deinit();
        const up_proj = try ops.matmul(self.ctx, x, w3_t);
        defer up_proj.deinit();

        const silu_gate = try ops.multiply(self.ctx, gate_proj, try ops.sigmoid(self.ctx, gate_proj));
        defer silu_gate.deinit();
        const hidden = try ops.multiply(self.ctx, silu_gate, up_proj);
        defer hidden.deinit();

        const w2_t = try ops.transpose(self.ctx, self.w2);
        defer w2_t.deinit();
        return ops.matmul(self.ctx, hidden, w2_t);
    }
};

/// MoE Gate: computes routing scores and selects top-k experts.
pub const DSV4Gate = struct {
    ctx: EagerContext,
    weight: Array, // [n_routed_experts, dim]
    bias: ?Array, // optional bias for score-based routing
    tid2eid: ?Array, // [vocab_size, topk] for hash-based routing
    topk: usize,
    n_routed_experts: usize,
    route_scale: f32,
    scoring_func: DSV4Config.ScoringFunc,
    is_hash: bool,

    pub fn deinit(self: *DSV4Gate) void {
        self.weight.deinit();
        if (self.bias) |b| b.deinit();
        if (self.tid2eid) |t| t.deinit();
    }

    pub fn forward(self: *DSV4Gate, hidden_states: Array, input_ids: ?Array, stream: c.c.mlx_stream) !struct { Array, Array } {
        const shape = hidden_states.shape();
        const batch = @as(usize, @intCast(shape[0]));
        const seq_len = if (shape.len >= 3) @as(usize, @intCast(shape[1])) else 1;
        const dim = @as(usize, @intCast(shape[shape.len - 1]));
        // Flatten to [B*S, dim]
        const flat = try ops.reshape(self.ctx, hidden_states, &[_]i32{ @intCast(batch * seq_len), @intCast(dim) });
        defer flat.deinit();

        // Scores: flat @ weight^T -> [B*S, n_routed_experts]
        const weight_t = try ops.transpose(self.ctx, self.weight);
        defer weight_t.deinit();
        const logits = try ops.matmul(self.ctx, flat, weight_t);
        defer logits.deinit();

        // Apply scoring function
        var scores: Array = undefined;
        switch (self.scoring_func) {
            .softmax => {
                scores = try ops.softmax(self.ctx, logits, &[_]i32{-1});
            },
            .sigmoid => {
                // sigmoid(x) = 1 / (1 + exp(-x))
                const neg = try ops.negative(self.ctx, logits);
                defer neg.deinit();
                const exp_neg = try ops.exp(self.ctx, neg);
                defer exp_neg.deinit();
                const one = try ops.scalarF32(self.ctx, 1.0);
                defer one.deinit();
                const denom = try ops.add(self.ctx, one, exp_neg);
                defer denom.deinit();
                scores = try ops.divide(self.ctx, one, denom);
            },
            .sqrtsoftplus => {
                // softplus(x) = log(1 + exp(x))
                // For numerical stability: softplus(x) = max(0, x) + log(1 + exp(-|x|))
                // Simplified: log(1 + exp(x))
                const exp_logits = try ops.exp(self.ctx, logits);
                defer exp_logits.deinit();
                const one = try ops.scalarF32(self.ctx, 1.0);
                defer one.deinit();
                const one_plus_exp = try ops.add(self.ctx, one, exp_logits);
                defer one_plus_exp.deinit();
                const softplus = try ops.log(self.ctx, one_plus_exp);
                defer softplus.deinit();
                scores = try ops.sqrt(self.ctx, softplus);
            },
        }
        defer scores.deinit();

        // Store original scores for weights
        const original_scores = try ops.copy(self.ctx, scores);
        defer original_scores.deinit();

        // Add bias for selection if present
        var scores_for_choice = scores;
        if (self.bias) |bias| {
            const bias_exp = try ops.expandDims(self.ctx, bias, 0);
            defer bias_exp.deinit();
            const biased = try ops.add(self.ctx, scores, bias_exp);
            scores_for_choice = biased;
        }
        defer if (scores_for_choice.inner.ctx != scores.inner.ctx) scores_for_choice.deinit();

        // Hash-based routing (first num_hash_layers) or score-based top-k
        var use_hash = false;
        var indices: Array = undefined;
        if (self.is_hash) {
            if (self.tid2eid) |t2e| {
                if (input_ids) |ids| {
                    // Lookup expert IDs from tid2eid table: [vocab_size, topk] -> [B, S, topk]
                    var looked_up = c.c.mlx_array_new();
                    try c.check(c.c.mlx_take_axis(&looked_up, t2e.inner, ids.inner, 0, stream));
                    const lu = Array.fromHandle(looked_up);
                    defer lu.deinit();

                    const lu_shape = lu.shape();
                    indices = try ops.reshape(self.ctx, lu, &[_]i32{ lu_shape[0] * lu_shape[1], lu_shape[2] });
                    use_hash = true;
                }
            }
        }

        if (!use_hash) {
            // MOE ROUTER INTEGRATION POINT (R20.1): The inline top-k selection below
            // (topkIndices + gatherScores + normalization) can be replaced with a single
            // call to `moe_router.MoERouter.topkRoute()`, which performs the same
            // argsort-based top-k selection, weight gathering via take_along_axis, and
            // optional weight normalization. Usage:
            //   const moe_router = @import("../moe_router.zig");
            //   const router = moe_router.MoERouter.init(.{
            //       .num_experts = self.n_routed_experts,
            //       .top_k = self.topk,
            //       .normalize_weights = (self.scoring_func != .softmax),
            //   });
            //   const route = try router.topkRoute(self.ctx, scores_for_choice);
            //   // route.indices replaces `indices`, route.weights replaces `final_weights`
            //   // Note: bias-adjusted scores_for_choice and route_scale still need
            //   // to be applied outside the router.
            indices = try topkIndices(self.ctx, scores_for_choice, @intCast(self.topk), self.n_routed_experts, stream);
        }

        // Gather weights for selected experts
        const weights = try gatherScores(original_scores, indices, self.ctx, stream);
        defer weights.deinit();

        // Normalize weights if not softmax
        var final_weights = weights;
        if (self.scoring_func != .softmax) {
            const sum_weights = try reduce_mod.sumAxis(self.ctx, weights, -1, true);
            defer sum_weights.deinit();
            const eps = try ops.scalarF32(self.ctx, 1e-20);
            defer eps.deinit();
            const denom = try ops.add(self.ctx, sum_weights, eps);
            defer denom.deinit();
            const normed = try ops.divide(self.ctx, weights, denom);
            final_weights = normed;
        }
        defer if (final_weights.inner.ctx != weights.inner.ctx) final_weights.deinit();

        // Scale weights
        const scale = try ops.scalarF32(self.ctx, self.route_scale);
        defer scale.deinit();
        const scaled_weights = try ops.multiply(self.ctx, final_weights, scale);

        return .{ scaled_weights, indices };
    }
};

/// MoE layer: routed experts + shared expert.
pub const DSV4MoE = struct {
    ctx: EagerContext,
    gate: DSV4Gate,
    experts: []DSV4Expert,
    shared_expert: DSV4Expert,
    n_routed_experts: usize,
    n_activated_experts: usize,

    pub fn deinit(self: *DSV4MoE) void {
        self.gate.deinit();
        for (self.experts) |*expert| {
            expert.deinit();
        }
        self.ctx.allocator.free(self.experts);
        self.shared_expert.deinit();
    }

    pub fn forward(self: *DSV4MoE, hidden_states: Array, input_ids: Array, stream: c.c.mlx_stream) !Array {
        const shape = hidden_states.shape();
        const batch = @as(usize, @intCast(shape[0]));
        const seq_len = @as(usize, @intCast(shape[1]));
        const dim = @as(usize, @intCast(shape[2]));

        // Flatten to [B*S, dim]
        const flat = try ops.reshape(self.ctx, hidden_states, &[_]i32{ @intCast(batch * seq_len), @intCast(dim) });
        defer flat.deinit();

        // Gate routing
        const weights, const indices = try self.gate.forward(flat, input_ids, stream);
        defer weights.deinit();
        defer indices.deinit();

        // Initialize output
        const output_shape = [_]i32{ @intCast(batch * seq_len), @intCast(dim) };
        var y = try array_mod.zeros(self.ctx.allocator, &output_shape, .float32);
        errdefer y.deinit();

        // MOE ROUTER INTEGRATION POINT (R20.2): The inline expert dispatch loop below
        // (mask computation, per-expert iteration, weighted accumulation) can be replaced
        // with `moe_router.MoERouter.expandTokens()` and `moe_router.MoERouter.reduceExperts()`.
        //
        // expandTokens duplicates each token k times (one per selected expert), producing
        // [N*k, dim]. Each expert processes its assigned slice. reduceExperts then performs
        // the weighted sum back to [N, dim]. Usage:
        //
        //   const moe_router = @import("../moe_router.zig");
        //   const route = moe_router.RouteResult{
        //       .indices = indices, .weights = weights, .k = self.n_activated_experts,
        //   };
        //   const expanded = try moe_router.MoERouter.expandTokens(self.ctx, flat, route);
        //   // ... run each expert on its slice of expanded tokens ...
        //   const y = try moe_router.MoERouter.reduceExperts(self.ctx, expert_outs, route);
        //
        // Note: The current mask-based approach runs all experts and masks inactive tokens
        // to zero, while expandTokens/reduceExperts uses a gather/scatter pattern. Both
        // produce equivalent results but the router approach avoids redundant expert
        // computation on zero-masked inputs.

        // Precompute all-expert mask in one broadcasted equal op:
        // indices: [N, topk] -> expand to [N, topk, 1]
        // expert_range: [1, 1, n_routed_experts]
        // all_masks: [N, topk, n_routed_experts]
        const indices_exp = try ops.expandDims(self.ctx, indices, 2);
        defer indices_exp.deinit();

        var expert_range_data = try self.ctx.allocator.alloc(f32, self.n_routed_experts);
        defer self.ctx.allocator.free(expert_range_data);
        for (0..self.n_routed_experts) |i| expert_range_data[i] = @floatFromInt(i);
        const expert_range = try Array.fromData(self.ctx.allocator, f32, expert_range_data, &[_]i32{ 1, 1, @intCast(self.n_routed_experts) });
        defer expert_range.deinit();

        const all_masks = try cmp_mod.equal(self.ctx, indices_exp, expert_range);
        defer all_masks.deinit();

        // Route each token to its selected experts
        for (0..self.n_routed_experts) |eid| {
            const expert = &self.experts[eid];

            // Extract mask for this expert: [N, topk, 1]
            const mask = try ops.slice(self.ctx, all_masks, &[_]i32{ 0, 0, @intCast(eid) }, &[_]i32{ @intCast(batch * seq_len), @intCast(self.n_activated_experts), @intCast(eid + 1) }, &[_]i32{});
            defer mask.deinit();

            // Check if any token is routed to this expert
            const has_tokens = try reduce_mod.sumAxes(self.ctx, mask, &[_]i32{ 0, 1, 2 }, false);
            defer has_tokens.deinit();
            const has_tokens_val = try has_tokens.dataPtr(f32);
            if (has_tokens_val[0] == 0.0) continue;

            // For each position in top-k
            for (0..self.n_activated_experts) |k| {
                const k_slice = try ops.slice(self.ctx, mask, &[_]i32{ 0, @intCast(k), 0 }, &[_]i32{ @intCast(batch * seq_len), @intCast(k + 1), 1 }, &[_]i32{});
                defer k_slice.deinit();
                const k_mask = try ops.reshape(self.ctx, k_slice, &[_]i32{ @intCast(batch * seq_len) });
                defer k_mask.deinit();

                // Gather tokens for this expert at this k position
                const mask_f32 = try ops.astype(self.ctx, k_mask, .float32);
                defer mask_f32.deinit();
                const mask_exp = try ops.expandDims(self.ctx, mask_f32, 1);
                defer mask_exp.deinit();

                const masked_input = try ops.multiply(self.ctx, flat, mask_exp);
                defer masked_input.deinit();

                // Run expert
                const expert_out = try expert.forward(masked_input, stream);
                defer expert_out.deinit();

                // Get weight for this k position
                const k_weight = try ops.slice(self.ctx, weights, &[_]i32{ 0, @intCast(k) }, &[_]i32{ @intCast(batch * seq_len), @intCast(k + 1) }, &[_]i32{});
                defer k_weight.deinit();
                const k_weight_flat = try ops.reshape(self.ctx, k_weight, &[_]i32{ @intCast(batch * seq_len) });
                defer k_weight_flat.deinit();
                const k_weight_exp = try ops.expandDims(self.ctx, k_weight_flat, 1);
                defer k_weight_exp.deinit();

                // Weighted output
                const weighted_out = try ops.multiply(self.ctx, expert_out, k_weight_exp);
                defer weighted_out.deinit();

                // Accumulate
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

/// FP4 Lightning Indexer for CSA sparse block selection.
///
/// DeepSeek V4's CSA uses a lightweight multi-head dot-product indexer that operates
/// in 4-bit quantized precision on the QK path. Instead of attending to ALL compressed
/// blocks, the indexer scores each block cheaply and selects the top-k most relevant
/// blocks for full-precision attention.
///
/// Architecture:
///   1. Project query and compressed keys into index space (index_head_dim per head)
///   2. Quantize Q and K projections to 4-bit for fast scoring
///   3. Compute dot-product attention scores: score = Q_fp4 @ K_fp4^T
///   4. Select top-k blocks by score
///   5. Gather only selected blocks for full attention
///
/// This reduces CSA attention cost from O(n_compressed_blocks) to O(index_topk).
pub const LightningIndexer = struct {
    ctx: EagerContext,
    wq_index: Array, // [index_n_heads * index_head_dim, head_dim] — query projection
    wk_index: Array, // [index_n_heads * index_head_dim, head_dim] — key projection
    index_n_heads: usize,
    index_head_dim: usize,
    index_topk: usize,

    pub fn deinit(self: *LightningIndexer) void {
        self.wq_index.deinit();
        self.wk_index.deinit();
    }

    /// Score compressed blocks and return indices of top-k blocks.
    ///
    /// q: [B, S, head_dim] — query states (after projection, before RoPE split)
    /// k_compressed: [B, N_blocks, head_dim] — compressed key blocks from compressKV
    /// Returns: top-k block indices [B, S, index_topk] as i32
    pub fn selectTopK(
        self: *LightningIndexer,
        q: Array, // [B, S, head_dim]
        k_compressed: Array, // [B, N_blocks, head_dim]
    ) !Array {
        const ctx = self.ctx;
        const q_shape = q.shape();
        const batch = @as(usize, @intCast(q_shape[0]));
        const seq_len = @as(usize, @intCast(q_shape[1]));
        const k_shape = k_compressed.shape();
        const n_blocks = @as(usize, @intCast(k_shape[1]));

        // If fewer blocks than topk, return all indices (no selection needed)
        if (n_blocks <= self.index_topk) {
            return self.allBlockIndices(batch, seq_len, n_blocks);
        }

        // Project Q into index space: [B, S, head_dim] @ [head_dim, index_dim] -> [B, S, index_dim]
        const wq_t = try ops.transpose(ctx, self.wq_index);
        defer wq_t.deinit();
        const q_index = try ops.matmul(ctx, q, wq_t);
        defer q_index.deinit();

        // Project K into index space: [B, N_blocks, head_dim] @ [head_dim, index_dim] -> [B, N_blocks, index_dim]
        const wk_t = try ops.transpose(ctx, self.wk_index);
        defer wk_t.deinit();
        const k_index = try ops.matmul(ctx, k_compressed, wk_t);
        defer k_index.deinit();

        // Quantize Q and K to 4-bit for fast scoring.
        // MLX doesn't have native FP4, so we simulate by quantizing to INT4 and dequantizing.
        // This gives the memory/compute savings of 4-bit while staying in the MLX graph.
        const q_fp4 = try self.quantize4bit(q_index);
        defer q_fp4.deinit();
        const k_fp4 = try self.quantize4bit(k_index);
        defer k_fp4.deinit();

        // Reshape to multi-head: [B, S, n_heads, head_dim] -> [B, n_heads, S, head_dim]
        const q_mh = try ops.reshape(ctx, q_fp4, &[_]i32{
            @intCast(batch),     @intCast(seq_len),
            @intCast(self.index_n_heads), @intCast(self.index_head_dim),
        });
        defer q_mh.deinit();
        const q_mh_t = try ops.transposeAxes(ctx, q_mh, &[_]i32{ 0, 2, 1, 3 });
        defer q_mh_t.deinit();

        const k_mh = try ops.reshape(ctx, k_fp4, &[_]i32{
            @intCast(batch),     @intCast(n_blocks),
            @intCast(self.index_n_heads), @intCast(self.index_head_dim),
        });
        defer k_mh.deinit();
        const k_mh_t = try ops.transposeAxes(ctx, k_mh, &[_]i32{ 0, 2, 1, 3 });
        defer k_mh_t.deinit();

        // Attention scores: [B, n_heads, S, head_dim] @ [B, n_heads, head_dim, N_blocks]
        //                 = [B, n_heads, S, N_blocks]
        const k_mh_transposed = try ops.transposeAxes(ctx, k_mh_t, &[_]i32{ 0, 1, 3, 2 });
        defer k_mh_transposed.deinit();
        const scores = try ops.matmul(ctx, q_mh_t, k_mh_transposed);
        defer scores.deinit();

        // Scale by 1/sqrt(index_head_dim)
        const scale_val = 1.0 / @sqrt(@as(f32, @floatFromInt(self.index_head_dim)));
        const scale = try ops.scalarF32(ctx, scale_val);
        defer scale.deinit();
        const scores_scaled = try ops.multiply(ctx, scores, scale);
        defer scores_scaled.deinit();

        // Average across heads: [B, n_heads, S, N_blocks] -> [B, S, N_blocks]
        const scores_avg = try reduce_mod.meanAxis(ctx, scores_scaled, 1, false);
        defer scores_avg.deinit();

        // Top-k selection: argsort descending, take first index_topk
        // Negate scores so ascending argsort gives descending order
        const neg_scores = try ops.negative(ctx, scores_avg);
        defer neg_scores.deinit();

        var sorted_handle = c.c.mlx_array_new();
        try c.check(c.c.mlx_argsort_axis(&sorted_handle, neg_scores.inner, -1, ctx.stream.inner));
        const sorted_indices = Array.fromHandle(sorted_handle);
        defer sorted_indices.deinit();

        // Take first index_topk indices: [B, S, index_topk]
        const topk_indices = try ops.slice(ctx, sorted_indices, &[_]i32{ 0, 0, 0 }, &[_]i32{
            @intCast(batch), @intCast(seq_len), @intCast(self.index_topk),
        }, &[_]i32{});
        try topk_indices.eval();
        return topk_indices;
    }

    /// Quantize a tensor to FP4 (E2M1) and immediately dequantize back to float32.
    /// Uses real MXFP4 quantization via mlx_quantize(mode="mxfp4") for true E2M1 precision,
    /// matching DeepSeek V4's Lightning Indexer FP4 QK path.
    fn quantize4bit(self: *LightningIndexer, tensor: Array) !Array {
        const ctx = self.ctx;
        const stream = ctx.stream.inner;

        // Reshape to 2D for mlx_quantize (requires 2D input)
        const shape = tensor.shape();
        const ndim = shape.len;
        var flat_rows: i32 = 1;
        for (0..ndim - 1) |i| flat_rows *= shape[i];
        const last_dim = shape[ndim - 1];

        // MXFP4 requires group_size=32, last dim must be divisible by 32.
        const group_size: i32 = 32;
        if (last_dim < group_size) {
            // Too small to quantize meaningfully, return as-is (copy)
            return ops.copy(ctx, tensor);
        }

        const flat = try ops.reshape(ctx, tensor, &[_]i32{ flat_rows, last_dim });
        defer flat.deinit();

        // Ensure float32 for quantization
        const flat_f32 = if (tensor.dtype() != .float32)
            try ops.astype(ctx, flat, .float32)
        else
            try ops.copy(ctx, flat);
        defer flat_f32.deinit();

        // Quantize to MXFP4 (E2M1) — true FP4 precision
        var vec = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(vec);

        const opt_group: c.c.mlx_optional_int = .{ .value = group_size, .has_value = true };
        const opt_bits: c.c.mlx_optional_int = .{ .value = 4, .has_value = true };

        try c.check(c.c.mlx_quantize(
            &vec,
            flat_f32.inner,
            opt_group,
            opt_bits,
            "mxfp4",
            .{ .ctx = null },
            stream,
        ));

        const vec_size = c.c.mlx_vector_array_size(vec);
        var packed_arr = c.c.mlx_array_new();
        var scales_arr = c.c.mlx_array_new();
        try c.check(c.c.mlx_vector_array_get(&packed_arr, vec, 0));
        try c.check(c.c.mlx_vector_array_get(&scales_arr, vec, 1));
        defer _ = c.c.mlx_array_free(packed_arr);
        defer _ = c.c.mlx_array_free(scales_arr);

        // MXFP4 may not return biases (uses global_scale instead)
        var biases_arr = c.c.mlx_array_new();
        if (vec_size >= 3) {
            try c.check(c.c.mlx_vector_array_get(&biases_arr, vec, 2));
        }
        defer _ = c.c.mlx_array_free(biases_arr);

        // Dequantize back to float32 using MXFP4 mode
        const no_dtype: c.c.mlx_optional_dtype = .{ .value = c.c.MLX_FLOAT32, .has_value = true };
        var deq = c.c.mlx_array_new();
        try c.check(c.c.mlx_dequantize(
            &deq,
            packed_arr,
            scales_arr,
            biases_arr,
            opt_group,
            opt_bits,
            "mxfp4",
            .{ .ctx = null },
            no_dtype,
            stream,
        ));

        const deq_arr = Array.fromHandle(deq);
        defer deq_arr.deinit();

        // Reshape back to original shape
        return ops.reshape(ctx, deq_arr, shape);
    }

    /// Generate indices [0..n_blocks) for all blocks when n_blocks <= index_topk.
    fn allBlockIndices(self: *LightningIndexer, batch: usize, seq_len: usize, n_blocks: usize) !Array {
        const ctx = self.ctx;
        // Create [n_blocks] range
        var idx_data = try ctx.allocator.alloc(i32, n_blocks);
        defer ctx.allocator.free(idx_data);
        for (0..n_blocks) |i| idx_data[i] = @intCast(i);

        const range = try Array.fromData(ctx.allocator, i32, idx_data, &[_]i32{ 1, 1, @intCast(n_blocks) });
        defer range.deinit();

        // Broadcast to [B, S, n_blocks]
        return ops.broadcastTo(ctx, range, &[_]i32{
            @intCast(batch), @intCast(seq_len), @intCast(n_blocks),
        });
    }

    /// Gather selected blocks from compressed KV using indexer output.
    /// k_compressed: [B, N_blocks, D] or [B, 1, N_blocks, D]
    /// indices: [B, S, topk] — block indices from selectTopK
    /// Returns: [B, topk, D] — selected blocks (using first query position's selection)
    pub fn gatherBlocks(self: *LightningIndexer, k_compressed: Array, indices: Array) !Array {
        const ctx = self.ctx;
        const k_shape = k_compressed.shape();
        const idx_shape = indices.shape();

        // Use first query position's selection for block gathering
        // indices: [B, S, topk] -> [B, topk] (take s=0)
        const batch = @as(usize, @intCast(idx_shape[0]));
        const topk = @as(usize, @intCast(idx_shape[idx_shape.len - 1]));

        const idx_first = try ops.slice(ctx, indices, &[_]i32{ 0, 0, 0 }, &[_]i32{
            @intCast(batch), 1, @intCast(topk),
        }, &[_]i32{});
        defer idx_first.deinit();
        const idx_flat = try ops.reshape(ctx, idx_first, &[_]i32{ @intCast(batch), @intCast(topk) });
        defer idx_flat.deinit();

        // Handle [B, 1, N_blocks, D] shape (expanded for broadcasting)
        var k_3d: Array = undefined;
        var needs_free = false;
        if (k_shape.len == 4) {
            k_3d = try ops.reshape(ctx, k_compressed, &[_]i32{ k_shape[0], k_shape[2], k_shape[3] });
            needs_free = true;
        } else {
            k_3d = k_compressed;
        }
        defer if (needs_free) k_3d.deinit();

        // Gather: for each batch, select topk blocks
        // k_3d: [B, N_blocks, D], idx_flat: [B, topk]
        // We need take_along_axis on dim 1
        const dim = @as(usize, @intCast(k_3d.shape()[2]));

        // Expand indices for gather: [B, topk] -> [B, topk, 1] -> broadcast to [B, topk, D]
        const idx_exp = try ops.expandDims(ctx, idx_flat, 2);
        defer idx_exp.deinit();
        const idx_broadcast = try ops.broadcastTo(ctx, idx_exp, &[_]i32{
            @intCast(batch), @intCast(topk), @intCast(dim),
        });
        defer idx_broadcast.deinit();

        // Use gather via take_along_axis on dim 1
        var res = c.c.mlx_array_new();
        try c.check(c.c.mlx_take_along_axis(&res, k_3d.inner, idx_broadcast.inner, 1, ctx.stream.inner));
        const result = Array.fromHandle(res);
        try result.eval();
        return result;
    }
};

/// MLA (Multi-head Latent Attention).
pub const DSV4Attention = struct {
    ctx: EagerContext,
    config: *const DSV4Config,
    layer_idx: usize,

    // Q projection with lora
    wq_a: Array,
    wq_b: Array,
    q_norm: nn.RMSNorm,

    // KV projection with compression
    wkv: Array,
    kv_norm: nn.RMSNorm,
    kv_b: ?Array,

    // O projection with lora
    wo_a: Array,
    wo_b: Array,

    // RoPE
    rope: DSV4YarnRoPE,

    // Compressor (for CSA/HCA layers)
    compress_ratio: usize,
    compress_gate_weight: ?Array, // [compress_ratio, dim] — learned gate projection
    compress_pos_bias: ?Array, // [compress_ratio] — positional bias for gate scores

    // Lightning Indexer (for CSA layers with sparse block selection)
    indexer: ?LightningIndexer,

    // Attention Sink: learnable logits added to softmax denominator
    // Allows attention scores to sum to < 1 (model can express "nothing relevant")
    sink_logits: ?Array,

    pub fn deinit(self: *DSV4Attention) void {
        self.wq_a.deinit();
        self.wq_b.deinit();
        self.wkv.deinit();
        if (self.kv_b) |kb| kb.deinit();
        self.wo_a.deinit();
        self.wo_b.deinit();
        self.rope.deinit();
        self.q_norm.weight.deinit();
        self.kv_norm.weight.deinit();
        if (self.compress_gate_weight) |gw| gw.deinit();
        if (self.compress_pos_bias) |pb| pb.deinit();
        if (self.indexer) |*idx| {
            var indexer = idx.*;
            indexer.deinit();
        }
        if (self.sink_logits) |sl| sl.deinit();
    }

    pub fn forward(
        self: *DSV4Attention,
        hidden_states: Array,
        mask: ?Array,
        cache: ?kvcache.KVCacheStrategy,
        start_pos: usize,
        stream: c.c.mlx_stream,
    ) !Array {
        _ = mask;
        const shape = hidden_states.shape();
        const batch = @as(usize, @intCast(shape[0]));
        const seq_len = @as(usize, @intCast(shape[1]));
        const num_heads = self.config.num_attention_heads;
        const head_dim = self.config.head_dim;
        const rope_dim = self.config.qk_rope_head_dim;
        const nope_dim = head_dim - rope_dim;
        const window_size = self.config.sliding_window;

        // 1. Q projection: x @ wq_a^T -> q_norm -> q @ wq_b^T
        const wq_a_t = try ops.transpose(self.ctx, self.wq_a);
        defer wq_a_t.deinit();
        const q_a = try ops.matmul(self.ctx, hidden_states, wq_a_t);
        defer q_a.deinit();
        const q_normed = try self.q_norm.forward(q_a);
        defer q_normed.deinit();
        const wq_b_t = try ops.transpose(self.ctx, self.wq_b);
        defer wq_b_t.deinit();
        const q_b = try ops.matmul(self.ctx, q_normed, wq_b_t);
        defer q_b.deinit();

        // Reshape to [B, S, n_heads, head_dim]
        const q_rs = try ops.reshape(self.ctx, q_b, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_heads), @intCast(head_dim) });
        defer q_rs.deinit();

        // Q RMSNorm (per-head): q *= rsqrt(mean(q^2) + eps)
        const q_scaled = try applyRMSNorm(self.ctx, q_rs, self.config.rms_norm_eps);
        defer q_scaled.deinit();

        // Split q into nope and pe parts
        const q_nope = try ops.slice(self.ctx, q_scaled, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_heads), @intCast(nope_dim) }, &[_]i32{});
        defer q_nope.deinit();
        const q_pe = try ops.slice(self.ctx, q_scaled, &[_]i32{ 0, 0, 0, @intCast(nope_dim) }, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_heads), @intCast(head_dim) }, &[_]i32{});
        defer q_pe.deinit();

        // Apply RoPE to q_pe
        // q_pe is [B, S, H, rope_dim]; reshape to [B*H, S, rope_dim] for apply
        const q_pe_rs = try ops.reshape(self.ctx, q_pe, &[_]i32{ @intCast(batch * num_heads), @intCast(seq_len), @intCast(rope_dim) });
        defer q_pe_rs.deinit();
        const q_pe_rot = try self.rope.apply(q_pe_rs, start_pos, stream);
        defer q_pe_rot.deinit();
        const q_pe_rot_rs = try ops.reshape(self.ctx, q_pe_rot, &[_]i32{ @intCast(batch), @intCast(num_heads), @intCast(seq_len), @intCast(rope_dim) });
        defer q_pe_rot_rs.deinit();
        const q_pe_rot_t = try ops.transposeAxes(self.ctx, q_pe_rot_rs, &[_]i32{ 0, 2, 1, 3 });
        defer q_pe_rot_t.deinit();

        // Concat q_nope and q_pe_rot back
        const q_rot = try shape_mod.concatenateAxis(self.ctx, &[_]Array{ q_nope, q_pe_rot_t }, 3);
        defer q_rot.deinit();

        // 2. KV projection: x @ wkv^T -> kv_norm
        const wkv_t = try ops.transpose(self.ctx, self.wkv);
        defer wkv_t.deinit();
        const kv = try ops.matmul(self.ctx, hidden_states, wkv_t);
        defer kv.deinit();
        const kv_normed = try self.kv_norm.forward(kv);
        defer kv_normed.deinit();

        // Apply RoPE to kv[..., -rope_dim:]
        const kv_nope = try ops.slice(self.ctx, kv_normed, &[_]i32{ 0, 0, 0 }, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(nope_dim) }, &[_]i32{});
        defer kv_nope.deinit();
        const kv_pe = try ops.slice(self.ctx, kv_normed, &[_]i32{ 0, 0, @intCast(nope_dim) }, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(head_dim) }, &[_]i32{});
        defer kv_pe.deinit();
        const kv_pe_rot = try self.rope.apply(kv_pe, start_pos, stream);
        defer kv_pe_rot.deinit();

        // FP8 KV storage: cast non-RoPE dims to FP8 (E4M3), RoPE dims to bfloat16.
        // DeepSeek V4 stores most KV dimensions in FP8 (reduced precision) while keeping
        // RoPE dimensions in BF16 for positional encoding fidelity.
        // Uses native mlx_to_fp8/mlx_from_fp8 for true FP8 storage.
        const use_fp8 = self.config.kv_storage_dtype != .float32;
        const kv_nope_stored = if (use_fp8)
            try ops.toFp8(self.ctx, kv_nope)
        else
            kv_nope;
        defer if (use_fp8) kv_nope_stored.deinit();

        const kv_pe_stored = if (use_fp8)
            try ops.astype(self.ctx, kv_pe_rot, .bfloat16)
        else
            kv_pe_rot;
        defer if (use_fp8) kv_pe_stored.deinit();

        // Before concatenation, restore FP8 nope dims to bfloat16 for uniform cache dtype.
        // MLX promotes to wider type during concat, so both must be bfloat16.
        const kv_nope_for_cat = if (use_fp8)
            try ops.fromFp8(self.ctx, kv_nope_stored, .bfloat16)
        else
            kv_nope_stored;
        defer if (use_fp8) kv_nope_for_cat.deinit();

        const kv_rot = try shape_mod.concatenateAxis(self.ctx, &[_]Array{ kv_nope_for_cat, kv_pe_stored }, 2);
        defer kv_rot.deinit();

        // 3. KV compression for CSA/HCA
        const k_compressed = try compressKV(self.ctx, kv_rot, self.compress_ratio, window_size, self.compress_gate_weight, self.compress_pos_bias, stream);
        defer if (self.compress_ratio > 1) k_compressed.deinit();

        // 3b. Lightning Indexer: sparse block selection for CSA
        // If indexer is present and we have compressed blocks, select top-k blocks
        // instead of attending to all compressed blocks.
        const k_final = if (self.compress_ratio > 1 and self.indexer != null) blk: {
            var idx = self.indexer.?;
            // q_rot is [B, S, n_heads, head_dim] — average across heads for indexer query
            const q_for_index = try reduce_mod.meanAxis(self.ctx, q_rot, 2, false);
            defer q_for_index.deinit();
            // k_compressed is [B, N_blocks, D]
            const top_indices = try idx.selectTopK(q_for_index, k_compressed);
            defer top_indices.deinit();
            const selected = try idx.gatherBlocks(k_compressed, top_indices);
            break :blk selected;
        } else if (self.compress_ratio > 1) k_compressed else kv_rot;
        defer if (self.compress_ratio > 1 and self.indexer != null) k_final.deinit();

        // 4. Sparse attention (simplified: full attention within window)
        // Transpose q to [B, n_heads, S, head_dim]
        const q_t = try ops.transposeAxes(self.ctx, q_rot, &[_]i32{ 0, 2, 1, 3 });
        defer q_t.deinit();

        // For MQA, kv is [B, S, head_dim]. Expand to [B, 1, S, head_dim] for broadcasting.
        const kv_exp = try ops.expandDims(self.ctx, k_final, 1);
        defer kv_exp.deinit();

        // KV Cache integration: append new KV and fetch full cached sequence
        var k_full: Array = kv_exp;
        var v_full: Array = kv_exp;
        var kv_slice: ?kvcache.KVSlice = null;
        if (cache) |cache_strategy| {
            kv_slice = try cache_strategy.updateAndFetch(kv_exp, kv_exp, stream);
            k_full = kv_slice.?.keys;
            v_full = kv_slice.?.values;
        }
        defer if (kv_slice) |slice| {
            slice.keys.deinit();
            slice.values.deinit();
        };

        // Cast KV back to float32 before attention computation for numerical accuracy.
        // The cache stores in reduced precision (bfloat16 after FP8 round-trip);
        // attention needs full precision.
        const k_for_attn = if (use_fp8)
            try ops.astype(self.ctx, k_full, .float32)
        else
            k_full;
        defer if (use_fp8) k_for_attn.deinit();

        const v_for_attn = if (use_fp8)
            try ops.astype(self.ctx, v_full, .float32)
        else
            v_full;
        defer if (use_fp8) v_for_attn.deinit();

        // Fast scaled dot-product attention
        const mask_mode: []const u8 = if (start_pos == 0 and seq_len > 1) "causal" else "";
        const scale = self.rope.scale;
        const attn_out = try fast_mod.scaledDotProductAttention(self.ctx, q_t, k_for_attn, v_for_attn, scale, mask_mode, null, self.sink_logits);
        defer attn_out.deinit();

        // Transpose back: [B, n_heads, S, head_dim] -> [B, S, n_heads, head_dim]
        const attn_t = try ops.transposeAxes(self.ctx, attn_out, &[_]i32{ 0, 2, 1, 3 });
        defer attn_t.deinit();

        // Apply inverse RoPE to output pe dimensions
        const out_nope = try ops.slice(self.ctx, attn_t, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_heads), @intCast(nope_dim) }, &[_]i32{});
        defer out_nope.deinit();
        const out_pe = try ops.slice(self.ctx, attn_t, &[_]i32{ 0, 0, 0, @intCast(nope_dim) }, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_heads), @intCast(head_dim) }, &[_]i32{});
        defer out_pe.deinit();
        // out_pe is [B, S, H, rope_dim]; reshape to [B*H, S, rope_dim] for applyInverse
        const out_pe_rs = try ops.reshape(self.ctx, out_pe, &[_]i32{ @intCast(batch * num_heads), @intCast(seq_len), @intCast(rope_dim) });
        defer out_pe_rs.deinit();
        const out_pe_derot = try self.rope.applyInverse(out_pe_rs, start_pos, stream);
        defer out_pe_derot.deinit();
        const out_pe_derot_rs = try ops.reshape(self.ctx, out_pe_derot, &[_]i32{ @intCast(batch), @intCast(num_heads), @intCast(seq_len), @intCast(rope_dim) });
        defer out_pe_derot_rs.deinit();
        const out_pe_derot_t = try ops.transposeAxes(self.ctx, out_pe_derot_rs, &[_]i32{ 0, 2, 1, 3 });
        defer out_pe_derot_t.deinit();
        const o_rot = try shape_mod.concatenateAxis(self.ctx, &[_]Array{ out_nope, out_pe_derot_t }, 3);
        defer o_rot.deinit();

        // 5. O projection with grouped lora
        const o_flat = try ops.reshape(self.ctx, o_rot, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_heads * head_dim) });
        defer o_flat.deinit();

        const out = if (self.wo_a.ndim() == 3) blk: {
            // Grouped LoRA path
            const wo_a_shape = self.wo_a.shape();
            const n_groups = @as(usize, @intCast(wo_a_shape[0]));
            const o_lora_rank = @as(usize, @intCast(wo_a_shape[1]));
            const heads_per_group = num_heads / n_groups;

            // o_rot: [B, S, n_heads, head_dim] -> [B, S, n_groups, heads_per_group, head_dim]
            const o_grouped = try ops.reshape(self.ctx, o_rot, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(n_groups), @intCast(heads_per_group), @intCast(head_dim) });
            defer o_grouped.deinit();

            // Process each group
            var group_outputs = try self.ctx.allocator.alloc(Array, n_groups);
            defer self.ctx.allocator.free(group_outputs);

            for (0..n_groups) |g| {
                // Extract group slice: [B, S, heads_per_group, head_dim]
                const g_slice = try ops.slice(self.ctx, o_grouped, &[_]i32{ 0, 0, @intCast(g), 0, 0 }, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(g + 1), @intCast(heads_per_group), @intCast(head_dim) }, &[_]i32{});
                defer g_slice.deinit();
                const g_flat = try ops.reshape(self.ctx, g_slice, &[_]i32{ @intCast(batch * seq_len), @intCast(heads_per_group * head_dim) });
                defer g_flat.deinit();

                // Extract wo_a for this group: [o_lora_rank, heads_per_group*head_dim]
                const wa_g = try ops.slice(self.ctx, self.wo_a, &[_]i32{ @intCast(g), 0, 0 }, &[_]i32{ @intCast(g + 1), @intCast(o_lora_rank), @intCast(heads_per_group * head_dim) }, &[_]i32{});
                defer wa_g.deinit();
                const wa_g_rs = try ops.reshape(self.ctx, wa_g, &[_]i32{ @intCast(o_lora_rank), @intCast(heads_per_group * head_dim) });
                defer wa_g_rs.deinit();
                const wa_g_t = try ops.transpose(self.ctx, wa_g_rs);
                defer wa_g_t.deinit();

                const g_out = try ops.matmul(self.ctx, g_flat, wa_g_t);
                group_outputs[g] = g_out;
            }
            defer for (group_outputs) |arr| arr.deinit();

            // Concatenate groups: [B*S, n_groups*o_lora_rank]
            const o_a_cat = try shape_mod.concatenateAxis(self.ctx, group_outputs, 1);
            defer o_a_cat.deinit();
            const o_a_rs = try ops.reshape(self.ctx, o_a_cat, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(n_groups * o_lora_rank) });
            defer o_a_rs.deinit();

            const wo_b_t = try ops.transpose(self.ctx, self.wo_b);
            defer wo_b_t.deinit();
            break :blk try ops.matmul(self.ctx, o_a_rs, wo_b_t);
        } else blk: {
            // Standard LoRA path
            const wo_a_t = try ops.transpose(self.ctx, self.wo_a);
            defer wo_a_t.deinit();
            const o_a = try ops.matmul(self.ctx, o_flat, wo_a_t);
            defer o_a.deinit();
            const wo_b_t = try ops.transpose(self.ctx, self.wo_b);
            defer wo_b_t.deinit();
            break :blk try ops.matmul(self.ctx, o_a, wo_b_t);
        };

        return out;
    }
};

/// Compress KV cache along sequence dimension for CSA/HCA layers.
/// - ratio <= 1: passthrough (no compression)
/// - ratio > 1 with gate weights: learned softmax-gated pooling with positional bias
/// - ratio > 1 without gate weights: fallback to mean-pool
pub fn compressKV(
    ctx: EagerContext,
    kv: Array, // [B, S, D]
    compress_ratio: usize,
    window_size: usize,
    gate_weight: ?Array, // [compress_ratio, dim] — learned gate projection
    pos_bias: ?Array, // [compress_ratio] — positional bias
    stream: c.c.mlx_stream,
) !Array {
    _ = stream;
    if (compress_ratio <= 1) return try ops.copy(ctx, kv);

    const shape = kv.shape();
    const batch = @as(usize, @intCast(shape[0]));
    const seq_len = @as(usize, @intCast(shape[1]));
    const dim = @as(usize, @intCast(shape[2]));

    if (seq_len <= window_size) return try ops.copy(ctx, kv);

    const prefix_len = seq_len - window_size;
    const num_groups = prefix_len / compress_ratio;
    const remainder = prefix_len % compress_ratio;

    // Sliding-window suffix is always kept uncompressed
    const suffix = try ops.slice(ctx, kv, &[_]i32{ 0, @intCast(prefix_len), 0 }, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(dim) }, &[_]i32{});
    defer suffix.deinit();

    if (num_groups == 0) return try ops.copy(ctx, kv);

    // Compressible prefix
    const prefix = try ops.slice(ctx, kv, &[_]i32{ 0, 0, 0 }, &[_]i32{ @intCast(batch), @intCast(num_groups * compress_ratio), @intCast(dim) }, &[_]i32{});
    defer prefix.deinit();

    // Reshape to [B, num_groups, compress_ratio, D]
    const prefix_rs = try ops.reshape(ctx, prefix, &[_]i32{ @intCast(batch), @intCast(num_groups), @intCast(compress_ratio), @intCast(dim) });
    defer prefix_rs.deinit();

    const prefix_comp = if (gate_weight != null and pos_bias != null)
        try softmaxGatedPool(ctx, prefix_rs, gate_weight.?, pos_bias.?, batch, num_groups, compress_ratio, dim)
    else
        try reduce_mod.meanAxis(ctx, prefix_rs, 2, false);
    defer prefix_comp.deinit();

    if (remainder > 0) {
        const rem = try ops.slice(ctx, kv, &[_]i32{ 0, @intCast(num_groups * compress_ratio), 0 }, &[_]i32{ @intCast(batch), @intCast(prefix_len), @intCast(dim) }, &[_]i32{});
        defer rem.deinit();

        const rem_rs = try ops.reshape(ctx, rem, &[_]i32{ @intCast(batch), 1, @intCast(remainder), @intCast(dim) });
        defer rem_rs.deinit();

        const rem_comp = if (gate_weight != null and pos_bias != null)
            try softmaxGatedPoolRemainder(ctx, rem_rs, gate_weight.?, pos_bias.?, batch, remainder, dim)
        else
            try reduce_mod.meanAxis(ctx, rem_rs, 2, false);
        defer rem_comp.deinit();

        const raw = try ctx.allocator.alloc(Array, 3);
        defer ctx.allocator.free(raw);
        raw[0] = prefix_comp;
        raw[1] = rem_comp;
        raw[2] = suffix;
        const result = try shape_mod.concatenateAxis(ctx, raw, 1);
        try result.eval();
        return result;
    } else {
        const raw = try ctx.allocator.alloc(Array, 2);
        defer ctx.allocator.free(raw);
        raw[0] = prefix_comp;
        raw[1] = suffix;
        const result = try shape_mod.concatenateAxis(ctx, raw, 1);
        try result.eval();
        return result;
    }
}

/// Learned softmax-gated pooling for the main prefix groups.
/// prefix_rs: [B, num_groups, compress_ratio, D]
/// gate_weight: [compress_ratio, D]
/// pos_bias: [compress_ratio]
/// Returns: [B, num_groups, D]
fn softmaxGatedPool(
    ctx: EagerContext,
    prefix_rs: Array, // [B, num_groups, compress_ratio, D]
    gate_weight: Array, // [compress_ratio, D]
    pos_bias: Array, // [compress_ratio]
    batch: usize,
    num_groups: usize,
    compress_ratio: usize,
    dim: usize,
) !Array {
    // Flatten to [B*num_groups, compress_ratio, D] for matmul
    const flat = try ops.reshape(ctx, prefix_rs, &[_]i32{ @intCast(batch * num_groups), @intCast(compress_ratio), @intCast(dim) });
    defer flat.deinit();

    // gate_weight^T: [D, compress_ratio]
    const gw_t = try ops.transpose(ctx, gate_weight);
    defer gw_t.deinit();

    // Gate scores: [B*num_groups, compress_ratio, D] @ [D, compress_ratio] -> [B*num_groups, compress_ratio, compress_ratio]
    // We want per-position scores, so we do: each position's D-dim vector dot gate_weight row
    // Simpler: reshape flat to [B*num_groups*compress_ratio, D], matmul with gw_t, reshape back
    const flat2 = try ops.reshape(ctx, flat, &[_]i32{ @intCast(batch * num_groups * compress_ratio), @intCast(dim) });
    defer flat2.deinit();

    // gate_weight: [compress_ratio, D] — we want each position to produce a scalar score
    // For each position i in compress_ratio, score_i = dot(x_i, gate_weight[i])
    // This is a diagonal operation: element-wise multiply then sum
    // But we can compute all scores as: (flat2 @ gate_weight^T) and take the diagonal
    // More efficiently: element-wise multiply flat with gate_weight (broadcast) and sum over D
    // flat: [B*num_groups, compress_ratio, D], gate_weight: [compress_ratio, D]
    // Broadcast multiply: [B*num_groups, compress_ratio, D]
    const gated = try ops.multiply(ctx, flat, gate_weight);
    defer gated.deinit();

    // Sum over D dimension to get scores: [B*num_groups, compress_ratio]
    const scores_raw = try reduce_mod.sumAxis(ctx, gated, 2, false);
    defer scores_raw.deinit();

    // Add positional bias: [compress_ratio] broadcasts to [B*num_groups, compress_ratio]
    const scores_biased = try ops.add(ctx, scores_raw, pos_bias);
    defer scores_biased.deinit();

    // Softmax over compress_ratio dimension (axis -1)
    const weights = try ops.softmax(ctx, scores_biased, &[_]i32{-1});
    defer weights.deinit();

    // Expand weights for broadcast multiply: [B*num_groups, compress_ratio, 1]
    const weights_exp = try ops.expandDims(ctx, weights, 2);
    defer weights_exp.deinit();

    // Weighted sum: [B*num_groups, compress_ratio, D] * [B*num_groups, compress_ratio, 1] -> sum over axis 1
    const weighted = try ops.multiply(ctx, flat, weights_exp);
    defer weighted.deinit();

    // Sum over compress_ratio dimension: [B*num_groups, D]
    const pooled = try reduce_mod.sumAxis(ctx, weighted, 1, false);
    defer pooled.deinit();

    // Reshape back to [B, num_groups, D]
    return ops.reshape(ctx, pooled, &[_]i32{ @intCast(batch), @intCast(num_groups), @intCast(dim) });
}

/// Learned softmax-gated pooling for the remainder group.
/// rem_rs: [B, 1, remainder, D]
/// gate_weight: [compress_ratio, D] — we use only the first `remainder` rows
/// pos_bias: [compress_ratio] — we use only the first `remainder` elements
/// Returns: [B, 1, D]
fn softmaxGatedPoolRemainder(
    ctx: EagerContext,
    rem_rs: Array, // [B, 1, remainder, D]
    gate_weight: Array, // [compress_ratio, D]
    pos_bias: Array, // [compress_ratio]
    batch: usize,
    remainder: usize,
    dim: usize,
) !Array {
    // Slice gate_weight to [remainder, D]
    const gw_rem = try ops.slice(ctx, gate_weight, &[_]i32{ 0, 0 }, &[_]i32{ @intCast(remainder), @intCast(dim) }, &[_]i32{});
    defer gw_rem.deinit();

    // Slice pos_bias to [remainder]
    const pb_rem = try ops.slice(ctx, pos_bias, &[_]i32{0}, &[_]i32{@intCast(remainder)}, &[_]i32{});
    defer pb_rem.deinit();

    // Reshape rem_rs to [B, remainder, D]
    const flat = try ops.reshape(ctx, rem_rs, &[_]i32{ @intCast(batch), @intCast(remainder), @intCast(dim) });
    defer flat.deinit();

    // Element-wise multiply: [B, remainder, D] * [remainder, D] (broadcast over B)
    const gated = try ops.multiply(ctx, flat, gw_rem);
    defer gated.deinit();

    // Sum over D: [B, remainder]
    const scores_raw = try reduce_mod.sumAxis(ctx, gated, 2, false);
    defer scores_raw.deinit();

    // Add positional bias: [B, remainder]
    const scores_biased = try ops.add(ctx, scores_raw, pb_rem);
    defer scores_biased.deinit();

    // Softmax over remainder dimension
    const weights = try ops.softmax(ctx, scores_biased, &[_]i32{-1});
    defer weights.deinit();

    // Expand: [B, remainder, 1]
    const weights_exp = try ops.expandDims(ctx, weights, 2);
    defer weights_exp.deinit();

    // Weighted sum: [B, remainder, D] * [B, remainder, 1]
    const weighted = try ops.multiply(ctx, flat, weights_exp);
    defer weighted.deinit();

    // Sum over remainder dim: [B, D]
    const pooled = try reduce_mod.sumAxis(ctx, weighted, 1, false);
    defer pooled.deinit();

    // Reshape to [B, 1, D]
    return ops.reshape(ctx, pooled, &[_]i32{ @intCast(batch), 1, @intCast(dim) });
}

/// Expand hidden states from [..., H] to [..., mhc_mult, H].
pub fn expandToMHC(ctx: EagerContext, hidden: Array, mhc_mult: usize, stream: c.c.mlx_stream) !Array {
    _ = stream;
    const shape = hidden.shape();
    const ndim = shape.len;
    const h = @as(usize, @intCast(shape[ndim - 1]));
    const unsqueezed = try ops.expandDims(ctx, hidden, @intCast(ndim - 1));
    defer unsqueezed.deinit();
    var new_shape = try ctx.allocator.alloc(i32, ndim + 1);
    defer ctx.allocator.free(new_shape);
    for (0..ndim - 1) |i| new_shape[i] = shape[i];
    new_shape[ndim - 1] = @intCast(mhc_mult);
    new_shape[ndim] = @intCast(h);
    return ops.broadcastTo(ctx, unsqueezed, new_shape);
}

/// MHC pre-norm fn: computes mixes from residual and fn weights.
/// residual: [B, S, mhc_mult, H]
/// fn_arr: [mhc_mult3, mhc_mult * H]
/// norm_weight: optional [mhc_mult * H]
/// Returns: mixes [B, S, mhc_mult3]
pub fn mhcPreNormFn(
    ctx: EagerContext,
    residual: Array,
    fn_arr: Array,
    norm_weight: ?Array,
    eps: f32,
    stream: c.c.mlx_stream,
) !Array {
    _ = stream;
    const fn_shape = fn_arr.shape();
    const mhc_mult3 = @as(usize, @intCast(fn_shape[0]));
    const mhc_hidden_size = @as(usize, @intCast(fn_shape[1]));
    const res_shape = residual.shape();
    const batch = @as(usize, @intCast(res_shape[0]));
    const seq_len = @as(usize, @intCast(res_shape[1]));

    const res_flat = try ops.reshape(ctx, residual, &[_]i32{ @intCast(batch * seq_len), @intCast(mhc_hidden_size) });
    defer res_flat.deinit();

    const fn_actual = if (norm_weight) |nw| blk: {
        const fn_scaled = try ops.multiply(ctx, fn_arr, nw);
        break :blk fn_scaled;
    } else fn_arr;
    defer if (norm_weight != null) fn_actual.deinit();

    const fn_t = try ops.transpose(ctx, fn_actual);
    defer fn_t.deinit();
    const mixes_flat = try ops.matmul(ctx, res_flat, fn_t);
    defer mixes_flat.deinit();

    const res_sq = try math_mod.square(ctx, res_flat);
    defer res_sq.deinit();
    const sqrsum = try reduce_mod.sumAxis(ctx, res_sq, 1, false);
    defer sqrsum.deinit();

    const group_size_f = try Array.fromData(ctx.allocator, f32, &[_]f32{ @floatFromInt(mhc_hidden_size) }, &[_]i32{1});
    defer group_size_f.deinit();
    const sqrsum_div = try ops.divide(ctx, sqrsum, group_size_f);
    defer sqrsum_div.deinit();
    const eps_arr = try Array.fromData(ctx.allocator, f32, &[_]f32{eps}, &[_]i32{1});
    defer eps_arr.deinit();
    const sqrsum_eps = try ops.add(ctx, sqrsum_div, eps_arr);
    defer sqrsum_eps.deinit();
    const norm = try math_mod.rsqrt(ctx, sqrsum_eps);
    defer norm.deinit();

    const norm_expanded = try ops.expandDims(ctx, norm, 1);
    defer norm_expanded.deinit();
    const mixes_scaled = try ops.multiply(ctx, mixes_flat, norm_expanded);
    defer mixes_scaled.deinit();

    const result = try ops.reshape(ctx, mixes_scaled, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(mhc_mult3) });
    try result.eval();
    return result;
}

/// Split mixes into pre_layer_mix, post_layer_mix, comb_res_mix.
/// mixes: [B, S, mhc_mult3]
/// scale: [3]
/// base: [mhc_mult3]
/// Returns: (pre_mix [B,S,mhc_mult,1], post_mix [B,S,mhc_mult,1], comb_mix [B,S,mhc_mult,mhc_mult])
pub fn mhcPreSplitMixes(
    ctx: EagerContext,
    mixes: Array,
    scale: Array,
    base: Array,
    mhc_mult: usize,
    post_mult_value: f32,
    pre_eps: f32,
    stream: c.c.mlx_stream,
) !struct { Array, Array, Array } {
    _ = stream;
    const shape = mixes.shape();
    const batch = @as(usize, @intCast(shape[0]));
    const seq_len = @as(usize, @intCast(shape[1]));
    const mhc_mult_sq = mhc_mult * mhc_mult;

    // Build expanded scale on CPU to avoid complex concat ops
    const scale_vals = try scale.dataSlice(f32);
    var scale_data = try ctx.allocator.alloc(f32, mhc_mult * 2 + mhc_mult_sq);
    defer ctx.allocator.free(scale_data);
    for (0..mhc_mult) |i| scale_data[i] = scale_vals[0];
    for (0..mhc_mult) |i| scale_data[mhc_mult + i] = scale_vals[1];
    for (0..mhc_mult_sq) |i| scale_data[2 * mhc_mult + i] = scale_vals[2];
    const scale_expanded = try Array.fromData(ctx.allocator, f32, scale_data, &[_]i32{ @intCast(mhc_mult * 2 + mhc_mult_sq) });
    defer scale_expanded.deinit();

    const mixes_scaled = try ops.multiply(ctx, mixes, scale_expanded);
    defer mixes_scaled.deinit();
    const mixes_biased = try ops.add(ctx, mixes_scaled, base);
    defer mixes_biased.deinit();

    const pre_mix_slice = try ops.slice(ctx, mixes_biased, &[_]i32{ 0, 0, 0 }, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(mhc_mult) }, &[_]i32{ 1, 1, 1 });
    defer pre_mix_slice.deinit();
    const post_mix_slice = try ops.slice(ctx, mixes_biased, &[_]i32{ 0, 0, @intCast(mhc_mult) }, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(2 * mhc_mult) }, &[_]i32{ 1, 1, 1 });
    defer post_mix_slice.deinit();
    const comb_mix_slice = try ops.slice(ctx, mixes_biased, &[_]i32{ 0, 0, @intCast(2 * mhc_mult) }, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(mhc_mult * 2 + mhc_mult_sq) }, &[_]i32{ 1, 1, 1 });
    defer comb_mix_slice.deinit();

    const pre_sigmoid = try ops.sigmoid(ctx, pre_mix_slice);
    defer pre_sigmoid.deinit();
    const pre_eps_arr = try Array.fromData(ctx.allocator, f32, &[_]f32{pre_eps}, &[_]i32{1});
    defer pre_eps_arr.deinit();
    const pre_mix = try ops.add(ctx, pre_sigmoid, pre_eps_arr);
    defer pre_mix.deinit();
    const pre_mix_expanded = try ops.expandDims(ctx, pre_mix, 3);
    try pre_mix_expanded.eval();

    const post_sigmoid = try ops.sigmoid(ctx, post_mix_slice);
    defer post_sigmoid.deinit();
    const post_mult_arr = try Array.fromData(ctx.allocator, f32, &[_]f32{post_mult_value}, &[_]i32{1});
    defer post_mult_arr.deinit();
    const post_mix = try ops.multiply(ctx, post_sigmoid, post_mult_arr);
    defer post_mix.deinit();
    const post_mix_expanded = try ops.expandDims(ctx, post_mix, 3);
    try post_mix_expanded.eval();

    const comb_mix = try ops.reshape(ctx, comb_mix_slice, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(mhc_mult), @intCast(mhc_mult) });
    try comb_mix.eval();

    return .{ pre_mix_expanded, post_mix_expanded, comb_mix };
}

/// Sinkhorn normalization on comb_res_mix.
pub fn sinkhornNormalize(
    ctx: EagerContext,
    x: Array,
    repeat: usize,
    eps: f32,
    stream: c.c.mlx_stream,
) !Array {
    _ = stream;
    const softmaxed = try ops.softmax(ctx, x, &[_]i32{ -1 });
    defer softmaxed.deinit();

    const eps_arr = try Array.fromData(ctx.allocator, f32, &[_]f32{eps}, &[_]i32{1});
    defer eps_arr.deinit();
    var current = try ops.add(ctx, softmaxed, eps_arr);

    // Initial col normalization
    const col_sum0 = try reduce_mod.sumAxis(ctx, current, -2, true);
    const col_sum0_eps = try ops.add(ctx, col_sum0, eps_arr);
    const next0 = try ops.divide(ctx, current, col_sum0_eps);
    col_sum0.deinit();
    col_sum0_eps.deinit();
    current.deinit();
    current = next0;

    var i: usize = 0;
    while (i < repeat - 1) : (i += 1) {
        const row_sum = try reduce_mod.sumAxis(ctx, current, -1, true);
        const row_sum_eps = try ops.add(ctx, row_sum, eps_arr);
        const row_normed = try ops.divide(ctx, current, row_sum_eps);
        row_sum.deinit();
        row_sum_eps.deinit();
        current.deinit();

        const col_sum = try reduce_mod.sumAxis(ctx, row_normed, -2, true);
        const col_sum_eps = try ops.add(ctx, col_sum, eps_arr);
        const next = try ops.divide(ctx, row_normed, col_sum_eps);
        col_sum.deinit();
        col_sum_eps.deinit();
        row_normed.deinit();

        current = next;
    }

    return current;
}

/// Apply pre-layer mix to residual.
pub fn mhcPreApplyMix(
    ctx: EagerContext,
    residual: Array,
    mix: Array,
    stream: c.c.mlx_stream,
) !Array {
    _ = stream;
    const weighted = try ops.multiply(ctx, residual, mix);
    defer weighted.deinit();
    const result = try reduce_mod.sumAxis(ctx, weighted, 2, false);
    try result.eval();
    return result;
}

/// MHC post: combine sublayer output with residual using post_mix and comb_mix.
pub fn mhcPost(
    ctx: EagerContext,
    x: Array,
    residual: Array,
    post_mix: Array,
    comb_mix: Array,
    stream: c.c.mlx_stream,
) !Array {
    _ = stream;
    const shape = x.shape();
    const batch = @as(usize, @intCast(shape[0]));
    const seq_len = @as(usize, @intCast(shape[1]));
    const h = @as(usize, @intCast(shape[2]));
    const res_shape = residual.shape();
    const mhc_mult = @as(usize, @intCast(res_shape[2]));

    const x_expanded = try ops.expandDims(ctx, x, 2);
    defer x_expanded.deinit();
    const term1 = try ops.multiply(ctx, x_expanded, post_mix);
    defer term1.deinit();

    const bs = batch * seq_len;
    const comb_2d = try ops.reshape(ctx, comb_mix, &[_]i32{ @intCast(bs), @intCast(mhc_mult), @intCast(mhc_mult) });
    defer comb_2d.deinit();
    const res_2d = try ops.reshape(ctx, residual, &[_]i32{ @intCast(bs), @intCast(mhc_mult), @intCast(h) });
    defer res_2d.deinit();
    const term2_2d = try ops.matmul(ctx, comb_2d, res_2d);
    defer term2_2d.deinit();
    const term2 = try ops.reshape(ctx, term2_2d, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(mhc_mult), @intCast(h) });
    defer term2.deinit();

    const result = try ops.add(ctx, term1, term2);
    try result.eval();
    return result;
}

/// MHC head compression: compress residual from [B,S,mhc_mult,H] to [B,S,H] for lm_head.
pub fn mhcHeadCompress(
    ctx: EagerContext,
    residual: Array,
    fn_arr: Array,
    scale: Array,
    base: Array,
    mhc_mult: usize,
    eps: f32,
    stream: c.c.mlx_stream,
) !Array {
    const mhc_mult3 = mhc_mult * (mhc_mult + 2);
    const fn_shape = fn_arr.shape();
    const fn_rows = @as(usize, @intCast(fn_shape[0]));

    const fn_padded = if (fn_rows < mhc_mult3) blk: {
        const pad_rows = mhc_mult3 - fn_rows;
        const zeros = try array_mod.zeros(ctx.allocator, &[_]i32{ @intCast(pad_rows), fn_shape[1] }, dtype_mod.float32);
        defer zeros.deinit();
        const padded = try shape_mod.concatenateAxis(ctx, &[_]Array{ fn_arr, zeros }, 0);
        break :blk padded;
    } else fn_arr;
    defer if (fn_rows < mhc_mult3) fn_padded.deinit();

    const mixes = try mhcPreNormFn(ctx, residual, fn_padded, null, eps, stream);
    defer mixes.deinit();

    const shape = mixes.shape();
    const batch = @as(usize, @intCast(shape[0]));
    const seq_len = @as(usize, @intCast(shape[1]));
    const mix_slice = try ops.slice(ctx, mixes, &[_]i32{ 0, 0, 0 }, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(mhc_mult) }, &[_]i32{ 1, 1, 1 });
    defer mix_slice.deinit();

    const mix_scaled = try ops.multiply(ctx, mix_slice, scale);
    defer mix_scaled.deinit();
    const mix_biased = try ops.add(ctx, mix_scaled, base);
    defer mix_biased.deinit();
    const mix_sigmoid = try ops.sigmoid(ctx, mix_biased);
    defer mix_sigmoid.deinit();
    const eps_arr = try Array.fromData(ctx.allocator, f32, &[_]f32{eps}, &[_]i32{1});
    defer eps_arr.deinit();
    const mix = try ops.add(ctx, mix_sigmoid, eps_arr);
    defer mix.deinit();

    const mix_expanded = try ops.expandDims(ctx, mix, 3);
    defer mix_expanded.deinit();

    return mhcPreApplyMix(ctx, residual, mix_expanded, stream);
}

/// mHC (Manifold-Constrained Hyper-Connections) parameters.
pub const DSV4HyperConn = struct {
    hc_fn: Array,
    hc_base: Array,
    hc_scale: Array,
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f32,

    pub fn deinit(self: *DSV4HyperConn) void {
        self.hc_fn.deinit();
        self.hc_base.deinit();
        self.hc_scale.deinit();
    }

    pub fn pre(self: *DSV4HyperConn, ctx: EagerContext, x: Array, stream: c.c.mlx_stream) !struct { Array, Array, Array } {
        const fn_shape = self.hc_fn.shape();
        const expected_rows = self.hc_mult * (self.hc_mult + 2);
        if (fn_shape.len < 2 or @as(usize, @intCast(fn_shape[0])) != expected_rows) {
            return error.InvalidMHCWeights;
        }
        const mixes = try mhcPreNormFn(ctx, x, self.hc_fn, null, self.hc_eps, stream);
        defer mixes.deinit();
        const pre_mix, const post_mix, const comb_mix = try mhcPreSplitMixes(ctx, mixes, self.hc_scale, self.hc_base, self.hc_mult, 1.0, self.hc_eps, stream);
        defer pre_mix.deinit();

        const comb_mix_norm = try sinkhornNormalize(ctx, comb_mix, self.hc_sinkhorn_iters, self.hc_eps, stream);
        try comb_mix_norm.eval();
        defer comb_mix.deinit();

        const layer_input = try mhcPreApplyMix(ctx, x, pre_mix, stream);
        return .{ layer_input, post_mix, comb_mix_norm };
    }

    pub fn post(self: *DSV4HyperConn, ctx: EagerContext, x: Array, residual: Array, post_weights: Array, comb: Array, stream: c.c.mlx_stream) !Array {
        _ = self;
        return mhcPost(ctx, x, residual, post_weights, comb, stream);
    }
};

/// Single transformer block with mHC.
pub const DSV4TransformerBlock = struct {
    ctx: EagerContext,
    config: *const DSV4Config,
    layer_idx: usize,
    attn_norm: nn.RMSNorm,
    ffn_norm: nn.RMSNorm,
    attn: DSV4Attention,
    ffn: DSV4MoE,
    hc_attn: DSV4HyperConn,
    hc_ffn: DSV4HyperConn,

    pub fn deinit(self: *DSV4TransformerBlock) void {
        self.attn_norm.weight.deinit();
        self.ffn_norm.weight.deinit();
        self.attn.deinit();
        self.ffn.deinit();
        self.hc_attn.deinit();
        self.hc_ffn.deinit();
    }

    pub fn forward(
        self: *DSV4TransformerBlock,
        hidden_states: Array,
        input_ids: ?Array,
        mask: ?Array,
        cache: ?kvcache.KVCacheStrategy,
        start_pos: usize,
        stream: c.c.mlx_stream,
    ) !Array {
        const use_mhc = self.config.use_mhc and self.hc_attn.hc_fn.shape().len >= 2;

        if (use_mhc) {
            // mHC path
            const attn_input, const attn_post_mix, const attn_comb_mix = try self.hc_attn.pre(self.ctx, hidden_states, stream);
            defer attn_post_mix.deinit();
            defer attn_comb_mix.deinit();
            defer attn_input.deinit();

            const attn_normed = try self.attn_norm.forward(attn_input);
            defer attn_normed.deinit();
            const attn_out = try self.attn.forward(attn_normed, mask, cache, start_pos, stream);
            defer attn_out.deinit();
            const after_attn = try self.hc_attn.post(self.ctx, attn_out, hidden_states, attn_post_mix, attn_comb_mix, stream);
            defer after_attn.deinit();

            const ffn_input, const ffn_post_mix, const ffn_comb_mix = try self.hc_ffn.pre(self.ctx, after_attn, stream);
            defer ffn_post_mix.deinit();
            defer ffn_comb_mix.deinit();
            defer ffn_input.deinit();

            const ffn_normed = try self.ffn_norm.forward(ffn_input);
            defer ffn_normed.deinit();
            const input_ids_actual = if (input_ids) |ids| ids else blk: {
                const shape = ffn_normed.shape();
                const ids_arr = try array_mod.zeros(self.ctx.allocator, &[_]i32{ shape[0], shape[1] }, .int32);
                break :blk ids_arr;
            };
            defer if (input_ids == null) input_ids_actual.deinit();
            const ffn_out = try self.ffn.forward(ffn_normed, input_ids_actual, stream);
            defer ffn_out.deinit();
            const after_ffn = try self.hc_ffn.post(self.ctx, ffn_out, after_attn, ffn_post_mix, ffn_comb_mix, stream);

            return after_ffn;
        } else {
            // Standard residual connection (mHC disabled)
            const attn_normed = try self.attn_norm.forward(hidden_states);
            defer attn_normed.deinit();
            const attn_out = try self.attn.forward(attn_normed, mask, cache, start_pos, stream);
            defer attn_out.deinit();
            const after_attn = try ops.add(self.ctx, hidden_states, attn_out);
            defer after_attn.deinit();

            const ffn_normed = try self.ffn_norm.forward(after_attn);
            defer ffn_normed.deinit();
            const input_ids_actual = if (input_ids) |ids| ids else blk: {
                const shape = ffn_normed.shape();
                const ids_arr = try array_mod.zeros(self.ctx.allocator, &[_]i32{ shape[0], shape[1] }, .int32);
                break :blk ids_arr;
            };
            defer if (input_ids == null) input_ids_actual.deinit();
            const ffn_out = try self.ffn.forward(ffn_normed, input_ids_actual, stream);
            defer ffn_out.deinit();
            const after_ffn = try ops.add(self.ctx, after_attn, ffn_out);

            return after_ffn;
        }
    }
};

/// Full DeepSeek-V4 model.
pub const DSV4Model = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    config: DSV4Config,
    embed_tokens: nn.Embedding,
    layers: []DSV4TransformerBlock,
    norm: nn.RMSNorm,
    head_norm: ?nn.RMSNorm,
    lm_head: Array,

    pub fn deinit(self: *DSV4Model) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
        self.norm.weight.deinit();
        if (self.head_norm) |*hn| {
            hn.weight.deinit();
        }
        self.lm_head.deinit();
    }

    pub fn forward(
        self: *DSV4Model,
        input_ids: Array,
        mask: ?Array,
        caches: ?[]kvcache.KVCacheStrategy,
        start_pos: usize,
        stream: c.c.mlx_stream,
    ) !Array {
        var arena = ScopedArrayArena.init(self.allocator);
        defer arena.deinit();

        // Embedding lookup
        var hidden = try arena.track(try self.embed_tokens.forward(input_ids));

        // Expand to mHC format if enabled
        if (self.config.use_mhc) {
            hidden = try arena.track(try expandToMHC(self.ctx, hidden, self.config.hc_mult, stream));
        }

        // Pass through layers
        for (self.layers, 0..) |*layer, i| {
            const cache = if (caches) |cache_arr| cache_arr[i] else null;
            hidden = try arena.track(try layer.forward(hidden, input_ids, mask, cache, start_pos, stream));
        }

        // Compress from mHC format before final norm if needed
        if (self.config.use_mhc and hidden.shape().len == 4) {
            hidden = try arena.track(try reduce_mod.meanAxis(self.ctx, hidden, 2, false));
        }

        // Final norm
        var final_hidden = try arena.track(try self.norm.forward(hidden));

        // Optional head norm
        if (self.head_norm) |*hn| {
            final_hidden = try arena.track(try hn.forward(final_hidden));
        }

        // LM head: [B, S, dim] @ [dim, vocab] -> [B, S, vocab] — final output NOT tracked
        const lm_head_t = try arena.track(try ops.transpose(self.ctx, self.lm_head));
        const logits = try ops.matmul(self.ctx, final_hidden, lm_head_t);
        return logits;
    }

    pub fn generate(
        self: *DSV4Model,
        prompt_tokens: []const u32,
        max_new_tokens: usize,
        sampler_config: *@import("../sampling.zig").SamplerConfig,
        caches: []kvcache.KVCacheStrategy,
        stream: c.c.mlx_stream,
    ) ![]u32 {
        const allocator = self.allocator;
        var tokens = try allocator.alloc(u32, prompt_tokens.len + max_new_tokens);
        defer allocator.free(tokens);
        @memcpy(tokens[0..prompt_tokens.len], prompt_tokens);

        var current_len = prompt_tokens.len;
        var start_pos: usize = 0;

        // Prefill: process all prompt tokens at once
        if (prompt_tokens.len > 0) {
            var arena = ScopedArrayArena.init(allocator);
            defer arena.deinit();

            const prompt_arr = try arena.track(try Array.fromData(allocator, u32, prompt_tokens, &[_]i32{1, @intCast(prompt_tokens.len)}));
            const logits = try self.forward(prompt_arr, null, caches, start_pos, stream);

            // Get last token logits
            const last_logits = try arena.track(try ops.slice(self.ctx, logits, &[_]i32{ 0, @intCast(prompt_tokens.len - 1), 0 }, &[_]i32{ 1, @intCast(prompt_tokens.len), @intCast(self.config.vocab_size) }, &[_]i32{}));
            const squeezed = try arena.track(try shape_mod.squeezeAxes(self.ctx, last_logits, &[_]i32{0}));
            const f32_logits = try arena.track(try ops.astype(self.ctx, squeezed, .float32));

            const next_token = try sampler_config.sample(f32_logits, allocator);
            tokens[current_len] = next_token;
            current_len += 1;
            start_pos = prompt_tokens.len;
        }

        // Generate new tokens autoregressively
        for (0..max_new_tokens - 1) |_| {
            if (current_len >= tokens.len) break;

            var arena = ScopedArrayArena.init(allocator);
            defer arena.deinit();

            const input_arr = try arena.track(try Array.fromData(allocator, u32, &[_]u32{tokens[current_len - 1]}, &[_]i32{1, 1}));
            const logits = try self.forward(input_arr, null, caches, start_pos, stream);

            const squeezed = try arena.track(try shape_mod.squeezeAxes(self.ctx, logits, &[_]i32{0, 1}));
            const f32_logits = try arena.track(try ops.astype(self.ctx, squeezed, .float32));

            const next_token = try sampler_config.sample(f32_logits, allocator);
            tokens[current_len] = next_token;
            current_len += 1;
            start_pos += 1;
        }

        // Return only generated tokens (skip prompt)
        const result = try allocator.alloc(u32, current_len - prompt_tokens.len);
        @memcpy(result, tokens[prompt_tokens.len..current_len]);
        return result;
    }
};

// ============================================================
// Lightning Indexer Unit Tests
// ============================================================

test "LightningIndexer.selectTopK returns correct shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = ctx.stream.inner;

    const batch: usize = 2;
    const seq_len: usize = 4;
    const head_dim: usize = 128;
    const n_blocks: usize = 32;
    const index_n_heads: usize = 4;
    const index_head_dim: usize = 64;
    const index_topk: usize = 8;
    const index_dim = index_n_heads * index_head_dim;

    // Create random Q: [B, S, head_dim]
    var q_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(&q_raw, (&[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(head_dim) }).ptr, 3, c.c.MLX_FLOAT32, 0.0, 1.0, .{ .ctx = null }, stream));
    const q = Array.fromHandle(q_raw);
    defer q.deinit();

    // Create random K: [B, N_blocks, head_dim]
    var k_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(&k_raw, (&[_]i32{ @intCast(batch), @intCast(n_blocks), @intCast(head_dim) }).ptr, 3, c.c.MLX_FLOAT32, 0.0, 1.0, .{ .ctx = null }, stream));
    const k = Array.fromHandle(k_raw);
    defer k.deinit();

    // Create indexer weights
    var wq_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(&wq_raw, (&[_]i32{ @intCast(index_dim), @intCast(head_dim) }).ptr, 2, c.c.MLX_FLOAT32, 0.0, 0.1, .{ .ctx = null }, stream));
    const wq = Array.fromHandle(wq_raw);

    var wk_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(&wk_raw, (&[_]i32{ @intCast(index_dim), @intCast(head_dim) }).ptr, 2, c.c.MLX_FLOAT32, 0.0, 0.1, .{ .ctx = null }, stream));
    const wk = Array.fromHandle(wk_raw);

    var indexer = LightningIndexer{
        .ctx = ctx,
        .wq_index = wq,
        .wk_index = wk,
        .index_n_heads = index_n_heads,
        .index_head_dim = index_head_dim,
        .index_topk = index_topk,
    };
    defer indexer.deinit();

    const result = try indexer.selectTopK(q, k);
    defer result.deinit();

    const result_shape = result.shape();
    try std.testing.expectEqual(@as(usize, 3), result_shape.len);
    try std.testing.expectEqual(@as(i32, @intCast(batch)), result_shape[0]);
    try std.testing.expectEqual(@as(i32, @intCast(seq_len)), result_shape[1]);
    try std.testing.expectEqual(@as(i32, @intCast(index_topk)), result_shape[2]);
}

test "LightningIndexer.selectTopK returns all indices when n_blocks <= topk" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = ctx.stream.inner;

    const batch: usize = 1;
    const seq_len: usize = 2;
    const head_dim: usize = 128;
    const n_blocks: usize = 4; // fewer than topk
    const index_n_heads: usize = 2;
    const index_head_dim: usize = 64;
    const index_topk: usize = 8; // topk > n_blocks
    const index_dim = index_n_heads * index_head_dim;

    var q_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(&q_raw, (&[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(head_dim) }).ptr, 3, c.c.MLX_FLOAT32, 0.0, 1.0, .{ .ctx = null }, stream));
    const q = Array.fromHandle(q_raw);
    defer q.deinit();

    var k_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(&k_raw, (&[_]i32{ @intCast(batch), @intCast(n_blocks), @intCast(head_dim) }).ptr, 3, c.c.MLX_FLOAT32, 0.0, 1.0, .{ .ctx = null }, stream));
    const k = Array.fromHandle(k_raw);
    defer k.deinit();

    var wq_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(&wq_raw, (&[_]i32{ @intCast(index_dim), @intCast(head_dim) }).ptr, 2, c.c.MLX_FLOAT32, 0.0, 0.1, .{ .ctx = null }, stream));
    const wq = Array.fromHandle(wq_raw);

    var wk_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(&wk_raw, (&[_]i32{ @intCast(index_dim), @intCast(head_dim) }).ptr, 2, c.c.MLX_FLOAT32, 0.0, 0.1, .{ .ctx = null }, stream));
    const wk = Array.fromHandle(wk_raw);

    var indexer = LightningIndexer{
        .ctx = ctx,
        .wq_index = wq,
        .wk_index = wk,
        .index_n_heads = index_n_heads,
        .index_head_dim = index_head_dim,
        .index_topk = index_topk,
    };
    defer indexer.deinit();

    const result = try indexer.selectTopK(q, k);
    defer result.deinit();

    // When n_blocks <= topk, should return all block indices
    const result_shape = result.shape();
    try std.testing.expectEqual(@as(usize, 3), result_shape.len);
    try std.testing.expectEqual(@as(i32, @intCast(batch)), result_shape[0]);
    try std.testing.expectEqual(@as(i32, @intCast(seq_len)), result_shape[1]);
    try std.testing.expectEqual(@as(i32, @intCast(n_blocks)), result_shape[2]); // n_blocks, not topk
}

test "LightningIndexer.gatherBlocks returns correct shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = ctx.stream.inner;

    const batch: usize = 2;
    const n_blocks: usize = 16;
    const dim: usize = 128;
    const topk: usize = 4;
    const index_dim: usize = 128;

    // Create K: [B, N_blocks, D]
    var k_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(&k_raw, (&[_]i32{ @intCast(batch), @intCast(n_blocks), @intCast(dim) }).ptr, 3, c.c.MLX_FLOAT32, 0.0, 1.0, .{ .ctx = null }, stream));
    const k = Array.fromHandle(k_raw);
    defer k.deinit();

    // Create indices: [B, S=1, topk]
    var idx_data = try allocator.alloc(i32, batch * 1 * topk);
    defer allocator.free(idx_data);
    for (0..batch) |b| {
        for (0..topk) |t| {
            idx_data[b * topk + t] = @intCast(t * 2); // select blocks 0, 2, 4, 6
        }
    }
    const indices = try Array.fromData(allocator, i32, idx_data, &[_]i32{ @intCast(batch), 1, @intCast(topk) });
    defer indices.deinit();

    var wq_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(&wq_raw, (&[_]i32{ @intCast(index_dim), @intCast(dim) }).ptr, 2, c.c.MLX_FLOAT32, 0.0, 0.1, .{ .ctx = null }, stream));
    const wq = Array.fromHandle(wq_raw);

    var wk_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(&wk_raw, (&[_]i32{ @intCast(index_dim), @intCast(dim) }).ptr, 2, c.c.MLX_FLOAT32, 0.0, 0.1, .{ .ctx = null }, stream));
    const wk = Array.fromHandle(wk_raw);

    var indexer = LightningIndexer{
        .ctx = ctx,
        .wq_index = wq,
        .wk_index = wk,
        .index_n_heads = 2,
        .index_head_dim = 64,
        .index_topk = topk,
    };
    defer indexer.deinit();

    const result = try indexer.gatherBlocks(k, indices);
    defer result.deinit();

    const result_shape = result.shape();
    try std.testing.expectEqual(@as(usize, 3), result_shape.len);
    try std.testing.expectEqual(@as(i32, @intCast(batch)), result_shape[0]);
    try std.testing.expectEqual(@as(i32, @intCast(topk)), result_shape[1]);
    try std.testing.expectEqual(@as(i32, @intCast(dim)), result_shape[2]);
}

test "LightningIndexer.quantize4bit preserves tensor shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const stream = ctx.stream.inner;

    const index_dim: usize = 128;

    // Create a tensor with last dim >= 64 (group_size)
    var t_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(&t_raw, (&[_]i32{ 4, 8, @intCast(index_dim) }).ptr, 3, c.c.MLX_FLOAT32, 0.0, 1.0, .{ .ctx = null }, stream));
    const tensor = Array.fromHandle(t_raw);
    defer tensor.deinit();

    var wq_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(&wq_raw, (&[_]i32{ @intCast(index_dim), @intCast(index_dim) }).ptr, 2, c.c.MLX_FLOAT32, 0.0, 0.1, .{ .ctx = null }, stream));
    const wq = Array.fromHandle(wq_raw);

    var wk_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(&wk_raw, (&[_]i32{ @intCast(index_dim), @intCast(index_dim) }).ptr, 2, c.c.MLX_FLOAT32, 0.0, 0.1, .{ .ctx = null }, stream));
    const wk = Array.fromHandle(wk_raw);

    var indexer = LightningIndexer{
        .ctx = ctx,
        .wq_index = wq,
        .wk_index = wk,
        .index_n_heads = 2,
        .index_head_dim = 64,
        .index_topk = 8,
    };
    defer indexer.deinit();

    const result = try indexer.quantize4bit(tensor);
    defer result.deinit();

    const orig_shape = tensor.shape();
    const result_shape = result.shape();
    try std.testing.expectEqual(orig_shape.len, result_shape.len);
    for (0..orig_shape.len) |i| {
        try std.testing.expectEqual(orig_shape[i], result_shape[i]);
    }
}
