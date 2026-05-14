/// DeepSeek-V4-Flash model architecture.
///
/// Features:
/// - MLA (Multi-head Latent Attention) with Q-lora and O-lora
/// - MoE (Mixture of Experts) with hash-based and score-based routing
/// - CSA/HCA (Compressed Sparse Attention / Heavily Compressed Attention)
/// - mHC (Manifold-Constrained Hyper-Connections)
/// - YARN RoPE scaling
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const reduce_mod = @import("mlx").reduce;
const cmp_mod = @import("mlx").comparison;
const math_mod = @import("mlx").math;
const dtype_mod = @import("mlx").dtype;
const nn = @import("mlx").nn;
const kvcache = @import("../kvcache.zig");
const lora_mod = @import("../lora.zig");
const fast_mod = @import("mlx").fast;
const array_arena_mod = @import("mlx").array_arena;
const quantize_mod = @import("mlx").quantize;
const expert_stream = @import("expert_stream.zig");

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
    o_groups: usize = 8,
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

    /// Deep-copy the config, duplicating any heap-allocated slices.
    pub fn clone(self: DSV4Config, allocator: std.mem.Allocator) !DSV4Config {
        var copy = self;
        copy.compress_ratios = try allocator.dupe(usize, self.compress_ratios);
        copy.quantize_default_mode = try allocator.dupe(u8, self.quantize_default_mode);
        return copy;
    }

    /// Release heap-allocated slices owned by this config.
    pub fn deinitClone(self: *const DSV4Config, allocator: std.mem.Allocator) void {
        allocator.free(self.compress_ratios);
        allocator.free(self.quantize_default_mode);
    }
};

/// HyperHead: compress mHC-expanded [B,S,mult,H] back to [B,S,H]
/// using RMSNorm-weighted learnable mixing (matching mlx-lm HyperHead).
pub const HyperHead = struct {
    ctx: EagerContext,
    fn_weight: Array, // [mix, mult*hidden_size] where mix = (2+mult)*mult
    base: Array, // [mix]
    scale: Array, // [3]
    hc_mult: usize,
    norm_eps: f32,

    pub fn deinit(self: *HyperHead) void {
        self.fn_weight.deinit();
        self.base.deinit();
        self.scale.deinit();
    }

    pub fn forward(self: *HyperHead, x: Array, stream: c.c.mlx_stream) !Array {
        _ = stream;
        const shape = x.shape();
        const batch = @as(usize, @intCast(shape[0]));
        const seq_len = @as(usize, @intCast(shape[1]));
        const mult = @as(usize, @intCast(shape[2]));
        const dim = @as(usize, @intCast(shape[3]));

        // flat = reshape(x, [B, S, mult*dim]).astype(f32)
        const flat = try ops.reshape(self.ctx, x, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(mult * dim) });
        defer flat.deinit();
        const flat_f32 = try ops.astype(self.ctx, flat, .float32);
        defer flat_f32.deinit();

        // RMSNorm rsqrt
        const sq = try ops.multiply(self.ctx, flat_f32, flat_f32);
        defer sq.deinit();
        const mean_sq = try reduce_mod.meanAxis(self.ctx, sq, -1, true);
        defer mean_sq.deinit();
        const eps_arr = try ops.scalarF32(self.ctx, self.norm_eps);
        defer eps_arr.deinit();
        const denom = try ops.add(self.ctx, mean_sq, eps_arr);
        defer denom.deinit();
        const rsqrt = try math_mod.rsqrt(self.ctx, denom);
        defer rsqrt.deinit();

        // mixes = (flat @ fn^T) * rsqrt
        const fn_t = try ops.transpose(self.ctx, self.fn_weight);
        defer fn_t.deinit();
        const mixes_raw = try ops.matmul(self.ctx, flat_f32, fn_t);
        defer mixes_raw.deinit();
        const mixes = try ops.multiply(self.ctx, mixes_raw, rsqrt);
        defer mixes.deinit();

        // pre = sigmoid(mixes * scale[0] + base) + eps
        const scale0 = try ops.slice(self.ctx, self.scale, &[_]i32{0}, &[_]i32{1}, &[_]i32{});
        defer scale0.deinit();
        const scale0_s = try shape_mod.squeeze(self.ctx, scale0);
        defer scale0_s.deinit();
        const mixes_scaled = try ops.multiply(self.ctx, mixes, scale0_s);
        defer mixes_scaled.deinit();
        const mixes_biased = try ops.add(self.ctx, mixes_scaled, self.base);
        defer mixes_biased.deinit();
        const sigmoided = try ops.sigmoid(self.ctx, mixes_biased);
        defer sigmoided.deinit();
        const hc_eps_arr = try ops.scalarF32(self.ctx, self.norm_eps);
        defer hc_eps_arr.deinit();
        const pre = try ops.add(self.ctx, sigmoided, hc_eps_arr);
        defer pre.deinit();

        // return (pre[..., None] * x.astype(f32)).sum(axis=2).astype(x.dtype)
        const x_f32 = try ops.astype(self.ctx, x, .float32);
        defer x_f32.deinit();
        const pre_exp = try ops.expandDims(self.ctx, pre, 3);
        defer pre_exp.deinit();
        const weighted = try ops.multiply(self.ctx, pre_exp, x_f32);
        defer weighted.deinit();
        const summed = try reduce_mod.sumAxis(self.ctx, weighted, 2, false);
        defer summed.deinit();
        return ops.astype(self.ctx, summed, x.dtype());
    }
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

    /// Apply RoPE using MLX ops (GPU-accelerated).
    /// Input shape: (..., seq_len, dim) where dim = self.dim
    /// Applies RoPE to the LAST self.dim dimensions. If input's last dim > self.dim,
    /// the first (last_dim - self.dim) dimensions are left unchanged (partial RoPE).
    pub fn apply(self: *DSV4YarnRoPE, input: Array, start_pos: usize, stream: c.c.mlx_stream, inverse: bool) !Array {
        _ = stream;
        const shape = input.shape();
        const last_dim = @as(usize, @intCast(shape[shape.len - 1]));
        const seq_len = @as(usize, @intCast(shape[shape.len - 2]));
        const dim = self.dim;
        const half_dim = dim / 2;
        const nope_dim = if (last_dim > dim) last_dim - dim else 0;

        // Slice cos/sin cache for the relevant positions: [seq_len, half_dim]
        const cos_slice = try ops.slice(self.ctx, self.cos_cache, &[_]i32{ @intCast(start_pos), 0 }, &[_]i32{ @intCast(start_pos + seq_len), @intCast(half_dim) }, &[_]i32{});
        defer cos_slice.deinit();
        const sin_slice = try ops.slice(self.ctx, self.sin_cache, &[_]i32{ @intCast(start_pos), 0 }, &[_]i32{ @intCast(start_pos + seq_len), @intCast(half_dim) }, &[_]i32{});
        defer sin_slice.deinit();

        // Cast cos/sin to input dtype
        const cos_typed = try ops.astype(self.ctx, cos_slice, input.dtype());
        defer cos_typed.deinit();
        var sin_typed = try ops.astype(self.ctx, sin_slice, input.dtype());
        defer sin_typed.deinit();

        if (inverse) {
            const neg_sin = try ops.negative(self.ctx, sin_typed);
            sin_typed.deinit();
            sin_typed = neg_sin;
        }

        // Extract the PE portion (last `dim` dimensions)
        var pe: Array = undefined;
        if (nope_dim > 0) {
            pe = try ops.slice(self.ctx, input, &[_]i32{ 0, 0, 0, @intCast(nope_dim) }, &[_]i32{ shape[0], shape[1], shape[2], shape[3] }, &[_]i32{});
        } else {
            pe = try ops.copy(self.ctx, input);
        }
        defer pe.deinit();

        // Reshape PE to (..., seq_len, half_dim, 2) for rotation
        const pe_shape = pe.shape();
        var new_shape_buf: [8]i32 = undefined;
        const ndim = pe_shape.len;
        for (0..ndim - 1) |i| new_shape_buf[i] = pe_shape[i];
        new_shape_buf[ndim - 1] = @intCast(half_dim);
        new_shape_buf[ndim] = 2;
        const pe_pairs = try ops.reshape(self.ctx, pe, new_shape_buf[0 .. ndim + 1]);
        defer pe_pairs.deinit();

        // Split into x0, x1
        const x0 = try ops.slice(self.ctx, pe_pairs, &[_]i32{ 0, 0, 0, 0, 0 }, &[_]i32{ pe_shape[0], pe_shape[1], @intCast(seq_len), @intCast(half_dim), 1 }, &[_]i32{});
        defer x0.deinit();
        const x1 = try ops.slice(self.ctx, pe_pairs, &[_]i32{ 0, 0, 0, 0, 1 }, &[_]i32{ pe_shape[0], pe_shape[1], @intCast(seq_len), @intCast(half_dim), 2 }, &[_]i32{});
        defer x1.deinit();
        const x0_sq = try ops.squeeze(self.ctx, x0);
        defer x0_sq.deinit();
        const x1_sq = try ops.squeeze(self.ctx, x1);
        defer x1_sq.deinit();

        // Rotate: out0 = x0*cos - x1*sin, out1 = x0*sin + x1*cos
        const x0_cos = try ops.multiply(self.ctx, x0_sq, cos_typed);
        defer x0_cos.deinit();
        const x1_sin = try ops.multiply(self.ctx, x1_sq, sin_typed);
        defer x1_sin.deinit();
        const out0 = try ops.subtract(self.ctx, x0_cos, x1_sin);
        defer out0.deinit();

        const x0_sin = try ops.multiply(self.ctx, x0_sq, sin_typed);
        defer x0_sin.deinit();
        const x1_cos = try ops.multiply(self.ctx, x1_sq, cos_typed);
        defer x1_cos.deinit();
        const out1 = try ops.add(self.ctx, x0_sin, x1_cos);
        defer out1.deinit();

        // Stack and reshape back: [out0, out1] → (..., seq_len, half_dim, 2) → (..., seq_len, dim)
        const out0_exp = try ops.expandDims(self.ctx, out0, -1);
        defer out0_exp.deinit();
        const out1_exp = try ops.expandDims(self.ctx, out1, -1);
        defer out1_exp.deinit();
        const stacked = try shape_mod.concatenateAxis(self.ctx, &[_]Array{ out0_exp, out1_exp }, -1);
        defer stacked.deinit();

        // Reshape back to original PE shape
        const pe_out = try ops.reshape(self.ctx, stacked, pe_shape);
        defer pe_out.deinit();

        // Concat nope + pe_out if partial RoPE
        if (nope_dim > 0) {
            const nope = try ops.slice(self.ctx, input, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ shape[0], shape[1], shape[2], @intCast(nope_dim) }, &[_]i32{});
            defer nope.deinit();
            return shape_mod.concatenateAxis(self.ctx, &[_]Array{ nope, pe_out }, -1);
        }
        return ops.copy(self.ctx, pe_out);
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
/// Weights may be null when the expert is not loaded (Smelt mode).
/// Supports quantized weights via quantizedMatmul when scales are present.
pub const DSV4Expert = struct {
    ctx: EagerContext,
    w1: ?Array, // gate projection
    w2: ?Array, // down projection
    w3: ?Array, // up projection
    // Optional quantized scales/biases for quantizedMatmul
    w1_scales: ?Array = null,
    w1_biases: ?Array = null,
    w2_scales: ?Array = null,
    w2_biases: ?Array = null,
    w3_scales: ?Array = null,
    w3_biases: ?Array = null,
    quant_group_size: i32 = 64,
    quant_bits: u8 = 4,
    quant_mode: quantize_mod.QuantMode = .affine,
    swiglu_limit: f32,

    pub fn deinit(self: *DSV4Expert) void {
        if (self.w1) |w| w.deinit();
        if (self.w2) |w| w.deinit();
        if (self.w3) |w| w.deinit();
        if (self.w1_scales) |s| s.deinit();
        if (self.w1_biases) |b| b.deinit();
        if (self.w2_scales) |s| s.deinit();
        if (self.w2_biases) |b| b.deinit();
        if (self.w3_scales) |s| s.deinit();
        if (self.w3_biases) |b| b.deinit();
    }

    /// Perform matmul: x @ w^T, using quantizedMatmul when scales are present.
    fn expertMatmul(self: *DSV4Expert, x: Array, w: Array, scales: ?Array, biases: ?Array) !Array {
        if (scales) |s| {
            const qw = quantize_mod.QuantizedWeight{
                .data = w,
                .scales = s,
                .biases = biases orelse Array.fromHandle(c.c.mlx_array_new()),
                .config = .{ .group_size = self.quant_group_size, .bits = self.quant_bits, .mode = self.quant_mode },
                .original_shape = &[_]i32{},
            };
            return quantize_mod.quantizedMatmul(self.ctx, x, qw, true);
        } else {
            const w_t = try ops.transpose(self.ctx, w);
            defer w_t.deinit();
            return ops.matmul(self.ctx, x, w_t);
        }
    }

    pub fn forward(self: *DSV4Expert, x: Array, stream: c.c.mlx_stream) !Array {
        _ = stream;
        const w1 = self.w1 orelse return error.UnloadedExpert;
        const w2 = self.w2 orelse return error.UnloadedExpert;
        const w3 = self.w3 orelse return error.UnloadedExpert;

        var gate_proj = try self.expertMatmul(x, w1, self.w1_scales, self.w1_biases);
        defer gate_proj.deinit();
        var up_proj = try self.expertMatmul(x, w3, self.w3_scales, self.w3_biases);
        defer up_proj.deinit();

        // Apply limited SwiGLU: clip gate and up to prevent numerical explosion
        if (self.swiglu_limit > 0) {
            const limit_pos = try ops.scalarF32(self.ctx, self.swiglu_limit);
            defer limit_pos.deinit();
            gate_proj = try math_mod.minimum(self.ctx, gate_proj, limit_pos);
            const limit_neg = try ops.scalarF32(self.ctx, -self.swiglu_limit);
            defer limit_neg.deinit();
            up_proj = try math_mod.maximum(self.ctx, up_proj, limit_neg);
            up_proj = try math_mod.minimum(self.ctx, up_proj, limit_pos);
        }

        const silu_gate = try ops.multiply(self.ctx, gate_proj, try ops.sigmoid(self.ctx, gate_proj));
        defer silu_gate.deinit();
        const hidden = try ops.multiply(self.ctx, silu_gate, up_proj);
        defer hidden.deinit();

        return self.expertMatmul(hidden, w2, self.w2_scales, self.w2_biases);
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
    /// Smelt mode: optional mask of resident experts. If present, non-resident
    /// experts receive a large negative routing bias to prevent selection.
    /// Owned by the gate; freed in deinit.
    smelt_mask: ?[]bool = null,
    allocator: ?std.mem.Allocator = null,

    pub fn deinit(self: *DSV4Gate) void {
        self.weight.deinit();
        if (self.bias) |b| b.deinit();
        if (self.tid2eid) |t| t.deinit();
        if (self.smelt_mask) |mask| {
            if (self.allocator) |alloc| alloc.free(mask);
        }
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

        // Promote to float32 for numerical stability (matching Python: logits.astype(mx.float32))
        const logits_f32 = try ops.astype(self.ctx, logits, .float32);
        defer logits_f32.deinit();

        // Apply scoring function
        var scores: Array = undefined;
        switch (self.scoring_func) {
            .softmax => {
                scores = try ops.softmax(self.ctx, logits_f32, &[_]i32{-1});
            },
            .sigmoid => {
                scores = try ops.sigmoid(self.ctx, logits_f32);
            },
            .sqrtsoftplus => {
                const exp_logits = try ops.exp(self.ctx, logits_f32);
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

        // Apply Smelt mode routing bias: non-resident experts get -inf score
        if (self.smelt_mask) |mask| {
            var bias_data = try self.ctx.allocator.alloc(f32, mask.len);
            defer self.ctx.allocator.free(bias_data);
            for (mask, 0..) |resident, i| {
                bias_data[i] = if (resident) 0.0 else -1e9;
            }
            const smelt_bias = try Array.fromData(self.ctx.allocator, f32, bias_data, &[_]i32{@intCast(mask.len)});
            defer smelt_bias.deinit();
            const bias_exp = try ops.expandDims(self.ctx, smelt_bias, 0);
            defer bias_exp.deinit();
            const biased = try ops.add(self.ctx, scores_for_choice, bias_exp);
            if (scores_for_choice.inner.ctx != scores.inner.ctx) scores_for_choice.deinit();
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

/// Fused SwitchGLU for MoE expert dispatch using gather_mm.
/// Stores expert weights in fused [n_experts, out, in] format and dispatches
/// via gather_mm (or gather_qmm for quantized) — matching mlx-lm's SwitchGLU.
pub const DSV4SwitchGLU = struct {
    ctx: EagerContext,
    // Fused expert weights: [n_experts, intermediate, hidden] or quantized equivalents
    gate_proj: Array, // w1: [n_experts, moe_intermediate_size, hidden_size]
    up_proj: Array, // w3: [n_experts, moe_intermediate_size, hidden_size]
    down_proj: Array, // w2: [n_experts, hidden_size, moe_intermediate_size]
    // Optional quantized scales/biases for gatherQmm
    gate_proj_scales: ?Array,
    gate_proj_biases: ?Array,
    up_proj_scales: ?Array,
    up_proj_biases: ?Array,
    down_proj_scales: ?Array,
    down_proj_biases: ?Array,
    is_quantized: bool,
    quant_group_size: i32,
    quant_bits: u8,
    quant_mode: []const u8,
    swiglu_limit: f32,
    sort_threshold: usize,

    pub fn deinit(self: *DSV4SwitchGLU) void {
        self.gate_proj.deinit();
        self.up_proj.deinit();
        self.down_proj.deinit();
        if (self.gate_proj_scales) |s| s.deinit();
        if (self.gate_proj_biases) |b| b.deinit();
        if (self.up_proj_scales) |s| s.deinit();
        if (self.up_proj_biases) |b| b.deinit();
        if (self.down_proj_scales) |s| s.deinit();
        if (self.down_proj_biases) |b| b.deinit();
    }

    /// Dispatch tokens to experts via gather_mm, matching mlx-lm DeepseekV4SwitchGLU.
    /// x: [N, hidden_size], indices: [N, topk] (expert IDs), scores: [N, topk] (weights)
    pub fn forward(self: *DSV4SwitchGLU, x: Array, indices: Array, scores: Array, stream: c.c.mlx_stream) !Array {
        _ = stream;
        const out_shape_arr = x.shape();
        const route_shape = indices.shape();
        const n_tokens = @as(usize, @intCast(out_shape_arr[0]));
        const hidden_dim = @as(usize, @intCast(out_shape_arr[1]));
        const topk = @as(usize, @intCast(route_shape[route_shape.len - 1]));

        // Expand x: [N, D] → [N, 1, 1, D] for gather_mm compatibility
        const x_4d = try ops.expandDims(self.ctx, x, -2);
        defer x_4d.deinit();
        const x_exp = try ops.expandDims(self.ctx, x_4d, -2);
        defer x_exp.deinit();

        // Sort tokens by expert ID for memory locality (when enough tokens)
        const total_indices = indices.size();
        const do_sort = total_indices >= self.sort_threshold;

        // Prepare transposed weights (swapaxes -1, -2 on [n_experts, out, in] → [n_experts, in, out])
        const gate_proj_t = try ops.transposeAxes(self.ctx, self.gate_proj, &[_]i32{ 0, 2, 1 });
        defer gate_proj_t.deinit();
        const up_proj_t = try ops.transposeAxes(self.ctx, self.up_proj, &[_]i32{ 0, 2, 1 });
        defer up_proj_t.deinit();
        const down_proj_t = try ops.transposeAxes(self.ctx, self.down_proj, &[_]i32{ 0, 2, 1 });
        defer down_proj_t.deinit();

        if (do_sort) {
            // Flatten indices, argsort, reorder
            const flat_indices = try ops.reshape(self.ctx, indices, &[_]i32{@intCast(total_indices)});
            defer flat_indices.deinit();

            var order_handle = c.c.mlx_array_new();
            try c.check(c.c.mlx_argsort_axis(&order_handle, flat_indices.inner, 0, self.ctx.stream.inner));
            const order = Array.fromHandle(order_handle);
            defer order.deinit();

            var inv_handle = c.c.mlx_array_new();
            try c.check(c.c.mlx_argsort_axis(&inv_handle, order.inner, 0, self.ctx.stream.inner));
            const inv_order = Array.fromHandle(inv_handle);
            defer inv_order.deinit();

            // Reorder x: x_exp.flatten(0, -3)[order // topk]
            const x_flat = try shape_mod.flatten(self.ctx, x_exp, 0, @as(i32, @intCast(x_exp.ndim())) - 3);
            defer x_flat.deinit();
            const topk_scalar = try ops.scalarI32(self.ctx, @intCast(topk));
            defer topk_scalar.deinit();
            const order_i32 = try ops.astype(self.ctx, order, .int32);
            defer order_i32.deinit();
            const token_idx_f = try ops.divide(self.ctx, order_i32, topk_scalar);
            defer token_idx_f.deinit();
            const token_idx = try ops.astype(self.ctx, token_idx_f, .int32);
            defer token_idx.deinit();
            const sx = try shape_mod.takeAxis(self.ctx, x_flat, token_idx, 0);
            defer sx.deinit();

            // Sorted indices and scores
            const si_raw = try shape_mod.take(self.ctx, flat_indices, order);
            defer si_raw.deinit();
            const si = try ops.astype(self.ctx, si_raw, .uint32);
            defer si.deinit();
            const flat_scores = try ops.reshape(self.ctx, scores, &[_]i32{@intCast(total_indices)});
            defer flat_scores.deinit();
            const ss = try shape_mod.take(self.ctx, flat_scores, order);
            defer ss.deinit();

            // Expert dispatch
            const x_up = try self.dispatchGatherMm(sx, self.up_proj, up_proj_t, self.up_proj_scales, self.up_proj_biases, si, true);
            defer x_up.deinit();
            const x_gate = try self.dispatchGatherMm(sx, self.gate_proj, gate_proj_t, self.gate_proj_scales, self.gate_proj_biases, si, true);
            defer x_gate.deinit();

            // LimitedSwiGLU
            const activated = try self.limitedSwiGLU(x_gate, x_up);
            defer activated.deinit();

            // Multiply scores before down_proj
            const ss_exp = try ops.expandDims(self.ctx, ss, -1);
            defer ss_exp.deinit();
            const ss_exp2 = try ops.expandDims(self.ctx, ss_exp, -1);
            defer ss_exp2.deinit();
            const weighted = try ops.multiply(self.ctx, activated, ss_exp2);
            defer weighted.deinit();

            // Down projection
            const x_down = try self.dispatchGatherMm(weighted, self.down_proj, down_proj_t, self.down_proj_scales, self.down_proj_biases, si, true);
            defer x_down.deinit();

            // Unsort
            const unsorted = try shape_mod.takeAxis(self.ctx, x_down, inv_order, 0);
            defer unsorted.deinit();
            const unflat = try shape_mod.unflatten(self.ctx, unsorted, 0, &[_]i32{ @intCast(n_tokens), @intCast(topk) });
            defer unflat.deinit();

            // Squeeze and sum over topk
            const squeezed = try ops.squeeze(self.ctx, unflat);
            defer squeezed.deinit();
            const summed = try reduce_mod.sumAxis(self.ctx, squeezed, -2, false);
            defer summed.deinit();
            return ops.reshape(self.ctx, summed, &[_]i32{ @intCast(n_tokens), @intCast(hidden_dim) });
        } else {
            // No sorting — direct dispatch
            const x_up = try self.dispatchGatherMm(x_exp, self.up_proj, up_proj_t, self.up_proj_scales, self.up_proj_biases, indices, false);
            defer x_up.deinit();
            const x_gate = try self.dispatchGatherMm(x_exp, self.gate_proj, gate_proj_t, self.gate_proj_scales, self.gate_proj_biases, indices, false);
            defer x_gate.deinit();

            const activated = try self.limitedSwiGLU(x_gate, x_up);
            defer activated.deinit();

            // Multiply scores before down_proj
            const s_exp = try ops.expandDims(self.ctx, scores, -1);
            defer s_exp.deinit();
            const s_exp2 = try ops.expandDims(self.ctx, s_exp, -1);
            defer s_exp2.deinit();
            const weighted = try ops.multiply(self.ctx, activated, s_exp2);
            defer weighted.deinit();

            const x_down = try self.dispatchGatherMm(weighted, self.down_proj, down_proj_t, self.down_proj_scales, self.down_proj_biases, indices, false);
            defer x_down.deinit();

            // Squeeze and sum over topk
            const squeezed = try ops.squeeze(self.ctx, x_down);
            defer squeezed.deinit();
            const summed = try reduce_mod.sumAxis(self.ctx, squeezed, -2, false);
            defer summed.deinit();
            return ops.reshape(self.ctx, summed, &[_]i32{ @intCast(n_tokens), @intCast(hidden_dim) });
        }
    }

    /// Forward pass WITHOUT score weighting — matches Python mlx-lm SwitchGLU.__call__.
    /// Returns [N, topk, D] (not summed over topk). Caller applies scores and sums.
    /// This is the correct behavior per Python reference:
    ///   y = switch_mlp(x, inds)                    # this function
    ///   y = (y * scores[..., None]).sum(axis=-2)    # caller does this
    pub fn forwardNoScores(self: *DSV4SwitchGLU, x: Array, indices: Array, stream: c.c.mlx_stream) !Array {
        _ = stream;
        const route_shape = indices.shape();
        const topk = @as(usize, @intCast(route_shape[route_shape.len - 1]));

        // Expand x: [N, D] → [N, 1, 1, D]
        const x_4d = try ops.expandDims(self.ctx, x, -2);
        defer x_4d.deinit();
        const x_exp = try ops.expandDims(self.ctx, x_4d, -2);
        defer x_exp.deinit();

        const total_indices = indices.size();
        const do_sort = total_indices >= self.sort_threshold;

        const gate_proj_t = try ops.transposeAxes(self.ctx, self.gate_proj, &[_]i32{ 0, 2, 1 });
        defer gate_proj_t.deinit();
        const up_proj_t = try ops.transposeAxes(self.ctx, self.up_proj, &[_]i32{ 0, 2, 1 });
        defer up_proj_t.deinit();
        const down_proj_t = try ops.transposeAxes(self.ctx, self.down_proj, &[_]i32{ 0, 2, 1 });
        defer down_proj_t.deinit();

        if (do_sort) {
            const flat_indices = try ops.reshape(self.ctx, indices, &[_]i32{@intCast(total_indices)});
            defer flat_indices.deinit();

            var order_handle = c.c.mlx_array_new();
            try c.check(c.c.mlx_argsort_axis(&order_handle, flat_indices.inner, 0, self.ctx.stream.inner));
            const order = Array.fromHandle(order_handle);
            defer order.deinit();

            var inv_handle = c.c.mlx_array_new();
            try c.check(c.c.mlx_argsort_axis(&inv_handle, order.inner, 0, self.ctx.stream.inner));
            const inv_order = Array.fromHandle(inv_handle);
            defer inv_order.deinit();

            const x_flat = try shape_mod.flatten(self.ctx, x_exp, 0, @as(i32, @intCast(x_exp.ndim())) - 3);
            defer x_flat.deinit();
            const topk_scalar = try ops.scalarI32(self.ctx, @intCast(topk));
            defer topk_scalar.deinit();
            const order_i32 = try ops.astype(self.ctx, order, .int32);
            defer order_i32.deinit();
            const token_idx_f = try ops.divide(self.ctx, order_i32, topk_scalar);
            defer token_idx_f.deinit();
            const token_idx = try ops.astype(self.ctx, token_idx_f, .int32);
            defer token_idx.deinit();
            const sx = try shape_mod.takeAxis(self.ctx, x_flat, token_idx, 0);
            defer sx.deinit();

            const si_raw = try shape_mod.take(self.ctx, flat_indices, order);
            defer si_raw.deinit();
            const si = try ops.astype(self.ctx, si_raw, .uint32);
            defer si.deinit();

            const x_up = try self.dispatchGatherMm(sx, self.up_proj, up_proj_t, self.up_proj_scales, self.up_proj_biases, si, true);
            defer x_up.deinit();
            const x_gate = try self.dispatchGatherMm(sx, self.gate_proj, gate_proj_t, self.gate_proj_scales, self.gate_proj_biases, si, true);
            defer x_gate.deinit();

            const activated = try self.limitedSwiGLU(x_gate, x_up);
            defer activated.deinit();

            // Down projection WITHOUT score weighting
            const x_down = try self.dispatchGatherMm(activated, self.down_proj, down_proj_t, self.down_proj_scales, self.down_proj_biases, si, true);
            defer x_down.deinit();

            // Unsort and unflatten to [N, topk, 1, D]
            const unsorted = try shape_mod.takeAxis(self.ctx, x_down, inv_order, 0);
            defer unsorted.deinit();
            const unflat = try shape_mod.unflatten(self.ctx, unsorted, 0, indices.shape());
            defer unflat.deinit();
            // Squeeze only axis -2 (the singleton dim from gather_mm), not all singletons
            // Python: return x.squeeze(-2)
            return shape_mod.squeezeAxes(self.ctx, unflat, &[_]i32{-2});
        } else {
            const x_up = try self.dispatchGatherMm(x_exp, self.up_proj, up_proj_t, self.up_proj_scales, self.up_proj_biases, indices, false);
            defer x_up.deinit();
            const x_gate = try self.dispatchGatherMm(x_exp, self.gate_proj, gate_proj_t, self.gate_proj_scales, self.gate_proj_biases, indices, false);
            defer x_gate.deinit();

            const activated = try self.limitedSwiGLU(x_gate, x_up);
            defer activated.deinit();

            // Down projection WITHOUT score weighting
            const x_down = try self.dispatchGatherMm(activated, self.down_proj, down_proj_t, self.down_proj_scales, self.down_proj_biases, indices, false);
            defer x_down.deinit();

            // Squeeze only axis -2 (the singleton dim from gather_mm)
            // Python: return x.squeeze(-2)
            return shape_mod.squeezeAxes(self.ctx, x_down, &[_]i32{-2});
        }
    }

    fn limitedSwiGLU(self: *DSV4SwitchGLU, gate: Array, up: Array) !Array {
        if (self.swiglu_limit > 0) {
            const limit_pos = try ops.scalarF32(self.ctx, self.swiglu_limit);
            defer limit_pos.deinit();
            const limit_neg = try ops.scalarF32(self.ctx, -self.swiglu_limit);
            defer limit_neg.deinit();
            const gate_clipped = try math_mod.minimum(self.ctx, gate, limit_pos);
            defer gate_clipped.deinit();
            const up_lo = try math_mod.maximum(self.ctx, up, limit_neg);
            defer up_lo.deinit();
            const up_clipped = try math_mod.minimum(self.ctx, up_lo, limit_pos);
            defer up_clipped.deinit();
            const silu_g = try ops.multiply(self.ctx, gate_clipped, try ops.sigmoid(self.ctx, gate_clipped));
            defer silu_g.deinit();
            return ops.multiply(self.ctx, silu_g, up_clipped);
        }
        const silu_g = try ops.multiply(self.ctx, gate, try ops.sigmoid(self.ctx, gate));
        defer silu_g.deinit();
        return ops.multiply(self.ctx, silu_g, up);
    }

    /// Dispatch matmul to either gatherMm (float) or gatherQmm (quantized).
    fn dispatchGatherMm(self: *DSV4SwitchGLU, x: Array, weight: Array, weight_t: Array, scales: ?Array, biases: ?Array, indices_arr: Array, sorted: bool) !Array {
        if (self.is_quantized) {
            const qconfig = quantize_mod.QuantConfig{
                .group_size = self.quant_group_size,
                .bits = self.quant_bits,
                .mode = if (std.mem.eql(u8, self.quant_mode, "mxfp4")) .mxfp4 else .affine,
            };
            return quantize_mod.gatherQmm(self.ctx, x, weight, scales.?, biases, null, indices_arr, true, qconfig, sorted);
        }
        return ops.gatherMm(self.ctx, x, weight_t, null, indices_arr, sorted);
    }
};

/// MoE layer: routed experts + shared expert.
/// Uses fused gather_mm dispatch (matching mlx-lm DeepseekV4MoE).
pub const DSV4MoE = struct {
    ctx: EagerContext,
    gate: DSV4Gate,
    switch_mlp: DSV4SwitchGLU,
    shared_expert: DSV4Expert,
    n_routed_experts: usize,
    n_activated_experts: usize,
    /// When smelt slices fused weights, maps original expert ID → sliced row index.
    /// null when all experts are loaded (no remapping needed).
    expert_remap: ?Array,
    /// When true, switch_mlp experts are loaded. When false, uses stream_provider or shared-expert-only.
    experts_loaded: bool,
    /// Expert streaming provider — loads expert weights from SSD on demand.
    /// When set, forward uses streaming instead of pre-loaded fused weights.
    stream_provider: ?*expert_stream.ExpertStreamProvider = null,
    /// Layer index for streaming (each MoE layer needs to know its index).
    layer_idx: usize = 0,

    pub fn deinit(self: *DSV4MoE) void {
        self.gate.deinit();
        if (self.experts_loaded) self.switch_mlp.deinit();
        self.shared_expert.deinit();
        if (self.expert_remap) |r| r.deinit();
        // stream_provider is owned by the model, not individual MoE layers
    }

    pub fn forward(self: *DSV4MoE, hidden_states: Array, input_ids: Array, stream: c.c.mlx_stream) !Array {
        const shape = hidden_states.shape();
        const batch = @as(usize, @intCast(shape[0]));
        const seq_len = @as(usize, @intCast(shape[1]));
        const dim = @as(usize, @intCast(shape[2]));

        // Flatten to [B*S, dim]
        const flat = try ops.reshape(self.ctx, hidden_states, &[_]i32{ @intCast(batch * seq_len), @intCast(dim) });
        defer flat.deinit();

        // Shared expert always runs
        const shared_out = try self.shared_expert.forward(flat, stream);
        defer shared_out.deinit();

        // Gate routing: returns (scores [N, topk], indices [N, topk])
        const weights_arr, const indices = try self.gate.forward(flat, input_ids, stream);
        defer weights_arr.deinit();
        defer indices.deinit();

        // Expert dispatch — choose path based on available resources
        const y = if (self.stream_provider) |sp| blk: {
            // Streaming path: load expert weights from SSD on demand
            break :blk try sp.streamForward(self.layer_idx, flat, indices, weights_arr);
        } else if (self.experts_loaded) blk: {
            // Pre-loaded path: use fused weights in memory
            const remapped_indices = if (self.expert_remap) |remap| blk2: {
                var res = c.c.mlx_array_new();
                try c.check(c.c.mlx_take(&res, remap.inner, indices.inner, self.ctx.stream.inner));
                break :blk2 Array.fromHandle(res);
            } else indices;
            defer if (self.expert_remap != null) remapped_indices.deinit();
            break :blk try self.switch_mlp.forward(flat, remapped_indices, weights_arr, stream);
        } else blk: {
            // No experts available — return zeros (shared expert only)
            var zeros_raw = c.c.mlx_array_new();
            try c.check(c.c.mlx_zeros_like(&zeros_raw, shared_out.inner, self.ctx.stream.inner));
            break :blk Array.fromHandle(zeros_raw);
        };
        defer y.deinit();

        // Add shared expert
        const final_out = try ops.add(self.ctx, y, shared_out);
        defer final_out.deinit();

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
/// Indexer module for CSA sparse block selection (matching mlx-lm Indexer).
/// Contains its own Compressor for independent pooling, plus Q projection and scoring.
pub const Indexer = struct {
    ctx: EagerContext,
    n_heads: usize,
    head_dim: usize,
    index_topk: usize,
    wq_b: Array, // Linear weight: [n_heads * head_dim, q_lora_rank]
    weights_proj: Array, // Linear weight: [n_heads, hidden_size]
    compressor: Compressor,
    scale: f32,

    pub fn deinit(self: *Indexer) void {
        self.wq_b.deinit();
        self.weights_proj.deinit();
        self.compressor.deinit();
    }

    /// Forward: score compressed blocks and return top-k indices.
    /// x: [B, L, hidden_size], q_residual: [B, L, q_lora_rank]
    /// compress_rope: RoPE for compressor, position_rope: RoPE for Q projection
    /// Returns: top-k indices [B, L, K] or null if no pooled blocks
    pub fn forward(
        self: *Indexer,
        x: Array,
        q_residual: Array,
        compress_rope: *DSV4YarnRoPE,
        position_rope: *DSV4YarnRoPE,
        cache: ?*kvcache.DeepseekV4Cache,
        start_pos: usize,
    ) !?Array {
        const x_shape = x.shape();
        const B = @as(usize, @intCast(x_shape[0]));
        const L = @as(usize, @intCast(x_shape[1]));

        // Run internal compressor
        const pooled = try self.compressor.forward(x, compress_rope, cache, start_pos, "indexer_state");
        defer pooled.deinit();

        if (pooled.shape()[1] == 0) return null;

        const N_pooled = @as(usize, @intCast(pooled.shape()[1]));

        // Q projection: wq_b(q_residual) → [B, L, n_heads, head_dim]
        const wq_t = try ops.transpose(self.ctx, self.wq_b);
        defer wq_t.deinit();
        const q_proj = try ops.matmul(self.ctx, q_residual, wq_t);
        defer q_proj.deinit();
        const q_rs = try ops.reshape(self.ctx, q_proj, &[_]i32{ @intCast(B), @intCast(L), @intCast(self.n_heads), @intCast(self.head_dim) });
        defer q_rs.deinit();
        const q_t = try ops.transposeAxes(self.ctx, q_rs, &[_]i32{ 0, 2, 1, 3 }); // [B, n_heads, L, head_dim]
        defer q_t.deinit();

        // Apply position RoPE to Q
        const q_roped = try position_rope.apply(q_t, start_pos, self.ctx.stream.inner, false);
        defer q_roped.deinit();

        // Scores: q @ pooled^T → [B, n_heads, L, N_pooled]
        const q_f32 = try ops.astype(self.ctx, q_roped, .float32);
        defer q_f32.deinit();
        const pooled_exp = try ops.expandDims(self.ctx, pooled, 1); // [B, 1, N_pooled, head_dim]
        defer pooled_exp.deinit();
        const pooled_t = try ops.transposeAxes(self.ctx, pooled_exp, &[_]i32{ 0, 1, 3, 2 }); // [B, 1, head_dim, N_pooled]
        defer pooled_t.deinit();
        const pooled_f32 = try ops.astype(self.ctx, pooled_t, .float32);
        defer pooled_f32.deinit();
        const scores_raw = try ops.matmul(self.ctx, q_f32, pooled_f32); // [B, n_heads, L, N_pooled]
        defer scores_raw.deinit();

        // max(0, scores) * scale
        const zero = try ops.scalarF32(self.ctx, 0.0);
        defer zero.deinit();
        const scores_relu = try math_mod.maximum(self.ctx, scores_raw, zero);
        defer scores_relu.deinit();
        const scale_arr = try ops.scalarF32(self.ctx, self.scale);
        defer scale_arr.deinit();
        const scores_scaled = try ops.multiply(self.ctx, scores_relu, scale_arr);
        defer scores_scaled.deinit();

        // weights_proj(x) weighting: [B, L, n_heads] → [B, n_heads, L] → [B, n_heads, L, 1]
        const wp_t = try ops.transpose(self.ctx, self.weights_proj);
        defer wp_t.deinit();
        const w_proj = try ops.matmul(self.ctx, x, wp_t); // [B, L, n_heads]
        defer w_proj.deinit();
        const w_f32 = try ops.astype(self.ctx, w_proj, .float32);
        defer w_f32.deinit();
        const n_heads_scale = try ops.scalarF32(self.ctx, 1.0 / @sqrt(@as(f32, @floatFromInt(self.n_heads))));
        defer n_heads_scale.deinit();
        const w_scaled = try ops.multiply(self.ctx, w_f32, n_heads_scale);
        defer w_scaled.deinit();
        // Transpose to [B, n_heads, L] then expand to [B, n_heads, L, 1]
        const w_t = try ops.transposeAxes(self.ctx, w_scaled, &[_]i32{ 0, 2, 1 }); // [B, n_heads, L]
        defer w_t.deinit();
        const w_exp = try ops.expandDims(self.ctx, w_t, -1); // [B, n_heads, L, 1]
        defer w_exp.deinit();

        // Weighted scores: (scores * weights[..., None]).sum(axis=1) → [B, L, N_pooled]
        const weighted_scores = try ops.multiply(self.ctx, scores_scaled, w_exp);
        defer weighted_scores.deinit();
        const summed = try reduce_mod.sumAxis(self.ctx, weighted_scores, 1, false); // [B, L, N_pooled]
        defer summed.deinit();

        // Top-k selection: argpartition(-scores, kth=k-1)[:, :, :k]
        const k = @min(self.index_topk, N_pooled);
        const neg_scores = try ops.negative(self.ctx, summed);
        defer neg_scores.deinit();

        // Use argsort + slice for top-k (argpartition not available in mlx-c)
        var sorted_handle = c.c.mlx_array_new();
        try c.check(c.c.mlx_argsort_axis(&sorted_handle, neg_scores.inner, -1, self.ctx.stream.inner));
        const sorted_indices = Array.fromHandle(sorted_handle);
        defer sorted_indices.deinit();

        // Take first k indices
        const topk_indices = try ops.slice(self.ctx, sorted_indices, &[_]i32{ 0, 0, 0 }, &[_]i32{ @intCast(B), @intCast(L), @intCast(k) }, &[_]i32{});

        return topk_indices;
    }
};

/// FP4 Lightning Indexer for CSA sparse block selection (LEGACY — replaced by Indexer).
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
            @intCast(batch),              @intCast(seq_len),
            @intCast(self.index_n_heads), @intCast(self.index_head_dim),
        });
        defer q_mh.deinit();
        const q_mh_t = try ops.transposeAxes(ctx, q_mh, &[_]i32{ 0, 2, 1, 3 });
        defer q_mh_t.deinit();

        const k_mh = try ops.reshape(ctx, k_fp4, &[_]i32{
            @intCast(batch),              @intCast(n_blocks),
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

        return topk_indices;
    }

    /// Quantize a tensor to FP4 (E2M1) and immediately dequantize back to float32.
    /// NOTE: This was originally a simulate path for FP4, but MLX qqmm does not support
    /// mxfp4 for two-dynamic-activation matmul. We now skip quantization and use
    /// bfloat16/float32 direct matmul like mlx-lm.
    fn quantize4bit(self: *LightningIndexer, tensor: Array) !Array {
        return ops.copy(self.ctx, tensor);
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

        return result;
    }
};

/// Apply score mask: if mask is bool, use where; if float, add.
fn applyScoreMask(ctx: EagerContext, scores_in: Array, mask_opt: ?Array) !Array {
    const mask_arr = mask_opt orelse return ops.copy(ctx, scores_in);
    if (mask_arr.dtype() == .bool_) {
        const neg_inf = try ops.scalarF32(ctx, -std.math.inf(f32));
        defer neg_inf.deinit();
        return ops.where(ctx, mask_arr, scores_in, neg_inf);
    }
    const mask_typed = try ops.astype(ctx, mask_arr, scores_in.dtype());
    defer mask_typed.deinit();
    return ops.add(ctx, scores_in, mask_typed);
}

/// Sparse pooled attention: separate local and pooled score computation with concat softmax.
/// Matches mlx-lm `_sparse_pooled_attention` (deepseek_v4.py:295-333).
///
/// q: [B, H, L, D], local_kv: [B, 1, S_local, D], pooled: [B, N_pooled, D],
/// topk: [B, L, K], local_mask: optional, pooled_mask: optional,
/// scale: attention scale, sinks: optional [H] attention sink logits
fn sparsePooledAttention(
    ctx: EagerContext,
    q: Array,
    local_kv: Array,
    pooled: Array,
    topk: Array,
    local_mask: ?Array,
    pooled_mask: ?Array,
    scale: f32,
    sinks: ?Array,
) !Array {
    const q_shape = q.shape();
    const B = q_shape[0];
    const H = q_shape[1];
    const L = q_shape[2];
    const D = q_shape[3];
    const pooled_shape = pooled.shape();
    const N_pooled = pooled_shape[1];

    // Gather pooled blocks using topk indices
    // idx: [B, 1, L, K, 1]
    const topk_exp = try ops.expandDims(ctx, topk, 1); // [B, 1, L, K]
    defer topk_exp.deinit();
    const idx = try ops.expandDims(ctx, topk_exp, -1); // [B, 1, L, K, 1]
    defer idx.deinit();

    // pooled_broadcast: [B, 1, L, N_pooled, D]
    const pooled_exp = try ops.expandDims(ctx, pooled, 1); // [B, 1, N_pooled, D]
    defer pooled_exp.deinit();
    const pooled_exp2 = try ops.expandDims(ctx, pooled_exp, 2); // [B, 1, 1, N_pooled, D]
    defer pooled_exp2.deinit();
    const pooled_bc = try ops.broadcastTo(ctx, pooled_exp2, &[_]i32{ B, 1, L, N_pooled, D });
    defer pooled_bc.deinit();

    // idx_broadcast: [B, 1, L, K, D]
    const topk_shape = topk.shape();
    const K = topk_shape[topk_shape.len - 1];
    const idx_bc = try ops.broadcastTo(ctx, idx, &[_]i32{ B, 1, L, K, D });
    defer idx_bc.deinit();

    // Gather: take_along_axis on axis 3
    var gathered_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_take_along_axis(&gathered_raw, pooled_bc.inner, idx_bc.inner, 3, ctx.stream.inner));
    const gathered_pooled = Array.fromHandle(gathered_raw); // [B, 1, L, K, D]
    defer gathered_pooled.deinit();

    // q_scaled = q * scale
    const scale_arr = try ops.scalarF32(ctx, scale);
    defer scale_arr.deinit();
    const q_scaled = try ops.multiply(ctx, q, scale_arr);
    defer q_scaled.deinit();

    // local_scores = q_scaled @ local_kv^T: [B, H, L, S_local]
    const local_kv_t = try ops.transposeAxes(ctx, local_kv, &[_]i32{ 0, 1, 3, 2 });
    defer local_kv_t.deinit();
    const local_scores_raw = try ops.matmul(ctx, q_scaled, local_kv_t);
    defer local_scores_raw.deinit();
    const local_scores = try applyScoreMask(ctx, local_scores_raw, local_mask);
    defer local_scores.deinit();

    // pooled_scores = (q_scaled[:,:,:,None] * gathered_pooled).sum(axis=-1): [B, H, L, K]
    const q_exp = try ops.expandDims(ctx, q_scaled, 3); // [B, H, L, 1, D]
    defer q_exp.deinit();
    // Broadcast gathered_pooled from [B, 1, L, K, D] to [B, H, L, K, D]
    const gp_bc = try ops.broadcastTo(ctx, gathered_pooled, &[_]i32{ B, H, L, K, D });
    defer gp_bc.deinit();
    const qp_prod = try ops.multiply(ctx, q_exp, gp_bc);
    defer qp_prod.deinit();
    const pooled_scores_raw = try reduce_mod.sumAxis(ctx, qp_prod, -1, false);
    defer pooled_scores_raw.deinit();
    const pooled_scores = try applyScoreMask(ctx, pooled_scores_raw, pooled_mask);
    defer pooled_scores.deinit();

    // Concatenate scores: [local_scores, pooled_scores] along last axis
    var all_scores: Array = undefined;
    var sink_offset: usize = 0;
    if (sinks) |sink_logits| {
        sink_offset = 1;
        // sink_scores: [B, H, L, 1]
        const sink_rs = try ops.reshape(ctx, sink_logits, &[_]i32{ 1, H, 1, 1 });
        defer sink_rs.deinit();
        const sink_bc = try ops.broadcastTo(ctx, sink_rs, &[_]i32{ B, H, L, 1 });
        defer sink_bc.deinit();
        const sink_typed = try ops.astype(ctx, sink_bc, local_scores.dtype());
        defer sink_typed.deinit();
        all_scores = try shape_mod.concatenateAxis(ctx, &[_]Array{ sink_typed, local_scores, pooled_scores }, -1);
    } else {
        all_scores = try shape_mod.concatenateAxis(ctx, &[_]Array{ local_scores, pooled_scores }, -1);
    }
    defer all_scores.deinit();

    // Softmax
    const weights_arr = try ops.softmax(ctx, all_scores, &[_]i32{-1});
    defer weights_arr.deinit();

    // Split weights back
    const local_len = local_kv.shape()[2];
    const sink_i32 = @as(i32, @intCast(sink_offset));
    const local_end = sink_i32 + local_len;
    const scores_last = all_scores.shape()[all_scores.ndim() - 1];

    // local_weights: weights[..., sink_offset : sink_offset + local_len]
    const local_w = try ops.slice(ctx, weights_arr, &[_]i32{ 0, 0, 0, sink_i32 }, &[_]i32{ B, H, L, local_end }, &[_]i32{});
    defer local_w.deinit();
    // pooled_weights: weights[..., sink_offset + local_len :]
    const pooled_w = try ops.slice(ctx, weights_arr, &[_]i32{ 0, 0, 0, local_end }, &[_]i32{ B, H, L, scores_last }, &[_]i32{});
    defer pooled_w.deinit();

    // out = local_weights @ local_kv
    const out_local = try ops.matmul(ctx, local_w, local_kv);
    defer out_local.deinit();

    // out += (pooled_weights[..., None] * gathered_pooled).sum(axis=-2)
    const pw_exp = try ops.expandDims(ctx, pooled_w, -1); // [B, H, L, K, 1]
    defer pw_exp.deinit();
    const pw_pooled = try ops.multiply(ctx, pw_exp, gp_bc);
    defer pw_pooled.deinit();
    const pooled_sum = try reduce_mod.sumAxis(ctx, pw_pooled, -2, false); // [B, H, L, D]
    defer pooled_sum.deinit();

    const out = try ops.add(ctx, out_local, pooled_sum);
    defer out.deinit();
    return ops.astype(ctx, out, q.dtype());
}

/// MLA (Multi-head Latent Attention).
pub const DSV4Attention = struct {
    ctx: EagerContext,
    config: DSV4Config,
    layer_idx: usize,

    // Q projection with lora (may be quantized)
    wq_a: Array,
    wq_a_scales: ?Array,
    wq_a_biases: ?Array,
    wq_b: Array,
    wq_b_scales: ?Array,
    wq_b_biases: ?Array,
    q_norm: nn.RMSNorm,

    // KV projection (may be quantized)
    wkv: Array,
    wkv_scales: ?Array,
    wkv_biases: ?Array,
    kv_norm: nn.RMSNorm,
    kv_b: ?Array,

    // O projection with lora (may be quantized)
    wo_a: Array,
    wo_a_scales: ?Array,
    wo_a_biases: ?Array,
    wo_b: Array,
    wo_b_scales: ?Array,
    wo_b_biases: ?Array,

    // Quantization config for attention weights
    attn_quant_group_size: i32,
    attn_quant_bits: u8,
    attn_quant_mode: quantize_mod.QuantMode = .affine,

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
        if (self.wq_a_scales) |s| s.deinit();
        if (self.wq_a_biases) |b| b.deinit();
        self.wq_b.deinit();
        if (self.wq_b_scales) |s| s.deinit();
        if (self.wq_b_biases) |b| b.deinit();
        self.wkv.deinit();
        if (self.wkv_scales) |s| s.deinit();
        if (self.wkv_biases) |b| b.deinit();
        if (self.kv_b) |kb| kb.deinit();
        self.wo_a.deinit();
        if (self.wo_a_scales) |s| s.deinit();
        if (self.wo_a_biases) |b| b.deinit();
        self.wo_b.deinit();
        if (self.wo_b_scales) |s| s.deinit();
        if (self.wo_b_biases) |b| b.deinit();
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
        const shape = hidden_states.shape();
        const batch = @as(usize, @intCast(shape[0]));
        const seq_len = @as(usize, @intCast(shape[1]));
        const num_heads = self.config.num_attention_heads;
        const head_dim = self.config.head_dim;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // === Q path ===
        // Use quantizedMatmul if weights are quantized, else plain matmul
        const q_a = if (self.wq_a_scales != null) blk: {
            const qw = quantize_mod.QuantizedWeight{ .data = self.wq_a, .scales = self.wq_a_scales.?, .biases = self.wq_a_biases orelse Array.fromHandle(c.c.mlx_array_new()), .config = .{ .group_size = self.attn_quant_group_size, .bits = self.attn_quant_bits, .mode = self.attn_quant_mode }, .original_shape = &[_]i32{} };
            break :blk try quantize_mod.quantizedMatmul(self.ctx, hidden_states, qw, true);
        } else blk: {
            const wq_a_t = try ops.transpose(self.ctx, self.wq_a);
            defer wq_a_t.deinit();
            break :blk try ops.matmul(self.ctx, hidden_states, wq_a_t);
        };
        defer q_a.deinit();
        const q_residual = try self.q_norm.forward(q_a);
        defer q_residual.deinit();
        const q_proj = if (self.wq_b_scales != null) blk: {
            const qw = quantize_mod.QuantizedWeight{ .data = self.wq_b, .scales = self.wq_b_scales.?, .biases = self.wq_b_biases orelse Array.fromHandle(c.c.mlx_array_new()), .config = .{ .group_size = self.attn_quant_group_size, .bits = self.attn_quant_bits, .mode = self.attn_quant_mode }, .original_shape = &[_]i32{} };
            break :blk try quantize_mod.quantizedMatmul(self.ctx, q_residual, qw, true);
        } else blk: {
            const wq_b_t = try ops.transpose(self.ctx, self.wq_b);
            defer wq_b_t.deinit();
            break :blk try ops.matmul(self.ctx, q_residual, wq_b_t);
        };
        defer q_proj.deinit();
        const q_rs = try ops.reshape(self.ctx, q_proj, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_heads), @intCast(head_dim) });
        defer q_rs.deinit();
        const q_normed = try applyRMSNorm(self.ctx, q_rs, self.config.rms_norm_eps);
        defer q_normed.deinit();
        const q_t = try ops.transposeAxes(self.ctx, q_normed, &[_]i32{ 0, 2, 1, 3 });
        defer q_t.deinit();
        const q = try self.rope.apply(q_t, start_pos, stream, false);
        defer q.deinit();

        // === KV path ===
        const kv_raw = if (self.wkv_scales != null) blk: {
            const qw = quantize_mod.QuantizedWeight{ .data = self.wkv, .scales = self.wkv_scales.?, .biases = self.wkv_biases orelse Array.fromHandle(c.c.mlx_array_new()), .config = .{ .group_size = self.attn_quant_group_size, .bits = self.attn_quant_bits, .mode = self.attn_quant_mode }, .original_shape = &[_]i32{} };
            break :blk try quantize_mod.quantizedMatmul(self.ctx, hidden_states, qw, true);
        } else blk: {
            const wkv_t = try ops.transpose(self.ctx, self.wkv);
            defer wkv_t.deinit();
            break :blk try ops.matmul(self.ctx, hidden_states, wkv_t);
        };
        defer kv_raw.deinit();
        const kv_normed = try self.kv_norm.forward(kv_raw);
        defer kv_normed.deinit();
        const kv_4d = try ops.reshape(self.ctx, kv_normed, &[_]i32{ @intCast(batch), 1, @intCast(seq_len), @intCast(head_dim) });
        defer kv_4d.deinit();
        const kv_roped = try self.rope.apply(kv_4d, start_pos, stream, false);
        defer kv_roped.deinit();

        // Update local cache
        var full_kv = kv_roped;
        var kv_slice: ?kvcache.KVSlice = null;
        if (cache) |cs| {
            kv_slice = try cs.updateAndFetch(kv_roped, kv_roped, stream);
            full_kv = kv_slice.?.keys;
        }
        defer if (kv_slice) |sl| {
            sl.keys.deinit();
            sl.values.deinit();
        };

        // === Attention (simplified: concat pooled to local KV for dense SDPA) ===
        var attn_out: Array = undefined;
        if (self.compress_ratio > 0 and self.compress_gate_weight != null) {
            const kv_3d = try ops.reshape(self.ctx, kv_roped, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(head_dim) });
            defer kv_3d.deinit();
            const pooled = try compressKV(self.ctx, kv_3d, self.compress_ratio, self.config.sliding_window, self.compress_gate_weight, self.compress_pos_bias, stream);
            defer pooled.deinit();
            if (pooled.shape()[1] > 0) {
                const pooled_4d = try ops.expandDims(self.ctx, pooled, 1);
                defer pooled_4d.deinit();
                const combined = try shape_mod.concatenateAxis(self.ctx, &[_]Array{ full_kv, pooled_4d }, 2);
                defer combined.deinit();
                // When explicit mask is provided, don't use "causal" string (they conflict in MLX)
                const mm: []const u8 = if (mask != null) "" else if (start_pos == 0 and seq_len > 1) "causal" else "";
                attn_out = try fast_mod.scaledDotProductAttention(self.ctx, q, combined, combined, scale, mm, mask, self.sink_logits);
            } else {
                const mm: []const u8 = if (mask != null) "" else if (start_pos == 0 and seq_len > 1) "causal" else "";
                attn_out = try fast_mod.scaledDotProductAttention(self.ctx, q, full_kv, full_kv, scale, mm, mask, self.sink_logits);
            }
        } else {
            const mm: []const u8 = if (mask != null) "" else if (start_pos == 0 and seq_len > 1) "causal" else "";
            attn_out = try fast_mod.scaledDotProductAttention(self.ctx, q, full_kv, full_kv, scale, mm, mask, self.sink_logits);
        }
        defer attn_out.deinit();

        // === Output: inverse partial RoPE → grouped projection → wo_b ===
        const out_deroped = try self.rope.apply(attn_out, start_pos, stream, true);
        defer out_deroped.deinit();

        const out = if (self.wo_a.ndim() == 3) blk: {
            // Grouped LoRA path: wo_a is [n_groups, o_lora_rank, group_feat]
            const wo_a_shape = self.wo_a.shape();
            const n_groups = @as(usize, @intCast(wo_a_shape[0]));
            const o_lora_rank = @as(usize, @intCast(wo_a_shape[1]));
            const heads_per_group = num_heads / n_groups;
            const o_rs = try ops.reshape(self.ctx, out_deroped, &[_]i32{ @intCast(batch), @intCast(n_groups), @intCast(heads_per_group), @intCast(seq_len), @intCast(head_dim) });
            defer o_rs.deinit();
            const o_tr = try ops.transposeAxes(self.ctx, o_rs, &[_]i32{ 1, 0, 3, 2, 4 });
            defer o_tr.deinit();
            const o_flat = try ops.reshape(self.ctx, o_tr, &[_]i32{ @intCast(n_groups), @intCast(batch * seq_len), @intCast(heads_per_group * head_dim) });
            defer o_flat.deinit();
            var grp_outs = try self.ctx.allocator.alloc(Array, n_groups);
            defer self.ctx.allocator.free(grp_outs);
            for (0..n_groups) |g| {
                const gi = try ops.slice(self.ctx, o_flat, &[_]i32{ @intCast(g), 0, 0 }, &[_]i32{ @intCast(g + 1), @intCast(batch * seq_len), @intCast(heads_per_group * head_dim) }, &[_]i32{});
                defer gi.deinit();
                const gi2 = try ops.reshape(self.ctx, gi, &[_]i32{ @intCast(batch * seq_len), @intCast(heads_per_group * head_dim) });
                defer gi2.deinit();
                const wa = try ops.slice(self.ctx, self.wo_a, &[_]i32{ @intCast(g), 0, 0 }, &[_]i32{ @intCast(g + 1), @intCast(o_lora_rank), @intCast(heads_per_group * head_dim) }, &[_]i32{});
                defer wa.deinit();
                const wa2 = try ops.reshape(self.ctx, wa, &[_]i32{ @intCast(o_lora_rank), @intCast(heads_per_group * head_dim) });
                defer wa2.deinit();
                const wat = try ops.transpose(self.ctx, wa2);
                defer wat.deinit();
                grp_outs[g] = try ops.matmul(self.ctx, gi2, wat);
            }
            defer for (grp_outs) |a| a.deinit();
            const cat = try shape_mod.concatenateAxis(self.ctx, grp_outs, 1);
            defer cat.deinit();
            const rs = try ops.reshape(self.ctx, cat, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(n_groups * o_lora_rank) });
            defer rs.deinit();
            break :blk if (self.wo_b_scales != null) blk2: {
                const qw = quantize_mod.QuantizedWeight{ .data = self.wo_b, .scales = self.wo_b_scales.?, .biases = self.wo_b_biases orelse Array.fromHandle(c.c.mlx_array_new()), .config = .{ .group_size = self.attn_quant_group_size, .bits = self.attn_quant_bits, .mode = self.attn_quant_mode }, .original_shape = &[_]i32{} };
                break :blk2 try quantize_mod.quantizedMatmul(self.ctx, rs, qw, true);
            } else blk2: {
                const wbt = try ops.transpose(self.ctx, self.wo_b);
                defer wbt.deinit();
                break :blk2 try ops.matmul(self.ctx, rs, wbt);
            };
        } else blk: {
            // Non-grouped path: wo_a is 2D
            const ot = try ops.transposeAxes(self.ctx, out_deroped, &[_]i32{ 0, 2, 1, 3 });
            defer ot.deinit();
            const of = try ops.reshape(self.ctx, ot, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(num_heads * head_dim) });
            defer of.deinit();
            // Use quantizedMatmul if wo_a has scales
            const oa = if (self.wo_a_scales != null) blk_qa: {
                const qw = quantize_mod.QuantizedWeight{
                    .data = self.wo_a,
                    .scales = self.wo_a_scales.?,
                    .biases = self.wo_a_biases orelse Array.fromHandle(c.c.mlx_array_new()),
                    .config = .{ .group_size = self.attn_quant_group_size, .bits = self.attn_quant_bits, .mode = self.attn_quant_mode },
                    .original_shape = &[_]i32{},
                };
                break :blk_qa try quantize_mod.quantizedMatmul(self.ctx, of, qw, true);
            } else blk_qa: {
                const wat = try ops.transpose(self.ctx, self.wo_a);
                defer wat.deinit();
                break :blk_qa try ops.matmul(self.ctx, of, wat);
            };
            defer oa.deinit();
            break :blk if (self.wo_b_scales != null) blk2: {
                const qw = quantize_mod.QuantizedWeight{ .data = self.wo_b, .scales = self.wo_b_scales.?, .biases = self.wo_b_biases orelse Array.fromHandle(c.c.mlx_array_new()), .config = .{ .group_size = self.attn_quant_group_size, .bits = self.attn_quant_bits, .mode = self.attn_quant_mode }, .original_shape = &[_]i32{} };
                break :blk2 try quantize_mod.quantizedMatmul(self.ctx, oa, qw, true);
            } else blk2: {
                const wbt = try ops.transpose(self.ctx, self.wo_b);
                defer wbt.deinit();
                break :blk2 try ops.matmul(self.ctx, oa, wbt);
            };
        };
        return out;
    }
};
/// Compressor module for KV compression (matching mlx-lm Compressor).
/// Performs learned softmax-gated pooling with optional overlap transform.
/// CSA (ratio=4): overlap=true, out_dim=head_dim*2
/// HCA (ratio=128): overlap=false, out_dim=head_dim
pub const Compressor = struct {
    ctx: EagerContext,
    wkv: Array, // Linear weight: [out_dim, hidden_size]
    wgate: Array, // Linear weight: [out_dim, hidden_size]
    ape: Array, // Positional bias: [compress_ratio, out_dim]
    norm_weight: Array, // RMSNorm weight: [head_dim]
    compress_ratio: usize,
    head_dim: usize,
    rope_head_dim: usize,
    overlap: bool,
    out_dim: usize,
    norm_eps: f32,

    pub fn deinit(self: *Compressor) void {
        self.wkv.deinit();
        self.wgate.deinit();
        self.ape.deinit();
        self.norm_weight.deinit();
    }

    /// Overlap transform for CSA (ratio=4): doubles the window by concatenating
    /// previous block's first half with current block's second half.
    fn overlapTransform(self: *Compressor, x: Array, fill_value: f32) !Array {
        // x: [B, W, R, out_dim] where out_dim = head_dim * 2
        const x_shape = x.shape();
        const B = x_shape[0];
        const W = x_shape[1];
        const R = x_shape[2];

        // second_half = x[:, :, :, head_dim:]
        const second_half = try ops.slice(self.ctx, x, &[_]i32{ 0, 0, 0, @intCast(self.head_dim) }, &[_]i32{ B, W, R, @intCast(self.out_dim) }, &[_]i32{});
        defer second_half.deinit();

        // fill_row = full([B, 1, R, head_dim], fill_value)
        var fill_raw = c.c.mlx_array_new();
        const fill_shape = [_]i32{ B, 1, R, @intCast(self.head_dim) };
        const fill_scalar = try ops.scalarF32(self.ctx, fill_value);
        defer fill_scalar.deinit();
        try c.check(c.c.mlx_full(&fill_raw, &fill_shape, fill_shape.len, fill_scalar.inner, @intCast(@intFromEnum(x.dtype())), self.ctx.stream.inner));
        const fill_row = Array.fromHandle(fill_raw);
        defer fill_row.deinit();

        // prev_first = concat([fill_row, x[:, :-1, :, :head_dim]], axis=1)
        const first_half = try ops.slice(self.ctx, x, &[_]i32{ 0, 0, 0, 0 }, &[_]i32{ B, W - 1, R, @intCast(self.head_dim) }, &[_]i32{});
        defer first_half.deinit();
        const prev_first = try shape_mod.concatenateAxis(self.ctx, &[_]Array{ fill_row, first_half }, 1);
        defer prev_first.deinit();

        // result = concat([prev_first, second_half], axis=2) → [B, W, 2R, head_dim]
        return shape_mod.concatenateAxis(self.ctx, &[_]Array{ prev_first, second_half }, 2);
    }

    /// Forward pass: compress input sequence into pooled blocks.
    /// x: [B, L, hidden_size]
    /// rope: RoPE module for position encoding on pooled output
    /// cache: optional DeepseekV4Cache for stateful accumulation
    /// start_pos: current position offset
    /// state_key: "compressor_state" or "indexer_state"
    /// Returns: pooled [B, N_pooled, head_dim]
    pub fn forward(
        self: *Compressor,
        x: Array,
        rope: *DSV4YarnRoPE,
        cache: ?*kvcache.DeepseekV4Cache,
        start_pos: usize,
        state_key: []const u8,
    ) !Array {
        const x_shape = x.shape();
        const B = @as(usize, @intCast(x_shape[0]));

        // Project: kv = wkv(x), gate = wgate(x)
        const wkv_t = try ops.transpose(self.ctx, self.wkv);
        defer wkv_t.deinit();
        const kv = try ops.matmul(self.ctx, x, wkv_t);
        defer kv.deinit();

        const wgate_t = try ops.transpose(self.ctx, self.wgate);
        defer wgate_t.deinit();
        const gate_raw = try ops.matmul(self.ctx, x, wgate_t);
        defer gate_raw.deinit();

        // Accumulate windows via cache (or direct computation)
        var ready_kv: Array = undefined;
        var ready_gate: Array = undefined;
        var pool_base: usize = undefined;

        if (cache) |v4_cache| {
            const result = try v4_cache.accumulateWindows(kv, gate_raw, state_key, self.compress_ratio, start_pos);
            ready_kv = result[0];
            ready_gate = result[1];
            pool_base = result[2];
        } else {
            const kv_len = @as(usize, @intCast(kv.shape()[1]));
            const usable = (kv_len / self.compress_ratio) * self.compress_ratio;
            if (usable > 0) {
                ready_kv = try ops.slice(self.ctx, kv, &[_]i32{ 0, 0, 0 }, &[_]i32{ @intCast(B), @intCast(usable), @intCast(self.out_dim) }, &[_]i32{});
                ready_gate = try ops.slice(self.ctx, gate_raw, &[_]i32{ 0, 0, 0 }, &[_]i32{ @intCast(B), @intCast(usable), @intCast(self.out_dim) }, &[_]i32{});
            } else {
                ready_kv = try array_mod.zeros(self.ctx.allocator, &[_]i32{ @intCast(B), 0, @intCast(self.out_dim) }, x.dtype());
                ready_gate = try array_mod.zeros(self.ctx.allocator, &[_]i32{ @intCast(B), 0, @intCast(self.out_dim) }, x.dtype());
            }
            pool_base = start_pos;
        }
        defer ready_kv.deinit();
        defer ready_gate.deinit();

        const ready_len = @as(usize, @intCast(ready_kv.shape()[1]));
        if (ready_len == 0) {
            const empty = try array_mod.zeros(self.ctx.allocator, &[_]i32{ @intCast(B), 0, @intCast(self.head_dim) }, x.dtype());
            if (cache) |v4_cache| {
                return v4_cache.updatePool(empty, state_key);
            }
            return empty;
        }

        const W = ready_len / self.compress_ratio;

        // Reshape to [B, W, ratio, out_dim]
        var kv_rs = try ops.reshape(self.ctx, ready_kv, &[_]i32{ @intCast(B), @intCast(W), @intCast(self.compress_ratio), @intCast(self.out_dim) });
        defer kv_rs.deinit();

        // gate + ape (positional bias)
        var gate_rs = try ops.reshape(self.ctx, ready_gate, &[_]i32{ @intCast(B), @intCast(W), @intCast(self.compress_ratio), @intCast(self.out_dim) });
        defer gate_rs.deinit();
        const ape_typed = try ops.astype(self.ctx, self.ape, gate_rs.dtype());
        defer ape_typed.deinit();
        const gate_biased = try ops.add(self.ctx, gate_rs, ape_typed);
        defer gate_biased.deinit();

        // Apply overlap transform if CSA (ratio=4)
        var kv_final: Array = kv_rs;
        var gate_final: Array = gate_biased;
        var kv_overlap_free = false;
        var gate_overlap_free = false;
        if (self.overlap) {
            kv_final = try self.overlapTransform(kv_rs, 0.0);
            kv_overlap_free = true;
            gate_final = try self.overlapTransform(gate_biased, -std.math.inf(f32));
            gate_overlap_free = true;
        }
        defer if (kv_overlap_free) kv_final.deinit();
        defer if (gate_overlap_free) gate_final.deinit();

        // Softmax gated pooling: weights = softmax(gate, axis=2), pooled = (kv * weights).sum(axis=2)
        const gate_f32 = try ops.astype(self.ctx, gate_final, .float32);
        defer gate_f32.deinit();
        const weights_arr = try ops.softmax(self.ctx, gate_f32, &[_]i32{2});
        defer weights_arr.deinit();
        const weights_typed = try ops.astype(self.ctx, weights_arr, kv_final.dtype());
        defer weights_typed.deinit();
        const weighted = try ops.multiply(self.ctx, kv_final, weights_typed);
        defer weighted.deinit();
        const new_pooled_raw = try reduce_mod.sumAxis(self.ctx, weighted, 2, false); // [B, W, out_dim or head_dim]
        defer new_pooled_raw.deinit();

        // RMSNorm on pooled output
        const normed = try fast_mod.rmsNorm(self.ctx, new_pooled_raw, self.norm_weight, self.norm_eps);
        defer normed.deinit();

        // Apply RoPE with compressed positions
        const normed_typed = try ops.astype(self.ctx, normed, x.dtype());
        defer normed_typed.deinit();

        // Positions: arange(W) * compress_ratio + pool_base
        // For simplicity, apply RoPE to the last rope_head_dim dimensions
        const normed_exp = try ops.expandDims(self.ctx, normed_typed, 1); // [B, 1, W, head_dim]
        defer normed_exp.deinit();

        // Apply partial RoPE (last rope_head_dim dims)
        // TODO: Use proper position computation with pool_base
        const roped = try rope.apply(normed_exp, pool_base, self.ctx.stream.inner, false);
        defer roped.deinit();
        const roped_sq = try shape_mod.squeezeAxis(self.ctx, roped, 1); // [B, W, head_dim]
        defer roped_sq.deinit();

        // Update cache pool
        if (cache) |v4_cache| {
            return v4_cache.updatePool(roped_sq, state_key);
        }
        return ops.copy(self.ctx, roped_sq);
    }
};

/// Compress KV cache along sequence dimension for CSA/HCA layers.
/// (Legacy pure-function interface — kept for backward compatibility)
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

        return result;
    } else {
        const raw = try ctx.allocator.alloc(Array, 2);
        defer ctx.allocator.free(raw);
        raw[0] = prefix_comp;
        raw[1] = suffix;
        const result = try shape_mod.concatenateAxis(ctx, raw, 1);

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

    const group_size_f = try Array.fromData(ctx.allocator, f32, &[_]f32{@floatFromInt(mhc_hidden_size)}, &[_]i32{1});
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

    // Build expanded scale using MLX ops (no dataSlice to avoid forcing evaluation)
    // scale has 3 values: [pre_scale, post_scale, comb_scale]
    // Expand to [mhc_mult * pre_scale, mhc_mult * post_scale, mhc_mult_sq * comb_scale]
    const scale_0 = try ops.slice(ctx, scale, &[_]i32{0}, &[_]i32{1}, &[_]i32{});
    defer scale_0.deinit();
    const scale_1 = try ops.slice(ctx, scale, &[_]i32{1}, &[_]i32{2}, &[_]i32{});
    defer scale_1.deinit();
    const scale_2 = try ops.slice(ctx, scale, &[_]i32{2}, &[_]i32{3}, &[_]i32{});
    defer scale_2.deinit();
    const pre_scales = try ops.broadcastTo(ctx, scale_0, &[_]i32{@intCast(mhc_mult)});
    defer pre_scales.deinit();
    const post_scales = try ops.broadcastTo(ctx, scale_1, &[_]i32{@intCast(mhc_mult)});
    defer post_scales.deinit();
    const comb_scales = try ops.broadcastTo(ctx, scale_2, &[_]i32{@intCast(mhc_mult_sq)});
    defer comb_scales.deinit();
    const scale_expanded = try shape_mod.concatenateAxis(ctx, &[_]Array{ pre_scales, post_scales, comb_scales }, 0);
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

    const post_sigmoid = try ops.sigmoid(ctx, post_mix_slice);
    defer post_sigmoid.deinit();
    const post_mult_arr = try Array.fromData(ctx.allocator, f32, &[_]f32{post_mult_value}, &[_]i32{1});
    defer post_mult_arr.deinit();
    const post_mix = try ops.multiply(ctx, post_sigmoid, post_mult_arr);
    defer post_mix.deinit();
    const post_mix_expanded = try ops.expandDims(ctx, post_mix, 3);

    const comb_mix = try ops.reshape(ctx, comb_mix_slice, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(mhc_mult), @intCast(mhc_mult) });

    return .{ pre_mix_expanded, post_mix_expanded, comb_mix };
}

/// Metal kernel source for fused hc_split_sinkhorn (matching mlx-lm).
/// Performs pre/post sigmoid + comb softmax + Sinkhorn normalization in a single dispatch.
const hc_split_sinkhorn_metal_source =
    \\uint idx = thread_position_in_grid.x;
    \\constexpr int MIX  = (2 + HC) * HC;
    \\constexpr int BASE = 2 * HC;
    \\const device float* mix = (const device float*)mixes + idx * MIX;
    \\device float* pre_out   = (device float*)pre  + idx * HC;
    \\device float* post_out  = (device float*)post + idx * HC;
    \\device float* comb_out  = (device float*)comb + idx * HC * HC;
    \\const float pre_scale  = scale[0];
    \\const float post_scale = scale[1];
    \\const float comb_scale = scale[2];
    \\const float epsv       = eps[0];
    \\{
    \\    float4 z = *(const device float4*)mix * pre_scale + *(const device float4*)base;
    \\    *(device float4*)pre_out = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
    \\}
    \\{
    \\    float4 z = *(const device float4*)(mix + HC) * post_scale + *(const device float4*)(base + HC);
    \\    *(device float4*)post_out = 2.0f * 1.0f / (1.0f + metal::fast::exp(-z));
    \\}
    \\float4 v0 = *(const device float4*)(mix + BASE) * comb_scale + *(const device float4*)(base + BASE);
    \\float4 v1 = *(const device float4*)(mix + BASE + 4) * comb_scale + *(const device float4*)(base + BASE + 4);
    \\float4 v2 = *(const device float4*)(mix + BASE + 8) * comb_scale + *(const device float4*)(base + BASE + 8);
    \\float4 v3 = *(const device float4*)(mix + BASE + 12) * comb_scale + *(const device float4*)(base + BASE + 12);
    \\float m0 = metal::max(metal::max(v0.x, v0.y), metal::max(v0.z, v0.w));
    \\float m1 = metal::max(metal::max(v1.x, v1.y), metal::max(v1.z, v1.w));
    \\float m2 = metal::max(metal::max(v2.x, v2.y), metal::max(v2.z, v2.w));
    \\float m3 = metal::max(metal::max(v3.x, v3.y), metal::max(v3.z, v3.w));
    \\float4 e0 = metal::fast::exp(v0 - m0);
    \\float4 e1 = metal::fast::exp(v1 - m1);
    \\float4 e2 = metal::fast::exp(v2 - m2);
    \\float4 e3 = metal::fast::exp(v3 - m3);
    \\float4 r0 = e0 * 1.0f / (e0.x + e0.y + e0.z + e0.w) + epsv;
    \\float4 r1 = e1 * 1.0f / (e1.x + e1.y + e1.z + e1.w) + epsv;
    \\float4 r2 = e2 * 1.0f / (e2.x + e2.y + e2.z + e2.w) + epsv;
    \\float4 r3 = e3 * 1.0f / (e3.x + e3.y + e3.z + e3.w) + epsv;
    \\float4 col = 1.0f / (r0 + r1 + r2 + r3 + epsv);
    \\r0 *= col; r1 *= col; r2 *= col; r3 *= col;
    \\for (int iter = 1; iter < ITERS; ++iter) {
    \\    r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
    \\    r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
    \\    r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
    \\    r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);
    \\    col = 1.0f / (r0 + r1 + r2 + r3 + epsv);
    \\    r0 *= col; r1 *= col; r2 *= col; r3 *= col;
    \\}
    \\*(device float4*)(comb_out) = r0;
    \\*(device float4*)(comb_out + 4) = r1;
    \\*(device float4*)(comb_out + 8) = r2;
    \\*(device float4*)(comb_out + 12) = r3;
;

/// Sinkhorn normalization on comb_res_mix.
pub fn sinkhornNormalize(
    ctx: EagerContext,
    x: Array,
    repeat: usize,
    eps: f32,
    stream: c.c.mlx_stream,
) !Array {
    _ = stream;
    // Use softmaxPrecise to match Python's precise=True behavior
    const softmaxed = try ops.softmaxPrecise(ctx, x, &[_]i32{-1});
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
/// Matches Python: (pre[..., None] * y).sum(axis=2).astype(x.dtype) where y = x.astype(f32)
pub fn mhcPreApplyMix(
    ctx: EagerContext,
    residual: Array,
    mix: Array,
    stream: c.c.mlx_stream,
) !Array {
    _ = stream;
    const orig_dtype = residual.dtype();
    // Promote to float32 for precision (matching Python .astype(mx.float32))
    const res_f32 = try ops.astype(ctx, residual, .float32);
    defer res_f32.deinit();
    const mix_f32 = try ops.astype(ctx, mix, .float32);
    defer mix_f32.deinit();
    const weighted = try ops.multiply(ctx, res_f32, mix_f32);
    defer weighted.deinit();
    const summed = try reduce_mod.sumAxis(ctx, weighted, 2, false);
    defer summed.deinit();
    // Cast back to original dtype
    return ops.astype(ctx, summed, orig_dtype);
}

/// MHC post: combine sublayer output with residual using post_mix and comb_mix.
/// Matches Python: all intermediate computation in float32, comb is transposed.
pub fn mhcPost(
    ctx: EagerContext,
    x: Array,
    residual: Array,
    post_mix: Array,
    comb_mix: Array,
    stream: c.c.mlx_stream,
) !Array {
    _ = stream;
    const orig_dtype = x.dtype();
    const shape = x.shape();
    const batch = @as(usize, @intCast(shape[0]));
    const seq_len = @as(usize, @intCast(shape[1]));
    const h = @as(usize, @intCast(shape[2]));
    const res_shape = residual.shape();
    const mhc_mult = @as(usize, @intCast(res_shape[2]));

    // Promote to float32 for precision (matching Python .astype(mx.float32))
    const x_f32 = try ops.astype(ctx, x, .float32);
    defer x_f32.deinit();
    const res_f32 = try ops.astype(ctx, residual, .float32);
    defer res_f32.deinit();
    const post_mix_f32 = try ops.astype(ctx, post_mix, .float32);
    defer post_mix_f32.deinit();

    const x_expanded = try ops.expandDims(ctx, x_f32, 2);
    defer x_expanded.deinit();
    const term1 = try ops.multiply(ctx, x_expanded, post_mix_f32);
    defer term1.deinit();

    const bs = batch * seq_len;
    // Python: mx.matmul(comb.swapaxes(-1, -2), residual) — comb is transposed
    const comb_2d = try ops.reshape(ctx, comb_mix, &[_]i32{ @intCast(bs), @intCast(mhc_mult), @intCast(mhc_mult) });
    defer comb_2d.deinit();
    const comb_2d_f32 = try ops.astype(ctx, comb_2d, .float32);
    defer comb_2d_f32.deinit();
    const comb_2d_t = try ops.transposeAxes(ctx, comb_2d_f32, &[_]i32{ 0, 2, 1 });
    defer comb_2d_t.deinit();
    const res_2d = try ops.reshape(ctx, res_f32, &[_]i32{ @intCast(bs), @intCast(mhc_mult), @intCast(h) });
    defer res_2d.deinit();
    const term2_2d = try ops.matmul(ctx, comb_2d_t, res_2d);
    defer term2_2d.deinit();
    const term2 = try ops.reshape(ctx, term2_2d, &[_]i32{ @intCast(batch), @intCast(seq_len), @intCast(mhc_mult), @intCast(h) });
    defer term2.deinit();

    const result_f32 = try ops.add(ctx, term1, term2);
    defer result_f32.deinit();
    // Cast back to original dtype
    return ops.astype(ctx, result_f32, orig_dtype);
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
        const pre_mix, const post_mix, const comb_mix = try mhcPreSplitMixes(ctx, mixes, self.hc_scale, self.hc_base, self.hc_mult, 2.0, self.hc_eps, stream);
        defer pre_mix.deinit();

        const comb_mix_norm = try sinkhornNormalize(ctx, comb_mix, self.hc_sinkhorn_iters, self.hc_eps, stream);

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
    config: DSV4Config,
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
    hc_head: ?HyperHead,
    lm_head: Array,

    pub fn deinit(self: *DSV4Model) void {
        self.config.deinitClone(self.allocator);
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
        self.norm.weight.deinit();
        if (self.hc_head) |*hh| {
            hh.deinit();
        }
        self.lm_head.deinit();
    }

    /// Check if any layer has expert weights loaded in memory.
    pub fn hasExpertsLoaded(self: *DSV4Model) bool {
        for (self.layers) |*layer| {
            if (layer.ffn.experts_loaded) return true;
        }
        return false;
    }

    /// Set expert stream provider on all MoE layers.
    pub fn setExpertStreamProvider(self: *DSV4Model, sp: *expert_stream.ExpertStreamProvider) void {
        for (self.layers, 0..) |*layer, i| {
            layer.ffn.stream_provider = sp;
            layer.ffn.layer_idx = i;
        }
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

        const forward_start = std.c.mach_absolute_time();

        // Embedding lookup
        var hidden = try arena.track(try self.embed_tokens.forward(input_ids));

        const embed_end = std.c.mach_absolute_time();

        // Expand to mHC format if enabled
        if (self.config.use_mhc) {
            hidden = try arena.track(try expandToMHC(self.ctx, hidden, self.config.hc_mult, stream));
        }

        // Pass through layers (eval after each to allow memory paging)
        var layer_total_ns: u64 = 0;
        for (self.layers, 0..) |*layer, i| {
            const layer_start = std.c.mach_absolute_time();
            const cache = if (caches) |cache_arr| cache_arr[i] else null;
            hidden = try arena.track(try layer.forward(hidden, input_ids, mask, cache, start_pos, stream));
            // Eval after each layer to materialize results and free lazy weight references
            // This allows MLX to page weights in/out of memory for large models
            try hidden.eval();
            const layer_end = std.c.mach_absolute_time();
            layer_total_ns += layer_end - layer_start;
        }

        // Compress from mHC format before final norm using HyperHead
        if (self.config.use_mhc and hidden.shape().len == 4) {
            if (self.hc_head) |*hh| {
                hidden = try arena.track(try hh.forward(hidden, stream));
            } else {
                // Fallback to simple mean if HyperHead weights not loaded
                hidden = try arena.track(try reduce_mod.meanAxis(self.ctx, hidden, 2, false));
            }
        }

        // Final norm
        const final_hidden = try arena.track(try self.norm.forward(hidden));

        // LM head: [B, S, dim] @ [dim, vocab] -> [B, S, vocab] — final output NOT tracked
        const lm_head_t = try arena.track(try ops.transpose(self.ctx, self.lm_head));
        const logits = try ops.matmul(self.ctx, final_hidden, lm_head_t);

        const forward_end = std.c.mach_absolute_time();
        const embed_ms = @as(f64, @floatFromInt(embed_end - forward_start)) / 1_000_000.0;
        const layers_ms = @as(f64, @floatFromInt(layer_total_ns)) / 1_000_000.0;
        const total_ms = @as(f64, @floatFromInt(forward_end - forward_start)) / 1_000_000.0;
        const head_ms = total_ms - layers_ms - embed_ms;
        const input_shape = input_ids.shape();
        const seq_len: usize = @intCast(input_shape[1]);
        std.log.info("[Forward] seq_len={d} total={d:.1}ms (embed={d:.1}ms layers={d:.1}ms head={d:.1}ms)", .{ seq_len, total_ms, embed_ms, layers_ms, head_ms });

        return logits;
    }

    /// Callback type for streaming generation.
    /// Called after each token is generated with the token ID and whether it's the final token.
    pub const StreamCallback = *const fn (ctx: *anyopaque, token: u32, is_final: bool) void;

    /// Generate tokens with optional streaming callback.
    /// When callback is provided, tokens are delivered immediately after generation.
    /// Returns only the generated tokens (excluding prompt).
    pub fn generateWithCallback(
        self: *DSV4Model,
        prompt_tokens: []const u32,
        max_new_tokens: usize,
        sampler_config: *@import("../sampling.zig").SamplerConfig,
        caches: []kvcache.KVCacheStrategy,
        stream: c.c.mlx_stream,
        callback_ctx: ?*anyopaque,
        callback: ?StreamCallback,
    ) ![]u32 {
        const allocator = self.allocator;

        if (max_new_tokens == 0) {
            return try allocator.alloc(u32, 0);
        }

        var tokens = try allocator.alloc(u32, prompt_tokens.len + max_new_tokens);
        defer allocator.free(tokens);
        @memcpy(tokens[0..prompt_tokens.len], prompt_tokens);

        var current_len = prompt_tokens.len;
        var start_pos: usize = 0;

        // Prefill: process all prompt tokens at once
        if (prompt_tokens.len > 0) {
            var arena = ScopedArrayArena.init(allocator);
            defer arena.deinit();

            const prompt_arr = try arena.track(try Array.fromData(allocator, u32, prompt_tokens, &[_]i32{ 1, @intCast(prompt_tokens.len) }));

            const prefill_mask = if (prompt_tokens.len > 1) blk: {
                const sl = prompt_tokens.len;
                const ws = self.config.sliding_window;
                var mask_data = try allocator.alloc(f32, sl * sl);
                defer allocator.free(mask_data);
                @memset(mask_data, 0);
                const neg_inf = -std.math.inf(f32);
                for (0..sl) |i| {
                    for (0..sl) |j| {
                        const causal = j > i;
                        const outside_window = ws > 0 and i >= ws and j < i + 1 - ws;
                        if (causal or outside_window) {
                            mask_data[i * sl + j] = neg_inf;
                        }
                    }
                }
                const mask_arr = try Array.fromData(allocator, f32, mask_data, &[_]i32{ 1, 1, @intCast(sl), @intCast(sl) });
                break :blk mask_arr;
            } else null;
            defer if (prefill_mask) |m| m.deinit();

            const logits = try self.forward(prompt_arr, prefill_mask, caches, start_pos, stream);

            const last_logits = try arena.track(try ops.slice(self.ctx, logits, &[_]i32{ 0, @intCast(prompt_tokens.len - 1), 0 }, &[_]i32{ 1, @intCast(prompt_tokens.len), @intCast(self.config.vocab_size) }, &[_]i32{}));
            const squeezed = try arena.track(try shape_mod.squeezeAxes(self.ctx, last_logits, &[_]i32{0}));
            const f32_logits = try arena.track(try ops.astype(self.ctx, squeezed, .float32));

            const next_token = (try sampler_config.sample(f32_logits, allocator)).token;
            tokens[current_len] = next_token;
            current_len += 1;
            start_pos = prompt_tokens.len;

            // Call callback immediately after first token is generated
            if (callback) |cb| {
                const is_final = max_new_tokens == 1 or next_token == 1;
                cb(callback_ctx.?, next_token, is_final);
            }

            if (next_token == 1) {
                std.log.info("EOS token generated after prefill, stopping", .{});
            }
        }

        // Generate new tokens autoregressively
        for (0..max_new_tokens - 1) |i| {
            if (current_len >= tokens.len) break;

            var arena = ScopedArrayArena.init(allocator);
            defer arena.deinit();

            const input_arr = try arena.track(try Array.fromData(allocator, u32, &[_]u32{tokens[current_len - 1]}, &[_]i32{ 1, 1 }));
            const logits = try self.forward(input_arr, null, caches, start_pos, stream);

            const squeezed = try arena.track(try shape_mod.squeezeAxes(self.ctx, logits, &[_]i32{ 0, 1 }));
            const f32_logits = try arena.track(try ops.astype(self.ctx, squeezed, .float32));

            const next_token = (try sampler_config.sample(f32_logits, allocator)).token;
            tokens[current_len] = next_token;
            current_len += 1;
            start_pos += 1;

            // Call callback immediately after each token is generated
            if (callback) |cb| {
                const is_final = i + 1 >= max_new_tokens - 1 or next_token == 1;
                cb(callback_ctx.?, next_token, is_final);
            }

            // Check for EOS token (ID 1 for DeepSeek V4)
            if (next_token == 1) {
                std.log.info("EOS token generated, stopping", .{});
                break;
            }
        }

        // Return only generated tokens (skip prompt)
        const result = try allocator.alloc(u32, current_len - prompt_tokens.len);
        @memcpy(result, tokens[prompt_tokens.len..current_len]);
        return result;
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

        // Guard against max_new_tokens=0 underflow (usize 0-1 wraps to maxInt)
        if (max_new_tokens == 0) {
            return try allocator.alloc(u32, 0);
        }

        var tokens = try allocator.alloc(u32, prompt_tokens.len + max_new_tokens);
        defer allocator.free(tokens);
        @memcpy(tokens[0..prompt_tokens.len], prompt_tokens);

        var current_len = prompt_tokens.len;
        var start_pos: usize = 0;

        // Prefill: process all prompt tokens at once
        if (prompt_tokens.len > 0) {
            var arena = ScopedArrayArena.init(allocator);
            defer arena.deinit();

            const prompt_arr = try arena.track(try Array.fromData(allocator, u32, prompt_tokens, &[_]i32{ 1, @intCast(prompt_tokens.len) }));

            // Create explicit causal mask with sliding window for prefill
            // Matches Python: create_attention_mask(return_array=True, window_size=sliding_window)
            const prefill_mask = if (prompt_tokens.len > 1) blk: {
                const sl = prompt_tokens.len;
                const ws = self.config.sliding_window;
                var mask_data = try allocator.alloc(f32, sl * sl);
                defer allocator.free(mask_data);
                @memset(mask_data, 0);
                const neg_inf = -std.math.inf(f32);
                for (0..sl) |i| {
                    for (0..sl) |j| {
                        const causal = j > i;
                        const outside_window = ws > 0 and i >= ws and j < i + 1 - ws;
                        if (causal or outside_window) {
                            mask_data[i * sl + j] = neg_inf;
                        }
                    }
                }
                const mask_arr = try Array.fromData(allocator, f32, mask_data, &[_]i32{ 1, 1, @intCast(sl), @intCast(sl) });
                break :blk mask_arr;
            } else null;
            defer if (prefill_mask) |m| m.deinit();

            const logits = try self.forward(prompt_arr, prefill_mask, caches, start_pos, stream);

            // Get last token logits
            const last_logits = try arena.track(try ops.slice(self.ctx, logits, &[_]i32{ 0, @intCast(prompt_tokens.len - 1), 0 }, &[_]i32{ 1, @intCast(prompt_tokens.len), @intCast(self.config.vocab_size) }, &[_]i32{}));
            const squeezed = try arena.track(try shape_mod.squeezeAxes(self.ctx, last_logits, &[_]i32{0}));
            const f32_logits = try arena.track(try ops.astype(self.ctx, squeezed, .float32));

            const next_token = (try sampler_config.sample(f32_logits, allocator)).token;
            tokens[current_len] = next_token;
            current_len += 1;
            start_pos = prompt_tokens.len;
        }

        // Generate new tokens autoregressively
        for (0..max_new_tokens - 1) |_| {
            if (current_len >= tokens.len) break;

            var arena = ScopedArrayArena.init(allocator);
            defer arena.deinit();

            const input_arr = try arena.track(try Array.fromData(allocator, u32, &[_]u32{tokens[current_len - 1]}, &[_]i32{ 1, 1 }));
            const logits = try self.forward(input_arr, null, caches, start_pos, stream);

            const squeezed = try arena.track(try shape_mod.squeezeAxes(self.ctx, logits, &[_]i32{ 0, 1 }));
            const f32_logits = try arena.track(try ops.astype(self.ctx, squeezed, .float32));

            const next_token = (try sampler_config.sample(f32_logits, allocator)).token;
            tokens[current_len] = next_token;
            current_len += 1;
            start_pos += 1;

            // Check for EOS token (ID 1 for DeepSeek V4)
            if (next_token == 1) {
                std.log.info("EOS token generated, stopping", .{});
                break;
            }
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
