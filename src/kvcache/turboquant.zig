/// TurboQuant — Near-optimal online vector quantization for KV cache.
///
/// Implements the TurboQuant algorithm (arXiv:2504.19874) for KV cache quantization.
/// Key properties:
///   - Data-oblivious (no calibration data needed — ideal for online KV cache quantization)
///   - Near-optimal MSE distortion (within ≈2.7x of information-theoretic lower bound)
///   - Unbiased inner product estimation (MSE quantizer + QJL residual correction)
///   - Accelerator-friendly (random rotation + per-coordinate scalar quantization)
///
/// Algorithm overview:
///   1. Random rotation: y = Π · x (induces Beta distribution on coordinates)
///   2. Scalar quantization: idx = nearest(y, codebook) per coordinate
///   3. Dequantization: ỹ = codebook[idx], x̃ = Π^T · ỹ
///   4. (Optional) QJL residual: unbiased inner product via 1-bit quantized residual
///
/// Codebooks are precomputed Lloyd-Max optimal quantizers for the Beta distribution
/// that arises from random rotation of unit-norm vectors.
///
/// Reference: TurboQuant paper arXiv:2504.19874
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const ops_mod = @import("../ops.zig");
const linalg_mod = @import("../ops/linalg.zig");
const reduce_mod = @import("../ops/reduce.zig");

const Array = array_mod.Array;
const EagerContext = ops_mod.EagerContext;

/// Precomputed Lloyd-Max codebooks for Beta distribution (high-dimensional limit ≈ Gaussian).
/// These are optimal scalar quantizer centroids for N(0, 1/d) distribution,
/// scaled by sqrt(d) so they work on unit-norm vectors after random rotation.
///
/// Values from TurboQuant paper Table 1 (normalized for unit variance Gaussian).
pub const Codebook = struct {
    /// 1-bit (2 centroids): {-0.7979, +0.7979} (= ±sqrt(2/π))
    pub const b1 = [_]f32{ -0.7979, 0.7979 };

    /// 2-bit (4 centroids): {-1.510, -0.4528, +0.4528, +1.510}
    pub const b2 = [_]f32{ -1.510, -0.4528, 0.4528, 1.510 };

    /// 3-bit (8 centroids)
    pub const b3 = [_]f32{ -2.152, -1.344, -0.7560, -0.2451, 0.2451, 0.7560, 1.344, 2.152 };

    /// 4-bit (16 centroids)
    pub const b4 = [_]f32{
        -2.733, -2.069, -1.618, -1.256, -0.9424, -0.6568, -0.3881, -0.1284,
        0.1284,  0.3881,  0.6568,  0.9424,  1.256,  1.618,  2.069,  2.733,
    };

    /// Get codebook for given bit-width.
    pub fn get(bits: u8) []const f32 {
        return switch (bits) {
            1 => &b1,
            2 => &b2,
            3 => &b3,
            4 => &b4,
            else => &b4,
        };
    }

    /// Get decision boundaries (midpoints between consecutive centroids).
    pub fn boundaries(bits: u8, buf: []f32) []f32 {
        const cb = get(bits);
        const n = cb.len - 1;
        for (0..n) |i| {
            buf[i] = (cb[i] + cb[i + 1]) / 2.0;
        }
        return buf[0..n];
    }
};

/// TurboQuant configuration.
pub const TurboQuantConfig = struct {
    bits: u8 = 4,
    /// Whether to apply QJL residual correction for unbiased inner products.
    /// Costs 1 extra bit per coordinate but eliminates inner product bias.
    use_qjl: bool = false,
};

/// State for TurboQuant quantizer (holds the random rotation matrix).
pub const TurboQuantState = struct {
    ctx: EagerContext,
    dim: usize,
    config: TurboQuantConfig,
    /// Random rotation matrix Π: [dim, dim], orthogonal.
    rotation: Array,
    /// Transpose of rotation matrix: Π^T.
    rotation_t: Array,
    /// Codebook centroids scaled by 1/sqrt(dim).
    scaled_codebook: Array,
    /// QJL random projection matrix S: [dim, dim], i.i.d. N(0,1). Only allocated if use_qjl=true.
    qjl_matrix: ?Array,
    /// Transpose of QJL matrix S^T. Only allocated if use_qjl=true.
    qjl_matrix_t: ?Array,

    pub fn init(ctx: EagerContext, dim: usize, config: TurboQuantConfig) !TurboQuantState {
        // Generate random rotation matrix via QR decomposition of random Gaussian matrix.
        const stream = ctx.stream.inner;
        const shape = &[_]i32{ @intCast(dim), @intCast(dim) };

        var rand_mat = c.c.mlx_array_new();
        try c.check(c.c.mlx_random_normal(
            &rand_mat, shape.ptr, shape.len, c.c.MLX_FLOAT32,
            0.0, 1.0, .{ .ctx = null }, stream,
        ));
        defer _ = c.c.mlx_array_free(rand_mat);

        // QR decomposition: rand_mat = Q · R, Q is orthogonal
        const qr_result = try linalg_mod.qr(ctx, Array.fromHandle(rand_mat));
        // qr returns { .q, .r } — we only need Q
        const q_mat = qr_result.q;
        const r_mat = qr_result.r;
        defer r_mat.deinit();

        const q_t = try ops_mod.transpose(ctx, q_mat);

        // Scale codebook by 1/sqrt(dim) for unit-norm vectors
        const cb = Codebook.get(config.bits);
        const scale_factor: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
        var scaled_data: [16]f32 = undefined;
        for (cb, 0..) |v, i| scaled_data[i] = v * scale_factor;

        const scaled_cb = try Array.fromData(
            ctx.allocator, f32,
            scaled_data[0..cb.len],
            &[_]i32{@intCast(cb.len)},
        );

        // QJL random projection matrix (only if use_qjl is enabled)
        var qjl_mat: ?Array = null;
        var qjl_mat_t: ?Array = null;
        if (config.use_qjl) {
            var s_raw = c.c.mlx_array_new();
            try c.check(c.c.mlx_random_normal(
                &s_raw, shape.ptr, shape.len, c.c.MLX_FLOAT32,
                0.0, 1.0, .{ .ctx = null }, stream,
            ));
            qjl_mat = Array.fromHandle(s_raw);
            qjl_mat_t = try ops_mod.transpose(ctx, qjl_mat.?);
        }

        return .{
            .ctx = ctx,
            .dim = dim,
            .config = config,
            .rotation = q_mat,
            .rotation_t = q_t,
            .scaled_codebook = scaled_cb,
            .qjl_matrix = qjl_mat,
            .qjl_matrix_t = qjl_mat_t,
        };
    }

    pub fn deinit(self: *TurboQuantState) void {
        self.rotation.deinit();
        self.rotation_t.deinit();
        self.scaled_codebook.deinit();
        if (self.qjl_matrix) |m| m.deinit();
        if (self.qjl_matrix_t) |m| m.deinit();
    }

    /// Quantize a vector: rotate → find nearest centroid per coordinate → store indices.
    /// Input: x [*, dim] (any batch dimensions)
    /// Returns: (indices [*, dim] as uint8/uint16, norm [*] as float32)
    pub fn quantize(self: *TurboQuantState, x: Array) !QuantizedVector {
        const ctx = self.ctx;
        const x_shape = x.shape();
        const dim_i = @as(i32, @intCast(self.dim));

        // Compute and store L2 norm for later rescaling
        const x_sq = try ops_mod.multiply(ctx, x, x);
        defer x_sq.deinit();
        const norm_sq = try reduce_mod.sumAxis(ctx, x_sq, @as(i32, @intCast(x_shape.len)) - 1, true);
        defer norm_sq.deinit();
        const norm = try ops_mod.sqrt(ctx, norm_sq);

        // Normalize to unit norm
        const x_normed = try ops_mod.divide(ctx, x, norm);
        defer x_normed.deinit();

        // Flatten batch dims: [*, dim] -> [N, dim]
        const batch_size = blk: {
            var n: i32 = 1;
            for (x_shape[0 .. x_shape.len - 1]) |s| n *= s;
            break :blk n;
        };
        const x_flat = try ops_mod.reshape(ctx, x_normed, &[_]i32{ batch_size, dim_i });
        defer x_flat.deinit();

        // Random rotation: y = x_flat @ Π^T  (each row rotated)
        const y = try ops_mod.matmul(ctx, x_flat, self.rotation_t);
        defer y.deinit();

        // Find nearest centroid per coordinate
        // y: [N, dim], codebook: [num_centroids]
        // Expand y to [N, dim, 1] and codebook to [1, 1, num_centroids]
        const y_exp = try ops_mod.expandDims(ctx, y, 2);
        defer y_exp.deinit();

        const cb_shape = self.scaled_codebook.shape();
        const num_centroids = cb_shape[0];
        const cb_reshaped = try ops_mod.reshape(ctx, self.scaled_codebook, &[_]i32{ 1, 1, num_centroids });
        defer cb_reshaped.deinit();

        // Distance: |y - codebook|  → [N, dim, num_centroids]
        const diff = try ops_mod.subtract(ctx, y_exp, cb_reshaped);
        defer diff.deinit();
        const dist = try ops_mod.abs(ctx, diff);
        defer dist.deinit();

        // Argmin over last axis → indices [N, dim]
        var idx_raw = c.c.mlx_array_new();
        try c.check(c.c.mlx_argmin_axis(&idx_raw, dist.inner, 2, false, ctx.stream.inner));
        const indices_flat = Array.fromHandle(idx_raw);

        // Reshape indices back to original batch shape
        const idx_shape = try ctx.allocator.alloc(i32, x_shape.len);
        defer ctx.allocator.free(idx_shape);
        @memcpy(idx_shape, x_shape);
        const indices = try ops_mod.reshape(ctx, indices_flat, idx_shape);
        indices_flat.deinit();

        // Reshape norm back (remove keepdim)
        const norm_shape = try ctx.allocator.alloc(i32, x_shape.len - 1);
        defer ctx.allocator.free(norm_shape);
        @memcpy(norm_shape, x_shape[0 .. x_shape.len - 1]);
        const norm_out = try ops_mod.reshape(ctx, norm, norm_shape);
        norm.deinit();

        return .{ .indices = indices, .norm = norm_out };
    }

    /// Compute QJL (Quantized Johnson-Lindenstrauss) on the residual for unbiased inner products.
    /// Given original x and MSE-reconstructed x̃, computes:
    ///   residual r = x - x̃
    ///   qjl_signs = sign(S · r)  — 1-bit quantization of projected residual
    ///   residual_norm = ‖r‖₂
    ///
    /// For inner product estimation: ⟨y, x⟩ ≈ ⟨y, x̃⟩ + (√(π/2)/d) · ‖r‖ · ⟨y, S^T · qjl_signs⟩
    /// This is unbiased: E[estimate] = ⟨y, x⟩
    pub fn computeQjl(self: *TurboQuantState, x: Array, x_hat: Array) !QjlData {
        const ctx = self.ctx;
        const s_mat = self.qjl_matrix orelse return error.QjlNotEnabled;

        // r = x - x̃
        const residual = try ops_mod.subtract(ctx, x, x_hat);
        defer residual.deinit();

        // ‖r‖₂
        const r_sq = try ops_mod.multiply(ctx, residual, residual);
        defer r_sq.deinit();
        const r_shape = residual.shape();
        const r_norm_sq = try reduce_mod.sumAxis(ctx, r_sq, @as(i32, @intCast(r_shape.len)) - 1, false);
        defer r_norm_sq.deinit();
        const residual_norm = try ops_mod.sqrt(ctx, r_norm_sq);

        // Flatten residual to [N, dim]
        const dim_i = @as(i32, @intCast(self.dim));
        const batch_size = blk: {
            var n: i32 = 1;
            for (r_shape[0 .. r_shape.len - 1]) |s| n *= s;
            break :blk n;
        };
        const r_flat = try ops_mod.reshape(ctx, residual, &[_]i32{ batch_size, dim_i });
        defer r_flat.deinit();

        // Project: S · r^T → sign  (r_flat @ S^T gives [N, dim], then sign)
        const projected = try ops_mod.matmul(ctx, r_flat, try ops_mod.transpose(ctx, s_mat));
        defer projected.deinit();

        // sign(projected) → {-1, +1}^dim
        var sign_raw = c.c.mlx_array_new();
        try c.check(c.c.mlx_sign(&sign_raw, projected.inner, ctx.stream.inner));
        const signs_flat = Array.fromHandle(sign_raw);

        // Reshape signs back to original batch shape
        const out_shape = try ctx.allocator.alloc(i32, r_shape.len);
        defer ctx.allocator.free(out_shape);
        @memcpy(out_shape, r_shape);
        const signs = try ops_mod.reshape(ctx, signs_flat, out_shape);
        signs_flat.deinit();

        return .{ .signs = signs, .residual_norm = residual_norm };
    }

    /// Reconstruct with QJL correction for unbiased inner product estimation.
    /// x̂ = x̃_mse + (√(π/2)/d) · ‖r‖ · S^T · signs
    pub fn dequantizeWithQjl(self: *TurboQuantState, qv: QuantizedVector, qjl: QjlData) !Array {
        const ctx = self.ctx;
        const s_t = self.qjl_matrix_t orelse return error.QjlNotEnabled;
        const dim_f: f32 = @floatFromInt(self.dim);

        // Base MSE reconstruction
        const x_hat = try self.dequantize(qv);
        defer x_hat.deinit();

        // QJL correction: (√(π/2)/d) · ‖r‖ · S^T · signs
        const coeff: f32 = @sqrt(std.math.pi / 2.0) / dim_f;
        const coeff_arr = try ops_mod.scalarF32(ctx, coeff);
        defer coeff_arr.deinit();

        // Flatten signs to [N, dim]
        const s_shape = qjl.signs.shape();
        const dim_i = @as(i32, @intCast(self.dim));
        const batch_size = blk: {
            var n: i32 = 1;
            for (s_shape[0 .. s_shape.len - 1]) |s| n *= s;
            break :blk n;
        };
        const signs_flat = try ops_mod.reshape(ctx, qjl.signs, &[_]i32{ batch_size, dim_i });
        defer signs_flat.deinit();

        // S^T · signs: [dim, dim] @ [N, dim]^T → we do signs_flat @ S_t^T = signs_flat @ S
        const correction_flat = try ops_mod.matmul(ctx, signs_flat, s_t);
        defer correction_flat.deinit();

        // Reshape back
        const out_shape = try ctx.allocator.alloc(i32, s_shape.len);
        defer ctx.allocator.free(out_shape);
        @memcpy(out_shape, s_shape);
        const correction = try ops_mod.reshape(ctx, correction_flat, out_shape);
        defer correction.deinit();

        // Scale by coeff * residual_norm
        const norm_exp = try ops_mod.expandDims(ctx, qjl.residual_norm, @as(i32, @intCast(s_shape.len)) - 1);
        defer norm_exp.deinit();
        const scaled_norm = try ops_mod.multiply(ctx, coeff_arr, norm_exp);
        defer scaled_norm.deinit();
        const scaled_correction = try ops_mod.multiply(ctx, scaled_norm, correction);
        defer scaled_correction.deinit();

        // x̂ = x̃ + correction
        return ops_mod.add(ctx, x_hat, scaled_correction);
    }

    /// Dequantize: look up centroids → inverse rotate → rescale by norm.
    /// Returns: reconstructed x̃ [*, dim]
    pub fn dequantize(self: *TurboQuantState, qv: QuantizedVector) !Array {
        const ctx = self.ctx;
        const idx_shape = qv.indices.shape();
        const dim_i = @as(i32, @intCast(self.dim));

        // Flatten indices to [N, dim]
        const batch_size = blk: {
            var n: i32 = 1;
            for (idx_shape[0 .. idx_shape.len - 1]) |s| n *= s;
            break :blk n;
        };
        const idx_flat = try ops_mod.reshape(ctx, qv.indices, &[_]i32{ batch_size, dim_i });
        defer idx_flat.deinit();

        // Look up centroids: scaled_codebook[indices] → [N, dim]
        var y_hat_raw = c.c.mlx_array_new();
        try c.check(c.c.mlx_take(&y_hat_raw, self.scaled_codebook.inner, idx_flat.inner, ctx.stream.inner));
        const y_hat = Array.fromHandle(y_hat_raw);
        defer y_hat.deinit();

        // Inverse rotation: x̃ = y_hat @ Π  (each row inverse-rotated)
        const x_hat_flat = try ops_mod.matmul(ctx, y_hat, self.rotation);
        defer x_hat_flat.deinit();

        // Reshape back to original shape
        const out_shape = try ctx.allocator.alloc(i32, idx_shape.len);
        defer ctx.allocator.free(out_shape);
        @memcpy(out_shape, idx_shape);
        const x_hat = try ops_mod.reshape(ctx, x_hat_flat, out_shape);
        defer x_hat.deinit();

        // Rescale by original norm
        const norm_exp = try ops_mod.expandDims(ctx, qv.norm, @as(i32, @intCast(idx_shape.len)) - 1);
        defer norm_exp.deinit();

        return ops_mod.multiply(ctx, x_hat, norm_exp);
    }
};

/// Quantized vector representation.
pub const QuantizedVector = struct {
    /// Centroid indices per coordinate: [*, dim] as int32.
    indices: Array,
    /// L2 norm of original vector: [*] as float32.
    norm: Array,

    pub fn deinit(self: QuantizedVector) void {
        self.indices.deinit();
        self.norm.deinit();
    }
};

/// QJL residual data for unbiased inner product correction.
pub const QjlData = struct {
    /// Sign of projected residual: [*, dim] as float32 ({-1, +1}).
    signs: Array,
    /// L2 norm of residual: [*] as float32.
    residual_norm: Array,

    pub fn deinit(self: QjlData) void {
        self.signs.deinit();
        self.residual_norm.deinit();
    }
};

// ============================================================
// Tests
// ============================================================

test "Codebook: get returns correct sizes" {
    try std.testing.expectEqual(@as(usize, 2), Codebook.get(1).len);
    try std.testing.expectEqual(@as(usize, 4), Codebook.get(2).len);
    try std.testing.expectEqual(@as(usize, 8), Codebook.get(3).len);
    try std.testing.expectEqual(@as(usize, 16), Codebook.get(4).len);
}

test "Codebook: centroids are symmetric around zero" {
    for ([_]u8{ 1, 2, 3, 4 }) |bits| {
        const cb = Codebook.get(bits);
        for (0..cb.len / 2) |i| {
            const diff = @abs(cb[i] + cb[cb.len - 1 - i]);
            try std.testing.expect(diff < 1e-3);
        }
    }
}

test "Codebook: boundaries are midpoints" {
    var buf: [15]f32 = undefined;
    const bounds = Codebook.boundaries(2, &buf);
    try std.testing.expectEqual(@as(usize, 3), bounds.len);
    // Midpoint of -1.510 and -0.4528 = -0.9814
    try std.testing.expect(@abs(bounds[0] - (-0.9814)) < 0.01);
}

test "TurboQuantConfig: default values" {
    const cfg = TurboQuantConfig{};
    try std.testing.expectEqual(@as(u8, 4), cfg.bits);
    try std.testing.expectEqual(false, cfg.use_qjl);
}

test "TurboQuantState: init and deinit" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    var state = try TurboQuantState.init(ctx, 16, .{ .bits = 2 });
    defer state.deinit();

    // Rotation matrix should be [16, 16]
    const r_shape = state.rotation.shape();
    try std.testing.expectEqual(@as(usize, 2), r_shape.len);
    try std.testing.expectEqual(@as(i32, 16), r_shape[0]);
    try std.testing.expectEqual(@as(i32, 16), r_shape[1]);

    // Scaled codebook should have 4 entries for 2-bit
    const cb_shape = state.scaled_codebook.shape();
    try std.testing.expectEqual(@as(i32, 4), cb_shape[0]);
}

test "TurboQuantState: quantize and dequantize round-trip preserves shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    var state = try TurboQuantState.init(ctx, 8, .{ .bits = 2 });
    defer state.deinit();

    // Create random input [4, 8]
    const stream = ctx.stream.inner;
    const shape = &[_]i32{ 4, 8 };
    var x_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_random_normal(
        &x_raw, shape.ptr, shape.len, c.c.MLX_FLOAT32,
        0.0, 1.0, .{ .ctx = null }, stream,
    ));
    const x = Array.fromHandle(x_raw);
    defer x.deinit();

    // Quantize
    const qv = try state.quantize(x);
    defer qv.deinit();

    // Check indices shape = [4, 8]
    const idx_shape = qv.indices.shape();
    try std.testing.expectEqual(@as(usize, 2), idx_shape.len);
    try std.testing.expectEqual(@as(i32, 4), idx_shape[0]);
    try std.testing.expectEqual(@as(i32, 8), idx_shape[1]);

    // Check norm shape = [4]
    const norm_shape = qv.norm.shape();
    try std.testing.expectEqual(@as(usize, 1), norm_shape.len);
    try std.testing.expectEqual(@as(i32, 4), norm_shape[0]);

    // Dequantize
    const x_hat = try state.dequantize(qv);
    defer x_hat.deinit();

    // Check reconstructed shape = [4, 8]
    const x_hat_shape = x_hat.shape();
    try std.testing.expectEqual(@as(usize, 2), x_hat_shape.len);
    try std.testing.expectEqual(@as(i32, 4), x_hat_shape[0]);
    try std.testing.expectEqual(@as(i32, 8), x_hat_shape[1]);
}
