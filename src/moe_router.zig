/// Reusable Mixture-of-Experts (MoE) routing module.
///
/// Provides top-k expert selection, token expansion to experts,
/// and weighted reduction of expert outputs back to token dimension.
/// All operations use mlx-c ops for GPU acceleration.
///
/// Requirements: R20.1, R20.2, R20.3
const std = @import("std");
const c = @import("c.zig");
const ops = @import("ops.zig");
const array_mod = @import("array.zig");
const reduce_mod = @import("ops/reduce.zig");
const shape_mod = @import("ops/shape.zig");
const sort_mod = @import("ops/sort.zig");
const cmp_mod = @import("ops/comparison.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

/// Result of top-k routing: selected expert indices and their weights.
pub const RouteResult = struct {
    /// Expert indices selected per token: [num_tokens, k]
    indices: Array,
    /// Routing weights for selected experts: [num_tokens, k]
    weights: Array,
    /// Number of experts selected per token
    k: usize,

    pub fn deinit(self: RouteResult) void {
        self.indices.deinit();
        self.weights.deinit();
    }
};

/// Configuration for the MoE router.
pub const MoERouterConfig = struct {
    num_experts: usize,
    top_k: usize,
    /// Whether to normalize weights to sum to 1
    normalize_weights: bool = true,
};

/// Reusable MoE router for top-k expert selection and dispatch.
pub const MoERouter = struct {
    config: MoERouterConfig,

    pub fn init(config: MoERouterConfig) MoERouter {
        return .{ .config = config };
    }

    /// Select top-k experts from routing scores using mlx-c ops.
    ///
    /// scores: [num_tokens, num_experts] — routing scores per token
    /// Returns RouteResult with indices [num_tokens, k] and weights [num_tokens, k].
    pub fn topkRoute(self: *const MoERouter, ctx: EagerContext, scores: Array) !RouteResult {
        const k = self.config.top_k;
        const num_experts = self.config.num_experts;

        // Use argsort ascending + slice last k (same pattern as DSV4Gate in deepseek_v4.zig)
        var sorted_handle = c.c.mlx_array_new();
        try c.check(c.c.mlx_argsort_axis(&sorted_handle, scores.inner, -1, ctx.stream.inner));
        const sorted = Array.fromHandle(sorted_handle);
        defer sorted.deinit();

        // Slice the last k elements (largest scores)
        const start_y = @as(i32, @intCast(num_experts - k));
        const stop_y = @as(i32, @intCast(num_experts));
        try sorted.eval();
        const num_tokens_i = sorted.shape()[0];
        const sliced = try ops.slice(ctx, sorted, &[_]i32{ 0, start_y }, &[_]i32{ num_tokens_i, stop_y }, &[_]i32{ 1, 1 });
        defer sliced.deinit();
        // Force contiguous layout (slice returns strided view; dataSlice reads raw memory)
        const indices = try shape_mod.contiguous(ctx, sliced);
        try indices.eval();

        // Gather the actual weights at those indices using take_along_axis
        var weights_handle = c.c.mlx_array_new();
        try c.check(c.c.mlx_take_along_axis(&weights_handle, scores.inner, indices.inner, 1, ctx.stream.inner));
        const weights = Array.fromHandle(weights_handle);

        // Optionally normalize weights
        if (self.config.normalize_weights) {
            const sum_w = try reduce_mod.sumAxis(ctx, weights, -1, true);
            defer sum_w.deinit();
            const eps = try ops.scalarF32(ctx, 1e-20);
            defer eps.deinit();
            const denom = try ops.add(ctx, sum_w, eps);
            defer denom.deinit();
            const normed = try ops.divide(ctx, weights, denom);
            weights.deinit();
            return RouteResult{
                .indices = indices,
                .weights = normed,
                .k = k,
            };
        }

        return RouteResult{
            .indices = indices,
            .weights = weights,
            .k = k,
        };
    }

    /// Expand input tokens to their assigned experts.
    ///
    /// For each token, creates k copies (one per selected expert).
    /// x: [num_tokens, hidden_dim]
    /// route: RouteResult from topkRoute
    /// Returns: [num_tokens * k, hidden_dim] — expanded tokens
    pub fn expandTokens(ctx: EagerContext, x: Array, route: RouteResult) !Array {
        const shape = x.shape();
        const num_tokens = @as(usize, @intCast(shape[0]));
        const hidden_dim = @as(usize, @intCast(shape[1]));
        const k = route.k;

        // Expand x from [N, D] to [N, k, D] via unsqueeze + broadcast
        const x_exp = try ops.expandDims(ctx, x, 1);
        defer x_exp.deinit();
        const x_broad = try ops.broadcastTo(ctx, x_exp, &[_]i32{
            @intCast(num_tokens),
            @intCast(k),
            @intCast(hidden_dim),
        });
        defer x_broad.deinit();

        // Reshape to [N*k, D]
        return ops.reshape(ctx, x_broad, &[_]i32{
            @intCast(num_tokens * k),
            @intCast(hidden_dim),
        });
    }

    /// Reduce expert outputs back to token dimension via weighted sum.
    ///
    /// expert_outs: [num_tokens * k, hidden_dim] — output from each expert
    /// route: RouteResult with weights [num_tokens, k]
    /// Returns: [num_tokens, hidden_dim] — weighted sum of expert outputs per token
    pub fn reduceExperts(ctx: EagerContext, expert_outs: Array, route: RouteResult) !Array {
        const out_shape = expert_outs.shape();
        const hidden_dim = @as(usize, @intCast(out_shape[1]));
        const weights_shape = route.weights.shape();
        const num_tokens = @as(usize, @intCast(weights_shape[0]));
        const k = route.k;

        // Reshape expert_outs from [N*k, D] to [N, k, D]
        const reshaped = try ops.reshape(ctx, expert_outs, &[_]i32{
            @intCast(num_tokens),
            @intCast(k),
            @intCast(hidden_dim),
        });
        defer reshaped.deinit();

        // Expand weights from [N, k] to [N, k, 1] for broadcasting
        const w_exp = try ops.expandDims(ctx, route.weights, 2);
        defer w_exp.deinit();

        // Weighted outputs: [N, k, D] * [N, k, 1] = [N, k, D]
        const weighted = try ops.multiply(ctx, reshaped, w_exp);
        defer weighted.deinit();

        // Sum over k dimension: [N, k, D] -> [N, D]
        return reduce_mod.sumAxis(ctx, weighted, 1, false);
    }
};

// === Unit Tests ===

test "MoERouter topkRoute selects correct top-k experts" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const router = MoERouter.init(.{
        .num_experts = 4,
        .top_k = 2,
        .normalize_weights = false,
    });

    // scores: [2, 4] — 2 tokens, 4 experts
    // Token 0: experts scored [0.1, 0.9, 0.3, 0.7] -> top-2 are experts 1 (0.9) and 3 (0.7)
    // Token 1: experts scored [0.8, 0.2, 0.6, 0.4] -> top-2 are experts 0 (0.8) and 2 (0.6)
    const scores = try Array.fromData(allocator, f32, &[_]f32{
        0.1, 0.9, 0.3, 0.7,
        0.8, 0.2, 0.6, 0.4,
    }, &[_]i32{ 2, 4 });
    defer scores.deinit();

    const result = try router.topkRoute(ctx, scores);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.k);

    // Check shapes
    try std.testing.expectEqual(@as(i32, 2), result.indices.shape()[0]);
    try std.testing.expectEqual(@as(i32, 2), result.indices.shape()[1]);
    try std.testing.expectEqual(@as(i32, 2), result.weights.shape()[0]);
    try std.testing.expectEqual(@as(i32, 2), result.weights.shape()[1]);

    // Verify weights are the top-2 scores per token (order may vary)
    const w_data = try result.weights.dataSlice(f32);

    // Token 0: top-2 scores should be {0.9, 0.7}
    const t0_w_sum = w_data[0] + w_data[1];
    try std.testing.expectApproxEqAbs(@as(f32, 1.6), t0_w_sum, 1e-5);

    // Token 1: top-2 scores should be {0.8, 0.6}
    const t1_w_sum = w_data[2] + w_data[3];
    try std.testing.expectApproxEqAbs(@as(f32, 1.4), t1_w_sum, 1e-5);
}

test "MoERouter topkRoute with weight normalization" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const router = MoERouter.init(.{
        .num_experts = 3,
        .top_k = 2,
        .normalize_weights = true,
    });

    // scores: [1, 3] — 1 token, 3 experts
    // Token 0: [0.2, 0.6, 0.4] -> top-2 are experts 1 (0.6) and 2 (0.4)
    // Normalized: 0.6/(0.6+0.4)=0.6, 0.4/(0.6+0.4)=0.4
    const scores = try Array.fromData(allocator, f32, &[_]f32{
        0.2, 0.6, 0.4,
    }, &[_]i32{ 1, 3 });
    defer scores.deinit();

    const result = try router.topkRoute(ctx, scores);
    defer result.deinit();

    const w_data = try result.weights.dataSlice(f32);
    // Weights should sum to ~1.0 (normalized)
    const sum = w_data[0] + w_data[1];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);

    // The two selected experts should be 1 (0.6) and 2 (0.4)
    const idx_data = try result.indices.dataSlice(u32);
    // Verify correct experts selected
    try std.testing.expect((idx_data[0] == 1 and idx_data[1] == 2) or (idx_data[0] == 2 and idx_data[1] == 1));

    // Verify normalized weights correspond to scores
    const score_data = try scores.dataSlice(f32);
    for (0..2) |i| {
        const expected_score = score_data[idx_data[i]];
        const expected_weight = expected_score / (0.6 + 0.4);
        try std.testing.expectApproxEqAbs(expected_weight, w_data[i], 1e-5);
    }
}

test "MoERouter expandTokens duplicates tokens for each expert" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // x: [2, 3] — 2 tokens, hidden_dim=3
    const x = try Array.fromData(allocator, f32, &[_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    }, &[_]i32{ 2, 3 });
    defer x.deinit();

    // Dummy route with k=2
    const indices = try Array.fromData(allocator, i32, &[_]i32{ 0, 1, 2, 3 }, &[_]i32{ 2, 2 });
    const weights = try Array.fromData(allocator, f32, &[_]f32{ 0.6, 0.4, 0.7, 0.3 }, &[_]i32{ 2, 2 });
    const route = RouteResult{ .indices = indices, .weights = weights, .k = 2 };
    defer route.deinit();

    const expanded = try MoERouter.expandTokens(ctx, x, route);
    defer expanded.deinit();

    // Should be [4, 3] — 2 tokens * 2 experts
    const shape = expanded.shape();
    try std.testing.expectEqual(@as(i32, 4), shape[0]);
    try std.testing.expectEqual(@as(i32, 3), shape[1]);

    const data = try expanded.dataSlice(f32);
    // Token 0 duplicated twice
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), data[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), data[5], 1e-5);
    // Token 1 duplicated twice
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), data[6], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), data[7], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), data[8], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), data[9], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), data[10], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), data[11], 1e-5);
}

test "MoERouter reduceExperts weighted sum" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // expert_outs: [4, 3] — 2 tokens * 2 experts, hidden_dim=3
    const expert_outs = try Array.fromData(allocator, f32, &[_]f32{
        1.0, 0.0, 0.0, // token 0, expert A
        0.0, 1.0, 0.0, // token 0, expert B
        0.0, 0.0, 1.0, // token 1, expert A
        1.0, 1.0, 1.0, // token 1, expert B
    }, &[_]i32{ 4, 3 });
    defer expert_outs.deinit();

    // weights: [2, 2]
    const weights = try Array.fromData(allocator, f32, &[_]f32{
        0.7, 0.3, // token 0: 0.7*expertA + 0.3*expertB
        0.4, 0.6, // token 1: 0.4*expertA + 0.6*expertB
    }, &[_]i32{ 2, 2 });
    const indices = try Array.fromData(allocator, i32, &[_]i32{ 0, 1, 2, 3 }, &[_]i32{ 2, 2 });
    const route = RouteResult{ .indices = indices, .weights = weights, .k = 2 };
    defer route.deinit();

    const reduced = try MoERouter.reduceExperts(ctx, expert_outs, route);
    defer reduced.deinit();

    // Should be [2, 3]
    const shape = reduced.shape();
    try std.testing.expectEqual(@as(i32, 2), shape[0]);
    try std.testing.expectEqual(@as(i32, 3), shape[1]);

    const data = try reduced.dataSlice(f32);
    // Token 0: 0.7*[1,0,0] + 0.3*[0,1,0] = [0.7, 0.3, 0.0]
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), data[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[2], 1e-5);
    // Token 1: 0.4*[0,0,1] + 0.6*[1,1,1] = [0.6, 0.6, 1.0]
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), data[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), data[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[5], 1e-5);
}

test "MoERouter end-to-end route-expand-reduce" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const router = MoERouter.init(.{
        .num_experts = 4,
        .top_k = 2,
        .normalize_weights = true,
    });

    // scores: [3, 4] — 3 tokens, 4 experts
    const scores = try Array.fromData(allocator, f32, &[_]f32{
        0.1, 0.5, 0.3, 0.1,
        0.4, 0.1, 0.1, 0.4,
        0.2, 0.2, 0.5, 0.1,
    }, &[_]i32{ 3, 4 });
    defer scores.deinit();

    // x: [3, 2] — 3 tokens, hidden_dim=2
    const x = try Array.fromData(allocator, f32, &[_]f32{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    }, &[_]i32{ 3, 2 });
    defer x.deinit();

    // Route
    const route = try router.topkRoute(ctx, scores);
    defer route.deinit();

    // Expand
    const expanded = try MoERouter.expandTokens(ctx, x, route);
    defer expanded.deinit();

    // expanded should be [6, 2] (3 tokens * 2 experts)
    const exp_shape = expanded.shape();
    try std.testing.expectEqual(@as(i32, 6), exp_shape[0]);
    try std.testing.expectEqual(@as(i32, 2), exp_shape[1]);

    // Simulate expert computation (identity for simplicity)
    // Reduce
    const reduced = try MoERouter.reduceExperts(ctx, expanded, route);
    defer reduced.deinit();

    // reduced should be [3, 2]
    const red_shape = reduced.shape();
    try std.testing.expectEqual(@as(i32, 3), red_shape[0]);
    try std.testing.expectEqual(@as(i32, 2), red_shape[1]);

    // Since we used identity expert and normalized weights sum to 1,
    // reduced should equal original x
    const red_data = try reduced.dataSlice(f32);
    const x_data = try x.dataSlice(f32);
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(x_data[i], red_data[i], 1e-4);
    }
}

test "MoERouter config with different num_experts and top_k" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Test with 8 experts, top-3
    const router = MoERouter.init(.{
        .num_experts = 8,
        .top_k = 3,
        .normalize_weights = false,
    });

    // scores: [1, 8]
    const scores = try Array.fromData(allocator, f32, &[_]f32{
        0.1, 0.2, 0.8, 0.3, 0.9, 0.4, 0.7, 0.5,
    }, &[_]i32{ 1, 8 });
    defer scores.deinit();

    const result = try router.topkRoute(ctx, scores);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 3), result.k);

    const idx_shape = result.indices.shape();
    try std.testing.expectEqual(@as(i32, 1), idx_shape[0]);
    try std.testing.expectEqual(@as(i32, 3), idx_shape[1]);

    // Top-3 experts should be 4 (0.9), 2 (0.8), 6 (0.7) — order may vary
    const idx_data = try result.indices.dataSlice(u32);
    // Verify the set of selected experts
    var found_4 = false;
    var found_2 = false;
    var found_6 = false;
    for (0..3) |i| {
        if (idx_data[i] == 4) found_4 = true;
        if (idx_data[i] == 2) found_2 = true;
        if (idx_data[i] == 6) found_6 = true;
    }
    try std.testing.expect(found_4);
    try std.testing.expect(found_2);
    try std.testing.expect(found_6);

    // Verify weights match scores at selected indices
    const w_data = try result.weights.dataSlice(f32);
    const score_data = try scores.dataSlice(f32);
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(score_data[idx_data[i]], w_data[i], 1e-5);
    }
}

// ============================================================
// Property-Based Test
// (Property 18)
//
// Feature: production-deployment, Property 18: MoE Top-K Selection
//
// For any routing score tensor and top-k value, the MoE_Router
// SHALL select the k experts with the highest scores. The output
// tensor shape SHALL match the input tensor shape along the token
// and hidden dimensions.
//
// **Validates: Requirements R20.1, R20.2**
// ============================================================

test "Property 18: MoE Top-K Selection — selected experts have highest scores and shapes are correct (100 iterations)" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    var prng = std.Random.DefaultPrng.init(18);
    const rand = prng.random();

    const token_choices = [_]usize{ 1, 2, 3, 4, 5, 8 };
    const expert_choices = [_]usize{ 4, 6, 8, 12, 16 };
    const topk_choices = [_]usize{ 1, 2, 3, 4 };
    const hidden_choices = [_]usize{ 2, 4, 8, 16 };

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // --- Random configuration ---
        const num_tokens = token_choices[rand.intRangeAtMost(usize, 0, token_choices.len - 1)];
        const num_experts = expert_choices[rand.intRangeAtMost(usize, 0, expert_choices.len - 1)];
        // top_k must be <= num_experts
        const max_k = @min(num_experts, topk_choices[topk_choices.len - 1]);
        var top_k = topk_choices[rand.intRangeAtMost(usize, 0, topk_choices.len - 1)];
        if (top_k > max_k) top_k = max_k;
        const hidden_dim = hidden_choices[rand.intRangeAtMost(usize, 0, hidden_choices.len - 1)];
        const normalize = rand.boolean();

        // --- Generate random routing scores [num_tokens, num_experts] ---
        const total_scores = num_tokens * num_experts;
        var score_data = try allocator.alloc(f32, total_scores);
        defer allocator.free(score_data);
        for (0..total_scores) |i| {
            score_data[i] = @as(f32, @floatFromInt(rand.intRangeAtMost(u32, 0, 10000))) / 10000.0;
        }

        const scores = try Array.fromData(allocator, f32, score_data, &[_]i32{
            @intCast(num_tokens),
            @intCast(num_experts),
        });
        defer scores.deinit();

        // --- Create router and run topkRoute ---
        const router = MoERouter.init(.{
            .num_experts = num_experts,
            .top_k = top_k,
            .normalize_weights = normalize,
        });

        const route = try router.topkRoute(ctx, scores);
        defer route.deinit();

        // --- Verify Property: indices shape is [num_tokens, k] ---
        const idx_shape = route.indices.shape();
        try std.testing.expectEqual(@as(usize, 2), idx_shape.len);
        try std.testing.expectEqual(@as(i32, @intCast(num_tokens)), idx_shape[0]);
        try std.testing.expectEqual(@as(i32, @intCast(top_k)), idx_shape[1]);

        // --- Verify Property: weights shape is [num_tokens, k] ---
        const w_shape = route.weights.shape();
        try std.testing.expectEqual(@as(usize, 2), w_shape.len);
        try std.testing.expectEqual(@as(i32, @intCast(num_tokens)), w_shape[0]);
        try std.testing.expectEqual(@as(i32, @intCast(top_k)), w_shape[1]);

        // --- Verify Property: selected indices correspond to top-k highest scores ---
        const scores_data = try scores.dataSlice(f32);
        const idx_data = try route.indices.dataSlice(u32);
        const w_data = try route.weights.dataSlice(f32);

        for (0..num_tokens) |t| {
            const token_scores = scores_data[t * num_experts .. (t + 1) * num_experts];
            const selected_start = t * top_k;
            const selected_indices = idx_data[selected_start .. selected_start + top_k];
            const selected_weights = w_data[selected_start .. selected_start + top_k];

            // Compute the expected top-k by sorting scores on CPU
            const ScoredExpert = struct { score: f32, idx: usize };
            var sorted_pairs = try allocator.alloc(ScoredExpert, num_experts);
            defer allocator.free(sorted_pairs);
            for (0..num_experts) |e| {
                sorted_pairs[e] = .{ .score = token_scores[e], .idx = e };
            }
            // Sort descending by score
            std.mem.sort(
                ScoredExpert,
                sorted_pairs,
                {},
                struct {
                    fn lessThan(_: void, a: ScoredExpert, b: ScoredExpert) bool {
                        return a.score > b.score;
                    }
                }.lessThan,
            );

            // The k-th highest score is the threshold for top-k selection
            const kth_score = sorted_pairs[top_k - 1].score;

            // Verify: each selected expert's score must be >= the k-th highest score
            // (handles ties correctly — any expert with score >= threshold is valid)
            for (0..top_k) |ki| {
                const sel_idx = selected_indices[ki];
                try std.testing.expect(sel_idx < num_experts);
                const sel_score = token_scores[sel_idx];
                try std.testing.expect(sel_score >= kth_score - 1e-5);
            }

            // Verify: no non-selected expert has score strictly greater than all selected scores
            var min_selected_score: f32 = std.math.floatMax(f32);
            for (0..top_k) |ki| {
                const sel_score = token_scores[selected_indices[ki]];
                if (sel_score < min_selected_score) {
                    min_selected_score = sel_score;
                }
            }
            for (0..num_experts) |e| {
                var is_selected = false;
                for (0..top_k) |ki| {
                    if (selected_indices[ki] == @as(u32, @intCast(e))) {
                        is_selected = true;
                        break;
                    }
                }
                if (!is_selected) {
                    // Non-selected expert's score must be <= min selected score (with tolerance for ties)
                    try std.testing.expect(token_scores[e] <= min_selected_score + 1e-5);
                }
            }

            // Verify weights match scores at selected indices (non-normalized)
            if (!normalize) {
                for (0..top_k) |ki| {
                    const expert_score = token_scores[selected_indices[ki]];
                    try std.testing.expectApproxEqAbs(expert_score, selected_weights[ki], 1e-5);
                }
            }

            // If normalized, weights should sum to ~1.0
            if (normalize) {
                var weight_sum: f32 = 0.0;
                for (0..top_k) |ki| {
                    weight_sum += selected_weights[ki];
                }
                try std.testing.expectApproxEqAbs(@as(f32, 1.0), weight_sum, 1e-4);
            }
        }

        // --- Verify Property: expandTokens output shape is [num_tokens * k, hidden_dim] ---
        const total_input = num_tokens * hidden_dim;
        var input_data = try allocator.alloc(f32, total_input);
        defer allocator.free(input_data);
        for (0..total_input) |i| {
            input_data[i] = @as(f32, @floatFromInt(rand.intRangeAtMost(u32, 0, 1000))) / 1000.0;
        }

        const x = try Array.fromData(allocator, f32, input_data, &[_]i32{
            @intCast(num_tokens),
            @intCast(hidden_dim),
        });
        defer x.deinit();

        const expanded = try MoERouter.expandTokens(ctx, x, route);
        defer expanded.deinit();

        const exp_shape = expanded.shape();
        try std.testing.expectEqual(@as(usize, 2), exp_shape.len);
        try std.testing.expectEqual(@as(i32, @intCast(num_tokens * top_k)), exp_shape[0]);
        try std.testing.expectEqual(@as(i32, @intCast(hidden_dim)), exp_shape[1]);

        // --- Verify Property: reduceExperts output shape is [num_tokens, hidden_dim] ---
        const reduced = try MoERouter.reduceExperts(ctx, expanded, route);
        defer reduced.deinit();

        const red_shape = reduced.shape();
        try std.testing.expectEqual(@as(usize, 2), red_shape.len);
        try std.testing.expectEqual(@as(i32, @intCast(num_tokens)), red_shape[0]);
        try std.testing.expectEqual(@as(i32, @intCast(hidden_dim)), red_shape[1]);
    }
}
