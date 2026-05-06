/// Tests for fused (compiled) composite operations (src/ops/fused.zig).
///
/// Verifies:
///   - compiledSwiGLU produces numerically equivalent results to unfused SwiGLU
///   - compiledAdamWStep produces numerically equivalent results to unfused AdamW step
///   - Fused ops handle various input shapes correctly
///
/// Requirements: R8.1, R8.2, R8.3
const std = @import("std");
const c = @import("mlx").c;
const ops = @import("mlx").ops;
const array_mod = @import("mlx").array;
const eval_mod = @import("mlx").eval;
const fused = @import("mlx").fused;
const random = @import("mlx").random;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

// ============================================================
// Helper: cosine similarity between two arrays
// ============================================================

fn cosineSimilarity(allocator: std.mem.Allocator, a: Array, b: Array) !f32 {
    try eval_mod.eval(a);
    try eval_mod.eval(b);

    const a_data = try a.dataSlice(f32);
    const b_data = try b.dataSlice(f32);

    if (a_data.len != b_data.len) return 0.0;
    if (a_data.len == 0) return 1.0;
    _ = allocator;

    var dot: f64 = 0.0;
    var norm_a: f64 = 0.0;
    var norm_b: f64 = 0.0;

    for (a_data, b_data) |av, bv| {
        const af: f64 = @floatCast(av);
        const bf: f64 = @floatCast(bv);
        dot += af * bf;
        norm_a += af * af;
        norm_b += bf * bf;
    }

    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom < 1e-12) return 1.0; // both near-zero
    return @floatCast(dot / denom);
}

// ============================================================
// SwiGLU Tests
// ============================================================

test "unfusedSwiGLU produces valid output shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const batch = 2;
    const seq_len = 4;
    const hidden_dim = 8;
    const intermediate_dim = 16;

    // Create random-ish inputs using normal distribution
    const x = try random.normal(ctx, &[_]i32{ batch, seq_len, hidden_dim }, .float32, 0.0, 1.0, null);
    defer x.deinit();
    const gate_w = try random.normal(ctx, &[_]i32{ intermediate_dim, hidden_dim }, .float32, 0.0, 0.1, null);
    defer gate_w.deinit();
    const up_w = try random.normal(ctx, &[_]i32{ intermediate_dim, hidden_dim }, .float32, 0.0, 0.1, null);
    defer up_w.deinit();
    const down_w = try random.normal(ctx, &[_]i32{ hidden_dim, intermediate_dim }, .float32, 0.0, 0.1, null);
    defer down_w.deinit();

    const output = try fused.unfusedSwiGLU(ctx, x, gate_w, up_w, down_w);
    defer output.deinit();

    try eval_mod.eval(output);
    const shape = output.shape();
    try std.testing.expectEqual(@as(usize, 3), shape.len);
    try std.testing.expectEqual(@as(i32, batch), shape[0]);
    try std.testing.expectEqual(@as(i32, seq_len), shape[1]);
    try std.testing.expectEqual(@as(i32, hidden_dim), shape[2]);
}

test "compiledSwiGLU matches unfused SwiGLU output" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const batch = 2;
    const seq_len = 3;
    const hidden_dim = 8;
    const intermediate_dim = 12;

    // Use seeded random key for reproducibility
    const rng_key = try random.key(42);
    defer rng_key.deinit();

    const x = try random.normal(ctx, &[_]i32{ batch, seq_len, hidden_dim }, .float32, 0.0, 1.0, null);
    defer x.deinit();
    const gate_w = try random.normal(ctx, &[_]i32{ intermediate_dim, hidden_dim }, .float32, 0.0, 0.1, null);
    defer gate_w.deinit();
    const up_w = try random.normal(ctx, &[_]i32{ intermediate_dim, hidden_dim }, .float32, 0.0, 0.1, null);
    defer up_w.deinit();
    const down_w = try random.normal(ctx, &[_]i32{ hidden_dim, intermediate_dim }, .float32, 0.0, 0.1, null);
    defer down_w.deinit();

    // Unfused result
    const unfused_out = try fused.unfusedSwiGLU(ctx, x, gate_w, up_w, down_w);
    defer unfused_out.deinit();

    // Compiled fused result
    const compiled = try fused.compiledSwiGLU(allocator);
    defer compiled.deinit();

    const inputs = [_]Array{ x, gate_w, up_w, down_w };
    const fused_result = try compiled.apply(&inputs, allocator);
    defer {
        for (fused_result) |arr| arr.deinit();
        allocator.free(fused_result);
    }

    try std.testing.expectEqual(@as(usize, 1), fused_result.len);

    const similarity = try cosineSimilarity(allocator, unfused_out, fused_result[0]);
    // Fused and unfused should be numerically equivalent (cosine sim ≥ 0.9999)
    try std.testing.expect(similarity >= 0.9999);
}

// ============================================================
// AdamW Step Tests
// ============================================================

test "unfusedAdamWStep produces valid outputs" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const param = try random.normal(ctx, &[_]i32{ 4, 8 }, .float32, 0.0, 1.0, null);
    defer param.deinit();
    const grad = try random.normal(ctx, &[_]i32{ 4, 8 }, .float32, 0.0, 0.1, null);
    defer grad.deinit();

    const m_shape = [_]i32{ 4, 8 };
    const m = try array_mod.zeros(allocator, &m_shape, .float32);
    defer m.deinit();
    const v = try array_mod.zeros(allocator, &m_shape, .float32);
    defer v.deinit();

    const result = try fused.unfusedAdamWStep(
        ctx,
        param,
        grad,
        m,
        v,
        0.001, // lr
        0.9, // beta1
        0.999, // beta2
        1e-8, // eps
        0.01, // weight_decay
        0.1, // bias_correction1 (1 - beta1^1)
        0.001, // bias_correction2 (1 - beta2^1)
    );
    defer result.param.deinit();
    defer result.m.deinit();
    defer result.v.deinit();

    try eval_mod.eval(result.param);
    try eval_mod.eval(result.m);
    try eval_mod.eval(result.v);

    // Check shapes match
    const p_shape = result.param.shape();
    try std.testing.expectEqual(@as(i32, 4), p_shape[0]);
    try std.testing.expectEqual(@as(i32, 8), p_shape[1]);
}

test "compiledAdamWStep matches unfused AdamW step" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const param = try random.normal(ctx, &[_]i32{ 4, 8 }, .float32, 0.0, 1.0, null);
    defer param.deinit();
    const grad = try random.normal(ctx, &[_]i32{ 4, 8 }, .float32, 0.0, 0.1, null);
    defer grad.deinit();

    const m_shape = [_]i32{ 4, 8 };
    const m = try array_mod.zeros(allocator, &m_shape, .float32);
    defer m.deinit();
    const v = try array_mod.zeros(allocator, &m_shape, .float32);
    defer v.deinit();

    const lr: f32 = 0.001;
    const beta1: f32 = 0.9;
    const beta2: f32 = 0.999;
    const eps_val: f32 = 1e-8;
    const weight_decay: f32 = 0.01;
    const bias_corr1: f32 = 1.0 - 0.9; // 1 - beta1^1
    const bias_corr2: f32 = 1.0 - 0.999; // 1 - beta2^1

    // Unfused result
    const unfused_result = try fused.unfusedAdamWStep(
        ctx,
        param,
        grad,
        m,
        v,
        lr,
        beta1,
        beta2,
        eps_val,
        weight_decay,
        bias_corr1,
        bias_corr2,
    );
    defer unfused_result.param.deinit();
    defer unfused_result.m.deinit();
    defer unfused_result.v.deinit();

    // Compiled fused result
    const compiled = try fused.compiledAdamWStep(allocator);
    defer compiled.deinit();

    const sc_lr = try ops.scalarF32(ctx, lr);
    defer sc_lr.deinit();
    const sc_beta1 = try ops.scalarF32(ctx, beta1);
    defer sc_beta1.deinit();
    const sc_beta2 = try ops.scalarF32(ctx, beta2);
    defer sc_beta2.deinit();
    const sc_eps = try ops.scalarF32(ctx, eps_val);
    defer sc_eps.deinit();
    const sc_wd = try ops.scalarF32(ctx, weight_decay);
    defer sc_wd.deinit();
    const sc_bc1 = try ops.scalarF32(ctx, bias_corr1);
    defer sc_bc1.deinit();
    const sc_bc2 = try ops.scalarF32(ctx, bias_corr2);
    defer sc_bc2.deinit();

    const inputs = [_]Array{ param, grad, m, v, sc_lr, sc_beta1, sc_beta2, sc_eps, sc_wd, sc_bc1, sc_bc2 };
    const fused_result = try compiled.apply(&inputs, allocator);
    defer {
        for (fused_result) |arr| arr.deinit();
        allocator.free(fused_result);
    }

    try std.testing.expectEqual(@as(usize, 3), fused_result.len);

    // Compare param, m, v
    const sim_param = try cosineSimilarity(allocator, unfused_result.param, fused_result[0]);
    const sim_m = try cosineSimilarity(allocator, unfused_result.m, fused_result[1]);
    const sim_v = try cosineSimilarity(allocator, unfused_result.v, fused_result[2]);

    try std.testing.expect(sim_param >= 0.9999);
    try std.testing.expect(sim_m >= 0.9999);
    try std.testing.expect(sim_v >= 0.9999);
}

// ============================================================
// Property-Based Test: Fused Operation Numerical Equivalence
// (Property 6)
//
// Feature: production-deployment, Property 6: Fused Operation
// Numerical Equivalence
//
// For any valid input tensors, a compiled fused operation
// (SwiGLU MLP, AdamW step) SHALL produce output with cosine
// similarity ≥ 0.9999 compared to the unfused implementation.
//
// **Validates: Requirements R8.1, R8.2, R8.3**
// ============================================================

test "Property 6: Fused SwiGLU numerical equivalence across random shapes (100 iterations)" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Compile once, reuse across iterations
    const compiled = try fused.compiledSwiGLU(allocator);
    defer compiled.deinit();

    var prng = std.Random.DefaultPrng.init(0xDEAD_BEEF);
    const rand = prng.random();

    const num_iterations: usize = 100;
    var min_similarity: f32 = 1.0;

    for (0..num_iterations) |_| {
        // Generate random shapes within reasonable bounds
        // batch: 1-4, seq_len: 1-8, hidden_dim: 4-32 (multiples of 4), intermediate_dim: 4-64 (multiples of 4)
        const batch: i32 = @intCast(rand.intRangeAtMost(u32, 1, 4));
        const seq_len: i32 = @intCast(rand.intRangeAtMost(u32, 1, 8));
        const hidden_dim: i32 = @intCast(rand.intRangeAtMost(u32, 1, 8) * 4);
        const intermediate_dim: i32 = @intCast(rand.intRangeAtMost(u32, 1, 16) * 4);

        // Create random input tensors
        const x = try random.normal(ctx, &[_]i32{ batch, seq_len, hidden_dim }, .float32, 0.0, 1.0, null);
        defer x.deinit();
        const gate_w = try random.normal(ctx, &[_]i32{ intermediate_dim, hidden_dim }, .float32, 0.0, 0.1, null);
        defer gate_w.deinit();
        const up_w = try random.normal(ctx, &[_]i32{ intermediate_dim, hidden_dim }, .float32, 0.0, 0.1, null);
        defer up_w.deinit();
        const down_w = try random.normal(ctx, &[_]i32{ hidden_dim, intermediate_dim }, .float32, 0.0, 0.1, null);
        defer down_w.deinit();

        // Unfused reference
        const unfused_out = try fused.unfusedSwiGLU(ctx, x, gate_w, up_w, down_w);
        defer unfused_out.deinit();

        // Compiled fused result
        const inputs = [_]Array{ x, gate_w, up_w, down_w };
        const fused_result = try compiled.apply(&inputs, allocator);
        defer {
            for (fused_result) |arr| arr.deinit();
            allocator.free(fused_result);
        }

        try std.testing.expectEqual(@as(usize, 1), fused_result.len);

        const similarity = try cosineSimilarity(allocator, unfused_out, fused_result[0]);
        if (similarity < min_similarity) min_similarity = similarity;

        // Property: cosine similarity ≥ 0.9999
        try std.testing.expect(similarity >= 0.9999);
    }

    // Log minimum similarity observed across all iterations
    std.debug.print("\n[Property 6 - SwiGLU] {d} iterations completed. Min cosine similarity: {d:.6}\n", .{ num_iterations, min_similarity });
}

test "Property 6: Fused AdamW numerical equivalence across random shapes (100 iterations)" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Compile once, reuse across iterations
    const compiled = try fused.compiledAdamWStep(allocator);
    defer compiled.deinit();

    var prng = std.Random.DefaultPrng.init(0xCAFE_BABE);
    const rand = prng.random();

    const num_iterations: usize = 100;
    var min_sim_param: f32 = 1.0;
    var min_sim_m: f32 = 1.0;
    var min_sim_v: f32 = 1.0;

    for (0..num_iterations) |_| {
        // Generate random 2D parameter shapes: rows 1-16, cols 4-32 (multiples of 4)
        const rows: i32 = @intCast(rand.intRangeAtMost(u32, 1, 16));
        const cols: i32 = @intCast(rand.intRangeAtMost(u32, 1, 8) * 4);
        const param_shape = [_]i32{ rows, cols };

        // Random hyperparameters within valid ranges
        const lr: f32 = @as(f32, @floatFromInt(rand.intRangeAtMost(u32, 1, 100))) * 0.0001; // 0.0001 to 0.01
        const beta1: f32 = 0.85 + @as(f32, @floatFromInt(rand.intRangeAtMost(u32, 0, 14))) * 0.01; // 0.85 to 0.99
        const beta2: f32 = 0.99 + @as(f32, @floatFromInt(rand.intRangeAtMost(u32, 0, 9))) * 0.001; // 0.99 to 0.999
        const eps_val: f32 = 1e-8;
        const weight_decay: f32 = @as(f32, @floatFromInt(rand.intRangeAtMost(u32, 0, 10))) * 0.01; // 0.0 to 0.1
        const bias_corr1: f32 = 1.0 - beta1; // simplified: step=1
        const bias_corr2: f32 = 1.0 - beta2; // simplified: step=1

        // Create random tensors
        const param = try random.normal(ctx, &param_shape, .float32, 0.0, 1.0, null);
        defer param.deinit();
        const grad = try random.normal(ctx, &param_shape, .float32, 0.0, 0.1, null);
        defer grad.deinit();
        const m_arr = try array_mod.zeros(allocator, &param_shape, .float32);
        defer m_arr.deinit();
        const v_arr = try array_mod.zeros(allocator, &param_shape, .float32);
        defer v_arr.deinit();

        // Unfused reference
        const unfused_result = try fused.unfusedAdamWStep(
            ctx,
            param,
            grad,
            m_arr,
            v_arr,
            lr,
            beta1,
            beta2,
            eps_val,
            weight_decay,
            bias_corr1,
            bias_corr2,
        );
        defer unfused_result.param.deinit();
        defer unfused_result.m.deinit();
        defer unfused_result.v.deinit();

        // Compiled fused result — pass hyperparams as scalar arrays
        const sc_lr = try ops.scalarF32(ctx, lr);
        defer sc_lr.deinit();
        const sc_beta1 = try ops.scalarF32(ctx, beta1);
        defer sc_beta1.deinit();
        const sc_beta2 = try ops.scalarF32(ctx, beta2);
        defer sc_beta2.deinit();
        const sc_eps = try ops.scalarF32(ctx, eps_val);
        defer sc_eps.deinit();
        const sc_wd = try ops.scalarF32(ctx, weight_decay);
        defer sc_wd.deinit();
        const sc_bc1 = try ops.scalarF32(ctx, bias_corr1);
        defer sc_bc1.deinit();
        const sc_bc2 = try ops.scalarF32(ctx, bias_corr2);
        defer sc_bc2.deinit();

        const inputs = [_]Array{ param, grad, m_arr, v_arr, sc_lr, sc_beta1, sc_beta2, sc_eps, sc_wd, sc_bc1, sc_bc2 };
        const fused_result = try compiled.apply(&inputs, allocator);
        defer {
            for (fused_result) |arr| arr.deinit();
            allocator.free(fused_result);
        }

        try std.testing.expectEqual(@as(usize, 3), fused_result.len);

        // Compare param, m, v
        const sim_param = try cosineSimilarity(allocator, unfused_result.param, fused_result[0]);
        const sim_m = try cosineSimilarity(allocator, unfused_result.m, fused_result[1]);
        const sim_v = try cosineSimilarity(allocator, unfused_result.v, fused_result[2]);

        if (sim_param < min_sim_param) min_sim_param = sim_param;
        if (sim_m < min_sim_m) min_sim_m = sim_m;
        if (sim_v < min_sim_v) min_sim_v = sim_v;

        // Property: cosine similarity ≥ 0.9999 for all outputs
        try std.testing.expect(sim_param >= 0.9999);
        try std.testing.expect(sim_m >= 0.9999);
        try std.testing.expect(sim_v >= 0.9999);
    }

    // Log minimum similarities observed
    std.debug.print("\n[Property 6 - AdamW] {d} iterations completed. Min cosine similarity — param: {d:.6}, m: {d:.6}, v: {d:.6}\n", .{ num_iterations, min_sim_param, min_sim_m, min_sim_v });
}
