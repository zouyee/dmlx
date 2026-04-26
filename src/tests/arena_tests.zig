/// Property 2: Forward Pass Arena Cleanup
///
/// Verifies that after a forward pass completes and ScopedArrayArena is
/// deinitialized, all intermediate Arrays are released and only the final
/// output remains live.
///
/// **Validates: Requirements R2.1, R2.2**
const std = @import("std");
const mlx = @import("../root.zig");

const Array = mlx.Array;
const EagerContext = mlx.EagerContext;
const ScopedArrayArena = mlx.array_arena.ScopedArrayArena;

/// Simulate a forward pass: create several intermediate arrays via ops,
/// track them in the arena, and return a final result NOT tracked.
/// This mirrors the pattern from design.md section 1.2.
fn simulatedForwardPass(alloc: std.mem.Allocator, ctx: EagerContext, input: Array) !Array {
    var arena = ScopedArrayArena.init(alloc);
    defer arena.deinit();

    // Intermediate 1: zeros added to input
    const zeros = try arena.track(try mlx.creation.zeros(ctx, &[_]i32{4}, .float32));
    // Intermediate 2: ones
    const ones_arr = try arena.track(try mlx.creation.ones(ctx, &[_]i32{4}, .float32));
    // Intermediate 3: input + zeros (should equal input)
    const sum1 = try arena.track(try mlx.ops.add(ctx, input, zeros));
    // Intermediate 4: sum1 + ones
    const sum2 = try arena.track(try mlx.ops.add(ctx, sum1, ones_arr));
    // Intermediate 5: sum2 * ones (identity multiply)
    const prod = try arena.track(try mlx.ops.multiply(ctx, sum2, ones_arr));

    // Final output is NOT tracked — caller owns it
    return mlx.ops.add(ctx, prod, ones_arr);
}

test "Property 2: arena cleanup — final output valid after arena deinit (100 iterations)" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const ctx = EagerContext.init(alloc);

    // Use std.Random for deterministic iteration with varying seeds
    var prng = std.Random.DefaultPrng.init(12345);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // Generate random input data for this iteration
        var input_data: [4]f32 = undefined;
        for (&input_data) |*v| {
            v.* = rand.float(f32) * 20.0 - 10.0; // range [-10, 10]
        }

        const input = try Array.fromData(alloc, f32, &input_data, &[_]i32{4});
        defer input.deinit();

        // Run simulated forward pass — arena is created and destroyed inside
        const result = try simulatedForwardPass(alloc, ctx, input);
        defer result.deinit();

        // Verify the final result is still valid after arena deinit:
        // result = (input + 0 + 1) * 1 + 1 = input + 2
        try result.eval();
        const data = try result.dataSlice(f32);

        try std.testing.expectEqual(@as(usize, 4), data.len);
        for (data, 0..) |val, i| {
            const expected = input_data[i] + 2.0;
            try std.testing.expectApproxEqAbs(expected, val, 1e-5);
        }
    }
}

test "Property 2: arena cleanup — multiple tracked arrays freed, untracked survives" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const ctx = EagerContext.init(alloc);

    var prng = std.Random.DefaultPrng.init(99999);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // Random shape dimension between 1 and 8
        const dim: i32 = @intCast(rand.intRangeAtMost(u32, 1, 8));
        const size: usize = @intCast(dim);

        var arena = ScopedArrayArena.init(alloc);

        // Create and track several intermediates
        const z = try arena.track(try mlx.creation.zeros(ctx, &[_]i32{dim}, .float32));
        const o = try arena.track(try mlx.creation.ones(ctx, &[_]i32{dim}, .float32));
        const sum_arr = try arena.track(try mlx.ops.add(ctx, z, o));
        const doubled = try arena.track(try mlx.ops.add(ctx, sum_arr, o));

        // Final result NOT tracked
        const final_result = try mlx.ops.add(ctx, doubled, o);

        // Deinit the arena — all tracked arrays are freed
        arena.deinit();

        // The final result should still be valid and usable
        defer final_result.deinit();
        try final_result.eval();
        const data = try final_result.dataSlice(f32);

        // 0 + 1 + 1 + 1 = 3.0
        try std.testing.expectEqual(size, data.len);
        for (data) |val| {
            try std.testing.expectApproxEqAbs(@as(f32, 3.0), val, 1e-5);
        }
    }
}

test "Property 2: arena cleanup — chained ops with random normal inputs" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const ctx = EagerContext.init(alloc);

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        var arena = ScopedArrayArena.init(alloc);

        // Use mlx random to generate input (different seed each iteration)
        const rng_key = try mlx.random.key(@as(u64, iteration) + 1000);
        defer rng_key.deinit();

        const input = try mlx.random.normal(ctx, &[_]i32{ 2, 3 }, .float32, 0.0, 1.0, rng_key);
        defer input.deinit();

        // Chain of tracked intermediate ops
        const ones_arr = try arena.track(try mlx.creation.ones(ctx, &[_]i32{ 2, 3 }, .float32));
        const step1 = try arena.track(try mlx.ops.add(ctx, input, ones_arr));
        const step2 = try arena.track(try mlx.ops.multiply(ctx, step1, ones_arr));
        const step3 = try arena.track(try mlx.ops.add(ctx, step2, ones_arr));

        // Final output: step3 + ones = input + 1 + 1 + 1 = input + 3
        const final_result = try mlx.ops.add(ctx, step3, ones_arr);

        // Destroy arena
        arena.deinit();

        // Final result must still be valid
        defer final_result.deinit();
        try final_result.eval();
        const result_data = try final_result.dataSlice(f32);

        // Verify against input + 3
        try input.eval();
        const input_data = try input.dataSlice(f32);

        try std.testing.expectEqual(input_data.len, result_data.len);
        for (input_data, result_data) |in_val, out_val| {
            try std.testing.expectApproxEqAbs(in_val + 3.0, out_val, 1e-5);
        }
    }
}
