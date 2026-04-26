const std = @import("std");
const memory = @import("../memory.zig");

const testing = std.testing;

// ============================================================
// Property-Based Test
// (Property 21)
//
// Feature: production-deployment, Property 21: Auto max_kv_size Formula
//
// For any model configuration (num_layers, num_kv_heads, head_dim,
// kv_bits) and device memory value, `autoMaxKvSizeWithTotalRam` SHALL
// return:
//   (total_RAM - model_bytes - safety_margin) /
//   (2 * num_kv_heads * head_dim * (kv_bits / 8) * num_layers)
//
// **Validates: Requirements R24.1, R24.2**
// ============================================================

/// Valid kv_bits values that produce non-zero bytes_per_element (kv_bits / 8 > 0).
const valid_kv_bits = [_]u8{ 8, 16 };

test "Property 21: Auto max_kv_size formula matches independent computation (100 iterations)" {
    const safety_margin: usize = 512 * 1024 * 1024; // 512 MB default

    var prng = std.Random.DefaultPrng.init(2024);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // --- Generate random valid config ---
        // total_ram: 4 GB to 128 GB
        const total_ram: usize = rand.intRangeAtMost(usize, 4 * 1024 * 1024 * 1024, 128 * 1024 * 1024 * 1024);
        // model_bytes: 0 to 32 GB
        const model_bytes: usize = rand.intRangeAtMost(usize, 0, 32 * 1024 * 1024 * 1024);
        // num_layers: 1 to 128
        const num_layers: usize = rand.intRangeAtMost(usize, 1, 128);
        // num_kv_heads: 1 to 64
        const num_kv_heads: usize = rand.intRangeAtMost(usize, 1, 64);
        // head_dim: 1 to 256
        const head_dim: usize = rand.intRangeAtMost(usize, 1, 256);
        // kv_bits: pick from valid values (8 or 16)
        const kv_bits = valid_kv_bits[rand.intRangeAtMost(usize, 0, valid_kv_bits.len - 1)];

        // --- Compute expected value independently ---
        const overhead = model_bytes +| safety_margin; // saturating add
        const expected: usize = if (total_ram <= overhead) blk: {
            break :blk 0;
        } else blk: {
            const available = total_ram - overhead;
            const bytes_per_element: usize = @as(usize, kv_bits) / 8;
            if (bytes_per_element == 0) break :blk 0;
            const bytes_per_token = 2 *| (num_kv_heads *| (head_dim *| (bytes_per_element *| num_layers)));
            if (bytes_per_token == 0) break :blk 0;
            break :blk available / bytes_per_token;
        };

        // --- Call the function under test ---
        const actual = memory.autoMaxKvSizeWithTotalRam(
            total_ram,
            model_bytes,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_bits,
        );

        // --- Assert equality ---
        try testing.expectEqual(expected, actual);
    }
}

test "Property 21: Auto max_kv_size returns 0 when overhead exceeds total_ram (100 iterations)" {
    const safety_margin: usize = 512 * 1024 * 1024;

    var prng = std.Random.DefaultPrng.init(9999);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // Generate total_ram that is small enough to be exceeded by model + safety.
        const total_ram: usize = rand.intRangeAtMost(usize, 0, 2 * 1024 * 1024 * 1024);
        // model_bytes large enough to exceed total_ram - safety_margin.
        const model_bytes: usize = rand.intRangeAtMost(usize, total_ram, total_ram +| (4 * 1024 * 1024 * 1024));

        const num_layers: usize = rand.intRangeAtMost(usize, 1, 64);
        const num_kv_heads: usize = rand.intRangeAtMost(usize, 1, 32);
        const head_dim: usize = rand.intRangeAtMost(usize, 1, 256);
        const kv_bits = valid_kv_bits[rand.intRangeAtMost(usize, 0, valid_kv_bits.len - 1)];

        _ = safety_margin;

        const actual = memory.autoMaxKvSizeWithTotalRam(
            total_ram,
            model_bytes,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_bits,
        );

        try testing.expectEqual(@as(usize, 0), actual);
    }
}

test "Property 21: Auto max_kv_size with kv_bits=4 yields ~4x tokens vs kv_bits=16" {
    // kv_bits=4 should yield approximately 4x the token capacity of kv_bits=16,
    // since 4-bit quantization uses 1/4 the bits per element.
    // Due to integer floor division, the ratio may not be exactly 4x.
    var prng = std.Random.DefaultPrng.init(1234);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        const total_ram: usize = rand.intRangeAtMost(usize, 8 * 1024 * 1024 * 1024, 128 * 1024 * 1024 * 1024);
        const model_bytes: usize = rand.intRangeAtMost(usize, 0, 4 * 1024 * 1024 * 1024);
        const num_layers: usize = rand.intRangeAtMost(usize, 1, 128);
        const num_kv_heads: usize = rand.intRangeAtMost(usize, 1, 64);
        const head_dim: usize = rand.intRangeAtMost(usize, 1, 256);

        const result_16 = memory.autoMaxKvSizeWithTotalRam(
            total_ram,
            model_bytes,
            num_layers,
            num_kv_heads,
            head_dim,
            16,
        );

        const result_4 = memory.autoMaxKvSizeWithTotalRam(
            total_ram,
            model_bytes,
            num_layers,
            num_kv_heads,
            head_dim,
            4,
        );

        // 4-bit should be non-zero when 16-bit is non-zero.
        if (result_16 > 0) {
            try testing.expect(result_4 > 0);
            // The ratio should be close to 4x. Due to floor division rounding,
            // result_4 may be slightly above or below 4 * result_16.
            // We allow a tolerance of ±1 token per unit of result_16.
            const lower = result_16 * 3;
            const upper = result_16 * 4 + 4; // small tolerance for rounding
            try testing.expect(result_4 >= lower);
            try testing.expect(result_4 <= upper);
        }
    }
}
