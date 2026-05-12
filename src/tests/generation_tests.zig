/// Tests for the three-layer generation API (src/generation.zig).
///
/// Uses a mock model that returns deterministic logits to verify:
///   - generateStep produces a single token from a forward pass
///   - streamGenerate delivers tokens via callback
///   - generate returns the full collected sequence
///   - streamGenerate and generate produce identical output (Property 3)
///   - stop tokens halt generation early
///   - output length ≤ max_tokens
const std = @import("std");
const c = @import("mlx").c;
const ops = @import("mlx").ops;
const array_mod = @import("mlx").array;
const generation = @import("../generation.zig");
const kvcache = @import("../kvcache.zig");
const sampling_mod = @import("../sampling.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const KVCacheStrategy = kvcache.KVCacheStrategy;

// ============================================================
// Mock model: returns logits where token `call_count % vocab_size`
// has the highest score, producing a deterministic sequence.
// ============================================================

const MockModel = struct {
    call_count: usize = 0,
    vocab_size: usize = 32,

    fn forwardFn(ctx_ptr: *anyopaque, input: Array, mask: ?Array, caches: ?[]KVCacheStrategy) anyerror!Array {
        _ = mask;
        _ = caches;
        const self: *MockModel = @ptrCast(@alignCast(ctx_ptr));

        const input_shape = input.shape();
        const seq_len = @as(usize, @intCast(input_shape[1]));
        const vocab = self.vocab_size;

        // Build logits: [1, seq_len, vocab_size]
        // For the last position, make token (call_count % vocab) have the highest logit
        const total = seq_len * vocab;
        var data = try std.heap.page_allocator.alloc(f32, total);
        defer std.heap.page_allocator.free(data);

        // Fill with small values
        @memset(data, -10.0);

        // Set the "winning" token at the last position
        const target_token = self.call_count % vocab;
        const last_pos_offset = (seq_len - 1) * vocab;
        data[last_pos_offset + target_token] = 100.0;

        self.call_count += 1;

        return Array.fromData(
            std.heap.page_allocator,
            f32,
            data,
            &[_]i32{ 1, @intCast(seq_len), @intCast(vocab) },
        );
    }

    fn deinitFn(_: *anyopaque, _: std.mem.Allocator) void {}

    fn asVTable(self: *MockModel) generation.ModelVTable {
        return .{
            .forward = @ptrCast(&forwardFn),
            .deinit = @ptrCast(&deinitFn),
            .config = .{
                .num_layers = 1,
                .num_kv_heads = 1,
                .head_dim = 8,
                .vocab_size = self.vocab_size,
                .hidden_size = 8,
            },
            .ptr = @ptrCast(self),
        };
    }
};

// ============================================================
// Stub KV cache (no-op, since mock model ignores caches)
// ============================================================

const StubKVCache = struct {
    fn updateAndFetch(_: *anyopaque, keys: Array, values: Array, _: c.c.mlx_stream) anyerror!kvcache.KVSlice {
        return .{ .keys = keys, .values = values };
    }
    fn currentLen(_: *anyopaque) usize {
        return 0;
    }
    fn reset(_: *anyopaque) void {}
    fn deinitFn(_: *anyopaque, _: std.mem.Allocator) void {}

    const vtable = kvcache.interface.VTable{
        .updateAndFetch = @ptrCast(&updateAndFetch),
        .currentLen = @ptrCast(&currentLen),
        .reset = @ptrCast(&reset),
        .filter = null,
        .rollback = null,
        .extend = null,
        .deinit = @ptrCast(&deinitFn),
    };

    fn asStrategy(self: *StubKVCache) KVCacheStrategy {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &vtable,
        };
    }
};

// ============================================================
// Tests
// ============================================================

test "generateStep returns a single token" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    var mock = MockModel{};
    const vtable = mock.asVTable();

    var stub_cache = StubKVCache{};
    var caches = [_]KVCacheStrategy{stub_cache.asStrategy()};

    var sampler = sampling_mod.SamplerConfig.init(42);
    sampler.temperature = 0.0; // greedy

    // Create a prompt: [1, 3]
    const prompt = try Array.fromData(allocator, u32, &[_]u32{ 1, 2, 3 }, &[_]i32{ 1, 3 });
    defer prompt.deinit();

    const result = try generation.generateStep(vtable, prompt, &caches, &sampler, ctx);

    // Mock model call_count was 0, so target token = 0 % 32 = 0
    try std.testing.expectEqual(@as(u32, 0), result.token);
}

test "generate returns sequence with length ≤ max_tokens" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    var mock = MockModel{};
    const vtable = mock.asVTable();

    var stub_cache = StubKVCache{};
    var caches = [_]KVCacheStrategy{stub_cache.asStrategy()};

    const config = generation.GenerateConfig{
        .max_tokens = 5,
        .temperature = 0.0,
        .seed = 42,
    };

    const result = try generation.generate(vtable, &[_]u32{ 10, 20 }, config, &caches, ctx);
    defer allocator.free(result);

    try std.testing.expect(result.len <= 5);
    try std.testing.expect(result.len > 0);
}

test "generate stops on stop token" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Mock produces tokens: 0, 1, 2, 3, 4, ...
    // If stop_tokens contains 2, generation should stop at or after producing token 2
    var mock = MockModel{};
    const vtable = mock.asVTable();

    var stub_cache = StubKVCache{};
    var caches = [_]KVCacheStrategy{stub_cache.asStrategy()};

    const stop_tokens = [_]u32{2};
    const config = generation.GenerateConfig{
        .max_tokens = 10,
        .temperature = 0.0,
        .seed = 42,
        .stop_tokens = &stop_tokens,
    };

    const result = try generation.generate(vtable, &[_]u32{10}, config, &caches, ctx);
    defer allocator.free(result);

    // Should have stopped when token 2 was generated (tokens: 0, 1, 2)
    try std.testing.expectEqual(@as(usize, 3), result.len);
    try std.testing.expectEqual(@as(u32, 2), result[result.len - 1]);
}

test "streamGenerate delivers tokens via callback" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    var mock = MockModel{};
    const vtable = mock.asVTable();

    var stub_cache = StubKVCache{};
    var caches = [_]KVCacheStrategy{stub_cache.asStrategy()};

    const config = generation.GenerateConfig{
        .max_tokens = 4,
        .temperature = 0.0,
        .seed = 42,
    };

    // Use a global to collect callback results (Zig test limitation: no closures)
    const S = struct {
        var tokens: [16]u32 = undefined;
        var done_flags: [16]bool = undefined;
        var count: usize = 0;

        fn callback(token: u32, is_done: bool) void {
            tokens[count] = token;
            done_flags[count] = is_done;
            count += 1;
        }
    };
    S.count = 0;

    try generation.streamGenerate(vtable, &[_]u32{10}, config, &caches, ctx, &S.callback);

    // Should have received exactly max_tokens callbacks
    try std.testing.expectEqual(@as(usize, 4), S.count);
    // Last callback should have is_done = true
    try std.testing.expect(S.done_flags[S.count - 1]);
    // All non-last callbacks should have is_done = false
    for (0..S.count - 1) |i| {
        try std.testing.expect(!S.done_flags[i]);
    }
}

test "streamGenerate and generate produce identical tokens (Property 3)" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    const config = generation.GenerateConfig{
        .max_tokens = 6,
        .temperature = 0.0,
        .seed = 123,
    };

    const prompt = [_]u32{ 5, 10, 15 };

    // Run generate
    var mock1 = MockModel{};
    const vtable1 = mock1.asVTable();
    var stub1 = StubKVCache{};
    var caches1 = [_]KVCacheStrategy{stub1.asStrategy()};

    const gen_result = try generation.generate(vtable1, &prompt, config, &caches1, ctx);
    defer allocator.free(gen_result);

    // Run streamGenerate with a fresh mock (same initial state)
    var mock2 = MockModel{};
    const vtable2 = mock2.asVTable();
    var stub2 = StubKVCache{};
    var caches2 = [_]KVCacheStrategy{stub2.asStrategy()};

    const S = struct {
        var tokens: [64]u32 = undefined;
        var count: usize = 0;

        fn callback(token: u32, _: bool) void {
            tokens[count] = token;
            count += 1;
        }
    };
    S.count = 0;

    try generation.streamGenerate(vtable2, &prompt, config, &caches2, ctx, &S.callback);

    // Both should produce the same tokens
    try std.testing.expectEqual(gen_result.len, S.count);
    for (gen_result, 0..) |tok, i| {
        try std.testing.expectEqual(tok, S.tokens[i]);
    }
}

test "generate with max_tokens=1 returns exactly one token" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    var mock = MockModel{};
    const vtable = mock.asVTable();

    var stub_cache = StubKVCache{};
    var caches = [_]KVCacheStrategy{stub_cache.asStrategy()};

    const config = generation.GenerateConfig{
        .max_tokens = 1,
        .temperature = 0.0,
        .seed = 42,
    };

    const result = try generation.generate(vtable, &[_]u32{1}, config, &caches, ctx);
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 1), result.len);
}

test "GenerateConfig has correct defaults" {
    const config = generation.GenerateConfig{};
    try std.testing.expectEqual(@as(usize, 256), config.max_tokens);
    try std.testing.expectEqual(@as(f32, 0.8), config.temperature);
    try std.testing.expectEqual(@as(usize, 50), config.top_k);
    try std.testing.expectEqual(@as(f32, 1.0), config.top_p);
    try std.testing.expectEqual(@as(usize, 0), config.stop_tokens.len);
    try std.testing.expectEqual(@as(u64, 0), config.seed);
}

// ============================================================
// Property-Based Test: Generation API Consistency (Property 3)
//
// **Validates: Requirements R5.1, R5.2, R5.3, R5.4**
//
// For any model, prompt, and generation config, the sequence of
// tokens produced by `streamGenerate` (collected via callback)
// SHALL be identical to the sequence returned by `generate`,
// and both SHALL have length ≤ `max_tokens`.
//
// Runs 100 iterations with randomly generated prompts and configs.
// ============================================================

/// Callback collector for streamGenerate — uses static storage since
/// Zig function pointers cannot capture state.
const StreamCollector = struct {
    var tokens: [512]u32 = undefined;
    var count: usize = 0;

    fn reset() void {
        count = 0;
    }

    fn callback(token: u32, _: bool) void {
        if (count < 512) {
            tokens[count] = token;
            count += 1;
        }
    }

    fn slice() []const u32 {
        return tokens[0..count];
    }
};

test "Property 3: Generation API Consistency — streamGenerate == generate, length ≤ max_tokens (100 iterations)" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    var prng = std.Random.DefaultPrng.init(0xDEAD_BEEF);
    const rand = prng.random();

    const num_iterations: usize = 100;

    for (0..num_iterations) |iteration| {
        // --- Generate random config ---
        const max_tokens = rand.intRangeAtMost(usize, 1, 32);
        const vocab_size = rand.intRangeAtMost(usize, 4, 64);
        // Use temperature 0 (greedy) for deterministic comparison
        // The property is about API consistency, not sampling randomness
        const seed = rand.int(u64);

        // Randomly decide whether to include stop tokens
        const use_stop_tokens = rand.boolean();
        var stop_token_buf: [1]u32 = undefined;
        var stop_tokens_slice: []const u32 = &.{};
        if (use_stop_tokens) {
            // Pick a stop token that's within vocab range
            stop_token_buf[0] = rand.intRangeLessThan(u32, 0, @intCast(vocab_size));
            stop_tokens_slice = &stop_token_buf;
        }

        const config = generation.GenerateConfig{
            .max_tokens = max_tokens,
            .temperature = 0.0, // greedy for determinism
            .top_k = 50,
            .top_p = 1.0,
            .stop_tokens = stop_tokens_slice,
            .seed = seed,
        };

        // --- Generate random prompt ---
        const prompt_len = rand.intRangeAtMost(usize, 1, 16);
        var prompt_buf: [16]u32 = undefined;
        for (0..prompt_len) |i| {
            prompt_buf[i] = rand.intRangeLessThan(u32, 0, @intCast(vocab_size));
        }
        const prompt = prompt_buf[0..prompt_len];

        // --- Run generate ---
        var mock_gen = MockModel{ .vocab_size = vocab_size };
        const vtable_gen = mock_gen.asVTable();
        var stub_gen = StubKVCache{};
        var caches_gen = [_]KVCacheStrategy{stub_gen.asStrategy()};

        const gen_result = try generation.generate(vtable_gen, prompt, config, &caches_gen, ctx);
        defer allocator.free(gen_result);

        // --- Run streamGenerate with fresh mock (same initial state) ---
        var mock_stream = MockModel{ .vocab_size = vocab_size };
        const vtable_stream = mock_stream.asVTable();
        var stub_stream = StubKVCache{};
        var caches_stream = [_]KVCacheStrategy{stub_stream.asStrategy()};

        StreamCollector.reset();
        try generation.streamGenerate(vtable_stream, prompt, config, &caches_stream, ctx, &StreamCollector.callback);
        const stream_result = StreamCollector.slice();

        // --- Property: output length ≤ max_tokens ---
        if (gen_result.len > max_tokens) {
            std.debug.print(
                "FAIL iteration {}: generate returned {} tokens, max_tokens={}\n",
                .{ iteration, gen_result.len, max_tokens },
            );
            return error.TestUnexpectedResult;
        }
        if (stream_result.len > max_tokens) {
            std.debug.print(
                "FAIL iteration {}: streamGenerate produced {} tokens, max_tokens={}\n",
                .{ iteration, stream_result.len, max_tokens },
            );
            return error.TestUnexpectedResult;
        }

        // --- Property: generate and streamGenerate produce identical tokens ---
        if (gen_result.len != stream_result.len) {
            std.debug.print(
                "FAIL iteration {}: length mismatch generate={} stream={} (max_tokens={}, vocab={}, prompt_len={}, stop={})\n",
                .{ iteration, gen_result.len, stream_result.len, max_tokens, vocab_size, prompt_len, use_stop_tokens },
            );
            return error.TestUnexpectedResult;
        }

        for (gen_result, 0..) |tok, i| {
            if (tok != stream_result[i]) {
                std.debug.print(
                    "FAIL iteration {}: token mismatch at index {} generate={} stream={}\n",
                    .{ iteration, i, tok, stream_result[i] },
                );
                return error.TestUnexpectedResult;
            }
        }
    }
}
