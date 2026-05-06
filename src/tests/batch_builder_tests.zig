const std = @import("std");
const batch_builder = @import("../batch_builder.zig");
const scheduler_mod = @import("../scheduler.zig");

const BatchResult = batch_builder.BatchResult;
const Request = scheduler_mod.Request;
const ScheduleResult = scheduler_mod.ScheduleResult;

const testing = std.testing;
const allocator = testing.allocator;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn makeRequest(id: u64, prompt: []const u32, max_tokens: usize) Request {
    return Request.init(allocator, id, prompt, max_tokens, &.{});
}

fn makePrefillRequest(id: u64, prompt: []const u32, max_tokens: usize) Request {
    var req = makeRequest(id, prompt, max_tokens);
    req.state = .prefilling;
    return req;
}

fn makeDecodeRequest(id: u64, prompt: []const u32, max_tokens: usize, generated: []const u32) !Request {
    var req = makeRequest(id, prompt, max_tokens);
    req.state = .decoding;
    req.prefill_offset = prompt.len;
    for (generated) |tok| {
        try req.generated_tokens.append(allocator, tok);
    }
    return req;
}

fn dummyCtx() @import("mlx").ops.EagerContext {
    return @import("mlx").ops.EagerContext.init(allocator);
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

test "batch_builder: empty schedule produces empty batch" {
    var ctx = dummyCtx();
    defer ctx.deinit();

    const schedule = ScheduleResult{
        .prefill_requests = &.{},
        .decode_requests = &.{},
        .blocks_needed = 0,
    };

    var result = try batch_builder.build(allocator, &schedule, ctx);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.total_tokens);
    try testing.expectEqual(@as(usize, 0), result.num_requests);
}

test "batch_builder: single prefill request" {
    var ctx = dummyCtx();
    defer ctx.deinit();

    const prompt = &[_]u32{ 10, 20, 30 };
    var req = makePrefillRequest(1, prompt, 10);
    defer req.deinit(allocator);

    var prefill_list = [_]*Request{&req};
    const schedule = ScheduleResult{
        .prefill_requests = &prefill_list,
        .decode_requests = &.{},
        .blocks_needed = 0,
    };

    var result = try batch_builder.build(allocator, &schedule, ctx);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.total_tokens);
    try testing.expectEqual(@as(usize, 1), result.num_requests);

    // Verify token ids.
    const tokens = try result.batched_tokens.dataSlice(u32);
    try testing.expectEqual(@as(u32, 10), tokens[0]);
    try testing.expectEqual(@as(u32, 20), tokens[1]);
    try testing.expectEqual(@as(u32, 30), tokens[2]);

    // Verify position ids: [0, 1, 2].
    const positions = try result.position_ids.dataSlice(i32);
    try testing.expectEqual(@as(i32, 0), positions[0]);
    try testing.expectEqual(@as(i32, 1), positions[1]);
    try testing.expectEqual(@as(i32, 2), positions[2]);

    // Verify attention mask: 3x3 causal within single request.
    const mask = try result.attention_mask.dataSlice(f32);
    // Row 0: can attend to col 0 only.
    try testing.expect(batch_builder.isAttending(mask, 3, 0, 0));
    try testing.expect(!batch_builder.isAttending(mask, 3, 0, 1));
    try testing.expect(!batch_builder.isAttending(mask, 3, 0, 2));
    // Row 1: can attend to cols 0, 1.
    try testing.expect(batch_builder.isAttending(mask, 3, 1, 0));
    try testing.expect(batch_builder.isAttending(mask, 3, 1, 1));
    try testing.expect(!batch_builder.isAttending(mask, 3, 1, 2));
    // Row 2: can attend to cols 0, 1, 2.
    try testing.expect(batch_builder.isAttending(mask, 3, 2, 0));
    try testing.expect(batch_builder.isAttending(mask, 3, 2, 1));
    try testing.expect(batch_builder.isAttending(mask, 3, 2, 2));
}

test "batch_builder: single decode request" {
    var ctx = dummyCtx();
    defer ctx.deinit();

    const prompt = &[_]u32{ 10, 20, 30 };
    var req = try makeDecodeRequest(1, prompt, 10, &[_]u32{42});
    defer req.deinit(allocator);

    var decode_list = [_]*Request{&req};
    const schedule = ScheduleResult{
        .prefill_requests = &.{},
        .decode_requests = &decode_list,
        .blocks_needed = 0,
    };

    var result = try batch_builder.build(allocator, &schedule, ctx);
    defer result.deinit();

    // Decode request contributes 1 token (last generated).
    try testing.expectEqual(@as(usize, 1), result.total_tokens);
    try testing.expectEqual(@as(usize, 1), result.num_requests);

    const tokens = try result.batched_tokens.dataSlice(u32);
    try testing.expectEqual(@as(u32, 42), tokens[0]);

    // Position should be seqLen - 1 = (3 + 1) - 1 = 3.
    const positions = try result.position_ids.dataSlice(i32);
    try testing.expectEqual(@as(i32, 3), positions[0]);
}

test "batch_builder: mixed prefill and decode — attention isolation" {
    var ctx = dummyCtx();
    defer ctx.deinit();

    // Prefill request: 2 prompt tokens.
    const prompt1 = &[_]u32{ 100, 200 };
    var req1 = makePrefillRequest(1, prompt1, 10);
    defer req1.deinit(allocator);

    // Decode request: 3 prompt tokens + 1 generated token → contributes 1 token.
    const prompt2 = &[_]u32{ 300, 400, 500 };
    var req2 = try makeDecodeRequest(2, prompt2, 10, &[_]u32{600});
    defer req2.deinit(allocator);

    var prefill_list = [_]*Request{&req1};
    var decode_list = [_]*Request{&req2};
    const schedule = ScheduleResult{
        .prefill_requests = &prefill_list,
        .decode_requests = &decode_list,
        .blocks_needed = 0,
    };

    var result = try batch_builder.build(allocator, &schedule, ctx);
    defer result.deinit();

    // Total: 2 (prefill) + 1 (decode) = 3 tokens.
    try testing.expectEqual(@as(usize, 3), result.total_tokens);
    try testing.expectEqual(@as(usize, 2), result.num_requests);

    const tokens = try result.batched_tokens.dataSlice(u32);
    try testing.expectEqual(@as(u32, 100), tokens[0]);
    try testing.expectEqual(@as(u32, 200), tokens[1]);
    try testing.expectEqual(@as(u32, 600), tokens[2]);

    // Attention isolation: request 1 tokens (0,1) should NOT attend to request 2 token (2).
    const mask = try result.attention_mask.dataSlice(f32);
    const n: usize = 3;

    // Request 1 tokens attend to each other (causal).
    try testing.expect(batch_builder.isAttending(mask, n, 0, 0));
    try testing.expect(batch_builder.isAttending(mask, n, 1, 0));
    try testing.expect(batch_builder.isAttending(mask, n, 1, 1));

    // Request 1 tokens do NOT attend to request 2 token.
    try testing.expect(!batch_builder.isAttending(mask, n, 0, 2));
    try testing.expect(!batch_builder.isAttending(mask, n, 1, 2));

    // Request 2 token does NOT attend to request 1 tokens.
    try testing.expect(!batch_builder.isAttending(mask, n, 2, 0));
    try testing.expect(!batch_builder.isAttending(mask, n, 2, 1));

    // Request 2 token attends to itself.
    try testing.expect(batch_builder.isAttending(mask, n, 2, 2));
}

test "batch_builder: multiple decode requests" {
    var ctx = dummyCtx();
    defer ctx.deinit();

    var req1 = try makeDecodeRequest(1, &[_]u32{ 1, 2 }, 10, &[_]u32{ 10, 20 });
    defer req1.deinit(allocator);
    var req2 = try makeDecodeRequest(2, &[_]u32{ 3, 4, 5 }, 10, &[_]u32{30});
    defer req2.deinit(allocator);

    var decode_list = [_]*Request{ &req1, &req2 };
    const schedule = ScheduleResult{
        .prefill_requests = &.{},
        .decode_requests = &decode_list,
        .blocks_needed = 0,
    };

    var result = try batch_builder.build(allocator, &schedule, ctx);
    defer result.deinit();

    // Each decode request contributes 1 token.
    try testing.expectEqual(@as(usize, 2), result.total_tokens);
    try testing.expectEqual(@as(usize, 2), result.num_requests);

    const tokens = try result.batched_tokens.dataSlice(u32);
    try testing.expectEqual(@as(u32, 20), tokens[0]); // last generated of req1
    try testing.expectEqual(@as(u32, 30), tokens[1]); // last generated of req2

    // Position ids: req1 seqLen=4, contributes 1 token → pos 3.
    //               req2 seqLen=4, contributes 1 token → pos 3.
    const positions = try result.position_ids.dataSlice(i32);
    try testing.expectEqual(@as(i32, 3), positions[0]);
    try testing.expectEqual(@as(i32, 3), positions[1]);

    // Attention: each token attends only to itself (single-token sequences).
    const mask = try result.attention_mask.dataSlice(f32);
    try testing.expect(batch_builder.isAttending(mask, 2, 0, 0));
    try testing.expect(!batch_builder.isAttending(mask, 2, 0, 1));
    try testing.expect(!batch_builder.isAttending(mask, 2, 1, 0));
    try testing.expect(batch_builder.isAttending(mask, 2, 1, 1));
}

test "batch_builder: new request joins without waiting (R13.3)" {
    var ctx = dummyCtx();
    defer ctx.deinit();

    // Simulate: req1 is already decoding, req2 is new (prefilling).
    var req1 = try makeDecodeRequest(1, &[_]u32{ 1, 2 }, 10, &[_]u32{10});
    defer req1.deinit(allocator);

    var req2 = makePrefillRequest(2, &[_]u32{ 3, 4 }, 10);
    defer req2.deinit(allocator);

    var prefill_list = [_]*Request{&req2};
    var decode_list = [_]*Request{&req1};
    const schedule = ScheduleResult{
        .prefill_requests = &prefill_list,
        .decode_requests = &decode_list,
        .blocks_needed = 0,
    };

    var result = try batch_builder.build(allocator, &schedule, ctx);
    defer result.deinit();

    // Both requests are batched together: 2 (prefill) + 1 (decode) = 3 tokens.
    try testing.expectEqual(@as(usize, 3), result.total_tokens);
    try testing.expectEqual(@as(usize, 2), result.num_requests);

    // Verify tokens: prefill first [3, 4], then decode [10].
    const tokens = try result.batched_tokens.dataSlice(u32);
    try testing.expectEqual(@as(u32, 3), tokens[0]);
    try testing.expectEqual(@as(u32, 4), tokens[1]);
    try testing.expectEqual(@as(u32, 10), tokens[2]);
}

test "batch_builder: tensor shapes are correct" {
    var ctx = dummyCtx();
    defer ctx.deinit();

    var req1 = makePrefillRequest(1, &[_]u32{ 1, 2, 3 }, 10);
    defer req1.deinit(allocator);
    var req2 = try makeDecodeRequest(2, &[_]u32{ 4, 5 }, 10, &[_]u32{6});
    defer req2.deinit(allocator);

    var prefill_list = [_]*Request{&req1};
    var decode_list = [_]*Request{&req2};
    const schedule = ScheduleResult{
        .prefill_requests = &prefill_list,
        .decode_requests = &decode_list,
        .blocks_needed = 0,
    };

    var result = try batch_builder.build(allocator, &schedule, ctx);
    defer result.deinit();

    // batched_tokens: [4]
    try testing.expectEqual(@as(usize, 1), result.batched_tokens.ndim());
    try testing.expectEqual(@as(i32, 4), result.batched_tokens.shape()[0]);

    // position_ids: [4]
    try testing.expectEqual(@as(usize, 1), result.position_ids.ndim());
    try testing.expectEqual(@as(i32, 4), result.position_ids.shape()[0]);

    // attention_mask: [4, 4]
    try testing.expectEqual(@as(usize, 2), result.attention_mask.ndim());
    try testing.expectEqual(@as(i32, 4), result.attention_mask.shape()[0]);
    try testing.expectEqual(@as(i32, 4), result.attention_mask.shape()[1]);
}

// ============================================================
// Property-Based Test
// (Property 11)
//
// Feature: production-deployment, Property 11: Continuous Batching
// Attention Isolation
//
// For any batch of N request sequences concatenated into a single
// tensor, the attention mask ensures each request's tokens attend
// only to tokens within the same request. The batched tensor length
// equals the sum of individual sequence lengths.
//
// **Validates: Requirements R13.1, R13.2**
// ============================================================

test "Property 11: Continuous Batching Attention Isolation — mask isolates requests and total_tokens == sum of seq lengths (100 iterations)" {
    const max_prefill = 4;
    const max_decode = 4;
    const max_prompt_len = 8;

    // Pre-allocate prompt token pools (stack-allocated).
    // Each prefill request gets a unique slice from this pool.
    var prefill_prompts: [max_prefill][max_prompt_len]u32 = undefined;
    for (0..max_prefill) |i| {
        for (0..max_prompt_len) |j| {
            prefill_prompts[i][j] = @intCast(100 * (i + 1) + j);
        }
    }
    var decode_prompts: [max_decode][max_prompt_len]u32 = undefined;
    for (0..max_decode) |i| {
        for (0..max_prompt_len) |j| {
            decode_prompts[i][j] = @intCast(1000 * (i + 1) + j);
        }
    }

    var prng = std.Random.DefaultPrng.init(0xDEAD_BEEF);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // --- Generate random request counts ---
        const num_prefill = rand.intRangeAtMost(usize, 0, max_prefill);
        const num_decode = rand.intRangeAtMost(usize, 0, max_decode);

        // Skip degenerate case where there are no requests at all.
        if (num_prefill == 0 and num_decode == 0) continue;

        var ctx = dummyCtx();
        defer ctx.deinit();

        // --- Create prefill requests with random prompt lengths ---
        var prefill_reqs: [max_prefill]Request = undefined;
        var prefill_lens: [max_prefill]usize = undefined;
        var prefill_ptrs: [max_prefill]*Request = undefined;

        for (0..num_prefill) |i| {
            const prompt_len = rand.intRangeAtMost(usize, 1, max_prompt_len);
            prefill_lens[i] = prompt_len;
            prefill_reqs[i] = makePrefillRequest(
                @intCast(i + 1),
                prefill_prompts[i][0..prompt_len],
                100,
            );
            prefill_ptrs[i] = &prefill_reqs[i];
        }

        // --- Create decode requests with random prompt lengths and 1+ generated tokens ---
        var decode_reqs: [max_decode]Request = undefined;
        var decode_ptrs: [max_decode]*Request = undefined;

        for (0..num_decode) |i| {
            const prompt_len = rand.intRangeAtMost(usize, 1, max_prompt_len);
            const num_generated = rand.intRangeAtMost(usize, 1, 4);

            decode_reqs[i] = makeRequest(
                @intCast(num_prefill + i + 1),
                decode_prompts[i][0..prompt_len],
                100,
            );
            decode_reqs[i].state = .decoding;
            decode_reqs[i].prefill_offset = prompt_len;

            // Add generated tokens.
            for (0..num_generated) |g| {
                try decode_reqs[i].generated_tokens.append(
                    allocator,
                    @intCast(2000 + i * 10 + g),
                );
            }
            decode_ptrs[i] = &decode_reqs[i];
        }

        // --- Build the schedule result ---
        const schedule = ScheduleResult{
            .prefill_requests = prefill_ptrs[0..num_prefill],
            .decode_requests = decode_ptrs[0..num_decode],
            .blocks_needed = 0,
        };

        // --- Build the batch ---
        var result = try batch_builder.build(allocator, &schedule, ctx);

        // --- Compute expected per-request sequence lengths ---
        // Prefill requests contribute their prompt length tokens.
        // Decode requests contribute exactly 1 token (last generated).
        var expected_total: usize = 0;
        var seq_lengths: [max_prefill + max_decode]usize = undefined;
        for (0..num_prefill) |i| {
            seq_lengths[i] = prefill_lens[i];
            expected_total += seq_lengths[i];
        }
        for (0..num_decode) |i| {
            seq_lengths[num_prefill + i] = 1; // decode always contributes 1 token
            expected_total += 1;
        }

        const total_reqs = num_prefill + num_decode;

        // --- Property 11a: total_tokens == sum of individual sequence lengths ---
        try testing.expectEqual(expected_total, result.total_tokens);

        // --- Property 11b: Attention mask isolates each request's tokens ---
        if (result.total_tokens > 0) {
            const mask = try result.attention_mask.dataSlice(f32);
            const n = result.total_tokens;

            // Verify mask dimensions.
            try testing.expectEqual(@as(usize, 2), result.attention_mask.ndim());
            try testing.expectEqual(@as(i32, @intCast(n)), result.attention_mask.shape()[0]);
            try testing.expectEqual(@as(i32, @intCast(n)), result.attention_mask.shape()[1]);

            // Walk through each pair of requests and verify isolation.
            var offset_i: usize = 0;
            for (0..total_reqs) |ri| {
                const len_i = seq_lengths[ri];
                var offset_j: usize = 0;
                for (0..total_reqs) |rj| {
                    const len_j = seq_lengths[rj];

                    if (ri == rj) {
                        // Same request: causal attention within the request's span.
                        for (0..len_i) |row| {
                            for (0..len_j) |col| {
                                const attending = batch_builder.isAttending(mask, n, offset_i + row, offset_j + col);
                                if (col <= row) {
                                    // Causal: token at row can attend to positions 0..row.
                                    try testing.expect(attending);
                                } else {
                                    // Future token: should be blocked.
                                    try testing.expect(!attending);
                                }
                            }
                        }
                    } else {
                        // Different requests: ALL cross-attention must be blocked.
                        for (0..len_i) |row| {
                            for (0..len_j) |col| {
                                const attending = batch_builder.isAttending(mask, n, offset_i + row, offset_j + col);
                                try testing.expect(!attending);
                            }
                        }
                    }

                    offset_j += len_j;
                }
                offset_i += len_i;
            }
        }

        // --- Cleanup ---
        result.deinit();
        for (0..num_prefill) |i| prefill_reqs[i].deinit(allocator);
        for (0..num_decode) |i| decode_reqs[i].deinit(allocator);
    }
}
