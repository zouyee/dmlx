const std = @import("std");
const scheduler_mod = @import("../scheduler.zig");

const Scheduler = scheduler_mod.Scheduler;
const Request = scheduler_mod.Request;
const BlockManager = scheduler_mod.BlockManager;
const TokenOutput = scheduler_mod.TokenOutput;

const testing = std.testing;
const allocator = testing.allocator;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn makeRequest(id: u64, prompt: []const u32, max_tokens: usize, stop_tokens: []const u32) Request {
    return Request.init(allocator, id, prompt, max_tokens, stop_tokens);
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

test "scheduler: empty schedule returns empty result" {
    var bm = BlockManager.init(16);
    var sched = Scheduler.init(allocator, &bm, 512);
    defer sched.deinit();

    const result = try sched.schedule();
    defer allocator.free(result.decode_requests);
    defer allocator.free(result.prefill_requests);

    try testing.expect(result.isEmpty());
    try testing.expectEqual(@as(usize, 0), result.totalRequests());
}

test "scheduler: single request goes through waiting → prefilling → decoding → done" {
    var bm = BlockManager.init(16);
    var sched = Scheduler.init(allocator, &bm, 512);
    defer sched.deinit();

    const prompt = &[_]u32{ 1, 2, 3 };
    var req = makeRequest(1, prompt, 5, &.{});
    defer req.deinit(allocator);

    try sched.addRequest(&req);
    try testing.expectEqual(scheduler_mod.RequestState.waiting, req.state);

    // Schedule: should promote to prefilling.
    const r1 = try sched.schedule();
    defer allocator.free(r1.decode_requests);
    defer allocator.free(r1.prefill_requests);

    try testing.expectEqual(@as(usize, 1), r1.prefill_requests.len);
    try testing.expectEqual(@as(usize, 0), r1.decode_requests.len);
    try testing.expectEqual(scheduler_mod.RequestState.prefilling, req.state);

    // Postprocess with first token → transitions to decoding.
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 100 }});
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req.state);
    try testing.expectEqual(@as(usize, 1), req.generated_tokens.items.len);

    // Schedule again: request should be in decode list.
    const r2 = try sched.schedule();
    defer allocator.free(r2.decode_requests);
    defer allocator.free(r2.prefill_requests);

    try testing.expectEqual(@as(usize, 0), r2.prefill_requests.len);
    try testing.expectEqual(@as(usize, 1), r2.decode_requests.len);
}

test "scheduler: request completes on stop token" {
    var bm = BlockManager.init(16);
    var sched = Scheduler.init(allocator, &bm, 512);
    defer sched.deinit();

    const stop_tokens = &[_]u32{999};
    var req = makeRequest(1, &[_]u32{ 1, 2 }, 100, stop_tokens);
    defer req.deinit(allocator);

    try sched.addRequest(&req);
    const r0 = try sched.schedule();
    allocator.free(r0.decode_requests);
    allocator.free(r0.prefill_requests);

    // First token (not stop) → decoding.
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 42 }});
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req.state);
    try testing.expectEqual(@as(usize, 1), sched.running.items.len);

    // Second token is stop → done, removed from running.
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 999 }});
    try testing.expectEqual(scheduler_mod.RequestState.done, req.state);
    try testing.expectEqual(@as(usize, 0), sched.running.items.len);
}

test "scheduler: request completes on max_tokens" {
    var bm = BlockManager.init(16);
    var sched = Scheduler.init(allocator, &bm, 512);
    defer sched.deinit();

    var req = makeRequest(1, &[_]u32{1}, 2, &.{});
    defer req.deinit(allocator);

    try sched.addRequest(&req);
    const r0 = try sched.schedule();
    allocator.free(r0.decode_requests);
    allocator.free(r0.prefill_requests);

    // Token 1.
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 10 }});
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req.state);

    // Token 2 → hits max_tokens, done.
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 20 }});
    try testing.expectEqual(scheduler_mod.RequestState.done, req.state);
    try testing.expectEqual(@as(usize, 0), sched.running.items.len);
}

test "scheduler: decode requests are prioritized over prefill" {
    var bm = BlockManager.init(16);
    var sched = Scheduler.init(allocator, &bm, 512);
    defer sched.deinit();

    // Add first request and promote it to running/decoding.
    var req1 = makeRequest(1, &[_]u32{ 1, 2 }, 10, &.{});
    defer req1.deinit(allocator);
    try sched.addRequest(&req1);
    const r1 = try sched.schedule();
    allocator.free(r1.decode_requests);
    allocator.free(r1.prefill_requests);
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 50 }});

    // Now add a second request (waiting).
    var req2 = makeRequest(2, &[_]u32{ 3, 4 }, 10, &.{});
    defer req2.deinit(allocator);
    try sched.addRequest(&req2);

    // Schedule: req1 should be in decode, req2 in prefill.
    const r2 = try sched.schedule();
    defer allocator.free(r2.decode_requests);
    defer allocator.free(r2.prefill_requests);

    try testing.expectEqual(@as(usize, 1), r2.decode_requests.len);
    try testing.expectEqual(@as(usize, 1), r2.prefill_requests.len);
    try testing.expectEqual(@as(u64, 1), r2.decode_requests[0].id);
    try testing.expectEqual(@as(u64, 2), r2.prefill_requests[0].id);
}

test "scheduler: requests stay waiting when blocks unavailable" {
    // Only 0 blocks available — nothing can be scheduled.
    var bm = BlockManager.init(0);
    var sched = Scheduler.init(allocator, &bm, 512);
    defer sched.deinit();

    var req = makeRequest(1, &[_]u32{ 1, 2 }, 10, &.{});
    defer req.deinit(allocator);

    try sched.addRequest(&req);

    const result = try sched.schedule();
    defer allocator.free(result.decode_requests);
    defer allocator.free(result.prefill_requests);

    // Request should remain in waiting queue.
    try testing.expectEqual(@as(usize, 0), result.prefill_requests.len);
    try testing.expectEqual(@as(usize, 0), result.decode_requests.len);
    try testing.expectEqual(@as(usize, 1), sched.waiting.items.len);
    try testing.expectEqual(scheduler_mod.RequestState.waiting, req.state);
    try testing.expectEqual(@as(usize, 1), result.blocks_needed);
}

test "scheduler: blocks freed when request completes" {
    var bm = BlockManager.init(2);
    var sched = Scheduler.init(allocator, &bm, 512);
    defer sched.deinit();

    var req = makeRequest(1, &[_]u32{1}, 1, &.{});
    defer req.deinit(allocator);

    try sched.addRequest(&req);
    const r0 = try sched.schedule();
    allocator.free(r0.decode_requests);
    allocator.free(r0.prefill_requests);

    // 1 block used.
    try testing.expectEqual(@as(usize, 1), bm.used_blocks);

    // Complete the request.
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 42 }});

    // Block should be freed.
    try testing.expectEqual(@as(usize, 0), bm.used_blocks);
    try testing.expectEqual(@as(usize, 0), req.block_ids.items.len);
}

test "scheduler: multiple requests scheduled and completed independently" {
    var bm = BlockManager.init(16);
    var sched = Scheduler.init(allocator, &bm, 512);
    defer sched.deinit();

    var req1 = makeRequest(1, &[_]u32{1}, 2, &.{});
    defer req1.deinit(allocator);
    var req2 = makeRequest(2, &[_]u32{2}, 3, &.{});
    defer req2.deinit(allocator);

    try sched.addRequest(&req1);
    try sched.addRequest(&req2);

    // Both should be promoted.
    const r1 = try sched.schedule();
    allocator.free(r1.decode_requests);
    allocator.free(r1.prefill_requests);

    try testing.expectEqual(@as(usize, 2), sched.running.items.len);
    try testing.expectEqual(@as(usize, 0), sched.waiting.items.len);

    // First tokens for both.
    try sched.postprocess(&[_]TokenOutput{
        .{ .request_id = 1, .token = 10 },
        .{ .request_id = 2, .token = 20 },
    });

    // req1 gets second token → done (max_tokens=2).
    // req2 still decoding.
    try sched.postprocess(&[_]TokenOutput{
        .{ .request_id = 1, .token = 11 },
        .{ .request_id = 2, .token = 21 },
    });

    try testing.expectEqual(scheduler_mod.RequestState.done, req1.state);
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req2.state);
    try testing.expectEqual(@as(usize, 1), sched.running.items.len);
    try testing.expectEqual(@as(u64, 2), sched.running.items[0].id);
}

test "scheduler: request isStopToken works correctly" {
    const stop_tokens = &[_]u32{ 10, 20, 30 };
    const req = makeRequest(1, &[_]u32{1}, 10, stop_tokens);

    try testing.expect(req.isStopToken(10));
    try testing.expect(req.isStopToken(20));
    try testing.expect(req.isStopToken(30));
    try testing.expect(!req.isStopToken(15));
    try testing.expect(!req.isStopToken(0));
}

test "scheduler: request seqLen tracks progress" {
    var req = makeRequest(1, &[_]u32{ 1, 2, 3 }, 10, &.{});
    defer req.deinit(allocator);

    try testing.expectEqual(@as(usize, 0), req.seqLen());

    // Simulate prefill completing.
    req.prefill_offset = 3;
    try testing.expectEqual(@as(usize, 3), req.seqLen());

    // Simulate generating tokens.
    try req.generated_tokens.append(allocator, 100);
    try testing.expectEqual(@as(usize, 4), req.seqLen());

    try req.generated_tokens.append(allocator, 200);
    try testing.expectEqual(@as(usize, 5), req.seqLen());
}

// ============================================================
// Property-Based Test
// (Property 7)
//
// Feature: production-deployment, Property 7: Scheduler
// Prioritization Invariant
//
// Verify schedule() output includes all running (decode) requests
// before any waiting (prefill) requests. Running requests with
// allocated blocks are always scheduled.
//
// **Validates: Requirements R9.2**
// ============================================================

test "Property 7: Scheduler Prioritization — running requests always scheduled, decode before prefill (100 iterations)" {
    const max_running = 8;
    const max_waiting = 8;
    // Enough prompt token storage for all requests across iterations.
    // Each request gets a unique 1-token prompt from this pool.
    const prompt_pool: [max_running + max_waiting]u32 = blk: {
        var arr: [max_running + max_waiting]u32 = undefined;
        for (0..arr.len) |i| arr[i] = @intCast(i + 1);
        break :blk arr;
    };

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // --- Generate random counts ---
        const num_running = rand.intRangeAtMost(usize, 0, max_running);
        const num_waiting = rand.intRangeAtMost(usize, 0, max_waiting);

        // Total blocks: enough for running + waiting so we can test the
        // full promotion path. Add extra so blocks are always available.
        const total_blocks = num_running + num_waiting + 4;
        var bm = BlockManager.init(total_blocks);
        var sched = Scheduler.init(allocator, &bm, 512);

        // Storage for request objects (stack-allocated, fixed-size arrays).
        var running_reqs: [max_running]Request = undefined;
        var waiting_reqs: [max_waiting]Request = undefined;

        // --- Phase 1: Create running (decode-phase) requests ---
        // We add them, schedule (promotes to prefilling), then postprocess
        // (transitions to decoding) so they end up in the running queue.
        for (0..num_running) |i| {
            running_reqs[i] = Request.init(
                allocator,
                @intCast(i + 1), // id 1..num_running
                prompt_pool[i .. i + 1], // 1-token prompt
                100, // generous max_tokens
                &.{},
            );
            try sched.addRequest(&running_reqs[i]);
        }

        if (num_running > 0) {
            // Schedule to promote waiting → prefilling and allocate blocks.
            const r_promote = try sched.schedule();
            allocator.free(r_promote.decode_requests);
            allocator.free(r_promote.prefill_requests);

            // Postprocess with a token for each to transition prefilling → decoding.
            var token_outputs: [max_running]TokenOutput = undefined;
            for (0..num_running) |i| {
                token_outputs[i] = .{
                    .request_id = @intCast(i + 1),
                    .token = 500 + @as(u32, @intCast(i)),
                };
            }
            try sched.postprocess(token_outputs[0..num_running]);
        }

        // Verify all running requests are now in decoding state.
        try testing.expectEqual(num_running, sched.running.items.len);
        for (sched.running.items) |req| {
            try testing.expectEqual(scheduler_mod.RequestState.decoding, req.state);
        }

        // --- Phase 2: Add new waiting requests ---
        for (0..num_waiting) |i| {
            const idx = num_running + i;
            waiting_reqs[i] = Request.init(
                allocator,
                @intCast(idx + 1), // ids after running
                prompt_pool[idx .. idx + 1],
                100,
                &.{},
            );
            try sched.addRequest(&waiting_reqs[i]);
        }

        // --- Phase 3: Schedule and verify the prioritization invariant ---
        const result = try sched.schedule();
        defer allocator.free(result.decode_requests);
        defer allocator.free(result.prefill_requests);

        // Property 7a: ALL running requests with allocated blocks appear in decode_requests.
        try testing.expectEqual(num_running, result.decode_requests.len);

        // Verify each running request is present in decode_requests.
        for (0..num_running) |i| {
            const expected_id: u64 = @intCast(i + 1);
            var found = false;
            for (result.decode_requests) |req| {
                if (req.id == expected_id) {
                    found = true;
                    // Must still be in decoding state.
                    try testing.expectEqual(scheduler_mod.RequestState.decoding, req.state);
                    // Must have allocated blocks.
                    try testing.expect(req.block_ids.items.len > 0);
                    break;
                }
            }
            try testing.expect(found);
        }

        // Property 7b: Waiting requests that got promoted appear in prefill_requests.
        // Since we have enough blocks, all waiting requests should be promoted.
        try testing.expectEqual(num_waiting, result.prefill_requests.len);

        for (result.prefill_requests) |req| {
            try testing.expectEqual(scheduler_mod.RequestState.prefilling, req.state);
        }

        // Property 7c: No prefill request ID appears in the decode list and vice versa.
        // This ensures the two lists are disjoint — decode (running) is separate from prefill (new).
        for (result.decode_requests) |d_req| {
            for (result.prefill_requests) |p_req| {
                try testing.expect(d_req.id != p_req.id);
            }
        }

        // --- Cleanup ---
        sched.deinit();
        for (0..num_running) |i| running_reqs[i].deinit(allocator);
        for (0..num_waiting) |i| waiting_reqs[i].deinit(allocator);
    }
}

test "Property 7: Scheduler Prioritization — running requests scheduled even under block pressure (100 iterations)" {
    const max_running = 6;
    const prompt_pool: [max_running]u32 = blk: {
        var arr: [max_running]u32 = undefined;
        for (0..arr.len) |i| arr[i] = @intCast(i + 1);
        break :blk arr;
    };

    var prng = std.Random.DefaultPrng.init(7777);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        const num_running = rand.intRangeAtMost(usize, 1, max_running);
        // Give exactly enough blocks for running requests — none left for waiting.
        const total_blocks = num_running;
        var bm = BlockManager.init(total_blocks);
        var sched = Scheduler.init(allocator, &bm, 512);

        var running_reqs: [max_running]Request = undefined;

        // Create and promote running requests.
        for (0..num_running) |i| {
            running_reqs[i] = Request.init(
                allocator,
                @intCast(i + 1),
                prompt_pool[i .. i + 1],
                100,
                &.{},
            );
            try sched.addRequest(&running_reqs[i]);
        }

        const r_promote = try sched.schedule();
        allocator.free(r_promote.decode_requests);
        allocator.free(r_promote.prefill_requests);

        var token_outputs: [max_running]TokenOutput = undefined;
        for (0..num_running) |i| {
            token_outputs[i] = .{
                .request_id = @intCast(i + 1),
                .token = 600 + @as(u32, @intCast(i)),
            };
        }
        try sched.postprocess(token_outputs[0..num_running]);

        // All blocks are now used by running requests. Add a waiting request.
        const waiting_prompt = &[_]u32{99};
        var waiting_req = Request.init(allocator, 99, waiting_prompt, 100, &.{});

        try sched.addRequest(&waiting_req);

        // Schedule: running requests MUST still be scheduled (decode).
        // Waiting request should stay waiting (no blocks available).
        const result = try sched.schedule();
        defer allocator.free(result.decode_requests);
        defer allocator.free(result.prefill_requests);

        // All running requests are in decode_requests.
        try testing.expectEqual(num_running, result.decode_requests.len);
        for (result.decode_requests) |req| {
            try testing.expectEqual(scheduler_mod.RequestState.decoding, req.state);
            try testing.expect(req.block_ids.items.len > 0);
        }

        // Waiting request could NOT be promoted — no blocks.
        try testing.expectEqual(@as(usize, 0), result.prefill_requests.len);
        try testing.expectEqual(@as(usize, 1), sched.waiting.items.len);
        try testing.expectEqual(scheduler_mod.RequestState.waiting, waiting_req.state);

        // Cleanup.
        sched.deinit();
        waiting_req.deinit(allocator);
        for (0..num_running) |i| running_reqs[i].deinit(allocator);
    }
}

test "block_manager: canAllocate and freeCount" {
    var bm = BlockManager.init(4);
    try testing.expectEqual(@as(usize, 4), bm.freeCount());
    try testing.expect(bm.canAllocate(4));
    try testing.expect(!bm.canAllocate(5));

    var req = makeRequest(1, &[_]u32{1}, 10, &.{});
    defer req.deinit(allocator);

    try bm.allocateBlocks(allocator, &req, 2);
    try testing.expectEqual(@as(usize, 2), bm.freeCount());
    try testing.expect(bm.canAllocate(2));
    try testing.expect(!bm.canAllocate(3));

    bm.freeBlocks(&req);
    try testing.expectEqual(@as(usize, 4), bm.freeCount());
}

// ===========================================================================
// Chunked Prefill Tests (Task 7.1)
//
// Validates: Requirements R14.1, R14.2, R14.3
// ===========================================================================

test "chunked prefill: short prompt fits in one chunk, no splitting" {
    // max_prefill_tokens = 10, prompt has 3 tokens → single chunk, no splitting.
    var bm = BlockManager.init(16);
    var sched = Scheduler.init(allocator, &bm, 10);
    defer sched.deinit();

    const prompt = &[_]u32{ 1, 2, 3 };
    var req = makeRequest(1, prompt, 5, &.{});
    defer req.deinit(allocator);

    try sched.addRequest(&req);

    // Schedule: promoted to prefilling.
    const r1 = try sched.schedule();
    defer allocator.free(r1.decode_requests);
    defer allocator.free(r1.prefill_requests);
    try testing.expectEqual(@as(usize, 1), r1.prefill_requests.len);
    try testing.expectEqual(scheduler_mod.RequestState.prefilling, req.state);
    try testing.expectEqual(@as(usize, 0), req.prefill_offset);

    // Postprocess: entire prompt fits in one chunk (3 <= 10).
    // prefill_offset advances to 3 (== prompt_tokens.len), transitions to decoding.
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 100 }});
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req.state);
    try testing.expectEqual(@as(usize, 3), req.prefill_offset);
    try testing.expectEqual(@as(usize, 1), req.generated_tokens.items.len);
    try testing.expectEqual(@as(u32, 100), req.generated_tokens.items[0]);
}

test "chunked prefill: long prompt split into multiple chunks" {
    // max_prefill_tokens = 4, prompt has 10 tokens → ceil(10/4) = 3 chunks.
    var bm = BlockManager.init(16);
    var sched = Scheduler.init(allocator, &bm, 4);
    defer sched.deinit();

    const prompt = &[_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var req = makeRequest(1, prompt, 5, &.{});
    defer req.deinit(allocator);

    try sched.addRequest(&req);

    // --- Chunk 1: tokens [0..4) ---
    const r1 = try sched.schedule();
    defer allocator.free(r1.decode_requests);
    defer allocator.free(r1.prefill_requests);
    try testing.expectEqual(@as(usize, 1), r1.prefill_requests.len);
    try testing.expectEqual(@as(usize, 0), req.prefill_offset);

    // After processing chunk 1, prefill_offset should advance to 4.
    // No generated token yet — still mid-prefill.
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 0 }});
    try testing.expectEqual(scheduler_mod.RequestState.prefilling, req.state);
    try testing.expectEqual(@as(usize, 4), req.prefill_offset);
    try testing.expectEqual(@as(usize, 0), req.generated_tokens.items.len);

    // --- Chunk 2: tokens [4..8) ---
    const r2 = try sched.schedule();
    defer allocator.free(r2.decode_requests);
    defer allocator.free(r2.prefill_requests);
    // Request is still prefilling, so it should be in prefill_requests.
    try testing.expectEqual(@as(usize, 1), r2.prefill_requests.len);
    try testing.expectEqual(@as(usize, 0), r2.decode_requests.len);

    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 0 }});
    try testing.expectEqual(scheduler_mod.RequestState.prefilling, req.state);
    try testing.expectEqual(@as(usize, 8), req.prefill_offset);
    try testing.expectEqual(@as(usize, 0), req.generated_tokens.items.len);

    // --- Chunk 3 (final): tokens [8..10) ---
    const r3 = try sched.schedule();
    defer allocator.free(r3.decode_requests);
    defer allocator.free(r3.prefill_requests);
    try testing.expectEqual(@as(usize, 1), r3.prefill_requests.len);

    // Final chunk: prefill_offset advances to 10 (== prompt_tokens.len).
    // Transitions to decoding, first generated token is appended.
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 42 }});
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req.state);
    try testing.expectEqual(@as(usize, 10), req.prefill_offset);
    try testing.expectEqual(@as(usize, 1), req.generated_tokens.items.len);
    try testing.expectEqual(@as(u32, 42), req.generated_tokens.items[0]);

    // --- Subsequent schedule: request should be in decode list ---
    const r4 = try sched.schedule();
    defer allocator.free(r4.decode_requests);
    defer allocator.free(r4.prefill_requests);
    try testing.expectEqual(@as(usize, 1), r4.decode_requests.len);
    try testing.expectEqual(@as(usize, 0), r4.prefill_requests.len);
}

test "chunked prefill: decode requests continue during chunked prefill" {
    // Req1 is already decoding. Req2 has a long prompt that requires chunked prefill.
    // Verify req1 continues to get decode steps while req2 is mid-prefill.
    var bm = BlockManager.init(16);
    var sched = Scheduler.init(allocator, &bm, 3); // max_prefill_tokens = 3
    defer sched.deinit();

    // Req1: short prompt, will be fully prefilled and transition to decoding.
    const prompt1 = &[_]u32{ 1, 2 };
    var req1 = makeRequest(1, prompt1, 10, &.{});
    defer req1.deinit(allocator);

    try sched.addRequest(&req1);
    const r0 = try sched.schedule();
    allocator.free(r0.decode_requests);
    allocator.free(r0.prefill_requests);
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 50 }});
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req1.state);

    // Req2: long prompt (7 tokens), needs ceil(7/3) = 3 chunks.
    const prompt2 = &[_]u32{ 10, 20, 30, 40, 50, 60, 70 };
    var req2 = makeRequest(2, prompt2, 10, &.{});
    defer req2.deinit(allocator);

    try sched.addRequest(&req2);

    // --- Step 1: req1 decoding, req2 chunk 1 ---
    const r1 = try sched.schedule();
    defer allocator.free(r1.decode_requests);
    defer allocator.free(r1.prefill_requests);
    try testing.expectEqual(@as(usize, 1), r1.decode_requests.len);
    try testing.expectEqual(@as(u64, 1), r1.decode_requests[0].id);
    try testing.expectEqual(@as(usize, 1), r1.prefill_requests.len);
    try testing.expectEqual(@as(u64, 2), r1.prefill_requests[0].id);

    try sched.postprocess(&[_]TokenOutput{
        .{ .request_id = 1, .token = 51 },
        .{ .request_id = 2, .token = 0 }, // mid-prefill, no real token
    });
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req1.state);
    try testing.expectEqual(scheduler_mod.RequestState.prefilling, req2.state);
    try testing.expectEqual(@as(usize, 2), req1.generated_tokens.items.len);
    try testing.expectEqual(@as(usize, 3), req2.prefill_offset);

    // --- Step 2: req1 decoding, req2 chunk 2 ---
    const r2 = try sched.schedule();
    defer allocator.free(r2.decode_requests);
    defer allocator.free(r2.prefill_requests);
    try testing.expectEqual(@as(usize, 1), r2.decode_requests.len);
    try testing.expectEqual(@as(u64, 1), r2.decode_requests[0].id);
    try testing.expectEqual(@as(usize, 1), r2.prefill_requests.len);
    try testing.expectEqual(@as(u64, 2), r2.prefill_requests[0].id);

    try sched.postprocess(&[_]TokenOutput{
        .{ .request_id = 1, .token = 52 },
        .{ .request_id = 2, .token = 0 },
    });
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req1.state);
    try testing.expectEqual(scheduler_mod.RequestState.prefilling, req2.state);
    try testing.expectEqual(@as(usize, 3), req1.generated_tokens.items.len);
    try testing.expectEqual(@as(usize, 6), req2.prefill_offset);

    // --- Step 3: req1 decoding, req2 final chunk (1 token) → transitions to decoding ---
    const r3 = try sched.schedule();
    defer allocator.free(r3.decode_requests);
    defer allocator.free(r3.prefill_requests);
    try testing.expectEqual(@as(usize, 1), r3.decode_requests.len);
    try testing.expectEqual(@as(usize, 1), r3.prefill_requests.len);

    try sched.postprocess(&[_]TokenOutput{
        .{ .request_id = 1, .token = 53 },
        .{ .request_id = 2, .token = 99 }, // first real generated token for req2
    });
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req1.state);
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req2.state);
    try testing.expectEqual(@as(usize, 4), req1.generated_tokens.items.len);
    try testing.expectEqual(@as(usize, 1), req2.generated_tokens.items.len);
    try testing.expectEqual(@as(u32, 99), req2.generated_tokens.items[0]);

    // --- Step 4: both decoding ---
    const r4 = try sched.schedule();
    defer allocator.free(r4.decode_requests);
    defer allocator.free(r4.prefill_requests);
    try testing.expectEqual(@as(usize, 2), r4.decode_requests.len);
    try testing.expectEqual(@as(usize, 0), r4.prefill_requests.len);
}

test "chunked prefill: exact multiple prompt length" {
    // max_prefill_tokens = 3, prompt has 6 tokens → exactly 2 chunks.
    var bm = BlockManager.init(16);
    var sched = Scheduler.init(allocator, &bm, 3);
    defer sched.deinit();

    const prompt = &[_]u32{ 1, 2, 3, 4, 5, 6 };
    var req = makeRequest(1, prompt, 5, &.{});
    defer req.deinit(allocator);

    try sched.addRequest(&req);

    // Chunk 1.
    const r1 = try sched.schedule();
    allocator.free(r1.decode_requests);
    allocator.free(r1.prefill_requests);
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 0 }});
    try testing.expectEqual(@as(usize, 3), req.prefill_offset);
    try testing.expectEqual(scheduler_mod.RequestState.prefilling, req.state);

    // Chunk 2 (final).
    const r2 = try sched.schedule();
    allocator.free(r2.decode_requests);
    allocator.free(r2.prefill_requests);
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 77 }});
    try testing.expectEqual(@as(usize, 6), req.prefill_offset);
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req.state);
    try testing.expectEqual(@as(usize, 1), req.generated_tokens.items.len);
}

test "chunked prefill: request helper hasPendingPrefill and currentPrefillChunkLen" {
    const prompt = &[_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var req = makeRequest(1, prompt, 5, &.{});

    // Initially: all 10 tokens pending.
    try testing.expect(req.hasPendingPrefill());
    try testing.expectEqual(@as(usize, 4), req.currentPrefillChunkLen(4));

    // After first chunk of 4.
    req.prefill_offset = 4;
    try testing.expect(req.hasPendingPrefill());
    try testing.expectEqual(@as(usize, 4), req.currentPrefillChunkLen(4));

    // After second chunk of 4 (offset=8, remaining=2).
    req.prefill_offset = 8;
    try testing.expect(req.hasPendingPrefill());
    try testing.expectEqual(@as(usize, 2), req.currentPrefillChunkLen(4));

    // After final chunk (offset=10, no remaining).
    req.prefill_offset = 10;
    try testing.expect(!req.hasPendingPrefill());
    try testing.expectEqual(@as(usize, 0), req.currentPrefillChunkLen(4));
}

test "chunked prefill: stop token on first generated token after prefill" {
    // Prompt of 5 tokens, max_prefill_tokens = 3 → 2 chunks.
    // First generated token is a stop token → request should go to done.
    var bm = BlockManager.init(16);
    var sched = Scheduler.init(allocator, &bm, 3);
    defer sched.deinit();

    const stop_tokens = &[_]u32{999};
    const prompt = &[_]u32{ 1, 2, 3, 4, 5 };
    var req = makeRequest(1, prompt, 10, stop_tokens);
    defer req.deinit(allocator);

    try sched.addRequest(&req);

    // Chunk 1.
    const r1 = try sched.schedule();
    allocator.free(r1.decode_requests);
    allocator.free(r1.prefill_requests);
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 0 }});
    try testing.expectEqual(scheduler_mod.RequestState.prefilling, req.state);
    try testing.expectEqual(@as(usize, 3), req.prefill_offset);

    // Chunk 2 (final) — first generated token is stop token.
    const r2 = try sched.schedule();
    allocator.free(r2.decode_requests);
    allocator.free(r2.prefill_requests);
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 999 }});
    try testing.expectEqual(scheduler_mod.RequestState.done, req.state);
    try testing.expectEqual(@as(usize, 5), req.prefill_offset);
    try testing.expectEqual(@as(usize, 1), req.generated_tokens.items.len);
    try testing.expectEqual(@as(usize, 0), sched.running.items.len);
}

// ============================================================
// Property-Based Test
// (Property 12)
//
// Feature: production-deployment, Property 12: Chunked Prefill
// Correctness
//
// For any prompt with length exceeding max_prefill_tokens, the
// Scheduler SHALL split it into ceil(prompt_len / max_prefill_tokens)
// chunks. While chunked prefill is in progress for one request,
// decode steps for other active requests SHALL continue to be
// scheduled.
//
// **Validates: Requirements R14.2, R14.3**
// ============================================================

test "Property 12: Chunked Prefill — correct chunk count and prefill_offset progression (100 iterations)" {
    // Maximum prompt length we'll generate (kept moderate to avoid huge arrays).
    const max_prompt_len = 64;
    // We need a static prompt pool large enough for the biggest prompt.
    const prompt_pool: [max_prompt_len]u32 = blk: {
        var arr: [max_prompt_len]u32 = undefined;
        for (0..arr.len) |i| arr[i] = @intCast(i + 1);
        break :blk arr;
    };

    var prng = std.Random.DefaultPrng.init(1234);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // --- Generate random parameters ---
        // prompt_len in [1..max_prompt_len]
        const prompt_len = rand.intRangeAtMost(usize, 1, max_prompt_len);
        // max_prefill_tokens in [1..32] (must be >= 1)
        const max_prefill_tokens = rand.intRangeAtMost(usize, 1, 32);

        const prompt = prompt_pool[0..prompt_len];

        // Expected number of chunks: ceil(prompt_len / max_prefill_tokens)
        const expected_chunks = (prompt_len + max_prefill_tokens - 1) / max_prefill_tokens;

        var bm = BlockManager.init(16);
        var sched = Scheduler.init(allocator, &bm, max_prefill_tokens);

        var req = makeRequest(1, prompt, 100, &.{});

        try sched.addRequest(&req);

        // Walk through each chunk and verify prefill_offset advances correctly.
        var chunk_count: usize = 0;
        var expected_offset: usize = 0;

        while (req.state != .decoding and req.state != .done) {
            const r = try sched.schedule();
            defer allocator.free(r.decode_requests);
            defer allocator.free(r.prefill_requests);

            // The request should appear in prefill_requests while mid-prefill.
            try testing.expect(r.prefill_requests.len >= 1);

            // Compute expected chunk size for this step.
            const remaining = prompt_len - expected_offset;
            const expected_chunk_len = @min(remaining, max_prefill_tokens);

            // Postprocess: advance the prefill.
            try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 0 }});

            expected_offset += expected_chunk_len;
            chunk_count += 1;

            // Verify prefill_offset matches expected.
            try testing.expectEqual(expected_offset, req.prefill_offset);

            // If this was the last chunk, the request should have transitioned.
            if (expected_offset >= prompt_len) {
                // Final chunk: postprocess with token=0 triggers transition to decoding.
                // But token 0 was already used above. The scheduler transitions on
                // the final chunk. Since token 0 is not a stop token and max_tokens=100,
                // the request should be in decoding state.
                // Note: token 0 is appended as the first generated token on the final chunk.
                break;
            } else {
                // Mid-prefill: request should still be prefilling.
                try testing.expectEqual(scheduler_mod.RequestState.prefilling, req.state);
            }
        }

        // Verify chunk count matches expected.
        try testing.expectEqual(expected_chunks, chunk_count);

        // Verify final prefill_offset equals prompt length.
        try testing.expectEqual(prompt_len, req.prefill_offset);

        // After all chunks, request should be in decoding state.
        try testing.expectEqual(scheduler_mod.RequestState.decoding, req.state);

        // Cleanup.
        sched.deinit();
        req.deinit(allocator);
    }
}

test "Property 12: Chunked Prefill — decode requests continue during chunked prefill (100 iterations)" {
    const max_prompt_len = 48;
    const prompt_pool: [max_prompt_len + 1]u32 = blk: {
        var arr: [max_prompt_len + 1]u32 = undefined;
        for (0..arr.len) |i| arr[i] = @intCast(i + 1);
        break :blk arr;
    };

    var prng = std.Random.DefaultPrng.init(9999);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // --- Generate random parameters ---
        // prompt_len for the chunked request: at least 2 so we get >1 chunk with small max_prefill_tokens
        const prompt_len = rand.intRangeAtMost(usize, 2, max_prompt_len);
        // max_prefill_tokens: pick a value that guarantees multiple chunks
        // (at most prompt_len - 1 so we get at least 2 chunks)
        const max_prefill_tokens = rand.intRangeAtMost(usize, 1, @max(1, prompt_len - 1));

        const expected_chunks = (prompt_len + max_prefill_tokens - 1) / max_prefill_tokens;

        // We need at least 2 chunks for this test to be meaningful.
        // If we accidentally got 1 chunk, skip this iteration.
        if (expected_chunks < 2) continue;

        var bm = BlockManager.init(32);
        var sched = Scheduler.init(allocator, &bm, max_prefill_tokens);

        // --- Set up a decode request (req1) that's already running ---
        const decode_prompt = prompt_pool[0..1]; // 1-token prompt, fits in one chunk
        var req1 = makeRequest(1, decode_prompt, 100, &.{});

        try sched.addRequest(&req1);

        // Promote req1 to decoding: schedule + postprocess.
        const r_init = try sched.schedule();
        allocator.free(r_init.decode_requests);
        allocator.free(r_init.prefill_requests);
        try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 500 }});
        try testing.expectEqual(scheduler_mod.RequestState.decoding, req1.state);

        // --- Add the chunked prefill request (req2) ---
        const prefill_prompt = prompt_pool[0..prompt_len];
        var req2 = makeRequest(2, prefill_prompt, 100, &.{});

        try sched.addRequest(&req2);

        // Track how many decode tokens req1 receives during req2's chunked prefill.
        const initial_decode_tokens = req1.generated_tokens.items.len;
        var decode_steps_during_prefill: usize = 0;

        // --- Walk through each chunk of req2's prefill ---
        var chunk_count: usize = 0;
        while (req2.state == .waiting or req2.state == .prefilling) {
            const r = try sched.schedule();
            defer allocator.free(r.decode_requests);
            defer allocator.free(r.prefill_requests);

            // Property 12 (R14.3): req1 (decode) MUST appear in decode_requests
            // during every step of req2's chunked prefill.
            var found_decode_req1 = false;
            for (r.decode_requests) |dreq| {
                if (dreq.id == 1) {
                    found_decode_req1 = true;
                    try testing.expectEqual(scheduler_mod.RequestState.decoding, dreq.state);
                    break;
                }
            }
            try testing.expect(found_decode_req1);

            // req2 should be in prefill_requests.
            var found_prefill_req2 = false;
            for (r.prefill_requests) |preq| {
                if (preq.id == 2) {
                    found_prefill_req2 = true;
                    break;
                }
            }
            try testing.expect(found_prefill_req2);

            // Postprocess both requests.
            try sched.postprocess(&[_]TokenOutput{
                .{ .request_id = 1, .token = 600 + @as(u32, @intCast(chunk_count)) },
                .{ .request_id = 2, .token = 0 }, // mid-prefill or final chunk token
            });

            decode_steps_during_prefill += 1;
            chunk_count += 1;

            // If req2 finished prefill, break.
            if (req2.state == .decoding or req2.state == .done) break;
        }

        // Verify req1 received a decode token on every chunk step.
        const total_decode_tokens = req1.generated_tokens.items.len - initial_decode_tokens;
        try testing.expectEqual(decode_steps_during_prefill, total_decode_tokens);

        // Verify chunk count matches expected.
        try testing.expectEqual(expected_chunks, chunk_count);

        // Cleanup.
        sched.deinit();
        req1.deinit(allocator);
        req2.deinit(allocator);
    }
}
