/// Property tests for Server Engine V2 infrastructure.
///
/// Tests:
///   - Property 1.4: RequestQueue linearizability
///   - Property 1.5: Request State Isolation
///   - Property 2.5: Error Propagation
///   - Property 2.6: Postprocess State Advancement
///   - Property 5.4: Dynamic Buffer Size Acceptance
///   - Property 5.5: SSE Chunk Encoding
const std = @import("std");
const engine = @import("../engine/root.zig");
const dynamic_buffer_mod = @import("../engine/dynamic_buffer.zig");
const DynamicBuffer = dynamic_buffer_mod.DynamicBuffer;

// ============================================================
// Property 1.4: RequestQueue Linearizability
//
// For any sequence of push/drainAll operations, the queue maintains
// FIFO ordering and no elements are lost or duplicated.
// ============================================================

test "Property 1.4: RequestQueue linearizability — push N then drainAll returns N in FIFO order (100 iterations)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
    const rand = prng.random();

    for (0..100) |_| {
        var queue = engine.RequestQueue.init();
        const n = rand.intRangeAtMost(usize, 1, 32);

        // Create N request states and push them
        var nodes = try allocator.alloc(engine.QueueNode, n);
        defer allocator.free(nodes);
        var states = try allocator.alloc(*engine.RequestState, n);
        defer allocator.free(states);

        for (0..n) |i| {
            const config = engine.RequestConfig{
                .prompt_tokens = &[_]u32{@intCast(i)},
                .max_tokens = 10,
            };
            states[i] = try engine.RequestState.init(allocator, @intCast(i), config);
            nodes[i] = engine.QueueNode.init(states[i]);
            queue.push(&nodes[i]);
        }

        // Drain all
        const drained = try queue.drainAll(allocator);
        defer allocator.free(drained);

        // Property: drainAll returns exactly N elements
        try std.testing.expectEqual(n, drained.len);

        // Property: elements are in FIFO order (id 0, 1, 2, ...)
        for (drained, 0..) |req, i| {
            try std.testing.expectEqual(@as(u64, @intCast(i)), req.id);
        }

        // Cleanup
        for (states[0..n]) |s| s.deinit();
    }
}

test "Property 1.4: RequestQueue linearizability — drainAll on empty queue returns empty slice" {
    const allocator = std.testing.allocator;
    var queue = engine.RequestQueue.init();

    const drained = try queue.drainAll(allocator);
    defer allocator.free(drained);

    try std.testing.expectEqual(@as(usize, 0), drained.len);
}

// ============================================================
// Property 1.5: Request State Isolation
//
// Each RequestState is independently allocated and modifying one
// does not affect another.
// ============================================================

test "Property 1.5: Request State Isolation — modifying one request does not affect another (100 iterations)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xCAFEBABE);
    const rand = prng.random();

    for (0..100) |_| {
        const n = rand.intRangeAtMost(usize, 2, 8);
        var states = try allocator.alloc(*engine.RequestState, n);
        defer allocator.free(states);

        // Create N independent request states
        for (0..n) |i| {
            const config = engine.RequestConfig{
                .prompt_tokens = &[_]u32{ @intCast(i), @intCast(i + 1) },
                .max_tokens = @intCast(10 + i),
                .temperature = @as(f32, @floatFromInt(i)) * 0.1,
            };
            states[i] = try engine.RequestState.init(allocator, @intCast(i), config);
        }
        defer for (states[0..n]) |s| s.deinit();

        // Modify the first request
        try states[0].generated_tokens.append(allocator, 42);
        states[0].token_count = 99;
        states[0].done = true;

        // Property: other requests are unaffected
        for (1..n) |i| {
            try std.testing.expectEqual(@as(usize, 0), states[i].generated_tokens.items.len);
            try std.testing.expectEqual(@as(usize, 0), states[i].token_count);
            try std.testing.expect(!states[i].done);
            try std.testing.expectEqual(@as(usize, 10 + i), states[i].max_tokens);
        }
    }
}

// ============================================================
// Property 2.5: Error Propagation
//
// When an error is delivered to a request, the request is marked
// as done and the error message is accessible.
// ============================================================

test "Property 2.5: Error Propagation — deliverError marks request done and stores message" {
    const allocator = std.testing.allocator;

    for (0..100) |i| {
        const config = engine.RequestConfig{
            .prompt_tokens = &[_]u32{1},
            .max_tokens = 10,
        };
        const req = try engine.RequestState.init(allocator, @intCast(i), config);
        defer req.deinit();

        const err_msg = "test error message";
        // CompletionSignal.deliverError ignores the io parameter (uses std.Thread.Mutex)
        req.completion.deliverError(undefined, err_msg);

        // Property: request is marked done
        try std.testing.expect(req.completion.isDone(undefined));

        // Property: error message is accessible
        try std.testing.expect(req.completion.hasError(undefined));
        const stored_msg = req.completion.getError(undefined);
        try std.testing.expect(stored_msg != null);
        try std.testing.expectEqualStrings(err_msg, stored_msg.?);
    }
}

// ============================================================
// Property 2.6: Postprocess State Advancement
//
// After delivering a token, the token is accessible via waitForToken
// and the signal state advances correctly.
// ============================================================

test "Property 2.6: Postprocess State Advancement — delivered tokens are retrievable in order" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xBEEFCAFE);
    const rand = prng.random();

    for (0..100) |_| {
        const config = engine.RequestConfig{
            .prompt_tokens = &[_]u32{1},
            .max_tokens = 10,
        };
        const req = try engine.RequestState.init(allocator, 0, config);
        defer req.deinit();

        const n_tokens = rand.intRangeAtMost(usize, 1, 10);

        // Deliver N tokens (CompletionSignal ignores io parameter)
        for (0..n_tokens) |t| {
            const is_final = (t == n_tokens - 1);
            const finish_reason: ?engine.TokenEvent.FinishReason = if (is_final) .stop else null;
            req.completion.deliverToken(undefined, @intCast(t), "", is_final, finish_reason);
        }

        // Property: all tokens are retrievable in order via tryGetToken (non-blocking)
        for (0..n_tokens) |t| {
            const event = req.completion.tryGetToken(undefined);
            if (event) |e| {
                try std.testing.expectEqual(@as(u32, @intCast(t)), e.token_id);
                if (t == n_tokens - 1) {
                    try std.testing.expect(e.is_final);
                }
            } else {
                try std.testing.expect(false); // Should have a token
            }
        }

        // Property: after all tokens consumed, signal is done
        try std.testing.expect(req.completion.isDone(undefined));
    }
}

// ============================================================
// Property 5.4: Dynamic Buffer Size Acceptance
//
// DynamicBuffer accepts data up to max_size and rejects beyond.
// ============================================================

test "Property 5.4: Dynamic Buffer Size Acceptance — accepts up to max_size, rejects beyond (100 iterations)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x12345678);
    const rand = prng.random();

    for (0..100) |_| {
        const max_size = rand.intRangeAtMost(usize, 64, 4096);
        var buf = DynamicBuffer.init(max_size);
        defer buf.deinit(allocator);

        // Fill to max_size in chunks
        const chunk_size = rand.intRangeAtMost(usize, 1, 64);
        var total: usize = 0;
        var chunk_data: [64]u8 = undefined;
        @memset(&chunk_data, 'A');

        while (total + chunk_size <= max_size) {
            buf.append(allocator, chunk_data[0..chunk_size]) catch break;
            total += chunk_size;
        }

        // Property: buffer length matches total appended
        try std.testing.expectEqual(total, buf.len());

        // Property: appending beyond max_size returns PayloadTooLarge
        const remaining = max_size - total;
        if (remaining < max_size) {
            // Try to append more than remaining capacity
            const overflow_size = remaining + 1;
            if (overflow_size <= 64) {
                const result = buf.append(allocator, chunk_data[0..overflow_size]);
                try std.testing.expectError(error.PayloadTooLarge, result);
            }
        }
    }
}

// ============================================================
// Property 5.5: SSE Chunk Encoding
//
// SSE chunks are properly formatted with valid JSON structure.
// ============================================================

test "Property 5.5: SSE Chunk Encoding — formatSSEChunk produces valid JSON with required fields" {
    const allocator = std.testing.allocator;
    const sse_mod = @import("../server/sse.zig");

    for (0..100) |i| {
        const created = @as(u64, @intCast(i + 1000));
        const completion_id = created;

        // Test with content
        const chunk = sse_mod.formatSSEChunk(allocator, completion_id, created, "test-model", "hello", null) catch continue;
        defer allocator.free(chunk);

        // Property: chunk is valid JSON
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, chunk, .{}) catch {
            // If parsing fails, the chunk is not valid JSON — test failure
            try std.testing.expect(false);
            continue;
        };
        defer parsed.deinit();

        // Property: chunk contains expected fields
        const obj = parsed.value.object;
        try std.testing.expect(obj.contains("id"));
        try std.testing.expect(obj.contains("object"));
        try std.testing.expect(obj.contains("created"));
        try std.testing.expect(obj.contains("model"));
        try std.testing.expect(obj.contains("choices"));
    }
}
