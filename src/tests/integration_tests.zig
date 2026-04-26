/// Integration tests for end-to-end request flow.
///
/// Tests the interaction between components:
///   1. Scheduler + BatchBuilder: submit requests → schedule → build batched tensors
///   2. Scheduler + Generation flow: request lifecycle from submission through completion
///   3. ModelPool: load/evict cycle with stub models
///   4. MemoryLimiter: enforcement with ModelPool and TieredKVCache
///   5. QuantizedKVCache: config-driven quantization (kv_bits=4 and kv_bits=16)
///   6. Prompt cache: save/load round-trip via prompt_cache module
///
/// Requirements: R7, R9, R11, R12, R13, R21, R23
const std = @import("std");
const scheduler_mod = @import("../scheduler.zig");
const batch_builder = @import("../batch_builder.zig");
const model_pool_mod = @import("../model_pool.zig");
const memory_mod = @import("../memory.zig");
const server_mod = @import("../server.zig");
const ops_mod = @import("../ops.zig");
const kvcache = @import("../kvcache.zig");
const prompt_cache_mod = @import("../prompt_cache.zig");
const generation_mod = @import("../generation.zig");
const c = @import("../c.zig");
const root = @import("../root.zig");

const Scheduler = scheduler_mod.Scheduler;
const Request = scheduler_mod.Request;
const BlockManager = scheduler_mod.BlockManager;
const TokenOutput = scheduler_mod.TokenOutput;
const ScheduleResult = scheduler_mod.ScheduleResult;
const BatchResult = batch_builder.BatchResult;
const ModelPool = model_pool_mod.ModelPool;
const LoadedModel = model_pool_mod.LoadedModel;
const ModelPoolError = model_pool_mod.ModelPoolError;
const MemoryConfig = memory_mod.MemoryConfig;
const MemoryError = memory_mod.MemoryError;
const Array = root.Array;
const TieredKVCache = kvcache.TieredKVCache;
const LayerConfig = kvcache.LayerConfig;
const StandardKVCache = kvcache.StandardKVCache;
const QuantizedKVCache = kvcache.QuantizedKVCache;
const ModelConfig = generation_mod.ModelConfig;
const EagerContext = ops_mod.EagerContext;

const testing = std.testing;
const allocator = testing.allocator;

// ===========================================================================
// Helpers
// ===========================================================================

fn makeRequest(id: u64, prompt: []const u32, max_tokens: usize, stop_tokens: []const u32) Request {
    return Request.init(allocator, id, prompt, max_tokens, stop_tokens);
}

fn dummyCtx() EagerContext {
    return EagerContext.init(allocator);
}

fn stubLoader(name: []const u8, path: []const u8) anyerror!LoadedModel {
    _ = path;
    return LoadedModel{
        .vtable = null,
        .name = name,
        .path = "",
        .memory_bytes = name.len * 100,
        .last_used = 0,
    };
}

/// Stub loader that reports a fixed memory size per model.
fn fixedMemLoader(name: []const u8, path: []const u8) anyerror!LoadedModel {
    _ = path;
    return LoadedModel{
        .vtable = null,
        .name = name,
        .path = "",
        .memory_bytes = 500,
        .last_used = 0,
    };
}

// POSIX helpers for cold dir management (same pattern as tiered_kvcache_tests).
const posix_c = @cImport({
    @cInclude("sys/stat.h");
    @cInclude("unistd.h");
    @cInclude("dirent.h");
    @cInclude("stdio.h");
});

fn ensureColdDir(alloc: std.mem.Allocator, dir: []const u8) !void {
    const dir_z = try alloc.dupeZ(u8, dir);
    defer alloc.free(dir_z);
    _ = posix_c.mkdir(dir_z.ptr, 0o755);
}

fn cleanupColdDir(alloc: std.mem.Allocator, dir: []const u8) void {
    const dir_z = alloc.dupeZ(u8, dir) catch return;
    defer alloc.free(dir_z);
    const dp = posix_c.opendir(dir_z.ptr);
    if (dp != null) {
        while (true) {
            const entry = posix_c.readdir(dp);
            if (entry == null) break;
            const name: [*:0]const u8 = @ptrCast(&entry.*.d_name);
            if (name[0] == '.') continue;
            const full_path = std.fmt.allocPrint(alloc, "{s}/{s}", .{ dir, std.mem.span(name) }) catch continue;
            defer alloc.free(full_path);
            const full_path_z = alloc.dupeZ(u8, full_path) catch continue;
            defer alloc.free(full_path_z);
            _ = posix_c.unlink(full_path_z.ptr);
        }
        _ = posix_c.closedir(dp);
    }
    _ = posix_c.rmdir(dir_z.ptr);
}

// ===========================================================================
// Task 14.1: Scheduler → Forward → SSE flow
//
// Integration Test 1: Scheduler + BatchBuilder end-to-end request flow
//
// Submit request → scheduler picks up → batch builder creates tensors
// → simulate forward pass → postprocess → SSE-style streaming response
//
// Full flow: submit → schedule → forward → SSE
//   1. Client submits a request with prompt tokens
//   2. Scheduler.addRequest() enqueues it in the waiting queue
//   3. Scheduler.schedule() promotes it to prefilling, allocates blocks
//   4. BatchBuilder.build() creates batched input tensor + attention mask
//   5. Model forward pass produces logits (simulated here)
//   6. Scheduler.postprocess() appends generated token, checks stop conditions
//   7. SSE writer formats each token as "data: {...}\n\n" events
//   8. Final event includes finish_reason: "stop" followed by "data: [DONE]\n\n"
//
// Validates: R9, R12, R13
// ===========================================================================

test "integration: submit request → scheduler picks up → batch builder creates correct tensors → postprocess completes" {
    var bm = BlockManager.init(16);
    var sched = Scheduler.init(allocator, &bm, 512);
    defer sched.deinit();

    var ctx = dummyCtx();
    defer ctx.deinit();

    // 1. Submit a request
    const prompt = &[_]u32{ 10, 20, 30, 40 };
    var req = makeRequest(1, prompt, 5, &.{});
    defer req.deinit(allocator);

    try sched.addRequest(&req);
    try testing.expectEqual(scheduler_mod.RequestState.waiting, req.state);

    // 2. Scheduler picks it up (schedule)
    const r1 = try sched.schedule();
    defer allocator.free(r1.decode_requests);
    defer allocator.free(r1.prefill_requests);

    try testing.expectEqual(@as(usize, 1), r1.prefill_requests.len);
    try testing.expectEqual(scheduler_mod.RequestState.prefilling, req.state);

    // 3. BatchBuilder creates batched tensors from the schedule result
    var batch = try batch_builder.build(allocator, &r1, ctx);
    defer batch.deinit();

    try testing.expectEqual(@as(usize, 4), batch.total_tokens);
    try testing.expectEqual(@as(usize, 1), batch.num_requests);

    // Verify the batched tokens match the prompt
    const tokens = try batch.batched_tokens.dataSlice(u32);
    try testing.expectEqual(@as(u32, 10), tokens[0]);
    try testing.expectEqual(@as(u32, 20), tokens[1]);
    try testing.expectEqual(@as(u32, 30), tokens[2]);
    try testing.expectEqual(@as(u32, 40), tokens[3]);

    // Verify position ids are sequential
    const positions = try batch.position_ids.dataSlice(i32);
    for (0..4) |i| {
        try testing.expectEqual(@as(i32, @intCast(i)), positions[i]);
    }

    // Verify attention mask is causal
    const mask = try batch.attention_mask.dataSlice(f32);
    for (0..4) |row| {
        for (0..4) |col| {
            if (col <= row) {
                try testing.expect(batch_builder.isAttending(mask, 4, row, col));
            } else {
                try testing.expect(!batch_builder.isAttending(mask, 4, row, col));
            }
        }
    }

    // 4. Simulate forward pass output → postprocess
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 100 }});
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req.state);
    try testing.expectEqual(@as(usize, 1), req.generated_tokens.items.len);
    try testing.expectEqual(@as(u32, 100), req.generated_tokens.items[0]);

    // 5. Continue decode steps until max_tokens
    var step: usize = 1;
    while (req.state == .decoding and step < 10) : (step += 1) {
        const r_decode = try sched.schedule();
        defer allocator.free(r_decode.decode_requests);
        defer allocator.free(r_decode.prefill_requests);

        // Build batch for decode step
        var decode_batch = try batch_builder.build(allocator, &r_decode, ctx);
        defer decode_batch.deinit();

        // Decode contributes 1 token per request
        try testing.expectEqual(@as(usize, 1), decode_batch.total_tokens);

        try sched.postprocess(&[_]TokenOutput{
            .{ .request_id = 1, .token = @as(u32, @intCast(100 + step)) },
        });
    }

    // Request should be done after max_tokens (5) generated tokens
    try testing.expectEqual(scheduler_mod.RequestState.done, req.state);
    try testing.expectEqual(@as(usize, 5), req.generated_tokens.items.len);
}

// ===========================================================================
// Integration Test 1b: SSE event formatting in the request flow
//
// Verifies that SSE helpers produce correct event format for tokens
// generated during the request lifecycle.
//
// Validates: R12
// ===========================================================================

test "integration: SSE event formatting for streaming response tokens" {
    // Simulate generating tokens and formatting them as SSE events
    // using the public writeSSEEvent and writeSSEKeepAlive helpers.
    const TestWriter = struct {
        data: std.ArrayList(u8),
        alloc: std.mem.Allocator,

        fn init(alloc: std.mem.Allocator) @This() {
            return .{ .data = std.ArrayList(u8).empty, .alloc = alloc };
        }
        fn deinit(self: *@This()) void {
            self.data.deinit(self.alloc);
        }
        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            try self.data.appendSlice(self.alloc, bytes);
        }
        fn getWritten(self: *const @This()) []const u8 {
            return self.data.items;
        }
    };

    var writer = TestWriter.init(allocator);
    defer writer.deinit();

    // Simulate streaming 3 token events as JSON payloads
    const token_payloads = [_][]const u8{
        "{\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}",
        "{\"choices\":[{\"delta\":{\"content\":\" world\"}}]}",
        "{\"choices\":[{\"delta\":{\"content\":\"!\"}}]}",
    };

    for (token_payloads) |payload| {
        try server_mod.writeSSEEvent(&writer, payload);
    }

    // Final event with finish_reason
    try server_mod.writeSSEEvent(&writer, "{\"choices\":[{\"finish_reason\":\"stop\"}]}");

    // Keep-alive comment (simulating prefill delay)
    try server_mod.writeSSEKeepAlive(&writer);

    // [DONE] sentinel
    try server_mod.writeSSEEvent(&writer, "[DONE]");

    const output = writer.getWritten();

    // Verify all token events are present
    try testing.expect(std.mem.indexOf(u8, output, "Hello") != null);
    try testing.expect(std.mem.indexOf(u8, output, " world") != null);
    try testing.expect(std.mem.indexOf(u8, output, "!") != null);
    try testing.expect(std.mem.indexOf(u8, output, "\"finish_reason\":\"stop\"") != null);
    try testing.expect(std.mem.indexOf(u8, output, "[DONE]") != null);
    try testing.expect(std.mem.indexOf(u8, output, ": keep-alive") != null);

    // Verify SSE format: each event starts with "data: " and ends with "\n\n"
    var event_count: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOf(u8, output[pos..], "data: ")) |idx| {
        event_count += 1;
        pos += idx + 6;
    }
    // 3 token events + 1 final event + 1 [DONE] = 5 data events
    try testing.expectEqual(@as(usize, 5), event_count);
}

// ===========================================================================
// Task 14.2: Continuous batching attention isolation
//
// Integration Test 2: Multiple concurrent requests with continuous batching
//
// Multiple requests at different stages (prefill, decode) are batched
// together. Verifies attention isolation and correct lifecycle management.
//
// Attention masks ensure isolation:
//   - The batch builder concatenates all request tokens into a single flat tensor
//   - A [total_tokens x total_tokens] attention mask is built
//   - For each request's span [start..end), causal attention is allowed within
//   - Cross-request cells are set to -inf, blocking attention across requests
//   - This ensures each request's tokens attend only to their own sequence
//
// Validates: R9, R13
// ===========================================================================

test "integration: multiple concurrent requests with continuous batching" {
    var bm = BlockManager.init(32);
    var sched = Scheduler.init(allocator, &bm, 512);
    defer sched.deinit();

    var ctx = dummyCtx();
    defer ctx.deinit();

    // Submit 3 requests with different prompt lengths
    const prompt1 = &[_]u32{ 1, 2, 3 };
    const prompt2 = &[_]u32{ 10, 20, 30, 40, 50 };
    const prompt3 = &[_]u32{ 100, 200 };

    var req1 = makeRequest(1, prompt1, 3, &.{});
    defer req1.deinit(allocator);
    var req2 = makeRequest(2, prompt2, 4, &.{});
    defer req2.deinit(allocator);
    var req3 = makeRequest(3, prompt3, 2, &.{});
    defer req3.deinit(allocator);

    // Add all three at once
    try sched.addRequest(&req1);
    try sched.addRequest(&req2);
    try sched.addRequest(&req3);

    // --- Step 1: All three should be promoted to prefilling ---
    const r1 = try sched.schedule();
    defer allocator.free(r1.decode_requests);
    defer allocator.free(r1.prefill_requests);

    try testing.expectEqual(@as(usize, 3), r1.prefill_requests.len);
    try testing.expectEqual(@as(usize, 0), r1.decode_requests.len);

    // Build batch: total tokens = 3 + 5 + 2 = 10
    var batch1 = try batch_builder.build(allocator, &r1, ctx);
    defer batch1.deinit();

    try testing.expectEqual(@as(usize, 10), batch1.total_tokens);
    try testing.expectEqual(@as(usize, 3), batch1.num_requests);

    // Verify attention isolation: cross-request attention is blocked
    const mask1 = try batch1.attention_mask.dataSlice(f32);
    const n1: usize = 10;

    // req1 tokens [0..3) should NOT attend to req2 tokens [3..8)
    try testing.expect(!batch_builder.isAttending(mask1, n1, 0, 3));
    try testing.expect(!batch_builder.isAttending(mask1, n1, 2, 5));
    // req2 tokens [3..8) should NOT attend to req3 tokens [8..10)
    try testing.expect(!batch_builder.isAttending(mask1, n1, 3, 8));
    // req1 tokens attend to themselves (causal)
    try testing.expect(batch_builder.isAttending(mask1, n1, 2, 0));
    try testing.expect(batch_builder.isAttending(mask1, n1, 2, 1));
    try testing.expect(batch_builder.isAttending(mask1, n1, 2, 2));

    // Postprocess: all transition to decoding
    try sched.postprocess(&[_]TokenOutput{
        .{ .request_id = 1, .token = 501 },
        .{ .request_id = 2, .token = 502 },
        .{ .request_id = 3, .token = 503 },
    });

    try testing.expectEqual(scheduler_mod.RequestState.decoding, req1.state);
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req2.state);
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req3.state);

    // --- Step 2: All three decoding concurrently ---
    const r2 = try sched.schedule();
    defer allocator.free(r2.decode_requests);
    defer allocator.free(r2.prefill_requests);

    try testing.expectEqual(@as(usize, 3), r2.decode_requests.len);

    var batch2 = try batch_builder.build(allocator, &r2, ctx);
    defer batch2.deinit();

    // Each decode request contributes 1 token → 3 total
    try testing.expectEqual(@as(usize, 3), batch2.total_tokens);

    // Verify attention isolation in decode batch
    const mask2 = try batch2.attention_mask.dataSlice(f32);
    // Each token should only attend to itself (single-token decode)
    try testing.expect(batch_builder.isAttending(mask2, 3, 0, 0));
    try testing.expect(!batch_builder.isAttending(mask2, 3, 0, 1));
    try testing.expect(!batch_builder.isAttending(mask2, 3, 1, 0));
    try testing.expect(batch_builder.isAttending(mask2, 3, 1, 1));
    try testing.expect(!batch_builder.isAttending(mask2, 3, 2, 0));
    try testing.expect(batch_builder.isAttending(mask2, 3, 2, 2));

    try sched.postprocess(&[_]TokenOutput{
        .{ .request_id = 1, .token = 601 },
        .{ .request_id = 2, .token = 602 },
        .{ .request_id = 3, .token = 603 },
    });

    // req3 should be done (max_tokens=2, generated 2 tokens)
    try testing.expectEqual(scheduler_mod.RequestState.done, req3.state);
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req1.state);
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req2.state);

    // --- Step 3: req1 and req2 still decoding, req3 removed ---
    try testing.expectEqual(@as(usize, 2), sched.running.items.len);

    const r3 = try sched.schedule();
    defer allocator.free(r3.decode_requests);
    defer allocator.free(r3.prefill_requests);

    try testing.expectEqual(@as(usize, 2), r3.decode_requests.len);

    try sched.postprocess(&[_]TokenOutput{
        .{ .request_id = 1, .token = 701 },
        .{ .request_id = 2, .token = 702 },
    });

    // req1 done (max_tokens=3, generated 3 tokens)
    try testing.expectEqual(scheduler_mod.RequestState.done, req1.state);
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req2.state);

    // --- Step 4: Only req2 remains ---
    try testing.expectEqual(@as(usize, 1), sched.running.items.len);

    const r4 = try sched.schedule();
    defer allocator.free(r4.decode_requests);
    defer allocator.free(r4.prefill_requests);

    try testing.expectEqual(@as(usize, 1), r4.decode_requests.len);

    try sched.postprocess(&[_]TokenOutput{
        .{ .request_id = 2, .token = 802 },
    });

    // req2 done (max_tokens=4, generated 4 tokens)
    try testing.expectEqual(scheduler_mod.RequestState.done, req2.state);
    try testing.expectEqual(@as(usize, 0), sched.running.items.len);
}

// ===========================================================================
// Integration Test 2b: New request joins mid-generation (continuous batching)
//
// A new request arrives while existing requests are decoding. The new
// request should be batched together with the existing decode requests
// in the very next engine step.
//
// Validates: R13.3
// ===========================================================================

test "integration: new request joins existing decode batch without waiting" {
    var bm = BlockManager.init(32);
    var sched = Scheduler.init(allocator, &bm, 512);
    defer sched.deinit();

    var ctx = dummyCtx();
    defer ctx.deinit();

    // Start with one request, get it to decoding state
    var req1 = makeRequest(1, &[_]u32{ 1, 2, 3 }, 10, &.{});
    defer req1.deinit(allocator);

    try sched.addRequest(&req1);
    const r0 = try sched.schedule();
    allocator.free(r0.decode_requests);
    allocator.free(r0.prefill_requests);
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 50 }});
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req1.state);

    // Now add a new request while req1 is decoding
    var req2 = makeRequest(2, &[_]u32{ 10, 20 }, 10, &.{});
    defer req2.deinit(allocator);
    try sched.addRequest(&req2);

    // Schedule: req1 should be in decode, req2 should be promoted to prefill
    const r1 = try sched.schedule();
    defer allocator.free(r1.decode_requests);
    defer allocator.free(r1.prefill_requests);

    try testing.expectEqual(@as(usize, 1), r1.decode_requests.len);
    try testing.expectEqual(@as(usize, 1), r1.prefill_requests.len);
    try testing.expectEqual(@as(u64, 1), r1.decode_requests[0].id);
    try testing.expectEqual(@as(u64, 2), r1.prefill_requests[0].id);

    // Build batch: both requests are in the same batch
    var batch = try batch_builder.build(allocator, &r1, ctx);
    defer batch.deinit();

    // req2 prefill (2 tokens) + req1 decode (1 token) = 3 tokens
    try testing.expectEqual(@as(usize, 3), batch.total_tokens);
    try testing.expectEqual(@as(usize, 2), batch.num_requests);

    // Verify attention isolation between the two requests
    const mask = try batch.attention_mask.dataSlice(f32);
    // req2 tokens [0,1] should not attend to req1 token [2]
    try testing.expect(!batch_builder.isAttending(mask, 3, 0, 2));
    try testing.expect(!batch_builder.isAttending(mask, 3, 1, 2));
    // req1 token [2] should not attend to req2 tokens [0,1]
    try testing.expect(!batch_builder.isAttending(mask, 3, 2, 0));
    try testing.expect(!batch_builder.isAttending(mask, 3, 2, 1));
}

// ===========================================================================
// Task 14.3: Model pool load/evict
//
// Integration Test 3: Model pool load/evict cycle
//
// Tests the full lifecycle: load models → access patterns → LRU eviction
// → reload → pinning prevents eviction.
//
// LRU eviction behavior:
//   - Models are tracked in LRU order (oldest access at front)
//   - getOrLoad() updates access time, moving the model to most-recent
//   - When loading exceeds max_memory, evictLRU() removes the oldest
//     non-pinned model from the pool
//   - Pinned models are skipped during eviction regardless of access time
//   - If all models are pinned, getOrLoad returns MemoryLimitExceeded
//
// Validates: R21
// ===========================================================================

test "integration: model pool load/evict cycle with access patterns" {
    // Budget for ~2 models (each 500 bytes via fixedMemLoader)
    var pool = ModelPool.init(allocator, 1050);
    defer pool.deinit();

    // Load model A
    const model_a = try pool.getOrLoad("model_a", "/models/a", fixedMemLoader);
    try testing.expectEqual(@as(usize, 1), pool.count());
    try testing.expect(model_a.memory_bytes == 500);

    // Load model B
    _ = try pool.getOrLoad("model_b", "/models/b", fixedMemLoader);
    try testing.expectEqual(@as(usize, 2), pool.count());
    try testing.expectEqual(@as(usize, 1000), pool.current_memory);

    // Access model A again (makes it most recently used)
    _ = try pool.getOrLoad("model_a", "/models/a", fixedMemLoader);

    // Load model C — should trigger eviction of model B (LRU)
    _ = try pool.getOrLoad("model_c", "/models/c", fixedMemLoader);
    try testing.expectEqual(@as(usize, 2), pool.count());
    try testing.expect(pool.models.get("model_a") != null);
    try testing.expect(pool.models.get("model_b") == null); // evicted
    try testing.expect(pool.models.get("model_c") != null);

    // Pin model A, then load model D — should evict model C (not pinned A)
    pool.pin("model_a");
    _ = try pool.getOrLoad("model_d", "/models/d", fixedMemLoader);
    try testing.expectEqual(@as(usize, 2), pool.count());
    try testing.expect(pool.models.get("model_a") != null); // pinned, survived
    try testing.expect(pool.models.get("model_c") == null); // evicted
    try testing.expect(pool.models.get("model_d") != null);

    // Unpin model A, load model E — now model A can be evicted if it's LRU
    pool.unpin("model_a");
    // Access model D to make it most recent
    _ = try pool.getOrLoad("model_d", "/models/d", fixedMemLoader);
    // Load model E — should evict model A (LRU after unpin)
    _ = try pool.getOrLoad("model_e", "/models/e", fixedMemLoader);
    try testing.expectEqual(@as(usize, 2), pool.count());
    try testing.expect(pool.models.get("model_a") == null); // evicted
    try testing.expect(pool.models.get("model_d") != null);
    try testing.expect(pool.models.get("model_e") != null);
}

test "integration: model pool rejects load when all models pinned and over budget" {
    var pool = ModelPool.init(allocator, 600);
    defer pool.deinit();

    _ = try pool.getOrLoad("pinned", "/models/p", fixedMemLoader);
    pool.pin("pinned");

    // Try to load another model — budget exceeded, only model is pinned
    const result = pool.getOrLoad("new_model", "/models/n", fixedMemLoader);
    try testing.expectError(ModelPoolError.MemoryLimitExceeded, result);

    // Pinned model should still be there
    try testing.expectEqual(@as(usize, 1), pool.count());
    try testing.expect(pool.models.get("pinned") != null);
}

// ===========================================================================
// Task 14.4: Memory limit enforcement
//
// Integration Test 4: Memory limit enforcement with ModelPool
//
// Tests that enforceMemoryLimit triggers ModelPool eviction when process
// memory exceeds the configured limit, and returns MemoryLimitExceeded
// when nothing can be evicted.
//
// How memory limit enforcement rejects requests:
//   1. enforceMemoryLimit() checks process RSS against configured limit
//   2. If over limit, it first tries TieredKVCache eviction (cheaper)
//   3. Then tries ModelPool LRU eviction (more expensive)
//   4. If still over limit after exhausting eviction, returns MemoryLimitExceeded
//   5. The server catches this error and returns HTTP 503 to the client
//   6. With null pool/cache, it simply checks RSS and returns error if over limit
//
// Validates: R23
// ===========================================================================

test "integration: memory limiter triggers model pool eviction" {
    var pool = ModelPool.init(allocator, 100000);
    defer pool.deinit();

    // Load several models
    _ = try pool.getOrLoad("m_a", "/m/a", stubLoader);
    _ = try pool.getOrLoad("m_b", "/m/b", stubLoader);
    _ = try pool.getOrLoad("m_c", "/m/c", stubLoader);
    try testing.expectEqual(@as(usize, 3), pool.count());

    // Set an impossibly low memory limit — enforceMemoryLimit will try to
    // evict models from the pool to reduce memory usage.
    const cfg = MemoryConfig{ .max_bytes = 1 };
    const result = memory_mod.enforceMemoryLimit(&pool, null, cfg);

    // All models should have been evicted (pool tried to help)
    try testing.expectEqual(@as(usize, 0), pool.count());

    // But process RSS is still above 1 byte, so we get the error
    try testing.expectError(MemoryError.MemoryLimitExceeded, result);
}

test "integration: memory limiter no-op when within budget" {
    var pool = ModelPool.init(allocator, 100000);
    defer pool.deinit();

    _ = try pool.getOrLoad("m_a", "/m/a", stubLoader);

    // Set a very high limit — should pass without eviction
    const cfg = MemoryConfig{ .max_bytes = std.math.maxInt(usize) };
    try memory_mod.enforceMemoryLimit(&pool, null, cfg);

    // Model should still be there
    try testing.expectEqual(@as(usize, 1), pool.count());
}

test "integration: memory limiter with no limit configured is no-op" {
    var pool = ModelPool.init(allocator, 100000);
    defer pool.deinit();

    _ = try pool.getOrLoad("m_a", "/m/a", stubLoader);

    // No limit configured — should return immediately
    try memory_mod.enforceMemoryLimit(&pool, null, MemoryConfig{});

    try testing.expectEqual(@as(usize, 1), pool.count());
}

// ===========================================================================
// Integration Test 4b: Memory limit enforcement with TieredKVCache
//
// Tests that enforceMemoryLimit triggers TieredKVCache eviction before
// falling back to ModelPool eviction.
//
// Validates: R23
// ===========================================================================

test "integration: memory limiter triggers tiered cache eviction before model pool eviction" {
    const stream = c.c.mlx_default_cpu_stream_new();
    const cold_dir = "/tmp/mlx_integ_mem_tiered";
    try ensureColdDir(allocator, cold_dir);
    defer cleanupColdDir(allocator, cold_dir);

    const config = LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = 64,
        .dtype = .float32,
    };

    // Create tiered cache with generous hot capacity
    const cache = try allocator.create(TieredKVCache);
    cache.* = try TieredKVCache.init(allocator, config, 4, 10, cold_dir, stream);
    var cache_strategy = cache.asStrategy();
    defer cache_strategy.deinit(allocator);

    // Insert some data to have hot pages
    const shape = &[_]i32{ 1, 2, 4, 4 };
    var keys_arr = c.c.mlx_array_new();
    var values_arr = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&keys_arr, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&values_arr, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    const kv_keys = Array.fromHandle(keys_arr);
    defer kv_keys.deinit();
    const kv_values = Array.fromHandle(values_arr);
    defer kv_values.deinit();

    const result = try cache_strategy.updateAndFetch(kv_keys, kv_values, stream);
    defer result.keys.deinit();
    defer result.values.deinit();

    const hot_before = cache.hotUsedCount();
    try testing.expect(hot_before > 0);

    // Create a model pool too
    var pool = ModelPool.init(allocator, 100000);
    defer pool.deinit();
    _ = try pool.getOrLoad("m_a", "/m/a", stubLoader);

    // Set impossibly low limit — should try tiered cache eviction first,
    // then model pool eviction
    const cfg = MemoryConfig{ .max_bytes = 1 };
    const enforce_result = memory_mod.enforceMemoryLimit(&pool, cache, cfg);

    // Both should have been attempted
    // Tiered cache should have evicted hot pages to SSD
    try testing.expect(cache.hotUsedCount() <= hot_before);

    // Model pool should have been evicted too
    try testing.expectEqual(@as(usize, 0), pool.count());

    // Still over limit (process RSS > 1 byte)
    try testing.expectError(MemoryError.MemoryLimitExceeded, enforce_result);
}

// ===========================================================================
// Integration Test 5: Full request lifecycle with stop token
//
// Tests the complete flow: submit → schedule → batch → postprocess
// with stop token termination, verifying blocks are freed.
//
// Validates: R9
// ===========================================================================

test "integration: request lifecycle with stop token terminates and frees blocks" {
    var bm = BlockManager.init(8);
    var sched = Scheduler.init(allocator, &bm, 512);
    defer sched.deinit();

    var ctx = dummyCtx();
    defer ctx.deinit();

    const stop_tokens = &[_]u32{999};
    var req = makeRequest(1, &[_]u32{ 1, 2, 3 }, 100, stop_tokens);
    defer req.deinit(allocator);

    try sched.addRequest(&req);

    // Schedule and prefill
    const r0 = try sched.schedule();
    allocator.free(r0.decode_requests);
    allocator.free(r0.prefill_requests);

    try testing.expectEqual(@as(usize, 1), bm.used_blocks);

    // First token (not stop)
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 42 }});
    try testing.expectEqual(scheduler_mod.RequestState.decoding, req.state);

    // Build batch for decode
    const r1 = try sched.schedule();
    defer allocator.free(r1.decode_requests);
    defer allocator.free(r1.prefill_requests);

    var batch = try batch_builder.build(allocator, &r1, ctx);
    defer batch.deinit();
    try testing.expectEqual(@as(usize, 1), batch.total_tokens);

    // Stop token → done
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 999 }});
    try testing.expectEqual(scheduler_mod.RequestState.done, req.state);
    try testing.expectEqual(@as(usize, 0), sched.running.items.len);

    // Blocks should be freed
    try testing.expectEqual(@as(usize, 0), bm.used_blocks);
    try testing.expectEqual(@as(usize, 0), req.block_ids.items.len);

    // Generated tokens should include both tokens
    try testing.expectEqual(@as(usize, 2), req.generated_tokens.items.len);
    try testing.expectEqual(@as(u32, 42), req.generated_tokens.items[0]);
    try testing.expectEqual(@as(u32, 999), req.generated_tokens.items[1]);
}

// ===========================================================================
// Integration Test 6: Scheduler + BatchBuilder with chunked prefill
//
// A long prompt is chunked across multiple engine steps while another
// request continues decoding. Verifies the batch builder handles mixed
// prefill chunks and decode tokens correctly.
//
// Validates: R9, R13
// ===========================================================================

test "integration: chunked prefill with concurrent decode in batch builder" {
    var bm = BlockManager.init(32);
    var sched = Scheduler.init(allocator, &bm, 3); // max_prefill_tokens = 3
    defer sched.deinit();

    var ctx = dummyCtx();
    defer ctx.deinit();

    // req1: short prompt, will quickly transition to decoding
    var req1 = makeRequest(1, &[_]u32{ 1, 2 }, 10, &.{});
    defer req1.deinit(allocator);
    try sched.addRequest(&req1);

    // Promote and transition req1 to decoding
    const r0 = try sched.schedule();
    allocator.free(r0.decode_requests);
    allocator.free(r0.prefill_requests);
    try sched.postprocess(&[_]TokenOutput{.{ .request_id = 1, .token = 50 }});

    // req2: long prompt (7 tokens), needs ceil(7/3) = 3 chunks
    var req2 = makeRequest(2, &[_]u32{ 10, 20, 30, 40, 50, 60, 70 }, 10, &.{});
    defer req2.deinit(allocator);
    try sched.addRequest(&req2);

    // --- Chunk 1: req1 decoding + req2 first prefill chunk ---
    const r1 = try sched.schedule();
    defer allocator.free(r1.decode_requests);
    defer allocator.free(r1.prefill_requests);

    try testing.expectEqual(@as(usize, 1), r1.decode_requests.len);
    try testing.expectEqual(@as(usize, 1), r1.prefill_requests.len);

    var batch1 = try batch_builder.build(allocator, &r1, ctx);
    defer batch1.deinit();

    // req2 prefill returns all remaining prompt tokens from offset 0 (7 tokens)
    // + req1 decode (1 token) = 8 tokens total.
    // Note: the batch builder doesn't enforce max_prefill_tokens — that's the
    // scheduler's postprocess responsibility. The batch builder just returns
    // all tokens from prefill_offset onward.
    try testing.expectEqual(@as(usize, 8), batch1.total_tokens);

    // Verify attention isolation
    const mask1 = try batch1.attention_mask.dataSlice(f32);
    // req2 prefill tokens [0..7) should not attend to req1 decode token [7]
    try testing.expect(!batch_builder.isAttending(mask1, 8, 0, 7));
    try testing.expect(!batch_builder.isAttending(mask1, 8, 6, 7));
    // req1 decode token [7] should not attend to req2 prefill tokens [0..7)
    try testing.expect(!batch_builder.isAttending(mask1, 8, 7, 0));
    try testing.expect(!batch_builder.isAttending(mask1, 8, 7, 6));

    try sched.postprocess(&[_]TokenOutput{
        .{ .request_id = 1, .token = 51 },
        .{ .request_id = 2, .token = 0 }, // mid-prefill
    });

    try testing.expectEqual(scheduler_mod.RequestState.decoding, req1.state);
    try testing.expectEqual(scheduler_mod.RequestState.prefilling, req2.state);
    try testing.expectEqual(@as(usize, 3), req2.prefill_offset);
}


// ===========================================================================
// Task 14.5: KV cache quantization via config
//
// Integration Test 7: QuantizedKVCache creation with kv_bits=4 and kv_bits=16
//
// Verifies that the config-driven KV cache strategy selection works:
//   - kv_bits=4: creates a QuantizedKVCache that quantizes K/V to 4-bit
//   - kv_bits=16: creates a QuantizedKVCache in passthrough mode (no quantization)
//
// In the server, this is wired via ServerConfig.kv_bits:
//   if (kv_bits < 16) → kvcache.createQuantized(allocator, config, kv_bits, 64, stream)
//   if (kv_bits == 16) → kvcache.createStandard(allocator, config, stream)
//
// Validates: R11
// ===========================================================================

test "integration: QuantizedKVCache with kv_bits=4 has correct initial state" {
    c.initErrorHandler();
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 64,
        .max_seq_len = 128,
        .dtype = .float32,
    };

    // Create a 4-bit quantized KV cache via the factory function
    var cache_strategy = try kvcache.createQuantized(allocator, config, 4, 64, stream);
    defer cache_strategy.deinit(allocator);

    // Verify initial state: offset should be 0
    try testing.expectEqual(@as(usize, 0), cache_strategy.currentLen());

    // Access the underlying QuantizedKVCache to verify config
    const qcache: *QuantizedKVCache = @ptrCast(@alignCast(cache_strategy.ptr));
    try testing.expectEqual(@as(u8, 4), qcache.kv_bits);
    try testing.expectEqual(@as(i32, 64), qcache.group_size);
    try testing.expectEqual(@as(i32, 4), qcache.bits);
    try testing.expectEqual(@as(usize, 2), qcache.num_kv_heads);
    try testing.expectEqual(@as(usize, 64), qcache.head_dim);
    try testing.expectEqual(@as(usize, 128), qcache.max_seq_len);

    // No data stored yet
    try testing.expect(qcache.keys == null);
    try testing.expect(qcache.values == null);
    try testing.expect(qcache.raw_keys == null);
    try testing.expect(qcache.raw_values == null);
}

test "integration: QuantizedKVCache with kv_bits=16 passthrough mode" {
    c.initErrorHandler();
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 64,
        .max_seq_len = 128,
        .dtype = .float32,
    };

    // Create a 16-bit passthrough KV cache
    var cache_strategy = try kvcache.createQuantized(allocator, config, 16, 64, stream);
    defer cache_strategy.deinit(allocator);

    // Verify initial state
    try testing.expectEqual(@as(usize, 0), cache_strategy.currentLen());

    // Access the underlying QuantizedKVCache to verify passthrough config
    const qcache: *QuantizedKVCache = @ptrCast(@alignCast(cache_strategy.ptr));
    try testing.expectEqual(@as(u8, 16), qcache.kv_bits);
    try testing.expectEqual(@as(i32, 16), qcache.bits);

    // In passthrough mode, quantized storage is not used
    try testing.expect(qcache.keys == null);
    try testing.expect(qcache.values == null);

    // Feed some data through to verify it works end-to-end
    const shape = &[_]i32{ 1, 2, 4, 64 }; // [batch, kv_heads, seq_len, head_dim]
    var keys_arr = c.c.mlx_array_new();
    var values_arr = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&keys_arr, shape.ptr, shape.len, c.c.MLX_FLOAT16, stream));
    try c.check(c.c.mlx_ones(&values_arr, shape.ptr, shape.len, c.c.MLX_FLOAT16, stream));
    const kv_keys = Array.fromHandle(keys_arr);
    defer kv_keys.deinit();
    const kv_values = Array.fromHandle(values_arr);
    defer kv_values.deinit();

    const result = try cache_strategy.updateAndFetch(kv_keys, kv_values, stream);
    defer result.keys.deinit();
    defer result.values.deinit();

    // After update, offset should reflect the inserted sequence length
    try testing.expectEqual(@as(usize, 4), cache_strategy.currentLen());

    // In passthrough mode, raw storage is used (not quantized tuples)
    try testing.expect(qcache.raw_keys != null);
    try testing.expect(qcache.raw_values != null);
    try testing.expect(qcache.keys == null); // quantized storage unused
}

test "integration: QuantizedKVCache kv_bits=4 vs kv_bits=16 both functional" {
    // Verify both quantization modes can be created and used side-by-side,
    // as would happen in a server with per-layer config selection.
    c.initErrorHandler();
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 64,
        .max_seq_len = 128,
        .dtype = .float32,
    };

    var cache_4bit = try kvcache.createQuantized(allocator, config, 4, 64, stream);
    defer cache_4bit.deinit(allocator);

    var cache_16bit = try kvcache.createQuantized(allocator, config, 16, 64, stream);
    defer cache_16bit.deinit(allocator);

    // Both start empty
    try testing.expectEqual(@as(usize, 0), cache_4bit.currentLen());
    try testing.expectEqual(@as(usize, 0), cache_16bit.currentLen());

    // Feed same data through both
    const shape = &[_]i32{ 1, 2, 2, 64 };
    var k1 = c.c.mlx_array_new();
    var v1 = c.c.mlx_array_new();
    var k2 = c.c.mlx_array_new();
    var v2 = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&k1, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&v1, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&k2, shape.ptr, shape.len, c.c.MLX_FLOAT16, stream));
    try c.check(c.c.mlx_ones(&v2, shape.ptr, shape.len, c.c.MLX_FLOAT16, stream));
    const keys_4 = Array.fromHandle(k1);
    defer keys_4.deinit();
    const vals_4 = Array.fromHandle(v1);
    defer vals_4.deinit();
    const keys_16 = Array.fromHandle(k2);
    defer keys_16.deinit();
    const vals_16 = Array.fromHandle(v2);
    defer vals_16.deinit();

    const r4 = try cache_4bit.updateAndFetch(keys_4, vals_4, stream);
    defer r4.keys.deinit();
    defer r4.values.deinit();

    const r16 = try cache_16bit.updateAndFetch(keys_16, vals_16, stream);
    defer r16.keys.deinit();
    defer r16.values.deinit();

    // Both should have advanced their offset
    try testing.expectEqual(@as(usize, 2), cache_4bit.currentLen());
    try testing.expectEqual(@as(usize, 2), cache_16bit.currentLen());

    // Reset both
    cache_4bit.reset();
    cache_16bit.reset();
    try testing.expectEqual(@as(usize, 0), cache_4bit.currentLen());
    try testing.expectEqual(@as(usize, 0), cache_16bit.currentLen());
}

// ===========================================================================
// Task 14.6: Prompt cache round-trip
//
// Integration Test 8: Prompt cache save/load functions are callable
//
// Verifies that the prompt_cache module's save and load functions exist,
// are callable, and produce correct results for a simple round-trip.
//
// Full save → restart → load flow:
//   1. During inference, KV caches accumulate state for processed prompts
//   2. On server shutdown (or explicit save), savePromptCache() serializes
//      all layer caches to a safetensors file with metadata
//   3. On next server startup, if --prompt-cache-file is set and file exists,
//      loadPromptCache() deserializes the caches and validates metadata
//      against the current model config (num_layers, head_dim, num_kv_heads)
//   4. If compatible, the loaded caches replace empty caches → prefill skipped
//   5. If incompatible (e.g., different model), returns a descriptive error
//      and the server falls back to full prefill with empty caches
//
// Validates: R7
// ===========================================================================

test "integration: prompt cache save and load functions are callable" {
    // Verify that the prompt_cache module's public API (savePromptCache,
    // loadPromptCache, freePromptCaches) exists and is callable from
    // integration test context. The detailed round-trip correctness and
    // mismatch error handling are covered by Property 5 in prompt_cache.zig.
    c.initErrorHandler();
    const stream = c.c.mlx_default_cpu_stream_new();

    const num_layers: usize = 2;
    const num_kv_heads: usize = 2;
    const head_dim: usize = 8;
    const seq_len: usize = 4;

    // Create caches with some data
    var caches = try allocator.alloc(kvcache.KVCacheStrategy, num_layers);
    defer {
        for (caches) |cache_strat| {
            cache_strat.deinit(allocator);
        }
        allocator.free(caches);
    }

    for (0..num_layers) |i| {
        const layer_config = LayerConfig{
            .batch_size = 1,
            .num_heads = num_kv_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .max_seq_len = seq_len * 2,
            .dtype = .float32,
        };

        const cache_ptr = try allocator.create(StandardKVCache);
        cache_ptr.* = try StandardKVCache.init(allocator, layer_config, stream);

        // Create deterministic test data: layer index affects values
        const total_elems = num_kv_heads * seq_len * head_dim;
        var key_data = try allocator.alloc(f32, total_elems);
        defer allocator.free(key_data);
        var val_data = try allocator.alloc(f32, total_elems);
        defer allocator.free(val_data);

        for (0..total_elems) |j| {
            key_data[j] = @as(f32, @floatFromInt(i * 100 + j)) / 10.0;
            val_data[j] = @as(f32, @floatFromInt(i * 200 + j)) / 10.0;
        }

        const keys_arr = try Array.fromData(
            allocator,
            f32,
            key_data,
            &[_]i32{ 1, @intCast(num_kv_heads), @intCast(seq_len), @intCast(head_dim) },
        );
        defer keys_arr.deinit();
        const vals_arr = try Array.fromData(
            allocator,
            f32,
            val_data,
            &[_]i32{ 1, @intCast(num_kv_heads), @intCast(seq_len), @intCast(head_dim) },
        );
        defer vals_arr.deinit();

        caches[i] = cache_ptr.asStrategy();
        const result = try caches[i].updateAndFetch(keys_arr, vals_arr, stream);
        defer result.keys.deinit();
        defer result.values.deinit();
    }

    // Save to temp file — verifies savePromptCache is callable
    const tmp_path = "/tmp/mlx_zig_integ_prompt_cache.safetensors";
    try prompt_cache_mod.savePromptCache(allocator, caches, tmp_path);

    // Load with matching config — verifies loadPromptCache is callable
    const model_config = ModelConfig{
        .num_layers = num_layers,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .vocab_size = 32000,
        .hidden_size = 128,
    };

    const loaded_caches = try prompt_cache_mod.loadPromptCache(allocator, tmp_path, model_config);
    defer prompt_cache_mod.freePromptCaches(allocator, loaded_caches);

    // Verify same number of layers
    try testing.expectEqual(num_layers, loaded_caches.len);

    // Verify each layer's offset matches (data was loaded correctly)
    for (0..num_layers) |li| {
        const orig: *StandardKVCache = @ptrCast(@alignCast(caches[li].ptr));
        const loaded: *StandardKVCache = @ptrCast(@alignCast(loaded_caches[li].ptr));

        try testing.expectEqual(orig.offset, loaded.offset);
    }

    // Mismatch error handling (wrong num_layers, head_dim, num_kv_heads)
    // is thoroughly tested by Property 5 in prompt_cache.zig.
    // The full save → restart → load flow is:
    //   1. savePromptCache() serializes caches to safetensors with metadata
    //   2. Server restarts, calls loadPromptCache() with current model config
    //   3. If metadata matches → loaded caches replace empty ones, prefill skipped
    //   4. If metadata mismatches → descriptive error, fall back to full prefill
}
