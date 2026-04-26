const std = @import("std");
const root = @import("../root.zig");

const Array = root.Array;
const kvcache = root.kvcache;
const EagerContext = root.EagerContext;
const c = @import("../c.zig");

/// Create a small KV tensor of shape [B=1, H=2, S=3, D=4] for testing.
fn makeTestKV(allocator: std.mem.Allocator, stream: c.c.mlx_stream) !struct { keys: Array, values: Array } {
    _ = allocator;
    const shape = &[_]i32{ 1, 2, 3, 4 };

    var keys = c.c.mlx_array_new();
    var values = c.c.mlx_array_new();

    try c.check(c.c.mlx_zeros(&keys, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_zeros(&values, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));

    return .{
        .keys = Array.fromHandle(keys),
        .values = Array.fromHandle(values),
    };
}

// ------------------------------------------------------------------
// StandardKVCache tests
// ------------------------------------------------------------------

test "StandardKVCache basic lifecycle" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = kvcache.LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = 16,
        .dtype = .float32,
    };

    var cache = try kvcache.StandardKVCache.init(allocator, config, stream);
    defer cache.keys.deinit();
    defer cache.values.deinit();

    try std.testing.expectEqual(@as(usize, 0), cache.offset);
    try std.testing.expectEqual(@as(usize, 1), cache.batch_size);
    try std.testing.expectEqual(@as(usize, 2), cache.num_kv_heads);
}

test "StandardKVCache asStrategy updateAndFetch" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = kvcache.LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = 16,
        .dtype = .float32,
    };

    const cache_ptr = try allocator.create(kvcache.StandardKVCache);
    defer allocator.destroy(cache_ptr);
    cache_ptr.* = try kvcache.StandardKVCache.init(allocator, config, stream);
    defer {
        cache_ptr.keys.deinit();
        cache_ptr.values.deinit();
    }

    var strategy = cache_ptr.asStrategy();

    // Initial state: empty.
    try std.testing.expectEqual(@as(usize, 0), strategy.currentLen());

    // First update: append 3 tokens.
    const kv = try makeTestKV(allocator, stream);
    defer kv.keys.deinit();
    defer kv.values.deinit();

    const result = try strategy.updateAndFetch(kv.keys, kv.values, stream);
    defer result.keys.deinit();
    defer result.values.deinit();

    try std.testing.expectEqual(@as(usize, 3), strategy.currentLen());

    // Second update: append another 3 tokens.
    const kv2 = try makeTestKV(allocator, stream);
    defer kv2.keys.deinit();
    defer kv2.values.deinit();

    const result2 = try strategy.updateAndFetch(kv2.keys, kv2.values, stream);
    defer result2.keys.deinit();
    defer result2.values.deinit();

    try std.testing.expectEqual(@as(usize, 6), strategy.currentLen());

    // Reset.
    strategy.reset();
    try std.testing.expectEqual(@as(usize, 0), strategy.currentLen());
}

// ------------------------------------------------------------------
// CacheManager tests
// ------------------------------------------------------------------

test "CacheManager multi-layer" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = kvcache.LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = 16,
        .dtype = .float32,
    };

    var manager = try kvcache.CacheManager.init(
        allocator,
        4, // 4 layers
        config,
        kvcache.createStandard,
        stream,
    );
    defer manager.deinit();

    try std.testing.expectEqual(@as(usize, 4), manager.numLayers());
    try std.testing.expectEqual(@as(usize, 0), manager.maxSeqLen());

    // Simulate forward pass on all layers.
    const kv = try makeTestKV(allocator, stream);
    defer kv.keys.deinit();
    defer kv.values.deinit();

    for (0..manager.numLayers()) |i| {
        const result = try manager.caches[i].updateAndFetch(kv.keys, kv.values, stream);
        defer result.keys.deinit();
        defer result.values.deinit();
    }

    try std.testing.expectEqual(@as(usize, 3), manager.maxSeqLen());

    // Reset all.
    manager.resetAll();
    try std.testing.expectEqual(@as(usize, 0), manager.maxSeqLen());
}

// ------------------------------------------------------------------
// RotatingKVCache tests
// ------------------------------------------------------------------

test "RotatingKVCache window bounded" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = kvcache.LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = 16,
        .dtype = .float32,
    };

    const cache_ptr = try allocator.create(kvcache.RotatingKVCache);
    defer allocator.destroy(cache_ptr);
    cache_ptr.* = try kvcache.RotatingKVCache.init(allocator, config, 4, stream);
    defer {
        cache_ptr.keys.deinit();
        cache_ptr.values.deinit();
    }

    var strategy = cache_ptr.asStrategy();

    const kv = try makeTestKV(allocator, stream);
    defer kv.keys.deinit();
    defer kv.values.deinit();

    // Append 3 tokens.
    const r1 = try strategy.updateAndFetch(kv.keys, kv.values, stream);
    defer r1.keys.deinit();
    defer r1.values.deinit();
    try std.testing.expectEqual(@as(usize, 3), strategy.currentLen());

    // Append another 3 (total 6, but window is 4).
    const r2 = try strategy.updateAndFetch(kv.keys, kv.values, stream);
    defer r2.keys.deinit();
    defer r2.values.deinit();

    // After window fills, length should be capped at window_size.
    // Note: currentLen returns min(total_tokens, window_size).
    // After second update of 3: total = 6, window = 4 -> len = 4.
    try std.testing.expectEqual(@as(usize, 4), strategy.currentLen());
}

// ------------------------------------------------------------------
// Strategy polymorphism test
// ------------------------------------------------------------------

test "KVCacheStrategy polymorphism" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = kvcache.LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = 16,
        .dtype = .float32,
    };

    // Create two different strategies and verify they both work through the same interface.
    const strategies = try allocator.alloc(kvcache.KVCacheStrategy, 2);
    defer allocator.free(strategies);

    strategies[0] = try kvcache.createStandard(allocator, config, stream);
    defer strategies[0].deinit(allocator);

    strategies[1] = try kvcache.createRotatingWithWindow(allocator, config, 8, stream);
    defer strategies[1].deinit(allocator);

    const kv = try makeTestKV(allocator, stream);
    defer kv.keys.deinit();
    defer kv.values.deinit();

    for (strategies) |strategy| {
        const result = try strategy.updateAndFetch(kv.keys, kv.values, stream);
        defer result.keys.deinit();
        defer result.values.deinit();

        try std.testing.expectEqual(@as(usize, 3), strategy.currentLen());
        strategy.reset();
        try std.testing.expectEqual(@as(usize, 0), strategy.currentLen());
    }
}

// ------------------------------------------------------------------
// BlockManager tests
// ------------------------------------------------------------------

const BlockManager = kvcache.BlockManager;

test "BlockManager: init creates correct number of free blocks" {
    const allocator = std.testing.allocator;
    var bm = try BlockManager.init(allocator, 8);
    defer bm.deinit();

    try std.testing.expectEqual(@as(usize, 8), bm.total_blocks);
    try std.testing.expectEqual(@as(usize, 8), bm.freeCount());
    try std.testing.expectEqual(@as(usize, 0), bm.usedCount());
    try std.testing.expect(bm.canAllocate(8));
    try std.testing.expect(!bm.canAllocate(9));
}

test "BlockManager: allocateBlocks reduces free pool and tracks ownership" {
    const allocator = std.testing.allocator;
    var bm = try BlockManager.init(allocator, 8);
    defer bm.deinit();

    const blocks = try bm.allocateBlocks(1, 3);
    defer allocator.free(blocks);

    try std.testing.expectEqual(@as(usize, 3), blocks.len);
    try std.testing.expectEqual(@as(usize, 5), bm.freeCount());
    try std.testing.expectEqual(@as(usize, 3), bm.usedCount());

    // Each allocated block should be marked used with ref_count 1.
    for (blocks) |bid| {
        const block = bm.block_pool.items[bid];
        try std.testing.expect(block.used);
        try std.testing.expectEqual(@as(usize, 1), block.ref_count);
    }
}

test "BlockManager: allocateBlocks fails when insufficient blocks" {
    const allocator = std.testing.allocator;
    var bm = try BlockManager.init(allocator, 2);
    defer bm.deinit();

    const result = bm.allocateBlocks(1, 3);
    try std.testing.expectError(error.InsufficientBlocks, result);
}

test "BlockManager: freeBlocks returns blocks to free pool" {
    const allocator = std.testing.allocator;
    var bm = try BlockManager.init(allocator, 8);
    defer bm.deinit();

    const blocks = try bm.allocateBlocks(1, 3);
    allocator.free(blocks);

    try std.testing.expectEqual(@as(usize, 5), bm.freeCount());

    bm.freeBlocks(1);

    try std.testing.expectEqual(@as(usize, 8), bm.freeCount());
    try std.testing.expectEqual(@as(usize, 0), bm.usedCount());
}

test "BlockManager: freeBlocks for unknown req_id is a no-op" {
    const allocator = std.testing.allocator;
    var bm = try BlockManager.init(allocator, 4);
    defer bm.deinit();

    // Should not panic or error.
    bm.freeBlocks(999);
    try std.testing.expectEqual(@as(usize, 4), bm.freeCount());
}

test "BlockManager: multiple requests tracked independently" {
    const allocator = std.testing.allocator;
    var bm = try BlockManager.init(allocator, 10);
    defer bm.deinit();

    const b1 = try bm.allocateBlocks(1, 3);
    defer allocator.free(b1);
    const b2 = try bm.allocateBlocks(2, 4);
    defer allocator.free(b2);

    try std.testing.expectEqual(@as(usize, 3), bm.freeCount());

    // Free request 1 — should free 3 blocks.
    bm.freeBlocks(1);
    try std.testing.expectEqual(@as(usize, 6), bm.freeCount());

    // Free request 2 — should free 4 blocks.
    bm.freeBlocks(2);
    try std.testing.expectEqual(@as(usize, 10), bm.freeCount());
}

test "BlockManager: canAllocate reflects current state" {
    const allocator = std.testing.allocator;
    var bm = try BlockManager.init(allocator, 4);
    defer bm.deinit();

    try std.testing.expect(bm.canAllocate(4));
    try std.testing.expect(!bm.canAllocate(5));

    const b = try bm.allocateBlocks(1, 2);
    allocator.free(b);

    try std.testing.expect(bm.canAllocate(2));
    try std.testing.expect(!bm.canAllocate(3));

    bm.freeBlocks(1);
    try std.testing.expect(bm.canAllocate(4));
}

test "BlockManager: copyOnWrite with ref_count 1 returns same block" {
    const allocator = std.testing.allocator;
    var bm = try BlockManager.init(allocator, 4);
    defer bm.deinit();

    const blocks = try bm.allocateBlocks(1, 1);
    defer allocator.free(blocks);

    const bid = blocks[0];
    try std.testing.expectEqual(@as(usize, 1), bm.block_pool.items[bid].ref_count);

    // CoW on a block with ref_count=1 should return the same block.
    const cow_id = try bm.copyOnWrite(bid);
    try std.testing.expectEqual(bid, cow_id);
}

test "BlockManager: copyOnWrite with ref_count > 1 creates a copy" {
    const allocator = std.testing.allocator;
    var bm = try BlockManager.init(allocator, 4);
    defer bm.deinit();

    const blocks = try bm.allocateBlocks(1, 1);
    defer allocator.free(blocks);

    const bid = blocks[0];

    // Simulate sharing: bump ref_count.
    bm.block_pool.items[bid].ref_count = 2;

    const cow_id = try bm.copyOnWrite(bid);

    // Should be a different block.
    try std.testing.expect(cow_id != bid);

    // Original ref_count decremented.
    try std.testing.expectEqual(@as(usize, 1), bm.block_pool.items[bid].ref_count);

    // New block has ref_count 1.
    try std.testing.expectEqual(@as(usize, 1), bm.block_pool.items[cow_id].ref_count);
    try std.testing.expect(bm.block_pool.items[cow_id].used);
}

test "BlockManager: copyOnWrite fails when no free blocks" {
    const allocator = std.testing.allocator;
    var bm = try BlockManager.init(allocator, 1);
    defer bm.deinit();

    const blocks = try bm.allocateBlocks(1, 1);
    defer allocator.free(blocks);

    // Simulate sharing.
    bm.block_pool.items[blocks[0]].ref_count = 2;

    // No free blocks for the copy.
    const result = bm.copyOnWrite(blocks[0]);
    try std.testing.expectError(error.InsufficientBlocks, result);
}

test "BlockManager: freeBlocks decrements shared block ref_count instead of freeing" {
    const allocator = std.testing.allocator;
    var bm = try BlockManager.init(allocator, 4);
    defer bm.deinit();

    // Allocate for req 1.
    const b1 = try bm.allocateBlocks(1, 1);
    defer allocator.free(b1);
    const bid = b1[0];

    // Simulate sharing: allocate same block for req 2 by manually adding.
    bm.block_pool.items[bid].ref_count = 2;
    const gop = try bm.req_to_blocks.getOrPut(2);
    if (!gop.found_existing) {
        gop.value_ptr.* = std.ArrayList(usize).empty;
    }
    try gop.value_ptr.append(allocator, bid);

    const free_before = bm.freeCount();

    // Free req 2 — should decrement ref_count, not return to free pool.
    bm.freeBlocks(2);

    try std.testing.expectEqual(@as(usize, 1), bm.block_pool.items[bid].ref_count);
    try std.testing.expect(bm.block_pool.items[bid].used);
    // Free count should not change since block is still in use.
    try std.testing.expectEqual(free_before, bm.freeCount());

    // Now free req 1 — ref_count goes to 0, block returned to free pool.
    bm.freeBlocks(1);
    try std.testing.expectEqual(@as(usize, 0), bm.block_pool.items[bid].ref_count);
    try std.testing.expect(!bm.block_pool.items[bid].used);
    try std.testing.expectEqual(free_before + 1, bm.freeCount());
}

test "BlockManager: block conservation — free + used == total" {
    const allocator = std.testing.allocator;
    var bm = try BlockManager.init(allocator, 10);
    defer bm.deinit();

    // Initial: all free.
    try std.testing.expectEqual(bm.total_blocks, bm.freeCount() + bm.usedCount());

    const b1 = try bm.allocateBlocks(1, 3);
    allocator.free(b1);
    try std.testing.expectEqual(bm.total_blocks, bm.freeCount() + bm.usedCount());

    const b2 = try bm.allocateBlocks(2, 5);
    allocator.free(b2);
    try std.testing.expectEqual(bm.total_blocks, bm.freeCount() + bm.usedCount());

    bm.freeBlocks(1);
    try std.testing.expectEqual(bm.total_blocks, bm.freeCount() + bm.usedCount());

    bm.freeBlocks(2);
    try std.testing.expectEqual(bm.total_blocks, bm.freeCount() + bm.usedCount());
}

test "BlockManager: hashBlock is deterministic" {
    const tokens = &[_]u32{ 10, 20, 30, 40 };

    const h1 = BlockManager.hashBlock(0, tokens);
    const h2 = BlockManager.hashBlock(0, tokens);
    try std.testing.expectEqual(h1, h2);

    // Different prev_hash → different result.
    const h3 = BlockManager.hashBlock(42, tokens);
    try std.testing.expect(h1 != h3);

    // Different tokens → different result.
    const other_tokens = &[_]u32{ 10, 20, 30, 41 };
    const h4 = BlockManager.hashBlock(0, other_tokens);
    try std.testing.expect(h1 != h4);
}

test "BlockManager: findCachedPrefix finds matching blocks" {
    const allocator = std.testing.allocator;
    var bm = try BlockManager.init(allocator, 8);
    defer bm.deinit();

    const block_size: usize = 4;
    const tokens = &[_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };

    // Allocate and register blocks for the first request.
    const b1 = try bm.allocateBlocks(1, 2);
    defer allocator.free(b1);

    // Register hashes for the two blocks.
    const hash0 = BlockManager.hashBlock(0, tokens[0..block_size]);
    bm.block_pool.items[b1[0]].tokens_used = block_size;
    try bm.registerBlockHash(b1[0], hash0);

    const hash1 = BlockManager.hashBlock(hash0, tokens[block_size .. 2 * block_size]);
    bm.block_pool.items[b1[1]].tokens_used = block_size;
    try bm.registerBlockHash(b1[1], hash1);

    // Now search for the same prefix — should find both blocks.
    const cached = try bm.findCachedPrefix(tokens, block_size);
    defer allocator.free(cached);

    try std.testing.expectEqual(@as(usize, 2), cached.len);
    try std.testing.expectEqual(b1[0], cached[0]);
    try std.testing.expectEqual(b1[1], cached[1]);
}

test "BlockManager: findCachedPrefix returns empty for no match" {
    const allocator = std.testing.allocator;
    var bm = try BlockManager.init(allocator, 4);
    defer bm.deinit();

    const tokens = &[_]u32{ 10, 20, 30, 40 };
    const cached = try bm.findCachedPrefix(tokens, 4);
    defer allocator.free(cached);

    try std.testing.expectEqual(@as(usize, 0), cached.len);
}

test "BlockManager: findCachedPrefix with zero block_size returns empty" {
    const allocator = std.testing.allocator;
    var bm = try BlockManager.init(allocator, 4);
    defer bm.deinit();

    const tokens = &[_]u32{ 1, 2, 3 };
    const cached = try bm.findCachedPrefix(tokens, 0);
    defer allocator.free(cached);

    try std.testing.expectEqual(@as(usize, 0), cached.len);
}

// ============================================================
// Property-Based Test
// (Property 8)
//
// Feature: production-deployment, Property 8: Block Conservation
//
// For any sequence of block allocation and deallocation operations
// on the BlockManager, the sum of free blocks and used blocks SHALL
// always equal the total block count. When a request completes and
// its blocks are freed, those blocks SHALL appear in the free pool.
//
// **Validates: Requirements R9.3, R9.4, R9.5, R10.1, R10.2, R10.3, R10.5**
// ============================================================

test "Property 8: Block Conservation — free + used == total after random alloc/free sequences (100 iterations)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xBEEF_CAFE);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // Random total_blocks between 4 and 64.
        const total_blocks = rand.intRangeAtMost(usize, 4, 64);
        var bm = try BlockManager.init(allocator, total_blocks);
        defer bm.deinit();

        // Invariant: must hold at init.
        try std.testing.expectEqual(total_blocks, bm.freeCount() + bm.usedCount());

        // Track active request IDs so we can free them later.
        var active_reqs = std.ArrayList(u64).empty;
        defer active_reqs.deinit(allocator);

        // Track allocated slices so we can free the allocator memory.
        var alloc_slices = std.ArrayList([]usize).empty;
        defer {
            for (alloc_slices.items) |s| allocator.free(s);
            alloc_slices.deinit(allocator);
        }

        var next_req_id: u64 = 1;

        // Perform a random number of operations (10–30 per iteration).
        const num_ops = rand.intRangeAtMost(usize, 10, 30);
        for (0..num_ops) |_| {
            // Decide: allocate (0) or free (1). Bias toward alloc when few
            // active requests, toward free when many.
            const do_free = active_reqs.items.len > 0 and
                (rand.intRangeAtMost(u8, 0, 3) == 0 or bm.freeCount() == 0);

            if (do_free) {
                // Pick a random active request to free.
                const idx = rand.intRangeLessThan(usize, 0, active_reqs.items.len);
                const req_id = active_reqs.items[idx];

                // Record free count before freeing.
                const free_before = bm.freeCount();

                // Count how many exclusively-owned blocks this request has
                // (ref_count == 1). Shared blocks (ref_count > 1) will only
                // have their ref_count decremented, not returned to free pool.
                var exclusively_owned: usize = 0;
                if (bm.req_to_blocks.get(req_id)) |block_list| {
                    for (block_list.items) |block_id| {
                        if (block_id < bm.block_pool.items.len) {
                            if (bm.block_pool.items[block_id].ref_count == 1) {
                                exclusively_owned += 1;
                            }
                        }
                    }
                }

                bm.freeBlocks(req_id);

                // Freed blocks with ref_count==1 should now be in the free pool.
                try std.testing.expectEqual(free_before + exclusively_owned, bm.freeCount());

                // Remove from active list.
                _ = active_reqs.swapRemove(idx);
            } else {
                // Allocate: pick a random block count that fits.
                const max_alloc = @min(bm.freeCount(), 8);
                if (max_alloc == 0) continue;
                const num_blocks = rand.intRangeAtMost(usize, 1, max_alloc);

                const req_id = next_req_id;
                next_req_id += 1;

                const blocks = try bm.allocateBlocks(req_id, num_blocks);
                try alloc_slices.append(allocator, blocks);
                try active_reqs.append(allocator, req_id);

                // Verify each allocated block is marked used with ref_count 1.
                for (blocks) |bid| {
                    try std.testing.expect(bm.block_pool.items[bid].used);
                    try std.testing.expectEqual(@as(usize, 1), bm.block_pool.items[bid].ref_count);
                }
            }

            // CORE INVARIANT: free + used == total after every operation.
            try std.testing.expectEqual(total_blocks, bm.freeCount() + bm.usedCount());
        }

        // Free all remaining active requests.
        for (active_reqs.items) |req_id| {
            bm.freeBlocks(req_id);
            // Invariant must hold after each free.
            try std.testing.expectEqual(total_blocks, bm.freeCount() + bm.usedCount());
        }

        // After freeing everything, all blocks should be free.
        try std.testing.expectEqual(total_blocks, bm.freeCount());
        try std.testing.expectEqual(@as(usize, 0), bm.usedCount());
    }
}

test "Property 8: Block Conservation — conservation holds with copyOnWrite operations (100 iterations)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xDEAD_BEEF);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // Random total_blocks between 6 and 32 (need headroom for CoW copies).
        const total_blocks = rand.intRangeAtMost(usize, 6, 32);
        var bm = try BlockManager.init(allocator, total_blocks);
        defer bm.deinit();

        try std.testing.expectEqual(total_blocks, bm.freeCount() + bm.usedCount());

        var alloc_slices = std.ArrayList([]usize).empty;
        defer {
            for (alloc_slices.items) |s| allocator.free(s);
            alloc_slices.deinit(allocator);
        }

        // Allocate blocks for two requests that will share blocks.
        const max_alloc = @min(bm.freeCount() / 2, 4);
        if (max_alloc == 0) continue;
        const num_blocks = rand.intRangeAtMost(usize, 1, max_alloc);

        const b1 = try bm.allocateBlocks(1, num_blocks);
        try alloc_slices.append(allocator, b1);
        try std.testing.expectEqual(total_blocks, bm.freeCount() + bm.usedCount());

        // Simulate sharing: bump ref_count on request 1's blocks and
        // register them under request 2 as well.
        const gop = try bm.req_to_blocks.getOrPut(2);
        if (!gop.found_existing) {
            gop.value_ptr.* = std.ArrayList(usize).empty;
        }
        for (b1) |bid| {
            bm.block_pool.items[bid].ref_count += 1;
            try gop.value_ptr.append(allocator, bid);
        }

        // Conservation still holds (shared blocks are still "used").
        try std.testing.expectEqual(total_blocks, bm.freeCount() + bm.usedCount());

        // Perform CoW on a random shared block (if free blocks available).
        if (bm.freeCount() > 0) {
            const cow_idx = rand.intRangeLessThan(usize, 0, b1.len);
            const cow_bid = b1[cow_idx];
            const new_bid = try bm.copyOnWrite(cow_bid);

            // CoW should produce a different block since ref_count > 1.
            try std.testing.expect(new_bid != cow_bid);

            // Original ref_count decremented, new block has ref_count 1.
            try std.testing.expect(bm.block_pool.items[cow_bid].ref_count >= 1);
            try std.testing.expectEqual(@as(usize, 1), bm.block_pool.items[new_bid].ref_count);

            // Conservation invariant after CoW.
            try std.testing.expectEqual(total_blocks, bm.freeCount() + bm.usedCount());
        }

        // Free request 2 (shared blocks get ref_count decremented).
        bm.freeBlocks(2);
        try std.testing.expectEqual(total_blocks, bm.freeCount() + bm.usedCount());

        // Free request 1 (blocks returned to free pool).
        bm.freeBlocks(1);
        try std.testing.expectEqual(total_blocks, bm.freeCount() + bm.usedCount());
    }
}

// ============================================================
// Property-Based Test
// (Property 9)
//
// Feature: production-deployment, Property 9: Copy-on-Write Isolation
//
// For any Block shared by two or more requests (ref_count > 1),
// when one request modifies the block, the Block_Manager SHALL
// create a copy before mutation. The other request's view of the
// block SHALL remain unchanged.
//
// **Validates: Requirements R10.4, R15.3**
// ============================================================

test "Property 9: Copy-on-Write Isolation — modifying a shared block creates a copy; other request's view unchanged (100 iterations)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0DE_1500);
    const rand = prng.random();

    const stream = c.c.mlx_default_cpu_stream_new();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // Random total_blocks between 4 and 32 (need headroom for CoW copies).
        const total_blocks = rand.intRangeAtMost(usize, 4, 32);
        var bm = try BlockManager.init(allocator, total_blocks);
        defer bm.deinit();

        // Allocate blocks for request 1 (1 to min(total_blocks/2, 4) blocks).
        const max_alloc = @min(total_blocks / 2, 4);
        if (max_alloc == 0) continue;
        const num_blocks = rand.intRangeAtMost(usize, 1, max_alloc);

        const req1_blocks = try bm.allocateBlocks(1, num_blocks);
        defer allocator.free(req1_blocks);

        // Assign random KV data to each allocated block using mlx random arrays.
        // Shape: [1, 2, 4, 8] — small tensors for testing.
        const shape = &[_]i32{ 1, 2, 4, 8 };
        for (req1_blocks) |bid| {
            var keys_arr = c.c.mlx_array_new();
            var values_arr = c.c.mlx_array_new();
            try c.check(c.c.mlx_random_normal(
                &keys_arr,
                shape.ptr,
                shape.len,
                c.c.MLX_FLOAT32,
                @as(f32, @floatFromInt(iteration * 1000 + bid)),
                1.0,
                .{ .ctx = null },
                stream,
            ));
            try c.check(c.c.mlx_random_normal(
                &values_arr,
                shape.ptr,
                shape.len,
                c.c.MLX_FLOAT32,
                @as(f32, @floatFromInt(iteration * 1000 + bid + 500)),
                1.0,
                .{ .ctx = null },
                stream,
            ));
            bm.block_pool.items[bid].keys = Array.fromHandle(keys_arr);
            bm.block_pool.items[bid].values = Array.fromHandle(values_arr);
            bm.block_pool.items[bid].tokens_used = 4;
        }

        // Simulate sharing: bump ref_count on all of request 1's blocks
        // and register them under request 2 as well.
        const gop = try bm.req_to_blocks.getOrPut(2);
        if (!gop.found_existing) {
            gop.value_ptr.* = std.ArrayList(usize).empty;
        }
        for (req1_blocks) |bid| {
            bm.block_pool.items[bid].ref_count += 1;
            try gop.value_ptr.append(allocator, bid);
        }

        // Pick a random shared block to perform CoW on.
        const cow_idx = rand.intRangeLessThan(usize, 0, req1_blocks.len);
        const original_bid = req1_blocks[cow_idx];

        // Snapshot the original block's key/value data for later comparison.
        // We evaluate the arrays to get concrete data before CoW.
        const orig_block = &bm.block_pool.items[original_bid];
        const orig_ref_count_before = orig_block.ref_count;

        // Verify the block is indeed shared (ref_count > 1).
        try std.testing.expect(orig_ref_count_before > 1);

        // Evaluate original keys/values to get concrete data for comparison.
        if (orig_block.keys) |k| {
            try c.check(c.c.mlx_array_eval(k.inner));
        }
        if (orig_block.values) |v| {
            try c.check(c.c.mlx_array_eval(v.inner));
        }

        // Perform Copy-on-Write.
        const new_bid = try bm.copyOnWrite(original_bid);

        // === Verify CoW created a separate copy ===

        // 1. New block ID must differ from original.
        try std.testing.expect(new_bid != original_bid);

        // 2. Original block's ref_count should be decremented by 1.
        try std.testing.expectEqual(orig_ref_count_before - 1, bm.block_pool.items[original_bid].ref_count);

        // 3. New block should have ref_count == 1.
        try std.testing.expectEqual(@as(usize, 1), bm.block_pool.items[new_bid].ref_count);

        // 4. New block should be marked as used.
        try std.testing.expect(bm.block_pool.items[new_bid].used);

        // 5. New block should have same tokens_used as original.
        try std.testing.expectEqual(
            bm.block_pool.items[original_bid].tokens_used,
            bm.block_pool.items[new_bid].tokens_used,
        );

        // 6. Verify the original block's data is unchanged (other request's view).
        //    The original block should still have its keys and values arrays.
        try std.testing.expect(bm.block_pool.items[original_bid].keys != null);
        try std.testing.expect(bm.block_pool.items[original_bid].values != null);

        // 7. Verify the new block has its own copy of keys and values.
        try std.testing.expect(bm.block_pool.items[new_bid].keys != null);
        try std.testing.expect(bm.block_pool.items[new_bid].values != null);

        // 8. Verify the new block's arrays are distinct handles from the original.
        try std.testing.expect(
            bm.block_pool.items[new_bid].keys.?.inner.ctx != bm.block_pool.items[original_bid].keys.?.inner.ctx,
        );
        try std.testing.expect(
            bm.block_pool.items[new_bid].values.?.inner.ctx != bm.block_pool.items[original_bid].values.?.inner.ctx,
        );

        // 9. Evaluate the new block's arrays and verify data matches original.
        const new_block = &bm.block_pool.items[new_bid];
        if (new_block.keys) |k| {
            try c.check(c.c.mlx_array_eval(k.inner));
        }
        if (new_block.values) |v| {
            try c.check(c.c.mlx_array_eval(v.inner));
        }

        // Compare key data element-wise: original and copy should be equal.
        if (bm.block_pool.items[original_bid].keys) |orig_k| {
            if (bm.block_pool.items[new_bid].keys) |new_k| {
                // Use mlx array_equal to verify data equality.
                var eq_arr = c.c.mlx_array_new();
                defer _ = c.c.mlx_array_free(eq_arr);
                try c.check(c.c.mlx_array_equal(&eq_arr, orig_k.inner, new_k.inner, false, stream));
                var all_eq = c.c.mlx_array_new();
                defer _ = c.c.mlx_array_free(all_eq);
                try c.check(c.c.mlx_all_axes(&all_eq, eq_arr, null, 0, false, stream));
                try c.check(c.c.mlx_array_eval(all_eq));
                var eq_val: bool = false;
                try c.check(c.c.mlx_array_item_bool(&eq_val, all_eq));
                try std.testing.expect(eq_val);
            }
        }

        // Compare value data element-wise.
        if (bm.block_pool.items[original_bid].values) |orig_v| {
            if (bm.block_pool.items[new_bid].values) |new_v| {
                var eq_arr = c.c.mlx_array_new();
                defer _ = c.c.mlx_array_free(eq_arr);
                try c.check(c.c.mlx_array_equal(&eq_arr, orig_v.inner, new_v.inner, false, stream));
                var all_eq = c.c.mlx_array_new();
                defer _ = c.c.mlx_array_free(all_eq);
                try c.check(c.c.mlx_all_axes(&all_eq, eq_arr, null, 0, false, stream));
                try c.check(c.c.mlx_array_eval(all_eq));
                var eq_val: bool = false;
                try c.check(c.c.mlx_array_item_bool(&eq_val, all_eq));
                try std.testing.expect(eq_val);
            }
        }

        // 10. Block conservation invariant still holds.
        try std.testing.expectEqual(total_blocks, bm.freeCount() + bm.usedCount());

        // Cleanup: free both requests.
        bm.freeBlocks(2);
        bm.freeBlocks(1);
    }
}

test "Property 9: Copy-on-Write Isolation — modifying copied block does not affect original (100 iterations)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0DE_AAAA);
    const rand = prng.random();

    const stream = c.c.mlx_default_cpu_stream_new();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // Random total_blocks between 4 and 24.
        const total_blocks = rand.intRangeAtMost(usize, 4, 24);
        var bm = try BlockManager.init(allocator, total_blocks);
        defer bm.deinit();

        // Allocate 1 block for request 1.
        const req1_blocks = try bm.allocateBlocks(1, 1);
        defer allocator.free(req1_blocks);
        const original_bid = req1_blocks[0];

        // Assign known key/value data to the block.
        const shape = &[_]i32{ 1, 2, 4, 8 };
        var keys_arr = c.c.mlx_array_new();
        var values_arr = c.c.mlx_array_new();
        // Use ones so we have a known baseline value.
        try c.check(c.c.mlx_ones(&keys_arr, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
        try c.check(c.c.mlx_ones(&values_arr, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
        try c.check(c.c.mlx_array_eval(keys_arr));
        try c.check(c.c.mlx_array_eval(values_arr));

        bm.block_pool.items[original_bid].keys = Array.fromHandle(keys_arr);
        bm.block_pool.items[original_bid].values = Array.fromHandle(values_arr);
        bm.block_pool.items[original_bid].tokens_used = 4;

        // Simulate sharing: bump ref_count and register under request 2.
        bm.block_pool.items[original_bid].ref_count += 1;
        const gop = try bm.req_to_blocks.getOrPut(2);
        if (!gop.found_existing) {
            gop.value_ptr.* = std.ArrayList(usize).empty;
        }
        try gop.value_ptr.append(allocator, original_bid);

        // Perform CoW — request 2 wants to modify the shared block.
        const new_bid = try bm.copyOnWrite(original_bid);
        try std.testing.expect(new_bid != original_bid);

        // Now "modify" the copied block by replacing its keys with a different value.
        // This simulates what a request would do after CoW.
        const new_block = &bm.block_pool.items[new_bid];
        if (new_block.keys) |k| k.deinit();
        var modified_keys = c.c.mlx_array_new();
        // Use a scalar multiplied by 2 to create different data.
        const two_scalar = c.c.mlx_array_new_float32(2.0);
        defer _ = c.c.mlx_array_free(two_scalar);
        var ones_arr = c.c.mlx_array_new();
        try c.check(c.c.mlx_ones(&ones_arr, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
        try c.check(c.c.mlx_multiply(&modified_keys, ones_arr, two_scalar, stream));
        _ = c.c.mlx_array_free(ones_arr);
        try c.check(c.c.mlx_array_eval(modified_keys));
        new_block.keys = Array.fromHandle(modified_keys);

        // Verify the ORIGINAL block's keys are still ones (unchanged).
        const orig_keys = bm.block_pool.items[original_bid].keys.?;
        try c.check(c.c.mlx_array_eval(orig_keys.inner));

        // Check that original keys are all 1.0 by summing and comparing.
        var sum_arr = c.c.mlx_array_new();
        defer _ = c.c.mlx_array_free(sum_arr);
        try c.check(c.c.mlx_sum(&sum_arr, orig_keys.inner, false, stream));
        try c.check(c.c.mlx_array_eval(sum_arr));
        var sum_val: f32 = 0;
        try c.check(c.c.mlx_array_item_float32(&sum_val, sum_arr));
        // 1*2*4*8 = 64 elements, all 1.0 → sum should be 64.0
        try std.testing.expectApproxEqAbs(@as(f32, 64.0), sum_val, 0.001);

        // Verify the NEW block's keys are all 2.0 (modified).
        const new_keys = bm.block_pool.items[new_bid].keys.?;
        var new_sum_arr = c.c.mlx_array_new();
        defer _ = c.c.mlx_array_free(new_sum_arr);
        try c.check(c.c.mlx_sum(&new_sum_arr, new_keys.inner, false, stream));
        try c.check(c.c.mlx_array_eval(new_sum_arr));
        var new_sum_val: f32 = 0;
        try c.check(c.c.mlx_array_item_float32(&new_sum_val, new_sum_arr));
        // 64 elements, all 2.0 → sum should be 128.0
        try std.testing.expectApproxEqAbs(@as(f32, 128.0), new_sum_val, 0.001);

        // Block conservation invariant.
        try std.testing.expectEqual(total_blocks, bm.freeCount() + bm.usedCount());

        // Cleanup.
        bm.freeBlocks(2);
        bm.freeBlocks(1);
    }
}

// ------------------------------------------------------------------
// QuantizedKVCache tests
// ------------------------------------------------------------------

/// Create a KV tensor of shape [B=1, H=2, S=seq_len, D=64] with random data.
/// head_dim=64 is required for quantization (must be divisible by group_size).
fn makeQuantTestKV(seq_len: usize, stream: c.c.mlx_stream) !struct { keys: Array, values: Array } {
    const shape = &[_]i32{ 1, 2, @intCast(seq_len), 64 };

    var keys = c.c.mlx_array_new();
    var values = c.c.mlx_array_new();

    try c.check(c.c.mlx_random_normal(
        &keys,
        shape.ptr,
        shape.len,
        c.c.MLX_FLOAT16,
        0.0,
        1.0,
        .{ .ctx = null },
        stream,
    ));
    try c.check(c.c.mlx_random_normal(
        &values,
        shape.ptr,
        shape.len,
        c.c.MLX_FLOAT16,
        0.0,
        1.0,
        .{ .ctx = null },
        stream,
    ));

    return .{
        .keys = Array.fromHandle(keys),
        .values = Array.fromHandle(values),
    };
}

test "QuantizedKVCache: 16-bit passthrough stores and retrieves without quantization" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = kvcache.LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 64,
        .max_seq_len = 32,
        .dtype = .float16,
    };

    const cache_ptr = try allocator.create(kvcache.QuantizedKVCache);
    cache_ptr.* = try kvcache.QuantizedKVCache.init(allocator, config, 64, 16, stream);
    var strategy = cache_ptr.asStrategy();
    defer strategy.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), strategy.currentLen());

    const kv = try makeQuantTestKV(4, stream);
    defer kv.keys.deinit();
    defer kv.values.deinit();

    const result = try strategy.updateAndFetch(kv.keys, kv.values, stream);
    defer result.keys.deinit();
    defer result.values.deinit();

    try std.testing.expectEqual(@as(usize, 4), strategy.currentLen());

    // Verify output shape: [1, 2, 4, 64]
    try std.testing.expectEqual(@as(i32, 1), result.keys.shape()[0]);
    try std.testing.expectEqual(@as(i32, 2), result.keys.shape()[1]);
    try std.testing.expectEqual(@as(i32, 4), result.keys.shape()[2]);
    try std.testing.expectEqual(@as(i32, 64), result.keys.shape()[3]);
}

test "QuantizedKVCache: 8-bit quantize and dequantize round-trip" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = kvcache.LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 64,
        .max_seq_len = 32,
        .dtype = .float16,
    };

    const cache_ptr = try allocator.create(kvcache.QuantizedKVCache);
    cache_ptr.* = try kvcache.QuantizedKVCache.init(allocator, config, 64, 8, stream);
    var strategy = cache_ptr.asStrategy();
    defer strategy.deinit(allocator);

    const kv = try makeQuantTestKV(4, stream);
    defer kv.keys.deinit();
    defer kv.values.deinit();

    const result = try strategy.updateAndFetch(kv.keys, kv.values, stream);
    defer result.keys.deinit();
    defer result.values.deinit();

    try std.testing.expectEqual(@as(usize, 4), strategy.currentLen());

    // Verify output shape matches input shape.
    try std.testing.expectEqual(@as(i32, 1), result.keys.shape()[0]);
    try std.testing.expectEqual(@as(i32, 2), result.keys.shape()[1]);
    try std.testing.expectEqual(@as(i32, 4), result.keys.shape()[2]);
    try std.testing.expectEqual(@as(i32, 64), result.keys.shape()[3]);
}

test "QuantizedKVCache: 4-bit quantize and dequantize round-trip" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = kvcache.LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 64,
        .max_seq_len = 32,
        .dtype = .float16,
    };

    const cache_ptr = try allocator.create(kvcache.QuantizedKVCache);
    cache_ptr.* = try kvcache.QuantizedKVCache.init(allocator, config, 64, 4, stream);
    var strategy = cache_ptr.asStrategy();
    defer strategy.deinit(allocator);

    const kv = try makeQuantTestKV(4, stream);
    defer kv.keys.deinit();
    defer kv.values.deinit();

    const result = try strategy.updateAndFetch(kv.keys, kv.values, stream);
    defer result.keys.deinit();
    defer result.values.deinit();

    try std.testing.expectEqual(@as(usize, 4), strategy.currentLen());
    try std.testing.expectEqual(@as(i32, 64), result.keys.shape()[3]);
}

test "QuantizedKVCache: multiple updates accumulate sequence length" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = kvcache.LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 64,
        .max_seq_len = 64,
        .dtype = .float16,
    };

    const cache_ptr = try allocator.create(kvcache.QuantizedKVCache);
    cache_ptr.* = try kvcache.QuantizedKVCache.init(allocator, config, 64, 8, stream);
    var strategy = cache_ptr.asStrategy();
    defer strategy.deinit(allocator);

    // First update: 4 tokens.
    const kv1 = try makeQuantTestKV(4, stream);
    defer kv1.keys.deinit();
    defer kv1.values.deinit();
    const r1 = try strategy.updateAndFetch(kv1.keys, kv1.values, stream);
    defer r1.keys.deinit();
    defer r1.values.deinit();
    try std.testing.expectEqual(@as(usize, 4), strategy.currentLen());

    // Second update: 2 more tokens.
    const kv2 = try makeQuantTestKV(2, stream);
    defer kv2.keys.deinit();
    defer kv2.values.deinit();
    const r2 = try strategy.updateAndFetch(kv2.keys, kv2.values, stream);
    defer r2.keys.deinit();
    defer r2.values.deinit();
    try std.testing.expectEqual(@as(usize, 6), strategy.currentLen());

    // Returned keys should have seq_len=6.
    try std.testing.expectEqual(@as(i32, 6), r2.keys.shape()[2]);
}

test "QuantizedKVCache: reset clears sequence length" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = kvcache.LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 64,
        .max_seq_len = 32,
        .dtype = .float16,
    };

    const cache_ptr = try allocator.create(kvcache.QuantizedKVCache);
    cache_ptr.* = try kvcache.QuantizedKVCache.init(allocator, config, 64, 8, stream);
    var strategy = cache_ptr.asStrategy();
    defer strategy.deinit(allocator);

    const kv = try makeQuantTestKV(4, stream);
    defer kv.keys.deinit();
    defer kv.values.deinit();
    const r = try strategy.updateAndFetch(kv.keys, kv.values, stream);
    defer r.keys.deinit();
    defer r.values.deinit();

    try std.testing.expectEqual(@as(usize, 4), strategy.currentLen());
    strategy.reset();
    try std.testing.expectEqual(@as(usize, 0), strategy.currentLen());
}

test "QuantizedKVCache: overflow returns error" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = kvcache.LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 64,
        .max_seq_len = 4,
        .dtype = .float16,
    };

    const cache_ptr = try allocator.create(kvcache.QuantizedKVCache);
    cache_ptr.* = try kvcache.QuantizedKVCache.init(allocator, config, 64, 16, stream);
    var strategy = cache_ptr.asStrategy();
    defer strategy.deinit(allocator);

    // Try to insert 8 tokens into a cache with max_seq_len=4.
    const kv = try makeQuantTestKV(8, stream);
    defer kv.keys.deinit();
    defer kv.values.deinit();

    const result = strategy.updateAndFetch(kv.keys, kv.values, stream);
    try std.testing.expectError(error.CacheOverflow, result);
}

test "QuantizedKVCache: invalid kv_bits returns error" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = kvcache.LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 64,
        .max_seq_len = 32,
        .dtype = .float16,
    };

    const result = kvcache.QuantizedKVCache.init(allocator, config, 64, 3, stream);
    try std.testing.expectError(error.InvalidQuantBits, result);
}

test "QuantizedKVCache: factory functions create correct strategies" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();

    const config = kvcache.LayerConfig{
        .batch_size = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 64,
        .max_seq_len = 32,
        .dtype = .float16,
    };

    // Test all three factory functions.
    var s4 = try kvcache.createQuantized4Bit(allocator, config, stream);
    defer s4.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 0), s4.currentLen());

    var s8 = try kvcache.createQuantized8Bit(allocator, config, stream);
    defer s8.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 0), s8.currentLen());

    var s16 = try kvcache.createQuantized16Bit(allocator, config, stream);
    defer s16.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 0), s16.currentLen());
}

test "QuantizedKVCache: QuantConfig validation" {
    // Valid configs.
    try (QuantConfig{ .kv_bits = 4, .group_size = 64 }).validate();
    try (QuantConfig{ .kv_bits = 8, .group_size = 32 }).validate();
    try (QuantConfig{ .kv_bits = 16, .group_size = 64 }).validate();

    // Invalid bits.
    try std.testing.expectError(
        error.InvalidQuantBits,
        (QuantConfig{ .kv_bits = 3, .group_size = 64 }).validate(),
    );
    try std.testing.expectError(
        error.InvalidQuantBits,
        (QuantConfig{ .kv_bits = 32, .group_size = 64 }).validate(),
    );

    // Invalid group_size.
    try std.testing.expectError(
        error.InvalidGroupSize,
        (QuantConfig{ .kv_bits = 8, .group_size = 0 }).validate(),
    );
    try std.testing.expectError(
        error.InvalidGroupSize,
        (QuantConfig{ .kv_bits = 8, .group_size = -1 }).validate(),
    );
}

const QuantConfig = kvcache.QuantConfig;

// ============================================================
// Property-Based Test
// (Property 10)
//
// Feature: production-deployment, Property 10: Quantize-Dequantize Round-Trip
//
// For any tensor (KV cache entry) and for any valid bit-width (4 or 8),
// quantizing with mlx_quantize and then dequantizing with mlx_dequantize
// SHALL produce a tensor with cosine similarity ≥ 0.99 to the original
// for 8-bit and ≥ 0.95 for 4-bit.
//
// **Validates: Requirements R11.2, R11.3, R18.1, R18.2**
// ============================================================

/// Compute cosine similarity between two arrays using raw mlx-c calls.
/// Both arrays are flattened to 1D, cast to float32, then:
///   cos_sim = sum(a*b) / (sqrt(sum(a*a)) * sqrt(sum(b*b)))
fn computeCosineSimilarity(a: c.c.mlx_array, b: c.c.mlx_array, stream: c.c.mlx_stream) !f32 {
    // Cast both to float32 for numerical stability.
    var a_f32 = c.c.mlx_array_new();
    defer _ = c.c.mlx_array_free(a_f32);
    var b_f32 = c.c.mlx_array_new();
    defer _ = c.c.mlx_array_free(b_f32);
    try c.check(c.c.mlx_astype(&a_f32, a, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_astype(&b_f32, b, c.c.MLX_FLOAT32, stream));

    // Flatten both to 1D.
    var a_flat = c.c.mlx_array_new();
    defer _ = c.c.mlx_array_free(a_flat);
    var b_flat = c.c.mlx_array_new();
    defer _ = c.c.mlx_array_free(b_flat);
    try c.check(c.c.mlx_flatten(&a_flat, a_f32, 0, 3, stream));
    try c.check(c.c.mlx_flatten(&b_flat, b_f32, 0, 3, stream));

    // dot = sum(a * b)
    var ab = c.c.mlx_array_new();
    defer _ = c.c.mlx_array_free(ab);
    try c.check(c.c.mlx_multiply(&ab, a_flat, b_flat, stream));
    var dot = c.c.mlx_array_new();
    defer _ = c.c.mlx_array_free(dot);
    try c.check(c.c.mlx_sum(&dot, ab, false, stream));

    // norm_a = sqrt(sum(a * a))
    var aa = c.c.mlx_array_new();
    defer _ = c.c.mlx_array_free(aa);
    try c.check(c.c.mlx_multiply(&aa, a_flat, a_flat, stream));
    var sum_aa = c.c.mlx_array_new();
    defer _ = c.c.mlx_array_free(sum_aa);
    try c.check(c.c.mlx_sum(&sum_aa, aa, false, stream));
    var norm_a = c.c.mlx_array_new();
    defer _ = c.c.mlx_array_free(norm_a);
    try c.check(c.c.mlx_sqrt(&norm_a, sum_aa, stream));

    // norm_b = sqrt(sum(b * b))
    var bb = c.c.mlx_array_new();
    defer _ = c.c.mlx_array_free(bb);
    try c.check(c.c.mlx_multiply(&bb, b_flat, b_flat, stream));
    var sum_bb = c.c.mlx_array_new();
    defer _ = c.c.mlx_array_free(sum_bb);
    try c.check(c.c.mlx_sum(&sum_bb, bb, false, stream));
    var norm_b = c.c.mlx_array_new();
    defer _ = c.c.mlx_array_free(norm_b);
    try c.check(c.c.mlx_sqrt(&norm_b, sum_bb, stream));

    // denom = norm_a * norm_b
    var denom = c.c.mlx_array_new();
    defer _ = c.c.mlx_array_free(denom);
    try c.check(c.c.mlx_multiply(&denom, norm_a, norm_b, stream));

    // cos_sim = dot / denom
    var cos_sim = c.c.mlx_array_new();
    defer _ = c.c.mlx_array_free(cos_sim);
    try c.check(c.c.mlx_divide(&cos_sim, dot, denom, stream));

    try c.check(c.c.mlx_array_eval(cos_sim));
    var result: f32 = 0;
    try c.check(c.c.mlx_array_item_float32(&result, cos_sim));
    return result;
}

/// Create a KV tensor of shape [B=1, H=num_kv_heads, S=seq_len, D=64] with random float16 data.
fn makeRandomQuantKV(num_kv_heads: usize, seq_len: usize, stream: c.c.mlx_stream) !struct { keys: Array, values: Array } {
    const shape = &[_]i32{ 1, @intCast(num_kv_heads), @intCast(seq_len), 64 };

    var keys = c.c.mlx_array_new();
    var values = c.c.mlx_array_new();

    try c.check(c.c.mlx_random_normal(
        &keys,
        shape.ptr,
        shape.len,
        c.c.MLX_FLOAT16,
        0.0,
        1.0,
        .{ .ctx = null },
        stream,
    ));
    try c.check(c.c.mlx_random_normal(
        &values,
        shape.ptr,
        shape.len,
        c.c.MLX_FLOAT16,
        0.0,
        1.0,
        .{ .ctx = null },
        stream,
    ));

    return .{
        .keys = Array.fromHandle(keys),
        .values = Array.fromHandle(values),
    };
}

test "Property 10: Quantize-Dequantize Round-Trip — 8-bit cosine similarity >= 0.99 (100 iterations)" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();
    var prng = std.Random.DefaultPrng.init(0xABCD_0008);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // Random num_kv_heads: 1..4, random seq_len: 1..16.
        const num_kv_heads = rand.intRangeAtMost(usize, 1, 4);
        const seq_len = rand.intRangeAtMost(usize, 1, 16);

        const config = kvcache.LayerConfig{
            .batch_size = 1,
            .num_heads = 4,
            .num_kv_heads = num_kv_heads,
            .head_dim = 64,
            .max_seq_len = 256,
            .dtype = .float16,
        };

        const cache_ptr = try allocator.create(kvcache.QuantizedKVCache);
        cache_ptr.* = try kvcache.QuantizedKVCache.init(allocator, config, 64, 8, stream);
        var strategy = cache_ptr.asStrategy();
        defer strategy.deinit(allocator);

        // Generate random KV tensors.
        const kv = try makeRandomQuantKV(num_kv_heads, seq_len, stream);
        defer kv.keys.deinit();
        defer kv.values.deinit();

        // Evaluate originals so we have concrete data to compare.
        try c.check(c.c.mlx_array_eval(kv.keys.inner));
        try c.check(c.c.mlx_array_eval(kv.values.inner));

        // Store (quantize) and retrieve (dequantize).
        const result = try strategy.updateAndFetch(kv.keys, kv.values, stream);
        defer result.keys.deinit();
        defer result.values.deinit();

        // Compute cosine similarity for keys.
        const keys_sim = try computeCosineSimilarity(kv.keys.inner, result.keys.inner, stream);
        if (keys_sim < 0.99) {
            std.debug.print(
                "FAIL iter={d}: 8-bit keys cosine_sim={d:.6} < 0.99 (heads={d}, seq={d})\n",
                .{ iteration, keys_sim, num_kv_heads, seq_len },
            );
            return error.TestUnexpectedResult;
        }

        // Compute cosine similarity for values.
        const values_sim = try computeCosineSimilarity(kv.values.inner, result.values.inner, stream);
        if (values_sim < 0.99) {
            std.debug.print(
                "FAIL iter={d}: 8-bit values cosine_sim={d:.6} < 0.99 (heads={d}, seq={d})\n",
                .{ iteration, values_sim, num_kv_heads, seq_len },
            );
            return error.TestUnexpectedResult;
        }
    }
}

test "Property 10: Quantize-Dequantize Round-Trip — 4-bit cosine similarity >= 0.95 (100 iterations)" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();
    var prng = std.Random.DefaultPrng.init(0xABCD_0004);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // Random num_kv_heads: 1..4, random seq_len: 1..16.
        const num_kv_heads = rand.intRangeAtMost(usize, 1, 4);
        const seq_len = rand.intRangeAtMost(usize, 1, 16);

        const config = kvcache.LayerConfig{
            .batch_size = 1,
            .num_heads = 4,
            .num_kv_heads = num_kv_heads,
            .head_dim = 64,
            .max_seq_len = 256,
            .dtype = .float16,
        };

        const cache_ptr = try allocator.create(kvcache.QuantizedKVCache);
        cache_ptr.* = try kvcache.QuantizedKVCache.init(allocator, config, 64, 4, stream);
        var strategy = cache_ptr.asStrategy();
        defer strategy.deinit(allocator);

        // Generate random KV tensors.
        const kv = try makeRandomQuantKV(num_kv_heads, seq_len, stream);
        defer kv.keys.deinit();
        defer kv.values.deinit();

        // Evaluate originals so we have concrete data to compare.
        try c.check(c.c.mlx_array_eval(kv.keys.inner));
        try c.check(c.c.mlx_array_eval(kv.values.inner));

        // Store (quantize) and retrieve (dequantize).
        const result = try strategy.updateAndFetch(kv.keys, kv.values, stream);
        defer result.keys.deinit();
        defer result.values.deinit();

        // Compute cosine similarity for keys.
        const keys_sim = try computeCosineSimilarity(kv.keys.inner, result.keys.inner, stream);
        if (keys_sim < 0.95) {
            std.debug.print(
                "FAIL iter={d}: 4-bit keys cosine_sim={d:.6} < 0.95 (heads={d}, seq={d})\n",
                .{ iteration, keys_sim, num_kv_heads, seq_len },
            );
            return error.TestUnexpectedResult;
        }

        // Compute cosine similarity for values.
        const values_sim = try computeCosineSimilarity(kv.values.inner, result.values.inner, stream);
        if (values_sim < 0.95) {
            std.debug.print(
                "FAIL iter={d}: 4-bit values cosine_sim={d:.6} < 0.95 (heads={d}, seq={d})\n",
                .{ iteration, values_sim, num_kv_heads, seq_len },
            );
            return error.TestUnexpectedResult;
        }
    }
}

// ============================================================
// Property-Based Test
// (Property 13)
//
// Feature: production-deployment, Property 13: Block Hash Determinism and Prefix Reuse
//
// For any sequence of token IDs, hashBlock(prev_hash, token_ids) SHALL be
// deterministic — the same inputs always produce the same hash. When two
// requests share a token prefix that aligns to block boundaries, the second
// request SHALL reuse the first request's cached blocks for the shared prefix.
//
// **Validates: Requirements R15.1, R15.2**
// ============================================================

test "Property 13: Block Hash Determinism — same inputs always produce same hash (100 iterations)" {
    var prng = std.Random.DefaultPrng.init(0xB10C_4A5E);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // Generate a random prev_hash.
        const prev_hash = rand.int(u64);

        // Generate a random token sequence of length 1..32.
        const token_len = rand.intRangeAtMost(usize, 1, 32);
        var tokens: [32]u32 = undefined;
        for (0..token_len) |i| {
            tokens[i] = rand.int(u32);
        }
        const token_slice = tokens[0..token_len];

        // Hash the same inputs twice — must produce identical results.
        const hash1 = BlockManager.hashBlock(prev_hash, token_slice);
        const hash2 = BlockManager.hashBlock(prev_hash, token_slice);
        try std.testing.expectEqual(hash1, hash2);

        // Hash a third time to be thorough.
        const hash3 = BlockManager.hashBlock(prev_hash, token_slice);
        try std.testing.expectEqual(hash1, hash3);

        // Verify different prev_hash produces a different hash (with overwhelming probability).
        const other_prev = prev_hash +% 1;
        const hash_other_prev = BlockManager.hashBlock(other_prev, token_slice);
        // Extremely unlikely to collide — assert they differ.
        try std.testing.expect(hash1 != hash_other_prev);

        // Verify different tokens produce a different hash (with overwhelming probability).
        if (token_len > 0) {
            var modified_tokens: [32]u32 = undefined;
            @memcpy(modified_tokens[0..token_len], token_slice);
            modified_tokens[0] = modified_tokens[0] +% 1;
            const hash_other_tokens = BlockManager.hashBlock(prev_hash, modified_tokens[0..token_len]);
            try std.testing.expect(hash1 != hash_other_tokens);
        }

        // Verify chained hashing is deterministic: hash(hash(0, block0), block1) is stable.
        if (token_len >= 2) {
            const mid = token_len / 2;
            const block0 = token_slice[0..mid];
            const block1 = token_slice[mid..token_len];

            const chain_hash_a = BlockManager.hashBlock(BlockManager.hashBlock(prev_hash, block0), block1);
            const chain_hash_b = BlockManager.hashBlock(BlockManager.hashBlock(prev_hash, block0), block1);
            try std.testing.expectEqual(chain_hash_a, chain_hash_b);
        }
    }
}

test "Property 13: Prefix Reuse — second request reuses first request's cached blocks for shared prefix (100 iterations)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xCAFE_1BCA);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // Random block_size: 2..8 tokens per block.
        const block_size = rand.intRangeAtMost(usize, 2, 8);

        // Random number of shared prefix blocks: 1..4.
        const num_prefix_blocks = rand.intRangeAtMost(usize, 1, 4);

        // Total blocks needed: prefix blocks + some extra for divergent suffix.
        const total_blocks = num_prefix_blocks + 8; // headroom
        var bm = try BlockManager.init(allocator, total_blocks);
        defer bm.deinit();

        // Generate random token sequence for the shared prefix.
        const prefix_len = num_prefix_blocks * block_size;
        const max_tokens = prefix_len + block_size; // prefix + one divergent block
        var token_buf: [40]u32 = undefined; // max 4*8 + 8 = 40
        for (0..max_tokens) |i| {
            token_buf[i] = rand.int(u32);
        }
        const prefix_tokens = token_buf[0..prefix_len];
        const full_tokens = token_buf[0..max_tokens];

        // --- First request: allocate blocks and register hashes ---
        const req1_blocks = try bm.allocateBlocks(1, num_prefix_blocks);
        defer allocator.free(req1_blocks);

        var prev_hash: u64 = 0;
        for (0..num_prefix_blocks) |i| {
            const block_start = i * block_size;
            const block_tokens = prefix_tokens[block_start .. block_start + block_size];
            const hash = BlockManager.hashBlock(prev_hash, block_tokens);

            bm.block_pool.items[req1_blocks[i]].tokens_used = block_size;
            try bm.registerBlockHash(req1_blocks[i], hash);

            prev_hash = hash;
        }

        // --- Second request: findCachedPrefix should reuse all prefix blocks ---
        const cached = try bm.findCachedPrefix(full_tokens, block_size);
        defer allocator.free(cached);

        // Must find exactly num_prefix_blocks cached blocks.
        try std.testing.expectEqual(num_prefix_blocks, cached.len);

        // Each cached block ID must match the first request's block IDs.
        for (0..num_prefix_blocks) |i| {
            try std.testing.expectEqual(req1_blocks[i], cached[i]);
        }

        // Verify ref_count was incremented for reused blocks.
        for (cached) |bid| {
            // Original alloc gave ref_count=1, findCachedPrefix incremented to 2.
            try std.testing.expectEqual(@as(usize, 2), bm.block_pool.items[bid].ref_count);
        }

        // Block conservation invariant still holds.
        try std.testing.expectEqual(total_blocks, bm.freeCount() + bm.usedCount());

        // Cleanup: free both requests' references.
        // Decrement ref_counts for the cached prefix (simulating req 2 freeing).
        for (cached) |bid| {
            bm.block_pool.items[bid].ref_count -= 1;
        }
        bm.freeBlocks(1);
    }
}
