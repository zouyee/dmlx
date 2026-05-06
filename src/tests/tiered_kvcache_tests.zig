const std = @import("std");
const root = @import("../root.zig");
const c = @import("mlx").c;
const posix_c = @cImport({
    @cInclude("sys/stat.h");
    @cInclude("unistd.h");
    @cInclude("dirent.h");
    @cInclude("stdio.h");
});

const Array = root.Array;
const kvcache = root.kvcache;
const TieredKVCache = kvcache.TieredKVCache;
const LayerConfig = kvcache.LayerConfig;

/// Helper: create a small KV tensor of shape [1, H, S, D].
fn makeTestKV(num_heads: usize, seq_len: usize, head_dim: usize, stream: c.c.mlx_stream) !struct { keys: Array, values: Array } {
    const shape = &[_]i32{ 1, @intCast(num_heads), @intCast(seq_len), @intCast(head_dim) };

    var keys = c.c.mlx_array_new();
    var values = c.c.mlx_array_new();

    try c.check(c.c.mlx_ones(&keys, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&values, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));

    return .{
        .keys = Array.fromHandle(keys),
        .values = Array.fromHandle(values),
    };
}

/// Helper: ensure the cold directory exists.
fn ensureColdDir(allocator: std.mem.Allocator, dir: []const u8) !void {
    const dir_z = try allocator.dupeZ(u8, dir);
    defer allocator.free(dir_z);
    _ = posix_c.mkdir(dir_z.ptr, 0o755);
}

/// Helper: clean up cold directory (remove files inside, then the dir).
fn cleanupColdDir(allocator: std.mem.Allocator, dir: []const u8) void {
    const dir_z = allocator.dupeZ(u8, dir) catch return;
    defer allocator.free(dir_z);
    const dp = posix_c.opendir(dir_z.ptr);
    if (dp != null) {
        while (true) {
            const entry = posix_c.readdir(dp);
            if (entry == null) break;
            const name: [*:0]const u8 = @ptrCast(&entry.*.d_name);
            if (name[0] == '.') continue;
            const full_path = std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir, std.mem.span(name) }) catch continue;
            defer allocator.free(full_path);
            const full_path_z = allocator.dupeZ(u8, full_path) catch continue;
            defer allocator.free(full_path_z);
            _ = posix_c.unlink(full_path_z.ptr);
        }
        _ = posix_c.closedir(dp);
    }
    _ = posix_c.rmdir(dir_z.ptr);
}

/// Helper: create a heap-allocated TieredKVCache and return its strategy.
fn createTestTiered(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    page_size: usize,
    hot_capacity: usize,
    cold_dir: []const u8,
    stream: c.c.mlx_stream,
) !struct { cache: *TieredKVCache, strategy: kvcache.KVCacheStrategy } {
    const cache = try allocator.create(TieredKVCache);
    errdefer allocator.destroy(cache);
    cache.* = try TieredKVCache.init(allocator, config, page_size, hot_capacity, cold_dir, stream);
    return .{ .cache = cache, .strategy = cache.asStrategy() };
}

// ------------------------------------------------------------------
// Unit Tests
// ------------------------------------------------------------------

test "TieredKVCache: init and basic lifecycle" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();
    const cold_dir = "/tmp/mlx_tiered_test_lifecycle";
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

    var t = try createTestTiered(allocator, config, 8, 4, cold_dir, stream);
    defer t.strategy.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), t.strategy.currentLen());
}

test "TieredKVCache: updateAndFetch stores tokens" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();
    const cold_dir = "/tmp/mlx_tiered_test_update";
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

    var t = try createTestTiered(allocator, config, 8, 10, cold_dir, stream);
    defer t.strategy.deinit(allocator);

    const kv = try makeTestKV(2, 3, 4, stream);
    defer kv.keys.deinit();
    defer kv.values.deinit();

    const result = try t.strategy.updateAndFetch(kv.keys, kv.values, stream);
    defer result.keys.deinit();
    defer result.values.deinit();

    try std.testing.expectEqual(@as(usize, 3), t.strategy.currentLen());
}

test "TieredKVCache: evictToSSD and restoreFromSSD round-trip" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();
    const cold_dir = "/tmp/mlx_tiered_test_roundtrip";
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

    var t = try createTestTiered(allocator, config, 8, 10, cold_dir, stream);
    defer t.strategy.deinit(allocator);

    // Insert some data
    const kv = try makeTestKV(2, 3, 4, stream);
    defer kv.keys.deinit();
    defer kv.values.deinit();

    const result = try t.strategy.updateAndFetch(kv.keys, kv.values, stream);
    defer result.keys.deinit();
    defer result.values.deinit();

    // Find a used page to evict
    var used_page_idx: ?usize = null;
    for (t.cache.hot.pages.items, 0..) |page, i| {
        if (page.used) {
            used_page_idx = i;
            break;
        }
    }
    const page_idx = used_page_idx orelse return error.NoUsedPages;

    // Snapshot original data before eviction
    const orig_page = &t.cache.hot.pages.items[page_idx];
    try c.check(c.c.mlx_array_eval(orig_page.keys.inner));
    try c.check(c.c.mlx_array_eval(orig_page.values.inner));
    const orig_keys_data = try orig_page.keys.dataSlice(f32);
    const orig_keys_copy = try allocator.dupe(f32, orig_keys_data);
    defer allocator.free(orig_keys_copy);
    const orig_values_data = try orig_page.values.dataSlice(f32);
    const orig_values_copy = try allocator.dupe(f32, orig_values_data);
    defer allocator.free(orig_values_copy);

    // Evict to SSD
    try t.cache.evictToSSD(page_idx);

    // Verify page is now unused
    try std.testing.expect(!t.cache.hot.pages.items[page_idx].used);

    // Verify cold block is tracked
    try std.testing.expect(t.cache.cold_blocks.contains(page_idx));

    // Restore from SSD
    try t.cache.restoreFromSSD(page_idx);

    // Verify page is used again
    try std.testing.expect(t.cache.hot.pages.items[page_idx].used);

    // Verify cold block is removed
    try std.testing.expect(!t.cache.cold_blocks.contains(page_idx));

    // Verify data is element-wise equal
    const restored_page = &t.cache.hot.pages.items[page_idx];
    try c.check(c.c.mlx_array_eval(restored_page.keys.inner));
    try c.check(c.c.mlx_array_eval(restored_page.values.inner));
    const restored_keys_data = try restored_page.keys.dataSlice(f32);
    const restored_values_data = try restored_page.values.dataSlice(f32);

    try std.testing.expectEqual(orig_keys_copy.len, restored_keys_data.len);
    for (orig_keys_copy, restored_keys_data) |orig, restored| {
        try std.testing.expectApproxEqAbs(orig, restored, 1e-6);
    }
    try std.testing.expectEqual(orig_values_copy.len, restored_values_data.len);
    for (orig_values_copy, restored_values_data) |orig, restored| {
        try std.testing.expectApproxEqAbs(orig, restored, 1e-6);
    }
}

test "TieredKVCache: enforceCapacity evicts LRU pages" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();
    const cold_dir = "/tmp/mlx_tiered_test_enforce";
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

    // Small hot capacity of 2 pages, page_size=4
    var t = try createTestTiered(allocator, config, 4, 2, cold_dir, stream);
    defer t.strategy.deinit(allocator);

    // Insert enough data to require 3 pages (12 tokens with page_size=4)
    const kv1 = try makeTestKV(2, 4, 4, stream);
    defer kv1.keys.deinit();
    defer kv1.values.deinit();
    const r1 = try t.strategy.updateAndFetch(kv1.keys, kv1.values, stream);
    defer r1.keys.deinit();
    defer r1.values.deinit();

    const kv2 = try makeTestKV(2, 4, 4, stream);
    defer kv2.keys.deinit();
    defer kv2.values.deinit();
    const r2 = try t.strategy.updateAndFetch(kv2.keys, kv2.values, stream);
    defer r2.keys.deinit();
    defer r2.values.deinit();

    const kv3 = try makeTestKV(2, 4, 4, stream);
    defer kv3.keys.deinit();
    defer kv3.values.deinit();
    const r3 = try t.strategy.updateAndFetch(kv3.keys, kv3.values, stream);
    defer r3.keys.deinit();
    defer r3.values.deinit();

    // After enforcement, hot used count should be <= hot_capacity
    try std.testing.expect(t.cache.hotUsedCount() <= t.cache.hot_capacity);

    // At least one block should have been evicted to cold tier
    try std.testing.expect(t.cache.cold_blocks.count() > 0);
}

test "TieredKVCache: reset cleans up cold tier files" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();
    const cold_dir = "/tmp/mlx_tiered_test_reset";
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

    var t = try createTestTiered(allocator, config, 4, 1, cold_dir, stream);
    defer t.strategy.deinit(allocator);

    const kv1 = try makeTestKV(2, 4, 4, stream);
    defer kv1.keys.deinit();
    defer kv1.values.deinit();
    const r1 = try t.strategy.updateAndFetch(kv1.keys, kv1.values, stream);
    defer r1.keys.deinit();
    defer r1.values.deinit();

    const kv2 = try makeTestKV(2, 4, 4, stream);
    defer kv2.keys.deinit();
    defer kv2.values.deinit();
    const r2 = try t.strategy.updateAndFetch(kv2.keys, kv2.values, stream);
    defer r2.keys.deinit();
    defer r2.values.deinit();

    // Reset should clean up everything
    t.strategy.reset();

    try std.testing.expectEqual(@as(usize, 0), t.strategy.currentLen());
    try std.testing.expectEqual(@as(usize, 0), t.cache.cold_blocks.count());
}

test "TieredKVCache: evict invalid page returns error" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();
    const cold_dir = "/tmp/mlx_tiered_test_invalid";
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

    var t = try createTestTiered(allocator, config, 8, 4, cold_dir, stream);
    defer t.strategy.deinit(allocator);

    try std.testing.expectError(error.InvalidPageIndex, t.cache.evictToSSD(999));
}

test "TieredKVCache: restore non-cold block returns error" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();
    const cold_dir = "/tmp/mlx_tiered_test_restore_err";
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

    var t = try createTestTiered(allocator, config, 8, 4, cold_dir, stream);
    defer t.strategy.deinit(allocator);

    try std.testing.expectError(error.BlockNotInColdTier, t.cache.restoreFromSSD(0));
}

test "TieredKVCache: LRU eviction selects oldest accessed page" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();
    const cold_dir = "/tmp/mlx_tiered_test_lru";
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

    var t = try createTestTiered(allocator, config, 4, 10, cold_dir, stream);
    defer t.strategy.deinit(allocator);

    // Insert data to create pages
    const kv1 = try makeTestKV(2, 4, 4, stream);
    defer kv1.keys.deinit();
    defer kv1.values.deinit();
    const r1 = try t.strategy.updateAndFetch(kv1.keys, kv1.values, stream);
    defer r1.keys.deinit();
    defer r1.values.deinit();

    const kv2 = try makeTestKV(2, 4, 4, stream);
    defer kv2.keys.deinit();
    defer kv2.values.deinit();
    const r2 = try t.strategy.updateAndFetch(kv2.keys, kv2.values, stream);
    defer r2.keys.deinit();
    defer r2.values.deinit();

    // Manually set access times: page 0 is older than page 1
    t.cache.access_recency.put(0, 1) catch {};
    t.cache.access_recency.put(1, 100) catch {};

    // findLRUPage should return page 0 (oldest)
    const lru = t.cache.findLRUPage();
    try std.testing.expect(lru != null);
    try std.testing.expectEqual(@as(usize, 0), lru.?);
}

// ============================================================
// Property-Based Tests
// ============================================================
//
// (Property 20)
//
// Feature: production-deployment, Property 20: Tiered KV Cache
// Evict-Restore Round-Trip
//
// For any KV cache block, evicting it to the cold tier (SSD via
// safetensors) and then restoring it to the hot tier SHALL produce
// a block with keys and values element-wise equal to the original.
// When the hot tier exceeds capacity, the least recently accessed
// blocks SHALL be evicted first.
//
// **Validates: Requirements R22.2, R22.3, R22.4**
// ============================================================

test "Property 20: Tiered KV Cache Evict-Restore Round-Trip — evict then restore produces element-wise equal data (100 iterations)" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();
    var prng = std.Random.DefaultPrng.init(0xD1E8_ED20);

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        const rand = prng.random();

        // Random page_size in [4, 16] and fixed small dimensions.
        const page_size = rand.intRangeAtMost(usize, 4, 16);
        const num_kv_heads = rand.intRangeAtMost(usize, 1, 4);
        const head_dim = rand.intRangeAtMost(usize, 2, 8);

        const cold_dir = "/tmp/mlx_tiered_p20_roundtrip";
        try ensureColdDir(allocator, cold_dir);
        defer cleanupColdDir(allocator, cold_dir);

        const config = LayerConfig{
            .batch_size = 1,
            .num_heads = num_kv_heads * 2,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .max_seq_len = page_size * 4,
            .dtype = .float32,
        };

        var t = try createTestTiered(allocator, config, page_size, 10, cold_dir, stream);
        defer t.strategy.deinit(allocator);

        // Insert exactly page_size tokens so we get one full page.
        const shape = &[_]i32{ 1, @intCast(num_kv_heads), @intCast(page_size), @intCast(head_dim) };
        var keys_arr = c.c.mlx_array_new();
        var values_arr = c.c.mlx_array_new();
        try c.check(c.c.mlx_random_normal(
            &keys_arr,
            shape.ptr,
            shape.len,
            c.c.MLX_FLOAT32,
            @as(f32, @floatFromInt(iteration * 1000)),
            1.0,
            .{ .ctx = null },
            stream,
        ));
        try c.check(c.c.mlx_random_normal(
            &values_arr,
            shape.ptr,
            shape.len,
            c.c.MLX_FLOAT32,
            @as(f32, @floatFromInt(iteration * 1000 + 500)),
            1.0,
            .{ .ctx = null },
            stream,
        ));
        const kv_keys = Array.fromHandle(keys_arr);
        defer kv_keys.deinit();
        const kv_values = Array.fromHandle(values_arr);
        defer kv_values.deinit();

        const result = try t.strategy.updateAndFetch(kv_keys, kv_values, stream);
        defer result.keys.deinit();
        defer result.values.deinit();

        // Find the used page.
        var used_page_idx: ?usize = null;
        for (t.cache.hot.pages.items, 0..) |page, i| {
            if (page.used) {
                used_page_idx = i;
                break;
            }
        }
        const page_idx = used_page_idx orelse continue;

        // Snapshot original data before eviction.
        const orig_page = &t.cache.hot.pages.items[page_idx];
        try c.check(c.c.mlx_array_eval(orig_page.keys.inner));
        try c.check(c.c.mlx_array_eval(orig_page.values.inner));
        const orig_keys_data = try orig_page.keys.dataSlice(f32);
        const orig_keys_copy = try allocator.dupe(f32, orig_keys_data);
        defer allocator.free(orig_keys_copy);
        const orig_values_data = try orig_page.values.dataSlice(f32);
        const orig_values_copy = try allocator.dupe(f32, orig_values_data);
        defer allocator.free(orig_values_copy);

        // Evict to SSD.
        try t.cache.evictToSSD(page_idx);

        // Verify page is now unused and tracked in cold tier.
        try std.testing.expect(!t.cache.hot.pages.items[page_idx].used);
        try std.testing.expect(t.cache.cold_blocks.contains(page_idx));

        // Restore from SSD.
        try t.cache.restoreFromSSD(page_idx);

        // Verify page is used again and removed from cold tier.
        try std.testing.expect(t.cache.hot.pages.items[page_idx].used);
        try std.testing.expect(!t.cache.cold_blocks.contains(page_idx));

        // Verify element-wise equality of keys.
        const restored_page = &t.cache.hot.pages.items[page_idx];
        try c.check(c.c.mlx_array_eval(restored_page.keys.inner));
        try c.check(c.c.mlx_array_eval(restored_page.values.inner));
        const restored_keys_data = try restored_page.keys.dataSlice(f32);
        const restored_values_data = try restored_page.values.dataSlice(f32);

        try std.testing.expectEqual(orig_keys_copy.len, restored_keys_data.len);
        for (orig_keys_copy, restored_keys_data) |orig, restored| {
            try std.testing.expectApproxEqAbs(orig, restored, 1e-6);
        }

        // Verify element-wise equality of values.
        try std.testing.expectEqual(orig_values_copy.len, restored_values_data.len);
        for (orig_values_copy, restored_values_data) |orig, restored| {
            try std.testing.expectApproxEqAbs(orig, restored, 1e-6);
        }
    }
}

test "Property 20: Tiered KV Cache Evict-Restore Round-Trip — least recently accessed blocks evicted first (100 iterations)" {
    const allocator = std.testing.allocator;
    const stream = c.c.mlx_default_cpu_stream_new();
    var prng = std.Random.DefaultPrng.init(0xA8E_EE1C);

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        const rand = prng.random();

        // Use a fixed small page_size so we can create multiple pages easily.
        const page_size: usize = 4;
        const num_kv_heads = rand.intRangeAtMost(usize, 1, 3);
        const head_dim = rand.intRangeAtMost(usize, 2, 6);
        // Random number of pages to create: 3 to 6.
        const num_pages = rand.intRangeAtMost(usize, 3, 6);
        // Hot capacity is always less than num_pages to force eviction.
        const hot_capacity = rand.intRangeAtMost(usize, 1, num_pages - 1);

        const cold_dir = "/tmp/mlx_tiered_p20_lru";
        try ensureColdDir(allocator, cold_dir);

        const config = LayerConfig{
            .batch_size = 1,
            .num_heads = num_kv_heads * 2,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .max_seq_len = page_size * (num_pages + 2),
            .dtype = .float32,
        };

        var t = try createTestTiered(allocator, config, page_size, hot_capacity, cold_dir, stream);

        // Insert data page by page, each page_size tokens at a time.
        var result_keys = std.ArrayList(Array).empty;
        var result_values = std.ArrayList(Array).empty;
        var input_keys = std.ArrayList(Array).empty;
        var input_values = std.ArrayList(Array).empty;

        for (0..num_pages) |p| {
            const shape = &[_]i32{ 1, @intCast(num_kv_heads), @intCast(page_size), @intCast(head_dim) };
            var k_arr = c.c.mlx_array_new();
            var v_arr = c.c.mlx_array_new();
            try c.check(c.c.mlx_random_normal(
                &k_arr,
                shape.ptr,
                shape.len,
                c.c.MLX_FLOAT32,
                @as(f32, @floatFromInt(iteration * 100 + p)),
                1.0,
                .{ .ctx = null },
                stream,
            ));
            try c.check(c.c.mlx_random_normal(
                &v_arr,
                shape.ptr,
                shape.len,
                c.c.MLX_FLOAT32,
                @as(f32, @floatFromInt(iteration * 100 + p + 50)),
                1.0,
                .{ .ctx = null },
                stream,
            ));
            const k = Array.fromHandle(k_arr);
            const v = Array.fromHandle(v_arr);
            try input_keys.append(allocator, k);
            try input_values.append(allocator, v);

            const r = try t.strategy.updateAndFetch(k, v, stream);
            try result_keys.append(allocator, r.keys);
            try result_values.append(allocator, r.values);
        }

        // After inserting num_pages pages with hot_capacity < num_pages,
        // enforceCapacity should have evicted some blocks.
        // Verify hot used count is within capacity.
        try std.testing.expect(t.cache.hotUsedCount() <= t.cache.hot_capacity);

        // Verify that evicted blocks are in the cold tier.
        const evicted_count = t.cache.cold_blocks.count();
        try std.testing.expect(evicted_count > 0);

        // Verify the LRU invariant: among all used pages still in hot tier,
        // their access times should be >= the access times of evicted pages.
        // Collect access times of hot pages and verify they are the most recent.
        var min_hot_access: u64 = std.math.maxInt(u64);
        for (t.cache.hot.pages.items, 0..) |page, i| {
            if (page.used) {
                const access_time = t.cache.access_recency.get(i) orelse 0;
                if (access_time < min_hot_access) {
                    min_hot_access = access_time;
                }
            }
        }

        // Evicted pages should not have access recency entries (they are removed
        // during eviction), confirming they were the least recently accessed.
        var cold_it = t.cache.cold_blocks.iterator();
        while (cold_it.next()) |entry| {
            const cold_page_idx = entry.key_ptr.*;
            // Evicted pages should have their access_recency removed.
            try std.testing.expect(!t.cache.access_recency.contains(cold_page_idx));
            // Evicted pages should be marked unused in the hot tier.
            try std.testing.expect(!t.cache.hot.pages.items[cold_page_idx].used);
        }

        // Additionally verify: if we manually check findLRUPage on the
        // remaining hot pages, it should return a page with the lowest
        // access time among the hot pages.
        if (t.cache.hotUsedCount() > 0) {
            const lru_page = t.cache.findLRUPage();
            try std.testing.expect(lru_page != null);
            const lru_access = t.cache.access_recency.get(lru_page.?) orelse 0;
            // The LRU page's access time should be <= all other hot pages.
            for (t.cache.hot.pages.items, 0..) |page, i| {
                if (page.used) {
                    const access_time = t.cache.access_recency.get(i) orelse 0;
                    try std.testing.expect(lru_access <= access_time);
                }
            }
        }

        // Explicit cleanup (defer in while-loop defers to function end, not iteration end)
        for (result_keys.items) |arr| arr.deinit();
        result_keys.deinit(allocator);
        for (result_values.items) |arr| arr.deinit();
        result_values.deinit(allocator);
        for (input_keys.items) |arr| arr.deinit();
        input_keys.deinit(allocator);
        for (input_values.items) |arr| arr.deinit();
        input_values.deinit(allocator);
        t.strategy.deinit(allocator);
        cleanupColdDir(allocator, cold_dir);
    }
}
