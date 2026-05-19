/// PrefixCacheManager — caches pre-filled KV caches for repeated prompts.
///
/// When a request with a previously-seen prompt arrives, the engine can
/// skip the prefill forward pass by cloning the cached KV caches.
/// This is especially effective for:
///   - Multi-turn chat (system prompt + previous turns are shared)
///   - Repeated queries with the same prefix
///   - Warmup prompts that overlap with real user prompts
const std = @import("std");
const kvcache = @import("../kvcache.zig");

const KVCacheStrategy = kvcache.KVCacheStrategy;

/// A single cached entry: prompt token sequence + pre-filled KV caches.
const CacheEntry = struct {
    prompt_tokens: []u32,
    caches: []KVCacheStrategy,
    last_access: u64, // Monotonic access counter for LRU eviction.
};

/// Hash a token sequence into a u64 key.
fn hashTokens(tokens: []const u32) u64 {
    var hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for (tokens) |t| {
        hash ^= t;
        hash *%= 0x100000001b3; // FNV-1a prime
    }
    return hash;
}

pub const PrefixCacheManager = struct {
    allocator: std.mem.Allocator,
    entries: std.AutoHashMap(u64, CacheEntry),
    max_entries: usize,
    stream: @import("mlx").c.c.mlx_stream,
    access_counter: u64, // Monotonically increasing counter for LRU.
    hits: u64,
    misses: u64,

    pub fn init(allocator: std.mem.Allocator, max_entries: usize, stream: @import("mlx").c.c.mlx_stream) PrefixCacheManager {
        return .{
            .allocator = allocator,
            .entries = std.AutoHashMap(u64, CacheEntry).init(allocator),
            .max_entries = max_entries,
            .stream = stream,
            .access_counter = 0,
            .hits = 0,
            .misses = 0,
        };
    }

    pub fn deinit(self: *PrefixCacheManager) void {
        var it = self.entries.valueIterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.prompt_tokens);
            for (entry.caches) |cache| {
                cache.deinit(self.allocator);
            }
            self.allocator.free(entry.caches);
        }
        self.entries.deinit();
    }

    /// Look up a cached entry by prompt tokens.
    /// Returns a cloned copy of the cached caches (caller owns the clone).
    pub fn lookup(self: *PrefixCacheManager, prompt_tokens: []const u32) !?[]KVCacheStrategy {
        const key = hashTokens(prompt_tokens);
        const entry_ptr = self.entries.getPtr(key) orelse {
            self.misses += 1;
            return null;
        };

        // Verify exact match (hash collision safety).
        if (entry_ptr.prompt_tokens.len != prompt_tokens.len) {
            self.misses += 1;
            return null;
        }
        if (!std.mem.eql(u32, entry_ptr.prompt_tokens, prompt_tokens)) {
            self.misses += 1;
            return null;
        }

        // Clone the cached caches.
        const cloned = try self.allocator.alloc(KVCacheStrategy, entry_ptr.caches.len);
        errdefer self.allocator.free(cloned);

        for (entry_ptr.caches, 0..) |cache, i| {
            if (cache.supportsClone()) {
                const copy = try cache.clone(self.allocator, self.stream);
                if (copy) |c| {
                    cloned[i] = c;
                } else {
                    // Clone failed for this layer — clean up and abort.
                    for (0..i) |j| {
                        cloned[j].deinit(self.allocator);
                    }
                    self.allocator.free(cloned);
                    self.misses += 1;
                    return null;
                }
            } else {
                // Strategy doesn't support cloning — can't use prefix cache.
                for (0..i) |j| {
                    cloned[j].deinit(self.allocator);
                }
                self.allocator.free(cloned);
                self.misses += 1;
                return null;
            }
        }

        // Update LRU access counter.
        self.access_counter += 1;
        entry_ptr.last_access = self.access_counter;
        self.hits += 1;
        return cloned;
    }

    /// Store a completed request's caches into the prefix cache.
    /// The caches are moved into the cache (caller must not deinit them).
    pub fn store(self: *PrefixCacheManager, prompt_tokens: []const u32, caches: []KVCacheStrategy) !void {
        const key = hashTokens(prompt_tokens);

        // If already cached, skip.
        if (self.entries.contains(key)) return;

        // Evict LRU entry if at capacity.
        if (self.entries.count() >= self.max_entries) {
            self.evictLRU();
        }

        // Store the prompt tokens and caches.
        const prompt_copy = try self.allocator.dupe(u32, prompt_tokens);
        errdefer self.allocator.free(prompt_copy);

        self.access_counter += 1;
        try self.entries.put(key, .{
            .prompt_tokens = prompt_copy,
            .caches = caches,
            .last_access = self.access_counter,
        });
    }

    /// Evict the least recently used entry.
    fn evictLRU(self: *PrefixCacheManager) void {
        var min_access: u64 = std.math.maxInt(u64);
        var evict_key: ?u64 = null;

        var it = self.entries.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.last_access < min_access) {
                min_access = kv.value_ptr.last_access;
                evict_key = kv.key_ptr.*;
            }
        }

        if (evict_key) |key| {
            if (self.entries.fetchRemove(key)) |removed| {
                self.allocator.free(removed.value.prompt_tokens);
                for (removed.value.caches) |cache| {
                    cache.deinit(self.allocator);
                }
                self.allocator.free(removed.value.caches);
            }
        }
    }

    /// Check if a prompt is already cached.
    pub fn isCached(self: *PrefixCacheManager, prompt_tokens: []const u32) bool {
        const key = hashTokens(prompt_tokens);
        const entry = self.entries.get(key) orelse return false;
        if (entry.prompt_tokens.len != prompt_tokens.len) return false;
        return std.mem.eql(u32, entry.prompt_tokens, prompt_tokens);
    }

    /// Get cache hit rate as a percentage (0.0-100.0).
    pub fn hitRate(self: *const PrefixCacheManager) f64 {
        const total = self.hits + self.misses;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.hits)) * 100.0 / @as(f64, @floatFromInt(total));
    }

    /// Get the number of entries currently in cache.
    pub fn count(self: *const PrefixCacheManager) usize {
        return self.entries.count();
    }

    /// Clear all cached entries and reset stats.
    pub fn clear(self: *PrefixCacheManager) void {
        var it = self.entries.valueIterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.prompt_tokens);
            for (entry.caches) |cache| {
                cache.deinit(self.allocator);
            }
            self.allocator.free(entry.caches);
        }
        self.entries.clearAndFree();
        self.hits = 0;
        self.misses = 0;
        self.access_counter = 0;
    }
};

// ============================================================
// Tests
// ============================================================

test "PrefixCacheManager basic store and lookup" {
    const allocator = std.testing.allocator;

    // Use a null stream for testing (clone operations won't be called
    // since we test with a mock that returns null on clone).
    var mgr = PrefixCacheManager.init(allocator, 4, std.mem.zeroes(@import("mlx").c.c.mlx_stream));
    defer mgr.deinit();

    // Test isCached on empty cache.
    const tokens = &[_]u32{ 1, 2, 3, 4 };
    try std.testing.expect(!mgr.isCached(tokens));
    try std.testing.expectEqual(@as(usize, 0), mgr.count());
}

test "PrefixCacheManager hash collision safety" {
    // Verify that different token sequences produce different hashes.
    const a = &[_]u32{ 1, 2, 3 };
    const b = &[_]u32{ 3, 2, 1 };
    const c = &[_]u32{ 1, 2, 3, 4 };

    const ha = hashTokens(a);
    const hb = hashTokens(b);
    const hc = hashTokens(c);

    try std.testing.expect(ha != hb);
    try std.testing.expect(ha != hc);
    try std.testing.expect(hb != hc);
}

test "PrefixCacheManager LRU eviction order" {
    const allocator = std.testing.allocator;

    var mgr = PrefixCacheManager.init(allocator, 2, std.mem.zeroes(@import("mlx").c.c.mlx_stream));
    defer mgr.deinit();

    // Store two entries (no caches, just tokens for eviction test).
    const tokens_a = &[_]u32{100};
    const tokens_b = &[_]u32{200};
    const tokens_c = &[_]u32{300};

    // Store empty cache slices (0-length).
    const empty_a = try allocator.alloc(KVCacheStrategy, 0);
    try mgr.store(tokens_a, empty_a);
    try std.testing.expectEqual(@as(usize, 1), mgr.count());

    const empty_b = try allocator.alloc(KVCacheStrategy, 0);
    try mgr.store(tokens_b, empty_b);
    try std.testing.expectEqual(@as(usize, 2), mgr.count());

    // Cache is full. Storing a third entry should evict the LRU (tokens_a).
    const empty_c = try allocator.alloc(KVCacheStrategy, 0);
    try mgr.store(tokens_c, empty_c);
    try std.testing.expectEqual(@as(usize, 2), mgr.count());

    // tokens_a should be evicted (oldest access).
    try std.testing.expect(!mgr.isCached(tokens_a));
    // tokens_b and tokens_c should still be cached.
    try std.testing.expect(mgr.isCached(tokens_b));
    try std.testing.expect(mgr.isCached(tokens_c));
}

test "PrefixCacheManager hit rate tracking" {
    const allocator = std.testing.allocator;

    var mgr = PrefixCacheManager.init(allocator, 4, std.mem.zeroes(@import("mlx").c.c.mlx_stream));
    defer mgr.deinit();

    // Initial hit rate should be 0.
    try std.testing.expectEqual(@as(f64, 0.0), mgr.hitRate());

    // Lookup a non-existent entry (counts as miss).
    const result = try mgr.lookup(&[_]u32{ 1, 2, 3 });
    try std.testing.expect(result == null);
    try std.testing.expectEqual(@as(u64, 0), mgr.hits);
    try std.testing.expectEqual(@as(u64, 1), mgr.misses);
}

test "PrefixCacheManager clear resets state" {
    const allocator = std.testing.allocator;

    var mgr = PrefixCacheManager.init(allocator, 4, std.mem.zeroes(@import("mlx").c.c.mlx_stream));
    defer mgr.deinit();

    const tokens = &[_]u32{ 10, 20, 30 };
    const empty = try allocator.alloc(KVCacheStrategy, 0);
    try mgr.store(tokens, empty);
    try std.testing.expectEqual(@as(usize, 1), mgr.count());

    mgr.clear();
    try std.testing.expectEqual(@as(usize, 0), mgr.count());
    try std.testing.expect(!mgr.isCached(tokens));
    try std.testing.expectEqual(@as(u64, 0), mgr.hits);
    try std.testing.expectEqual(@as(u64, 0), mgr.misses);
}

test "PrefixCacheManager duplicate store is no-op" {
    const allocator = std.testing.allocator;

    var mgr = PrefixCacheManager.init(allocator, 4, std.mem.zeroes(@import("mlx").c.c.mlx_stream));
    defer mgr.deinit();

    const tokens = &[_]u32{ 5, 10, 15 };
    const empty1 = try allocator.alloc(KVCacheStrategy, 0);
    try mgr.store(tokens, empty1);
    try std.testing.expectEqual(@as(usize, 1), mgr.count());

    // Storing same prompt again should be a no-op.
    // Allocate new caches that won't be stored (caller still owns them).
    const empty2 = try allocator.alloc(KVCacheStrategy, 0);
    try mgr.store(tokens, empty2);
    try std.testing.expectEqual(@as(usize, 1), mgr.count());

    // Free the caches that weren't stored (caller still owns them).
    allocator.free(empty2);
}
