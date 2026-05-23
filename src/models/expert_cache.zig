/// LRU cache for expert weight tensors in stream mode MoE inference.
///
/// Stores individual expert weight slices keyed by (layer_index, tensor_name, expert_id),
/// enabling O(1) lookup and LRU eviction when the memory budget is exceeded.
/// This avoids redundant disk reads for experts reused across adjacent token generation steps.
///
/// Design: doubly-linked list threaded through CacheEntry nodes for O(1) promotion/eviction,
/// backed by an AutoHashMap for O(1) key lookup.
const std = @import("std");
const array_mod = @import("mlx").array;
const Array = array_mod.Array;

/// Default cache budget: 6GB
/// Tuned for 48GB Mac running DeepSeek-V4-Flash-4bit (141GB model).
/// 6GB balances cache hit rate vs OS page cache pressure:
/// - Too large (10GB+): squeezes backbone pages, causes page thrashing
/// - Too small (2GB): low hit rate, more SSD I/O per token
/// - 6GB: steady-state 6.5 tok/s, HTTP converges to ~32s by request #3
pub const DEFAULT_MAX_BYTES: usize = 6 * 1024 * 1024 * 1024;

/// Cache key identifying a specific expert tensor slice.
pub const CacheKey = struct {
    layer_idx: u32,
    tensor_name_hash: u64,
    expert_id: u32,
};

/// Hash context for CacheKey in AutoHashMap.
pub const CacheKeyContext = struct {
    pub fn hash(_: CacheKeyContext, key: CacheKey) u64 {
        var h = std.hash.Wyhash.init(0);
        h.update(std.mem.asBytes(&key.layer_idx));
        h.update(std.mem.asBytes(&key.tensor_name_hash));
        h.update(std.mem.asBytes(&key.expert_id));
        return h.final();
    }

    pub fn eql(_: CacheKeyContext, a: CacheKey, b: CacheKey) bool {
        return a.layer_idx == b.layer_idx and
            a.tensor_name_hash == b.tensor_name_hash and
            a.expert_id == b.expert_id;
    }
};

/// A single cache entry, threaded into the LFU doubly-linked list.
pub const CacheEntry = struct {
    key: CacheKey,
    tensor: Array,
    byte_size: usize,
    /// Access frequency counter for LFU eviction policy
    frequency: u64,
    /// Previous entry in LFU list (toward head / most frequent)
    lru_prev: ?*CacheEntry,
    /// Next entry in LFU list (toward tail / least frequent)
    lru_next: ?*CacheEntry,
};

/// Snapshot of cache statistics for diagnostics.
pub const CacheStats = struct {
    hits: u64,
    misses: u64,
    current_bytes: usize,
    max_bytes: usize,
    entry_count: usize,
};

/// LFU cache for expert weight tensors.
///
/// Uses Least Frequently Used (LFU) eviction policy instead of LRU.
/// Head = most frequently used, tail = least frequently used.
/// On `get()` hit, the entry's frequency is incremented and it may be promoted.
/// On `put()`, least frequent entries are evicted until the budget allows the new entry.
pub const ExpertCache = struct {
    allocator: std.mem.Allocator,
    map: std.HashMap(CacheKey, *CacheEntry, CacheKeyContext, std.hash_map.default_max_load_percentage),
    max_bytes: usize,
    current_bytes: usize,
    hits: u64,
    misses: u64,

    /// LFU list head (most frequently used)
    lru_head: ?*CacheEntry,
    /// LFU list tail (least frequently used)
    lru_tail: ?*CacheEntry,

    /// Initialize an empty cache with the given memory budget.
    pub fn init(allocator: std.mem.Allocator, max_bytes: usize) ExpertCache {
        return ExpertCache{
            .allocator = allocator,
            .map = std.HashMap(CacheKey, *CacheEntry, CacheKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
            .max_bytes = max_bytes,
            .current_bytes = 0,
            .hits = 0,
            .misses = 0,
            .lru_head = null,
            .lru_tail = null,
        };
    }

    /// Free all entries, their tensors, and the map.
    pub fn deinit(self: *ExpertCache) void {
        // Walk the LRU list and free every entry + its tensor
        var current = self.lru_head;
        while (current) |entry| {
            const next = entry.lru_next;
            entry.tensor.deinit();
            self.allocator.destroy(entry);
            current = next;
        }
        self.map.deinit();
        self.lru_head = null;
        self.lru_tail = null;
        self.current_bytes = 0;
    }

    /// Look up a cached expert tensor by key.
    ///
    /// Returns the tensor on hit (increments frequency and may reorder in LFU list),
    /// or null on miss. Updates hit/miss counters.
    pub fn get(self: *ExpertCache, key: CacheKey) ?Array {
        if (self.map.get(key)) |entry| {
            self.hits += 1;
            entry.frequency += 1;
            // Reorder in LFU list based on new frequency
            self.reorderByFrequency(entry);
            return entry.tensor;
        }
        self.misses += 1;
        return null;
    }

    /// Insert a tensor into the cache.
    ///
    /// If `byte_size` exceeds `max_bytes`, the tensor is not cached (Req 1.6).
    /// Otherwise, least frequent entries are evicted until there is room, then the new
    /// entry is inserted with frequency=1 at the appropriate position in the LFU list.
    pub fn put(self: *ExpertCache, key: CacheKey, tensor: Array, byte_size: usize) void {
        // Skip caching if the single tensor exceeds the entire budget
        if (byte_size > self.max_bytes) {
            std.log.debug("ExpertCache: tensor too large ({d} bytes > {d} max), skipping cache", .{ byte_size, self.max_bytes });
            return;
        }

        // If the key already exists, update it
        if (self.map.get(key)) |existing| {
            // Remove old entry's bytes, deinit old tensor, update with new
            self.current_bytes -= existing.byte_size;
            existing.tensor.deinit();

            // Evict until we have room for the new size
            self.evictUntil(byte_size);

            existing.tensor = tensor;
            existing.byte_size = byte_size;
            existing.frequency += 1; // Increment frequency on update
            self.current_bytes += byte_size;
            self.reorderByFrequency(existing);
            return;
        }

        // Evict until we have room
        self.evictUntil(byte_size);

        // Allocate and insert new entry with frequency=1
        const entry = self.allocator.create(CacheEntry) catch {
            std.log.debug("ExpertCache: allocation failed, skipping cache insert", .{});
            return;
        };
        entry.* = CacheEntry{
            .key = key,
            .tensor = tensor,
            .byte_size = byte_size,
            .frequency = 1,
            .lru_prev = null,
            .lru_next = null,
        };

        self.map.put(key, entry) catch {
            self.allocator.destroy(entry);
            std.log.debug("ExpertCache: map insert failed, skipping cache insert", .{});
            return;
        };

        self.current_bytes += byte_size;
        self.insertByFrequency(entry);
    }

    /// Evict least-frequently-used entries until `current_bytes + needed_bytes <= max_bytes`.
    fn evictUntil(self: *ExpertCache, needed_bytes: usize) void {
        while (self.current_bytes + needed_bytes > self.max_bytes) {
            const tail = self.lru_tail orelse break;

            std.log.debug("ExpertCache: evicting layer={d} expert={d} (freq={d})", .{ tail.key.layer_idx, tail.key.expert_id, tail.frequency });

            // Remove from LRU list
            self.removeFromList(tail);

            // Remove from map
            _ = self.map.remove(tail.key);

            // Update accounting and free resources
            self.current_bytes -= tail.byte_size;
            tail.tensor.deinit();
            self.allocator.destroy(tail);
        }
    }

    /// Return a snapshot of current cache statistics.
    pub fn stats(self: *const ExpertCache) CacheStats {
        return CacheStats{
            .hits = self.hits,
            .misses = self.misses,
            .current_bytes = self.current_bytes,
            .max_bytes = self.max_bytes,
            .entry_count = self.map.count(),
        };
    }

    // ── LFU list helpers ──

    /// Reorder an entry in the LFU list based on its current frequency.
    /// Moves the entry toward the head if its frequency is higher than its neighbors.
    fn reorderByFrequency(self: *ExpertCache, entry: *CacheEntry) void {
        // Remove from current position
        self.removeFromList(entry);
        // Re-insert at the correct position based on frequency
        self.insertByFrequency(entry);
    }

    /// Insert an entry into the LFU list at the correct position based on frequency.
    /// List is ordered from head (highest frequency) to tail (lowest frequency).
    fn insertByFrequency(self: *ExpertCache, entry: *CacheEntry) void {
        // Empty list case
        if (self.lru_head == null) {
            entry.lru_prev = null;
            entry.lru_next = null;
            self.lru_head = entry;
            self.lru_tail = entry;
            return;
        }

        // Find insertion point: first entry with frequency <= new entry's frequency
        var current = self.lru_head;
        while (current) |node| {
            if (node.frequency <= entry.frequency) {
                // Insert before this node
                entry.lru_next = node;
                entry.lru_prev = node.lru_prev;
                if (node.lru_prev) |prev| {
                    prev.lru_next = entry;
                } else {
                    self.lru_head = entry;
                }
                node.lru_prev = entry;
                return;
            }
            current = node.lru_next;
        }

        // If we get here, entry has lowest frequency - insert at tail
        entry.lru_prev = self.lru_tail;
        entry.lru_next = null;
        if (self.lru_tail) |tail| {
            tail.lru_next = entry;
        }
        self.lru_tail = entry;
    }

    /// Move an existing entry to the head of the LRU list (most recently used).
    /// NOTE: This is kept for backward compatibility but not used in LFU mode.
    fn moveToHead(self: *ExpertCache, entry: *CacheEntry) void {
        if (self.lru_head == entry) return; // already at head
        self.removeFromList(entry);
        self.pushHead(entry);
    }

    /// Insert an entry at the head of the LRU list.
    /// NOTE: This is kept for backward compatibility but not used in LFU mode.
    fn pushHead(self: *ExpertCache, entry: *CacheEntry) void {
        entry.lru_prev = null;
        entry.lru_next = self.lru_head;
        if (self.lru_head) |old_head| {
            old_head.lru_prev = entry;
        }
        self.lru_head = entry;
        if (self.lru_tail == null) {
            self.lru_tail = entry;
        }
    }

    /// Remove an entry from the LRU list (does not free it).
    fn removeFromList(self: *ExpertCache, entry: *CacheEntry) void {
        if (entry.lru_prev) |prev| {
            prev.lru_next = entry.lru_next;
        } else {
            // entry was head
            self.lru_head = entry.lru_next;
        }
        if (entry.lru_next) |next| {
            next.lru_prev = entry.lru_prev;
        } else {
            // entry was tail
            self.lru_tail = entry.lru_prev;
        }
        entry.lru_prev = null;
        entry.lru_next = null;
    }
};

/// Hash a tensor name string to a u64 for use in CacheKey.
pub fn hashTensorName(name: []const u8) u64 {
    return std.hash.Wyhash.hash(0, name);
}

// ── Tests ──

test "ExpertCache: init and deinit empty cache" {
    const allocator = std.testing.allocator;
    var cache = ExpertCache.init(allocator, 1024);
    defer cache.deinit();

    const s = cache.stats();
    try std.testing.expectEqual(@as(u64, 0), s.hits);
    try std.testing.expectEqual(@as(u64, 0), s.misses);
    try std.testing.expectEqual(@as(usize, 0), s.current_bytes);
    try std.testing.expectEqual(@as(usize, 1024), s.max_bytes);
    try std.testing.expectEqual(@as(usize, 0), s.entry_count);
}

test "ExpertCache: put and get round-trip" {
    const allocator = std.testing.allocator;
    var cache = ExpertCache.init(allocator, 4096);
    defer cache.deinit();

    const key = CacheKey{ .layer_idx = 1, .tensor_name_hash = hashTensorName("gate_proj"), .expert_id = 42 };

    // Create a small test tensor
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const tensor = try Array.fromData(allocator, f32, &data, &[_]i32{4});

    cache.put(key, tensor, 16);

    // get should return the tensor
    const result = cache.get(key);
    try std.testing.expect(result != null);

    const s = cache.stats();
    try std.testing.expectEqual(@as(u64, 1), s.hits);
    try std.testing.expectEqual(@as(u64, 0), s.misses);
    try std.testing.expectEqual(@as(usize, 16), s.current_bytes);
    try std.testing.expectEqual(@as(usize, 1), s.entry_count);
}

test "ExpertCache: get miss increments misses" {
    const allocator = std.testing.allocator;
    var cache = ExpertCache.init(allocator, 4096);
    defer cache.deinit();

    const key = CacheKey{ .layer_idx = 0, .tensor_name_hash = 123, .expert_id = 0 };
    const result = cache.get(key);
    try std.testing.expect(result == null);

    const s = cache.stats();
    try std.testing.expectEqual(@as(u64, 0), s.hits);
    try std.testing.expectEqual(@as(u64, 1), s.misses);
}

test "ExpertCache: LFU eviction order" {
    const allocator = std.testing.allocator;
    // Budget for exactly 2 entries of 100 bytes each
    var cache = ExpertCache.init(allocator, 200);
    defer cache.deinit();

    const key_a = CacheKey{ .layer_idx = 0, .tensor_name_hash = 1, .expert_id = 0 };
    const key_b = CacheKey{ .layer_idx = 0, .tensor_name_hash = 2, .expert_id = 0 };
    const key_c = CacheKey{ .layer_idx = 0, .tensor_name_hash = 3, .expert_id = 0 };

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const t_a = try Array.fromData(allocator, f32, &data, &[_]i32{4});
    const t_b = try Array.fromData(allocator, f32, &data, &[_]i32{4});
    const t_c = try Array.fromData(allocator, f32, &data, &[_]i32{4});

    cache.put(key_a, t_a, 100); // [A freq=1]
    cache.put(key_b, t_b, 100); // [A freq=1, B freq=1] or [B freq=1, A freq=1]

    // Cache is full (200/200). Inserting C should evict the least frequent entry.
    // Since A and B both have freq=1, either could be evicted (implementation-dependent).
    // For LFU with same frequency, we evict the tail (last inserted with that frequency).
    cache.put(key_c, t_c, 100);

    // At least one of A or B should be evicted, C should be present
    const a_present = cache.get(key_a) != null;
    const b_present = cache.get(key_b) != null;
    const c_present = cache.get(key_c) != null;

    try std.testing.expect(c_present); // C should always be present
    try std.testing.expect(!(a_present and b_present)); // At least one of A or B was evicted
}

test "ExpertCache: get promotes entry by frequency, changes eviction order" {
    const allocator = std.testing.allocator;
    var cache = ExpertCache.init(allocator, 200);
    defer cache.deinit();

    const key_a = CacheKey{ .layer_idx = 0, .tensor_name_hash = 1, .expert_id = 0 };
    const key_b = CacheKey{ .layer_idx = 0, .tensor_name_hash = 2, .expert_id = 0 };
    const key_c = CacheKey{ .layer_idx = 0, .tensor_name_hash = 3, .expert_id = 0 };

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const t_a = try Array.fromData(allocator, f32, &data, &[_]i32{4});
    const t_b = try Array.fromData(allocator, f32, &data, &[_]i32{4});
    const t_c = try Array.fromData(allocator, f32, &data, &[_]i32{4});

    cache.put(key_a, t_a, 100); // [A freq=1]
    cache.put(key_b, t_b, 100); // [A freq=1, B freq=1]

    // Access A multiple times to increase its frequency: freq=4
    _ = cache.get(key_a);
    _ = cache.get(key_a);
    _ = cache.get(key_a);

    // Now A has freq=4, B has freq=1
    // Inserting C should evict B (lowest frequency), not A
    cache.put(key_c, t_c, 100);

    try std.testing.expect(cache.get(key_b) == null); // B evicted (freq=1)
    try std.testing.expect(cache.get(key_a) != null); // A still present (freq=4)
    try std.testing.expect(cache.get(key_c) != null); // C present (freq=1)
}

test "ExpertCache: skip caching when tensor exceeds max_bytes" {
    const allocator = std.testing.allocator;
    var cache = ExpertCache.init(allocator, 100);
    defer cache.deinit();

    const key = CacheKey{ .layer_idx = 0, .tensor_name_hash = 1, .expert_id = 0 };
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const tensor = try Array.fromData(allocator, f32, &data, &[_]i32{4});
    defer tensor.deinit(); // We must deinit since cache won't own it

    // byte_size > max_bytes → should not be cached
    cache.put(key, tensor, 200);

    try std.testing.expect(cache.get(key) == null);
    try std.testing.expectEqual(@as(usize, 0), cache.stats().current_bytes);
    try std.testing.expectEqual(@as(usize, 0), cache.stats().entry_count);
}

test "ExpertCache: memory tracking invariant" {
    const allocator = std.testing.allocator;
    var cache = ExpertCache.init(allocator, 500);
    defer cache.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    // Insert 5 entries of 100 bytes each (fills the cache)
    var keys: [5]CacheKey = undefined;
    for (0..5) |i| {
        keys[i] = CacheKey{ .layer_idx = 0, .tensor_name_hash = @intCast(i), .expert_id = 0 };
        const t = try Array.fromData(allocator, f32, &data, &[_]i32{4});
        cache.put(keys[i], t, 100);
    }

    try std.testing.expectEqual(@as(usize, 500), cache.stats().current_bytes);
    try std.testing.expectEqual(@as(usize, 5), cache.stats().entry_count);
    try std.testing.expect(cache.stats().current_bytes <= cache.stats().max_bytes);

    // Insert one more — should evict the oldest and maintain invariant
    const key_new = CacheKey{ .layer_idx = 1, .tensor_name_hash = 0, .expert_id = 0 };
    const t_new = try Array.fromData(allocator, f32, &data, &[_]i32{4});
    cache.put(key_new, t_new, 100);

    try std.testing.expectEqual(@as(usize, 500), cache.stats().current_bytes);
    try std.testing.expectEqual(@as(usize, 5), cache.stats().entry_count);
    try std.testing.expect(cache.stats().current_bytes <= cache.stats().max_bytes);
}

test "ExpertCache: hit/miss counter accuracy" {
    const allocator = std.testing.allocator;
    var cache = ExpertCache.init(allocator, 4096);
    defer cache.deinit();

    const key = CacheKey{ .layer_idx = 0, .tensor_name_hash = 1, .expert_id = 0 };
    const missing_key = CacheKey{ .layer_idx = 99, .tensor_name_hash = 99, .expert_id = 99 };

    const data = [_]f32{1.0};
    const tensor = try Array.fromData(allocator, f32, &data, &[_]i32{1});
    cache.put(key, tensor, 4);

    // 3 hits
    _ = cache.get(key);
    _ = cache.get(key);
    _ = cache.get(key);

    // 2 misses
    _ = cache.get(missing_key);
    _ = cache.get(missing_key);

    const s = cache.stats();
    try std.testing.expectEqual(@as(u64, 3), s.hits);
    try std.testing.expectEqual(@as(u64, 2), s.misses);
    // hits + misses == total get() calls
    try std.testing.expectEqual(@as(u64, 5), s.hits + s.misses);
}

test "ExpertCache: hashTensorName produces consistent hashes" {
    const h1 = hashTensorName("model.layers.5.ffn.switch_mlp.gate_proj.weight");
    const h2 = hashTensorName("model.layers.5.ffn.switch_mlp.gate_proj.weight");
    const h3 = hashTensorName("model.layers.5.ffn.switch_mlp.up_proj.weight");

    try std.testing.expectEqual(h1, h2);
    try std.testing.expect(h1 != h3);
}

test "ExpertCache: update existing key" {
    const allocator = std.testing.allocator;
    var cache = ExpertCache.init(allocator, 4096);
    defer cache.deinit();

    const key = CacheKey{ .layer_idx = 0, .tensor_name_hash = 1, .expert_id = 0 };

    const data1 = [_]f32{ 1.0, 2.0 };
    const t1 = try Array.fromData(allocator, f32, &data1, &[_]i32{2});
    cache.put(key, t1, 8);

    try std.testing.expectEqual(@as(usize, 8), cache.stats().current_bytes);

    // Update with a larger tensor
    const data2 = [_]f32{ 3.0, 4.0, 5.0, 6.0 };
    const t2 = try Array.fromData(allocator, f32, &data2, &[_]i32{4});
    cache.put(key, t2, 16);

    // Should have replaced, not duplicated
    try std.testing.expectEqual(@as(usize, 1), cache.stats().entry_count);
    try std.testing.expectEqual(@as(usize, 16), cache.stats().current_bytes);
}

test "ExpertCache: default max bytes constant" {
    try std.testing.expectEqual(@as(usize, 6 * 1024 * 1024 * 1024), DEFAULT_MAX_BYTES);
}

// ── Property-Based Tests ──

test "Feature: stream-mode-performance, Property 1: Cache round-trip" {
    // **Validates: Requirements 1.1, 1.2**
    // For any valid cache key and tensor value, inserting via put() and retrieving
    // via get() with the same key SHALL return the original tensor, provided no
    // eviction has occurred for that key.
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // Random cache budget large enough to hold all entries this iteration
        const n_entries = random.intRangeAtMost(u32, 1, 10);
        const entry_size: usize = 16; // 4 f32 values
        const budget = @as(usize, n_entries) * entry_size + 100; // extra headroom

        var cache = ExpertCache.init(allocator, budget);
        defer cache.deinit();

        // Generate random keys and insert tensors
        var keys: [10]CacheKey = undefined;
        for (0..n_entries) |i| {
            keys[i] = CacheKey{
                .layer_idx = random.intRangeAtMost(u32, 0, 42),
                .tensor_name_hash = random.int(u64),
                .expert_id = random.intRangeAtMost(u32, 0, 255),
            };

            const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
            const tensor = try Array.fromData(allocator, f32, &data, &[_]i32{4});
            cache.put(keys[i], tensor, entry_size);
        }

        // Verify round-trip: every inserted key should be retrievable
        for (0..n_entries) |i| {
            const result = cache.get(keys[i]);
            try std.testing.expect(result != null);
        }
    }
}

test "Feature: stream-mode-performance, Property 2: LFU eviction order" {
    // **Validates: Requirements 1.3**
    // For any sequence of put()/get() operations on a small-budget cache,
    // when a put() triggers eviction, the evicted entry SHALL be the one
    // with the lowest frequency count among all entries.
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(123);
    const random = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        const entry_size: usize = 100;
        // Budget for exactly 3 entries
        var cache = ExpertCache.init(allocator, 300);
        defer cache.deinit();

        // Insert 3 entries: A, B, C
        var keys: [4]CacheKey = undefined;
        for (0..3) |i| {
            keys[i] = CacheKey{
                .layer_idx = @intCast(i),
                .tensor_name_hash = random.int(u64),
                .expert_id = 0,
            };
            const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
            const tensor = try Array.fromData(allocator, f32, &data, &[_]i32{4});
            cache.put(keys[i], tensor, entry_size);
        }

        // Access one entry multiple times to make it high-frequency
        const hot_idx = random.intRangeAtMost(usize, 0, 2);
        const n_accesses = random.intRangeAtMost(usize, 3, 10);
        for (0..n_accesses) |_| {
            _ = cache.get(keys[hot_idx]);
        }

        // The hot entry now has freq >= 4, others have freq = 1
        // Insert a 4th entry — should evict one of the low-frequency entries, not the hot one
        keys[3] = CacheKey{
            .layer_idx = 99,
            .tensor_name_hash = random.int(u64),
            .expert_id = 0,
        };
        const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const tensor = try Array.fromData(allocator, f32, &data, &[_]i32{4});
        cache.put(keys[3], tensor, entry_size);

        // The hot entry should still be present
        try std.testing.expect(cache.get(keys[hot_idx]) != null);
        // keys[3] should be present
        try std.testing.expect(cache.get(keys[3]) != null);

        // At least one of the low-frequency entries should be evicted
        var evicted_count: usize = 0;
        for (0..3) |i| {
            if (i != hot_idx and cache.get(keys[i]) == null) {
                evicted_count += 1;
            }
        }
        try std.testing.expect(evicted_count >= 1);
    }
}

test "Feature: stream-mode-performance, Property 3: Cache memory tracking invariant" {
    // **Validates: Requirements 1.4**
    // For any sequence of put(), get(), and eviction operations,
    // current_bytes SHALL equal the sum of byte_size of all entries,
    // and current_bytes SHALL never exceed max_bytes.
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(456);
    const random = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        const max_bytes: usize = 500;
        var cache = ExpertCache.init(allocator, max_bytes);
        defer cache.deinit();

        // Perform a random sequence of put/get operations
        const n_ops = random.intRangeAtMost(usize, 5, 20);
        for (0..n_ops) |_| {
            const op = random.intRangeAtMost(u8, 0, 1);
            const key = CacheKey{
                .layer_idx = random.intRangeAtMost(u32, 0, 5),
                .tensor_name_hash = random.intRangeAtMost(u64, 0, 5),
                .expert_id = random.intRangeAtMost(u32, 0, 5),
            };

            if (op == 0) {
                // put
                const byte_size = random.intRangeAtMost(usize, 10, 150);
                const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
                const tensor = try Array.fromData(allocator, f32, &data, &[_]i32{4});
                cache.put(key, tensor, byte_size);
            } else {
                // get
                _ = cache.get(key);
            }

            // Invariant: current_bytes <= max_bytes
            try std.testing.expect(cache.current_bytes <= cache.max_bytes);

            // Invariant: current_bytes == sum of all entry byte_sizes
            var sum: usize = 0;
            var node = cache.lru_head;
            while (node) |entry| {
                sum += entry.byte_size;
                node = entry.lru_next;
            }
            try std.testing.expectEqual(sum, cache.current_bytes);
        }
    }
}

test "Feature: stream-mode-performance, Property 4: Cache hit/miss counter accuracy" {
    // **Validates: Requirements 1.7**
    // For any sequence of get() operations, hits + misses SHALL equal
    // the total number of get() calls.
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(789);
    const random = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        var cache = ExpertCache.init(allocator, 4096);
        defer cache.deinit();

        // Insert some entries
        const n_entries = random.intRangeAtMost(u32, 1, 8);
        for (0..n_entries) |i| {
            const key = CacheKey{
                .layer_idx = @intCast(i),
                .tensor_name_hash = 0,
                .expert_id = 0,
            };
            const data = [_]f32{1.0};
            const tensor = try Array.fromData(allocator, f32, &data, &[_]i32{1});
            cache.put(key, tensor, 4);
        }

        // Perform random get() calls and track total count
        var total_gets: u64 = 0;
        const n_gets = random.intRangeAtMost(usize, 5, 30);
        for (0..n_gets) |_| {
            const key = CacheKey{
                .layer_idx = random.intRangeAtMost(u32, 0, 15), // some will miss
                .tensor_name_hash = 0,
                .expert_id = 0,
            };
            _ = cache.get(key);
            total_gets += 1;

            // Invariant: hits + misses == total get() calls
            try std.testing.expectEqual(total_gets, cache.hits + cache.misses);
        }
    }
}
