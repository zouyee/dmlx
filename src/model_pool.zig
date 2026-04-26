/// Model Pool — multi-model management with LRU eviction and pinning.
///
/// Maintains a pool of loaded models keyed by name. When the pool's memory
/// budget is exceeded, the least recently used non-pinned model is evicted.
/// Models can be pinned to prevent eviction regardless of access recency.
///
/// Design reference: Section 6.1 of the production-deployment design doc.
const std = @import("std");
const generation = @import("generation.zig");

pub const ModelVTable = generation.ModelVTable;
pub const ModelConfig = generation.ModelConfig;

// ============================================================
// LoadedModel — metadata wrapper around a loaded model instance
// ============================================================

pub const LoadedModel = struct {
    vtable: ?ModelVTable,
    name: []const u8,
    path: []const u8,
    memory_bytes: usize,
    last_used: u64,
};

// ============================================================
// ModelPool
// ============================================================

pub const ModelPoolError = error{
    ModelNotFound,
    AllModelsPinned,
    MemoryLimitExceeded,
};

pub const ModelPool = struct {
    allocator: std.mem.Allocator,
    models: std.StringHashMap(LoadedModel),
    lru_order: std.ArrayList([]const u8),
    max_memory: usize,
    current_memory: usize,
    pinned: std.StringHashMap(void),
    access_counter: u64,

    /// Stub loader function type. In production this would actually load
    /// model weights; for now it returns an estimated memory size.
    pub const LoaderFn = *const fn (name: []const u8, path: []const u8) anyerror!LoadedModel;

    pub fn init(allocator: std.mem.Allocator, max_memory: usize) ModelPool {
        return .{
            .allocator = allocator,
            .models = std.StringHashMap(LoadedModel).init(allocator),
            .lru_order = std.ArrayList([]const u8).empty,
            .max_memory = max_memory,
            .current_memory = 0,
            .pinned = std.StringHashMap(void).init(allocator),
            .access_counter = 0,
        };
    }

    pub fn deinit(self: *ModelPool) void {
        // Free all owned name/path strings and deinit vtables.
        var it = self.models.iterator();
        while (it.next()) |entry| {
            const model = entry.value_ptr;
            if (model.vtable) |vtable| {
                vtable.deinit(vtable.ptr, self.allocator);
            }
            self.allocator.free(model.path);
            self.allocator.free(model.name);
        }
        self.models.deinit();
        // Free duped name strings in lru_order.
        for (self.lru_order.items) |name| {
            self.allocator.free(name);
        }
        self.lru_order.deinit(self.allocator);
        self.pinned.deinit();
    }

    /// Return a cached model or load it using the provided loader function.
    /// If loading would exceed max_memory, evicts LRU non-pinned models first.
    pub fn getOrLoad(self: *ModelPool, name: []const u8, path: []const u8, loader: LoaderFn) !*LoadedModel {
        // If already loaded, update access time and return.
        if (self.models.getPtr(name)) |model| {
            self.access_counter += 1;
            model.last_used = self.access_counter;
            self.touchLru(name);
            return model;
        }

        // Load the model via the provided loader.
        var loaded = try loader(name, path);

        // Evict until we have room (or fail if all are pinned).
        while (self.current_memory + loaded.memory_bytes > self.max_memory) {
            self.evictLRU() catch |err| switch (err) {
                ModelPoolError.AllModelsPinned => return ModelPoolError.MemoryLimitExceeded,
                else => return err,
            };
        }

        // Dupe name and path so the pool owns the memory.
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_path = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(owned_path);

        self.access_counter += 1;
        loaded.name = owned_name;
        loaded.path = owned_path;
        loaded.last_used = self.access_counter;

        try self.models.put(owned_name, loaded);
        self.current_memory += loaded.memory_bytes;

        // Append to LRU order (most recent at end).
        const lru_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(lru_name);
        try self.lru_order.append(self.allocator, lru_name);

        return self.models.getPtr(owned_name).?;
    }

    /// Evict the least recently used non-pinned model.
    /// Returns `AllModelsPinned` if every model in the pool is pinned.
    pub fn evictLRU(self: *ModelPool) !void {
        // Walk lru_order from front (oldest) to find first non-pinned model.
        var idx: usize = 0;
        while (idx < self.lru_order.items.len) {
            const candidate = self.lru_order.items[idx];
            if (!self.pinned.contains(candidate)) {
                // Found a non-pinned model — evict it.
                self.evictByName(candidate);
                // Free the LRU entry string and remove from list.
                self.allocator.free(self.lru_order.orderedRemove(idx));
                return;
            }
            idx += 1;
        }
        return ModelPoolError.AllModelsPinned;
    }

    /// Pin a model so it cannot be evicted.
    pub fn pin(self: *ModelPool, name: []const u8) void {
        self.pinned.put(name, {}) catch {};
    }

    /// Unpin a model, allowing it to be evicted again.
    pub fn unpin(self: *ModelPool, name: []const u8) void {
        _ = self.pinned.remove(name);
    }

    /// Check if a model is currently pinned.
    pub fn isPinned(self: *const ModelPool, name: []const u8) bool {
        return self.pinned.contains(name);
    }

    /// Number of models currently in the pool.
    pub fn count(self: *const ModelPool) usize {
        return self.models.count();
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Remove a model from the models map and free its resources.
    fn evictByName(self: *ModelPool, name: []const u8) void {
        if (self.models.fetchRemove(name)) |kv| {
            const model = kv.value;
            if (model.memory_bytes <= self.current_memory) {
                self.current_memory -= model.memory_bytes;
            } else {
                self.current_memory = 0;
            }
            if (model.vtable) |vtable| {
                vtable.deinit(vtable.ptr, self.allocator);
            }
            self.allocator.free(model.path);
            self.allocator.free(model.name);
        }
    }

    /// Move `name` to the end of lru_order (most recently used).
    fn touchLru(self: *ModelPool, name: []const u8) void {
        var idx: usize = 0;
        while (idx < self.lru_order.items.len) {
            if (std.mem.eql(u8, self.lru_order.items[idx], name)) {
                const entry = self.lru_order.orderedRemove(idx);
                self.lru_order.append(self.allocator, entry) catch {};
                return;
            }
            idx += 1;
        }
    }
};

// ============================================================
// Tests
// ============================================================

/// Stub loader for testing — creates a LoadedModel with a given memory size.
/// Uses a simple hash of the name to produce a deterministic memory_bytes value
/// unless overridden by the test.
fn stubLoader(name: []const u8, path: []const u8) anyerror!LoadedModel {
    _ = path;
    // Use name length * 100 as a simple deterministic memory estimate.
    const mem = name.len * 100;
    return LoadedModel{
        .vtable = null,
        .name = name,
        .path = "",
        .memory_bytes = if (mem > 0) mem else 100,
        .last_used = 0,
    };
}

test "ModelPool: init and deinit empty pool" {
    const allocator = std.testing.allocator;
    var pool = ModelPool.init(allocator, 1024);
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 0), pool.count());
    try std.testing.expectEqual(@as(usize, 0), pool.current_memory);
}

test "ModelPool: getOrLoad caches model" {
    const allocator = std.testing.allocator;
    var pool = ModelPool.init(allocator, 10000);
    defer pool.deinit();

    const model = try pool.getOrLoad("alpha", "/models/alpha", stubLoader);
    try std.testing.expectEqual(@as(usize, 1), pool.count());
    try std.testing.expect(model.memory_bytes > 0);
    const first_access = model.last_used;

    // Second call returns the same cached model (no new entry).
    const model2 = try pool.getOrLoad("alpha", "/models/alpha", stubLoader);
    try std.testing.expectEqual(@as(usize, 1), pool.count());
    // Same pointer — it's the cached entry.
    try std.testing.expectEqual(model, model2);
    // Access counter should have advanced.
    try std.testing.expect(model2.last_used > first_access);
}

test "ModelPool: evictLRU removes oldest non-pinned model" {
    const allocator = std.testing.allocator;
    // Small budget to force eviction.
    var pool = ModelPool.init(allocator, 2000);
    defer pool.deinit();

    _ = try pool.getOrLoad("aaa", "/m/aaa", stubLoader); // 300 bytes
    _ = try pool.getOrLoad("bbb", "/m/bbb", stubLoader); // 300 bytes
    _ = try pool.getOrLoad("ccc", "/m/ccc", stubLoader); // 300 bytes

    try std.testing.expectEqual(@as(usize, 3), pool.count());

    // Evict LRU — should remove "aaa" (loaded first, not accessed since).
    try pool.evictLRU();
    try std.testing.expectEqual(@as(usize, 2), pool.count());
    try std.testing.expect(pool.models.get("aaa") == null);
    try std.testing.expect(pool.models.get("bbb") != null);
    try std.testing.expect(pool.models.get("ccc") != null);
}

test "ModelPool: pinned models are not evicted" {
    const allocator = std.testing.allocator;
    var pool = ModelPool.init(allocator, 2000);
    defer pool.deinit();

    _ = try pool.getOrLoad("aaa", "/m/aaa", stubLoader);
    _ = try pool.getOrLoad("bbb", "/m/bbb", stubLoader);

    // Pin the oldest model.
    pool.pin("aaa");
    try std.testing.expect(pool.isPinned("aaa"));

    // Evict should skip "aaa" and evict "bbb".
    try pool.evictLRU();
    try std.testing.expectEqual(@as(usize, 1), pool.count());
    try std.testing.expect(pool.models.get("aaa") != null);
    try std.testing.expect(pool.models.get("bbb") == null);
}

test "ModelPool: evictLRU returns error when all pinned" {
    const allocator = std.testing.allocator;
    var pool = ModelPool.init(allocator, 2000);
    defer pool.deinit();

    _ = try pool.getOrLoad("aaa", "/m/aaa", stubLoader);
    pool.pin("aaa");

    const result = pool.evictLRU();
    try std.testing.expectError(ModelPoolError.AllModelsPinned, result);
}

test "ModelPool: unpin allows eviction" {
    const allocator = std.testing.allocator;
    var pool = ModelPool.init(allocator, 2000);
    defer pool.deinit();

    _ = try pool.getOrLoad("aaa", "/m/aaa", stubLoader);
    _ = try pool.getOrLoad("bbb", "/m/bbb", stubLoader);

    pool.pin("aaa");
    pool.pin("bbb");

    // All pinned — eviction fails.
    try std.testing.expectError(ModelPoolError.AllModelsPinned, pool.evictLRU());

    // Unpin "aaa" — now it can be evicted.
    pool.unpin("aaa");
    try std.testing.expect(!pool.isPinned("aaa"));
    try pool.evictLRU();
    try std.testing.expectEqual(@as(usize, 1), pool.count());
    try std.testing.expect(pool.models.get("aaa") == null);
}

test "ModelPool: getOrLoad auto-evicts when memory exceeded" {
    const allocator = std.testing.allocator;
    // Budget for ~2 models (each ~300 bytes with 3-char names).
    var pool = ModelPool.init(allocator, 650);
    defer pool.deinit();

    _ = try pool.getOrLoad("aaa", "/m/aaa", stubLoader); // 300 bytes
    _ = try pool.getOrLoad("bbb", "/m/bbb", stubLoader); // 300 bytes
    try std.testing.expectEqual(@as(usize, 2), pool.count());

    // Loading a third model should auto-evict the LRU ("aaa").
    _ = try pool.getOrLoad("ccc", "/m/ccc", stubLoader); // 300 bytes
    try std.testing.expect(pool.models.get("aaa") == null);
    try std.testing.expect(pool.models.get("ccc") != null);
}

test "ModelPool: LRU order updates on access" {
    const allocator = std.testing.allocator;
    var pool = ModelPool.init(allocator, 650);
    defer pool.deinit();

    _ = try pool.getOrLoad("aaa", "/m/aaa", stubLoader);
    _ = try pool.getOrLoad("bbb", "/m/bbb", stubLoader);

    // Access "aaa" again — it becomes most recently used.
    _ = try pool.getOrLoad("aaa", "/m/aaa", stubLoader);

    // Now "bbb" is LRU. Loading "ccc" should evict "bbb", not "aaa".
    _ = try pool.getOrLoad("ccc", "/m/ccc", stubLoader);
    try std.testing.expect(pool.models.get("aaa") != null);
    try std.testing.expect(pool.models.get("bbb") == null);
    try std.testing.expect(pool.models.get("ccc") != null);
}

test "ModelPool: getOrLoad fails when pinned models block eviction" {
    const allocator = std.testing.allocator;
    var pool = ModelPool.init(allocator, 350);
    defer pool.deinit();

    _ = try pool.getOrLoad("aaa", "/m/aaa", stubLoader); // 300 bytes
    pool.pin("aaa");

    // Trying to load another model that exceeds budget should fail.
    const result = pool.getOrLoad("bbb", "/m/bbb", stubLoader);
    try std.testing.expectError(ModelPoolError.MemoryLimitExceeded, result);
}

// ============================================================
// Property-Based Test
// (Property 19)
//
// Feature: production-deployment, Property 19: Model Pool LRU Eviction with Pinning
//
// For any set of loaded models in the Model_Pool, when eviction is
// triggered, the least recently used non-pinned model SHALL be evicted
// first. Pinned models SHALL never be evicted regardless of their
// access recency.
//
// **Validates: Requirements R21.3, R21.4**
// ============================================================

test "Property 19: Model Pool LRU Eviction with Pinning — LRU non-pinned evicted first, pinned never evicted (100 iterations)" {
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(19);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // --- Random configuration ---
        // Generate between 3 and 8 models (need at least 3 for meaningful LRU testing).
        const num_models = rand.intRangeAtMost(usize, 3, 8);

        // Each model uses a 4-char name → 400 bytes via stubLoader.
        // Set max_memory large enough to hold all models so we control eviction manually.
        const mem_per_model: usize = 400; // 4-char name * 100
        var pool = ModelPool.init(allocator, num_models * mem_per_model + 1);
        defer pool.deinit();

        // Build unique 4-char model names: "m_00", "m_01", ..., "m_07"
        var names: [8][4]u8 = undefined;
        var name_slices: [8][]const u8 = undefined;
        var path_bufs: [8][7]u8 = undefined;
        var path_slices: [8][]const u8 = undefined;
        for (0..num_models) |i| {
            names[i] = .{ 'm', '_', '0' + @as(u8, @intCast(i / 10)), '0' + @as(u8, @intCast(i % 10)) };
            name_slices[i] = &names[i];
            path_bufs[i] = .{ '/', 'm', '/', 'm', '_', '0' + @as(u8, @intCast(i / 10)), '0' + @as(u8, @intCast(i % 10)) };
            path_slices[i] = &path_bufs[i];
        }

        // Load all models in order (first loaded = oldest LRU).
        for (0..num_models) |i| {
            _ = try pool.getOrLoad(name_slices[i], path_slices[i], stubLoader);
        }
        try std.testing.expectEqual(num_models, pool.count());

        // --- Random access pattern to shuffle LRU order ---
        // Perform some random accesses to change the LRU ordering.
        const num_accesses = rand.intRangeAtMost(usize, 0, num_models * 2);
        // Track access order: last access time per model index.
        var last_access = [_]usize{0} ** 8;
        // Initial LRU order: model 0 is oldest (access_time = 1), model N-1 is newest.
        for (0..num_models) |i| {
            last_access[i] = i + 1;
        }
        var access_clock: usize = num_models;
        for (0..num_accesses) |_| {
            const idx = rand.intRangeLessThan(usize, 0, num_models);
            _ = try pool.getOrLoad(name_slices[idx], path_slices[idx], stubLoader);
            access_clock += 1;
            last_access[idx] = access_clock;
        }

        // --- Random pinning pattern ---
        // Pin a random subset, but ensure at least one model is NOT pinned
        // so eviction can succeed.
        var pinned = [_]bool{false} ** 8;
        var num_pinned: usize = 0;
        for (0..num_models) |i| {
            if (rand.boolean() and num_pinned < num_models - 1) {
                pool.pin(name_slices[i]);
                pinned[i] = true;
                num_pinned += 1;
            }
        }

        // --- Property A: LRU non-pinned model is evicted first ---
        // Find the expected eviction target: the non-pinned model with the
        // smallest last_access value (i.e., least recently used).
        var expected_evict_idx: ?usize = null;
        var min_access: usize = std.math.maxInt(usize);
        for (0..num_models) |i| {
            if (!pinned[i] and last_access[i] < min_access) {
                min_access = last_access[i];
                expected_evict_idx = i;
            }
        }

        // There must be at least one non-pinned model.
        try std.testing.expect(expected_evict_idx != null);
        const evict_idx = expected_evict_idx.?;

        // Record which models exist before eviction.
        const count_before = pool.count();

        // Trigger one eviction.
        try pool.evictLRU();

        // Verify the expected model was evicted.
        try std.testing.expectEqual(count_before - 1, pool.count());
        try std.testing.expect(pool.models.get(name_slices[evict_idx]) == null);

        // --- Property B: Pinned models are NEVER evicted ---
        for (0..num_models) |i| {
            if (pinned[i]) {
                try std.testing.expect(pool.models.get(name_slices[i]) != null);
            }
        }

        // --- Additional evictions: keep evicting until only pinned remain ---
        // Verify pinned models survive all evictions.
        var remaining_non_pinned = num_models - 1 - num_pinned; // one already evicted
        while (remaining_non_pinned > 0) {
            try pool.evictLRU();
            remaining_non_pinned -= 1;
        }

        // Now only pinned models should remain.
        try std.testing.expectEqual(num_pinned, pool.count());

        // Verify every pinned model is still present.
        for (0..num_models) |i| {
            if (pinned[i]) {
                try std.testing.expect(pool.models.get(name_slices[i]) != null);
            }
        }

        // If there are pinned models, further eviction should fail.
        if (num_pinned > 0) {
            try std.testing.expectError(ModelPoolError.AllModelsPinned, pool.evictLRU());
        }
    }
}
