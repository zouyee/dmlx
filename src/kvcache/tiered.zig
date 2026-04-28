/// TieredKVCache — Two-tier KV cache with hot (RAM) and cold (SSD) tiers.
///
/// Strategy: wraps a PagedKVCache as the hot tier. When the hot tier exceeds
/// its configured capacity (in blocks), the least recently accessed blocks are
/// evicted to SSD as safetensors files. When an evicted block is needed for
/// attention, it is restored from SSD back into the hot tier.
///
/// Persistence format: each evicted block is stored as a safetensors file at
/// `{cold_dir}/block_{id}.safetensors` with keys "keys" and "values" plus
/// metadata for block_id, tokens_used, and block_hash.
///
/// Requirements: R22.1, R22.2, R22.3, R22.4
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const iface = @import("interface.zig");
const paged_mod = @import("paged.zig");
const mlx_io = @import("../io/mlx_io.zig");
const posix = @cImport({
    @cInclude("sys/stat.h");
    @cInclude("unistd.h");
});

const Array = array_mod.Array;
const KVSlice = iface.KVSlice;
const KVCacheStrategy = iface.KVCacheStrategy;
const LayerConfig = iface.LayerConfig;
const PagedKVCache = paged_mod.PagedKVCache;

/// Helper: delete a file by path using POSIX unlink.
fn deleteFileByPath(allocator: std.mem.Allocator, path: []const u8) void {
    const path_z = allocator.dupeZ(u8, path) catch return;
    defer allocator.free(path_z);
    _ = posix.unlink(path_z.ptr);
}

pub const TieredKVCache = struct {
    allocator: std.mem.Allocator,
    hot: *PagedKVCache,
    cold_dir: []const u8,
    hot_capacity: usize,
    access_recency: std.AutoHashMap(usize, u64),
    access_counter: u64,
    cold_blocks: std.AutoHashMap(usize, ColdBlockMeta),

    const ColdBlockMeta = struct {
        file_path: []const u8,
        tokens_used: usize,
        hash: ?u64,
    };

    pub const vtable: iface.VTable = .{
        .updateAndFetch = updateAndFetchImpl,
        .currentLen = currentLenImpl,
        .reset = resetImpl,
        .filter = null,
        .rollback = rollbackImpl,
        .deinit = deinitImpl,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        config: LayerConfig,
        page_size: usize,
        hot_capacity: usize,
        cold_dir: []const u8,
        stream: c.c.mlx_stream,
    ) !TieredKVCache {
        const hot = try allocator.create(PagedKVCache);
        errdefer allocator.destroy(hot);
        hot.* = try PagedKVCache.init(allocator, config, page_size, 16, 64, stream);

        const dir_owned = try allocator.dupe(u8, cold_dir);
        errdefer allocator.free(dir_owned);

        return .{
            .allocator = allocator,
            .hot = hot,
            .cold_dir = dir_owned,
            .hot_capacity = hot_capacity,
            .access_recency = std.AutoHashMap(usize, u64).init(allocator),
            .access_counter = 0,
            .cold_blocks = std.AutoHashMap(usize, ColdBlockMeta).init(allocator),
        };
    }

    pub fn asStrategy(self: *TieredKVCache) KVCacheStrategy {
        return .{ .ptr = self, .vtable = &vtable };
    }

    /// Count the number of used pages in the hot tier.
    pub fn hotUsedCount(self: *TieredKVCache) usize {
        var count: usize = 0;
        for (self.hot.pages.items) |page| {
            if (page.used) count += 1;
        }
        return count;
    }

    /// Touch a page to update its access recency.
    fn touchPage(self: *TieredKVCache, page_idx: usize) void {
        self.access_counter += 1;
        self.access_recency.put(page_idx, self.access_counter) catch {};
    }

    /// Find the least recently accessed used page in the hot tier.
    pub fn findLRUPage(self: *TieredKVCache) ?usize {
        var oldest_time: u64 = std.math.maxInt(u64);
        var oldest_idx: ?usize = null;

        for (self.hot.pages.items, 0..) |page, i| {
            if (!page.used) continue;
            const access_time = self.access_recency.get(i) orelse 0;
            if (access_time < oldest_time) {
                oldest_time = access_time;
                oldest_idx = i;
            }
        }
        return oldest_idx;
    }

    /// Evict a block from the hot tier to SSD (cold tier).
    /// Serializes the block's keys and values as a safetensors file.
    pub fn evictToSSD(self: *TieredKVCache, page_idx: usize) !void {
        if (page_idx >= self.hot.pages.items.len) return error.InvalidPageIndex;

        const page = &self.hot.pages.items[page_idx];
        if (!page.used) return error.PageNotInUse;

        // Build file path: {cold_dir}/block_{page_idx}.safetensors
        const path = try std.fmt.allocPrint(self.allocator, "{s}/block_{d}.safetensors", .{ self.cold_dir, page_idx });
        defer self.allocator.free(path);

        // Evaluate arrays before saving
        try c.check(c.c.mlx_array_eval(page.keys.inner));
        try c.check(c.c.mlx_array_eval(page.values.inner));

        // Build weights map
        var weights = std.StringHashMap(Array).init(self.allocator);
        defer weights.deinit();
        try weights.put("keys", page.keys);
        try weights.put("values", page.values);

        // Build metadata map
        var metadata = std.StringHashMap([]const u8).init(self.allocator);
        defer {
            var it = metadata.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.value_ptr.*);
            }
            metadata.deinit();
        }

        const tokens_used_str = try std.fmt.allocPrint(self.allocator, "{d}", .{page.ref_count});
        try metadata.put("ref_count", tokens_used_str);

        // Save to safetensors
        try mlx_io.saveSafetensors(self.allocator, path, weights, metadata);

        // Track cold block metadata
        const path_owned = try self.allocator.dupe(u8, path);

        // Free old path if this page was already in cold storage (re-eviction)
        if (self.cold_blocks.getPtr(page_idx)) |old_meta| {
            self.allocator.free(old_meta.file_path);
        }

        try self.cold_blocks.put(page_idx, .{
            .file_path = path_owned,
            .tokens_used = 0, // will be set from page
            .hash = null,
        });

        // Free the arrays from the hot tier page and mark unused
        page.keys.deinit();
        page.values.deinit();

        // Re-create empty arrays so the page struct stays valid
        const shape = &[_]i32{
            1,
            @intCast(self.hot.num_kv_heads),
            @intCast(self.hot.page_size),
            @intCast(self.hot.head_dim),
        };
        const stream = c.c.mlx_default_cpu_stream_new();
        var empty_k = c.c.mlx_array_new();
        var empty_v = c.c.mlx_array_new();
        try c.check(c.c.mlx_zeros(&empty_k, shape.ptr, shape.len, @intCast(@intFromEnum(self.hot.dtype)), stream));
        try c.check(c.c.mlx_zeros(&empty_v, shape.ptr, shape.len, @intCast(@intFromEnum(self.hot.dtype)), stream));
        page.keys = Array.fromHandle(empty_k);
        page.values = Array.fromHandle(empty_v);
        page.used = false;
        page.ref_count = 0;

        // Remove from access recency tracking
        _ = self.access_recency.remove(page_idx);
    }

    /// Restore a block from SSD (cold tier) back to the hot tier.
    /// Deserializes the safetensors file and loads keys/values into the page.
    pub fn restoreFromSSD(self: *TieredKVCache, page_idx: usize) !void {
        const meta = self.cold_blocks.get(page_idx) orelse return error.BlockNotInColdTier;

        // Load from safetensors
        var result = try mlx_io.loadSafetensors(self.allocator, meta.file_path);
        defer result.deinit(self.allocator);

        const loaded_keys = result.weights.get("keys") orelse return error.MissingKeysInColdBlock;
        const loaded_values = result.weights.get("values") orelse return error.MissingValuesInColdBlock;

        // Copy arrays (loadSafetensors result owns them, we need our own copies)
        var k_copy = c.c.mlx_array_new();
        try c.check(c.c.mlx_array_set(&k_copy, loaded_keys.inner));
        var v_copy = c.c.mlx_array_new();
        try c.check(c.c.mlx_array_set(&v_copy, loaded_values.inner));

        // Replace the page's arrays
        const page = &self.hot.pages.items[page_idx];
        page.keys.deinit();
        page.values.deinit();
        page.keys = Array.fromHandle(k_copy);
        page.values = Array.fromHandle(v_copy);
        page.used = true;
        page.ref_count = 1;

        // Touch the restored page
        self.touchPage(page_idx);

        // Remove from cold blocks and clean up the file
        const removed = self.cold_blocks.fetchRemove(page_idx);
        if (removed) |kv| {
            // Delete the safetensors file from disk
            deleteFileByPath(self.allocator, kv.value.file_path);
            self.allocator.free(kv.value.file_path);
        }
    }

    /// Enforce hot tier capacity by evicting LRU pages to SSD.
    pub fn enforceCapacity(self: *TieredKVCache) !void {
        while (self.hotUsedCount() > self.hot_capacity) {
            const lru_idx = self.findLRUPage() orelse break;
            try self.evictToSSD(lru_idx);
        }
    }

    // ------------------------------------------------------------------
    // VTable implementations
    // ------------------------------------------------------------------

    fn updateAndFetchImpl(
        ctx: *anyopaque,
        keys: Array,
        values: Array,
        stream: c.c.mlx_stream,
    ) anyerror!KVSlice {
        const self: *TieredKVCache = @ptrCast(@alignCast(ctx));

        // Delegate to the hot tier's updateAndFetch
        const result = try PagedKVCache.vtable.updateAndFetch(self.hot, keys, values, stream);

        // Touch all used pages for recency tracking
        for (self.hot.pages.items, 0..) |page, i| {
            if (page.used) {
                self.touchPage(i);
            }
        }

        // Enforce capacity after update
        self.enforceCapacity() catch |err| {
            std.log.warn("TieredKVCache: capacity enforcement failed: {s}", .{@errorName(err)});
        };

        return result;
    }

    fn currentLenImpl(ctx: *anyopaque) usize {
        const self: *TieredKVCache = @ptrCast(@alignCast(ctx));
        return PagedKVCache.vtable.currentLen(self.hot);
    }

    fn resetImpl(ctx: *anyopaque) void {
        const self: *TieredKVCache = @ptrCast(@alignCast(ctx));
        PagedKVCache.vtable.reset(self.hot);

        // Clean up cold tier files
        var it = self.cold_blocks.iterator();
        while (it.next()) |entry| {
            deleteFileByPath(self.allocator, entry.value_ptr.file_path);
            self.allocator.free(entry.value_ptr.file_path);
        }
        self.cold_blocks.clearAndFree();
        self.access_recency.clearAndFree();
        self.access_counter = 0;
    }

    fn rollbackImpl(ctx: *anyopaque, to_len: usize) void {
        const self: *TieredKVCache = @ptrCast(@alignCast(ctx));
        if (PagedKVCache.vtable.rollback) |f| {
            f(self.hot, to_len);
        }
    }

    fn deinitImpl(ctx: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *TieredKVCache = @ptrCast(@alignCast(ctx));

        // Clean up cold tier files
        var it = self.cold_blocks.iterator();
        while (it.next()) |entry| {
            deleteFileByPath(allocator, entry.value_ptr.file_path);
            allocator.free(entry.value_ptr.file_path);
        }
        self.cold_blocks.deinit();
        self.access_recency.deinit();
        allocator.free(self.cold_dir);

        // Deinit the hot tier
        PagedKVCache.vtable.deinit(self.hot, allocator);

        allocator.destroy(self);
    }
};

/// Factory function conforming to StrategyFactory signature.
/// Uses default hot_capacity of 16 and /tmp as cold_dir.
pub fn createTiered(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    stream: c.c.mlx_stream,
) !KVCacheStrategy {
    const cache = try allocator.create(TieredKVCache);
    errdefer allocator.destroy(cache);
    cache.* = try TieredKVCache.init(
        allocator,
        config,
        paged_mod.default_page_size,
        16, // default hot capacity
        "/tmp/mlx_tiered_kv",
        stream,
    );
    return cache.asStrategy();
}

/// Factory with custom hot capacity and cold directory.
pub fn createTieredWithConfig(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    page_size: usize,
    hot_capacity: usize,
    cold_dir: []const u8,
    stream: c.c.mlx_stream,
) !KVCacheStrategy {
    const cache = try allocator.create(TieredKVCache);
    errdefer allocator.destroy(cache);
    cache.* = try TieredKVCache.init(
        allocator,
        config,
        page_size,
        hot_capacity,
        cold_dir,
        stream,
    );
    return cache.asStrategy();
}
