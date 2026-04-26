/// PrefixDiskCache — On-disk shared-prefix reuse for heterogeneous KV caches.
///
/// Bridges `prompt_cache.zig` (safetensors persistence) with `BlockManager.findCachedPrefix`
/// (hash-based reuse). Persists prefix KV cache blocks to disk after first computation
/// and restores them for subsequent requests with matching prefix.
///
/// Handles heterogeneous cache formats where different layers have different KV shapes
/// based on `compress_ratios` in `ModelConfig` (DeepSeek V4).
///
/// Persistence format: each block is stored as a safetensors file at
/// `{cache_dir}/prefix_{hash:016x}.safetensors` with per-layer keys/values
/// stored as `layer_{i}_keys` and `layer_{i}_values`, plus metadata for
/// `layer_count`, `block_size`, and per-layer shape info.
///
/// _Gap: V4 "heterogeneous KV cache with on-disk storage for shared-prefix reuse"_
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const mlx_io = @import("../io/mlx_io.zig");
const paged_mod = @import("paged.zig");
const generation = @import("../generation.zig");

const Array = array_mod.Array;
const BlockManager = paged_mod.BlockManager;
const ModelConfig = generation.ModelConfig;

const posix = @cImport({
    @cInclude("sys/stat.h");
    @cInclude("unistd.h");
});

/// Per-layer keys and values loaded from a prefix block on disk.
pub const PrefixBlockData = struct {
    layer_keys: []Array,
    layer_values: []Array,

    pub fn deinit(self: *PrefixBlockData, allocator: std.mem.Allocator) void {
        for (self.layer_keys) |arr| arr.deinit();
        allocator.free(self.layer_keys);
        for (self.layer_values) |arr| arr.deinit();
        allocator.free(self.layer_values);
    }
};

/// Result of restoring a prefix from disk into the BlockManager.
pub const RestoredPrefix = struct {
    block_ids: []usize,
    num_tokens: usize,

    pub fn deinit(self: *RestoredPrefix, allocator: std.mem.Allocator) void {
        allocator.free(self.block_ids);
    }
};

pub const PrefixDiskCache = struct {
    allocator: std.mem.Allocator,
    cache_dir: []const u8,
    block_size: usize,

    pub fn init(allocator: std.mem.Allocator, cache_dir: []const u8, block_size: usize) !PrefixDiskCache {
        const dir_owned = try allocator.dupe(u8, cache_dir);
        errdefer allocator.free(dir_owned);

        // Ensure cache directory exists via POSIX mkdir.
        const dir_z = try allocator.dupeZ(u8, cache_dir);
        defer allocator.free(dir_z);
        _ = posix.mkdir(dir_z.ptr, 0o755);

        return .{
            .allocator = allocator,
            .cache_dir = dir_owned,
            .block_size = block_size,
        };
    }

    pub fn deinit(self: *PrefixDiskCache) void {
        self.allocator.free(self.cache_dir);
    }

    /// Build the file path for a block hash: {cache_dir}/prefix_{hash:016x}.safetensors
    fn blockPath(self: *const PrefixDiskCache, block_hash: u64) ![]u8 {
        return std.fmt.allocPrint(self.allocator, "{s}/prefix_{x:0>16}.safetensors", .{ self.cache_dir, block_hash });
    }

    /// Check if a block with the given hash exists on disk.
    pub fn hasBlock(self: *const PrefixDiskCache, block_hash: u64) bool {
        const path = self.blockPath(block_hash) catch return false;
        defer self.allocator.free(path);
        const path_z = self.allocator.dupeZ(u8, path) catch return false;
        defer self.allocator.free(path_z);
        return posix.access(path_z.ptr, posix.F_OK) == 0;
    }

    /// Save a computed prefix block to disk. Handles heterogeneous layer shapes.
    /// `layer_keys` and `layer_values` contain per-layer KV data for this block.
    pub fn saveBlock(
        self: *PrefixDiskCache,
        block_hash: u64,
        layer_keys: []const Array,
        layer_values: []const Array,
        model_config: ModelConfig,
    ) !void {
        _ = model_config;
        const num_layers = layer_keys.len;
        if (num_layers != layer_values.len) return error.LayerCountMismatch;

        const path = try self.blockPath(block_hash);
        defer self.allocator.free(path);

        // Build weights map with per-layer keys/values.
        var weights = std.StringHashMap(Array).init(self.allocator);
        defer {
            var it = weights.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
            weights.deinit();
        }

        // Build metadata map.
        var metadata = std.StringHashMap([]const u8).init(self.allocator);
        defer {
            var it = metadata.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                self.allocator.free(entry.value_ptr.*);
            }
            metadata.deinit();
        }

        for (0..num_layers) |i| {
            // Evaluate lazy arrays before saving.
            try c.check(c.c.mlx_array_eval(layer_keys[i].inner));
            try c.check(c.c.mlx_array_eval(layer_values[i].inner));

            const key_name = try std.fmt.allocPrint(self.allocator, "layer_{d}_keys", .{i});
            const val_name = try std.fmt.allocPrint(self.allocator, "layer_{d}_values", .{i});

            try weights.put(key_name, layer_keys[i]);
            try weights.put(val_name, layer_values[i]);

            // Store per-layer shape info in metadata for heterogeneous cache validation.
            const shape = layer_keys[i].shape();
            const shape_str = try std.fmt.allocPrint(self.allocator, "{d}", .{shape.len});
            const shape_key = try std.fmt.allocPrint(self.allocator, "layer_{d}_ndim", .{i});
            try metadata.put(shape_key, shape_str);

            for (shape, 0..) |dim, d| {
                const dim_key = try std.fmt.allocPrint(self.allocator, "layer_{d}_shape_{d}", .{ i, d });
                const dim_val = try std.fmt.allocPrint(self.allocator, "{d}", .{dim});
                try metadata.put(dim_key, dim_val);
            }
        }

        // Store block-level metadata.
        const layer_count_key = try self.allocator.dupe(u8, "layer_count");
        const layer_count_val = try std.fmt.allocPrint(self.allocator, "{d}", .{num_layers});
        try metadata.put(layer_count_key, layer_count_val);

        const block_size_key = try self.allocator.dupe(u8, "block_size");
        const block_size_val = try std.fmt.allocPrint(self.allocator, "{d}", .{self.block_size});
        try metadata.put(block_size_key, block_size_val);

        const hash_key = try self.allocator.dupe(u8, "block_hash");
        const hash_val = try std.fmt.allocPrint(self.allocator, "{d}", .{block_hash});
        try metadata.put(hash_key, hash_val);

        try mlx_io.saveSafetensors(self.allocator, path, weights, metadata);
    }

    /// Load a prefix block from disk. Returns per-layer keys and values.
    /// Caller owns the returned PrefixBlockData.
    pub fn loadBlock(
        self: *PrefixDiskCache,
        block_hash: u64,
        model_config: ModelConfig,
    ) !PrefixBlockData {
        _ = model_config;
        const path = try self.blockPath(block_hash);
        defer self.allocator.free(path);

        var result = try mlx_io.loadSafetensors(self.allocator, path);
        defer result.deinit(self.allocator);

        // Parse layer count from metadata.
        const layer_count_str = result.metadata.get("layer_count") orelse return error.MissingMetadata;
        const num_layers = std.fmt.parseInt(usize, layer_count_str, 10) catch return error.InvalidMetadata;

        var layer_keys = try self.allocator.alloc(Array, num_layers);
        errdefer {
            for (layer_keys) |arr| arr.deinit();
            self.allocator.free(layer_keys);
        }
        var layer_values = try self.allocator.alloc(Array, num_layers);
        errdefer {
            for (layer_values) |arr| arr.deinit();
            self.allocator.free(layer_values);
        }

        // Initialize to avoid undefined values in errdefer.
        for (0..num_layers) |i| {
            layer_keys[i] = Array.fromHandle(c.c.mlx_array_new());
            layer_values[i] = Array.fromHandle(c.c.mlx_array_new());
        }

        for (0..num_layers) |i| {
            const key_name = try std.fmt.allocPrint(self.allocator, "layer_{d}_keys", .{i});
            defer self.allocator.free(key_name);
            const val_name = try std.fmt.allocPrint(self.allocator, "layer_{d}_values", .{i});
            defer self.allocator.free(val_name);

            const loaded_keys = result.weights.get(key_name) orelse return error.MissingLayerData;
            const loaded_values = result.weights.get(val_name) orelse return error.MissingLayerData;

            // Copy arrays (loadSafetensors result owns them).
            var k_copy = c.c.mlx_array_new();
            try c.check(c.c.mlx_array_set(&k_copy, loaded_keys.inner));
            var v_copy = c.c.mlx_array_new();
            try c.check(c.c.mlx_array_set(&v_copy, loaded_values.inner));

            // Free the placeholder before replacing.
            layer_keys[i].deinit();
            layer_values[i].deinit();
            layer_keys[i] = Array.fromHandle(k_copy);
            layer_values[i] = Array.fromHandle(v_copy);
        }

        return .{
            .layer_keys = layer_keys,
            .layer_values = layer_values,
        };
    }

    /// High-level: given token IDs, find which prefix blocks exist on disk,
    /// load them, and register with the BlockManager.
    pub fn restorePrefix(
        self: *PrefixDiskCache,
        block_manager: *BlockManager,
        token_ids: []const u32,
        model_config: ModelConfig,
    ) !RestoredPrefix {
        if (self.block_size == 0) {
            return .{
                .block_ids = try self.allocator.alloc(usize, 0),
                .num_tokens = 0,
            };
        }

        var block_ids = std.ArrayList(usize).empty;
        errdefer block_ids.deinit(self.allocator);

        var prev_hash: u64 = 0;
        var offset: usize = 0;

        while (offset + self.block_size <= token_ids.len) {
            const block_tokens = token_ids[offset .. offset + self.block_size];
            const hash = BlockManager.hashBlock(prev_hash, block_tokens);

            if (!self.hasBlock(hash)) break;

            // Load block data from disk.
            var block_data = self.loadBlock(hash, model_config) catch break;
            defer block_data.deinit(self.allocator);

            // Allocate a block in the BlockManager for this restored data.
            // Use a synthetic request ID based on the hash to track ownership.
            const req_id = hash;
            const allocated = block_manager.allocateBlocks(req_id, 1) catch break;
            defer self.allocator.free(allocated);

            const block_id = allocated[0];

            // Register the block hash so findCachedPrefix can discover it.
            block_manager.registerBlockHash(block_id, hash) catch break;

            // Mark the block as having tokens_used = block_size.
            block_manager.block_pool.items[block_id].tokens_used = self.block_size;

            try block_ids.append(self.allocator, block_id);

            prev_hash = hash;
            offset += self.block_size;
        }

        const num_tokens = offset;
        return .{
            .block_ids = try block_ids.toOwnedSlice(self.allocator),
            .num_tokens = num_tokens,
        };
    }

    /// Remove a cached block file from disk.
    pub fn removeBlock(self: *PrefixDiskCache, block_hash: u64) void {
        const path = self.blockPath(block_hash) catch return;
        defer self.allocator.free(path);
        const path_z = self.allocator.dupeZ(u8, path) catch return;
        defer self.allocator.free(path_z);
        _ = posix.unlink(path_z.ptr);
    }
};

// ============================================================
// Tests
// ============================================================

const posix_dir = @cImport({
    @cInclude("dirent.h");
});

/// Helper: ensure the cache directory exists.
fn ensureCacheDir(allocator: std.mem.Allocator, dir: []const u8) !void {
    const dir_z = try allocator.dupeZ(u8, dir);
    defer allocator.free(dir_z);
    _ = posix.mkdir(dir_z.ptr, 0o755);
}

/// Helper: clean up cache directory.
fn cleanupCacheDir(allocator: std.mem.Allocator, dir: []const u8) void {
    const dir_z = allocator.dupeZ(u8, dir) catch return;
    defer allocator.free(dir_z);
    const dp = posix_dir.opendir(dir_z.ptr);
    if (dp != null) {
        while (true) {
            const entry = posix_dir.readdir(dp);
            if (entry == null) break;
            const name: [*:0]const u8 = @ptrCast(&entry.*.d_name);
            if (name[0] == '.') continue;
            const full_path = std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir, std.mem.span(name) }) catch continue;
            defer allocator.free(full_path);
            const full_path_z = allocator.dupeZ(u8, full_path) catch continue;
            defer allocator.free(full_path_z);
            _ = posix.unlink(full_path_z.ptr);
        }
        _ = posix_dir.closedir(dp);
    }
    _ = posix.rmdir(dir_z.ptr);
}

test "PrefixDiskCache: init and deinit" {
    const allocator = std.testing.allocator;
    const cache_dir = "/tmp/mlx_prefix_disk_test_init";
    try ensureCacheDir(allocator, cache_dir);
    defer cleanupCacheDir(allocator, cache_dir);

    var cache = try PrefixDiskCache.init(allocator, cache_dir, 32);
    defer cache.deinit();

    try std.testing.expectEqual(@as(usize, 32), cache.block_size);
}

test "PrefixDiskCache: hasBlock returns false for non-existent block" {
    const allocator = std.testing.allocator;
    const cache_dir = "/tmp/mlx_prefix_disk_test_has";
    try ensureCacheDir(allocator, cache_dir);
    defer cleanupCacheDir(allocator, cache_dir);

    var cache = try PrefixDiskCache.init(allocator, cache_dir, 32);
    defer cache.deinit();

    try std.testing.expect(!cache.hasBlock(0x1234));
}

test "PrefixDiskCache: saveBlock and hasBlock round-trip" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const cache_dir = "/tmp/mlx_prefix_disk_test_save";
    try ensureCacheDir(allocator, cache_dir);
    defer cleanupCacheDir(allocator, cache_dir);

    var cache = try PrefixDiskCache.init(allocator, cache_dir, 4);
    defer cache.deinit();

    const stream = c.c.mlx_default_cpu_stream_new();
    const num_layers: usize = 2;
    const config = ModelConfig{
        .num_layers = num_layers,
        .num_kv_heads = 2,
        .head_dim = 4,
        .vocab_size = 100,
        .hidden_size = 32,
    };

    // Create per-layer KV arrays with different shapes (heterogeneous).
    var layer_keys: [2]Array = undefined;
    var layer_values: [2]Array = undefined;

    // Layer 0: shape [1, 2, 4, 4] (full head_dim)
    const shape0 = &[_]i32{ 1, 2, 4, 4 };
    var k0 = c.c.mlx_array_new();
    var v0 = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&k0, shape0.ptr, shape0.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&v0, shape0.ptr, shape0.len, c.c.MLX_FLOAT32, stream));
    layer_keys[0] = Array.fromHandle(k0);
    layer_values[0] = Array.fromHandle(v0);
    defer layer_keys[0].deinit();
    defer layer_values[0].deinit();

    // Layer 1: shape [1, 2, 4, 1] (compressed, different head_dim)
    const shape1 = &[_]i32{ 1, 2, 4, 1 };
    var k1 = c.c.mlx_array_new();
    var v1 = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&k1, shape1.ptr, shape1.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&v1, shape1.ptr, shape1.len, c.c.MLX_FLOAT32, stream));
    layer_keys[1] = Array.fromHandle(k1);
    layer_values[1] = Array.fromHandle(v1);
    defer layer_keys[1].deinit();
    defer layer_values[1].deinit();

    const block_hash: u64 = 0xDEADBEEF;
    try cache.saveBlock(block_hash, &layer_keys, &layer_values, config);

    try std.testing.expect(cache.hasBlock(block_hash));
    try std.testing.expect(!cache.hasBlock(0x12345678));
}

test "PrefixDiskCache: saveBlock and loadBlock round-trip with heterogeneous shapes" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const cache_dir = "/tmp/mlx_prefix_disk_test_load";
    try ensureCacheDir(allocator, cache_dir);
    defer cleanupCacheDir(allocator, cache_dir);

    var cache = try PrefixDiskCache.init(allocator, cache_dir, 4);
    defer cache.deinit();

    const stream = c.c.mlx_default_cpu_stream_new();
    const num_layers: usize = 3;
    const config = ModelConfig{
        .num_layers = num_layers,
        .num_kv_heads = 2,
        .head_dim = 8,
        .vocab_size = 100,
        .hidden_size = 32,
        .compress_ratios = &[_]usize{ 1, 4, 128 },
    };

    // Create heterogeneous per-layer KV arrays.
    var layer_keys: [3]Array = undefined;
    var layer_values: [3]Array = undefined;

    // Layer 0: sliding-window (ratio=1), full head_dim=8
    const shape0 = &[_]i32{ 1, 2, 4, 8 };
    var k0 = c.c.mlx_array_new();
    var v0 = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&k0, shape0.ptr, shape0.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&v0, shape0.ptr, shape0.len, c.c.MLX_FLOAT32, stream));
    layer_keys[0] = Array.fromHandle(k0);
    layer_values[0] = Array.fromHandle(v0);
    defer layer_keys[0].deinit();
    defer layer_values[0].deinit();

    // Layer 1: CSA (ratio=4), compressed head_dim=2
    const shape1 = &[_]i32{ 1, 2, 4, 2 };
    var k1 = c.c.mlx_array_new();
    var v1 = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&k1, shape1.ptr, shape1.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&v1, shape1.ptr, shape1.len, c.c.MLX_FLOAT32, stream));
    layer_keys[1] = Array.fromHandle(k1);
    layer_values[1] = Array.fromHandle(v1);
    defer layer_keys[1].deinit();
    defer layer_values[1].deinit();

    // Layer 2: HCA (ratio=128), heavily compressed head_dim=1
    const shape2 = &[_]i32{ 1, 1, 4, 1 };
    var k2 = c.c.mlx_array_new();
    var v2 = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&k2, shape2.ptr, shape2.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&v2, shape2.ptr, shape2.len, c.c.MLX_FLOAT32, stream));
    layer_keys[2] = Array.fromHandle(k2);
    layer_values[2] = Array.fromHandle(v2);
    defer layer_keys[2].deinit();
    defer layer_values[2].deinit();

    const block_hash: u64 = 0xCAFEBABE;
    try cache.saveBlock(block_hash, &layer_keys, &layer_values, config);

    // Load and verify shapes match.
    var loaded = try cache.loadBlock(block_hash, config);
    defer loaded.deinit(allocator);

    try std.testing.expectEqual(num_layers, loaded.layer_keys.len);
    try std.testing.expectEqual(num_layers, loaded.layer_values.len);

    // Verify each layer's shape is preserved.
    const expected_shapes = [_][]const i32{ shape0, shape1, shape2 };
    for (0..num_layers) |i| {
        const loaded_shape = loaded.layer_keys[i].shape();
        try std.testing.expectEqual(expected_shapes[i].len, loaded_shape.len);
        for (expected_shapes[i], loaded_shape) |expected, actual| {
            try std.testing.expectEqual(expected, actual);
        }
    }

    // Verify data is element-wise equal.
    for (0..num_layers) |i| {
        try c.check(c.c.mlx_array_eval(loaded.layer_keys[i].inner));
        try c.check(c.c.mlx_array_eval(loaded.layer_values[i].inner));
        const orig_k = try layer_keys[i].dataSlice(f32);
        const loaded_k = try loaded.layer_keys[i].dataSlice(f32);
        try std.testing.expectEqual(orig_k.len, loaded_k.len);
        for (orig_k, loaded_k) |a, b| {
            try std.testing.expectEqual(a, b);
        }
    }
}

test "PrefixDiskCache: restorePrefix with BlockManager integration" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const cache_dir = "/tmp/mlx_prefix_disk_test_restore";
    try ensureCacheDir(allocator, cache_dir);
    defer cleanupCacheDir(allocator, cache_dir);

    const block_size: usize = 4;
    var cache = try PrefixDiskCache.init(allocator, cache_dir, block_size);
    defer cache.deinit();

    const stream = c.c.mlx_default_cpu_stream_new();
    const num_layers: usize = 2;
    const config = ModelConfig{
        .num_layers = num_layers,
        .num_kv_heads = 2,
        .head_dim = 4,
        .vocab_size = 100,
        .hidden_size = 32,
    };

    // Simulate saving two prefix blocks for token sequence [1,2,3,4, 5,6,7,8].
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };

    var prev_hash: u64 = 0;
    for (0..2) |block_idx| {
        const block_tokens = tokens[block_idx * block_size .. (block_idx + 1) * block_size];
        const hash = BlockManager.hashBlock(prev_hash, block_tokens);

        var layer_keys: [2]Array = undefined;
        var layer_values: [2]Array = undefined;
        for (0..num_layers) |li| {
            const shape = &[_]i32{ 1, 2, @intCast(block_size), 4 };
            var k = c.c.mlx_array_new();
            var v = c.c.mlx_array_new();
            try c.check(c.c.mlx_ones(&k, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
            try c.check(c.c.mlx_ones(&v, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
            layer_keys[li] = Array.fromHandle(k);
            layer_values[li] = Array.fromHandle(v);
        }
        defer for (0..num_layers) |li| {
            layer_keys[li].deinit();
            layer_values[li].deinit();
        };

        try cache.saveBlock(hash, &layer_keys, &layer_values, config);
        prev_hash = hash;
    }

    // Now restore the prefix using a BlockManager.
    var bm = try BlockManager.init(allocator, 16);
    defer bm.deinit();

    var restored = try cache.restorePrefix(&bm, &tokens, config);
    defer restored.deinit(allocator);

    // Should have restored 2 blocks = 8 tokens.
    try std.testing.expectEqual(@as(usize, 2), restored.block_ids.len);
    try std.testing.expectEqual(@as(usize, 8), restored.num_tokens);

    // Verify blocks are registered in BlockManager's hash map.
    prev_hash = 0;
    for (0..2) |block_idx| {
        const block_tokens = tokens[block_idx * block_size .. (block_idx + 1) * block_size];
        const hash = BlockManager.hashBlock(prev_hash, block_tokens);
        try std.testing.expect(bm.block_hashes.contains(hash));
        prev_hash = hash;
    }

    // Clean up by freeing blocks via their request IDs (which are the hashes).
    prev_hash = 0;
    for (0..2) |block_idx| {
        const block_tokens = tokens[block_idx * block_size .. (block_idx + 1) * block_size];
        const hash = BlockManager.hashBlock(prev_hash, block_tokens);
        bm.freeBlocks(hash);
        prev_hash = hash;
    }
}

test "PrefixDiskCache: restorePrefix with partial match" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const cache_dir = "/tmp/mlx_prefix_disk_test_partial";
    try ensureCacheDir(allocator, cache_dir);
    defer cleanupCacheDir(allocator, cache_dir);

    const block_size: usize = 4;
    var cache = try PrefixDiskCache.init(allocator, cache_dir, block_size);
    defer cache.deinit();

    const stream = c.c.mlx_default_cpu_stream_new();
    const config = ModelConfig{
        .num_layers = 1,
        .num_kv_heads = 1,
        .head_dim = 2,
        .vocab_size = 100,
        .hidden_size = 16,
    };

    // Save only the first block.
    const tokens = [_]u32{ 10, 20, 30, 40 };
    const hash = BlockManager.hashBlock(0, &tokens);

    var layer_keys = [_]Array{undefined};
    var layer_values = [_]Array{undefined};
    const shape = &[_]i32{ 1, 1, 4, 2 };
    var k = c.c.mlx_array_new();
    var v = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&k, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&v, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    layer_keys[0] = Array.fromHandle(k);
    layer_values[0] = Array.fromHandle(v);
    defer layer_keys[0].deinit();
    defer layer_values[0].deinit();

    try cache.saveBlock(hash, &layer_keys, &layer_values, config);

    // Try to restore with a longer sequence — only first block should match.
    const longer_tokens = [_]u32{ 10, 20, 30, 40, 50, 60, 70, 80 };
    var bm = try BlockManager.init(allocator, 16);
    defer bm.deinit();

    var restored = try cache.restorePrefix(&bm, &longer_tokens, config);
    defer restored.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), restored.block_ids.len);
    try std.testing.expectEqual(@as(usize, 4), restored.num_tokens);

    // Clean up.
    bm.freeBlocks(hash);
}

test "PrefixDiskCache: removeBlock deletes file" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const cache_dir = "/tmp/mlx_prefix_disk_test_remove";
    try ensureCacheDir(allocator, cache_dir);
    defer cleanupCacheDir(allocator, cache_dir);

    var cache = try PrefixDiskCache.init(allocator, cache_dir, 4);
    defer cache.deinit();

    const stream = c.c.mlx_default_cpu_stream_new();
    const config = ModelConfig{
        .num_layers = 1,
        .num_kv_heads = 1,
        .head_dim = 2,
        .vocab_size = 100,
        .hidden_size = 16,
    };

    const shape = &[_]i32{ 1, 1, 4, 2 };
    var k = c.c.mlx_array_new();
    var v = c.c.mlx_array_new();
    try c.check(c.c.mlx_ones(&k, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    try c.check(c.c.mlx_ones(&v, shape.ptr, shape.len, c.c.MLX_FLOAT32, stream));
    const key_arr = Array.fromHandle(k);
    const val_arr = Array.fromHandle(v);
    defer key_arr.deinit();
    defer val_arr.deinit();

    const block_hash: u64 = 0xBEEF;
    try cache.saveBlock(block_hash, &[_]Array{key_arr}, &[_]Array{val_arr}, config);
    try std.testing.expect(cache.hasBlock(block_hash));

    cache.removeBlock(block_hash);
    try std.testing.expect(!cache.hasBlock(block_hash));
}

test "PrefixDiskCache: restorePrefix with empty tokens" {
    const allocator = std.testing.allocator;
    const cache_dir = "/tmp/mlx_prefix_disk_test_empty";
    try ensureCacheDir(allocator, cache_dir);
    defer cleanupCacheDir(allocator, cache_dir);

    var cache = try PrefixDiskCache.init(allocator, cache_dir, 4);
    defer cache.deinit();

    const config = ModelConfig{
        .num_layers = 1,
        .num_kv_heads = 1,
        .head_dim = 2,
        .vocab_size = 100,
        .hidden_size = 16,
    };

    var bm = try BlockManager.init(allocator, 8);
    defer bm.deinit();

    const empty_tokens = [_]u32{};
    var restored = try cache.restorePrefix(&bm, &empty_tokens, config);
    defer restored.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), restored.block_ids.len);
    try std.testing.expectEqual(@as(usize, 0), restored.num_tokens);
}
