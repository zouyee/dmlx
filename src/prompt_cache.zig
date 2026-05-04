/// Prompt Cache Persistence — Save/load KV cache state to/from safetensors files.
///
/// Enables skipping redundant prefill computation for recurring long prompts
/// (system prompts, RAG context) by persisting KV cache state to disk.
///
/// Serialization format:
///   - Each layer's keys stored as "layer_{i}_keys"
///   - Each layer's values stored as "layer_{i}_values"
///   - Metadata: num_layers, head_dim, num_kv_heads, seq_len, dtype
///
/// On load, metadata is validated against the current model config.
/// A descriptive error is returned on mismatch.
const std = @import("std");
const io = @import("io/mlx_io.zig");
const kvcache = @import("kvcache.zig");
const generation = @import("generation.zig");
const array_mod = @import("array.zig");
const dtype_mod = @import("dtype.zig");
const c = @import("c.zig");

const Array = array_mod.Array;
const KVCacheStrategy = kvcache.KVCacheStrategy;
const StandardKVCache = kvcache.StandardKVCache;
const ModelConfig = generation.ModelConfig;
const Dtype = dtype_mod.Dtype;

pub const PromptCacheError = error{
    NumLayersMismatch,
    HeadDimMismatch,
    NumKvHeadsMismatch,
    MissingMetadata,
    InvalidMetadata,
    MissingCacheData,
};

/// Save KV cache state to a safetensors file.
///
/// For each layer, extracts the current keys and values arrays and stores them
/// with metadata describing the cache configuration for later validation.
pub fn savePromptCache(
    allocator: std.mem.Allocator,
    caches: []KVCacheStrategy,
    path: []const u8,
) !void {
    var weights = std.StringHashMap(Array).init(allocator);
    defer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            // Don't deinit arrays — they are views into the cache
        }
        weights.deinit();
    }

    var metadata = std.StringHashMap([]const u8).init(allocator);
    defer {
        var it = metadata.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        metadata.deinit();
    }

    // Extract keys/values from each layer's cache
    var seq_len: usize = 0;
    var head_dim: usize = 0;
    var num_kv_heads: usize = 0;
    var cache_dtype: Dtype = .float32;

    for (caches, 0..) |cache, i| {
        // Use VTable getState to safely access cache internals
        const state = cache.getState() orelse {
            std.log.warn("savePromptCache: layer {d} cache type does not support getState, skipping", .{i});
            continue;
        };

        const current_len = state.offset;
        if (current_len == 0) continue;

        // Slice the valid portion of keys and values
        const stream = c.c.mlx_default_cpu_stream_new();
        defer _ = c.c.mlx_stream_free(stream);
        const keys = try sliceCache(state.keys, current_len, stream);
        const values = try sliceCache(state.values, current_len, stream);

        // Store shape info from first non-empty layer
        if (seq_len == 0) {
            const shape = keys.shape();
            // Shape: [batch, num_kv_heads, seq_len, head_dim]
            num_kv_heads = @intCast(shape[1]);
            seq_len = @intCast(shape[2]);
            head_dim = @intCast(shape[3]);
            cache_dtype = keys.dtype();
        }

        const key_name = try std.fmt.allocPrint(allocator, "layer_{d}_keys", .{i});
        const val_name = try std.fmt.allocPrint(allocator, "layer_{d}_values", .{i});

        try weights.put(key_name, keys);
        try weights.put(val_name, values);
    }

    // Store metadata
    const num_layers_str = try std.fmt.allocPrint(allocator, "{d}", .{caches.len});
    const num_layers_key = try allocator.dupe(u8, "num_layers");
    try metadata.put(num_layers_key, num_layers_str);

    const head_dim_str = try std.fmt.allocPrint(allocator, "{d}", .{head_dim});
    const head_dim_key = try allocator.dupe(u8, "head_dim");
    try metadata.put(head_dim_key, head_dim_str);

    const num_kv_heads_str = try std.fmt.allocPrint(allocator, "{d}", .{num_kv_heads});
    const num_kv_heads_key = try allocator.dupe(u8, "num_kv_heads");
    try metadata.put(num_kv_heads_key, num_kv_heads_str);

    const seq_len_str = try std.fmt.allocPrint(allocator, "{d}", .{seq_len});
    const seq_len_key = try allocator.dupe(u8, "seq_len");
    try metadata.put(seq_len_key, seq_len_str);

    const dtype_str = try allocator.dupe(u8, dtypeToString(cache_dtype));
    const dtype_key = try allocator.dupe(u8, "dtype");
    try metadata.put(dtype_key, dtype_str);

    try io.saveSafetensors(allocator, path, weights, metadata);
}

/// Load KV cache state from a safetensors file, validating against model config.
///
/// Returns an error with a descriptive message if the cached state is
/// incompatible with the current model configuration.
pub fn loadPromptCache(
    allocator: std.mem.Allocator,
    path: []const u8,
    model_config: ModelConfig,
) ![]KVCacheStrategy {
    var result = try io.loadSafetensors(allocator, path);
    defer result.deinit(allocator);

    // Validate metadata
    const num_layers = try parseMetadataUsize(result.metadata, "num_layers") orelse
        return PromptCacheError.MissingMetadata;
    const head_dim = try parseMetadataUsize(result.metadata, "head_dim") orelse
        return PromptCacheError.MissingMetadata;
    const num_kv_heads = try parseMetadataUsize(result.metadata, "num_kv_heads") orelse
        return PromptCacheError.MissingMetadata;
    const seq_len = try parseMetadataUsize(result.metadata, "seq_len") orelse
        return PromptCacheError.MissingMetadata;

    // Validate against model config
    if (num_layers != model_config.num_layers) {
        std.log.warn(
            "Prompt cache incompatible: num_layers mismatch (cache={d}, model={d})",
            .{ num_layers, model_config.num_layers },
        );
        return PromptCacheError.NumLayersMismatch;
    }
    if (head_dim != model_config.head_dim) {
        std.log.warn(
            "Prompt cache incompatible: head_dim mismatch (cache={d}, model={d})",
            .{ head_dim, model_config.head_dim },
        );
        return PromptCacheError.HeadDimMismatch;
    }
    if (num_kv_heads != model_config.num_kv_heads) {
        std.log.warn(
            "Prompt cache incompatible: num_kv_heads mismatch (cache={d}, model={d})",
            .{ num_kv_heads, model_config.num_kv_heads },
        );
        return PromptCacheError.NumKvHeadsMismatch;
    }

    // Determine dtype from metadata or first array
    var cache_dtype: Dtype = .float32;
    if (result.metadata.get("dtype")) |dtype_str| {
        cache_dtype = stringToDtype(dtype_str) orelse .float32;
    }

    // Reconstruct KV caches
    const stream = c.c.mlx_default_cpu_stream_new();
    defer _ = c.c.mlx_stream_free(stream);
    var caches = try allocator.alloc(KVCacheStrategy, num_layers);
    errdefer {
        for (caches) |*cache_strategy| {
            cache_strategy.deinit(allocator);
        }
        allocator.free(caches);
    }

    for (0..num_layers) |i| {
        const key_name = try std.fmt.allocPrint(allocator, "layer_{d}_keys", .{i});
        defer allocator.free(key_name);
        const val_name = try std.fmt.allocPrint(allocator, "layer_{d}_values", .{i});
        defer allocator.free(val_name);

        const config = kvcache.LayerConfig{
            .batch_size = 1,
            .num_heads = num_kv_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .max_seq_len = seq_len * 2, // Allow room for additional generation
            .dtype = cache_dtype,
        };

        const cache_ptr = try allocator.create(StandardKVCache);
        errdefer allocator.destroy(cache_ptr);
        cache_ptr.* = try StandardKVCache.init(allocator, config, stream);

        // If we have saved data for this layer, load it into the cache
        const maybe_keys = result.weights.get(key_name);
        const maybe_values = result.weights.get(val_name);

        if (maybe_keys != null and maybe_values != null) {
            const keys = maybe_keys.?;
            const values = maybe_values.?;

            // Copy the loaded arrays into the cache buffer via slice update
            try sliceUpdateCache(&cache_ptr.keys, keys, 0, seq_len, stream);
            try sliceUpdateCache(&cache_ptr.values, values, 0, seq_len, stream);
            cache_ptr.offset = seq_len;
        }

        caches[i] = cache_ptr.asStrategy();
    }

    return caches;
}

/// Free an array of KV cache strategies returned by loadPromptCache.
pub fn freePromptCaches(allocator: std.mem.Allocator, caches: []KVCacheStrategy) void {
    for (caches) |cache| {
        cache.deinit(allocator);
    }
    allocator.free(caches);
}

// ============================================================
// Internal helpers
// ============================================================

/// Slice the valid portion of a cache buffer: buffer[..., 0:len, :]
fn sliceCache(buffer: Array, len: usize, stream: c.c.mlx_stream) !Array {
    const start = &[_]i32{ 0, 0, 0, 0 };
    const stop = &[_]i32{ std.math.maxInt(i32), std.math.maxInt(i32), @intCast(len), std.math.maxInt(i32) };
    const strides = &[_]i32{ 1, 1, 1, 1 };

    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_slice(
        &res,
        buffer.inner,
        start.ptr,
        start.len,
        stop.ptr,
        stop.len,
        strides.ptr,
        strides.len,
        stream,
    ));
    return Array.fromHandle(res);
}

/// In-place update: buffer[..., offset:end, :] = src
fn sliceUpdateCache(
    buffer: *Array,
    src: Array,
    offset: usize,
    end: usize,
    stream: c.c.mlx_stream,
) !void {
    const start = &[_]i32{ 0, 0, @intCast(offset), 0 };
    const stop = &[_]i32{ std.math.maxInt(i32), std.math.maxInt(i32), @intCast(end), std.math.maxInt(i32) };
    const strides = &[_]i32{ 1, 1, 1, 1 };

    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_slice_update(
        &res,
        buffer.inner,
        src.inner,
        start.ptr,
        start.len,
        stop.ptr,
        stop.len,
        strides.ptr,
        strides.len,
        stream,
    ));

    buffer.deinit();
    buffer.* = Array.fromHandle(res);
}

/// Parse a usize value from metadata string map.
fn parseMetadataUsize(
    metadata: std.StringHashMap([]const u8),
    key: []const u8,
) !?usize {
    const val_str = metadata.get(key) orelse return null;
    return std.fmt.parseInt(usize, val_str, 10) catch return PromptCacheError.InvalidMetadata;
}

/// Convert Dtype enum to string for metadata storage.
fn dtypeToString(dt: Dtype) []const u8 {
    return switch (dt) {
        .float32 => "float32",
        .float16 => "float16",
        .bfloat16 => "bfloat16",
        .float64 => "float64",
        .int32 => "int32",
        .int64 => "int64",
        .uint32 => "uint32",
        .uint64 => "uint64",
        .int8 => "int8",
        .int16 => "int16",
        .uint8 => "uint8",
        .uint16 => "uint16",
        .bool_ => "bool",
        .complex64 => "complex64",
    };
}

/// Convert string back to Dtype enum.
fn stringToDtype(s: []const u8) ?Dtype {
    const map = std.StaticStringMap(Dtype).initComptime(.{
        .{ "float32", .float32 },
        .{ "float16", .float16 },
        .{ "bfloat16", .bfloat16 },
        .{ "float64", .float64 },
        .{ "int32", .int32 },
        .{ "int64", .int64 },
        .{ "uint32", .uint32 },
        .{ "uint64", .uint64 },
        .{ "int8", .int8 },
        .{ "int16", .int16 },
        .{ "uint8", .uint8 },
        .{ "uint16", .uint16 },
        .{ "bool", .bool_ },
        .{ "complex64", .complex64 },
    });
    return map.get(s);
}

// ============================================================
// Tests
// ============================================================

test "dtypeToString round-trip" {
    const types = [_]Dtype{ .float32, .float16, .bfloat16, .int32 };
    for (types) |dt| {
        const s = dtypeToString(dt);
        const recovered = stringToDtype(s).?;
        try std.testing.expectEqual(dt, recovered);
    }
}

test "parseMetadataUsize valid" {
    const allocator = std.testing.allocator;
    var map = std.StringHashMap([]const u8).init(allocator);
    defer map.deinit();
    try map.put("num_layers", "32");
    try map.put("head_dim", "128");

    const nl = (try parseMetadataUsize(map, "num_layers")).?;
    try std.testing.expectEqual(@as(usize, 32), nl);
    const hd = (try parseMetadataUsize(map, "head_dim")).?;
    try std.testing.expectEqual(@as(usize, 128), hd);
}

test "parseMetadataUsize missing key" {
    const allocator = std.testing.allocator;
    var map = std.StringHashMap([]const u8).init(allocator);
    defer map.deinit();

    const result = try parseMetadataUsize(map, "missing");
    try std.testing.expectEqual(@as(?usize, null), result);
}

// ============================================================
// Property-Based Test: Prompt Cache Round-Trip (Property 5)
//
// **Validates: Requirements R7.1, R7.2, R7.3, R7.4**
//
// For any valid KV cache state:
//   1. save → load with matching config produces element-wise equal KV data
//   2. load with mismatched config (num_layers, head_dim, num_kv_heads) returns error
//
// Runs 100 iterations with randomly generated cache configurations.
// ============================================================

test "Property 5: Prompt Cache Round-Trip — save/load equality and mismatch error (100 iterations)" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(0xCAFE_BABE);
    const rand = prng.random();

    const num_iterations: usize = 100;

    for (0..num_iterations) |iteration| {
        _ = iteration;

        // --- Generate random config (keep small to avoid huge allocations) ---
        const num_layers = rand.intRangeAtMost(usize, 1, 4);
        const num_kv_heads = rand.intRangeAtMost(usize, 1, 4);
        const head_dim_choices = [_]usize{ 4, 8, 16 };
        const head_dim = head_dim_choices[rand.intRangeLessThan(usize, 0, head_dim_choices.len)];
        const seq_len = rand.intRangeAtMost(usize, 1, 16);

        const stream = c.c.mlx_default_cpu_stream_new();
        defer _ = c.c.mlx_stream_free(stream);
        var caches = try allocator.alloc(KVCacheStrategy, num_layers);
        defer {
            for (caches) |cache_strat| {
                cache_strat.deinit(allocator);
            }
            allocator.free(caches);
        }

        for (0..num_layers) |layer_idx| {
            const config = kvcache.LayerConfig{
                .batch_size = 1,
                .num_heads = num_kv_heads,
                .num_kv_heads = num_kv_heads,
                .head_dim = head_dim,
                .max_seq_len = seq_len * 2,
                .dtype = .float32,
            };

            const cache_ptr = try allocator.create(StandardKVCache);
            errdefer allocator.destroy(cache_ptr);
            cache_ptr.* = try StandardKVCache.init(allocator, config, stream);

            // Generate random key/value data: [1, num_kv_heads, seq_len, head_dim]
            const total_elems = num_kv_heads * seq_len * head_dim;
            var key_data = try allocator.alloc(f32, total_elems);
            defer allocator.free(key_data);
            var val_data = try allocator.alloc(f32, total_elems);
            defer allocator.free(val_data);

            for (0..total_elems) |j| {
                key_data[j] = @as(f32, @floatFromInt(rand.intRangeAtMost(i32, -1000, 1000))) / 100.0;
                val_data[j] = @as(f32, @floatFromInt(rand.intRangeAtMost(i32, -1000, 1000))) / 100.0;
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

            // Update the cache via the strategy interface
            caches[layer_idx] = cache_ptr.asStrategy();
            _ = try caches[layer_idx].updateAndFetch(keys_arr, vals_arr, stream);
        }

        // --- Save to temp file ---
        const tmp_path = "/tmp/mlx_zig_pbt_prompt_cache.safetensors";
        try savePromptCache(allocator, caches, tmp_path);

        // --- Load with matching config → verify element-wise equality ---
        const matching_config = generation.ModelConfig{
            .num_layers = num_layers,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .vocab_size = 32000,
            .hidden_size = 128,
        };

        const loaded_caches = try loadPromptCache(allocator, tmp_path, matching_config);
        defer freePromptCaches(allocator, loaded_caches);

        // Verify same number of layers
        try std.testing.expectEqual(num_layers, loaded_caches.len);

        // Verify element-wise equality for each layer
        for (0..num_layers) |li| {
            const orig_cache: *StandardKVCache = @ptrCast(@alignCast(caches[li].ptr));
            const loaded_cache: *StandardKVCache = @ptrCast(@alignCast(loaded_caches[li].ptr));

            // Both should have the same offset
            try std.testing.expectEqual(orig_cache.offset, loaded_cache.offset);

            // Compare keys element-wise
            const orig_keys = try sliceCache(orig_cache.keys, orig_cache.offset, stream);
            defer orig_keys.deinit();
            const loaded_keys = try sliceCache(loaded_cache.keys, loaded_cache.offset, stream);
            defer loaded_keys.deinit();

            const orig_k_data = try orig_keys.dataSlice(f32);
            const loaded_k_data = try loaded_keys.dataSlice(f32);
            try std.testing.expectEqual(orig_k_data.len, loaded_k_data.len);
            for (orig_k_data, loaded_k_data) |a, b| {
                try std.testing.expectEqual(a, b);
            }

            // Compare values element-wise
            const orig_vals = try sliceCache(orig_cache.values, orig_cache.offset, stream);
            defer orig_vals.deinit();
            const loaded_vals = try sliceCache(loaded_cache.values, loaded_cache.offset, stream);
            defer loaded_vals.deinit();

            const orig_v_data = try orig_vals.dataSlice(f32);
            const loaded_v_data = try loaded_vals.dataSlice(f32);
            try std.testing.expectEqual(orig_v_data.len, loaded_v_data.len);
            for (orig_v_data, loaded_v_data) |a, b| {
                try std.testing.expectEqual(a, b);
            }
        }

        // --- Load with mismatched num_layers → verify error ---
        const bad_layers_config = generation.ModelConfig{
            .num_layers = num_layers + 1,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .vocab_size = 32000,
            .hidden_size = 128,
        };
        if (loadPromptCache(allocator, tmp_path, bad_layers_config)) |bad_caches| {
            freePromptCaches(allocator, bad_caches);
            return error.TestUnexpectedResult;
        } else |err| {
            try std.testing.expectEqual(PromptCacheError.NumLayersMismatch, err);
        }

        // --- Load with mismatched head_dim → verify error ---
        const bad_head_config = generation.ModelConfig{
            .num_layers = num_layers,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim + 1,
            .vocab_size = 32000,
            .hidden_size = 128,
        };
        if (loadPromptCache(allocator, tmp_path, bad_head_config)) |bad_caches| {
            freePromptCaches(allocator, bad_caches);
            return error.TestUnexpectedResult;
        } else |err| {
            try std.testing.expectEqual(PromptCacheError.HeadDimMismatch, err);
        }

        // --- Load with mismatched num_kv_heads → verify error ---
        const bad_kv_heads_config = generation.ModelConfig{
            .num_layers = num_layers,
            .num_kv_heads = num_kv_heads + 1,
            .head_dim = head_dim,
            .vocab_size = 32000,
            .hidden_size = 128,
        };
        if (loadPromptCache(allocator, tmp_path, bad_kv_heads_config)) |bad_caches| {
            freePromptCaches(allocator, bad_caches);
            return error.TestUnexpectedResult;
        } else |err| {
            try std.testing.expectEqual(PromptCacheError.NumKvHeadsMismatch, err);
        }

        // Temp file is overwritten each iteration; no explicit cleanup needed.
    }
}
