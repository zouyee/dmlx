/// Integration tests for cache-aware streamingForward behavior.
///
/// Tests the ExpertCache + PartialTensorReader integration that underpins
/// the cache-first loading strategy in streamingForward. Since
/// `loadExpertSlicesCached` is private to ExpertStreamProvider, these tests
/// exercise the same logic directly: check cache → on miss load via
/// PartialTensorReader → insert into cache → verify stats.
///
/// _Requirements: 5.1, 5.2_
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const ops = @import("../ops.zig");
const safetensors_reader = @import("../io/safetensors_reader.zig");
const expert_cache = @import("../models/expert_cache.zig");

const Array = array_mod.Array;
const TensorIndex = safetensors_reader.TensorIndex;
const FdPool = safetensors_reader.FdPool;
const PartialTensorReader = safetensors_reader.PartialTensorReader;
const ExpertCache = expert_cache.ExpertCache;
const CacheKey = expert_cache.CacheKey;

const posix_c = @cImport({
    @cInclude("fcntl.h");
    @cInclude("unistd.h");
});

// ── Helpers ──

/// Write a minimal safetensors file containing a single fused expert tensor.
/// Each expert row is filled with a deterministic pattern: expert i has all values = (i+1)*0.1.
fn writeMockSafetensors(
    allocator: std.mem.Allocator,
    path: []const u8,
    tensor_name: []const u8,
    n_experts: u32,
    dim1: u32,
    dim2: u32,
) !void {
    const row_elements: usize = @as(usize, dim1) * @as(usize, dim2);
    const total_elements: usize = @as(usize, n_experts) * row_elements;
    const elem_bytes: usize = 4; // f32
    const total_data_bytes: usize = total_elements * elem_bytes;

    const data = try allocator.alloc(f32, total_elements);
    defer allocator.free(data);

    // Fill each expert row with a known pattern
    for (0..n_experts) |eid| {
        const val: f32 = @as(f32, @floatFromInt(eid + 1)) * 0.1;
        const row_start = eid * row_elements;
        for (0..row_elements) |j| {
            data[row_start + j] = val;
        }
    }

    const header_json = try std.fmt.allocPrint(
        allocator,
        "{{\"{s}\": {{\"dtype\": \"F32\", \"shape\": [{d}, {d}, {d}], \"data_offsets\": [0, {d}]}}}}",
        .{ tensor_name, n_experts, dim1, dim2, total_data_bytes },
    );
    defer allocator.free(header_json);

    const header_len: u64 = @intCast(header_json.len);

    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    const fd = posix_c.open(path_z.ptr, posix_c.O_WRONLY | posix_c.O_CREAT | posix_c.O_TRUNC, @as(c_uint, 0o644));
    if (fd < 0) return error.FileCreateFailed;
    defer _ = posix_c.close(fd);

    var header_len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &header_len_buf, header_len, .little);
    _ = posix_c.write(fd, &header_len_buf, 8);
    _ = posix_c.write(fd, header_json.ptr, header_json.len);

    const data_bytes: [*]const u8 = @ptrCast(data.ptr);
    _ = posix_c.write(fd, data_bytes, total_data_bytes);
}

/// Write a safetensors file with two tensors (simulating gate_proj and up_proj).
fn writeMockSafetensorsMulti(
    allocator: std.mem.Allocator,
    path: []const u8,
    tensor_name_1: []const u8,
    tensor_name_2: []const u8,
    n_experts: u32,
    dim1: u32,
    dim2: u32,
) !void {
    const row_elements: usize = @as(usize, dim1) * @as(usize, dim2);
    const total_elements: usize = @as(usize, n_experts) * row_elements;
    const elem_bytes: usize = 4;
    const tensor_data_bytes: usize = total_elements * elem_bytes;
    const total_data_bytes: usize = tensor_data_bytes * 2; // two tensors

    const data1 = try allocator.alloc(f32, total_elements);
    defer allocator.free(data1);
    const data2 = try allocator.alloc(f32, total_elements);
    defer allocator.free(data2);

    for (0..n_experts) |eid| {
        const val1: f32 = @as(f32, @floatFromInt(eid + 1)) * 0.1;
        const val2: f32 = @as(f32, @floatFromInt(eid + 1)) * 0.2;
        const row_start = eid * row_elements;
        for (0..row_elements) |j| {
            data1[row_start + j] = val1;
            data2[row_start + j] = val2;
        }
    }

    const header_json = try std.fmt.allocPrint(
        allocator,
        "{{\"{s}\": {{\"dtype\": \"F32\", \"shape\": [{d}, {d}, {d}], \"data_offsets\": [0, {d}]}}, " ++
            "\"{s}\": {{\"dtype\": \"F32\", \"shape\": [{d}, {d}, {d}], \"data_offsets\": [{d}, {d}]}}}}",
        .{
            tensor_name_1,    n_experts, dim1, dim2, tensor_data_bytes,
            tensor_name_2,    n_experts, dim1, dim2, tensor_data_bytes,
            total_data_bytes,
        },
    );
    defer allocator.free(header_json);

    const header_len: u64 = @intCast(header_json.len);

    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    const fd = posix_c.open(path_z.ptr, posix_c.O_WRONLY | posix_c.O_CREAT | posix_c.O_TRUNC, @as(c_uint, 0o644));
    if (fd < 0) return error.FileCreateFailed;
    defer _ = posix_c.close(fd);

    var header_len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &header_len_buf, header_len, .little);
    _ = posix_c.write(fd, &header_len_buf, 8);
    _ = posix_c.write(fd, header_json.ptr, header_json.len);

    const data1_bytes: [*]const u8 = @ptrCast(data1.ptr);
    _ = posix_c.write(fd, data1_bytes, tensor_data_bytes);
    const data2_bytes: [*]const u8 = @ptrCast(data2.ptr);
    _ = posix_c.write(fd, data2_bytes, tensor_data_bytes);
}

fn removeTempFile(path: []const u8, allocator: std.mem.Allocator) void {
    const path_z = allocator.dupeZ(u8, path) catch return;
    defer allocator.free(path_z);
    _ = posix_c.unlink(path_z.ptr);
}

/// Build a cache key for a (layer, tensor_name, expert_ids) combination,
/// matching the hashing strategy used in loadExpertSlicesCached.
fn buildCacheKey(layer_idx: u32, tensor_name: []const u8, expert_ids: []const u32) CacheKey {
    const name_hash = expert_cache.hashTensorName(tensor_name);
    var ids_hasher = std.hash.Wyhash.init(name_hash);
    ids_hasher.update(std.mem.sliceAsBytes(expert_ids));
    const ids_hash = ids_hasher.final();
    return CacheKey{
        .layer_idx = layer_idx,
        .tensor_name_hash = ids_hash,
        .expert_id = @intCast(expert_ids.len),
    };
}

/// Simulate the cache-first loading strategy from loadExpertSlicesCached:
/// 1. Check cache for the key
/// 2. On miss, load via PartialTensorReader
/// 3. Insert a copy into the cache
/// Returns the loaded tensor (caller owns it).
fn cacheAwareLoad(
    allocator: std.mem.Allocator,
    cache_inst: *ExpertCache,
    reader: *PartialTensorReader,
    tensor_name: []const u8,
    expert_ids: []const u32,
    layer_idx: u32,
) !Array {
    const key = buildCacheKey(layer_idx, tensor_name, expert_ids);

    // Check cache
    if (cache_inst.get(key)) |cached_tensor| {
        // Cache hit — return a copy (caller owns it, cache retains original)
        const ctx = ops.EagerContext.init(allocator);
        return ops.copy(ctx, cached_tensor);
    }

    // Cache miss — load via partial reader
    const tensor = try reader.readExpertRows(tensor_name, expert_ids);

    // Compute byte size for cache tracking
    const info = reader.index.entries.get(tensor_name) orelse return error.TensorNotFound;
    const range = reader.computeExpertByteRange(&info, 0);
    const total_bytes = range.length * expert_ids.len;

    // Insert a copy into cache (cache owns the copy, we return the original)
    const ctx = ops.EagerContext.init(allocator);
    const cached_copy = try ops.copy(ctx, tensor);
    cache_inst.put(key, cached_copy, total_bytes);

    return tensor;
}

// ── Tests ──

test "cache integration: cache hit path — no disk I/O on second call" {
    // Validates: Requirement 5.1 (cache checked before disk I/O)
    // Validates: Requirement 5.2 (cache miss triggers partial read + insert)
    const allocator = std.testing.allocator;

    const tmp_path = "/tmp/dmlx_cache_hit_test.safetensors";
    const tensor_name = "layer0.gate_proj.weight";
    const n_experts: u32 = 8;
    const dim1: u32 = 4;
    const dim2: u32 = 4;

    try writeMockSafetensors(allocator, tmp_path, tensor_name, n_experts, dim1, dim2);
    defer removeTempFile(tmp_path, allocator);

    // Build infrastructure
    var index = TensorIndex.init(allocator);
    defer index.deinit();
    try index.addShard(tmp_path);

    var fd_pool = FdPool.init(allocator);
    defer fd_pool.deinit();
    try fd_pool.openAll(&index);

    // Large cache budget so nothing gets evicted
    var cache_inst = ExpertCache.init(allocator, 1024 * 1024);
    defer cache_inst.deinit();

    var reader = PartialTensorReader.init(allocator, &index, &fd_pool);

    const expert_ids = [_]u32{ 1, 3, 5 };
    const layer_idx: u32 = 0;

    // First call: should be a cache miss, loads from disk
    const result1 = try cacheAwareLoad(allocator, &cache_inst, &reader, tensor_name, &expert_ids, layer_idx);
    defer result1.deinit();

    const stats1 = cache_inst.stats();
    try std.testing.expectEqual(@as(u64, 0), stats1.hits);
    try std.testing.expectEqual(@as(u64, 1), stats1.misses);
    try std.testing.expectEqual(@as(usize, 1), stats1.entry_count);

    // Second call with same params: should be a cache hit, no disk I/O
    const result2 = try cacheAwareLoad(allocator, &cache_inst, &reader, tensor_name, &expert_ids, layer_idx);
    defer result2.deinit();

    const stats2 = cache_inst.stats();
    try std.testing.expectEqual(@as(u64, 1), stats2.hits);
    try std.testing.expectEqual(@as(u64, 1), stats2.misses); // no new misses
    try std.testing.expectEqual(@as(usize, 1), stats2.entry_count); // same entry

    // Verify both results have correct data
    try result1.eval();
    try result2.eval();
    const data1 = try result1.dataSlice(f32);
    const data2 = try result2.dataSlice(f32);
    try std.testing.expectEqual(data1.len, data2.len);
    for (data1, data2) |v1, v2| {
        try std.testing.expectEqual(v1, v2);
    }

    // Verify shapes are [3, 4, 4] (3 selected experts)
    const shape1 = result1.shape();
    try std.testing.expectEqual(@as(usize, 3), shape1.len);
    try std.testing.expectEqual(@as(i32, 3), shape1[0]);
    try std.testing.expectEqual(@as(i32, 4), shape1[1]);
    try std.testing.expectEqual(@as(i32, 4), shape1[2]);
}

test "cache integration: cache miss path — partial reads used for new experts" {
    // Validates: Requirement 5.2 (cache miss loads via partial read + FdPool)
    const allocator = std.testing.allocator;

    const tmp_path = "/tmp/dmlx_cache_miss_test.safetensors";
    const tensor_name = "layer0.gate_proj.weight";
    const n_experts: u32 = 8;
    const dim1: u32 = 4;
    const dim2: u32 = 4;

    try writeMockSafetensors(allocator, tmp_path, tensor_name, n_experts, dim1, dim2);
    defer removeTempFile(tmp_path, allocator);

    var index = TensorIndex.init(allocator);
    defer index.deinit();
    try index.addShard(tmp_path);

    var fd_pool = FdPool.init(allocator);
    defer fd_pool.deinit();
    try fd_pool.openAll(&index);

    var cache_inst = ExpertCache.init(allocator, 1024 * 1024);
    defer cache_inst.deinit();

    var reader = PartialTensorReader.init(allocator, &index, &fd_pool);

    // Load first set of experts
    const ids_a = [_]u32{ 0, 2 };
    const result_a = try cacheAwareLoad(allocator, &cache_inst, &reader, tensor_name, &ids_a, 0);
    defer result_a.deinit();

    try std.testing.expectEqual(@as(u64, 0), cache_inst.stats().hits);
    try std.testing.expectEqual(@as(u64, 1), cache_inst.stats().misses);

    // Load a DIFFERENT set of experts — should be another cache miss
    const ids_b = [_]u32{ 4, 6, 7 };
    const result_b = try cacheAwareLoad(allocator, &cache_inst, &reader, tensor_name, &ids_b, 0);
    defer result_b.deinit();

    try std.testing.expectEqual(@as(u64, 0), cache_inst.stats().hits);
    try std.testing.expectEqual(@as(u64, 2), cache_inst.stats().misses);
    try std.testing.expectEqual(@as(usize, 2), cache_inst.stats().entry_count);

    // Verify result_a has correct expert data: experts 0 and 2
    try result_a.eval();
    const data_a = try result_a.dataSlice(f32);
    // Expert 0 → val = 0.1, Expert 2 → val = 0.3
    const row_elems: usize = @as(usize, dim1) * @as(usize, dim2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), data_a[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), data_a[row_elems], 1e-6);

    // Verify result_b has correct expert data: experts 4, 6, 7
    try result_b.eval();
    const data_b = try result_b.dataSlice(f32);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), data_b[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), data_b[row_elems], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), data_b[2 * row_elems], 1e-6);
}

test "cache integration: mixed hit/miss — some experts cached, some not" {
    // Validates: Requirements 5.1, 5.2
    // Simulates the real streamingForward pattern where some projections
    // for a layer are already cached (from a previous token step) and
    // others are new.
    const allocator = std.testing.allocator;

    const tmp_path = "/tmp/dmlx_cache_mixed_test.safetensors";
    const gate_name = "layer0.gate_proj.weight";
    const up_name = "layer0.up_proj.weight";
    const n_experts: u32 = 8;
    const dim1: u32 = 4;
    const dim2: u32 = 4;

    try writeMockSafetensorsMulti(allocator, tmp_path, gate_name, up_name, n_experts, dim1, dim2);
    defer removeTempFile(tmp_path, allocator);

    var index = TensorIndex.init(allocator);
    defer index.deinit();
    try index.addShard(tmp_path);

    var fd_pool = FdPool.init(allocator);
    defer fd_pool.deinit();
    try fd_pool.openAll(&index);

    var cache_inst = ExpertCache.init(allocator, 1024 * 1024);
    defer cache_inst.deinit();

    var reader = PartialTensorReader.init(allocator, &index, &fd_pool);

    const expert_ids = [_]u32{ 1, 3, 5 };
    const layer_idx: u32 = 0;

    // --- Token step 1: Load gate_proj and up_proj (both miss) ---
    const gate1 = try cacheAwareLoad(allocator, &cache_inst, &reader, gate_name, &expert_ids, layer_idx);
    defer gate1.deinit();
    const up1 = try cacheAwareLoad(allocator, &cache_inst, &reader, up_name, &expert_ids, layer_idx);
    defer up1.deinit();

    try std.testing.expectEqual(@as(u64, 0), cache_inst.stats().hits);
    try std.testing.expectEqual(@as(u64, 2), cache_inst.stats().misses);
    try std.testing.expectEqual(@as(usize, 2), cache_inst.stats().entry_count);

    // --- Token step 2: Same experts, same layer ---
    // Both gate_proj and up_proj should be cache hits
    const gate2 = try cacheAwareLoad(allocator, &cache_inst, &reader, gate_name, &expert_ids, layer_idx);
    defer gate2.deinit();
    const up2 = try cacheAwareLoad(allocator, &cache_inst, &reader, up_name, &expert_ids, layer_idx);
    defer up2.deinit();

    try std.testing.expectEqual(@as(u64, 2), cache_inst.stats().hits);
    try std.testing.expectEqual(@as(u64, 2), cache_inst.stats().misses); // no new misses

    // --- Token step 3: Different experts for gate_proj, same for up_proj ---
    // gate_proj with new experts → miss; up_proj with same experts → hit
    const new_expert_ids = [_]u32{ 0, 2, 7 };
    const gate3 = try cacheAwareLoad(allocator, &cache_inst, &reader, gate_name, &new_expert_ids, layer_idx);
    defer gate3.deinit();
    const up3 = try cacheAwareLoad(allocator, &cache_inst, &reader, up_name, &expert_ids, layer_idx);
    defer up3.deinit();

    // gate_proj with new ids → 1 new miss; up_proj with same ids → 1 new hit
    try std.testing.expectEqual(@as(u64, 3), cache_inst.stats().hits);
    try std.testing.expectEqual(@as(u64, 3), cache_inst.stats().misses);
    try std.testing.expectEqual(@as(usize, 3), cache_inst.stats().entry_count);

    // Verify gate3 has correct data for experts 0, 2, 7
    try gate3.eval();
    const gate3_data = try gate3.dataSlice(f32);
    const row_elems: usize = @as(usize, dim1) * @as(usize, dim2);
    // gate_proj: expert i → val = (i+1)*0.1
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), gate3_data[0], 1e-6); // expert 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), gate3_data[row_elems], 1e-6); // expert 2
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), gate3_data[2 * row_elems], 1e-6); // expert 7

    // Verify up3 is still correct (from cache) for experts 1, 3, 5
    try up3.eval();
    const up3_data = try up3.dataSlice(f32);
    // up_proj: expert i → val = (i+1)*0.2
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), up3_data[0], 1e-6); // expert 1
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), up3_data[row_elems], 1e-6); // expert 3
    try std.testing.expectApproxEqAbs(@as(f32, 1.2), up3_data[2 * row_elems], 1e-6); // expert 5
}

test "cache integration: cross-layer caching — different layers use separate cache entries" {
    // Validates: Requirement 5.1 (cache keyed by layer index)
    const allocator = std.testing.allocator;

    const tmp_path = "/tmp/dmlx_cache_crosslayer_test.safetensors";
    const tensor_name = "gate_proj.weight";
    const n_experts: u32 = 8;
    const dim1: u32 = 4;
    const dim2: u32 = 4;

    try writeMockSafetensors(allocator, tmp_path, tensor_name, n_experts, dim1, dim2);
    defer removeTempFile(tmp_path, allocator);

    var index = TensorIndex.init(allocator);
    defer index.deinit();
    try index.addShard(tmp_path);

    var fd_pool = FdPool.init(allocator);
    defer fd_pool.deinit();
    try fd_pool.openAll(&index);

    var cache_inst = ExpertCache.init(allocator, 1024 * 1024);
    defer cache_inst.deinit();

    var reader = PartialTensorReader.init(allocator, &index, &fd_pool);

    const expert_ids = [_]u32{ 1, 3 };

    // Load for layer 0
    const r0 = try cacheAwareLoad(allocator, &cache_inst, &reader, tensor_name, &expert_ids, 0);
    defer r0.deinit();

    // Load for layer 1 with same expert_ids — should be a separate cache miss
    // because the layer_idx differs in the cache key
    const r1 = try cacheAwareLoad(allocator, &cache_inst, &reader, tensor_name, &expert_ids, 1);
    defer r1.deinit();

    try std.testing.expectEqual(@as(u64, 0), cache_inst.stats().hits);
    try std.testing.expectEqual(@as(u64, 2), cache_inst.stats().misses);
    try std.testing.expectEqual(@as(usize, 2), cache_inst.stats().entry_count);

    // Now re-request layer 0 — should be a cache hit
    const r0_again = try cacheAwareLoad(allocator, &cache_inst, &reader, tensor_name, &expert_ids, 0);
    defer r0_again.deinit();

    try std.testing.expectEqual(@as(u64, 1), cache_inst.stats().hits);
    try std.testing.expectEqual(@as(u64, 2), cache_inst.stats().misses);
}
