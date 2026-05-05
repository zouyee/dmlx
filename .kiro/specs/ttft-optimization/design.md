# Design: TTFT Performance Optimization

## Overview

This design addresses the 137-second TTFT for DeepSeek-V4-Flash-4bit by optimizing two phases: index building (67s → 1s via binary cache) and weight loading (65s → 40s via sequential I/O + zero-copy). The optimizations are layered and independent — each can be implemented and verified separately.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    loadOrBuildIndex (P0)                 │
│  ┌──────────────┐    ┌──────────────────────────────┐   │
│  │ model.mlxidx │───▶│ deserializeIndex (mmap, ~1s) │   │
│  │ (binary cache)│    └──────────────────────────────┘   │
│  └──────────────┘                                       │
│        miss ↓                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │ buildIndexFromDirectoryParallel (P0.5)           │   │
│  │  Io.Group.async × 33 shards → mergeIndices      │   │
│  │  → serializeIndex → model.mlxidx                 │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼ TensorIndex
┌─────────────────────────────────────────────────────────┐
│              loadWeightsSelective (P3 + P2)              │
│  1. Group entries by shard_path                         │
│  2. Sort each group by data_offset_start                │
│  3. For each entry (sequential order):                  │
│     mmap_pool.getSlice → mlx_array_new_data_managed_    │
│     payload (P2, zero-copy attempt)                     │
│     → fallback: mlx_array_new_data (memcpy)             │
└─────────────────────────────────────────────────────────┘
```

## Component Design

### 1. Binary Index Cache (`model.mlxidx`)

**File**: `src/io/safetensors_reader.zig`

**Binary format**:

```
Offset  Size     Field
──────────────────────────────────────────
0       4        magic: "MLXI"
4       4        version: u32 (= 1)
8       8        index_json_size: u64
16      8        index_json_mtime: i64
24      4        entry_count: u32
28      4        padding: u32 (= 0)
32      8        string_table_offset: u64
──────────────────────────────────────────
40      N×8      Entry Offset Table: entry_offsets[entry_count]
──────────────────────────────────────────
varies  varies   Entry[0..entry_count] (variable-length):
                   name_off: u32      (offset into string table)
                   name_len: u16
                   dtype: u8          (maps to mlx_dtype enum)
                   ndim: u8
                   shape[ndim]: i64[] (ndim × 8 bytes)
                   offset_start: u64
                   offset_end: u64
                   shard_path_off: u32
                   shard_path_len: u16
                   padding: u16       (align to 8 bytes)
──────────────────────────────────────────
varies  varies   String Table: concatenated name and path strings
```

**New functions**:

```zig
/// Serialize a TensorIndex to binary cache file.
pub fn serializeIndex(index: *TensorIndex, cache_path: []const u8) !void;

/// Deserialize a TensorIndex from binary cache file.
pub fn deserializeIndex(allocator: Allocator, cache_path: []const u8) !TensorIndex;

/// Check if cache is valid by comparing mtime + file_size of index.json.
pub fn isCacheValid(cache_path: []const u8, dir_path: []const u8) bool;

/// Top-level entry point: load from cache or build + cache.
pub fn loadOrBuildIndex(allocator: Allocator, dir_path: []const u8) !TensorIndex;
```

**Cache invalidation**: Compare `stat.st_mtime` and `stat.st_size` of `model.safetensors.index.json` against values stored in the cache header. If either differs, rebuild.

### 2. Zero-Copy mmap Loading

**Files**: `src/io/safetensors_reader.zig`, `src/models/deepseek_v4_loader.zig`

**Change**: Replace `mlx_array_new_data` with `mlx_array_new_data_managed_payload` at 3 mmap call sites.

```zig
// Noop deleter — MmapPool owns the mmap lifecycle
fn noopDeleter(_: ?*anyopaque) callconv(.c) void {}

// Zero-copy path (mmap):
const arr = c.c.mlx_array_new_data_managed_payload(
    @constCast(@ptrCast(slice.ptr)),
    shape_i32.ptr,
    @intCast(shape_i32.len),
    mlx_dtype,
    null,
    noopDeleter,
);

// Copy path (pread fallback) — unchanged:
const arr = c.c.mlx_array_new_data(buf.ptr, shape_i32.ptr, ...);
```

**Call sites to modify**:
1. `safetensors_reader.zig:163` — `TensorIndex.loadTensor()` mmap path
2. `deepseek_v4_loader.zig:651` — `loadWeightsSelective()` mmap path
3. `safetensors_reader.zig:741` — `PartialTensorReader` mmap path

**Pointer alignment caveat**: `mmap_pool.getSlice()` returns `base_ptr + data_offset_start`. The base is page-aligned (mmap guarantee), but `data_offset_start` depends on tensor packing in the safetensors file and is usually NOT page-aligned (16KB on Apple Silicon). Metal's `newBuffer(ptr, ...)` requires page-aligned ptr, so most tensors will fall back to memcpy. The change is still safe (automatic fallback) and low-risk.

**Lifecycle safety**: MmapPool.deinit() is called after all model arrays are freed (in `ModelState.deinit()` or `ExpertStreamProvider.deinit()`). The noop deleter ensures no premature munmap.

### 3. Sequential I/O Loading

**File**: `src/models/deepseek_v4_loader.zig`

**Change**: Refactor `loadWeightsSelective` to group-and-sort before loading.

```zig
pub fn loadWeightsSelective(...) !StringHashMap(Array) {
    // Phase 1: Collect and filter entries
    var by_shard = StringHashMap(ArrayList(EntryWithName)).init(allocator);
    var idx_it = index.entries.iterator();
    while (idx_it.next()) |entry| {
        if (shouldSkip(entry)) continue;
        const list = try by_shard.getOrPut(entry.value_ptr.shard_path);
        if (!list.found_existing) list.value_ptr.* = ArrayList(EntryWithName).init(allocator);
        try list.value_ptr.append(.{ .name = entry.key_ptr.*, .info = entry.value_ptr.* });
    }

    // Phase 2: Sort each shard group by offset
    var shard_it = by_shard.iterator();
    while (shard_it.next()) |shard_entry| {
        std.sort.pdq(EntryWithName, shard_entry.value_ptr.items, {}, offsetLessThan);
    }

    // Phase 3: Load in sequential order
    shard_it = by_shard.iterator();
    while (shard_it.next()) |shard_entry| {
        for (shard_entry.value_ptr.items) |item| {
            // ... load tensor (same logic as current, but in offset order)
        }
    }
}
```

**Why this helps**: Current HashMap iteration visits entries in hash-bucket order, causing random mmap page faults across the ~4.3GB shard file. Sequential access lets `madvise(MADV_SEQUENTIAL)` (already set by MmapPool) trigger effective OS readahead.

### 4. fd Reuse

**File**: `src/io/safetensors_reader.zig`

**Change**: `addShard` gains a `keep_fd` variant that returns the fd instead of closing it. `buildIndexFromDirectory` collects fds and returns them alongside the index.

```zig
pub fn addShardKeepFd(self: *TensorIndex, shard_path: []const u8) !c_int {
    // ... same as addShard but skip close(fd) ...
    return fd;
}

pub fn buildIndexFromDirectoryWithFds(allocator: Allocator, dir_path: []const u8) !struct {
    index: TensorIndex,
    fds: StringHashMap(c_int),
} { ... }
```

`FdPool` gains `initFromFds(fds: StringHashMap(c_int))` to accept pre-opened fds.

### 5. Parallel Header Parsing

**File**: `src/io/safetensors_reader.zig`

**Change**: New function `buildIndexFromDirectoryParallel` using `Io.Group`.

```zig
pub fn buildIndexFromDirectoryParallel(
    allocator: Allocator,
    io: std.Io,
    dir_path: []const u8,
    shard_paths: []const []const u8,
) !TensorIndex {
    var per_shard = try allocator.alloc(TensorIndex, shard_paths.len);
    defer allocator.free(per_shard);
    for (per_shard) |*idx| idx.* = TensorIndex.init(allocator);

    var group: std.Io.Group = .init;
    for (shard_paths, 0..) |path, i| {
        group.async(io, parseShardWorker, .{ &per_shard[i], allocator, path });
    }
    try group.await(io);

    return mergeIndices(allocator, per_shard);
}
```

**Thread safety**: Each shard writes to its own `TensorIndex` — no shared state. `mergeIndices` runs single-threaded after `group.await`.

**Concurrency limit**: `Io.Group.async` on the Threaded backend falls back to synchronous execution when `busy_count >= async_limit` (default: CPU cores - 1). On a 10-core Mac, 9 shards run in parallel, 24 run serially on the calling thread. No deadlock risk (confirmed via source analysis of `Threaded.zig:2197`).

### 6. TTFT Instrumentation

**File**: `src/models/deepseek_v4_loader.zig`

**Change**: Add `std.time.nanoTimestamp()` timing around each phase in `loadWeightsSelective` and `loadOrBuildIndex`. Log a single summary line.

For P2 zero-copy tracking: maintain a counter of zero-copy hits vs. fallbacks during the loading loop.

## Module-Document Mapping

| Source File | Design Section | Layer |
|---|---|---|
| `src/io/safetensors_reader.zig` | §1 Binary Cache, §4 fd Reuse, §5 Parallel Parsing | L1/L2 |
| `src/models/deepseek_v4_loader.zig` | §2 Zero-Copy, §3 Sequential I/O, §6 Instrumentation | L1/L2 |

## Correctness Properties

1. **Cache equivalence**: `deserializeIndex(serializeIndex(idx))` produces a TensorIndex with identical entries to `idx`.
2. **Load equivalence**: Sequential-order loading produces the same `StringHashMap(Array)` as random-order loading (same keys, same tensor data byte-for-byte).
3. **Zero-copy safety**: No use-after-munmap — MmapPool outlives all MLX arrays referencing its regions.
4. **Parallel merge equivalence**: `mergeIndices(parallel_results)` produces the same TensorIndex as serial `buildIndexFromDirectory`.

## Error Handling

- Cache file corruption (bad magic, truncated): treat as cache miss, rebuild from scratch, log warning.
- Cache write failure (disk full, permissions): log warning, continue without cache. Do not fail the load.
- Parallel shard parse failure: `parseShardWorker` logs the error and leaves its TensorIndex empty. `mergeIndices` skips empty indices. If all shards fail, the merged index is empty and downstream loading will fail with appropriate errors.
