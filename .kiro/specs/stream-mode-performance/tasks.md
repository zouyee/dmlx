# Implementation Plan: Stream Mode Performance Optimizations

## Overview

Implement four interlocking performance optimizations for stream mode expert loading in mlx-zig: FdPool (file descriptor pooling), ExpertCache (LRU cache), PartialTensorReader (per-expert pread), and their integration into `streamingForward`. An optional LayerPrefetcher and diagnostic logging round out the plan. The implementation order follows dependency chains: FdPool first (no deps), then ExpertCache (no deps), then PartialTensorReader (depends on FdPool), then integration, then prefetching and diagnostics.

## Tasks

- [x] 1. Implement FdPool for file descriptor pooling
  - [x] 1.1 Create `FdPool` struct in `src/io/safetensors_reader.zig`
    - Add `FdPool` struct with `fds: std.StringHashMap(std.posix.fd_t)` and allocator
    - Implement `init(allocator)` to create an empty pool
    - Implement `deinit()` to close all open file descriptors and free path keys
    - Implement `openAll(index: *TensorIndex)` that iterates unique shard paths from `index.entries` and opens each with `O_RDONLY`, storing the fd keyed by shard path
    - Implement `getFd(shard_path)` that returns the pre-opened fd or `error.ShardNotInPool`
    - Return `error.FileNotFound` from `openAll` if any shard file cannot be opened, including the inaccessible path
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 1.2 Write property test for FdPool (Property 7)
    - **Property 7: FdPool provides valid fds for all indexed shards**
    - Create temporary safetensors files with valid headers, build a TensorIndex, call `openAll()`, verify `getFd()` returns valid (non-negative) fds for all shard paths, and verify idempotency (same fd on repeated calls)
    - Run 100 iterations with varying numbers of shard files
    - **Validates: Requirements 3.1, 3.2**

  - [ ]* 1.3 Write unit tests for FdPool error handling
    - Test `openAll` returns `error.FileNotFound` for missing shard files
    - Test `getFd` returns `error.ShardNotInPool` for unknown paths
    - Test `deinit` properly closes all fds (verify with `fstat` after close)
    - _Requirements: 3.3, 3.4_

- [x] 2. Implement ExpertCache with LRU eviction
  - [x] 2.1 Create `src/models/expert_cache.zig` with ExpertCache struct
    - Define `CacheKey` struct: `{ layer_idx: u32, tensor_name_hash: u64, expert_id: u32 }`
    - Define `CacheEntry` struct with key, tensor (Array), byte_size, and LRU prev/next pointers
    - Define `CacheStats` struct: `{ hits, misses, current_bytes, max_bytes, entry_count }`
    - Implement `init(allocator, max_bytes)` with default 4GB budget
    - Implement `deinit()` to free all entries and their tensors
    - Implement the `AutoHashMap(CacheKey, *CacheEntry)` for O(1) lookup
    - Implement doubly-linked LRU list with `lru_head` (MRU) and `lru_tail` (LRU) pointers
    - _Requirements: 1.1, 1.4, 1.5_

  - [x] 2.2 Implement `get()` and `put()` methods on ExpertCache
    - `get(key)`: look up in map, return null on miss (increment `misses`), on hit move entry to LRU head (increment `hits`), return tensor
    - `put(key, tensor, byte_size)`: if `byte_size > max_bytes`, skip caching (Req 1.6); otherwise evict LRU entries until `current_bytes + byte_size <= max_bytes`; allocate entry, insert into map and LRU head; update `current_bytes`
    - Implement `evictUntil(needed_bytes)`: remove entries from LRU tail, deinit their tensors, update `current_bytes` and map; log evicted tensor info at debug level
    - Implement `stats()` returning `CacheStats`
    - _Requirements: 1.2, 1.3, 1.4, 1.6, 1.7_

  - [ ]* 2.3 Write property test for cache round-trip (Property 1)
    - **Property 1: Cache round-trip**
    - Generate random CacheKeys and mock tensors with random byte sizes, insert via `put()`, retrieve via `get()`, verify the returned tensor matches the original
    - Vary cache budget, number of entries, and access patterns across 100 iterations
    - **Validates: Requirements 1.1, 1.2**

  - [ ]* 2.4 Write property test for LRU eviction order (Property 2)
    - **Property 2: LRU eviction order**
    - Generate random sequences of `put()`/`get()` on a small-budget cache, track access timestamps manually, after each eviction-triggering `put()` verify the evicted entry was the LRU one
    - Run 100 iterations with varying operation sequences
    - **Validates: Requirements 1.3**

  - [ ]* 2.5 Write property test for cache memory tracking (Property 3)
    - **Property 3: Cache memory tracking invariant**
    - Generate random operation sequences, after each operation verify `current_bytes` equals sum of all entry sizes and `current_bytes <= max_bytes`
    - Run 100 iterations
    - **Validates: Requirements 1.4**

  - [ ]* 2.6 Write property test for hit/miss counter accuracy (Property 4)
    - **Property 4: Cache hit/miss counter accuracy**
    - Generate random `put()`/`get()` sequences, after each `get()` verify `hits + misses` equals total `get()` count
    - Run 100 iterations
    - **Validates: Requirements 1.7**

- [x] 3. Checkpoint — Ensure FdPool and ExpertCache tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement PartialTensorReader for per-expert sliced loading
  - [x] 4.1 Create `PartialTensorReader` struct in `src/io/safetensors_reader.zig`
    - Add struct with `allocator`, `index: *TensorIndex`, `fd_pool: *FdPool`
    - Implement `init(allocator, index, fd_pool)`
    - Implement `computeExpertByteRange(info, expert_id)` returning `{ offset: u64, length: usize }`:
      - Compute `row_elements = D1 * D2 * ...` from tensor shape (all dims except axis 0)
      - Compute `row_bytes = row_elements * elem_bytes` (for mxfp4 uint8 packed: `row_elements` bytes; for bfloat16 scales: `row_elements * 2` bytes)
      - Return `offset = info.data_offset_start + expert_id * row_bytes`, `length = row_bytes`
      - Ensure mxfp4 group alignment: offset and length must be multiples of `group_size / 2` bytes
    - _Requirements: 2.1, 2.3_

  - [x] 4.2 Implement `readExpertRow()` and `readExpertRows()` on PartialTensorReader
    - `readExpertRow(tensor_name, expert_id)`: look up tensor info in index, compute byte range, get fd from FdPool, pread the byte range, create MLX Array with shape `[1, D1, D2, ...]` via `mlx_array_new_data`
    - `readExpertRows(tensor_name, expert_ids)`: call `readExpertRow` for each expert_id, concatenate into a mini-fused tensor with shape `[n_experts, D1, D2, ...]`
    - Return `error.IncompleteRead` if pread returns fewer bytes than expected
    - Return `error.TensorNotFound` if tensor name not in index
    - Return `error.ExpertIdOutOfRange` if expert_id >= n_experts
    - Apply same partial read strategy for scale tensors
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 4.3 Write property test for byte offset computation (Property 5)
    - **Property 5: Expert byte offset computation with mxfp4 alignment**
    - Generate random tensor shapes `[n_experts, D1, D2]` with random dtypes (uint8 for mxfp4, bfloat16 for scales), for random expert IDs verify offset = `data_offset_start + expert_id * row_bytes` and verify mxfp4 alignment
    - Run 100 iterations
    - **Validates: Requirements 2.1, 2.3**

  - [ ]* 4.4 Write property test for mini-fused tensor assembly (Property 6)
    - **Property 6: Mini-fused tensor assembly correctness**
    - Create test fused tensors with known data (expert `i` has all values = `i`), select random subsets of expert IDs, assemble mini-fused tensor, verify each row matches the corresponding expert's data
    - Run 100 iterations
    - **Validates: Requirements 2.4**

  - [ ]* 4.5 Write unit tests for PartialTensorReader error handling
    - Test `error.IncompleteRead` when pread returns short
    - Test `error.TensorNotFound` for missing tensor names
    - Test `error.ExpertIdOutOfRange` for invalid expert IDs
    - _Requirements: 2.6_

- [x] 5. Checkpoint — Ensure PartialTensorReader tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Integrate optimizations into streamingForward
  - [x] 6.1 Extend `ExpertStreamProvider` with new performance fields
    - Add `cache: ?*ExpertCache`, `fd_pool: ?*FdPool`, `partial_reader: ?*PartialTensorReader` fields to `ExpertStreamProvider` in `src/models/expert_stream.zig`
    - Add diagnostic counters: `total_bytes_read: u64`, `token_step_count: u64`, `token_step_start_ns: i128`
    - Update `initWithStrategy` to initialize FdPool (call `openAll`), ExpertCache (with configurable budget), and PartialTensorReader when strategy is `.stream`
    - Update `deinit` to clean up all new components
    - _Requirements: 5.1, 5.2, 3.1_

  - [x] 6.2 Rewrite `streamingForward` to use cache-first loading with partial reads
    - For each required projection (gate_proj, up_proj, down_proj, and their scales):
      1. Check ExpertCache for each expert_id using `cache.get(CacheKey{layer_idx, hash(tensor_name), expert_id})`
      2. Collect cache misses into a list of expert_ids to load
      3. For cache misses, use `partial_reader.readExpertRows(tensor_name, missing_ids)` instead of `loadExpertSlices`
      4. Insert newly loaded expert rows into ExpertCache via `cache.put()`
      5. Assemble the mini-fused tensor from cached + newly loaded rows
    - Preserve the existing remap, gatherQmm/gatherMm, SwiGLU, and weighted-sum logic unchanged
    - Fall back to full tensor loading via `loadExpertSlices` if partial_reader or cache is null
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 6.3 Write property test for numerical equivalence (Property 8)
    - **Property 8: Numerical equivalence with unoptimized path**
    - Create small mock expert tensors on disk, run both the old `loadExpertSlices` (full load + slice) and the new partial-read + cache path with the same inputs, verify outputs are bitwise identical
    - Run 100 iterations with random expert selections
    - **Validates: Requirements 5.3**

  - [x] 6.4 Write integration tests for cache-aware streamingForward
    - Test cache hit path: load experts, call streamingForward again with same layer, verify no disk I/O on second call
    - Test cache miss path: call streamingForward with new experts, verify partial reads are used
    - Test mixed hit/miss: some experts cached, some not
    - _Requirements: 5.1, 5.2_

- [x] 7. Checkpoint — Ensure integration tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement LayerPrefetcher for async I/O
  - [x] 8.1 Create `src/models/layer_prefetcher.zig` with LayerPrefetcher struct
    - Define struct with `allocator`, `reader: *PartialTensorReader`, `cache: *ExpertCache`, `layer_meta: []const LayerExpertMeta`
    - Add thread synchronization: `thread: ?std.Thread`, `mutex: std.Thread.Mutex`, `condition: std.Thread.Condition`
    - Add request state: `request_layer: ?usize`, `request_expert_ids: ?[]const u32`, `request_done: bool`, `should_stop: bool`
    - Implement `init()` that spawns the background prefetch worker thread
    - Implement `deinit()` that signals `should_stop`, wakes the thread, and joins it
    - _Requirements: 4.1, 4.4_

  - [x] 8.2 Implement `prefetch()` and `waitForCompletion()` on LayerPrefetcher
    - `prefetch(layer_idx, expert_ids)`: acquire mutex, set request_layer and request_expert_ids, signal condition variable, release mutex — non-blocking
    - `waitForCompletion()`: acquire mutex, wait on condition until `request_done` is true, release mutex
    - Implement `prefetchWorker()` background thread: loop waiting on condition, on wake load all 6 projections for requested experts via `reader.readExpertRows()`, insert into cache, set `request_done`, signal condition
    - Limit prefetch memory to one layer's worth (~36MB for 6 experts × 6 projections)
    - Handle prefetch read failures gracefully: log warning, skip insertion, main thread will load synchronously
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 8.3 Wire LayerPrefetcher into streamingForward
    - Add `prefetcher: ?*LayerPrefetcher` field to `ExpertStreamProvider`
    - At the end of `streamingForward` for layer N, call `prefetcher.prefetch(layer_idx + 1, expert_ids)` to kick off background loading for layer N+1
    - At the start of `streamingForward` for layer N+1, call `prefetcher.waitForCompletion()` — prefetched experts will already be in the cache
    - If actual expert selections differ from prefetched, cache still holds overlapping experts; missing ones are loaded synchronously
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ]* 8.4 Write unit tests for LayerPrefetcher
    - Test prefetch with matching expert predictions (all prefetched experts used)
    - Test prefetch with mismatching predictions (some prefetched, some loaded sync)
    - Test graceful degradation when prefetch thread fails to spawn
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 9. Implement diagnostic logging and performance metrics
  - [x] 9.1 Add `TokenStepMetrics` struct and logging to ExpertStreamProvider
    - Define `TokenStepMetrics` struct: `{ step_number, wall_clock_ms, bytes_read, cache_hits, cache_misses, cache_memory_bytes, prefetch_hits, prefetch_misses }`
    - At the start of each token generation step, record `token_step_start_ns` via `std.time.nanoTimestamp()` and log current cache memory usage
    - At the end of each token step, compute wall-clock time, collect cache stats, log total bytes read, hit/miss counts, and wall-clock time
    - Log evicted tensor info (layer index, tensor name) at debug level from `ExpertCache.evictUntil()`
    - _Requirements: 6.1, 6.2, 6.3_

  - [ ]* 9.2 Write unit tests for diagnostic metrics
    - Test that `TokenStepMetrics` fields are populated correctly after a mock token step
    - Test that cache eviction logs include layer index and tensor name
    - Test that cache memory usage is logged at step start
    - _Requirements: 6.1, 6.2, 6.3_

- [x] 10. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The implementation uses Zig throughout, matching the existing codebase
- FdPool and ExpertCache have no dependencies and can be implemented in parallel
- PartialTensorReader depends on FdPool; integration depends on all three core components
- LayerPrefetcher (task 8) can be deferred — the system works correctly without it, just without I/O overlap
- The system degrades gracefully: if any optimization component is null/disabled, streamingForward falls back to the current full-tensor loading path
