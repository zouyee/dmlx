# Implementation Plan: TTFT Performance Optimization

## Overview

Reduce TTFT from ~137s to ~41-46s for DeepSeek-V4-Flash-4bit through binary index caching (P0), zero-copy mmap loading (P2), sequential I/O (P3), fd reuse (P4), parallel header parsing (P0.5), and instrumentation. Tasks are ordered by dependency: instrumentation first (enables measurement), then P0 (biggest single win), then P2+P4 (low-risk quick wins), then P3 (requires refactoring the load loop), then P0.5 (optional, first-run only).

## Tasks

- [ ] 1. Add TTFT instrumentation (R6)
  - [ ] 1.1 Add timing to `loadWeightsSelective` in `deepseek_v4_loader.zig`
    - Insert `std.time.nanoTimestamp()` before/after: index loading, FdPool.openAll, MmapPool.mmapAll, weight loading loop
    - Log single summary line: `TTFT breakdown: index={d}ms fd={d}ms mmap={d}ms load={d}ms total={d}ms`
    - _Requirements: R6.1, R6.2, R6.4_

  - [ ] 1.2 Add zero-copy hit/miss counter
    - Add `zero_copy_count` and `fallback_copy_count` variables to the weight loading loop
    - After loading, log: `Zero-copy: {d}/{d} tensors ({d}% hit rate)`
    - Counter logic: after `mlx_array_new_data_managed_payload`, compare `mlx_array_data_uint8(arr)` with `slice.ptr` — equal means zero-copy
    - _Requirements: R6.3_

  - [ ] 1.3 Verify instrumentation with baseline run
    - Run `zig build` and execute with DeepSeek-V4-Flash-4bit model
    - Confirm TTFT breakdown log line appears with reasonable values (~67s index, ~65s load)
    - Record baseline numbers for comparison after each optimization
    - _Requirements: R6.1, R6.2_

- [ ] 2. Implement binary index cache — P0 (R1)
  - [ ] 2.1 Implement `serializeIndex` in `safetensors_reader.zig`
    - Write binary header: magic "MLXI", version=1, index_json_size, index_json_mtime, entry_count, string_table_offset
    - Write Entry Offset Table (entry_count × u64)
    - Write variable-length entries: name_off/len, dtype, ndim, shape[ndim], offset_start/end, shard_path_off/len
    - Write string table: concatenated tensor names and shard paths
    - Use `std.fs.File` for writing (not C posix) for simplicity
    - _Requirements: R1.2, R1.5_

  - [ ] 2.2 Implement `deserializeIndex` in `safetensors_reader.zig`
    - Read and validate header: check magic "MLXI", version=1
    - Read Entry Offset Table
    - For each entry: parse fields, resolve string table references, insert into TensorIndex.entries HashMap
    - On corruption (bad magic, truncated file): return error, caller treats as cache miss
    - _Requirements: R1.1, R1.5, R1.6_

  - [ ] 2.3 Implement `isCacheValid` in `safetensors_reader.zig`
    - Stat `model.safetensors.index.json` to get mtime and file_size
    - Stat `model.mlxidx` to check existence
    - Read first 40 bytes of cache file, compare stored mtime and file_size
    - Return false if any check fails (missing file, mismatch, read error)
    - _Requirements: R1.3, R1.4_

  - [ ] 2.4 Implement `loadOrBuildIndex` entry point
    - Call `isCacheValid`; if valid, call `deserializeIndex` and return
    - If invalid: call `buildIndexFromDirectory`, then `serializeIndex`, return index
    - On `serializeIndex` failure: log warning, return index without caching
    - _Requirements: R1.1, R1.2_

  - [ ] 2.5 Wire `loadOrBuildIndex` into `loadWeightsSelective` and `loadWeightsSelectiveLazy`
    - Replace `buildIndexFromDirectory(allocator, dir_path)` calls with `loadOrBuildIndex(allocator, dir_path)`
    - Verify both eager and lazy loading paths use the new entry point
    - _Requirements: R1.1, R1.2_

  - [ ] 2.6 Test cache round-trip correctness
    - Build index from model directory, serialize, deserialize, compare entry count and spot-check 10 random entries (name, dtype, shape, offsets, shard_path)
    - Test cache invalidation: modify index.json mtime, verify cache is rebuilt
    - Test corruption handling: truncate cache file, verify fallback to full build
    - Run `bash scripts/best_test.sh` — 7/7 pass
    - Run `bash scripts/e2e_server.sh` — 13/13 pass (core 7 + extended 6)
    - _Requirements: R1.3, R1.4, R1.6, R1.7_

  - [ ] 2.7 Measure P0 impact
    - Run with fresh cache (first run builds + caches)
    - Run again (second run loads from cache)
    - Compare TTFT breakdown: index phase should drop from ~67s to <2s
    - _Requirements: R1.7_

- [ ] 3. Implement zero-copy mmap loading — P2 (R2)
  - [ ] 3.1 Add `noopDeleter` function in `safetensors_reader.zig`
    - `fn noopDeleter(_: ?*anyopaque) callconv(.c) void {}`
    - _Requirements: R2.1_

  - [ ] 3.2 Replace `mlx_array_new_data` with `mlx_array_new_data_managed_payload` at mmap call sites
    - Site 1: `safetensors_reader.zig` — `TensorIndex.loadTensor()` mmap path (~line 163)
    - Site 2: `deepseek_v4_loader.zig` — `loadWeightsSelective()` mmap path (~line 651)
    - Site 3: `safetensors_reader.zig` — `PartialTensorReader` mmap path (~line 741)
    - Keep pread fallback path unchanged (site at ~line 670)
    - _Requirements: R2.1, R2.6_

  - [ ] 3.3 Verify correctness and measure zero-copy hit rate
    - Run `bash scripts/best_test.sh` — 7/7 pass
    - Run `bash scripts/e2e_server.sh` — 13/13 pass
    - Check zero-copy counter log (from task 1.2): record hit rate
    - If hit rate is 0%, the change is still safe (all fallback to copy) — document finding
    - _Requirements: R2.2, R2.3, R2.7_

- [ ] 4. Implement fd reuse — P4 (R4)
  - [ ] 4.1 Add `addShardKeepFd` variant to `TensorIndex`
    - Same as `addShard` but does not close the fd; returns it to caller
    - _Requirements: R4.1_

  - [ ] 4.2 Add `buildIndexFromDirectoryWithFds` function
    - Same as `buildIndexFromDirectory` but uses `addShardKeepFd`
    - Returns `struct { index: TensorIndex, fds: StringHashMap(c_int) }`
    - _Requirements: R4.1, R4.2_

  - [ ] 4.3 Add `FdPool.initFromFds` to accept pre-opened fds
    - Takes ownership of the fd map; `deinit` closes all fds
    - _Requirements: R4.2_

  - [ ] 4.4 Wire fd reuse into the loading path
    - When building index from scratch (no cache): use `buildIndexFromDirectoryWithFds`, pass fds to `FdPool.initFromFds`
    - When loading from cache (P0): use existing `FdPool.openAll` (no fds to reuse)
    - _Requirements: R4.2, R4.3_

  - [ ] 4.5 Verify correctness
    - Run `bash scripts/best_test.sh` — 7/7 pass
    - Run `bash scripts/e2e_server.sh` — 13/13 pass
    - _Requirements: R4.4_

- [ ] 5. Implement sequential I/O loading — P3 (R3)
  - [ ] 5.1 Refactor `loadWeightsSelective` to group entries by shard
    - Create `EntryWithName` struct: `{ name: []const u8, info: TensorInfo }`
    - Iterate `index.entries`, filter (skip expert/metadata), collect into `StringHashMap(ArrayList(EntryWithName))` keyed by shard_path
    - _Requirements: R3.1_

  - [ ] 5.2 Sort each shard group by `data_offset_start`
    - Use `std.sort.pdq` with comparator on `info.data_offset_start`
    - _Requirements: R3.1_

  - [ ] 5.3 Replace the loading loop with sequential-order iteration
    - Iterate shard groups, then entries within each group (already sorted)
    - Apply same loading logic (mmap/pread, dtype, shape, name mapping, HashMap insert)
    - _Requirements: R3.2_

  - [ ] 5.4 Verify correctness and measure impact
    - Run `bash scripts/best_test.sh` — 7/7 pass
    - Run `bash scripts/e2e_server.sh` — 13/13 pass
    - Compare TTFT breakdown: load phase should decrease
    - Compare page fault count: `/usr/bin/time -l` before vs. after
    - _Requirements: R3.3, R3.4, R3.5_

- [ ] 6. Implement parallel header parsing — P0.5 (R5)
  - [ ] 6.1 Implement `buildIndexFromDirectoryParallel` in `safetensors_reader.zig`
    - Accept `allocator`, `io: std.Io`, `dir_path`, `shard_paths`
    - Allocate `per_shard: []TensorIndex`, init each
    - Use `Io.Group.async` to dispatch `parseShardWorker` for each shard
    - Call `group.await(io)` to block until all complete
    - _Requirements: R5.1, R5.2, R5.6_

  - [ ] 6.2 Implement `mergeIndices` function
    - Iterate each per-shard TensorIndex, move entries into a single merged TensorIndex
    - Deinit per-shard indices after merge
    - _Requirements: R5.3_

  - [ ] 6.3 Implement `parseShardWorker` function
    - Signature: `fn(result: *TensorIndex, allocator: Allocator, shard_path: []const u8) Io.Cancelable!void`
    - Call `result.addShard(shard_path)`, log error on failure
    - _Requirements: R5.2, R5.5_

  - [ ] 6.4 Wire parallel parsing into `loadOrBuildIndex`
    - When cache miss: use `buildIndexFromDirectoryParallel` instead of `buildIndexFromDirectory`
    - Requires threading `io: std.Io` parameter through `loadOrBuildIndex`
    - After parallel build: serialize to cache (P0)
    - _Requirements: R5.7_

  - [ ] 6.5 Verify correctness and measure impact
    - Delete `model.mlxidx` cache, run with parallel parsing
    - Compare first-run TTFT: index phase should drop from ~67s to ~15-20s
    - Verify cache file is created after first run
    - Run `bash scripts/best_test.sh` — 7/7 pass
    - Run `bash scripts/e2e_server.sh` — 13/13 pass
    - _Requirements: R5.4, R5.5_
