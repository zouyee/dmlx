# Requirements: TTFT Performance Optimization

## Introduction

TTFT (Time To First Token) for DeepSeek-V4-Flash-4bit on mlx-zig is ~137 seconds on a MacBook Pro 48GB RAM. This is split roughly 50/50 between index building (~67s, serial parsing of 33 safetensors shard headers) and weight loading (~65-68s, mmap + memcpy of 2223 backbone tensors into MLX arrays). This feature targets reducing TTFT to under 50 seconds through binary index caching, zero-copy mmap loading, sequential I/O optimization, and parallel header parsing.

Reference document: `docs/TTFT-OPTIMIZATION.md` (v3)

## Glossary

- **TTFT**: Time To First Token — the wall-clock time from program start to the first generated token, dominated by model loading.
- **TensorIndex**: A HashMap mapping tensor names to `TensorInfo` (dtype, shape, file offset, shard path), built by parsing safetensors headers. Defined in `src/io/safetensors_reader.zig`.
- **buildIndexFromDirectory**: The function that reads `model.safetensors.index.json`, collects unique shard files, and calls `addShard` on each to parse its JSON header. Currently serial, ~67s for 33 shards.
- **addShard**: Parses one safetensors file header (open → pread 8-byte length → pread JSON → parse → close) and inserts tensor entries into the TensorIndex HashMap.
- **FdPool**: Pool of pre-opened file descriptors for shard files, avoiding repeated open/close during weight loading. Defined in `src/io/safetensors_reader.zig`.
- **MmapPool**: Pool of memory-mapped shard files for zero-syscall reads. Each shard is mmap'd once; provides `getSlice(shard, offset, len)` for direct pointer access. Defined in `src/io/safetensors_reader.zig`.
- **loadWeightsSelective**: The function in `src/models/deepseek_v4_loader.zig` that iterates TensorIndex entries, filters expert/metadata weights, and loads backbone tensors via mmap or pread into MLX arrays.
- **mlx_array_new_data**: MLX C API function that creates an array by copying data from an external buffer (always memcpy).
- **mlx_array_new_data_managed_payload**: MLX C API function that creates an array from an external buffer with a custom deleter. Attempts zero-copy via `MTL::Device::newBuffer`; falls back to memcpy if pointer is not page-aligned.
- **model.mlxidx**: The proposed binary cache file storing a serialized TensorIndex, enabling sub-second index loading on subsequent runs.
- **Io.Group**: Zig 0.16 `std.Io.Group` — an unordered task group that dispatches work to a thread pool (Threaded backend) or GCD (macOS Dispatch backend). Supports `group.async` + `group.await`.

## Requirements

### Requirement 1: Binary Index Cache (P0)

**User Story:** As a user starting mlx-zig inference, I want the safetensors index to be cached as a binary file after the first run, so that subsequent startups skip the 67-second header parsing phase and load the index in under 2 seconds.

#### Acceptance Criteria

1. WHEN `loadOrBuildIndex` is called and a valid `model.mlxidx` cache file exists in the model directory, THEN the function SHALL deserialize the TensorIndex from the binary cache without parsing any safetensors shard headers.
2. WHEN `loadOrBuildIndex` is called and no valid cache file exists, THEN the function SHALL call `buildIndexFromDirectory` to parse all shard headers, then serialize the resulting TensorIndex to `model.mlxidx`.
3. THE cache validity check SHALL compare both the `mtime` and `file_size` of `model.safetensors.index.json` against the values stored in the cache header.
4. WHEN the `mtime` or `file_size` of `model.safetensors.index.json` differs from the cached values, THEN the cache SHALL be considered invalid and rebuilt from scratch.
5. THE binary cache format SHALL use an Entry Offset Table to support O(1) random access to variable-length entries (shape arrays have variable ndim).
6. THE deserialized TensorIndex SHALL contain identical entries (name, dtype, shape, offsets, shard_path) to a freshly built TensorIndex from the same model directory.
7. WHEN loading from a valid cache, THE index loading time SHALL be under 2 seconds for a 2481-entry index on NVMe storage.

### Requirement 2: Zero-Copy mmap Loading (P2)

**User Story:** As a user loading model weights, I want backbone tensors to be created directly from mmap'd memory without an intermediate memcpy, so that weight loading time is reduced by eliminating redundant data copies.

#### Acceptance Criteria

1. WHEN loading a backbone tensor from an mmap'd shard region, THE loader SHALL use `mlx_array_new_data_managed_payload` with a noop deleter instead of `mlx_array_new_data`.
2. WHEN `mlx_array_new_data_managed_payload` is called and the Metal backend's `make_buffer` succeeds (pointer page-aligned, size page-aligned), THEN the MLX array SHALL reference the mmap'd memory directly without copying.
3. WHEN `mlx_array_new_data_managed_payload` is called and `make_buffer` fails (pointer not page-aligned), THEN MLX SHALL automatically fall back to malloc + memcpy, producing a correct array.
4. THE noop deleter SHALL not munmap or free any memory, because MmapPool manages the mmap lifecycle.
5. THE MmapPool SHALL NOT be deinitialized before all MLX arrays referencing its mmap'd regions are freed.
6. THE pread fallback path (non-mmap) SHALL continue using `mlx_array_new_data` (always copies), since its buffer is stack/heap-allocated and temporary.
7. AFTER the change, all 7 prompts in `scripts/best_test.sh` SHALL pass with identical output, and all 13 tests in `scripts/e2e_server.sh` SHALL pass.

### Requirement 3: Sequential I/O Weight Loading (P3)

**User Story:** As a user loading model weights, I want tensors to be loaded in sequential file offset order within each shard, so that OS readahead is effective and page fault overhead is minimized.

#### Acceptance Criteria

1. WHEN `loadWeightsSelective` loads backbone tensors, IT SHALL group entries by shard_path and sort each group by `data_offset_start` in ascending order before loading.
2. WHEN loading tensors within a shard group, THE loader SHALL iterate in offset-sorted order so that mmap page accesses are sequential.
3. THE sequential loading SHALL produce the same `StringHashMap(Array)` result as the current random-order loading (same keys, same tensor data).
4. AFTER the change, THE weight loading phase SHALL show reduced page fault count compared to the baseline (measurable via `/usr/bin/time -l`).
5. AFTER the change, all 7 prompts in `scripts/best_test.sh` SHALL pass with identical output, and all 13 tests in `scripts/e2e_server.sh` SHALL pass.

### Requirement 4: fd Reuse (P4)

**User Story:** As a developer maintaining the codebase, I want file descriptors opened during index building to be reused by FdPool, so that the code avoids redundant open/close cycles and is simpler to reason about.

#### Acceptance Criteria

1. WHEN `buildIndexFromDirectory` parses shard headers, IT SHALL keep file descriptors open instead of closing them after each `addShard`.
2. THE returned fd map SHALL be passed to `FdPool` to avoid re-opening the same 33 shard files.
3. WHEN the index is loaded from binary cache (P0), THE fd reuse path SHALL be skipped and `FdPool.openAll` SHALL open fds normally.
4. THE change SHALL not alter any observable behavior or output.

### Requirement 5: Parallel Header Parsing (P0.5)

**User Story:** As a user running mlx-zig for the first time (no binary cache), I want shard header parsing to run in parallel using Zig 0.16's `std.Io.Group`, so that the first-run index build time is reduced from 67 seconds to under 20 seconds.

#### Acceptance Criteria

1. WHEN `buildIndexFromDirectory` is called without a valid binary cache, IT SHALL parse shard headers in parallel using `Io.Group.async`.
2. EACH shard SHALL be parsed into an independent `TensorIndex` instance to avoid shared-state race conditions.
3. AFTER all parallel tasks complete (`group.await`), THE per-shard indices SHALL be merged into a single TensorIndex in a single-threaded merge step.
4. THE merged TensorIndex SHALL contain identical entries to a serially-built TensorIndex from the same model directory.
5. WHEN the number of shards exceeds `async_limit` (CPU cores - 1), THE excess tasks SHALL execute synchronously on the calling thread (Zig 0.16 `groupAsyncEager` fallback), not hang or deadlock.
6. THE parallel parsing SHALL accept an `std.Io` parameter, consistent with the project's existing `std.Io` usage in `server.zig`.
7. WHEN parallel parsing completes, THE result SHALL be serialized to `model.mlxidx` (P0 cache) so subsequent runs benefit from the cache.

### Requirement 6: TTFT Instrumentation

**User Story:** As a developer optimizing TTFT, I want each loading phase to be individually timed and logged, so that I can measure the actual impact of each optimization and identify remaining bottlenecks.

#### Acceptance Criteria

1. THE loader SHALL log wall-clock time in milliseconds for each phase: index loading, FdPool open, MmapPool mmap, weight loading, and total TTFT.
2. THE log format SHALL be a single line: `TTFT breakdown: index={d}ms fd={d}ms mmap={d}ms load={d}ms total={d}ms`.
3. WHEN P2 (zero-copy) is active, THE loader SHALL count and log the number of tensors that achieved zero-copy vs. those that fell back to memcpy.
4. THE instrumentation SHALL use `std.time.nanoTimestamp()` for sub-millisecond precision.

## Non-Goals

- Modifying the mlx-c or mlx C++ library code.
- Optimizing the prefill or token generation phases (already optimized in prior work).
- Supporting models other than DeepSeek-V4-Flash-4bit (though the optimizations are model-agnostic).
- Implementing P5 (index-load pipeline) — deferred as low-ROI given P0 cache.
