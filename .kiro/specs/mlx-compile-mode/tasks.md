# Implementation Plan: MLX Compile Mode

## Overview

Add layer-group MLX graph compilation to dmlx as an opt-in performance mode. The implementation creates a standalone `src/compile_mode.zig` module, integrates it with CLI flag parsing in `main.zig`, and modifies the forward loop in `deepseek_v4.zig` to support compiled execution alongside the existing eager path.

## Tasks

- [x] 1. Create CompileConfig and CompileModule skeleton
  - [x] 1.1 Create `src/compile_mode.zig` with `MixedCompileStrategy` enum, `CompileConfig` struct, and `CompileModule` struct definitions
    - Define `MixedCompileStrategy` enum with `group_only` and `attention_compiled_moe_eager` variants
    - Define `CompileConfig` with fields: `enabled`, `group_size`, `auto_detect`, `strategy`, `per_layer_bytes`, `smelt_overhead_bytes`, `num_layers`
    - Define `CompileModule` with fields: `config`, `cache` ([]?Closure), `failed_groups` ([]bool), `total_compile_time_ns`, `compiled_count`, `allocator`
    - Add Phase 2 doc comments on `attention_compiled_moe_eager` describing future intent
    - _Requirements: 2.1, 2.2, 10.1, 10.2, 10.4, 10.5_

  - [x] 1.2 Implement `CompileModule.init` and `CompileModule.deinit`
    - `init`: validate config (group_size >= 2, clamp group_size > num_layers), allocate cache and failed_groups arrays, call `enableCompile()` when enabled
    - `deinit`: free all cached closures via `Closure.deinit()`, free arrays, call `mlx_detail_compile_clear_cache()`
    - Return unimplemented error when `strategy = .attention_compiled_moe_eager`
    - _Requirements: 2.4, 2.5, 6.3, 6.4, 6.5, 10.3_

  - [x] 1.3 Implement `numGroups` and `groupLayerRange` helper functions
    - `numGroups`: return `(num_layers + group_size - 1) / group_size`
    - `groupLayerRange`: return `{ start: group_idx * group_size, end: min((group_idx + 1) * group_size, num_layers) }`
    - _Requirements: 3.1, 3.4, 3.5_

  - [x]* 1.4 Write property test for layer partition correctness (Property 2)
    - **Property 2: Layer Partition Correctness**
    - Generate random `num_layers` in [1, 200] and `group_size` in [2, num_layers]
    - Verify: exactly `ceil(num_layers / group_size)` groups produced
    - Verify: union of all group ranges equals [0, num_layers)
    - Verify: each group (except last) has exactly `group_size` layers
    - Verify: last group has `num_layers mod group_size` layers (or `group_size` if evenly divisible)
    - 100 iterations with `std.Random`
    - **Validates: Requirements 3.1, 3.4**

- [x] 2. Implement getOrCompileGroup with caching and fallback
  - [x] 2.1 Implement `getOrCompileGroup` core logic
    - Check `failed_groups[group_idx]` — if true, return null (eager fallback)
    - Check `cache[group_idx]` — if non-null, return cached closure (cache hit)
    - On cache miss: call `mlx-zig compile.compile(closure, shapeless=false)`, measure time with `std.time.nanoTimestamp`
    - On success: store in cache, increment `compiled_count`, add to `total_compile_time_ns`
    - On failure: mark `failed_groups[group_idx] = true`, log error with group index and layer range, return null
    - _Requirements: 2.3, 3.2, 6.1, 6.2, 7.1, 7.2, 7.3, 7.5_

  - [x] 2.2 Implement all-groups-failed detection
    - After marking a group as failed, check if all groups have failed
    - If all failed: log summary warning, set `config.enabled = false` to disable compile mode entirely
    - _Requirements: 7.4_

  - [x]* 2.3 Write property test for closure cache idempotence (Property 4)
    - **Property 4: Closure Cache Idempotence**
    - Create a mock/test closure, call `getOrCompileGroup` twice with same group_idx
    - Verify second call returns same closure without re-invoking compile
    - Verify `compiled_count` only increments once per group
    - 100 iterations across random group indices
    - **Validates: Requirements 6.1, 6.2**

  - [x]* 2.4 Write property test for compilation failure fallback (Property 5)
    - **Property 5: Compilation Failure Fallback Without Retry**
    - Simulate compilation failure for a group (inject error)
    - Verify `getOrCompileGroup` returns null on all subsequent calls for that group
    - Verify no retry attempt is made (compile not called again)
    - 100 iterations with random group indices
    - **Validates: Requirements 7.1, 7.5**

- [x] 3. Implement auto-detect group size
  - [x] 3.1 Implement `autoDetectGroupSize` function
    - Query system memory via `mlx-zig` memory API (`memory.getMemoryLimit()` or equivalent)
    - Query active memory via `memory.getActiveMemory()`
    - Calculate: `available = system_memory - active_memory - smelt_overhead - safety_margin` (safety_margin = 2GB)
    - Calculate: `group_size = clamp(floor(available / per_layer_bytes), 2, num_layers)`
    - When available >= per_layer_bytes * num_layers + smelt_overhead: set group_size = num_layers (full graph compile)
    - Log detection result with available memory, per-layer estimate, and chosen group_size
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 3.2 Wire auto-detect into `CompileModule.init`
    - When `config.auto_detect = true`, call `autoDetectGroupSize` before allocating cache arrays
    - Account for Smelt overhead when Smelt is active (`smelt_overhead_bytes` field)
    - _Requirements: 5.4_

  - [x]* 3.3 Write property test for auto-detect formula (Property 3)
    - **Property 3: Auto-Detect Group Size Formula**
    - Generate random `available_memory`, `per_layer_bytes`, `num_layers`, `smelt_overhead`
    - Verify: result equals `clamp(floor((available - smelt_overhead) / per_layer_bytes), 2, num_layers)`
    - Verify: when available >= per_layer_bytes * num_layers + smelt_overhead, result = num_layers
    - Verify: result is always >= 2
    - 100 iterations
    - **Validates: Requirements 4.2, 4.3, 4.4, 5.4**

- [x] 4. Checkpoint - Verify core module
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Add CLI flag parsing
  - [x] 5.1 Add `--compile` and `--compile-group-size` flags to `ChatCommand` and `ServerCommand` in `main.zig`
    - Add `compile: bool = false` and `compile_group_size: ?u32 = null` fields to both command structs
    - Parse `--compile` as boolean flag (no value, decrement i)
    - Parse `--compile-group-size` as u32 value
    - _Requirements: 1.1, 1.3_

  - [x] 5.2 Add validation logic for compile flags
    - If `--compile-group-size` is provided without `--compile`: log error and exit with code 1
    - If `--compile` is provided without `--compile-group-size`: set `auto_detect = true`
    - Validate group_size >= 2 when explicitly provided
    - _Requirements: 1.2, 1.4, 1.5_

  - [x] 5.3 Construct `CompileConfig` from parsed flags and pass to model initialization
    - Build `CompileConfig` from CLI flags
    - Set `smelt_overhead_bytes` based on Smelt configuration when both `--compile` and `--smelt` are active
    - Pass config through to `DSV4Model` initialization path
    - _Requirements: 5.1, 5.5_

  - [x]* 5.4 Write property test for CLI config parsing (Property 1)
    - **Property 1: CLI Config Parsing Consistency**
    - Generate random valid N in [2, 43], verify parsing `--compile --compile-group-size=N` produces `enabled=true, group_size=N`
    - Generate `--compile-group-size=N` without `--compile`, verify config error
    - Generate `--compile` alone, verify `auto_detect=true`
    - 100 iterations
    - **Validates: Requirements 1.1, 1.3, 1.5**

- [x] 6. Implement closure construction and forward loop integration
  - [x] 6.1 Add `compile_module` field to `DSV4Model` struct in `deepseek_v4.zig`
    - Add `compile_module: ?*compile_mode.CompileModule = null` field
    - Initialize from `CompileConfig` during model setup when `config.enabled = true`
    - Free in `DSV4Model.deinit()`
    - _Requirements: 2.1, 6.3_

  - [x] 6.2 Implement `buildGroupClosure` in `deepseek_v4.zig`
    - Define `GroupClosurePayload` struct capturing: layers slice, input_ids, mask, caches slice, start_pos, stream, allocator
    - Implement `groupForwardFn` matching MLX closure signature `fn([]const Array, Allocator) ![]Array`
    - Inside `groupForwardFn`: iterate layers in range, call each layer's forward, eval between layers within group
    - Construct `Closure` from payload using `Closure.initWithPayload` or equivalent pattern from `closure.zig`
    - _Requirements: 3.2, 9.2_

  - [x] 6.3 Modify `DSV4Model.forward` to support compiled path
    - When `self.compile_module` is non-null: iterate over groups using `numGroups()` and `groupLayerRange()`
    - For each group: build closure via `buildGroupClosure`, call `getOrCompileGroup`
    - On compiled closure: call `compiled.apply(&[_]Array{hidden})`, track result
    - On null (eager fallback): execute layers individually with per-layer eval
    - Call `eval()` between groups to allow Smelt paging
    - When `group_size == num_layers`: skip intermediate eval (single group = full graph)
    - Preserve original eager path when `compile_module` is null
    - _Requirements: 3.1, 3.2, 3.3, 3.5, 5.2, 5.3, 9.1, 9.2, 9.3_

- [x] 7. Checkpoint - Verify forward loop integration
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Add logging and observability
  - [x] 8.1 Add startup log message when compile mode is enabled
    - Log format: "Compile mode: enabled, group_size=N, groups=M" at model initialization
    - _Requirements: 8.1_

  - [x] 8.2 Add per-group JIT compilation time logging
    - Log compilation time in milliseconds when a group is compiled for the first time
    - Log auto-detect result with available memory and per-layer estimate
    - Log fallback events with MLX error message and group index/layer range
    - _Requirements: 8.2, 8.3, 8.4, 8.5_

- [x] 9. Write correctness property tests
  - [x]* 9.1 Write property test for compiled-eager numerical equivalence (Property 6)
    - **Property 6: Compiled-Eager Numerical Equivalence**
    - Create a small test model or mock layer group
    - Run same input through compiled closure and eager sequential execution
    - Verify bitwise-equal output tensors
    - 20 iterations (GPU-bound)
    - **Validates: Requirements 9.1, 9.4**

  - [x]* 9.2 Write property test for KV cache state equivalence (Property 7)
    - **Property 7: KV Cache State Equivalence**
    - Run same input through compiled and eager paths with KV caches
    - Verify identical KV cache state (keys, values, sequence positions) after execution
    - 20 iterations (GPU-bound)
    - **Validates: Requirements 9.3**

- [x] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- The implementation language is Zig, matching the existing codebase and design document
- `src/compile_mode.zig` is standalone with no model-specific dependencies — it operates on generic closures from `mlx-zig`
- The forward loop modification preserves the existing eager path as the default when `--compile` is not specified
