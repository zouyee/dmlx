# Requirements Document

## Introduction

This feature adds layer-group MLX graph compilation to dmlx as an opt-in performance optimization. Rather than compiling the entire forward pass as a single graph (which would require all layer weights in memory simultaneously), this approach compiles groups of N consecutive transformer layers into fused closures, with `eval()` calls between groups to allow memory paging.

Key insight: layer-group compilation is fully compatible with Smelt stream mode. Within a group, MLX fuses kernels for reduced dispatch overhead. Between groups, `eval()` materializes results and allows Smelt to page expert weights in/out. This gives the best of both worlds — kernel fusion within groups and memory efficiency across groups.

Performance characteristics:
- group_size=4 (default): ~14GB peak for one group, ~10-15% throughput improvement from kernel fusion
- group_size=43 (full graph): ~151GB peak, ~30% improvement from cross-layer fusion, only for large-memory machines
- When memory is sufficient to hold all layers (no Smelt needed), group_size auto-sets to num_layers for maximum fusion

Architecture: DeepSeek V4 Flash 4-bit has 43 layers at ~3.5GB backbone weights per layer. On a 48GB Mac with Smelt stream, group_size=4 fits comfortably while leaving headroom for KV cache and activations.

## Glossary

- **Compile_Module**: The new dmlx module (`src/compile_mode.zig`) that manages layer-group compilation strategy and compiled closure lifecycle
- **Layer_Group**: A contiguous sequence of `group_size` transformer layers compiled into a single fused closure
- **Group_Size**: The number of consecutive layers compiled together as one closure (configurable, default 4)
- **Compiled_Closure**: An MLX closure that has been passed through `mlx_compile()`, producing fused Metal kernels on first invocation
- **Eager_Mode**: The current execution model where each layer is executed and eval'd individually without kernel fusion
- **Compile_Mode**: The execution model where layers are grouped into compiled closures with eval between groups
- **Smelt_Stream**: The memory paging mode where expert weights are loaded on-demand from SSD, requiring eval between groups to trigger paging
- **Smelt_Preload**: The memory mode where expert weights are preloaded into memory, still compatible with layer-group compile
- **Forward_Loop**: The loop in `deepseek_v4.zig` that iterates over transformer layers, currently calling eval after each layer
- **Auto_Detect**: The strategy that selects optimal group_size based on available system memory and model size
- **Full_Graph_Compile**: The special case where group_size equals num_layers, compiling the entire forward pass as one closure (no Smelt needed)

## Requirements

### Requirement 1: Compile Mode CLI Configuration

**User Story:** As a dmlx user, I want to enable layer-group compilation via CLI flags, so that I can opt into compilation without changing the default eager behavior.

#### Acceptance Criteria

1. WHEN the `--compile` CLI flag is provided, THE dmlx CLI SHALL enable layer-group compilation for inference
2. WHEN the `--compile` CLI flag is not provided, THE dmlx CLI SHALL use Eager_Mode with per-layer eval
3. WHEN the `--compile-group-size=N` flag is provided, THE dmlx CLI SHALL set Group_Size to N
4. WHEN `--compile` is provided without `--compile-group-size`, THE Compile_Module SHALL use Auto_Detect to determine optimal Group_Size
5. IF `--compile-group-size` is provided without `--compile`, THEN THE dmlx CLI SHALL report a configuration error and exit

### Requirement 2: Compile Mode Module

**User Story:** As a dmlx developer, I want a standalone compile mode module, so that compilation logic is encapsulated and testable independently of the model.

#### Acceptance Criteria

1. THE Compile_Module SHALL be implemented as `src/compile_mode.zig` with no direct dependencies on model-specific code
2. THE Compile_Module SHALL expose a `CompileConfig` struct containing `enabled` (bool), `group_size` (u32), and `auto_detect` (bool) fields
3. THE Compile_Module SHALL expose a `compileLayerGroup` function that accepts a closure over N layers and returns a Compiled_Closure
4. THE Compile_Module SHALL expose an `init` function that accepts CompileConfig and validates parameters
5. THE Compile_Module SHALL call `mlx-zig`'s `enableCompile()` when initialized with `enabled=true`

### Requirement 3: Layer-Group Compilation Strategy

**User Story:** As a dmlx user, I want layers compiled in groups with eval between groups, so that I get kernel fusion benefits while maintaining memory paging compatibility.

#### Acceptance Criteria

1. WHEN Compile_Mode is enabled, THE Forward_Loop SHALL partition the model's layers into consecutive groups of Group_Size layers
2. WHEN a Layer_Group is executed, THE Compile_Module SHALL compile the group's forward computation as a single closure with `shapeless=false`
3. WHEN a Layer_Group finishes execution, THE Forward_Loop SHALL call `eval()` on the group output to materialize results
4. IF the total number of layers is not evenly divisible by Group_Size, THEN THE Compile_Module SHALL compile the remaining layers as a smaller final group
5. WHEN Group_Size equals the total number of layers, THE Forward_Loop SHALL compile all layers as one closure and skip intermediate eval calls

### Requirement 4: Auto-Detection of Optimal Group Size

**User Story:** As a dmlx user, I want the system to automatically choose the best group size for my hardware, so that I get optimal performance without manual tuning.

#### Acceptance Criteria

1. WHEN Auto_Detect is active, THE Compile_Module SHALL query available system memory via MLX's memory API
2. WHEN available memory is sufficient to hold all layer weights simultaneously, THE Compile_Module SHALL set Group_Size to num_layers (Full_Graph_Compile)
3. WHEN available memory is constrained (Smelt active), THE Compile_Module SHALL calculate Group_Size as `floor(available_memory / per_layer_memory_estimate)` capped at num_layers
4. THE Compile_Module SHALL ensure the auto-detected Group_Size is at least 2 (compiling a single layer provides minimal benefit)
5. WHEN Auto_Detect completes, THE Compile_Module SHALL log the selected Group_Size and the reasoning (available memory, per-layer estimate)

### Requirement 5: Smelt Compatibility

**User Story:** As a dmlx user running on a 48GB Mac with Smelt stream, I want compile mode to work alongside Smelt, so that I get both memory paging and kernel fusion benefits.

#### Acceptance Criteria

1. WHEN `--compile` and `--smelt` are both provided, THE Compile_Module SHALL enable layer-group compilation (they are NOT mutually exclusive)
2. WHILE Smelt_Stream is active, THE Forward_Loop SHALL call `eval()` between Layer_Groups to allow expert weight paging
3. WHILE Smelt_Preload is active, THE Forward_Loop SHALL call `eval()` between Layer_Groups to allow weight management
4. WHEN Smelt is active with Auto_Detect, THE Compile_Module SHALL account for Smelt's memory overhead when calculating Group_Size
5. THE Compile_Module SHALL NOT require disabling Smelt to use compilation

### Requirement 6: Compiled Closure Lifecycle Management

**User Story:** As a dmlx developer, I want compiled closures to be properly created, cached, and freed, so that there are no resource leaks and recompilation is minimized.

#### Acceptance Criteria

1. THE Compile_Module SHALL cache Compiled_Closures after first compilation, keyed by group index
2. WHEN a cached Compiled_Closure exists for a group, THE Forward_Loop SHALL reuse it without recompilation
3. WHEN the model is deinitialized, THE Compile_Module SHALL free all cached Compiled_Closures via `Closure.deinit()`
4. THE Compile_Module SHALL provide a `deinit` function that releases all resources including the MLX compile cache
5. WHEN `deinit` is called, THE Compile_Module SHALL call `mlx-zig`'s compile cache clear function

### Requirement 7: Fallback to Eager on Compilation Failure

**User Story:** As a dmlx user, I want the system to gracefully fall back to eager mode if compilation fails, so that inference continues even if compile encounters an error.

#### Acceptance Criteria

1. IF `compileLayerGroup` returns an error from MLX, THEN THE Compile_Module SHALL fall back to Eager_Mode for that group
2. IF compilation fails for any group, THEN THE Compile_Module SHALL log the error with the group index and layer range
3. WHEN a group falls back to Eager_Mode, THE Forward_Loop SHALL execute those layers individually with per-layer eval
4. IF all groups fail compilation, THEN THE Compile_Module SHALL disable Compile_Mode entirely and log a summary warning
5. THE Compile_Module SHALL NOT retry failed compilations within the same inference session

### Requirement 8: Observability and Diagnostics

**User Story:** As a dmlx user, I want to see compile mode status and performance metrics, so that I can verify compilation is working and measure its impact.

#### Acceptance Criteria

1. WHEN Compile_Mode is enabled, THE dmlx CLI SHALL log "Compile mode: enabled, group_size=N, groups=M" at startup
2. WHEN a Layer_Group is compiled for the first time, THE Compile_Module SHALL log the JIT compilation time in milliseconds for that group
3. WHEN Auto_Detect selects a Group_Size, THE Compile_Module SHALL log the detection result with available memory and per-layer estimate
4. WHEN a fallback to Eager_Mode occurs, THE Compile_Module SHALL log the reason including the MLX error message
5. WHILE Compile_Mode is active, THE Compile_Module SHALL track and expose total compilation time across all groups

### Requirement 9: Correctness Preservation

**User Story:** As a dmlx user, I want compiled output to be numerically identical to eager mode output, so that I can trust compilation does not alter model behavior.

#### Acceptance Criteria

1. THE Compiled_Closure output for any Layer_Group SHALL be numerically identical to executing those layers in Eager_Mode
2. THE Compile_Module SHALL NOT alter the order of operations within a Layer_Group compared to eager execution
3. WHEN KV caches are used, THE Compiled_Closure SHALL correctly read from and write to the same cache entries as Eager_Mode
4. FOR ALL valid inputs, parsing the compiled output then comparing to eager output SHALL produce equivalent tensors (round-trip correctness)
5. THE Compile_Module SHALL preserve the model's dtype throughout compilation (no implicit precision changes)

### Requirement 10: Phase 2 Placeholder — Layer-Internal Mixed Compile

**User Story:** As a dmlx developer, I want a documented extension point for future layer-internal mixed compilation, so that we can later compile attention while keeping MoE eager within a single layer.

#### Acceptance Criteria

1. THE Compile_Module SHALL define a `MixedCompileStrategy` enum with values: `group_only` (Phase 1), `attention_compiled_moe_eager` (Phase 2 future)
2. THE CompileConfig SHALL include a `strategy` field defaulting to `group_only`
3. WHEN `strategy` is set to `attention_compiled_moe_eager`, THE Compile_Module SHALL return an "unimplemented" error with a descriptive message
4. THE Compile_Module source code SHALL include documentation comments describing the Phase 2 design intent: compile attention blocks while keeping MoE routing and expert execution in eager mode
5. THE Compile_Module interface SHALL be designed so that Phase 2 can be added without breaking changes to the public API
