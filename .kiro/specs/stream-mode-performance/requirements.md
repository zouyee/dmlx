# Requirements Document

## Introduction

Stream mode in mlx-zig enables DeepSeek V4 inference on memory-constrained hardware (48GB Mac) by loading expert weights on-demand from disk instead of preloading them all into memory. While stream mode is functionally correct, it currently takes 10+ minutes per token because it reloads all expert weights from disk on every token generation step. This feature addresses three critical performance bottlenecks: lack of expert caching, full tensor loading when only a subset is needed, and synchronous I/O that prevents overlap between disk reads and GPU computation. The goal is to reduce per-token latency from minutes to seconds while keeping memory usage under 20GB.

## Glossary

- **Stream_Mode**: The expert loading strategy in mlx-zig that loads MoE expert weights on-demand from SSD during inference, designed for systems with insufficient memory to hold all expert weights simultaneously.
- **Expert_Cache**: An in-memory LRU (Least Recently Used) cache that retains recently loaded expert weight tensors across token generation steps, avoiding redundant disk reads for experts that are reused.
- **Expert_Tensor**: A fused weight tensor stored on disk in safetensors format, containing weights for all 256 experts along axis 0. Each MoE layer has six expert tensors: gate_proj, up_proj, down_proj, and their corresponding quantization scales.
- **Router**: The component within each MoE layer that selects which experts (topk=6 out of 256) process a given token, producing expert indices and routing scores.
- **LRU_Eviction**: A cache replacement policy that removes the least recently used entry when the cache is full, prioritizing retention of frequently accessed experts.
- **Tensor_Index**: The safetensors header index that maps tensor names to file offsets, enabling pread-based random access without loading entire shard files.
- **Partial_Read**: A disk read operation that loads only the byte ranges corresponding to selected expert rows from a fused expert tensor, rather than loading the full tensor containing all 256 experts.
- **Prefetch_Pipeline**: A mechanism that overlaps disk I/O for the next layer's expert weights with GPU computation for the current layer, hiding disk latency behind compute time.
- **MoE_Layer**: A Mixture-of-Experts transformer layer in DeepSeek V4, containing a router and 256 experts. The model has 43 MoE layers.
- **Shard_File**: One of the 33 safetensors files (~4.3GB each) that together store the full 151GB DeepSeek V4 Flash 4-bit model on disk.
- **mxfp4**: The 4-bit mixed-precision floating point quantization format used by DeepSeek V4 Flash, which packs multiple expert values into contiguous byte groups that cannot be arbitrarily sliced at the byte level.
- **gatherQmm**: The quantized gather matrix multiplication operation that dispatches input tokens to their assigned experts using index-based weight selection from a fused expert tensor.
- **Token_Generation_Step**: One iteration of the autoregressive inference loop, where the model processes the current token through all 43 MoE layers to produce the next token.
- **File_Descriptor_Pool**: A pool of pre-opened file descriptors for shard files, avoiding the overhead of repeated open/close system calls during expert loading.

## Requirements

### Requirement 1: Expert Weight LRU Cache

**User Story:** As a user running DeepSeek V4 inference in stream mode on a 48GB Mac, I want expert weights to be cached in memory across token generation steps, so that experts reused between adjacent tokens are not redundantly reloaded from disk.

#### Acceptance Criteria

1. WHEN the Stream_Mode loads an Expert_Tensor for a given MoE_Layer and expert index, THE Expert_Cache SHALL store the loaded tensor keyed by layer index and tensor name.
2. WHEN the Stream_Mode requires an Expert_Tensor that exists in the Expert_Cache, THE Stream_Mode SHALL return the cached tensor without performing any disk I/O.
3. WHEN the Expert_Cache reaches its configured maximum memory capacity, THE Expert_Cache SHALL evict entries using LRU_Eviction to make room for new entries.
4. THE Expert_Cache SHALL track its current memory usage by summing the byte sizes of all cached tensors.
5. WHEN the Expert_Cache is initialized, THE Expert_Cache SHALL accept a configurable maximum memory budget in bytes, defaulting to 4GB.
6. IF the Expert_Cache receives a request for a tensor larger than the maximum memory budget, THEN THE Expert_Cache SHALL bypass caching and load the tensor directly from disk.
7. THE Expert_Cache SHALL report cache hit and miss counts for diagnostic logging.

### Requirement 2: Per-Expert Sliced Tensor Loading

**User Story:** As a user running stream mode inference, I want the system to load only the expert rows selected by the Router from each fused Expert_Tensor, so that disk I/O is reduced from ~256MB per tensor to ~4-8MB per tensor.

#### Acceptance Criteria

1. WHEN the Router selects a set of expert indices for a given MoE_Layer, THE Stream_Mode SHALL compute the byte offset and byte length for each selected expert row within the fused Expert_Tensor on disk.
2. WHEN loading selected expert rows, THE Stream_Mode SHALL issue pread calls targeting only the byte ranges of the selected experts, rather than reading the full fused tensor.
3. WHEN the quantization format is mxfp4, THE Stream_Mode SHALL load expert rows at mxfp4 group-aligned boundaries to preserve the quantization packing format.
4. THE Stream_Mode SHALL assemble the individually loaded expert rows into a mini fused tensor with shape [n_selected, ...] that is compatible with gatherQmm.
5. WHEN loading scale tensors for quantized experts, THE Stream_Mode SHALL apply the same per-expert partial read strategy used for weight tensors.
6. IF a pread call for a partial expert row returns fewer bytes than expected, THEN THE Stream_Mode SHALL return an error indicating an incomplete read.

### Requirement 3: File Descriptor Pooling

**User Story:** As a user running stream mode inference, I want shard file descriptors to be opened once and reused across token generation steps, so that repeated open/close system call overhead is eliminated.

#### Acceptance Criteria

1. WHEN the Stream_Mode is initialized, THE File_Descriptor_Pool SHALL open file descriptors for all Shard_Files referenced by the Tensor_Index.
2. WHEN the Stream_Mode requests a tensor from a specific Shard_File, THE File_Descriptor_Pool SHALL return the pre-opened file descriptor for that shard without calling open().
3. WHEN the Stream_Mode is deinitialized, THE File_Descriptor_Pool SHALL close all open file descriptors.
4. IF a Shard_File cannot be opened during initialization, THEN THE File_Descriptor_Pool SHALL return an error identifying the inaccessible shard path.

### Requirement 4: Asynchronous I/O and Layer Prefetching

**User Story:** As a user running stream mode inference, I want disk reads for the next MoE layer's experts to overlap with GPU computation for the current layer, so that disk latency is partially hidden behind compute time.

#### Acceptance Criteria

1. WHILE the Stream_Mode is computing the forward pass for MoE_Layer N, THE Prefetch_Pipeline SHALL begin loading expert weights for MoE_Layer N+1 using the Router's predicted expert selections.
2. WHEN the forward pass for MoE_Layer N completes and the Stream_Mode advances to MoE_Layer N+1, THE Prefetch_Pipeline SHALL provide the prefetched expert weights without additional disk I/O if the prefetched experts match the required experts.
3. IF the Router's actual expert selections for MoE_Layer N+1 differ from the prefetched selections, THEN THE Prefetch_Pipeline SHALL load the missing experts from disk and discard unused prefetched experts.
4. THE Prefetch_Pipeline SHALL limit prefetch memory usage to one layer's worth of expert weights to avoid exceeding the memory budget.

### Requirement 5: Cache-Aware Expert Loading Integration

**User Story:** As a user running stream mode inference, I want the Expert_Cache, partial reads, and file descriptor pooling to work together seamlessly in the streamingForward path, so that the combined optimizations deliver end-to-end latency improvement.

#### Acceptance Criteria

1. WHEN the streamingForward function is called for a given MoE_Layer, THE Stream_Mode SHALL first check the Expert_Cache for each required Expert_Tensor before attempting any disk I/O.
2. WHEN a cache miss occurs for an Expert_Tensor, THE Stream_Mode SHALL load the tensor using per-expert partial reads through the File_Descriptor_Pool, then insert the result into the Expert_Cache.
3. THE Stream_Mode SHALL produce numerically identical output to the current stream mode implementation for the same input token and expert selections.
4. WHEN all optimizations are active, THE Stream_Mode SHALL generate a token in under 30 seconds on a 48GB Apple Silicon Mac with SSD read speeds of 3GB/s, for single-token generation with DeepSeek V4 Flash 4-bit.
5. WHILE all optimizations are active, THE Stream_Mode SHALL maintain peak memory usage below 20GB for DeepSeek V4 Flash 4-bit inference on a 48GB Mac.

### Requirement 6: Diagnostic Logging and Performance Metrics

**User Story:** As a developer tuning stream mode performance, I want the system to report cache statistics and per-token I/O metrics, so that I can identify remaining bottlenecks and verify optimization effectiveness.

#### Acceptance Criteria

1. WHEN a Token_Generation_Step completes, THE Stream_Mode SHALL log the total disk bytes read, cache hit count, cache miss count, and wall-clock time for that step.
2. WHEN the Expert_Cache evicts an entry, THE Expert_Cache SHALL log the evicted tensor's layer index and tensor name at debug log level.
3. THE Stream_Mode SHALL log the total Expert_Cache memory usage at the start of each Token_Generation_Step.
