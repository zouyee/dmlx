# Requirements Document

## Introduction

This document specifies the requirements for taking mlx-zig from its current state (2 model architectures, partial GPU utilization, single-threaded server, memory leaks) to a production-level LLM inference engine on Apple Silicon. The scope covers six phases: foundation fixes, inference engine refactoring, service layer, advanced inference, quantization & training, and production operations. Requirements are derived from the production roadmap, codebase audit, and analysis of vLLM, mlx-lm, oMLX, and TileKernels.

## Glossary

- **MLX_Engine**: The mlx-zig inference engine — the top-level system that orchestrates model loading, request scheduling, generation, and response delivery
- **NN_Layer**: A neural network layer implementation in mlx-zig (Linear, RMSNorm, RoPE, Embedding, LSTM, GRU, Attention, etc.)
- **Forward_Pass**: A single execution of a model's computation graph from input tokens to output logits
- **KV_Cache**: Key-Value cache storing attention state across generation steps to avoid recomputation
- **Paged_KV_Cache**: A KV_Cache implementation that allocates memory in fixed-size blocks rather than contiguous buffers
- **Tiered_KV_Cache**: A KV_Cache with two storage tiers — hot (RAM) for active blocks and cold (SSD via safetensors) for evicted blocks
- **Block**: A fixed-size unit of KV_Cache storage (default 16 tokens per block) used by Paged_KV_Cache
- **Block_Manager**: The component that allocates, frees, and tracks Block ownership for Paged_KV_Cache
- **Copy_on_Write (CoW)**: A memory optimization where multiple requests share the same Block until one modifies it, at which point a copy is made
- **Scheduler**: The component that selects requests from waiting/running queues, allocates KV_Cache blocks, and orchestrates engine steps
- **Engine_Step**: One iteration of the Scheduler's three-phase loop: schedule → forward → postprocess
- **Continuous_Batching**: A serving strategy where multiple requests at different generation stages are batched into a single Forward_Pass
- **Chunked_Prefill**: Splitting a long prompt's prefill computation into multiple chunks across Engine_Steps to reduce latency for other requests
- **Prefix_Caching**: Reusing previously computed KV_Cache blocks for prompts that share a common prefix, identified by block-level hashing
- **Speculative_Decoding**: An acceleration technique where a lightweight draft model proposes multiple tokens that the target model verifies in a single forward pass
- **Guided_Decoding**: Constraining token generation to follow a grammar (JSON schema, regex) by masking logits at each step via a finite state machine
- **Model_Registry**: A compile-time lookup table mapping architecture names to their VTable implementations
- **Model_Pool**: A runtime manager that loads, caches, and evicts multiple models with LRU policy and memory limits
- **SSE_Streaming**: Server-Sent Events protocol for delivering tokens incrementally to HTTP clients
- **Arena_Allocator**: A bulk-free memory allocator used to release all intermediate Arrays created during a Forward_Pass in one operation
- **EagerContext**: The mlx-zig execution context holding an allocator and MLX stream for dispatching operations
- **mlx-c**: The C API bindings to Apple's MLX framework that provide GPU-accelerated operations via Metal
- **fast.zig**: mlx-zig module binding MLX fused GPU kernels (rms_norm, rope, sdpa, layer_norm)
- **kv_bits**: The bit-width parameter (4, 8, or 16) controlling KV_Cache quantization precision
- **max_kv_size**: The maximum number of tokens whose KV state can be cached, auto-configured based on device memory
- **Quantization**: Reducing weight or activation precision (INT4/INT8/FP8) to decrease memory usage and increase throughput
- **QLoRA**: Quantized Low-Rank Adaptation — fine-tuning with 4-bit quantized base weights and trainable low-rank adapters
- **MoE_Router**: Mixture-of-Experts routing component that selects top-k experts per token and dispatches computation

## Requirements

### Requirement 1: Error Handling with Context

**User Story:** As a developer debugging inference failures, I want MLX error messages to include the original C++ exception text, so that I can diagnose issues without guessing.

#### Acceptance Criteria

1. WHEN an mlx-c function returns a non-zero return code, THE MLX_Engine SHALL log the return code and the error message retrieved from the MLX error handler before returning an error
2. WHEN an mlx-c function returns a non-zero return code and no error message is available from the error handler, THE MLX_Engine SHALL log the return code alone and return an error
3. THE MLX_Engine SHALL register a Zig error handler with mlx-c at initialization via `mlx_set_error_handler` to capture C++ exception messages

### Requirement 2: Memory Safety for Forward Passes

**User Story:** As an operator running long inference sessions, I want intermediate Arrays created during forward passes to be automatically released, so that the process does not leak memory over time.

#### Acceptance Criteria

1. THE MLX_Engine SHALL use an Arena_Allocator scoped to each Forward_Pass to track all intermediate Array allocations
2. WHEN a Forward_Pass completes, THE MLX_Engine SHALL release all intermediate Arrays allocated during that pass via the Arena_Allocator
3. THE MLX_Engine SHALL NOT use `@constCast` on MLX array data pointers; all data mutations SHALL use mlx-c operator chains that respect Copy_on_Write semantics

### Requirement 3: NN Layer GPU Acceleration

**User Story:** As a user running inference on Apple Silicon, I want all neural network layers to execute on the GPU via MLX fused kernels, so that inference is 100-1000x faster than CPU scalar loops.

#### Acceptance Criteria

1. THE NN_Layer RMSNorm SHALL delegate to `fast.rmsNorm()` for its forward computation
2. THE NN_Layer RoPE SHALL delegate to `fast.rope()` for rotary position embedding computation
3. THE NN_Layer MultiHeadAttention SHALL delegate to `fast.scaledDotProductAttention()` for attention computation
4. THE NN_Layer Embedding SHALL use the `mlx_take` operation for index-based lookup
5. WHEN a loss function computes its output, THE MLX_Engine SHALL use mlx-c operator graph composition (following the crossEntropyGraph pattern) instead of CPU scalar loops
6. WHEN an LSTM, GRU, or RNN layer computes its forward pass, THE NN_Layer SHALL use mlx-c matmul, sigmoid, and tanh operator chains instead of CPU scalar loops

### Requirement 4: Build System Portability

**User Story:** As a developer on a non-Homebrew macOS setup or a CI environment, I want the build system to locate mlx-c without hardcoded paths, so that the project builds on any Apple Silicon machine.

#### Acceptance Criteria

1. THE MLX_Engine build system SHALL use pkg-config to locate mlx-c include and library paths as the primary discovery method
2. WHERE a `-Dmlx_prefix` build option is provided, THE MLX_Engine build system SHALL use that path instead of pkg-config
3. THE MLX_Engine build system SHALL NOT contain hardcoded `/opt/homebrew` paths
4. THE MLX_Engine build system SHALL pin the zig-regex dependency to a fixed release tag or commit hash rather than a branch name

### Requirement 5: Three-Layer Generation Architecture

**User Story:** As a developer integrating mlx-zig into an application, I want a layered generation API (single-step, streaming, complete), so that I can choose the right abstraction for my use case.

#### Acceptance Criteria

1. THE MLX_Engine SHALL provide a `generateStep` function that executes one forward pass and returns a single sampled token
2. THE MLX_Engine SHALL provide a `streamGenerate` function that yields tokens incrementally via a callback as they are generated
3. THE MLX_Engine SHALL provide a `generate` function that returns the complete sequence of generated tokens after generation finishes
4. THE `streamGenerate` and `generate` functions SHALL be implemented in terms of `generateStep`

### Requirement 6: Model Architecture Registry

**User Story:** As a user, I want to load any supported model architecture by name from a config file, so that I do not need to modify code to switch between LLaMA, DeepSeek, Mistral, or Qwen models.

#### Acceptance Criteria

1. THE MLX_Engine SHALL maintain a Model_Registry mapping architecture name strings to their corresponding model VTable implementations
2. WHEN a model config file specifies an `architectures` field, THE MLX_Engine SHALL look up the architecture in the Model_Registry and instantiate the corresponding model
3. THE Model_Registry SHALL support at minimum: LLaMA, DeepSeek V4, Mistral, Qwen2, and Gemma architectures
4. IF a model config specifies an architecture not present in the Model_Registry, THEN THE MLX_Engine SHALL return a descriptive error naming the unsupported architecture

### Requirement 7: Prompt Cache Persistence

**User Story:** As a user with recurring long prompts (system prompts, RAG context), I want to save and reload KV cache state to disk, so that I skip redundant prefill computation on subsequent requests.

#### Acceptance Criteria

1. THE MLX_Engine SHALL provide a `savePromptCache` function that serializes KV_Cache state to a safetensors file at a specified path
2. THE MLX_Engine SHALL provide a `loadPromptCache` function that deserializes KV_Cache state from a safetensors file
3. WHEN a prompt cache file is loaded, THE MLX_Engine SHALL validate that the cached state is compatible with the current model configuration (number of layers, head dimensions, KV head count)
4. IF a prompt cache file is incompatible with the current model, THEN THE MLX_Engine SHALL return a descriptive error and fall back to full prefill

### Requirement 8: Operator Fusion via mlx_compile

**User Story:** As a user seeking maximum throughput, I want composite operations (SwiGLU MLP, AdamW step) to be compiled into fused GPU operations, so that temporary Array allocations and kernel launch overhead are minimized.

#### Acceptance Criteria

1. THE MLX_Engine SHALL compile the SwiGLU MLP forward function into a fused operation using `mlx_compile`
2. THE MLX_Engine SHALL compile the AdamW optimizer step into a fused operation using `mlx_compile`
3. WHEN a compiled fused operation is executed, THE MLX_Engine SHALL produce numerically equivalent results to the unfused implementation

### Requirement 9: Request Scheduler

**User Story:** As an operator serving multiple concurrent users, I want a scheduler that manages request queues and allocates KV cache blocks, so that the engine can serve multiple requests efficiently.

#### Acceptance Criteria

1. THE Scheduler SHALL maintain separate waiting and running request queues
2. WHEN an Engine_Step begins, THE Scheduler SHALL prioritize running requests (decode phase) over waiting requests (prefill phase)
3. WHEN a request enters the running queue, THE Scheduler SHALL allocate KV_Cache blocks for that request via the Block_Manager
4. WHEN a request completes (stop token or max length reached), THE Scheduler SHALL free the KV_Cache blocks associated with that request
5. IF insufficient KV_Cache blocks are available for a new request, THEN THE Scheduler SHALL keep the request in the waiting queue until blocks become available

### Requirement 10: PagedAttention Implementation

**User Story:** As an operator, I want KV cache memory allocated in fixed-size blocks with on-demand allocation, so that memory waste is reduced from 60-80% to under 4%.

#### Acceptance Criteria

1. THE Paged_KV_Cache SHALL allocate memory in fixed-size Blocks (default 16 tokens per block)
2. THE Block_Manager SHALL maintain a free block pool and allocate blocks on demand as sequences grow
3. WHEN a request completes, THE Block_Manager SHALL return all of that request's blocks to the free pool
4. THE Paged_KV_Cache SHALL support Copy_on_Write: WHEN multiple requests share a Block and one request modifies it, THE Block_Manager SHALL copy the Block before mutation
5. THE Paged_KV_Cache SHALL track block-to-request mappings so that blocks can be freed when their owning request completes

### Requirement 11: KV Cache Quantization

**User Story:** As a user running long-context inference on a memory-constrained Apple Silicon device, I want to quantize KV cache entries to 4-bit or 8-bit precision, so that I can fit 4x more context in the same RAM.

#### Acceptance Criteria

1. THE KV_Cache SHALL accept a `kv_bits` configuration parameter with valid values of 4, 8, or 16 (no quantization)
2. WHEN `kv_bits` is set to 4 or 8, THE KV_Cache SHALL quantize key and value tensors using `mlx_quantize` before storage with a configurable group size (default 64)
3. WHEN quantized KV entries are needed for attention computation, THE KV_Cache SHALL dequantize them using `mlx_dequantize` before passing to the attention kernel
4. WHEN `kv_bits` is 16, THE KV_Cache SHALL store keys and values without quantization

### Requirement 12: SSE Streaming Response

**User Story:** As a client application, I want the HTTP server to stream tokens as Server-Sent Events, so that I see partial responses immediately instead of waiting for full generation.

#### Acceptance Criteria

1. WHEN a chat completion request has `stream: true`, THE MLX_Engine HTTP server SHALL respond with `Content-Type: text/event-stream` and send each generated token as an SSE `data:` event in OpenAI-compatible format
2. WHEN generation completes for a streaming request, THE MLX_Engine HTTP server SHALL send a final SSE event with `finish_reason: "stop"` followed by `data: [DONE]`
3. WHILE a streaming response is in progress and the prefill phase takes longer than 5 seconds, THE MLX_Engine HTTP server SHALL send SSE keep-alive comments to prevent client read timeouts

### Requirement 13: Continuous Batching

**User Story:** As an operator, I want multiple requests at different generation stages to be processed in a single forward pass, so that GPU utilization is maximized.

#### Acceptance Criteria

1. THE MLX_Engine SHALL concatenate token sequences from multiple active requests into a single batched input tensor for each Forward_Pass
2. THE MLX_Engine SHALL maintain per-request position indices so that each request's tokens attend only to their own sequence via attention masks
3. WHEN a new request arrives while other requests are generating, THE Scheduler SHALL add the new request to the next Engine_Step without waiting for existing requests to complete

### Requirement 14: Chunked Prefill

**User Story:** As an operator, I want long prompt prefills to be split across multiple engine steps, so that short requests are not blocked by a single long prefill.

#### Acceptance Criteria

1. THE MLX_Engine SHALL accept a `max_prefill_tokens` configuration parameter that limits the number of tokens processed in a single prefill chunk
2. WHEN a request's prompt exceeds `max_prefill_tokens`, THE Scheduler SHALL split the prefill into multiple chunks processed across consecutive Engine_Steps
3. WHILE a request is being chunked-prefilled, THE Scheduler SHALL continue processing decode steps for other active requests in the same Engine_Steps

### Requirement 15: Prefix Caching

**User Story:** As a user sending many requests with shared system prompts, I want the engine to reuse previously computed KV cache blocks for common prefixes, so that redundant prefill computation is eliminated.

#### Acceptance Criteria

1. THE Paged_KV_Cache SHALL compute a hash for each Block based on the previous block's hash and the token IDs in the current block
2. WHEN a new request's prompt prefix matches cached block hashes, THE Block_Manager SHALL reuse the existing cached blocks instead of recomputing them
3. WHEN a cached block is reused by a new request that will modify subsequent blocks, THE Block_Manager SHALL apply Copy_on_Write to preserve the shared prefix blocks

### Requirement 16: Speculative Decoding

**User Story:** As a user, I want the engine to use speculative decoding to generate multiple tokens per forward pass of the target model, so that end-to-end generation latency is reduced.

#### Acceptance Criteria

1. THE MLX_Engine SHALL support an n-gram draft proposal mechanism that searches the existing generated context for matching n-gram suffixes and proposes continuation tokens
2. WHEN draft tokens are proposed, THE MLX_Engine SHALL verify all proposed tokens in a single forward pass of the target model
3. THE MLX_Engine SHALL accept or reject each proposed token based on the target model's probability distribution, ensuring statistical equivalence to standard autoregressive sampling

### Requirement 17: Guided Decoding

**User Story:** As a developer building structured output applications, I want to constrain generation to follow a JSON schema or regex pattern, so that model output is always parseable.

#### Acceptance Criteria

1. WHEN a generation request includes a JSON schema constraint, THE MLX_Engine SHALL construct a finite state machine from the schema and apply a logits mask at each generation step that allows only tokens valid in the current FSM state
2. WHEN a generation request includes a regex constraint, THE MLX_Engine SHALL construct a finite state machine from the regex and apply a logits mask at each generation step
3. THE MLX_Engine SHALL set disallowed token logits to negative infinity before sampling

### Requirement 18: Weight Quantization Infrastructure

**User Story:** As a user deploying models on memory-constrained devices, I want to quantize model weights to INT4/INT8 precision, so that I can run larger models within available RAM.

#### Acceptance Criteria

1. THE MLX_Engine SHALL provide a `quantize` function that converts model weight tensors to a specified bit-width (4 or 8) with configurable group size using `mlx_quantize`
2. THE MLX_Engine SHALL provide a `dequantize` function that restores quantized tensors to their original precision using `mlx_dequantize`
3. THE MLX_Engine SHALL support loading pre-quantized model weights (GPTQ format) and performing inference with on-the-fly dequantization in each layer's forward pass

### Requirement 19: QLoRA Fine-Tuning

**User Story:** As a researcher fine-tuning large models on a single Apple Silicon machine, I want QLoRA support, so that I can train with 4-bit quantized base weights and low-rank adapters within limited memory.

#### Acceptance Criteria

1. THE MLX_Engine SHALL support 4-bit NormalFloat (NF4) quantization for base model weights during QLoRA training
2. THE MLX_Engine SHALL apply LoRA adapters on top of quantized base weights, computing the forward pass as `dequantize(W_base) * x + (B @ A) * x * scaling`
3. WHEN computing gradients during QLoRA training, THE MLX_Engine SHALL compute gradients only for the LoRA adapter parameters, not for the quantized base weights

### Requirement 20: MoE Routing Pipeline

**User Story:** As a user running Mixture-of-Experts models (DeepSeek V4), I want a complete MoE routing pipeline, so that expert selection and dispatch is efficient on Apple Silicon.

#### Acceptance Criteria

1. THE MoE_Router SHALL compute top-k expert selection from routing scores using mlx-c operations
2. THE MoE_Router SHALL expand input tokens to their assigned experts, compute expert outputs, and reduce (weighted sum) the results back to the original token dimension
3. THE MoE_Router SHALL support configurable number of experts and top-k selection count

### Requirement 21: Multi-Model Management

**User Story:** As an operator serving multiple models, I want the engine to manage a pool of loaded models with automatic eviction, so that I can serve different models without manual loading/unloading.

#### Acceptance Criteria

1. THE Model_Pool SHALL maintain a mapping of model names to loaded model instances
2. WHEN a request specifies a model not currently loaded, THE Model_Pool SHALL load the model and add it to the pool
3. WHEN loading a new model would exceed the configured memory limit, THE Model_Pool SHALL evict the least recently used model to free memory
4. WHERE a model is marked as pinned, THE Model_Pool SHALL NOT evict that model regardless of LRU ordering

### Requirement 22: Tiered KV Cache (RAM + SSD)

**User Story:** As a user running long-context inference, I want inactive KV cache blocks to be offloaded to SSD and restored on demand, so that I can handle contexts larger than available RAM.

#### Acceptance Criteria

1. THE Tiered_KV_Cache SHALL maintain a hot tier in RAM for actively accessed blocks and a cold tier on SSD for evicted blocks
2. WHEN the hot tier exceeds its configured capacity, THE Tiered_KV_Cache SHALL evict the least recently accessed blocks to the cold tier by serializing them as safetensors files
3. WHEN a block in the cold tier is needed for attention computation, THE Tiered_KV_Cache SHALL restore it to the hot tier by deserializing the safetensors file
4. THE Tiered_KV_Cache SHALL track block access recency to make eviction decisions

### Requirement 23: Process Memory Limits

**User Story:** As an operator, I want to set a maximum memory budget for the inference process, so that the engine does not cause system instability by consuming all available RAM.

#### Acceptance Criteria

1. THE MLX_Engine SHALL accept a `--max-process-memory` configuration parameter specifying the maximum memory usage as an absolute value or percentage of system RAM
2. WHILE the process memory usage exceeds the configured limit, THE MLX_Engine SHALL trigger LRU eviction of Model_Pool entries and Tiered_KV_Cache cold-tier offloading until usage falls below the limit
3. IF memory usage cannot be reduced below the limit after eviction, THEN THE MLX_Engine SHALL reject new requests with a descriptive error until memory is freed

### Requirement 24: Automatic max_kv_size Configuration

**User Story:** As a user, I want the engine to automatically determine the maximum KV cache size based on my device's available memory, so that I get optimal context length without manual tuning or OOM crashes.

#### Acceptance Criteria

1. WHEN `max_kv_size` is set to `auto`, THE MLX_Engine SHALL compute the maximum KV cache token capacity as: `(total_device_RAM - model_weight_bytes - safety_margin) / bytes_per_token_kv`
2. THE MLX_Engine SHALL compute `bytes_per_token_kv` as: `2 * num_kv_heads * head_dim * (kv_bits / 8) * num_layers`
3. WHERE `max_kv_size` is set to a specific integer value, THE MLX_Engine SHALL use that value directly without auto-computation
4. THE MLX_Engine SHALL expose `max_kv_size` via CLI as `--max-kv-size auto` (default) or `--max-kv-size <integer>`

### Requirement 25: Benchmark Tool

**User Story:** As a developer evaluating mlx-zig performance, I want a benchmark command that measures key inference metrics, so that I can compare performance across configurations and track regressions.

#### Acceptance Criteria

1. THE MLX_Engine SHALL provide a `benchmark` CLI command that runs inference with configurable input and output token counts
2. WHEN a benchmark run completes, THE MLX_Engine SHALL report: time to first token (TTFT), inter-token latency (ITL), throughput (tokens/second), and peak memory usage
3. THE MLX_Engine SHALL accept `--model`, `--input-tokens`, and `--output-tokens` parameters for the benchmark command

### Requirement 26: Numerical Verification Test Framework

**User Story:** As a developer, I want automated tests that verify mlx-zig's numerical output matches Python MLX reference outputs, so that I can catch correctness regressions.

#### Acceptance Criteria

1. THE MLX_Engine test suite SHALL include golden tests that compare each NN_Layer's output against pre-computed reference data generated by Python MLX
2. THE MLX_Engine test suite SHALL verify numerical equivalence using cosine similarity with a threshold of 0.9999 for float32 operations
3. THE MLX_Engine test suite SHALL include an end-to-end inference test using a small reference model (TinyLlama or equivalent) comparing output token sequences against Python MLX output
