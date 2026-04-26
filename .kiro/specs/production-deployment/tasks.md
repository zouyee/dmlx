# Implementation Plan: Production Deployment

## Overview

Transform mlx-zig from a prototype into a production-grade LLM inference engine on Apple Silicon. Implementation follows six dependency-ordered phases: foundation fixes, inference engine, service layer, advanced inference, quantization & training, and production operations. All computation flows through the MLX computation graph — no CPU scalar loops for tensor operations.

> **Audit Status (2026-04-25)**: Phase 0–5 modules are all implemented as standalone files.
> The primary remaining work is **integration** (Task 13) and **identified gaps** (Task 15).
> server.zig still hardcodes `createStandard` for KV cache and does not use Scheduler,
> Paged KV Cache, Continuous Batching, Prompt Cache, or Speculative/Guided Decoding.

## Tasks

- [x] 1. Phase 0: Foundation — Error handling, memory safety, NN GPU, build (R1–R4)
  - [x] 1.1 Verify error handler registration and contextual error logging in `src/c.zig`
    - Confirm `mlxErrorHandler` export captures C++ exception text via `mlx_set_error_handler`
    - Confirm `check()` logs rc + error message, and logs rc alone when no message is available
    - Add `initErrorHandler()` call at engine startup if not already present
    - _Requirements: R1.1, R1.2, R1.3_

  - [x] 1.2 Adopt ScopedArrayArena consistently across all model forward passes
    - Modify `src/models/llama.zig` and `src/models/deepseek_v4.zig` forward methods to wrap intermediate Arrays in `ScopedArrayArena` from `src/array_arena.zig`
    - Ensure only the final output Array escapes the arena scope
    - Remove any `@constCast` usage on MLX array data pointers; replace with mlx-c operator chains
    - _Requirements: R2.1, R2.2, R2.3_

  - [x] 1.3 Write property test for forward pass arena cleanup (Property 2)
    - **Property 2: Forward Pass Arena Cleanup**
    - Verify that after a forward pass completes and ScopedArrayArena is deinitialized, all intermediate Arrays are released and only the final output remains live
    - Minimum 100 iterations with random input tensors
    - **Validates: Requirements R2.1, R2.2**

  - [x] 1.4 GPU-accelerate remaining NN layers in `src/ops/nn.zig`
    - Rewrite `Embedding.forward` to use `mlx_take(weight, indices, 0)` instead of CPU scalar loop
    - Rewrite `LSTM.forward`, `GRU.forward`, `RNN.forward` to use `ops.matmul` + `ops.sigmoid` + `ops.tanh` operator chains
    - Migrate remaining loss functions in `src/ops/loss.zig` to graph-mode composition following the `crossEntropyGraph` pattern
    - _Requirements: R3.1, R3.2, R3.3, R3.4, R3.5, R3.6_

  - [x] 1.5 Write property test for NN layer GPU numerical equivalence (Property 1)
    - **Property 1: NN Layer GPU Numerical Equivalence**
    - Create `tests/generate_golden.py` script to generate reference data from Python MLX for each NN layer
    - Write `src/tests/golden_test.zig` comparing mlx-zig output against golden data with cosine similarity ≥ 0.9999
    - Cover: RMSNorm, RoPE, SDPA, Embedding, LSTM, GRU, loss functions
    - **Validates: Requirements R3.1, R3.2, R3.3, R3.4, R3.5, R3.6, R26.1, R26.2**

  - [x] 1.6 Fix build system portability in `build.zig`
    - Replace hardcoded `/opt/homebrew` fallback with `pkg-config` as primary mlx-c discovery
    - Keep `-Dmlx_prefix` as override, remove hardcoded path
    - Pin `zig-regex` dependency to a fixed commit hash in `build.zig.zon`
    - _Requirements: R4.1, R4.2, R4.3, R4.4_

- [x] 2. Checkpoint — Phase 0 complete

- [x] 3. Phase 1: Inference Engine — Generation API, model registry, prompt cache, fusion (R5–R8)
  - [x] 3.1 Implement three-layer generation API in `src/generation.zig`
    - _Requirements: R5.1, R5.2, R5.3, R5.4_
  - [x] 3.2 Write property test for generation API consistency (Property 3)
  - [x] 3.3 Implement model architecture registry in `src/model_registry.zig`
    - _Requirements: R6.1, R6.2, R6.3, R6.4_
  - [x] 3.4 Write property test for model registry lookup (Property 4)
  - [x] 3.5 Implement prompt cache persistence in `src/prompt_cache.zig`
    - _Requirements: R7.1, R7.2, R7.3, R7.4_
  - [x] 3.6 Write property test for prompt cache round-trip (Property 5)
  - [x] 3.7 Implement operator fusion in `src/ops/fused.zig`
    - _Requirements: R8.1, R8.2, R8.3_
  - [x] 3.8 Write property test for fused operation equivalence (Property 6)

- [x] 4. Checkpoint — Phase 1 complete

- [x] 5. Phase 2: Service Layer — Scheduler, paged attention, KV quant, SSE, batching (R9–R13)
  - [x] 5.1 Implement request scheduler in `src/scheduler.zig`
    - _Requirements: R9.1, R9.2, R9.3, R9.4, R9.5_
  - [x] 5.2 Write property test for scheduler prioritization (Property 7)
  - [x] 5.3 Implement block manager in `src/kvcache/paged.zig` (BlockManager with CoW + prefix hash)
    - _Requirements: R10.1, R10.2, R10.3, R10.4, R10.5_
  - [x] 5.4 Write property test for block conservation (Property 8)
  - [x] 5.5 Write property test for copy-on-write isolation (Property 9)
  - [x] 5.6 Implement KV cache quantization in `src/kvcache/quantized.zig` (4/8/16-bit with passthrough)
    - _Requirements: R11.1, R11.2, R11.3, R11.4_
  - [x] 5.7 Write property test for quantize-dequantize round-trip (Property 10)
  - [x] 5.8 Implement SSE streaming in `src/server.zig` (writeSSEEvent, handleStreamingCompletion)
    - _Requirements: R12.1, R12.2, R12.3_
  - [x] 5.9 Implement continuous batching in `src/batch_builder.zig`
    - _Requirements: R13.1, R13.2, R13.3_
  - [x] 5.10 Write property test for continuous batching attention isolation (Property 11)

- [x] 6. Checkpoint — Phase 2 complete

- [x] 7. Phase 3: Advanced Inference — Chunked prefill, prefix cache, spec decode, guided decode (R14–R17)
  - [x] 7.1 Implement chunked prefill in scheduler
    - _Requirements: R14.1, R14.2, R14.3_
  - [x] 7.2 Write property test for chunked prefill correctness (Property 12)
  - [x] 7.3 Implement prefix caching in block manager (hashBlock + findCachedPrefix)
    - _Requirements: R15.1, R15.2, R15.3_
  - [x] 7.4 Write property test for block hash determinism and prefix reuse (Property 13)
  - [x] 7.5 Implement speculative decoding in `src/speculative.zig`
    - _Requirements: R16.1, R16.2, R16.3_
  - [x] 7.6 Write property tests for speculative decoding (Properties 14, 15)
  - [x] 7.7 Implement guided decoding in `src/guided.zig`
    - _Requirements: R17.1, R17.2, R17.3_
  - [x] 7.8 Write property test for guided decoding constraint satisfaction (Property 16)

- [x] 8. Checkpoint — Phase 3 complete

- [x] 9. Phase 4: Quantization & Training — Weight quant, QLoRA, MoE (R18–R20)
  - [x] 9.1 Implement weight quantization infrastructure in `src/quantize.zig`
    - _Requirements: R18.1, R18.2, R18.3_
  - [x] 9.2 Implement QLoRA fine-tuning in `src/qlora.zig`
    - _Requirements: R19.1, R19.2, R19.3_
  - [x] 9.3 Write property test for QLoRA forward correctness (Property 17)
  - [x] 9.4 Extract reusable MoE router module in `src/moe_router.zig`
    - _Requirements: R20.1, R20.2, R20.3_
  - [x] 9.5 Write property test for MoE top-k selection (Property 18)

- [x] 10. Checkpoint — Phase 4 complete

- [x] 11. Phase 5: Production Ops — Model pool, tiered KV, memory, auto config, benchmark, tests (R21–R26)
  - [x] 11.1 Implement model pool in `src/model_pool.zig`
    - _Requirements: R21.1, R21.2, R21.3, R21.4_
  - [x] 11.2 Write property test for model pool LRU eviction with pinning (Property 19)
  - [x] 11.3 Implement tiered KV cache in `src/kvcache/tiered.zig`
    - _Requirements: R22.1, R22.2, R22.3, R22.4_
  - [x] 11.4 Write property test for tiered KV cache evict-restore round-trip (Property 20)
  - [x] 11.5 Implement process memory limiter in `src/memory.zig`
    - _Requirements: R23.1, R23.2, R23.3_
  - [x] 11.6 Implement auto max_kv_size configuration in `src/memory.zig`
    - _Requirements: R24.1, R24.2, R24.3, R24.4_
  - [x] 11.7 Write property test for auto max_kv_size formula (Property 21)
  - [x] 11.8 Implement benchmark tool in `src/benchmark.zig`
    - _Requirements: R25.1, R25.2, R25.3_
  - [x] 11.9 Implement numerical verification test framework
    - _Requirements: R26.1, R26.2, R26.3_

- [x] 12. Checkpoint — Phase 5 complete

- [x] 13. Final integration and wiring
  - [x] 13.1 Wire KV cache strategy selection into server
    - Replace hardcoded `createStandard` at `src/server.zig:189` with config-driven selection
    - Add `--kv-bits 4|8|16` CLI parameter to `ServerConfig` (default 4 for Apple Silicon)
    - When `kv_bits < 16`: use `createQuantized(allocator, config, kv_bits, 64, stream)`
    - When `kv_bits == 16`: use `createStandard(allocator, config, stream)`
    - Add `--kv-strategy standard|paged|quantized` CLI parameter for explicit strategy override
    - _Requirements: R11.1, R11.4_
    - _Gap: roadmap §8.3 缺口 1 — `--kv-bits` CLI 参数_

  - [x] 13.2 Wire Scheduler + Continuous Batching into server request loop
    - Replace serial `handleRequest` → `generateChatCompletion` with Scheduler-driven engine loop
    - Integrate `src/scheduler.zig` schedule() → forward → postprocess cycle
    - Integrate `src/batch_builder.zig` for multi-request batched forward passes
    - New requests enter Scheduler.waiting queue instead of blocking
    - _Requirements: R9, R13_

  - [x] 13.3 Wire Generation API into server
    - Use `streamGenerate` for SSE streaming requests (`stream: true`)
    - Use `generate` for non-streaming requests
    - Replace inline generation logic in server with `generation_mod` calls
    - _Requirements: R5_

  - [x] 13.4 Wire Prompt Cache into server and CLI
    - Add `--prompt-cache-file <path>` CLI parameter
    - On startup: if cache file exists, load via `prompt_cache.loadPromptCache`
    - On shutdown or explicit save: serialize via `prompt_cache.savePromptCache`
    - Validate cache compatibility with loaded model config
    - _Requirements: R7_

  - [x] 13.5 Wire Speculative Decoding into generation pipeline
    - Add `--speculative-ngram <n>` CLI parameter (default: disabled)
    - When enabled, wrap `generateStep` with NgramDrafter.propose → verifyDraft cycle
    - _Requirements: R16_

  - [x] 13.6 Wire Guided Decoding into generation pipeline
    - Add `response_format` field to chat completion request parsing
    - When JSON schema or regex provided, construct FSM and apply `GuidedDecoder.maskLogits` per step
    - _Requirements: R17_

  - [x] 13.7 Wire ModelPool for multi-model management
    - Replace single-model loading in `loadModel` with `ModelPool.getOrLoad`
    - Route requests to correct model based on `model` field in request body
    - Trigger LRU eviction when memory limit exceeded
    - _Requirements: R21_

  - [x] 13.8 Wire Tiered KV Cache into server
    - Add `--kv-tier ram|ssd` and `--kv-cold-dir <path>` CLI parameters
    - When `ssd`: use `TieredKVCache` wrapping `PagedKVCache` as hot tier
    - _Requirements: R22_

  - [x] 13.9 Wire CLI subcommands into `src/main.zig`
    - Add `serve` subcommand (currently inline in main)
    - Add `benchmark` subcommand → `src/benchmark.zig`
    - Add `quantize` subcommand → `src/quantize.zig`
    - Add `--kv-bits`, `--max-kv-size`, `--kv-strategy`, `--kv-tier` to `serve`
    - _Requirements: R24.4, R25_

  - [x] 13.10 Wire operator fusion into model forward passes
    - Replace unfused SwiGLU in LLaMA/DeepSeek V4 MLP with `compiledSwiGLU` from `src/ops/fused.zig`
    - Replace unfused AdamW step with `compiledAdamWStep` in training path
    - _Requirements: R8_

  - [x] 13.11 Wire MoE Router into DeepSeek V4
    - Refactor `DSV4Gate`/`DSV4MoE` in `deepseek_v4.zig` to delegate to `src/moe_router.zig`
    - Remove duplicated routing logic from model file
    - _Requirements: R20_

- [x] 14. Integration tests
  - [x] 14.1 Test: submit request → scheduler → forward → SSE streaming response
    - _Requirements: R9, R12, R13_
  - [x] 14.2 Test: multiple concurrent requests with continuous batching
    - _Requirements: R13_
  - [x] 14.3 Test: model pool load/evict cycle
    - _Requirements: R21_
  - [x] 14.4 Test: memory limit enforcement rejects requests when over budget
    - _Requirements: R23_
  - [x] 14.5 Test: `--kv-bits 4` produces quantized KV cache with correct cosine similarity
    - _Requirements: R11_
  - [x] 14.6 Test: prompt cache save → restart → load skips prefill
    - _Requirements: R7_

- [x] 15. Identified gaps from audit (Apple Silicon + DeepSeek V4)
  - [x] 15.1 Paged + Quantized combination strategy
    - `PagedKVCache` accepts `kv_bits` parameter (4/8/16) in constructor
    - When `kv_bits < 16`: quantizes KV data at block level via `quantizeArray()`, dequantizes on read
    - Factory `createPagedQuantized(allocator, config, kv_bits, group_size, stream)` available
    - Server wires via `--kv-strategy paged_quantized` + `--kv-bits 4|8|16`
    - _Gap: roadmap §8.3 缺口 2 — Paged + Quantized 组合_

  - [x] 15.2 DeepSeek V4: replace mean-pool compression with learned softmax-gated pooling
    - `compressKV()` dispatches to `softmaxGatedPool()` when gate weights present
    - Falls back to `meanAxis` only when gate weights are absent from checkpoint
    - Loader reads `compress_gate_weight` and `compress_pos_bias` from checkpoint
    - _Gap: V4 CSA compression fidelity_

  - [x] 15.3 DeepSeek V4: upgrade FP8 KV storage to use native mlx_to_fp8/mlx_from_fp8
    - Current code at `deepseek_v4.zig:1055-1080` uses `astype(float16)` as FP8 proxy
    - mlx-c 0.6.0 has `mlx_to_fp8`/`mlx_from_fp8`, Zig bindings added in `ops.zig`
    - Replace `astype(kv_storage_dtype)` with `ops.toFp8()` for non-RoPE KV dimensions
    - Keep `astype(.bfloat16)` for RoPE dimensions
    - Add `fromFp8()` call before attention computation to restore precision
    - _Gap: V4 FP8 KV storage — upgrade from float16 proxy to native FP8_

  - [x] 15.4 DeepSeek V4: add FP4 Lightning Indexer for CSA
    - `LightningIndexer` struct with `selectTopK()`, `gatherBlocks()`, `quantize4bit()`
    - Integrated into `DSV4Attention.forward`; loader reads indexer weights from checkpoint
    - _Gap: V4 CSA sparse selection_

  - [x] 15.5 DeepSeek V4: add Attention Sink
    - `sink_logits` field passed to `fast_mod.scaledDotProductAttention()`
    - Loader reads `attn.sink_logits` from checkpoint
    - _Gap: V4 Attention Sink_

  - [x] 15.6 DeepSeek V4: heterogeneous KV cache per layer type
    - `compress_ratios` parsed from config.json per layer in loader
    - Each `DSV4Attention` gets its own `compress_ratio`
    - `compressKV()` applies different compression ratios (4x CSA, 128x HCA, 1x passthrough)
    - _Gap: V4 heterogeneous KV cache_

  - [x] 15.7 DeepSeek V4: on-disk shared-prefix reuse
    - `PrefixDiskCache` in `src/kvcache/prefix_disk.zig` (702 lines, 8 tests)
    - Handles heterogeneous layer shapes via per-layer keys in safetensors
    - Server wires via `--kv-tier ssd` + `--kv-cold-dir <path>`
    - _Gap: V4 "heterogeneous KV cache with on-disk storage for shared-prefix reuse"_

  - [x]* 15.8 TurboQuant KV cache quantization (optional, Phase 4+)
    - Implement Lloyd-Max codebook precomputation for b=1,2,3,4
    - Implement random rotation (QR decomposition via mlx linalg)
    - Implement scalar quantize/dequantize with precomputed codebook
    - Implement QJL residual correction for unbiased inner products
    - Add `--kv-quant simple|turbo` CLI option (default: simple)
    - _Reference: TurboQuant paper arXiv:2504.19874_

- [x] 16. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 17. Integration wiring gaps (identified from deep audit 2026-04-26)
  - [x] 17.1 Wire prefix caching into PagedKVCache write path
    - Call `registerBlockHash` in `PagedKVCache.updateAndFetchImpl` after writing a full block
    - Compute block hash from token IDs stored in the block
    - This activates the existing `findCachedPrefix` / `hashBlock` code which is currently dead
    - _Gap: Prefix sharing is implemented but never registers hashes during normal operation_

  - [x] 17.2 Wire prompt cache save into server shutdown
    - Current `server.zig:96-97` has TODO comment for save
    - Call `prompt_cache_mod.savePromptCache` in `ModelState.deinit` when `config.prompt_cache_file` is set
    - Load path at L285-293 already works
    - _Gap: Prompt cache load works but save is a TODO_

  - [x]* 17.3 Wire PrefixDiskCache into server (optional)
    - Import `prefix_disk` module in `server.zig`
    - Create `PrefixDiskCache` during model load when `kv_cold_dir` is configured
    - Call `restorePrefix` before generation to skip redundant prefill
    - Call `saveBlock` after generation to persist computed prefix blocks
    - _Gap: prefix_disk.zig (702 lines, 8 tests) is complete but never imported by server_

- [x] 18. Final verification — Build + all tests pass

- [x] 19. Concurrent request handling via std.Io.async + Scheduler-driven engine loop
  - [x] 19.1 Replace serial request loop with io.async per connection
    - Wrap `handleRequest` call in `io.async(handleConnection, .{...})` in the main loop
    - Each connection handled concurrently via Zig 0.16.0 std.Io (GCD on macOS)
    - Connection close moves into `handleConnection` (not deferred in main loop)
    - _Gap: server.zig main loop is serial — accept blocks until request completes_

  - [x] 19.2 Add request completion notification to scheduler Request
    - Add `done: bool` and `result: ?GenerationResult` fields to `scheduler.Request`
    - Add `waitForCompletion(io: std.Io)` method that polls `done` with `io.sleep` backoff
    - Add `markComplete(result)` method called by engine loop after postprocess
    - _Gap: No mechanism for handleConnection to wait for scheduler to finish a request_

  - [x] 19.3 Implement engine loop as io.async background task
    - Start `engineLoop` via `io.async` after scheduler initialization in `start()`
    - Engine loop: `schedule() → buildBatch() → forward() → postprocess()` cycle
    - Sleep 1ms when no requests pending to avoid busy-wait
    - Stop when `state.running` flag is set to false
    - _Gap: Scheduler is created but never drives the forward pass_

  - [x] 19.4 Wire handleConnection to submit requests to scheduler
    - Parse chat completion request → create `scheduler.Request`
    - Call `scheduler.addRequest()` to enqueue
    - Call `request.waitForCompletion(io)` to block this connection's async fiber
    - Write response (SSE or JSON) from completed result
    - _Gap: handleRequest does generation inline instead of via scheduler_

- [x] 20. Final verification — Build + all tests pass

- [x] 21. Quantized weight loading support for mlx-lm format models
  - [x] 21.1 Detect quantized weights in LLaMA loader
    - In `llama_loader.zig`, check if weight names have `.scales`/`.biases` suffixes
    - Group (weight, scales, biases) into QuantizedWeight structs from `quantize.zig`
    - Store both regular Array weights and QuantizedWeight in the model
  - [x] 21.2 Add quantized linear forward path in LLaMA model
    - In `llama.zig` Attention and MLP, detect if weight is quantized
    - Use `quantize.quantizedMatmul()` instead of `ops.matmul()` for quantized weights
    - Fall back to regular matmul for non-quantized weights
  - [x] 21.3 Support multi-shard safetensors loading
    - Load all `weights.XX.safetensors` or `model-XXXXX-of-XXXXX.safetensors` files
    - Merge weights from all shards into a single HashMap
  - [x]* 21.4 End-to-end test with TinyLlama-1.1B-Chat-v1.0-4bit (optional)
    - Load model from ~/models/TinyLlama-1.1B-Chat-v1.0-4bit
    - Run inference with a simple prompt
    - Verify output is non-empty and reasonable

- [x] 22. Final verification — Build + all tests pass

- [x] 23. Complete quantization mode parity with MLX 0.31.1
  - [x] 23.1 Add nvfp4 and mxfp8 to QuantMode enum in quantize.zig
    - Add `nvfp4` (NVIDIA FP4, group_size=16, bits=4) and `mxfp8` (Microscaling FP8, group_size=32, bits=8)
    - Add validation rules for each mode
    - Add convenience constructors: `QuantConfig.nvfp4()`, `QuantConfig.mxfp8()`
  - [x] 23.2 Upgrade LightningIndexer to use real FP4 quantization
    - Replace `quantize4bit()` INT4 simulation with `mlx_quantize(mode="mxfp4")`
    - Use `mlx_dequantize(mode="mxfp4")` for restoration
    - This gives true E2M1 FP4 precision instead of INT4 approximation

- [x] 24. Final verification — Build + all tests pass

- [x] 25. Fix quantized weight memory leak and per-weight mixed quantization
  - [x] 25.1 Fix QuantizedWeight original_shape leak in LLaMA model deinit
    - `LlamaAttention.deinit` and `LlamaMLP.deinit` must free `?QuantizedWeight` fields
    - Call `qw.deinit(allocator)` for each non-null quantized weight
    - Same for `lm_head_quant` in `LlamaModel.deinit`
  - [x] 25.2 Support per-weight mixed quantization mode in V4 loader
    - Parse `quantization` / `quantization_config` from V4 config.json
    - Each weight can have independent `{mode, bits, group_size}` (e.g., expert MLP uses mxfp4, attention uses affine)
    - In `deepseek_v4_loader.zig`, look up per-weight config when creating QuantizedWeight
    - Fall back to global default if no per-weight config exists

- [x] 26. Final verification — Build + all tests pass

---

## Phase 7: User Experience & Ecosystem Parity (from deep-gap-analysis.md)

- [x] 27. P0: Core usability fixes
  - [x] 27.1 Streaming token output in generate loop
    - Print each token as it's generated, not after all tokens complete
    - In `llama.zig` generate(), call tokenizer.decode(token) and print immediately per iteration
    - In server SSE path, send each token as an SSE event
    - _Reference: mlx-lm generate_step yield pattern_
  - [x] 27.2 EOS token stop condition
    - Read `eos_token_id` from config.json
    - In generate loop, check if sampled token == eos_token_id and break early
    - Also check against configurable stop token list
    - _Reference: mlx-lm stop condition logic_
  - [x] 27.3 Chat template for LLaMA/Qwen/Mistral path
    - Parse `chat_template` from tokenizer_config.json (Jinja2 format)
    - Implement basic Jinja2 template rendering (or simplified subset)
    - Apply template to wrap user prompt with system/user/assistant markers
    - Currently only DeepSeek V4 path uses chat template
    - _Reference: mlx-lm chat_templates module_
  - [x] 27.4 Fix generate performance for non-quantized models
    - Profile Qwen2.5-0.5B decode step (~30s per token is too slow)
    - Verify KV cache incremental update works correctly (not re-prefilling)
    - Ensure only last token is passed after first prefill (already fixed but verify)
    - Check if float16 matmul on Metal GPU is being used (not CPU fallback)

- [x] 28. P1: Model architecture expansion
  - [x] 28.1 Implement Gemma loader (currently returns "not yet implemented")
    - Gemma uses LLaMA-like architecture with GeGLU activation and different norm
    - Map Gemma weight names to internal format
    - _Reference: mlx-lm models/gemma.py_
  - [x] 28.2 Add Phi-3/Phi-4 architecture support
    - Register `PhiForCausalLM` / `Phi3ForCausalLM` in model_registry
    - Phi uses partial rotary embedding and different MLP structure
    - _Reference: mlx-lm models/phi3.py_
  - [x]* 28.3 Add Qwen3 architecture support (optional)
    - Register `Qwen3ForCausalLM` (non-VL, non-MoE variant)
    - Likely similar to Qwen2 with minor changes
  - [x]* 28.4 Add GLM-4 architecture support (optional)
    - Register `Glm4ForCausalLM` / `Glm4MoeLiteForCausalLM`
    - GLM uses different attention pattern (prefix LM)

- [x] 29. P1: Real batch forward in engine loop
  - [x] 29.1 Implement batch_builder tensor construction
    - Concatenate multiple request tokens into single tensor with position offsets
    - Build attention mask to prevent cross-request attention
    - _Reference: vLLM continuous batching, oMLX BatchGenerator_
  - [x] 29.2 Wire batch_builder into engine loop
    - Replace per-request forward with batched forward in engineLoop()
    - Split batched output logits back to per-request results
    - _Reference: vLLM scheduler postprocess_

- [x] 30. P1: Token decode integration for LLaMA chat
  - [x] 30.1 Use tokenizer.decode() in LLaMA chat output
    - Currently outputs raw token IDs or garbled text
    - Call BpeTokenizer.decode(generated_ids) to get readable text
    - Handle special tokens (BOS/EOS/PAD) in decode

- [x] 31. P2: OpenAI API completeness
  - [x] 31.1 Add `stop` parameter support
    - Parse `stop` field from chat completion request
    - Check generated text against stop strings after each token
  - [x] 31.2 Add `logprobs` parameter support
    - Return top-k log probabilities per token when requested
  - [x]* 31.3 Add `tool_calls` / function calling support (optional)
    - Parse tool definitions from request
    - Detect tool call patterns in generated text
    - Return structured tool_call response
  - [x]* 31.4 Add Anthropic Messages API compatibility (optional)
    - Support `/v1/messages` endpoint with Anthropic format
    - Map between OpenAI and Anthropic message formats

- [x] 32. P2: Inference quality improvements
  - [x] 32.1 Implement repetition penalty
    - Track generated token frequencies
    - Apply penalty to logits before sampling
    - _Reference: mlx-lm sample_utils.py repetition_penalty_
  - [x] 32.2 Implement min_p sampling
    - Filter tokens below min_p * max_prob threshold
    - _Reference: mlx-lm sample_utils.py_
  - [x]* 32.3 SSE keep-alive during long prefill (optional)
    - Send empty SSE comments as heartbeat during prefill
    - Prevents client read timeout on long prompts
    - _Reference: oMLX Claude Code optimization_

- [x] 33. P2: Advanced speculative decoding
  - [x]* 33.1 EAGLE speculative decoding (optional)
    - Train/load lightweight draft head on top of base model
    - More efficient than n-gram for structured outputs
    - _Reference: vLLM EAGLE implementation_

- [x] 34. P3: Ecosystem tools
  - [x]* 34.1 Custom Metal kernel registration (optional)
    - Bind mlx custom kernel API for user-defined Metal shaders
    - Enable specialized kernels for MoE expert dispatch
    - _Reference: mlx custom_kernel API_
  - [x]* 34.2 Multi-Mac distributed inference via mlx_distributed (optional)
    - Bind mlx-c distributed API (send/recv/all_reduce)
    - Implement tensor parallelism across multiple Macs
    - _Reference: mlx distributed module_
  - [x]* 34.3 Model conversion tool (optional)
    - Convert HuggingFace models to MLX safetensors format
    - Support quantization during conversion
    - _Reference: mlx-lm convert.py_
  - [x]* 34.4 Perplexity evaluation tool (optional)
    - Compute perplexity on a test dataset
    - Validate quantization quality
    - _Reference: mlx-lm evaluate.py_

- [x] 35. Final verification — Build + all tests pass

## Notes

- Phase 0–5 modules (Tasks 1–12) are all implemented as standalone files
- The primary remaining work is integration (Task 13) and gap fixes (Task 15)
- Task 13 is broken into 11 sub-tasks for granular tracking of each integration point
- Task 15 captures gaps identified from Apple Silicon best practices audit and DeepSeek V4 paper analysis
- Each task references specific requirements (R1–R26) or identified gaps for traceability
- Property tests validate the 21 correctness properties defined in the design document
- All code is Zig, targeting Apple Silicon via mlx-c bindings
