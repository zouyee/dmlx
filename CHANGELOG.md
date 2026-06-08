# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.4] - 2026-05-14

### Added
- **Server Engine V2**: Production-grade HTTP server with continuous batching
  - OpenAI-compatible `/v1/chat/completions` (streaming + non-streaming)
  - Anthropic Messages API `/v1/messages` compatibility
  - True streaming via SSE with per-token delivery
  - Speculative decoding (N-gram drafter + verification)
  - Guided decoding (JSON schema + regex constraints via FSM)
  - Graceful shutdown (SIGTERM/SIGINT + `/shutdown` endpoint)
  - Error isolation (per-request error handling, no server crash)
  - Request logging (duration, token count, tokens/sec)
  - Health endpoint with model info and active request count

- **Expert Stream Mode**: Run 141GB DeepSeek V4 Flash 4-bit on 48GB Mac
  - On-demand expert loading from SSD via mmap + pread
  - LFU (Least Frequently Used) expert cache with configurable budget
  - Expert deduplication across batch tokens (20.8% I/O reduction on prefill)
  - `madvise(WILLNEED)` async prefetch for expert data
  - Skip mmap during backbone loading to reduce virtual memory pressure
  - Configurable cache budget via `--smelt-cache <MB>` flag

- **Continuous Batching Infrastructure**
  - `BatchKVCache` with merge/filter/extend for multi-request batching
  - MPSC atomic request queue (lock-free push, batch drain)
  - Cross-thread token delivery via Darwin ulock (zero-latency wake)
  - Per-request KV cache isolation (no cross-contamination)

- **Block-based KV Cache Manager** (PagedKVCache)
  - Reference-counted blocks with Copy-on-Write
  - Prefix caching via chain-based content hashing
  - LRU eviction with O(1) doubly-linked list

### Performance
- **Cold start (first prompt)**: 61s for 10 tokens (SSD I/O bound, 43 layers × 6 experts)
- **Warm cache (subsequent prompts)**: 2.5s for 5 tokens, 2.03 tok/s
- **Decode speed**: ~100-124ms/token (ReleaseFast)
- **Model loading**: 46s (ReleaseFast) vs 110s (Debug)
- **Memory**: RSS stable at ~15GB backbone + 2GB cache = ~17GB total

### Fixed
- `mlx_eval()` 20s/token regression: set default stream to GPU stream
- Tokenizer segfault: heap-allocate BpeTokenizer (AutoHashMap not memcpy-safe)
- Weight lifetime bug: transfer ownership to VTable adapter
- P4 performance regression: remove 5,160 excessive `eval()` calls per token
- Streaming latency: replace io.sleep polling with Darwin ulock cross-thread wake
- OOM during startup: skip mmap for backbone loading in stream mode
- Cache budget: reduce default from 4096→2048MB to prevent OS kills

## [Unreleased]

### Added
- **Native MLX-free Engine for DeepSeek V4 Flash** (`src/native_engine.zig`, `src/metal_infer/`)
  - Full attention + mHC + MoE pipeline in C/Metal without MLX runtime
  - `--native` flag as new recommended default for DSV4-Flash-4bit
  - Requires pre-packed experts: `python3 scripts/repack_experts.py <model_dir>`
  - Smoke test: `NATIVE=1 bash scripts/dsv4_smoke.sh` → 2/2 PASS
  - Correctness diagnostic skill: `.kiro/steering/native-engine-debug.md`

### Fixed
- **MXFP4 E8M0 scale bias correction** (`src/models/moe_kernel.metal`)
  - Root cause: `exp2(scale - 128)` should be `exp2(scale - 127)` per MLX `fp8_e8m0` spec
  - Impact: routed expert outputs were 7.8× too small, causing completely wrong logits
  - Fix: change bias from 128 → 127 in all MXFP4 Metal kernels
  - Same fix applied to `scripts/verify_mxfp4_gate.py` and `scripts/verify_mxfp4_lut.py`

- **Hash routing per-token fix** (`src/native_engine.zig`)
  - Hash routing (layers 0-2) was using last prompt token's ID for all tokens
  - Fix: pass `token_ids[]` array to `forwardBatch` so each token uses its own expert table row

- **`pipe_mhc_pre_bfloat` kernel fix** (`src/metal_infer/engine.c`)
  - Was loading `mhc_pre_gpu_f16` (f16 truncation) instead of `mhc_pre_gpu` (bf16 truncation)
  - Fix: load correct `mhc_pre_gpu` kernel matching MLX's bf16 precision

- **FFN normed_bf16_direct stale bug** (`src/metal_infer/engine.c`)
  - Routing gate was using attn-norm's output instead of FFN-norm's output
  - Fix: update `normed_bf16_direct` after FFN RMSNorm step


  - Caches pre-filled KV states for repeated prompts (skip prefill on cache hit)
  - Proper LRU eviction via monotonic access counter (replaces naive iteration order)
  - FNV-1a token sequence hashing with collision safety (exact match verification)
  - Hit/miss tracking with `hitRate()` reporting
  - Configurable capacity via `--prefix-cache-entries <n>` CLI flag (default: 16)
  - 6 unit tests covering store/lookup, LRU eviction, hit rate, clear, duplicates

- **KV Cache Clone Interface** (`src/kvcache/interface.zig`)
  - `clone()` method on KVCacheStrategy VTable for deep-copying cache state
  - `supportsClone()` predicate for strategy capability detection
  - Implementations for Standard, Rotating, and DeepSeek V4 cache strategies

- **mlock/munlock Backbone Weights** (`src/models/deepseek_v4_loader.zig`)
  - `mlockBackboneWeights()`: batch-eval + POSIX mlock all backbone tensors (~4GB)
  - `munlockBackboneWeights()`: cleanup unlock on shutdown
  - Prevents OS paging out backbone during expert cache activity
  - Activated via `--mlock-backbone` CLI flag

- **Serve-mode benchmark pipeline** (`scripts/run_benchmark.sh`)
  - Full HTTP API testing (30-token + 100-token generation)
  - Automatic report generation with delta comparison vs previous commit
  - Metrics: prefill, ITL, tok/s, cache hit rate, HTTP latency, RSS, startup time
  - 7-prompt correctness validation via serve API

- **kqueue-based accept loop** (`src/server.zig`)
  - Non-blocking connection accept to mitigate macOS VM pressure delays
  - Fallback to blocking accept if kqueue unavailable

- **Expert cache warmup at startup** (`src/engine/engine_loop.zig`, `src/server.zig`)
  - Run 5 diverse prompts before accepting connections
  - Pre-populates cache with common expert routing paths
  - Reduces first-request cache misses by ~85% (18,630 → 2,709)

### Performance (Prefix Cache)
- **Throughput**: 17.8 → 18.8 tok/s (+6%)
- **Steady-state ITL**: 56.2ms → 53.2ms (+5%)
- **Prefix cache TTFR reduction**: 25-48% on repeated prompts (skip prefill + expert loads)
- **Cache hit rate**: 23.7% (71,217 hits / 229,740 misses)
- **Tests**: 401 total (400 passed, 1 skipped), 7/7 E2E pass

### Changed
- **Expert cache default: 4GB → 10GB** (`src/models/expert_cache.zig`)
  - Safe on 48GB Mac (total RSS ~15GB)
  - Improves cache hit rate for longer sessions

### Performance
- **Prefill**: 247ms → 216ms (+12%)
- **Serve-mode tok/s**: 7.1 (cold) → 9.1 (warm cache)
- **100-token server-side**: 10.98s
- **Cache hit rate**: 42.3% (10GB cache, stream mode)
- **Startup time**: 48s (including warmup)
- **7/7 correctness**: All prompts pass in serve mode

### Performance
- **44% end-to-end speedup**: 7-prompt test suite reduced from 2400s to 1340s
- Remove hot-path Layer forward debug prints (86 write syscalls/token)
- Remove MoE diagnostic sync block (4× forced eval + 2× GPU→CPU copy per token)
- Remove prefill logits diagnostic (128K vocab iteration ×2)
- Replace `std.sort.insertion` with `std.mem.sort` in model loader (O(n²) → O(n log n))
- Add `POSIX_MADV_RANDOM` mmap advisory for MoE expert streaming (fixes page cache thrashing)

### Fixed
- **DeepSeek V4 chat template special tokens**: Corrected special token format from
  full-width characters (`<｜begin▁of▁sentence｜>`) to half-width ASCII
  (`<|begin_of_sentence|>`). This fixes garbled output caused by tokenizer
  splitting special tokens into sub-tokens. Added prompt validation to detect
  formatting errors early. (Issue: BOS token ID should be 100000, not split tokens)
- **DeepSeek V4 prompt formatting**: Added proper spacing and newlines in chat
  template (`<|User|>: {content}\n\n` instead of `<|User|>{content}`). Matches
  official DeepSeek V4 format specification.

### Added
- Comprehensive troubleshooting guide for DeepSeek V4 (`docs/en/deepseek-v4/troubleshooting.md`)
- Chat template unit tests to validate special token formatting
- Automatic prompt validation with detailed error messages for debugging
- Performance optimization tracking document (`docs/en/analysis/perf-optimization-log.md`)

## [0.0.3] - 2026-04-21

### Breaking
- Architecture rebuilt from a pure-Zig MLX rewrite (~30K lines) to Zig-native
  bindings over Apple's official `mlx-c` C library (~3.8K lines).
- All backend code (`backend/`, `primitive.zig`, `scheduler.zig`, `graph.zig`)
  has been removed. Computation is now delegated to MLX's unified Metal/CPU
  runtime via `mlx-c`.

### Added
- 200+ operations across dedicated sub-modules:
  `comparison`, `math`, `shape`, `reduce`, `sort`, `creation`, `random`,
  `linalg`, `fft`, `conv`, `fast`
- `EagerContext` for eager execution with default stream management
- Autograd and transforms: `eval`, `asyncEval`, `Closure`, `valueAndGrad`,
  `vjp`, `jvp`, `compile`, `enableCompile`, `disableCompile`, `setCompileMode`
- I/O layer rebuilt on `mlx-c`: `loadSafetensors`, `saveSafetensors`, `load`, `save`
- Retained pure-Zig `.npy` reader/writer

### Removed
- ~25K lines of pure-Zig backend (CPU SIMD/BLAS, Metal wrapper, CUDA scaffold,
  scheduler, graph engine, primitive dispatch)
- Old pure-Zig `safetensors.zig` and `gguf.zig` parsers (replaced by `mlx-c` I/O)

### Fixed
- Zig 0.16.0 compatibility (`DebugAllocator`, removed `refAllDeclsRecursive`,
  `addCSourceFile` API changes)
- Fixed `nn.zig` LSTM/GRU scope bugs caused by Zig 0.16 shadowing rules
- Fixed segfault from uninitialized `mlx_device` before `mlx_get_default_stream`

## [0.0.2] - 2026-04-17

### Added
- 100% MLX C++ core API parity (all operations migrated)
- 100% test suite alignment (all 19 MLX C++ test files have Zig equivalents)
- 369 tests total across 17 test files
- Dedicated test suites: autograd, FFT, linalg, random, einsum, scheduler,
  device, allocator, I/O, compile
- `logical_and`, `logical_or` operations
- `sliceUpdateAdd`, `sliceUpdateProd`, `sliceUpdateMax`, `sliceUpdateMin`
- `gather_mm` (matrix product with matrix-level gather)
- `segmented_mm` (segmented matrix multiply)
- Project documentation: README, LICENSE, CONTRIBUTING, CODE_OF_CONDUCT,
  ACKNOWLEDGMENTS, CHANGELOG, .gitignore

### Fixed
- FFT `shape[axis]` type mismatch (i32 vs usize) in advanced.zig
- PCG random number generator invalid bit shift (>> 75 on u64)
- `randpermutation` comptime array size issue

## [0.0.1] - 2025-01-01

### Added
- Phase 1: Core data structures (Array, Dtype, Device, Stream)
- Phase 2: Lazy graph + autograd (VJP/JVP, grad, vmap, compile)
- Phase 3: CPU backend optimization (BLAS, SIMD, parallel)
- Phase 4: Metal GPU backend (C wrapper, kernel dispatch)
- Phase 5: Full operation coverage
  - 30 unary ops, 20 binary ops, 12 reduction ops
  - Convolution (1D/2D/3D + transpose variants)
  - FFT (fft, ifft, rfft, irfft, fft2, ifft2, fftn, ifftn, fftshift)
  - Linear algebra (norm, det, inv, solve, cholesky, QR, SVD, LU, eig)
  - 15 random distributions
  - 21 activation functions, 10 loss functions, 12 NN layers
  - 10 pooling operations, sparse ops, distance functions
  - Quantization (quantize, dequantize, quantizedMatmul, qqmm, gatherQmm)
  - I/O (safetensors, GGUF, npy)
  - Distributed ops (all_sum, all_gather, send, recv, all_max, all_min)
  - Memory management (active/peak tracking, limits, cache)
  - Scheduler (per-stream task execution, sync events)
