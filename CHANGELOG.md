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
