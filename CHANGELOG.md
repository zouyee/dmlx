# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-21

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

## [0.2.0] - 2026-04-17

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

## [0.1.0] - 2025-01-01

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
