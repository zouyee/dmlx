# MLX-Zig

[**Quick Start**](#quick-start) | [**Installation**](#installation) | [**Architecture**](#architecture)

MLX-Zig provides Zig-native bindings for [Apple's MLX](https://github.com/ml-explore/mlx)
machine learning framework, built on top of the official `mlx-c` C library.

## Features

- **Official MLX backend**: Full access to Metal GPU, CPU (Accelerate/BLAS), unified memory,
  and Steel GEMM via Apple's `mlx-c` — no reimplemented kernels.

- **Zig-native API**: Type-safe wrappers around `mlx-c` with idiomatic Zig patterns:
  `Array`, `Dtype`, `Device`, `Stream`, and `EagerContext`.

- **200+ operations**: Comprehensive coverage across comparison, math, shape manipulation,
  reductions, sorting, creation, random, linear algebra, FFT, convolution, and fast custom ops
  (layer norm, RMS norm, RoPE, scaled dot-product attention).

- **Automatic differentiation**: `grad`, `value_and_grad`, `vjp`, `jvp`, and graph `compile`
  via `mlx-c` transforms.

- **Model I/O**: Load and save Safetensors, GGUF, and NumPy `.npy` formats.

- **Neural network layers**: Built-in Linear, LSTM, GRU, MultiHeadAttention, 21 activations,
  and 10 loss functions — all backed by MLX.

- **LLM inference engine**: Production-grade inference with LLaMA and DeepSeek V4 support,
  OpenAI-compatible HTTP server with SSE streaming, continuous batching, speculative decoding,
  and guided decoding (JSON schema / regex).
  
  > **Note:** DeepSeek V4 chat template was fixed in commit `a18bc24` to use correct ASCII
  > special tokens. If you experience garbled output, see [`docs/QUICKFIX-DEEPSEEK-V4.md`](docs/QUICKFIX-DEEPSEEK-V4.md)
  > for verification steps.

- **KV cache**: 6 strategies (Standard, Rotating, Quantized 4/8-bit, Paged with CoW,
  Paged+Quantized, Tiered RAM+SSD) with prefix caching and on-disk shared-prefix reuse.

- **Quantization**: Affine INT4/INT8, MXFP4 (Microscaling FP4), FP8 (E4M3), and
  TurboQuant (near-optimal Lloyd-Max + QJL for unbiased inner products).

- **Quantized model loading**: Direct loading of mlx-lm 4-bit/8-bit quantized models
  with fused `quantizedMatmul` for inference — no dequantization overhead.

- **Training**: QLoRA fine-tuning, AdamW optimizer with compiled fusion, SFT Trainer.

## Quick Start

```zig
const std = @import("std");
const mlx = @import("mlx-zig");

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const b_data = [_]f32{ 5, 6, 7, 8 };
    const a = try mlx.Array.fromData(allocator, f32, &a_data, &[_]i32{ 2, 2 });
    defer a.deinit();
    const b = try mlx.Array.fromData(allocator, f32, &b_data, &[_]i32{ 2, 2 });
    defer b.deinit();

    const ctx = mlx.EagerContext.init(allocator);
    const c = try mlx.ops.matmul(ctx, a, b);
    defer c.deinit();

    std.debug.print("A @ B = {any}\n", .{try c.dataSlice(f32)});

    const logits = try mlx.Array.fromSlice(allocator, f32, &[_]f32{ 2.0, 1.0, 0.1 });
    defer logits.deinit();
    const probs = try mlx.ops.softmax(ctx, logits);
    defer probs.deinit();
    std.debug.print("softmax = {any}\n", .{try probs.dataSlice(f32)});
}
```

## Installation

### Requirements

- Zig **0.16.0** or later
- macOS with Apple Silicon (primary target)
- `mlx-c` installed via Homebrew:
  ```bash
  brew install mlx-c
  ```

### As a Zig dependency

Add to your `build.zig.zon`:

```zig
.dependencies = .{
    .mlx_zig = .{
        .url = "https://github.com/zouyee/mlx-zig/archive/refs/tags/v0.3.0.tar.gz",
        .hash = "...",
    },
},
```

### Build from source

```bash
git clone https://github.com/zouyee/mlx-zig.git
cd mlx-zig
zig build          # Build library + example
zig build test     # Run tests
zig build run      # Run demo
```

## Architecture

```
mlx-zig/
├── build.zig              # Build configuration (links mlx-c + frameworks)
├── build.zig.zon          # Package manifest
├── src/
│   ├── c.zig              # @cImport of mlx-c headers
│   ├── array.zig          # Array wrapper (creation, eval, data access)
│   ├── dtype.zig          # Dtype enum + comptime mapping
│   ├── device.zig         # Device / Stream
│   ├── ops.zig            # Core ops (unary, binary, matmul, reductions)
│   ├── ops/
│   │   ├── comparison.zig # equal, greater, all, any, isclose, ...
│   │   ├── math.zig       # floor, clip, logaddexp, erf, ...
│   │   ├── shape.zig      # reshape, slice, transpose, take, ...
│   │   ├── reduce.zig     # sum, mean, argmax, cumsum, topk, ...
│   │   ├── sort.zig       # sort, argsort, partition, ...
│   │   ├── creation.zig   # zeros, ones, eye, arange, linspace, ...
│   │   ├── random.zig     # normal, uniform, categorical, ...
│   │   ├── linalg.zig     # cholesky, inv, svd, qr, solve, ...
│   │   ├── fft.zig        # fft, rfft, fftshift, ...
│   │   ├── conv.zig       # conv1d/2d/3d, conv_transpose, ...
│   │   ├── fast.zig       # layer_norm, rms_norm, rope, sdpa
│   │   ├── nn.zig         # Linear, LSTM, GRU, MultiHeadAttention
│   │   ├── activations.zig # 21 activation functions
│   │   └── loss.zig       # 10 loss functions
│   ├── io/
│   │   ├── mlx_io.zig     # Safetensors / GGUF via mlx-c
│   │   └── npy.zig        # NumPy .npy read/write
│   ├── eval.zig           # eval / async_eval
│   ├── closure.zig        # Closure wrapper for transforms
│   ├── grad.zig           # grad, value_and_grad, vjp, jvp
│   ├── compile.zig        # compile, enable_compile, compile modes
│   └── tests/
│       └── core_tests.zig # Core test suite
└── README.md              # This file
```

## Modules

### Core types

- `mlx.Array` — multi-dimensional array backed by `mlx_array`
- `mlx.Dtype` — type enum (`float32`, `int32`, `bool`, ...)
- `mlx.Device` / `mlx.Stream` — execution device and stream
- `mlx.EagerContext` — execution context for eager ops

### Operations

All operations are organized into sub-modules:

```zig
const mlx = @import("mlx-zig");
const ctx = mlx.EagerContext.init(allocator);

// Element-wise
const s = try mlx.math.sin(ctx, a);
const eq = try mlx.comparison.equal(ctx, a, b);

// Shape
const r = try mlx.shape.reshape(ctx, a, &.{ 2, 5 });

// Reduce
const total = try mlx.reduce.sumAxis(ctx, a, 0, false);

// Linear algebra
const inv_a = try mlx.linalg.inv(ctx, a);

// Random
const noise = try mlx.random.normal(ctx, &.{ 3, 3 }, .float32, 0.0, 1.0, null);

// FFT
const spectrum = try mlx.fft.fft(ctx, a, 256, -1);
```

### Autograd

```zig
const closure = try mlx.closure.Closure.init(myForwardFn, allocator);
defer closure.deinit();

// Value and gradient
const vg = try mlx.grad.valueAndGrad(closure, &.{0});
defer vg.deinit();
const result = try vg.apply(&.{x}, allocator);
defer allocator.free(result.value);
defer allocator.free(result.grad);

// VJP
const vjp_result = try mlx.grad.vjp(closure, &.{x}, &.{dy}, allocator);
defer allocator.free(vjp_result.outputs);
defer allocator.free(vjp_result.grads);
```

### I/O

```zig
// Safetensors
const st = try mlx.io.loadSafetensors(allocator, "model.safetensors");
defer st.deinit(allocator);
const weight = st.weights.get("layer1.weight").?;

// Save
try mlx.io.saveSafetensors(allocator, "out.safetensors", weights, metadata);
```

## Platform Support

| Platform | Status | Backend |
|----------|--------|---------|
| macOS Apple Silicon | ✅ Primary | Metal + CPU (Accelerate) |
| macOS Intel | ⚠️ Experimental | CPU only |
| Linux | ⚠️ Experimental | CPU + CUDA (if available) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

MLX-Zig is inspired by and built on [Apple's MLX](https://github.com/ml-explore/mlx)
and the official `mlx-c` C bindings. See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for details.

## License

MLX-Zig is released under the [MIT License](LICENSE).
