# MLX-Z: Zig Bindings for Apple MLX

Zig wrappers for [Apple's MLX C API](https://github.com/ml-explore/mlx-c) (`mlx-c`), providing type-safe, idiomatic Zig access to the Metal GPU-accelerated array framework.

## Features

- **200+ MLX operations** — math, linear algebra, FFT, convolution, sorting, random, and more
- **Autograd** — `grad`, `value_and_grad`, `vjp`, `jvp`
- **Graph compilation** — `compile`, `enable_compile`
- **NN layers** — Linear, LSTM, GRU, MultiHeadAttention, Embedding, Dropout
- **Activations & Loss** — 21 activation functions, 10 loss functions
- **I/O** — Safetensors/GGUF load/save, NumPy `.npy` read/write
- **Quantization** — Affine INT4/INT8, MXFP4, FP8 (E4M3)
- **Optimizers** — AdamW with compiled fusion

## Requirements

- Zig **0.16.0** or later
- macOS with Apple Silicon (primary target)
- `mlx-c` installed via Homebrew: `brew install mlx-c`

## Installation

Add to your project's `build.zig.zon`:

```zig
.dependencies = .{
    .mlx_z = .{
        .path = "path/to/mlx-zig",
    },
},
```

Then in `build.zig`:

```zig
const mlx_z_dep = b.dependency("mlx_z", .{
    .target = target,
    .optimize = optimize,
});
const mlx_z_module = mlx_z_dep.module("mlx");
exe.root_module.addImport("mlx", mlx_z_module);
```

## Quick Start

```zig
const std = @import("std");
const mlx = @import("mlx");

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
}
```

## Build

```bash
zig build          # Build library
zig build test     # Run tests
```

## mlx-c Discovery

mlx-zig discovers `mlx-c` via (in priority order):
1. `-Dmlx_prefix=/path/to/mlx-c` build flag
2. `MLX_C_PREFIX` environment variable
3. `pkg-config --variable=prefix mlxc`
4. `/opt/homebrew` fallback

## mlx-c API Coverage

| mlx-c 0.6.0 Module | Zig Wrapper | Coverage | Notes |
|---------------------|-------------|----------|-------|
| `array.h` | `src/array.zig` | ✅ 80% | creation, data, eval, shape/strides (missing: scalar item accessors, `data_managed`, `set_data`) |
| `ops.h` | `src/ops.zig` `src/ops/*.zig` | ✅ 85% | 265+ ops: math, comparison, shape, reduce, sort, creation, random, linalg, fft, conv, nn, activations, loss, fused, batch (missing: ~25 fringe ops) |
| `transforms.h` | `src/grad.zig` | ✅ 90% | `grad`, `value_and_grad`, `vjp`, `jvp`, `eval`, `checkpoint`, `custom_vjp` |
| `compile.h` | `src/compile.zig` | ✅ 80% | `compile`, `enable/disable`, `mode` (missing: `detail_compile_*`) |
| `memory.h` | `src/memory.zig` | ✅ 100% | `clear_cache`, `get/set_memory_limit`, `peak_memory`, `cache_limit`, `wired_limit` |
| `metal.h` | `src/metal.zig` | ✅ 100% | `is_available`, `start/stop_capture` |
| `device.h` | `src/device.zig` | ✅ 85% | `Device`, `DeviceType`, `DeviceInfo`, `device_count`, `is_available` |
| `stream.h` | `src/device.zig` | ✅ 80% | `Stream` wrapper |
| `fast.h` | `src/ops/fast.zig` `src/ops/custom_kernel.zig` | ✅ 90% | LayerNorm, RMSNorm, RoPE, SDPA, Metal custom kernels |
| `io.h` | `src/io/mlx_io.zig` | ✅ 80% | Safetensors load/save, GGUF load/save, NPY |
| `random.h` | `src/ops/random.zig` | ✅ 100% | all 19 random functions |
| `fft.h` / `linalg.h` | `src/ops/fft.zig` `src/ops/linalg.zig` | ✅ 100% | all FFT + all linalg operations |
| `distributed.h` | — | ⛔ 0% | multi-GPU collectives (macOS single-device, deferred) |
| `graph_utils.h` | — | ⛔ 0% | `node_namer`, `export_to_dot` (debug only) |
| `export.h` | — | ⛔ 0% | function serialization (deferred) |

### Version Matrix

| mlx-zig | mlx-c (minimum) | mlx (upstream) | Zig | Notes |
|---------|-----------------|----------------|-----|-------|
| v0.0.1 | **0.6.0** | 0.23.x | 0.16.0 | Initial coverage: ~85% of C API |

## License

MIT
