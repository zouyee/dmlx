# Chapter 1: Overall Architecture and Module Distribution

## 1.1 Six-Layer Architecture

dmlx adopts a six-layer architecture (from `.kiro/specs/production-deployment/design.md`):

```
┌─────────────────────────────────────────┐
│  Layer 6: Tooling Layer                 │
│  main.zig (CLI) / benchmark.zig         │
├─────────────────────────────────────────┤
│  Layer 5: Model Layer                   │
│  models/ (LLaMA, DeepSeek V4, Nemotron) │
│  expert_stream.zig (MoE expert streaming)│
├─────────────────────────────────────────┤
│  Layer 4: Memory Layer                  │
│  kvcache/ (6 strategies) / model_pool.zig│
│  memory.zig / prompt_cache.zig          │
├─────────────────────────────────────────┤
│  Layer 3: Service Layer                 │
│  server.zig / scheduler.zig             │
│  batch_builder.zig                      │
├─────────────────────────────────────────┤
│  Layer 2: Inference Engine              │
│  generation.zig / speculative.zig       │
│  guided.zig / model_registry.zig        │
├─────────────────────────────────────────┤
│  Layer 1: Foundation Layer              │
│  c.zig / array.zig / ops.zig + ops/     │
│  fast.zig (fused kernels) / fused.zig   │
└─────────────────────────────────────────┘
```

## 1.2 Module Size Distribution (Top 20)

```
3,091  models/deepseek_v4.zig
2,071  models/deepseek_v4_loader.zig
1,764  main.zig
1,517  server.zig
1,354  ops/nn.zig
1,223  speculative.zig
1,152  kvcache/paged.zig
1,129  guided.zig
1,045  io/safetensors_reader.zig
  912  tokenizer/pre_tokenizer.zig
  872  quantize.zig
  773  qlora.zig
  744  tokenizer/bpe.zig
  725  models/llama.zig
  712  models/minimax.zig
  702  kvcache/prefix_disk.zig
  673  models/expert_cache.zig
  649  models/expert_stream.zig
  640  trainer.zig
  563  prompt_cache.zig
```

## 1.3 Core Hubs (Import Frequency)

| Imported Module | Frequency | Role |
|-----------|------|------|
| `std` | 128 | Standard library |
| `../c.zig` | 60 | C binding layer |
| `../array.zig` | 55 | Array wrapper |
| `../ops.zig` | 48 | Operation entry point |
| `../dtype.zig` | 35 | Type system |

## 1.4 Detailed Layer Dependency Analysis

### Layer 1: C Binding Layer (`src/c.zig`)
- The thinnest wrapper layer, wrapping `mlx-c`'s C API with Zig error handling
- `mlxErrorHandler`: global C error handler, captures C++ exception text into `last_error_buffer[2048]`
- `check(rc)`: unified error checking, auto-clears buffer after consumption
- Type re-exports: `mlx_array`→`Array`, `mlx_dtype`→`Dtype`, `mlx_stream`→`Stream`

### Layer 2: Core Types
- `Array`: idiomatic Zig wrapper, provides `fromHandle`/`fromData`/`fromSlice`/`zeros`/`ones`
- `eval()`: explicitly uses the vectorized version of `mlx_eval` to support cross-device scheduling
- `dataPtr<T>()` / `dataSlice<T>()`: comptime type-safe access
- `dataSliceMut<T>()`: **dangerous method**, bypasses CoW semantics via `@constCast`

### Layer 3: Operations Layer
- `ops.zig`: core entry point, 200+ operations, provides `EagerContext` execution mode
- `fast.zig`: binds MLX fused kernels (rms_norm/rope/sdpa/layer_norm)
- `fused.zig`: `mlx_compile` fused graph wrapper, includes `compiledAdamWStep`

### Layer 4: Automatic Differentiation and Parameter Tree
- `grad.zig`: `valueAndGrad`, `vjp`, `jvp`
- `tree.zig`: recursively traverses nested structs via Zig comptime reflection

### Layer 5: Model Layer
- `llama.zig`: standard LLaMA/Mistral/Qwen/Gemma/Phi (725 lines)
- `deepseek_v4.zig`: project's largest file (3,091 lines), includes MLA + MoE + YARN RoPE + mHC

### Layer 6: Inference Engine
- `ModelVTable`: runtime polymorphic interface
- `generateStep`: single forward + sampling, `ScopedArrayArena` tracks temporary arrays
- `streamGenerateSpeculative`: PLD n-gram speculative decoding
- `streamGenerateEagle`: EAGLE speculative decoding

### Layer 7: Service Layer
- `server.zig` (1,517 lines): OpenAI-compatible HTTP + SSE + tool calling
- `scheduler.zig`: request scheduler
- **Active issue**: `batch_builder.zig` is implemented but not fully integrated into the engine loop
