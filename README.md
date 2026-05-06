# dmlx — Run Frontier LLMs on Your Mac

[**How It Works**](#how-it-works-5-layer-memory-optimization) | [**Performance**](#real-world-performance) | [**Scenarios**](#application-scenarios) | [**Quick Start**](#quick-start) | [**Installation**](#installation) | [**Architecture**](#architecture)

> **Run a 284B-parameter MoE model (13B activated per token) on a 48GB MacBook Pro. No cloud. No GPU cluster. Just your laptop.**
>
> dmlx combines Apple's MLX Metal backend with five layers of memory optimization to make the
> impossible possible — and wraps it in a single static Zig binary with an OpenAI-compatible API.

---

## Why dmlx?

The DeepSeek V4 Flash model is **284 billion parameters** across 256 routed experts (with 6 activated
per token, plus 1 shared expert). At BF16 precision, its weights require **~568 GB** — more than
10× the memory of a 48GB MacBook Pro. Even at 4-bit quantization (~40 GB), it won't fit without
aggressive memory management.

dmlx solves this through **five complementary layers of memory optimization**, plus a production-grade
inference stack that makes frontier LLMs practical on consumer hardware.

| Capability | mlx-lm (Python) | dmlx (Zig) |
|-----------|-----------------|-----------------|
| DeepSeek V4 on 48GB Mac | ❌ OOM (~40GB weights needed) | ✅ ~6GB via SMELT 15% |
| DeepSeek V4 on 96GB+ Mac | ✅ (if RAM sufficient) | ✅ |
| KV cache strategies | 1 (fixed) | 6 (runtime-switchable, incl. Tiered SSD) |
| Max context on 48GB | Limited by RAM | 128K+ tokens (RAM+SSD) |
| Deployment | Python + pip + venv (~500MB+) | Single static binary (~5-15MB) |
| Deterministic latency | ❌ Python GC (10–100ms pauses) | ✅ Zero GC (sub-ms) |
| Model architectures | 50+ | 8 (LLaMA, DeepSeek V4, Qwen2/3, Mistral, Gemma, GLM-4, Phi) |
| iOS/macOS embedding | ❌ No Python runtime on iOS | ✅ C ABI → Swift |

---

## How It Works: 5-Layer Memory Optimization

### Layer 1: MoE Expert Streaming → 138 GB → 10 GB

DeepSeek V4 Flash activates only **top-6 of 256 routed experts** (plus 1 shared expert) per token.
dmlx exploits this sparsity via `expert_stream.zig` (649 lines):

- **On-demand loading**: Only active expert weights are loaded via `PartialTensorReader`
- **LRU expert cache**: Frequently-used experts stay resident; cold ones are evicted
- **Layer prefetching**: Next layer's experts are fetched during current layer computation
- **Result**: Peak memory drops from ~138 GB (all experts) to ~10 GB (streaming only active ones)

```
Source: src/models/expert_stream.zig | src/models/moe_router.zig
```

### Layer 2: 4-bit Quantization + SMELT → 40 GB → 6 GB

Six quantization formats (INT4/8, MXFP4, FP8 E4M3, TurboQuant) with the **SMELT system** for
partial expert loading:

| Mode | Experts Loaded | Memory | Latency Impact |
|------|---------------|--------|----------------|
| Full 4-bit | 256 (100%) | ~40 GB | Baseline |
| SMELT 30% | ~77 | ~12 GB | +10–15% |
| SMELT 15% | ~38 | **~6 GB** | +10–15% |

SMELT auto-detects how many experts actually exist in the model files, loads only those, and
applies a routing bias (`smelt_mask`) to prevent selecting unloaded experts. Missing experts
fall back to streaming on-demand.

```
Source: docs/en/technical/4bit-smelt.md | docs/en/technical/smelt-flow.md
```

### Layer 3: CSA + HCA Hybrid Attention (KV Cache Compression)

DeepSeek V4 introduces a **hybrid attention architecture** combining Compressed Sparse Attention
(CSA) and Heavily Compressed Attention (HCA) in an interleaved configuration:

- **CSA** (compression rate m=4): Compresses every 4 KV entries into 1, then applies sparse
  top-k selection (k=512) via a lightning indexer
- **HCA** (compression rate m'=128): Compresses every 128 KV entries into 1 for extreme reduction
- **FP8 storage**: Non-RoPE KV dimensions stored as FP8 (E4M3), further halving memory
- **Result**: KV cache is ~9.5× smaller than DeepSeek-V3.2 at 1M context

```
Source: src/models/deepseek_v4.zig (CSA+HCA implementation, 3,091 lines)
```

### Layer 4: Six-Level KV Cache Strategy System

Runtime-switchable strategies for any memory budget:

| Strategy | Memory Profile | Best For |
|----------|---------------|----------|
| **Standard** | Full KV buffer | Short sequences, single request |
| **Rotating** | Fixed window ring buffer | Ultra-long sequences (avoid OOM) |
| **Quantized** | 4/8/16-bit KV compression | Memory-constrained scenarios |
| **Paged** ⭐ | 32-token pages + CoW | Continuous batching (production default) |
| **PagedQuantized** | Paged + Quantized combined | Extreme memory optimization |
| **Tiered** | RAM hot + SSD cold + LRU | Ultra-long context (128K+) + multi-model |

```
Source: src/kvcache/paged.zig (1,152 lines) | src/kvcache/tiered.zig
```

### Layer 5: Zero-Copy Model Loading (TTFT Optimization)

Eliminates unnecessary memory copies during model loading:

| Phase | What | Before | After |
|-------|------|--------|-------|
| P0 | Binary index cache | 67s parsing 33 shards | ~1s mmap read |
| P2 | Zero-copy weight loading | ~7 GB memcpy | 0 (direct mmap) |
| P3 | Batched shard I/O | Random reads | Sequential OS readahead |

Combined: model loading from **~137s → ~41–46s** (66–70% reduction).

```
Source: docs/en/technical/ttft-optimization.md
```

---

## Real-World Performance

**Hardware**: Apple M4 Pro, 48 GB unified memory
**Model**: DeepSeek-V4-Flash-4bit, 33 shards (~40 GB 4-bit quantized)
**Mode**: SMELT + stream, ExpertCache 4GB, temperature=0
**Commit**: `7e72a7` (2026-05-05) — [benchmark log](docs/en/analysis/performance-benchmark.md)

| Metric | Value |
|--------|-------|
| Prefill (token 1) | 370.5 ms |
| Steady-state (tokens 3-10 avg) | 82.2 ms |
| Throughput (steady-state) | **~12.2 tok/s** |
| Memory (SMELT 15%) | ~6 GB weights + KV cache |
| Max context (Paged + Tiered) | 128K+ tokens |
| 7-prompt correctness | **7/7 PASS, 0 FAIL** |

| Benchmark Trend | Initial (`a024bee`) | Current (`7e72a7`) | Improvement |
|-----------------|---------------------|---------------------|-------------|
| Prefill | 716ms | **370.5ms** | +48% |
| Steady-state avg | ~125ms | **82.2ms** | +34% |
| Throughput | ~8 tok/s | **~12.2 tok/s** | **+52%** |

> **Why this matters**: mlx-lm cannot run DeepSeek V4 on 48GB Macs at all — it requires loading
> all ~40GB of 4-bit weights simultaneously, causing OOM. dmlx's SMELT system runs the same
> model with ~6GB of weights. Raw Metal compute is similar (same `mlx-c` backend), so on larger
> Macs (96GB+) where mlx-lm can fit, performance is comparable.
>
> **dmlx's advantage is not raw speed — it's that the model runs at all on small Macs.**

[→ Performance Dashboard](https://dmlx.ai/) | [→ Benchmarking Guide](docs/en/technical/benchmarks.md) | [→ Competitive Analysis](docs/en/analysis/competitive-advantages.md)

---

## Application Scenarios

### 1. Local LLM Inference

Run GPT-4-class intelligence entirely on-device. All computation happens on your Mac's Metal GPU
via Apple's unified memory architecture — zero network egress, no API keys, no per-token pricing.

### 2. Privacy-First Applications

HIPAA, GDPR, and enterprise compliance: all data stays on-device. Air-gapped deployment supported.
Your models, your data, your hardware — no third-party processors.

### 3. Edge Deployment — Mac mini Inference Server

Deploy a Mac mini as a private team inference server with OpenAI-compatible API:

```bash
dmlx server --model ~/models/deepseek-v4-flash-4bit \
  --port 8080 --kv-strategy paged

# Any OpenAI client works as a drop-in replacement
curl http://mac-mini:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4","messages":[{"role":"user","content":"Hello"}]}'
```

### 4. Offline & Censored-Region Access

Download once, run anywhere — no internet required. Full model capability without content filtering.
Works on trains, planes, remote locations, and secure facilities.

### 5. Development & Testing Without GPU Clusters

Iterate on LLM applications locally — no A100 reservations at $3+/hr. Develop locally, validate,
then optionally scale up to cloud.

### 6. Research & Experimentation

Full-stack access for ML research: modify MoE routing (`moe_router.zig`), swap KV cache strategies
at runtime, test quantization formats (INT4, MXFP4, TurboQuant), compare speculative decoding
strategies (PLD vs EAGLE).

[→ Detailed Scenario Guide](docs/en/scenarios/README.md) | [→ DeepSeek MoE Deep Dive](docs/en/deepseek-moe/README.md)

---

## Features

### LLM Inference Engine

- **9 model architectures**: DeepSeek V4, LLaMA, Mistral, Qwen2, Qwen3, Gemma, GLM-4, Phi, Phi-3
- **OpenAI-compatible HTTP server** with SSE streaming and continuous batching
- **Speculative decoding**: PLD (Prompt Lookup Decoding) + EAGLE draft head for faster generation
- **Guided decoding**: JSON Schema / Regex FSM for constrained, structured output
- **6-level KV cache**: Standard, Rotating, Quantized, Paged (CoW), PagedQuantized, Tiered (RAM+SSD)
- **Quantization**: Affine INT4/INT8, MXFP4, FP8 (E4M3), TurboQuant (Lloyd-Max + QJL)
- **Expert streaming**: SMELT partial loading + on-demand stream mode for MoE models
- **Training**: QLoRA fine-tuning, AdamW optimizer with compiled fusion, SFT Trainer
- **Model I/O**: Safetensors, GGUF, NumPy `.npy` loading and saving
- **Custom Metal kernels**: TileKernels reproduction — fused Sinkhorn normalization, fused SwitchGLU with gather_mm

### Core MLX Library

- **200+ operations**: Comparison, math, shape manipulation, reductions, sorting, creation,
  random, linear algebra, FFT, convolution, fast custom ops (layer norm, RMS norm, RoPE, SDPA)
- **Autograd**: `grad`, `value_and_grad`, `vjp`, `jvp`, graph `compile`
- **NN layers**: Linear, LSTM, GRU, MultiHeadAttention, 21 activations, 10 loss functions
- **Zig-native API**: Type-safe wrappers with idiomatic Zig patterns — `Array`, `Dtype`, `Device`, `Stream`, `EagerContext`
- **Official MLX backend**: Full access to Metal GPU, CPU (Accelerate/BLAS), unified memory

---

## Quick Start

### One-Command Chat

```bash
# Start chatting with DeepSeek V4 on your Mac
dmlx chat --model ~/models/DeepSeek-V4-Flash-4bit \
  --prompt "Explain quantum computing in one sentence" \
  --smelt --smelt-experts 0.15
```

### Core Library Usage

```zig
const std = @import("std");
const mlx = @import("dmlx");

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

---

## Installation

### Requirements

- Zig **0.16.0** or later
- macOS with Apple Silicon (primary target)
- `mlx-c` installed via Homebrew:
  ```bash
  brew install mlx-c
  ```

### As a Zig Dependency

Add to your `build.zig.zon`:

```zig
.dependencies = .{
    .dmlx = .{
        .url = "https://github.com/zouyee/dmlx/archive/refs/tags/v0.3.0.tar.gz",
        .hash = "...",
    },
},
```

### Build from Source

```bash
git clone https://github.com/zouyee/dmlx.git
cd dmlx
zig build          # Build library + example
zig build test     # Run tests
zig build run      # Run demo
```

---

## Architecture

```
dmlx/
├── build.zig              # Build configuration (links mlx-c + Metal/Accelerate)
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
│   ├── models/            # LLM architectures
│   │   ├── deepseek_v4.zig        # DeepSeek V4 (3,091 lines)
│   │   ├── deepseek_v4_loader.zig # Weight loading + SMELT config
│   │   ├── expert_stream.zig      # MoE on-demand streaming (649 lines)
│   │   ├── expert_cache.zig       # LRU expert cache
│   │   ├── moe_router.zig         # Top-k MoE routing (629 lines)
│   │   ├── llama.zig              # LLaMA architecture
│   │   └── llama_loader.zig       # LLaMA weight loading
│   ├── kvcache/            # KV cache strategies
│   │   ├── standard.zig    # Full buffer
│   │   ├── rotating.zig    # Ring buffer for long sequences
│   │   ├── quantized.zig   # 4/8/16-bit compressed KV
│   │   ├── paged.zig       # 32-token pages + CoW (1,152 lines)
│   │   ├── tiered.zig      # RAM hot + SSD cold + LRU
│   │   ├── turboquant.zig  # TurboQuant: Lloyd-Max + QJL
│   │   └── prefix_disk.zig # On-disk prefix caching
│   ├── speculative.zig     # PLD + EAGLE speculative decoding
│   ├── guided.zig          # JSON Schema / Regex guided decoding
│   ├── server.zig          # OpenAI-compatible HTTP server
│   ├── batch_builder.zig   # Continuous batching builder
│   ├── memory.zig          # Auto memory budgeting + enforcement
│   ├── generation.zig      # Three-layer generation API
│   ├── io/
│   │   ├── mlx_io.zig      # Safetensors / GGUF via mlx-c
│   │   └── npy.zig         # NumPy .npy read/write
│   ├── eval.zig            # eval / async_eval
│   ├── closure.zig         # Closure wrapper for transforms
│   ├── grad.zig            # grad, value_and_grad, vjp, jvp
│   ├── compile.zig         # compile, enable_compile, compile modes
│   └── tests/              # Comprehensive test suite (350+ tests)
├── docs/                   # Bilingual EN/ZH documentation
└── README.md
```

---

## Documentation

Comprehensive documentation is available in [docs/](docs/index.md) (bilingual EN/ZH):

| Section | Description |
|---------|-------------|
| [DeepSeek MoE Deep Dive](docs/en/deepseek-moe/README.md) | How 284B runs on 48GB — 5-layer optimization |
| [Application Scenarios](docs/en/scenarios/README.md) | 6 real-world use cases |
| [Competitive Analysis](docs/en/analysis/competitive-advantages.md) | dmlx vs mlx-lm, llama.cpp, LM Studio — verified benchmarks |
| [User Guide](docs/en/user-guide/) | Quick fixes and troubleshooting |
| [Technical Docs](docs/en/technical/) | Benchmarks, TTFT, SMELT, roadmap |
| [Analysis Reports](analysis-report/) | Comprehensive project analysis (52K+ lines) |
| [Contributing Guide](CONTRIBUTING.md) | Developer guidelines |

→ [Documentation Index](docs/index.md)

---

## Platform Support

| Platform | Status | Backend |
|----------|--------|---------|
| macOS Apple Silicon | ✅ Primary | Metal + CPU (Accelerate) |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Acknowledgments

dmlx is inspired by and built on [Apple's MLX](https://github.com/ml-explore/mlx)
and the official `mlx-c` C bindings. Custom Metal kernels reproduce optimizations from
[DeepSeek's TileKernels](https://github.com/deepseek-ai/tilekernels), adapted from
CUDA to Apple Silicon.

See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for details.

---

## License

dmlx is released under the [MIT License](LICENSE).
