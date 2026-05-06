# DeepSeek V4 MoE on Small Macs — Technical Deep Dive

> **How dmlx runs a ~150GB MoE model on a 48GB MacBook Pro**

---

## The Problem

DeepSeek V4 is a 671B-parameter Mixture-of-Experts model. Even at 4-bit quantization, the full model weighs ~40GB. In FP16, the weights alone exceed 150GB. Running this on a consumer MacBook Pro with 48GB unified memory seems impossible — yet dmlx does it.

## The Solution: Five Layers of Memory Optimization

### Layer 1: MoE Expert Streaming (138GB → 10GB)

DeepSeek V4 uses 256 routed experts + shared experts with top-k routing. Per token, only a small subset (typically top-8) of experts are activated. dmlx's `expert_stream.zig` (649 lines) exploits this sparsity:

- **On-demand loading**: Only active experts are loaded into memory via `PartialTensorReader` (pread-based partial tensor reads)
- **LRU expert cache**: Frequently used experts stay resident; cold experts are evicted
- **Layer prefetching**: The next layer's experts are prefetched during current layer computation
- **Memory reduction**: From ~138GB (all experts loaded) to ~10GB (streaming active experts only)

```
Source: src/models/expert_stream.zig (649 lines)
        src/models/moe_router.zig (top-k routing)
```

### Layer 2: 4-bit Quantization (40GB → 6-12GB with SMELT)

dmlx supports six quantization formats, with the SMELT system enabling partial expert loading:

| Mode | Experts Loaded | Memory | Latency Impact |
|------|---------------|--------|----------------|
| Full 4-bit | 256 (100%) | ~40GB | Baseline |
| SMELT 30% | ~77 | ~12GB | +10-15% |
| SMELT 15% | ~38 | ~6GB | +10-15% |

**How SMELT works:**
- Auto-detects how many experts actually exist in the model files (not all 256 may be present)
- Loads only the weights for present experts
- Applies a routing bias (`smelt_mask`) to prevent the router from selecting unloaded experts
- Falls back to streaming for occasionally-needed missing experts

```
Source: docs/en/technical/4bit-smelt.md
        docs/en/technical/smelt-flow.md
```

### Layer 3: MLA KV Cache Compression (2×heads×dim → 2×latent_dim)

Multi-head Latent Attention (MLA) compresses the KV cache via low-rank projection:

- **Before MLA**: KV cache size = 2 × n_heads × head_dim (huge for long contexts)
- **After MLA**: KV cache size = 2 × latent_dim (dramatically smaller)
- **FP8 storage**: Non-RoPE KV dimensions stored as FP8 (E4M3), further halving memory

```
Source: src/models/deepseek_v4.zig (MLA implementation, 3,091 lines)
```

### Layer 4: Six-Level KV Cache Strategy System

Choose the right strategy for your memory budget:

| Strategy | Memory Profile | Best For |
|----------|---------------|----------|
| **Standard** | Full KV buffer | Short sequences, single request |
| **Rotating** | Fixed window ring buffer | Ultra-long sequences (avoid OOM) |
| **Quantized** | 4/8/16-bit KV compression | Memory-constrained scenarios |
| **Paged** ⭐ | 32-token pages + CoW | Continuous batching (production default) |
| **PagedQuantized** | Paged + Quantized combined | Extreme memory optimization |
| **Tiered** | RAM hot + SSD cold + LRU | Ultra-long context + multi-model |

```
Source: src/kvcache/paged.zig (1,152 lines)
        src/kvcache/tiered.zig
```

### Layer 5: Zero-Copy Model Loading (7GB memcpy → 0)

The TTFT optimization plan eliminates unnecessary memory copies during model loading:

| Phase | What | Before | After |
|-------|------|--------|-------|
| P0 | Binary index cache | 67s parsing 33 shards | ~1s mmap read |
| P2 | Zero-copy weight loading | ~7GB memcpy | 0 (direct mmap) |
| P3 | Batched shard I/O | Random reads | Sequential OS readahead |

```
Source: docs/en/technical/ttft-optimization.md
```

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────┐
│                    dmlx Inference Engine               │
├─────────────────────────────────────────────────────────┤
│  Model (DeepSeek V4, 3,091 lines)                        │
│  ├── MLA (Multi-head Latent Attention)                   │
│  ├── MoE Router (top-k, 256 experts)                     │
│  ├── Expert Stream (on-demand loading, 10GB peak)        │
│  ├── YARN RoPE (1M+ context)                             │
│  └── mHC (multi-Hyper Connection)                        │
├─────────────────────────────────────────────────────────┤
│  KV Cache (6 strategies, runtime-switchable)             │
│  ├── Paged (32-token pages, CoW, Wyhash prefixes)        │
│  ├── Tiered (RAM + SSD, LRU eviction)                    │
│  └── Prompt Cache (save/restore to safetensors)          │
├─────────────────────────────────────────────────────────┤
│  Quantization (6 formats)                                │
│  ├── Affine INT4/INT8 (group_size: 32/64/128)            │
│  ├── MXFP4 / NVFP4 (microscaling)                       │
│  ├── FP8 E4M3                                            │
│  └── TurboQuant (Lloyd-Max + QJL)                       │
├─────────────────────────────────────────────────────────┤
│  Generation Engine                                        │
│  ├── Speculative Decoding (PLD + EAGLE)                  │
│  ├── Guided Decoding (JSON Schema / Regex FSM)           │
│  └── Continuous Batching (OpenAI-compatible API)         │
└─────────────────────────────────────────────────────────┘
```

---

## Why Zig?

| Aspect | Python (mlx-lm) | Zig (dmlx) |
|--------|----------------|---------------|
| **Memory control** | GC + implicit | Explicit, compile-time checked |
| **Concurrency** | GIL-limited | True parallelism, no GIL |
| **Deployment** | Python runtime + deps | Single static binary |
| **Metal access** | Via mlx-c FFI | Direct kernel integration |
| **Safety** | Runtime errors | Compile-time guarantees |
| **Startup** | ~2s interpreter warmup | Instant native launch |

---

## Real-World Performance

**Hardware**: MacBook Pro M4 Max, 48GB unified memory  
**Model**: DeepSeek-V4-Flash-4bit, 33 shards (~150GB raw)

| Metric | Value |
|--------|-------|
| TTFT (32-token prompt) | 200-500ms |
| ITL (inter-token latency) | 250-500ms |
| Throughput | 2-4 tokens/s |
| Memory (with SMELT 15%) | ~6GB for weights + KV cache |
| Max context (Paged + Tiered) | 128K+ tokens |

---

## TileKernels Reproduction — Custom Metal Kernels

dmlx doesn't just call MLX ops — it reproduces key optimizations from [DeepSeek's TileKernels](https://github.com/deepseek-ai/tilekernels), the production-grade GPU kernel library powering DeepSeek's training and inference.

### Fused hc_split_sinkhorn Metal Kernel

The most significant reproduction: a **hand-written Metal GPU kernel** that fuses 4 operations into a single GPU dispatch:

```metal
// deepseek_v4.zig:2341-2393 — Custom Metal kernel
// Single dispatch performs:
//   1. Pre-sigmoid activation:   sigmoid(x) + eps
//   2. Post-sigmoid activation:  2 * sigmoid(x)
//   3. Comb softmax:             row-wise softmax with max-subtraction
//   4. Sinkhorn normalization:   iterative row/col normalization → doubly-stochastic
// All with optimized float4 vectorization for Apple Silicon GPU
```

**Why this matters:**
- TileKernels' core philosophy is **"fuse multiple logical operations into a single GPU kernel"** to minimize global memory round-trips
- Without fusion: 4-5 kernel launches, 4-5 memory round-trips
- With fusion: 1 kernel launch, 1 memory round-trip
- **This is a direct reproduction of TileKernels' `mhc/` module**, adapted from CUDA to Metal

```
Source: src/models/deepseek_v4.zig:2341-2393 (hc_split_sinkhorn_metal_source)
        src/models/deepseek_v4.zig:2395-2441 (sinkhornNormalize — MLX fallback)
```

### Fused SwitchGLU with gather_mm

Reproducing TileKernels' `moe/` module dispatch pattern:

- **Fused expert weights**: `[n_experts, intermediate, hidden]` format (matching TileKernels' fused layout)
- **gather_mm dispatch**: Loads only selected expert weights via `mlx_gather_mm`, matching TileKernels' "dispatch only necessary weights" approach
- **gather_qmm for quantized**: Quantized expert path uses `mlx_gather_qmm` — fused dequant + matmul in one step

```zig
// deepseek_v4.zig:670-703 — DSV4SwitchGLU
// Stores expert weights in fused [n_experts, out, in] format
// Dispatches via gather_mm (or gather_qmm for quantized)
// — matching mlx-lm's SwitchGLU and TileKernels' MoE dispatch
```

```
Source: src/models/deepseek_v4.zig:670-703 (DSV4SwitchGLU)
        src/models/deepseek_v4.zig:968-1033 (DSV4MoE forward with streaming + gather_mm)
```

### Sinkhorn Normalization

The Sinkhorn algorithm iteratively normalizes a matrix to be doubly-stochastic (rows and columns both sum to 1). This is used in DeepSeek V4's multi-Hyper Connection (mHC) for combining multiple residual streams.

- **MLX path**: `sinkhornNormalize()` — iterative row/col normalization using MLX ops (20 iterations default)
- **Fused Metal path**: The custom Metal kernel above — Sinkhorn baked into the fusion
- **Tested**: Unit test verifies the output is doubly-stochastic within tolerance

```
Source: src/models/deepseek_v4.zig:2395-2441
        src/tests/deepseek_v4_tests.zig:203 — test "sinkhornNormalize produces doubly stochastic"
```

### What Makes This Impressive

| Aspect | Challenge | dmlx Solution |
|--------|-----------|-----------------|
| **Platform** | TileKernels targets NVIDIA CUDA (H100/B200) | Adapted to Apple Metal GPU |
| **Language** | TileKernels uses Python/TileLang | Hand-written Zig + Metal Shading Language |
| **Vectorization** | CUDA warps (32 threads) | Metal SIMD groups + float4 vectorization |
| **Memory model** | Explicit HBM management | Apple Unified Memory (zero-copy CPU/GPU) |
| **Correctness** | Numerical equivalence critical | Unit tests verify vs Python reference |

---

## Application Scenarios

→ See [Application Scenarios](../scenarios/README.md) for detailed use cases.

| Scenario | Why dmlx |
|----------|-------------|
| **Local LLM inference** | Run 671B model on a laptop — no cloud needed |
| **Privacy-first applications** | All data stays on-device, zero network egress |
| **Edge deployment** | Mac mini as private inference server for team |
| **Offline/censored-region access** | Full LLM capability without internet |
| **Development & testing** | Iterate on LLM apps without GPU cluster costs |
| **Research** | Experiment with MoE routing, KV cache strategies |

---

## Related Documentation

- [DeepSeek V4 Fix History](../deepseek-v4/) — Chat template fixes, optimization plans
- [KV Cache Deep Dive](../analysis/04-kv-cache-subsystem.md) — Six strategies in detail
- [Quantization Pipeline](../analysis/06-quantization-training.md) — All formats explained
- [TTFT Optimization](../technical/ttft-optimization.md) — Model loading optimization
- [SMELT System](../technical/smelt-flow.md) — Expert streaming architecture
- [Project Roadmap](../../ROADMAP.md) — Future improvements
