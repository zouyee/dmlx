# dmlx — Run Frontier LLMs on Your Mac

[![CI](https://github.com/zouyee/dmlx/actions/workflows/ci.yml/badge.svg)](https://github.com/zouyee/dmlx/actions/workflows/ci.yml)

> **284B-parameter DeepSeek V4 Flash on a 48GB MacBook Pro. No cloud. No GPU cluster. Just your laptop.**

dmlx is a high-performance LLM inference engine for Apple Silicon, built in Zig 0.16 on Apple's MLX Metal backend. It delivers **18-26 tok/s** on hardware that can't even load the model with other tools.

```
284B params → 4-bit quantized (40GB) → SMELT 20% (8GB) → runs on 48GB Mac
```

---

## Quick Start

```bash
# Install dependencies
brew install zig mlx-c

# Build
git clone https://github.com/zouyee/dmlx.git && cd dmlx
make

# Chat (single prompt)
./zig-out/bin/dmlx chat --model ~/models/DeepSeek-V4-Flash-4bit \
  --prompt "Explain quantum computing in one sentence" \
  --smelt --smelt-experts 0.2

# Serve (OpenAI-compatible API, Trust OS mode — no custom cache)
./zig-out/bin/dmlx serve --model ~/models/DeepSeek-V4-Flash-4bit \
  --port 8080 --smelt --smelt-experts 0.2

# Query the server
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'
```

---

## Performance

**Hardware**: M4 Pro 48GB | **Model**: DeepSeek-V4-Flash 4-bit (284B, 33 shards) | **Mode**: SMELT 20%, Trust OS, packed experts

| Metric | Value |
|--------|-------|
| Throughput (steady-state) | **~1.1 tok/s** (99.2s/100tok) |
| 30-token latency (warm) | **~23.2s** |
| Steady-state ITL | ~887 ms |
| Memory usage | ~2.5 GB (Trust OS) |
| 7-prompt correctness | **7/7 PASS** |

> **Trust OS (cache=0) with packed experts is the recommended default.** Score-free DyMoE skips 1 expert per layer (17% I/O reduction) without breaking MLX lazy fusion. See [analysis](docs/analysis/flash-moe-alignment-plan.md).

<details>
<summary><b>Optimization history</b></summary>

| Metric | Initial (cached) | Current (Trust OS + packed) | Notes |
|--------|-----------------|---------------------------|-------|
| Client 100-token | 193s | **99.2s** | +94% faster |
| First-request TTFR | 113s | **23.2s** | +388% faster |
| Server RSS | 4.7GB | **2.5GB** | -47% memory |
| E2E correctness | 7/7 | **7/7** | maintained |


---

## Why dmlx?

| | mlx-lm (Python) | dmlx (Zig) |
|---|---|---|
| DeepSeek V4 on 48GB | OOM | **8GB via SMELT** |
| KV cache strategies | 1 fixed | 6 runtime-switchable |
| Max context (48GB) | RAM-limited | **128K+ (RAM+SSD)** |
| Binary size | ~500MB+ (Python stack) | ~10MB static binary |
| Latency jitter | 10-100ms GC pauses | Zero GC, sub-ms |
| iOS embedding | No | Yes (C ABI) |

**The core advantage is not raw speed — it's that the model runs at all on consumer Macs.**

---

## How It Works: 5-Layer Memory Optimization

```
Full model (568GB BF16)
  │
  ├─ Layer 1: MoE Expert Streaming    138GB → 10GB   (load only active 6/256 experts)
  ├─ Layer 2: 4-bit Quant + SMELT      40GB →  8GB   (partial expert preloading)
  ├─ Layer 3: CSA/HCA Attention         KV cache 9.5× smaller than V3
  ├─ Layer 4: 6 KV Cache Strategies     Standard|Rotating|Quantized|Paged|Tiered
  └─ Layer 5: Zero-Copy Loading         137s → 46s startup (mmap + batched I/O)
```

<details>
<summary><b>Layer 1: MoE Expert Streaming (138GB → 10GB)</b></summary>

DeepSeek V4 activates top-6 of 256 experts per token. dmlx loads only active experts on-demand from SSD via pread (Trust OS mode — no custom cache, relying on macOS page cache). Expert caching is available via `--smelt-cache <MB>` for larger RAM configurations.

```
Source: src/models/expert_stream.zig | src/models/expert_cache.zig
```
</details>

<details>
<summary><b>Layer 2: SMELT — Selective Model Expert Loading (40GB → 8GB)</b></summary>

| Mode | Experts Preloaded | Memory | Use Case |
|------|-------------------|--------|----------|
| Full 4-bit | 256 (100%) | ~40 GB | 96GB+ Mac |
| SMELT 30% | ~77 | ~10 GB | 64GB Mac |
| **SMELT 20%** | **~51** | **~8 GB** | **48GB Mac** |
| SMELT 15% | ~38 | ~6 GB | Memory-tight |

SMELT preloads frequently-used experts and applies routing bias to avoid loading unused ones. Cache misses fall back to on-demand streaming.
</details>

<details>
<summary><b>Layer 3: CSA + HCA Hybrid Attention</b></summary>

DeepSeek V4's native architecture — Compressed Sparse Attention (m=4) interleaved with Heavily Compressed Attention (m'=128), plus FP8 KV storage. Result: **9.5x smaller KV cache** than V3.

```
Source: src/models/deepseek_v4.zig (3,091 lines)
```
</details>

<details>
<summary><b>Layer 4: Six KV Cache Strategies</b></summary>

| Strategy | Best For |
|----------|----------|
| Standard | Short sequences |
| Rotating | Ultra-long sequences (ring buffer) |
| Quantized | Memory-constrained (4/8/16-bit KV) |
| **Paged** (default) | Production (32-token pages + CoW) |
| PagedQuantized | Extreme memory optimization |
| Tiered | 128K+ context (RAM hot + SSD cold) |

Plus **Prefix Cache** with LRU eviction — skip prefill entirely on repeated prompts.
</details>

<details>
<summary><b>Layer 5: Zero-Copy Model Loading</b></summary>

| Optimization | Before | After |
|-------------|--------|-------|
| Binary index cache | 67s parsing | ~1s mmap |
| Weight loading | 7GB memcpy | Direct mmap |
| Shard I/O | Random reads | Sequential readahead |

Combined: **137s → 46s** (66% reduction).
</details>

---

## Features

**Inference Engine**
- 9 architectures: DeepSeek V4, LLaMA, Mistral, Qwen2/3, Gemma, GLM-4, Phi, Phi-3
- OpenAI + Anthropic API compatible HTTP server (SSE streaming)
- Speculative decoding (PLD + EAGLE), guided decoding (JSON Schema / Regex FSM)
- Prefix cache with LRU eviction (25-48% TTFR reduction on repeated prompts)
- Quantization: INT4/INT8, MXFP4, FP8, TurboQuant
- QLoRA fine-tuning, tool calling, vision (LLaVA)

**MLX Bindings**
- 200+ operations, autograd, NN layers, compiled op fusion
- Type-safe Zig API over mlx-c (Metal GPU + CPU Accelerate)

---

## Installation

**Requirements**: macOS Apple Silicon (M1-M4) + Zig 0.16.0 + mlx-c

```bash
brew install zig mlx-c
```

**Build from source:**

```bash
git clone https://github.com/zouyee/dmlx.git
cd dmlx
make              # Build (ReleaseFast)
make test         # Unit tests (400+)
make benchmark    # Full benchmark (build + unit tests + perf + 7-prompt e2e + report)
make check        # Everything: build + test + verify + benchmark
```

The benchmark script is the single source of truth for performance and correctness:

```bash
bash scripts/run_benchmark.sh                              # defaults: 0.20 experts, Trust OS (cache=0)
bash scripts/run_benchmark.sh ~/models/DeepSeek-V4-Flash-4bit 0.15 0  # custom smelt fraction
```
Packed experts are auto-detected from `<model_path>/packed_experts/`. Generate with `scripts/repack_experts.py`.

**As a Zig dependency** (`build.zig.zon`):

```zig
.dependencies = .{
    .dmlx = .{
        .url = "https://github.com/zouyee/dmlx/archive/refs/tags/v0.3.0.tar.gz",
        .hash = "...",
    },
},
```

---

## Architecture

```
src/
├── main.zig                # CLI (chat, serve, benchmark)
├── models/                 # DeepSeek V4, LLaMA, Qwen, Gemma, ...
│   ├── deepseek_v4.zig    # MLA + CSA/HCA + MoE (3K lines)
│   ├── expert_stream.zig  # On-demand expert loading from SSD
│   └── expert_cache.zig   # Expert cache (disabled by default — Trust OS)
├── engine/                 # Inference engine
│   ├── engine_loop.zig    # Main loop + prefix cache integration
│   └── prefix_cache.zig   # LRU prefix cache
├── server/                 # HTTP server (OpenAI + Anthropic API)
├── kvcache/                # 6 strategies (standard → tiered)
├── tokenizer/              # BPE tokenizer
├── speculative.zig         # PLD + EAGLE
├── guided.zig              # JSON/Regex constrained decoding
└── trainer.zig             # QLoRA + AdamW
```

---

## Use Cases

- **Local inference** — GPT-4-class intelligence on-device, zero API costs
- **Privacy/compliance** — HIPAA/GDPR: all data stays on your hardware
- **Edge server** — Mac mini as a team inference server (OpenAI API drop-in)
- **Offline** — Download once, run anywhere without internet
- **Research** — Modify MoE routing, swap KV strategies, test quantization formats

---

## Documentation

| | |
|---|---|
| [Technical Deep Dive](docs/en/deepseek-moe/README.md) | How 284B runs on 48GB |
| [Performance Benchmark](docs/en/analysis/performance-benchmark.md) | Latest numbers |
| [Optimization Roadmap](docs/analysis/optimization-roadmap.md) | What's next |
| [Contributing](CONTRIBUTING.md) | Developer guidelines |

---

## Acknowledgments

Built on [Apple MLX](https://github.com/ml-explore/mlx) and `mlx-c`. Custom Metal kernels adapted from [DeepSeek TileKernels](https://github.com/deepseek-ai/tilekernels).

## License

[MIT](LICENSE)
