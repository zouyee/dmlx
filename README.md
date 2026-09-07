# dmlx — Run Frontier LLMs on Your Mac

[![CI](https://github.com/zouyee/dmlx/actions/workflows/ci.yml/badge.svg)](https://github.com/zouyee/dmlx/actions/workflows/ci.yml)

> **284B-parameter DeepSeek V4 Flash on a 48GB MacBook Pro. No cloud. No GPU cluster. Just your laptop.**

dmlx runs frontier Mixture-of-Experts models on Apple Silicon through a native C/Metal inference path with packed expert streaming. The latest reproducible DeepSeek V4 native result is **1.236 tok/s** for the standard short-run median and **1.950 tok/s** for the separately classified 100-token diagnostic on an M4 Pro 48GB; see the [machine-readable artifact](docs/en/analysis/performance-benchmark.json).

```
284B parameters → packed Q4 experts on SSD → bounded SMELT N=20 hot cache → runs within 48GB
```

---

## Quick Start

```bash
# Install dependencies
brew install zig mlx-c

# Build
git clone https://github.com/zouyee/dmlx.git && cd dmlx
make

# Generate packed experts (one-time, required for --native mode)
python3 scripts/repack_experts.py ~/models/DeepSeek-V4-Flash-4bit

# Serve — Native mode (recommended for DeepSeek V4, no MLX runtime overhead)
./zig-out/bin/dmlx serve --model ~/models/DeepSeek-V4-Flash-4bit \
  --port 8080 \
  --expert-packed-dir ~/models/DeepSeek-V4-Flash-4bit/packed_experts \
  --native

# Serve — MLX mode (other models, or fallback)
./zig-out/bin/dmlx serve --model ~/models/DeepSeek-V4-Flash-4bit \
  --port 8080 --smelt --smelt-experts 0.2

# Query the server
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'
```

---

## Performance

**Hardware**: M4 Pro 48GB | **Model**: DeepSeek-V4-Flash-4bit | **Mode**: native, packed experts, `NATIVE_SMELT_N=20`

| Metric | Current measured value |
|---|---:|
| Standard short-run effective throughput | **1.236 tok/s** (median of 3 requests, 5 completion tokens each) |
| Long-decode effective throughput | **1.950 tok/s** (100 completion tokens, 51.292 s; diagnostic only) |
| Startup to health-ready | **6.163 s** |
| True HTTP TTFT / independently bounded prefill | **unknown — not measured** |
| Engine first-token latency (after tokenization) | **2.498 s** median |
| Steady-state ITL p50 / p95 / max | **428.721 / 471.672 / 472.575 ms** |
| Critical-path expert I/O | **5,271 `pread` calls; 70,469,812,224 bytes; 6,350.424 ms wait; 0 short reads** |
| Peak RSS | **unknown — not measured** |
| Correctness | **Paris PASS; 7/7 E2E PASS** |

These throughput values use HTTP request wall time. The 100-token run is explicitly excluded from the standard median and must not be presented as the standard benchmark. Exact run values, configuration, null fields, and timestamps are in [`performance-benchmark.json`](docs/en/analysis/performance-benchmark.json); the readable report is [`performance-benchmark.md`](docs/en/analysis/performance-benchmark.md).

> **Native mode (`--native`) is the recommended path for DeepSeek V4 Flash 4-bit.** It runs attention, mHC, and MoE through the native C/Metal engine. Packed experts are required, and the safe default is `NATIVE_SMELT_N=20`.

---

## Why dmlx?

| | mlx-lm (Python) | dmlx (Zig) |
|---|---|---|
| DeepSeek V4 on 48GB | OOM | **bounded N=20 cache + SSD expert streaming** |
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
  ├─ Layer 2: Exact Q4 + SMELT N=20       ~11GB hot expert cache (safe 48GB default)
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
<summary><b>Layer 2: SMELT — Bounded Hot Expert Residency</b></summary>

| Configuration | Approximate expert-cache memory | Use case |
|---|---:|---|
| Full 256-expert preload | ~147 GB across 43 layers | **Prohibited on 48GB** |
| `NATIVE_SMELT_N=51` | ~28 GB | Near the machine limit; diagnostic only |
| **`NATIVE_SMELT_N=20`** | **~11 GB** | **Safe default for 48GB** |

SMELT keeps a bounded hot set resident and loads cold misses from packed expert files. Native mode must provide `--expert-packed-dir`; full preload is not a supported 48GB configuration.
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

The native benchmark emits both the human-readable report and the authoritative schema-v1 JSON artifact:

```bash
NATIVE_SMELT_N=20 bash scripts/run_benchmark.sh

# Optional alternate artifact location
BENCHMARK_JSON=/tmp/dmlx-benchmark.json NATIVE_SMELT_N=20 bash scripts/run_benchmark.sh
```

The standard result is always the median of three 5-token requests. The 100-token request is a separate diagnostic. Packed experts are required at `<model_path>/packed_experts/`, and the script performs a clean ReleaseFast build before measurement.

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
