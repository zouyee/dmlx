# dmlx — The Local AI-Native Engine for Apple Silicon

[![CI](https://github.com/zouyee/dmlx/actions/workflows/ci.yml/badge.svg)](https://github.com/zouyee/dmlx/actions/workflows/ci.yml)

> **Frontier models, fully on your Mac — and faster than the frameworks built for it.**

dmlx is a Zig-native inference engine for Apple Silicon: no Python runtime, a custom Metal
kernel layer over MLX, and a serving core built for local-first AI — private, offline,
zero API cost. It runs **Qwen3.8-27B with MTP speculative decoding faster than omlx-mtp**
(the current fastest MLX-based MTP stack) and **284B-parameter DeepSeek V4 Flash on a
48GB MacBook Pro** through packed expert streaming.

```
Local-first: your weights, your memory, your wire speed — nothing leaves the machine.
```

---

## Benchmarks: dmlx vs omlx-mtp (Qwen3.8-27B-4bit, M4 Pro 48GB, 2026-11-06)

Same model, same greedy workloads, interleaved A/B (reps=3-5, prefix-cache nonce):

| Metric | dmlx | omlx-mtp | Δ |
|---|---:|---:|---:|
| Decode headline (non-thinking) | **43.2 t/s** | 36.1 t/s | **+19.7%** |
| Decode headline (thinking, aligned) | **40.5 t/s** | 35.6 t/s | **+13.8%** |
| — code (thinking) | 47.1 | 37.7 | +24.9% |
| — math (thinking) | 46.6 | 42.3 | +10.2% |
| TTFT (short prompt) | **0.28 s** | 0.91 s | **3.2× faster** |
| TTFT @512 tokens | **4.34 s** | 4.62 s | +6% |
| TTFT @2048 tokens | 16.15 s | 15.94 s | parity (−1.3%, within noise) |

Prefill on both engines is bound by the same stock 4-bit qmm rate (6.7 TF/s on M4 Pro);
dmlx ships the steel BM=64 tile (bit-exact, 1.05-1.12× over stock) plus fused GDN
conv/tail kernels, closing prefill to parity. Decode wins come from the verify engine:
fused GDN verify tail, sg8/split-K verify qmm routing, a compiled drafter step, and an
adaptive draft-depth controller with per-content acceptance tracking.

Artifacts: `artifacts/final_nothink_20261106.json`, `artifacts/final_think_20261106.json`,
`artifacts/think_latch_20261106.json`. Harness: `scripts/compare_engines.py`.

---

## Quick Start

```bash
# Install dependencies
brew install zig mlx-c

# Build
git clone https://github.com/zouyee/dmlx.git && cd dmlx
make

# Serve Qwen3.8-27B with MTP speculative decoding (recommended)
./zig-out/bin/dmlx serve --model ~/models/Qwen3.8-27B-4bit \
  --speculative-mtp 7 --port 8080

# DeepSeek V4 Flash (native mode; one-time expert packing required)
python3 scripts/repack_experts.py ~/models/DeepSeek-V4-Flash-4bit
./zig-out/bin/dmlx serve --model ~/models/DeepSeek-V4-Flash-4bit \
  --port 8080 \
  --expert-packed-dir ~/models/DeepSeek-V4-Flash-4bit/packed_experts \
  --native

# Query the server (OpenAI-compatible)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'
```

---

## Positioning: Local AI-Native

Not "a Python framework ported to Mac" — an engine designed from the metal up for
**running frontier models locally**:

| | Python stacks (mlx-lm / vLLM-ish) | dmlx |
|---|---|---|
| Runtime | ~500MB+ Python, GC pauses | ~10MB static Zig binary, zero GC |
| Kernels | stock MLX dispatch | custom Metal layer: fused GDN conv/tail, verify-regime qmm (sg8/split-K/mma), compiled drafter steps |
| Speculative decoding | fixed depth | adaptive controller, per-content acceptance evidence, greedy GPU-chained drafts |
| Long context | RAM-limited | RAM+SSD tiered KV, 128K+ |
| MoE at 284B | OOM on 48GB | packed expert streaming, bounded hot cache |
| iOS embedding | No | Yes (C ABI) |

**The point is not just speed — it is that the model is yours: offline, private,
inspectable down to the kernel.**

---

## The Two Flagship Paths

### Qwen3.8-27B (dense hybrid GDN + attention, MTP)

```bash
./zig-out/bin/dmlx serve --model ~/models/Qwen3.8-27B-4bit --speculative-mtp 7
```

- MTP speculative decoding with an adaptive depth controller (1-7), greedy GPU-chained
  draft chain (one sync per round), target-verify with commit/rollback
- Hybrid GDN (linear attention) + partial-RoPE full attention; fused prefill conv (v2,
  vectorized bf16-direct) and fused out-gate norm tails on both prefill and verify
- Verify qmm routing: split-K (M=2-5) / sg8 simdgroup-matrix (M=6-8) / experimental mma
  (`DMLX_VERIFY_QMM_MMA=1`); steel BM=64 tile on prefill projections (bit-exact)
- Thinking-mode aligned comparisons supported (`dmlx-think-aligned` arm)

### DeepSeek V4 Flash (284B MoE on 48GB)

| Metric | Measured value |
|---|---:|
| Standard short-run effective throughput | **1.236 tok/s** (median of 3×5-token requests) |
| Long-decode diagnostic | 1.950 tok/s (100 tokens, diagnostic only) |
| Startup to health-ready | 6.163 s |
| Correctness | Paris PASS; 7/7 E2E PASS |

Machine-readable artifact: [`performance-benchmark.json`](docs/en/analysis/performance-benchmark.md).
**Native mode (`--native`) is the recommended path**, with `NATIVE_SMELT_N=20` as the
safe 48GB default.

---

## How It Works: 5-Layer Memory Optimization (DSV4)

```
Full model (568GB BF16)
  │
  ├─ Layer 1: MoE Expert Streaming    138GB → 10GB   (load only active 6/256 experts)
  ├─ Layer 2: Exact Q4 + SMELT N=20     ~11GB hot expert cache (safe 48GB default)
  ├─ Layer 3: CSA/HCA Attention         KV cache 9.5× smaller than V3
  ├─ Layer 4: 6 KV Cache Strategies     Standard|Rotating|Quantized|Paged|Tiered
  └─ Layer 5: Zero-Copy Loading         137s → 46s startup (mmap + batched I/O)
```

<details>
<summary><b>Layer details</b></summary>

- **Layer 1: MoE Expert Streaming** — top-6 of 256 experts per token, on-demand pread
  from SSD (Trust OS mode). `src/models/expert_stream.zig`, `src/models/expert_cache.zig`.
- **Layer 2: SMELT** — bounded hot expert residency; full 256-expert preload (~147GB)
  is prohibited on 48GB; `NATIVE_SMELT_N=20` ≈ 11GB is the safe default.
- **Layer 3: CSA + HCA hybrid attention** — Compressed Sparse (m=4) + Heavily
  Compressed (m'=128), FP8 KV: 9.5× smaller KV cache than V3.
- **Layer 4: Six KV cache strategies** — Standard / Rotating / Quantized / Paged
  (default, 32-token pages + CoW) / PagedQuantized / Tiered (RAM hot + SSD cold).
- **Layer 5: Zero-Copy Loading** — binary index cache (67s → ~1s), direct mmap,
  sequential readahead. 137s → 46s startup.

</details>

---

## Features

**Inference Engine**
- Architectures: DeepSeek V4, Qwen3.5/3.8 (hybrid GDN), LLaMA, Mistral, Qwen2/3,
  Gemma, GLM-4, Phi, Phi-3
- OpenAI + Anthropic API compatible HTTP server (SSE streaming)
- Speculative decoding: MTP (Qwen3.8, adaptive depth), PLD + EAGLE
- Guided decoding (JSON Schema / Regex FSM), prefix cache with LRU eviction
- Quantization: INT4/INT8, MXFP4, FP8, TurboQuant
- QLoRA fine-tuning, tool calling, vision (LLaVA)

**MLX Bindings**
- 200+ operations, autograd, NN layers, mx.compile op fusion
- Type-safe Zig API over mlx-c (Metal GPU + CPU Accelerate)

**Custom Metal kernel layer** (code-generated at runtime, all with kill switches)
- GDN: fused prefill conv+silu, fused out-gate norm tails (prefill + verify regimes),
  pipelined delta-rule recurrence (ported from oMLX, Apache-2.0)
- verify qmm: split-K / sg8 / mma morphologies with shape-based routing
- coarse 2-bit draft head + exact top-k rescore
- steel BM=64 prefill qmm tile (flattened steel headers, bit-exact vs stock)

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
make test         # Unit tests (460+)
make benchmark    # Full benchmark (build + unit tests + perf + 7-prompt e2e + report)
make check        # Everything: build + test + verify + benchmark
```

The native benchmark emits both the human-readable report and the authoritative
schema-v1 JSON artifact:

```bash
NATIVE_SMELT_N=20 bash scripts/run_benchmark.sh
```

The standard result is always the median of three 5-token requests. The 100-token
request is a separate diagnostic. Packed experts are required at
`<model_path>/packed_experts/`.

**As a Zig dependency** (`build.zig.zon`):

```zig
.dependencies = .{
    .dmlx = .{
        .url = "https://github.com/zouyee/dmlx/archive/refs/tags/v0.0.5.tar.gz",
        .hash = "...",
    },
},
```

---

## Architecture

```
src/
├── main.zig                # CLI (chat, serve, benchmark)
├── models/                 # DeepSeek V4, Qwen3.8, LLaMA, Qwen, Gemma, ...
│   ├── deepseek_v4.zig     # MLA + CSA/HCA + MoE
│   ├── qwen38.zig          # Qwen3.8 serving model (hybrid GDN + attention)
│   ├── qwen38_gdn.zig      # GDN block: fused conv/tail kernels, pipelined recurrence
│   ├── qwen38_mtp.zig      # MTP drafter: compiled decode steps, coarse draft head
│   ├── verify_qmm.zig      # verify-regime quantized matmul routing (sg8/split-K/mma)
│   ├── qwen38_qmm_v8.zig   # steel BM=64 prefill qmm tile
│   ├── expert_stream.zig   # On-demand expert loading from SSD
│   └── expert_cache.zig    # Expert cache (disabled by default — Trust OS)
├── engine/                 # Inference engine
│   ├── engine_loop.zig     # Main loop + prefix cache + MTP depth controller
│   └── prefix_cache.zig    # LRU prefix cache
├── server/                 # HTTP server (OpenAI + Anthropic API)
├── kvcache/                # 6 strategies (standard → tiered)
├── tokenizer/              # BPE tokenizer
├── speculative.zig         # PLD + EAGLE
├── guided.zig              # JSON/Regex constrained decoding
└── trainer.zig             # QLoRA + AdamW
```

---

## Use Cases

- **Local inference** — frontier-class intelligence on-device, zero API costs
- **Privacy/compliance** — HIPAA/GDPR: all data stays on your hardware
- **Edge server** — Mac mini as a team inference server (OpenAI API drop-in)
- **Offline** — download once, run anywhere without internet
- **Research** — modify MoE routing, swap KV strategies, test quantization formats,
  iterate on speculative-decoding depth policy

---

## Documentation

| | |
|---|---|
| [Technical Deep Dive](docs/en/deepseek-moe/README.md) | How 284B runs on 48GB |
| [Performance Benchmark](docs/en/analysis/performance-benchmark.md) | DSV4 numbers |
| [HANDOFF.md](HANDOFF.md) | Qwen3.8 MTP campaign log (kernel-level attribution) |
| [Optimization Roadmap](docs/analysis/optimization-roadmap.md) | What's next |
| [Contributing](CONTRIBUTING.md) | Developer guidelines |

---

## Acknowledgments

Built on [Apple MLX](https://github.com/ml-explore/mlx) and `mlx-c`. Custom Metal
kernels adapted from [DeepSeek TileKernels](https://github.com/deepseek-ai/tilekernels);
verify-qmm and GDN recurrence morphologies ported from oMLX,
[MTPLX](https://github.com/youssofal/mtplx), and
[Splash](https://github.com/incoai/splash) (Apache-2.0); steel GEMM headers
from MLX (MIT).

## License

[MIT](LICENSE)
