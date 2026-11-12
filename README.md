# dmlx

**A Zig inference runtime for Apple Silicon, built for model-specific speculative decoding and memory-bounded local execution.**

[![CI](https://github.com/zouyee/dmlx/actions/workflows/ci.yml/badge.svg)](https://github.com/zouyee/dmlx/actions/workflows/ci.yml)
[![macOS](https://img.shields.io/badge/platform-macOS-black?logo=apple)](https://support.apple.com/macos)
[![Apple Silicon](https://img.shields.io/badge/hardware-Apple%20Silicon-555)](https://support.apple.com/en-us/116943)
[![Zig](https://img.shields.io/badge/Zig-0.16%2B-F7A41D?logo=zig&logoColor=white)](https://ziglang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`dmlx` runs local language models through a Zig-controlled serving and inference stack backed by Apple MLX, custom Metal kernels, and a separate native path for SSD-streamed MoE models.

Its two flagship paths are:

- **Qwen3.8-27B-4bit + MTP** — model-specific speculative decoding with GPU-chained drafts, multi-token verification, rollback, and an adaptive draft-depth controller. Image input is supported end to end (CLI `--image`, server `image_url` content parts, bf16 vision tower with MRoPE, MTP-compatible).
- **DeepSeek V4 Flash + native expert streaming** — a 284B-parameter MoE checkpoint running within a 48GB unified-memory budget by keeping a bounded expert hot set in RAM and reading cold experts from SSD.

> **Project status:** experimental. The flagship configurations below are benchmarked and exercised regularly. Other registered model architectures are compatibility paths, not equivalent claims of production readiness.

---

## Why dmlx

### Model-aware speculative decoding

The Qwen3.8 path does more than call a generic draft model:

- a dedicated MTP head with its own cache
- GPU-chained greedy drafting with one synchronization per round
- one target forward over the proposed block
- transactional commit and rollback of target and drafter state
- online depth selection from observed round cost and conditional acceptance
- shape-routed quantized matmul kernels for verify workloads

### Models larger than unified memory

The DeepSeek V4 native path does not try to preload every routed expert. It combines:

- packed expert files with direct `pread`
- a bounded SMELT hot-expert cache
- asynchronous cold-expert loading
- native Metal kernels for attention and routed MoE execution

On a 48GB Mac, the safe default is `NATIVE_SMELT_N=20`. Full expert preload would require roughly 147GB and is intentionally unsupported on this hardware class.

### An inspectable local runtime

The service, scheduler, model adapters, cache management, and performance-critical dispatch logic live in Zig. Python is used for conversion, oracle checks, and benchmark tooling, but the serving process does not require a Python runtime.

The MLX-backed path still depends on `mlx-c`; “Zig-controlled” does not mean “MLX-free.” The DeepSeek V4 `--native` path is the separate native execution path.

---

## Choose a path

| Path | Execution | Best for | Status |
|---|---|---|---|
| Qwen3.8-27B-4bit + MTP | Zig runtime + `mlx-c` + custom Metal kernels | Fast local decode and speculative-decoding research | **Flagship / benchmarked** |
| DeepSeek V4 Flash | Native engine + packed SSD-streamed experts | Running a 284B MoE model under a 48GB memory budget | **Flagship / benchmarked** |
| Llama-like, Mistral, Qwen2/3, Gemma, GLM-4, Phi/Phi-3, LLaVA, GLM-5 Next | MLX-backed registry adapters | Compatibility and development | **Implemented / model-dependent** |
| n-gram speculation, PLD, EAGLE, DSpark and alternate Metal paths | Mixed | Kernel and decoding experiments | **Experimental** |

---

## Quick start

### Requirements

- macOS on Apple Silicon
- Zig 0.16 or newer
- `mlx-c`, discoverable through `pkg-config` or an explicit prefix

```bash
brew install zig mlx-c

zig version
pkg-config --modversion mlxc
```

If `pkg-config` cannot locate `mlx-c`, set `MLX_C_PREFIX` or pass `-Dmlx_prefix=/path/to/mlx-c` to `zig build`.

### Build

```bash
git clone https://github.com/zouyee/dmlx.git
cd dmlx
zig build -Doptimize=ReleaseFast
```

The binary is written to `zig-out/bin/dmlx`.

### Serve Qwen3.8-27B with MTP

The target model and MTP drafter are separate checkpoints.

```bash
export DMLX_QWEN38_MTP_DIR=~/models/Qwen3.8-27B-MTP-4bit

./zig-out/bin/dmlx serve \
  --model ~/models/Qwen3.8-27B-4bit \
  --mlx \
  --speculative-mtp 7 \
  --port 8080
```

`--speculative-mtp 7` sets the adaptive controller’s ceiling; it does not force every round to draft seven tokens.

MTP performance measurements must use non-streaming requests. The current streaming route does not exercise the same speculative path.

### Serve DeepSeek V4 Flash in native mode

Pack the routed experts once:

```bash
python3 scripts/repack_experts.py ~/models/DeepSeek-V4-Flash-4bit
```

Then start one server instance:

```bash
pkill -f "dmlx serve" 2>/dev/null; sleep 1

NATIVE_SMELT_N=20 ./zig-out/bin/dmlx serve \
  --model ~/models/DeepSeek-V4-Flash-4bit \
  --expert-packed-dir ~/models/DeepSeek-V4-Flash-4bit/packed_experts \
  --native \
  --port 8080
```

Do not use full expert preload on a 48GB machine. One DeepSeek V4 server already holds model weights and an expert cache; running two instances can exhaust unified memory.

### Query the server

```bash
curl http://127.0.0.1:8080/health

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Explain unified memory briefly."}],
    "max_tokens": 64,
    "temperature": 0,
    "stream": false
  }'
```

The server exposes OpenAI-style Chat Completions and an Anthropic-style Messages endpoint. It does not claim full compatibility with every endpoint or response field from either hosted API.

---

## Benchmarks

Benchmark claims in this README are deliberately narrow. Each result names its model, hardware, content mode, and raw artifact. Do not compare results across different prompt templates, thinking modes, sampling settings, cache states, or metric definitions.

### Qwen3.8-27B-4bit speculative decode

- **Hardware:** Apple M4 Pro, 48GB unified memory
- **Model:** local Qwen3.8-27B-4bit target with the same MTP drafter for both engines
- **Protocol:** greedy, non-streaming, one server at a time, prefix cache disabled or defeated, interleaved A/B
- **Metric:** streamed token-equivalent content deltas per second during decode; the protocol observed one delta per token for these engines, but SSE does not guarantee this generally

| Workload | dmlx | omlx-mtp | Difference | Source |
|---|---:|---:|---:|---|
| Non-thinking decode headline | **43.2/s** | 36.1/s | **+19.7%** | [`final_nothink_20261106.json`](artifacts/final_nothink_20261106.json) |
| Aligned thinking decode headline | **40.5/s** | 35.6/s | **+13.8%** | [`think_latch_20261106.json`](artifacts/think_latch_20261106.json) |
| Thinking, code | **47.1/s** | 37.7/s | **+24.9%** | [`think_latch_20261106.json`](artifacts/think_latch_20261106.json) |
| Thinking, math | **46.6/s** | 42.3/s | **+10.2%** | [`think_latch_20261106.json`](artifacts/think_latch_20261106.json) |
| Short-prompt TTFT, non-thinking | **0.28s** | 0.91s | **3.2× lower** | [`final_nothink_20261106.json`](artifacts/final_nothink_20261106.json) |

The headline is the arithmetic mean of the `code`, `prose`, and `math` prompt results. The intentionally predictable `count` prompt is excluded. “Aligned thinking” means both arms receive the same explicit reasoning directive; unaligned thinking outputs are not comparable because speculative acceptance depends on the text each engine generates.

The harness, prompt set, aggregation, cache controls, and metric definitions are in [`scripts/compare_engines.py`](scripts/compare_engines.py). The raw artifacts currently report engine versions as `?`; treat the figures as a recorded local result, not a portable guarantee.

Historical comparisons against `mlx-vlm` and `mlx-lm` are not used as headline claims because several earlier runs mixed decode-only rates with end-to-end rates or compared speculative and non-speculative execution. The common-basis result against `mlx-vlm` was parity, not a demonstrated win. See [`HANDOFF.md`](HANDOFF.md) for the measurement history and withdrawn conclusions.

### DeepSeek V4 Flash native path

- **Hardware:** Apple M4 Pro, 48GB
- **Configuration:** native mode, packed experts, `NATIVE_SMELT_N=20`
- **Source of truth:** [`performance-benchmark.json`](docs/en/analysis/performance-benchmark.json)

| Metric | Result |
|---|---:|
| Standard effective completion throughput | **1.236 tok/s** — median of three 5-token HTTP requests |
| Long-decode diagnostic | 1.950 tok/s — 100 tokens, excluded from the standard median |
| Startup to health-ready | 7.218s |
| Peak RSS | 21,829,861,376 bytes, about 20.3GiB |
| Correctness | Paris PASS; 7/7 end-to-end prompts PASS |

See the [human-readable report](docs/en/analysis/performance-benchmark.md) for request-level timings, telemetry coverage, configuration, and unavailable metrics. In particular, the report does **not** claim a true HTTP TTFT measurement for this run.

### Reproduce locally

```bash
# Qwen3.8 benchmark
EXTRA_SERVE_FLAGS="--speculative-mtp 7" \
  bash scripts/run_benchmark_qwen38.sh ~/models/Qwen3.8-27B-4bit

# DeepSeek V4 correctness
bash scripts/dsv4_smoke.sh
bash scripts/run_kernel_tests.sh
bash scripts/run_mla_attention_test.sh

# DeepSeek V4 standard benchmark
NATIVE_SMELT_N=20 bash scripts/run_benchmark.sh --native \
  ~/models/DeepSeek-V4-Flash-4bit
```

Benchmark numbers are hardware-, model-, prompt-, and version-specific. Please attach the raw artifact when reporting a result.

---

## How the Qwen3.8 MTP path works

```text
prompt
  │
  ├─ target prefill ───────────────► first target token
  │                                      │
  │                                 MTP head cache
  │                                      │
  └─► adaptive draft chain: d1 → d2 → … → dN
                                         │
                         one target verify over [token, d1…dN]
                                         │
                         accept longest matching prefix
                                         │
                         commit accepted state / rollback rejected state
                                         └─► next round
```

Important implementation points:

- `src/models/qwen38_mtp.zig` — MTP head, cache, chained draft body, coarse candidate head
- `src/models/qwen38_gdn.zig` — Gated DeltaNet prefill, verify, decode, and custom kernels
- `src/models/verify_qmm.zig` — shape-based split-K and simdgroup-matrix routing
- `src/model_registry.zig` — target/drafter adapter, prompt priming, commit, and rollback
- `src/engine/engine_loop.zig` — speculative request loop and adaptive depth controller

---

## How DeepSeek V4 fits in 48GB

The routed MoE has 256 experts per layer and activates six per token. Loading every expert for every layer would require about 147GB even in the packed format, so the native path uses a bounded working set:

```text
request
  │
  ├─ resident non-expert weights
  ├─ SMELT hot experts in unified memory
  └─ cold expert miss
       └─ async pread from packed_experts/
            └─ routed Metal kernel
```

Approximate memory budget for the tested 48GB machine:

| Component | Approximate memory |
|---|---:|
| Non-expert model weights | ~5GB |
| Attention and mHC weights | ~2GB |
| SMELT expert cache, `N=20` | ~11GB |
| Metal buffers | ~2–3GB |
| OS and other processes | ~3–5GB |

`N=20` leaves room for the OS page cache and is the recommended default. Larger hot sets can improve hit rate but approach the machine’s memory limit quickly.

---

## API and runtime boundaries

Currently exposed routes:

- `POST /v1/chat/completions`
- `POST /v1/responses` (minimal Responses API compatibility layer)
- `POST /v1/messages`
- `POST /v1/tokenize` (also `/tokenize`)
- `POST /v1/embeddings` (last-hidden pooling, Qwen3.8 path)
- `GET /v1/models`
- `GET /metrics` (Prometheus text format, also `/v1/metrics`)
- `GET /health`
- `POST /shutdown`

Current boundaries:

- the HTTP service starts with one configured model
- request handling is currently serialized through the active engine path
- MTP performance measurements use non-streaming requests
- registered architecture support is broader than the set of continuously benchmarked checkpoints
- experimental speculative methods and alternate Metal paths may have no stable server entry point

These boundaries are intentional documentation, not a roadmap promise.

---

## Build, test, and benchmark

```bash
make                 # ReleaseFast build
make test            # build + unit tests
make verify          # model-backed DeepSeek V4 verification
make benchmark       # DeepSeek V4 benchmark; requires local model and packed experts
make check           # test + verify + benchmark
```

Metal changes should also run:

```bash
bash scripts/run_kernel_tests.sh
bash scripts/run_mla_attention_test.sh
```

Model-backed commands require the corresponding local checkpoints and enough unified memory. They are not part of a lightweight dependency-only test run.

---

## Repository map

```text
src/
├── main.zig                 CLI and command dispatch
├── model_registry.zig       model adapters and MTP integration
├── engine/                  request loop, scheduling, prefix cache
├── server/                  HTTP endpoints and request handling
├── models/
│   ├── qwen38.zig           Qwen3.8 target model
│   ├── qwen38_mtp.zig       Qwen3.8 MTP drafter
│   ├── qwen38_gdn.zig       GDN implementation and kernels
│   ├── verify_qmm.zig       verify-shape quantized matmul routing
│   ├── deepseek_v4.zig      DeepSeek V4 model path
│   ├── expert_stream.zig    packed expert I/O
│   └── expert_cache.zig     bounded expert residency
├── kvcache/                 KV-cache strategies
├── tokenizer/               tokenizer implementations
├── guided.zig               regex and JSON-constrained decoding
└── trainer.zig              local training experiments

scripts/                     conversion, correctness, profiling, benchmarks
artifacts/                   raw benchmark and diagnostic outputs
docs/                        design notes and measured reports
```

---

## Documentation

| Document | Purpose |
|---|---|
| [DeepSeek V4 benchmark report](docs/en/analysis/performance-benchmark.md) | Current native benchmark, protocol, and telemetry |
| [DeepSeek MoE technical notes](docs/en/deepseek-moe/README.md) | Packed experts and native MoE execution |
| [Qwen3.8 MTP engineering history](HANDOFF.md) | Optimization campaign, measurement corrections, and remaining limits |
| [MTP framework plan](docs/analysis/mtp-framework-plan.md) | Direction for a reusable speculative-decoding framework |
| [Contributing](CONTRIBUTING.md) | Development workflow |
| [Changelog](CHANGELOG.md) | Release history |

Some files under `docs/analysis/` are engineering journals rather than current user documentation. Prefer the benchmark artifact and current source when a historical note conflicts with a measured result.

---

## Acknowledgments

Built on [Apple MLX](https://github.com/ml-explore/mlx) and [`mlx-c`](https://github.com/ml-explore/mlx-c).

Custom kernels and techniques include work adapted from:

- [oMLX](https://github.com/jundot/omlx)
- [MTPLX](https://github.com/youssofal/mtplx)
- [Splash](https://github.com/incoai/splash)
- [DeepSeek TileKernels](https://github.com/deepseek-ai/TileKernels)

See source headers for per-kernel provenance and licenses. Adapted Apache-2.0 and MIT code retains its required attribution.

## License

[MIT](LICENSE)
