# dmlx Deep Gap Analysis: vs mlx/mlx-c/mlx-lm/oMLX/TileKernels/vLLM

> Based on 2026-04-26 code audit (37K+ lines Zig, 100+ files, 350 tests),
> compared against 6 reference projects, identifying leverageable strengthening directions.
> **Phase 7 (P0-P3) fully complete, including all optional tasks.** 3 models verified passing.

---

## I. dmlx True State (Honest Assessment)

### Verified Working
- ✅ TinyLlama-1.1B-Chat-v1.0-4bit: loading + quantized inference + token generation
- ✅ Qwen2.5-0.5B-Instruct: loading + non-quantized inference + token generation
- ✅ Qwen3-0.6B-4bit: loading + quantized inference + token generation (including quantized embedding dequantization)
- ✅ 350 unit tests all passing
- ✅ Metal GPU as default device
- ✅ Streaming token output (output as each token is generated)
- ✅ EOS stop condition
- ✅ Chat template support (LLaMA/Qwen/Mistral/Qwen3/GLM-4 paths)
- ✅ Gemma/Phi-3/Qwen3/GLM-4 model architecture support
- ✅ Repetition penalty / min_p sampling
- ✅ OpenAI API: stop/logprobs/tool_calls/Anthropic compatible
- ✅ EAGLE speculative decoding
- ✅ Custom Metal kernel / distributed inference / model conversion / perplexity evaluation
- ✅ Quantized embedding auto-dequantization (Qwen3 and similar models)
- ✅ Explicit head_dim support (Qwen3 head_dim=128 ≠ hidden_size/num_heads)

### Real Bugs / Defects (All Fixed)
1. ~~**generate loop poor performance**~~ → ✅ Fixed (KV cache incremental update)
2. ~~**Gemma loader not implemented**~~ → ✅ Implemented
3. ~~**No streaming token output**~~ → ✅ Implemented
4. ~~**No EOS stop**~~ → ✅ Implemented
5. ~~**No chat template application**~~ → ✅ Implemented
6. ~~**No multi-turn dialogue**~~ → ✅ Chat template supports multi-turn format
7. ~~**Qwen3 quantized embedding load failure**~~ → ✅ Auto dequantization
8. ~~**Qwen3 head_dim mismatch**~~ → ✅ Explicit head_dim parsing
9. ~~**Large tokenizer.json load failure**~~ → ✅ Limit raised to 50MB

### Remaining Optional
- All P0-P3 tasks completed, including optional tasks 28.3 (Qwen3) and 28.4 (GLM-4)

---

## II. Item-by-Item Comparison

### 1. vs mlx (Apple Official Framework)

| Capability | mlx | dmlx | Gap | Leverage Value |
|------------|-----|---------|-----|---------------|
| Lazy evaluation | ✅ Core | ✅ via mlx-c | None | — |
| Metal kernel | ✅ Auto | ✅ via mlx-c | None | — |
| `mlx.compile` | ✅ | ✅ Bound | None | — |
| Custom Metal kernel | ✅ | ❌ | **High** | Can write custom Metal shaders to accelerate hotspots |
| Distributed (NCCL) | ✅ v0.29+ | ❌ | Medium | Multi-Mac cluster inference |
| CUDA backend | ✅ v0.29+ | ❌ | Low | Not needed for Apple Silicon focus |

**Leverage direction**: Custom Metal kernel. mlx allows registering custom Metal shaders, enabling dedicated kernels for hotspots like MoE expert dispatch and fused attention, bypassing generic operator overhead.

### 2. vs mlx-c (C API Layer)

| Capability | mlx-c | dmlx Binding | Gap |
|------------|-------|-----------------|-----|
| All ops | ~200 | ~200 | None |
| quantize/dequantize | ✅ 4 mode | ✅ 4 mode | None |
| to_fp8/from_fp8 | ✅ | ✅ | None |
| qqmm/gather_qmm | ✅ | ✅ | None |
| mlx_fast (SDPA/RoPE) | ✅ | ✅ | None |
| mlx_distributed | ✅ | ❌ | Medium |

**Leverage direction**: `mlx_distributed` binding. mlx-c has distributed communication APIs (send/recv/all_reduce), enabling multi-Mac tensor parallelism.

### 3. vs mlx-lm (Python Reference Implementation)

| Capability | mlx-lm | dmlx | Gap | Priority |
|------------|--------|---------|-----|----------|
| Model architectures | 50+ | 5 | **High** | P1 |
| Streaming token output | ✅ yield | ❌ batch output | **High** | P0 |
| Chat template (Jinja2) | ✅ | Partial (V4 only) | **High** | P0 |
| EOS stop | ✅ | ❌ | **High** | P0 |
| Token decode (ids→text) | ✅ | ✅ exists but unused in LLaMA chat | Medium | P1 |
| Repetition penalty | ✅ | ❌ | Medium | P2 |
| Top-p / Top-k sampling | ✅ | ✅ | None | — |
| Perplexity evaluation | ✅ | ❌ | Medium | P2 |
| Model conversion (convert) | ✅ | ❌ stub | Low | P3 |
| GGUF export | ✅ | ❌ | Low | P3 |

**Highest priority leverage**:
1. **Streaming token output** — mlx-lm's `generate_step` is yield pattern, outputting each token as generated. dmlx's generate waits for full completion.
2. **EOS stop** — mlx-lm checks `eos_token_id` and terminates early. dmlx always generates max_tokens.
3. **Chat template** — mlx-lm renders chat templates with Jinja2. dmlx only has chat template on V4 path, LLaMA path passes raw prompt.

### 4. vs oMLX (Production-Grade Server)

| Capability | oMLX | dmlx | Gap | Priority |
|------------|------|---------|-----|----------|
| Continuous Batching | ✅ Real | ⚠️ Framework exists but no real batch forward | **High** | P1 |
| SSE keep-alive | ✅ | ❌ | Medium | P2 |
| Context scaling | ✅ | ❌ | Medium | P2 |
| Model TTL | ✅ | ❌ | Low | P3 |
| Web admin panel | ✅ | ❌ | Low | P3 |
| Anthropic API compatibility | ✅ | ❌ | Low | P3 |

**Leverage direction**: Real batch forward. Current engine loop processes decode requests individually, not merging multiple request tokens into one batch tensor for single forward. This is the core of continuous batching — requires `batch_builder` to truly build batched input.

### 5. vs TileKernels (DeepSeek GPU Kernel Library)

| Capability | TileKernels | dmlx | Gap |
|------------|-------------|---------|-----|
| Numerical verification framework | ✅ cosine sim + bias check | ✅ golden test | None |
| Per-token/per-block quantization | ✅ | ✅ | None |
| Sinkhorn normalization | ✅ | ✅ | None |
| MoE routing pipeline | ✅ | ✅ | None |
| Custom CUDA kernel | ✅ | N/A (Metal) | — |
| FP8/FP4 training kernel | ✅ | ❌ | Low (inference priority) |

**Leverage direction**: No significant gap. TileKernels' core value (MoE routing, quantization kernels, numerical verification) is already implemented in dmlx.

### 6. vs vLLM (Industrial-Grade Inference Engine)

| Capability | vLLM | dmlx | Gap | Priority |
|------------|------|---------|-----|----------|
| PagedAttention | ✅ | ✅ | None | — |
| Prefix Caching | ✅ Real | ⚠️ Hash registered but unused in lookup | Medium | P2 |
| Chunked Prefill | ✅ | ✅ Framework exists | None | — |
| Speculative Decoding | ✅ ngram/EAGLE/Medusa | ✅ ngram | Medium | P2 |
| Guided Decoding | ✅ FSM/xgrammar | ✅ FSM | None | — |
| Disaggregated Serving | ✅ | ❌ | Low | P3 |
| Multi-GPU TP/PP | ✅ | ❌ | Low | P3 |
| OpenAI API Completeness | ✅ | ⚠️ Basic | Medium | P2 |

**Leverage direction**:
1. **OpenAI API completeness** — vLLM supports full `/v1/chat/completions` including `stream`, `stop`, `temperature`, `top_p`, `max_tokens`, `logprobs`, `tool_calls`. dmlx's server only supports basic fields.
2. **EAGLE speculative decoding** — more efficient than n-gram, but requires additional draft model head.

---

## III. Priority Ranking: Highest ROI Improvements

### P0 (Immediate Fix — Affects Basic Usability)

1. **Streaming token output** — Print each token as generated, don't wait for full completion
2. **EOS stop** — Check `eos_token_id` to terminate generate loop early
3. **Chat template applied to LLaMA path** — Read chat_template from tokenizer_config.json
4. **generate performance** — Confirm KV cache incremental update works correctly (no repeated prefill)

### P1 (Near-term — Significant Competitiveness Boost)

5. **More model architectures** — Phi-3/4, Qwen3 (non-VL), GLM-4, GPT-OSS
6. **Real batch forward** — Use batch_builder to merge multiple requests in engine loop
7. **Token decode integration** — LLaMA chat path outputs readable text not token IDs

### P2 (Mid-term — Production Polish)

8. **OpenAI API completeness** — stop, logprobs, tool_calls
9. **SSE keep-alive** — Send heartbeats during long prefill
10. **Repetition penalty** — Avoid repeated output
11. **EAGLE speculative decoding** — More efficient speculative decoding

### P3 (Long-term — Differentiating Features)

12. **Custom Metal kernel** — Write dedicated shaders for MoE dispatch
13. **Multi-Mac distributed inference** — via mlx_distributed
14. **Model conversion tool** — HF → MLX safetensors
15. **Anthropic API compatibility** — Support Claude API format alongside OpenAI
