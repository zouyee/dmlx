# Handoff — Qwen3.8-27B text-only support (2026-10-20)

Branch `glm53-mlx-support`, HEAD after Task 7: see `git log` (commits `fb826f1`..`ae35a88` + acceptance commit).

## What works

- **Qwen3.8-27B-4bit (qwen3_5, hybrid GDN + full attention)** end to end:
  config parsing (`src/models/qwen38_loader.zig`), GDN linear-attention module
  (`qwen38_gdn.zig`), partial-RoPE full attention (`qwen38_attn.zig`), model
  assembly aligned to the mlx-lm oracle (cos >= 0.9999, `qwen38.zig`), serving
  model + registry arch `Qwen3_5ForConditionalGeneration` + CLI/server wiring
  (ChatML template, multi-EOS stop tokens 248046/248044).
- CLI: `dmlx chat --model /Users/zouyee/models/Qwen3.8-27B-4bit --prompt ... --max-tokens N`
  → coherent completions, clean `<|im_end|>` termination.
- Server: `dmlx server --model ... --port 8080`, `/v1/chat/completions`
  streaming + non-streaming; 7-prompt acceptance (short QA / code / long
  instruction) all returned `finish_reason: "stop"`, server survives
  sequential requests (per-layer GDN/attention caches reset on each prefill).
- Vision input rejected explicitly: token ids 248053–248057
  (`<|vision_start|>`, `<|vision_end|>`, `<|vision_pad|>`, `<|image_pad|>`,
  `<|video_pad|>`) raise `error.VisionInputNotSupported` from
  `Qwen38ServingModel.forwardTokens` (`qwen38.rejectVisionTokens`).

## Measured performance (M4 Pro, 48 GB)

**Decode, long-diagnostic (2026-10-21, after perf phases 1–5b):**

| Engine | tok/s |
|---|---|
| dmlx before optimization | 6.72 |
| **dmlx now (profiler A/B mean)** | **12.8** |
| mlx-lm 0.31.3 + mlx 0.32.3 (parity target) | 15.6 |
| omlx (no custom kernels) | 13.4 |
| flash-moe | n/a (MoE-only) |

Phases: single eval barrier/token (5940174), embedding dequant-once (2f89b63),
slab KV cache + native GQA SDPA (f748a02), GDN op-count cuts (88c6def),
persistent layer modules (43abe0d, perf-neutral cleanup), mx.compile of the
GDN step + SwiGLU core (863dee6) + attention pre-cache segment with
ropeDynamic offset (fe04c3e). Root cause was CPU dispatch-bound:
~3900 mlx-c calls/token at ~18 µs each; compiled closures cut the graph
count but the remaining ~60 ms/token CPU sits in eager attention/norm
remnants and closure-apply overhead. Next lever (not done): whole-step
mega-closure (all layer weights as inputs, one apply per token) or a fused
custom Metal GDN kernel via the existing custom_kernel.zig binding.

Benchmark: `SKIP_PURGE=1 SKIP_BENCHMARK_BUILD=1 bash scripts/run_benchmark_qwen38.sh`
(Paris + 7-prompt E2E + median/decode tok/s). Profiler harness:
`zig build run-qwen38-profile -Doptimize=ReleaseFast -- --mode both --decode 50`
(src/test_qwen38_profile.zig; buildsplit shows CPU-build vs GPU-wait).

- CLI: model load + 32 tokens in ~15 s total.
- Prefill ~13 tokens in <1 s (GDN prefill loop is still per-token; not yet optimized).

## Fixes made during acceptance (Task 7)

- **Sampler O(n²) fix (`src/sampling.zig`)**: `topKSampler` /
  `topKTopPSampler` full-sorted the entire vocab with insertion sort — fine at
  32k vocab, ~minutes/token at Qwen3.8's 248,320 vocab. Now a stable O(n·k)
  top-k selection (`selectTopK`); `topPSampler` uses stable `std.sort.block`.
  Semantics preserved (stable ordering, ties keep lower token id).
- **finish_reason threaded to the wire** (commit after acceptance): the engine
  always computed the real verdict but `openai.zig` / `streaming.zig` /
  `anthropic.zig` hardcoded `"stop"` / `"end_turn"`. Now threaded end-to-end:
  `StreamCallbackCtx` got `stop_tokens` (EOS → `.stop`, cap → `.length`), and
  the non-streaming/speculative/guided/DSV4 engine paths use
  `finishReasonForTokens` in `engine_loop.zig`. Anthropic maps
  `.stop`→`end_turn`, `.length`→`max_tokens`. Verified live on Qwen3.8:
  EOS → `stop`, max_tokens=8 → `length` (streaming final chunk too),
  Anthropic → `end_turn` / `max_tokens`.

## Known gaps / concerns

- **Streaming SSE text loses spaces**: the engine decodes each token
  individually (`engine_loop.zig` `tokenizer.decode(&[_]u32{token})`) and
  `bpe.zig` decode strips the leading space (step 3), so streamed deltas
  concatenate without spaces ("Theuserwants…"). Non-streaming output is
  correct. Pre-existing, affects all models; needs incremental decode state.
- glm53-inherited loader error-path leaks (minor, only on failure paths).
- CLI `--temperature 0.8` is the global default *and* a sentinel: qwen38 chat
  maps "left at default" to the recommended temp 1.0 / top_k 20 / top_p 0.95,
  so an explicit `--temperature 0.8` cannot be distinguished from the default.
- `zig build test` reports a failed step due to the Zig 0.16 `--listen=-`
  runner quirk; real signal is the test binary tail line
  (`440 passed; 1 skipped; 0 failed`).

## Regression status

- Full unit suite: 440 passed, 1 skipped, 0 failed (baseline was 438; +2 new
  vision-rejection tests).
- GLM-5.3 smoke (checkpoint present at
  `~/models/GLM-5.3-Flash-MLX-mixed-4_8bit-dmlx`): served on :8081, one
  non-streaming request → coherent text, `finish_reason: stop`, server alive.
  Note: `dmlx chat` CLI dispatch fails for GLM (`detectModelType` reads the
  first `model_type` key, which is the nested `glm5_next_text`); the server
  path detects by substring and works. Pre-existing.
