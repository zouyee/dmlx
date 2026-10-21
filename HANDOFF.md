# Handoff — Qwen3.8-27B text-only support (2026-10-20)

Branch `glm53-mlx-support`, HEAD after Task 7 + GDN prefill optimization: see
`git log` (commits `fb826f1`..`ae35a88` + acceptance commit + `9a42856`,
`26274ba`).

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

**Prefill, L=512 prompt (2026-10-21, `9a42856` + `26274ba`):**

| Engine | tok/s |
|---|---|
| dmlx before (per-token GDN prefill loop) | 32.6 |
| dmlx Task 1 (`9a42856`, vectorized projections/conv/gates/out) | ~92 |
| **dmlx Task 2 (`26274ba`, single-kernel Metal recurrence)** | **~110–116** |
| mlx-lm (warm, reference) | ~95 |

- Task 1: one batched qmm per projection over `[1,L,*]`, whole-sequence
  causal conv (`convAccumSeq`), vectorized gates/out-gate; recurrence stayed
  a per-step eager loop (`deltaRulePrefillEager`, state synced every 16
  steps — never eval the stacked per-step ys: live Metal shared events are
  released only on array free and L×48 of them exhaust the pool).
- Task 2: the recurrence is one Metal kernel per GDN layer
  (`deltaRulePrefillFused`, port of mlx-lm `gated_delta.py`: grid
  (32, Dv, Hv), threadgroup (32,4,1), state in registers, serial for-t loop,
  `simd_sum` over Dk). Eligible when D%32==0 + f32; toy/small dims fall back
  to the eager loop. `prefill` dispatches via `deltaRulePrefill`.
- Numerics gates hold: prefill-vs-per-token unit test (L=3/64, 1e-5),
  kernel-vs-eager recurrence test (HV=48/D=128, L=37/64/128, 1e-5), oracle
  cos ≥ 0.9999 all segments, benchmark Paris + 7/7 E2E.
- Gotcha fixed en route: inline `try ops.slice(...)` temporaries passed to
  `ops.reshape` leak their handle; each leaked slice pinned its 12.6 MB
  parent buffer (+2.2 GB per prefill iteration).

**Decode, long-diagnostic (2026-10-21, after round 2):**

| Engine | ms/token | tok/s |
|---|---|---|
| dmlx before optimization | 157 | 6.72 |
| dmlx after round 1 (phases 1–5b) | 78 | 12.8 |
| **dmlx now (fully fused GDN tail, 9025567 + bdb417e + 6c93624)** | **~69** | **~14.4** |
| mlx-lm 0.31.3 + mlx 0.32.3 (parity target) | 64 | 15.6 |

Round 4/5 (final, kernel-level):
- GPU kernel timeline (micro-bench, single-command-buffer timing): qmm is at
  **97% of the 52.8 ms DRAM roofline** (14.4 GB packed weights/token) — not
  the target. MLP qmm alone = 37.6 ms.
- **v3 conv+tail kernel** (bdb417e): the depthwise causal conv folded into
  the gdn_tail kernel (threadgroup-per-head; q/k conv via threadgroup-shared
  stage, v in registers; new conv state written in-kernel). GDN layer is now
  4 qmm + 1 kernel + out_proj qmm. Plus **fused add-rmsnorm** kernel for all
  128 ln/residual sites. A/B (200-token): 70.9 → 69.2 ms/token.
- **gate+up packed concat + silu_mul kernel** (6c93624): 128 fewer kernel
  launches/token; perf-neutral on wall (69.45 vs 69.25) — dispatch overhead
  is fully hidden behind GPU busy; kept for CPU-submit headroom.
- Stopping point rationale: wall ≈ GPU busy ≈ 69 ms; the residual gap to
  mlx-lm (64 ms) is inside qmm wall + tail-kernel state traffic; further
  gains need sub-roofline tricks (e.g. bf16 ssm state — numerics risk) or
  MLX qmm kernel work itself. Declared done at ~93% parity.
| omlx (no custom kernels) | — | 13.4 |
| flash-moe | n/a (MoE-only) | |

Round-1 phases: single eval barrier/token (5940174), embedding dequant-once
(2f89b63), slab KV cache + native GQA SDPA (f748a02), GDN op-count cuts
(88c6def), persistent layer modules (43abe0d, perf-neutral), mx.compile of
GDN step + SwiGLU core (863dee6) + attention pre-cache segment (fe04c3e).

Round-2 findings (measurement-driven):
- Stage 0 attribution: closure applies are only ~2.4 ms/token; the 63 ms
  "CPU" inside asyncEval is **GPU back-pressure** — MLX's scheduler blocks
  after 10 in-flight command buffers (`wait_for_one`), so wall ≈ GPU-bound
  at ~176 GB/s effective (roofline 273 GB/s). Eval-structure A/Bs and
  mlx 0.31→0.32.4 upgrade: no effect.
- DeepInfer (../DeepInfer) investigated: CUDA SM100/Ascend-only, no GDN, no
  int4 GEMV — nothing borrowable.
- **Fully fused GDN tail Metal kernel** (9025567): post-conv qkv + raw gate
  projections + f32 state → out_proj input + new state, ONE kernel per GDN
  layer (rmsNorm q/k with pre-folded scales, softplus/exp/sigmoid gates,
  delta rule, out-gate silu + gated rmsNorm, exact dtype round-trips).
  Interleaved A/B: 83.4 → 70.8 ms/token (Δ12.6 ms). NOTE: round-2's v1
  delta-rule kernel (9892ce7) was dtype-gated OFF in the real path (beta is
  bf16 there) — v2 is the first fusion that actually runs; v1 remains as the
  fallback inside the eager tail.
- Remaining ~5 ms gap to mlx-lm: wall ≈ GPU busy ≈ 69 ms; qmm is at 97% of
  the DRAM roofline. Further gains need sub-roofline tricks (bf16 ssm state —
  numerics risk vs the oracle gate; GDN state r/w is ~6.3 MB/layer) or MLX
  qmm kernel improvements upstream. Diminishing returns reached.

Environment notes:
- mlx-c 0.7.0 (mlx 0.32.4) via homebrew. mlx_z dependency is upstream
  zouyee/mlx-zig **v0.0.3** (URL dep in build.zig.zon) — the 0.7.0 compat
  patches (SDPA force_fused, cumsum/cumprod axis API, metal-kernel new
  signature, vector add_value→append_value) and safetensors FP8 dtypes were
  upstreamed and released there. Note: qwen38_gdn.zig's fused kernel calls
  the mlx-c metal-kernel C API directly (written before the binding was
  fixed); migrating it to the v0.0.3 `custom_kernel.CustomMetalKernel`
  binding is a cleanup option.
- `zig build test` silently used a stale binary for a while (2>/dev/null
  hid compile failures after the mlx-c upgrade); always verify by running
  the newest test binary directly and checking the count.

Benchmark: `SKIP_PURGE=1 SKIP_BENCHMARK_BUILD=1 bash scripts/run_benchmark_qwen38.sh`
(Paris + 7-prompt E2E + median/decode tok/s). Profiler harness:
`zig build run-qwen38-profile -Doptimize=ReleaseFast -- --mode both --decode 50`
(src/test_qwen38_profile.zig; buildsplit shows CPU-build vs GPU-wait;
`DMLX_PROF_APPLY=1` adds apply/eval breakdown).

- CLI: model load + 32 tokens in ~15 s total.
- Prefill is vectorized + single-kernel recurrence (see table above): L=512
  prompt at ~110 tok/s warm.

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
  (`454 passed; 1 skipped; 0 failed`).

## Regression status

- Full unit suite: 454 passed, 1 skipped, 0 failed (440 at Task 7; +2
  vision-rejection tests, +1 extended GDN prefill equivalence, +1 fused
  prefill recurrence kernel test).
- GLM-5.3 smoke (checkpoint present at
  `~/models/GLM-5.3-Flash-MLX-mixed-4_8bit-dmlx`): served on :8081, one
  non-streaming request → coherent text, `finish_reason: stop`, server alive.
  Note: `dmlx chat` CLI dispatch fails for GLM (`detectModelType` reads the
  first `model_type` key, which is the nested `glm5_next_text`); the server
  path detects by substring and works. Pre-existing.
