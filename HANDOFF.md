# Handoff — Qwen3.8-27B text-only support (2026-10-20)

Branch `glm53-mlx-support`, HEAD after Task 7 + GDN prefill optimization: see
`git log` (commits `fb826f1`..`ae35a88` + acceptance commit + `9a42856`,
`26274ba`).

## What works

- **Qwen3.8 MTP speculative decoding (2026-10-23)** — see the dedicated
  section below. `--speculative-mtp N` (server/CLI) enables it;
  ~1.6-1.7× effective decode on greedy workloads.

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

## MTP speculative decoding (2026-10-23)

**Design** (commits `2bd5b34`, `41b4907`, `23d6321` + memory-hardening fix):
- Drafter: `mlx-community/Qwen3.8-27B-MTP-4bit` (239 MB, 4-bit gs=64) at
  `/Users/zouyee/models/Qwen3.8-27B-MTP-4bit`; override via
  `DMLX_QWEN38_MTP_DIR`. Loaded lazily on first `--speculative-mtp` request.
- `src/models/qwen38_mtp.zig` (`Qwen38Mtp`): pre_fc norms + fc(2H→H) + one
  full-attention decoder layer (own KV cache) + final norm → shared
  embed/lm_head. Chaining convention (verified against omlx/HF): the next
  pass consumes the POST-final-norm output.
- Backbone verify continuation (`forwardVerifyImpl`/`commitVerify`/
  `rollbackVerify` in `qwen38.zig`): L>1 without reset; GDN per-position
  states retained per round and committed/rolled back (`prefillVerify`,
  per-step recurrence — verify rounds are tiny, N+1 ≤ 4). Attention rollback
  is `len_` truncation. MLX fast SDPA `"causal"` is right-aligned (verified
  empirically) — no explicit mask needed.
- Engine loop `processMtpRequest` (non-streaming only; guided decoding is
  untouched): draft chain → one backbone verify → greedy longest-prefix at
  temperature 0 / speculative sampling with (p−q)+ resample at temp>0 →
  commit/rollback → bonus token on full acceptance. Drafter prompt pass at
  prefill (head processes the prompt so its KV/RoPE offsets are correct).
- Commit semantics (2026-10-24, `afe2886`): on commit the drafter cache is
  truncated to round start + t_next, then each accepted draft is REPLAYED
  through the head with the TARGET's pre-final-norm verify hidden rows
  (mlx-vlm `accept_verified_tokens` semantics; `MtpOps.commit` now takes the
  verify token ids). Replaces the full-acceptance tail-draft hack. A/B on
  fixed temp-0 prompts: acceptance 85.1/65.5/88.0% vs 85.1/63.4/88.0%
  (code/prose/math) — neutral-to-slightly-better; 128-token request wall
  5.39 s warm both sides. Acceptance is content-bound (mirrors mlx-vlm's
  98/61/88 spread), not cache-semantics-bound. Gotcha documented in
  `qwen38_mtp.loadNormF32`: the community drafter checkpoint is flat-layout
  (norms scale-encoded, NO +1 shift — adding it drops acceptance ~85→63%).
  Env-gated debug: `DMLX_MTP_STAGE` (per-stage checksums in the head),
  `DMLX_MTP_DEBUG` (per-round engine trace).
- Flags: `--speculative-mtp N` (drafts/round; N=2 default choice, N=3 also
  works). Benchmark script: `EXTRA_SERVE_FLAGS="--speculative-mtp 2"`.

**Measured (temp 0, M4 Pro 48 GB):**

| Metric | MTP off | MTP N=2 | MTP N=3 |
|---|---|---|---|
| 128-token request, code (best of 2) | 9.14 s | 5.54 s | 5.82 s |
| 128-token request, prose | 9.20 s | 5.98 s | 6.35 s |
| 128-token request, math | 9.44 s | 5.95 s | 5.94 s |
| Benchmark long-decode diagnostic | 13.8 tok/s | **24.05 tok/s** | — |
| Acceptance rate (code/prose/math) | — | 83–85% | 72–82% |

→ N=2 is the sweet spot (~1.6-1.7× effective decode; N=3 slightly worse).

**Correctness / losslessness (honest version):**
- MTP-on is fully deterministic at temperature 0 (after fixing the
  `req.sampler` bug below).
- 2/3 fixed prompts bitwise identical to MTP-off; the third diverges at one
  bf16 near-tie flip. Root cause measured: batched-verify vs decode-step
  logits drift up to 0.0625 (bf16 rounding) — argmax flips only when top-2
  gap < ~0.06. Strict bitwise equality is infeasible at bf16 (same for
  omlx/vLLM). Instead verified: self-consistency probe — every emitted token
  equals the model's own argmax in a fresh single batched re-prefill (9/9),
  i.e. MTP emits exactly the target model's greedy path under batched
  scoring.
- Benchmark with `--speculative-mtp 2`: Paris ✓, E2E 7/7.

**Memory (hardened before measurement):**
- All MTP round exits free the verify box + retained per-step GDN states
  (single `defer freeVerify`; previously error/max_tokens paths leaked
  ~150 MB × positions × 48 layers per round). h_cur/hidden_next ownership
  audited; stale drafter round state cleared in freeVerify.
- Per-round transient is bounded by design: (N+1) × ~150 MB × 48 GDN layers
  retained only within one round. Peak RSS (0.5 s polling during
  2×128-token requests): off 11.56 GB vs N=2 11.48 GB — no delta.
- Drafter loads once per adapter (idempotent enable), freed at adapter deinit.

**Bugs found and fixed along the way:**
- `req.sampler` was never initialized from the request in the serial engine
  path (only `prefillBatch` does) — `SamplerConfig` defaults to
  temperature **1.0**, so temperature=0 requests still sampled stochastically
  in the MTP loop. Fixed inside `processMtpRequest` (init from gen_config).
  **Known issue, unfixed**: the pre-existing ngram speculative path
  (`processSpeculativeRequest` fallback `generateStep` calls) uses the same
  uninitialized `req.sampler` and has the same latent bug.
- Emission overshoot past max_tokens (capped).
- mlx-lm has no native MTP for this arch (it strips the weights at load);
  omlx monkey-patches. dmlx's is a native implementation.

## Measured performance (M4 Pro, 48 GB)

**Prefill scaling (2026-10-22, warm best-of-3, interleaved same-session;
dmlx = Task 1 + Task 2 + sync-cadence fix):**

| L | dmlx tok/s | mlx-lm tok/s | omlx (custom kernels) tok/s |
|---|---|---|---|
| 512 | 112.6 | 117.5 | 123.7 |
| 1024 | 117.5 | 124.6 | 127.2 |
| 2048 | 117.2 | 124.8 | 128.2 |
| 4096 | 114.8 | 124.1 | 116.8 |

omlx measured 2026-10-22 via `omlx serve` + /v1/chat/completions
(prompt_tokens / prompt_eval_duration, fresh random-word prompts each
attempt — prefix cache confirmed defeated via cached_tokens=0; actual prompt
sizes undershot the target by up to ~8% at 2k/4k, harmless given flat
scaling). omlx's L=4096 attempts degraded across reps (116.8 → 93 → 73,
thermal/memory pressure), so its 4k number is a best-case outlier of a
declining series; dmlx was stable at 113-117 there. Interleaving note: dmlx
numbers above are from a solo run; when measured with the omlx server
co-resident (15 GB), dmlx lost ~10% at L=1024 to memory pressure.

**Speculative decode comparison (2026-10-23, temp 0, N=2 drafts both sides):**

| Engine | code tok/s | prose tok/s | math tok/s | acceptance |
|---|---|---|---|---|
| dmlx `--speculative-mtp 2` | 22.6 | 20.9 | 21.9 | 83–85% |
| mlx-vlm 0.7.8 `--draft-kind mtp` | 31.1 | 26.1 | 31.0 | 61–98% (avg draft 2) |

(mlx-vlm numbers are its own generation tok/s; dmlx numbers are effective
tok/s over 128-token requests incl. prefill + thinking — dmlx decode-only
diagnostic is 24.05. omlx's native `mtp_enabled` does NOT work with the
local 4-bit checkpoint: it was stripped of `mtp.*` weights, and omlx's
native MTP needs them in the main checkpoint ("Lightning MTP is inactive …
checkpoint has no MTP weights"); the mlx-vlm row uses the same community
drafter we use. Notable: acceptance varies by content on both sides
(mlx-vlm: 98% code / 61% prose / 88% math; dmlx: flat ~85% everywhere) —
yet mlx-vlm is faster even where OUR acceptance is higher (prose), so its
edge is pipeline efficiency (BatchGenerator verify loop, faster decode
baseline) more than acceptance. Verdict: dmlx is BELOW mlx-vlm speculative
today (~1.3–1.4× gap), roughly at parity nowhere but within ~20% on math.)

**Decode, same session (2026-10-22):**

| Engine | tok/s |
|---|---|
| dmlx | 14.0 |
| mlx-lm | 15.41 |
| omlx (custom kernels) | 15.42 |

Baseline before this work was 32.6 tok/s at L=512 (per-token GDN loop) and
flat ~32 at any L, so the improvement is ~3.5× across the range. NOTE: the
earlier "mlx-lm warm ~95" reference was measured at L=61 where fixed
overheads dominate; at matched lengths mlx-lm is ~123-124 tok/s, so dmlx is
within 4-7% at prefill (not ahead). The residual gap is 4-bit qmm compute
throughput at long-L shapes — prefill at long L is compute-bound (MLP ≈ 55%
of barrier-inflated section time at L=4096; SDPA measured in isolation:
40 ms/call at L=4096 = <2% of wall, so the omlx fa256 port is NOT
warranted). Decode: omlx ≈ mlx-lm ≈ 15.4 vs dmlx 14.0 — the ~9% delta is
the known residual qmm-wall/dispatch gap (decode SDPA kernels measured
wall-neutral at every context length), not attention.

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
- Sync cadence (2026-10-22): `prefill(..., sync)` — the serving loop syncs
  every 8th GDN layer + the last layer instead of every layer. Per-layer
  `out.eval()` was added for a memory blow-up that actually came from a
  slice-handle leak (fixed separately); removing the 48 per-prefill drains
  bought ~4-8% (L=4096: 107 → 115 tok/s). Peak MLX memory at L=4096 ≈ 35 GB
  regardless of cadence (dominated by the [1,L,248320] logits + qmm
  transients, not layer intermediates).
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
- **Decode SDPA port (9c9c166 + follow-up commit)**: omlx's
  `omlx_sdpa_decode` (mlx PR #4294) ported as the decode-path SDPA in
  `qwen38_attn.postCache`: 1-pass kernel (one 1024-thread group per q-head,
  KV striped over 32 simdgroups in 4-key tiles, fp32 online softmax,
  vectorized bf16 loads) plus the 2-pass split-KV variant for kv_len ≥ 1024
  (pass 1: grid (HKV,1,blocks) × (32,gqa,1) threads, ~256-key chunks, fp32
  partials + per-block max/sum; pass 2: LSE merge per q-head; block count is
  a runtime scalar so no per-length recompile). Gated to qL=1/bf16/D%32==0/
  contiguous slabs, stock SDPA fallback otherwise; `DMLX_NO_SDPA_DECODE=1`
  disables. Measured: barrier-isolated stock SDPA ≈ 2.7 ms/token at kv~300,
  but interleaved A/B shows **no wall gain** — ~71.4 vs ~71.3 ms/token at
  kv~300 and ~69.2 vs ~68.9 at kv~4100 (±4 ms run variance). The decode step
  is bound by the 14 GB weight read; SDPA traffic (≤25 MB/token even at 4k
  context) is irrelevant. Kept for correctness-validated parity with omlx
  and as the vehicle for any future long-context work, but it does NOT move
  decode. (Two-pass verified vs stock SDPA at kv_len 1025/2048/4097, 5e-3.)

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
- Prefill is vectorized + single-kernel recurrence + 8-layer sync cadence
  (see table above): ~113-117 tok/s warm from L=512 to 4096.

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
