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
- Flags: `--speculative-mtp N`. **As of the 2026-10-24 depth-controller work
  below, N is the UPPER BOUND on draft depth, not a fixed depth** — the
  engine picks the per-round depth online. `DMLX_MTP_FIXED_N=1` pins it to N
  (for A/B). Benchmark script: `EXTRA_SERVE_FLAGS="--speculative-mtp 4"`.

**Measured (temp 0, M4 Pro 48 GB):**

| Metric | MTP off | MTP N=2 | MTP N=3 |
|---|---|---|---|
| 128-token request, code (best of 2) | 9.14 s | 5.54 s | 5.82 s |
| 128-token request, prose | 9.20 s | 5.98 s | 6.35 s |
| 128-token request, math | 9.44 s | 5.95 s | 5.94 s |
| Benchmark long-decode diagnostic | 13.8 tok/s | **24.05 tok/s** | — |
| Acceptance rate (code/prose/math) | — | 83–85% | 72–82% |

→ N=2 was the sweet spot at the time (~1.6-1.7× effective decode); after the
2026-10-24 replay-semantics + round-overhead fixes, **N=3 is the sweet spot**
(see the update sections below).

**2026-10-24 update (fused GDN verify, `ab078d5`):** the verify recurrence
runs as ONE Metal kernel per GDN layer with per-position states in `s_all`
(`deltaRulePrefillFusedRetain`), replacing the per-step eager retain loop.
Per-round profile (ms, N=2, code prompt): draft 11.0 / verify 82.4 /
accept 3.4 / commit 0.1 (verify was 94.2). Request wall 5.39 → **4.84 s**;
long-decode diagnostic 24.05 → **26.83 tok/s**; acceptance 86.2% (fused vs
eager numeric path flips near-ties). Kernel-side gotcha fixed en route: MLX
passes inputs smaller than its constant-space cutoff as `constant` — both
prefill recurrence kernel bodies now declare small-input pointers
(`g`/`beta`, [L, HV]) as `auto` instead of `const device float*`.

**2026-10-24 update 2 (batched argmax + GPU draft chain, `34ffb13`):**
accept-phase per-row sampling (L × full-vocab CPU pull+scan) replaced by one
batched GPU argmax (`MtpOps.verifyArgmax`): accept 3.4 → 0.2 ms/round.
Greedy draft chain now keeps token ids on GPU (`MtpOps.draftGreedy`,
`Qwen38Mtp.forwardGreedyGpu`), one sync per chain: draft 11.2 → 9.0
ms/round. With per-round overheads down, **N=3 is the new sweet spot**:
replay semantics keep acceptance flat at 85.2% through depth 3, while N=4
degrades (72% code / 51% prose — chained drafts past depth 3 are weak) and
loses. Results (temp 0, M4 Pro): 128-token code request 5.39 → 4.84 →
**4.45 s (N=3)**; long-decode diagnostic 24.05 → 26.83 → **29.37 tok/s
(N=3)** — within ~6% of mlx-vlm's 31.1 (was a 1.3-1.4× gap). Verify-cost
split (DMLX_MTP_DEBUG): N=2 verify 82.4 ms = 64.7 ms graph build (GPU mostly
overlaps it) + 17.8 ms eval drain; +1 verify position costs ~19 ms, mostly
build. **The "graph build" reading in this paragraph is a measurement
artifact — see the correction below.**

**2026-10-24: cross-engine standing, measured on a common basis.** The
"dmlx is 1.3-1.4x BELOW mlx-vlm speculative" verdict recorded further down
this file was a MEASUREMENT BASIS ARTIFACT — it compared dmlx's effective
tok/s (including prefill and the `<think>` block) against mlx-vlm's
generation-only rate. `scripts/compare_engines.py` now drives every engine
through its own OpenAI-compatible server with one client and one metric
definition, deriving a prefill-free decode rate from two non-streaming
requests (`(c_long - c_short) / (t_long - t_short)`):

| engine | code | prose | math | headline | TTFT |
|---|---|---|---|---|---|
| dmlx (adaptive MTP + verify kernels) | 27.8 | **26.7** | **33.3** | **29.3** | **0.25-0.38 s** |
| mlx-vlm 0.7.8 (`--draft-kind mtp`, same drafter) | **32.5** | 21.5 | 32.3 | 28.8 | 0.44-0.53 s |
| mlx-lm 0.31.3 (no speculation) | 15.6 | 15.4 | 16.6 | 15.9 | 0.85-1.13 s |
| omlx 0.7.1 (no speculation — see below) | 15.6 | 14.8 | 15.2 | 15.2 | 0.27-0.28 s |

**Scorecard, stated honestly: dmlx has NOT surpassed any of them.** With
speculation OFF — the only like-for-like decode comparison — dmlx is 13.9
headline vs mlx-lm 15.4 and omlx 15.1, i.e. **10% BEHIND both**; the 29.3
above comes entirely from speculation. The "1.84x" below is therefore not a
capability claim: it pits speculative dmlx against engines that cannot
speculate on THIS checkpoint (omlx's native MTP needs `mtp.*` in the main
checkpoint, which the community 4-bit export stripped; mlx-lm has no MTP for
this arch). The one true apples-to-apples comparison is dmlx vs mlx-vlm —
same drafter, same speculative semantics — and that is a TIE. **TTFT / prefill now measured directly** (§15 of the plan; the earlier TTFT
figure was DERIVED from an 8-token request and covered only ~20-token
prompts). `max_tokens=1` wall gives TTFT; differencing two PROMPT lengths at
`max_tokens=1` gives prefill throughput with the single decode step and all
fixed costs cancelled. Note raw `ttft@L` is NOT comparable across engines —
the same text tokenizes to 520/524/565 tokens — so compare slope (throughput)
and intercept (fixed overhead):

| engine | prefill tok/s | fixed overhead | short TTFT | ttft@2k (normalized) |
|---|---|---|---|---|
| dmlx | 117.4 → **123.7-129.6** (2026-11-05) | **-0.036 s** | **0.277 s** | 17.41 → ~16.3-17.3 s |
| mlx-vlm 0.7.8 | 133.1 | 0.224 s | 0.376 s | 15.61 s |
| omlx 0.7.1 | **134.5 → 137.5-139.4** (0.7.3) | 0.354 s | 0.762 s | **15.58-15.91 s** |

2026-11-05 (commits `5e8ef27`, `3356161`, `8ad3fd3`): the prefill gap was
-11.6% and is now ~-8.5% (median of medians; same-code spread is ±4%). The
biggest single piece was the GDN causal conv silently running its eager
fallback in every serving prefill (qmm returns bf16, the fused kernel only
accepted f32) — ~0.9 s at L=1790. Also landed: runtime-L conv kernel (one
compile per process, not per prompt length), a fused prefill out-gate
kernel, and v8-tile routing in `QuantLinear.apply`. Ruled out by
measurement: GEMM-floor optimism (cold-weight qmm streams at the same
6.4-6.7 TF/s), omlx's W4A8 backend (NAX/M5-only), omlx fused GDN prework
(Qwen3.5 prefill keeps the stock path), the prefill sync cadence, and — via
a synthetic one-layer bench (`b6480e2`) — any hidden structural overhead:
the serving per-layer cost equals the op-level sum. The "~15% over the v8
floor" reading in between was thermal drift. Priced and deliberately
deferred: the out_proj gate is f32 (conv emits f32), which bars the v8 tile
— ~240 ms at L=1901 — but rounding it bf16 is a dtype-policy change across
every tail path (numpy fixtures key on f32), not a local patch.

dmlx has by far the lowest fixed overhead (2.75x better than omlx) but the
lowest prefill throughput (-13%), so it wins TTFT only below ~258 tokens
(vs mlx-vlm) / ~360 tokens (vs omlx) and loses at agent-sized prompts.
Fixed en route: prefill applied lm_head to all L positions and discarded
every row but the last (778 ms wasted at L=2048) — prefill 109.6 -> 117.4
tok/s. Full scorecard and what is still missing:
`docs/analysis/beat-omlx-mlxvlm-plan.md` §14-§15.

So dmlx is at PARITY with mlx-vlm and 1.84x both non-speculative engines.
**Parity, not a win**: the rank flips between runs (28.9 vs 29.0, then 29.3 vs
28.8) and the 0.7% mean gap is smaller than dmlx's own 0.40 run-to-run spread.
Wins split by content: mlx-vlm +17% on code, dmlx +24% on prose, tied on math
— consistent with mlx-vlm using a fixed deeper draft while dmlx adapts depth
online. omlx is a decode/prefill reference only: its native `mtp_enabled` is
inactive on this checkpoint because the community 4-bit export stripped the
`mtp.*` weights that omlx expects in the MAIN checkpoint. Harness validation:
mlx-lm's 15.4-16.6 and omlx's 15.2 here reproduce the 15.41-15.6 / 15.42
recorded independently below.

Two harness traps worth remembering: the measurement MUST be non-streaming
(dmlx's `processMtpRequest` is non-streaming only, so a streaming harness
measures dmlx *without* speculation — 14.8 vs 27.9 tok/s), and mlx-lm's
server resolves the request's `model` field against HuggingFace, so a
placeholder id 404s every request. Details, environment repair, and the
remaining code-prompt gap: `docs/analysis/beat-omlx-mlxvlm-plan.md` §13.

**2026-10-24 correction: verify is GPU-bound, not build-bound.** The
`DMLX_MTP_DEBUG` lap timers attribute the blocking part of
`eval_mod.asyncEval` to whichever lap contains the call, and
`Qwen38GDN.prefillVerify` called it once per GDN layer (48×) to materialize
the retained states. That is why "build" appeared to be ~80 ms concentrated
in `gdn=`. Experiment (parked in `git stash` as the one-batched-asyncEval
variant): hoisting all 48 submissions into a single `asyncEval` after the
layer loop in `forwardVerifyImpl` moved the cost rather than removing it —

| variant | gdn | attn | other | eval | build+eval |
|---|---|---|---|---|---|
| per-layer asyncEval (shipped) | 80.0 | 0.2 | 0.3 | 22.4 | 102.8 |
| one batched asyncEval | 2.2 | 0.3 | 88.9 | 16.3 | 107.7 |

True CPU graph-build for a whole N=3 verify is **~2.8 ms**; the compiled
MLP/attn closures already cover the stack. Clean A/B at N=3 without the
debug env (benchmark long-decode diagnostic): 28.76 tok/s shipped vs 28.57
tok/s batched — neutral, so the variant was NOT merged.

Consistency check on the ~100 ms: an MTP-off decode step (full 27B 4-bit
forward at L=1) is 72.5 ms (13.8 tok/s), i.e. roughly the time to stream
~15 GB of weights once. An L=4 verify measures 103.7 ms, so the marginal
cost of the 3 extra verify positions is ~31 ms (~10 ms/position) of GDN
recurrence GPU work — weight reads are already amortized across positions.

**2026-10-24 update 3: the verify curve is MLX's qmv, and depth is now
chosen online.**

Measured per-round phase costs across depth (code prompt, temp 0, no debug
env; `draft` is perfectly linear at ~4.5 ms/position, which validates the
instrumentation):

| N | L=N+1 | draft ms | verify ms | acceptance | 128-tok wall |
|---|---|---|---|---|---|
| 1 | 2 | 4.9 | 77.0 | 85.5% | 6.00 s |
| 2 | 3 | 9.2 | 84.6 | 77.0% | 5.02 s |
| 3 | 4 | 13.7 | 102.7 | 63.6% | 5.45 s |
| 4 | 5 | 18.1 | 126.5 | 57.1% | 5.96 s |

Verify's marginal cost per position ACCELERATES (+7.6 / +18.1 / +23.8 ms),
which a weight-bandwidth-bound forward cannot do. `src/bench_verify_m.zig`
(scratch, `zig build run-bench-verify-m`) isolates the cause: it is entirely
MLX's 4-bit `quantizedMatmul` as a function of row count M. Summed over one
64-layer forward at the real shapes:

| M | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 14 | 18 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qmm ms | 59.4 | 62.0 | 68.5 | 84.6 | 105.8 | 136.0 | 169.3 | 213.4 | 254.4 | 257.2 | 259.0 | 259.2 |

The qmm marginals (+6.5 / +16.2 / +21.2 at M=3/4/5) match the measured verify
marginals (+7.6 / +18.1 / +23.8) almost exactly — GDN recurrence, `s_all`,
and attention are all noise by comparison. Mechanism, from
`mlx/backend/metal/quantized.cpp` `QuantizedMatmul::eval_gpu`: M below
`get_qmv_batch_limit(K, N)` goes to the batched `qmv` kernel, at or above it
to the split-K GEMM. For K=5120 (> 4096) the limit is 6..12 by arch gen, so
verify always rides `qmv`. **Padding M up to reach the GEMM path does not
help**: the GEMM path only engages at M>=14 for these shapes and plateaus at
~257 ms, ~3x worse in absolute terms than `qmv` at M=4. No *stock* MLX kernel
is cheaper for verify-sized batches.

**RETRACTED (same day): this paragraph originally concluded "~21 ms/position
is intrinsic". That is wrong — it only holds for stock MLX.** omlx ships
custom verify-shape Metal kernels (`omlx/patches/qwen35_verify_qmm.py`) whose
docstring states the same diagnosis we measured ("Stock MLX qmm is tuned for
M=1 and large-M; at M=3..6 it pays a steep per-row penalty") and then fixes
it with four morphologies: split-K (default), msg (N >= 100k, lm_head),
mma (7..32 rows, dequantizes a 16-column weight tile into threadgroup memory
once and multiplies it against every row via simdgroup matrices — a batch of
rows costs ONE pass over the weights), and sg8 (4..8 rows, exact `128 + q`
simdgroup operands in registers, plus a fused gate/up/SiLU launch). Reported:
`lm_head 2.6-3.3x at M=3-4`. They also fuse residual-add with the next
RMSNorm bit-exactly and submit the partial graph every few layers.

Crucially, omlx must gate this to `N >= 16384` because "each
`mx.fast.metal_kernel` invocation pays more Python/dispatch overhead than the
built-in fast path" across ~330 projections per verify. **That is a Python
tax, not a Metal one** — `qwen38_gdn.zig` already calls the mlx-c
metal-kernel C API directly, so dmlx can route every projection. See
`docs/analysis/beat-omlx-mlxvlm-plan.md` §0.5 for the port plan, the
projected round costs, and the licensing/attribution requirements (MTPLX /
dflash-mlx / Splash, all Apache-2.0).

Two knock-on effects: the optimal depth shifts deeper once the per-row cost
curve flattens, and the tree-draft rejection in lever 7 below must be
re-measured because it was derived from the convex qmv curve.

Consequence: the optimal depth is a per-content tradeoff between a fixed,
known, convex cost curve and a content-dependent acceptance curve. Round
economics from the table above (tokens/round = 1 + acceptance x N):

| N | round ms | tok/round | ms/token | model tok/s | measured tok/s |
|---|---|---|---|---|---|
| 1 | 81.9 | 1.85 | 44.2 | 22.6 | 21.3 |
| 2 | 93.8 | 2.54 | 36.9 | 27.1 | 25.5 |
| 3 | 116.4 | 2.91 | 40.0 | 25.0 | 23.5 |
| 4 | 144.6 | 3.28 | 44.0 | 22.7 | 21.5 |

So N=2 wins on a code prompt while N=3 wins on `"Count from 1 to 50."` —
**and the benchmark's long-decode diagnostic uses exactly that counting
prompt**, so the committed "N=3 is the sweet spot" was overfitted to an
unusually predictable prompt. Do not tune depth on that prompt alone.

Fix: `EngineLoop.MtpDepthController` picks the depth per round (see its doc
comment for the estimator design and for the EMA-of-realized-ms/token
estimator that was tried first and FAILED — tokens/round is a high-variance
integer, so per-arm ratio estimates collapse the controller onto depth 1).
The working estimator separates the two factors and pools acceptance
evidence across depths: `cost(n)` as an EMA of round wall time (near
deterministic, +-1 ms; a depth switch was measured to cost nothing, so there
is no switching penalty to amortize), and conditional per-depth acceptance
`p_i` observed by every round that reached depth i, giving analytic
`tokens(n) = 1 + sum_i prod_{j<=i} p_j` for every n including unrun ones.

Final interleaved A/B (3 reps each alternating to cancel thermal drift;
reps were within +-0.01 s):

| prompt | fixed N=3 | adaptive (bound 4) | delta | tok/s |
|---|---|---|---|---|
| code | 5.64 s | 5.24 s | **-7.1%** | 22.7 -> 24.4 |
| count | 4.40 s | 4.48 s | +1.7% | 29.1 -> 28.6 |
| prose | 6.21 s | 5.61 s | **-9.6%** | 20.6 -> 22.8 |

The +1.7% on the counting prompt is the price of the probe rounds; realistic
prompts gain 7-10%. Unit tests 468/469 (1 skipped); benchmark Paris ok,
E2E 7/7, peak RSS 15.0 GiB, long-decode diagnostic 27.7 tok/s (vs 28.8
fixed-N=3 — same counting-prompt effect).

The controller lives on `EngineLoop`, not on the request, so the cost
estimates survive across requests and only acceptance resets
(`resetForRequest`). Confirmed in the benchmark log: later requests show no
calibration rounds at all, just the chosen depth plus the periodic probe
(`MTP depth N=2:11r/34.7ms/tok N=4:1r/40.2ms/tok`). This matters for short
requests, where spending 4 of ~12 rounds calibrating was a third of the
request.

Remaining levers, re-ordered by all of the above:
1. **Bytes per token.** Decode qmm runs at 258 GB/s = ~97% of the DRAM
   roofline, so no kernel work can beat mlx-lm/omlx — the only structural
   win is reading fewer bytes. `src/bench_bits.zig` (scratch,
   `zig build run-bench-bits`) measured the payoff curve at the real decode
   shapes (M=1): 3b/gs128 45.5 ms / 11.2 GB/token vs 4b/gs64 61.9 ms /
   16.0 GB (8b is 1.98x, i.e. time is linear in bytes). The cost model built
   from it predicts 13.3 tok/s for today's config against 14.0 measured, so
   it is trustworthy: 3b/gs128 projects to ~17.1 tok/s decode, clearing
   mlx-lm's 15.6 and omlx's 15.42. Prerequisite already fixed:
   `mlx-zig/src/quantize.zig` restricted affine bits to {4, 8} while MLX
   supports {2,3,4,5,6,8} x gs {32,64,128} (all instantiated in
   `quantized.metal`); widened to match MLX, needs an mlx-zig v0.0.4 release
   since the vendored `zig-pkg/` copy is untracked and regenerable.
2. ~~**Acceptance.**~~ **Dead end — acceptance is NOT the gap.** mlx-vlm's
   qwen3_5_mtp drafter config sets `block_size: 3`, i.e. its default depth is
   2 drafts — exactly what dmlx's controller picks on code. Same drafter
   checkpoint plus same depth means the same acceptance curve (measured
   p1=0.895, p2=0.735, 2.553 tokens/round), so the code-prompt gap (32.5 vs
   27.8 tok/s) is entirely per-round machinery: dmlx's round is 91.8 ms with
   60.9 ms of qmm, mlx-vlm's is 78.5 ms with 67.3 ms of qmm — i.e. dmlx's qmm
   is FASTER (verify kernels) but it carries **19.7 ms/round more non-qmm
   overhead** (draft 9.0, of which ~3.0 ms/step is the drafter's full-vocab
   lm_head; verify non-qmm ~17.0; accept+commit ~0.5). Halving that reaches
   31.1 tok/s, removing it 35.4. Corollary at forced depth 4, dmlx is 25.2 vs
   mlx-vlm 11.4 — **2.2x faster**, so the verify kernels do work; mlx-vlm wins
   on code only by operating shallow where its fixed cost is lower. The
   drafter-training route (MTP-D / FastMTP, days of work for ~+7.5%
   acceptance) is therefore dropped in favour of the 19.7 ms, which is all in
   dmlx's own code. Details: plan doc §21.
3. Drafter-step compile (~4 ms of the 13.9 ms draft phase) — ~3% of a round.
4. `s_all` write elision (~2.6 ms GPU/round, needs failable commit) — ~2%.
5. ~~Compiled verify closures~~ — **dead**, only ~2.8 ms of build exists.
6. ~~Padding M into MLX's GEMM path~~ — **dead**, measured 3x worse.
7. ~~Tree / multi-branch drafts~~ — **dead**. An earlier version of this
   section called a branching verify attractive because "extra rows cost
   ~21 ms while a round costs ~95 ms"; that omitted the break-even
   threshold. Adding a row pays only if expected tokens rise by
   `tokens x c/cost` = 2.5 x 21/95 = 0.55; a second branch at depth 1 covers
   ~8% more probability mass x ~2 continuation tokens = ~0.16. Off by 3.4x,
   and the qmm curve is convex (M=5->6 is +30 ms, 6->8 is +33 ms), so wide
   trees lose harder.

Full plan to overtake both engines, with the measured cost model, the
excluded options and their evidence, and milestone gates:
`docs/analysis/beat-omlx-mlxvlm-plan.md`. Note its P0: the existing
cross-engine numbers are NOT on a common basis (mlx-vlm's are its own
generation tok/s, dmlx's are effective tok/s including prefill and
thinking), and `mlx_vlm` / `mlx_lm` / `omlx` are not currently installed, so
"beat them" is unfalsifiable until a shared harness exists.

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

## MTP round-cost ablation (2026-10-28) — plan doc §22

Priced the per-round overhead by **消除法**: one env switch per candidate, fixed
depth (`DMLX_MTP_FIXED_N=2`, verify M=3), median of `MTP round: ... ms=`.
Tools: `scripts/ab_verify_round.sh`, `scripts/verify_prof_once.sh`.

| removed | switch | round delta |
|---|---|---|
| verify per-position conv-state copies (~290 dispatches/round) | `DMLX_NO_VERIFY_LAZY_CONV=1` (reverse) | −0.3…−0.9 ms (noise) |
| `s_all` L-way state writes (288 MB/round) | `DMLX_VERIFY_NO_SALL=1` (wrong numerics) | −1.3 ms (−1.4%) |
| drafter full-vocab lm_head (636 MB × 2 steps) | `DMLX_MTP_NO_LM_HEAD=1` (wrong numerics) | −7.0 ms (−7.8%) |

Three hypotheses killed:
- **Not dispatch-bound.** Cutting ~290 small dispatches — an order of magnitude
  more than the conv fusion that gave −0.4% — still reads as noise. op-count is
  a dead end.
- **`s_all` has no headroom.** 1.3 ms for 288 MB ≈ 220 GB/s on a 273 GB/s part.
  (The earlier 2.6 ms estimate assumed 380 GB/s, which this machine does not have.)
- **drafter lm_head is at roofline, so FastMTP vocab compression is off the table
  for good.** 3.5 ms/step vs a 2.33 ms floor (636 MB / 273 GB/s) = 75% of peak;
  `bench_verify_kernel` shows stock at 3.22 ms (M=2) and our own sg8 flat at
  3.75-3.95 ms, i.e. we cannot beat stock at M=1 either.

Corrected round budget (N=2, ~90 ms): target verify qmm ~63.7 + drafter 10.2
(of which lm_head 7.0 is itself qmm) + verify non-qmm **~16** + accept/commit 0.4.
§21.2's "non-qmm 30.9 ms" counted the drafter's qmm as non-qmm.

Only remaining candidate for the ~16 ms: the **GDN recurrence kernel**
(`prefill_rec_retain_kernel_body`). Barriered share 9.8% → ~8 ms/round real
vs a 2.1 ms memory floor, and the write side is already proven at bandwidth.
Suspect the kernel shape: 4 state floats per thread, two 32-lane `simd_sum`
per step against ~16 FLOP of useful work. Fixing it is a real kernel rewrite
(wider per-thread state, narrower reduction) worth at most ~6 ms (6.7%).

Landed (performance-neutral, kept for simplicity/instrumentation):
`PrefillVerifyResult.conv_window` + `convStateAt` (conv state sliced at commit,
mirroring the `s_all` path), `Qwen38Mtp.forwardBodyIds` (lm_head split out of
`forwardIds`), and the two diagnostic switches above.

Verification: unit suite 472/473 (1 skip, unchanged); `--speculative-mtp 4`
decode 30.05 / 29.66 tok/s (recorded range 29.3-31.3 — neutral); Paris ✓;
E2E 7/7; peak RSS 15.0 GiB. `run_kernel_tests.sh` fails one DSV4 *performance*
threshold (`down target-shape M=4`, 0.730 vs 0.706 ms, wants ≥5%) — pre-existing
and unrelated.

### Correction (same day): the GDN recurrence costs 0 ms — plan doc §23

The "~6 ms kernel-rewrite headroom" claimed above is **wrong**. Two errors:

1. **No arithmetic check before proposing it.** ~36 instructions per thread-step
   × 196608 threads × 3 steps = 21 M thread-instructions per layer ≈ 5.6 µs on a
   3.8 T/s GPU → **0.27 ms for all 48 layers**. The reduction/compute ratio is
   indeed lopsided, but both sides are irrelevant in magnitude. Ratio describes
   structure, not size.
2. **The 8 ms baseline came from the barriered share**, which §22.6 had just
   declared unusable as an absolute number.

Decisive measurement (`DMLX_VERIFY_NO_RECUR=1`, zero-cost stand-in for the whole
recurrence — 48 launches, 600 MB of state traffic and all ALU removed):
**−0.7 ms, i.e. noise.** The recurrence fully overlaps the qmm work and is not on
the critical path. The 9.8% barriered share was a measurement artifact.

New lesson, stronger than §21.3: barrier-per-section does not merely inflate
absolute times by 37% — **it manufactures hotspots that do not exist**. Its
relative shares must not be used to pick optimization targets.

With all four candidates now priced (dispatches ~0.7 ms / `s_all` 1.3 ms /
drafter lm_head 7.0 ms / recurrence 0 ms) and none recoverable, the residual
re-estimates to **~9 ms** against a full-stock qmm baseline (69.6 + 10.2 + 0.4 =
80.2 vs ~89 measured), spread across prefill-front conv/gates/norms, embedding,
sampling and CPU graph build — no single item worth attacking.

**Honest scorecard: ~89 ms/round vs mlx-vlm's 78.5 at matched depth, and the
11 ms gap resists attribution after four rounds of ablation.** This closes the
"cut our own overhead" direction opened in §21.4. What remains are the two
routes the user already rejected (quantization / vocab compression).

### Found it: the decode GDN fused conv+tail kernel — plan doc §24/§25

§23 closed the "cut our own overhead" direction too early. All four things priced
there were **verify-only** code (per-position conv states, `s_all`, the
recurrence); the largest item lives in the **decode** layer both paths share and
had never been priced on its own.

**The overhead is a per-forward constant, not verify-specific.** `bench_verify_m`
already printed both halves: plain decode (M=1, MTP off) = 72.5 ms wall vs
58.72 ms of qmm → **13.8 ms non-qmm**; verify (M=3) = 84.6 vs 67.90 →
**16.7 ms**. Nearly equal. Same constant also explains the 10% plain-decode
deficit vs mlx-lm (64.9 ms/token, ~6 ms non-qmm). One target, two payoffs.

Ruled out along the way:
- **CPU is innocent.** `DMLX_PROF_APPLY=1`: graph build totals **0.6 ms/token**
  (0.9% of wall). Decode already has exactly ONE sync point per token
  (`asyncEvalCacheStates` + `logits.eval`), so there is no sync-point win either.
  The 78% on "barrier submit" is just where GPU work is forced to materialize.
- **More elementwise fusion is capped at 2.5 ms.** `DMLX_QWEN38_NO_COMPILE=1`
  turns off all four compiled closures (GDN step / MLP swiglu / fused gate_up /
  attn pre-cache): 69.83 → 72.32 ms/token. That is the entire value of the
  existing fusion work.

**The item: `gdnConvTailFused`** — the single non-qmm GPU kernel in a GDN decode
layer (conv + q/k rmsNorm + gates + delta rule + out-gate silu, already one
Metal kernel). `DMLX_DECODE_NO_GDN_TAIL=1` swaps it for a zero-cost stand-in
(all five qmm kept, so bytes/token is unchanged; identical-shape `reshape` is a
free view): **73.65 → 61.57 ms/token, −12.1 ms**, reproduced in all three
interleaved pairs. That is 0.254 ms/layer against a 23 µs memory floor — **11×**.

Root cause is the launch geometry (`qwen38_gdn.zig:811`, and `:335`/`:900` for
the two tail variants): `set_grid(d, hv, 1)` = **6144 threads**, one thread per
state row, each looping over D=128 serially. The prefill/verify recurrence uses
`set_grid(32, d, hv)` = **196608 threads** with a simdgroup per row — and §23
measured that one as free.

**Option A implemented and rejected.** `DMLX_DECODE_WIDE_REC=1` routes the decode
delta rule onto the prefill kernel's wide mapping, which requires bypassing both
monolithic kernels (hence `DMLX_DECODE_EAGER_TAIL=1` to price that separately):
**78.83 ms median, ~5 ms WORSE than mono.** Breaking the fusion costs more than
the re-mapping saves, and it also shows the 12 ms is **not** mostly the
recurrence — the conv/norm/gate/out-gate work is the bulk.

**Option B is the only route: re-map `gdnConvTailFused` in place** — grid
`(32, d, hv)` / tg `(32, 4, 1)`, row work split across lanes with `simd_sum`,
scalar gates on lane 0, fusion kept intact. Upside is bounded by 12.1 ms (the
stand-in deletes the work) and floored by the 1.1 ms memory cost of actually
doing it; landing within 3× of the floor is ~9 ms → ~65 ms/token ≈ 15.4 tok/s,
parity with mlx-lm without touching quality. The kernel is codegen'd
(`createConvTailKernel` / `ConvTailKernelKey`), and the M1 register-spill rule
applies: compile-time-constant indices, no thread-local arrays.

**Measurement protocol (nearly got burned again):** the first mono/eager/wide
batch read 72.0 / 76.9 / 77.4 — but re-measuring mono at the end of the same
sequence gave **77.99**. The machine heats up across consecutive runs and the
whole batch was thermally drifted. Any ms/token A/B must interleave, cool down
between runs, and bracket the sequence with a baseline at both ends. Sequential
blocks (A A B B) are unusable on this machine.

Default path is unchanged (all switches are env-gated and off); unit suite
472/473 (1 skip).

### Test status after §22-§25 (verified, not from memory)

| suite | result |
|---|---|
| `zig build test` | **472/473 pass** (1 skip) |
| └ `test-exact-q4-producer` step | fails under `zig build test`, passes **3/3 standalone** → its 1% timing assertion is squeezed by concurrent test steps; DSV4 `metal_infer` path, disjoint from any change here |
| oracle vs mlx-lm (`compare_qwen38_oracle.py`, 7 segments) | **7/7 PASS**, cos 0.99993-1.00000, max_abs_err ≤ 0.56 on logits |
| **MTP self-consistency (new, `scripts/mtp_self_consistency.py`)** | **5/5 byte-identical** between plain greedy and `--speculative-mtp 4` |
| `run_kernel_tests.sh` | correctness **58/58 OK**; one *performance* gate fails (DSV4 `down target-shape M=4`) |
| benchmark, no speculation | decode 13.07 tok/s, Paris ✓, E2E 7/7 |
| benchmark, `--speculative-mtp 4` | decode 30.05 / 29.66 tok/s, Paris ✓, E2E 7/7, peak RSS 15.0 GiB |

**New: `scripts/mtp_self_consistency.py`.** At temperature 0 speculative decoding
must emit exactly the greedy sequence (verify accepts a draft only when it equals
the target's argmax). This is the only real-dimension test covering
`commitVerifyStates` / the lazy conv-window path (§22.5) and the verify_qmm
kernels — the existing unit test runs at toy dims (D=4), where the fused kernels
are not even eligible. Runs the two configs serially, never two serves at once.

The DSV4 kernel perf gate flips direction between runs (earlier: bucket 0.730 vs
scalar4 0.706, i.e. slower; now bucket 0.929 vs 0.975, i.e. faster but only 4.7%
against a 5% gate), so it sits on the threshold rather than regressing.

**One false green worth recording.** The first version of the self-consistency
probe was a shell script; zsh mangled the nested quoting in the JSON builder, no
request was ever sent, both output files were empty — and the script reported
`PASS 5/5`, because empty output is trivially identical. The Python rewrite
asserts non-empty completions and exits 2 (harness failure, distinct from
divergence) if any completion is under 2 characters. **A comparison harness must
fail on absent data, not pass.**

Known gaps: the diagnostic switches (`DMLX_VERIFY_NO_SALL`,
`DMLX_MTP_NO_LM_HEAD`, `DMLX_VERIFY_NO_RECUR`, `DMLX_DECODE_NO_GDN_TAIL`,
`DMLX_DECODE_WIDE_REC`, `DMLX_DECODE_EAGER_TAIL`) are deliberately
numerics-breaking and have no tests; nothing automatically guards the §25
finding, so a future change to `gdnConvTailFused`'s launch geometry would need
the ablation re-run by hand.

### omlx native MTP now works — scorecard retracted, we are last on the speculative axis (plan doc §26)

The old reason omlx ran as a no-speculation row ("the community 4-bit export
stripped the `mtp.*` weights") was **a wrong diagnosis**. The weights ship as a
separate checkpoint (`Qwen3.8-27B-MTP-4bit`) in FLAT layout (`fc.weight`,
`layers.0.*`, `norm.weight`, `pre_fc_norm_{hidden,embedding}.weight`), and
omlx's `oq.combine_mtp_donor` only accepts donors whose keys already carry an
`mtp.` prefix. omlx's own head module is built from exactly those five
component names, so `mtp.` + flat key is the layout it wants.

`scripts/omlx_enable_native_mtp.py` normalizes the donor, builds a recipient
whose 14 GB of backbone shards are **symlinks** (~240 MB total cost), and calls
omlx's own `combine_mtp_donor` so the per-layer quantization entries and the
raw-HF-vs-MLX RMSNorm `+1` rule are handled by omlx rather than by hand. Every
`validate_mtp_donor_pair` precondition already held (family `qwen3_5` both
sides, `mtp_num_hidden_layers=1`, all geometry fields equal, byte-identical
`tokenizer.json`). Originals untouched. omlx then auto-sets `mtp_enabled` and
logs `Lightning MTP (qwen3_5, active)` + `MTP path activated`.

Validated before trusting any number: `scripts/omlx_mtp_consistency.py` checks
that omlx on the grafted checkpoint (MTP active) is **byte-identical at
temperature 0** to omlx on the original (plain decode) — the backbone shards are
the same files. **3/3 identical**, and the script fails if the log shows no MTP
activation. Without this, a graft that weakened the drafter would have looked
like a slower competitor and flattered us.

**Six-engine, common basis (interleaved, one server at a time):**

| axis | dmlx | best competitor | our rank |
|---|---|---|---|
| speculative decode (HEADLINE) | 28.5 | **omlx-mtp 33.2** (mlx-vlm 28.2) | **3/3, −16%** |
| plain decode | 13.7 | omlx 15.4 (mlx-lm 15.3) | **3/3, −11%** |
| TTFT fixed overhead | **0.281s** | mlx-vlm 0.386s (omlx 0.773s, mlx-lm 1.078s) | **1/5, 2.7-3.8× ahead** |
| prefill throughput | 112.4 t/s | omlx 137.0 | **5/5, −18%** |
| ttft@512 | 4.513s | mlx-vlm 4.290s | 2/5 |
| ttft@2048 | 18.210s | omlx 15.955s | 5/5 |

**RETRACTED:** the earlier headline "dmlx 29.3 / mlx-vlm 28.8 / mlx-lm 15.9 /
omlx 15.2 (tied for first)". That tie existed only because omlx could not
speculate.

**This reopens §21.4.** Per-prompt-class: omlx-mtp leads on predictable content
(code +29%, math +30%, count +43%) while **we lead only on prose** (25.7 vs
22.3). That is an acceptance/effective-depth difference, not a per-step speed
difference — at forced matched depth we measured 2.2× faster than mlx-vlm
(§21.3). §21.4 concluded "same drafter, same depth ⇒ identical acceptance, so
acceptance is not the gap", but omlx uses the **same drafter weights** and still
gets +30% on predictable text (count 50.2 vs our 35.2 is hard to explain any
other way than deeper chains). So that conclusion holds for mlx-vlm only, not
for omlx: next step is to read omlx's depth policy
(`mtp_adaptive_max_depth`, `patches/mlx_lm_mtp/batch_policy.py`) against our
`MtpDepthController`'s climb behaviour when p_i is high.

Meanwhile §25 remains the only known quality-neutral win on the plain-decode
axis: −12 ms/token would take dmlx-nospec 13.7 → ~16.3, past omlx's 15.4.

### §26.4's "deeper chains" claim is wrong — the gap is cost(n) slope (plan doc §27)

omlx's own log prints its depth distribution and it stops at **d4** — the same
ceiling our `--speculative-mtp 4` used (`mtp_adaptive_max_depth` defaults to 4
for Qwen 27B). The "deeper chains" story came from comparing OUR code-prompt
numbers against THEIR prose-prompt log. `scripts/mtp_round_decompose.py` now
puts both on the same prompt and splits the rate into its two factors.

**Same prompt (code, max_tokens=128, both capped at depth 4):**

| | dmlx | omlx-mtp |
|---|---|---|
| ms / round | **91.95** | **91.78** |
| tokens / round | 2.65 | **3.20** |
| mean depth used | 2.30 | ~4 |
| ms / emitted token | 39.7 | 31.2 |

**The same round cost buys them depth 4 and us depth 2.3.** Their cost(n) is
flat in depth; ours rises ~14 ms per level.

**The depth controller is not broken.** Forcing 3 deep rounds per request
(`DMLX_MTP_EXPLORE_ROUNDS`, now default **0**) raised tokens/round 2.65→2.93 and
depth 2.30→3.00 but ms/round 91.9→111.4, and the realized rate got **worse**
(39.7 → 43.5 ms/token). Its shallow choice was correct for our cost curve. Kept
as a tunable because the optimum moves up the moment cost(n) is flattened.

**Where their flatness comes from** (`batch_generator.py` `_CoarseDraftHead`):
a one-time **3-bit gs128 copy of lm_head** drives the draft chain, then only the
**top-64** candidates are rescored against the exact head's gathered rows. This
is strictly quality-neutral — a draft is only a guess, acceptance is decided by
the target's own argmax, so a coarser head can lower acceptance slightly but
**cannot change the temperature-0 output**; the top-64 exact rescore removes most
of even that loss.

Budget it by BYTES, not by their timer label: their `mtp=1.65ms` covers only the
head's 1-layer transformer, not the coarse lm_head (which lands in `backbone`).
I first misread that as 14× cheaper drafting.

| | weights | scales+biases | total | @231 GB/s |
|---|---|---|---|---|
| ours 4-bit gs64 | 636 MB | 79 MB | 715 MB | 3.10 ms/step (matches measurement) |
| 3-bit gs128 coarse | 477 MB | 40 MB | 517 MB | **2.24 ms/step** |

**Ranked, quantified, all quality-neutral (per round at depth 4):**

| item | gain | status |
|---|---|---|
| coarse 3-bit draft head + top-64 exact rescore | ~5 ms | not implemented, technique confirmed |
| drafter body dtype round-trips (`rmsNormBf16` does astype→norm→astype, 4 norms/step; theirs 0.41 ms/step vs our 1.6 ms) | ~4.8 ms (est) | unpriced |
| route verify `lm_head` to sg8 (`DMLX_VERIFY_QMM_LM_HEAD=1`; stock climbs 3.22→6.59 ms over M=2..5, sg8 is flat at 3.75) | ~2.8 ms | switch exists, needs the self-consistency probe first |
| | **~12.6 ms** | |

Flattening cost(n) is **multiplicative**: ms/round falls AND the controller then
picks deeper, so tokens/round climbs toward 3.20. But arithmetic says it is not
sufficient on its own — depth 4 at ~112 ms with 3.19 tok/round is 35.1 ms/token
vs omlx's 31.2, so part of §24's ~14 ms per-forward constant still has to come
back. §24/§25 and §27 are two fronts of the same fight, not alternatives.

Unit suite after the revert: 472/473 (1 skip).

**Method note:** comparing a single metric across different prompts manufactured
a causal story ("deeper chains") that did not exist. Decompose into
tokens/round × ms/round on an identical prompt before attributing anything.

### §27.5's three items, verified one by one — two hold, one retracted, net is break-even (plan doc §28)

**(c) verify `lm_head` routing — downside removed, still default-off.**
The fidelity gate passed (`mtp_self_consistency.py` 5/5 byte-identical) but the
performance was *negative* at the depth the controller actually picks: M=3 gave
86.5/87.6 ms default vs 90.5/87.8 ms routed, because sg8 needs M>=4 so at M=3
`lm_head` fell back to **split-K**, which the bench already showed loses on that
shape (stock 3.22/3.82/4.80 vs kp=2 3.22/4.11/5.59 at M=2/3/4). Fixed by making
`lm_head` **sg8-only** in `verify_qmm.pick`; M=3 is now neutral and the gain
lands only at M>=4 (M=5: 6.59 → 3.78 ms, −2.8 ms/round). Left default-off: five
prompts x 64 tokens is not enough to clear a change that alters the *emitted*
token (the ~0.07 logit deviation is the size of an argmax-flipping top-2 gap).

**(b) drafter body dtype round-trips — RETRACTED.** Two errors in my own
estimate. (1) Bytes forbid it: the whole drafter checkpoint is **228 MB**, so a
body step has a 0.99 ms floor at 273 GB/s and our measured 1.6 ms is already
62% of roofline — at most 0.6 ms/step of headroom. omlx's 0.41 ms/step would
need 556 GB/s, i.e. **impossible**, so their `mtp=` timer covers neither the
lm_head nor the body's GPU time (same misreading as §24.2). (2) Wrong magnitude:
8 extra small ops per step at ~2.4 us is **0.02 ms**, not 1.2 ms.
*Lesson: never budget from a competitor's timer label without a byte-level
feasibility check first.*

**New finding: the verify kernels deliver only part of their benchmarked gain.**
A/B against `DMLX_NO_VERIFY_QMM=1`: −2.8 ms at M=4 and −19.1 ms at M=5, versus
bench predictions of −16.6 / −27.7. Measured cost(n) per round
(`DMLX_MTP_FIXED_N`): 77.0 / 93.0 / 112.1 / 122.3 ms at N=1..4.
**It is not a routing-coverage problem** — new `DMLX_VERIFY_QMM_TALLY=1`
(`scripts/verify_qmm_tally.sh`) tallies hits per (M,K,N) and at M=5 everything
lands on sg8 except `lm_head` (and the sub-`min_route_n` tiny GDN projections,
skipped by design). So the shortfall is in-situ efficiency — the same effect as
§24.4's isolated-qmm 245 GB/s vs in-situ 206 GB/s — not something routing can
recover.

**Totals at depth 4:** lm_head sg8 −2.8 ms + coarse 3-bit draft head −4.4 ms =
**−7.2 ms**, taking the round 122.3 → ~115 ms. With tokens/round 3.30 that is
**34.8 ms/token**, against our current depth-2.3 figure of **34.7**. Both items
together only make depth 4 break even; they make no real progress against
omlx's 28.7.

**So the priority is unchanged and it is §25, not §27.** Subtracting qmm (stock
column) and the drafter from cost(n) leaves a ~12 ms **constant** present at
every depth (13% of a round) — the same per-forward constant §24.1 measured in
plain decode. In plain decode it is located and priced (§25: `gdnConvTailFused`
thread mapping, 12.2 ms, fix known = option B, re-map grid to `(32,d,hv)` in
place). In MTP rounds it is **not** located, because verify runs the prefill
path whose GDN mapping is the good one and whose recurrence §23 proved free.

Do §25 option B first (biggest, known fix, plain-decode axis), then locate the
verify-path 12 ms; treat the 7.2 ms above as follow-on. Doing it in the other
order spends the coarse-head implementation effort to buy a break-even.

Unit suite: 472/473 (1 skip).

### §25 option B executed: v4 wide tail is now the default, worth ~3% (plan doc §29)

`gdnConvTailWide` + `wide_tail_kernel_body` / `out_gate_kernel_body` in
`qwen38_gdn.zig`. **Default on**; `DMLX_DECODE_NO_WIDE_TAIL=1` restores v3.

v4 flips the geometry from grid `(D, HV)` / 6144 threads / one thread per state
row to grid `(32, D, HV)` / tg `(32, 4, 1)` = **1536 threadgroups**, with lane
`dk` owning float4 `dk` of the row so a simdgroup issues one coalesced 512 B
transaction instead of 32 separate 128 B lines (the 8x read+write amplification
§25.2 identified). The two row-length dot products become `simd_sum`, as does
the q/k RMSNorm reduction.

Two consequences handled: a threadgroup now holds only 4 state rows but still
needs the head's full D-length q/k, so phase 1 recomputes the whole head's conv
with one channel per thread (tg size == D == 128) — free in DRAM terms because
`conv_state`/`conv1d_t` are ~123 KB and stay L2-resident, with `conv_out` writes
gated to the first row-block so they happen once. And the out-gate RMSNorm spans
the head's whole y row, so it moved to a second cheap kernel. Requires D == 128;
anything else falls back to v3.

**Gain: ~2.2 ms/token (3.1%), which is SMALLER than this machine's noise.** Two
independent interleaved+cooldown experiments, 6 pairs total:

| | v3 | v4 |
|---|---|---|
| exp 1 | 73.98 / 76.04 / 68.55 | 71.80 / 70.73 / 67.63 |
| exp 2 | 71.84 / 67.48 / 70.91 | 71.99 / 67.30 / 67.43 |
| **median of all** | **71.4** | **69.2** |

v4 wins 4 pairs, ties 1, loses 1 by 0.15 ms. Reporting only exp 1 would have
claimed 3.25 ms. The benchmark's single-shot long-decode reading is useless here
(13.705 v4 vs 13.652 v3 — no visible gain at all).

**Why so far below the 13.9 ms headroom.** `DMLX_DECODE_NO_GDN_TAIL=1` is
60.1 ms; v4 is 69.2. Costing v4 by bytes gives only ~2 ms (state 1.05 + conv
redundancy 0.6 + extra launch 0.24 + conv DRAM 0.04). **The remaining ~7 ms is
not work — it looks like dependent kernel-launch latency**: each GDN layer is a
serial `qmm → conv_tail → out_gate → qmm` chain, and v4 makes that **two**
dependent kernels per layer, i.e. 96 dependent launches per token; at ~70-100 us
each that is 7-9.6 ms. This also explains §24.3 (four compiled fusions worth only
2.5 ms): they cut elementwise op *count*, not dependency *depth*.

**Real tension:** the wide mapping forces the out gate into its own kernel (its
reduction spans a head), while cutting dependent launches wants one kernel. The
only route that fixes coalescing WITHOUT adding dependency depth is transposing
the SSM state layout to `[HV, D/4, D, 4]` so v3's single-kernel shape coalesces
too — but that touches every decode/prefill/verify kernel, the eager fallbacks,
`commitVerifyStates`, cache init and the state-comparing tests.

**New gate, and it was mandatory:** `scripts/decode_env_ab_text.py` byte-compares
greedy output under two env configs. The GDN decode kernels are the one part of
the model **neither** existing gate covers — unit tests run at D=4 where
`gdnConvTailWide` is ineligible, and the mlx-lm oracle dump is a *prefill* pass.
Result **6/6 byte-identical**.

All gates: unit 472/473 (1 skip); oracle 7/7 (cos 0.99993-1.0); MTP
self-consistency 5/5; benchmark no-spec decode 13.7-14.0, Paris ✓, E2E 7/7;
benchmark `--speculative-mtp 4` decode **30.859** tok/s, Paris ✓, E2E 7/7,
peak RSS 15.0 GiB.

**One unreproduced anomaly, logged not dismissed:** one no-spec benchmark run
reported E2E 3/7 with `expected 'Paris', got ''`. A clean re-run gave 7/7 for
**both** v4 and v3. Empty strings point at client timeout rather than numerics (a
numeric fault yields *wrong* text, not none), and that run followed the oracle
dump process while memory was still being released. If it recurs, check the E2E
script's `--max-time`.

**Effect on the goal:** plain decode 13.7 → ~14.1 tok/s (+3%), still **8%**
behind omlx's 15.4. §25.4's estimate of "13.7 → 15.6-16.1, parity with mlx-lm"
did **not** materialise: it assumed ~9 of the 12.2 ms was recoverable, and only
2.2 ms was. The dominant term is dependency depth, not bandwidth.

### CORRECTION: §25's 12.2 ms was a measurement error — the real figure is 1.3 ms (plan doc §30)

Before committing to the SSM-state-layout transpose (large blast radius) on the
strength of §29.3's *inferred* "dependent launch latency" attribution, I priced
dependency depth directly. Two things collapsed.

**1. Dependency depth costs nothing.** `DMLX_DECODE_EXTRA_LAUNCH=N` inserts N
numerically-identity pass-through kernels into each GDN layer's serial chain
(+48 dependent launches per token each):

| extra | +launches/token | ms/token (2 runs) |
|---|---|---|
| 0 | 0 | 69.08 / 69.53 |
| 1 | +48 | 70.13 / 67.82 |
| 2 | +96 | 72.39 / 67.55 |
| 4 | +192 | 68.68 / 68.33 |

Zero slope. 192 extra dependent launches are free, so a dependent launch costs
**< 5 us**, not the 70-100 us §29.3 assumed. That attribution is retracted.

**2. The ablation behind all of §25 silently removed a qmm.** In the
`DMLX_DECODE_NO_GDN_TAIL` stand-in, `y_sub` was built from `z` alone, leaving
`qkv_lin` **unconsumed** — and MLX is lazy, so the whole `in_proj_qkv` qmm
([10240, 5120] 4-bit = 29.5 MB/layer = **1.42 GB/token ≈ 6.1 ms**) was pruned
along with the tail. §25.1 even claimed "all five qmm are kept, so bytes/token
is unchanged"; that sentence checked only the arrays I returned, not which
*inputs* had become dead code. Fixed by consuming both projections
(`add(slice(qkv_lin, 0..HV*D), z)`, ~24 KB).

**Corrected measurements (3 interleaved passes, cooldown):**

| | median ms/token |
|---|---|
| v4 (default) | **68.32** |
| v3 | 69.29 |
| tail removed (fixed stand-in) | **67.04** |

- The entire fused conv+tail kernel costs **1.3 ms/token**, not 12.2 — §25.1 was
  inflated ~9x.
- **v4 beats v3 by 1.0 ms, winning all three pairs**, i.e. it captures 77% of the
  headroom that actually exists. v4 stays the default (byte-identical output,
  6/6; better threadgroup scaling per §25.7), but it is a 1.4% win, not 3%.

**Retracted:** §25.1 ("12.2 ms, largest single inefficiency in the engine"),
§25.4 ("13.7 → ~16.3 tok/s, past omlx"), §29.2 ("v4 worth 2.2 ms"), §29.3
("remaining ~7 ms is dependent launch latency").

**Corrected global picture.** Plain decode is now 68.3 ms/token against a
58.72 ms qmm floor, so non-qmm is **~9.6 ms**, made up of: GDN conv+tail 1.3,
dispatch/dependency ~0, CPU graph build 0.6, and **~7.7 ms spread across the 16
attention layers, MLP activations, norms, residuals, embed and sampling** with no
single item above ~1-2 ms. That is the opposite of the "there is one 12 ms
hotspot" picture §24.1-§25 built. **The SSM-state-layout transpose proposed in
§29.3 is no longer worth doing** — its entire rationale was recovering the 8x
amplification, and with the kernel at 1.3 ms total and v4 already holding 1.0,
the remaining ceiling is 0.3 ms.

**Method rule added.** This is the second time lazy evaluation produced a wrong
number this session (§19.2 was the mirror image: a segment timer that failed to
force one output charged its qmm to the next segment). When writing an ablation
stand-in, **enumerate every input of the replaced block and confirm each is still
consumed downstream** — checking the outputs is not enough. The cheap self-check
is that the stand-in must read the same weight bytes per token; computing "does
the no-tail path still read in_proj_qkv's 29.5 MB/layer?" would have caught this
in minutes, instead of invalidating most of §25-§29.

Gates after the fix: unit 472/473 (1 skip), `decode_env_ab_text.py` 6/6
byte-identical.

### Optimization exhausted at the measurable op level (plan doc §31-§32)

**Landed this round**

1. **v4 wide GDN conv+tail mapping** (default; `DMLX_DECODE_NO_WIDE_TAIL=1`
   reverts) — 1.0 ms of the 1.3 ms that kernel actually costs; 6/6
   byte-identical.
2. **Coarse 3-bit draft head** (default; `DMLX_MTP_NO_COARSE_HEAD=1` reverts) —
   a 3-bit gs128 copy of lm_head proposes draft tokens, then the **top-64** are
   rescored against the exact head's gathered rows. **0.8 ms/step** (N=2:
   92.8→91.3; N=4: 129.8→126.5), matching the byte prediction (3.10→2.24 ms).
   Strictly quality-neutral because a draft is only a guess and acceptance is
   decided by the target's own argmax — confirmed by **MTP self-consistency
   5/5 byte-identical**. Costs +430 MB resident (peak RSS 15.0 → 15.4 GiB).
   Ported from omlx `batch_generator._CoarseDraftHead`.
3. `lm_head` made sg8-only in `verify_qmm.pick` so routing it can no longer
   lose at M=3. Still default-off: the gain only exists at M>=4 and the
   controller picks M≈3, plus it changes the emitted token.

**Plain-decode levers: all measured, all ~zero**

| lever | measured |
|---|---|
| kernel launch / dispatch count | **0** (+192 identity launches/token are free) |
| dependency depth | **0** (same probe) |
| **qmm call count** | **0** (`DMLX_QWEN38_NO_FUSE_GATEUP=1`, +64 calls/token: 72.07 vs 72.13) |
| GDN fused conv+tail | 1.3 ms (1.0 captured by v4) |
| attention non-qmm (16 layers) | 0.6 ms |
| CPU graph build | 0.6 ms |
| glue (128 addRmsNorm + 64 swiglu) by bytes | **0.04 ms** |

Notably the existing `fuseGateUp` has no measurable value, which also kills the
most promising-looking idea (fusing GDN's four input projections to remove 144
qmm calls) — calibrated at zero.

14.42 GB must stream per token: isolated qmm bench 58.7 ms (245 GB/s), mlx-lm
65.4 (222), dmlx ~72 (206). The ~7 ms gap to mlx-lm **is not attributable to any
ablatable op-level item.** The only remaining mechanism that could explain
"same bytes, same kernels, 7% less bandwidth" is weight memory layout / DRAM page
locality (we allocate per tensor; a per-layer arena would change it). That cannot
be priced by ablation — it has to be built first — so the risk/benefit is poor.

**Speculative axis standing** (same code prompt): ms/round dmlx **91.6** vs
omlx 96.6 — **we are now faster per round**. The entire remaining gap is
tokens/round (2.49 vs 3.20), and §27.3 measured that forcing ourselves deeper is
net-negative (cost(n) slope ~13-14 ms/level). Unexplained: their conditional
acceptance *rises* with depth (0.757/0.821/0.826/0.941) while ours falls
(0.863/0.682). Candidate causes are their prompt priming (`primed=62`) and how
the chained hidden is sourced — **unverified; do not act on it**, every
unverified attribution in this session was overturned by measurement.

**Two invalid-ablation rules learned (both cost real retractions)**
- Every input of the replaced block must still be consumed, or MLX's lazy graph
  prunes a qmm and inflates the result (§30: §25's 12.2 ms was really 1.3 ms).
- The stand-in must keep values bounded. `DMLX_DECODE_NO_NORMS` made the
  residual stream blow up and read **+22 ms** — an ablation that makes things
  slower is measuring its own stand-in, not the removed code.

Gates: unit 472/473 (1 skip), oracle 7/7, decode text A/B 6/6, MTP
self-consistency 5/5, benchmark spec decode 29.5 tok/s / Paris ✓ / E2E 7/7.

### Acceptance mystery closed; batched commit fold; depth>7 rejected by data (plan doc §43, 2026-10-29)

Commit `ee8cee2`. Session goal: close the speculative-decode gap to omlx-mtp.

- **Re-baseline** (interleaved, nonce, 3 reps; omlx 0.7.1 unchanged): dmlx 34.0
  vs omlx-mtp 41.3 vs mlx-vlm 30.7. The user's table (46.7) did not reproduce.
  NOTE: omlx-mtp's readings climb across a session (33.2 → 41.3 → 46.3 over
  three days) with prefix cache ruled out (§40.1) — mechanism still unknown;
  treat its absolute number as a moving target.
- **Chain-state audit** (`DMLX_MTP_CHAIN_DUMP=1` in model_registry.zig): every
  position invariant holds; RoPE offset == drafter cache.len(). Excluded.
- **Token-level drafter oracle** (`OMLX_CHAIN_DUMP` patch in ~/work/code/omlx,
  env-gated, uncommitted there): with the SAME raw prompt both engines emit
  byte-identical text; at matched positions draft mismatches are symmetric
  (5-5-3). **Same drafter weights → equal quality; the "p3 decay" (§34-§35)
  was a depth-mix/statistics artifact, not a defect.** Coarse-head bits and
  verify kernels measurably do NOT affect acceptance (4-way A/B identical
  p_i) — more evidence the coarse head is strictly neutral.
- **Trunk post-norm head input** (`DMLX_MTP_TRUNK_POSTNORM=1`, omlx's
  HEAD_HIDDEN_POST_NORM convention): fixed-depth p_i improved on code
  (E[tok] 3.41→3.60) but the interleaved harness says TIE (35.2 vs 35.0).
  Default OFF. Lesson: fixed-depth p_i wins do not guarantee end-to-end wins.
- **Batched commit fold**: commit replay of k accepted drafts is now ONE
  prefill-mode head forward (omlx fold semantics; `DMLX_MTP_NO_BATCH_FOLD=1`
  reverts). Interleaved harness A/B: fold ON 31.7 vs OFF 31.0 (prose +8%),
  i.e. neutral-to-positive — kept as default. Gates: self-consistency 5/5
  byte-identical, fidelity NEUTRAL (identical divergence positions),
  unit 464+1skip, benchmark Paris ✓ + E2E 7/7, RSS 15.0 GiB.
- **Depth 8-12 probed** (stock-priced rows, `DMLX_MTP_FIXED_N=1 --speculative-mtp 12`):
  p4 is the cliff (~0.54-0.64), p8+ starves; E[tokens] gain ≤ +0.5/round even
  if deep rows were free → **mma kernel port rejected by data** (plan doc §43.5).
- **Math gap attribution**: ms/round EQUAL (93.05 vs 92.99); the residual is
  tokens/round (3.56 vs 3.88) driven by cost(n) slope — all attributed items
  are at their measured floors.
- pre-commit hook fix: a failing test binary aborted the scan silently under
  `set -e` (`.git/hooks/pre-commit`, added `|| true` — .git is untracked,
  reapplied manually if hooks are reinstalled).

**Watch item:** cross-session dmlx drift this day: 34.0 (morning) → 31.7
(evening) with both fold arms equal in-session — ambient/thermal state, not
the fold. Compare same-session only.

## 2026-10-30 — catch-omlx-mtp: leak fix + chat-template parity (branch glm53-mlx-support @ 732316a)

- **Metal buffer leak fixed** in `qwen38_gdn.commitVerifyStates` (replay branch never
  freed the five retained recurrence inputs): +0.68 GB/request of leaked MLX active
  memory, progressive verify slowdown (95 ms -> 41 s/round), then metal::malloc /
  shared-event crashes at ~20 sustained MTP requests. Found with new
  `DMLX_MEM_PROBE` / `DMLX_MEM_PROBE_ROUNDS` instrumentation. Verified dead-flat
  over 25 temp-0.7 requests; self-consistency 5/5; unit 472/473+1skip.
- **Root cause of the omlx-mtp "gap" decomposed**: NOT weights (same drafter
  checkpoint, p_i equal at fixed depth 4: 0.87/0.78/0.80/0.78 vs 0.77/0.79/0.73/0.79),
  NOT controller (tokens/round 3.06 vs 3.04), NOT base decode (both ~93% of the
  231 GB/s roofline). Two real confounds: (1) chat template — omlx sends
  enable_thinking=false (empty think block), dmlx free-thought (fixed via
  `DMLX_CHAT_EMPTY_THINK=1` / `dmlx-nothink` harness arm); (2) our progressive
  slowdown from the leak. Residual real gap after both: ~7-10% steady-state decode
  (verify marginal ~3 ms/round at depth 3-4 + their lower host overhead); within
  cross-session drift. Effective throughput (incl TTFT) is a tie: 0.28s vs 0.8-1.2s TTFT.
- MTP-D self-distillation pipeline built but ON HOLD per user decision
  (scripts/mtpd_*.py, plan docs/superpowers/plans/2026-10-30-mtp-drafter-training.md).
  Corpus: artifacts/mtpd/corpus.jsonl (183 sequences).

## 2026-11-02 — fusion program refuted; trunk-postnorm flip (prose gap root cause)

- Session state: branch glm53-mlx-support, commits 490ef5f..de0ac2c. The
  whole close-decode-gap ledger is in
  docs/superpowers/plans/2026-10-30-close-decode-gap.md (2026-11-02 entries).
- **Fusion program (F1 sg8 group / F2 sg8 swiglu) ported from omlx and
  REFUTED by A/B at the parking depth** (worse at M=4, better at M=6 but
  deep chains still lose ms/token). Both kernels kept, OPT-IN via
  DMLX_VERIFY_GROUP_SG8=1 / DMLX_VERIFY_SWIGLU_SG8=1.
- **prose-gap root cause = trunk-hidden norm convention.** omlx feeds the
  MTP head the POST-final-norm trunk hidden (HEAD_HIDDEN_POST_NORM=True);
  we fed pre-norm. Flipped to default on (DMLX_MTP_NO_TRUNK_POSTNORM=1
  reverts): prose p1 0.64->0.78 (omlx 0.74), code p4 0.27->0.68,
  tok/round +7-12% at equal round cost. Harness effective: code and math
  flipped to WINS (+1.1% / +1.3%), prose gap halved (-29% -> -12%).
- New toggles for future A/B: DMLX_MTP_REPLAY_FREE=1 (omlx replay-free
  commit; measured NOT the acceptance difference).
- Harness methodology: compare_engines.py now records per-request raw
  samples (per_request_raw) — the best-of-N per-prompt pick compresses
  variance and hid the prose gap's true size. omlx decode numbers have
  stdev 5-10 tok/s vs our ~1; their headline is max-tail biased.
- Latent bug fixed: prefillVerify errdefer+defer double-free of y
  (segfaulted the server when a Metal compile failure propagated).
- verify_qmm test-gate repair: CPU-stream guard via mlx_stream_equal
  (mlx_stream_get_device SEGFAULTS in mlx-c 0.7.0 — do not use it).

### Final scorecard (2026-11-02, post postnorm+explore=9; raw-sample means, effective tok/s)

| prompt | dmlx (before -> final) | omlx-mtp (3 runs) | verdict |
|---|---|---|---|
| code  | 34.7 -> **42.7** (+23%) | 35.9 / 37.8 / 33.9 | **dmlx wins** |
| prose | 21.7 -> 22.3 (+3%) | 30.7 / 25.3 / 23.8 | -6%, inside omlx's own cross-run swing |
| math  | 33.9 -> 33.9 | 36.6 / 33.8 / 37.0 | tie (omlx swings 33.8-37.0) |

HEADLINE decode: 33.9 -> 38.0 (theirs 50.7/46.4/45.8, still max-tail biased).
omlx's cross-run spread is +/-10-20% per domain; dmlx's is +/-0.5 tok/s.
TTFT unchanged: 0.28s vs 0.78s; prefill 123 vs 141 tok/s (they keep the
prefill edge at 2048: 17.2s vs 15.7s TTFT).

### 2026-11-02 (later) — content-parity fix + fair-harness final state

- **The prose/math "gaps" were partly a CONTENT mismatch**: omlx defaults
  to thinking-ON + reasoning-effort system prompt (66-70 token prompts,
  110/128 reasoning tokens on prose); dmlx-nothink decoded pure answer
  prose. Fixed harness-side with the omlx-mtp-nothink arm
  (enable_thinking=false). Fair 6-rep: code eff +10.9% (we win),
  prose -8.6%, math -14.0%.
- **ChatML template now matches the official jinja byte-for-byte** (6/6
  cases vs HF apply_chat_template): default = system prompt + thinking ON;
  DMLX_CHAT_EMPTY_THINK=1 keeps the empty-think arm. NOTE: server default
  behavior for Qwen3.5 changed to thinking-on (HF parity).
  DMLX_CHAT_REASONING_EFFORT selects xhigh/medium/low.
- Residual honest gaps (fair basis): prose -8.6% / math -14% effective.
  math decomposes to the known verify marginal at depth 4-5 + deep-tail
  acceptance measured on non-alignable streams (backbone greedy diverges
  at char ~36 cross-engine, so per-position draft comparison is
  impossible on this content). Partial-graph flush tried, neutral, opt-in
  (DMLX_VERIFY_FLUSH=1).
- Self-consistency byte-exactness is content-length-bound: under
  thinking-on content it hits the documented fidelity class (near-tie
  flips). Gate = mtp_fidelity_profile (currently neutral: median
  first-divergence 209, 4/6 identical, both arms equal).

### FINAL fair scorecard (2026-11-02, both engines empty-think, 6-rep interleaved, raw-sample means)

| prompt | dmlx | omlx-mtp | delta |
|---|---|---|---|
| code  | 42.3 (dec 47.7) | 38.3 (dec 44.6) | **+10.3% dmlx wins** |
| prose | 22.7 (dec 25.2) | 23.9 (dec 25.5) | -5.3% |
| math  | 34.1 (dec 38.7) | 39.9 (dec 46.8) | -14.5% (real, see below) |
| count | 54.0 (dec 61.6) | 42.3 (dec 50.5) | **+27.7% dmlx wins** |

TTFT 0.28s vs 0.78s; prefill 122.8 vs 138.5 tok/s (they keep the long-prefill
edge: ttft@2048 17.5s vs 15.9s).

math residual (-14.5%) decomposes to: (a) the known verify marginal row at
depth 4-5 (~5.5 ms/round), (b) per-request calibration cost — each fresh
request pays explore rounds, and the harness restarts servers per visit so
no warm-up carries (their controller starts deep and sheds, paying nothing
upfront on easy content). math probes within a warm session measured 39.5
vs their 39.3 (parity) — the harness gap is partly cold-start calibration.

## 2026-11-05 — thinking-regime parity -> win: round-cost attribution and the fused verify tail

The aligned thinking pair (`dmlx-think-aligned` vs `omlx-mtp-aligned`,
SAME reasoning directive on both arms so content AND prompt length match)
opened the day at **-4.4%** (34.9 vs 36.5) and closed at **+8.2%**
(39.7 vs 36.7; code +12.9%, prose +6.8%, math +3.7%). Full attribution
chain, each step measured:

1. NOT drafter weights: dmlx's drafter (`~/models/Qwen3.8-27B-MTP-4bit`)
   and omlx's (`...-4bit-mtp/model-mtp.safetensors`) are the same 31
   tensors, byte-identical.
2. NOT acceptance semantics: the harness sends temperature 0, so both
   engines run greedy longest-prefix walks (ours: batched argmax compare
   in engine_loop.zig; theirs: mlx-vlm `_speculative_walk`).
3. NOT depth policy: fix3 == fix5 == adaptive == ~-4.4%. omlx pins
   block_size=3 and only extends when the last-32-round full-depth hit
   rate >= 65% (mlx-vlm `_effective_mtp_block_size`); observed tok/cycle
   3.46-4.13 with depth reaching d4.
4. REAL: round economics. dmlx adaptive won tok/round (~4.1 vs 3.46) but
   paid 110-114 ms/round vs omlx's ~90 ms. verify GPU grew +6.5 ms per
   verify token (L=4: 82 ms, L=8: 108 ms; DMLX_VERIFY_LAYER_SUBMIT=0 to
   isolate GPU from host build), and the draft chain cost 3.3 ms/step.

Commits, in order:

- `1cca39c` verify_qmm sg8 CT default 1->2 (-2.3 ms/round at M=8, neutral
  at M=4, bit-identical: CT only repacks output columns per threadgroup)
  + bf16 argpartition in the coarse draft head (draft stream
  bit-identical). Also added compare_engines `dmlx-think-fix3/fix5` arms
  and `scripts/mine_mtp_logs.py` (per-prompt round-economics miner for
  CMP_KEEP_PASS_LOGS server logs).
- `74de036` group/swiglu sg8 auto-enabled at M>=6 (they lose at M=4-5,
  win at M>=6; the controller parks deep on thinking content). verify
  104.8 -> 102.6 ms at L=8.
- `0ad07a0` **the big one**: fused GDN verify tail kernel (port of
  omlx's `_norm_gate`, one simdgroup per (token,head) row, MLX rms_norm
  reduction order, gated to L<=16) + bf16 gate storage while verify_qmm
  is armed so out_proj routes to the fast verify qmm instead of the
  stock f32 path (the 5e8ef27 conv-f32 promotion had silently excluded
  it). tail 0.66 -> 0.25 ms/layer; **verify 102.6 -> 84.7 ms/round at
  L=8 (-17.5%)**. Kill switches: DMLX_NO_VERIFY_TAIL_FUSION,
  DMLX_VERIFY_TAIL_F32_GATE. Watch item: the bf16 gate shifts the draft
  stream at ULP level; L=4 acceptance on one probe moved 58.3->50.7%
  (target-verify still decides tokens; thinking-pair result improved
  regardless).
- `de911fb` `zig build run-bench-verify-sg8`: per-shape GB/s harness for
  the production verify_qmm routing. Baseline: 160-200 GB/s at M=8 —
  the SAME class as omlx's kernels (shared lineage), which is what
  exonerated qmm bandwidth and pointed at the tail.

Refuted today: CHAIN_PRENORM for the p6/p7 acceptance cliff (p1 0.79 ->
0.49, rejected), lm_head sg8 routing (no A/B delta), GROUP/SWIGLU at
M=4 (confirmed worse, hence the M-gate), DMLX_VERIFY_QMM_SG8_NSG=4.

Remaining thinking-mode margin vs omlx is now POSITIVE; the known next
kernel-level upside (not needed for the win, would extend it): verify
qmm bandwidth 180 -> ~230 GB/s (mlx-lm's demonstrated 81% class; no
reference design at these shapes — research, not port), draft-step body
fusion (1.12 ms/step floor ~0.4). The prefill@2048 gap (~-4%, 16.2s vs
15.6s) is untouched by today's work; its remaining known item is the
out_proj f32 dtype policy (~270 ms) deferred earlier.

## 2026-11-06 — prefill out_proj bf16 gate (task A) + verify_qmm mma morphology (task C)

**A (commit `9599ced`)**: the last dtype-policy gap on the prefill side —
the GDN prefill/conv path handed an f32 gate to out_proj, which the dtype
policy kept off the fast quantized qmm route. All stored state stays f32;
only the gate handed to out_proj is bf16-rounded (and only when
`o_scales != null`; new `outGateDt()` decides; decode wide path stores
bf16 in-kernel; verify keeps `0ad07a0` behavior). `prefillTailFused`
untouched (serving never hits it; follow-up).
Measured: GDN prefill tail 1116→980 ms (−130 ms); ab_prefill span
−0.1..−0.26 s (inside ±0.3 s noise); decode throughput unchanged; L=8
acceptance 88/274 vs 87/275 (unchanged).
End-to-end prefill@2048 after A, two interleaved runs
(`artifacts/prefill_check_a1/a2_20261106.json`): dmlx 16.88/16.63 s vs
omlx 15.81/15.72 s → **−0.9..−1.1 s remains (−5.4..−6.3%)**, worse than
the 2026-11-05 baseline (−0.53 s) but BOTH engines drifted +0.1..+0.7 s
slower today, so A is neutral-to-small-positive and the residual gap is
NOT the out_proj dtype. Decode headline meanwhile: dmlx 46.1 vs omlx
36.7 (+25.6%). **Open: the remaining ~0.9 s prefill gap needs a fresh
span profile — the earlier ~1.0 s "unattributed" term is still the
whole story.**

**C (in progress)**: verify_qmm third morphology `mma`, adapted from
upstream MLX `affine_qmv_mma` (MIT): one uint4 (16 B) weight load per
lane per 128-wide K block (sg8 does four 4 B loads), fully unrolled
block body, inline dequant (half `1024+q` + fma scale/bias into the
fp32 A fragment — no xsum shuffle epilogue), WK=4 simdgroups splitting
K with a threadgroup reduction. Deliberate deviation: scale index is
`(k + 16*fn_)/GS` (the group of the nibbles the lane actually loaded);
upstream's `k/GS` is only right at GS=128. Gate: `DMLX_VERIFY_QMM_MMA=1`
(opt-in, M 4..8, K%128==0, N%16==0, BN=32 preferred/16 fallback).
Codegen keeps the no-thread-local-arrays discipline; unit tests assert
the source properties and the pick() gates.

**C VERDICT (commit `6c76797`)**: mma morphology implemented and
numerically validated (one-hot probes + bench rel diff 1e-5..4e-3;
two bugs found en route — upstream's per-block scale index is only
exact at GS=128, fixed per-element via `(fn_+e)/(GS/16)`; and a
Zig `@divExact(3,2)`-is-UB-in-ReleaseFast emission bug that scrambled
the g=3 weight words). **Bandwidth: 157-182 GB/s vs sg8's 165-198 —
parity.** Two morphologies with completely different load widths
(4x4B vs 1x16B per lane-block), epilogues (xsum shuffle vs inline
dequant), and unroll structure converge on ~175 GB/s, so the verify-qmm
ceiling is the DRAM access pattern of weight streaming at M<=8, NOT
kernel code. The mlx-lm 81% number must come from different
shapes/conditions. Kept opt-in (`DMLX_VERIFY_QMM_MMA=1`), not routed by
default. Remaining decode-side upside is elsewhere: draft-step fusion
(1.12 ms/step vs ~0.4 floor), and the ~0.9 s prefill@2048 residual
needs a fresh span profile.

## 2026-11-06 (pm) — conv v2 (prefill 残差) + drafter 编译步 + draft 阶段重新定性

**Prefill@2048 残差收敛（-1.0s -> -0.67s，已进噪声带 ±4%）**。
新归因（DMLX_LAYER_PROF + GDN_PREFILL_PROF @L=2345）：GDN front 的
conv 段 26 ms/层（~24 GB/s 有效）是最大项；PRICE_CONV 定价 0.61 s/
请求。修复（commit `b2c21fd`）：conv kernel v2 = 每线程 channel-quad
向量化（float4 / vec<bf16,4>）+ x bf16 直读（全张量 f32 promotion
变 lazy）。bench-gdn-chains：conv 1.18 ms/层（原定价 12.7 ms/层）。
位平等：同 prompt 温度 0 与 eager fallback 逐字一致。
compare_engines 复测（`artifacts/prefill_conv_fix_20261106.json`）：
ttft@2048 dmlx 16.22 vs omlx 15.55（-4.1%，前值 -5.4..-6.3%）。

**Drafter 编译步（commit `17f268f`）**：drafter decode 步 48 次 eager
dispatch 经 mx.compile 双段（pre/post，中段 KV append + decode SDPA
保持 eager）压到 ~13 次。kill-switch DMLX_MTP_NO_COMPILE；pre 段失败
在 cache append 前回退（状态安全）。**实测 wall time 中性**
（draft=9.9 vs 9.8 ms/round，acceptance 52.1% 逐 token 一致）——
因为单 sync 的 GPU 链里 CPU 构建时间本就藏在 GPU 后面；价值在并发
场景的 CPU 释放，不在单流 decode。

**Draft 阶段重新定性（推翻 launch-bound 假设）**：draft=9.9ms/round
(7 步) 是 GPU-bound：coarse 2-bit lm_head（318MB 读，microbench
1.69ms @188GB/s）+ argpartition top-64（1.33ms！）每步 ~3.0ms。
注意 microbench 与 in-context 的 2 倍差异未解（in-context 更快），
下一步先给 coarseDraftToken 加 in-context 计时再动手。
**下一个 decode 杠杆：coarse qmm + top-k 单 kernel 融合**（tile 局部
top-64 + merge，省 1MB logits 写回 + argpartition），预期
3.0 -> ~1.4ms/step，depth 7 时约 -10ms/round。

C 任务结论已在前一节（mma morphology，带宽上限判定）。

### 追平→超越分析（2026-11-06 pm2）

逐项定价（ab_prefill @~1900 tok, 单 rep, ±0.3-0.5s 噪声）：
- NORMS/TAIL/BCAST ≈ 0（bcast 是惰性死代码，profiler 假成本）
- RECUR +0.9-1.2s 是 verify 路径 hook，max_tokens=1 下被长短抵消，存疑
- **关键结构事实：omlx 的 packed_linear（tensor-unit GEMM）是 NAX/M5
  专属，本机 M4 Pro 不可用——两边 prefill GEMM 都是 stock MLX qmm
  6.67 TF/s，追平只需清非-GEMM 项，超越需要 GEMM 级突破（独立课题）**

修复序列：conv v2（b2c21fd，~0.4-0.6s）→ prefill tail 接受 bf16
norm_w（本次，~0.1s，md5 位平等）。tail 融合从此在 serving 激活。

**结果（61f94b2 后，artifacts/prefill_tail_fix_20261106.json）**：
ttft@2048 **dmlx 15.89s vs omlx 15.70s = -1.2%**（起点 -11.6%，噪声
±4% 内）——prefill 战线宣告实质性追平。轨迹：16.88 -> 16.63
(bf16 门控) -> 16.22 (conv v2) -> 15.89 (tail 融合生效)。

## 终验（2026-11-06 晚，reps=5/reps=3，artifacts/final_{nothink,think}_20261106.json）

**prefill@2048 追平确认**（两个 pair 一致）：
- 非思考对：dmlx 16.16s vs omlx 15.94s = **-1.3%**
- 思考对：dmlx 16.35s vs omlx 16.04s = **-1.9%**
- 均在 ±4% 机器噪声带内；dmlx 六场样本 15.89-16.88。战役起点 -11.6%。

**decode headline（同场，机器热漂移后）**：非思考对 dmlx 43.2 vs
omlx 36.1（**+19.7%**）；思考对 dmlx 38.0 vs omlx 35.7
（**+6.4%**）。全天峰值记录：非思考 46.9（+25.6%）、思考 39.7（+8.2%）。

全战线终态：decode 两对全胜，prefill 追平。coarse-head 融合证伪
（前提数字来自错误的 mlx 构建；真实上限 ~0.5ms/步 ≈ 3%，见
bench-coarse-head）。剩余长课题：Soft-MTP 接受率上限、M5/NAX
packed GEMM 移植（换机时）、ANE 混合 prefill（私有 API，灰色）。

## prefill 抖动归因（2026-11-06 晚）

用户观察到 dmlx prefill 抖动大于 omlx。对照探针（同一 server 10 连发
~2100-token 请求，max_tokens=1）：

- dmlx-plain: stdev 0.33s / dmlx-mtp: 0.59s / omlx-mtp: 0.32s
  （omlx 需排除 rep0 的 kernel 编译 warmup 22.1s）
- 之前引用的"四场 16.88/16.63/16.22/15.89"不是纯噪声——主体是
  conv/tail 修复的真实改进 + 跨场热漂移。
- MTP 模式残存稍高的方差来自更多 kernel 暴露于环境波动（非单调，
  无泄漏迹象；无状态累积）。

**顺带抓到一个生产 bug（08ba203）**：plain（非 MTP）路径的
prefix-cache store 缺 vtable_serial 门控 —— lookup 永不命中导致每
请求都 store（48 层 GDN 状态 ~160MB/请求，6-14s/请求！），且
pc_prompt 非引擎持有 → HTTP 释放后 dupe 触发 @memcpy 别名 panic
（~第 6 个长请求必崩）。修复后 plain 长 prompt 23-31s → 16.6-17.6s。

## 思考对 math 红格修复（2372805 + 复测 artifacts/think_latch_20261106.json）

深度彩票根因：梯子探针在 incumbent+2=5 打出 5/5 满接受，但
"近满接受重探索"门栓要求 n≥max_n-1=6 —— 探针永远够不到。慢局卡
N=3 到请求结束，快局与慢局差 38%。修复：门栓 floor 下探到
max_n-2（DMLX_MTP_REEXPLORE_FLOOR 可调）。

复测（思考对 reps=5，前值→现值）：
- math decode 38.5→**46.6** vs omlx 42.3：**-7.0% → +10.2%** ✅
- code 47.1 vs 37.7：**+24.9%**；prose +3.7%
- headline 38.0→**40.5** vs 35.6：+6.4% → **+13.8%**
- 思考对实现全领域无红格。

## "GEMM tile 差距"证伪（2026-11-06 深夜）——v8 tile 早已在分支上

调查 prefill 残差时发现：omlx 的 GEMM 优势（自定义 qmm tile
BM=64/BK=32/BN=64，实测 +4%）**本分支早已移植**——
`src/models/qwen38_qmm_v8.zig`（commit `10cc8f7`，flatten_qmm_v8.py
预扁平化 4813 行 steel header，M≥256 路由，bit-exact）。子代理重跑
run-bench-prefill-v8：7 个投影 shape 1.045-1.123x，dmlx tiled ≈
omlx tile（±2% 噪声内）。所以 prefill GEMM 从未落后——今日残差
-0.2~0.3s 全部是非 GEMM 散项（conv/tail 已清）+ 噪声。

子代理新增的冗余路由层（M≥1024 再路由到同一 kernel）已回退；
功能与 harness 都是重复建设。教训：开工前先 grep 现有 kernel 目录。

另一个被证伪的假设：MLX 运行时 JIT **不能**解析
`#include "mlx/backend/..."`（custom_kernel.cpp 只拼接扁平化
utils 文本，无 include 搜索路径；python 侧的"成功"是 lazy eval
没编译的假阳性）——v8 当年的预扁平化是必要的。

## 抖动治理终局 + Soft-MTP/pi-attn 评估(2026-11-06 深夜)

**抖动归因定论**(探针 DMLX_PREFILL_TRACE,commit 36eda65):
HTTP 墙钟方差 ≈ 引擎内部方差,host 侧增量恒定 ≤30ms —— 抖动 100%
在 GPU 执行内部(热/功耗/带宽),非代码可调度部分。三天数据 MTP
0.59/0.25/0.48、plain 0.33/0.53/0.17,模式间大小关系跨会话翻转,
omlx 0.32 同带。"dmlx 抖动更大"不成立,是采样时机差。无修复项。

**Soft-MTP 评估(~/work/code/Soft-MTP)**:推理侧 = 经典 Leviathan
speculative sampling(min(1,p/q) 接受 +(p-q)+ 重采样),dmlx 早已实现
(engine_loop.zig mtpAcceptProb/mtpResampleAdjusted,温度>0 无损)。
其增益全在训练侧(KL 梯度恢复 trick),与我们"不重训"定位冲突,不适用。

**pi-attn 评估(~/work/code/pi-attn)**:CUDA 专用(sm_80/89),核心
价值是 bitwise 确定性(fold-group attention 程序、row-invariant GEMM)
而非速度。移植到 Metal 是自研 kernel 大工程,无吞吐收益;若将来做
"确定性模式"差异化特性可再议。

## AV 接受率估计器评估(2026-11-06 深夜)——早退证伪,Phase 2 不做

缘起:vLLM 2026-10 PR #52228(Adaptive Verification 在线估计器,
logistic 回归 σ(ω·logit(q_max)+β_i),IRLS 在线拟合,免训练)。
评估是否移植到 dmlx 的 MTP 草稿链。

**Phase 1(DMLX_MTP_AVTRACE,本 commit)**:greedy GPU 链逐步从
coarse head 精确 rescore 提取 (l_max, top1-top2 margin) 特征,verify
后按 vLLM 规则打标签(首个拒绝位之前=1,拒绝位=0),写 JSONL。
4 prompt × 5 reps × 128 tok 采集 2178 条标注。

离线拟合(标准化特征,共享斜率+每位置截距,IRLS):
- **判别力优秀**:pooled AUC 0.918(vLLM 报 0.834),logloss 0.269,
  margin 特征权重最大(3.73 vs lmax 0.70,标准化后)
- **但早退模拟全线亏损**:τ=0.1..0.7 扫参,最好 +0.1%,普遍 -1~-11%。
  原因:我们 greedy+MTP 头的每位置接受率 0.76-0.96,survival 很难塌到
  阈值以下;draft 步成本(2-10ms)相对 verify(82-148ms)和被放弃的
  已接受 token(每个 ~25ms)太小,省一步不抵丢一个 token。
- vLLM 自己也有同样 caveat:"a smaller fixed length often outperforms
  AV"。他们的收益场景是 batch=64 compute-bound 裁 verify 宽度,与我们
  单请求带宽受限场景无关。

**结论**:早退(Phase 2)不做,EV 不成立。估计器唯一残余价值是给深度
控制器的 E[t] 模型供预测 p̂(解决跨深度证据陈旧问题),但需对未观测
深度外推,预期收益小,缓议。AVTRACE 作为诊断工具保留(默认关闭,
开销可忽略:64 元素 sort/步)。

## SpecExtend 前提实测(2026-11-07)——B 证伪,不适用

问题:我们的 MTP 头在长上下文是否像 EAGLE 系 drafter 那样退化
(论文:EAGLE-3 在 4K 后 τ 4.65→1.82)。方法:MTP server,重复
filler 词构造 1K/4K/8K/16K prompt,greedy 128 tok,每档 3 发,
读 server 日志的逐请求接受率:

- 1K: 52.0/50.0/44.9(均值 ~49%)
- 4K: 53.3/54.9/52.8(~54%)
- 8K: 49.2/48.0/53.6(~50%)
- 16K: 50.0/48.9/63.3(~54%)

接受率完全平坦(条件接受率 p1 稳定 0.71-0.87)。厂商 MTP 头训练
覆盖了长上下文,SpecExtend/CMR 针对的问题在我们这里不存在。
verify 时间随上下文增长(87→272ms/round)是 attention 成本,与
draft 质量无关。CMR 类方案不修不存在的问题,关闭。
caveat:filler 词内容只压 KV 长度不压语义检索;长推理(32K 生成)
场景未覆盖,但 16K 已远超 EAGLE-3 崩溃点(4K),前提已足够证伪。

## NPUMoE/ANE 方向 spike(2026-11-07)——D 证伪,不做

方法:scripts/bench_ane_gemm.py(coremltools 9.0,~/venvs/dmlx-ane
py3.13;cmp venv 的 py3.14 无 coremltools native wheel)。CoreML FP16
静态 matmul,CPU_AND_NE,M=2048,7 个 prefill 投影 shape vs dmlx v8
qmm 同 shape 实测(run-bench-prefill-v8,本机同日下午):

| shape | CoreML FP16 | dmlx v8 4bit | 比 |
|---|---|---|---|
| gdn_qkv | 38.8ms | 31.4ms | 0.81x |
| gdn_z | 24.0 | 19.0 | 0.79x |
| gdn_out | 34.8 | 18.9 | 0.54x |
| attn_q | 46.1 | 37.8 | 0.82x |
| attn_o | 34.6 | 18.8 | 0.54x |
| mlp_gate | 65.2 | 53.1 | 0.81x |
| mlp_down | 359.3(疑 fallback) | 53.4 | 0.15x |

加权全 pass:CoreML 33.1s(2.28 TF/s)vs dmlx 11.0s(6.75 TF/s)。
即使修掉 mlp_down 异常、扣掉 fp32 I/O cast 开销,天花板 ~5.6 TF/s
FP16 仍低于 4-bit qmm 的 6.75。另有结构性成本未计:CPU↔ANE 同步、
routing/top-k 留 CPU、GDN 层(64 层中 48 层)CoreML 无算子。

结论:ANE 的价值是相对 CPU 基线(NPUMoE 论文的基线正是 CoreML
CPU),不是相对调优 GPU;能效场景另说,但对"prefill 反超"目标
无益。方向关闭,不用 CoreML/ANE 做 GEMM 加速。
