# Close the residual decode gap vs omlx-mtp

Date: 2026-10-30. Status: executing.

## Latest measured state (all same-session, interleaved, post leak-fix + tokenizer-fix)

| metric | dmlx | omlx-mtp | verdict |
|---|---|---|---|
| HEADLINE decode t/s (harness) | 35.2 (dmlx-nothink) | 46.2 | behind (inflated, see below) |
| effective t/s incl TTFT (code/prose/math) | 35.6/23.8/34.1 | 35.0/27.7/34.5 | TIE |
| TTFT short | 0.282s | 0.80-1.16s | we win 3-4x |
| cold prefill, 1618 tok, nonce | 14.2s | 13.6s | parity (was 26s pre-tokenizer-fix) |
| plain decode | 68.6 ms/tok (~93% of 231 GB/s roofline) | same | parity |
| fixed-depth-4 p_i | 0.87/0.78/0.80/0.78 | 0.77/0.79/0.73/0.79 | equal (same weights) |
| tokens/round (adaptive, 6 prompts) | 3.06 | 3.04 | equal |
| ms/round (adaptive) | ~97 | ~90 | **−7%, the real residual** |
| sustained serving stability | clean (post leak fix) | — | fixed today |

omlx's headline 46.2/51.9-math readings are inflated by (a) two-point windows
ending at their early stop (fastest segment only), (b) best-of-6 selection;
their own per-cycle logs show ~3.5 tok/cycle @ ~89 ms ≈ 39-40 tok/s steady.
The honest residual is the ~7 ms/round gap at depth 3-4.

## Decomposition of the 7 ms/round (depth 3-4, same weights, same content)

- draft chain: ours ~10 ms visible (3.2 ms/step: head fwd ~0.6 + 2-bit coarse
  qmm ~1.55 + rescore/argmax ~0.3 + per-step overhead ~0.7). Theirs is
  fire-and-forget async, GPU cost hidden inside their 86 ms backbone figure.
- verify: ours ~86 ms; theirs ~80 ms (their backbone 86 minus chain ~6).
- accept/commit: ~1 ms both.

## Plan (each step: implement -> fixed-depth A/B -> interleaved harness gate)

- [ ] **S1. Compile the draft step.** The decode path already uses mx.compile
  closures (GDN step, SwiGLU, attn pre-cache); the MTP draft step still runs
  ~10 eager ops per step (head layer + fc + norms + coarse qmm + rescore).
  One compiled closure per draft step removes the ~0.7 ms/step overhead and
  fuses elementwise tails. Gate: draft ms/step 3.2 -> <=2.3 at fixed depth 4
  (DMLX_MTP_DEBUG), acceptance p_i unchanged (self-consistency 5/5).
- [ ] **S2. Verify-front qmm cross-bench vs omlx.** Bench our splitk/sg8
  routing against omlx's custom verify qmm on the six verify shapes at
  M=3..5 (their kernel is Python MLX; bench in ~/venvs/dmlx-omlx). Adopt
  whichever wins per shape (theirs is Apache-2.0; port the tile config, not
  the code). Kill criterion: if the delta is <1 ms/round total, stop here.
- [ ] **S3. GPU-resident verify input.** forwardVerifyImpl takes host
  []const u32, forcing a sync between draft and verify. omlx verifies
  mx.concatenate([next_main, drafts]) fully GPU-side. Feed the verify from
  the GPU-chained draft tokens directly; the single per-round sync moves to
  the acceptance readback (omlx's structure). Est. -2-4 ms/round.
- [ ] **S4. Re-run the controller exploration sweep** (DMLX_MTP_EXPLORE_ROUNDS
  0/3/6) ONLY if S1-S3 measurably flattened cost(n) — twice already a knob's
  optimum flipped when cost(n) changed.
- [ ] **S5. Final gates:** interleaved harness dmlx-nothink vs omlx-mtp
  (3 reps, nonce) HEADLINE delta beyond run-to-run spread; self-consistency
  5/5; unit 472/473+1; fidelity neutral; commit.

## Explicitly out of scope

- Drafter training (user vetoed 2026-10-30: same weights => no training gap).
- ANE prefill (their qwen35_ane_prefill path; we are at cold-prefill parity
  without it after the tokenizer fix).
- mma kernel morphology for depth > 7 (vetoed by the acceptance cliff data).

## Execution log (2026-10-30)

- [x] **Final scorecard (post leak+tokenizer+parity fixes, 3 reps nonce)**:
  dmlx-nothink 34.2 vs omlx-mtp 43.9 HEADLINE decode; effective tie
  (35.0/22.2/33.2 vs 34.5/23.6/34.3); TTFT 0.283 vs 0.791 (we win 2.8x);
  prefill 121.5 vs 137.5 tok/s (tokenizer fix; was 49.2); ttft@2048 17.9s
  vs 16.0s (was 37.2s).
- [x] **S2 evaluation — premise REFUTED**: verify-qmm routing tally at M=3-5
  shows every projection shape routed (splitk/sg8), zero stock fallbacks;
  qmm marginals are ~0.01 ms/row (weight-amortized). The verify marginal row
  (8.2 ms vs omlx ~3.1) is NOT qmm. Differential DMLX_VERIFY_PROF (N=2 vs
  N=5, same session): front +5.3 ms/row (conv+gates — the 4 qmms are
  amortized), tail +2.9 ms/row (out-gate norm + out_proj), recur ~flat.
  => the residual lives in the GDN conv/gates + tail ops at verify M,
  inside the user's heavily-optimized in-flight verify path.
- [ ] S1/S3 deferred: compile draft step ~+2.5%, GPU verify input ~+1% —
  combined below the cross-session drift floor; revisit if the front/tail
  marginal is fixed first.

## Verify marginal deep-dive (2026-10-30, all same-session, fixed-depth A/B)

Cycle/round totals at fixed depth 1/2/4 (same code prompt):
- omlx: 70.6 / 75.9 / 89.3 ms per cycle -> marginal ~6.2-6.7 ms per
  (verify row + draft step). With their draft step ~3.2 ms (same head, full
  4-bit lm_head), their verify marginal row ~= 3.0-3.5 ms.
- dmlx: 77.7 / — / 111.3 ms per round -> marginal 11.2 ms per (row+step) =
  draft 3.1 + verify row 8.1.

**The whole residual gap = verify marginal row: 8.1 vs ~3.2 ms.**

Refinements:
- verify_qmm routing tally at M=3-5: every projection routed (splitk/sg8),
  zero stock fallbacks. sg8-everything (DMLX_VERIFY_QMM_SG8_MIN_M=4) measured
  WORSE in context (+3 ms/round) — bench routing stands.
- Front sub-splits (DMLX_GDN_PREFILL_PROF during verify, N=2 vs N=5): conv
  FLAT (0.183->0.195 ms/layer-pass), gates FLAT, bcast FLAT; the growth is
  entirely in the 4 projection qmms (2.09 -> 2.60 ms/layer-pass,
  +0.17 ms/row/layer x48 = 8.2 ms/row) — even though the isolated bench shows
  the same kernels nearly M-flat (0.015-0.056 ms/row). This is the §28
  in-context penalty: 3-4x worse than isolated, growing with M.
- omlx's split-K kernel source is near-identical to ours (same lane-strided
  packs, 4-col tiles, threadgroup partials) — kernel morphology is NOT the
  differentiator at M=3-5. What differs: their verify forward fuses far more
  of the non-qmm work (custom_kernels/qwen35_prefill/, vk_swiglu_sg8 fused
  gate_up+silu+mul, one-chunk GDN window processing), so the ~30+ small
  per-layer ops that serialize with our qmms in context largely vanish.

**Precise remaining lever**: consolidate the verify forward's per-layer
non-qmm ops into fused kernels (omlx-style), or port their fused swiglu +
chunked GDN verify. Est. +10-15% decode at depth 3-4, and it flattens
cost(n) enough that the controller's deeper depths (4-5) become profitable
— the two effects multiply. Effort ~1 day, gated by self-consistency +
fidelity + harness. Everything cheaper has been measured out.
- [x] **Fused gate|up verify MLP: TRIED, NEUTRAL, REVERTED** (verify 98.6-101.7
  vs 96.6-97.0 ms/round at fixed depth 4). Halving projection launches does
  not touch the in-context marginal — confirming the marginal is per-kernel
  in-context behavior, not launch count.
- [x] **Prefill qmm port evaluated, REJECTED by data**: omlx's native
  affine_q4_qmm_t (variant 8) vs stock MLX at M=2048 MLP shape = 1.05x, bit
  exact. Their prefill edge is not the qmm kernel (ANE path is disabled in
  their shipped settings too). Post-tokenizer-fix prefill is ~parity.
- [x] **In-context vs isolated kernel divergence confirmed once more**: the
  verify marginal (+0.17 ms/row/layer, = the entire 8.2 ms/row) appears only
  in-context; the isolated bench shows the same kernels M-flat. Identical
  split-K morphology on the omlx side rules out kernel source differences;
  the cause is GPU scheduling/L2 interaction across the ~30 mixed kernels per
  layer — i.e. kernel COUNT around the qmms, not the qmms (launch-count
  fusion already tried and neutral). What remains untried and matched to the
  evidence: omlx's per-layer mega-fusion (their custom_kernels/qwen35_prefill
  GDN chunk + vk_swiglu_sg8), which removes the small kernels BETWEEN the
  qmms. Est +10-15% decode at depth 3-4; ~1 day; all gates in place.
- [x] **split-K BN=8 (x re-read halving): TRIED, REFUTED, REVERTED.** Theory:
  each column tile re-reads the activation rows, traffic ~ N/BN x M. Correct
  after a write-back fix (acceptance identical 56.1%/69.6%), but verify
  SLOWED 96-125 -> 190-209 ms/round: halving the threadgroup count costs
  more occupancy than the saved x traffic buys. The verify qmms are
  occupancy-bound, not bandwidth-bound, at these shapes.

## Fusion program design (2026-11-02) — omlx verify mega-fusion port

Inventory (both sides, same weights, M = 1 + depth rows):

- omlx verify layer: `vk_group_sg8` (every same-input projection tuple in
  ONE sg8 launch; grid y = sum(N)/8, per-threadgroup tile picks its matrix
  via an if-chain), `vk_swiglu_sg8` (gate/up + silu + mul in one launch,
  epilogue rounds both to T first), fused add+rmsnorm that also emits
  per-64 group sums consumed by the sg8 kernels, partial-graph flush every
  8 layers, chunked-GDN Metal kernels.
- dmlx verify layer: conv and the delta-rule recurrence are already single
  fused kernels; add+rmsNorm is already fused. What is NOT fused: the 4 GDN
  in-projections (qkv/a/b/z, shared input), the 3 attention projections
  (q/k/v, shared input), and the MLP gate/up + silu + mul chain. Those are
  the ~30 mixed kernels per layer that the in-context penalty lives in.

Scope for this port (phased, each phase A/B gated at fixed depth):

- **F1. sg8 group kernel** (`maybeGroupQmm`): GDN front (qkv 10240 + a 48 +
  b 48 + z 6144 = 16480 cols, 2060 tiles, one launch) and attention
  (q/k/v) at 4 <= M <= 8, 4-bit gs in {32,64,128}. K=5120 % (64*8) = 0 OK.
  M < 4 stays on the existing per-projection split-K routing.
- **F2. sg8 swiglu kernel** (`maybeSwiglu`): gate/up (N=17408) + silu + mul
  in one launch, epilogue rounds gate/up to bf16 first (bit-parity with the
  separate ops), then down_proj as today.
- **F3 (only if F1/F2 pay)**: group-sums producer in the fused add+rmsnorm
  (gs=64 == omlx's _SUM_GROUP, so the sums variant applies) and the
  every-8-layers partial-graph flush in forwardVerifyImpl.

Kill switches: both are OPT-IN (DMLX_VERIFY_GROUP_SG8=1 / DMLX_VERIFY_SWIGLU_SG8=1) after the 2026-11-02 A/B refuted them at the parking depth.
Kill criterion (same as S2): if the combined delta is <1 ms/round at fixed
depth 4, stop and revert.

## Execution log (2026-11-02)

- [x] **F1+F2 implemented**: sg8 group kernel (one launch per same-input
  projection tuple: GDN qkv/a/b/z, attention q/k/v) and sg8 swiglu kernel
  (gate/up + silu + mul, epilogue rounds to bf16 first for bit-parity),
  ported from omlx `vk_group_sg8`/`vk_swiglu_sg8` into
  `src/models/verify_qmm.zig` (`maybeGroupQmm`/`maybeSwiglu`), wired into
  `qwen38_gdn.prefillFront`, `qwen38_attn.forward` (L>1) and
  `qwen38.mlpForwardPersistent`. Armed-only, 4 <= M <= 8, 4-bit affine,
  kill switches DMLX_VERIFY_NO_GROUP_SG8 / DMLX_VERIFY_NO_SWIGLU_SG8.
  Self-consistency 5/5 with both live.
- [x] **Latent double-free fixed en route**: `prefillVerify` registered both
  `errdefer y.deinit()` and `defer y.deinit()` — any error after `y` was
  assigned segfaulted the server in unwinding (first seen when a Metal
  source typo failed the swiglu compile mid-request).
- [x] **F1+F2 A/B at fixed depth 3 (M=4), median ms/round, same session**:
  both off 94.1 / group only 95.4 / swiglu only 96.2 / both on 99.7.
  WORSE monotonically. Consistent with the bench table: split-K beats sg8
  per-kernel at M=4-5 on every narrow shape, so replacing 4 split-K launches
  with 1 sg8 launch trades a winning morphology for launch-count savings —
  and launch count was already measured neutral (fused gate|up). omlx wins
  with this because THEIR alternative is stock qmv (much worse at M=4), not
  split-K. Depth-5 (M=6, where sg8 wins per-kernel) checked before revert.
- [x] **Depth-5 check (M=6) + final verdict — fusion program REFUTED at the
  operating point**: N=5 median 114.0 on vs 117.6 off (fusion wins at M=6,
  as the per-kernel bench predicts). But ms/token (same session, code
  prompt): depth 3 off 94.1/2.98 = 31.6; depth 5 on 114.0/3.47 = 32.8;
  depth 5 off 33.9. The deeper chain's acceptance (3.47 tok/round, not the
  pooled-p_i 3.86) does not repay its rows even fused. Both kernels flipped
  to OPT-IN (DMLX_VERIFY_GROUP_SG8 / DMLX_VERIFY_SWIGLU_SG8), code kept for
  an M >= 6 operating point. Combined with fused-gate|up-neutral and
  split-K-BN8-refuted, the "kernel count between qmms" hypothesis is now
  measured out three ways. The residual ~7% ms/round gap vs omlx stays
  unexplained by any single lever tried; remaining untried: their
  add_rms+group-sums (F3) and 8-layer partial-graph flush — deprioritized,
  since F1/F2 were the load-bearing half and went backwards.
- [ ] S5 final gates for the session's actual changes (test-gate fix,
  double-free fix): unit 465/466+1skip green; self-consistency 5/5 (run
  with the fusion kernels live, pre-opt-in-flip).

## 2026-11-02 — prose-gap root cause: trunk-hidden norm convention (FIXED)

The 6-rep interleaved harness with per-request raw samples showed the prose
gap is REAL, not noise (effective 21.7 vs 30.7 tok/s, no overlap), and NOT
the verify-marginal story: at the prose operating point per-round costs are
identical (83.6 vs 83.2 ms). The gap was tok/round: 2.02 vs 2.25.

Decomposition by conditional acceptance (empty-think prose, matched
semantics to omlx's depth[d=a/b] log):
  p1..p4: dmlx 0.64/0.34/0.11/0.11   omlx 0.74/0.68/0.33/0.50

Eliminated by A/B (each bit-identical or unchanged): CHAIN_PRENORM
(chain on pre-norm head output), NO_COARSE_HEAD (2-bit proposal head),
REPLAY_FREE (keep drafter's own chain states instead of replaying with
target hiddens — new toggle, kept for future A/B).

Cause: **omlx feeds the head the trunk's POST-final-norm hidden and folds
committed history with post-norm rows** (`HEAD_HIDDEN_POST_NORM = True`,
prompt_priming.py: "accepts a few points higher than PR 990's pre-norm at
every depth"); we fed PRE-norm everywhere. The `DMLX_MTP_TRUNK_POSTNORM`
toggle (built 10-30, never flipped) covers priming rows, draft-0 input,
verify chaining rows and commit replay rows. Measured with it ON:
- prose: p1 0.643 -> 0.776 (omlx: 0.736), tok/round 2.02 -> 2.17, ms/tok
  41.6 -> 37.6 (-10%);
- code:  p1 0.960 -> 0.977, p4 0.267 -> 0.677, tok/round 3.74 -> 4.18 (+12%).

Flipped to DEFAULT ON (`DMLX_MTP_NO_TRUNK_POSTNORM=1` reverts). Gates:
unit 465/466+1skip, self-consistency 5/5, fidelity neutral (median
first-divergence 591 both sides).

Residual p2 gap on prose (0.36 vs 0.68) is suspected SELECTION, not defect:
omlx's controller only runs d2 in 38/57 cycles (high-confidence content),
so their p2 is conditioned on favourable rounds; our parked-2 measures all
rounds uniformly. p1 — the only selection-free statistic — now favours us.
