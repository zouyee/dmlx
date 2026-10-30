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
