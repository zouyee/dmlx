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
