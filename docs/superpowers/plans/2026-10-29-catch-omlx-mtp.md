# Catch omlx-mtp: Optimization Plan (Qwen3.8-27B-4bit, speculative decode axis)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Under the common-basis harness (`scripts/compare_engines.py`, interleaved reps, nonce-defeated prefix caches), close dmlx's speculative-decode HEADLINE gap to omlx-mtp. Re-baselined 2026-10-29: dmlx **34.0** vs omlx-mtp **41.3** tok/s (−18%; the user-reported 46.7 did not reproduce — omlx 0.7.1 unchanged, no MTP-related commits).

**Architecture:** The gap decomposes as tok/s = tokens/round ÷ ms/round. Per-round cost is already ours (91.6 vs 96.6 ms, §32.4); the deficit lives in cost(n) slope (~85% of the gap, §35.2) with acceptance contributing little (§35.2) — and the token-level oracle comparison below shows **no drafter-quality difference at matched positions**. So the main line is flattening cost(n); the trunk-post-norm input convention is a secondary, already-implemented acceptance tweak.

**Tech stack:** Zig 0.16 + mlx-c 0.7.0 (custom Metal kernels already shipped in `verify_qmm.zig`), Python harness `scripts/compare_engines.py`, omlx source readable at `~/work/code/omlx` (editable install; carries our `OMLX_CHAIN_DUMP` diagnostic patch).

**Dead ends (measured, do not redo):** tree drafts (§12.3), quality-for-speed quantization / vocabulary compression (rejected by user), op-level plain-decode work (§31: all zero), prefill GEMM (§19: hardware ceiling), verify-attention kernels (§41.1: unneeded), entropy-gated depth (§4.5), drafter-quality root cause for p3 decay (refuted below).

**Measurement discipline (hard-won, mandatory):**
1. All A/B: interleaved arms + cooldown + baseline re-measured at both ends (§25.6 thermal drift).
2. Diagnostic statistics read at FIXED depth only (`DMLX_MTP_FIXED_N`) — adaptive controllers contaminate them (§34.2).
3. Ablation stand-ins must keep every input of the replaced block consumed, keep values bounded, and must make total time go DOWN, else the reading is void (§30.6/§31.2).
4. Acceptance is a first-class correctness signal: temp-0 self-consistency CANNOT see a weakened drafter (verify fixes bad drafts) (§34.1(5)); fidelity gate = `scripts/mtp_fidelity_profile.py` (first-divergence distribution, §36.3).
5. Single per-prompt readings are noise on this machine; only the interleaved harness decides (§39.3).
6. Before testing a hypothesis, check it isn't already contradicted by the measurement method (§40.1).

---

## Execution log (2026-10-29)

- [x] **Task 0 — Re-baseline.** omlx 0.7.1 unchanged. Interleaved 3 reps + nonce: dmlx 34.0 (code 36.6 / prose 29.0 / math 36.3 / count 48.3), omlx-mtp 41.3 (41.3/26.2/56.4/49.4), mlx-vlm 30.7. Gap −18%, concentrated in math (−36%). TTFT short: dmlx 0.278 s (best) vs omlx-mtp 0.782 s.
- [x] **Task 1 — Chain-state audit (candidate 1 refuted).** `DMLX_MTP_CHAIN_DUMP=1` added to `src/model_registry.zig` (prefill/draft/commit/replay drafter-cache lengths). At fixed depth 4 every invariant holds: draft step i sees `cache_len == round_start + i`; commit truncates to old_len+1 and replays accepted drafts; the next round's first draft continues exactly there. RoPE offset == cache.len() (`qwen38_attn.zig:1011`), so the position accounting is closed.
- [x] **Task 2 — Step-by-step drafter oracle (conclusion flipped).** Added `OMLX_CHAIN_DUMP` to omlx's `_chain_next_drafts`. With the SAME raw prompt (omlx `/v1/completions` vs dmlx's ChatML expansion), both engines emitted **byte-identical** 407-char outputs. Position-aligned draft comparison: 80 comparable positions, 13 divergences, perfectly symmetric (dmlx right 5 / omlx right 5 / both wrong 3). **Same drafter weights → equal quality at matched context.** The §35.1 p_i shape difference (0.600 vs 0.955) is an artifact of depth-mix/sampling, not a defect. Bonus A/Bs: coarse head 2-bit vs 3-bit vs off, and stock vs custom verify qmm — acceptance identical in all four; the coarse head is strictly neutral as designed.
- [ ] **Task 4' — Trunk post-norm head input: NOT ADOPTED.** Fixed-depth-4 p_i promised a code win (E[tok] 3.41→3.60), but the interleaved harness A/B (3 reps, nonce) says **tie: 35.2 vs 35.0** (per-prompt: prose +10%, math −8%, code flat — all within noise). Per the honest-verdict rule the flag stays available (`DMLX_MTP_TRUNK_POSTNORM=1`) but default OFF. Another lesson: fixed-depth p_i improvements do not guarantee end-to-end wins.

- [x] **Task 5 (first fix) — Batched commit fold.** The differential profile (N=2 vs N=4) showed the verify marginal is now ~11.2 ms/row with only ~1.7 ms/row unattributed after the §37/§38/§42 fixes (front marginal +6.7/row ≈ the qmm bench's +6.9; tail +3.55/row includes out_proj qmm growth). The commit replay ran k SEQUENTIAL drafter body forwards per round; replaced with ONE batched prefill-mode fold (`DMLX_MTP_NO_BATCH_FOLD=1` reverts). **Perf-neutral (110.0 vs 110.0 ms/round)** — the lazy replay work overlapped the next verify anyway — but it matches omlx's fold structurally and shrinks the graph. Gates: self-consistency 5/5 byte-identical, fidelity profile NEUTRAL (5/6 both arms, identical divergence positions), unit suite 464+1 skip unchanged.

**Plan correction from these findings:** the acceptance axis shows no real deficit; **cost(n) slope (Task 5) is the main line** — at fixed depth 4 omlx runs 95.6 ms/round vs our 120.3 (§35.2).

---

### Task 4' completion: post-norm verdict and gate

- [ ] **Step 1: Read the interleaved A/B** (background task `bash-pzsa0wu2`). Adopt only if the HEADLINE delta exceeds dmlx's own run-to-run spread (~0.4 tok/s).
- [ ] **Step 2: Fidelity gate.** `scripts/mtp_fidelity_profile.py` with the flag on vs off; variant must not diverge EARLIER than baseline (§36.3 semantics).
- [ ] **Step 3: If adopted, flip the default** (invert the env to `DMLX_MTP_NO_TRUNK_POSTNORM`), run unit suite (≥472/473) + benchmark (`--speculative-mtp 7`: Paris ✓, E2E 7/7), update the analysis doc, commit only the files this task touched:
```bash
git add src/model_registry.zig scripts/compare_engines.py docs/superpowers/plans/2026-10-29-catch-omlx-mtp.md
git commit -m "perf(qwen38-mtp): feed head post-final-norm trunk hidden (omlx convention)"
```

### Task 5: cost(n) slope — locate the unattributed ~6.5 ms/row in the GDN front

cost(n) marginal is ~13-16 ms/row; bench accounts for qmm (+6.9) and drafter (+2.65); the rest sits in the verify front section (§37.1: front marginal +18.9 ms over +2 rows).

- [ ] **Step 1: Differential fixed-depth profile.** For N=2 and N=4: `N=<n> bash scripts/verify_prof_once.sh` (DMLX_VERIFY_PROF=1 built in). Subtract; barrier inflation is a same-magnitude constant offset in both, so the DIFFERENCE is trustworthy. Decompose the front marginal into its four sub-items (4 projection qmms / causal conv / gates / q,k broadcast). De-dup against the qmm bench's +6.9.
- [ ] **Step 2: Price the largest sub-item by ablation** (discipline rule 3: consume all inputs, bounded values, must get faster).
- [ ] **Step 3: Implement the fix + gates + commit** (same gates as Task 4' Step 3).
- [ ] **Step 4: Re-measure cost(n)** (`DMLX_MTP_FIXED_N=1..7`, 2 rounds each, median). If the curve flattened, RE-TEST the controller knobs (`DMLX_MTP_EXPLORE_ROUNDS` etc.) — twice already this project a knob's optimum flipped when cost(n) changed (§28.1, §38.2).

### Task 6 (conditional): mma morphology for depth > 7

Start only if after Tasks 4'/5 the depth-7 marginal is flat AND p_i at depths 5-7 stays ≥0.6. sg8 covers M≤8 (verify M = depth+1); deeper needs omlx's `mma` morphology (7..24 rows, simdgroup-matrix tiles; Apache-2.0 attribution per plan doc §0.5).

- [ ] **Step 1: Micro-bench M=9..12 in `src/bench_verify_kernel.zig`.** Gate: mma must cost ≤ 1.3× the sg8 M=8 flat line, else abandon (deep acceptance won't pay for pricier rows). Implementation rules from §9.5: codegen-unrolled named registers, no thread-local arrays, bf16 results astype→f32 before reading.
- [ ] **Step 2: Wire into `verify_qmm.pick` (M≥9 → mma), raise the depth ceiling, re-test the controller.**
- [ ] **Step 3: Gates + commit.**

### Task 7: Final six-engine scorecard

- [ ] **Step 1:** `python3 scripts/compare_engines.py --engines dmlx,omlx-mtp,mlx-vlm,omlx,mlx-lm --reps 3 --nonce`
- [ ] **Step 2: Honest verdict** (§13.4 rule: a delta smaller than run-to-run spread is a TIE, not a win). Success: HEADLINE ≥ omlx-mtp beyond noise. Partial: report the remaining decomposition (tokens/round vs ms/round).
- [ ] **Step 3: All gates green:** unit ≥472/473, oracle 7/7, fidelity profile neutral, benchmark Paris ✓ + E2E 7/7, peak RSS < 20 GiB.
- [ ] **Step 4: Write the new section** in `docs/analysis/beat-omlx-mlxvlm-plan.md` + update `HANDOFF.md`, including every refuted hypothesis and the final numbers.
- [ ] **Step 5: Commit** (`docs: catch-omlx-mtp final scorecard`).

## Self-review notes

- **Coverage:** plan → confirm → execute-to-goal maps to this file → user approval → Tasks 0-7; the goal's decision rule is Task 7 Step 2.
- **Placeholders:** Task 5's fix depends on what the differential profile finds — inherent to measurement-driven work; the diagnostic steps are concrete (exact commands, expected reads) and kill criteria are stated.
- **Consistency:** env switches (`DMLX_MTP_FIXED_N` / `DMLX_VERIFY_PROF` / `DMLX_MTP_CHAIN_DUMP` / `DMLX_MTP_TRUNK_POSTNORM`) and tools (`compare_engines.py` / `mtp_fidelity_profile.py` / `verify_prof_once.sh`) all exist in the tree now.
- **Repo hygiene:** the working tree carries the user's large uncommitted prior work; commits stage ONLY files touched by this plan's tasks.
