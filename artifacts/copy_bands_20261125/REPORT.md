# Copy-chain band model (TensorFold lookup.zig port) — 2026-11-25

Machine: M4 Pro 48 GB. Worktree `mtp-copy-bands` @ 562943f + this change.
Binary: local ReleaseFast, single server at a time, GPU idle otherwise.
Model: mlx-community/Qwen3.8-27B-4bit (HF snapshot b935172), drafter
z-lab/Qwen3.8-27B-DFlash2 @ 50307d4c. Serve flags: `--speculative-mtp 7
--system-mem-guard 0 --temperature 0.8 --top-k 50 --top-p 1.0`.

## What shipped (opt-in)

`DMLX_MTP_COPIES_BANDS=1` (requires `DMLX_MTP_COPIES=1`) replaces the plain
copy proposal (first matching n ∈ {4,3,2}, latest position, depth ≤ n_round)
with a TensorFold `lookup.zig`-style band model:

- candidates ranked by (band tier of available continuation → suffix
  specificity n → available length, latest position);
- band caps on depth: available ≥8/16/32 → cap 5/6/7 (`copy_bands`,
  `src/speculative.zig`); depth = `min(7, max(n_round, band_cap))`;
- cap is pinned at 7 because tuned verify qmm paths stop at M=8 (M≥9 falls
  back to stock kernels — the 2026-11-19 tree-verify negative result).

Copies remain exact-match proposals; verification decides acceptance, so
output identity is unaffected by construction.

## Method

Arms: A = `DMLX_MTP_COPIES=1`; B = A + `DMLX_MTP_COPIES_BANDS=1`. Same
binary. Two passes A→B and B→A, 2 reps per cell per pass → n=4 per arm per
cell, drift-cancelled. Driver: `copy_bands_client.py` (thinking on/off ×
short/2k/8k, filler code-transcript prompt, max_tokens 256, temp 0.8,
streaming). Analyzer: `analyze.py`.

## Client decode tok/s (median of n=4)

| cell | A (copies) | B (+bands) | delta |
|---|---|---|---|
| think-off/short | 29.2 [26.2-32.8] | 34.0 [32.3-45.3] | **+16.4%** |
| think-off/2k | 33.0 [26.1-38.4] | 33.8 [30.1-35.0] | +2.4% (noise) |
| think-off/8k | 28.3 [25.0-33.6] | 32.8 [29.5-33.5] | **+15.7%** |
| think-on/short | 30.4 [27.5-40.2] | 39.2 [30.6-42.0] | **+29.0%** |
| think-on/2k | 35.7 [30.8-37.8] | 39.6 [34.8-42.1] | +10.9% |
| think-on/8k | 30.6 [24.2-31.2] | 28.5 [24.7-36.2] | −6.5% (ranges overlap) |

Output comparability: every request finished `length` at n=256 in both arms.

## Server mechanism evidence (per pass, 13 requests)

| arm | copy rounds | copy share (med) | copy acceptance (med) | verify ms (med) |
|---|---|---|---|---|
| A pass1 | 148 | 15% | 60% | 75.0 |
| A pass2 | 172 | 16% | 66% | 71.5 |
| B pass1 | 198 | 24% | 71% | 74.1 |
| B pass2 | 185 | 22% | 70% | 69.9 |

Bands engage copies ~1.5× more often AND pick better candidates
(+5-11 pts acceptance) at equal verify cost. This is the intended
mechanism; the think-on/8k client cell is inside sampling noise
(half-width ~3 tok/s at n=4, temp 0.8).

## Verdict

GO as opt-in. Gains are workload-dependent (they scale with exact-match
opportunities, like copies themselves); no cell regressed beyond noise and
verify bytes did not blow up. Combine with `DMLX_MTP_COPIES=1`.

## DMLX_MTP_TOPK headroom diagnostic (same session, greedy, 12 requests)

Per-depth share of verify positions where the target's own (greedy) token
was in the drafter's top-k, n≈710-752 per depth:

| depth | top1 | top2 | top4 | top8 |
|---|---|---|---|---|
| d1 | 0.834 | 0.916 | 0.956 | 0.974 |
| d2 | 0.711 | 0.800 | 0.907 | 0.949 |
| d3 | 0.629 | 0.775 | 0.815 | 0.883 |
| d4 | 0.549 | 0.663 | 0.753 | 0.852 |
| d5 | 0.463 | 0.581 | 0.705 | 0.786 |

Chain E[t] at top1 = 3.10; idealized survival at top2/top4/top8 marginals
= 3.81 / 4.44 / 4.95. The idealized numbers overestimate a real tree
(per-depth marginals are not jointly realizable), and the 2026-11-19/20
tree-verify experiments already measured the realized gain at ~zero under
linear verify-byte costs. Interpretation: drafter top-1 quality is the
binding constraint (p1 0.83 decaying to 0.46 by d5); the top1→top8 gap is
the theoretical ceiling a DNeo-class drafter (better top-1, no trees)
could approach. Raw data: `server_topk.log`.
