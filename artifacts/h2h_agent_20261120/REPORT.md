# dmlx vs DF2-equipped engine B + agent e2e — 2026-11-20 (M4 Pro 48GB, serial)

Model: Qwen3.8-27B-pi-merged-4bit, both engines with the z-lab DFlash2
drafter, bench = scripts/pi_perf_bench.py (temp 0.8, 256 tok budget).
Decode tok/s: engine-B numbers are server-side (its SSE coalesces spec
batches, client chunk counting undercounts); dmlx client ≈ server (gap
report 2026-11-18). Machine carried 3-7GB swap all day; ranking is the
signal, absolutes drift.

## Decode matrix

| thinking | prompt | dmlx | engine-B | verdict |
|---|---|---|---|---|
| off | short | 31.9-37.4 | 35.5 | parity/win |
| off | mid-2k | 32.6-36.8 | 42.5 | -13% |
| off | long-8k | 19.6-28.3 | 19.8 | parity (win w/ cache) |
| on  | short | 27.7 | 28.7 | parity |
| on  | mid-2k | 24.8 | 34.2 | -27% |
| on  | long-8k | 15.9-17.3 | 19.0 | -12% |

## TTFT (the dmlx structural win)

| prompt | dmlx (disk snapshot) | engine-B |
|---|---|---|
| 2k warm/restart | 0.41s | ~24s |
| 8k warm/restart | 0.36-2.9s | ~76s |
| 8k cold | ~92s | ~77s |

Multi-turn extension across a restart: 3836/5514 tokens restored from disk,
wall 43.0s -> 15.1s (turn-boundary-aligned snapshots; BPE merge at the
generation header was diagnosed as the exact-prefix breaker and fixed).

## Optimizations landed this round

- sampling: single-pass top-k (no 3MB/vocab-entry ScoredToken
  materialization) + logprob only when requested — DF2 draft phase
  46-64ms -> 24-36ms/round (commit 96ebb37).
- GPU residual resample (Gumbel-max, exact, statistically tested): measured
  NEUTRAL, reverted. The accept phase's residual cost is sync absorption,
  not host compute.

## Where the remaining gap is

Thinking-mode E[t]: dmlx 2.4-2.7 vs engine-B 3.4-4.5 tokens/round on
reasoning content. Engine-B's edge is multi-candidate verify (tree +
context copies); our tree implementation exists but is shelved (sweep:
E[t] saturates at 6.18 @10 nodes, round cost grows linearly — pays only
after fused GDN tree kernels exist). DNeo (arXiv:2611.21925) is the
drafter-quality route, no Qwen3.8-27B weights yet.

## Agent e2e (codex + pi)

- codex (responses wire, workspace-write): two consecutive coding tasks
  completed (mul branch; div branch + zero guard), apply_patch executed,
  self-verified runs passed. One session hit 503s from the memory guard
  under heavy machine swap (7GB+ from repeated benchmark churn); codex
  retried and completed. Guard worked as designed (no OOM).
- pi: KeyError-on-empty-section task fixed and verified; 7 in-session
  prefix snapshot hits; server log clean.

## Known issues carried forward

- Memory guard shedding is user-visible (503) on swap-polluted machines;
  consider softer policy (queue instead of reject at warn level).
- DF2 drafter context disk cache is still single-slot (df2_latest) — with
  backbone multi-slot snapshots, a mismatched drafter context forces a full
  tapped prefill.
