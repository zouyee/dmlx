# DSpine Phase 0 — six-scenario probe on M4 Pro (2026-11-25)

## Verdict

**GO for a bounded dmlx integration spike; not yet a production/default-mode
decision.** The community MLX implementation, using stock target kernels, beat
the strongest measured dmlx DFlash2+copy-bands baseline in all six decode-only
cells: **+6.7% to +25.6% median throughput**. `block=8` was the best sampled
configuration in every cell; `block=16` was consistently worse than `block=8`.

This is enough evidence to justify measuring DSpine inside dmlx. It is not an
AB/BA result on one binary, and DSpine has only two sampled repetitions per
cell, so the exact deltas remain provisional.

## Reproducibility

- Machine: M4 Pro 48 GB; one GPU workload at a time.
- Target: `mlx-community/Qwen3.8-27B-4bit`, snapshot `b935172`.
- Draft: `tencent/Qwen3.8-27B-DSpine-b16`, snapshot
  `260a6c23070ca873ff1a5f33b7119ac3f6a245ee`, quantized in memory to 4-bit,
  group size 64, affine.
- Code: `~/work/code/DSpine`, commit
  `75cade7ab0307fc2daac8eda95337705226b217e`; `dspine-mlx 0.1.0`,
  `mlx 0.32.5`, `mlx-lm 0.32.2`.
- Driver: `bench_six.py`; raw results: `results.tsv` (42 data rows, process
  exit 0). Prompt cache remained enabled within each cell; prefill is excluded
  from decode tok/s.
- Sampled protocol: thinking on/off x short/2k/8k, temperature 0.8, top-k 50,
  top-p 1.0, max 256 tokens, block {5,8}, two repetitions. Greedy controls used
  block {5,8,16}, one repetition.
- Baseline: `../copy_bands_20261125/REPORT.md`, dmlx DFlash2 + copies + bands,
  four sampled repetitions per cell on the same prompts and sampling settings.

## Sampled decode results (block=8)

| cell | DSpine tok/s, median [range], n=2 | dmlx DF2+bands tok/s, median, n=4 | delta | DSpine tokens/round |
|---|---:|---:|---:|---:|
| think-off/short | 37.28 [35.61-38.95] | 34.0 | **+9.6%** | 4.12 |
| think-off/2k | 42.47 [36.83-48.10] | 33.8 | **+25.6%** | 5.41 |
| think-off/8k | 36.72 [36.42-37.01] | 32.8 | **+11.9%** | 4.88 |
| think-on/short | 41.84 [39.25-44.44] | 39.2 | **+6.7%** | 4.22 |
| think-on/2k | 45.33 [41.88-48.78] | 39.6 | **+14.5%** | 4.83 |
| think-on/8k | 32.53 [29.75-35.32] | 28.5 | **+14.2%** | 4.03 |

For reference, the block=5 sampled medians were 30.30 / 32.16 / 20.93 / 35.31
/ 30.75 / 24.22 tok/s in the same cell order. Block=8 is therefore the clear
Phase 1 starting point.

### Where the time goes at block=8

Median per-round phase costs across the two sampled runs:

| cell | verify ms | draft ms | tail ms |
|---|---:|---:|---:|
| think-off/short | 96.4 | 8.9 | 6.1 |
| think-off/2k | 98.8 | 20.5 | 5.2 |
| think-off/8k | 97.7 | 30.2 | 4.8 |
| think-on/short | 87.6 | 8.0 | 5.1 |
| think-on/2k | 89.9 | 11.4 | 4.8 |
| think-on/8k | 97.8 | 20.1 | 5.0 |

Verify remains dominant in the stock MLX stack. The shape is compatible with
dmlx's existing tuned path: `generate.py:185` sets `round_T=block_size`,
`_draft_round` emits `round_T-1` drafts, and `generate.py:208-212` verifies the
anchor plus drafts. Thus **block=8 means M=8 target rows**, not M=9; no M>8
kernel expansion is required for the initial integration.

Peak memory over all runs was 20.4-24.9 GB. First-prefill costs in this Python
stack (about 16 s at 2k and 63 s at 8k) are recorded in the raw data but are
not part of the decode comparison.

## Block=16 control

Greedy block=16 changed throughput relative to block=8 by -23.3%, -29.5%,
-21.2%, -42.0%, -16.3%, and -28.6% across the six cells. Although it raised
tokens/round, the larger target pass more than cancelled that gain. Do not make
M>8 verify kernels a Phase 1 blocker.

## Limitations

- Different runtime stacks and different measurement dates/processes; this is
  a directional cross-stack probe, not a same-binary AB/BA.
- DSpine sampled n=2 per cell, so sampling noise is material, especially where
  the baseline ranges overlap.
- Some think-off runs ended before 256 tokens (short: 120 and 49; 2k block=8:
  40 and 256; 8k block=8: 256 and 88). Decode rates are still measured over
  generated tokens, but these cells need full-length AB/BA confirmation.
- `tokens/round` includes the bonus token and must not be compared directly to
  dmlx's depth-indexed `E[t]` logs without normalizing definitions.
- No output-identity, cache-rollback, server termination, or concurrency
  validation was performed in Phase 0.

## Phase 1 recommendation

Implement DSpine behind an opt-in drafter selection, initially fixed at
`block=8`, retaining DFlash2 as the fallback. Reuse the existing target verify,
accept/commit, rollback, depth-control, and copy-band machinery where their
contracts fit; the new work is the DSpine block-drafter forward, target-hidden
bridge, tail/rescoring, and drafter cache lifecycle.

The decision gate must be a same-ReleaseFast-binary DF2-vs-DSpine AB/BA over
thinking on/off x short/2k/8k, with full output/finish-reason comparability,
correct cache rollback and termination, and no memory regression. Only those
results should determine whether DSpine becomes the default. Preserve this
Phase 0 artifact as the feasibility evidence, not as the final performance
claim.
