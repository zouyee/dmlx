# DSpine Phase 1 verdict — 2026-11-25

## Decision

**Keep DSpine as a sampled-oriented experimental opt-in. Do not make it the
default and do not replace DFlash2.** After optimizing the sparse sampled path,
five of six sampled cells improved (+5.5% to +12.6%), but thinking-on/2k
regressed 6.6%. Greedy controls regressed in five of six cells. MLX active
memory increased 0.39 GB and peak memory increased 0.13 GB in the short control.

This is useful forward progress from drafter quality, but it does not meet a
uniform default-mode promotion bar.

## Usage

```bash
DMLX_QWEN38_DSPINE_DIR="$HOME/.cache/huggingface/hub/models--tencent--Qwen3.8-27B-DSpine-b16/snapshots/260a6c23070ca873ff1a5f33b7119ac3f6a245ee" \
DMLX_DSPINE_BLOCK=8 \
DMLX_MTP_COPIES=1 DMLX_MTP_COPIES_BANDS=1 \
zig-out/bin/dmlx serve --model "$HOME/models/Qwen3.8-27B-4bit" \
  --speculative-mtp 7 --system-mem-guard 0 \
  --temperature 0.8 --top-k 50 --top-p 1.0
```

`DMLX_QWEN38_DSPINE_DIR` takes precedence over the DF2 variable. The released
checkpoint is trained at block=16 but Phase 1 deliberately serves block=8:
anchor + seven proposals and M=8 target verify, using the existing tuned verify
kernels. Draft backbone linears are quantized to 4-bit affine gs=64; bridge
weights and codebook remain BF16. `DMLX_DSPINE_DRAFTER_BITS=0` disables draft
quantization for diagnostics.

Limits: five layers, K=16, text-only, no DSpine tree verify or predraft, no
on-disk DSpine prompt-context persistence, and no block>8 support.
`DMLX_DSPINE_BLOCK` accepts 2..8 and defaults to 8; the block=5 experiment did
not justify changing that default (`block5/REPORT.md`).

## Implementation

- `src/models/qwen38_dspine.zig`: strict config/weight loader, projected-KV
  sliding-window ring, causal block attention, BF16 code bridge, direct serial
  refinement, greedy chain and conditional sparse-q helpers.
- `src/model_registry.zig`: explicit DSpine selection, target taps
  `1,16,31,46,61`, in-memory DSpine prompt shadow, sampled/greedy draft paths,
  accepted-only commit observation, rollback/reset/deinit cleanup.
- `src/sampling.zig`: skip `-inf` in top-k sampling, use the actual retained
  candidate count, and reject an all-`-inf` distribution.
- Tests cover released-config parsing, invalid block size, serial greedy chain,
  predecessor-conditioned sparse q, missing-predecessor errors, and sparse
  top-k behavior.

No target verify or engine-loop kernel was changed.

## Numerical validation

The official and dmlx targets produce different hidden states and greedy text
for the smoke prompt. A dmlx target-only control matched the DSpine-enabled
output regime, so final-text differences across runtimes were not attributed to
the drafter.

The decisive comparison exported dmlx's raw five-layer tap and fed exactly that
input to the official `dspine_mlx` drafter with the same anchor and block=8.
Official and Zig results matched:

- all seven greedy proposal IDs exactly;
- all 112 top-16 candidate IDs exactly and in order;
- projected-context values exactly in the inspected prefix;
- block hidden only at BF16-rounding-scale differences.

Greedy and sampled 64-token smokes both completed finish=length without cache,
termination, NaN, or predecessor-contract errors. See `NUMERICAL.md` and
`reference_same_dmlx_tap.log`.

## Sampled AB/BA after optimization

A = DFlash2+copies+bands; B = DSpine block=8+copies+bands. Same ReleaseFast
binary, pass order A→B then B→A, two repetitions per cell per pass (n=4 per arm
per cell), temperature 0.8/top-k 50/top-p 1.0/max 256. All requests in both arms
finished length at 256 tokens.

| cell | A decode tok/s, median [range] | B decode tok/s, median [range] | delta |
|---|---:|---:|---:|
| think-off/short | 34.75 [28.50-39.40] | 38.35 [30.60-44.60] | **+10.4%** |
| think-off/2k | 31.90 [27.30-40.30] | 33.65 [26.40-38.00] | **+5.5%** |
| think-off/8k | 30.30 [29.50-32.90] | 33.10 [31.40-37.10] | **+9.2%** |
| think-on/short | 37.15 [33.50-38.30] | 41.75 [37.40-44.40] | **+12.4%** |
| think-on/2k | 37.40 [31.10-42.30] | 34.95 [28.50-39.40] | **-6.6%** |
| think-on/8k | 30.15 [29.10-30.60] | 33.95 [28.80-36.70] | **+12.6%** |

The thinking-on/2k ranges overlap broadly, so the cell needs more repetitions
before a promotion decision; it is not treated as a correctness failure, nor is
it hidden by averaging across cells.

Full-pass server medians:

| arm/pass | acceptance | E[t] | draft ms | verify ms | accept ms |
|---|---:|---:|---:|---:|---:|
| A pass1 | 32.3% | 2.54 | 11.7 | 80.1 | 23.4 |
| A pass2 | 31.7% | 2.66 | 12.5 | 70.6 | 23.3 |
| B pass1 | 37.3% | 2.90 | 18.0 | 70.9 | 23.4 |
| B pass2 | 39.0% | 2.90 | 17.2 | 70.4 | 22.1 |

Raw data: `round2/`; analyzer output: `analysis_round2.txt`.

## Greedy controls

One repetition per cell, same binary and prompts, greedy decoding. Every request
finished length at 256 tokens.

| cell | A tok/s | B tok/s | delta |
|---|---:|---:|---:|
| think-off/short | 42.9 | 40.8 | **-4.9%** |
| think-off/2k | 43.7 | 40.8 | **-6.6%** |
| think-off/8k | 35.9 | 33.6 | **-6.4%** |
| think-on/short | 40.9 | 39.5 | **-3.4%** |
| think-on/2k | 36.1 | 40.3 | **+11.6%** |
| think-on/8k | 27.8 | 25.2 | **-9.4%** |

Raw data: `client_greedy_greedy-{A,B}.log`; summary: `analysis_greedy.txt`.
Single-repetition greedy results are directional, but they clearly do not
justify presenting DSpine as uniformly faster.

## Memory

Short sampled control with `DMLX_MEM_PROBE=1`:

| metric | DF2 | DSpine | delta |
|---|---:|---:|---:|
| MLX active | 23.52 GB | 23.91 GB | **+0.39 GB** |
| MLX peak | 26.12 GB | 26.25 GB | **+0.13 GB** |

Raw logs: `server_memory_{A,B}.log`. DSpine therefore has a small memory cost;
the implementation avoids the reference tail's vocabulary-wide TG/TSV tables,
which would add about 762 MB BF16 on their own.

## Optimization evidence

The initial AB/BA had DSpine draft medians of 31.5-32.7 ms and sampled deltas
from -17.0% to +9.3% (`round1/`). Profiling showed a stable block cost of about
17 ms but a sampled server draft cost of about 35 ms. The difference came from
`topKTopPSampler`: sparse `-inf` entries repeatedly took the equal-score
insertion path, causing O(vocab x 50) work per drafted token.

Skipping zero-probability entries reduced stable sampled draft cost to about
21-22 ms in the profile smoke and 17.2-18.0 ms in the full second matrix. Full
tests and ReleaseFast passed after the fix. Details: `OPTIMIZATION.md` and
`server_prof{,_after_sampler}.log`.

## Remaining work

- More sampled repetitions for thinking-on/2k before any default-mode proposal.
- If further speed is required, add a candidate-aware sampled interface to avoid
  scanning the full vocabulary for each of the seven sparse proposals.
- Optional disk persistence for DSpine prompt context and vision support are
  outside this phase and must not be inferred from DF2 support.
