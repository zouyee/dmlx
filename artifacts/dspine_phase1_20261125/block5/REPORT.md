# DSpine block=5 versus block=8 — 2026-11-25

## Verdict

**Keep block=8 as the default. Block=8 won every sampled and every greedy cell.**
The official MLX guide's block=5 recommendation for stock 4-bit MLX does not
transfer to dmlx's tuned M=5..8 verify paths on this M4 Pro.

`DMLX_DSPINE_BLOCK` now accepts 2..8 and defaults to 8. `MtpOps.num_drafts` and
all active forward/candidate/refinement/sparse-q lengths use `block-1`; fixed
arrays are capacity-only. Invalid values fail enable explicitly.

## Numerical validation

Using the same exported dmlx target tap and anchor 1596, block=5 produced:

- four of four greedy proposal IDs exactly matching official dspine_mlx;
- 78 of 80 initial candidate IDs exactly matching and in order;
- two candidate differences at top-16 boundaries, with hidden differences at
  the same BF16/kernel-rounding scale already observed at block=8;
- greedy and sampled 64-token requests both finishing length without cache,
  predecessor, or termination errors.

Raw: `server_smoke.log`, `reference_same_tap.log`, `reference_tensors.log`.

## Sampled AB/BA

A = block=5, B = block=8. Same ReleaseFast binary, A→B then B→A, n=4 per arm per
cell, temperature 0.8/top-k 50/top-p 1.0/max 256, copies+bands enabled. Every
request in both arms finished length at 256 tokens.

| cell | block=5 tok/s, median [range] | block=8 tok/s, median [range] | block=8 delta |
|---|---:|---:|---:|
| think-off/short | 34.30 [25.10-38.50] | 40.10 [38.70-46.20] | **+16.9%** |
| think-off/2k | 36.20 [27.50-37.70] | 39.65 [35.50-43.70] | **+9.5%** |
| think-off/8k | 25.95 [21.40-28.70] | 33.10 [31.30-35.50] | **+27.6%** |
| think-on/short | 35.45 [33.20-38.40] | 37.20 [32.20-44.30] | **+4.9%** |
| think-on/2k | 34.00 [26.80-40.70] | 41.60 [38.00-42.70] | **+22.4%** |
| think-on/8k | 26.85 [25.50-29.70] | 35.65 [31.80-38.20] | **+32.8%** |

Mechanism medians:

| arm/pass | E[t] | draft ms | verify ms | accept ms |
|---|---:|---:|---:|---:|
| block=5 pass1 | 2.73 | 22.4 | 68.3 | 21.2 |
| block=5 pass2 | 2.72 | 21.8 | 73.2 | 23.3 |
| block=8 pass1 | 2.83 | 15.8 | 68.9 | 21.0 |
| block=8 pass2 | 3.07 | 16.9 | 72.9 | 22.0 |

Block=5 both reduced acceptance per round and failed to reduce per-round
latency. Raw logs and full analyzer output: `analysis_sampled.txt` and the
`client_pass*.log`/`server_pass*.log` files in this directory.

## Greedy controls

One repetition per cell, same binary and prompts. Every request finished length
at 256 tokens.

| cell | block=5 tok/s | block=8 tok/s | block=8 delta |
|---|---:|---:|---:|
| think-off/short | 37.2 | 41.7 | **+12.1%** |
| think-off/2k | 38.4 | 42.4 | **+10.4%** |
| think-off/8k | 33.0 | 36.4 | **+10.3%** |
| think-on/short | 37.9 | 42.3 | **+11.6%** |
| think-on/2k | 38.2 | 39.9 | **+4.5%** |
| think-on/8k | 33.5 | 35.3 | **+5.4%** |

Summary: `analysis_greedy.txt`.

## Memory

Across the seven greedy requests, block=5 and block=8 reported the same MLX
active-memory trajectory and peak values within 0.01 GB. Both reached 28.47 GB
peak; there is no memory basis for preferring block=5.

## Scope

This experiment does not change the Phase 1 verdict that DSpine remains an
experimental, sampled-oriented opt-in rather than the default drafter. It adds
a supported block override for future measurements but no dynamic block
controller.
