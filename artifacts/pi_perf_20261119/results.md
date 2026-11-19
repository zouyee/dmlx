# pi-adapted cross-engine client perf — 2026-11-19 rerun (M4 Pro 48GB, serial)

Rerun of artifacts/pi_perf_20261118 with swap pressure relieved (sudo purge;
memory_pressure 92% free, residual swap ~3.3GB from user apps). Same model,
same harness (scripts/pi_perf_bench.py, 256-token budget, temp 0.8, streaming).

IMPORTANT configuration asymmetry: only dmlx ran a speculative drafter
(DFlash2 via --speculative-mtp 7 + DMLX_QWEN38_DFLASH2_DIR=z-lab snapshot).
TensorFold/omlx/mlx_lm ran the bare merged model — their drafter flags were
not engaged. See "Fairness note" below.

Decode tok/s:

| engine            | thinking | short(~65tok) | mid(2.1k) | long(8.5k) | TTFT 2k / 8k |
|-------------------|----------|---------------|-----------|------------|--------------|
| dmlx + DFlash2    | off      | 37.6          | 31.5      | 32.8       | 20.6s / 3.9s* |
| dmlx + DFlash2    | on       | 27.3          | 20.7      | 18.7       | 21.2s / 72.3s |
| TensorFold        | off      | 14.0          | 13.5      | 12.3       | 23.1s / 73.8s |
| TensorFold        | on       | 14.1          | 13.6      | 12.0       | 23.6s / 73.1s |
| mlx_lm serve      | on       | 14.1          | 13.8      | 13.3       | 21.3s / 86.1s |
| omlx 0.8.0rc1     | default  | 4.9           | 4.9       | 4.7        | 20.1s / 82.5s |

* dmlx long-8k TTFT 3.9s is a disk-persistent prompt-cache FULL HIT
(commit 0039a15: quantized KV snapshot reused across server restarts).
The thinking-on run used a different chat template (different prefix hash)
so it missed the snapshot and paid the cold 72.3s prefill.

Notes:
- mlx_lm thinking-off is not measured: its server exposes thinking only via
  the chat template default (on); no off switch used.
- omlx has no thinking toggle for this model path; single default config.
- omlx numbers unchanged from 11-18 (~5 tok/s): VLM legacy path, drafter
  never engages for this model. omlx-bench venv is broken (mlx_vlm missing
  d1_lfm3_vl); the dmlx-omlx venv (0.8.0rc1, mlx_vlm 0.7.11) was used.

Ranking (decode, all prompt sizes, both thinking modes):
dmlx+DF2 1.4-2.7x > TensorFold ~= mlx_lm >> omlx.

vs 2026-11-18 run: identical ranking; dmlx numbers recovered ~2x once
memory pressure dropped (37.6 vs 16.3 short-off), confirming yesterday's
"machine hygiene dominates" caveat. TensorFold/mlx_lm/omlx unchanged.

Fairness note: this is "each engine at its best-supported config" — dmlx is
the only one with a working DF2 speculative path. TensorFold (--drafter)
and mlx_lm (--draft-model) have drafter flags but were not configured with
the z-lab DF2 weights (whether they can load that architecture unmodified
is unverified). An "all-with-DF2" arm is future work.

---

## Addendum: TensorFold + DFlash2 (same day)

Verified: TensorFold natively supports the z-lab DF2 weights
(`--drafter <path> --drafter-bits 4`, "DFlash2 drafts for all streams").
mlx_lm does NOT: it loads the draft config as plain qwen3 and dies with
"ValueError: Received 81 parameters not in model" (DFlash2DraftModel is a
custom block-diffusion architecture; upstream supports only SGLang/vLLM).
omlx has no drafter engagement for this model at all.

IMPORTANT methodology correction: with DF2 enabled, TensorFold coalesces
accepted draft tokens into fewer SSE chunks, so pi_perf_bench chunk-counting
drastically undercounts (client showed 9.4/7.9/6.3 tok/s while the server
log proves otherwise). For this arm the SERVER log tok/s (tokens/decode-time,
excl. TTFT — same definition as the dmlx client numbers, which the gap
report verified equal server-side) is the valid metric:

| engine (server-side decode tok/s, excl TTFT) | thinking | short | mid-2k | long-8k |
|---|---|---|---|---|
| dmlx + DF2 | off | 37.6 | 31.5 | 32.8 |
| TensorFold + DF2 | off | 58.5 | 40.5 | 22.3 |
| dmlx + DF2 | on (256 tok) | 27.3 | 20.7 | 18.7 |
| TensorFold + DF2 | on (256 tok) | 24.4 | 24.1 | 18.3 |

Reading:
- thinking-off generations stopped early (finish=stop, 82-165 tokens), so
  those windows are short and noisy; sampling params also differ slightly
  (bench sends temp 0.8; TF defaults top_k 20 / top_p 0.95 shown at boot).
- TF+DF2 accepts ~7 draft tokens/round with "copies" (its log shows
  e.g. 68/91 accepted short, 133/240 mid, 82/229 long = acceptance
  75%/55%/36%), vs dmlx MTP depth-7 E[t]=4.86, acceptance 61.8%.
- Net: TF+DF2 wins short/mid thinking-off by a wide margin (58/40 vs
  38/32), dmlx wins thinking-off long (32.8 vs 22.3) and thinking-on short
  (27.3 vs 24.4); thinking-on mid/long are a tie within noise.
- dmlx's long-context advantage is consistent with the accept-phase cost
  analysis: TF's acceptance collapses at 8k (36%) while dmlx holds up.
- Actionable: dmlx DF2 integration is competitive but leaks ~20 tok/s on
  short prompts vs TF's copy-based multi-candidate verify. Worth studying
  TF's "DFlash2 + copies" verify path (likely tree/multiple candidate
  verification per round) as the next optimization reference.

Raw logs: this directory (bench_*.txt = client-side per-scenario lines,
*.log = server logs).
