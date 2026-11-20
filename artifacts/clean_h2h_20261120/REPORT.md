# Clean H2H: dmlx vs engine-B (same DF2 drafter) — 2026-11-20

Machine: M4 Pro 48GB. Conditions: no inference server running before each
arm; dmlx memory guard DISABLED for the bench (`--system-mem-guard 0`)
because swap-growth escalation from unrelated apps was 503-ing requests.
Residual swap 4-7GB from user apps; both arms share the same machine state.
Model: Qwen3.8-27B-pi-merged-4bit; drafter z-lab/Qwen3.8-27B-DFlash2 on both.
Bench: clean_h2h.py — thinking on/off x short/mid/long x 2 reps,
temp 0.8, max_tokens 256, streamed.

Decode tok/s: dmlx = client-side (its SSE is per-token; verified ≈ server
round math). engine-B = server-side `done` lines (its SSE coalesces spec
batches, client chunk counting undercounts).

## Decode tok/s

| thinking | prompt | dmlx r1 | dmlx r2 | B r1 | B r2 | verdict |
|---|---|---|---|---|---|---|
| off | short | 37.9 | 30.1 | 65.4* | 44.9* | B ahead* |
| off | mid-2k | 29.6 | 41.6 | 46.4 | 34.7 | mixed |
| off | long-8k | 26.3 | 28.1 | 31.7* | 41.5* | B ahead* |
| on  | short | 32.5 | 33.2 | 34.0 | 35.8 | B +5% |
| on  | mid-2k | 32.9 | 30.0 | 32.7 | 29.2 | dmlx +2% |
| on  | long-8k | 30.5 | 23.9 | 26.0 | 25.1 | parity |

* thinking-off runs are polluted by early stops: B emitted only 40-158
tokens on several cells (short decode windows amortize differently), so
those numbers are not comparable to the 256-token cells.

## TTFT

| prompt | dmlx cold | dmlx warm (disk snapshot) | B cold | B warm (RAM cache) |
|---|---|---|---|---|
| short | 1.4s | 0.12s | 1.1-1.6s | 0.12-0.13s |
| mid-2k | 15.7s | 0.15s | 16.8-17.0s | 0.27s |
| long-8k | 65-67s | 0.20-0.30s | 53.5-70.1s | 0.30-0.32s |

In-session warm TTFT is now parity (both cache in RAM); dmlx's disk
snapshot advantage only shows across server restarts (0.2-0.3s vs full
re-prefill 53-70s for B).

## Server-side dmlx round anatomy (all 12 requests)

- acceptance 34.6-58.7%, verify 66-96 ms/round, draft 24-34 ms, accept
  21-31 ms, commit <1.5 ms.
- accept phase is back to ~25 ms/round (20% of round) — the single largest
  non-roofline item. B's round-resolution is one fused GPU kernel; ours
  still pulls L+1 verify rows + resample separately. Next target.
- verify 66-70ms at short/mid context ≈ the DRAM weight-streaming floor;
  growth to 85-96ms at 8k is attention KV reads.

## E[t] comparison (thinking-on, server-side)

| prompt | dmlx E[t] | B E[t] | dmlx acc | B acc |
|---|---|---|---|---|
| short | ~3.9 | 4.00-4.06 | 41-44% | 42-43% |
| mid-2k | ~3.9 | 4.06 / 3.61 | 42.6-48.4% | 42.8% / 37.2% |
| long-8k | ~4.0 | 4.00 / 4.06 | 44.8-48.4% | 42.3% / 41.5% |

Acceptance and E[t] are at parity on identical content. The remaining
short-prompt gap is round-time (accept phase), not acceptance.

## Raw artifacts

- `clean_h2h.py` — bench driver
- `clean_dmlx2.log` — dmlx server log (per-request MTP stats)
- `clean_tf.log` — engine-B server log (per-request done lines)
