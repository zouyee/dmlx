# pi-adapted cross-engine client perf — 2026-11-18 (M4 Pro 48GB, serial, swap-degraded)

All engines served ~/models/Qwen3.8-27B-pi-merged-4bit (offline LoRA merge,
scripts/merge_pi_adapter.py — omlx/TensorFold/mlx-serve lack LoRA support).
dmlx also runs DFlash2 (--speculative-mtp 7, sampled temp=0.8 path).
Caveat: machine had 5.5-7 GB swap in use throughout (user apps + churn);
absolutes are depressed ~30-40% vs idle-machine numbers from earlier today
(dmlx short-prompt decode hit 29-37 tok/s when memory was clean).

Decode tok/s (256-token budget, temp 0.8, streaming):

| engine        | thinking | short(~65tok) | mid(2.1k) | long(8.5k) | TTFT 2k / 8k |
|---------------|----------|---------------|-----------|------------|--------------|
| dmlx          | off      | 16.3          | 14.5      | 12.4       | 21s / 73s    |
| TensorFold    | off      | 13.6          | 13.3      | 12.2       | 24s / 76s    |
| omlx 0.8.0rc1 | off      | 5.0           | 5.0       | 4.8        | 20s / 81s    |
| mlx_lm serve  | on       | 13.8          | 13.6      | 13.0       | 0.3s* / 0.3s*|
| dmlx          | on       | 9.2-16        | 11.9      | 11.2       | 22s / 73s    |
| omlx          | on       | 5.0           | 4.9       | 4.7        | 20s / 81s    |

* mlx_lm prompt-cache hits on repeated benchmark prompts (its TTFT is not
comparable cold; first-touch walls were ~20/40/107s ≈ same ~110 tok/s
prefill as everyone else).

Takeaways:
- Prefill throughput is ~105-130 tok/s on ALL engines for this hybrid GDN
  model — hardware-bound parity, no engine differentiates.
- dmlx >= TensorFold > mlx_lm >> omlx on decode. omlx's ~5 tok/s (VLM
  legacy path, no drafter engaged for this model) is 2.5-3x slower than the
  field — consistent across thinking on/off and prompt sizes.
- dmlx's spec advantage is largest on short/mid prompts; at 8k context the
  sampled accept phase (~33-135ms/round) eats it to a tie with TensorFold.
- Machine hygiene dominates: after the swap filled, all dmlx numbers
  dropped ~40%. Re-run on a clean boot for publishable absolutes.

Fixes landed during this benchmark (commit 5ab9368):
- MTP draft-failure fallback double-free segfault (h_cur/hidden alias +
  epilogue re-free) — reproduced via crash report, now safe.
- DFlash2 + DMLX_PREFILL_CHUNK: tap was dropped in the chunk loop ->
  DFlash2EmptyContext on every long prompt; tap threaded through,
  HiddenTap concatenates per-chunk rows.

Codex integration (B7): dmlx speaks the Responses API (user WIP committed
as src/server/responses/). `codex exec` connects and completes turns, but
the base model emits patch text instead of native tool_calls, and
pi-adapted emits pi's XML tool format — neither matches codex's
function_call item protocol, so apply_patch never executes. Needs a
tool-call translation layer in the responses path (model-native
tool_calls -> responses function_call items).
