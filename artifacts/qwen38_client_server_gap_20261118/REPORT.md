# qwen38 client/server tok/s gap — 2026-11-18

Build: ReleaseFast, `dmlx serve --model qwen3.8-27b-4bit --port 8080` (NON-MTP,
single variable). Machine state caveat: ~6GB swap in use from earlier churn —
absolute tok/s are depressed vs a clean boot, but the GAP (the subject of
this measurement) is unaffected since both clocks see the same machine.

Raw data: `server.log` (per-request RequestLog = server-side truth),
`short.json` / `long8k.json` (per-SSE-chunk timestamps, python stdlib client).

## Results

| scenario | rep | prompt tok | out tok | client tok/s | server tok/s | gap (server/client) | client ITL med |
|---|---|---|---|---|---|---|---|
| short (数到100) | 0 | 67 | 100 | 14.61 | 14.30 | 0.98 | 70.2 ms |
| short | 1 | 67 | 100 | 14.59 | 14.28 | 0.98 | 70.5 ms |
| short | 2 | 67 | 100 | 14.55 | 14.24 | 0.98 | 70.6 ms |
| long 8k | 0 | 12,411 | 50 | 14.15 | 0.49* | — | 75.0 ms |
| long 8k | 1 | 12,411 | 50 | 14.10 | 13.40 | 0.95 | 75.3 ms |
| long 8k | 2 | 12,411 | 50 | 14.15 | 13.48 | 0.95 | 74.9 ms |
| burst 5×short | 0-4 | 63 | 60 | ~14.4 (steady) | 12.59 / 14.39 / 14.42 / 14.38 / 14.44 | ~1.0 | — |

* rep0 long = cold prefill (98.8s TTFT, 12411 tok → 126 tok/s); excluded
  from the gap. Reps 1-2 hit the prefix snapshot (TTFT 0.18s).

client tok/s = completion_tokens / (wall − TTFT), i.e. steady-state decode
rate as observed by the SSE client. server tok/s = RequestLog
tokens_per_sec. ITL median = median inter-chunk gap from SSE timestamps.

## Verdict

**Gap ≈ 0.95-1.0× across every scenario — the streaming pipeline is
healthy.** The historical DeepSeek-V4 pathology (44.7 server vs 1.02 client
tok/s, 30×, VM thrash blocking network I/O — flash-moe-alignment-analysis
§0) does NOT reproduce on qwen38: weights are fully resident (~15GB), no
swap-pressure-driven I/O stalls, SSE chunks leave the server at compute
speed.

Absolute per-token latency: client 70ms/token vs server-implied ~70ms —
delta ≤ 1ms/token. No flush/queue accumulation.

TTFT path likewise clean: warm short prompt TTFT 0.15s; snapshot-restored
12k prompt TTFT 0.18s.

## Consequence for prioritization

Client tok/s bottleneck is the COMPUTE side. Next optimizations belong to
kernel/scheduling routes (speculative depth policy, long-context sampled
accept — the accept_gpu work landed today, MTP/dflash2 routing), not the
streaming pipeline. Bare decode ~14.5 tok/s (swap-degraded; ~15-16 clean)
is the number to beat with speculation, not a floor imposed by I/O.

## If a gap ever reappears, check list (not executed)

1. TTFT gap ≫ decode gap → prefill queueing / snapshot restore stalls.
2. Steady ITL gap → SSE flush batching (per-token vs per-chunk flush),
   fiber scheduling under `waitForToken` spin.
3. RequestLog duration vs engine-internal phase timers (draft/verify/accept
   split) to localize.
4. `vm_stat` pageouts during the run — any nonzero pageout rate invalidates
   the "resident weights" premise.
