# Phase Q4 — Copy-Free Bounded Admission Decision (Task 4.1)

**Decision: NOT_TRIGGERED**

**Verdict date:** 2026-08-20  
**Baseline:** Q0 (Task 0.3, retained 1.207636 tok/s, N=20, scalar decode)

---

## Evidence

### 1. CACHE_ADMISSION stage measured at 0 ms/token

Task 0.3 exact-Q4 telemetry (`enabled1-standard.exact-q4.json`, 12 steady tokens).
The complete set of stage IDs across all events at
`requests[N].tokens[M].events[K].stage` is `{0, 2, 3, 6, 7, 9, 12}`.
Stage 10 produces zero events in all 24 tokens (3 requests × 8 tokens):

| Stage | ID | ms/token | % of wall |
|---|---|---|---|
| MOE_GPU_WAIT | 9 | 307 | 72% |
| IO_EXPOSED_WAIT | 6 | 185 | 44% |
| ATTENTION_GPU_WAIT | 2 | 61 | 14% |
| ROUTING | 3 | 50 | 12% |
| LOGITS_SAMPLE | 12 | 3 | 0.8% |
| ATTENTION_ENCODE | 0 | <1 | <0.1% |
| MOE_ENCODE | 7 | <1 | <0.1% |
| **CACHE_ADMISSION (s10)** | **10** | **0** | **0%** |

Stage 10 (`DMLX_EXACT_Q4_STAGE_CACHE_ADMISSION`) is **absent from the artifact** — it
produces no events and therefore contributes zero nanoseconds to any token.

### 2. LRU admission is a stub — not implemented by default

Inspection of `src/metal_infer/engine.c` (3,519 lines, commit `9a7e1ef`):

- `cache_lru_admit_misses` — **not found** (0 matches)
- `moe_infer_set_experimental_cache` — declared in `engine.h:1341` but **not implemented**
  in any `.c` file
- `cache_lru_recency[N_LAYERS][N_EXPERTS]` — struct field exists in `engine.h:789` but
  the function that would write/read it has no body in engine.c
- `NATIVE_EXPERIMENTAL_CACHE` env flag — always `0` in Q0 baseline runs; the entire LRU
  branch is gated behind this flag

The experimental cache is a **design stub**: the engine struct carries metadata fields
(`cache_lru_clock`, `cache_lru_recency`, `cache_lru_admissions`, `cache_lru_evictions`)
but the admission logic itself was never written into engine.c. Enabling
`NATIVE_EXPERIMENTAL_CACHE=1` activates the policy flag but has no admission code path
to run, so the `exact_q4_records_admission` guard in the instrumentation block never
triggers.

### 3. No expert-sized memcpy on the default path

Current expert resolution priority:
1. SMELT fixed-hot RAM cache — zero-copy pointer
2. Predictive prefetch (`expert_buf_pred[k]`) — already DMA'd by io_pool
3. Blocking pread into `expert_buf[k]` — kernel I/O, no CPU memcpy

There is **no `memcpy` of expert payload** (~13.4 MB per expert) anywhere in the default
decode path. The only CPU copies per layer are small data-movement operations (residual
syncs, ~360 KB/layer total) that are already classified as per-token-unavoidable in the
Task 1.1 audit.

### 4. Stop gate condition

Per spec §4.1 stop gate:

> `NOT_TRIGGERED` ends Q4; current LRU remains rejected/default-off.

The condition is:
> `TRIGGERED` only if measured admission/copy is a material critical-path contributor
> and a bounded design can fit RSS budget.

Neither condition holds:
- Admission cost is **not measurable** (zero; feature is a stub)
- There is no expert-sized memcpy on the critical path
- The primary bottlenecks are MOE_GPU_WAIT (307 ms/token, 72%) and IO_EXPOSED_WAIT
  (185 ms/token, 44%), both of which are unrelated to cache admission

---

## Conclusion

**Q4 = NOT_TRIGGERED.**

The experimental bounded-LRU admission feature is not implemented in the default binary.
CACHE_ADMISSION stage is absent from all Q0 telemetry. No victim scan, expert-sized
memcpy, or wrapper/map update occurs on the measured critical path. Implementing and
measuring admission cost would require first completing the LRU implementation
(`moe_infer_set_experimental_cache` + `cache_lru_admit_misses`), which is itself a new
feature not covered by this spec.

Phase Q4 is complete. Tasks 4.2 and 4.3 do not run.  
Q0 (1.207636 tok/s) remains the accepted baseline for Q5.
