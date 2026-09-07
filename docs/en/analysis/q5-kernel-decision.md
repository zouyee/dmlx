# Phase Q5 — Exact-Q4 Kernel Decision (Task 5.1)

**Decision: NOT_TRIGGERED**

**Verdict date:** 2026-08-20  
**Baseline:** Q0 (Task 0.3, retained 1.207636 tok/s, N=20, scalar decode)  
**Q1/Q2/Q3/Q4 status:** Q1 REJECTED (+0.69%); Q2 REJECTED (−13.50%); Q3 BLOCKED; Q4 NOT_TRIGGERED

---

## Stage ranking after all accepted Q0 modifications

Source: Task 0.3 exact-Q4 telemetry, `requests[N].tokens[M].events[K].stage`.

### Standard decode (enabled1+enabled2, 12 steady tokens each)

| Stage | ID | ms/token | % of stage sum* | Events/token |
|---|---|---|---|---|
| MOE_GPU_WAIT | 9 | 307 | 50% | 43 (one per layer) |
| IO_EXPOSED_WAIT | 6 | 185 | 30% | ~40 |
| ATTENTION_GPU_WAIT | 2 | 61 | 10% | 43 |
| ROUTING | 3 | 50 | 8% | 43 |
| LOGITS_SAMPLE | 12 | 3 | 0.8% | 1 |
| ATTENTION_ENCODE | 0 | <1 | <0.1% | 43 |
| MOE_ENCODE | 7 | <1 | <0.1% | 43 |
| CACHE_ADMISSION | 10 | 0 | 0% | 0 |

\* Stages overlap (GPU dispatch and SSD reads run concurrently across layers), so their
sum (606 ms) exceeds actual token wall time (~437 ms average). As a fraction of true
wall time: MOE_GPU_WAIT ≈ **72%**, IO_EXPOSED_WAIT ≈ 44%, ATTN_GPU_WAIT ≈ 14%.
The "% of stage sum" column reflects each stage's share of all measured stage durations.

Exclusive attribution (per steady token, **222 steady tokens pooled from all four enabled
canonical runs: enabled1-standard 12 + enabled2-standard 12 + enabled1-long 99 +
enabled2-long 99**; long-run weighted due to token count):

| Category | ms/token | % |
|---|---|---|
| `gpu_unknown_cpu_wait_ns` | 255 | 49% |
| `io_exposed_wait_ns` | 198 | 38% |
| `cpu_active_ns` | 62 | 12% |
| `unaccounted_ns` | <1 | 0% |

### Long diagnostic (enabled1+enabled2, 99 steady tokens each)

| Stage | ms/token | % |
|---|---|---|
| MOE_GPU_WAIT (s9) | 378–401 | 53% |
| IO_EXPOSED_WAIT (s6) | 199–201 | 27% |
| ATTN_GPU_WAIT (s2) | 69–77 | 10% |
| ROUTING (s3) | 55–60 | 8% |

MOE_GPU_WAIT is consistently **50–53% of token wall time** across standard and long
decode workloads. It is the single largest remaining bottleneck.

---

## Trigger condition analysis

Per spec §5.1:

> `TRIGGERED` only if MoE GPU execution is a **principal remaining bottleneck**
> **and isolated stage measurement is trustworthy**.

### Condition 1 — MoE GPU execution is a principal remaining bottleneck

**MET.** MOE_GPU_WAIT accounts for 50–53% of steady-state token wall time, 43 events
per token (one per layer). No other stage approaches this contribution. The kernel
responsible is `pipe_gather6_gate_up` + `pipe_gather6_down` + `pipe_moe_combine` inside
`moe_forward_layer`, dispatched via `commandBufferWithUnretainedReferences` per layer.

### Condition 2 — Isolated stage measurement is trustworthy

**NOT MET.** The `MOE_GPU_WAIT` stage (s9) carries the
`DMLX_EXACT_Q4_EVENT_GPU_TIME_UNKNOWN` flag. Per the telemetry design (design.md §3.2):

> "Until an absolute Metal timestamp mapping to the process monotonic clock is separately
> proven, attention and MoE execution remain unknown: emit CPU completion waits with
> `GPU_TIME_UNKNOWN` and do not emit fabricated `GPU_EXECUTION` spans."

The 307 ms/token figure is a **CPU-wait measurement** — the time from `[cb commit]` to
`[cb waitUntilCompleted]` returning on the CPU thread. It includes:

- Actual GPU kernel execution time
- Metal command buffer scheduling latency
- GPU driver overhead
- Any CPU–GPU synchronization cost

These cannot be separated with the current instrumentation. A kernel change that
reduces GPU computation time by 10% might produce a 3–8% wall-clock change in
`MOE_GPU_WAIT` depending on how much of the wait is scheduling versus execution — the
spec requires "stage GPU time improves >=10%" as a proof predicate, but the current
stage does not measure GPU time directly.

To satisfy the "trustworthy" gate, the project would need to add authoritative
`MTLCounterSampleBuffer` timestamps (Metal GPU timer queries) to the MoE command buffer,
proving the kernel execution portion separately from scheduling. This is a non-trivial
instrumentation investment outside the current spec scope.

### Why a wall-clock A/B alone is insufficient

A V3 benchmark could show throughput change from a kernel modification, but:

1. The spec requires `TRIGGERED` based on "isolated stage measurement" — a trusted
   measurement that can be attributed exclusively to the kernel.
2. Without GPU timestamps, any observed throughput change conflates kernel speed,
   scheduling, page-fault/DMA pressure, and thermal variation (~8% run-to-run noise
   observed in Q1/Q2 A/B data).
3. The spec's ≥10% stage-GPU-time criterion cannot be verified without a GPU timer.

---

## Conclusion

**Q5 = NOT_TRIGGERED.**

`MOE_GPU_WAIT` is the dominant remaining bottleneck (50–53% of token wall) and
`pipe_gather6_gate_up`/`pipe_gather6_down` are clear kernel candidates. However, the
existing `GPU_TIME_UNKNOWN` measurement does not provide an authoritative GPU timer that
could certify a ≥10% kernel execution improvement independently of scheduling/CB
overhead. The trigger condition "isolated stage measurement is trustworthy" is not
satisfied with current instrumentation.

Phase Q5 is complete with no Metal edits. Tasks 5.2 does not run.  
Q0 (1.207636 tok/s) remains the accepted baseline for Task 6.1 (final closure).

---

## Accepted changes at Q0

| Phase | Task | Status | Net effect |
|---|---|---|---|
| Q1 | 1.1 Audit | APPROVED | Inventory only; no runtime change |
| Q1 | 1.2 Hoist weights/zero_resid | REJECTED (+0.69%) | Code retained; not a performance change |
| Q2 | 2.1 Partition design | APPROVED | Design only; no runtime change |
| Q2 | 2.2 Overlap implementation | REJECTED (−13.50%) | Fully reverted |
| Q3 | 3.1 Incremental trigger | BLOCKED_BY_REJECTED_DEPENDENCY | Not run |
| Q4 | 4.1 Admission trigger | NOT_TRIGGERED | Not run |
| Q5 | 5.1 Kernel trigger | NOT_TRIGGERED | Not run |

**Effective Q0 binary** is the unmodified baseline. The hoisting changes from Task 1.2
(`buf_weights[43]`, `buf_zero_resid[43]`, `blob_wrap` generation guard, deinit fix)
are retained in engine.c/engine.h as they are structurally correct and produce no
regression, but they did not move the accepted baseline.
