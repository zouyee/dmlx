# Exact-Q4 Latency Optimization — Final Report (Task 6.1)

**Generated:** 2026-08-21  
**Workspace commit (HEAD):** `9a7e1ef894ec5885793a1b551f57aa44ae0a8cc4`  
**Note:** `exact-q4-latency-final.md` is an untracked file generated at spec closure; all
referenced source/artifact changes are present in the tracked workspace working tree.  
**Spec:** `.kiro/specs/dsv4-exact-q4-latency/`  
**Status:** All phases complete — every task is in a valid terminal state.

---

## Accepted baseline

**Q0 — retained 1.207636 tok/s** (Task 0.3, N=20 fixed-hot scalar decode, M4 Pro 48GB,
`DeepSeek-V4-Flash-4bit`). No optimization phase produced a qualifying improvement;
this baseline is the definitive authoritative result of this spec.

### Q0 collection evidence

Authoritative repeated V3 A/B (baseline1 → enabled1 → baseline2 → enabled2):

| Role | Standard tok/s | Peak RSS | Correctness |
|---|---|---|---|
| baseline1 | 1.161221 | 18.0 GiB | PASS |
| enabled1  | 1.185331 | 20.0 GiB | PASS |
| baseline2 | 1.165278 | 20.0 GiB | PASS |
| enabled2  | 1.201045 | 20.0 GiB | PASS |

All gates passed: baseline drift within ±5%, enabled coverage ≥95%, RSS < 44 GiB,
correctness all PASS.

Aggregate artifact: `docs/en/analysis/exact-q4-task03-artifacts/exact-q4-task03-aggregate.json`  
Content SHA-256 (schema-validated, from `derived.content_sha256` field): `3de39aaf4163ba5bcc2db30f72c7ed0a12c3c268ddc78719254917202f1b16ae`  
Physical file SHA-256 (on-disk, untracked): `e75c8cab76b1c0b4716d40076fb0ce61d663775b6edb06153afd5bef21a51d32`  
(Content hash is validated by `python3 scripts/benchmark_report.py --validate-exact-q4-task03`; physical hash tracks the untracked file.)

---

## Phase-by-phase summary

### Phase Q0 — Telemetry and baseline

| Task | Status | Summary |
|---|---|---|
| 0.1 | ✅ APPROVED | Froze exact-Q4 telemetry ABI v1 — 14 stage IDs, 6 flags, 4 read sources, frozen fixture |
| 0.2.1 | ✅ APPROVED | Bounded per-layer producer storage and publication API |
| 0.2.2 | ✅ APPROVED | Expert sources and per-read lifecycle instrumentation |
| 0.2.3 | ✅ APPROVED | Scalar stages and native request lifecycle wiring |
| 0.2.4 | ✅ APPROVED | Enabled reconciliation and disabled overhead boundary (≤1%) |
| 0.2 | ✅ APPROVED | Aggregate: immutable per-read records, IO_EXPOSED_WAIT ownership, drain boundaries |
| 0.3 | ✅ APPROVED | Authoritative N=20 repeated baseline; all gates PASS; Q0 = 1.207636 tok/s |

### Phase Q1 — Remove steady-state host overhead

| Task | Status | Summary |
|---|---|---|
| 1.1 | ✅ APPROVED | Audit: 86 MTLBuffer alloc/token (weights_buf+zero_resid) identified as engine-lifetime candidates; blob_wrap stale-pointer and deinit hazards documented; 1116 µs/token encode overhead = 0.26% of wall |
| 1.2 | REJECTED (+0.69%) | Hoisted `buf_weights[43]`/`buf_zero_resid[43]`, added blob_wrap generation guard, fixed deinit leaks. V3 A/B: candidate median 1.202 vs baseline median 1.194 tok/s = +0.69% < required ≥5%. Root cause: encode overhead is 0.26% of wall; primary bottlenecks GPU-wait/I/O-wait are unaffected. Code retained (no regression). |

### Phase Q2 — Hide same-layer cold I/O

| Task | Status | Summary |
|---|---|---|
| 2.1 | ✅ APPROVED | Dependency partition design: shared expert CB independent of cold payload; state machine, buffer conflict proof, arithmetic order invariants, 9-unit focused test suite |
| 2.2 | REJECTED (−13.50%) | In-layer overlap: moved shared expert CB into read window (io_pool_start → shared CB → io_pool_wait_dependency). V3: candidate 0.988 vs baseline 1.142 = −13.50%. Root cause: M4 Pro unified LPDDR5X memory — GPU and SSD DMA share same bus; shared CB during cold reads causes GPU↔DMA bandwidth contention. All changes reverted. |

### Phase Q3 — Conditional incremental completion

| Task | Status | Summary |
|---|---|---|
| 3.1 | BLOCKED_BY_REJECTED_DEPENDENCY | Q2 REJECTED → Q3 blocked per stop gate |
| 3.2 | Not run | |
| 3.3 | Not run | |

### Phase Q4 — Conditional copy-free admission

| Task | Status | Summary |
|---|---|---|
| 4.1 | NOT_TRIGGERED | CACHE_ADMISSION stage absent from all telemetry (0 ms/token). LRU admission implementation is a stub (`moe_infer_set_experimental_cache` declared but not implemented; `cache_lru_admit_misses` not found in engine.c). No expert-sized memcpy on default decode path. Stop gate: NOT_TRIGGERED ends Q4. |
| 4.2 | Not run | |
| 4.3 | Not run | |

### Phase Q5 — Conditional exact-Q4 kernel

| Task | Status | Summary |
|---|---|---|
| 5.1 | NOT_TRIGGERED | MOE_GPU_WAIT is 72% of true token wall (principal bottleneck) but carries `GPU_TIME_UNKNOWN` flag — measurement is CPU-wait (CB submit to waitUntilCompleted), not an authoritative Metal GPU timer. Cannot satisfy spec's "isolated stage measurement is trustworthy" gate; cannot verify ≥10% stage-GPU-time improvement without `MTLCounterSampleBuffer`. Stop gate: NOT_TRIGGERED ends Q5 with no Metal edits. |
| 5.2 | Not run | |

---

## Accepted modifications to the binary

One set of structural changes from Task 1.2 is **retained** in the codebase despite
REJECTED promotion (structurally correct, no regression):

- `engine.h`: `buf_weights[N_LAYERS]`, `buf_zero_resid[N_LAYERS]` (engine-lifetime
  per-layer buffers); `blob_wrap_pread_ptr/pred_ptr/pred2_ptr[6]` (backing-pointer
  capture fields for future stale-wrapper detection; currently cleared in deinit,
  not yet read in the forward path — the working guard is pointer-identity comparison)
- `engine.c`: lazy-init + memcpy pattern for `buf_weights[layer_idx]` /
  `buf_zero_resid[layer_idx]` replacing per-call `newBufferWithBytes` /
  `newBufferWithLength` in `moe_forward_layer`; stale-pointer invalidation before
  NoCopy wrapper reuse; release loops in `moe_infer_deinit` for previously-leaked
  `buf_weights`, `buf_zero_resid`, and `blob_wrap_pread/pred/pred2` slots

`git diff --stat -- src/metal_infer/engine.c src/metal_infer/engine.h` shows 28
lines changed (23 insertions, 5 deletions) over two files — the complete Task 1.2
change set.

---

## Bottleneck analysis — why gains were elusive

The Q0 telemetry shows why no optimization phase succeeded:

| Stage | ms/token | % of wall | Character |
|---|---|---|---|
| MOE_GPU_WAIT (s9) | 307 | 72% | GPU-bound: gather6 MXFP4 kernel execution + CB overhead (GPU_TIME_UNKNOWN) |
| IO_EXPOSED_WAIT (s6) | 185 | 44% | I/O-bound: SSD pread blocking join at cold-miss experts |
| ATTENTION_GPU_WAIT (s2) | 61 | 14% | GPU-bound: MLA attention CB |
| ROUTING (s3) | 50 | 12% | CPU-bound: 256-candidate top-6 selection |

(Stages overlap; sum exceeds 100%. True wall = ~422 ms/token at N=20.)

**Structural constraint**: on the M4 Pro, GPU execution (via Metal) and SSD DMA both
compete for the unified LPDDR5X memory bus. Moving GPU work into the SSD read window
(Q2) directly increased contention and degraded throughput by 13.5%. This is a
fundamental architectural property of the unified memory platform:
- Prefetch (next-layer async pread) works because it fires after current-layer GPU completes,
  using idle bus time
- Same-layer overlap does not work because the bus is saturated during the expert read

**Primary remaining levers** (outside this spec's scope):
1. Reduce per-token pread count via better SMELT routing alignment (fewer cold misses)
2. Metal `MTLCounterSampleBuffer` instrumentation to obtain trustworthy GPU timers
   enabling justified kernel optimization (Q5 trigger requires this)
3. Expert quantization to reduce per-read payload size

---

## Spec completion certificate

All tasks are in valid terminal states per spec execution rules:

```
Q0: 0.1 APPROVED, 0.2.1–0.2.4 APPROVED, 0.2 APPROVED, 0.3 APPROVED
Q1: 1.1 APPROVED, 1.2 REJECTED (code retained, no revert required)
Q2: 2.1 APPROVED, 2.2 REJECTED (engine.c fully reverted)
Q3: 3.1 BLOCKED_BY_REJECTED_DEPENDENCY, 3.2–3.3 not run
Q4: 4.1 NOT_TRIGGERED, 4.2–4.3 not run
Q5: 5.1 NOT_TRIGGERED, 5.2 not run
6.1: this report
```

No phase was bypassed. Every REJECTED path records root cause. Every NOT_TRIGGERED
path binds to measured evidence. No threshold was lowered to avoid a REJECTED state.

**Accepted baseline: Q0 = 1.207636 tok/s (unchanged)**
