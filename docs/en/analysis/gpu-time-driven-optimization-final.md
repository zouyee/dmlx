# DSV4 GPU-Time-Driven Optimization — Final Closure

## Decision

The authoritative GPU-time-driven optimization path is **closed as `NOT_PROVEN`**. Task 1.3
completed its sole authorized valid paired RSS session, but the frozen RSS-comparability
numerical gates failed. No threshold relaxation or additional session is authorized.
Consequently Task 2.2 could not collect enabled production timestamps or prove a CPU↔GPU
clock mapping, and Tasks 3.1–5.2 are terminally
`BLOCKED_BY_NOT_PROVEN_DEPENDENCY`.

No production optimization candidate was selected, implemented, benchmark-promoted, or
reverted by this spec.

## Accepted path

The following work remains accepted and retained:

- the reconciled exact-Q4 runtime source baseline;
- bounded raw completed-command-buffer timestamp storage and default-disabled production
  instrumentation;
- the deterministic offline clock-correlation analyzer and its fixtures;
- sampled completed-CB duration telemetry, explicitly non-authoritative for wall-time or
  exclusive attribution;
- native selector/residency correctness gates and subsequent diagnostic investigations;
- the production correctness fix that executes the shared expert exactly once on all-hot as
  well as cold-miss routed-MoE layers.

Task 2.3 is `APPROVED`: after its P4 correctness fix, the full standard N=20 benchmark passed
Paris and E2E 7/7, stayed below 44 GiB, and emitted positive sampled durations with zero
overflow. Those sampled duration masses rank diagnostic stages only and do not substitute
for Task 2.2 clock proof.

## Authoritative path termination

Task 1.3's final indivisible `B1 → I1 → B2 → I2` session passed lifecycle, correctness,
series-completeness, cleanup, and the independent 44 GiB safety limit. It failed the frozen
RSS-equivalence gates: pair 1, baseline drift, and aggregate mean absolute/relative gates did
not pass. Its terminal decision is `NOT_PROVEN`; the frozen protocol explicitly forbids a
rerun.

Task 2.2 therefore was not run. There is no real enabled Task 2.2 raw artifact, no strict
mapping report, and no `PROVEN` affine mapping. Exact-Q4 GPU intervals remain unknown. The
following consequences are terminal:

| Task | Disposition | Consequence |
|---|---|---|
| 2.2 | `BLOCKED_BY_NOT_PROVEN_DEPENDENCY` | No clock mapping decision or enabled timestamp artifact |
| 3.1 | `BLOCKED_BY_NOT_PROVEN_DEPENDENCY` | No `GPU_TIME_KNOWN` publication |
| 3.2 | `BLOCKED_BY_NOT_PROVEN_DEPENDENCY` | No reconciliation/coverage gate |
| 4.1 | `BLOCKED_BY_NOT_PROVEN_DEPENDENCY` | No authoritative exclusive decomposition |
| 4.2 | `BLOCKED_BY_NOT_PROVEN_DEPENDENCY` | No production branch selection |
| 5.1 | `BLOCKED_BY_NOT_PROVEN_DEPENDENCY` | No production candidate implementation |
| 5.2 | `BLOCKED_BY_NOT_PROVEN_DEPENDENCY` | No promotion run |

## Diagnostic investigations do not override closure

The 14-stage production diagnostic ranked `mhc_pre` first and retained its isolated dense-dot
path as an investigation candidate, without proving recoverable production time. A separate,
pre-registered standalone screen then compared SIMDgroup-wave8 and 24×8 mappings. Neither
met the frozen `candidate/serial <= 0.80` gate in all three processes, so Task 2.9 ended
`REJECTED`. No production implementation authorization was requested.

These diagnostics establish neither clock-domain mapping, wall/exclusive/recoverable time,
nor production speedup and cannot unlock Tasks 3.1–5.2.

## Retained baseline and thresholds

The frozen promotion throughput baseline remains **1.207636 tok/s**. Because no G4 candidate
was selected, no V3 comparison was performed and the baseline was neither replaced nor
reinterpreted. The later Task 2.3 diagnostic standard benchmark median of 1.018876 tok/s is
retained as diagnostic evidence only, not as a promotion baseline.

No acceptance, RSS, correctness, ratio, component-improvement, or throughput threshold was
lowered after observing results.

## Next direction

There is no authorized continuation inside this spec. Future work requires a separate,
reviewed spec with a new evidence contract. Legitimate options include redesigning RSS
comparability, establishing a different authoritative clock-mapping method, or proposing a
new bounded performance hypothesis. Such work must not resume sealed Task 1.3 sessions,
reuse Task 2.9 as production authority, or modify production kernels before its own gates.

## Closure validation

- every prior task has an approved, rejected, not-proven, or dependency-blocked terminal
  disposition;
- no Task 2.2 production collection or G3–G5 implementation was run during closure;
- no threshold was changed;
- final Task 2.9 evidence replay and terminal closure semantic review passed;
- final semantic review was `APPROVED` with zero issues;
- no `dmlx serve` remained after closure.
