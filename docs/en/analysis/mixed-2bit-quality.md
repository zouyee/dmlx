# Mixed-2bit offline quality gate

## Decision

**REJECTED. Tasks 7.4–7.6 are not authorized. Q4 remains authoritative and default.**

Task 7.3 applies the unchanged thresholds frozen by `mixed-2bit-rfc.md`. The first mandatory artifact-and-provenance gate fails: the repository contains only the reduced Task 7.2 fixture, whose canonical sidecar explicitly records `runtime_qualified=false`. It does not contain the frozen calibration and held-out manifests, imatrix, allocation trace, complete 43-layer mixed artifact, matched uniform-int2 control, pinned MPFR evaluator, or Unicode 15.1 source/table bindings required by the RFC.

The checked-in `dsv4_native_logits.json` contains one authoritative Q4 reference vector only. It contains no mixed, naive uniform-int2, or matched-control logits. Consequently, tensor reconstruction, held-out logits, canonical-seven, larger-suite, and serialized-storage gates are **unavailable**, not measured passes. The reduced fixture proves bounded encoding/validation behavior only and is not full-model quality evidence.

No speed result can waive this decision, and no threshold is relaxed. Producing full-model artifacts now would violate the Task 7.2 rollback gate, so the missing evidence is archived as a failed feasibility result rather than generated after inspecting held-out gates.

## Frozen gates

The machine-readable report preserves the complete predeclared threshold families:

- tensor: weighted NRMSE `<=0.090`, weighted SNR `>=20.9151498112135 dB`, int2-unit p99/max NRMSE `<=0.35/0.75`, absolute-error multiplier `<=8`;
- logits: top-1 `>=0.98`, mean/p1 top-5 `>=0.970/0.600`, mean/p99 JS `<=0.0020/0.020`, mean/p99 centered NRMSE `<=0.060/0.150`;
- matched control: JS reduction `>=0.20`, centered-NRMSE reduction `>=0.15`, top-1 improvement `>=1.0` percentage point;
- E2E: Q4 and mixed both `7/7`, immutable suite at least 256 prompts, overall/stratum deltas `>=-1.0/-3.0` points, paired Q4-pass retention `>=0.95`;
- storage: complete serialized mixed bytes `<=0.75` of Q4.

## Reproduction

```bash
python3 scripts/verify_mixed_2bit.py --quality-self-test
python3 scripts/verify_mixed_2bit.py --quality-report docs/en/analysis/mixed-2bit-quality.json
shasum -a 256 docs/en/analysis/mixed-2bit-quality.json
```

The JSON is canonical (sorted keys, compact separators, one trailing newline), contains no timestamp or host-dependent path, binds every inspected input by SHA-256, and explicitly distinguishes `failed` from `unavailable`. Its expected decision is `REJECTED`, with `runtime_work_authorized=false` and `stop_tasks=["7.4","7.5","7.6"]`.
