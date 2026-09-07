# Exact-Q4 latency baseline (Task 0.3)

> Generated from the canonical aggregate. Task 0.3 remains unchecked pending collection/review.

## Collection summary

- Retained baseline: **1.207636 tok/s**
- Aggregate decision: **PASS**
- Commit: 9a7e1ef (workspace_revision from identity)

## Per-run results

| Role | Pair | Standard tok/s | Peak RSS GiB | Correctness | Coverage |
|---|---:|---:|---:|---|---|
| baseline1 | 1 | 1.161221 | 18.1 | PASS | - |
| enabled1 | 1 | 1.185331 | 21.0 | PASS | long=99%, standard=99% |
| baseline2 | 2 | 1.165278 | 20.2 | PASS | - |
| enabled2 | 2 | 1.201045 | 20.1 | PASS | long=99%, standard=99% |

## Gates

- baseline_within_retained_5_percent_or_root_caused: **PASS**
- correctness_all_runs: **PASS**
- enabled_coverage_at_least_95_percent: **PASS**
- peak_rss_below_44_gib: **PASS**

## Artifact

Aggregate: `/Users/zouyee/work/code/dmlx/docs/en/analysis/exact-q4-task03-artifacts/exact-q4-task03-aggregate.json`

Content SHA-256: `3de39aaf4163ba5bcc2db30f72c7ed0a12c3c268ddc78719254917202f1b16ae`
