---
report_generated_utc: 2026-08-19T07:35:29.545543+00:00
benchmark_run_date: 2026-08-19
benchmark_timestamp_utc: 2026-08-19T07:35:29.391791+00:00
commit: 9a7e1ef
branch: main
artifact: baseline2-benchmark.json
schema_version: 1
---

# dmlx Native Performance Benchmark

> Machine-readable source of truth: [`baseline2-benchmark.json`](./baseline2-benchmark.json). Values below are rounded for display; the artifact retains measured precision and JSON `null` for unavailable metrics.

## Reproducible result

| Phase | Classification | Prompt / completion tokens | HTTP wall time | Effective completion throughput |
|---|---|---:|---:|---:|
| Standard short-run median | Standard; median of 3 | 5 / 5 per run | See runs below | **1.165 tok/s** |
| Long decode | Diagnostic only; excluded from standard median | 12 / 100 | 61.729 s | **1.620 tok/s** |

The throughput rows use HTTP request wall time. Engine-scoped first-token latency, decode-phase timing, ITL, expert-I/O, and routed-MoE geometry below come from opt-in server telemetry attached to exactly the three standard requests. True HTTP TTFT and an independently bounded prefill phase remain unavailable because telemetry begins after tokenization. The 100-token result remains diagnostic and is not mixed into the standard median.

### Standard 5-token runs

| Run | Prompt tokens | Completion tokens | Wall time (s) | Effective tok/s |
|---:|---:|---:|---:|---:|
| 1 | 5 | 5 | 4.291 | 1.165 |
| 2 | 5 | 5 | 4.518 | 1.107 |
| 3 | 5 | 5 | 4.219 | 1.185 |

## Measurement coverage

| Metric | Value |
|---|---|
| Startup to health-ready | 7.161 s |
| Full benchmark wall time | 267.000 s |
| Performance-phase wall time | 133.000 s |
| TTFT (HTTP request received → first response token) | unknown (not measured) |
| Independently bounded prefill latency | unknown (not measured) |
| Unqualified first-token latency | unknown (not measured) |
| Engine first-token latency (after tokenization) | 2.461 s |
| Decode phase (median per standard request) | 1.792 s |
| ITL p50 / p95 / max | 430.28499999999997 ms / 532.93085 ms / 558.337 ms |
| Peak RSS | 21656174592 bytes |
| Workspace high-water | unknown (not measured) |
| Expert `pread` calls / bytes / I/O wait | 5256 / 70269272064 bytes / 6296.408 ms |
| Expert short reads | 0 |
| Expert hot / prefetch / batch-pool / cold | 936 / 0 / 0 / 5256 |
| Routed-MoE dispatches / command buffers | 3096 / 1032 |
| Routed-MoE threadgroups / active-lane proxy | 4771968 / 1.0 |

Unavailable metrics are intentionally reported as **unknown**, not zero. This artifact records workspace high-water and peak RSS when measured; complete attention GPU-stage timing and hardware occupancy remain unavailable in schema v1.

## Correctness gates

| Gate | Result |
|---|---|
| Paris prompt | PASS |
| 7-prompt E2E | **7/7 PASS**, 0 FAIL |
| Unit tests | FAIL |

## Run configuration

| Parameter | Value |
|---|---|
| Mode | native |
| Model | `/Users/zouyee/models/DeepSeek-V4-Flash-4bit` |
| Packed experts | `/Users/zouyee/models/DeepSeek-V4-Flash-4bit/packed_experts` |
| Hardware | Apple M4 Pro, 48GB |
| `NATIVE_SMELT_N` | 20 |
| `NATIVE_TELEMETRY` | 1 |
| `NATIVE_BATCH_PREFILL` | 0 |
| `NATIVE_BATCH_MLA` | 0 |
| `NATIVE_EXPERT_BUCKETS` | 0 |
| `NATIVE_BATCH_EXPERT_KERNELS` | 0 |
| `NATIVE_DSPARK_BATCH_VERIFY` | 0 |
| Warmup | 5 initial + 4 re-warm requests |
| Standard measured requests | 3 |
| Command | `NATIVE_SMELT_N=20 NATIVE_EXPERIMENTAL_CACHE=0 NATIVE_BATCH_PREFILL=0 NATIVE_BATCH_MLA=0 NATIVE_EXPERT_BUCKETS=0 NATIVE_BATCH_EXPERT_KERNELS=0 NATIVE_DSPARK_BATCH_VERIFY=0 DSPARK_DIR='' SKIP_PURGE=0 bash scripts/run_benchmark.sh --native /Users/zouyee/models/DeepSeek-V4-Flash-4bit` |

## Provenance

- Benchmark run date (machine-local): **2026-08-19**
- Benchmark timestamp (UTC): **2026-08-19T07:35:29.391791+00:00**
- Report generation timestamp (UTC): **2026-08-19T07:35:29.545543+00:00**
- Workspace revision: `9a7e1ef`
- Artifact schema: `1`

This report does not reuse older headline throughput, latency, or memory values as current native results. Only measurements present in the linked artifact are treated as current.
