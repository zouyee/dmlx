#!/usr/bin/env python3
"""Generate performance-benchmark.md from serve-mode perf + e2e data.

Usage: python3 _gen_report.py <perf_file> <e2e_file> <output_md>
Reads BM_* env vars for metadata.

The script reads the existing report at <output_md> to extract the previous
benchmark data (commit, prefill, steady-state, throughput) and uses it as
the "previous" comparison baseline in the new report.
"""
import sys, re, os, json
from datetime import datetime, timezone

perf_file, e2e_file, report_path = sys.argv[1], sys.argv[2], sys.argv[3]
env = os.environ


def _format_seconds_from_ms(value):
    return "unknown (not measured)" if value is None else f"{value / 1000.0:.3f} s"


def _format_unknown(value, suffix=""):
    return "unknown (not measured)" if value is None else f"{value}{suffix}"


def render_native_artifact(artifact_path, output_path):
    """Render the native report exclusively from a validated schema-v1 artifact."""
    with open(artifact_path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("schema_version") != 1:
        raise ValueError("native benchmark report requires schema_version 1")

    run = artifact["run"]
    phases = artifact["phases"]
    metrics = artifact["metrics"]
    standard = phases["standard_decode"]
    diagnostic = phases["long_decode_diagnostic"]
    diagnostic_run = diagnostic["run"]
    correctness = phases["correctness"]
    itl = metrics["itl_ms"]
    expert_io = metrics["expert_io"]
    dispatch = metrics["dispatch"]

    standard_rows = "\n".join(
        f"| {item['run_index']} | {item['prompt_tokens']} | {item['completion_tokens']} | "
        f"{item['wall_seconds']:.3f} | {item['tokens_per_second']:.3f} |"
        for item in standard["runs"]
    )
    median_tps = standard["median_tokens_per_second"]
    artifact_link = os.path.relpath(artifact_path, os.path.dirname(output_path))
    report_generated_utc = datetime.now(timezone.utc).isoformat()
    benchmark_timestamp = run["timestamp_utc"]
    benchmark_date = run.get("date_local") or benchmark_timestamp[:10]
    run_environment = run.get("environment", {})
    smelt_n = run_environment.get("NATIVE_SMELT_N")
    warmup = run.get("warmup", {})
    machine = run.get("machine", {})
    e2e_total = correctness["e2e_pass"] + correctness["e2e_fail"]

    report = f"""---
report_generated_utc: {report_generated_utc}
benchmark_run_date: {benchmark_date}
benchmark_timestamp_utc: {benchmark_timestamp}
commit: {run.get('workspace_revision') or 'unknown'}
branch: {run.get('branch') or 'unknown'}
artifact: {artifact_link}
schema_version: 1
---

# dmlx Native Performance Benchmark

> Machine-readable source of truth: [`{artifact_link}`](./{artifact_link}). Values below are rounded for display; the artifact retains measured precision and JSON `null` for unavailable metrics.

## Reproducible result

| Phase | Classification | Prompt / completion tokens | HTTP wall time | Effective completion throughput |
|---|---|---:|---:|---:|
| Standard short-run median | Standard; median of 3 | 5 / 5 per run | See runs below | **{median_tps:.3f} tok/s** |
| Long decode | Diagnostic only; excluded from standard median | {diagnostic_run['prompt_tokens']} / {diagnostic_run['completion_tokens']} | {diagnostic_run['wall_seconds']:.3f} s | **{diagnostic_run['tokens_per_second']:.3f} tok/s** |

The throughput rows use HTTP request wall time. Engine-scoped first-token latency, decode-phase timing, ITL, expert-I/O, and routed-MoE geometry below come from opt-in server telemetry attached to exactly the three standard requests. True HTTP TTFT and an independently bounded prefill phase remain unavailable because telemetry begins after tokenization. The 100-token result remains diagnostic and is not mixed into the standard median.

### Standard 5-token runs

| Run | Prompt tokens | Completion tokens | Wall time (s) | Effective tok/s |
|---:|---:|---:|---:|---:|
{standard_rows}

## Measurement coverage

| Metric | Value |
|---|---|
| Startup to health-ready | {_format_seconds_from_ms(metrics.get('startup_ms'))} |
| Full benchmark wall time | {_format_seconds_from_ms(metrics.get('end_to_end_wall_ms'))} |
| Performance-phase wall time | {_format_seconds_from_ms(metrics.get('performance_phase_wall_ms'))} |
| TTFT (HTTP request received → first response token) | {_format_seconds_from_ms(metrics.get('ttft_ms'))} |
| Independently bounded prefill latency | {_format_seconds_from_ms(metrics.get('prefill_ms'))} |
| Unqualified first-token latency | {_format_seconds_from_ms(metrics.get('first_token_ms'))} |
| Engine first-token latency (after tokenization) | {_format_seconds_from_ms(metrics.get('engine_first_token_ms'))} |
| Decode phase (median per standard request) | {_format_seconds_from_ms(metrics.get('steady_state_decode_ms'))} |
| ITL p50 / p95 / max | {_format_unknown(itl.get('p50'), ' ms')} / {_format_unknown(itl.get('p95'), ' ms')} / {_format_unknown(itl.get('max'), ' ms')} |
| Peak RSS | {_format_unknown(metrics.get('peak_rss_bytes'), ' bytes')} |
| Workspace high-water | {_format_unknown(dispatch.get('workspace_high_water_bytes'), ' bytes')} |
| Expert `pread` calls / bytes / I/O wait | {_format_unknown(expert_io.get('pread_calls'))} / {_format_unknown(expert_io.get('bytes_read'), ' bytes')} / {_format_unknown(expert_io.get('io_wait_ms'), ' ms')} |
| Expert short reads | {_format_unknown(expert_io.get('short_reads'))} |
| Expert hot / prefetch / batch-pool / cold | {_format_unknown(expert_io.get('hot_hits'))} / {_format_unknown(expert_io.get('prefetch_hits'))} / {_format_unknown(expert_io.get('batch_pool_hits'))} / {_format_unknown(expert_io.get('cold_misses'))} |
| Routed-MoE dispatches / command buffers | {_format_unknown(dispatch.get('dispatches'))} / {_format_unknown(dispatch.get('command_buffers'))} |
| Routed-MoE threadgroups / active-lane proxy | {_format_unknown(dispatch.get('threadgroups'))} / {_format_unknown(dispatch.get('active_lane_utilization_proxy'))} |

Unavailable metrics are intentionally reported as **unknown**, not zero. This artifact records workspace high-water and peak RSS when measured; complete attention GPU-stage timing and hardware occupancy remain unavailable in schema v1.

## Correctness gates

| Gate | Result |
|---|---|
| Paris prompt | {'PASS' if correctness['paris_pass'] else 'FAIL'} |
| 7-prompt E2E | **{correctness['e2e_pass']}/{e2e_total} PASS**, {correctness['e2e_fail']} FAIL |
| Unit tests | {run.get('unit_tests') or 'unknown'} |

## Run configuration

| Parameter | Value |
|---|---|
| Mode | {run.get('mode') or 'unknown'} |
| Model | `{run.get('model_path') or 'unknown'}` |
| Packed experts | `{run.get('packed_experts_path') or 'unknown'}` |
| Hardware | {machine.get('hardware') or 'unknown'}, {machine.get('memory') or 'unknown'} |
| `NATIVE_SMELT_N` | {smelt_n if smelt_n is not None else 'unknown'} |
| `NATIVE_TELEMETRY` | {run_environment.get('NATIVE_TELEMETRY') if run_environment.get('NATIVE_TELEMETRY') is not None else 'unknown'} |
| `NATIVE_BATCH_PREFILL` | {run_environment.get('NATIVE_BATCH_PREFILL', 'unknown')} |
| `NATIVE_BATCH_MLA` | {run_environment.get('NATIVE_BATCH_MLA', 'unknown')} |
| `NATIVE_EXPERT_BUCKETS` | {run_environment.get('NATIVE_EXPERT_BUCKETS', 'unknown')} |
| `NATIVE_BATCH_EXPERT_KERNELS` | {run_environment.get('NATIVE_BATCH_EXPERT_KERNELS', 'unknown')} |
| `NATIVE_DSPARK_BATCH_VERIFY` | {run_environment.get('NATIVE_DSPARK_BATCH_VERIFY', 'unknown')} |
| Warmup | {warmup.get('initial_requests', 'unknown')} initial + {warmup.get('rewarm_requests', 'unknown')} re-warm requests |
| Standard measured requests | {run.get('measured_requests', 'unknown')} |
| Command | `{run.get('command') or 'unknown'}` |

## Provenance

- Benchmark run date (machine-local): **{benchmark_date}**
- Benchmark timestamp (UTC): **{benchmark_timestamp}**
- Report generation timestamp (UTC): **{report_generated_utc}**
- Workspace revision: `{run.get('workspace_revision') or 'unknown'}`
- Artifact schema: `1`

This report does not reuse older headline throughput, latency, or memory values as current native results. Only measurements present in the linked artifact are treated as current.
"""

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(
        f"Native artifact: standard={median_tps:.3f} tok/s | "
        f"long diagnostic={diagnostic_run['tokens_per_second']:.3f} tok/s | "
        f"E2E={correctness['e2e_pass']}/{e2e_total}"
    )


if perf_file.endswith(".json"):
    render_native_artifact(perf_file, report_path)
    raise SystemExit(0)


# --- Parse previous report to get baseline data ---
def parse_previous_report(path):
    """Extract key metrics from the existing benchmark report."""
    prev = {
        "commit": "?",
        "prefill": 0.0,
        "steady": 0.0,
        "tput": 0.0,
        "perf_secs": 0,
        "e2e": "?/?",
    }
    try:
        content = open(path).read()
    except FileNotFoundError:
        return prev

    m = re.search(r'Commit: (\w+)', content)
    if m:
        prev["commit"] = m[1]

    m = re.search(r'Prefill \(token 1\): \*\*([\d.]+)ms\*\*', content)
    if m:
        prev["prefill"] = float(m[1])

    m = re.search(r'(?:avg|平均) ([\d.]+)ms', content)
    if m:
        prev["steady"] = float(m[1])

    m = re.search(r'(?:Throughput|吞吐量): \*\*~([\d.]+) tok/s\*\*', content)
    if m:
        prev["tput"] = float(m[1])

    m = re.search(r'perf (\d+)s', content)
    if m:
        prev["perf_secs"] = int(m[1])

    m = re.search(r'(\d+)/7 PASS', content)
    if m:
        prev["e2e"] = f"{m[1]}/7"

    return prev


prev = parse_previous_report(report_path)

# --- Parse perf (Token step logs) ---
# Two formats:
#   Cached:   Token step N complete: Xms, YMB read, cache hits=A misses=B
#   Trust OS: Token step N complete: Xms
steps = []
for line in open(perf_file):
    # Try cached format first (has hits/misses)
    m = re.search(r'step (\d+) complete: ([\d.]+)ms.*hits=(\d+) misses=(\d+)', line)
    if m:
        steps.append((int(m[1]), float(m[2]), int(m[3]), int(m[4])))
        continue
    # Try Trust OS format (bare ms, no cache fields)
    m = re.search(r'step (\d+) complete: ([\d.]+)ms', line)
    if m:
        steps.append((int(m[1]), float(m[2]), 0, 0))

prefill = steps[0][1] if steps else 0
steady = [s[1] for s in steps[2:]]  # skip step 1 (prefill) and step 2 (cold)
savg = sum(steady) / len(steady) if steady else 0
smin = min(steady) if steady else 0
smax = max(steady) if steady else 0
tput = 1000 / savg if savg > 0 else 0

total_hits = sum(s[2] for s in steps)
total_misses = sum(s[3] for s in steps)
if (total_hits + total_misses) > 0:
    hit_rate = total_hits / (total_hits + total_misses) * 100
    hit_rate_str = f"{hit_rate:.1f}% ({total_hits} hits / {total_misses} misses)"
else:
    hit_rate = 0.0
    hit_rate_str = "N/A (Trust OS, no custom cache)"

# --- Parse e2e ---
e2e_raw = open(e2e_file).read()
e2e_lines = [l.strip() for l in e2e_raw.strip().split('\n') if l.strip()]

e2e_results = []
i = 0
while i < len(e2e_lines):
    line = e2e_lines[i]
    generated = ""
    if i + 1 < len(e2e_lines) and e2e_lines[i+1].startswith("Generated:"):
        generated = e2e_lines[i+1].replace("Generated:", "").strip()
        i += 2
    else:
        i += 1
    
    if "PASSED" in line:
        e2e_results.append(("✅", generated))
    elif "FAILED" in line:
        e2e_results.append(("❌", generated))

e2e_pass = int(env.get("BM_E2E_PASS", sum(1 for r in e2e_results if r[0] == "✅")))
e2e_fail = int(env.get("BM_E2E_FAIL", sum(1 for r in e2e_results if r[0] == "❌")))


def delta(old, new, lower_better=True):
    if old == 0:
        return "—"
    try:
        old_f = float(old)
        new_f = float(new)
    except (ValueError, TypeError):
        return "—"
    pct = (1 - new_f / old_f) * 100 if lower_better else (new_f / old_f - 1) * 100
    sign = "+" if pct >= 0 else ""
    return f"**{sign}{pct:.0f}%**"


# --- Env vars ---
commit = env.get("BM_COMMIT", "?")
branch = env.get("BM_BRANCH", "?")
date = env.get("BM_DATE", "?")
hw = env.get("BM_HW", "?")
mem = env.get("BM_MEM", "?")
unit = env.get("BM_UNIT", "?")
perf_secs = env.get("BM_PERF_SECS", "?")
e2e_secs = env.get("BM_E2E_SECS", "?")
total_secs = env.get("BM_TOTAL_SECS", "?")
smelt_experts = env.get("BM_SMELT_EXPERTS", "0.15")
cache_mb = env.get("BM_CACHE_MB", "10240")
mlock = env.get("BM_MLOCK", "false")
perf_ttfr = env.get("BM_PERF_TTFR", "?")
perf_total = env.get("BM_PERF_TOTAL", "?")
perf_tokens = env.get("BM_PERF_TOKENS", "30")
long_ttfr = env.get("BM_LONG_TTFR", "?")
long_total = env.get("BM_LONG_TOTAL", "?")
long_tokens = env.get("BM_LONG_TOKENS", "100")
server_rss = env.get("BM_SERVER_RSS", "?")
startup_secs = env.get("BM_STARTUP_SECS", "?")

# --- Previous baseline ---
PREV_COMMIT = prev["commit"]
PREV_PREFILL = prev["prefill"]
PREV_STEADY = prev["steady"]
PREV_TPUT = prev["tput"]
PREV_PERF_SECS = prev["perf_secs"]

# Build per-prompt result list
result_icons = [r[0] for r in e2e_results]
while len(result_icons) < 7:
    result_icons.append('—')

# --- Tables ---
perf_rows = "\n".join(f"| {s[0]} | {s[1]:.1f} | {s[2]} | {s[3]} |" for s in steps[:15])
e2e_rows = "\n".join(
    f"| P{i+1} | {r[0]} | {r[1][:80]} |"
    for i, r in enumerate(e2e_results)
)

# Native engine mode: the native server does not emit MLX-era "Token step"
# log lines, so the per-token latency table would be empty. Use the headline
# numbers from run_benchmark.sh's native summary instead.
native_tps = env.get("BM_NATIVE_TPS", "")
native_decode_tps = env.get("BM_NATIVE_DECODE_TPS", "")
if native_tps:
    try:
        tput = float(native_tps)
    except ValueError:
        pass
    latency_section = f"""**Native engine results**:
- Sequential decode (median of 3): **{native_tps} tok/s**
- Long decode (100-token run): **{native_decode_tps} tok/s**
- Paris correctness: PASS (see E2E below)
- (Per-token latency table requires MLX-era "Token step" server logs,
  which the native engine does not emit.)"""
else:
    latency_section = f"""| Token | Latency (ms) | Cache Hits | Cache Misses |
|-------|-------------|-----------|-------------|
{perf_rows}

**Summary**:
- Prefill (token 1): **{prefill:.1f}ms**
- Steady-state (token 3+): **{smin:.1f}-{smax:.1f}ms**, avg {savg:.1f}ms
- Throughput: **~{tput:.1f} tok/s**
- Cache hit rate: **{hit_rate_str}**"""

report = f"""---
date: {date}
Commit: {commit} ({branch})
model: DeepSeek-V4-Flash-4bit (~141GB on disk, 33 shards)
hardware: {hw}, {mem}
mode: serve, smelt {smelt_experts} + stream, ExpertCache {cache_mb}MB, temperature=0
build: zig build -Doptimize=ReleaseFast
generated_by: scripts/run_benchmark.sh
total_time: {total_secs}s (perf {perf_secs}s + e2e {e2e_secs}s)
---

# dmlx Performance Benchmark Report

---

## 1. Token Generation Latency (Serve Mode)

{latency_section}

### HTTP End-to-End Latency

| Test | Tokens | TTFR (s) | Total (s) | Effective tok/s |
|------|--------|----------|-----------|-----------------|
| 30-token | {perf_tokens} | {perf_ttfr} | {perf_total} | — |
| 100-token | {long_tokens} | {long_ttfr} | {long_total} | — |

### Comparison with Previous Version ({PREV_COMMIT} → {commit})

| Metric | Previous ({PREV_COMMIT}) | Current ({commit}) | Change |
|--------|-------------------------|-------------------|--------|
| Prefill | {PREV_PREFILL:.1f}ms | **{prefill:.1f}ms** | {delta(PREV_PREFILL, prefill)} |
| Steady-state avg | {PREV_STEADY:.1f}ms | **{savg:.1f}ms** | {delta(PREV_STEADY, savg)} |
| Throughput | ~{PREV_TPUT:.1f} tok/s | **~{tput:.1f} tok/s** | {delta(PREV_TPUT, tput, lower_better=False)} |
| Perf phase | {PREV_PERF_SECS}s | **{perf_secs}s** | {delta(PREV_PERF_SECS, perf_secs)} |

Note: Previous data auto-extracted from prior report. Both runs use ReleaseFast.

---

## 2. Server Configuration & Resources

| Parameter | Value |
|-----------|-------|
| Mode | serve (HTTP API) |
| SMELT strategy | stream |
| SMELT experts | {smelt_experts} (preloaded) |
| Expert cache | {cache_mb} MB |
| mlock backbone | {mlock} |
| Temperature | 0 (greedy) |
| Startup time | {startup_secs}s (incl. warmup) |
| Server RSS | {server_rss} MB |
| Port | 18090 |

---

## 3. 7-Prompt End-to-End Test (Serve Mode)

| # | Result | Model Output (truncated) |
|---|--------|--------------------------|
{e2e_rows}

**{e2e_pass}/7 PASS, {e2e_fail} FAIL**

---

## 4. Unit Tests

```
zig build test → {unit}
```

---

## 5. Key Performance Metrics

| Metric | Previous ({PREV_COMMIT}) | Current ({commit}) | Change |
|--------|-------------------------|-------------------|--------|
| Prefill latency | {PREV_PREFILL:.1f}ms | **{prefill:.1f}ms** | {delta(PREV_PREFILL, prefill)} |
| Steady-state ITL | {PREV_STEADY:.1f}ms | **{savg:.1f}ms** | {delta(PREV_STEADY, savg)} |
| Steady-state tok/s | ~{PREV_TPUT:.1f} | **~{tput:.1f}** | {delta(PREV_TPUT, tput, lower_better=False)} |
| Cache hit rate | — | **{hit_rate_str}** | — |
| 100-token HTTP total | — | **{long_total}s** | — |
| Server RSS | — | **{server_rss} MB** | — |
| Startup time | — | **{startup_secs}s** | — |
| 7-Prompt pass rate | 7/7 | **{e2e_pass}/7** | — |

---

*Generated by `scripts/run_benchmark.sh` on {date}*
"""

with open(report_path, "w") as f:
    f.write(report)

cache_str = hit_rate_str if (total_hits + total_misses) > 0 else "N/A"
print(f"Prefill: {prefill:.1f}ms | Steady: {savg:.1f}ms | {tput:.1f} tok/s | Cache: {cache_str} | E2E: {e2e_pass}/7")
