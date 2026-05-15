#!/usr/bin/env python3
"""Generate performance-benchmark.md from serve-mode perf + e2e data.

Usage: python3 _gen_report.py <perf_file> <e2e_file> <output_md>
Reads BM_* env vars for metadata.

The script reads the existing report at <output_md> to extract the previous
benchmark data (commit, prefill, steady-state, throughput) and uses it as
the "previous" comparison baseline in the new report.
"""
import sys, re, os

perf_file, e2e_file, report_path = sys.argv[1], sys.argv[2], sys.argv[3]
env = os.environ


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
steps = []
for line in open(perf_file):
    m = re.search(r'step (\d+) complete: ([\d.]+)ms.*hits=(\d+) misses=(\d+)', line)
    if m:
        steps.append((int(m[1]), float(m[2]), int(m[3]), int(m[4])))

prefill = steps[0][1] if steps else 0
steady = [s[1] for s in steps[2:]]  # skip step 1 (prefill) and step 2 (cold)
savg = sum(steady) / len(steady) if steady else 0
smin = min(steady) if steady else 0
smax = max(steady) if steady else 0
tput = 1000 / savg if savg > 0 else 0

total_hits = sum(s[2] for s in steps)
total_misses = sum(s[3] for s in steps)
hit_rate = total_hits / (total_hits + total_misses) * 100 if (total_hits + total_misses) > 0 else 0

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
smelt_experts = env.get("BM_SMELT_EXPERTS", "0.1")
cache_mb = env.get("BM_CACHE_MB", "10240")
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

| Token | Latency (ms) | Cache Hits | Cache Misses |
|-------|-------------|-----------|-------------|
{perf_rows}

**Summary**:
- Prefill (token 1): **{prefill:.1f}ms**
- Steady-state (token 3+): **{smin:.1f}-{smax:.1f}ms**, avg {savg:.1f}ms
- Throughput: **~{tput:.1f} tok/s**
- Cache hit rate: **{hit_rate:.1f}%** ({total_hits} hits / {total_misses} misses)

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
| Cache hit rate | — | **{hit_rate:.1f}%** | — |
| 100-token HTTP total | — | **{long_total}s** | — |
| Server RSS | — | **{server_rss} MB** | — |
| Startup time | — | **{startup_secs}s** | — |
| 7-Prompt pass rate | 7/7 | **{e2e_pass}/7** | — |

---

*Generated by `scripts/run_benchmark.sh` on {date}*
"""

with open(report_path, "w") as f:
    f.write(report)

print(f"Prefill: {prefill:.1f}ms | Steady: {savg:.1f}ms | {tput:.1f} tok/s | Cache: {hit_rate:.1f}% | E2E: {e2e_pass}/7")
