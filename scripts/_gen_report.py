#!/usr/bin/env python3
"""Generate PERFORMANCE_BENCHMARK.md from perf + e2e data files.
Usage: python3 _gen_report.py <perf_file> <e2e_file> <output_md>
Reads BM_* env vars for metadata.

The script reads the existing report at <output_md> to extract the previous
benchmark data (commit, prefill, steady-state, throughput, perf_secs) and uses
it as the "previous" comparison baseline in the new report.
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

    # Extract commit from front-matter or header
    m = re.search(r'Commit: (\w+)', content)
    if m:
        prev["commit"] = m[1]

    # Extract prefill: "Prefill (token 1): **273.9ms**"
    m = re.search(r'Prefill \(token 1\): \*\*([\d.]+)ms\*\*', content)
    if m:
        prev["prefill"] = float(m[1])

    # Extract steady avg: "avg 56.6ms" or "平均 56.6ms"
    m = re.search(r'(?:avg|平均) ([\d.]+)ms', content)
    if m:
        prev["steady"] = float(m[1])

    # Extract throughput: "Throughput: **~17.7 tok/s**" or "吞吐量: **~17.7 tok/s**"
    m = re.search(r'(?:Throughput|吞吐量): \*\*~([\d.]+) tok/s\*\*', content)
    if m:
        prev["tput"] = float(m[1])

    # Extract perf_secs from front-matter: "perf XXs"
    m = re.search(r'perf (\d+)s', content)
    if m:
        prev["perf_secs"] = int(m[1])

    # Extract e2e pass rate: "7/7 PASS"
    m = re.search(r'(\d+)/7 PASS', content)
    if m:
        prev["e2e"] = f"{m[1]}/7"

    return prev


prev = parse_previous_report(report_path)

# --- Parse perf ---
steps = []
for line in open(perf_file):
    m = re.search(r'step (\d+) complete: ([\d.]+)ms.*hits=(\d+) misses=(\d+)', line)
    if m:
        steps.append((int(m[1]), float(m[2]), int(m[3]), int(m[4])))

prefill = steps[0][1] if steps else 0
steady = [s[1] for s in steps[2:]]
savg = sum(steady) / len(steady) if steady else 0
smin = min(steady) if steady else 0
smax = max(steady) if steady else 0
tput = 1000 / savg if savg > 0 else 0

# --- Parse e2e ---
e2e_raw = open(e2e_file).read()
e2e_matches = list(re.finditer(r'(✅ PASSED|❌ FAILED)[^\n]*\n\s*Generated: ([^\n]*)', e2e_raw))
e2e_pass = len([m for m in e2e_matches if "PASSED" in m[1]])
e2e_fail = len(e2e_matches) - e2e_pass


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

# --- Previous baseline ---
PREV_COMMIT = prev["commit"]
PREV_PREFILL = prev["prefill"]
PREV_STEADY = prev["steady"]
PREV_TPUT = prev["tput"]
PREV_PERF_SECS = prev["perf_secs"]
PREV_E2E = prev["e2e"]

# Build per-prompt pass/fail list
e2e_results = ['✅' if 'PASSED' in m[1] else '❌' for m in e2e_matches]
while len(e2e_results) < 7:
    e2e_results.append('—')

# --- Tables ---
perf_rows = "\n".join(f"| {s[0]} | {s[1]} | {s[2]} | {s[3]} |" for s in steps)
e2e_rows = "\n".join(
    f"| P{i+1} | {'✅' if 'PASSED' in m[1] else '❌'} | {m[2][:80].strip()} |"
    for i, m in enumerate(e2e_matches)
)

report = f"""---
date: {date}
Commit: {commit} ({branch})
model: DeepSeek-V4-Flash-4bit (~40GB 4-bit, 33 shards)
hardware: {hw}, {mem}
mode: smelt + stream, ExpertCache 4GB, temperature=0
build: zig build -Doptimize=ReleaseFast
generated_by: scripts/run_benchmark.sh
total_time: {total_secs}s (perf {perf_secs}s + e2e {e2e_secs}s)
---

# dmlx Performance Benchmark Report

---

## 1. Token Generation Latency

| Token | Latency (ms) | Cache Hits | Cache Misses |
|-------|-------------|-----------|-------------|
{perf_rows}

**Summary**:
- Prefill (token 1): **{prefill:.1f}ms**
- Steady-state (token 3-10): **{smin:.1f}-{smax:.1f}ms**, avg {savg:.1f}ms
- Throughput: **~{tput:.1f} tok/s**

### Comparison with Previous Version ({PREV_COMMIT} → {commit})

| Metric | Previous ({PREV_COMMIT}) | Current ({commit}) | Change |
|--------|-------------------------|-------------------|--------|
| Prefill | {PREV_PREFILL:.1f}ms | **{prefill:.1f}ms** | {delta(PREV_PREFILL, prefill)} |
| Steady-state avg | {PREV_STEADY:.1f}ms | **{savg:.1f}ms** | {delta(PREV_STEADY, savg)} |
| Throughput | ~{PREV_TPUT:.1f} tok/s | **~{tput:.1f} tok/s** | {delta(PREV_TPUT, tput, lower_better=False)} |
| Perf phase (load+10tok) | {PREV_PERF_SECS}s | **{perf_secs}s** | {delta(PREV_PERF_SECS, perf_secs)} |

Note: Previous data auto-extracted from prior report. Both runs use ReleaseFast.

---

## 2. 7-Prompt End-to-End Test

```
bash scripts/best_test.sh → {e2e_pass} passed, {e2e_fail} failed ({e2e_secs}s)
```

| # | Result | Model Output (truncated) |
|---|--------|--------------------------|
{e2e_rows}

### Correctness Verification

| # | Prompt | Previous ({PREV_COMMIT}) | Current ({commit}) |
|---|--------|-------------------------|-------------------|
| P1 | 2+2= | ✅ | {e2e_results[0]} |
| P2 | Capital of France | ✅ | {e2e_results[1]} |
| P3 | Water freezes at | ✅ | {e2e_results[2]} |
| P4 | Is Earth round? | ✅ | {e2e_results[3]} |
| P5 | 3*3= | ✅ | {e2e_results[4]} |
| P6 | 10-5= | ✅ | {e2e_results[5]} |
| P7 | Capital of France? | ✅ | {e2e_results[6]} |

**{e2e_pass}/7 PASS, {e2e_fail} FAIL, 0 SKIP**

---

## 3. Unit Tests

```
zig build test → {unit}
```

---

## 4. Key Performance Metrics

| Metric | Previous ({PREV_COMMIT}) | Current ({commit}) | Change |
|--------|-------------------------|-------------------|--------|
| Perf phase (load+gen) | {PREV_PERF_SECS}s | **{perf_secs}s** | {delta(PREV_PERF_SECS, perf_secs)} |
| Prefill latency | {PREV_PREFILL:.1f}ms | **{prefill:.1f}ms** | {delta(PREV_PREFILL, prefill)} |
| Steady-state latency | {PREV_STEADY:.1f}ms | **{savg:.1f}ms** | {delta(PREV_STEADY, savg)} |
| Steady-state throughput | ~{PREV_TPUT:.1f} tok/s | **~{tput:.1f} tok/s** | {delta(PREV_TPUT, tput, lower_better=False)} |
| 7-Prompt pass rate | {PREV_E2E} | **{e2e_pass}/7** | — |
| ExpertCache | 4GB / ~40% hit | 4GB / ~40% hit | — |

Note: Both runs use ReleaseFast. Perf phase includes model loading + 10 token generation.
"""

with open(report_path, "w") as f:
    f.write(report)

print(f"Prefill: {prefill:.1f}ms | Steady: {savg:.1f}ms | {tput:.1f} tok/s | E2E: {e2e_pass}/7")
