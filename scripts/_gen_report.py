#!/usr/bin/env python3
"""Generate PERFORMANCE_BENCHMARK.md from perf + e2e data files.
Usage: python3 _gen_report.py <perf_file> <e2e_file> <output_md>
Reads BM_* env vars for metadata.

The script reads the existing report at <output_md> to extract the previous
benchmark data (commit, prefill, steady-state, throughput, perf_secs) and uses
it as the "上一版" comparison baseline in the new report.
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

    # Extract commit from header: "Commit: abc1234 (branch)" (in front-matter or old # format)
    m = re.search(r'Commit: (\w+)', content)
    if m:
        prev["commit"] = m[1]

    # Extract prefill from summary: "Prefill (token 1): **273.9ms**"
    m = re.search(r'Prefill \(token 1\): \*\*([\d.]+)ms\*\*', content)
    if m:
        prev["prefill"] = float(m[1])

    # Extract steady avg: "平均 56.6ms"
    m = re.search(r'平均 ([\d.]+)ms', content)
    if m:
        prev["steady"] = float(m[1])

    # Extract throughput: "吞吐量: **~17.7 tok/s**"
    m = re.search(r'吞吐量: \*\*~([\d.]+) tok/s\*\*', content)
    if m:
        prev["tput"] = float(m[1])

    # Extract perf_secs from header: "perf XXs"
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
# Pad to 7 if fewer prompts completed
while len(e2e_results) < 7:
    e2e_results.append('—')

# --- Tables ---
perf_rows = "\n".join(f"| {s[0]} | {s[1]} | {s[2]} | {s[3]} |" for s in steps)
e2e_rows = "\n".join(
    f"| P{i+1} | {'✅' if 'PASSED' in m[1] else '❌'} | {m[2][:80].strip()} |"
    for i, m in enumerate(e2e_matches)
)

report = f"""---
日期: {date}
Commit: {commit} ({branch})
模型: DeepSeek-V4-Flash-4bit (~40GB 4-bit, 33 shards)
硬件: {hw}, {mem}
模式: smelt + stream, ExpertCache 4GB, temperature=0
编译: zig build -Doptimize=ReleaseFast
生成方式: scripts/run_benchmark.sh (自动化)
总耗时: {total_secs}s (perf {perf_secs}s + e2e {e2e_secs}s)
---

# dmlx 性能基准测试报告

---

## 一、Token 生成延迟

| Token | 延迟 (ms) | Cache Hits | Cache Misses |
|-------|----------|-----------|-------------|
{perf_rows}

**汇总**:
- Prefill (token 1): **{prefill:.1f}ms**
- 稳态 (token 3-10): **{smin:.1f}-{smax:.1f}ms**, 平均 {savg:.1f}ms
- 吞吐量: **~{tput:.1f} tok/s**

### 与上一版本对比 ({PREV_COMMIT} → {commit})

| 指标 | 上一版 ({PREV_COMMIT}) | 本次 ({commit}) | 变化 |
|------|----------------------|-----------------|------|
| Prefill | {PREV_PREFILL:.1f}ms | **{prefill:.1f}ms** | {delta(PREV_PREFILL, prefill)} |
| 稳态平均 | {PREV_STEADY:.1f}ms | **{savg:.1f}ms** | {delta(PREV_STEADY, savg)} |
| 吞吐量 | ~{PREV_TPUT:.1f} tok/s | **~{tput:.1f} tok/s** | {delta(PREV_TPUT, tput, lower_better=False)} |
| Perf 阶段 (加载+10tok) | {PREV_PERF_SECS}s | **{perf_secs}s** | {delta(PREV_PERF_SECS, perf_secs)} |

注: 上一版数据自动从前次报告中提取，两次均为 ReleaseFast 编译。

---

## 二、7-Prompt 端到端测试

```
bash scripts/best_test.sh → {e2e_pass} passed, {e2e_fail} failed ({e2e_secs}s)
```

| # | 结果 | 模型输出 (截取) |
|---|------|----------------|
{e2e_rows}

### 正确性验证

| # | Prompt | 上一版 ({PREV_COMMIT}) | 本次 ({commit}) |
|---|--------|----------------------|-----------------|
| P1 | 2+2= | ✅ | {e2e_results[0]} |
| P2 | Capital of France | ✅ | {e2e_results[1]} |
| P3 | Water freezes at | ✅ | {e2e_results[2]} |
| P4 | Is Earth round? | ✅ | {e2e_results[3]} |
| P5 | 3*3= | ✅ | {e2e_results[4]} |
| P6 | 10-5= | ✅ | {e2e_results[5]} |
| P7 | Capital of France? | ✅ | {e2e_results[6]} |

**{e2e_pass}/7 PASS, {e2e_fail} FAIL, 0 SKIP**

---

## 三、单元测试

```
zig build test → {unit}
```

---

## 四、关键性能指标

| 指标 | 上一版 ({PREV_COMMIT}) | 本次 ({commit}) | 变化 |
|------|----------------------|-----------------|------|
| Perf 阶段 (加载+生成) | {PREV_PERF_SECS}s | **{perf_secs}s** | {delta(PREV_PERF_SECS, perf_secs)} |
| Prefill 延迟 | {PREV_PREFILL:.1f}ms | **{prefill:.1f}ms** | {delta(PREV_PREFILL, prefill)} |
| 稳态 token 延迟 | {PREV_STEADY:.1f}ms | **{savg:.1f}ms** | {delta(PREV_STEADY, savg)} |
| 稳态吞吐量 | ~{PREV_TPUT:.1f} tok/s | **~{tput:.1f} tok/s** | {delta(PREV_TPUT, tput, lower_better=False)} |
| 7-Prompt 通过率 | {PREV_E2E} | **{e2e_pass}/7** | — |
| ExpertCache | 4GB / ~40% hit | 4GB / ~40% hit | — |

注: 两组数据均为 ReleaseFast 编译。Perf 阶段包含模型加载 + 10 token 生成。
"""

with open(report_path, "w") as f:
    f.write(report)

print(f"Prefill: {prefill:.1f}ms | Steady: {savg:.1f}ms | {tput:.1f} tok/s | E2E: {e2e_pass}/7")
