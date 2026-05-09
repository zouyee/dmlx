---
date: 2026-05-09
Commit: 538f930 (tuning)
model: DeepSeek-V4-Flash-4bit (~40GB 4-bit, 33 shards)
hardware: Apple M4 Pro, 48GB
mode: smelt + stream, ExpertCache 4GB, temperature=0
build: zig build -Doptimize=ReleaseFast
generated_by: scripts/run_benchmark.sh
total_time: 1739s (perf 92s + e2e 1616s)
---

# dmlx Performance Benchmark Report

---

## 1. Token Generation Latency

| Token | Latency (ms) | Cache Hits | Cache Misses |
|-------|-------------|-----------|-------------|
| 1 | 247.4 | 0 | 7920 |
| 2 | 101.2 | 22 | 1795 |
| 3 | 50.6 | 840 | 1176 |
| 4 | 44.6 | 816 | 1182 |
| 5 | 43.8 | 882 | 1134 |
| 6 | 67.2 | 708 | 1266 |
| 7 | 79.0 | 846 | 1134 |
| 8 | 89.3 | 558 | 1344 |
| 9 | 91.9 | 708 | 1242 |
| 10 | 85.1 | 750 | 1230 |

**Summary**:
- Prefill (token 1): **247.4ms**
- Steady-state (token 3-10): **43.8-91.9ms**, avg 68.9ms
- Throughput: **~14.5 tok/s**

### Comparison with Previous Version (dff154d → 538f930)

| Metric | Previous (dff154d) | Current (538f930) | Change |
|--------|-------------------|-------------------|--------|
| Prefill | 261.0ms | **247.4ms** | **+5%** |
| Steady-state avg | 95.4ms | **68.9ms** | **+28%** |
| Throughput | ~10.5 tok/s | **~14.5 tok/s** | **+38%** |
| Perf phase (load+10tok) | 102s | **92s** | **+10%** |
| E2E total | 1707s | **1616s** | **+5%** |

Note: Both runs use ReleaseFast with the fixed run_benchmark.sh (rebuild after zig build test).

---

## 2. 7-Prompt End-to-End Test

```
bash scripts/best_test.sh → 7 passed, 0 failed (1616s)
```

| # | Result | Model Output (truncated) |
|---|--------|--------------------------|
| P1 | ✅ | . The user's query is "2+2=?". The assistant's response is "4". The user's query |
| P2 | ✅ | . The capital of France is Paris. The capital of France is Paris. The capital of |
| P3 | ✅ | .</think> The temperature at which water freezes is 0 degrees Celsius. This is a |
| P4 | ✅ | , but the user's question is "Is the Earth round?" The answer is yes. The user's |
| P5 | ✅ | . The user's query is "3*3=". This is a simple multiplication problem. The answe |
| P6 | ✅ | . The user's query is "10-5=". This is a simple arithmetic subtraction problem. |
| P7 | ✅ | to user's query. The user's query is "What is capital of France?" The correct an |

### Correctness Verification

| # | Prompt | Previous (dff154d) | Current (538f930) |
|---|--------|-------------------|-------------------|
| P1 | 2+2= | ✅ | ✅ |
| P2 | Capital of France | ✅ | ✅ |
| P3 | Water freezes at | ✅ | ✅ |
| P4 | Is Earth round? | ✅ | ✅ |
| P5 | 3*3= | ✅ | ✅ |
| P6 | 10-5= | ✅ | ✅ |
| P7 | Capital of France? | ✅ | ✅ |

**7/7 PASS, 0 FAIL, 0 SKIP**

---

## 3. Unit Tests

```
zig build test → PASS (430+)
```

---

## 4. Key Performance Metrics

| Metric | Previous (dff154d) | Current (538f930) | Change |
|--------|-------------------|-------------------|--------|
| Perf phase (load+gen) | 102s | **92s** | **+10%** |
| Prefill latency | 261.0ms | **247.4ms** | **+5%** |
| Steady-state latency | 95.4ms | **68.9ms** | **+28%** |
| Steady-state throughput | ~10.5 tok/s | **~14.5 tok/s** | **+38%** |
| 7-Prompt pass rate | 7/7 | **7/7** | — |
| ExpertCache | 4GB / ~40% hit | 4GB / ~40% hit | — |

Note: Both runs use ReleaseFast. Perf phase includes model loading + 10 token generation.
dff154d data measured on same machine, same day (bench-dff154d branch).
