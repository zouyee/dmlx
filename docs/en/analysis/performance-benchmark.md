---
日期: 2026-05-09
Commit: cd8d29f (tuning)
模型: DeepSeek-V4-Flash-4bit (~40GB 4-bit, 33 shards)
硬件: Apple M4 Pro, 48GB
模式: smelt + stream, ExpertCache 4GB, temperature=0
编译: zig build -Doptimize=ReleaseFast
生成方式: scripts/run_benchmark.sh (自动化)
总耗时: 1582s (perf 85s + e2e 1491s)
---

# dmlx 性能基准测试报告

---

## 一、Token 生成延迟

| Token | 延迟 (ms) | Cache Hits | Cache Misses |
|-------|----------|-----------|-------------|
| 1 | 273.9 | 0 | 7920 |
| 2 | 104.8 | 22 | 1795 |
| 3 | 54.2 | 840 | 1176 |
| 4 | 46.9 | 816 | 1182 |
| 5 | 43.0 | 882 | 1134 |
| 6 | 47.3 | 708 | 1266 |
| 7 | 40.9 | 846 | 1134 |
| 8 | 68.1 | 558 | 1344 |
| 9 | 68.1 | 708 | 1242 |
| 10 | 84.1 | 750 | 1230 |

**汇总**:
- Prefill (token 1): **273.9ms**
- 稳态 (token 3-10): **40.9-84.1ms**, 平均 56.6ms
- 吞吐量: **~17.7 tok/s**

### 与上一版本对比 (dff154d → cd8d29f)

| 指标 | 上一版 dff154d (ReleaseFast) | 本次 cd8d29f (ReleaseFast) | 变化 |
|------|----------------------------|---------------------------|------|
| Prefill | 278.3ms | **273.9ms** | **-2%** |
| 稳态平均 | 102.3ms | **56.6ms** | **-45%** |
| 吞吐量 | ~9.8 tok/s | **~17.7 tok/s** | **+81%** |
| Perf 阶段 (加载+10tok) | 104s | **85s** | **-18%** |

注: dff154d 数据来自同机器 ReleaseFast 实测 (bench-baseline 分支, 2026-05-09)。

### 与初始版本对比 (a024bee → cd8d29f)

| 指标 | 初始 (a024bee) | 本次 (cd8d29f) | 总变化 |
|------|---------------|----------------|--------|
| Prefill | 716ms | **273.9ms** | **-62%** |
| 稳态平均 | ~125ms | **56.6ms** | **-55%** |
| 吞吐量 | ~8 tok/s | **~17.7 tok/s** | **+121%** |
| 7-Prompt 通过率 | 1/7 | **7/7** | — |

---

## 二、7-Prompt 端到端测试

```
bash scripts/best_test.sh → 7 passed, 0 failed (1491s)
```

| # | 结果 | 模型输出 (截取) |
|---|------|----------------|
| P1 | ✅ | . The user's query is "2+2=?". The assistant's response is "4". The user's query |
| P2 | ✅ | . The capital of France is Paris. The capital of France is Paris. The capital of |
| P3 | ✅ | .</think> The temperature at which water freezes is 0 degrees Celsius. This is a |
| P4 | ✅ | , but the user's question is "Is the Earth round?" The answer is yes. The user's |
| P5 | ✅ | . The user's query is "3*3=". This is a simple multiplication problem. The answe |
| P6 | ✅ | . The user's query is "10-5=". This is a simple arithmetic subtraction problem. |
| P7 | ✅ | to user's query. The user's query is "What is capital of France?" The correct an |

### 正确性验证 (dff154d → cd8d29f)

| # | Prompt | 上一版 (dff154d) | 本次 (cd8d29f) |
|---|--------|-----------------|-----------------|
| P1 | 2+2= | ✅ | ✅ |
| P2 | Capital of France | ✅ | ✅ |
| P3 | Water freezes at | ✅ | ✅ |
| P4 | Is Earth round? | ✅ | ✅ |
| P5 | 3*3= | ✅ | ✅ |
| P6 | 10-5= | ✅ | ✅ |
| P7 | Capital of France? | ✅ | ✅ |

**7/7 PASS, 0 FAIL, 0 SKIP**

---

## 三、单元测试

```
zig build test → PASS (430+)
```

---

## 四、关键性能指标

| 指标 | 初始 (a024bee) | TTFT优化 (8ced50c) | 5项优化 (dff154d) | + copy removal (cd8d29f) |
|------|---------------|-------------------|-------------------|--------------------------|
| Prefill 延迟 | ~716ms | 446.8ms | 278.3ms | **273.9ms** |
| 稳态 token 延迟 | ~125ms | 106.5ms | 102.3ms | **56.6ms** |
| 稳态吞吐量 | ~8 tok/s | ~9.4 tok/s | ~9.8 tok/s | **~17.7 tok/s** |
| 7-Prompt 通过率 | 1/7 | 7/7 | 7/7 | **7/7** |
| ExpertCache | 4GB / ~40% hit | 4GB / ~40% hit | 4GB / ~40% hit | **4GB / ~40% hit** |

### dff154d → cd8d29f 增量对比 (实测)

| 指标 | dff154d (上一版) | cd8d29f (最新) | 变化 |
|------|-----------------|----------------|------|
| Prefill | 278.3ms | **273.9ms** | **-2%** |
| 稳态延迟 (平均) | 102.3ms | **56.6ms** | **-45%** |
| 吞吐量 | ~9.8 tok/s | **~17.7 tok/s** | **+81%** |
| Perf 阶段总耗时 | 104s | **85s** | **-18%** |

注: 两组数据均为 ReleaseFast 编译，同机器同条件实测 (2026-05-09)。
