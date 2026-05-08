# dmlx 性能基准测试报告
# 日期: 2026-05-09
# Commit: b2e7113 (tuning)
# 模型: DeepSeek-V4-Flash-4bit (~150GB, 33 shards)
# 硬件: Apple M4 Pro, 48GB
# 模式: smelt + stream, ExpertCache 4GB, temperature=0
# 生成方式: scripts/run_benchmark.sh (自动化)
# 总耗时: 1582s (perf 85s + e2e 1491s)

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

### 与初始版本对比

| 指标 | 初始 (a024bee) | 最终 (b2e7113) | 变化 |
|------|---------------|-----------------|------|
| Prefill | 716ms | **273.9ms** | **+62%** |
| 稳态平均 | ~125ms | **56.6ms** | **+55%** |
| 吞吐量 | ~8 tok/s | **~17.7 tok/s** | **+121%** |

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

### 修复前后对比

| # | Prompt | 修复前 (a024bee) | 修复后 (b2e7113) |
|---|--------|-----------------|-----------------|
| P1 | 2+2= | ✅ (记忆型) | ✅ |
| P2 | Capital of France | ⚠️ 15 token | ✅ |
| P3 | Water freezes at | ⚠️ 20 token | ✅ |
| P4 | Is Earth round? | ❌ 安全文本 | ✅ |
| P5 | 3*3= | ❌ 错误 token | ✅ |
| P6 | 10-5= | ❌ 乱码 | ✅ |
| P7 | Capital of France? | ⚠️ 15 token | ✅ |

**7/7 PASS, 0 FAIL, 0 SKIP**

---

## 三、单元测试

```
zig build test → PASS (430+)
```

---

## 四、关键性能指标

| 指标 | 初始 (a024bee) | 最终 (b2e7113) | 变化 |
|------|---------------|-----------------|------|
| TTFT (模型加载→首 token) | ~140s | ~85s | — |
| Prefill 延迟 | ~716ms | 273.9ms | **+62%** |
| 稳态 token 延迟 | ~125ms | 56.6ms | **+55%** |
| 稳态吞吐量 | ~8 tok/s | ~17.7 tok/s | **+121%** |
| 7-Prompt 通过率 | 1/7 | 7/7 | — |
| ExpertCache | 4GB / ~40% hit | 4GB / ~40% hit | — |

注: TTFT 瓶颈是模型加载 (33 shard 索引 × 2)，prefill 本身仅 273.9ms。
