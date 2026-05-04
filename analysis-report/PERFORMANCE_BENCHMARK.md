# mlx-zig 性能基准测试报告
# 日期: 2026-05-04
# Commit: 13a6e6b (tuning branch)
# 模型: DeepSeek-V4-Flash-4bit (~150GB, 33 shards)
# 硬件: MacBook Pro, 48GB RAM
# 模式: smelt + stream, ExpertCache 4GB, temperature=0

---

## 一、Token 生成延迟

### Test 1: 短 prompt ("Hello", 5 tokens input, 10 tokens output)

| Token | 延迟 (ms) | Cache Hits | Cache Misses | 说明 |
|-------|----------|-----------|-------------|------|
| 1 | 384.6 | 0 | 7920 | 冷启动 + prefill |
| 2 | 143.1 | 22 | 1795 | cache 预热中 |
| 3 | 109.8 | 840 | 1176 | cache 命中率上升 |
| 4 | 111.1 | 816 | 1182 | |
| 5 | 97.3 | 882 | 1134 | |
| 6 | 109.3 | 708 | 1266 | |
| 7 | 105.7 | 846 | 1134 | |
| 8 | 112.8 | 558 | 1344 | |
| 9 | 118.4 | 708 | 1242 | |
| 10 | 106.2 | 750 | 1230 | |

**汇总**:
- Prefill (token 1): **385ms**
- 稳态 (token 3-10): **97-118ms/token**, 平均 ~109ms
- 吞吐量: ~9.2 tokens/s
- ExpertCache 命中率: 从 0% 爬升到 ~40%

### 与修复前对比

| 指标 | 修复前 (commit 58d7752) | 修复后 (commit 13a6e6b) | 变化 |
|------|------------------------|------------------------|------|
| Prefill | 716ms | 385ms | **-46%** |
| 稳态平均 | ~125ms | ~109ms | **-13%** |
| 吞吐量 | ~8 tok/s | ~9.2 tok/s | **+15%** |

提升原因: 显式 causal mask 让 MLX SDPA 走更优 kernel 路径（无需内部构建 mask）。

---

## 二、7-Prompt 端到端测试

```
bash scripts/best_test.sh → 7 passed, 0 failed
```

| # | Prompt | 结果 |
|---|--------|------|
| P1 | 2+2= → 4 | ✅ |
| P2 | The capital of France is → Paris | ✅ |
| P3 | Water freezes at (Celsius) → 0 | ✅ |
| P4 | Is the Earth round? → yes | ✅ |
| P5 | 3*3= → 9 | ✅ |
| P6 | 10-5= → 5 | ✅ |
| P7 | What is capital of France? → Paris | ✅ |

**7/7 PASS, 0 FAIL, 0 SKIP**

---

## 三、单元测试

```
zig build test → exit code 0
```

430+ 测试全部通过。

---

## 四、关键性能指标

| 指标 | 值 |
|------|-----|
| Prefill 延迟 | ~385ms (8 tokens) |
| 稳态 token 延迟 | ~109ms |
| 稳态吞吐量 | ~9.2 tok/s |
| ExpertCache 大小 | 4GB |
| ExpertCache 稳态命中率 | ~40% |
| 模型加载 | smelt stream (按需从 SSD 加载) |
| 内存占用 | <48GB (适配 MacBook Pro) |
