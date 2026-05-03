# mlx-zig 性能基准测试报告
# 日期: 2026-05-04
# Commit: 93ed7a6 (tuning branch)
# 模型: DeepSeek-V4-Flash-4bit (~150GB, 33 shards)
# 硬件: MacBook Pro, 48GB RAM
# 模式: smelt + stream, ExpertCache 4GB, temperature=0

---

## 一、Token 生成延迟

### Test 1: 短 prompt ("Hello", 5 tokens input, 20 tokens output)

| Token | 延迟 (ms) | Cache Hits | Cache Misses | 说明 |
|-------|----------|-----------|-------------|------|
| 1 | 715.9 | 0 | 12390 | 冷启动 + prefill |
| 2 | 148.3 | 12 | 1800 | cache 预热中 |
| 3 | 113.0 | 918 | 1122 | cache 命中率上升 |
| 4 | 118.5 | 768 | 1218 | |
| 5 | 120.3 | 936 | 1110 | |
| 6-10 | 114-131 | 576-1068 | 1056-1338 | 稳态 |
| 11-20 | 115-138 | 480-1164 | 1002-1428 | 稳态 |

**汇总**:
- Prefill (token 1): **716ms**
- 稳态 (token 3-20): **113-138ms/token**, 平均 ~125ms
- 吞吐量: ~8 tokens/s
- ExpertCache 命中率: 从 0% 爬升到 ~45%

### Test 2: 数学 prompt ("2+2=", 8 tokens input, 10 tokens output)

| Token | 延迟 (ms) | Cache Hits | Cache Misses |
|-------|----------|-----------|-------------|
| 1 | 688.1 | 0 | 12390 |
| 2 | 155.9 | 12 | 1800 |
| 3 | 126.1 | 918 | 1122 |
| 4-10 | 123-134 | 576-1068 | 1056-1338 |

**汇总**:
- Prefill: **688ms**
- 稳态: **123-134ms/token**, 平均 ~128ms
- 与 Test 1 一致，prompt 长度对稳态延迟无显著影响

---

## 二、7-Prompt 端到端测试

```
bash scripts/best_test.sh
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
| Prefill 延迟 | ~700ms (8 tokens) |
| 稳态 token 延迟 | ~125ms |
| 稳态吞吐量 | ~8 tok/s |
| ExpertCache 大小 | 4GB |
| ExpertCache 稳态命中率 | ~40-45% |
| 模型加载 | smelt stream (按需从 SSD 加载) |
| 内存占用 | <48GB (适配 MacBook Pro) |
