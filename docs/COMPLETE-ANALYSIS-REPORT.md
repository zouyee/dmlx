# mlx-zig DeepSeek V4 Flash — 完整分析报告
# 2026-05-03 | 三方对照: mlx-lm / Rapid-MLX / mlx-zig

## 一、7 Prompt 测试状态 (chat / stream)

| # | Prompt | 期望 | 状态 |
|---|--------|------|------|
| 1 | 2+2= | 4 | ✅ |
| 2 | The capital of France is | Paris | ⚠️ 15 token |
| 3 | Water freezes at | 0 | ⚠️ 20 token |
| 4 | Is Earth round? yes/no: | yes | ❌ |
| 5 | 3*3= | 9 | ❌ 安全文本 |
| 6 | 10-5= | 5 | ❌ |
| 7 | What is capital of France? | Paris | ⚠️ 15 token |

## 二、已修复 (10)

1. mxfp4 反量化
2. mhcPreApplyMix float32
3. mhcPost float32
4. ExpertCache 启用
5. max_new_tokens guard
6. Chat Template (官方 Jinja)
7. --raw flag
8. pre-commit hook
9. pre-push hook
10. post_mult = 2.0

## 三、三方对照 — 残存 6 项偏差

对照源: ../mlx-lm / ../Rapid-MLX / mlx-zig

### HC Split Sinkhorn (86次/forward)

| 偏差 | mlx-lm | Rapid-MLX | mlx-zig |
|------|--------|-----------|---------|
| mixes.astype(f32) | ✅ | ✅ | ❌ |
| scale.astype(f32) | ✅ | ✅ | ❌ |
| base.astype(f32) | ✅ | ✅ | ❌ |
| comb softmax precise | ✅ | ✅ | ❌ |

### Gate (43次/forward)

| 偏差 | mlx-lm | Rapid-MLX | mlx-zig |
|------|--------|-----------|---------|
| logits.astype(f32) | ✅ | ✅ | ❌ |
| sqrtsoftplus stable | ✅ | ✅ | ❌ |

### Attention

| 偏差 | mlx-lm | Rapid-MLX | mlx-zig |
|------|--------|-----------|---------|
| softmax precise | ✅ | ✅ | ❌ |

## 四、关键发现

- `<ds_safety>` 是模型文本，非系统过滤器
- mhcPreNormFn 数学等价 (cos sim=1.0)
- Rapid-MLX hc_mult=4 走 Metal kernel，Zig 只用 ops

## 五、基础设施

| 组件 | 位置 |
|------|------|
| pre-commit | .git/hooks/pre-commit |
| pre-push | .git/hooks/pre-push |
| 测试方案 | scripts/best_test.sh |
| Makefile | make check |
| 报告 | docs/COMPLETE-ANALYSIS-REPORT.md |
