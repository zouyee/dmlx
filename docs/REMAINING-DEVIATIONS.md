# dmlx DeepSeek V4 — 未修复偏差清单
# 2026-05-03 | 对照源: mlx-lm / Rapid-MLX

## 状态: 7/7 测试已通过，以下为精度优化项（非阻塞）

| # | 偏差 | 文件:行 | 当前 | Python | 影响评估 |
|---|------|---------|------|--------|---------|
| 1 | `mhcPreSplitMixes` 未转 f32 | `deepseek_v4.zig:L2322` | 无 float32 转换 | `mixes/scale/base.astype(f32)` | sigmoid 精度，86次/forward |
| 2 | `DSV4Gate` logits 未转 f32 | `deepseek_v4.zig:L540` | 无 float32 转换 | `logits.astype(f32)` | gate 路由精度，43次/forward |
| 3 | `sqrtsoftplus` 不稳定 | `deepseek_v4.zig:L568` | `log(1+exp(x))` (x>50溢出) | `nn.softplus` (稳定) | gate 溢出风险，43次/forward |
| 4 | Attention softmax 非 precise | `deepseek_v4.zig:L1518` | `ops.softmax` (precise=false) | `softmax(precise=true)` | attention 精度，每层 |

## 已修复项 (无需处理)

- post_mult = 2.0 ✅
- mhcPost float32 + comb transpose ✅
- mhcPreApplyMix float32 ✅
- sinkhornNormalize softmaxPrecise ✅
- max_new_tokens guard ✅
- ExpertCache 启用 ✅
