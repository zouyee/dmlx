# 剩余问题修复计划
# 日期: 2026-05-04

## 修复项

| # | 问题 | 优先级 | 方案 | 风险 | 状态 |
|---|------|--------|------|------|------|
| 1 | prompt_cache 类型安全 | P0 | 方案 B: vtable 指针比较断言 | 低 | 待修复 |
| 2 | Gate float32 | P1 | logits.astype(f32) 在 scoring 前 | 低 | 待修复 |
| 3 | Attention mask return_array | P1 | 预填充时构建显式 mask 含 sliding window | 中 | 待修复 |
| 4 | dataSliceMut CoW | P1 | 添加 debug assert ref_count==1 | 低 | 待修复 |

不修复:
- Server batch forward (P1): 架构级重构，不在本轮范围
