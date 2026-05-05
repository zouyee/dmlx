# 剩余修复计划
# 日期：2026-05-04

## 修复项

| # | 问题 | 优先级 | 方案 | 风险 | 状态 |
|---|-------|----------|----------|------|--------|
| 1 | prompt_cache 类型安全 | P0 | 方案 B：vtable 指针比较断言 | 低 | 待处理 |
| 2 | Gate float32 | P1 | 评分前 logits.astype(f32) | 低 | 待处理 |
| 3 | Attention mask return_array | P1 | 预填充阶段构建带滑动窗口的显式 mask | 中 | 待处理 |
| 4 | dataSliceMut CoW | P1 | 添加 debug assert ref_count==1 | 低 | 待处理 |

不修复：
- Server batch forward (P1)：架构级重构，超出本轮范围

(End of file - total 14 lines)
