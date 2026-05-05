# 故障排除：调度器与 Block 管理器

**组件**：Scheduler、BlockManager、请求队列
**规格**：`.kiro/specs/production-deployment/design.md` §3.1、§3.2
**属性**：属性 7（优先级排序）、属性 8（Block 保护）

## 症状：请求无限期停留在等待队列中

**根本原因**：BlockManager.freeCount() 报告的空闲 block 数低于实际值。

### 规格参考

- `design.md` §3.2 — Block 管理器接口
- 属性 8：Block 保护 — "空闲 block 数量与已使用 block 数量之和应始终等于 block 总数"

### 代码检查点

1. `src/scheduler.zig:46` — `canAllocate()` 逻辑
2. `src/scheduler.zig:284` — 等待队列提升
3. `src/kvcache/paged.zig:89` — `freeCount()` 计算

### 修复历史

- **2026-04-20**：将 BlockManager 改为独立跟踪 ref_count，与 used/allocated 状态分离。共享 block 现在仅递减 freeCount 一次，而非每个请求递减一次。

## 症状：解码请求在长时间 prefill 运行时被饿死

**根本原因**：分块 prefill 未正确分割；整个提示词在一步中处理。

### 规格参考

- 属性 12：分块 Prefill 正确性
- `design.md` §3.1 — 调度器 `max_prefill_tokens`

### 代码检查点

1. `src/scheduler.zig:183` — `currentPrefillChunkLen()`
2. `src/scheduler.zig:275` — `schedule()` 中的 prefill 与解码分类
3. `src/scheduler.zig:335` — `postprocess()` 中的 `prefill_offset` 推进

### 诊断步骤

1. 检查 `max_prefill_tokens` 配置 — 是否设置为合理值（例如 512）？
2. 验证 `Request.hasPendingPrefill()` 在完整提示词消费完毕前返回 true
3. 检查 `schedule()` 是否将 prefill 请求放入 `prefill_list` 而非 `decode_list`

## 症状：负载测试中出现 "InsufficientBlocks" 错误

**根本原因**：Block 池耗尽；已完成请求没有逐出策略。

### 规格参考

- 属性 8：Block 保护
- `design.md` §3.2 — `freeBlocks()`

### 代码检查点

1. `src/scheduler.zig:290` — 等待队列提升 block 检查
2. `src/scheduler.zig:370` — `postprocess()` 中的已完成请求清理
3. `src/kvcache/paged.zig:72` — `freeBlocks()` 实现

### 诊断步骤

1. 检查 `postprocess()` 是否为已完成的请求释放 block
2. 验证 `freeCount() + usedCount() == total_blocks` 不变量是否成立
3. 如果不变量被破坏，搜索重复释放或遗漏释放的路径
