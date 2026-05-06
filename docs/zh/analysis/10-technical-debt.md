# 第十章 技术债务评估与修复建议

## 10.1 债务热力图

```
高影响 ↑
  P0 │  [Prompt Cache 类型漏洞]
     │  [NN 层 GPU 化（剩余34处）]
  P1 │  [dataSliceMut 安全隐患]
     │  [Sampling 排序性能]
     │  [AdamW 临时对象风暴]
     │  [Batched Forward 未集成]
  P2 │  [EagerContext stream 泄漏]
     │  [allocator 参数误导]
     │  [ops.zig API 冗余]
     └─────────────────────────────→ 高频率
```

## 10.2 修复优先级

### Immediate（1-2 周内，高 ROI）

| 优先级 | 问题 | 工作量 | 影响 |
|--------|------|--------|------|
| 1 | `prompt_cache.zig` 类型安全漏洞 | 1-2 天 | **生产崩溃**（默认配置下必现） |
| 2 | `sampling.zig` `insertion` → `pdq`/`mlx_topk` | 半天 | 显著降低 128K vocab 延迟 |
| 3 | `nn.zig` BatchNorm `var_buf` 未初始化 | 1 小时 | 数值错误（审计已指出） |

### Short-term（1-2 月内）

| 优先级 | 问题 | 工作量 | 收益 |
|--------|------|--------|------|
| 4 | `nn.zig` GPU 化迁移 | 2-3 周 | 消除最大技术债务，启用 GPU 加速 |
| 5 | `AdamW.step` 融合优化 | 3-5 天 | 训练速度提升 5-10x |
| 6 | `EagerContext` 添加 `deinit` | 半天 | 修复 stream 泄漏 |
| 7 | `batch_builder` 集成 | 1-2 周 | 释放连续批处理吞吐潜力 |

### Medium-term（3-6 月内）

| 优先级 | 问题 | 工作量 |
|--------|------|--------|
| 8 | 错误类型细分 | 2-3 天 |
| 9 | 测试覆盖补全（nn_tests, grad_tests, golden） | 1-2 周 |
| 10 | API 清理（移除冗余和未使用参数） | 3-5 天 |

## 10.3 具体修复方案

### Prompt Cache 类型安全（P0）

**方案 A（推荐）**：扩展 VTable
```zig
pub const VTable = struct {
    // ... 现有方法 ...
    saveState: ?*const fn (ctx: *anyopaque, path: []const u8) anyerror!void,
    loadState: ?*const fn (ctx: *anyopaque, path: []const u8) anyerror!void,
};
```

**方案 B（快速修复）**：添加运行时断言
```zig
const std_cache: *StandardKVCache = @ptrCast(@alignCast(cache.ptr));
std.debug.assert(cache.vtable == &StandardKVCache.vtable);  // 安全检查
```

### Sampling 排序优化（P1）

**方案 A（推荐）**：GPU top-k
```zig
// 替换 insertion sort
const topk_result = try ops.topk(ctx, logits, @intCast(top_k));
```

**方案 B（CPU 优化）**：`std.sort.pdq`
```zig
std.sort.pdq(ScoredToken, scored[0..vocab_size], {}, scoredGreater);
```

### NN 层 GPU 化（P0→P1）

优先级排序：
1. `Linear.init`：权重初始化改用 `mlx_random_normal`（当前用 CPU 随机数）
2. `Embedding.forward`：改用 `mlx_take`（当前用 CPU lookup）
3. `BatchNorm.forward`：改用 `fast.layerNorm` + mean/variance 算子
4. `MultiHeadAttention`：改用 `fast.scaledDotProductAttention`
5. `LSTM`/`GRU`/`RNN`：改为 mlx-c 算子链（matmul + sigmoid + tanh）

## 10.4 架构建议

1. **KV Cache VTable 扩展**：增加 `saveState`/`loadState`/`clone`，消除 `prompt_cache.zig` 的类型不安全假设

2. **NN 层统一抽象**：
   - 所有 NN 层通过 mlx-c 算子链实现
   - `dataSliceMut` 标记为 `deprecated`，仅用于测试/调试
   - 新增 `nn/gpu/` 子模块存放 GPU 化实现

3. **Stream 生命周期统一**：
   - `EagerContext` 支持 `deinit` 释放 stream
   - 或改为全局默认 stream 引用（推荐）

4. **采样后端切换**：
   - vocab > 32K：默认 `mlx_topk` GPU
   - vocab <= 32K：回退到 CPU `pdq`

5. **构建时 feature flags**：
   - `-Dmetal` / `-Daccelerate` / `-Ddistributed`

## 10.5 与生产路线图的对比

`ROADMAP.md` 声称所有 Phase 0–7 + Task 13–34 已完成。

| 维度 | 声称 | 实际 | 偏差 |
|------|------|------|------|
| 功能完成度 | 全部完成 | 基本属实 | 投机解码、引导解码、MoE、分层缓存均已落地 |
| 质量完成度 | 全部完成 | **部分 overclaim** | NN GPU 化、@constCast 清除未完全完成 |
| 测试完成度 | 350 测试通过 | 测试数量属实 | 结构性缺口：NN 层、Autograd、真实权重 |

**建议**：在路线图中将 `nn.zig` 的 GPU 化迁移从"已完成"调整为"部分完成（activations已完成，nn层待清理）"。
