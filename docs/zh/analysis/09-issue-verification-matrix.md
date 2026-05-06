# 第九章 问题验证矩阵

## 9.1 与项目自我审计的交叉验证

`docs/deep-analysis.md` 是 v0.3.0 时的自我审计文档，`ROADMAP.md` 声称所有问题已修复。本分析逐条验证：

| 原问题 | 原严重度 | 项目声称 | 实际状态 | 偏差 |
|--------|---------|---------|---------|------|
| 系统性内存泄漏 | P0 | ✅ 修复 | **部分修复** | `ScopedArrayArena` 已引入，但 `nn.zig` CPU 路径不经过 Arena |
| 错误信息丢失 | P0 | ✅ 修复 | **已修复** | `mlxErrorHandler` 正确捕获 C++ 异常文本 |
| NN/Activation 绕过 GPU | P0 | ✅ 修复 | **部分修复** | `activations.zig` 全部 GPU 化；`nn.zig` 仍有 34 处 `dataSliceMut` |
| Sampling insertion sort | P2 | 未提及 | **未修复** | 4 处调用仍在 |
| `dataSliceMut` @constCast | P1 | ✅ 修复 | **未修复** | 全库 10 处 `@constCast` + `nn.zig` 34 处 `dataSliceMut` |
| 硬编码 Homebrew | P1 | ✅ 修复 | **已修复** | 四级探测已实现 |
| EagerContext stream 泄漏 | P1 | 未提及 | **未修复** | 仍无 `deinit` |
| Attention mask 忽略 | P1 | 未提及 | **待验证** | `nn.zig` TransformerEncoderLayer 需确认 |
| allocator 参数误导 | P2 | 未提及 | **未修复** | `array.zig` 3 处 `_ = allocator` |
| ops.zig 与 ops/ 重复 | P2 | 未提及 | **未修复** | 两套 API 并存 |
| zig-regex 指向 main | P1 | ✅ 修复 | **已修复** | 已指向固定 hash |
| NN 层无测试 | P1 | ✅ 修复 | **部分修复** | 无 `nn_tests`，`numerical_equivalence_test` 覆盖部分 |
| Autograd 无测试 | P1 | 未提及 | **未修复** | 无 `grad_tests` |
| 缺少 golden test | P1 | ✅ 修复 | **部分修复** | 有 golden 文件但使用随机权重 |

## 9.2 修复完成度统计

```
P0 问题（3个）
├── 系统性内存泄漏     部分修复  ⚠️
├── 错误信息丢失       已修复   ✅
└── NN/Activation GPU  部分修复  ⚠️

P1 问题（6个）
├── dataSliceMut       未修复   ❌
├── 硬编码 Homebrew    已修复   ✅
├── EagerContext 泄漏  未修复   ❌
├── Attention mask     待验证   ❓
├── NN 测试           部分修复  ⚠️
└── golden test        部分修复  ⚠️

P2 问题（4个）
├── insertion sort     未修复   ❌
├── allocator 误导     未修复   ❌
├── ops 重复          未修复   ❌
└── scalar 忽略 dtype  待验证   ❓
```

## 9.3 新增问题（本分析发现）

| 新问题 | 严重度 | 位置 | 说明 |
|--------|--------|------|------|
| Prompt Cache 类型安全漏洞 | **P0** | `prompt_cache.zig:74` | `@ptrCast` 假设所有缓存为 StandardKVCache |
| DistributedGroup deinit 为空 | P1 | `distributed.zig:83` | 资源泄漏 |
| ModelPool vtable null 风险 | P2 | `model_pool.zig:66` | 模型资源可能不释放 |
| EagleDrafter 简化实现 | P2 | `speculative.zig:146` | 仅单 token draft 有效 |
| `strides()` 64 位假设 | P2 | `array.zig` | 32 位平台截断风险 |

## 9.4 偏差分析

**最大偏差**：项目声称 NN 层 GPU 化和 `@constCast` 清除"全部完成"，但实际 `nn.zig` 仍有 34 处 `dataSliceMut`（间接通过 `@constCast`），`minimax.zig` 和 `deepseek_v4.zig` 中仍有直接 `@constCast`。

可能原因：
1. `activations.zig` 的 GPU 化被误认为"全部 NN 层"已修复
2. `dataSliceMut` 的调用统计在修复时被遗漏
3. 新模型文件（`minimax.zig`、`deepseek_v4.zig`）引入了新的 `@constCast`
