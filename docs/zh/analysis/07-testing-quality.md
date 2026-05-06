# 第七章 测试体系与质量

## 7.1 测试模块全景

`tests.zig` 注册了 50+ 测试模块，按类别分组：

### 算子测试（核心正确性）
- `core_tests`：基础 Array 操作
- `comparison_tests`、`math_tests`、`shape_tests`、`reduce_tests`、`sort_tests`
- `creation_tests`、`random_tests`、`linalg_tests`、`fft_tests`

### 模型与推理测试（功能验证）
- `e2e_tests`（302 行）：tiny random model forward + generate + GQA
- `deepseek_v4_tests`（611 行）：`compressKV` 各种模式、slice 操作
- `generation_tests`、`speculative_tests`、`guided_tests`

### 数值等价性测试（精度验证）
- `numerical_equivalence_test.zig`（814 行）：**属性测试**，100 次迭代
  - RMSNorm：cosine similarity ≥ 0.9999
  - RoPE、SDPA、Embedding、LSTM、GRU、多种 loss 函数
  - 与 Python MLX 参考输出比较

### MoE 与专家测试
- `expert_remap_tests`、`expert_cache_tests`、`expert_stream_tests`
- `moe_router_tests`：top-k 路由正确性

### 集成测试
- `cache_integration_tests`、`integration_tests`
- `model_smoke_tests`、golden tests

### 基础设施测试
- `kvcache_tests`、`tiered_kvcache_tests`、`prefix_disk_tests`
- `memory_tests`、`memory_property_tests`
- `arena_tests`：ScopedArrayArena 功能验证

## 7.2 测试质量评估

### 优势

- **属性测试**（100 次随机输入迭代）比单点测试更可靠
- **数值等价性测试**使用 cosine similarity 阈值：
  - float32: 0.9999
  - int8: 0.99
  - int4: 0.95
- E2E 测试包含 tiny model forward + generate + KV cache 组合验证
- DeepSeek V4 有专门的单元测试

### 缺口

| 缺口 | 严重度 | 说明 |
|------|--------|------|
| 无 `nn_tests` | P1 | Linear/BatchNorm/LSTM/GRU/RNN/MultiHeadAttention 无直接测试 |
| 无 `grad_tests` | P1 | 自动微分正确性未直接验证 |
| 无真实权重 golden test | P1 | 所有模型测试使用随机权重 |
| `trainer_tests` 可能为骨架 | P2 | 需验证是否含实际训练循环 |

## 7.3 E2E 测试实证分析

`e2e_tests.zig` 使用微型模型（128 vocab / 32 hidden / 2 layer / 4 heads）：

```zig
const config = LlamaConfig{
    .vocab_size = 128,
    .hidden_size = 32,
    .num_hidden_layers = 2,
    .num_attention_heads = 4,
    // ...
};
```

验证内容：
- forward 输出 shape `[batch, seq_len, vocab]`
- generate 生成长度正确
- 有/无 KV cache 的生成结果一致

**局限性**：微型模型无法暴露大模型的数值问题（FP16 溢出、量化误差累积）。

## 7.4 性能基准（`benchmark_run.log`）

```
⚡ DeepSeek V4 Performance Regression Test
Model: DeepSeek-V4-Flash-4bit
Absolute thresholds: TTFT ≤ 500ms | ITL ≤ 150ms | TPS ≥ 5
Regression threshold: +20% vs baseline
```

当前性能回归框架已建立，但日志未显示完整运行结果。
