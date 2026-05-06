# 第六章 量化、训练与生态

## 6.1 量化体系（`quantize.zig`，872 行）

支持 4 种前沿格式 + 2 种内部格式：

| 格式 | group_size | 说明 |
|------|-----------|------|
| affine | 32/64/128 | 标准对称/非对称量化 |
| mxfp4 | 32 | Microscaling FP4（AMD 标准） |
| nvfp4 | 16 | NVIDIA FP4（黑石架构） |
| mxfp8 | 32 | Microscaling FP8 |
| fp8_e4m3 | - | 原生 FP8 E4M3 |
| turboquant | - | Lloyd-Max + QJL 自适应量化 |

核心类型：
- `QuantizedWeight`：打包数据 + scales + biases + config + original_shape
- `quantizedMatmul`：融合反量化 + 矩阵乘
- `gatherQmm`：量化 gather matmul，用于 MoE batched/indexed 推理

## 6.2 专家流式加载（`expert_stream.zig`，649 行）

**核心能力**：将 DeepSeek V4 内存从 ~138GB 降至 ~10GB（仅加载活跃专家）

| 模式 | 策略 | 适用场景 |
|------|------|---------|
| Preload | 加载指定比例的 experts 到内存 | 内存充足 |
| Stream | 按需从磁盘流式加载 | 内存受限 |

Stream 模式特性：
- LRU cache 管理活跃专家
- `PartialTensorReader`：基于 `pread` 的部分张量读取，避免全文件加载
- 层预取器（layer prefetcher）：预加载下一层需要的专家

## 6.3 训练

### AdamW（`optim.zig`，217 行）

```zig
pub fn step(self: *AdamW, grads: []const Array, stream: mlx_stream) !void {
    // 每参数每步创建 ~15 个临时 mlx_array
    const sc_lr = c.c.mlx_array_new_float32(self.lr);
    const sc_eps = c.c.mlx_array_new_float32(self.eps);
    // ... 约 15 个标量 Array
}
```

**量化影响**：
- 7B 模型约 200 个参数矩阵 → 每步 ~3000 个临时对象
- 逐个参数串行执行，无法利用 GPU 批量并行

**已知优化点**（代码注释已标明）：
```zig
// FUSION INTEGRATION POINT (R8.2):
// 可替换为 compiledAdamWStep（fused.zig），将整个 step 编译为单 kernel
```

### Trainer（`trainer.zig`，640 行）

- SFT（监督微调）训练循环
- **缺失**：梯度裁剪（`clip_grad_norm` 为 TODO）

### LoRA（`lora.zig`，227 行）

- `LoRALayer`：A（Gaussian init）+ B（Zero init）+ scale
- `LoRAModel`：多层适配器管理

### QLoRA（`qlora.zig`，773 行）

量化 + LoRA 组合训练：
- 基础权重保持量化状态
- 仅训练 LoRA A/B 矩阵
- 支持 `quantizedMatmul` + `lora.apply` 融合前向

## 6.4 safetensors 随机访问读取器（`io/safetensors_reader.zig`，1,045 行）

**亮点**：基于 `pread` 的零拷贝随机访问，无需全文件加载

核心组件：
- `TensorInfo`：dtype、shape、data_offsets、shard_path
- `TensorIndex`：跨分片的哈希索引
- `addShard`：解析单文件 header（8-byte LE u64 + JSON）
- `loadTensor`：`pread` 按偏移读取原始数据
- `buildIndexFromDirectory`：从 `model.safetensors.index.json` 构建全局索引
