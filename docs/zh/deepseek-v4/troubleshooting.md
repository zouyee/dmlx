# DeepSeek V4 故障排查指南

## 概述

本文档提供 mlx-zig 中 DeepSeek V4 模型推理问题的故障排查指导。

## 常见问题

### 1. 输出乱码 / 无效 Token

**症状：**
- 模型生成无意义文本
- 输出包含随机字符或符号
- Token 似乎超出词汇表范围

**根本原因：**
聊天模板格式不正确，导致分词器将特殊 token 拆分为子 token。

**诊断方法：**
检查推理运行时的日志输出：

```bash
mlx-zig chat --model ~/models/deepseek-v4 --prompt "Hello"
```

查看 prompt token 验证信息：
```
✅ Prompt correctly formatted with BOS token 100000
Prompt tokens (8): [100000, 100003, 1234, 5678, 100006]
```

如果看到如下错误：
```
❌ BOS token mismatch! Expected 100000, got 60
```

这表明聊天模板使用了不正确的特殊 token。

**解决方案：**
确保聊天模板使用正确的特殊 token 格式：

- ✅ 正确：`<|begin_of_sentence|>`（半角竖线 `|`，下划线 `_`）
- ❌ 错误：`<｜begin▁of▁sentence｜>`（全角竖线 `｜`，特殊空格 `▁`）

**已在提交中修复：** `fix: correct DeepSeek V4 chat template special tokens`

---

### 2. 特殊 Token 参考

DeepSeek V4 使用以下特殊 token：

| Token | ID | 用途 |
|-------|-----|-------|
| `<|begin_of_sentence|>` | 100000 | 对话开始 |
| `<|end_of_sentence|>` | 100001 | 助手回复结束 |
| `<|User|>` | 100003 | 用户消息标记 |
| `<|Assistant|>` | 100006 | 助手消息标记 |

**正确的 Prompt 格式：**
```
<|begin_of_sentence|>{system}\n\n<|User|>: {user_message}\n\n<|Assistant|>: 
```

**示例：**
```
<|begin_of_sentence|>You are a helpful assistant.\n\n<|User|>: Hello, how are you?\n\n<|Assistant|>: 
```

---

### 3. Token 验证

实现中包含对 prompt 格式的自动验证：

```zig
// Validates that the first token is BOS (100000)
if (prompt_tokens[0] != 100000) {
    std.log.err("❌ BOS token mismatch! Expected 100000, got {d}", .{prompt_tokens[0]});
    return error.InvalidPromptFormat;
}
```

这可以在推理开始前及早发现聊天模板格式错误。

---

### 4. 内存问题

**症状：**
- 内存不足错误
- 推理极慢（每 token > 1 秒）
- 系统 swap 使用

**诊断方法：**
检查日志中的 MLX 内存配置：
```
MLX memory: wired_limit=40960MB cache_limit=38400MB (system=48000MB)
```

**解决方案：**

1. **启用 Smelt 模式**（部分专家加载）：
```bash
mlx-zig chat --model ~/models/deepseek-v4 \
  --smelt --smelt-experts 0.15 \
  --prompt "Hello"
```

2. **使用量化 KV Cache**：
```bash
mlx-zig serve --model ~/models/deepseek-v4 \
  --kv-strategy paged_quantized \
  --kv-bits 4
```

3. **减少上下文长度**：
```bash
mlx-zig chat --model ~/models/deepseek-v4 \
  --max-kv-size 4096 \
  --prompt "Hello"
```

---

### 5. 推理速度慢

**预期性能（M4 Max，48GB，4-bit 量化）：**
- TTFT（首 token 延迟）：32-token prompt 下 200-500ms
- ITL（token 间延迟）：每 token 250-500ms
- 吞吐量：2-4 tokens/s

**如果比预期慢：**

1. **检查权重是否已量化：**
```bash
# 模型路径中应出现 "4-bit" 或 "quantized"
ls -lh ~/models/deepseek-v4-flash-4bit/
```

2. **验证 GPU 使用：**
```bash
# MLX 应默认使用 GPU
# 检查日志中是否有："Set default device to GPU"
```

3. **启用推测解码**（后续优化）：
```bash
mlx-zig chat --model ~/models/deepseek-v4 \
  --speculative-ngram 4 \
  --prompt "Hello"
```

---

### 6. 模型加载错误

**症状：**
- "Missing weight" 错误
- "Unsupported architecture" 错误
- 加载过程中段错误

**常见原因：**

1. **模型格式不正确：**
   - 确保模型是 MLX 格式（非 PyTorch 或 GGUF）
   - 使用 `mlx_lm.convert` 转换 HuggingFace 模型

2. **下载不完整：**
   - 验证所有 shard 文件均存在
   - 检查文件大小是否匹配预期值

3. **配置不匹配：**
   - 确保 `config.json` 匹配模型架构
   - 检查 `model_type` 字段是否为 `"deepseek_v4"`

**诊断方法：**
```bash
# 检查模型文件
ls -lh ~/models/deepseek-v4-flash-4bit/

# 应看到：
# - config.json
# - tokenizer.json
# - model.safetensors（或 model-00001-of-NNNNN.safetensors）
```

---

## 调试技巧

### 启用详细日志

将日志级别设为 debug：
```bash
export RUST_LOG=debug  # 如果使用 Rust 组件
mlx-zig chat --model ~/models/deepseek-v4 --prompt "Test"
```

### 检查 Logits

实现中包含 logits 的诊断日志：
```
Logits: len=129280 max=12.3456 min=-8.9012 mean=0.0234 argmax=1234 nan=0 inf=0
Top tokens: [1234]=12.35 [5678]=11.23 [9012]=10.45
```

检查以下内容：
- NaN 或 Inf 值（表明数值不稳定）
- 极大/极小值（表明缩放问题）
- 均匀分布（表明模型未学习）

### 使用简单 Prompt 测试

从最简 prompt 开始，隔离问题：
```bash
# 测试 1：单个 token
mlx-zig chat --model ~/models/deepseek-v4 --prompt "Hi" --max-tokens 5

# 测试 2：仅英文
mlx-zig chat --model ~/models/deepseek-v4 --prompt "Hello" --max-tokens 10

# 测试 3：中文（如果模型支持）
mlx-zig chat --model ~/models/deepseek-v4 --prompt "你好" --max-tokens 10
```

---

## 性能基准测试

运行基准测试工具来测量性能：

```bash
mlx-zig benchmark --model ~/models/deepseek-v4-flash-4bit \
  --input-tokens 32 \
  --output-tokens 128 \
  --num-runs 3
```

预期输出：
```
=== Benchmark Results ===
  TTFT (time to first token): 350.00 ms
  ITL  (inter-token latency): 300.00 ms
  Throughput:                  3.33 tokens/s
  Peak memory:                 42000.0 MB
```

---

## 已知限制

1. **仅支持单请求吞吐**
   - 尚未支持 continuous batching
   - 并发请求被串行化

2. **CPU 采样瓶颈**
   - 采样在 CPU 上进行，非 GPU
   - 导致 GPU 流水线停顿

3. **无图缓存**
   - 每个 token 触发完整图编译
   - 后续优化：缓存解码图

4. **MoE 路由开销**
   - batch=1 时 GPU 上 top-k 选择效率低
   - 后续优化：融合 MoE kernel

---

## 报告问题

报告问题时，请包含以下信息：

1. **系统信息：**
   - macOS 版本
   - Apple Silicon 芯片（M1/M2/M3/M4）
   - 总内存

2. **模型信息：**
   - 模型名称和变体
   - 量化级别（4-bit、8-bit、FP16）
   - 磁盘上的模型大小

3. **使用的命令：**
   ```bash
   mlx-zig chat --model <path> --prompt "<prompt>" --max-tokens <n>
   ```

4. **日志输出：**
   - 包含命令的完整日志输出
   - 特别是 prompt token 验证和 logits 诊断信息

5. **预期 vs 实际行为：**
   - 您预期发生什么
   - 实际发生什么
   - 任何错误消息

---

## 参考资料

- [DeepSeek V4 Paper](https://arxiv.org/abs/2501.12948)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [mlx-lm Repository](https://github.com/ml-explore/mlx-examples/tree/main/llms)
