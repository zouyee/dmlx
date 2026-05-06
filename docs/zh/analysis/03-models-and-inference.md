# 第三章 模型与推理引擎

## 3.1 DeepSeek V4（`models/deepseek_v4.zig`，3,091 行）

项目最大、最复杂的模型文件，实现以下前沿机制：

### MLA（Multi-head Latent Attention）
- 通过低秩压缩将 KV Cache 从 `2×n_heads×head_dim` 降至 `2×latent_dim`
- 显著降低长序列的内存占用

### MoE（Mixture of Experts）
- 256 路由专家 + 共享专家
- 通过 `moe_router.zig`（629 行）的 top-k 路由调度
- `expert_stream.zig`（649 行）支持将内存从 ~138GB 降至 ~10GB

### YARN RoPE
- 频率插值支持 1M+ 上下文
- 预计算旋转频率表，GPU 加速应用

### mHC（multi-Hyper Connection）
- `HyperHead` 实现 RMSNorm 加权的可学习混合头

### FP8 KV 存储
- 非 RoPE 维度使用 `mlx_to_fp8`/`mlx_from_fp8` 压缩

### KV 压缩策略
- `compressKV` 支持：mean pooling、softmax-gated pooling、attention sink

## 3.2 DeepSeek V4 加载器（`models/deepseek_v4_loader.zig`，2,071 行）

- 解析 `model.safetensors.index.json` 处理分片权重
- HF 命名到内部命名映射：`gate_proj` → `w1`/`w3`/`w2`
- 自动反量化：检测 `.scales`/`.biases` 后缀
- `SmeltConfig`：专家加载策略（preload 子集 vs stream 按需流式）
- `sliceFusedExperts`：按掩码选取专家子集

## 3.3 投机解码（`speculative.zig`，1,223 行）

### 双轨制实现

**PLD（Prompt Lookup Decoding）**：`NgramDrafter`
- 在已生成上下文上搜索匹配 n-gram 后缀
- 无需草稿模型，纯查找机制
- 实现简洁，约 100 行核心逻辑

**EAGLE**：`EagleDrafter`
- 使用轻量级 MLP draft head 投影隐藏态到 vocab logits
- 支持 KV cache rollback（验证失败时回滚）
- **已知限制**：第 2 个及以后的 draft token 只是重复第一个 token

### 共享验证逻辑

`verifyDraft` 函数实现 speculative sampling 的 accept/reject 算法，确保统计等价性。

## 3.4 引导解码（`guided.zig`，1,129 行）

基于有限状态机（FSM）的约束生成：

- `FiniteStateMachine.fromJsonSchema`：支持 string/integer/boolean/enum
- `FiniteStateMachine.fromRegex`：从正则表达式构建 FSM
- `GuidedDecoder.maskLogits`：使用 MLX `where` 算子将非法 token 的 logits 设为 -inf
- **依赖**：`zig_regex` 包

## 3.5 生成引擎三层 API（`generation.zig`）

| API | 用途 | 特点 |
|-----|------|------|
| `generateStep` | 单次前向+采样 | 使用 `ScopedArrayArena` 追踪临时数组 |
| `streamGenerate` | 逐 token 流式生成 | SSE 事件输出 |
| `generate` | 批量生成 | 返回完整 token 序列 |
| `streamGenerateSpeculative` | PLD 投机解码 | KV cache rollback 支持 |
| `streamGenerateEagle` | EAGLE 投机解码 | 需 `forwardWithHidden` |

## 3.6 模型注册表（`model_registry.zig`）

支持 9 种架构：LLaMA、Mistral、Qwen2、Qwen3、Gemma、GLM-4、Phi、Phi-3、DeepSeek V4

运行时通过 VTable 多态切换，无需重新编译。
