# mlx-zig 深度差距分析：对比 mlx/mlx-c/mlx-lm/oMLX/TileKernels/vLLM

> 基于 2026-04-26 代码审计（37K+ 行 Zig、100+ 文件、350 测试），
> 对比 6 个参考项目，识别可借鉴的强化方向。
> **Phase 7 (P0-P3) 已全部完成，包括所有可选任务。** 3 个模型验证通过。

---

## I. mlx-zig 真实状态（诚实评估）

### 已验证可运行
- ✅ TinyLlama-1.1B-Chat-v1.0-4bit：加载 + 量化推理 + token 生成
- ✅ Qwen2.5-0.5B-Instruct：加载 + 非量化推理 + token 生成
- ✅ Qwen3-0.6B-4bit：加载 + 量化推理 + token 生成（含量化 embedding 反量化）
- ✅ 350 单元测试全部通过
- ✅ Metal GPU 作为默认设备
- ✅ 流式 token 输出（每个 token 生成时即输出）
- ✅ EOS 停止条件
- ✅ Chat template 支持（LLaMA/Qwen/Mistral/Qwen3/GLM-4 路径）
- ✅ Gemma/Phi-3/Qwen3/GLM-4 模型架构支持
- ✅ 重复惩罚 / min_p 采样
- ✅ OpenAI API：stop/logprobs/tool_calls/Anthropic 兼容
- ✅ EAGLE 投机解码
- ✅ 自定义 Metal kernel / 分布式推理 / 模型转换 / 困惑度评估
- ✅ 量化 embedding 自动反量化（Qwen3 及类似模型）
- ✅ 显式 head_dim 支持（Qwen3 head_dim=128 ≠ hidden_size/num_heads）

### 真实缺陷（均已修复）
1. ~~**generate 循环性能差**~~ → ✅ 已修复（KV cache 增量更新）
2. ~~**Gemma loader 未实现**~~ → ✅ 已实现
3. ~~**无流式 token 输出**~~ → ✅ 已实现
4. ~~**无 EOS 停止**~~ → ✅ 已实现
5. ~~**无 chat template 应用**~~ → ✅ 已实现
6. ~~**无多轮对话**~~ → ✅ Chat template 支持多轮格式
7. ~~**Qwen3 量化 embedding 加载失败**~~ → ✅ 自动反量化
8. ~~**Qwen3 head_dim 不匹配**~~ → ✅ 显式 head_dim 解析
9. ~~**大 tokenizer.json 加载失败**~~ → ✅ 限制提升至 50MB

### 剩余可选项
- 所有 P0-P3 任务已完成，包括可选任务 28.3 (Qwen3) 和 28.4 (GLM-4)

---

## II. 逐项对比

### 1. 对比 mlx (Apple 官方框架)

| 能力 | mlx | mlx-zig | 差距 | 借鉴价值 |
|------------|-----|---------|-----|---------------|
| 惰性求值 | ✅ 核心 | ✅ 通过 mlx-c | 无 | — |
| Metal kernel | ✅ 自动 | ✅ 通过 mlx-c | 无 | — |
| `mlx.compile` | ✅ | ✅ 已绑定 | 无 | — |
| 自定义 Metal kernel | ✅ | ❌ | **高** | 可编写自定义 Metal shader 加速热点 |
| 分布式 (NCCL) | ✅ v0.29+ | ❌ | 中 | 多 Mac 集群推理 |
| CUDA 后端 | ✅ v0.29+ | ❌ | 低 | Apple Silicon 定位不需要 |

**借鉴方向**：自定义 Metal kernel。mlx 允许注册自定义 Metal shader，可为 MoE expert dispatch 和 fused attention 等热点编写专用 kernel，绕过通用算子开销。

### 2. 对比 mlx-c (C API 层)

| 能力 | mlx-c | mlx-zig 绑定 | 差距 |
|------------|-------|-----------------|-----|
| 全部算子 | ~200 | ~200 | 无 |
| quantize/dequantize | ✅ 4 模式 | ✅ 4 模式 | 无 |
| to_fp8/from_fp8 | ✅ | ✅ | 无 |
| qqmm/gather_qmm | ✅ | ✅ | 无 |
| mlx_fast (SDPA/RoPE) | ✅ | ✅ | 无 |
| mlx_distributed | ✅ | ❌ | 中 |

**借鉴方向**：`mlx_distributed` 绑定。mlx-c 有分布式通信 API（send/recv/all_reduce），可实现多 Mac tensor parallelism。

### 3. 对比 mlx-lm (Python 参考实现)

| 能力 | mlx-lm | mlx-zig | 差距 | 优先级 |
|------------|--------|---------|-----|----------|
| 模型架构数 | 50+ | 5 | **高** | P1 |
| 流式 token 输出 | ✅ yield | ❌ 批量输出 | **高** | P0 |
| Chat template (Jinja2) | ✅ | 部分 (仅 V4) | **高** | P0 |
| EOS 停止 | ✅ | ❌ | **高** | P0 |
| Token decode (ids→text) | ✅ | ✅ 存在但 LLaMA chat 中未用 | 中 | P1 |
| 重复惩罚 | ✅ | ❌ | 中 | P2 |
| Top-p / Top-k 采样 | ✅ | ✅ | 无 | — |
| 困惑度评估 | ✅ | ❌ | 中 | P2 |
| 模型转换 (convert) | ✅ | ❌ stub | 低 | P3 |
| GGUF 导出 | ✅ | ❌ | 低 | P3 |

**最高优先级借鉴**：
1. **流式 token 输出** — mlx-lm 的 `generate_step` 是 yield 模式，每个 token 生成即输出。mlx-zig 的 generate 等待全部完成。
2. **EOS 停止** — mlx-lm 检查 `eos_token_id` 并提前终止。mlx-zig 始终生成 max_tokens。
3. **Chat template** — mlx-lm 用 Jinja2 渲染 chat template。mlx-zig 仅在 V4 路径有 chat template，LLaMA 路径传递原始 prompt。

### 4. 对比 oMLX (生产级服务器)

| 能力 | oMLX | mlx-zig | 差距 | 优先级 |
|------------|------|---------|-----|----------|
| Continuous Batching | ✅ 真实 | ⚠️ 框架存在但无真实批量前向 | **高** | P1 |
| SSE keep-alive | ✅ | ❌ | 中 | P2 |
| Context scaling | ✅ | ❌ | 中 | P2 |
| Model TTL | ✅ | ❌ | 低 | P3 |
| Web 管理面板 | ✅ | ❌ | 低 | P3 |
| Anthropic API 兼容 | ✅ | ❌ | 低 | P3 |

**借鉴方向**：真实批量前向。当前引擎循环逐个处理解码请求，未将多个请求 token 合并为一个批量张量进行单次前向。这是 continuous batching 的核心——需要 `batch_builder` 真正构建批量输入。

### 5. 对比 TileKernels (DeepSeek GPU Kernel 库)

| 能力 | TileKernels | mlx-zig | 差距 |
|------------|-------------|---------|-----|
| 数值验证框架 | ✅ cosine sim + bias check | ✅ golden test | 无 |
| Per-token/per-block 量化 | ✅ | ✅ | 无 |
| Sinkhorn 归一化 | ✅ | ✅ | 无 |
| MoE 路由管线 | ✅ | ✅ | 无 |
| 自定义 CUDA kernel | ✅ | N/A (Metal) | — |
| FP8/FP4 训练 kernel | ✅ | ❌ | 低 (推理优先) |

**借鉴方向**：无明显差距。TileKernels 的核心价值（MoE 路由、量化 kernel、数值验证）已在 mlx-zig 中实现。

### 6. 对比 vLLM (工业级推理引擎)

| 能力 | vLLM | mlx-zig | 差距 | 优先级 |
|------------|------|---------|-----|----------|
| PagedAttention | ✅ | ✅ | 无 | — |
| Prefix Caching | ✅ 真实 | ⚠️ Hash 已注册但查找未使用 | 中 | P2 |
| Chunked Prefill | ✅ | ✅ 框架存在 | 无 | — |
| Speculative Decoding | ✅ ngram/EAGLE/Medusa | ✅ ngram | 中 | P2 |
| Guided Decoding | ✅ FSM/xgrammar | ✅ FSM | 无 | — |
| Disaggregated Serving | ✅ | ❌ | 低 | P3 |
| Multi-GPU TP/PP | ✅ | ❌ | 低 | P3 |
| OpenAI API 完整性 | ✅ | ⚠️ 基础 | 中 | P2 |

**借鉴方向**：
1. **OpenAI API 完整性** — vLLM 支持完整 `/v1/chat/completions`，包括 `stream`、`stop`、`temperature`、`top_p`、`max_tokens`、`logprobs`、`tool_calls`。mlx-zig 的服务器仅支持基础字段。
2. **EAGLE 投机解码** — 比 n-gram 更高效，但需要额外的 draft model head。

---

## III. 优先级排序：ROI 最高的改进项

### P0 (立即修复 — 影响基础可用性)

1. **流式 token 输出** — 每个 token 生成即打印，不等待全部完成
2. **EOS 停止** — 检查 `eos_token_id` 提前终止 generate 循环
3. **LLaMA 路径应用 chat template** — 从 tokenizer_config.json 读取 chat_template
4. **generate 性能** — 确认 KV cache 增量更新正确运行（无重复 prefill）

### P1 (近期 — 显著提升竞争力)

5. **更多模型架构** — Phi-3/4、Qwen3（非 VL）、GLM-4、GPT-OSS
6. **真实批量前向** — 使用 batch_builder 在引擎循环中合并多个请求
7. **token decode 集成** — LLaMA chat 路径输出可读文本而非 token ID

### P2 (中期 — 生产级打磨)

8. **OpenAI API 完整性** — stop、logprobs、tool_calls
9. **SSE keep-alive** — 长 prefill 期间发送心跳
10. **重复惩罚** — 避免重复输出
11. **EAGLE 投机解码** — 更高效的投机解码

### P3 (长期 — 差异化特性)

12. **自定义 Metal kernel** — 为 MoE dispatch 编写专用 shader
13. **多 Mac 分布式推理** — 通过 mlx_distributed
14. **模型转换工具** — HF → MLX safetensors
15. **Anthropic API 兼容** — 同时支持 Claude API 格式与 OpenAI

(End of file - total 164 lines)
