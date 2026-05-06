# 附录：关键文件与文档索引

## A.1 关键源码文件索引

| 文件 | 行数 | 职责 | 风险等级 |
|------|-------|---------------|------------|
| `src/models/deepseek_v4.zig` | 3,091 | DeepSeek V4 完整实现 (MLA+MoE+YARN+mHC) | 中 (2 @constCast) |
| `src/models/deepseek_v4_loader.zig` | 2,071 | V4 权重加载器 (分片+反量化+专家策略) | 低 |
| `src/main.zig` | 1,764 | CLI 入口点 (chat/serve/benchmark/quantize/lora-train) | 低 |
| `src/server.zig` | 1,517 | HTTP 服务器 (OpenAI 兼容+SSE+工具调用) | 中 (批量未集成) |
| `src/ops/nn.zig` | 1,354 | NN 层 (Linear/BatchNorm/LSTM/GRU/RNN/Attention) | **高** (34 dataSliceMut) |
| `src/speculative.zig` | 1,223 | 双轨投机解码 (PLD+EAGLE) | 低 |
| `src/kvcache/paged.zig` | 1,152 | Paged KV Cache (BlockManager+CoW+prefix hashing) | 低 |
| `src/guided.zig` | 1,129 | 引导解码 FSM (JSON Schema/Regex 约束) | 低 |
| `src/io/safetensors_reader.zig` | 1,045 | Safetensors 随机访问读取器 (pread 零拷贝) | 低 |
| `src/quantize.zig` | 872 | 量化基础设施 (affine/mxfp4/nvfp4/mxfp8/turboquant) | 低 |
| `src/qlora.zig` | 773 | QLoRA 量化+低秩适配器训练 | 低 |
| `src/tokenizer/bpe.zig` | 744 | BPE Tokenizer (HF tokenizer.json 格式) | 低 |
| `src/models/llama.zig` | 725 | LLaMA/Mistral/Qwen/Gemma/Phi 标准架构 | 低 |
| `src/models/minimax.zig` | 712 | MiniMax 模型适配器 | 中 (4 @constCast) |
| `src/prompt_cache.zig` | 563 | Prompt cache 持久化 (safetensors) | **高** (类型漏洞) |
| `src/distributed.zig` | 222 | 分布式推理 (多 Mac tensor parallelism) | 中 (空 deinit) |
| `src/optim.zig` | 217 | AdamW 优化器 | 中 (临时对象风暴) |
| `src/c.zig` | ~200 | C 绑定层 (mlxErrorHandler+check+类型再导出) | 低 |

## A.2 文档目录索引

| 文档 | 类型 | 内容 |
|----------|------|---------|
| `docs/deep-analysis.md` | 审计报告 | v0.3.0 自审计，P0-P3 问题列表 |
| `ROADMAP.md` | 路线图 | Phase 0-7 进度跟踪，Task 13-34 完成状态 |
| `.kiro/specs/production-deployment/design.md` | 设计文档 | 六层架构设计，Mermaid 图表 |
| `.kiro/specs/production-deployment/design-paged-kv-cache.md` | 设计文档 | PagedKVCache 算法细节 (updateAndFetch 六步法) |
| `.kiro/specs/production-deployment/design-server.md` | 设计文档 | 服务器架构、请求流、调度器设计 |
| `docs/ecosystem-analysis.md` | 调研 | 五项目对比：vLLM/mlx-lm/oMLX/TileKernels/mlx-rs |
| `docs/BENCHMARK.md` | 性能 | 基准测试方法与阈值定义 |
| `docs/DEEPSEEK-V4-FIX-PLAN.md` | 修复计划 | V4 问题诊断与修复方案 |
| `docs/FIX-REPORT-DEEPSEEK-V4.md` | 修复报告 | V4 修复验证结果 |
| `docs/tilekernels-analysis.md` | 调研 | TileKernels 算子融合与量化分析 |
| `docs/deepseek-v4-optimization-plan.md` | 优化 | V4 性能优化策略 |
| `docs/DEEPSEEK-V4-CHAT-ANALYSIS.md` | 分析 | V4 Chat 模式行为分析 |
| `docs/competitive-advantages.md` | 策略 | 项目竞争优势分析 |

## A.3 构建与依赖

### 外部依赖

- `mlx-c`：Apple MLX C API 绑定
  - 检测优先级：`-Dmlx_prefix` > `MLX_C_PREFIX` env > `pkg-config --variable=prefix mlxc` > `/opt/homebrew`
- `zig_regex`：正则表达式库（Zig 包，固定哈希，非 main 分支）

### 构建产物

- `libdmlx.a`：静态库
- `dmlx`：CLI 工具
  - `chat`：交互式对话
  - `serve`/`server`：HTTP 服务
  - `benchmark`：性能基准测试
  - `quantize`：权重量化
  - `lora-train`：LoRA 微调
  - `convert`：格式转换（TODO）
  - `evaluate`：困惑度评估
- `example`：示例程序
- `test`：测试运行器（50+ 模块，350 测试）

### 平台支持

| 平台 | 状态 | 备注 |
|----------|--------|-------|
| macOS Apple Silicon | ✅ 主力 | Metal GPU + UMA 统一内存 |
| macOS Intel | ⚠️ 可能可用 | 无 Metal，回退到 CPU |

## A.4 代码风格 (`.sisyphus/zig-style.md`)

- **命名**：PascalCase（类型）、camelCase（函数）、snake_case（常量）
- **禁止**：`// ====` 分隔符、`//!` 模块文档、`as any`、空 catch 块
- **要求**：`zig build` 通过、`zig build test` 通过、`zig fmt src/` 格式化
- **许可证**：AGPL-3.0

## A.5 版本信息

```zig
// src/root.zig
pub const version = "0.3.0-mlx-c";
```

(End of file - total 85 lines)
