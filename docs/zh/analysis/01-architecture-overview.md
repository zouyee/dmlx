# 第一章 整体架构与模块分布

## 1.1 六层架构

dmlx 采用六层架构（引自 `.kiro/specs/production-deployment/design.md`）：

```
┌─────────────────────────────────────────┐
│  Layer 6: Tooling Layer                 │
│  main.zig (CLI) / benchmark.zig         │
├─────────────────────────────────────────┤
│  Layer 5: Model Layer                   │
│  models/ (LLaMA, DeepSeek V4, Nemotron) │
│  expert_stream.zig (MoE 专家流式)        │
├─────────────────────────────────────────┤
│  Layer 4: Memory Layer                  │
│  kvcache/ (6种策略) / model_pool.zig    │
│  memory.zig / prompt_cache.zig          │
├─────────────────────────────────────────┤
│  Layer 3: Service Layer                 │
│  server.zig / scheduler.zig             │
│  batch_builder.zig                      │
├─────────────────────────────────────────┤
│  Layer 2: Inference Engine              │
│  generation.zig / speculative.zig       │
│  guided.zig / model_registry.zig        │
├─────────────────────────────────────────┤
│  Layer 1: Foundation Layer              │
│  c.zig / array.zig / ops.zig + ops/     │
│  fast.zig (融合kernel) / fused.zig      │
└─────────────────────────────────────────┘
```

## 1.2 模块规模分布（Top 20）

```
3,091  models/deepseek_v4.zig
2,071  models/deepseek_v4_loader.zig
1,764  main.zig
1,517  server.zig
1,354  ops/nn.zig
1,223  speculative.zig
1,152  kvcache/paged.zig
1,129  guided.zig
1,045  io/safetensors_reader.zig
  912  tokenizer/pre_tokenizer.zig
  872  quantize.zig
  773  qlora.zig
  744  tokenizer/bpe.zig
  725  models/llama.zig
  712  models/minimax.zig
  702  kvcache/prefix_disk.zig
  673  models/expert_cache.zig
  649  models/expert_stream.zig
  640  trainer.zig
  563  prompt_cache.zig
```

## 1.3 核心枢纽统计（import 频次）

| 被导入模块 | 频次 | 角色 |
|-----------|------|------|
| `std` | 128 | 标准库 |
| `../c.zig` | 60 | C 绑定层 |
| `../array.zig` | 55 | Array 包装器 |
| `../ops.zig` | 48 | 算子入口 |
| `../dtype.zig` | 35 | 类型系统 |

## 1.4 分层依赖关系详解

### Layer 1: C 绑定层（`src/c.zig`）
- 最薄包装层，将 `mlx-c` 的 C API 封装为 Zig 错误处理
- `mlxErrorHandler`：全局 C 错误处理器，捕获 C++ 异常文本到 `last_error_buffer[2048]`
- `check(rc)`：统一错误检查，消费后自动清空缓冲区
- 类型重导出：`mlx_array`→`Array`，`mlx_dtype`→`Dtype`，`mlx_stream`→`Stream`

### Layer 2: 核心类型
- `Array`：Zig 惯用包装器，提供 `fromHandle`/`fromData`/`fromSlice`/`zeros`/`ones`
- `eval()`：显式使用 `mlx_eval` 向量版以支持跨设备调度
- `dataPtr<T>()` / `dataSlice<T>()`：comptime 类型安全访问
- `dataSliceMut<T>()`：**危险方法**，通过 `@constCast` 绕过 CoW 语义

### Layer 3: 算子层
- `ops.zig`：核心入口，200+ 操作，提供 `EagerContext` 执行模式
- `fast.zig`：绑定 MLX 融合 kernel（rms_norm/rope/sdpa/layer_norm）
- `fused.zig`：`mlx_compile` 融合图封装，含 `compiledAdamWStep`

### Layer 4: 自动微分与参数树
- `grad.zig`：`valueAndGrad`、`vjp`、`jvp`
- `tree.zig`：通过 Zig comptime 反射递归遍历嵌套结构体

### Layer 5: 模型层
- `llama.zig`：标准 LLaMA/Mistral/Qwen/Gemma/Phi（725 行）
- `deepseek_v4.zig`：项目最大文件（3,091 行），含 MLA + MoE + YARN RoPE + mHC

### Layer 6: 推理引擎
- `ModelVTable`：运行时多态接口
- `generateStep`：单次前向 + 采样，`ScopedArrayArena` 追踪临时数组
- `streamGenerateSpeculative`：PLD n-gram 投机解码
- `streamGenerateEagle`：EAGLE 投机解码

### Layer 7: 服务层
- `server.zig`（1,517 行）：OpenAI-compatible HTTP + SSE + 工具调用
- `scheduler.zig`：请求调度器
- **活跃问题**：`batch_builder.zig` 已构建但未完全集成到 engine loop
