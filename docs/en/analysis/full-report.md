# MLX-Zig 深度技术分析报告集

> 分析日期：2026-05-03  
> 分析轮次：三轮递进式深度分析  
> 覆盖范围：~52,933 行 Zig 源码，50+ 测试模块，25 份设计文档  
> 目标平台：macOS Apple Silicon

---

## 报告文件索引

| 文件 | 大小 | 内容 |
|------|------|------|
| `00-executive-summary.md` | ~3KB | 执行摘要与核心结论 |
| `01-architecture-overview.md` | ~6KB | 六层架构全景 + 模块规模分布 |
| `02-core-infrastructure.md` | ~5KB | C绑定层、Array包装器、算子层、EagerContext |
| `03-models-and-inference.md` | ~7KB | DeepSeek V4、投机解码、引导解码 |
| `04-kv-cache-subsystem.md` | ~4KB | 六级策略、Paged/Tiered/Prompt Cache |
| `05-server-and-service.md` | ~3KB | HTTP服务器、Scheduler、Batched Forward |
| `06-quantization-training.md` | ~4KB | 量化体系、专家流式、LoRA/QLoRA/AdamW |
| `07-testing-quality.md` | ~4KB | 50+测试模块分析、数值等价性、覆盖缺口 |
| `08-security-audit.md` | ~5KB | @constCast统计、类型安全漏洞、资源泄漏 |
| `09-issue-verification-matrix.md` | ~4KB | 与v0.3.0自我审计的交叉验证 |
| `10-technical-debt.md` | ~4KB | 债务热力图、修复优先级、架构建议 |
| `appendix-file-index.md` | ~2KB | 关键文件索引、文档索引、构建依赖 |

---

## 最关键的发现（按严重程度排序）

### 🔴 P0 - 生产崩溃风险

1. **`prompt_cache.zig:74` 类型安全漏洞**
   - `const std_cache: *StandardKVCache = @ptrCast(@alignCast(cache.ptr));`
   - 当使用默认配置 `--kv-strategy paged_quantized` + `--prompt-cache-file` 时直接崩溃
   - 因为 `cache.ptr` 实际指向 `PagedKVCache`，但强制转换为 `StandardKVCache`

2. **`nn.zig` 34 处 `dataSliceMut` 未清除**
   - `Linear`/`BatchNorm`/`LSTM`/`GRU`/`RNN`/`MultiHeadAttention` 仍用纯 CPU 标量循环
   - 完全绕过 Metal GPU 加速，且 `@constCast` 破坏 MLX CoW 语义

### 🟡 P1 - 性能与安全

3. **`sampling.zig` insertion sort**
   - 4处调用对 128K vocab 每 token ~82亿次比较
   - 改为 `pdq` 或 `mlx_topk` 仅需半天

4. **Batched Forward 未集成**
   - `batch_builder.zig` 已构建但未接入 `server.zig` engine loop
   - 连续批处理吞吐潜力未释放

5. **AdamW 临时对象风暴**
   - 每参数每步 ~15 个临时 mlx_array，7B 模型每步 ~3000 个

### 🟢 P2 - API 与可维护性

6. **`allocator` 参数误导**（`array.zig` 3 处）
7. **`EagerContext` stream 泄漏**（无 `deinit`）
8. **ops.zig 与 ops/ 子模块功能重复**

---

## 分析方法论

**第一轮**：整体架构分层梳理
- 精读 8 个核心文件：`c.zig`, `array.zig`, `ops.zig`, `generation.zig`, `server.zig`, `deepseek_v4.zig` 等
- 产出六层架构全景图和模块依赖统计

**第二轮**：模块耦合、安全边界、子系统专项
- 定位全库 38 处 `dataSliceMut`、4 处 `insertion` sort
- 分析 IO 层（`safetensors_reader.zig`）、Tokenizer、Vision、Diffusion、MoE Router
- 与项目自我审计文档 `deep-analysis.md` 交叉验证

**第三轮**：测试质量、构建系统、性能实证、代码风格
- 分析 50+ 测试模块的实际内容
- 统计全库 10 处 `@constCast`
- 评估文档完整性（25 份文档）和代码风格一致性
- 整合形成最终报告
# 执行摘要

> **版本**：基于代码库当前 HEAD（v0.3.0-mlx-c）  
> **分析轮次**：三轮递进式深度分析  
> **覆盖范围**：~52,933 行 Zig 源码（含测试），50+ 测试模块，25 份设计文档  
> **分析日期**：2026-05-03  
> **目标平台**：macOS Apple Silicon

---

## 项目定位

MLX-Zig 是基于 Apple MLX C 绑定（`mlx-c`）的全栈 LLM 推理与训练系统，以 Zig 语言编写，目标平台为 macOS Apple Silicon。项目经历了从原型到生产级的显著跃迁，当前具备与 Python vLLM/mlx-lm 对标的功能深度。

## 核心结论

1. **工程成熟度极高**：350 个测试全部通过，Phase 0–7 路线图全部完成，代码结构清晰，文档体系完善
2. **前沿功能齐全**：DeepSeek V4（3091 行）、投机解码（PLD+EAGLE）、引导解码（JSON Schema/Regex）、MoE 路由、QLoRA、TurboQuant 均已落地
3. **技术债务仍存**：`nn.zig` 中 34 处 `dataSliceMut` 调用未完全清除、`sampling.zig` 的 `insertion` sort 性能瓶颈、`prompt_cache.zig` 存在类型安全漏洞
4. **最大风险点**：`prompt_cache.zig` 对运行时多态的 `KVCacheStrategy` 做 `@ptrCast` 强制类型转换，在 Paged/Quantized/Tiered 模式下会导致崩溃

## 项目规模

| 指标 | 数值 |
|------|------|
| 总代码行数 | ~52,933 行 |
| 源码行数（不含测试） | ~42,455 行 |
| 测试模块数 | 50+ |
| 通过测试数 | 350 |
| 最大源文件 | `models/deepseek_v4.zig`（3,091 行） |
| 文档文件数 | 25 份 |
| 外部依赖 | `mlx-c`（C 库）、`zig_regex`（Zig 包） |

## 最关键的 5 个发现

| 排名 | 问题 | 严重度 | 位置 |
|------|------|--------|------|
| 1 | Prompt Cache 类型安全漏洞：将 `PagedKVCache` 强制转为 `StandardKVCache` | **P0** | `prompt_cache.zig:74` |
| 2 | NN 层 34 处 `dataSliceMut`：纯 CPU 标量循环绕过 GPU | **P0** | `ops/nn.zig` |
| 3 | Sampling 对 128K vocab 使用 insertion sort，~82亿次比较/token | **P1** | `sampling.zig`（4处） |
| 4 | Batched forward 未集成到 engine loop | **P1** | `server.zig` |
| 5 | AdamW 每步创建 ~3000 个临时 mlx_array | **P1** | `optim.zig` |

## 子系统成熟度速览

| 子系统 | 成熟度 | 关键风险 |
|--------|--------|---------|
| DeepSeek V4 模型 | ⭐⭐⭐⭐⭐ | 极复杂但已验证 |
| KV Cache (6种策略) | ⭐⭐⭐⭐☆ | prompt_cache 类型漏洞 |
| 服务器 | ⭐⭐⭐⭐☆ | batch 未集成 |
| 投机/引导解码 | ⭐⭐⭐⭐⭐ | EAGLE 仅单 token draft |
| NN 层 (`ops/nn.zig`) | ⭐⭐☆☆☆ | **最大技术债务** |
| 采样 | ⭐⭐⭐☆☆ | insertion sort |
| 分布式推理 | ⭐⭐⭐☆☆ | deinit 为空 |
# 第一章 整体架构与模块分布

## 1.1 六层架构

MLX-Zig 采用六层架构（引自 `.kiro/specs/production-deployment/design.md`）：

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
# 第二章 核心基础设施深度分析

## 2.1 `c.zig`：错误处理的演进

### 已修复：v0.3.0 → 当前

v0.3.0 审计发现错误处理为 P0 问题：`check(rc)` 仅返回 `error.MlxError`，无上下文。

当前已修复：

```zig
pub fn check(rc: c_int) !void {
    if (rc != 0) {
        const msg = mlx_get_last_error();
        std.log.err("MLX error: {s}", .{std.mem.span(msg)});
        return error.MlxError;
    }
}
```

### 仍未修复

错误联合类型仍单一（`error.MlxError`），未根据 mlx-c 错误码细分为：
- `MlxOOM`（内存不足）
- `MlxInvalidArg`（非法参数）
- `MlxDeviceError`（设备错误）

## 2.2 `array.zig`：API 设计矛盾

### `allocator` 参数误导（P2）

`fromData`、`zeros`、`ones` 接受 `allocator` 参数但完全忽略：

```zig
pub fn fromData(allocator: std.mem.Allocator, comptime T: type, ...) !Array {
    _ = allocator;  // 完全没用
    // ...
}
```

因为 mlx-c 内部管理内存，这误导用户以为 Zig allocator 在管理 Array 内存。

### `strides()` 64 位假设（P2）

```zig
pub fn strides(self: Array) []i64 {
    // 假设 size_t == i64（64位平台）
}
```

在 32 位平台上会截断。

## 2.3 `ops.zig`：API 冗余

`ops.zig` 与 `ops/shape.zig`、`ops/math.zig` 存在功能重复：

| 操作 | ops.zig 位置 | ops/ 子模块位置 |
|------|-------------|----------------|
| reshape | `ops.zig` | `ops/shape.zig` |
| transpose | `ops.zig` | `ops/shape.zig` |
| softmax | `ops.zig` | `ops/math.zig` |
| relu | `ops.zig` | `ops/activations.zig` |

建议：`ops.zig` 只保留 EagerContext 和最核心的二元/一元算子，其余委托给子模块。

## 2.4 `EagerContext`：Stream 生命周期缺陷

```zig
pub fn init(allocator: std.mem.Allocator) EagerContext {
    return .{
        .allocator = allocator,
        .stream = .{ .inner = c.c.mlx_default_cpu_stream_new() },
    };
}
```

**问题**：每次调用 `init` 都会创建新的 mlx_stream，但 `EagerContext` 仍无 `deinit` 方法释放它。

**影响**：频繁创建 `EagerContext` 会产生 stream 资源泄漏。

**建议**：添加 `deinit` 方法，或改为获取全局默认 stream 的引用（非创建新实例）。

## 2.5 构建系统（`build.zig`）

### 四级 mlx-c 探测（已修复 P1 问题）

优先级：`-Dmlx_prefix` > `MLX_C_PREFIX` env > `pkg-config --variable=prefix mlxc` > `/opt/homebrew` fallback

```zig
const mlx_prefix = blk: {
    if (b.option([]const u8, "mlx_prefix", ...)) |p| break :blk p;
    if (b.graph.environ_map.get("MLX_C_PREFIX")) |p| break :blk p;
    if (pkgConfigMlxPrefix(b)) |p| break :blk p;
    break :blk "/opt/homebrew";
};
```

### 构建产物

- `libmlx-zig.a`：静态库
- `mlx-zig`：CLI 工具
- `example`：示例程序
- `test`：测试 runner

### 平台框架链接

构建系统检测目标平台，在 macOS 下链接 Accelerate、Metal、Foundation 框架以启用 GPU 加速。
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
# 第四章 KV Cache 子系统

## 4.1 策略接口设计（`kvcache/interface.zig`）

VTable 运行时多态设计：

```zig
pub const VTable = struct {
    updateAndFetch: *const fn (ctx: *anyopaque, keys: Array, values: Array, stream: mlx_stream) anyerror!KVSlice,
    currentLen: *const fn (ctx: *anyopaque) usize,
    reset: *const fn (ctx: *anyopaque) void,
    filter: ?*const fn (ctx: *anyopaque, indices: []const usize, allocator: Allocator) anyerror!void,
    rollback: ?*const fn (ctx: *anyopaque, to_len: usize) void,
    deinit: *const fn (ctx: *anyopaque, allocator: Allocator) void,
};
```

设计亮点：运行时策略切换 + comptime 内部特化 + 注意力层完全解耦。

## 4.2 六级策略对比

| 策略 | 特点 | 适用场景 | 代码位置 |
|------|------|---------|---------|
| Standard | 简单连续缓冲区 | 单请求、短序列 | `kvcache/standard.zig` |
| Rotating | 环形缓冲区，固定窗口 | 超长序列（避免 OOM） | `kvcache/rotating.zig` |
| Quantized | 4/8/16 bit KV 压缩 | 内存受限 | `kvcache/quantized.zig` |
| Paged | 32-token 页 + 页表 + CoW | 连续批处理（**默认**） | `kvcache/paged.zig` |
| PagedQuantized | Paged + Quantized 组合 | 极致内存优化 | `kvcache/paged.zig` |
| Tiered | RAM hot + SSD cold + LRU | 超长上下文 + 多模型 | `kvcache/tiered.zig` |

## 4.3 PagedKVCache（`kvcache/paged.zig`，1,152 行）

### 核心设计

- **页大小**：默认 32 tokens（针对 Apple Silicon Metal GPU 内存对齐调优）
- **BlockManager**：管理 free pool、per-request 块映射、CoW 机制
- **前缀哈希**：`hashBlock` 使用 Wyhash 计算滚动哈希
- **Copy-on-Write**：共享块 `ref_count > 1` 时分配新块并拷贝数据

### updateAndFetch 算法流程

1. **分配页**：`new_total = cached_len + seq_len`，按需分配新页
2. **写入 KV**：通过 `mlx_slice_update` 将 keys/values 写入对应页
3. **注册哈希**：页写满时计算哈希，注册到 `page_hashes` 映射
4. **Gather 输出**：将分散的页拼接为连续的 `[batch, heads, seq, dim]` 数组
5. **量化路径**： quantized 页需先 dequantize 再 concatenate

## 4.4 TieredKVCache（`kvcache/tiered.zig`）

- 包装 `PagedKVCache` 作为 hot tier
- 超出 `hot_capacity` 时 LRU 页写入 SSD：`{cold_dir}/block_{id}.safetensors`
- `restoreFromSSD`：从 safetensors 文件恢复块到 hot tier

## 4.5 Prompt Cache（`prompt_cache.zig`，563 行）

支持 save/load KV cache 状态到 safetensors 文件。

### 🔴 高危漏洞

```zig
// prompt_cache.zig:74
const std_cache: *StandardKVCache = @ptrCast(@alignCast(cache.ptr));
```

**问题**：`savePromptCache` 接收 `[]KVCacheStrategy`（运行时多态），但直接将 `cache.ptr` 强制转换为 `*StandardKVCache`。

**后果**：
- `PagedKVCache` 的 `ptr` 指向 `PagedKVCache` 结构体，字段布局完全不同
- 访问 `.offset` 时读取的是 `pages` 指针的一部分——数值无意义
- 访问 `.keys`/`.values` 时可能读取到 `PageTableEntry` 数组的指针，导致 segfault

**触发条件**：使用默认配置 `--kv-strategy paged_quantized` + `--prompt-cache-file` 时必现。

**修复建议**：
1. 在 `KVCacheStrategy.VTable` 中增加 `saveState`/`loadState` 方法
2. 或添加运行时类型检查：`std.debug.assert(cache.vtable == &StandardKVCache.vtable)`
# 第五章 服务器与服务层

## 5.1 `server.zig`（1,517 行）

OpenAI-compatible HTTP 服务器，功能覆盖：

- **并发模型**：每个连接通过 `io.async` 并发处理
- **Engine Loop**：后台 async fiber 驱动 scheduler
  - schedule → batch → forward → postprocess
- **SSE 流式**：`text/event-stream` 格式，`data: {...}` 事件
- **工具调用**：OpenAI functions 格式解析与执行
- **引导解码**：集成 `GuidedDecoder`，支持请求级 JSON Schema / Regex 约束

## 5.2 服务器配置

```zig
ServerConfig{
    .kv_strategy = .paged_quantized,   // 默认分页量化
    .kv_quant = .simple,                // 量化算法
    .kv_tier = .ram,                    // 存储层级
    .speculative_ngram = null,          // 投机解码 n-gram 大小
    .smelt = false,                     // 专家流式加载
    .smelt_strategy = "preload",
    .distributed = false,               // 分布式推理
}
```

## 5.3 Scheduler（`scheduler.zig`）

三阶段循环：
1. **Schedule**：从 waiting/running 队列选择请求，分配 KV cache blocks
2. **Forward**：执行模型前向传播 + 采样
3. **Postprocess**：追加 token、检查停止条件、释放已完成请求的 blocks

## 5.4 活跃问题：Batched Forward 未完成

`batch_builder.zig`（256 行）已实现请求合并逻辑：

```zig
// batch_builder.zig: 将多个 decode 请求合并为单个 batched input tensor
```

但 `server.zig` 的 `engineLoop` 中 decode 仍按单请求处理：

```zig
// TODO: batch_builder would merge all decode requests into a single forward pass
```

**影响**：
- 连续批处理的最大吞吐潜力未释放
- 每个请求独立 forward，GPU 利用率低
- 这是当前推理吞吐量的最大瓶颈

**修复工作量**：预计 1-2 周，需将 `batch_builder.buildBatch()` 集成到 scheduler 的 forward 阶段。

## 5.5 分布式推理（`distributed.zig`，222 行）

支持多 Mac 间的 tensor parallelism（MPI 风格集合通信）：

- `allSum` / `allGather` / `allMax` / `allMin`：集合归约
- `send` / `recv`：点对点通信
- `sumScatter`：reduce-scatter 梯度聚合

### ⚠️ 资源泄漏

```zig
pub fn deinit(self: *DistributedGroup) void {
    _ = self;
    // mlx_distributed_group has no explicit free in this mlx-c version
}
```

`deinit` 为空实现。频繁创建/销毁 `DistributedGroup` 会产生资源泄漏。
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
# 第八章 安全边界与代码审查

## 8.1 `@constCast` 全库统计

全库共 **10 处** `@constCast` 调用：

| 位置 | 行号 | 用途 | 风险 |
|------|------|------|------|
| `array.zig` | 150 | `dataSliceMut`：将 const 指针转为可变 | **高** |
| `tree.zig` | 302 | `treeMapInPlace`：递归遍历字段指针 | 中 |
| `tree.zig` | 317 | `treeToArrayPtrs`：收集 Array 指针 | 低 |
| `guided.zig` | 85 | `FiniteStateMachine.deinit`：`[]State` → `*State` | 低 |
| `safetensors_reader.zig` | 494 | `mmap` 区域指针转换 | 中 |
| `safetensors_reader.zig` | 520 | `munmap` 解除 const 限制 | 低 |
| `minimax.zig` | 59-60 | RoPE sin/cos cache 初始化 | **高** |
| `deepseek_v4.zig` | 198-199 | YARN RoPE sin/cos cache 初始化 | **高** |
| `deepseek_v4.zig` | 399 | Attention mask 初始化 | **高** |

### `dataSliceMut` 泛滥

`array.zig:148`：
```zig
pub fn dataSliceMut(self: Array, comptime T: type) ![]T {
    const ptr = try self.dataPtr(T);
    return @constCast(ptr)[0..self.size()];
}
```

调用统计：
- `ops/nn.zig`：**34 处**
- `models/minimax.zig`：**4 处**
- 合计：**38 处**

**项目声称已修复**（`production-roadmap.md`）："安全：`@constCast` 绕过 CoW → 全部改用 mlx-c 算子链 ✅"

**实际状态**：修复未完全完成。`nn.zig` 中 BatchNorm、LSTM、GRU、RNN、MultiHeadAttention、RoPE、Embedding 仍通过 `dataSliceMut` 使用纯 CPU 标量循环。

## 8.2 `prompt_cache.zig` 类型安全漏洞（P0）

```zig
// prompt_cache.zig:74
const std_cache: *StandardKVCache = @ptrCast(@alignCast(cache.ptr));
```

**问题**：接收 `[]KVCacheStrategy`（运行时多态），但强制转换为 `*StandardKVCache`。

**PagedKVCache 与 StandardKVCache 布局差异**：

```zig
// StandardKVCache
pub const StandardKVCache = struct {
    keys: Array,
    values: Array,
    offset: usize,
};

// PagedKVCache
pub const PagedKVCache = struct {
    pages: []Page,
    sequences: []SequenceState,
    page_size: usize,
    page_hashes: std.HashMap(...),
    // ...
};
```

当 `cache.ptr` 实际指向 `PagedKVCache` 时：
- `std_cache.offset` 读取的是 `PagedKVCache.pages` 指针的低 64 位——无意义
- `std_cache.keys` 读取的是 `PagedKVCache.sequences` 指针——后续 `sliceCache` 操作 segfault

**触发条件**：默认配置 `--kv-strategy paged_quantized` + `--prompt-cache-file` 时必现。

## 8.3 `distributed.zig` 资源泄漏

```zig
pub fn deinit(self: *DistributedGroup) void {
    _ = self;
    // mlx_distributed_group has no explicit free in this mlx-c version
}
```

频繁创建/销毁会产生资源泄漏，长期运行服务中风险累积。

## 8.4 `model_pool.zig` VTable 可选类型

```zig
pub const LoadedModel = struct {
    vtable: ?ModelVTable,  // 可选！
    // ...
};
```

`getOrLoad` 加载后 `vtable` 设为 null，`deinit` 中仅当 `vtable != null` 时释放资源——如果始终为 null，模型资源泄漏。
# 第九章 问题验证矩阵

## 9.1 与项目自我审计的交叉验证

`docs/deep-analysis.md` 是 v0.3.0 时的自我审计文档，`production-roadmap.md` 声称所有问题已修复。本分析逐条验证：

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

`production-roadmap.md` 声称所有 Phase 0–7 + Task 13–34 已完成。

| 维度 | 声称 | 实际 | 偏差 |
|------|------|------|------|
| 功能完成度 | 全部完成 | 基本属实 | 投机解码、引导解码、MoE、分层缓存均已落地 |
| 质量完成度 | 全部完成 | **部分 overclaim** | NN GPU 化、@constCast 清除未完全完成 |
| 测试完成度 | 350 测试通过 | 测试数量属实 | 结构性缺口：NN 层、Autograd、真实权重 |

**建议**：在路线图中将 `nn.zig` 的 GPU 化迁移从"已完成"调整为"部分完成（activations已完成，nn层待清理）"。
# 附录：关键文件与文档索引

## A.1 关键源文件索引

| 文件 | 行数 | 职责 | 风险等级 |
|------|------|------|---------|
| `src/models/deepseek_v4.zig` | 3,091 | DeepSeek V4 完整实现（MLA+MoE+YARN+mHC） | 中（2处@constCast） |
| `src/models/deepseek_v4_loader.zig` | 2,071 | V4 权重加载器（分片+反量化+专家策略） | 低 |
| `src/main.zig` | 1,764 | CLI 入口（chat/serve/benchmark/quantize/lora-train） | 低 |
| `src/server.zig` | 1,517 | HTTP 服务器（OpenAI兼容+SSE+工具调用） | 中（batch未集成） |
| `src/ops/nn.zig` | 1,354 | NN 层（Linear/BatchNorm/LSTM/GRU/RNN/Attention） | **高**（34处dataSliceMut） |
| `src/speculative.zig` | 1,223 | 投机解码双轨制（PLD+EAGLE） | 低 |
| `src/kvcache/paged.zig` | 1,152 | 分页 KV Cache（BlockManager+CoW+前缀哈希） | 低 |
| `src/guided.zig` | 1,129 | 引导解码 FSM（JSON Schema/Regex约束） | 低 |
| `src/io/safetensors_reader.zig` | 1,045 | Safetensors 随机访问读取器（pread零拷贝） | 低 |
| `src/quantize.zig` | 872 | 量化基础设施（affine/mxfp4/nvfp4/mxfp8/turboquant） | 低 |
| `src/qlora.zig` | 773 | QLoRA 量化+低秩适配器训练 | 低 |
| `src/tokenizer/bpe.zig` | 744 | BPE Tokenizer（HF tokenizer.json 格式） | 低 |
| `src/models/llama.zig` | 725 | LLaMA/Mistral/Qwen/Gemma/Phi 标准架构 | 低 |
| `src/models/minimax.zig` | 712 | MiniMax 模型适配 | 中（4处@constCast） |
| `src/prompt_cache.zig` | 563 | Prompt 缓存持久化（safetensors） | **高**（类型漏洞） |
| `src/distributed.zig` | 222 | 分布式推理（多Mac tensor parallelism） | 中（deinit为空） |
| `src/optim.zig` | 217 | AdamW 优化器 | 中（临时对象风暴） |
| `src/c.zig` | ~200 | C 绑定层（mlxErrorHandler+check+类型重导出） | 低 |

## A.2 文档目录索引

| 文档 | 类型 | 内容 |
|------|------|------|
| `docs/deep-analysis.md` | 审计报告 | v0.3.0 自我审计，P0-P3 问题清单 |
| `docs/production-roadmap.md` | 路线图 | Phase 0-7 进度追踪，Task 13-34 完成状态 |
| `.kiro/specs/production-deployment/design.md` | 设计文档 | 六层架构设计、Mermaid 图表 |
| `.kiro/specs/production-deployment/design-paged-kv-cache.md` | 设计文档 | PagedKVCache 算法细节（updateAndFetch六步法） |
| `.kiro/specs/production-deployment/design-server.md` | 设计文档 | 服务器架构、请求流、Scheduler 设计 |
| `docs/ecosystem-analysis.md` | 调研 | vLLM/mlx-lm/oMLX/TileKernels/mlx-rs 五项目对比 |
| `docs/BENCHMARK.md` | 性能 | 基准测试方法论和阈值定义 |
| `docs/DEEPSEEK-V4-FIX-PLAN.md` | 修复计划 | V4 问题诊断与修复方案 |
| `docs/FIX-REPORT-DEEPSEEK-V4.md` | 修复报告 | V4 修复验证结果 |
| `docs/tilekernels-analysis.md` | 调研 | TileKernels 算子融合与量化分析 |
| `docs/deepseek-v4-optimization-plan.md` | 优化 | V4 性能优化策略 |
| `docs/DEEPSEEK-V4-CHAT-ANALYSIS.md` | 分析 | V4 Chat 模式行为分析 |
| `docs/competitive-advantages.md` | 战略 | 项目竞争优势分析 |

## A.3 构建与依赖

### 外部依赖

- `mlx-c`：Apple MLX 的 C API 绑定
  - 探测优先级：`-Dmlx_prefix` > `MLX_C_PREFIX` env > `pkg-config --variable=prefix mlxc` > `/opt/homebrew`
- `zig_regex`：正则表达式库（Zig 包，固定 hash，非 main 分支）

### 构建产物

- `libmlx-zig.a`：静态库
- `mlx-zig`：CLI 工具
  - `chat`：交互式聊天
  - `serve`/`server`：HTTP 服务
  - `benchmark`：性能基准
  - `quantize`：权重量化
  - `lora-train`：LoRA 微调
  - `convert`：格式转换（TODO）
  - `evaluate`：困惑度评估
- `example`：示例程序
- `test`：测试 runner（50+ 模块，350 测试）

### 平台支持

| 平台 | 状态 | 说明 |
|------|------|------|
| macOS Apple Silicon | ✅ 主平台 | Metal GPU + UMA 统一内存 |
| macOS Intel | ⚠️ 可能工作 | 无 Metal，回退 CPU |

## A.4 代码风格（`.sisyphus/zig-style.md`）

- **命名**：PascalCase（类型）、camelCase（函数）、snake_case（常量）
- **禁止**：`// ====` 分隔符、`//!` 模块文档、`as any`、空 catch 块
- **要求**：`zig build` 通过、`zig build test` 通过、`zig fmt src/` 格式化
- **许可证**：AGPL-3.0

## A.5 版本信息

```zig
// src/root.zig
pub const version = "0.3.0-mlx-c";
```
