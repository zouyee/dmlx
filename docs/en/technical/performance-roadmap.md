# MLX-Zig 生产级部署路线图

> 基于 mlx-zig 代码库深度审计，以及 vLLM、mlx-lm、oMLX、TileKernels、mlx-rs 五个项目的
> 系统性分析，制定从当前状态到生产级部署的完整路线。

### 进度追踪（更新于 2026-04-26）

> **所有 Phase 0–5 + 集成 + 缺口修复 + Phase 7 (P0-P3) 任务已完成。** 350 个测试全部通过。

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 0: 地基修复 | ✅ 全部完成 | 错误处理、内存安全（ScopedArrayArena）、NN 层 GPU 化、构建系统 pkg-config |
| Phase 1: 推理引擎 | ✅ 全部完成 | 三层生成 API、模型注册表（5 架构）、Prompt Cache 持久化、算子融合 |
| Phase 2: 服务层 | ✅ 全部完成 | Scheduler、PagedAttention（CoW + prefix hash）、KV 量化 4/8/16-bit、SSE streaming、Continuous Batching |
| Phase 3: 高级推理 | ✅ 全部完成 | Chunked Prefill、Prefix Caching、Speculative Decoding（n-gram）、Guided Decoding（FSM） |
| Phase 4: 量化与训练 | ✅ 全部完成 | 权重量化（affine + MXFP4）、QLoRA、MoE Router、FP8/FP4 绑定 |
| Phase 5: 生产运维 | ✅ 全部完成 | ModelPool LRU、Tiered KV Cache（RAM+SSD）、内存限制、autoMaxKvSize、Benchmark |
| 集成接线 (Task 13) | ✅ 全部完成 | 11 个子任务：KV 策略选择、Scheduler、Generation API、Prompt Cache、Speculative/Guided Decoding、ModelPool、Tiered Cache、CLI 子命令、算子融合、MoE Router |
| V4 缺口 (Task 15) | ✅ 全部完成 | Paged+Quantized 组合、learned pooling、FP8 原生 KV 存储、Lightning Indexer、Attention Sink、异构 KV Cache、On-disk prefix reuse、TurboQuant |
| 集成修复 (Task 17) | ✅ 全部完成 | Prefix hash 自动注册、Prompt Cache save 接入、PrefixDiskCache 接入 server |
| 并发服务 (Task 19) | ✅ 全部完成 | io.async 并发连接、Request 完成通知、Engine loop 后台 fiber、Scheduler 驱动 |
| 量化模型加载 (Task 21) | ✅ 全部完成 | 量化权重检测、quantizedMatmul forward、多分片加载、lm_head 量化、端到端验证 |
| 量化 mode 完整对齐 (Task 23) | ✅ 全部完成 | nvfp4/mxfp8 枚举、LightningIndexer 升级为真正 MXFP4 量化、341 测试通过 |
| 内存泄漏修复 + 混合量化 (Task 25) | ✅ 全部完成 | QuantizedWeight shape 泄漏修复、V4 per-weight 量化 config 解析 |
| 量化模型加载 (Task 21) | ✅ 全部完成 | mlx-lm 量化格式加载、quantizedMatmul 前向路径、多分片 safetensors 合并 |
| Phase 7 P0: 核心可用性 (Task 27) | ✅ 全部完成 | Streaming token 输出、EOS 停止、Chat template、generate 性能修复 |
| Phase 7 P1: 模型扩展 (Task 28-30) | ✅ 全部完成 | Gemma/Phi-3 loader、batch forward、token decode 集成 |
| Phase 7 P2: API 完整性 (Task 31-33) | ✅ 全部完成 | stop/logprobs/tool_calls、repetition penalty、min_p、SSE keep-alive、EAGLE |
| Phase 7 P3: 生态工具 (Task 34) | ✅ 全部完成 | Custom Metal kernel、分布式推理、模型转换、perplexity 评估 |

>
> 参考文档：`deep-analysis.md`（代码审计）、`ecosystem-analysis.md`（mlx-lm/oMLX/mlx-rs）、
> `tilekernels-analysis.md`（TileKernels）、`deepseek-v4-turboquant-analysis.md`（V4+TurboQuant 论文分析）。

---

## 一、mlx-zig 当前状态总览（更新于 2026-04-26）

### 已有能力

- 200+ mlx-c 算子绑定（math/shape/reduce/linalg/fft/conv/random/creation）
- `fast.zig` 绑定了 MLX 融合 kernel（rms_norm/rope/sdpa/layer_norm）
- LLaMA + DeepSeek V4 两种模型架构 + 模型注册表（9 架构：LLaMA/Mistral/Qwen2/Qwen3/Gemma/GLM-4/Phi/Phi-3/DeepSeek V4）
- KV Cache 6 种策略（Standard/Rotating/Quantized/Paged/PagedQuantized/Tiered）+ TurboQuant
- 量化：affine 4/8-bit + MXFP4 + NVFP4 + MXFP8 + FP8 (E4M3) + TurboQuant（Lloyd-Max + QJL）
- 量化模型加载：直接加载 mlx-lm 4-bit/8-bit 量化模型（packed uint32 + scales + biases）
- mlx-lm 量化模型加载：自动检测 .scales/.biases 后缀，quantizedMatmul 前向路径，多分片 safetensors 合并
- LoRA/QLoRA 适配器 + AdamW 优化器 + SFT Trainer
- BPE Tokenizer + HuggingFace config 解析 + Safetensors I/O
- CLI（chat/serve/benchmark/quantize）+ OpenAI 兼容 HTTP 服务器（SSE streaming）
- Scheduler + Continuous Batching + Chunked Prefill
- Speculative Decoding（n-gram）+ Guided Decoding（FSM）
- ModelPool（LRU 淘汰）+ 进程内存限制 + autoMaxKvSize
- Tiered KV Cache（Hot RAM + Cold SSD）+ PrefixDiskCache（on-disk prefix reuse）
- Prompt Cache 持久化（save/load safetensors）
- DeepSeek V4 完整支持：CSA/HCA 压缩、Lightning Indexer、Attention Sink、FP8 KV 存储、mHC

### 已解决的核心缺陷

| 类别 | 原问题 | 解决方案 | 状态 |
|------|--------|---------|------|
| 性能 | NN 层绕过 GPU | 全部改用 mlx-c 算子链 + fast.zig 融合 kernel | ✅ |
| 内存 | 中间 Array 泄漏 | ScopedArrayArena 批量释放 | ✅ |
| 错误 | 无上下文错误信息 | `mlxErrorHandler` 捕获 C++ 异常文本 | ✅ |
| 服务 | 单线程阻塞 | Scheduler + Continuous Batching + SSE streaming | ✅ |
| 安全 | `@constCast` 绕过 CoW | 全部改用 mlx-c 算子链 | ✅ |
| 测试 | 零覆盖 | 350 个测试（含 21 个 property tests） | ✅ |
| 构建 | 硬编码路径 | pkg-config + `-Dmlx_prefix` + 固定依赖版本 | ✅ |

---

## 二、外部项目关键架构提炼

> **注意**：以下"mlx-zig 对应"描述的是实现前的原始状态。所有提到的缺失功能
> 已在 Phase 0-5 + Task 13-22 中全部实现。当前状态见上方进度追踪表。

### vLLM — 工业级推理引擎的标杆

vLLM 是当前最成熟的 LLM 推理引擎，其架构定义了生产级推理的标准。
以下是 mlx-zig 必须理解和借鉴的核心机制：

#### PagedAttention（分页注意力）

vLLM 的核心创新。KV cache 不再为每个请求分配连续内存，
而是分成固定大小的 block（默认 16 tokens/block），按需从 `free_block_queue` 分配。

```
block_size = 2 * block_size_tokens * num_kv_heads * head_size * dtype_bytes
```

- 内存浪费从 60-80% 降到 < 4%
- 支持 Copy-on-Write（prefix sharing 时多个请求共享 block）
- block 回收后可被新请求复用

**mlx-zig 对应**：`kvcache/paged.zig` 是骨架，需要实现完整的 block 分配/回收/CoW。

#### Continuous Batching（连续批处理）

所有序列被展平拼接为一个"超级序列"，通过 position indices 和 attention masks
确保每个序列只 attend 自己的 tokens。新请求可以在任意 engine step 加入。

```
[seq1_tok1, seq1_tok2, seq2_tok1, seq3_tok1, seq3_tok2, seq3_tok3]
 ← 3 个不同长度的序列拼接在一起 →
```

**mlx-zig 对应**：`server.zig` 串行处理请求，需要实现 batch scheduler。

#### Scheduler 三阶段循环

每个 engine step：
1. **Schedule** — 从 waiting/running 队列选择请求，分配 KV cache blocks
2. **Forward** — 执行模型前向传播 + 采样
3. **Postprocess** — 追加 token、检查停止条件、释放已完成请求的 blocks

**mlx-zig 对应**：当前 `generate` 是单请求循环，需要重构为 scheduler 驱动。

#### Chunked Prefill（分块预填充）

长 prompt 的 prefill 分成多个 chunk（如每次 8 tokens），
避免单个长请求独占 engine step，降低其他请求的延迟。

**mlx-zig 对应**：当前无此机制。

#### Prefix Caching（前缀缓存）

将 prompt tokens 按 block_size 分块，每块计算 hash。
相同前缀的请求可以复用已计算的 KV cache blocks，跳过重复 prefill。

```
hash(block) = hash(prev_block_hash, token_ids, metadata)
```

**mlx-zig 对应**：`kvcache/radix.zig` 有骨架，但未实现 hash-based 查找。

#### Speculative Decoding（推测解码）

用小模型（或 n-gram）提议 k 个 token，大模型一次验证，
accept/reject 保证统计等价于逐 token 采样。

vLLM V1 支持：n-gram、EAGLE、Medusa 三种 draft 方案。

**mlx-zig 对应**：完全缺失。

#### Guided Decoding（约束解码）

通过 FSM（有限状态机）在每步 mask logits，
确保输出符合指定语法（JSON schema、regex 等）。

**mlx-zig 对应**：完全缺失。

### mlx-lm — MLX 生态的参考实现

- **50+ 模型架构**注册表 vs mlx-zig 的 2 个
- **三层生成架构**：`generate_step` → `stream_generate` → `generate`
- **Prompt cache 持久化**到 safetensors
- **量化体系**：GPTQ / AWQ / DWQ
- **完整 CLI**：generate / chat / convert / evaluate / benchmark / manage / share

### oMLX — Apple Silicon 生产级服务器

- **分层 KV Cache**：Hot RAM + Cold SSD（safetensors 格式 offload）
- **Continuous Batching**：基于 mlx-lm BatchGenerator
- **多模型管理**：EnginePool + LRU 淘汰 + TTL + 模型固定
- **进程内存限制**：`--max-process-memory`
- **Claude Code 优化**：context scaling + SSE keep-alive

### TileKernels — 算子融合与量化

- **极致算子融合**：SwiGLU + 量化单 kernel
- **完整量化栈**：FP8/FP4/E5M6，per-token/per-block/per-channel
- **MoE 路由管线**：topk → expand → compute → reduce
- **参考实现 + 数值验证**的工程纪律

### mlx-rs — 同类项目的经验

- **自动微分显式输入规则**的文档化
- **Feature flags** 控制 Metal/Accelerate 后端

---

## 三、生产级部署路线

### Phase 0：地基修复（阻塞一切后续工作）

> 不修复这些问题，任何功能开发都是在沙子上建楼。

#### 0.1 错误处理（1 天）

```zig
// c.zig — 接入 mlx_get_last_error
pub fn check(rc: c_int) !void {
    if (rc != 0) {
        const msg = c.mlx_get_last_error();
        std.log.err("MLX error: {s}", .{std.mem.span(msg)});
        return error.MlxError;
    }
}
```

#### 0.2 内存安全（1 周）

- 为 forward pass 引入 ArenaAllocator，批量释放中间 Array
- 移除 `dataSliceMut` 的 `@constCast`，所有数据操作改用 mlx-c 算子
- 为 EagerContext 添加 `deinit`

#### 0.3 NN 层 GPU 化（1 周）

这是性能提升 100-1000x 的关键步骤：

| 当前实现 | 改为 |
|----------|------|
| `nn.RMSNorm.forward` — CPU 标量循环 | `fast.rmsNorm()` |
| `nn.RoPE.apply` — CPU 标量循环 | `fast.rope()` |
| `nn.MultiHeadAttention` — CPU 标量循环 | `fast.scaledDotProductAttention()` |
| `nn.Embedding.forward` — CPU 标量循环 | `mlx_take` |
| `activations.gelu` 等 21 个 — CPU 标量循环 | mlx-c 算子组合 |
| `loss.mseLoss` 等 9 个 — CPU 标量循环 | 参照 `crossEntropyGraph` 图模式 |
| `nn.LSTM/GRU/RNN` — CPU 标量循环 | mlx-c matmul + sigmoid + tanh 算子链 |

#### 0.4 构建系统（2 天）

- 消除硬编码 `/opt/homebrew` 路径，改用 pkg-config + `-Dmlx_prefix`
- zig-regex 依赖固定版本
- build.zig 去重

### Phase 1：推理引擎重构（参照 vLLM + mlx-lm）

> 目标：从"能跑"到"跑得好"。

#### 1.1 三层生成架构（参照 mlx-lm）

```zig
// 单步生成 — 核心原语
pub fn generateStep(model: *Model, tokens: Array, cache: []KVCache) !Token { ... }

// 流式生成 — 每个 token 回调
pub fn streamGenerate(model: *Model, prompt: []u32, config: GenConfig,
    callback: fn(Token) void) !void { ... }

// 完整生成 — 返回全部 tokens
pub fn generate(model: *Model, prompt: []u32, config: GenConfig) ![]u32 { ... }
```

#### 1.2 模型架构注册表（参照 mlx-lm）

```zig
pub const ModelVTable = struct {
    forward: *const fn(...) !Array,
    loadWeights: *const fn(...) !void,
    // ...
};

pub const model_registry = std.StaticStringMap(ModelVTable).initComptime(.{
    .{ "llama", llama_vtable },
    .{ "deepseek_v4", deepseek_v4_vtable },
    .{ "mistral", mistral_vtable },
    .{ "qwen2", qwen2_vtable },
});
```

优先添加：Mistral、Qwen2/3、Gemma（覆盖 90% 使用场景）。

#### 1.3 Prompt Cache 持久化（参照 mlx-lm + oMLX）

利用已有的 safetensors I/O，将 KV cache 序列化到磁盘：

```zig
pub fn savePromptCache(cache: []KVCache, path: []const u8) !void { ... }
pub fn loadPromptCache(path: []const u8) ![]KVCache { ... }
```

#### 1.4 算子融合（参照 TileKernels + MLX compile）

```zig
// 将 SwiGLU MLP 编译为融合操作
const swiglu_fn = try Closure.init(swiGluForward, allocator);
const compiled_swiglu = try compile.compile(swiglu_fn, false);

// 将 AdamW.step 编译为融合操作（消除每步 ~15 个临时 Array）
const adamw_fn = try Closure.init(adamwStep, allocator);
const compiled_adamw = try compile.compile(adamw_fn, false);
```

### Phase 2：服务层重构（参照 vLLM + oMLX）

> 目标：从"能响应"到"能服务"。

#### 2.1 Scheduler（参照 vLLM）

实现 vLLM 的三阶段 engine step：

```zig
pub const Scheduler = struct {
    waiting: Queue(Request),
    running: Queue(Request),
    kv_cache_manager: KVCacheManager,

    pub fn schedule(self: *Scheduler) ScheduleResult {
        // 1. 优先处理 running 队列（decode 请求）
        // 2. 处理 waiting 队列（prefill 请求）
        // 3. 分配 KV cache blocks
    }
};

pub fn engineStep(scheduler: *Scheduler, model: *Model) ![]Output {
    const scheduled = scheduler.schedule();
    const outputs = try model.forward(scheduled.batch);
    return try postprocess(outputs, scheduler);
}
```

#### 2.2 PagedAttention 完整实现（参照 vLLM）

```zig
pub const BlockManager = struct {
    free_blocks: DoublyLinkedList(Block),
    req_to_blocks: HashMap(RequestId, []Block),
    cached_block_hash: HashMap(BlockHash, *Block),

    pub fn allocateSlots(self: *BlockManager, req: *Request, num_tokens: usize) !void { ... }
    pub fn freeBlocks(self: *BlockManager, req_id: RequestId) void { ... }
};
```

#### 2.5 KV Cache 量化（kv_bits=4/8）（P1，来自 Apple Silicon 优化分析）

Apple Silicon M3/M4 对小位宽运算有专门优化，KV Cache 量化是长上下文推理的硬性前提。
8B 模型 32K context 下 float16 KV Cache 约 8GB，4-bit 量化后降到 2GB。

```zig
pub const KVQuantConfig = struct {
    kv_bits: u8 = 16,          // 4, 8, 或 16（不量化）
    group_size: i32 = 64,      // 量化分组大小
};

// 在 KV Cache update 路径中插入量化
pub fn updateQuantized(self: *KVCache, key: Array, value: Array, config: KVQuantConfig) !void {
    if (config.kv_bits < 16) {
        const q_key = try quantize(ctx, key, config.kv_bits, config.group_size);
        const q_value = try quantize(ctx, value, config.kv_bits, config.group_size);
        // 存储量化后的 KV
    }
    // Attention 计算前反量化回来
}
```

与 PagedAttention 天然配合：block 粒度的量化可以在 block 分配时决定精度。
利用 mlx-c 已有的 `mlx_quantize`/`mlx_dequantize` 实现。

#### 2.3 SSE Streaming

```zig
fn writeSSEEvent(conn: Connection, data: []const u8) !void {
    try conn.write("data: ");
    try conn.write(data);
    try conn.write("\n\n");
    try conn.flush();
}
```

#### 2.4 Continuous Batching

所有序列展平拼接，通过 position indices 区分：

```zig
// 3 个请求：长度分别为 2, 1, 3
// 拼接为：[tok1_1, tok1_2, tok2_1, tok3_1, tok3_2, tok3_3]
// positions: [0, 1, 0, 0, 1, 2]
```

### Phase 3：高级推理特性（参照 vLLM）

#### 3.1 Chunked Prefill

长 prompt 分块处理，每步最多处理 `max_prefill_tokens` 个 token：

```zig
if (request.remaining_prefill_tokens > max_prefill_tokens) {
    request.chunk_size = max_prefill_tokens;
} else {
    request.chunk_size = request.remaining_prefill_tokens;
}
```

#### 3.2 Prefix Caching

```zig
pub fn hashBlock(prev_hash: u64, token_ids: []const u32) u64 {
    var hasher = std.hash.Wyhash.init(prev_hash);
    hasher.update(std.mem.sliceAsBytes(token_ids));
    return hasher.final();
}
```

#### 3.3 Speculative Decoding

优先实现 n-gram 方案（最简单，无需额外模型）：

```zig
pub fn ngramPropose(context: []const u32, k: usize) ?[]const u32 {
    // 在已生成的 context 中查找最后 n 个 token 的匹配
    // 返回匹配位置之后的 k 个 token 作为 draft
}
```

#### 3.4 Guided Decoding

通过 logits mask 约束输出格式：

```zig
pub fn applyGrammarMask(logits: Array, allowed_tokens: []const u32) !Array {
    // 将不允许的 token 的 logit 设为 -inf
}
```

### Phase 4：量化与训练（参照 TileKernels + mlx-lm）

#### 4.1 量化基础设施

```zig
pub const QuantConfig = struct {
    format: enum { int4_nf4, int8, fp8_e4m3 },
    group_size: i32 = 64,
    bits: u8 = 4,
};

// 绑定 mlx-c
pub fn quantize(ctx: EagerContext, weight: Array, config: QuantConfig) !struct { quantized: Array, scales: Array } { ... }
pub fn dequantize(ctx: EagerContext, quantized: Array, scales: Array, config: QuantConfig) !Array { ... }
```

#### 4.2 QLoRA

- 4-bit NormalFloat 量化（NF4）
- 双量化（Double Quantization）

#### 4.3 MoE 路由（参照 TileKernels）

```zig
pub fn moeRoute(ctx: EagerContext, scores: Array, num_topk: i32) !MoeRouteResult { ... }
pub fn moeExpand(ctx: EagerContext, x: Array, mapping: MoeRouteResult) !Array { ... }
pub fn moeReduce(ctx: EagerContext, expanded: Array, weights: Array, mapping: MoeRouteResult) !Array { ... }
```

### Phase 5：生产级运维（参照 oMLX）

#### 5.1 多模型管理

```zig
pub const ModelPool = struct {
    models: HashMap([]const u8, *LoadedModel),
    lru: LRUCache([]const u8),
    max_memory: usize,

    pub fn getOrLoad(self: *ModelPool, name: []const u8) !*LoadedModel { ... }
    pub fn evictLRU(self: *ModelPool) void { ... }
};
```

#### 5.2 分层 KV Cache（参照 oMLX）

```
Hot Tier (RAM) ←→ Cold Tier (SSD, safetensors)
    ↑ write-back        ↑ restore on cache hit
```

#### 5.3 进程内存限制

```zig
pub fn enforceMemoryLimit(max_bytes: usize) void {
    const current = getProcessMemoryUsage();
    if (current > max_bytes) {
        // 触发 LRU 淘汰
    }
}
```

#### 5.4 max_kv_size 自动配置（P2，来自 Apple Silicon 优化分析）

根据设备内存自动推荐 KV Cache 上限，避免用户手动调参导致 OOM 或浪费。

```zig
pub fn autoMaxKvSize(model_bytes: usize, kv_bits: u8) usize {
    const total_ram = getSystemMemoryBytes();
    const available = total_ram - model_bytes - safety_margin;
    const bytes_per_token_kv = 2 * num_kv_heads * head_dim * (kv_bits / 8) * num_layers;
    return available / bytes_per_token_kv;
}
```

| 设备内存 | kv_bits=4 建议 max_kv_size | 典型场景 |
|---------|--------------------------|---------|
| 16 GB | 8,192 – 16,384 | 基础对话 |
| 64 GB | 32,768 – 65,536 | 代码编写、文档分析 |
| 128 GB+ | 131,072+ | RAG、超长上下文 |

CLI 暴露为 `--max-kv-size auto`（默认）或 `--max-kv-size 32768`（手动指定）。

#### 5.4 Benchmark 工具（参照 vLLM + mlx-lm）

```bash
mlx-zig benchmark --model <path> --input-tokens 32 --output-tokens 128
# 输出：TTFT, ITL, throughput, memory usage
```

---

## 四、测试与验证体系（参照 TileKernels + vLLM）

### 数值验证

```zig
// 参照 TileKernels testing/numeric.py
fn calcDiff(x: []const f32, y: []const f32) f64 {
    // 余弦相似度：1 - 2*dot(x,y) / (dot(x,x) + dot(y,y))
}

fn checkBias(x: []const f32, ref: []const f32) !void {
    // 统计检验：量化误差是否无偏
}
```

### Golden Test

1. Python MLX 生成参考输出（每个 NN 层、每个 activation、每个 loss）
2. Zig 测试加载参考数据，对比 mlx-zig 输出
3. CI 自动运行

### 端到端验证

- TinyLlama 1.1B 推理输出对比 Python MLX
- Perplexity 评估（参照 mlx-lm evaluate.py）

---

## 五、版本发布计划（更新于 2026-04-26）

| 版本 | 阶段 | 关键交付 | 状态 |
|------|------|----------|------|
| v0.4.1 | Phase 0 | 错误处理 + 内存安全 + NN 层 GPU 化 + 构建修复 | ✅ 已完成 |
| v0.5.0 | Phase 1 | 三层生成 + 模型注册表 + Prompt cache + 算子融合 | ✅ 已完成 |
| v0.6.0 | Phase 2 | Scheduler + PagedAttention + SSE + Continuous Batching | ✅ 已完成 |
| v0.7.0 | Phase 3 | Chunked prefill + Prefix caching + Speculative decoding | ✅ 已完成 |
| v0.8.0 | Phase 4 | 量化 + QLoRA + MoE 路由 | ✅ 已完成 |
| v0.9.0 | Phase 5 | 多模型管理 + 分层 KV Cache + 内存限制 + Benchmark | ✅ 已完成 |
| v1.0.0 | GA | 生产级部署：完整 CLI + 文档 + CI + 性能验证 | ✅ 已完成（350 测试通过） |

---

## 六、架构演进

### 当前（v1.0 — 已实现）
```
┌──────────────────────────────────────────────────┐
│  CLI (chat/serve/benchmark/quantize/lora-train)  │
│  HTTP Server (SSE streaming, OpenAI compat)      │
├──────────────────────────────────────────────────┤
│  Scheduler (continuous batching, chunked PF)     │
│  Request Queue → Schedule → Forward → Post       │
├──────────────────────────────────────────────────┤
│  ModelPool (多模型 LRU, 内存限制)                  │
│  ModelRegistry (9 架构: LLaMA/DSV4/Mistral/      │
│       Qwen2/Qwen3/Gemma/GLM-4/Phi/Phi-3)        │
├──────────────────────────────────────────────────┤
│  Speculative Decoding (n-gram)                   │
│  Guided Decoding (FSM logits mask)               │
│  Prefix Caching (hash-based block reuse)         │
├──────────────────────────────────────────────────┤
│  Quantization (affine 4/8-bit, MXFP4, FP8)      │
│  TurboQuant (Lloyd-Max + QJL, optional)          │
│  MoE Router (topk → expand → reduce)             │
│  QLoRA fine-tuning                               │
├──────────────────────────────────────────────────┤
│  NN Layer (全部走 mlx-c 算子图)                   │
│  Compiled Fused Ops (SwiGLU, AdamW)              │
│  fast.zig (rms_norm / rope / sdpa)               │
├──────────────────────────────────────────────────┤
│  PagedAttention (block alloc/free/CoW/prefix)    │
│  Paged+Quantized (block-level 4-bit)             │
│  Tiered KV Cache (Hot RAM + Cold SSD)            │
│  PrefixDiskCache (on-disk shared-prefix reuse)   │
├──────────────────────────────────────────────────┤
│  DeepSeek V4: CSA/HCA compression, Lightning     │
│  Indexer, Attention Sink, FP8 KV, mHC            │
├──────────────────────────────────────────────────┤
│  mlx-c 0.6.0 → Metal GPU / Accelerate CPU       │
└──────────────────────────────────────────────────┘
```

---
├──────────────────────────────────────────────┤
│  mlx-c → Metal GPU / Accelerate CPU          │
---

## 七、借鉴来源速查

| 特性 | vLLM | mlx-lm | oMLX | TileKernels | mlx-rs |
|------|------|--------|------|-------------|--------|
| PagedAttention | ✅ 原创 | — | ✅ 借鉴 | — | — |
| Continuous Batching | ✅ 核心 | ✅ BatchGen | ✅ 借鉴 | — | — |
| Chunked Prefill | ✅ | — | — | — | — |
| Prefix Caching | ✅ hash-based | ✅ safetensors | ✅ + SSD | — | — |
| Speculative Decoding | ✅ ngram/EAGLE/Medusa | — | — | — | — |
| Guided Decoding | ✅ FSM/xgrammar | — | — | — | — |
| Scheduler 三阶段 | ✅ 核心 | — | FCFS | — | — |
| 模型注册表 | ✅ | ✅ 50+ | 继承 mlx-lm | — | — |
| 三层生成架构 | — | ✅ | — | — | — |
| 量化 GPTQ/AWQ | ✅ | ✅ | 继承 | FP8/FP4 | — |
| 算子融合 | CUDA graph | 自动 | 继承 | 手写 kernel | — |
| 分层 KV Cache | — | — | ✅ Hot+Cold | — | — |
| 多模型管理 | — | — | ✅ LRU+TTL | — | — |
| MoE 路由 | ✅ EP | — | — | ✅ 完整管线 | — |
| 数值验证框架 | — | — | — | ✅ | — |
| 自动微分文档 | — | — | — | — | ✅ |

---

## 八、KV Cache 架构总览（Apple Silicon 深度优化）

> 基于 Apple Silicon UMA 特性、mlx-lm 最佳实践、DeepSeek V4 技术报告、
> TurboQuant 论文（arXiv:2504.19874）的综合分析，定义 mlx-zig KV Cache 的最终架构。

### 8.1 最终架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLI / HTTP Server                            │
│  --kv-bits 4|8|16    --max-kv-size auto|<int>    --kv-tier ram|ssd │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐   │
│  │   通用模型路径        │    │   DeepSeek V4 专用路径            │   │
│  │  (LLaMA/Mistral/    │    │                                  │   │
│  │   Qwen/Gemma)       │    │  CSA 4x 压缩 (learned pooling)  │   │
│  │                     │    │  HCA 128x 压缩 (dense attn)     │   │
│  │  ┌───────────────┐  │    │  MLA 低秩投影 (q/o_lora_rank)   │   │
│  │  │ PagedKVCache  │  │    │  FP8 KV 存储 + FP4 indexer     │   │
│  │  │  + 内置量化    │  │    │  滑动窗口 128 token             │   │
│  │  │  (kv_bits)    │  │    │  Attention Sink                 │   │
│  │  └───────┬───────┘  │    └──────────────┬───────────────────┘   │
│  │          │          │                   │                       │
│  │  ┌───────▼───────┐  │    ┌──────────────▼───────────────────┐   │
│  │  │ Block Manager │  │    │  mHC 残差流 (hc_mult=4)          │   │
│  │  │  alloc/free   │  │    │  Sinkhorn 归一化 (Birkhoff)      │   │
│  │  │  CoW          │  │    │  MoE 路由 (256 experts, top-6)  │   │
│  │  │  prefix hash  │  │    └──────────────────────────────────┘   │
│  │  └───────┬───────┘  │                                           │
│  └──────────┼──────────┘                                           │
│             │                                                       │
│  ┌──────────▼──────────────────────────────────────────────────┐   │
│  │              Tiered Cache (Hot RAM + Cold SSD)               │   │
│  │                                                              │   │
│  │  Hot Tier (RAM)              Cold Tier (SSD)                 │   │
│  │  ┌──────────────────┐       ┌──────────────────┐            │   │
│  │  │ Active blocks    │ ───── │ safetensors files │            │   │
│  │  │ LRU eviction     │ evict │ restore on hit    │            │   │
│  │  └──────────────────┘ ◄──── └──────────────────┘            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Memory Manager                                  │   │
│  │  autoMaxKvSize(device_ram, model_bytes, kv_bits)            │   │
│  │  enforceMemoryLimit(max_process_memory)                     │   │
│  │  sysctl hw.memsize → 动态计算                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Apple Silicon UMA (Zero-copy)                   │   │
│  │  mlx_slice_update (lazy eval, MLX 内部优化内存复用)           │   │
│  │  CPU/GPU 共享内存 — 无 PCIe 搬运开销                         │   │
│  │  Metal GPU 原生 4-bit/8-bit 算力                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 五大设计要点 → 实施对照表

| # | Apple Silicon 设计要点 | 代码位置 | Spec Requirement | Spec Task | 状态 |
|---|----------------------|---------|-----------------|-----------|------|
| 1 | **Zero-copy UMA** — `mlx_slice_update` 原位更新，禁止 `@constCast` | `kvcache/standard.zig` `sliceUpdateKV` | R2.3 | Task 1.2 | ✅ 已完成 |
| 2a | **kv_bits 量化** — 4-bit/8-bit KV cache 量化 | `kvcache/quantized.zig` | R11.1–R11.4 | Task 5.6, 5.7 | ✅ 已完成 |
| 2b | **`--kv-bits` CLI 参数** — serve 命令暴露量化选项 | `main.zig` `--kv-bits 4\|8\|16` | R11.4 | Task 13.1 | ✅ 已完成 |
| 3 | **RAM+SSD 二级缓存** — Hot/Cold 分层，safetensors offload | `kvcache/tiered.zig` | R22.1–R22.4 | Task 11.3, 11.4 | ✅ 已完成 |
| 4a | **Paged Attention** — block 分配/回收/CoW/prefix sharing | `kvcache/paged.zig` | R10.1–R10.5 | Task 5.3–5.5 | ✅ 已完成 |
| 4b | **Paged + Quantized 组合** — block 内部量化存储 | `kvcache/paged.zig` `kv_bits` 参数 | R10.6 | Task 15.1 | ✅ 已完成 |
| 5 | **max_kv_size 自动配置** — 根据设备内存动态计算 | `memory.zig` `autoMaxKvSize` | R24.1–R24.4 | Task 11.6, 11.7 | ✅ 已完成 |

### 8.3 已识别缺口及补充方案

#### 缺口 1：`--kv-bits` CLI 参数

**问题**：`QuantizedKVCache` 已实现 4-bit/8-bit 量化，但用户无法通过 CLI 或 server config 开启。
R11 说"接受 kv_bits 配置参数"，但没有对应的 CLI requirement。

**补充方案**：
- `serve` 命令增加 `--kv-bits 4|8|16` 参数（默认 4，Apple Silicon 最佳实践）
- `chat` 命令同步增加
- server.zig 启动时根据 `kv_bits` 选择 `createQuantized4Bit` / `createQuantized8Bit` / `createStandard`

**对应 Spec 变更**：
- R12（SSE Streaming）或新增 R12.5：serve 命令 SHALL 接受 `--kv-bits` 参数
- Task 13.1（最终集成）增加 `--kv-bits` 接入

#### 缺口 2：Paged + Quantized 组合策略

**问题**：`PagedKVCache` 和 `QuantizedKVCache` 是两个平级的 `KVCacheStrategy`，
不能同时使用。生产环境需要"分页管理内存 + 量化压缩数据"的组合。

**补充方案**（两选一，推荐方案 A）：

**方案 A：PagedKVCache 内置量化选项**
```zig
pub const PagedKVCache = struct {
    // ... 现有字段 ...
    kv_bits: u8 = 16,        // 新增：4, 8, 或 16
    group_size: i32 = 64,    // 新增

    fn allocPage(self: *PagedKVCache, stream: c.c.mlx_stream) !usize {
        // 分配时根据 kv_bits 决定存储精度
        if (self.kv_bits < 16) {
            // 存储 (packed, scales, biases) 而非原始 array
        }
    }
};
```

**方案 B：组合策略包装器**
```zig
pub const PagedQuantizedKVCache = struct {
    paged: PagedKVCache,       // 内存管理
    kv_bits: u8,               // 量化精度
    // 实现 KVCacheStrategy VTable
    // updateAndFetch: 量化后写入 paged block，读取时反量化
};
```

**对应 Spec 变更**：
- 新增 R10.6：PagedKVCache SHALL 支持可选的 kv_bits 参数，在 block 级别量化存储
- Task 5.3 增加子任务：PagedKVCache 构造函数接受 kv_bits 参数

### 8.4 通用模型 vs DeepSeek V4：KV Cache 路径分叉

两类模型的 KV cache 压缩策略完全不同，不应混用：

```
                    ┌─────────────────┐
                    │  Model Registry │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐       ┌───────────▼───────────┐
    │   通用模型          │       │   DeepSeek V4          │
    │   (LLaMA/Mistral/  │       │                       │
    │    Qwen/Gemma)     │       │                       │
    └─────────┬─────────┘       └───────────┬───────────┘
              │                             │
    ┌─────────▼─────────┐       ┌───────────▼───────────┐
    │  Post-hoc 量化     │       │  架构内置压缩          │
    │                   │       │                       │
    │  PagedKVCache     │       │  CSA 4x + HCA 128x   │
    │  + kv_bits=4/8    │       │  + MLA 低秩投影       │
    │  + CoW            │       │  + FP8 dtype cast     │
    │  + prefix sharing │       │  + FP4 indexer        │
    │                   │       │  + 滑动窗口 128       │
    │  可选：            │       │                       │
    │  TurboQuant       │       │  不需要额外量化        │
    │  (理论最优,        │       │  KV cache 已是 V3.2   │
    │   3.5-bit 无损)   │       │  的 2%                │
    └─────────┬─────────┘       └───────────┬───────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Tiered Cache   │
                    │  (Hot RAM +     │
                    │   Cold SSD)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Memory Manager │
                    │  autoMaxKvSize  │
                    └─────────────────┘
```

**关键决策**：
- 通用模型默认 `kv_bits=4`（Apple Silicon 最佳实践，M4 Pro 上 TPS 不降反升）
- DeepSeek V4 不走通用量化路径，其 forward path 内部直接做 FP8/FP4 dtype cast
- Tiered Cache 和 Memory Manager 对两条路径通用
- TurboQuant 作为通用模型的可选高级量化方案（Phase 4+），不适用于 V4

### 8.5 Apple Silicon 设备内存 → KV Cache 推荐配置

| 设备 | 内存 | kv_bits=4 max_kv_size | kv_bits=8 max_kv_size | 典型场景 |
|------|------|----------------------|----------------------|---------|
| M3 MacBook Air | 16 GB | 8,192 – 16,384 | 4,096 – 8,192 | 基础对话 |
| M4 Pro Mac Mini | 48 GB | 24,576 – 49,152 | 12,288 – 24,576 | 代码编写 |
| M4 Max MacBook Pro | 64 GB | 32,768 – 65,536 | 16,384 – 32,768 | 文档分析 |
| M4 Ultra Mac Studio | 128 GB | 65,536 – 131,072 | 32,768 – 65,536 | RAG、超长上下文 |
| M4 Ultra Mac Pro | 192 GB+ | 131,072+ | 65,536+ | 多模型并发 |

> 公式：`max_kv_size = (device_ram - model_bytes - 512MB) / (2 × num_kv_heads × head_dim × (kv_bits/8) × num_layers)`
>
> 以 Llama-3-8B (GQA-8, head_dim=128, 32 layers) + kv_bits=4 为例：
> `bytes_per_token = 2 × 8 × 128 × 0.5 × 32 = 32,768 bytes ≈ 32KB/token`
> 64GB 设备，模型占 ~4GB → 可用 ~59GB → max_kv_size ≈ 59GB / 32KB ≈ 1,900,000 tokens
