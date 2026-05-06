# dmlx 完整代码审查报告

**审查日期:** 2026-05-01  
**审查范围:** `src/` 下 130+ 个 Zig 文件（~52,572 行），`build.zig`，`tests/`，`docs/`  
**Zig 版本:** 0.16.0（`build.zig.zon` 声明）  
**目标平台:** macOS Apple Silicon（Metal）  
**审查轮次:** 6 轮独立维度审视  
**文档状态:** 经源码交叉验证

---

## 执行摘要

### 项目定位
`dmlx` 是 Apple `mlx-c` C 库的 Zig 语言绑定，提供 LLM 推理引擎。支持多种架构（LLaMA、DeepSeek V4、MiniMax、Nemotron-H 等）、多种 KV 缓存策略、量化、连续批处理、服务器模式。

### 总体评分

| 审视维度 | 得分 | 说明 |
|----------|------|------|
| 架构设计 | 7.5/10 | 模块化良好，200+ ops，多种缓存策略 |
| 深度实现 | 7.5/10 | 核心逻辑基本正确，但存在关键算法错误 |
| 设计模式 | 6.5/10 | VTable 正确，但 Array 所有权歧义、stub 模块多 |
| 构建/安全/资源 | 6.5/10 | 无 CI/CD，安全白名单危险，文档虚假声明 |
| 并发/API/性能 | 5.5/10 | 完全单线程，~432 函数 camelCase，采样器 O(V²) |
| 数学/边界/协议 | 4.7/10 | Llama causal mask 缺失，大量除零/OOB/溢出 |
| **DeepSeek V4 专项** | **4/10** | 算法缺陷、30GB+ 缓存、大量边界条件 |
| **综合评分** | **5.5/10** | 架构雄心与实现质量严重不匹配 |

### 最关键发现（Top 10）

1. **🔴 Llama Prefill 双向注意力** — `llama.zig` 预填充阶段传递 `mask=null`，注意力非因果，破坏自回归正确性
2. **🔴 采样器 O(V²) 插入排序** — 129K 词汇表每令牌 ~160 亿次比较，CPU 瓶颈
3. **🔴 879 个 `@intCast` ReleaseFast UB** — `build.zig` 支持 ReleaseFast，所有形状转换溢出为未定义行为
4. **🔴 函数级静态存储指向栈局部变量** — `server.zig` `StreamState` 存储 `&sse`、`req.model` 指针
5. **🔴 `custom_kernel.zig` 字符串悬挂指针** — `defer free` 后立即返回包含该指针的内核对象
6. **🔴 `last_error_buffer` 全局非线程安全** — 多线程 MLX 操作竞争写入
7. **🔴 `tool_executor.zig` Shell 白名单可绕过** — `python3 -c "..."` 可通过白名单执行任意代码
8. **🔴 文档虚假声明** — `competitive-advantages.md` 声称 350 测试通过，实际崩溃；`FIX-REPORT-DEEPSEEK-V4.md` 声称已修复但未修复
9. **🔴 `grad.zig`/`fused.zig`/`eval.zig` 硬编码 CPU Stream** — GPU 闲置，性能下降 10-100x
10. **🔴 DeepSeek V4 YARN RoPE 缓存 30GB+** — `max_seq_len=1M` 时 61 层共需 ~30.5GB 预计算缓存

---

## 1. 架构与设计审视（第一次）

### 1.1 正面评价
- **模块化清晰**: `ops/` 200+ 操作按功能分文件，`kvcache/` 9 种策略通过 VTable 统一接口
- **模型架构覆盖广**: LLaMA、DeepSeek V4、MiniMax、Nemotron-H、Gemma、GLM4、Qwen、Phi、LLaVA、Flux
- **量化支持**: W4A16/W8A8、MXFP4、TurboQuant、GPTQ 格式加载
- **连续批处理**: Scheduler + BatchBuilder 架构正确
- **分布式**: MPI 风格多 Mac 张量并行

### 1.2 架构缺陷
- **模型规模失控**: `deepseek_v4.zig` 2949 行，`deepseek_v4_loader.zig` 1983 行——单一文件承担过多子系统
- **KV Cache 策略过多**: 9 种后端（Standard、Paged、Quantized、Rotating、Radix、Tiered、PrefixDisk、TurboQuant、DeepSeekV4），无统一工厂
- **服务器设计虚假繁荣**: 声称支持"连续批处理"，实际 `engineLoop` 单纤程串行
- **无 CI/CD**: 无 `.github/workflows`，无自动化质量门

---

## 2. 深度实现问题（第二次）

### 2.1 线程安全
- **`last_error_buffer`**: `src/c.zig:32` 全局 2048 字节缓冲区，C 回调 `mlxErrorHandler` 直接写入，无锁、无 `threadlocal`
- **模型池非线程安全**: `ModelPool` 无 Mutex/RwLock，`lru_order` 在 `evictByName` 后包含悬空指针
- **专家缓存非线程安全**: `ExpertCache` 的 `hits`/`misses` 是普通 `u64`，并发访问会竞争

### 2.2 排序与采样
- **O(n²) 插入排序**: `src/sampling.zig` 对完整词汇表使用 `std.sort.insertion`。对于 128K 词汇量，最坏情况约 160 亿次比较
- **Top-p 归一化**: 在过滤后重新 softmax，正确

### 2.3 CLI 解析
- `main.zig` 使用手动字符串比较解析参数，无 `std.process.args` 结构化解析
- 缺少参数验证（如 `--temperature -1` 会被直接传递）

---

## 3. 设计模式与代码质量（第三次）

### 3.1 危险的 `@ptrCast` 假设
- `array.zig:strides()`: `const cast_ptr: [*]const i64 = @ptrCast(@alignCast(ptr));` — 假设 MLX 的 `size_t*` 与 `i64` 对齐且大小相同。在 64 位平台上成立，但无文档说明
- `safetensors_reader.zig`: `@ptrCast(result)` 用于 mmap 内存，无 `@alignCast`

### 3.2 内存泄漏模式
- `page_allocator` 多处泄漏：`batch_builder.zig` `emptyBatch` 使用 `page_allocator` 直接分配，调用者不释放
- `Array.zeros`/`ones` 接受 allocator 但忽略（`_ = allocator;`），C API 内部分配
- `EagerContext.deinit` 释放 stream，但如果 stream 是 default stream，可能不安全

### 3.3 Array 所有权歧义
- 无文档说明哪些函数接收所有权、哪些借用
- `closureCallback` 释放输入 Array，但 MLX 可能为梯度保留引用
- `tree.zig` 的 `treeMapInPlace` 语义不清晰

### 3.4 Stub 模块
- `diffusion/vae.zig`: 所有卷积权重零初始化，解码器输出恒为零
- `vision/llava.zig`: 视觉 tower 标注 "not yet implemented"
- `jang_quantizer.zig`: `analyzeSensitivity` 返回空，`quantizeModel` 不写文件

---

## 4. 构建系统、安全与资源（第四次）

### 4.1 构建系统
- `build.zig` `pkgConfigMlxPrefix` 成功路径未释放 `stdout` 缓冲区
- 无 MLX-C 版本检查：`pkg-config --modversion mlxc` 从未调用
- 仓库中追踪不明二进制文件：`check_types`(1.8MB)、`main`、`test_arr`、`test_mlx_c` 等 Mach-O 文件无源代码

### 4.2 安全边界
- **`tool_executor.zig` SHELL_WHITELIST 包含解释器**: `python3`、`python`、`zig`、`git`、`curl`、`wget`、`rm`
- 只检查第一个 token：`python3 -c "import os; os.system('rm -rf /')"` 可通过白名单
- `server.zig` HTTP 解析器：64KB 固定缓冲区，无请求大小限制，Slowloris 可行
- 无认证、无速率限制

### 4.3 文档虚假声明
- `docs/competitive-advantages.md`: "350 个单元测试全部通过" — **虚假**，`zig build test` 崩溃
- `docs/FIX-REPORT-DEEPSEEK-V4.md`: 声称修复全角字符和 BOS 验证(token 100000) — **未修复**，源码仍含 `｜`；BOS 验证存在但使用 token ID `0` 而非文档声称的 `100000`

### 4.4 资源生命周期
- `safetensors_reader.zig`: 每次 `loadTensor` 执行 open/pread/close，大量系统调用
- `tiered.zig`/`prefix_disk.zig`: 临时文件无清理机制，崩溃后残留
- `mlx_default_cpu_stream_new()` 创建的 stream 多处不释放（`array.zig:zeros/ones`、`eval.zig`）

---

## 5. 并发、API、性能与弹性（第五次）

### 5.1 并发模型（4/10）
- **零 Mutex/RwLock/Thread.Pool** 整个 `src/` 树
- 唯一 OS 线程：`layer_prefetcher.zig` 后台预取工作线程
- 服务器使用 `std.Io.async(...)` 协作纤程（GCD），非 Zig `async/await`
- `engineLoop` 单纤程串行，所有请求竞争同一个无锁 `ModelState`
- `ModelPool`、`ExpertCache`、`KVCacheStrategy` 均非线程安全

### 5.2 API 设计（5/10）
- **~432 个公共函数使用 camelCase**，违反 Zig 惯例（应为 snake_case）
- 无统一错误类型：`error.MlxError` + `MemoryError` + `RegistryError` + 大量临时错误
- `Array` 无任何算子方法，不支持链式调用
- NN 层所有字段 `pub`，暴露实现细节
- `std.mem.indexOf` 与 `std.mem.find` 混用（0.16 迁移不完全）

### 5.3 性能关键路径（5/10）
- **采样器**: `std.sort.insertion` 129K 元素 + `logits.eval()` GPU→CPU 同步每令牌 + 重复惩罚全词汇表 CPU→GPU 回拷
- **RotatingKVCache**: 预填充逐令牌循环，4S 次内核启动（S=序列长度）
- **BatchBuilder**: 每步重建 O(N²) mask，即使纯解码请求也全量计算
- **Stream 模式**: 每次加载完整专家张量（GB 级磁盘→GPU 拷贝）
- **无 SIMD**: 搜索 `@Vector`/`simd` 返回零匹配

### 5.4 弹性设计（4/10）
- 无信号处理（SIGINT/SIGTERM）— 缓存不保存，临时文件不清理
- 无请求级超时 — 慢请求永久阻塞
- 无重试、无断路器
- 缺失权重 = 硬失败，无部分加载回退
- `TieredKVCache.evictToSSD` 磁盘满时崩溃

### 5.5 数据格式兼容性（5/10）
- ❌ 无 GGUF 原生加载器
- ❌ 无 SentencePiece、TikToken、WordPiece
- ❌ 无 Jinja2 聊天模板（仅 4 种硬编码）
- ❌ 无图像解码器（PNG/JPEG/WebP）
- NPY 不支持 big-endian、int8/uint8/f16/bf16
- Safetensors 自定义读取器跳过 `__metadata__`

---

## 6. 数学正确性、边界条件与协议（第六次）

### 6.1 数学正确性（6/10）

**🔴 P0 — Llama Prefill 缺少 Causal Mask**
`src/models/llama.zig:657`:
```zig
const logits = try arena.track(try self.forward(input, null, caches));
```
预填充阶段 `mask=null`，注意力双向。DeepSeek V4 在同位置正确使用 `mask_mode="causal"`。

**🟡 P1 — Softplus Float16 溢出**
`src/ops/activations.zig:90` 阈值 20.0 > float16 `exp` 安全上限 ~11.5。

**✅ 数学正确的组件**: RoPE（所有变体）、RMSNorm、Softmax、Cross-Entropy、Temperature、Top-p、AdamW。

### 6.2 边界条件（4/10）

**边界条件缺陷清单（经源码验证）:**

| 缺陷 | 位置 | 触发条件 | 后果 |
|------|------|----------|------|
| OOB 读取空 logits | `sampling.zig` | `vocab_size == 0` | 崩溃 |
| `seq_len - 1` 下溢 | `generation.zig:114` | `seq_len == 0` | 崩溃 |
| `max_new_tokens - 1` 下溢 | `deepseek_v4.zig:2807` | `max_new_tokens == 0` | 无限循环 |
| `base == 1.0` 除零 | `deepseek_v4.zig:336` | `rope_theta = 1.0` | NaN/Inf |
| `compress_ratio == 0` 除零 | `deepseek_v4.zig:1889` | 配置错误 | 崩溃 |
| `window_size == 0` 除零 | `rotating.zig:127` | 配置错误 | 崩溃 |
| `page_size == 0` 除零 | `paged.zig:358` | 配置错误 | 崩溃 |
| `n_experts - k` 下溢 | `deepseek_v4.zig:376` | `k > n_experts` | 崩溃 |
| mask 缓冲区溢出 | `batch_builder.zig:140` | `total_tokens² > usize.MAX` | 堆损坏 |
| 无 ndim 检查 | `deepseek_v4.zig:1604` | `ndim < 3` | OOB panic |
| RoPE cache OOB | `deepseek_v4.zig:252` | `start_pos+seq_len > max_seq_len` | OOB |

**879 个 `@intCast`**：ReleaseFast 模式下溢出 = 未定义行为。

### 6.3 协议合规性（4/10）

- OpenAI API 缺失 `top_p`、`top_k`、`frequency_penalty`、`presence_penalty`、`logprobs`、`n` 等字段
- `std.json.parseFromSlice` 默认 `ignore_unknown_fields = false` — 客户端发送额外字段直接 400
- HTTP 解析：大小写敏感 header、无 chunked encoding、64KB 固定缓冲区、无持久连接
- SSE：无 `id`/`retry`、多行数据不安全
- JSON 转义：缺失 `\b`、`\f` 和所有控制字符 `\u00XX`
- 错误响应格式不一致：大部分是 `{"error":"string"}` 而非 OpenAI 的嵌套对象
- 无认证

### 6.4 Zig 语言陷阱（4/10）

**🔴 P0 — 函数级静态存储指向栈局部变量**
`server.zig:1122-1132`:
```zig
const StreamState = struct {
    var s_sse: *SSEWriter = undefined;      // ← &sse (栈局部)
    var s_model_name: []const u8 = "";      // ← req.model
    // ...
};
```

**🔴 P0 — `custom_kernel.zig` 悬挂指针**
```zig
const name_z = try allocator.dupeZ(u8, name);
defer allocator.free(name_z);  // 返回后释放
return .{ .inner = c.c.mlx_fast_metal_kernel_new(name_z.ptr, ...) };
```

**🟡 P1 — `catch {}` 静默丢弃错误**
- `server.zig:934`: 生成期间 token 静默丢弃
- `tool_executor.zig:144`: 目录创建失败静默忽略
- `memory.zig`: 逐出失败静默忽略

**🟡 P1 — 资源泄漏路径**
- `ops/fused.zig`: `swigluForward`、`adamwStepForward`、`unfusedAdamWStep` 多处缺少 `defer`/`errdefer`
- `closure.zig`: `closureCallback` 部分失败时 `out_arrs` 泄漏；`Closure.apply` `out_vec` 无 `errdefer`

**🟡 P1 — C API 返回值忽略**
- `array.zig:165`: `_ = c.c.mlx_array_tostring(...)` — 失败时 `str` 未定义
- `device.zig:36`: `_ = c.c.mlx_device_get_type(...)`

---

## DeepSeek V4 模块专项分析（优先）

### 文件规模与复杂度

| 文件 | 行数 | 子系统 |
|------|------|--------|
| `deepseek_v4.zig` | 2949 | MLA、MoE、CSA/HCA、mHC、YARN RoPE、Gate、Router、Compressor、Indexer、Attention、Tokenizer adapter |
| `deepseek_v4_loader.zig` | 1983 | 权重名映射、量化反量化、Smelt 配置、层构建 |
| `expert_stream.zig` | ~500 | Stream 模式专家加载、预取 |
| `expert_preload.zig` | ~400 | Preload 模式专家加载 |
| `kvcache/deepseek_v4_cache.zig` | ~600 | CSA/HCA KV cache、窗口管理 |

**评估**: 单一文件 2949 行承载 5+ 个复杂子系统，远超可维护性阈值（~500 行/文件）。

### 数学与算法问题

| 问题 | 位置 | 验证状态 | 说明 |
|------|------|----------|------|
| Prefill causal mask | — | N/A | DS V4 **正确**使用 `mask_mode="causal"` |
| `findCorrectionDim` 除零 | `z:336` | ✅ 确认 | `base==1.0` → `@log(1.0)==0` → 除零 |
| YARN RoPE 缓存内存 | `z:192-225` | ✅ 确认 | `max_seq_len * half_dim * 4B` × 2 缓存/层 × 61 层 ≈ **30.5 GB** |
| RoPE cache OOB | `z:252` | ✅ 确认 | `start_pos + seq_len > max_seq_len` 时越界 |
| `createCausalMask` shape 溢出 | `z:397` | ⚠️ 部分 | 4 个 `@intCast` 到 `i32` 可溢出 |
| `topkIndices` 下溢 | `z:376` | ✅ 确认 | `k > n_experts` 时 `usize` 下溢 |
| `compress_ratio == 0` 除零 | `z:1889` | ✅ 确认 | `W = ready_len / self.compress_ratio` |

### 内存与性能问题

| 问题 | 位置 | 影响 |
|------|------|------|
| YARN RoPE 预计算 30GB+ | `z:192-225` | 61 层 × 512MB = 超出大多数 Mac 内存 |
| Stream 模式全张量加载 | `expert_stream.zig:256` | 每次 MoE 层加载完整权重（GB 级） |
| 逐层 `hidden.eval()` | `z:~1700` | 阻止跨层内核融合 |
| `RotatingKVCache` 逐 token | `kvcache/rotating.zig` | 4S 次内核启动/预填充 |

### 边界条件与崩溃风险

| 问题 | 位置 | 触发 | 后果 |
|------|------|------|------|
| `max_new_tokens==0` 无限循环 | `z:2807` | `for (0..max_new_tokens-1)` | 内存耗尽 |
| `DSV4Attention` ndim<2 OOB | `z:1604` | `hidden_states` 非 3D | Panic |
| `DSV4YarnRoPE.apply` ndim<2 | `z:242` | 输入非至少 2D | Panic |
| `head_dim==0` scale=+inf | `z:1609` | `1.0/@sqrt(0)` | NaN 传播 |
| `page_size==0` 除零 | `paged.zig:358` | 配置错误 | Panic |
| `window_size==0` 除零 | `rotating.zig:127` | 配置错误 | Panic |

### 量化与加载问题

| 问题 | 位置 | 验证 | 说明 |
|------|------|------|------|
| `dequantIfNeeded` 硬编码 affine | `loader.zig:122` | ✅ | 忽略配置的 `mxfp4`/`nvfp4`/`mxfp8` |
| `dispatchGatherMm` 只认 mxfp4 | `z:967` | ✅ | 其他模式回退到 `.affine` |
| `dequantFp4` 浮点除法 | `loader.zig:678` | ✅ | `divide(packed, 16)` 非 `rightShift` |
| `dequantFp8` scale 不匹配 | `loader.zig:726` | ✅ | padding 权重但未 padding scale |
| 专家部分读取被绕过 | `expert_stream.zig:310` | ✅ | 明确注释 "bypassed" |
| `expandQuantizedBuffer` 形状错误 | `quantized.zig:373` | ✅ | `seq_len * el_per_int` 误作 head_dim |

### 代码质量与技术债务

| 问题 | 位置 | 说明 |
|------|------|------|
| Python 伪代码注释 | `z:162` | `// return (pre[..., None] * x.astype(f32)).sum(axis=2)` |
| TODO: pool_base | `z:1943` | `// TODO: Use proper position computation with pool_base` |
| MoE Router 集成注释 | `z:637` | 15 行正确代码被注释掉，使用手动 top-k |
| 代码重复 | 多个加载器 | `getFloat`/`getInt`/`mapLayerComponent` 在 4 个文件中重复 |
| `@intCast` 泛滥 | 整个模块 | `usize`→`i32` 形状转换无溢出检查 |

---

## 全部缺陷汇总

### 🔴 P0 缺陷（崩溃/安全/算法错误）

| # | 缺陷 | 来源 | 验证 |
|---|------|------|------|
| 1 | Llama prefill 双向注意力 | 第六次 | ✅ |
| 2 | 采样器 O(V²) 插入排序 | 第二次 | ✅ |
| 3 | 879 个 `@intCast` ReleaseFast UB | 第六次 | ✅ |
| 4 | `StreamState` 静态存储指向栈局部变量 | 第六次 | ✅ |
| 5 | `custom_kernel.zig` 字符串悬挂指针 | 第六次 | ✅ |
| 6 | `last_error_buffer` 全局非线程安全 | 第二次 | ✅ |
| 7 | Shell 白名单包含解释器且可绕过 | 第四次 | ✅ |
| 8 | 文档虚假声明（350 测试通过、DS V4 修复） | 第四次 | ✅ |
| 9 | `grad.zig`/`fused.zig`/`eval.zig` 硬编码 CPU Stream | 第三次 | ✅ |
| 10 | `expert_remap_test.zig` 崩溃 | 第四次 | ✅ |
| 11 | `model_pool.zig` LRU 悬空指针 | 第三次 | ✅ |
| 12 | `npy.zig` `dtype.val` 编译错误 | 第三次 | ✅ |
| 13 | 采样器空 logits OOB | 第六次 | ✅ |
| 14 | `seq_len - 1` 下溢 | 第六次 | ✅ |
| 15 | `max_new_tokens - 1` 下溢无限循环 | 第六次 | ✅ |
| 16 | `findCorrectionDim` base==1.0 除零 | 第六次 | ✅ |
| 17 | `createCausalMask`/`batch_builder` mask 溢出 | 第六次 | ⚠️ |
| 18 | `topkIndices` k>n_experts 下溢 | 第六次 | ✅ |
| 19 | `compress_ratio==0` 除零 | 第六次 | ✅ |
| 20 | `window_size==0` 除零 | 第六次 | ✅ |
| 21 | `page_size==0` 除零 | 第六次 | ✅ |
| 22 | `expandQuantizedBuffer` 形状计算错误 | 第六次 | ✅ |
| 23 | `@divTrunc` 导致量化缓冲区过小 | 第六次 | ✅ |
| 24 | 仓库中不明二进制文件 | 第四次 | ✅ |
| 25 | `tool_executor.zig` 目录创建失败静默忽略 | 第四次 | ✅ |
| 26 | `guided.zig` JSON 解析可被欺骗 | 第三次 | ✅ |
| 27 | `server.zig` 64KB 缓冲区溢出风险 | 第四次 | ✅ |

### 🟡 P1 缺陷（性能/功能/稳定性）

| # | 缺陷 | 来源 |
|---|------|------|
| 28 | AdamW 每步 9 次标量分配（注释称 ~15） | 第三次 |
| 29 | `trainer.zig` 双释放风险 | 第三次 |
| 30 | `guided.zig` 字符串搜索非 JSON 解析 | 第三次 |
| 31 | `batch_builder.zig` O(N²) mask | 第三次 |
| 32 | `optim.zig` 引用 `compiledAdamWStep` 但未使用 | 第三次 |
| 33 | `closure.zig` 所有权 bug | 第三次 |
| 34 | `benchmark.zig` 4096 静默截断 | 第三次 |
| 35 | `diffusion/vae.zig` 零权重 | 第三次 |
| 36 | `build.zig` 内存泄漏 | 第四次 |
| 37 | MLX-C 无版本检查 | 第五次 |
| 38 | `zig-pkg/` 被 git 追踪（双源风险） | 第五次 |
| 39 | 服务器无信号处理 | 第五次 |
| 40 | 无请求级超时 | 第五次 |
| 41 | 无 MLX 错误分类 | 第五次 |
| 42 | CJK 正则范围不匹配 | 第五次 |
| 43 | HTTP 缺少 charset | 第五次 |
| 44 | `jsonEscapeInto` 不完整 | 第五次 |
| 45 | `tool_calling.zig` 引号跳过 bug | 第五次 |
| 46 | `deepseek_v4.zig` YARN RoPE 30GB | 第六次 |
| 47 | Stream 模式全张量加载 | 第六次 |
| 48 | `RotatingKVCache` 逐 token 循环 | 第六次 |
| 49 | `DSV4YarnRoPE.apply` OOB (ndim/start_pos) | 第六次 |
| 50 | `DSV4Attention` ndim 假设 | 第六次 |
| 51 | `dequantIfNeeded` 硬编码 affine | 第六次 |
| 52 | `dispatchGatherMm` 只认 mxfp4 | 第六次 |
| 53 | `dequantFp4` 浮点除法 | 第六次 |
| 54 | `dequantFp8` scale 不匹配 | 第六次 |
| 55 | QJL 矩阵转置错误 | 第六次 |
| 56 | `ops/fused.zig` 泄漏路径 | 第六次 |
| 57 | `closure.zig` 泄漏路径 | 第六次 |
| 58 | C API 返回值忽略 | 第六次 |
| 59 | Softplus float16 溢出 | 第六次 |
| 60 | 7 个幽灵功能 | 第六次 |
| 61 | OpenAI API 字段缺失 | 第六次 |
| 62 | `ignore_unknown_fields=false` | 第六次 |
| 63 | HTTP 大小写敏感 header | 第六次 |
| 64 | SSE 多行数据不安全 | 第六次 |
| 65 | 错误响应格式不一致 | 第六次 |

### 🟢 P2 缺陷（人体工学/规范/轻微）

| # | 缺陷 | 来源 |
|---|------|------|
| 66 | ~432 函数 camelCase | 第五次 |
| 67 | 无统一错误类型 | 第五次 |
| 68 | `Array` 无方法链 | 第五次 |
| 69 | NN 层字段全部 pub | 第五次 |
| 70 | `Array.zeros`/`ones` 误导性 allocator | 第五次 |
| 71 | `std.mem.indexOf`/`find` 混用 | 第五次 |
| 72 | 无 SIMD | 第五次 |
| 73 | 测试无 mocking 框架 | 第六次 |
| 74 | 无性能回归测试 | 第六次 |
| 75 | KV cache "comptime shape" 虚假声明 | 第五次 |
| 76 | 无 Unicode 规范化 | 第五次 |
| 77 | `safety_margin_bytes` 硬编码 512MB | 第四次 |
| 78 | `LlamaConfig.rope_theta` 默认 10000 | 第四次 |
| 79 | `ServerConfig.kv_bits` 默认 4 | 第四次 |

---

## 修复建议（按优先级排序）

### 紧急（本周内）
1. **修复 Llama causal mask** — 在 `llama.zig:657` 的 prefill 路径传递 causal mask
2. **修复采样器** — 将 `std.sort.insertion` 替换为 `std.mem.sort`；或优先使用 MLX GPU `topk`
3. **修复所有除零路径** — 在 `findCorrectionDim`、`compress_ratio`、`window_size`、`page_size` 添加 `> 0` 验证
4. **修复 `max_new_tokens - 1` 下溢** — 使用 `saturatingSub` 或前置验证
5. **修复 `StreamState` 静态存储** — 改为通过上下文指针传递
6. **删除/隔离仓库中的不明二进制文件** — `check_types`、`main`、`test_*`

### 高优先级（本月内）
7. **修复 DeepSeek V4 YARN RoPE 内存** — 按需计算而非预计算全部 30GB；或文档声明仅支持较小 max_seq_len
8. **修复 `expandQuantizedBuffer` 形状错误** — `dim` 应使用 `head_dim` 而非 `seq_len * el_per_int`
9. **修复 `custom_kernel.zig` 悬挂指针** — 确保字符串生命周期覆盖内核对象生命周期
10. **修复 `last_error_buffer` 线程安全** — 添加 `threadlocal` 或改为线程局部存储
11. **修复 `tool_executor.zig` 白名单** — 移除所有解释器和网络工具；或添加参数级验证
12. **修复文档虚假声明** — 更正 `competitive-advantages.md` 和 `FIX-REPORT-DEEPSEEK-V4.md`
13. **添加 MLX-C 版本检查** — `pkg-config --modversion mlxc`

### 中优先级（季度内）
14. **修复 Stream 模式专家加载** — 恢复部分读取（`PartialTensorReader`）而非加载完整张量
15. **修复 ` RotatingKVCache` 逐 token 循环** — 使用批量 slice_update
16. **修复 BatchBuilder mask 重建** — 缓存解码请求的 trivial mask
17. **修复 `grad.zig`/`fused.zig`/`eval.zig` CPU stream** — 传递正确 GPU stream
18. **统一命名规范** — snake_case 公共函数
19. **修复 JSON 转义** — 添加 `\b`、`\f` 和所有控制字符 `\u00XX`
20. **修复 OpenAI API 兼容性** — `ignore_unknown_fields = true`，添加缺失字段

### 低优先级（技术债务）
21. 拆分 `deepseek_v4.zig` 和 `deepseek_v4_loader.zig`
22. 为 KV cache 添加统一工厂函数
23. 删除/隔离 7 个幽灵功能
24. 添加信号处理（SIGINT/SIGTERM）
25. 添加请求级超时
26. 添加 CI/CD（GitHub Actions）

---

*本报告基于对 dmlx 代码库的六次独立维度审视，所有声明均经源码交叉验证。报告生成日期：2026-05-01。*
