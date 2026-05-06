# Continuous Batching 设计文档

> **状态**：框架已完备。Server 集成待完成（预计 1-2 周）。
> **源文件**：`src/scheduler.zig`、`src/batch_builder.zig`、`src/kvcache/paged.zig`、`src/kvcache/interface.zig`
> **需求追踪**：R13.1, R13.2, R13.3

---

## 1. 架构总览

dmlx 的 continuous batching 架构基于 **三阶段 Engine Step 循环**：

```
                   ┌──────────────────────────────────────────┐
                   │              Engine Loop                  │
                   │                                          │
  addRequest() ──► │  ┌──────────┐  ┌──────────┐  ┌────────┐ │
                   │  │ schedule │─►│ forward  │─►│ post   │ │
                   │  └──────────┘  └──────────┘  └────────┘ │
                   │       │             ▲              │      │
                   │       ▼             │              ▼      │
                   │  BatchBuilder    Model.forward   cleanup  │
                   │       │             │              │      │
                   │       ▼             │              ▼      │
                   │  [batched_tensor]   │      free blocks    │
                   └─────────────────────┴────────────────────┘
```

### 组件映射

| 组件 | 文件 | 行数 | 职责 |
|-----------|------|------|------|
| **Scheduler** | `src/scheduler.zig` | 387 | 请求队列管理、块分配、停止条件检查 |
| **BatchBuilder** | `src/batch_builder.zig` | 256 | Token 拼接、位置编码、因果注意力掩码 |
| **BlockManager** | `src/scheduler.zig`（包装）+ `src/kvcache/paged.zig`（真实） | 73 + 1152 | KV cache 块池，支持 CoW 和前缀哈希 |
| **KVCacheStrategy** | `src/kvcache/interface.zig` | 149 | 所有缓存策略的多态 filter/update 接口 |
| **Server** | `src/server.zig` | 1517 | HTTP 服务器、连接处理、SSE 流式输出 |

---

## 2. 请求生命周期

### 2.1 状态机

```
  addRequest()
       │
       ▼
  ┌─────────┐   schedule()    ┌────────────┐  预填充     ┌──────────┐
  │ waiting │ ──────────────► │ prefilling │ ──────────► │ decoding │
  └─────────┘                 └────────────┘  完成       └──────────┘
       ▲                           │                          │
       │   无可用块                  │  分块预填充               │  stop / max_tokens
       │                           │  (未完成)                 │
       │                           ▼                          ▼
       │                      保持在                      ┌──────┐
       └────────────────── prefilling                    │ done │
                                                        └──────┘
                                                           │
                                                    freeBlocks()
```

### 2.2 Request 结构体 (`src/scheduler.zig:108-187`)

```zig
pub const Request = struct {
    id: u64,                          // 唯一请求标识符
    prompt_tokens: []const u32,       // 原始提示词 token 序列
    generated_tokens: ArrayList(u32), // 生成的输出 token
    state: RequestState,              // waiting | prefilling | decoding | done
    block_ids: ArrayList(usize),      // 已分配的 KV cache 块 ID
    max_tokens: usize,                // 生成限制
    stop_tokens: []const u32,         // EOS / 自定义停止 token
    prefill_offset: usize,            // 分块预填充的游标
    done: bool,                       // 完成标志（供外部轮询）
    result_tokens: ?[]const u32,      // 最终输出（完成时设置）
};
```

### 2.3 关键方法

| 方法 | 说明 |
|--------|-------------|
| `seqLen()` | 总序列长度 = prefill_offset + generated_tokens.len |
| `isStopToken(token)` | 检查 token 是否匹配任何停止 token |
| `hasPendingPrefill()` | 提示词是否未完全消费 |
| `currentPrefillChunkLen(max)` | 下一预填充块大小（受上限约束） |
| `markComplete()` | 设置 result_tokens，done=true |
| `waitForCompletion(io)` | 阻塞轮询，间隔 1ms |

---

## 3. Scheduler（`src/scheduler.zig`）

### 3.1 设计

```zig
pub const Scheduler = struct {
    allocator: std.mem.Allocator,
    waiting: ArrayList(*Request),       // FCFS 队列
    running: ArrayList(*Request),       // 活跃请求
    block_manager: *BlockManager,       // KV cache 块池
    max_prefill_tokens: usize,          // 分块预填充限制（默认：512）
    blocks_per_request: usize,          // 每请求块数（默认：1）
};
```

### 3.2 `schedule()` — 请求选择

优先级顺序：
1. **Running 请求** — 始终优先纳入（保证解码连续性）
2. **Waiting 请求** — 仅在有可用块时提升

```zig
pub fn schedule(self: *Scheduler) !ScheduleResult {
    // 阶段 1：对 running 请求分类
    for (self.running.items) |req| {
        if (req.state == .prefilling and req.hasPendingPrefill())
            prefill_list.append(req)   // 继续分块预填充
        else
            decode_list.append(req)    // 解码
    }

    // 阶段 2：有可用块时提升 waiting 请求
    for (self.waiting.items) |req| {
        if (self.block_manager.canAllocate(self.blocks_per_request)) {
            self.block_manager.allocateBlocks(allocator, req, ...);
            req.state = .prefilling;
            prefill_list.append(req);
            self.running.append(req);
        } else {
            still_waiting.append(req); // 继续等待
        }
    }

    return ScheduleResult{
        .prefill_requests = prefill_list,
        .decode_requests = decode_list,
        .blocks_needed = blocks_needed,
    };
}
```

### 3.3 `postprocess()` — 输出处理 + 清理

```zig
pub fn postprocess(self: *Scheduler, outputs: []const TokenOutput) !void {
    // 阶段 1：应用生成的 token
    for (outputs) |output| {
        if (req.state == .prefilling) {
            // 按块大小推进 prefill_offset
            req.prefill_offset += chunk_len;
            if (!req.hasPendingPrefill()) {
                req.state = .decoding;           // 切换到解码
                req.generated_tokens.append(output.token); // 首生成 token
            }
        } else {
            // 解码：追加 token，检查停止条件
            req.generated_tokens.append(output.token);
            if (req.isStopToken(token) or at_max) req.state = .done;
        }
    }

    // 阶段 2：移除完成的请求，释放 KV cache 块
    while (i < self.running.items.len) {
        if (req.state == .done) {
            self.block_manager.freeBlocks(req);
            _ = self.running.orderedRemove(i);
        } else { i += 1; }
    }
}
```

### 3.4 Engine Step 集成

`server.zig:185-229` 中的 `engineLoop` 应实现此循环：

```
1. scheduled = scheduler.schedule()
2. batch    = batch_builder.build(&scheduled, ctx)
3. logits   = model.forward(batch.batched_tokens, batch.position_ids, batch.attention_mask, caches)
4. tokens   = sample(logits) → []TokenOutput
5. outputs  = scheduler.postprocess(tokens)
6. 对每个完成的请求 → SSE 流式返回结果给客户端
```

> **当前状态**：步骤 2-3 未接线。engineLoop 桩代码仅做状态转换，无实际 forward pass。见 `server.zig:211-216`。

---

## 4. BatchBuilder（`src/batch_builder.zig`）

### 4.1 设计

将 `ScheduleResult`（分类的 prefill + decode 请求）转换为单一 batched input tensor，供一次模型 forward pass 使用。

### 4.2 BatchResult

```zig
pub const BatchResult = struct {
    batched_tokens: Array,     // [total_tokens] — 拼接后的 token ID
    position_ids: Array,       // [total_tokens] — 每 token 的位置索引
    attention_mask: Array,     // [total_tokens, total_tokens] — 0.0=可关注, -inf=屏蔽
    total_tokens: usize,       // batch 中的总 token 数
    num_requests: usize,       // batch 中的请求数
};
```

### 4.3 Token 选择逻辑

```zig
fn getRequestTokens(req: *const Request) []const u32 {
    switch (req.state) {
        .prefilling, .waiting =>
            // 返回 prefill_offset 到末尾的提示词 token
            return req.prompt_tokens[req.prefill_offset..],
        .decoding =>
            // 返回单个 token：最后生成的（或提示词末尾，如果还没有生成）
            return &[_]u32{ req.generated_tokens[last] },
        .done =>
            return &[_]u32{},
    }
}
```

### 4.4 注意力掩码构建

每个请求的 token 在 batch 中形成连续片段。掩码确保：

- **请求内**（`R13.2`）：因果注意力 — 位置 `row` 的 token 可关注位置 `0..row`
- **请求间**：完全屏蔽 — 不同请求的 token 永不相见

```
3 个请求（序列长度：3, 2, 2）的 Mask：
          req0        req1      req2
tok:    0  1  2   |  3  4  |  5  6
      ┌────────────┬─────────┬───────┐
  0   │ 0  -  -   │  -  -  │  -  - │
  1   │ 0  0  -   │  -  -  │  -  - │
  2   │ 0  0  0   │  -  -  │  -  - │
      ├────────────┼─────────┼───────┤
  3   │ -  -  -   │  0  -  │  -  - │
  4   │ -  -  -   │  0  0  │  -  - │
      ├────────────┼─────────┼───────┤
  5   │ -  -  -   │  -  -  │  0  - │
  6   │ -  -  -   │  -  -  │  0  0 │
      └────────────┴─────────┴───────┘

图例: 0 = 可关注 (0.0), - = 屏蔽 (-inf)
```

### 4.5 位置 ID

- **预填充请求**：`prefill_offset + j`（序列中的绝对位置）
- **解码请求**：`seqLen() - 1`（当前位置）

---

## 5. 分块预填充（Chunked Prefill）

### 5.1 动机

大提示词（> `max_prefill_tokens`）被分割为多个块，防止：
- 解码请求饥饿（解码步骤与预填充块交错进行）
- 内存峰值（每个块使用有界的 KV cache 空间）

### 5.2 流程

```
步骤 N：   schedule() → prefill_list: [req_A(chunk_0)], decode_list: [req_B, req_C]
          forward pass → 处理 req_A 的 chunk_0 + 解码 req_B, req_C
          postprocess() → req_A.prefill_offset += 512，保持 prefilling

步骤 N+1： schedule() → prefill_list: [req_A(chunk_1)], decode_list: [req_B, req_C]
          forward pass → 处理 req_A 的 chunk_1 + 解码 req_B, req_C
          postprocess() → req_A.prefill_offset += 512，保持 prefilling

...       （重复直到提示词完全消费）

步骤 N+K： schedule() → prefill_list: [req_A(final_chunk)], decode_list: [req_B, req_C]
          forward pass → 处理最终块
          postprocess() → req_A 切换到 .decoding
```

### 5.3 关键不变量

- 解码请求**永不被**预填充阻塞 — 每步始终包含它们
- 预填充请求在多个步骤间保持在 `running` 中
- `prefill_offset` 是每步前进 `max_prefill_tokens` 的游标
- 仅当 `prefill_offset >= prompt_tokens.len` 时才切换到 `.decoding`

---

## 6. BlockManager — KV Cache 块分配

### 6.1 接口 (`src/scheduler.zig:22-95`)

```zig
pub const BlockManager = struct {
    total_blocks: usize,
    used_blocks: usize,
    real: ?*RealBlockManager,     // 支持真实 paged KV cache（CoW、前缀哈希）

    pub fn init(total_blocks: usize) BlockManager;
    pub fn canAllocate(num_blocks: usize) bool;
    pub fn allocateBlocks(allocator, req: *Request, num_blocks: usize) !void;
    pub fn freeBlocks(req: *Request) void;
    pub fn freeCount() usize;
};
```

### 6.2 真实 BlockManager (`src/kvcache/paged.zig:44`)

真实 `BlockManager` 管理 `Block` 结构体池，包含：

| 特性 | 说明 |
|---------|-------------|
| **Copy-on-Write** | 共享块（`ref_count > 1`）在修改前被复制 |
| **前缀哈希** | 基于 Wyhash 的滚动哈希，用于缓存命中检测 |
| **每请求映射** | `req_to_blocks: HashMap(u64, ArrayList(usize))` |
| **块哈希** | `block_hashes: HashMap(u64, usize)` 用于查找 |
| **访问追踪** | `last_access` 时间戳，用于 LRU 淘汰 |
| **默认页大小** | 32 token（针对 Apple Silicon Metal GPU 对齐调优） |

### 6.3 块生命周期

```
allocateBlocks(req_id, N):
  1. 从 free_blocks 池弹出 N 个块
  2. 添加到 req_to_blocks[req_id]
  3. used_blocks += N

freeBlocks(req_id):
  1. 对 req_to_blocks[req_id] 中的每个块：
     a. ref_count -= 1
     b. 如果 ref_count == 0：推回 free_blocks
  2. 从 req_to_blocks 中移除 req_id
```

---

## 7. Continuous Batching 的 KV Cache 策略

### 7.1 VTable 接口 (`src/kvcache/interface.zig`)

```zig
pub const VTable = struct {
    updateAndFetch: *const fn (ctx, keys, values, stream) anyerror!KVSlice,
    currentLen:    *const fn (ctx) usize,
    reset:         *const fn (ctx) void,
    filter:        ?*const fn (ctx, indices, allocator) anyerror!void, // ← CB 关键
    rollback:      ?*const fn (ctx, to_len) void,
    deinit:        *const fn (ctx, allocator) void,
};
```

### 7.2 策略支持矩阵

| 策略 | `filter` | 使用场景 |
|----------|----------|----------|
| **Paged** | ✅ 支持 | **Continuous batching 默认策略** |
| **PagedQuantized** | ✅ 支持 | 内存受限的 CB |
| **Tiered** | ✅ 支持（热层 = Paged） | 长上下文 CB + SSD 溢出 |
| **Quantized** | ✅ 支持 | 紧凑 CB |
| **Standard** | ✅ 支持 | 简单单请求 |
| **Rotating** | ❌ 不支持 | 固定窗口，无动态 batch |

### 7.3 `filter()` — 动态 Batch 大小调整

请求完成（`.done`）时，调用 `filter()` 移除其 KV cache 条目：

```zig
// 移除已完成的请求的 KV cache
cache.filter(&[_]usize{0, 2}, allocator) // 保留 batch 索引 0 和 2
```

这确保每次 `postprocess()` 后 KV cache 大小与活跃 batch 大小匹配。

---

## 8. Server 集成蓝图

### 8.1 当前状态

```
POST /v1/chat/completions
  → handleStreamingCompletion()     // 串行：一次一个请求
  → generateStep() 循环              // 未使用 batch builder
```

### 8.2 目标状态

```
POST /v1/chat/completions
  → state.scheduler.addRequest(req) // 入队（非阻塞）
  → req.waitForCompletion(io)       // 轮询直到完成（在连接 fiber 中）

engineLoop (异步 fiber):
  while (state.running):
    scheduled = scheduler.schedule()
    if scheduled.isEmpty(): sleep(1ms); continue

    batch = batch_builder.build(&scheduled, ctx)
    logits = model.forward(batch.batched_tokens, batch.position_ids,
                           batch.attention_mask, caches)
    tokens = sample(logits, scheduled)
    scheduler.postprocess(tokens)

    for 完成的请求:
      SSE 流式返回结果给连接
      cache.filter(remaining_indices)  // 移除完成的条目
```

### 8.3 集成步骤（来自 `server.zig:530-540`）

1. 从解析的聊天补全请求创建 `scheduler_mod.Request`
2. `state.scheduler.addRequest(&req)` — 入队
3. 在引擎循环中：`scheduler.schedule()` 选择请求
4. `batch_builder.build()` 创建批输入 tensors
5. `state.vtable.forward(...)` — 所有请求的单次 forward pass
6. `scheduler.postprocess(outputs)` — 应用 token，检查停止条件
7. SSE 流式返回结果给每个请求的连接

---

## 9. 测试覆盖

### 9.1 Batch Builder 测试 (`src/tests/batch_builder_tests.zig`)

| 测试 | 验证 | 状态 |
|------|------|------|
| `single prefill request` | Token 拼接、注意力隔离 | ✅ 通过 |
| `single decode request` | 单 token 提取 | ✅ 通过 |
| `mixed prefill and decode` | R13.2 — 跨状态注意力隔离 | ✅ 通过 |
| `multiple decode requests` | 多请求解码批处理 | ✅ 通过 |
| `new request joins without waiting` | **R13.3** — continuous batching 加入 | ✅ 通过 |
| `tensor shapes are correct` | 输出维度、位置 ID | ✅ 通过 |
| `attention mask structure` | 请求内因果、请求间屏蔽 | ✅ 通过 |

### 9.2 集成测试 (`src/tests/integration_tests.zig`)

| 测试 | 验证 | 状态 |
|------|------|------|
| `submit → schedule → batch → postprocess` | 完整流程（无 forward pass） | ✅ 通过 |
| `multiple concurrent requests with CB` | R13 — 多请求批处理 | ✅ 通过 |
| `new request joins existing decode batch` | R13.3 — 生成中途加入 | ✅ 通过 |
| `chunked prefill with concurrent decode` | 分块预填充 + 解码交错 | ✅ 通过 |

---

## 10. 已知缺口

| 缺口 | 影响 | 预估工作量 |
|-----|--------|-----------------|
| Server engineLoop 未连接 BatchBuilder | 无真批 forward — 每请求串行 | **1-2 周** |
| 无每请求 RoPE delta 追踪 | mRoPE 模型（DeepSeek V4）需要每序列位置偏移 | 2-3 天 |
| 无外部预填充 | 大提示词在批内处理，可能导致 TTFT 峰值 | 3-5 天 |
| 无内存感知调度 | 并发请求无 OOM 保护 | 2-3 天 |
| SSE 流式输出未集成调度器输出 | 结果需路由回正确的连接 fiber | 2-3 天 |

---

## 11. 与 oMLX 对比

| 维度 | oMLX | dmlx（当前） | dmlx（目标） |
|-----------|------|-------------------|---------------------|
| **语言** | Python | Zig | Zig |
| **批引擎** | mlx-lm `BatchGenerator` | 自定义 `BatchBuilder` | 同当前 |
| **调度器** | `Scheduler`（4322 行） | `Scheduler`（387 行） | 同当前 |
| **队列模型** | `waiting` deque + `running` dict | `waiting` ArrayList + `running` ArrayList | 同当前 |
| **批插入** | `batch_generator.insert()` | `batch_builder.build()` | 同当前 |
| **批移除** | `batch_generator.remove()` | `KVCacheStrategy.filter()` | 同当前 |
| **分块预填充** | `prefill_step_size=2048` | `max_prefill_tokens=512` | 同当前 |
| **Paged KV** | `PagedCacheManager` + CoW | `BlockManager` + CoW | 同当前 |
| **外部预填充** | ✅ `_do_external_prefill()` | ❌ 未实现 | 需要 |
| **推测预填充** | ✅ `specprefill` + draft model | ❌ 未实现 | 未来 |
| **Server 集成** | ✅ 完全集成 | ❌ engineLoop 桩代码 | 1-2 周 |
| **并发连接** | FastAPI + asyncio | `io.async` fibers | 基本就绪 |

---

## 12. 参考资料

- **源码**：`src/scheduler.zig`、`src/batch_builder.zig`、`src/kvcache/paged.zig`、`src/server.zig`
- **测试**：`src/tests/batch_builder_tests.zig`、`src/tests/integration_tests.zig`
- **分析**：`docs/zh/analysis/05-server-and-service.md`、`docs/zh/analysis/04-kv-cache-subsystem.md`
- **故障排查**：`docs/zh/user-guide/troubleshooting/scheduler.md`
- **规格**：`.kiro/specs/production-deployment/design.md` §3.1、§3.2
