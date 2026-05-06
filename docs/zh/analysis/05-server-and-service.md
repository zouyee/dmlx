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
