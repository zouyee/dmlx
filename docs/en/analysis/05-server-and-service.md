# Chapter 5: Server and Service Layer

## 5.1 `server.zig` (1,517 lines)

OpenAI-compatible HTTP server, covering:

- **Concurrency model**: each connection processed concurrently via `io.async`
- **Engine Loop**: background async fiber drives the scheduler
  - schedule → batch → forward → postprocess
- **SSE streaming**: `text/event-stream` format, `data: {...}` events
- **Tool calling**: OpenAI functions format parsing and execution
- **Guided decoding**: integrates `GuidedDecoder`, supports request-level JSON Schema / Regex constraints

## 5.2 Server Configuration

```zig
ServerConfig{
    .kv_strategy = .paged_quantized,   // default paged quantized
    .kv_quant = .simple,                // quantization algorithm
    .kv_tier = .ram,                    // storage tier
    .speculative_ngram = null,          // speculative decoding n-gram size
    .smelt = false,                     // expert streaming
    .smelt_strategy = "preload",
    .distributed = false,               // distributed inference
}
```

## 5.3 Scheduler (`scheduler.zig`)

Three-stage loop:
1. **Schedule**: select requests from waiting/running queues, allocate KV cache blocks
2. **Forward**: execute model forward propagation + sampling
3. **Postprocess**: append tokens, check stop conditions, release blocks for completed requests

## 5.4 Active Issue: Batched Forward Not Complete

`batch_builder.zig` (256 lines) has implemented request merging logic:

```zig
// batch_builder.zig: merges multiple decode requests into a single batched input tensor
```

But `server.zig`'s `engineLoop` still processes decode per single request:

```zig
// TODO: batch_builder would merge all decode requests into a single forward pass
```

**Impact**:
- Maximum throughput potential of continuous batching is not unlocked
- Each request has an independent forward, GPU utilization is low
- This is the largest bottleneck for current inference throughput

**Fix effort**: estimated 1-2 weeks, requires integrating `batch_builder.buildBatch()` into the scheduler's forward phase.

## 5.5 Distributed Inference (`distributed.zig`, 222 lines)

Supports tensor parallelism across multiple Macs (MPI-style collective communication):

- `allSum` / `allGather` / `allMax` / `allMin`: collective reductions
- `send` / `recv`: point-to-point communication
- `sumScatter`: reduce-scatter gradient aggregation

### ⚠️ Resource Leak

```zig
pub fn deinit(self: *DistributedGroup) void {
    _ = self;
    // mlx_distributed_group has no explicit free in this mlx-c version
}
```

`deinit` is an empty implementation. Frequent creation/destruction of `DistributedGroup` causes resource leaks.
