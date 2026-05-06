# Continuous Batching — Design Document

> **Status**: Framework complete. Server integration pending (estimated 1-2 weeks).
> **Source files**: `src/scheduler.zig`, `src/batch_builder.zig`, `src/kvcache/paged.zig`, `src/kvcache/interface.zig`
> **Requirements**: R13.1, R13.2, R13.3

---

## 1. Architecture Overview

MLX-Zig's continuous batching architecture follows a **three-phase Engine Step loop**:

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

### Component Map

| Component | File | Lines | Role |
|-----------|------|-------|------|
| **Scheduler** | `src/scheduler.zig` | 387 | Request queue management, block allocation, stop condition checking |
| **BatchBuilder** | `src/batch_builder.zig` | 256 | Token concatenation, position IDs, causal attention mask |
| **BlockManager** | `src/scheduler.zig` (wrapper) + `src/kvcache/paged.zig` (real) | 73 + 1152 | KV cache block pool with CoW and prefix hashing |
| **KVCacheStrategy** | `src/kvcache/interface.zig` | 149 | Polymorphic filter/update interface for all cache strategies |
| **Server** | `src/server.zig` | 1517 | HTTP server, connection handling, SSE streaming |

---

## 2. Request Lifecycle

### 2.1 State Machine

```
  addRequest()
       │
       ▼
  ┌─────────┐   schedule()    ┌────────────┐  prefill    ┌──────────┐
  │ waiting │ ──────────────► │ prefilling  │ ──────────► │ decoding │
  └─────────┘                 └────────────┘  complete   └──────────┘
       ▲                           │                          │
       │   no blocks               │  chunked prefill          │ stop / max_tokens
       │                           │  (partial)               │
       │                           ▼                          ▼
       │                      stays in                   ┌──────┐
       └────────────────── prefilling                   │ done │
                                                        └──────┘
                                                           │
                                                    freeBlocks()
```

### 2.2 Request Struct (`src/scheduler.zig:108-187`)

```zig
pub const Request = struct {
    id: u64,                          // Unique request identifier
    prompt_tokens: []const u32,       // Original prompt token sequence
    generated_tokens: ArrayList(u32), // Generated output tokens
    state: RequestState,              // waiting | prefilling | decoding | done
    block_ids: ArrayList(usize),      // Allocated KV cache block IDs
    max_tokens: usize,                // Generation limit
    stop_tokens: []const u32,         // EOS / custom stop tokens
    prefill_offset: usize,            // Cursor for chunked prefill
    done: bool,                       // Completion flag (for external polling)
    result_tokens: ?[]const u32,      // Final output (set on completion)
};
```

### 2.3 Key Methods

| Method | Description |
|--------|-------------|
| `seqLen()` | Total sequence length = prefill_offset + generated_tokens.len |
| `isStopToken(token)` | Check if token matches any stop token |
| `hasPendingPrefill()` | True if prompt not fully consumed |
| `currentPrefillChunkLen(max)` | Size of next prefill chunk (capped) |
| `markComplete()` | Set result_tokens, done=true |
| `waitForCompletion(io)` | Blocking poll with 1ms sleep intervals |

---

## 3. Scheduler (`src/scheduler.zig`)

### 3.1 Design

```zig
pub const Scheduler = struct {
    allocator: std.mem.Allocator,
    waiting: ArrayList(*Request),       // FCFS queue
    running: ArrayList(*Request),       // Active requests
    block_manager: *BlockManager,       // KV cache block pool
    max_prefill_tokens: usize,          // Chunked prefill limit (default: 512)
    blocks_per_request: usize,          // Blocks per new request (default: 1)
};
```

### 3.2 `schedule()` — Request Selection

Priority order:
1. **Running requests** — always included first (guarantees decode continuity)
2. **Waiting requests** — promoted only if blocks available

```zig
pub fn schedule(self: *Scheduler) !ScheduleResult {
    // Phase 1: Categorize running requests
    for (self.running.items) |req| {
        if (req.state == .prefilling and req.hasPendingPrefill())
            prefill_list.append(req)   // Continuing chunked prefill
        else
            decode_list.append(req)    // Decoding
    }

    // Phase 2: Promote waiting requests if blocks available
    for (self.waiting.items) |req| {
        if (self.block_manager.canAllocate(self.blocks_per_request)) {
            self.block_manager.allocateBlocks(allocator, req, ...);
            req.state = .prefilling;
            prefill_list.append(req);
            self.running.append(req);
        } else {
            still_waiting.append(req); // Keep waiting
        }
    }

    return ScheduleResult{
        .prefill_requests = prefill_list,
        .decode_requests = decode_list,
        .blocks_needed = blocks_needed,
    };
}
```

### 3.3 `postprocess()` — Output Handling + Cleanup

```zig
pub fn postprocess(self: *Scheduler, outputs: []const TokenOutput) !void {
    // Phase 1: Apply generated tokens
    for (outputs) |output| {
        if (req.state == .prefilling) {
            // Advance prefill_offset by chunk size
            req.prefill_offset += chunk_len;
            if (!req.hasPendingPrefill()) {
                req.state = .decoding;           // Transition to decode
                req.generated_tokens.append(output.token); // First gen token
            }
        } else {
            // Decoding: append token, check stop
            req.generated_tokens.append(output.token);
            if (req.isStopToken(token) or at_max) req.state = .done;
        }
    }

    // Phase 2: Remove completed requests, free KV cache blocks
    while (i < self.running.items.len) {
        if (req.state == .done) {
            self.block_manager.freeBlocks(req);
            _ = self.running.orderedRemove(i);
        } else { i += 1; }
    }
}
```

### 3.4 Engine Step Integration

The `engineLoop` in `server.zig:185-229` should implement this cycle:

```
1. scheduled = scheduler.schedule()
2. batch    = batch_builder.build(&scheduled, ctx)
3. logits   = model.forward(batch.batched_tokens, batch.position_ids, batch.attention_mask, caches)
4. tokens   = sample(logits) → []TokenOutput
5. outputs  = scheduler.postprocess(tokens)
6. for each finished request → SSE stream result to client
```

> **Current status**: Step 2-3 are not wired up. The engineLoop stub only does state transitions without actual forward passes. See `server.zig:211-216`.

---

## 4. BatchBuilder (`src/batch_builder.zig`)

### 4.1 Design

Converts a `ScheduleResult` (categorized prefill + decode requests) into a single batched input tensor suitable for one model forward pass.

### 4.2 BatchResult

```zig
pub const BatchResult = struct {
    batched_tokens: Array,     // [total_tokens] — concatenated token IDs
    position_ids: Array,       // [total_tokens] — per-token position indices
    attention_mask: Array,     // [total_tokens, total_tokens] — 0.0=attend, -inf=block
    total_tokens: usize,       // Total tokens in the batch
    num_requests: usize,       // Number of requests in the batch
};
```

### 4.3 Token Selection Logic

```zig
fn getRequestTokens(req: *const Request) []const u32 {
    switch (req.state) {
        .prefilling, .waiting =>
            // Return prompt tokens from prefill_offset to end
            return req.prompt_tokens[req.prefill_offset..],
        .decoding =>
            // Return single token: last generated (or prompt tail if none yet)
            return &[_]u32{ req.generated_tokens[last] },
        .done =>
            return &[_]u32{},
    }
}
```

### 4.4 Attention Mask Construction

Each request's tokens form a contiguous segment in the batch. The mask ensures:

- **Within request** (`R13.2`): Causal attention — token at position `row` can attend to positions `0..row`
- **Across requests**: Fully blocked — tokens from different requests never attend to each other

```
Mask for batch with 3 requests (seq lens: 3, 2, 2):
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

Legend: 0 = attend (0.0), - = blocked (-inf)
```

### 4.5 Position IDs

- **Prefill requests**: `prefill_offset + j` (absolute positions in the sequence)
- **Decode requests**: `seqLen() - 1` (the current position)

---

## 5. Chunked Prefill

### 5.1 Motivation

Large prompts (> `max_prefill_tokens`) are split into chunks to prevent:
- Decode request starvation (decode steps interleave with prefill chunks)
- Memory spikes (each chunk uses bounded KV cache space)

### 5.2 Flow

```
Step N:   schedule() → prefill_list: [req_A(chunk_0)], decode_list: [req_B, req_C]
          forward pass → processes chunk_0 of req_A + decodes req_B, req_C
          postprocess() → req_A.prefill_offset += 512, stays prefilling

Step N+1: schedule() → prefill_list: [req_A(chunk_1)], decode_list: [req_B, req_C]
          forward pass → processes chunk_1 of req_A + decodes req_B, req_C
          postprocess() → req_A.prefill_offset += 512, stays prefilling

...       (repeats until prompt fully consumed)

Step N+K: schedule() → prefill_list: [req_A(final_chunk)], decode_list: [req_B, req_C]
          forward pass → processes final chunk
          postprocess() → req_A transitions to .decoding
```

### 5.3 Key Invariants

- Decode requests are **never blocked** by prefill — they are always included in each step
- A prefilling request stays in `running` across multiple steps
- `prefill_offset` is the cursor that advances by `max_prefill_tokens` each step
- The transition to `.decoding` happens only when `prefill_offset >= prompt_tokens.len`

---

## 6. BlockManager — KV Cache Block Allocation

### 6.1 Interface (`src/scheduler.zig:22-95`)

```zig
pub const BlockManager = struct {
    total_blocks: usize,
    used_blocks: usize,
    real: ?*RealBlockManager,     // Backs real paged KV cache (CoW, prefix hashing)

    pub fn init(total_blocks: usize) BlockManager;
    pub fn canAllocate(num_blocks: usize) bool;
    pub fn allocateBlocks(allocator, req: *Request, num_blocks: usize) !void;
    pub fn freeBlocks(req: *Request) void;
    pub fn freeCount() usize;
};
```

### 6.2 Real BlockManager (`src/kvcache/paged.zig:44`)

The real `BlockManager` manages a pool of `Block` structs with:

| Feature | Description |
|---------|-------------|
| **Copy-on-Write** | Shared blocks (`ref_count > 1`) are copied before modification |
| **Prefix hashing** | Wyhash-based rolling hashes for cache hit detection |
| **Per-request mapping** | `req_to_blocks: HashMap(u64, ArrayList(usize))` |
| **Block hashing** | `block_hashes: HashMap(u64, usize)` for lookup |
| **Access tracking** | `last_access` timestamp for LRU eviction |
| **Default page size** | 32 tokens (tuned for Apple Silicon Metal GPU alignment) |

### 6.3 Block Lifecycle

```
allocateBlocks(req_id, N):
  1. Pop N blocks from free_blocks pool
  2. Add to req_to_blocks[req_id]
  3. used_blocks += N

freeBlocks(req_id):
  1. For each block in req_to_blocks[req_id]:
     a. ref_count -= 1
     b. If ref_count == 0: push back to free_blocks
  2. Remove req_id from req_to_blocks
```

---

## 7. KV Cache Strategies for Continuous Batching

### 7.1 VTable Interface (`src/kvcache/interface.zig`)

```zig
pub const VTable = struct {
    updateAndFetch: *const fn (ctx, keys, values, stream) anyerror!KVSlice,
    currentLen:    *const fn (ctx) usize,
    reset:         *const fn (ctx) void,
    filter:        ?*const fn (ctx, indices, allocator) anyerror!void, // ← CB critical
    rollback:      ?*const fn (ctx, to_len) void,
    deinit:        *const fn (ctx, allocator) void,
};
```

### 7.2 Strategy Support Matrix

| Strategy | `filter` | Use Case |
|----------|----------|----------|
| **Paged** | ✅ Supported | **Default for continuous batching** |
| **PagedQuantized** | ✅ Supported | Memory-constrained CB |
| **Tiered** | ✅ Supported (hot tier = Paged) | Long-context CB with SSD spill |
| **Quantized** | ✅ Supported | Compact CB |
| **Standard** | ✅ Supported | Simple single-request |
| **Rotating** | ❌ Not supported | Fixed-window, no dynamic batch |

### 7.3 `filter()` — Dynamic Batch Resize

When a request completes (`.done`), `filter()` is called to remove its KV cache entries:

```zig
// Remove completed requests from KV cache
cache.filter(&[_]usize{0, 2}, allocator) // Keep batch indices 0 and 2
```

This ensures the KV cache size matches the active batch size after each `postprocess()`.

---

## 8. Server Integration Blueprint

### 8.1 Current State

```
POST /v1/chat/completions
  → handleStreamingCompletion()     // Serial: one request at a time
  → generateStep() loop             // No batch builder used
```

### 8.2 Target State

```
POST /v1/chat/completions
  → state.scheduler.addRequest(req) // Enqueue (non-blocking)
  → req.waitForCompletion(io)       // Poll until done (in connection fiber)

engineLoop (async fiber):
  while (state.running):
    scheduled = scheduler.schedule()
    if scheduled.isEmpty(): sleep(1ms); continue

    batch = batch_builder.build(&scheduled, ctx)
    logits = model.forward(batch.batched_tokens, batch.position_ids,
                           batch.attention_mask, caches)
    tokens = sample(logits, scheduled)
    scheduler.postprocess(tokens)

    for finished requests:
      SSE stream result to connection
      cache.filter(remaining_indices)  // Remove completed entries
```

### 8.3 Integration Steps (from `server.zig:530-540`)

1. Create `scheduler_mod.Request` from parsed chat completion
2. `state.scheduler.addRequest(&req)` — enqueue
3. In engine loop: `scheduler.schedule()` to select requests
4. `batch_builder.build()` to create batched input tensors
5. `state.vtable.forward(...)` — single forward pass for all requests
6. `scheduler.postprocess(outputs)` — apply tokens, check stop conditions
7. SSE stream results back to each request's connection

---

## 9. Test Coverage

### 9.1 Batch Builder Tests (`src/tests/batch_builder_tests.zig`)

| Test | Validates | Status |
|------|-----------|--------|
| `single prefill request` | Token concatenation, attention isolation | ✅ Passing |
| `single decode request` | Single-token extraction | ✅ Passing |
| `mixed prefill and decode` | R13.2 — attention isolation across states | ✅ Passing |
| `multiple decode requests` | Multi-request decode batching | ✅ Passing |
| `new request joins without waiting` | **R13.3** — continuous batching join | ✅ Passing |
| `tensor shapes are correct` | Output dimensions, position IDs | ✅ Passing |
| `attention mask structure` | Causal within request, blocked across | ✅ Passing |

### 9.2 Integration Tests (`src/tests/integration_tests.zig`)

| Test | Validates | Status |
|------|-----------|--------|
| `submit → schedule → batch → postprocess` | Full flow without forward pass | ✅ Passing |
| `multiple concurrent requests with CB` | R13 — multi-request batching | ✅ Passing |
| `new request joins existing decode batch` | R13.3 — mid-generation join | ✅ Passing |
| `chunked prefill with concurrent decode` | Chunked prefill + decode interleaving | ✅ Passing |

---

## 10. Known Gaps

| Gap | Impact | Estimated Effort |
|-----|--------|-----------------|
| Server engineLoop not wired to BatchBuilder | No true batch forward — each request serial | **1-2 weeks** |
| No per-request RoPE delta tracking | mRoPE models (DeepSeek V4) require per-sequence position offsets | 2-3 days |
| No external prefill | Large prompts processed inside batch, may cause TTFT spikes | 3-5 days |
| No memory-aware scheduling | No OOM guard for concurrent requests | 2-3 days |
| SSE streaming not integrated with scheduler output | Results need to route back to correct connection fiber | 2-3 days |

---

## 11. Comparison with oMLX

| Dimension | oMLX | mlx-zig (current) | mlx-zig (target) |
|-----------|------|-------------------|------------------|
| **Language** | Python | Zig | Zig |
| **Batch engine** | mlx-lm `BatchGenerator` | Custom `BatchBuilder` | Same |
| **Scheduler** | `Scheduler` (4322 lines) | `Scheduler` (387 lines) | Same |
| **Queue model** | `waiting` deque + `running` dict | `waiting` ArrayList + `running` ArrayList | Same |
| **Batch insert** | `batch_generator.insert()` | `batch_builder.build()` | Same |
| **Batch remove** | `batch_generator.remove()` | `KVCacheStrategy.filter()` | Same |
| **Chunked prefill** | `prefill_step_size=2048` | `max_prefill_tokens=512` | Same |
| **Paged KV** | `PagedCacheManager` + CoW | `BlockManager` + CoW | Same |
| **External prefill** | ✅ `_do_external_prefill()` | ❌ Not implemented | Needed |
| **Speculative prefill** | ✅ `specprefill` with draft model | ❌ Not implemented | Future |
| **Server integration** | ✅ Fully integrated | ❌ Stub engineLoop | 1-2 weeks |
| **Concurrent connections** | FastAPI + asyncio | `io.async` fibers | Mostly ready |

---

## 12. References

- **Source**: `src/scheduler.zig`, `src/batch_builder.zig`, `src/kvcache/paged.zig`, `src/server.zig`
- **Tests**: `src/tests/batch_builder_tests.zig`, `src/tests/integration_tests.zig`
- **Analysis**: `docs/en/analysis/05-server-and-service.md`, `docs/en/analysis/04-kv-cache-subsystem.md`
- **Troubleshooting**: `docs/en/user-guide/troubleshooting/scheduler.md`
- **Specs**: `.kiro/specs/production-deployment/design.md` §3.1, §3.2
