# Design: HTTP Server & Service Layer

## Overview

The HTTP server (`src/server.zig`) provides an OpenAI-compatible REST API for
LLM inference. It bridges the Generation API (inference engine) with external
clients via HTTP/SSE, handles concurrent connections via async I/O, and
orchestrates model loading, KV cache management, and request scheduling.

**Scope:** L1 (architecture/interface) + L2 (algorithm/state machine).
L3 implementation details (JSON escaping, HTTP header parsing) are out of scope.

## Components and Interfaces

### 1. Server Lifecycle

```
start(allocator, io, config)
  └─> loadModel(allocator, io, config) → ModelState
        ├─> detectArchitecture(config.json) → arch_name
        ├─> ModelRegistry.getLoader(arch_name) → loader
        ├─> loader(allocator, config_json, model_path, ctx, stream, io, smelt) → ModelVTable
        ├─> load tokenizer.json
        ├─> create KV caches (per-layer, heterogeneous)
        ├─> optionally load prompt cache from disk
        ├─> init ModelPool (multi-model LRU, budget = 50% RAM)
        └─> init BlockManager
  └─> Scheduler.init(allocator, &block_manager, max_prefill_tokens)
  └─> io.async(engineLoop, .{ io, &state })       // background fiber
  └─> listen(0.0.0.0:port)
        └─> io.async(handleConnection, ...) // per-connection fiber
```

**Key invariant:** `Scheduler` holds a pointer to `ModelState.block_manager`,
so it MUST be initialized AFTER `loadModel` returns and `ModelState` has a
stable address (not on the stack).

### 2. Request Routing

| Endpoint | Method | Handler | Notes |
|----------|--------|---------|-------|
| `/v1/chat/completions` | POST | `generateChatCompletion` / `handleStreamingCompletion` | OpenAI-compatible; supports streaming |
| `/v1/messages` | POST | `handleAnthropicMessages` | Anthropic Messages API compatibility |
| `/health` | GET | `writeJsonResponse(200, {"status":"ok"})` | Liveness probe |

**Request parsing flow:**
1. Read raw HTTP request into 64KB buffer
2. Parse headers + body (Content-Length aware)
3. Parse JSON body into `ChatCompletionRequest` or `AnthropicRequest`
4. Route to handler based on path prefix

### 3. Generation Pipeline (per-request)

Both streaming and non-streaming paths share the same pipeline:

```
1. Parse request JSON
2. If tools provided → inject tool system prompt as first message
3. Build message list → apply chat template
4. Tokenize (template already includes special tokens, encode(add_bos=false))
5. Configure GenerateConfig (max_tokens, temperature, seed)
   <!-- DIVERGENCE: OpenAI API supports per-request top_k/top_p.
        Code currently uses server_config.top_k/top_p for all requests.
        **为准方**: 设计。ChatCompletionRequest 应添加 top_k/top_p 字段，
        让客户端能覆盖服务器默认值。代码 TODO。-->
6. Generate tokens (see §3.1)
7. Decode tokens → text
8. Check stop strings → truncate if matched
9. If tools provided → detect/parse tool calls → execute → format response
10. Format OpenAI-compatible (or Anthropic) JSON response
```

**Step 6 — Generation dispatch:**

```
if speculative_ngram configured:
    streamGenerateSpeculative(vtable, prompt, config, caches, ctx, PldDrafter, callback)
else if streaming:
    streamGenerate(vtable, prompt, config, caches, ctx, callback)
else:
    generate(vtable, prompt, config, caches, ctx) → []u32
```

### 4. SSE Streaming Protocol

**Connection lifecycle:**

```
Client          Server
  |                |
  |-- POST /v1/chat/completions (stream=true) -->
  |                |
  |<-- HTTP/1.1 200 (text/event-stream) --------|
  |<-- data: {"id":"...","choices":[{"delta":{"content":"Hello"}}]}\n\n
  |<-- data: [DONE]\n\n
  |                |
```

**Keep-alive mechanism:** During long prefill, a background fiber sends
`: keep-alive\n\n` every 5 seconds until the first token is generated.
This prevents proxy/read timeouts.

**Callback architecture (streaming):** Zig lacks closures, so the stream
callback uses a function-scoped struct with static variables:

```zig
const StreamState = struct {
    var s_allocator: std.mem.Allocator = undefined;
    var s_sse: *SSEWriter = undefined;
    var s_token_count: usize = 0;
    // ... other statics
    fn callback(token: u32, is_done: bool) void { ... }
};
```

### 5. Engine Loop (Scheduler Integration)

The `engineLoop` runs as a background async fiber and drives the scheduler:

```
while running:
    scheduled = scheduler.schedule() catch { sleep(10ms); continue }
    if scheduled.isEmpty(): sleep(1ms); continue

    for each decode_request:
        if tokens >= max_tokens: markComplete(); state = .done
        // TODO: batch_builder.merge → single forward pass

    for each prefill_request:
        if !hasPendingPrefill(): state = .decoding

    free(scheduled.prefill_requests)
    free(scheduled.decode_requests)
```

**Current state:** The engine loop manages request state transitions but does
NOT yet perform batched forward passes. Each request is processed individually
via the generation pipeline. True continuous batching requires:
1. `batch_builder` to merge decode requests into a batched input tensor
2. Batched `forward()` via `ModelVTable`
3. `scheduler.postprocess(outputs)` to distribute results

## Data Models

### ServerConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_path` | `[]const u8` | — | Path to model directory (contains config.json, weights) |
| `port` | `u16` | 8080 | HTTP listen port |
| `max_tokens` | `usize` | 256 | Default max generation length |
| `temperature` | `f32` | 0.8 | Sampling temperature |
| `top_k` | `usize` | 50 | Top-k sampling cutoff |
| `top_p` | `f32` | 1.0 | Nucleus sampling threshold |
| `memory_config` | `memory_mod.MemoryConfig` | `.{}` | Memory limit configuration |
| `max_kv_size` | `memory_mod.MaxKvSize` | `.auto` | Max KV cache sequence length |
| `kv_bits` | `u8` | 4 | KV quantization bit width |
| `kv_strategy` | `KvStrategy` | `.paged_quantized` | KV cache layout strategy |
| `kv_quant` | `KvQuant` | `.simple` | Quantization algorithm (simple / turbo) |

<!-- DIVERGENCE: kv_quant 字段存在于 ServerConfig 但 loadModel() 从未读取。
     设计意图是支持 simple（per-group scale+bias）和 turbo（Lloyd-Max codebook）
     两种量化路径。代码目前无论 kv_quant 设置何值都只走 simple 路径。
     **为准方**: 设计。代码需补充 turbo 分支。-->
| `kv_tier` | `KvTier` | `.ram` | Storage tier for KV cache |
| `kv_cold_dir` | `?[]const u8` | null | SSD cold storage directory |
| `prompt_cache_file` | `?[]const u8` | null | Path to load/save prompt cache |
| `speculative_ngram` | `?usize` | null | PLD n-gram size (enables speculative decoding) |
| `smelt` | `bool` | false | Enable SMoE layer-temperature loading |
| `smelt_experts` | `f32` | 1.0 | Fraction of experts to load (SMoE) |
| `allow_unsafe_tools` | `bool` | false | Allow shell execution in tool calls |

### ModelState (internal)

Runtime state held for the lifetime of the server process. Not exported;
defined as `const` (module-private).

| Field | Type | Description |
|-------|------|-------------|
| `allocator` | `std.mem.Allocator` | Allocator passed from main |
| `io` | `std.Io` | Async I/O handle |
| `ctx` | `EagerContext` | MLX eager evaluation context |
| `stream` | `c.c.mlx_stream` | MLX compute stream |
| `vtable` | `ModelVTable` | Polymorphic model interface (forward, deinit, config) |
| `tokenizer_strategy` | `root.tokenizer.TokenizerStrategy` | Encode/decode interface |
| `tokenizer_backend` | `root.tokenizer.BpeTokenizer` | Actual tokenizer instance |
| `chat_template` | `root.tokenizer.ChatTemplate` | Message formatting (DeepSeek-style default) |
| `caches` | `[]kvcache.KVCacheStrategy` | Per-layer KV cache (heterogeneous sizes possible) |
| `model_pool` | `?ModelPool` | Multi-model LRU cache (currently single-model) |
| `block_manager` | `?scheduler_mod.BlockManager` | KV block allocator for scheduler |
| `scheduler` | `?scheduler_mod.Scheduler` | Request queue manager (initialized after stable address) |
| `prefix_disk_cache` | `?prefix_disk_mod.PrefixDiskCache` | SSD spill for evicted KV blocks |
| `speculative_ngram` | `?usize` | PLD configuration (mirrors ServerConfig) |
| `prompt_cache_file` | `?[]const u8` | Prompt cache persist path |
| `running` | `bool` | Engine loop control flag |

### ChatCompletionRequest (internal, OpenAI-compatible)

Module-private request type. Fields and defaults:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `[]const u8` | — | Model identifier (for multi-model routing) |
| `messages` | `[]Message` | — | Chat history (role, content) |
| `stream` | `?bool` | `false` | Enable SSE streaming |
| `temperature` | `?f32` | `null` | Override default temperature |
| `max_tokens` | `?u32` | `null` | Override default max_tokens |
| `seed` | `?u64` | `null` | Deterministic sampling seed |
| `stop` | `?[]const []const u8` | `null` | Stop strings |
| `response_format` | `?ResponseFormat` | `null` | Guided decoding hint (parsed, not enforced) |
| `tools` | `?[]const ToolDefinition` | `null` | Function definitions for tool calling |

### AnthropicRequest (internal)

Module-private type for Anthropic Messages API compatibility.
Fields: `model`, `messages`, `max_tokens`, `temperature`, `stream`, `system`.

## Key Design Decisions

### KD-S1: Single-threaded async I/O (not multi-threaded)

The server uses `std.Io.async` fibers for concurrency, not OS threads.
Each connection and the engine loop run as independent fibers scheduled by
the Zig event loop (GCD on macOS, io_uring on Linux). This avoids thread
safety concerns in the MLX C API but limits true parallel forward passes.

**Consequence:** Continuous batching is cooperative, not parallel.
The scheduler picks which requests to advance, but forward passes are
serialized.

### KD-S2: Heterogeneous per-layer KV caches

Layers with `compress_ratio > 1` (CSA/HCA layers in DeepSeek V4) store
fewer effective tokens. Each layer's `max_seq_len` is divided by its
compression ratio:

```zig
layer_max_seq = if (compress_ratio > 1) effective_max_seq / compress_ratio
                else effective_max_seq;
```

This prevents over-allocation for compressed layers while maintaining
full capacity for standard layers.

### KD-S3: Prompt cache persistence

On startup: if `prompt_cache_file` exists, load cached KV states to skip
prefill for known prompts.
On shutdown: save current KV caches to disk (in `ModelState.deinit`).

This is a cold-start optimization, not a runtime cache.

### KD-S4: Tool calling as post-generation pattern matching

Tool calls are detected AFTER generation completes by scanning output text
for JSON objects containing `"name"` and `"arguments"` keys. This is a
heuristic, not a constrained decoder.

When tools are provided in the request, a system prompt describing available
functions is injected as the first message. The model is expected to emit
tool call JSON in its response.

## Boundaries

### Upstream: Model Registry

- **Interface:** `model_registry_mod.getLoader(arch_name) → ModelLoader`
- **Contract:** Loader returns a fully initialized `ModelVTable` with valid
  `config` (num_layers, num_kv_heads, head_dim, vocab_size, compress_ratios)
- **Server responsibility:** Detect architecture from config.json, call loader,
  manage vtable lifetime (deinit on shutdown)

### Upstream: Generation API

- **Interface:** `generate()`, `streamGenerate()`, `streamGenerateSpeculative()`
- **Contract:** Takes `ModelVTable`, prompt tokens, `GenerateConfig`, KV caches,
  `EagerContext`. Returns generated tokens (or calls callback per token).
- **Server responsibility:** Build `GenerateConfig` from request parameters,
  manage KV cache array, handle speculative drafter lifecycle.

### Downstream: Clients

- **OpenAI API:** `/v1/chat/completions` with SSE streaming
- **Anthropic API:** `/v1/messages` (non-streaming only)
- **Health:** `/health` for load balancer probes

### Internal: Scheduler (future integration)

The server creates a `Scheduler` and `BlockManager` but the engine loop is
currently a simplified placeholder. Full integration requires:
- `batch_builder` to construct batched inputs from scheduled requests
- Batched forward pass support in model implementations
- Per-request output routing in `scheduler.postprocess()`

## Dependencies

| Module | Usage |
|--------|-------|
| `generation` | `generate`, `streamGenerate`, `streamGenerateSpeculative`, `GenerateConfig` |
| `model_registry` | `getLoader`, architecture detection |
| `kvcache` | `createStandard`, `createPaged`, `createQuantized`, `createPagedQuantized`, `createTieredWithConfig` |
| `scheduler` | `Scheduler`, `BlockManager` (future batched serving) |
| `batch_builder` | Batched input construction — **imported but not integrated** |

<!-- DIVERGENCE: batch_builder_mod 已导入但 engineLoop 中从未调用。
     设计意图是 schedule → batch_builder.build() → batched forward → postprocess。
     代码当前仍逐个请求处理。
     **为准方**: 设计。engineLoop 需升级到调用 batch_builder。-->
| `tokenizer` | `BpeTokenizer`, `TokenizerStrategy`, `ChatTemplate`, `ChatMessage` |
| `tool_calling` | `detectFamily`, `buildToolSystemPrompt`, `parse` |
| `tool_executor` | `execute` (with `allow_unsafe_tools` gate) |
| `speculative` | `PldDrafter` (Prompt Lookup Decoding) |
| `prompt_cache` | `loadPromptCache`, `savePromptCache` |
| `memory` | `autoMaxKvSize`, `enforceMemoryLimit`, `MemoryConfig` |
| `prefix_disk` | `PrefixDiskCache` (SSD spill) |

## Known Divergences

<!-- DIVERGENCE: Engine loop is a simplified placeholder. Design intent:
     schedule → batch_builder.build() → batched forward → postprocess.
     Code reality: manages state transitions only; forward passes are per-request.
     **为准方**: 设计。engineLoop 需升级到完整调度循环。代码 TODO。 -->

<!-- DIVERGENCE: Multi-model routing via ModelPool is placeholder code.
     Design intent: parse `model` field from request → ModelPool.getOrLoad() →
     route to requested model. Code reality: all requests use the single model
     loaded at startup.
     **为准方**: 设计。ModelPool 已初始化但请求路由尚未接线。代码 TODO。 -->

<!-- DIVERGENCE: response_format (json_schema / regex) is parsed from request
     but not enforced during generation. Design intent: FSM-guided decoding
     constrains token sampling to match schema/regex. Code reality: parsed
     into ChatCompletionRequest.response_format but never passed to sampler.
     **为准方**: 设计。guided_mod 已导入但未集成到生成流水线。代码 TODO。 -->

<!-- DIVERGENCE: kv_quant field exists in ServerConfig but is never read.
     Design intent: support simple (per-group scale+bias) and turbo
     (Lloyd-Max codebook) quantization algorithms. Code reality: loadModel()
     ignores kv_quant; all paths use simple quantization regardless of setting.
     **为准方**: 设计。ServerConfig 保留该字段；loadModel() 需补充 turbo 分支。 -->
