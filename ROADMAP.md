# DMLX Project Roadmap

> Based on deep audit of the dmlx codebase and systematic analysis of five projects: vLLM,
> mlx-lm, oMLX, TileKernels, and mlx-rs. Defines the complete path from current state to
> production-grade deployment.

### Progress Tracking (Updated 2026-04-26)

> **All Phase 0–5 + Integration + Gap Fixes + Phase 7 (P0-P3) tasks completed.** 350 tests passing.

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0: Foundation Fixes | ✅ Complete | Error handling, memory safety (ScopedArrayArena), NN layer GPU migration, build system pkg-config |
| Phase 1: Inference Engine | ✅ Complete | Three-tier generation API, model registry (5 architectures), Prompt Cache persistence, op fusion |
| Phase 2: Service Layer | ✅ Complete | Scheduler, PagedAttention (CoW + prefix hash), KV quantization 4/8/16-bit, SSE streaming, Continuous Batching |
| Phase 3: Advanced Inference | ✅ Complete | Chunked Prefill, Prefix Caching, Speculative Decoding (n-gram), Guided Decoding (FSM) |
| Phase 4: Quantization & Training | ✅ Complete | Weight quantization (affine + MXFP4), QLoRA, MoE Router, FP8/FP4 bindings |
| Phase 5: Production Operations | ✅ Complete | ModelPool LRU, Tiered KV Cache (RAM+SSD), memory limits, autoMaxKvSize, Benchmark |
| Integration Wiring (Task 13) | ✅ Complete | 11 subtasks: KV strategy selection, Scheduler, Generation API, Prompt Cache, Speculative/Guided Decoding, ModelPool, Tiered Cache, CLI subcommands, op fusion, MoE Router |
| V4 Gaps (Task 15) | ✅ Complete | Paged+Quantized combined, learned pooling, FP8 native KV storage, Lightning Indexer, Attention Sink, heterogeneous KV Cache, on-disk prefix reuse, TurboQuant |
| Integration Fixes (Task 17) | ✅ Complete | Prefix hash auto-registration, Prompt Cache save integration, PrefixDiskCache server integration |
| Concurrent Serving (Task 19) | ✅ Complete | io.async concurrent connections, Request completion notification, engine loop background fiber, Scheduler-driven |
| Quantized Model Loading (Task 21) | ✅ Complete | Quantized weight detection, quantizedMatmul forward, multi-shard loading, lm_head quantization, end-to-end verification |
| Quant Mode Full Alignment (Task 23) | ✅ Complete | nvfp4/mxfp8 enums, LightningIndexer upgraded to true MXFP4 quantization, 341 tests passing |
| Memory Leak Fixes + Mixed Quant (Task 25) | ✅ Complete | QuantizedWeight shape leak fix, V4 per-weight quant config parsing |
| Phase 7 P0: Core Usability (Task 27) | ✅ Complete | Streaming token output, EOS stopping, Chat template, generate performance fix |
| Phase 7 P1: Model Expansion (Task 28-30) | ✅ Complete | Gemma/Phi-3 loader, batch forward, token decode integration |
| Phase 7 P2: API Completeness (Task 31-33) | ✅ Complete | stop/logprobs/tool_calls, repetition penalty, min_p, SSE keep-alive, EAGLE |
| Phase 7 P3: Ecosystem Tools (Task 34) | ✅ Complete | Custom Metal kernel, distributed inference, model conversion, perplexity evaluation |

>
> References: `deep-analysis.md` (code audit), `ecosystem-analysis.md` (mlx-lm/oMLX/mlx-rs),
> `tilekernels-analysis.md` (TileKernels), `deepseek-v4-turboquant-analysis.md` (V4+TurboQuant paper).

---

## 1. DMLX Current State Overview (Updated 2026-04-26)

### Existing Capabilities

- 200+ mlx-c operator bindings (math/shape/reduce/linalg/fft/conv/random/creation)
- `fast.zig` binding MLX fused kernels (rms_norm/rope/sdpa/layer_norm)
- LLaMA + DeepSeek V4 model architectures + model registry (9 architectures: LLaMA/Mistral/Qwen2/Qwen3/Gemma/GLM-4/Phi/Phi-3/DeepSeek V4)
- KV Cache 6 strategies (Standard/Rotating/Quantized/Paged/PagedQuantized/Tiered) + TurboQuant
- Quantization: affine 4/8-bit + MXFP4 + NVFP4 + MXFP8 + FP8 (E4M3) + TurboQuant (Lloyd-Max + QJL)
- Quantized model loading: direct loading of mlx-lm 4-bit/8-bit quantized models (packed uint32 + scales + biases)
- mlx-lm quantized model loading: auto-detect .scales/.biases suffixes, quantizedMatmul forward path, multi-shard safetensors merging
- LoRA/QLoRA adapters + AdamW optimizer + SFT Trainer
- BPE Tokenizer + HuggingFace config parsing + Safetensors I/O
- CLI (chat/serve/benchmark/quantize) + OpenAI-compatible HTTP server (SSE streaming)
- Scheduler + Continuous Batching + Chunked Prefill
- Speculative Decoding (n-gram) + Guided Decoding (FSM)
- ModelPool (LRU eviction) + process memory limits + autoMaxKvSize
- Tiered KV Cache (Hot RAM + Cold SSD) + PrefixDiskCache (on-disk prefix reuse)
- Prompt Cache persistence (save/load safetensors)
- DeepSeek V4 full support: CSA/HCA compression, Lightning Indexer, Attention Sink, FP8 KV storage, mHC

### Resolved Core Defects

| Category | Original Issue | Solution | Status |
|----------|---------------|----------|--------|
| Performance | NN layers bypassed GPU | All migrated to mlx-c op chains + fast.zig fused kernels | ✅ |
| Memory | Intermediate Array leaks | ScopedArrayArena batch release | ✅ |
| Errors | No context in error messages | `mlxErrorHandler` captures C++ exception text | ✅ |
| Service | Single-threaded blocking | Scheduler + Continuous Batching + SSE streaming | ✅ |
| Safety | `@constCast` bypassing CoW | All replaced with mlx-c op chains | ✅ |
| Testing | Zero coverage | 350 tests (21 property tests) | ✅ |
| Build | Hardcoded paths | pkg-config + `-Dmlx_prefix` + pinned dependency versions | ✅ |

---

## 2. Key Architecture Lessons from External Projects

> **Note**: The "dmlx counterpart" descriptions below describe the original state before
> implementation. All mentioned missing features have been implemented in Phase 0-5 + Task 13-22.
> Current state is reflected in the progress tracking table above.

### vLLM — The Gold Standard for Industrial Inference Engines

vLLM is the most mature LLM inference engine. Its architecture defines the standard for production-grade inference.
Below are the core mechanisms dmlx must understand and leverage:

#### PagedAttention

vLLM's core innovation. KV cache no longer allocates contiguous memory per request,
but divides into fixed-size blocks (default 16 tokens/block), allocated on-demand from `free_block_queue`.

```
block_size = 2 * block_size_tokens * num_kv_heads * head_size * dtype_bytes
```

- Memory waste reduced from 60-80% to < 4%
- Supports Copy-on-Write (multiple requests share blocks during prefix sharing)
- Blocks can be reclaimed and reused by new requests

**dmlx counterpart**: `kvcache/paged.zig` is the skeleton; full block alloc/free/CoW needed.

#### Continuous Batching

All sequences are flattened and concatenated into a "super-sequence", with position indices and attention masks
ensuring each sequence only attends to its own tokens. New requests can join at any engine step.

```
[seq1_tok1, seq1_tok2, seq2_tok1, seq3_tok1, seq3_tok2, seq3_tok3]
 ← 3 sequences of different lengths concatenated →
```

**dmlx counterpart**: `server.zig` processes requests serially; needs a batch scheduler.

#### Scheduler Three-Phase Loop

Each engine step:
1. **Schedule** — Select requests from waiting/running queues, allocate KV cache blocks
2. **Forward** — Execute model forward propagation + sampling
3. **Postprocess** — Append tokens, check stop conditions, release blocks for completed requests

**dmlx counterpart**: Current `generate` is a single-request loop; needs refactoring to scheduler-driven.

#### Chunked Prefill

Long prompt prefill split into multiple chunks (e.g., 8 tokens each),
preventing a single long request from monopolizing engine steps and reducing latency for other requests.

**dmlx counterpart**: No equivalent mechanism currently.

#### Prefix Caching

Split prompt tokens into blocks by block_size, compute hash per block.
Requests with matching prefixes can reuse already-computed KV cache blocks, skipping redundant prefill.

```
hash(block) = hash(prev_block_hash, token_ids, metadata)
```

**dmlx counterpart**: `kvcache/radix.zig` has skeleton; hash-based lookup not yet implemented.

#### Speculative Decoding

Use a small model (or n-gram) to propose k tokens, verified in one pass by the large model.
Accept/reject guarantees statistical equivalence to token-by-token sampling.

vLLM V1 supports: n-gram, EAGLE, Medusa — three draft methods.

**dmlx counterpart**: Completely missing.

#### Guided Decoding

Use FSM (Finite State Machine) to mask logits at each step,
ensuring output conforms to specified grammar (JSON schema, regex, etc.).

**dmlx counterpart**: Completely missing.

### mlx-lm — The MLX Ecosystem Reference Implementation

- **50+ model architecture** registry vs dmlx's 2
- **Three-tier generation architecture**: `generate_step` → `stream_generate` → `generate`
- **Prompt cache persistence** to safetensors
- **Quantization suite**: GPTQ / AWQ / DWQ
- **Complete CLI**: generate / chat / convert / evaluate / benchmark / manage / share

### oMLX — Apple Silicon Production-Grade Server

- **Tiered KV Cache**: Hot RAM + Cold SSD (safetensors format offload)
- **Continuous Batching**: Based on mlx-lm BatchGenerator
- **Multi-model management**: EnginePool + LRU eviction + TTL + model pinning
- **Process memory limits**: `--max-process-memory`
- **Claude Code optimizations**: context scaling + SSE keep-alive

### TileKernels — Operator Fusion and Quantization

- **Extreme operator fusion**: SwiGLU + quantization in a single kernel
- **Complete quantization stack**: FP8/FP4/E5M6, per-token/per-block/per-channel
- **MoE routing pipeline**: topk → expand → compute → reduce
- **Reference implementation + numerical verification** engineering discipline

### mlx-rs — Lessons from a Peer Project

- **Autodiff explicit input rules** documented
- **Feature flags** controlling Metal/Accelerate backends

---

## 3. Production-Grade Deployment Path

### Phase 0: Foundation Fixes (Blocks all subsequent work)

> Without fixing these issues, any feature development is building on sand.

#### 0.1 Error Handling (1 day)

```zig
// c.zig — integrate mlx_get_last_error
pub fn check(rc: c_int) !void {
    if (rc != 0) {
        const msg = c.mlx_get_last_error();
        std.log.err("MLX error: {s}", .{std.mem.span(msg)});
        return error.MlxError;
    }
}
```

#### 0.2 Memory Safety (1 week)

- Introduce ArenaAllocator for forward pass, batch-release intermediate Arrays
- Remove `dataSliceMut`'s `@constCast`, migrate all data operations to mlx-c operators
- Add `deinit` to EagerContext

#### 0.3 NN Layer GPU Migration (1 week)

This is the critical step for 100-1000x performance improvement:

| Current Implementation | Replace With |
|------------------------|--------------|
| `nn.RMSNorm.forward` — CPU scalar loops | `fast.rmsNorm()` |
| `nn.RoPE.apply` — CPU scalar loops | `fast.rope()` |
| `nn.MultiHeadAttention` — CPU scalar loops | `fast.scaledDotProductAttention()` |
| `nn.Embedding.forward` — CPU scalar loops | `mlx_take` |
| `activations.gelu` etc. 21 functions — CPU scalar loops | mlx-c operator chains |
| `loss.mseLoss` etc. 9 functions — CPU scalar loops | Refer to `crossEntropyGraph` graph mode |
| `nn.LSTM/GRU/RNN` — CPU scalar loops | mlx-c matmul + sigmoid + tanh operator chains |

#### 0.4 Build System (2 days)

- Eliminate hardcoded `/opt/homebrew` paths, use pkg-config + `-Dmlx_prefix`
- Pin zig-regex dependency version
- Deduplicate build.zig

### Phase 1: Inference Engine Refactoring (Based on vLLM + mlx-lm)

> Goal: From "it runs" to "it runs well".

#### 1.1 Three-Tier Generation Architecture (Based on mlx-lm)

```zig
// Single-step generation — core primitive
pub fn generateStep(model: *Model, tokens: Array, cache: []KVCache) !Token { ... }

// Streaming generation — callback per token
pub fn streamGenerate(model: *Model, prompt: []u32, config: GenConfig,
    callback: fn(Token) void) !void { ... }

// Full generation — return all tokens
pub fn generate(model: *Model, prompt: []u32, config: GenConfig) ![]u32 { ... }
```

#### 1.2 Model Architecture Registry (Based on mlx-lm)

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

Priority additions: Mistral, Qwen2/3, Gemma (covers 90% of usage scenarios).

#### 1.3 Prompt Cache Persistence (Based on mlx-lm + oMLX)

Leverage existing safetensors I/O to serialize KV cache to disk:

```zig
pub fn savePromptCache(cache: []KVCache, path: []const u8) !void { ... }
pub fn loadPromptCache(path: []const u8) ![]KVCache { ... }
```

#### 1.4 Operator Fusion (Based on TileKernels + MLX compile)

```zig
// Compile SwiGLU MLP into fused operation
const swiglu_fn = try Closure.init(swiGluForward, allocator);
const compiled_swiglu = try compile.compile(swiglu_fn, false);

// Compile AdamW.step into fused operation (eliminates ~15 temporary Arrays per step)
const adamw_fn = try Closure.init(adamwStep, allocator);
const compiled_adamw = try compile.compile(adamw_fn, false);
```

### Phase 2: Service Layer Refactoring (Based on vLLM + oMLX)

> Goal: From "it responds" to "it serves".

#### 2.1 Scheduler (Based on vLLM)

Implement vLLM's three-phase engine step:

```zig
pub const Scheduler = struct {
    waiting: Queue(Request),
    running: Queue(Request),
    kv_cache_manager: KVCacheManager,

    pub fn schedule(self: *Scheduler) ScheduleResult {
        // 1. Prioritize running queue (decode requests)
        // 2. Process waiting queue (prefill requests)
        // 3. Allocate KV cache blocks
    }
};

pub fn engineStep(scheduler: *Scheduler, model: *Model) ![]Output {
    const scheduled = scheduler.schedule();
    const outputs = try model.forward(scheduled.batch);
    return try postprocess(outputs, scheduler);
}
```

#### 2.2 Full PagedAttention Implementation (Based on vLLM)

```zig
pub const BlockManager = struct {
    free_blocks: DoublyLinkedList(Block),
    req_to_blocks: HashMap(RequestId, []Block),
    cached_block_hash: HashMap(BlockHash, *Block),

    pub fn allocateSlots(self: *BlockManager, req: *Request, num_tokens: usize) !void { ... }
    pub fn freeBlocks(self: *BlockManager, req_id: RequestId) void { ... }
};
```

#### 2.5 KV Cache Quantization (kv_bits=4/8) (P1, from Apple Silicon optimization analysis)

Apple Silicon M3/M4 has dedicated optimizations for small-bitwidth operations. KV Cache quantization is
a hard prerequisite for long-context inference. An 8B model at 32K context with float16 KV Cache is ~8GB;
4-bit quantization reduces to 2GB.

```zig
pub const KVQuantConfig = struct {
    kv_bits: u8 = 16,          // 4, 8, or 16 (no quantization)
    group_size: i32 = 64,      // Quantization group size
};

// Insert quantization into KV Cache update path
pub fn updateQuantized(self: *KVCache, key: Array, value: Array, config: KVQuantConfig) !void {
    if (config.kv_bits < 16) {
        const q_key = try quantize(ctx, key, config.kv_bits, config.group_size);
        const q_value = try quantize(ctx, value, config.kv_bits, config.group_size);
        // Store quantized KV
    }
    // Dequantize before attention computation
}
```

Naturally compatible with PagedAttention: block-level quantization can decide precision at allocation time.
Leverages existing `mlx_quantize`/`mlx_dequantize` from mlx-c.

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

All sequences flattened and concatenated, differentiated by position indices:

```zig
// 3 requests: lengths 2, 1, 3
// Concatenated: [tok1_1, tok1_2, tok2_1, tok3_1, tok3_2, tok3_3]
// positions: [0, 1, 0, 0, 1, 2]
```

### Phase 3: Advanced Inference Features (Based on vLLM)

#### 3.1 Chunked Prefill

Long prompts processed in chunks, max `max_prefill_tokens` tokens per step:

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

Prioritize n-gram implementation (simplest, no additional model needed):

```zig
pub fn ngramPropose(context: []const u32, k: usize) ?[]const u32 {
    // Search generated context for matches to the last n tokens
    // Return k tokens after the match position as draft
}
```

#### 3.4 Guided Decoding

Constrain output format via logits mask:

```zig
pub fn applyGrammarMask(logits: Array, allowed_tokens: []const u32) !Array {
    // Set logit of disallowed tokens to -inf
}
```

### Phase 4: Quantization & Training (Based on TileKernels + mlx-lm)

#### 4.1 Quantization Infrastructure

```zig
pub const QuantConfig = struct {
    format: enum { int4_nf4, int8, fp8_e4m3 },
    group_size: i32 = 64,
    bits: u8 = 4,
};

// Bind mlx-c
pub fn quantize(ctx: EagerContext, weight: Array, config: QuantConfig) !struct { quantized: Array, scales: Array } { ... }
pub fn dequantize(ctx: EagerContext, quantized: Array, scales: Array, config: QuantConfig) !Array { ... }
```

#### 4.2 QLoRA

- 4-bit NormalFloat quantization (NF4)
- Double Quantization

#### 4.3 MoE Routing (Based on TileKernels)

```zig
pub fn moeRoute(ctx: EagerContext, scores: Array, num_topk: i32) !MoeRouteResult { ... }
pub fn moeExpand(ctx: EagerContext, x: Array, mapping: MoeRouteResult) !Array { ... }
pub fn moeReduce(ctx: EagerContext, expanded: Array, weights: Array, mapping: MoeRouteResult) !Array { ... }
```

### Phase 5: Production Operations (Based on oMLX)

#### 5.1 Multi-Model Management

```zig
pub const ModelPool = struct {
    models: HashMap([]const u8, *LoadedModel),
    lru: LRUCache([]const u8),
    max_memory: usize,

    pub fn getOrLoad(self: *ModelPool, name: []const u8) !*LoadedModel { ... }
    pub fn evictLRU(self: *ModelPool) void { ... }
};
```

#### 5.2 Tiered KV Cache (Based on oMLX)

```
Hot Tier (RAM) ←→ Cold Tier (SSD, safetensors)
    ↑ write-back        ↑ restore on cache hit
```

#### 5.3 Process Memory Limits

```zig
pub fn enforceMemoryLimit(max_bytes: usize) void {
    const current = getProcessMemoryUsage();
    if (current > max_bytes) {
        // Trigger LRU eviction
    }
}
```

#### 5.4 max_kv_size Auto-Configuration (P2, from Apple Silicon optimization analysis)

Automatically recommend KV Cache limit based on device memory, avoiding manual tuning that leads to OOM or waste.

```zig
pub fn autoMaxKvSize(model_bytes: usize, kv_bits: u8) usize {
    const total_ram = getSystemMemoryBytes();
    const available = total_ram - model_bytes - safety_margin;
    const bytes_per_token_kv = 2 * num_kv_heads * head_dim * (kv_bits / 8) * num_layers;
    return available / bytes_per_token_kv;
}
```

| Device RAM | kv_bits=4 suggested max_kv_size | Typical Scenario |
|------------|-------------------------------|------------------|
| 16 GB | 8,192 – 16,384 | Basic conversation |
| 64 GB | 32,768 – 65,536 | Code writing, document analysis |
| 128 GB+ | 131,072+ | RAG, ultra-long context |

CLI exposed as `--max-kv-size auto` (default) or `--max-kv-size 32768` (manual).

#### 5.4 Benchmark Tool (Based on vLLM + mlx-lm)

```bash
dmlx benchmark --model <path> --input-tokens 32 --output-tokens 128
# Output: TTFT, ITL, throughput, memory usage
```

---

## 4. Testing & Verification System (Based on TileKernels + vLLM)

### Numerical Verification

```zig
// Based on TileKernels testing/numeric.py
fn calcDiff(x: []const f32, y: []const f32) f64 {
    // Cosine similarity: 1 - 2*dot(x,y) / (dot(x,x) + dot(y,y))
}

fn checkBias(x: []const f32, ref: []const f32) !void {
    // Statistical test: whether quantization error is unbiased
}
```

### Golden Tests

1. Python MLX generates reference outputs (each NN layer, each activation, each loss)
2. Zig tests load reference data, compare against dmlx output
3. CI runs automatically

### End-to-End Verification

- TinyLlama 1.1B inference output comparison with Python MLX
- Perplexity evaluation (based on mlx-lm evaluate.py)

---

## 5. Phase Delivery Plan (Updated 2026-04-26, based on v0.0.3)

| Phase | Key Deliverables | Status |
|-------|-----------------|--------|
| Phase 0 | Error handling + memory safety + NN layer GPU migration + build fixes | ✅ Complete |
| Phase 1 | Three-tier generation + model registry + Prompt cache + operator fusion | ✅ Complete |
| Phase 2 | Scheduler + PagedAttention + SSE + Continuous Batching (framework layer) | ✅ Complete |
| Phase 3 | Chunked prefill + Prefix caching + Speculative decoding | ✅ Complete |
| Phase 4 | Quantization + QLoRA + MoE routing | ✅ Complete |
| Phase 5 | Multi-model management + Tiered KV Cache + memory limits + Benchmark | ✅ Complete |
| v0.0.x | Current iteration | Continuous Batching server integration, multi-request forward batching | 🚧 In Progress |
| v0.1.0 | Target | Production deployment: complete CLI + docs + CI + performance verification | 🎯 Planned |

---

## 6. Architecture Evolution

### Current (v0.0.3 — Implemented)
```
┌──────────────────────────────────────────────────┐
│  CLI (chat/serve/benchmark/quantize/lora-train)  │
│  HTTP Server (SSE streaming, OpenAI compat)      │
├──────────────────────────────────────────────────┤
│  Scheduler (continuous batching, chunked PF)     │
│  Request Queue → Schedule → Forward → Post       │
├──────────────────────────────────────────────────┤
│  ModelPool (multi-model LRU, memory limits)      │
│  ModelRegistry (9 architectures: LLaMA/DSV4/     │
│       Mistral/Qwen2/Qwen3/Gemma/GLM-4/Phi/      │
│       Phi-3)                                     │
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
│  NN Layer (all via mlx-c op graph)               │
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

## 7. External Project Reference Quick Lookup

| Feature | vLLM | mlx-lm | oMLX | TileKernels | mlx-rs |
|---------|------|--------|------|-------------|--------|
| PagedAttention | ✅ Original | — | ✅ Adopted | — | — |
| Continuous Batching | ✅ Core | ✅ BatchGen | ✅ Adopted | — | — |
| Chunked Prefill | ✅ | — | — | — | — |
| Prefix Caching | ✅ hash-based | ✅ safetensors | ✅ + SSD | — | — |
| Speculative Decoding | ✅ ngram/EAGLE/Medusa | — | — | — | — |
| Guided Decoding | ✅ FSM/xgrammar | — | — | — | — |
| Scheduler 3-phase | ✅ Core | — | FCFS | — | — |
| Model Registry | ✅ | ✅ 50+ | Inherits mlx-lm | — | — |
| 3-tier Generation Architecture | — | ✅ | — | — | — |
| Quantization GPTQ/AWQ | ✅ | ✅ | Inherits | FP8/FP4 | — |
| Operator Fusion | CUDA graph | Automatic | Inherits | Hand-written kernel | — |
| Tiered KV Cache | — | — | ✅ Hot+Cold | — | — |
| Multi-Model Management | — | — | ✅ LRU+TTL | — | — |
| MoE Routing | ✅ EP | — | — | ✅ Full pipeline | — |
| Numerical Verification Framework | — | — | — | ✅ | — |
| Autodiff Docs | — | — | — | — | ✅ |

---

## 8. KV Cache Architecture Overview (Apple Silicon Deep Optimization)

> Based on Apple Silicon UMA characteristics, mlx-lm best practices, DeepSeek V4 technical report,
> TurboQuant paper (arXiv:2504.19874), defining dmlx's final KV Cache architecture.

### 8.1 Final Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLI / HTTP Server                            │
│  --kv-bits 4|8|16    --max-kv-size auto|<int>    --kv-tier ram|ssd │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐   │
│  │   Generic Model Path│    │   DeepSeek V4 Dedicated Path     │   │
│  │   (LLaMA/Mistral/   │    │                                  │   │
│  │    Qwen/Gemma)      │    │  CSA 4x compression (learned     │   │
│  │                     │    │    pooling)                      │   │
│  │  ┌───────────────┐  │    │  HCA 128x compression (dense    │   │
│  │  │ PagedKVCache  │  │    │    attn)                        │   │
│  │  │ + built-in    │  │    │  MLA low-rank projection        │   │
│  │  │   quantization │  │    │    (q/o_lora_rank)             │   │
│  │  │   (kv_bits)   │  │    │  FP8 KV storage + FP4 indexer  │   │
│  │  └───────┬───────┘  │    │  Sliding window 128 tokens      │   │
│  │          │          │    │  Attention Sink                 │   │
│  │  ┌───────▼───────┐  │    └──────────────┬───────────────────┘   │
│  │  │ Block Manager │  │                   │                       │
│  │  │  alloc/free   │  │    ┌──────────────▼───────────────────┐   │
│  │  │  CoW          │  │    │  mHC residual stream (hc_mult=4) │   │
│  │  │  prefix hash  │  │    │  Sinkhorn normalization          │   │
│  │  └───────┬───────┘  │    │    (Birkhoff)                    │   │
│  └──────────┼──────────┘    │  MoE routing (256 experts,       │   │
│             │               │    top-6)                        │   │
│  ┌──────────▼───────────────┴──────────────────────────────────┐   │
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
│  │  sysctl hw.memsize → dynamic calculation                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Apple Silicon UMA (Zero-copy)                   │   │
│  │  mlx_slice_update (lazy eval, MLX internal memory reuse)     │   │
│  │  CPU/GPU shared memory — no PCIe transfer overhead           │   │
│  │  Metal GPU native 4-bit/8-bit compute                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Five Design Principles → Implementation Mapping

| # | Apple Silicon Design Principle | Code Location | Spec Requirement | Spec Task | Status |
|---|------------------------------|---------------|------------------|-----------|--------|
| 1 | **Zero-copy UMA** — `mlx_slice_update` in-place update, no `@constCast` | `kvcache/standard.zig` `sliceUpdateKV` | R2.3 | Task 1.2 | ✅ Complete |
| 2a | **kv_bits quantization** — 4-bit/8-bit KV cache quantization | `kvcache/quantized.zig` | R11.1–R11.4 | Task 5.6, 5.7 | ✅ Complete |
| 2b | **`--kv-bits` CLI parameter** — serve command exposes quantization options | `main.zig` `--kv-bits 4\|8\|16` | R11.4 | Task 13.1 | ✅ Complete |
| 3 | **RAM+SSD two-tier cache** — Hot/Cold tiering, safetensors offload | `kvcache/tiered.zig` | R22.1–R22.4 | Task 11.3, 11.4 | ✅ Complete |
| 4a | **Paged Attention** — block alloc/free/CoW/prefix sharing | `kvcache/paged.zig` | R10.1–R10.5 | Task 5.3–5.5 | ✅ Complete |
| 4b | **Paged + Quantized combined** — block-internal quantized storage | `kvcache/paged.zig` `kv_bits` parameter | R10.6 | Task 15.1 | ✅ Complete |
| 5 | **max_kv_size auto-config** — dynamic calculation based on device RAM | `memory.zig` `autoMaxKvSize` | R24.1–R24.4 | Task 11.6, 11.7 | ✅ Complete |

### 8.3 Identified Gaps and Remediation Plans

#### Gap 1: `--kv-bits` CLI Parameter

**Problem**: `QuantizedKVCache` implements 4-bit/8-bit quantization, but users cannot enable it via CLI or server config.
R11 says "accept kv_bits config parameter" but no corresponding CLI requirement exists.

**Remediation**:
- `serve` command add `--kv-bits 4|8|16` parameter (default 4, Apple Silicon best practice)
- `chat` command add synchronously
- server.zig selects `createQuantized4Bit` / `createQuantized8Bit` / `createStandard` based on `kv_bits` at startup

**Corresponding Spec Change**:
- R12 (SSE Streaming) or new R12.5: serve command SHALL accept `--kv-bits` parameter
- Task 13.1 (final integration) add `--kv-bits` integration

#### Gap 2: Paged + Quantized Combined Strategy

**Problem**: `PagedKVCache` and `QuantizedKVCache` are peer-level `KVCacheStrategy`, cannot be used simultaneously.
Production needs "paged memory management + quantized data compression" combined.

**Remediation** (Two options, Option A recommended):

**Option A: PagedKVCache with built-in quantization**
```zig
pub const PagedKVCache = struct {
    // ... existing fields ...
    kv_bits: u8 = 16,        // New: 4, 8, or 16
    group_size: i32 = 64,    // New

    fn allocPage(self: *PagedKVCache, stream: c.c.mlx_stream) !usize {
        // Decide storage precision based on kv_bits at allocation time
        if (self.kv_bits < 16) {
            // Store (packed, scales, biases) instead of raw array
        }
    }
};
```

**Option B: Composite Strategy Wrapper**
```zig
pub const PagedQuantizedKVCache = struct {
    paged: PagedKVCache,       // Memory management
    kv_bits: u8,               // Quantization precision
    // Implement KVCacheStrategy VTable
    // updateAndFetch: quantize then write to paged block, dequantize on read
};
```

**Corresponding Spec Change**:
- New R10.6: PagedKVCache SHALL support optional kv_bits parameter, quantized storage at block level
- Task 5.3 add sub-task: PagedKVCache constructor accepts kv_bits parameter

### 8.4 Generic Models vs DeepSeek V4: KV Cache Path Bifurcation

The two model categories have completely different KV cache compression strategies and should not be mixed:

```
                    ┌─────────────────┐
                    │  Model Registry │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐       ┌───────────▼───────────┐
    │   Generic Models   │       │   DeepSeek V4          │
    │   (LLaMA/Mistral/  │       │                       │
    │    Qwen/Gemma)     │       │                       │
    └─────────┬─────────┘       └───────────┬───────────┘
              │                             │
    ┌─────────▼─────────┐       ┌───────────▼───────────┐
    │  Post-hoc Quant   │       │  Architecture-native   │
    │                   │       │  Compression            │
    │  PagedKVCache     │       │                       │
    │  + kv_bits=4/8    │       │  CSA 4x + HCA 128x   │
    │  + CoW            │       │  + MLA low-rank proj  │
    │  + prefix sharing │       │  + FP8 dtype cast     │
    │                   │       │  + FP4 indexer        │
    │  Optional:        │       │  + Sliding window 128 │
    │  TurboQuant       │       │                       │
    │  (theoretically   │       │  No additional quant  │
    │   optimal,        │       │  needed — KV cache    │
    │   3.5-bit lossless)│      │  already 2% of V3.2   │
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

**Key Decisions**:
- Generic models default `kv_bits=4` (Apple Silicon best practice, TPS does not drop and may increase on M4 Pro)
- DeepSeek V4 does not use the generic quantization path; its forward path internally does FP8/FP4 dtype casts
- Tiered Cache and Memory Manager are common to both paths
- TurboQuant as an optional advanced quantization for generic models (Phase 4+), not applicable to V4

### 8.5 Apple Silicon Device RAM → KV Cache Recommended Configuration

| Device | RAM | kv_bits=4 max_kv_size | kv_bits=8 max_kv_size | Typical Scenario |
|--------|-----|----------------------|----------------------|------------------|
| M3 MacBook Air | 16 GB | 8,192 – 16,384 | 4,096 – 8,192 | Basic conversation |
| M4 Pro Mac Mini | 48 GB | 24,576 – 49,152 | 12,288 – 24,576 | Code writing |
| M4 Max MacBook Pro | 64 GB | 32,768 – 65,536 | 16,384 – 32,768 | Document analysis |
| M4 Ultra Mac Studio | 128 GB | 65,536 – 131,072 | 32,768 – 65,536 | RAG, ultra-long context |
| M4 Ultra Mac Pro | 192 GB+ | 131,072+ | 65,536+ | Multi-model concurrent |

> Formula: `max_kv_size = (device_ram - model_bytes - 512MB) / (2 × num_kv_heads × head_dim × (kv_bits/8) × num_layers)`
>
> Example with Llama-3-8B (GQA-8, head_dim=128, 32 layers) + kv_bits=4:
> `bytes_per_token = 2 × 8 × 128 × 0.5 × 32 = 32,768 bytes ≈ 32KB/token`
> 64GB device, model ~4GB → available ~59GB → max_kv_size ≈ 59GB / 32KB ≈ 1,900,000 tokens
