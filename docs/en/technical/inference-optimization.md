# Inference Speed Optimization Plan v1.0

> **Goal**: Reduce per-token latency from 82.2ms → 25-30ms (12.2 tok/s → 33-40 tok/s),
> bringing a 100-token request from ~8s down to ~3-4s target on a 48GB Mac.

---

## 1. Current Baseline

**Hardware**: Apple M4 Pro, 48 GB unified memory
**Model**: DeepSeek-V4-Flash-4bit, 33 shards (~141 GB on disk, 4-bit quantized)
**Mode**: SMELT 15% + stream + ExpertCache 4GB + temperature=0
**Commit**: `7e72a7` (2026-05-05)

| Metric | Value |
|--------|-------|
| Prefill (token 1) | 370.5 ms |
| Steady-state ITL (tokens 3-10 avg) | **82.2 ms** |
| Throughput (steady-state) | **12.2 tok/s** |
| Weights in RAM | ~6 GB (SMELT 15%) |
| Expert cache budget | 4 GB (LFU) |
| NVMe bandwidth (Apple) | ~5-7 GB/s |
| UMA bandwidth (M4 Pro) | ~400 GB/s |
| Metal GPU (M4 Pro) | ~15 TFLOPS FP16 |

---

## 2. Hot Path Decomposition (Decode Phase, Per Token)

```
82.2ms total per token (43 layers)
│
├── 0.5ms   Token embedding lookup
├── 0.1ms   Expand to mHC [B,1,4,H]
│
├── 78ms    43 Transformer Layers (1.81ms avg per layer)
│   │
│   ├── 0.15ms  mHC attention pre-processing (Sinkhorn normalization)
│   ├── 0.05ms  RMSNorm (fused fast kernel)
│   ├── 0.30ms  MLA Attention
│   │   ├── 0.05ms Q/KV projection (q_lora_rank 1024)
│   │   ├── 0.05ms RoPE apply (YARN-scaled, partial dim 64/512)
│   │   ├── 0.05ms KV cache update (CSA/HCA compression)
│   │   ├── 0.05ms CSA indexing (LightningIndexer top-512 selection)
│   │   ├── 0.05ms SDPA (scaled dot product attention)
│   │   └── 0.05ms O-lora projection (8 groups)
│   ├── 0.10ms  mHC attention post-processing
│   ├── 0.05ms  mHC FFN pre-processing
│   ├── 0.05ms  RMSNorm
│   ├── **0.90ms  MoE Forward ← PRIMARY BOTTLENECK (50% of layer time)**
│   │   ├── 0.05ms Gate routing (score-based top-6 of 256 experts)
│   │   ├── 0.03ms Shared expert forward
│   │   ├── **0.50ms Expert streaming / cache I/O**
│   │   │   ├── Cache lookup (LFU, ~70% hit rate)
│   │   │   ├── Cache miss → PartialTensorReader disk read (~5-7 GB/s NVMe)
│   │   │   └── Array construction from mmap/expert cache
│   │   ├── 0.20ms SwitchGLU compute (gate_proj + up_proj + SiLU + down_proj)
│   │   ├── 0.02ms Score re-weighting and sum
│   │   └── 0.10ms mHC FFN post-processing
│   └── 0.10ms  hidden.eval() ← forced GPU synchronization
│
├── 0.20ms  HyperHead compress (mHC [4×H] → [H])
├── 0.05ms  Final RMSNorm
├── 0.50ms  LM head [1, 4096] × [4096, 129280] = ~530M FLOPs
└── 0.35ms  Sampling (slice→squeeze→astype f32→sample)
```

### Bottleneck Summary

| Rank | Bottleneck | Location | Est. Cost | Category |
|------|-----------|----------|-----------|----------|
| B1 | MLX per-op dispatch overhead | `DSV4Model.forward()` | ~30ms/tok | Compute |
| B2 | Expert streaming I/O latency | `expert_stream.zig:streamingForward()` | ~21ms/tok | I/O |
| B3 | Per-layer `hidden.eval()` sync | `deepseek_v4.zig:2753` | ~4.3ms/tok | Sync |
| B4 | Sequential compute-I/O pipeline | `layer_prefetcher.zig` | ~15ms/tok | Pipelining |
| B5 | Per-layer ScopedArrayArena churn | `deepseek_v4.zig` layer loop | ~5ms/tok | Memory |
| B6 | LM head matmul (530M FLOPs) | `deepseek_v4.zig:2772` | ~0.5ms/tok | Compute |

---

## 3. Physical Limits

### Memory Bandwidth Limit

M4 Pro has ~400 GB/s unified memory bandwidth (CPU + GPU shared).

Per-token data traffic for decode (single token, batch=1):

| Operation | Data per layer | ×43 layers | Bandwidth time |
|-----------|---------------|------------|----------------|
| Attention weights (MLA Q/KV/O) | ~40 MB | 1.7 GB | 4.3 ms |
| MoE weights (7 experts, 4-bit → dequant) | ~42 MB | 1.8 GB | 4.5 ms |
| mHC + norms + residuals | ~5 MB | 0.2 GB | 0.5 ms |
| LM head | ~530 MB | 0.5 GB | 1.3 ms |
| **Total** | **~87 MB** | **~4.2 GB** | **~10.6 ms** |

**Bandwidth-limited maximum: ~94 tok/s** (10.6ms/token)

### Metal Compute Limit

M4 Pro ~15 TFLOPS FP16:
- MoE per layer: ~700M FLOPs → 43 layers → ~30G FLOPs
- Attention: ~50M FLOPs per layer → ~2G FLOPs
- LM head: ~0.5G FLOPs
- Total: ~32.5G FLOPs per token → 2.2ms compute → **~460 tok/s**

**Conclusion: Entirely bandwidth-limited. Compute is never the bottleneck.**

### SSD I/O (Stream Mode)

With SMELT + stream, expert weights come from NVMe SSD (~5-7 GB/s):
- Per layer: 42 MB expert data × (1 - cache_hit_rate) actual disk reads
- Without cache: 42 MB / 6 GB/s = 7 ms per layer → 301 ms total per token (impossible)
- With 70% cache hit: ~12.6 MB / 6 GB/s = 2.1 ms per layer → 90 ms → matches observation
- With 95% cache hit: ~2.1 MB / 6 GB/s = 0.35 ms per layer → 15 ms

**Key insight: Expert cache hit rate directly determines decode speed.**

---

## 4. Optimization Plan

### Phase 1: Low-Hanging Fruit (Sprint 1, ~1 week)

#### P1.1 — Conditional `eval()` Skip for Decode ⭐

**Code**: `src/models/deepseek_v4.zig:2747-2756`

**Problem**: `hidden.eval()` forces Metal GPU synchronization after every layer. The comment says "This allows MLX to page weights in/out of memory for large models" — which matters during prefill (many tokens → many intermediate tensors ≥ memory pressure) but NOT during decode (1 token → minimal intermediate tensors).

**Fix**: Add `is_decode` parameter; only eval at layer 0 and the last layer during decode.

```zig
// In DSV4Model.forward(), after each layer:
if (!is_decode or i == 0 or i == self.layers.len - 1) {
    try hidden.eval(); // sync only at boundaries during single-token decode
}
```

**Expected**: ITL 82.2ms → 78ms (**-5%**)
**Risk**: Low. Decode has minimal memory pressure. Prefill path unchanged.
**Verification**: Run `make benchmark` before/after. Ensure 7/7 correctness tests pass.

#### P1.2 — Expand Expert Cache Budget ⭐⭐

**Code**: `src/models/expert_cache.zig:14` (`DEFAULT_MAX_BYTES`)

**Problem**: 4 GB cache holds ~682 expert entries → ~16 per layer. Each layer needs 6 (routed) + 1 (shared) = 7 experts. Hit rate ~70% → ~2.1 disk reads per layer at ~7ms each.

**Fix**: Increase cache from 4 GB to 10 GB. On 48 GB Mac: 6 GB SMELT preloads + 6 GB backbone + 10 GB cache + 2 GB KV cache = 24 GB total, well within 48 GB.

```zig
// src/models/expert_cache.zig
pub const DEFAULT_MAX_BYTES: usize = 10 * 1024 * 1024 * 1024; // 10 GB
```

**Alternative**: Make cache size configurable via CLI (`--expert-cache-gb 10`).

**Expected**: 
- Cache entries: 682 → ~1700 (40 per layer)
- Hit rate: 70% → ~95%
- Disk I/O reduction: ~15ms/token
- ITL 78ms → 63ms (**-19%**)

**Risk**: None if total memory stays under 48GB. Add a check in `memory.zig` to warn if expert cache + model > available RAM.

#### P1.3 — Zero-Copy MLX Arrays ⭐

**Code**: `src/io/safetensors_reader.zig`, `src/models/deepseek_v4_loader.zig`, `src/models/expert_stream.zig`

**Problem**: `mlx_array_new_data()` copies data from mmap into MLX-managed memory. For 7 GB backbone weights + per-token expert slices, this adds ~2-3ms cumulatively.

**Fix**: Use `mlx_array_new_data_managed_payload()` with a noop deleter, as documented in [TTFT Optimization](ttft-optimization.md#p2-mmap-zero-copy).

```zig
// Replace:
const arr = c.c.mlx_array_new_data(slice.ptr, shape_i32.ptr, ndim, dtype);

// With:
fn noopDeleter(_: ?*anyopaque) callconv(.c) void {}
const arr = c.c.mlx_array_new_data_managed_payload(
    @constCast(@ptrCast(slice.ptr)),
    shape_i32.ptr, ndim, dtype,
    null, noopDeleter,
);
```

**Locations to patch**: 3 call sites in mmap paths (not pread fallback).

**Expected**: ITL 63ms → 61ms (**-3%**)
**Risk**: Low. MmapPool covers lifetime. If `newBuffer` rejects non-page-aligned pointer, MLX auto-falls back to copy.
**Note**: Most expert tensors (safetensors offsets) are NOT page-aligned, so actual zero-copy hit rate may be low. But the change is trivial and can't hurt.

---

### Phase 2: MLX Compile Fusion (Sprint 2, ~2 weeks)

#### P2.1 — Layer-Level `compile()` Fusion ⭐⭐⭐⭐

**Code**: `src/models/deepseek_v4.zig` — `DSV4TransformerBlock.forward()`

**Problem**: Each layer's forward pass dispatches ~50+ individual MLX operations (matmul, add, multiply, reshape, transpose...), each triggering a separate Metal kernel launch. Kernel launch overhead is ~10-50μs per op → ~1-2.5ms per layer → ~43-108ms total per token. Additionally, MLX lazy graph construction and scheduling adds overhead.

**Fix**: Wrap the layer forward pass in `mlx_compile()` to fuse all operations into a single Metal compute graph. During decode, shapes are static (batch=1, seq=1), making the graph fully compilable.

```zig
// Compile the decode-phase forward once at initialization
const decode_closure = try Closure.init(decodeForward, allocator);
const compiled_decode = try compile.compile(decode_closure, false);

// Per-token: re-execute the compiled graph (single Metal dispatch!)
const logits = try compiled_decode.execute(input_arr, null, caches, start_pos, stream);
```

**Challenges**:

1. **KV cache update path**: `mlx_slice_update()` used for KV cache updates IS compile-compatible (it's a pure function with array inputs).

2. **MoE routing**: The `topk()` and `argpartition()` operations in gate routing are compile-compatible.

3. **Expert streaming**: The `ExpertStreamProvider.streamingForward()` loads tensors from disk — this is NOT compile-compatible. Two solutions:
   - **Option A (recommended)**: Split the layer into two compiled segments: `attention_block` (fully compiled) and `moe_compute` (compiled, fed pre-loaded expert arrays from cache).
   - **Option B**: Pre-load all experts for the token before entering the compiled graph, pass them as inputs.

4. **Dynamic shapes during prefill**: Prefill has variable seq_len → a separate compiled graph or fallback to eager mode.

**Architecture**:

```
Token generation loop
│
├── Expert pre-loading phase (I/O, not compiled)
│   └── For each layer: cache lookup / disk read → expert arrays in memory
│
└── Compiled model execution (single Metal dispatch)
    └── 43 layers + LM head → one fused Metal graph
        ├── Attention (MLA + CSA/HCA + SDPA) — compiled
        ├── MoE gate routing — compiled
        ├── SwitchGLU (using pre-loaded experts) — compiled
        ├── Shared expert — compiled
        └── mHC pre/post — compiled
```

**Implementation Steps**:

1. Create a `decodeForward()` function that takes pre-loaded expert arrays as inputs
2. Compile it once after model loading
3. Per-token: load experts → execute compiled graph → sample

**Expected**: Eliminates ~30ms of Metal kernel launch + graph construction overhead.
ITL 61ms → **31ms** (**-49%**)

**Risk**: Medium. `compile()` may not support all MLX ops used in CSA/HCA attention. Needs op-by-op compatibility testing.

#### P2.2 — Attention Block Compilation (Fallback if full model compile fails)

If full model compilation is infeasible, compile the attention block separately (the most op-heavy section).

**Expected**: ITL 31ms → ~25ms if full compile works; ~55ms if only attention compiled.

---

### Phase 3: I/O Pipeline Optimization (Sprint 3, ~2 weeks)

#### P3.1 — Multi-Layer Expert Prefetching ⭐⭐⭐

**Code**: `src/models/layer_prefetcher.zig`

**Problem**: Current prefetcher loads only layer N+1 while layer N computes. SSD latency (~7ms per cold read) means compute waits for I/O. With 95% cache hit after P1.2, this is less critical, but still matters for the first few tokens of a request.

**Fix**: Replace single-layer prefetch with a **multi-layer ring buffer**:

```
Layer N computing on GPU
│
├── Worker 1: prefetching experts for layer N+1
├── Worker 2: prefetching experts for layer N+2
└── Worker 3: prefetching experts for layer N+3
```

```zig
pub const LayerPipeline = struct {
    workers: [3]std.Thread,
    ring: RingBuffer(PrefetchRequest, 8), // circular buffer, non-blocking
    cache: *ExpertCache,
    reader: *PartialTensorReader,

    pub fn stage(self: *LayerPipeline, layer_idx: usize, expert_ids: []const u32) void {
        // Non-blocking: push request, worker picks it up
        self.ring.push(.{ .layer = layer_idx, .experts = expert_ids });
    }

    pub fn ensureReady(self: *LayerPipeline, layer_idx: usize) void {
        // Block only if this layer isn't ready yet (should be rare with 3-stage pipeline)
        while (!self.ring.isReady(layer_idx)) {
            std.Thread.yield() catch {};
        }
    }
};
```

**Expected**: Hides 80-100% of remaining SSD latency. ITL 31ms → **27ms** (**-13%**)

**Risk**: Medium. Thread coordination needs careful implementation. The `PartialTensorReader` must be thread-safe for concurrent reads from the same safetensors file (pread is thread-safe on POSIX, but fd sharing needs attention).

**Alternative**: If multi-threading is too complex, implement **non-blocking single prefetch**: use `mach_absolute_time()` to measure how long compute takes, and start prefetch for N+1 WITHOUT blocking at the end of layer N. Only block if you reach layer N+1 before prefetch completes.

#### P3.2 — Sequential Expert Layout in Safetensors

**Code**: `src/models/expert_stream.zig:loadExpertSlicesCached()`

**Problem**: `PartialTensorReader` reads gate_proj, up_proj, down_proj as 3 separate file reads for each expert. safetensors layout has all experts fused along axis 0, so reading expert 42's gate_proj reads a single contiguous slice — good. But the 3 projections for expert 42 are separated by other experts' tensors in the file → 3 seeks per expert.

**Fix**: When building the TensorIndex, annotate expert tensor groups so the partial reader can issue a single vectored read (preadv) for all 3 projections of one expert.

**Expected**: Reduces per-expert I/O syscalls from 3 to 1. Modest latency improvement (~1ms/tok).

---

### Phase 4: Compute Optimization (Sprint 4, ~3 weeks)

#### P4.1 — Fused SwitchGLU Metal Kernel ⭐⭐⭐

**Code**: New file `src/models/metal/switchglu_fused.metal`

**Problem**: SwitchGLU forward does 4 matrix multiplies + SiLU activation + element-wise multiply → 6 Metal kernel launches per layer. Each kernel launch means: building MTLComputeCommandEncoder → setting buffers → dispatching → ending encoding.

**Fix**: Write a custom Metal kernel that fuses the entire SwitchGLU into ONE dispatch:

```metal
// switchglu_fused.metal
kernel void switchglu_fused(
    device const float* x,        // [N, D] input
    device const float* gate_w,   // [N_experts, intermediate, D] packed
    device const float* up_w,     // [N_experts, intermediate, D] packed
    device const float* down_w,   // [N_experts, D, intermediate] packed
    device const uint* indices,   // [N, topk] expert routing
    device float* y,              // [N, D] output
    constant uint& N,
    constant uint& D,
    constant uint& intermediate,
    constant uint& topk,
    // ... threadgroup memory for tiled matmul ...
) {
    // Fused: gather experts → gate_proj @ x → SiLU → up_proj @ x → multiply → down_proj
    // All in one kernel, zero intermediate allocations
}
```

**Integration with MLX**: Register as a custom MLX op via `mlx_register_custom_op()` or call directly via `mlx_array_new_data()` wrapping Metal buffer.

**Expected**: Eliminates 5 of 6 kernel launches per layer.
ITL 27ms → **24ms** (**-11%**)

**Risk**: High. Requires Metal Shading Language expertise. Needs numerical verification against Python reference (as ds4 does with test vectors).

**Reference**: DeepSeek's [TileKernels](https://github.com/deepseek-ai/tilekernels) (CUDA) — dmlx already references this for Sinkhorn normalization and fused SwitchGLU with gather_mm.

#### P4.2 — KV Cache Update Optimization

**Code**: `src/kvcache/` and `src/models/deepseek_v4.zig` attention path

**Problem**: KV cache is updated via multiple MLX slice/update calls per attention head. The CSA/HCA compressor adds additional computation.

**Fix**: For decode (single token), KV cache update is just appending one token's K and V vectors. This can be a single `mlx_slice_update()` per layer instead of the multi-step CSA/HCA compression pipeline (which is designed for prefill/pooling). Verify that the existing code already optimizes for this case.

**Expected**: Small improvement (~1ms) if not already optimized.

---

### Phase 5: Speculative Decoding (Already Built — Enable by Default)

#### P5.1 — Enable PLD by Default for Greedy Decoding ⭐⭐⭐

**Code**: `src/generation.zig:401` (`streamGenerateSpeculative`), `src/speculative.zig`

**Status**: PLD (Prompt Lookup Decoding) and EAGLE speculative decoding are already implemented but not enabled by default. PLD searches recent context for matching n-grams and proposes 2-3 draft tokens, verified in one forward pass.

**Fix**: Make PLD the default for greedy decoding (temperature=0). With ~70% acceptance rate and 3-token drafts → effective 2.1x throughput.

**Expected with P1-P4 optimizations**: 24ms ITL × 2.1x = effective **11.4ms/token** → **~88 tok/s**
**Without P1-P4** (current baseline + PLD): 82ms × 2.1x = effective **39ms/token** → **~26 tok/s**

**Risk**: Low. PLD is statistically equivalent to autoregressive sampling. Already verified.

---

## 5. Combined Impact Summary

| Phase | Optimization | Savings | New ITL | New tok/s | Effort |
|-------|-------------|---------|---------|-----------|--------|
| — | **Current Baseline** | — | 82.2ms | 12.2 | — |
| P1.1 | Conditional eval skip | 4ms | 78ms | 12.8 | 1 line |
| P1.2 | Expand cache 4→10 GB | 15ms | 63ms | 15.9 | 1 constant |
| P1.3 | Zero-copy arrays | 2ms | 61ms | 16.4 | 3 call sites |
| P2.1 | MLX layer compile | 30ms | **31ms** | **32.3** | 2 weeks |
| P3.1 | Multi-layer prefetch | 4ms | **27ms** | **37.0** | 2 weeks |
| P4.1 | Fused SwitchGLU kernel | 3ms | **24ms** | **41.7** | 3 weeks |
| P5.1 | PLD speculative (enable) | 2.1× | eff. 11.4ms | **eff. 88** | Already built |

| Scenario | ITL | tok/s | 100-token request | 200-token request |
|----------|-----|-------|-------------------|-------------------|
| Current | 82.2ms | 12.2 | **8.2s** | **16.4s** |
| After Phase 1 | 61ms | 16.4 | 6.1s | 12.2s |
| After Phase 2 | 31ms | 32.3 | **3.1s** ✅ | 6.2s |
| After Phase 3 | 27ms | 37.0 | **2.7s** ✅ | 5.4s |
| After Phase 4 | 24ms | 41.7 | **2.4s** ✅ | 4.8s |
| + PLD (P5) | eff. 11.4ms | eff. 88 | **1.1s** ✅ | **2.3s** ✅ |

**Target achieved at Phase 2: 100-token request in ~3s.**

---

## 6. Implementation Roadmap

```
Week 1-2: Phase 1 (Low risk, immediate gain)
  ├── Day 1: P1.1 conditional eval skip
  │   └── 1-line change + test
  ├── Day 2: P1.2 expand expert cache
  │   └── Constant change + memory check
  └── Day 3-4: P1.3 zero-copy arrays
      └── 3 call site replacements + verify no regression
  Expected: 82ms → 61ms (+35% tok/s)

Week 3-4: Phase 2 (MLX compile — highest impact)
  ├── Day 5-7: Op compatibility audit
  │   └── Test each op in forward() for compile() support
  ├── Day 8-10: Implement decodeForward() wrapper
  │   └── Pre-load experts → pass as inputs → compile
  ├── Day 11-12: Integration testing
  │   └── Correctness: 7/7 prompts + logits diff < 1e-5
  └── Day 13-14: Performance validation
  Expected: 61ms → 31ms (+97% tok/s)

Week 5-6: Phase 3 (I/O pipeline)
  ├── Day 15-18: Multi-layer prefetcher
  │   └── Ring buffer + worker thread pool
  ├── Day 19-20: Thread safety for PartialTensorReader
  └── Day 21: Integration + benchmark
  Expected: 31ms → 27ms (+15% tok/s)

Week 7-9: Phase 4 (Metal kernel — deep optimization)
  ├── Day 22-25: Write fused SwitchGLU Metal kernel
  ├── Day 26-28: Numerical verification vs Python reference
  ├── Day 29-31: MLX custom op integration
  └── Day 32-33: Full pipeline benchmark
  Expected: 27ms → 24ms (+13% tok/s)

Ongoing: Phase 5 (Speculative — already built)
  └── Enable PLD by default for greedy decoding
  Expected: 2.1× effective throughput on top of all optimizations
```

---

## 7. Verification Protocol

### Before/After Benchmarks

```bash
# Baseline
make benchmark

# After each phase, re-run:
make benchmark
MAX_TTFT_MS=300 MAX_ITL_MS=40 make benchmark
```

### Correctness Validation

```bash
# After every change, ensure 7/7 prompts pass
make verify MODEL_PATH=~/models/DeepSeek-V4-Flash-4bit
make e2e MODEL_PATH=~/models/DeepSeek-V4-Flash-4bit
```

### Logit Diff Check

```bash
# Compare logits before/after optimization
./zig-out/bin/dmlx benchmark --model ~/models/DeepSeek-V4-Flash-4bit \
  --dump-logprobs /tmp/before.json --temp 0 -p "The capital of France is"
# ... apply optimization ...
./zig-out/bin/dmlx benchmark --model ~/models/DeepSeek-V4-Flash-4bit \
  --dump-logprobs /tmp/after.json --temp 0 -p "The capital of France is"
# diff /tmp/before.json /tmp/after.json — should be identical for greedy decoding
```

### Memory Safety

```bash
# Check process RSS doesn't exceed 48GB
/usr/bin/time -l ./zig-out/bin/dmlx chat --model ~/models/DeepSeek-V4-Flash-4bit \
  --smelt --smelt-experts 0.15 -p "Hello" 2>&1 | grep "maximum resident"
```

---

## 8. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| `compile()` doesn't support CSA/HCA ops | Medium | High | Fall back to attention-only compile (P2.2) |
| Multi-threaded PartialTensorReader race | Low | High | Use per-thread fd handles; pread is thread-safe on POSIX |
| 10GB cache OOM on 48GB | Low | Medium | Add memory budget enforcement in `memory.zig` |
| Fused SwitchGLU numerical mismatch | Medium | Medium | Verify against Python reference; ds4 test vector approach |
| PLD produces worse quality | Low | Low | PLD is statistically equivalent; only for greedy mode |

---

## 9. Key References

| Project | What to Reference |
|---------|-------------------|
| [ds4](https://github.com/antirez/ds4) | Custom Metal graph executor, asymmetric quantization, fused kernels |
| [oMLX](https://github.com/jundot/omlx) | Tiered KV cache, multi-model management, continuous batching patterns |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | `mx.compile()` usage, `stream_generate` architecture |
| [TileKernels](https://github.com/deepseek-ai/tilekernels) | Fused SwitchGLU + Sinkhorn Metal kernels (CUDA → Metal adaptation) |
| [vLLM](https://github.com/vllm-project/vllm) | PagedAttention, scheduler 3-phase loop, chunked prefill |

---

## Appendix A: Source Files Modified

| File | Phase | Change Summary |
|------|-------|----------------|
| `src/models/deepseek_v4.zig` | P1.1, P2.1 | Conditional eval skip + compile wrapper |
| `src/models/expert_cache.zig` | P1.2 | Default cache size constant |
| `src/models/expert_stream.zig` | P1.3, P3.1 | Zero-copy array + pipeline prefetch |
| `src/io/safetensors_reader.zig` | P1.3 | `loadTensor` zero-copy path |
| `src/models/deepseek_v4_loader.zig` | P1.3 | `loadWeightsSelective` zero-copy path |
| `src/models/layer_prefetcher.zig` | P3.1 | Multi-layer ring buffer pipeline |
| `src/models/metal/switchglu_fused.metal` | P4.1 | New: fused SwitchGLU kernel |
| `src/generation.zig` | P5.1 | Enable PLD by default for greedy |
| `src/memory.zig` | P1.2 | Expert cache memory budget check |
| `src/main.zig` | P1.2 | `--expert-cache-gb` CLI flag |

---

## Appendix B: Memory Budget on 48GB Mac

```
Total unified memory:                    48.0 GB
├── macOS + other apps:                  ~8.0 GB (reserve)
├── Available for dmlx:                  ~40.0 GB
│
├── Backbone weights (mmap):             ~6.0 GB  (attention + norms + shared + routing)
├── SMELT preloads (15% experts):        ~6.0 GB  (38 experts × ~160MB each)
├── Expert cache (P1.2: 10GB):          ~10.0 GB  (LFU, near-100% hit rate)
├── KV cache (paged, 32K context):       ~1.5 GB  (CSA/HCA compressed)
├── MLX runtime + Metal:                 ~2.0 GB  (buffers, command queues)
│
└── Remaining headroom:                  ~14.5 GB  (safe for spikes)
```
