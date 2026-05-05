# Chapter 10: Technical Debt Assessment and Remediation Recommendations

## 10.1 Debt Heat Map

```
High Impact ↑
  P0 │  [Prompt Cache type vulnerability]
     │  [NN layer GPU-ization (remaining 34)]
  P1 │  [dataSliceMut safety risk]
     │  [Sampling sort performance]
     │  [AdamW temporary object storm]
     │  [Batched Forward not integrated]
  P2 │  [EagerContext stream leak]
     │  [allocator parameter misleading]
     │  [ops.zig API redundancy]
     └─────────────────────────────→ High Frequency
```

## 10.2 Fix Priority

### Immediate (1-2 weeks, high ROI)

| Priority | Issue | Effort | Impact |
|--------|------|--------|------|
| 1 | `prompt_cache.zig` type safety vulnerability | 1-2 days | **Production crash** (always triggered under default config) |
| 2 | `sampling.zig` `insertion` → `pdq`/`mlx_topk` | half day | Significantly reduces 128K vocab latency |
| 3 | `nn.zig` BatchNorm `var_buf` uninitialized | 1 hour | Numerical errors (already flagged by audit) |

### Short-term (1-2 months)

| Priority | Issue | Effort | Benefit |
|--------|------|--------|------|
| 4 | `nn.zig` GPU migration | 2-3 weeks | Eliminates largest technical debt, enables GPU acceleration |
| 5 | `AdamW.step` fusion optimization | 3-5 days | Training speed improvement 5-10x |
| 6 | Add `deinit` to `EagerContext` | half day | Fixes stream leak |
| 7 | `batch_builder` integration | 1-2 weeks | Unlocks continuous batching throughput potential |

### Medium-term (3-6 months)

| Priority | Issue | Effort |
|--------|------|--------|
| 8 | Error type subdivision | 2-3 days |
| 9 | Test coverage completion (nn_tests, grad_tests, golden) | 1-2 weeks |
| 10 | API cleanup (remove redundancy and unused parameters) | 3-5 days |

## 10.3 Specific Fix Plans

### Prompt Cache Type Safety (P0)

**Plan A (Recommended)**: Extend VTable
```zig
pub const VTable = struct {
    // ... existing methods ...
    saveState: ?*const fn (ctx: *anyopaque, path: []const u8) anyerror!void,
    loadState: ?*const fn (ctx: *anyopaque, path: []const u8) anyerror!void,
};
```

**Plan B (Quick fix)**: Add runtime assertion
```zig
const std_cache: *StandardKVCache = @ptrCast(@alignCast(cache.ptr));
std.debug.assert(cache.vtable == &StandardKVCache.vtable);  // safety check
```

### Sampling Sort Optimization (P1)

**Plan A (Recommended)**: GPU top-k
```zig
// Replace insertion sort
const topk_result = try ops.topk(ctx, logits, @intCast(top_k));
```

**Plan B (CPU Optimization)**: `std.sort.pdq`
```zig
std.sort.pdq(ScoredToken, scored[0..vocab_size], {}, scoredGreater);
```

### NN Layer GPU-ization (P0→P1)

Priority ordering:
1. `Linear.init`: switch weight initialization to `mlx_random_normal` (currently uses CPU random)
2. `Embedding.forward`: switch to `mlx_take` (currently uses CPU lookup)
3. `BatchNorm.forward`: switch to `fast.layerNorm` + mean/variance ops
4. `MultiHeadAttention`: switch to `fast.scaledDotProductAttention`
5. `LSTM`/`GRU`/`RNN`: switch to mlx-c op chains (matmul + sigmoid + tanh)

## 10.4 Architecture Recommendations

1. **KV Cache VTable Extension**: add `saveState`/`loadState`/`clone`, eliminate `prompt_cache.zig`'s type-unsafe assumptions

2. **NN Layer Unified Abstraction**:
   - All NN layers implemented via mlx-c op chains
   - `dataSliceMut` marked as `deprecated`, used only for testing/debugging
   - New `nn/gpu/` submodule for GPU-ized implementations

3. **Stream Lifecycle Unification**:
   - `EagerContext` supports `deinit` to release stream
   - Or switch to global default stream reference (recommended)

4. **Sampling Backend Switching**:
   - vocab > 32K: default `mlx_topk` GPU
   - vocab <= 32K: fallback to CPU `pdq`

5. **Build-time Feature Flags**:
   - `-Dmetal` / `-Daccelerate` / `-Ddistributed`

## 10.5 Comparison with Production Roadmap

`production-roadmap.md` claims all Phase 0–7 + Task 13–34 are completed.

| Dimension | Claim | Actual | Deviation |
|------|------|------|------|
| Feature completion | All completed | Basically accurate | Speculative decoding, guided decoding, MoE, tiered cache all implemented |
| Quality completion | All completed | **Partial overclaim** | NN GPU-ization, @constCast removal not fully completed |
| Test completion | 350 tests passing | Test count accurate | Structural gaps: NN layers, Autograd, real weights |

**Recommendation**: update roadmap to mark `nn.zig` GPU-ization migration from "completed" to "partially completed (activations done, nn layers pending cleanup)".
