# dmlx Optimization Roadmap

> **Date**: 2026-05-16
> **Baseline**: commit f970d9f (SMELT 10% + 6GB cache + warmup)
> **Hardware**: Apple M4 Pro, 48GB unified memory

---

## Current Performance (2026-05-16)

| Metric | Value | Notes |
|--------|-------|-------|
| **Server-side tok/s (warm)** | **17-24** | After cache warmup |
| **Server-side tok/s (cold)** | 8.6 | First request |
| **Steady-state ITL** | 59.2ms | Token 8+ average |
| **Prefill** | 32.4ms | Single token |
| **Cache hit rate** | 23.7% | 6GB cache |
| **HTTP end-to-end (cold)** | 145s | macOS memory pressure |
| **HTTP end-to-end (warm)** | ~32s | Converges by request #3 |
| **100-token server-side** | 4.2s | 24 tok/s |
| **RSS** | 3.4GB | Very efficient |
| **Startup** | 104s | Including warmup |
| **Correctness** | 7/7 | All prompts pass |

---

## Performance Evolution

| Commit | Date | Config | tok/s | ITL | Key Change |
|--------|------|--------|-------|-----|------------|
| dff154d | 05-05 | 4GB cache | 10.5 | 95ms | Initial baseline |
| 538f930 | 05-09 | 4GB cache | 14.5 | 69ms | Tuning |
| 6d339e0 | 05-14 | 10GB cache | 10.7 | 94ms | Cache too large (regression) |
| **f970d9f** | **05-16** | **6GB cache + warmup** | **16.9-24** | **59ms** | **Optimal cache size** |

---

## Verified Findings

### What Works

| Optimization | Impact | Status |
|-------------|--------|--------|
| Expert cache 6GB (not 4, not 10) | +58% tok/s vs 10GB | ✅ Deployed |
| Cache warmup before accept | -85% first-request misses | ✅ Deployed |
| mmap for expert loading | 2x tok/s vs pread | ✅ Kept |
| ReleaseFast build | ~2x vs Debug | ✅ Always |

### What Doesn't Work (Do Not Retry)

| Approach | Why It Failed |
|----------|---------------|
| Expert cache 10GB+ | Squeezes backbone pages → page thrashing |
| SMELT 15% (more preload) | Extra 3GB preload squeezes OS page cache |
| pread replaces mmap | -44% tok/s (loses OS readahead) |
| OS threads for HTTP | 0% improvement (not a fiber scheduling issue) |
| posix write bypass | 0% improvement (write is not the bottleneck) |
| P1.1 eval skip (stream mode) | -40% ITL (stream needs eval for page-in) |
| Zero-copy mmap arrays | <7% gain, not page-aligned |
| PLD speculative decoding | -15% tok/s (low n-gram match rate for this model) |
| LRU cache eviction | -36% tok/s (cache thrashing: 258 new entries/token evict all) |

### Hardware Limits (Cannot Fix in Code)

| Issue | Cause | Mitigation |
|-------|-------|------------|
| 38s HTTP cold start | 141GB model on 48GB RAM → page thrashing | Warmup + smaller cache |
| 145s first HTTP request | Backbone page-in after warmup fills cache | Accept trade-off |
| Cache hit rate 24% | 256 experts × 43 layers too large for 6GB | Architectural (see below) |

---

## Next Optimization Priorities

### Priority 1: MLX Compile Fusion ⭐⭐⭐⭐ (Expected: ITL 59ms → 30ms)

**Current bottleneck**: 43 layers × ~50 MLX ops per layer = ~2150 individual Metal
kernel dispatches per token. Each dispatch has ~10-50μs overhead → 20-100ms total.

**Solution**: `mlx_compile()` fuses the entire decode forward pass into a single
Metal compute graph. Shapes are static during decode (batch=1, seq=1).

**Steps**:
1. Spike test: compile a 2-layer simplified model to verify op compatibility
2. If CSA/HCA ops are compatible → full model compile
3. If not → attention-only compile (fallback, ~6ms savings)

**Risk**: Medium. Expert dynamic binding and CSA/HCA ops may not be compile-compatible.

**Effort**: 2-4 weeks.

### Priority 2: ~~Cache Strategy Optimization~~ ❌ Verified Ineffective

**Tested**: LRU eviction (replace LFU).

**Result**: -36% tok/s. LRU causes cache thrashing in MoE — each token routes to
6 experts × 43 layers = 258 new cache entries, which evict all previous entries.
Next token routes to different experts → 100% miss. LFU is better because it
preserves high-frequency experts that are shared across multiple tokens.

**Conclusion**: LFU is already the optimal eviction policy for MoE routing patterns.
Further cache strategy improvements would require fundamentally different approaches
(e.g., layer-partitioned budgets or routing-prediction-based prefetch), which have
diminishing returns given the 6GB budget constraint.

### Priority 3: ~~PLD Speculative Decoding~~ ❌ Verified Ineffective

**Tested**: `--speculative-ngram 3` (n-gram draft proposal).

**Result**: -15% tok/s (warm). The model's output rarely repeats prompt content,
so n-gram match rate is very low. Most draft tokens are rejected, wasting the
verification forward pass.

**Conclusion**: PLD is not suitable for this model/task combination. Would need
a trained draft model (EAGLE) for effective speculative decoding.

### Priority 4: Reduce HTTP Cold Start ⭐⭐ (Expected: 145s → 60-80s)

**Options**:
1. **Smarter warmup**: Instead of 5 random prompts, use prompts that maximize
   expert coverage across layers (analyze routing statistics).
2. **Backbone pinning**: Use `mlock()` on backbone weight pages to prevent OS eviction.
   Risk: may trigger OOM if total locked memory exceeds available.
3. **Lazy backbone loading**: Load backbone weights on-demand during first request
   instead of at startup (shifts latency but doesn't eliminate it).

**Effort**: 1 week.

---

## Target Milestones

| Milestone | Target | Timeline | Key Metric |
|-----------|--------|----------|------------|
| ~~M1: PLD enabled~~ | ~~50+ effective tok/s~~ | ❌ | N-gram match too low |
| ~~M2: Cache strategy~~ | ~~50%+ hit rate~~ | ❌ | LRU causes thrashing |
| M3: Compile fusion | 30ms ITL, 33 tok/s | +4 weeks | Server tok/s |
| M4: 100-token < 3s | 3s server-side | M3 achieved | E2E latency |

**Remaining viable optimization**: MLX compile fusion is the only path to
significantly improve server-side tok/s beyond the current 17-24 tok/s.
All other low-hanging fruit has been exhausted and verified.

---

## Memory Budget (48GB Mac)

```
Current allocation (f970d9f):
├── macOS + apps:           ~8 GB
├── Backbone (mmap):        ~6 GB (page-cached by OS)
├── SMELT preloads (10%):   ~6 GB (25 experts × 43 layers)
├── Expert cache (LFU):     ~6 GB (configured)
├── KV cache:               ~0.5 GB
├── MLX runtime + Metal:    ~2 GB
├── OS page cache headroom: ~19.5 GB ← critical for mmap performance
└── Total:                  ~48 GB

Key insight: OS page cache headroom directly determines mmap performance.
Every GB added to expert cache is 1 GB less for OS page cache.
6GB cache is the sweet spot where both expert cache AND backbone
pages can coexist in physical memory.
```

---

## References

- `docs/en/technical/inference-optimization.md` — full optimization plan (Phases 1-5)
- `docs/analysis/inference-optimization-review.md` — critical review with corrections
- `docs/analysis/socket-write-latency.md` — HTTP latency root cause
- `docs/analysis/pread-expert-loading.md` — mmap vs pread experiments
- `docs/analysis/cache-benchmark-results.md` — 4GB vs 8GB cache data
