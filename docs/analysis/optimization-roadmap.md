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

### Priority 2: Cache Strategy Optimization ⭐⭐⭐ (Expected: hit rate 24% → 50%+)

**Current issue**: LFU eviction is suboptimal for MoE routing patterns.
Different tokens route to different experts, making frequency counts unreliable.

**Options**:
1. **Layer-partitioned LRU**: Reserve cache budget per layer (6GB / 43 = 140MB/layer).
   Each layer keeps its most recently used experts. Prevents cross-layer eviction.
2. **Predictive prefetch**: After gate routing computes expert indices for layer N,
   prefetch layer N+1's likely experts (based on historical routing correlation).
3. **Adaptive cache sizing**: Monitor per-layer hit rates, allocate more budget to
   layers with higher miss rates.

**Effort**: 1-2 weeks.

### Priority 3: PLD Speculative Decoding ⭐⭐⭐ (Expected: 1.5-2x effective throughput)

**Status**: Already implemented (`src/speculative.zig`), just needs enabling.

PLD (Prompt Lookup Decoding) proposes 2-3 draft tokens from recent context,
verified in a single forward pass. With ~70% acceptance rate → 1.5-2.1x effective
throughput multiplier on top of all other optimizations.

**Expected result**: 24 tok/s × 1.7 = **~41 tok/s effective** (warm cache).

**Effort**: 1-2 days (already built, just enable for greedy decoding).

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
| M1: Compile fusion | 30ms ITL, 33 tok/s | +4 weeks | Server tok/s |
| M2: PLD enabled | 50+ effective tok/s | +1 day | Effective throughput |
| M3: Cache strategy | 50%+ hit rate | +2 weeks | Cache hit rate |
| M4: 100-token < 3s | 3s server-side | M1 + M2 | E2E latency |

**Ultimate target** (from inference-optimization.md):
- ITL: 24ms → 41.7 tok/s
- With PLD: effective 11.4ms → 88 tok/s
- 100-token request: < 3s server-side ✅ (already 4.2s, close)

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
