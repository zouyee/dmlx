# dmlx Optimization Roadmap

> **Date**: 2026-05-16
> **Baseline**: commit f970d9f (SMELT 10% + 6GB cache + warmup)
> **Hardware**: Apple M4 Pro, 48GB unified memory

---

## Current Performance (2026-05-16)

| Metric | Value | Notes |
|--------|-------|-------|
| **Server-side tok/s (warm)** | **14-15** | SMELT 20% + 6GB cache |
| **Server-side tok/s (cold)** | 5.5 | First request |
| **Client HTTP latency (warm)** | **27-30s** | macOS memory pressure |
| **Client HTTP latency (cold)** | 75s | First request |
| **Steady-state ITL** | ~67ms | Token 7+ |
| **Prefill** | 32.4ms | Single token |
| **Startup time** | 46s | SMELT 20% (faster than 10%) |
| **RSS** | ~3.4GB | Process memory (excl. mmap) |
| **Correctness** | 7/7 | All prompts pass |

### Optimal Configuration

```bash
dmlx serve --model ~/models/DeepSeek-V4-Flash-4bit \
  --smelt --smelt-strategy stream --smelt-experts 0.2 \
  --smelt-cache 6144 --temperature 0
```

---

## Performance Evolution

| Commit | Date | Config | Server tok/s | Client Latency | Key Change |
|--------|------|--------|-------------|----------------|------------|
| dff154d | 05-05 | SMELT 10%, 4GB cache | 10.5 | — | Initial baseline |
| 538f930 | 05-09 | SMELT 10%, 4GB cache | 14.5 | — | Tuning |
| 6d339e0 | 05-14 | SMELT 10%, 10GB cache | 10.7 | 38-40s | Cache too large |
| f970d9f | 05-16 | SMELT 10%, 6GB cache | 12-13 | 31-34s | Optimal cache size |
| **latest** | **05-16** | **SMELT 20%, 6GB cache** | **14-15** | **27-30s** | **Optimal SMELT ratio** |

---

## Verified Findings

### What Works

| Optimization | Impact | Status |
|-------------|--------|--------|
| **SMELT 20%** (not 10%, not 30%) | +15% tok/s, -12% client latency vs 10% | ✅ Optimal |
| Expert cache 6GB (not 4, not 10) | +58% tok/s vs 10GB | ✅ Deployed |
| Cache warmup before accept | -85% first-request misses | ✅ Deployed |
| mmap for expert loading | 2x tok/s vs pread | ✅ Kept |
| ReleaseFast build | ~2x vs Debug | ✅ Always |

### What Doesn't Work (Do Not Retry)

| Approach | Why It Failed |
|----------|---------------|
| Expert cache 10GB+ | Squeezes backbone pages → page thrashing |
| SMELT 15% (more preload) | Extra 3GB preload squeezes OS page cache |
| SMELT 30%+ | Initial improvement then degradation — memory pressure builds |
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

| Milestone | Target | Status | Result |
|-----------|--------|--------|--------|
| ~~M1: PLD enabled~~ | ~~50+ effective tok/s~~ | ❌ Done | N-gram match too low (-15%) |
| ~~M2: Cache strategy~~ | ~~50%+ hit rate~~ | ❌ Done | LRU causes thrashing (-36%) |
| ~~M3: Compile fusion~~ | ~~30ms ITL~~ | ❌ Done | Only +6% in stream mode (I/O bound) |
| **M4: SMELT tuning** | **Best client latency** | ✅ Done | **SMELT 20% = 28s client, 15 tok/s** |

**All viable optimizations have been exhausted for 48GB Mac + stream mode.**

Current performance ceiling:
- Server: 14-15 tok/s (warm)
- Client: 27-30s per request (macOS memory pressure, hardware limit)
- 100-token server-side: ~7s

Further improvement requires:
1. Hardware upgrade (64GB+ Mac) — eliminates page thrashing entirely
2. Smaller/different model — reduces memory footprint
3. Apple Silicon with more memory bandwidth — faster page-in

---

## Memory Budget (48GB Mac)

```
Current allocation (SMELT 20% + 6GB cache):
├── macOS + apps:           ~8 GB
├── Backbone (mmap):        ~6 GB (page-cached by OS)
├── SMELT preloads (20%):   ~8 GB (51 experts × 43 layers)
├── Expert cache (LFU):     ~6 GB (configured)
├── KV cache:               ~0.5 GB
├── MLX runtime + Metal:    ~2 GB
├── OS page cache headroom: ~17.5 GB ← critical for mmap performance
└── Total:                  ~48 GB

SMELT ratio tuning results:
├── 10%: 6GB preload, 22GB headroom → 12-13 tok/s, 31s client
├── 20%: 8GB preload, 20GB headroom → 14-15 tok/s, 28s client ← OPTIMAL
├── 30%: 10GB preload, 18GB headroom → 13-14 tok/s, 33s client (degraded)
└── Conclusion: 20% is the sweet spot where preload benefit > headroom cost
```

---

## References

- `docs/en/technical/inference-optimization.md` — full optimization plan (Phases 1-5)
- `docs/analysis/inference-optimization-review.md` — critical review with corrections
- `docs/analysis/socket-write-latency.md` — HTTP latency root cause
- `docs/analysis/pread-expert-loading.md` — mmap vs pread experiments
- `docs/analysis/cache-benchmark-results.md` — 4GB vs 8GB cache data
