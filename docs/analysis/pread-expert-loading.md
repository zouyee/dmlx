# Expert Loading: mmap → pread Migration

> **Date**: 2026-05-16
> **Status**: Investigated — partial improvement, architectural limit reached
> **Goal**: Eliminate HTTP accept() latency caused by mmap VM pressure

---

## Problem

When the server is generating tokens, expert weight loading via mmap triggers
massive page faults (141GB model, 33 shards). The macOS kernel's VM subsystem
becomes saturated, blocking unrelated syscalls — including `accept()` on the
listening socket.

**Evidence** (from `socket-write-latency.md`):
- Server-side generation: 4.2s for 30 tokens
- HTTP end-to-end: 175s for the same request
- Gap: **~170s** of kernel-level accept() blocking
- `sample` tool confirms: main thread 100% in mmap page fault I/O

## Root Cause (Revised)

Initial hypothesis: mmap page faults block `accept()` syscall.

**Actual root cause after investigation**: The 38-40s HTTP overhead is NOT caused
by a single mechanism. It's a combination of:

1. **Zig IO fiber scheduling starvation** (~partially confirmed)
   - Engine loop runs on main thread doing blocking I/O (mmap or pread)
   - Zig `io.async` fibers cannot be scheduled while main thread is blocked
   - Affects: request read, response write, waitForToken polling

2. **macOS memory pressure / page cache thrashing** (~confirmed)
   - 141GB model on 48GB Mac → OS constantly evicts pages
   - After warmup loads expert weights, backbone weights get evicted
   - First real request must re-page-in backbone from SSD
   - This is an OS-level issue, not solvable in application code

3. **waitForToken polling overhead** (~minor)
   - Uses nanosleep(100μs) polling loop
   - Not the primary cause (only adds ~1ms per token)

## Experiments Conducted

### Experiment 1: pread replaces mmap (eliminate VM pressure)

```
ExpertStreamProvider: skip MmapPool, use FdPool + pread fallback
```

| Metric | mmap | pread | Change |
|--------|------|-------|--------|
| HTTP Cold | 175s | 56s | **-68%** |
| HTTP Warm | 57s | 43s | **-25%** |
| Server tok/s (warm) | 8.7 | 4.9 | **-44%** |

**Conclusion**: pread reduces HTTP latency but halves server throughput.
mmap's OS readahead is critical for tok/s performance.

### Experiment 2: pread + coalesced reads + F_RDAHEAD

```
Merge consecutive expert IDs into single pread calls.
Enable F_RDAHEAD on file descriptors.
```

| Metric | pread v1 | pread v2 | Change |
|--------|----------|----------|--------|
| HTTP Cold | 56s | 56s | — |
| Server tok/s (cold) | 3.1 | 3.7 | +19% |
| Prefill | 373ms | 317ms | +15% |

**Conclusion**: Coalescing helps cold start slightly, negligible for warm.

### Experiment 3: pread + warmup (pre-populate cache before accept)

```
Run 5 dummy prompts before starting accept loop.
```

| Metric | pread no warmup | pread + warmup | Change |
|--------|-----------------|----------------|--------|
| HTTP Cold | 56s | **39.8s** | **-29%** |
| HTTP Warm | 43s | **37.9s** | **-12%** |
| Server tok/s | 4.9 | **5.5** | **+12%** |
| Startup time | 39s | 81s | +108% |

**Conclusion**: Warmup significantly helps by pre-populating expert cache.
Trade-off: longer startup time.

### Experiment 4: OS threads replace io.async fibers

```
Accept loop + HTTP handlers on dedicated OS threads.
Blocking libc read/write, no Zig IO dependency.
```

| Metric | io.async + pread | OS thread + pread | Change |
|--------|------------------|-------------------|--------|
| HTTP Cold | 39.8s | 39.8s | **0%** |
| Server tok/s | 5.5 | 5.5 | **0%** |

**Conclusion**: Fiber scheduling is NOT the bottleneck. The 38s delay persists
regardless of threading model. It's caused by macOS memory management.

### Experiment 5: posix write (bypass Zig IO for response)

```
writeJsonResponse: blocking mode + direct libc write + TCP_NODELAY
```

| Metric | Zig IO write | posix write | Change |
|--------|-------------|-------------|--------|
| HTTP total | 39.8s | 39.8s | **0%** |

**Conclusion**: Response write is not the bottleneck.

## Final Diagnosis

The ~38s HTTP overhead is caused by **macOS unified memory management** when
running a 141GB model on 48GB physical RAM:

1. Warmup loads expert weights → fills 10GB cache
2. OS evicts backbone weight pages to make room
3. First real request triggers backbone page-in from SSD (~5-7 GB/s)
4. 6GB backbone / 6 GB/s ≈ 1s for sequential read, but random access pattern
   and OS scheduling overhead inflate this to ~38s

This is a **hardware limitation** — not solvable by code changes alone.

## Recommended Configuration

Based on all experiments, the optimal configuration for 4-bit model on 48GB Mac:

```bash
dmlx serve \
  --model ~/models/DeepSeek-V4-Flash-4bit \
  --smelt --smelt-strategy stream --smelt-experts 0.1 \
  --smelt-cache 10240 \
  --temperature 0
```

With pread + warmup + 10GB cache:
- **Server-side**: 5.2-5.5 tok/s (warm)
- **HTTP end-to-end**: ~38-40s (first request), improving with subsequent requests
- **Startup**: ~80s (including warmup)
- **Correctness**: 7/7 pass

## What Would Actually Help

| Solution | Expected Impact | Feasibility |
|----------|----------------|-------------|
| 64GB+ Mac | Eliminates page thrashing entirely | Hardware upgrade |
| Smaller model (e.g., 7B) | No memory pressure | Different model |
| MLX compile fusion | Reduce forward pass time → less I/O blocking | 2-4 weeks dev |
| Preload mode (if memory allows) | 100% cache hit, no SSD I/O | Needs ~20GB free |
| Reduce backbone size | Less page-in needed | Model architecture change |

## References

- `docs/analysis/socket-write-latency.md` — original investigation
- `c678821` — backbone loading uses pread in stream mode
- `87943c0` (reverted) — OS thread HTTP handling attempt
- `6d339e0` — revert of OS thread changes

