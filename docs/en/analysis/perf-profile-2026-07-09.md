# Native Engine Performance Profile — 2026-07-09

## Test Conditions

- Commit: `ac6e291` (feat/ds4-ize-stage1)
- Hardware: M4 Pro 48GB
- Model: DeepSeek-V4-Flash-4bit (MXFP4 experts, affine attention)
- SMELT: N=51, warm state (routing stats loaded)
- Measurement: `NATIVE_TIME_LAYERS=1`, stderr→file (block-buffered, minimal overhead)
- Metric: wall-clock per layer during 2nd decode token (pos=5)

## Results

### Per-layer breakdown (decode token, pos=5, warm SMELT N=51)

| Layer | Total (ms) |
|-------|-----------|
| L0-L2 (hash routing, 256 experts preloaded) | 217-240ms |
| L3-L42 (score routing, top-6 from SMELT pool) | 40-150ms |
| **Average L3-L42** | **~60ms** |

### Component breakdown (from [MOE-IO] / [MOE-TIME] logs)

| Component | Per-layer avg | 43 layers | % of total |
|-----------|--------------|-----------|-----------|
| Expert I/O (pread from SMELT pool) | ~5ms | 215ms | 7% |
| MoE GPU (gate+up+swiglu+down, 6 experts) | ~8ms | 344ms | 12% |
| **Attention + mhc + routing + shared expert** | **~52ms** | **2240ms** | **77%** |
| Compressor/Indexer (DSA, CPU) | included above | | ~4% (est) |

### Total decode budget

- 43 layers × 67ms/layer = **2890ms/token**
- Prefill: 4 tokens × ~3.5s = ~14s (dominated by cold expert loading)
- End-to-end: 5 tokens in ~10s = **0.5 tok/s**

## Key Finding: Attention Path is the Bottleneck (77%)

The previous optimization doc (native-engine-4toks-plan.md) stated "MoE GPU 占 90%".
That was measured at commit `e01aed5` BEFORE the v2 kernel optimization.

**Current state**: After kernel v2 (x_shared + ROWS_PER_TG=8), MoE GPU is only 12%.
The bottleneck shifted to the **attention + mhc hyperconnection** path.

## Attention Path Architecture

Per layer, the attention path consists of:

### merged_cb (1 GPU sync):
1. `mhc_pre_split_weighted_sum_norm` — collapse 4 HC streams → normed input (1 encoder)
2. `wq_a` matmul — [1024, 4096] affine 4-bit dequant matvec (bf16→bf16)
3. `q_norm` — RMSNorm [1024]
4. `wq_b` matmul — [32768, 1024] affine 4-bit dequant matvec (bf16→bf16)
5. `per-head norm` — RMSNorm per head
6. `RoPE` — rotary position embedding (partial, 64 dims)
7. `wkv` matmul — [512, 4096] affine 4-bit dequant matvec (bf16→bf16)
8. `kv_norm` — RMSNorm [512]
9. `kv RoPE` — rotary on KV
10. `KV cache blit` — write new KV entry
11. `SDPA` — scaled dot-product attention (per-head, 32 threads)
12. `wo_a` — 8-group matmul (Q8_0 or dense f32)
13. `wo_b` — [4096, 8192] affine dequant matvec (bf16→f32)

**13 encoders in ONE command buffer.**

### cb2cmd2 (1 GPU sync):
1. `f32→bf16` residual conversion
2. `mhc_post_bfloat` — attention residual update
3. `bf16→f32` writeback
4. `mhc_pre_bfloat` — FFN HC split
5. `ffn_RMSNorm`
6. `routing_gate matmul` — [256, 4096] f32 (SIMD parallel)
7. `moe_route_gpu` — sqrtsoftplus + penalty + bitonic top-6

**7 encoders in ONE command buffer.**

### Summary: 20 GPU encoders per layer in 2 CBs before MoE even starts.

## Attention SDPA Kernel: NOT FlashMLA

Current: `mla_sdpa_decode_bfloat` — simple per-head kernel, 32 threads/head,
serial KV iteration. No tiling, no online softmax, no flash attention.

For short sequences (pos<100), SDPA is fast (<1ms). For long sequences it will dominate.
At pos=5 in our test, SDPA is negligible.

## Actual Bottleneck: Weight Matmuls

The dominant operations in the 52ms attention path are the **dequantization matmuls**:
- `wq_b` [32768, 1024]: 32K output × 1K input → largest single matmul per layer
- `wo_b` [4096, 8192]: 4K output × 8K input → second largest
- `wq_a` [1024, 4096]: moderate
- `wkv` [512, 4096]: moderate
- `wo_a` 8× [1024, 4096]: moderate (parallelized via 8 groups)

These use `dequant_matvec_affine_bf16in_bf16out` kernel — same dispatch pattern
as MoE v2 (no x_shared for attention weights, since weights are fixed per layer
and already in persistent GPU buffers via AttnBufCache).

## Optimization Directions (Priority Order)

### 1. ~~Attention kernel optimization~~ (RULED OUT)

The `dequant_matvec_affine_bf16in_bf16out` uses 1-thread-per-row dispatch.
A v2 SIMD-parallel version exists but was explicitly **ruled out** in a previous experiment:

> "v2 (SIMD-parallel) dispatch was tried for attention matmuls but regresses
> performance due to high TG-scheduling overhead (e.g. wq_b needs 16384 TGs vs 128
> for naive). Apple GPU L2 cache (32-64MB) absorbs non-coalesced reads for these
> relatively small matrices."

The naive dispatch with `MTLSizeMake(out_dim, 1, 1), threads=256` is already optimal
for these sizes on Apple Silicon with unified memory + large L2.

### 2. Merge merged_cb + cb2cmd2 (eliminate 1 GPU sync/layer)

Currently: merged_cb [commit+wait] → CPU readback → cb2cmd2 [commit+wait]

The CPU readback between them reads `attn_input`, `normed`, `post/comb` for the
compressor/indexer (DSA) and for routing. If DSA compressor/indexer can be deferred
or read from GPU buffer after cb2cmd2, the merge saves 1 sync/layer.

**Estimated savings**: depends on actual GPU sync overhead (need to measure).

### 3. Reduce CPU memcpy between CBs

Multiple per-layer CPU↔GPU copies: normed (DIM×4=16KB), residual (4×DIM×4=64KB),
ffn_input (DIM×4=16KB), ffn_out (DIM×4=16KB). Total ~112KB/layer × 43 = ~4.8MB/token.
On M4 Pro unified memory, memcpy cost is minimal (~0.5ms/layer), probably not significant.

### 4. Profile with Metal GPU timestamps

Use `cb.GPUStartTime`/`cb.GPUEndTime` to measure actual GPU execution time per CB,
separating GPU compute from CPU scheduling/waiting overhead. This will reveal whether
the 52ms/layer is mostly GPU compute or mostly CPU↔GPU sync latency.

## GPU Timestamp Profiling Results (2nd decode token, L3-L42)

Using `cb.GPUStartTime`/`cb.GPUEndTime` (hardware GPU timer, zero overhead):

| CB | GPU time avg | GPU time total (40 layers) |
|----|-------------|---------------------------|
| merged_cb (13 encoders: mhc+Q+KV+SDPA+wo) | **4.3ms** | 172ms |
| cb2cmd2 (7 encoders: mhc_post+pre+norm+route) | **0.7ms** | 28ms |
| **GPU subtotal** | **5.0ms** | **200ms** |

### CPU-side profiling (clock_gettime around operations):

| Operation | Per-layer avg | 40 layers total |
|-----------|--------------|-----------------|
| merged_cb waitUntilCompleted (wall) | 5.0ms | 199ms |
| cb2cmd2 waitUntilCompleted (wall) | 1.1ms | 44ms |
| compressor_step (CPU affine matvec) | **5.1ms** | **204ms** |
| wait overhead (wall - GPU) | 0.2-0.5ms | ~15ms |

### Key Insight

- **GPU executes in ~5ms/layer** — already fast, limited by bandwidth
- **CPU compressor_step costs 5.1ms/layer** — same magnitude as entire GPU!
- **waitUntilCompleted overhead is negligible** (0.2-0.5ms)
- **Total measured: ~11ms/layer.** Actual total: ~23ms/layer (without profiling fprintf overhead)

The remaining time per layer is:
- MoE forward (GPU + I/O): ~13ms
- Shared expert (GPU): ~3ms  
- cb3 mhc_post_ffn (GPU): ~1ms
- CPU memcpy: ~1ms

**Without any fprintf profiling, true decode = ~23ms/layer × 43 = ~990ms + prefill overhead → 0.5 tok/s matches 5 tokens in 10s (including 4-token prefill at ~1.5s).**

## Root Cause: Compressor CPU Matmul

`moe_infer_compressor_step` performs 2 CPU affine 4-bit matvecs per layer:
- CSA layers (ratio=4): [1024, 4096] × 2 = 8.4M ops each
- HCA layers (ratio=128): [512, 4096] × 2 = 4.2M ops each

Using `cpu_affine_matvec_safe` — naive C loop, single-threaded, no SIMD.
Total: 5.1ms × 41 layers (all except L0, L1, L43) = **209ms/token (21% of decode time)**

## Optimization Plan

### Priority 1: Move compressor matmuls to GPU (−209ms, +25%)

The compressor wkv/wgate weights are fixed per layer (quantized affine 4-bit, same format
as attention weights). They should use the same `dequant_matvec_affine` GPU kernel that
attention already uses. Pre-allocate GPU buffers for comp_wkv/comp_wgate at init time.

Expected: 5.1ms → <0.1ms per layer (GPU matvec for [1024,4096] is ~0.2ms from cb2cmd2 data).

### Priority 2: Overlap MoE with cb2cmd2 (−44ms, +5%)

cb2cmd2 computes routing (which experts to load). The MoE I/O (pread from SMELT pool) could
start AS SOON as routing results are available, overlapping with cb2cmd2's waitUntilCompleted.
Currently, I/O starts AFTER cb2cmd2 wait completes.

### Priority 3: Reduce MoE I/O (−100ms, +10%)

Even with SMELT N=51 (top-51 cached), some experts still need pread (~5ms/layer).
Increasing N or using predictive prefetch (pre-read next layer's likely experts based
on token embedding similarity) could eliminate most I/O.

## Attention Implementation Note

**NOT FlashMLA.** Uses custom `mla_sdpa_decode_bfloat` — simple per-head kernel (32 threads,
serial KV iteration). Adequate for short sequences (pos<100 → <1ms).
For long-context scenarios, FlashMLA tiled kernel would be needed.

