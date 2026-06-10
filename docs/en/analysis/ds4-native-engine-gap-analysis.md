# Native Engine Performance Gap Analysis: dmlx vs ds4
date: 2026-06-10

## Summary

ds4 achieves **22 tok/s** on M4 Max (48GB) running DeepSeek V4 Flash.
dmlx native engine achieves **~0.1 tok/s** on M3 Max (48GB) with SMELT_N=51.

This document explains exactly why and what it would take to close the gap.

---

## Root Cause: GPU Sync Count

| | dmlx (current) | ds4 |
|---|---|---|
| GPU waitUntilCompleted per token | **~172–215×** (4–5 per layer × 43 layers) | **1×** |
| GPU CommandBuffers per token | ~172–215 | 2 |
| Encoder open/close per token | ~172–215 | 2 |

This alone explains most of the gap. Each `waitUntilCompleted` on Apple Silicon costs
~2–13ms overhead (measured: ~13ms in SMELT mode). At 172 syncs × 13ms = **2.2 seconds**
of pure synchronization overhead per token — before any GPU computation.

---

## How ds4 Achieves 1 Sync/Token

### Batch Command Buffer Pattern

```objc
// Every decode step:
ds4_gpu_begin_commands()          // open CB #1
metal_graph_encode_token(...)     // encode ALL 43 layers into CB #1/#2
  // after layer 4: flush_commands() → commit CB #1 async, open CB #2
  //                CPU continues encoding layers 5..42 into CB #2
  //                GPU runs CB #1 concurrently
ds4_gpu_end_commands()            // commit CB #2, wait(CB#1), wait(CB#2)
logits readback                   // single memcpy at the end
```

**One encoder is reused** across all kernel dispatches within a CB (`g_batch_enc`).
Metal encoder open/close is expensive; ds4 keeps it open for the entire CB.

### GPU-Only Routing

MoE routing (top-K selection for 256 experts) is done entirely on GPU:
- `kernel_dsv4_router_finalize_one` — top-6 selection in one GPU dispatch
- `kernel_dsv4_router_weights_one` — weight normalization in one GPU dispatch
- Outputs `router_selected[6]` and `router_weights[6]` as GPU buffers
- Next kernel (`routed_moe_one_tensor`) reads them directly from GPU — **no CPU sees routing results**

### GPU-Only KV Cache

KV cache is fully GPU-resident. The ring-buffer write index `pos % raw_cap` is
computed CPU-side from the position counter (no GPU readback needed). FP8-quantized
in the head's non-RoPE half. CPU never touches cache contents during decode.

### Expert Data via mmap

All expert weights are mmap'd into a single contiguous buffer at startup
(`model->map + abs_offset`). GPU kernels access them directly via MTLBuffer NoCopy.
**No pread during decode** — the OS page cache handles caching automatically.

### Key Fused Kernels

ds4 fuses operations that dmlx does separately:
- `dsv4_qkv_rms_norm_rows` — Q-norm + KV-norm in one kernel
- `hc_split_weighted_sum_norm` — HC Sinkhorn + RMS norm + split
- `shared_down_hc_expand_q8_0` — shared-expert down-proj + add routed + HC post-update
- `shared_gate_up_swiglu` — gate + up + SwiGLU
- `router_finalize_one` — softplus/sqrt + topK + bias + group constraints

---

## dmlx Bottlenecks (Measured)

Per-layer timing at SMELT hot state (pos=7):
```
[MOE-TIME] gpu=~10ms
[TIME] total=~60–90ms
→ non-MoE overhead: ~50–80ms/layer
  = ~4–5 × 13ms GPU sync overhead
  + attention GPU compute (~30–40ms)
  + mHC CPU overhead
```

Per-token (43 layers × ~75ms average):
```
GPU sync overhead:   172 syncs × ~13ms  = ~2.2s
MLA attention GPU:   ~40ms × 43         = ~1.7s
MoE GPU:             ~10ms × 43         = ~0.4s
mHC pre/post CPU:    ~5ms × 43          = ~0.2s
─────────────────────────────────────────────────
Theoretical minimum (after sync removal):  ~2.3s/token ≈ 0.43 tok/s
Current measured:                         ~8–9s/token ≈ 0.11 tok/s
```

---

## Why dmlx Has So Many Syncs

### 1. CPU-Side Routing (Forces GPU Sync After Gate Projection)

In `moe_infer_forward_layer`:
```
CMD2 (gate proj GPU) → waitUntilCompleted → CPU top-K → expert pread → MoE GPU
```
CMD2 includes the routing gate matmul. The CPU reads gate scores to do `cpu_moe_route()`.
This **requires** a GPU sync after CMD2.

**ds4**: routing top-K is done in a GPU kernel, no CPU sees the gate scores.

### 2. mHC Requires CPU Readback (3 Syncs/Layer for mHC Alone)

mHC (HyperConnection) needs `post[]` and `comb[]` arrays on CPU for the next kernel
dispatch (to set `setBytes:`). Currently:
```
CB-A → wait → CPU reads post/comb → cb2 → wait → CMD2 → wait → cb3 → wait
```
4 syncs just for mHC, per layer.

**ds4**: No mHC. DS4 also implements HyperConnection but does it fully GPU-side
with persistent GPU buffers — `post` and `comb` weights stay in GPU buffers and
are consumed by next kernels directly.

### 3. Expert pread Blocks Between CMD2 and MoE GPU

In SSD mode:
```
CMD2 wait → routing → pread (6–12ms) → MoE GPU
```
Even in SMELT mode (RAM), the SMELT cache lookup returns immediately but the
architectural split between routing and MoE GPU still requires a CPU sync point.

**ds4**: Expert data is mmap'd, GPU reads directly via `abs_offset` into the
already-mapped weight buffer. No pread, no explicit IO step.

---

## Remediation Plan and Expected Gains

### Priority 1: mHC GPU-Only (Estimated: -1.1s/token)

Currently mHC uses 4 separate CBs per layer with CPU readback between each.
Move `post[]` and `comb[]` to persistent GPU buffers; kernels read them directly.

**Removes**: 3 of the 4 mHC-related syncs per layer
**Saves**: ~3 × 13ms × 43 = ~1.7s/token (conservative: ~1.1s/token net)
**Difficulty**: Medium — requires adding persistent GPU buffers for post/comb scalars

### Priority 2: GPU Routing Kernel (Estimated: -560ms/token)

Implement top-K routing as a GPU kernel. Output `selected[6]` and `weights[6]`
as GPU buffers consumed by the MoE kernel.

**Requires**: Expert data available to GPU without CPU-side selection
**Currently blocked by**: SMELT pread model (CPU needs to know which experts to load)
**Unblocked if**: Using mmap (experts always available to GPU) or SMELT 35GB all in GPU-mapped pool

**Saves**: 1 sync × 13ms × 43 = ~560ms/token

### Priority 3: Expert mmap (Prerequisite for GPU Routing)

Replace SMELT 35GB pread model with mmap of all expert data.
48GB machine: expert data per layer ≈ 256 × 13.4MB = 3.4GB × 43 layers = 146GB → too large.

**Feasible partial**: mmap the layer files, let OS page cache decide what to keep.
Same approach as ds4's "Trust the OS".

**Memory**: ~146GB virtual address space (no physical requirement), OS handles caching.
**Risk**: Cold start latency on cache miss (each 13.4MB expert: ~0.8ms cold read).
**Expected steady state**: OS caches hot experts (~35GB working set), cold misses only on routing variance.

### Priority 4: Encoder Reuse (Estimated: -50ms/token)

Reuse a single `MTLComputeCommandEncoder` across kernel dispatches within one CB.
Currently every `setComputePipelineState + setBuffer + dispatch + endEncoding` is
a separate encoder open/close.

**Saves**: encoder open/close overhead, estimated ~1ms × 43 layers = ~43ms/token.

---

## Theoretical Ceiling

With all optimizations applied:
```
MLA attention GPU:         ~40ms × 43  = 1.72s
MoE GPU:                   ~10ms × 43  = 0.43s
mHC GPU (optimized):       ~3ms × 43   = 0.13s
Routing GPU:               ~1ms × 43   = 0.04s
GPU sync (2 per token):    2 × ~13ms   = 0.03s
─────────────────────────────────────────────
Total:                     ~2.35s/token ≈ 0.43 tok/s
```

**Gap to ds4 (22 tok/s)**: ds4's attention is ~5× faster because:
- Q8 quantization instead of BF16 (half the memory bandwidth)
- Flash Attention kernel (O(1) memory vs O(n²)) 
- Fused Q/K/V projections in one kernel pass
- mHC is fully fused into attention kernels

**Realistic target for dmlx**: 0.4–0.5 tok/s after Priority 1+2 above (~18–20× improvement over current 0.11 tok/s).

---

## Action Plan

1. **Immediate**: mHC GPU-only (Priority 1) — highest ROI, no architectural change needed
2. **Short term**: Expert mmap + GPU routing (Priority 2+3) — removes the fundamental pread bottleneck
3. **Long term**: Fused MLA attention kernel matching ds4's approach — closes the remaining gap

The current Path B work (GPU-resident residual) was a step in the right direction
(eliminated 3 CPU memcpy per layer) but hit a wall because mHC still forces
4 GPU syncs per layer. The next step must target mHC.
