# GPU Routing Feasibility Analysis
date: 2026-06-10

## Objective

Move MoE routing (top-K expert selection) from CPU to GPU, eliminating the
mandatory GPU sync after CMD2 that currently forces `waitUntilCompleted`
to read gate scores for CPU-side top-K.

---

## Current dmlx Routing Flow (CPU-side)

```
CMD2 (6 encoders in one CB):
  Enc1: mhc_pre_ffn
  Enc2: ffn_RMSNorm
  Enc3: gate_proj matmul → buf_routing_scores (GPU writes)
  Enc4: mhc_post_attn
  Enc5: mhc_pre_ffn (overlap)
  Enc6: routing gate
→ waitUntilCompleted   ← FORCED because CPU needs gate scores
→ CPU reads buf_routing_scores
→ CPU: sqrtsoftplus(scores) + bias + SMELT penalty + top-6 selection
→ CPU: L1-normalize × 1.5
→ SMELT cache lookup → expert_buf[k] pointers
→ MoE GPU dispatch
```

**Why the wait is unavoidable now**: CPU must read gate scores to do top-K,
then select which expert buffers to load/dispatch.

---

## ds4 GPU Routing Implementation

ds4 uses 2 Metal kernel dispatches (all in the same batch CB, no sync):

### Step 1: `kernel_dsv4_softplus_sqrt_f32_4`
- Input: `logits[256]` (f32)
- Output: `probs[256]` = sqrt(softplus(logit)) for each expert
- Dispatch: 256/4 = 64 threads (processes 4 elements per thread)

### Step 2: `kernel_dsv4_router_finalize_one`
- Input: `probs[256]`, optional `bias[256]`
- Output: `selected[6]` (int32 indices)
- Algorithm: bitonic sort on 256 elements using threadgroup memory
  - 256 threads, 1 threadgroup
  - threadgroup memory: 256 × float (scores) + 256 × int32 (indices) = 2KB
  - Supports bias addition, hash-mode routing (lookup table), and token buffer
- Important: handles group constraints for DeepSeek routing

### Step 3: `kernel_dsv4_router_weights_one`
- Input: `probs[256]`, `selected[6]`
- Output: `weights[6]` = probs[selected[k]] / sum × 1.5
- Dispatch: 6 threads (one per expert)

**Total**: 3 encoders, 0 CPU syncs, all in same command buffer.

---

## dmlx-specific Requirements

dmlx routing must match `cpu_moe_route()` exactly:

```c
1. scores[i] = sqrt(log(1 + exp(logits[i])))      // sqrtsoftplus
2. biased[i] = scores[i] + bias[i]                 // e_score_correction_bias
              - smelt_penalty if !cache_ptr[i]     // SMELT penalty
3. top-6 by biased scores
4. weights[k] = scores[selected[k]]               // original scores (not biased)
5. weights[k] = weights[k] / sum × 1.5            // L1-normalize × route_scale
```

**SMELT penalty complication**: dmlx applies `-1e9` penalty to uncached experts.
In SMELT mode with routing bias active, all 6 selected experts are guaranteed
to be in cache (bias steers routing). So the penalty logic only fires during
warmup or when cache misses occur. Post-warmup: penalty doesn't change top-6
selection (all cached experts have unpenalized scores).

**Hash routing (layers 0-2)**: Uses `tid2eid[token_id]` lookup table instead
of top-K. ds4 handles this in `kernel_dsv4_router_finalize_one` with hash_mode.
dmlx needs the same. The hash lookup can be done on GPU: same as ds4.

---

## Feasibility Assessment

### Can we port ds4's kernels directly?

**Yes, with minor adaptations**:

1. `sqrtsoftplus`: Pure element-wise, trivial to implement in Metal.
2. `router_finalize_one` (bitonic sort top-6 of 256): ds4 has the exact code,
   works for DSV4 (N_EXPERTS=256, N_ACTIVE=6). Can be copied with SMELT penalty added.
3. `router_weights_one`: 6-thread normalization, trivial.

### What about SMELT penalty on GPU?

In steady-state SMELT mode (all hot experts cached, penalty doesn't fire):
- We can simplify: just add bias to logits, skip penalty.
- Output of routing is `selected[6]` as GPU buffer.

### What about expert buffer dispatch?

This is the KEY difference:

- **ds4**: all expert weights are mmap'd into one buffer. The `selected[6]`
  GPU buffer is passed directly to `ds4_gpu_routed_moe_one_tensor()` which
  reads the correct expert offsets from `selected` × expert_stride.

- **dmlx SMELT**: expert data is in `expert_mem_pool[layer]` (RAM, Shared).
  The pool is contiguous: `pool[slot × EXPERT_SIZE]`. The `smelt_pool_pos[layer][eid]`
  table maps `expert_id → pool_slot`.

**This is solvable**: we already have `buf_gather_gate_W[layer]` which is a
NoCopy MTLBuffer over the entire pool. The gather kernels already use
`pool_pos × EXPERT_SIZE_U32` as the expert offset. But currently `pool_pos`
is looked up on CPU and written to `buf_gather_expert_ids`.

**If routing is on GPU**: GPU outputs `selected[6]` as expert_ids. We need
a GPU kernel that converts `expert_id → pool_slot` using `smelt_pool_pos`
table (which is a CPU array). This table can be uploaded as a GPU buffer once
at SMELT warmup.

---

## Concrete Implementation Plan

### Phase 1: GPU routing kernels (no architectural change yet)

1. Add `dsv4_moe_route` Metal kernel:
   - 256 threads, 1 TG
   - Input: `logits[256]`, `bias[256]`, TG memory for bitonic sort
   - Output: `selected[6]` (int32), `scores[256]` (intermediate)

2. Add `dsv4_moe_weights` Metal kernel:
   - 6 threads
   - Input: `scores[256]`, `selected[6]`
   - Output: `weights[6]` (f32)

3. Add `smelt_pool_pos_buf[N_LAYERS]` GPU buffer: upload `smelt_pool_pos[layer]`
   table (256 × int32 = 1KB) once after SMELT warmup.

4. Add `dsv4_expert_id_to_pool_pos` kernel (or blit+remap):
   - Input: `selected[6]`, `pool_pos_table[256]`
   - Output: `pool_pos[6]` for gather kernel dispatch

### Phase 2: Eliminate CMD2 waitUntilCompleted

With GPU routing, CMD2 no longer needs a CPU wait after it. The flow becomes:

```
CMD2 (encoders):
  mhc_post_attn (f32→bf16 + mhc_post + bf16→f32)
  mhc_pre_ffn + ffn_norm + gate_proj  ← writes buf_routing_scores
  dsv4_moe_route                      ← reads buf_routing_scores, writes selected[6]
  dsv4_moe_weights                    ← writes weights[6]
  dsv4_expert_id_to_pool_pos          ← writes pool_pos[6]
→ commit async (no wait!)
```

Then MoE dispatch uses `pool_pos[6]` from GPU buffer, and gather mode
kernels access expert data directly.

**Remaining CPU sync**: Only for reading `normed_bf16_direct` (attention input)
and `attn_out` (from CB1) — these are genuinely required.

### Expected after Phase 1+2:

Layers per token:
- CB-A (mhc_pre_attn + norm): 1 sync (normed_bf16_direct needed for attention)
- CB1 (Q/KV + SDPA): 1 sync (attn_out needed)
- CMD2 (mhc_post + routing + weights → async): 0 sync
- MoE + shared + cb3: 0 sync (deferred/chained)
- **Total: 2 syncs/layer × 43 = 86 syncs/token** (vs current 258+)

Estimated savings: reduce from ~300ms attention-side sync overhead to ~114ms.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Bitonic sort precision mismatch | Low | Validate against cpu_moe_route() |
| SMELT penalty during warmup causes wrong routing | Medium | Keep CPU fallback during warmup |
| Hash routing layers (0-2) work differently | Low | ds4 already handles hash_mode |
| pool_pos lookup adds latency | Very low | 256 × int32 table lookup is trivial |
| gather mode still slow (cache-unfriendly) | High | Known issue — but at least removes CPU sync |

**Biggest risk**: The gather kernel reads 6 experts × 13.4MB = 80MB with
13MB stride, which was measured to be 10-15× slower than separate mode.
GPU routing only helps if we can also make MoE dispatch fast.

**Mitigation**: Keep separate expert dispatch (6 per-expert kernels),
just pass the routing result through GPU buffer to eliminate the CPU sync.
The `expert_bufs[]` pointers can be derived from `pool_pos[6]` on CPU
via a tiny memcpy of 6 pointers — CPU doesn't need to wait for GPU routing
to complete to *start* preparing buffers, only before submitting MoE kernel.

Actually this is still a problem: CPU needs pool_pos to set up expert_bufs
before MoE dispatch. Unless we use gather mode.

---

## Revised Plan (Simpler)

Instead of full GPU routing pipeline, use a **hybrid approach**:

1. Add GPU routing kernels to CMD2 (async commit)
2. After CMD2 commit (non-blocking), CPU reads `selected[6]` from the GPU
   buffer **after** CB-A+CB1 attention completes (using that time to let
   GPU finish routing)
3. CPU does expert buffer setup from `selected[6]`
4. MoE dispatch proceeds normally

This way:
- CMD2 commits without waiting
- CPU does attention (CB-A + CB1) — during this time GPU finishes CMD2+routing
- When CB1 wait completes, CMD2+routing is also done (GPU serializes them)
- CPU reads selected[6] (tiny: 6 × int32 = 24 bytes)
- MoE dispatch with normal expert buffers

**This eliminates the CMD2 wait entirely**, at the cost of one tiny CPU read
(24 bytes) after CB1 completes. The GPU time for CMD2+routing overlaps with
CB1 attention execution.

This is the **correct and achievable** plan.

---

## Conclusion

**Feasible**: Yes. The GPU routing kernels are straightforward (ds4 code can
be adapted directly). The revised hybrid plan eliminates the CMD2 wait without
requiring gather mode.

**Expected gain**: 1 × waitUntilCompleted/layer × 43 layers × ~13ms = ~560ms/token.

**Prerequisites**:
- `smelt_pool_pos` table as GPU buffer (trivial, 43 × 1KB)
- GPU routing kernels (3 encoders, adapted from ds4)
- CMD2 must commit async

**Implementation sequence**:
1. Add Metal kernels (sqrtsoftplus + top-6 + weights)
2. Upload smelt_pool_pos as GPU buffer at warmup
3. Append routing kernels to CMD2, commit async
4. After CB1 wait: read selected[6] from GPU, do normal SMELT cache lookup
5. MoE dispatch proceeds as before

**This is a clean, testable incremental change.**
