# Phase Q2 Layer Transaction/Dependency Partition Design

> Task 2.1 deliverable — read-only design analysis. No runtime behavior changed.

## 1. Current layer execution sequence

Per `moe_infer_forward_layer_impl` (engine.c), each decode layer currently
executes the following steps in strict sequence, one after the other:

```
merged_cb (CB-A+CB1) commit+wait      ← MLA attention (Q/KV/SDPA + wo_b)
CPU: read attn_out, normed, normed_bf16
cb2cmd2 commit+wait                   ← mhc_post(attn) + mhc_pre(ffn) + routing gate + GPU topK
CPU: read residual, ffn_input, normed (ffn), routing results
CPU: route resolution (SMELT/prefetch/CPU topK)
CPU: memcpy normed → buf_normed
CPU: SMELT hit resolution
CPU: prefetch_consume
CPU: build miss_ids[], miss_bufs[]
io_pool_dispatch(n_miss preads)       ← BLOCKS until all cold reads complete
CPU: expert_data[] pointers set
CPU: prefetch_issue(layer+1)          ← async next-layer pread (cross-layer, not in scope)
moe_forward_layer                     ← routed expert CB (gather6 + moe_combine) + wait
CPU: memcpy buf_hidden → ffn_out
shared expert CB commit+wait          ← gate + up + SwiGLU + down using buf_shared_w[layer]
CPU: ffn_out[j] += sv[j]
cb3 (mhc_post_ffn) commit+wait
CPU: memcpy buf_residual_gpu → residual
```

The shared expert CB runs **after** `io_pool_dispatch` blocks and after the
routed expert CB completes, even though it has no data dependency on cold reads.

## 2. Dependency partition table

Each unit of work per layer is classified by its data dependencies:

| Work unit | Inputs needed | Cold I/O dependency | Classification |
|---|---|---|---|
| `buf_normed` memcpy | `normed[]` (from cb2cmd2) | **None** | Independent |
| Route metadata prep (`expert_weights[]`, rank order) | Route scores (from cb2cmd2) | **None** | Independent |
| SMELT/prefetch/batch-pool hit resolution | `expert_mem_cache`, `expert_buf_pred` | **None** | Independent |
| `miss_ids[]`, `miss_bufs[]` build | Route IDs | **None** | Independent |
| **Shared expert CB** (gate+up+SwiGLU+down) | `normed` (buf_mhc_attn_in via bx), `buf_shared_w[layer][0..8]` | **None** | **Independent — eligible for read window** |
| Hot/prefetch routed expert kernels | `expert_data[k]` where pointer already valid | **None for hot** | Independent (hot only) |
| `io_pool_start` (pread submit) | `miss_ids[]`, `miss_bufs[]` | N/A (the submit itself) | Async submit point |
| `io_pool_wait_dependency` (join) | Cold pread results | **YES** | Cold join point |
| `moe_forward_layer` routed CB | `blob_bufs[k]` = NoCopy(expert_buf[k]) — cold bytes | **YES** | Cold-dependent |
| `moe_combine` (inside routed CB) | `buf_expert_contiguous` (rank-indexed) | Indirect YES | Cold-dependent |
| `ffn_out[j] += sv[j]` (shared output add) | shared CB result + routed combine result | Both needed | Combine point — after join |
| `cb3` (mhc_post_ffn) | `ffn_out`, `residual` | Indirect | After combine |

### Arithmetic order invariants

The following ordering properties must be preserved by any reordering:

1. **Route ranks are frozen** at routing time (before any I/O). `moe_combine`
   iterates `buf_expert_contiguous` in rank-indexed order (rank 0..K-1 at
   contiguous offsets); async completion never defines combine order.

2. **Shared expert output** is added via `ffn_out[j] += sv[j]` after
   `moe_combine` writes `buf_hidden`. This order (`routed_sum + shared_out`)
   must be preserved: shared CB may complete before or after cold join, but
   the CPU `+=` can only execute after both routed and shared GPU results exist.

3. **`buf_normed` is read-only** during shared expert execution — it is written
   once (memcpy from `normed`) and never mutated until the next layer. Shared
   expert kernel (`pipe_dequant_matvec_affine` uses `bx = buf_mhc_attn_in`,
   which is updated from `normed` before the shared CB) does not conflict with
   the routing or pread paths.

4. **No cross-layer route speculation** is permitted (per R4.2). Only
   `prefetch_issue(layer+1)` fires an async pread for the next layer's
   *predicted* experts, but routes are not re-used across layers.

## 3. Target execution sequence

Moving the shared expert CB into the read window, submitting cold preads first:

```
merged_cb (CB-A+CB1) commit+wait           ← unchanged
cb2cmd2 commit+wait                        ← unchanged
CPU: route resolution + metadata prep
CPU: memcpy normed → buf_normed            ← unchanged (needed by routed CB only)
CPU: build miss_ids[], miss_bufs[]
io_pool_start(n_miss preads)               ← submits tasks and returns generation token
                                              NOTE: io_pool_start first calls io_pool_wait
                                              to drain any prior generation on the same pool;
                                              under normal decode flow the pool is idle so
                                              this returns immediately. Task 2.2 must call
                                              io_pool_start directly (not io_pool_dispatch)
                                              to split submit from join.
── READ WINDOW (independent work) ──────────────────────────────────
shared expert CB commit+wait               ← MOVED: runs during cold reads
──────────────────────────────────────────────────────────────────────
io_pool_wait_dependency(generation, ...)   ← join: blocks only if reads still in-flight
CPU: expert_data[] pointers set
moe_forward_layer                          ← routed CB + moe_combine + wait
CPU: ffn_out[j] += sv[j]                   ← ordering preserved: += after both results
cb3 commit+wait                            ← unchanged
```

### Why this is safe

- `io_pool_start` posts K `pread` tasks to I/O worker threads and returns a
  generation token. It first calls `io_pool_wait` to drain any prior generation on
  the same pool; under normal single-decode flow the pool is idle so this is a
  no-op. After it returns, tasks are in-flight.
- Task 2.2 must use `io_pool_start` directly (not `io_pool_dispatch`, which
  combines start + wait into one blocking call) and insert the shared expert CB
  between start and `io_pool_wait_dependency`. This is a targeted refactoring:
  split `io_pool_dispatch` at its existing call site into `io_pool_start` +
  shared-CB + `io_pool_wait_dependency` + error check.
- Shared expert CB has zero dependency on `expert_buf[k]`, `miss_bufs[k]`,
  or any `blob_wrap` slot. It reads only `buf_shared_w[layer][0..8]` (always
  present, initialized first-token and persistent) and `buf_mhc_attn_in` (bx,
  written before this function call).
- `io_pool_wait_dependency` (line 2744 in engine.c) records `IO_EXPOSED_WAIT`
  only when it actually blocks. If cold reads complete before the shared expert
  CB does, the join returns immediately with zero exposed wait.
- If cold reads fail (short read or I/O error), `io_pool_wait_dependency`
  still returns, `io_status` is set non-zero, and the layer returns `-1` before
  any routed expert CB is encoded — no partial output is published.
- **Buffer safety**: shared CB and routed CB share `buf_hidden` (shared CB as
  gate scratch, routed CB as combine output). Safety is guaranteed by temporal
  sequencing — shared CB commits+waits before routed CB is encoded. See §5.

## 4. State machine

```
[*] → Routed
  After cb2cmd2 wait, route resolution, metadata prep

Routed → ReadsSubmitted
  io_pool_start(n_miss) called; generation token held
  [if n_miss == 0: skip to IndependentWork directly]

ReadsSubmitted → IndependentWork
  Shared expert CB submitted and executing

IndependentWork → WaitingForDependencies
  Shared expert CB committed and waited

WaitingForDependencies → ColdWork
  io_pool_wait_dependency(generation) returns; all cold payloads valid

WaitingForDependencies → Failed
  io_pool_wait_dependency returns but io_status != 0 (short read / error)

ColdWork → Combined
  moe_forward_layer (routed CB) committed and waited
  ffn_out[j] += sv[j] executed

ColdWork → Failed
  GPU error during routed CB

Combined → Committed
  cb3 (mhc_post_ffn) committed and waited; layer output published

Failed → Drained
  All I/O owners and GPU CBs drained; layer returns -1
```

### Failure invariants

- **Submit failure** (n_miss > 0, io_pool_start returns bad generation): treat
  as n_miss = 0 and skip cold work; routed experts for cold ranks receive
  zeroed expert_data; layer returns -1 before encoding any routed CB.
- **Short read / read error**: `io_pool_wait_dependency` returns; `io_status != 0`;
  layer returns -1 without calling `moe_forward_layer`. Partial layer output
  is never published.
- **GPU error** (Metal CB status != Complete): layer returns -1 after draining
  the failed CB; no residual update committed.
- **Cancellation** (request abort from `native_engine.zig`): abort path calls
  `exact_q4_abort_request` which drains I/O pools and deferred GPU work before
  releasing request storage. No partial result reaches the caller.

## 5. Command-buffer and buffer conflict proof

> **Safety note**: The shared expert CB and routed expert CB share some MTLBuffer
> objects (`buf_hidden`, `buf_attn_out`). Safety is guaranteed by **temporal
> sequencing** — the shared expert CB is fully committed and waited before the routed
> expert CB is encoded. Buffer disjointness does NOT hold and must not be assumed.
> Any future Task 2.2 implementation that introduces GPU-side pipeline parallelism
> between the two CBs would create a silent write-write race and is forbidden.

The shared expert CB buffer assignments (engine.c:4447–4449):
```c
bx    = buf_mhc_attn_in  // input: normed ffn activation [DIM]
bgate = buf_hidden        // gate intermediate output [INTERMEDIATE]
bup   = buf_h_mid         // up intermediate output [INTERMEDIATE]
bdown = buf_attn_out      // down projection output [DIM]
```

The routed expert CB (`moe_forward_layer`, engine.c:3313):
- `gather6_gate_up` reads `blob_bufs[k]` (cold data) and `buf_normed`; writes `buf_expert_mid[k]`
- `gather6_down` reads `buf_expert_mid[k]`; writes `buf_expert_contiguous`
- `moe_combine` reads `buf_expert_contiguous`; writes **`buf_hidden`** (atIndex:3)

| Buffer | Written by shared CB | Read/Written by routed CB | Alias? | Safe? |
|---|---|---|---|---|
| `buf_mhc_attn_in` (bx) | No (read only) | No | No | ✓ |
| `buf_hidden` (bgate) | **Writes** (gate intermediary) | **Writes** (moe_combine output) | **YES** | ✓ by sequencing |
| `buf_h_mid` (bup) | **Writes** (up intermediary) | No | No | ✓ |
| `buf_attn_out` (bdown) | **Writes** (down output) | No | No | ✓ |
| `buf_shared_w[l][0..8]` | Reads (weight data) | No | No | ✓ |
| `buf_normed` | No | Reads (gather6 input) | No | ✓ |
| `blob_bufs[k]` (cold data) | No | **Reads** (gather6_gate_up) | No | ✓ |
| `buf_expert_contiguous` | No | **Writes** (gather6_down) | No | ✓ |

**buf_hidden alias**: shared CB uses `buf_hidden` as a temporary intermediate buffer for
the gate projection output (`[INTERMEDIATE]` elements). `moe_combine` writes `buf_hidden`
with the routed combine result (`[DIM]` elements). These do not race because the shared
CB completes and waits **before** the routed CB is encoded. After the shared CB commits,
the CPU reads `bdown` (= `buf_attn_out`) for the `ffn_out += sv[j]` step. At that point,
`buf_hidden` has completed its role as a gate scratch buffer and is safe for moe_combine
to overwrite.

**buf_attn_out**: used as the down-projection output of the shared expert CB. The
attention `wo_b` projection also writes `buf_attn_out` (engine.c:~3795), but that write
occurs in `merged_cb` which is committed and waited before the MoE block begins. After
`merged_cb`, the CPU reads `attn_out` from `buf_attn_out` contents, copies to `float
attn_out[]`, and no further engine CB reads `buf_attn_out` until the next layer's
`merged_cb`. The shared expert CB is therefore the sole writer of `buf_attn_out` during
the MoE block; no read-after-write hazard with any other in-window CB.

**Conclusion for Task 2.2**: The proposed ordering (shared CB committed+waited → routed
CB encoded) is safe. The two CBs must NOT be submitted to the GPU concurrently. Any
implementation that submits them to the same `commandBufferWithUnretainedReferences` or
to overlapping Metal pipelines would create a `buf_hidden` write-write race.

## 6. Scope limits (what Task 2.2 must NOT do)

Per R4.2 and the stop gate in tasks.md:

- No cross-layer route speculation (next-layer prefetch already exists and is unchanged)
- No increase in physical pread calls (same K cold misses, same io_pool workers)
- No change to `moe_combine` rank ordering (buf_expert_contiguous remains rank-indexed)
- No change to Q4 bytes (same expert data, same kernels)
- No partial layer output on any failure (all failure paths must return -1 with no GPU residual written)

If any of these cannot be guaranteed under the target ordering, Task 2.2 must
record `REJECTED` per the stop gate and leave the current ordering unchanged.

## 7. Test matrix for Task 2.1 focused tests

The tests (`tests/test_q2_layer_partition.c`) verify the partition logic
independently of Metal by driving the io_pool and a simulated layer state.

| Test case | n_miss | n_hot | n_prefetch | Expected state |
|---|---|---|---|---|
| All hot (0 miss) | 0 | 6 | 0 | No pread; join immediate; routed CB fires |
| All cold (6 miss) | 6 | 0 | 0 | Pread fires; join may block; routed CB fires after join |
| Mixed 3 hot 3 cold | 3 | 3 | 0 | 3 preads; shared CB runs in window |
| Mixed 2 hot 2 prefetch 2 cold | 2 | 2 | 2 | 2 preads |
| Submit failure | — | — | — | n_miss skipped; returns -1 before routed CB |
| Short read (1 of 3) | 3 | 0 | 0 | io_status != 0; returns -1; no routed CB |
| All cold, GPU error (routed CB) | 6 | 0 | 0 | Drained; -1 returned |
| Cancellation mid-window | 3 | 3 | 0 | Drain; abort; no partial output |
| Pinned lifetime: shared CB overlaps cold reads | 3 | 0 | 3 | Shared CB completes; cold join returns; combine correct |
| Arithmetic order: rank order preserved | 6 | 0 | 0 | moe_combine rank 0..5 matches route_ranks[] |

For each case, the test asserts:
- `io_pool_wait_dependency` is called iff n_miss > 0
- `IO_EXPOSED_WAIT` span recorded iff join actually blocked
- Layer returns -1 on any I/O or GPU failure
- `ffn_out[j] += sv[j]` executes only after both routed and shared results valid
- Route ranks are unchanged between routing time and moe_combine call

## 8. Oracle dry run

The oracle verifies that the partition model is consistent with the existing
Task 0.3 telemetry evidence:

- `IO_EXPOSED_WAIT` stage (s6) = 185 ms/token in Q0 baseline → this is the
  maximum potential reduction from moving shared expert into the read window.
- Shared expert CB observed cost ≈ 8 µs/layer × 43 = 344 µs/token
  (from ATTENTION_ENCODE + MOE_ENCODE stages, where MOE_ENCODE = 335 µs includes
  shared expert encode setup).
- If shared expert CB (344 µs) hides within the pread window (measured
  cold-miss latency avg ~185 ms) → overlap potential ≈ 344 µs, well within
  the window. The reduction in IO_EXPOSED_WAIT is bounded by how much of the
  shared CB overlaps with the slowest cold read.
- R4.4 gate: IO_EXPOSED_WAIT must fall ≥15%. The shared expert CB is 344 µs
  vs 185 ms average exposed wait → 0.19% of the gap. Hiding it reduces
  exposed wait by at most 0.19%, far below 15%.
- **Stop gate assessment**: The partition is geometrically correct (shared CB
  is independent of cold reads, can safely run in the window), but the
  measurable reduction in exposed wait from shared CB alone is ~0.19% — well
  below the R4.4 15% gate. Task 2.2 may proceed to implementation, but the
  promotion gate will require either (a) the join itself returning earlier
  because pread completes during shared CB execution, or (b) additional
  independent work being identified. The oracle confirms the current ordering
  is the only correctness barrier; geometry allows overlap but scale of benefit
  is uncertain and subject to R4.4 measurement. This is a valid `TRIGGERED`
  assessment — the partition is safe and tests confirm ordering invariants.
