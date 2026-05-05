# DeepSeek V4 Optimization and Roadmap

> **Consolidated from:** optimization-plan.md, turboquant-analysis.md  
> **Last updated:** 2026-04-27 (optimization plan v3), 2026-04-26 (turboquant analysis)  
> **Cross-references:** [Fixes and Details](DEEPSEEK-V4-FIXES-AND-DETAILS.md) | [Chat Analysis & Troubleshooting](DEEPSEEK-V4-CHAT-ANALYSIS-AND-TROUBLESHOOTING.md)

---

## Executive Summary

Based on deep audit of Apple's official `mlx-lm/mlx_lm/models/deepseek_v4.py` (2153 lines) and mlx-zig's current capabilities (including `CustomMetalKernel` support), this document provides a 5-week executable optimization roadmap. Three key architectural differences between mlx-zig and mlx-lm drive the optimization strategy: (1) MoE weight format (fused vs split), (2) KV compression subsystem (stateful vs pure function), and (3) Indexer design (Apple redesign vs paper-original).

Additionally, TurboQuant analysis identifies concrete steps for upgrading FP8 KV storage (now possible with new `mlx-c` 0.6.0 bindings) and a path toward optimal KV cache quantization.

---

## Part 1: Optimization Plan v3 (from optimization-plan.md)

**Revision v3** — Audit date: 2026-04-27  
v2 corrected 8 issues from v1; v3 corrected 3 factual errors (gatherMm already bound, RoPE already unified, HCA Indexer condition fixed), added 2 missing tasks (RoPE GPU-ization, O-LoRA batch matmul), removed 3 completed tasks, timeline compressed from 6 to 5 weeks.

### 1. Background

`mlx-lm` released official `deepseek_v4.py` (2026-04), the authoritative DeepSeek-V4 inference reference covering:
- **CSA/HCA hybrid attention**: Complete `Compressor` + `Indexer` + `_sparse_pooled_attention`
- **mHC custom Metal kernel**: `hc_split_sinkhorn` and `hc_sinkhorn_collapse` two inline shaders
- **Efficient MoE dispatch**: `DeepseekV4SwitchGLU` using `gather_mm` + `sort_threshold=8`
- **Sophisticated cache design**: `DeepseekV4Cache` managing `local` + `compressor_state` + `indexer_state`

### 2. Key Architectural Differences

#### 2.1 MoE Weight Format: Fused vs Split

| Dimension | mlx-lm | mlx-zig |
|-----------|--------|---------|
| **Storage** | `switch_mlp.{gate,up,down}_proj.weight` as `[n_experts, out, in]` (fused) | Loader splits fused tensor into 256 independent experts (`ffn.experts.{e}.{w1,w2,w3}`) |
| **Dispatch** | `mx.gather_mm` / `mx.gather_qmm` loads selected experts at once | Iterates 256 `DSV4Expert`, mask shorts experts with no tokens |
| **Quantized state** | `QuantizedSwitchLinear` keeps MXFP4/affine packed, dequant inside kernel | Loader `mlx_dequantize` dequantizes to float16/float32 |

**Conclusion:** mlx-zig MoE is `fused → split → dequantize → per-expert loop` path, opposite of mlx-lm's `fused → gather_mm`. Optimization requires **reverse refactoring**.

#### 2.2 KV Compression Subsystem: Module State vs Pure Function

| Dimension | mlx-lm | mlx-zig |
|-----------|--------|---------|
| **Structure** | `Compressor` is stateful `nn.Module` with `wkv`, `wgate`, `ape`, `norm` | `compressKV()` is pure function, weights from caller |
| **Cache integration** | `Compressor.__call__` internally calls `cache.accumulate_windows` | `compressKV()` completely unaware of cache |
| **Overlap** | `_overlap_transform` is member method, accesses `self.head_dim`, `self.overlap` | No overlap logic |

**Conclusion:** mlx-zig lacks `Compressor` module and `DeepseekV4Cache` state machine. Not a simple `compressKV` function fix.

#### 2.3 Indexer: Two Different Architectures

| Dimension | mlx-lm `Indexer` | mlx-zig `LightningIndexer` |
|-----------|-------------------|---------------------------|
| **Q projection** | `wq_b: Linear(q_lora_rank → n_heads * head_dim)` | `wq_index: [index_n_heads * index_head_dim, head_dim]` |
| **K source** | **Internally nests `Compressor`** generating pooled | Directly receives external `k_compressed` |
| **Scoring** | `q @ pooled^T`, `max(0, score)`, weighted by `weights_proj` | `q_index @ k_index^T`, scaled, mean over heads |
| **FP4** | ❌ All bfloat16/float32 | ⚠️ Has `quantize4bit` simulation (should be removed) |

**Conclusion:** Two Indexers have completely different weight shapes, input sources, and scoring logic. mlx-zig's `LightningIndexer` is closer to DeepSeek V3 paper design, while mlx-lm's is Apple's redesign.

### 3. Priority Overview

| Priority | Tasks | Core Theme |
|----------|-------|------------|
| **P0** | 3 | Correctness fixes + max performance bottleneck (incl. Attention dual-branch, elevated from P1) |
| **P1** | 5 | Architecture alignment + engineering simplification + Custom Kernel + new RoPE GPU + O-LoRA batch matmul |
| **P2** | 2 | Quantization preservation + weight loading |
| **P3** | 1 | Detail polish |

**v3 changes:** P2.3 (`mlx_gather_mm` binding) removed (`ops.zig:318` already has full binding); P1.6 (RoPE inverse simplify) removed (`apply()` already has `inverse: bool`); P0.3 (HCA remove Indexer) removed (loader condition already `compress_ratio == 4`); P1.2 (Attention dual-branch) elevated to P0; Added P1.7 (RoPE GPU-ize) and P1.8 (O-LoRA batch matmul).

### 4. P0: Correctness & Core Performance Bottlenecks

> **Recommended timeline: Week 1-2**

#### 4.1 MoE Dispatch Reverse Refactoring: Gather/Sort + `gather_mm`

| Attribute | Content |
|-----------|---------|
| **Problem** | `DSV4MoE.forward()` iterates all 256 experts (mask shorts experts with no tokens, but still O(n_experts)). More fundamentally: loader already split fused `switch_mlp` into independent experts, losing gather capability |
| **mlx-lm reference** | `DeepseekV4SwitchGLU` (`deepseek_v4.py:125-155`), inherits `SwitchGLU` (`switch_layers.py:160-199`) |
| **Key mechanism** | 1. `_gather_sort`: sort tokens by expert ID (`sort_threshold=8`, covers nearly all V4 cases)<br>2. `mx.gather_mm` / `mx.gather_qmm`: only load selected expert weights (fused `[n_experts, out, in]`)<br>3. `scores` multiplied **before** `down_proj`, reduces one reduce<br>4. `_scatter_unsort` restores original token order |
| **Prerequisite** | ~~`mlx_gather_mm` Zig binding missing~~ — **v3 correction: `ops.zig:318` already has complete `pub fn gatherMm` binding** |
| **mlx-zig actions** | 1. ~~Bind `mlx_gather_mm`~~ (done)<br>2. **Stop splitFusedExperts**: loader keeps `switch_mlp.weight` as `[n_experts, out, in]` fused<br>3. **Refactor DSV4MoE**: remove `DSV4Expert` array, use fused weights + `gatherMm`/`gatherQmm`<br>4. **Integrate sort**: `argsort` tokens by expert ID after routing, call gather with `sorted_indices=true`<br>5. **Pre-fuse scores**: multiply before down-proj |
| **Expected benefit** | **10-100x** expert computation speedup, eliminate redundant kernel launches, reduce VRAM fragmentation |
| **Related files** | `src/models/deepseek_v4.zig:638-777`, `src/models/deepseek_v4_loader.zig:495-536`, `src/quantize.zig:323-358`, `src/ops.zig` |

#### 4.2 KV Compression Subsystem Refactoring: `Compressor` + `DeepseekV4Cache`

| Attribute | Content |
|-----------|---------|
| **Problem** | `compressKV` is pure function, no state management, two problems:<br>1. **No `_overlap_transform`**: CSA (ratio=4) layers lose block overlap, compressed blocks only aggregate `m` tokens not `2m`<br>2. **No `accumulate_windows`**: tail tokens directly discarded (`num_groups = prefix_len / compress_ratio`, remainder not buffered), generate phase L=1 can't accumulate |
| **mlx-lm reference** | `Compressor` module (`deepseek_v4.py:1481-1551`), `DeepseekV4Cache.accumulate_windows` (`deepseek_v4.py:1028-1126`) |
| **Key mechanism** | 1. `Compressor` as stateful module, internally calls `cache.accumulate_windows(kv, gate, state_key, ratio, start_pos)`<br>2. `_overlap_transform`: CSA concats `[prev_first, second_half]`, gate fills with `-inf`<br>3. `accumulate_windows`: maintains `buffer_kv`/`buffer_gate`, tail < `compress_ratio` preserved for next forward; supports variable-length batch (`lengths` + `right_padding`) |
| **mlx-zig actions** | 1. **Introduce `Compressor` struct**: encapsulate `wkv`, `wgate`, `ape`, `norm`, `overlap` flag<br>2. **Introduce `DeepseekV4Cache`**: extend or add cache type, manage `local` + `compressor_state` + `indexer_state`<br>3. **Implement `_overlap_transform`**: add overlap branch in `Compressor` (only `compress_ratio == 4`)<br>4. **Implement `accumulate_windows`**: maintain buffer state in `DeepseekV4Cache` |
| **Expected benefit** | **CSA layer correctness fix** (otherwise attention loses half context), support arbitrary length input and correct generate phase accumulation |
| **Related files** | `src/models/deepseek_v4.zig:1367-1441` (compressKV), `src/kvcache.zig`, `src/models/deepseek_v4.zig:1117-1234` (Attention forward) |

#### ~~4.3 HCA Layer (ratio=128) Remove Lightning Indexer~~ ✅ Complete (v3)

Already fixed. `deepseek_v4_loader.zig:935` condition is `compress_ratio == 4`, only CSA layers create Indexer. No action needed.

#### 4.4 Attention Dual-Branch: Local / Pooled (L==1 vs L>1)

| Attribute | Content |
|-----------|---------|
| **Problem** | mlx-zig uniformly uses `gatherBlocks → concat → dense SDPA` for all CSA layers, doesn't distinguish generate (L=1) and prefill (L>1). **Correctness issue**: generate phase L=1, compressed blocks gather semantics differ (applies last prefill's index result) |
| **mlx-lm reference** | `V4Attention.__call__` (`deepseek_v4.py:1769-1794`) |
| **Key mechanism** | - **L==1** (generate): `take_along_axis` gather selected pooled blocks, concat, standard `scaled_dot_product_attention`<br>- **L>1** (prefill): use `_sparse_pooled_attention` (`deepseek_v4.py:295-333`) |
| **mlx-zig actions** | 1. Add `if (seq_len == 1)` gather + dense SDPA branch in `DSV4Attention.forward`<br>2. Implement `seq_len > 1` `_sparse_pooled_attention` equivalent path |
| **Expected benefit** | Decode phase correctness fix + latency reduction, prefill phase pooled computation reduction |
| **Related files** | `src/models/deepseek_v4.zig:1113-1361` |

### 5. P1: Architecture Alignment & Engineering Simplification

> **Recommended timeline: Week 3-4**

#### 5.1 Remove Indexer FP4 Simulation + Clarify Indexer Architecture

| Attribute | Content |
|-----------|---------|
| **Problem** | `LightningIndexer.quantize4bit()` (`deepseek_v4.zig:905-986`) does `quantize(mxfp4) → dequantize → float32 matmul`, no speed or precision benefit. More importantly: mlx-zig `LightningIndexer` vs mlx-lm `Indexer` architectures are **completely different** (§2.3) |
| **mlx-zig actions** | **Plan A (recommended)**: Align with mlx-lm Indexer — delete `LightningIndexer`, create `Indexer` struct (nest `Compressor`, `wq_b` projection, `weights_proj`); remove `quantize4bit`<br>**Plan B**: Keep paper-original Lightning Indexer, just remove `quantize4bit`, use bfloat16 matmul |
| **Expected benefit** | Code simplification, precision improvement, reference implementation alignment |
| **Related files** | `src/models/deepseek_v4.zig:794-1056` |

#### 5.2 ~~Attention Dual-Branch~~ → Elevated to P0.4 (v3)

#### 5.3 mHC Integration with `CustomMetalKernel`

| Attribute | Content |
|-----------|---------|
| **Problem** | `DSV4HyperConn.pre` (`deepseek_v4.zig:1876-1893`) uses pure Zig ops for Sinkhorn iteration (`sinkhornNormalize`, line 1713-1757), multiple kernel launches, no SIMD optimization |
| **mlx-lm reference** | `_make_hc_sinkhorn_collapse_kernel` (`deepseek_v4.py:486-633`) and `_make_hc_split_sinkhorn_kernel` (`deepseek_v4.py:365-448`) |
| **Key mechanism** | Single Metal dispatch, `simd_sum` column normalization, `bfloat4` vectorized load, FMA chain, branchless lanes |
| **Prerequisite** | Metal 3.1+ (supports `bfloat16_t` and `vec<bfloat16_t, 4>`) |
| **mlx-zig actions** | 1. Translate mlx-lm Metal source strings to Zig strings<br>2. Register `CustomMetalKernel` via `src/ops/custom_kernel.zig`<br>3. Trigger: `!training && hc_mult==4 && dtype==bfloat16 && gpu`<br>4. Static init kernel at module load (like mlx-lm's `_hc_sinkhorn_collapse_kernel = _make_hc_sinkhorn_collapse_kernel()`)<br>5. Fallback to existing ops path |
| **Expected benefit** | mHC forward pass reduces 80%+ kernel launch overhead, 43-layer cumulative effect significant |
| **Related files** | `src/models/deepseek_v4.zig:1861-1899`, `src/ops/custom_kernel.zig` |

#### 5.4 Compressor CSA vs HCA Strategy Differentiation

| Attribute | Content |
|-----------|---------|
| **Problem** | `compressKV` uses uniform gated pooling for all `compress_ratio > 1`, doesn't distinguish CSA (4x) and HCA (128x) |
| **mlx-lm reference** | `Compressor.__init__` (`deepseek_v4.py:1483-1493`): `self.overlap = compress_ratio == 4`, `self.out_dim = head_dim * (2 if overlap else 1)` |
| **Key mechanism** | CSA (4x): gated pool + overlap + Indexer, output `head_dim * 2`<br>HCA (128x): gated pool, no overlap, no Indexer, output `head_dim` |
| **mlx-zig actions** | Select strategy by `compress_ratio` in `Compressor`: 128x layers simplified (no overlap, out_dim = head_dim) |
| **Expected benefit** | HCA layers avoid wasted computation, strict paper architecture alignment |
| **Related files** | `src/models/deepseek_v4.zig` (refactored Compressor) |

#### 5.5 Sink Handling in Generate/Prefill Attention Branches

| Attribute | Content |
|-----------|---------|
| **Problem** | mlx-zig `sink_logits` **already aligned** with mlx-lm dense SDPA path (both pass to `fast_scaled_dot_product_attention` `sinks` param). Real difference in `_sparse_pooled_attention` path: mlx-lm manually concats sink to scores front (`deepseek_v4.py:319-324`), mlx-zig lacks sparse_pooled path |
| **mlx-zig actions** | When implementing `_sparse_pooled_attention` equivalent path, explicitly concat sink as prefix to scores (`scores = concat([sink_scores, local_scores, pooled_scores])`) |
| **Expected benefit** | sparse_pooled path behavior consistent with mlx-lm |
| **Related files** | `src/models/deepseek_v4.zig:1089-1091`, `src/ops/fast.zig:39-47` |

#### 5.6 RoPE Inverse Simplification

| Attribute | Content |
|-----------|---------|
| **Problem** | mlx-zig `DSV4YarnRoPE` may have separate `apply` and `applyInverse` methods, code duplication |
| **mlx-lm reference** | `_apply_partial_rope(..., inverse=True)` (`deepseek_v4.py:266-284`) reuses same function |
| **mlx-zig actions** | Merge `applyInverse` into `apply`, add `inverse: bool = false` param |
| **Expected benefit** | Reduce code duplication, lower maintenance cost |
| **Related files** | `src/models/deepseek_v4.zig` (RoPE related) |

### 6. P2: Quantization, Loading & Binding Gap Fill

> **Recommended timeline: Week 5**

#### 6.1 MoE Expert Weights Keep Quantized State

| Attribute | Content |
|-----------|---------|
| **Problem** | `loader.zig` (line 539-653) dequantizes all quantized weights (incl. experts) via `mlx_dequantize` to float16/float32. Expert weights are full precision at inference |
| **mlx-lm reference** | `QuantizedSwitchLinear` (`switch_layers.py:27-90`): weights stay MXFP4/affine packed, dequant inside kernel via `mx.gather_qmm` |
| **mlx-zig actions** | 1. Keep packed weight + scales at load (don't dequantize)<br>2. Use `gatherQmm` for expert matmul (`quantize.zig:324-358`)<br>3. Note: requires P0.1 fused format refactoring first (no split) |
| **Expected benefit** | MoE expert weight memory to 1/4-1/8, relieve VRAM pressure |
| **Related files** | `src/models/deepseek_v4_loader.zig:539-653`, `src/quantize.zig:324-358` |

#### 6.2 O-LoRA Grouped Projection Quantized Path

| Attribute | Content |
|-----------|---------|
| **Problem** | mlx-zig O-LoRA manually implements grouped matmul (`deepseek_v4.zig:1305-1358`), only float32/float16 path, **no quantized matmul** |
| **mlx-lm reference** | `V4Attention._grouped_output_projection` (`deepseek_v4.py:1680-1709`): when `wo_a` is `QuantizedLinear`, uses `mx.quantized_matmul` |
| **mlx-zig actions** | In grouped projection, detect if `wo_a` is quantized, if so call `quantizedMatmul` (`quantize.zig:259-285`) |
| **Expected benefit** | Quantized model end-to-end inference performance improvement |
| **Related files** | `src/models/deepseek_v4.zig:1305-1358` |

#### 6.3 ~~Bind `mlx_gather_mm`~~ ✅ Complete (v3)

`ops.zig:318` already has complete `pub fn gatherMm` binding. No action needed.

### 7. P3: Detail Polish

> **Recommended timeline: Week 6**

#### 7.1 Weight Loading Naming Mapping & Format Alignment

| Attribute | Content |
|-----------|---------|
| **Problem** | mlx-lm `sanitize` (`deepseek_v4.py:1992-2127`) handles FP4/FP8 custom dequant (`dequant_fp4`, `dequant_fp8`), naming remap (`w1→gate_proj`), and **independent expert → fused `switch_mlp` stack**. mlx-zig direction is opposite (fused → split) |
| **mlx-zig actions** | 1. Confirm mlx-lm quant format (packed uint8 + scales specific layout) compatible with `mlx_dequantize`<br>2. With P0.1 refactoring, loader ultimately **stops split**, keeps fused (same direction as mlx-lm) |
| **Expected benefit** | 1:1 weight format compatibility with mlx-lm, reduce conversion overhead |
| **Related files** | `src/models/deepseek_v4_loader.zig:471-661` |

### 8. Timeline

```text
Week 1-2 (P0) — Correctness & Core Performance
├── Task 1: Bind mlx_gather_mm (P2.3 prerequisite) ✅ Complete
├── Task 2: MoE reverse refactoring (fused format + gather_mm dispatch)
├── Task 3: KV compression subsystem refactoring (Compressor + DeepseekV4Cache + overlap + accumulate_windows)
└── Task 4: HCA remove Indexer ✅ Complete

Week 3-4 (P1) — Architecture Alignment & Custom Kernel
├── Task 5: Indexer architecture choice + remove FP4 simulation
├── Task 6: Attention dual-branch (L==1 gather + L>1 sparse_pooled)
├── Task 7: mHC CustomMetalKernel (with Metal bfloat16 prerequisite check)
├── Task 8: CSA/HCA strategy differentiation
├── Task 9: Sparse_pooled Sink concat
└── Task 10: RoPE inverse simplification

Week 5 (P2) — Quantization & Binding Gap Fill
├── Task 11: MoE experts keep quantized state (gather_qmm)
├── Task 12: O-LoRA quantized path
└── Task 13: Weight loading fused format alignment

Week 6 (P3) — Detail Polish
└── Task 14: Naming mapping & format compatibility verification
```

### 9. Key Reference File Index

| File | Path | Description |
|------|------|-------------|
| `deepseek_v4.py` | `mlx-lm/mlx_lm/models/deepseek_v4.py` | Official complete implementation (2153 lines) |
| `switch_layers.py` | `mlx-lm/mlx_lm/models/switch_layers.py` | MoE SwitchGLU / gather_mm / gather_qmm |
| `base.py` | `mlx-lm/mlx_lm/models/base.py` | `scaled_dot_product_attention` (sinks param) |
| `custom_kernel.zig` | `mlx-zig/src/ops/custom_kernel.zig` | mlx-zig CustomMetalKernel (full API) |
| `deepseek_v4.zig` | `mlx-zig/src/models/deepseek_v4.zig` | mlx-zig V4 implementation |
| `deepseek_v4_loader.zig` | `mlx-zig/src/models/deepseek_v4_loader.zig` | Weight loading |
| `moe_router.zig` | `mlx-zig/src/moe_router.zig` | MoE routing module (not integrated into DSV4MoE) |
| `quantize.zig` | `mlx-zig/src/quantize.zig` | Quantization / gather_qmm |
| `ops.h` | `mlx-c/mlx/c/ops.h` | `mlx_gather_mm` / `mlx_gather_qmm` C API |

### 10. Appendix: mlx-lm vs mlx-zig Capability Matrix

| Feature | mlx-lm (Python) | mlx-zig | Gap | Notes |
|---------|-----------------|---------|-----|-------|
| MoE gather_mm dispatch | ✅ `SwitchLinear` + `gather_mm` | ❌ split + per-expert loop | **High** | Need reverse refactoring |
| CSA overlap transform | ✅ `Compressor._overlap_transform` | ❌ pure fn compressKV | **High** | Need Compressor module |
| Cache accumulate_windows | ✅ `DeepseekV4Cache` | ❌ no buffer state | **High** | Need custom cache type |
| HCA no Indexer | ✅ `compress_ratio==4` only | ❌ `compress_ratio>1` | Mid | One-line condition change |
| Indexer architecture | ✅ Apple redesign (wq_b + weights_proj) | ⚠️ Paper design (wq/wk + dot) | Mid | Architecture choice |
| FP4 Indexer | ❌ Skipped | ⚠️ Simulate (should remove) | Low | Just delete |
| Local/pooled separate attention | ✅ `_sparse_pooled_attention` + L==1 branch | ❌ concat dense | **High** | Need dual-branch |
| mHC Metal kernel | ✅ 2 fused kernels | ❌ (has CustomMetalKernel, not integrated) | Mid | Need shader port |
| O-LoRA quantization | ✅ `quantized_matmul` fallback | ❌ float matmul only | Mid | Need detect + quantized path |
| Expert weights quantized | ✅ `QuantizedSwitchLinear` | ❌ Full precision | Mid | Keep packed + gather_qmm |
| CustomMetalKernel | ✅ `mx.fast.metal_kernel` | ✅ `mlx_fast_metal_kernel` | None | Capability complete |
| `mlx_gather_mm` binding | N/A (Python direct call) | ✅ Bound | None | `ops.zig:318` |
| Attention Sink | ✅ `sinks` param / manual concat | ✅ `sinks` param | None | Dense path aligned |
| RoPE inverse | ✅ `inverse=True` param | ⚠️ Possible separate method | Low | Merge |

### 11. Audit Changelog (v1 → v2)

| Change | v1 Issue | v2 Correction |
|--------|----------|---------------|
| P0 3.1 | Incorrectly described "256 unconditional launches"; didn't mention gather_mm binding | Corrected to mask short-circuit; added binding prerequisite; noted loader needs reverse refactoring |
| P0 3.2 | Only mentioned "compressKV add overlap" | Noted missing Compressor module and DeepseekV4Cache; needs subsystem refactoring |
| P1 4.1 | Only mentioned "remove FP4 simulation" | Added Indexer architecture difference; provided Plan A/B |
| P1 4.2 | Only mentioned "separate local/pooled" | Added L==1 branch; distinguished generate vs prefill |
| P1 4.3 | Only mentioned "accumulate_windows" | Merged into P0 3.2 (Compressor + Cache refactoring) |
| P3 6.1 | Misleading "sink_logits needs refactoring" | Noted dense path aligned; corrected to sparse_pooled sink concat |
| P2 5.2 | Suggested "sanitize align + stack" | Noted mlx-zig direction opposite; changed to "stop split, keep fused" |
| New | — | §2 Key architectural differences (fused vs split, Compressor state, Indexer designs) |
| New | — | P2.2 O-LoRA quantized path |
| New | — | P2.3 Bind `mlx_gather_mm` |

---

## Part 2: TurboQuant & FP8 Analysis (from turboquant-analysis.md)

> Based on DeepSeek V4 technical report (2026-04-24) and TurboQuant paper (arXiv:2504.19874), combined with mlx-zig code audit (2026-04-26) and newly completed MXFP4/FP8 bindings.

### 1. Current Implementation Status Audit

#### Completed (code exists and functional)

| Task | Implementation Location | Status |
|------|------------------------|--------|
| 15.2 learned softmax-gated pooling | `compressKV()` → `softmaxGatedPool()` at L1231-1430 | ✅ Complete, with remainder handling |
| 15.4 FP4 Lightning Indexer | `LightningIndexer` struct at L648-930 | ✅ Complete, with INT4 quantization simulation, top-k selection, block gather |
| 15.5 Attention Sink | `sink_logits` field at L959, passed to `scaledDotProductAttention` at L1144 | ✅ Integrated into fast SDPA |
| 15.3 FP8 KV storage | `kv_storage_dtype` + `astype` at L1055-1080 | ⚠️ Using float16 as FP8 proxy (comment notes MLX lacks native FP8) |
| 15.6 heterogeneous KV cache | `compress_ratios` from config, per-layer `compress_ratio` at L950 | ⚠️ Compression ratios per-layer, but KV cache strategy still uniform |

#### Key Findings

1. **Task 15.2/15.4/15.5 marked `[x]` are accurate** — code is indeed implemented
2. **Task 15.3 FP8 now solvable** — mlx-c 0.6.0 has `mlx_to_fp8`/`mlx_from_fp8`, Zig bindings added in `quantize.zig` and `ops.zig`
3. **Task 15.6 partially complete** — `compress_ratios` per-layer config, but KV cache strategy allocation not per-layer
4. **Task 15.1 (Paged + Quantized) is the only unstarted** — marked `[ ]`

### 2. DeepSeek V4 Paper: Precise Guidance for Remaining Work

#### 2.1 Task 15.3 — FP8 KV Storage: Now Can Use True FP8

**Paper requirement:** V4 stores most KV dimensions as FP8 (E4M3), only RoPE dimensions keep BF16.

**Current implementation:** `deepseek_v4.zig:1055-1080` uses `astype(float16)` as FP8 proxy, comment says "MLX lacks native FP8".

**Current situation:** mlx-c 0.6.0 now has `mlx_to_fp8`/`mlx_from_fp8`, we just added `toFp8()`/`fromFp8()` bindings in `ops.zig`. Can directly replace.

**Specific changes:**
```zig
// Current (L1059-1063):
const kv_nope_stored = if (kv_storage_dtype != .float32)
    try ops.astype(self.ctx, kv_nope, kv_storage_dtype)  // float16 proxy
else kv_nope;

// Changed to:
const kv_nope_stored = try ops.toFp8(self.ctx, kv_nope);  // True FP8
// On read:
const kv_nope_restored = try ops.fromFp8(self.ctx, kv_nope_stored, .bfloat16);
```

**Note:** FP8 is not a `mlx_dtype` enum member, cannot use `astype`. Must use dedicated `toFp8`/`fromFp8`. RoPE dimensions continue with `astype(.bfloat16)`.

**Priority:** 🔴 High — Small change (~10 lines), big effect (memory halved vs float16).

#### 2.2 Task 15.6 — Heterogeneous KV Cache: Missing Per-Layer Strategy

**Paper requirement:** V4-Pro 61-layer distribution:
- Layer 0-1: HCA (128x compression)
- Layer 2-60: CSA (4x) and HCA (128x) alternating
- Per-layer KV cache shapes differ (compressed sequence lengths differ)

**Current implementation:**
- ✅ `compress_ratios` read from config.json, passed per-layer to `DSV4Attention`
- ✅ `compressKV()` does different compression based on `compress_ratio`
- ❌ `server.zig` and `loadModel` use same `KVCacheStrategy` for all layers
- ❌ No cache buffer size allocation based on `compress_ratio`

**Needed changes:** In `loadModel`, allocate different cache per layer based on `compress_ratios[i]`:
- CSA layers (ratio=4): cache sequence dim = max_seq_len / 4 + window_size
- HCA layers (ratio=128): cache sequence dim = max_seq_len / 128 + window_size
- No compression layers (ratio=0 or 1): cache sequence dim = max_seq_len

**Priority:** 🟡 Medium — Current impl works (cache oversized but no errors), optimization saves significant memory.

#### 2.3 Task 15.1 — Paged + Quantized Combination

**Paper reference:** V4 itself doesn't need this (built-in compression sufficient), but critical for general models (LLaMA/Mistral/Qwen) in long context.

**Current state:** `PagedKVCache` and `QuantizedKVCache` are sibling `KVCacheStrategy`, cannot be used simultaneously.

**TurboQuant paper insight:** 3.5-bit quantization is lossless, so Paged + 4-bit combination achieves dual benefits of paged memory management (reduces fragmentation) and quantization compression (reduces total size).

**Priority:** 🟡 Medium — Valuable for general model long context scenarios.

### 3. TurboQuant Paper: Guidance for General Model Quantization

#### 3.1 Comparison with Current `kvcache/quantized.zig`

Current implementation uses MLX built-in `mlx_quantize` (affine mode), uniform quantization. TurboQuant provides theoretically optimal alternative:

| Dimension | Current affine | TurboQuant | MXFP4 (newly bound) |
|-----------|---------------|------------|---------------------|
| Quantization method | Uniform per-group | Lloyd-Max optimal | E2M1 per-block |
| Inner product bias | Biased | Unbiased (+QJL) | Biased |
| Theoretical guarantee | None | ≈2.7x optimal | None |
| Implementation complexity | Low (existing) | Medium | Low (already bound) |
| Hardware acceleration | Metal native | Needs self-implementation | Metal native |
| Applicable scenario | Weight quantization | KV cache | Weight quantization |

#### 3.2 MXFP4 vs TurboQuant: Optimal Choice by Scenario

**Weight quantization:** MXFP4 more suitable
- Metal kernel acceleration (`mlx_quantized_matmul` with mode="mxfp4")
- Training-compatible (QAT)
- Good ecosystem support (HuggingFace, mlx-lm integrated)

**KV cache quantization:** TurboQuant more suitable
- Online quantization (no calibration data needed, KV cache dynamically generated)
- Unbiased inner products (attention score precision higher)
- 3.5-bit lossless (more memory efficient and better quality than 4-bit affine)

**Recommended path:**
1. Current: Use existing affine 4-bit for KV cache quantization (already implemented)
2. Near-term: Use MXFP4 for weight quantization (newly bound, directly usable)
3. Long-term: Implement TurboQuant for KV cache quantization (Phase 4+)

#### 3.3 TurboQuant Zig Implementation Plan

Core algorithm needs only 4 steps, all implementable with mlx-c operators:

```
1. Random rotation:  y = Π · x          → mlx_matmul (Π pre-generated, QR decomposition)
2. Scalar quantization: idx = nearest(y, codebook) → mlx_argmin + precomputed codebook
3. Dequantization:  ỹ = codebook[idx]   → mlx_take
4. Inverse rotation: x̃ = Π^T · ỹ       → mlx_matmul
```

QJL residual correction (unbiased inner products):
```
5. Residual: r = x - x̃          → mlx_subtract
6. QJL: z = sign(S · r)         → mlx_matmul + mlx_sign (S pre-generated)
7. Reconstruction: x̂ = x̃ + √(π/2)/d · ‖r‖ · S^T · z  → standard operator chain
```

**Storage overhead:**
- Π matrix: d×d float32, d=128 → 64KB (per-model, one-time)
- S matrix: d×d float32, same
- Codebook: 2^b float32, 4-bit → just 64 bytes
- Quantized data: b bits per coordinate (vs 16 bits float16)

### 4. Updated Priority Ranking

#### Immediately Doable (small change, big benefit)

1. **Task 15.3 Upgrade to true FP8** — Replace `astype(float16)` with newly bound `toFp8()`/`fromFp8()`
   - Change: ~10 lines in `deepseek_v4.zig`
   - Benefit: KV cache memory halved (vs float16)

#### Near-term (requires some work)

2. **Task 15.6 per-layer cache sizing** — Allocate different cache buffer sizes based on `compress_ratio`
   - Change: Cache allocation logic in `loadModel`
   - Benefit: More precise V4 memory usage

3. **Task 15.1 Paged + Quantized** — Built-in quantization option in `PagedKVCache`
   - Change: Add `kv_bits` param to `kvcache/paged.zig`
   - Benefit: General model long context memory optimization

#### Long-term (Phase 4+)

4. **MXFP4 weight quantization integration** — Support MXFP4 model loading with newly bound `quantize(mode="mxfp4")`
5. **TurboQuant KV cache quantization** — Implement paper algorithm, replace affine quantization

### 5. Suggested tasks.md Revisions

Task 15.3 description update:
```markdown
- [ ] 15.3 DeepSeek V4: upgrade FP8 KV storage to use native mlx_to_fp8/mlx_from_fp8
    - Current code at deepseek_v4.zig:1055-1080 uses astype(float16) as FP8 proxy
    - mlx-c 0.6.0 now has mlx_to_fp8/mlx_from_fp8, Zig bindings added in ops.zig
    - Replace astype(kv_storage_dtype) with ops.toFp8() for non-RoPE KV dimensions
    - Keep astype(.bfloat16) for RoPE dimensions
    - Add fromFp8() call before attention computation to restore precision
    - Remove kv_storage_dtype config field (no longer needed, FP8 is the V4 default)
```

New Task 15.8:
```markdown
- [ ]* 15.8 TurboQuant KV cache quantization (optional, Phase 4+)
    - Implement Lloyd-Max codebook precomputation for b=1,2,3,4
    - Implement random rotation (QR decomposition via mlx linalg)
    - Implement scalar quantize/dequantize with precomputed codebook
    - Implement QJL residual correction for unbiased inner products
    - Add --kv-quant simple|turbo CLI option (default: simple)
    - _Reference: TurboQuant paper arXiv:2504.19874_
```

---

## References

- [DeepSeek V4 Paper](https://arxiv.org/abs/2501.12948)
- [TurboQuant Paper](https://arxiv.org/abs/2504.19874)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [mlx-lm Repository](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [Fixes and Details](DEEPSEEK-V4-FIXES-AND-DETAILS.md)
- [Chat Analysis & Troubleshooting](DEEPSEEK-V4-CHAT-ANALYSIS-AND-TROUBLESHOOTING.md)
