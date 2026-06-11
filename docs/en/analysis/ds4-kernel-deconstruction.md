# ds4 Metal Kernel Deconstruction
date: 2026-06-11
status: complete analysis

---

## 1. 文件结构总览

| 文件 | 功能 |
|------|------|
| `dsv4_hc.metal` | **mHC (HyperConnection) kernels** — 这是 dmlx 最缺的部分 |
| `dsv4_misc.metal` | router selection, indexed mixed attention, indexer scores |
| `dsv4_kv.metal` | KV cache FP8 quantize/store, compressor |
| `dsv4_rope.metal` | Partial RoPE (non-RoPE prefix copy + RoPE tail) |
| `norm.metal` | RMSNorm + fused Q/KV norm (`kernel_dsv4_qkv_rms_norm_f32_4`) |
| `dense.metal` | General matvec (Q8_0) + fused shared expert gate/up/SwiGLU |
| `moe.metal` | Routed MoE SwiGLU+weight kernel, sum6, expert matmul (Q2_K/Q4_K) |
| `flash_attn.metal` | FlashAttention decode/prefill |

---

## 2. 关键 Kernel 逐一分析

### 2.1 mHC (dsv4_hc.metal) — **最关键**

这是 dmlx native engine 性能差距的根本原因之一。

**`kernel_dsv4_hc_split_weighted_sum`**:
- 功能: HC pre-split (Sinkhorn) + weighted sum，合并为一次 dispatch
- 在 decode 时：**1 TG = 1 token row，tid=0 做 Sinkhorn，所有 thread 做 weighted sum**
- 不需要多次 CB，`pre[]` 存在 shared memory，所有 thread 复用
- dmlx 当前：CB-A 做 mhc_pre (GPU)，wait，CPU 读 post/comb，再 CMD2。**2 个 wait**
- ds4 做法：1 次 dispatch 搞定，**0 extra wait**

**`kernel_dsv4_hc_split_weighted_sum_norm4`**:
- 功能: HC split + weighted sum + **fused RMSNorm**，全部在一个 dispatch
- 1024 threads，处理 4096-wide row
- 直接输出 normed attention input，省掉单独的 RMSNorm dispatch

**`kernel_dsv4_shared_down_hc_expand4_q8_0`**:
- 功能: shared expert down + HC expand，**完全融合**
- Q8_0 matvec + route 结果合并 + HC expand → 一次 dispatch
- dmlx：shared expert CB(wait) + mhc_post_ffn CB(wait) = **2 extra waits**

**`kernel_dsv4_hc_expand4`**:
- 功能: 单独的 HC expand（post gate × block + comb × residual）
- 每个 thread 处理 1 (d, t)，计算 4 个 HC stream，没有 shared memory
- 非常高效，无 occupancy 问题

**`kernel_dsv4_q8_hc_expand4_q8_0`**:
- 功能: attention output (wo_b 等价物) + HC expand 融合
- **attention output matmul + HC post 合为一次 dispatch**

---

### 2.2 QKV RMSNorm (norm.metal)

**`kernel_dsv4_qkv_rms_norm_f32_4`**:
- 同时 normalize Q-lora 和 KV，**一次 dispatch 处理两个 norm**
- dmlx：wq_a output norm + wkv output norm 是分开的 encoder（在 CB1 内）
- ds4：一次 kernel 搞定，但 dmlx 已经合并在 CB1 里了，差异不大

---

### 2.3 Partial RoPE (dsv4_rope.metal)

**`kernel_dsv4_rope_tail_f32`**:
- 前 n_nope 个 element 直接 copy，后 n_dims 个 element 做 YaRN RoPE
- 和 dmlx 的 `pipe_rope_tail_bf16` 等价，但 ds4 是 f32
- 不是性能瓶颈

---

### 2.4 Router (dsv4_misc.metal)

**`kernel_dsv4_router_finalize_one`**:
- 256 threads，bitonic sort，输出 selected[6]
- hash mode 或 score mode
- dmlx 已有 `moe_route_gpu`，功能完全一致

**`kernel_dsv4_router_weights_one`**:
- 6 threads，gather + normalize weights
- 极简

**`kernel_dsv4_indexed_mixed_attention_heads8`**:
- CSA/HCA 的 indexed attention，8 heads per TG
- dmlx 的 `pipe_mla_sdpa_bfloat` 处理的是不同的 attention 形式（全量 KV cache）
- ds4 的是压缩注意力，不直接对应

---

### 2.5 KV Cache (dsv4_kv.metal)

**`kernel_dsv4_fp8_kv_quantize_f32`** / **`kernel_dsv4_kv_fp8_store_f32`**:
- KV cache 存 FP8（E4M3）格式
- 大幅节省 KV 内存，也减少 SDPA 时的 bandwidth
- dmlx 当前 KV cache 是 f16，无 FP8

---

### 2.6 MoE Expert (moe.metal)

**`kernel_dsv4_moe_swiglu_weight`**:
- gate + up + SwiGLU + route_weight 融合，**包含 route weight multiply**
- dmlx 的 `gather_gate_up_swiglu` 类似，但不包含 route weight
- 差异不大

**`kernel_mul_mv_id_q4_K_*`**:
- Q4_K (GGUF 格式) expert matmul
- dmlx 用 affine 4-bit，不同量化格式

---

## 3. 核心差距汇总

| 功能 | ds4 实现 | dmlx 当前 | 差距 |
|------|---------|----------|------|
| mHC pre | 1 dispatch，与 RMSNorm 融合 | CB-A(wait) + CPU readback | **2 extra waits** |
| mHC post | 融合进 down proj | cb3(deferred wait) | **1 wait** |
| attention output + HC | 1 fused dispatch | CB3(wo_a×8+concat+wo_b) + cb3(mhc_post) | **wo_a 128MB f32 + 1 wait** |
| KV cache | FP8，节省 bandwidth | f16 | KV bandwidth 2× |
| QKV norm | fused | 分开 encoder，在 CB1 内 | 差异小 |
| Router | GPU bitonic | 已实现 | 持平 |
| SDPA | Flash Attention | mla_sdpa_bfloat | 差距大（无 Flash） |

---

## 4. 为什么 ds4 快：3 个根本原因

### 原因 1：mHC 完全 GPU-side，无 CPU round-trip

ds4 的 `kernel_dsv4_hc_split_weighted_sum_norm4` 在一次 dispatch 里完成：
1. Sinkhorn normalize (tid=0)
2. Weighted sum (全部 thread，shared memory 复用 pre[4])
3. RMSNorm (1024 threads，float4，4096-wide row)

dmlx 需要：CB-A(wait) → CPU(read post/comb) → CMD2(mhc_post+mhc_pre_ffn+norm)

**这就是 CB-A wait = 32ms/token（实测）的来源。**

### 原因 2：attention output + HC expand 融合

`kernel_dsv4_q8_hc_expand4_q8_0` 在 Q8_0 down projection 的同一次 dispatch 里完成 HC expand。

相当于：wo 投影（dmlx 的 wo_a×8+wo_b）+ mhc_post_ffn = 1 dispatch。

dmlx 分成：CB3(8×wo_a+blit+wo_b, wait) + cb3(mhc_post_ffn, deferred)。

### 原因 3：attention 权重 Q8_0，不是 f32 dense

ds4 的 wo_a 等效物是 Q8_0，每行 `4096×1B = 4KB`（vs dmlx wo_a f32 `4096×4B = 16KB`）。
4× bandwidth reduction on wo_a。

---

## 5. 移植可行性评估

| 组件 | 移植难度 | 收益 |
|------|---------|------|
| `kernel_dsv4_hc_split_weighted_sum_norm4` | 高（需要改 engine.c 流水线） | **最大**（消除 CB-A wait + RMSNorm wait） |
| `kernel_dsv4_hc_expand4` | 低（独立 kernel，替换 cb3 mhc_post_ffn） | 中（消除 cb3 deferred wait开销） |
| `kernel_dsv4_shared_down_hc_expand4_q8_0` | 高（需要 Q8_0 权重） | 高（消除 shared expert wait + mhc_post_ffn wait） |
| wo_a 改 Q8_0 | 中（loader + kernel） | 高（4× bandwidth reduction vs f32） |
| FP8 KV cache | 高（SDPA 对应改动） | 中（KV bandwidth 2×） |

---

## 6. 可立即实施的最小改动

不需要重写整个流水线，只需要：

### Step 1: `kernel_dsv4_hc_expand4` 移植

替换 dmlx 的 mhc_post_ffn（cb3）。

这个 kernel 极简：每个 thread 处理 1 (d, t)，4 HC streams，没有 shared memory。
输入：block_out (ffn output), residual_hc, post (per-HC gate), comb (HC×HC matrix)
输出：new_residual[HC, DIM]

对应 dmlx：`pipe_mhc_post_bfloat` + bf16→f32 conversions

**好处**：消除 f32→bf16, mhc_post, bf16→f32 三个 encoder + 它们的开销，单个 f32 kernel 搞定。

### Step 2: mhc_pre + input_norm fusion

`kernel_dsv4_hc_split_weighted_sum_norm4` 的 HC-pre 部分（1 dispatch，产生 attn input + normed）。

替换 dmlx 的 CB-A（mhc_pre + input_norm，2 encoders，1 wait）。

**好处**：不减少 wait 次数，但减少 kernel launch overhead。

---

## 7. 结论

**dmlx native engine 比 ds4/flash-moe 慢的核心原因不是 kernel 效率，而是 mHC 导致的架构性问题：**

1. mHC 要求 CPU 读 post/comb 系数（CB-A wait = 32ms/token）
2. wo_a f32 dense 128MB bandwidth（wo_a+wo_b = 365ms/token）
3. mhc_post_ffn 是独立 CB（cb3 deferred = 5ms/token）

ds4 通过 fused kernels 消除了 1 和 3，通过 Q8_0 quantization 解决了 2。

**最高 ROI 的下一步**：移植 `kernel_dsv4_hc_expand4` 替换 mhc_post_ffn cb3，验证正确性后再移植 `kernel_dsv4_hc_split_weighted_sum_norm4`。
