# CB-A Wait 消除 — 流水线重构方案

date: 2026-06-11
current: commit `6b704ef`，0.778 tok/s
target: 消除 CB-A wait（32ms/token）+ CB1/CB3 间隔，目标 ≥ 1.0 tok/s

---

## 0. 问题分析

### 0.1 当前流水线（4 个 GPU sync 点）

```
L(N-1) cb3 deferred wait ──→ CPU 读 residual
  ↓
CB-A wait ──→ CPU 读 attn_input + normed          ← 32ms/token
  ↓
CB1 wait ──→ (internal, MLA attention Q/KV/SDPA)
  ↓
CB3 wait ──→ CPU 读 attn_out
  ↓
CMD2 wait ──→ CPU 读 residual + routing            ← 52ms/token
  ↓
MoE + shared expert wait ──→ CPU 读 ffn_out
  ↓
cb3 deferred (no wait) ──→ next layer
```

每个 token 43 层，每层 ~18ms，其中 GPU sync 等待 ~84ms（CB-A 32ms + CMD2 52ms）。

### 0.2 ds4 的做法

ds4 将整个 layer 编码进**一个** command buffer：
```
begin_cb → mhc_pre → attention → hc_expand → mhc_pre_ffn → FFN → hc_expand → commit+wait
```

一次 `waitUntilCompleted`，中间零 CPU 干预。dmlx 无法直接照搬，因为：
- dmlx 的 expert I/O 需要 CPU 做路由决策 + 文件读取（ds4 也是 CPU routing + I/O）
- dmlx 的 MLA attention 接口要求 CPU pointer 入参

### 0.3 最小可行目标

**消除 CB-A wait（32ms）**：将 mhc_pre 编码合并进 MLA attention 的 CB1，消除中间的 CPU↔GPU 同步。

CMD2 wait（52ms）暂时保留——它涉及 expert I/O，需要 CPU 侧路由决策，改动更大。

---

## 1. 重构方案

### 1.1 核心思路

将 `mhc_pre_split_weighted_sum_norm`（当前在 `engine.c` 的 CB-A 中）移入 `mla_attention_decode_bf16` 的 CB1 中，作为第一个 encoder。CB1 的输出（normed_bf16）直接在 GPU 上传递给后续的 Q/KV chain，无需 CPU 读回。

### 1.2 修改后的流水线

```
L(N-1) cb3 deferred wait ──→ CPU 读 residual
  ↓
CB-MEGA (CB-A + CB1 + CB3 合并，一个 CB，一次 wait)
  Enc 1: mhc_pre_split_weighted_sum_norm → post/comb/normed_bf16 (GPU)
  Enc 2..N: Q chain + KV chain + SDPA (读 normed_bf16, GPU 内传递)
  Enc N+1..: wo_a × 8 + wo_b (读 SDPA 输出, GPU 内传递)
  wait ──→ CPU 读 attn_out + attn_input
  ↓
CMD2 wait ──→ (不变)
  ↓
MoE + shared expert wait ──→ (不变)
  ↓
cb3 deferred ──→ (不变)
```

**节省**：CB-A wait 32ms/token。

### 1.3 对 tok/s 的影响

当前 771ms/token → 771 - 32 = 739ms/token → **1.35 tok/s** (+73% over 0.778)。

---

## 2. 具体改动

### 2.1 `mla_attention_decode_bf16` 接口变更

**文件**：`src/metal_infer/mla_attention.h`

```c
// 新增 GPU-input 变体：x 来自 GPU buffer 而非 CPU pointer
int mla_attention_decode_bf16_gpu_in(
    MlaPipes *pipes, const AttnWeights *aw,
    id<MTLBuffer> x_bf16,              // [DIM] bf16，GPU buffer（替代 CPU uint16_t *x）
    uint16_t *kv_cache, int cache_len,
    int pos, float *out, void *kv_cache_gpu_buf);
```

或者更简洁的方案——在现有函数中检测 `x == NULL` 时从 GPU buffer 读：

```c
// x: 如果非 NULL，同原来。如果 NULL，从 x_gpu_buf 读
// x_gpu_buf: 可选的 GPU buffer（当 x == NULL 时使用）
int mla_attention_decode_bf16(MlaPipes *pipes, const AttnWeights *aw,
    const uint16_t *x, uint16_t *kv_cache, int cache_len,
    int pos, float *out, void *kv_cache_gpu_buf, void *x_gpu_buf);
```

**推荐第二种**：改动最小，向后兼容。

### 2.2 `mla_attention_decode_bf16` 内部改动

**文件**：`src/metal_infer/mla_attention.m`

改动点（~620 行附近）：

```objc
// 旧代码：
id<MTLBuffer> bx = mkbuf(d, x, DIM * sizeof(uint16_t));

// 新代码：
id<MTLBuffer> bx;
if (x_gpu_buf) {
    bx = (__bridge id<MTLBuffer>)x_gpu_buf;
} else {
    bx = mkbuf(d, x, DIM * sizeof(uint16_t));
}
```

仅此一处改动。后续 Q chain、KV chain、SDPA 都不变——它们已经通过 `bx` 使用 GPU buffer。

### 2.3 CB-A 移入 `engine.c` → 调用 MLA 时传入 GPU buffer

**文件**：`src/metal_infer/engine.c`（CB-A 段，~1078 行）

```c
// === 旧代码：CB-A 独立编码 + wait + CPU 读回 ===
{
    id<MTLCommandBuffer> cb = ...;
    // mhc_pre... (1 encoder)
    [cb commit]; [cb waitUntilCompleted];
    memcpy(attn_input, ...);
    memcpy(normed_bf16_direct, ...);
    // bf16→f32 转换 normed
}

// MLA attention(normed, ...)

// === 新代码：CB-A 移入 MLA，CB-A + CB1 合并 ===
// MLA attention 内部做 mhc_pre + Q/KV/SDPA
// 传入 GPU buffers 替代 CPU pointers
mla_attention_decode_bf16(..., 
    NULL,                    // x = NULL（使用 GPU buffer）
    kv_cache, cache_len, pos, attn_out,
    kv_cache_gpu_buf,        // 现有
    eng->buf_mhc_attn_norm_bf16  // x_gpu_buf: CB-A 在此 buffer 中
);

// CB-A 的 mhc_pre 不在此处编码——它已在 MLA 内部完成。
// attn_input 仍需要 CPU 读回（给 compressor），但可以异步。
```

**关键**：`buf_mhc_attn_norm_bf16` 现在由 MLA 内部的 mhc_pre encoder 写入，而不是 engine.c 的独立 CB-A。

但这意味着 mhc_pre 的编码必须在 MLA 内部完成。有两种实现方式：

**方式 A**：在 `mla_attention_decode_bf16` 内部编码 mhc_pre

- 优点：简洁，CB-A + CB1 真正合并为一个 CB
- 缺点：MLA 函数需要知道 mhc_pre 的 buffer 布局（fn_weight, base, scale, residual）

**方式 B**：在 `engine.c` 中预先将 mhc_pre 编码进一个 CB，然后将此 CB 的 encoder 传递给 MLA

- 优点：不改 MLA 内部结构
- 缺点：Metal 不支持跨函数共享 encoder

**选择方式 A**。需要给 MLA 函数传递 mhc_pre 所需的 buffers：

```c
// 扩展 MlaPipes 或新增参数
int mla_attention_decode_bf16_with_hc_pre(
    MlaPipes *pipes, const AttnWeights *aw,
    // mhc_pre 参数：
    id<MTLBuffer> hc_fn, id<MTLBuffer> hc_base, id<MTLBuffer> hc_scale,
    id<MTLBuffer> residual_gpu,
    id<MTLBuffer> post_out, id<MTLBuffer> comb_out,
    id<MTLBuffer> attn_input_out,
    id<MTLBuffer> norm_weight,
    // 原有参数：
    uint16_t *kv_cache, int cache_len, int pos,
    float *out, void *kv_cache_gpu_buf);
```

这个签名太长了。更实用的做法：**给 MLA 函数传入一个预编码的 mhc_pre CB**，MLA 将其作为 CB1 的第一个 encoder。

但 Metal 不支持这种操作。encoder 必须在 CB 内创建。

**最实用的方案**：在 `engine.c` 中创建一个新的 CB，将 mhc_pre + MLA attention 全部编码进去。不调用 `mla_attention_decode_bf16`，而是内联它的编码逻辑。

实际上，让我重新思考。当前 `mla_attention_decode_bf16` 创建了自己的 CB（CB1 和 CB3）。如果把 mhc_pre 也放进去，就变成了一个包含 mhc_pre + Q chain + KV chain + SDPA + wo_a + wo_b 的超大 CB。

一个折中方案：**在 engine.c 中创建一个 CB，先编码 mhc_pre，再调用 MLA 的 Q/KV/SDPA 编码（不创建新 CB）**。这需要将 MLA 的内部编码逻辑拆分为可重用的函数。

但这改动太大。让我退一步思考最简单的可行方案。

### 2.4 最简方案：GPU buffer pass-through

**不改 MLA 内部结构**，只消除 CPU 读回 normed_bf16：

当前：
```
CB-A → buf_mhc_attn_norm_bf16 (GPU)
CB-A wait
CPU read buf_mhc_attn_norm_bf16 → normed_bf16_direct → normed (f32)
MLA: normed → bf16 → bx (GPU buffer) → CB1
```

改为：
```
CB-A + CB1 合并为一个 CB:
  Enc 1: mhc_pre → buf_mhc_attn_norm_bf16 (GPU)
  Enc 2: Q chain (读 buf_mhc_attn_norm_bf16) → ...
  ...
  [commit + wait]
```

关键变化：在 engine.c 中，不创建独立的 CB-A。而是在一个 CB 中先编码 mhc_pre，再编码 MLA attention 的 CB1。

但这需要 engine.c 能够直接编码 MLA attention 的 CB1。当前 MLA attention 封装了这个逻辑。

**最终决策**：采用两阶段重构。

**Phase A**（最小改动，验证可行）：eliminate normed CPU readback，但仍保留 CB-A 作为独立 CB（与 CB1 不合并）。

具体做法：
1. CB-A 将 normed_bf16 写在 GPU buffer `buf_mhc_attn_norm_bf16`
2. CB-A commit + wait（保留）
3. 不再 CPU 读回 normed
4. 将 `buf_mhc_attn_norm_bf16` 作为 `x_gpu_buf` 传给 MLA
5. MLA 中跳过 CPU→GPU 上传

Phase A 不改 CB-A wait，只消除 CPU↔GPU 的数据搬运。wait 时间基本不变。

**Phase B**（真正合并 CB-A + CB1）：将 mhc_pre 编码移入 MLA 函数，或者将 MLA 编码逻辑暴露给 engine.c。

让我设计 Phase B 的具体实现。

### 2.5 Phase B 详细设计：CB-A + CB1 合并

**新增函数**：`mla_attention_encode_qkv_sdpa` — 编码 Q/KV/SDPA 到指定的 command buffer

```c
// 将 Q chain + KV chain + SDPA 编码到给定的 command buffer
// x_bf16_buf: GPU buffer [DIM] bf16（已经 normed）
static void mla_encode_qkv_sdpa(
    MlaPipes *P, id<MTLCommandBuffer> cb,
    const AttnWeights *aw, AttnBufCache *abc,
    id<MTLBuffer> x_bf16_buf,
    id<MTLBuffer> kvcache_buf, int cache_len, int pos,
    id<MTLBuffer> bcos, id<MTLBuffer> bsin,
    id<MTLBuffer> out_sdpa_bf16  // [N_HEADS, HEAD_DIM] bf16 output
);
```

然后在 `engine.c` 中：
```c
// Phase 1: CB-A + CB1 merged
{
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    
    // Encoder 1: mhc_pre (原 CB-A)
    encode_mhc_pre(cb, ...);
    
    // Encoder 2..N: Q chain + KV chain + SDPA (原 CB1)
    mla_encode_qkv_sdpa(P, cb, aw, abc, 
        eng->buf_mhc_attn_norm_bf16,  // GPU buffer, written by Enc 1
        ...);
    
    [cb commit]; [cb waitUntilCompleted];
    
    // CPU reads: attn_input, SDPA output (for CB3)
}
```

---

## 3. 实施步骤

### Step 1: GPU buffer pass-through for MLA input（Phase A）

**文件**：`mla_attention.h`, `mla_attention.m`
- 新增 `x_gpu_buf` 参数到 `mla_attention_decode_bf16`（或新增变体函数）
- 当 `x == NULL && x_gpu_buf != NULL` 时，直接使用 GPU buffer
- 改动量：~10 行

**文件**：`engine.c`
- CB-A 后不再 CPU 读回 normed_bf16
- 将 `buf_mhc_attn_norm_bf16` 作为 `x_gpu_buf` 传给 MLA
- 改动量：~15 行删除 + 5 行新增

**验证**：benchmark — 应保持 Paris ✓，性能略有提升（消除 2 次 memcpy）

### Step 2: 提取 MLA Q/KV/SDPA 编码逻辑（Phase B 预备）

**文件**：`mla_attention.m`
- 新增 `mla_encode_qkv_sdpa()` 函数
- 从 `mla_attention_decode_bf16` 中提取 CB1 的编码逻辑
- 改动量：~50 行新增 + 原有函数调用新函数

### Step 3: CB-A + CB1 合并（Phase B 主体）

**文件**：`engine.c`
- 删除独立的 CB-A
- 新增 merged CB：mhc_pre + Q/KV/SDPA
- CPU 读取 attn_input + SDPA 输出（用于 CB3）
- 改动量：~40 行

**文件**：`mla_attention.m`
- CB1 → 被 engine.c 的 merged CB 替代
- 原有 CB1 编码逻辑通过 `mla_encode_qkv_sdpa` 调用
- 改动量：~10 行

### Step 4: CB3 + CB1 合并（将 wo_a+wo_b 也放进 merged CB）

**文件**：`engine.c` + `mla_attention.m`
- 将 CB3 的 wo_a+wo_b 编码也加入 merged CB
- 完全消除 CB1→CB3 的间隔
- 改动量：~30 行

### Step 5: 验证

每步完成后：
```bash
bash scripts/run_benchmark.sh
# Paris ✓，tok/s 不退化
```

---

## 4. 文件改动清单

| 步骤 | 文件 | 改动 |
|------|------|------|
| Step 1 | `mla_attention.h` | 新增 `x_gpu_buf` 参数 |
| Step 1 | `mla_attention.m` | GPU buffer 分支（~10 行） |
| Step 1 | `engine.c` | 消除 normed CPU readback（~20 行） |
| Step 2 | `mla_attention.m` | 提取 `mla_encode_qkv_sdpa`（~50 行） |
| Step 3 | `engine.c` | CB-A + CB1 合并（~40 行新增，~30 行删除） |
| Step 4 | `engine.c` | CB3 合并进 merged CB（~30 行） |

---

## 5. 预期收益

| 步骤 | 消除项 | 预期 tok/s |
|------|--------|-----------|
| Step 1 | normed CPU readback | ~0.79（+1-2%） |
| Step 3 | CB-A wait（32ms） | ~1.0-1.1（+30-40%） |
| Step 4 | CB3 sync（~5ms） | ~1.05-1.15 |

累计：0.778 → ~1.1 tok/s（+41%）

---

## 6. 风险

| 风险 | 缓解 |
|------|------|
| CB-A + CB1 合并后 CB3 读 SDPA 输出时序错误 | CB3 在同一 CB 内，encoder 顺序保证正确性 |
| `buf_mhc_attn_norm_bf16` 被 CB1 的 encoder 覆盖 | 确认 buffer 无别名 |
| compressor/indexer 需要 attn_input | CPU 仍可从 `buf_mhc_attn_in` 读回 |
| 重构后 Paris 失败 | 逐 step 提交，每步 benchmark 验证 |