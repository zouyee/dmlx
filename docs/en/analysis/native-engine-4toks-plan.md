# Native Engine：4 tok/s 性能优化 — 执行记录 + ds4 路径

date: 2026-06-11（持续更新）
baseline: commit `e01aed5`，实测 **0.709 tok/s**（SMELT N=51，热状态，M4 Pro 48GB）
target: **≥ 4 tok/s**
current: commit `40425ab`，实测 **~0.72 tok/s**（benchmark 噪声范围，无统计显著改进）

---

## 1. 已完成的优化

### 1.1 性能演进

| Commit | 内容 | tok/s | 提升 | 备注 |
|--------|------|-------|------|------|
| `e01aed5` | **baseline** | **0.709** | — | — |
| `b4b63f0` | Phase 1+2+3: mHC fusion + Q8_0 wo_a | 0.752 | +6.1% | 实质收益 |
| `6b704ef` | Phase 4: coalesced wo_b v2 | 0.778 | +3.5% | 实质收益 |
| `ea18a9c` | GPU buffer pass-through | ~0.72 | 0 | 接口重构，无收益 |
| `40425ab` | CB-A + CB1 merge | ~0.72 | 0 | 接口重构，无收益 |

### 1.2 各 Phase 详情

见下方"已移植的 ds4 内核"章节。

---

## 2. CB-A Wait 消除尝试 — 为什么失败了

### 2.1 尝试的内容

两个 commit：

1. **GPU buffer pass-through** (`ea18a9c`)：消除 CB-A → MLA 之间的 CPU normed 读回，传入 GPU buffer
2. **CB-A + CB1 merge** (`40425ab`)：将 mhc_pre 编码合并进 CB1，消除独立的 CB-A

### 2.2 为什么性能没提升

根因：**CB1 和 CB3 之间仍然有 CPU 读回**。

```
merged CB:
  Enc 1:  mhc_pre → buf_mhc_attn_norm_bf16 (GPU)
  Enc 2..N: Q/KV/SDPA → battn (GPU)
  [commit + wait]           ← 这个 wait 仍然存在
  CPU: 读 battn → 提取 wo_a 8 groups → bconcat (CPU)  ← 强制 CPU 读回
  CB3: wo_a × 8 + wo_b
  [commit + wait]
```

mhc_pre 和 Q/KV/SDPA 合并在同一个 CB 里了，但 CB 内部的 `[commit + wait]` 本身没变——GPU 执行的工作量完全一样，wait 时间也完全一样。只是少了一个 Metal encoder 的创建开销（微秒级，不可测）。

**真正的瓶颈是 CB1→CB3 之间的 CPU battn 读回**：mla_attention_decode_bf16 在 SDPA 之后必须把 `battn` 读到 CPU，做 head grouping（8 组 wo_a 各自提取对应的 head 输出），然后才能编码 wo_a+wo_b。这个 CPU 读回强制了 CB1 和 CB3 不能合并。

### 2.3 结论

**局部 CB 合并不起作用。必须遵循 ds4 的方式：整个 layer 一个 CB，所有中间数据留在 GPU 上。**

---

## 3. ds4 的正确路径：单 CB 全层流水线

### 3.1 ds4 架构

ds4 把每个 layer 的所有操作编码进**一个** command buffer：

```
begin_cb
  → mhc_pre (split + weighted sum + norm)
  → attention (Q/K/V/SDPA)
  → attention output + HC expand (FUSED: Q8_0 matvec + mhc_post)
  → mhc_pre_ffn (split + weighted sum + norm)
  → FFN (expert routing + compute + shared + combine)
  → FFN output + HC expand (FUSED)
commit + wait (一次)
```

关键差别：
- ds4 的 attention 输出直接做 HC expand，**不需要 CPU 读回 SDPA 输出做 head grouping**
- ds4 的 FFN 输出直接做 HC expand，**不需要 CPU 读回 ffn_out**
- 全程 GPU-side，零中间 CPU 干预

### 3.2 dmlx 要改成 ds4 方式需要做什么

**核心改动**：消除两个 CPU 读回点。

**读回点 1：SDPA 输出 → wo_a head grouping**

当前 `mla_attention_decode_bf16`：
```objc
// CB1: SDPA → battn (GPU, bf16)
[cb1 commit]; [cb1 waitUntilCompleted];
uint16_t *attn_bf16 = [battn contents];  // ← CPU 读回
for (int g = 0; g < 8; g++) {
    // CPU memcpy: 提取 group g 的 head 输出 → bgv
    memcpy(gv_bf16_data + hh * HEAD_DIM, attn_bf16 + ..., HEAD_DIM * sizeof(uint16_t));
    // CB3: wo_a matvec
    enc_matvec_q8_0(..., bgv, ...);
}
```

改为 GPU blit：
```objc
// CB1: SDPA → battn (GPU, bf16)
// NO commit, NO wait, NO CPU readback
// GPU blit: 直接提取 group g 的 head 输出 → bgv (GPU)
for (int g = 0; g < 8; g++) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    // blit battn[g*heads_per_group*HEAD_DIM .. (g+1)*heads_per_group*HEAD_DIM]
    //   → bgv[0 .. group_feat]
    [blit copyFromBuffer:battn sourceOffset:... toBuffer:bgv destinationOffset:0 size:...];
    [blit endEncoding];
    // CB3: wo_a matvec (same CB, after blit)
    enc_matvec_q8_0(..., bgv, ...);
}
// ONE commit + wait at the end
```

**读回点 2：ffn_out → mhc_post_ffn**

已解决（Phase 1 的 `mhc_post_ffn_expand4` 直接读 `buf_ffn_out_f32`）。

### 3.3 实施计划

#### Step A: GPU blit wo_a grouping（消除读回点 1）

**文件**：`mla_attention.m`（CB3 部分，~810-870 行）

改动：
- 不再 CPU 读 `battn`，用 Metal blit encoder 在 GPU 上提取每个 group 的 head 输出
- CB1 和 CB3 合并为一个 CB
- 一个 commit + wait

**涉及函数**：`mla_attention_decode_bf16`（或新增 `mla_attention_decode_bf16_merged`）

**关键**：blit encoder 必须在 compute encoder 之间穿插，Metal 支持在同一 CB 中混合 blit 和 compute。

#### Step B: CB-A + CB1 + CB3 全合并（在 Step A 基础上）

**文件**：`engine.c` + `mla_attention.m`

改动：
- engine.c 不再单独编码 CB-A
- mhc_pre 作为第一个 encoder 编入合并的 CB
- 然后 Q/KV/SDPA（GPU path）
- 然后 GPU blit wo_a grouping + wo_a×8 + wo_b
- 一个 commit + wait

**预期收益**：消除 CB-A wait（32ms） + CB1→CB3 间隔。tok/s: 0.72 → ~0.85-0.95。

#### Step C: CMD2 合并（远期，消除第二个 wait）

**文件**：`engine.c`

改动：
- CMD2（mhc_post + mhc_pre_ffn + routing）合并进同一个 CB
- Expert I/O 仍然需要 CPU 决策，但可以在 commit 之前编码 CMD2
- 一个 commit + wait per layer（ds4 方式）

**预期收益**：消除 CMD2 wait（52ms）。tok/s: ~0.95 → ~1.3-1.5。

---

## 4. 已移植的 ds4 内核

| dmlx 内核 | ds4 原版 | 作用 | Commit |
|-----------|---------|------|--------|
| `mhc_post_ffn_expand4` | `kernel_dsv4_hc_expand4` | cb3: 3 encoders → 1 | `b4b63f0` |
| `mhc_pre_split_weighted_sum_norm` | `kernel_dsv4_hc_split_weighted_sum_norm4` | CB-A: 2 encoders → 1 | `b4b63f0` |
| `matvec_q8_0_f32` | `kernel_mul_mv_q8_0_f32` | wo_a f32→Q8_0 | `b4b63f0` |
| `dequant_matvec_affine_v2` | ds4 Q8_0 模式适配 affine 4-bit | wo_b coalesced | `6b704ef` |

---

## 5. 关键教训

1. **bf16 精度匹配**：移植内核必须保留 `float(bfloat(...))` cast，否则 43 层累积误差导致 Paris 失败
2. **Q8_0 量化用 `roundf()`**：`(int)(v+0.5f)` 对负数舍入错误
3. **局部 CB 合并不起作用**：只要中间有 CPU 读回，合并就无法消除 wait。必须把所有操作放在一个 CB 中，全程 GPU-side
4. **GPU blit 是消除 CPU 读回的关键**：Metal blit encoder 可以在同一 CB 中重新排列 GPU buffer 数据，替代 CPU memcpy

---

## 6. 不做的事（已验证）

| 方案 | 结果 | 原因 |
|------|------|------|
| flash-moe v3 | 退步 | x_shared[4096]=16KB→occupancy 崩溃 |
| shared expert v2 | Paris ✗ | bug 未定位 |
| 局部 CB merge（无 GPU blit） | 无收益 | CB3 CPU 读回强制 wait |
| batched_wo_a / shared_mem xs / 4-bit 等 | 退步 | 均未解决 non-coalesced access |

---

## 7. 验收标准

```bash
sudo purge
bash scripts/run_benchmark.sh  # SMELT N=51，顺序，热状态
# 要求：Paris ✓，tok/s 不退化
```