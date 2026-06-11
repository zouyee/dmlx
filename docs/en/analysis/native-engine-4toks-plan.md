# Native Engine：4 tok/s 性能优化 — 执行记录 + 瓶颈分析

date: 2026-06-11（持续更新）
baseline: commit `e01aed5`，实测 **0.709 tok/s**（SMELT N=51，热状态，M4 Pro 48GB）
target: **≥ 4 tok/s**
current: commit `b750770`，实测 **~0.778 tok/s**（benchmark 固有不稳定性 ±8%，无统计显著改进）

---

## 0. 当前性能 profile（NATIVE_TIME_LAYERS=1, pos=8, warm state）

```
Hash layers (0-2):  88ms (3.6%)  — L0 9.5ms, L1 7.3ms, L2 71.0ms
Score layers odd:   ~42ms × 20 = 840ms (34.6%)
Score layers even:  ~75ms × 20 = 1500ms (61.8%)
─────────────────────────────────
Total:              ~2529ms (with profiling overhead)
Estimated raw:      ~1264ms → 0.79 tok/s ✓ (matches benchmark ~0.78)
```

**关键发现：score layers 交替出现 42ms/75ms 模式**。差值的 33ms ✕ 20 层 = 660ms 是最大单一损失。

---

## 1. 已完成优化（5 个 commit, 639 insertions）

| Commit | 内容 | 类型 | tok/s | 收益 |
|--------|------|------|-------|------|
| `b4b63f0` | Phase 1+2+3: mHC fusion + Q8_0 wo_a | **kernel fusion + bandwidth** | 0.752 | +6.1% |
| `6b704ef` | Phase 4: coalesced wo_b v2 | **bandwidth** | 0.778 | +3.5% |
| `ea18a9c` | GPU buffer pass-through | CB merge (无效) | ~0.72 | 0 |
| `40425ab` | CB-A + CB1 merge | CB merge (无效) | ~0.67 | 0 |
| `b750770` | GPU blit CB1+CB3 merge | CB merge (无效) | ~0.72 | 0 |

**结论：只有减少 GPU 计算量/带宽的优化有收益。CB 级合并不减少 GPU 计算，无效。**

---

## 2. 关键教训：CB 合并 ≠ Kernel 融合

### 2.1 文档此前没有区分清楚

`ds4-kernel-deconstruction.md` 原文：
> "CB-A wait = 32ms/token（实测）的来源（是 CB 边界导致的）"
> "最高 ROI：移植 kernel_dsv4_hc_split_weighted_sum_norm4（消除 CB-A wait）"

**这是错误的**。32ms 是 GPU 执行 mhc_pre 的时间，不是 "CB 边界的 idle wait"。

### 2.2 正确理解

| | CB 合并 | Kernel 融合 |
|---|---|---|
| 做了什么 | 多个 kernel dispatch → 同一 CB | 多个操作 → 一个 kernel dispatch |
| 减少 GPU 计算？ | ❌ 不 | ✅ 是（消除中间读写） |
| 减少 dispatch 开销？ | ✅ 微秒级 | ✅ 同左 |
| 实际收益 | 0（已验证 3 次） | +6.1%（Phase 1-2） |

### 2.3 ds4 真正优势是 Kernel 融合，不是 CB 合并

ds4 的单个 CB 包含的不是多个独立 kernel dispatch，而是 **fused kernel dispatch**：
- `kernel_dsv4_hc_split_weighted_sum_norm4`: 3 个操作融合为 1 个 kernel（我们已移植 ✓）
- `kernel_dsv4_q8_hc_expand4_q8_0`: attention 输出 + HC expand 融合为 1 个 kernel（未移植）
- `kernel_dsv4_shared_down_hc_expand4_q8_0`: shared expert + HC expand 融合（未移植）

---

## 3. Phase-level profiling 结果（NATIVE_PHASE_TIME=1，pos=1 warm-state）

### 3.1 每层耗时分解

| Phase | 时间（with overhead） | 占比 | 估算 raw（÷10） | 43层 raw |
|-------|---------------------|------|----------------|---------|
| MLA (merged CB-A+CB1+CB3) | 15.6ms | 5.5% | 1.6ms | 69ms |
| CMD2 | 8.1ms | 2.8% | 0.8ms | 34ms |
| Expert I/O | 0.6ms | 0.2% | 0.06ms | 3ms |
| **MoE GPU** | **258.5ms** | **90.3%** | **25.9ms** | **1111ms** |
| Shared expert | 3.6ms | 1.2% | 0.4ms | 17ms |
| **Total** | **286ms** | | **29ms** | **1234ms** |

> Note: Profiling adds ~10× overhead from clock_gettime at each phase boundary.
> Relative ratios are accurate; absolute numbers projected from benchmark 0.778 tok/s.
> Estimated raw matches benchmark (1234ms vs 1285ms measured).

---

### 3.3 结论：MoE GPU 是下一个优化目标

MoE GPU 占 per-layer 时间的 90%，估计 43 层合计 **~1100ms**（benchmark 总 1285ms 的 86%）。

当前 separate mode: 12 dispatches/layer (6 × fused_gate_up_swiglu + 6 × dequant_matvec_4bit).
每个 dispatch 使用 x_shared[4096]=16KB（flash-moe v3 pattern）。
Gather mode 默认禁用（13MB expert-stride 导致 scattered memory access，实测比 separate 更慢）。

### 3.4 MoE v2 尝试：no-x_shared pattern

**结果**：Paris ✓，0.704 tok/s — 无改进。

**原因**：v1（256 threads, x_shared[4096], 8 rows/TG）vs v2（128 threads, no x_shared, 2 rows/TG）。v1 的 x_shared 成本分摊到 8 行，v2 减少的并行度抵消了 x_shared 的节省。

**结论**：MoE GPU 是 **compute-bound**，不是 occupancy-bound。当前 flash-moe v3 pattern 已接近最优。进一步优化需改 expert 量化格式（如 ds4 的 Q8_0）。

---

## 4. Expert I/O 现状

- SMELT N=51：51 个 experts/layer 缓存 在 RAM（~29GB）
- penalty routing：确保 6 个选中 experts 都在缓存中
- `io_pool_dispatch_cached`：全命中时零 I/O（直接返回 RAM 指针）
- gather mode：默认禁用（13MB expert-stride 导致 scattered access，比 separate 更慢）
- **结论：expert I/O 瓶颈已被 SMELT 解决，剩余 208ms 是纯 GPU compute + dispatch overhead**

---

## 5. 已移植的 ds4 内核

| dmlx 内核 | ds4 原版 | 类型 | Commit |
|-----------|---------|------|--------|
| `mhc_post_ffn_expand4` | `kernel_dsv4_hc_expand4` | kernel fusion | `b4b63f0` |
| `mhc_pre_split_weighted_sum_norm` | `kernel_dsv4_hc_split_weighted_sum_norm4` | kernel fusion | `b4b63f0` |
| `matvec_q8_0_f32` | `kernel_mul_mv_q8_0_f32` | bandwidth | `b4b63f0` |
| `dequant_matvec_affine_v2` | ds4 Q8_0 模式适配 | bandwidth | `6b704ef` |

## 6. 未移植的 ds4 内核 + flash-moe 对比

| 内核/技术 | 原因 | 备注 |
|-----------|------|------|
| `kernel_dsv4_q8_hc_expand4_q8_0` | 需要 wo_b 改为 Q8_0 格式 | Q8_0 带宽需求高，M4 Pro 120GB/s 可能不够 |
| `kernel_dsv4_shared_down_hc_expand4_q8_0` | 需要 shared expert 改为 Q8_0 格式 | 同上 |
| Expert MXFP4→affine 4-bit | flash-moe 用此格式达到 4.36 tok/s | **SMELT 兼容，内存不变** |

### 6.1 flash-moe 对比：affine 4-bit vs MXFP4

| | flash-moe (Qwen3.5, 4.36 tok/s) | dmlx (DSV4, 0.78 tok/s) |
|---|---|---|
| Expert 量化 | affine 4-bit (g=64): w = nibble * scale + bias | MXFP4 E2M1 (g=32): w = LUT[nibble] * exp2(scale-127) |
| Dequant 操作 | 简单: multiply + add | 复杂: LUT + exp2 + multiply |
| Expert 大小 | ~10.5MB | ~10.5MB |
| GPU 瓶颈 | I/O (SSD → RAM) | GPU compute (LUT+exp2 overhead) |

**结论**：将 expert 从 MXFP4 改为 affine 4-bit 可能消除 MoE GPU compute 瓶颈。SMELT N=51 仍可在 48GB 内存中运行，I/O 不变，仅为 dequant 算法替换。

---

## 7. 验收标准

```bash
sudo purge
NATIVE_TIME_LAYERS=1 bash scripts/run_benchmark.sh  # 获得 phase-level 时间分解
# 要求：Paris ✓，tok/s 不退化
```