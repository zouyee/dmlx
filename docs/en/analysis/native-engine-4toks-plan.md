# Native Engine：4 tok/s 性能优化 — 执行记录

date: 2026-06-11（持续更新）
baseline: commit `e01aed5`，实测 **0.709 tok/s**（SMELT N=51，热状态，M4 Pro 48GB）
target: **≥ 4 tok/s**
current: commit `6b704ef`，实测 **0.778 tok/s** (+9.7%)

---

## 相关分析文档

- **[ds4 Kernel 深度解构](ds4-kernel-deconstruction.md)** — ds4 Metal kernels 逐文件分析
- 本文件：ds4 移植执行记录 + 后续计划

---

## 1. 实测 Profiling 数据（已验证，2026-06-10）

### 1.1 43层全量相位分解（NATIVE_PHASE_TIME，热状态）

| 阶段 | 43层合计 | 占比 |
|------|---------|------|
| **MLA attention**（CB1+CB3） | **429ms** | **56%** |
| expertIO + MoE GPU | 208ms | 27% |
| CMD2 wait | 52ms | 7% |
| shared expert | 45ms | 6% |
| CB-A wait | 32ms | 4% |
| deferred cb3 | 5ms | 1% |
| **合计** | **771ms** | **100%** |

### 1.2 MLA 内部分解

| 阶段 | 占MLA比例 | 说明 |
|------|---------|------|
| **wo_a×8 + wo_b** | **84.7%** | 核心瓶颈，wo_a = 128MB f32 |
| wq_b + q_rope | 9.4% | wq_b=[32768×1024] |
| wq_a + q_norm | 2.8% | |
| wkv + blit | 2.6% | |
| SDPA | 0.6% | |

---

## 2. 为什么要移植 ds4 内核

### flash-moe v3 路径已被验证失败

flash-moe 的 `dequant_matvec_4bit_v3`（256 threads, `x_shared[4096]`=16KB）在 dmlx 上实测退步。根因：M4 Pro GPU threadgroup memory = 32KB，16KB 的 x_shared 导致 occupancy 崩溃。

### ds4 的 3 个关键优势

| ds4 内核 | shared memory | 为何适用 dmlx |
|---------|--------------|-------------|
| `kernel_dsv4_hc_expand4` | **0** | 纯寄存器计算，无 occupancy 风险 |
| `kernel_dsv4_hc_split_weighted_sum_norm4` | 16KB | 仅 1 TG/layer，43 次/token，非高频调用 |
| `kernel_mul_mv_q8_0_f32` | **256B** | NR0=2, NSG=4，極小 shared memory |

---

## 3. 已完成的优化（Phase 1-4）

### 3.1 性能演进

| Commit | 内容 | tok/s | 提升 |
|--------|------|-------|------|
| `e01aed5` | **baseline** | **0.709** | — |
| `b4b63f0` | Phase 1+2+3: mHC fusion + Q8_0 wo_a | 0.752 | +6.1% |
| `6b704ef` | Phase 4: coalesced wo_b v2 | 0.778 | +3.5% |
| **累计** | | **0.778** | **+9.7%** |

所有 commit Paris ✓ 通过，unit tests 430+ PASS。

### 3.2 Phase 1: `mhc_post_ffn_expand4`（`kernel_dsv4_hc_expand4`）

**来源**：ds4 `dsv4_hc.metal:579-620`

**改动**：替换 cb3 mhc_post_ffn 的 3 encoders（f32→bf16 + mhc_post_bfloat + bf16→f32）为 1 个纯 f32 dispatch。

**关键教训**：必须保留 bf16 精度截断（`float(bfloat(...))` cast），否则 43 层累积误差导致 Paris 失败。ds4 原版 kernel 没有这些 cast——因为 ds4 不使用 bf16 中间格式，全程 f32。

**内核特点**：1 thread/dim，计算全部 4 个 HC stream，零 shared memory。

### 3.3 Phase 2: `mhc_pre_split_weighted_sum_norm`（`kernel_dsv4_hc_split_weighted_sum_norm4`）

**来源**：ds4 `dsv4_hc.metal:395-536`

**改动**：替换 CB-A 的 2 encoders（mhc_pre_gpu + rms_norm）为 1 个 fused dispatch。

**关键教训**：同样需要 bf16 精度截断——weighted sum 的结果（collapsed row）需要用 `float(bfloat(acc))` 截断后再做 RMSNorm，以匹配旧 2-encoder 路径的 bf16 round-trip。

**内核特点**：256 threads，~17.5KB shared memory（row_shmem 16KB + mixes/comb/ss_buf ~1.5KB）。

### 3.4 Phase 3: `matvec_q8_0_f32` + wo_a Q8_0 量化（`kernel_mul_mv_q8_0_f32`）

**来源**：ds4 `dense.metal:108-176`

**改动**：
- wo_a 从 f32 dense（128MB/layer, 每行 16KB）改为 Q8_0（36MB/layer, 每行 4.5KB），3.56× bandwidth 减少
- 新增 Q8_0 量化代码（在 `AttnBufCache` 初始化时，`mla_attention.m`）：`d = amax/127, qs[i] = roundf(x[i]/d)`
- 新增 coalesced matvec kernel：NR0=2 rows/TG, NSG=4 simdgroups, 256B shared memory

**关键教训**：
- 必须使用 `roundf()` 而非 `(int)(v+0.5f)`（后者对负数舍入错误）
- Q8_0 量化在 AttnBufCache 中完成（lazy init, CPU→GPU upload），不需要改动 Zig loader
- 保留 f32 dense 作为 fallback

**内核特点**：128 threads/TG，512 TGs for [1024,4096]，coalesced weight reads via simd_lane striding。

### 3.5 Phase 4: `dequant_matvec_affine_v2`（ds4 Q8_0 模式适配 affine 4-bit）

**来源**：ds4 `kernel_mul_mv_q8_0_f32` 架构适配 affine 4-bit dequant

**改动**：wo_b 使用 coalesced affine 4-bit dequant kernel（NR0=2, NSG=4, 128 threads/TG）。

**内核特点**：
- 8 threads/group，每组处理 1 个 uint32 word（8 nibbles）
- 32 threads/SIMD group × 4 SIMD groups = 128 threads/TG
- FMA 优化：pre-compute `scale*x` 和 `bias*x`，用 `fma(nibble, sx, bx)`
- 256B shared memory（仅用于 reduction）
- 无 x_shared——避免 occupancy 问题

---

## 4. 尝试但失败/未采用的优化

| 方案 | 结果 | 原因 |
|------|------|------|
| shared expert 用 v2 coalesced kernel | Paris ✗，已 revert | bug 未定位（可能与 out_dim=2048 边缘条件相关） |
| 不加 bf16 截断的 mHC kernel | Paris ✗ | 43 层 bf16→f32 精度偏差累积 |
| Q8_0 量化用 `(int)(v+0.5f)` 舍入 | Paris ✗ | 负数舍入错误，改用 `roundf()` 后通过 |
| flash-moe v3 | 退步 | x_shared[4096]=16KB→occupancy 崩溃 |
| batched_wo_a / shared_mem / 4-bit 等 | 退步 | 均未解决 non-coalesced access 根因 |

---

## 5. 剩余机会分析

### 5.1 已做的 vs 未做的

| 优化项 | 状态 | 估算收益 |
|--------|------|---------|
| wo_a f32→Q8_0 | ✅ Phase 3 | +6.1% |
| wo_b coalesced dequant | ✅ Phase 4 | +3.5% |
| mHC encoder fusion (Phase 1+2) | ✅ | 持平（纯架构改进） |
| shared expert coalesced | ❌ 失败 | 待修复 |
| MoE expert GPU coalesced | ❌ 未做 | ~中（208ms 含 I/O，GPU 占比未知） |
| wq_b coalesced | ❌ 未做 | ~小（9.4% of MLA） |
| CB-A wait 消除 | ❌ 未做 | **最大**（32ms/token，需流水线重构） |
| CMD2 wait 消除 | ❌ 未做 | 中（52ms/token，需 GPU-side routing+mHC） |
| expert I/O 优化 | ❌ 未做 | 大（flash-moe 证明 I/O 是主要瓶颈） |

### 5.2 最高 ROI 后续步骤

1. **CB-A wait 消除**（32ms/token → 约 +4.5% tok/s）
   - 改变 `mla_attention_decode_bf16` 接口：接受 GPU buffer 而非 CPU pointer
   - 将 CB-A 结果通过 GPU buffer chain 传递给 MLA attention
   - 消除 CPU readback 的 wait

2. **shared expert v2 修复**（45ms/token → 约 +2-3%）
   - 定位 dequant_matvec_affine_v2 在 shared expert 上的 bug
   - 可能与 buffer 布局或 edge case 相关

3. **expert I/O + MoE GPU 优化**（208ms/token → 潜力最大）
   - flash-moe 证明 SSD I/O 是瓶颈，需异步预取或更高效的 I/O 策略
   - MoE GPU 内核可以用 ds4 模式改写（但收益受 I/O 限制）

---

## 6. 历史记录

| commit | 优化 | tok/s | Paris |
|--------|------|-------|-------|
| `214070a` | CB2+CMD2 合并 | 0.49 | ✓ |
| `a064cec` | GPU routing | 0.679 | ✓ |
| `e01aed5` | cb3 deferred | **0.709** | ✓ |
| `b4b63f0` | mHC fusion + Q8_0 wo_a | 0.752 | ✓ |
| `6b704ef` | coalesced wo_b v2 | **0.778** | ✓ |

---

## 7. 验收标准

每步改动前后都运行：
```bash
sudo purge
bash scripts/run_benchmark.sh  # SMELT N=51，顺序，热状态
# 要求：Paris ✓，tok/s ≥ 上一步
```