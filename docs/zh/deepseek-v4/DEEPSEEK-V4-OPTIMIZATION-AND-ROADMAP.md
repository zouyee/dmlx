# DeepSeek V4 优化与路线图

> **合并自:** optimization-plan.md (ZH), turboquant-analysis.md (ZH)  
> **最后更新:** 2026-04-27（优化计划 v3），2026-04-26（TurboQuant 分析）  
> **交叉引用:** [修复与细节](DEEPSEEK-V4-FIXES-AND-DETAILS.md) | [聊天分析与故障排查](DEEPSEEK-V4-CHAT-ANALYSIS-AND-TROUBLESHOOTING.md)

---

## 执行摘要

基于对 Apple 官方 `mlx-lm/mlx_lm/models/deepseek_v4.py`（2153 行）的深度审计以及 mlx-zig 当前能力（含 `CustomMetalKernel` 支持），本文档提供了可执行的 5 周优化路线图。mlx-zig 与 mlx-lm 之间的三个关键架构差异驱动了优化策略：(1) MoE 权重格式（fused vs split），(2) KV 压缩子系统（有状态 vs 纯函数），(3) Indexer 设计（Apple 重设计 vs 论文原始设计）。

此外，TurboQuant 分析确定了升级 FP8 KV 存储的具体步骤（现在可通过新的 `mlx-c` 0.6.0 绑定实现）以及实现最优 KV cache 量化的路径。

---

## 第一部分：优化计划 v3（来自 optimization-plan.md）

**修订版 v3** — 审计日期：2026-04-27  
v2 修正了 v1 中的 8 类问题；v3 修正了 v2 中的 3 处事实性错误，新增 2 项遗漏任务，删除 3 项已完成任务，时间线从 6 周压缩至 5 周。

### 1. 背景

`mlx-lm` 于 2026-04 发布了官方 `deepseek_v4.py`，覆盖：
- **CSA/HCA 混合注意力**：完整的 `Compressor` + `Indexer` + `_sparse_pooled_attention`
- **mHC 自定义 Metal Kernel**：`hc_split_sinkhorn` 和 `hc_sinkhorn_collapse`
- **高效 MoE Dispatch**：`DeepseekV4SwitchGLU` 使用 `gather_mm` + `sort_threshold=8`
- **精巧的 Cache 设计**：`DeepseekV4Cache` 管理 `local` + `compressor_state` + `indexer_state`

### 2. 关键架构差异

#### 2.1 MoE 权重格式：Fused vs Split

| 维度 | mlx-lm | mlx-zig |
|------|--------|---------|
| **存储格式** | `switch_mlp.weight` 为 `[n_experts, out, in]`（fused） | loader 将 fused tensor **split 成 256 个独立专家** |
| **Dispatch** | `mx.gather_mm` 一次性加载选中专家 | 遍历 256 个 `DSV4Expert` |
| **量化态** | `QuantizedSwitchLinear` 保持打包 | loader 反量化为 float16/float32 |

**结论：** mlx-zig 的 MoE 是 `fused → split → dequantize → per-expert loop` 路径，与 mlx-lm 的 `fused → gather_mm` 路径完全相反。优化需要**反向重构**。

#### 2.2 KV 压缩子系统：模块状态 vs 纯函数

| 维度 | mlx-lm | mlx-zig |
|------|--------|---------|
| **结构** | `Compressor` 是有状态的 `nn.Module` | `compressKV()` 是纯函数 |
| **Cache 集成** | `Compressor.__call__` 内部调用 `cache.accumulate_windows` | `compressKV()` 完全不知道 cache 存在 |
| **Overlap** | `_overlap_transform` 是成员方法 | 无 overlap 逻辑 |

**结论：** mlx-zig 缺少 `Compressor` 模块和 `DeepseekV4Cache` 状态机。

#### 2.3 Indexer：两种不同架构

| 维度 | mlx-lm `Indexer` | mlx-zig `LightningIndexer` |
|------|-------------------|---------------------------|
| **Q 投影** | `wq_b: Linear(q_lora_rank → n_heads * head_dim)` | `wq_index: [index_n_heads * index_head_dim, head_dim]` |
| **K 来源** | **内部嵌套 `Compressor`** | 直接接收外部 `k_compressed` |
| **FP4** | ❌ 全程 bfloat16/float32 | ⚠️ 有 `quantize4bit` 模拟（应删除） |

**结论：** 两个 Indexer 架构完全不同。mlx-zig 的更接近 DeepSeek V3 论文设计，mlx-lm 的是 Apple 重设计。

### 3. 优先级总览

| 优先级 | 任务数 | 核心主题 |
|--------|--------|----------|
| **P0** | 3 | 正确性修复 + 最大性能瓶颈（含 Attention 双分支） |
| **P1** | 5 | 架构对齐 + 工程简化 + Custom Kernel + RoPE GPU 化 + O-LoRA batch matmul |
| **P2** | 2 | 量化保持 + 权重加载 |
| **P3** | 1 | 细节打磨 |

### 4. P0：正确性与核心性能瓶颈

> **建议工期：Week 1-2**

#### 4.1 MoE Dispatch 反向重构：Gather/Sort + `gather_mm`

| 属性 | 内容 |
|------|------|
| **问题** | `DSV4MoE.forward()` 遍历全部 256 个专家。loader 已将 fused `switch_mlp` split 为独立专家，丢失了 gather 的可能性 |
| **mlx-lm 参考** | `DeepseekV4SwitchGLU` (`deepseek_v4.py:125-155`) |
| **关键机制** | `_gather_sort`：按专家 ID 排序（`sort_threshold=8`）；`mx.gather_mm`/`mx.gather_qmm`：只加载选中专家；`scores` 在 `down_proj` 前乘入；`_scatter_unsort` 恢复顺序 |
| **mlx-zig 行动** | 1. **停止 splitFusedExperts**：保留 fused 格式<br>2. **重构 DSV4MoE**：删除 `DSV4Expert` 数组<br>3. **接入 sort**：路由后按专家 ID `argsort`<br>4. **scores 前置融合** |
| **预期收益** | **10-100x** 专家计算提速 |
| **相关文件** | `src/models/deepseek_v4.zig:638-777`, `src/models/deepseek_v4_loader.zig:495-536` |

#### 4.2 KV 压缩子系统重构：`Compressor` + `DeepseekV4Cache`

| 属性 | 内容 |
|------|------|
| **问题** | 无 `_overlap_transform`：CSA 层丢失块重叠；无 `accumulate_windows`：尾部 token 被丢弃 |
| **mlx-zig 行动** | 1. **引入 `Compressor` 结构体**<br>2. **引入 `DeepseekV4Cache`**<br>3. **实现 `_overlap_transform`**<br>4. **实现 `accumulate_windows`** |
| **预期收益** | CSA 层正确性修复，支持任意长度输入 |
| **相关文件** | `src/models/deepseek_v4.zig:1367-1441`, `src/kvcache.zig` |

#### 4.3 Attention 双分支：Local / Pooled（L==1 vs L>1）

| 属性 | 内容 |
|------|------|
| **问题** | 所有 CSA 层统一走 `gatherBlocks → concat → dense SDPA`，未区分 generate（L=1）和 prefill（L>1） |
| **mlx-zig 行动** | 1. 增加 `if (seq_len == 1)` gather + dense SDPA 分支<br>2. 实现 `seq_len > 1` 的 `_sparse_pooled_attention` |
| **预期收益** | 解码阶段正确性修复 + latency 降低 |
| **相关文件** | `src/models/deepseek_v4.zig:1113-1361` |

### 5. P1：架构对齐与工程简化

> **建议工期：Week 3-4**

#### 5.1 删除 Indexer FP4 模拟 + 明确架构选择

**方案 A（推荐）：** 对齐 mlx-lm Indexer — 删除 `LightningIndexer`，创建 `Indexer` 结构体（嵌套 `Compressor`，`wq_b` 投影，`weights_proj`）

**方案 B：** 保留论文原始设计，仅删除 `quantize4bit`

#### 5.3 mHC 接入 `CustomMetalKernel`

将 mlx-lm 的 Metal source 翻译为 Zig 字符串，通过 `src/ops/custom_kernel.zig` 注册。触发条件：`!training && hc_mult==4 && dtype==bfloat16 && gpu`。Fallback 到现有 ops 路径。

**预期收益：** mHC 前向传播减少 80%+ kernel launch 开销。

#### 5.4 Compressor 区分 CSA 与 HCA 策略

CSA (4x)：gated pool + overlap + Indexer，输出 `head_dim * 2`
HCA (128x)：gated pool，无 overlap，无 Indexer，输出 `head_dim`

#### 5.5 Sparse_pooled 中的 Sink 处理

实现 `_sparse_pooled_attention` 时，显式 concat sink 到 scores 前端。

#### 5.6 RoPE 逆变换简化

将 `applyInverse` 合并到 `apply`，增加 `inverse: bool = false` 参数。

### 6. P2：量化、加载与绑定补齐

> **建议工期：Week 5**

#### 6.1 MoE 专家权重保持量化态

加载时保留 packed weight + scales（不 dequantize），改用 `gatherQmm`。

#### 6.2 O-LoRA Grouped Projection 支持量化路径

检测 `wo_a` 是否为量化权重，是则调用 `quantizedMatmul`。

### 7. 时间线

```text
Week 1-2 (P0) — 正确性与核心性能
├── Task 1: MoE 反向重构（fused 格式 + gather_mm dispatch）
├── Task 2: KV 压缩子系统重构（Compressor + DeepseekV4Cache）
└── Task 3: Attention 双分支（L==1 gather + L>1 sparse_pooled）

Week 3-4 (P1) — 架构对齐与 Custom Kernel
├── Task 4: Indexer 架构选择 + 删除 FP4 模拟
├── Task 5: mHC CustomMetalKernel
├── Task 6: CSA/HCA 策略区分
├── Task 7: Sparse_pooled Sink concat
└── Task 8: RoPE 逆变换简化

Week 5 (P2) — 量化与绑定补齐
├── Task 9: MoE 专家保持量化态（gather_qmm）
├── Task 10: O-LoRA 量化路径
└── Task 11: 权重加载 fused 格式对齐
```

### 8. 附录：mlx-lm vs mlx-zig 能力矩阵

| 特性 | mlx-lm (Python) | mlx-zig | 差距 |
|------|-----------------|---------|------|
| MoE gather_mm dispatch | ✅ | ❌ split + per-expert loop | **高** |
| CSA overlap transform | ✅ | ❌ 纯函数 compressKV | **高** |
| Cache accumulate_windows | ✅ | ❌ 无 buffer 状态 | **高** |
| Local/pooled 分离注意力 | ✅ | ❌ concat dense | **高** |
| mHC Metal kernel | ✅ 2 个 fused kernel | ❌ 未接入 | 中 |
| O-LoRA 量化 | ✅ | ❌ 仅 float matmul | 中 |
| 专家权重量化态 | ✅ | ❌ 全精度 | 中 |
| CustomMetalKernel | ✅ | ✅ | 无 |
| `mlx_gather_mm` 绑定 | N/A | ✅ `ops.zig:318` | 无 |
| Attention Sink | ✅ | ✅ | 无 |

---

## 第二部分：TurboQuant 与 FP8 分析（来自 turboquant-analysis.md）

> 基于 DeepSeek V4 技术报告（2026-04-24）和 TurboQuant 论文（arXiv:2504.19874），结合 mlx-zig 代码审计和 MXFP4/FP8 绑定实现。

### 1. 当前实现状态

| 任务 | 实现位置 | 状态 |
|------|---------|------|
| 15.2 可学习 softmax-gated pooling | `compressKV()` L1231-1430 | ✅ 完整实现 |
| 15.4 FP4 Lightning Indexer | `LightningIndexer` L648-930 | ✅ 完整实现 |
| 15.5 Attention Sink | `sink_logits` L959 | ✅ 已接入 fast SDPA |
| 15.3 FP8 KV 存储 | `kv_storage_dtype` L1055-1080 | ⚠️ 用 float16 代替 FP8 |
| 15.6 异构 KV cache | `compress_ratios` per-layer | ⚠️ KV cache strategy 仍统一 |

### 2. Task 15.3 — FP8 KV 存储：现在可用真正的 FP8

**论文要求：** V4 将大部分 KV 维度存储为 FP8（E4M3），仅 RoPE 维度保持 BF16。

**当前实现：** 用 `astype(float16)` 作为 FP8 代替。

**现在：** mlx-c 0.6.0 已有 `mlx_to_fp8`/`mlx_from_fp8`，新增了 `toFp8()`/`fromFp8()` 绑定。

**具体改动：**
```zig
// 当前：
const kv_nope_stored = if (kv_storage_dtype != .float32)
    try ops.astype(self.ctx, kv_nope, kv_storage_dtype)  // float16 代替
else kv_nope;

// 改为：
const kv_nope_stored = try ops.toFp8(self.ctx, kv_nope);  // 真正的 FP8
// 读取时：
const kv_nope_restored = try ops.fromFp8(self.ctx, kv_nope_stored, .bfloat16);
```

**优先级：** 🔴 高 — 改动小（约 10 行），效果大（内存减半 vs float16）。

### 3. Task 15.6 — 异构 KV Cache：缺少 per-layer strategy

**需要的改动：** 根据 `compress_ratios[i]` 为每层分配不同 cache：
- CSA 层（ratio=4）：cache 序列维度 = max_seq_len / 4 + window_size
- HCA 层（ratio=128）：cache 序列维度 = max_seq_len / 128 + window_size
- 无压缩层：cache 序列维度 = max_seq_len

**优先级：** 🟡 中 — 当前可以工作，优化后节省大量内存。

### 4. MXFP4 vs TurboQuant：不同场景的最优选择

| 维度 | 当前 affine | TurboQuant | MXFP4 |
|------|------------|------------|-------|
| 量化方式 | 均匀 per-group | Lloyd-Max 最优 | E2M1 per-block |
| 内积偏差 | 有偏 | 无偏（+QJL） | 有偏 |
| 理论保证 | 无 | ≈2.7x 最优 | 无 |
| 硬件加速 | Metal 原生 | 需自实现 | Metal 原生 |
| 适用场景 | 权重量化 | KV cache | 权重量化 |

**建议路径：**
1. 当前：已有的 affine 4-bit KV cache 量化
2. 近期：MXFP4 权重量化
3. 远期：TurboQuant KV cache 量化（Phase 4+）

### 5. TurboQuant Zig 实现方案

核心算法只需 4 个步骤，全部可用 mlx-c 算子实现：

```
1. 随机旋转：y = Π · x          → mlx_matmul（Π 预生成，QR 分解）
2. 标量量化：idx = nearest(y, codebook)  → mlx_argmin + 预计算 codebook
3. 反量化：  ỹ = codebook[idx]   → mlx_take
4. 逆旋转：  x̃ = Π^T · ỹ       → mlx_matmul
```

QJL 残差修正（无偏内积）：
```
5. 残差：    r = x - x̃          → mlx_subtract
6. QJL：     z = sign(S · r)     → mlx_matmul + mlx_sign
7. 重建：    x̂ = x̃ + √(π/2)/d · ‖r‖ · S^T · z
```

**存储开销：**
- Π 矩阵：d×d float32，d=128 时 64KB
- S 矩阵：同上
- Codebook：4-bit 时仅 64 bytes
- 量化后数据：b bits per coordinate（vs 16 bits float16）

### 6. 更新后的优先级排序

#### 立即可做
1. **Task 15.3 升级为真正的 FP8** — ~10 行，KV cache 内存减半

#### 近期
2. **Task 15.6 per-layer cache sizing** — 根据 `compress_ratio` 分配不同 buffer
3. **Task 15.1 Paged + Quantized** — `PagedKVCache` 内置量化选项

#### 远期（Phase 4+）
4. **MXFP4 权重量化集成**
5. **TurboQuant KV cache 量化**

### 7. tasks.md 修订建议

Task 15.3 更新：
```markdown
- [ ] 15.3 DeepSeek V4: upgrade FP8 KV storage to use native mlx_to_fp8/mlx_from_fp8
    - Current code at deepseek_v4.zig:1055-1080 uses astype(float16) as FP8 proxy
    - mlx-c 0.6.0 now has mlx_to_fp8/mlx_from_fp8, Zig bindings added in ops.zig
    - Replace astype(kv_storage_dtype) with ops.toFp8() for non-RoPE KV dimensions
    - Keep astype(.bfloat16) for RoPE dimensions
```

新增 Task 15.8：
```markdown
- [ ]* 15.8 TurboQuant KV cache quantization (optional, Phase 4+)
    - Implement Lloyd-Max codebook precomputation for b=1,2,3,4
    - Implement random rotation (QR decomposition via mlx linalg)
    - Implement QJL residual correction for unbiased inner products
    - _Reference: TurboQuant paper arXiv:2504.19874_
```

---

## 参考资料

- [DeepSeek V4 论文](https://arxiv.org/abs/2501.12948)
- [TurboQuant 论文](https://arxiv.org/abs/2504.19874)
- [MLX 文档](https://ml-explore.github.io/mlx/)
- [mlx-lm 仓库](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [修复与细节](DEEPSEEK-V4-FIXES-AND-DETAILS.md)
- [聊天分析与故障排查](DEEPSEEK-V4-CHAT-ANALYSIS-AND-TROUBLESHOOTING.md)
