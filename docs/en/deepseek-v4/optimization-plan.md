# DeepSeek V4 优化计划：对照 mlx-lm 官方实现

> 基于对 Apple 官方 `mlx-lm/mlx_lm/models/deepseek_v4.py`（2153 行完整实现）的深度审计，
> 以及 `dmlx` 实际能力（含 `CustomMetalKernel` 自定义 Metal kernel 支持），
> 制定的可执行优化路线图。
>
> **修订版 v3** — 审计日期：2026-04-27
> v2 修正了 v1 中的 8 类问题；v3 修正了 v2 中的 3 处事实性错误（gatherMm 已绑定、RoPE 已统一、HCA Indexer 条件已修复），
> 新增 2 项遗漏任务（RoPE GPU 化、O-LoRA batch matmul），删除 3 项已完成任务，时间线从 6 周压缩至 5 周。

---

## 一、背景

`mlx-lm` 于 2026-04 发布了官方的 `deepseek_v4.py`，这是目前最权威的 DeepSeek-V4 推理参考实现。该实现覆盖了：

- **CSA/HCA 混合注意力**：完整的 `Compressor` + `Indexer` + `_sparse_pooled_attention`
- **mHC 自定义 Metal Kernel**：`hc_split_sinkhorn` 和 `hc_sinkhorn_collapse` 两个内联 Shader
- **高效 MoE Dispatch**：`DeepseekV4SwitchGLU` 使用 `gather_mm` + `sort_threshold=8`
- **精巧的 Cache 设计**：`DeepseekV4Cache` 管理 `local` + `compressor_state` + `indexer_state`

本次审计发现 `dmlx` 在以下方面存在可优化的空间，按影响程度分为 P0-P3 四级。

---

## 二、关键架构差异（v2 新增）

在制定具体任务前，必须理解 `dmlx` 与 `mlx-lm` 在**核心模块设计**上的深层差异。这些差异不是简单的"实现缺失"，而是架构路径不同，需要针对性重构。

### 2.1 MoE 权重格式：Fused vs Split

| 维度 | mlx-lm | dmlx |
|------|--------|---------|
| **存储格式** | `switch_mlp.{gate,up,down}_proj.weight` 为 `[n_experts, out, in]`（fused） | loader 将 fused tensor **split 成 256 个独立专家**（`ffn.experts.{e}.{w1,w2,w3}`） |
| **Dispatch** | `mx.gather_mm` / `mx.gather_qmm` 一次性加载选中专家权重 | 遍历 256 个 `DSV4Expert`，mask 短路跳过无 token 的专家 |
| **量化态** | `QuantizedSwitchLinear` 保持 MXFP4/affine 打包，kernel 内反量化 | loader 中 `mlx_dequantize` 反量化为 float16/float32 |

**结论**：dmlx 的 MoE 是一条 `fused → split → dequantize → per-expert loop` 路径，与 mlx-lm 的 `fused → gather_mm` 路径完全相反。优化需要**反向重构**。

### 2.2 KV 压缩子系统：模块状态 vs 纯函数

| 维度 | mlx-lm | dmlx |
|------|--------|---------|
| **结构** | `Compressor` 是有状态的 `nn.Module`，含 `wkv`、`wgate`、`ape`、`norm` | `compressKV()` 是纯函数，权重从 caller 传入 |
| **Cache 集成** | `Compressor.__call__` 内部调用 `cache.accumulate_windows` | `compressKV()` 完全不知道 cache 存在 |
| **Overlap** | `_overlap_transform` 是成员方法，访问 `self.head_dim`、`self.overlap` | 无 overlap 逻辑 |

**结论**：dmlx 缺少 `Compressor` 模块和 `DeepseekV4Cache` 状态机，不是简单修改 `compressKV` 函数即可。

### 2.3 Indexer：两种不同架构

| 维度 | mlx-lm `Indexer` | dmlx `LightningIndexer` |
|------|-------------------|---------------------------|
| **Q 投影** | `wq_b: Linear(q_lora_rank → n_heads * head_dim)` | `wq_index: [index_n_heads * index_head_dim, head_dim]` |
| **K 来源** | **内部嵌套 `Compressor`** 生成 pooled | 直接接收外部 `k_compressed` |
| **打分方式** | `q @ pooled^T`，`max(0, score)`，乘 `weights_proj` | `q_index @ k_index^T`，scaled，mean over heads |
| **FP4** | ❌ 全程 bfloat16/float32 | ⚠️ 有 `quantize4bit` 模拟（应删除） |

**结论**：两个 Indexer 的权重形状、输入来源、打分逻辑**完全不同**。dmlx 的 `LightningIndexer` 更接近 DeepSeek V3 论文原始设计，而 mlx-lm 的是 Apple 的重新实现。需要明确选择对齐方向。

---

## 三、优先级总览

| 优先级 | 任务数 | 核心主题 |
|--------|--------|----------|
| **P0** | 3 | 正确性修复 + 最大性能瓶颈（含 Attention 双分支，从 P1 提升） |
| **P1** | 5 | 架构对齐 + 工程简化 + Custom Kernel + 新增 RoPE GPU 化 + O-LoRA batch matmul |
| **P2** | 2 | 量化保持 + 权重加载（~~绑定补齐~~ 已完成） |
| **P3** | 1 | 细节打磨 |

> **v3 变更**：P2.3（绑定 `mlx_gather_mm`）已删除（`ops.zig:318` 已有完整绑定）；
> P1.6（RoPE 逆变换简化）已删除（`apply()` 已有 `inverse: bool` 参数）；
> P0.3（HCA 移除 Indexer）已删除（loader 条件已为 `compress_ratio == 4`）；
> P1.2（Attention 双分支）提升至 P0（generate 正确性问题）；
> 新增 P1.7（RoPE GPU 化）和 P1.8（O-LoRA batch matmul）。

---

## 四、P0：正确性与核心性能瓶颈

> **建议工期：Week 1-2**

### 4.1 MoE Dispatch 反向重构：Gather/Sort + `gather_mm`

| 属性 | 内容 |
|------|------|
| **问题** | `DSV4MoE.forward()` 遍历全部 256 个专家（有 mask 短路：无 token 专家跳过，但仍 O(n_experts)）。更根本的是：loader 已将 fused `switch_mlp` split 为独立专家，丢失了 gather 的可能性 |
| **mlx-lm 参考** | `DeepseekV4SwitchGLU`（`deepseek_v4.py:125-155`），继承自 `SwitchGLU`（`switch_layers.py:160-199`） |
| **关键机制** | 1. `_gather_sort`：按专家 ID 对 token 排序（`sort_threshold = 8`，V4 覆盖几乎所有情况）<br>2. `mx.gather_mm` / `mx.gather_qmm`：只加载选中专家权重（fused `[n_experts, out, in]` 格式）<br>3. `scores` 在 `down_proj` **之前**乘入，减少一次 reduce<br>4. `_scatter_unsort` 恢复原始 token 顺序 |
| **前置依赖** | ~~`mlx_gather_mm` Zig 绑定缺失~~（**v3 修正：`ops.zig:318` 已有完整的 `pub fn gatherMm` 绑定，无前置依赖**） |
| **dmlx 行动** | 1. ~~绑定 `mlx_gather_mm`~~（已完成）<br>2. **停止 splitFusedExperts**：loader 中保留 `switch_mlp.weight` 为 `[n_experts, out, in]` fused 格式<br>3. **重构 DSV4MoE**：删除 `DSV4Expert` 数组，改用 fused 权重 + `gatherMm`/`gatherQmm`<br>4. **接入 sort**：路由后按专家 ID `argsort` token，用 `sorted_indices=true` 调用 gather<br>5. **scores 前置融合**：在 down-proj 前乘入 |
| **预期收益** | **10-100x** 专家计算提速，消除冗余 kernel launch，减少显存碎片 |
| **相关文件** | `src/models/deepseek_v4.zig:638-777`, `src/models/deepseek_v4_loader.zig:495-536`, `src/quantize.zig:323-358`, `src/ops.zig` |

### 4.2 KV 压缩子系统重构：`Compressor` + `DeepseekV4Cache`

| 属性 | 内容 |
|------|------|
| **问题** | `compressKV` 是纯函数，无状态管理，导致两个问题：<br>1. **无 `_overlap_transform`**：CSA (ratio=4) 层丢失前后块重叠，压缩块只聚合 `m` 个 token 而非论文要求的 `2m`<br>2. **无 `accumulate_windows`**：序列尾部 token 被直接丢弃（`num_groups = prefix_len / compress_ratio`，remainder 未缓冲），且 generate 阶段 L=1 无法正确累积 |
| **mlx-lm 参考** | `Compressor` 模块（`deepseek_v4.py:1481-1551`）、`DeepseekV4Cache.accumulate_windows`（`deepseek_v4.py:1028-1126`） |
| **关键机制** | 1. `Compressor` 作为有状态模块，内部调用 `cache.accumulate_windows(kv, gate, state_key, ratio, start_pos)`<br>2. `_overlap_transform`：CSA 时 `concat([prev_first, second_half], axis=2)`，gate 用 `-inf` fill<br>3. `accumulate_windows`：维护 `buffer_kv` / `buffer_gate`，不足 `compress_ratio` 的尾部保留到下一次 forward；支持变长 batch（`lengths` + `right_padding`） |
| **dmlx 行动** | 1. **引入 `Compressor` 结构体**：封装 `wkv`、`wgate`、`ape`、`norm`、`overlap` 标志<br>2. **引入 `DeepseekV4Cache`**：扩展或新增 cache 类型，管理 `local` + `compressor_state` + `indexer_state`<br>3. **实现 `_overlap_transform`**：在 `Compressor` 中增加 overlap 分支（仅 `compress_ratio == 4`）<br>4. **实现 `accumulate_windows`**：在 `DeepseekV4Cache` 中维护 buffer 状态 |
| **预期收益** | **CSA 层正确性修复**（否则注意力上下文丢失一半），支持任意长度输入和 generate 阶段正确累积 |
| **相关文件** | `src/models/deepseek_v4.zig:1367-1441`（compressKV）, `src/kvcache.zig`, `src/models/deepseek_v4.zig:1117-1234`（Attention forward） |

### ~~4.3 HCA 层（ratio=128）移除 Lightning Indexer~~ ✅ 已完成（v3）

| 属性 | 内容 |
|------|------|
| **问题** | ~~dmlx 的 loader 对所有 `compress_ratio > 1` 层创建 `LightningIndexer`~~ |
| **v3 状态** | **已修复**。`deepseek_v4_loader.zig:935` 条件已为 `compress_ratio == 4`，仅 CSA 层创建 Indexer。无需额外操作 |

### 4.4 Attention 分离 Local / Pooled 计算（L==1 与 L>1 双分支）（v3 从 P1 提升）

| 属性 | 内容 |
|------|------|
| **问题** | dmlx 当前对所有 CSA 层统一走 `gatherBlocks → concat → dense SDPA`，未区分 generate（L=1）和 prefill（L>1）。**这是正确性问题**：generate 阶段 L=1 时 compressed blocks 的 gather 语义不同（应用上次 prefill 的 index 结果） |
| **mlx-lm 参考** | `V4Attention.__call__`（`deepseek_v4.py:1769-1794`） |
| **关键机制** | - **`L == 1`**（generate）：`take_along_axis` gather 选中的 pooled blocks，concat 后走标准 `scaled_dot_product_attention`<br>- **`L > 1`**（prefill）：走 `_sparse_pooled_attention`（`deepseek_v4.py:295-333`） |
| **dmlx 行动** | 1. 在 `DSV4Attention.forward` 中增加 `if (seq_len == 1)` gather + dense SDPA 分支<br>2. 实现 `seq_len > 1` 的 `_sparse_pooled_attention` 等效路径 |
| **预期收益** | 解码阶段正确性修复 + latency 降低，prefill 阶段 pooled 计算量减少 |
| **相关文件** | `src/models/deepseek_v4.zig:1113-1361` |

---

## 五、P1：架构对齐与工程简化

> **建议工期：Week 3-4**

### 5.1 删除 Indexer FP4 模拟 + 明确 Indexer 架构选择

| 属性 | 内容 |
|------|------|
| **问题** | `LightningIndexer.quantize4bit()`（`deepseek_v4.zig:905-986`）做 `quantize(mxfp4) → dequantize → float32 matmul`，既无加速也无精度优势。但更重要的是：dmlx 的 `LightningIndexer` 与 mlx-lm 的 `Indexer` **架构完全不同**（见 §2.3） |
| **mlx-lm 参考** | `Indexer.__call__`（`deepseek_v4.py:1567-1598`）全程 bfloat16/float32，使用内部 `Compressor` + `wq_b` 投影 + `weights_proj` 加权 |
| **dmlx 行动** | **方案 A（推荐）**：对齐 mlx-lm 的 Indexer 架构<br>1. 删除 `LightningIndexer`，改为 `Indexer` 结构体（嵌套 `Compressor`，`wq_b` 投影，`weights_proj`）<br>2. 删除 `quantize4bit`<br><br>**方案 B**：保留论文原始 Lightning Indexer 设计，仅删除 `quantize4bit`，改用 bfloat16 matmul |
| **预期收益** | 代码简化，精度提升，与参考实现对齐 |
| **相关文件** | `src/models/deepseek_v4.zig:794-1056` |

### 5.2 ~~Attention 分离 Local / Pooled 计算~~ → 已提升至 P0.4（v3）

### 5.3 mHC 接入 `CustomMetalKernel`

| 属性 | 内容 |
|------|------|
| **问题** | `DSV4HyperConn.pre`（`deepseek_v4.zig:1876-1893`）使用纯 Zig ops 实现 Sinkhorn 迭代（`sinkhornNormalize`，line 1713-1757），多次 kernel launch，无 SIMD 优化 |
| **mlx-lm 参考** | `_make_hc_sinkhorn_collapse_kernel`（`deepseek_v4.py:486-633`）和 `_make_hc_split_sinkhorn_kernel`（`deepseek_v4.py:365-448`） |
| **关键机制** | 单 Metal dispatch，使用 `simd_sum` 列归一化，`bfloat4` 向量化加载，FMA 链，branchless lanes |
| **前置条件** | Metal 3.1+（支持 `bfloat16_t` 和 `vec<bfloat16_t, 4>`） |
| **dmlx 行动** | 1. 将 mlx-lm 的 Metal source 字符串翻译为 Zig 字符串<br>2. 通过 `src/ops/custom_kernel.zig` 注册 `CustomMetalKernel`<br>3. 触发条件：`!training && hc_mult==4 && dtype==bfloat16 && gpu`<br>4. 模块加载时静态初始化 kernel（仿照 mlx-lm 的 `_hc_sinkhorn_collapse_kernel = _make_hc_sinkhorn_collapse_kernel()`）<br>5. Fallback 到现有 ops 路径 |
| **预期收益** | mHC 前向传播减少 80%+ kernel launch 开销，43 层累积效果显著 |
| **相关文件** | `src/models/deepseek_v4.zig:1861-1899`, `src/ops/custom_kernel.zig` |

### 5.4 Compressor 区分 CSA 与 HCA 策略

| 属性 | 内容 |
|------|------|
| **问题** | `compressKV` 对所有 `compress_ratio > 1` 统一使用 gated pooling，未区分 CSA (4x) 和 HCA (128x) 的不同特性 |
| **mlx-lm 参考** | `Compressor.__init__`（`deepseek_v4.py:1483-1493`）：`self.overlap = compress_ratio == 4`，`self.out_dim = head_dim * (2 if overlap else 1)` |
| **关键机制** | CSA (4x)：gated pool + overlap + Indexer，输出 `head_dim * 2`<br>HCA (128x)：gated pool，无 overlap，无 Indexer，输出 `head_dim` |
| **dmlx 行动** | 在 `Compressor` 中按 `compress_ratio` 值选择策略：128x 层简化计算（无 overlap，out_dim = head_dim） |
| **预期收益** | HCA 层减少无效计算，与论文架构严格对齐 |
| **相关文件** | `src/models/deepseek_v4.zig`（重构后的 Compressor） |

### 5.5 生成/预填充 Attention 分支中的 Sink 处理

| 属性 | 内容 |
|------|------|
| **问题** | dmlx 的 `sink_logits` 实现与 mlx-lm dense SDPA 路径**已经对齐**（都传给 `fast_scaled_dot_product_attention` 的 `sinks` 参数）。真正的差异在 `_sparse_pooled_attention` 路径：mlx-lm 手动将 sink concat 到 scores 前端（`deepseek_v4.py:319-324`），dmlx 没有 sparse_pooled 路径 |
| **dmlx 行动** | 在实现 `_sparse_pooled_attention` 等效路径时，显式将 sink 作为前缀 concat 到 `scores`（`scores = concat([sink_scores, local_scores, pooled_scores])`） |
| **预期收益** | sparse_pooled 路径与 mlx-lm 行为一致 |
| **相关文件** | `src/models/deepseek_v4.zig:1089-1091`, `src/ops/fast.zig:39-47` |

### 5.6 RoPE 逆变换简化

| 属性 | 内容 |
|------|------|
| **问题** | dmlx 中 `DSV4YarnRoPE` 可能有独立的 `apply` 和 `applyInverse` 方法，代码重复 |
| **mlx-lm 参考** | `_apply_partial_rope(..., inverse=True)`（`deepseek_v4.py:266-284`）复用同一函数 |
| **dmlx 行动** | 将 `applyInverse` 合并到 `apply` 中，增加 `inverse: bool = false` 参数 |
| **预期收益** | 减少代码重复，降低维护成本 |
| **相关文件** | `src/models/deepseek_v4.zig`（RoPE 相关） |

---

## 六、P2：量化、加载与绑定补齐

> **建议工期：Week 5**

### 6.1 MoE 专家权重保持量化态

| 属性 | 内容 |
|------|------|
| **问题** | `loader.zig`（line 539-653）将所有量化的权重（含专家权重）`mlx_dequantize` 为 float16/float32。推理时专家权重是全精度 |
| **mlx-lm 参考** | `QuantizedSwitchLinear`（`switch_layers.py:27-90`）：权重保持 MXFP4/affine 打包，用 `mx.gather_qmm` 在 kernel 内反量化 |
| **dmlx 行动** | 1. 加载时保留 packed weight + scales（不 dequantize）<br>2. 专家矩阵乘改用 `gatherQmm`（`quantize.zig:324-358`）<br>3. 注意：这要求先完成 P0.1 的 fused 格式重构（不 split） |
| **预期收益** | MoE 专家权重内存降至 1/4-1/8，缓解显存压力 |
| **相关文件** | `src/models/deepseek_v4_loader.zig:539-653`, `src/quantize.zig:324-358` |

### 6.2 O-LoRA Grouped Projection 支持量化路径

| 属性 | 内容 |
|------|------|
| **问题** | dmlx 的 O-LoRA 手动实现 grouped matmul（`deepseek_v4.zig:1305-1358`），只有 float32/float16 路径，**无量化 matmul** |
| **mlx-lm 参考** | `V4Attention._grouped_output_projection`（`deepseek_v4.py:1680-1709`）：当 `wo_a` 是 `QuantizedLinear` 时，使用 `mx.quantized_matmul` |
| **dmlx 行动** | 在 grouped projection 中检测 `wo_a` 是否为量化权重，是则调用 `quantizedMatmul`（`quantize.zig:259-285`） |
| **预期收益** | 量化模型端到端推理性能提升 |
| **相关文件** | `src/models/deepseek_v4.zig:1305-1358` |

### 6.3 绑定 `mlx_gather_mm`（非量化 gather matmul）

| 属性 | 内容 |
|------|------|
| **问题** | `mlx-c` 提供 `mlx_gather_mm`（`ops.h:467`：非量化 gather matmul，支持 `lhs_indices`/`rhs_indices`/`sorted_indices`），但 dmlx 未绑定。当前只有 `gatherQmm`（量化版） |
| **dmlx 行动** | 在 `ops.zig` 或新模块中绑定 `mlx_gather_mm`：<br>```zig<br>pub fn gatherMm(ctx, a, b, lhs_indices, rhs_indices, sorted_indices) !Array<br>``` |
| **预期收益** | P0.1 MoE gather_mm dispatch 的前提条件 |
| **相关文件** | `src/ops.zig`, `src/c.zig` |

---

## 七、P3：细节打磨

> **建议工期：Week 6**

### 7.1 权重加载命名映射与格式对齐

| 属性 | 内容 |
|------|------|
| **问题** | mlx-lm `sanitize`（`deepseek_v4.py:1992-2127`）处理 FP4/FP8 自定义反量化（`dequant_fp4`、`dequant_fp8`）、命名 remap（`w1→gate_proj`）、以及 **独立 expert → fused `switch_mlp` 的 stack**。dmlx 的方向相反（fused → split） |
| **dmlx 行动** | 1. 确认 mlx-lm 量化格式（packed uint8 + scales 的特定布局）是否与 `mlx_dequantize` 兼容<br>2. 随着 P0.1 重构，loader 最终应**停止 split**，改为保留 fused 格式（与 mlx-lm 同向） |
| **预期收益** | 与 mlx-lm 权重格式 1:1 兼容，减少转换开销 |
| **相关文件** | `src/models/deepseek_v4_loader.zig:471-661` |

---

## 八、时间线

```text
Week 1-2 (P0) — 正确性与核心性能
├── Task 1: 绑定 mlx_gather_mm（P2.3 前置）
├── Task 2: MoE 反向重构（fused 格式 + gather_mm dispatch）
├── Task 3: KV 压缩子系统重构（Compressor + DeepseekV4Cache + overlap + accumulate_windows）
└── Task 4: HCA 移除 Indexer

Week 3-4 (P1) — 架构对齐与 Custom Kernel
├── Task 5: Indexer 架构选择 + 删除 FP4 模拟
├── Task 6: Attention 双分支（L==1 gather + L>1 sparse_pooled）
├── Task 7: mHC CustomMetalKernel（含 Metal bfloat16 前提检查）
├── Task 8: CSA/HCA 策略区分
├── Task 9: sparse_pooled 中的 Sink concat
└── Task 10: RoPE 逆变换简化

Week 5 (P2) — 量化与绑定补齐
├── Task 11: MoE 专家保持量化态（gather_qmm）
├── Task 12: O-LoRA 量化路径
└── Task 13: 权重加载 fused 格式对齐

Week 6 (P3) — 细节打磨
└── Task 14: 命名映射与格式兼容性验证
```

---

## 九、关键参考文件索引

| 文件 | 路径 | 说明 |
|------|------|------|
| `deepseek_v4.py` | `mlx-lm/mlx_lm/models/deepseek_v4.py` | 官方完整实现（2153 行） |
| `switch_layers.py` | `mlx-lm/mlx_lm/models/switch_layers.py` | MoE SwitchGLU / gather_mm / gather_qmm |
| `base.py` | `mlx-lm/mlx_lm/models/base.py` | `scaled_dot_product_attention`（sinks 参数） |
| `custom_kernel.zig` | `dmlx/src/ops/custom_kernel.zig` | dmlx CustomMetalKernel（完整 API） |
| `deepseek_v4.zig` | `dmlx/src/models/deepseek_v4.zig` | dmlx V4 实现（2332 行） |
| `deepseek_v4_loader.zig` | `dmlx/src/models/deepseek_v4_loader.zig` | 权重加载（1169 行） |
| `moe_router.zig` | `dmlx/src/moe_router.zig` | MoE 路由模块（未接入 DSV4MoE） |
| `quantize.zig` | `dmlx/src/quantize.zig` | 量化 / gather_qmm / 无 gather_mm |
| `ops.h` | `mlx-c/mlx/c/ops.h` | `mlx_gather_mm` / `mlx_gather_qmm` C API |

---

## 十、附录：mlx-lm vs dmlx 能力矩阵（v2 修订）

| 特性 | mlx-lm (Python) | dmlx | 差距 | 备注 |
|------|-----------------|---------|------|------|
| MoE gather_mm dispatch | ✅ `SwitchLinear` + `gather_mm` | ❌ split + per-expert loop | **高** | 需反向重构 + 绑定 gather_mm |
| CSA overlap transform | ✅ `Compressor._overlap_transform` | ❌ 纯函数 compressKV | **高** | 需引入 Compressor 模块 |
| Cache accumulate_windows | ✅ `DeepseekV4Cache` | ❌ 无 buffer 状态 | **高** | 需自定义 cache 类型 |
| HCA 无 Indexer | ✅ `compress_ratio==4` 才创建 | ❌ `compress_ratio>1` 创建 | 中 | 一行条件修改 |
| Indexer 架构 | ✅ Apple 重设计（wq_b + weights_proj） | ⚠️ 论文原始设计（wq/wk + dot） | 中 | 架构选择问题 |
| FP4 Indexer | ❌ 跳过 | ⚠️ simulate（应删除） | 低 | 删除即可 |
| Local/pooled 分离注意力 | ✅ `_sparse_pooled_attention` + L==1 分支 | ❌ concat dense | **高** | 需双分支实现 |
| mHC Metal kernel | ✅ 2 个 fused kernel | ❌（有 CustomMetalKernel 能力，未接入） | 中 | 需移植 shader |
| O-LoRA 量化 | ✅ `quantized_matmul` fallback | ❌ 仅 float matmul | 中 | 需检测 + 量化路径 |
| 专家权重量化态 | ✅ `QuantizedSwitchLinear` | ❌ 全精度 | 中 | 需保留 packed + gather_qmm |
| CustomMetalKernel | ✅ `mx.fast.metal_kernel` | ✅ `mlx_fast_metal_kernel` | 无 | 能力完整 |
| `mlx_gather_mm` 绑定 | N/A（Python 直接调用） | ❌ 未绑定 | 中 | P2.3 前置任务 |
| Attention Sink | ✅ `sinks` 参数 / 手动 concat | ✅ `sinks` 参数 | 无 | dense 路径已对齐 |
| RoPE 逆变换 | ✅ `inverse=True` 参数 | ⚠️ 可能独立方法 | 低 | 合并即可 |

---

## 十一、审计变更日志（v1 → v2）

| 变更项 | v1 问题 | v2 修正 |
|--------|---------|---------|
| P0 3.1 | 错误描述"256 次无条件 launch"；未提 gather_mm 绑定缺失 | 修正为有 mask 短路；增加绑定前置任务；指出 loader 需反向重构 |
| P0 3.2 | 仅提 "compressKV 增加 overlap" | 指出缺少 Compressor 模块和 DeepseekV4Cache；需子系统级重构 |
| P1 4.1 | 仅提"删除 FP4 模拟" | 增加 Indexer 架构差异说明；提供方案 A/B |
| P1 4.2 | 仅提"分离 local/pooled" | 增加 L==1 分支；区分 generate 与 prefill 路径 |
| P1 4.3 (原 P1 4.3) | 仅提"accumulate_windows" | 合并入 P0 3.2（Compressor + Cache 重构） |
| P3 6.1 (原 P3 6.1) | 误导性描述"sink_logits 需重构" | 指出 dense 路径已对齐；修正为 sparse_pooled 中的 sink concat |
| P2 5.2 (原 P2 5.2) | 建议"sanitize 对齐 + stack" | 指出 dmlx 方向与 mlx-lm 相反；改为"停止 split，保留 fused" |
| 新增 | — | §2 关键架构差异（fused vs split、Compressor 状态、Indexer 两种设计） |
| 新增 | — | P2.2 O-LoRA 量化路径 |
| 新增 | — | P2.3 绑定 `mlx_gather_mm` |
