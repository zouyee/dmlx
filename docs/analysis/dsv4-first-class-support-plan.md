# DeepSeek-V4-Flash-4bit 一等公民支持方案（Metal-First）

> **日期**: 2026-06-01
> **硬件**: Apple M4 Pro, 48GB
> **战略方向**: **native 为目标主路径**。性能达标且全链路数值对齐后，**废弃 MLX 推理路径**。
> **MLX 角色（过渡期）**: 数值对拍 oracle / 正确性参考，不再是长期主路径
> **首要目标**: `native` 端到端正确 (E2E 7/7) + 达到性能目标 (3.0 tok/s)
> **基线**: `origin/main` (`e3be289`，== `../dm/dmlx`)，纯 MLX 路径已验证输出正确（`The capital of France is` → `Paris`），作为对拍基准

---

## 0. 背景与决策依据

2026-06-01 的诊断（见 `flash-moe-alignment-plan.md` §1.5）确认：

- 之前一版未提交的「注意力 / RoPE 重写」是乱码根因，已 `git stash` (`stash@{0}`)
- 当前工作区已回到正确基线，仅额外恢复了独立安全的 `greedy.zig`（tokenizer BOS/EOS/PAD 健壮性）
- stash 内的「FP8 loader」对本模型 **完全无用**（见下方实测）

### ⚠️ 关键现状：HEAD 的 metal 路径注意力是占位实现

读 `src/metal_infer/engine.c` 的 `moe_infer_forward_layer` 确认：metal 路径**只有 MoE expert matmul 段是实质实现**，注意力是占位：

```c
// Simplified SDPA: single token self-attention = V (...)
// For now: use Q as attention output (simplified)
memcpy(attn_out, q, DIM * sizeof(float));   // 直接拿 Q 当注意力输出
```

即 metal 路径**缺失**：真正的 MLA SDPA、`wq_b`/`q_norm`/`wo_a`/`wo_b`、KV cache、`attn_sink`、mHC、最终 norm + LM head、`compressor`/`indexer`。
要让 metal-moe 替代 MLX，必须在 Metal/C 侧**完整实现 DSV4 的注意力 + RoPE + mHC + sink**。

### 可行性：ds4 有可移植的 V4 Metal kernel

`../ds4/metal/` 已有相对完整的 V4 kernel，显著降低从零实现的成本：

| 文件 | 内容 | 复用 |
|------|------|------|
| `dsv4_misc.metal` | MLA 混合注意力 (`kernel_dsv4_indexed_mixed_attention_heads8`) | ✅ 核心 |
| `dsv4_rope.metal` | V4 YaRN tail RoPE | ✅ |
| `dsv4_hc.metal` | mHC / HyperConnection | ✅ |
| `dsv4_kv.metal` | FP8 KV cache 量化 | ⬜ |
| `flash_attn.metal` | 多阶段 FlashAttention（prefill 加速） | ⬜ |
| `moe.metal` | MoE（含路由/combine） | ⬜ 对照 |
| `norm.metal` | RMSNorm | ✅ |

### 0.1 模型真实格式（实测 33 shards header）

| 项 | 值 |
|----|-----|
| safetensors `format` | `mlx` |
| dtype 分布 | BF16×1364, U32×641, F32×344, U8×129, I64×3 |
| FP8 / `F8_E4M3` / `F8_E8M0` 张量 | **0** |
| `*_scale_inv` 张量 | **0** |

### 0.2 量化结构（混合，来自 config.json `quantization`）

| 权重组 | mode | group_size | bits |
|--------|------|-----------|------|
| MoE experts (`switch_mlp.{gate,up,down}_proj`) | **mxfp4** | 32 | 4 |
| attn (`wq_a/wq_b/wkv/wo_a/wo_b`)、`embed`、`lm_head`、`shared_experts`、`compressor`、`indexer` | **affine** | 64 | 4 |

### 0.3 架构关键参数（config.json）

| 参数 | 值 | 备注 |
|------|-----|------|
| `num_hidden_layers` | 43 | |
| `num_attention_heads` | 64 | |
| `head_dim` | 512 | = qk_nope(448) + qk_rope(64) |
| `qk_rope_head_dim` | 64 | YaRN partial RoPE 仅作用于 tail 64 维 |
| `q_lora_rank` | 1024 | MLA：wq_a → q_norm → wq_b |
| `kv_lora_rank` | — | wkv 联合投影 |
| `n_routed_experts` | 256 | |
| `num_experts_per_tok` | 6 | K=6 |
| `n_shared_experts` | 1 | |
| `vocab_size` | 129280 | |
| `use_mhc` / `hc_mult` | true / 4 | mHC HyperConnection |
| `rms_norm_eps` / `hc_eps` | 1e-6 / 1e-6 | **二者独立，不可混用** |
| `attn_sink` | F32[64] per layer | 注意力 softmax 分母必须含 sink |
| `compressor` / `indexer` | 部分层存在 | CSA 压缩 + 稀疏索引（V4 高级特性） |

### 0.4 结论

> **本模型由 MLX 原生量化产出，HEAD 的 `loadShardedWeights` (`mlx_load_safetensors`) 路径已能正确加载。
> 不需要任何 FP8/E8M0 解量逻辑。** stash 中的 `dequantFp8Weights` 等改动不予恢复。

---

## 1. 设计原则（Metal-First）

战略是「metal-moe 成为主路径、达标后废弃 MLX」。但 MLX 当前是唯一已验证正确的实现，因此 **MLX 在过渡期降级为对拍 oracle，而非立即删除**。

### P1. MLX 是过渡期「正确性真值」，分阶段退役
- metal 每一段（注意力 / RoPE / mHC / MoE / norm / LM head）以「与 MLX 同 prompt 逐层对齐」为验收
- **废弃 MLX 的硬门槛**：metal 全链路 E2E 7/7 + 性能达标 (≥3.0 tok/s) + 连续多次 benchmark 稳定，三者同时满足后才下线 MLX 推理路径（loader 与对拍脚本保留）

### P2. 逐段替换，灰度推进（非一次性 big-bang）
- 在 MLX forward 逐层循环内，每段可独立选 metal/MLX；未实现段回落 MLX，避免「全 metal 但全错」无法定位
- per-segment 开关（`--metal-attn` / `--metal-mhc` / `--metal-moe`），逐段点亮、逐段对拍
- 每点亮一段，smoke 必须仍输出 `Paris`，否则回退该段

### P3. 数值对齐优先于性能
- 任一 metal 段先做到「与 MLX max diff < 阈值」，再谈 kernel 优化（SIMD / tiling / coalesce / FMA）
- 优化不得改变已对齐段的数值语义

### P4. 复用 ds4 kernel，不从零写
- MLA 注意力 / RoPE / mHC / norm 优先移植 `../ds4/metal/`，按 DSV4-Flash-4bit 维度适配
- 移植严守正确性契约：`attn_sink`、YaRN tail-only、`hc_eps`≠`rms_norm_eps`、grouped `wo_a`(8 组)

### P5. 加载与量化保持格式探测（兼容性）
- 量化 mode（mxfp4/affine）、group_size、bits 从 `config.json` 按权重名查询（已有 `dequantIfNeeded`）；metal dequant kernel 必须区分 experts(mxfp4/gs32) 与 attn(affine/gs64)
- 不引入对本模型无用的 FP8/E8M0 逻辑

---

## 2. 现状盘点（HEAD + greedy.zig）

| 能力 | MLX 路径 | Metal 路径 | 位置 |
|------|---------|-----------|------|
| 分片 safetensors 加载 | ✅ 正确 | 共用 MLX 加载 + 提取 f32 指针 | `loadShardedWeights` |
| affine / mxfp4 按名解量 | ✅ | ⚠️ 仅 experts mxfp4 | `dequantIfNeeded` / `engine.c` |
| MLA 注意力 | ✅ 正确 | ❌ **占位** `memcpy(attn_out, q)` | `DSV4Attention` / `engine.c` |
| YaRN partial RoPE (tail 64) | ✅ 正确 | ⚠️ 有 `apply_rope_tail`，未接正确 SDPA | `DSV4YarnRoPE` / `engine.c` |
| attn_sink | ✅ 正确 | ❌ 缺失 | `DSV4Attention` |
| mHC (hc_mult=4) | ✅ 正确 | ❌ 缺失 | `expandToMHC` / `engine.c` |
| MoE (256 exp, K=6, shared) | ✅ 正确 | ⚠️ expert matmul 有，combine/shared/数值未对齐 | `DSV4SwitchGLU` / `engine.c` |
| 最终 norm + LM head | ✅ 正确 | ❌ 缺失 | `engine.c` |
| SMELT 流式 expert 加载 | ✅ | ✅ I/O pool | `expert_stream.zig` / `engine.c` |
| tokenizer special-id | ✅ 已恢复 | — | `greedy.zig` |

> **结论**：metal 路径离「替代 MLX」尚缺整个注意力子系统 + mHC + 输出层。当前 `--metal-moe` 实测乱码（§1.5 诊断），与此一致。
> **基线性能**：MLX ~0.36–0.45 tok/s（cold），历史 warm ~1.0 tok/s。目标 3.0 tok/s。

---

## 3. 实施路线（Metal-First，逐段灰度）

> 总思路：在 MLX forward 的逐层循环里，让每段可独立选 metal/MLX，逐段点亮、逐段对拍。MLX 作为 oracle 保留到全链路达标。

### Phase 0 — 固化正确基线（已完成）

- [x] 诊断结论写入 `flash-moe-alignment-plan.md` §1.5
- [x] 恢复 `greedy.zig`（独立安全）
- [x] 本方案文档（metal-first）
- [x] commit：`3a018f2` (tokenizer fix) + `88a86a6` (docs)
- [x] smoke 脚本入库（`scripts/dsv4_smoke.sh`）

**验收**：✅ clean build + smoke 输出 `Paris`（且 `2+2=` 续写含 `4`）。

### Phase 1 — 对拍 oracle 基础设施（已完成）

没有逐层对拍，metal 段无法判断对错。先建护栏。

- [x] `scripts/dsv4_smoke.sh`：启动 serve → 2 续写型 prompt → 断言含 `Paris` / `4`
      - 注：该模型指令跟随弱但事实续写稳，故用续写 prompt + 足够 token（France=16, 2+2=64）
- [x] 逐层 activation dump：`src/models/activation_dump.zig`，由 **`DSV4_DUMP_DIR` 环境变量**门控（默认关，零开销）
      - 导出 `layer_00..42.npy` + `final_norm.npy` + `logits.npy`（float32 NumPy）
      - MLX 与 metal 两条路径共用 forward 循环里的同一 hook
- [x] 比对脚本 `scripts/compare_metal_mlx.py`：逐文件 max_abs / mean_abs / rel_L2，标出首个发散层
- [ ] （可选）用 `../transformers` golden 再校 MLX 自身，确认 oracle 可信

**验收**：✅ 一键得到「metal vs MLX 逐层偏差表」。用法：
```bash
DSV4_DUMP_DIR=/tmp/mlx_ref   PORT=8935            bash scripts/dsv4_smoke.sh
DSV4_DUMP_DIR=/tmp/metal_out PORT=8936 METAL_MOE=1 bash scripts/dsv4_smoke.sh
python3 scripts/compare_metal_mlx.py /tmp/mlx_ref /tmp/metal_out
```

### Phase 2 — 混合方案先行（plan c，已部分执行 2026-06-01）

> 决策：先做「MLX backbone + Metal 路由 expert」的混合方案，用最小改动拿真实性能/正确性数据，
> 再决定注意力是否值得搬到 Metal。理由：归档文档实测 **I/O 占每 token ~97%，GPU 计算仅 ~30%**，
> 注意力投影仅 ~1.2ms/层 —— 把注意力搬 Metal 对性能几乎无收益，瓶颈在 MoE expert I/O。

#### 2a. 接线修正（已完成）

- [x] 发现 `--metal-moe` 同时启用了**两条冲突路径**：
  - `server.zig`：`metal_moe.setEnabled(true)` → 正确的混合路径（`expert_stream.tryMetalPath`，
    MLX 跑 attention/mHC/shared/gate，Metal 只跑路由 expert）
  - `state.zig`：设 `model.metal_engine` → **engine.c 全层引擎**（占位注意力 `memcpy(q→attn_out)`），
    会短路 MLX `layer.forward`，导致 `tryMetalPath` 永不执行 + 注意力失效 → 乱码
- [x] 移除 `state.zig` 设 `metal_engine` 的代码块。`--metal-moe` 现在 = 纯混合路径（engine.c 全层引擎不再使用）

#### 2b. 混合方案实测与修复（2026-06-01）

初测乱码，逐层对拍定位到 **metal_moe.zig MoE kernel 两处 bug**，已修复：

1. **moe_combine 多加了 residual** —— `moe_metal_wrapper.c` 把 `hidden_buf`（MoE 输入）当 residual 加进
   combine 输出。但 `DSV4MoE.forward` 返回的是路由 expert 和 `y`，之后才 `+ shared_out` + block residual，
   导致输入被重复计入。改为绑定**零 residual buffer**，combine 只返回纯路由 expert 加权和。
2. **fused_gate_up_swiglu 缺 limited-SwiGLU 截断** —— MLX 用 `gate=min(g,10)`、`up=clamp(u,-10,10)`
   （swiglu_limit=10）后再 silu。kernel 原为无截断的 `silu(g)*u`。已加匹配截断。

| 项 | 修复前 | 修复后 |
|----|--------|--------|
| 正确性 | ❌ 乱码 | ✅ **France→`The capital of France is Paris.`**（匹配 MLX oracle），2+2= 连贯 |
| 性能 | ~4.5s/token | ~4.5s/token（未变，见 §Phase 4/5） |

> ⚠️ 对拍说明：metal-moe 模式下 **prefill 走 MLX batch**（每层 expert 数 >6，`tryMetalPath` 跳过回落 MLX），
> 仅 **decode 单 token** 才走 metal MoE kernel。故 `--max-tokens 1` 的逐层 dump 对拍显示全 0 偏差（都是 MLX prefill），
> 真正验证 metal kernel 正确性的是 **E2E decode 输出**（已匹配 `Paris`）。

#### 2c. 现状

- [x] metal-moe 混合路径**正确性达标**（E2E 输出匹配 MLX）
- [x] **性能瓶颈实测（2026-06-01）** —— 见 §2e，结论颠覆「I/O 是瓶颈」的旧假设
- [ ] decode 路径逐层对拍自动化（需构造 decode-only dump，当前 dump 主要覆盖 prefill）

#### 2e. 性能瓶颈实测 —— 瓶颈不是 I/O，是 MLX 同步屏障

decode 单 token 计时（warm cache，5 token 平均 ~3.7s/token）：

| 组件 | 耗时 | 占比 | 说明 |
|------|------|------|------|
| **other** | ~2400-2800ms | **~70%** | MLX backbone（attention/mHC/gate/shared expert）+ 每层同步 |
| io | ~460-940ms | ~20% | `readAndAssembleAll` SSD 读 + 数组组装（warm cache，读本身仅 ~1ms） |
| metal | ~400-760ms | ~12% | Metal MoE kernel dispatch（naive，含 6 expert 串行 + waitUntilCompleted） |

**关键结论（颠覆旧假设）**：

1. **I/O 不是瓶颈**（仅 ~20%，且 warm cache 下纯读 ~1ms）。归档文档「I/O 占 97%」是 cold-start 估算，
   warm 运行下不成立。→ **时序预测预取、双缓冲等 I/O 优化对当前瓶颈收益有限。**
2. **真正瓶颈是 "other" ~70%**：纯 MLX 路径 backbone 仅 ~0.5s/token，但混合路径 "other" 高达 ~2.6s。
   5x 劣化的原因：`tryMetalPath` 每层强制 `eval()` + `dataSlice` + `Array.fromData` + `reshape`，
   **把 MLX 的惰性图融合打断成 43 次同步屏障**（每层一次 materialize），丧失 MLX 的 lazy/fusion 优势。
3. metal kernel 本身（~12%）是第三位，naive kernel + 6 expert 串行 + 每次 `waitUntilCompleted` 同步。

**修正后的优化优先级（Phase 4/5）**：

- **P0：消除每层 MLX 同步屏障**（收益最大，~70%）。让 metal MoE 输出**作为 MLX lazy array 回插图**，
  而非 `eval()`+`fromData` 往返；或反过来，把整层（attn+moe）都搬出 MLX 同步点。
  这是当前 ~3.7s → 接近纯 MLX ~0.5s 的关键。
- **P1：metal kernel 优化**（~12%）：6 expert 并行 dispatch（单 command buffer 不 per-expert 等待）、
  SIMD reduction、threadgroup tiling、去掉 `waitUntilCompleted` 改 deferred。
- **P2：I/O**（~20%）：当前 warm 已不是瓶颈；cold-start 才需预取。优先级最低。

> ⚠️ 这说明「metal-first 终局」的性能前提存疑：只要 backbone 还在 MLX，每层 metal↔MLX 往返的同步屏障
> 就吃掉大部分时间。要么 **全 metal**（消除往返，回到 Phase 2d-5 的大工程），要么 **纯 MLX + MLX 量化 MoE**
> （根本不引入 metal 往返）。当前「MLX backbone + metal MoE」混合方案**性能上是最差组合**（两边的同步代价都付）。

#### 2d. 注意力搬 Metal（推迟）

> 仅当混合方案正确 + MoE 提速后仍需更快时再做。届时按下述移植 ds4 kernel：
> 权重接入（`wq_a/q_norm/wq_b/wkv/wo_a/wo_b/attn_sink`）→ Q 链 → wkv → tail RoPE →
> 含 sink 的 SDPA（`dsv4_misc.metal`）→ grouped `wo_a`→`wo_b`，逐层对拍。
> ⚠️ 注意 metal 路径收到的 hidden 是 mHC 展开后的 `[1,1,4,4096]`，engine.h 的注意力常量
> （N_HEADS=32/HEAD_DIM=128/KV_HEADS=8）是 GQA 占位，与 V4 MLA（64/512/1）不符，需重设计。


### Phase 3-5 — 全 Metal Layer（已选定，full-metal 终局）

> 决策（2026-06-01，§2e 实测后）：混合方案性能注定差（每层 MLX↔metal 同步屏障吃 ~70%）。
> 走 **全 metal**：整层（attn + mHC + MoE）都在 C/Metal 内完成，**一个 token 一次性跑完 43 层，零 MLX 往返**。
> MLX 仅保留为对拍 oracle，达标后退役。

#### 关键约束与已验证事实

1. **注意力权重必须 Metal 内 on-the-fly 解量**，禁止预解量到 f32（43 层 × ~400MB = ~17GB → 旧 OOM）。
2. **两种量化解量公式**（kernel 必须区分）：
   - **attn / embed / lm_head / shared（affine, gs=64）**：`w = scale_g * nibble + bias_g`
     （nibble = `(packed >> 4*i) & 0xF`，scale/bias 为 bf16 per-group）。MLX 源：`cpu/quantized.cpp:131`
   - **MoE experts（mxfp4, gs=32）**：`w = NIBBLE_TO_FLOAT[nibble] * exp2(scale_e8m0 - 128)`，无 bias（§10）
3. **真实 MLA 维度（layer 0，与 engine.h 现有 GQA 占位常量完全不符，需重写）**：

   | 权重 | 逻辑 shape | 量化 | 作用 |
   |------|-----------|------|------|
   | wq_a | [1024, 4096] | affine gs64 | hidden → q_lora(1024) |
   | q_norm | [1024] | bf16 | RMSNorm |
   | wq_b | [32768, 1024] | affine gs64 | q_lora → 64 head × 512 |
   | wkv | [512, 4096] | affine gs64 | hidden → kv_lora(512) |
   | kv_norm | [512] | bf16 | RMSNorm |
   | wo_a | [8, 1024, 4096] | affine gs64 | grouped：8 组，每组 4096→1024 |
   | wo_b | [4096, 8192] | affine gs64 | concat(8×1024)=8192 → hidden(4096) |
   | attn_sink | [64] | f32 | 每 head 一个 sink logit |

   配置：64 heads，head_dim=512（nope 448 + rope 64），1 KV head（MQA 广播），mHC mult=4。

#### 构建顺序（逐 kernel 对拍，每步 smoke 必须仍 `Paris`）

- [ ] **S0 脚手架**：重写 `engine.h` 权重模型为 MLA（删 GQA 占位常量），扩展 `extractWeightsForEngine`
      提取 attn 权重的**量化原始指针**（packed u32 + bf16 scales + bf16 biases，不解量）
- [ ] **S1 affine-dequant matvec kernel**：写 `dequant_matvec_affine`（`w=scale*q+bias`），
      **单测对拍** vs MLX `quantizedMatmul`（固定输入向量，max diff 阈值）——这是全 metal 的成败前提，先证它
- [ ] **S2 Q 链**：wq_a → RMSNorm(q_norm) → wq_b → reshape[64,512] → per-head RMSNorm → tail RoPE，对拍 Q
      - [x] 算法对拍（`scripts/verify_q_chain.py`）：kernel-style 交错 RoPE == mlx-style，max_abs=0；
            RMSNorm（带权 + weightless）vs MLX `fast.rms_norm` max_abs~1e-6；
            sanity 确认 split-half RoPE 会差 5.8（证明旧 engine.c 的 split-half 是错的）
      - [x] Metal kernels：`rms_norm_rows`（per-row，带/不带权重）、`rope_tail_interleaved`（交错对，YaRN tail）
      - [ ] host 编排 + 真实 Q dump 对拍（依赖 attn 权重提取，与 S3 一起接）
- [ ] **S3 KV**：wkv → RMSNorm(kv_norm) → [1,512] → tail RoPE；KV cache（43 层 × [seq,512]）
      - [x] 数值无新风险：wkv matvec=affine（S1 证）、kv_norm=learned RMSNorm（S2 证）、tail RoPE=交错（S2 证）；KV 单 head 512 维、无 per-head norm
      - [x] attn 权重提取（`extractWeightsForEngine`）：每层填 `AttnWeightPtrs`（packed u32 原始指针 + scales/biases bf16→f32 + q_norm/kv_norm/attn_sink）；astype'd f32 数组挂 `model.engine_f32_arrays` 存活
      - [ ] KV cache 分配 + host 编排（与 S4 一起接，届时 Q+KV 真实 dump 对拍）
- [ ] **S4 SDPA + sink**：移植 `dsv4_misc.metal`，MQA 广播 1→64，含 `attn_sink`，对拍 attn_out
      - [x] 算法对拍（`scripts/verify_sdpa_sink.py`）：kernel-style online-softmax + sink-fold vs MLX `fast.scaled_dot_product_attention(sinks=)`，max_abs=3e-8；sanity 确认 sink 改变输出 0.30
      - [x] sink 精确语义（MLX `fast.cpp`）：`scores=concat([sink_h, q·k·scale])` → softmax → 切掉 sink 列 ≡ 把 `exp(sink_h)` 计入 softmax 分母、不贡献输出
      - [x] Metal kernel `mla_sdpa_decode`：每 head 一 threadgroup，online-softmax，MQA 单 KV head 广播
      - [ ] host 编排（KV cache + 串 S2/S3/S4）+ 真实 attn_out dump 对拍（runtime 验证 kernel 语法+数值）
- [ ] **S5 输出投影**：grouped wo_a（8 组）→ inverse tail RoPE → concat → wo_b，对拍 attn 层输出
      - [x] 布局对拍（`scripts/verify_out_proj.py`）：简单标量索引（group=h//8，head-major flatten）vs MLX reshape/transpose 链，max_abs=0；完整 out-proj max_abs=0
      - [x] 无新 kernel：grouped wo_a = 8× `dequant_matvec_affine`（S1）、concat = host memcpy、wo_b = `dequant_matvec_affine`（S1）、inverse RoPE = `rope_tail_interleaved` inverse=1（S2）
      - [ ] host 编排（与 S7 整层一起接）
- [ ] **S6 mHC**：移植 `dsv4_hc.metal`（expand/compress/preNormFn），注意 `hc_eps`≠`rms_norm_eps`
      - [x] 算法对拍（`scripts/verify_mhc.py`）：sinkhorn（softmax→+eps→col-norm→[row,col]×19）max_abs=5e-8；mhcPost（`comb^T @ residual` + `post_mix*x`）max_abs=0；sanity 确认 sinkhorn 后行列和≈1（双随机）
      - [x] 关键发现：mHC 维度极小（HC=4，comb 是 4×4，mixes 12 宽），compute 量可忽略 → S7 可在 host CPU 上算 mHC 小算子，不必写 Metal kernel（省复杂度，对性能无影响）
      - [ ] host 编排（hc_fn/hc_scale/hc_base 提取 + pre/post + sinkhorn，CPU 实现，与 S7 一起接）
- [ ] **S7 整层串联**：attn + mHC + 已有 MoE kernel 在 engine.c 内跑完单层，**层间不回 MLX**，逐层对拍
      - [x] **Metal kernel runtime 验证**（最大 S7 风险已消除）：独立测试台 `scripts/metal_kernel_test.m`
            + `run_kernel_tests.sh`，runtime 编译 `moe_kernel.metal` 并对拍每个 kernel（~2s 反馈，
            不必 50s 起服务）。全部通过：`dequant_matvec_affine`=0、`rms_norm_rows`≤1e-7、
            `rope_tail_interleaved`=1.5e-8、`mla_sdpa_decode(+sink)`=0。Metal 语法/绑定/threadgroup 归约/
            online-softmax 全部在真实 GPU 上确认正确。
      - [x] host 编排：`mla_attention.m`（`mla_attention_decode`）串 Q链→KV链→SDPA+sink→逆RoPE→grouped wo_a→wo_b，全程 Metal 无 MLX 往返；mHC 小算子留 CPU（S6 决定）
      - [x] **独立 host 对拍 GO**：`gen_attn_golden.py` 生成 layer-0 真实权重+golden，`mla_attention_test.m` 跑完整注意力 vs golden → **rel_L2=1.9e-6**（~2s，不必起服务）。`run_mla_attention_test.sh` 可复跑
      - [x] mHC CPU 实现（`mhc.{h,c}`）：`mhc_pre`（preNormFn + scale/base 切分 + sinkhorn）/`mhc_post`（转置 comb）/`mhc_head_compress`，对拍 golden（`gen_mhc_golden.py` + `mhc_test.c`）全部 ≤6e-8
      - [x] 接入 engine.c `moe_infer_forward_layer`：完整 mHC-wrapped 全 metal layer
            （mhc_pre→input RMSNorm→`mla_attention_decode`→mhc_post→mhc_pre→ffn RMSNorm→gate→MoE→mhc_post），
            `hidden` 即 mHC 展开的 `[MHC_MULT,DIM]` 原地残差；KV cache 惰性分配；engine.c MoE combine 改零 residual
            （routed-only，与 metal_moe.zig 一致）；build.zig 加 `mla_attention.m`/`mhc.c`。编译通过，隔离测试仍 GO
      - [x] 接线完成：`extractWeightsForEngine` 填 attn(含 grouped wo_a 切片)+ hc 权重；`engine.zig` setWeights 调 `set_layer_attn`/`set_layer_hc`；`--metal-full` flag + `state.zig` 设 `metal_engine`；build.zig 编译 mla_attention.m/mhc.c
      - [x] 修复 #1：gate.weight 是 BF16，旧 `dataPtr(f32)` 2x 过读 → segfault。改 `keepF32`（astype f32）
      - [x] 修复 #2：mHC 残差 `expandToMHC` 是 broadcast view（stride-0），引擎读 16384 元素越界 → 改 host 侧 materialize 连续 f32 + 稳定 heap buffer
      - [x] **全 metal 跑通(无 crash)**：修了 4 个集成 bug — gate.weight bf16 过读、mHC broadcast 残差越界（改 mlx_contiguous 物化）、MLX↔C 指针（改稳定 host buffer）、**wo_a 是 dense bf16 不是 packed**（loader `dequantIfNeeded` 解掉了，wo_a_scales=null）→ 改 dense f32 + `matvec_f32`。warmup + 43 层全跑通
      - [ ] ⚠️ **正确性调试中(进展)**：发现并修了 3 个集成 bug — (a) `generate()` 对 metal-full 仍走 batch prefill（[1,9,...] 喂给单 token 引擎）→ 改 token-by-token；(b) MoE 读 `eng->buf_normed` 但新 RMSNorm 没写进去 → ffn_out 全 0，已修；(c) 引擎 input/ffn norm 换用已验证的 `rms_norm_rows`。MF_DBG 追踪：ffn_out 从 0 → 0.19（MoE 现在有输出）
      - [ ] 仍未对齐：layer_00 输出 norm 远小于 MLX（metal ~0.5 vs MLX ~533），层在衰减而非放大；`post_mix=[0.032,0,0,0]` 待与 MLX 逐项核对；shared expert 仍缺
      - [ ] 下一步：对比 MLX layer-0 的 mhc_pre 内部量（post/comb/attn_input），补 shared expert
- [ ] **S8 全 43 层 + final norm + lm_head**：engine 内跑完整 forward，E2E 对拍 logits → smoke `Paris`
- [ ] **S9 性能**：去同步屏障后测 tok/s；再做 kernel 优化（SIMD/tiling/coalesce）+ 6-expert 并行 dispatch
- [ ] **S10 达标门**：≥3.0 tok/s + 7/7 + 稳定 → 改默认 flag、退役 MLX 推理路径（保留 loader + 对拍）

**里程碑验收**：S1 对拍通过（affine dequant 可行）是 go/no-go 关卡；S8 smoke `Paris` 是正确性终点；S10 是性能终点。

---

## 4. 测试纪律（强制）

```bash
# Step 1: clean build
rm -rf .zig-cache zig-out && zig build -Doptimize=ReleaseFast

# Step 2: smoke (纯 MLX，正确性护栏)
./zig-out/bin/dmlx serve --model ~/models/DeepSeek-V4-Flash-4bit \
    --port 8930 --max-tokens 8 --temperature 0 \
    --smelt --smelt-strategy stream --smelt-experts 0.20 --smelt-cache 0 \
    --expert-packed-dir ~/models/DeepSeek-V4-Flash-4bit/packed_experts \
    > /tmp/dmlx_smoke.log 2>&1 &
while ! curl -sf http://localhost:8930/health >/dev/null 2>&1; do sleep 1; done
curl -s http://localhost:8930/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"default","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":8,"temperature":0}' \
    | python3 -c "import sys,json;print(repr(json.load(sys.stdin)['choices'][0]['message']['content']))"
# 期望包含 "Paris"
```

### 决策门

| smoke 结果 | 行动 |
|-----------|------|
| 含 `Paris` / 算术正确 | ✅ 可继续 / 可 benchmark |
| 乱码、重复退化 | ❌ 立即回退最近点亮的 metal 段 |
| BOS 重复 | ❌ 注意力 sink/mask 问题 |
| OOM / crash | ❌ 内存或加载问题 |

- 每点亮一个 metal 段（`--metal-attn` / `--metal-mhc` / `--metal-moe`），先跑该段开启、其余回落 MLX 的 smoke，再跑逐层对拍（§3 Phase 1）
- benchmark 仅在 smoke 通过后执行，7/7 才可 commit
- MLX 作为 oracle，对拍偏差超阈值即视为该 metal 段未对齐

---

## 5. stash 处置结论

| stash 内容 | 处置 | 理由 |
|-----------|------|------|
| 注意力 / RoPE 重写 (`deepseek_v4.zig`) | ❌ 永久丢弃 | 乱码根因；丢失 sink_logits + RoPE 布局翻转 |
| engine.c 掏空成 stub | ❌ 丢弃 | HEAD 版本更完整（含真实 MoE kernel） |
| FP8 loader (`dequantFp8Weights` 等) | ❌ 丢弃 | 本模型零 FP8/scale_inv，且耦合坏注意力的 hc_eps/norm_eps 字段 |
| `expert_preload.zig` remap -1 屏蔽 | ⬜ 待评估 | 与 SMELT 部分加载相关，需独立验证后再单独引入 |
| `greedy.zig` special-id | ✅ 已恢复 | 独立、安全、已验证 |

> stash 保留备查，不删除。未来若需要其中某个独立特性，**在干净 HEAD 上重做该特性**，禁止整体 `stash pop`。

---

## 6. 风险与对策

| 风险 | 对策 |
|------|------|
| **Metal 重写注意力再现乱码**（最大风险） | 逐段灰度 + 逐层对拍（Phase 1 oracle 先行）；任一段不达标立即回落 MLX |
| Metal 路径数值漂移难定位 | dump-activations 逐层 max diff，定位首个偏差层 |
| 移植 ds4 kernel 维度不匹配（ds4 与本模型 head/group 不同） | 移植前核对 head_dim=512 / qk_rope=64 / o_groups=8 / experts gs32 |
| 过早废弃 MLX 导致无对拍基准 | §1-P1 硬门槛：三条件齐备才下线 MLX |
| MLX↔Metal 互操作（eval 阻塞 command buffer） | metal 段在 MLX eval 后取 f32 指针，段间显式同步 |
| compressor/indexer（CSA/稀疏注意力）未实现 | 短序列 full_kv 已够用；长上下文再按 V4 论文补 stateful 压缩 |
| 内存：metal 注意力权重 dequant 到 f32 | 逐层惰性 dequant + 释放（参考 stash 思路，但在干净基线上重做） |

---

## 7. 工作量与现实预期

> ⚠️ Metal-first 是**大工程**。把已被 MLX 正确实现的 MLA + YaRN + mHC + sink + MoE 全部用 Metal/C 重写并逐层对齐，是数周级别的工作，注意力子系统（Phase 2）是最大难点。

- **短期可交付**：Phase 0（固化基线）+ Phase 1（对拍设施）——风险低、价值高，让后续每一步可验证
- **中期**：Phase 2-4 逐段点亮，每段都有 MLX 兜底，不影响可用性
- **里程碑**：任何阶段 MLX 路径始终可用（默认或 fallback），避免「为追 metal 把可用功能弄丢」
- **MLX 退役**：仅在 Phase 5 达标门全部满足后执行，且代码层面先改默认 flag、观察稳定后再删除

---

## 8. 参考实现对照（合并自 flash-moe-alignment-plan.md）

> 本节汇总两个外部参考引擎，作为 Phase 2-5 的移植索引。
> 行号为撰写时快照，移植前以实际文件为准。

### 8.1 ds4（DwarfStar，`../ds4/`）— 主参考

antirez 的 **DeepSeek-V4-Flash 专用自包含推理引擎**，目标模型与本项目一致，且经「官方 logits 对齐验证」。
**关键差异**：ds4 用 GGUF + 2-bit(IQ2) 量化；本项目用 MLX 原生 4bit safetensors（mxfp4 experts gs32 + affine attn gs64）。
→ **算法逻辑可直接对照/移植，但 dequant 必须替换为我们的 mxfp4/affine。**

| 文件 / 符号 | 内容 | 用途 |
|------------|------|------|
| `metal/dsv4_misc.metal` `kernel_dsv4_indexed_mixed_attention_heads8` | MLA 混合注意力核心（含 sink、window、raw+compressed KV、top-k 索引） | Phase 2 SDPA 主参考 |
| `metal/dsv4_rope.metal` `kernel_dsv4_rope_tail_f32` | V4 YaRN tail-only RoPE | Phase 2 RoPE |
| `metal/dsv4_hc.metal` | mHC / HyperConnection | Phase 3 mHC |
| `metal/norm.metal` | RMSNorm | Phase 2/3 |
| `metal/dsv4_kv.metal` `kernel_dsv4_fp8_kv_quantize_f32` | KV cache 量化 | Phase 2 KV（可选） |
| `metal/flash_attn.metal` | 多阶段 FlashAttention | prefill 加速（后期） |
| `metal/moe.metal` | MoE（路由 + combine） | Phase 4 对照 |
| `ds4_metal.m` `ds4_gpu_encode_rope_tail_inplace()` | RoPE 调度参考 | Phase 2 host 侧 |
| `ds4.c` | host 侧 forward 编排（层循环、KV 状态机） | 整体架构参考 |

> ds4 attention kernel 关键结构（`dsv4_attend_*`）：每 head 8 SIMD groups，threadgroup 缓存 KV 行，
> online-softmax（运行 M/S 累加），最后用 `dsv4_attend_sink(sinks[head], M, S, ...)` 把 sink 并入分母——
> 与我们 MLX 版 `scaledDotProductAttention(..., sink_logits)` 语义一致。**这是我们要复刻的正确算法。**

### 8.2 flash-moe（`../flash-moe/metal_infer/`）— 仅 MoE/IO 流水线参考

flash-moe 是 **Qwen GQA** 模型引擎，注意力与 V4 MLA 不同，**注意力部分对本项目无用**。
仅其 MoE expert 流水线与 I/O 设计可借鉴。

| 文件 / 符号 | 内容 | 价值 |
|------------|------|------|
| `infer.m:2124` `full_attention_forward()` | Qwen GQA 注意力 | ❌ 不适用（V4 用 MLA） |
| `shaders.metal` `dequant_matvec_4bit_v3` | 优化版 4bit matvec（tiling/coalesce/SIMD/FMA） | ⬜ Phase 5 kernel 优化参考 |
| `shaders.metal` `fused_gate_up_swiglu` | 融合 gate+up+SwiGLU | ⬜ Phase 4 MoE |
| `shaders.metal` `moe_combine_residual` | expert 加权和 + residual（K 硬编码，需改 K=6） | ⬜ Phase 4 |
| `shaders.metal` `rms_norm_sum_sq` / `rms_norm_apply_bf16` | 两段式 RMSNorm | ⬜ |
| `infer.m:3060` `async_pread_start/wait` | GCD 异步 I/O | ⬜ Phase 5 I/O |

---

## 9. flash-moe MoE/IO 流水线技术点（Phase 5 性能参考）

> 这些是 flash-moe 把 60 层 K=4 跑到 4.36 tok/s 的关键技术。Phase 5 性能优化时参考，
> **但必须在 Phase 2-4 全链路数值对齐之后**才动，且每步对拍。

### 9.1 三命令缓冲流水线（每层）

```
CMD3(N-1) deferred → CMD1: attention 投影
                   → CPU: 结果刷出
                   → CMD2: o_proj + norm + routing + shared expert
                   → CPU: softmax + topK 路由
                   → I/O: 并行 pread K experts
                   → CMD3: expert forward + GPU combine + norm (DEFERRED, 不等待)
```

- **CMD3 延迟提交**：`[cmd commit]` 后不 `waitUntilCompleted`，GPU 算 CMD3 时 CPU 已进入下一层
- **GPU-side combine**：CMD3 内 3 个 encoder 串联 —— `moe_combine_residual` + `rms_norm_sum_sq` +
  `rms_norm_apply`（用**下一层**的 norm weight），输出直接是下一层输入，**消除 CPU 往返**

### 9.2 时序 expert 预测 + 双缓冲

- token N-1 完成后存下每层 routing indices → 预测表
- token N 开始时用预测 indices 异步预取到 B 缓冲集；到该层时命中则零 I/O，未命中同步读 A 缓冲集
- flash-moe 命中率 ~71%（OS page cache 辅助）
- ⚠️ **V4 expert 局部性仅 ~35%**（实测），远低于 flash-moe 的 71%，故时序预测对 V4 收益有限，
  **不作为 V4 的主优化手段**

### 9.3 持久化 I/O 线程池

- N 个持久 pthread（每 expert 一个）+ generation counter + condition variable
- `io_pool_dispatch()` 填任务 → broadcast → wait
- HEAD 的 `engine.c` 已有等价实现（6 持久 pthread）

### 9.4 Metal kernel 优化要点（Phase 5）

- threadgroup tiling（如 8 rows/group，256 threads，8 SIMD groups）
- shared memory 缓存输入向量，256 线程协作加载
- coalesced 全局内存读取（SIMD lane stride 32）
- FMA 解量：`fma(nibble, scale*x, bias*x)` 一条指令完成解量+乘
- SIMD reduction：`simd_sum(acc)`
- ⚠️ 历史教训：SIMD reduction 版曾有 bug（87-97% 输出为 0），优化版必须逐 kernel 对拍

---

## 10. MXFP4 解量公式（已验证）

MoE experts 走 mxfp4（group_size=32）。解量公式（Python 端与 MLX `quantized_matmul` 实测 max diff 0.000000）：

```
NIBBLE_TO_FLOAT[16] = {0,1,2,3,4,6,8,12, -0,-1,-2,-3,-4,-6,-8,-12}
w = NIBBLE_TO_FLOAT[nibble] * exp2(scale - 128.0)
```

attn / embed / lm_head / shared_experts 走 affine（group_size=64），用标准 `w = q * scale + bias`。
metal dequant kernel 必须按权重名区分这两种 mode（见 §1-P5）。

---

## 11. 性能预估（参考，来自 flash-moe 类比）

flash-moe 实测 4.36 tok/s（60 层 K=4）。V4 43 层 K=6 的理论拆解：

| 阶段 | 耗时/layer | 43 层 |
|------|-----------|-------|
| attention 投影 | ~1.2ms | ~52ms |
| CPU attention | ~0.5ms | ~22ms |
| o_proj + norm + routing | ~0.55ms | ~24ms |
| I/O pread（预测命中时） | ~0.5ms | ~22ms |
| expert forward (deferred) | ~0.04ms | ~2ms |
| **合计** | ~2.8ms | **~120ms** |

理论 ~8 tok/s，保守含 overhead **3-5 tok/s** → 与目标 3.0 tok/s 吻合。
瓶颈仍是 SSD I/O：每 token ~3.35GB（43×6×13.4MB），SSD 有效带宽 ~1GB/s。
I/O 优化关键：降低有效 I/O 量（部分加载 SMELT + 缓存），而非单纯靠时序预测（V4 局部性低）。

---

## 12. 阶段性总结（2026-06-01）

> 本节记录从「混合方案乱码」到「全 metal 43 层跑通」的完整过程，作为后续调试的参考基线。

### 12.1 起点与战略转向

**起点**：`--metal-moe`（MLX backbone + Metal 路由 expert）输出乱码，逐层对拍 layer_00 rel_L2≈1.9。

**根因**（2026-06-01 诊断）：两个 MoE kernel bug——combine 多加了 residual（double-count）+ fused_gate_up_swiglu 缺 limited-SwiGLU 截断。修复后混合方案输出 `Paris`，但性能 ~4.5s/token（比纯 MLX 慢 8x）。

**性能测量**（decode 单 token 分解）：
- other（MLX backbone + 每层同步屏障）~70%
- io（expert SSD 读）~20%
- metal kernel ~12%

**战略转向**：混合方案每层 MLX↔Metal 往返是最差组合。选择**全 metal**（消除往返）。

### 12.2 S0-S6：算法验证阶段（全部 GO）

| 步骤 | 验证内容 | 误差 | 关键发现 |
|------|---------|------|---------|
| S0 | engine.h MLA 权重模型重写 | — | 旧 GQA 占位(N_HEADS=32/HEAD_DIM=128)全错 |
| S1 | affine 4bit dequant `w=scale*q+bias` | **0** | 可行，绕开 17GB 预解量 OOM |
| S2 | Q 链：交错 RoPE + 两种 RMSNorm | ≤1e-6 | split-half RoPE 差 5.8（旧 engine.c 是错的）|
| S3 | KV 链 + attn 权重提取 | 复用 S1/S2 | wo_a 是 dense bf16（loader 解量），不是 packed |
| S4 | SDPA + sink | 3e-8 | sink 精确语义：`exp(sink_h)` 计入分母不贡献输出 |
| S5 | grouped wo_a 布局 | **0** | 简单 head-major flatten == MLX 复杂 transpose |
| S6 | mHC sinkhorn + post | ≤6e-8 | HC=4 compute 可忽略，CPU 实现即可 |

**方法论**：每步用真实 layer-0 权重 + numpy/MLX 对拍，sanity check 证明"做错会差多少"。

### 12.3 S7：系统集成阶段（进行中）

**S7a** kernel runtime 验证（~2s 测试台）：全部 GPU 上通过，Metal 语法/绑定/threadgroup 归约确认正确。

**S7b** 完整注意力 host 编排（`mla_attention_decode`）：独立对拍 rel_L2=1.9e-6，~2s 验证。

**S7c** engine.c 全层串联：mhc_pre→attn→mhc_post→mhc_pre→MoE→mhc_post，编译通过。

**S7d** 真服务集成——修了 **7 个集成 bug**（按发现顺序）：

| # | Bug | 现象 | 修复 |
|---|-----|------|------|
| 1 | gate.weight 是 BF16 | segfault（2x 过读） | `keepF32` |
| 2 | mHC broadcast view（stride-0） | segfault（16384 越界） | `mlx_contiguous` + size 断言 |
| 3 | MLX buffer 指针不能交给 C | segfault（GPU 地址 CPU 读） | 稳定 host buffer + memcpy |
| 4 | wo_a 是 dense bf16（loader 解量） | segfault（scale ptr=0x4） | dense f32 + `matvec_f32` |
| 5 | `generate()` batch prefill | 9 token 喂单 token 引擎 → 全乱 | metal-full 时 token-by-token |
| 6 | MoE 读 `buf_normed` 但未写入 | ffn_out 恒为 0 | 写入 `buf_normed` 再 dispatch |
| 7 | 引擎 input/ffn norm 未验证 | 潜在精度问题 | 换用已验证 `rms_norm_rows` |

**当前状态（2026-06-01）**：

| 指标 | 值 |
|------|-----|
| 全 metal 43 层 | ✅ 跑通，无 crash |
| MoE 有输出 | ✅ ffn_out norm=0.19（修 bug#6 后） |
| E2E 正确性 | ❌ 乱码；layer_00 norm ~0.5 vs MLX ~533 |
| 性能 | ~1.7s/token（比混合方案快 2x，但仍慢） |
| 护栏 | ✅ 隔离测试全 GO；纯 MLX smoke `Paris` |

### 12.4 剩余工作（S7d 继续 → S8-S10）

**S7d 正确性（当前阻塞）**：

1. **layer_00 严重衰减**（norm 0.5 vs 533）：`MF_DBG` 追踪显示 `post_mix=[0.032,0,0,0]`，需与 MLX layer-0 mhc_pre 内部量逐项对比，定位是 mhc_pre 计算错还是权重传递错。
2. **shared expert 缺失**：每层 MoE 少了共享专家的贡献（`n_shared_experts=1`）。
3. **generateWithCallback 未加 token-by-token 保护**（HTTP 服务路径）。

**S8-S10**：全 43 层 E2E 对齐 → smoke `Paris` → 性能优化 → 退役 MLX。

### 12.5 调试工具清单

| 工具 | 用途 |
|------|------|
| `bash scripts/run_kernel_tests.sh` | ~2s Metal kernel 正确性（每次改 kernel 必跑） |
| `bash scripts/run_mla_attention_test.sh` | ~2s 完整注意力 host 对拍 |
| `python3 scripts/compare_metal_mlx.py ref/ cmp/` | 逐层 max_abs/rel_L2，标出首个发散层 |
| `DSV4_DUMP_DIR=/tmp/x bash scripts/dsv4_smoke.sh` | 生成逐层激活 dump |
| `MF_DBG=1 --metal-full` | 引擎内部逐段 norm 追踪（layer 0） |
| `bash scripts/dsv4_smoke.sh` | 端到端正确性门（Paris + 4） |

---

## 13. 阶段性总结（2026-06-02，commit 5639f94）

### 13.1 本轮调查目标

解决 hash routing 崩溃 + 验证 attention 正确性 + 定位真实误差来源。

### 13.2 修复项（全部已提交）

| # | Bug | 现象 | 修复 |
|---|-----|------|------|
| 8 | `tid2eid` 指针未对齐 | SIGBUS（MLX `dataPtr(i64)` 返回非 8 字节对齐指针） | 新增 `keepI64()`，将数据 copy 到 Zig allocator 分配的对齐 buffer |
| 9 | `EngineWeights` 未初始化 nullable 数组 | Debug 构建下 `tid2eid/gate_biases/attn` 等字段为 `0xaa`，被 C engine 误判为非 null | `extractWeightsForEngine` 显式初始化所有 nullable 数组 |
| 10 | `moe_infer_reset_kv` 只清 `len` 不清数据 | 多请求间旧 KV 数据可能污染（当前未必触发，防御性修复） | `memset(kv_cache, 0, ...)` |
| 11 | Hash routing 时 Zig debug 构建 `tid2eid` 为 `0xaa` | E2E 输出乱码（非 hash 路由时偶然正确是因为 0xaa 作 float 是极小负数，等效于不加 bias） | bug #9 修复 |

### 13.3 关键发现：误差归因

通过精确的逐步对比（同一 token、同一 position）确认：

| 组件 | 误差 | 状态 |
|------|------|------|
| `mla_attention_decode` 单步 | rel_L2=1.9e-6 | ✅ 正确 |
| `mla_attention_decode` 多步（8 prefill + decode） | rel_L2=2.2e-6 | ✅ 正确 |
| attn_normed（attention RMSNorm 输入）| rel_L2=0.26%, cosine=0.9999 | ✅ 正确 |
| attn_out（attention 输出）| rel_L2=0.67%, cosine=0.9999 | ✅ 正确 |
| mhc_post residual（attn 后 mHC 残差）| norm 8.895 vs 8.894 | ✅ 正确 |
| **MoE ffn_out**（MoE layer 输出）| norm 29.3 vs 30.1（~3% 差） | ⚠️ **误差来源** |
| layer_00 hidden state（整层后） | rel_L2=17%, cosine=0.986 | ⚠️ 累积中 |

**结论**：注意力（mla_attention_decode）、mHC（mhc_pre/post）均已正确。  
**唯一误差来源：MoE 子层（~3% per layer），43 层累积后 → 最终输出严重偏差**。

### 13.4 Hash routing 状态

- 数据正确性：已验证（C engine 查找结果与 safetensors 表完全一致）
- 当前状态：**暂时禁用**（`use_hash_routing = false`）
- 原因：在 MoE 误差未修复前，hash routing 使结果更差（正确的 expert 选择 + 错误的注意力 = 更坏的组合）
- 重启条件：待 MoE 对齐后再开启，预期能进一步改善正确性

### 13.5 下一步：MoE 误差根因

MoE 子层的 ~3% per-layer 误差来自哪里？候选：

1. **fused_gate_up_swiglu 截断（SwiGLU limit）**：MLX 用 `gate=min(g,10)`, `up=clamp(u,-10,10)` 后再 silu，而 Metal kernel 的截断逻辑是否完全一致？需对拍 `L0_ffn_input`（MoE 的 normed 输入）→ `swiglu output` → `down_proj output`。
2. **shared expert 的 SwiGLU limit**：shared expert 的 gate/up 截断是否与路由 expert 一致？
3. **moe_combine 权重精度**：`expert_weights` 的归一化 + scale 是否与 MLX 完全一致（sqrtsoftplus + L1-norm + ×1.5）？
4. **mxfp4 dequant 精度**：routed expert 用 mxfp4（`NIBBLE_TO_FLOAT * exp2(scale-128)`），是否与 MLX 的 `quantizedMatmul` 完全一致？

**优先动作**：在 `engine.c` 里 dump layer-0 MoE 的 `normed`（ffn input）、`ffn_out`（before mhc_post），与 MLX 的 `L0_ffn_out.npy` 对比，缩小范围。

### 13.6 调试工具更新

| 工具 | 用途 | 状态 |
|------|------|------|
| `bash scripts/run_mla_attention_test.sh` | 单步注意力对拍（rel_L2=1.9e-6） | ✅ 已更新（wo_a_dense 接口） |
| `python3 scripts/gen_multistep_golden.py` | 多步 KV cache golden 生成 | ✅ 新增 |
| `/tmp/mat_ms` | 多步 KV cache 对拍（rel_L2=2.2e-6） | ✅ 新增 |
| `MF_DBG=1 --metal-full` | 各段 norm 追踪；需仔细区分 dump 是哪次 forward | ✅ 可用 |
| `DSV4_DUMP_DIR` 逐层 dump | 注意：需用**同一 token 同一 position** 对比才有效 | ✅ 可用（方法确立）|

---

## 14. 深度精度对齐调查（2026-06-02，commit 47345a8）

### 14.1 调查目标

从 commit `1c1af0b` 已知 metal-full 有 76% logits 误差，根因是 f32 vs bf16 精度路径不同。本次调查尝试了多种精度对齐方案，全部失败，但得出了系统性结论。

### 14.2 每层误差分析

通过 `DSV4_DUMP_DIR` 逐层对比（metal vs MLX，同一 token 同一 position）：

| 层 | rel_L2 | 说明 |
|----|--------|------|
| L00 | 0.17 | MoE expert swap 贡献 17% |
| L01 | 0.21 | 累积 |
| L02 | 0.27 | 累积 |
| **L03** | **0.44** | **暴增！score-based routing 首层放大误差** |
| L10 | 0.99 | 接近随机 |
| L12+ | ~1.0 | 完全发散，方向随机 |

**关键发现**：L02→L03 误差从 0.27 跳到 0.44（+63%）。L03 是第一个 score-based routing 层，验证了 score-based routing 是主要误差放大器。

### 14.3 Layer-0 误差链

精确定位每个子步骤的误差：

| 步骤 | 相对于 MLX 的误差 |
|------|---------------|
| 输入残差（来自 embedding） | **0%**（bf16→f32 精确转换）|
| attn_input（mhc_pre 输出） | ~0.5% |
| attn normed（RMSNorm 后） | ~0.26% |
| attn_out（attention 输出） | **0.67%** |
| ffn_input（第二次 mhc_pre） | ~0.5% |
| ffn_normed（RMSNorm 后） | **0.51%** |
| **expert selection** | **5/6 匹配**（expert 90 vs 130，差值 0.001） |
| MoE ffn_out | **18%**（expert swap + 权重差异） |
| layer-0 output | 17% |

### 14.4 尝试的修复方案（全部失败）

| 方案 | 结果 | 原因 |
|------|------|------|
| bf16 truncation of `residual` after mhc_post | 更差 | 扰乱后续 mhc_pre |
| bf16 truncation of `attn_out` | 无效 | 误差源不在此 |
| bf16 truncation of `attn_input` (mhc_pre output) | 更差 | 引入额外截断误差 |
| bf16 truncation of KV cache | 更差 | 破坏 KV 精度 |
| bf16 Q chain（wq_a/wq_b/q_norm → bf16 output） | **无效（1.00x）** | attn_input 本身是 f32，不匹配 |
| hash routing enabled | L0-1 改善，L4+ 更差 | 混沌系统路径依赖 |
| normed bf16 truncation before gate | 无效 | normed 差异 0.51% 导致 expert swap 不论是否截断 |

### 14.5 根本原因（最终定论）

f32 Metal 路径与 bf16 MLX 训练路径在 43 层后的完全发散，是**系统性问题，非简单 bug**：

1. **每个 bf16 操作在不同时刻截断**（matmul 后、激活后、写入 tensor 时），这些截断是确定性的但顺序相关的，无法通过简单的"在某点截断"来重现。
2. **0.51% ffn_normed 误差**导致 expert 90（Metal）vs 130（MLX）的选择 flip，两者分数差仅 0.001。
3. **18% MoE 误差 = expert swap（~7.6%）+ 5个共同 expert 的权重差异（~10.4%）**。
4. 每层 ~17% 的误差在 43 层后通过非线性（SwiGLU、softmax）放大成完全随机。

### 14.6 新增基础设施

在 `moe_kernel.metal` 里新增三个 bf16-output kernel（Metal 3.1+ `bfloat` 类型）：
- `dequant_matvec_affine_bf16out`：affine matmul → bf16 输出
- `rms_norm_rows_bf16out`：RMSNorm → bf16 输出
- `bf16_to_f32`：bfloat → float 转换

这些 kernel 已通过 runtime 编译验证，可用于未来完整 bf16 计算链实现。

### 14.7 正确修复路径

要使 metal-full 输出与 MLX 对齐，必须**全链路使用 bf16**：

1. **`mhc.c` 改为 bf16 运算**：`mhc_pre` 的 `fn @ residual_flat` 计算输出 bf16，`attn_input` 以 bf16 传给 attention kernel
2. **attention kernel 全程 bf16**：Q/KV chain 的所有中间结果 → bf16
3. **KV cache 以 bf16 存储**：每个 token 写入 KV cache 时截断为 bf16
4. **MoE normed bf16**：ffn_normed 传给 gate matmul 时以 bf16，确保 expert selection 匹配

完整 bf16 对齐预期能使 expert selection 6/6 匹配，每层误差从 17% 降到 ~0.1%，43 层不再发散。

**这是正确的 S7d 修复路径**，工作量约 2-3 天（主要是 mhc.c + mla_attention.m 的 bf16 数据流）。

### 14.8 现实评估

如果不做 bf16 对齐：
- metal-full 输出在 12 层后完全发散，无法通过 smoke test
- 但 --metal-moe（混合路径）仍然正确（Paris ✓）

优先级建议：先完成 bf16 mhc + attention 数据流，再重测 metal-full。

---

## 15. 最终结论：metal-full 精度对齐（2026-06-02，commit bc21ec3）

### 15.1 尝试了什么（全部失败）

经过系统性的尝试，所有精度对齐方案均失败：

| 方案 | 结果 | 失败原因 |
|------|------|---------|
| bf16 Q chain（wq_a/wq_b/q_norm 输出 bf16） | 无效（1.00x） | attn_input 本身是 f32，Q chain 输入不匹配 |
| attn_input bf16 截断 | **灾难性（1.0+）** | attention 对输入极度敏感，微小截断导致 Q/K 完全错误 |
| ffn_input bf16 截断 | 更差（~20%+） | 同上 |
| mhc_pre 输出 bf16 截断 | **灾难性（1.0+）** | 同 attn_input（因为就是 mhc_pre 输出） |
| KV cache bf16 存储 | 更差 | KV 精度破坏 |
| residual bf16 截断（mhc_post 后） | 更差 | 影响下一层 mhc_pre |
| mhc_pre f32→float32 累加改变 | 无效（1.00x） | f32/f64 差异远小于 0.51% 误差 |
| hash routing（layers 0-2） | L0-1 改善，L3+ 更差 | 混沌路径依赖 |
| GPU mhc_blend_bf16 kernel | **灾难性（1.0+）** | 与 CPU bf16 截断等效，同样灾难性 |

### 15.2 关键认识：attn_input 是混沌系统的敏感点

`attn_input`（从 mhc_pre 输出，norm ~16）经过 `wq_a`（[1024, 4096]）后变成 Q_LORA（norm ~10），再经过 `wq_b`（[32768, 1024]）变成 Q（64×512）。

0.03 的 attn_input 变化（bf16 最大截断量）通过 wq_b 的 32768 维放大后导致每个 head 的 Q 向量偏差约 0.5-1.0（相对于 norm ~2 的 Q 向量），这使 SDPA 的 attention pattern 完全改变。

**这就是为什么 attn_input bf16 截断是灾难性的**：它改变了 64 个 attention head 的每一个，导致 attention output 完全错误（rel_L2 从 0.67% 跳到 >100%）。

### 15.3 技术结论（定论）

**metal-full 的正确性只能通过以下方式实现：**
将整个 mhc_pre 运算搬到 Metal GPU kernel，使用 native `bfloat` 类型，让 `fn @ residual_flat` 的内部计算精度与 MLX 一致。

**不行的方案**：在任何步骤截断 f32 中间值到 bf16（会破坏后续计算的稳定性）。

**唯一可行的方案**：GPU kernel 内部全程 bfloat，不做截断，让 Metal 驱动原生地处理 bfloat 算术（包括 SIMD 累加精度）。

### 15.4 已有基础设施

以下 Metal kernel 已实现（metal 3.1 bfloat 类型，runtime 编译通过）：
- `dequant_matvec_affine_bf16out`：affine matmul → bfloat 输出
- `rms_norm_rows_bf16out`：RMSNorm → bfloat 输出
- `bf16_to_f32`：bfloat → float 转换
- `mhc_blend_bf16`：mhc_pre 最终 blend 步骤 → bfloat 输出

以及 `mhc_pre_with_premix()`（暴露 pre_mix 供 GPU 调用）。

### 15.5 下一步工作

要完成 metal-full 正确性，需要写一个 **mhc_pre_gpu** Metal kernel，实现：
1. RMSNorm（f32 accumulation within kernel）→ mixes[24]，bfloat 存储
2. sigmoid → pre_mix[4]，bfloat 存储
3. `out_input = sum_m pre_mix[m] * residual[m,:]` → bfloat 输出（即 `mhc_blend_bf16` 的扩展版）

关键：kernel 内部使用 `bfloat` 做中间存储，使计算路径与 MLX 完全匹配。

**估计工作量**：1天（实现 + 验证单层 mhc_pre 输出与 MLX 一致），然后端到端测试。

**当前状态**：
- metal-full: 17% layer-0 误差，输出"Let's think step by step"（语义相关但错）
- mlx oracle: Paris ✓, 4 ✓（smoke PASS）
- 工具链完整，可快速迭代


---

## 16. 2026-06-04 会话：bfloat16 全链路尝试与根本原因分析

### 16.1 本次会话工作

本次会话尝试了以下方案（均以失败告终）：

| 方案 | 结果 | 原因 |
|------|------|------|
| bfloat mhc_pre/post (GPU) + f32 SDPA | **Layer 29 爆炸**，ffn_out rms=505 | BOS token 的 bfloat residual 与 Expert 126 的 gate 方向高度对齐，权重=1.493，输出爆炸 |
| bfloat mhc + f32 SDPA + clamp(±200) | 仍输出乱码 | clamp 只阻止了爆炸传播，但 bfloat mhc 的 residual 本身就是错的 |
| f32 mhc + bfloat SDPA | 比基线更差 | Q 链仍是 f32，转换到 bf16 后与 MLX 的 bf16 Q 不同，导致更差的注意力输出 |
| f32 mhc + bf16 attn_out 截断 | 更差 | 截断到错误的 bf16 bin，与 MLX 不一致 |

### 16.2 关键发现

**Layer 29 爆炸的精确机制**：
- Expert 126 在 Layer 29 有一个特殊的 gate weight 方向（norm=3.52，高于平均 3.1）
- bfloat mhc 路径下的 BOS token (token_id=0) 经过 29 层后，ffn_normed 恰好高度对齐 gate_w[126]
- 这导致 Expert 126 的路由权重=1.493（理论最大值 1.5），其 down_proj 产生 rms=505 的输出
- f32 mhc 路径下，BOS token 的残差方向不同，不触发此对齐

**残差量级系统性漂移**：
- L00: metal/MLX 比值=1.02（几乎完全一致）
- L03: 比值=1.29（metal 比 MLX 大）
- L17: 比值=0.97（开始比 MLX 小）
- L29: 比值=0.63，L42: 比值=0.54

L03 的突然跳变说明 Layer 3 的 expert 选择错误（borderline expert swap 导致选了不同的 expert，其输出恰好更大）。这个错误在后续层持续传播和放大。

**bfloat SDPA 无效的原因**：
我们的 Q 链是 f32，转换为 bf16 后得到的是 `bf16(f32_Q)`，而 MLX 的 Q 是 `bf16_chain(bf16_x, bf16_wqa, ...)` - 这两个 bf16 值不同。仅改变 SDPA 计算精度而不改变整个 Q 链的精度，会适得其反。

### 16.3 为什么 f32 mhc (1c1af0b 状态) 是目前最好的

f32 mhc 路径之所以优于所有 bfloat 路径：
1. mhc_pre/post 用 f32，比 bf16 更精确（接近 f64 参考值）
2. Layer 0 残差几乎完全正确（ratio=1.02）
3. 注意力输出 rel_L2=0.67%（非常接近 MLX f64 参考）
4. Layer 3+ 的 routing 误差来自 0.67% 的注意力误差积累

当前状态（c9be92c commit）：
- metal-full France: "of France is... Let's think step by step"（含 france，不含 paris）
- metal-full 2+2: 错误（routing 在 L3 开始偏离）
- MLX oracle: 全部正确

### 16.4 真正的解决方案

要让 metal-full 通过 smoke test，需要满足以下条件之一：

**方案 A（推荐）：完整 bfloat16 注意力链**
- wq_a 输出截断到 bf16
- q_norm 在 bf16 上操作
- wq_b 输出截断到 bf16
- per-head norm 在 bf16 上操作
- **关键**：Q 链和 KV 链都在 bf16 操作
- SDPA 用 mla_sdpa_decode_bfloat（已实现）
- mhc_pre/post 继续用 GPU bf16 kernel（已实现）
- 添加 FFN 输出安全 clamp（防 Expert 126 爆炸）

**方案 B（更保守）：数值校正**
- 分析 L0-L2 的 expert 选择差异
- 在 metal 路径中加入 routing score 的系统性校正
- 工作量较大，效果不确定

**方案 A 的预估工作量**：2-3天（实现完整 bf16 Q/KV 链 + 安全 clamp 调试）

已有基础设施：
- `mla_sdpa_decode_bfloat` kernel（已实现）
- `dequant_matvec_affine_bf16in_bf16out` kernel（已实现）
- `rms_norm_rows_bf16in_bf16out` kernel（已实现）
- `rope_tail_interleaved_bf16` kernel（已实现）
- `mhc_pre_bfloat` / `mhc_post_bfloat` GPU kernel（已实现）

只需要把这些 kernel 正确串联，并在合适位置加安全 clamp。


---

## 17. 2026-06-04 会话追加：非确定性分析 + ds4 精度对比

### 17.1 关键发现：Metal GPU 计算非确定性

server 的输出**在不同 run 之间不确定**：
- Run 1: "of France is... Let's think step by step"（含 france）
- Run 2-3: "to the // question, and, dear reader"

这不是 KV cache 污染（`resetKv` 每次请求前调用，清零 KV）。**根本原因**：Metal GPU 的 `simd_sum` 等归约操作在不同执行中因执行顺序不同而产生微小差异（FP 不确定性），在 expert 选择的边界区域导致不同的 topK 结果。

### 17.2 ds4 精度分析

从代码分析得出 ds4 的精度链：
1. **Q chain**: 全程 f32（`matvec_q8_0`, `rms_norm_weight`, `head_rms_norm_inplace`）
2. **KV 存储**: f32 → f16 → f32 round trip（`ds4_gpu_encode_f16_round_copy_for_raw_store`）
3. **SDPA**: `half4` Q·K（f16 精度），f32 累加
4. **SwiGLU clamp**: ±10（`DS4_SWIGLU_CLAMP_EXP = 10.0f`）

ds4 用 **f16（half）** 精度，不是 bf16。但 ds4 可以正确产生 France→Paris, 2+2→4。

### 17.3 单步精度测试结果

| 方案 | rel_L2 vs f64 | 备注 |
|------|--------------|------|
| f32 SDPA (1c1af0b) | 0.40% | 当前基线 |
| bf16 KV + bfloat SDPA | 0.23% | 服务器输出乱码 |
| bf16 KV + f32 SDPA | 0.15% | **最佳单步精度**，服务器输出仍乱码 |

bf16 KV round trip + f32 SDPA 的单步精度最好（0.15%），但服务器仍输出乱码。这说明单步精度改善不足以解决多步推理的发散问题。

### 17.4 真正的根本原因

Metal GPU 的 f32 算术非确定性 + borderline expert selection = 随机输出

解决路径只有一个：**完全复制 MLX 的 bf16 计算链**，使 expert 选择结果确定且与训练时一致。具体需要：
1. Q chain 每步截断到 bf16（wq_a → bf16 → q_norm → bf16 → wq_b → bf16 → head_norm → bf16）
2. KV chain 每步截断到 bf16
3. bfloat SDPA
4. mhc_pre/post 全部用 GPU bf16 kernel

这等价于回到 `7b7bfe4` 的完整 bfloat16 路径，但加上 FFN 安全 clamp 防止 Expert 126 爆炸。

**当前状态**：HEAD=c9be92c (1c1af0b f32 基线)，偶发含 france，无 paris，2+2 无 4。


---

## 18. 关键结论与现状更新（2026-06-04 最终）

### 18.1 §16.4 方案 A 已经尝试并失败

§16.4 中的"方案 A"（完整 bfloat16 链 + FFN clamp）在本次会话中已全部尝试，均失败：

| 尝试 | clamp 值 | 结果 |
|------|---------|------|
| 7b7bfe4 路径（bfloat mhc + f32 SDPA）+ clamp ±200 | ±200 | 乱码 |
| 7b7bfe4 路径 + clamp ±30 | ±30 | 乱码 |
| 7b7bfe4 路径 + clamp ±20 | ±20 | 乱码 |
| f32 mhc + bfloat SDPA + bf16 KV | 无 | 乱码 |
| f32 mhc + f32 SDPA + bf16 KV（round trip）| 无 | 乱码 |

**失败的根本原因**（已确认）：不是 clamp 值的问题，而是非确定性问题：
- Metal GPU `simd_sum` 在不同 run 之间产生微小差异
- 这导致 borderline expert 选择在不同 run 之间不一致
- **所有改变精度的方案都让结果更差或没有改善**，因为任何与 MLX 的偏差都会导致错误的 expert 路由

### 18.2 ds4 的精确实现与我们的差距

ds4 使用 f16（不是 bf16）：
- Q chain: f32 全程（不截断）
- KV: f32 计算，存储前做 f32→f16→f32 round trip
- SDPA: Q(f32→half) · K(f32→half) 点积
- 结果: 能正确产出 France→Paris, 2+2→4

我们实现了类似方案（bf16 KV + f32 SDPA），单步精度提升到 0.15%（从 0.40%），但服务器仍然输出乱码。原因：

```
单步精度改善 ≠ 多步推理正确性
```

因为即使 0.15% 的单步误差，在 Layer 3+ 的 borderline expert 选择中仍然导致错误路由。每层的误差积累，最终偏离正确的计算路径。

### 18.3 实际当前状态（HEAD = 5e46be4）

**代码状态**：1c1af0b 基线（f32 mhc + f32 SDPA + f32 KV）

**metal-full 实测行为**：
- France: 一致输出 "to the\n// question, and, dear reader"（不含 paris/france）
- 2+2: 一致输出 "to 'just' after '2' after..."（不含 4）
- **完全不通过 smoke test**

注：之前声称的"偶发含 france"是不准确的——那是特定 Metal GPU 状态下的偶然，并不可重现。系统性测试（8次连续请求）显示 0/8 含 france。

**MLX 路径**：正常工作，France→Paris，2+2→4（smoke test PASS）

### 18.4 下一步（已更新）

§16.4 中方案 A 已宣告失败。需要新的思路：

**可能有效的方案**：

**方案 C（最有可能成功）**：ds4 的 f16 路径
- 使用 f16（half）而非 bf16 做 KV round trip
- 使用 f16 精度的 SDPA（需要新的 `mla_sdpa_decode_f16` kernel）
- ds4 用此方案成功，原因可能是 f16（10-bit mantissa）比 bf16（7-bit mantissa）精度更高
- 预估工作量：1天（实现 f16 SDPA kernel + 测试）

**方案 D（回退策略）**：
- 接受 metal-full 目前不通过 smoke test 的事实
- 专注于 `--metal-moe` 路径（MoE 在 Metal，attention 在 MLX）的性能优化
- 这是确定性正确且已验证的路径

**方案 E（从源头修复）**：
- 在 mhc_post 之后的 residual 上加一个 "精度注入"：从 MLX 拿到正确的 Layer N residual，用于校正 metal 路径的积累误差
- 这需要 MLX/metal 混合执行，复杂度高

**关键教训**：
- 改变精度（f32→bf16/f16）在 attention 层级的改善不足以修复 Layer 3+ 的 routing 误差
- Metal GPU 的非确定性是一个真实障碍，不是代码 bug
- ds4 能工作说明 f16 精度有效，但需要 f16 kernel（我们目前只有 bf16 kernel）

## 19. 精度路径选择回顾：为什么从 f32 出发，为什么 bf16 失败，为什么最终选 f16（2026-06-05）

### 19.1 为什么当初设计为 f32？

回顾 S0-S7 的构建过程，f32 路径是自然的工程起点，原因有三：

**（1）S1 的对拍需求**
第一个 go/no-go 关卡是 `dequant_matvec_affine` kernel 能否在固定输入下与 MLX `quantizedMatmul` 逐元素对齐（max diff ≈ 0）。MLX `eval()` 后输出是 f32，因此 kernel 输出 f32 最直接、最方便用 numpy 做阈值比对。如果当时选 bf16/f16，对拍脚本需要额外处理精度转换，增加第一轮验证的复杂度。

**（2）开发便利性与安全假设**
f32 不会溢出、不会下溢、simd_sum 行为最可预测。团队在 S7d 阶段需要连修 7 个集成 bug（segfault、buffer 对齐、broadcast view 等），如果中间还叠加 bf16/f16 的精度调试，定位难度会指数上升。当时的判断是：**先让 43 层无 crash 跑通，再谈精度压缩**。

**（3）错误的直觉假设**
团队隐含认为：f32 比 bf16 精度高，因此 f32 Metal 路径应该「至少不比 MLX bf16 差」，甚至可能更接近「真实数学值」。这个假设在单步验证中似乎成立（attn_out rel_L2 = 0.67%），但在多步推理中被彻底推翻。

### 19.2 为什么「直接选 bf16」也失败了？

§16.4 的方案 A（完整 bfloat16 注意力链）在 `7b7bfe4` 已实现并测试，结果乱码。所有 bf16 变体（bf16 mhc、bfloat SDPA、bf16 KV round trip）全部失败。

根本原因是：**MLX 的 bf16 不是「每个算子后截断」这么简单**。MLX 拥有 lazy evaluation 和 graph fusion，它的 bf16 截断时机与 fusion 边界、tile 划分、simd reduction 顺序深度绑定。我们自行编写的 Metal kernel 即使使用 Metal 3.1 的 `bfloat` 类型，也无法复现这些框架内建行为：

- **simd_sum 归约顺序不同** → 累加误差不同
- **matmul tile 划分不同** → 中间累加精度不同  
- **norm/activation fusion 边界不同** → 截断时机不同

因此，我们的「完整 bf16 链」和 MLX 的 bf16 链是**两条不同的链**。团队试图通过调整截断点、加 clamp、改 round mode 来逼近 MLX，结果越调越乱（§15 全部尝试失败）。

### 19.3 为什么 ds4 的 f16 能工作？

ds4 与我们的关键差异在于：**ds4 从不试图逐层匹配 MLX**。

| 维度 | ds4 | dmlx metal-full |
|------|-----|-----------------|
| 目标 | 端到端正确即可 | 逐层对齐 MLX oracle |
| 权重格式 | GGUF + IQ2（量化已冻结训练截断） | MLX safetensors（原生 bf16） |
| Q chain | f32 全程 | f32（与 mlx bf16 不一致） |
| KV / SDPA | f32→f16→f32 round trip + half 点积 | f32 全程（与 mlx bf16 不一致） |
| 精度哲学 | "我用自己的一致精度算完 43 层" | "我必须在每一层都和 mlx 一致" |

ds4 的 f16 路径是**自洽的独立系统**：KV cache 和 SDPA 都遵循同一套 f16 规则，43 层累积的误差模式稳定，不会在某一层突然翻转 expert 选择。类比：MLX bf16 是方言 A，ds4 f16 是方言 B，两者不一致但各自内部通顺。我们的 f32 则是"试图模仿方言 A 的外国人"——每个词都对，但语调不对。

此外，f16 拥有 10-bit 尾数，bf16 只有 7-bit。f16 的 round trip 误差更小，可能恰好足够稳定，不会在 borderline expert 处 flip。

### 19.4 结论：方案 C（f16）的决策依据

> **放弃「逐层匹配 MLX」的验收标准，转而建立一条自洽的 f16 精度链，像 ds4 一样独立跑完 43 层，只要求端到端正确。**

具体实施：
1. 新增 `mla_sdpa_decode_f16` kernel（Q/K 用 `half4` 点积，f32 累加）
2. KV cache 改为 f32→f16→f32 round trip（与 ds4 一致）
3. Q chain 保持 f32（与 ds4 一致）
4. 不对拍 MLX 逐层 hidden state，只对拍端到端输出（France→Paris, 2+2→4）

预估工作量：1 天（实现 kernel + 集成 + 端到端验证）。

---

## OOM 风险提示

**不要在同一台机器上同时运行多个 dmlx serve 实例。**

每个实例都会独立加载模型权重（~4GB+ RSS），同时运行两个或更多实例会迅速耗尽物理内存，导致系统 OOM 或被 macOS 内存压缩机制拖垮。测试 `--metal-full` 和 `--metal-moe` 时必须串行进行：先关闭前一个进程（`kill <pid>`），再启动下一个。

---

## 20. 代码审计与精度诊断（2026-06-06）

### 20.1 关键 bug 修复

**`fused_gate_up_swiglu` 占位符 bug（`src/models/moe_kernel.metal` line 63）**

发现 `fused_gate_up_swiglu` kernel 最后一行是：
```metal
float act = g_c / (1.0f + exp(-g_c));
out[tid] = 999.0f;  // ← BUG: act * u_c 从未写出
```

已修复为：
```metal
out[tid] = act * u_c;
```

**影响分析**：这个 bug 导致 `--metal-moe` 之前能输出 Paris 是"假正确"——`999 × mxfp4_down_weights` 在各维方向抵消约等于 0，路由 expert 贡献被消除，整层 MoE 只有 shared expert 在起作用。shared expert 恰好足够让 MLX backbone 输出 Paris。

修复后，路由 expert 有真实输出，暴露了真正的精度问题。

### 20.2 系统性精度诊断

**测试方法**：用 `DSV4_DUMP_DIR` dump 逐层激活，对比 `--metal-moe` 与纯 MLX 的 layer outputs。

**关键发现**：
- `ffn_normed`（MoE 输入）：`--metal-moe` vs MLX rel_L2=**0.0**（完全一致，因为 MLX 做 attn/mhc/norm）
- `layer_00` 输出：rel_L2=**0.24**（修复后）

这说明：同一 normed 输入 + 同一 expert IDs（MLX routing 给出），Metal f32 mxfp4 compute 与 MLX bf16 mxfp4 compute 有 24% 误差。

**精度体制差异（根本原因）**：
- MLX `switch_mlp`：x 是 bf16，mxfp4 matmul 内部用 bf16 精度
- Metal `fused_gate_up_swiglu`：x 是 f32（来自 MLX eval → dataPtr(f32)），mxfp4 matmul 全 f32

这 24% 的误差在 43 层后完全发散（L3 就到 0.44，L12 接近随机），无法通过后处理修复。

### 20.3 bf16 input 尝试（失败）

尝试了 `fused_gate_up_swiglu_bfloat_in` kernel（bfloat input），并在 wrapper 里先做 `f32_to_bf16` 转换。结果：rel_L2 从 0.24 **恶化到 0.97**（near-orthogonal）。

原因：MLX 内部的 bf16 chain 不等于 `f32_to_bf16(f32_from_mlx_eval)`。MLX 的 bf16 tensor 经过 lazy fusion 优化，其截断时机与 `f32→bf16→metal` 两次转换路径完全不同。bf16 input 方案已放弃。

### 20.4 当前代码状态

| 变更 | 文件 | 说明 |
|------|------|------|
| ✅ 修复 `fused_gate_up_swiglu` 999 bug | `src/models/moe_kernel.metal` | `out[tid] = act * u_c` |
| ✅ 添加 `fused_gate_up_swiglu_bfloat_in` | `src/models/moe_kernel.metal` | 保留供未来参考，未使用 |
| ✅ 添加 `dequant_matvec_4bit_bfloat_in` | `src/models/moe_kernel.metal` | 保留供未来参考，未使用 |
| ✅ mhc_pre 切换 f16 权重路径 | `src/metal_infer/engine.c` | `metal-full` 用 `attn_hc_fn_f16` |
| ✅ mla_attention 切换全 f16 chain | `src/metal_infer/engine.c` | `mla_attention_decode` 替代 `mla_attention_decode_f16kv` |
| ⚠️ `moe_metal_wrapper.c` 保持 f32 path | — | bf16 path 更差，已回滚 |

**注意**：`engine.c` 的 `mhc_pre` + `mla_attention_decode` 切换（f16 path）仅影响 `--metal-full` 路径，`--metal-moe` 仍用 `moe_metal_wrapper.c`（f32）。

### 20.5 当前性能参考

| 模式 | tok/s | 正确性 | 说明 |
|------|-------|--------|------|
| 纯 MLX | ~0.36-0.45 | ✅ Paris | 对拍基准 |
| `--metal-moe` (f32) | ~4.5 | ❌ 乱码 | 24% per-layer 误差，43层发散 |
| `--metal-full` (f16) | ~1.7 | ❌ 乱码 | attn+mhc精度链不匹配 |

### 20.6 下一步路径分析

**路径 A（放弃 metal kernel 精度对齐，接受 ~10% 误差）**：

研究 `--metal-moe` 路径中，24% layer误差是否可以通过"归一化截断"来降低。具体：在 combine 之后、写回 MLX 之前，对 Metal 的输出做方向校正（scaling），使其与 MLX 的量级对齐。这不改变语义，只是让误差不在每层累积。

**路径 B（纯 MLX + SMELT + 性能优化）**：

放弃 Metal compute，只用 MLX。性能瓶颈在于 `switch_mlp` 的 MLX lazy eval 同步，可以通过 `--smelt` 的 expert 预加载策略 + MLX stream 优化来改善。这是已知正确的路径，专注性能而非精度。

**路径 C（从 ds4 导入 bf16 mxfp4 kernel）**：

`../ds4/metal/moe.metal` 里可能有 bf16 精度的 mxfp4 kernel，参考其实现。但 ds4 用的是 GGUF IQ2 格式而非 MLX mxfp4，kernel 可能需要从头写。

**最推荐的短期路径**：路径 B（纯 MLX + 性能优化）是风险最低、最有可能在短期内出成果的方向。metal compute 在精度对齐上遇到了系统性障碍（§14-§20 均失败），继续投入的边际收益递减。

---

## 22. 操作规范（强制，每次操作前必读）

> 本节记录所有在调试过程中发现的操作约束。遇到新约束立即补充到此处。

### 22.1 进程管理

**只能同时运行一个 `dmlx serve` 实例。**
- 启动新 serve 前必须先 kill 旧进程
- 不允许同时开两个 serve 做对比测试
- 对比测试用"先生成 dump，停掉，再跑另一个读 dump"的顺序进行

### 22.2 MLX serve 启动参数

**必须用 `stream` 模式，禁止 `preload`，禁止不加 `--smelt`：**

```bash
# ✅ 正确
./zig-out/bin/dmlx serve \
    --model ~/models/DeepSeek-V4-Flash-4bit \
    --expert-packed-dir ~/models/DeepSeek-V4-Flash-4bit/packed_experts \
    --smelt --smelt-strategy stream --smelt-experts 0.20 --smelt-cache 0 \
    --port 8960 --max-tokens N --temperature 0

# ❌ 禁止（把全部 256 专家加载进内存，极慢且 OOM）
--smelt-strategy preload

# ❌ 禁止（不加 --smelt 同样 OOM，全部专家会被加载）
# 不带 --smelt 的 MLX serve
```

### 22.3 Native serve 启动参数

```bash
# ✅ native 模式（不需要 --smelt，engine.c 内部用 pread）
./zig-out/bin/dmlx serve --native \
    --model ~/models/DeepSeek-V4-Flash-4bit \
    --expert-packed-dir ~/models/DeepSeek-V4-Flash-4bit/packed_experts \
    --port 8961 --max-tokens N --temperature 0

# --expert-packed-dir 是 native 模式的必填项
```

### 22.4 标准调试流程（MLX golden 对比）

```bash
# Step 1: 生成 MLX golden dump（stream 模式）
DSV4_DUMP_DIR=/tmp/mlx_golden MF_DBG=1 \
    ./zig-out/bin/dmlx serve --model ~/models/DeepSeek-V4-Flash-4bit \
    --expert-packed-dir ~/models/DeepSeek-V4-Flash-4bit/packed_experts \
    --smelt --smelt-strategy stream --smelt-experts 0.20 --smelt-cache 0 \
    --port 8960 --max-tokens 2 --temperature 0 &
SERVER_PID=$!
# 等待启动
while ! curl -sf http://localhost:8960/health >/dev/null 2>&1; do sleep 1; done
# 发一次请求触发 dump
curl -s http://localhost:8960/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"default","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":2,"temperature":0}'
kill $SERVER_PID; wait $SERVER_PID 2>/dev/null

# Step 2: 跑 native，生成 native dump
MF_DBG=1 DSV4_DUMP_DIR=/tmp/native_dbg \
    ./zig-out/bin/dmlx serve --native \
    --model ~/models/DeepSeek-V4-Flash-4bit \
    --expert-packed-dir ~/models/DeepSeek-V4-Flash-4bit/packed_experts \
    --port 8961 --max-tokens 2 --temperature 0 &
SERVER_PID=$!
while ! curl -sf http://localhost:8961/health >/dev/null 2>&1; do sleep 1; done
curl -s http://localhost:8961/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"default","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":2,"temperature":0}'
kill $SERVER_PID; wait $SERVER_PID 2>/dev/null

# Step 3: 对比
python3 scripts/compare_metal_mlx.py /tmp/mlx_golden /tmp/native_dbg
```

### 22.5 Smoke test 命令

```bash
# MLX 路径（正确性基准）
bash scripts/dsv4_smoke.sh

# Native 路径
NATIVE=1 bash scripts/dsv4_smoke.sh
```

### 22.6 构建

```bash
zig build -Doptimize=ReleaseFast
# 只看错误，忽略 mlx-c warning
zig build -Doptimize=ReleaseFast 2>&1 | grep -v "warning: mlx"
```

### 22.8 测试 token 数量

- smoke test / 正确性验证：`--max-tokens 30`（足够让模型完整回答简单事实题）
- 性能测试：`--max-tokens 30`
- 调试 dump（减少等待时间）：`--max-tokens 1`（只需要看第一个生成 token）
### 22.7 目标模型

- 路径：`~/models/DeepSeek-V4-Flash-4bit`
- 格式：MLX safetensors，mxfp4（专家 gs=32）+ affine 4bit（注意力/embed gs=64）
- 全精度模型（`~/models/deepseek-ai/DeepSeek-V4-Flash`，148.7GB BF16）**不适用**（超出 48GB 内存限制）
- packed experts 目录：`~/models/DeepSeek-V4-Flash-4bit/packed_experts/`（必须已用 `repack_experts.py` 生成）

### 21.1 战略方向

放弃 MLX 依赖，走纯 C/Metal/Zig 路径（`--native` 模式）。参照 ds4 架构：

- **native_loader**：直接读 safetensors，无 MLX runtime
- **engine.c**：完整的 forward pass（attention + mHC + MoE + compressor/indexer）
- **目标**：`--native` 输出 `Paris` for `"The capital of France is"`

### 21.2 已完成

| 组件 | 状态 | 说明 |
|------|------|------|
| native_loader (safetensors.zig/config.zig/weights.zig) | ✅ | 2143ms 加载全量权重，值对齐验证通过 |
| moe_infer_embed / get_logits | ✅ | cblas_sgemv lm_head |
| mla_attention_decode / _f16kv | ✅ | f32 Q + f16 KV 精度链，rel_L2=1.9e-6 |
| mhc_pre / mhc_post (CPU) | ✅ | ≤6e-8 |
| CompressorState + compressor_step | ✅ | 已按 ds4 per-dimension softmax pooling 实现 |
| mla_attention_decode_mixed | ✅ | 已实现（替代原来的 stub），CPU mixed SDPA |
| IO pool race bug | ✅ | 修复：6个专家现在都有输出（之前只有expert[0]有输出）|
| server routing (nativeEngineLoop) | ✅ | server.zig 已有 native 请求处理路径 |
| dsv4_smoke.sh --native 支持 | ✅ | `NATIVE=1 bash scripts/dsv4_smoke.sh` |

### 21.3 当前状态（IO race 修复后）

- `--native` 服务能启动，处理请求
- 6 个路由专家均有输出（修复 IO race 后），`ffn_out(moe only) norm` 提升到 ~0.5-0.9
- 但 **输出仍错误**：`"The capital of France is"` → `"to the question..."` 而非 `"Paris"`
- `logits max=304 (' to')` vs 期望 11111 (' Paris')

### 21.4 已修复的 Bug

| # | Bug | 现象 | 修复 |
|---|-----|------|------|
| 1 | IO pool race condition | expert[1..5] mid_norm=0（只有expert[0]有输出）| 在持锁时原子标记任务为已claimed，保存原始fd再释放锁 |
| 2 | mla_attention_decode_mixed 是 stub | 41/43 层 comp_kv 被忽略 | 实现真正的 CPU mixed SDPA（SWA raw f16 + comp_kv f32）|
| 3 | compressor pooling 算法错误 | 使用 per-token 标量 sum 作为 softmax 权重 | 改为 ds4 的 per-dimension softmax pooling |

### 21.5 当前状态与结论（2026-06-07 全天深度调试）

修复 IO race 后，6个专家都有输出，但答案仍错误。经过全天深度调试，结论如下：

**已修复的 bug**：

| # | Bug | 现象 | 修复 |
|---|-----|------|------|
| 1 | IO pool race condition | expert[1..5] 全为零输出 | 持锁时原子 claim 任务 |
| 2 | `mla_attention_decode` 使用未初始化 BF16 pipeline | attention 输出为零/随机 | 改用 `mla_attention_decode_bf16`（bf16 Q+KV） |
| 3 | `mla_attention_decode_mixed` 导致 degeneration | decode 后期输出乱码循环 | 只在 `kvc->len > SWA_WINDOW` 时使用 |
| 4 | Serial prefill（每 token 跑完 43 层） | 与 MLX batch prefill 顺序不一致 | 实现 `moe_infer_forward_batch`（layer-first） |
| 5 | mhc_pre/post 用 `half`（f16）而非 `bfloat`（bf16）| GPU kernel 解读 bf16 数据为 f16，输出 265K norm | 改用 `mhc_pre_gpu`（f32→f32） + `mhc_post_bfloat`（bfloat I/O）|

**已确认正确的组件**（Python 验证）：
embed dequant、lm_head dequant、Q chain、MoE mxfp4、expert routing、shared expert、attn_sink、mhc_pre（GPU版与CPU完全一致）

**精度分析**（关键数据）：

MLX golden dump（`max_tokens=1, DSV4_DUMP_DIR`）结果：
- MLX top logit for pos=8（`</think>`）: `.`(16)=**18.70** >> `to`(304)=17.37 → 首个 token=`.`(正确)
- Native top logit: `to`(304)=**18.6** >> `.`(16)≈15.7 → 首个 token=` to`(错误)

Layer 对比：
| 层 | cosine | rel_L2 | 说明 |
|----|--------|--------|------|
| Layer 0 out | 0.992 | 0.129 | GPU mhc 后依然有 3% 误差（Stream 3：MLX=17.47, Native=16.97）|
| Layer 42 out | 0.740 | 0.724 | 指数放大，最终 logit 方向完全偏离 |

**3% Layer-0 误差根因**：
- `attn_out`（来自 `mla_attention_decode_bf16` vs MLX `fast.scaledDotProductAttention`）有 0.88% 差异
- 这通过 FFN `mhc_post_bfloat` 放大：`post_ffn[3]=0.537`，Stream 3 = `0.537 × 31.7 × delta`
- 43 层指数放大 → L42 cosine 仅 0.74

**SDPA 是根本瓶颈**：Metal `mla_sdpa_decode_bfloat` kernel 的浮点归约顺序与 MLX batch prefill SDPA 不同，导致不可避免的 ~3% per-layer 差异。

**当前 `--native` 输出**：
- 不再乱码循环 ✓
- 语义连贯：`to the capital of France is. The capital of France is.` ✓
- 不输出 Paris ✗

**要输出 Paris 的路径**：
1. 实现与 MLX batch prefill SDPA 相同浮点归约顺序的 Metal kernel（高难度）
2. 或使用 MLX backbone（= `--metal-moe`，已验证输出 Paris）

### 21.6 关键操作约束

> 见 **§22 操作规范**（强制）。核心：同时只能跑一个 serve；MLX serve 必须用 stream 模式。

### 21.7 下一步

根据精度分析，SDPA 是根本瓶颈（不是 mhc）。已完成：
- ✅ `mhc_pre_gpu`（f32→f32 bf16-truncated）已接入
- ✅ `mhc_post_bfloat`（bfloat I/O，匹配 MLX 的 `.astype(x.dtype)=bfloat16`）已接入

**剩余问题**：Metal `mla_sdpa_decode_bfloat` 的逐 token decode 归约模式 ≠ MLX batch prefill SDPA 归约模式。误差 ~0.88% per attention output，43 层后放大到 72% rel_L2。

**可选路径**：
1. **实现 batch prefill SDPA kernel**：处理 N 个 query，与 MLX batch SDPA 归约顺序一致
2. **改用 `--metal-moe` + rename**：MLX backbone 已验证正确，绕过 SDPA 精度问题
3. **接受当前行为**：输出语义连贯但不含 Paris，专注性能和长文本（pos > 128 才开启 mixed）

---

## §23 诊断轮次二（2026-06-08）

### 23.1 先前结论的错误

§21 记录的"3% layer-0 误差来源于 SDPA 归约顺序"**基于错误的对比数据**：
- `/tmp/mlx_ref/` 里的 MLX golden 是用**不同 prompt**（system/query 格式）生成的，而 native 处理的是 France prompt
- 两个 prompt 的 `L0_attn_out` 比较 cosine ≈ -0.02，根本不是精度问题，而是输入完全不同
- 因此之前"72% rel_L2"的结论无效

### 23.2 正确诊断流程

重新生成了与 native 相同 prompt 的 MLX golden（`/tmp/mlx_ref_new/`），得到以下结论：

#### 23.2.1 经 Python 精确验证的正确组件

| 组件 | 验证方式 | 结论 |
|------|----------|------|
| `embed → mHC_pre → L0_attn_normed` | Python 重现 | cosine=1.000，rel_L2=0.005 ✓ |
| `mla_attention_decode_bf16`（单步） | `mla_attention_test` | cosine=1.0 ✓ |
| `mla_attention_decode_bf16`（多步 KV 缓存） | `mla_attention_multistep_test` | rel_L2=4.7e-4，**RESULT: GO** ✓ |
| SDPA + attention sink 数学 | `verify_sdpa_sink.py` | max_abs=3e-8，**RESULT: GO** ✓ |

#### 23.2.2 Layer-0 attention 对比（Python vs MLX golden，France prompt）

使用 MLX golden 的 `L0_attn_normed` 作为输入，Python 重现注意力输出：

| token | rel_L2（f32 KV） | cosine | 说明 |
|-------|-----------------|--------|------|
| 0 | 0.011 | **1.0000** | 单 KV 行，完全匹配 ✓ |
| 1 | 0.238 | 0.971 | KV 精度差异开始影响 |
| 8（最后） | 0.652 | 0.785 | 7 步 KV 积累差异 |

token 1-8 的误差**不是算法 bug**，而是：
- MLX 使用 FP8+F16 round-trip KV cache（`kernel_dsv4_kv_fp8_store_f32`）
- Python/native 用 bf16/f32 KV，精度更高但与 MLX 训练时的数值走向不同
- 更换为 f16 KV 或 FP8+f16 KV，误差完全一样（因为 FP8 噪声极小，差异来源于 KV 内容本身的微小不同）

#### 23.2.3 本轮做的改动

1. **`mla_sdpa_decode_bfloat` kernel 重写**（`src/models/moe_kernel.metal`）：
   - 参照 ds4 `kernel_flash_attn_ext_vec_f16_dk512_dv512` 的结构
   - Q: bf16 → float4，KV: f16（half）匹配 ds4
   - 用 `dot(float4, float4)` FMA 替代标量累加
   - 32 线程（1 simdgroup），每线程处理 4 个 float4 = 16 维
   - Dispatch 改为 `threadsPerThreadgroup:32`

2. **KV cache 精度：bf16 → f16**（`src/metal_infer/mla_attention.m`）：
   - 写入 KV cache 时加 bf16→f16 显式转换
   - SDPA kernel 的 KV 输入从 `bfloat*` 改为 `half*`

3. **`@autoreleasepool` 包裹**（`src/metal_infer/engine.c`, `mla_attention.m`）：
   - `moe_infer_forward_layer` 整体包裹，每次调用后释放临时 Metal buffer
   - `mla_attention_decode_bf16` 整体包裹

4. **Per-layer residual dump**（`src/metal_infer/engine.c`）：
   - `MF_DBG=1 DSV4_DUMP_DIR=...` 时每层写 `L{NN}_residual_last.bin`

### 23.3 根本问题：OOM

**所有改动后 smoke test 结果不变（仍输出 `to`），原因是 OOM**：

每次 `moe_infer_forward_layer` 调用通过 `newBufferWithBytes`/`newBufferWithLength` 动态分配 ~71 个临时 Metal buffer，包括：
- `attn_hc_fn` 权重复制：`24 × 4 × 4096 × 4 = 1.5 MB` per call
- Q/KV chain 的多个 buffer（bq_a, bq_res, bq, bq_n, bkv, bkv_n 等）

对于 France prompt（9 tokens × 43 layers = 387 次调用），MLX warmup 再加上这 387 次调用导致系统 OOM kill。

**`@autoreleasepool` 减缓了内存压力但没有根治**，因为：
1. MLX warmup（通过 `deepseek_v4.zig` 的 Zig/MLX 路径）在 native request 之前先跑一遍所有 43 层，消耗了 ~2-3 GB 额外内存
2. Native engine 的逐 token 逐层 Metal buffer 分配没有持久化复用

### 23.4 正确的修复方向

问题不在 SDPA 精度，而在 **Metal buffer 分配策略**：

1. **持久化 Metal buffer**：在 `MoEInferEngine` 初始化时预分配所有需要复用的 buffer（Q/KV chain 的中间结果 buffer，注意力权重的 GPU 拷贝），每次 forward_layer 只复用，不重新分配

2. **权重 GPU 缓存**：`attn_hc_fn`（1.5MB/layer × 43 layers = 65MB）、`input_norms`、`attn_norms` 等应在 init 时上传到 GPU，后续 dispatch 直接引用，不每次 `newBufferWithBytes`

3. **KV chain weights GPU cache**：`wq_a.packed`/scales/biases、`wkv.packed` 等也应预上传

这是性能架构改造，不是数值修复。完成后 native engine 的速度会从当前约 5s/token（全量 buffer 分配）降到目标 1s/token 以内，且不再 OOM。

### 23.5 新的 smoke test 运行约束

> **强制**：见 §22

额外约束（本轮发现）：
- `MF_DBG=1 NATIVE=1` 模式下，**不要**跑超过 3 个 prefill tokens（改用极短 prompt），否则 OOM
- 诊断 dump 只能通过独立 C test 程序（`scripts/mla_attention_multistep_test.m`）做，不通过 server

### 23.6 数值精度现状总结（截止 2026-06-08）

| 位置 | 状态 | 验证方式 |
|------|------|----------|
| embed dequant | ✓ 正确 | Native loader vs safetensors 手算 |
| mHC_pre → L0_attn_normed | ✓ cosine=1.000 | Python 重现 |
| mla_attention_decode（单步+多步） | ✓ rel_L2<0.001 | mla_attention_multistep_test |
| SDPA + sink 数学 | ✓ max_abs=3e-8 | verify_sdpa_sink.py |
| KV cache 精度（bf16/f16） | ✓ 无影响 | Python f16/FP8 对比 |
| 全 43 层 residual 对比 | ❌ 无数据（OOM 无法获取） | 待 buffer 持久化后 |
| MoE FFN 正确性 | ❌ 未验证 | 待 buffer 持久化后 |

**结论**：attention 链路数值正确，预测错误（`to` vs `Paris`）的根本原因是 OOM 导致推理无法完成，而不是算法错误。需要先解决 Metal buffer 持久化问题。

---

## §24 根本原因精确定位（2026-06-08 续）

### 24.1 Layer-by-layer 残差对比（France prompt，最后一个 prefill token）

通过 `DSV4_DUMP_DIR` + `MF_DBG` dump 获得了 native 全 43 层的残差，与 MLX golden 对比：

| 层 | stream 0 cos | stream 1 cos | stream 2 cos | stream 3 cos |
|----|-------------|-------------|-------------|-------------|
| L00 | **0.9750** | **1.0000** | **1.0000** | **0.9809** |
| L01 | 0.919 | 1.000 | **0.688** ← | 0.980 |
| L02 | 0.660 | 0.997 | 0.757 | 0.975 |
| L03 | 0.644 | 0.440 | 0.793 | 0.961 |
| ... | 下降 | 下降 | 下降 | 缓慢下降 |
| L42 | 0.717 | 0.730 | 0.710 | **0.965** |

**关键发现**：Layer 1 的 stream 2 从 1.0 暴跌到 0.688，仅经过一层。

### 24.2 根本原因链

```
embed 正确（Python 验证 norm=4.04）
→ mhc_pre 正确（Python 验证 cos=1.0）
→ attn_normed 和 MLX 一致（cos=1.0）
→ attention 算法正确（multistep test: rel_L2=0.0005）
↓
但 KV 内容和 MLX 不同：
  - MLX 的 KV 使用 batch prefill（9 个 token 并行，simdgroup matmul）
  - Native 的 KV 使用 decode（逐 token，simd_sum）
  → Token 8 的 attn_out: Python 验证 rel_L2=0.65, cos=0.785（vs MLX golden）
  → Native 的 attn_out: 约 5% rel_L2（推算）
↓
mhc_post(attn_out, residual) → layer 0 stream 0 cos=0.975（2.5% 方向误差）
↓
Layer 1 的 mhc_pre 使用了偏离的 layer 0 残差
→ fn @ res_normed 产生不同 mixes
→ sinkhorn 放大小差异：comb[2,1] 差异 12.8%
→ stream 2 更新出错（cos 从 1.0 → 0.688）
↓
误差通过 43 层传播放大 → logit 翻转（.→to）
```

### 24.3 关键技术结论

1. **attention 算法本身是正确的**：multistep test rel_L2=0.0005 ✓
2. **KV 精度不是根因**：FP8+f16 vs f16 的差异极小（Python 验证误差不变）
3. **根本原因是 prefill 方式**：
   - MLX：batch prefill（所有 9 个 token 同时处理，simdgroup 8×8 矩阵乘法）
   - Native：decode 方式逐 token（token 0→1→...→8，每次 SDPA 只看当前 KV）
   - 两者对 KV 内容的计算路径不同，导致 ~5% 的 attention 输出误差
4. **sinkhorn 放大器**：mhc_pre 的 sinkhorn 归一化将 2.5% 的残差误差放大为 12.8% 的 comb 矩阵误差

### 24.4 修复方向

**必须实现 batch prefill SDPA**，与 MLX 的 `kernel_flash_attn_ext_vec_f16_dk512_dv512` 或 `kernel_flash_attn_ext_f16_dk512_dv512` 数值路径一致。

具体要求：
- 同时处理所有 N 个 prefill tokens
- Q×K^T 使用 simdgroup_multiply_accumulate（8×8 tiles）
- 降低每个 token 的 attention 输出误差到 <0.1%

当前的 `mla_sdpa_prefill_bfloat` kernel 使用的是 `simd_sum(partial)` 而非 simdgroup matmul，因此不能解决根本问题。

### 24.5 当前状态

- OOM 问题已修复（persistent Metal buffer）✓
- Smoke test 能正常完成（两个 request 都能跑完）✓
- 输出 `to` 而非 `Paris`，分析确认原因 ✓
- 需要实现 simdgroup matmul 版的 batch prefill SDPA

### 24.6 操作约束（再次强调）

- **仅运行一个 dmlx serve**，不能同时运行 MLX 和 native server
- **MLX serve** 必须使用 `--smelt --smelt-strategy stream --smelt-experts 0.20 --smelt-cache 0`
- **smoke test**: `NATIVE=1 bash scripts/dsv4_smoke.sh`
- **max-tokens 30** 用于正确性测试

---

## §25 诊断轮次三——关键发现（2026-06-08 续续）

### 25.1 所有之前的分析都基于错误数据

**根本错误**：smoke test 跑两个请求（capital-of-france + two-plus-two），第二个请求覆盖了第一个请求的 dump 文件。因此：
- `L0_kvcache_prefill.bin` = two-plus-two 的 KV（8 tokens），不是 France 的（9 tokens）
- `L0*_metal.bin` = two-plus-two 最后 token 的数据
- `L{NN}_residual_last.bin` = two-plus-two 最后 token 的残差

之前所有 "KV[2..7] cos=0.37-0.53" 的分析都是把 two-plus-two 的 KV 和 France prompt 的 MLX normed 对比，完全无效。

### 25.2 正确的验证结果

**KV cache 是完全正确的**：
```
two-plus-two 所有 8 个 token 的 KV:
  ttp tok[0]=0:       cos=1.0000
  ttp tok[1]=128803:  cos=1.0000
  ttp tok[2]=20:      cos=1.0000  (token '2')
  ttp tok[3]=13:      cos=1.0000  (token '+')
  ttp tok[4]=20:      cos=1.0000  (token '2', 重复)
  ttp tok[5]=31:      cos=1.0000  (token '=')
  ttp tok[6]=128804:  cos=1.0000
  ttp tok[7]=128822:  cos=1.0000
```

**所有验证全部 cos=1.0！** KV cache 是正确的。注意 tok[2]=tok[4]=20（同一个 token），所以它们的 nope 部分相同是预期的，不是 bug。

### 25.3 Consecutive vs Interleaved RoPE 发现

`verify_attention_python.py` 使用了 **interleaved** RoPE（`j0=NOPE+i, j1=NOPE+half+i`），但：
- Metal kernel `rope_tail_interleaved_bf16` 实际使用 **consecutive** 配对（`j0=NOPE+2*i, j1=j0+1`）
- `gen_multistep_golden.py` 也使用 **consecutive**
- ds4 的 Metal kernel 也是 **consecutive**

用正确的 consecutive RoPE 重新运行 attention 验证：**所有 9 个 token 的 cos=1.0！**

| 实现 | RoPE 方式 | token 0-8 attention cos |
|------|-----------|------------------------|
| verify_attention_python.py（旧） | interleaved | 1.0, 0.97, 0.83, 0.78, ... |
| gen_multistep_golden.py | consecutive | 1.0, 1.0, 1.0, ... |
| mla_attention_multistep_test.m | consecutive | rel_L2=0.0005 |

**结论**：native attention 是完全正确的。之前分析的"21% 误差"完全是由于用了错误的 Python RoPE。

### 25.4 真正的问题所在

用 Python 实现完整的 layer 0（embed → mhc_pre → attention → mhc_post → FFN mhc_pre），在 FFN 输出为 0 的情况下：
- stream 0: cos=0.970
- stream 1/2: cos=1.0
- stream 3: cos=0.035（几乎没有相关性！）

**stream 3 主要依赖 FFN 输出**（因为 `post_ffn[3]=0.244`，comb 矩阵显示 FFN 输出对 stream 3 贡献 `0.244 * ffn_out`）。

MLX 的 stream 3 有 norm=17.47，而没有 FFN 的情况下只有 norm≈4。这意味着 FFN 贡献了大约 13-14 的 norm 增量到 stream 3。

**当前状态**：native 的 FFN（MoE + shared expert）输出和 MLX 有差异。具体原因待查。

### 25.5 已确认正确的组件

| 组件 | 验证状态 | 证据 |
|------|----------|------|
| embed | ✓ 正确 | fprintf 直接打印，与 Python 计算完全一致 |
| KV cache（all 8 tokens） | ✓ cos=1.0 | two-plus-two dump 验证 |
| RoPE（consecutive） | ✓ 正确 | gen_multistep_golden.py 通过 |
| attention（attention 函数） | ✓ cos=1.0 all tokens | mla_attention_multistep_test 通过 |
| mhc_pre（attn） | ✓ cos=1.0 | Python 验证 |
| mhc_post（attn）— streams 1,2 | ✓ cos=1.0 | Python 层 0 验证（FFN=0 条件下） |
| MoE FFN | ❓ 未充分验证 | native FFN norm=23 vs MLX=32.5 |

### 25.6 下一步

1. **验证 MoE FFN 正确性**：用 `moe_isolation_test.py` 或直接比较 `L0_ffn_out_metal.bin` vs `L0_ffn_out.npy`（需要在同一个请求内 dump，不被第二个请求覆盖）
2. **运行单请求 dump**：修改 smoke test 只跑 France prompt 一个请求，然后对比 dump 数据
3. **如果 FFN 不对**：检查 expert packing、gate computation、shared expert 计算

### 25.7 实现状态

- ✅ OOM 修复：大权重 buffer 持久化（fn 1.5MB × 43 layers × 2 = 129MB），小 buffer 仍然 per-call
- ✅ RoPE 类型：已验证 consecutive 是正确的
- ✅ KV 精度：bf16 → f16 转换（无 FP8 量化）
- ✅ SDPA kernel：ds4 风格 float4 dot product
- 🔄 FFN 精度：待验证

---

## §26 MoE FFN 根因确认（2026-06-08 最终）

### 26.1 所有组件验证汇总

| 组件 | 状态 | 验证方法 | 结论 |
|------|------|----------|------|
| embed | ✅ | fprintf 直接打印 | France tok2/tok4 值与 Python 完全一致 |
| KV cache（所有 token） | ✅ | two-plus-two 8 token dump | 全部 cos=1.0 |
| RoPE 方向 | ✅ | consecutive = 正确 | 之前所有"21%误差"分析全部无效 |
| Attention | ✅ | mla_attention_multistep_test | rel_L2=0.0005，RESULT: GO |
| mhc_pre | ✅ | Python 验证 | cos=1.0 |
| mhc_post（attn） | ✅ | Python layer 0（FFN=0 时 s0/s1/s2 高） | 结构正确 |
| **MoE FFN** | ❌ | moe_isolation_test | **rel_L2=1.28，FAIL** |

### 26.2 moe_isolation_test 结果

```
[RESULT] rel L2: 1.280945e+00
[FAIL] MoE isolation: significant drift (rel_L2=1.281e+00); bug likely in MoE path
```

注：test 使用 `--metal-full`（MLX backbone + Metal MoE），不是 `--native`。`MOE_TEST_INJECT_NORMED` 在代码中未实现，test 实际是在 metal-full 模式下比较 FFN 输出与 MLX golden，但输入未必相同。

### 26.3 为什么 stream 3 在 FFN=0 时几乎为 0

从 Python 分析 FFN mhc_post 的 comb 矩阵（最后一个 prefill token）：

```
comb_ffn[k,m] (k=source stream, m=dest stream):
[[0.475  0.     0.     0.515]
 [0.     0.974  0.113  0.   ]
 [0.     0.     0.887  0.042]
 [0.524  0.026  0.     0.443]]

post_ffn = [0.001, 0.0, 0.0, 0.244]
```

Stream 3 输出 = `0.244 * ffn_out + 0.515*res[0] + 0.443*res[3]`

`post_ffn[3] = 0.244` 意味着 FFN 输出对 stream 3 贡献 24.4%，这解释了 MLX stream 3 的大 norm（17.47）主要来自 FFN 输出。如果 FFN 错误，stream 3 就会严重偏离。

### 26.4 根本原因

**MoE FFN 输出不正确**（rel_L2=1.28）。根据之前的 layer 0 后 stream 3 的 cos=0.035（FFN=0 条件），以及 moe_isolation_test 的 FAIL，确认问题在 MoE/shared expert 计算。

候选原因：
1. expert packing 格式错误（`repack_experts.py` 产生的 packed binary）
2. expert 权重读取（`io_pool_dispatch` + pread 路径）
3. `moe_forward_layer` 的 gate/up/down 计算
4. expert weights 归一化/量化错误
5. shared expert 计算错误

### 26.5 下一步诊断

验证路径：
1. **只跑 1 个 expert，比较其输出** — 用 Python 手动计算 expert 0 的 gate/up/swiglu/down，与 native dump 对比
2. **检查 packed binary 格式** — 确认 `layer_00.bin` 里 expert 0 的 gate 权重读取正确
3. **比较 gate scores** — native 的 routing scores 是否和 MLX 一致（决定路由到哪些 expert）

### 26.6 操作约束（不变）

- 只跑一个 `dmlx serve`
- MLX serve：`--smelt --smelt-strategy stream --smelt-experts 0.20 --smelt-cache 0`
- Smoke test：`NATIVE=1 bash scripts/dsv4_smoke.sh`
- 目标：`capital-of-france` 输出包含 `paris`

---

## §27 深度诊断轮次四——根因链条完整还原（2026-06-08 续）

### 27.1 本轮目标

从 §26 的「MoE FFN rel_L2=1.28」出发，逐一验证每个候选根因，还原完整 bug 链。

### 27.2 已修复的三个 Bug（本轮）

#### Bug A：FFN norm 后 `normed_bf16_direct` 未更新

**位置**：`src/metal_infer/engine.c` `moe_infer_forward_layer`

**问题**：FFN RMSNorm 计算后，只更新了 f32 `normed[]` 数组，但 `normed_bf16_direct[]`（bf16 版本）停留在**注意力 norm 的输出**（第一次 RMSNorm）。routing gate 读取 `normed_bf16_direct` 进行矩阵乘，结果是用**错误的 normed 输入**做路由。

**影响**：路由到错误的 expert，FFN 输出完全错误。

**修复**：FFN RMSNorm 后同时更新 `normed_bf16_direct`：
```c
memcpy(normed_bf16_direct, bf16_out, DIM * sizeof(uint16_t));
```

#### Bug B：`pipe_mhc_pre_bfloat` 加载了错误的 kernel

**位置**：`src/metal_infer/engine.c` engine 初始化

**问题**：
```c
eng->pipe_mhc_pre_bfloat = [lib newFunctionWithName:@"mhc_pre_gpu_f16"];  // 错！
```
`mhc_pre_gpu_f16` 将 `out_input` 截断为 **f16**（半精度），而 MLX 使用 **bf16** 截断。两种截断的误差模式不同，导致 mhc_pre 输出有系统性偏差。

**修复**：改为 `mhc_pre_gpu`（bf16 截断，与 MLX 一致）。

#### Bug C：hash routing 禁用

**位置**：`src/metal_infer/engine.c`

**问题**：`use_hash_routing = false` 导致 layers 0-2 使用 score-based routing，而 MLX 对这 3 层使用 hash routing（`tid2eid` 查表）。两者选出的 expert 不同，导致 layers 0-2 输出就有偏差。

**修复**：恢复为 `use_hash_routing = (eng->tid2eid[layer] != NULL && eng->current_token_id >= 0)`。

### 27.3 修复后验证结果

所有三个 Bug 修复后，对 layer-0 关键组件进行了逐一 Python 验证：

| 组件 | 验证方法 | 结果 |
|------|---------|------|
| MoE 路由 expert 计算（6 个 expert） | Python MXFP4 前向 vs metal dump | **cos=1.0, rel_L2=0** ✅ |
| Shared expert 计算 | Python affine 前向 vs engine debug norm | **norm=31.67 完全一致** ✅ |
| mhc_pre GPU kernel | Python CPU 实现 vs Metal GPU | **cos=1.0, rel_L2=0** ✅ |
| 专家路由（Expert IDs） | Python sqrtsoftplus + topk vs engine debug | **完全一致 [130,166,248,90,61,113]** ✅ |

### 27.4 逐层对比分析（修复后）

修复所有 3 个 bug 后，对比 MLX vs native layer residuals（最后一个 prefill token，France prompt 9 tokens）：

| 层 | MLX norm | Native norm | cos | 状态 |
|----|---------|-------------|-----|------|
| L00 | 18.82 | 18.37 | **0.9937** | ✓ 良好 |
| L01 | 17.34 | 17.01 | **0.9862** | ✓ 良好 |
| L02 | 17.22 | 17.39 | **0.9592** | ✓ 可接受 |
| L03 | 17.70 | 19.35 | **0.8659** | ✗ 首个显著下降 |
| L04 | 18.05 | 20.69 | **0.8034** | ✗ 持续恶化 |
| L17 | 55.52 | 41.39 | **0.3963** | ✗ 严重偏离 |
| L42 | 862.6 | 414.9 | **0.7412** | ✗ 大幅偏离 |

**关键发现**：L03 是首个显著发散层（cos 从 0.96 跳到 0.87）。

### 27.5 Per-stream 分析（L00 层）

L00 residual 的 4 个流对比：

| 流 | cos | 说明 |
|----|-----|------|
| stream 0 | **1.0000** | 完全一致 |
| stream 1 | **1.0000** | 完全一致 |
| stream 2 | **1.0000** | 完全一致 |
| stream 3 | **0.9927** | 轻微偏差（~0.7%）|

Stream 3 负责 FFN 输出的主要贡献（`post_ffn[3]*ffn_out`），偏差最大。

### 27.6 L03 发散的潜在原因

L03 是 `compress_ratio=128`（HCA）层，layers 2-42 都有 compressor 权重。调查发现：

**关键问题**：native engine 的 `moe_infer_compressor_step` 会在每个 position 积累 compressed KV blocks，但当序列长度小于 `SWA_WINDOW=128` 时，**MLX 不执行 KV 压缩**（`compressKV` 在 `prefix_len < 0` 时跳过）。

修复方向：`use_comp = (n_comp > 0 && kvc->len > SWA_WINDOW)` — 只在 raw KV 溢出滑动窗口时才使用压缩块，短序列走纯 bf16 路径。

但该修复后 L03 仍然 cos=0.87，说明还有其他因素。

### 27.7 当前诊断边界

- **Layer 0 stream 3 cos=0.9927**：偏差在 mhc_post 计算，stream 3 的 `post_mix[3]` 和 `comb` 系数与 MLX 不同
- **L03 开始发散**：L02（ratio=4 CSA）是第一个有 compressor 的层，L03 累积了 L02 的误差
- **误差来源初步定位**：mhc_pre 的 `post_mix` 和 `comb` 矩阵与 MLX 实际有偏差

### 27.8 下一步行动

**最高优先级**：直接对比每层 mhc_pre 的 `post` 和 `comb` 输出，确认是否与 MLX 一致。

具体方案：
1. 在 engine.c layer-0 的 mhc_pre 中 dump `post[]` 和 `comb[]` 到文件
2. 同时在 MLX 路径中 dump 同一 token 的 mhc_pre 输出
3. 比较两者是否一致

**次优先级**：确认 `mhc_pre_bfloat` kernel（现在是 `mhc_pre_gpu`）的输出是否有 bf16 截断误差导致 stream 3 的 0.73% 偏差。

### 27.9 操作约束（不变）

- 只跑一个 `dmlx serve`
- MLX serve：`--smelt --smelt-strategy stream --smelt-experts 0.20 --smelt-cache 0`
- Native smoke test：`NATIVE=1 bash scripts/dsv4_smoke.sh`
- 测试 max-tokens：**30**（正确性测试）
- 目标：`capital-of-france` 输出包含 `paris`

### 27.10 代码变更汇总

| 文件 | 变更 | 目的 |
|------|------|------|
| `src/metal_infer/engine.c` | FFN norm 后更新 `normed_bf16_direct` | Bug A 修复 |
| `src/metal_infer/engine.c` | `pipe_mhc_pre_bfloat` 加载 `mhc_pre_gpu` | Bug B 修复 |
| `src/metal_infer/engine.c` | `use_hash_routing` 恢复条件判断 | Bug C 修复 |
| `src/metal_infer/engine.c` | `use_comp = (n_comp>0 && kvc->len>SWA_WINDOW)` | 短序列不使用压缩块 |
| `src/metal_infer/engine.c` | 混合 attention 路径（`mla_attention_decode_mixed`）接入 | 长序列使用压缩 KV |

---

## §28 根本原因完整链条（2026-06-08 最终诊断）

### 28.1 总结

Native 引擎输出 "to"（token 304），MLX 输出 "."（token 16），最终正确续写包含 "Paris"。根本原因是**量化精度路径差异**，经过 mHC 矩阵放大后导致 logits 偏差 1.33，最终 argmax 选到错误 token。

### 28.2 完整因果链

```
1. affine 4-bit shared expert 精度差异
   ├─ MLX GPU quantizedMatmul (bf16路径) → shared_out norm = 32.5
   └─ Native CPU dequant_matvec_affine → shared_out norm = 31.7
   └─ 差异: 2.5% (0.81/32.5)
   
2. ffn_out 精度差异
   └─ ffn_out = moe_only + shared_out
   └─ moe_only (mxfp4) 精度完全正确 (cos=1.0)
   └─ shared_out 差 2.5% → ffn_out 差 2.5%
   
3. mHC 矩阵放大
   └─ Layer 0 FFN mhc_pre: post_ffn[3] = 0.537
   └─ Stream 3 = 0.537 × ffn_out + comb × residual
   └─ Stream 3 从 MLX 的 17.47 缩小到 native 的 16.99
   └─ cos = 0.9927 (0.7% 误差)
   
4. 层间累积放大
   └─ L01: stream 2 变坏 (post_ffn[2]=0.578, cos=0.90)
   └─ L03: stream 1 崩溃 (cos=0.43)
   └─ L17: cos=0.35 (完全发散)
   └─ 最终 logits 偏差 1.33 → argmax 选到 token 304 而非 16
```

### 28.3 关键验证

| 组件 | 验证方法 | native 结果 | MLX 结果 | 差异来源 |
|------|---------|------------|---------|---------|
| mxfp4 routed expert | Python CPU vs native | **cos=1.0** | — | 无差异 |
| affine shared expert | Python CPU vs native debug | **31.67=31.67** | 32.51 | **MLX quantizedMatmul vs CPU dequant** |
| mhc_pre | GPU vs CPU | **完全一致** | — | 无差异 |
| KV cache | cos 验证 | **cos=1.0** | — | 无差异 |
| attention (单步) | mla_attention_test | **rel_L2=1.9e-6** | — | 无差异 |
| embedding | bf16 broadcast | 正确 | 量化路径稍不同 | 极小 (<0.01%) |

### 28.4 关于 MLX residual_in 流不相同

MLX `L0_residual_in.npy` 的 4 个 HC 流不完全相同（差 ~1.9%），这**不是** native 需要复现的"正确行为"。ds4 和 native 均使用相同的 embed broadcast（4 流完全一致），这是正确的初始化方式。MLX 的差异来自其内部 batch quantized matmul 的特殊数值行为。

### 28.5 修复方向

#### 方向 A：接受精度差异，调整 mHC 参数
- mHC 的放大机制是设计如此（doubly stochastic constraint）
- 问题在于 `post_ffn[3]=0.537` 对误差过于敏感
- **不可行**：无法改变模型权重

#### 方向 B：提升 shared expert 计算精度（推荐）
- 在 native 引擎中对 shared expert 使用与 MLX 一致的计算路径
- 方案：用 MLX 的 `quantizedMatmul`（bf16 matmul）替代当前 CPU affine dequant
- 或：在 `dequant_matvec_affine` Metal kernel 中加入 bf16 中间计算
- **预期效果**：消除 2.5% shared expert 误差，stream 3 cos 0.9927 → 接近 1.0

#### 方向 C：对 shared expert 使用 bf16 中间累积（最小改动）
当前 kernel 用 f32 累积；改成先 bf16 量化 matmul 再累积，模拟 MLX 的行为
- 改动点：`dequant_matvec_affine` kernel，添加 bf16 中间版本

#### 方向 D：直接使用 MLX 计算 shared expert（过渡方案）
在 native 引擎的 MoE 部分，对 shared expert 回落到 MLX 计算
- 缺点：打破 native 纯 C/Metal 架构
- 优点：立即解决精度问题，最小代码改动

### 28.6 代码状态（2026-06-08）

**已修复的 bug（本轮）**：
1. FFN norm 后 `normed_bf16_direct` 未更新 → routing gate 用了错误的 normed
2. `pipe_mhc_pre_bfloat` 加载了 f16 截断 kernel → 改为 bf16 截断的 `mhc_pre_gpu`
3. `use_hash_routing = false` → 恢复条件判断
4. 批量 prefill 中 hash routing 用了最后一个 token 的 ID → `forwardBatch` 传入 token_ids 数组

**当前构建状态**：clean build，smoke test 输出 "to the question..." → 仍 FAIL

**sinkhorn 实现**：已验证与论文/MLX/ds4 等价（max diff < 4e-5）

**mhc_post comb 索引**：已验证与 MLX/ds4 一致（`comb[k, m]` = src k → dst m）

### 28.7 下一步

优先执行方向 C 或 D（最小改动，最快解决）：
1. **方向 D（立即测试）**：在 `moe_infer_forward_layer` 的 shared expert 部分，改为调用 MLX `quantizedMatmul` 
2. **方向 C（中期）**：修改 `dequant_matvec_affine` kernel，添加 bf16 中间路径匹配 MLX

关键约束（不变）：
- 只跑一个 `dmlx serve`
- Native smoke test：`NATIVE=1 bash scripts/dsv4_smoke.sh`
- 目标：`capital-of-france` 输出包含 `paris`

---

## §29 根本 Bug 修复：MXFP4 E8M0 Scale Bias 错误（2026-06-08 终结）

### 29.1 发现过程

通过直接用 MLX 的 `quantized_matmul(mode='mxfp4')` 计算 layer-0 单 token 路由 expert 输出：
- MLX mxfp4 routed norm = **4.51**
- Native packed_experts CPU 计算 routed norm = **0.58**（差了 7.8×）

进一步测试发现所有 nibble 值 MLX 的结果是 native 的 2 倍（对于单层 gate/up/down）。

### 29.2 根本原因

**MXFP4 E8M0 scale 的 bias 用错了**：

| 来源 | 公式 | Bias |
|------|------|------|
| MLX `fp8_e8m0`（`fp_quantized.h`） | `scale = 2^(byte - 127)` | **127** |
| Native engine（`moe_kernel.metal`）| `scale = exp2(byte - 128)` | **128** |

每个 scale 相差 `2^(127-128) / 2^0 = 2^(-1)` vs `2^0`... 不对，应该是：
- MLX: `2^(s - 127)` 
- Native: `2^(s - 128) = 2^(s-127) / 2`

所以 native 的每个 scale 比 MLX **小 2 倍**，每个 expert 输出小 2 倍，经过 SwiGLU 非线性复合后 moe_only 输出小 7.8 倍（gate/up/down 每级 ×2，但 SwiGLU 的截断改变了放大比例）。

### 29.3 修复

将 `src/models/moe_kernel.metal` 中所有 MXFP4 kernel 的 scale 计算从：
```metal
float sf = exp2((float)sc[g] - 128.0f);
```
改为：
```metal
float sf = exp2((float)sc[g] - 127.0f);
```

影响的 kernel：
- `fused_gate_up_swiglu`（gate/up scale）
- `dequant_matvec_4bit`（down scale）
- `fused_gate_up_swiglu_bfloat_in`
- `dequant_matvec_4bit_bfloat_in`
- 以及所有其他 mxfp4 kernel 变体

### 29.4 验证结果

修复后 `NATIVE=1 bash scripts/dsv4_smoke.sh`：

```
✓ capital-of-france: 'The capital of France is **Paris**.'
✓ two-plus-two: '4'
SMOKE PASS
```

### 29.5 为什么之前没发现

之前的 moe_isolation_test 和验证脚本（`repack_experts.py`, `verify_mxfp4_gate.py` 等）都**一致地**使用了 bias=128，所以 Python 和 native 的结果总是匹配（两边同样错误）。只有当与 MLX 的 `mode='mxfp4'` 直接比较时才能发现差异。

### 29.6 对架构的影响

此 bug 修复：
1. 完全解决了 native 引擎的正确性问题
2. 所有之前"cos=1.0"的验证结论仍然有效（Python 和 native 都用了同样的 bias=128）
3. 之前分析的"MLX 批量 GPU 精度差异"结论是错误的，实际上精度差异来自 scale bias 错误
4. packed_experts 里的数据是正确的（MLX 导出时用 fp8_e8m0 bias=127），native 解释时用了错误的 bias

### 29.7 后续工作

- `repack_experts.py` 和 `verify_mxfp4_gate.py` 中的 `exp2(scale - 128.0)` 需要改为 `exp2(scale - 127.0)` 以保持一致性（虽然这些是验证脚本，不影响推理）
- 注意：**affine 量化**（attn/shared expert/lm_head）的 scales 是 BF16 格式，不受此 bug 影响
- 只有 **MXFP4 的 U8 E8M0 scales** 受影响（`switch_mlp.{gate,up,down}_proj.scales`）

---

## §30 Native Engine 正确性诊断方法论（完整手册）

> 本节将整个排查过程提炼为可复用的方法，供未来调试类似问题时参考。

### 30.1 诊断分层原则

正确性问题的黄金诊断顺序：**从最小单元向上，每层用 Python/MLX 对拍，cos≥0.999 才认为通过。**

```
Level 1: 单 kernel 数值（Python 复现 Metal kernel 算法）
Level 2: 单组件对拍（mhc_pre、attention、moe_forward）
Level 3: 单层 E2E 对拍（layer_00_residual vs MLX layer_00.npy）
Level 4: 全层 per-stream 分析（找发散首层和发散流）
Level 5: 逐段追因（attn_input → attn_out → after_attn → ffn_normed → ffn_out）
```

### 30.2 必须区分的 Dump 数据性质

| Dump 文件 | 写入时机 | 对应哪个 token/pos |
|-----------|---------|-----------------|
| `L00_residual_last.bin` | `moe_infer_forward_batch` 每层结束 | prefill 最后一个 token |
| `L0_attn_out_metal.bin` | `MF_DBG` 每次 `moe_infer_forward_layer` | 最后一次写入（可能是 decode step）|
| `L0_normed_ffn_in.bin` | `MF_DBG` | 最后一次写入 |
| MLX `layer_00.npy` | `activation_dump.dump` | 所有 prefill tokens [1,9,4,4096] |
| MLX `L0_ffn_out.npy` | `activation_dump.dump` | 所有 prefill tokens [1,9,4096] |

**黄金规则**：只在 **相同 prompt、相同 position** 的数据之间做比较。

### 30.3 MXFP4 Quantization 快速验证 Checklist

当 native MoE 输出和 MLX 不一致时，先做这个 3 步 check：

```python
# Step 1: 单 expert 验证（判断 mxfp4 kernel 是否正确）
import mlx.core as mx, numpy as np

# 1a. 用 MLX 直接算 expert eid 的输出
x = mx.array(normed_np).astype(mx.bfloat16).reshape(1, -1)
g_out = mx.quantized_matmul(x, g_w, g_s, transpose=True, group_size=32, bits=4, mode='mxfp4')
# 1b. 用 Python CPU 算同一 expert
sf = np.exp2(float(scale_byte) - 127.0)  # ← 注意 bias=127
# 比较两者 cos

# Step 2: 如果单 expert 差 2x，检查 scale bias
# MLX fp8_e8m0: scale = 2^(byte - 127)
# 正确代码: exp2(scale - 127.0f)
# 错误代码: exp2(scale - 128.0f) → 每个值小 2x

# Step 3: 比较 moe_only norm
# native debug: [mf-dbg] L0 ffn_out(moe only) norm=X
# MLX 单 token: float(mx.sqrt(mx.sum(routed[0].astype(mx.float32)**2)))
# 两者应 cos≥0.999
```

### 30.4 诊断 Dump 生成标准流程

```bash
# 1. 生成 MLX golden（1 请求，max-tokens=1）
pkill -f "dmlx serve"
mkdir -p /tmp/mlx_ref
DSV4_DUMP_DIR=/tmp/mlx_ref ./zig-out/bin/dmlx serve \
  --model ~/models/DeepSeek-V4-Flash-4bit \
  --port 8930 --max-tokens 1 --temperature 0 \
  --smelt --smelt-strategy stream --smelt-experts 0.20 --smelt-cache 0 \
  --expert-packed-dir ~/models/DeepSeek-V4-Flash-4bit/packed_experts &
# 等待 ready 后发 France prompt
curl -s ... | python3 -c "print(json.load(sys.stdin)['choices'][0]['message']['content'])"
pkill -f "dmlx serve"

# 2. 生成 native dump（同样配置）
mkdir -p /tmp/native_ref
MF_DBG=1 DSV4_DUMP_DIR=/tmp/native_ref ./zig-out/bin/dmlx serve \
  --native --expert-packed-dir ... --port 8930 --max-tokens 1 &
# 发同一请求
pkill -f "dmlx serve"

# 3. 比较
python3 - << 'EOF'
import numpy as np, os
mlx_d, nat_d = "/tmp/mlx_ref", "/tmp/native_ref"
for i in range(5):
    mf = f"{mlx_d}/layer_{i:02d}.npy"
    nf = f"{nat_d}/L{i:02d}_residual_last.bin"
    if not (os.path.exists(mf) and os.path.exists(nf)): continue
    m = np.load(mf).astype(np.float32).reshape(-1,4,4096)[-1].ravel()
    n = np.fromfile(nf, dtype=np.float32).ravel()
    cos = np.dot(m,n)/(np.linalg.norm(m)*np.linalg.norm(n)+1e-10)
    print(f"L{i:02d}: cos={cos:.4f}")
EOF
```

### 30.5 已知 Bug 模式库

| 症状 | 根因 | 修复位置 | 修复方式 |
|------|------|---------|---------|
| moe_only norm 比 MLX 小 7-8× | MXFP4 scale bias=128（应为 127） | `moe_kernel.metal` | `exp2(s - 128)` → `exp2(s - 127)` |
| routing 选错 expert（prefill） | hash routing 用了最后一个 token ID | `native_engine.zig` | `forwardBatch` 传入每 token 的 token_ids |
| attn_out 正确但 stream 发散 | mhc_pre kernel 截断精度 | `engine.c` | 改用 `mhc_pre_gpu`（bf16 截断）而非 `mhc_pre_gpu_f16`（f16 截断） |
| FFN 输出为 0 | `buf_normed` 未写入 | `engine.c` | 写入 `buf_normed` 再 dispatch MoE |
| 路由 gate 用了 attn normed | `normed_bf16_direct` 未在 FFN norm 后更新 | `engine.c` | FFN RMSNorm 后同步更新 `normed_bf16_direct` |
| prefill 用了 wrong KV layout | `generate()` 走 batch prefill 喂给单 token 引擎 | `state.zig` | metal-full 时改 token-by-token |

### 30.6 关键精度对齐事实

| 组件 | native vs MLX | 注意事项 |
|------|-------------|---------|
| MXFP4 routed experts | ✅ cos=1.0（修 bias 后）| scale bias=127（fp8_e8m0），非 128 |
| Affine shared expert | ✅ cos=1.0 | BF16 scales，bias 不适用 |
| mhc_pre | ✅ GPU=CPU=Python | F32 输入，BF16 截断输出 |
| mhc_post | ✅ Python=MLX | comb 是 [src,dst] row-major |
| KV cache | ✅ cos=1.0 all tokens | BF16 存储，f16 写入 |
| Attention | ✅ rel_L2=1.9e-6 | 单步验证，多步累积误差正常 |
| Sinkhorn | ✅ max diff<4e-5 | 与论文/MLX/ds4 算法等价 |

### 30.7 下次出问题先看这里

**正确性问题 5 分钟初诊**：

```
Q1: smoke test 输出什么？
  - 退化（重复）→ 检查 mHC/RoPE
  - 完全随机 → 检查 embed/KV cache
  - 接近但错 token → 检查 MoE 输出量级

Q2: 逐层 cos 哪层开始发散？
  - L00 就很低 → embed/mhc_pre/attention 基础问题
  - 某层突然跳低 → 该层特有结构（CSA/HCA/hash routing）
  - 线性衰减 → 量化精度累积误差

Q3: Per-stream cos：哪个 stream 先发散？
  - stream 3（大 norm 流）→ FFN mhc_post 放大 FFN 误差
  - stream 1 在某层跳低 → post_ffn[1] 大，FFN 误差被放大

Q4: FFN moe_only norm 和 MLX 相比？
  - 比 MLX 小 7-8× → MXFP4 scale bias 错（128→127）
  - 比 MLX 小 2× → 某个 projection 的 scale 错
  - cos≈0 → expert weights 读取顺序/格式错

Q5: Shared expert 和 moe_only 分开验证？
  - Shared cos≈1 但 total cos低 → moe_only 有问题
  - 两者都低 → normed 输入有问题
```

### 30.8 repack_experts.py 注意事项

`scripts/repack_experts.py` 生成 `packed_experts/layer_XX.bin`。里面的 U8 scales 是直接从 safetensors 的 U8 E8M0 scales 复制的，**不做任何偏置转换**。native engine 读取时需要用 bias=127 解码（`exp2(scale - 127.0)`）。

若发现验证脚本（`verify_mxfp4_gate.py` 等）仍用 `exp2(scale - 128.0)`，修改如下：
```python
# 错误
sf = 2.0 ** (float(scales[g]) - 128.0)
# 正确
sf = 2.0 ** (float(scales[g]) - 127.0)
```

---

## §31 状态总结（2026-06-08 最终）

### 31.1 最终状态

| 指标 | 值 |
|------|-----|
| native smoke test (France + 2+2) | ✅ **2/2 PASS** |
| 根本 Bug | ✅ 已修复（MXFP4 scale bias 128→127） |
| 默认模式 | **`--native`**（dsv4_smoke.sh 已更新） |
| 构建 | ✅ clean `zig build -Doptimize=ReleaseFast` |

### 31.2 操作规范（最终版）

```bash
# Smoke test（默认 native 模式）
bash scripts/dsv4_smoke.sh

# MLX oracle 对照
NATIVE=0 bash scripts/dsv4_smoke.sh

# Serve（生产用）
./zig-out/bin/dmlx serve \
  --model ~/models/DeepSeek-V4-Flash-4bit \
  --port 8080 \
  --expert-packed-dir ~/models/DeepSeek-V4-Flash-4bit/packed_experts \
  --native
```

### 31.3 已修复 Bug 完整列表

| # | Bug | 文件 | 修复 |
|---|-----|------|------|
| A | FFN norm 后 `normed_bf16_direct` 未更新 | `engine.c` | FFN RMSNorm 后同步更新 |
| B | `pipe_mhc_pre_bfloat` 加载了 f16 kernel | `engine.c` | 改为 `mhc_pre_gpu`（bf16）|
| C | hash routing 用了最后 token ID | `native_engine.zig` | `forwardBatch` 传 token_ids[] |
| D | **MXFP4 scale bias=128（根本 Bug）** | `moe_kernel.metal` | `exp2(s-128)` → `exp2(s-127)` |

### 31.4 诊断参考

完整诊断方法论见 §30，skill 文件见 `.kiro/steering/native-engine-debug.md`。

---

## §32 Native Engine 性能优化计划（2026-06-08）

### 32.1 当前性能基准

| 指标 | 值 |
|------|-----|
| 实测吞吐 | ~0.13-0.15 tok/s（5 token 请求约 33-40s）|
| 上一版 MLX 路径 | ~1.0 tok/s（warm，Trust OS）|
| 目标 | ≥3.0 tok/s |
| 差距 | 20-23×（vs 目标），6-7×（vs 旧 MLX）|

### 32.2 性能瓶颈分析

通过代码计数，每个 decode token 需要：

| 开销来源 | 数量 | 估计时间 | 原因 |
|---------|------|---------|------|
| `waitUntilCompleted` | **387 次**（9/层 × 43 层）| ~0.4s | 每次 Metal dispatch 同步等待 |
| `newBufferWithBytes*` MTLBuffer 分配 | **1419 次**（33/层 × 43 层）| ~0.7s | 每次 kernel 都临时分配 buffer |
| SSD I/O（6 experts × 13.4MB × 43 层）| 3.46 GB/tok | ~2.3s | expert 权重读取 |
| Metal compute | 43 层 × ~6ms | ~0.26s | 实际 GPU 计算 |
| 其他（memcpy、autorelease）| — | ~0.5s | 内存复制、ObjC 开销 |

**核心问题**：
1. **1419 次 MTLBuffer 临时分配/释放** — 每个 kernel 调用都重新分配缓冲区
2. **387 次 GPU 同步屏障** — 每次 dispatch 后立即 `waitUntilCompleted`，GPU 空转等待 CPU
3. **6 个 expert 串行 dispatch** — 可并行却串行执行

### 32.3 优化路线图（参考 flash-moe + ds4）

#### P0：持久化 MTLBuffer（最大收益，最简单）

**原理**：把所有固定大小的临时 buffer 改为持久化分配，消除 1419 次 alloc/dealloc。

**当前代码（每次调用都分配）**：
```objc
// 在 moe_infer_forward_layer 里，每层每次：
id<MTLBuffer> bx = [d newBufferWithBytes:attn_input length:DIM*sizeof(float) ...];
// ...用完就释放
```

**目标代码（持久化）**：
```objc
// 初始化时一次性分配，存在 eng->buf_* 里
eng->buf_attn_input = [d newBufferWithLength:DIM*sizeof(float) ...];
// 每次使用时直接 memcpy 进去
memcpy([eng->buf_attn_input contents], attn_input, DIM*sizeof(float));
```

已有的持久化 buffer 示范：`eng->buf_attn_hc_fn[layer]`（大权重已经持久化了），但小的 scratch buffer 还没有。

**预期收益**：~0.7s/tok → ~0.4 tok/s（+2.5×）

**工作量**：2 天（主要是 engine.c 里的 `moe_infer_forward_layer` 函数）

#### P1：6 Expert 并行 Dispatch

**原理**：把 6 个 expert 的 gate/up/swiglu/down 从串行改为单个 command buffer 内并行。

**当前（`moe_forward_layer`）**：
```objc
id<MTLCommandBuffer> cb = [...commandBuffer];
for (int k = 0; k < K; k++) {
    // gate+up+swiglu kernel for expert k
    [enc endEncoding];
}
[cb commit]; [cb waitUntilCompleted];  // 等待所有 expert 完成
```

**目标**：已经是单 cb 里 6 个 encoder，只需一次 `waitUntilCompleted` — 这部分其实已经做了！真正的问题是 **每个 encoder 前还要分配 buffer**。

**工作量**：与 P0 合并

#### P2：合并 Command Buffers（减少 GPU 同步）

**原理**：每层当前有 9 个独立的 command buffer + wait。应该合并成 2-3 个：
1. **CB-ATT**：mhc_pre_attn + attn_norm + attention + mhc_post_attn（可以串联，无中间 CPU 依赖）
2. **CB-FFN**：mhc_pre_ffn + ffn_norm + routing_gate（可以串联）
3. **CB-MOE**：expert forward（已有，等 I/O 完成后 dispatch）

**当前 9 次 wait**：
1. mhc_pre_gpu (attn)
2. input_RMSNorm
3. mhc_post (attn)
4. mhc_pre_gpu (ffn)
5. ffn_RMSNorm
6. routing_gate
7. moe_forward (expert kernels)
8. shared_expert (gate+up)
9. shared_expert (down)

**目标 3 次 wait**：CB1（mhc_pre+norm）→ 等 I/O →CB2（moe+shared）→ CB3（mhc_post_ffn）

**预期收益**：减少 6 次不必要的 wait（~6ms/层 × 43 = ~0.26s/tok）

**工作量**：3-4 天

#### P3：Deferred Expert Forward（flash-moe CMD3 技术）

**原理**：expert forward（最重的 GPU 计算）不 `waitUntilCompleted`，让 GPU 异步执行，CPU 立即处理下一层。下一层开始前只需同步。

```
Layer N:
  CPU: routing → I/O pread experts → dispatch CMD3 (no wait!)
  GPU: (running CMD3 for layer N)
  CPU: immediately starts layer N+1 routing
  ...
  CPU: before dispatching layer N+1 CMD3 → wait for layer N CMD3
```

这是 flash-moe 的核心性能技术之一。

**注意**：V4 expert 局部性只有 35%，时序预测预取对 V4 收益有限（flash-moe 文档已证明）。但 deferred 本身不依赖预测，只是让 GPU 和 CPU 并行工作。

**预期收益**：GPU 计算（~0.26s/tok）几乎完全 overlap

**工作量**：3 天（需要修改 `moe_forward_layer` 和层循环结构）

#### P4：Expert I/O 优化

**现状**：`io_pool_dispatch` 每层 6 个 expert 并行 pread，有线程池。但：
1. pread 后数据在 CPU 内存 → `newBufferWithBytesNoCopy` 避免复制（已有）
2. OS page cache 是主要手段

**潜在优化**：
- 预分配 `Metal shared memory` 用于 expert data（当前是 `posix_memalign`，已是对齐的）
- 使用 `F_NOCACHE` flag 避免 page cache 污染（flash-moe 文档提到，但对 V4 效果不确定）

**工作量**：1-2 天试验

### 32.4 实施优先级

```
P0 持久化 MTLBuffer        → +2.5×  (0.13 → ~0.33 tok/s)  ★★★★★
P1+P2 合并 CommandBuffer   → +1.5×  (0.33 → ~0.50 tok/s)  ★★★★
P3 Deferred CMD3           → +1.3×  (0.50 → ~0.65 tok/s)  ★★★
P4 I/O 优化                → +1.1×  (0.65 → ~0.72 tok/s)  ★★
```

**综合预期**：0.13 → ~0.7-0.8 tok/s（5-6× 提升）

### 32.5 与 flash-moe / ds4 对比的结构性差距

即使做完所有优化，native 仍与 flash-moe 4.36 tok/s 有差距：

| 差距来源 | flash-moe | native (目标) | 说明 |
|---------|-----------|--------------|------|
| Expert I/O 量 | 43×K=4×~11MB ≈ 1.9GB/tok | 43×K=6×13.4MB ≈ 3.46GB/tok | V4 更多 expert，更大权重 |
| SSD 速度 | M3 Max ~5GB/s | M4 Pro ~3-4GB/s | 硬件差异 |
| 模型大小 | 60 层（Qwen GQA） | 43 层（V4 MLA）| 层数不同，但单层更重 |
| Expert 局部性 | 71% hit rate | ~35% hit rate | V4 路由更分散 |

**理论最大值**（I/O 限制）：3.46 GB × 1/(3.5 GB/s SSD) = ~1s/tok = ~1 tok/s

**3 tok/s 的路径**：需要提升 expert 局部性（缓存 hot experts）或降低每 token I/O 量（SMELT 更激进预加载 + OS page cache 预热）

### 32.6 近期可执行的最快修复

**立即可做（1 天工作量，预计 2-3× 提升）**：

在 `moe_infer_forward_layer` 里，把所有 `[d newBufferWithBytes:... length:N*sizeof(float) options:MTLResourceStorageModeShared]` 改为使用预分配的持久化 buffer。

主要改动点（engine.c）：
1. `buf_ain`（attn_input, DIM float） → 已有 `eng->buf_mhc_attn_in` 未使用
2. `buf_post`, `buf_comb2`（mhc output） → 已有 `eng->buf_mhc_post_weights`, `eng->buf_mhc_comb_weights`
3. `bx`（attn_input for RMSNorm）→ 复用 `buf_ain`
4. `bfx`（ffn_input for RMSNorm）→ 已有 `eng->buf_mhc_ffn_in`
5. `bx_bf16`（gate routing input）→ 可用 `eng->buf_mhc_attn_norm_bf16`

这些 buffer 在 `init_metal` 里已经分配了，但 `moe_infer_forward_layer` 里没有使用！

### 32.7 实测更新（2026-06-08）

P0 优化（持久化 MTLBuffer）实施后，**性能无明显变化**：仍约 0.14 tok/s。

**原因**：真正的瓶颈是 **SSD I/O**，而非 GPU 内存管理。

| 指标 | 值 |
|------|-----|
| 每 token I/O 量 | 3.46 GB（43×6×13.4MB） |
| 实测有效 SSD 带宽 | ~0.54 GB/s（远低于额定 5-6 GB/s）|
| 理论每 token 时间 | 3.46/0.54 ≈ 6.4s |
| 实测每 token 时间 | ~6.2s（吻合！）|
| 8 次预热后变化 | 无改善（page cache 无法保存 3.46GB × N）|

**根本原因**：native 引擎每层 pread 6 个 expert，每 token 读 258 次，共 3.46GB。OS page cache 在 48GB 系统中无法长期保留，每次都是冷读取。

### 32.8 正确的优化方向

**最高优先级：Expert 内存缓存（对应 MLX 的 SMELT）**

MLX 路径之所以能达到 ~1 tok/s，核心原因是 `--smelt-experts 0.20` 预加载了 20% 的 expert（~51 个）到内存，极大减少了 SSD 读取次数（只在 cache miss 时才读 SSD）。

native 引擎当前完全没有 expert 缓存！需要添加：

1. **Expert 内存池**（最小版本）：启动时预分配 N GB RAM，预读取最常用的 experts（可以用 hash routing 的固定 expert 列表作为种子）
2. **LFU/LRU 缓存**：复用已有的 `expert_cache.zig` 逻辑，在 native engine 层面接入
3. **Hash routing 层（0-2）expert 预固定**：这 3 层用 hash routing，expert 完全确定，可以永久缓存（每层只需 6 × 13.4MB = 80MB × 3 = 240MB）

**具体实施路径**：

```
A. 最小改动（1-2 天）：
   - 在 moe_infer_init 时，对 hash routing layers 0-2 直接 preload 所有选中的 experts 到内存
   - 对 decode 时频繁出现的 experts 建立简单 LRU（16GB 可缓存 ~1200 个 expert）
   
B. 完整优化（1 周）：
   - 移植 expert_stream.zig 的 IOPool + LFU cache 到 native engine
   - 支持 --smelt-cache N 参数
   - Expert 预测：上一个 token 的 routing 结果用于异步 pread 下一层
```

**预期收益**：
- hash routing 3 层完全缓存：节省 3/43 ≈ 7% I/O
- 20% expert 缓存：cache hit ~60-70%（类似 MLX SMELT）→ 约 3× I/O 减少 → ~0.4 tok/s
- 60% expert 缓存：大部分命中 → ~1 tok/s（接近旧 MLX）

**关键数字**：
- 48GB RAM - ~8GB 模型 = ~40GB 可用
- 256 experts × 13.4MB = 3.43GB 可以全量缓存！
- 全量缓存所有 expert：每 token 从 RAM 读 3.46GB @100 GB/s = 35ms/tok = ~29 tok/s

### 32.9 缓存策略分析与修正（2026-06-08）

#### 尝试结果

1. **P0：持久化 MTLBuffer** — 无明显收益（~0.14 tok/s，与之前相同）
   - 原因：I/O 才是瓶颈，不是 GPU 内存分配

2. **10GB Expert 缓存（random 18 experts/layer）** — 无明显收益
   - 原因：18/256 = 7% cache hit rate，score-based routing 路由到这 18 个的概率极低

#### 性能约束数学

```
每 token I/O = 43 层 × 6 experts × 13.4 MB = 3.46 GB
实测有效带宽 ≈ 0.54 GB/s（OS page cache 无法保持，SSD 冷读）
理论时间 = 3.46 / 0.54 = 6.4 s/tok → 0.16 tok/s ← 与实测吻合！
```

这是**硬件/模型结构约束**，不是软件 bug。

#### 为什么 MLX 路径能达到 1.02 tok/s？

MLX 用 `--smelt-experts 0.20`（预加载 20% expert = 51个）+ streaming 模式：
- 51 个常用 expert 在内存（~680MB），每次访问这些时**零 I/O**
- 只有 cache miss 才读 SSD
- 实测 hit rate ~23-27% → 有效 I/O 减少到 ~2.5 GB/tok → ~0.54/2.5 × 0.16 ≈ 不对...

实际上：MLX SMELT 加速来自**只加载了 20% 的 expert + 路由 bias 让它避开未加载的 expert**。这使得每层实际参与的 expert 变少，减少了 I/O。

#### 正确的优化方案

**方案 A：All-expert 内存缓存（全量，理论最优）**
- 256 experts × 43 layers × 13.4MB = **147 GB** — 无法在 48GB 机器上实现

**方案 B：全量缓存单层（最实用的完整优化）**
- 每层的 256 experts = 256 × 13.4MB = 3.43 GB
- 40GB 可用 → 可以完整缓存 **11-12 层**
- 这 12 层的 I/O 为 0，其他 31 层仍读 SSD
- 预期收益：时间 = (31/43 × 6.4s) + 0 = 4.6s/tok ≈ 0.22 tok/s（+40%）

**方案 C：SMELT 等价——只缓存 top-K 最常用的 expert（最优性价比）**
- 目标：每层缓存 N 个，使 routing 命中率 > 60%
- 需要统计：做若干次推理，记录每层被选中的 expert 频率，缓存最热的 N 个
- 预期：10GB 缓存 18 个/层，如果是热门 expert 则 hit rate ~60% → 有效 I/O 降到 ~40% → 速度翻 2.5×

**方案 D：ds4 的方案——时序预测 + 双缓冲**
- 用上一个 token 的 expert 选择来预测下一个 token 的 expert
- V4 命中率约 35%（历史记录），比随机好 35/7% ≈ 5× 对于 18 个缓存
- 但 ds4 文档指出 V4 局部性很低，对 V4 效果有限

**立即可实施的最优方案（2 天工作量）**：

1. 收集 token routing 统计（在 forward 循环中记录每层最常用的 expert）
2. 按频率预加载最热的 N 个 expert 到每层缓存
3. 在后续 token 中命中缓存时直接使用 RAM 数据

预期能从 0.14 tok/s → 0.5-1.0 tok/s（3-7× 提升）。

---

## §33 Native Engine 性能差距根因完整分析（2026-06-08）

### 33.1 性能对比

| 路径 | tok/s | 方案 |
|------|-------|------|
| 旧 MLX 路径（最优配置） | **~1.02** | `--smelt --smelt-experts 0.20 --smelt-cache 0` |
| Native engine（当前） | **~0.14** | `--native` |
| flash-moe 参考 | **4.36** | Trust OS，手写 Metal kernel |
| **目标** | **≥3.0** | — |

**差距**：native vs MLX = 7×；native vs 目标 = 21×。

### 33.2 旧 MLX 路径为什么能到 1 tok/s

来自 `flash-moe-alignment-analysis.md` 实测（commit f9b3cd8，Trust OS + packed experts）：

1. **SMELT 0.20**：预加载 51/256 个最常用 expert 到内存，命中时**零 I/O**
2. **Trust OS（cache=0）**：不用自定义 LFU cache，依赖 OS page cache，避免 VM 抖动
3. **DyMoE Skip**：每步 skip 1/6 低分 expert，减少 ~17% I/O，实测 +19-57% client 延迟改善
4. **结果**：有效 I/O 从 3.46 GB/tok 降到约 1-2 GB/tok，配合 OS page cache → ~1 tok/s

### 33.3 Native Engine 为什么只有 0.14 tok/s

**完全没有任何 I/O 优化**：
- 没有 SMELT：每层每次 pread 全部 6 个 expert（3.46 GB/tok）
- 没有 DyMoE Skip：不跳过低分 expert
- 没有 OS page cache 利用：每次请求都是冷读

**还有 GPU 同步串行问题**：
- 每层有 9 次 `waitUntilCompleted`（9 个 command buffer 全部同步等待）
- flash-moe 的 Deferred CMD3 完全没有实现
- GPU 和 CPU 完全串行，没有任何 overlap

### 33.4 flash-moe 的三个核心技术（文档 §9 + flash-moe-alignment-plan.md §2）

#### 技术 A：Deferred CMD3（GPU/CPU 并行）

```
layer N:
  CPU: pread 6 experts → dispatch CMD3(expert forward) → 不等待！
       → 立即开始处理 layer N+1 (mhc_pre, attention, routing...)
  GPU: (同时在跑 layer N 的 CMD3 expert 计算)
  ...等到 layer N+1 需要 GPU 输出时 → waitUntilCompleted
```

**当前代码**：每个 CB 都立即 `waitUntilCompleted`，没有任何重叠。

#### 技术 B：GPU-side Combine + 下一层 RMSNorm（消除 CPU 往返）

flash-moe CMD3 内有 3 个 encoder 串联：
1. `moe_combine_residual`：expert 加权和 + residual
2. `rms_norm_sum_sq`：计算平方和
3. `rms_norm_apply`：用**下一层**的 norm weight 做归一化

输出直接就是下一层的输入，GPU 内完成，不需要 CPU 读回再写入。

**当前代码**：combine 在单独 CB 里，之后还要 CPU memcpy，完全串行。

#### 技术 C：时序预测 + 双缓冲（减少 I/O 等待）

- Token N-1 结束后记录每层 routing indices
- Token N 开始时，用预测 indices 异步 pread 到 B 缓冲区
- 如果命中：零 I/O；未命中：同步 pread 到 A 缓冲区

flash-moe 命中率 ~71%。**V4 局部性仅 ~35%**，效益有限——文档已经验证过不作为 V4 主优化手段。

### 33.5 代码实现状态对比

| 优化 | flash-moe | engine.c 当前 |
|------|-----------|--------------|
| Deferred CMD3（不等待 GPU） | ✅ `[cmd commit]` only | ❌ 全部 `waitUntilCompleted` |
| GPU-side combine + next-layer norm | ✅ 3 encoders in CMD3 | ❌ 单独 CB + CPU memcpy |
| 时序预测 + 双缓冲 | ✅ ~71% hit | ❌ 只有 TODO 注释 |
| 并行 pread 线程池 | ✅ 4 pthreads | ✅ 6 pthreads（已实现） |
| Packed per-layer bin | ✅ | ✅（已实现） |
| Expert RAM 缓存 | ❌（Trust OS） | ✅ `moe_infer_preload_experts`（框架已有但 hit rate 低）|
| DyMoE Skip | N/A | ❌ 没有移植到 native |
| SMELT（选择性预加载热门 expert）| ❌（Trust OS）| ❌ 没有实现 |

### 33.6 达到 1 tok/s 的最短路径

根据旧 MLX 路径的实测经验，从 0.14 → 1.0 tok/s 最快路径是**移植 SMELT 到 native engine**：

#### 步骤 1：SMELT-style 热门 expert 预加载（预期 +4-5×）

原理：预加载最常用的 N 个 expert 到内存，并在 routing 时通过 bias 惩罚未预加载的 expert（让它们得分更低，避免被选中）。

```c
// 在 cpu_moe_route 里，对未缓存的 expert 加 -1e9 的 bias
if (eng->expert_mem_cache[layer][expert_ids[k]] == NULL) {
    biased_scores[k] -= 1e9f;  // 强制不选未缓存 expert
}
```

配合 `moe_infer_preload_experts`（已有框架），只需：
1. 统计首 20 个 token 的 routing indices（热身）
2. 按频率排序，预加载 top-51（相当于 20%）expert 到内存
3. 在 `cpu_moe_route` 里对未缓存 expert 加大惩罚

预期命中率 ~60-70%（类似 MLX SMELT），有效 I/O 降到 ~1 GB/tok → 约 1 tok/s。

**工作量**：2-3 天（统计收集 + bias 接入 + 测试）

#### 步骤 2：DyMoE Skip 移植（预期额外 +20%）

把 `expert_stream.zig` 里的 DyMoE 逻辑移植到 `engine.c`：
- 热身阶段统计每层 expert 被选中频率
- skip mask：对从未在 top-5 中出现的 expert，在 routing 时直接 skip（减少 pread 次数）

**工作量**：1 天

### 33.7 达到 3 tok/s 的完整路径

在 1 tok/s 基础上继续：

#### 步骤 3：Deferred CMD3（预期 +1.3-1.5×）

```c
// 当前（错误）：串行等待
[cb_expert commit]; [cb_expert waitUntilCompleted];  // 阻塞 CPU ~6ms/layer

// 目标（deferred）：
[cb_expert commit];  // 不等待，立即返回
// CPU 继续处理下一层 attention/mhc/routing
// 在需要 expert 输出时才等待
```

需要跨层保存 in-flight command buffer，在下一层开始前 wait。

**工作量**：3-4 天（需要重构 `moe_infer_forward_layer` 的层循环结构）

#### 步骤 4：GPU-side Combine（预期额外 +20%）

把 combine + next-layer RMSNorm 放进同一个 CMD3 里，消除 CPU memcpy 往返。

**工作量**：2 天

#### 步骤 5：SIMD 优化 dequant kernel（预期 +10-15%）

参考 `flash-moe/shaders.metal: dequant_matvec_4bit_v3`：
- threadgroup tiling：8 rows/group
- 共享内存缓存 input vector
- SIMD reduction：`simd_sum(acc)`
- FMA 解量

⚠️ 历史教训：SIMD reduction 版本曾有 87-97% 输出为 0 的 bug，必须逐 kernel 对拍验证。

**工作量**：2-3 天（含对拍）

### 33.8 理论性能预估（完整优化后）

基于 flash-moe 类比（§11 的理论拆解）：

| 阶段 | 耗时/layer | 43 层 |
|------|-----------|-------|
| attention 投影 | ~1.2ms | ~52ms |
| mhc + attn | ~0.5ms | ~22ms |
| o_proj + norm + routing | ~0.55ms | ~24ms |
| I/O pread（SMELT 命中时 ~0.5ms，miss ~4ms×miss_rate） | ~0.8ms | ~34ms |
| expert forward（deferred overlap） | ~0.04ms | ~2ms |
| **合计** | ~3.1ms | **~133ms** |

理论 ~7.5 tok/s，保守含 overhead → **3-5 tok/s**。

### 33.9 优先级排序（按 ROI）

| 优先级 | 优化 | 预期收益 | 工作量 | 依赖 |
|--------|------|---------|--------|------|
| **P0** | SMELT-style expert 预加载 + routing bias | **0.14→1.0 tok/s (7×)** | 2-3天 | 无 |
| **P1** | DyMoE Skip 移植 | +20% | 1天 | P0 |
| **P2** | Deferred CMD3 | +30-50% | 3-4天 | P0 |
| **P3** | GPU-side combine | +20% | 2天 | P2 |
| **P4** | SIMD dequant kernel | +10-15% | 2-3天 | P0，需逐 kernel 对拍 |

**核心结论**：Native engine 现在 0.14 tok/s 主要是因为**没有 SMELT**（没有 expert I/O 缓存/筛选机制），其次是 **GPU 串行**（deferred CMD3 未实现）。这两者在旧 MLX 路径里都已实现，但迁移到 native engine 时没有同步。

### 33.10 参考资料

- `docs/analysis/flash-moe-alignment-analysis.md` — 所有实测数据和 A/B 对比结论
- `docs/analysis/flash-moe-alignment-plan.md §2` — flash-moe 架构详细分析
- `docs/analysis/flash-moe-plan.md` — P1/P2/P3 实现状态
- `docs/analysis/pread-expert-loading.md` — mmap vs pread 实验记录
- `docs/analysis/dsv4-first-class-support-plan.md §9, §11` — flash-moe 技术点摘录
- `src/models/expert_stream.zig` — DyMoE 和 SMELT 的 Zig 实现（待移植）
- `../flash-moe/metal_infer/infer.m:5354-5434` — Deferred CMD3 原始实现
- `../flash-moe/metal_infer/shaders.metal:251` — SIMD dequant_matvec_4bit_v3

### 修复: BF16 SDPA → f16 SDPA

将 mla_attention.m 中的 SDPA 核从 `mla_sdpa_decode_bfloat` 改为 `mla_sdpa_decode_f16in_f16out` 后 **消除 NaN**。

但模型仍输出空内容（所有 token 解码为空字符串），还需进一步调查。

该修复已 commit 到 engine.c/h 的正确 offset 修复之上。


---

## 34. 2026-06-15 Native serve 调试进展

### 34.1 已修复的关键 bug

本轮针对 `--native` serve 模式进行深度调试，修复/确认了以下问题：

1. **BOS/EOS 剥离错误** (`src/native_engine.zig`)
   - 原代码把 `EOS_TOKEN` 当成 BOS 判断，导致非 BOS token 被误剥，prompt 被破坏。
   - 已改为正确判断 `prompt_tokens[0] == BOS_TOKEN`。

2. **MoE 输入被错误截断为 bf16** (`src/metal_infer/engine.c`)
   - `fused_gate_up_swiglu_v2` 期望 f32 输入，但原代码把 `buf_normed` 截成 bf16 再喂给 kernel，导致 expert 输出失真。
   - 已移除该截断，保持 f32 输入。

3. **attention weight buffer 对齐问题** (`src/metal_infer/mla_attention.m`)
   - `newBufferWithBytesNoCopy` 在部分 Metal/CPU buffer 组合下出现对齐或生命周期问题。
   - 已把 attention 相关 weight buffer 改为 `newBufferWithBytes` 拷贝，确保稳定。

4. **MXFP4 expert packing offsets 错误** (`src/metal_infer/engine.h` / `engine.c`)
   - `06423da` 引入的 offset 把 6 个 slot 当成 8 个 slot（包含了不存在的 bias buffer），导致 expert 数据读错位。
   - 已恢复为 6-slot 布局：`GATE_W/GATE_S/UP_W/UP_S/DOWN_W/DOWN_S`。

5. **CPU routing 未加 gate bias** (`src/metal_infer/engine.c` 的 `cpu_moe_route`)
   - `e_score_correction_bias` 根本没参与计算，导致 score layer 的 expert 选择完全错误。
   - 已改为 `logits[i] + bias[i]` 后再做 `sqrtsoftplus`。
   - **这是本轮最关键修复**：修复前 `penalty=0` 输出为乱码；修复后输出变得连贯（但仍不正确）。

### 34.2 已验证的事实

- `penalty=1e9`（强制只选缓存 expert）时输出连贯但错误：`"to the question: ..."` / `"to the capital of ..."`，且长续写会重复。
- `penalty=0`（修复 bias 后）输出从乱码变成连贯但错误：`"to the capital of France. The capital of France is ..."`。
- 最终 norm + lm_head 计算是正确的：用 native 的 `final_normed` 和 MLX `lm_head` 重算 logits，与 native 输出 logits 完全一致。
  - 因此 bug 不在 final norm / lm head，而在前面 43 层 transformer 产生的 hidden state。
- packed_experts 二进制数据与 MLX safetensors 逐 byte 对齐（已抽检 expert 35 的 6 个 component）。
- compressor/indexer 禁用后输出不变，暂时排除它们是当前主要根因。
- gather mode 禁用/启用对当前正确性无本质影响；问题不在 gather kernel。

### 34.3 本节遗留问题（已在 §35 解决）

本节记录时 `bash scripts/dsv4_smoke.sh` 仍失败：

| prompt | 当前 native 输出 | 期望 |
|--------|----------------|------|
| `"The capital of France is"` | `"to the capital of France. The capital of France is the capital of France."` | 含 `paris` |
| `"2+2="` | `":??,??,??,..."` | 含 `4` |

后续按 §35 进行逐层对拍，定位到最早发散点为 `L0_moe_only_out`，根因为 `dequant_matvec_4bit` 的 dispatch 与 kernel `ROWS_PER_TG` 不匹配导致 50% 输出为零。修复后 smoke 通过。详见 §35。

---

## 35. 2026-06-15 Native serve 关键修复：MoE down_proj dispatch 错误

### 35.1 本轮目标

用户明确要求：
- 只测 `--native` serve 模式；
- 不绕开，定位并修复 native hidden state 与正确参考之间的系统性偏差。

### 35.2 调试手段

在 `src/metal_infer/engine.c` 中增加 `dump_if_requested` 工具，按 `DSV4_DUMP_DIR` + `NATIVE_DUMP_TOKEN_POS` 导出每层关键张量：

- `L{layer}_attn_in_pos{pos}.bin`：进入该层的 residual。
- `L{layer}_residual_postattn_pos{pos}.bin`：attention 之后的 residual。
- `L{layer}_normed_ffn_in_pos{pos}.bin`：FFN RMSNorm 后的输入。
- `L{layer}_post_ffn_pos{pos}.bin` / `comb_ffn_pos{pos}.bin`：mHC 后处理权重。
- `L{layer}_moe_only_out.bin`：MoE combine 之后、shared expert 之前的纯 routed-expert 输出。
- `L{layer+1}_in_pos{pos}.bin`：该层最终输出（下一层输入）。

参考版本使用 `git checkout 0286342` 在 `/tmp/dmlx-good` 独立构建，并接入相同 dump 逻辑，确保比较公平。

### 35.3 发现：最早发散点在 Layer 0 输出

| 张量 | current `f740757` | known-good `0286342` | 状态 |
|------|-------------------|----------------------|------|
| `L00_in` (embed / pos 0) | norm ≈ 16.8 | norm ≈ 16.8 | ✅ 一致 |
| `L0_residual_postattn` | norm ≈ 17.3 | norm ≈ 17.3 | ✅ 一致 |
| `L0_normed_ffn_in` | norm ≈ 0.48 | norm ≈ 0.48 | ✅ 一致 |
| `L0_post_ffn` / `comb_ffn` | 一致 | 一致 | ✅ mHC 权重一致 |
| `L0_moe_only_out` | norm ≈ 1.24，**2048 个零（50%）** | norm ≈ 4.92，无零 | ❌ 发散 |
| `L01_in` (layer 0 最终输出) | norm ≈ 22 | norm ≈ 528 | ❌ 严重发散 |

结论：**mHC 后处理本身是对的**，它把错误的 `moe_only_out` 继续传播，导致最终输出偏差被放大。

### 35.4 根因：down_proj kernel dispatch 与 kernel 不匹配

当前 separate mode 使用：

```c
[enc dispatchThreadgroups:MTLSizeMake((DIM + 1) / 2, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];   // 128 threads = 4 simd-groups
```

但 `dequant_matvec_4bit` 内核硬编码：

```metal
const uint ROWS_PER_TG = 8;
uint row = tgid * ROWS_PER_TG + simd_group;
```

`threadsPerThreadgroup=(32,4,1)` 只有 4 个 simd-group，每个 threadgroup 却只能覆盖 `row = tgid*8 + 0..3`，即 **每 8 行跳过 4 行**。结果 `buf_expert_out` 每隔 4 行就是 0，MoE combine 后 50% 输出为零。

这是 `571fd49` 把 separate-mode down_proj 从 affine-4bit 切到 MXFP4 时，顺手把 dispatch 改成与 `fused_gate_up_swiglu_v2` 一致（128 threads），但没有同步修改 `dequant_matvec_4bit` 的 `ROWS_PER_TG`。

### 35.5 修复

`src/metal_infer/engine.c` 中 down_proj dispatch 恢复为与 kernel 匹配：

```c
// dequant_matvec_4bit covers 8 rows per threadgroup (ROWS_PER_TG=8).
[enc dispatchThreadgroups:MTLSizeMake(DIM/8, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
```

同时修复 `src/native_engine.zig` 中 BOS/EOS 剥离逻辑：
- 原 `f740757` 把 `EOS_TOKEN` 误当成 BOS 判断条件；已改为 `BOS_TOKEN`。
- 正确剥离 prompt 首部的 BOS 与尾部的 EOS。
- 生成结果中如果最后一个 token 是 EOS，则从返回结果中剔除，避免输出 `<end of sentence>`。

### 35.6 验证结果

```bash
zig build -Doptimize=ReleaseFast
bash scripts/dsv4_smoke.sh
```

输出：

```
✓ capital-of-france: Paris.
✓ two-plus-two: 4
SMOKE PASS
```

`--native` serve 模式首次通过 smoke gate。

### 35.7 仍保留的 precision 差异

修复 dispatch 后 `L0_moe_only_out` 已无零，但当前使用 `fused_gate_up_swiglu_v2` + `dequant_matvec_4bit` 的 **bfloat 输出 cast**（`out[d] = (bfloat)val;`），而 `0286342` 使用 `fused_gate_up_swiglu` + `dequant_matvec_4bit` 的 float32 输出。两者内部激活不完全相同，但端到端输出已正确。后续若需要更严格数值对齐，可再评估是否将 MoE 输出改回 float32。

### 35.8 经验教训

- **dispatch 与 kernel 常量必须成对修改**：改 threadgroup 形状时，必须同步核对 kernel 里的 `ROWS_PER_TG` / `NR0` / `NSG` 等常量。
- **50% 零输出是强信号**：一旦 hidden state 出现规则性零块，首先怀疑 thread/threadgroup 覆盖是否完整。
- **端到端 smoke 是最硬标准**：中间激活可接受小幅 precision 差异，只要最终 greedy 输出正确即可放行。


---

## 36. 2026-06-15 后续：benchmark 7-prompt E2E 与 SMELT N 调整

### 36.1 发现：`run_benchmark.sh` 默认 `SMELT N=51` 在 48GB Mac 上 OOM

将 7-prompt E2E 接入 native 分支后跑 `./scripts/run_benchmark.sh --native`：

```
tok/s:   0.592
Paris:   ✓ PASS
E2E:     2/7 passed
```

失败的不是答案错误，而是 server 在第 3 个 prompt 时被系统 `SIGKILL`：

```
scripts/run_benchmark.sh: line 221: 36961 Killed: 9
```

根因：`SMELT N=51` 预加载约 33GB expert cache，加上主权重/KV cache/GPU buffer 后，48GB 内存被吃满，连续请求触发 OOM。

### 36.2 调整：默认 `NATIVE_SMELT_N` 从 51 降到 20

修改：
- `scripts/run_benchmark.sh`: `NATIVE_SMELT_N="${NATIVE_SMELT_N:-20}"`
- `scripts/native_bench.sh`: `SMELT_N="${NATIVE_SMELT_N:-20}"`

验证：

```bash
bash scripts/run_benchmark.sh --native
```

结果：

```
tok/s:   0.616
Paris:   ✓ PASS
E2E:     7/7 passed
unit:    PASS (430+)
time:    149s
```

7 个 prompt 全部通过，server 不再 OOM。

### 36.3 仍存在的问题

- **tok/s 0.616 仍低于历史 0.7 基线**。需要后续性能优化（`wo_a` Q8_0、MXFP4 v2 coalesced down_proj、deferred CMD3 等）。
- `SMELT N=51` 在 48GB 上仍然无法稳跑完整 7-prompt benchmark。若要在更高 SMELT N 下不 OOM，需要降低 native serve 的 per-request 内存峰值或引入 mmap/expert 换出机制。


---

## 37. 2026-06-15 深度源码性能分析：native engine 优化路径

### 37.1 分析方法

基于当前通过 correctness 的代码（`f740757` + 本节未提交修正），对以下模块做了只读源码分析：

- `src/models/moe_kernel.metal`：所有 Metal kernel 实现。
- `src/metal_infer/engine.c`：层循环、command buffer 编排、SMELT I/O、`moe_forward_layer`、shared expert、mHC 前后处理。
- `src/metal_infer/mla_attention.m`：注意力权重缓存、Q/KV/SDPA/`wo_a`/`wo_b` 路径。
- `src/metal_infer/mhc.c` / `moe_kernel.metal` 中 mHC 相关 kernel。
- `src/native_engine.zig`：SMELT 初始化、penalty、gather 模式开关。

当前基线：`NATIVE_SMELT_N=20`，`bash scripts/run_benchmark.sh --native` 得 **0.616 tok/s**，7-prompt E2E 全过。

### 37.2 关键发现：0.616 tok/s 的主要瓶颈不是 kernel 算力，而是 I/O 与同步

#### 37.2.1 每 token 的层循环要经过 5 次 `waitUntilCompleted`

`moe_infer_forward_layer` 的 decode 路径：

1. 等上一层 deferred CB3（mHC post-ffn）。
2. CB-A+CB1 merged：mHC pre(attn) + 完整 attention，1 次 wait。
3. CB2+CMD2 merged：mHC post(attn) + mHC pre(ffn) + RMSNorm + routing，1 次 wait。
4. CMD3 MoE：6 gate-up + 6 down-proj + combine，1 次 wait。
5. Shared expert：gate/up/SwiGLU/down，1 次 wait。
6. CB3 mHC post(ffn)：commit 但不 wait，留到下一层开头。

即每层 **5 次 blocking GPU wait**，43 层共约 215 次同步。Apple Silicon 每次 wait 实测 2–13 ms，仅此就占 430 ms–1.08 s/token。

#### 37.2.2 `SMELT N=20` + `penalty=0` 导致缓存形同虚设

`src/native_engine.zig:106`：

```zig
metal.smeltInit(engine, 0, smelt_n, 0.0);
```

第三个参数是 `smelt_penalty`，当前为 **0.0**。后果：

- CPU routing `cpu_moe_route` 选择真实 top-6，**不偏向已缓存的 20 个 expert**。
- GPU routing `moe_route_gpu` 被 `engine.c:1315` 的条件 `eng->smelt_penalty > 0.0f` 屏蔽，永不启用。
- 缓存命中率低，每 token 可能触发大量 SSD `pread`：43 层 × 6 experts × 13.4 MB ≈ 3.46 GB I/O。

这是当前 tok/s 远低于历史 0.7 基线的**首要原因**。

#### 37.2.3 每层大量临时 buffer 分配

| 位置 | 每层分配内容 |
|------|-------------|
| `engine.c:866-867` | `weights_buf`（6 floats）、`zero_resid`（DIM floats） |
| `engine.c:1523-1579` | shared expert gate/up/down 的 weight/scale/bias buffer 9 个 |
| `mla_attention.m:679-680` | `bcos`、`bsin`（RoPE cos/sin，64 bytes） |

这些分配廉价但累积起来有开销，且 `zero_resid` 依赖 Apple 对新 Shared buffer 的零初始化，有隐式正确性依赖。

#### 37.2.4 已存在但未启用的更快路径

| 更快路径 | 当前状态 | 阻碍 |
|----------|----------|------|
| `mhc_post_ffn_expand4`（单 encoder f32 mHC post） | 已编译，未使用 | 之前被认为有数值漂移 |
| `gather_gate_up_swiglu` / `gather_down`（gather mode） | 已编译，默认禁用 | 之前测得比 separate mode 慢 |
| `matvec_q8_0_f32`（Q8_0 `wo_a`） | `AttnBufCache` 已量化并缓存，但 dispatch 强制走 f32 dense | “numerical stability” |
| `moe_route_gpu` | 已编译，因 penalty=0 未启用 | penalty 配置 |

### 37.3 推荐的优化路径（按 ROI 与风险排序）

#### P0：启用 SMELT routing bias / GPU routing（最高 ROI，风险可控）

**改动**：
- `src/native_engine.zig:106`：`metal.smeltInit(engine, 0, smelt_n, 1e9f)`。
- 验证 `cpu_moe_route` 与 `moe_route_gpu` 在 penalty 作用下行为一致。

**预期效果**：
- 强制 top-6 从缓存的 20 个 expert 里选，命中率接近 100%。
- SSD I/O 从 ~3.46 GB/token 降到接近 0。
- 速度可能从 0.616 提升到 **1.8–3.0 tok/s** 区间（历史文档对有效 SMELT 的预期）。

**风险**：
- 强制选缓存 expert 可能选到非真实 top-6，引入数值误差。必须重跑 7-prompt E2E 验证 correctness。

#### P1：合并 CMD3 与 shared expert（中低风险，明显收益）

**改动**：
- `src/metal_infer/engine.c`：把 MoE combine 后的 `buf_hidden` 直接作为 shared expert 输入，在同一个 command buffer 里编码 shared expert，减少一次 `waitUntilCompleted`。

**预期效果**：
- 每层减少 1 次 wait，43 层 × 2–5 ms ≈ **85–215 ms/token**。
- 约 **+5–13% tok/s**。

**风险**：
- 需要 persistent buffer 保存 shared expert 输出，供 mHC post-ffn 读取。需验证 buffer lifetime。

#### P2：持久化 shared expert 权重与 combine buffer（低风险，中等收益）

**改动**：
- 在 `MoEInferEngine` 里增加 `buf_shared_gate_W/S/B`、`buf_shared_up_W/S/B`、`buf_shared_down_W/S/B`（每层一组），初始化时上传。
- 增加 `buf_moe_weights`、`buf_moe_zero`，每层 `memcpy` 更新。

**预期效果**：
- 消除 ~9 个 `newBufferWithBytes` + 2 个 `newBufferWithBytes`/`newBufferWithLength` 每层。
- 约 **+2–5% tok/s**，并消除 `zero_resid` 的隐式零初始化依赖。

#### P3：修复并启用 `mhc_post_ffn_expand4`（中低风险，中等收益）

**改动**：
- 用 `mhc_post_ffn_expand4` 替代当前 3-encoder bf16 路径（f32→bf16 → `mhc_post_bfloat` → bf16→f32）。
- 用 fixed prompt 与当前路径做 bitwise 对比，确认 comb 方向与 `POST_MULT` 处理正确。

**预期效果**：
- 每层 mHC post-ffn 从 3 encoder 降到 1 encoder。
- 约 **+5–10% tok/s**。

#### P4：启用 Q8_0 `wo_a` 路径（中等风险，中等收益）

**改动**：
- `src/metal_infer/mla_attention.m:862-868`：当 `abc->wo_a_q8_gpu[g]` 存在时 dispatch `enc_matvec_q8_0`，否则 fallback f32。

**预期效果**：
- `wo_a` 权重内存流量从 128 MB/layer 降到 ~36 MB/layer。
- 每 token 减少约 3–4 GB 内存流量。
- 约 **+5–15% tok/s**。

**风险**：
- Q8_0 量化引入数值漂移。必须通过 `scripts/run_benchmark.sh --native` 7-prompt 验证。

#### P5：Deferred CMD3 / 层间重叠（高风险，高潜在收益）

**改动**：
- 把当前层的 MoE CB（CMD3）commit 后不 wait，让 GPU 继续算 MoE，同时 CPU 开始下一层的 attention 前处理 / routing / I/O。
- 在下一层需要 residual 前再 wait 上一层的 CMD3。

**预期效果**：
- 把 MoE GPU 计算与下一层 CPU/I/O 工作重叠，隐藏 200–400 ms/tok。
- 约 **+10–20% tok/s**。

**风险**：
- `expert_buf[k]` 等 scratch buffer 会被下一层覆盖，必须等 GPU 读完才能复用。
- 实现复杂，正确性风险高。

### 37.4 不建议优先做的方向

| 方向 | 原因 |
|------|------|
 纯优化 MoE down_proj kernel（写 MXFP4 v2 coalesced） | MoE GPU 时间仅占 ~12%，I/O 和同步才是大头；等 I/O 解决后再做 |
 重写 SDPA 占用率 | 当前 decode SDPA 不是瓶颈，且 online softmax 跨 simdgroup 容易引入数值漂移 |
 增大 `NATIVE_SMELT_N` 到 51 | 48GB 机器会 OOM，除非先降低 per-request 内存峰值 |

### 37.5 建议的下一步执行顺序

1. **P0**：改 `smelt_penalty=1e9`，跑 `bash scripts/run_benchmark.sh --native`，确认 7/7 E2E 仍通过，记录 tok/s。
2. 如果 P0 后内存仍够，尝试把 `NATIVE_SMELT_N` 从 20 提到 30–40，重新测量。
3. **P1**：合并 CMD3 + shared expert，重跑 benchmark。
4. **P2 + P3**：持久化 buffer + 单 encoder mHC post，重跑 benchmark。
5. **P4**：启用 Q8_0 `wo_a`，重跑 benchmark。
6. 最后再评估 P5 deferred CMD3 是否值得。

### 37.6 关键源码引用

| 文件 | 行号 | 内容 |
|------|------|------|
| `src/native_engine.zig` | 106 | `metal.smeltInit(engine, 0, smelt_n, 0.0)` — penalty 为 0 |
| `src/metal_infer/engine.c` | 1315 | GPU routing被 `smelt_penalty > 0` 条件屏蔽 |
| `src/metal_infer/engine.c` | 717-886 | `moe_forward_layer`，6 gate-up + 6 down + combine |
| `src/metal_infer/engine.c` | 866-867 | 每层临时分配 `weights_buf`、`zero_resid` |
| `src/metal_infer/engine.c` | 1513-1585 | shared expert，每层新建 weight buffer |
| `src/metal_infer/engine.c` | 1587-1650 | mHC post-ffn，3 encoder bf16 路径 |
| `src/metal_infer/mla_attention.m` | 862-868 | `wo_a` 强制走 f32 dense，Q8_0 已缓存但未用 |
| `src/models/moe_kernel.metal` | 1684-1725 | `mhc_post_ffn_expand4` 单 encoder f32 kernel |


---

## 38. 2026-06-15 P0 实验：SMELT routing bias / warmup 的尝试与失败

### 38.1 实验 1：penalty=1e9（强制选缓存 expert）

**改动**：`src/native_engine.zig:106` `metal.smeltInit(engine, 0, smelt_n, 1e9)`。

**结果**：

```
tok/s:   0.239
Paris:   ✗ FAIL (输出 "France is the capital")
E2E:     1/7 passed
```

**结论**：强制从缓存的 0..19 号 expert 里选，与真实 top-6 偏差太大，直接破坏 correctness。此路不通。

### 38.2 实验 2：warmup=64 + penalty=0.5（收集路由统计后预加载热 expert）

**改动**：
- `warmup_tokens=64`
- `penalty=0.5`
- 不立即调用 `smeltFinishWarmup`，让 decode token 计数触发 async preload。

**结果**：

```
tok/s:   0.568
Paris:   ✓ PASS
E2E:     7/7 passed
```

**问题**：benchmark 整个流程只产生 ~45 个 decode token，小于 `warmup_tokens=64`，async preload 根本没触发。速度没有提升。

### 38.3 实验 3：warmup=24 + penalty=0.5

**改动**：把 `warmup_tokens` 降到 24，让 benchmark 的 warmup 请求能触发预加载。

**结果**：

```
[smelt] Async preload started (routing bias inactive until complete)
[smelt] Warmup complete (24 tokens). Preloading experts per layer...
[smelt] Async preload started (routing bias inactive until complete)
[smelt] Warmup complete (25 tokens). Preloading experts per layer...
[smelt] Async preload started (routing bias inactive until complete)
[smelt] Warmup complete (26 tokens). Preloading experts per layer...
...
scripts/run_benchmark.sh: line 170: 90133 Abort trap: 6
```

**问题**：async preload 被**重复触发多次**，多个后台线程同时读写 `expert_mem_pool` / `expert_mem_cache` / `smelt_pool_pos`，导致数据竞争，最终 `Abort trap: 6`（SIGABRT）。server 崩溃后所有 E2E 失败。

根因：`moe_infer_smelt_preload_async` 在 preload 完成前没有设置 `smelt_warmup_done=true`，而 `moe_infer_forward_layer` 里的计数器继续累加，每次 `smelt_tokens_seen >= warmup_tokens` 都会再次启动一个 preload 线程。

### 38.4 回退到稳定配置

当前代码回退为：

```zig
metal.smeltInit(engine, 0, smelt_n, 0.0);
const n_loaded = metal.smeltFinishWarmup(engine);
```

- `warmup_tokens=0`：立即同步预加载 experts 0..N-1。
- `penalty=0.0`：路由不偏向缓存，保持 correctness。
- `NATIVE_SMELT_N=20`：48GB 机器上能稳跑 7-prompt E2E 不 OOM。

验证：`bash scripts/dsv4_smoke.sh` 通过，`Paris.` / `4` 正确。

### 38.5 修正后的性能路径

P0（启用 routing bias）不能直接做，必须分成两步：

1. **先修 async preload 竞态 bug**：在 `moe_infer_smelt_preload_async` 启动时立即设置 `smelt_warmup_done=true`（或一个 `preloading` 标志），防止重复触发。
2. **再启用 warmup + 小 penalty**：warmup_tokens 建议 64–128，penalty 建议 0.2–0.5，预加载真实热 expert 后再服务请求。

在 bug 修复前，更安全的性能优化是：

- **P1**：合并 CMD3 + shared expert（减少一次 wait/layer）。
- **P2**：持久化 shared expert 权重与 combine buffer（减少分配）。
- **P3**：修复并启用 `mhc_post_ffn_expand4`（减少 encoder 数量）。
- **P4**：启用 Q8_0 `wo_a`（减少 attention 权重内存流量）。

这些改动风险较低，预计能把 tok/s 从 0.568 提升到 **0.65–0.75** 区间，接近或超过历史 0.7 基线。


---

## 39. 2026-06-15 MXFP4 v2 coalesced down-proj kernel：可行性 + 正确性分析

### 39.1 目标

把当前 separate-mode 的 down_proj 从 naive kernel `dequant_matvec_4bit` 换成 v2 coalesced kernel，复刻 `fused_gate_up_swiglu_v2` 的并行策略，同时保持 MXFP4 的 E8M0 scale 解码。

### 39.2 现有 kernel 对比

| 属性 | `fused_gate_up_swiglu_v2`（gate_up，在用） | `dequant_matvec_4bit`（down_proj，在用） | `dequant_matvec_affine_v2`（affine down，参考） |
|------|----------------------------------------|----------------------------------------|-----------------------------------------------|
 数据格式 | MXFP4, gs=32 | MXFP4, gs=32 | affine 4-bit, gs=64 |
 并行策略 | v2 coalesced, NR0=2, NSG=4, NQ=8, TPG=4 | naive one-row-per-thread, ROWS_PER_TG=8 | v2 coalesced, NR0=2, NSG=4, NQ=4, TPG=8 |
 输出 | gate + up 两个值 | 一个值 | 一个值 |
 scale | uint8_t E8M0 (`exp2(s-127)`) | uint8_t E8M0 | float scale + float bias |
 共享内存 | 512 B (2 gate + 2 up rows) | 16 KB (`x_shared[4096]`) | 256 B |
 线程组 | (32,4,1)=128 threads | (256,1,1)=256 threads | (32,4,1)=128 threads |
 threadgroups | (out_dim+1)/2 | out_dim/8 | (out_dim+1)/2 |

### 39.3 关键观察：gate_up_v2 的 tiling 已经适配 gs=32

`fused_gate_up_swiglu_v2` 参数：

```metal
const short NR0 = 2, NSG = 4, NW = 32, NQ = 8, TPG = 4;
```

- `TPG=4` 对应 `packed_per_group = group_size/8 = 32/8 = 4`。
- `NQ=8`、`NSG=4` → 每个 SIMD group 处理 `g0 = sgitg*8 + ix`（ix=0..7），gg 步长 `NSG*NQ=32`。
- 对 `num_groups=128`（down_proj in_dim=4096, gs=32），每个 thread 处理 4 个 group，4 个 SIMD group 无重叠覆盖全部 128 group。

这个 tiling 对 gs=32 是**正确且已验证**的（smoke test 通过）。

### 39.4 新 kernel 设计

创建 `dequant_matvec_4bit_v2`：

```metal
kernel void dequant_matvec_4bit_v2(
    device const uint32_t* W_packed [[buffer(0)]],
    device const uint8_t*  scales   [[buffer(1)]],
    device const float*    x        [[buffer(2)]],
    device float*          out      [[buffer(3)]],
    constant uint&         out_dim  [[buffer(4)]],
    constant uint&         in_dim   [[buffer(5)]],
    constant uint&         group_size [[buffer(6)]],
    threadgroup float*     shmem    [[threadgroup(0)]],
    uint3  tgpig  [[threadgroup_position_in_grid]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]]
)
```

实现要点：
1. 复用 gate_up_v2 的 NR0/NSG/NW/NQ/TPG 参数。
2. 只保留一组 weight + scale（去掉 up、gate、SwiGLU）。
3. scale 用 `exp2((float)sc[gg] - 127.0f)`，与 naive kernel 完全一致。
4. weight 用 `NIBBLE_TO_FLOAT[(pw >> n*4) & 0xF]`，与 naive kernel 完全一致。
5. 输出 `out[d] = (bfloat)simd_sum_result`。

### 39.5 engine.c 改动点

1. **新增 pipeline**：在 `moe_metal_build_pipeline` 里加入 `dequant_matvec_4bit_v2`。
2. **新增 pipeline 指针**：`MoEInferEngine.pipe_dequant_matvec_v2`。
3. **down_proj dispatch 改为 v2 形状**：
   - `threadsPerThreadgroup: MTLSizeMake(32,4,1)`
   - `threadgroups: MTLSizeMake((DIM+1)/2,1,1)`
   - `setThreadgroupMemoryLength:256 atIndex:0`（2 rows × 32 lanes × 4 bytes）
4. **buffer binding**：只需 W 和 scale（去掉 bias），index 相应调整。

### 39.6 正确性风险与缓解

| 风险 | 可能性 | 缓解措施 |
|------|--------|----------|
 scale 解码不一致（E8M0 特殊值 NaN/inf） | 低 | 完全复用 naive kernel 的 `exp2(s-127)` 公式 |
 NIBBLE_TO_FLOAT LUT 不一致 | 低 | 直接使用同一个 LUT |
 Tiling 越界（row >= out_dim） | 低 | 照搬 gate_up_v2 的 `if (r >= out_dim) continue` |
 共享内存 reduction 边界 | 低 | 复用 gate_up_v2 的 `simd_sum` + `shmem` 模式 |
 group_size 参数未来变化 | 中 | kernel 内部用 `group_size` 动态计算，但 TPG=4 只保证 gs=32；gs=64 需新变体或调参 |

### 39.7 预期收益

- 当前 down-proj 是 naive one-thread-per-row，SIMD 利用率和内存合并度差。
- v2 与 gate_up_v2 同构，预计 down-proj GPU 时间降到原来的 **1/2 ~ 1/4**。
- 单独此项预计 tok/s 从 0.568 → **0.62–0.68**。
- 若未来 SMELT cache 命中率提高（I/O 不再是瓶颈），v2 down-proj 的收益会进一步放大。

### 39.8 投入评估

| 项目 | 估算 |
|------|------|
 新增 kernel 代码 | ~120 行 Metal |
 engine.c pipeline + dispatch 改动 | ~30 行 |
 编译 + smoke 验证 | 10 分钟 |
 完整 7-prompt benchmark 验证 | 3–5 分钟 |
 回退成本 | 低（改回 `pipe_dequant_matvec` 即可） |

### 39.9 建议

**建议实施**。理由：
1. 实现简单，有现成模板（gate_up_v2）。
2. 正确性风险低，公式/LUT/tiling 都与现有正确 kernel 一致。
3. 收益正向且可量化，是当前 ROI 较高的纯 kernel 优化。
4. 不依赖 SMELT bug 修复即可落地。

实施顺序：先写 kernel → 接 pipeline → 改 dispatch → smoke → 7-prompt benchmark → 对比 tok/s。


---

## 40. 2026-06-15 重新审视：ds4 / flash-moe 路线 vs 当前 dmlx native engine

### 40.1 已读文档

- `docs/en/analysis/ds4-native-engine-gap-analysis.md`
- `docs/en/analysis/ds4-kernel-deconstruction.md`
- `docs/en/analysis/native-engine-4toks-plan.md`
- `docs/analysis/flash-moe-alignment-analysis.md`
- `docs/analysis/flash-moe-alignment-plan.md`
- `docs/analysis/flash-moe-plan.md`
- `ds4` 源码仓库：`/Users/zouyee/work/code/ds4`（C/ObjC + Metal，~70K 行）

### 40.2 ds4 / flash-moe 的核心结论

| 项目 | ds4 | flash-moe | 当前 dmlx native |
|------|-----|-----------|-----------------|
| **真实 tok/s** | ~2+ tok/s（文档另有 22 tok/s 声明） | 4.36 tok/s（Qwen3.5） | **0.568 tok/s** |
| **每 token GPU sync** | **1–2 次** | 约 3 次 | **~215 次** |
| **CommandBuffer 策略** | 整层图 batch 到 1–2 个 CB | 3 个 CB，deferred CMD3 | 每层 4–5 个 CB |
| **路由** | 全 GPU | CPU topK | CPU topK（SMELT penalty=0 时） |
| **mHC** | 全 GPU， fused kernels | 无 mHC | CPU 读 post/comb，多 CB |
| **Expert IO** | mmap + NoCopy GPU buffer | Trust OS direct pread | pread fallback / SMELT cache |
| **模型格式** | **GGUF** | GGUF / custom | MLX safetensors |

### 40.3 为什么 dmlx 慢：sync 是主因，不是 kernel

`ds4-native-engine-gap-analysis.md` 明确指出：

> Each `waitUntilCompleted` on Apple Silicon costs ~2–13ms. At 172 syncs × 13ms = **2.2 seconds of pure synchronization overhead per token**.

当前 dmlx 每层 4–5 次 wait，43 层 ≈ 215 次。即便把每个 kernel 加速到 0ms，tok/s 也到不了 1.0，因为 sync 开销已经占满时间。

真正的性能杠杆按 ROI 排序：

1. **消除 CPU-GPU 往返**（mHC post/comb 读回、路由 gate score 读回）。
2. **Batch 所有层到 1–2 个 CommandBuffer**。
3. **mmap expert 权重 + NoCopy GPU buffer**（替代 pread/SMELT）。
4. **Kernel 融合与格式优化**（affine 4-bit / Q8_0）。

### 40.4 “直接迁移 ds4” 的障碍

`ds4` 是一个独立完整的 DeepSeek V4 推理引擎（~70K 行 C/ObjC + Metal），只支持 **GGUF** 格式。当前 dmlx 使用 **MLX safetensors**。直接迁移意味着：

| 迁移方式 | 工作量 | 障碍 |
|----------|--------|------|
| **整体替换 dmlx native engine 为 ds4** | 极大（数周~数月） | 模型格式 GGUF vs safetensors；tokenizer、server API、KV cache 全部不同 |
| **把 ds4 编译成库供 dmlx 调用** | 大 | 需要把 safetensors 权重转 ds4 所需 GGUF 格式；API 适配 |
| **只移植 ds4 的 Metal kernels** | 中 | 能解决 kernel 效率，但不解决 sync 架构问题 |
| **移植 ds4 的调度架构（batch CB + GPU mHC + GPU routing）** | 很大 | 需要重写 `engine.c` 层循环，但保留 dmlx 的模型加载和 server |

### 40.5 推荐方案：分阶段“ds4 化”，而非一次性迁移

#### Phase 1：先验证 ds4 本身能否跑通我们的模型（1–2 天）

目标：确认 ds4 的 2+ tok/s 在我们的硬件/模型上可复现。

步骤：
1. 检查 ds4 是否已有对应 DeepSeek-V4-Flash-4bit 的 GGUF。
2. 若没有，用 `llama.cpp` / `convert.py` 把当前 safetensors 转成 GGUF（Q4_K_M 或对应 MXFP4 的 GGUF type）。
3. 跑 `ds4 serve` + 同样 7-prompt E2E，记录 tok/s。

**如果 ds4 跑不到 2+ tok/s**，说明文档数据有水分或硬件差异，不值得迁移。
**如果 ds4 能到 2+ tok/s**，则进入 Phase 2。

#### Phase 2：移植 ds4 调度架构到 dmlx（核心，2–4 周）

这是 ROI 最高的方向，按 ds4 文档估计可消除 **~2s/token 的 sync 开销**。

具体改动：

1. **GPU-only mHC**：
   - 把 `post[]` / `comb[]` 放到 persistent GPU buffer。
   - 用 `kernel_dsv4_hc_split_weighted_sum_norm4` 替代当前 CB-A + CPU 读回 + CMD2 的流程。
   - 参考：`/Users/zouyee/work/code/ds4/metal/dsv4_hc.metal`

2. **GPU-only routing**：
   - 当前 `moe_route_gpu` 已实现但默认禁用。
   - 让 gate matmul 输出留在 GPU，直接接 GPU router kernel，输出 `selected[6]` + `weights[6]`。
   - 参考：`/Users/zouyee/work/code/ds4/metal/dsv4_misc.metal`

3. **Batch CommandBuffer**：
   - 把整个 43 层 encode 进 1–2 个 CB，只在最后 `waitUntilCompleted` 一次。
   - 需要把专家 IO 从 sync pread 改为 async / mmap，否则 CB 无法连续 encode。

4. **mmap expert + NoCopy**：
   - 用 `mmap` 映射 packed_experts 或 safetensors，GPU kernel 直接通过 `newBufferWithBytesNoCopy` 读。
   - 替代当前的 `pread → CPU scratch → GPU` 路径。

#### Phase 3：移植 ds4 关键 fused kernels（1–2 周）

在调度架构改完后，再引入 ds4 的 kernel 融合：

- `kernel_dsv4_shared_down_hc_expand4_q8_0`：shared expert down + routed combine + HC expand 一次 dispatch。
- `kernel_dsv4_q8_hc_expand4_q8_0`：attention output + HC expand 一次 dispatch。
- `kernel_dsv4_qkv_rms_norm_f32_4`：Q/KV norm 合并。

#### Phase 4：量化格式迁移（可选，高风险）

- ds4 使用 GGUF Q4_K / Q8_0，flash-moe 使用 affine 4-bit。
- 若想把 dmlx 权重也换成这些格式，需要重新打包模型并验证数值。
- 这是“最后 2×”的优化，应放在架构迁移之后。

### 40.6 为什么不先写 MXFP4 v2 down-proj kernel

之前 §39 分析的 v2 down-proj kernel 是**纯 kernel 优化**，预计只能把 tok/s 从 0.568 拉到 0.62–0.68。在 ds4 路线面前，它是**次要矛盾**。

因为：
- 当前 dmlx 的 MoE GPU 时间只占总 token 时间的 ~15–20%。
- 80% 以上时间花在 **CPU-GPU sync、路由 CPU 处理、mHC CPU 读回** 上。
- 把 down-proj kernel 加速 4×，整体只快 5–10%。

所以正确策略是：**先搞 ds4 调度架构，再补 kernel 优化**。

### 40.7 决策建议

建议执行顺序：

1. **立即**：验证 ds4 在本机 + DeepSeek-V4-Flash 上的真实性能（Phase 1）。
2. **如果验证通过**：开始 Phase 2，优先做 **GPU-only mHC**（文档估计可省 ~1.1s/token）。
3. **并行**：把 `moe_route_gpu` 从默认禁用改为启用，验证 correctness。
4. **最后**：批量 CommandBuffer + mmap expert。

这个路线比 “直接迁移整个 ds4” 更可控，且目标是让 dmlx native engine 逐步具备 ds4 的核心架构优势，而不是放弃 dmlx 现有代码。


---

## 41. 2026-06-15 源码级结论：不能直接迁移 ds4/flash-moe，只能借鉴其架构与算法

### 41.1 已完成的源码阅读

| 源码 | 规模 | 关键发现 |
|------|------|----------|
| `/Users/zouyee/work/code/ds4/ds4.c` | 27,725 行 | 完整 GGUF 加载 + Metal graph 调度 + CPU reference |
| `/Users/zouyee/work/code/ds4/ds4_metal.m` | 26,629 行 | ObjC Metal runtime，batch CB，mmap NoCopy，streaming cache |
| `/Users/zouyee/work/code/ds4/metal/dsv4_hc.metal` | 885 行 | mHC fused kernels（Sinkhorn + weighted sum + expand） |
| `/Users/zouyee/work/code/ds4/metal/dsv4_misc.metal` | 1,327 行 | Router + indexer + mixed attention kernels |
| `/Users/zouyee/work/code/flash-moe/metal_infer/infer.m` | ~7,151 行 | Qwen3.5 专用 engine，3-CB deferred pipeline |
| `/Users/zouyee/work/code/flash-moe/metal_infer/shaders.metal` | 1,296 行 | affine 4-bit / GatedDeltaNet kernels |

### 41.2 为什么不能直接迁移 ds4

#### 41.2.1 模型格式完全不同

| 维度 | ds4 | dmlx native |
|------|-----|-------------|
| 文件格式 | **GGUF** | **MLX safetensors** |
| Expert 量化 | Q2_K / Q4_K / IQ2_XXS (GGUF block) | **MXFP4** (`uint8 E8M0 scale + nibble LUT`) |
| Attention/共享专家 | Q8_0 / F16 | affine 4-bit / MXFP4 |
| 加载方式 | 单文件 mmap，NoCopy GPU buffer | safetensors → 上传 GPU buffer |
| 权重命名 | `blk.N.*` (GGUF) | `model.layers.N.*` (HF/mlx-lm) |

ds4 的 Metal kernels 直接解码 GGUF `block_q4_K` / `block_q8_0` 布局，这些内存在 dmlx 中根本不存在。要跑 ds4 kernel，必须先转换权重格式。

#### 41.2.2 模型架构不同

| 维度 | ds4 / flash-moe | dmlx (DeepSeek V4) |
|------|-----------------|-------------------|
| 层数 | 43 (ds4) / 60 (flash-moe) | 43 |
| Attention | Full SDPA + compressed KV indexer | **MLA** + compressor/indexer |
| 线性注意力 | GatedDeltaNet / conv1d (flash-moe) | 无 |
| Hyper-Connection | ds4 有，全 GPU | dmlx 有，CPU 读 post/comb |
| Experts | 256/384 (ds4), 512 (flash-moe) | 256 |
| K (active experts) | 6 (ds4), 4 (flash-moe) | 6 |

flash-moe 的 GatedDeltaNet / conv1d 内核在 DeepSeek V4 上完全无用。ds4 的 indexer/mixed attention 内核虽然思想相近，但维度、head 数、KV 格式都对不上。

#### 41.2.3 调度模型虽然相似但实现深度耦合

ds4 的 batch CB 调度分散在 C 引擎和 ObjC Metal runtime 之间，与 GGUF offset、streaming cache、CPU reference 路径强耦合。把它“直接”搬到 dmlx 等于重写整个 native engine。

### 41.3 不能直接迁移，但能借鉴什么

#### 41.3.1 最值得借鉴的 ds4 思想

| ds4 特性 | 收益 | 移植到 dmlx 的难度 |
|----------|------|-------------------|
| **GPU-only mHC**（`dsv4_hc.metal`） | 消除每层 2–3 次 CPU 读回 | 中 |
| **Batch 所有层到 1–2 个 CB** | 消除 ~215 次 wait/layer | 高 |
| **GPU-only routing** | 消除 gate score CPU 读回 | 中 |
| **mmap + NoCopy expert buffer** | 省掉 pread / GPU upload | 中 |
| **FP8 KV cache** | 省内存 + bandwidth | 中 |
| **Q8_0 / Q4_K GGUF 内核** | 不适用 | — |

#### 41.3.2 可 cherry-pick 的具体内核

| 内核 | 来源 | 用途 | 移植前提 |
|------|------|------|----------|
| `kernel_dsv4_hc_split_weighted_sum_norm4` | ds4 | mHC pre + RMSNorm 融合 | 改成 dmlx 的 bf16/f32 buffer 布局 |
| `kernel_dsv4_hc_expand4` | ds4 | mHC post 单 dispatch | 已有 `mhc_post_ffn_expand4` 类似实现 |
| `kernel_dsv4_shared_down_hc_expand4_q8_0` | ds4 | shared down + routed + HC expand | 需 Q8_0 权重或重写为 MXFP4 |
| `dequant_matvec_4bit_v3/v4/v5` | flash-moe | affine 4-bit matvec tiling | dmlx attention 权重已是 affine，可直接用思路 |
| `moe_combine_residual` | flash-moe | GPU combine + residual | dmlx 已有 `moe_combine`，可融合 shared gate |

### 41.4 修正后的正确路线

**放弃“直接迁移 ds4/flash-moe 实现”**。改为：**以 ds4 为参考，把其核心架构优势逐步移植到 dmlx native engine**。

#### Stage A：立即验证 ds4 真实性能（1 天）

用现有 ds4 binary + GGUF 模型（或临时转换 safetensors → GGUF）跑同样 7-prompt E2E，确认它在本机是否真能到 2+ tok/s。

**目的**：避免基于文档数字盲目投入。

#### Stage B：GPU-only mHC（1–2 周，最高 ROI）

这是 ds4 路线里**最快、最稳、收益最大**的一步。

- 把 `post[]`/`comb[]` 放到 persistent GPU buffer。
- 用 `kernel_dsv4_hc_split_weighted_sum_norm4` 的思路替换当前 CB-A + CPU 读回 + CMD2 流程。
- 预计收益：省掉 ~3 wait/layer × 43 × 2–13ms ≈ **0.25–1.7s/token**。

#### Stage C：GPU-only routing（1 周）

- 启用 `moe_route_gpu`（已存在）。
- gate matmul 输出不读回 CPU，直接进 GPU router kernel。
- 预计收益：省掉 CMD2 后的一次 wait + CPU softmax/topK。

#### Stage D：Batch 层到 1–2 个 CB（2–4 周）

- 重构 `engine.c` 层循环，把 43 层 encode 进 1–2 个 command buffer。
- 需要先把 expert IO 从 sync pread 改为 async / mmap，否则无法连续 encode。

#### Stage E：Kernel 细节优化（持续）

- MXFP4 v2 coalesced down-proj
- `moe_combine_residual` 式融合
- attention `wo_a` Q8_0 启用

### 41.5 为什么不先转 GGUF 再跑 ds4

可以做一个 sidecar 方案（dmlx server 调用 ds4 推理），但这意味着：

1. 维护两套模型格式（safetensors + GGUF）。
2. dmlx 的 native engine 路径被废弃，server/tokenizer/KV cache 都要与 ds4 对接。
3. 失去 dmlx 现有架构的灵活性（MLX path、flash-moe path、native path 统一）。

除非 ds4 性能遥遥领先且无法通过移植追上，否则不应走 sidecar。

### 41.6 决策建议

| 方案 | 推荐度 | 理由 |
|------|--------|------|
| 直接整体迁移 ds4 | ❌ 不推荐 | 格式/架构不匹配，等于重写引擎 |
| 直接整体迁移 flash-moe | ❌ 不推荐 | Qwen3.5 专用，DSV4 用不上 |
| **借鉴 ds4 架构，逐步移植到 dmlx** | ✅ **推荐** | 可控、保留现有代码、ROI 高 |
| 先做 sidecar ds4 验证性能 | ⬜ 可选 | 快速确认天花板，但不应作为最终方案 |

**下一步**：先做 ds4 真实性能验证（Stage A），然后优先投入 GPU-only mHC（Stage B）。


---

## 42. 2026-06-15 GPU-only mHC post 落地：结果与下一步

### 42.1 改动内容

基于 §41 结论，实施第一阶段 ds4 架构移植：**GPU-only mHC post**。

修改 `src/metal_infer/engine.c`：

1. **CB2 的 mHC post(attn)**：把原来的
   ```
   f32→bf16 + mhc_post_bfloat + bf16→f32
   ```
   换成单 encoder `mhc_post_ffn_expand4`。

2. **cb3 的 mHC post(ffn)**：同样换成单 encoder `mhc_post_ffn_expand4`。

3. **消除 CPU 往返**：
   - 不再把 `buf_mhc_post_weights` / `buf_mhc_comb_weights` 读回 CPU。
   - 不再从 CPU 数组 `post` / `comb` 拷回 GPU。
   - `ffn_out` 只在最后一次性上传到 GPU scratch buffer `buf_hidden`。

### 42.2 验证结果

```bash
bash scripts/run_benchmark.sh --native
```

| 指标 | 改动前 | 改动后 | 变化 |
|------|--------|--------|------|
| tok/s | 0.568 | **0.621** | **+9.3%** |
| Paris | ✓ PASS | ✓ PASS | — |
| 7-prompt E2E | 7/7 | 7/7 | — |
| unit tests | PASS (430+) | PASS (430+) | — |

### 42.3 关键发现

- `mhc_post_ffn_expand4` 本身是正确的。之前被标记为 "buggy" 是误判，真实原因很可能是当时并发的其他改动（如 attention 重写、MoE dispatch bug）。
- 单这一项优化就带来 **+9.3%**，说明 ds4 文档里关于 **“mHC CPU round-trip 是主要 overhead 之一”** 的判断是对的。
- 但 0.621 tok/s 仍低于历史 0.7 基线，更低于 ds4 的 2+ tok/s。主因仍是：
  - CB2 末尾仍有 `waitUntilCompleted`（routing scores 读回 CPU）。
  - 每层仍有多个 CB wait。
  - Expert I/O 仍是 sync pread。

### 42.4 下一步建议（按 ROI 排序）

#### 1. 启用 GPU-only routing（预计 +5–15%）

当前 `moe_route_gpu` 已实现但只在 `smelt_penalty > 0` 时启用。正确做法是：

- 让 `moe_route_gpu`  always 跑，输出 `selected[6]` + `weights[6]` 到 GPU buffer。
- CPU 只在需要时从 GPU buffer 读取 selected/weights（用于 SSD pread 或 cache lookup）。
- 这样可以去掉 CB2 末尾的 `waitUntilCompleted`。

这是 **投入小、收益明确** 的下一步。

#### 2. 合并 shared expert 与 MoE combine（预计 +5–10%）

当前：
- MoE → `buf_hidden` → CPU copy → `ffn_out` → CPU shared add → GPU mHC post。
- shared expert 单独一个 CB + wait。

改成：
- MoE 输出保留在 `buf_hidden`。
- shared expert 输出加到 `buf_hidden` 上（GPU in-place add）。
- mHC post 直接读 `buf_hidden`。

这样省掉一次 CPU↔GPU 和 shared expert 的 CB wait。

#### 3. Batch CommandBuffer（预计 +30–50%，但工作量大）

把 43 层 encode 进 1–2 个 CB。这是 ds4 最大的架构优势，也是达到 1.0+ tok/s 的关键。

#### 4. MXFP4 v2 down-proj kernel（预计 +5–10%，但已不是主瓶颈）

### 42.5 建议立即执行的下一项

推荐 **GPU-only routing**。它直接消除 CB2 末尾的 wait，与本次 mHC post 优化形成接力，预计能把 tok/s 从 0.621 推到 **0.7–0.75**。


---

## 43. 2026-06-15 GPU-only routing 全部启用

### 43.1 改动内容

把 `src/metal_infer/engine.c` 中 GPU routing 的启用条件从

```c
if (!use_hash_routing && eng->smelt_warmup_done && eng->smelt_penalty > 0.0f && ...)
```

放宽为

```c
if (!use_hash_routing && eng->smelt_warmup_done && ...)
```

即：**所有预热完成后的 score-based 层都走 GPU routing**，无论 SMELT penalty 是否为 0。

同时修复 `src/models/moe_kernel.metal` 中的 `moe_route_gpu`，使其与 `cpu_moe_route` 严格一致：

1. **bias 位置**：之前把 bias 加到 sqrtsoftplus 后的 score 上，这是错的。MLX 的 `e_score_correction_bias` 是加在 logits 上再跑 softplus。
2. **bf16 截断**：CPU 路由前会把 gate logits 截断到 bf16；GPU 现在也做同样的截断。
3. **penalty 只影响选择**：权重仍使用未惩罚的原始 score，与 CPU 一致。
4. **has_smelt=0 时忽略 cached 标志**：当 penalty=0 时，不再读取 `buf_cached_flags`。

### 43.2 验证结果

```bash
bash scripts/run_benchmark.sh --native
```

| 指标 | 改动前 (mHC post only) | GPU routing 初版 | GPU routing 修正后 |
|------|------------------------|------------------|--------------------|
| Paris | ✓ | ✗ | ✓ |
| E2E | 7/7 | 5/7 | 7/7 |
| tok/s | 0.630 | 0.253 (cache miss 导致 SSD 回退) | **0.621** |
| unit | PASS | PASS | PASS |

### 43.3 关键发现

- **初版 GPU routing 把 bias 加错位置**，导致选出的 top-6 expert 与 CPU 不同，SMELT cache hit 率暴跌，性能掉到 0.25 tok/s（大量走 SSD）。
- 修正后 correctness 恢复，但 **tok/s 与 CPU routing 持平**（0.621 vs 0.630）。
- 这说明：**当前瓶颈不是 CPU sort 的开销，而是 CB2 末尾的 `waitUntilCompleted`**。只要还需要把 routing 结果同步读回 CPU 来启动 SSD/RAM expert I/O，就无法省掉这个 wait。

### 43.4 结论

GPU routing 本身是正确的、架构上更干净的选择（减少 CPU 工作、减少 readback、为后续 batch/async 铺路），但在现有 **"每 token 同步读回 expert ids"** 的架构下，它不会提升 tok/s。

要真正跨过 0.7 tok/s 回到历史基线，必须减少 command buffer wait 的次数。下一步有两个方向：

1. **合并 CB2/CB3 与 MoE 的前置步骤**：如果能把 routing 输出直接喂给 MoE 的专家选择逻辑，并且专家数据已经在 GPU/RAM，就可以去掉 CB2 的 wait。
2. **batch 多层进一个 command buffer**：把 43 层的 attention/mhc/routing encode 到 1–2 个 CB，消除每层的 3 次 wait。这是 ds4 达到高速的核心架构。

---

## 44. 2026-06-16 源码级深度分析与最终决策建议

> **调查目标**：在 §41–§43 的基础上，再次深入阅读 `../ds4/`、`../flash-moe/` 与 dmlx 当前 native engine 的源码，回答核心问题——**是否应该直接迁移 ds4 的实现来提升性能？** 并给出可执行的下一步路径。
> **阅读范围**：
> - `/Users/zouyee/work/code/ds4/ds4.c`（27,725 行，GGUF 加载 + CPU reference + 层循环）
> - `/Users/zouyee/work/code/ds4/ds4_metal.m`（26,629 行，Metal runtime + batch CB + streaming cache）
> - `/Users/zouyee/work/code/ds4/metal/dsv4_hc.metal`、`dsv4_misc.metal`、`dsv4_rope.metal`、`norm.metal`、`moe.metal`
> - `/Users/zouyee/work/code/flash-moe/metal_infer/infer.m`（~7,151 行，Qwen3.5 engine）
> - `/Users/zouyee/work/code/flash-moe/metal_infer/shaders.metal`（1,296 行，affine 4-bit / GatedDeltaNet kernels）
> - dmlx `src/metal_infer/engine.c`、`engine.h`、`mla_attention.m`、`native_engine.zig`、`src/models/moe_kernel.metal`

### 44.1 结论先行

**不应直接迁移 ds4 或 flash-moe 的实现。** 两者与 dmlx 在模型格式、量化方式、attention 架构上存在根本性差异，直接迁移等价于重写引擎。

**正确路线**：以 ds4 为架构参考，将其核心调度思想（GPU-only mHC / GPU routing / batch command buffer / mmap NoCopy）逐步移植到 dmlx native engine，保留 dmlx 现有的 MLX safetensors 加载、SMELT 缓存、server 框架。flash-moe 仅作为 kernel 微优化与 I/O 工程经验的参考，其 attention 路径对 V4 MLA 不可用。

### 44.2 ds4 源码关键发现

#### 44.2.1 模型格式与量化：与 dmlx 不兼容

| 维度 | ds4 | dmlx native |
|------|-----|-------------|
| 文件格式 | **GGUF** | MLX safetensors |
| Expert 量化 | `Q2_K` / `Q4_K` / `IQ2_XXS`（GGUF block） | **MXFP4**（`uint8 E8M0 scale + nibble LUT`） |
| Attention / shared | `Q8_0` / F16 | affine 4-bit / MXFP4 |
| 权重命名 | `blk.N.*` | `model.layers.N.*` |
| 加载方式 | 单文件 mmap + `newBufferWithBytesNoCopy` | safetensors → GPU buffer |

ds4 的 Metal kernels 直接解码 GGUF block 布局，例如 `kernel_mul_mv_id_q4_k_f32`（`metal/moe.metal:892`）、`kernel_mul_mv_id_q8_0_f32`（`metal/moe.metal:895`）。这些 block 结构在 dmlx 中不存在，因此 ds4 的 expert matvec kernel **不能原样复用**。

#### 44.2.2 Attention：思想可借鉴，维度需重写

ds4 的 `kernel_dsv4_indexed_mixed_attention_heads8`（`metal/dsv4_misc.metal:577`）把 raw sliding-window KV + ratio-4/ratio-128 压缩 KV + top-k 索引 + sink logit 融进一个 decode kernel。该算法与 V4 的 MLA + compressor/indexer 语义相近，但实现参数不同：
- ds4：`head_dim=128`，`n_head=64`，`n_head_kv=1`（`ds4.c:115, 3943`）
- dmlx：`HEAD_DIM=512`，`N_HEADS=64`，`KV_LORA_RANK=512`（`engine.h:34–39`）

因此不能直接搬 kernel，需要把 128-wide Q/K dot 改成 512-wide MLA latent，KV cache 从 raw+comp 改成 MLA latent。

#### 44.2.3 mHC：最可移植、ROI 最高的部分

ds4 已经把 mHC pre split + weighted sum + RMSNorm fuse 成单一 kernel：
- `kernel_dsv4_hc_split_weighted_sum_norm4`（`metal/dsv4_hc.metal:394`）
- `kernel_dsv4_hc_expand4`（`metal/dsv4_hc.metal:579`）
- `kernel_dsv4_q8_hc_expand4_q8_0`：attention output + HC expand 融合（`metal/dsv4_hc.metal:752`）
- `kernel_dsv4_shared_down_hc_expand4_q8_0`：shared down + routed combine + HC expand 融合（`metal/dsv4_hc.metal:631`）

这与 dmlx `MHC_MULT=4`（`engine.h:44`）对应，但权重 shape 与精度链需调整。dmlx 已经在 §42 中迈出了第一步（`mhc_post_ffn_expand4`），验证了 +9.3% 的收益。

#### 44.2.4 调度架构：ds4 真正快的原因

ds4 使用全局 batch command buffer：`g_batch_cb` / `g_batch_enc`（`ds4_metal.m:543–572`），`ds4_gpu_begin_commands` / `flush_commands` / `end_commands`（`ds4_metal.m:6223–6422`）。decode 每 token 通常只 `end` 一次，即 **每 token 1 次 GPU 同步**。

dmlx 当前 `moe_infer_forward_layer`（`engine.c:1051–1636`）内部多次 `waitUntilCompleted`（CB1、CB2、CMD3 deferred），43 层约 215 次 wait/token。这是 dmlx 0.621 tok/s 与 ds4 2+ tok/s 差距的**主要来源**。

#### 44.2.5 Expert I/O：ds4 的 streaming cache 成熟

ds4 使用 slab + mlock + parallel pread 线程池（`ds4_metal.m:671–10930`），单专家单文件、mmap NoCopy。dmlx 的 SMELT 是其简化版，已验证在 Trust OS 模式下 RSS 更低、速度更快。ds4 的 I/O 工程经验可以借鉴，但不应替换 SMELT（V4 expert 更大，SSD-only 路径更慢，SMELT RAM 命中是核心优势）。

### 44.3 flash-moe 源码关键发现

flash-moe 是 **Qwen3.5-397B 专用引擎**，其 GQA + GatedDeltaNet attention 对 V4 MLA **完全不可用**。仅以下技术可借鉴：

#### 44.3.1 可直接借鉴的 kernel / I/O 技术

| 技术 | 来源 | 移植难度 | 预期收益 |
|------|------|----------|----------|
| FMA 反量化代数重组：`fma(nibble, scale*x, bias*x)` | `shaders.metal:314–330` | 低 | ~12%（flash-moe 实测） |
| 单专家连续打包 + 单 `pread` | `infer.m:2736–2746` | 中 | 减少小读取次数 |
| 2MB 对齐 DMA buffer + `newBufferWithBytesNoCopy` | `infer.m:1118–1142` | 中 | warm cache DMA 3.6× |
| 持久 I/O 线程池（pthread + generation counter） | `infer.m:2973–3058` | 中 | 减少线程调度开销 |
| CMD2 融合：o_proj + residual + norm + routing + shared gate | `infer.m:4749–5027` | 高 | 减少一次 CB round-trip |
| CMD3 deferred + GPU combine + next-layer norm | `infer.m:5354–5430` | 高 | 消除 CB3 wait |
| Batched K expert encoding（2 encoder / expert） | `infer.m:1695–1786` | 中 | 减少 encoder 数量 |

#### 44.3.2 不可直接借鉴的技术

| 技术 | 失败/不适用原因 |
|------|-----------------|
| “Trust the OS” 无缓存 | V4 已用 SMELT 预加载到 RAM，与 SSD streaming 是两套内存模型 |
| 时序路由预测 + 双缓冲 | V4 K=6 且 expert 局部性 ~35%，全中概率 `0.35^6 ≈ 1.8e-3`，预测收益为负 |
| GQA / GatedDeltaNet kernel | V4 是 MLA，kernel 结构完全不同 |
| LZ4 / 2-bit expert 压缩 | flash-moe 实测 2-bit 破坏 JSON/tool calling；LZ4 解压开销 > 收益 |
| 后台 SSD prefetch（`F_RDADVISE`） | Apple Silicon 统一内存：SSD DMA 与 GPU 共享内存控制器，拖慢 GPU |

### 44.4 dmlx 当前 native engine 的关键瓶颈（源码级）

结合 `src/metal_infer/engine.c` 与 `native_engine.zig`：

1. **每层多次 `waitUntilCompleted`**：`moe_infer_forward_layer` 中 CB1（attention / mHC pre）、CB2（o_proj / norm / routing / shared）、CMD3（MoE expert / combine / mHC post）各自 commit+wait，43 层累计 ~215 次同步。
2. **mHC post 已部分 GPU 化**：§42 把 `mhc_post_ffn_expand4` 接入 CB2/CB3，去掉了 CPU 读回，带来 +9.3%。但 `mhc_pre` 仍有 CPU 计算。
3. **GPU routing 已启用但受限于同步读回**：§43 把 `moe_route_gpu` 放宽到所有预热后的 score-based 层，正确性恢复，但 tok/s 不变。因为 routing 结果仍需同步读回 CPU 以启动 expert I/O，CB2 末尾的 wait 省不掉。
4. **Expert I/O 仍是 sync pread**：`engine.c` 中 `readAndAssembleAll` 在 CB2/CB3 之间同步等待专家数据，无法与 GPU 重叠。
5. **临时 MTLBuffer 分配**：每层仍有较多 `newBufferWithBytes` 或 wrapper 分配，ds4 使用持久 GPU-resident buffer。
6. **Q8_0 `wo_a` 已缓存但未启用**：`mla_attention.m:862–868` 附近仍走 f32 dense path，启用后可减少 attention 带宽。

### 44.5 直接迁移 ds4/flash-moe 的可行性分析

| 迁移方案 | 推荐度 | 理由 |
|----------|--------|------|
| **整体替换 dmlx native engine 为 ds4** | ❌ 不推荐 | 模型格式 GGUF vs safetensors；tokenizer、server API、KV cache 全不同；等于重写引擎 |
| **把 ds4 编译成库供 dmlx 调用** | ❌ 不推荐 | 需 safetensors → GGUF 转换；API 适配复杂；丧失 dmlx 多路径统一性 |
| **整体迁移 flash-moe** | ❌ 不推荐 | Qwen3.5 专用，GatedDeltaNet 对 V4 无用 |
| **只移植 ds4/flash-moe 的 Metal kernels** | ⚠️ 局部可行 | 能解决 kernel 效率，但不解决 sync 架构问题；且量化格式需重写 |
| **移植 ds4 调度架构到 dmlx** | ✅ 推荐 | ROI 最高：batch CB、GPU-only mHC/routing、mmap NoCopy；保留 dmlx 现有基础设施 |

### 44.6 正确的性能优化路径（按 ROI 与风险排序）

#### Stage A：kernel 微优化（低风险、立即可做）

1. **FMA 反量化代数重组**
   - 在 `src/models/moe_kernel.metal` / `moe_kernel_f16.metal` 中把 `(nibble*scale+bias)*x` 改为 `fma(nibble, scale*x, bias*x)`。
   - 参考 flash-moe `shaders.metal:314–330`，预计 +10% MoE kernel 时间。

2. **启用已缓存的 Q8_0 `wo_a`**
   - `mla_attention.m:862–868` 附近解除 f32 dense 强制路径。
   - 减少 attention 输出投影带宽。

3. **合并 shared expert 与 MoE combine**
   - 当前 shared expert 单独一个 CB + wait；改为 GPU in-place add 到 `buf_hidden`，再进 mHC post。
   - 参考 ds4 `kernel_dsv4_shared_down_hc_expand4_q8_0`，但需改为 MXFP4 / affine。

#### Stage B：GPU-only mHC pre（中风险、高收益）

- 当前 `mhc_pre` 仍在 CPU 算 sinkhorn + weighted sum。
- 移植 ds4 `kernel_dsv4_hc_split_weighted_sum_norm4` 思想，把 RMSNorm + mix + weighted sum 放到 GPU。
- 预计收益：省掉 CB1 前/后的 CPU 读回，与 §42 的 mHC post 形成闭环。

#### Stage C：GPU routing 真正异步化（中高风险、关键）

- §43 已把 `moe_route_gpu` 启用，但 routing 结果仍需同步读回 CPU 启动 I/O。
- **关键改造**：让 expert 数据在 GPU/RAM 中已就绪（SMELT cache 命中），routing 输出直接用于 GPU 端 expert selection，不再读回 CPU。
- 只有当 cache miss 时才回退到 CPU routing + sync I/O。
- 这一步能真正去掉 CB2 末尾的 wait，是跨越 0.7 tok/s 的关键。

#### Stage D：Batch 多层到 1–2 个 CommandBuffer（高风险、最大收益）

- 重构 `engine.c` 层循环，把 43 层的 attention/mhc/routing/MoE encode 到 1–2 个 command buffer。
- 前提：
  - expert I/O 改为 async / mmap / NoCopy，或 SMELT 保证所有 needed expert 已在 RAM。
  - GPU routing 完全异步。
  - 层间 buffer 全部 persistent GPU-resident。
- 参考 ds4 `g_batch_cb` / `g_batch_enc`（`ds4_metal.m:543–572`）。
- 预计收益：从 0.621 tok/s 提升到 1.0+ tok/s，是达到 3.0 tok/s 目标的必由之路。

#### Stage E：I/O 与内存优化（长期）

- mmap expert + `newBufferWithBytesNoCopy`：省掉 pread → CPU scratch → GPU。
- FP8 KV cache：ds4 已有 `kernel_dsv4_fp8_kv_quantize_f32`（`metal/dsv4_kv.metal`），可借鉴减少 KV 内存。
- 持久 buffer pool：消除每 layer 的 MTLBuffer 分配。

### 44.7 决策建议

1. **放弃“直接迁移 ds4 实现”的想法**。ds4 与 dmlx 格式/架构不匹配，直接迁移成本极高且收益不确定。
2. **立即执行 Stage A**：FMA 反量化 + 启用 Q8_0 `wo_a` + shared expert 合并。这是风险最低、最快速的收益。
3. **本阶段重点攻坚 Stage C**：让 GPU routing 真正异步化。只有 routing 不阻塞 CB2，才能释放 §42/§43 的架构收益。
4. **把 Stage D（batch CB）作为中期里程碑**。不要跳过 Stage C 直接做 Stage D，否则 I/O 会打断 batch。
5. **保持 MLX oracle 路径可用**，所有 native engine 改动必须 7/7 E2E 通过才能合并。

### 44.8 测试与内存安全纪律（鉴于近期 OOM 事故）

- **严禁同时运行多个 serve 实例**。每个实例都会独立加载模型权重，48GB 机器上两个实例即可触发 macOS compressor / OOM。
- **测试必须串行**：`--native` 测试结束后 `kill` 进程，确认端口释放、RSS 下降，再启动下一测试。
- **监控 RSS 与 swap**：benchmark 过程中用 `vm_stat 1` 或 `Activity Monitor` 观察，若 swap 增长立即停测。
- **保守调整 SMELT**：`NATIVE_SMELT_N` 从 20 开始逐步上调，不要一次性调到 51（已知 OOM）。
- **改 kernel 后先跑隔离测试**：
  ```bash
  bash scripts/run_kernel_tests.sh        # ~2s
  bash scripts/run_mla_attention_test.sh  # ~2s
  bash scripts/dsv4_smoke.sh              # E2E 正确性
  bash scripts/run_benchmark.sh --native  # 性能（串行、单实例）
  ```

### 44.9 一句话总结

**不要迁移 ds4，而要“ds4 化” dmlx**：保留 dmlx 的模型加载与 server，把 ds4 的 batch command buffer、GPU-only mHC/routing、mmap NoCopy 调度思想逐步移植进来；flash-moe 只取其 kernel 微优化与 I/O 工程经验。当前最高优先级是让 GPU routing 真正异步化，并减少 command buffer wait 次数，而非重写 expert kernel。**

---

## 45. 2026-06-16 「ds4 化 dmlx」完整移植路线图

> **目标**：在保留 dmlx 现有 MLX safetensors 加载、server 框架、SMELT 缓存的前提下，把 ds4 的调度架构优势逐步移植到 native engine，实现性能从当前 **~0.62 tok/s** 到 **≥1.5 tok/s（中期）**、最终 **≥3.0 tok/s** 的提升。
> **总原则**：逐阶段灰度、每阶段必须有 7/7 E2E 通过、性能不退化才合并。

### 45.1 阶段总览

| 阶段 | 主题 | 主要改动文件 | 预期 tok/s | 风险 | 依赖 |
|------|------|-------------|-----------|------|------|
| **Stage 0** | 基线与护栏 | `scripts/run_benchmark.sh`, `scripts/dsv4_smoke.sh` | 0.621（基线） | 低 | 无 |
| **Stage 1** | Kernel 微优化 | `src/models/moe_kernel.metal`, `src/metal_infer/mla_attention.m`, `src/metal_infer/engine.c` | 0.70–0.75 | 低 | Stage 0 |
| **Stage 2** | GPU-only mHC pre | `src/metal_infer/engine.c`, `src/models/moe_kernel.metal`, 新增 `src/metal_infer/mhc_gpu.metal` | 0.80–0.90 | 中 | Stage 1 |
| **Stage 3** | GPU routing 真正异步化 | `src/metal_infer/engine.c`, `src/models/moe_kernel.metal`, `src/native_engine.zig` | 1.00–1.20 | 高 | Stage 2 |
| **Stage 4** | 合并 CB2/CB3 + MoE combine | `src/metal_infer/engine.c`, `src/models/moe_kernel.metal` | 1.30–1.60 | 高 | Stage 3 |
| **Stage 5** | Batch 多层 CommandBuffer | `src/metal_infer/engine.c`, `src/native_engine.zig`, `src/metal_infer/engine.h` | 2.00–3.00 | 很高 | Stage 4 |
| **Stage 6** | mmap NoCopy expert buffer | `src/models/expert_stream.zig`, `src/models/expert_pread.zig`, `src/metal_infer/engine.c` | 2.50–3.50（与 Stage 5 叠加） | 中 | Stage 4/5 |
| **Stage 7** | Attention kernel 深度优化 | `src/metal_infer/mla_attention.m`, `src/models/moe_kernel.metal` | 3.00–4.00 | 中 | Stage 5/6 |

> **注**：以上收益为估算，实际取决于每阶段成功消除的 sync 数量与 kernel 效率提升。Stage 1–4 是「追上 MLX 1.6 tok/s」的关键；Stage 5–7 是「超越 MLX、接近 flash-moe」的关键。

### 45.2 Stage 0：基线验证与护栏（0.5 天）

#### 目标
- 确立可复现的性能与正确性基线。
- 确保后续任何改动都有明确的回退标准。

#### 关键改动
1. 在 `scripts/run_benchmark.sh` 中固定以下参数并打印到日志：
   - `NATIVE_SMELT_N=20`
   - `SMELT_N=0.20`
   - 温度 0，max_tokens=64
2. 在 `scripts/dsv4_smoke.sh` 中增加 `--native` 路径的独立运行选项。
3. 记录当前基线：
   ```bash
   bash scripts/run_benchmark.sh --native
   # 期望：tok/s ≈ 0.621，Paris ✓，7/7 E2E ✓，unit tests ✓
   ```

#### 验收标准
- 连续 3 次 `run_benchmark.sh --native` 的 tok/s 波动 < 5%。
- `dsv4_smoke.sh` 7/7 通过。
- 430+ unit tests 通过。

#### 风险
- 基线不可复会掩盖后续优化效果。必须固定 SMELT_N 与硬件状态（插电、无其他重负载）。

---

### 45.3 Stage 1：低风险 Kernel 微优化（2–3 天）

#### 目标
在不改变调度架构的前提下，通过 kernel 微优化和启用已有缓存提升性能。

#### 关键改动

**1.1 FMA 反量化代数重组（flash-moe 经验）**
- 文件：`src/models/moe_kernel.metal`, `src/models/moe_kernel_f16.metal`
- 改动：将 `(nibble * scale + bias) * x` 重排为 `fma(nibble, scale * x, bias * x)`。
- 参考 flash-moe `shaders.metal:314–330`。
- 影响 kernel：`dequant_matvec_4bit_*`, `dequant_matvec_affine_*`。

**1.2 启用已缓存的 Q8_0 `wo_a`**
- 文件：`src/metal_infer/mla_attention.m:862–868` 附近
- 改动：解除 f32 dense 强制路径，当 `wo_a_scales` 存在时走 Q8_0/affine 4-bit matvec。
- 验证：跑 `run_mla_attention_test.sh`，rel_L2 不劣化。

**1.3 合并 shared expert 与 MoE combine**
- 文件：`src/metal_infer/engine.c`
- 改动：
  - 当前：MoE → `buf_hidden` → CPU copy → `ffn_out` → CPU shared add → GPU mHC post。
  - 目标：shared expert 输出直接 GPU in-place 加到 `buf_hidden`，再进 mHC post，省掉一次 CPU↔GPU 和 shared expert 的 CB wait。
- 可能需要新 kernel：`shared_add_mhc_post` 或扩展 `moe_combine_residual` 语义。

#### 验收标准
- `run_kernel_tests.sh` 全部通过。
- `run_mla_attention_test.sh` rel_L2 ≤ 1.9e-6。
- `dsv4_smoke.sh` 7/7 通过。
- `run_benchmark.sh --native` tok/s ≥ 0.70（目标 0.75）。

#### 风险
- FMA 重排可能引入微小数值差异，需逐 kernel 对拍。
- shared expert 合并可能改变 mHC post 的输入精度，需 E2E 验证。

#### 回退
- 任一 kernel 测试失败即回退该 kernel 改动，其他改动可独立合并。

---

### 45.4 Stage 2：GPU-only mHC pre（5–7 天）

#### 目标
消除 mHC pre 阶段的 CPU↔GPU 往返，把 `post[]` / `comb[]` 放到 persistent GPU buffer，用 GPU kernel 完成 sinkhorn + weighted sum + RMSNorm。

#### 关键改动

**2.1 Persistent GPU buffer for post/comb**
- 文件：`src/metal_infer/engine.h`, `src/metal_infer/engine.c`
- 改动：
  - 在 `MoEInferEngine` 中新增 `id<MTLBuffer> buf_post_persistent`、`buf_comb_persistent`。
  - 每层初始化时从 host 上传一次 `post` / `comb` 权重到 GPU，后续不再每 token 上传。

**2.2 GPU mHC pre kernel**
- 新增文件：`src/metal_infer/mhc_gpu.metal`（或复用 `src/models/moe_kernel.metal`）
- 实现参考 ds4 `kernel_dsv4_hc_split_weighted_sum_norm4`（`metal/dsv4_hc.metal:394`）：
  - 输入：`residual[MHC_MULT, DIM]`（来自上一层 mHC post 输出）
  - 中间：`mixes` → sigmoid → `pre_mix[4]`
  - 输出：`out_input[DIM]` = `sum_m pre_mix[m] * residual[m, :]`
  - 可选：在同一 kernel 内接 input RMSNorm。
- 精度：保持 f32 accumulation，输出 bf16/f32 视后续 Stage 决定。

**2.3 修改 `moe_infer_forward_layer` 调用顺序**
- 文件：`src/metal_infer/engine.c`
- 改动：
  - 移除 `mhc_pre` 的 CPU 计算路径。
  - CB1 开头直接 dispatch GPU mHC pre kernel。
  - 不再从 GPU readback `post` / `comb`。

#### 验收标准
- 新增 `run_mhc_gpu_test.sh`（或用 `gen_mhc_golden.py` 生成 golden），GPU mHC pre vs CPU reference max_abs ≤ 1e-6。
- `run_mla_attention_test.sh` 仍通过。
- `dsv4_smoke.sh` 7/7 通过。
- `run_benchmark.sh --native` tok/s ≥ 0.80。

#### 风险
- mHC pre 是 attention 的输入，精度敏感。GPU 归约顺序可能与 CPU 有微小差异，需与 MLX 端到端验证（不追求逐层 hidden state 一致，只要求输出正确）。
- sinkhorn 双随机矩阵的数值稳定性需重点验证。

#### 回退
- 若 E2E 正确性不达标，保留 persistent buffer，回退到 CPU sinkhorn + GPU weighted sum 的混合方案。

---

### 45.5 Stage 3：GPU routing 真正异步化（7–10 天）

#### 目标
让 routing 结果不再同步读回 CPU， expert 数据在 SMELT cache 命中时直接由 GPU 端选择并计算。

#### 关键改动

**3.1 GPU router kernel 修正与增强**
- 文件：`src/models/moe_kernel.metal` 中 `moe_route_gpu`
- 改动：
  - 输出 `selected[6]` 和 `weights[6]` 到 GPU buffer（已是当前行为，但需确保格式稳定）。
  - 增加 `has_smelt=1` 路径：直接输出 `expert_ids_for_gpu[6]`，供 GPU expert selection 使用。
  - 保持 `has_smelt=0` 或 cache miss 时回退到 CPU routing + sync I/O。

**3.2 SMELT cache 命中路径改造**
- 文件：`src/native_engine.zig`, `src/models/expert_stream.zig`, `src/metal_infer/engine.c`
- 改动：
  - 在 SMELT warmup 阶段，把预加载的 expert 数据组织为 GPU 可访问的 buffer（或保证在 RAM 中可被快速 bind）。
  - 当 `moe_route_gpu` 输出的 6 个 expert 全部命中 SMELT 时，CB2 不 `waitUntilCompleted`，直接由 GPU 读取专家权重并继续。
  - cache miss 时：fallback 到当前 sync pread 路径。

**3.3 引入 "GPU-resident expert slab"**
- 文件：`src/metal_infer/engine.h`, `src/metal_infer/engine.c`
- 改动：
  - 分配一块 persistent GPU buffer（或 pinned RAM buffer）作为 expert slab。
  - SMELT 命中的 expert 数据以固定 stride 放在 slab 中。
  - GPU expert matvec kernel 通过 `expert_id` 计算 offset，而不是通过 CPU 传指针。

**3.4 修改 `moe_infer_forward_layer` 层循环**
- 文件：`src/metal_infer/engine.c`
- 目标结构：
  ```
  CB1: mHC pre → input RMSNorm → attention → mHC post
  CB2: ffn RMSNorm → gate proj → GPU routing → (no wait if SMELT hit) → MoE expert dispatch
  CB3: MoE combine + shared add + mHC post (deferred)
  ```
- CB2 末尾的 `waitUntilCompleted` 仅在 cache miss 时执行。

#### 验收标准
- `dsv4_smoke.sh` 7/7 通过。
- `run_benchmark.sh --native` tok/s ≥ 1.00。
- SMELT cache hit 率 ≥ 95%（从日志或 MF_DBG 确认）。
- cache miss fallback 路径仍正确（临时调低 SMELT_N 测试）。

#### 风险
- **这是最高风险阶段**。一旦 GPU routing 与 CPU routing 不一致，会导致 expert 选择错误、输出乱码。
- SMELT cache 命中判断必须在 GPU 端或 host 端与 GPU 同步，设计不当会引入新的 sync。
- GPU slab 内存占用大，需仔细管理（48GB 机器上 20 experts × 13.4MB ≈ 268MB，可接受）。

#### 回退
- 若正确性无法稳定，保留 `moe_route_gpu` 但恢复 CB2 wait，仅作为代码结构优化合并，不强行提升性能。

---

### 45.6 Stage 4：合并 CB2/CB3 + MoE combine 融合（5–7 天）

#### 目标
进一步减少 command buffer 数量，把 routing、MoE expert、combine、shared add、mHC post 尽量合并到同一 CB。

#### 关键改动

**4.1 GPU combine + residual + shared gate**
- 文件：`src/models/moe_kernel.metal`
- 新增/扩展 kernel：`moe_combine_residual_shared`
- 功能：
  ```
  hidden = h_mid + Σ weights[k] * expert_down[k] + sigmoid(shared_gate) * shared_out
  ```
- 参考 flash-moe `moe_combine_residual`（`shaders.metal:1261–1296`）和 ds4 `kernel_dsv4_shared_down_hc_expand4_q8_0`。

**4.2 合并 CB2 与 CB3**
- 文件：`src/metal_infer/engine.c`
- 改动：
  - 当前：CB2（routing）→ wait → CB3（MoE + combine + mHC post）。
  - 目标：在 SMELT hit 时，CB2 编码 routing + MoE + combine + mHC post，一次 commit，deferred wait。
  - 这需要 Stage 3 的 GPU-resident expert slab 支持。

**4.3 消除 routing 后 CPU 参与**
- 在 Stage 3 基础上，确保 combine 后的 mHC post 输入直接是下一层输入，不需要 CPU 中转。

#### 验收标准
- `dsv4_smoke.sh` 7/7 通过。
- `run_benchmark.sh --native` tok/s ≥ 1.30。
- 每 token command buffer wait 次数从 ~215 降到 ~100 以下。

#### 风险
- kernel 融合可能改变数值精度，需与 Stage 1 前的基线 E2E 对比。
- shared gate 的 sigmoid 精度需与 MLX 对齐。

---

### 45.7 Stage 5：Batch 多层 CommandBuffer（2–4 周）

#### 目标
实现 ds4 最大的架构优势：把 43 层 encode 进 1–2 个 command buffer，每 token 只 wait 1–2 次。

#### 关键改动

**5.1 Persistent layer state on GPU**
- 文件：`src/metal_infer/engine.h`, `src/metal_infer/engine.c`
- 改动：
  - KV cache、hidden state、expert slab、post/comb weights 全部 GPU-resident。
  - 层间不读回 CPU，只传递 position/seq_len 等标量。

**5.2 重构层循环为 batch encoder**
- 文件：`src/metal_infer/engine.c` 中 `moe_infer_forward`
- 目标结构：
  ```objc
  ds4_gpu_begin_commands();  // open CB #1
  for layer = 0..42:
      encode_layer(layer, encoder);  // 复用同一个 MTLComputeCommandEncoder
      if (layer == 4) flush_commands();  // commit CB #1 async, open CB #2
  ds4_gpu_end_commands();  // commit CB #2, wait both
  logits readback
  ```
- 参考 ds4 `g_batch_cb` / `g_batch_enc`（`ds4_metal.m:543–572`）和 `ds4_gpu_begin/end_commands`（`ds4_metal.m:6223–6422`）。

**5.3 Encoder 复用**
- 在同一 CB 内，所有 kernel dispatch 复用同一个 `MTLComputeCommandEncoder`。
- 当前每个 kernel 都 `endEncoding` + 新建 encoder，开销大。

**5.4 层权重指针缓存**
- 文件：`src/native_engine.zig` 或 `src/metal_infer/engine.c`
- 改动：启动时缓存每层所有 weight buffer 指针，运行时不再 `snprintf` + hash 查找。
- 参考 flash-moe `infer.m:3644–3804`。

#### 验收标准
- `dsv4_smoke.sh` 7/7 通过。
- `run_benchmark.sh --native` tok/s ≥ 2.00。
- 每 token `waitUntilCompleted` 次数 ≤ 5。

#### 风险
- **这是工程最大、风险最高的阶段**。需要重写 `engine.c` 层循环。
- 所有中间 buffer 生命周期需重新设计，极易 segfault。
- 43 层全部在一个 CB 中可能导致单个 CB 过大，驱动/OS 限制需分 CB（如 ds4 在 layer 4 分一次）。

#### 回退
- 可分两步：先 batch 2–4 层，验证稳定后再 batch 全部 43 层。

---

### 45.8 Stage 6：mmap NoCopy expert buffer（5–7 天，可与 Stage 5 并行）

#### 目标
用 mmap + `newBufferWithBytesNoCopy` 替代 pread → CPU scratch → GPU upload，减少 expert I/O 开销。

#### 关键改动

**6.1 专家文件 mmap**
- 文件：`src/models/expert_stream.zig`, `src/models/expert_pread.zig`
- 改动：
  - 对 packed_experts 或 safetensors shard 做 `mmap`。
  - 用 `posix_memalign(2MB)` + `newBufferWithBytesNoCopy` 创建 Metal buffer。
  - 参考 flash-moe `infer.m:1118–1142`。

**6.2 GPU kernel 通过 offset 访问 expert**
- 文件：`src/models/moe_kernel.metal`
- 改动：expert matvec kernel 接受 base buffer + expert_id offset，而不是每个 expert 单独 bind buffer。
- 参考 ds4 `model->map + abs_offset` 和 flash-moe `setBuffer:offset:atIndex:`。

**6.3 与 SMELT 的关系**
- SMELT 预加载到 RAM 仍然是主要路径。
- mmap 作为 SSD fallback 和 cold-start 路径，让 OS page cache 自动管理。

#### 验收标准
- `dsv4_smoke.sh` 7/7 通过。
- 冷启动首 token 延迟不劣化。
- 48GB 机器上不触发 OOM（mmap 虚拟地址空间大，但 physical RSS 需监控）。

#### 风险
- mmap 文件与 GPU NoCopy buffer 的对齐要求严格。
- 文件修改后需 munmap/remap，生命周期管理复杂。
- 与 SMELT 的交互需仔细设计，避免双重缓存浪费内存。

---

### 45.9 Stage 7：Attention kernel 深度优化（1–2 周）

#### 目标
优化 attention 阶段的 GPU 计算，追赶 ds4 的 attention 效率。

#### 关键改动

**7.1 Fused MLA attention kernel**
- 文件：`src/metal_infer/mla_attention.m`, `src/models/moe_kernel.metal`
- 目标：把 Q chain、KV chain、SDPA、wo_a、wo_b 尽量融合到更少 kernel dispatch。
- 参考 ds4 `kernel_dsv4_indexed_mixed_attention_heads8`（`metal/dsv4_misc.metal:577`）。

**7.2 Q8_0 / 4-bit attention 权重**
- 文件：`src/metal_infer/mla_attention.m`, `src/models/deepseek_v4_loader.zig`
- 目标：attention 权重也走量化 matvec，减少带宽。
- 注意：当前 loader 可能已把部分权重解量到 bf16，需保留 packed 路径。

**7.3 Flash Attention 风格 prefill（可选）**
- 新增 kernel：多阶段 FlashAttention for prefill。
- 参考 ds4 `metal/flash_attn.metal`。
- 优先级较低，因为当前瓶颈在 decode。

#### 验收标准
- `run_mla_attention_test.sh` rel_L2 ≤ 1.9e-6。
- `dsv4_smoke.sh` 7/7 通过。
- `run_benchmark.sh --native` tok/s 继续提升（目标 ≥ 3.00）。

#### 风险
- Attention 数值精度敏感，融合后更难 debug。
- 需在 `run_kernel_tests.sh` 中增加更多 attention kernel 单测。

---

### 45.10 跨阶段测试纪律

每个 Stage 合并前必须依次通过：

```bash
# 1. 隔离 kernel 测试（每次改 kernel 必跑）
bash scripts/run_kernel_tests.sh

# 2. 注意力 host 编排对拍
bash scripts/run_mla_attention_test.sh

# 3. E2E 正确性
bash scripts/dsv4_smoke.sh
# 期望：France → 含 Paris；2+2 → 含 4；7/7 PASS

# 4. 性能基线（串行、单实例、插电）
bash scripts/run_benchmark.sh --native
# 期望：tok/s 不低于该 Stage 目标，且 Paris ✓、7/7 ✓

# 5. 内存监控
vm_stat 1 > /tmp/vmstat.log &
# 跑 benchmark 过程中观察 swap 是否增长
```

---

### 45.11 回退策略

| 场景 | 回退动作 |
|------|----------|
| 某 Stage E2E 7/7 失败 | 回退该 Stage 所有改动，保留之前已合并 Stage |
| 性能提升但正确性退化 | 禁止合并；必须同时满足正确性 |
| OOM / swap 增长 | 降低 SMELT_N，检查是否有新的 MTLBuffer 泄漏 |
| GPU routing 不稳定 | 保留 kernel，恢复 CB2 wait，作为架构准备合并 |
| Batch CB 导致 segfault | 减少 batch 层数，或分更多 CB |

---

### 45.12 时间预估与里程碑

| 里程碑 | 时间 | 目标 tok/s | 验收 |
|--------|------|-----------|------|
| Stage 0–1 完成 | 第 1 周结束 | ≥ 0.70 | 7/7 + 微优化收益 |
| Stage 2 完成 | 第 2 周结束 | ≥ 0.85 | GPU-only mHC pre |
| Stage 3 完成 | 第 3–4 周结束 | ≥ 1.10 | GPU routing 异步化 |
| Stage 4 完成 | 第 5 周结束 | ≥ 1.50 | CB2/CB3 合并 |
| Stage 5 完成 | 第 7–8 周结束 | ≥ 2.50 | Batch 多层 CB |
| Stage 6 完成 | 第 8–9 周结束 | ≥ 2.80 | mmap NoCopy |
| Stage 7 完成 | 第 10 周结束 | ≥ 3.00 | Attention 优化 |

> **总工期**：约 8–10 周（单线程、保守估算）。若 Stage 3 或 Stage 5 提前完成，可显著缩短。

---

### 45.13 关键成功因素

1. **每阶段必须有 7/7 E2E 通过**，性能提升只是加分项。
2. **不要跳过 Stage 3 直接做 Stage 5**。没有 GPU routing 异步化，batch CB 会被 I/O 打断。
3. **保持 MLX oracle 路径可用**，作为正确性最后防线。
4. **优先消除 sync，再优化 kernel**。当前 0.621 tok/s 的 80% 时间花在同步上。
5. **严格控制内存**。每新增一个 persistent GPU buffer 都要评估 48GB 机器的承受能力。

---

### 45.14 一句话总结路线图

**Stage 1 快速止血 → Stage 2 消除 mHC CPU 往返 → Stage 3 让 routing 不再阻塞 → Stage 4 合并 MoE pipeline → Stage 5 一次性 batch 43 层 → Stage 6 mmap  expert → Stage 7 优化 attention。每一步都 7/7 E2E 通过才继续，否则回退。**

---

## 46. 2026-06-16 深入可行性分析：flash-moe 与 dmlx 前期优化尝试的教训

> **目标**：在 §45 路线图基础上，结合 flash-moe 34 个实验的实测结论与 dmlx 自身前期优化尝试的失败记录，重新评估每条路径的真实可行性，避免重复踩坑，并给出风险修正建议。

### 46.1 核心前提修正

§45 路线图假设「ds4 化 dmlx」能按阶段稳步推进并最终达到 3.0 tok/s。但深入阅读 flash-moe 与 dmlx 历史优化记录后，必须加入以下**硬性约束**：

1. **全 metal 路径的 bf16/f16 精度对齐仍是未解决的根本风险**（§14–§19）。在单层/单步精度改善（0.15% rel_L2）的情况下，多步推理仍会发散。
2. **Apple Silicon 统一内存架构决定了 SSD I/O 与 GPU compute 无法 profitable overlap**。任何依赖「后台 prefetch 与 GPU 并行」的方案都已失败。
3. **DeepSeek-V4 的 expert 局部性（~35%）远低于 flash-moe 的 Qwen3.5（~71%）**，时序预测、 speculation、co-occurrence 等方案的收益天花板更低。
4. **V4 的 score-based routing 对数值精度极度敏感**。borderline expert 的 score 差距约 0.001，kernel 精度或非确定性（`simd_sum`）会导致 expert swap，43 层后输出随机化。

因此，§45 的 Stage 1–7 不是一条平坦的升级路径，而是一条**充满已知陷阱、需要硬止损条件**的高风险路径。

### 46.2 flash-moe 的实验结论（直接来自 `../flash-moe/docs/`）

#### 46.2.1 成功方案

| 方案 | 效果 | 对 dmlx 的适用性 |
|------|------|-----------------|
| **Trust OS / 删除自定义缓存** | +38%（4.36 → 5.74 tok/s） | ✅ dmlx 已默认 `--smelt-cache 0`，适用 |
| **Parallel pread（4 线程）** | +9.2× vs sequential | ⚠️ dmlx 已接入 `expert_pread.zig`，但 Trust OS 下收益被测量噪声覆盖 |
| **2 MB 对齐 DMA buffer** | +3.6× isolated，全管道 ~+5% | ✅ 可移植到 Stage 6 mmap |
| **FMA 反量化 kernel** | +2.6%（4.36 tok/s） | ✅ Stage 1 已规划 |
| **2-bit expert 量化** | 文件 -44%，I/O -42% | ❌ dmlx 因质量门控未采用；flash-moe 也承认 2-bit 破坏 JSON/tool calling |

#### 46.2.2 失败方案（对 dmlx 的直接警示）

| 方案 | flash-moe 结果 | 对 dmlx Stage 1–7 的影响 |
|------|----------------|-------------------------|
| **自定义 Metal LRU cache** | 9.8 GB cache 比无缓存慢；删除后 +38% | ⚠️ Stage 6 若引入用户态 expert slab/cache，必须控制大小，避免挤压 OS page cache |
| **mmap expert files** | 比 pread 慢 5×（page fault 风暴） | ⚠️ Stage 6 mmap 必须只用于 warm/hot path，cold bulk read 仍用 pread |
| **`F_RDADVISE` / `MADV_*` / kernel hints** | 中性或有害 | ❌ 不应再试 |
| **Temporal expert 预测 + 双缓冲** | -18%，命中率 25.6%，全中 0.4% | ❌ Stage 5/6 不应依赖时序预测；V4 K=6 + 35% 局部性更差 |
| **LZ4 / LZFSE expert 压缩** | -13%（warm 下解压 > I/O 节省） | ❌ 不应引入 |
| **dispatch_io / aio_read** | -70% / -7% | ❌ macOS 用户态调度不如 pread |
| **GPU private buffer compression** | -20% 全管道 | ❌ Stage 7 不应采用 |
| **Spin-poll GPU wait** | -23%（CPU thermal） | ❌ 统一架构下抢热预算 |
| **MTP / PLD speculative decoding** | break-even 或更差 | ❌ MoE 每 speculative token 都要 I/O，不划算 |

### 46.3 dmlx 前期优化尝试的失败记录

来自 `docs/analysis/flash-moe-alignment-analysis.md` §3.1、`docs/en/analysis/native-engine-4toks-plan.md` §4 及历史 commit：

| 方案 | 结果 | 失败根因 |
|------|------|----------|
| **Fate cross-layer expert prediction** | 2.2–2.5× 退化 | mHC 表征不匹配，64% 准确率；后台 pread 与主线程争 SSD |
| **Cache-aware routing bias/swap（P1.4）** | 三种方案全失败 | additive bias 低分 expert 涌入；multiplicative bias 开销>收益；post-selection swap 的 `eval()` 破坏 MLX lazy graph |
| **Background pread prefetch** | 2.2× 退化 | SSD 竞争 + LFU 驱逐 |
| **Cross-tensor madvise prefetch** | -50% server tok/s | madvise CPU 开销 > page-in 收益 |
| **Eval skip (every 2 layers)** | -5% | lazy graph 增大内存压力 |
| **LRU cache eviction / 大 cache（10GB+）** | -36% / swap | 挤占 backbone page cache，触发 compressor |
| **Hash routing 确定性预加载** | 负面 | 仅覆盖 3/43 层；Config A cold -39%、warm -24% |
| **Expert Wave Pipeline** | 不适用 | Apple Silicon UMA 无法并行 I/O+compute |
| **SIMD reduction kernel（早期）** | 87–97% 输出为 0 | reduction bug，已回退 naive | |
| **CB merge 多次尝试** | 0 收益 | `b750770`, `40425ab`, `ea18a9c` 等 commit 显示 CB 合并在当前 sync 架构下无效 |
| **mHC fusion + Q8_0 wo_a** | +6.1% | 少数正向优化之一，但收益有限 |
| **coalesced wo_b v2** | +3.5% | 同上 |

### 46.4 对 §45 各 Stage 的可行性重新评估

| Stage | 原计划收益 | 可行性 | 主要风险 |
|-------|-----------|--------|----------|
| **Stage 1：Kernel 微优化** | 0.62 → 0.75 | ✅ **高** | FMA 数值差异、shared combine 精度 |
| **Stage 2：GPU-only mHC pre** | 0.75 → 0.90 | ⚠️ **中** | sinkhorn 数值稳定性；GPU 归约顺序可能与 CPU 不同 |
| **Stage 3：GPU routing 异步化** | 0.90 → 1.20 | ⚠️ **中-低** | 必须配合 SMELT slab 常驻 RAM/GPU；cache miss fallback 会重新引入 sync |
| **Stage 4：CB2/CB3 合并** | 1.20 → 1.50 | ⚠️ **中** | kernel 融合改变精度；shared gate sigmoid 需对齐 |
| **Stage 5：Batch 多层 CB** | 1.50 → 2.50 | ❓ **低-中** | 需要前面所有条件成立；43 层 CB 可能触发驱动/OS 限制；工程风险最高 |
| **Stage 6：mmap NoCopy** | 2.50 → 2.80 | ⚠️ **中** | mmap cold read 慢；与 SMELT 关系需仔细设计 |
| **Stage 7：Attention 优化** | 2.80 → 3.00+ | ⚠️ **中** | attention 精度敏感，融合后更难 debug |

**关键结论**：
- Stage 1 收益确定，应**立即执行**。
- Stage 2–4 是**中等风险、中等收益**，需要严格的数值护栏。
- Stage 5 的 **2.50 tok/s 目标很可能过于乐观**。即使所有 sync 消除，GPU compute 本身的带宽与 kernel 效率可能限制在 1.5–2.0 tok/s。
- **Stage 6 mmap 不是 silver bullet**。flash-moe 的 mmap 优势建立在 Trust OS + 小 expert（7MB）+ 高局部性上；V4 expert 13.4MB、局部性低，mmap cold miss 惩罚更大。

### 46.5 必须加入的硬止损条件

§45 已要求每阶段 7/7 E2E 通过，但还需要：

1. **全 metal 路线止损**：
   - 若 Stage 2（GPU-only mHC pre）完成后，仍无法在 **连续 8 次 run** 中稳定输出 `Paris`（排除 `simd_sum` 非确定性），则终止 metal-first，回退到 MLX 路径优化。
   - 理由：§17 已证明 Metal 非确定性是真实障碍，继续投入可能无法收敛。

2. **内存止损**：
   - 任一 Stage 合并后若 `vm_stat` 显示 `swapouts` 或 `compressor` 活动增长 > 20%，立即 revert 该 Stage。

3. **性能止损**：
   - 若 Stage 3 完成后 tok/s < 0.90，说明 GPU routing 异步化未能消除 CB2 wait，不应继续 Stage 4/5，而是回退分析根因。

4. **时间止损**：
   - Stage 5 若 2 周内无法实现 ≥1.8 tok/s 且 7/7 稳定，则放弃 full batch CB，改为 partial batch（如每 4–8 层一个 CB）。

### 46.6 修正后的优先级与并行路径

基于可行性分析，建议把 §45 的单线顺序改为 **三条并行轨道**：

#### 轨道 A：低风险快速收益（必做）
- Stage 1 kernel 微优化
- 2 MB 对齐 DMA buffer
- DyMoE skip 调优（已存在，需质量门控）
- Expert co-occurrence clustering 离线分析

#### 轨道 B：高风险 metal 架构（设止损）
- Stage 2 GPU-only mHC pre
- Stage 3 GPU routing 异步化
- Stage 4 CB2/CB3 合并
- 若任一里程碑失败，整体停止 metal-first

#### 轨道 C：MLX 路径备份优化（与 B 并行）
- 消除 MLX per-op eval 同步（`mx.compile`、batch decode、op fusion）
- 优化 safetensors 读取路径（header 缓存、按层分 bin）
- 动态 SMELT 预算（避免挤压 OS page cache）

> **轨道 C 是轨道 B 的保险**。若 metal-first 在止损点失败，可立即切换到 MLX 优化，避免项目整体卡死。

### 46.7 不应再进入 Roadmap 的方案

以下方案已在 flash-moe 或 dmlx 自身实验中被证伪，不应浪费人力：

- ❌ Cross-layer / temporal expert prediction
- ❌ Cache-aware routing bias/swap
- ❌ Background SSD prefetch / `F_RDADVISE` / `MADV_*`
- ❌ `dispatch_io` / `aio_read`
- ❌ LZ4 / LZFSE / GPU private expert 压缩
- ❌ MTP / PLD speculative decoding 作为主要优化
- ❌ 自定义大容量 expert cache（> 物理内存 25%）
- ❌ Spin-poll GPU wait
- ❌ mmap 用于 cold bulk expert read

### 46.8 对 3.0 tok/s 目标的现实评估

| 路径 | 最高现实目标 | 置信度 | 条件 |
|------|-------------|--------|------|
| metal-first 成功（Stage 1–7 全部达成） | 2.5–3.0 tok/s | 30% | bf16/f16 精度对齐稳定 + batch CB 成功 + mmap 有效 |
| metal 部分成功（Stage 1–4） | 1.2–1.6 tok/s | 50% | 追上 MLX |
| 回退 MLX + I/O 优化 | 1.0–1.5 tok/s | 60% | 消除 eval 同步 + 优化 safetensors 读取 |
| 维持现状 | 0.62 tok/s | 100% | — |

> **结论**：3.0 tok/s 不是不可能，但需要多个高风险 Stage 同时成功。更现实的目标是 **1.5–2.0 tok/s**，且应以 **轨道 A + 轨道 B（设止损）+ 轨道 C 备份** 的方式推进。

### 46.9 一句话修正总结

**§45 路线图在技术上成立，但成功率被高估。真正确定可行的只有 Stage 1；Stage 2–4 需要硬数值护栏；Stage 5 需要硬时间止损；Stage 6–7 依赖太多前置条件。同时必须并行准备 MLX 路径备份，并彻底放弃 flash-moe/dmlx 都已证伪的 prediction/prefetch/compression/cache 方案。**

---

## 47. 2026-06-16 Stage 1 执行记录：FMA 反量化改写实测失败

> **实验**：按照 §45 Stage 1 Task 1，将 `src/models/moe_kernel.metal` 与 `src/models/moe_kernel_f16.metal` 中所有 `(scale*nibble + bias) * x` 与 `NIBBLE_TO_FLOAT[nibble] * scale * x` 模式改写为 `fma(nibble, scale*x, bias*x)` / `fma(NIBBLE_TO_FLOAT[nibble], scale*x, 0.0f)`。
> **依据**：flash-moe `shaders.metal:314–330` 声称该优化带来 +2.6% / +12% 收益。

### 47.1 改动范围

- `src/models/moe_kernel.metal`：473 行改动，覆盖 `fused_gate_up_swiglu*`、`dequant_matvec_4bit*`、`dequant_matvec_affine*`、`gather_gate_up_swiglu`、`gather_down` 等 kernel。
- `src/models/moe_kernel_f16.metal`：6 行改动，覆盖 `dequant_matvec_affine_f16*` kernel。
- `scripts/metal_kernel_test.m`：1 处测试输入 signedness 修复（已随 FMA 回退一并还原）。

### 47.2 测试结果

```bash
rm -rf .zig-cache zig-out && zig build -Doptimize=ReleaseFast
bash scripts/run_kernel_tests.sh        # ✅ PASS
bash scripts/run_mla_attention_test.sh  # ✅ PASS (rel_L2=3.659e-04)
bash scripts/dsv4_smoke.sh              # ✅ PASS (Paris ✓, 2+2 ✓)
bash scripts/run_benchmark.sh --native  # ❌ 退化
```

**Benchmark 结果（FMA 改写后）**：

| 指标 | 基线（改写前） | FMA 改写后 | 变化 |
|------|---------------|-----------|------|
| tok/s | 0.621 | **0.555** | **-10.6%** |
| Paris | ✓ | ✗ FAIL | 正确性退化 |
| E2E 7-prompt | 7/7 | **6/7** | 1 个失败 |
| unit tests | PASS | PASS | — |

### 47.3 失败原因分析

1. **浮点运算顺序改变**：
   - 原版：`(scale * nibble + bias) * x`，先算 `scale*nibble+bias`，再乘 `x`。
   - FMA 版：`nibble * (scale*x) + (bias*x)`。中间量 `scale*x` 和 `bias*x` 被预先计算并截断，累加顺序也与原版不同。
   - 在 V4 的 borderline expert selection 中，0.001 级别的 score 差异即可导致 expert swap，43 层放大后输出偏离。

2. **MXFP4 LUT 模式**：
   - 原版：`NIBBLE_TO_FLOAT[nibble] * sf * x`
   - FMA 版：`fma(NIBBLE_TO_FLOAT[nibble], sf*x, 0.0f)`
   - 同样改变了乘法结合顺序，引入舍入差异。

3. **寄存器压力增加**：
   - unrolled 8-nibble 版本中，FMA 版需要额外 8 个 `scale*x` 寄存器 + 8 个 `bias*x` 寄存器。
   - 这可能增加 register spilling，抵消 FMA 的 ALU 收益，导致 tok/s 反而下降。

### 47.4 处置

- **已回退 FMA 改写**：`src/models/moe_kernel.metal`、`src/models/moe_kernel_f16.metal`、`scripts/metal_kernel_test.m` 恢复到改写前状态。
- **回退后验证**：`dsv4_smoke.sh` 再次 PASS。

### 47.5 对 Stage 1 的修正

**FMA 反量化不是 dmlx V4 的免费收益**。在 flash-moe 上有效的原因可能是：
- Qwen3.5 使用 affine 4-bit（`w = nibble*scale+bias`），没有 MXFP4 的 LUT 环节；
- Qwen3.5 routing 对微小数值差异不敏感；
- flash-moe 的 kernel 结构与 dmlx 不同（shared memory tiling、SIMD reduction 已优化，register pressure 可控）。

**修正后的 Stage 1**：
- ❌ 移除 "FMA 反量化代数重组" 作为 Stage 1 任务。
- ✅ 保留 "启用 Q8_0 wo_a"。
- ✅ 保留 "合并 shared expert 与 MoE combine"，但需更谨慎验证。
- ⬜ 新增候选：针对具体瓶颈 kernel 做 profiled 优化，而非 blanket FMA 重写。

### 47.6 教训

**即使是被外部项目验证过的优化，也可能因模型格式（MXFP4 vs affine）、routing 敏感度、kernel register pressure 差异而在 dmlx V4 上失败。Stage 1 的每个改动都必须经过 `run_benchmark.sh --native` 而非仅 `smoke.sh` 验证。**

### 47.7 Stage 1 Task 2 执行记录：Q8_0 wo_a 部分成功

> **实验**：启用 `src/metal_infer/mla_attention.m` 中已创建但未使用的 Q8_0 `wo_a` 量化路径。`wo_a_q8_gpu[g]` 已在 `set_layer_attn` 中分配并量化；原代码在 dispatch 处强制走 f32 dense。

#### 改动

```objc
// src/metal_infer/mla_attention.m (~line 862)
static int use_q8_woa = -1;
if (use_q8_woa < 0) use_q8_woa = getenv("DMLX_USE_Q8_WOA") ? 1 : 0;
if (use_q8_woa && abc && abc->wo_a_q8_gpu[g]) {
    enc_matvec_q8_0(P, cb3, abc->wo_a_q8_gpu[g], bgv, bog_arr[g], O_LORA_RANK, group_feat);
} else {
    // f32 dense fallback (unchanged)
}
```

默认保持 f32 dense；设置 `DMLX_USE_Q8_WOA=1` 启用 Q8_0 路径。

#### 测试结果

| 配置 | tok/s | Paris | E2E 7-prompt | 备注 |
|------|-------|-------|-------------|------|
| 基线（f32 dense） | 0.621 | ✓ | 7/7 | — |
| Q8_0 wo_a ON | **0.657** | ✗ | **6/7** | P3 算术题失败 |
| Q8_0 wo_a OFF（默认） | 0.621 | ✓ | 7/7 | 与基线一致 |

#### 结论

Q8_0 `wo_a` 能带来 **+5.8%** 的性能提升，但会引入数值误差，导致 stricter correctness check（benchmark 内部 Paris 检查）和 P3 算术题失败。该误差可能来自 Q8_0 量化/反量化的舍入，在 V4 的 borderline expert selection 中被放大。

**处置**：代码已合入为 **opt-in**（默认关闭），通过环境变量 `DMLX_USE_Q8_WOA=1` 启用。后续可进一步研究：
- 改进 Q8_0 量化方案（如 per-channel scale、更大 block size）以减少误差；
- 或对 attention 输出加数值 clamp，降低对 expert routing 的敏感度。

在默认路径下，dmlx 仍保持 7/7 正确性；需要性能且能接受 6/7 的场景可手动开启。

---

## 48. 2026-06-16 Stage 1 中期总结与 Task 3 方案

### 48.1 已执行结果

| Task | 状态 | 结果 | 对默认路径性能影响 |
|------|------|------|-------------------|
| Task 1: FMA 反量化 | ❌ 失败并回退 | kernel/attention/smoke 通过，但 benchmark 0.555 tok/s（-10.6%），Paris/E2E 退化 | 0 |
| Task 2: Q8_0 wo_a | ⚠️ 部分成功 | benchmark 0.657 tok/s（+5.8%），但 Paris/E2E 退化；已改为 opt-in | 默认 0，可手动 +5.8% |
| Task 3: shared expert 合并 | ⏸️ 待实施 | 方案已确定，需新增 GPU add kernel 并调整 buffer 分配 | 待测 |

### 48.2 当前代码状态

- 分支：`feat/ds4-ize-stage1`
- 默认路径（`DMLX_USE_Q8_WOA` 未设置）：与基线一致，f32 dense `wo_a`，7/7 E2E 通过。
- Opt-in 路径（`DMLX_USE_Q8_WOA=1`）：启用 Q8_0 `wo_a`，+5.8% tok/s，但 6/7 E2E。
- FMA 改写已完全回退。

### 48.3 Task 3 实施方案

**目标**：把 shared expert 的 CPU read/add/upload 改为 GPU-side in-place add，减少一次 CPU↔GPU 往返和一个 CB wait。

**当前流程**（`src/metal_infer/engine.c:1415–1590`）：
1. MoE → `buf_hidden`（GPU）
2. CPU read `buf_hidden` → `ffn_out`（CPU）
3. Shared expert gate 占用 `buf_hidden`（INTERMEDIATE），up → `buf_h_mid`，down → `buf_attn_out`
4. CPU read shared down → `sv`，CPU add `sv` 到 `ffn_out`
5. CPU upload `ffn_out` → `buf_hidden`
6. mHC post on `buf_hidden`

**问题**：`buf_hidden` 同时是 MoE 输出缓冲和 shared gate 缓冲，无法直接合并。

**建议修改**：
1. **重新分配 shared expert 中间缓冲**：
   - gate → `buf_h_mid`（INTERMEDIATE）
   - up → `buf_attn_out`（INTERMEDIATE，`buf_attn_out` 分配大小为 `buf_size = max(DIM, INTERMEDIATE)`）
   - swiglu in-place on `buf_h_mid`
   - down → `buf_ffn_out_f32`（DIM）
2. **保持 MoE 输出在 `buf_hidden`**，不再 CPU readback。
3. **新增 GPU kernel `vec_add_f32`**：
   - 输入：`buf_hidden`（MoE output）和 `buf_ffn_out_f32`（shared down）
   - 输出：`buf_hidden += shared_down`
   - 也可扩展现有 `moe_combine` kernel 增加 shared-expert 加项。
4. **mHC post 直接读 `buf_hidden`**，无需 CPU upload。

**预期收益**：
- 省掉 CPU read `buf_hidden`（~DIM×4B memcpy）
- 省掉 CPU add（~DIM f32 add）
- 省掉 CPU upload `ffn_out`（~DIM×4B memcpy）
- 可能省掉 shared expert CB 末尾的 `waitUntilCompleted`（如果与 MoE 共用一个 CB 或 deferred）
- 估算：~10–20ms/token 开销减少，tok/s 从 0.62 → 0.65–0.68。

**风险**：
- 需要新增 Metal kernel 并在 `run_kernel_tests.sh` 中对拍。
- shared expert 与 MoE 的 buffer 冲突必须处理正确，否则 segfault。
- 当前 shared expert 使用 `buf_hidden` 作为 gate 输出是历史选择，改动需验证所有路径（native、metal-moe、MLX fallback）。

### 48.4 下一步建议

1. **先完成 Task 3**：这是 Stage 1 剩余收益最确定的一项，且不涉及数值精度（只是 buffer 搬运和加法）。
2. **Task 3 后再跑完整 benchmark**：确认 Stage 1 整体收益。
3. **若 Task 3 成功**，再考虑是否继续 Stage 2（GPU-only mHC pre）；若 Task 3 失败，按 §46 止损条件回退到 MLX 备份路径。

### 48.5 测试纪律（再次强调）

- 每次改动后必须：`run_kernel_tests.sh` → `run_mla_attention_test.sh` → `dsv4_smoke.sh` → `run_benchmark.sh --native`。
- 任何 benchmark 中 Paris 失败或 E2E < 7/7 的改动都必须回退或改为 opt-in。
- 严禁多实例、监控 swap。


---

## §49 性能优化路径更新（2026-06-19）

> 本节与 `docs/en/analysis/native-engine-4toks-plan.md §7` 交叉关联。
> 在完整审读 flash-moe/ds4 参考实现后，对 Stage 1 后续路径做如下修正与补充。

### 49.1 §47 失败的根本原因澄清

§47 的 FMA 改写失败，是**浮点运算顺序改变**导致数值发散，而不是"kernel 优化方向错误"。

两类优化必须严格区分：

| 优化类型 | 是否改变数值 | V4 安全性 | 说明 |
|---------|------------|-----------|------|
| **FMA 代数重组**（`nibble*(scale*x)+bias*x`） | ✅ 改变舍入顺序 | ❌ **危险** | 改变 expert score 0.001 级差距 → routing flip → 43 层发散 |
| **ROWS_PER_TG + x_shared + simd_sum**（并行化） | ❌ **不改变数值** | ✅ **安全** | 只是让更多 thread 并行完成同一个 dot product，数学结果完全相同 |
| **fused gate+up+swiglu（同一 encoder）** | ❌ 不改变 | ✅ **安全** | 只是合并 dispatch，计算本身不变 |
| **批量 6-expert dispatch** | ❌ 不改变 | ✅ **安全** | 改变 GPU scheduler 顺序，不改变每个 expert 的数值 |
| **GPU-side combine（moe_combine kernel）** | ⚠️ 微小 f32 累加顺序差异 | ⚠️ 需验证 | 加法结合律在 f32 下不精确，需 smoke 验证 |

**§47 的教训应该精确表述为**：FMA 代数重组在 V4 上不安全。ROWS_PER_TG 并行化在 V4 上是安全的，因为它不改变任何浮点运算顺序，只改变谁先计算谁后计算（在同一个 dot product 内用 `simd_sum` 做 reduction，与串行累加的数值完全一致）。

### 49.2 为什么当前 kernel 慢 10-15×（性能差距根因）

当前 `fused_gate_up_swiglu` / `dequant_matvec_4bit` 使用的 dispatch pattern：

```objc
// 当前（naive, 1 thread per output row）:
[enc dispatchThreads:MTLSizeMake(out_dim, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
```

每个 thread 独立串行完成 4096 个 MAC，从 global memory 读 x 向量 4096 次（非合并，严重内存瓶颈）。

flash-moe v3 kernel 使用的 dispatch pattern：

```objc
// v3（256 threads per tg, ROWS_PER_TG=8）:
uint32_t num_tgs = (out_dim + 7) / 8;
[enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
```

256 个 thread 分 8 个 SIMD group，每 SIMD group (32 thread) 负责一行，且 256 个 thread 协作加载 x_shared[4096] 一次。收益：
- **x 加载**：从 N_threads × 4096 次 global read → 1 次 256-thread 协作加载到 threadgroup memory
- **dot product**：32 thread 并行 + `simd_sum` → 只需 4096/32=128 次 FMA per thread，而非 4096 次
- **occupancy**：43 层 × 6 expert × (256+512) tg = 大量 threadgroup 同时在 GPU，硬件利用率高

这是**纯并行化改进，不改变任何数学结果**。

### 49.3 下一个 Stage 1 任务：v3-style MXFP4 kernel

> 与 `native-engine-4toks-plan.md §7.1` 一一对应。Task 3（shared expert 合并）与此正交，可并行推进。

**改动范围**：`src/models/moe_kernel.metal`，新增 `mxfp4_matvec_v3` 和 `fused_gate_up_swiglu_mxfp4_v3`。

**关键约束**（来自 flash-moe 代码注释 + §47 教训）：

1. `threadgroup_barrier` 必须在所有 `return` 之前——out-of-bounds thread 提前 return 会导致 x_shared 部分未加载，有效行收到 garbage 输入（历史 SIMD bug 的根因）。

2. scale bias 必须是 **127**（`exp2(byte - 127.0f)`），不是 128（§29 根本 bug，此处只是再次强调）。

3. `simd_sum` 结果只在 `simd_lane == 0` 有意义，其他 lane 不得写 `out[]`。

4. dispatch 时 `out_dim` 可能不被 8 整除，`(out_dim + 7) / 8` 算 tg 数，kernel 内用 `if (row >= out_dim) return`（但必须在 barrier 之后）。

5. 对 `fused_gate_up_swiglu_mxfp4_v3`：SwiGLU clamp 必须保留（`min(gate, 10.0f)`, `clamp(up, -10.0f, 10.0f)`）——§20.1 的 999 bug 说明这不是可选项。

**验收**（与 §47 同标准，必须全部通过）：

```bash
bash scripts/run_kernel_tests.sh             # kernel 单元对拍
bash scripts/run_mla_attention_test.sh       # attention 单测
bash scripts/dsv4_smoke.sh                   # Paris + 2+2
bash scripts/run_benchmark.sh --native       # 预期 ≥ 1.2 tok/s（若低于 0.7 则退化，立即回退）
```

**预期收益**：MoE GPU 从 ~1100ms → ~200ms，总 tok/s 从 0.62 → ~2.0（参考 §7.7 表格）。

### 49.4 Stage 1 后续路径修正（综合 §47 + §7 新信息）

| Task | 安全性 | 预期收益 | 顺序 |
|------|--------|---------|------|
| Task 3: shared expert 合并 | ✅ 安全 | ~+3-5% | 可与下方并行 |
| **新 Task 4: v3-style MXFP4 kernel** | ✅ 安全（仅并行化） | **~+2×** | **最高优先级** |
| Task 5: fused gate+up+swiglu（encoder 合并） | ✅ 安全 | ~+10-15% | Task 4 完成后 |
| Task 6: GPU-side combine（CMD3 内完成） | ⚠️ 需验证 combine 数值 | ~+30% | Task 5 完成后 |
| Task 7: Deferred CMD3 | ⚠️ 依赖 Task 6 | ~+15% | Task 6 完成后 |

> **原 §46.4 对 Stage 1 收益（0.62→0.75）的预估过于保守**。Task 4（v3 kernel）单独即可带来 ~2× 提升，是真正的决定性改动。§45 原方案中 Stage 1 遗漏了 dispatch pattern 优化，导致预期收益被严重低估。

### 49.5 与现有文档的关系

| 文档 | 内容 | 与本节关系 |
|------|------|-----------|
| `docs/en/analysis/native-engine-4toks-plan.md §7` | v3 kernel + fused gate+up + 6-slot + GPU combine + deferred CMD3 的详细实现 | 本节的技术细节全部在那里 |
| `docs/analysis/dsv4-first-class-support-plan.md §47` | FMA 失败记录 | 本节的安全性分析以此为基础 |
| `docs/analysis/dsv4-first-class-support-plan.md §45` | 原 Stage 1–7 路线图 | §45 Stage 1 需要补入 Task 4（v3 kernel） |
| `.kiro/specs/native-engine-perf/design.md` | P0-P3 架构设计（CB 合并、deferred、SMELT） | 对应本节 Task 6-7，CB 合并部分已实验证明无效（§4） |

### 49.6 当前实际状态（2026-06-19）

| 指标 | 值 |
|------|-----|
| 默认路径 tok/s | **0.621** (SMELT N=20, `run_benchmark.sh`) |
| 热状态 tok/s | **0.709** (SMELT N=51, 热缓存) |
| 正确性 | ✅ 7/7 E2E PASS |
| 下一个动作 | 实施 v3-style MXFP4 kernel（Task 4） |
| 目标 | ≥ 3.0 tok/s（修正后现实目标 1.5-2.5 tok/s，见 §46.8） |
