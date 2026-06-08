# DeepSeek-V4-Flash-4bit 一等公民支持方案（Metal-First）

> **日期**: 2026-06-01
> **硬件**: Apple M4 Pro, 48GB
> **战略方向**: **metal-moe 为目标主路径**。性能达标且全链路数值对齐后，**废弃 MLX 推理路径**。
> **MLX 角色（过渡期）**: 数值对拍 oracle / 正确性参考，不再是长期主路径
> **首要目标**: `--metal-moe` 端到端正确 (E2E 7/7) + 达到性能目标 (3.0 tok/s)
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
