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
