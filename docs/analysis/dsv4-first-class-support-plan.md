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

### Phase 2 — Metal 注意力子系统（最大难点，引入 `--metal-attn`）

按 §0「可行性」移植 ds4 kernel，逐 kernel 对拍。

- [ ] 权重接入：扩展 `extractWeightsForEngine` 提取 `wq_a/q_norm/wq_b/wkv/wo_a(grouped)/wo_b/attn_sink` 的 f32 指针
- [ ] Q 投影链 `wq_a → RMSNorm(q_norm) → wq_b`，对拍 Q
- [ ] `wkv` 联合投影 + 拆分；KV cache 分配（43 层）
- [ ] RoPE：接 `dsv4_rope.metal` tail-only，对拍 roped Q/K
- [ ] SDPA：移植 `dsv4_misc.metal` MLA 混合注意力，**含 attn_sink**，对拍 attn_out
- [ ] 输出：grouped `wo_a`(8 组) → `wo_b`，对拍 attn 层输出
- [ ] 逐层 max diff 达标后，`--metal-attn` 随 `--metal-moe` 默认开启

**验收**：`--metal-attn`（MoE 仍 MLX）smoke 输出 `Paris`，逐层达标。

### Phase 3 — Metal mHC + norm + 输出层

- [ ] mHC：移植 `dsv4_hc.metal`，`expandToMHC`/`compressFromMHC`/`mhcPreNormFn`，注意 `hc_eps`
- [ ] 最终 `norm` + `lm_head`（或暂留 MLX，影响小）
- [ ] 逐层对拍

**验收**：注意力 + mHC 全 metal，MoE 可切 MLX，smoke `Paris`。

### Phase 4 — Metal MoE 数值对齐

- [ ] expert dequant 是 **mxfp4/gs32**，对拍 `dequant_matvec_4bit` vs MLX `gather_qmm`
- [ ] `moe_combine_residual` 改 K=6（flash-moe 移植为 K=8 硬编码）+ shared expert 接入
- [ ] 路由 gate 对拍（softmax + topK + `e_score_correction_bias`）
- [ ] 逐层对拍

**验收**：`--metal-moe` 全链路 smoke `Paris`，逐层达标。

### Phase 5 — 全 Metal E2E + 性能 + 废弃 MLX

- [ ] 全段 metal，关闭 MLX forward，E2E benchmark 7/7
- [ ] kernel 优化：SIMD reduction / threadgroup tiling / coalesced 读 / FMA dequant
- [ ] I/O：时序预测预取（双缓冲 A/B）、CMD3 延迟提交流水线
- [ ] **达标门**：≥3.0 tok/s + 7/7 + 连续稳定
- [ ] **达标后**：移除 MLX 推理路径（保留 loader + 对拍脚本），metal 成为默认 flag，更新 README

**验收**：默认即 metal，3.0 tok/s，7/7。

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

