# DSpark 投机解码集成技术设计文档

> **日期**: 2026-07-08
> **目标**: 在 dmlx native engine 中集成 DeepSeek-V4-Flash-DSpark 的完整 MTP 投机解码
> **预期收益**: decode 从 1.22 tok/s → 2.5-5.0 tok/s（block_size=5, 接受率 ~3-4 token/step）
> **硬件**: Apple M4 Pro, 48GB

---

## 1. 现状分析

### 1.1 已有实现（简化版 Markov-only DSpark）

当前 `src/dspark.zig` + `src/native_engine.zig` 已实现了简化版投机解码：

- **Propose**: 使用 Markov Head（W1/W2 转移矩阵）在 **target base_logits** 上加偏置
- **Verify**: forwardBatch 验证 draft tokens，greedy 比对 accept/reject
- **Rollback**: rollbackKv 回退未接受位置的 KV cache

**局限性**：所有 draft position 共用同一个 base_logits（target model 最后一步的输出），
而非每个 position 有独立的 backbone hidden states。这导致接受率低（~1-2 token/step），
收益有限。

### 1.2 完整 DSpark（本设计目标）

完整 DSpark 的推理流程（参考 `inference/model.py`）：

```
每个 decode step:
  1. Target forward（43 层）→ output_ids + main_hidden（layer 40/41/42 的 hidden 平均）
  2. DSpark MTP forward（3 层 MoE）→ 5 个 draft token + logits + confidence
  3. Target verify（43 层 × 5 token batch）→ 接受/拒绝
  4. 更新 KV cache（target + MTP 各自独立）
```

关键区别：MTP 的 3 层 backbone 为每个 draft position 生成独立的 hidden state + logits，
然后 Markov Head 在这些 logits 上做顺序修正。这比简化版的"共用 base_logits"质量高得多。

---

## 2. DSpark MTP 模型架构

### 2.1 Config（from `~/models/DeepSeek-V4-Flash-DSpark-meta/inference/config.json`）

| 参数 | 值 | 说明 |
|------|-----|------|
| n_mtp_layers | 3 | MTP backbone 层数 |
| dspark_block_size | 5 | 每步起草 token 数 |
| dspark_target_layer_ids | [40, 41, 42] | 从 target 哪些层提取 hidden |
| dspark_markov_rank | 256 | Markov Head 低秩 |
| dspark_noise_token_id | 128799 | noise embedding token |

### 2.2 MTP 层结构（每层，与 target 层几乎相同）

```
mtp.{0,1,2}:
  ├─ main_proj: [4096, 12288] FP8 — 仅 mtp.0 有（3 × target hidden concat → dim）
  ├─ main_norm: [4096] BF16 — 仅 mtp.0 有
  ├─ attn (DSparkAttention):
  │   ├─ wq_a: [1024, 4096] FP8       ┐
  │   ├─ q_norm: [1024] BF16          │ 与 target MLA 完全相同
  │   ├─ wq_b: [32768, 1024] FP8      │ 但无 compressor/indexer
  │   ├─ wkv: [512, 4096] FP8         │ 仅用 sliding window
  │   ├─ kv_norm: [512] BF16          │
  │   ├─ wo_a: [8192, 4096] FP8       │
  │   ├─ wo_b: [4096, 8192] FP8       │
  │   └─ attn_sink: [64] F32          ┘
  ├─ ffn (MoE, 256 experts):
  │   ├─ gate.weight: [256, 4096] BF16
  │   ├─ gate.bias: [256] F32
  │   └─ experts.{0-255}: w1[2048,2048] w2[4096,1024] w3[2048,2048] — INT8 + E8M0 scale
  ├─ hc_attn_fn/base/scale — Hyper-Connection（与 target 相同）
  ├─ hc_ffn_fn/base/scale
  ├─ attn_norm / ffn_norm: [4096] BF16
  │
  └─ 仅 mtp.2 额外有:
      ├─ norm: [4096] BF16 — 最终 RMSNorm
      ├─ hc_head_fn/base/scale — HC head compression
      ├─ markov_head.markov_w1: [129280, 256] BF16
      ├─ markov_head.markov_w2: [129280, 256] BF16
      └─ confidence_head.proj: [1, 4352] BF16
```

### 2.3 DSparkAttention vs Attention（关键区别）

| 特性 | Target Attention | DSparkAttention |
|------|-----------------|-----------------|
| KV 来源 | 自身 hidden | **main_x（target hidden）+ 自身 draft hidden** |
| Compressor | 有（ratio 4/128） | **无**（dense only） |
| Indexer | 有（ratio==4 层） | **无** |
| Window | sliding_window=128 | sliding_window=128（仅 main_x 的 window 部分） |
| Prefill 行为 | 正常 SDPA | **仅缓存 main_x 的 KV，不做 SDPA** |
| Decode 行为 | window + compress KV | **window(main) + draft_block KV** |

推理流程（decode 阶段）：
```python
# DSparkAttention.forward:
# 1. main_kv = wkv(main_x) → 写入 sliding window KV cache
# 2. draft_q = wq_a(draft_x) → wq_b → RoPE
# 3. draft_kv = wkv(draft_x) → RoPE
# 4. kv = concat([window_cache, draft_kv])  # [win+block_size, head_dim]
# 5. topk_idxs = [0..min(win,pos+1), win+0..win+block_size]
# 6. o = sparse_attn(draft_q, kv, attn_sink, topk_idxs)
# 7. o_proj = wo_b(wo_a(inverse_rope(o)))
```

### 2.4 Expert 量化格式

**关键**：MTP experts 使用 **INT8**（不是 target 的 FP4/MXFP4）：

| 权重 | shape | dtype | scale shape | scale dtype |
|------|-------|-------|-------------|-------------|
| w1 (gate) | [2048, 2048] | I8 | [2048, 128] | F8_E8M0 |
| w2 (down) | [4096, 1024] | I8 | [4096, 64] | F8_E8M0 |
| w3 (up) | [2048, 2048] | I8 | [2048, 128] | F8_E8M0 |

解量化公式：`w_float = int8_val * scale_e8m0_to_float(scale[row/bs][col/bs])`
其中 block_size 从 shape 推断：`bs = weight_cols / scale_cols`（w1: 2048/128=16, w2: 1024/64=16）

---

## 3. 推理流程设计

### 3.1 完整一步推理流程

```
┌─────────────────────────────────────────────────────────┐
│ Step N: 当前有 anchor_token（上一步 target decode 输出）  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Target forward 1 token                              │
│     └─ hidden[0..42] → 保存 hidden[40], hidden[41],    │
│        hidden[42] 的 hc_head_compress 输出              │
│     └─ logits → sample → anchor_token_next              │
│     └─ main_hidden = cat([h40, h41, h42]) ∈ R^12288    │
│                                                         │
│  2. MTP forward (3 层, block_size=5 tokens 并行)        │
│     └─ main_proj(main_hidden) → main_x ∈ R^4096        │
│     └─ embed([anchor, noise, noise, noise, noise])      │
│     └─ mtp.0: DSparkAttention(draft, main_x) + MoE     │
│     └─ mtp.1: DSparkAttention(draft, main_x) + MoE     │
│     └─ mtp.2: DSparkAttention(draft, main_x) + MoE     │
│     └─ hc_head + norm → draft_hidden[0..4]             │
│     └─ lm_head(draft_hidden) → draft_logits[5, vocab]  │
│     └─ Markov Head 顺序修正 → corrected_logits         │
│     └─ argmax → draft_tokens[5]                        │
│                                                         │
│  3. Target verify (forwardBatch, 5 tokens)              │
│     └─ 对 draft_tokens[0..4] 做 target forward batch   │
│     └─ 逐位置比较 target_sample vs draft_tokens         │
│     └─ 接受前 k 个匹配的 + 1 个 bonus token            │
│                                                         │
│  4. 更新状态                                            │
│     └─ KV cache: target 保留到 pos + k + 1             │
│     └─ MTP KV cache: 重置（每步重新算）                 │
│     └─ output: 输出 anchor + accepted[0..k] tokens      │
│     └─ 下一步的 anchor_token = accepted 的最后一个      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 简化方案（Phase 1: 不跑 MTP backbone，只用 Markov + target logits）

当前已有实现，接受率 ~1-2 tokens。保留作为 fallback。

### 3.3 完整方案（Phase 2: MTP backbone + Markov Head）

需要实现完整的 3 层 MTP forward。这是本文档的主要设计目标。

### 3.4 MTP forward 的关键简化

参考 `inference/model.py:DSparkBlock.forward`：

```python
def forward(self, x, start_pos, input_ids, main_x):
    if start_pos > 0:
        return super().forward(x, start_pos, input_ids, main_x)
    # only compute KV cache in prefill stage
    return self.attn(x, start_pos, main_x)
```

**Prefill 阶段**：MTP 层只缓存 main_x 的 KV，不做完整 forward。
**Decode 阶段**：MTP 层做完整 forward（hc_pre + attn + hc_post + hc_pre + MoE + hc_post）。

对于 dmlx native engine 的单序列推理，每次 target decode 之后立即跑 MTP decode，
MTP 不需要维护跨 step 的 KV cache——因为 draft 只看当前 window 的 main_x KV。

---

## 4. 内存预算

### 4.1 权重内存

| 组件 | FP8 大小 | 转 4bit 后 | 说明 |
|------|---------|-----------|------|
| MTP attention (×3 层) | ~100 MB | ~50 MB | wq_a/wq_b/wkv/wo_a/wo_b |
| MTP experts (256×3 层) | ~9.8 GB | **不转，保持 INT8** | I8+E8M0 scale |
| MTP gate/norm/hc (×3) | ~30 MB | 保持 BF16/F32 | 小张量 |
| main_proj | ~50 MB | FP8 | 只有 mtp.0 |
| Markov W1+W2 | ~132 MB | BF16 | 129280×256×2×2B |
| Confidence head | ~9 KB | BF16 | 极小 |
| **MTP 总计** | **~10.1 GB** | — | — |

### 4.2 运行时内存

| 组件 | 大小 | 说明 |
|------|------|------|
| Target model (4bit MLX safetensors) | ~5-8 GB | 已有 |
| SMELT cache N=20 | ~11 GB | 已有 |
| MTP weights (INT8 experts on SSD) | ~0.1 GB (non-expert) | attention/norm 加载到 RAM |
| MTP expert I/O per step | ~240 MB/step | 3×6 experts×13.4MB，从 SSD pread |
| MTP KV cache (3 层) | ~3 KB | window=128, dim=512, 极小 |
| Draft buffers | ~5 MB | logits + hidden states |
| **新增总计** | ~0.4 GB RAM + SSD I/O | **fit easily** |

### 4.3 结论

48GB 机器上完全 fit。MTP 的 attention 权重（~100MB）可以常驻 RAM，
MTP experts（INT8, ~9.8GB total）走 SSD pread（与 target 相同路径），
每步额外 I/O ~240MB（3 层 × 6 experts × ~13MB）。

---

## 5. 实现计划

### 5.1 Phase 1: 权重加载 + INT8 dequant kernel（2-3 天）

**目标**: 从 DSpark safetensors 加载 MTP 权重到 native engine。

**任务**:
1. 扩展 `native_loader` 支持加载 `mtp.*` 权重（当前显式跳过）
2. 实现 INT8 + E8M0 scale 的 dequant 逻辑（新 kernel 或 CPU 解量化）
3. 实现 `main_proj`（FP8 matmul [4096, 12288] × [12288] → [4096]）
4. MTP packed_experts 格式转换（`repack_mtp_experts.py`）

**关键文件**:
- `src/native_loader/loader.zig` — 加入 `loadMTPWeights()`
- `src/native_loader/weights.zig` — 新增 `MTPWeightStore` struct
- `scripts/repack_mtp_experts.py` — INT8 experts 打包成 per-layer binary

**INT8 dequant 公式**:
```
w_float[i,j] = (float)w_int8[i,j] * exp2((float)scale_e8m0[i/bs, j/bs] - 127.0)
```
其中 block_size 从 shape 推断。注意 E8M0 bias 是 **127**（与 MXFP4 相同，§29 教训）。

### 5.2 Phase 2: MTP forward engine（3-5 天）

**目标**: 实现 MTP 的 3 层 forward pass。

**设计**：复用现有 `engine.c` 的 `moe_infer_forward_layer` 框架，但有以下修改：

| 功能 | Target layer | MTP layer | 修改 |
|------|-------------|-----------|------|
| Attention | MLA + compressor + indexer | DSparkAttention（sliding window only）| 新增 `dspark_attention_forward` |
| KV cache | 滑动窗口 + 压缩 KV | 仅滑动窗口（main_x 的 KV）| 简化 KV 管理 |
| Expert routing | score-based + hash (L0-2) | score-based only | 无 hash，但有 gate.bias |
| Expert dtype | MXFP4 (gs=32) | INT8 (bs=16) | 新增 INT8 matmul kernel |
| hc_pre/post | 与 target 相同 | 相同 | 复用 |
| Input | embed(token) | embed(noise) + main_proj(main_hidden) | mtp.0 特殊处理 |

**新增 C 函数**:
```c
// src/metal_infer/dspark_engine.c
void dspark_mtp_forward(
    MoEInferEngine* eng,
    float* main_hidden,       // [12288] from target layers 40/41/42
    int anchor_token_id,      // 当前 anchor token
    float* draft_logits_out,  // [5, vocab_size] output
    float* confidence_out     // [5] confidence scores
);
```

**新增 Metal kernel**:
```metal
// INT8 + E8M0 scale matvec (for MTP experts)
kernel void dequant_matvec_int8_e8m0(
    device const int8_t* W,
    device const uint8_t* scales,
    device const float* x,
    device float* out,
    constant uint& out_dim,
    constant uint& in_dim,
    constant uint& block_size, ...
);
```

### 5.3 Phase 3: Markov Head 顺序采样 + 验证循环（1-2 天）

**目标**: 在 MTP forward 输出的 logits 上做 Markov Head 修正 + greedy 采样，
然后用 target model batch verify。

**修改 `native_engine.zig`**:
```zig
// 在 decode loop 中，替换当前简化版 dspark.propose:
// 1. 从 target forward 中提取 main_hidden = cat([L40_out, L41_out, L42_out])
// 2. 调用 dspark_mtp_forward → draft_logits[5, vocab]
// 3. Markov Head 顺序修正（复用现有 dspark.addMarkovBias）
// 4. argmax → draft_tokens[5]
// 5. target forwardBatch verify（已有逻辑）
// 6. accept/reject + rollback（已有逻辑）
```

### 5.4 Phase 4: MTP SMELT + expert 预加载（2-3 天）

**目标**: MTP 的 256 experts（INT8）也走 SMELT 缓存路径，避免每步 SSD I/O。

**方案**: 复用现有 SMELT 框架，为 MTP 层单独维护一套 expert cache：
- MTP N_smelt = 20（每层 20 个 hot experts）
- 内存增量：20 × 3 × 13MB ≈ 780MB（极小）
- 或共享 target 的 SSD 路径（按需读取），初期不做预加载

### 5.5 Phase 5: 性能优化 + 验证（2-3 天）

**目标**: tok/s 提升验证 + 正确性对齐。

**验证方法**:
1. `bash scripts/dsv4_smoke.sh` — Paris + 2+2 必须仍然通过
2. 新增 `scripts/dspark_bench.sh` — 测量投机解码的接受率和 tok/s
3. 对比 draft quality: DSpark MTP vs 简化版 Markov-only
4. 如果 MTP 接受率 < 2.5 tokens/step，分析根因（INT8 精度？attention 错误？）

---

## 6. 关键技术挑战

### 6.1 DSparkAttention 的 KV 管理

MTP attention 与 target attention 的核心区别：

```
Target attention KV cache:
  - 滑动窗口 [128, 512]
  - 压缩 KV [max_seq/ratio, 512]
  - 跨多步持续增长

MTP DSparkAttention KV cache:
  - 从 target 的 sliding window 复制 main_x 的 KV（最多 128 个）
  - 加上 draft block 自己的 KV（5 个）
  - 每步 target decode 后重置（因为 main_x window 变了）
```

**实现**：MTP 不维护自己的持久 KV cache。每次 MTP forward：
1. 取 target 当前 sliding window position 对应的 main KV（从 target 的 forward 中保存）
2. 为 5 个 draft token 生成 KV
3. SDPA 时 key/value = concat([main_kv_window, draft_kv])

### 6.2 target hidden state 提取

target forward 过程中需要保存 layer 40/41/42 的 hidden state 输出。

当前 `moe_infer_forward_layer` 在每层结束后更新 `eng->residual`，
需要在 layer 40/41/42 完成时额外保存一份 `hidden[layer]` 用于 MTP。

**修改 `engine.c`**:
```c
// 在 moe_infer_forward 的层循环中:
if (layer_id == 40 || layer_id == 41 || layer_id == 42) {
    // 保存 hc_head_compress(residual) 到 eng->dspark_target_hidden[layer_id - 40]
    hc_head_compress(residual, eng->dspark_target_hidden + (layer_id - 40) * DIM);
}
```

注意：参考 `inference/model.py` 中 `main_hiddens.append(h.mean(dim=2))`，
target 保存的是 `hc_mult` 维度的均值（即 4 个 stream 的平均），不是 hc_head_compress。

### 6.3 INT8 Expert 的性能影响

INT8 experts 比 target 的 FP4 (MXFP4) experts 大 2×：
- MXFP4: 每元素 4 bit + scale → ~13.4 MB/expert
- INT8: 每元素 8 bit + scale → ~18-20 MB/expert（估算）

实际从 shard 分析：w1[2048,2048]+w2[4096,1024]+w3[2048,2048] 全是 I8，
= (2048×2048 + 4096×1024 + 2048×2048) × 1 byte = (4M + 4M + 4M) = 12MB/expert
加上 scales ≈ 12.5 MB/expert。

每步 MTP I/O: 3 层 × 6 experts × 12.5MB = **225MB** — 约 0.4s @ 0.54GB/s SSD。

**这是性能瓶颈**：MTP forward 会增加 ~0.4s/step 的 I/O 时间。
但如果接受 3-4 个 token/step，净收益仍然正向：
- 当前：1 token/step @ 0.82s = 1.22 tok/s
- DSpark：(1+3) token/step @ (0.82 + 0.4)s = 3.3 tok/s

### 6.4 MTP Expert 预加载策略

为消除 MTP I/O 瓶颈，可以预加载 MTP 的 hot experts：
- 3 层 × 20 hot experts × 12.5MB = **750MB** — 非常小
- 加到 SMELT cache 中即可
- 预期：MTP I/O → 0，MTP forward ≈ 50-100ms（GPU compute only）
- 最终：(1+3) token/step @ (0.82 + 0.1)s = **4.3 tok/s**

---

## 7. 风险与缓解

| 风险 | 严重度 | 缓解方案 |
|------|--------|---------|
| INT8 dequant 精度不够 → 接受率低 | 中 | 先用 Python 验证 INT8 experts 的输出是否与 HF transformers 一致 |
| MTP I/O 瓶颈 > 收益 | 中 | Phase 4 预加载 MTP hot experts（仅 750MB） |
| DSparkAttention KV 管理复杂 | 高 | 简化：每步重新计算 main KV（不做 incremental） |
| 48GB 内存不够 | 低 | MTP weights 仅增 ~0.5GB RAM，experts 走 SSD |
| main_hidden 提取影响 target 性能 | 低 | 仅 3 个 memcpy（layer 40/41/42），~50KB |
| Markov Head 顺序采样延迟 | 低 | rank=256, vocab=129280，每步 ~33M MAC，CPU ~10ms |

---

## 8. 验收标准

### 8.1 正确性

- `bash scripts/dsv4_smoke.sh` 必须 PASS（Paris + 2+2）
- DSpark 接受率 ≥ 2.5 tokens/step（在 France prompt 上验证）
- 投机解码输出必须与普通 decode 数学等价（greedy verification 保证 lossless）

### 8.2 性能

| 指标 | Phase 1 (Markov-only) | Phase 2 (MTP+Markov) | Phase 4 (MTP+预加载) |
|------|----------------------|---------------------|---------------------|
| tok/s | ~1.5 (估) | ~2.5 (估) | **~4.0** (目标) |
| 接受率 | ~1-2 tok/step | ~3-4 tok/step | ~3-4 tok/step |
| 额外 I/O | 0 | ~225MB/step | ~0 (预加载) |
| 额外内存 | ~132MB (W1/W2) | ~0.6GB (attn weights) | ~1.35GB (+750MB cache) |

### 8.3 测试命令

```bash
# 正确性
bash scripts/dsv4_smoke.sh

# DSpark benchmark
DSPARK_DIR=~/models/DeepSeek-V4-Flash-DSpark bash scripts/dspark_bench.sh

# 接受率统计
DSPARK_STATS=1 ./zig-out/bin/dmlx serve --native \
  --model ~/models/DeepSeek-V4-Flash-4bit \
  --expert-packed-dir ~/models/DeepSeek-V4-Flash-4bit/packed_experts \
  --dspark-dir ~/models/DeepSeek-V4-Flash-DSpark \
  --port 8930 --max-tokens 30 --temperature 0
```

---

## 9. 文件变更清单

| 文件 | 变更类型 | 内容 |
|------|---------|------|
| `src/native_loader/loader.zig` | 修改 | 加入 `loadMTPWeights()` |
| `src/native_loader/weights.zig` | 修改 | 新增 `MTPWeightStore` struct |
| `src/native_loader/config.zig` | 修改 | 解析 `dspark_*` config 字段 |
| `src/metal_infer/dspark_engine.c` | **新增** | MTP 3 层 forward 实现 |
| `src/metal_infer/dspark_engine.h` | **新增** | MTP engine 头文件 |
| `src/metal_infer/engine.c` | 修改 | target forward 中提取 layer 40/41/42 hidden |
| `src/metal_infer/engine.zig` | 修改 | 暴露 `dsparkMTPForward` 绑定 |
| `src/models/moe_kernel.metal` | 修改 | 新增 `dequant_matvec_int8_e8m0` kernel |
| `src/dspark.zig` | 修改 | 加入 `proposeWithMTP()` 方法 |
| `src/native_engine.zig` | 修改 | 集成 MTP forward 到 decode loop |
| `scripts/repack_mtp_experts.py` | **新增** | INT8 experts 打包工具 |
| `scripts/extract_markov_weights.py` | 修改 | 从 DSpark shard 提取 BF16 → f32 |
| `scripts/dspark_bench.sh` | **新增** | DSpark 性能+接受率 benchmark |
| `build.zig` | 修改 | 编译 dspark_engine.c |

---

## 10. 时间线

| 阶段 | 工作量 | 里程碑 |
|------|--------|--------|
| Phase 1: 权重加载 + INT8 kernel | 2-3 天 | MTP experts 能正确 dequant |
| Phase 2: MTP forward engine | 3-5 天 | MTP 3 层能跑通，输出 logits |
| Phase 3: Markov + verify 循环 | 1-2 天 | 完整 speculative decode 跑通 |
| Phase 4: MTP SMELT 预加载 | 2-3 天 | MTP I/O → 0 |
| Phase 5: 性能验证 + 调优 | 2-3 天 | smoke PASS + ≥3.0 tok/s |
| **总计** | **10-16 天** | |

---

## 11. 附录

### 11.1 权重文件位置

```
~/models/DeepSeek-V4-Flash-DSpark/
  model-00046-of-00048.safetensors  (3.6GB, mtp.0)
  model-00047-of-00048.safetensors  (3.6GB, mtp.1)
  model-00048-of-00048.safetensors  (3.7GB, mtp.2 + markov + confidence)

~/models/DeepSeek-V4-Flash-DSpark-meta/
  config.json                       (DSpark config)
  model.safetensors.index.json      (weight → shard mapping)
  inference/                        (官方推理代码参考)
    model.py                        (完整推理实现)
    config.json                     (推理 config)
    generate.py                     (生成入口)
    kernel.py                       (CUDA kernels)
    convert.py                      (HF → 自有格式转换)
```

### 11.2 与现有代码的关系

- `src/dspark.zig`：保留现有 Markov Head 加载/propose 逻辑，扩展为 `proposeWithMTP()`
- `src/native_engine.zig`：已有 propose+verify 框架，只需替换 propose 数据来源
- `src/metal_infer/engine.c`：复用 `moe_forward_layer` 的 MoE dispatch 框架
- `src/models/moe_kernel.metal`：新增 INT8 dequant kernel，其余 kernel 复用

### 11.3 E8M0 Scale 解码（与 §29 保持一致）

```c
// FP8 E8M0 to float: scale = 2^(byte - 127)
// 用 IEEE 754 bit-shift 等效（MLX 方式，无 FPU transcendental）:
float e8m0_to_float(uint8_t s) {
    uint32_t bits = (uint32_t)s << 23;  // 填入 float32 exponent field
    return *(float*)&bits;              // = 2^(s - 127)
}
```
