# Batched MoE for Multi-Token Prefill/Verification

## 目标

让 `forwardBatch(n_tokens)` 真正并行处理 MoE 层，而不是逐 token 串行。
主要收益场景：prefill（4-100+ tokens）、DSpark verification（5 tokens）。

## 当前状态

`moe_infer_forward_batch` 是 layer-major token-sequential：
```c
for (layer) {
    for (token) {
        moe_infer_forward_layer(eng, layer, hidden_t, pos+t);  // 完整单 token decode
    }
}
```

每个 token 独立走 merged_cb + cb2cmd2 + MoE + shared + cb3。

## MLX 的做法

MLX 用 `gather_qmv_fast` / `gather_qmm`：
- `lhs_indices[i]` → 选 input token
- `rhs_indices[i]` → 选 expert weight
- 一次 dispatch 处理所有 (token, expert) 对
- Grid: `(M_tokens, N_output_tiles, B_batch)`

## dmlx 已有的基础

`gather_gate_up_swiglu` 已实现 single-token × K experts batch：
- Grid Y = K experts（并行）
- 共享 input x（单 token）
- 从 SMELT pool 按 expert_id 索引权重

## 实现计划

### Phase 1: Multi-token attention batch（减少 GPU sync）

**目标**：让 merged_cb 一次处理 N tokens 的 attention matmul。

**改动**：
1. `mla_attention_prefill_bfloat` 已存在但逐 token 创建 CB。改为：
   - 所有 wq_a dispatch 放在 1 个 CB（N encoders，N tokens）
   - 所有 wq_b dispatch 放在 1 个 CB
   - SDPA 已有 prefill batch kernel

**不需要新 kernel** — 现有 matvec kernel 用 N 次 encoder dispatch in 1 CB。
减少：N × 5 GPU syncs → 5 GPU syncs per layer。

**预计收益**：prefill 从 N×43×23ms → 43×(5ms×N + overhead)

### Phase 2: Multi-token MoE gather（核心改动）

**目标**：N tokens 的 MoE 用 1 次 gather dispatch。

**改动**：

1. **新 kernel `gather_gate_up_swiglu_mt`**（multi-token）：
```metal
kernel void gather_gate_up_swiglu_mt(
    device const uint32_t* pool,        // SMELT expert pool
    device const float* x_batch,        // [N_tokens, IN_DIM]
    device float* out,                  // [total_activations, INTERMEDIATE]
    constant uint* expert_ids,          // [total_activations] which expert
    constant uint* token_ids,           // [total_activations] which token's x to use
    constant uint& total_activations,   // N_tokens × K
    // grid: (INTERMEDIATE/8, total_activations, 1)
)
```

2. **新 routing batch**：对 N tokens 一次性 routing → 得到 `expert_ids[N×K]` 和 `token_ids[N×K]`

3. **新 combine kernel**：收集所有 expert outputs，按 token 加权求和

**但**——routing 需要先知道每个 token 的 ffn_normed input。这依赖 attention 的输出。所以 MoE batch 不能和 attention batch 融合——它们仍然是顺序的（attention → routing → MoE）。

### Phase 3: 完整 batched forward layer

每层的完整 batch 路径：
```
1. Attention batch: 1 CB with N× matmul encoders (wq_a, norm, wq_b, norm, rope, wkv, norm, rope)
2. SDPA prefill: 1 CB (existing kernel)  
3. wo_a, wo_b: 1 CB with N× matmul encoders
4. mhc_post_attn: N× in 1 CB
5. mhc_pre_ffn + norm + routing: N× in 1 CB → get expert_ids[N×K]
6. MoE gather: 1 CB (new multi-token gather kernel)
7. Shared expert batch: 1 CB with N× matmul encoders
8. mhc_post_ffn: N× in 1 CB
```

Syncs per layer: 8（vs 当前 N×5）。对 N=5：从 25 syncs → 8 syncs = -68%。

## 实施顺序

**Step 1（最快收益）**：把当前 prefill 的 per-token CB 合并为 per-layer single CB（不需要新 kernel）。
- 文件：`mla_attention.m` 的 `mla_attention_prefill_bfloat`
- 改动：去掉 per-token `[cb commit]; [cb waitUntilCompleted]`，改为所有 N tokens 的 encoders 放在 1 个 CB 中。

**Step 2**：mhc pre/post batch dispatch（engine.c forwardBatch 路径）。

**Step 3**：MoE multi-token gather kernel。

## Step 1 的具体改动

文件：`src/metal_infer/mla_attention.m` 函数 `mla_attention_prefill_bfloat`

当前（line 1310-1330）：
```objc
for (int t = 0; t < n_tokens; t++) {
    { id<MTLCommandBuffer> cb = [P->queue commandBuffer];
      enc_dequant_matvec_bf16in_bf16out(P, cb, &aw->wq_a, bx, bq_a);
      [cb commit]; [cb waitUntilCompleted]; }
    { id<MTLCommandBuffer> cb = [P->queue commandBuffer];
      enc_rms_norm_rows_bf16in_bf16out(P, cb, bq_a, bqn_w, bq_res, 1, Q_LORA_RANK, 1);
      [cb commit]; [cb waitUntilCompleted]; }
    // ... 5 more commit+wait per token
}
```

改为：
```objc
id<MTLCommandBuffer> cb = [P->queue commandBuffer];
for (int t = 0; t < n_tokens; t++) {
    // 每个 token 的 matmul 作为独立 encoder 追加到同一 CB
    enc_dequant_matvec_bf16in_bf16out(P, cb, &aw->wq_a, bx_t, bq_a_t);
    enc_rms_norm_rows_bf16in_bf16out(P, cb, bq_a_t, bqn_w, bq_res_t, 1, Q_LORA_RANK, 1);
    enc_dequant_matvec_bf16in_bf16out(P, cb, &aw->wq_b, bq_res_t, bq_t);
    enc_rms_norm_rows_bf16in_bf16out(P, cb, bq_t, NULL, bq_n_t, N_HEADS, HEAD_DIM, 0);
    enc_rope_bf16(P, cb, bq_n_t, bcos_t, bsin_t, N_HEADS, 0);
}
[cb commit]; [cb waitUntilCompleted];
// 然后 KV chain 类似处理
```

这把 Q chain 的 N×5 syncs → 1 sync。
