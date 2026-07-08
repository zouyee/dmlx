// DSparkAttention — Sliding-window attention for DSpark draft layers.
//
// Key differences from target Attention (mla_attention_decode in mla_attention.m):
//   - No compressor/indexer (all layers are dense, compress_ratio=0)
//   - KV source: concat(main_kv_window[0..win], draft_kv[0..block_size])
//   - Q comes from draft tokens only (not main_x)
//   - During prefill (start_pos==0): only caches main_x KV, no SDPA
//   - During decode (start_pos>0): full Q/K/V with sparse_attn over window+draft
//
// Architecture per position (decode mode):
//   q = per_head_norm(wq_b(q_norm(wq_a(draft_x))))  → [N_HEADS, HEAD_DIM]
//   RoPE(q[..., -ROPE_DIM:], pos=start_pos+seqlen+k)
//   kv = kv_norm(wkv(draft_x))  → [KV_LORA_RANK]
//   RoPE(kv[..., -ROPE_DIM:], pos=start_pos+seqlen+k)
//   Write main_kv[pos%win] = wkv(main_x) (from target's latest token)
//   full_kv = concat(kv_cache_window, draft_kv)  → [win+block, KV_LORA_RANK]
//   topk_idxs = [0..min(win, pos+1), win..win+block_size]  (attend to all)
//   o = sparse_attn(q, full_kv, attn_sink, topk_idxs, softmax_scale)
//   inverse_RoPE(o[..., -ROPE_DIM:])
//   x = wo_b(wo_a_grouped(o))
//
// For Phase 2 initial implementation: CPU-only placeholder (identity passthrough).
// Will be replaced with Metal kernel dispatch in Phase 3.
//
#include "dspark_engine.h"
#include "engine.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

// Placeholder: DSparkAttention forward (CPU, identity for initial bring-up).
// In production this will dispatch Metal kernels for Q/KV/SDPA/O-proj.
void dspark_attention_forward(
    DSparkEngine *eng, int layer_idx,
    const float *normed_input,   // [DSPARK_BLOCK_SIZE, DIM] — attn_norm output per position
    const float *main_x,         // [DIM] — main_x (the projected target hidden)
    float *attn_out,             // [DSPARK_BLOCK_SIZE, DIM] output
    int start_pos
) {
    (void)eng;
    (void)layer_idx;
    (void)main_x;
    (void)start_pos;

    // PLACEHOLDER: pass through input as output (identity attention)
    // This allows the rest of the pipeline to be tested end-to-end.
    // The MoE layers and Markov Head will still produce meaningful draft tokens
    // even without real attention — just with lower quality (worse acceptance rate).
    memcpy(attn_out, normed_input, DSPARK_BLOCK_SIZE * DIM * sizeof(float));

    // TODO Phase 3: Implement real DSparkAttention:
    // 1. For each draft position k:
    //    a. Q chain: wq_a → q_norm → wq_b → reshape[N_HEADS, HEAD_DIM] → per-head norm → RoPE
    //    b. KV: wkv(draft_x[k]) → kv_norm → RoPE → write to draft_kv[k]
    // 2. Also process main_x through wkv → kv_norm → RoPE → write to main_kv[pos%win]
    // 3. Build full_kv = concat(main_kv_window, draft_kv)
    // 4. SDPA with attn_sink
    // 5. inverse RoPE on output
    // 6. wo_a (grouped) → wo_b → attn_out
}
