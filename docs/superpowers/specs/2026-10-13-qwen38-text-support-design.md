# Qwen3.8-27B Text-Only Support — Design Document

Date: 2026-10-13
Target model: `/Users/zouyee/models/Qwen3.8-27B-4bit` (mlx-community 4-bit affine gs=64, 15 GB, 3 shards)
Architecture: `Qwen3_5ForConditionalGeneration` (model_type `qwen3_5`)

## 1. Scope

**In scope**: language-model text-only inference, end-to-end on both the CLI chat and server paths; ChatML template; 4-bit affine gs=64 quantized compute.

**Out of scope (YAGNI)**:
- Vision tower (27-layer ViT) / image & video input — encountering `image_token_id` (248056) or `video_token_id` (248057) raises an explicit error
- MTP (`mtp_num_hidden_layers=1`) speculative decoding
- Layered/expert streaming weight loading (the 15 GB of weights stay fully resident in memory; a loading-strategy abstraction is kept as an extension point)
- MRoPE multimodal position encoding — the text-only path degrades to 1D positions

## 2. Model Structure Summary (from config.json)

- 64 layers; `layer_types` cycles `"linear_attention" ×3 + "full_attention"` (48 GDN layers + 16 full-attention layers)
- hidden 5120, vocab 248320, `tie_word_embeddings=false`
- GDN (linear attention, Qwen3-Next style gated delta net):
  - 16 key heads / 48 value heads, head dim 128 for both, conv kernel 4, ssm dtype float32
  - Projections (HF naming): `in_proj_qkvz` (fused q/k/v/z output gate) + `in_proj_ba` (beta + a)
  - Per-head `A_log` / `dt_bias`; swish (silu) output gate; `o_norm` RMSNorm + `out_proj`
  - State: conv sliding window + ssm state `[48, 128, 128]` in f32; k/q broadcast to the 48 value heads by kv-group
- Full attention: 24 q heads / 4 kv heads, head_dim 256, `partial_rotary_factor=0.25` (rotary dims 64), `rope_theta=1e7`, `attn_output_gate=true` (sigmoid gate), no attention bias
- Quantization: 4-bit affine group_size=64, `weight/scales/biases` triplets (MLX community layout)

## 3. Overall Architecture

Mirrors the proven GLM-5.3 structure. Three new files; no changes to the existing llama/dsv4/glm53 paths:

```
src/models/qwen38_loader.zig  — Qwen38Config parsing (nested text_config) + weight loading
src/models/qwen38_gdn.zig     — GDN linear-attention module (forked from glm53_kda.zig)
src/models/qwen38.zig         — Qwen38ServingModel assembly + full-attention layer + forward
```

Registry integration: `model_registry.zig` gains `"Qwen3_5ForConditionalGeneration" → qwen38Loader`, returning a `ModelVTable`. The adapter pattern copies `Glm53VTableAdapter` (the model manages its caches internally; the VTable `caches`/`mask` parameters are ignored; `forward` takes token ids and returns logits).

### 3.1 GDN Module (qwen38_gdn.zig)

Forked from the skeleton at `src/models/glm53_kda.zig:146-324`, keeping: the conv1d sliding-window cache, the gated delta rule recurrence (`S ← S·g + k⊗(v−(S·k))·β`), the f32 compute path, per-step eval to prevent lazy-graph accumulation, and the token-by-token prefill loop. Three changes:

1. **Asymmetric heads**: 16 key heads / 48 value heads, ssm state `[48,128,128]`; q/k broadcast by kv-group (each group of 3 value heads shares one key head)
2. **Projection split**: `in_proj_qkvz` splits into q/k/v/z (z is the output gate); `in_proj_ba` splits into beta/a — replacing GLM's `qkv_proj` + `fbg_a_proj` + low-rank `f_b_proj`/`g_b_proj`
3. **Output gate**: `silu(z)` (GLM uses `sigmoid(gate)`); `dt_bias` is per-head `[48]` (GLM is per-channel)

The decay formula is structurally identical: `g = exp(lower_bound · sigmoid(exp(A_log) · (a + dt_bias)))` (per head).

### 3.2 Full-Attention Layer (inside qwen38.zig)

Written fresh — llama.zig is not touched (it lacks both partial rotary and the output gate; modifying it would put Qwen2/3, Mistral, Phi, and every architecture mapped to it at regression risk):

- GQA: 24 q / 4 kv, head_dim 256; partial rotary via the low-level `fast.rope` with `dims=64` (first 64 dims rotated, remaining 192 pass through)
- `attn_output_gate`: after concatenating attention outputs, multiply by `sigmoid(gate_proj(x))` before `o_proj`
- KV cache lives in the model's internal cache (managed together with GDN state), not the engine's `KVCacheStrategy`
- q/k norm: per-head RMSNorm (Qwen3 convention; exact behavior confirmed against actual weight keys)

### 3.3 Loading and Quantization

- Reuses the proven GLM-5.3 path: `WeightSource.quantized(prefix, in_features)` reads `weight/scales/biases`; bits/group_size are derived from the packed shape; compute goes through `quantizedMatmul` without dequantization
- Weight key prefix: confirmed against the actual shard index json (expected `model.language_model.*` or `language_model.*`; the loader asserts and reports a readable error on mismatch)
- Loading-strategy abstraction: the loader accepts a `WeightSource` interface, defaulting to fully resident; a future `PartialTensorReader` streaming backend (FlashMoE-style layered loading) swaps the implementation without touching model code
- Embedding/lm_head are not quantized (confirmed against the actual index); read directly as f32/bf16

### 3.4 Chat Template and Sampling

- Reuses the existing ChatML (`applyChatML` in `chat_template.zig`); the server template selection in `state.zig` gains a ChatML branch for `qwen3_5`
- EOS: token id list `[248046, 248044]` — sampling stop checks are extended to support multiple EOS ids (if currently single-EOS only)
- Default sampling parameters follow generation_config: temperature 1.0 / top_k 20 / top_p 0.95 (defaults only; user-overridable)

## 4. Data Flow

```
token ids → embed → for layer in 0..64:
    linear_attention layer: GDN.forward(x, conv_state, ssm_state)   [f32 state, per-step eval]
    full_attention layer:   attn(q/k norm → partial RoPE(dims=64) → GQA + KV cache
                                 → sigmoid output gate → o_proj)
  → final RMSNorm → lm_head (quantizedMatmul) → logits
```

Quantized layers stay packed end to end; GDN projection outputs are reshaped to heads and cast to f32 before entering the recurrence.

## 5. Error Handling

- Missing config field: parse error names the field path (e.g. `text_config.linear_num_value_heads`)
- Missing weight key / shape mismatch: load-time error printing expected vs actual shape
- Vision token in input: explicit error `vision input not supported (token id N)` — never silently produces garbage
- Unknown `layer_types` entry: load-time hard failure (config/implementation mismatch must not continue)

## 6. Testing and Acceptance

1. **GDN unit tests**: single-step state-update determinism (same input + same state → bit-identical), shape assertions
2. **Oracle alignment harness** (following the GLM Task 7 pattern): golden dumps from transformers/mlx-vlm, aligned segment by segment — embed → layer0 GDN → layer3 full attention → full-segment logits, cos ≥ 0.9999; dump script `scripts/dump_qwen38_oracle.py`
3. **E2E**: 7-prompt generation sanity + server streaming/non-streaming stable serving; regression: existing architectures (llama/glm53/dsv4) stay green

## 7. FlashMoE Borrowing Notes

FlashMoE's expert offloading has no direct target in the dense Qwen3.8; what we borrow is its pattern — "predictive prefetch + bounded cache + on-demand streaming reads" — applied as **loading-layer extensibility**: weight access is funneled through the `WeightSource` abstraction, so a future layered streaming backend for larger models only needs a `PartialTensorReader`-backed implementation (generic row-wise pread, see `expert_stream.zig:490-491`) swapped in at load time, leaving the forward code unchanged. The streaming strategy itself is not implemented in this milestone.
