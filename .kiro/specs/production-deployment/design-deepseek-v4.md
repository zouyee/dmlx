# Design: DeepSeek V4 Model Architecture

**Parent**: `design.md` §Components and Interfaces
**Scope**: Weight loading, attention, cache, and model-level fixes for DeepSeek V4 Flash 4-bit parity with mlx-lm.

## Overview

DeepSeek V4 uses a heterogeneous architecture with dual-path attention, compressed sparse attention (CSA), hyper-connection (mHC), and fused expert dispatch. This design covers the Zig-side implementation needed to match `mlx-lm/mlx_lm/models/deepseek_v4.py` (2153 lines).

## Architecture

### High-Level Flow

```
Input tokens
    → Embedding
    → mHC Pre-Norm
    → [Layer × N]
        → DSV4Attention (dual-path: Q + KV-local + pooled + indexer)
        → mHC Post-Connection
        → mHC Pre-Connection
        → DSV4MoE (fused gather_mm dispatch) OR DSV4Expert (shared)
        → mHC Post-Connection
    → Final Norm
    → LM Head
```

### Key Design Decisions

1. **Fused expert weights**: Keep `switch_mlp.{gate,up,down}_proj.weight` as `[n_experts, out, in]` instead of splitting into per-expert arrays.
2. **Dual-path attention**: Q path uses full head dim with RoPE; KV path uses shared KV with compression; pooled path uses `Compressor`; indexer path uses `Indexer` for sparse selection.
3. **Heterogeneous cache**: `compress_ratio > 0` layers use `DeepseekV4Cache`; `compress_ratio == 0` layers use `RotatingKVCache`.

## Components

### 1. Weight Loading & Sanitization

#### FP4 Expert Dequantization

mlx-lm uses a custom lookup table (16 entries) for FP4 expert weights, not standard `mxfp4`.

```zig
pub fn dequantFp4(weight: Array, scale: Array, block_size: i32) !Array {
    // Lookup table: 0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
    // Unpack uint8 → 2×fp32, multiply by block-wise scale
}
```

**Detection**: `.ffn.experts.` in key AND `weight.dtype in (int8, uint8)` AND `scale.shape[-1] * 16 == weight.shape[-1]`

#### FP8 Attention Dequantization

```zig
pub fn dequantFp8(weight: Array, scale: Array, block_size: i32) !Array {
    // ops.fromFp8() → pad to 128 alignment → reshape → multiply scale → truncate
}
```

**Detection**: `weight.dtype == uint8` AND not matching FP4 pattern.

#### Selective float32 Preservation

These weights must stay float32 regardless of model dtype:
- `attn_sink`
- `e_score_correction_bias`
- All HyperConnection weights (`attn_hc.*`, `ffn_hc.*`)
- `hc_head.*`

### 2. DSV4Attention (Dual-Path)

#### Q Path

```
wq_a(x) → q_norm → wq_b → reshape [B, L, n_heads, head_dim]
→ per-head L2 RMSNorm → transpose [B, n_heads, L, head_dim]
→ _apply_partial_rope
```

#### KV Path

```
wkv(x) → kv_norm → reshape [B, 1, L, head_dim]
→ _apply_partial_rope → cache.update_and_fetch(kv, kv)
// key == value, stored in local RotatingKVCache
```

#### Pooled Path (CSA)

When `compress_ratio > 0`:
```
compressor(x, rope, cache, offset) → pooled [B, N_pooled, head_dim]
```

#### Indexer Path (Sparse Selection)

When `compress_ratio == 4`:
```
indexer(x, q_residual, ...) → topk indices for sparse selection
```

#### Attention Dispatch (4 Cases)

| Case | Condition | Behavior |
|---|---|---|
| 1. select_all | indexer exists AND `max_pooled_length <= index_topk` | Use ALL pooled blocks; add `pooled_bias = log(L)`; dense SDPA |
| 2. generate + topk | `L == 1` + indexer + topk returned | `take_along_axis` gather selected pooled → dense SDPA |
| 3. prefill + topk | `L > 1` + indexer + topk returned | `_sparse_pooled_attention`: separate local + pooled scores + sink |
| 4. no indexer | HCA or no compression | concat pooled to local KV → dense SDPA with mask padding |

#### Output Path

```
_apply_partial_rope(out, inverse=True) → _grouped_output_projection → wo_b
```

#### Grouped Output Projection

```zig
// Reshape: [B, n_heads, L, head_dim] → [B, o_groups, heads_per_group, L, head_dim]
// Transpose to [o_groups, B, L, heads_per_group * head_dim]
// If wo_a quantized: quantizedMatmul with reshaped scales/biases
// If wo_a float: einsum/batched matmul equivalent
// Reshape to [B, L, o_groups * o_lora_rank] → wo_b
```

### 3. DSV4MoE (Fused Expert Dispatch)

```zig
pub const DSV4MoE = struct {
    // Fused weights: [n_experts, out, in] packed uint32
    gate_proj: Array,  // or QuantizedWeight
    up_proj: Array,
    down_proj: Array,
    is_quantized: bool,

    pub fn forward(self: *DSV4MoE, x: Array, ctx: EagerContext) !Array {
        // _gather_sort: flatten indices, argsort, reorder tokens by expert ID
        // gatherMm or gatherQmm (if quantized) with sorted_indices=true
        // Multiply scores before down_proj
        // _scatter_unsort to restore original token order
        // Sort threshold: indices.size >= 8
    }
};
```

### 4. DeepseekV4Cache

```zig
pub const DeepseekV4Cache = struct {
    local: RotatingKVCache,          // sliding window
    compressor_state: BranchState,
    indexer_state: BranchState,

    pub fn update_and_fetch(self: *DeepseekV4Cache, kv: Array, ...) !Array;
    pub fn accumulate_windows(self: *DeepseekV4Cache, kv, gate, state_key, ratio, start_pos) !WindowResult;
    pub fn update_pool(self: *DeepseekV4Cache, new_pooled, state_key) !void;
};
```

### 5. Compressor

```zig
pub const Compressor = struct {
    wkv: Linear,
    wgate: Linear,
    ape: Array,           // parameter, not a Linear
    norm: RMSNorm,
    compress_ratio: usize,
    overlap: bool,        // true when compress_ratio == 4
    out_dim: usize,       // head_dim * (2 if overlap else 1)

    pub fn forward(self: *Compressor, x: Array, rope, cache, offset) !Array {
        // kv = wkv(x), gate = wgate(x)
        // cache.accumulate_windows(kv, gate, ...)
        // reshape to [B, W, ratio, out_dim]
        // add ape to gate
        // if overlap: _overlap_transform
        // softmax(gate) * kv → sum(axis=2)
        // norm → apply RoPE with compressed positions → cache.update_pool
    }
};
```

### 6. Indexer

```zig
pub const Indexer = struct {
    wq_b: Linear,                 // q_lora_rank → n_heads * head_dim
    weights_proj: Linear,         // hidden_size → n_heads
    compressor: Compressor,
    scale: f32,
    n_heads: usize,
    head_dim: usize,
    index_topk: usize,

    pub fn forward(self: *Indexer, x: Array, q_residual: Array, compress_rope, position_rope, cache, start_pos) !TopKResult {
        // pooled = compressor(x, compress_rope, cache, start_pos, "indexer_state")
        // q = wq_b(q_residual) → reshape multi-head → apply position_rope
        // scores = q @ pooled^T
        // max(0, scores) * scale → weights_proj(x) weighting
        // argpartition top-k
    }
};
```

### 7. mHC (HyperConnection) Expand

```zig
// block_out: 3D [B, L, D]
// residual: 4D [B, L, mult, D]
// y = post[..., None] * block_out[:, :, None, :].astype(f32) + comb @ residual.astype(f32)
```

## Data Models

### Per-Layer Configuration

```zig
const LayerConfig = struct {
    compress_ratio: usize,    // 0 = sliding-window-only, ~4 = CSA, ~128 = HCA
    rope_theta: f32,          // 10000.0 for standard, 160000.0 for compress layers
    use_indexer: bool,        // true when compress_ratio == 4
};
```

### Quantization Config

| Component | Mode | group_size | bits | biases |
|---|---|---|---|---|
| Expert MLP | mxfp4 | 32 | 4 | no |
| Attention / Shared Experts | affine | 64 | 4 | yes |
| Embedding / LM Head | affine | 64 | 4 | yes |

## Correctness Properties (V4-Specific)

### Property V1: FP4/FP8 Dequantization Parity

*For any* FP4 or FP8 packed weight tensor, the Zig `dequantFp4` / `dequantFp8` output SHALL be element-wise equal to mlx-lm `sanitize()` output for the same checkpoint.

**Validates: Tasks 36.1, 36.2**

### Property V2: Attention Dispatch Completeness

*For any* input sequence length `L` and indexer state, `DSV4Attention.forward` SHALL dispatch to exactly one of the 4 attention cases without overlap or omission.

**Validates: Tasks 37.1**

### Property V3: Gather-Scatter Round-Trip

*For any* token batch and expert routing indices, `_gather_sort` followed by `_scatter_unsort` SHALL restore the original token order with expert outputs correctly placed.

**Validates: Tasks 36.4**

## Error Handling

| Error | Source | Handling |
|---|---|---|
| `error.UnsupportedQuantFormat` | `dequantFp4` / `dequantFp8` | Log weight name and dtype, fall back to float path if available |
| `error.CompressorShapeMismatch` | `Compressor.forward` | Assert `usable == (total_len / ratio) * ratio`, log buffer state |
| `error.IndexerTopKInvalid` | `Indexer.forward` | If `topk > pooled_count`, fall back to `select_all` case |

## Testing Strategy

- **Golden test**: Compare DeepSeek V4 Flash 4-bit logits against mlx-lm for fixed prompt and seed
- **Unit test**: Each attention dispatch case with mocked cache
- **Property test**: `gather_sort` + `scatter_unsort` round-trip with random tokens and indices (min 100 iterations)
