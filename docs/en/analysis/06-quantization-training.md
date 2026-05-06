# Chapter 6: Quantization, Training, and Ecosystem

## 6.1 Quantization System (`quantize.zig`, 872 lines)

Supports 4 cutting-edge formats + 2 internal formats:

| Format | group_size | Description |
|------|-----------|------|
| affine | 32/64/128 | Standard symmetric/asymmetric quantization |
| mxfp4 | 32 | Microscaling FP4 (AMD standard) |
| nvfp4 | 16 | NVIDIA FP4 (Blackwell architecture) |
| mxfp8 | 32 | Microscaling FP8 |
| fp8_e4m3 | - | Native FP8 E4M3 |
| turboquant | - | Lloyd-Max + QJL adaptive quantization |

Core types:
- `QuantizedWeight`: packed data + scales + biases + config + original_shape
- `quantizedMatmul`: fused dequantization + matrix multiplication
- `gatherQmm`: quantized gather matmul, used for MoE batched/indexed inference

## 6.2 Expert Streaming (`expert_stream.zig`, 649 lines)

**Core capability**: reduces DeepSeek V4 memory from ~138GB to ~10GB (loading only active experts)

| Mode | Strategy | Use Case |
|------|------|---------|
| Preload | Load specified proportion of experts into memory | Sufficient memory |
| Stream | Stream-load from disk on demand | Memory-constrained |

Stream mode features:
- LRU cache manages active experts
- `PartialTensorReader`: partial tensor reading via `pread`, avoids loading entire files
- Layer prefetcher: preloads experts needed for the next layer

## 6.3 Training

### AdamW (`optim.zig`, 217 lines)

```zig
pub fn step(self: *AdamW, grads: []const Array, stream: mlx_stream) !void {
    // Creates ~15 temporary mlx_array per parameter per step
    const sc_lr = c.c.mlx_array_new_float32(self.lr);
    const sc_eps = c.c.mlx_array_new_float32(self.eps);
    // ... approximately 15 scalar Arrays
}
```

**Quantified impact**:
- 7B model has ~200 parameter matrices → ~3000 temporary objects per step
- Per-parameter serial execution cannot utilize GPU batch parallelism

**Known optimization point** (noted in code comments):
```zig
// FUSION INTEGRATION POINT (R8.2):
// Can be replaced with compiledAdamWStep (fused.zig), compiling the entire step into a single kernel
```

### Trainer (`trainer.zig`, 640 lines)

- SFT (Supervised Fine-Tuning) training loop
- **Missing**: gradient clipping (`clip_grad_norm` is TODO)

### LoRA (`lora.zig`, 227 lines)

- `LoRALayer`: A (Gaussian init) + B (Zero init) + scale
- `LoRAModel`: multi-layer adapter management

### QLoRA (`qlora.zig`, 773 lines)

Quantization + LoRA combined training:
- Base weights remain quantized
- Only LoRA A/B matrices are trained
- Supports `quantizedMatmul` + `lora.apply` fused forward

## 6.4 Safetensors Random Access Reader (`io/safetensors_reader.zig`, 1,045 lines)

**Highlight**: zero-copy random access via `pread`, no full file loading required

Core components:
- `TensorInfo`: dtype, shape, data_offsets, shard_path
- `TensorIndex`: hash index across shards
- `addShard`: parses single file header (8-byte LE u64 + JSON)
- `loadTensor`: `pread` reads raw data by offset
- `buildIndexFromDirectory`: builds global index from `model.safetensors.index.json`
