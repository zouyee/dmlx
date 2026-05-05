# Chapter 3: Models and Inference Engine

## 3.1 DeepSeek V4 (`models/deepseek_v4.zig`, 3,091 lines)

The project's largest and most complex model file, implementing the following cutting-edge mechanisms:

### MLA (Multi-head Latent Attention)
- Compresses KV Cache from `2×n_heads×head_dim` to `2×latent_dim` via low-rank compression
- Significantly reduces memory usage for long sequences

### MoE (Mixture of Experts)
- 256 routed experts + shared experts
- Top-k routing via `moe_router.zig` (629 lines)
- `expert_stream.zig` (649 lines) enables reducing memory from ~138GB to ~10GB

### YARN RoPE
- Frequency interpolation supporting 1M+ context
- Pre-computed rotation frequency tables, applied with GPU acceleration

### mHC (multi-Hyper Connection)
- `HyperHead` implements RMSNorm-weighted learnable mixture heads

### FP8 KV Storage
- Non-RoPE dimensions compressed via `mlx_to_fp8`/`mlx_from_fp8`

### KV Compression Strategies
- `compressKV` supports: mean pooling, softmax-gated pooling, attention sink

## 3.2 DeepSeek V4 Loader (`models/deepseek_v4_loader.zig`, 2,071 lines)

- Parses `model.safetensors.index.json` for sharded weight handling
- HF naming to internal naming mapping: `gate_proj` → `w1`/`w3`/`w2`
- Automatic dequantization: detects `.scales`/`.biases` suffixes
- `SmeltConfig`: expert loading strategy (preload subset vs stream on-demand)
- `sliceFusedExperts`: selects expert subset by mask

## 3.3 Speculative Decoding (`speculative.zig`, 1,223 lines)

### Dual-Track Implementation

**PLD (Prompt Lookup Decoding)**: `NgramDrafter`
- Searches for matching n-gram suffixes over already-generated context
- No draft model required, pure lookup mechanism
- Clean implementation, ~100 lines of core logic

**EAGLE**: `EagleDrafter`
- Uses a lightweight MLP draft head to project hidden states to vocab logits
- Supports KV cache rollback (reverts on verification failure)
- **Known limitation**: the 2nd and subsequent draft tokens are simply repetitions of the first token

### Shared Verification Logic

The `verifyDraft` function implements the speculative sampling accept/reject algorithm, ensuring statistical equivalence.

## 3.4 Guided Decoding (`guided.zig`, 1,129 lines)

FSM-based constrained generation:

- `FiniteStateMachine.fromJsonSchema`: supports string/integer/boolean/enum
- `FiniteStateMachine.fromRegex`: builds FSM from regular expressions
- `GuidedDecoder.maskLogits`: uses MLX `where` operator to set illegal token logits to -inf
- **Dependency**: `zig_regex` package

## 3.5 Generation Engine Three-Tier API (`generation.zig`)

| API | Purpose | Characteristics |
|-----|------|------|
| `generateStep` | Single forward + sampling | Uses `ScopedArrayArena` to track temporary arrays |
| `streamGenerate` | Per-token streaming generation | SSE event output |
| `generate` | Batch generation | Returns complete token sequence |
| `streamGenerateSpeculative` | PLD speculative decoding | KV cache rollback support |
| `streamGenerateEagle` | EAGLE speculative decoding | Requires `forwardWithHidden` |

## 3.6 Model Registry (`model_registry.zig`)

Supports 9 architectures: LLaMA, Mistral, Qwen2, Qwen3, Gemma, GLM-4, Phi, Phi-3, DeepSeek V4

Runtime polymorphic switching via VTable, no recompilation needed.
