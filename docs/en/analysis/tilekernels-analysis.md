# TileKernels Deep Analysis and dmlx Reference Directions

> Based on per-module audit of the DeepSeek TileKernels codebase, extracting its design philosophy and engineering practices,
> identifying key patterns dmlx can adopt to fully leverage MLX's advantages on Apple Silicon.

---

## Part 1: TileKernels Project Analysis

### 1. Project Positioning

TileKernels is a high-performance GPU kernel library built by the DeepSeek team on [TileLang](https://github.com/tile-ai/tilelang),
targeting NVIDIA SM90/SM100 architectures (H100/B200) for critical path optimization in LLM training and inference.
The project is already in production use within DeepSeek's training and inference scenarios.

### 2. Module Architecture

```
tile_kernels/
├── moe/          # MoE routing: topk selection, token expand/reduce, weight normalization
├── quant/        # Quantization: FP8(e4m3)/FP4(e2m1)/E5M6 per-token/per-block/per-channel
├── engram/       # Engram gating: fused RMSNorm + gating + residual
├── mhc/          # Manifold HyperConnection: Sinkhorn normalization + mix split/apply
├── transpose/    # Batch transpose
├── modeling/     # torch.autograd.Function wrappers (engram gate, mHC pipeline)
├── torch/        # PyTorch reference implementations (for correctness verification)
└── testing/      # Test utilities (numerical comparison, benchmark, random generators)
```

### 3. Core Design Philosophy

#### 3.1 Extreme Operator Fusion

TileKernels' core design philosophy is **fusing multiple logical operations into a single GPU kernel**,
minimizing GPU global memory reads and writes.

**Typical case — `swiglu_forward_and_per_token_cast`:**

This kernel completes in a single GPU launch:
1. Load `x[..., :hidden]` and `x[..., hidden:]` from global memory
2. Compute SwiGLU: `silu(x_left) * x_right`
3. Optional: multiply by MoE routing weight
4. Optional: clamp activation values and count clamp occurrences
5. Compute per-token quantization scaling factor (reduce_absmax)
6. Quantize to FP8 (e4m3)
7. Write back quantized result and scaling factor

Without fusion, this would require 4-5 kernel launches and 4-5 global memory round-trips.

**Another case — `engram_gate_fwd`:**

Single kernel completes:
1. Load hidden_states, k, v, weight
2. Compute rstd for RMSNorm(x) and RMSNorm(k)
3. Compute dot(RMSNorm(x) * w, RMSNorm(k))
4. Compute signed_sqrt + sigmoid gating score
5. Compute output = x + gate * v
6. Save intermediate results for backward pass

#### 3.2 Hardware-Aware Tile Partitioning

Each kernel carefully selects tile sizes based on hardware characteristics:

```python
# Determine persistent block count based on SM count
num_persistent_blocks = num_sms * blocks_per_sm // hc_mult

# Select block dimension based on shared memory size
smem_bytes = hidden_size * 2 + blk_d * 4
blocks_per_sm = min(get_max_smem_per_sm() // smem_bytes, 16)

# Align to vectorization width
num_vectorize = min(get_best_vectorize_size(in_config.dtype), ...)
```

#### 3.3 Async Pipeline

`engram_gate_kernel` demonstrates fine-grained async data prefetching:

```python
# Compute current tile while asynchronously loading next tile
T.async_copy(k[i_s + 1, pid_h, 0:blk_d], kv_smem[0, :])
T.ptx_wait_group(2)  # Wait for first 2 async copies to complete
# ... compute current tile ...
T.ptx_wait_group(0)  # Wait for all async copies to complete
```

This double-buffering + async copy pattern fully overlaps computation and memory access.

#### 3.4 Reference Implementation + Numerical Verification Engineering Discipline

Each optimized kernel has a corresponding PyTorch reference implementation (`torch/` directory),
and the test framework provides strict numerical verification tools:

```python
# Not only checks numerical closeness, but also checks the statistical distribution of quantization bias
def check_bias(x, ref_x):
    # Verify quantization error is unbiased (less_ratio ≈ 0.5)
    # Use Central Limit Theorem to compute confidence interval
    allowed_diff_ratio = 10 / math.sqrt(x.numel())
    assert abs(less_ratio - 0.5) < allowed_diff_ratio
```

#### 3.5 Layered Abstraction

```
Low-level kernel (tilelang DSL)
    ↓ Wrapped
Python functions (per_token_cast, topk_gate, ...)
    ↓ Composed
modeling layer (torch.autograd.Function, supports forward+backward)
    ↓ Integrated
model layer (directly replaces PyTorch standard layers)
```

---

## Part 2: Key Patterns dmlx Can Adopt

### Adoption 1: Operator Fusion — Solving dmlx's Biggest Performance Problem

**TileKernels approach:** Fuse SwiGLU + quantization, RMSNorm + gating + residual etc. into single kernels.

**dmlx current problem:** `nn.zig` layer implementations completely bypass MLX computation graph, using CPU scalar loops.

**Adoption plan:**

mlx-c already provides fused kernel entry points:
- `mlx_fast_layer_norm` — Fused LayerNorm
- `mlx_fast_rms_norm` — Fused RMSNorm
- `mlx_fast_rope` — Fused RoPE
- `mlx_fast_scaled_dot_product_attention` — Fused SDPA

dmlx's `fast.zig` has already bound these functions, but `nn.zig` doesn't call them at all.

**Specific actions:**

```zig
// Current nn.zig RMSNorm.forward — pure CPU scalar loop
pub fn forward(self: *RMSNorm, input: Array) !Array {
    // ... manually compute mean_sq, rms, element-wise multiply ...
}

// Should be changed to:
pub fn forward(self: *RMSNorm, input: Array) !Array {
    return fast.rmsNorm(self.ctx, input, self.weight, self.eps);
}
```

Similarly, LLaMA Attention should call `fast.scaledDotProductAttention`,
RoPE should call `fast.rope`.

**Going further:** Referencing TileKernels' SwiGLU+quantization fusion pattern, dmlx can use
`mlx_compile` to compile multiple mlx-c operators into fused graphs:

```zig
// Compile SwiGLU MLP's gate_proj + up_proj + silu + multiply into a fused operation
const swiglu_closure = try Closure.init(swiGluForward, allocator);
const compiled_swiglu = try compile.compile(swiglu_closure, false);
```

### Adoption 2: Quantization Infrastructure — dmlx's QLoRA Roadmap

**TileKernels approach:**

Complete quantization stack:
- Formats: FP8 (e4m3), FP4 (e2m1), E5M6
- Granularity: per-token, per-block, per-channel
- Fusion: Quantization fused with computation (SwiGLU+cast, cast+transpose)
- Scaling factor management: Supports TMA alignment, column-major, packed UE8M0

Key data structure:
```python
@dataclass(frozen=True)
class CastOutputConfig:
    torch_dtype: torch.dtype
    sf_block: tuple[int, int]        # Scaling factor block granularity
    round_sf: bool                    # Whether to round SF to power of 2
    use_packed_ue8m0: bool           # Whether to use packed format
```

**dmlx current state:** Quantization infrastructure fully implemented (affine/MXFP4/FP8), QLoRA implemented.

**Adoption plan:**

1. Design `QuantConfig` struct, referencing TileKernels' `CastOutputConfig`:

```zig
pub const QuantConfig = struct {
    format: QuantFormat,          // .fp8_e4m3, .fp4_e2m1, .int4_nf4
    sf_block: [2]i32,             // Scaling factor block
    round_sf: bool,               // SF round to power of 2
};
```

2. mlx-c already provides `mlx_quantize` and `mlx_dequantize`, can be directly bound
3. Reference TileKernels' per-token/per-block layering to implement different granularity quantization strategies

### Adoption 3: MoE Routing — DeepSeek V4 (Implemented)

**TileKernels approach:**

Complete MoE routing pipeline:
```
scores → topk_gate → get_fused_mapping → expand_to_fused
    → expert compute → reduce_fused → output
```

Each stage has optimized kernels:
- `topk_gate`: GPU top-k selection (repeated max finding, avoids full sort)
- `expand_to_fused`: Group tokens by expert and expand
- `reduce_fused`: Weight expert outputs by routing weight and reduce
- `normalize_weight`: Routing weight normalization
- `aux_fi`: Load balancing auxiliary loss

**dmlx current state:** `models/deepseek_v4.zig` + `moe_router.zig` fully implement MoE routing pipeline.

**Adoption plan:**

Reference TileKernels' `torch/moe.py` reference implementation, implement using mlx-c operator combinations:

```zig
// topk_gate: use mlx_topk
pub fn topkGate(ctx: EagerContext, scores: Array, num_topk: i32) !struct { indices: Array, values: Array } {
    // mlx-c provides mlx_topk
}

// reduce_fused: use mlx_take + mlx_multiply + mlx_sum
pub fn reduceFused(ctx: EagerContext, expanded: Array, weights: Array, mapping: Array) !Array {
    // gather + weighted sum
}
```

### Adoption 4: Reference Implementation + Numerical Verification Engineering Discipline

**TileKernels approach:**

```
tile_kernels/torch/moe.py      ← Pure PyTorch reference implementation
tile_kernels/moe/topk_gate.py  ← Optimized kernel
tests/moe/test_topk_gate.py    ← Comparison test
```

Each optimized implementation has:
1. Pure PyTorch reference implementation (`torch/` directory)
2. Strict numerical comparison tests
3. Bias statistical checks (`check_bias`)
4. Benchmark tools

**dmlx current problem:** Zero test coverage for NN layers and Autograd.

**Adoption plan:**

1. Create Python MLX reference implementations for each NN layer, generate golden test data
2. Zig tests load golden data, compare with dmlx output
3. Reference TileKernels' `calc_diff` to implement numerical similarity checks:

```zig
fn calcDiff(x: []const f32, y: []const f32) f64 {
    var xy_sum: f64 = 0;
    var xx_sum: f64 = 0;
    var yy_sum: f64 = 0;
    for (x, y) |xi, yi| {
        xy_sum += @as(f64, xi) * @as(f64, yi);
        xx_sum += @as(f64, xi) * @as(f64, xi);
        yy_sum += @as(f64, yi) * @as(f64, yi);
    }
    const denom = xx_sum + yy_sum;
    if (denom == 0) return 0;
    return 1.0 - 2.0 * xy_sum / denom;
}
```

### Adoption 5: Sinkhorn Normalization — Advanced Attention Mechanism

**TileKernels approach:**

`mhc/sinkhorn_kernel.py` implements Sinkhorn normalization (alternating row/column normalization),
used for attention weight matrix bistochastic normalization in Manifold HyperConnection.
Includes complete forward and backward propagation implementations.

**dmlx adoption value:**

This is a key component of the DeepSeek V4 architecture. If dmlx is to fully support DeepSeek V4,
Sinkhorn normalization needs to be implemented. Can use mlx-c operator combinations:

```zig
pub fn sinkhorn(ctx: EagerContext, matrix: Array, num_iters: usize, eps: f32) !Array {
    var result = try ops.softmax(ctx, matrix, &[_]i32{-1});
    // Alternate row/column normalization
    for (0..num_iters) |_| {
        const col_sum = try reduce.sumAxis(ctx, result, -2, true);
        result = try ops.divide(ctx, result, try ops.add(ctx, col_sum, eps_arr));
        const row_sum = try reduce.sumAxis(ctx, result, -1, true);
        result = try ops.divide(ctx, result, try ops.add(ctx, row_sum, eps_arr));
    }
    return result;
}
```

### Adoption 6: autograd.Function Layered Encapsulation Pattern

**TileKernels approach:**

```python
# Low-level: independent fwd/bwd kernels
engram_gate_fwd(...)  → output, dot, gate_score, rstd_x, rstd_k
engram_gate_bwd(...)  → grad_x, grad_k, grad_v, grad_w_partial

# High-level: torch.autograd.Function wrapper
class EngramGateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ...):
        # Call fwd kernel
        # ctx.save_for_backward(intermediate results)
    @staticmethod
    def backward(ctx, grad_output):
        # Call bwd kernel
```

**dmlx adoption:**

dmlx's `closure.zig` + `grad.zig` already provide similar mechanisms,
but current NN layers (Linear, LSTM, etc.) don't leverage them.

Should implement closure-based forward for every layer that needs gradients,
enabling participation in `valueAndGrad`'s autograd:

```zig
// Wrap Linear.forward as Closure
fn linearForward(inputs: []const Array, allocator: std.mem.Allocator) ![]Array {
    const x = inputs[0];
    const weight = inputs[1];
    const result = try ops.matmul(ctx, x, try ops.transpose(ctx, weight));
    // ... return result
}
```

---

## Part 3: Strategies for Leveraging MLX's Unique Advantages

TileKernels targets NVIDIA CUDA optimization, but its design philosophy can be mapped to MLX/Metal:

### Operator Fusion on Metal

MLX's `mlx_compile` is equivalent to TileKernels' hand-written fused kernels,
but done automatically:

```zig
// Compile entire Transformer block into fused graph
const block_closure = try Closure.init(transformerBlockForward, allocator);
const compiled_block = try compile.compile(block_closure, false);
// Subsequent calls to compiled_block.apply() will automatically fuse operators
```

MLX's compiler will automatically:
- Eliminate intermediate Array memory allocations
- Fuse element-wise operations
- Optimize Metal shader scheduling

### Leveraging Unified Memory

TileKernels needs explicit CPU↔GPU data transfer management,
while MLX's unified memory architecture naturally avoids this problem.

dmlx should **stop** manipulating data on the CPU side via `dataSliceMut`,
letting all computation stay within MLX's lazy evaluation graph, with MLX deciding the optimal execution location.

### Metal Tile Groups

MLX's Steel GEMM and Metal shader are already optimized for Apple Silicon's
GPU architecture (TBDR, tile memory, SIMD groups).
dmlx calling these optimized implementations via mlx-c is more reliable than hand-writing Metal shaders.

---

## Part 4: Specific Action Items

| Priority | Action | Reference TileKernels | dmlx Implementation |
|--------|------|-----------------|-----------------|
| P0 | NN layers switch to mlx-c operators | All kernels go through GPU | Call `fast.zig` bindings |
| P0 | Enable `mlx_compile` fusion | SwiGLU+cast fusion | `compile.compile(closure)` |
| P1 | Quantization infrastructure | `quant/common.py` | Bind `mlx_quantize`/`mlx_dequantize` |
| P1 | MoE routing complete implementation | `moe/*.py` | Compose with mlx-c operators |
| P1 | Numerical verification test framework | `testing/numeric.py` | Implement `calcDiff` + golden test |
| P2 | Sinkhorn normalization | `mhc/sinkhorn_kernel.py` | mlx-c operator composition |
| P2 | Engram gating | `engram/engram_gate_kernel.py` | Fuse RMSNorm+gate+residual |
| P3 | Benchmark framework | `testing/bench.py` | Timing + bandwidth/FLOPS reporting |

---

## Appendix: TileKernels Key Technical Details

### A. Quantization Format Support Matrix

| Format | Bit Width | Range | Use Case |
|------|------|------|------|
| FP8 e4m3 | 8-bit | ±448 | Training/inference activations |
| FP4 e2m1 | 4-bit | ±6 | Weight quantization |
| E5M6 | 12-bit | Custom | High-precision intermediate values |

### B. Scaling Factor Strategies

- **per-token (row-wise)**: One SF per row, suitable for activations
- **per-block**: One SF per (M, K) block, suitable for weights
- **per-channel (column-wise)**: One SF per column, suitable for weight transposes

### C. MoE Routing Pipeline

```
logits [T, E]
  → topk_gate → topk_idx [T, K], topk_weights [T, K]
  → get_fused_mapping → token_topk_to_pos [T, K], pos_to_expert [T*K]
  → expand_to_fused → expanded_x [T*K, H]
  → expert_compute (per-expert matmul)
  → reduce_fused → output [T, H]
```

Where T=tokens, E=experts, K=topk, H=hidden.
