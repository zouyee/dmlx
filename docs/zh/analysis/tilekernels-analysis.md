# TileKernels 深度分析与 mlx-zig 借鉴方向

> 基于 DeepSeek TileKernels 代码库的逐模块审计，提炼其设计哲学和工程实践，
> 识别 mlx-zig 可以借鉴的关键模式，以充分发挥 MLX 在 Apple Silicon 上的优势。

---

## 第一部分：TileKernels 项目分析

### 1. 项目定位

TileKernels 是 DeepSeek 团队基于 [TileLang](https://github.com/tile-ai/tilelang) 构建的
高性能 GPU kernel 库，面向 NVIDIA SM90/SM100 架构（H100/B200），用于 LLM 训练和推理的
关键路径优化。项目已在 DeepSeek 内部的训练和推理场景中实际使用。

### 2. 模块架构

```
tile_kernels/
├── moe/          # MoE 路由：topk 选择、token 展开/归约、权重归一化
├── quant/        # 量化：FP8(e4m3)/FP4(e2m1)/E5M6 per-token/per-block/per-channel
├── engram/       # Engram 门控：融合 RMSNorm + 门控 + 残差
├── mhc/          # Manifold HyperConnection：Sinkhorn 归一化 + mix 分裂/应用
├── transpose/    # 批量转置
├── modeling/     # torch.autograd.Function 封装（engram gate, mHC pipeline）
├── torch/        # PyTorch 参考实现（用于正确性验证）
└── testing/      # 测试工具（数值比较、benchmark、随机生成器）
```

### 3. 核心设计哲学

#### 3.1 极致的算子融合（Operator Fusion）

TileKernels 最核心的设计理念是**将多个逻辑操作融合到单个 GPU kernel 中**，
最大限度减少 GPU 全局内存读写。

**典型案例 — `swiglu_forward_and_per_token_cast`：**

这个 kernel 在一次 GPU launch 中完成：
1. 从全局内存加载 `x[..., :hidden]` 和 `x[..., hidden:]`
2. 计算 SwiGLU：`silu(x_left) * x_right`
3. 可选：乘以 MoE routing weight
4. 可选：clamp 激活值并统计 clamp 次数
5. 计算 per-token 量化 scaling factor（reduce_absmax）
6. 量化为 FP8 (e4m3)
7. 写回量化结果和 scaling factor

如果不融合，这需要 4-5 次 kernel launch 和 4-5 次全局内存往返。

**另一个案例 — `engram_gate_fwd`：**

单个 kernel 完成：
1. 加载 hidden_states、k、v、weight
2. 计算 RMSNorm(x) 和 RMSNorm(k) 的 rstd
3. 计算 dot(RMSNorm(x) * w, RMSNorm(k))
4. 计算 signed_sqrt + sigmoid 门控分数
5. 计算 output = x + gate * v
6. 保存中间结果供反向传播使用

#### 3.2 硬件感知的 Tile 分块

每个 kernel 都根据硬件特性精心选择 tile 大小：

```python
# 根据 SM 数量决定持久化 block 数
num_persistent_blocks = num_sms * blocks_per_sm // hc_mult

# 根据共享内存大小选择 block 维度
smem_bytes = hidden_size * 2 + blk_d * 4
blocks_per_sm = min(get_max_smem_per_sm() // smem_bytes, 16)

# 根据向量化宽度对齐
num_vectorize = min(get_best_vectorize_size(in_config.dtype), ...)
```

#### 3.3 异步流水线（Async Pipeline）

`engram_gate_kernel` 展示了精细的异步数据预取：

```python
# 当前 tile 计算的同时，异步加载下一个 tile
T.async_copy(k[i_s + 1, pid_h, 0:blk_d], kv_smem[0, :])
T.ptx_wait_group(2)  # 等待前 2 个异步拷贝完成
# ... 计算当前 tile ...
T.ptx_wait_group(0)  # 等待所有异步拷贝完成
```

这种 double-buffering + async copy 模式让计算和内存访问完全重叠。

#### 3.4 参考实现 + 数值验证的工程纪律

每个优化 kernel 都有对应的 PyTorch 参考实现（`torch/` 目录），
测试框架提供了严格的数值验证工具：

```python
# 不仅检查数值接近，还检查量化偏差的统计分布
def check_bias(x, ref_x):
    # 验证量化误差是否无偏（less_ratio ≈ 0.5）
    # 使用中心极限定理计算置信区间
    allowed_diff_ratio = 10 / math.sqrt(x.numel())
    assert abs(less_ratio - 0.5) < allowed_diff_ratio
```

#### 3.5 分层抽象

```
低层 kernel（tilelang DSL）
    ↓ 封装
Python 函数（per_token_cast, topk_gate, ...）
    ↓ 组合
modeling 层（torch.autograd.Function，支持前向+反向）
    ↓ 集成
模型层（直接替换 PyTorch 标准层）
```

---

## 第二部分：mlx-zig 可借鉴的关键模式

### 借鉴 1：算子融合 — 解决 mlx-zig 最大的性能问题

**TileKernels 的做法：** 将 SwiGLU + 量化、RMSNorm + 门控 + 残差等融合为单个 kernel。

**mlx-zig 当前问题：** `nn.zig` 中的层实现完全绕过 MLX 计算图，用 CPU 标量循环。

**借鉴方案：**

mlx-c 已经提供了融合 kernel 的入口：
- `mlx_fast_layer_norm` — 融合 LayerNorm
- `mlx_fast_rms_norm` — 融合 RMSNorm
- `mlx_fast_rope` — 融合 RoPE
- `mlx_fast_scaled_dot_product_attention` — 融合 SDPA

mlx-zig 的 `fast.zig` 已经绑定了这些函数，但 `nn.zig` 完全没有调用。

**具体行动：**

```zig
// 当前 nn.zig 的 RMSNorm.forward — 纯 CPU 标量循环
pub fn forward(self: *RMSNorm, input: Array) !Array {
    // ... 手动计算 mean_sq, rms, 逐元素乘 ...
}

// 应改为：
pub fn forward(self: *RMSNorm, input: Array) !Array {
    return fast.rmsNorm(self.ctx, input, self.weight, self.eps);
}
```

同理，LLaMA Attention 应该调用 `fast.scaledDotProductAttention`，
RoPE 应该调用 `fast.rope`。

**更进一步：** 参照 TileKernels 的 SwiGLU+量化融合模式，mlx-zig 可以通过
`mlx_compile` 将多个 mlx-c 算子编译为融合图：

```zig
// 将 SwiGLU MLP 的 gate_proj + up_proj + silu + multiply 编译为融合操作
const swiglu_closure = try Closure.init(swiGluForward, allocator);
const compiled_swiglu = try compile.compile(swiglu_closure, false);
```

### 借鉴 2：量化基础设施 — mlx-zig 的 QLoRA 路线图

**TileKernels 的做法：**

完整的量化栈：
- 格式：FP8 (e4m3)、FP4 (e2m1)、E5M6
- 粒度：per-token、per-block、per-channel
- 融合：量化与计算融合（SwiGLU+cast、cast+transpose）
- Scaling factor 管理：支持 TMA 对齐、列主序、packed UE8M0

关键数据结构：
```python
@dataclass(frozen=True)
class CastOutputConfig:
    torch_dtype: torch.dtype
    sf_block: tuple[int, int]        # scaling factor 的分块粒度
    round_sf: bool                    # 是否将 SF 取整为 2 的幂
    use_packed_ue8m0: bool           # 是否使用 packed 格式
```

**mlx-zig 当前状态：** 量化基础设施已完整实现（affine/MXFP4/FP8），QLoRA 已实现。

**借鉴方案：**

1. 设计 `QuantConfig` 结构体，参照 TileKernels 的 `CastOutputConfig`：

```zig
pub const QuantConfig = struct {
    format: QuantFormat,          // .fp8_e4m3, .fp4_e2m1, .int4_nf4
    sf_block: [2]i32,             // scaling factor 分块
    round_sf: bool,               // SF 取整为 2 的幂
};
```

2. mlx-c 已提供 `mlx_quantize` 和 `mlx_dequantize`，可以直接绑定
3. 参照 TileKernels 的 per-token/per-block 分层，实现不同粒度的量化策略

### 借鉴 3：MoE 路由 — DeepSeek V4（已实现）

**TileKernels 的做法：**

完整的 MoE 路由管线：
```
scores → topk_gate → get_fused_mapping → expand_to_fused
    → expert compute → reduce_fused → output
```

每个环节都有优化 kernel：
- `topk_gate`：GPU 上的 top-k 选择（重复找最大值，避免全排序）
- `expand_to_fused`：将 token 按 expert 分组展开
- `reduce_fused`：将 expert 输出按 routing weight 加权归约
- `normalize_weight`：routing weight 归一化
- `aux_fi`：负载均衡辅助损失

**mlx-zig 当前状态：** `models/deepseek_v4.zig` + `moe_router.zig` 完整实现 MoE 路由管线。

**借鉴方案：**

参照 TileKernels 的 `torch/moe.py` 参考实现，用 mlx-c 算子组合实现：

```zig
// topk_gate: 使用 mlx_topk
pub fn topkGate(ctx: EagerContext, scores: Array, num_topk: i32) !struct { indices: Array, values: Array } {
    // mlx-c 提供 mlx_topk
}

// reduce_fused: 使用 mlx_take + mlx_multiply + mlx_sum
pub fn reduceFused(ctx: EagerContext, expanded: Array, weights: Array, mapping: Array) !Array {
    // gather + weighted sum
}
```

### 借鉴 4：参考实现 + 数值验证的工程纪律

**TileKernels 的做法：**

```
tile_kernels/torch/moe.py      ← 纯 PyTorch 参考实现
tile_kernels/moe/topk_gate.py  ← 优化 kernel
tests/moe/test_topk_gate.py    ← 对比测试
```

每个优化实现都有：
1. 纯 PyTorch 参考实现（`torch/` 目录）
2. 严格的数值对比测试
3. 偏差统计检查（`check_bias`）
4. Benchmark 工具

**mlx-zig 当前问题：** NN 层和 Autograd 零测试覆盖。

**借鉴方案：**

1. 为每个 NN 层创建 Python MLX 参考实现，生成 golden test data
2. Zig 测试加载 golden data，对比 mlx-zig 输出
3. 参照 TileKernels 的 `calc_diff` 实现数值相似度检查：

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

### 借鉴 5：Sinkhorn 归一化 — 高级注意力机制

**TileKernels 的做法：**

`mhc/sinkhorn_kernel.py` 实现了 Sinkhorn 归一化（交替行/列归一化），
用于 Manifold HyperConnection 中的注意力权重矩阵双随机化。
包含完整的前向和反向传播实现。

**mlx-zig 的借鉴价值：**

这是 DeepSeek V4 架构的关键组件。如果 mlx-zig 要完整支持 DeepSeek V4，
需要实现 Sinkhorn 归一化。可以用 mlx-c 算子组合：

```zig
pub fn sinkhorn(ctx: EagerContext, matrix: Array, num_iters: usize, eps: f32) !Array {
    var result = try ops.softmax(ctx, matrix, &[_]i32{-1});
    // 交替行/列归一化
    for (0..num_iters) |_| {
        const col_sum = try reduce.sumAxis(ctx, result, -2, true);
        result = try ops.divide(ctx, result, try ops.add(ctx, col_sum, eps_arr));
        const row_sum = try reduce.sumAxis(ctx, result, -1, true);
        result = try ops.divide(ctx, result, try ops.add(ctx, row_sum, eps_arr));
    }
    return result;
}
```

### 借鉴 6：autograd.Function 的分层封装模式

**TileKernels 的做法：**

```python
# 低层：独立的 fwd/bwd kernel
engram_gate_fwd(...)  → output, dot, gate_score, rstd_x, rstd_k
engram_gate_bwd(...)  → grad_x, grad_k, grad_v, grad_w_partial

# 高层：torch.autograd.Function 封装
class EngramGateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ...):
        # 调用 fwd kernel
        # ctx.save_for_backward(中间结果)
    @staticmethod
    def backward(ctx, grad_output):
        # 调用 bwd kernel
```

**mlx-zig 的借鉴：**

mlx-zig 的 `closure.zig` + `grad.zig` 已经提供了类似机制，
但当前的 NN 层（Linear、LSTM 等）没有利用它。

应该为每个需要梯度的层实现 closure-based forward，
使其能参与 `valueAndGrad` 的自动微分：

```zig
// 将 Linear.forward 包装为 Closure
fn linearForward(inputs: []const Array, allocator: std.mem.Allocator) ![]Array {
    const x = inputs[0];
    const weight = inputs[1];
    const result = try ops.matmul(ctx, x, try ops.transpose(ctx, weight));
    // ... 返回结果
}
```

---

## 第三部分：MLX 特有优势的发挥策略

TileKernels 针对 NVIDIA CUDA 优化，但其设计理念可以映射到 MLX/Metal：

### Metal 上的算子融合

MLX 的 `mlx_compile` 相当于 TileKernels 的手写融合 kernel，
但是自动完成的：

```zig
// 将整个 Transformer block 编译为融合图
const block_closure = try Closure.init(transformerBlockForward, allocator);
const compiled_block = try compile.compile(block_closure, false);
// 后续调用 compiled_block.apply() 会自动融合算子
```

MLX 的编译器会自动：
- 消除中间 Array 的内存分配
- 融合逐元素操作
- 优化 Metal shader 调度

### 统一内存的利用

TileKernels 需要显式管理 CPU↔GPU 数据传输，
而 MLX 的统一内存架构天然避免了这个问题。

mlx-zig 应该**停止**在 CPU 端通过 `dataSliceMut` 操作数据，
让所有计算留在 MLX 的惰性求值图中，由 MLX 决定最优执行位置。

### Metal 的 Tile 分组

MLX 的 Steel GEMM 和 Metal shader 已经针对 Apple Silicon 的
GPU 架构（TBDR、tile memory、SIMD groups）做了优化。
mlx-zig 通过 mlx-c 调用这些优化实现，比手写 Metal shader 更可靠。

---

## 第四部分：具体行动项

| 优先级 | 行动 | 参照 TileKernels | mlx-zig 实现方式 |
|--------|------|-----------------|-----------------|
| P0 | NN 层改用 mlx-c 算子 | 所有 kernel 都走 GPU | 调用 `fast.zig` 绑定 |
| P0 | 启用 `mlx_compile` 融合 | SwiGLU+cast 融合 | `compile.compile(closure)` |
| P1 | 量化基础设施 | `quant/common.py` | 绑定 `mlx_quantize`/`mlx_dequantize` |
| P1 | MoE 路由完整实现 | `moe/*.py` | 用 mlx-c 算子组合 |
| P1 | 数值验证测试框架 | `testing/numeric.py` | 实现 `calcDiff` + golden test |
| P2 | Sinkhorn 归一化 | `mhc/sinkhorn_kernel.py` | mlx-c 算子组合 |
| P2 | Engram 门控 | `engram/engram_gate_kernel.py` | 融合 RMSNorm+gate+residual |
| P3 | Benchmark 框架 | `testing/bench.py` | 计时 + 带宽/FLOPS 报告 |

---

## 附录：TileKernels 关键技术细节

### A. 量化格式支持矩阵

| 格式 | 位宽 | 范围 | 用途 |
|------|------|------|------|
| FP8 e4m3 | 8-bit | ±448 | 训练/推理激活值 |
| FP4 e2m1 | 4-bit | ±6 | 权重量化 |
| E5M6 | 12-bit | 自定义 | 高精度中间值 |

### B. Scaling Factor 策略

- **per-token (row-wise)**：每行一个 SF，适合激活值
- **per-block**：每个 (M, K) 块一个 SF，适合权重
- **per-channel (column-wise)**：每列一个 SF，适合权重转置后

### C. MoE 路由管线

```
logits [T, E]
  → topk_gate → topk_idx [T, K], topk_weights [T, K]
  → get_fused_mapping → token_topk_to_pos [T, K], pos_to_expert [T*K]
  → expand_to_fused → expanded_x [T*K, H]
  → expert_compute (per-expert matmul)
  → reduce_fused → output [T, H]
```

其中 T=tokens, E=experts, K=topk, H=hidden。
