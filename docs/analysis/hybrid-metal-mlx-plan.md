# 混合 Metal/MLX MoE 方案

> **目标**: MoE compute 换裸 Metal，MLX 保留 attention。实现 I/O-GPU 重叠 + score-based DyMoE。
> **预期**: 2.5-3.0 tok/s（flash-moe K=6 等效），7/7 正确性。
> **工作量**: 2-3 周。

---

## 1. 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    per-layer pipeline                    │
├─────────────────────────────────────────────────────────┤
│  MLX: attention (MLA/CSA/HCA)                           │
│    └─ input_norm → qkv_proj → RoPE → attn → o_proj     │
│                                                         │
│  CPU: router (MLX) → scores + indices                   │
│    └─ dataSlice(scores)  ← 读 scores，不破坏 fusion    │
│    └─ DyMoE: skip 低分 expert                           │
│                                                         │
│  CPU: pread experts → Metal buffers                     │
│    └─ thread pool (已有), 2MB aligned (已有)            │
│    └─ pread 直接写入 Metal buffer (无 CPU 中间拷贝)    │
│                                                         │
│  Metal: expert forward (custom kernel)                  │
│    └─ dequant + gate_proj + SwiGLU + down_proj          │
│    └─ combine: weighted sum × scores + residual         │
│    └─ output → MLX array for next layer                 │
│                                                         │
│  Pipeline overlap:                                      │
│    layer N-1 Metal compute ‖ layer N CPU pread          │
│    (Metal CMD async commit, CPU 不等待)                 │
└─────────────────────────────────────────────────────────┘
```

## 2. 模块设计

### 2.1 MetalMoEExecutor（新增文件 `src/models/metal_moe.zig`）

```
MetalMoEExecutor:
  // 生命周期
  init(device, allocator) → self
  deinit()

  // 每层调用（替代 streamingForward 的 MoE 部分）
  forward(layer_idx, hidden_states, scores, indices, expert_ids) → output

内部流程:
  1. pread K experts → Metal buffers (复用 ExpertPreadLoader)
  2. encode Metal compute: dequant + gate/up/SwiGLU/down + combine
  3. commit async (不等待)
  4. 返回 output array (MLX-compatible)
```

### 2.2 ExpertPreadLoader 扩展

```
现有: pread → CPU buffer → mlx_array_new_data (CPU→GPU copy)
新增: pread → Metal buffer (zero-copy, SSD→GPU)

接口:
  // 现有: CPU 路径
  readExperts(layer_idx, expert_ids) → CPU buffers

  // 新增: GPU-direct 路径
  readExpertsToMetal(layer_idx, expert_ids, metal_buffers) → GPU buffers
  
Metal buffer 管理:
  - 预分配 ring buffer (2 × K 个 buffer，支持 double-buffering)
  - 2MB aligned (已有 posix_memalign)
  - Metal MTLBuffer 从 aligned memory 创建 (newBufferWithBytesNoCopy)
```

### 2.3 Metal Kernel（`src/models/metal_moe_kernel.metal`）

```
kernel expert_forward_4bit(
  // 输入: expert data [gate_w, gate_s, up_w, up_s, down_w, down_s]
  constant uint *expert_data  [[buffer(0)]],
  constant float *hidden      [[buffer(1)]],  // [hidden_dim]
  device   float *output      [[buffer(2)]],  // [hidden_dim]
  constant float &weight      [[buffer(3)]],  // router score
  device   float *accumulator [[buffer(4)]],  // shared across experts
  uint     tid [[thread_position_in_grid]]
)

每个 thread 处理 output 的一个元素:
  // 1. 定位 expert data offset
  // 2. dequant gate_proj row[tid] (4-bit mxfp4 → float)
  // 3. dot product gate_proj[tid] · hidden → gate_val
  // 4. dequant up_proj row[tid]
  // 5. dot product up_proj[tid] · hidden → up_val
  // 6. SwiGLU: gate_val * sigmoid(gate_val) * up_val
  // 7. dequant down_proj row[tid]
  // 8. dot product down_proj[tid] · act → expert_out
  // 9. accumulator[tid] += expert_out * weight
```

### 2.4 DyMoE 集成

```
Router (MLX) 输出 scores + indices 后:
  1. dataSlice(scores) → CPU ← 此时 MLX graph 未受影响
  2. 在 CPU 上做 DyMoE skip 决策（score-based，准确）
  3. 只 pread 需要的 experts
  4. Metal kernel 处理剩余 experts

时机关键: router 在 attention 之后、MoE 之前。
  scores 在 MLX graph 中只用于 expandDims→multiply→sum.
  替换为 Metal kernel 后，scores 不再走 MLX graph.
  MLX 只看到 router 输出 (indices)，不再处理 scores.
```

### 2.5 MLX 集成接口

```
// deepseek_v4.zig: DSV4MoE.forward 改为:

fn forward(self, hidden, scores, indices) {
    if (metal_moe_enabled) {
        // Metal path: pread + kernel + combine
        return metal_executor.forward(layer_idx, hidden, scores, indices, expert_ids);
    } else {
        // MLX fallback (现有代码)
        return self.streamingForward(layer_idx, hidden, indices, scores);
    }
}
```

## 3. 实施计划

### Phase 1: Metal Kernel 移植（1-2天）

**源**: `../ds4/ds4_metal.m` — 完整 DeepSeek V4 Metal 实现 (15,813行)
   `../flash-moe/metal_infer/infer.m` — Qwen MoE Metal 实现
   `../flash-moe/metal_infer/shaders.metal` — Metal 参考 shader

**目标**: 从 ds4 移植以下 kernel（同模型、同量化格式）:
1. `g_moe_mul_mv_id_q4_k_pair_swiglu` — gate+up 融合 dequant+matvec+SwiGLU
2. `g_moe_mul_mv_id_q4_k` — down_proj dequant+matvec
3. `g_moe_sum6_pipeline` — K=6 expert 加权求和+combine

**封装**: Zig `@cImport` 包装 ds4 Metal C API，或直接写 Zig Metal binding。
  参考: `../flash-moe/metal_infer/infer.m` 的 `gpu_expert_forward` 函数 (~50行)

### Phase 2: I/O 集成（2-3天）

1. ExpertPreadLoader 扩展 `readExpertsToMetal`: pread → Metal buffer
2. Ring buffer 管理 (2 × K 个 buffer，支持 double-buffering)
3. Async commit: Metal command buffer 提交后不等待

### Phase 3: DyMoE 恢复（1-2天）

1. Router scores dataSlice → CPU (时机正确，不影响 graph)
2. Score-based skip 决策
3. 验证: 7/7 正确性

### Phase 4: 全流程集成（2-3天）

1. DSV4MoE.forward 加 Metal 分支
2. 每层 pipeline: MLX attention → Metal MoE → MLX residual
3. Per-layer async: Metal CMD(N) commit → CPU pread(N+1)
4. Benchmark: 目标 2.5-3.0 tok/s, 7/7

### Phase 5: 稳定性与优化（2-3天）

1. MLX fallback: Metal 失败时自动回退
2. 内存管理: buffer 复用, 避免 leak
3. 多量化格式: mxfp4 优先, int4 支持
4. 文档: API 文档, 使用说明

## 4. 接口契约

### MetalMoEForward 输入/输出

```
输入:
  layer_idx: u32          — 层索引 (0-42)
  hidden: Array           — [hidden_dim] MLX array (attention 输出)
  scores: Array           — [topk] router scores
  indices: Array          — [topk] expert indices
  expert_ids: []u32       — 去重后的 expert ID 列表

输出:
  output: Array           — [hidden_dim] MoE 加权输出

保证:
  - 不修改输入 arrays
  - output 可直接用于 MLX residual add
  - Metal buffer 在返回前已 sync
```

### ExpertPreadToMetal

```
输入:
  layer_idx: u32
  expert_ids: []u32       — K 个 expert ID
  metal_buffers: []MetalBuffer — 预分配的 Metal buffer 数组 (K 个)

输出:
  loaded: usize           — 成功加载的 expert 数量

保证:
  - 每个 expert 写入对应 metal_buffer
  - pread 数据已 sync (F_NOCACHE 可选)
  - 失败时返回 < K，调用方 fallback
```

## 5. 风险与缓解

| 风险 | 缓解 |
|------|------|
| Metal kernel 数值精度差异 | RMS error < 1e-3 阈值，超出回退 MLX |
| pread → Metal buffer 兼容性 | 先用 CPU buffer + memcpy，验证后再优化 |
| Metal/MLX 内存管理冲突 | Metal buffer 独立管理，MLX 只读 output |
| 开发期间正确性回归 | `--metal-moe` CLI flag，默认 OFF |
| 多量化格式维护成本 | 先只支持 mxfp4 (主要格式) |

## 6. 文件清单

| 文件 | 用途 |
|------|------|
| `src/models/metal_moe.zig` | MetalMoEExecutor: Metal 设备管理, kernel encode |
| `src/models/metal_moe_kernel.metal` | Metal shader: dequant+matvec+SwiGLU+combine |
| `src/models/expert_pread.zig` | 扩展: readExpertsToMetal (GPU-direct pread) |
| `src/models/expert_stream.zig` | 简化: 移除并行 pread, 由 MetalMoE 替代 |
| `src/models/deepseek_v4.zig` | 加 Metal 分支: DSV4MoE.forward |
| `src/main.zig` | 加 `--metal-moe` CLI flag |
