# Native Engine：4 tok/s 性能优化 — 可实施执行计划（v3）

date: 2026-06-11（基于 ds4 src 反编译 + flash-moe 验证）
baseline: commit `e01aed5`，实测 **0.709 tok/s**（SMELT N=51，热状态，M4 Pro 48GB）
target: **≥ 4 tok/s**

---

## 0. 重新确认优化方向（避免错误）

### 0.1 flash-moe v3 为什么在 dmlx 上失败了

flash-moe 的 `dequant_matvec_4bit_v3` （256 threads, `x_shared[4096]`=16KB）已在 dmlx 上尝试并**确认退步**。

根因：M4 Pro GPU threadgroup memory = 32KB。`x_shared[4096]` 占用 16KB（50%），导致每个 SM 可并发的 threadgroup 从 ~8 降到 2-4，occupancy 崩溃。flash-moe 在 M3 Max 上可行（40 核 vs M4 Pro 的核数差异 + 更大的 L2 cache），但 dmlx 的 workload 特征不同（MLA wo_a 的 8 组 matvec 串行，不同于 flash-moe 的单次 dequant_matvec）。

**→ flash-moe v3 的 kernel 模式本身不适用于 dmlx 当前架构。不再尝试。**

### 0.2 ds4 为什么能成功

ds4 三个关键 kernel 的 shared memory 使用：

| kernel | shared memory | 说明 |
|--------|--------------|------|
| `kernel_dsv4_hc_expand4` | **0** | 每 thread 独立处理 1 (d,t)，纯寄存器计算 |
| `kernel_dsv4_hc_split_weighted_sum_norm4` | 16KB + 128B | 但 1024 threads 做浮点运算 + RMSNorm，一次性消除 CB-A 的 wait |
| `kernel_mul_mv_q8_0_f32` | **256B** (32×2×4B) | NR0=2 rows/TG, NSG=4 simdgroups，极简 reduction |

**ds4 成功的核心原因不是"更聪明的 kernel"，而是"消除架构性浪费"**：
1. mHC 完全 GPU-side，**消除 CPU round-trip wait**（dmlx CB-A wait = 32ms/token）
2. 融合多个 operation 为一次 dispatch，消除 encoder overhead + wait
3. Q8_0 格式：wo_a 等效物每行 4KB vs dmlx f32 dense 的 16KB

### 0.3 修正后的优化方向

| 之前（错误） | 现在（正确） | 原因 |
|------------|------------|------|
| flash-moe v3 kernel for wo_a | ds4 mHC GPU kernels | v3 的 x_shared 16KB 导致 dmlx 上 occupancy 崩溃 |
| wo_a 改 flash-moe v3 dequant | wo_a 改 Q8_0 + ds4 kernel_mul_mv_q8_0_f32 | Q8_0 kernel 只有 256B shared memory，无 occupancy 问题 |
| 单独改 wo_a kernel | 先消除 mHC 的 CB-A/CMD2 wait，再改 wo_a 格式 | wait 是比 non-coalesced access 更大的 bottleneck |

---

## 1. 执行步骤总览

```
Phase 1: 移植 kernel_dsv4_hc_expand4 → 替换 mhc_post_ffn cb3
Phase 2: 移植 kernel_dsv4_hc_split_weighted_sum_norm4 → 替换 CB-A
Phase 3: wo_a 改 Q8_0 + 移植 kernel_mul_mv_q8_0_f32
Phase 4 (远期): kernel_dsv4_q8_hc_expand4_q8_0 融合 attention out + HC expand
```

---

## 2. Phase 1: kernel_dsv4_hc_expand4 移植（最小改动，最高验证价值）

### 目标

替换 dmlx 的 mhc_post_ffn cb3（3 encoders: f32→bf16 + mhc_post + bf16→f32），用 ds4 的一次 dispatch 完成。

### 收益分析

| 指标 | 当前 | 改后 | 收益 |
|------|------|------|------|
| encoders | 3 | 1 | 消除 2 个 encoder |
| 数据类型转换 | f32→bf16→f32（2 次 round trip） | 纯 f32 | 消除精度损失 |
| dispatch pattern | 3 × DIM×HC threads | 1 × DIM threads（每 thread 算 4 HC） | 减少 GPU launch overhead |
| cb3 deferred wait | 5ms/token（实测） | 预期 <2ms/token | ~3ms/token 节省 |

### 2.1 新增 kernel：`mhc_post_ffn_expand4`

**文件**：`src/models/moe_kernel.metal`（在 `mhc_post_bfloat` 之后插入）

```metal
// mhc_post_ffn_expand4: ds4 kernel_dsv4_hc_expand4 adapted for dmlx.
//
// Replaces the 3-encoder mhc_post_ffn cb3 with a single dispatch:
//   Before: f32→bf16, mhc_post_bfloat, bf16→f32 (3 encoders)
//   After:  mhc_post_ffn_expand4 (1 encoder, pure f32)
//
// Design (from ds4 kernel_dsv4_hc_expand4, dsv4_hc.metal:579-620):
//   - 1 thread per (dimension d, token t) → for decode: DIM threads
//   - Each thread computes all 4 HC output streams independently
//   - Reuses block_out (ffn output) and residual_hc for all 4 streams
//   - ZERO shared memory — no occupancy issues
//   - Pure f32 compute — eliminates f32↔bf16 conversion overhead
//
// ds4 original uses args struct with byte-offset strides.
// dmlx adaptation: fixed dimensions, explicit buffer layout.
//
// 参数（对应 dmlx 的 cb3 dispatch）：
//   block_out: ffn output [DIM] f32
//   residual:  current residual [4, DIM] f32
//   post:      per-HC gate coefficients [4] f32
//   comb:      HC×HC comb matrix [4, 4] f32 (column-major: comb[src_hc*4+dst_hc])
//   dst:       new residual [4, DIM] f32

kernel void mhc_post_ffn_expand4(
    device const float* block_out  [[buffer(0)]],  // [DIM]
    device const float* residual   [[buffer(1)]],  // [4, DIM]
    device const float* post       [[buffer(2)]],  // [4]
    device const float* comb       [[buffer(3)]],  // [4, 4]
    device float*       dst        [[buffer(4)]],  // [4, DIM]
    constant uint&      dim        [[buffer(5)]],  // DIM = 4096
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dim) return;

    float block_v = block_out[gid];

    // Load 4 residual values for this dimension
    float r0 = residual[0 * dim + gid];
    float r1 = residual[1 * dim + gid];
    float r2 = residual[2 * dim + gid];
    float r3 = residual[3 * dim + gid];

    // Compute 4 HC output streams
    // HC 0
    float acc0 = block_v * post[0];
    acc0 += comb[0*4 + 0] * r0 + comb[0*4 + 1] * r1 + comb[0*4 + 2] * r2 + comb[0*4 + 3] * r3;
    dst[0 * dim + gid] = acc0;

    // HC 1
    float acc1 = block_v * post[1];
    acc1 += comb[1*4 + 0] * r0 + comb[1*4 + 1] * r1 + comb[1*4 + 2] * r2 + comb[1*4 + 3] * r3;
    dst[1 * dim + gid] = acc1;

    // HC 2
    float acc2 = block_v * post[2];
    acc2 += comb[2*4 + 0] * r0 + comb[2*4 + 1] * r1 + comb[2*4 + 2] * r2 + comb[2*4 + 3] * r3;
    dst[2 * dim + gid] = acc2;

    // HC 3
    float acc3 = block_v * post[3];
    acc3 += comb[3*4 + 0] * r0 + comb[3*4 + 1] * r1 + comb[3*4 + 2] * r2 + comb[3*4 + 3] * r3;
    dst[3 * dim + gid] = acc3;
}
```

### 2.2 修改 `mhc_post_ffn` cb3 dispatch

**文件**：`src/metal_infer/engine.c`（lines 1511-1571）

**旧代码**（3 encoders: f32→bf16 + mhc_post_bfloat + bf16→f32）：

```c
// Encoder 1: f32→bf16 on GPU for residual
// Encoder 2: mhc_post(ffn) bfloat
// Encoder 3: bf16→f32 writeback to buf_residual_gpu
// cb3 committed, DEFERRED — no wait
```

**新代码**（1 encoder: mhc_post_ffn_expand4）：

```c
// --- mhc_post(ffn) — single dispatch via ds4 kernel_dsv4_hc_expand4 pattern ---
{
    // Copy post/comb to GPU buffers (same as before)
    memcpy([(id<MTLBuffer>)eng->buf_mhc_post_weights contents], post, MHC_MULT*sizeof(float));
    memcpy([(id<MTLBuffer>)eng->buf_mhc_comb_weights contents], comb, MHC_MULT*MHC_MULT*sizeof(float));

    id<MTLCommandBuffer> cb3 = [(id<MTLCommandQueue>)eng->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb3 computeCommandEncoder];
    [e setComputePipelineState:(id<MTLComputePipelineState>)eng->pipe_mhc_post_ffn_expand4];
    // Buffer 0: ffn_out (block_out) — from buf_mhc_attn_out_bf16 or a new f32 scratch buffer
    [e setBuffer:(id<MTLBuffer>)eng->buf_ffn_out_f32       offset:0 atIndex:0];
    // Buffer 1: residual — from buf_residual_gpu (f32)
    [e setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu       offset:0 atIndex:1];
    // Buffer 2: post [4] f32
    [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_post_weights   offset:0 atIndex:2];
    // Buffer 3: comb [16] f32
    [e setBuffer:(id<MTLBuffer>)eng->buf_mhc_comb_weights   offset:0 atIndex:3];
    // Buffer 4: dst = buf_residual_gpu (in-place: reads old, writes new)
    [e setBuffer:(id<MTLBuffer>)eng->buf_residual_gpu       offset:0 atIndex:4];
    uint dim = DIM;
    [e setBytes:&dim length:4 atIndex:5];
    // Dispatch: DIM threads, each computes 4 HC output streams
    [e dispatchThreads:MTLSizeMake(DIM, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];

    [cb3 commit]; // DEFERRED — no wait (same deferred pattern as before)

    // Store deferred state
    // ... (same deferred state management as current code)
}
```

**⚠️ 注意**：需要新增 `buf_ffn_out_f32` buffer 来存储 f32 格式的 ffn 输出（当前代码先将 ffn_out 转为 bf16 存入 `buf_mhc_attn_out_bf16`，再在 mhc_post_bfloat 中读回）。新 kernel 直接读 f32，所以需要 ffn_out 保持 f32。

**对应改动**：
- 在 `init_metal` 中新增 `eng->buf_ffn_out_f32` 分配（DIM * sizeof(float)）
- 在 `moe_infer_forward_layer` 的 shared expert 完成后，直接将 ffn_out 写入 `buf_ffn_out_f32`（当前是通过 bf16 中间格式写入 `buf_mhc_attn_out_bf16`）
- 删除 cb3 中的 Encoder 1（f32→bf16）和 Encoder 3（bf16→f32），只保留 mhc_post_ffn_expand4

### 2.3 新增 pipeline state

**文件**：`src/metal_infer/engine.c`（`init_metal` 函数）

```c
// In init_metal, after other pipeline creations:
{
    id<MTLFunction> fn = [(id<MTLLibrary>)eng->lib newFunctionWithName:@"mhc_post_ffn_expand4"];
    NSError *err = nil;
    eng->pipe_mhc_post_ffn_expand4 = [(id<MTLDevice>)eng->dev newComputePipelineStateWithFunction:fn error:&err];
    if (err) { /* log and return error */ }
}
```

### 2.4 Benchmark 验证

```bash
bash scripts/run_benchmark.sh
# 要求：Paris 7/7，tok/s ≥ 0.709（不退步即可，这个改动的收益主要来自消除 3 encoder overhead）
```

### 2.5 Commit（如通过）

```bash
git add src/models/moe_kernel.metal src/metal_infer/engine.c src/metal_infer/engine.h
git commit -m "perf: transplant ds4 kernel_dsv4_hc_expand4 as mhc_post_ffn_expand4

Replace 3-encoder mhc_post_ffn cb3 (f32→bf16, mhc_post_bfloat, bf16→f32)
with single f32 dispatch. 1 thread per (dim), computes all 4 HC streams.

Zero shared memory, no occupancy issues. Eliminates f32↔bf16 round-trip
precision loss and reduces encoder overhead.

Benchmark: <report-path> — <tok/s> tok/s, Paris <X>/7
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 3. Phase 2: kernel_dsv4_hc_split_weighted_sum_norm4 移植

### 目标

替换 dmlx 的 CB-A（mhc_pre_gpu + RMSNorm，2 encoders + wait），用 ds4 的一次 dispatch 完成。

### 收益分析

| 指标 | 当前 | 改后 |
|------|------|------|
| CB-A wait | 32ms/token | 不减（wait 是 CB 级别，仍需要） |
| GPU encoder | 2（mhc_pre_bfloat + rms_norm） | 1（hc_split_weighted_sum_norm） |
| Sinkhorn | 单线程 GPU（256 threads 中 tid=0 计算） | 单线程 GPU |
| CPU post/comb readback | 存于 GPU buffer，CMD2 直接读 | 存于 GPU buffer，CMD2 直接读 |

**⚠️ Phase 2 的收益来自减少 encoder overhead，CK-A wait 本身不减**（仍需等待 GPU 完成才能继续 CPU 操作）。预估收益约 1-2ms/token。

### 3.1 新增 kernel：`mhc_pre_split_weighted_sum_norm`

ds4 的 `kernel_dsv4_hc_split_weighted_sum_norm4` 完整移植。关键要素：

```metal
// ds4 kernel_dsv4_hc_split_weighted_sum_norm4 adapted for dmlx.
//
// Replaces CB-A's 2 encoders (mhc_pre_bfloat + rms_norm) with 1 dispatch.
//
// Design (from ds4 dsv4_hc.metal:395-536):
//   - 1 threadgroup = 1 row (1 token, decode)
//   - 1024 threads, shared memory = 4096*4 + 4*4 + 32*4 bytes
//   - Steps:
//     1. tid=0: Compute HC coefficients (Sinkhorn on 4×4 comb via float4)
//     2. All threads: weighted sum (collapse 4 residual streams → 1 row)
//     3. All threads: RMSNorm on collapsed row (float4, 1024-wide)
//
// Inputs:
//   mixes: fn_weight [24, MHC_H] f32 + base [24] f32 + scale [3] f32
//          (flattened, architecture-dependent layout)
//   x: residual [4, DIM] f32
//   norm_weight: RMSNorm weight [DIM] f32
// Outputs:
//   split: HC coefficients [24] f32 (for diagnostics / next step)
//   dst: collapsed row [DIM] f32 (attn_input)
//   norm_dst: RMSNorm'd row [DIM] f32 (attn_input normed)
//
// Threadgroup memory: shared[4096 + 4 + 32] floats
//   = 16384 + 16 + 128 = 16528 bytes < 32KB OK
```

### 3.2 修改 CB-A dispatch

**文件**：`src/metal_infer/engine.c`（lines 1077-1122）

将当前的 2 encoders 改为 1 encoder。

**需要适配的关键差异**：
- ds4 的 `mix` 参数包含 fn_weight、base、scale 的混合布局，需要适配 dmlx 的 `MhcWeights` 结构
- 当前 dmlx 的 `mhc_pre_bfloat` 从 `buf_attn_hc_fn[layer]` + `buf_attn_hc_base[layer]` + `buf_attn_hc_scale[layer]` 读参数

**最简单的适配方式**：不改变 dmlx 的 MhcWeights buffer 布局，在 kernel 内做 ds4 适配。但更干净的方式是直接修改 kernel 以匹配 dmlx 的 buffer 布局（fn[24×16384], base[24], scale[3]），这样不改 engine.c 的 buffer 设置。

### 3.3 验证

```bash
bash scripts/run_benchmark.sh
# 要求：Paris 7/7，tok/s ≥ Phase 1 的值
```

---

## 4. Phase 3: wo_a 改 Q8_0 + kernel_mul_mv_q8_0_f32 移植

### 目标

将 wo_a 从 f32 dense（128MB/layer，每行 16KB）改为 Q8_0（每行 4KB + 16B scale），4× bandwidth 减少。

### 收益分析

| 指标 | 当前（f32 dense） | 改后（Q8_0） |
|------|------------------|-------------|
| wo_a 大小/layer | 128MB | ~34MB（8 groups × 1024 × 4096 × 1B + scales） |
| 理论读取时间 @50% BW | 2.2ms/layer | 0.55ms/layer |
| 每行带宽 | 16KB | 4KB + scale info |
| shared memory | N/A（当前无） | 256B（NR0=2, 32×2×4B） |

### 4.1 Q8_0 格式说明（block_q8_0）

ds4 使用的 Q8_0 格式（来自 GGUF/llama.cpp）：
```c
#define QK8_0 32
typedef struct {
    float   d;           // scale (delta)
    int8_t  qs[QK8_0];  // quantized values
} block_q8_0;
```

对于 in_dim=4096：4096/32 = 128 blocks per row。
每行存储：128 × (4 + 32) = 4608 bytes ≈ 4.5KB（vs f32 的 16KB）。

### 4.2 Loader 改动

**文件**：`src/models/deepseek_v4.zig`

将 `keepF32(a.wo_a)` 改为 load + quantize 到 Q8_0：

```zig
// 当前：wo_a dequantized to f32
ap.wo_a_dense = try self.keepF32(a.wo_a);

// 改为：wo_a loaded as bf16, quantized to Q8_0 per group
// 需要实现 Q8_0 quantization（或从 safetensors 直接加载 Q8_0 格式）
ap.wo_a_q8 = try self.loadQ8_0(a.wo_a);  // [O_GROUPS=8, O_LORA_RANK=1024, group_feat=4096]
```

### 4.3 AttnWeights struct 改动

**文件**：`src/metal_infer/engine.h`

```c
// 旧：
const float *wo_a_dense; // [O_GROUPS * O_LORA_RANK * 4096]

// 新：
typedef struct {
    float    d;       // block scale
    int8_t   qs[32];  // quantized values (block_q8_0 equivalent)
} Q8_0Block;

Q8_0Block *wo_a_q8[8]; // [O_GROUPS=8] → each [1024*(4096/32)] blocks
```

### 4.4 新增 kernel：`matvec_q8_0_f32`

移植 ds4 的 `kernel_mul_mv_q8_0_f32`（dense.metal:108-176）：

```metal
// matvec_q8_0_f32: ds4 kernel_mul_mv_q8_0_f32 adapted for dmlx wo_a.
//
// Design (from ds4 dense.metal:108-176):
//   - NR0=2 rows per threadgroup
//   - NSG=4 simdgroups, 32 threads per simdgroup → 128 threads/TG
//   - shared memory: 32 * 2 * sizeof(float) = 256 bytes (極小)
//   - Coalesced access: ix = tiisg/(32/8), il = tiisg%(32/8)
//     Each thread loads NQ=8 int8 values per block
//   - Reduction: simd_sum per row → threadgroup scatter → final simd_sum
//
// For wo_a [1024, 4096]:
//   - 1024/2 = 512 threadgroups
//   - 128 threads per TG = 65,536 total threads
//   - Per thread: 4096/(32*8) = 16 blocks per thread
```

### 4.5 修改 MLA attention dispatch

**文件**：`src/metal_infer/mla_attention.m`

```objc
// 旧代码（wo_a 循环）：
for (int g = 0; g < O_GROUPS; g++) {
    enc_matvec_f32_bf16in(P, cb3, bwg, bgv, bog_arr[g], O_LORA_RANK, group_feat);
}

// 新代码：
for (int g = 0; g < O_GROUPS; g++) {
    enc_matvec_q8_0(P, cb3, aw->wo_a_q8[g], bgv, bog_arr[g], O_LORA_RANK, group_feat);
}
```

### 4.6 验证

```bash
bash scripts/run_benchmark.sh
# 要求：Paris 7/7（Q8_0 精度应足以保持正确性）
```

---

## 5. Phase 4（远期）: kernel_dsv4_q8_hc_expand4_q8_0 融合

### 目标

融合 attention output projection（wo_a+wo_b 等价物）+ HC expand 为一次 dispatch。

ds4 的 `kernel_dsv4_q8_hc_expand4_q8_0`（dsv4_hc.metal:752-859）:
- Q8_0 matvec（wo_b 等价物）
- 输出直接做 HC expand（mhc_post_ffn）
- 一次 dispatch 替代 dmlx 的：wo_a×8 + wo_b + mhc_post_ffn

**⚠️ 这个 kernel 假设 attention 权重已经是 Q8_0 格式，且直接产生 HC expand 结果（不需要中间 attn_out）。移植需要：**
1. Phase 1-3 全部完成
2. wo_a + wo_b 都改为 Q8_0 格式
3. mhc_post_ffn 在此 kernel 内完成（与 Phase 1 的功能重叠）

**暂不实施。Phase 1-3 完成后再评估。**

---

## 6. 不做的事（已双重验证确认）

| 方案 | 原因 | 验证来源 |
|------|------|---------|
| flash-moe v3 kernel（x_shared[4096]） | occupancy 崩溃，已实测退步 | 分析文档 §2 |
| flash-moe fast kernel（64 threads） | dequant overhead > 节省 | 分析文档 §2 |
| wo_a 4-bit affine + v3 kernel | 同样 x_shared 问题 | 分析文档 §2 |
| batched_wo_a | non-coalesced，更慢 | 分析文档 §2 |
| shared memory xs[4096] in oLD kernel | 16KB→occupancy 崩溃 | 分析文档 §2 |
| CB-A no-wait | 破坏数据依赖 | 分析文档 §5 |
| shared expert 合并进 CMD2 | 增加 GPU 时间 > 节省 | 分析文档 §5 |
| GPU routing 独立使用 | CMD2 wait 只有 1.2ms | 分析文档 §5 |
| mmap expert | pread 让 tok/s 减半 | 分析文档 §5 |

---

## 7. 文件改动清单总览

### Phase 1

| 文件 | 改动 |
|------|------|
| `src/models/moe_kernel.metal` | 新增 `mhc_post_ffn_expand4` kernel（~55 行） |
| `src/metal_infer/engine.c` | 替换 cb3 mhc_post_ffn block（3 encoders → 1），新增 buf_ffn_out_f32 |
| `src/metal_infer/engine.h` | 新增 `pipe_mhc_post_ffn_expand4`、`buf_ffn_out_f32` |

### Phase 2

| 文件 | 改动 |
|------|------|
| `src/models/moe_kernel.metal` | 新增 `mhc_pre_split_weighted_sum_norm` kernel（~150 行，ds4 移植） |
| `src/metal_infer/engine.c` | 替换 CB-A block（2 encoders → 1），新增 pipeline |
| `src/metal_infer/engine.h` | 新增 `pipe_mhc_pre_split_weighted_sum_norm` |

### Phase 3

| 文件 | 改动 |
|------|------|
| `src/models/deepseek_v4.zig` | `keepF32` → Q8_0 load（需实现 Q8_0 quantization） |
| `src/metal_infer/engine.h` | `wo_a_dense` → `wo_a_q8[8]`，新增 Q8_0Block 类型 |
| `src/models/moe_kernel.metal` | 新增 `matvec_q8_0_f32` kernel（~100 行，ds4 移植） |
| `src/metal_infer/mla_attention.m` | wo_a dispatch 改用 `matvec_q8_0` |

---

## 8. 验证流程（每个 Phase 完成后）

```bash
# 1. 编译验证
# (项目的编译命令)

# 2. 正确性验证
bash scripts/run_benchmark.sh
# 要求：Paris 7/7

# 3. 性能验证
# 要求：tok/s ≥ 上一 Phase 的值

# 4. 每次连续测试前
sudo purge
```

---

## 9. 关键风险

| 风险 | 概率 | Phase | 缓解 |
|------|------|-------|------|
| Q8_0 精度不足以保持 Paris 7/7 | 低 | Phase 3 | 先用 MLX 验证 wo_a Q8_0 精度；如失败可换 4-bit affine（仍比 f32 省 75%） |
| `kernel_dsv4_hc_split_weighted_sum_norm4` 的 16KB shared memory 导致 occupancy 问题 | 中 | Phase 2 | 1024 threads + 16KB → M4 Pro 仍有 2-3 TG/SM，可接受。如失败可降级为只融合 weighted sum（不加 norm），norm 单独 dispatch |
| dmlx 的 MhcWeights buffer 布局与 ds4 `mix` 不兼容 | 中 | Phase 2 | 不改 buffer 布局，在 kernel 内适配 dmlx 的参数布局 |
| cb3 deferred 写入 `buf_residual_gpu` 后下一层 CB-A 直接读的冲突 | 低 | Phase 1 | 当前 cb3 deferred 已经使用 `buf_residual_gpu`，Phase 1 不改这个行为 |

---

## 10. 预期最终 tok/s（Phase 1+2+3 全部完成后）

基于分析文档的 profiling 数据，保守估计：

| 阶段 | 当前 | Phase 1 | Phase 2 | Phase 3 | 合计节省 |
|------|------|---------|---------|---------|---------|
| CB-A wait | 32ms | 32ms | 30ms | 30ms | -2ms |
| MLA attention | 430ms | 430ms | 430ms | ~200ms | -230ms |
| CB3 deferred | 5ms | ~2ms | ~2ms | ~2ms | -3ms |
| 其他 | 304ms | 304ms | 304ms | 304ms | 0 |
| **合计/token** | 771ms | 768ms | 766ms | 536ms | **-235ms** |
| **tok/s** | **0.709** | **0.713** | **0.715** | **~1.02** | |

**注意**：Phase 3 的 wo_a Q8_0 收益最大（MLA 从 430ms → ~200ms），但 Phase 1+2 的贡献较小（主要是消除 encoder overhead + mhc 精度改进）。如果 Phase 2 能实现 **mhc_pre + weighted sum + norm 全部 GPU-side 且 CB-A 不再需要 CPU readback**，CB-A wait 可完全消除（-32ms），这将使 tok/s 达到 ~1.22。

但 CB-A wait 消除需要将 CMD2 与 CB-A 合并为一个 CB（如同 ds4 的做法），这超出了 Phase 2 的范围，需要更大的流水线重构。