# Native Engine：4 tok/s 性能优化 — 执行记录 + 修正分析

date: 2026-06-11（持续更新）
baseline: commit `e01aed5`，实测 **0.709 tok/s**（SMELT N=51，热状态，M4 Pro 48GB）
target: **≥ 4 tok/s**
reference: MLX 路径 **~1.6 tok/s**，flash-moe (Qwen3.5) **4.36 tok/s**，ds4 **~2+ tok/s**

---

## 0. 硬件与性能基线（修正）

| 系统 | 模型 | tok/s | 硬件 |
|------|------|-------|------|
| **dmlx MLX** | DSV4 | **~1.6** | M4 Pro 48GB |
| **dmlx native** | DSV4 | **~0.78** | M4 Pro 48GB |
| flash-moe | Qwen3.5 | **4.36** | M3 Max 48GB |
| ds4 | DSV4 | **~2+** | M3/M4 class |

**结论**：M4 Pro 48GB **不是**瓶颈。dmlx native engine 有 2× 的提升空间（追上 MLX），flash-moe 证明 4× 是可达的（换模型+格式）。

---

## 1. 当前瓶颈（NATIVE_PHASE_TIME profiling）

| Phase | 占比 | 估计 ms/token |
|-------|------|-------------|
| MoE GPU | 90.3% | ~1100ms |
| MLA | 5.5% | ~70ms |
| CMD2 | 2.8% | ~36ms |
| Shared | 1.2% | ~15ms |
| I/O | 0.2% | ~3ms |

MoE GPU 占 90%，是唯一的优化目标。SMELT 已消除 I/O，MLA 已通过 Phase 1-4 优化到仅 5.5%。

---

## 2. 为什么 MLX 快 2×

MLX 的 MoE GPU 内核是 Apple 优化的 Metal 实现，可能比我们的 custom kernel 快 2-3×。如果能把 MoE GPU 从 1100ms 降到 400ms，total 就从 1285ms → 585ms → **1.71 tok/s**。

**MLX 优势**：
1. Apple 工程师编写的 Metal kernel（可能用了我们不知道的优化）
2. 自动 kernel fusion（mx.compile 合并操作）
3. 高效的内存池管理

**差距验证**：应该在 MLX 路径上跑 NATIVE_PHASE_TIME 等效测量，确认 MLX 的 MoE 阶段耗时。

---

## 3. 优化路径总览

> **详细实现见 §7**。本节仅概述方向。

### 已确认有效方向

MoE GPU 占 90% 时间（1100ms/token），且：
- CB 合并无效（§4 已验证，多次尝试均为 0 收益）
- 数值精度改动危险（§47 FMA 改写 -10.6% 且 Paris 失败）
- 只有**减少 GPU 计算量和内存访问量**的改动有效

### 主路径：MXFP4 kernel dispatch pattern 优化（§7 Steps 1-5）

核心是将 naive 1-thread/row dispatch 改为 flash-moe v3 的 256-thread tiling + x_shared。这是纯并行化改动，**不改变任何浮点数值**，对 V4 routing 稳定。

预期从 0.71 → 3.8 tok/s，分 5 步完成，见 §7。

### 备选：Hybrid MLX MoE（§7.8）

如果 Steps 1-5 后仍落后 MLX，可考虑调用 MLX C API 执行 expert matmul。风险是引入 eval() 同步屏障，先不做。

---

## 4. 已完成优化

| Commit | 内容 | 类型 | 收益 |
|--------|------|------|------|
| `b4b63f0` | mHC fusion + Q8_0 wo_a | kernel fusion + bandwidth | +6.1% |
| `6b704ef` | coalesced wo_b v2 | bandwidth | +3.5% |
| `ea18a9c` | GPU buffer pass-through | CB merge | 0 |
| `40425ab` | CB-A + CB1 merge | CB merge | 0 |
| `b750770` | GPU blit CB1+CB3 merge | CB merge | 0 |
| `6133380` | MoE v2 no-x_shared | kernel pattern | 0 |

**只有减少 GPU 计算/带宽的优化有效。CB 合并和 kernel pattern 替换均无效。**

## 5. 当前状态与下一步

| 任务 | 状态 | 结果 |
|------|------|------|
| FMA 代数重组 | ❌ 回退 | -10.6%，Paris 失败（见 dsv4 doc §47）|
| Q8_0 wo_a | ⚠️ opt-in | +5.8%，6/7 E2E（`DMLX_USE_Q8_WOA=1`）|
| **v3-style MXFP4 kernel（§7 Step 1）** | ⏸️ **待实施** | 预期 ~2× |

**下一步：实施 §7 Step 1**（`mxfp4_matvec_v3` kernel）。

## 6. 验收标准

```bash
sudo purge
bash scripts/run_benchmark.sh  # SMELT N=51，要求 Paris ✓，tok/s 不退化
```

---

## 7. 完整优化路径（2026-06-19，基于 flash-moe + ds4 代码审计）

> 本节在读完 `flash-moe/metal_infer/shaders.metal`、`infer.m`、`ds4/metal/moe.metal`、`ds4/ds4_metal.m`
> 以及当前 dmlx 所有坑点后撰写。每步给出**精确的改动点、预期收益、验收命令、坑点**。

### 7.0 根本原因诊断（为什么 MoE GPU 占 90%）

当前 MXFP4 kernel 有四个根本性效率问题：

| 问题 | 当前 dmlx | flash-moe v3 | 影响 |
|------|----------|--------------|------|
| dispatch pattern | 1 thread/row（全顺序）| 256 threads, ROWS_PER_TG=8, simd_sum | ~8-16× |
| x 读取 | 每 thread 独立读 global mem | 256 threads 协作加载到 x_shared[4096] | ~4× |
| scale 计算 | `exp2(byte - 127)` per group | `bf16_to_f32(scale)` per group | ~2× |
| 每 expert dispatch 数 | gate+up+swiglu+down = 4 个 encoder | fused gate+up+swiglu = 1 encoder + down = 1 | ~2× encoder 开销 |

**理论乘数**：4 个问题叠加 → kernel 效率差 10-30×，与实测 MoE GPU 1100ms（MLX 同路径 ~200ms）一致。

---

### 7.1 Step 1：MXFP4 v3 kernel（核心，预期 5-10× MoE 加速）

**改动文件**：`src/models/moe_kernel.metal`

**新 kernel：`mxfp4_matvec_v3`**

关键设计（直接对应 flash-moe `dequant_matvec_4bit_v3`，适配 MXFP4 格式）：

```metal
#define ROWS_PER_TG_MXFP4 8

// NIBBLE_TO_FLOAT: 固定 16-entry LUT（MXFP4 格式，与 §29 bug 修复一致）
constant float NIBBLE_TO_FLOAT_MXFP4[16] = {
    0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f,
   -0.f,-1.f,-2.f,-3.f,-4.f,-6.f,-8.f,-12.f
};

kernel void mxfp4_matvec_v3(
    device const uint32_t* W_packed  [[buffer(0)]],  // [out_dim, in_dim/8]
    device const uint8_t*  scales    [[buffer(1)]],  // [out_dim, in_dim/32]  E8M0 bytes
    device const float*    x         [[buffer(2)]],  // [in_dim]
    device float*          out       [[buffer(3)]],  // [out_dim]
    constant uint&         out_dim   [[buffer(4)]],
    constant uint&         in_dim    [[buffer(5)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG_MXFP4 + simd_group;

    // ⚠️ 坑点 1：x_shared 加载必须所有线程参与，不能提前 return
    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;  // 必须在 barrier 之后

    uint packed_cols   = in_dim / 8;
    uint packed_per_gs = 4;   // group_size=32 / 8 nibbles_per_u32 = 4

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint8_t*  s_row = scales   + row * (in_dim / 32);

    float acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / packed_per_gs;
        // ⚠️ 坑点 2：bias 必须是 127，不是 128（§29 根本 bug）
        float sf = exp2((float)s_row[g] - 127.0f);

        // v5 LUT pattern：每 group 构建一次 LUT，避免反复 float(nibble) 转换
        float lut[16];
        for (uint v = 0; v < 16; v++) lut[v] = NIBBLE_TO_FLOAT_MXFP4[v] * sf;

        uint32_t packed = w_row[col];
        uint x_base = col * 8;
        acc += lut[(packed >>  0) & 0xF] * x_shared[x_base + 0];
        acc += lut[(packed >>  4) & 0xF] * x_shared[x_base + 1];
        acc += lut[(packed >>  8) & 0xF] * x_shared[x_base + 2];
        acc += lut[(packed >> 12) & 0xF] * x_shared[x_base + 3];
        acc += lut[(packed >> 16) & 0xF] * x_shared[x_base + 4];
        acc += lut[(packed >> 20) & 0xF] * x_shared[x_base + 5];
        acc += lut[(packed >> 24) & 0xF] * x_shared[x_base + 6];
        acc += lut[(packed >> 28) & 0xF] * x_shared[x_base + 7];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) out[row] = sum;
}
```

**dispatch 变更**（`engine.c` / `moe_metal_wrapper.c`）：

```objc
// 旧（每 thread 一行）:
uint32_t num_tgs_old = out_dim;
[enc dispatchThreads:MTLSizeMake(num_tgs_old,1,1)
    threadsPerThreadgroup:MTLSizeMake(1,1,1)];

// 新（8 rows / tg, 256 threads）:
uint32_t num_tgs_new = (out_dim + 7) / 8;
[enc dispatchThreadgroups:MTLSizeMake(num_tgs_new,1,1)
    threadsPerThreadgroup:MTLSizeMake(256,1,1)];
```

**验收**：
```bash
bash scripts/run_kernel_tests.sh          # kernel 单元测试
bash scripts/dsv4_smoke.sh                # Paris + 2+2 正确性
bash scripts/run_benchmark.sh             # 预期：MoE GPU ≤ 200ms，tok/s ≥ 2.0
```

**已知坑点**：
- ⚠️ `threadgroup_barrier` 之前不能 `return`，否则 x_shared 只被部分 thread 加载，out-of-bounds row 的 thread 不参与加载 → 剩余行得到 garbage（flash-moe 注释：_"ALL threads must participate in this load + barrier, even if their row is out of bounds"_）
- ⚠️ scale bias = **127**，不是 128（§29 根本 bug）
- ⚠️ simd_sum 结果只在 simd_lane == 0 有意义，其余 lane 写 out 会造成 race（已在 v3 中正确处理：`if (simd_lane == 0)`）

---

### 7.2 Step 2：fused_gate_up_swiglu_mxfp4（减少 encoder 数量）

**改动文件**：`src/models/moe_kernel.metal`

**新 kernel：`fused_gate_up_swiglu_mxfp4`**

将当前的 gate matmul + up matmul + swiglu = 3 个 encoder 合并为 1 个。基于 flash-moe `fused_gate_up_swiglu`，适配 MXFP4 格式：

```metal
kernel void fused_gate_up_swiglu_mxfp4(
    device const uint32_t* gate_W  [[buffer(0)]],
    device const uint8_t*  gate_s  [[buffer(1)]],
    device const uint32_t* up_W    [[buffer(2)]],
    device const uint8_t*  up_s    [[buffer(3)]],
    device const float*    x       [[buffer(4)]],
    device float*          out     [[buffer(5)]],
    constant uint&         out_dim [[buffer(6)]],
    constant uint&         in_dim  [[buffer(7)]],
    constant float&        route_w [[buffer(8)]],  // 路由权重（从 combine 移入）
    uint tgid, uint lid, uint simd_lane, uint simd_group ...
) {
    uint row = tgid * ROWS_PER_TG_MXFP4 + simd_group;

    // x 加载一次，gate 和 up 共用
    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) x_shared[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    uint packed_cols = in_dim / 8, packed_per_gs = 4;
    device const uint32_t* gw = gate_W + row * packed_cols;
    device const uint8_t*  gs = gate_s + row * (in_dim / 32);
    device const uint32_t* uw = up_W   + row * packed_cols;
    device const uint8_t*  us = up_s   + row * (in_dim / 32);

    float g_acc = 0.0f, u_acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint grp = col / packed_per_gs;
        float gsf = exp2((float)gs[grp] - 127.0f);
        float usf = exp2((float)us[grp] - 127.0f);
        float glut[16], ulut[16];
        for (uint v = 0; v < 16; v++) {
            glut[v] = NIBBLE_TO_FLOAT_MXFP4[v] * gsf;
            ulut[v] = NIBBLE_TO_FLOAT_MXFP4[v] * usf;
        }
        uint32_t gp = gw[col], up = uw[col];
        uint xb = col * 8;
        for (uint i = 0; i < 8; i++) {
            float xv = x_shared[xb + i];
            g_acc += glut[(gp >> (i*4)) & 0xF] * xv;
            u_acc += ulut[(up >> (i*4)) & 0xF] * xv;
        }
    }

    float g = simd_sum(g_acc), u = simd_sum(u_acc);
    if (simd_lane == 0) {
        // ⚠️ 坑点 3：SwiGLU clamp 必须保留（§20.1 历史 bug：曾有 out[tid]=999.0f 占位符）
        g = min(g, 10.0f);
        u = clamp(u, -10.0f, 10.0f);
        float silu = g / (1.0f + exp(-g));
        out[row] = silu * u * route_w;
    }
}
```

**dispatch 变更**：gate+up+swiglu 从 3 个 encoder 合并为 1 个，encoder 数从 `4/expert` → `2/expert`（fused gate+up+swiglu + down）。

**对 combine 的影响**：`route_w` 直接乘在 swiglu 输出上（flash-moe `fused_gate_up_swiglu` 不含，但 ds4 `kernel_dsv4_moe_swiglu_weight` 在 swiglu 后直接乘），可在 combine kernel 中省去每 element 的权重乘法。

**验收**：smoke test + 逐层对拍（`scripts/compare_metal_mlx.py`）确认 layer_00 rel_L2 不退化。

---

### 7.3 Step 3：batched 6-expert dispatch（ds4 slots6 模式）

**改动文件**：`src/models/moe_kernel.metal`、`src/metal_infer/engine.c`

参考 ds4 `kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32`，将 6 个 expert 的 gate+up+swiglu 合并为一次 dispatch：

```metal
kernel void fused_gate_up_swiglu_mxfp4_6slots(
    // 6 组 gate/up weights，通过 buffer index 区分
    device const uint32_t* gate_W0, device const uint8_t* gate_s0,
    device const uint32_t* up_W0,   device const uint8_t* up_s0,
    // ... gate_W1..5, up_W1..5 (共 24 个 weight buffer)
    device const float*    x        [[buffer(24)]],
    device float*          out      [[buffer(25)]],  // [6 * intermediate_dim]
    device const float*    route_ws [[buffer(26)]],  // [6] 路由权重
    constant uint& out_dim, constant uint& in_dim,
    uint tgid, uint lid, uint simd_lane, uint simd_group
) {
    // tgid 的高位 = expert index (0-5), 低位 = row within expert
    uint expert_idx = tgid / ((out_dim + 7) / 8);
    uint local_tgid = tgid % ((out_dim + 7) / 8);
    uint row = local_tgid * ROWS_PER_TG_MXFP4 + simd_group;

    // x_shared 对所有 expert 相同（x 是共享输入）
    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) x_shared[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    // 根据 expert_idx 选择对应的 gate/up weight 指针
    // （使用 switch 或通过 device const uint64_t* 地址表）
    // ... 计算 g_acc, u_acc（同 Step 2）
    // 写入 out[expert_idx * out_dim + row]
}
```

**dispatch**：
```objc
uint32_t num_tgs = 6 * (intermediate_dim + 7) / 8;
[enc dispatchThreadgroups:MTLSizeMake(num_tgs,1,1)
    threadsPerThreadgroup:MTLSizeMake(256,1,1)];
```

**收益**：encoder 数量从 `6 (gate+up+swiglu)` → `1`，减少 5 次 encoder 创建 + endEncoding 开销。

**更优方案（ds4 addr 模式）**：维护 `gate_addrs[256]`、`up_addrs[256]` 两个 uint64_t 数组（GPU-side 指针），kernel 根据路由 ID 直接 deref：

```metal
// kernel 内：
uint64_t gate_ptr = gate_addrs[expert_id];
device const uint32_t* gw = reinterpret_cast<device const uint32_t*>(gate_ptr);
```

SMELT 命中时 Metal buffer 已在 GPU，其 `[buffer contents]` 指针即为有效 GPU 地址。要求 Metal buffer 使用 `MTLResourceStorageModeShared`（已满足）。

⚠️ **坑点 4**：addr 模式需要检查 `gate_ptr == 0`（cache miss），为 null 时该 expert 的输出置 0（ds4 的 masked 变体），随后走 CPU combine 时跳过该 expert。

---

### 7.4 Step 4：GPU-side combine + RMSNorm（CMD3 内完成，消除 CPU 往返）

**改动文件**：`src/metal_infer/engine.c`、`src/models/moe_kernel.metal`

在 CMD3 的 expert forward 之后追加 2 个 encoder：

```
CMD3 内 encoder 序列（6-slot batched 后）：
  Enc 0: fused_gate_up_swiglu_mxfp4_6slots → buf_expert_mid[0..5]
  Enc 1: down_proj_mxfp4_v3 (6 experts batched) → buf_expert_out[0..5]
  Enc 2: moe_combine_residual_6 → buf_moe_hidden  (同 flash-moe kernel)
  Enc 3: rms_norm_sum_sq(buf_moe_hidden) → buf_norm_sq
  Enc 4: rms_norm_apply(buf_moe_hidden, next_layer_norm_w, buf_norm_sq) → buf_input
```

**`moe_combine_residual_6`**（适配 V4 的 6-expert，无 shared_gate sigmoid，shared expert 已在 CMD2 中处理）：

```metal
kernel void moe_combine_6(
    device const float* ffn_residual  [[buffer(0)]],  // mhc_post_ffn 后的残差
    device const float* shared_out    [[buffer(1)]],  // shared expert 输出
    device const float* expert_out0   [[buffer(2)]],
    // ... expert_out1..5
    device float*       out           [[buffer(8)]],
    device const float* weights       [[buffer(9)]],  // [6] 路由权重
    constant uint& dim, uint tid
) {
    if (tid >= dim) return;
    float moe = 0.f;
    moe += weights[0] * expert_out0[tid];
    moe += weights[1] * expert_out1[tid];
    // ...
    out[tid] = ffn_residual[tid] + moe + shared_out[tid];
}
```

⚠️ **坑点 5**：`rms_norm_apply` 写入 `buf_input`，但最后一层（layer 42）没有 next_layer_norm_w。需要特判：`if (layer == N_LAYERS - 1)` 跳过 rms_norm，或写到独立的 `buf_final_hidden`。

⚠️ **坑点 6**：CMD3 内的 blit encoder（用于拼接 expert 输出）必须在 compute encoders 之后排列。Metal 保证同一 CB 内 encoder 串行。

---

### 7.5 Step 5：Deferred CMD3（GPU/CPU 并行流水线）

**改动文件**：`src/metal_infer/engine.c`

参考 flash-moe `fused_layer_forward` 的 `DeferredExpertState` 模式（已在 design.md 设计）：

```c
// Layer N 结束时（CMD3 已 encode 完成）:
[cmd3 commit];          // 提交但不等待
[cmd3 retain];          // ⚠️ 坑点 7：必须 retain，防止 autorelease pool 提前释放
eng->deferred.cmd3 = cmd3;
eng->deferred.active = true;

// Layer N+1 开始时（在 CMD1 提交前）:
if (eng->deferred.active) {
    [eng->deferred.cmd3 waitUntilCompleted];  // GPU 已在跑 N 的 CMD3
    [eng->deferred.cmd3 release];
    eng->deferred.cmd3 = nil;
    eng->deferred.active = false;
    // GPU-side combine 已完成：buf_input 已就绪，直接提交 CMD1
}
```

**快速路径（prev_gpu_combined = true）**：
- CMD3(N) 写了 `buf_input`（下一层的 norm 输入）
- CMD1(N+1) 可立即提交，GPU queue 自动序列化 CMD3(N) → CMD1(N+1)
- CPU 在 CMD1 wait 之后再 read back hidden（buf_moe_hidden）

⚠️ **坑点 8**：`moe_infer_reset_kv()` 必须先 waitUntilCompleted + release 任何 in-flight CMD3，再清空状态。否则下一个请求会并发访问 GPU buffer → 数据污染。

⚠️ **坑点 9**：最后一层（layer 42）的 deferred CMD3 必须在 `moe_infer_forward` 返回前等待完成（不是在 layer 43 开始前），否则 `moe_infer_get_logits` 读到 stale 数据。

---

### 7.6 坑点总结（来自 flash-moe/ds4/dmlx 历史）

| # | 坑点 | 根因 | 修复 |
|---|------|------|------|
| 1 | x_shared 部分加载 | out-of-bounds thread 在 barrier 前 return | barrier 必须先于所有 return |
| 2 | MXFP4 scale 值差 2× | bias=128 而非 127 | `exp2(byte - 127.0f)` |
| 3 | MoE 输出恒为 0 | `out[tid] = 999.0f` 占位符未替换（§20.1）| `out[tid] = silu * u * route_w` |
| 4 | addr-mode 空指针 | cache miss 时 gate_addr=0 | dispatch 前检查；mask kernel 跳过 null addr |
| 5 | last layer combine 越界 | rms_norm_apply 读 next_layer norm weight 越界 | layer == 42 时跳过 rms_norm |
| 6 | SIMD reduce 87-97% 为 0 | 多级归约时 shared[] 竞争 | 严格按 flash-moe v3 的两级归约顺序 |
| 7 | CMD3 被 autorelease 提前释放 | ObjC retain count = 0 | commit 后显式 [cmd3 retain] |
| 8 | 请求间 KV 污染 | reset_kv 未等待 in-flight CMD3 | reset_kv 内先 wait+release deferred |
| 9 | 最后层 logits stale | layer 42 deferred CMD3 未等待 | forward() 末尾强制 waitUntilCompleted |
| 10 | mHC buffer 别名 | attn_post 和 ffn_pre 共用 buf_mhc_post | 分配独立 attn_post_buf 和 ffn_pre_buf |
| 11 | SwiGLU 缺少 clamp | gate/up 无截断，极端值炸 layer 29 | `min(gate,10)`, `clamp(up,-10,10)` |

---

### 7.7 预期性能收益

**数学上限（I/O 已消除）**：

```
DSV4 per token MoE ops：
  43 层 × 6 expert × (gate[2048,4096] + up[2048,4096] + down[4096,2048])
  = 43 × 6 × 3 × 2048 × 4096 MACs ≈ 6.5G MACs

M4 Pro GPU 峰值（大约 ~17 TFlops FP32）→ 6.5G MACs ≈ 0.38ms（理论极限）
实际有效带宽限制（Metal 4bit kernel，~30% 利用率）→ ~1.3ms/token
```

**现实预期**：

| 优化步骤 | MoE GPU (ms) | Total (ms) | tok/s |
|---------|-------------|------------|-------|
| 当前基线 | 1100 | 1410 | 0.71 |
| Step 1: v3 kernel | ~200 | ~510 | ~2.0 |
| Step 1+2: fused gate+up | ~160 | ~470 | ~2.1 |
| Step 1+2+3: 6-slot batched | ~130 | ~440 | ~2.3 |
| Step 1+2+3+4: GPU combine | ~130 | ~310 | ~3.2 |
| Step 1-5: deferred CMD3 | ~130 | ~260 | **~3.8** |

Step 1 是决定性的一步（5× MoE 加速）。Step 4（GPU combine）是第二大收益（消除 ~130ms CPU 往返）。

**能否达到 4 tok/s**：理论上可以，但取决于 M4 Pro 的实际 kernel 吞吐。如果 Step 1 后仍 ≤ 2 tok/s，需考虑「Hybrid MLX MoE」（§7.8）。

---

### 7.8 备选路径：Hybrid MLX MoE

如果 Steps 1-5 后 MoE GPU 仍比 MLX 路径慢：

**方案**：在 native engine 的 MoE 部分调用 MLX C API (`mlx_quantized_matmul`，mode=mxfp4)：

```c
// engine.c 内，对每个 selected expert：
mlx_array gate_out = mlx_quantized_matmul(
    ffn_normed_array,        // [1, 4096] bfloat16
    expert_gate_weights[e],  // [2048, 512] uint32 + scales
    /* transpose */ true,
    /* group_size */ 32,
    /* bits */ 4,
    /* mode */ "mxfp4"
);
```

**缺点**：引入 MLX eval() 同步屏障（§2e 的教训），每个 expert 一次 eval → 慢。只适合「共享 expert」（每层只有 1 个，eval 开销相对小）。

**风险**：MLX C API 的 expert by expert 调用会重新引入同步屏障，可能比 Step 1 的 native kernel 更慢。**先实施 Step 1 再决定**。

---

### 7.9 实施顺序与验收门

```
Step 1 → 运行 benchmark → 达到 ≥2.0 tok/s → 继续
Step 2 → 运行 smoke + benchmark
Step 3 → 运行 smoke + benchmark（addr 模式先 stub，slots6 先做）
Step 4 → 运行 smoke + kernel 单测 + benchmark
Step 5 → 运行 smoke + benchmark → 目标 ≥3.5 tok/s
```

每步验收命令：
```bash
# kernel 单测（~2s）
bash scripts/run_kernel_tests.sh

# 正确性（smoke）
bash scripts/dsv4_smoke.sh

# 性能
NATIVE_SMELT_N=20 bash scripts/run_benchmark.sh
```

**回退策略**：任一 step 引入 smoke 失败 → `git stash`，用 `scripts/compare_metal_mlx.py` 定位首个发散层，先修正确性再继续性能优化。
