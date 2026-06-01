# dmlx → flash-moe 对齐方案（已归档 / SUPERSEDED）

> **⚠️ 本文档已归档。** 有效方案与最新进展见
> **[`dsv4-first-class-support-plan.md`](./dsv4-first-class-support-plan.md)**（metal-first 路线）。
>
> 本文档保留作为历史记录：
> - §1.5 诊断记录（2026-06-01 乱码根因分析）已并入新文档 §1.5 / §0
> - §2/§4 的 flash-moe / ds4 架构分析与 kernel 对照表已并入新文档 §8-§11
> - §1/§3/§5 的旧状态表与「已放弃」结论部分**已过时**（其中「混合方案数值无法对齐」为误判，
>   真因是未提交的注意力重写改坏了 backbone，见新文档 §1.5）
>
> 不再更新本文件。

---

# （以下为历史内容）

# dmlx → flash-moe 对齐方案

> **日期**: 2026-05-26
> **硬件**: Apple M4 Pro, 48GB
> **模型**: DeepSeek-V4-Flash-4bit (43 层, K=6, 256 experts, MXFP4)
> **目标**: 3.0 tok/s, E2E 7/7

---

## 1. 当前状态

| 指标 | 值 |
|------|-----|
| E2E 正确性 | 7/7 |
| Token/s (cold) | ~0.4 |
| Token/s (warm) | ~1.0 |
| 每 token I/O 量 | ~3.35 GB (43 层 × 6 experts × 13.4 MB) |
| SSD 有效带宽 | ~1 GB/s |
| 单 token 理论下限 | 3.35s (纯 I/O) |

**瓶颈**：SSD I/O。GPU 计算仅占 ~30%（Metal kernel 约 ~1.8ms/layer × 43 = 77ms），I/O 占 ~97%。

**mach_absolute_time Bug**：已修复。Forward 日志之前将 ticks 当纳秒，少报 41.67x。实际每 token 3.3s，非 80ms。

**bufPrintZ Bug**：已修复。Zig 0.16 的 `bufPrint` 不再 null-terminate，导致 `fopen(path)` 失败。

---

## 1.5 诊断记录 (2026-06-01)

> **结论：之前一版未提交的"注意力 / RoPE 重写"是乱码根因，已 `git stash`。当前工作区已回到 `origin/main`(`e3be289`) 基线并验证输出正确。**

### 背景

某一版（未提交）的工作区改动把进度文档大量条目标记为 "✅ 已完成"（正确 MQA 注意力、BOS 修复、Lazy dequant、KV Cache、多 token SDPA 等），但这些从未通过 smoke test。实测两个 prompt 均为乱码：

| Prompt | 工作区(重写版) | 期望 |
|--------|---------------|------|
| `2+2=` | `algebraically free 2` | `4` |
| `The capital of France is` | `mesigned geopolitical geopolitical...` | `Paris` |

去掉 `--metal-moe`（纯 MLX 路径）同样乱码 → 排除 Metal MoE kernel，问题在共享 backbone。

### 对照实验

以 `../dm/dmlx` 为参考（经校验与本仓库 `origin/main` `e3be289` **逐字节一致**，文档记录此版本 7/7 正确）。

将全部未提交改动 `git stash` 后，回到 HEAD 基线重建，纯 MLX 路径输出：

| Prompt | HEAD 基线 | 判定 |
|--------|----------|------|
| `2+2=` | `. The user's query` | ✅ 连贯 |
| `The capital of France is` | `. The capital of France is Paris.` | ✅ 正确 |

**根因确认：污染在未提交的注意力 / RoPE 重写，不在 Metal MoE，也不在已提交历史。**

### 具体污染点（已 stash 到 `stash@{0}`）

`DSV4Attention.forward` 被整段改写，致命改动：

1. **手写 SDPA 替换 MLX `fast_scaled_dot_product_attention`** —— 重写版**丢失 `sink_logits`**（DeepSeek-V4 attention sink），mask 处理也变更。
2. **RoPE pair 布局翻转** —— `(half_dim, 2)` → `(2, half_dim)`，slice/stack 轴随之改变，改变了 Q/K 旋转语义。
3. **compressKV（CSA/HCA 压缩注意力）路径被禁用**。

> ⚠️ 更正：之前文档「❌ 已放弃 - 混合 MLX+Metal 方案 - 数值无法对齐」的结论属于**误判**。真正原因是 backbone 注意力被改坏，与 Metal/MLX 互操作无关。

### 后续处置

stash 内混合三类改动，须区分对待：

| 类别 | 处置 |
|------|------|
| (A) 注意力 / RoPE 重写 | ❌ **丢弃**（乱码根因） |
| (B) FP8 loader (`dequantFp8Weights`) | ⬜ 可保留，需独立验证 |
| (C) Metal engine 脚手架 | ⬜ 可保留（已提交部分在 HEAD） |

纪律：注意力如需改动，**必须保留 `sink_logits` 与原 RoPE 布局**，且每改一处立即 smoke。

---

## 2. flash-moe 架构分析

flash-moe 源码位于 `../flash-moe/metal_infer/infer.m` (7151 行) + `shaders.metal` (1296 行)。

### 2.1 核心流水线（每层 3 命令缓冲）

```
CMD3(N-1) deferred → CMD1: attention 投影 (1.22ms GPU)
                   → CPU: 结果刷出 (0.01ms)
                   → CMD2: o_proj + norm + routing + shared expert (0.55ms GPU)
                   → CPU: softmax + topK 路由 (0.003ms)
                   → I/O: 并行 pread K=4 experts (2.41ms SSD)
                   → CMD3: expert forward + GPU combine + norm (0.04ms encode, DEFERRED)
```

**CMD3 延迟提交**（`infer.m:5434`）：
```objc
[cmd_experts commit];  // 不 waitUntilCompleted — 立即返回
```
GPU 处理 CMD3 的同时，CPU 已进入下一层或下一 token。

**GPU-side combine**（`infer.m:5354-5431`）：CMD3 内部包含 3 个 encoder：
1. `moe_combine_residual` — 8 expert 输出加权和 + residual + shared gate
2. `rms_norm_sum_sq` — 归一化所需的平方和
3. `rms_norm_apply_bf16` — 用**下一层**的 norm weight 做归一化 → 直接写入下一层的输入缓冲

**消除 CPU 往返**：当 GPU-side combine 激活时，CMD3 的输出直接在 GPU 上完成 combine+norm，无需 CPU 读取、combine、写回。

### 2.2 时序 Expert 预测

`infer.m:4196-4215, 5196-5248`：

```
token N-1 完成后：每层存下 routing indices → g_pred_experts[layer][0..K-1]
token N 开始时：async_pread_start() 用预测的 experts 预取到 B 缓冲集
token N 到达该层时：检查预测是否命中
  - 命中：直接用 buf_multi_expert_data_B[p]（零 I/O）
  - 未命中：同步 io_pool_dispatch() 读取到 A 缓冲集
```

命中率 ~71%（OS page cache 的热 fds 辅助）。

### 2.3 双层缓冲

`infer.m:938-939`：
```objc
buf_multi_expert_data[MAX_K]     // 集 A — 当前层
buf_multi_expert_data_B[MAX_K]   // 集 B — 预测预取
```

### 2.4 I/O 线程池

`infer.m:2940-3058`：
- 4 个持久化 pthread（`NUM_IO_THREADS = 4`），每 expert 一个
- Generation counter + condition variable 同步
- `io_pool_dispatch()` 填充任务 → broadcast → wait 完成

我们已有类似实现（`expert_pread.zig`），但当前用 `std.Thread.spawn` per expert，不是持久化线程池。

### 2.5 Metal Kernel 优化

`shaders.metal:251`（`dequant_matvec_4bit_v3`）：
- **Threadgroup tiling**: 8 rows/group, 256 threads, 8 SIMD groups
- **Shared memory 输入缓存**: `threadgroup float x_shared[4096]` — 256 线程协作加载
- **Coalesced 读取**: SIMD lane 按 stride 32 读列，相邻 lane 读相邻 uint32
- **FMA 优化**: 预计算 `scale*x` 和 `bias*x`，`fma(nibble, scale*x, bias*x)` 一条指令完成解量+乘法
- **SIMD reduction**: `simd_sum(acc)` 单指令归约

我们的 kernel 是 naive 单线程 per row，无 shared memory，无 coalesced 读取，无 SIMD reduction。

---

## 3. 实施进度 (2026-05-27)

### ✅ 已完成

| 组件 | 文件 | 说明 |
|------|------|------|
| MXFP4 公式 | `moe_kernel.metal` | `NIBBLE_TO_FLOAT[16] * exp2(scale-128)`, Python 验证与 MLX 一致 |
| Naive MoE kernel | `moe_kernel.metal` | gate_up_swiglu + dequant_matvec + moe_combine, 99.9% 匹配 Python |
| SIMD kernel | 已回退 | 87-97% 结果错误，bug 在 reduction，待后续修复 |
| bufPrintZ 修复 | `expert_pread.zig` | Zig 0.16 `bufPrint` 不再 null-terminate |
| mach_absolute_time 修复 | 5 文件 | tick→ms 换算，之前少报 41.67x |
| 时序预测测量 | `expert_stream.zig` | 命中率 20-54%，V4 expert 局部性低 |
| 权重提取 | `deepseek_v4.zig` | `extractWeightsForEngine()` — embed/norm/gate_proj float32 指针 |
| **Metal 推理引擎** | `src/metal_infer/` | |
| ├ MoE forward | `engine.c` | gate/up/SwiGLU/down/combine, Metal GPU dispatch |
| ├ I/O thread pool | `engine.c` | 6 持久化 pthread + cond var, flash-moe 模式 |
| ├ RMSNorm | `engine.c` + `moe_kernel.metal` | Metal GPU rms_norm_sum_sq + rms_norm_apply |
| ├ Routing gate | `engine.c` | Metal matvec [256, 4096] @ [4096] → CPU softmax+topK |
| ├ Q/K/V 投影 | `engine.c` | Metal matvec_f32 kernel ready, 权重待接入 |
| ├ CPU RoPE | `engine.c` | ds4 YaRN partial RoPE (mode=2, tail-only) |
| ├ Per-layer forward | `engine.c` | `moe_infer_forward_layer()` 串联全部组件 |
| └ **Server 集成** | `state.zig` + `deepseek_v4.zig` | 43 层全通无 crash, --metal-moe flag |
| 混合方案 | 已放弃 | MLX attention + Metal MoE: 数值漂移致命 (5/5 prompt 失败) |
| **结论** | | Metal MoE 无法替代 MLX matmul, 必须完整 Metal 引擎 |

### ⬜ 进行中

| 组件 | 优先级 | 说明 |
|------|--------|------|
| Attention 权重 dequant | P0 | wq_b/wo_b 量化 → dequant → float32, 当前 OOM (43层×16MB×2=1.4GB) |
| CPU SDPA | P0 | 单 token self-attention |
| MLA attention 迁移 | P1 | 从 ds4 迁移 indexed_mixed_attention |

### ❌ 已知问题

| 问题 | 现象 | 原因 |
|------|------|------|
| Attention dequant OOM | Server 启动 crash | 43层 attention 权重 dequant 内存超限 |
| SIMD kernel 错误 | 87-97% 输出值为 0 | reduction bug, 已回退为 naive |
| BOS token 重复 | 输出全是 `<｜begin▁of▁sentence｜>` | attention pass-through, 无有效 attention 计算 |

### 测试方式

```bash
# 构建
rm -rf .zig-cache zig-out && zig build -Doptimize=ReleaseFast

# 启动 server (30-60s 加载)
./zig-out/bin/dmlx serve \
    --model ~/models/DeepSeek-V4-Flash-4bit \
    --port 8930 --max-tokens 30 --temperature 0 \
    --smelt --smelt-strategy stream --smelt-experts 0.20 \
    --smelt-cache 0 \
    --expert-packed-dir ~/models/DeepSeek-V4-Flash-4bit/packed_experts \
    --metal-moe

# 测试 (另一个终端)
curl -s http://localhost:8930/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"default","messages":[{"role":"user","content":"2+2="}],"max_tokens":5,"temperature":0}' \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['choices'][0]['message']['content'])"

# Benchmark (完整测试, ~10min)
bash scripts/run_benchmark.sh
```

### 调试方式

```bash
# 引擎 debug 输出 (stderr)
./zig-out/bin/dmlx serve ... 2>/tmp/engine_stderr.log

# Python 公式验证
python3 -c "
import numpy as np
# 读取 hidden state dump 对比 Metal 输出
hidden = np.fromfile('/tmp/metal_hidden.bin', dtype=np.float32)
# ... NIBBLE_TO_FLOAT * exp2(scale-128) 计算 ...
"
```

### ❌ 已放弃

| 方案 | 原因 |
|------|------|
| Metal MoE 替换 MLX matmul | 1e-7 × 258 matmuls = token 级 logit 偏差 |
| SIMD kernel (flash-moe v3 移植) | reduction bug, 87-97% 输出错误 |
| 时序预测 as primary optimization | V4 expert 局部性仅 35% vs flash-moe 71% |
| 混合 MLX+Metal 方案 | 数值无法对齐, 根本性限制 |

## 4. 引擎架构

### flash-moe (`../flash-moe/metal_infer/`)

| 文件 | 内容 | 迁移价值 |
|------|------|---------|
| `shaders.metal:251` | `dequant_matvec_4bit_v3` | ✅ 已迁移（适配 MXFP4 + V4 维度） |
| `shaders.metal:169` | `fused_gate_up_swiglu` | ✅ 已迁移 |
| `shaders.metal:1261` | `moe_combine_residual` | ⬜ K=8 硬编码，需改为 K=6 |
| `shaders.metal:745` | `rms_norm_sum_sq` | ✅ 已迁移 |
| `shaders.metal:779` | `rms_norm_apply` | ✅ 已迁移 |
| `infer.m:2124` | `full_attention_forward()` | ❌ Qwen 的 GQA attention，V4 用 MLA |
| `infer.m:2025` | `apply_rotary_emb()` | ⬜ 标准部分 RoPE，可参考但 V4 用 YaRN tail |
| `infer.m:3060` | `async_pread_start/wait` | ⬜ GCD async I/O 模式可迁移 |

### ds4 (`../ds4/metal/`)

| 文件 | 内容 | 迁移价值 |
|------|------|---------|
| `dsv4_rope.metal:68` | `kernel_dsv4_rope_tail_f32` | ✅ V4 YaRN tail RoPE，可直接迁移 |
| `dsv4_misc.metal:577` | `kernel_dsv4_indexed_mixed_attention_heads8` | ✅ V4 MLA 混合注意力核心 |
| `dsv4_kv.metal:104` | `kernel_dsv4_fp8_kv_quantize_f32` | ⬜ FP8 KV 缓存量化 |
| `flash_attn.metal:139` | Multi-stage FlashAttention | ⬜ prefill 加速 |
| `dsv4_hc.metal` | Head composition (HC) | ⬜ 注意力头合成 |
| `ds4_metal.m:2697` | `ds4_gpu_encode_rope_tail_inplace()` | ✅ RoPE 调度参考 |

## 5. 实施进度 (2026-05-27)

### ✅ 已完成

| 组件 | 状态 | 文件 |
|------|------|------|
| MXFP4 公式 | ✅ LUT + exp2 | `moe_kernel.metal` |
| Naive MoE kernel | ✅ 99.9% 匹配 Python | `moe_kernel.metal` |
| SIMD kernel | ❌ 有 bug，87-97% 错误 | 已回退为 naive |
| RMSNorm kernel | ✅ | `moe_kernel.metal` |
| Matvec kernel | ✅ | `moe_kernel.metal` |
| I/O thread pool | ✅ | `engine.c` |
| 权重提取 | ✅ MLX → float32 指针 | `deepseek_v4.zig` |
| 混合方案 (MLX attn + Metal MoE) | ❌ 数值漂移，不可行 | 已放弃 |
| 时序预测 | ✅ 命中率 20-54% | `expert_stream.zig` |

### ⬜ 进行中

| 组件 | 文件 |
|------|------|
| 引擎集成到 server | `state.zig`, `engine.{h,c}` |
| Attention (简化版) | `engine.c` |
| End-to-end 正确性验证 | 手动测试 + benchmark |

### 下一步

1. 将引擎集成到 server 生成循环
2. 实现简化 attention（Metal Q/K/V/O matvec + CPU SDPA）
3. 验证端到端正确性
4. 渐进迁移 ds4 的 MLA attention kernel

移植 flash-moe `shaders.metal` 的关键 kernel，适配 DeepSeek V4 维度：

| Kernel | flash-moe 维度 | V4 维度 |
|--------|--------------|---------|
| `fused_gate_up_swiglu` | [1024, 4096] | [2048, 4096] |
| `dequant_matvec_4bit` | [4096, 1024] | [4096, 2048] |
| `moe_combine_residual` | K=4, add shared gate | K=6, no shared gate |
| `rms_norm_sum_sq` | 4096 | 4096 |
| `rms_norm_apply_bf16` | 4096 | 4096 |

**关键改进**（vs 之前的 kernel）：
- Threadgroup tiling + shared memory 输入缓存
- Coalesced 全局内存读取（stride 32）
- SIMD reduction（`simd_sum`）
- FMA 解量：`fma(nibble, scale*x, 0.0)` 替代 `nibble * scale * x`
- MXFP4 LUT：`NIBBLE_TO_FLOAT[16]` = `{0,1,2,3,4,6,8,12, -0,-1,-2,-3,-4,-6,-8,-12}`
- 公式：`w = LUT[nibble] * exp2(scale - 128.0)`

**已验证正确**：Python 端 manual 计算与 MLX `quantized_matmul` 完美匹配（max diff 0.000000）。

### Phase 2: 时序 Expert 预测

在 `expert_stream.zig` 中添加：
```
g_pred_experts: [43][6]u32  // 每层上一 token 的 expert indices
g_pred_valid: bool           // 首 token 后设为 true

// token N 开始时（layer 0 入口）：
if (g_pred_valid) {
    for each layer L:
        async_prefetch(layer L, g_pred_experts[L])  // 预测预取到 B 缓冲
}

// 每层 MoE 计算前：
actual = router.topk(hidden)  // 实际路由
for each expert e in actual:
    if predicted[L].contains(e):
        use B_buffer[e]  // 命中，零 I/O
    else:
        io_pool_dispatch(A_buffer[e])  // 未命中，同步读取
```

### Phase 3: GPU-side Combine + RMSNorm

CMD3 末尾添加 `moe_combine_residual` + `rms_norm_sum_sq` + `rms_norm_apply`，使输出直接在 GPU 上完成归一化，消除 CPU 往返。

需要传下一层的 attention input_norm weight（`layer.ln_in.weight`）。

### Phase 4: 延迟 CMD3 + 流水线

```
Token 的 layer L:
1. CMD3(L-1) 完成 → GPU-side combine 已将 normed hidden 写入 buf_input
2. CMD1(L): attention 投影（GPU）
3. CPU: attention 计算（RoPE, KV cache, SDPA 等）
4. CMD2(L): o_proj + residual + routing + shared expert
5. CPU: softmax + topK → 确定 experts
6. I/O: 读取 expert 权重
7. CMD3(L): 专家 forward + combine + norm → DEFERRED (不等待)
8. 进入 layer L+1

Token 结束时:
- complete_deferred_experts() 等待最后一个 CMD3
```

### 数据流

```
MLX 域 (attention, embedding, LM head)
    ↓ hidden state (CPU buffer)
Metal 域 (switch_mlp 的 gate/up/down matmul)
    ↓
Metal 域 (combine + RMSNorm)
    ↓ normed hidden (GPU buffer → CPU buffer for next MLX layer)
```

**关键接口**：每层 MoE 需要从 MLX 获取 hidden state（float32），传给 Metal，Metal 返回 combine+norm 后的 hidden。

---

## 4. 文件变更清单

| 文件 | 变更 | 行数估计 |
|------|------|---------|
| `src/models/moe_kernel.metal` | 重写：SIMD 优化 kernel (v3 风格) | ~400 |
| `src/models/moe_metal_wrapper.c` | 新增：CMD3 流水线、combine+norm encoder | ~300 |
| `src/models/metal_moe.zig` | 新增：Zig 绑定、预测表、双层缓冲 | ~150 |
| `src/models/expert_pread.zig` | 修改：添加 async prefetch API | ~50 |
| `src/models/expert_stream.zig` | 修改：Metal MoE 路径 + 预测逻辑 | ~100 |
| `src/main.zig` | 恢复 --metal-moe flag | ~5 |
| `build.zig` | Metal/Foundation 框架链接 | ~10 |

---

## 5. 性能预估

flash-moe 实测 4.36 tok/s（60 层 K=4）。V4 模型 43 层 K=6：

| 阶段 | 耗时/layer | 43 层总计 |
|------|-----------|----------|
| CMD1: attention 投影 | 1.22ms | 52.5ms |
| CPU: attention 计算 | ~0.5ms | 21.5ms |
| CMD2: o_proj + norm + routing | 0.55ms | 23.7ms |
| I/O: pread（预测命中） | 0.5ms | 21.5ms |
| CMD3: expert forward (deferred) | 0.04ms | 1.7ms |
| **总计** | **~2.8ms** | **~121ms** |

理论：1000/121 = **8.3 tok/s**。保守估计含 overhead 达到 **3-5 tok/s**。

I/O 优化关键：时序预测命中率 ≥70% → 有效 I/O 量降至 ~1 GB/token（从 3.35 GB）。

---

## 6. 验证计划

1. **Kernel 正确性**：Python 对比 Metal vs MLX 输出（已建立测试方法）
2. **单层 benchmark**：Metal kernel 延迟测量 vs MLX `gather_qmm`
3. **E2E 正确性**：7-prompt benchmark，目标 7/7
4. **性能回归**：`bash scripts/run_benchmark.sh` → 目标 3.0+ tok/s
5. **Commit 规则**：仅 benchmark 验证后 commit，7/7 才 push

---

## 7. 风险

- **MLX-Metal 互操作**：MLX eval 可能阻塞 Metal command buffer 执行
- **内存压力**：双层缓冲 + GPU buffer 增加 ~100-200MB 内存
- **正确性回归**：GPU-side combine+norm 的数值精度可能略有偏差
- **复杂度**：Metal pipeline 难以调试（无 printf，需 readback buffer）>
