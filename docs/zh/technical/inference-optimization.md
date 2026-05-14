# 推理速度优化方案 v1.0

> **目标**: 将每 token 延迟从 82.2ms 降至 25-30ms（12.2 tok/s → 33-40 tok/s），
> 在 48GB Mac 上将 100 token 请求从 ~8s 降至目标 ~3-4s。

---

## 1. 当前基线

**硬件**: Apple M4 Pro，48 GB 统一内存
**模型**: DeepSeek-V4-Flash-4bit，33 个分片（~141 GB 磁盘，4-bit 量化）
**模式**: SMELT 15% + stream + ExpertCache 4GB + temperature=0
**Commit**: `7e72a7`（2026-05-05）

| 指标 | 数值 |
|------|------|
| Prefill（首 token） | 370.5 ms |
| 稳态 ITL（token 3-10 均值） | **82.2 ms** |
| 吞吐（稳态） | **12.2 tok/s** |
| 权重占内存 | ~6 GB（SMELT 15%） |
| 专家缓存预算 | 4 GB（LFU） |
| NVMe 带宽（Apple） | ~5-7 GB/s |
| UMA 带宽（M4 Pro） | ~400 GB/s |
| Metal GPU（M4 Pro） | ~15 TFLOPS FP16 |

---

## 2. 热路径分解（Decode 阶段，每 Token）

```
82.2ms / token（43 层）
│
├── 0.5ms   Token 嵌入查询
├── 0.1ms   扩展至 mHC [B,1,4,H]
│
├── 78ms    43 个 Transformer 层（每层 1.81ms 均值）
│   │
│   ├── 0.15ms  mHC 注意力预处理（Sinkhorn 归一化）
│   ├── 0.05ms  RMSNorm（融合 fast 内核）
│   ├── 0.30ms  MLA 注意力
│   │   ├── 0.05ms Q/KV 投影（q_lora_rank 1024）
│   │   ├── 0.05ms RoPE 应用（YARN 缩放，部分维度 64/512）
│   │   ├── 0.05ms KV 缓存更新（CSA/HCA 压缩）
│   │   ├── 0.05ms CSA 索引（LightningIndexer top-512 选择）
│   │   ├── 0.05ms SDPA（缩放点积注意力）
│   │   └── 0.05ms O-lora 投影（8 组）
│   ├── 0.10ms  mHC 注意力后处理
│   ├── 0.05ms  mHC FFN 预处理
│   ├── 0.05ms  RMSNorm
│   ├── **0.90ms  MoE 前向 ← 主要瓶颈（占层时间 50%）**
│   │   ├── 0.05ms Gate 路由（256 专家中 score-based top-6）
│   │   ├── 0.03ms 共享专家前向
│   │   ├── **0.50ms 专家流式加载 / 缓存 I/O**
│   │   │   ├── 缓存查找（LFU，~70% 命中率）
│   │   │   ├── 缓存未命中 → PartialTensorReader 磁盘读取
│   │   │   └── 从 mmap/专家缓存构建 Array
│   │   ├── 0.20ms SwitchGLU 计算（gate_proj + up_proj + SiLU + down_proj）
│   │   ├── 0.02ms 分数重加权并求和
│   │   └── 0.10ms mHC FFN 后处理
│   └── 0.10ms  hidden.eval() ← 强制 GPU 同步
│
├── 0.20ms  HyperHead 压缩（mHC [4×H] → [H]）
├── 0.05ms  最终 RMSNorm
├── 0.50ms  LM head [1, 4096] × [4096, 129280] = ~530M FLOPs
└── 0.35ms  采样（slice→squeeze→astype f32→sample）
```

### 瓶颈汇总

| 排名 | 瓶颈 | 代码位置 | 预估耗时 | 类别 |
|------|------|----------|----------|------|
| B1 | MLX 逐 op 调度开销 | `DSV4Model.forward()` | ~30ms/tok | 计算 |
| B2 | 专家流式 I/O 延迟 | `expert_stream.zig:streamingForward()` | ~21ms/tok | I/O |
| B3 | 逐层 `hidden.eval()` 同步 | `deepseek_v4.zig:2753` | ~4.3ms/tok | 同步 |
| B4 | 计算-I/O 串行流水线 | `layer_prefetcher.zig` | ~15ms/tok | 流水线 |
| B5 | 逐层 ScopedArrayArena 分配/释放 | `deepseek_v4.zig` 层循环 | ~5ms/tok | 内存 |
| B6 | LM head 矩阵乘（530M FLOPs） | `deepseek_v4.zig:2772` | ~0.5ms/tok | 计算 |

---

## 3. 物理极限分析

### 内存带宽极限

M4 Pro 拥有 ~400 GB/s 统一内存带宽（CPU + GPU 共享）。

Decode 阶段每 token 数据传输量（单 token，batch=1）：

| 操作 | 每层数据量 | ×43 层 | 带宽耗时 |
|------|-----------|--------|----------|
| 注意力权重（MLA Q/KV/O） | ~40 MB | 1.7 GB | 4.3 ms |
| MoE 权重（7 专家，4-bit → 反量化） | ~42 MB | 1.8 GB | 4.5 ms |
| mHC + norms + 残差 | ~5 MB | 0.2 GB | 0.5 ms |
| LM head | ~530 MB | 0.5 GB | 1.3 ms |
| **合计** | **~87 MB** | **~4.2 GB** | **~10.6 ms** |

**带宽极限：~94 tok/s**（10.6ms/token）

### GPU 计算极限

M4 Pro ~15 TFLOPS FP16：
- MoE 每层：~700M FLOPs → 43 层 → ~30G FLOPs
- 注意力：~50M FLOPs/层 → ~2G FLOPs
- LM head：~0.5G FLOPs
- 合计：~32.5G FLOPs/token → 2.2ms 计算 → **~460 tok/s**

**结论：完全受限于内存带宽。计算从来不是瓶颈。**

### SSD I/O（流式模式）

SMELT + stream 模式下，专家权重来自 NVMe SSD（~5-7 GB/s）：
- 每层：42 MB 专家数据 ×（1 - 缓存命中率）的实际磁盘读取
- 无缓存：42 MB / 6 GB/s = 7 ms/层 → 301 ms/token（不可接受）
- 70% 缓存命中：~12.6 MB / 6 GB/s = 2.1 ms/层 → 90 ms → 与观测值吻合
- 95% 缓存命中：~2.1 MB / 6 GB/s = 0.35 ms/层 → 15 ms

**核心洞察：专家缓存命中率直接决定 decode 速度。**

---

## 4. 优化方案

### 阶段 1：低风险快速收益（Sprint 1，~1 周）

#### P1.1 — Decode 阶段选择性跳过 `eval()` ⭐

**代码**: `src/models/deepseek_v4.zig:2747-2756`

**问题**: `hidden.eval()` 在每层之后强制 Metal GPU 同步。注释说"允许 MLX 在内存中换入/换出大型模型的权重"——这对 prefill 有意义（多 token → 大量中间张量 ≥ 内存压力），但对 decode 阶段无意义（1 token → 极少的中间张量）。

**修复**: 添加 `is_decode` 参数；仅在 layer 0 和最后一层执行 eval。

```zig
// 在 DSV4Model.forward() 中，每层之后：
if (!is_decode or i == 0 or i == self.layers.len - 1) {
    try hidden.eval(); // decode 阶段仅在边界同步
}
```

**预期**: ITL 82.2ms → 78ms（**-5%**）
**风险**: 低。Decode 阶段内存压力极小。Prefill 路径不变。
**验证**: 执行前后 `make benchmark`，确保 7/7 正确性测试通过。

#### P1.2 — 扩大专家缓存预算 ⭐⭐

**代码**: `src/models/expert_cache.zig:14`（`DEFAULT_MAX_BYTES`）

**问题**: 4 GB 缓存容纳 ~682 个专家条目 → 每层 ~16 个。每层需要 6（路由）+ 1（共享）= 7 个专家。命中率 ~70% → 每层 ~2.1 次磁盘读取，每次 ~7ms。

**修复**: 将缓存从 4 GB 增加到 10 GB。48GB Mac 上：6 GB SMELT 预加载 + 6 GB backbone + 10 GB 缓存 + 2 GB KV 缓存 = 24 GB，完全在 48 GB 内。

```zig
// src/models/expert_cache.zig
pub const DEFAULT_MAX_BYTES: usize = 10 * 1024 * 1024 * 1024; // 10 GB
```

**备选**: 通过 CLI 可配置（`--expert-cache-gb 10`）。

**预期**:
- 缓存条目：682 → ~1700（每层 40 个）
- 命中率：70% → ~95%
- 磁盘 I/O 减少：~15ms/token
- ITL 78ms → 63ms（**-19%**）

**风险**: 如果总内存不超过 48GB 则无风险。在 `memory.zig` 中添加检查，如果专家缓存 + 模型超过可用 RAM 则发出警告。

#### P1.3 — 零拷贝 MLX 数组 ⭐

**代码**: `src/io/safetensors_reader.zig`、`src/models/deepseek_v4_loader.zig`、`src/models/expert_stream.zig`

**问题**: `mlx_array_new_data()` 将 mmap 数据拷贝到 MLX 管理的内存中。7 GB backbone 权重 + 每 token 专家切片，累计额外增加 ~2-3ms。

**修复**: 使用 `mlx_array_new_data_managed_payload()` 配合空操作 deleter，详见 [TTFT 优化方案](ttft-optimization.md#p2-mmap-zero-copy)。

```zig
// 替换:
const arr = c.c.mlx_array_new_data(slice.ptr, shape_i32.ptr, ndim, dtype);

// 为:
fn noopDeleter(_: ?*anyopaque) callconv(.c) void {}
const arr = c.c.mlx_array_new_data_managed_payload(
    @constCast(@ptrCast(slice.ptr)),
    shape_i32.ptr, ndim, dtype,
    null, noopDeleter,
);
```

**需修改的位置**: 3 处 mmap 路径调用点（非 pread 回退路径）。

**预期**: ITL 63ms → 61ms（**-3%**）
**风险**: 低。MmapPool 覆盖生命周期。若 `newBuffer` 因非页对齐指针拒绝，MLX 自动回退到拷贝路径。
**注意**: safetensors 内的专家张量通常不满足页对齐，实际零拷贝命中率可能较低。但改动无风险。

---

### 阶段 2：MLX 编译融合（Sprint 2，~2 周）

#### P2.1 — 层级 `compile()` 融合 ⭐⭐⭐⭐

**代码**: `src/models/deepseek_v4.zig` — `DSV4TransformerBlock.forward()`

**问题**: 每层前向传播调度 50 个以上独立 MLX 操作（matmul, add, multiply, reshape, transpose...），每个触发一次 Metal kernel 启动。Kernel 启动开销 ~10-50μs/op → 每层 ~1-2.5ms → 每 token ~43-108ms。此外 MLX 惰性图构建和调度还有额外开销。

**修复**: 将层前向传播包裹在 `mlx_compile()` 中以将所有操作融合为单个 Metal 计算图。Decode 阶段 shape 固定（batch=1, seq=1），图可完全编译。

**挑战**:

1. **KV 缓存更新路径**: KV 缓存更新使用的 `mlx_slice_update()` 是可编译的（纯函数，数组输入）。

2. **MoE 路由**: Gate 路由中的 `topk()` 和 `argpartition()` 操作可编译。

3. **专家流式加载**: `ExpertStreamProvider.streamingForward()` 从磁盘加载张量——不可编译。两种方案：
   - **方案 A（推荐）**: 将层分为两个编译段：`attention_block`（完全编译）和 `moe_compute`（编译，接收预加载的专家数组作为输入）。
   - **方案 B**: 在进入编译图前预加载该 token 的所有专家，作为输入传入。

4. **Prefill 阶段动态 shape**: 可变 seq_len → 单独编译图或回退到 eager 模式。

**架构**:

```
Token 生成循环
│
├── 专家预加载阶段（I/O，不编译）
│   └── 每层：缓存查找 / 磁盘读取 → 专家数组在内存中
│
└── 编译模型执行（单次 Metal dispatch）
    └── 43 层 + LM head → 一个融合 Metal 图
        ├── 注意力（MLA + CSA/HCA + SDPA）— 已编译
        ├── MoE gate 路由 — 已编译
        ├── SwitchGLU（使用预加载的专家）— 已编译
        ├── 共享专家 — 已编译
        └── mHC 预处理/后处理 — 已编译
```

**实现步骤**:

1. 创建 `decodeForward()` 函数，以预加载的专家数组作为输入
2. 模型加载后编译一次
3. 每 token：加载专家 → 执行编译图 → 采样

**预期**: 消除 ~30ms Metal kernel 启动 + 图构建开销。
ITL 61ms → **31ms**（**-49%**）

**风险**: 中。CSA/HCA 注意力中使用的某些 MLX op 可能不支持 `compile()`。需逐 op 兼容性测试。

#### P2.2 — 注意力模块编译（若全模型编译不可行的备选方案）

**预期**: 若全模型编译可行 ITL → ~25ms；仅编译注意力模块 → ~55ms。

---

### 阶段 3：I/O 流水线优化（Sprint 3，~2 周）

#### P3.1 — 多层专家预取 ⭐⭐⭐

**代码**: `src/models/layer_prefetcher.zig`

**问题**: 当前预取器仅在层 N 计算时加载层 N+1。SSD 延迟（~7ms/冷读取）导致计算等待 I/O。P1.2 后缓存命中率达 95%，此问题对首几个 token 仍有影响。

**修复**: 将单层预取替换为**多层环形缓冲区**：

```
GPU 上正在计算层 N
│
├── Worker 1: 预取层 N+1 的专家
├── Worker 2: 预取层 N+2 的专家
└── Worker 3: 预取层 N+3 的专家
```

```zig
pub const LayerPipeline = struct {
    workers: [3]std.Thread,
    ring: RingBuffer(PrefetchRequest, 8), // 环形缓冲区，非阻塞
    cache: *ExpertCache,
    reader: *PartialTensorReader,

    pub fn stage(self: *LayerPipeline, layer_idx: usize, expert_ids: []const u32) void {
        // 非阻塞：推送请求，worker 自行获取
        self.ring.push(.{ .layer = layer_idx, .experts = expert_ids });
    }

    pub fn ensureReady(self: *LayerPipeline, layer_idx: usize) void {
        // 仅在层未就绪时阻塞（3 级流水线下极少发生）
        while (!self.ring.isReady(layer_idx)) {
            std.Thread.yield() catch {};
        }
    }
};
```

**预期**: 隐藏 80-100% 剩余 SSD 延迟。ITL 31ms → **27ms**（**-13%**）

**风险**: 中。线程协调需谨慎实现。`PartialTensorReader` 必须支持同一 safetensors 文件的并发读取（POSIX 上 pread 是线程安全的，但 fd 共享需注意）。

**备选方案**: 若多线程过于复杂，实现**非阻塞单层预取**：使用 `mach_absolute_time()` 测量计算耗时，在层 N 结束时立即启动 N+1 的预取（不阻塞）。仅在到达层 N+1 且预取尚未完成时才阻塞。

#### P3.2 — safetensors 中专家权重顺序布局

**代码**: `src/models/expert_stream.zig:loadExpertSlicesCached()`

**问题**: `PartialTensorReader` 对每个专家的 gate_proj、up_proj、down_proj 分别进行 3 次文件读取。safetensors 将所有专家沿 axis 0 融合，但专家 42 的 3 个投影在文件中被其他专家的张量分隔 → 每个专家 3 次 seek。

**修复**: 构建 TensorIndex 时标注专家张量组，partial reader 可以用单次向量读取（preadv）获取一个专家的全部 3 个投影。

**预期**: 减少每专家 I/O syscall 从 3 到 1。延迟改善 ~1ms/tok。

---

### 阶段 4：计算优化（Sprint 4，~3 周）

#### P4.1 — 融合 SwitchGLU Metal 内核 ⭐⭐⭐

**代码**: 新文件 `src/models/metal/switchglu_fused.metal`

**问题**: SwitchGLU 前向执行 4 次矩阵乘 + SiLU 激活 + 逐元素乘 → 每层 6 次 Metal kernel 启动。每次启动意味着：构建 MTLComputeCommandEncoder → 设置缓冲区 → 调度 → 结束编码。

**修复**: 编写自定义 Metal 内核，将整个 SwitchGLU 融合为**一次**调度：

```metal
// switchglu_fused.metal
kernel void switchglu_fused(
    device const float* x,        // [N, D] 输入
    device const float* gate_w,   // [N_experts, intermediate, D] 打包
    device const float* up_w,     // [N_experts, intermediate, D] 打包
    device const float* down_w,   // [N_experts, D, intermediate] 打包
    device const uint* indices,   // [N, topk] 专家路由
    device float* y,              // [N, D] 输出
    constant uint& N,
    constant uint& D,
    constant uint& intermediate,
    constant uint& topk,
    // ...
) {
    // 融合: gather experts → gate_proj @ x → SiLU → up_proj @ x → multiply → down_proj
    // 单次内核调用，零中间分配
}
```

**与 MLX 集成**: 通过 `mlx_register_custom_op()` 注册为自定义 MLX op，或直接通过 `mlx_array_new_data()` 包装 Metal buffer 调用。

**预期**: 每层消除 6 次 kernel 启动中的 5 次。
ITL 27ms → **24ms**（**-11%**）

**风险**: 高。需要 Metal Shading Language 专业知识。需对照 Python 参考进行数值验证（如 ds4 使用测试向量的方式）。

**参考**: DeepSeek 的 [TileKernels](https://github.com/deepseek-ai/tilekernels)（CUDA）——dmlx 已引用于 Sinkhorn 归一化和融合 SwitchGLU。

#### P4.2 — KV 缓存更新优化

**代码**: `src/kvcache/` 和 `src/models/deepseek_v4.zig` 注意力路径

**问题**: KV 缓存通过多个 MLX slice/update 调用更新每个注意力头。CSA/HCA 压缩器增加了额外计算。

**修复**: Decode（单 token）时，KV 缓存更新只是追加一个 token 的 K 和 V 向量。这应该每个层只需一次 `mlx_slice_update()`，而不是多步 CSA/HCA 压缩管线（后者为 prefill/pooling 设计）。确认现有代码已对此优化。

**预期**: 小幅度改善（~1ms），若尚未优化。

---

### 阶段 5：推测解码（已构建 — 默认启用）

#### P5.1 — 默认启用 PLD ⭐⭐⭐

**代码**: `src/generation.zig:401`（`streamGenerateSpeculative`）、`src/speculative.zig`

**状态**: PLD（Prompt Lookup Decoding）和 EAGLE 推测解码已实现但未默认启用。PLD 在最近上下文中搜索匹配的 n-gram 并提出 2-3 个草稿 token，一次前向验证。

**修复**: 对贪婪解码（temperature=0）默认启用 PLD。~70% 接受率 × 3-token 草稿 → 有效 2.1× 吞吐。

**含 P1-P4 优化的预期**: 24ms ITL × 2.1× = 有效 **11.4ms/token** → **~88 tok/s**
**不含 P1-P4**（当前基线 + PLD）: 82ms × 2.1× = 有效 **39ms/token** → **~26 tok/s**

**风险**: 低。PLD 统计等价于自回归采样。已验证。

---

## 5. 组合效果汇总

| 阶段 | 优化 | 节省 | 新 ITL | 新 tok/s | 工作量 |
|------|------|------|--------|----------|--------|
| — | **当前基线** | — | 82.2ms | 12.2 | — |
| P1.1 | 选择性跳过 eval | 4ms | 78ms | 12.8 | 1 行 |
| P1.2 | 扩大缓存 4→10 GB | 15ms | 63ms | 15.9 | 1 常量 |
| P1.3 | 零拷贝数组 | 2ms | 61ms | 16.4 | 3 处调用 |
| P2.1 | MLX 层级编译 | 30ms | **31ms** | **32.3** | 2 周 |
| P3.1 | 多层预取 | 4ms | **27ms** | **37.0** | 2 周 |
| P4.1 | 融合 SwitchGLU 内核 | 3ms | **24ms** | **41.7** | 3 周 |
| P5.1 | PLD 推测（启用） | 2.1× | 有效 11.4ms | **有效 88** | 已构建 |

| 场景 | ITL | tok/s | 100 token 请求 | 200 token 请求 |
|------|-----|-------|----------------|----------------|
| 当前 | 82.2ms | 12.2 | **8.2s** | **16.4s** |
| 阶段 1 后 | 61ms | 16.4 | 6.1s | 12.2s |
| 阶段 2 后 | 31ms | 32.3 | **3.1s** ✅ | 6.2s |
| 阶段 3 后 | 27ms | 37.0 | **2.7s** ✅ | 5.4s |
| 阶段 4 后 | 24ms | 41.7 | **2.4s** ✅ | 4.8s |
| + PLD（P5） | 有效 11.4ms | 有效 88 | **1.1s** ✅ | **2.3s** ✅ |

**阶段 2 完成即达成目标：100 token 请求 ~3s。**

---

## 6. 实施路线图

```
第 1-2 周：阶段 1（低风险，立竿见影）
  ├── 第 1 天：P1.1 选择性跳过 eval
  │   └── 1 行修改 + 测试
  ├── 第 2 天：P1.2 扩大专家缓存
  │   └── 常量修改 + 内存检查
  └── 第 3-4 天：P1.3 零拷贝数组
      └── 3 处调用替换 + 验证无回退
  预期：82ms → 61ms（tok/s +35%）

第 3-4 周：阶段 2（MLX 编译 — 最大收益）
  ├── 第 5-7 天：Op 兼容性审查
  │   └── 测试 forward() 中每个 op 对 compile() 的支持
  ├── 第 8-10 天：实现 decodeForward() 封装
  │   └── 预加载专家 → 作为输入传递 → 编译
  ├── 第 11-12 天：集成测试
  │   └── 正确性：7/7 prompts + logits diff < 1e-5
  └── 第 13-14 天：性能验证
  预期：61ms → 31ms（tok/s +97%）

第 5-6 周：阶段 3（I/O 流水线）
  ├── 第 15-18 天：多层预取器
  │   └── 环形缓冲区 + worker 线程池
  ├── 第 19-20 天：PartialTensorReader 线程安全
  └── 第 21 天：集成 + 基准测试
  预期：31ms → 27ms（tok/s +15%）

第 7-9 周：阶段 4（Metal 内核 — 深度优化）
  ├── 第 22-25 天：编写融合 SwitchGLU Metal 内核
  ├── 第 26-28 天：对照 Python 参考数值验证
  ├── 第 29-31 天：MLX 自定义 op 集成
  └── 第 32-33 天：全流水线基准测试
  预期：27ms → 24ms（tok/s +13%）

持续：阶段 5（推测解码 — 已构建）
  └── 默认启用 PLD（贪婪解码模式）
  预期：在所有优化之上获得 2.1× 有效吞吐
```

---

## 7. 验证协议

### 优化前后基准测试

```bash
# 基线
make benchmark

# 每个阶段后重新运行：
make benchmark
MAX_TTFT_MS=300 MAX_ITL_MS=40 make benchmark
```

### 正确性验证

```bash
# 每次修改后确保 7/7 prompt 通过
make verify MODEL_PATH=~/models/DeepSeek-V4-Flash-4bit
make e2e MODEL_PATH=~/models/DeepSeek-V4-Flash-4bit
```

### Logit 差异检查

```bash
# 对比优化前后的 logits
./zig-out/bin/dmlx benchmark --model ~/models/DeepSeek-V4-Flash-4bit \
  --dump-logprobs /tmp/before.json --temp 0 -p "法国的首都是"
# ... 应用优化 ...
./zig-out/bin/dmlx benchmark --model ~/models/DeepSeek-V4-Flash-4bit \
  --dump-logprobs /tmp/after.json --temp 0 -p "法国的首都是"
# diff /tmp/before.json /tmp/after.json — 贪婪解码下应完全一致
```

### 内存安全

```bash
# 检查进程 RSS 不超过 48GB
/usr/bin/time -l ./zig-out/bin/dmlx chat --model ~/models/DeepSeek-V4-Flash-4bit \
  --smelt --smelt-experts 0.15 -p "Hello" 2>&1 | grep "maximum resident"
```

---

## 8. 风险登记

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| `compile()` 不支持 CSA/HCA 操作 | 中 | 高 | 回退为仅编译注意力模块（P2.2） |
| 多线程 PartialTensorReader 竞态 | 低 | 高 | 每线程独立 fd；POSIX 上 pread 线程安全 |
| 10GB 缓存在 48GB 上 OOM | 低 | 中 | `memory.zig` 中增加内存预算检查 |
| 融合 SwitchGLU 数值偏差 | 中 | 中 | 对照 Python 参考验证；ds4 测试向量方法 |
| PLD 产生更低质量输出 | 低 | 低 | PLD 统计等价；仅用于贪婪模式 |

---

## 9. 关键参考

| 项目 | 参考内容 |
|------|----------|
| [ds4](https://github.com/antirez/ds4) | 自定义 Metal 图执行器、非对称量化、融合内核 |
| [oMLX](https://github.com/jundot/omlx) | 分层 KV 缓存、多模型管理、continuous batching 模式 |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | `mx.compile()` 用法、`stream_generate` 架构 |
| [TileKernels](https://github.com/deepseek-ai/tilekernels) | 融合 SwitchGLU + Sinkhorn Metal 内核（CUDA → Metal 适配） |
| [vLLM](https://github.com/vllm-project/vllm) | PagedAttention、调度器三阶段循环、chunked prefill |

---

## 附录 A：修改源文件清单

| 文件 | 阶段 | 修改摘要 |
|------|------|----------|
| `src/models/deepseek_v4.zig` | P1.1, P2.1 | 条件 eval 跳过 + compile 封装 |
| `src/models/expert_cache.zig` | P1.2 | 默认缓存大小常量 |
| `src/models/expert_stream.zig` | P1.3, P3.1 | 零拷贝数组 + 流水线预取 |
| `src/io/safetensors_reader.zig` | P1.3 | `loadTensor` 零拷贝路径 |
| `src/models/deepseek_v4_loader.zig` | P1.3 | `loadWeightsSelective` 零拷贝路径 |
| `src/models/layer_prefetcher.zig` | P3.1 | 多层环形缓冲区流水线 |
| `src/models/metal/switchglu_fused.metal` | P4.1 | 新增：融合 SwitchGLU 内核 |
| `src/generation.zig` | P5.1 | 贪婪模式默认启用 PLD |
| `src/memory.zig` | P1.2 | 专家缓存内存预算检查 |
| `src/main.zig` | P1.2 | `--expert-cache-gb` CLI 标志 |

---

## 附录 B：48GB Mac 内存预算

```
总统一内存:                          48.0 GB
├── macOS + 其他应用:                 ~8.0 GB（预留）
├── dmlx 可用:                       ~40.0 GB
│
├── Backbone 权重（mmap）:            ~6.0 GB  （注意力 + norms + 共享 + 路由）
├── SMELT 预加载（15% 专家）:          ~6.0 GB  （38 个专家 × ~160MB/个）
├── 专家缓存（P1.2: 10GB）:          ~10.0 GB  （LFU，接近 100% 命中率）
├── KV 缓存（paged, 32K 上下文）:     ~1.5 GB  （CSA/HCA 压缩后）
├── MLX 运行时 + Metal:              ~2.0 GB  （缓冲区、命令队列）
│
└── 剩余余量:                        ~14.5 GB  （安全应对峰值）
```
