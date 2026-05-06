# DeepSeek V4 MoE 在小内存 Mac 上的技术深度解析

> **dmlx 如何在 48GB MacBook Pro 上运行 ~150GB 的 MoE 模型**

---

## 问题所在

DeepSeek V4 是一个 671B 参数的 Mixture-of-Experts 模型。即使在 4-bit 量化下，完整模型仍重约 40GB。在 FP16 下，仅权重就超过 150GB。在一台只有 48GB 统一内存的消费级 MacBook Pro 上运行它看似不可能——但 dmlx 做到了。

## 解决方案：五层内存优化

### 第一层：MoE Expert Streaming（138GB → 10GB）

DeepSeek V4 使用 256 个路由专家 + 共享专家，采用 top-k 路由。每个 token 仅激活一小部分专家（通常 top-8）。dmlx 的 `expert_stream.zig`（649 行）利用了这种稀疏性：

- **按需加载**：仅通过 `PartialTensorReader`（基于 pread 的部分张量读取）将活跃专家加载到内存中
- **LRU 专家缓存**：常用专家常驻内存；冷门专家被淘汰
- **层预取**：在当前层计算期间预取下一层的专家
- **内存削减**：从 ~138GB（全部专家加载）降至 ~10GB（仅流式加载活跃专家）

```
Source: src/models/expert_stream.zig (649 lines)
        src/models/moe_router.zig (top-k routing)
```

### 第二层：4-bit 量化（40GB → 6-12GB，借助 SMELT）

dmlx 支持六种量化格式，SMELT 系统实现了部分专家加载：

| 模式 | 专家加载数 | 内存 | 延迟影响 |
|------|---------------|--------|----------------|
| Full 4-bit | 256 (100%) | ~40GB | 基线 |
| SMELT 30% | ~77 | ~12GB | +10-15% |
| SMELT 15% | ~38 | ~6GB | +10-15% |

**SMELT 工作原理：**
- 自动检测模型文件中实际存在多少专家（并非全部 256 个都可能存在）
- 仅加载存在专家的权重
- 施加路由偏置（`smelt_mask`）以防止路由器选择未加载的专家
- 对偶尔需要的缺失专家回退到流式加载

```
Source: docs/en/technical/4bit-smelt.md
        docs/en/technical/smelt-flow.md
```

### 第三层：MLA KV Cache 压缩（2×heads×dim → 2×latent_dim）

Multi-head Latent Attention (MLA) 通过低秩投影压缩 KV Cache：

- **MLA 之前**：KV cache 大小 = 2 × n_heads × head_dim（长上下文时巨大）
- **MLA 之后**：KV cache 大小 = 2 × latent_dim（显著缩小）
- **FP8 存储**：非 RoPE KV 维度以 FP8 (E4M3) 存储，进一步将内存减半

```
Source: src/models/deepseek_v4.zig (MLA implementation, 3,091 lines)
```

### 第四层：六级 KV Cache 策略系统

根据你的内存预算选择合适策略：

| 策略 | 内存特征 | 最佳场景 |
|----------|---------------|----------|
| **Standard** | 完整 KV 缓冲区 | 短序列，单次请求 |
| **Rotating** | 固定窗口环形缓冲区 | 超长序列（避免 OOM） |
| **Quantized** | 4/8/16-bit KV 压缩 | 内存受限场景 |
| **Paged** ⭐ | 32-token 页面 + CoW | 连续批处理（生产默认） |
| **PagedQuantized** | Paged + Quantized 组合 | 极致内存优化 |
| **Tiered** | RAM 热数据 + SSD 冷数据 + LRU | 超长上下文 + 多模型 |

```
Source: src/kvcache/paged.zig (1,152 lines)
        src/kvcache/tiered.zig
```

### 第五层：零拷贝模型加载（7GB memcpy → 0）

TTFT 优化计划消除了模型加载期间不必要的内存拷贝：

| 阶段 | 内容 | 优化前 | 优化后 |
|-------|------|--------|-------|
| P0 | 二进制索引缓存 | 解析 33 个分片 67s | ~1s mmap 读取 |
| P2 | 零拷贝权重加载 | ~7GB memcpy | 0（直接 mmap） |
| P3 | 分批分片 I/O | 随机读取 | 顺序 OS 预读 |

```
Source: docs/en/technical/ttft-optimization.md
```

---

## 架构一览

```
┌─────────────────────────────────────────────────────────┐
│                    dmlx Inference Engine               │
├─────────────────────────────────────────────────────────┤
│  Model (DeepSeek V4, 3,091 lines)                        │
│  ├── MLA (Multi-head Latent Attention)                   │
│  ├── MoE Router (top-k, 256 experts)                     │
│  ├── Expert Stream (on-demand loading, 10GB peak)        │
│  ├── YARN RoPE (1M+ context)                             │
│  └── mHC (multi-Hyper Connection)                        │
├─────────────────────────────────────────────────────────┤
│  KV Cache (6 strategies, runtime-switchable)             │
│  ├── Paged (32-token pages, CoW, Wyhash prefixes)        │
│  ├── Tiered (RAM + SSD, LRU eviction)                    │
│  └── Prompt Cache (save/restore to safetensors)          │
├─────────────────────────────────────────────────────────┤
│  Quantization (6 formats)                                │
│  ├── Affine INT4/INT8 (group_size: 32/64/128)            │
│  ├── MXFP4 / NVFP4 (microscaling)                       │
│  ├── FP8 E4M3                                            │
│  └── TurboQuant (Lloyd-Max + QJL)                       │
├─────────────────────────────────────────────────────────┤
│  Generation Engine                                        │
│  ├── Speculative Decoding (PLD + EAGLE)                  │
│  ├── Guided Decoding (JSON Schema / Regex FSM)           │
│  └── Continuous Batching (OpenAI-compatible API)         │
└─────────────────────────────────────────────────────────┘
```

---

## 为什么选择 Zig？

| 方面 | Python (mlx-lm) | Zig (dmlx) |
|--------|----------------|---------------|
| **内存控制** | GC + 隐式 | 显式，编译期检查 |
| **并发** | GIL 受限 | 真正的并行，无 GIL |
| **部署** | Python 运行时 + 依赖 | 单个静态二进制 |
| **Metal 访问** | 通过 mlx-c FFI | 直接内核集成 |
| **安全性** | 运行时错误 | 编译期保证 |
| **启动** | ~2s 解释器预热 | 瞬时原生启动 |

---

## 实际性能

**硬件**：MacBook Pro M4 Max，48GB 统一内存  
**模型**：DeepSeek-V4-Flash-4bit，33 个分片（原始 ~150GB）

| 指标 | 数值 |
|--------|-------|
| TTFT (32-token prompt) | 200-500ms |
| ITL (inter-token latency) | 250-500ms |
| 吞吐量 | 2-4 tokens/s |
| 内存（SMELT 15%） | ~6GB 权重 + KV cache |
| 最大上下文（Paged + Tiered） | 128K+ tokens |

---

## 应用场景

→ 详细用例请参见 [应用场景](../scenarios/README.md)。

| 场景 | dmlx 的优势 |
|----------|-------------|
| **本地 LLM 推理** | 在笔记本上运行 671B 模型——无需云端 |
| **隐私优先应用** | 所有数据保留在设备上，零网络出站 |
| **边缘部署** | Mac mini 作为团队的私有推理服务器 |
| **离线/受限区域访问** | 无需互联网即可获得完整 LLM 能力 |
| **开发与测试** | 无需 GPU 集群成本即可迭代 LLM 应用 |
| **研究** | 实验 MoE 路由、KV Cache 策略 |

---

## 相关文档

- [DeepSeek V4 修复历史](../deepseek-v4/) — Chat 模板修复、优化计划
- [KV Cache 深度解析](../analysis/04-kv-cache-subsystem.md) — 六大策略详解
- [量化流水线](../analysis/06-quantization-training.md) — 所有格式解析
- [TTFT 优化](../technical/ttft-optimization.md) — 模型加载优化
- [SMELT 系统](../technical/smelt-flow.md) — 专家流式加载架构
- [项目路线图](../../ROADMAP.md) — 未来改进
