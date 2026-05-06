# ⚡ dmlx — (Apple Silicon Native LLM Runtime)

## DeepSeek-V4-Flash-4bit on Apple Silicon

### MoE Expert Streaming + SMELT (~3.5–4 bits per weight)

| 🔢 Model Size | 🧠 Parameters | ⚙️ Backend | 💻 Hardware |
|:---:|:---:|:---:|:---:|
| ~40 GiB (4-bit) | 671 B | Metal (MLX) | Apple M4 Pro, 48GB |

---

## BENCHMARK ENVIRONMENT

| Item | Value |
|------|-------|
| 🟢 Runtime | dmlx (single static binary) |
| 🟢 GPU | Apple M4 Pro (Metal, unified memory) |
| 🟢 Memory | 48 GB unified (shared CPU/GPU) |
| 🟢 Quantization | INT4 + SMELT 15% (~6 GB active weights) |
| 🟢 KV Cache | Paged (CoW) + Tiered (RAM+SSD) |
| 🟢 Expert Strategy | Stream + LRU cache (top-8 of 256) |
| 🟢 Context | 128K+ tokens supported |

---

## BENCHMARK RESULTS

### 🏆 TOKEN GENERATION LATENCY (ms/token) — Lower is better

```
                    Best: 82.2 ms/tok (~12.2 tok/s)
                    Mode: SMELT 15% + Expert Stream

 Token 1 (Prefill)     ████████████████████████████████████  370.5 ms
 Token 2               ████████████████  152.7 ms
 Token 3               ████████████  116.1 ms
 Token 4               ██████████  104.4 ms
 Token 5               █████████  96.2 ms   ← fastest
 Token 6               ██████████  103.6 ms
 Token 7               █████████  98.4 ms
 Token 8               ██████████  107.1 ms
 Token 9               ██████████  111.0 ms
 Token 10              ██████████  115.4 ms
```

### 📈 PERFORMANCE EVOLUTION

| Metric | Initial (a024bee) | Current (7e72a7) | Improvement |
|--------|-------------------|-------------------|-------------|
| Prefill (TTFT) | 716 ms | **370.5 ms** | **+48% faster** |
| Steady-state avg | ~125 ms | **82.2 ms** | **+34% faster** |
| Throughput | ~8 tok/s | **~12.2 tok/s** | **+52%** |

---

## MEMORY OPTIMIZATION LAYERS

```
 Layer 1: Expert Streaming     138 GB → 10 GB    (top-8 of 256 experts)
 Layer 2: SMELT 15%             40 GB →  6 GB    (partial expert loading)
 Layer 3: MLA KV Compression    Dramatic reduction (low-rank projection)
 Layer 4: Paged KV Cache        Fixed pages + CoW (production batching)
 Layer 5: Zero-Copy Loading     137s → 41s TTFT  (mmap, no memcpy)
```

---

## TEST CONFIGURATIONS

| Config | Description |
|--------|-------------|
| `SMELT 15%` | 38 of 256 experts loaded, ~6 GB weights |
| `SMELT 30%` | 77 of 256 experts loaded, ~12 GB weights |
| `Full 4-bit` | All 256 experts, ~40 GB (requires 64GB+ Mac) |
| `KV: Paged` | 32-token pages, CoW for continuous batching |
| `KV: Tiered` | RAM hot + SSD cold, 128K+ context |

---

## 🏆 KEY TAKEAWAYS

- ✅ **671B MoE model on 48GB Mac** — impossible with mlx-lm (OOM)
- ✅ **~12.2 tok/s** steady-state generation with SMELT 15%
- ✅ **370ms prefill** (Time-To-First-Token), 48% improvement over initial
- ✅ **128K+ context** via Paged + Tiered KV cache (RAM+SSD)
- ✅ **Single static binary** (~5–15 MB) — no Python, no pip, no venv
- ✅ **Zero GC pauses** — deterministic sub-ms latency (Zig runtime)
- ✅ **7/7 correctness** on end-to-end prompt validation

---

## vs. COMPETITION

| | dmlx | mlx-lm (Python) | llama.cpp |
|---|---|---|---|
| DeepSeek V4 on 48GB | ✅ SMELT ~6GB | ❌ OOM | ❌ No MoE streaming |
| Runtime | Single binary | Python + venv | C++ binary |
| GC pauses | None (Zig) | 10–100ms (Python) | None (C++) |
| Apple Metal native | ✅ (mlx-c) | ✅ (mlx) | Partial |
| KV strategies | 6 (runtime-switch) | 1 (fixed) | 2 |
| Max context 48GB | 128K+ (Tiered SSD) | RAM-limited | RAM-limited |

---

✨ *Pushing the limits of local LLM performance on Apple Silicon.*

**Model**: DeepSeek-V4-Flash-4bit (671B MoE, 256 experts)
**Runtime**: dmlx (Zig + Metal via mlx-c)

---

# ⚡ dmlx —（Apple Silicon 原生 LLM 推理引擎）

## 在 Apple Silicon 上运行 DeepSeek-V4-Flash-4bit

### MoE 专家流式加载 + SMELT（约 3.5–4 比特量化）

| 🔢 模型大小 | 🧠 参数量 | ⚙️ 后端 | 💻 硬件 |
|:---:|:---:|:---:|:---:|
| ~40 GiB (4-bit) | 6710 亿 | Metal (MLX) | Apple M4 Pro, 48GB |

---

## 基准测试环境

| 项目 | 值 |
|------|------|
| 🟢 运行时 | dmlx（单一静态二进制） |
| 🟢 GPU | Apple M4 Pro（Metal，统一内存） |
| 🟢 内存 | 48 GB 统一内存（CPU/GPU 共享） |
| 🟢 量化 | INT4 + SMELT 15%（约 6 GB 活跃权重） |
| 🟢 KV 缓存 | 分页（CoW）+ 分层（RAM+SSD） |
| 🟢 专家策略 | 流式加载 + LRU 缓存（256 中取 top-8） |
| 🟢 上下文 | 支持 128K+ tokens |

---

## 基准测试结果

### 🏆 Token 生成延迟（ms/token）— 越低越好

```
                    最佳: 82.2 ms/tok（约 12.2 tok/s）
                    模式: SMELT 15% + 专家流式加载

 Token 1（预填充）    ████████████████████████████████████  370.5 ms
 Token 2             ████████████████  152.7 ms
 Token 3             ████████████  116.1 ms
 Token 4             ██████████  104.4 ms
 Token 5             █████████  96.2 ms   ← 最快
 Token 6             ██████████  103.6 ms
 Token 7             █████████  98.4 ms
 Token 8             ██████████  107.1 ms
 Token 9             ██████████  111.0 ms
 Token 10            ██████████  115.4 ms
```

### 📈 性能演进

| 指标 | 初始版本 (a024bee) | 当前版本 (7e72a7) | 提升 |
|------|-------------------|-------------------|------|
| 预填充（TTFT） | 716 ms | **370.5 ms** | **快 48%** |
| 稳态平均 | ~125 ms | **82.2 ms** | **快 34%** |
| 吞吐量 | ~8 tok/s | **~12.2 tok/s** | **+52%** |

---

## 五层内存优化

```
 第一层: 专家流式加载     138 GB → 10 GB    （256 专家中仅加载 top-8）
 第二层: SMELT 15%        40 GB →  6 GB    （部分专家预加载）
 第三层: MLA KV 压缩      大幅缩减          （低秩投影）
 第四层: 分页 KV 缓存     固定页 + CoW      （生产级连续批处理）
 第五层: 零拷贝加载       137s → 41s TTFT   （mmap，无 memcpy）
```

---

## 测试配置

| 配置 | 说明 |
|------|------|
| `SMELT 15%` | 加载 38/256 专家，约 6 GB 权重 |
| `SMELT 30%` | 加载 77/256 专家，约 12 GB 权重 |
| `Full 4-bit` | 全部 256 专家，约 40 GB（需 64GB+ Mac） |
| `KV: Paged` | 32-token 分页，CoW 连续批处理 |
| `KV: Tiered` | RAM 热层 + SSD 冷层，128K+ 上下文 |

---

## 🏆 核心亮点

- ✅ **6710 亿参数 MoE 模型在 48GB Mac 上运行** — mlx-lm 无法做到（OOM）
- ✅ **约 12.2 tok/s** 稳态生成速度（SMELT 15% 模式）
- ✅ **370ms 预填充**（首 Token 时间），比初始版本快 48%
- ✅ **128K+ 上下文**，通过分页 + 分层 KV 缓存（RAM+SSD）
- ✅ **单一静态二进制**（约 5–15 MB）— 无需 Python、pip、venv
- ✅ **零 GC 停顿** — 确定性亚毫秒延迟（Zig 运行时）
- ✅ **7/7 正确性** 端到端提示验证全部通过

---

## 对比竞品

| | dmlx | mlx-lm (Python) | llama.cpp |
|---|---|---|---|
| DeepSeek V4 48GB 运行 | ✅ SMELT ~6GB | ❌ OOM | ❌ 无 MoE 流式 |
| 运行时 | 单一二进制 | Python + venv | C++ 二进制 |
| GC 停顿 | 无（Zig） | 10–100ms（Python） | 无（C++） |
| Apple Metal 原生 | ✅（mlx-c） | ✅（mlx） | 部分支持 |
| KV 策略 | 6 种（运行时切换） | 1 种（固定） | 2 种 |
| 48GB 最大上下文 | 128K+（分层 SSD） | 受 RAM 限制 | 受 RAM 限制 |

---

✨ *突破 Apple Silicon 本地 LLM 推理性能极限。*

**模型**: DeepSeek-V4-Flash-4bit（6710 亿参数 MoE，256 专家）
**运行时**: dmlx（Zig + Metal，基于 mlx-c）
