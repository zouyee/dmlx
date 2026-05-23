# dmlx Optimization Roadmap

> **Date**: 2026-05-19 (深度分析综合版)
> **Baseline**: commit 2eb821fc (SMELT 15% + 10GB cache + mlock-backbone)
> **Hardware**: Apple M4 Pro, 48GB unified memory
> **Source Analysis**: expert_cache.zig, expert_stream.zig, expert_preload.zig, layer_prefetcher.zig, engine_loop.zig, prefix_cache.zig, deepseek_v4.zig

---

## Executive Summary

dmlx 是一个基于 Zig + MLX 的推理引擎，目标是在 Apple Silicon（48GB）上运行 141GB DeepSeek-V4-Flash-4bit MoE 模型。核心挑战是**内存带宽瓶颈**：模型体积是物理内存的 3 倍，必须通过智能缓存和 I/O 调度来弥补。

**当前状态**：Server 端 16-19 tok/s（warm, 内部计时），Client 端 ~113s/30-token cold、~30s/10-token warm（SMELT 20% + 6GB）。
**目标**：Warm client 吞吐 6-7 tok/s（100-token < 15s）。
**路径**：减少 cache miss 总量（减轻 VM 压力）→ I/O-compute overlap → 减少 page fault。

> **⚠️ 指标说明**：Server 端 `RequestLog` 的 tok/s 只计量 generate 阶段 CPU 时间，
> 不含 macOS VM 子系统阻塞 socket write 的延迟。真实用户感知以 curl `time_starttransfer` 为准。
> "27-30s/100-token" 来自旧配置（05-16, SMELT 20% + 6GB cache, 多次请求后 cache 极度 warm），
> 当前 Config C 实测为 ~190s/100-token（cold）、~54s/10-token（warm 第 3 次请求）。

---

## 1. Current Performance (2026-05-19, Post Phase 1)

### Before Phase 1 (commit 0243aeb baseline)

| Metric | Value | Notes |
|--------|-------|-------|
| **Server-side tok/s (steady)** | **22.0** | SMELT 15% + 10GB cache + mlock (Config C, 05-22) |
| **Server-side tok/s (warm, latest)** | **22-26** | commit 0243aeb |
| **100-token HTTP total** | **193s** | bench: client 吞吐 ~0.5 tok/s |
| **30-token HTTP total** | **102s** | bench: client 吞吐 ~0.3 tok/s |
| **Steady-state ITL** | ~56ms | Token 7+ |
| **Cache hit rate** | **23.7%** | 71217 hits / 229740 misses |
| **Startup time** | 86s | SMELT 15% + mlock (Config C) |

### After Phase 1 (layer-partitioned cache + routing stats + yield)

| Metric | Value | Change | Notes |
|--------|-------|--------|-------|
| **Server-side tok/s (steady)** | **22.8** | — | Median ITL 40.1ms |
| **Steady-state ITL (median)** | **40.1ms** | **-28%** | Was 56ms |
| **Steady-state ITL (mean)** | **43.9ms** | **-22%** | Last 100 steps |
| **ITL P90** | **84.4ms** | — | Much less variance |
| **100-token HTTP total** | **192s** | -0.5% | Still VM-dominated |
| **30-token HTTP (warm)** | **63-71s** | — | New: 2nd request 67s |
| **Cache hit rate (overall)** | **22.6%** | -1.1% | Includes cold warmup |
| **Cache hit rate (per-step avg)** | **26.7%** | **+3%** | Post-warmup average |
| **Cache hit rate (best 30-step group)** | **33.2%** | **+9.5%** | Steps 31-60 |
| **Cache hit rate (per-step peak)** | **56.0%** | **+32%** | Single step max |
| **Startup time** | **160s** | +60s | Extended warmup (8 prompts) |
| **Cache entries (post-warmup)** | **7594** | — | Up from ~3800 |
| **Per-layer budget** | **143MB** | — | ~72 entries/layer, ~24 experts/layer |
| **Unique routing experts seen** | **1990/11008** | — | 18% after 8 warmup prompts |

**关键发现**：Server 端 100 token 只需 ~5.6s，但 client 端需要 193s。
差距 **34x** 完全来自 macOS page thrashing（141GB 模型 on 48GB RAM）。

### Phase 1 Benchmark Analysis (2026-05-19)

**收益分析**:
```
ITL 改善:          56ms → 40ms (median), -28%
Server tok/s:      维持 22.8 (与 baseline 22-26 一致)
Client 延迟:       未改善 (仍被 VM pressure 主导)
```

**为什么 client 延迟没有改善**:
- Server 端 ITL 改善了 28%（56→40ms），说明 cache 分区确实减少了 I/O
- 但 client 延迟仍被 macOS VM 子系统压力主导（page fault → kernel busy → syscall blocked）
- 每次 cache miss 仍然触发 mmap page fault，40ms ITL 中 ~30ms 是等待 SSD I/O
- **需要减少 miss 绝对数量**（异步预取）而不仅仅是优化 hit 路径

**Layer-partitioned cache 的实际效果**:
- 每层 72 entries，能缓存约 24 个 expert（需要 gate+up+down 3 个 tensor slot）
- 256 experts 中缓存 24 个 = 9.4% 覆盖率
- 但由于 routing 偏斜，实际 hit rate 达到 26.7%（比随机好 3x）
- Peak hit rate 56% 说明 routing 确实高度集中于少数热门 expert

**下一步优先级调整**:
- P1.1 异步预取的优先级进一步提升：client 延迟的瓶颈是 miss 导致的 page fault，
  不是 hit 路径的速度。需要通过提前 pread 来避免 page fault，而非仅仅提高 hit rate。

### Optimal Configuration

```bash
zig build -Doptimize=ReleaseFast

dmlx serve --model ~/models/DeepSeek-V4-Flash-4bit \
  --smelt --smelt-strategy stream --smelt-experts 0.15 \
  --smelt-cache 10240 --mlock-backbone --temperature 0
```

> **注意**：必须使用 `ReleaseFast` 构建。Debug 模式性能约为 Release 的 1/2。

---

## 2. Architecture Deep Dive (源码级分析)

### 2.1 组件状态总览

| Component | File | Lines | Status | Key Detail |
|-----------|------|-------|--------|------------|
| Expert Cache | `expert_cache.zig` | 745 | **Enabled** | Global LFU, 10GB, single-threaded, Wyhash |
| Expert Stream | `expert_stream.zig` | 672 | **Enabled** | mmap + ExpertCache + dedup + PartialTensorReader |
| Expert Preload | `expert_preload.zig` | 415 | **Enabled** | Sequential expert selection (0..N), cache_bias=-1000 |
| Layer Prefetcher | `layer_prefetcher.zig` | 234 | **Disabled** | Lock-free FSM, blocked by MLX thread-safety |
| Prefix Cache | `prefix_cache.zig` | 323 | **NEW** | LRU, FNV-1a hash, skip repeated prefill |
| Engine Loop | `engine_loop.zig` | 1515 | **Enabled** | Serial processing, batch decode, warmup, guided |
| DSV4 Model | `deepseek_v4.zig` | 3221 | **Enabled** | MLA + MoE (256 experts, top-6), FP8 KV |

### 2.2 数据流分析

```
Request → engine_loop.zig
  → DSV4Model.forward (deepseek_v4.zig)
    → 43 layers × {
        MLA Attention (latent compressed, FP8 KV)
        MoE Router → top-6 expert IDs
        streamingForward (expert_stream.zig)
          → ExpertCache.get (expert_cache.zig)
            HIT  → 直接返回 tensor
            MISS → PartialTensorReader.readExpertRows → mmap page fault
                 → ExpertCache.put → LFU eviction if over budget
      }
  → sampling → token output
```

### 2.3 关键源码结构分析

**ExpertCache (expert_cache.zig)**:
- 全局单实例 LFU 策略，`AutoHashMap(CacheKey → *CacheEntry)`
- CacheKey = `(layer_idx: u32, tensor_name_hash: u64, expert_id: u32)`
- 每次 get() 增加 frequency，可能触发 promote（插入排序维护链表顺序）
- 每次 put() 从 tail 驱逐 lowest-frequency entries
- **结构性缺陷**：浅层 expert 天然 frequency 高（每 token 必经），驱逐深层 expert

**ExpertStreamProvider (expert_stream.zig)**:
- 支持 `preload` 和 `stream` 两种策略
- Stream 模式下使用 `PartialTensorReader` (mmap-based) + `ExpertCache`
- 已实现 expert deduplication（prefill 阶段 30-50% fewer loads）
- 支持 `FdPool`（pread）和 `MmapPool` 双路径

**LayerPrefetcher (layer_prefetcher.zig)**:
- 完整的 lock-free 状态机（IDLE→PENDING→DONE→STOPPED）
- 后台 thread 通过 `std.Thread.spawn` 创建
- **禁用原因**：worker 内调用 `PartialTensorReader.readExpertRows` 触发 MLX array 构造（不线程安全）
- 改造方向明确：分离 raw I/O 和 MLX Array 构造

**Engine Loop (engine_loop.zig)**:
- 串行处理请求（一次一个），支持 batch decode
- Warmup 使用 5 个固定 prompt 预热 cache
- 集成 guided decoding (JSON schema/regex)
- 支持 streaming token delivery via CompletionSignal

### 2.4 Already Deployed Optimizations

| Optimization | Source | Impact | Date |
|-------------|--------|--------|------|
| Expert Deduplication | `expert_stream.zig` | 30-50% fewer expert loads during prefill | 05-16 |
| **Layer-Partitioned LFU Cache (6GB)** | `expert_cache.zig` | Per-layer isolation, eliminates cross-layer eviction bias | **05-19** |
| **Routing Frequency Statistics** | `expert_stream.zig` | Per-layer expert activation tracking | **05-19** |
| **Statistics-Based Smart Warmup** | `engine_loop.zig` | 8 warmup prompts (5+3 extra), routing-guided | **05-19** |
| **I/O Throttling (Layer Yield)** | `expert_stream.zig` | Yield every 8 layers to reduce VM storm | **05-19** |
| Prefix Cache | `prefix_cache.zig` | Skip prefill on repeated prompts | 05-19 |
| **~~Cache-Aware Routing Bias (P1.4)~~** | `deepseek_v4.zig` | **❌ 已验证不可行（三种方案均失败）** | **05-22** |
| Batch Decode | `engine_loop.zig` | Continuous batching for concurrent requests | 05-12 |
| FP8 KV Storage | `deepseek_v4.zig` | 50% KV memory reduction | 05-10 |
| SMELT 20% Preload | `expert_preload.zig` | +15% tok/s vs 10% (superseded by Config C: 15%+mlock) | 05-16 |
| ReleaseFast Build | build.zig | ~2x vs Debug | — |

---

## 3. Root Cause Analysis

### 3.1 为什么 Client 延迟 >> Server 延迟

**核心问题**：Server 端 100 token 只需 5.6s，但 Client 端需要 193s（cold）/ 54s+/10-token（warm）。

> 注：之前文档记录 "27-30s/100-token (warm)" 来自 05-16 旧配置（SMELT 20% + 6GB cache），
> 在该配置下连续跑 5+ 次请求后 cache 高度 warm（hit rate 上升到 40-50%），page fault 减少，
> 从而 VM 压力降低、socket write 延迟改善。Config C（SMELT 15% + 10GB + mlock）虽然
> server-internal tok/s 更高 (+24%)，但 client 延迟未改善 — 因为 mlock 保护 backbone
> 但 expert cache miss 仍然 76%，page fault 总量不变。

**因果链**：
```
141GB 模型 on 48GB Mac
    ↓
76% expert 访问是 cache miss → 触发 mmap page fault
    ↓
每 token 258 次 expert 访问 × 76% miss = ~196 次 page fault
    ↓
macOS kernel VM 子系统饱和
    ↓
连带拖慢 socket accept/read/write 等系统调用
    ↓
Client 延迟 = Server 计算时间 + OS 内存管理开销
```

### 3.2 193s 的时间分解

```
├── ~98s: Cold start（首次请求 backbone + expert page-in）
│         证据: 30-token 102s 中，server 端只需 4.2s → 98s 是 cold start
├── ~89s: 持续 page thrashing（70 额外 token × 1.24s/token OS 开销）
│         证据: 增量 91s / 70 token = 1.3s/token，server 只需 0.056s
└── ~5.6s: 实际推理时间（server 端 100 token × 56ms）
```

### 3.3 交叉验证（来自 5 个独立实验）

| 实验 | 来源文档 | 关键发现 |
|------|---------|---------|
| mmap vs pread | `pread-expert-loading.md` | pread 消除 VM 压力但损失 44% tok/s |
| OS thread 替代 fiber | `socket-write-latency.md` | 0% improvement → 非调度问题 |
| posix write bypass | `socket-write-latency.md` | 0% improvement → 非 write 问题 |
| 4GB vs 8GB cache | `cache-benchmark-results.md` | 8GB: -30% ITL, +102% hit rate |
| 6GB vs 10GB cache | optimization-roadmap | 10GB optimal with mlock (6GB was optimal without mlock) |

**结论**：优化方向聚焦于 **提升 cache hit rate** 和 **降低 page fault 对 OS 的冲击**。

---

## 4. Next Optimization Priorities

所有优化围绕两个目标：
1. **提升 cache hit rate**（减少 page fault 总量）
2. **降低 page fault 对 OS 的冲击**（减少 VM 压力对 client 延迟的影响）

---

### P0.1: 层级分区 Cache ⭐⭐⭐

**Expected**: hit rate 24% → 35%+
**Risk**: 低
**Effort**: 2-3 天
**Code**: `src/models/expert_cache.zig`
**Status**: 🔄 未实现

**实现**:
- 新增 `LayerPartitionedCache` struct，包装 `[]ExpertCache`（每层独立 LFU）
- 每层预算 = 6GB / num_layers ≈ 143MB（约 24 个 expert slot）
- API 兼容：`get(key)` / `put(key, tensor, size)` / `stats()` 接口不变
- `ExpertStreamProvider` 已切换使用 `LayerPartitionedCache`
- `LayerPrefetcher` 已适配新类型
- 新增 `layerStats(layer_idx)` 用于 per-layer 诊断

**Problem**: 当前全局 LFU 有结构性缺陷。浅层 expert（layer 0-5）每个 token 都被访问，
frequency 天然高于深层 expert。LFU 驱逐深层 expert 来保留浅层 → 深层永远 miss。

**源码分析** (`expert_cache.zig:1-100`):
- 当前结构: `AutoHashMap(CacheKey → *CacheEntry)` + doubly-linked list (LFU)
- CacheKey: `(layer_idx: u32, tensor_name_hash: u64, expert_id: u32)`
- 已有 layer_idx 作为 key 一部分，改造为 per-layer 只需重构存储结构
- 无线程安全要求（单线程），重构风险低
- Wyhash 用于 key hashing，性能良好

**Fix**: 将全局 `ExpertCache` 拆分为 per-layer 独立 cache：

```zig
// 替换全局 cache:
cache: ?*ExpertCache = null,

// 为 per-layer 数组:
layer_caches: ?[]ExpertCache = null, // [43]ExpertCache, 每层独立 LFU

// 每层 143MB ≈ 24 个 expert（每个 ~6MB for gate+up+down）
const per_layer_budget = cache_budget_mb * 1024 * 1024 / num_layers;
```

**进阶**：按层的 routing 集中度动态分配预算（先收集 per-layer routing 统计）。

**Verification**:
1. 对比 per-layer cache 前后的 hit rate
2. 对比 server tok/s 和 client 延迟
3. 确保 7/7 correctness pass

---

### P0.2: Routing 统计收集 + 精准预加载 ⭐⭐⭐ ✅ IMPLEMENTED (05-19)

**Expected**: warmup hit rate 提升 10-15%
**Risk**: 低
**Effort**: 2-3 天
**Code**: `src/models/expert_stream.zig`, `src/engine/engine_loop.zig`
**Status**: ✅ 已实现 routing 统计 + 扩展 warmup

**实现**:
- 新增 `RoutingLayerStats` struct（per-layer [256]u64 频率数组）
- `streamingForward()` 每次 routing 后自动记录 expert activation
- 新增 `getHotExperts(layer, top_n, buf)` 返回热门 expert IDs
- 新增 `getRoutingStats()` 返回全量统计
- `warmupExpertCache()` 第一阶段后调用 `warmupFromRoutingStats()`
- 扩展 warmup 增加 3 个长 prompt（更高 expert 覆盖率）
- Warmup 后输出 cache 状态日志

**Problem**: 当前 warmup（`engine_loop.zig`）使用 5 个固定 prompt 预热 cache。
预热效果取决于这 5 个 prompt 的 expert 覆盖率——覆盖不到的 expert 在首次真实请求时仍会 miss。

**源码分析** (`engine_loop.zig` warmup section):
- 当前: 5 个硬编码 prompt → `processRequest()` → 自然触发 cache
- 覆盖率: ~1290 expert slots / 11008 total = 11.7%
- 问题: 固定 prompt 无法覆盖真实用户的 routing 分布

**Fix (两阶段)**:

Phase A — 添加 routing 统计收集:
```zig
// 在 streamingForward 的 routing 完成后记录
var routing_freq: [43][256]u64 = undefined; // per-layer frequency
for (indices_data) |eid| {
    routing_freq[layer_idx][eid] += 1;
}
// 导出: dmlx bench --routing-stats --output routing_stats.json
```

Phase B — 基于统计的精准 warmup:
```zig
// 加载 routing_stats.json，预热每层 top-N experts
// 替代当前的 5 固定 prompt warmup
for (0..43) |layer| {
    for (hot_experts[layer][0..target_count]) |eid| {
        cache.prefetch(layer, eid);
    }
}
```

**注意**: 在 stream 模式下 router 不受 `cache_bias` 约束（-1000 bias 仅在 preload 模式生效）。
Stream 模式下所有 256 experts 均可被路由到，因此精准预热确实有价值。

**Verification**:
1. 比较 warmup 后首次请求的 cache hit rate
2. 导出 routing 统计验证分布集中度

---

### P1.1: 异步 Expert 预取（Cross-Layer Gate Prediction）⭐⭐⭐⭐ ❌ VERIFIED INEFFECTIVE (05-20)

**Expected**: ITL -8~12ms (隐藏 60-70% I/O 延迟); 社区数据暗示可能 3-16x 改善
**Risk**: 低（Fate 论文验证了 93%+ 预测准确率，MoE-Infinity 开源实现可参考）
**Effort**: 5-7 天
**Code**: `src/models/layer_prefetcher.zig`, `src/models/expert_stream.zig` (Fate prediction)

**实现状态 (2026-05-20)**:
- ✅ Raw pread prefetcher: worker 线程只做 POSIX pread (线程安全, 不涉及 MLX)
- ✅ flushToCache(): 主线程将 raw buffer 包装为 MLX Array 并插入 cache
- ✅ Fate Timing C (gate-input): `flat_x @ next_gate_weight^T → top-k` → **64%** accuracy
- ✅ Fate Timing B (layer-output): `hidden_N[HC_slot0] @ next_gate_weight^T → top-k` → **42%** accuracy
- ✅ Fate Timing B (layer-output + mean): `mean(hidden_N, HC) @ gate^T` → **63%** accuracy
- ❌ **mHC 模型 Timing B 效果差**: layer output 经过 expandToMHC 后与 gate input 表征空间不同
- ❌ **背景线程 prefetch: 2.5x 退化 (458s vs 212s baseline)**
- ❌ **主线程 pre-warming: 2.3x 退化 (497s vs 212s baseline)**
- ❌ **matmul overhead**: triggerFatePrediction 的 eval() 每层增加 ~1.5ms, 43层×100token = +6.5s
- ❌ **准确率 63-64% << 论文 93%**: mHC 变换 + 256-expert 模型层间信号衰减太大

**实测结论 (2026-05-20 ReleaseFast)**:

| 配置 | 100-token 延迟 | Server ITL | 说明 |
|------|---------------|------------|------|
| Baseline (no prefetch, no mmap) | **212s** | 40ms | Phase 1 优化后基线 |
| + Fate Timing C only (measure) | **225s** | 40-55ms | 预测开销 ~6%, 无改善 |
| + Fate Timing B (HC slot0, no prefetch) | **358s** | - | matmul overhead, 42% accuracy |
| + Fate Timing B (HC mean, prefetcher) | **432s** | - | SSD contention + LFU eviction |
| + Fate + background prefetcher (Timing C) | **458s** | 94-125ms | 2.2x 退化, SSD 竞争 |
| + Fate + main-thread pre-warm | **497s** | 106-125ms | 2.3x 退化, I/O 翻倍 |
| + Prefetcher (naive, no Fate) | **463s** | 120ms | 旧方案, SSD 竞争 |
| + mmap_pool connected to reader | **crash** | — | OOM: zero-copy concat exhausts memory |

**Fate 预测准确率分析**:
- Timing C (flat_x = MoE input): **64.2%** (17967/27972)
- Timing B HC slot0 (raw layer output): **42%** (维度匹配但表征不同)
- Timing B HC mean (layer output mean over HC): **63%** (等效于 Timing C)
- 论文 (DeepSeek V2 Lite, 16 experts): **93.03%**
- 论文 (Qwen3-30B): **94.69%**
- **差距根因**: DeepSeek V4 Flash 使用 mHC (Multi-Head Clustering) 架构,
  layer output 经过 `expandToMHC` → broadcast 后与 gate input (经 `hc_ffn.pre()` mixing)
  处于不同表征空间。256 experts 进一步降低准确率。

**预热失败原因分析**:
1. **64% 准确率 → 36% 浪费 I/O**: 预测错误的 experts 白白加载, 增加总 I/O
2. **LFU 缓存驱逐**: 新预热的 entries (freq=1) 被 LFU 立即驱逐, 来不及被使用
3. **I/O 翻倍**: 每层加载当前 + 下一层 = 2x I/O, 净效果为退化

**根因总结**:
- 141GB 模型 on 48GB RAM → kernel buffer cache 已饱和
- 后台 pread 不是"隐藏延迟"，而是"增加 I/O 队列深度"
- 64% 预测准确率不足以让主线程 SSD 访问降低到可忽略水平
- Fate 论文的 93%+ 准确率基于较小模型 (16-64 experts), 不适用于 256-expert 模型

**当前状态**: Fate prediction 完全禁用 (Timing B 在 model forward 中注释掉, prefetcher=null)。
代码保留 triggerFatePrediction() 和 accuracy measurement 基础设施供未来使用。

**Do Not Retry**:
- 背景线程 pread (SSD 竞争无法避免 on 48GB/141GB system)
- 主线程 pre-warming (I/O 翻倍 + LFU 驱逐)
- Fate prediction on mHC model (Timing B 与 gate input 表征空间不匹配)
- Fate Timing C on 256-expert model (63% 准确率不足以抵消 prefetch 开销)
- 每层 matmul eval() 开销 (43层×1.5ms = 65ms/token 净增)

#### 学术验证（社区已有数据）

| 论文 | 核心发现 | 对 dmlx 的意义 |
|------|---------|---------------|
| **Fate** (arXiv 2502.12224) | Layer N 的 gate input 预测 Layer N+1 expert，准确率 93-99% | P1.1 的 routing 预测策略 |
| **MoE-Infinity** (arXiv 2401.14361) | Trace-guided prefetch + caching → 3.1-16.7x latency 改善 | 验证了 prefetch 的巨大收益 |
| **MoE Lens** (OpenReview GS4WXncwSF) | DeepSeekMoE: 少数 expert 处理 >50% routing | P0.2 精准预加载的理论基础 |
| **MoE-Sieve** (arXiv 2603.24044) | Per-layer routing 高度偏斜，25% expert 处理大部分 token | 验证了 routing 集中度假设 |
| **ExpertFlow** (arXiv 2510.26730) | Hybrid cross-layer prediction: pregating + intermediate states | Fate 的增强版方案 |
| **MoE-Beyond** (arXiv 2508.17137) | Learning-based predictor: 97.5% accuracy, cache hit 17%→72% | 更高级的预测方案（备选） |

**现成资源**:
- 开源实现: https://github.com/EfficientMoE/MoE-Infinity (Python/PyTorch)
- Expert trace 数据集: https://huggingface.co/datasets/core12345/MoE_expert_selection_trace
  - 包含 DeepSeek-R1 (671B) 的完整 per-layer per-token expert activation trace
  - 可直接用于离线验证相邻层 overlap 和 routing 分布

#### 实现方式（基于 Fate 论文）

**Fate 核心算法**:
```
1. Layer N 的 attention 输出 hidden_state: [1, D]
2. 用 Layer N+1 的 gate_weight: [D, num_experts] 做 matmul
   → predicted_scores: [1, num_experts]
3. 取 top-k → 预测的 expert IDs
4. 后台线程 pread 这些 expert weights 到 raw buffer
5. Layer N+1 实际 routing 时，93%+ 的 expert 已在 buffer 中

关键: gate_weight 是静态模型权重，不需要额外训练。
预测开销: 一次 [1,4096]×[4096,256] matmul ≈ 0.01ms（可忽略）。
```

**Fate 的 Shallow-Favoring Caching 策略**:
```
- 浅层 expert 更容易预测（routing 更稳定）→ 给浅层更大 cache
- 深层 expert 预测不准时 fallback 到 on-demand load
- 组合效果: 99% expert hit rate
```

**dmlx 适配方案**:

```zig
// Step 1: 在 DSV4Model.forward() 中，attention 完成后立即预测
fn predictNextLayerExperts(
    hidden: Array,           // Layer N attention output
    next_gate_weight: Array, // Layer N+1 的 gate projection weight
    k: usize,               // top-k (6 for DeepSeek V4)
    ctx: EagerContext,
) ![]u32 {
    // 轻量 matmul: [1, D] × [D, 256] → [1, 256]
    const scores = try ops.matmul(ctx, hidden, next_gate_weight);
    defer scores.deinit();
    try scores.eval();
    // top-k selection (CPU-side, 256 elements trivial)
    const scores_data = try scores.dataSlice(f32);
    return topkIndices(scores_data, k);
}

// Step 2: 后台线程只做 pread（线程安全，不涉及 MLX）
fn prefetchWorkerRawIO(self: *LayerPrefetcher) void {
    // pread expert rows into pre-allocated byte buffers
    // NO MLX operations — pure POSIX I/O
    for (self.predicted_experts) |eid| {
        const offset = self.getExpertOffset(self.target_layer, eid);
        const size = self.expert_row_bytes;
        _ = std.posix.pread(self.fd, self.buffer_pool[eid].ptr, size, offset);
    }
    self.state.store(STATE_DONE, .release);
}

// Step 3: 主线程从 raw buffer 构造 MLX Array（需要时）
fn getPreloadedExpert(self: *LayerPrefetcher, layer: usize, eid: u32) ?Array {
    if (self.raw_buffers[layer][eid]) |buf| {
        // 构造 MLX Array（主线程，线程安全）
        return Array.fromData(self.allocator, buf, self.shape, self.dtype);
    }
    return null; // fallback to normal load path
}
```

**与 dmlx 现有架构的兼容性**:
- `LayerPrefetcher` 已有 lock-free 状态机（IDLE→PENDING→DONE→STOPPED），只需改造 worker
- Gate weights 已在 `DSV4Model` 中加载（`ffn_gate_inp` per layer），可直接引用
- `FdPool` 已初始化，`pread` 路径已验证可用
- `streamingForward` 已有 prefetcher 集成点（`pf.prefetch()` / `pf.waitForCompletion()`）

#### Verification Plan

1. **离线验证**（0.5 天）: 下载 HuggingFace trace 数据集，分析 DeepSeek 相邻层 overlap
2. **实现 raw I/O prefetcher**（3 天）: 改造 LayerPrefetcher worker
3. **实现 cross-layer gate prediction**（2 天）: 在 forward 中添加预测逻辑
4. **集成测试**（1 天）: TSAN 验证 + ITL 对比 + 命中率监控
5. **目标**: 预取命中率 > 90%，ITL 改善 > 15%

---

### P1.2: I/O 节流（层间 Yield）⭐⭐

**Expected**: client latency -3~5s
**Risk**: 极低
**Effort**: 0.5 天
**Code**: `src/models/expert_stream.zig` (streamingForward)
**Status**: 🔄 未实现

**实现**: 在 `streamingForward()` 中每 8 层调用 `std.Thread.yield()`，
让 OS 有机会处理积压的 VM 操作，减少 page fault 风暴对 client 延迟的影响。

**Problem**: 43 层串行执行，每层 cache miss 立即触发 page fault。~196 次 miss 形成 page fault 风暴。

**Fix**:
```zig
// 每处理 N 层后，yield 让 OS 处理积压的 VM 操作
if (layer_idx % 8 == 0) {
    std.Thread.yield() catch {};
}
```

**Verification**: 对比 yield 前后的 client 延迟，确保 server tok/s 不退化。

---

### P1.3: mmap/pread 混合模式 ⭐⭐

**Expected**: 减少 VM 压力，client ~54s/10tok → ~35s/10tok（估算）
**Risk**: 中高（已验证 pread 完全替代无效 -44% tok/s）
**Effort**: 3-5 天
**Code**: `src/models/expert_stream.zig`

**Problem**: 当前全部使用 mmap。Expert 随机访问导致 VM 映射压力，连带影响 backbone page cache。

**已验证数据**:
| 模式 | Server tok/s | Client latency |
|------|-------------|----------------|
| 纯 mmap | 14-15 | 27-30s |
| 纯 pread | 4.9 | 39.8s (with warmup) |

**Fix**: Backbone 保持 mmap（利用 readahead），Expert miss 改用 pread（不产生 VM 映射）。

**Decision Gate**: 如果混合模式的 client 延迟改善 < 5s，则不值得 server tok/s 的损失。

**源码约束** (`expert_stream.zig:68-93`):
- `ExpertStreamProvider` 已支持 `FdPool`（pread）和 `MmapPool` 双路径
- 改造只需在 cache miss 时选择 FdPool 而非 MmapPool
- 已有 `PartialTensorReader` 支持两种读取方式

---

### P1.4: Cache-Aware Routing Optimization ❌ 已验证不可行

**Expected**: hit rate 24% → 32-38%，性能净正收益
**Actual**: 三种方案均无法同时满足正确性与效率
**Effort**: 3 天（05-20 ~ 05-22）
**Code**: `src/models/deepseek_v4.zig` (DSV4Gate.forward), `src/models/expert_stream.zig`
**Status**: ❌ 已关闭 — MoE routing 对任何 expert 替换高度敏感，代码已回退

---

#### 迭代历史

**P1.4a: Additive Bias (05-20, ❌ 废弃)**

| Bias | Cache Hit Rate | 质量 | 问题 |
|------|----------------|------|------|
| 0.0 (baseline) | 27% | 7/7 PASS | — |
| 0.5 | 61% | 2/7 PASS | 低分 expert 被拉入 top-6 |
| 1.0 | 88% | ❌ 退化 | Routing 锁定 |

**失败原因**：绝对 bias 对 ALL cached experts 提升相同值。score=0.5 的垃圾 expert
和 score=1.3 的优质 expert 获得相同 +0.5 提升 → 大量低分 expert 涌入 top-6。

**P1.4b: Multiplicative Bias + Warmup Bypass (05-21, ❌ 废弃)**

| Factor | Cache Hit Rate | 质量 | Server tok/s | 问题 |
|--------|----------------|------|-------------|------|
| 1.10 + bypass | 24.4% | 7/7 PASS | 15.87 (-11%) | 计算开销 > 收益 |
| 1.50 + bypass | 27.2% | 7/7 PASS | 13.22 (-26%) | 同上，更严重 |

**失败原因**：
1. 每层每步 `multiply + add + expandDims` = ~13ms/token 额外开销
2. Multiplicative 太"保守"——只影响恰好在 top-6 边缘的 expert，hit rate 改善微不足道
3. 无 warmup bypass 时，warmup 期间 bias → cache feedback loop → 正确性崩溃
4. **净效果为负**：微小的 hit rate 提升不足以抵消计算开销

**关键发现**：
- Warmup bypass 是正确性必需条件（自然填充 cache 后再启用 bias）
- 但 score-space bias（无论 additive/multiplicative）的效率/正确性矛盾无法调和
- 需要完全不同的机制：**不修改 scoring，而是 post-selection swap**

---

#### P1.4c: Post-Selection Swap（05-21~22, ❌ 废弃）

**核心思想**：正常 top-6 选择（零开销），然后检查是否有"近似等价"的 cached expert
可以替换 non-cached expert。只做 score 差距极小的替换。

**实验结果**：

| Threshold | Max Swap/Step | 质量 | 问题 |
|-----------|---------------|------|------|
| 0.05 | 无限制 | 5/7 PASS | eval() 破坏 MLX lazy graph |
| 0.02 | 无限制 | 3/7 PASS | 同上 |
| 0.05 | 1 | 4/7 PASS | 同上 |
| 0.005 | 1 | 4/7 PASS | 同上 |
| **no-op stub** | — | **7/7 PASS** | **证明问题在 eval() 本身** |

**根本失败原因**：

1. **MLX Lazy Evaluation 冲突**：swap 需要在 forward pass 中间调用 `eval()` 读取
   indices 和 scores 到 CPU。这会强制同步整个计算图，破坏 Metal GPU 的 pipeline
   调度，导致数值结果不确定。no-op stub（只有函数调用壳，不做任何 eval/swap）
   立即恢复 7/7 PASS，证明 eval() 是唯一致命点。

2. **缺少 GPU-native 实现路径**：MLX-Zig 绑定只提供 `where`、comparison、arithmetic
   ops，没有 `take`/`gather`（按 indices 索引 array）。无法用纯 GPU ops 表达
   "查询 cache → 选择候选 → 条件替换" 逻辑。

3. **MoE Routing 级联敏感性**：即使只替换 1 expert/step（score gap <0.5%），
   60 层 × 30 token = 1800 次潜在路由扰动。每次 swap 改变 hidden state →
   下层 routing 偏移 → 级联累积 → 生成轨迹完全偏离。

---

#### P1.4 最终结论

**三种方案全部失败**：

| 方案 | 正确性 | 性能 | 失败根因 |
|------|--------|------|----------|
| Additive Bias | 2/7 PASS | — | 低分 expert 涌入 top-6 |
| Multiplicative Bias | 7/7 PASS | -11~26% | 计算开销 > cache 收益 |
| Post-Selection Swap | 3-5/7 PASS | — | eval() 破坏 MLX 计算图 |

**核心矛盾**：MoE expert 选择对模型输出的贡献是非线性、不可替代的。
"score 接近"不等于"功能等价"——两个 expert 即使 gate score 只差 0.5%，
它们编码的知识/功能可能完全不同。Cache-aware routing 的假设前提
（score 接近 = 可互换）在 DeepSeek V4 的 256-expert MoE 中不成立。

**Baseline 保持不变**：7/7 PASS, 17.8 tok/s, 23.7% hit rate (commit 2eb821fc)

**推荐后续优化方向**（不修改 routing）：
- P1.5 DyMoE: skip 低重要性 miss（不替换，直接跳过）
- 增大 cache（6GB → 10GB，如果 VM 允许）
- I/O 并行化（多线程 expert 加载）
- Expert 更高压缩比（int4 → int2 或 fp4）

---

### P1.5: DyMoE — Skip 边缘 Expert（Importance-Aware Loading）⭐⭐⭐⭐

**Expected**: cache miss I/O 减少 30-50%，client 吞吐 +30-50%
**Risk**: 低（只跳过 score 最低的 miss expert，质量损失可控）
**Effort**: 1-2 天
**Code**: `src/models/expert_stream.zig` (streamingForward)
**Status**: 🔄 下一优先级

**论文依据**：DyMoE (arXiv 2603.19172) — importance-aware prioritization，
只加载"重要"的 cache miss expert，跳过"边缘"expert。

**核心思路**：当 top-6 中有 expert cache miss 时，检查其 routing score：
- Score 高（重要）→ 必须从 SSD 加载
- Score 低（边缘，接近 top-6 门槛）→ 跳过，权重自动重新归一化到剩余 expert

**为什么这在 dmlx 上可行**：
1. DeepSeek V4 已有权重归一化逻辑（`sum_weights → normed`），跳过 expert 后自动补偿
2. Shared expert 始终参与计算（不受 routing 影响），提供 baseline 质量保障
3. 只跳过 cache miss 的边缘 expert → cache hit 的 expert 不受影响
4. 不修改 routing 决策，只跳过加载（vs P1.4 尝试修改 routing 失败）

**预期效果**：
```
当前: 258 accesses × 76% miss = 196 page faults/token
+ P1.5 DyMoE (skip 50% low-score miss): 196 × 50% = ~98 page faults/token

OS 开销: 0.22s × (98/196) = 0.11s/token
预期: 吞吐 +30-50%
```

**实现方式**：

```zig
// 在 streamingForward 中，routing 完成后、expert 加载前:
// indices_data: [N, 6] — 选中的 expert IDs
// scores_data: 对应的 routing scores（归一化前）

// Step 1: 分离 cached 和 uncached experts
var experts_to_load = ArrayList(u32).init(allocator);
var experts_to_skip = ArrayList(u32).init(allocator);

for (unique_ids) |eid| {
    const key = CacheKey{ .layer_idx = lx, .tensor_name_hash = gate_hash, .expert_id = eid };
    if (cache_inst.get(key) != null) {
        // Cache hit — 正常使用
        try experts_to_load.append(eid);
    } else {
        // Cache miss — 检查 importance
        const score = getScoreForExpert(scores_data, eid);
        if (score > importance_threshold) {
            // 重要 expert — 必须加载
            try experts_to_load.append(eid);
        } else {
            // 边缘 expert — 跳过
            try experts_to_skip.append(eid);
        }
    }
}

// Step 2: 只加载 experts_to_load（不含 skipped）
// Step 3: 在 remap 中将 skipped experts 映射到 0 权重
// Step 4: 权重自动重新归一化（已有逻辑）
```

**importance_threshold 的确定**：
- 方案 A: 固定阈值（如 score < top6_min_score × 0.9 → 跳过）
- 方案 B: 动态阈值（如果 miss 数量 > 3，跳过 score 最低的 miss expert）
- 方案 C: 只跳过 score 排名最后 1-2 个的 miss expert（最保守）

**质量保障**：
- Shared expert 始终参与（不受 skip 影响）→ 提供 baseline
- 权重归一化自动补偿 → 剩余 expert 权重增大
- 只跳过 cache miss 的 → cache hit 的 expert 不受影响
- 最坏情况：跳过 1-2 个 expert，质量损失 ~10-15%（单层）
- 43 层中只有部分层会触发 skip → 整体质量损失更小

**Verification**:
1. 7-prompt correctness test（不同 threshold 值）
2. 对比 skip 前后的 cache miss I/O 量
3. 对比 client 延迟
4. 监控 skip 频率（每 token 平均跳过多少 expert）

---

### P2: Reduce HTTP Cold Start ⭐⭐

**Expected**: 75s → 40-50s
**Risk**: 低-中
**Effort**: 2-3 天

**Options**:
1. **Smarter warmup**: 基于 routing 统计，使用最大化 expert 覆盖率的 warmup（与 P0.2 协同）
2. **Backbone pinning**: `mlock()` 锁定 backbone weight pages 防止 OS 换出
3. **渐进式 warmup**: 先接受请求（快速响应），后台继续 warmup cache

---

## 5. Abandoned Priorities (Do Not Retry)

| 方案 | 原因 | 验证来源 |
|------|------|---------|
| **Fate Cross-Layer Prediction** | **mHC 架构表征空间不匹配，64% 准确率不足，prefetch 反而退化 +104%** | **实测 05-20** |
| **Background pread prefetch** | **SSD 竞争 + LFU 驱逐，2.2x 退化** | **实测 05-20** |
| **Main-thread pre-warming** | **I/O 翻倍 + LFU 驱逐，2.3x 退化** | **实测 05-20** |
| **BuddyMoE Expert Substitution** | DeepSeek V4 的 256 experts 高度专门化（论文目标即 "ultimate specialization"），冗余性低，buddy 质量损失不可接受 | 分析 05-21 |
| **SliceMoE Bit-Sliced Cache** | 与"不做 expert 压缩"决定冲突，需离线预处理 + 双精度 dequant 路径 | 分析 05-21 |
| **NPUMoE (Apple NPU offload)** | MLX 不支持 NPU dispatch，需 Core ML/ANE 直接调用，实现成本极高 | 分析 05-21 |
| Expert 压缩 (2-bit) | 已决定不再投入 | — |
| MLX Compile Fusion | Stream 模式 I/O-bound，仅 +6% | inference-optimization-review.md |
| MTP (Multi-Token Prediction) | MTP head 含完整 MoE 层，加剧 page thrashing；I/O-bound 场景无效 | 分析 05-19 |
| PLD Speculative Decoding | N-gram match rate too low, -15% tok/s | bench 7ea49aa |
| LRU Cache Eviction | 258 new entries/token → cache thrashing, -36% tok/s | bench |
| P1.1 eval skip (stream mode) | Stream 模式依赖 eval() 触发 page-in, -40% ITL | bench |
| Zero-copy mmap arrays | safetensors offset 非 page-aligned, <7% gain | PERF_PLAN |
| pread 完全替代 mmap | 丧失 OS readahead, -44% tok/s | pread-expert-loading.md |
| OS threads for HTTP | 0% improvement, 证明非 fiber 调度问题 | socket-write-latency.md |
| posix write bypass | 0% improvement, 证明 write 不是瓶颈 | socket-write-latency.md |
| Cache 扩大到 10GB+ | 挤占 backbone page cache, tok/s -36% | bench f970d9f |
| Cache 扩大到 20GB | 触发系统 swap | PERF_PLAN |

---

## 6. Verified Findings

### What Works

| Optimization | Impact | Status |
|-------------|--------|--------|
| **SMELT 15% + 10GB cache + mlock** | +24% tok/s, -46% startup (vs 20%+6GB) | ✅ Deployed |
| Expert cache 10GB + mlock (was 6GB) | +24% tok/s, mlock prevents backbone eviction | ✅ Deployed |
| Cache warmup before accept | -85% first-request misses | ✅ Deployed |
| mmap for expert loading | 2x tok/s vs pread | ✅ Kept |
| ReleaseFast build | ~2x vs Debug | ✅ Always |
| Expert deduplication (P4.1) | 30-50% fewer expert loads during prefill | ✅ Deployed |
| Prefix cache | Skip repeated prefill | ✅ Deployed |
| FP8 KV storage | 50% KV memory saving | ✅ Deployed |
| ~~Cache-Aware Routing Bias (P1.4)~~ | ~~-55% latency~~ | **❌ 不可行** |

### Hardware Limits (Cannot Fix in Code)

| Issue | Cause | Status |
|-------|-------|--------|
| ~54s/10-token HTTP warm latency | 141GB model on 48GB RAM → VM pressure → socket write blocked | 物理限制 |
| ~190s HTTP cold start | Backbone page-in + cache cold | 物理限制 |
| Cache hit rate ceiling ~24% | 256 experts × 43 layers, routing 不可预测 | 架构限制 |

### What Doesn't Work (Do Not Retry)

| Optimization | Result | Date |
|-------------|--------|------|
| Cross-tensor madvise prefetch | -50% server tok/s（madvise CPU 开销 > parallel page-in 收益）| 05-22 |
| Cache-Aware Routing Bias (P1.4) | 3 approaches all failed（eval 中断 MLX lazy graph）| 05-22 |
| Eval skip (every 2 layers) | -5%（2-layer lazy graph 增加内存压力）| 05-22 |
| Async prefetcher (pread) | -2.5x（与主线程竞争 SSD 带宽）| 05-20 |
| mmap concat reader | OOM crash | 05-20 |
| PLD speculative (ngram=3) | -15% tok/s | 05-19 |
| pread 完全替代 mmap | -44% tok/s | 05-16 |
| OS thread 替代 fiber | 无效 | 05-16 |
| Zero-copy mmap | 只省 7%，非瓶颈 | 05-16 |

### 关于 "27-30s/100-token (warm)" 的说明

此数据来自 05-16 的旧配置（commit 0243aeb, SMELT 20% + 6GB cache），在以下条件下测得：
1. Server 已运行多次请求（5+ rounds），expert cache 高度 warm
2. 重复/相似的 prompt，routing pattern 稳定，cache hit rate 上升到 40-50%
3. 此时 page fault 减少约 50%，macOS VM 压力显著缓解，socket write 延迟下降

**Config C 为何没有达到相同的 client 延迟**：
- mlock 保护了 backbone（server ITL 改善），但 expert cache miss 仍 76%
- 10GB cache 容量够大，但 MoE routing 分散度极高（256 experts × 43 layers）
- Page fault 总量不变 → VM 压力不变 → client 延迟不变
- 唯一改善路径：**减少 cache miss 绝对数量**（非提高 cache 命中路径速度）

---

## 7. Performance Evolution

| Commit | Date | Config | Server tok/s | Client Latency | Key Change |
|--------|------|--------|-------------|----------------|------------|
| dff154d | 05-05 | SMELT 10%, 4GB cache | 10.5 | — | Initial baseline |
| 538f930 | 05-09 | SMELT 10%, 4GB cache | 14.5 | — | Tuning |
| 6d339e0 | 05-14 | SMELT 10%, 10GB cache | 10.7 | 38-40s | Cache too large |
| f970d9f | 05-16 | SMELT 10%, 6GB cache | 12-13 | 31-34s | Optimal cache size |
| latest | 05-16 | SMELT 20%, 6GB cache | 14-15 | 27-30s | Optimal SMELT ratio |
| **0243aeb** | **05-16** | **SMELT 20%, 6GB cache** | **17.8** | **27-30s** | **Previous best** |
| **P1.4** | **05-22** | **SMELT 20%, 6GB** | **17.8** | **—** | **❌ Cache-Aware Routing 不可行（三种方案全部失败）** |
| **Config C** | **05-22** | **SMELT 15%, 10GB, mlock** | **22.0** | **~190s (cold), ~54s/10tok (warm)** | **✅ 新 baseline (+24% tok/s, -46% startup)** |
| **Hash Prefetch** | **05-22** | **Config C + hash prefetch** | **—** | **~154s (cold), ~32s/10tok (warm)** | **❌ Config 依赖：仅 Config C 有效，Config A 下负面** |

---

## 8. Target Milestones

**阶段性目标：Warm client 吞吐改善（P1.5 + I/O 优化）**

| Milestone | Target | Status | Approach | 置信度 |
|-----------|--------|--------|----------|--------|
| ~~M5: Cache hit rate~~ | ~~50-60%~~ | ❌ | ~~P1.4 Cache-Aware Routing~~ 不可行 | — |
| ~~M6: Client latency (warm)~~ | ~~< 250s/100tok~~ | ❌ | ~~P1.4~~ routing 修改导致质量退化 | — |
| **M6.5: Hash Prefetch** | **< 40s/10tok** | ❌ | Hash Routing 确定性预加载 — 最优配置下无效 | — |
| **M7: Client throughput** | **6-7 tok/s (15s)** | 🔄 Next | P1.5 DyMoE skip 边缘 expert | 中-高 |
| **M8: Client throughput (stretch)** | **8-12 tok/s** | 🔄 Future | P1.5 + I/O 并行 + 更大 cache | 中 |
| ~~M9: Fate prefetch~~ | ~~15+ tok/s~~ | ❌ | mHC 模型不适用 | — |

**P1.4 最终实测结论（05-22 关闭）**：
- Additive bias=0.5: hit rate 61% 但 2/7 PASS（正确性不可接受）
- Multiplicative bias: 7/7 PASS 但 -11~26% 性能（开销 > 收益）
- Post-Selection Swap: eval() 破坏 MLX lazy evaluation（根本不兼容）
- **结论**：MoE routing 对 expert 替换高度敏感，无法在保持正确性的同时获益
- **Baseline 不变**：7/7 PASS, 17.8 tok/s, 23.7% hit rate (commit 2eb821fc)

**P1.5 预期效果**：
```
当前 baseline: hit rate 23.7%, 17.8 tok/s
每 token ~258 page faults (76.3% miss × 256 expert + overhead)
P1.5 skip 50% low-score miss: ~129 page faults/token
预期: 吞吐提升 30-50%

注: 不依赖 P1.4（已废弃），独立生效。
```

---

## 8.5 DeepSeek V4 论文优化方案分析 (2026-05-22)

> 来源：`DeepSeek_V4.pdf` §3 General Infrastructures

### 论文方案 vs dmlx 已验证结论

| 论文方案 | 论文场景 | dmlx 适用性 | 已验证？ | 结论 |
|---------|---------|------------|---------|------|
| **Expert Wave Pipeline** (§3.1) | GPU EP 多节点通信 | ⚠️ 需改造 | 部分（async prefetch 失败）| 见下方分析 |
| **Communication-Computation Overlap** (§3.1) | Dispatch/Combine + Linear overlap | ⚠️ 需改造 | 是（pread async 失败）| SSD I/O ≠ 网络通信 |
| **Hash Routing 前 3 层** (§2.1) | 确定性路由 | ✅ **直接可用** | ✅ 已验证 | **❌ 最优配置下负面，已放弃** |
| **On-Disk KV Cache** (§3.6.2) | 共享 prefix 避免重 prefill | ✅ 已实现 | ✅ Prefix Cache | 已部署 |
| **KV Cache 压缩** (§3.6.1) | CSA/HCA 压缩 KV | ⚠️ 间接相关 | 否 | 减少 KV 内存可为 expert cache 让出空间 |
| **FP4 QAT** (§3.4) | expert 权重 MXFP4 | ✅ 已实现 | ✅ | 模型已是 4-bit |
| **SwiGLU → 轻量激活** (§3.1 观察) | 减少 expert compute | ❌ 不可行 | — | 需重训模型 |
| **MegaMoE Kernel Fusion** (§3.1) | GPU CUDA mega-kernel | ❌ 不适用 | — | Metal/MLX 无法复制 |
| **TileLang 自动优化** (§3.2) | DSL → 高效 kernel | ❌ 不适用 | — | Zig + MLX C API |
| **Batch-Invariant Kernels** (§3.3) | 训练复现性 | ❌ 不相关 | — | 推理场景不需要 |

### 详细分析

#### 1. Expert Wave Pipeline — ⚠️ 需重新设计（非直接 async prefetch）

**论文原理**（§3.1 p.15-16）：
- 将 256 experts 按 "wave" 分组（如 6 个一组）
- Wave N 计算时，Wave N+1 同时做通信（在 GPU EP 中是 all-to-all）
- 理论加速 1.92x（vs naive 顺序执行）

**dmlx 映射**：
- "通信" = 从 SSD 读取 expert weights（mmap page fault）
- "计算" = MLX matmul（GPU compute）
- 当前是 `load gate → compute gate → load up → compute up → load down → compute down`（全串行）

**为什么之前的 async prefetch 失败**：
- pread async: 后台线程和主线程竞争同一个 SSD 通道 → -2.5x
- madvise prefetch: CPU 遍历开销 > page-in 收益 → -50%
- mmap concat: OOM

**Wave Pipeline 的不同之处**：
- 不是"提前读取"（prefetch），而是**重排 compute 顺序**使得 I/O 自然被 compute 隐藏
- 关键约束：macOS 上 SSD 和 GPU memory 共享同一总线，不能真正并行
- **结论**：在 Apple Silicon UMA 架构上，I/O 和 compute 无法真正 overlap（同一内存控制器）
- **状态**：❌ 不适用于 Apple Silicon（GPU ↔ SSD 共享带宽）

#### 2. Hash Routing 确定性预加载 — ❌ 已验证，放弃

**论文原理**（§2.1 p.7，§4.2.1 p.25）：
- DeepSeek-V4-Flash 前 3 层使用 Hash routing（不是 learned routing）
- Hash routing: `expert_id = tid2eid[token_id]` 查表（[129280, 6]）
- Expert 选择 **完全由 input token ID 决定**，与 hidden state 无关

**实现与测试**（2026-05-22, 已 revert）：
- 实现了完整的 prefetch pipeline（tid2eid CPU cache → prefetchForTokens → engine 调用）
- Config C (15%+10GB+mlock) 下看似有效：warm 10-tok 54s→32s (-41%)
- 但切换到最优配置 Config A (20%+6GB, no mlock) 后效果为负：

| 配置 | 无 prefetch | 有 prefetch | 变化 |
|------|-------------|-------------|------|
| Config A Cold 30-tok | 113s | 157s | **-39%（更差）** |
| Config A Warm 100-tok | 149s | 185s | **-24%（更差）** |
| Config A Warm 10-tok | 30.6s | 33.9s | -10% |

**根因分析**：
1. Hash routing 只覆盖 3/43 层（7%），对整体 page fault 影响极小
2. `loadExpertSlicesCached` 调用本身有开销（cache lookup + LFU 逻辑）
3. Cold 时 prefetch 同步触发大量 page fault，加剧 VM thrashing
4. Config A 的 25GB OS headroom 下，OS page cache 已能高效复用，手动 prefetch 画蛇添足

**结论**：在最优配置下收益为负，代码已 revert。仅在极端内存受限配置 (headroom < 16GB) 下可能有正向效果。

#### 3. KV Cache 压缩让出内存 — ⚠️ 间接，低优先级

**当前 KV cache**: ~500MB（FP8）
**如果压缩到 1/4**: 省 ~375MB → expert cache 可多 375MB → 多缓存 ~20 experts

在当前 10GB cache 下，多 375MB 是 3.75% 增量，效果有限。**低优先级**。

### 新增可行方案（不来自论文，但受论文启发）

#### 4. Reduce MoE layers（DyMoE Skip）— 已在 roadmap P1.5

论文提到 "6 experts will be activated for each token"。如果某些 miss expert 的 score 远低于
top-1，跳过它们 = 减少 page fault。这就是 P1.5 DyMoE 的思路。

#### 5. 重复 Prompt 的 Expert 路由复用

论文的 On-Disk KV Cache 避免重复 prefill。类似思路：
- 如果用户发送相似 prompt，routing pattern 会很相似
- 可以缓存 routing decisions（哪些 layer 选了哪些 expert）
- 下次相同/相似 prompt 来时，直接预加载这些 experts

**这解释了为什么旧配置在多次请求后达到 27-30s** — routing pattern 稳定后 cache hit 自然上升。

### 优先级排序（2026-05-22 更新）

```
❌ 已验证无效：Hash Routing 确定性预加载（3 层）
  - 仅 Config C (15%+10GB+mlock) 下看似有效（headroom 极小时）
  - 最优配置 (20%+6GB) 下为负面（cold -39%, warm -24%）
  - 代码已 revert

❌ 已验证无效：Cross-tensor madvise prefetch
  - A/B 测试证明 -50% 性能退化

第 1 优先：P1.5 DyMoE Skip 边缘 Expert
  - 中等风险（可能影响输出质量）
  - 预期：-30-50% page fault

第 2 优先：Routing Pattern 缓存 + 预加载
  - 利用"重复 prompt → 重复 routing"的规律
  - 多次请求后 cache 自然 warm 的原因

不做：
  - Expert Wave Pipeline（Apple Silicon UMA 无法并行 I/O+compute）
  - MegaMoE Kernel（CUDA only）
  - SwiGLU 替换（需重训）
  - KV 压缩（增量太小）
```

---

## 9. Implementation Plan

```
Phase 1 — 已完成:
├── P1.4 Cache-Aware Routing Bias ✅ (05-20)
│   ├── bias=0.5 最优，hit rate 24%→61%，延迟 456s→207s (-55%)
│   └── 分离 original_scores / scores_for_choice 避免权重污染
├── Fate Cross-Layer Prediction ❌ 已验证无效 (05-20)
│   └── mHC 表征空间不匹配，64% 准确率，所有 prefetch 变体退化
└── Prefix Cache ✅
    └── 跳过重复 prompt 的 prefill

Phase 2 — 下一步 (P1.5 DyMoE):
├── Step 1: P1.5 DyMoE Skip 边缘 Expert [1-2 天] ⭐ 最高优先级
│   ├── 在 streamingForward 中，对 cache miss expert 检查 score
│   ├── Score < threshold → 跳过加载，权重归一化到剩余 expert
│   ├── 测试 threshold 值（top6_min × 0.85, 0.9, 0.95）
│   ├── 7-prompt correctness test 验证质量
│   └── 对比 skip 前后的 page fault 数量和 client 延迟
├── Step 2: Benchmark 验证 P1.4 + P1.5 组合效果 [0.5 天]
│   └── 跑 run_benchmark.sh，记录 hit rate + skip rate + latency
└── Step 3: P0.1 层级分区 Cache [2-3 天]（如果 P1.5 效果不足）
    ├── 新增 LayerPartitionedCache（43 个独立 LFU）
    └── 验证是否进一步提升 hit rate

Phase 3 — 辅助优化:
├── P0.2 Routing 统计 + 精准 warmup [2-3 天]
├── P1.2 I/O 节流 [0.5 天]
├── P2 Cold Start 优化 [2-3 天]
└── P1.3 mmap/pread 混合模式 [3-5 天]（Decision Gate）

已放弃:
├── Fate Cross-Layer Prediction ❌ (mHC 不兼容)
├── Background pread prefetch ❌ (SSD 竞争)
├── BuddyMoE Expert Substitution ❌ (DeepSeek 256 expert 高度专门化，冗余性低)
├── SliceMoE Bit-Sliced Cache ❌ (与"不做 expert 压缩"决定冲突)
└── NPUMoE ❌ (MLX 不支持 NPU dispatch)
```

---

## 10. Expected Outcomes

### Prefetch 方案对比分析

基于学术论文验证的数据，三种 prefetch 方案的预期 client 吞吐：

| 方案 | 核心机制 | 预测准确率 | Client tok/s (保守) | Client tok/s (乐观) | 实现难度 | 推荐 |
|------|---------|-----------|-------------------|-------------------|---------|------|
| **Fate** | Layer N gate input 预测 Layer N+1 expert | 93%+ | **12** | **15** | 低（5-7天） | ⭐⭐⭐⭐⭐ |
| **MoE-Infinity** | Trace-guided sparsity pattern | 70-80% (token-level) | **10-11** | **12-14** | 中（1-2周） | ⭐⭐⭐ |
| **MoE-Beyond** | Trained predictor network | 97.5% | **13-14** | **17** | 高（3-4周） | ⭐⭐ |

#### 推导过程

**Warm baseline**: ~54s/10-token（05-22 Config C 实测），推算 ~500s/100-token。
旧配置 (SMELT 20% + 6GB) cache 极度 warm 后曾达到 27-30s/100-token，其中：
- 实际推理: 5.6s（100 × 56ms ITL）
- OS page thrashing: ~22-24s（每 token ~0.22s，来自 ~196 次 page fault）

**Fate 方案推导**:
```
预测准确率 93% → 93% 的 cache miss 被 pread 预取命中（不产生 page fault）
剩余 page fault: 196 × 7% = ~14 次/token
OS 开销: 0.22s × (14/196) = 0.016s/token
ITL 改善: 56ms → 46-50ms（I/O 部分被 compute 隐藏）
Client time: 100 × 50ms + 100 × 16ms = 6.6s → 15 tok/s (乐观)
保守（85% accuracy）: 100 × 50ms + 100 × 34ms = 8.4s → 12 tok/s
```

**MoE-Infinity 方案推导**:
```
Cache 策略改善: hit rate 24% → 40-50%（trace-guided LFU）
Token-level 预测: ~70-80% 准确率
Page fault: 258 × (1-50%) × (1-75%) = ~32 次/token
OS 开销: 0.22s × (32/196) = 0.036s/token
需要 warmup 期收集 trace（前 10-20 token 无预测能力）
Client time: ~8.6-10s → 10-12 tok/s
```

**MoE-Beyond 方案推导**:
```
97.5% 准确率，但需要训练 predictor
Page fault: 196 × 2.5% = ~5 次/token
OS 开销: 0.22s × (5/196) = 0.006s/token
Predictor 推理开销: +0.5-1ms/token
Client time: ~5.6-7s → 14-17 tok/s
但: 训练成本高 + 额外内存占用 + 实现复杂
```

### 推荐路径

**选择 Fate 方案（修正版）**：

⚠️ **实测发现**：直接用 `flat_x_N`（layer N 的 MoE 输入）× `gate_weight_{N+1}` 预测，
准确率仅 **64%**（vs 论文 93%）。原因分析：

1. **Expert 数量差异**：V4 Flash 256 experts vs V2 Lite 16 experts — 预测空间大 16 倍
2. **信号衰减**：`flat_x_N` 到 `flat_x_{N+1}` 之间经过 MoE 输出 + 残差 + attention + norm
3. **预测输入不对**：论文用的是同层 gate input（attention 输出后），不是上一层 MoE 输入

**修正方案**：改用 **layer N 的完整输出**（MoE + 残差之后）做预测：
```zig
// 时机 B: layer N 完成后（包含 MoE + 残差），预测 layer N+1
const hidden_after_layer_n = ...; // 完整 layer N 输出
const predicted = ops.matmul(ctx, hidden_after_layer_n, gate_weight_n1);
// 预期准确率: 75-85%（比 flat_x_N 的 64% 高，但仍低于论文 93%）
```

**修正后的预期**：
- 准确率 75-85%（用 layer N 完整输出）
- page fault: 196 × 20% = ~39 次/token
- OS 开销: 0.22s × (39/196) = 0.044s/token
- client time: 100 × 50ms + 100 × 44ms = 9.4s → **~10.6 tok/s**

**如果 75-85% 仍不够**，备选方案：
- 时机 A（同层预测）：在 attention 输出后立即做 gate projection → ~100% 准确率，但 prefetch 窗口短
- 组合策略：时机 B 预取 top-8（覆盖面广），时机 A 精确确认 top-6（准确率高）

### 综合预期（P0 + P1.4 Cache-Aware Routing Bias）

| 阶段 | 100-token (warm) | Client tok/s | 说明 |
|------|-----------------|--------------|------|
| **当前** | ~54s/10tok (推算 ~500s/100tok) | ~0.2 | Config C baseline（cache miss 76%）|
| **P0 完成后** | 18-22s | 4.5-5.5 | 层级分区 + 精准预加载 + I/O 节流 |
| **P0 + P1.4 (bias=5)** | 14-17s | **6-7** | Cache-Aware Routing Bias（温和） |
| **P0 + P1.4 (bias=10)** | 9-12s | **8-11** | Cache-Aware Routing Bias（强，需质量验证） |
| ~~P0 + Fate（保守 85%）~~ | ~~8-10s~~ | ~~10-12~~ | ❌ mHC 模型不适用，实测 64% 且退化 |

**P1.4 是当前最有潜力的方案**：
1. 零额外 I/O（不像 prefetch 那样增加 SSD 负载）
2. 零额外内存（不像扩大 cache 那样挤占 headroom）
3. 架构原生支持（DeepSeek V4 的 routing 本身就用 bias 控制）
4. 质量可控（从 bias=0 渐进到 bias=10，找最优平衡点）
5. 实现极简（gate routing 后加几行代码）

**Fate 失败后的关键认知**：
- 在 48GB/141GB 系统上，任何增加 I/O 的方案都会退化
- 唯一有效的方向是**从源头减少 cache miss 的发生**
- P1.4 正是这个方向——不是隐藏 I/O 延迟，而是消除 I/O 需求

---

## 11. Memory Budget (48GB Mac)

```
Current allocation (Config C: SMELT 15% + 10GB cache + mlock):
├── macOS + apps:           ~8 GB
├── Backbone (mlocked):     ~6 GB (locked in RAM, never evicted)
├── SMELT preloads (15%):   ~6 GB (38 experts × 43 layers)
├── Expert cache (LFU):    ~10 GB (configured, fills dynamically)
├── KV cache:               ~0.5 GB (FP8)
├── MLX runtime + Metal:    ~2 GB
├── OS page cache headroom: ~15.5 GB ← for mmap expert loading
└── Total:                  ~48 GB

Key insight (05-22):
├── mlock backbone: prevents 6GB eviction that made 10GB cache unsafe
├── SMELT 20%→15%: frees 2GB headroom (preload marginal value drops with larger cache)
├── Cache 6GB→10GB: +67% capacity → hit rate improves significantly
└── Net result: +24% server tok/s, -46% startup time
```

---

## 12. Risk Matrix

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| Layer-partitioned cache 收益不如预期 | 低 | 中 | 收集 routing 统计后动态调整每层预算 |
| Async prefetch data race | 中 | 高 | TSAN 验证 + 严格分离 I/O 和 MLX 操作 |
| mmap/pread 混合模式 tok/s 下降过多 | 中 | 中 | Decision Gate: 改善 < 5s 则回退 |
| Routing 分布不集中（无热点 expert） | 低 | 中 | 先收集统计数据再决定是否投入 |
| mlock() 导致 OOM | 低 | 高 | 只 mlock 最关键的 backbone 部分（~2GB） |

---

## References

### 项目内部文档
- `docs/analysis/inference-optimization-review.md` — 深度分析与修订建议
- `docs/analysis/socket-write-latency.md` — HTTP latency root cause (VM pressure)
- `docs/analysis/pread-expert-loading.md` — mmap vs pread experiments
- `docs/analysis/cache-benchmark-results.md` — 4GB vs 8GB cache data

### 学术论文与社区资源

| 论文/资源 | 链接 | 与 dmlx 的关系 |
|-----------|------|---------------|
| **Fate: Cross-Layer Gate Prediction** | https://arxiv.org/abs/2502.12224 | P1.1 routing 预测的核心方法（93-99% 准确率） |
| **MoE-Infinity** | https://arxiv.org/abs/2401.14361 / https://github.com/EfficientMoE/MoE-Infinity | Trace-guided prefetch 开源实现（3.1-16.7x 改善） |
| **MoE Lens: An Expert Is All You Need** | https://openreview.net/forum?id=GS4WXncwSF | DeepSeekMoE routing 分布分析（少数 expert 处理 >50% token） |
| **MoE-Sieve: Routing-Guided LoRA** | https://arxiv.org/abs/2603.24044 | Per-layer routing 偏斜度验证（25% expert 处理大部分 token） |
| **ExpertFlow** | https://arxiv.org/abs/2510.26730 | Hybrid cross-layer prediction + adaptive scheduling |
| **MoE-Beyond** | https://arxiv.org/abs/2508.17137 | Learning-based predictor（97.5% accuracy, hit 17%→72%） |
| **Expert Selection Trace Dataset** | https://huggingface.co/datasets/core12345/MoE_expert_selection_trace | DeepSeek-R1 per-layer per-token expert activation traces |
| **DuoServe-MoE** | https://arxiv.org/abs/2509.07379 | Dual-phase prefetch: prefill vs decode 不同策略 |

---

## 13. Community Research (2026-05-19)

社区数据验证了本 roadmap 的核心假设，并提供了关键实现指导：

### Routing 分布高度偏斜 — P0.2 前提确认

- **"MoE Lens"** (OpenReview 2025): 少数 expert 处理 >50% routing 决策
- **"MoE-Sieve"** (arXiv 2603.24044): 仅 25% expert 即可达到 full LoRA 效果
- **结论**: 精准预加载 top-N 热门 expert 确定有显著收益

### 相邻层 Expert 可精准预测 — P1.1 可行性提升

- **"Fate"** (arXiv 2502.12224): 用 layer N 的 gate input 预测 layer N+1 expert 选择
  - DeepSeek V2 Lite: **93.03%** 预测准确率
  - Qwen3-30B: **94.69%** 预测准确率
  - 配合 shallow-favoring caching → **99% hit rate**
- **影响**: P1.1 的 routing 预测从 "60-70% 相邻层相关性" 提升到 **93%+ gate-input 预测**
- **实现变更**: 预取策略应使用 gate linear projection 结果，不再依赖粗糙的"上一层复用"

### Trace-Guided Prefetch 已验证有效

- **"MoE-Infinity"** (arXiv 2401.14361): 通过 expert selection traces 指导 cache/prefetch
  - 在 DeepSeek 和 Mixtral 上实现 **3.1-16.7x per-token latency 改善**
- **HuggingFace 数据集** `core12345/MoE_expert_selection_trace`: 
  - 包含 DeepSeek-R1 (671B) 的 expert selection profiling traces
  - 可直接分析 per-layer 频率分布和 token 间重用率

### 对 Roadmap 的影响

| 原假设 | 社区验证 | 修正 |
|--------|---------|------|
| P0.2 热门 expert 占比未知 | >50% token 由少数 expert 处理 | 置信度 低→高 |
| P1.1 相邻层预测 60-70% | Fate gate-input 预测 93%+ | 策略升级 |
| P1.1 异步预取收益 ITL -8~12ms | MoE-Infinity 3.1-16.7x | 收益可能被低估 |
| Cache hit rate 天花板 50% | Fate shallow-favoring → 99% | 需研究 Fate 策略适用性 |

**结论: P1.1 置信度从"中"提升到"高"，6-7 tok/s 目标达成概率显著提升。**
