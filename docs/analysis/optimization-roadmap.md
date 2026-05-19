# dmlx Optimization Roadmap

> **Date**: 2026-05-19 (updated)
> **Baseline**: commit 0243aeb (SMELT 20% + 6GB cache + warmup)
> **Hardware**: Apple M4 Pro, 48GB unified memory

---

## Current Performance (2026-05-19)

| Metric | Value | Notes |
|--------|-------|-------|
| **Server-side tok/s (steady)** | **17.8** | SMELT 20% + 6GB cache (bench commit 7ea49aa) |
| **Server-side tok/s (warm, latest)** | **22-26** | commit 0243aeb |
| **100-token HTTP total** | **193s** | bench: client 吞吐 ~0.5 tok/s |
| **30-token HTTP total** | **102s** | bench: client 吞吐 ~0.3 tok/s |
| **Steady-state ITL** | ~56ms | Token 7+ |
| **Prefill** | 36ms | Single token |
| **Cache hit rate** | **23.7%** | 71217 hits / 229740 misses |
| **Startup time** | 100s | SMELT 20% (incl. warmup) |
| **RSS** | ~3.9GB | Process memory (excl. mmap) |
| **Correctness** | 7/7 | All prompts pass |

**关键发现**：Server 端 100 token 只需 ~5.6s，但 client 端需要 193s。
差距 **34x** 完全来自 macOS page thrashing（141GB 模型 on 48GB RAM）。

### Optimal Configuration

```bash
zig build -Doptimize=ReleaseFast

dmlx serve --model ~/models/DeepSeek-V4-Flash-4bit \
  --smelt --smelt-strategy stream --smelt-experts 0.2 \
  --smelt-cache 6144 --temperature 0
```

> **注意**：必须使用 `ReleaseFast` 构建。Debug 模式性能约为 Release 的 1/2。

---

## Root Cause Analysis: Why Client Latency >> Server Latency

**核心问题**：Server 端 100 token 只需 5.6s，但 Client 端需要 193s（cold）/ 27-30s（warm）。

**Benchmark 数据解读**（commit 7ea49aa）：
```
30-token HTTP:  102s (TTFR ≈ Total → 非 streaming，一次性返回)
100-token HTTP: 193s
增量: (193-102)s / 70 tokens = 1.3s/token (client 视角)
Server ITL: 56ms/token
每 token OS 开销: 1.3s - 0.056s ≈ 1.24s/token (page thrashing)
```

**193s 的组成**：
```
├── ~98s: Cold start（首次请求 backbone + expert page-in）
│         证据: 30-token 102s 中，server 端只需 4.2s → 98s 是 cold start
├── ~89s: 持续 page thrashing（70 额外 token × 1.24s/token OS 开销）
│         证据: 增量 91s / 70 token = 1.3s/token，server 只需 0.056s
└── ~5.6s: 实际推理时间（server 端 100 token × 56ms）
```

**Warm 状态（0243aeb README 数据 27-30s）的组成**：
```
├── ~0s: Cold start 已过（backbone 已 page-in）
├── ~22-25s: 持续 page thrashing（100 token × 0.22-0.25s/token）
└── ~5.6s: 实际推理时间
```

Warm 状态下每 token OS 开销从 1.24s 降到 ~0.22s（backbone 常驻后 VM 压力大幅降低）。

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

**关键数据**：
- Cache hit rate: 24%（6GB cache / 11008 total entries = 9% 覆盖率）
- 每次 cache miss: mmap page fault → SSD random read (~0.1-0.5ms)
- 每 token page fault 总量: ~196 次 → OS VM 压力极大
- Backbone weights 被 expert page fault 挤出后需要 re-page-in

**结论**：优化方向应聚焦于 **提升 cache hit rate** 和 **降低 page fault 对 OS 的冲击**。

---

## Performance Evolution

| Commit | Date | Config | Server tok/s | Client Latency | Key Change |
|--------|------|--------|-------------|----------------|------------|
| dff154d | 05-05 | SMELT 10%, 4GB cache | 10.5 | — | Initial baseline |
| 538f930 | 05-09 | SMELT 10%, 4GB cache | 14.5 | — | Tuning |
| 6d339e0 | 05-14 | SMELT 10%, 10GB cache | 10.7 | 38-40s | Cache too large |
| f970d9f | 05-16 | SMELT 10%, 6GB cache | 12-13 | 31-34s | Optimal cache size |
| latest | 05-16 | SMELT 20%, 6GB cache | 14-15 | 27-30s | Optimal SMELT ratio |
| **0243aeb** | **05-16** | **SMELT 20%, 6GB cache** | **22-26** | **27-30s** | **Latest optimizations** |

---

## Verified Findings

### What Works

| Optimization | Impact | Status |
|-------------|--------|--------|
| **SMELT 20%** (not 10%, not 30%) | +15% tok/s, -12% client latency vs 10% | ✅ Optimal |
| Expert cache 6GB (not 4, not 10) | +58% tok/s vs 10GB | ✅ Deployed |
| Cache warmup before accept | -85% first-request misses | ✅ Deployed |
| mmap for expert loading | 2x tok/s vs pread | ✅ Kept |
| ReleaseFast build | ~2x vs Debug | ✅ Always |
| Expert deduplication (P4.1) | 30-50% fewer expert loads during prefill | ✅ Deployed |

### What Doesn't Work (Do Not Retry)

| Approach | Why It Failed |
|----------|---------------|
| Expert cache 10GB+ | Squeezes backbone pages → page thrashing |
| SMELT 15% (more preload) | Extra 3GB preload squeezes OS page cache |
| SMELT 30%+ | Initial improvement then degradation — memory pressure builds |
| pread replaces mmap (全量) | -44% tok/s (loses OS readahead) |
| OS threads for HTTP | 0% improvement (not a fiber scheduling issue) |
| posix write bypass | 0% improvement (write is not the bottleneck) |
| P1.1 eval skip (stream mode) | -40% ITL (stream needs eval for page-in) |
| Zero-copy mmap arrays | <7% gain, not page-aligned |
| PLD speculative decoding | -15% tok/s (low n-gram match rate for this model) |
| LRU cache eviction | -36% tok/s (cache thrashing: 258 new entries/token evict all) |
| MTP (Multi-Token Prediction) | 不适用：I/O-bound 场景下 MTP head 增加内存压力和 cache 竞争 |
| MLX compile fusion (stream mode) | Only +6% (I/O bound, not compute bound) |

### Hardware Limits (Cannot Fix in Code)

| Issue | Cause | Mitigation |
|-------|-------|------------|
| 27-30s HTTP warm latency | 141GB model on 48GB RAM → VM pressure | See new priorities below |
| 75s HTTP cold start | Backbone page-in after warmup fills cache | Warmup + smaller cache |
| Cache hit rate 24% | 256 experts × 43 layers too large for 6GB | Architectural (see below) |

---

## Next Optimization Priorities (I/O Focused)

基于根因分析，所有优化围绕两个目标：
1. **提升 cache hit rate**（减少 page fault 总量）
2. **降低 page fault 对 OS 的冲击**（减少 VM 压力对 client 延迟的影响）

---

### Priority 0: 层级分区 Cache ⭐⭐⭐ (Expected: hit rate 24% → 35%)

**Code**: `src/models/expert_cache.zig`

**Problem**: 当前全局 LFU 有结构性缺陷。浅层 expert（layer 0-5）每个 token 都被访问，
frequency 天然高于深层 expert。LFU 驱逐深层 expert 来保留浅层 → 深层永远 miss。
但实际上浅层 routing 更集中（少数 expert 覆盖大部分流量），深层 routing 更分散。

**Fix**: 将全局 `ExpertCache` 拆分为 per-layer 独立 cache：

```zig
// 替换:
cache: ?*ExpertCache = null,

// 为:
layer_caches: ?[]ExpertCache = null, // [43]ExpertCache, 每层独立 LFU

// 初始化:
const per_layer_budget = cache_budget_mb * 1024 * 1024 / num_layers; // ~143MB/layer
var caches = try allocator.alloc(ExpertCache, num_layers);
for (caches) |*c| {
    c.* = ExpertCache.init(allocator, per_layer_budget);
}
```

每层 143MB ≈ 可缓存 ~24 个 expert（每个 ~6MB for gate+up+down）。
256 experts 中缓存 24 个 = 9.4% per layer，但热门 expert 集中度高，
实际 hit rate 预期 30-40%。

**进阶**：按层的 routing 集中度动态分配预算（集中度高的层给更多 budget）。
需要先收集 per-layer routing 统计数据。

**Expected**: hit rate 24% → 35%，page fault 数量减少 ~15%，ITL 和 client 延迟均改善。

**Risk**: 低。最坏情况与当前持平（全局 LFU 退化为 per-layer LFU 的特殊情况）。

**Effort**: 1-2 天。

**Verification**:
1. 对比 per-layer cache 前后的 hit rate（从日志 `Token step N: cache hits=X misses=Y`）
2. 对比 server tok/s 和 client 延迟
3. 确保 7/7 correctness pass

---

### Priority 0: 热门 Expert 精准预加载 ⭐⭐⭐ (Expected: hit rate +10-15%)

**Code**: `src/models/expert_preload.zig`, SMELT 初始化逻辑

**Problem**: SMELT 20% 预加载 51 个 expert × 43 层 = 2193 entries（8GB）。
但这 51 个 expert 是按 index 顺序选择的（expert 0-50），不一定是实际最常被路由到的。
如果预加载命中率只有 50%，则 4GB 预加载内存被浪费。

**Fix**:

Phase A — 离线统计（1 天）：
```bash
# 跑 10-20 个代表性 prompt，记录每层每个 expert 的被选中次数
dmlx bench --model ~/models/DeepSeek-V4-Flash-4bit \
  --routing-stats --output routing_stats.json
```

在 `streamingForward` 中添加 routing 统计收集：
```zig
// 记录每层每个 expert 的被选中次数
var routing_freq: [43][256]u64 = .{.{0} ** 256} ** 43;
// 在 routing 完成后:
for (indices_data) |eid| {
    routing_freq[layer_idx][eid] += 1;
}
```

Phase B — 精准预加载（1 天）：
```zig
// 按频率排序，预加载每层 top-N（而非顺序的前 51 个）
const hot_experts = loadRoutingStats("routing_stats.json");
for (0..43) |layer| {
    // 预加载该层最热门的 N 个 expert
    for (hot_experts[layer][0..smelt_count]) |eid| {
        preloadExpert(layer, eid);
    }
}
```

**Expected**: 预加载命中率从 ~50% 提升到 80%+。等效于 cache hit rate 额外提升 10-15%。

**Risk**: 低。最坏情况：统计分布与实际使用不匹配，退回当前行为。

**Effort**: 2-3 天。

**Verification**:
1. 对比预加载前后的 cache hit rate
2. 对比 cold start 时的 first-request 延迟
3. 确保预加载的 expert 与 cache 中的 expert 不重复浪费空间

---

### Priority 1: mmap/pread 混合模式 ⭐⭐⭐ (Expected: client 30s → 18-22s)

**Code**: `src/models/expert_stream.zig` (`ExpertStreamProvider.initWithStrategy`)

**Problem**: 当前全部使用 mmap。Expert 的随机访问模式导致大量无效 readahead +
VM 映射压力。Backbone 的顺序访问模式适合 mmap readahead。

已验证的数据：
- 纯 mmap: server 14-15 tok/s, client 27-30s
- 纯 pread: server 4.9 tok/s, client 56s → 39.8s (with warmup)

**Fix**: Backbone 保持 mmap（利用 OS readahead），Expert cache miss 改用 pread（避免 VM 压力）：

```zig
// ExpertStreamProvider.initWithStrategy (stream mode):
// 1. MmapPool 只用于 backbone weights（顺序访问，readahead 有效）
// 2. Expert cache miss 走 FdPool + pread（随机访问，不产生 VM 映射）
// 3. Expert cache hit 直接从用户空间 cache 取（零 syscall）

// PartialTensorReader 改为使用 pread:
reader.* = safetensors_reader.PartialTensorReader.init(allocator, index, fd_pool);
// 不再通过 mmap_pool 读取 expert weights
```

**关键洞察**：
- Expert cache hit（24-35%）：直接从用户空间 cache 取，零 VM 压力
- Expert cache miss（65-76%）：pread 到用户空间 buffer，不产生 VM 映射
- Backbone weights：保持 mmap，OS readahead 对顺序访问有效
- 结果：消除 expert loading 对 OS VM 子系统的冲击

**Expected**:
- Server tok/s: 可能从 22-26 降到 16-20（pread 无 readahead）
- Client latency: 从 27-30s 降到 18-22s（VM 压力大幅降低）
- **净效果**：用户体验改善（client 延迟是用户感知的指标）

**Risk**: 中。Server tok/s 可能下降。需要实测确认 client 延迟改善是否值得。

**Effort**: 3-5 天。

**Verification**:
1. A/B 对比：纯 mmap vs 混合模式的 server tok/s 和 client 延迟
2. 监控 `vm_stat` 的 page fault 率变化
3. 确保 7/7 correctness pass
4. 如果 server tok/s 下降 > 30%，考虑回退

**Decision Gate**: 如果混合模式的 client 延迟改善 < 5s，则不值得 server tok/s 的损失。

---

### Priority 1: I/O 节流 ⭐⭐ (Expected: client latency -3~5s)

**Code**: `src/models/expert_stream.zig` (`streamingForward`)

**Problem**: 43 层串行执行，每层的 cache miss 立即触发 page fault。一个 token 有
~196 次 miss，形成 page fault 风暴，瞬时压垮 OS VM 子系统。

**Fix**: 在 token step 级别限制并发 page fault 数量：

```zig
// 方案 A: 批量化 I/O（减少 syscall 频率）
// 每层加载 expert 时，先收集所有 miss 的 expert IDs，
// 然后一次性 preadv 加载（而非逐个触发 page fault）

// 方案 B: 层间 yield（给 OS 喘息空间）
// 每处理 N 层后，yield 一次让 OS 处理积压的 VM 操作
if (layer_idx % 8 == 0) {
    std.Thread.yield() catch {};
}
```

**Expected**: 减少 OS VM 子系统的瞬时压力，让 accept()/socket 操作有机会在
page fault 间隙执行。Client 延迟改善 3-5s。

**Risk**: 低。方案 B 几乎零成本，最坏情况无效果。

**Effort**: 半天。

**Verification**:
1. 对比 yield 前后的 client 延迟
2. 确保 server tok/s 不退化

---

### Priority 1: 异步 Expert 预取（Raw I/O）⭐⭐⭐ (Expected: ITL -8~12ms)

**Code**: `src/models/layer_prefetcher.zig`, `src/models/expert_stream.zig`

**Problem**: `LayerPrefetcher` 已实现但被禁用。注释说 "MLX tensor operations are
not thread-safe"。但实际上预取线程只需要做 `pread`（纯 POSIX I/O），不需要调用 MLX ops。

**Fix**: 重构 prefetcher，将 I/O 和 MLX Array 构造分离：

```zig
// 后台线程：只做 pread 到 raw buffer（线程安全）
fn prefetchWorkerRawIO(self: *LayerPrefetcher) void {
    // pread expert rows into pre-allocated byte buffers
    // NO MLX operations here — pure POSIX I/O
    const bytes = pread(fd, buffer.ptr, size, offset);
    self.raw_buffers[layer][expert_id] = buffer[0..bytes];
    self.state.store(STATE_DONE, .release);
}

// 主线程：从 raw buffer 构造 MLX Array（在需要时）
fn getPreloadedExpert(self: *LayerPrefetcher, layer: usize, eid: u32) ?Array {
    if (self.raw_buffers[layer][eid]) |buf| {
        // 构造 MLX Array（主线程，线程安全）
        return Array.fromData(self.allocator, buf, shape, dtype);
    }
    return null;
}
```

**Routing 预测策略**（用于决定预取哪些 expert）：
- 策略 A: 复用 Layer N 的 routing 结果预取 Layer N+1（相邻层相关性 ~60-70%）
- 策略 B: 维护 per-layer 热门 expert top-8 列表，始终预取这 8 个
- 策略 C: 两者结合——预取 Layer N 的 routing ∪ per-layer top-8

**Expected**: 隐藏 60-70% 的 I/O 延迟在 GPU 计算时间内。
每层 attention ~0.3ms × 43 = 13ms 可用于预取 → ITL 从 56ms 降到 ~44-48ms。

**Risk**: 中。Routing 预测不准确时预取浪费 I/O 带宽。但不会比当前更差。

**Effort**: 1 周。

**Verification**:
1. 对比启用/禁用 prefetcher 的 ITL
2. 监控预取命中率（预取的 expert 实际被使用的比例）
3. 确保无 data race（raw buffer 的生命周期管理）
4. 确保 7/7 correctness pass

---

### Priority 3: Reduce HTTP Cold Start ⭐⭐ (Expected: 75s → 40-50s)

**Options**:
1. **Smarter warmup**: 基于 routing 统计，使用能最大化 expert 覆盖率的 warmup prompts
2. **Backbone pinning**: `mlock()` 锁定 backbone weight pages 防止 OS 换出
   Risk: 可能触发 OOM（总 locked memory 超过可用 RAM）
3. **渐进式 warmup**: 先接受请求（快速响应），后台继续 warmup cache

**Effort**: 1 week.

---

## Abandoned Priorities

### ~~MLX Compile Fusion~~ ❌ Verified: Only +6% in Stream Mode

**Tested**: Compile fusion for decode forward pass.

**Result**: In stream mode (I/O bound), compile fusion only provides +6% improvement.
The bottleneck is SSD I/O, not Metal kernel dispatch overhead.

**Conclusion**: Compile fusion is only valuable when the model is fully memory-resident
(64GB+ Mac). On 48GB Mac with stream mode, I/O dominates and compute optimization
has minimal impact.

### ~~MTP (Multi-Token Prediction)~~ ❌ Not Applicable

**Analysis**: MTP 解决的是 compute-bound 场景。dmlx 在 48GB Mac 上是 I/O-bound：
- MTP head 本身包含完整 MoE 层（~2-4GB 额外权重）
- 会与 target model 竞争 6GB expert cache 空间
- 增加内存压力 → 加剧 page thrashing
- ds4 项目（128GB Mac，权重全部常驻）也只有 "slight speedup"

**Conclusion**: 等硬件升级到权重全部常驻内存后再考虑。

### ~~PLD Speculative Decoding~~ ❌ Verified Ineffective

N-gram match rate too low for this model. -15% tok/s.

### ~~LRU Cache Eviction~~ ❌ Verified Ineffective

Cache thrashing: 258 new entries/token evict all previous entries. -36% tok/s.

---

## Target Milestones

**阶段性目标：Warm client 吞吐 6-7 tok/s（100-token < 15s）**
**Stretch 目标：10-12 tok/s（需 P1 混合模式验证通过）**

| Milestone | Target | Status | Approach | 置信度 |
|-----------|--------|--------|----------|--------|
| **M5: Cache hit rate** | **40%+** | 🔄 Next | Layer-partitioned cache + smart preload | 高 |
| **M6: Client throughput (warm)** | **6-7 tok/s** | 🔄 Next | P0 cache 优化 + 异步预取 | 高 |
| **M7: Stretch — Client throughput** | **10-12 tok/s** | 🔄 Conditional | P1 混合模式（需 Decision Gate 通过） | 低(40%) |
| M4: SMELT tuning | Best client latency | ✅ | SMELT 20% = 28s client | — |
| ~~M1: PLD enabled~~ | ~~50+ effective tok/s~~ | ❌ | N-gram match too low | — |
| ~~M2: Cache strategy~~ | ~~50%+ hit rate~~ | ❌ | LRU causes thrashing | — |
| ~~M3: Compile fusion~~ | ~~30ms ITL~~ | ❌ | Only +6% in stream mode | — |

---

## Implementation Plan

```
Week 1 (确定性收益，低风险):
├── Day 1-2: P0 层级分区 Cache
│   ├── 改 ExpertCache 为 [43]ExpertCache
│   ├── 验证 hit rate 变化
│   └── 如果 hit rate < 30%，尝试动态预算分配
├── Day 3-4: P0 热门 Expert 精准预加载
│   ├── 添加 routing 统计收集
│   ├── 跑 benchmark 收集 routing_stats
│   └── 修改 SMELT 预加载逻辑使用统计结果
└── Day 5: P1 I/O 节流
    ├── 添加层间 yield
    └── 验证 client 延迟变化

Week 2 (中等风险，需要 A/B 测试):
├── Day 1-3: P1 mmap/pread 混合模式
│   ├── Backbone 保持 mmap
│   ├── Expert miss 改用 pread
│   ├── A/B 对比 server tok/s vs client latency
│   └── Decision gate: client 改善 > 5s 才保留
└── Day 4-5: P1 异步 Expert 预取
    ├── 重构 LayerPrefetcher 为 raw I/O 模式
    ├── 实现 routing 预测（策略 B: per-layer top-8）
    └── 验证 ITL 改善
```

---

## Expected Outcomes

**阶段性目标：Client 吞吐达到 10-12 tok/s（48GB Mac 物理极限）**

### Benchmark 数据解读（commit 7ea49aa）

```
30-token HTTP total:  102s (TTFR ≈ Total，非 streaming)
100-token HTTP total: 193s
增量: 70 token / 91s = 1.3s/token (client 视角)
Server ITL: 56ms/token
每 token OS 开销: 1.3s - 0.056s ≈ 1.24s/token

193s 分解:
├── ~98s: Cold start（首次请求 backbone + expert page-in）
├── ~5.6s: 实际推理时间（server 端 100 token）
└── ~89s: 持续 page thrashing（70 token × 1.24s OS 开销）
```

**关键认知**：
- Cold start 占 193s 的 ~51%（98s）
- 持续 page thrashing 占 ~46%（89s）
- 实际推理只占 ~3%（5.6s）

README 中 0243aeb 的 "27-30s" 应为 **warm 状态后续请求**（cold start 已过）。
如果以 warm 请求为 baseline：100 token / 27-30s ≈ **3.3-3.7 tok/s**。

### 预期结果

| 场景 | 100-token (cold) | 100-token (warm) | Client tok/s (warm) | Cache Hit Rate |
|------|-----------------|-----------------|---------------------|----------------|
| **当前 (bench 7ea49aa)** | 193s | ~27-30s* | 3.3-3.7 | 24% |
| **P0 完成后** | 120-150s | 18-22s | 4.5-5.5 | 40% |
| **P0+P1 完成后（保守）** | 60-90s | 12-18s | 5.5-8 | 40% |
| **P0+P1 完成后（乐观）** | 30-50s | 8-12s | 8-12 | 45% |
| **物理极限** | ~15-20s | **8-10s** | **10-12** | ~50% |

*\* 27-30s 为 0243aeb README 数据，未经正式 benchmark 验证*

**优化对两个阶段的影响**：
1. **Cold start（98s）**：P0 精准预加载直接减少 → 预期降到 30-50s
2. **持续 page thrashing（1.24s/token）**：P0 cache hit rate 提升 + P1 混合模式 → 预期降到 0.3-0.5s/token

**达到 10-12 tok/s (warm) 的条件**：
- 每 token OS 开销从 1.24s 降到 < 0.05s（几乎消除 page thrashing）
- 这要求 cache hit rate > 80% 或 expert miss 不产生 VM 压力
- P0 单独无法达到（hit rate 最多到 45%）
- P1 混合模式是关键——但已有数据显示纯 pread client 更慢（39.8s vs 27-30s）

**修正后的现实预期**：
- P0+P1 全部完成后 warm client: **8-15s**（5-12 tok/s）
- 达到 10 tok/s 需要混合模式有效（Decision Gate 通过）
- 如果混合模式无效，warm client 停在 ~15-18s（5.5-6.5 tok/s）

如果 P0+P1 全部完成后仍未达到 10 tok/s，剩余差距来自：
- OS 内存管理的固有开销（不可消除）
- Backbone page-in 的随机性（取决于 OS 调度）
- 需要考虑 P2 方案（2-bit 压缩）或硬件升级

---

## Memory Budget (48GB Mac)

```
Current allocation (SMELT 20% + 6GB cache):
├── macOS + apps:           ~8 GB
├── Backbone (mmap):        ~6 GB (page-cached by OS)
├── SMELT preloads (20%):   ~8 GB (51 experts × 43 layers)
├── Expert cache (LFU):     ~6 GB (configured)
├── KV cache:               ~0.5 GB
├── MLX runtime + Metal:    ~2 GB
├── OS page cache headroom: ~17.5 GB ← critical for mmap performance
└── Total:                  ~48 GB

Optimization impact on memory budget:
├── Layer-partitioned cache: 0 change (same 6GB, different structure)
├── Smart preload: 0 change (same 8GB, better expert selection)
├── mmap/pread hybrid: +0 (expert mmap pages freed → more headroom)
├── Async prefetch buffers: +50-100MB (raw I/O buffers)
└── Net: slightly more headroom available with hybrid mode
```

---

## References

- `docs/en/technical/inference-optimization.md` — full optimization plan (Phases 1-5)
- `docs/analysis/inference-optimization-review.md` — critical review with corrections
- `docs/analysis/socket-write-latency.md` — HTTP latency root cause (VM pressure)
- `docs/analysis/pread-expert-loading.md` — mmap vs pread experiments
- `docs/analysis/cache-benchmark-results.md` — 4GB vs 8GB cache data
