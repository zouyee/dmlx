# Inference Optimization Plan — 深度分析与修订建议

> **基于**: `docs/en/technical/inference-optimization.md` v1.0
> **日期**: 2026-05-14
> **交叉参考**: `PERF_PLAN.md` (已实施 P0-P4), `socket-write-latency.md` (cold start 分析)

---

## 1. 整体评价

优化方案质量很高，体现了对 Apple Silicon 统一内存架构下 MoE 模型推理的深刻理解。
方案从物理极限出发（bandwidth-limited 10.6ms/tok），逐层分解瓶颈，按 ROI 排序优化项。

**核心路径正确**：cache hit rate → compile fusion → speculative decoding 是 Apple Silicon
上 MoE 推理优化的黄金路径。

---

## 2. 核心洞察验证

### 2.1 "完全带宽受限" ✅ 正确（但有隐含前提）

文档计算了 M4 Pro 的两个物理极限：
- **带宽极限**: 400 GB/s → 10.6ms/tok (94 tok/s)
- **算力极限**: 15 TFLOPS → 2.2ms/tok (460 tok/s)

batch=1 decode 完全受内存带宽约束——这是 LLM 推理的标准结论。

**⚠️ 隐含假设**：10.6ms 的带宽极限假设所有权重都在 UMA 中（400 GB/s）。
实际上 SMELT stream 模式下，大部分 expert 权重在 NVMe SSD 上（5-7 GB/s），
两者差距 60-80x。所以当前真正的瓶颈不是 UMA 带宽，而是 **SSD I/O + cache hit rate**。
文档在 Section 3 末尾也指出了这一点，但 Section 4 的优化排序未充分反映这个事实。

### 2.2 Expert Cache Hit Rate 是决定性因素 ✅ 正确

关键公式：
```
ITL ≈ f(1 - hit_rate) × SSD_latency_per_layer × 43
```

从 PERF_PLAN.md 的实测数据验证：
- 70% hit rate → 82ms/tok（当前基线）
- 95% hit rate → 理论 ~15ms SSD I/O（与 P1.2 预期吻合）

### 2.3 B1 与 B3 实际上是同一问题

B1 "MLX per-op dispatch overhead ~30ms" 和 B3 "Per-layer eval() sync ~4.3ms" 是同一问题的两面：

- `eval()` 强制切断 MLX lazy graph 为 43 段
- 每段独立 dispatch → 无法跨层 fusion
- 消除 eval() 是 compile fusion 的前提条件

两者不应叠加计算收益。实际上 P1.1（skip eval）和 P2.1（compile）的收益有重叠。

---

## 3. 各 Phase 逐项评估

### Phase 1: Low-Hanging Fruit

| 项目 | 评估 | 置信度 | 备注 |
|------|------|--------|------|
| P1.1 eval skip | ⚠️ 收益可能被高估 | 中 | 跳过 eval 可能导致 graph 无限增长，需验证 decode 时 graph 大小 |
| P1.2 cache 4→10GB | ✅ 最确定的收益 | 高 | PERF_PLAN 提到 20GB 触发 swap，10GB 应安全 |
| P1.3 zero-copy | ❌ **建议移除** | 低 | PERF_PLAN 明确标注"已验证无效，只省 7%，memcpy 不是瓶颈" |

**关键发现**：PERF_PLAN.md（实际实施经验）与本文档（理论规划）在 P1.3 上有直接矛盾。
safetensors 中 expert tensor 通常不是 page-aligned，zero-copy 命中率极低。
PERF_PLAN 的实测结论更可信。

**P1.1 风险细节**：
- Decode 时每层只产生 1 个 token 的中间 tensor（~几 KB），内存压力极低
- 但如果完全不 eval，43 层的 lazy graph 会累积 ~2000+ 节点
- MLX graph scheduler 处理大 graph 本身有开销
- **建议**：改为每 8-10 层 eval 一次，而非完全跳过

### Phase 2: MLX Compile Fusion

P2.1 是整个方案的核心（预期 -49%，从 61ms 到 31ms）。

**优势**：
- 理论完全正确——消除 per-op dispatch 是 MLX 推理优化的标准做法
- mlx-lm 已证明 `mx.compile()` 对 decode 有效

**关键挑战（文档低估了复杂度）**：

1. **Expert 动态绑定问题**：
   - compile 要求所有输入 shape 静态已知
   - 虽然 top-6 数量固定，但 **哪 6 个** expert 是动态的
   - 每次 token 需重新绑定 43×7=301 个 expert tensor 到 compiled graph
   - 这个绑定开销本身可能抵消部分 compile 收益

2. **CSA/HCA 兼容性**：
   - 文档承认可能不兼容
   - 如果 fallback 到 P2.2（只编译 attention），收益从 30ms 降到 ~6ms
   - 这意味着方案的 "目标在 Phase 2 达成" 的结论不成立

3. **KV cache in-place update**：
   - `mlx_slice_update()` 在 compile 模式下的行为需要验证
   - MLX compile 对 side-effect 操作支持有限

**时间估算修正**：文档估计 2 周，实际考虑到 op 兼容性审计 + expert 预加载架构重构 + 调试，
**3-4 周更现实**。

**建议**：在投入 P2.1 之前，花 1-2 天做 **spike test**：
```zig
// 用一个简化的 2-layer model 测试 compile 兼容性
const test_model = try SimplifiedDSV4.init(2); // 只有 2 层
const compiled = try mlx_compile(test_model.decodeForward);
// 验证：MLA attention + MoE routing + SwitchGLU 是否都能 compile
```

### Phase 3: I/O Pipeline

P3.1 Multi-layer prefetch 设计合理，但有一个关键前提：

**P1.2 将 cache hit rate 提升到 95% 后，SSD I/O 已经不是主要瓶颈。**

计算：95% hit rate 时，每层只有 ~2.1MB 需要从 SSD 读取，43 层共 ~90MB，
在 6 GB/s SSD 上只需 15ms。Multi-layer prefetch 能隐藏的也就是这 15ms 中的一部分。
文档预期 4ms 收益是合理的。

**但如果 P1.2 的 95% hit rate 达不到**（比如实际只有 85%），P3.1 的价值大幅提升。
两个优化有互补关系——P3.1 是 P1.2 效果不理想时的保险。

### Phase 4: Fused SwitchGLU

P4.1 的 3ms 收益（从 27ms 到 24ms）相对于 3 周开发时间，**ROI 最低**。

**深层问题**：在 compile 模式下（P2.1），MLX 已经会自动 fuse 连续的 matmul + activation。
手写 Metal kernel 的额外价值有限。

更重要的是，batch=1 decode 时 SwitchGLU 的 matmul 是 `[1, D] × [D, intermediate]`——
这是 GEMV 而非 GEMM。Apple GPU 的 GEMV 已被 MLX Metal kernel 高度优化，
手写 tiled kernel 的边际收益接近 0。

**结论**：如果 P2.1 compile 成功，P4.1 可以跳过。
P4.1 只在 P2.1 失败时作为 fallback 有价值。

### Phase 5: Speculative Decoding (PLD)

PLD 的 2.1x 乘数是建立在所有前序优化之上的"免费"乘数。

**被忽略的问题**：PLD acceptance rate 高度依赖输出可预测性：
- 对话/翻译场景：~70% acceptance（文档假设）
- 代码生成/数学推理：可能只有 40-50%
- 创意写作：可能低于 40%

**建议**：文档应注明 2.1x 是乐观估计，保守估计为 1.5-1.7x。

---

## 4. 文档中的关键矛盾与遗漏

### 4.1 与 PERF_PLAN.md 的矛盾

| 本文档 | PERF_PLAN.md (实测) | 结论 |
|--------|---------------------|------|
| P1.3 zero-copy 预期 -3% | "零拷贝 mmap Array 只省 7%，已验证无效" | **移除 P1.3** |
| P1.2 扩大 cache 到 10GB | "增大 cache 到 20GB 触发 swap" | 10GB 安全，但需实测边界 |
| 基线 82.2ms/tok | PERF_PLAN 测到 100ms/tok (decode) | 可能是不同 commit/配置 |
| eval skip 预期 -5% | P4 修复中"移除过度 eval() 调用"已部分实施 | 需确认当前状态 |

### 4.2 Cold Start 问题未覆盖

从 `socket-write-latency.md` 的分析：
- Cold start 时 backbone weights 的 mmap page-in 需要 **17-19s**
- 这是用户体验的第一印象，比 steady-state ITL 更影响感知

**建议新增 Phase 0: Cold Start Mitigation**：

```
Phase 0: Cold Start (新增，1-2 天)
├── P0.1: Backbone warmup — 启动时 madvise(WILLNEED) 预热 6GB backbone
├── P0.2: Pre-accept — forward pass 期间不阻塞新连接 accept
└── P0.3: 减少 mmap 范围 — 只 mmap 需要的 shard 文件
预期：首请求从 ~20s 降到 ~3-5s
```

### 4.3 Prefill 优化缺失

文档标题是 "Inference Speed Optimization"，但几乎只讨论 decode phase。
Prefill 的 370ms（token 1）对短对话场景占比不小。

PERF_PLAN 的 P4（expert deduplication）已在解决这个问题（12390 → 6000-8000 unique loads），
但本文档未提及。建议补充 prefill 优化的交叉引用。

### 4.4 Batch Size > 1 场景

整个方案假设 batch=1。dmlx 有 server 模式，concurrent requests 会导致：
- KV cache 内存翻倍
- Expert cache 竞争加剧（不同请求路由到不同 expert）
- compile graph 可能需要支持动态 batch

建议至少注明 "本方案仅针对 batch=1 单请求场景" 的适用范围。

---

## 5. 修订后的优先级排序

基于以上分析，建议的实施顺序：

```
第 1 周（确定性收益，低风险）:
  1. P1.2 扩大 expert cache 到 10GB        → 预期 -15ms ⭐⭐⭐
     - 最确定的收益，1 行代码改动
     - 验证：cache hit rate 从 70% → 95%
  2. P1.1 conditional eval skip (仅 decode) → 预期 -4ms ⭐
     - 改为每 8 层 eval 一次（非完全跳过）
     - 验证：graph 节点数不超过 500
  3. P0 Cold start warmup (新增)            → 首请求 -15s ⭐⭐
     - madvise(WILLNEED) 预热 backbone
     - 验证：首请求 < 5s

第 2 周（Spike Test — 决策门）:
  4. P2.0 Compile 兼容性 spike test         → 0ms (决策用)
     - 2-layer 简化模型测试 compile 兼容性
     - 测试 MLA attention + MoE routing + SwitchGLU
     - 结果决定后续走 P2.1 还是 P2.2

第 3-5 周（高收益，中风险）:
  如果 spike test 通过:
    5a. P2.1 Full model compile             → 预期 -30ms ⭐⭐⭐⭐
  如果 spike test 失败:
    5b. P2.2 Attention-only compile         → 预期 -6ms ⭐⭐
    5c. P3.1 Multi-layer prefetch           → 预期 -4ms ⭐⭐
    5d. P4.1 Fused SwitchGLU (此时才有价值) → 预期 -3ms ⭐

第 6 周（乘数效果）:
  6. P5.1 Enable PLD                        → 1.5-2.1x 乘数 ⭐⭐⭐
     - 仅 greedy decoding (temperature=0)
     - 注明 acceptance rate 因任务类型而异
```

### 修订后的预期结果

| 路径 | 场景 | 最终 ITL | tok/s | 100-token |
|------|------|----------|-------|-----------|
| 乐观路径 | P2.1 compile 成功 | 27ms | 37 | **2.7s** ✅ |
| 乐观路径 + PLD | + speculative | eff. 13ms | eff. 77 | **1.3s** ✅ |
| 保守路径 | P2.1 失败，走 P2.2+P3+P4 | 48ms | 21 | **4.8s** ⚠️ |
| 保守路径 + PLD | + speculative | eff. 28ms | eff. 36 | **2.8s** ✅ |

**关键结论**：
- 乐观路径在 Phase 2 即可达标（3s/100-token）
- 保守路径需要 PLD 才能达标
- 无论哪条路径，P1.2（扩大 cache）都是必须的第一步

---

## 6. 新增风险项

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| P2.1 expert 动态绑定开销抵消 compile 收益 | 中 | 高 | spike test 中测量绑定开销 |
| P1.1 完全跳过 eval 导致 graph OOM | 中 | 中 | 改为每 N 层 eval 一次 |
| 10GB cache 在实际负载下 hit rate < 95% | 低 | 中 | 监控 hit rate，动态调整 |
| Cold start warmup 与 expert streaming 竞争 I/O | 低 | 低 | warmup 完成后再接受请求 |
| PLD acceptance rate 远低于 70% | 中 | 中 | 按任务类型动态启用/禁用 |

---

## 7. 已验证无效的方案（Do Not Retry）

来源：PERF_PLAN.md 实测结论 + 2026-05-16 排查

| 方案 | 原因 |
|------|------|
| P1.3 零拷贝 mmap Array | memcpy 不是瓶颈，safetensors offset 非 page-aligned，实际命中率极低 |
| P1.1 eval skip (stream 模式) | stream 模式依赖 eval() 触发 mmap page-in，跳过导致 ITL 退化 40% |
| 增大 cache 到 20GB | 触发系统 swap，反而更慢 |
| Preload 模式 (全量加载) | 48GB 内存不够，OOM |
| OS thread 替代 fiber | 实测证明 fiber 调度不是瓶颈，HTTP 延迟不变 |
| posix write 替代 Zig IO write | 实测证明 response write 不是瓶颈 |
| pread 完全替代 mmap | HTTP 延迟 -68% 但 tok/s -44%，得不偿失 |
| Warmup with dummy prompts (旧版) | 旧版无效，但新版（pread + cache warmup）有效 |

### 2026-05-16 新增发现

**38s HTTP 延迟的真正根因**：不是 accept 阻塞、不是 fiber 调度、不是 write 延迟。
是 **macOS 内存管理在 48GB Mac 上运行 141GB 模型的物理限制**：
- warmup 加载 expert weights 后，backbone weights 被 OS 换出
- 第一个请求需要重新 page-in 6GB backbone
- SSD 随机读取 6GB 需要 ~30-40s

**这是硬件限制，不可通过代码优化消除。** 唯一的解决方案是：
1. 更大内存的 Mac (64GB+)
2. 更小的模型
3. MLX compile fusion 减少 forward pass 时间（间接减少 I/O 窗口）

---

## 8. 总结

### 方案的核心正确性

优化路径 **cache hit rate → compile fusion → speculative decoding** 完全正确。
物理极限分析、瓶颈分解、per-layer timing 都是高质量的工程分析。

### 最大不确定性

P2.1（MLX compile）贡献了 49% 的预期收益。如果 compile 不兼容 CSA/HCA 或
expert 动态绑定开销过大：
- 目标（82ms → 24ms）退化为（82ms → 48ms）
- 仍然是 40% 提升，但需要 PLD 才能达到 3s/100-token 目标

### 行动建议

1. **立即执行** P1.2（扩大 cache）— 最高确定性，最低风险
2. **第 2 周** 做 compile spike test — 决定后续路径
3. **不要投入** P1.3（zero-copy）— 已验证无效
4. **补充** cold start 优化 — 用户体验的第一印象
5. **注明** batch=1 适用范围 — 避免 server 场景的误导
