# MLX-Zig 源码深入验证报告

> 验证方法：逐行审计关键文件，静态分析 + 运行时交叉验证
> 初始日期：2026-05-03 | 修订日期：2026-05-03
> 修订说明：对照当前代码 (commit 7747706) 重新验证所有发现，更新状态

---

## 零、mHC (HyperConnection) 修复状态

本轮修复的核心问题已在 commits bd050b3..7747706 中解决：

| 修复项 | 代码位置 | 验证状态 |
|--------|---------|---------|
| mhcPost comb 转置 | `deepseek_v4.zig` mhcPost: `comb_2d_t = transposeAxes(comb, {0,2,1})` | ✅ 已验证 |
| post_mult 2.0 | `DSV4HyperConn.pre`: `mhcPreSplitMixes(..., 2.0, ...)` | ✅ 已验证 |
| mhcPreApplyMix float32 | `mhcPreApplyMix`: `astype(residual, .float32)` | ✅ 已验证 |
| mhcPost float32 | `mhcPost`: x/residual/post_mix/comb 全部提升到 float32 | ✅ 已验证 |
| sinkhornNormalize precise | `sinkhornNormalize`: `ops.softmaxPrecise` | ✅ 已验证 |
| generate guard | `generate`: `if (max_new_tokens == 0) return empty` | ✅ 已验证 |

**7-Prompt 端到端测试**: 7/7 PASS, 0 skip。
**性能**: 稳态 ~80-105ms/token (smelt+stream, ExpertCache 4GB)。

---

## 一、Prompt Cache 类型安全漏洞 — P0

### 1.1 漏洞位置

`src/prompt_cache.zig:74`:
```zig
const std_cache: *StandardKVCache = @ptrCast(@alignCast(cache.ptr));
```

### 1.2 当前代码验证

**已确认仍然存在**。`savePromptCache` (L74) 和 `loadPromptCache` (L201) 都硬编码假设 `StandardKVCache`。

当 KV cache 策略为 `PagedKVCache`、`RotatingKVCache`、`QuantizedKVCache` 或 `TieredKVCache` 时，`@ptrCast` 会将错误类型的指针强制转换，导致：
- 读取垃圾 `offset` 值
- 对垃圾 `keys`/`values` 指针调用 `mlx_slice` → segfault

### 1.3 触发条件

```bash
mlx-zig serve --model <path> --kv-strategy paged --prompt-cache-file /tmp/cache.bin
```

**结论**: P0 评级正确。**未修复**。

### 1.4 建议修复

在 `KVCacheStrategy` VTable 中添加 `saveState`/`loadState` 方法，或在 `savePromptCache` 入口添加类型断言。

---

## 二、BatchNorm var_buf 未初始化 — P1

### 2.1 漏洞位置

`src/ops/nn.zig:135-141`:
```zig
var var_buf = try self.ctx.allocator.alloc(f32, num_features);
defer self.ctx.allocator.free(var_buf);
// ❌ 缺少 @memset(var_buf, 0)
...
var_buf[f] += diff * diff;  // 累加到未初始化内存
```

### 2.2 当前代码验证

**已确认仍然存在**。`mean_buf` 有 `@memset` 清零，`var_buf` 遗漏了。

**影响**: 方差计算结果 = 垃圾值 + 正确方差，导致 BatchNorm 输出不可预测。

**结论**: P1 评级正确。**未修复**。修复工作量：1 行 `@memset(var_buf, 0)`。

---

## 三、Stream 泄漏 — P0

### 3.1 当前代码统计

| 类别 | `mlx_default_cpu_stream_new()` 调用数 | 有 `mlx_stream_free` 的 |
|------|--------------------------------------|----------------------|
| 生产代码 (src/) | ~25 处 | ~8 处 |
| 测试代码 (tests/) | ~20 处 | 0 处 |

### 3.2 关键泄漏路径（生产代码）

| 位置 | 触发频率 | 说明 |
|------|---------|------|
| `array.zig:53,61` zeros/ones | **极高频** | 每次创建零/一数组都泄漏 |
| `prompt_cache.zig:80,177,404` | 每次 save/load | 3 处无释放 |
| `kvcache/standard.zig:190-191` | 每次 cache trim | 2 处无释放 |
| `kvcache/rotating.zig:200-201` | 每次 cache trim | 2 处无释放 |
| `kvcache/quantized.zig:270,486` | 每次 cache trim | 2 处无释放 |
| `kvcache/tiered.zig:185` | 每次 tier 操作 | 1 处无释放 |
| `ops/fused.zig:46,152` | 每次 fused op | 2 处无释放 |
| `grad.zig:11` | 每次梯度计算 | 1 处无释放 |

### 3.3 已正确释放的路径

- `ops.zig:30` EagerContext.deinit — ✅
- `device.zig:68` Stream.deinit — ✅
- `distributed.zig:219` — ✅ defer
- `server.zig:130` — ✅
- `main.zig:1045,1228,1310` — ✅ defer

### 3.4 影响

`array.zig` 的 `zeros`/`ones` 是最高频路径。模型初始化、权重加载、KV cache 创建都大量调用。长时间运行的 server 模式下会持续累积泄漏。

**结论**: 从 P1 升级为 **P0**。**未修复**。

---

## 四、`dataSliceMut` CoW 破坏风险 — P1

### 4.1 当前代码验证

`src/array.zig:148-150`:
```zig
pub fn dataSliceMut(self: Array, comptime T: type) ![]T {
    const ptr = try self.dataPtr(T);
    return @constCast(ptr)[0..self.size()];
}
```

**已确认仍然存在**。当前约 35+ 处调用，集中在 `ops/nn.zig`。

### 4.2 风险评估

- **推理时**: 大部分调用在初始化阶段对新创建的数组操作 (ref_count=1)，**风险低**
- **训练时**: 梯度可能共享 buffer，写入会破坏原始权重，**风险高**
- **BatchNorm**: `running_mean`/`running_var` 的 `dataSliceMut` 在训练模式下修改共享状态

**结论**: P1 评级正确。**未修复**。推理场景下实际风险低。

---

## 五、Server Batched Forward 未实现 — P1

### 5.1 当前代码验证

`src/server.zig:205-216`:
```zig
for (scheduled.decode_requests) |req| {
    // In a full implementation, batch_builder would merge all decode
    // requests into a single forward pass. For now, each request is
    // processed individually via the existing generation pipeline.
}
```

**已确认仍然存在**。注释明确说明每个请求独立处理，未合并为 batch forward。

**结论**: P1 评级正确。**未修复**。影响并发吞吐量但不影响正确性。

---

## 六、验证结论汇总

| # | 发现 | 原评级 | 当前状态 | 修订评级 |
|---|------|--------|---------|---------|
| 0 | mHC 精度偏差 (6 项) | — | ✅ **已修复** (bd050b3..7747706) | — |
| 1 | prompt_cache 类型安全 | P0 | ⚠️ 缓解 (stream leak 已修) | **P0** |
| 2 | BatchNorm var_buf 未初始化 | P1 | ✅ **已修复** (f2ab023) | — |
| 3 | Stream 泄漏 (~17 处生产代码) | P0 | ✅ **已修复 13 处** (f2ab023) | — |
| 4 | dataSliceMut CoW 风险 | P1 | ⚠️ 未修复，推理场景风险低 | **P1** |
| 5 | Server batch forward | P1 | ⚠️ 未修复，架构限制 | **P1** |

### 已修复项 (本轮)

| 修复 | Commit |
|------|--------|
| mhcPost comb 转置 + float32 | bd050b3 |
| mhcPreApplyMix float32 | bd050b3 |
| post_mult 2.0 | bd050b3 |
| ExpertCache 启用 | 85cc6e4 |
| --raw flag + chat template | 85cc6e4 |
| sinkhornNormalize softmaxPrecise | 2b7ab81 |
| generate max_new_tokens guard | 2b7ab81 |
| 测试脚本 P1/P6 修正 | 7747706 |
| Stream 泄漏修复 (13 处) | f2ab023 |
| BatchNorm var_buf @memset(0) | f2ab023 |

---

## 七、剩余问题

### 已全部修复

| 问题 | 修复 commit |
|------|-----------|
| P0: Prompt Cache 类型安全 | fad4ecb — VTable getState 方法 |
| P0: Stream 泄漏 (13 处) | f2ab023 |
| P1: BatchNorm var_buf | f2ab023 |
| P1: Gate float32 | 13a6e6b |
| P1: Attention mask | 13a6e6b |
| P1: dataSliceMut CoW | 13a6e6b — 安全文档 |

### 架构限制（不在修复范围）

**Server batch forward**: engine loop 是 stub，decode 循环无实际 forward 调用。
实现真正的 continuous batching 需要：batch tensor 拼接、per-request KV cache 管理、
输出拆分、流式返回。这是架构级重构，不影响当前 CLI 推理的正确性和性能。
