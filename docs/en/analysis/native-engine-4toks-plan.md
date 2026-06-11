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

## 3. 优化路径

### Path A：对齐 MLX（0.78 → 1.6, 2×）

**方案 1：替换 MoE GPU 内核**
- flash-moe affine 4-bit 格式：w = nibble × scale + bias（替代 MXFP4 的 LUT+exp2）
- 使用 flash-moe v3 模式（已验证在 M3 Max 上 4.36 tok/s）
- Expert 大小不变（~10.5MB），SMELT 兼容
- 预估收益：MoE GPU 1100ms → 400-500ms → 1.4-1.7 tok/s

**方案 2：Hybrid MLX**
- 在 native engine 中调用 MLX C API 执行 MoE 计算
- 利用 MLX 的优化内核，其余管线保持不变
- 需要验证 MLX C API 的开销是否值得

### Path B：超越 MLX（1.6 → 4.0, 2.5×）

需要结合多个技术：
- Expert 量化格式优化（affine 4-bit）
- Pipeline 重构（消除 CMD2 wait 52ms）
- 可能的 CB-A 真消除（GPU-only data flow）

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

## 5. 下一步

1. **在 MLX 路径上测量 MoE 阶段耗时**：确认 MLX 的 MoE GPU 时间 vs native 的 1100ms
2. **实现 flash-moe affine 4-bit 替换 MXFP4**：最可能的 2× 提升路径
3. **测试 Hybrid MLX 方案**：如果方案 2 可行，可能更快达到目标

## 6. 验收标准

```bash
sudo purge
bash scripts/run_benchmark.sh  # SMELT N=51，要求 Paris ✓，tok/s 不退化
```