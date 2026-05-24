# dmlx → flash-moe 对齐：最终执行记录

> **日期**: 2026-05-25
> **硬件**: Apple M4 Pro, 14核(10P+4E), 48GB, SSD ~23 GB/s (warm seq)
> **模型**: DeepSeek-V4-Flash-4bit (141GB, 43层, K=6, 256 experts)

---

## 1. 结论

dmlx 从 0.7 tok/s → 2.5 tok/s（3.6x 提升）。flash-moe = 4.36 tok/s，差距 1.7x。

差距来源：
- **I/O 量**: dmlx 3,457 MB/token vs flash-moe 1,620 MB/token (2.1x)
  - K=6 vs K=4 (1.5x)：模型训练决定，不可改（K=4 正确性崩坏）
  - 专家 13.4MB vs 6.75MB (2.0x)：DeepSeek V4 架构决定
- **有效 I/O 速率**: dmlx ~8 GB/s vs flash-moe ~11 GB/s (1.4x)
  - safetensors 33 分片 → packed expert per-layer 文件改善显著
  - thread-per-expert 并行 pread

Server 与 client 端到端延迟完全一致（无 gap——之前的 gap 是 mach_absolute_time 单位换算 bug）。

---

## 2. 已实施方案

| 优化 | 效果 | 状态 |
|------|------|------|
| Trust OS (ExpertCache 移除) | TTFR -24%, 内存 -27% | ✅ |
| DyMoE (skip 1/6 experts) | client 延迟 -19~57% | ✅ |
| ulock_wait (nanosleep 替代) | CPU 开销降低 | ✅ |
| warmup (backbone 预热) | 首次 -25%, 后续 -36% | ✅ |
| 并行 projection pread (3 线程) | +23% tok/s | ✅ |
| packed expert + readAndAssembleAll | +100% tok/s (1.3→2.6) | ✅ |
| F_NOCACHE | 无效 (更慢) | ❌ |
| mlock backbone | 无效 (backbone 未被换出) | ❌ |
| K=6→4 | 正确性崩坏 | ❌ |

## 3. 性能演进

```
配置                                     30tok warm    100tok      tok/s
──────────────────────────────────────────────────────────────────────
原始 (Trust OS, safetensors)             25.6s         135.8s      0.7
+ warmup                                 17.3s         109.4s      1.0
+ 并行 pread (3 proj)                    14.0s          87.2s      1.3
+ packed expert + readAndAssembleAll     11.7s          72.9s      2.5
──────────────────────────────────────────────────────────────────────
总计提升                                  -54%          -46%        +257%
```

## 4. mach_absolute_time 换算 Bug

`mach_absolute_time()` 返回 ticks，非纳秒。Apple Silicon 上 1 tick = 125/3 ns ≈ 41.67 ns。

所有 `ticks / 1_000_000` 得到的"毫秒"比实际小 41.67 倍。修正后：
- "13.8ms ITL" → 实际 575ms
- "30 tok/s server" → 实际 0.72 tok/s
- Server 与 client 完全一致，从未有 "gap"

修正文件: `expert_stream.zig`, `engine_loop.zig`

## 5. 关键代码改动

| 文件 | 改动 |
|------|------|
| `expert_pread.zig` | 新文件：packed expert 并行 pread + readAndAssembleAll |
| `expert_stream.zig` | ExpertCache 移除, DyMoE, 并行 projection, packed 路径, 诊断打点, unit fix |
| `engine_loop.zig` | warmupBackbone 恢复, RequestLog 打点修正 |
| `server.zig` | warmup 调用, accept 打点 |
| `server/http.zig` | writeJsonResponse POSIX write, 诊断打点, shutdown |
| `server/openai.zig` | 诊断打点 |
| `server/config.zig` + `state.zig` + `main.zig` | packed expert CLI flag 传参 |
| `_gen_report.py` | Trust OS 解析修复 |

## 6. 未解决问题

- **数学正确性 4/7**: "2+2=", "3*3=", "10-5=" 失败，根因待定位
- **K=6→4 不可行**: 模型用 K=6 训练，router 权重不可向下兼容
- **3.0 tok/s 未达**: 差距 17%，来自 thread spawn overhead + SSD 寻道损耗

## 7. flash-moe 关键差异

| | flash-moe | dmlx |
|------|-----------|------|
| 模型 | Qwen3.5-397B (K=4) | DeepSeek V4 (K=6) |
| GPU 框架 | 手写 Metal (bare C) | MLX (Apple 框架) |
| 端到端 | 4.36 tok/s | 2.5 tok/s |
| I/O 量 | 1,620 MB/token | 3,457 MB/token |
| 有效 I/O | 11.2 GB/s | ~8 GB/s |
| 并行 I/O | GCD dispatch groups | std.Thread.spawn |
| 线程模型 | 持久化 thread pool | per-call spawn/join |
