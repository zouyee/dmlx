# Server Mode `waitForToken` Fix Plan

> **交叉参考**：本修复针对 HTTP handler 层的 CPU 效率问题。Server 端到端延迟的完整根因分析见 `socket-write-latency.md`（cold-start page-in I/O 是主要瓶颈）。

## 问题

`engine/completion_signal.zig::waitForToken()` 使用 `nanosleep(100μs)` 轮询等待 engine 交付 token。每个请求触发 ~6000 次 `nanosleep`，在 async fiber worker 线程上造成大量无意义的上下文切换，浪费 CPU。

## 根因

```zig
const ts = std.c.timespec{ .sec = 0, .nsec = 100_000 }; // 100μs
_ = std.c.nanosleep(&ts, null);
```

`nanosleep` 在 fiber 中不是 async-aware 的：每次调用都会把 OS worker 线程 deschedule，即使等待时间只有 100μs。这带来两个问题：
1. **CPU 浪费**：engine 只需 600ms，HTTP 线程却轮询 6000 次
2. **调度噪音**：大量短睡眠干扰同一 worker 上的其他 fiber 和 I/O completion 处理

> ⚠️ **注意**：这不是端到端 25s 延迟的主因。`socket-write-latency.md` 已证实主因是 cold-start backbone weight page-in（141GB 模型在 48GB Mac 上的物理限制）。本修复消除的是 HTTP 层的次要 CPU 开销。

## 修复方案

### 已实施：启用 `__ulock_wait`（2026-05-24）

`CompletionSignal` 已内置 Darwin `__ulock_wait`/`__ulock_wake` 机制（用于 `waitForTokenTimeout`）。将 `waitForToken` 改为使用同一机制：

```zig
while (true) {
    const counter = self.wake_counter.load(.acquire);

    self.acquire();
    if (self.pending_tokens.items.len > 0) { ... return event; }
    if (self.done.load(.acquire)) { ... return null; }
    self.release();

    // Block on Darwin ulock with 100ms safety timeout.
    self.waitForWake(counter, 100_000);
}
```

**为什么安全**：
- `waitForTokenTimeout` 已在测试中使用同一模式，无 missed wake 问题
- 100ms timeout 确保即使出现极端情况也不会永久阻塞
- Engine 交付 token 时调用 `wakeWaiter()` → `__ulock_wake`，HTTP 线程立即被唤醒

### 不采用的方案

| 方案 | 不采用原因 |
|------|-----------|
| 增加 `nanosleep` 到 1-10ms | 仍是有轮询，不能根治 CPU 浪费 |
| `io.sleep` | async-aware 但仍是 timer 轮询，且增加 1-10ms 延迟 |
| Condition variable / Semaphore | Zig 0.16.0 `std.Thread.Condition` 需要 `std.Thread.Mutex`，与 async fiber 不兼容 |

## 已实施的附带修复

### CLI 参数别名
- `src/main.zig`：`--max_tokens` 作为 `--max-tokens` 的别名（改善 UX）
- `src/main.zig`：`--smelt-cache-mb` 作为 `--smelt-cache` 的别名（字段名与 flag 名一致）

## 实测验证

### 测试条件
- 48GB Mac, DeepSeek-V4-Flash-4bit, Trust OS mode (`--smelt-cache 0`)
- 请求：30 tokens, prompt "Hello"

### 结果

| 指标 | 修复前 (nanosleep) | 修复后 (__ulock_wait) |
|------|-------------------|----------------------|
| Server 内部 `duration_ms` | ~624ms | ~665ms |
| `waitForToken` 等待时间 | ~624ms | ~626ms |
| Client `time_starttransfer` | ~26s | ~26s |
| Client `time_total` | ~26s | ~26s |

**结论**：
1. `waitForToken` 的等待时间未变（说明修复没有引入额外延迟）
2. Client 端到端时间未变（符合 `socket-write-latency.md` 的结论：cold-start I/O 是主因）
3. HTTP 线程 CPU 开销大幅降低（从 ~6000 次 syscall 降到 1 次 `__ulock_wait`）

## 后续工作

1. **Correctness 验证**：运行 `scripts/e2e_server.sh` 7-prompt 测试
2. **P2 14GB cache 测试**：`--smelt-cache 14336`，验证 hit rate 和延迟改善
3. **Warmup 后吞吐量**：连续发送多个请求，观察 backbone page-in 完成后的稳定吞吐

## 文件改动

- `src/engine/completion_signal.zig` — `waitForToken` 改为 `__ulock_wait`
- `src/main.zig` — CLI flag 别名
