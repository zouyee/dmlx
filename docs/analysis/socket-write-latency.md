# HTTP Response Latency Analysis (2026-05-14)

## 问题

Server 端推理只需 ~500ms（3 tokens），但 curl 端到端延迟 ~20-23s。

## 测试方法

```bash
# 1. 构建 ReleaseFast
zig build -Doptimize=ReleaseFast

# 2. 启动 server
./zig-out/bin/dmlx serve --model ~/models/DeepSeek-V4-Flash-4bit --port 18090 \
  --smelt --smelt-strategy stream --smelt-experts 0.1 --smelt-cache 2048 \
  --temperature 0 --max-tokens 5

# 3. 等待 server ready
curl -sf http://localhost:18090/health

# 4. 测试（带 curl 计时）
curl --max-time 120 -s -w "\ncurl_total: %{time_total}s starttransfer: %{time_starttransfer}s\n" \
  http://localhost:18090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"hi"}],"max_tokens":3,"temperature":0}'
```

## 排查过程

### Round 1: 添加 Forward 计时

在 `DSV4Model.forward` 中添加 `mach_absolute_time` 计时：

```
[Forward] seq_len=5 total=178.1ms (embed=0.0ms layers=178.1ms head=0.0ms)
```

结论：Forward 总计 ~439ms（3 tokens），全部在 layers 中。

### Round 2: 添加 HTTP 各阶段计时

```
[HTTP] Request received: 244 bytes (read took 1ms)
[HTTP] Template=0ms Tokenize=0ms prompt_len=5
[HTTP] Request 1 submitted to queue
[HTTP] Timing: push→wait_start=0ms, wait_duration=510ms
[HTTP] Response: gen=510ms write=0ms total=510ms
[HTTP] Connection lifetime: 510ms
```

结论：整个 `handleConnection` 函数只用了 510ms。

### Round 3: 对比 curl 时间

| 指标 | Server 端 | curl 端 |
|------|-----------|---------|
| HTTP read | 1ms | - |
| Template + Tokenize | 0ms | - |
| Engine forward | ~400ms | - |
| waitForToken | 510ms | - |
| Response write | 0ms | - |
| **Connection lifetime** | **510ms** | - |
| **curl total** | - | **21.27s** |
| **差距** | | **~20.8s** |

### Round 4: 定位延迟位置

`Connection lifetime: 510ms` 证明从 `handleConnection` 函数入口到出口只有 510ms。
但 curl 显示 21.27s。

**结论：20.8s 延迟在 `io.async(handleConnection, ...)` 被调度执行之前。**

即：accept loop 调用 `io.async(handleConnection, ...)` 创建 fiber 后，
该 fiber 要等待 ~20s 才被 Zig IO 调度器分配执行时间。

## 根因

### Zig 0.16 Threaded IO Backend Fiber 调度延迟

1. Server 架构：
   - 主线程：engine loop（`engineLoopRun`）— 处理 GPU forward pass
   - Worker threads：HTTP handler fibers（通过 `io.async` 创建）

2. Engine loop 空闲时的行为：
   ```zig
   // engine_loop.zig run() 循环
   if (no_work) {
       threadSleepMs(1);  // 主线程 sleep
   }
   ```

3. Accept loop 在 `io.async` fiber 中运行，接受连接后创建新 fiber：
   ```zig
   _ = io.async(handleConnection, .{...});
   ```

4. **问题**：Zig Threaded IO backend 的 fiber 调度依赖事件循环。
   当主线程在 `threadSleepMs(1)` 中 sleep 时，IO 事件循环不会被驱动，
   新创建的 fiber 无法被及时调度到 worker thread 上执行。

5. Fiber 最终被调度的触发条件可能是：
   - 主线程 sleep 结束后的下一次事件循环迭代
   - 或者某个 IO 事件（如 socket 可读）触发了调度器唤醒
   - 实际延迟 ~20s 暗示调度器的唤醒机制存在严重问题

### 排除的原因

- ❌ HTTP 读取延迟（实测 1ms）
- ❌ Template/Tokenize（实测 0ms）
- ❌ Engine forward（实测 ~400ms）
- ❌ Socket write 延迟（改用 posix write 后仍然 20s）
- ❌ Connection close 延迟（改用 libc close 后仍然 20s）
- ❌ TCP Nagle 算法（设置 TCP_NODELAY 后仍然 20s）

## 修复方案

将 HTTP handler 从 `io.async` fiber 改为 `std.Thread.spawn` 独立 OS 线程。
这样 HTTP handler 的调度不再依赖 Zig IO 事件循环，而是由 OS 内核直接调度。

```zig
// Before (fiber-based, ~20s scheduling delay):
_ = io.async(handleConnection, .{allocator, io, state, connection, config});

// After (thread-based, immediate execution):
_ = std.Thread.spawn(.{}, handleConnectionThread, .{allocator, state, connection, config}) catch continue;
```

## 预期效果

修复后 curl 端到端延迟应从 ~21s 降至 ~510ms（= server 处理时间）。
