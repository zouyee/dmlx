# Socket Write Latency Analysis (2026-05-14)

## 问题

Server 端推理只需 ~470ms（3 tokens），但 curl 端到端延迟 ~19s。
差距 ~18.5s 不在 HTTP 读取、模板处理、tokenization、engine forward 中。

## 排查过程

### 1. 添加 Forward 计时

在 `DSV4Model.forward` 中添加 `mach_absolute_time` 计时：

```
[Forward] seq_len=5 total=178.1ms (embed=0.0ms layers=178.1ms head=0.0ms)
[Forward] seq_len=1 total=132.3ms (embed=0.0ms layers=132.3ms head=0.0ms)
[Forward] seq_len=1 total=128.8ms (embed=0.0ms layers=128.8ms head=0.0ms)
```

结论：Forward 总计 ~439ms，全部在 layers 中（attention + MoE expert loading）。

### 2. 添加 HTTP 读取计时

```
[HTTP] Request received: 244 bytes (read took 1ms)
```

结论：HTTP 读取正常，1ms 完成。

### 3. 添加 Template/Tokenize 计时

```
[HTTP] Template=0ms Tokenize=0ms prompt_len=5
```

结论：模板和分词瞬间完成。

### 4. 添加 waitForToken 计时

```
[HTTP] Timing: push→wait_start=0ms, wait_duration=468ms
[RequestLog] duration_ms=468.94 tokens_per_sec=6.40
[Signal] waitForToken got id=0 waited 468ms
```

结论：从 queue push 到拿到所有 tokens 只需 468ms。

### 5. 添加响应写入计时

```
[HTTP] Response: gen=468ms write=0ms total=468ms
```

结论：`writeJsonResponse` 返回耗时 0ms。

### 6. 最终定位

| 阶段 | Server 端计时 | 说明 |
|------|--------------|------|
| HTTP read | 1ms | ✅ |
| Template + Tokenize | 0ms | ✅ |
| Queue push → wait start | 0ms | ✅ |
| waitForToken (engine) | 468ms | ✅ |
| writeJsonResponse | 0ms | ✅ |
| **Server 总计** | **469ms** | |
| **curl 总计** | **19.5s** | |
| **差距** | **~19s** | ❌ |

## 根因

### Socket 非阻塞模式 + Zig IO 调度器延迟

1. 连接 socket 在 accept 后被设为 `O_NONBLOCK`：
   ```zig
   _ = fc.fcntl(connection.socket.handle, fc.F_SETFL, flags | fc.O_NONBLOCK);
   ```

2. `writeJsonResponse` 使用 Zig IO writer：
   ```zig
   var writer = stream.writer(io, &buf);
   try writer.interface.writeAll(data);
   try writer.interface.flush();
   ```

3. Zig 的 `writer.interface.flush()` 在非阻塞 socket 上：
   - 如果 kernel send buffer 满或 TCP 窗口受限，write 返回 EAGAIN
   - Zig IO backend 将 fiber yield，等待 socket 可写
   - Fiber 重新调度依赖 Zig Threaded IO backend 的 epoll/kqueue 循环

4. **关键问题**：当 engine loop 在主线程做 GPU forward pass 时，
   Zig IO backend 的事件循环无法及时处理 socket 可写事件，
   导致 HTTP worker fiber 被延迟调度。

### 为什么 `write=0ms` 但 curl 收到数据要 19s？

`writeJsonResponse` 中的 `flush()` 可能只是将数据放入了 Zig IO 的
内部缓冲区，实际的 socket write 系统调用被延迟到 fiber 下次被调度时执行。
或者 `flush()` 确实调用了 write syscall，但由于 O_NONBLOCK，
只写入了部分数据，剩余数据需要等待 fiber 重新调度后继续写入。

### 对比 HTTP 读取

HTTP 读取使用 `std.posix.read`（直接系统调用）+ 手动 `io.sleep(1ms)` 轮询，
绕过了 Zig IO writer。这就是为什么读取只需 1ms 而写入需要 19s。

## 修复方案

将 `streamWriteAll` 改为直接使用 `std.posix.write`（阻塞模式），
与 HTTP 读取的 `std.posix.read` 方式保持一致。

在写响应前临时切回阻塞模式，写完后恢复非阻塞模式（供下次读取使用）。

## 预期效果

修复后 curl 端到端延迟应从 ~19s 降至 ~500ms（= engine 处理时间）。
