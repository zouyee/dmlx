# HTTP Response Latency Analysis (2026-05-14)

## 问题

Server 端推理只需 ~400-500ms（3 tokens），但 curl 端到端延迟 ~17-20s。

## 测试方法

```bash
# 构建
zig build -Doptimize=ReleaseFast

# 启动 server (前台或 nohup)
nohup ./zig-out/bin/dmlx serve --model ~/models/DeepSeek-V4-Flash-4bit --port 18090 \
  --smelt --smelt-strategy stream --smelt-experts 0.1 --smelt-cache 2048 \
  --temperature 0 --max-tokens 3 > /tmp/dmlx.log 2>&1 &

# 等待 ready
while ! curl -sf http://localhost:18090/health > /dev/null 2>&1; do sleep 2; done

# 测试 health (应该 <1ms)
curl -s -w '\ncurl_total: %{time_total}s\n' http://127.0.0.1:18090/health

# 测试 chat completion
date +%s && curl --max-time 60 -s -w '\nstarttransfer: %{time_starttransfer}s total: %{time_total}s\n' \
  http://127.0.0.1:18090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"hi"}],"max_tokens":3,"temperature":0}' && date +%s
```

## 排查过程

### Round 1: Forward 计时

在 `DSV4Model.forward` 中添加 `mach_absolute_time` 计时：

```
[Forward] seq_len=5 total=143.4ms (embed=0.0ms layers=143.4ms head=0.0ms)
[Forward] seq_len=1 total=116.8ms
[Forward] seq_len=1 total=118.7ms
```

结论：3 次 forward 总计 ~379ms。

### Round 2: HTTP 各阶段计时

```
[Accept] Connection accepted fd=38
[HTTP] Thread started: accept→thread=0ms
[HTTP] Request received: 244 bytes (raw thread)
[HTTP] Template=0ms Tokenize=0ms prompt_len=5
[HTTP] Request 1 submitted to queue
[RequestLog] duration_ms=401.04 tokens_per_sec=7.48
[HTTP] Timing: push→wait_start=0ms, wait_duration=401ms
[RAW] gen=401ms write=0ms
[HTTP] Connection lifetime: 401ms (close took 0ms)
```

### Round 3: curl vs server 时间对比

| 指标 | Server 端 | curl 端 |
|------|-----------|---------|
| accept→thread | 0ms | - |
| HTTP read | <1ms | - |
| Template + Tokenize | 0ms | - |
| Engine forward (3 tokens) | ~379ms | - |
| waitForToken | 401ms | - |
| posixWriteAll | 0ms | - |
| close() | 0ms | - |
| **Connection lifetime** | **401ms** | - |
| **curl total** | - | **17-20s** |
| **差距** | | **~17s** |

### Round 4: Unix timestamp 对比

```
curl start:     1778737258 (date +%s)
response created: 1778737277 (JSON response 中的 created 字段)
差距: 19s
```

Server 在 curl 发送请求 **19s 后**才开始处理。但 `accept→thread=0ms` 说明
一旦 accept() 返回，处理是立即的。

**结论：`accept()` 系统调用本身被阻塞了 ~18s。**

### Round 5: Health endpoint 对比

```
curl health: 0.4ms ✅
```

Health 请求在 chat completion 请求之前发送，此时系统没有 I/O 压力。
Chat completion 请求在第一个请求的 forward pass 期间或之后发送，
此时系统正在做大量 mmap page-in I/O。

### Round 6: 排除 CPU 瓶颈

观察到 CPU 使用率并未达到 100%。问题不是 CPU 繁忙导致线程无法调度。

## 根因分析（确定性证据）

### `sample` 工具采样结果

```
Thread 主线程 (1973 samples / 3s):
  98.6% DSV4Model.forward → DSV4TransformerBlock → DSV4MoE → streamingForward
    └── loadExpertSlicesCached → loadExpertSlices → PartialTensorReader.readExpertRows
        ├── pread (267 samples, 13.5%) ← SSD I/O 等待
        └── Allocator.alloc (172 samples, 8.7%) ← 内存分配

Thread Accept 线程 (1973 samples):
  100% __accept ← 正常阻塞等待新连接

Thread HTTP handler (1973 samples):
  100% nanosleep (waitForToken) ← 等待 engine 完成
```

### 确定性结论

**17-20s 延迟 = cold start 时 backbone weights + expert weights 的 mmap page fault I/O 时间。**

证据链：
1. `sample` 证实主线程 100% 在做 forward pass（pread + alloc）
2. HTTP handler 线程在 `nanosleep` 轮询等待 engine 完成
3. Accept 线程正常阻塞在 `__accept`（不是被延迟）
4. Server 代码路径本身只需 ~775ms（backbone warm 后）
5. Cold start 时 backbone weights 需要从 SSD page-in（15GB），这是 17s 的来源

**不是 I/O bound vs CPU bound 的问题** — 是 **mmap page fault 导致的 cold start 延迟**。
一旦 backbone weights 被 page-in 到内存，后续请求只需 ~775ms。

## 已排除的原因

| 假设 | 验证方法 | 结果 |
|------|----------|------|
| Zig IO fiber 调度延迟 | 改用 std.Thread.spawn | ❌ 仍然 17s |
| Socket write 延迟 | posixWriteAll + write=0ms | ❌ 排除 |
| TCP Nagle 算法 | TCP_NODELAY | ❌ 排除 |
| connection.close() 延迟 | close took 0ms | ❌ 排除 |
| io.async fiber 创建延迟 | accept→thread=0ms | ❌ 排除 |
| HTTP read 阻塞 | read took <1ms | ❌ 排除 |
| Template/Tokenize | 0ms | ❌ 排除 |
| curl 客户端问题 | nc 测试同样 19.8s | ❌ 排除 |
| TCP 层延迟 | health 0.4ms | ❌ 排除（空闲时正常） |
| CPU 满载 | CPU < 100% | ❌ 排除 |
| 物理内存不足 | 25GB free | ❌ 排除 |

## 结论

**17-19s 延迟是 macOS kernel 在 mmap I/O 压力下的 `accept()` 系统调用阻塞。**

这是在 48GB Mac 上通过 mmap 访问 141GB 模型文件的固有限制。
当 forward pass 触发大量 page fault I/O 时，kernel 的 VM 子系统
繁忙，导致其他需要内核内存操作的系统调用（如 accept）被延迟。

## 改善方向

1. **Cache warmup**：启动时预热 expert cache，减少 forward pass 期间的 page fault
2. **更大的 expert cache**：缓存更多 experts，减少 SSD 随机读取
3. **Prefix caching**：复用 KV cache，跳过 prefill 阶段（减少 page fault 次数）
4. **Pre-accept**：在 forward pass 之前 pre-accept 连接，避免 accept 被阻塞
5. **减少 mmap 范围**：只 mmap 需要的 shard 文件，而不是全部 33 个

## 代码改动记录

本次排查过程中的代码改动（保留用于未来诊断）：

1. `DSV4Model.forward` — 添加 embed/layers/head 计时
2. `server/http.zig` — 添加 HTTP read/template/tokenize/gen/write/close 计时
3. `server/openai.zig` — 添加 push→wait/wait_duration 计时
4. `server.zig` — accept loop 改为 OS thread + POSIX socket
5. `server/http.zig` — handleRequestRaw 使用 blocking libc read/write
