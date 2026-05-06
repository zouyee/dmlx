# 应用场景 —— 小内存 Mac 上的 DeepSeek V4

> **核心价值**：在 48GB MacBook Pro 上运行 284B 参数的 MoE 模型。无需云端。无需 GPU 集群。只需你的笔记本。

---

## 场景一：本地 LLM 推理

**问题**：你需要 GPT-4 级别的智能，但不能将敏感数据发送到云端 API。

**解决方案**：dmlx 完全在设备上运行 DeepSeek V4。所有计算在你的 Mac 的 Metal GPU 上通过 Apple 的统一内存架构完成。

| 方面 | 云端 API (OpenAI/Claude) | dmlx 本地 |
|--------|--------------------------|---------------|
| 数据隐私 | ❌ 数据离开设备 | ✅ 零网络出站 |
| 延迟 | ~500ms-2s（网络） | 200-500ms TTFT, 250-500ms/tok |
| 成本 | 按 token 定价 | 免费（你的硬件） |
| 可用性 | 需要互联网 | 离线可用 |
| 审查 | 受 API 过滤限制 | 完整模型能力 |

**硬件要求**：Apple Silicon Mac，48GB+ 统一内存（推荐 M4 Max）

---

## 场景二：隐私优先应用

**问题**：医疗、法律、金融或企业数据不能离开设备。

**解决方案**：dmlx 的单二进制部署 + 零网络依赖意味着：

- **HIPAA/GDPR 合规**：无数据传输，无第三方处理器
- **物理隔离部署**：在无互联网连接的机器上运行
- **审计追踪**：所有推理在本地进行，完全可审计
- **数据主权**：你的模型，你的数据，你的硬件

```
┌─────────────────────────────────────┐
│           Your Mac                   │
│  ┌─────────────────────────────┐    │
│  │     dmlx binary           │    │
│  │  ┌───────────────────────┐  │    │
│  │  │  DeepSeek V4 Model    │  │    │
│  │  │  + KV Cache (local)   │  │    │
│  │  └───────────────────────┘  │    │
│  │  Zero network egress ←───→  │ 🔒 │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

---

## 场景三：边缘部署 —— Mac mini 推理服务器

**问题**：一个小团队需要共享 LLM 访问，但不能依赖云端。

**解决方案**：将 Mac mini 部署为私有推理服务器：

```bash
# Start the OpenAI-compatible server
dmlx server --model ~/models/deepseek-v4-flash-4bit \
  --port 8080 --kv-strategy paged

# Team members connect via standard OpenAI client
curl http://mac-mini:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4","messages":[{"role":"user","content":"Hello"}]}'
```

**能力**：
- OpenAI 兼容 API（任何 OpenAI 客户端的直接替代）
- SSE 流式传输，实时 token 输出
- 连续批处理（PagedKVCache）支持并发请求
- 推测解码（PLD + EAGLE）加速生成
- 引导解码（JSON Schema / Regex）输出结构化结果

**硬件**：Mac mini M4 Pro（48-64GB）可服务 2-5 个并发用户

---

## 场景四：离线 / 受限区域访问

**问题**：云端 LLM API 在某些地区被屏蔽、审查或不可用。

**解决方案**：dmlx 的单二进制 + 内置模型权重意味着：

1. 一次性下载模型（通过任何可用渠道）
2. 完全离线运行——无需 API key，无需网络调用
3. 完整模型能力，无内容过滤
4. 适用于火车、飞机、偏远地区、安全设施

```bash
# One-time setup
brew install mlx-c
git clone https://github.com/zouyee/dmlx.git
cd dmlx && zig build

# Run anywhere, anytime — no internet needed
./zig-out/bin/dmlx chat --model /path/to/model --prompt "Hello"
```

---

## 场景五：无需 GPU 集群的开发与测试

**问题**：LLM 应用开发需要昂贵的 GPU 实例进行测试。

**解决方案**：在你的 Mac 上本地开发与测试：

| 任务 | 云端 GPU | dmlx 本地 |
|------|-----------|---------------|
| Prompt 工程 | 部署 → 测试 → 重新部署 | 即时迭代 |
| 模型评估 | 预留 A100 ($3+/hr) | 你的 Mac（免费） |
| 集成测试 | 依赖网络 | 本地，确定性 |
| CI/CD | 昂贵的 GPU runner | Mac runner（或 mock） |

**工作流**：本地开发 → 验证 → 部署到生产环境（可选云端扩展）

---

## 场景六：研究与实验

**问题**：研究人员需要实验 MoE 路由、KV Cache 策略和量化，但不能有云端开销。

**解决方案**：dmlx 暴露完整技术栈用于实验：

- **MoE 路由**：在 `moe_router.zig` 中修改 top-k、路由偏置、专家选择
- **KV Cache 策略**：通过 VTable 多态在运行时切换 6 种策略
- **量化**：在真实硬件上测试 INT4、MXFP4、TurboQuant
- **推测解码**：比较 PLD 与 EAGLE 草稿策略
- **引导解码**：为结构化生成构建自定义 FSM

```zig
// Example: Swap KV cache strategy at runtime
const strategy = switch (config.mode) {
    .long_context => KVCacheStrategy.initRotating(allocator, window_size),
    .multi_user => KVCacheStrategy.initPaged(allocator, page_size, num_pages),
    .low_memory => KVCacheStrategy.initPagedQuantized(allocator, page_size, num_pages),
    else => KVCacheStrategy.initStandard(allocator, max_len),
};
```

---

## 为何这些场景选择 dmlx

| 需求 | dmlx 如何满足 |
|-------------|---------------------|
| **小内存** | 五层优化：streaming + SMELT + MLA + 量化 + 分层 KV |
| **无需云端** | 单个静态二进制，零网络依赖 |
| **隐私** | 所有计算在设备上，Metal GPU，统一内存 |
| **确定性延迟** | 无 GC，Zig 编译期保证 |
| **Apple Silicon 原生** | 直接 Metal/Accelerate 链接，无 Python/CFFI 开销 |
| **生产就绪** | OpenAI 兼容 API，SSE 流式，连续批处理 |
| **可扩展** | 完整源码访问，编译期特化，VTable 多态 |

---

## 入门指南

1. [安装](../user-guide/) — 在你的 Mac 上设置 dmlx
2. [DeepSeek V4 快速修复](../user-guide/deepseek-v4-quickfix.md) — 修复乱码输出
3. [DeepSeek MoE 深度解析](../deepseek-moe/README.md) — 技术原理
4. [性能基准](../technical/benchmarks.md) — 性能数据
5. [项目路线图](../../ROADMAP.md) — 未来改进
