# Application Scenarios — DeepSeek V4 on Small Macs

> **Core Value**: Run a 671B-parameter MoE model on a 48GB MacBook Pro. No cloud. No GPU cluster. Just your laptop.

---

## Scenario 1: Local LLM Inference

**Problem**: You need GPT-4-class intelligence but can't send sensitive data to cloud APIs.

**Solution**: mlx-zig runs DeepSeek V4 entirely on-device. All computation happens on your Mac's Metal GPU via Apple's unified memory architecture.

| Aspect | Cloud API (OpenAI/Claude) | mlx-zig Local |
|--------|--------------------------|---------------|
| Data privacy | ❌ Data leaves device | ✅ Zero network egress |
| Latency | ~500ms-2s (network) | 200-500ms TTFT, 250-500ms/tok |
| Cost | Per-token pricing | Free (your hardware) |
| Availability | Requires internet | Works offline |
| Censorship | Subject to API filtering | Full model capability |

**Hardware requirement**: Apple Silicon Mac with 48GB+ unified memory (M4 Max recommended)

---

## Scenario 2: Privacy-First Applications

**Problem**: Healthcare, legal, financial, or enterprise data cannot leave the device.

**Solution**: mlx-zig's single-binary deployment + zero network dependency means:

- **HIPAA/GDPR compliance**: No data transmission, no third-party processors
- **Air-gapped deployment**: Run on machines with no internet connection
- **Audit trail**: All inference happens locally, fully auditable
- **Data sovereignty**: Your models, your data, your hardware

```
┌─────────────────────────────────────┐
│           Your Mac                   │
│  ┌─────────────────────────────┐    │
│  │     mlx-zig binary           │    │
│  │  ┌───────────────────────┐  │    │
│  │  │  DeepSeek V4 Model    │  │    │
│  │  │  + KV Cache (local)   │  │    │
│  │  └───────────────────────┘  │    │
│  │  Zero network egress ←───→  │ 🔒 │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

---

## Scenario 3: Edge Deployment — Mac mini Inference Server

**Problem**: A small team needs shared LLM access without cloud dependency.

**Solution**: Deploy a Mac mini as a private inference server:

```bash
# Start the OpenAI-compatible server
mlx-zig server --model ~/models/deepseek-v4-flash-4bit \
  --port 8080 --kv-strategy paged

# Team members connect via standard OpenAI client
curl http://mac-mini:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4","messages":[{"role":"user","content":"Hello"}]}'
```

**Capabilities**:
- OpenAI-compatible API (drop-in replacement for any OpenAI client)
- SSE streaming for real-time token output
- Continuous batching (PagedKVCache) for concurrent requests
- Speculative decoding (PLD + EAGLE) for faster generation
- Guided decoding (JSON Schema / Regex) for structured output

**Hardware**: Mac mini M4 Pro (48-64GB) can serve 2-5 concurrent users

---

## Scenario 4: Offline / Censored-Region Access

**Problem**: Cloud LLM APIs are blocked, censored, or unavailable in certain regions.

**Solution**: mlx-zig's single binary + bundled model weights means:

1. Download model once (via any available channel)
2. Run entirely offline — no API key, no network call
3. Full model capability without content filtering
4. Works on trains, planes, remote locations, secure facilities

```bash
# One-time setup
brew install mlx-c
git clone https://github.com/zouyee/mlx-zig.git
cd mlx-zig && zig build

# Run anywhere, anytime — no internet needed
./zig-out/bin/mlx-zig chat --model /path/to/model --prompt "Hello"
```

---

## Scenario 5: Development & Testing Without GPU Clusters

**Problem**: LLM application development requires expensive GPU instances for testing.

**Solution**: Develop and test locally on your Mac:

| Task | Cloud GPU | mlx-zig Local |
|------|-----------|---------------|
| Prompt engineering | Deploy → test → redeploy | Instant iteration |
| Model evaluation | Reserve A100 ($3+/hr) | Your Mac (free) |
| Integration testing | Network-dependent | Local, deterministic |
| CI/CD | Expensive GPU runners | Mac runners (or mock) |

**Workflow**: Develop locally → validate → deploy to production (optional cloud scale-up)

---

## Scenario 6: Research & Experimentation

**Problem**: Researchers need to experiment with MoE routing, KV cache strategies, and quantization without cloud overhead.

**Solution**: mlx-zig exposes the full stack for experimentation:

- **MoE routing**: Modify top-k, routing bias, expert selection in `moe_router.zig`
- **KV cache strategies**: Swap between 6 strategies at runtime via VTable polymorphism
- **Quantization**: Test INT4, MXFP4, TurboQuant on real hardware
- **Speculative decoding**: Compare PLD vs EAGLE draft strategies
- **Guided decoding**: Build custom FSMs for structured generation

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

## Why mlx-zig for These Scenarios

| Requirement | How mlx-zig Delivers |
|-------------|---------------------|
| **Small memory** | 5-layer optimization: streaming + SMELT + MLA + quantization + tiered KV |
| **No cloud** | Single static binary, zero network dependency |
| **Privacy** | All computation on-device, Metal GPU, unified memory |
| **Deterministic latency** | No GC, Zig compile-time guarantees |
| **Apple Silicon native** | Direct Metal/Accelerate linking, no Python/CFFI overhead |
| **Production-ready** | OpenAI-compatible API, SSE streaming, continuous batching |
| **Extensible** | Full source access, compile-time specialization, VTable polymorphism |

---

## Getting Started

1. [Installation](../user-guide/) — Set up mlx-zig on your Mac
2. [DeepSeek V4 Quick Fix](../user-guide/deepseek-v4-quickfix.md) — Fix garbled output
3. [DeepSeek MoE Deep Dive](../deepseek-moe/README.md) — How it works technically
4. [Benchmarks](../technical/benchmarks.md) — Performance data
5. [Performance Roadmap](../technical/performance-roadmap.md) — Future improvements
