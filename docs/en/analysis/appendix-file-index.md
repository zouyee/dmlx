# Appendix: Key Files and Documentation Index

## A.1 Key Source File Index

| File | Lines | Responsibility | Risk Level |
|------|-------|---------------|------------|
| `src/models/deepseek_v4.zig` | 3,091 | DeepSeek V4 complete implementation (MLA+MoE+YARN+mHC) | Medium (2 @constCast) |
| `src/models/deepseek_v4_loader.zig` | 2,071 | V4 weight loader (sharding+dequantization+expert strategy) | Low |
| `src/main.zig` | 1,764 | CLI entry point (chat/serve/benchmark/quantize/lora-train) | Low |
| `src/server.zig` | 1,517 | HTTP server (OpenAI-compatible+SSE+tool calling) | Medium (batch not integrated) |
| `src/ops/nn.zig` | 1,354 | NN layers (Linear/BatchNorm/LSTM/GRU/RNN/Attention) | **High** (34 dataSliceMut) |
| `src/speculative.zig` | 1,223 | Dual-track speculative decoding (PLD+EAGLE) | Low |
| `src/kvcache/paged.zig` | 1,152 | Paged KV Cache (BlockManager+CoW+prefix hashing) | Low |
| `src/guided.zig` | 1,129 | Guided decoding FSM (JSON Schema/Regex constraints) | Low |
| `src/io/safetensors_reader.zig` | 1,045 | Safetensors random-access reader (pread zero-copy) | Low |
| `src/quantize.zig` | 872 | Quantization infrastructure (affine/mxfp4/nvfp4/mxfp8/turboquant) | Low |
| `src/qlora.zig` | 773 | QLoRA quantized+low-rank adapter training | Low |
| `src/tokenizer/bpe.zig` | 744 | BPE Tokenizer (HF tokenizer.json format) | Low |
| `src/models/llama.zig` | 725 | LLaMA/Mistral/Qwen/Gemma/Phi standard architecture | Low |
| `src/models/minimax.zig` | 712 | MiniMax model adapter | Medium (4 @constCast) |
| `src/prompt_cache.zig` | 563 | Prompt cache persistence (safetensors) | **High** (type vulnerability) |
| `src/distributed.zig` | 222 | Distributed inference (multi-Mac tensor parallelism) | Medium (empty deinit) |
| `src/optim.zig` | 217 | AdamW optimizer | Medium (temporary object storm) |
| `src/c.zig` | ~200 | C binding layer (mlxErrorHandler+check+type re-export) | Low |

## A.2 Documentation Directory Index

| Document | Type | Content |
|----------|------|---------|
| `docs/deep-analysis.md` | Audit report | v0.3.0 self-audit, P0-P3 issue list |
| `docs/production-roadmap.md` | Roadmap | Phase 0-7 progress tracking, Task 13-34 completion status |
| `.kiro/specs/production-deployment/design.md` | Design doc | Six-layer architecture design, Mermaid diagrams |
| `.kiro/specs/production-deployment/design-paged-kv-cache.md` | Design doc | PagedKVCache algorithm details (updateAndFetch six-step method) |
| `.kiro/specs/production-deployment/design-server.md` | Design doc | Server architecture, request flow, Scheduler design |
| `docs/ecosystem-analysis.md` | Research | Five-project comparison: vLLM/mlx-lm/oMLX/TileKernels/mlx-rs |
| `docs/BENCHMARK.md` | Performance | Benchmark methodology and threshold definitions |
| `docs/DEEPSEEK-V4-FIX-PLAN.md` | Fix plan | V4 issue diagnosis and fix plan |
| `docs/FIX-REPORT-DEEPSEEK-V4.md` | Fix report | V4 fix verification results |
| `docs/tilekernels-analysis.md` | Research | TileKernels operator fusion and quantization analysis |
| `docs/deepseek-v4-optimization-plan.md` | Optimization | V4 performance optimization strategy |
| `docs/DEEPSEEK-V4-CHAT-ANALYSIS.md` | Analysis | V4 Chat mode behavior analysis |
| `docs/competitive-advantages.md` | Strategy | Project competitive advantage analysis |

## A.3 Build & Dependencies

### External Dependencies

- `mlx-c`: Apple MLX C API bindings
  - Detection priority: `-Dmlx_prefix` > `MLX_C_PREFIX` env > `pkg-config --variable=prefix mlxc` > `/opt/homebrew`
- `zig_regex`: Regular expression library (Zig package, fixed hash, not main branch)

### Build Artifacts

- `libmlx-zig.a`: Static library
- `mlx-zig`: CLI tool
  - `chat`: Interactive chat
  - `serve`/`server`: HTTP service
  - `benchmark`: Performance benchmark
  - `quantize`: Weight quantization
  - `lora-train`: LoRA fine-tuning
  - `convert`: Format conversion (TODO)
  - `evaluate`: Perplexity evaluation
- `example`: Example program
- `test`: Test runner (50+ modules, 350 tests)

### Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| macOS Apple Silicon | ✅ Primary | Metal GPU + UMA unified memory |
| macOS Intel | ⚠️ May work | No Metal, falls back to CPU |

## A.4 Code Style (`.sisyphus/zig-style.md`)

- **Naming**: PascalCase (types), camelCase (functions), snake_case (constants)
- **Forbidden**: `// ====` separators, `//!` module docs, `as any`, empty catch blocks
- **Required**: `zig build` passes, `zig build test` passes, `zig fmt src/` formatted
- **License**: AGPL-3.0

## A.5 Version Info

```zig
// src/root.zig
pub const version = "0.3.0-mlx-c";
```
