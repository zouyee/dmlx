# Executive Summary

> **Version**: Based on codebase current HEAD (v0.3.0-mlx-c)  
> **Analysis Rounds**: Three rounds of progressive deep analysis  
> **Coverage**: ~52,933 lines of Zig source (including tests), 50+ test modules, 25 design documents  
> **Analysis Date**: 2026-05-03  
> **Target Platform**: macOS Apple Silicon

---

## Project Positioning

dmlx is a full-stack LLM inference and training system built on Apple MLX C bindings (`mlx-c`), written in Zig and targeting macOS Apple Silicon. The project has undergone a significant transition from prototype to production-grade, currently reaching feature depth comparable to Python vLLM/mlx-lm.

## Core Conclusions

1. **High engineering maturity**: 350 tests all passing, Phase 0–7 roadmap fully completed, clean code structure, comprehensive documentation system
2. **Complete cutting-edge features**: DeepSeek V4 (3,091 lines), speculative decoding (PLD+EAGLE), guided decoding (JSON Schema/Regex), MoE routing, QLoRA, TurboQuant all implemented
3. **Technical debt remains**: 34 `dataSliceMut` calls in `nn.zig` not fully removed; `insertion` sort performance bottleneck in `sampling.zig`; type safety vulnerability in `prompt_cache.zig`
4. **Highest risk**: `prompt_cache.zig` uses `@ptrCast` to forcefully cast the runtime-polymorphic `KVCacheStrategy` — this causes crashes in Paged/Quantized/Tiered modes

## Project Scale

| Metric | Value |
|------|------|
| Total lines of code | ~52,933 lines |
| Source lines (excluding tests) | ~42,455 lines |
| Test modules | 50+ |
| Passing tests | 350 |
| Largest source file | `models/deepseek_v4.zig` (3,091 lines) |
| Documentation files | 25 |
| External dependencies | `mlx-c` (C library), `zig_regex` (Zig package) |

## Top 5 Critical Findings

| Rank | Issue | Severity | Location |
|------|------|--------|------|
| 1 | Prompt Cache type safety vulnerability: force-casting `PagedKVCache` as `StandardKVCache` | **P0** | `prompt_cache.zig:74` |
| 2 | 34 `dataSliceMut` in NN layers: pure CPU scalar loops bypass GPU | **P0** | `ops/nn.zig` |
| 3 | Sampling uses insertion sort for 128K vocab, ~8.2 billion comparisons/token | **P1** | `sampling.zig` (4 locations) |
| 4 | Batched forward not integrated into engine loop | **P1** | `server.zig` |
| 5 | AdamW creates ~3000 temporary mlx_array per step | **P1** | `optim.zig` |

## Subsystem Maturity at a Glance

| Subsystem | Maturity | Key Risk |
|--------|--------|---------|
| DeepSeek V4 Model | ⭐⭐⭐⭐⭐ | Extremely complex but verified |
| KV Cache (6 strategies) | ⭐⭐⭐⭐☆ | prompt_cache type vulnerability |
| Server | ⭐⭐⭐⭐☆ | Batch not integrated |
| Speculative/Guided Decoding | ⭐⭐⭐⭐⭐ | EAGLE single-token draft only |
| NN Layers (`ops/nn.zig`) | ⭐⭐☆☆☆ | **Largest technical debt** |
| Sampling | ⭐⭐⭐☆☆ | insertion sort |
| Distributed Inference | ⭐⭐⭐☆☆ | deinit is empty |
