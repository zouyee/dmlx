# Chapter 7: Testing System and Quality

## 7.1 Test Module Panorama

`tests.zig` registers 50+ test modules, grouped by category:

### Operation Tests (Core Correctness)
- `core_tests`: basic Array operations
- `comparison_tests`, `math_tests`, `shape_tests`, `reduce_tests`, `sort_tests`
- `creation_tests`, `random_tests`, `linalg_tests`, `fft_tests`

### Model and Inference Tests (Functional Verification)
- `e2e_tests` (302 lines): tiny random model forward + generate + GQA
- `deepseek_v4_tests` (611 lines): `compressKV` various modes, slice operations
- `generation_tests`, `speculative_tests`, `guided_tests`

### Numerical Equivalence Tests (Precision Verification)
- `numerical_equivalence_test.zig` (814 lines): **property tests**, 100 iterations
  - RMSNorm: cosine similarity ≥ 0.9999
  - RoPE, SDPA, Embedding, LSTM, GRU, various loss functions
  - Compared against Python MLX reference output

### MoE and Expert Tests
- `expert_remap_tests`, `expert_cache_tests`, `expert_stream_tests`
- `moe_router_tests`: top-k routing correctness

### Integration Tests
- `cache_integration_tests`, `integration_tests`
- `model_smoke_tests`, golden tests

### Infrastructure Tests
- `kvcache_tests`, `tiered_kvcache_tests`, `prefix_disk_tests`
- `memory_tests`, `memory_property_tests`
- `arena_tests`: ScopedArrayArena functional verification

## 7.2 Test Quality Assessment

### Strengths

- **Property tests** (100 random input iterations) are more reliable than single-point tests
- **Numerical equivalence tests** use cosine similarity thresholds:
  - float32: 0.9999
  - int8: 0.99
  - int4: 0.95
- E2E tests include tiny model forward + generate + KV cache combined verification
- DeepSeek V4 has dedicated unit tests

### Gaps

| Gap | Severity | Description |
|------|--------|------|
| No `nn_tests` | P1 | Linear/BatchNorm/LSTM/GRU/RNN/MultiHeadAttention have no direct tests |
| No `grad_tests` | P1 | Automatic differentiation correctness not directly verified |
| No real-weight golden test | P1 | All model tests use random weights |
| `trainer_tests` possibly skeleton | P2 | Needs verification whether actual training loop is included |

## 7.3 E2E Test Empirical Analysis

`e2e_tests.zig` uses a tiny model (128 vocab / 32 hidden / 2 layer / 4 heads):

```zig
const config = LlamaConfig{
    .vocab_size = 128,
    .hidden_size = 32,
    .num_hidden_layers = 2,
    .num_attention_heads = 4,
    // ...
};
```

Verification content:
- Forward output shape `[batch, seq_len, vocab]`
- Generate produces correct length
- Generation results with/without KV cache are consistent

**Limitation**: tiny models cannot expose numerical issues of large models (FP16 overflow, quantization error accumulation).

## 7.4 Performance Benchmark (`benchmark_run.log`)

```
⚡ DeepSeek V4 Performance Regression Test
Model: DeepSeek-V4-Flash-4bit
Absolute thresholds: TTFT ≤ 500ms | ITL ≤ 150ms | TPS ≥ 5
Regression threshold: +20% vs baseline
```

The current performance regression framework is established, but the log does not show complete run results.
