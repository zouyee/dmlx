# Troubleshooting: DeepSeek V4

**Component**: DeepSeek V4 Model (`src/models/deepseek_v4.zig`, `src/models/deepseek_v4_loader.zig`)
**Spec**: `.kiro/specs/production-deployment/design-deepseek-v4.md`
**Properties**: Property V1 (Dequantization Parity), Property V2 (Attention Dispatch)

## Symptom: Model generates garbled / random characters

**First seen**: 2026-04-29 (Commit `a18bc24`, `e11c3b9`)
**Root cause**: Chat template using full-width Unicode characters (`｜` U+FF5C, `▁` U+2581) instead of ASCII (`|`, `_`), causing the tokenizer to split special tokens into sub-tokens.

### Spec References

- `design-deepseek-v4.md` §1 — Weight Loading (indirect: tokenizer config is part of model loading)
- `requirements.md` R6.2 — Model Registry lookup

### Code Checkpoints

1. `src/tokenizer/chat_template.zig` — Special token definitions
2. `src/main.zig` — BOS token validation (expected: 100000)
3. `src/tokenizer/bpe.zig` — Sub-token splitting logic

### Diagnosis Steps

1. Print first 10 token IDs of the prompt. Expected: `[100000, ...]` (single BOS)
2. If BOS is split into multiple tokens → check chat template Unicode characters
3. Verify special tokens use ASCII: `<|begin_of_sentence|>`, `<|User|>`, `<|Assistant|>`

### Fix Summary

Correct special tokens from full-width to ASCII format:
```diff
- .bos_token = "<｜begin▁of▁sentence｜>",
+ .bos_token = "<|begin_of_sentence|>",
```

### Prevention

- `src/tests/chat_template_tests.zig` — Unit tests verify tokenization round-trip
- `scripts/verify-deepseek-v4-fix.sh` — Automated verification script

---

## Symptom: `mhcPreNormFn` crash with reshape error

**First seen**: 2026-04-27
**Root cause**: `hc_head.fn` weight not consumed by loader (appeared as "Unused weight"). Additionally, all `.scales`/`.biases` quantized metadata keys were unused.

### Spec References

- `design-deepseek-v4.md` §1 — Weight Loading & Sanitization
- Property V1: FP4/FP8 Dequantization Parity

### Code Checkpoints

1. `src/models/deepseek_v4_loader.zig:mapV4LayerWeight` — Weight name mapping
2. `src/models/deepseek_v4_loader.zig:buildDSV4Model` — `hc_head` loading block
3. `src/models/deepseek_v4_loader.zig:consumeWeightAndMeta` — Quantized metadata consumption

### Diagnosis Steps

1. Run with verbose logging — check for "Unused weight" warnings
2. If `hc_head.*` or `*.scales`/ `*.biases` appear as unused → loader is not consuming them
3. Check `consumeWeightAndMeta()` is applied to all weight loading paths

### Fix Summary

- Created `consumeWeightAndMeta()` helper that removes `.weight`, `.scales`, `.biases` together
- Applied to all weight loading in `buildDSV4Model`
- Fixed `hc_head` weight name mapping (`hc_head.fn` vs `hc_head.fn_weight`)

### Prevention

- Loader should fail on any remaining "Unused weight" at end of load (strict mode)
- Add unit test: after `buildDSV4Model`, HashMap should be empty

---

## Symptom: OOM when loading DeepSeek V4 Flash 4-bit on 48GB Mac

**First seen**: 2026-04-27
**Root cause**: Eager dequantization of expert weights (4-bit → float16) expanded memory 4×. mlx-c's `mlx_load_safetensors` loads full shard data eagerly, unlike Python's lazy memory-mapped arrays.

### Spec References

- `design-deepseek-v4.md` §1 — Weight Loading
- `design.md` §6.3 — Memory Limiter

### Code Checkpoints

1. `src/models/deepseek_v4_loader.zig:loadShardedWeights` — Shard loading strategy
2. `src/main.zig:runDeepSeekV4Chat` — MLX memory limits (`mlx_set_wired_limit`)
3. `src/models/deepseek_v4_loader.zig:buildDSV4Model` — Dequantization calls

### Diagnosis Steps

1. Profile RSS during `buildDSV4Model`. Expected: ~400MB with lazy loading.
2. If RSS > 10GB → check for eager `dequantIfNeeded` calls
3. Check `wired_limit` and `cache_limit` are set (50% and 25% of system memory)

### Fix Summary

- Removed bfloat16→float32 conversion in `loadShardedWeights`
- Removed eager dequantization in `splitFusedExperts`
- Set `wired_limit` to 50% system memory, `cache_limit` to 25%
- Switched to `c_allocator` instead of `DebugAllocator`

### Prevention

- Add memory benchmark: RSS during model load must be < 1GB (without weights materialized)
- Document: "Never call astype/dequantize during load unless absolutely required"

---

## Symptom: Attention dequantize + forward is decode bottleneck (~10min+ per token)

**First seen**: 2026-04-28
**Status**: PARTIALLY RESOLVED — blocked on full expert loading
**Root cause**: Attention weights dequantized on every forward pass instead of keeping quantized + using `quantizedMatmul`.

### Spec References

- `design-deepseek-v4.md` §2 — DSV4Attention
- `design-deepseek-v4.md` §1 — Quantization Config

### Code Checkpoints

1. `src/models/deepseek_v4.zig:DSV4Attention.forward` — Matmul calls for wq_a, wq_b, wkv, wo_b
2. `src/models/deepseek_v4_loader.zig` — Whether attention weights are stored as `QuantizedWeight` or `Array`
3. `src/quantize.zig:quantizedMatmul` — Fused dequantize+matmul availability

### Diagnosis Steps

1. Profile with Metal System Trace — identify kernel launch count per decode step
2. If attention matmul shows ~10 kernel launches per weight → using plain matmul on packed weights
3. Check if `DSV4Attention` fields are `?quantize_mod.QuantizedWeight` (should be) vs plain `Array`

### Fix Summary (Partial)

- `wq_a`, `wq_b`, `wkv`, `wo_b` dequantized at load time as temporary fix
- Long-term: Use `quantizedMatmul` in attention forward to avoid dequantize overhead
- Task 51 in tasks.md tracks this optimization

### Prevention

- Benchmark: decode step must complete in < 500ms per token on target hardware
- Add performance regression test in CI

---

## Symptom: `wo_a` dequantize produces wrong shape

**First seen**: 2026-04-28
**Root cause**: `mlx_dequantize` on lazy `wo_a` weight `[8192, 512]` uint32 returns array with size 4194304 instead of expected `[8192, 4096]` (unpacked).

### Spec References

- `design-deepseek-v4.md` §2 — Grouped Output Projection
- Property V1: FP4/FP8 Dequantization Parity

### Code Checkpoints

1. `src/models/deepseek_v4_loader.zig:buildDSV4Model` — wo_a loading block
2. Check `wo_a_raw.shape()`, `wo_a_raw.dtype()`, `wo_a_deq.shape()`

### Diagnosis Steps

1. Print packed weight shape and dtype: expected `[8192, 512]` uint32 (affine 4-bit, group_size=64)
2. Print dequantized shape: expected `[8192, 4096]` bfloat16
3. If shape mismatch → check dequantize parameters (group_size, bits, mode) match actual quantization

### Fix Summary

- Root cause was lazy array shape metadata mismatch
- Workaround: forced evaluation before reshape
- Proper fix: ensure `mlx_dequantize` parameters match checkpoint format exactly

### Prevention

- Add shape assertion after every dequantize call in loader
- Golden test: load one layer, verify all weight shapes match expected

---

## Symptom: Model generates wrong tokens (e.g., `2+2=` → `"That's a classic!"` instead of `4`)

**First seen**: 2026-05-01
**Status**: RESOLVED
**Root cause**: `dequantIfNeeded` and embed/lm_head dequantize in `deepseek_v4_loader.zig` hardcoded quantization mode `"affine"` with `group_size=64`. DeepSeek V4 Flash 4-bit uses **mxfp4** (no biases, `group_size=32`) for expert weights and embeddings. MLX interpreted packed uint32 weight data with wrong dequantization parameters, producing garbage embeddings that propagated through all 43 layers.

### Spec References

- `design-deepseek-v4.md` §1 — Weight Loading & Quantization
- Property V1: FP4/FP8 Dequantization Parity

### Code Checkpoints

1. `src/models/deepseek_v4_loader.zig:dequantIfNeeded` — Dequantization mode detection
2. `src/models/deepseek_v4_loader.zig:buildDSV4Model` — Embed/lm_head dequantize block
3. `src/models/deepseek_v4.zig:DSV4Expert.expertMatmul` — `quant_mode` field usage
4. `src/models/deepseek_v4.zig:DSV4Attention` — `attn_quant_mode` field usage

### Diagnosis Steps

1. Check quantization config from `config.json`: `mode="mxfp4"`, `group_size=32`, no `bias` present
2. Verify `dequantIfNeeded` detects mxfp4 vs affine correctly:
   - mxfp4: `biases == null` → `group_size=32`, `mode="mxfp4"`
   - affine: `biases != null` → `group_size=64`, `mode="affine"`
3. Check `DSV4Expert.quant_mode` and `DSV4Attention.attn_quant_mode` are set from loader config, not hardcoded `.affine`
4. If model generates plausible-looking but semantically wrong text → suspect dequantization mismatch (not routing or attention bug)

### Fix Summary

- **`dequantIfNeeded`**: Auto-detect mxfp4 vs affine based on `biases == null`:
  ```zig
  const is_mxfp4 = biases == null;
  const gs = if (is_mxfp4) 32 else config.quantize_default_group_size;
  try c.check(c.c.mlx_dequantize(&res, weight.inner, scales.inner, biases_inner, opt_group, opt_bits,
      if (is_mxfp4) "mxfp4" else "affine", null_array, no_dtype, ctx.stream.inner));
  ```
- **Embed/lm_head dequantize**: Same auto-detection applied.
- **`DSV4Expert`**: Added `quant_mode: quantize_mod.QuantMode` field (default `.affine` → now set from loader).
- **`DSV4Attention`**: Added `attn_quant_mode` field, passed to all 6 `quantizedMatmul` calls.

### Prevention

- **Never hardcode quantization mode**: Always derive from checkpoint metadata (`biases` presence, `config.json` fields).
- **Add golden test**: Load a single expert weight, dequantize, and compare first few values with Python reference.
- **Add integration test**: Prompt `"2+2="` must generate token sequence containing the correct answer token.

### Related Notes

- **CORRECTNESS TEST pitfall**: The stream-mode `CORRECTNESS TEST` uses `TensorIndex.loadTensor` which loads raw bytes without calling `dequantIfNeeded`. It passed even while the actual model forward was broken because it tested a different code path. Always verify the production forward path.
- **STREAM WEIGHTS TEST discrepancy**: Stream-loaded sliced weights (`readExpertRows`/`takeAxis`) produce slightly different `gatherQmm` output than full weights (~2× difference in first element). This is a secondary precision issue that does not block correct generation but may warrant future investigation.
