# dmlx DeepSeek V4 Flash — Complete Analysis Report
# Date: 2026-05-03 (Revised)
# Baseline: mlx-lm `hyper_connection.py` + `deepseek_v4.py`

---

## I. 7-Prompt End-to-End Test Results

Run method: `scripts/best_test.sh` (smelt + stream, chat mode, temperature=0)

| # | Prompt | Expected | max_tokens | Status |
|---|--------|----------|-----------|--------|
| 1 | `2+2=` | `4` | 30 | ✅ |
| 2 | `The capital of France is` | `Paris` | 20 | ✅ |
| 3 | `What temperature does water freeze at in Celsius? Just give the number.` | `0` | 30 | ✅ |
| 4 | `Is the Earth round? Reply with only yes or no.` | `yes` | 30 | ✅ |
| 5 | `3*3=` | `9` | 30 | ✅ |
| 6 | `10-5=` | `5` | 30 | ✅ |
| 7 | `What is capital of France?` | `Paris` | 30 | ✅ |

**7/7 all passed, 0 skip.**

Note: In chat mode, the model outputs analysis text before the answer, so sufficient max_tokens is needed.
Prompt tokens match Python tokenizer exactly (verified: `[0, 128803, 20, 13, 20, 31, 128804, 128822]`).

---

## II. Fixed BUGs (7 items)

### P0 — Root Cause Fixes

| # | BUG | File | Vs Python | Description |
|---|-----|------|-----------|-------------|
| 1 | **mhcPost comb not transposed** | `deepseek_v4.zig` mhcPost | `comb.swapaxes(-1,-2) @ residual` | Python does comb^T @ res, Zig did comb @ res. 43 layers×2=86 accumulated errors. **Most critical fix** |
| 2 | **post_mult 1.0 → 2.0** | `deepseek_v4.zig` DSV4HyperConn.pre | `2 * mx.sigmoid(...)` | ops path passed 1.0, Metal kernel used 2.0. 86 accumulated deviations |
| 3 | **mxfp4 dequantization** | `deepseek_v4_loader.zig` | `biases==null → mxfp4` | group_size=32, mode="mxfp4". Fixed model crash/garbled output |

### P1 — Precision Fixes

| # | BUG | File | Vs Python | Description |
|---|-----|------|-----------|-------------|
| 4 | **mhcPreApplyMix missing float32** | `deepseek_v4.zig` | `(pre * y).sum()` where `y=x.astype(f32)` | Weighted sum precision loss under bfloat16 |
| 5 | **mhcPost missing float32** | `deepseek_v4.zig` | `x.astype(f32)`, `residual.astype(f32)` | matmul/combine precision loss |
| 6 | **sinkhornNormalize missing precise** | `deepseek_v4.zig` | `mx.softmax(comb, precise=True)` | Sinkhorn initial softmax precision |

### P2 — Safety/Functionality

| # | BUG | File | Description |
|---|-----|------|-------------|
| 7 | **generate max_new_tokens=0 underflow** | `deepseek_v4.zig` | `0-1` usize underflow causes infinite loop |

---

## III. Python ↔ Zig Function-by-Function Comparison

### 3.1 HyperConnection.__call__ → mhcPreNormFn + mhcPreSplitMixes

```
Python:                                    Zig:
y = x.astype(f32)                          (No explicit conversion, fn weights are f32 so matmul output is f32)
z = rms_norm(y.flatten(-2), None, eps)     Manual RMSNorm: square→sum→rsqrt→multiply
mixes = z @ self.fn.T                      mixes = res_normed @ fn^T
pre, post, comb = split_sinkhorn(mixes)    pre, post, comb = mhcPreSplitMixes(mixes)
  post = 2 * sigmoid(...)                    post = 2.0 * sigmoid(...)  ✅
  comb = softmax(precise=True) + eps         comb = softmaxPrecise + eps  ✅
  sinkhorn(comb)                             sinkhornNormalize(comb)  ✅
return (pre[...,None] * y).sum(2)          return mhcPreApplyMix(residual, pre)
  .astype(x.dtype)                           float32 promotion + astype(orig_dtype)  ✅
```

**Status**: ✅ Aligned

### 3.2 hc_expand → mhcPost

```
Python:                                    Zig:
y = post[...,None] * x[:,None,:].f32()     term1 = x_expanded_f32 * post_mix_f32  ✅
y += comb.swapaxes(-1,-2) @ res.f32()      term2 = comb_2d_t @ res_2d  ✅ (transposed)
return y.astype(x.dtype)                   return astype(result_f32, orig_dtype)  ✅
```

**Status**: ✅ Aligned

### 3.3 Chat Template

```
Python (Jinja):                            Zig:
<bos><User>content<Assistant></think>      <bos><User>content<Assistant></think>
```

Prompt tokens match exactly: `[0, 128803, 20, 13, 20, 31, 128804, 128822]`

**Status**: ✅ Aligned

### 3.4 Attention Mask

```
Python: create_attention_mask(return_array=True, window_size=128)
Zig:    "causal" string (no sliding window limit)
```

**Status**: ⚠️ Potential difference. When seq_len > sliding_window (128), Python's mask limits attention range,
Zig's "causal" does not. No impact for short prompts (8 tokens), possible differences for long text.

### 3.5 Gate (MoE Router)

```
Python: logits.astype(f32) → sqrtsoftplus → argpartition
Zig:    logits (original dtype) → sqrtsoftplus → argsort slice
```

**Status**: ⚠️ Zig does not promote logits to float32. Minor impact on short prompts, possible accumulation for long sequences.

---

## IV. Modified File List

| File | Changes |
|------|---------|
| `src/models/deepseek_v4.zig` | mhcPost comb transpose + float32, mhcPreApplyMix float32, post_mult 2.0, softmaxPrecise, generate guard |
| `src/models/deepseek_v4_loader.zig` | mxfp4 dequantization detection |
| `src/models/expert_stream.zig` | ExpertCache enabled, diagnostic log cleanup |
| `src/ops.zig` | softmaxPrecise function |
| `src/tokenizer/chat_template.zig` | thinking_mode support, official Jinja alignment |
| `src/main.zig` | --raw flag, smelt_strategy passthrough |
| `src/server.zig` | smelt_strategy configuration |
| `scripts/best_test.sh` | 7 prompt all PASS, 0 skip |

---

## V. Remaining Known Differences (not affecting 7-prompt test)

1. **Attention mask**: Zig uses `"causal"` string, Python uses `return_array=True` with sliding window. No impact on short prompts.
2. **Gate float32**: Python promotes logits to float32 in MoE gate, Zig does not.
3. **Metal kernel not used**: Python uses fused Metal kernel (`hc_sinkhorn_collapse`) on GPU, Zig only uses ops path. Functionally equivalent but performance difference.
4. **Compressor overlap**: Python's compress_ratio=4 layers use overlap compression, Zig implementation may differ (not deep-compared).

---

## VI. Verification Status

| Check Item | Status |
|------------|--------|
| `zig build` | ✅ Compiles |
| `zig build test` | ✅ All unit tests pass (exit code 0) |
| `scripts/best_test.sh` | ✅ 7/7 PASS, 0 skip |
| Prompt tokens comparison | ✅ Exactly matches Python tokenizer |
| pre-commit hook | ✅ fmt + build + test |
