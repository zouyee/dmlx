# Python mlx-lm vs Zig mlx-zig Item-by-Item Comparison
# Date: 2026-05-03

---

## I. Chat Template Comparison

### Python (chat_template.jinja + deepseek_v32.py)

For a single user message `"2+2="`, thinking_mode="chat":

```
<｜begin▁of▁sentence｜><｜User｜>2+2=<｜Assistant｜></think>
```

Key behaviors:
- When `add_generation_prompt=True`, if the last message is from user, Jinja template does **not** append `<｜Assistant｜></think>` (because user message already appends it)
- Python `deepseek_v32.py`'s `apply_chat_template` with `add_generation_prompt=True` and last message is user, it `removesuffix("<｜Assistant｜><think>")` (only in thinking mode)

### Zig (chat_template.zig)

```zig
// add_generation_prompt: only when last message is NOT user
if (add_generation_prompt and ... and !std.mem.eql(u8, messages[messages.len - 1].role, "user")) {
    try result.appendSlice(self.allocator, "<｜Assistant｜></think>");
}
```

**Issue**: Zig calls `chat_template.apply(messages.items, true)` in `main.zig` with `add_generation_prompt=true`.
But since the last message is from user, this `add_generation_prompt` branch does not execute.
The user message itself already appends `</think>`, so the final result is correct:

```
<｜begin▁of▁sentence｜><｜User｜>2+2=<｜Assistant｜></think>
```

**Conclusion**: ✅ Chat template behavior consistent.

---

## II. HyperConnection (mHC) Comparison

### 2.1 HyperConnection.__call__ (pre phase)

**Python** (`hyper_connection.py`):
```python
def __call__(self, x: mx.array):
    B, L, H, D = x.shape
    y = x.astype(mx.float32)                          # ← float32 promotion
    z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
    mixes = z @ self.fn.T
    # calls _hc_ops or _hc_kernel
```

**Zig** (`deepseek_v4.zig` mhcPreNormFn):
```zig
// No explicit astype(float32), operates directly on input dtype
const res_flat = try ops.reshape(ctx, residual, ...);
// RMSNorm manually implemented, no prior float32 conversion
```

**Difference**: Python immediately converts x to float32 when entering HyperConnection, Zig does not.
But in Zig's mhcPreNormFn, RMSNorm is manually implemented (square → sum → rsqrt),
and fn weights themselves are float32, so matmul results are also float32.
**Partially aligned**, but input residual is not first converted to float32.

### 2.2 _hc_split_sinkhorn_ops (split + sinkhorn)

**Python**:
```python
@mx.compile
def _hc_split_sinkhorn_ops(mixes, scale, base, hc_mult, sinkhorn_iters, eps):
    mixes = mixes.astype(mx.float32)
    scale = scale.astype(mx.float32)
    base = base.astype(mx.float32)
    pre_scale, post_scale, comb_scale = scale[0], scale[1], scale[2]

    pre = mx.sigmoid(mixes[..., :hc_mult] * pre_scale + base[:hc_mult]) + eps
    post = 2 * mx.sigmoid(                                                    # ← 2.0
        mixes[..., hc_mult:2*hc_mult] * post_scale + base[hc_mult:2*hc_mult]
    )
    comb = ... * comb_scale + base[2*hc_mult:].reshape(hc_mult, hc_mult)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps                       # ← precise=True
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb
```

**Zig** (`mhcPreSplitMixes` + `sinkhornNormalize`):
- `post_mult_value`: was `1.0`, **fixed to `2.0`** ✅
- `sinkhornNormalize`: was `ops.softmax`, **fixed to `ops.softmaxPrecise`** ✅
- **But**: Python's sinkhorn does softmax first, then col norm, then loops (row norm → col norm)
  Zig's sinkhorn also does softmax → col norm → loop (row norm → col norm), **consistent** ✅

### 2.3 _hc_ops (collapse / pre_apply_mix)

**Python**:
```python
def _hc_ops(x, y, mixes, scale, base, hc_mult, sinkhorn_iters, eps):
    pre, post, comb = _hc_split_sinkhorn_ops(...)
    return (pre[..., None] * y).sum(axis=2).astype(x.dtype), post, comb
    #                       ↑ y is x.astype(float32)
```

**Zig** (`mhcPreApplyMix`):
```zig
// Fixed: added float32 promotion
const res_f32 = try ops.astype(ctx, residual, .float32);
const mix_f32 = try ops.astype(ctx, mix, .float32);
const weighted = try ops.multiply(ctx, res_f32, mix_f32);
const summed = try reduce_mod.sumAxis(ctx, weighted, 2, false);
return ops.astype(ctx, summed, orig_dtype);
```

**Conclusion**: ✅ Aligned.

### 2.4 hc_expand (post phase)

**Python**:
```python
@mx.compile
def _hc_expand_op(x, residual, post, comb):
    y = post[..., None] * x[:, :, None, :].astype(mx.float32)     # ← float32
    y = y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))  # ← float32
    return y.astype(x.dtype)
```

Note: Python uses `comb.swapaxes(-1, -2)` i.e. **comb transposed**!

**Zig** (`mhcPost`):
```zig
// Fixed: added float32 promotion
// However:
const comb_2d = try ops.reshape(ctx, comb_mix_f32, ...);
const term2_2d = try ops.matmul(ctx, comb_2d, res_2d);  // ← comb NOT transposed!
```

**⚠️ Critical difference**: Python does `comb.swapaxes(-1, -2) @ residual`,
Zig does `comb @ residual` (no transpose).
This means Zig's mhcPost computes `comb @ res` instead of `comb^T @ res`.

### 2.5 Metal Kernel Comparison

**Python Metal kernel** (`hc_sinkhorn_collapse_kernel`):
```metal
float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));  // ← 2.0
```

**Zig Metal kernel** (`hc_split_sinkhorn_metal_source`):
```metal
*(device float4*)post_out = 2.0f * 1.0f / (1.0f + metal::fast::exp(-z));  // ← 2.0
```

**Consistent** ✅ (both are 2.0 * sigmoid)

---

## III. Attention Mask Comparison

**Python**:
```python
mask = create_attention_mask(
    h[:, :, 0, :],
    mask_cache,
    window_size=self.args.sliding_window,
    return_array=True,                    # ← force return array
)
```

`return_array=True` means even when N>1 it returns actual mask array rather than "causal" string.

**Zig**:
```zig
const mm: []const u8 = if (start_pos == 0 and seq_len > 1) "causal" else "";
attn_out = try fast_mod.scaledDotProductAttention(ctx, q, full_kv, full_kv, scale, mm, mask, ...);
```

Zig passes `"causal"` string during prefill, while Python passes actual mask array.
MLX's `scaled_dot_product_attention` should handle `"causal"` string and actual mask array equivalently,
but Python chooses `return_array=True` possibly to work with sliding window.

**Potential difference**: If sliding_window < seq_len, Python's mask includes window limit,
while Zig's `"causal"` string does not include window limit.

---

## IV. Key Findings Summary

| # | Item | Python | Zig | Status |
|---|------|--------|-----|--------|
| 1 | post_mult | `2.0` | `2.0` (fixed) | ✅ |
| 2 | softmaxPrecise | `precise=True` | `softmaxPrecise` (fixed) | ✅ |
| 3 | mhcPreApplyMix float32 | `y = x.astype(f32)` | Added float32 | ✅ |
| 4 | mhcPost float32 | `x.astype(f32)`, `residual.astype(f32)` | Added float32 | ✅ |
| 5 | **mhcPost comb transpose** | `comb.swapaxes(-1, -2) @ residual` | `comb @ residual` (no transpose) | **❌ BUG** |
| 6 | max_new_tokens guard | N/A | Added | ✅ |
| 7 | Chat template | Consistent | Consistent | ✅ |
| 8 | Attention mask | `return_array=True` | `"causal"` string | ⚠️ Possible difference |
