# Python mlx-lm 与 Zig dmlx 逐项对比
# 日期：2026-05-03

---

## I. Chat Template 对比

### Python (chat_template.jinja + deepseek_v32.py)

对于单条用户消息 `"2+2="`，thinking_mode="chat"：

```
<｜begin▁of▁sentence｜><｜User｜>2+2=<｜Assistant｜></think>
```

关键行为：
- 当 `add_generation_prompt=True` 时，如果最后一条消息来自 user，Jinja 模板**不会**追加 `<｜Assistant｜></think>`（因为 user 消息已经追加）
- Python `deepseek_v32.py` 的 `apply_chat_template` 在 `add_generation_prompt=True` 且最后一条消息来自 user 时，会 `removesuffix("<｜Assistant｜><think>")`（仅在 thinking 模式）

### Zig (chat_template.zig)

```zig
// add_generation_prompt: only when last message is NOT user
if (add_generation_prompt and ... and !std.mem.eql(u8, messages[messages.len - 1].role, "user")) {
    try result.appendSlice(self.allocator, "<｜Assistant｜></think>");
}
```

**结论**：Zig 在 `main.zig` 中以 `chat_template.apply(messages.items, true)` 调用，`add_generation_prompt=true`。
但由于最后一条消息来自 user，该 `add_generation_prompt` 分支不会执行。
user 消息本身已追加 `</think>`，因此最终结果正确：

```
<｜begin▁of▁sentence｜><｜User｜>2+2=<｜Assistant｜></think>
```

**结论**：✅ Chat template 行为一致。

---

## II. HyperConnection (mHC) 对比

### 2.1 HyperConnection.__call__（pre 阶段）

**Python** (`hyper_connection.py`)：
```python
def __call__(self, x: mx.array):
    B, L, H, D = x.shape
    y = x.astype(mx.float32)                          # ← float32 提升
    z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
    mixes = z @ self.fn.T
    # 调用 _hc_ops 或 _hc_kernel
```

**Zig** (`deepseek_v4.zig` mhcPreNormFn)：
```zig
// No explicit astype(float32), operates directly on input dtype
const res_flat = try ops.reshape(ctx, residual, ...);
// RMSNorm manually implemented, no prior float32 conversion
```

**差异**：Python 在进入 HyperConnection 时立即将 x 转换为 float32，Zig 则未执行此操作。
但在 Zig 的 mhcPreNormFn 中，RMSNorm 是手动实现的（square → sum → rsqrt），
且 fn 权重本身为 float32，因此 matmul 结果也是 float32。
**部分对齐**，但输入 residual 没有首先转换为 float32。

### 2.2 _hc_split_sinkhorn_ops（split + sinkhorn）

**Python**：
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

**Zig** (`mhcPreSplitMixes` + `sinkhornNormalize`)：
- `post_mult_value`：原为 `1.0`，**已修复为 `2.0`** ✅
- `sinkhornNormalize`：原为 `ops.softmax`，**已修复为 `ops.softmaxPrecise`** ✅
- **但**：Python 的 sinkhorn 先 softmax，然后列归一化，再循环（行归一化 → 列归一化）
  Zig 的 sinkhorn 也是 softmax → 列归一化 → 循环（行归一化 → 列归一化），**一致** ✅

### 2.3 _hc_ops（collapse / pre_apply_mix）

**Python**：
```python
def _hc_ops(x, y, mixes, scale, base, hc_mult, sinkhorn_iters, eps):
    pre, post, comb = _hc_split_sinkhorn_ops(...)
    return (pre[..., None] * y).sum(axis=2).astype(x.dtype), post, comb
    #                       ↑ y is x.astype(float32)
```

**Zig** (`mhcPreApplyMix`)：
```zig
// Fixed: added float32 promotion
const res_f32 = try ops.astype(ctx, residual, .float32);
const mix_f32 = try ops.astype(ctx, mix, .float32);
const weighted = try ops.multiply(ctx, res_f32, mix_f32);
const summed = try reduce_mod.sumAxis(ctx, weighted, 2, false);
return ops.astype(ctx, summed, orig_dtype);
```

**结论**：✅ 已对齐。

### 2.4 hc_expand（post 阶段）

**Python**：
```python
@mx.compile
def _hc_expand_op(x, residual, post, comb):
    y = post[..., None] * x[:, :, None, :].astype(mx.float32)     # ← float32
    y = y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))  # ← float32
    return y.astype(x.dtype)
```

注意：Python 使用 `comb.swapaxes(-1, -2)` 即 **comb 转置**！

**Zig** (`mhcPost`)：
```zig
// Fixed: added float32 promotion
// However:
const comb_2d = try ops.reshape(ctx, comb_mix_f32, ...);
const term2_2d = try ops.matmul(ctx, comb_2d, res_2d);  // ← comb NOT transposed!
```

**⚠️ 关键差异**：Python 执行 `comb.swapaxes(-1, -2) @ residual`，
Zig 执行 `comb @ residual`（未转置）。
这意味着 Zig 的 mhcPost 计算 `comb @ res` 而非 `comb^T @ res`。

### 2.5 Metal Kernel 对比

**Python Metal kernel** (`hc_sinkhorn_collapse_kernel`)：
```metal
float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));  // ← 2.0
```

**Zig Metal kernel** (`hc_split_sinkhorn_metal_source`)：
```metal
*(device float4*)post_out = 2.0f * 1.0f / (1.0f + metal::fast::exp(-z));  // ← 2.0
```

**一致** ✅（两者均为 2.0 * sigmoid）

---

## III. Attention Mask 对比

**Python**：
```python
mask = create_attention_mask(
    h[:, :, 0, :],
    mask_cache,
    window_size=self.args.sliding_window,
    return_array=True,                    # ← 强制返回数组
)
```

`return_array=True` 意味着即使 N>1 也会返回实际 mask 数组而非 "causal" 字符串。

**Zig**：
```zig
const mm: []const u8 = if (start_pos == 0 and seq_len > 1) "causal" else "";
attn_out = try fast_mod.scaledDotProductAttention(ctx, q, full_kv, full_kv, scale, mm, mask, ...);
```

Zig 在预填充阶段传递 `"causal"` 字符串，而 Python 传递实际 mask 数组。
MLX 的 `scaled_dot_product_attention` 应等价处理 `"causal"` 字符串和实际 mask 数组，
但 Python 选择 `return_array=True` 可能是为了与滑动窗口配合。

**潜在差异**：如果 sliding_window < seq_len，Python 的 mask 包含窗口限制，
而 Zig 的 `"causal"` 字符串不包含窗口限制。

---

## IV. 关键发现汇总

| # | 项目 | Python | Zig | 状态 |
|---|------|--------|-----|--------|
| 1 | post_mult | `2.0` | `2.0`（已修复）| ✅ |
| 2 | softmaxPrecise | `precise=True` | `softmaxPrecise`（已修复）| ✅ |
| 3 | mhcPreApplyMix float32 | `y = x.astype(f32)` | 已添加 float32 | ✅ |
| 4 | mhcPost float32 | `x.astype(f32)`, `residual.astype(f32)` | 已添加 float32 | ✅ |
| 5 | **mhcPost comb 转置** | `comb.swapaxes(-1, -2) @ residual` | `comb @ residual`（未转置）| **❌ BUG** |
| 6 | max_new_tokens 守卫 | N/A | 已添加 | ✅ |
| 7 | Chat template | 一致 | 一致 | ✅ |
| 8 | Attention mask | `return_array=True` | `"causal"` 字符串 | ⚠️ 可能差异 |

(End of file - total 200 lines)
