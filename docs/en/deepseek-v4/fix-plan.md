# DeepSeek V4 修复计划（基于 Python mlx-lm 对比验证）

**验证日期:** 2026-05-01  
**参考实现:** `../mlx-lm/mlx_lm/models/deepseek_v4.py` (1171 行)  
**目标实现:** `src/models/deepseek_v4.zig` (2949 行) + `src/models/deepseek_v4_loader.zig` (1983 行)  
**结论:** 已定位导致 argmax=16 的根本差异

---

## 1. Python vs Zig 关键差异（经源码逐行对比）

### 🔴 差异 1：Attention Mask 构造方式（根本原因）

**Python (`base.py:45-55`)：**
```python
def create_attention_mask(h, cache=None, window_size=None, return_array=False):
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None
    if return_array or (window_size and N > window_size):
        return create_causal_mask(N, window_size=window_size)  # ← 显式数组
    return "causal"
```

Python `DeepseekV4Model.__call__` **总是**传入 `return_array=True`：
```python
mask = create_attention_mask(
    h[:, :, 0, :],
    mask_cache,
    window_size=self.args.sliding_window,  # 128
    return_array=True,   # ← 强制返回数组，从不返回字符串
)
```

Python `create_causal_mask` (`base.py:24-42`) 构造的数组包含 **sliding window 约束**：
```python
mask = linds >= rinds  # causal
if window_size is not None:
    mask = mask & (linds < rinds + window_size)  # sliding window
```

**Zig (`deepseek_v4.zig:1712-1720`)：**
```zig
const mm: []const u8 = if (start_pos == 0 and seq_len > 1) "causal" else "";
attn_out = try fast_mod.scaledDotProductAttention(
    self.ctx, q, full_kv, full_kv, scale, mm, mask, self.sink_logits
);
```

Zig **从不构造 mask 数组**：
- `DSV4Model.generate` 对预填充和解码都传递 `mask = null`
- 仅依赖 `"causal"` 字符串模式
- **`sliding_window` 约束完全缺失**

**为什么这导致 argmax=16：**
- DeepSeek V4 使用 `sliding_window=128`。在短提示（7 tokens）时，滑动窗口不会截断，causal mask 单独就能工作
- 但如果 MLX C API 的 `"causal"` 字符串模式在 `mask_arr=null` 时**不生效**（或当存在 `sinks` 参数时行为异常），注意力会变成双向
- 预填充时的双向注意力会破坏提示语义 → 所有提示产生相同的退化分布 → argmax=16

**验证方法：** 在 Python 中运行同样的模型，对比 `mask="causal"` vs `mask=create_causal_mask(...)` 的输出。如果 `"causal"` 字符串模式在 DeepSeek V4 的特定配置下（single KV head, sink logits）行为异常，则确认此根因。

### 🟢 差异 2：kv_b（不是原因）

Python `deepseek_v4.py` 中没有 `kv_b` 权重。Zig 中 `.kv_b = null`，且 `DSV4Attention.forward` 从未引用 `self.kv_b`。这不是导致 argmax=16 的原因。

### 🟡 差异 3：HyperConnection 扩展

Python：
```python
h = self.embed_tokens(inputs)
h = mx.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]))
h = mx.contiguous(h)
```

Zig 需要验证 `expandToMHC` 是否等价于 `broadcast_to(..., hc_mult)` + `contiguous`。

### 🟡 差异 4：RoPE 调用

Python RoPE (`traditional=True`)：
```python
return mx.fast.rope(x, head_dim, traditional=True, base=None, scale=1.0, offset=offset, freqs=freqs)
```

Zig `DSV4YarnRoPE.apply` 调用 `mlx_fast_rope` 时 `traditional=false`。需要确认 `traditional` 参数是否影响数值。

---

## 2. 修复方案

### P0 — 修复 1：显式构造 Attention Mask 数组

在 `DSV4Model.generate` 的预填充路径中，模仿 Python 构造显式 causal mask：

```zig
// 新增函数（类似 Python create_causal_mask）
fn createCausalMaskArray(ctx: EagerContext, seq_len: usize, window_size: usize, start_pos: usize) !Array {
    const n = seq_len;
    const total = start_pos + n;
    
    // 构造 [n, total] 的 boolean mask
    var mask_data = try ctx.allocator.alloc(u8, n * total);
    defer ctx.allocator.free(mask_data);
    
    for (0..n) |qi| {
        const q_pos = start_pos + qi;
        for (0..total) |ki| {
            const k_pos = ki;
            const causal = k_pos <= q_pos;  // 可以 attend 到当前及之前位置
            const in_window = if (window_size > 0)
                q_pos < window_size or k_pos >= q_pos + 1 - window_size
            else
                true;
            mask_data[qi * total + ki] = if (causal and in_window) 1 else 0;
        }
    }
    
    const shape = &[_]i32{@intCast(n), @intCast(total)};
    return try Array.fromData(ctx.allocator, u8, mask_data, shape);
}
```

在 `DSV4Model.generate` 中使用：
```zig
// 预填充时
var mask_arr: ?Array = null;
if (prompt_tokens.len > 1) {
    mask_arr = try createCausalMaskArray(self.ctx, prompt_tokens.len, self.config.sliding_window, 0);
}
const logits = try self.forward(prompt_arr, mask_arr, caches, 0, stream);
if (mask_arr) |m| m.deinit();
```

在 `DSV4Attention.forward` 中，将 mask 数组传给 SDPA：
```zig
// 将传入的 mask 数组扩展为 [B, H, L, S] 或传给 C API
const m_ptr = if (mask) |m| m.inner else c.c.mlx_array_empty;
c.c.mlx_fast_scaled_dot_product_attention(..., mm, m_ptr, ...);
```

### P0 — 修复 2：验证并修复 `traditional` RoPE 参数

Python 使用 `traditional=True`，Zig 使用 `traditional=false`。对于非交错的 RoPE（standard LLaMA 格式），`traditional=True` 是标准选择。

需要验证 Zig 的 `mlx_fast_rope` 调用是否应改为 `traditional=true`：
```zig
// 当前：
try c.check(c.c.mlx_fast_rope(&res, input.inner, dims, false, ...));

// 应改为：
try c.check(c.c.mlx_fast_rope(&res, input.inner, dims, true, ...));
```

### P0 — 修复 3：验证 HyperConnection 扩展

确认 `expandToMHC` 等价于 Python 的：
```python
mx.broadcast_to(h[:, :, None, :], (B, L, hc_mult, hidden_size))
mx.contiguous(h)
```

### P1 — 修复 4：启用 ExpertCache

替换 `expert_stream.zig` 的 bypass：
```zig
// 在 loadExpertSlicesCached 中实际使用 self.cache
if (self.cache) |ec| {
    if (ec.get(cache_key, expert_ids)) |cached| return cached;
    const result = try self.loadExpertSlices(...);
    ec.put(cache_key, expert_ids, result) catch {};
    return result;
}
```

### P1 — 修复 5：启用 PartialTensorReader

将 `loadExpertSlices` 从完整张量加载改为部分读取：
```zig
// 从：
const full_tensor = try self.index.loadTensor(tensor_name);
// 改为：
const partial = try self.partial_reader.readExpertRows(tensor_name, expert_ids);
```

### P2 — 修复 6：其他边界条件
- `max_new_tokens - 1` 下溢
- `findCorrectionDim` base==1.0 除零
- `n_experts - k` 下溢

---

## 3. 验证计划

修复后应按以下顺序验证：

1. **Python 基线**：运行 `mlx-lm` 生成 `"2+2="` 的首个 token，记录 argmax 和 logits
2. **Zig 修复后**：运行相同提示，对比首个 token 是否匹配
3. **Stream 模式性能**：启用 ExpertCache 后，测量每 token 耗时（目标 < 5 秒/令牌）
4. **完整回答**：验证 `"2+2="` 能生成 `"4"` 或相关数学答案

---

*本计划基于 `mlx-lm` Python 参考实现与 `mlx-zig` Zig 实现的逐行源码对比。*
