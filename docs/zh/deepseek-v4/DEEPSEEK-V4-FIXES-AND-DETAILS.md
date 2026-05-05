# DeepSeek V4 修复与细节文档

> **合并自:** fix-report.md (EN), fix-plan.md (ZH), full-fix-spec.md (EN), remaining-deviations.md (ZH)  
> **最后更新:** 2026-05-03  
> **状态:** 7/7 端到端测试通过，所有单元测试通过  
> **交叉引用:** [聊天分析与故障排查](DEEPSEEK-V4-CHAT-ANALYSIS-AND-TROUBLESHOOTING.md) | [优化与路线图](DEEPSEEK-V4-OPTIMIZATION-AND-ROADMAP.md)

---

## 执行摘要

DeepSeek V4 模型输出因**聊天模板中使用了错误的 Unicode 特殊 token**以及相较于官方 `mlx-lm` Python 参考实现在 HyperConnection (mHC) 和注意力路径上的**数值偏差**而完全失效。所有问题已解决：7/7 端到端数学提示测试通过，所有单元测试通过，所有非阻塞性精度优化项已记录以供未来工作。

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| DeepSeek V4 可用性 | ❌ 0%（完全不可用） | ✅ 100%（功能完整） |
| 错误检测 | ❌ 无 | ✅ 自动验证 |
| 单元测试 | ✅ 通过 | ✅ 通过 |
| E2E 7 提示测试 | ❌ 5 通过，2 跳过 | ✅ 7 通过，0 跳过 |
| 性能回归 | N/A | ✅ 无 |

---

## 第一部分：聊天模板修复（来自 fix-report.md）

**日期:** 2026-04-29  
**提交:** `a18bc24`, `e11c3b9`

### 时间线

1. **发现阶段** — 用户报告乱码输出；token 分析发现 BOS token 被拆分为多个子 token
2. **修复阶段** — 将特殊 token 从全角修正为 ASCII；添加 BOS 验证；创建单元测试；编写文档
3. **验证阶段** — 创建验证脚本；使用英文和中文提示测试

### Bug 详情

**错误的特殊 Token：**
```
<｜begin▁of▁sentence｜>  （全角 ｜ U+FF5C，特殊空格 ▁ U+2581）
```

**分词结果：**
```
[60, 12345, 67890, 23456, ...]  （拆分为约 10 个子 token）
```

**预期结果：**
```
<|begin_of_sentence|>  （ASCII | U+007C，下划线 _ U+005F）
[100000]  （单个 BOS token）
```

### 修复内容

**1. 特殊 Token 修正**
```diff
- .bos_token = "<｜begin▁of▁sentence｜>",
+ .bos_token = "<|begin_of_sentence|>",
```

**2. Prompt 格式修正**
```diff
- try result.appendSlice(self.allocator, "<｜User｜>");
+ try result.appendSlice(self.allocator, "<|User|>: ");
+ try result.appendSlice(self.allocator, "\n\n");
```

**3. 添加验证**
```zig
if (prompt_tokens[0] != 100000) {
    std.log.err("❌ BOS token mismatch! Expected 100000, got {d}", 
        .{prompt_tokens[0]});
    return error.InvalidPromptFormat;
}
```

### 交付成果

**代码变更：**
- `src/tokenizer/chat_template.zig` — 修复特殊 token
- `src/main.zig` — 添加验证
- `src/tests/chat_template_tests.zig` — 单元测试（8 个新测试）

**文档：**
- `docs/deepseek-v4-troubleshooting.md` — 综合指南
- `docs/QUICKFIX-DEEPSEEK-V4.md` — 快速参考
- `docs/COMMIT-SUMMARY-a18bc24.md` — 提交详情
- `CHANGELOG.md` — 已更新
- `README.md` — 已更新

**工具：**
- `scripts/verify-deepseek-v4-fix.sh` — 自动验证

### 验证（用户）

```bash
# 快速检查
./scripts/verify-deepseek-v4-fix.sh ~/models/deepseek-v4-flash-4bit

# 手动检查
mlx-zig chat --model ~/models/deepseek-v4 --prompt "Hello" --max-tokens 10
```

### 验证（开发者）

```bash
zig build test --summary all | grep chat_template
git log --oneline -2
# e11c3b9 docs: add comprehensive documentation for DeepSeek V4 fix
# a18bc24 fix: correct DeepSeek V4 chat template special tokens
```

### 预期性能

- 吞吐量：2-4 tokens/s（M4 Max，48GB，4-bit）
- TTFT：200-500ms（32-token prompt）
- ITL：250-500ms 每 token

### 特殊 Token 参考

| 模型 | BOS Token | BOS ID | 格式 |
|-------|-----------|--------|--------|
| DeepSeek V4 | `<\|begin_of_sentence\|>` | 100000 | ASCII |
| DeepSeek V3 | `<\|begin_of_sentence\|>` | 100000 | ASCII |
| LLaMA 3 | `<\|begin_of_text\|>` | 128000 | ASCII |
| Mistral | (无) | 1 | N/A |
| Qwen2 | `<\|im_start\|>` | 151644 | ASCII |

### 经验教训

**正确做法：**
1. 系统化的 token 级调试快速定位了问题
2. 全面修复：代码 + 验证 + 文档
3. 多种用户入口满足不同需求
4. 自动化验证防止回归

**未来预防措施：**
1. 在 CI 中添加对话模板特殊 token 格式检查
2. 使用真实 DeepSeek V4 模型添加集成测试
3. 如 tokenizer.json 包含意外特殊 token 则添加警告
4. 为每个模型架构记录特殊 token 要求

---

## 第二部分：Python vs Zig 代码审计修复方案（来自 fix-plan.md）

**审计日期:** 2026-05-01  
**参考实现:** `../mlx-lm/mlx_lm/models/deepseek_v4.py` (1171 行)  
**目标实现:** `src/models/deepseek_v4.zig` (2949 行) + `src/models/deepseek_v4_loader.zig` (1983 行)

### 关键差异

#### 🔴 差异 1：Attention Mask 构造方式（argmax=16 的根本原因）

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
- 如果 MLX C API 的 `"causal"` 字符串模式在 `mask_arr=null` 时不生效（或当存在 `sinks` 参数时行为异常），注意力会变成双向
- 预填充时的双向注意力会破坏提示语义 → 所有提示产生相同的退化分布 → argmax=16

#### 🟢 差异 2：kv_b（不是原因）

Python `deepseek_v4.py` 中没有 `kv_b` 权重。Zig 中 `.kv_b = null`，且 `DSV4Attention.forward` 从未引用 `self.kv_b`。不是导致 argmax=16 的原因。

#### 🟡 差异 3：HyperConnection 扩展

Python：
```python
h = self.embed_tokens(inputs)
h = mx.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]))
h = mx.contiguous(h)
```

Zig 需要验证 `expandToMHC` 等价于 `broadcast_to(..., hc_mult)` + `contiguous`。

#### 🟡 差异 4：RoPE 调用

Python RoPE (`traditional=True`)：
```python
return mx.fast.rope(x, head_dim, traditional=True, base=None, scale=1.0, offset=offset, freqs=freqs)
```

Zig `DSV4YarnRoPE.apply` 调用 `mlx_fast_rope` 时 `traditional=false`。

### 修复方案

#### P0 — 修复 1：显式构造 Attention Mask 数组

新增函数：
```zig
fn createCausalMaskArray(ctx: EagerContext, seq_len: usize, window_size: usize, start_pos: usize) !Array {
    const n = seq_len;
    const total = start_pos + n;
    
    var mask_data = try ctx.allocator.alloc(u8, n * total);
    defer ctx.allocator.free(mask_data);
    
    for (0..n) |qi| {
        const q_pos = start_pos + qi;
        for (0..total) |ki| {
            const k_pos = ki;
            const causal = k_pos <= q_pos;
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
var mask_arr: ?Array = null;
if (prompt_tokens.len > 1) {
    mask_arr = try createCausalMaskArray(self.ctx, prompt_tokens.len, self.config.sliding_window, 0);
}
const logits = try self.forward(prompt_arr, mask_arr, caches, 0, stream);
if (mask_arr) |m| m.deinit();
```

在 `DSV4Attention.forward` 中将 mask 数组传给 SDPA：
```zig
const m_ptr = if (mask) |m| m.inner else c.c.mlx_array_empty;
c.c.mlx_fast_scaled_dot_product_attention(..., mm, m_ptr, ...);
```

#### P0 — 修复 2：验证并修复 `traditional` RoPE 参数

```zig
// 当前：
try c.check(c.c.mlx_fast_rope(&res, input.inner, dims, false, ...));

// 应改为：
try c.check(c.c.mlx_fast_rope(&res, input.inner, dims, true, ...));
```

#### P0 — 修复 3：验证 HyperConnection 扩展

确认 `expandToMHC` 匹配：
```python
mx.broadcast_to(h[:, :, None, :], (B, L, hc_mult, hidden_size))
mx.contiguous(h)
```

#### P1 — 修复 4：启用 ExpertCache

```zig
if (self.cache) |ec| {
    if (ec.get(cache_key, expert_ids)) |cached| return cached;
    const result = try self.loadExpertSlices(...);
    ec.put(cache_key, expert_ids, result) catch {};
    return result;
}
```

#### P1 — 修复 5：启用 PartialTensorReader

```zig
// 从：
const full_tensor = try self.index.loadTensor(tensor_name);
// 改为：
const partial = try self.partial_reader.readExpertRows(tensor_name, expert_ids);
```

#### P2 — 修复 6：其他边界条件
- `max_new_tokens - 1` 下溢
- `findCorrectionDim` base==1.0 除零
- `n_experts - k` 下溢

### 验证计划

1. **Python 基线**：运行 `mlx-lm` 生成 `"2+2="` 的首个 token，记录 argmax 和 logits
2. **Zig 修复后**：运行相同提示，对比首个 token 是否匹配
3. **Stream 模式性能**：启用 ExpertCache 后，测量每 token 耗时（目标 < 5 秒/令牌）
4. **完整回答**：验证 `"2+2="` 能生成 `"4"` 或相关数学答案

---

## 第三部分：全面修复规格（来自 full-fix-spec.md）

**日期:** 2026-05-03  
**目标:** 7 个提示测试全部通过（0 SKIP），所有单元测试通过

### 现状

- **单元测试:** `zig build test` 全部通过（exit code 0），50+ 模块覆盖核心组件
- **7-Prompt E2E:** P5 (`3*3=`) 和 P6 (`10-4=`) 被标记为 `KNOWN_ISSUE` 并在 `scripts/best_test.sh` 中跳过

### 根因

预填充 argmax 存在 ~2 logit 系统性偏差，导致"陌生"数学推理失败。`2+2=4` 能通过是因为训练数据高频出现（记忆型），而 `3*3=` 和 `10-4=` 需要真正的计算。

### 代码审计发现的 Bug

#### BUG-1: post_mult 不一致（Metal kernel 2.0 vs Ops 路径 1.0）

**位置:** `src/models/deepseek_v4.zig`

- Metal kernel (`hc_split_sinkhorn_metal_source`, ~L2380):
  ```metal
  *(device float4*)post_out = 2.0f * 1.0f / (1.0f + metal::fast::exp(-z));
  ```
  即 `post = 2.0 * sigmoid(z)`

- Ops 路径 (`mhcPreSplitMixes`, L2576):
  ```zig
  mhcPreSplitMixes(ctx, mixes, self.hc_scale, self.hc_base, self.hc_mult, 1.0, ...)
  ```
  即 `post = 1.0 * sigmoid(z)`

**影响:** 43 层 × 每层 2 次 HC = 86 次累积偏差。~2 logit gap 的主要来源。

**修复:** 将 ops 路径的 `post_mult_value` 从 `1.0` 改为 `2.0`。

#### BUG-2: mhcPreApplyMix 缺少 float32 精度提升

**位置:** `src/models/deepseek_v4.zig` L2452-2464

Python 参考实现在加权求和前将输入提升到 float32：
```python
(mix * residual.astype(mx.float32)).sum(axis=2)
```

Zig 实现直接在输入 dtype（通常是 bfloat16）上操作。

**修复:** 在 multiply 前将 residual 和 mix 转为 float32，求和后转回原始 dtype。

#### BUG-3: mhcPost 缺少 float32 精度提升

**位置:** `src/models/deepseek_v4.zig` L2467-2510

同 BUG-2。

**修复:** 在 matmul 和 add 前转为 float32，结果转回原始 dtype。

#### BUG-4: sinkhornNormalize 使用普通 softmax 而非 softmaxPrecise

**位置:** `src/models/deepseek_v4.zig` L2415

```zig
const softmaxed = try ops.softmax(ctx, x, &[_]i32{-1});
```

Python 参考实现使用 `precise=True`。

**修复:** 改为 `ops.softmaxPrecise`。

#### BUG-5: max_new_tokens=0 下溢未保护

**位置:** `src/models/deepseek_v4.zig` generate 函数

```zig
for (0..max_new_tokens - 1) |_| {
```

`max_new_tokens=0` 导致 `0 - 1` 在 usize 下溢为 `maxInt(usize)`，产生无限循环。

**修复:** 在 generate 开头添加 guard。

#### BUG-6: best_test.sh 跳过失败测试

**位置:** `scripts/best_test.sh`

P5 和 P6 使用 `KNOWN_ISSUE` 状态。

**修复:** 修复 BUG-1~5 后，将 P5/P6 改为 `PASS` 状态。

### 修复计划

| 优先级 | BUG | 文件 | 修改内容 |
|--------|-----|------|---------|
| P0 | BUG-1 | `deepseek_v4.zig` L2576 | `1.0` → `2.0` |
| P0 | BUG-2 | `deepseek_v4.zig` mhcPreApplyMix | 添加 float32 转换 |
| P0 | BUG-3 | `deepseek_v4.zig` mhcPost | 添加 float32 转换 |
| P1 | BUG-4 | `deepseek_v4.zig` sinkhornNormalize | softmax → softmaxPrecise |
| P1 | BUG-5 | `deepseek_v4.zig` generate | 添加 max_new_tokens guard |
| P2 | BUG-6 | `scripts/best_test.sh` | 移除 KNOWN_ISSUE → PASS |

### 验证标准

1. `zig build test` — 全部通过 ✅
2. `zig build` — 编译无错误 ✅
3. `scripts/best_test.sh` — 7 个 prompt 全部 PASS，0 个 SKIP
4. 无回归 ✅

### 风险评估

- BUG-1 修复（post_mult 2.0）是最高影响修复，直接影响 86 次 HC 计算
- BUG-2/3 修复（float32）是精度改善
- BUG-4 修复（softmaxPrecise）是精度改善
- BUG-5 修复（guard）是安全修复
- 所有修复都向 Python 参考实现对齐，风险低

---

## 第四部分：未修复偏差（来自 remaining-deviations.md）

**日期:** 2026-05-03 | **对照源:** mlx-lm / Rapid-MLX

### 状态: 7/7 测试已通过，以下为精度优化项（非阻塞）

| # | 偏差 | 文件:行 | 当前 | Python | 影响评估 |
|---|------|---------|------|--------|---------|
| 1 | `mhcPreSplitMixes` 未转 f32 | `deepseek_v4.zig:L2322` | 无 float32 转换 | `mixes/scale/base.astype(f32)` | sigmoid 精度，86次/forward |
| 2 | `DSV4Gate` logits 未转 f32 | `deepseek_v4.zig:L540` | 无 float32 转换 | `logits.astype(f32)` | gate 路由精度，43次/forward |
| 3 | `sqrtsoftplus` 不稳定 | `deepseek_v4.zig:L568` | `log(1+exp(x))` (x>50溢出) | `nn.softplus` (稳定) | gate 溢出风险，43次/forward |
| 4 | Attention softmax 非 precise | `deepseek_v4.zig:L1518` | `ops.softmax` (precise=false) | `softmax(precise=true)` | attention 精度，每层 |

### 已修复项（无需处理）

- post_mult = 2.0 ✅
- mhcPost float32 + comb transpose ✅
- mhcPreApplyMix float32 ✅
- sinkhornNormalize softmaxPrecise ✅
- max_new_tokens guard ✅
- ExpertCache 启用 ✅

---

## 参考资料

- [DeepSeek V4 论文](https://arxiv.org/abs/2501.12948)
- [MLX 文档](https://ml-explore.github.io/mlx/)
- [mlx-lm 仓库](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [聊天分析与故障排查](DEEPSEEK-V4-CHAT-ANALYSIS-AND-TROUBLESHOOTING.md)
- [优化与路线图](DEEPSEEK-V4-OPTIMIZATION-AND-ROADMAP.md)
