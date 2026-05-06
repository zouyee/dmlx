# dmlx DeepSeek V4 Flash — 完整分析报告
# 日期：2026-05-03（修订）
# 基线：mlx-lm `hyper_connection.py` + `deepseek_v4.py`

---

## 一、7 题端到端测试结果

运行方法：`scripts/best_test.sh`（smelt + stream，chat 模式，temperature=0）

| # | 提示 | 预期结果 | max_tokens | 状态 |
|---|--------|----------|-----------|--------|
| 1 | `2+2=` | `4` | 30 | ✅ |
| 2 | `The capital of France is` | `Paris` | 20 | ✅ |
| 3 | `What temperature does water freeze at in Celsius? Just give the number.` | `0` | 30 | ✅ |
| 4 | `Is the Earth round? Reply with only yes or no.` | `yes` | 30 | ✅ |
| 5 | `3*3=` | `9` | 30 | ✅ |
| 6 | `10-5=` | `5` | 30 | ✅ |
| 7 | `What is capital of France?` | `Paris` | 30 | ✅ |

**7/7 全部通过，0 跳过。**

说明：Chat 模式下模型在答案前会输出分析文字，因此需要足够的 max_tokens。
Prompt tokens 与 Python tokenizer 完全一致（已验证：`[0, 128803, 20, 13, 20, 31, 128804, 128822]`）。

---

## 二、已修复的 BUG（共 7 项）

### P0 — 根因修复

| # | BUG | 文件 | 与 Python 对比 | 说明 |
|---|-----|------|-----------|-------------|
| 1 | **mhcPost comb 未转置** | `deepseek_v4.zig` mhcPost | `comb.swapaxes(-1,-2) @ residual` | Python 执行 comb^T @ res，Zig 执行 comb @ res。43 层×2=86 处累积误差。**最关键修复** |
| 2 | **post_mult 1.0 → 2.0** | `deepseek_v4.zig` DSV4HyperConn.pre | `2 * mx.sigmoid(...)` | ops 路径传递 1.0，Metal kernel 使用 2.0。86 处累积偏差 |
| 3 | **mxfp4 反量化** | `deepseek_v4_loader.zig` | `biases==null → mxfp4` | group_size=32, mode="mxfp4"。修复模型崩溃/乱码输出 |

### P1 — 精度修复

| # | BUG | 文件 | 与 Python 对比 | 说明 |
|---|-----|------|-----------|-------------|
| 4 | **mhcPreApplyMix 缺少 float32** | `deepseek_v4.zig` | `(pre * y).sum()` 其中 `y=x.astype(f32)` | bfloat16 下加权和精度损失 |
| 5 | **mhcPost 缺少 float32** | `deepseek_v4.zig` | `x.astype(f32)`, `residual.astype(f32)` | matmul/combine 精度损失 |
| 6 | **sinkhornNormalize 缺少 precise** | `deepseek_v4.zig` | `mx.softmax(comb, precise=True)` | Sinkhorn 初始 softmax 精度 |

### P2 — 安全/功能

| # | BUG | 文件 | 说明 |
|---|-----|------|-------------|
| 7 | **generate max_new_tokens=0 下溢** | `deepseek_v4.zig` | `0-1` usize 下溢导致无限循环 |

---

## 三、Python ↔ Zig 逐函数对比

### 3.1 HyperConnection.__call__ → mhcPreNormFn + mhcPreSplitMixes

```
Python:                                    Zig:
y = x.astype(f32)                          (无显式转换，fn 权重为 f32 故 matmul 输出为 f32)
z = rms_norm(y.flatten(-2), None, eps)     手动 RMSNorm: square→sum→rsqrt→multiply
mixes = z @ self.fn.T                      mixes = res_normed @ fn^T
pre, post, comb = split_sinkhorn(mixes)    pre, post, comb = mhcPreSplitMixes(mixes)
  post = 2 * sigmoid(...)                    post = 2.0 * sigmoid(...)  ✅
  comb = softmax(precise=True) + eps         comb = softmaxPrecise + eps  ✅
  sinkhorn(comb)                             sinkhornNormalize(comb)  ✅
return (pre[...,None] * y).sum(2)          return mhcPreApplyMix(residual, pre)
  .astype(x.dtype)                           float32 提升 + astype(orig_dtype)  ✅
```

**状态**：✅ 已对齐

### 3.2 hc_expand → mhcPost

```
Python:                                    Zig:
y = post[...,None] * x[:,None,:].f32()     term1 = x_expanded_f32 * post_mix_f32  ✅
y += comb.swapaxes(-1,-2) @ res.f32()      term2 = comb_2d_t @ res_2d  ✅ (已转置)
return y.astype(x.dtype)                   return astype(result_f32, orig_dtype)  ✅
```

**状态**：✅ 已对齐

### 3.3 Chat Template

```
Python (Jinja):                            Zig:
<bos><User>content<Assistant></think>      <bos><User>content<Assistant></think>
```

Prompt tokens 完全一致：`[0, 128803, 20, 13, 20, 31, 128804, 128822]`

**状态**：✅ 已对齐

### 3.4 Attention Mask

```
Python: create_attention_mask(return_array=True, window_size=128)
Zig:    "causal" 字符串（无滑动窗口限制）
```

**状态**：⚠️ 潜在差异。当 seq_len > sliding_window (128) 时，Python 的 mask 会限制 attention 范围，
Zig 的 "causal" 则不会。对短 prompt（8 tokens）无影响，长文本可能存在差异。

### 3.5 Gate (MoE Router)

```
Python: logits.astype(f32) → sqrtsoftplus → argpartition
Zig:    logits (原始 dtype) → sqrtsoftplus → argsort slice
```

**状态**：⚠️ Zig 未将 logits 提升为 float32。对短 prompt 影响微小，长序列可能累积误差。

---

## 四、修改文件列表

| 文件 | 变更内容 |
|------|---------|
| `src/models/deepseek_v4.zig` | mhcPost comb 转置 + float32、mhcPreApplyMix float32、post_mult 2.0、softmaxPrecise、generate 守卫 |
| `src/models/deepseek_v4_loader.zig` | mxfp4 反量化检测 |
| `src/models/expert_stream.zig` | ExpertCache 启用，诊断日志清理 |
| `src/ops.zig` | softmaxPrecise 函数 |
| `src/tokenizer/chat_template.zig` | thinking_mode 支持，官方 Jinja 对齐 |
| `src/main.zig` | --raw 标志，smelt_strategy 传递 |
| `src/server.zig` | smelt_strategy 配置 |
| `scripts/best_test.sh` | 7 题全部通过，0 跳过 |

---

## 五、剩余已知差异（不影响 7 题测试）

1. **Attention mask**：Zig 使用 `"causal"` 字符串，Python 使用 `return_array=True` 带滑动窗口。对短 prompt 无影响。
2. **Gate float32**：Python 在 MoE gate 中将 logits 提升为 float32，Zig 未执行此操作。
3. **Metal kernel 未使用**：Python 在 GPU 上使用融合 Metal kernel（`hc_sinkhorn_collapse`），Zig 仅使用 ops 路径。功能等价但存在性能差异。
4. **Compressor overlap**：Python 的 compress_ratio=4 层使用 overlap compression，Zig 实现可能不同（未深入对比）。

---

## 六、验证状态

| 检查项 | 状态 |
|------------|--------|
| `zig build` | ✅ 编译通过 |
| `zig build test` | ✅ 所有单元测试通过（退出码 0）|
| `scripts/best_test.sh` | ✅ 7/7 通过，0 跳过 |
| Prompt tokens 对比 | ✅ 与 Python tokenizer 完全一致 |
| pre-commit hook | ✅ fmt + build + test |

(End of file - total 148 lines)
