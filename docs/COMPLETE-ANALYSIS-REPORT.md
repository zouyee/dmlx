# mlx-zig DeepSeek V4 Flash — 完整分析报告
# 日期: 2026-05-03 (修订版)
# 基准: mlx-lm `hyper_connection.py` + `deepseek_v4.py`

---

## 一、7-Prompt 端到端测试结果

运行方式: `scripts/best_test.sh` (smelt + stream, chat mode, temperature=0)

| # | Prompt | 期望 | max_tokens | 状态 |
|---|--------|------|-----------|------|
| 1 | `2+2=` | `4` | 30 | ✅ |
| 2 | `The capital of France is` | `Paris` | 20 | ✅ |
| 3 | `What temperature does water freeze at in Celsius? Just give the number.` | `0` | 30 | ✅ |
| 4 | `Is the Earth round? Reply with only yes or no.` | `yes` | 30 | ✅ |
| 5 | `3*3=` | `9` | 30 | ✅ |
| 6 | `10-5=` | `5` | 30 | ✅ |
| 7 | `What is capital of France?` | `Paris` | 30 | ✅ |

**7/7 全部通过，0 skip。**

注: chat 模式下模型先输出分析文本再给答案，因此需要足够的 max_tokens。
Prompt tokens 与 Python tokenizer 完全一致 (已验证: `[0, 128803, 20, 13, 20, 31, 128804, 128822]`)。

---

## 二、已修复的 BUG (7 项)

### P0 — 根因修复

| # | BUG | 文件 | 对照 Python | 说明 |
|---|-----|------|------------|------|
| 1 | **mhcPost comb 未转置** | `deepseek_v4.zig` mhcPost | `comb.swapaxes(-1,-2) @ residual` | Python 做 comb^T @ res，Zig 做 comb @ res。43层×2次=86次错误累积。**最关键修复** |
| 2 | **post_mult 1.0 → 2.0** | `deepseek_v4.zig` DSV4HyperConn.pre | `2 * mx.sigmoid(...)` | ops 路径传 1.0，Metal kernel 用 2.0。86次累积偏差 |
| 3 | **mxfp4 反量化** | `deepseek_v4_loader.zig` | `biases==null → mxfp4` | group_size=32, mode="mxfp4"。解决模型崩溃/乱码 |

### P1 — 精度修复

| # | BUG | 文件 | 对照 Python | 说明 |
|---|-----|------|------------|------|
| 4 | **mhcPreApplyMix 无 float32** | `deepseek_v4.zig` | `(pre * y).sum()` 其中 `y=x.astype(f32)` | bfloat16 下加权求和精度损失 |
| 5 | **mhcPost 无 float32** | `deepseek_v4.zig` | `x.astype(f32)`, `residual.astype(f32)` | matmul/combine 精度损失 |
| 6 | **sinkhornNormalize 无 precise** | `deepseek_v4.zig` | `mx.softmax(comb, precise=True)` | Sinkhorn 初始 softmax 精度 |

### P2 — 安全/功能

| # | BUG | 文件 | 说明 |
|---|-----|------|------|
| 7 | **generate max_new_tokens=0 下溢** | `deepseek_v4.zig` | `0-1` usize 下溢导致无限循环 |

---

## 三、Python ↔ Zig 逐函数对照

### 3.1 HyperConnection.__call__ → mhcPreNormFn + mhcPreSplitMixes

```
Python:                                    Zig:
y = x.astype(f32)                          (无显式转换，fn 权重是 f32 所以 matmul 输出为 f32)
z = rms_norm(y.flatten(-2), None, eps)     手动 RMSNorm: square→sum→rsqrt→multiply
mixes = z @ self.fn.T                      mixes = res_normed @ fn^T
pre, post, comb = split_sinkhorn(mixes)    pre, post, comb = mhcPreSplitMixes(mixes)
  post = 2 * sigmoid(...)                    post = 2.0 * sigmoid(...)  ✅
  comb = softmax(precise=True) + eps         comb = softmaxPrecise + eps  ✅
  sinkhorn(comb)                             sinkhornNormalize(comb)  ✅
return (pre[...,None] * y).sum(2)          return mhcPreApplyMix(residual, pre)
  .astype(x.dtype)                           float32 提升 + astype(orig_dtype)  ✅
```

**状态**: ✅ 已对齐

### 3.2 hc_expand → mhcPost

```
Python:                                    Zig:
y = post[...,None] * x[:,None,:].f32()     term1 = x_expanded_f32 * post_mix_f32  ✅
y += comb.swapaxes(-1,-2) @ res.f32()      term2 = comb_2d_t @ res_2d  ✅ (转置)
return y.astype(x.dtype)                   return astype(result_f32, orig_dtype)  ✅
```

**状态**: ✅ 已对齐

### 3.3 Chat Template

```
Python (Jinja):                            Zig:
<bos><User>content<Assistant></think>      <bos><User>content<Assistant></think>
```

Prompt tokens 完全一致: `[0, 128803, 20, 13, 20, 31, 128804, 128822]`

**状态**: ✅ 已对齐

### 3.4 Attention Mask

```
Python: create_attention_mask(return_array=True, window_size=128)
Zig:    "causal" 字符串 (无 sliding window 限制)
```

**状态**: ⚠️ 潜在差异。当 seq_len > sliding_window (128) 时，Python 的 mask 会限制注意力范围，
Zig 的 "causal" 不会。对于短 prompt (8 tokens) 无影响，长文本可能有差异。

### 3.5 Gate (MoE Router)

```
Python: logits.astype(f32) → sqrtsoftplus → argpartition
Zig:    logits (原始 dtype) → sqrtsoftplus → argsort slice
```

**状态**: ⚠️ Zig 未将 logits 提升到 float32。对短 prompt 影响小，长序列可能累积。

---

## 四、修改文件清单

| 文件 | 变更 |
|------|------|
| `src/models/deepseek_v4.zig` | mhcPost comb 转置 + float32, mhcPreApplyMix float32, post_mult 2.0, softmaxPrecise, generate guard |
| `src/models/deepseek_v4_loader.zig` | mxfp4 反量化检测 |
| `src/models/expert_stream.zig` | ExpertCache 启用, 诊断日志清理 |
| `src/ops.zig` | softmaxPrecise 函数 |
| `src/tokenizer/chat_template.zig` | thinking_mode 支持, 官方 Jinja 对齐 |
| `src/main.zig` | --raw flag, smelt_strategy 传递 |
| `src/server.zig` | smelt_strategy 配置 |
| `scripts/best_test.sh` | 7 prompt 全 PASS, 0 skip |

---

## 五、残存已知差异 (不影响 7-prompt 测试)

1. **Attention mask**: Zig 用 `"causal"` 字符串，Python 用 `return_array=True` 含 sliding window。短 prompt 无影响。
2. **Gate float32**: Python 在 MoE gate 中将 logits 提升到 float32，Zig 未做。
3. **Metal kernel 未使用**: Python 在 GPU 上用 fused Metal kernel (`hc_sinkhorn_collapse`)，Zig 只走 ops 路径。功能等价但性能差异。
4. **Compressor overlap**: Python 的 compress_ratio=4 层用 overlap 压缩，Zig 实现可能有差异（未深入对比）。

---

## 六、验证状态

| 检查项 | 状态 |
|--------|------|
| `zig build` | ✅ 编译通过 |
| `zig build test` | ✅ 全部单元测试通过 (exit code 0) |
| `scripts/best_test.sh` | ✅ 7/7 PASS, 0 skip |
| Prompt tokens 对照 | ✅ 与 Python tokenizer 完全一致 |
| pre-commit hook | ✅ fmt + build + test |
