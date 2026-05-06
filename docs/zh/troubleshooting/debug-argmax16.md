# DEBUG: Stream模式输出始终为 `.`（argmax=16）

## 症状
- 输入: `2+2=`（DeepSeek V4 chat template）
- 期望输出: `4`（token 22）
- 实际输出: `.`（token 16, logit=17.65），token 22 logit=13.14
- 每次运行结果一致（确定性bug，非随机）
- MOE DIAG: routed expert `y mean=-0.088, max=15.7` vs shared `mean=0.003, max=0.46`

## 约束
- 48GB Mac，mlx-lm 无法运行 DeepSeek V4（OOM ~138GB）
- 只能通过代码对照 Python mlx-lm 源码来排查
- 非smelt模式也会OOM

## 排查路径

### P1: Gate routing — 选错了expert？
**检查点**: gate.forward 返回的 indices 和 scores 是否合理
- [ ] P1.1: Python `_expert_select` 先 `logits.astype(mx.float32)` 再算 scores，Zig 是否也做了 float32 转换？
- [ ] P1.2: Python 用 `argpartition(-biased, kth=top_k-1)[..., :top_k]`，Zig 用 `argsort` + slice last k。结果应该等价但需确认
- [ ] P1.3: Hash routing（前3层）— `tid2eid[input_ids]` 的 input_ids 形状是否正确？Python 传的是 flat input_ids `[B*S]`，Zig 传的是什么？
- [ ] P1.4: `e_score_correction_bias` 是否正确加载？Python 在 `_expert_select` 中 `biased = scores + e_score_correction_bias`

### P2: SwitchGLU — expert计算本身有bug？
- [ ] P2.1: `dispatchGatherMm` 的 `transpose=True` 参数是否正确？Python `gather_qmm(..., transpose=True)` 意味着 weight 不需要预先转置
- [ ] P2.2: Zig 代码同时做了 `transposeAxes(weight, {0,2,1})` 和传 `transpose=True` 给 gatherQmm — 是否双重转置了？
- [ ] P2.3: `limitedSwiGLU` 参数顺序：Python 是 `_limited_swiglu(gate, up, limit)` 即 `silu(gate) * up`，Zig 的 `limitedSwiGLU(gate, up)` 是否一致？
- [ ] P2.4: `forwardNoScores` 的 sort 路径中 `x_flat = flatten(x_exp, 0, ndim-3)` 是否正确？Python 是 `x.flatten(0, -3)`

### P3: Stream模式特有 — 权重加载/切片
- [ ] P3.1: `loadExpertSlices` 加载的权重形状是否正确？应该是 `[n_unique, out, in]`
- [ ] P3.2: 加载的权重是否包含 scales？mxfp4 需要 group_size=32, bits=4
- [ ] P3.3: remap 逻辑是否正确？`remapped[i] = remap[indices[i]]` 将全局expert ID映射到局部索引
- [ ] P3.4: `loadExpertSlicesCached` 当前直接调用 `loadExpertSlices`（绕过cache），`loadExpertSlices` 是否正确？

### P4: 注意力层
- [ ] P4.1: DSV4Attention 使用 `scaledDotProductAttention` + `"causal"` mode，这与 Python 的 `scaled_dot_product_attention` 是否等价？
- [ ] P4.2: RoPE 实现是否正确？YARN scaling 参数是否匹配？
- [ ] P4.3: KV cache 更新是否正确？

### P5: 模型结构
- [ ] P5.1: mHC (HyperConnection) — `expandToMHC` 和 `HyperHead` 是否正确实现？
- [ ] P5.2: Layer forward 中 residual connection 是否正确？
- [ ] P5.3: 最终 norm + lm_head 是否正确？

### P6: 权重加载
- [ ] P6.1: 权重名称映射是否正确？`switch_mlp.gate_proj.weight` vs `ffn.switch_mlp.gate_proj.weight`
- [ ] P6.2: shared expert 权重是否正确加载？
- [ ] P6.3: attention 权重是否正确加载？

## 排查优先级
1. **P2.2** — 最可疑：双重转置会导致 expert 输出完全错误，解释 y max=15.7
2. **P1.1** — float32 转换缺失可能导致 scoring 精度问题
3. **P1.3** — hash routing input_ids 形状错误会导致前3层选错expert
4. **P3** — stream 模式权重加载
5. **P4, P5, P6** — 如果以上都正确再查

## 检查结果记录

### P2.2 检查结果
✅ 没有双重转置。量化路径传原始 weight + transpose=true，非量化路径传 weight_t。正确。

### P1.1 检查结果
⚠️ Python 先 `logits.astype(mx.float32)` 再算 scores，Zig 没有。可能影响精度但不太可能导致 argmax=16。需修复但非根因。

### P1.3 检查结果
✅ Hash routing input_ids 形状处理正确。tid2eid[ids] 的 reshape 逻辑与 Python 等价。

### P5.2 检查结果 — ⚠️ 发现关键 bug！
**mhcPost (HyperConnection post) 中 comb 矩阵缺少转置！**

Python `_hc_expand_op`:
```python
y = post[..., None] * x[:, :, None, :].astype(mx.float32)
y = y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))
#                  ^^^^^^^^^^^^^^^^^^^^ comb 转置了！
```

Zig `mhcPost`:
```zig
const term2_2d = try ops.matmul(ctx, comb_2d, res_2d);
//                                    ^^^^^^^ comb 没有转置！
```

**影响**：每一层的 residual connection 都用了错误的 comb 矩阵，导致所有层输出偏差。这会累积到最终 logits，可能是 argmax=16 的根本原因。

### P5.2b — mhcPost 还缺少 float32 转换
Python 在 post 中做了 `.astype(mx.float32)` 再计算，最后 `.astype(x.dtype)` 转回。Zig 没有做 float32 转换。 → **已修复**

### 修复后测试结果（第一轮）
- argmax 仍然是 16（`.`），但 token 22 logit 从 13.14 → 15.78（改善明显）
- Top tokens: [16]=17.76 [304]=16.93 [343]=16.92
- 说明 comb 转置修复有效果但还有其他问题

### 继续排查

### P5.2c — mhcPreSplitMixes post_mult_value 错误！
Python: `post = 2 * mx.sigmoid(...)` — 乘以 2
Zig: `post_mult_value = 1.0` — 乘以 1
**影响**：每一层 sublayer output 在 residual 中的权重只有正确值的一半。43层累积效应巨大。
→ **已修复**：改为 2.0

### P5.2d — mhcPreApplyMix 缺少 float32 转换
Python: `(pre[..., None] * y).sum(axis=2).astype(x.dtype)` 其中 `y = x.astype(mx.float32)`
Zig: 直接用 bfloat16 residual 计算
→ **已修复**：添加 float32 转换

### 第二轮修复后测试结果
- post_mult=2.0 修复后：argmax 仍然是 16
- Logits: max=19.02, token 22 logit=14.13（反而比第一轮的 15.78 降了）
- 说明 post_mult=2.0 可能不是正确的修复方向，或者有其他问题抵消了

### 已排除的路径
- P2.2: 双重转置 — ✅ 没有问题
- P1.3: Hash routing input_ids — ✅ 形状正确
- P3.1-P3.4: Stream 权重加载 — ✅ loadExpertSlices 用 takeAxis 切片，正确
- P4.1: SDPA — ✅ 使用 "causal" mode，与 Python 等价
- P6.2: Shared expert 量化 — ✅ affine group_size=64 bits=4，与 config.json 一致
- P5.1: expandToMHC — ✅ 与 Python 等价
- P5.3: HyperHead — ✅ 实现与 Python 一致

### 已修复但未解决 argmax=16
- P5.2: comb 转置 — 修复了，token 22 logit 从 13.14→15.78
- P5.2c: post_mult 1.0→2.0 — 修复了，但 argmax 仍然是 16
- P5.2d: mhcPreApplyMix float32 — 修复了
- P1.1: gate logits float32 — 修复了

### 待检查
- P2.3: limitedSwiGLU 参数顺序
- P2.4: forwardNoScores sort 路径的 flatten
- P4.2: RoPE YARN scaling
- P5.2e: mhcPreNormFn 是否需要 float32 转换
- P6.1: 权重名称映射
- P6.3: attention 权重加载

### 新增排查路径
- P7: 检查 Python sanitize 中的权重变换（view(uint32), repeat scales 等）是否在 Zig 中正确处理
- P8: 检查 per-weight quantization config 是否正确读取和应用

### 已排除的更多路径
- P7: Python sanitize view(uint32) — ✅ safetensors 中已经是 uint32
- P6.2: shared expert quantization — ✅ affine group_size=64 bits=4，与 config.json 一致
- P6.3: attention quantization — ✅ affine group_size=64 bits=4，与 config.json 一致
- P5.1: expandToMHC — ✅ 与 Python 等价
- P4.1: SDPA mask_mode — ✅ "causal" string literal 是 null-terminated
- mhcPreNormFn RMS norm — ✅ 数学等价（先 matmul 再 norm = 先 norm 再 matmul）

### 下一步：添加诊断打印
在 layer 0 的 attention 输出后打印统计信息，隔离问题是在 attention 还是 MoE 还是 mHC。
如果 attention 输出就已经不对，问题在 attention。
如果 attention 输出正确但 MoE 后不对，问题在 MoE。
如果 MoE 后正确但 mHC post 后不对，问题在 mHC。

### 诊断结果
- LAYER0 attn_out: mean=-0.006544, max=4.02, min=-4.20 — 合理
- LAYER0 ffn_out (MoE): mean=-0.087266, max=16.29, min=-21.71 — **routed expert max=16.08 太大**
- shared expert: mean=0.003328, max=0.46 — 正常

### 问题定位
routed expert 输出量级不对（max=16 vs shared max=0.46）。
问题在 stream 模式的 expert 计算（forwardNoScores + 切片权重）。

### 关键假设
correctness test 用全量 256-expert 权重 + indices=[0..5] 匹配 Python。
stream 模式用切片后 6-expert 权重 + remapped indices=[0..5]。
如果 takeAxis 切片破坏了 mxfp4 tensor 的内部格式（strides 不连续），gather_qmm 可能产生错误结果。

### 下一步
在 streamingForward 中添加诊断：
1. 打印切片后权重的 shape, dtype, strides
2. 对切片后权重做 eval() + copy() 强制连续化
3. 对比切片前后的 gather_qmm 结果

---

## 2025-05-01 修复验证

### 运行结果
- 输入: `2+2=`
- 实际输出: `2+2=4` ✅（生成的 tokens: `{20, 13, 20, 31, 22}`，最终 token 22='4'）
- 之前输出: `"That's a classic!"` 或 `"to question 2 ="` ❌

### 关键发现：根因不是 routed expert，而是 **embed/lm_head 的 dequantize 模式错误**

**`dequantIfNeeded` 和 embed dequantize 硬编码了 `"affine"` + `group_size=64`**：

```zig
// 旧代码（错误）
const gs = config.quantize_default_group_size;  // 64
...mlx_dequantize(..., "affine", ...);
```

如果 embed/lm_head 的量化配置是 **mxfp4**（`group_size=32`，无 biases），旧代码用错误的 mode/group_size dequantize，产生 **garbage embeddings**。这污染了所有层的输入/输出，导致模型完全胡说。

### 修复内容
在 `deepseek_v4_loader.zig` 中，根据 `biases == null` 检测 mxfp4：

```zig
const is_mxfp4 = biases == null;
const gs = if (is_mxfp4) 32 else config.quantize_default_group_size;
...mlx_dequantize(..., if (is_mxfp4) "mxfp4" else "affine", ...);
```

同样修复了 embed dequantize。

### 修复后诊断对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 输出 | `"That's a classic!"` | `2+2=4` ✅ |
| Prefill argmax | 20 ('2', logit~29) | 20 ('2', logit=19.49) |
| Token 22 ('4') logit | ~10-13 | **17.36** |
| Decode routed expert max | ~16 | ~0.55 |

### 剩余问题
- Prefill 阶段 `argmax` 仍是 20（'2'）而不是 22（'4'），但差距已从 ~15 缩小到 ~2
- 模型在 decode 阶段自校正，最终输出正确结果 `4`
- 可能是 gate routing 或 sort 路径仍有微小偏差

### 已排除的路径
- P3 (Stream 权重加载): `readExpertRows` + `copy()` 后 strides 是标准 contiguous `{1048576, 512, 1}`
- P2.2 (双重转置): 代码逻辑正确
- P6.2/P6.3 (shared expert/attention quantization): 实际为 affine，有 biases

### 真正根因
**P8: per-weight quantization config 未正确应用** — `dequantIfNeeded` 硬编码 affine，未检测 mxfp4。

### 下一步
1. 验证 embed/lm_head 是否确实为 mxfp4（检查 safetensors 中是否有 `.biases`）
2. 进一步优化 prefill logits，使 argmax 直接为 22
