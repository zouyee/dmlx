# DeepSeek V4 + TurboQuant 论文分析：对 mlx-zig 的参考价值（更新版）

> 基于 DeepSeek V4 技术报告（2026-04-24）和 TurboQuant 论文（arXiv:2504.19874）的分析，
> 结合 mlx-zig 代码审计（2026-04-26）和刚完成的 MXFP4/FP8 绑定实现。

---

## 一、当前实现状态审计

在分析论文参考价值之前，先确认 mlx-zig 中 V4 相关功能的真实实现状态：

### 已完成（代码已存在且功能完整）

| Task | 实现位置 | 状态 |
|------|---------|------|
| 15.2 learned softmax-gated pooling | `compressKV()` → `softmaxGatedPool()` at L1231-1430 | ✅ 完整实现，含 remainder 处理 |
| 15.4 FP4 Lightning Indexer | `LightningIndexer` struct at L648-930 | ✅ 完整实现，含 INT4 量化模拟、top-k 选择、block gather |
| 15.5 Attention Sink | `sink_logits` 字段 at L959，传入 `scaledDotProductAttention` at L1144 | ✅ 已接入 fast SDPA |
| 15.3 FP8 KV 存储 | `kv_storage_dtype` + `astype` at L1055-1080 | ⚠️ 用 float16 代替 FP8（注释说明 MLX 缺原生 FP8） |
| 15.6 异构 KV cache | `compress_ratios` 从 config 读取，per-layer `compress_ratio` at L950 | ⚠️ 压缩比已 per-layer，但 KV cache strategy 仍统一 |

### 关键发现

1. **Task 15.2/15.4/15.5 标记为 `[x]` 是准确的** — 代码确实已实现
2. **Task 15.3 的 FP8 问题现在可以解决了** — 我们刚确认 mlx-c 0.6.0 有 `mlx_to_fp8`/`mlx_from_fp8`，且已在 `quantize.zig` 和 `ops.zig` 中新增了 Zig 绑定
3. **Task 15.6 部分完成** — `compress_ratios` 已 per-layer 配置，但 KV cache strategy（Standard/Paged/Quantized）没有 per-layer 分配
4. **Task 15.1（Paged + Quantized 组合）是唯一未开始的** — 标记为 `[ ]`

---

## 二、DeepSeek V4 论文：对剩余工作的精确指导

### 2.1 Task 15.3 — FP8 KV 存储：现在可以用真正的 FP8

**论文要求**：V4 将大部分 KV 维度存储为 FP8（E4M3），仅 RoPE 维度保持 BF16。

**当前实现**：`deepseek_v4.zig:1055-1080` 用 `astype(float16)` 作为 FP8 的代替，
注释写着 "MLX lacks native FP8"。

**现在的情况**：mlx-c 0.6.0 已有 `mlx_to_fp8`/`mlx_from_fp8`，我们刚在 `ops.zig` 中
新增了 `toFp8()`/`fromFp8()` 绑定。可以直接替换。

**具体改动**：
```zig
// 当前（L1059-1063）：
const kv_nope_stored = if (kv_storage_dtype != .float32)
    try ops.astype(self.ctx, kv_nope, kv_storage_dtype)  // float16 代替
else kv_nope;

// 改为：
const kv_nope_stored = try ops.toFp8(self.ctx, kv_nope);  // 真正的 FP8
// 读取时：
const kv_nope_restored = try ops.fromFp8(self.ctx, kv_nope_stored, .bfloat16);
```

**注意**：FP8 不是 mlx_dtype 枚举成员，不能用 `astype`。必须用专用的 `toFp8`/`fromFp8`。
RoPE 维度继续用 `astype(.bfloat16)`。

**优先级**：🔴 高 — 改动小（约 10 行），效果大（内存减半 vs float16）。

### 2.2 Task 15.6 — 异构 KV Cache：缺少 per-layer cache strategy

**论文要求**：V4-Pro 61 层的分布：
- Layer 0-1：HCA（128x 压缩）
- Layer 2-60：CSA（4x）和 HCA（128x）交替
- 每层的 KV cache 形状不同（压缩后序列长度不同）

**当前实现**：
- ✅ `compress_ratios` 已从 config.json 读取，per-layer 传入 `DSV4Attention`
- ✅ `compressKV()` 根据 `compress_ratio` 做不同压缩
- ❌ `server.zig` 和 `loadModel` 中所有层使用同一个 `KVCacheStrategy`
- ❌ 没有根据 `compress_ratio` 分配不同大小的 cache buffer

**需要的改动**：在 `loadModel` 中根据 `compress_ratios[i]` 为每层分配不同的 cache：
- CSA 层（ratio=4）：cache 序列维度 = max_seq_len / 4 + window_size
- HCA 层（ratio=128）：cache 序列维度 = max_seq_len / 128 + window_size
- 无压缩层（ratio=0 或 1）：cache 序列维度 = max_seq_len

**优先级**：🟡 中 — 当前实现可以工作（cache 过大但不会出错），优化后节省大量内存。

### 2.3 Task 15.1 — Paged + Quantized 组合

**论文参考**：V4 本身不需要这个（内置压缩已足够），但对通用模型（LLaMA/Mistral/Qwen）
在长上下文下至关重要。

**当前状态**：`PagedKVCache` 和 `QuantizedKVCache` 是两个平级的 `KVCacheStrategy`，
不能同时使用。

**TurboQuant 论文的启示**：3.5-bit 量化即可无损，所以 Paged + 4-bit 量化的组合
可以同时获得分页内存管理（减少碎片）和量化压缩（减少总量）的双重收益。

**优先级**：🟡 中 — 对通用模型的长上下文场景有价值。

---

## 三、TurboQuant 论文：对通用模型量化的指导

### 3.1 与当前 `kvcache/quantized.zig` 的对比

当前实现使用 MLX 内置的 `mlx_quantize`（affine 模式），这是均匀量化。
TurboQuant 提供了理论最优的替代方案：

| 维度 | 当前 affine 量化 | TurboQuant | MXFP4（刚绑定） |
|------|-----------------|------------|----------------|
| 量化方式 | 均匀 per-group | Lloyd-Max 最优 | E2M1 per-block |
| 内积偏差 | 有偏 | 无偏（+QJL） | 有偏 |
| 理论保证 | 无 | ≈2.7x 最优 | 无 |
| 实现复杂度 | 低（已有） | 中 | 低（已绑定） |
| 硬件加速 | Metal 原生 | 需自实现 | Metal 原生 |
| 适用场景 | 权重量化 | KV cache | 权重量化 |

### 3.2 MXFP4 vs TurboQuant：不同场景的最优选择

**权重量化**：MXFP4 更合适
- 已有 Metal kernel 加速（`mlx_quantized_matmul` with mode="mxfp4"）
- 训练时可用（量化感知训练）
- 生态支持好（HuggingFace、mlx-lm 已集成）

**KV cache 量化**：TurboQuant 更合适
- 在线量化（无需校准数据，KV cache 是动态生成的）
- 无偏内积（attention score 精度更高）
- 3.5-bit 无损（比 4-bit affine 更省内存且质量更好）

**建议路径**：
1. 当前：用已有的 affine 4-bit 做 KV cache 量化（已实现）
2. 近期：用 MXFP4 做权重量化（刚绑定，可直接用）
3. 远期：实现 TurboQuant 做 KV cache 量化（Phase 4+）

### 3.3 TurboQuant 在 Zig 中的实现方案

核心算法只需要 4 个步骤，全部可用 mlx-c 算子实现：

```
1. 随机旋转：y = Π · x          → mlx_matmul（Π 预生成，QR 分解）
2. 标量量化：idx = nearest(y, codebook)  → mlx_argmin + 预计算 codebook
3. 反量化：  ỹ = codebook[idx]   → mlx_take
4. 逆旋转：  x̃ = Π^T · ỹ       → mlx_matmul
```

QJL 残差修正（无偏内积）：
```
5. 残差：    r = x - x̃          → mlx_subtract
6. QJL：     z = sign(S · r)     → mlx_matmul + mlx_sign（S 预生成）
7. 重建：    x̂ = x̃ + √(π/2)/d · ‖r‖ · S^T · z  → 标准算子链
```

**存储开销**：
- Π 矩阵：d×d float32，d=128 时 64KB（per-model，一次性）
- S 矩阵：d×d float32，同上
- Codebook：2^b 个 float32，4-bit 时仅 64 bytes
- 量化后数据：b bits per coordinate（vs 16 bits float16）

---

## 四、更新后的优先级排序

### 立即可做（改动小，收益大）

1. **Task 15.3 升级为真正的 FP8** — 用刚绑定的 `toFp8()`/`fromFp8()` 替换 `astype(float16)`
   - 改动：~10 行 in `deepseek_v4.zig`
   - 收益：KV cache 内存减半（vs float16）

### 近期（需要一定工作量）

2. **Task 15.6 per-layer cache sizing** — 根据 `compress_ratio` 分配不同大小的 cache buffer
   - 改动：`loadModel` 中的 cache 分配逻辑
   - 收益：V4 内存使用更精确

3. **Task 15.1 Paged + Quantized** — 在 `PagedKVCache` 中内置量化选项
   - 改动：`kvcache/paged.zig` 新增 `kv_bits` 参数
   - 收益：通用模型长上下文内存优化

### 远期（Phase 4+）

4. **MXFP4 权重量化集成** — 用刚绑定的 `quantize(mode="mxfp4")` 支持 MXFP4 模型加载
5. **TurboQuant KV cache 量化** — 实现论文算法，替代 affine 量化

---

## 五、对 tasks.md 的修订建议

Task 15.3 的描述需要更新，反映 FP8 现在可用的事实：

```markdown
- [ ] 15.3 DeepSeek V4: upgrade FP8 KV storage to use native mlx_to_fp8/mlx_from_fp8
    - Current code at deepseek_v4.zig:1055-1080 uses astype(float16) as FP8 proxy
    - mlx-c 0.6.0 now has mlx_to_fp8/mlx_from_fp8, Zig bindings added in ops.zig
    - Replace astype(kv_storage_dtype) with ops.toFp8() for non-RoPE KV dimensions
    - Keep astype(.bfloat16) for RoPE dimensions
    - Add fromFp8() call before attention computation to restore precision
    - Remove kv_storage_dtype config field (no longer needed, FP8 is the V4 default)
```

新增 Task 15.8：

```markdown
- [ ]* 15.8 TurboQuant KV cache quantization (optional, Phase 4+)
    - Implement Lloyd-Max codebook precomputation for b=1,2,3,4
    - Implement random rotation (QR decomposition via mlx linalg)
    - Implement scalar quantize/dequantize with precomputed codebook
    - Implement QJL residual correction for unbiased inner products
    - Add --kv-quant simple|turbo CLI option (default: simple)
    - _Reference: TurboQuant paper arXiv:2504.19874_
```
