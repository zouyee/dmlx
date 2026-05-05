# mlx-zig DeepSeek V4 Flash — 全面修复规格文档
# 日期: 2026-05-03
# 目标: 7 Prompt 测试全部通过（无 SKIP），单元测试全部通过

---

## 一、现状总结

### 1.1 单元测试
`zig build test` 全部通过（exit code 0），50+ 模块覆盖核心组件。

### 1.2 7-Prompt 端到端测试
`scripts/best_test.sh` 中 P5 (`3*3=`) 和 P6 (`10-4=`) 被标记为 `KNOWN_ISSUE` 并跳过。
这不是真正的全部通过。

### 1.3 根因
预填充 argmax 存在 ~2 logit 系统性偏差，导致"陌生"数学推理失败。
`2+2=4` 能通过是因为训练数据高频出现（记忆型），而 `3*3=` 和 `10-4=` 需要真正的计算。

---

## 二、代码审计发现的问题

### BUG-1: post_mult 不一致（Metal kernel 2.0 vs Ops 路径 1.0）

**位置**: `src/models/deepseek_v4.zig`

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

**影响**: 43 层 × 每层 2 次 HC = 86 次累积偏差。这是 ~2 logit gap 的主要来源。

**修复**: 将 ops 路径的 `post_mult_value` 从 `1.0` 改为 `2.0`。

### BUG-2: mhcPreApplyMix 缺少 float32 精度提升

**位置**: `src/models/deepseek_v4.zig` L2452-2464

Python 参考实现在加权求和前将输入提升到 float32：
```python
(mix * residual.astype(mx.float32)).sum(axis=2)
```

Zig 实现直接在输入 dtype（通常是 bfloat16）上操作，没有精度提升。

**修复**: 在 multiply 前将 residual 和 mix 转为 float32，求和后转回原始 dtype。

### BUG-3: mhcPost 缺少 float32 精度提升

**位置**: `src/models/deepseek_v4.zig` L2467-2510

同 BUG-2，Python 在 matmul/combine 前提升到 float32。

**修复**: 在 matmul 和 add 前转为 float32，结果转回原始 dtype。

### BUG-4: sinkhornNormalize 使用普通 softmax 而非 softmaxPrecise

**位置**: `src/models/deepseek_v4.zig` L2415

```zig
const softmaxed = try ops.softmax(ctx, x, &[_]i32{-1});
```

Python 参考实现使用 `precise=True`。

**修复**: 改为 `ops.softmaxPrecise`。

### BUG-5: max_new_tokens=0 下溢未保护

**位置**: `src/models/deepseek_v4.zig` generate 函数

```zig
for (0..max_new_tokens - 1) |_| {
```

当 `max_new_tokens=0` 时，`0 - 1` 在 usize 下溢为 `maxInt(usize)`，导致无限循环。

**修复**: 在 generate 开头添加 guard。

### BUG-6: best_test.sh 跳过失败测试

**位置**: `scripts/best_test.sh`

P5 和 P6 使用 `KNOWN_ISSUE` 状态跳过，不是真正的测试通过。

**修复**: 修复 BUG-1~5 后，将 P5 和 P6 改为 `PASS` 状态，增加 max_tokens 以适应 chat 模式输出。

---

## 三、修复计划

| 优先级 | BUG | 文件 | 修改内容 |
|--------|-----|------|---------|
| P0 | BUG-1 | `deepseek_v4.zig` L2576 | `1.0` → `2.0` |
| P0 | BUG-2 | `deepseek_v4.zig` mhcPreApplyMix | 添加 float32 转换 |
| P0 | BUG-3 | `deepseek_v4.zig` mhcPost | 添加 float32 转换 |
| P1 | BUG-4 | `deepseek_v4.zig` sinkhornNormalize | softmax → softmaxPrecise |
| P1 | BUG-5 | `deepseek_v4.zig` generate | 添加 max_new_tokens guard |
| P2 | BUG-6 | `scripts/best_test.sh` | 移除 KNOWN_ISSUE，改为 PASS |

---

## 四、验证标准

1. `zig build test` — 全部通过（exit code 0）✅ 已验证
2. `zig build` — 编译无错误 ✅ 已验证
3. `scripts/best_test.sh` — 7 个 prompt 全部 PASS，0 个 SKIP，0 个 FAIL（需模型文件验证）
4. 无回归：现有单元测试不受影响 ✅ 已验证

---

## 五、风险评估

- BUG-1 修复（post_mult 2.0）是最高影响修复，直接影响 86 次 HC 计算
- BUG-2/3 修复（float32）是精度改善，不会改变正确输入的结果
- BUG-4 修复（softmaxPrecise）是精度改善
- BUG-5 修复（guard）是安全修复，不影响正常路径
- 所有修复都是向 Python 参考实现对齐，风险低
