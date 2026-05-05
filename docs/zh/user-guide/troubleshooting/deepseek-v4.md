# 故障排除：DeepSeek V4

**组件**：DeepSeek V4 模型（`src/models/deepseek_v4.zig`、`src/models/deepseek_v4_loader.zig`）
**规格**：`.kiro/specs/production-deployment/design-deepseek-v4.md`
**属性**：属性 V1（反量化一致性）、属性 V2（注意力分发）

## 症状：模型生成乱码/随机字符

**首次发现**：2026-04-29（提交 `a18bc24`、`e11c3b9`）
**根本原因**：聊天模板使用了全角 Unicode 字符（`｜` U+FF5C、`▁` U+2581）而非 ASCII（`|`、`_`），导致 tokenizer 将特殊 token 分割为子 token。

### 规格参考

- `design-deepseek-v4.md` §1 — 权重加载（间接：tokenizer 配置是模型加载的一部分）
- `requirements.md` R6.2 — 模型注册表查找

### 代码检查点

1. `src/tokenizer/chat_template.zig` — 特殊 token 定义
2. `src/main.zig` — BOS token 验证（预期：100000）
3. `src/tokenizer/bpe.zig` — 子 token 分割逻辑

### 诊断步骤

1. 打印提示词的前 10 个 token ID。预期：`[100000, ...]`（单个 BOS）
2. 如果 BOS 被分割为多个 token → 检查聊天模板中的 Unicode 字符
3. 验证特殊 token 使用 ASCII 格式：`<|begin_of_sentence|>`、`<|User|>`、`<|Assistant|>`

### 修复摘要

将特殊 token 从全角格式修正为 ASCII 格式：
```diff
- .bos_token = "<｜begin▁of▁sentence｜>",
+ .bos_token = "<|begin_of_sentence|>",
```

### 预防措施

- `src/tests/chat_template_tests.zig` — 单元测试验证 token 化往返一致性
- `scripts/verify-deepseek-v4-fix.sh` — 自动化验证脚本

---

## 症状：`mhcPreNormFn` 因 reshape 错误而崩溃

**首次发现**：2026-04-27
**根本原因**：`hc_head.fn` 权重未被加载器消费（显示为"未使用的权重"）。此外，所有 `.scales`/`.biases` 量化元数据键均未被使用。

### 规格参考

- `design-deepseek-v4.md` §1 — 权重加载与清理
- 属性 V1：FP4/FP8 反量化一致性

### 代码检查点

1. `src/models/deepseek_v4_loader.zig:mapV4LayerWeight` — 权重名称映射
2. `src/models/deepseek_v4_loader.zig:buildDSV4Model` — `hc_head` 加载块
3. `src/models/deepseek_v4_loader.zig:consumeWeightAndMeta` — 量化元数据消费

### 诊断步骤

1. 使用详细日志运行 — 检查"未使用的权重"警告
2. 如果 `hc_head.*` 或 `*.scales`/ `*.biases` 显示为未使用 → 加载器未消费它们
3. 检查所有权重加载路径是否都应用了 `consumeWeightAndMeta()`

### 修复摘要

- 创建了 `consumeWeightAndMeta()` 辅助函数，可同时移除 `.weight`、`.scales`、`.biases`
- 应用于 `buildDSV4Model` 中的所有权重加载
- 修正了 `hc_head` 权重名称映射（`hc_head.fn` vs `hc_head.fn_weight`）

### 预防措施

- 加载器应在加载结束时对任何遗留的"未使用权重"报错（严格模式）
- 添加单元测试：`buildDSV4Model` 执行后，HashMap 应为空

---

## 症状：在 48GB Mac 上加载 DeepSeek V4 Flash 4-bit 时出现 OOM

**首次发现**：2026-04-27
**根本原因**：专家权重的即时反量化（4-bit → float16）将内存扩展了 4 倍。mlx-c 的 `mlx_load_safetensors` 会即时加载完整的分片数据，不同于 Python 的延迟内存映射数组。

### 规格参考

- `design-deepseek-v4.md` §1 — 权重加载
- `design.md` §6.3 — 内存限制器

### 代码检查点

1. `src/models/deepseek_v4_loader.zig:loadShardedWeights` — 分片加载策略
2. `src/main.zig:runDeepSeekV4Chat` — MLX 内存限制（`mlx_set_wired_limit`）
3. `src/models/deepseek_v4_loader.zig:buildDSV4Model` — 反量化调用

### 诊断步骤

1. 在 `buildDSV4Model` 期间监控 RSS。预期：延迟加载时约 ~400MB。
2. 如果 RSS > 10GB → 检查是否有即时 `dequantIfNeeded` 调用
3. 检查 `wired_limit` 和 `cache_limit` 是否已设置（分别为系统内存的 50% 和 25%）

### 修复摘要

- 移除了 `loadShardedWeights` 中的 bfloat16→float32 转换
- 移除了 `splitFusedExperts` 中的即时反量化
- 将 `wired_limit` 设置为系统内存的 50%，`cache_limit` 设为 25%
- 切换为使用 `c_allocator` 而非 `DebugAllocator`

### 预防措施

- 添加内存基准测试：模型加载期间的 RSS 必须 < 1GB（未实现权重）
- 文档说明："除非绝对必要，否则绝不在加载期间调用 astype/dequantize"

---

## 症状：注意力反量化 + 前向传播成为解码瓶颈（每 token 约 10 分钟以上）

**首次发现**：2026-04-28
**状态**：部分解决 — 受限于完整的专家加载
**根本原因**：注意力权重在每次前向传播时都进行反量化，而非保持量化状态 + 使用 `quantizedMatmul`。

### 规格参考

- `design-deepseek-v4.md` §2 — DSV4Attention
- `design-deepseek-v4.md` §1 — 量化配置

### 代码检查点

1. `src/models/deepseek_v4.zig:DSV4Attention.forward` — wq_a、wq_b、wkv、wo_b 的 Matmul 调用
2. `src/models/deepseek_v4_loader.zig` — 注意力权重是否存储为 `QuantizedWeight` 还是 `Array`
3. `src/quantize.zig:quantizedMatmul` — 融合反量化+matmul 的可用性

### 诊断步骤

1. 使用 Metal System Trace 进行分析 — 统计每个解码步骤的内核启动次数
2. 如果注意力 matmul 每个权重显示约 10 次内核启动 → 在打包权重上使用了普通 matmul
3. 检查 `DSV4Attention` 字段是否为 `?quantize_mod.QuantizedWeight`（应为）还是普通 `Array`

### 修复摘要（部分）

- `wq_a`、`wq_b`、`wkv`、`wo_b` 在加载时进行反量化作为临时修复
- 长期方案：在注意力前向传播中使用 `quantizedMatmul` 以避免反量化开销
- tasks.md 中的 Task 51 跟踪此优化

### 预防措施

- 基准测试：在目标硬件上，解码步骤每 token 必须在 500ms 内完成
- 在 CI 中添加性能回归测试

---

## 症状：`wo_a` 反量化产生错误形状

**首次发现**：2026-04-28
**根本原因**：在延迟加载的 `wo_a` 权重 `[8192, 512]` uint32 上调用 `mlx_dequantize` 返回大小为 4194304 的数组，而非预期的 `[8192, 4096]`（解包后）。

### 规格参考

- `design-deepseek-v4.md` §2 — 分组输出投影
- 属性 V1：FP4/FP8 反量化一致性

### 代码检查点

1. `src/models/deepseek_v4_loader.zig:buildDSV4Model` — wo_a 加载块
2. 检查 `wo_a_raw.shape()`、`wo_a_raw.dtype()`、`wo_a_deq.shape()`

### 诊断步骤

1. 打印打包权重的形状和 dtype：预期 `[8192, 512]` uint32（affine 4-bit, group_size=64）
2. 打印反量化后的形状：预期 `[8192, 4096]` bfloat16
3. 如果形状不匹配 → 检查反量化参数（group_size、bits、mode）与实际量化是否一致

### 修复摘要

- 根本原因是延迟加载数组的形状元数据不匹配
- 临时方案：在 reshape 之前强制执行求值
- 正式修复：确保 `mlx_dequantize` 参数与检查点格式完全一致

### 预防措施

- 在加载器中每次反量化调用后添加形状断言
- 黄金测试：加载一层，验证所有权重形状符合预期

---

## 症状：模型生成错误的 token（例如 `2+2=` → `"That's a classic!"` 而非 `4`）

**首次发现**：2026-05-01
**状态**：已解决
**根本原因**：`deepseek_v4_loader.zig` 中的 `dequantIfNeeded` 和 embed/lm_head 反量化硬编码了量化模式 `"affine"` 且 `group_size=64`。DeepSeek V4 Flash 4-bit 对专家权重和嵌入层使用 **mxfp4**（无 biases，`group_size=32`）。MLX 以错误的去量化参数解释打包的 uint32 权重数据，产生错误的嵌入向量，并传播至全部 43 层。

### 规格参考

- `design-deepseek-v4.md` §1 — 权重加载与量化
- 属性 V1：FP4/FP8 反量化一致性

### 代码检查点

1. `src/models/deepseek_v4_loader.zig:dequantIfNeeded` — 反量化模式检测
2. `src/models/deepseek_v4_loader.zig:buildDSV4Model` — Embed/lm_head 反量化块
3. `src/models/deepseek_v4.zig:DSV4Expert.expertMatmul` — `quant_mode` 字段使用
4. `src/models/deepseek_v4.zig:DSV4Attention` — `attn_quant_mode` 字段使用

### 诊断步骤

1. 从 `config.json` 检查量化配置：`mode="mxfp4"`、`group_size=32`、不存在 `bias`
2. 验证 `dequantIfNeeded` 是否正确区分 mxfp4 与 affine：
   - mxfp4：`biases == null` → `group_size=32`、`mode="mxfp4"`
   - affine：`biases != null` → `group_size=64`、`mode="affine"`
3. 检查 `DSV4Expert.quant_mode` 和 `DSV4Attention.attn_quant_mode` 是否从加载器配置中设置，而非硬编码 `.affine`
4. 如果模型生成看似合理但语义错误的文本 → 怀疑反量化不匹配（而非路由或注意力 bug）

### 修复摘要

- **`dequantIfNeeded`**：基于 `biases == null` 自动检测 mxfp4 与 affine：
  ```zig
  const is_mxfp4 = biases == null;
  const gs = if (is_mxfp4) 32 else config.quantize_default_group_size;
  try c.check(c.c.mlx_dequantize(&res, weight.inner, scales.inner, biases_inner, opt_group, opt_bits,
      if (is_mxfp4) "mxfp4" else "affine", null_array, no_dtype, ctx.stream.inner));
  ```
- **Embed/lm_head 反量化**：应用了相同的自动检测。
- **`DSV4Expert`**：添加了 `quant_mode: quantize_mod.QuantMode` 字段（默认 `.affine` → 现在从加载器设置）。
- **`DSV4Attention`**：添加了 `attn_quant_mode` 字段，传递给所有 6 个 `quantizedMatmul` 调用。

### 预防措施

- **绝不硬编码量化模式**：始终从检查点元数据（`biases` 是否存在、`config.json` 字段）推导。
- **添加黄金测试**：加载单个专家权重，进行反量化，并将前几个值与 Python 参考值比较。
- **添加集成测试**：提示词 `"2+2="` 必须生成包含正确答案 token 的 token 序列。

### 相关说明

- **CORRECTNESS TEST 陷阱**：流模式 `CORRECTNESS TEST` 使用 `TensorIndex.loadTensor`，它加载原始字节而不调用 `dequantIfNeeded`。即使实际模型前向传播存在问题时它也通过了测试，因为它测试的是不同的代码路径。务必验证生产环境的前向传播路径。
- **STREAM WEIGHTS TEST 差异**：流加载的切片权重（`readExpertRows`/`takeAxis`）产生的 `gatherQmm` 输出与完整权重略有不同（第一个元素差异约 2 倍）。这是一个次要的精度问题，不阻塞正确生成，但将来可能需要调查。
