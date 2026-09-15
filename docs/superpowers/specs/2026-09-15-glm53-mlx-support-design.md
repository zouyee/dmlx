# GLM-5.3-Flash MLX 路径支持设计

> **日期**：2026-09-15
> **状态**：已获批准（方案 A）
> **目标模型**：`~/models/GLM-5.3-Flash-MLX-mixed-4_8bit`（125GB，纯文本权重，MLX affine 4/8bit 混合量化）
> **Oracle**：mlx-vlm `glm5_next`（Python MLX 参考实现，`mlx_vlm/models/glm5_next/language.py`）

## 1. 背景与目标

GLM-5.3-Flash（`Glm5NextForConditionalGeneration`）是 Z.ai 的 320B-A18B 混合注意力 MoE：45 层 + MTP，34 层 KDA 线性注意力 + 11 层 NoPE MLA+DSA 稀疏注意力，Hyper-Connections（hc_mult=4），288 routed experts top-8 + 1 shared。

**目标**：在 dmlx 的 Zig + MLX 栈上支持该模型的文本推理（server + CLI），正确性优先，不定 tok/s 指标。

**明确不在本期范围**：MTP 投机层、vision tower（本地副本已剥离）、native metal kernel 化、prefill 的 KDA chunked scan 优化。

## 2. 架构映射与复用

mlx-vlm 参考实现确认 GLM-5.3-Flash 与 dmlx 已有 DSV4 实现高度同族：

| GLM-5.3 组件 | dmlx 复用 | 备注 |
|---|---|---|
| HyperConnection（attn_hc/ffn_hc，hc_mult=4，Sinkhorn 20 次） | ✅ `deepseek_v4.zig:2176-2592` mHC 函数族 | mlx-vlm 直接从 deepseek_v4 模块 import |
| MoE gate（sigmoid + e_score_correction_bias + noaux_tc + norm_topk_prob + scaling 2.5） | ✅ `DSV4Gate`（:522-687） | sigmoid+bias 已支持 |
| SwitchGLU experts + Limited SwiGLU（limit=10） | ✅ `DSV4SwitchGLU`/`DSV4Expert`/`DSV4MoE` | limit 与 DSV4 相同 |
| shared expert | ✅ `DSV4MoE` 内置 | |
| dense MLP（gate_up_proj fused + limited swiglu） | ✅ `DSV4Expert` 充当 | 前 3 层 dense（first_k_dense_replace=3） |
| NoPE MLA + DSA（11 层） | ⚠️ 新写 `GlmAttention`/`GlmIndexer` | latent KV rank 512、per-head absorbed 投影（embed_q/unembed_out MultiLinear）、indexer kpool=4 池化压缩 + top-2048 + always-select-tail；**无 RoPE**；可参考 `deepseek_v4.zig` Indexer 模式 |
| KDA 线性注意力（34 层） | ❌ 全新 `GlmKDA` | 见 §3 |
| 主干（embed → repeat HC → 45 层 → h.mean(HC) → norm → lm_head） | 新写（模式同 DSV4Model） | lm_head 不 tied |

## 3. KDA 模块设计

参考 mlx-vlm `Glm5NextLinearAttention` + `gated_delta_update`（kimi_linear 同族）：

- `qkv_proj` → depthwise short-conv1d（k=4，silu，带 conv window 缓存）→ split q/k/v（64 头 × head_dim 128）
- q/k 经 RMSNorm + scale 调整
- gated delta rule 循环：`A_log` 衰减、`dt_bias`、gate lower_bound=-5
- 输出 `o_norm(out) * sigmoid(gate)` → `o_proj`
- **decode**：逐 token 更新 SSM 状态（纯 MLX ops，无专用 kernel）
- **prefill**：naive O(L) 逐 token 循环（正确性优先，已知慢，接受）
- 状态：conv window（k-1）+ SSM recurrent state，新增 `GlmKdaCache`（见 §5）

**为什么本期不写专用 kernel**（回应评审问题）：mlx-vlm 的 `gated_delta_update` 专用 kernel 面向 GPU-bound 数据中心吞吐（chunked prefill scan）；本机是 SSD I/O-bound（DSV4 decode wall ~360ms/token）。纯 MLX ops 的 decode 开销估算：SSM 状态 ~4MB/层（fp32，[64,128,128]）× 34 层，dispatch ~2-5ms/token + 状态流量 ~5ms/token，合计 <10ms/token，淹没在 I/O 噪声里。prefill naive 循环 100-token 约 +0.5-1s，一次性可接受。**触发条件**：M3 oracle 对齐后 profile，若 KDA MLX 路径实测 >20ms/token（如 lazy eval 图碎片化导致状态反复物化），再下沉 fused Metal kernel——届时数学已被 oracle 锁定，有金标准可对。

## 4. 新增/改动文件

### 新增

- `src/models/glm53.zig` — `GlmKDA`、`GlmAttention`（含 `GlmIndexer`）、`GlmDecoderLayer`、`Glm53Model`；复用 `DSV4Gate`/`DSV4SwitchGLU`/`DSV4MoE`/`DSV4Expert`/mHC 函数族
- `src/models/glm53_loader.zig` — config.json 解析（text_config 嵌套 + linear_attn_config + quantization_config per-tensor 覆盖）+ 权重加载。affine 混合量化按 `llama_loader.zig:243-270` `tryGetQuantizedWeight` 模式 per-tensor 探测 `.scales`/`.biases`（bits 4/8，gs64）。以下 tensor 保持 fp32 不反量化：`A_log`、`dt_bias`、`e_score_correction_bias`、`attn_hc.*`、`ffn_hc.*`、`index_kpool_compress_*`（与 mlx-vlm cast/quant predicate 一致）。embed_tokens（8bit）load 时反量化到 bf16

### 注册与 dispatch

- `src/model_registry.zig` — 注册 `"Glm5NextForConditionalGeneration"` → `glm53Loader`；mxfp4/gs32 硬编码（:326-328）改为从 config quantization 字段读取
- `src/main.zig` — `detectModelType()`/`runChat()` 增加 `glm5_next` 分支（config `model_type=glm5_next`，architectures 为 `Glm5NextForConditionalGeneration`）
- `src/server/state.zig` — template 初始化分支

### Chat template

- `src/tokenizer/chat_template.zig` — enum 加 `glm`，按本地 `chat_template.jinja` 手写等价 `applyGlm()` + `initGlm()`：前缀 `[gMASK]<sop>`，system 段用 `<|system|>`（可选 `Reasoning Effort: Max` 行），EOS 按 config eos_token_id 列表（154820/154827/154829）；工具调用/reasoning_effort/clear_thinking 等高级分支本期不支持，按无 tools 纯对话路径实现；`main.zig:595` 与 `state.zig:150` 检测分支。tokenizer 本体走 tokenizer.json，无需改动

## 5. Expert streaming（125GB on 48GB，必须）

- `src/models/expert_stream.zig` 硬编码泛化：`pred_experts: [43][6]u32` → config 驱动（45 层 / top-8）；DyMoE `[256]u32` → 288
- `LayerExpertMeta`（`expert_stream.zig:52-65`）增加 **biases** 名字与字节数字段；`expert_stream.zig:799-803` 的 `*_biases = null` 改为实际加载（affine 格式必需）
- GLM sanitize 后的 fused 3D 布局（`switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`，[288, ...]）与 streaming 逐 expert 切片约定兼容，tensor 命名映射在 `LayerExpertMeta` 构建处适配 GLM 的 `mlp.switch_mlp.*` 前缀
- SMELT preload eager 路径（含 biases，`expert_preload.zig:265-280`）直接可用

## 6. KV cache

- MLA 层：latent cache（[B, 1, N, 512]），用现有标准 KV cache 接口；indexer 侧需要 pool cache（kpool 压缩状态），参照 mlx-vlm `PoolingCache` 在模型内自管
- KDA 层：conv window + SSM 状态，新增 `GlmKdaCache`，实现 `KVCacheStrategy` VTable（`src/kvcache/interface.zig:85`）或模型内自管（参照 `DeepseekV4Cache` 先例）
- FP8 KV 不在本期（DSV4 的 FP8 KV 是模型内实现，非通用能力）

## 7. 验收

1. **mlx-vlm oracle 逐段对齐**：Python 跑 mlx-vlm dump embed 输出 / 首个 KDA 层输出 / 首个 MLA 层输出 / 首个 MoE 层输出 / final logits，dmlx 对应段 cos ≥ 0.999
2. **层级 golden**：复用 `tests/golden`（RMSNorm/SDPA/Embedding 等，模型无关）
3. **7-prompt e2e**：适配 `scripts/run_benchmark.sh`（MODEL_PATH + 期望子串 + chat template），greedy `--temperature 0`，全 PASS
4. **稳定 serve**：server 模式流式与非流式各一次完整 100-token 生成无崩溃

## 8. 风险

| 风险 | 缓解 |
|---|---|
| CLI/server dispatch 都不完全走 registry，容易漏改分支 | §4 清单逐个核对，e2e 覆盖两条路径 |
| stream 路径缺 biases 是 affine 模型的硬阻塞 | §5 先行实现，作为第一个里程碑 |
| KDA prefill naive 循环慢 | 已知并接受；decode 为主路径 |
| 288 experts / 45 层 / top-8 超出现有硬编码数组 | §5 泛化先行 |
| GLM indexer 的 kpool 压缩/always-select-tail 逻辑复杂 | 以 mlx-vlm 代码为准逐行对齐，oracle 分段验证 |
| mixed 4/8bit per-tensor bits 探测遗漏 | loader 对 `.scales` shape 反推 bits，与 mlx-vlm sanitize 的 bits 推导一致 |

## 9. 里程碑

1. **M1**：expert stream biases + 硬编码泛化（§5），不破坏 DSV4 回归（`zig build test` + DSV4 7-prompt）
2. **M2**：loader + 注册 + chat template，eager 路径能 load（不接 streaming，小内存下至少 backbone 可建）
3. **M3**：KDA + GlmAttention/Indexer + HC + MoE 全模型 eager forward，oracle 逐段对齐通过
4. **M4**：expert streaming 接入，125GB 全量加载 serve
5. **M5**：7-prompt e2e + 100-token 稳定生成，收尾
