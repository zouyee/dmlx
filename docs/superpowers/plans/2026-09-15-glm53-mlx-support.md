# GLM-5.3-Flash MLX 路径支持实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 dmlx 的 Zig + MLX 栈上支持 GLM-5.3-Flash（`Glm5NextForConditionalGeneration`）文本推理，正确性优先，server + CLI 可用。

**Architecture:** 复用 DSV4 组件（mHC、DSV4Gate/DSV4MoE/DSV4SwitchGLU、expert streaming），新写 KDA 线性注意力与 NoPE MLA+DSA 注意力；mlx-vlm `glm5_next` 作为数值 oracle 逐段对齐。离线 prep 脚本把 per-expert safetensors 堆叠成 streaming 需要的 fused 3D 布局。

**Tech Stack:** Zig（模型/loader/引擎）、MLX C API（计算图）、Python（权重 prep + oracle dump）、Metal（仅复用现有 affine kernel，本期不写新 kernel）。

**Spec:** `docs/superpowers/specs/2026-09-15-glm53-mlx-support-design.md`（架构映射、里程碑 M1-M5、验收标准以此为准）

**参考实现（oracle，执行时先下载到本地）：**

```bash
mkdir -p /tmp/glm5_next_ref && cd /tmp/glm5_next_ref
for f in config.py language.py glm5_next.py __init__.py; do
  curl -sLO "https://raw.githubusercontent.com/Blaizzy/mlx-vlm/main/mlx_vlm/models/glm5_next/$f"
done
curl -sLo gated_delta.py https://raw.githubusercontent.com/Blaizzy/mlx-vlm/main/mlx_vlm/models/gated_delta.py
```

**目标模型：** `~/models/GLM-5.3-Flash-MLX-mixed-4_8bit`（纯文本权重，prefix `language_model.model.`，per-expert tensors，affine 4bit gs64 为主 + 敏感层 8bit）

**关键 config 值（text_config，loader 以此为准）：**
hidden 4096 / layers 45 / vocab 154880 / first_k_dense_replace 3 / hc_mult 4 / hc_sinkhorn_iters 20 / hc_eps 1e-6 / kv_lora_rank 512 / intermediate 12288 / moe_intermediate 2048 / n_routed_experts 288 / num_experts_per_tok 8 / n_shared_experts 1 / norm_topk_prob true / routed_scaling_factor 2.5 / topk_method noaux_tc / swiglu_limit 10.0 / index_n_heads 32 / index_head_dim 128 / index_topk 2048 / index_kpool 4 / index_kpool_always_select_tail true / index_kpool_compress true / linear heads 64 / linear head_dim 128 / conv kernel 4 / gate_lower_bound -5.0 / full_attn_layers [3,7,11,15,19,23,27,31,35,39,43] / eos_token_id [154820,154827,154829] / model_type `glm5_next`

---

## Task 1: 权重 prep 脚本（offline sanitize + expert 堆叠）

**Files:**
- Create: `scripts/prep_glm53_weights.py`
- Test: 手动运行 + 输出校验（步骤内）

**背景：** 本地副本是 per-expert 布局（`...mlp.experts.{0..287}.{gate,up,down}_proj.{weight,scales,biases}`）。dmlx expert streaming（`expert_stream.zig` 的 `LayerExpertMeta` + `PartialTensorReader`）要求 fused 3D 张量 `[n_experts, out, packed_in]` 按 expert 维均匀切片。同时 mlx-vlm 的 `sanitize()` 还做：合并 `gate_up_proj`、`qkv_a_proj`、`qkv_proj`/`fbg_a_proj`（KDA）、`kv_b_proj` 拆成 `embed_q`/`unembed_out`、HC 重命名（`hc_attn_fn` → `attn_hc.fn` 等）。用 Python 一次性做完，Zig loader 只认处理后的布局。

**输出布局约定**（写入 `~/models/GLM-5.3-Flash-MLX-mixed-4_8bit-dmlx/`）：
- 目录用硬链接起步（`os.link` 复制全部 shard 和大文件，不占双份磁盘），仅对需要变换的 tensor 写新 shard
- 新 index：`model.safetensors.index.json` 重写
- tensor 命名（Zig loader 唯一约定，prefix 保留 `language_model.model.`）：
  - MoE 层：`...mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`（[288, ...] 堆叠）
  - MoE 层 shared：`...mlp.shared_experts.gate_up_proj.*`（gate/up concat）、`...mlp.shared_experts.down_proj.*`
  - dense 层：`...mlp.gate_up_proj.*`、`...mlp.down_proj.*`
  - KDA 层：`...self_attn.qkv_proj.*`（q/k/v concat）、`...self_attn.fbg_a_proj.*`（f_a/b/g_a concat）、`...self_attn.qkv_conv.weight`（3 个 depthwise conv concat，[3*proj, 1, k]→[3*proj, k, 1] 保持 MLX conv 布局不变，记录实际 shape 到 manifest）
  - MLA 层：`...self_attn.qkv_a_proj.*`（q_a + kv_a_with_mqa concat）、`...self_attn.embed_q.*` / `...self_attn.unembed_out.*`（kv_b 拆分转置，**保持量化格式**：拆后对 wk/wv 分别重新 affine 量化——直接用 `mlx.core.quantize`，与原 bits/gs 一致）
  - HC：`hc_{attn,ffn}_{fn,base,scale}` → `{attn,ffn}_hc.{fn,base,scale}` 重命名
  - 其余 tensor 原样（硬链接）
- 写 `dmlx_manifest.json`：per-tensor bits/group_size（从 quantization_config 的 per-tensor 覆盖推导）、conv weight 实际 shape、处理版本号

- [ ] **Step 1: 探查本地副本实际 key 布局**

```bash
python3 - <<'EOF'
from safetensors import safe_open
import glob, re
from collections import Counter
pats = Counter()
for f in sorted(glob.glob('/Users/zouyee/models/GLM-5.3-Flash-MLX-mixed-4_8bit/model-*.safetensors')):
    with safe_open(f, framework='numpy') as sf:
        for k in sf.keys():
            k2 = re.sub(r'\.\d+\.', '.N.', k)
            k2 = re.sub(r'experts\.N\.', 'experts.E.', k2)
            pats[k2] += 1
for p, c in sorted(pats.items()):
    print(c, p)
EOF
```
Expected: 打印全部 key pattern。确认 experts 是 per-expert 还是已堆叠；确认 KDA 层 conv/投影命名；确认 HC 命名。把输出贴到脚本头部注释作为 ground truth。

- [ ] **Step 2: 写 `scripts/prep_glm53_weights.py`**

核心结构（照着 mlx-vlm `sanitize()` 的语义写，以上一步探查的真实 key 为准）：

```python
#!/usr/bin/env python3
"""GLM-5.3-Flash MLX mixed-4_8bit → dmlx fused 布局（expert 堆叠 + 投影合并）。

用法: python3 scripts/prep_glm53_weights.py SRC_DIR DST_DIR
硬链接未变换 tensor；变换 tensor 写新 shard（mlx.core 做 quantize/dequantize）。
"""
import json, os, sys, shutil
import numpy as np
import mlx.core as mx
from safetensors import safe_open
from safetensors.numpy import save_file as st_save

N_EXPERTS = 288
FULL_ATTN = {3,7,11,15,19,23,27,31,35,39,43}
N_LAYERS = 45
FP32_KEEP = ("A_log", "dt_bias", "e_score_correction_bias",
             "attn_hc", "ffn_hc", "hc_attn", "hc_ffn", "index_kpool_compress")

def load_all(src):
    tensors, index = {}, {}
    for name in sorted(os.listdir(src)):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(src, name), framework="mlx") as sf:
                for k in sf.keys():
                    tensors[k] = sf.get_tensor(k)
                    index[k] = name
    return tensors, index

def fuse(tensors, keys, axis=0):
    """concat 存在的一组同名后缀 tensor；返回 None 若不存在。"""
    present = [k for k in keys if k in tensors]
    if not present or len(present) != len(keys):
        return None
    out = mx.concatenate([tensors.pop(k) for k in keys], axis=axis)
    return out

# 主流程（伪代码骨架，执行者按 sanitize 语义补全并打印每步计数）：
# for layer in range(N_LAYERS):
#   p = f"language_model.model.layers.{layer}"
#   1. HC 重命名: .hc_attn_fn. → .attn_hc.fn. 等（纯 key rename）
#   2. MoE 层 (layer >= 3):
#      for proj in gate/up/down: 堆叠 experts → f"{p}.mlp.switch_mlp.{proj}_proj.{sfx}"
#      (sfx ∈ weight/scales/biases, axis=0 → [288, ...])
#      shared_experts gate/up → gate_up_proj concat axis=0
#      gate bias: f"{p}.mlp.gate.e_score_correction_bias" 原样保留
#   3. dense 层 (layer < 3): mlp.gate_proj+up_proj → gate_up_proj
#   4. KDA 层 (layer not in FULL_ATTN):
#      q/k/v_proj → qkv_proj; f_a/b/g_a_proj → fbg_a_proj; (各 sfx concat axis=0)
#      q/k/v_conv1d.weight → qkv_conv.weight concat (记录 shape)
#   5. MLA 层 (layer in FULL_ATTN):
#      q_a_proj + kv_a_proj_with_mqa → qkv_a_proj
#      kv_b_proj → dequantize → reshape [64, nope+v, 512] → split →
#      wk [64,512,nope]→ embed_q, wv → unembed_out, 各自重新 quantize 同 bits/gs
#      indexer 权重原样（wk/k_norm/weights_proj/wq_b/index_kpool_compress_*）
# 写出: 变换 tensor 合并进新 shard(safetensors, 每个 ≤4GB)，未变换 tensor 硬链接。
# 重写 index json + dmlx_manifest.json。打印变换统计。
```

- [ ] **Step 3: 运行 prep（全量，约 10-30 分钟）**

```bash
python3 scripts/prep_glm53_weights.py \
  ~/models/GLM-5.3-Flash-MLX-mixed-4_8bit \
  ~/models/GLM-5.3-Flash-MLX-mixed-4_8bit-dmlx
```
Expected: 末尾打印 `switch_mlp stacked: 42 layers × 3 proj`、`kda fused: 34`、`mla fused: 11`、`hardlinked: N`，`du -sh` 目标目录 ≈ 125GB（硬链接）+ 变换 shard 增量。

- [ ] **Step 4: 校验输出**

```python
# scripts/prep_glm53_weights.py 内置 --verify 模式：
# 1. 每层 switch_mlp.gate_proj.weight shape[0] == 288
# 2. 随机抽 layer 10 expert 7：堆叠后切片 == 源 per-expert tensor（bit-exact）
# 3. embed_q/unembed_out 反量化后与源 kv_b_proj 反量化值 cos ≥ 0.9999
# 4. FP32_KEEP 名单 tensor 全部存在且 dtype 为 float32/bfloat16
```
Expected: 4 项全 PASS。

- [ ] **Step 5: Commit**

```bash
git add scripts/prep_glm53_weights.py
git commit -m "feat(glm53): 权重 prep 脚本——expert 堆叠 + 投影合并为 dmlx fused 布局"
```

---

## Task 2: oracle dump 脚本（mlx-vlm 数值金标准）

**Files:**
- Create: `scripts/glm53_oracle_dump.py`
- 产物: `tests/fixtures/glm53_oracle/`（git LFS 不管，直接 .npy，几十 MB 内）

**背景：** 用 mlx-vlm 加载**源模型**（未 prep 的原始副本），对固定 prompt 逐段 dump 中间激活，作为 Zig 实现的对齐基准（cos ≥ 0.999）。固定输入用 DSV4 batch oracle 同款 fixture token 序列的变体：`[17,42,314,2718,7,99,1234]`（7 token，避免长 prefill 的 KDA naive 循环拖累后续 Zig 侧对齐）。

- [ ] **Step 1: 写 dump 脚本**

```python
#!/usr/bin/env python3
"""mlx-vlm glm5_next oracle dump：固定 7-token 输入的逐段激活金标准。
用法: python3 scripts/glm53_oracle_dump.py MODEL_DIR OUT_DIR
输出: embed.npy, layer{L}_{kda|mla}_out.npy（每类首层 + 第 10/30 层）,
      moe_out.npy（第 10 层）, hidden_final.npy, logits.npy, meta.json
实现要点:
- mlx_vlm 加载模型（仅 text：model.language_model）
- input_ids = mx.array([[17,42,314,2718,7,99,1234]])
- 手动逐层调用（仿 Glm5NextTextModel.__call__ 但逐段 eval + 保存）：
  h = embed → repeat hc_mult → 每层 (attn 分支输出, mlp 输出) 分别 dump 指定层
- logits = lm_head(norm(mean(h, axis=2)))
- meta.json: input_ids, dtype, mlx/mlx-vlm 版本, 各 tensor shape
"""
```
（执行者按 `/tmp/glm5_next_ref/language.py` 的实际调用签名实现；dump 全部用 `mx.eval` 后 `.astype(mx.float32)` 存 npy。）

- [ ] **Step 2: 安装依赖并运行**

```bash
pip install -U "mlx-vlm>=0.6.17" safetensors numpy
python3 scripts/glm53_oracle_dump.py \
  ~/models/GLM-5.3-Flash-MLX-mixed-4_8bit \
  tests/fixtures/glm53_oracle
ls -la tests/fixtures/glm53_oracle/
```
Expected: 生成约 10 个 .npy + meta.json；`logits.npy` shape `[1, 7, 154880]`。

- [ ] **Step 3: sanity check——oracle 自洽**

脚本内附 `--self-check`：同一输入跑两次（eager 与 `use_kernel=True/False` 的 KDA 路径），logits cos ≥ 0.9999。不通过说明 dump 环境不确定，需先修环境再继续。

- [ ] **Step 4: Commit**

```bash
git add scripts/glm53_oracle_dump.py tests/fixtures/glm53_oracle/meta.json
git commit -m "test(glm53): mlx-vlm oracle dump 脚本 + fixture 元信息"
```
（.npy 体积若 >50MB，只 commit 脚本与 meta.json，NOTES 里写重新生成命令；fixture 生成命令必须可复现。）

---

## Task 3: Glm53Config 解析（loader 第一步）

**Files:**
- Create: `src/models/glm53_loader.zig`
- Test: `src/models/glm53_loader.zig`（文件内 `test` block，仓库惯例）

- [ ] **Step 1: 写 failing test**

```zig
// src/models/glm53_loader.zig 末尾
test "parseGlm53Config: text_config 嵌套 + linear_attn_config" {
    const allocator = std.testing.allocator;
    const json_text =
        \\{"model_type":"glm5_next","text_config":{
        \\  "hidden_size":4096,"num_hidden_layers":45,"vocab_size":154880,
        \\  "first_k_dense_replace":3,"hc_mult":4,"hc_sinkhorn_iters":20,
        \\  "kv_lora_rank":512,"intermediate_size":12288,"moe_intermediate_size":2048,
        \\  "n_routed_experts":288,"num_experts_per_tok":8,"n_shared_experts":1,
        \\  "norm_topk_prob":true,"routed_scaling_factor":2.5,"swiglu_limit":10.0,
        \\  "index_n_heads":32,"index_head_dim":128,"index_topk":2048,"index_kpool":4,
        \\  "index_kpool_always_select_tail":true,"index_kpool_compress":true,
        \\  "rms_norm_eps":1e-6,"eos_token_id":[154820,154827,154829],
        \\  "linear_attn_config":{"num_heads":64,"head_dim":128,
        \\    "short_conv_kernel_size":4,"gate_lower_bound":-5.0,
        \\    "full_attn_layers":[3,7,11]}
        \\}}
    ;
    const cfg = try parseGlm53Config(allocator, json_text);
    defer cfg.deinit();
    try std.testing.expectEqual(@as(u32, 45), cfg.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 288), cfg.n_routed_experts);
    try std.testing.expectEqual(@as(u32, 64), cfg.linear_num_heads);
    try std.testing.expect(cfg.isFullAttention(3));
    try std.testing.expect(!cfg.isFullAttention(4));
    try std.testing.expectEqual(@as(usize, 3), cfg.eos_token_ids.len);
}
```

- [ ] **Step 2: 运行确认失败**

```bash
zig build test 2>&1 | head -20
```
Expected: 编译错误（`parseGlm53Config` 未定义）。注：先把 `glm53_loader.zig` 挂进测试 root——检查 `build.zig` 中 test 目标的 root 文件（DSV4 loader 的 test 如何被包含，照其 import 链加入）。

- [ ] **Step 3: 实现 Glm53Config + parseGlm53Config**

```zig
// src/models/glm53_loader.zig
const std = @import("std");

pub const Glm53Config = struct {
    allocator: std.mem.Allocator,
    hidden_size: u32 = 4096,
    num_hidden_layers: u32 = 45,
    vocab_size: u32 = 154880,
    first_k_dense_replace: u32 = 3,
    hc_mult: u32 = 4,
    hc_sinkhorn_iters: u32 = 20,
    hc_eps: f32 = 1e-6,
    kv_lora_rank: u32 = 512,
    q_lora_rank: u32 = 1536, // 以 config 实际值为准（text_config.q_lora_rank）
    num_attention_heads: u32 = 64, // config.num_attention_heads
    qk_nope_head_dim: u32 = 128, // config.qk_nope_head_dim（NoPE：无 rope 维）
    v_head_dim: u32 = 128, // config.v_head_dim
    intermediate_size: u32 = 12288,
    moe_intermediate_size: u32 = 2048,
    n_routed_experts: u32 = 288,
    num_experts_per_tok: u32 = 8,
    n_shared_experts: u32 = 1,
    norm_topk_prob: bool = true,
    routed_scaling_factor: f32 = 2.5,
    swiglu_limit: f32 = 10.0,
    rms_norm_eps: f32 = 1e-6,
    index_n_heads: u32 = 32,
    index_head_dim: u32 = 128,
    index_topk: u32 = 2048,
    index_kpool: u32 = 4,
    index_kpool_always_select_tail: bool = true,
    index_kpool_compress: bool = true,
    linear_num_heads: u32 = 64,
    linear_head_dim: u32 = 128,
    linear_conv_kernel: u32 = 4,
    linear_gate_lower_bound: f32 = -5.0,
    eos_token_ids: []u32 = &.{},
    full_attn_layers: []u32 = &.{},
    // quantization（从 quantization_config + dmlx_manifest.json 合并）
    quant_bits_default: u8 = 4,
    quant_group_size: u32 = 64,

    pub fn deinit(self: *Glm53Config) void {
        self.allocator.free(self.eos_token_ids);
        self.allocator.free(self.full_attn_layers);
    }
    pub fn isFullAttention(self: *const Glm53Config, layer: u32) bool {
        for (self.full_attn_layers) |l| if (l == layer) return true;
        return false;
    }
    pub fn isMoE(self: *const Glm53Config, layer: u32) bool {
        return layer >= self.first_k_dense_replace;
    }
};

pub fn parseGlm53Config(allocator: std.mem.Allocator, json_text: []const u8) !Glm53Config {
    // std.json 解析；text_config 嵌套取值（缺失回退 text_config 同级顶层），
    // linear_attn_config 子对象取 linear_* 与 full_attn_layers。
    // 参照 deepseek_v4_loader.zig:310 parseDSV4Config 的解析模式。
    ...
}
```
（解析实现照 `deepseek_v4_loader.zig:310` 的 `std.json` 模式；所有字段从上表读，缺字段用默认值。）

- [ ] **Step 4: 测试通过 + 真 config 冒烟**

```bash
zig build test 2>&1 | tail -5
python3 -c "import json; print(json.load(open('/Users/zouyee/models/GLM-5.3-Flash-MLX-mixed-4_8bit-dmlx/config.json'))['text_config']['q_lora_rank'])"
```
Expected: test PASS；打印的真实 q_lora_rank 与代码默认值一致（不一致则以真值为准修正）。

- [ ] **Step 5: Commit**

```bash
git add src/models/glm53_loader.zig build.zig
git commit -m "feat(glm53): Glm53Config 解析"
```

---

## Task 4: expert stream 泛化 + biases 支持（Spec M1，先行里程碑）

**Files:**
- Modify: `src/models/expert_stream.zig`（:52-65 `LayerExpertMeta`、:94-104 硬编码数组、:790-810 biases=null 处）
- Modify: `src/models/expert_pread.zig`（若 pread 路径也组装 gatherQmm 参数）
- Test: 现有 `zig build test` 全绿 + DSV4 回归

**背景：** GLM 需要 45 层/top-8/288 experts（现为 [43][6]/[256] 硬编码），且 affine 格式必须有 biases（现 stream 路径写死 null，见探索发现 `expert_stream.zig:799-803`）。

- [ ] **Step 1: 写 failing test——LayerExpertMeta biases 字段**

```zig
// src/models/expert_stream.zig 内 test
test "LayerExpertMeta: biases 名字与字节数" {
    const meta = LayerExpertMeta{
        .gate_proj_name = "language_model.model.layers.5.mlp.switch_mlp.gate_proj.weight",
        .up_proj_name = "language_model.model.layers.5.mlp.switch_mlp.up_proj.weight",
        .down_proj_name = "language_model.model.layers.5.mlp.switch_mlp.down_proj.weight",
        .gate_scales_name = "language_model.model.layers.5.mlp.switch_mlp.gate_proj.scales",
        .up_scales_name = "language_model.model.layers.5.mlp.switch_mlp.up_proj.scales",
        .down_scales_name = "language_model.model.layers.5.mlp.switch_mlp.down_proj.scales",
        .gate_biases_name = "language_model.model.layers.5.mlp.switch_mlp.gate_proj.biases",
        .up_biases_name = "language_model.model.layers.5.mlp.switch_mlp.up_proj.biases",
        .down_biases_name = "language_model.model.layers.5.mlp.switch_mlp.down_proj.biases",
        .expert_row_bytes = 2048 * 4096 / 2, // 4bit packed
        .expert_scale_row_bytes = 2048 * (4096 / 64) * 2, // bf16 scales
        .expert_bias_row_bytes = 2048 * (4096 / 64) * 2, // bf16 biases
        .n_experts = 288,
    };
    try std.testing.expectEqual(@as(usize, 288), meta.n_experts);
    try std.testing.expect(meta.gate_biases_name != null);
}
```

- [ ] **Step 2: 确认编译失败**（字段不存在）

```bash
zig build test 2>&1 | grep -m3 error
```
Expected: `no field named 'gate_biases_name'` 类错误。

- [ ] **Step 3: 实现泛化**

具体改动：
1. `LayerExpertMeta` 增加：
```zig
    gate_biases_name: ?[]const u8 = null,
    up_biases_name: ?[]const u8 = null,
    down_biases_name: ?[]const u8 = null,
    expert_bias_row_bytes: usize = 0,
```
2. `pred_experts: [43][6]u32` → 改为运行期分配：`pred_experts: [][]u32`（`init` 时按 `n_layers × topk` 分配；DSV4 传 43×6，行为不变）。`dymoe_top5_count/dymoe_total_count: ?[][256]u32` → `?[][]u32`（per-layer 长度 n_experts）。所有使用处（`finishWarmup` 等）同步改索引。
3. stream 加载路径中组装 `gatherQmm`/expert tensor 处（:790-810 附近），`gate_proj_biases = null` 改为：当 `gate_biases_name != null` 时用 `PartialTensorReader` 读对应 expert 的 biases 行（与 scales 完全同构的读取路径，复用同一函数）。
4. **DSV4 行为保护**：biases 为 null 时生成代码与现状逐字节一致（mxfp4 无 biases）。

- [ ] **Step 4: 全量测试 + DSV4 冒烟回归**

```bash
zig build test 2>&1 | tail -3
# DSV4 server 冒烟（有模型时）:
./zig-out/bin/dmlx serve --model ~/models/DeepSeek-V4-Flash-4bit --port 8931 &
sleep 60 && curl -s localhost:8931/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"2+2="}],"max_tokens":8,"temperature":0}' | head -c 300
kill %1
```
Expected: 单测全绿；DSV4 输出含 `4`（stream 路径回归无破坏）。

- [ ] **Step 5: Commit**

```bash
git add src/models/expert_stream.zig src/models/expert_pread.zig
git commit -m "feat(stream): LayerExpertMeta biases + 层数/topk/expert 数泛化（GLM-5.3 前置）"
```

---

## Task 5: chat template（glm 类型）

**Files:**
- Modify: `src/tokenizer/chat_template.zig`（:14-26 enum、:75-87 apply、新增 applyGlm/initGlm）
- Modify: `src/main.zig:595`（detectChatTemplate）、`src/server/state.zig:150`（template 初始化分支）
- Test: `chat_template.zig` 内 test block

**模板语义**（从本地 `chat_template.jinja` 无 tools、无 reasoning 分支提取）：
前缀 `[gMASK]<sop>`；system 消息 → `<|system|>{content}`；user → `<|user|>{content}`；assistant → `<|assistant|>{content}`；generation prompt → `<|assistant|>`。执行时先完整读一遍 jinja 确认角色标记的精确文本（含换行）。

- [ ] **Step 1: failing test**

```zig
test "applyGlm: system+user+generation prompt" {
    const allocator = std.testing.allocator;
    var tpl = ChatTemplate.initGlm(allocator);
    const msgs = [_]ChatMessage{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "Hi" },
    };
    const out = try tpl.apply(&msgs, true);
    defer allocator.free(out);
    try std.testing.expect(std.mem.startsWith(u8, out, "[gMASK]<sop>"));
    try std.testing.expect(std.mem.indexOf(u8, out, "<|system|>You are helpful.") != null);
    try std.testing.expect(std.mem.endsWith(u8, out, "<|assistant|>"));
}
```

- [ ] **Step 2: 确认失败** → **Step 3: 实现**（enum 加 `glm`、`fromString` 加 `"glm"`、`applyGlm` 照 jinja 语义、`initGlm` 设 bos=`[gMASK]<sop>` eos=``——EOS 用 token id 列表在采样侧处理，模板层只负责文本）

- [ ] **Step 4: 测试 + 接 dispatch**

`main.zig:595` `detectChatTemplate()`：config.json 含 `"glm5_next"` → `.glm`；`state.zig:150`：同样条件调 `initGlm`。

```bash
zig build test 2>&1 | tail -3
```
Expected: 全绿。

- [ ] **Step 5: Commit**

```bash
git add src/tokenizer/chat_template.zig src/main.zig src/server/state.zig
git commit -m "feat(glm53): GLM chat template（[gMASK]<sop> 系）"
```

---

## Task 6: GlmKDA 模块（线性注意力，34 层）

**Files:**
- Create: `src/models/glm53_kda.zig`
- Test: `src/models/glm53_kda.zig` 内 test + Task 7 的 oracle 对齐

**数学定义**（照 mlx-vlm `Glm5NextLinearAttention` + `gated_delta.py`，以 `/tmp/glm5_next_ref` 本地副本为准逐行对齐）：
1. `qkv = silu(depthwise_conv1d(qkv_proj(x), k=4))`（conv 带 window 缓存，prefill 时 padding 语义照参考）
2. split q/k/v → [B,L,64,128]；`q = scale²·rmsnorm(q)`、`k = scale·rmsnorm(k)`（scale=128^-0.5，eps=1e-6/128）
3. `f_a, b_raw, g_a = split(fbg_a_proj(x))`；`a = f_b_proj(f_a)`、`gate = g_b_proj(g_a)`
4. gated delta rule（逐 token；S 为 [64,128,128] 循环状态）：
   - 衰减 `g = exp(A_log)`（per-head 标量，经 lower_bound=-5 clamp 的 log 域处理——以参考实现 `gated_delta.py` 为准）
   - `β`（b）经 sigmoid + dt_bias 处理（同上，以参考为准）
   - delta rule: `S = g*S + β*(k ⊗ (v - kᵀS))` 形式的标准 gated delta net 更新（**必须照参考逐行翻译**，含数值稳定细节）
   - `out_h = qᵀS`（per head）
5. `out = o_norm(out_h) * sigmoid(gate)` → `o_proj`

- [ ] **Step 1: failing test——单步状态更新形状与确定性**

```zig
test "GlmKDA.step: 输出形状与状态推进" {
    // 构造 64 头 head_dim=128 的小配置（heads=2, head_dim=4 缩小版），
    // 随机 x，跑 step() 两次：
    // 1. 输出 shape == [1, hidden]
    // 2. 相同初始状态 + 相同输入 → 相同输出（确定性）
    // 3. 状态 S 被修改（不全等于初始）
}
```

- [ ] **Step 2: 确认失败** → **Step 3: 实现 `GlmKDA`**

```zig
pub const GlmKdaCache = struct {
    conv_state: Array,  // [conv_kernel-1, 3*proj_dim]
    ssm_state: Array,   // [heads, head_dim, head_dim] f32
    pub fn deinit(self: *GlmKdaCache) void { ... }
};

pub const GlmKDA = struct {
    // 权重：qkv_proj(quant), qkv_conv, fbg_a_proj(quant), f_b/g_b_proj,
    // A_log/dt_bias (f32), o_norm, o_proj(quant)
    pub fn forward(self: *GlmKDA, ctx: EagerContext, x: Array, cache: *GlmKdaCache) !Array
    pub fn prefill(self: *GlmKDA, ctx: EagerContext, x: Array, cache: *GlmKdaCache) !Array
    // prefill: L 次 forward 循环（naive，spec §3 已决策）
};
```
conv：用 `mlx` 卷积 C API 或手写 k=4 滑窗乘加（k 很小，手写 elementwise 更简单且无 API 风险）。

- [ ] **Step 4: 单测通过 + oracle 对齐（Task 7 的统一 harness 先行跑 KDA 段）**

```bash
zig build test 2>&1 | tail -3
```
Expected: 单测 PASS；KDA 段 cos ≥ 0.999（见 Task 7 Step 3 命令）。

- [ ] **Step 5: Commit**

```bash
git add src/models/glm53_kda.zig
git commit -m "feat(glm53): KDA 线性注意力模块（gated delta rule，MLX ops）"
```

---

## Task 7: oracle 对齐 harness（Zig 侧）

**Files:**
- Create: `src/tests/glm53_oracle_test.zig`（或挂入现有 golden 框架，照 `src/tests/golden_test.zig` 模式）
- Create: `scripts/compare_glm53_oracle.py`（npy vs Zig dump bin 的 cos 计算）

- [ ] **Step 1: Zig dump 二进制**

写一个 `zig build run-glm53-oracle-dump` 目标（照 `run-batch-oracle-model` 模式）：加载 prep 后模型的指定层权重，对 fixture token `[17,42,314,2718,7,99,1234]` 计算各段输出，写 `artifacts/glm53_oracle/{seg}.bin`（f32 row-major + shape header）。

- [ ] **Step 2: compare 脚本**

```python
# scripts/compare_glm53_oracle.py A.npy B.bin → 打印 shape/cos/max_abs_err
# cos >= 0.999 PASS（bf16 量化差异容忍到 0.998，FP32_KEEP 段要求 0.9999）
```

- [ ] **Step 3: 逐段对齐循环**

```bash
python3 scripts/compare_glm53_oracle.py tests/fixtures/glm53_oracle/layer0_kda_out.npy artifacts/glm53_oracle/layer0_kda_out.bin
```
Expected: cos ≥ 0.999。不达标时的排查顺序：①权重名映射（prep 输出 vs loader 期望）②conv padding/window 语义 ③delta rule 的 gate/decay 逐行对照 ④bf16 vs f32 累加顺序。

- [ ] **Step 4: Commit**

```bash
git add src/tests/glm53_oracle_test.zig scripts/compare_glm53_oracle.py
git commit -m "test(glm53): oracle 对齐 harness"
```

---

## Task 8: GlmAttention + GlmIndexer（NoPE MLA + DSA，11 层）

**Files:**
- Create: `src/models/glm53_attention.zig`
- Test: oracle 对齐（Task 7 harness 扩展段：layer3_mla_out、layer3_indexer_topk）

**要点**（照 `/tmp/glm5_next_ref/language.py` 的 `Glm5NextAttention`/`Glm5NextIndexer`）：
1. `qkv_a_proj` → split q_a/kv_a → RMSNorm → `q_b_proj`；latent KV cache [B,1,N,512]
2. absorbed 投影：`embed_q`（q_nope → k 空间）/ `unembed_out`（latent → v 空间）per-head MultiLinear
3. Indexer：`wk`+LayerNorm、`weights_proj`、kpool=4 softmax 池化压缩（`index_kpool_compress_ape/gate` fp32）、top-(2048/4) 池选择 + 展开 + always-select-tail（尾 3 个）——**无 RoPE**
4. decode：gather topk latent → SDPA（scale = qk_nope_head_dim^-0.5）；prefill：chunked sparse SDPA（可先 naive 逐 query）
5. cache：latent KV（标准接口）+ pool cache（kpool 窗口累积，模型内自管，参照 mlx-vlm PoolingCache 语义）

- [ ] **Step 1: failing test（indexer topk 对齐）**

对 fixture 输入跑 indexer，dump topk indices 与 oracle `layer3_indexer_topk.npy` 比较：**集合一致率 ≥ 99%**（argpartition 顺序无关，比集合）。

- [ ] **Step 2: 确认失败** → **Step 3: 实现**

- [ ] **Step 4: oracle 对齐 layer3_mla_out cos ≥ 0.999**

```bash
python3 scripts/compare_glm53_oracle.py tests/fixtures/glm53_oracle/layer3_mla_out.npy artifacts/glm53_oracle/layer3_mla_out.bin
```

- [ ] **Step 5: Commit**

```bash
git add src/models/glm53_attention.zig
git commit -m "feat(glm53): NoPE MLA + DSA indexer 注意力"
```

---

## Task 9: Glm53Model 组装 + eager forward

**Files:**
- Create: `src/models/glm53.zig`
- Modify: `src/models/glm53_loader.zig`（buildGlm53Model：weights map → 模型 struct）
- Test: oracle 全段 + final logits

**组装**（照 `Glm5NextTextModel`）：
```
h = embed_tokens(ids) → repeat(hc_mult=4)
for layer in 0..45:
    attn = isFullAttention(l) ? GlmAttention : GlmKDA
    mlp  = isMoE(l) ? Glm53MoE(=DSV4MoE 复用) : dense MLP(=DSV4Expert)
    h = attn_hc.apply_branch(h, input_norm, attn)
    h = ffn_hc.apply_branch(h, post_norm, mlp)
    （topk indices 仅在 MLA 层间传递；本地 config 全 "full" 无 shared 层）
h = norm(mean(h, axis=HC))
logits = lm_head(h)
```
mHC 复用 `deepseek_v4.zig:2176-2592` 的函数族（hc_mult/sinkhorn_iters 从 Glm53Config 传）。MoE gate 用 `DSV4Gate` 的 sigmoid+bias 模式 + norm_topk_prob + scaling 2.5。

- [ ] **Step 1: failing test——logits 对齐（eager 子集）**

先用 **3 层子模型**（layer 0 KDA + layer 3 MLA + layer 4 KDA，loader 支持 `--glm53-layers 0,3,4` 调试开关）跑通 forward，各段对 oracle。

- [ ] **Step 2-3: 实现 + 子集对齐**

- [ ] **Step 4: 全 45 层 eager logits 对齐**

```bash
zig build run-glm53-oracle-dump -- --full
python3 scripts/compare_glm53_oracle.py tests/fixtures/glm53_oracle/logits.npy artifacts/glm53_oracle/logits.bin
```
Expected: cos ≥ 0.999（45 层复合后允许到 0.995，低于此值先逐段二分定位发散层）。
注意：全量 eager 需要 backbone 全部进内存——45 层 attention/dense 部分约 20GB 可行；MoE expert 此时可用「仅保留每层 top 预测所需」的调试开关跳过（logits 会偏，只对齐到 MoE 输入段），或接受全量加载慢（125GB 全读，内存不够就先只对齐 hidden_final 之前的非 MoE 段）。**执行时按内存实况选择并在任务笔记记录**。

- [ ] **Step 5: Commit**

```bash
git add src/models/glm53.zig src/models/glm53_loader.zig build.zig
git commit -m "feat(glm53): Glm53Model 组装 + eager forward 全段对齐"
```

---

## Task 10: registry + CLI/server dispatch + expert streaming 接入

**Files:**
- Modify: `src/model_registry.zig`（注册表 + :259-345 streaming 接入 + :326-328 quant 硬编码改 config 驱动）
- Modify: `src/main.zig:546-595`（detectModelType/runChat 分支）
- Modify: `src/server/state.zig:232`（arch dispatch，architectures[0]=`Glm5NextForConditionalGeneration`）
- Modify: `src/models/glm53_loader.zig`（buildLayerExpertMeta：GLM 命名 + biases）

- [ ] **Step 1: 注册 + VTableAdapter**

照 `model_registry.zig:198-230` DeepseekV4VTableAdapter 模式写 `Glm53VTableAdapter`（forward/deinit/ModelConfig{num_layers:45, num_kv_heads:1（latent）, head_dim:512, vocab_size, hidden_size, compress_ratios: 空}），注册 `.{"Glm5NextForConditionalGeneration", glm53Loader}`。

- [ ] **Step 2: streaming 接入**

`smelt.enabled` 时构建 `[]LayerExpertMeta`（45 层，仅 MoE 层填，dense 层跳过），名字用 Task 1 的 prep 布局，biases 字段填上；`quant_mode = "affine"`、gs=64、bits per-tensor（4；8bit tensor 走 eager 不进 stream——检查 8bit 覆盖是否涉及 expert 权重，若是则 stream 路径需支持 per-tensor bits，否则记录为 eager）。

- [ ] **Step 3: CLI/server 冒烟（真模型）**

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/dmlx serve --model ~/models/GLM-5.3-Flash-MLX-mixed-4_8bit-dmlx \
  --smelt --smelt-strategy stream --smelt-experts 0.15 --smelt-cache 10240 \
  --port 8932 --temperature 0
curl -s localhost:8932/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":16,"temperature":0}'
```
Expected: 返回文本含 `Paris`（首个端到端里程碑）。

- [ ] **Step 4: Commit**

```bash
git add src/model_registry.zig src/main.zig src/server/state.zig src/models/glm53_loader.zig
git commit -m "feat(glm53): registry 注册 + CLI/server dispatch + expert streaming"
```

---

## Task 11: 7-prompt e2e + 稳定 serve 验收（Spec M5）

**Files:**
- Modify: `scripts/run_benchmark.sh`（MODEL_PATH/期望子串参数化，或新增 `scripts/e2e_glm53.sh`）

- [ ] **Step 1: 7-prompt correctness（greedy，temperature=0）**

复用 DSV4 的 7 prompt 集，逐个人工/脚本校验期望子串（GLM 期望输出与 DSV4 可能不同，先跑一遍记录实际输出，人工确认正确后固化为期望）。

- [ ] **Step 2: 100-token 流式 + 非流式各一次**

确认无崩溃、无乱码、SSE 正常终止（[DONE]）。

- [ ] **Step 3: DSV4 回归**

`zig build test` 全绿 + DSV4 7-prompt 快速回归（registry/expert_stream 被改过）。

- [ ] **Step 4: Commit + 收尾**

```bash
git add scripts/e2e_glm53.sh
git commit -m "test(glm53): 7-prompt e2e + 稳定 serve 验收"
```
更新 `docs/superpowers/specs/2026-09-15-glm53-mlx-support-design.md` 状态为已实现（或在 ROADMAP/README 模型支持列表加 GLM-5.3，照现有文档惯例）。

---

## 风险与回退

- **Task 6/8 的 oracle 对齐是最大耗时点**：每段卡住先对照 `/tmp/glm5_next_ref` 逐行，再二分 dump 中间 tensor。
- **任何阶段 DSV4 回归破裂**：立即停下修复，DSV4 是主线资产。
- **内存不可行**（125GB + 48GB 机器极端情况）：回退到提高 smelt 比例以外的手段前先记录数据，不与 DSV4 时期的"已证伪方向"重复（见 native-predictive-expert-prefetch-plan.md §20-26）。
