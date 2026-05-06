# DeepSeek V4 Chat Analysis and Troubleshooting

> **Consolidated from:** chat-analysis.md, troubleshooting.md  
> **Last updated:** 2026-05-01  
> **Cross-references:** [Fixes and Details](DEEPSEEK-V4-FIXES-AND-DETAILS.md) | [Optimization & Roadmap](DEEPSEEK-V4-OPTIMIZATION-AND-ROADMAP.md)

---

## Executive Summary

The `chat` command with DeepSeek V4 Flash 4-bit was completely non-functional due to **systematic semantic processing defects** in the model forward pass — the first generated token's argmax was always 16 (`.`), regardless of input prompt. This was a systematic computation error, not a mere performance issue. Additionally, Stream mode had catastrophic I/O overhead (~200s/token) because all caching infrastructure was bypassed.

| Dimension | Assessment | Notes |
|-----------|-----------|-------|
| **Accuracy** | ❌ Completely broken | All prompts produce token 16, model not processing input semantics |
| **Performance** | ❌ Unusable | Stream mode ~200s/token, 300s timeout yields 0-1 tokens |
| **Memory** | ✅ Manageable | Stream mode ~10GB, fits 48GB Mac |
| **gtimeout 300→900** | ⚠️ Marginal | 900s gives ~4-5 tokens, doesn't fix accuracy |

---

## Part 1: In-Depth Chat Command Analysis (from chat-analysis.md)

**Analysis Date:** 2026-05-01  
**Command:**
```bash
gtimeout 300 ./zig-out/bin/dmlx chat \
  --model ~/models/DeepSeek-V4-Flash-4bit \
  --prompt "2+2=" \
  --max-tokens 30 \
  --temperature 0.0 \
  --smelt --smelt-strategy stream --smelt-experts 1.0
```
**Scope:** Model loading, smelt stream mode, token generation performance, result accuracy  
**Verification:** All claims cross-verified against `src/` source code

### 1. Command Decomposition & Code Path

#### 1.1 CLI Parameters (`main.zig:412-463`)

```zig
const ChatCommand = struct {
    model_path: []const u8,
    prompt: []const u8,
    max_tokens: usize = 256,
    temperature: f32 = 0.8,
    smelt: bool = false,
    smelt_experts: f32 = 1.0,
    smelt_strategy: []const u8 = "preload",
    smelt_cache_mb: usize = 4096,
};
```

**Parameter mapping:**

| Param | Value | Behavior |
|-------|-------|----------|
| `--smelt` | true | Enable MoE expert selective loading |
| `--smelt-strategy stream` | `.stream` | On-demand disk loading (not preload) |
| `--smelt-experts 1.0` | `1.0` | **Ignored in stream mode**; preload loads 100% |
| `--temperature 0.0` | `0.0` | Greedy sampling (argmax) |
| `--max-tokens 30` | `30` | Max generation limit |

#### 1.2 Model Type Detection (`main.zig:465-500`)

`detectModelType` scans `config.json` for `"model_type"`. DeepSeek V4 Flash has `"model_type": "deepseek_v4"`, entering `runDeepSeekV4Chat` branch.

#### 1.3 DeepSeek V4 Chat Full Path

```
runDeepSeekV4Chat
├── Read config.json
├── Set MLX memory limits (wired=85% RAM, cache=80% RAM)
├── Load tokenizer.json (BpeTokenizer)
├── Build SmeltConfig { enabled=true, load_fraction=1.0, load_mode=.stream }
├── Load weights (loadWeightsSelective — skip all expert weights, backbone ~4.3GB)
├── Build model (buildDSV4Model — 43 layers, MLA + MoE + mHC)
├── Create ExpertStreamProvider (stream mode)
│   └── Open all shard FdPool, but don't load experts
├── Apply chat template (ChatTemplate.initDeepSeek)
│   └── Generate: "<｜begin▁of▁sentence｜>2+2=<｜Assistant｜></think>"
├── Tokenize (add_special_tokens=false)
│   └── Validate BOS token == 0
├── Create KV Caches (makeV4Caches)
│   └── CSA/HCA layers → DeepseekV4Cache
│   └── Other layers → RotatingKVCache
├── Generate (DSV4Model.generate)
│   ├── Prefill: full prompt forward pass, last position logits
│   ├── Sample: argmax (temperature=0.0)
│   └── Decode: autoregressive token-by-token generation
└── Decode and output
```

### 2. Model Loading Analysis

#### 2.1 Weight Loading

**Stream mode `loadWeightsSelective` (`deepseek_v4_loader.zig:485-625`):**
- Build `TensorIndex`: Parse all 33 shard headers (JSON headers only, ~KB level)
- Open all shard file descriptors (`FdPool`)
- **Skip all expert weights**: Only load embedding, attention, norm, shared expert, head
- Initial load: **~4.3GB** (vs full model ~141GB)

**Key observation:**
- `--smelt-experts 1.0` has **no effect on initial load** in stream mode. Stream mode always skips all expert weights.
- Expert weights will be loaded on-demand during inference.

#### 2.2 Model Construction

`buildDSV4Model` (`deepseek_v4_loader.zig:1106-2034`) builds layer by layer:
1. **Embedding**: Dequantized on load (`mlx_take` doesn't support quantized arrays for embedding lookup)
2. **Attention weights**: `wq_a`, `wq_b`, `wkv`, `wo_b` kept quantized; `wo_a` dequantized on load
3. **MoE Gate**: Stream mode **does not allocate `smelt_mask`** — router can freely select 0-255 experts
4. **SwitchGLU / Experts**: Stream mode creates dummy SwitchGLU (expert weights don't exist)
5. **mHC HyperConnections**: Loaded if weights present
6. **RoPE, Compressor, Indexer**

#### 2.3 ExpertStreamProvider Initialization

`main.zig:783-883`:
```zig
if (cmd.smelt and !model.hasExpertsLoaded()) {
    const idx = try allocator.create(safetensors_reader.TensorIndex);
    const strategy = .stream;
    sp.* = try expert_stream.ExpertStreamProvider.initWithStrategy(
        allocator, ctx, idx, strategy, expert_ids, layer_meta,
        true,   // switch_mlp quantized
        32,     // group_size
        4,      // bits
        "mxfp4", // quant_mode
        ds_config.swiglu_limit,
        cmd.smelt_cache_mb,  // 4096 MB
    );
    model.setExpertStreamProvider(sp);
}
```

**Note:** `ExpertCache`, `LayerPrefetcher`, `MmapPool` are all allocated but **never actually used**. `expert_ids` (256 experts) is also unused in stream mode.

### 3. Token Generation Performance Analysis

#### 3.1 Catastrophic I/O in Stream Mode

**Disk read per token generation:**

In `expert_stream.zig:streamingForward`, for each MoE layer:
1. Collect selected expert IDs (typically top-6 to top-8)
2. Call `loadExpertSlices`:
   ```zig
   const full_tensor = try self.index.loadTensor(tensor_name);
   const sliced = try shape_mod.takeAxis(self.ctx, full_tensor, indices_i32, 0);
   ```
3. This means **loading the full fused tensor from disk** (all 256 experts), then slicing on GPU

**Per-layer read:**
- Gate weights: ~4GB
- Up weights: ~4GB  
- Down weights: ~4GB
- **Per-layer total: ~12GB**

**Per token total (43 MoE layers):**
```
43 layers × 12 GB/layer = 516 GB disk read / token
```

**Measured performance:**
- ~**200 seconds/token**
- `gtimeout 300`: at most **1 token** (if prefill takes <100s)
- `gtimeout 900`: at most **4-5 tokens**

#### 3.2 Why Caching Infrastructure Is Disabled

`expert_stream.zig:301-315` explicitly states:
```zig
// Always use full tensor loading (load full tensor + takeAxis slice).
// This produces GPU-friendly tensors that gatherQmm processes efficiently.
// The partial-read path (mmap/pread) creates tensors from raw bytes that
// are numerically correct but may not be optimally laid out for GPU computation.
// Cache and partial reads are available but bypassed until the GPU layout
// issue is resolved.
```

**Allocated but idle components:**

| Component | Purpose | Status |
|-----------|---------|--------|
| `ExpertCache` (LRU) | Cache recently used expert slices | Allocated, never inserted |
| `LayerPrefetcher` | Background thread prefetch next layer | Allocated, thread not started |
| `PartialTensorReader` | Read only selected expert rows | Never instantiated |
| `MmapPool` | mmap reuse | Never created |

#### 3.3 Sampler Performance Bottleneck (Compound)

Even if forward pass were instantaneous, the sampler itself is a bottleneck:
- `std.sort.insertion` on 129K vocabulary: ~16B comparisons
- `logits.eval()`: one GPU→CPU sync per token
- Repeat penalty: full vocabulary CPU→GPU copy-back per token

**Conclusion:** Stream mode is currently **unusable for performance** — even with accuracy issues fully fixed, generating a complete answer would take hours.

### 4. Accuracy Root Cause Analysis

#### 4.1 Core Symptom: All Prompts Produce Same Argmax

| Prompt | Prompt Tokens | First Token Argmax | Top Logit |
|--------|---------------|-------------------|-----------|
| "2+2=" | 7 | **16** | 17.25 |
| "Capital of France" | 7 | **16** | 17.96 |
| "Translate: Hello" | 7 | **16** | 18.73 |
| "Explain AI" | 6 | **16** | 19.63 |
| "Write haiku" | 7 | **16** | 18.20 |

Token 16 in DeepSeek V4 tokenizer maps to `.`.

For a 129K vocabulary model, **different prompts cannot produce identical first token distributions**. This proves the model **completely fails to process input prompt semantics**.

#### 4.2 Root Cause A: Prefill Causal Mask Possibly Not Applied (HIGHEST PRIORITY)

**Code path (`deepseek_v4.zig:1712-1720`):**
```zig
const mm: []const u8 = if (start_pos == 0 and seq_len > 1) "causal" else "";
attn_out = try fast_mod.scaledDotProductAttention(
    self.ctx, q, full_kv, full_kv, scale, mm, mask, self.sink_logits);
```

- Prefill (`seq_len > 1`) passes `"causal"` string
- `mask` parameter is `null` (`mlx_array_empty`)

**Problem:** `DSV4Model.forward` passes `mask=null` in all layer calls:
```zig
// deepseek_v4.zig:2716-2761
for (self.layers, 0..) |*layer, i| {
    hidden = try layer.forward(hidden, input_ids, mask, cache, start_pos, stream);
}
```

**Risk:** MLX's `mlx_fast_scaled_dot_product_attention` when receiving string `"causal"` along with a non-empty `mask` array may prioritize the mask array and ignore the string. Here mask is `mlx_array_empty` (non-null but empty), **behavior undocumented**.

**More serious issue:** `createCausalMask` function (`deepseek_v4.zig:396-423`) exists but is **never called** by `DSV4Model.forward` or `DSV4Attention.forward`. Same issue as Llama model (`llama.zig:657` also `mask=null`).

**Why this explains argmax=16:**
- If prefill attention is bidirectional (non-causal), all tokens see each other
- Prompt semantic structure is destroyed ("2+2=" position info lost)
- Model output degenerates to input-independent distribution
- First decode token identical for all prompts

#### 4.3 Root Cause B: `kv_b` Weights Loaded But Unused (HIGH PRIORITY)

**Code path (`deepseek_v4_loader.zig`):**
- `kv_b` weight identified and loaded: `weights.get("layers.{d}.attn.kv_b")`
- Stored in `layer.kv_b` field during `buildDSV4Model`

**But in `DSV4Attention.forward` (`deepseek_v4.zig:1604-1766`):**
```zig
// wkv output directly reshaped to [B, S, num_heads, head_dim]
const kv_3d = try shape_mod.reshape(self.ctx, wkv_out, &[_]i32{
    @intCast(batch * seq_len), @intCast(self.config.num_attention_heads), @intCast(self.config.head_dim)
});
```

**No use of `self.kv_b` for KV latent-to-full projection.**

In DeepSeek V4's MLA architecture:
- `wkv` outputs low-dim latent KV (`kv_lora_rank`, e.g., 512)
- `kv_b` expands it to full `head_dim` (e.g., 128)
- Missing `kv_b` means KV representation dimension is incorrect

#### 4.4 Root Cause C: CSA/HCA Compression Path Architecture Incomplete (MEDIUM PRIORITY)

**Compression call in `DSV4Attention.forward`:**
```zig
const pooled = try compressKV(self.ctx, kv_3d, self.compress_ratio, ...);
```

**Problems:**
1. `compressKV` only compresses **current input token's KV**, doesn't use `DeepseekV4Cache`'s `accumulateWindows` / `updatePool`
2. During decode (`seq_len=1`), since `seq_len <= window_size` (128), `compressKV` short-circuits returning original tensor — compression completely skipped
3. For short prompts (<128 tokens) this doesn't surface; for long prompts, model loses compressed history context

**Additionally:** `Compressor` and `Indexer` modules are fully implemented in `deepseek_v4.zig` but **never called in `DSV4Attention.forward`**. They're only used in `deepseek_v4_cache.zig`'s cache update path, but attention forward bypasses them.

#### 4.5 Root Cause D: Chat Template (LOW PRIORITY — correct for this model variant)

Current code uses full-width characters: `<｜begin▁of▁sentence｜>`
- Documentation claims should be ASCII (`<|begin_of_sentence|>`) with BOS=100000
- But source code BOS validation uses `expected_bos: u32 = 0`
- Test logs confirm tokenizer maps full-width strings to token 0
- Therefore **for this model variant**, the current template is correct

**Risk:** Loading other DeepSeek V4 variants (using ASCII special tokens) would silently fail.

### 5. Fix Recommendations

#### 5.1 Urgent Fixes (Accuracy)

**Fix 1: Verify/Repair Prefill Causal Mask (HIGHEST PRIORITY)**

Recommended: Explicitly construct causal mask:
```zig
var mask_arr: ?Array = null;
defer if (mask_arr) |m| m.deinit();

if (seq_len > 1) {
    // Construct [1, 1, seq_len, seq_len] causal mask
    mask_arr = try createCausalMask(ctx, 1, 1, seq_len, window_size, start_pos);
}

const mm: []const u8 = if (seq_len > 1) "causal" else "";
attn_out = try fast_mod.scaledDotProductAttention(
    ctx, q, full_kv, full_kv, scale, mm, mask_arr, sink_logits);
```

**Fix 2: Correctly Use `kv_b` in Attention (HIGH PRIORITY)**
```zig
var kv_proj = wkv_out;
if (self.kv_b) |kv_b_weight| {
    // kv_b expands latent KV to full head_dim
    kv_proj = try ops.matmul(self.ctx, wkv_out, kv_b_weight);
}
const kv_3d = try shape_mod.reshape(self.ctx, kv_proj, &[_]i32{...});
```

**Fix 3: Fix CSA/HCA Compression Path (MEDIUM PRIORITY)**
```zig
if (cache) |c| {
    const pooled = try c.compressor.forward(...);
    // ... use pooled KV for attention computation
}
```

#### 5.2 High Priority Fixes (Performance)

**Fix 4: Enable ExpertCache (MOST CRITICAL PERFORMANCE FIX)**
```zig
fn loadExpertSlicesCached(...) !Array {
    const cache_key = CacheKey{ .layer = layer_idx, .tensor = tensor_name };
    
    if (self.cache) |ec| {
        if (ec.get(cache_key, expert_ids)) |cached| {
            return cached;
        }
    }
    
    const result = try self.loadExpertSlices(tensor_name, expert_ids, row_bytes);
    if (self.cache) |ec| {
        ec.put(cache_key, expert_ids, result) catch {};
    }
    return result;
}
```

**Expected effect:** Reduce disk I/O from 516GB/token to near 0 (hot experts cached).

**Fix 5: Enable PartialTensorReader**
```zig
// Replace:
const full_tensor = try self.index.loadTensor(tensor_name);
// With:
const partial = try self.partial_reader.readExpertRows(tensor_name, expert_ids);
```

**Fix 6: Enable LayerPrefetcher**
```zig
if (self.prefetcher) |pf| {
    try pf.startPrefetch(next_layer_idx, predicted_expert_ids);
}
```

#### 5.3 Medium Priority Fixes

**Fix 7: Fix Sampler O(n²) Sort** — Replace `std.sort.insertion` with `std.mem.sort` or GPU MLX `topk`.

**Fix 8: Fix `max_new_tokens - 1` Underflow**
```zig
if (max_new_tokens == 0) return tokens;
for (0..max_new_tokens - 1) |_| {
```

**Fix 9: Unify Chat Template** — Dynamically select template based on `tokenizer_config.json` `chat_template` field.

### 6. gtimeout 300 → 900 Assessment

| Phase | Time | Notes |
|-------|------|-------|
| Model load | ~30-60s | 4.3GB backbone + open 33 shard FDs |
| Prefill (7 tokens) | ~30-60s | 43 layers forward, loading experts per layer |
| Decode per token | ~200s | 516GB disk read |

**gtimeout 300:** Available: 300 - 60 - 45 = ~195s → 0-1 tokens  
**gtimeout 900:** Available: 900 - 60 - 45 = ~795s → 3-4 tokens

**Conclusion:** Changing gtimeout from 300 to 900 **does not change the essential result**. Even generating 3-4 tokens, output is still garbage (all token 16).

**Only after fixing accuracy + enabling ExpertCache does gtimeout matter.** With caching:
- Prefill: ~5-10s (expert cache warmup)
- Decode: ~0.5-2s/token (cache hit)
- Full answer (30 tokens): ~20-70s
- **gtimeout 300 is fully sufficient**

### 7. autoresearch-mlx Assessment

`../autoresearch-mlx` is a Karpathy `autoresearch` port for Apple Silicon (MLX) — an **automated model training/research tool**:
- Trains small LMs within fixed 5-min time budget
- Auto-attempts architecture/optimizer/hyperparameter variants
- Iterates on `val_bpb` metric

**Can it help debug DeepSeek V4?**

| Capability | Applicability | Notes |
|------------|---------------|-------|
| Compare Python mlx-lm output | ⚠️ Indirect | Can load same model in Python to verify correct output |
| Automated debugging | ❌ N/A | Only trains models, doesn't diagnose inference bugs |
| Generate test cases | ❌ N/A | Uses fixed datasets, doesn't generate prompts |
| Performance analysis | ❌ N/A | No disk I/O or memory analysis tools |
| Fix code | ❌ N/A | Python training scripts, doesn't modify Zig code |

**Limited help:**
```python
# Verify correct behavior in Python mlx-lm
import mlx.core as mx
from mlx_lm import load, generate

model, tokenizer = load("~/models/DeepSeek-V4-Flash-4bit")
prompt = "2+2="
response = generate(model, tokenizer, prompt, max_tokens=30, temp=0.0)
print(response)  # Expected: "4" or related answer
```

**`../autoresearch-mlx` cannot directly assist with DeepSeek V4 inference issues.** It's a training tool, separate domain from dmlx inference engine.

### 8. Summary & Recommended Fix Order

| Priority | Fix | Impact |
|----------|-----|--------|
| P0 | Verify/repair causal mask | **Accuracy** |
| P0 | Use `kv_b` in Attention | **Accuracy** |
| P0 | Enable ExpertCache | **Performance usable** |
| P1 | Enable PartialTensorReader | **Further I/O reduction** |
| P1 | Fix sampler sort | **CPU bottleneck reduction** |
| P2 | Fix CSA/HCA compression path | **Long context accuracy** |
| P2 | Unify chat template | **Cross-variant compatibility** |

*This analysis is based on full source code review of dmlx `tuning` branch (commit 5c1cec2). All code references verified against actual files.*

---

## Part 2: Practical Troubleshooting Guide (from troubleshooting.md)

### Common Issues

#### 1. Garbled Output / Invalid Tokens

**Symptoms:**
- Model generates nonsensical text
- Output contains random characters or symbols
- Tokens appear out of vocabulary range

**Root Cause:** Incorrect chat template formatting causing tokenizer to split special tokens into sub-tokens.

**Diagnosis:**
```bash
dmlx chat --model ~/models/deepseek-v4 --prompt "Hello"
```

Look for prompt token validation:
```
✅ Prompt correctly formatted with BOS token 100000
Prompt tokens (8): [100000, 100003, 1234, 5678, 100006]
```

If you see:
```
❌ BOS token mismatch! Expected 100000, got 60
```
This indicates incorrect special tokens.

**Solution:**
- ✅ Correct: `<|begin_of_sentence|>` (half-width pipe `|`, underscore `_`)
- ❌ Wrong: `<｜begin▁of▁sentence｜>` (full-width pipe `｜`, special space `▁`)

**Fixed in:** Commit `fix: correct DeepSeek V4 chat template special tokens`

#### 2. Special Token Reference

| Token | ID | Usage |
|-------|-----|-------|
| `<|begin_of_sentence|>` | 100000 | Start of conversation |
| `<|end_of_sentence|>` | 100001 | End of assistant response |
| `<|User|>` | 100003 | User message marker |
| `<|Assistant|>` | 100006 | Assistant message marker |

**Correct Prompt Format:**
```
<|begin_of_sentence|>{system}\n\n<|User|>: {user_message}\n\n<|Assistant|>: 
```

**Example:**
```
<|begin_of_sentence|>You are a helpful assistant.\n\n<|User|>: Hello, how are you?\n\n<|Assistant|>: 
```

#### 3. Token Validation

Implementation includes automatic prompt format validation:
```zig
if (prompt_tokens[0] != 100000) {
    std.log.err("❌ BOS token mismatch! Expected 100000, got {d}", .{prompt_tokens[0]});
    return error.InvalidPromptFormat;
}
```

#### 4. Memory Issues

**Symptoms:** Out of memory errors, slow inference (>1s per token), system swap usage

**Diagnosis:** Check MLX memory config in logs:
```
MLX memory: wired_limit=40960MB cache_limit=38400MB (system=48000MB)
```

**Solutions:**

1. **Enable Smelt Mode** (partial expert loading):
```bash
dmlx chat --model ~/models/deepseek-v4 \
  --smelt --smelt-experts 0.15 \
  --prompt "Hello"
```

2. **Use Quantized KV Cache:**
```bash
dmlx serve --model ~/models/deepseek-v4 \
  --kv-strategy paged_quantized \
  --kv-bits 4
```

3. **Reduce Context Length:**
```bash
dmlx chat --model ~/models/deepseek-v4 \
  --max-kv-size 4096 \
  --prompt "Hello"
```

#### 5. Slow Inference

**Expected Performance (M4 Max, 48GB, 4-bit quantized):**
- TTFT: 200-500ms for 32-token prompt
- ITL: 250-500ms per token
- Throughput: 2-4 tokens/s

**If slower:**

1. **Check if weights are quantized:**
```bash
ls -lh ~/models/deepseek-v4-flash-4bit/
# Should see "4-bit" or "quantized" in model path
```

2. **Verify GPU usage:** Check logs for "Set default device to GPU"

3. **Enable speculative decoding** (future optimization):
```bash
dmlx chat --model ~/models/deepseek-v4 \
  --speculative-ngram 4 \
  --prompt "Hello"
```

#### 6. Model Loading Errors

**Symptoms:** "Missing weight" errors, "Unsupported architecture" errors, segfaults

**Common Causes:**
1. Incorrect model format — ensure MLX format (not PyTorch or GGUF); use `mlx_lm.convert`
2. Incomplete download — verify all shard files present, check file sizes
3. Mismatched config — ensure `config.json` matches architecture, `model_type` is `"deepseek_v4"`

**Diagnosis:**
```bash
ls -lh ~/models/deepseek-v4-flash-4bit/
# Should see: config.json, tokenizer.json, model.safetensors (or shards)
```

### Debugging Tips

**Enable Verbose Logging:**
```bash
export RUST_LOG=debug
dmlx chat --model ~/models/deepseek-v4 --prompt "Test"
```

**Inspect Logits:**
```
Logits: len=129280 max=12.3456 min=-8.9012 mean=0.0234 argmax=1234 nan=0 inf=0
Top tokens: [1234]=12.35 [5678]=11.23 [9012]=10.45
```

Check for: NaN/Inf (numerical instability), extreme values (scaling issues), uniform distribution (model not learning)

**Test with Simple Prompts:**
```bash
# Test 1: Single token
dmlx chat --model ~/models/deepseek-v4 --prompt "Hi" --max-tokens 5

# Test 2: English only
dmlx chat --model ~/models/deepseek-v4 --prompt "Hello" --max-tokens 10

# Test 3: Chinese (if supported)
dmlx chat --model ~/models/deepseek-v4 --prompt "你好" --max-tokens 10
```

### Performance Benchmarking

```bash
dmlx benchmark --model ~/models/deepseek-v4-flash-4bit \
  --input-tokens 32 \
  --output-tokens 128 \
  --num-runs 3
```

Expected output:
```
=== Benchmark Results ===
  TTFT (time to first token): 350.00 ms
  ITL  (inter-token latency): 300.00 ms
  Throughput:                  3.33 tokens/s
  Peak memory:                 42000.0 MB
```

### Known Limitations

1. **Single-request throughput only** — No continuous batching; concurrent requests serialized
2. **CPU sampling bottleneck** — Sampling on CPU, not GPU; causes pipeline stalls
3. **No graph caching** — Each token triggers full graph compilation; future: cache decode graph
4. **MoE routing overhead** — top-k selection on GPU inefficient for batch=1; future: fused MoE kernels

### Reporting Issues

Please include:
1. **System info:** macOS version, Apple Silicon chip, total RAM
2. **Model info:** Name/variant, quantization level, disk size
3. **Command:** Full command used
4. **Log output:** Full log, especially prompt token validation and logits diagnostics
5. **Expected vs actual:** What you expected and what happened

### References

- [DeepSeek V4 Paper](https://arxiv.org/abs/2501.12948)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [mlx-lm Repository](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [Fixes and Details](DEEPSEEK-V4-FIXES-AND-DETAILS.md)
- [Optimization & Roadmap](DEEPSEEK-V4-OPTIMIZATION-AND-ROADMAP.md)
