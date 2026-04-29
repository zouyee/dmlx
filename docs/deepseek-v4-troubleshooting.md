# DeepSeek V4 Troubleshooting Guide

## Overview

This document provides troubleshooting guidance for DeepSeek V4 model inference issues in mlx-zig.

## Common Issues

### 1. Garbled Output / Invalid Tokens

**Symptoms:**
- Model generates nonsensical text
- Output contains random characters or symbols
- Tokens appear to be out of vocabulary range

**Root Cause:**
Incorrect chat template formatting causing the tokenizer to split special tokens into sub-tokens.

**Diagnosis:**
Check the log output when running inference:

```bash
mlx-zig chat --model ~/models/deepseek-v4 --prompt "Hello"
```

Look for the prompt token validation:
```
✅ Prompt correctly formatted with BOS token 100000
Prompt tokens (8): [100000, 100003, 1234, 5678, 100006]
```

If you see an error like:
```
❌ BOS token mismatch! Expected 100000, got 60
```

This indicates the chat template is using incorrect special tokens.

**Solution:**
Ensure the chat template uses the correct special token format:

- ✅ Correct: `<|begin_of_sentence|>` (half-width pipe `|`, underscore `_`)
- ❌ Wrong: `<｜begin▁of▁sentence｜>` (full-width pipe `｜`, special space `▁`)

**Fixed in:** Commit `fix: correct DeepSeek V4 chat template special tokens`

---

### 2. Special Token Reference

DeepSeek V4 uses the following special tokens:

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

---

### 3. Token Validation

The implementation includes automatic validation of prompt formatting:

```zig
// Validates that the first token is BOS (100000)
if (prompt_tokens[0] != 100000) {
    std.log.err("❌ BOS token mismatch! Expected 100000, got {d}", .{prompt_tokens[0]});
    return error.InvalidPromptFormat;
}
```

This catches chat template formatting errors early before inference begins.

---

### 4. Memory Issues

**Symptoms:**
- Out of memory errors
- Slow inference (>1s per token)
- System swap usage

**Diagnosis:**
Check MLX memory configuration in logs:
```
MLX memory: wired_limit=40960MB cache_limit=38400MB (system=48000MB)
```

**Solutions:**

1. **Enable Smelt Mode** (partial expert loading):
```bash
mlx-zig chat --model ~/models/deepseek-v4 \
  --smelt --smelt-experts 0.15 \
  --prompt "Hello"
```

2. **Use Quantized KV Cache**:
```bash
mlx-zig serve --model ~/models/deepseek-v4 \
  --kv-strategy paged_quantized \
  --kv-bits 4
```

3. **Reduce Context Length**:
```bash
mlx-zig chat --model ~/models/deepseek-v4 \
  --max-kv-size 4096 \
  --prompt "Hello"
```

---

### 5. Slow Inference

**Expected Performance (M4 Max, 48GB, 4-bit quantized):**
- TTFT (Time to First Token): 200-500ms for 32-token prompt
- ITL (Inter-Token Latency): 250-500ms per token
- Throughput: 2-4 tokens/s

**If slower than expected:**

1. **Check if weights are quantized:**
```bash
# Should see "4-bit" or "quantized" in model path
ls -lh ~/models/deepseek-v4-flash-4bit/
```

2. **Verify GPU usage:**
```bash
# MLX should default to GPU
# Check logs for: "Set default device to GPU"
```

3. **Enable speculative decoding** (future optimization):
```bash
mlx-zig chat --model ~/models/deepseek-v4 \
  --speculative-ngram 4 \
  --prompt "Hello"
```

---

### 6. Model Loading Errors

**Symptoms:**
- "Missing weight" errors
- "Unsupported architecture" errors
- Segmentation faults during loading

**Common Causes:**

1. **Incorrect model format:**
   - Ensure model is in MLX format (not PyTorch or GGUF)
   - Use `mlx_lm.convert` to convert HuggingFace models

2. **Incomplete download:**
   - Verify all shard files are present
   - Check file sizes match expected values

3. **Mismatched config:**
   - Ensure `config.json` matches the model architecture
   - Check `model_type` field is `"deepseek_v4"`

**Diagnosis:**
```bash
# Check model files
ls -lh ~/models/deepseek-v4-flash-4bit/

# Should see:
# - config.json
# - tokenizer.json
# - model.safetensors (or model-00001-of-NNNNN.safetensors)
```

---

## Debugging Tips

### Enable Verbose Logging

Set log level to debug:
```bash
export RUST_LOG=debug  # If using Rust components
mlx-zig chat --model ~/models/deepseek-v4 --prompt "Test"
```

### Inspect Logits

The implementation includes diagnostic logging for logits:
```
Logits: len=129280 max=12.3456 min=-8.9012 mean=0.0234 argmax=1234 nan=0 inf=0
Top tokens: [1234]=12.35 [5678]=11.23 [9012]=10.45
```

Check for:
- NaN or Inf values (indicates numerical instability)
- Extremely large/small values (indicates scaling issues)
- Uniform distribution (indicates model not learning)

### Test with Simple Prompts

Start with minimal prompts to isolate issues:
```bash
# Test 1: Single token
mlx-zig chat --model ~/models/deepseek-v4 --prompt "Hi" --max-tokens 5

# Test 2: English only
mlx-zig chat --model ~/models/deepseek-v4 --prompt "Hello" --max-tokens 10

# Test 3: Chinese (if model supports)
mlx-zig chat --model ~/models/deepseek-v4 --prompt "你好" --max-tokens 10
```

---

## Performance Benchmarking

Run the benchmark tool to measure performance:

```bash
mlx-zig benchmark --model ~/models/deepseek-v4-flash-4bit \
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

---

## Known Limitations

1. **Single-request throughput only**
   - No continuous batching support yet
   - Concurrent requests are serialized

2. **CPU sampling bottleneck**
   - Sampling happens on CPU, not GPU
   - Causes GPU pipeline stalls

3. **No graph caching**
   - Each token triggers full graph compilation
   - Future optimization: cache decode graph

4. **MoE routing overhead**
   - Top-k selection on GPU is inefficient for batch=1
   - Future optimization: fused MoE kernels

---

## Reporting Issues

When reporting issues, please include:

1. **System information:**
   - macOS version
   - Apple Silicon chip (M1/M2/M3/M4)
   - Total RAM

2. **Model information:**
   - Model name and variant
   - Quantization level (4-bit, 8-bit, FP16)
   - Model size on disk

3. **Command used:**
   ```bash
   mlx-zig chat --model <path> --prompt "<prompt>" --max-tokens <n>
   ```

4. **Log output:**
   - Include full log output from command
   - Especially prompt token validation and logits diagnostics

5. **Expected vs actual behavior:**
   - What you expected to happen
   - What actually happened
   - Any error messages

---

## References

- [DeepSeek V4 Paper](https://arxiv.org/abs/2501.12948)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [mlx-lm Repository](https://github.com/ml-explore/mlx-examples/tree/main/llms)
