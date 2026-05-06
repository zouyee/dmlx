# Quick Fix Reference: DeepSeek V4 Chat Template

## TL;DR

**Problem:** DeepSeek V4 generates garbled output  
**Cause:** Wrong special token format in chat template  
**Fix:** Commit `a18bc24` - corrects special tokens from full-width to ASCII  
**Verification:** Run `./scripts/verify-deepseek-v4-fix.sh <model_path>`

---

## What Changed

### Before (❌ Wrong)
```zig
.bos_token = "<｜begin▁of▁sentence｜>",  // Full-width ｜ and special ▁
.eos_token = "<｜end▁of▁sentence｜>",
```

Prompt format:
```
<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜>
```

### After (✅ Correct)
```zig
.bos_token = "<|begin_of_sentence|>",  // ASCII | and _
.eos_token = "<|end_of_sentence|>",
```

Prompt format:
```
<|begin_of_sentence|><|User|>: Hello\n\n<|Assistant|>: 
```

---

## How to Verify the Fix

### Method 1: Automated Script
```bash
./scripts/verify-deepseek-v4-fix.sh ~/models/deepseek-v4-flash-4bit
```

Expected output:
```
✅ All tests passed!
✅ Chat template uses correct special tokens (<|begin_of_sentence|>)
✅ BOS token ID validation working (expected: 100000)
```

### Method 2: Manual Test
```bash
dmlx chat --model ~/models/deepseek-v4 --prompt "Hello" --max-tokens 10
```

Look for this in the logs:
```
✅ Prompt correctly formatted with BOS token 100000
Prompt tokens (5): [100000, 100003, 1234, 5678, 100006]
```

If you see this instead, the fix didn't apply:
```
❌ BOS token mismatch! Expected 100000, got 60
```

---

## Special Token Reference

| Token | ID | ASCII Format | Wrong Format |
|-------|-----|--------------|--------------|
| BOS | 100000 | `<\|begin_of_sentence\|>` | `<｜begin▁of▁sentence｜>` |
| EOS | 100001 | `<\|end_of_sentence\|>` | `<｜end▁of▁sentence｜>` |
| User | 100003 | `<\|User\|>` | `<｜User｜>` |
| Assistant | 100006 | `<\|Assistant\|>` | `<｜Assistant｜>` |

**Key Differences:**
- ✅ ASCII pipe `|` (U+007C) vs ❌ Full-width `｜` (U+FF5C)
- ✅ Underscore `_` (U+005F) vs ❌ Special space `▁` (U+2581)

---

## Files Changed

| File | Change |
|------|--------|
| `src/tokenizer/chat_template.zig` | Fixed special tokens + prompt format |
| `src/main.zig` | Added BOS token validation |
| `docs/en/deepseek-v4/troubleshooting.md` | Comprehensive troubleshooting guide |
| `src/tests/chat_template_tests.zig` | Unit tests for chat templates |
| `scripts/verify-deepseek-v4-fix.sh` | Automated verification script |
| `CHANGELOG.md` | Documented the fix |

---

## Troubleshooting

### Still seeing garbled output?

1. **Rebuild the project:**
   ```bash
   zig build
   ```

2. **Check git status:**
   ```bash
   git log --oneline -1
   # Should show: a18bc24 fix: correct DeepSeek V4 chat template special tokens
   ```

3. **Verify tokenizer.json:**
   ```bash
   # Check that your model's tokenizer.json contains the correct special tokens
   grep "begin_of_sentence" ~/models/deepseek-v4/tokenizer.json
   # Should show: "<|begin_of_sentence|>" (not <｜begin▁of▁sentence｜>)
   ```

4. **Check model format:**
   ```bash
   # Ensure model is in MLX format (not PyTorch/GGUF)
   ls ~/models/deepseek-v4/
   # Should see: config.json, tokenizer.json, model.safetensors (or shards)
   ```

### Error: "BOS token mismatch"

This means the tokenizer is not recognizing the special tokens. Possible causes:

1. **Wrong tokenizer.json:** Model was converted incorrectly
   - Solution: Re-download or re-convert the model

2. **Tokenizer cache:** Old tokenizer data cached
   - Solution: Clear cache and restart

3. **Model mismatch:** Using a different model variant
   - Solution: Verify model is DeepSeek V4 (not V2/V3)

---

## Performance Notes

After this fix, you should see:
- **TTFT:** 200-500ms (32-token prompt)
- **ITL:** 250-500ms per token
- **Throughput:** 2-4 tokens/s (M4 Max, 48GB, 4-bit)

If performance is significantly worse, see the full troubleshooting guide:
`docs/en/deepseek-v4/troubleshooting.md`

---

## Related Documentation

- Full troubleshooting guide: `docs/en/deepseek-v4/troubleshooting.md`
- Verification script: `scripts/verify-deepseek-v4-fix.sh`
- Unit tests: `src/tests/chat_template_tests.zig`
- Changelog: `CHANGELOG.md` (Unreleased section)

---

## Questions?

If you're still experiencing issues after applying this fix:

1. Run the verification script and save the output
2. Check the troubleshooting guide for your specific error
3. Open an issue with:
   - Output from verification script
   - Model path and variant
   - System information (macOS version, chip, RAM)
   - Full log output from `dmlx chat` command

---

**Commit:** `a18bc24`  
**Date:** 2026-04-29  
**Author:** dmlx team
