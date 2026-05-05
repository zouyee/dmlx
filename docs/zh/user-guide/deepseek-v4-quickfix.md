# 快速修复参考：DeepSeek V4 聊天模板

## TL;DR

**问题：** DeepSeek V4 生成乱码输出
**原因：** 聊天模板中的特殊 token 格式不正确
**修复：** 提交 `a18bc24` - 将特殊 token 从全角字符修正为 ASCII 格式
**验证：** 运行 `./scripts/verify-deepseek-v4-fix.sh <model_path>`

---

## 变更内容

### 修复前 (❌ 错误)
```zig
.bos_token = "<｜begin▁of▁sentence｜>",  // 全角 ｜ 和特殊 ▁
.eos_token = "<｜end▁of▁sentence｜>",
```

提示词格式：
```
<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜>
```

### 修复后 (✅ 正确)
```zig
.bos_token = "<|begin_of_sentence|>",  // ASCII | 和 _
.eos_token = "<|end_of_sentence|>",
```

提示词格式：
```
<|begin_of_sentence|><|User|>: Hello\n\n<|Assistant|>: 
```

---

## 如何验证修复

### 方法 1：自动化脚本
```bash
./scripts/verify-deepseek-v4-fix.sh ~/models/deepseek-v4-flash-4bit
```

预期输出：
```
✅ All tests passed!
✅ Chat template uses correct special tokens (<|begin_of_sentence|>)
✅ BOS token ID validation working (expected: 100000)
```

### 方法 2：手动测试
```bash
mlx-zig chat --model ~/models/deepseek-v4 --prompt "Hello" --max-tokens 10
```

在日志中查找以下内容：
```
✅ Prompt correctly formatted with BOS token 100000
Prompt tokens (5): [100000, 100003, 1234, 5678, 100006]
```

如果看到以下内容，说明修复未生效：
```
❌ BOS token mismatch! Expected 100000, got 60
```

---

## 特殊 Token 参考

| Token | ID | ASCII 格式 | 错误格式 |
|-------|-----|--------------|--------------|
| BOS | 100000 | `<\|begin_of_sentence\|>` | `<｜begin▁of▁sentence｜>` |
| EOS | 100001 | `<\|end_of_sentence\|>` | `<｜end▁of▁sentence｜>` |
| User | 100003 | `<\|User\|>` | `<｜User｜>` |
| Assistant | 100006 | `<\|Assistant\|>` | `<｜Assistant｜>` |

**关键区别：**
- ✅ ASCII 竖线 `|` (U+007C) vs ❌ 全角 `｜` (U+FF5C)
- ✅ 下划线 `_` (U+005F) vs ❌ 特殊空格 `▁` (U+2581)

---

## 变更文件

| 文件 | 变更内容 |
|------|--------|
| `src/tokenizer/chat_template.zig` | 修正特殊 token + 提示词格式 |
| `src/main.zig` | 添加 BOS token 验证 |
| `docs/zh/deepseek-v4/troubleshooting.md` | 全面的故障排除指南 |
| `src/tests/chat_template_tests.zig` | 聊天模板单元测试 |
| `scripts/verify-deepseek-v4-fix.sh` | 自动化验证脚本 |
| `CHANGELOG.md` | 记录修复内容 |

---

## 故障排除

### 仍然看到乱码输出？

1. **重新构建项目：**
   ```bash
   zig build
   ```

2. **检查 git 状态：**
   ```bash
   git log --oneline -1
   # 应显示: a18bc24 fix: correct DeepSeek V4 chat template special tokens
   ```

3. **验证 tokenizer.json：**
   ```bash
   # 检查模型的 tokenizer.json 是否包含正确的特殊 token
   grep "begin_of_sentence" ~/models/deepseek-v4/tokenizer.json
   # 应显示: "<|begin_of_sentence|>" (而非 <｜begin▁of▁sentence｜>)
   ```

4. **检查模型格式：**
   ```bash
   # 确保模型为 MLX 格式（非 PyTorch/GGUF）
   ls ~/models/deepseek-v4/
   # 应看到: config.json, tokenizer.json, model.safetensors (或分片文件)
   ```

### 错误："BOS token mismatch"

这表示 tokenizer 未能识别特殊 token。可能的原因：

1. **错误的 tokenizer.json：** 模型转换不正确
   - 解决方案：重新下载或重新转换模型

2. **Tokenizer 缓存：** 旧的 tokenizer 数据被缓存
   - 解决方案：清除缓存并重启

3. **模型不匹配：** 使用了不同的模型变体
   - 解决方案：确认模型为 DeepSeek V4（而非 V2/V3）

---

## 性能说明

修复后，您应该看到：
- **TTFT：** 200-500ms（32 token 提示词）
- **ITL：** 每 token 250-500ms
- **吞吐量：** 2-4 tokens/s（M4 Max, 48GB, 4-bit）

如果性能明显更差，请参阅完整的故障排除指南：
`docs/zh/deepseek-v4/troubleshooting.md`

---

## 相关文档

- 完整故障排除指南：`docs/zh/deepseek-v4/troubleshooting.md`
- 验证脚本：`scripts/verify-deepseek-v4-fix.sh`
- 单元测试：`src/tests/chat_template_tests.zig`
- 变更日志：`CHANGELOG.md`（未发布部分）

---

## 有问题？

如果应用此修复后仍然遇到问题：

1. 运行验证脚本并保存输出
2. 在故障排除指南中查找您的具体错误
3. 提交 issue，附上：
   - 验证脚本的输出
   - 模型路径和变体
   - 系统信息（macOS 版本、芯片、RAM）
   - `mlx-zig chat` 命令的完整日志输出

---

**提交：** `a18bc24`
**日期：** 2026-04-29
**作者：** mlx-zig 团队
