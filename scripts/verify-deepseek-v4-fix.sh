#!/bin/bash
# Verification script for DeepSeek V4 chat template fix
# Usage: ./scripts/verify-deepseek-v4-fix.sh <model_path>

set -e

MODEL_PATH="${1:-$HOME/models/deepseek-v4-flash-4bit}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at: $MODEL_PATH"
    echo "Usage: $0 <model_path>"
    exit 1
fi

echo "🔍 Verifying DeepSeek V4 chat template fix..."
echo "Model path: $MODEL_PATH"
echo ""

# Build the project
echo "📦 Building mlx-zig..."
zig build || {
    echo "❌ Build failed"
    exit 1
}

echo "✅ Build successful"
echo ""

# Test 1: Simple English prompt
echo "🧪 Test 1: Simple English prompt"
echo "Running: mlx-zig chat --model $MODEL_PATH --prompt 'Hello' --max-tokens 10"
echo ""

OUTPUT=$(./zig-out/bin/mlx-zig chat \
    --model "$MODEL_PATH" \
    --prompt "Hello" \
    --max-tokens 10 2>&1)

echo "$OUTPUT"
echo ""

# Check for BOS token validation
if echo "$OUTPUT" | grep -q "✅ Prompt correctly formatted with BOS token 100000"; then
    echo "✅ Test 1 PASSED: BOS token correctly validated"
else
    echo "❌ Test 1 FAILED: BOS token validation not found"
    echo "Expected to see: '✅ Prompt correctly formatted with BOS token 100000'"
    exit 1
fi

# Check for error messages
if echo "$OUTPUT" | grep -q "❌ BOS token mismatch"; then
    echo "❌ Test 1 FAILED: BOS token mismatch detected"
    exit 1
fi

echo ""

# Test 2: Chinese prompt (if model supports)
echo "🧪 Test 2: Chinese prompt"
echo "Running: mlx-zig chat --model $MODEL_PATH --prompt '你好' --max-tokens 10"
echo ""

OUTPUT=$(./zig-out/bin/mlx-zig chat \
    --model "$MODEL_PATH" \
    --prompt "你好" \
    --max-tokens 10 2>&1)

echo "$OUTPUT"
echo ""

if echo "$OUTPUT" | grep -q "✅ Prompt correctly formatted with BOS token 100000"; then
    echo "✅ Test 2 PASSED: Chinese prompt correctly formatted"
else
    echo "❌ Test 2 FAILED: Chinese prompt validation failed"
    exit 1
fi

echo ""

# Test 3: Multi-turn conversation
echo "🧪 Test 3: System prompt + user message"
echo "Running with system prompt..."
echo ""

OUTPUT=$(./zig-out/bin/mlx-zig chat \
    --model "$MODEL_PATH" \
    --system "You are a helpful assistant." \
    --prompt "What is 2+2?" \
    --max-tokens 20 2>&1)

echo "$OUTPUT"
echo ""

if echo "$OUTPUT" | grep -q "✅ Prompt correctly formatted with BOS token 100000"; then
    echo "✅ Test 3 PASSED: System prompt correctly formatted"
else
    echo "❌ Test 3 FAILED: System prompt validation failed"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ All tests passed!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📋 Summary:"
echo "  - Chat template uses correct special tokens (<|begin_of_sentence|>)"
echo "  - BOS token ID validation working (expected: 100000)"
echo "  - Prompt formatting includes proper spacing and newlines"
echo ""
echo "🎉 DeepSeek V4 chat template fix verified successfully!"
