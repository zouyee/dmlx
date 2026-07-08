#!/bin/bash
# ============================================================
# DSpark Speculative Decoding Smoke Test
# ============================================================
# Prerequisites:
#   1. Build: zig build -Doptimize=ReleaseFast
#   2. Extract DSpark weights:
#      python3 scripts/extract_markov_weights.py \
#        --model deepseek-ai/DeepSeek-V4-Flash-DSpark \
#        --output ~/models/DeepSeek-V4-Flash-4bit/dspark/
#
# This script runs the same Paris/2+2 checks as dsv4_smoke.sh
# but with DSpark speculative decoding enabled.
#
# Expected: identical output quality (DSpark is lossless with greedy sampling)
# Expected: fewer forward passes per token (acceptance rate > 1.0)
# ============================================================
set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${DIR}/.." && pwd)"
CLI="${PROJECT_DIR}/zig-out/bin/dmlx"
MODEL="${HOME}/models/DeepSeek-V4-Flash-4bit"
PACKED="${MODEL}/packed_experts"
DSPARK_DIR="${MODEL}/dspark"
PORT=8931

# Verify DSpark weights exist
if [ ! -f "${DSPARK_DIR}/markov_w1.bin" ] || [ ! -f "${DSPARK_DIR}/markov_w2.bin" ]; then
    echo "ERROR: DSpark weights not found at ${DSPARK_DIR}/"
    echo "Run: python3 scripts/extract_markov_weights.py --model deepseek-ai/DeepSeek-V4-Flash-DSpark --output ${DSPARK_DIR}/"
    exit 1
fi

echo "=== DSpark Smoke Test ==="
echo "Model: ${MODEL}"
echo "DSpark: ${DSPARK_DIR}"
echo ""

# Kill any existing dmlx serve (project constraint: only one at a time)
pkill -f "dmlx serve" 2>/dev/null; sleep 1

# Start server with DSpark
echo "Starting server with DSpark on port ${PORT}..."
"${CLI}" serve \
    --model "${MODEL}" \
    --expert-packed-dir "${PACKED}" \
    --native \
    --dspark "${DSPARK_DIR}" \
    --port ${PORT} \
    --max-tokens 30 \
    --temperature 0 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "Server ready (${i}s)"
        break
    fi
    sleep 1
done

if ! curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "FAIL: Server did not start"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Test 1: Paris
echo ""
echo "--- Test 1: What is the capital of France? ---"
RESPONSE1=$(curl -s "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        "temperature": 0,
        "max_tokens": 30
    }' | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('message',{}).get('content',''))" 2>/dev/null)

echo "Response: ${RESPONSE1}"
if echo "${RESPONSE1}" | grep -qi "paris"; then
    echo "✓ Paris found"
    PARIS_PASS=1
else
    echo "✗ Paris NOT found"
    PARIS_PASS=0
fi

# Test 2: 2+2
echo ""
echo "--- Test 2: What is 2+2? ---"
RESPONSE2=$(curl -s "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "temperature": 0,
        "max_tokens": 30
    }' | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('message',{}).get('content',''))" 2>/dev/null)

echo "Response: ${RESPONSE2}"
if echo "${RESPONSE2}" | grep -q "4"; then
    echo "✓ 2+2=4 found"
    MATH_PASS=1
else
    echo "✗ 4 NOT found"
    MATH_PASS=0
fi

# Cleanup
echo ""
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

# Results
echo "==========================="
if [ $PARIS_PASS -eq 1 ] && [ $MATH_PASS -eq 1 ]; then
    echo "DSPARK SMOKE PASS ✓"
    exit 0
else
    echo "DSPARK SMOKE FAIL ✗"
    exit 1
fi
