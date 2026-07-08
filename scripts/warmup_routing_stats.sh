#!/bin/bash
# warmup_routing_stats.sh — collect routing stats from diverse prompts (Phase 1)
# This must run BEFORE benchmark to ensure Phase 2 has high-quality routing stats.
# Runs each prompt 3 times to build robust per-layer routing_counts.
#
# Usage: bash scripts/warmup_routing_stats.sh

set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLI="${PROJECT_DIR}/zig-out/bin/dmlx"
MODEL_PATH="${MODEL_PATH:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
PACKED_DIR="${PACKED_DIR:-${MODEL_PATH}/packed_experts}"
STATS_FILE="${PACKED_DIR}/.smelt_routing_stats.bin"
NATIVE_SMELT_N="${NATIVE_SMELT_N:-20}"
PORT=18091
SERVER_URL="http://localhost:${PORT}"

echo "════════════════════════════════════"
echo "  Routing Stats Warmup (Phase 1)"
echo "  SMELT N=${NATIVE_SMELT_N}"
echo "════════════════════════════════════"

# Delete stats to ensure Phase 1 (unbiased routing)
rm -f "${STATS_FILE}"
echo "  Stats file deleted → Phase 1 mode"

# Kill any existing server on this port
pkill -f "dmlx serve.*${PORT}" 2>/dev/null || true
sleep 1

# Start server in Phase 1 (no stats → penalty=0, unbiased routing)
NATIVE_SMELT_N="${NATIVE_SMELT_N}" "${CLI}" serve \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --native \
    --expert-packed-dir "${PACKED_DIR}" \
    > /tmp/warmup_serve.log 2>&1 &
SRV_PID=$!
trap "kill ${SRV_PID} 2>/dev/null; sleep 2; pkill -f 'dmlx serve.*${PORT}' 2>/dev/null || true" EXIT

# Wait for server ready
echo -n "  Waiting for server..."
for i in $(seq 1 120); do
    if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
        echo " ready (${i}s)"
        break
    fi
    echo -n "."
    sleep 1
done
if ! curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo -e "\nERROR: Server failed to start"; exit 1
fi

# Diverse prompts — same as E2E + benchmark prompts
PROMPTS=(
    "2+2="
    "The capital of France is"
    "What temperature does water freeze at in Celsius? Just give the number."
    "Is the Earth round? Reply with only yes or no."
    "3*3="
    "10-5="
    "What is capital of France?"
    "Hi"
    "Hello, how are you?"
    "What is machine learning?"
    "Explain transformers in one sentence."
    "Who wrote Romeo and Juliet?"
    "What is the speed of light?"
    "1+1="
    "Name a primary color."
)

TOTAL=${#PROMPTS[@]}
REPEATS=3
echo "  Sending ${TOTAL} prompts × ${REPEATS} repeats = $((TOTAL * REPEATS)) requests..."

for rep in $(seq 1 ${REPEATS}); do
    for i in "${!PROMPTS[@]}"; do
        prompt="${PROMPTS[$i]}"
        n=$((i + 1))
        # Use max_tokens=15 — enough to generate meaningful routing patterns without being slow
        curl -s --max-time 90 "${SERVER_URL}/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"d\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":15,\"temperature\":0}" \
            > /dev/null 2>&1 || true
    done
    echo "  Repeat ${rep}/${REPEATS} done"
done

echo "  Shutting down gracefully (saves routing stats)..."
if curl -sf --max-time 5 -X POST "${SERVER_URL}/shutdown" > /dev/null 2>&1; then
    sleep 3
fi
pkill -f "dmlx serve.*${PORT}" 2>/dev/null || true
sleep 1

if [[ -f "${STATS_FILE}" ]]; then
    SIZE=$(wc -c < "${STATS_FILE}")
    echo "  ✓ Stats saved: ${STATS_FILE} (${SIZE} bytes)"
else
    echo "  ✗ Stats file not found — check server log: /tmp/warmup_serve.log"
    exit 1
fi

echo "════════════════════════════════════"
echo "  Routing stats warmup DONE"
echo "  Phase 2 will activate on next server start"
echo "════════════════════════════════════"
