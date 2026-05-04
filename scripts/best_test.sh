#!/bin/bash
# ============================================================
# mlx-zig Best Test Plan — 7-Prompt Correctness Verification
# ============================================================
# Mode: chat (thinking OFF, </think> in prompt)
# Strategy: smelt + stream
# Updated: 2026-05-03
#
# Usage:
#   MODEL_PATH=~/models/DeepSeek-V4-Flash-4bit bash scripts/best_test.sh
# ============================================================

set -euo pipefail

MODEL_PATH="${1:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
CLI="${2:-${PWD}/zig-out/bin/mlx-zig}"
SMELT_FLAGS=(--smelt --smelt-strategy stream --smelt-experts 0.1)

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

PASSED=0
FAILED=0

echo "═══════════════════════════════════════════════════════"
echo "🔍 mlx-zig 7-Prompt Test Plan (smelt + stream, chat mode)"
echo "═══════════════════════════════════════════════════════"
echo "Model:  $MODEL_PATH"
echo "Mode:   chat (thinking OFF — \`</think>\` in prompt)"
echo ""

# ------------------------------------------------------------------
# Test Case: run with optimal max_tokens per prompt type
# In chat mode with </think>, the model often generates boilerplate
# before the actual answer. We use generous max_tokens to allow
# the answer to appear within the generated text.
# ------------------------------------------------------------------
run_test() {
    local name="$1" prompt="$2" max_tokens="$3" expected="$4"

    echo ""
    echo "🧪 ${name}"
    echo "   Prompt: '${prompt}'"
    echo "   Max tokens: ${max_tokens}"
    echo "   Expected: contains '${expected}'"

    local output
    output=$("${CLI}" chat \
        --model "${MODEL_PATH}" \
        --prompt "${prompt}" \
        --max-tokens "${max_tokens}" \
        --temperature 0 \
        "${SMELT_FLAGS[@]}" 2>&1) || {
        echo -e "${RED}   ❌ CRASHED${NC}"
        FAILED=$((FAILED + 1))
        return
    }

    local generated
    generated=$(echo "${output}" | grep -v '^info:' | grep -v '^debug:' | grep -v '^Starting generation' | grep -v '^Layer ' | tail -1 | tr -d '\r')

    if echo "${generated}" | grep -qi "${expected}"; then
        echo -e "${GREEN}   ✅ PASSED${NC}"
        echo "   Generated: ${generated:0:120}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}   ❌ FAILED${NC}"
        echo "   Generated: ${generated:0:120}"
        FAILED=$((FAILED + 1))
    fi
}

# ==================================================================
# TEST CASES (chat mode, thinking OFF)
# All 7 prompts must PASS — no skips allowed.
# Chat mode generates boilerplate before answers, so we use generous
# max_tokens to ensure the answer appears in the output.
# ==================================================================

# --- Prompt 1: Math ---
run_test "P1: 2+2=" \
    "2+2=" 30 "4"

# --- Prompt 2: Geography completion ---
run_test "P2: Capital of France (completion)" \
    "The capital of France is" 20 "Paris"

# --- Prompt 3: Science completion ---
run_test "P3: Water freezes at" \
    "What temperature does water freeze at in Celsius? Just give the number." 30 "0"

# --- Prompt 4: Yes/No ---
run_test "P4: Earth round?" \
    "Is the Earth round? Reply with only yes or no." 30 "yes"

# --- Prompt 5: Math ---
run_test "P5: 3*3=" \
    "3*3=" 30 "9"

# --- Prompt 6: Math ---
run_test "P6: 10-5=" \
    "10-5=" 50 "answer is 5"

# --- Prompt 7: Geography question ---
run_test "P7: Capital of France (question)" \
    "What is capital of France?" 30 "Paris"

# ==================================================================
# SUMMARY
# ==================================================================
echo ""
echo "═══════════════════════════════════════════════════════"
echo "Results: ${PASSED} passed, ${FAILED} failed"
echo "═══════════════════════════════════════════════════════"

if [ ${FAILED} -eq 0 ]; then
    echo -e "${GREEN}✅ All 7 tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ ${FAILED} test(s) failed${NC}"
    exit 1
fi
