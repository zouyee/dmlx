#!/bin/bash
# End-to-end correctness tests for DeepSeek V4 stream mode.
#
# These are "smoke tests" that verify the model produces semantically
# correct outputs for well-known prompts. They use greedy sampling
# (--temperature 0) to eliminate randomness.
#
# Usage:
#   ./scripts/e2e_correctness.sh [model_path]
#
# Environment:
#   MLX_METAL_FAST_SYNCH=1 is set automatically for stability.
#   SMELT_FLAGS="--smelt --smelt-strategy stream --smelt-experts 0.1"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${1:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
CLI="${PROJECT_DIR}/zig-out/bin/dmlx"

# Smelt configuration (required for 48GB Mac)
SMELT_FLAGS=(--smelt --smelt-strategy stream --smelt-experts 0.1)

# Ensure Metal fast synch for stability
export MLX_METAL_FAST_SYNCH=1

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

# ------------------------------------------------------------------
# Helper: run a single correctness case
# Args: $1=test_name $2=prompt $3=max_tokens $4=expected_substring
# ------------------------------------------------------------------
run_case() {
    local name="$1"
    local prompt="$2"
    local max_tokens="$3"
    local expected="$4"

    echo ""
    echo "🧪 ${name}"
    echo "   Prompt: ${prompt}"
    echo "   Expected output to contain: '${expected}'"

    local output
    output=$("${CLI}" chat \
        --model "${MODEL_PATH}" \
        --prompt "${prompt}" \
        --max-tokens "${max_tokens}" \
        --temperature 0 \
        "${SMELT_FLAGS[@]}" 2>&1) || {
        echo -e "${RED}   ❌ FAILED: command crashed${NC}"
        echo "   Output: ${output}"
        ((FAILED++)) || true || true
        return
    }

    # Extract generated text (last non-empty line)
    local generated
    generated=$(echo "${output}" | grep -v '^info:' | grep -v '^Starting generation' | tail -1 | tr -d '\r')

    if echo "${generated}" | grep -qi "${expected}"; then
        echo -e "${GREEN}   ✅ PASSED${NC}"
        echo "   Generated: ${generated}"
        ((PASSED++)) || true
    else
        echo -e "${RED}   ❌ FAILED${NC}"
        echo "   Generated: ${generated}"
        echo "   Full output:"
        echo "${output}"
        ((FAILED++)) || true
    fi
}

# ------------------------------------------------------------------
# Pre-flight checks
# ------------------------------------------------------------------
echo "═══════════════════════════════════════════════════════════════"
echo "🔍 DeepSeek V4 E2E Correctness Tests"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Model path: ${MODEL_PATH}"
echo "CLI:        ${CLI}"
echo ""

if [ ! -d "${MODEL_PATH}" ]; then
    echo -e "${RED}❌ Model not found at: ${MODEL_PATH}${NC}"
    echo "Usage: $0 [model_path]"
    exit 1
fi

if [ ! -f "${CLI}" ]; then
    echo "📦 Building dmlx..."
    (cd "${PROJECT_DIR}" && zig build -Doptimize=ReleaseFast) || {
        echo -e "${RED}❌ Build failed${NC}"
        exit 1
    }
    echo -e "${GREEN}✅ Build successful${NC}"
else
    echo -e "${YELLOW}⚠️  Using existing binary. Run 'zig build' first for latest code.${NC}"
fi

# ------------------------------------------------------------------
# Smoke Test Cases
# ------------------------------------------------------------------
# These are lightweight sanity checks (3-5 tokens each) to verify
# the model produces semantically correct outputs. They do NOT
# replace full lm-eval benchmarks (GSM8k, Hellaswag, MMLU, etc.).

# --- Math & Logic ---
run_case "Math: 2+2" "2+2=" 5 "4"
run_case "Math: 3×3" "3*3=" 5 "9"
run_case "Math: 10-5" "10-5=" 5 "5"
run_case "Logic: Yes/No" "Is the Earth round? Answer yes or no:" 3 "yes"

# --- Geography & Facts ---
run_case "Geography: France" "The capital of France is" 5 "Paris"
run_case "Geography: Japan" "The capital of Japan is" 5 "Tokyo"
run_case "Geography: China" "The capital of China is" 5 "Beijing"
run_case "Geography: USA" "The capital of the United States is" 5 "Washington"

# --- Science & Common Sense ---
run_case "Science: Water" "Water freezes at" 5 "0"
run_case "Science: Sky color" "The color of the sky on a clear day is" 5 "blue"
run_case "Science: Sun" "The closest star to Earth is the" 5 "Sun"

# --- Language & Code ---
run_case "Code: Python print" 'In Python, print("Hello") outputs' 5 "Hello"
run_case "Language: Hello CN" "Translate 'Hello' to Chinese:" 5 "你好"
run_case "Language: Cat" "A domestic feline is commonly called a" 5 "cat"

# --- History & Culture ---
run_case "History: 2024" "The year 2024 is in the" 5 "21st"
run_case "History: WW2" "World War II ended in" 5 "1945"

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "═══════════════════════════════════════════════════════════════"
if [ ${FAILED} -eq 0 ]; then
    echo -e "${GREEN}✅ All ${PASSED} tests passed!${NC}"
    echo "═══════════════════════════════════════════════════════════════"
    exit 0
else
    echo -e "${RED}❌ ${FAILED} of $((PASSED + FAILED)) tests failed${NC}"
    echo "═══════════════════════════════════════════════════════════════"
    exit 1
fi
