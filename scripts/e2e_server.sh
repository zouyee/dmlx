#!/bin/bash
# End-to-end correctness tests using dmlx server mode.
#
# This avoids reloading the 141GB model for every test by starting the
# server once and sending requests via HTTP API.
#
# Server KV cache compatibility: the server now creates specialized per-layer
# KV caches for DeepSeek V4 (DeepseekV4Cache / RotatingWithWindow) matching
# the model's heterogeneous attention architecture. Both the engine loop
# (vtable.forward → DSV4Model.forward) and the native generate path
# (nativeGenerateFn) receive the correct cache types.
#
# Usage:
#   ./scripts/e2e_server.sh [model_path]
#
# Requirements:
#   curl, jq (for JSON parsing)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${1:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
CLI="${PROJECT_DIR}/zig-out/bin/dmlx"
PORT=18080
SERVER_URL="http://localhost:${PORT}"

SMELT_FLAGS="--smelt --smelt-strategy stream --smelt-experts 0.1"

export MLX_METAL_FAST_SYNCH=1

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0
CORE_PASSED=0
CORE_FAILED=0
EXT_PASSED=0
EXT_FAILED=0
SERVER_PID=""

# ------------------------------------------------------------------
# Cleanup: kill server on exit
# ------------------------------------------------------------------
cleanup() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo ""
        echo "🛑 Stopping server (PID ${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ------------------------------------------------------------------
# Helper: start server and wait for readiness
# ------------------------------------------------------------------
start_server() {
    echo "🚀 Starting server on port ${PORT}..."
    echo "   Command: ${CLI} serve --model ${MODEL_PATH} --port ${PORT} ${SMELT_FLAGS}"
    echo ""

    # Start server in background, redirect output to log
    ${CLI} serve \
        --model "${MODEL_PATH}" \
        --port "${PORT}" \
        --max-tokens 256 \
        --temperature 0 \
        ${SMELT_FLAGS} > "${PROJECT_DIR}/server.log" 2>&1 &
    SERVER_PID=$!

    echo "   Server PID: ${SERVER_PID}"

    # Wait for server to be ready (max 300s for model loading)
    local retries=0
    local max_retries=300
    while [ ${retries} -lt ${max_retries} ]; do
        if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Server ready${NC}"
            echo ""
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo -e "${RED}❌ Server crashed during startup${NC}"
            echo "   Last log lines:"
            tail -20 "${PROJECT_DIR}/server.log"
            exit 1
        fi
        sleep 1
        retries=$((retries + 1))
        if [ $((retries % 30)) -eq 0 ]; then
            echo "   ⏳ Still loading model... (${retries}s)"
        fi
    done

    echo -e "${RED}❌ Server failed to start within ${max_retries}s${NC}"
    kill "${SERVER_PID}" 2>/dev/null || true
    exit 1
}

# ------------------------------------------------------------------
# Helper: send chat completion request
# ------------------------------------------------------------------
run_case() {
    local name="$1"
    local prompt="$2"
    local max_tokens="$3"
    local expected="$4"

    echo "🧪 ${name}"
    echo "   Prompt: ${prompt}"
    echo "   Expected: '${expected}'"

    local response
    response=$(curl -sf "${SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"default\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${prompt}\"}],
            \"max_tokens\": ${max_tokens},
            \"temperature\": 0
        }" 2>&1) || {
        echo -e "${RED}   ❌ FAILED: request error${NC}"
        echo "   ${response}"
        ((EXT_FAILED++)) || true
        ((FAILED++)) || true
        return
    }

    # Extract generated text from response
    local generated
    generated=$(echo "${response}" | jq -r '.choices[0].message.content // empty' 2>/dev/null || echo "")

    # Clean up whitespace
    generated=$(echo "${generated}" | tr -d '\r\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    if [ -z "${generated}" ]; then
        echo -e "${RED}   ❌ FAILED: empty response${NC}"
        echo "   Raw: ${response}"
        ((EXT_FAILED++)) || true
        ((FAILED++)) || true
        return
    fi

    if echo "${generated}" | grep -qi "${expected}"; then
        echo -e "${GREEN}   ✅ PASSED${NC}"
        echo "   Generated: ${generated}"
        ((EXT_PASSED++)) || true
        ((PASSED++)) || true
    else
        echo -e "${RED}   ❌ FAILED${NC}"
        echo "   Generated: ${generated}"
        ((EXT_FAILED++)) || true
        ((FAILED++)) || true
    fi
}

# ------------------------------------------------------------------
# Helper: run a core prompt test (aligned with best_test.sh)
# ------------------------------------------------------------------
run_core_case() {
    local name="$1"
    local prompt="$2"
    local max_tokens="$3"
    local expected="$4"

    echo "🧪 [CORE] ${name}"
    echo "   Prompt: ${prompt}"
    echo "   Expected: '${expected}'"

    local response
    response=$(curl -sf "${SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"default\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${prompt}\"}],
            \"max_tokens\": ${max_tokens},
            \"temperature\": 0
        }" 2>&1) || {
        echo -e "${RED}   ❌ FAILED: request error${NC}"
        echo "   ${response}"
        ((CORE_FAILED++)) || true
        ((FAILED++)) || true
        return
    }

    local generated
    generated=$(echo "${response}" | jq -r '.choices[0].message.content // empty' 2>/dev/null || echo "")
    generated=$(echo "${generated}" | tr -d '\r\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    if [ -z "${generated}" ]; then
        echo -e "${RED}   ❌ FAILED: empty response${NC}"
        echo "   Raw: ${response}"
        ((CORE_FAILED++)) || true
        ((FAILED++)) || true
        return
    fi

    if echo "${generated}" | grep -qi "${expected}"; then
        echo -e "${GREEN}   ✅ PASSED${NC}"
        echo "   Generated: ${generated}"
        ((CORE_PASSED++)) || true
        ((PASSED++)) || true
    else
        echo -e "${RED}   ❌ FAILED${NC}"
        echo "   Generated: ${generated}"
        ((CORE_FAILED++)) || true
        ((FAILED++)) || true
    fi
}

# ------------------------------------------------------------------
# Pre-flight
# ------------------------------------------------------------------
echo "═══════════════════════════════════════════════════════════════"
echo "🔍 DeepSeek V4 E2E Tests (Server Mode)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Model path: ${MODEL_PATH}"
echo "Server:     ${SERVER_URL}"
echo ""

if [ ! -d "${MODEL_PATH}" ]; then
    echo -e "${RED}❌ Model not found at: ${MODEL_PATH}${NC}"
    exit 1
fi

if [ ! -f "${CLI}" ]; then
    echo "📦 Building dmlx..."
    (cd "${PROJECT_DIR}" && zig build -Doptimize=ReleaseFast) || {
        echo -e "${RED}❌ Build failed${NC}"
        exit 1
    }
fi

if ! command -v curl >/dev/null 2>&1; then
    echo -e "${RED}❌ curl not found${NC}"
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    echo -e "${RED}❌ jq not found (install: brew install jq)${NC}"
    exit 1
fi

# ------------------------------------------------------------------
# Start server
# ------------------------------------------------------------------
start_server

# ------------------------------------------------------------------
# Test cases
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Core prompt tests (aligned with best_test.sh — 7 prompts)
# ------------------------------------------------------------------
echo ""
echo "── Core Prompts (aligned with best_test.sh) ──"
echo ""

run_core_case "P1: 2+2=" \
    "2+2=" 30 "4"

run_core_case "P2: Capital of France (completion)" \
    "The capital of France is" 20 "Paris"

run_core_case "P3: Water freezes at" \
    "What temperature does water freeze at in Celsius? Just give the number." 30 "0"

run_core_case "P4: Earth round?" \
    "Is the Earth round? Reply with only yes or no." 30 "yes"

run_core_case "P5: 3*3=" \
    "3*3=" 30 "9"

run_core_case "P6: 10-5=" \
    "10-5=" 50 "answer is 5"

run_core_case "P7: Capital of France (question)" \
    "What is capital of France?" 30 "Paris"

# ------------------------------------------------------------------
# Extended test cases
# ------------------------------------------------------------------
echo ""
echo "── Extended Tests ──"
echo ""

# --- Geography ---
run_case "Geography: Japan" "What is the capital of Japan? Answer in one word:" 3 "Tokyo"
run_case "Geography: China" "What is the capital of China? Answer in one word:" 3 "Beijing"

# --- Science ---
run_case "Science: Sky color" "What color is the sky on a clear day?" 3 "blue"

# --- Language ---
run_case "Language: Translate" "Translate 'Hello' to Chinese:" 3 "你好"

# --- Code ---
run_case "Code: Python print" 'In Python, what does print("Hello") output?' 5 "Hello"

# --- History ---
run_case "History: WW2" "In what year did World War II end?" 3 "1945"

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Core Prompts:    ${CORE_PASSED}/7 passed, ${CORE_FAILED} failed"
echo "Extended Tests:  ${EXT_PASSED}/$((EXT_PASSED + EXT_FAILED)) passed, ${EXT_FAILED} failed"
echo "Total:           ${PASSED}/$((PASSED + FAILED)) passed, ${FAILED} failed"
echo "═══════════════════════════════════════════════════════════════"

if [ ${CORE_FAILED} -gt 0 ]; then
    echo -e "${RED}❌ Core prompt(s) failed — exit 1${NC}"
    exit 1
elif [ ${EXT_FAILED} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  All core prompts passed, but ${EXT_FAILED} extended test(s) failed${NC}"
    exit 0
else
    echo -e "${GREEN}✅ All ${PASSED} tests passed!${NC}"
    exit 0
fi
