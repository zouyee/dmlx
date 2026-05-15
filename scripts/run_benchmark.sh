#!/bin/bash
# ============================================================
# dmlx Performance Benchmark — Serve Mode
# ============================================================
# Generates: docs/en/analysis/performance-benchmark.md
#
# Usage:
#   bash scripts/run_benchmark.sh [model_path] [smelt_experts] [cache_mb]
#
# Examples:
#   bash scripts/run_benchmark.sh                              # defaults
#   bash scripts/run_benchmark.sh ~/models/DeepSeek-V4-Flash-4bit 0.1 10240
# ============================================================

set -uo pipefail

MODEL_PATH="${1:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
SMELT_EXPERTS="${2:-0.1}"
CACHE_MB="${3:-10240}"

DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${DIR}/.." && pwd)"
CLI="${PROJECT_DIR}/zig-out/bin/dmlx"
REPORT="${PROJECT_DIR}/docs/en/analysis/performance-benchmark.md"
PORT=18090
SERVER_URL="http://localhost:${PORT}"

export MLX_METAL_FAST_SYNCH=1
export BM_COMMIT=$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo "?")
export BM_BRANCH=$(git -C "$PROJECT_DIR" branch --show-current 2>/dev/null || echo "?")
export BM_DATE=$(date +%Y-%m-%d)
export BM_HW=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "?")
export BM_MEM=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0fGB",$1/1073741824}' || echo "?")

T0=$(date +%s)

# ------------------------------------------------------------------
# Phase 0: Build
# ------------------------------------------------------------------
echo "🔧 Build (ReleaseFast)..."
(cd "$PROJECT_DIR" && zig build -Doptimize=ReleaseFast 2>/dev/null) || {
    echo "❌ Build failed"; exit 1
}

# ------------------------------------------------------------------
# Phase 1: Unit Tests
# ------------------------------------------------------------------
echo "🧪 Unit tests..."
T_UNIT=$(date +%s)
if (cd "$PROJECT_DIR" && zig build test >/dev/null 2>&1); then
    export BM_UNIT="PASS (430+)"
else
    export BM_UNIT="FAIL"
fi
echo "   $BM_UNIT ($(($(date +%s)-T_UNIT))s)"

# Rebuild after test (zig build test compiles in Debug, overwrites binary)
(cd "$PROJECT_DIR" && zig build -Doptimize=ReleaseFast 2>/dev/null)

# ------------------------------------------------------------------
# Phase 2: Serve Mode Performance Test
# ------------------------------------------------------------------
echo "📊 Serve mode perf test (smelt=${SMELT_EXPERTS}, cache=${CACHE_MB}MB)..."

# Cleanup
cleanup() {
    pkill -9 dmlx 2>/dev/null || true
}
trap cleanup EXIT
cleanup
sleep 2

# Start server
T_PERF=$(date +%s)
"$CLI" serve \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --max-tokens 256 \
    --temperature 0 \
    --smelt --smelt-strategy stream --smelt-experts "$SMELT_EXPERTS" \
    --smelt-cache "$CACHE_MB" > /tmp/benchmark_serve.log 2>&1 &
SERVER_PID=$!

# Wait for server ready
echo "   Waiting for server..."
STARTUP_SECS=0
for i in {1..180}; do
    if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
        STARTUP_SECS=$i
        echo "   Server ready (${i}s)"
        break
    fi
    sleep 1
done

if ! curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo "❌ Server failed to start"
    tail -30 /tmp/benchmark_serve.log
    exit 1
fi

# --- Perf: Generate 30 tokens to measure steady-state latency ---
echo "   Generating 30 tokens (perf measurement)..."
PERF_RESULT=$(curl -sf -w '\n%{time_starttransfer}|%{time_total}' \
    --max-time 300 \
    "${SERVER_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"Hello"}],"max_tokens":30,"temperature":0}' 2>&1)

PERF_BODY=$(echo "$PERF_RESULT" | sed '$d')
PERF_TIMING=$(echo "$PERF_RESULT" | tail -1)
PERF_TTFR=$(echo "$PERF_TIMING" | cut -d'|' -f1)
PERF_TOTAL=$(echo "$PERF_TIMING" | cut -d'|' -f2)
PERF_TOKENS=$(echo "$PERF_BODY" | jq -r '.usage.completion_tokens // 0' 2>/dev/null || echo "0")

echo "   30-token: TTFR=${PERF_TTFR}s total=${PERF_TOTAL}s tokens=${PERF_TOKENS}"

# --- Perf: Generate 100 tokens to measure long-generation performance ---
echo "   Generating 100 tokens (long-gen measurement)..."
LONG_RESULT=$(curl -sf -w '\n%{time_starttransfer}|%{time_total}' \
    --max-time 600 \
    "${SERVER_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"Explain the concept of machine learning in simple terms."}],"max_tokens":100,"temperature":0}' 2>&1)

LONG_BODY=$(echo "$LONG_RESULT" | sed '$d')
LONG_TIMING=$(echo "$LONG_RESULT" | tail -1)
LONG_TTFR=$(echo "$LONG_TIMING" | cut -d'|' -f1)
LONG_TOTAL=$(echo "$LONG_TIMING" | cut -d'|' -f2)
LONG_TOKENS=$(echo "$LONG_BODY" | jq -r '.usage.completion_tokens // 0' 2>/dev/null || echo "0")

echo "   100-token: TTFR=${LONG_TTFR}s total=${LONG_TOTAL}s tokens=${LONG_TOKENS}"

# Extract token step data from server log
PF=$(mktemp)
grep "Token step.*complete" /tmp/benchmark_serve.log > "$PF"
export BM_PERF_SECS=$(($(date +%s)-T_PERF))
echo "   Perf phase: ${BM_PERF_SECS}s"

# --- E2E: 7-Prompt correctness test via serve mode ---
echo "✅ E2E (7 prompts via serve)..."
T_E2E=$(date +%s)

PROMPTS=(
    "2+2=|4"
    "The capital of France is|Paris"
    "What temperature does water freeze at in Celsius? Just give the number.|0"
    "Is the Earth round? Reply with only yes or no.|yes"
    "3*3=|9"
    "10-5=|5"
    "What is capital of France?|Paris"
)

EF=$(mktemp)
E2E_PASS=0
E2E_FAIL=0

for idx in "${!PROMPTS[@]}"; do
    IFS='|' read -r prompt expected <<< "${PROMPTS[$idx]}"
    
    result=$(curl -sf --max-time 300 \
        "${SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":30,\"temperature\":0}" 2>&1)
    
    content=$(echo "$result" | jq -r '.choices[0].message.content // ""' 2>/dev/null || echo "")
    
    if echo "$content" | grep -qi "$expected"; then
        echo "   P$((idx+1)): ✅ PASSED"
        echo "✅ PASSED P$((idx+1)): ${prompt}" >> "$EF"
        echo "   Generated: ${content:0:80}" >> "$EF"
        E2E_PASS=$((E2E_PASS + 1))
    else
        echo "   P$((idx+1)): ❌ FAILED (expected '${expected}' in output)"
        echo "❌ FAILED P$((idx+1)): ${prompt}" >> "$EF"
        echo "   Generated: ${content:0:80}" >> "$EF"
        E2E_FAIL=$((E2E_FAIL + 1))
    fi
    sleep 1
done

export BM_E2E_SECS=$(($(date +%s)-T_E2E))
echo "   Results: ${E2E_PASS} passed, ${E2E_FAIL} failed (${BM_E2E_SECS}s)"

# --- Extract server-side RequestLog metrics ---
echo ""
echo "📈 Server-side metrics:"
grep "RequestLog" /tmp/benchmark_serve.log | tail -10

# --- Memory usage ---
SERVER_RSS=$(ps -o rss= -p $SERVER_PID 2>/dev/null | awk '{printf "%.0f", $1/1024}' || echo "0")
echo "   Server RSS: ${SERVER_RSS}MB"

# Stop server
cleanup

export BM_TOTAL_SECS=$(($(date +%s)-T0))

# ------------------------------------------------------------------
# Phase 3: Generate Report
# ------------------------------------------------------------------
echo ""
echo "📝 Generating report..."

# Export additional env vars for report
export BM_SMELT_EXPERTS="$SMELT_EXPERTS"
export BM_CACHE_MB="$CACHE_MB"
export BM_E2E_PASS="$E2E_PASS"
export BM_E2E_FAIL="$E2E_FAIL"
export BM_PERF_TTFR="$PERF_TTFR"
export BM_PERF_TOTAL="$PERF_TOTAL"
export BM_PERF_TOKENS="$PERF_TOKENS"
export BM_LONG_TTFR="$LONG_TTFR"
export BM_LONG_TOTAL="$LONG_TOTAL"
export BM_LONG_TOKENS="$LONG_TOKENS"
export BM_SERVER_RSS="$SERVER_RSS"
export BM_STARTUP_SECS="$STARTUP_SECS"

python3 "$DIR/_gen_report.py" "$PF" "$EF" "$REPORT"
rm -f "$PF" "$EF"

echo "✅ Done → $REPORT (${BM_TOTAL_SECS}s total)"
