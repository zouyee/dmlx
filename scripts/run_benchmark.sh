#!/bin/bash
# ============================================================
# dmlx Performance Benchmark — Serve Mode
# ============================================================
# Generates: docs/en/analysis/performance-benchmark.md
#
# Supports two modes:
#
#   NATIVE mode (default, --native engine with SMELT N=51):
#     bash scripts/run_benchmark.sh
#     bash scripts/run_benchmark.sh --native
#     NATIVE_SMELT_N=20 bash scripts/run_benchmark.sh  # lighter SMELT for low-RAM
#
#   MLX mode (legacy):
#     bash scripts/run_benchmark.sh --mlx [model_path] [smelt_experts] [cache_mb]
#
# Native mode measures tok/s correctly:
#   1. Start server with SMELT N=51 (waits for 35GB RAM preload)
#   2. Send warmup request to heat GPU caches
#   3. Send 3 timed requests (max_tokens=5) → report median tok/s
#   4. Correctness gate: Paris must appear in response
# ============================================================

set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${DIR}/.." && pwd)"
CLI="${PROJECT_DIR}/zig-out/bin/dmlx"
REPORT="${PROJECT_DIR}/docs/en/analysis/performance-benchmark.md"
PORT=18090
SERVER_URL="http://localhost:${PORT}"

# Detect mode
MODE="native"
if [[ "${1:-}" == "--mlx" ]]; then
    MODE="mlx"
    shift
elif [[ "${1:-}" == "--native" ]]; then
    MODE="native"
    shift
fi

MODEL_PATH="${1:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
PACKED_DIR="${MODEL_PATH}/packed_experts"
NATIVE_SMELT_N="${NATIVE_SMELT_N:-20}"

# MLX-mode legacy params
SMELT_EXPERTS="${2:-0.20}"
CACHE_MB="${3:-0}"
EXPERT_PARALLEL="${4:-18}"

# Purge page cache for clean baseline (requires sudo, skip if not available).
if command -v purge &>/dev/null && [[ "${SKIP_PURGE:-0}" != "1" ]]; then
    sudo -n purge 2>/dev/null || echo "⚠️  sudo purge skipped (no password) — results may vary"
fi

export BM_COMMIT=$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo "?")
export BM_BRANCH=$(git -C "$PROJECT_DIR" branch --show-current 2>/dev/null || echo "?")
export BM_DATE=$(date +%Y-%m-%d)
export BM_HW=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "?")
export BM_MEM=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0fGB",$1/1073741824}' || echo "?")

T0=$(date +%s)

echo "════════════════════════════════════════════════"
echo "  dmlx Benchmark  mode=${MODE}  commit=${BM_COMMIT}"
echo "  hw: ${BM_HW} ${BM_MEM}"
echo "════════════════════════════════════════════════"

# ------------------------------------------------------------------
# Phase 0: Build (clean cache to avoid stale struct layout issues)
# ------------------------------------------------------------------
echo "🔧 Build (clean + ReleaseFast)..."
(cd "$PROJECT_DIR" && rm -rf .zig-cache && zig build -Doptimize=ReleaseFast 2>/dev/null) || {
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

# Cleanup helper
cleanup() {
    # Use graceful shutdown (SIGTERM) so routing stats are saved via deinit()
    # Fall back to SIGKILL only if graceful shutdown times out
    if curl -sf --max-time 3 -X POST "http://localhost:${PORT}/shutdown" > /dev/null 2>&1; then
        sleep 2
    fi
    pkill -f "dmlx serve.*${PORT}" 2>/dev/null || true
    sleep 1
    # Delete routing stats before each benchmark run to ensure reproducible results.
    # With penalty=0 (natural routing), stats-based SMELT may hurt benchmark performance
    # for prompts not in the training stats. Phase 1 (default experts 0..N-1) gives
    # consistent baseline numbers. Production servers benefit from persistent stats;
    # benchmarks need fresh state for fair comparisons.
    rm -f "${PACKED_DIR}/.smelt_routing_stats.bin" 2>/dev/null || true
}
trap cleanup EXIT
cleanup

# ==================================================================
# NATIVE MODE: SMELT N=51, warmup + sequential requests
# ==================================================================
if [[ "$MODE" == "native" ]]; then
    echo "📊 Native engine perf test (SMELT N=${NATIVE_SMELT_N})..."

    # Check available memory
    AVAIL_GB=$(python3 -c "
import subprocess
out = subprocess.run(['vm_stat'],capture_output=True,text=True).stdout
pages={}
for line in out.split('\n'):
    for k in ['Pages free','Pages inactive','Pages speculative']:
        if k in line: pages[k]=int(line.split(':')[1].strip().rstrip('.'))
pg=16384
print(int((pages.get('Pages free',0)+pages.get('Pages inactive',0)+pages.get('Pages speculative',0))*pg/1024**3))
" 2>/dev/null || echo "0")
    NEEDED_GB=$(python3 -c "print(int(${NATIVE_SMELT_N} * 43 * 13.4 / 1024) + 5)")
    if (( AVAIL_GB < NEEDED_GB )); then
        echo "⚠️  Only ${AVAIL_GB}GB available, SMELT N=${NATIVE_SMELT_N} needs ~${NEEDED_GB}GB"
    fi

    T_PERF=$(date +%s)
    NATIVE_SMELT_N="${NATIVE_SMELT_N}" "$CLI" serve \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --native \
        --expert-packed-dir "$PACKED_DIR" \
        > /tmp/benchmark_serve.log 2>&1 &

    # Wait for server ready (up to 180s for SMELT preload)
    echo -n "   Waiting for server (SMELT preload)..."
    STARTUP_SECS=0
    for i in $(seq 1 180); do
        if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
            STARTUP_SECS=$i
            echo " ready (${i}s)"
            break
        fi
        echo -n "."
        sleep 1
    done
    if ! curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
        echo -e "\n❌ Server failed to start"; tail -20 /tmp/benchmark_serve.log; exit 1
    fi

    # Warmup: heat GPU caches — send 5 requests to fully warm pipeline state
    # (Metal PSO compilation + GPU thread warmup can take first 3-4 requests)
    echo "   Warmup (5 requests)..."
    for i in 1 2 3 4 5; do
        curl -s --max-time 60 "${SERVER_URL}/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model":"d","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0}' \
            > /dev/null 2>&1 || true
    done

    # Correctness: Paris (sequential, measured but not counted in perf)
    echo "   Correctness check..."
    PARIS_RESP=$(curl -s --max-time 120 "${SERVER_URL}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"d","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":10,"temperature":0}')
    PARIS_TEXT=$(echo "$PARIS_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
    if echo "$PARIS_TEXT" | grep -qi "paris"; then
        echo "   ✓ Paris correct: \"${PARIS_TEXT}\""
        NATIVE_CORRECT=1
    else
        echo "   ✗ Paris FAILED: \"${PARIS_TEXT}\""
        NATIVE_CORRECT=0
    fi

    # Re-warm GPU after Paris (Paris uses a different prompt, which can re-cold the pipeline)
    echo "   Re-warm after Paris (4 requests)..."
    for i in 1 2 3 4; do
        curl -s --max-time 60 "${SERVER_URL}/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model":"d","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0}' \
            > /dev/null 2>&1 || true
    done

    # Perf: 3 sequential requests, report median tok/s
    echo "   Performance (3 sequential runs, max_tokens=5)..."
    NATIVE_TPS_LIST=()
    for i in 1 2 3; do
        T_REQ=$(python3 -c "import time; print(time.time())")
        RESP=$(curl -s --max-time 60 "${SERVER_URL}/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model":"d","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0}')
        T_DONE=$(python3 -c "import time; print(time.time())")
        RESP_TEXT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
        TPS=$(python3 -c "print(f'{5/($T_DONE-$T_REQ):.3f}')")
        ELAPSED=$(python3 -c "print(f'{$T_DONE-$T_REQ:.1f}')")
        echo "   run${i}: ${ELAPSED}s | ${TPS} tok/s | \"${RESP_TEXT}\""
        NATIVE_TPS_LIST+=("$TPS")
    done
    NATIVE_MEDIAN_TPS=$(python3 -c "
vals=sorted([${NATIVE_TPS_LIST[0]},${NATIVE_TPS_LIST[1]},${NATIVE_TPS_LIST[2]}])
print(f'{vals[1]:.3f}')")
    echo "   Median: ${NATIVE_MEDIAN_TPS} tok/s"

    export BM_NATIVE_TPS="$NATIVE_MEDIAN_TPS"
    export BM_NATIVE_CORRECT="$NATIVE_CORRECT"
    export BM_PERF_SECS=$(($(date +%s)-T_PERF))
    export BM_SMELT_N="$NATIVE_SMELT_N"

    # --- E2E: 7-Prompt correctness test via native serve mode ---
    echo ""
    echo "✅ E2E (7 prompts via native serve)..."
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

    cleanup

    # --- DSpark benchmark (optional, if weights available) ---
    DSPARK_DIR="${HOME}/models/DeepSeek-V4-Flash-DSpark/dspark_weights"
    DSPARK_PACKED="${HOME}/models/DeepSeek-V4-Flash-DSpark/packed_mtp_experts"
    if [ -f "${DSPARK_DIR}/markov_w1.bin" ] && [ -d "${DSPARK_PACKED}" ]; then
        echo ""
        echo "📊 DSpark benchmark (SMELT N=${NATIVE_SMELT_N})..."
        cleanup

        NATIVE_SMELT_N="${NATIVE_SMELT_N}" "$CLI" serve \
            --model "$MODEL_PATH" \
            --port "$PORT" \
            --native \
            --expert-packed-dir "$PACKED_DIR" \
            --dspark "$DSPARK_DIR" \
            --max-tokens 30 \
            --temperature 0 \
            > /tmp/benchmark_dspark.log 2>&1 &

        echo -n "   Waiting for DSpark server..."
        for i in $(seq 1 120); do
            if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
                echo " ready (${i}s)"
                break
            fi
            echo -n "."
            sleep 1
        done

        if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
            # Warmup
            curl -s --max-time 120 "${SERVER_URL}/v1/chat/completions" \
                -H 'Content-Type: application/json' \
                -d '{"model":"d","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0,"stream":false}' > /dev/null 2>&1

            # Test with diverse prompt to measure acceptance
            echo "   DSpark perf (max_tokens=30, diverse prompt)..."
            T_DSPARK=$(python3 -c "import time; print(time.time())")
            DSPARK_RESP=$(curl -s --max-time 300 "${SERVER_URL}/v1/chat/completions" \
                -H 'Content-Type: application/json' \
                -d '{"model":"d","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":30,"temperature":0,"stream":false}')
            T_DSPARK_DONE=$(python3 -c "import time; print(time.time())")
            DSPARK_TEXT=$(echo "$DSPARK_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null || echo "")
            DSPARK_TOKS=$(echo "$DSPARK_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "0")
            DSPARK_ELAPSED=$(python3 -c "print(f'{$T_DSPARK_DONE-$T_DSPARK:.1f}')")
            DSPARK_TPS=$(python3 -c "t=$DSPARK_TOKS; e=$T_DSPARK_DONE-$T_DSPARK; print(f'{t/e:.3f}' if e>0 and t>0 else '0')")

            # Get DSpark stats from server log
            DSPARK_STATS=$(grep "dspark-stats" /tmp/benchmark_dspark.log | tail -1 || echo "")

            echo "   DSpark result: ${DSPARK_TOKS} tokens in ${DSPARK_ELAPSED}s = ${DSPARK_TPS} tok/s"
            echo "   DSpark output: \"${DSPARK_TEXT:0:60}\""
            echo "   DSpark stats: ${DSPARK_STATS}"
            if echo "$DSPARK_TEXT" | grep -qi "paris"; then
                echo "   ✓ Paris correct with DSpark"
            else
                echo "   ✗ Paris NOT found with DSpark"
            fi
        else
            echo "   ⚠️  DSpark server failed to start"
        fi
        cleanup
    else
        echo ""
        echo "⚠️  DSpark weights not found, skipping DSpark benchmark"
        echo "   Expected: ${DSPARK_DIR}/markov_w1.bin"
    fi

    # Report
    echo ""
    echo "════════════════════════════════════════"
    echo "  Native Engine Benchmark Results"
    echo "  commit:  ${BM_COMMIT}"
    echo "  SMELT N: ${NATIVE_SMELT_N}"
    echo "  tok/s:   ${NATIVE_MEDIAN_TPS} (median of 3 sequential)"
    echo "  Paris:   $([ "$NATIVE_CORRECT" -eq 1 ] && echo '✓ PASS' || echo '✗ FAIL')"
    echo "  E2E:     ${E2E_PASS}/7 passed"
    echo "  unit:    ${BM_UNIT}"
    echo "  time:    ${BM_PERF_SECS}s"
    echo "════════════════════════════════════════"
    exit $([ "$NATIVE_CORRECT" -eq 1 ] && [ "$E2E_FAIL" -eq 0 ] && echo 0 || echo 1)
fi

# ==================================================================
# MLX MODE (legacy)
# ==================================================================
echo "📊 MLX serve mode perf test (smelt=${SMELT_EXPERTS}, cache=${CACHE_MB}MB)..."

# Validate packed expert directory for MLX mode
if [ ! -d "$PACKED_DIR" ]; then
    echo "⚠️  Packed expert directory not found: $PACKED_DIR"
    PACKED_DIR=""
    EXPERT_PARALLEL=0
fi

T_PERF=$(date +%s)
if [ -n "$PACKED_DIR" ] && [ -d "$PACKED_DIR" ]; then
    "$CLI" serve \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --max-tokens 256 \
        --temperature 0 \
        --smelt --smelt-strategy stream --smelt-experts "$SMELT_EXPERTS" \
        --smelt-cache "$CACHE_MB" \
        --expert-packed-dir "$PACKED_DIR" \
        --expert-parallel "$EXPERT_PARALLEL" > /tmp/benchmark_serve.log 2>&1 &
else
    "$CLI" serve \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --max-tokens 256 \
        --temperature 0 \
        --smelt --smelt-strategy stream --smelt-experts "$SMELT_EXPERTS" \
        --smelt-cache "$CACHE_MB" > /tmp/benchmark_serve.log 2>&1 &
fi
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
export BM_MLOCK="false"
export BM_E2E_PASS="$E2E_PASS"
export BM_E2E_FAIL="$E2E_FAIL"
export BM_PERF_TTFR="$PERF_TTFR"
export BM_PERF_TOTAL="$PERF_TOTAL"
export BM_PERF_TOKENS="$PERF_TOKENS"
export BM_LONG_TTFR="$LONG_TTFR"
export BM_LONG_TOTAL="$LONG_TOTAL"
export BM_LONG_TOKENS="$LONG_TOKENS"
export BM_PACKED_DIR="$PACKED_DIR"
export BM_PARALLEL="$EXPERT_PARALLEL"
export BM_SERVER_RSS="$SERVER_RSS"
export BM_STARTUP_SECS="$STARTUP_SECS"

python3 "$DIR/_gen_report.py" "$PF" "$EF" "$REPORT"
rm -f "$PF" "$EF"

echo "✅ Done → $REPORT (${BM_TOTAL_SECS}s total)"
