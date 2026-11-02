#!/bin/bash
# ============================================================
# dmlx Performance Benchmark — Qwen3.8-27B (MLX serve path)
# ============================================================
# Lightweight counterpart to run_benchmark.sh (which is DSV4-only:
# --native engine, packed experts, NATIVE_TELEMETRY). Qwen3.8 is a dense
# model served through the registry vtable path, so this script only uses
# plain `dmlx serve` and HTTP-level measurements.
#
# Usage:
#   bash scripts/run_benchmark_qwen38.sh [model_path]
#
# Env:
#   SKIP_BENCHMARK_BUILD=1   reuse zig-out/bin/dmlx as-is
#   SKIP_PURGE=1             skip sudo purge
#   BENCHMARK_JSON=<path>    override JSON output path
#   EXTRA_SERVE_FLAGS="..."  extra flags for `dmlx serve` (e.g. --speculative-mtp 2)
# ============================================================

set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${DIR}/.." && pwd)"
CLI="${PROJECT_DIR}/zig-out/bin/dmlx"
PORT=18091
SERVER_URL="http://localhost:${PORT}"
SERVE_LOG="/tmp/benchmark_qwen38_serve.log"

MODEL_PATH="${1:-${HOME}/models/Qwen3.8-27B-4bit}"
RESULT_JSON="${BENCHMARK_JSON:-${PROJECT_DIR}/docs/en/analysis/performance-benchmark-qwen38.json}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "❌ Model not found: $MODEL_PATH"
    exit 1
fi

export BM_COMMIT=$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo "?")
export BM_BRANCH=$(git -C "$PROJECT_DIR" branch --show-current 2>/dev/null || echo "?")
export BM_DATE=$(date +%Y-%m-%d)
export BM_HW=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "?")
export BM_MEM=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0fGB",$1/1073741824}' || echo "?")

T0=$(date +%s)

echo "════════════════════════════════════════════════"
echo "  dmlx Benchmark — Qwen3.8 (MLX serve path)"
echo "  commit: ${BM_COMMIT}  branch: ${BM_BRANCH}"
echo "  hw: ${BM_HW} ${BM_MEM}"
echo "  model: ${MODEL_PATH}"
echo "  standard: warmup=5, measured=3, max_tokens=5"
echo "  diagnostic: 1 run, max_tokens=100 (excluded from median)"
echo "════════════════════════════════════════════════"

if command -v purge &>/dev/null && [[ "${SKIP_PURGE:-0}" != "1" ]]; then
    sudo -n purge 2>/dev/null || echo "⚠️  sudo purge skipped (no password) — results may vary"
fi

# ------------------------------------------------------------------
# Build
# ------------------------------------------------------------------
if [[ "${SKIP_BENCHMARK_BUILD:-0}" == "1" ]]; then
    [[ -x "$CLI" ]] || { echo "❌ $CLI missing"; exit 1; }
    echo "🔧 Build skipped (SKIP_BENCHMARK_BUILD=1)"
else
    echo "🔧 Build (ReleaseFast)..."
    (cd "$PROJECT_DIR" && zig build -Doptimize=ReleaseFast 2>&1 | grep -E "^error" && exit 1)
    echo "✅ Build passed"
fi

cleanup() {
    if [[ -n "${RSS_SAMPLER_PID:-}" ]]; then
        kill "$RSS_SAMPLER_PID" 2>/dev/null || true
        wait "$RSS_SAMPLER_PID" 2>/dev/null || true
        RSS_SAMPLER_PID=""
    fi
    if curl -sf --max-time 3 -X POST "${SERVER_URL}/shutdown" > /dev/null 2>&1; then
        sleep 2
    fi
    pkill -f "dmlx serve.*--port ${PORT}" 2>/dev/null || true
    sleep 1
}
trap cleanup EXIT
cleanup

# ------------------------------------------------------------------
# Serve
# ------------------------------------------------------------------
T_PERF=$(date +%s)
"$CLI" serve --model "$MODEL_PATH" --port "$PORT" ${EXTRA_SERVE_FLAGS:-} > "$SERVE_LOG" 2>&1 &
SERVER_PID=$!

RSS_SAMPLE_FILE=$(mktemp)
echo 0 > "$RSS_SAMPLE_FILE"
(
    peak=0
    while kill -0 "$SERVER_PID" 2>/dev/null; do
        rss_kib=$(ps -o rss= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')
        if [[ "$rss_kib" =~ ^[0-9]+$ ]] && (( rss_kib * 1024 > peak )); then
            peak=$((rss_kib * 1024)); echo "$peak" > "$RSS_SAMPLE_FILE"
        fi
        sleep 0.2
    done
) &
RSS_SAMPLER_PID=$!

echo -n "   Waiting for server (27B weight load)..."
STARTUP_SECS=0
for i in $(seq 1 300); do
    if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
        STARTUP_SECS=$i
        echo " ready (${i}s)"
        break
    fi
    echo -n "."
    sleep 1
done
if ! curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo -e "\n❌ Server failed to start"; tail -20 "$SERVE_LOG"; exit 1
fi

request() { # $1=prompt $2=max_tokens -> response body on stdout
    curl -s --max-time 600 "${SERVER_URL}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"d\",\"messages\":[{\"role\":\"user\",\"content\":$1}],\"max_tokens\":$2,\"temperature\":0}"
}

# ------------------------------------------------------------------
# Warmup (Metal PSO compilation + pipeline warmup)
# ------------------------------------------------------------------
echo "   Warmup (5 requests)..."
for i in 1 2 3 4 5; do
    request '"Hi"' 5 > /dev/null 2>&1 || { echo "❌ Warmup request failed"; exit 1; }
done

# ------------------------------------------------------------------
# Correctness: Paris. Generous max_tokens because Qwen3.8 thinks first
# (<think> block); match on post-think text as well as raw.
# ------------------------------------------------------------------
echo "   Correctness check..."
PARIS_TEXT=$(request '"The capital of France is"' 64 | python3 -c "
import re, sys, json
print(re.sub(r'<think>.*?</think>', '', json.load(sys.stdin)['choices'][0]['message']['content'], flags=re.S))" 2>/dev/null || echo "")
if echo "$PARIS_TEXT" | grep -qi "paris"; then
    echo "   ✓ Paris correct: \"${PARIS_TEXT}\""
    QWEN_CORRECT=1
else
    echo "   ✗ Paris FAILED: \"${PARIS_TEXT}\""
    QWEN_CORRECT=0
fi

# ------------------------------------------------------------------
# Standard decode: 3 sequential 5-token runs, median tok/s
# ------------------------------------------------------------------
echo "   STANDARD_DECODE: 3 sequential runs, max_tokens=5..."
TPS_LIST=()
ELAPSED_LIST=()
for i in 1 2 3; do
    T_REQ_NS=$(python3 -c "import time; print(time.monotonic_ns())")
    RESP=$(request '"Hi"' 5)
    T_DONE_NS=$(python3 -c "import time; print(time.monotonic_ns())")
    read -r N_PROMPT N_COMPLETION <<< "$(echo "$RESP" | python3 -c "import sys,json; u=json.load(sys.stdin).get('usage',{}); print(int(u.get('prompt_tokens',0)), int(u.get('completion_tokens',0)))" 2>/dev/null || echo '0 0')"
    if ! [[ "$N_PROMPT" =~ ^[0-9]+$ && "$N_COMPLETION" =~ ^[0-9]+$ ]] || (( N_PROMPT <= 0 || N_COMPLETION <= 0 || N_COMPLETION > 5 )); then
        echo "❌ Invalid STANDARD_DECODE usage for run${i}: prompt=${N_PROMPT:-?} completion=${N_COMPLETION:-?}"
        exit 1
    fi
    ELAPSED=$(python3 -c "print(f'{($T_DONE_NS-$T_REQ_NS)/1_000_000_000:.6f}')")
    TPS=$(python3 -c "print(f'{$N_COMPLETION/$ELAPSED:.6f}')")
    echo "   run${i}: $(python3 -c "print(f'{$ELAPSED:.1f}')")s | $(python3 -c "print(f'{$TPS:.3f}')") tok/s | prompt=${N_PROMPT} completion=${N_COMPLETION}"
    TPS_LIST+=("$TPS")
    ELAPSED_LIST+=("$ELAPSED")
done
MEDIAN_TPS=$(python3 -c "
vals=sorted([${TPS_LIST[0]},${TPS_LIST[1]},${TPS_LIST[2]}])
print(f'{vals[1]:.6f}')")
echo "   Standard median (5-token runs): $(python3 -c "print(f'{${MEDIAN_TPS}:.3f}')") tok/s"

# ------------------------------------------------------------------
# Long-decode diagnostic: 1 x 100 tokens (excluded from median)
# ------------------------------------------------------------------
echo "   LONG_DECODE_DIAGNOSTIC: 1 run, max_tokens=100..."
T_REQ_NS=$(python3 -c "import time; print(time.monotonic_ns())")
RESP=$(request '"Count from 1 to 50."' 100)
T_DONE_NS=$(python3 -c "import time; print(time.monotonic_ns())")
read -r LONG_PROMPT N_COMPLETION <<< "$(echo "$RESP" | python3 -c "import sys,json; u=json.load(sys.stdin).get('usage',{}); print(int(u.get('prompt_tokens',0)), int(u.get('completion_tokens',0)))" 2>/dev/null || echo '0 0')"
if ! [[ "$LONG_PROMPT" =~ ^[0-9]+$ && "$N_COMPLETION" =~ ^[0-9]+$ ]] || (( LONG_PROMPT <= 0 || N_COMPLETION <= 0 || N_COMPLETION > 100 )); then
    echo "❌ Invalid LONG_DECODE_DIAGNOSTIC usage: prompt=${LONG_PROMPT:-?} completion=${N_COMPLETION:-?}"
    exit 1
fi
LONG_ELAPSED=$(python3 -c "print(f'{($T_DONE_NS-$T_REQ_NS)/1_000_000_000:.6f}')")
LONG_TPS=$(python3 -c "print(f'{$N_COMPLETION/$LONG_ELAPSED:.6f}')")
echo "   long-decode diagnostic: $(python3 -c "print(f'{$LONG_ELAPSED:.1f}')")s | $(python3 -c "print(f'{$LONG_TPS:.3f}')") tok/s"

# ------------------------------------------------------------------
# E2E: 7 prompts
# ------------------------------------------------------------------
echo ""
echo "✅ E2E (7 prompts)..."
PROMPTS=(
    "2+2=|4"
    "The capital of France is|Paris"
    "What temperature does water freeze at in Celsius? Just give the number.|0"
    "Is the Earth round? Reply with only yes or no.|yes"
    "3*3=|9"
    "10-5=|5"
    "What is capital of France?|Paris"
)
E2E_PASS=0
E2E_FAIL=0
for idx in "${!PROMPTS[@]}"; do
    IFS='|' read -r prompt expected <<< "${PROMPTS[$idx]}"
    content=$(request "\"${prompt}\"" 64 | python3 -c "
import re, sys, json
print(re.sub(r'<think>.*?</think>', '', json.load(sys.stdin)['choices'][0]['message']['content'], flags=re.S))" 2>/dev/null || echo "")
    if echo "$content" | grep -qi "$expected"; then
        echo "   P$((idx+1)): ✅ PASSED"
        E2E_PASS=$((E2E_PASS + 1))
    else
        echo "   P$((idx+1)): ❌ FAILED (expected '${expected}', got '${content:0:60}')"
        E2E_FAIL=$((E2E_FAIL + 1))
    fi
    sleep 1
done
echo "   Results: ${E2E_PASS} passed, ${E2E_FAIL} failed"

if [[ -n "${RSS_SAMPLER_PID:-}" ]]; then
    kill "$RSS_SAMPLER_PID" 2>/dev/null || true
    wait "$RSS_SAMPLER_PID" 2>/dev/null || true
fi
PEAK_RSS_BYTES=$(cat "$RSS_SAMPLE_FILE" 2>/dev/null || echo 0)
rm -f "$RSS_SAMPLE_FILE"

# ------------------------------------------------------------------
# JSON artifact + summary
# ------------------------------------------------------------------
BM_PERF_SECS=$(($(date +%s)-T_PERF))
BM_TOTAL_SECS=$(($(date +%s)-T0))

python3 - "$RESULT_JSON" <<PY
import json
from pathlib import Path
doc = {
    "schema": "dmlx-qwen38-benchmark-v1",
    "commit": "${BM_COMMIT}",
    "branch": "${BM_BRANCH}",
    "date": "${BM_DATE}",
    "hardware": "${BM_HW}",
    "memory": "${BM_MEM}",
    "model_path": "${MODEL_PATH}",
    "startup_secs": ${STARTUP_SECS},
    "standard_median_tps": ${MEDIAN_TPS},
    "standard_tps_runs": [${TPS_LIST[0]}, ${TPS_LIST[1]}, ${TPS_LIST[2]}],
    "long_diagnostic_tps": ${LONG_TPS},
    "long_diagnostic_elapsed_s": ${LONG_ELAPSED},
    "paris_correct": ${QWEN_CORRECT} == 1,
    "e2e_pass": ${E2E_PASS},
    "e2e_fail": ${E2E_FAIL},
    "peak_rss_bytes": ${PEAK_RSS_BYTES},
    "perf_secs": ${BM_PERF_SECS},
    "total_secs": ${BM_TOTAL_SECS},
    "serve_flags": "${EXTRA_SERVE_FLAGS:-}",
    "command": "EXTRA_SERVE_FLAGS=\"${EXTRA_SERVE_FLAGS:-}\" bash scripts/run_benchmark_qwen38.sh ${MODEL_PATH}",
}
Path("$RESULT_JSON").parent.mkdir(parents=True, exist_ok=True)
Path("$RESULT_JSON").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
PY

echo ""
echo "════════════════════════════════════════"
echo "  Qwen3.8 Benchmark Results"
echo "  commit:  ${BM_COMMIT}"
echo "  tok/s:   $(python3 -c "print(f'{${MEDIAN_TPS}:.3f}')") (median of 3 sequential)"
echo "  decode:  $(python3 -c "print(f'{${LONG_TPS}:.3f}')") tok/s (long diagnostic)"
echo "  Paris:   $([ "$QWEN_CORRECT" -eq 1 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "  E2E:     ${E2E_PASS}/7 passed"
echo "  peakRSS: $(python3 -c "print(f'{${PEAK_RSS_BYTES}/2**30:.1f}')") GiB"
echo "  time:    ${BM_PERF_SECS}s"
echo "  JSON:    ${RESULT_JSON}"
echo "════════════════════════════════════════"

exit $([ "$QWEN_CORRECT" -eq 1 ] && [ "$E2E_FAIL" -eq 0 ] && echo 0 || echo 1)
