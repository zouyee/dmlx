#!/bin/bash
# native_bench.sh — Native engine (SMELT N=51) performance benchmark.
#
# Measures tok/s in the correct way:
#   1. Start server with SMELT N=51 (35GB RAM preload)
#   2. Send a warmup request to heat up GPU caches
#   3. Send 3 timed requests (max_tokens=5, short prompt)
#   4. Report median tok/s and correctness
#
# Correctness gate: Paris must appear in response.
# Performance gate: median tok/s must be >= MIN_TPS (default 0.3).
#
# Usage:
#   bash scripts/native_bench.sh
#   MIN_TPS=0.4 bash scripts/native_bench.sh   # stricter threshold
#   NATIVE_SMELT_N=20 bash scripts/native_bench.sh  # lighter SMELT for low-RAM machines

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
PACKED_DIR="${MODEL_PATH}/packed_experts"
CLI="${PROJECT_DIR}/zig-out/bin/dmlx"
PORT="${PORT:-8932}"
MIN_TPS="${MIN_TPS:-0.30}"
SMELT_N="${NATIVE_SMELT_N:-51}"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo "════════════════════════════════════════════════"
echo "  Native Engine Benchmark (SMELT N=${SMELT_N})"
echo "  commit: $(git -C "${PROJECT_DIR}" rev-parse --short HEAD 2>/dev/null)"
echo "  model:  ${MODEL_PATH}"
echo "════════════════════════════════════════════════"

# Prereq check
if [[ ! -x "${CLI}" ]]; then
    echo -e "${RED}✗ binary not found. Run: zig build -Doptimize=ReleaseFast${NC}"
    exit 1
fi
if [[ ! -d "${PACKED_DIR}" ]]; then
    echo -e "${RED}✗ packed_experts not found: ${PACKED_DIR}${NC}"
    exit 1
fi

# Check available memory (SMELT N=51 needs ~35GB)
AVAIL_GB=$(python3 -c "
import subprocess
out = subprocess.run(['vm_stat'],capture_output=True,text=True).stdout
pages={}
for line in out.split('\n'):
    for k in ['Pages free','Pages inactive','Pages speculative']:
        if k in line:
            pages[k]=int(line.split(':')[1].strip().rstrip('.'))
pg=16384
avail=(pages.get('Pages free',0)+pages.get('Pages inactive',0)+pages.get('Pages speculative',0))*pg/1024**3
print(f'{avail:.0f}')
" 2>/dev/null || echo "0")

NEEDED_GB=$(python3 -c "print(int(${SMELT_N} * 43 * 13.4 / 1024) + 5)")
if (( AVAIL_GB < NEEDED_GB )); then
    echo -e "${YELLOW}⚠  Only ${AVAIL_GB}GB available, SMELT N=${SMELT_N} needs ~${NEEDED_GB}GB${NC}"
    echo "   Consider: NATIVE_SMELT_N=20 bash scripts/native_bench.sh"
fi

# Cleanup on exit
cleanup() { pkill -f "dmlx serve.*${PORT}" 2>/dev/null; sleep 1; }
trap cleanup EXIT
cleanup

# Start server
LOG=$(mktemp -t native_bench_XXXXX.log)
echo "▶ Starting server (SMELT N=${SMELT_N}, loading ${NEEDED_GB}GB to RAM)..."
NATIVE_SMELT_N="${SMELT_N}" "${CLI}" serve \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --native \
    --expert-packed-dir "${PACKED_DIR}" \
    > "${LOG}" 2>&1 &
SRV_PID=$!

# Wait for ready (up to 180s for SMELT preload)
echo -n "  waiting for server"
for i in $(seq 1 180); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo " ready (${i}s)"
        break
    fi
    echo -n "."
    sleep 1
done
if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo -e "\n${RED}✗ server failed to start${NC}"
    tail -20 "${LOG}"
    exit 1
fi

# ── Warmup request (GPU cache cold → hot) ──────────────────────────
echo ""
echo "▷ Warmup request (GPU cache cold → hot)..."
curl -s --max-time 120 "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"d","messages":[{"role":"user","content":"Hi"}],"max_tokens":3,"temperature":0}' \
    > /dev/null 2>&1 || true

# ── Correctness test ───────────────────────────────────────────────
echo "▷ Correctness check (Paris)..."
PARIS_RESP=$(curl -s --max-time 60 "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"d","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":5,"temperature":0}')
PARIS_TEXT=$(echo "${PARIS_RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
if echo "${PARIS_TEXT}" | grep -qi "paris"; then
    echo -e "  ${GREEN}✓ Paris correct: \"${PARIS_TEXT}\"${NC}"
    CORRECT=1
else
    echo -e "  ${RED}✗ Paris FAILED: \"${PARIS_TEXT}\"${NC}"
    CORRECT=0
fi

# ── Performance measurement (3 runs, GPU hot) ──────────────────────
echo ""
echo "▷ Performance (3 runs, GPU hot, max_tokens=5, prompt='Hi')..."
MAX_TOKENS=5
TPS_VALUES=()
for i in 1 2 3; do
    T0=$(python3 -c "import time; print(time.time())")
    RESP=$(curl -s --max-time 60 "http://localhost:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"d\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":${MAX_TOKENS},\"temperature\":0}")
    T1=$(python3 -c "import time; print(time.time())")
    ELAPSED=$(python3 -c "print(f'{$T1 - $T0:.2f}')")
    TEXT=$(echo "${RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "ERROR")
    TPS=$(python3 -c "print(f'{${MAX_TOKENS}/($T1-$T0):.3f}')")
    echo "  run ${i}: ${ELAPSED}s → ${TPS} tok/s  \"${TEXT}\""
    TPS_VALUES+=("${TPS}")
done

# Median of 3
MEDIAN_TPS=$(python3 -c "
vals = sorted([${TPS_VALUES[0]}, ${TPS_VALUES[1]}, ${TPS_VALUES[2]}])
print(f'{vals[1]:.3f}')
")

echo ""
echo "════════════════════════════════════════════════"
echo "  Median tok/s:  ${MEDIAN_TPS}"
echo "  Threshold:      ${MIN_TPS}"

# Final verdict
PASS=1
if [[ "${CORRECT}" -eq 0 ]]; then
    echo -e "  ${RED}✗ CORRECTNESS FAIL${NC}"
    PASS=0
fi

TPS_OK=$(python3 -c "print('1' if float('${MEDIAN_TPS}') >= float('${MIN_TPS}') else '0')")
if [[ "${TPS_OK}" == "1" ]]; then
    echo -e "  ${GREEN}✓ PERFORMANCE PASS (${MEDIAN_TPS} >= ${MIN_TPS} tok/s)${NC}"
else
    echo -e "  ${RED}✗ PERFORMANCE FAIL (${MEDIAN_TPS} < ${MIN_TPS} tok/s)${NC}"
    PASS=0
fi

echo "════════════════════════════════════════════════"

rm -f "${LOG}"
[[ "${PASS}" -eq 1 ]] && exit 0 || exit 1
