#!/bin/bash
# DeepSeek-V4-Flash-4bit smoke test — correctness gate.
#
# Starts the dmlx server, sends 2 well-known greedy prompts, and asserts the
# output is coherent (not gibberish / BOS repetition). This is the mandatory
# gate before any benchmark or before lighting up a new metal segment.
#
# See docs/analysis/dsv4-first-class-support-plan.md §4 and §29.
#
# Usage:
#   bash scripts/dsv4_smoke.sh                 # native MLX-free engine (DEFAULT)
#   NATIVE=0 bash scripts/dsv4_smoke.sh        # pure MLX path (oracle)
#   METAL_MOE=1 bash scripts/dsv4_smoke.sh     # add --metal-moe
#   DSV4_DUMP_DIR=/tmp/mlx_ref bash scripts/dsv4_smoke.sh   # also dump activations
#
# Env:
#   MODEL_PATH   (default: ~/models/DeepSeek-V4-Flash-4bit)
#   PORT         (default: 8930)
#   NATIVE       (default: 1 → native MLX-free engine; set to 0 for pure MLX)
#   METAL_MOE    (default: unset → ignored unless NATIVE=0)
#   DSV4_DUMP_DIR(default: unset → no dump)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
PACKED_DIR="${MODEL_PATH}/packed_experts"
CLI="${PROJECT_DIR}/zig-out/bin/dmlx"
PORT="${PORT:-8930}"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

if [[ ! -x "${CLI}" ]]; then
    echo -e "${RED}✗ binary not found: ${CLI}${NC}  (run: zig build -Doptimize=ReleaseFast)"
    exit 1
fi

EXTRA_FLAGS=()
if [[ "${NATIVE:-1}" == "1" ]]; then
    EXTRA_FLAGS+=(--native --expert-packed-dir "${PACKED_DIR}")
    echo -e "${YELLOW}mode: native (MLX-free) — default${NC}"
elif [[ "${METAL_MOE:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--metal-moe)
    echo -e "${YELLOW}mode: metal-moe${NC}"
else
    echo -e "${YELLOW}mode: pure MLX (oracle)${NC}"
fi

LOG="$(mktemp -t dsv4_smoke.XXXXXX.log)"
echo "server log: ${LOG}"

# Start server in background.
if [[ "${NATIVE:-1}" == "1" ]]; then
    # Keep stderr on the TTY (line-buffered) for NATIVE_TIME_LAYERS to work correctly.
    # NATIVE_TIME_LAYERS requires fprintf→write() to call the TTY syscall each layer,
    # which gives Metal's GCD cleanup threads the scheduling opportunity they need.
    # If stderr is redirected to a file (block-buffered), the write() is delayed and
    # the Metal @autoreleasepool crash reappears.
    NATIVE_TIME_LAYERS=1 "${CLI}" serve \
        --model "${MODEL_PATH}" \
        --port "${PORT}" --max-tokens 10 --temperature 0 \
        ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
        > "${LOG}" &
else
    "${CLI}" serve \
        --model "${MODEL_PATH}" \
        --port "${PORT}" --max-tokens 64 --temperature 0 \
        --smelt --smelt-strategy stream --smelt-experts 0.20 --smelt-cache 0 \
        --expert-packed-dir "${PACKED_DIR}" \
        ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
        > "${LOG}" 2>&1 &
fi
SERVER_PID=$!

cleanup() {
    kill "${SERVER_PID}" 2>/dev/null
    wait "${SERVER_PID}" 2>/dev/null
}
trap cleanup EXIT

# Wait for health (up to 180s for cold model load).
echo -n "waiting for server"
for _ in $(seq 1 180); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo " ready"
        break
    fi
    echo -n "."
    sleep 1
done
if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo -e "\n${RED}✗ server did not become healthy — see ${LOG}${NC}"
    tail -20 "${LOG}"
    exit 1
fi
# Extra settling time: allow Metal to complete its internal initialization
# (e.g., AttnBufCache, pipeline compilation) before the first inference request.
sleep 5

# ---- test cases: name | prompt | max_tokens | expected_substring (lowercase) ----
run_case() {
    local name="$1" prompt="$2" max_tokens="$3" expected="$4"
    local resp text
    resp="$(curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":${max_tokens},\"temperature\":0}")"
    text="$(echo "${resp}" | python3 -c "import sys,json
try:
    d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])
except Exception as e:
    print('<parse-error: %s>' % e)" 2>/dev/null)"

    local lc; lc="$(echo "${text}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${lc}" == *"${expected}"* ]]; then
        echo -e "${GREEN}✓${NC} ${name}: $(printf '%q' "${text}")"
        return 0
    else
        echo -e "${RED}✗${NC} ${name}: $(printf '%q' "${text}")  (expected to contain '${expected}')"
        return 1
    fi
}

FAIL=0
# Warmup: first request cold-starts Metal compressor state; a "Hi" request
# initialises all 43 layers so subsequent tests don't hit a cold-start crash.
# (benchmark does the same: several "Hi" warmup requests before Paris check)
curl -s "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0}' \
    > /dev/null
# Both are continuation-style prompts (the model follows instructions poorly
# but continues facts reliably). Token budgets are sized so the answer is
# actually reached before the model's restating habit kicks in.
run_case "capital-of-france" "The capital of France is" 16 "paris" || FAIL=1
run_case "two-plus-two"       "2+2="                     64 "4"     || FAIL=1

echo ""
if [[ "${FAIL}" == "0" ]]; then
    echo -e "${GREEN}SMOKE PASS${NC}"
    exit 0
else
    echo -e "${RED}SMOKE FAIL${NC} — do NOT benchmark / do NOT light up next metal segment"
    exit 1
fi
