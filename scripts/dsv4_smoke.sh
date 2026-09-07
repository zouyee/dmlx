#!/bin/bash
# DeepSeek-V4-Flash-4bit smoke test — correctness gate.
#
# Starts one dmlx server, sends 2 deterministic prompts, and asserts the
# expected answers. This is the mandatory correctness gate before benchmark
# or enabling a new Metal/native segment.
#
# Usage:
#   bash scripts/dsv4_smoke.sh                 # native MLX-free engine (DEFAULT)
#   NATIVE=0 bash scripts/dsv4_smoke.sh        # pure MLX path (serial oracle)
#   METAL_MOE=1 NATIVE=0 bash scripts/dsv4_smoke.sh
#   DSV4_DUMP_DIR=/tmp/native_ref bash scripts/dsv4_smoke.sh
#
# Env:
#   MODEL_PATH   (default: ~/models/DeepSeek-V4-Flash-4bit)
#   PORT         (default: 8930)
#   NATIVE       (default: 1)
#   METAL_MOE    (default: unset; only used when NATIVE=0)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
PACKED_DIR="${MODEL_PATH}/packed_experts"
CLI="${PROJECT_DIR}/zig-out/bin/dmlx"
PORT="${PORT:-8930}"
MAX_TOKENS=30
SERVER_PID=""

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

cleanup() {
    if [[ -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
    # The workspace invariant is stronger than port-local cleanup: no dmlx
    # serve process may overlap a subsequent native/MLX oracle run.
    pkill -f "dmlx serve" 2>/dev/null || true
    sleep 1
}
trap cleanup EXIT INT TERM

if [[ ! -x "${CLI}" ]]; then
    echo -e "${RED}✗ binary not found: ${CLI}${NC}  (run: zig build -Doptimize=ReleaseFast)"
    exit 1
fi

EXTRA_FLAGS=()
if [[ "${NATIVE:-1}" == "1" ]]; then
    if [[ ! -d "${PACKED_DIR}" ]]; then
        echo -e "${RED}✗ packed experts not found: ${PACKED_DIR}${NC}"
        exit 1
    fi
    EXTRA_FLAGS+=(--native --expert-packed-dir "${PACKED_DIR}")
    echo -e "${YELLOW}mode: native (MLX-free) — default${NC}"
elif [[ "${METAL_MOE:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--metal-moe)
    echo -e "${YELLOW}mode: metal-moe${NC}"
else
    echo -e "${YELLOW}mode: pure MLX (serial oracle)${NC}"
fi

LOG="$(mktemp -t dsv4_smoke.XXXXXX.log)"
echo "server log: ${LOG}"
echo "max tokens: ${MAX_TOKENS}"
echo "batch flags: prefill=${NATIVE_BATCH_PREFILL:-0} mla=${NATIVE_BATCH_MLA:-0} buckets=${NATIVE_EXPERT_BUCKETS:-0} expert_kernels=${NATIVE_BATCH_EXPERT_KERNELS:-0} dspark_verify=${NATIVE_DSPARK_BATCH_VERIFY:-0}"

# Mandatory single-server guard before every start.
pkill -f "dmlx serve" 2>/dev/null || true
sleep 1

if [[ "${NATIVE:-1}" == "1" ]]; then
    # Run under a pseudo-TTY so native stderr keeps its Metal cleanup/timing
    # behavior while both stdout and stderr are retained as mandatory evidence.
    NATIVE_TIME_LAYERS=1 script -q "${LOG}" \
        "${CLI}" serve \
        --model "${MODEL_PATH}" \
        --port "${PORT}" --max-tokens "${MAX_TOKENS}" --temperature 0 \
        ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} &
else
    "${CLI}" serve \
        --model "${MODEL_PATH}" \
        --port "${PORT}" --max-tokens "${MAX_TOKENS}" --temperature 0 \
        --smelt --smelt-strategy stream --smelt-experts 0.20 --smelt-cache 0 \
        --expert-packed-dir "${PACKED_DIR}" \
        ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
        > "${LOG}" 2>&1 &
fi
SERVER_PID=$!

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
sleep 5

run_case() {
    local name="$1" prompt="$2" expected="$3"
    local resp text
    resp="$(curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":${MAX_TOKENS},\"temperature\":0}")"
    text="$(echo "${resp}" | python3 -c "import sys,json
try:
    d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])
except Exception as e:
    print('<parse-error: %s>' % e)" 2>/dev/null)"

    local lc; lc="$(echo "${text}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${lc}" == *"${expected}"* ]]; then
        echo -e "${GREEN}✓${NC} ${name}: $(printf '%q' "${text}")"
        return 0
    fi
    echo -e "${RED}✗${NC} ${name}: $(printf '%q' "${text}")  (expected to contain '${expected}')"
    return 1
}

FAIL=0
curl -s "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0}' \
    > /dev/null

run_case "capital-of-france" "The capital of France is" "paris" || FAIL=1
run_case "two-plus-two"       "2+2="                    "4"     || FAIL=1

echo ""
if [[ "${FAIL}" == "0" ]]; then
    if [[ "${NATIVE:-1}" == "1" ]]; then
        sleep 1
        echo "native batch/chunk evidence:"
        FLAGS_EVIDENCE="$(grep -E "native_engine: batch flags" "${LOG}" | tail -1 || true)"
        CHUNK_EVIDENCE="$(grep -E "native_engine: prefill chunk" "${LOG}" | tail -20 || true)"
        COMPLETE_EVIDENCE="$(grep -E "native_engine: prefill complete" "${LOG}" | tail -3 || true)"
        EVIDENCE="${FLAGS_EVIDENCE}"$'\n'"${CHUNK_EVIDENCE}"$'\n'"${COMPLETE_EVIDENCE}"
        printf '%s\n' "${EVIDENCE}"
        if [[ -z "${FLAGS_EVIDENCE}" ||
              -z "${CHUNK_EVIDENCE}" ||
              -z "${COMPLETE_EVIDENCE}" ]]; then
            echo -e "${RED}✗ required native batch/chunk evidence is missing${NC}"
            FAIL=1
        fi
        if [[ "${NATIVE_BATCH_PREFILL:-0}" == "1" &&
              "${NATIVE_BATCH_MLA:-0}" == "1" &&
              "${NATIVE_EXPERT_BUCKETS:-0}" == "1" &&
              "${NATIVE_BATCH_EXPERT_KERNELS:-0}" == "1" ]]; then
            if [[ "${EVIDENCE}" != *"batch=true"* ||
                  ! "${EVIDENCE}" =~ batched_chunks=[1-9][0-9]* ]]; then
                echo -e "${RED}✗ enabled v2 smoke did not retain batched chunk evidence${NC}"
                FAIL=1
            fi
        fi
    fi
    if [[ "${FAIL}" == "0" ]]; then
        echo -e "${GREEN}SMOKE PASS${NC}"
        exit 0
    fi
fi

echo -e "${RED}SMOKE FAIL${NC} — do NOT benchmark / do NOT enable the next native segment"
exit 1
