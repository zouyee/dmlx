#!/bin/bash
# Performance regression test for DeepSeek V4 stream mode.
#
# Measures TTFT (time-to-first-token) and ITL (inter-token latency),
# saves results to a JSON baseline file, and compares against historical
# runs to detect performance regressions.
#
# Usage:
#   ./scripts/performance_regression.sh [model_path]
#
# First run on a machine auto-saves the baseline. Subsequent runs compare
# against it. If a metric degrades beyond the regression threshold, the
# script exits with code 1.
#
# Environment:
#   MAX_TTFT_MS       — Absolute TTFT threshold in ms (default: 500)
#   MAX_ITL_MS        — Absolute ITL threshold in ms (default: 150)
#   MIN_TPS           — Minimum throughput in tokens/sec (default: 5)
#   OUTPUT_TOKENS     — Number of tokens to generate (default: 20)
#   REGRESSION_PCT    — % degradation to flag regression (default: 20)
#   BASELINE_DIR      — Where to store baseline JSON (default: scripts/baselines)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${1:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
CLI="${PROJECT_DIR}/zig-out/bin/mlx-zig"

# Thresholds
MAX_TTFT_MS="${MAX_TTFT_MS:-500}"
MAX_ITL_MS="${MAX_ITL_MS:-150}"
MIN_TPS="${MIN_TPS:-5}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-20}"
REGRESSION_PCT="${REGRESSION_PCT:-20}"
BASELINE_DIR="${BASELINE_DIR:-${SCRIPT_DIR}/baselines}"

# Smelt configuration
SMELT_FLAGS=(--smelt --smelt-strategy stream --smelt-experts 0.1)

export MLX_METAL_FAST_SYNCH=1

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Derive baseline file from hostname + model name
HOSTNAME=$(hostname -s)
MODEL_BASENAME=$(basename "${MODEL_PATH}")
BASELINE_FILE="${BASELINE_DIR}/performance_${HOSTNAME}_${MODEL_BASENAME}.json"

# ------------------------------------------------------------------
# Pre-flight
# ------------------------------------------------------------------
echo "═══════════════════════════════════════════════════════════════"
echo "⚡ DeepSeek V4 Performance Regression Test"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Model path:        ${MODEL_PATH}"
echo "Baseline file:     ${BASELINE_FILE}"
echo "Absolute thresholds:"
echo "   TTFT ≤ ${MAX_TTFT_MS} ms | ITL ≤ ${MAX_ITL_MS} ms | TPS ≥ ${MIN_TPS}"
echo "Regression threshold: +${REGRESSION_PCT}% vs baseline"
echo ""

if [ ! -d "${MODEL_PATH}" ]; then
    echo -e "${RED}❌ Model not found at: ${MODEL_PATH}${NC}"
    exit 1
fi

mkdir -p "${BASELINE_DIR}"

if [ ! -f "${CLI}" ]; then
    echo "📦 Building mlx-zig..."
    (cd "${PROJECT_DIR}" && zig build -Doptimize=ReleaseFast) || {
        echo -e "${RED}❌ Build failed${NC}"
        exit 1
    }
fi

# ------------------------------------------------------------------
# Run benchmark
# ------------------------------------------------------------------
echo "🏃 Running benchmark (prompt='2+2=', tokens=${OUTPUT_TOKENS})..."
echo ""

OUTPUT=$("${CLI}" chat \
    --model "${MODEL_PATH}" \
    --prompt "2+2=" \
    --max-tokens "${OUTPUT_TOKENS}" \
    --temperature 0 \
    "${SMELT_FLAGS[@]}" 2>&1) || {
    echo -e "${RED}❌ Benchmark run crashed${NC}"
    echo "${OUTPUT}"
    exit 1
}

# Parse token step timings
STEP_TIMES=$(echo "${OUTPUT}" | grep "Token step .* complete:" | sed -E 's/.*complete: ([0-9.]+)ms.*/\1/')

if [ -z "${STEP_TIMES}" ]; then
    echo -e "${RED}❌ Could not parse timing data from output${NC}"
    echo "${OUTPUT}"
    exit 1
fi

# TTFT = first step time (prefill)
TTFT_MS=$(echo "${STEP_TIMES}" | head -1)

# ITL = mean of decode steps (steps 2..N)
DECODE_TIMES=$(echo "${STEP_TIMES}" | tail -n +2)
N_DECODE=$(echo "${DECODE_TIMES}" | wc -l | tr -d ' ')

if [ "${N_DECODE}" -gt 0 ]; then
    ITL_MS=$(echo "${DECODE_TIMES}" | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
    TPS=$(echo "${ITL_MS}" | awk '{printf "%.1f", 1000/$1}')
else
    ITL_MS="N/A"
    TPS="N/A"
fi

# Gather git info
GIT_COMMIT=$(git -C "${PROJECT_DIR}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git -C "${PROJECT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# ------------------------------------------------------------------
# Load or create baseline
# ------------------------------------------------------------------
BASELINE_EXISTS=false
if [ -f "${BASELINE_FILE}" ]; then
    BASELINE_EXISTS=true
    BASELINE_TTFT=$(jq -r '.ttft_ms // "null"' "${BASELINE_FILE}")
    BASELINE_ITL=$(jq -r '.itl_ms // "null"' "${BASELINE_FILE}")
    BASELINE_TPS=$(jq -r '.tps // "null"' "${BASELINE_FILE}")
    BASELINE_COMMIT=$(jq -r '.git_commit // "unknown"' "${BASELINE_FILE}")
    BASELINE_DATE=$(jq -r '.timestamp // "unknown"' "${BASELINE_FILE}")
fi

# ------------------------------------------------------------------
# Build current result JSON
# ------------------------------------------------------------------
CURRENT_JSON=$(cat <<EOF
{
  "hostname": "${HOSTNAME}",
  "model": "${MODEL_BASENAME}",
  "git_commit": "${GIT_COMMIT}",
  "git_branch": "${GIT_BRANCH}",
  "timestamp": "${TIMESTAMP}",
  "ttft_ms": ${TTFT_MS},
  "itl_ms": ${ITL_MS},
  "tps": ${TPS},
  "output_tokens": ${OUTPUT_TOKENS},
  "thresholds": {
    "max_ttft_ms": ${MAX_TTFT_MS},
    "max_itl_ms": ${MAX_ITL_MS},
    "min_tps": ${MIN_TPS},
    "regression_pct": ${REGRESSION_PCT}
  }
}
EOF
)

# ------------------------------------------------------------------
# Display results with baseline comparison
# ------------------------------------------------------------------
echo ""
echo "📊 Results:"
printf "   %-8s %10s" "Metric" "Current"
if [ "${BASELINE_EXISTS}" = true ]; then
    printf " %10s %10s %10s\n" "Baseline" "Δ" "Status"
else
    printf " %10s\n" "Status"
fi

# Helper: compare value against baseline
# $1=label $2=current $3=baseline $4=lower_is_better(true/false)
compare_metric() {
    local label="$1"
    local current="$2"
    local baseline="$3"
    local lower_is_better="$4"

    if [ "${BASELINE_EXISTS}" = false ]; then
        printf "   %-8s %10s %10s\n" "${label}" "${current}" "(new)"
        return
    fi

    if [ "${baseline}" = "null" ] || [ "${baseline}" = "N/A" ]; then
        printf "   %-8s %10s %10s %10s %10s\n" "${label}" "${current}" "N/A" "—" "(new)"
        return
    fi

    local delta_pct
    delta_pct=$(echo "scale=1; (${current} - ${baseline}) / ${baseline} * 100" | bc -l)
    local delta_str
    if [ "${lower_is_better}" = true ]; then
        # For latency: positive delta is bad
        local abs_pct
        abs_pct=$(echo "${delta_pct}" | sed 's/^-//')
        if [ "${delta_pct%.*}" -gt "${REGRESSION_PCT}" ] 2>/dev/null; then
            delta_str="${RED}+${delta_pct}%${NC}"
        elif [ "${delta_pct%.*}" -lt "-$((REGRESSION_PCT / 2))" ] 2>/dev/null; then
            delta_str="${GREEN}${delta_pct}%${NC}"
        else
            delta_str="${delta_pct}%"
        fi
    else
        # For throughput: positive delta is good
        if [ "${delta_pct%.*}" -lt "-${REGRESSION_PCT}" ] 2>/dev/null; then
            delta_str="${RED}${delta_pct}%${NC}"
        elif [ "${delta_pct%.*}" -gt "$((REGRESSION_PCT / 2))" ] 2>/dev/null; then
            delta_str="${GREEN}+${delta_pct}%${NC}"
        else
            delta_str="${delta_pct}%"
        fi
    fi

    printf "   %-8s %10s %10s %10b %10s\n" "${label}" "${current}" "${baseline}" "${delta_str}" ""
}

compare_metric "TTFT" "${TTFT_MS}" "${BASELINE_TTFT:-null}" true
if [ "${ITL_MS}" != "N/A" ]; then
    compare_metric "ITL" "${ITL_MS}" "${BASELINE_ITL:-null}" true
    compare_metric "TPS" "${TPS}" "${BASELINE_TPS:-null}" false
fi

# ------------------------------------------------------------------
# Absolute threshold checks
# ------------------------------------------------------------------
echo ""
ABS_FAILED=0

TTFT_OK=$(echo "${TTFT_MS} <= ${MAX_TTFT_MS}" | bc -l)
if [ "${TTFT_OK}" -eq 1 ]; then
    echo -e "${GREEN}✅ TTFT within absolute threshold${NC}"
else
    echo -e "${RED}❌ TTFT exceeds absolute threshold: ${TTFT_MS} ms > ${MAX_TTFT_MS} ms${NC}"
    ABS_FAILED=1
fi

if [ "${ITL_MS}" != "N/A" ]; then
    ITL_OK=$(echo "${ITL_MS} <= ${MAX_ITL_MS}" | bc -l)
    if [ "${ITL_OK}" -eq 1 ]; then
        echo -e "${GREEN}✅ ITL within absolute threshold${NC}"
    else
        echo -e "${RED}❌ ITL exceeds absolute threshold: ${ITL_MS} ms > ${MAX_ITL_MS} ms${NC}"
        ABS_FAILED=1
    fi

    TPS_OK=$(echo "${TPS} >= ${MIN_TPS}" | bc -l)
    if [ "${TPS_OK}" -eq 1 ]; then
        echo -e "${GREEN}✅ Throughput above minimum${NC}"
    else
        echo -e "${RED}❌ Throughput below minimum: ${TPS} < ${MIN_TPS}${NC}"
        ABS_FAILED=1
    fi
fi

# ------------------------------------------------------------------
# Regression checks vs baseline
# ------------------------------------------------------------------
REG_FAILED=0

if [ "${BASELINE_EXISTS}" = true ]; then
    echo ""
    echo "🔍 Regression analysis (baseline from ${BASELINE_COMMIT} @ ${BASELINE_DATE}):"

    if [ "${BASELINE_TTFT}" != "null" ] && [ "${BASELINE_TTFT}" != "N/A" ]; then
        TTFT_REG=$(echo "scale=1; (${TTFT_MS} - ${BASELINE_TTFT}) / ${BASELINE_TTFT} * 100" | bc -l)
        TTFT_REG_INT=$(echo "${TTFT_REG}" | sed 's/^-//' | cut -d. -f1)
        if [ "${TTFT_REG_INT}" -gt "${REGRESSION_PCT}" ]; then
            echo -e "${RED}   ❌ TTFT regression: +${TTFT_REG}% vs baseline${NC}"
            REG_FAILED=1
        else
            echo -e "${GREEN}   ✅ TTFT no regression: ${TTFT_REG}% vs baseline${NC}"
        fi
    fi

    if [ "${ITL_MS}" != "N/A" ] && [ "${BASELINE_ITL}" != "null" ] && [ "${BASELINE_ITL}" != "N/A" ]; then
        ITL_REG=$(echo "scale=1; (${ITL_MS} - ${BASELINE_ITL}) / ${BASELINE_ITL} * 100" | bc -l)
        ITL_REG_INT=$(echo "${ITL_REG}" | sed 's/^-//' | cut -d. -f1)
        if [ "${ITL_REG_INT}" -gt "${REGRESSION_PCT}" ]; then
            echo -e "${RED}   ❌ ITL regression: +${ITL_REG}% vs baseline${NC}"
            REG_FAILED=1
        else
            echo -e "${GREEN}   ✅ ITL no regression: ${ITL_REG}% vs baseline${NC}"
        fi
    fi

    if [ "${TPS}" != "N/A" ] && [ "${BASELINE_TPS}" != "null" ] && [ "${BASELINE_TPS}" != "N/A" ]; then
        TPS_REG=$(echo "scale=1; (${BASELINE_TPS} - ${TPS}) / ${BASELINE_TPS} * 100" | bc -l)
        TPS_REG_INT=$(echo "${TPS_REG}" | sed 's/^-//' | cut -d. -f1)
        if [ "${TPS_REG_INT}" -gt "${REGRESSION_PCT}" ]; then
            echo -e "${RED}   ❌ TPS regression: -${TPS_REG}% vs baseline${NC}"
            REG_FAILED=1
        else
            echo -e "${GREEN}   ✅ TPS no regression: ${TPS_REG}% vs baseline${NC}"
        fi
    fi
fi

# ------------------------------------------------------------------
# Save baseline
# ------------------------------------------------------------------
echo ""
if [ "${BASELINE_EXISTS}" = false ]; then
    echo "💾 No baseline found. Saving current results as baseline..."
    echo "${CURRENT_JSON}" | jq . > "${BASELINE_FILE}"
    echo -e "${BLUE}   Baseline saved to: ${BASELINE_FILE}${NC}"
else
    echo "💾 Updating baseline with latest results..."
    echo "${CURRENT_JSON}" | jq . > "${BASELINE_FILE}"
    echo -e "${BLUE}   Baseline updated: ${BASELINE_FILE}${NC}"
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "═══════════════════════════════════════════════════════════════"
if [ ${ABS_FAILED} -eq 0 ] && [ ${REG_FAILED} -eq 0 ]; then
    echo -e "${GREEN}✅ All performance metrics OK${NC}"
    echo "═══════════════════════════════════════════════════════════════"
    exit 0
else
    if [ ${ABS_FAILED} -ne 0 ]; then
        echo -e "${RED}❌ Absolute threshold violations detected${NC}"
    fi
    if [ ${REG_FAILED} -ne 0 ]; then
        echo -e "${RED}❌ Performance regression detected vs baseline${NC}"
    fi
    echo "═══════════════════════════════════════════════════════════════"
    exit 1
fi
