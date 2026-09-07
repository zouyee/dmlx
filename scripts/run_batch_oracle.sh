#!/bin/bash
# V4: real-model, in-process scalar/final-logits/chunk-partition oracle.
#
# Ordinary runs are validation-only and never rewrite the golden fixture.
# Regeneration is an explicit two-step operation after reviewing the scalar run:
#   python3 scripts/compare_batch_oracle.py ... --regenerate-fixture \
#     --regeneration-reason "documented reason"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${HOME}/models/DeepSeek-V4-Flash-4bit"
PACKED_DIR="${MODEL_PATH}/packed_experts"
FIXTURE="${PROJECT_DIR}/tests/fixtures/dsv4_native_logits.json"
DEVELOPER_DIR="/Library/Developer/CommandLineTools"

# A promotion run re-execs once under tee so every V4 suite has retained,
# hashable output. pipefail preserves the inner runner's failure status.
if [[ -n "${BATCH_ORACLE_JSON:-}" && "${BATCH_ORACLE_CAPTURED:-0}" != "1" ]]; then
    BATCH_ORACLE_LOG="${BATCH_ORACLE_JSON%.json}-capture.log"
    mkdir -p "$(dirname "$BATCH_ORACLE_JSON")"
    set +e
    BATCH_ORACLE_CAPTURED=1 BATCH_ORACLE_LOG="$BATCH_ORACLE_LOG" \
        BATCH_ORACLE_JSON="$BATCH_ORACLE_JSON" bash "${BASH_SOURCE[0]}" 2>&1 | \
        tee "$BATCH_ORACLE_LOG"
    ORACLE_STATUS="${PIPESTATUS[0]}"
    set -e
    rm -f "$BATCH_ORACLE_LOG"
    exit "$ORACLE_STATUS"
fi

RESULT="$(mktemp -t dmlx_batch_oracle.XXXXXX.json)"

cleanup() {
    # V4 is in-process and does not start a server, but preserve the workspace's
    # stronger lifecycle invariant before and after every oracle run.
    pkill -f "dmlx serve" 2>/dev/null || true
    sleep 1
    rm -f "${RESULT}"
}
trap cleanup EXIT INT TERM

if [[ ! -f "${MODEL_PATH}/config.json" || ! -f "${MODEL_PATH}/tokenizer.json" ]]; then
    echo "batch oracle FAILED: fixed model is incomplete: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -d "${PACKED_DIR}" ]]; then
    echo "batch oracle FAILED: packed experts not found: ${PACKED_DIR}" >&2
    exit 1
fi
if [[ ! -f "${FIXTURE}" ]]; then
    echo "batch oracle FAILED: frozen fixture not found: ${FIXTURE}" >&2
    echo "ordinary validation never creates or updates the golden" >&2
    exit 1
fi

# Mandatory single-server guard, even though this runner uses no HTTP server.
pkill -f "dmlx serve" 2>/dev/null || true
sleep 1

printf 'batch oracle config: model=%s packed=%s NATIVE_SMELT_N=20 max_generated_tokens=0\n' \
    "${MODEL_PATH}" "${PACKED_DIR}"

python3 "${SCRIPT_DIR}/compare_batch_oracle.py" --self-test
python3 "${SCRIPT_DIR}/check_expert_io_invariants.py" --self-test

# Task 4.5 enabled pass. The in-process oracle temporarily raises the C
# engine capacity from 8 to 16, proving that the independent hard B=8 boundary
# (not the configured capacity) forces its 9-token fallback assertion.
printf 'batch oracle path: enabled (expert buckets=1 kernels=1 hard-fallback=9>8)\n'
NATIVE_SMELT_N=20 \
NATIVE_BATCH_PREFILL=0 \
NATIVE_EXPERT_BUCKETS=1 \
NATIVE_BATCH_EXPERT_KERNELS=1 \
DEVELOPER_DIR="${DEVELOPER_DIR}" \
zig build run-batch-oracle-model -Doptimize=ReleaseFast -- --output "${RESULT}"

python3 "${SCRIPT_DIR}/compare_batch_oracle.py" \
    --result "${RESULT}" \
    --fixture "${FIXTURE}" \
    --model "${MODEL_PATH}" \
    --packed-dir "${PACKED_DIR}"

# Task 5.2 production-v2 pass. Keep this independent from the legacy grouped-
# MoE pass above because Tasks 5.3-5.4 still use the legacy entry for DSpark
# verification. This pass calls forwardPlannedBatch directly and proves all
# four required flags, B=1/2/3/5/8, nonzero starts and partition invariance.
printf 'batch oracle path: production-v2 (prefill=1 mla=1 buckets=1 kernels=1)\n'
sleep 2
NATIVE_SMELT_N=20 \
NATIVE_BATCH_PREFILL=1 \
NATIVE_BATCH_MLA=1 \
NATIVE_EXPERT_BUCKETS=1 \
NATIVE_BATCH_EXPERT_KERNELS=1 \
NATIVE_DSPARK_BATCH_VERIFY=0 \
DEVELOPER_DIR="${DEVELOPER_DIR}" \
zig build run-batch-oracle-model -Doptimize=ReleaseFast -- --production-v2

# Preserve and prove the rollback path independently of caller environment.
printf 'batch oracle path: disabled (expert buckets=0 kernels=0)\n'
sleep 2
NATIVE_SMELT_N=20 \
NATIVE_BATCH_PREFILL=0 \
NATIVE_EXPERT_BUCKETS=0 \
NATIVE_BATCH_EXPERT_KERNELS=0 \
NATIVE_BATCH_MAX_PREFILL_CHUNK=8 \
DEVELOPER_DIR="${DEVELOPER_DIR}" \
zig build run-batch-oracle-model -Doptimize=ReleaseFast

# Optional Task 5.5 provenance: this point is reached only after scalar,
# production-v2 and rollback paths all exit successfully. Retain the scalar
# result and emit a source-hashed, model-fingerprinted V4 evidence document.
if [[ -n "${BATCH_ORACLE_JSON:-}" ]]; then
    ORACLE_RESULT_JSON="${BATCH_ORACLE_JSON%.json}-result.json"
    ORACLE_LOG_JSON="${BATCH_ORACLE_JSON%.json}-run.log"
    mkdir -p "$(dirname "$BATCH_ORACLE_JSON")"
    cp "$RESULT" "$ORACLE_RESULT_JSON"
    # Snapshot the active tee log before evidence generation. Subsequent
    # generator/validator messages continue only in the capture log and cannot
    # invalidate this immutable suite transcript.
    cp "$BATCH_ORACLE_LOG" "$ORACLE_LOG_JSON"
    ORACLE_COMMIT=$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo "?")
    python3 "${SCRIPT_DIR}/benchmark_report.py" \
        --oracle-evidence-output "$BATCH_ORACLE_JSON" \
        --oracle-result "$ORACLE_RESULT_JSON" \
        --oracle-fixture "$FIXTURE" \
        --oracle-model "$MODEL_PATH" \
        --oracle-packed "$PACKED_DIR" \
        --oracle-runner "$0" \
        --oracle-log "$ORACLE_LOG_JSON" \
        --oracle-workspace-revision "$ORACLE_COMMIT"
    python3 "${SCRIPT_DIR}/benchmark_report.py" \
        --validate-oracle-evidence "$BATCH_ORACLE_JSON"
fi

# No server may remain after V4.
if pgrep -f "dmlx serve" >/dev/null 2>&1; then
    echo "batch oracle FAILED: dmlx serve process remains" >&2
    exit 1
fi

echo "BATCH ORACLE PASS"
