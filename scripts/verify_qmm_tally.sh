#!/bin/bash
# Dump which projection shapes actually reach the verify-shape kernels.
#
# Needed because the benchmarked speedup is not what production delivers: an
# A/B against DMLX_NO_VERIFY_QMM=1 measured -2.8 ms at M=4 and -19.1 ms at
# M=5, against bench predictions of -16.6 / -27.7 (plan doc §28). Every call
# site routes through maybeQmm, so the loss is in the eligibility predicates —
# this names the shapes that fall through instead of guessing.
#
# Rows flagged "<- MISS" are M>=4 shapes still on stock.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
CLI="$(cd "${DIR}/.." && pwd)/zig-out/bin/dmlx"
MODEL="${MODEL:-${HOME}/models/Qwen3.8-27B-4bit}"
PORT="${PORT:-18113}"
N="${N:-4}"
LOG=/tmp/verify_qmm_tally.log
pkill -f "dmlx serve" 2>/dev/null; pkill -f omlx 2>/dev/null; sleep 2
env DMLX_VERIFY_QMM_TALLY=1 DMLX_MTP_FIXED_N="$N" ${EXTRA_ENV:-} \
    "$CLI" serve --model "$MODEL" --speculative-mtp "$N" --port "$PORT" \
    > "$LOG" 2>&1 &
PID=$!
for _ in $(seq 1 240); do
    curl -sf --max-time 2 "http://localhost:${PORT}/health" >/dev/null && break
    sleep 1
done
curl -sf --max-time 900 "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"m","messages":[{"role":"user","content":"Write a Python function that merges two sorted lists."}],"max_tokens":24,"temperature":0,"stream":false}' \
    > /dev/null
sleep 2
kill "$PID" 2>/dev/null; sleep 1
pkill -f "dmlx serve" 2>/dev/null
echo "=== N=${N} (verify M=$((N+1))) ==="
sed -n '/verify_qmm routing tally/,/^$/p' "$LOG" | head -60
