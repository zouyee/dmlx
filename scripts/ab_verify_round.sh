#!/bin/bash
# A/B the per-MTP-round wall cost at a FIXED draft depth.
#
# Fixed depth is essential: the online depth controller changes N between
# runs, and ms/round is only comparable at matched depth (plan doc §21.2).
# Env vars listed in VARIANTS are applied to the serve process; the script
# parses `MTP round: ... ms=` lines from the server log and reports the
# median (rounds are integer-token events, so the mean is high-variance).
#
# Usage: bash scripts/ab_verify_round.sh "ENV1=1" "ENV2=1" ...
#        (an empty string "" means the baseline with no extra env)
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${DIR}/.." && pwd)"
CLI="${PROJECT_DIR}/zig-out/bin/dmlx"
MODEL="${MODEL:-${HOME}/models/Qwen3.8-27B-4bit}"
PORT="${PORT:-18097}"
N="${N:-2}"
MAXTOK="${MAXTOK:-80}"
PROMPT="${PROMPT:-Write a Python function that merges two sorted lists.}"

run_one() {
    local extra="$1" log="/tmp/ab_verify_$$.log"
    pkill -f "dmlx serve" 2>/dev/null; sleep 2
    env $extra DMLX_MTP_DEBUG=1 DMLX_MTP_FIXED_N="$N" \
        "$CLI" serve --model "$MODEL" --speculative-mtp "$N" \
        --port "$PORT" > "$log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 180); do
        curl -sf --max-time 2 "http://localhost:${PORT}/health" >/dev/null && break
        sleep 1
    done
    # Warmup then measure.
    for i in 1 2; do
      curl -sf --max-time 600 "http://localhost:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":${MAXTOK},\"temperature\":0,\"stream\":false}" \
        >/dev/null
    done
    kill "$pid" 2>/dev/null; sleep 1
    # Second request's rounds only: drop the first half of the lines.
    local stats
    # Anchor on MTP-ROUND lines. A bare 'ms=' also matches `duration_ms=` in
    # [RequestLog], which put a 9.5e7 "round" into the sample and made both
    # the mean and (with few real rounds) the median meaningless.
    stats=$(grep -ao 'MTP round: n=[0-9]* ms=[0-9.]*' "$log" | grep -o 'ms=[0-9.]*$' | sed 's/ms=//' | python3 -c "
import sys,statistics
v=[float(x) for x in sys.stdin]
v=v[len(v)//2:]
print(f'{len(v)} rounds  median={statistics.median(v):.1f} ms  mean={statistics.fmean(v):.1f}')
" 2>/dev/null)
    # Host-side GDN graph-build time per round, from the same log. Reported
    # next to the round time because the two together say whether the build is
    # real CPU work or the host blocking on the GPU — a distinction this
    # project has already got wrong once (an 80 ms "graph build" that was
    # asyncEval blocking mis-attributed).
    local hb
    hb=$(grep -ao 'verify build parts: gdn=[0-9.]*' "$log" | sed 's/.*gdn=//' | python3 -c "
import sys,statistics
v=[float(x) for x in sys.stdin]
v=v[len(v)//2:]
print(f'hostbuild median={statistics.median(v):.1f} ms' if v else 'hostbuild n/a')
" 2>/dev/null)
    echo "  [${extra:-baseline}] ${stats:-NO DATA (see $log)}  ${hb:-}"
}

echo "verify-round A/B  N=${N}  max_tokens=${MAXTOK}"
for v in "$@"; do run_one "$v"; done
pkill -f "dmlx serve" 2>/dev/null
