#!/bin/bash
# One shot: run a fixed-depth MTP request with DMLX_VERIFY_PROF=1 and dump the
# GDN verify section split. Barriers inflate the round by ~37% at M=3, so read
# the SHARES, not the absolute ms (plan doc §21.3).
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
CLI="$(cd "${DIR}/.." && pwd)/zig-out/bin/dmlx"
MODEL="${MODEL:-${HOME}/models/Qwen3.8-27B-4bit}"
PORT="${PORT:-18098}"
N="${N:-2}"
LOG="${LOG:-/tmp/vprof.log}"
pkill -f "dmlx serve" 2>/dev/null; sleep 2
env ${EXTRA_ENV:-} DMLX_VERIFY_PROF=1 DMLX_MTP_FIXED_N="$N" \
    "$CLI" serve --model "$MODEL" --speculative-mtp "$N" --port "$PORT" > "$LOG" 2>&1 &
PID=$!
for _ in $(seq 1 180); do
    curl -sf --max-time 2 "http://localhost:${PORT}/health" >/dev/null && break
    sleep 1
done
curl -sf --max-time 900 "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function that merges two sorted lists.\"}],\"max_tokens\":${MAXTOK:-80},\"temperature\":0,\"stream\":false}" >/dev/null
sleep 2
kill "$PID" 2>/dev/null; sleep 1
pkill -f "dmlx serve" 2>/dev/null
grep -A8 "GDN VERIFY sections" "$LOG" | tail -12
