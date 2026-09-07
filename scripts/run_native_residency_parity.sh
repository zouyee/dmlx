#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${DIR}/.." && pwd)"
CLI="${PROJECT_DIR}/zig-out/bin/dmlx"
MODEL_PATH="${1:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
PACKED_DIR="${MODEL_PATH}/packed_experts"
STATS_FILE="${PACKED_DIR}/.smelt_routing_stats.bin"
PORT="${RESIDENCY_PARITY_PORT:-8930}"
SERVER_URL="http://127.0.0.1:${PORT}"

# Quiesce every possible routing-stats writer before taking the backup.
pkill -f "dmlx serve" 2>/dev/null || true
sleep 1
if pgrep -f "dmlx serve" >/dev/null 2>&1; then
    echo "FAIL: unable to quiesce existing dmlx serve before stats backup" >&2
    exit 1
fi

TMP_DIR="$(mktemp -d)"
BACKUP_FILE="${TMP_DIR}/routing-stats.backup"
HAD_STATS=0
SERVER_PID=""

if [[ -f "$STATS_FILE" ]]; then
    cp -p "$STATS_FILE" "$BACKUP_FILE"
    HAD_STATS=1
fi

stop_server() {
    if curl -sf --max-time 3 -X POST "${SERVER_URL}/shutdown" >/dev/null 2>&1; then
        for _ in $(seq 1 30); do
            [[ -z "$SERVER_PID" ]] || ! kill -0 "$SERVER_PID" 2>/dev/null || { sleep 1; continue; }
            break
        done
    fi
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
    pkill -f "dmlx serve" 2>/dev/null || true
    sleep 1
    ! pgrep -f "dmlx serve" >/dev/null 2>&1
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    stop_server || rc=1
    if (( HAD_STATS )); then
        cp -p "$BACKUP_FILE" "$STATS_FILE" || rc=1
        cmp -s "$BACKUP_FILE" "$STATS_FILE" || rc=1
    else
        rm -f "$STATS_FILE" || rc=1
    fi
    rm -rf "$TMP_DIR"
    exit "$rc"
}
trap cleanup EXIT INT TERM

start_server() {
    local log_file="$1"
    pkill -f "dmlx serve" 2>/dev/null || true
    sleep 1
    if pgrep -f "dmlx serve" >/dev/null 2>&1; then
        echo "FAIL: residual dmlx serve before start" >&2
        return 1
    fi
    env -u NATIVE_TELEMETRY \
        NATIVE_SMELT_N=20 NATIVE_EXPERIMENTAL_CACHE=0 \
        NATIVE_PREFETCH=0 NATIVE_SMELT_PREFETCH=0 \
        NATIVE_BATCH_PREFILL=0 NATIVE_BATCH_MLA=0 \
        NATIVE_EXPERT_BUCKETS=0 NATIVE_BATCH_EXPERT_KERNELS=0 \
        NATIVE_DSPARK_BATCH_VERIFY=0 DSPARK_DIR= \
        "$CLI" serve --model "$MODEL_PATH" \
        --expert-packed-dir "$PACKED_DIR" --native --port "$PORT" \
        --max-tokens 30 --temperature 0 >"$log_file" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 180); do
        if curl -sf "${SERVER_URL}/health" >/dev/null 2>&1; then return 0; fi
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -40 "$log_file" >&2; return 1; }
        sleep 1
    done
    echo "FAIL: server readiness timeout" >&2
    tail -40 "$log_file" >&2
    return 1
}

capture_requests() {
    local output="$1"
    : >"$output"
    python3 - "$SERVER_URL" "$output" <<'PY'
import json, sys, urllib.request
url, output = sys.argv[1], sys.argv[2]
prompts = [
    "The capital of France is",
    "2+2=",
    "Is the Earth round? Reply with only yes or no.",
]
with open(output, "w", encoding="utf-8") as f:
    for prompt in prompts:
        body = json.dumps({"model":"default","messages":[{"role":"user","content":prompt}],"max_tokens":30,"temperature":0}).encode()
        req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            doc = json.load(response)
        choice = doc["choices"][0]
        record = {"prompt":prompt,"content":choice["message"]["content"],"finish_reason":choice["finish_reason"],"completion_tokens":doc["usage"]["completion_tokens"]}
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",",":")) + "\n")
PY
}

[[ -x "$CLI" ]] || { echo "FAIL: missing ReleaseFast binary: $CLI" >&2; exit 1; }
[[ -d "$PACKED_DIR" ]] || { echo "FAIL: missing packed experts: $PACKED_DIR" >&2; exit 1; }
BINARY_SHA="$(shasum -a 256 "$CLI" | awk '{print $1}')"
rm -f "$STATS_FILE"

FRESH_LOG="${TMP_DIR}/fresh.log"
STATS_LOG="${TMP_DIR}/stats.log"
FRESH_JSONL="${TMP_DIR}/fresh.jsonl"
STATS_JSONL="${TMP_DIR}/stats.jsonl"

echo "Residency parity: fresh native N=20"
start_server "$FRESH_LOG"
grep -q "no saved stats" "$FRESH_LOG"
grep -q "Phase 1" "$FRESH_LOG"
capture_requests "$FRESH_JSONL"
stop_server
[[ -s "$STATS_FILE" ]] || { echo "FAIL: graceful fresh shutdown did not create routing stats" >&2; exit 1; }
GENERATED_STATS_SHA="$(shasum -a 256 "$STATS_FILE" | awk '{print $1}')"
[[ "$(shasum -a 256 "$CLI" | awk '{print $1}')" == "$BINARY_SHA" ]]

echo "Residency parity: stats-backed native N=20"
start_server "$STATS_LOG"
grep -q "load_stats: loaded" "$STATS_LOG"
grep -q "Phase 2" "$STATS_LOG"
capture_requests "$STATS_JSONL"
stop_server
[[ "$(shasum -a 256 "$CLI" | awk '{print $1}')" == "$BINARY_SHA" ]]

python3 - "$FRESH_JSONL" "$STATS_JSONL" <<'PY'
import json, sys
from pathlib import Path
def load(path): return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
fresh, backed = load(sys.argv[1]), load(sys.argv[2])
if fresh != backed: raise SystemExit("FAIL: fresh/stats-backed exact responses differ")
if len(fresh) != 3: raise SystemExit("FAIL: expected three parity prompts")
checks = ["paris" in fresh[0]["content"].lower(), "4" in fresh[1]["content"], fresh[2]["content"].strip().lower() in {"yes", "yes."}]
if not all(checks): raise SystemExit("FAIL: equal responses did not satisfy semantic gates")
print("Responses:")
for item in fresh: print("  " + json.dumps(item, ensure_ascii=False, sort_keys=True))
PY

echo "Residency parity PASS: binary=${BINARY_SHA} generated_stats=${GENERATED_STATS_SHA}"
