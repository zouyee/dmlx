#!/bin/bash
# Copy-bands AB/BA: A = DMLX_MTP_COPIES=1, B = + DMLX_MTP_COPIES_BANDS=1.
# Two passes (A,B then B,A); each run does warmup + 6 cells x 2 reps, so
# every arm/cell gets n=4 total, drift-cancelled by the alternating order.
set -uo pipefail
WT="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$WT/zig-out/bin/dmlx"
MODEL="$(echo ${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/*/)"
if [ ! -f "${MODEL}config.json" ]; then
    echo "model snapshot not ready: $MODEL" >&2; exit 1
fi
DF2_DIR=$(echo ${HOME}/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/*/)
OUT="$(cd "$(dirname "$0")" && pwd)"
PORT=18411
CLIENT="$OUT/copy_bands_client.py"

run_arm() { # tag, extra env...
    local tag="$1"; shift
    local log="$OUT/server_${tag}.log"
    pkill -f "dmlx serve" 2>/dev/null; sleep 2
    env DMLX_QWEN38_DFLASH2_DIR="$DF2_DIR" "$@" \
        "$CLI" serve --model "$MODEL" --speculative-mtp 7 \
        --system-mem-guard 0 --temperature 0.8 --top-k 50 --top-p 1.0 \
        --port "$PORT" > "$log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 300); do
        curl -sf --max-time 2 "http://localhost:${PORT}/health" >/dev/null && break
        sleep 1
    done
    python3 "$CLIENT" "$PORT" "$tag" 2 >> "$OUT/client_${tag}.log"
    kill "$pid" 2>/dev/null; sleep 2
}

: > "$OUT/client_pass1.log"
run_arm pass1-A DMLX_MTP_COPIES=1
run_arm pass1-B DMLX_MTP_COPIES=1 DMLX_MTP_COPIES_BANDS=1
run_arm pass2-B DMLX_MTP_COPIES=1 DMLX_MTP_COPIES_BANDS=1
run_arm pass2-A DMLX_MTP_COPIES=1
pkill -f "dmlx serve" 2>/dev/null || true
echo "done"
