#!/bin/bash
set -uo pipefail
WT="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$WT/zig-out/bin/dmlx"
MODEL="$HOME/models/Qwen3.8-27B-4bit"
OUT="$(cd "$(dirname "$0")" && pwd)"
PORT=18414

run_arm() {
    local tag="$1" drafter_name="$2" drafter_dir="$3"
    pkill -f "dmlx serve" 2>/dev/null; sleep 2
    env "$drafter_name=$drafter_dir" DMLX_MEM_PROBE=1 DMLX_MTP_COPIES=1 DMLX_MTP_COPIES_BANDS=1 \
        "$CLI" serve --model "$MODEL" --speculative-mtp 7 --system-mem-guard 0 \
        --temperature 0.8 --top-k 50 --top-p 1.0 --port "$PORT" \
        > "$OUT/server_memory_${tag}.log" 2>&1 &
    local pid=$!
    local ready=0
    for _ in $(seq 1 300); do
        if curl -sf --max-time 2 "http://localhost:${PORT}/health" >/dev/null; then ready=1; break; fi
        sleep 1
    done
    if [ "$ready" -ne 1 ]; then kill "$pid" 2>/dev/null; return 1; fi
    python3 "$OUT/smoke_client.py" "$PORT" 0.8 > "$OUT/client_memory_${tag}.log"
    local status=$?
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null || true
    sleep 2
    return "$status"
}

DF2_DIR="$HOME/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
DSPINE_DIR="$HOME/.cache/huggingface/hub/models--tencent--Qwen3.8-27B-DSpine-b16/snapshots/260a6c23070ca873ff1a5f33b7119ac3f6a245ee"
run_arm A DMLX_QWEN38_DFLASH2_DIR "$DF2_DIR" || exit 1
run_arm B DMLX_QWEN38_DSPINE_DIR "$DSPINE_DIR" || exit 1
pkill -f "dmlx serve" 2>/dev/null || true
echo "done"
