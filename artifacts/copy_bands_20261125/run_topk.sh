#!/bin/bash
# DMLX_MTP_TOPK headroom diagnostic: greedy, 2 reps per cell, one server.
# Server log's "MTP topk:" lines carry per-depth top1/top2/top4/top8.
set -uo pipefail
WT="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$WT/zig-out/bin/dmlx"
MODEL="$(echo ${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/*/)"
DF2_DIR=$(echo ${HOME}/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/*/)
OUT="$(cd "$(dirname "$0")" && pwd)"
PORT=18413

if [ ! -f "${MODEL}config.json" ]; then
    echo "model snapshot not ready: $MODEL" >&2; exit 1
fi

pkill -f "dmlx serve" 2>/dev/null; sleep 2
env DMLX_QWEN38_DFLASH2_DIR="$DF2_DIR" DMLX_MTP_TOPK=1 \
    "$CLI" serve --model "$MODEL" --speculative-mtp 7 \
    --system-mem-guard 0 --temperature 0 --port "$PORT" \
    > "$OUT/server_topk.log" 2>&1 &
pid=$!
for _ in $(seq 1 300); do
    curl -sf --max-time 2 "http://localhost:${PORT}/health" >/dev/null && break
    sleep 1
done
python3 "$OUT/copy_bands_client.py" "$PORT" topk 2 0.0 > "$OUT/client_topk.log" 2>&1
kill "$pid" 2>/dev/null
pkill -f "dmlx serve" 2>/dev/null || true
echo "done"
