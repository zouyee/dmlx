#!/bin/bash
# Supervisor: wait for the HF snapshot, then AB/BA, then the topk diagnostic.
set -uo pipefail
OUT="$(cd "$(dirname "$0")" && pwd)"
MODEL_GLOB="${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/*/"
for _ in $(seq 1 720); do
    # shellcheck disable=SC2086
    for d in $MODEL_GLOB; do
        [ -f "${d}config.json" ] && [ -f "${d}model-00003-of-00003.safetensors" ] && break 2
    done
    sleep 10
done
if ! ls $MODEL_GLOB/config.json >/dev/null 2>&1; then
    echo "model never became ready" >&2; exit 1
fi
echo "model ready: $(echo $MODEL_GLOB)"
bash "$OUT/run_abba.sh"
bash "$OUT/run_topk.sh"
