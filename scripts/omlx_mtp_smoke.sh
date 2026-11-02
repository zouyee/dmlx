#!/bin/bash
# Verify omlx's NATIVE MTP actually activates on the grafted checkpoint built
# by scripts/omlx_enable_native_mtp.py, and report its decode rate.
#
# omlx auto-writes `mtp_enabled: true` into model_settings.json the first time
# it discovers a model for which `lightning_mtp_supported()` is true, so no
# flag is needed — but that must be CONFIRMED in the log, not assumed. The
# telltale line is "MTP path activated for uid=..." from
# omlx/patches/mlx_lm_mtp/batch_generator.py.
set -uo pipefail
OMLX_PY="${OMLX_PY:-$HOME/venvs/dmlx-omlx/bin/python}"
MODEL_DIR="${MODEL_DIR:-${HOME}/models}"
MODEL_ID="${MODEL_ID:-Qwen3.8-27B-4bit-mtp}"
PORT="${PORT:-18110}"
LOG=/tmp/omlx_mtp_smoke.log

pkill -f "dmlx serve" 2>/dev/null
pkill -f "omlx" 2>/dev/null
sleep 2
OMLX_HEADLESS=1 "$OMLX_PY" -m omlx.cli serve --model-dir "$MODEL_DIR" \
    --port "$PORT" > "$LOG" 2>&1 &
PID=$!
for _ in $(seq 1 300); do
    curl -sf --max-time 2 "http://localhost:${PORT}/v1/models" >/dev/null && break
    sleep 1
done

echo "--- models visible:"
curl -s "http://localhost:${PORT}/v1/models" | python3 -c "
import json,sys
for m in json.load(sys.stdin).get('data',[]): print('   ', m.get('id'))
" 2>/dev/null

echo "--- request (warm + measured):"
for i in 1 2; do
  curl -sf --max-time 900 "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function that merges two sorted lists.\"}],\"max_tokens\":128,\"temperature\":0,\"stream\":false}" \
    | python3 -c "
import json,sys
d=json.load(sys.stdin)
u=d.get('usage',{})
print('   tokens:', u.get('completion_tokens'), 'finish:', d['choices'][0].get('finish_reason'))
" 2>/dev/null || echo "   request failed"
done

sleep 2
kill "$PID" 2>/dev/null; sleep 1
pkill -f "omlx" 2>/dev/null

echo "--- MTP evidence in log:"
grep -iE "MTP path activated|mtp_enabled|Lightning MTP|MTP .*(accept|acceptance)|not eligible|no mtp" "$LOG" | tail -15
echo "--- errors:"
grep -iE "^Traceback|Error|error:" "$LOG" | tail -10
echo "(full log: $LOG)"
