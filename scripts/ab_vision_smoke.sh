#!/bin/bash
# Vision smoke test (Task 7): start dmlx serve on the Qwen3.8 checkpoint,
# POST an OpenAI chat completion whose user message is content parts
# (image_url data URL + text), and assert a non-error finish_reason:"stop".
#
# Usage: bash scripts/ab_vision_smoke.sh
# Env:   MODEL (default ~/models/Qwen3.8-27B-4bit), PORT (default 18098),
#        MAXTOK (default 64), MTP (set to e.g. 4 to enable --speculative-mtp)
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${DIR}/.." && pwd)"
CLI="${PROJECT_DIR}/zig-out/bin/dmlx"
MODEL="${MODEL:-${HOME}/models/Qwen3.8-27B-4bit}"
PORT="${PORT:-18098}"
MAXTOK="${MAXTOK:-256}"
IMAGE="${IMAGE:-${PROJECT_DIR}/tests/fixtures/vision/photo.png}"
LOG="/tmp/ab_vision_smoke_$$.log"

MTP_ARGS=()
if [ -n "${MTP:-}" ]; then
    MTP_ARGS=(--speculative-mtp "$MTP")
fi

pkill -f "dmlx serve" 2>/dev/null; sleep 2
# Empty-think keeps the smoke short: the model answers directly instead of
# burning the token budget on a thinking block before the description.
DMLX_CHAT_EMPTY_THINK=1 "$CLI" serve --model "$MODEL" --port "$PORT" ${MTP_ARGS[@]+"${MTP_ARGS[@]}"} >"$LOG" 2>&1 &
pid=$!
cleanup() { kill "$pid" 2>/dev/null; }
trap cleanup EXIT

ready=0
for _ in $(seq 1 240); do
    curl -sf --max-time 2 "http://localhost:${PORT}/health" >/dev/null && { ready=1; break; }
    sleep 1
done
if [ "$ready" != "1" ]; then
    echo "FAIL: server did not become healthy (log: $LOG)"
    tail -20 "$LOG"
    exit 1
fi

b64=$(base64 -i "$IMAGE" | tr -d '\n')
payload=$(python3 -c "
import json, sys
b64 = open('/dev/stdin').read().strip()
print(json.dumps({
    'model': 'm',
    'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64}},
        {'type': 'text', 'text': 'Describe this image.'},
    ]}],
    'max_tokens': ${MAXTOK},
    'temperature': 0,
    'stream': False,
}))
" <<< "$b64")

resp=$(curl -sf --max-time 600 "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$payload")
if [ -z "$resp" ]; then
    echo "FAIL: empty/error response (log: $LOG)"
    tail -20 "$LOG"
    exit 1
fi

echo "$resp" | python3 -c "
import json, sys
r = json.load(sys.stdin)
if 'error' in r:
    print('FAIL: server returned error:', r['error'])
    sys.exit(1)
ch = r['choices'][0]
fr = ch.get('finish_reason')
content = ch['message'].get('content') or ''
print('finish_reason:', fr)
print('content:', content[:400])
assert fr == 'stop', f'finish_reason {fr!r} != stop'
assert len(content.strip()) > 0, 'empty content'
print('PASS: vision smoke')
"
