#!/usr/bin/env bash
# A/B the drafter rope-delta convention on a vision prompt.
# Usage: bash scripts/ab_vision_drafter_delta.sh [image] [max_tokens]
set -u
IMG=${1:-/tmp/bench_photo_1024.png}
MAXTOK=${2:-128}
PORT=8186
MODEL=~/models/Qwen3.8-27B-4bit
DMLX=/Users/zouyee/work/code/dmlx/zig-out/bin/dmlx
B64=$(base64 -i "$IMG" | tr -d '\n')
BODY=$(python3 -c "
import json,sys
b64=sys.argv[1]
print(json.dumps({'model':'default','temperature':0,'max_tokens':$MAXTOK,'stream':False,
 'messages':[{'role':'user','content':[
   {'type':'image_url','image_url':{'url':'data:image/png;base64,'+b64}},
   {'type':'text','text':'Describe this image in detail.'}]}]}))" "$B64")

run_arm() {
  local tag=$1 delta=$2
  DMLX_MTP_VISION_DRAFTER_DELTA=$delta $DMLX serve --model "$MODEL" --port $PORT --speculative-mtp 4 > /tmp/vision_delta_$tag.log 2>&1 &
  local pid=$!
  for i in $(seq 1 120); do curl -s -o /dev/null http://127.0.0.1:$PORT/health && break; sleep 2; done
  curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "$BODY" \
    | python3 -c "import json,sys; o=json.load(sys.stdin); print('$tag wall usage:', o['usage'])"
  sleep 1
  grep -E "MTP request|cond-acceptance" /tmp/vision_delta_$tag.log | tail -4
  kill $pid; wait $pid 2>/dev/null
  sleep 3
}

run_arm delta_on 1
run_arm delta_off 0
