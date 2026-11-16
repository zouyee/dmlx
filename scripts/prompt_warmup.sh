#!/bin/bash
# Replay the largest captured chat request (DMLX_PROMPT_DUMP) against a freshly
# started dmlx server so the KV prefix snapshot exists before the first real
# turn. Usage: scripts/prompt_warmup.sh [port] [dump_path]
set -u
PORT="${1:-8080}"
DUMP="${2:-/tmp/dmlx_prompt_dump.json}"
[ -s "$DUMP" ] || { echo "no prompt dump at $DUMP yet"; exit 0; }
python3 - "$DUMP" "$PORT" <<'EOF'
import json, sys, urllib.request
body = json.load(open(sys.argv[1]))
body["max_tokens"] = 1
body["stream"] = False
req = urllib.request.Request(
    f"http://127.0.0.1:{sys.argv[2]}/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"})
try:
    urllib.request.urlopen(req, timeout=900).read()
    print("warmup done")
except Exception as e:
    print(f"warmup failed: {e}")
EOF
