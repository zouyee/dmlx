#!/bin/bash
# One git-bisect probe for the "second 1k prefill hangs" regression.
# Usage: bisect_probe.sh <worktree_dir>   — expects binary already built there.
# Prints PASS (both requests complete) or HANG (second request stalls).
set -u
WT="$1"
CLI="$WT/zig-out/bin/dmlx"
PORT=18211
pkill -9 -f "zig-out/bin/dmlx serve" 2>/dev/null; sleep 2
"$CLI" serve --model "$HOME/models/Qwen3.8-27B-4bit" --port $PORT > /tmp/bisect_serve.log 2>&1 &
SRV=$!
for i in $(seq 1 240); do curl -sf --max-time 2 "http://localhost:$PORT/health" >/dev/null && break; sleep 1; done
mk() { python3 -c "import random,sys; random.seed(int(sys.argv[1])); print(' '.join(random.choice('alpha beta gamma delta epsilon zeta eta theta iota kappa'.split()) for _ in range(800)))" "$1"; }
PA=$(mk 1); PB=$(mk 2)
verdict=PASS
for P in "$PA" "$PB"; do
  code=$(curl -s --max-time 120 "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":1,\"temperature\":0}" \
    -o /dev/null -w "%{http_code}")
  if [[ "$code" != "200" ]]; then verdict=HANG; break; fi
done
kill -9 $SRV 2>/dev/null; pkill -9 -f "zig-out/bin/dmlx serve" 2>/dev/null; sleep 1
echo "$verdict" > /tmp/bisect_verdict.txt
echo "$verdict"
