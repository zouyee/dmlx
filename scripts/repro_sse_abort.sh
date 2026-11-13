#!/bin/bash
# Regression repro for: SSE stream aborted mid-flight segfaulting the server.
#
# Root causes fixed in src/server/streaming.zig + src/server/http.zig:
#   1. The SSE keep-alive fiber was spawned fire-and-forget and borrowed
#      stack locals of handleStreamingCompletion — after the handler
#      returned (e.g. on a write error from an aborted client) the fiber
#      could still wake up and write through dangling stack pointers.
#   2. Writes to a socket whose peer closed raise SIGPIPE on macOS
#      (default disposition terminates the process); SO_NOSIGPIPE is now
#      set per connection so writes fail with EPIPE instead.
#
# Usage:
#   scripts/repro_sse_abort.sh [model_path] [port] [iterations]
# Defaults: ~/models/Qwen3.8-27B-4bit, port 18123, 10 iterations.
#
# Starts a fresh server, fires streaming chat completions, and cuts each
# client connection after the first bytes (`head -c`). The server must be
# alive and responsive after all iterations.

set -u

MODEL="${1:-$HOME/models/Qwen3.8-27B-4bit}"
PORT="${2:-18123}"
ITER="${3:-10}"
BIN="$(cd "$(dirname "$0")/.." && pwd)/zig-out/bin/dmlx"
LOG="$(mktemp -t dmlx_sse_abort)"

if [ ! -x "$BIN" ]; then
    echo "dmlx binary not found at $BIN — run 'zig build' first" >&2
    exit 1
fi

echo "Starting server (model=$MODEL, port=$PORT, log=$LOG) ..."
"$BIN" serve --model "$MODEL" --port "$PORT" >"$LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

# Wait for the port to accept connections.
for _ in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FAIL: server died during startup; log tail:" >&2
        tail -20 "$LOG" >&2
        exit 1
    fi
    sleep 1
done

echo "Server up (pid $SERVER_PID). Firing $ITER aborted streams ..."
for i in $(seq 1 "$ITER"); do
    # Cut the connection after the first 64 bytes of the SSE stream.
    curl -sN "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"m","messages":[{"role":"user","content":"Write a long story about a dragon."}],"stream":true,"max_tokens":64}' \
        | head -c 64 >/dev/null
    # Also exercise a hard cut BEFORE the first token (keep-alive window):
    curl -sN --max-time 2 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"m","messages":[{"role":"user","content":"Count to one thousand."}],"stream":true,"max_tokens":64}' \
        >/dev/null 2>&1
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FAIL: server died after aborted stream #$i; log tail:" >&2
        tail -30 "$LOG" >&2
        exit 1
    fi
done

# Let any in-flight keep-alive fibers hit their 5s wake-up against
# now-torn-down handler state — the original crash showed up here.
sleep 7

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FAIL: server died during the post-abort settle window; log tail:" >&2
    tail -30 "$LOG" >&2
    exit 1
fi

# Final liveness check: a normal (non-aborted) streaming request must work.
# Aborted requests drain their remaining generation server-side before the
# handler can unwind, so allow generous time for the queue to flush.
OUT=$(curl -sN --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"m","messages":[{"role":"user","content":"Say hi."}],"stream":true,"max_tokens":8}')
if ! printf '%s' "$OUT" | grep -q '\[DONE\]'; then
    echo "FAIL: post-abort streaming request did not complete cleanly:" >&2
    printf '%s\n' "$OUT" | tail -5 >&2
    exit 1
fi

echo "PASS: server survived $ITER aborted streams and still serves requests."
