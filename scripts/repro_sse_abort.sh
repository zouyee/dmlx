#!/bin/bash
# Regression repro for: SSE stream aborted mid-flight segfaulting the server.
#
# Root causes fixed in src/server/streaming.zig + src/server/responses.zig +
# src/server/http.zig:
#   1. The HTTP handler frame owns the engine RequestState; on an SSE write
#      error (client gone), cancellation, or OOM it must cancel the request
#      and drain the completion signal until the engine reports done —
#      otherwise req_state.deinit() frees the signal while the engine thread
#      is still calling deliverToken into it (use-after-free → segfault).
#      Applies to both /v1/chat/completions and /v1/responses streaming.
#   2. The SSE keep-alive fiber (chat path) was spawned fire-and-forget and
#      borrowed stack locals of the handler — it is now joined via cancel.
#   3. Writes to a socket whose peer closed raise SIGPIPE on macOS
#      (default disposition terminates the process); SO_NOSIGPIPE is now
#      set per connection so writes fail with EPIPE instead.
#
# Usage:
#   scripts/repro_sse_abort.sh [model_path] [port] [iterations]
# Defaults: ~/models/Qwen3.8-27B-4bit, port 18123, 10 iterations.
#
# Starts a fresh server, fires streaming requests against BOTH streaming
# endpoints, and cuts each client connection mid-stream / pre-first-token.
# The server must be alive and responsive after all iterations.

set -u -o pipefail

MODEL="${1:-$HOME/models/Qwen3.8-27B-4bit}"
PORT="${2:-18123}"
ITER="${3:-10}"
BIN="$(cd "$(dirname "$0")/.." && pwd)/zig-out/bin/dmlx"
LOG="$(mktemp -t dmlx_sse_abort)"

if [ ! -x "$BIN" ]; then
    echo "dmlx binary not found at $BIN — run 'zig build' first" >&2
    exit 1
fi

# Refuse to run against a stale server already occupying the port —
# results must reflect the binary we start here.
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "FAIL: port $PORT is already in use by another process:" >&2
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2
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

# Wait for the port to accept connections, then assert the listener is the
# server we just started (not a stale instance).
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
LISTENER_PID=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1)
if [ "$LISTENER_PID" != "$SERVER_PID" ]; then
    echo "FAIL: port $PORT is served by pid ${LISTENER_PID:-none}, not our server pid $SERVER_PID" >&2
    exit 1
fi

fire_aborted_pair() {
    local endpoint="$1" payload="$2"
    # Cut the connection after the first 64 bytes of the SSE stream.
    # (curl exits nonzero when its write end is cut by head — expected.)
    curl -sN "http://127.0.0.1:$PORT$endpoint" \
        -H 'Content-Type: application/json' \
        -d "$payload" \
        | head -c 64 >/dev/null || true
    # Also exercise a hard cut BEFORE the first token (keep-alive window):
    curl -sN --max-time 2 "http://127.0.0.1:$PORT$endpoint" \
        -H 'Content-Type: application/json' \
        -d "$payload" \
        >/dev/null 2>&1 || true
}

CHAT_PAYLOAD='{"model":"m","messages":[{"role":"user","content":"Write a long story about a dragon."}],"stream":true,"max_tokens":64}'
RESPONSES_PAYLOAD='{"model":"m","input":"Write a long story about a dragon.","stream":true,"max_output_tokens":64}'

echo "Server up (pid $SERVER_PID). Firing $ITER aborted streams per endpoint ..."
for i in $(seq 1 "$ITER"); do
    fire_aborted_pair /v1/chat/completions "$CHAT_PAYLOAD"
    fire_aborted_pair /v1/responses "$RESPONSES_PAYLOAD"
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

# Final liveness check: normal (non-aborted) streaming requests against both
# endpoints must work. Aborted requests drain their remaining generation
# server-side before the handler can unwind, so allow generous time for the
# queue to flush.
OUT=$(curl -sN --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"m","messages":[{"role":"user","content":"Say hi."}],"stream":true,"max_tokens":8}')
if ! printf '%s' "$OUT" | grep -q '\[DONE\]'; then
    echo "FAIL: post-abort chat streaming request did not complete cleanly:" >&2
    printf '%s\n' "$OUT" | tail -5 >&2
    exit 1
fi
OUT=$(curl -sN --max-time 600 "http://127.0.0.1:$PORT/v1/responses" \
    -H 'Content-Type: application/json' \
    -d '{"model":"m","input":"Say hi.","stream":true,"max_output_tokens":8}')
if ! printf '%s' "$OUT" | grep -q 'response.completed'; then
    echo "FAIL: post-abort /v1/responses streaming request did not complete cleanly:" >&2
    printf '%s\n' "$OUT" | tail -5 >&2
    exit 1
fi

echo "PASS: server survived $ITER aborted streams per endpoint and still serves requests."
