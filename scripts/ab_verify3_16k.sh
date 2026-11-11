#!/bin/bash
# Interleaved 3-arm verify-cost pricing at 16K context (one 16K request per
# server visit, 2 reps): default (split SDPA on) vs NO_ATTN_CORE stand-in
# (removes cacheAppend + SDPA + gate; wrong numerics, pricing only) vs split
# off (stock composed SDPA). Prints per-request verify ms/round.
set -u
cd "$(dirname "$0")/.."
for i in 1 2; do
  for arm in default nocore nosplit; do
    pkill -f "dmlx serve" 2>/dev/null; sleep 3
    log=/tmp/verify3_${arm}_${i}.log
    case "$arm" in
      nocore)  DMLX_VERIFY_NO_ATTN_CORE=1 ./zig-out/bin/dmlx serve --model ~/models/Qwen3.8-27B-4bit --speculative-mtp 7 --port 18205 > "$log" 2>&1 & ;;
      nosplit) DMLX_VERIFY_SDPA_SPLIT=0 ./zig-out/bin/dmlx serve --model ~/models/Qwen3.8-27B-4bit --speculative-mtp 7 --port 18205 > "$log" 2>&1 & ;;
      *)       ./zig-out/bin/dmlx serve --model ~/models/Qwen3.8-27B-4bit --speculative-mtp 7 --port 18205 > "$log" 2>&1 & ;;
    esac
    for _ in $(seq 1 40); do curl -s localhost:18205/health >/dev/null 2>&1 && break; sleep 3; done
    PORT=18205 PROBE_WORDS=12000 python3 scripts/verify_ctx_probe.py 1 >/dev/null
    v=$(grep "MTP request" "$log" | tail -1 | grep -o "verify=[0-9.]*")
    echo "rep$i arm=$arm  $v"
  done
done
pkill -f "dmlx serve" 2>/dev/null
