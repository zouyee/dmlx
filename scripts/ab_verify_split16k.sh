#!/bin/bash
# Interleaved A/B for the verify split-KV kernel at 16K context:
# alternates DMLX_VERIFY_SDPA_SPLIT on/off, ONE 16K request per server visit,
# so thermal drift hits both arms equally. Prints the per-request verify
# ms/round from each server's log.
set -u
cd "$(dirname "$0")/.."
for i in 1 2 3; do
  for arm in 1 0; do
    pkill -f "dmlx serve" 2>/dev/null; sleep 3
    log=/tmp/verify_ab_${arm}_${i}.log
    if [ "$arm" = "0" ]; then
      DMLX_VERIFY_SDPA_SPLIT=0 ./zig-out/bin/dmlx serve --model ~/models/Qwen3.8-27B-4bit --speculative-mtp 7 --port 18201 > "$log" 2>&1 &
    else
      ./zig-out/bin/dmlx serve --model ~/models/Qwen3.8-27B-4bit --speculative-mtp 7 --port 18201 > "$log" 2>&1 &
    fi
    for _ in $(seq 1 40); do curl -s localhost:18201/health >/dev/null 2>&1 && break; sleep 3; done
    PROBE_WORDS=12000 python3 scripts/verify_ctx_probe.py 1 >/dev/null
    v=$(grep "MTP request" "$log" | tail -1 | grep -o "verify=[0-9.]*")
    echo "rep$i arm_split=$arm  $v"
  done
done
pkill -f "dmlx serve" 2>/dev/null
