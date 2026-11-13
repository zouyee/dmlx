#!/bin/bash
# Interleaved A/B for chunked long prefill at 16K context:
# alternates DMLX_PREFILL_CHUNK=0 (monolithic) and default (2048),
# ONE 16K request per server visit so thermal drift hits both arms equally.
# PROBE_MAXTOK=8 keeps decode negligible so wall time ~= prefill.
set -u
cd "$(dirname "$0")/.."
for i in 1 2; do
  for arm in 1 0; do
    pkill -f "dmlx serve" 2>/dev/null; sleep 3
    log=/tmp/prefill_chunk_ab_${arm}_${i}.log
    if [ "$arm" = "0" ]; then
      DMLX_PREFILL_CHUNK=0 ./zig-out/bin/dmlx serve --model ~/models/Qwen3.8-27B-4bit --port 18202 > "$log" 2>&1 &
    else
      ./zig-out/bin/dmlx serve --model ~/models/Qwen3.8-27B-4bit --port 18202 > "$log" 2>&1 &
    fi
    for _ in $(seq 1 40); do curl -s localhost:18202/health >/dev/null 2>&1 && break; sleep 3; done
    out=$(PORT=18202 PROBE_WORDS=12000 PROBE_MAXTOK=8 python3 scripts/verify_ctx_probe.py 1 2>/dev/null | grep '^times:')
    echo "rep$i arm_chunk=$arm  $out"
  done
done
pkill -f "dmlx serve" 2>/dev/null
