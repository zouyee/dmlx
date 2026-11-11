#!/bin/bash
# Interleaved A/B: prefill qmm routing — v8 custom tile (default) vs stock
# MLX quantized_matmul (DMLX_PREFILL_QMM_V8=0). One 16K request per server
# visit, maxtok=8 so wall ~= prefill.
set -u
cd "$(dirname "$0")/.."
for i in 1 2; do
  for arm in 1 0; do
    pkill -f "dmlx serve" 2>/dev/null; sleep 3
    log=/tmp/qmmv8_ab_${arm}_${i}.log
    if [ "$arm" = "0" ]; then
      DMLX_PREFILL_QMM_V8=0 ./zig-out/bin/dmlx serve --model ~/models/Qwen3.8-27B-4bit --port 18207 > "$log" 2>&1 &
    else
      ./zig-out/bin/dmlx serve --model ~/models/Qwen3.8-27B-4bit --port 18207 > "$log" 2>&1 &
    fi
    for _ in $(seq 1 40); do curl -s localhost:18207/health >/dev/null 2>&1 && break; sleep 3; done
    out=$(PORT=18207 PROBE_WORDS=12000 PROBE_MAXTOK=8 python3 scripts/verify_ctx_probe.py 1 2>/dev/null | grep '^times:')
    echo "rep$i arm_v8=$arm  $out"
  done
done
pkill -f "dmlx serve" 2>/dev/null
