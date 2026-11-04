#!/bin/bash
# Price blocks of the PREFILL path by duplication, the same way
# scripts/ab_verify_round.sh prices the verify round.
#
# Why: the prefill gap against omlx is 1.66 s of non-GEMM work in our own
# code (see the "prefill gap LOCATED" commit). All GEMM for the measured
# 1536-token span is 73.3 TFLOP = 11.10 s at the 6.6 TF/s stock qmm achieves,
# omlx runs at 11.42 s (+3% over that floor) and we run at 12.76 s (+13%).
# This splits our +1.66 s.
#
# A max_tokens=1 request is prefill plus one decode step plus fixed overhead,
# so differencing a long prompt against a short one cancels everything that
# does not scale with prompt length. The DMLX_VERIFY_PRICE_* hooks run a block
# twice and average, which is exact for finite values, so the output stays
# correct and the time rise is an upper bound on one pass of that block.
#
# Usage: bash scripts/ab_prefill.sh "" "DMLX_VERIFY_PRICE_CONV=1" ...
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "${DIR}/.." && pwd)"
CLI="${PROJ}/zig-out/bin/dmlx"
MODEL="${MODEL:-${HOME}/models/Qwen3.8-27B-4bit}"
PORT="${PORT:-18201}"
SHORT_WORDS="${SHORT_WORDS:-64}"
LONG_WORDS="${LONG_WORDS:-1600}"
REPS="${REPS:-2}"

mk() { python3 -c "
import random,sys
random.seed(int(sys.argv[2]))
W=('alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa')
print(' '.join(random.choice(W) for _ in range(int(sys.argv[1]))))" "$1" "$2"; }

probe() { # $1=words $2=seed -> wall seconds for a 1-token completion
    local p; p=$(mk "$1" "$2")
    python3 -c "
import json,sys,time,urllib.request
body=json.dumps({'model':'m','messages':[{'role':'user','content':sys.stdin.read()}],
                 'max_tokens':1,'temperature':0,'stream':False}).encode()
r=urllib.request.Request('http://localhost:${PORT}/v1/chat/completions',data=body,
                          headers={'Content-Type':'application/json'})
t=time.perf_counter()
urllib.request.urlopen(r,timeout=1800).read()
print(f'{time.perf_counter()-t:.4f}')" <<<"$p"
}

run_one() {
    local extra="$1" log="/tmp/ab_prefill_$$.log"
    pkill -f "dmlx serve" 2>/dev/null; sleep 2
    env $extra "$CLI" serve --model "$MODEL" --port "$PORT" > "$log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 300); do
        curl -sf --max-time 2 "http://localhost:${PORT}/health" >/dev/null && break
        sleep 1
    done
    probe 32 1 >/dev/null 2>&1   # warm
    local bs=99999 bl=99999 s l
    for i in $(seq 1 "$REPS"); do
        s=$(probe "$SHORT_WORDS" "$((100+i))"); l=$(probe "$LONG_WORDS" "$((200+i))")
        bs=$(python3 -c "print(min($bs,$s))"); bl=$(python3 -c "print(min($bl,$l))")
    done
    kill "$pid" 2>/dev/null; sleep 1
    python3 -c "
s,l=$bs,$bl
print(f'  [${extra:-baseline}]  short={s:.3f}s  long={l:.3f}s  delta={l-s:.3f}s')"
}

echo "prefill A/B  short=${SHORT_WORDS}w  long=${LONG_WORDS}w  reps=${REPS}"
for v in "$@"; do run_one "$v"; done
pkill -f "dmlx serve" 2>/dev/null
