#!/bin/bash
# A/B the re-explore latch floor (max_n-2 vs old max_n-1) on the
# thinking-regime math lottery, watching code/prose for overshoot.
# Interleaved blocks cancel thermal drift. Each arm: 8 math + 4 code + 4 prose.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "${DIR}/.." && pwd)"
CLI="${PROJ}/zig-out/bin/dmlx"
MODEL="${MODEL:-${HOME}/models/Qwen3.8-27B-4bit}"
PORT=18209

run_arm() { # $1=label $2=extra-env
    pkill -f "dmlx serve" 2>/dev/null; sleep 2
    env $2 "$CLI" serve --model "$MODEL" --speculative-mtp 7 --port "$PORT" > "/tmp/latch_$1.log" 2>&1 &
    sleep 55
    python3 - "$1" <<'PYEOF'
import json, time, urllib.request, sys
label=sys.argv[1]
directive='Before answering, reason step by step inside <think> and </think> tags. Keep the reasoning brief and structured, then give the final answer.'
prompts={'math':'What is 17 * 23? Show your working briefly.','code':'Write a Python function that merges two sorted lists.','prose':'Explain in one paragraph why unified memory helps local LLM inference.'}
plan=[('math',8),('code',4),('prose',4)]
for kind,n in plan:
    for i in range(n):
        body=json.dumps({'model':'m','messages':[{'role':'system','content':directive},{'role':'user','content':f'[{label}-{kind}-{i}] '+prompts[kind]}],'max_tokens':128,'temperature':0,'stream':False}).encode()
        r=urllib.request.Request('http://localhost:18209/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
        t=time.perf_counter(); resp=json.loads(urllib.request.urlopen(r,timeout=300).read()); dt=time.perf_counter()-t
        tok=resp['usage']['completion_tokens']
        print(f'{label} {kind} {tok/dt:.2f}', flush=True)
PYEOF
}

for blk in 1 2; do
    run_arm "new_b$blk" ""
    run_arm "old_b$blk" "DMLX_MTP_REEXPLORE_FLOOR=6"
done
pkill -f "dmlx serve" 2>/dev/null
echo done
