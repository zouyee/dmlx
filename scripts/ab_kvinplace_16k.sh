#!/bin/bash
# Interleaved A/B for the in-place KV slab append at 16K context:
# DMLX_KV_INPLACE=0 (slice_update copy) vs default (in-place kernel).
# One 16K MTP request per server visit; checks wall time, verify ms/round,
# and that greedy output text is bit-identical across arms.
set -u
cd "$(dirname "$0")/.."
rm -f /tmp/kvinplace_text_*.txt
for i in 1 2; do
  for arm in 1 0; do
    pkill -f "dmlx serve" 2>/dev/null; sleep 3
    log=/tmp/kvinplace_${arm}_${i}.log
    if [ "$arm" = "0" ]; then
      DMLX_KV_INPLACE=0 ./zig-out/bin/dmlx serve --model ~/models/Qwen3.8-27B-4bit --speculative-mtp 7 --port 18206 > "$log" 2>&1 &
    else
      ./zig-out/bin/dmlx serve --model ~/models/Qwen3.8-27B-4bit --speculative-mtp 7 --port 18206 > "$log" 2>&1 &
    fi
    for _ in $(seq 1 40); do curl -s localhost:18206/health >/dev/null 2>&1 && break; sleep 3; done
    ARM=$arm REP=$i PROBE_WORDS=12000 PORT=18206 python3 - <<'EOF'
import json, os, random, time, urllib.request
random.seed(42)
W = ('alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa')
prompt = ' '.join(random.choice(W) for _ in range(int(os.environ.get('PROBE_WORDS','12000'))))
port = os.environ.get('PORT','18206')
body = json.dumps({'model':'m','messages':[{'role':'user','content':prompt}],
                   'max_tokens':128,'temperature':0,'stream':False}).encode()
r = urllib.request.Request(f'http://localhost:{port}/v1/chat/completions', data=body,
                           headers={'Content-Type':'application/json'})
t = time.perf_counter()
resp = json.loads(urllib.request.urlopen(r, timeout=1800).read())
dt = time.perf_counter() - t
text = resp['choices'][0]['message']['content']
open(f"/tmp/kvinplace_text_{os.environ['ARM']}_{os.environ['REP']}.txt",'w').write(text)
print(f"wall={dt:.2f}s textlen={len(text)}")
EOF
    v=$(grep "MTP request" "$log" | tail -1 | grep -o "verify=[0-9.]*")
    eng=$(grep -c "kv-inplace" "$log")
    echo "rep$i arm_inplace=$arm  $v  engaged_marks=$eng"
  done
done
pkill -f "dmlx serve" 2>/dev/null
echo "--- text equality across arms ---"
md5 /tmp/kvinplace_text_*.txt
