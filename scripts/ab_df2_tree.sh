#!/bin/bash
# U5 tree-verify A/B: DMLX_DF2_TREE=0 (chain) vs =1 (tree+copies).
# Same greedy prompt; outputs must be token-identical.
set -uo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLI="${PROJECT_DIR}/zig-out/bin/dmlx"
MODEL="${HOME}/models/Qwen3.8-27B-4bit"
DF2_DIR=$(echo ${HOME}/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/*/)
PORT=18111
MAXTOK=200
PROMPT="Write a Python function that merges two sorted lists, then explain its time complexity."
OUT="${PROJECT_DIR}/artifacts/dflash2_tree_u5_$(date +%Y%m%d)"
mkdir -p "$OUT"

run_one() {
    local tag="$1"; shift
    local log="${OUT}/server_${tag}.log" out="${OUT}/out_${tag}.json"
    pkill -f "dmlx serve" 2>/dev/null; sleep 2
    env DMLX_QWEN38_DFLASH2_DIR="$DF2_DIR" "$@" \
        "$CLI" serve --model "$MODEL" --speculative-mtp 7 --port "$PORT" > "$log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 300); do
        curl -sf --max-time 2 "http://localhost:${PORT}/health" >/dev/null && break
        sleep 1
    done
    # warmup (not measured)
    curl -sf --max-time 600 "http://localhost:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":${MAXTOK},\"temperature\":0,\"stream\":false}" \
        > /dev/null
    # measured request
    curl -sf --max-time 600 "http://localhost:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":${MAXTOK},\"temperature\":0,\"stream\":false}" \
        > "$out"
    kill "$pid" 2>/dev/null; sleep 2
}

run_one chain
run_one tree ${TREE_ENV:+$TREE_ENV}
pkill -f "dmlx serve" 2>/dev/null || true

python3 - "$OUT" <<'EOF'
import json, sys, re
out = sys.argv[1]
texts = {}
for tag in ("chain", "tree"):
    with open(f"{out}/out_{tag}.json") as f:
        texts[tag] = json.load(f)["choices"][0]["message"]["content"]
identical = texts["chain"] == texts["tree"]
print(f"OUTPUT IDENTICAL: {identical}  (chain {len(texts['chain'])} chars, tree {len(texts['tree'])} chars)")
if not identical:
    a, b = texts["chain"], texts["tree"]
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print(f"first diff at char {i}: chain={a[max(0,i-40):i+20]!r} tree={b[max(0,i-40):i+20]!r}")
            break
for tag in ("chain", "tree"):
    log = open(f"{out}/server_{tag}.log", errors="replace").read()
    # last request's lines (after warmup)
    req = re.findall(r"\[Engine\] MTP request .*", log)
    tree = re.findall(r"\[Engine\] MTP tree: .*", log)
    toks = re.findall(r"tokens_per_sec=([0-9.]+)", log)
    print(f"--- {tag} ---")
    for l in req[-1:]: print(l)
    for l in tree[-1:]: print(l)
    if toks: print("tok/s (req1 warmup, req2 measured):", toks[-2:])
EOF
