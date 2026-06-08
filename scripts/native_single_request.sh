#!/bin/bash
# Run a single native request with dump and compare FFN output
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
PACKED_DIR="${MODEL_PATH}/packed_experts"
CLI="${PROJECT_DIR}/zig-out/bin/dmlx"
PORT="${PORT:-8969}"
DUMP_DIR="${1:-/tmp/native_single_dump}"

rm -rf "${DUMP_DIR}" && mkdir -p "${DUMP_DIR}"

# Start native server with dump
DSV4_DUMP_DIR="${DUMP_DIR}" MF_DBG=1 NATIVE=1 "${CLI}" serve \
    --model "${MODEL_PATH}" \
    --expert-packed-dir "${PACKED_DIR}" \
    --port "${PORT}" --max-tokens 1 --temperature 0 &
SERVER_PID=$!

cleanup() { kill "${SERVER_PID}" 2>/dev/null; wait "${SERVER_PID}" 2>/dev/null; }
trap cleanup EXIT

# Wait for ready
echo -n "waiting..."
for _ in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo " ready"
        break
    fi
    echo -n "."
    sleep 1
done

# Single France prompt request
echo "Sending France prompt..."
RESPONSE=$(curl -s --max-time 300 "http://localhost:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"default","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":1,"temperature":0}')
echo "Response: ${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin.readline().split(': ',1)[1] if False else sys.stdin); print('Token:', d['choices'][0]['message']['content'])" 2>/dev/null || echo "Response: ${RESPONSE}"

echo ""
echo "Dump files in ${DUMP_DIR}:"
ls "${DUMP_DIR}/"

# Compare FFN output  
python3 << PYEOF
import numpy as np, os

REF='/tmp/mlx_ref_new'
NAT='${DUMP_DIR}'

mlx_ffn = np.load(f'{REF}/L0_ffn_out.npy')[0,-1]  # last token
print(f"MLX L0_ffn_out (last token) norm: {np.linalg.norm(mlx_ffn):.4f}")

nat_ffn_path = f'{NAT}/L0_ffn_out_metal.bin'
if os.path.exists(nat_ffn_path):
    nat_ffn = np.fromfile(nat_ffn_path, dtype=np.float32)
    print(f"Native L0_ffn_out norm: {np.linalg.norm(nat_ffn):.4f}")
    cos = np.dot(nat_ffn, mlx_ffn)/(np.linalg.norm(nat_ffn)*np.linalg.norm(mlx_ffn)+1e-12)
    rl2 = np.linalg.norm(nat_ffn-mlx_ffn)/(np.linalg.norm(mlx_ffn)+1e-12)
    print(f"cos vs MLX: {cos:.4f}  rel_L2: {rl2:.4f}")
else:
    print("L0_ffn_out_metal.bin not found!")
PYEOF
