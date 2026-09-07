#!/bin/bash
# Standalone test of the full MLA attention host orchestration
# (src/metal_infer/mla_attention.m) against a Python golden, in ~2s.
#
# Regenerates the golden + raw layer-0 weights, builds the harness, runs it.
# Use during S7/S8 attention development instead of the ~50s server load.
#
# Usage: bash scripts/run_mla_attention_test.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
python3 scripts/gen_attn_golden.py "${1:-7}"
BIN="$(mktemp -t mat.XXXXXX)"
clang -framework Metal -framework Foundation -framework Accelerate \
    -Wno-deprecated-declarations -I src/metal_infer -x objective-c \
    scripts/mla_attention_test.m src/metal_infer/mla_attention.m \
    src/metal_infer/engine.c src/metal_infer/routing_trace.c \
    src/metal_infer/mhc.c src/metal_infer/mach_compat.c \
    src/metal_infer/dspark_engine.c src/metal_infer/dspark_attention.c \
    src/metal_infer/batch_oracle.c -o "${BIN}"
"${BIN}"
rc=$?
rm -f "${BIN}"
exit $rc
