#!/bin/bash
# Fast (~2s) standalone Metal kernel test loop for the attention kernels.
# Runtime-compiles src/models/moe_kernel.metal and checks each new kernel
# (dequant_matvec_affine, rms_norm_rows, rope_tail_interleaved, mla_sdpa_decode)
# against a CPU reference. Use during S7/S8 kernel development instead of the
# ~50s server load.
#
# Usage: bash scripts/run_kernel_tests.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="$(mktemp -t mkt.XXXXXX)"
clang -framework Metal -framework Foundation -fobjc-arc \
    "${SCRIPT_DIR}/metal_kernel_test.m" -o "${BIN}"
"${BIN}" "${PROJECT_DIR}/src/models/moe_kernel.metal"
rc=$?
rm -f "${BIN}"
exit $rc
