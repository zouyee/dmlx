#!/bin/bash
# Build the two Python environments the engine comparison needs.
#
# These used to live in /tmp, which macOS cleans — and when they vanished,
# `compare_engines.py` happily skipped all four competitors and printed a
# comparison-shaped table containing only dmlx's own rows. They now live under
# ~/venvs so they survive, and the harness refuses to report a partial
# comparison unless `--allow-skips` is passed.
#
#   ~/venvs/dmlx-cmp   python3.14, --system-site-packages
#                      mlx 0.32.4 + mlx-lm 0.31.3 + mlx-vlm 0.7.8
#                      (mlx-vlm < 0.7 cannot load `qwen3_5_mtp`)
#   ~/venvs/dmlx-omlx  python3.13 (omlx requires >=3.11,<3.14)
#                      omlx 0.7.1, editable from ~/work/code/omlx
#
# Idempotent: re-running only reinstalls, and pip's cache makes that quick.
set -euo pipefail
CMP="${HOME}/venvs/dmlx-cmp"
OMLX="${HOME}/venvs/dmlx-omlx"
PY314="${PY314:-/opt/homebrew/bin/python3.14}"
PY313="${PY313:-/opt/homebrew/bin/python3.13}"
OMLX_SRC="${OMLX_SRC:-${HOME}/work/code/omlx}"

mkdir -p "${HOME}/venvs"

echo "== ${CMP} (mlx-lm / mlx-vlm)"
[[ -x "$CMP/bin/python" ]] || "$PY314" -m venv --system-site-packages "$CMP"
"$CMP/bin/pip" install -q "mlx==0.32.4" "mlx-lm==0.31.3" "mlx-vlm==0.7.8"
"$CMP/bin/python" -c "import mlx.core as m,mlx_lm,mlx_vlm;print('   mlx',m.__version__,'mlx_lm',mlx_lm.__version__,'mlx_vlm',mlx_vlm.__version__)"

echo "== ${OMLX} (omlx)"
[[ -d "$OMLX_SRC" ]] || { echo "   missing omlx checkout: $OMLX_SRC" >&2; exit 1; }
[[ -x "$OMLX/bin/python" ]] || "$PY313" -m venv "$OMLX"
"$OMLX/bin/pip" install -q -e "$OMLX_SRC"
"$OMLX/bin/python" -c "import omlx;print('   omlx',getattr(omlx,'__version__','?'))"

echo
echo "Also required for the omlx-mtp row (native Lightning MTP):"
echo "  ${OMLX}/bin/python scripts/omlx_enable_native_mtp.py"
echo "Verify with: python3 scripts/compare_engines.py --list"
