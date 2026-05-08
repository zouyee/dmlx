#!/bin/bash
set -uo pipefail

MODEL_PATH="${1:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
CLI="${PWD}/zig-out/bin/dmlx"
REPORT="docs/en/analysis/performance-benchmark.md"
DIR="$(cd "$(dirname "$0")" && pwd)"

export BM_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
export BM_BRANCH=$(git branch --show-current 2>/dev/null || echo "?")
export BM_DATE=$(date +%Y-%m-%d)
export BM_HW=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "?")
export BM_MEM=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0fGB",$1/1073741824}' || echo "?")
T0=$(date +%s)

echo "🔧 Build (ReleaseFast)..."
zig build -Doptimize=ReleaseFast 2>/dev/null

echo "🧪 Unit tests..."
T1=$(date +%s)
if zig build test >/dev/null 2>&1; then export BM_UNIT="PASS (430+)"; else export BM_UNIT="FAIL"; fi
echo "   $BM_UNIT ($(($(date +%s)-T1))s)"

echo "📊 Perf (10 tok)..."
T2=$(date +%s)
PERF=$("$CLI" chat --model "$MODEL_PATH" --prompt "Hello" --max-tokens 10 --temperature 0 --smelt --smelt-strategy stream --smelt-experts 0.1 2>&1)
export BM_PERF_SECS=$(($(date +%s)-T2))
echo "   ${BM_PERF_SECS}s"
PF=$(mktemp); echo "$PERF" | grep "Token step.*complete" > "$PF"

echo "✅ E2E (7 prompts)..."
T3=$(date +%s)
E2E=$(bash scripts/best_test.sh "$MODEL_PATH" "$CLI" 2>&1) || true
export BM_E2E_SECS=$(($(date +%s)-T3))
echo "   $(echo "$E2E" | grep '^Results:') (${BM_E2E_SECS}s)"
EF=$(mktemp); echo "$E2E" > "$EF"

export BM_TOTAL_SECS=$(($(date +%s)-T0))

echo "📝 Report..."
python3 "$DIR/_gen_report.py" "$PF" "$EF" "$REPORT"
rm -f "$PF" "$EF"
echo "✅ Done → $REPORT (${BM_TOTAL_SECS}s)"
