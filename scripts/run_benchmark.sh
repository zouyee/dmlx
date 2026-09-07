#!/bin/bash
# ============================================================
# dmlx Performance Benchmark — Serve Mode
# ============================================================
# Generates: docs/en/analysis/performance-benchmark.md
#
# Supports two modes:
#
#   NATIVE mode (default, --native engine with safe SMELT N=20):
#     bash scripts/run_benchmark.sh
#     bash scripts/run_benchmark.sh --native
#     NATIVE_SMELT_N=20 bash scripts/run_benchmark.sh
#
#   MLX mode (legacy):
#     bash scripts/run_benchmark.sh --mlx [model_path] [smelt_experts] [cache_mb]
#
# Native mode measures two explicitly separate phases:
#   1. Start one server with the configured SMELT cache (default N=20)
#   2. Warm GPU caches
#   3. Send 3 standard requests (max_tokens=5) and report median tok/s
#   4. Run an optional 100-token long-decode diagnostic, excluded from median
#   5. Run correctness gates serially
# ============================================================

set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${DIR}/.." && pwd)"
CLI="${PROJECT_DIR}/zig-out/bin/dmlx"
REPORT="${BENCHMARK_REPORT:-${PROJECT_DIR}/docs/en/analysis/performance-benchmark.md}"
PORT=18090
SERVER_URL="http://localhost:${PORT}"
SERVE_LOG="/tmp/benchmark_serve.log"
TELEMETRY_JSONL=""

# Exact-Q4 Task 0.3 tooling mode. This is intentionally explicit: the default
# benchmark path and performance-benchmark.md remain unchanged.
if [[ "${1:-}" == "--exact-q4-task-0.3" ]]; then
    shift
    TASK03_MODEL_PATH="${1:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
    TASK03_PACKED_DIR="${TASK03_MODEL_PATH}/packed_experts"
    TASK03_DIR="${PROJECT_DIR}/docs/en/analysis/exact-q4-task03-artifacts"
    TASK03_AGGREGATE="${TASK03_DIR}/exact-q4-task03-aggregate.json"
    TASK03_REPORT="${PROJECT_DIR}/docs/en/analysis/exact-q4-latency-baseline.md"
    mkdir -p "$TASK03_DIR"
    TASK03_MODEL_FINGERPRINT=$(PYTHONPATH="$DIR" python3 - "$TASK03_MODEL_PATH" "$TASK03_PACKED_DIR" <<'PY'
import sys
from pathlib import Path
from compare_batch_oracle import compute_fingerprints
print(compute_fingerprints(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())['model_manifest_sha256'])
PY
) || exit 1
    [[ "$TASK03_MODEL_FINGERPRINT" =~ ^[0-9a-f]{64}$ ]] || { echo "❌ invalid Task 0.3 model fingerprint"; exit 1; }

    task03_cleanup() {
        pkill -f "dmlx serve" 2>/dev/null || true
        sleep 1
        if pgrep -f "dmlx serve" >/dev/null 2>&1; then
            echo "❌ exact-Q4 Task 0.3 cleanup left a dmlx serve process"
            return 1
        fi
    }
    task03_write_identity() {
        local out="$1" role="$2" pair="$3" capture="$4"
        python3 - "$out" "$role" "$pair" "$capture" "$PROJECT_DIR" "$CLI" "$TASK03_MODEL_PATH" "$TASK03_PACKED_DIR" "${SKIP_PURGE:-0}" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
out, role, pair, capture, workspace, binary, model, packed, skip = sys.argv[1:]
def file_hash(path):
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()
def tree_hash(root, predicate=lambda p: True):
    root=Path(root); h=hashlib.sha256()
    for p in sorted(x for x in root.rglob('*') if x.is_file() and predicate(x)):
        rel=p.relative_to(root).as_posix(); h.update(rel.encode()+b'\0'+bytes.fromhex(file_hash(p)))
    return h.hexdigest()
def command_output(args, fallback='?'):
    try: return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip() or fallback
    except (OSError, subprocess.CalledProcessError): return fallback
revision=command_output(['git','-C',workspace,'rev-parse','HEAD'])
sys.path.insert(0, str(Path(workspace)/'scripts'))
from compare_batch_oracle import compute_fingerprints
fingerprints=compute_fingerprints(Path(model), Path(packed))
doc={'workspace_revision':revision,'binary_sha256':file_hash(binary),
     'model_sha256':fingerprints['model_manifest_sha256'],
     'tokenizer_sha256':fingerprints['tokenizer_sha256'],
     'packed_sha256':tree_hash(packed, lambda p: not any(x.startswith('.') for x in p.relative_to(packed).parts)),
     'command':f'bash scripts/run_benchmark.sh --exact-q4-task-0.3 {model}',
     'environment':{'NATIVE_SMELT_N':'20','NATIVE_EXACT_Q4_CAPTURE':capture,
       'NATIVE_EXPERIMENTAL_CACHE':'0','NATIVE_PREFETCH':'0','NATIVE_SMELT_PREFETCH':'0',
       'NATIVE_BATCH_PREFILL':'0','NATIVE_BATCH_MLA':'0','NATIVE_EXPERT_BUCKETS':'0',
       'NATIVE_BATCH_EXPERT_KERNELS':'0','NATIVE_DSPARK_BATCH_VERIFY':'0','SKIP_PURGE':skip},
     'role':role,'pair_ordinal':int(pair)}
Path(out).write_text(json.dumps(doc,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
PY
        local actual_binary_sha
        actual_binary_sha=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$CLI") || return 1
        [[ "$actual_binary_sha" == "$TASK03_FROZEN_BINARY_SHA" ]] || {
            echo "❌ Task 0.3 frozen binary changed during ${role}"
            return 1
        }
    }
    trap task03_cleanup EXIT INT TERM
    task03_cleanup || exit 1

    TASK03_SOURCES=()
    echo "🔧 exact-Q4 Task 0.3 one-time ReleaseFast build/test (binary frozen across all four runs)..."
    (cd "$PROJECT_DIR" && rm -rf .zig-cache && zig build -Doptimize=ReleaseFast 2>/dev/null) || exit 1
    (cd "$PROJECT_DIR" && zig build test -Doptimize=ReleaseFast >/dev/null 2>&1) || exit 1
    (cd "$PROJECT_DIR" && zig build -Doptimize=ReleaseFast 2>/dev/null) || exit 1
    TASK03_FROZEN_BINARY_SHA=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$CLI") || exit 1
    ROLES=(baseline1 enabled1 baseline2 enabled2)
    for role in "${ROLES[@]}"; do
        pair=1; [[ "$role" == *2 ]] && pair=2
        benchmark_json="${TASK03_DIR}/${role}-benchmark.json"
        raw_jsonl="${TASK03_DIR}/${role}-raw.jsonl"
        identity_json="${TASK03_DIR}/${role}-identity.json"
        standard_json="${TASK03_DIR}/${role}-standard.exact-q4.json"
        long_json="${TASK03_DIR}/${role}-long.exact-q4.json"
        child_report="${TASK03_DIR}/${role}-benchmark.md"
        rm -f "$benchmark_json" "$raw_jsonl" "$identity_json" "$standard_json" "$long_json" "$child_report"
        task03_cleanup || exit 1
        echo "▶ exact-Q4 Task 0.3 ${role}"
        COMMON_ENV=(NATIVE_SMELT_N=20 NATIVE_EXPERIMENTAL_CACHE=0 NATIVE_PREFETCH=0 NATIVE_SMELT_PREFETCH=0
            NATIVE_BATCH_PREFILL=0 NATIVE_BATCH_MLA=0 NATIVE_EXPERT_BUCKETS=0
            NATIVE_BATCH_EXPERT_KERNELS=0 NATIVE_DSPARK_BATCH_VERIFY=0 DSPARK_DIR=
            SKIP_PURGE="${SKIP_PURGE:-0}" SKIP_BENCHMARK_BUILD=1 BM_MODEL_FINGERPRINT="$TASK03_MODEL_FINGERPRINT" BENCHMARK_JSON="$benchmark_json" BENCHMARK_REPORT="$child_report")
        if [[ "$role" == enabled* ]]; then
            env "${COMMON_ENV[@]}" NATIVE_EXACT_Q4_CAPTURE=1 NATIVE_EXACT_Q4_RAW_JSONL="$raw_jsonl" \
                NATIVE_EXACT_Q4_MAX_PROMPT_TOKENS=256 bash "$0" --native "$TASK03_MODEL_PATH" || exit 1
            task03_write_identity "$identity_json" "$role" "$pair" 1 || exit 1
            python3 "$DIR/benchmark_report.py" --exact-q4-raw-jsonl "$raw_jsonl" --exact-q4-identity "$identity_json" --exact-q4-select 10,11,12 --exact-q4-output "$standard_json" || exit 1
            python3 "$DIR/benchmark_report.py" --exact-q4-raw-jsonl "$raw_jsonl" --exact-q4-identity "$identity_json" --exact-q4-select 13 --exact-q4-output "$long_json" || exit 1
            python3 "$DIR/benchmark_report.py" --validate-exact-q4-latency "$standard_json" || exit 1
            python3 "$DIR/benchmark_report.py" --validate-exact-q4-latency "$long_json" || exit 1
            TASK03_SOURCES+=("${role}:${benchmark_json}:${standard_json}:${long_json}:${identity_json}")
        else
            env -u NATIVE_EXACT_Q4_CAPTURE -u NATIVE_EXACT_Q4_RAW_JSONL "${COMMON_ENV[@]}" \
                bash "$0" --native "$TASK03_MODEL_PATH" || exit 1
            task03_write_identity "$identity_json" "$role" "$pair" 0 || exit 1
            TASK03_SOURCES+=("${role}:${benchmark_json}:-:-:${identity_json}")
        fi
        task03_cleanup || exit 1
    done

    TASK03_ARGS=(--exact-q4-task03-output "$TASK03_AGGREGATE")
    [[ -n "${TASK03_DRIFT_ROOT_CAUSE:-}" ]] && TASK03_ARGS+=(--exact-q4-task03-drift-root-cause "$TASK03_DRIFT_ROOT_CAUSE")
    for source in "${TASK03_SOURCES[@]}"; do TASK03_ARGS+=(--exact-q4-task03-source "$source"); done
    python3 "$DIR/benchmark_report.py" "${TASK03_ARGS[@]}" || exit 1
    python3 "$DIR/benchmark_report.py" --validate-exact-q4-task03 "$TASK03_AGGREGATE" || exit 1
    python3 - "$TASK03_AGGREGATE" "$TASK03_REPORT" <<'PY'
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
lines=['# Exact-Q4 latency baseline (Task 0.3)', '', '> Generated from the canonical aggregate. Task 0.3 remains unchecked pending collection/review.', '', '| Role | Pair | Standard tok/s | Peak RSS bytes | Correctness |', '|---|---:|---:|---:|---|']
for r in d['runs']:
    lines.append(f"| {r['role']} | {r['pair_ordinal']} | {r['standard_median_tokens_per_second']:.6f} | {r['peak_rss_bytes']} | {'PASS' if r['correctness_pass'] else 'FAIL'} |")
lines += ['', f"Decision: **{'PASS' if d['decision']['pass'] else 'FAIL'}**", '', f"Aggregate: `{Path(sys.argv[1]).resolve()}`", '']
Path(sys.argv[2]).write_text('\n'.join(lines), encoding='utf-8')
PY
    task03_cleanup || exit 1
    exit 0
fi

# Task 6.5: repeated, strictly serial fixed-hot-vs-bounded-LRU cache gate.
if [[ "${1:-}" == "--cache-promotion-gate" ]]; then
    shift
    CACHE_PROMOTION_MODEL_PATH="${1:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
    CACHE_PROMOTION_REPEATS="${CACHE_PROMOTION_REPEATS:-2}"
    CACHE_PROMOTION_ARTIFACT_DIR="${PROJECT_DIR}/docs/en/analysis"
    CACHE_PROMOTION_SUMMARY="${CACHE_PROMOTION_ARTIFACT_DIR}/performance-cache-promotion.json"

    if ! [[ "$CACHE_PROMOTION_REPEATS" =~ ^[0-9]+$ ]] || (( CACHE_PROMOTION_REPEATS < 2 )); then
        echo "❌ CACHE_PROMOTION_REPEATS must be an integer >= 2"
        exit 1
    fi
    CACHE_PROMOTION_MODEL_FINGERPRINT=$(PYTHONPATH="${DIR}" python3 - "$CACHE_PROMOTION_MODEL_PATH" <<'PY'
import sys
from pathlib import Path
from compare_batch_oracle import compute_fingerprints
model = Path(sys.argv[1]).expanduser().resolve()
print(compute_fingerprints(model, model / "packed_experts")["model_manifest_sha256"])
PY
) || exit 1
    if ! [[ "$CACHE_PROMOTION_MODEL_FINGERPRINT" =~ ^[0-9a-f]{64}$ ]]; then
        echo "❌ cache promotion model fingerprint is invalid"
        exit 1
    fi

    cache_promotion_cleanup() {
        pkill -f "dmlx serve" 2>/dev/null || true
        sleep 1
    }
    trap cache_promotion_cleanup EXIT INT TERM

    FIXED_ARTIFACTS=()
    LRU_ARTIFACTS=()
    echo "════════════════════════════════════════════════"
    echo "  Task 6.5 cache-default promotion gate"
    echo "  repeats/group: ${CACHE_PROMOTION_REPEATS}"
    echo "  order: fixed1, lru1, fixed2, lru2, ..."
    echo "════════════════════════════════════════════════"

    for run_index in $(seq 1 "$CACHE_PROMOTION_REPEATS"); do
        fixed_json="${CACHE_PROMOTION_ARTIFACT_DIR}/performance-cache-promotion-fixed-run${run_index}.json"
        lru_json="${CACHE_PROMOTION_ARTIFACT_DIR}/performance-cache-promotion-lru-run${run_index}.json"
        FIXED_ARTIFACTS+=("$fixed_json")
        LRU_ARTIFACTS+=("$lru_json")

        echo "▶ fixed-hot run ${run_index}/${CACHE_PROMOTION_REPEATS}"
        cache_promotion_cleanup
        env NATIVE_SMELT_N=20 NATIVE_EXPERIMENTAL_CACHE=0 NATIVE_PREFETCH=0 NATIVE_SMELT_PREFETCH=0 \
            NATIVE_BATCH_PREFILL=0 NATIVE_BATCH_MLA=0 NATIVE_EXPERT_BUCKETS=0 \
            NATIVE_BATCH_EXPERT_KERNELS=0 NATIVE_DSPARK_BATCH_VERIFY=0 DSPARK_DIR= \
            BM_MODEL_FINGERPRINT="$CACHE_PROMOTION_MODEL_FINGERPRINT" \
            BENCHMARK_JSON="$fixed_json" SKIP_PURGE="${SKIP_PURGE:-0}" \
            bash "$0" --native "$CACHE_PROMOTION_MODEL_PATH" || exit 1
        cache_promotion_cleanup

        echo "▶ bounded-LRU run ${run_index}/${CACHE_PROMOTION_REPEATS}"
        env NATIVE_SMELT_N=20 NATIVE_EXPERIMENTAL_CACHE=1 NATIVE_PREFETCH=0 NATIVE_SMELT_PREFETCH=0 \
            NATIVE_BATCH_PREFILL=0 NATIVE_BATCH_MLA=0 NATIVE_EXPERT_BUCKETS=0 \
            NATIVE_BATCH_EXPERT_KERNELS=0 NATIVE_DSPARK_BATCH_VERIFY=0 DSPARK_DIR= \
            BM_MODEL_FINGERPRINT="$CACHE_PROMOTION_MODEL_FINGERPRINT" \
            BENCHMARK_JSON="$lru_json" SKIP_PURGE="${SKIP_PURGE:-0}" \
            bash "$0" --native "$CACHE_PROMOTION_MODEL_PATH" || exit 1
        cache_promotion_cleanup
    done

    CACHE_PROMOTION_ARGS=(--cache-promotion-output "$CACHE_PROMOTION_SUMMARY")
    for artifact in "${FIXED_ARTIFACTS[@]}"; do
        CACHE_PROMOTION_ARGS+=(--cache-promotion-fixed "$artifact")
    done
    for artifact in "${LRU_ARTIFACTS[@]}"; do
        CACHE_PROMOTION_ARGS+=(--cache-promotion-candidate "$artifact")
    done
    python3 "${DIR}/benchmark_report.py" "${CACHE_PROMOTION_ARGS[@]}" || exit 1
    python3 "${DIR}/benchmark_report.py" --validate-cache-promotion "$CACHE_PROMOTION_SUMMARY" || exit 1
    exit 0
fi

# Task 5.5: repeated, strictly serial scalar-vs-fully-batched promotion gate.
# Each child benchmark retains an independent artifact; the aggregate validator
# rejects configuration contamination and stale/tampered source files.
if [[ "${1:-}" == "--promotion-gate" ]]; then
    shift
    PROMOTION_MODEL_PATH="${1:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
    PROMOTION_DSPARK_DIR="${DSPARK_DIR:-${HOME}/models/DeepSeek-V4-Flash-DSpark/dspark_weights}"
    PROMOTION_REPEATS="${PROMOTION_REPEATS:-2}"
    PROMOTION_ARTIFACT_DIR="${PROJECT_DIR}/docs/en/analysis"
    PROMOTION_SUMMARY="${PROMOTION_ARTIFACT_DIR}/performance-batch-promotion.json"

    if ! [[ "$PROMOTION_REPEATS" =~ ^[0-9]+$ ]] || (( PROMOTION_REPEATS < 2 )); then
        echo "❌ PROMOTION_REPEATS must be an integer >= 2"
        exit 1
    fi
    if [[ "$(basename "$PROMOTION_DSPARK_DIR")" != "dspark_weights" ||
          ! -d "$PROMOTION_DSPARK_DIR" ||
          ! -d "$(dirname "$PROMOTION_DSPARK_DIR")/packed_mtp_experts" ]]; then
        echo "❌ promotion candidate requires DSPARK_DIR=<...>/dspark_weights with sibling packed_mtp_experts"
        exit 1
    fi

    promotion_cleanup() {
        pkill -f "dmlx serve" 2>/dev/null || true
        sleep 1
    }
    trap promotion_cleanup EXIT INT TERM
    promotion_cleanup

    SCALAR_ARTIFACTS=()
    CANDIDATE_ARTIFACTS=()
    echo "════════════════════════════════════════════════"
    echo "  Task 5.5 batch-default promotion gate"
    echo "  repeats/group: ${PROMOTION_REPEATS}"
    echo "  SMELT N: 20"
    echo "  order: V4 oracle, scalar1, fully-batched1, scalar2, fully-batched2, ..."
    echo "════════════════════════════════════════════════"

    PROMOTION_ORACLE="${PROMOTION_ARTIFACT_DIR}/performance-promotion-oracle.json"
    echo "▶ V4 batch oracle + model fingerprint"
    BATCH_ORACLE_JSON="$PROMOTION_ORACLE" \
        NATIVE_SMELT_N=20 bash "${DIR}/run_batch_oracle.sh" || exit 1
    promotion_cleanup
    PROMOTION_MODEL_FINGERPRINT=$(python3 - "$PROMOTION_ORACLE" <<'PY'
import json
import sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value["configuration"]["model_fingerprint"])
PY
)
    if ! [[ "$PROMOTION_MODEL_FINGERPRINT" =~ ^[0-9a-f]{64}$ ]]; then
        echo "❌ V4 evidence did not provide a valid model fingerprint"
        exit 1
    fi

    for run_index in $(seq 1 "$PROMOTION_REPEATS"); do
        scalar_json="${PROMOTION_ARTIFACT_DIR}/performance-promotion-scalar-run${run_index}.json"
        candidate_json="${PROMOTION_ARTIFACT_DIR}/performance-promotion-batched-run${run_index}.json"
        SCALAR_ARTIFACTS+=("$scalar_json")
        CANDIDATE_ARTIFACTS+=("$candidate_json")

        echo "▶ scalar run ${run_index}/${PROMOTION_REPEATS}"
        promotion_cleanup
        env \
            NATIVE_SMELT_N=20 \
            NATIVE_BATCH_PREFILL=0 \
            NATIVE_BATCH_MLA=0 \
            NATIVE_EXPERT_BUCKETS=0 \
            NATIVE_BATCH_EXPERT_KERNELS=0 \
            NATIVE_DSPARK_BATCH_VERIFY=0 \
            BM_MODEL_FINGERPRINT="$PROMOTION_MODEL_FINGERPRINT" \
            DSPARK_DIR= \
            BENCHMARK_JSON="$scalar_json" \
            SKIP_PURGE="${SKIP_PURGE:-0}" \
            bash "$0" --native "$PROMOTION_MODEL_PATH" || exit 1
        promotion_cleanup

        echo "▶ fully-batched run ${run_index}/${PROMOTION_REPEATS}"
        env \
            NATIVE_SMELT_N=20 \
            NATIVE_BATCH_PREFILL=1 \
            NATIVE_BATCH_MLA=1 \
            NATIVE_EXPERT_BUCKETS=1 \
            NATIVE_BATCH_EXPERT_KERNELS=1 \
            NATIVE_DSPARK_BATCH_VERIFY=1 \
            BM_MODEL_FINGERPRINT="$PROMOTION_MODEL_FINGERPRINT" \
            DSPARK_DIR="$PROMOTION_DSPARK_DIR" \
            BENCHMARK_JSON="$candidate_json" \
            SKIP_PURGE="${SKIP_PURGE:-0}" \
            bash "$0" --native "$PROMOTION_MODEL_PATH" || exit 1
        promotion_cleanup
    done

    PROMOTION_ARGS=(--promotion-output "$PROMOTION_SUMMARY" --promotion-oracle "$PROMOTION_ORACLE")
    for artifact in "${SCALAR_ARTIFACTS[@]}"; do
        PROMOTION_ARGS+=(--promotion-scalar "$artifact")
    done
    for artifact in "${CANDIDATE_ARTIFACTS[@]}"; do
        PROMOTION_ARGS+=(--promotion-candidate "$artifact")
    done
    python3 "${DIR}/benchmark_report.py" "${PROMOTION_ARGS[@]}" || exit 1
    python3 "${DIR}/benchmark_report.py" --validate-promotion "$PROMOTION_SUMMARY" || exit 1

    python3 - "$PROMOTION_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
decision = report["decision"]
comparison = report["comparison"]["candidate_vs_scalar"]
print("════════════════════════════════════════════════")
print("  Promotion decision:", "PASS" if decision["promote_fully_batched_defaults"] else "FAIL — retain scalar/off defaults")
print("  standard delta: %.3f%%" % comparison["standard_tokens_per_second_delta_percent"])
print("  long diagnostic delta: %.3f%%" % comparison["long_diagnostic_tokens_per_second_delta_percent"])
print("  failed gates:", ", ".join(decision["failed_gates"]) or "none")
print("  JSON:", sys.argv[1])
print("════════════════════════════════════════════════")
PY
    exit 0
fi

# Prefer the standalone Command Line Tools SDK for non-interactive builds when
# the caller did not explicitly select a developer directory. This avoids
# coupling benchmark reproducibility to the full Xcode application's license
# state while still using an installed Apple SDK.
if [[ -z "${DEVELOPER_DIR:-}" && -d "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk" ]]; then
    export DEVELOPER_DIR="/Library/Developer/CommandLineTools"
fi

# Detect mode
MODE="native"
if [[ "${1:-}" == "--mlx" ]]; then
    MODE="mlx"
    shift
elif [[ "${1:-}" == "--native" ]]; then
    MODE="native"
    shift
fi

MODEL_PATH="${1:-${HOME}/models/DeepSeek-V4-Flash-4bit}"
PACKED_DIR="${MODEL_PATH}/packed_experts"
NATIVE_SMELT_N="${NATIVE_SMELT_N:-20}"
NATIVE_EXPERIMENTAL_CACHE_EFFECTIVE=$([[ "${NATIVE_EXPERIMENTAL_CACHE:-0}" == "1" ]] && echo 1 || echo 0)
NATIVE_BATCH_PREFILL_EFFECTIVE=$([[ "${NATIVE_BATCH_PREFILL:-0}" == "1" ]] && echo 1 || echo 0)
NATIVE_BATCH_MLA_EFFECTIVE=$([[ "${NATIVE_BATCH_MLA:-0}" == "1" ]] && echo 1 || echo 0)
NATIVE_EXPERT_BUCKETS_EFFECTIVE=$([[ "${NATIVE_EXPERT_BUCKETS:-0}" == "1" ]] && echo 1 || echo 0)
NATIVE_BATCH_EXPERT_KERNELS_EFFECTIVE=$([[ "${NATIVE_BATCH_EXPERT_KERNELS:-0}" == "1" ]] && echo 1 || echo 0)
NATIVE_DSPARK_BATCH_VERIFY_EFFECTIVE=$([[ "${NATIVE_DSPARK_BATCH_VERIFY:-0}" == "1" ]] && echo 1 || echo 0)
DSPARK_DIR="${DSPARK_DIR:-}"
DSPARK_ARGS=()
if [[ -n "$DSPARK_DIR" ]]; then
    DSPARK_ARGS=(--dspark "$DSPARK_DIR")
fi
validate_dspark_layout() {
    local weights_dir="$1"
    [[ -n "$weights_dir" &&
       "$(basename "$weights_dir")" == "dspark_weights" &&
       -d "$weights_dir" &&
       -d "$(dirname "$weights_dir")/packed_mtp_experts" ]]
}
if [[ "${1:-}" == "--self-test-dspark-layout" ]]; then
    SELF_TEST_ROOT=$(mktemp -d)
    trap 'rm -rf "$SELF_TEST_ROOT"' EXIT
    mkdir -p "$SELF_TEST_ROOT/valid/dspark_weights"
    if validate_dspark_layout "$SELF_TEST_ROOT/valid/dspark_weights"; then
        echo "❌ DSpark layout self-test accepted missing packed_mtp_experts"
        exit 1
    fi
    mkdir -p "$SELF_TEST_ROOT/valid/packed_mtp_experts" "$SELF_TEST_ROOT/wrong_name"
    if validate_dspark_layout "$SELF_TEST_ROOT/wrong_name"; then
        echo "❌ DSpark layout self-test accepted wrong weights basename"
        exit 1
    fi
    if ! validate_dspark_layout "$SELF_TEST_ROOT/valid/dspark_weights"; then
        echo "❌ DSpark layout self-test rejected valid sibling layout"
        exit 1
    fi
    echo "DSpark benchmark layout self-test: PASS"
    exit 0
fi
if [[ "$NATIVE_DSPARK_BATCH_VERIFY_EFFECTIVE" == "1" ]]; then
    if ! validate_dspark_layout "$DSPARK_DIR"; then
        echo "❌ NATIVE_DSPARK_BATCH_VERIFY=1 requires DSPARK_DIR=<...>/dspark_weights with sibling packed_mtp_experts"
        exit 1
    fi
fi

# MLX-mode legacy params
SMELT_EXPERTS="${2:-0.20}"
CACHE_MB="${3:-0}"
EXPERT_PARALLEL="${4:-18}"

# Purge page cache for clean baseline (requires sudo, skip if not available).
if command -v purge &>/dev/null && [[ "${SKIP_PURGE:-0}" != "1" ]]; then
    sudo -n purge 2>/dev/null || echo "⚠️  sudo purge skipped (no password) — results may vary"
fi

export BM_COMMIT=$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo "?")
export BM_BRANCH=$(git -C "$PROJECT_DIR" branch --show-current 2>/dev/null || echo "?")
export BM_DATE=$(date +%Y-%m-%d)
export BM_HW=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "?")
export BM_MEM=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0fGB",$1/1073741824}' || echo "?")

T0=$(date +%s)

echo "════════════════════════════════════════════════"
echo "  dmlx Benchmark  mode=${MODE}  commit=${BM_COMMIT}"
echo "  hw: ${BM_HW} ${BM_MEM}"
echo "  model: ${MODEL_PATH}"
echo "  packed experts: ${PACKED_DIR}"
echo "  developer dir: ${DEVELOPER_DIR:-system default}"
if [[ "${MODE}" == "native" ]]; then
    echo "  SMELT N: ${NATIVE_SMELT_N}"
    echo "  cache policy: $([[ "$NATIVE_EXPERIMENTAL_CACHE_EFFECTIVE" == "1" ]] && echo lru_observed_hot_seeded || echo fixed_hot_observed) (NATIVE_EXPERIMENTAL_CACHE=${NATIVE_EXPERIMENTAL_CACHE_EFFECTIVE})"
    echo "  batch flags: prefill=${NATIVE_BATCH_PREFILL_EFFECTIVE} mla=${NATIVE_BATCH_MLA_EFFECTIVE} buckets=${NATIVE_EXPERT_BUCKETS_EFFECTIVE} expert_kernels=${NATIVE_BATCH_EXPERT_KERNELS_EFFECTIVE} dspark_verify=${NATIVE_DSPARK_BATCH_VERIFY_EFFECTIVE}"
    echo "  DSpark weights: ${DSPARK_DIR:-disabled}"
    echo "  standard: warmup=5+4, measured=3, max_tokens=5"
    echo "  diagnostic: measured=1, max_tokens=100 (excluded from standard median)"
fi
echo "════════════════════════════════════════════════"

# ------------------------------------------------------------------
# Phase 0/1: Build and unit tests. Task 0.3 freezes one prevalidated binary
# across all four children so physical binary identity is comparable.
# ------------------------------------------------------------------
if [[ "${SKIP_BENCHMARK_BUILD:-0}" == "1" ]]; then
    [[ -x "$CLI" ]] || { echo "❌ frozen benchmark binary is missing"; exit 1; }
    export BM_UNIT="PASS (prevalidated ReleaseFast)"
    echo "🔧 Build skipped (using Task 0.3 frozen ReleaseFast binary)"
    echo "🧪 Unit tests: $BM_UNIT"
else
    echo "🔧 Build (clean + ReleaseFast)..."
    (cd "$PROJECT_DIR" && rm -rf .zig-cache && zig build -Doptimize=ReleaseFast 2>/dev/null) || {
        echo "❌ Build failed"; exit 1
    }

    echo "🧪 Unit tests..."
    T_UNIT=$(date +%s)
    if (cd "$PROJECT_DIR" && zig build test >/dev/null 2>&1); then
        export BM_UNIT="PASS (430+)"
    else
        export BM_UNIT="FAIL"
    fi
    echo "   $BM_UNIT ($(($(date +%s)-T_UNIT))s)"

    # Rebuild after test (zig build test compiles in Debug, overwrites binary)
    (cd "$PROJECT_DIR" && zig build -Doptimize=ReleaseFast 2>/dev/null)
fi

# Cleanup helper
cleanup() {
    if [[ -n "${RSS_SAMPLER_PID:-}" ]]; then
        kill "$RSS_SAMPLER_PID" 2>/dev/null || true
        wait "$RSS_SAMPLER_PID" 2>/dev/null || true
        RSS_SAMPLER_PID=""
    fi
    # Workspace invariant: every start is preceded by a global dmlx serve stop.
    # Use graceful shutdown first so routing stats can be saved, then enforce the
    # single-server rule regardless of port or prior script ownership.
    if curl -sf --max-time 3 -X POST "http://localhost:${PORT}/shutdown" > /dev/null 2>&1; then
        sleep 2
    fi
    pkill -f "dmlx serve" 2>/dev/null || true
    sleep 1
    # Delete routing stats before each benchmark run to ensure reproducible results.
    # With penalty=0 (natural routing), stats-based SMELT may hurt benchmark performance
    # for prompts not in the training stats. Phase 1 (default experts 0..N-1) gives
    # consistent baseline numbers. Production servers benefit from persistent stats;
    # benchmarks need fresh state for fair comparisons.
    rm -f "${PACKED_DIR}/.smelt_routing_stats.bin" 2>/dev/null || true
}

cleanup_all() {
    cleanup
    if [[ -n "${TELEMETRY_JSONL:-}" ]]; then
        rm -f "$TELEMETRY_JSONL"
    fi
}

telemetry_count() {
    if [[ ! -f "$SERVE_LOG" ]]; then
        echo 0
        return
    fi
    grep -c '^\[NATIVE_TELEMETRY\] ' "$SERVE_LOG" 2>/dev/null || true
}

append_latest_telemetry() {
    grep '^\[NATIVE_TELEMETRY\] ' "$SERVE_LOG" | tail -n 1 | \
        sed 's/^\[NATIVE_TELEMETRY\] //' >> "$TELEMETRY_JSONL"
}

trap cleanup_all EXIT
cleanup

# ==================================================================
# NATIVE MODE: safe SMELT N=20 by default, warmup + sequential requests
# ==================================================================
if [[ "$MODE" == "native" ]]; then
    echo "📊 Native engine perf test (SMELT N=${NATIVE_SMELT_N})..."

    # Check available memory
    AVAIL_GB=$(python3 -c "
import subprocess
out = subprocess.run(['vm_stat'],capture_output=True,text=True).stdout
pages={}
for line in out.split('\n'):
    for k in ['Pages free','Pages inactive','Pages speculative']:
        if k in line: pages[k]=int(line.split(':')[1].strip().rstrip('.'))
pg=16384
print(int((pages.get('Pages free',0)+pages.get('Pages inactive',0)+pages.get('Pages speculative',0))*pg/1024**3))
" 2>/dev/null || echo "0")
    NEEDED_GB=$(python3 -c "print(int(${NATIVE_SMELT_N} * 43 * 13.4 / 1024) + 5)")
    if (( AVAIL_GB < NEEDED_GB )); then
        echo "⚠️  Only ${AVAIL_GB}GB available, SMELT N=${NATIVE_SMELT_N} needs ~${NEEDED_GB}GB"
    fi

    T_PERF=$(date +%s)
    TELEMETRY_JSONL=$(mktemp)
    # Hard single-server gate immediately before every serve launch. The startup
    # clock intentionally begins after the mandatory safety delay.
    pkill -f "dmlx serve" 2>/dev/null || true
    sleep 1
    STARTUP_BEGIN_NS=$(python3 -c "import time; print(time.monotonic_ns())")
    NATIVE_TELEMETRY=1 NATIVE_SMELT_N="${NATIVE_SMELT_N}" NATIVE_EXPERIMENTAL_CACHE="${NATIVE_EXPERIMENTAL_CACHE_EFFECTIVE}" "$CLI" serve \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --native \
        --expert-packed-dir "$PACKED_DIR" \
        "${DSPARK_ARGS[@]+"${DSPARK_ARGS[@]}"}" \
        > "$SERVE_LOG" 2>&1 &
    SERVER_PID=$!
    RSS_SAMPLE_FILE=$(mktemp)
    echo 0 > "$RSS_SAMPLE_FILE"
    (
        peak=0
        while kill -0 "$SERVER_PID" 2>/dev/null; do
            rss_kib=$(ps -o rss= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')
            if [[ "$rss_kib" =~ ^[0-9]+$ ]] && (( rss_kib * 1024 > peak )); then
                peak=$((rss_kib * 1024)); echo "$peak" > "$RSS_SAMPLE_FILE"
            fi
            sleep 0.2
        done
    ) &
    RSS_SAMPLER_PID=$!

    # Wait for server ready (up to 180s for SMELT preload)
    echo -n "   Waiting for server (SMELT preload)..."
    STARTUP_SECS=0
    for i in $(seq 1 180); do
        if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
            STARTUP_SECS=$i
            echo " ready (${i}s)"
            break
        fi
        echo -n "."
        sleep 1
    done
    if ! curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
        echo -e "\n❌ Server failed to start"; tail -20 "$SERVE_LOG"; exit 1
    fi
    STARTUP_END_NS=$(python3 -c "import time; print(time.monotonic_ns())")
    STARTUP_MS=$(python3 -c "print(f'{($STARTUP_END_NS-$STARTUP_BEGIN_NS)/1_000_000:.3f}')")

    # Warmup: heat GPU caches — send 5 requests to fully warm pipeline state
    # (Metal PSO compilation + GPU thread warmup can take first 3-4 requests)
    echo "   Warmup (5 requests)..."
    for i in 1 2 3 4 5; do
        curl -sf --max-time 60 "${SERVER_URL}/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model":"d","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0}' \
            > /dev/null 2>&1 || { echo "❌ Warmup request failed; refusing ordinal drift"; exit 1; }
    done

    # Correctness: Paris (sequential, measured but not counted in perf)
    echo "   Correctness check..."
    PARIS_RESP=$(curl -s --max-time 120 "${SERVER_URL}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"d","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":10,"temperature":0}')
    PARIS_TEXT=$(echo "$PARIS_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
    if echo "$PARIS_TEXT" | grep -qi "paris"; then
        echo "   ✓ Paris correct: \"${PARIS_TEXT}\""
        NATIVE_CORRECT=1
    else
        echo "   ✗ Paris FAILED: \"${PARIS_TEXT}\""
        NATIVE_CORRECT=0
    fi

    # Re-warm GPU after Paris (Paris uses a different prompt, which can re-cold the pipeline)
    echo "   Re-warm after Paris (4 requests)..."
    for i in 1 2 3 4; do
        curl -sf --max-time 60 "${SERVER_URL}/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model":"d","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0}' \
            > /dev/null 2>&1 || { echo "❌ Warmup request failed; refusing ordinal drift"; exit 1; }
    done

    # Perf: 3 sequential requests, report median tok/s. Use response usage
    # counts rather than assuming every request generated five tokens.
    echo "   STANDARD_DECODE: 3 sequential runs, max_tokens=5 (median source only)..."
    NATIVE_TPS_LIST=()
    NATIVE_ELAPSED_LIST=()
    NATIVE_PROMPT_TOKEN_LIST=()
    NATIVE_TOKEN_LIST=()
    for i in 1 2 3; do
        TELEM_BEFORE=$(telemetry_count)
        T_REQ_NS=$(python3 -c "import time; print(time.monotonic_ns())")
        RESP=$(curl -s --max-time 60 "${SERVER_URL}/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model":"d","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0}')
        T_DONE_NS=$(python3 -c "import time; print(time.monotonic_ns())")
        RESP_TEXT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
        read -r N_PROMPT N_COMPLETION <<< "$(echo "$RESP" | python3 -c "import sys,json; u=json.load(sys.stdin).get('usage',{}); print(int(u.get('prompt_tokens',0)), int(u.get('completion_tokens',0)))" 2>/dev/null || echo '0 0')"
        if ! [[ "$N_PROMPT" =~ ^[0-9]+$ && "$N_COMPLETION" =~ ^[0-9]+$ ]] || (( N_PROMPT <= 0 || N_COMPLETION != 5 )); then
            echo "❌ Invalid STANDARD_DECODE usage for run${i}: prompt=${N_PROMPT:-?} completion=${N_COMPLETION:-?}"
            exit 1
        fi
        TELEM_AFTER=$(telemetry_count)
        if (( TELEM_AFTER != TELEM_BEFORE + 1 )); then
            echo "❌ STANDARD_DECODE run${i} emitted $((TELEM_AFTER-TELEM_BEFORE)) telemetry records (expected 1)"
            exit 1
        fi
        if ! append_latest_telemetry; then
            echo "❌ Failed to capture STANDARD_DECODE run${i} telemetry"
            exit 1
        fi
        ELAPSED=$(python3 -c "print(f'{($T_DONE_NS-$T_REQ_NS)/1_000_000_000:.6f}')")
        TPS=$(python3 -c "print(f'{$N_COMPLETION/$ELAPSED:.6f}')")
        DISPLAY_ELAPSED=$(python3 -c "print(f'{$ELAPSED:.1f}')")
        DISPLAY_TPS=$(python3 -c "print(f'{$TPS:.3f}')")
        echo "   run${i}: ${DISPLAY_ELAPSED}s | ${DISPLAY_TPS} tok/s | prompt=${N_PROMPT} completion=${N_COMPLETION} | \"${RESP_TEXT}\""
        NATIVE_TPS_LIST+=("$TPS")
        NATIVE_ELAPSED_LIST+=("$ELAPSED")
        NATIVE_PROMPT_TOKEN_LIST+=("$N_PROMPT")
        NATIVE_TOKEN_LIST+=("$N_COMPLETION")
    done
    NATIVE_MEDIAN_TPS=$(python3 -c "
vals=sorted([${NATIVE_TPS_LIST[0]},${NATIVE_TPS_LIST[1]},${NATIVE_TPS_LIST[2]}])
print(f'{vals[1]:.6f}')")
    echo "   Standard median (5-token runs only): $(python3 -c "print(f'{${NATIVE_MEDIAN_TPS}:.3f}')") tok/s"

    # Long-decode diagnostic: 1 request x at most 100 tokens. This is reported
    # separately and is never included in the standard 5-token median.
    echo "   LONG_DECODE_DIAGNOSTIC: 1 run, max_tokens=100 (excluded from standard median)..."
    T_REQ_NS=$(python3 -c "import time; print(time.monotonic_ns())")
    RESP=$(curl -s --max-time 600 "${SERVER_URL}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"d","messages":[{"role":"user","content":"Count from 1 to 50."}],"max_tokens":100,"temperature":0}')
    T_DONE_NS=$(python3 -c "import time; print(time.monotonic_ns())")
    read -r LONG_PROMPT_TOKENS N_COMPLETION <<< "$(echo "$RESP" | python3 -c "import sys,json; u=json.load(sys.stdin).get('usage',{}); print(int(u.get('prompt_tokens',0)), int(u.get('completion_tokens',0)))" 2>/dev/null || echo '0 0')"
    if ! [[ "$LONG_PROMPT_TOKENS" =~ ^[0-9]+$ && "$N_COMPLETION" =~ ^[0-9]+$ ]] || (( LONG_PROMPT_TOKENS <= 0 || N_COMPLETION <= 0 || N_COMPLETION > 100 )); then
        echo "❌ Invalid LONG_DECODE_DIAGNOSTIC usage: prompt=${LONG_PROMPT_TOKENS:-?} completion=${N_COMPLETION:-?}"
        exit 1
    fi
    LONG_ELAPSED=$(python3 -c "print(f'{($T_DONE_NS-$T_REQ_NS)/1_000_000_000:.6f}')")
    NATIVE_DECODE_TPS=$(python3 -c "print(f'{$N_COMPLETION/$LONG_ELAPSED:.6f}')")
    echo "   long-decode diagnostic: $(python3 -c "print(f'{$LONG_ELAPSED:.1f}')")s | $(python3 -c "print(f'{$NATIVE_DECODE_TPS:.3f}')") tok/s (prompt=${LONG_PROMPT_TOKENS}, completion=${N_COMPLETION})"

    if [[ -n "${RSS_SAMPLER_PID:-}" ]]; then
        kill "$RSS_SAMPLER_PID" 2>/dev/null || true
        wait "$RSS_SAMPLER_PID" 2>/dev/null || true
    fi
    PERF_PEAK_RSS_BYTES=$(cat "$RSS_SAMPLE_FILE" 2>/dev/null || echo 0)
    rm -f "$RSS_SAMPLE_FILE"
    if ! [[ "$PERF_PEAK_RSS_BYTES" =~ ^[0-9]+$ ]] || (( PERF_PEAK_RSS_BYTES <= 0 )); then
        echo "❌ Failed to capture native peak RSS"
        exit 1
    fi
    python3 - "$TELEMETRY_JSONL" "$PERF_PEAK_RSS_BYTES" <<'PY'
import json, sys
from pathlib import Path
path=Path(sys.argv[1]); peak=int(sys.argv[2]); records=[]
for line in path.read_text(encoding='utf-8').splitlines():
    record=json.loads(line); record['dispatch']['peak_rss_bytes']=peak; records.append(record)
path.write_text(''.join(json.dumps(record,separators=(',',':'))+'\n' for record in records),encoding='utf-8')
PY

    export BM_NATIVE_TPS="$NATIVE_MEDIAN_TPS"
    export BM_NATIVE_CORRECT="$NATIVE_CORRECT"
    export BM_PERF_SECS=$(($(date +%s)-T_PERF))
    export BM_SMELT_N="$NATIVE_SMELT_N"

    # Restart the server before E2E: after the perf phase the process has spent
    # ~250s streaming ~3.4GB/token of expert data through the page cache, and on
    # a 48GB machine the accumulated pressure gets the server Jetsam-killed
    # mid-E2E (observed 2/2 at P7, SIGKILL "Killed: 9"). A fresh process halves
    # the per-process lifetime and cache churn; perf numbers are already captured.
    echo "   Restarting server before E2E (avoid Jetsam kill from accumulated memory pressure)..."
    cleanup
    # Verify the default-off path in a fresh process; do not inherit a caller's
    # NATIVE_TELEMETRY setting into the E2E correctness phase.
    pkill -f "dmlx serve" 2>/dev/null || true
    sleep 1
    env -u NATIVE_TELEMETRY -u NATIVE_EXACT_Q4_CAPTURE -u NATIVE_EXACT_Q4_RAW_JSONL NATIVE_SMELT_N="${NATIVE_SMELT_N}" NATIVE_EXPERIMENTAL_CACHE="${NATIVE_EXPERIMENTAL_CACHE_EFFECTIVE}" "$CLI" serve \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --native \
        --expert-packed-dir "$PACKED_DIR" \
        "${DSPARK_ARGS[@]+"${DSPARK_ARGS[@]}"}" \
        > "$SERVE_LOG" 2>&1 &
    echo -n "   Waiting for server (SMELT preload)..."
    for i in $(seq 1 180); do
        if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
            echo " ready (${i}s)"
            break
        fi
        echo -n "."
        sleep 1
    done
    if ! curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
        echo -e "\n❌ Server failed to restart"; tail -20 "$SERVE_LOG"; exit 1
    fi

    # --- E2E: 7-Prompt correctness test via native serve mode ---
    echo ""
    echo "✅ E2E (7 prompts via native serve)..."
    T_E2E=$(date +%s)

    PROMPTS=(
        "2+2=|4"
        "The capital of France is|Paris"
        "What temperature does water freeze at in Celsius? Just give the number.|0"
        "Is the Earth round? Reply with only yes or no.|yes"
        "3*3=|9"
        "10-5=|5"
        "What is capital of France?|Paris"
    )

    EF=$(mktemp)
    E2E_PASS=0
    E2E_FAIL=0

    for idx in "${!PROMPTS[@]}"; do
        IFS='|' read -r prompt expected <<< "${PROMPTS[$idx]}"

        result=$(curl -sf --max-time 300 \
            "${SERVER_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":30,\"temperature\":0}" 2>&1)

        content=$(echo "$result" | jq -r '.choices[0].message.content // ""' 2>/dev/null || echo "")

        if echo "$content" | grep -qi "$expected"; then
            echo "   P$((idx+1)): ✅ PASSED"
            echo "✅ PASSED P$((idx+1)): ${prompt}" >> "$EF"
            echo "   Generated: ${content:0:80}" >> "$EF"
            E2E_PASS=$((E2E_PASS + 1))
        else
            echo "   P$((idx+1)): ❌ FAILED (expected '${expected}' in output)"
            echo "❌ FAILED P$((idx+1)): ${prompt}" >> "$EF"
            echo "   Generated: ${content:0:80}" >> "$EF"
            E2E_FAIL=$((E2E_FAIL + 1))
        fi
        sleep 1
    done

    export BM_E2E_SECS=$(($(date +%s)-T_E2E))
    echo "   Results: ${E2E_PASS} passed, ${E2E_FAIL} failed (${BM_E2E_SECS}s)"

    if (( $(telemetry_count) != 0 )); then
        echo "❌ E2E default-off server unexpectedly emitted native telemetry"
        exit 1
    fi
    echo "   ✓ E2E telemetry default-off verified"

    cleanup

    # Emit the authoritative schema-v1 artifact before the human-readable
    # report. Server-side TTFT/ITL/expert-I/O/dispatch metrics come only from
    # the three standard telemetry records; unavailable RSS/workspace fields stay null.
    export BM_TOTAL_SECS=$(($(date +%s)-T0))
    export BM_MODE="native"
    export BM_MODEL_PATH="$MODEL_PATH"
    export BM_PACKED_DIR="$PACKED_DIR"
    export BM_DSPARK_DIR="$DSPARK_DIR"
    export BM_WARMUP_INITIAL="5"
    export BM_WARMUP_REWARM="4"
    export BM_STANDARD_TPS_CSV="$(IFS=,; echo "${NATIVE_TPS_LIST[*]}")"
    export BM_STANDARD_ELAPSED_CSV="$(IFS=,; echo "${NATIVE_ELAPSED_LIST[*]}")"
    export BM_STANDARD_PROMPT_TOKENS_CSV="$(IFS=,; echo "${NATIVE_PROMPT_TOKEN_LIST[*]}")"
    export BM_STANDARD_TOKENS_CSV="$(IFS=,; echo "${NATIVE_TOKEN_LIST[*]}")"
    export BM_LONG_PROMPT_TOKENS="$LONG_PROMPT_TOKENS"
    export BM_LONG_TOKENS="$N_COMPLETION"
    export BM_LONG_ELAPSED="$LONG_ELAPSED"
    export BM_NATIVE_DECODE_TPS="$NATIVE_DECODE_TPS"
    export BM_NATIVE_CORRECT="$NATIVE_CORRECT"
    export BM_E2E_PASS="$E2E_PASS"
    export BM_E2E_FAIL="$E2E_FAIL"
    export BM_STARTUP_SECS="$STARTUP_SECS"
    export BM_STARTUP_MS="$STARTUP_MS"
    export BM_TELEMETRY_ENABLED="1"
    export BM_EXPERIMENTAL_CACHE="$NATIVE_EXPERIMENTAL_CACHE_EFFECTIVE"
    export BM_PREFETCH="${NATIVE_PREFETCH:-0}"
    export BM_SMELT_PREFETCH="${NATIVE_SMELT_PREFETCH:-0}"
    export BM_SKIP_PURGE="${SKIP_PURGE:-0}"
    export BM_BATCH_PREFILL="$NATIVE_BATCH_PREFILL_EFFECTIVE"
    export BM_BATCH_MLA="$NATIVE_BATCH_MLA_EFFECTIVE"
    export BM_EXPERT_BUCKETS="$NATIVE_EXPERT_BUCKETS_EFFECTIVE"
    export BM_BATCH_EXPERT_KERNELS="$NATIVE_BATCH_EXPERT_KERNELS_EFFECTIVE"
    export BM_DSPARK_BATCH_VERIFY="$NATIVE_DSPARK_BATCH_VERIFY_EFFECTIVE"
    export BM_TELEMETRY_JSONL="$TELEMETRY_JSONL"
    printf -v BM_COMMAND 'NATIVE_SMELT_N=%q NATIVE_EXPERIMENTAL_CACHE=%q NATIVE_BATCH_PREFILL=%q NATIVE_BATCH_MLA=%q NATIVE_EXPERT_BUCKETS=%q NATIVE_BATCH_EXPERT_KERNELS=%q NATIVE_DSPARK_BATCH_VERIFY=%q DSPARK_DIR=%q SKIP_PURGE=%q bash %q --native %q' \
        "$NATIVE_SMELT_N" "$NATIVE_EXPERIMENTAL_CACHE_EFFECTIVE" "$NATIVE_BATCH_PREFILL_EFFECTIVE" "$NATIVE_BATCH_MLA_EFFECTIVE" \
        "$NATIVE_EXPERT_BUCKETS_EFFECTIVE" "$NATIVE_BATCH_EXPERT_KERNELS_EFFECTIVE" \
        "$NATIVE_DSPARK_BATCH_VERIFY_EFFECTIVE" "$DSPARK_DIR" "${SKIP_PURGE:-0}" \
        "scripts/run_benchmark.sh" "$MODEL_PATH"
    export BM_COMMAND

    if [[ -n "${BENCHMARK_JSON:-}" ]]; then
        RESULT_JSON="$BENCHMARK_JSON"
    elif [[ "$NATIVE_DSPARK_BATCH_VERIFY_EFFECTIVE" == "1" ]]; then
        # Preserve the canonical default-off artifact as the matched baseline;
        # DSpark evidence must never overwrite the comparison it cites.
        RESULT_JSON="${PROJECT_DIR}/docs/en/analysis/performance-benchmark-dspark.json"
    else
        RESULT_JSON="${PROJECT_DIR}/docs/en/analysis/performance-benchmark.json"
    fi
    if ! python3 "$DIR/benchmark_report.py" --output "$RESULT_JSON"; then
        FAILED_TELEMETRY="/tmp/dmlx_benchmark_telemetry_failed.jsonl"
        cp "$TELEMETRY_JSONL" "$FAILED_TELEMETRY" 2>/dev/null || true
        echo "❌ Machine-readable benchmark generation failed (telemetry preserved at $FAILED_TELEMETRY)"
        rm -f "$EF"
        exit 1
    fi
    if ! python3 "$DIR/benchmark_report.py" --validate "$RESULT_JSON"; then
        echo "❌ Machine-readable benchmark validation failed"
        rm -f "$EF"
        exit 1
    fi

    # Report
    echo ""
    echo "════════════════════════════════════════"
    echo "  Native Engine Benchmark Results"
    echo "  commit:  ${BM_COMMIT}"
    echo "  SMELT N: ${NATIVE_SMELT_N}"
    echo "  tok/s:   $(python3 -c "print(f'{${NATIVE_MEDIAN_TPS}:.3f}')") (median of 3 sequential)"
    echo "  decode:  $(python3 -c "print(f'{${NATIVE_DECODE_TPS}:.3f}')") tok/s (long diagnostic)"
    echo "  Paris:   $([ "$NATIVE_CORRECT" -eq 1 ] && echo '✓ PASS' || echo '✗ FAIL')"
    echo "  E2E:     ${E2E_PASS}/7 passed"
    echo "  unit:    ${BM_UNIT}"
    echo "  time:    ${BM_PERF_SECS}s"
    echo "  JSON:    ${RESULT_JSON}"
    echo "════════════════════════════════════════"

    # Refresh the Markdown report from the same schema-v1 artifact. The E2E
    # file remains an argument for legacy generator compatibility, but native
    # metrics and correctness come exclusively from RESULT_JSON.
    python3 "$DIR/_gen_report.py" "$RESULT_JSON" "$EF" "$REPORT" || echo "⚠️ report generation failed"
    rm -f "$EF"
    exit $([ "$NATIVE_CORRECT" -eq 1 ] && [ "$E2E_FAIL" -eq 0 ] && echo 0 || echo 1)
fi

# ==================================================================
# MLX MODE (legacy)
# ==================================================================
echo "📊 MLX serve mode perf test (smelt=${SMELT_EXPERTS}, cache=${CACHE_MB}MB)..."

# Validate packed expert directory for MLX mode
if [ ! -d "$PACKED_DIR" ]; then
    echo "⚠️  Packed expert directory not found: $PACKED_DIR"
    PACKED_DIR=""
    EXPERT_PARALLEL=0
fi

T_PERF=$(date +%s)
if [ -n "$PACKED_DIR" ] && [ -d "$PACKED_DIR" ]; then
    "$CLI" serve \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --max-tokens 256 \
        --temperature 0 \
        --smelt --smelt-strategy stream --smelt-experts "$SMELT_EXPERTS" \
        --smelt-cache "$CACHE_MB" \
        --expert-packed-dir "$PACKED_DIR" \
        --expert-parallel "$EXPERT_PARALLEL" > "$SERVE_LOG" 2>&1 &
else
    "$CLI" serve \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --max-tokens 256 \
        --temperature 0 \
        --smelt --smelt-strategy stream --smelt-experts "$SMELT_EXPERTS" \
        --smelt-cache "$CACHE_MB" > "$SERVE_LOG" 2>&1 &
fi
SERVER_PID=$!

# Wait for server ready
echo "   Waiting for server..."
STARTUP_SECS=0
for i in {1..180}; do
    if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
        STARTUP_SECS=$i
        echo "   Server ready (${i}s)"
        break
    fi
    sleep 1
done

if ! curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo "❌ Server failed to start"
    tail -30 "$SERVE_LOG"
    exit 1
fi

# --- Perf: Generate 30 tokens to measure steady-state latency ---
echo "   Generating 30 tokens (perf measurement)..."
PERF_RESULT=$(curl -sf -w '\n%{time_starttransfer}|%{time_total}' \
    --max-time 300 \
    "${SERVER_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"Hello"}],"max_tokens":30,"temperature":0}' 2>&1)

PERF_BODY=$(echo "$PERF_RESULT" | sed '$d')
PERF_TIMING=$(echo "$PERF_RESULT" | tail -1)
PERF_TTFR=$(echo "$PERF_TIMING" | cut -d'|' -f1)
PERF_TOTAL=$(echo "$PERF_TIMING" | cut -d'|' -f2)
PERF_TOKENS=$(echo "$PERF_BODY" | jq -r '.usage.completion_tokens // 0' 2>/dev/null || echo "0")

echo "   30-token: TTFR=${PERF_TTFR}s total=${PERF_TOTAL}s tokens=${PERF_TOKENS}"

# --- Perf: Generate 100 tokens to measure long-generation performance ---
echo "   Generating 100 tokens (long-gen measurement)..."
LONG_RESULT=$(curl -sf -w '\n%{time_starttransfer}|%{time_total}' \
    --max-time 600 \
    "${SERVER_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"Explain the concept of machine learning in simple terms."}],"max_tokens":100,"temperature":0}' 2>&1)

LONG_BODY=$(echo "$LONG_RESULT" | sed '$d')
LONG_TIMING=$(echo "$LONG_RESULT" | tail -1)
LONG_TTFR=$(echo "$LONG_TIMING" | cut -d'|' -f1)
LONG_TOTAL=$(echo "$LONG_TIMING" | cut -d'|' -f2)
LONG_TOKENS=$(echo "$LONG_BODY" | jq -r '.usage.completion_tokens // 0' 2>/dev/null || echo "0")

echo "   100-token: TTFR=${LONG_TTFR}s total=${LONG_TOTAL}s tokens=${LONG_TOKENS}"

# Extract token step data from server log
PF=$(mktemp)
grep "Token step.*complete" "$SERVE_LOG" > "$PF"
export BM_PERF_SECS=$(($(date +%s)-T_PERF))
echo "   Perf phase: ${BM_PERF_SECS}s"

# --- E2E: 7-Prompt correctness test via serve mode ---
echo "✅ E2E (7 prompts via serve)..."
T_E2E=$(date +%s)

PROMPTS=(
    "2+2=|4"
    "The capital of France is|Paris"
    "What temperature does water freeze at in Celsius? Just give the number.|0"
    "Is the Earth round? Reply with only yes or no.|yes"
    "3*3=|9"
    "10-5=|5"
    "What is capital of France?|Paris"
)

EF=$(mktemp)
E2E_PASS=0
E2E_FAIL=0

for idx in "${!PROMPTS[@]}"; do
    IFS='|' read -r prompt expected <<< "${PROMPTS[$idx]}"
    
    result=$(curl -sf --max-time 300 \
        "${SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":30,\"temperature\":0}" 2>&1)
    
    content=$(echo "$result" | jq -r '.choices[0].message.content // ""' 2>/dev/null || echo "")
    
    if echo "$content" | grep -qi "$expected"; then
        echo "   P$((idx+1)): ✅ PASSED"
        echo "✅ PASSED P$((idx+1)): ${prompt}" >> "$EF"
        echo "   Generated: ${content:0:80}" >> "$EF"
        E2E_PASS=$((E2E_PASS + 1))
    else
        echo "   P$((idx+1)): ❌ FAILED (expected '${expected}' in output)"
        echo "❌ FAILED P$((idx+1)): ${prompt}" >> "$EF"
        echo "   Generated: ${content:0:80}" >> "$EF"
        E2E_FAIL=$((E2E_FAIL + 1))
    fi
    sleep 1
done

export BM_E2E_SECS=$(($(date +%s)-T_E2E))
echo "   Results: ${E2E_PASS} passed, ${E2E_FAIL} failed (${BM_E2E_SECS}s)"

# --- Extract server-side RequestLog metrics ---
echo ""
echo "📈 Server-side metrics:"
grep "RequestLog" "$SERVE_LOG" | tail -10

# --- Memory usage ---
SERVER_RSS=$(ps -o rss= -p $SERVER_PID 2>/dev/null | awk '{printf "%.0f", $1/1024}' || echo "0")
echo "   Server RSS: ${SERVER_RSS}MB"

# Stop server
cleanup

export BM_TOTAL_SECS=$(($(date +%s)-T0))

# ------------------------------------------------------------------
# Phase 3: Generate Report
# ------------------------------------------------------------------
echo ""
echo "📝 Generating report..."

# Export additional env vars for report
export BM_SMELT_EXPERTS="$SMELT_EXPERTS"
export BM_CACHE_MB="$CACHE_MB"
export BM_MLOCK="false"
export BM_E2E_PASS="$E2E_PASS"
export BM_E2E_FAIL="$E2E_FAIL"
export BM_PERF_TTFR="$PERF_TTFR"
export BM_PERF_TOTAL="$PERF_TOTAL"
export BM_PERF_TOKENS="$PERF_TOKENS"
export BM_LONG_TTFR="$LONG_TTFR"
export BM_LONG_TOTAL="$LONG_TOTAL"
export BM_LONG_TOKENS="$LONG_TOKENS"
export BM_PACKED_DIR="$PACKED_DIR"
export BM_PARALLEL="$EXPERT_PARALLEL"
export BM_SERVER_RSS="$SERVER_RSS"
export BM_STARTUP_SECS="$STARTUP_SECS"

python3 "$DIR/_gen_report.py" "$PF" "$EF" "$REPORT"
rm -f "$PF" "$EF"

echo "✅ Done → $REPORT (${BM_TOTAL_SECS}s total)"
