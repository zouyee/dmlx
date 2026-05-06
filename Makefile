# dmlx Makefile — Aggregates build, test, and verification commands.
#
# Usage:
#   make              — Build the CLI binary
#   make test         — Run Zig unit tests
#   make verify       — Run DeepSeek V4 verification suite
#   make e2e          — Run end-to-end correctness tests
#   make benchmark    — Run performance regression test
#   make check        — Run everything (build + test + verify + benchmark)
#
# Environment variables:
#   MODEL_PATH        — Path to DeepSeek V4 model (default: ~/models/DeepSeek-V4-Flash-4bit)
#   MLX_PREFIX        — Path to mlx-c installation (optional)
#   MAX_TTFT_MS       — TTFT threshold for benchmark (default: 500)
#   MAX_ITL_MS        — ITL threshold for benchmark (default: 150)

.PHONY: all build test clean verify e2e benchmark check

# Default target
all: build

# Paths
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
MODEL_PATH  ?= $(HOME)/models/DeepSeek-V4-Flash-4bit
CLI         := $(PROJECT_DIR)/zig-out/bin/dmlx

# Build options
ZIG_BUILD_OPTS := -Doptimize=ReleaseFast

# Metal stability (required for DeepSeek V4 on Apple Silicon)
export MLX_METAL_FAST_SYNCH := 1
ifdef MLX_PREFIX
	ZIG_BUILD_OPTS += -Dmlx_prefix=$(MLX_PREFIX)
endif

# ------------------------------------------------------------------
# Build
# ------------------------------------------------------------------
build:
	@echo "📦 Building dmlx..."
	cd "$(PROJECT_DIR)" && zig build $(ZIG_BUILD_OPTS)
	@echo "✅ Build complete: $(CLI)"

# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------
test: build
	@echo ""
	@echo "🧪 Running Zig unit tests..."
	cd "$(PROJECT_DIR)" && zig build test
	@echo "✅ Unit tests passed"

# ------------------------------------------------------------------
# Clean
# ------------------------------------------------------------------
clean:
	@echo "🧹 Cleaning build artifacts..."
	cd "$(PROJECT_DIR)" && rm -rf zig-out .zig-cache
	@echo "✅ Clean complete"

# ------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------
verify: build
	@echo ""
	@echo "🔍 Running DeepSeek V4 verification suite..."
	"$(PROJECT_DIR)/scripts/verify-deepseek-v4-fix.sh" "$(MODEL_PATH)"

# ------------------------------------------------------------------
# End-to-end correctness
# ------------------------------------------------------------------
e2e: build
	@echo ""
	@echo "🎯 Running end-to-end correctness tests..."
	"$(PROJECT_DIR)/scripts/e2e_correctness.sh" "$(MODEL_PATH)"

# ------------------------------------------------------------------
# Performance regression
# ------------------------------------------------------------------
benchmark: build
	@echo ""
	@echo "⚡ Running performance regression test..."
	"$(PROJECT_DIR)/scripts/performance_regression.sh" "$(MODEL_PATH)"

# ------------------------------------------------------------------
# End-to-end correctness tests (server mode — faster, model loads once)
# ------------------------------------------------------------------
e2e-server: build
	@echo ""
	@echo "🎯 Running end-to-end correctness tests (server mode)..."
	"$(PROJECT_DIR)/scripts/e2e_server.sh" "$(MODEL_PATH)"

# ------------------------------------------------------------------
# Upload benchmark results to benchmark-results branch
# ------------------------------------------------------------------
upload-benchmark: benchmark
	@echo ""
	@echo "🚀 Uploading benchmark results..."
	python3 "$(PROJECT_DIR)/scripts/upload_benchmark.py"

# ------------------------------------------------------------------
# Setup benchmark-results branch (run once per clone)
# ------------------------------------------------------------------
setup-benchmark-branch:
	@echo "🔧 Setting up benchmark-results branch..."
	@git branch benchmark-results 2>/dev/null || true
	@echo "Done. Run 'make upload-benchmark' after running 'make benchmark'."

# ------------------------------------------------------------------
# Full check — everything (local only)
# ------------------------------------------------------------------
check: test verify e2e benchmark
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "✅ ALL CHECKS PASSED"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Completed:"
	@echo "  • Zig unit tests"
	@echo "  • DeepSeek V4 verification (chat template + greedy correctness)"
	@echo "  • End-to-end correctness (math, geography, common knowledge)"
	@echo "  • Performance regression (TTFT + ITL thresholds)"
	@echo ""
	@echo "To upload results to dashboard: make upload-benchmark"
