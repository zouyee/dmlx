# Performance Benchmarking & Dashboard

This document describes the performance benchmarking infrastructure for mlx-zig.

## Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐
│  Local Mac  │────▶│ benchmark-   │────▶│ GitHub Pages        │
│  (make      │     │ results      │     │ (auto-deployed)     │
│   benchmark)│     │ branch       │     │                     │
└─────────────┘     └──────────────┘     └─────────────────────┘
```

- **Local**: Run benchmarks on your machine (requires DeepSeek V4 model)
- **Upload**: Push results to `benchmark-results` branch
- **Dashboard**: Auto-generated HTML with trend charts, deployed to GitHub Pages

## Quick Start

### 1. Run benchmark locally

```bash
cd /Users/zouyee/work/code/mlx-zig
make benchmark
```

This measures TTFT, ITL, and TPS against absolute thresholds and baseline history.

### 2. Upload results

```bash
make upload-benchmark
```

This appends the latest baseline to `benchmark-results/history.jsonl` and pushes to origin.

### 3. View dashboard

After push, GitHub Actions automatically generates and deploys the dashboard:

```
https://dmlx.ai/
```

## Commands

| Command | Description |
|---------|-------------|
| `make benchmark` | Run performance benchmark |
| `make upload-benchmark` | Upload results to `benchmark-results` branch |
| `make setup-benchmark-branch` | Initialize the `benchmark-results` branch (run once) |
| `make check` | Run all local tests (build + test + verify + e2e + benchmark) |

## Configuration

Environment variables for `make benchmark`:

```bash
MAX_TTFT_MS=300       # TTFT threshold (default: 500)
MAX_ITL_MS=80         # ITL threshold (default: 150)
MIN_TPS=10            # Minimum throughput (default: 5)
REGRESSION_PCT=10     # Regression alert threshold % (default: 20)
OUTPUT_TOKENS=50      # Tokens to generate (default: 20)
```

Example:

```bash
MAX_TTFT_MS=300 MAX_ITL_MS=80 make benchmark
```

## Benchmark Result Format

Each benchmark produces a JSON record:

```json
{
  "hostname": "MacBook-Pro",
  "model": "DeepSeek-V4-Flash-4bit",
  "git_commit": "a18bc24",
  "git_branch": "main",
  "timestamp": "2026-05-01T10:00:00Z",
  "ttft_ms": 215.0,
  "itl_ms": 35.0,
  "tps": 28.5,
  "output_tokens": 20,
  "thresholds": {
    "max_ttft_ms": 500,
    "max_itl_ms": 150,
    "min_tps": 5,
    "regression_pct": 20
  }
}
```

## Dashboard Features

The auto-generated dashboard includes:

- **Trend charts**: TTFT, ITL, TPS over time (Plotly.js)
- **History table**: All runs with commit hashes and dates
- **Regression highlighting**: Cells turn red when metrics degrade beyond threshold
- **Per-machine grouping**: Results are grouped by `(hostname, model)`

## CI Integration

### What runs in CI (GitHub Actions)

The `.github/workflows/ci.yml` runs on every PR:
- Build check (`zig build`)
- Unit tests (`zig build test`)
- Format check (`zig fmt --check`)

### What does NOT run in CI

- Performance benchmarks (requires 141GB model, unavailable in GitHub runners)
- End-to-end correctness tests (requires 141GB model)

These are run locally and results are uploaded manually.

### Dashboard deployment

The `.github/workflows/dashboard.yml` runs when:
- Someone pushes to `benchmark-results` branch
- Manually triggered via `workflow_dispatch`

It generates HTML from `history.jsonl` and deploys to GitHub Pages.

## Adding a New Machine

Each machine maintains its own baseline automatically:

```bash
# First run on a new machine — creates baseline
make benchmark
make upload-benchmark

# Subsequent runs compare against this machine's baseline
make benchmark
```

Baseline files are keyed by `hostname` and `model` name.

## Troubleshooting

### "No baseline found" error

Run `make benchmark` first to generate a baseline, then `make upload-benchmark`.

### Dashboard not updating

Check GitHub Actions tab for the "Deploy Performance Dashboard" workflow. It may take 1-2 minutes after push.

### Git push fails

Ensure you have push access to the repository and the `benchmark-results` branch exists:

```bash
git branch benchmark-results
make upload-benchmark
```
