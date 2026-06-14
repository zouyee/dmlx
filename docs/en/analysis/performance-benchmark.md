---
date: 2026-06-13
Commit: 3294fb3 (main)
model: DeepSeek-V4-Flash-4bit (~141GB on disk, 33 shards)
hardware: Apple M4 Pro, 48GB
mode: serve, smelt 0.20 + stream, ExpertCache 0MB, temperature=0
build: zig build -Doptimize=ReleaseFast
generated_by: scripts/run_benchmark.sh
total_time: 87s (perf + e2e)
---

# dmlx Performance Benchmark Report

## Summary

| Mode | SMELT N | tok/s (median) | Paris |
|------|---------|---------------|-------|
| serve | 51 | 1.347 | ❌ FAIL |

## Changes
- Fixed engine.c dispatch: 1D grid sizes from (INTERMEDIATE+1)/2 → INTERMEDIATE/8 for gate-up, (DIM+1)/2 → DIM/8 for down-proj
- Fixed repack_affine.py: LUT kernel quantization with max(|x|)/6 scale (same as original MXFP4)
- Repack time: ~87s total

## Test Results
- Unit tests: 430+ PASS
- Correctness: ❌ Paris FAIL (see above)
- Perf: 1.347 tok/s median (3 sequential runs, max_tokens=5)
