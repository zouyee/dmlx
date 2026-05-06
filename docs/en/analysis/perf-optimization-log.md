# dmlx Performance Optimization Log

**Date**: 2026-05-06
**Spec**: `.kiro/specs/perf-regression-fix/`
**Baseline**: commit 8ced50c — 9.4 tok/s, 446.8ms prefill
**Target**: ≥12 tok/s, ≤380ms prefill

---

## Optimization 1: Remove Layer Debug Prints

**File**: `src/models/deepseek_v4.zig:2761-2766`
**Change**: Removed 2× `std.debug.print("Layer {d}/43 forward start/done")` from per-layer loop
**Impact**: Eliminates 86 write(2) syscalls per token (43 layers × 2 prints)
**Status**: ✅ Applied, build passes

---

## Optimization 2: Remove MoE Diagnostic Synchronization

**File**: `src/models/deepseek_v4.zig:1038-1066`
**Change**: Removed entire `if (self.layer_idx == 0 and self.stream_provider != null)` diagnostic block
**Impact**: Eliminates 4× forced `eval()` + 2× GPU→CPU `dataSlice` copies per token at layer 0
**Status**: ✅ Applied, build passes

---

## Optimization 3: Remove Prefill Logits Diagnostic

**File**: `src/models/deepseek_v4.zig:2850-2895`
**Change**: Removed 45-line block that iterates 128K vocab array twice for statistics
**Impact**: Eliminates forced GPU sync + O(128K) CPU iteration during prefill
**Status**: ✅ Applied, build passes

---

## Optimization 4: Replace Insertion Sort → std.mem.sort

**File**: `src/models/deepseek_v4_loader.zig:614-625`
**Change**: `std.sort.insertion` → `std.mem.sort` (pattern-defeating quicksort, O(n log n))
**Impact**: Reduces model loading sort overhead from O(n²) to O(n log n) for ~75 entries/shard × 33 shards
**Status**: ✅ Applied, build passes

---

## Optimization 5: Fix mmap Advisory for Expert Streaming

**File**: `src/models/expert_stream.zig:216+`
**Change**: Added `posix_madvise(POSIX_MADV_RANDOM)` override after `mmap.mmapAll(index)` in expert stream init
**Impact**: Prevents OS from thrashing page cache on random MoE expert access patterns
**Status**: ✅ Applied, build passes

---

## Verification Results

### P1 Quick Test (2+2=)

```
Command: zig-out/bin/dmlx chat --model ~/models/DeepSeek-V4-Flash-4bit \
  --prompt "2+2=" --max-tokens 30 --temperature 0 \
  --smelt --smelt-strategy stream --smelt-experts 0.1

Output: ". The user's query is "2+2=?". The assistant's response is "4"..."
Result: ✅ PASS (contains "4")
Token 30 latency: 147.1ms (cache warming phase, not steady-state)
```

### Full 7-Prompt Suite

**Status**: Pending (requires ~20-35 min with model loaded)

---

## Summary

| # | Optimization | Expected Gain | Status |
|---|---|---|---|
| 1 | Remove layer prints | ~8 ms/tok | ✅ Applied |
| 2 | Remove MoE DIAG | ~10 ms/tok | ✅ Applied |
| 3 | Remove logits diag | ~3 ms prefill | ✅ Applied |
| 4 | Sort O(n²)→O(n log n) | ~20ms loading | ✅ Applied |
| 5 | mmap MADV_RANDOM | ~5-10 ms/tok (reduced page faults) | ✅ Applied |
| **Total expected** | **~24 ms/tok reduction** | **9.4 → ~12+ tok/s** |

---

## Next Steps

1. Run full `bash scripts/best_test.sh` to confirm 7/7 PASS
2. Run `scripts/run_benchmark.sh` for precise tok/s measurement
3. Commit changes with performance results
