# Troubleshooting: Scheduler & Block Manager

**Component**: Scheduler, BlockManager, Request Queue
**Spec**: `.kiro/specs/production-deployment/design.md` §3.1, §3.2
**Properties**: Property 7 (Prioritization), Property 8 (Block Conservation)

## Symptom: Requests remain in waiting queue indefinitely

**Root cause**: BlockManager.freeCount() under-reports free blocks.

### Spec References

- `design.md` §3.2 — Block Manager interface
- Property 8: Block Conservation — "sum of free blocks and used blocks SHALL always equal the total block count"

### Code Checkpoints

1. `src/scheduler.zig:46` — `canAllocate()` logic
2. `src/scheduler.zig:284` — waiting queue promotion
3. `src/kvcache/paged.zig:89` — `freeCount()` calculation

### Fix History

- **2026-04-20**: Changed BlockManager to track ref_count separately from used/allocated state. Shared blocks now decrement freeCount only once, not per-request.

## Symptom: Decode requests starved while long prefill runs

**Root cause**: Chunked prefill not splitting correctly; entire prompt processed in one step.

### Spec References

- Property 12: Chunked Prefill Correctness
- `design.md` §3.1 — Scheduler `max_prefill_tokens`

### Code Checkpoints

1. `src/scheduler.zig:183` — `currentPrefillChunkLen()`
2. `src/scheduler.zig:275` — prefill vs decode categorization in `schedule()`
3. `src/scheduler.zig:335` — `prefill_offset` advancement in `postprocess()`

### Diagnosis Steps

1. Check `max_prefill_tokens` config — is it set to a reasonable value (e.g., 512)?
2. Verify `Request.hasPendingPrefill()` returns true until full prompt consumed
3. Check that `schedule()` places prefilling requests in `prefill_list`, not `decode_list`

## Symptom: "InsufficientBlocks" error during load test

**Root cause**: Block pool exhausted; no eviction policy for completed requests.

### Spec References

- Property 8: Block Conservation
- `design.md` §3.2 — `freeBlocks()`

### Code Checkpoints

1. `src/scheduler.zig:290` — waiting queue promotion block check
2. `src/scheduler.zig:370` — completed request cleanup in `postprocess()`
3. `src/kvcache/paged.zig:72` — `freeBlocks()` implementation

### Diagnosis Steps

1. Check if `postprocess()` is freeing blocks for completed requests
2. Verify `freeCount() + usedCount() == total_blocks` invariant holds
3. If invariant broken, search for double-free or missing-free paths
