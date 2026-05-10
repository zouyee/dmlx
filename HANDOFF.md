# Server Engine V2 — Critical Fixes (2026-05-09)

## Summary

Two blocking issues resolved today:
1. **20s/token eval delay** → fixed by `mlx_set_default_stream`
2. **Empty content + streaming segfault** → fixed by heap-allocating `BpeTokenizer`

## Fix 1: MLX eval performance (`mlx_set_default_stream`)

**Root cause**: All MLX ops were created on an explicit GPU stream (`ctx.stream.inner`), but `mlx_eval()` (called in `sampling.zig` via `logits.eval()`) uses the *default stream*. Since we never called `mlx_set_default_stream()`, the default stream was not our GPU stream — eval fell back to CPU execution, causing ~20s delay per token.

**Fix**: Call `mlx_set_default_stream(stream)` immediately after creating the GPU stream in both server and CLI paths.

**Files changed**:
- `src/server/state.zig` — `loadModel()`: `mlx_set_default_stream(stream)` after `mlx_default_gpu_stream_new()`
- `src/server/state.zig` — `ServerState.deinit()`: reset default stream to CPU before freeing GPU stream
- `src/main.zig` — `runChat()`: same pattern with defer cleanup

**Verification**:
- Before: ~20,000ms/token
- After: ~0.3ms forward + ~3ms eval = ~3.3ms/token total (Qwen2.5-0.5B, M4 Pro)

## Fix 2: Tokenizer segfault → empty content

**Root cause**: `BpeTokenizer` contains `std.AutoHashMap` fields that are NOT safe to `memcpy` (they hold internal heap pointers). `ServerState` was returned by value from `loadModel()`, causing a bitwise copy of `BpeTokenizer`. The copied `ids_to_tokens` HashMap had corrupted internal pointers, so `get(id)` returned garbage — manifesting as:
- `count()` returning ~2.3B (uninitialized memory)
- `get(id)` returning null for all tokens → decode returning `""`
- In some cases, the corrupted `SequenceDecoder.decoders` slice pointer (null) caused EXC_BAD_ACCESS

**Fix**: Heap-allocate `BpeTokenizer` so its address is stable and never copied.

**Files changed**:
- `src/server/state.zig` — `ServerState.tokenizer_backend`: changed from `BpeTokenizer` to `*BpeTokenizer`
- `src/server/state.zig` — `loadModel()`: `allocator.create(BpeTokenizer)` + `tokenizer_backend.* = BpeTokenizer.init(allocator)`
- `src/server/state.zig` — `ServerState.deinit()`: `allocator.destroy(self.tokenizer_backend)` after `deinit()`
- `src/server/state.zig` — `loadModel()`: `tokenizer_strategy` fixup happens after `ServerState` construction so `ptr` points to the stable heap address

**Verification**:
- Before: non-streaming returns `""`, streaming segfaults after 1-2 tokens
- After: both streaming and non-streaming return correct text, no crashes

## E2E Verification

Tested with Qwen2.5-0.5B-Instruct on M4 Pro:
- `2+2=` → `" 2+2=4\n\nAssistant: 4\n\n..."` ✅
- `"The capital of France is"` → contains `"Paris"` ✅
- Streaming 20 tokens → 22 SSE chunks, no crash ✅
- Performance: ~3.3ms/token (300 tok/s)

## Remaining work

See `.kiro/specs/server-engine-v2/tasks.md` for full task list. The blocking infra bugs are now resolved; remaining items are feature work (continuous batching, speculative decoding, guided decoding, etc.).
