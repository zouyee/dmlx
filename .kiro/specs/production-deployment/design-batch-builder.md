# Design: Continuous Batching (BatchBuilder)

## Overview

`BatchBuilder` (`src/batch_builder.zig`) constructs batched input tensors from
multiple active inference requests, enabling a single forward pass to serve
concurrent users. It is the bridge between the Scheduler's request selection
and the model's `forward()` function.

**Scope:** L1 (interface + data structures) + L2 (mask construction algorithm).
L3 (buffer allocation, MLX array creation) is out of scope.

## Interface

```zig
pub const BatchResult = struct {
    batched_tokens: Array,     // shape [total_tokens]
    position_ids: Array,       // shape [total_tokens]
    attention_mask: Array,     // shape [total_tokens, total_tokens]
    total_tokens: usize,
    num_requests: usize,

    pub fn deinit(self: *BatchResult) void {
        self.batched_tokens.deinit();
        self.position_ids.deinit();
        self.attention_mask.deinit();
    }
};

/// Build a BatchResult from a ScheduleResult.
pub fn build(
    allocator: std.mem.Allocator,
    schedule_result: *const ScheduleResult,
    ctx: EagerContext,
) !BatchResult;

/// Query whether a position in the attention mask is allowed to attend.
/// Returns true if mask value is 0.0 (attend), false if blocked (-inf).
pub fn isAttending(mask: []const f32, total_tokens: usize, row: usize, col: usize) bool;
```

## Algorithm: Batched Input Construction

### Step 1 — Collect Request Tokens

For each scheduled request, extract the token sequence to feed into the model:

| Request State | Tokens Extracted | Rationale |
|---------------|------------------|-----------|
| `prefilling` / `waiting` | `prompt_tokens[prefill_offset..]` | Process remaining prompt tokens (chunked prefill) |
| `decoding` | `generated_tokens.last()` (or prompt tail) | Single-token advance per step |
| `done` | empty | Skip — request is finished |

**Edge cases:**
- If `prefill_offset >= prompt_tokens.len` AND prompt is non-empty: return the
  last prompt token as a single-token decode.
- If the prompt is empty (`prompt_tokens.len == 0`): return an empty slice.
- If a decode request has neither generated tokens nor prompt tokens: return an
  empty slice.

### Step 2 — Concatenate into Flat Tensor

All token sequences are concatenated into a 1D array:

```
Request A (prefill): [t_A0, t_A1, t_A2]
Request B (decode):  [t_B0]
Request C (prefill): [t_C0, t_C1]
─────────────────────────────────────
Batched tokens:      [t_A0, t_A1, t_A2, t_B0, t_C0, t_C1]
```

Prefill requests are placed before decode requests. This ordering is a
current convention, not a hard requirement.

### Step 3 — Build Position IDs

Each token's position is its absolute position in the request's sequence:

| Token | Source Request | Position |
|-------|---------------|----------|
| t_A0 | A (prefill, offset=0) | 0 |
| t_A1 | A (prefill, offset=0) | 1 |
| t_A2 | A (prefill, offset=0) | 2 |
| t_B0 | B (decode, seq_len=5) | 4 |
| t_C0 | C (prefill, offset=2) | 2 |
| t_C1 | C (prefill, offset=2) | 3 |

Prefill positions start at `prefill_offset`. Decode positions are
`seq_len - seq_len_input` (the position of the input token being fed,
which for single-token decode equals `seq_len - 1`).

### Step 4 — Build Causal Attention Mask

The attention mask is a `total_tokens × total_tokens` matrix initialized to
`-inf` (block all). For each request's span, a causal mask is applied:

```
        t_A0  t_A1  t_A2  t_B0  t_C0  t_C1
      ┌─────┬─────┬─────┬─────┬─────┬─────┐
t_A0  │ 0.0 │-inf │-inf │-inf │-inf │-inf │
t_A1  │ 0.0 │ 0.0 │-inf │-inf │-inf │-inf │
t_A2  │ 0.0 │ 0.0 │ 0.0 │-inf │-inf │-inf │
      ├─────┼─────┼─────┼─────┼─────┼─────┤
t_B0  │-inf │-inf │-inf │ 0.0 │-inf │-inf │
      ├─────┼─────┼─────┼─────┼─────┼─────┤
t_C0  │-inf │-inf │-inf │-inf │ 0.0 │-inf │
t_C1  │-inf │-inf │-inf │-inf │ 0.0 │ 0.0 │
      └─────┴─────┴─────┴─────┴─────┴─────┘
```

**Key invariant:** Tokens from different requests NEVER attend to each other.
Within a request, token at position `row` can attend to positions `0..row`
(causal).

**Memory cost:** O(total_tokens²) for the mask. At 4K tokens, the mask is
64M floats (256MB). This is acceptable for small batches but will need
sparsity/optimization for large-scale serving.

## Key Design Decisions

### KD-BB1: Prefill-before-decode ordering

Prefill requests are concatenated before decode requests in the batch.
This groups long sequences together, which may improve memory locality.
The ordering does not affect correctness because the attention mask isolates
requests regardless of position in the batch.

### KD-BB2: Dense attention mask (not sparse/block-diagonal)

The mask is stored as a dense `total_tokens × total_tokens` matrix.
This is the simplest correct implementation. Future optimizations:
- Block-diagonal storage (only store per-request sub-matrices)
- Use MLX's `mx.fast.sdpa` with a custom mask argument
- Flash Attention-style variable-length sequence support

### KD-BB3: Single-token decode per step

Each decode request contributes exactly one token per batch step.
This is the standard vLLM continuous batching pattern. A decode request
cannot generate multiple tokens in a single forward pass because the next
token depends on the KV cache update from the current token.

## Boundaries

### Upstream: Scheduler

- **Interface:** `ScheduleResult` containing `prefill_requests` and `decode_requests`
- **Contract:** Each request has valid `state`, `prompt_tokens`, `generated_tokens`,
  and `prefill_offset`
- **BatchBuilder responsibility:** Read request state, do NOT mutate requests

### Downstream: Model Forward Pass

- **Interface:** `forward(batched_tokens, position_ids, attention_mask, caches)`
- **Contract:** Model must support batched input (shape `[batch, seq_len]` or
  flattened equivalent)
- **Current state:** Most model implementations expect single-request input.
  Batched forward is a future requirement for true continuous batching.

## Known Divergences

<!-- DIVERGENCE: Batched forward pass is not yet implemented in model layers.
     BatchBuilder produces correct batched tensors, but model `forward()`
     implementations expect single-request input.
     **为准方**: 设计。BatchBuilder 输出格式是正确的；模型层需升级到支持
     batched input。代码 TODO：(1) batched embedding lookup, (2) batched
     attention with per-request position_ids, (3) batched output splitting. -->

<!-- DIVERGENCE: Empty batch handling uses `std.heap.page_allocator` as a
     temporary workaround instead of the caller's allocator.
     **为准方**: 设计。`emptyBatch()` 应接受 allocator 参数。当前 workaround
     不泄漏但不符合模块约定。代码 TODO。 -->
