# Design: Paged KV Cache

## Overview

`PagedKVCache` (`src/kvcache/paged.zig`) implements a block/page-based KV cache
strategy for continuous batching. Unlike a standard KV cache that allocates a
single contiguous buffer per sequence, the paged approach:

1. Allocates KV storage in fixed-size **pages** (default 32 tokens)
2. Maps each sequence to a **page table** (logical position → physical page)
3. Enables **page reuse** across sequences via reference counting
4. Supports **Copy-on-Write** for prefix sharing
5. Supports **per-page quantization** (4/8/16 bit) to reduce memory footprint

**Scope:** L1 (interface + data structures) + L2 (page management algorithm).
L3 (MLX slice_update, quantization/dequantization helpers) is out of scope.

## Components

### 1. BlockManager

A standalone block allocator used by the Scheduler for request-level block
management. It is separate from PagedKVCache's internal page pool.

| Method | Purpose |
|--------|---------|
| `init(allocator, total_blocks)` | Pre-allocate block pool and free list |
| `deinit()` | Free all block pools and hash maps |
| `allocateBlocks(req_id, n)` | Allocate n blocks, track per-request ownership |
| `freeBlocks(req_id)` | Return blocks to free pool (CoW-aware) |
| `copyOnWrite(block_id)` | Clone a shared block, return new block_id (!usize, may fail) |
| `canAllocate(n)` | Check free pool capacity |
| `freeCount()` | Number of free blocks available |
| `usedCount()` | Number of used blocks |
| `hashBlock(prev_hash, tokens)` | Compute rolling block hash for prefix caching |
| `findCachedPrefix(tokens, block_size)` | Find reusable blocks by hash chain |
| `registerBlockHash(block_id, hash)` | Register a block's hash for prefix caching |

**CoW semantics:** When `freeBlocks` encounters a block with `ref_count > 1`,
it decrements the ref_count instead of freeing. This allows multiple sequences
to share prefix blocks.

**Block types:**
- `Block` — used by BlockManager (keys, values, used, ref_count, hash, last_access, tokens_used)
- `Page` — used by PagedKVCache (keys, values, quantized_keys, quantized_values, used, ref_count)
- `PageTableEntry` — maps logical page index to physical page index
- `SequenceState` — per-batch-entry page table list and cached length

**Factory functions:**
- `createPaged(allocator, config, stream) → KVCacheStrategy`
- `createPagedWithSize(allocator, config, page_size, stream) → KVCacheStrategy`
- `createPagedQuantized(allocator, config, kv_bits, group_size, stream) → KVCacheStrategy`
- `createPagedQuantized4Bit(allocator, config, stream) → KVCacheStrategy`

### 2. PagedKVCache

The main `KVCacheStrategy` implementation. Key fields:

| Field | Description |
|-------|-------------|
| `pages` | Physical page pool (each page holds keys + values for `page_size` tokens) |
| `sequences` | Per-batch-entry page tables and cached lengths |
| `page_size` | Tokens per page (default 32) |
| `batch_size` | Number of concurrent sequences |
| `num_kv_heads` | Number of KV heads |
| `head_dim` | Dimension per head |
| `max_pages` | Maximum number of physical pages |
| `dtype` | Element dtype (float32, float16, bfloat16) |
| `kv_bits` | Quantization width (4, 8, or 16 = no quantization) |
| `group_size` | Quantization group size |
| `el_per_int` | Elements packed per integer (derived from kv_bits) |
| `cached_keys/values` | Cached contiguous arrays per batch entry (updated on each `updateAndFetch`) |
| `page_hashes` | Hash → physical page mapping for prefix caching |
| `seq_prev_hashes` | Per-sequence running hash for prefix chain computation |

## Algorithm: updateAndFetch

This is the core lifecycle method called by the model's attention layer on
every forward pass.

### Step 1 — Allocate pages for new tokens

```
new_total = cached_len + seq_len
pages_needed = ceil(new_total / page_size)
while sequence.pages.len < pages_needed:
    phys_idx = allocPage(self, stream)  // from physical page pool
    sequence.pages.append({physical: phys_idx})
```

`allocPage` searches for an unused page; if none exist and `pages.len < max_pages`,
it creates a new page via `mlx_zeros`.

### Step 2 — Write KV data into pages

For each batch entry and each token position:
- Compute `page_idx = global_pos / page_size`, `page_offset = global_pos % page_size`
- Slice the new keys/values from the input tensor
- Write into the page via `mlx_slice_update`

**Quantized path (kv_bits < 16):**
1. Quantize the input slice via `mlx_quantize`
2. Write packed data, scales, and biases into the page's `QuantizedTuple`
   via three separate `mlx_slice_update` calls

**Standard path (kv_bits = 16):**
1. Direct `mlx_slice_update` into the page's float arrays

### Step 3 — Register page hashes (prefix caching)

When a page is fully written (`page_offset + write_len == page_size`),
compute a hash and register it in `page_hashes`:

```
page_hash = Wyhash(prev_hash, physical_idx, page_idx)
page_hashes[page_hash] = physical_idx
seq_prev_hash = page_hash
```

### Step 4 — Gather pages into contiguous output

Attention layers require contiguous `[batch, num_kv_heads, seq_len, head_dim]`
arrays. After writing, pages are gathered:

**Single page:** Slice to actual length (`new_total`).

**Multiple pages:** Slice each page to its actual length, then `mlx_concatenate_axis(2)`
along the sequence dimension.

**Quantized path:** Dequantize each page first, then concatenate.

### Step 5 — Batch concatenation

If `batch > 1`, concatenate all per-batch outputs along axis 0.

### Step 6 — Cache for next call

Store copies of the output arrays in `cached_keys/values` so that the
caller can independently free the returned `KVSlice` without losing the
contiguous view for subsequent operations.

## Key Design Decisions

### KD-PK1: Page size = 32 tokens

Tuned for Apple Silicon Metal GPU memory alignment. Smaller pages reduce
internal fragmentation but increase page table overhead. Larger pages improve
locality but waste memory for short sequences.

### KD-PK2: Contiguous gather on every updateAndFetch

The paged structure is for memory management; attention still receives
contiguous arrays. This means every forward pass pays a gather cost
(concatenate pages). The alternative — paged attention kernels that can read
from non-contiguous pages — is future work.

### KD-PK3: Lazy page allocation

Pages are created on-demand in `allocPage`, not at initialization. This
avoids wasting memory for sequences shorter than `max_seq_len`. The tradeoff
is allocation overhead during the first forward pass.

### KD-PK4: Separate BlockManager and PagedKVCache page pools

`BlockManager` manages blocks at the Scheduler/Request level (for planning
and prefix caching). `PagedKVCache` manages pages at the Layer level (for
actual KV storage). This separation allows the scheduler to plan block
allocations without touching layer-specific quantization state.

## VTable Methods

| Method | Behavior |
|--------|----------|
| `updateAndFetch(keys, values, stream)` | Write new KV into pages, gather into contiguous output |
| `currentLen()` | Max cached length across all batch entries |
| `reset()` | Mark all pages unused, clear all sequences, free cached arrays |
| `rollback(to_len)` | Truncate all sequences to `to_len`, mark excess pages unused, free cached arrays |
| `filter(indices, allocator)` | Keep only specified batch entries, mark others' pages unused |
| `deinit(allocator)` | Free all pages, sequences, cached arrays |

## Boundaries

### Upstream: Model Forward Pass

- **Interface:** `KVCacheStrategy.updateAndFetch(keys, values) → KVSlice`
- **Contract:** Model provides new KV tensors of shape `[batch, num_kv_heads, seq_len, head_dim]`
- **PagedKVCache responsibility:** Store KV persistently across forward passes,
  return contiguous arrays for attention

### Upstream: Scheduler / BlockManager

- **Interface:** `BlockManager.allocateBlocks(req_id, n)` → block IDs
- **Contract:** Scheduler ensures sufficient blocks before admitting requests
- **PagedKVCache responsibility:** Manage its own page pool independently;
  BlockManager is for planning, not direct KV storage

### Downstream: Attention Layer

- **Interface:** `KVSlice { keys, values }` — contiguous `[batch, heads, seq, dim]`
- **Contract:** Attention reads KV but does NOT modify it
- **PagedKVCache responsibility:** Maintain KV persistence and correctness

## Known Divergences

<!-- DIVERGENCE: Prefix caching hash currently uses (physical_idx, page_idx)
     as a proxy for content. A full implementation would hash the actual token
     IDs and KV tensor content for true content-addressable caching.
     **为准方**: 设计。当前代理哈希可能产生误命中（不同内容映射到同一哈希）。
     代码 TODO：改用 token ID bytes + Wyhash 做内容寻址。 -->

<!-- DIVERGENCE: PagedKVCache and BlockManager maintain separate page/block
     pools. In vLLM these are unified (BlockManager IS the page allocator).
     Our separation means the scheduler plans block allocations without seeing
     layer-specific quantization state, but PagedKVCache manages its own
     on-demand page creation.
     **为准方**: 代码。当前实现可工作，但增加了调度器协调复杂度。
     未来若统一池子，需让 BlockManager 感知量化配置（kv_bits, group_size）。 -->
