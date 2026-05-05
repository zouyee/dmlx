# Chapter 4: KV Cache Subsystem

## 4.1 Strategy Interface Design (`kvcache/interface.zig`)

VTable runtime polymorphic design:

```zig
pub const VTable = struct {
    updateAndFetch: *const fn (ctx: *anyopaque, keys: Array, values: Array, stream: mlx_stream) anyerror!KVSlice,
    currentLen: *const fn (ctx: *anyopaque) usize,
    reset: *const fn (ctx: *anyopaque) void,
    filter: ?*const fn (ctx: *anyopaque, indices: []const usize, allocator: Allocator) anyerror!void,
    rollback: ?*const fn (ctx: *anyopaque, to_len: usize) void,
    deinit: *const fn (ctx: *anyopaque, allocator: Allocator) void,
};
```

Design highlights: runtime strategy switching + comptime internal specialization + full decoupling from attention layers.

## 4.2 Six Strategy Comparison

| Strategy | Characteristics | Use Case | Code Location |
|------|------|---------|---------|
| Standard | Simple contiguous buffer | Single request, short sequences | `kvcache/standard.zig` |
| Rotating | Ring buffer, fixed window | Ultra-long sequences (avoid OOM) | `kvcache/rotating.zig` |
| Quantized | 4/8/16 bit KV compression | Memory-constrained | `kvcache/quantized.zig` |
| Paged | 32-token pages + page table + CoW | Continuous batching (**default**) | `kvcache/paged.zig` |
| PagedQuantized | Paged + Quantized combined | Extreme memory optimization | `kvcache/paged.zig` |
| Tiered | RAM hot + SSD cold + LRU | Ultra-long context + multi-model | `kvcache/tiered.zig` |

## 4.3 PagedKVCache (`kvcache/paged.zig`, 1,152 lines)

### Core Design

- **Page size**: default 32 tokens (tuned for Apple Silicon Metal GPU memory alignment)
- **BlockManager**: manages free pool, per-request block mapping, CoW mechanism
- **Prefix hashing**: `hashBlock` uses Wyhash to compute rolling hashes
- **Copy-on-Write**: when shared block `ref_count > 1`, allocates new block and copies data

### updateAndFetch Algorithm Flow

1. **Allocate pages**: `new_total = cached_len + seq_len`, allocate new pages as needed
2. **Write KV**: write keys/values into corresponding pages via `mlx_slice_update`
3. **Register hashes**: compute hash when page is full, register in `page_hashes` map
4. **Gather output**: concatenate scattered pages into contiguous `[batch, heads, seq, dim]` arrays
5. **Quantized path**: quantized pages must be dequantized before concatenation

## 4.4 TieredKVCache (`kvcache/tiered.zig`)

- Wraps `PagedKVCache` as the hot tier
- When exceeding `hot_capacity`, LRU pages are written to SSD: `{cold_dir}/block_{id}.safetensors`
- `restoreFromSSD`: restores blocks from safetensors files to hot tier

## 4.5 Prompt Cache (`prompt_cache.zig`, 563 lines)

Supports save/load KV cache state to safetensors files.

### 🔴 High Severity Vulnerability

```zig
// prompt_cache.zig:74
const std_cache: *StandardKVCache = @ptrCast(@alignCast(cache.ptr));
```

**Problem**: `savePromptCache` receives `[]KVCacheStrategy` (runtime polymorphic), but directly casts `cache.ptr` to `*StandardKVCache`.

**Consequences**:
- `PagedKVCache`'s `ptr` points to a `PagedKVCache` struct with a completely different field layout
- Accessing `.offset` reads part of the `pages` pointer — the value is meaningless
- Accessing `.keys`/`.values` may read pointers from the `PageTableEntry` array, causing segfault

**Trigger**: using default config `--kv-strategy paged_quantized` + `--prompt-cache-file` will always trigger this.

**Fix Suggestions**:
1. Add `saveState`/`loadState` methods to `KVCacheStrategy.VTable`
2. Or add runtime type check: `std.debug.assert(cache.vtable == &StandardKVCache.vtable)`
