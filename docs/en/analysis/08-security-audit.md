# Chapter 8: Security Boundaries and Code Audit

## 8.1 `@constCast` Repository-Wide Statistics

A total of **10** `@constCast` calls across the repository:

| Location | Line | Purpose | Risk |
|------|------|------|------|
| `array.zig` | 150 | `dataSliceMut`: converts const pointer to mutable | **High** |
| `tree.zig` | 302 | `treeMapInPlace`: recursively traverses field pointers | Medium |
| `tree.zig` | 317 | `treeToArrayPtrs`: collects Array pointers | Low |
| `guided.zig` | 85 | `FiniteStateMachine.deinit`: `[]State` → `*State` | Low |
| `safetensors_reader.zig` | 494 | `mmap` region pointer conversion | Medium |
| `safetensors_reader.zig` | 520 | `munmap` removes const restriction | Low |
| `minimax.zig` | 59-60 | RoPE sin/cos cache initialization | **High** |
| `deepseek_v4.zig` | 198-199 | YARN RoPE sin/cos cache initialization | **High** |
| `deepseek_v4.zig` | 399 | Attention mask initialization | **High** |

### `dataSliceMut` Rampancy

`array.zig:148`:
```zig
pub fn dataSliceMut(self: Array, comptime T: type) ![]T {
    const ptr = try self.dataPtr(T);
    return @constCast(ptr)[0..self.size()];
}
```

Call statistics:
- `ops/nn.zig`: **34 locations**
- `models/minimax.zig`: **4 locations**
- Total: **38 locations**

**Project claims fixed** (`production-roadmap.md`): "Safety: `@constCast` bypasses CoW → all replaced with mlx-c op chains ✅"

**Actual status**: the fix is not fully completed. `nn.zig`'s BatchNorm, LSTM, GRU, RNN, MultiHeadAttention, RoPE, Embedding still use pure CPU scalar loops via `dataSliceMut`.

## 8.2 `prompt_cache.zig` Type Safety Vulnerability (P0)

```zig
// prompt_cache.zig:74
const std_cache: *StandardKVCache = @ptrCast(@alignCast(cache.ptr));
```

**Problem**: receives `[]KVCacheStrategy` (runtime polymorphic), but forcefully casts to `*StandardKVCache`.

**Layout difference between PagedKVCache and StandardKVCache**:

```zig
// StandardKVCache
pub const StandardKVCache = struct {
    keys: Array,
    values: Array,
    offset: usize,
};

// PagedKVCache
pub const PagedKVCache = struct {
    pages: []Page,
    sequences: []SequenceState,
    page_size: usize,
    page_hashes: std.HashMap(...),
    // ...
};
```

When `cache.ptr` actually points to `PagedKVCache`:
- `std_cache.offset` reads the lower 64 bits of `PagedKVCache.pages` pointer — meaningless
- `std_cache.keys` reads the `PagedKVCache.sequences` pointer — subsequent `sliceCache` operations segfault

**Trigger**: default config `--kv-strategy paged_quantized` + `--prompt-cache-file` always triggers this.

## 8.3 `distributed.zig` Resource Leak

```zig
pub fn deinit(self: *DistributedGroup) void {
    _ = self;
    // mlx_distributed_group has no explicit free in this mlx-c version
}
```

Frequent creation/destruction causes resource leaks, risk accumulates in long-running services.

## 8.4 `model_pool.zig` VTable Optional Type

```zig
pub const LoadedModel = struct {
    vtable: ?ModelVTable,  // optional!
    // ...
};
```

`getOrLoad` sets `vtable` to null after loading. `deinit` only releases resources when `vtable != null` — if always null, model resources leak.
