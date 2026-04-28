/// KV Cache Strategy Interface — Plugin abstraction for multi-strategy KV cache management.
///
/// Design goals:
///   1. Runtime polymorphism: switch cache strategy without recompiling model code.
///   2. Zero-overhead internals: each strategy uses comptime shape specialization.
///   3. Attention decoupling: Attention layer depends only on KVCacheStrategy, not concrete type.
///   4. Unified lifecycle: CacheManager owns all layer caches; individual strategies own their buffers.
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");

const Array = array_mod.Array;

/// A slice of cached keys and values returned to the attention layer.
/// Shape: keys [B, H, S, D_k], values [B, H, S, D_v]
/// where S is the current sequence length (including the newly appended tokens).
pub const KVSlice = struct {
    keys: Array,
    values: Array,
};

/// VTable for runtime polymorphic KV cache strategies.
/// Every concrete strategy implements these functions and registers them in its VTable.
pub const VTable = struct {
    /// Append new keys/values to the cache and return the full cached sequence.
    updateAndFetch: *const fn (
        ctx: *anyopaque,
        keys: Array,
        values: Array,
        stream: c.c.mlx_stream,
    ) anyerror!KVSlice,

    /// Return the current cached sequence length (number of tokens stored so far).
    currentLen: *const fn (ctx: *anyopaque) usize,

    /// Reset the cache to empty state (sequence length = 0).
    reset: *const fn (ctx: *anyopaque) void,

    /// Filter the cache to keep only the batch entries at the given indices.
    /// Used in continuous batching when some sequences complete or are evicted.
    /// null if the strategy does not support dynamic batch filtering.
    filter: ?*const fn (
        ctx: *anyopaque,
        indices: []const usize,
        allocator: std.mem.Allocator,
    ) anyerror!void,

    /// Roll back the cache to a previous sequence length.
    /// Used in speculative decoding when draft tokens are rejected.
    /// null if the strategy does not support rollback.
    rollback: ?*const fn (ctx: *anyopaque, to_len: usize) void,

    /// Release all resources held by this strategy.
    deinit: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) void,
};

/// Runtime-polymorphic handle to a KV cache strategy.
/// Constructed via concrete strategy's `asStrategy()` method.
pub const KVCacheStrategy = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    /// Append new KV and return full cached sequence.
    pub fn updateAndFetch(
        self: KVCacheStrategy,
        keys: Array,
        values: Array,
        stream: c.c.mlx_stream,
    ) !KVSlice {
        return self.vtable.updateAndFetch(self.ptr, keys, values, stream);
    }

    /// Current sequence length in the cache.
    pub fn currentLen(self: KVCacheStrategy) usize {
        return self.vtable.currentLen(self.ptr);
    }

    /// Reset to empty.
    pub fn reset(self: KVCacheStrategy) void {
        return self.vtable.reset(self.ptr);
    }

    /// Whether this strategy supports batch filtering.
    pub fn supportsFilter(self: KVCacheStrategy) bool {
        return self.vtable.filter != null;
    }

    /// Filter to keep only specified batch indices.
    pub fn filter(
        self: KVCacheStrategy,
        indices: []const usize,
        allocator: std.mem.Allocator,
    ) !void {
        if (self.vtable.filter) |f| {
            return f(self.ptr, indices, allocator);
        }
        return error.FilterNotSupported;
    }

    /// Roll back to a previous sequence length.
    pub fn rollback(self: KVCacheStrategy, to_len: usize) void {
        if (self.vtable.rollback) |f| {
            return f(self.ptr, to_len);
        }
    }

    /// Release resources.
    pub fn deinit(self: KVCacheStrategy, allocator: std.mem.Allocator) void {
        return self.vtable.deinit(self.ptr, allocator);
    }
};

/// Per-layer KV cache configuration.
/// Passed to strategy constructors to describe the shape requirements.
pub const LayerConfig = struct {
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    dtype: @import("../dtype.zig").Dtype,
};

/// Factory signature for creating a KV cache strategy instance.
/// Used by CacheManager to instantiate per-layer caches with the same strategy.
pub const StrategyFactory = *const fn (
    allocator: std.mem.Allocator,
    config: LayerConfig,
    stream: c.c.mlx_stream,
) anyerror!KVCacheStrategy;
