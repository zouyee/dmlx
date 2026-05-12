/// DeepseekV4Cache — Specialized KV cache for DeepSeek V4's CSA/HCA attention.
///
/// Manages three state branches:
///   - `local`: RotatingKVCache for sliding-window local attention
///   - `compressor_state`: buffer + pooled state for the Compressor module
///   - `indexer_state`: buffer + pooled state for the Indexer module
///
/// Each branch state contains:
///   - buffer_kv / buffer_gate: remainder tokens not yet forming a full compression window
///   - pooled: accumulated compressed KV blocks
///   - buffer_lengths / pooled_lengths: per-batch-item lengths for variable-length support
///
/// Reference: mlx-lm `DeepseekV4Cache` (deepseek_v4.py:888-1480)
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const reduce_mod = @import("mlx").reduce;
const rotating = @import("rotating.zig");
const iface = @import("interface.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const RotatingKVCache = rotating.RotatingKVCache;
const KVCacheStrategy = iface.KVCacheStrategy;
const KVSlice = iface.KVSlice;
const LayerConfig = iface.LayerConfig;

/// State for one compression branch (compressor or indexer).
pub const BranchState = struct {
    buffer_kv: ?Array,
    buffer_gate: ?Array,
    pooled: ?Array,
    buffer_lengths: ?[]usize,
    pooled_lengths: ?[]usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) BranchState {
        return .{
            .buffer_kv = null,
            .buffer_gate = null,
            .pooled = null,
            .buffer_lengths = null,
            .pooled_lengths = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BranchState) void {
        if (self.buffer_kv) |b| b.deinit();
        if (self.buffer_gate) |b| b.deinit();
        if (self.pooled) |p| p.deinit();
        if (self.buffer_lengths) |l| self.allocator.free(l);
        if (self.pooled_lengths) |l| self.allocator.free(l);
    }
};

pub const DeepseekV4Cache = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    local: *RotatingKVCache,
    compressor_state: BranchState,
    indexer_state: BranchState,
    sliding_window: usize,
    owns_local: bool,

    pub fn init(
        allocator: std.mem.Allocator,
        config: LayerConfig,
        sliding_window: usize,
        stream: c.c.mlx_stream,
    ) !DeepseekV4Cache {
        const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });
        const local = try allocator.create(RotatingKVCache);
        errdefer allocator.destroy(local);
        local.* = try RotatingKVCache.init(allocator, config, sliding_window, stream);

        return .{
            .allocator = allocator,
            .ctx = ctx,
            .local = local,
            .compressor_state = BranchState.init(allocator),
            .indexer_state = BranchState.init(allocator),
            .sliding_window = sliding_window,
            .owns_local = true,
        };
    }

    pub fn deinit(self: *DeepseekV4Cache) void {
        self.compressor_state.deinit();
        self.indexer_state.deinit();
        if (self.owns_local) {
            self.local.keys.deinit();
            self.local.values.deinit();
            self.allocator.destroy(self.local);
        }
    }

    /// Delegate to local rotating cache for KV update.
    pub fn updateAndFetch(self: *DeepseekV4Cache, keys: Array, values: Array, stream: c.c.mlx_stream) !KVSlice {
        return self.local.asStrategy().updateAndFetch(keys, values, stream);
    }

    /// Get current offset (total tokens seen by local cache).
    pub fn offset(self: *DeepseekV4Cache) usize {
        return self.local.total_tokens;
    }

    /// Check if cache is empty.
    pub fn empty(self: *DeepseekV4Cache) bool {
        return self.local.total_tokens == 0;
    }

    /// Get branch state by key.
    fn branchState(self: *DeepseekV4Cache, state_key: []const u8) *BranchState {
        if (std.mem.eql(u8, state_key, "indexer_state")) {
            return &self.indexer_state;
        }
        return &self.compressor_state;
    }

    /// Accumulate windows for compression.
    /// Buffers incomplete windows and returns only complete windows for pooling.
    ///
    /// Returns: (ready_kv, ready_gate, pool_base)
    /// - ready_kv: [B, usable, dim] — complete windows ready for pooling
    /// - ready_gate: [B, usable, dim] — corresponding gate values
    /// - pool_base: position offset for RoPE on pooled output
    pub fn accumulateWindows(
        self: *DeepseekV4Cache,
        kv: Array,
        gate: Array,
        state_key: []const u8,
        ratio: usize,
        start_pos: usize,
    ) !struct { Array, Array, usize } {
        const state = self.branchState(state_key);
        const kv_shape = kv.shape();
        const B = @as(usize, @intCast(kv_shape[0]));
        _ = B;
        const dim = @as(usize, @intCast(kv_shape[2]));
        _ = dim;

        // Simple path (no variable-length batch support for now):
        // Concat buffer with new kv/gate, compute usable windows, store remainder
        var combined_kv: Array = undefined;
        var combined_gate: Array = undefined;
        var buf_len: usize = 0;

        if (state.buffer_kv) |buf| {
            buf_len = @intCast(buf.shape()[1]);
            combined_kv = try shape_mod.concatenateAxis(self.ctx, &[_]Array{ buf, kv }, 1);
            combined_gate = try shape_mod.concatenateAxis(self.ctx, &[_]Array{ state.buffer_gate.?, gate }, 1);
            // Free old buffers
            buf.deinit();
            state.buffer_gate.?.deinit();
            state.buffer_kv = null;
            state.buffer_gate = null;
        } else {
            combined_kv = try ops.copy(self.ctx, kv);
            combined_gate = try ops.copy(self.ctx, gate);
        }

        const total_len = @as(usize, @intCast(combined_kv.shape()[1]));
        const usable = (total_len / ratio) * ratio;
        const remainder = total_len - usable;

        // Store remainder in buffer
        if (remainder > 0) {
            const combined_shape = combined_kv.shape();
            state.buffer_kv = try ops.slice(self.ctx, combined_kv, &[_]i32{ 0, @intCast(usable), 0 }, &[_]i32{ combined_shape[0], @intCast(total_len), combined_shape[2] }, &[_]i32{});
            state.buffer_gate = try ops.slice(self.ctx, combined_gate, &[_]i32{ 0, @intCast(usable), 0 }, &[_]i32{ combined_shape[0], @intCast(total_len), combined_shape[2] }, &[_]i32{});
        }

        // Extract usable portion
        if (usable > 0) {
            const combined_shape = combined_kv.shape();
            const ready_kv = try ops.slice(self.ctx, combined_kv, &[_]i32{ 0, 0, 0 }, &[_]i32{ combined_shape[0], @intCast(usable), combined_shape[2] }, &[_]i32{});
            const ready_gate = try ops.slice(self.ctx, combined_gate, &[_]i32{ 0, 0, 0 }, &[_]i32{ combined_shape[0], @intCast(usable), combined_shape[2] }, &[_]i32{});
            combined_kv.deinit();
            combined_gate.deinit();

            const pool_base = if (start_pos >= buf_len) start_pos - buf_len else 0;
            return .{ ready_kv, ready_gate, pool_base };
        } else {
            // No complete windows — return empty
            const combined_shape = combined_kv.shape();
            const empty_kv = try array_mod.zeros(self.allocator, &[_]i32{ combined_shape[0], 0, combined_shape[2] }, .float32);
            const empty_gate = try array_mod.zeros(self.allocator, &[_]i32{ combined_shape[0], 0, combined_shape[2] }, .float32);
            combined_kv.deinit();
            combined_gate.deinit();
            return .{ empty_kv, empty_gate, start_pos };
        }
    }

    /// Update pooled state by concatenating new pooled blocks.
    pub fn updatePool(self: *DeepseekV4Cache, new_pooled: Array, state_key: []const u8) !Array {
        const state = self.branchState(state_key);

        if (new_pooled.shape()[1] == 0) {
            // No new pooled blocks
            if (state.pooled) |p| {
                return ops.copy(self.ctx, p);
            }
            // Return empty
            const np_shape = new_pooled.shape();
            return array_mod.zeros(self.allocator, &[_]i32{ np_shape[0], 0, np_shape[2] }, new_pooled.dtype());
        }

        if (state.pooled) |existing| {
            const merged = try shape_mod.concatenateAxis(self.ctx, &[_]Array{ existing, new_pooled }, 1);
            existing.deinit();
            state.pooled = try ops.copy(self.ctx, merged);
            return merged;
        } else {
            state.pooled = try ops.copy(self.ctx, new_pooled);
            return ops.copy(self.ctx, new_pooled);
        }
    }

    /// Get current pooled lengths for a branch.
    pub fn pooledLengths(self: *DeepseekV4Cache, state_key: []const u8) ?[]usize {
        return self.branchState(state_key).pooled_lengths;
    }

    /// Implement KVCacheStrategy interface for compatibility.
    pub const vtable: iface.VTable = .{
        .updateAndFetch = updateAndFetchImpl,
        .currentLen = currentLenImpl,
        .reset = resetImpl,
        .filter = null,
        .rollback = null,
        .extend = null,
        .deinit = deinitImpl,
    };

    fn updateAndFetchImpl(ctx_ptr: *anyopaque, keys: Array, values: Array, stream: c.c.mlx_stream) anyerror!KVSlice {
        const self: *DeepseekV4Cache = @ptrCast(@alignCast(ctx_ptr));
        return self.updateAndFetch(keys, values, stream);
    }

    fn currentLenImpl(ctx_ptr: *anyopaque) usize {
        const self: *DeepseekV4Cache = @ptrCast(@alignCast(ctx_ptr));
        return self.local.total_tokens;
    }

    fn resetImpl(ctx_ptr: *anyopaque) void {
        const self: *DeepseekV4Cache = @ptrCast(@alignCast(ctx_ptr));
        self.local.total_tokens = 0;
        self.local.cursor = 0;
    }

    fn deinitImpl(ctx_ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *DeepseekV4Cache = @ptrCast(@alignCast(ctx_ptr));
        self.deinit();
        allocator.destroy(self);
    }

    pub fn asStrategy(self: *DeepseekV4Cache) KVCacheStrategy {
        return .{ .ptr = self, .vtable = &vtable };
    }
};

/// Factory for DeepseekV4Cache.
pub fn createDeepseekV4Cache(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    sliding_window: usize,
    stream: c.c.mlx_stream,
) !KVCacheStrategy {
    const cache = try allocator.create(DeepseekV4Cache);
    errdefer allocator.destroy(cache);
    cache.* = try DeepseekV4Cache.init(allocator, config, sliding_window, stream);
    return cache.asStrategy();
}
