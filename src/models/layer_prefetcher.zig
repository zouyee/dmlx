/// Asynchronous layer prefetcher for stream mode MoE inference.
///
/// Prefetches expert weights for the next MoE layer while the current layer's
/// forward pass is computing on the GPU. Prefetched tensors are inserted into
/// the ExpertCache so that the next layer's streamingForward finds them as
/// cache hits, hiding disk latency behind compute time.
///
/// Implementation uses atomic flags with yield for synchronization.
/// A background thread polls for prefetch requests and loads expert weights
/// via PartialTensorReader into the ExpertCache.
const std = @import("std");
const expert_cache = @import("expert_cache.zig");
const expert_stream = @import("expert_stream.zig");
const safetensors_reader = @import("mlx").safetensors_reader;

const ExpertCache = expert_cache.ExpertCache;
const PartialTensorReader = safetensors_reader.PartialTensorReader;
const LayerExpertMeta = expert_stream.LayerExpertMeta;

pub const LayerPrefetcher = struct {
    allocator: std.mem.Allocator,
    reader: *PartialTensorReader,
    cache: *ExpertCache,
    layer_meta: []const LayerExpertMeta,

    /// Background thread handle
    thread: ?std.Thread = null,

    /// Atomic flags for lock-free synchronization
    /// 0 = idle, 1 = request pending, 2 = done, 3 = stopped
    state: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),

    /// Request parameters (written by main thread, read by worker)
    request_layer: usize = 0,
    /// Owned copy of expert_ids for the worker thread
    request_expert_ids_buf: ?[]u32 = null,
    request_expert_ids_len: usize = 0,

    const STATE_IDLE: u32 = 0;
    const STATE_PENDING: u32 = 1;
    const STATE_DONE: u32 = 2;
    const STATE_STOPPED: u32 = 3;

    /// Initialize the prefetcher and spawn the background worker thread.
    pub fn init(
        allocator: std.mem.Allocator,
        reader: *PartialTensorReader,
        cache: *ExpertCache,
        layer_meta: []const LayerExpertMeta,
    ) !LayerPrefetcher {
        var self = LayerPrefetcher{
            .allocator = allocator,
            .reader = reader,
            .cache = cache,
            .layer_meta = layer_meta,
        };

        // Spawn the background prefetch worker thread
        self.thread = std.Thread.spawn(.{}, prefetchWorker, .{&self}) catch |err| {
            std.log.warn("LayerPrefetcher: failed to spawn worker thread: {}, prefetching disabled", .{err});
            return self;
        };

        return self;
    }

    /// Stop the background thread and clean up.
    pub fn deinit(self: *LayerPrefetcher) void {
        self.state.store(STATE_STOPPED, .release);

        if (self.thread) |t| {
            t.join();
            self.thread = null;
        }

        if (self.request_expert_ids_buf) |buf| {
            self.allocator.free(buf);
            self.request_expert_ids_buf = null;
        }
    }

    /// Request prefetch of experts for a given layer (non-blocking).
    pub fn prefetch(self: *LayerPrefetcher, layer_idx: usize, expert_ids: []const u32) void {
        if (self.thread == null) return;
        if (layer_idx >= self.layer_meta.len) return;

        // Only queue if worker is idle (don't overwrite in-flight request)
        if (self.state.load(.acquire) != STATE_IDLE) return;

        // Copy expert_ids into our owned buffer
        if (self.request_expert_ids_buf) |buf| {
            if (buf.len < expert_ids.len) {
                self.allocator.free(buf);
                self.request_expert_ids_buf = self.allocator.alloc(u32, expert_ids.len) catch return;
            }
        } else {
            self.request_expert_ids_buf = self.allocator.alloc(u32, expert_ids.len) catch return;
        }
        @memcpy(self.request_expert_ids_buf.?[0..expert_ids.len], expert_ids);
        self.request_expert_ids_len = expert_ids.len;
        self.request_layer = layer_idx;

        // Signal the worker
        self.state.store(STATE_PENDING, .release);
    }

    /// Wait for the current prefetch request to complete (blocking).
    pub fn waitForCompletion(self: *LayerPrefetcher) void {
        if (self.thread == null) return;

        // Wait for completion with yield (avoids busy-wait CPU usage)
        while (self.state.load(.acquire) == STATE_PENDING) {
            std.Thread.yield() catch {};
        }
        // Reset to idle for next request
        if (self.state.load(.acquire) == STATE_DONE) {
            self.state.store(STATE_IDLE, .release);
        }
    }

    /// Background worker thread entry point.
    fn prefetchWorker(self: *LayerPrefetcher) void {
        while (self.state.load(.acquire) != STATE_STOPPED) {
            // Wait for a pending request with yield
            if (self.state.load(.acquire) != STATE_PENDING) {
                std.Thread.yield() catch {};
                continue;
            }

            // Capture request parameters
            const layer_idx = self.request_layer;
            const expert_ids = self.request_expert_ids_buf.?[0..self.request_expert_ids_len];

            // Perform the actual prefetch work
            self.doPrefetch(layer_idx, expert_ids);

            // Mark as done
            self.state.store(STATE_DONE, .release);
        }
    }

    /// Execute the prefetch: load all projections for the requested experts
    /// and insert them into the cache.
    fn doPrefetch(self: *LayerPrefetcher, layer_idx: usize, expert_ids: []const u32) void {
        if (layer_idx >= self.layer_meta.len) return;
        const meta = self.layer_meta[layer_idx];

        // Load weight projections
        const tensor_names = [_][]const u8{
            meta.gate_proj_name,
            meta.up_proj_name,
            meta.down_proj_name,
        };
        const scale_names = [_]?[]const u8{
            meta.gate_scales_name,
            meta.up_scales_name,
            meta.down_scales_name,
        };

        for (tensor_names) |tensor_name| {
            self.prefetchTensor(tensor_name, expert_ids, layer_idx);
        }
        for (scale_names) |maybe_name| {
            if (maybe_name) |tensor_name| {
                self.prefetchTensor(tensor_name, expert_ids, layer_idx);
            }
        }
    }

    /// Load a single tensor for the given experts and insert into cache.
    fn prefetchTensor(
        self: *LayerPrefetcher,
        tensor_name: []const u8,
        expert_ids: []const u32,
        layer_idx: usize,
    ) void {
        const name_hash = expert_cache.hashTensorName(tensor_name);
        const lx: u32 = @intCast(layer_idx);

        for (expert_ids) |eid| {
            const key = expert_cache.CacheKey{
                .layer_idx = lx,
                .tensor_name_hash = name_hash,
                .expert_id = eid,
            };

            // Skip if this expert row is already cached
            if (self.cache.get(key) != null) continue;

            // Read single expert row via partial reader
            const row = self.reader.readExpertRow(tensor_name, eid) catch |err| {
                std.log.warn("LayerPrefetcher: failed to read {s} expert {d} for layer {d}: {}", .{ tensor_name, eid, layer_idx, err });
                continue;
            };

            const sz = row.nbytes();
            self.cache.put(key, row, sz);
        }
    }
};

// ── Tests ──

test "LayerPrefetcher: prefetch for out-of-bounds layer is ignored" {
    const allocator = std.testing.allocator;

    var prefetcher = LayerPrefetcher{
        .allocator = allocator,
        .reader = undefined,
        .cache = undefined,
        .layer_meta = &[_]LayerExpertMeta{},
    };

    prefetcher.prefetch(0, &[_]u32{ 1, 2, 3 });
    prefetcher.prefetch(999, &[_]u32{ 1, 2, 3 });
    prefetcher.waitForCompletion();
    prefetcher.deinit();
}

test "LayerPrefetcher: init/deinit lifecycle without real I/O" {
    const allocator = std.testing.allocator;

    var prefetcher = LayerPrefetcher{
        .allocator = allocator,
        .reader = undefined,
        .cache = undefined,
        .layer_meta = &[_]LayerExpertMeta{},
    };

    try std.testing.expect(prefetcher.thread == null);
    try std.testing.expectEqual(@as(u32, 0), prefetcher.state.load(.acquire));

    prefetcher.deinit();
}
