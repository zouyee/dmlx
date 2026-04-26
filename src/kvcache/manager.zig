/// CacheManager — Manages per-layer KV cache strategies for a transformer model.
///
/// A transformer with N layers has N KV caches (one per attention layer).
/// CacheManager owns all caches and provides batch operations across all layers:
///   - reset all caches (e.g., new conversation)
///   - filter all caches (e.g., continuous batching eviction)
///   - query total memory usage
///
/// Usage:
///   var manager = try CacheManager.init(allocator, num_layers, config, factory, stream);
///   defer manager.deinit();
///
///   // Per-layer access during forward pass
///   const kv = try manager.caches[layer_idx].updateAndFetch(keys, values, stream);
const std = @import("std");
const c = @import("../c.zig");
const iface = @import("interface.zig");

const KVCacheStrategy = iface.KVCacheStrategy;
const LayerConfig = iface.LayerConfig;
const StrategyFactory = iface.StrategyFactory;

/// Manages a collection of per-layer KV cache strategies.
pub const CacheManager = struct {
    allocator: std.mem.Allocator,
    caches: []KVCacheStrategy,

    /// Initialize a CacheManager with `num_layers` caches, all using the same factory.
    pub fn init(
        allocator: std.mem.Allocator,
        num_layers: usize,
        config: LayerConfig,
        factory: StrategyFactory,
        stream: c.c.mlx_stream,
    ) !CacheManager {
        const caches = try allocator.alloc(KVCacheStrategy, num_layers);
        errdefer allocator.free(caches);

        for (0..num_layers) |i| {
            caches[i] = try factory(allocator, config, stream);
            errdefer {
                // Roll back already-created caches on failure
                for (0..i) |j| {
                    caches[j].deinit(allocator);
                }
            }
        }

        return .{
            .allocator = allocator,
            .caches = caches,
        };
    }

    /// Initialize with a per-layer configuration array (e.g., GQA different heads per layer).
    pub fn initPerLayer(
        allocator: std.mem.Allocator,
        configs: []const LayerConfig,
        factory: StrategyFactory,
        stream: c.c.mlx_stream,
    ) !CacheManager {
        const caches = try allocator.alloc(KVCacheStrategy, configs.len);
        errdefer allocator.free(caches);

        for (configs, 0..) |cfg, i| {
            caches[i] = try factory(allocator, cfg, stream);
            errdefer {
                for (0..i) |j| {
                    caches[j].deinit(allocator);
                }
            }
        }

        return .{
            .allocator = allocator,
            .caches = caches,
        };
    }

    /// Release all caches and the manager itself.
    pub fn deinit(self: *CacheManager) void {
        for (self.caches) |cache| {
            cache.deinit(self.allocator);
        }
        self.allocator.free(self.caches);
        self.caches = &.{};
    }

    /// Number of layers (caches).
    pub fn numLayers(self: CacheManager) usize {
        return self.caches.len;
    }

    /// Reset all caches to empty state.
    pub fn resetAll(self: CacheManager) void {
        for (self.caches) |cache| {
            cache.reset();
        }
    }

    /// Filter all caches to keep only the specified batch indices.
    /// Used in continuous batching when sequences complete or are evicted.
    pub fn filterAll(
        self: CacheManager,
        indices: []const usize,
    ) !void {
        for (self.caches, 0..) |cache, i| {
            cache.filter(indices, self.allocator) catch |err| {
                // On partial failure, we can't easily roll back.
                // Log and continue — caller should handle by resetting.
                std.log.warn("CacheManager: filter failed on layer {d}: {s}", .{ i, @errorName(err) });
                return err;
            };
        }
    }

    /// Whether all caches support batch filtering.
    pub fn allSupportFilter(self: CacheManager) bool {
        for (self.caches) |cache| {
            if (!cache.supportsFilter()) return false;
        }
        return true;
    }

    /// Return the maximum current sequence length across all layers.
    /// In a well-behaved system all layers should have the same length.
    pub fn maxSeqLen(self: CacheManager) usize {
        var max_len: usize = 0;
        for (self.caches) |cache| {
            const len = cache.currentLen();
            if (len > max_len) max_len = len;
        }
        return max_len;
    }
};
