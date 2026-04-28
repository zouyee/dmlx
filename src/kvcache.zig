/// Plugin-based KV Cache Management System.
///
/// Provides runtime-polymorphic KV cache strategies with a unified interface.
/// Models and attention layers depend only on `KVCacheStrategy`, not on concrete types.
///
/// Available strategies:
///   - StandardKVCache: pre-allocated fixed-length buffer
///   - RotatingKVCache: sliding-window circular buffer
///   - QuantizedKVCache: 8-bit / 4-bit quantized storage
///   - PagedKVCache: block/page-based for continuous batching
///   - RadixKVCache: prefix-tree for multi-turn conversation
///
/// Usage:
///   const kvcache = @import("kvcache.zig");
///   var manager = try kvcache.CacheManager.init(allocator, num_layers, config, kvcache.standard.create, stream);
///   defer manager.deinit();
///
///   // In attention forward:
///   const kv = try manager.caches[layer_idx].updateAndFetch(keys, values, stream);
const std = @import("std");

// Core interface
pub const interface = @import("kvcache/interface.zig");
pub const KVCacheStrategy = interface.KVCacheStrategy;
pub const KVSlice = interface.KVSlice;
pub const LayerConfig = interface.LayerConfig;
pub const StrategyFactory = interface.StrategyFactory;

// Manager
pub const manager = @import("kvcache/manager.zig");
pub const CacheManager = manager.CacheManager;

// Strategy implementations
pub const standard = @import("kvcache/standard.zig");
pub const StandardKVCache = standard.StandardKVCache;

pub const rotating = @import("kvcache/rotating.zig");
pub const RotatingKVCache = rotating.RotatingKVCache;

pub const quantized = @import("kvcache/quantized.zig");
pub const QuantizedKVCache = quantized.QuantizedKVCache;

pub const paged = @import("kvcache/paged.zig");
pub const PagedKVCache = paged.PagedKVCache;
pub const BlockManager = paged.BlockManager;
pub const Block = paged.Block;
pub const default_page_size = paged.default_page_size;

pub const radix = @import("kvcache/radix.zig");
pub const RadixKVCache = radix.RadixKVCache;
pub const MatchResult = radix.MatchResult;

pub const tiered = @import("kvcache/tiered.zig");
pub const TieredKVCache = tiered.TieredKVCache;
pub const createTiered = tiered.createTiered;
pub const createTieredWithConfig = tiered.createTieredWithConfig;

pub const prefix_disk = @import("kvcache/prefix_disk.zig");
pub const PrefixDiskCache = prefix_disk.PrefixDiskCache;
pub const PrefixBlockData = prefix_disk.PrefixBlockData;
pub const RestoredPrefix = prefix_disk.RestoredPrefix;

pub const turboquant = @import("kvcache/turboquant.zig");
pub const TurboQuantState = turboquant.TurboQuantState;
pub const TurboQuantConfig = turboquant.TurboQuantConfig;
pub const TurboQuantCodebook = turboquant.Codebook;

pub const dsv4_cache = @import("kvcache/deepseek_v4_cache.zig");
pub const DeepseekV4Cache = dsv4_cache.DeepseekV4Cache;
pub const BranchState = dsv4_cache.BranchState;
pub const createDeepseekV4Cache = dsv4_cache.createDeepseekV4Cache;

// Convenience factory re-exports
pub const createStandard = standard.createStandard;
pub const createRotating = rotating.createRotating;
pub const createRotatingWithWindow = rotating.createRotatingWithWindow;
pub const createQuantized = quantized.createQuantized;
pub const createQuantized8Bit = quantized.createQuantized8Bit;
pub const createQuantized4Bit = quantized.createQuantized4Bit;
pub const createQuantized16Bit = quantized.createQuantized16Bit;
pub const QuantConfig = quantized.QuantConfig;
pub const createPaged = paged.createPaged;
pub const createPagedWithSize = paged.createPagedWithSize;
pub const createPagedQuantized = paged.createPagedQuantized;
pub const createPagedQuantized4Bit = paged.createPagedQuantized4Bit;
pub const createRadix = radix.createRadix;
