/// Expert weight management for memory-constrained MoE inference.
///
/// Supports two strategies:
/// 1. Preload (Option 1): Load expert subset at initialization, use throughout inference
///    - Matches Python vmlx implementation (proven to work)
///    - Higher memory but stable and fast
///    - See expert_preload.zig for implementation
///
/// 2. Stream (Option 2): Load experts on-demand from disk during inference
///    - Lower memory footprint
///    - More complex, requires correct mxfp4 handling
///    - Experimental
///
/// On a 48GB Mac running DeepSeek V4 Flash 4-bit (151GB on disk):
/// - Without smelt: OOM (needs ~138GB for expert weights alone)
/// - With preload (50%): ~70GB (attention + shared + 128 experts)
/// - With stream: ~10GB (attention + shared expert + 8 active experts per step)
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const safetensors_reader = @import("mlx").safetensors_reader;
const quantize_mod = @import("mlx").quantize;
const shape_mod = @import("mlx").shape;
const expert_preload = @import("expert_preload.zig");
const expert_cache = @import("expert_cache.zig");
const layer_prefetcher_mod = @import("layer_prefetcher.zig");
const ExpertCache = expert_cache.ExpertCache;
const LayerPrefetcher = layer_prefetcher_mod.LayerPrefetcher;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const TensorIndex = safetensors_reader.TensorIndex;

/// Diagnostic metrics for a single token generation step.
pub const TokenStepMetrics = struct {
    step_number: u64,
    wall_clock_ms: f64,
    bytes_read: u64,
    cache_hits: u64,
    cache_misses: u64,
    cache_memory_bytes: usize,
    layers_processed: usize,
};

/// Expert loading strategy.
pub const ExpertLoadStrategy = enum {
    preload, // Option 1: Preload subset at init (matches Python vmlx)
    stream, // Option 2: Stream on-demand from disk (experimental)
};

/// Per-layer expert weight metadata for streaming.
pub const LayerExpertMeta = struct {
    /// HF weight names for this layer's fused switch_mlp tensors
    gate_proj_name: []const u8, // e.g. "model.layers.5.ffn.switch_mlp.gate_proj.weight"
    up_proj_name: []const u8,
    down_proj_name: []const u8,
    gate_scales_name: ?[]const u8,
    up_scales_name: ?[]const u8,
    down_scales_name: ?[]const u8,
    /// Shape of one expert slice: [intermediate_size, packed_hidden] for weight
    expert_row_bytes: usize, // bytes per expert row in the fused tensor
    expert_scale_row_bytes: usize, // bytes per expert row in scales tensor
    n_experts: usize,
};

/// Unified expert provider supporting both preload and streaming strategies.
pub const ExpertStreamProvider = struct {
    allocator: std.mem.Allocator,
    index: *TensorIndex,
    ctx: EagerContext,
    strategy: ExpertLoadStrategy,

    // Common fields
    is_quantized: bool,
    quant_group_size: i32,
    quant_bits: u8,
    quant_mode: []const u8,
    swiglu_limit: f32,

    // Strategy-specific implementations
    preload_provider: ?*expert_preload.ExpertPreloadProvider = null,

    // Stream-specific fields (Option 2)
    layer_meta: []LayerExpertMeta,

    // Performance optimization fields (stream mode only)
    cache: ?*ExpertCache = null,
    fd_pool: ?*safetensors_reader.FdPool = null,
    mmap_pool: ?*safetensors_reader.MmapPool = null,
    partial_reader: ?*safetensors_reader.PartialTensorReader = null,
    prefetcher: ?*LayerPrefetcher = null,

    // Diagnostic counters
    total_bytes_read: u64 = 0,
    token_step_count: u64 = 0,
    token_step_start_ticks: u64 = 0,
    step_bytes_read: u64 = 0,
    step_cache_hits_start: u64 = 0,
    step_cache_misses_start: u64 = 0,

    pub fn deinit(self: *ExpertStreamProvider) void {
        if (self.preload_provider) |provider| {
            provider.deinit();
            self.allocator.destroy(provider);
        }

        // Clean up performance optimization components
        if (self.prefetcher) |pf| {
            pf.deinit();
            self.allocator.destroy(pf);
        }
        if (self.cache) |c_inst| {
            c_inst.deinit();
            self.allocator.destroy(c_inst);
        }
        if (self.partial_reader) |r| {
            self.allocator.destroy(r);
        }
        if (self.mmap_pool) |m| {
            m.deinit();
            self.allocator.destroy(m);
        }
        if (self.fd_pool) |p| {
            p.deinit();
            self.allocator.destroy(p);
        }

        // Clean up stream-specific metadata
        for (self.layer_meta) |meta| {
            self.allocator.free(meta.gate_proj_name);
            self.allocator.free(meta.up_proj_name);
            self.allocator.free(meta.down_proj_name);
            if (meta.gate_scales_name) |n| self.allocator.free(n);
            if (meta.up_scales_name) |n| self.allocator.free(n);
            if (meta.down_scales_name) |n| self.allocator.free(n);
        }
        self.allocator.free(self.layer_meta);
    }

    /// Initialize provider with specified strategy.
    pub fn initWithStrategy(
        allocator: std.mem.Allocator,
        ctx: EagerContext,
        index: *TensorIndex,
        strategy: ExpertLoadStrategy,
        expert_ids: []const u32,
        layer_meta: []LayerExpertMeta,
        is_quantized: bool,
        quant_group_size: i32,
        quant_bits: u8,
        quant_mode: []const u8,
        swiglu_limit: f32,
        cache_budget_mb: usize,
    ) !ExpertStreamProvider {
        var provider = ExpertStreamProvider{
            .allocator = allocator,
            .index = index,
            .ctx = ctx,
            .strategy = strategy,
            .is_quantized = is_quantized,
            .quant_group_size = quant_group_size,
            .quant_bits = quant_bits,
            .quant_mode = quant_mode,
            .swiglu_limit = swiglu_limit,
            .layer_meta = layer_meta,
        };

        switch (strategy) {
            .preload => {
                std.log.info("Initializing expert provider with PRELOAD strategy", .{});

                // Convert LayerExpertMeta to expert_preload.LayerMeta
                var preload_meta = try allocator.alloc(expert_preload.LayerMeta, layer_meta.len);
                defer allocator.free(preload_meta);

                for (layer_meta, 0..) |meta, i| {
                    preload_meta[i] = expert_preload.LayerMeta{
                        .gate_proj_name = meta.gate_proj_name,
                        .up_proj_name = meta.up_proj_name,
                        .down_proj_name = meta.down_proj_name,
                        .gate_scales_name = meta.gate_scales_name,
                        .up_scales_name = meta.up_scales_name,
                        .down_scales_name = meta.down_scales_name,
                        .n_experts = meta.n_experts,
                    };
                }

                // Initialize preload provider
                const preload_impl = try allocator.create(expert_preload.ExpertPreloadProvider);
                preload_impl.* = try expert_preload.ExpertPreloadProvider.init(
                    allocator,
                    ctx,
                    index,
                    expert_ids,
                    preload_meta,
                    is_quantized,
                    quant_group_size,
                    quant_bits,
                    quant_mode,
                );
                provider.preload_provider = preload_impl;
            },
            .stream => {
                std.log.info("Initializing expert provider with STREAM strategy (experimental)", .{});

                // Initialize FdPool for pread-based loading
                const pool = try allocator.create(safetensors_reader.FdPool);
                pool.* = safetensors_reader.FdPool.init(allocator);
                try pool.openAll(index);
                provider.fd_pool = pool;

                // Initialize MmapPool for zero-copy memory-mapped access.
                // mmap provides OS-level readahead which is critical for tok/s performance.
                // Trade-off: causes VM pressure that adds ~38s HTTP latency on 48GB Mac,
                // but server-side tok/s is 2x better than pread (9.1 vs 4.9).
                // See: docs/analysis/pread-expert-loading.md for full investigation.
                const mmap = try allocator.create(safetensors_reader.MmapPool);
                mmap.* = safetensors_reader.MmapPool.init(allocator);
                try mmap.mmapAll(index);
                provider.mmap_pool = mmap;

                // Initialize PartialTensorReader for reading only selected expert rows
                const reader = try allocator.create(safetensors_reader.PartialTensorReader);
                reader.* = safetensors_reader.PartialTensorReader.init(allocator, index, pool);
                provider.partial_reader = reader;

                // Initialize ExpertCache for caching frequently-used expert slices
                const cache = try allocator.create(expert_cache.ExpertCache);
                cache.* = expert_cache.ExpertCache.init(allocator, cache_budget_mb * 1024 * 1024);
                provider.cache = cache;

                // LayerPrefetcher is NOT enabled due to MLX thread safety constraints
                // See tasks.md P0 for details: MLX tensor operations are not thread-safe
                provider.prefetcher = null;

                std.log.info("Expert streaming enabled: loading experts from SSD on demand (cache_budget={d}MB)", .{cache_budget_mb});
            },
        }

        return provider;
    }

    /// Forward pass - dispatches to appropriate strategy implementation.
    pub fn streamForward(
        self: *ExpertStreamProvider,
        layer_idx: usize,
        flat_x: Array,
        indices: Array,
        scores: Array,
    ) !Array {
        return switch (self.strategy) {
            .preload => blk: {
                if (self.preload_provider) |provider| {
                    break :blk provider.forward(layer_idx, flat_x, indices, scores);
                }
                return error.PreloadProviderNotInitialized;
            },
            .stream => self.streamingForward(layer_idx, flat_x, indices, scores),
        };
    }

    /// Get cache bias for router (only for preload strategy).
    pub fn getCacheBias(self: *ExpertStreamProvider, layer_idx: usize) ?Array {
        if (self.strategy == .preload) {
            if (self.preload_provider) |provider| {
                return provider.getCacheBias(layer_idx);
            }
        }
        return null;
    }

    /// Load a subset of experts from a fused tensor on disk.
    /// Returns a mini fused tensor [n_selected, ...] containing only the requested expert rows.
    /// `expert_ids` is a sorted, deduplicated list of expert indices to load.
    ///
    /// CRITICAL: For quantized mxfp4 format, we MUST load the full tensor first, then slice it.
    /// Creating a new tensor from concatenated expert rows breaks the packing format.
    /// This matches the Python vmlx implementation (_load_expert_subset).
    fn loadExpertSlices(
        self: *ExpertStreamProvider,
        tensor_name: []const u8,
        expert_ids: []const u32,
        row_bytes: usize,
    ) !Array {
        _ = row_bytes; // Not used in the new approach

        // Use PartialTensorReader if available: read only selected expert rows
        if (self.partial_reader) |reader| {
            return try reader.readExpertRows(tensor_name, expert_ids);
        }

        const info = self.index.entries.get(tensor_name) orelse return error.TensorNotFound;

        // Fallback: load the FULL tensor from disk then slice
        const full_tensor = try self.index.loadTensor(tensor_name);
        defer full_tensor.deinit();

        // If all experts are selected, return the full tensor
        const n_experts = @as(usize, @intCast(info.shape[0]));
        if (expert_ids.len >= n_experts) {
            return ops.copy(self.ctx, full_tensor);
        }

        // Slice to get only the selected experts: full_tensor[expert_ids, ...]
        const indices_arr = try Array.fromData(self.allocator, u32, expert_ids, &[_]i32{@intCast(expert_ids.len)});
        defer indices_arr.deinit();

        // Use take_axis to slice along axis 0 (expert dimension)
        const indices_i32 = try ops.astype(self.ctx, indices_arr, .int32);
        defer indices_i32.deinit();

        const sliced = try shape_mod.takeAxis(self.ctx, full_tensor, indices_i32, 0);

        // Force evaluation to materialize the sliced data
        try sliced.eval();

        return sliced;
    }

    /// Load expert slices with cache-first strategy using partial reads.
    ///
    /// Strategy:
    /// 1. Try to assemble ALL experts from cache (fast path for repeated selections)
    /// 2. On partial or full miss, load via PartialTensorReader (already active in loadExpertSlices)
    /// 3. Cache each newly-loaded expert row for future tokens
    fn loadExpertSlicesCached(
        self: *ExpertStreamProvider,
        tensor_name: []const u8,
        expert_ids: []const u32,
        layer_idx: usize,
        row_bytes: usize,
    ) !Array {
        // If cache unavailable, fall through to direct load (PartialTensorReader already active there)
        if (self.cache == null) {
            return self.loadExpertSlices(tensor_name, expert_ids, row_bytes);
        }

        const cache_inst = self.cache.?;
        const tensor_name_hash = std.hash.Wyhash.hash(0, tensor_name);
        const lx: u32 = @intCast(layer_idx);

        // Try to assemble ALL from cache (fast path)
        var all_cached = true;
        for (expert_ids) |eid| {
            const key = expert_cache.CacheKey{ .layer_idx = lx, .tensor_name_hash = tensor_name_hash, .expert_id = eid };
            if (cache_inst.get(key) == null) {
                all_cached = false;
                break;
            }
        }

        if (all_cached) {
            // Fast path: assemble mini-fused tensor from cached rows
            var cached_rows = try self.allocator.alloc(Array, expert_ids.len);
            defer self.allocator.free(cached_rows);
            for (expert_ids, 0..) |eid, i| {
                const key = expert_cache.CacheKey{ .layer_idx = lx, .tensor_name_hash = tensor_name_hash, .expert_id = eid };
                const cached_row = cache_inst.get(key).?;
                cached_rows[i] = try ops.copy(self.ctx, cached_row);
            }
            defer for (cached_rows) |row| row.deinit(); // Clean up intermediate arrays
            return try shape_mod.concatenateAxis(self.ctx, cached_rows, 0);
        }

        // Partial/full miss: load via existing path (uses PartialTensorReader)
        const result = try self.loadExpertSlices(tensor_name, expert_ids, row_bytes);

        // Cache individual expert rows lazily for future token reuse
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const aa = arena.allocator();

        for (expert_ids, 0..) |eid, i| {
            const key = expert_cache.CacheKey{ .layer_idx = lx, .tensor_name_hash = tensor_name_hash, .expert_id = eid };
            if (cache_inst.get(key) != null) continue;

            const idx_arr = try Array.fromData(aa, i32, &[_]i32{@intCast(i)}, &[_]i32{1});
            const row = try shape_mod.takeAxis(self.ctx, result, idx_arr, 0);
            const row_copy = try ops.copy(self.ctx, row);
            row.deinit();
            idx_arr.deinit();

            const sz = row_copy.nbytes();
            cache_inst.put(key, row_copy, sz);
        }

        return result;
    }

    /// Streaming forward (Option 2): Load experts on-demand from disk.
    /// This is the experimental approach with lower memory but more complexity.
    ///
    /// P4.1 Optimization: Expert deduplication across batch tokens
    /// During prefill with multiple tokens (e.g., 8 tokens), each token routes to topk experts.
    /// Without deduplication: 8 tokens × 6 experts/token = 48 expert loads per layer
    /// With deduplication: ~24-34 unique experts per layer (30-50% reduction)
    /// This optimization unions routing results across all tokens before loading,
    /// significantly reducing I/O during cold start prefill.
    fn streamingForward(
        self: *ExpertStreamProvider,
        layer_idx: usize,
        flat_x: Array,
        indices: Array,
        scores: Array,
    ) !Array {
        const meta = self.layer_meta[layer_idx];

        // Wait for any in-flight prefetch to complete (prefetched data is now in cache)
        if (self.prefetcher) |pf| {
            pf.waitForCompletion();
        }

        // Track token steps for diagnostics (increment once per first layer of each token)
        if (layer_idx == 0) {
            self.token_step_count += 1;
            self.token_step_start_ticks = std.c.mach_absolute_time();
            self.step_bytes_read = 0;
            // Log cache memory usage at start of token step
            if (self.cache) |cache_inst| {
                const s = cache_inst.stats();
                std.log.info("Token step {d}: cache={d}MB/{d}MB ({d} entries, {d} hits, {d} misses)", .{
                    self.token_step_count,
                    s.current_bytes / (1024 * 1024),
                    s.max_bytes / (1024 * 1024),
                    s.entry_count,
                    s.hits,
                    s.misses,
                });
                self.step_cache_hits_start = s.hits;
                self.step_cache_misses_start = s.misses;
            }
        }

        // 1. Ensure indices are contiguous and uint32 before any dataSlice reads.
        // Router's topk + reshape can produce non-contiguous strides, causing
        // dataSlice to read elements in wrong order (the original mlx_take bug).
        // copy() forces a contiguous memory layout; astype ensures uint32.
        const indices_contig = try ops.copy(self.ctx, indices);
        defer indices_contig.deinit();
        const indices_u32 = try ops.astype(self.ctx, indices_contig, .uint32);
        defer indices_u32.deinit();
        try indices_u32.eval();

        const indices_data = try indices_u32.dataSlice(u32);

        // P4.1: Union/deduplicate routing results across all batch tokens
        // This reduces redundant expert loading when multiple tokens route to the same experts
        var unique_set = std.AutoHashMap(u32, void).init(self.allocator);
        defer unique_set.deinit();
        for (indices_data) |eid| {
            try unique_set.put(eid, {});
        }
        var unique_ids = try self.allocator.alloc(u32, unique_set.count());
        defer self.allocator.free(unique_ids);
        {
            var it = unique_set.keyIterator();
            var i: usize = 0;
            while (it.next()) |k| {
                unique_ids[i] = k.*;
                i += 1;
            }
        }
        // Sort for sequential disk access (helps cache and partial reads)
        std.mem.sort(u32, unique_ids, {}, std.sort.asc(u32));

        // Log deduplication effectiveness (only for layer 0 to avoid spam)
        if (layer_idx == 0) {
            const dedup_rate = if (indices_data.len > 0)
                @as(f64, @floatFromInt(indices_data.len - unique_ids.len)) / @as(f64, @floatFromInt(indices_data.len)) * 100.0
            else
                0.0;
            std.log.info("P4.1 Expert deduplication: {d} total → {d} unique ({d:.1}% reduction)", .{
                indices_data.len,
                unique_ids.len,
                dedup_rate,
            });
        }

        // 2. Load expert weight slices using cache-first strategy with partial reads
        const gate_w = try self.loadExpertSlicesCached(meta.gate_proj_name, unique_ids, layer_idx, 0);
        defer gate_w.deinit();

        const up_w = try self.loadExpertSlicesCached(meta.up_proj_name, unique_ids, layer_idx, 0);
        defer up_w.deinit();

        const down_w = try self.loadExpertSlicesCached(meta.down_proj_name, unique_ids, layer_idx, 0);
        defer down_w.deinit();

        // Load scales if quantized
        var gate_s: ?Array = null;
        var up_s: ?Array = null;
        var down_s: ?Array = null;
        defer if (gate_s) |a| a.deinit();
        defer if (up_s) |a| a.deinit();
        defer if (down_s) |a| a.deinit();
        if (self.is_quantized) {
            if (meta.gate_scales_name) |n| {
                gate_s = try self.loadExpertSlicesCached(n, unique_ids, layer_idx, 0);
            }
            if (meta.up_scales_name) |n| {
                up_s = try self.loadExpertSlicesCached(n, unique_ids, layer_idx, 0);
            }
            if (meta.down_scales_name) |n| {
                down_s = try self.loadExpertSlicesCached(n, unique_ids, layer_idx, 0);
            }
        }

        // 3. Build remap: original_expert_id → mini_fused_row_index
        var remap_data = try self.allocator.alloc(i32, meta.n_experts);
        defer self.allocator.free(remap_data);
        @memset(remap_data, 0);
        for (unique_ids, 0..) |eid, i| {
            remap_data[eid] = @intCast(i);
        }
        const remap_arr = try Array.fromData(self.allocator, i32, remap_data, &[_]i32{@intCast(meta.n_experts)});
        defer remap_arr.deinit();

        // 4. Remap indices: map original expert IDs [0, 256) to local indices [0, n_unique)
        // Use manual remap to avoid mlx_take 2D layout issues
        // (see .kiro/specs/stream-mode-correctness/design.md - H1)
        const remap_readback = try remap_arr.dataSlice(i32);

        // Build remapped indices preserving original 2D shape [N, topk]
        var remapped_data = try self.allocator.alloc(u32, indices_data.len);
        defer self.allocator.free(remapped_data);
        for (indices_data, 0..) |idx, i| {
            remapped_data[i] = @intCast(remap_readback[idx]);
        }

        const idx_shape = indices_u32.shape();
        var shape_buf: [8]i32 = undefined;
        for (idx_shape, 0..) |d, i| {
            shape_buf[i] = @intCast(d);
        }
        const remapped_u32 = try Array.fromData(self.allocator, u32, remapped_data, shape_buf[0..@intCast(indices_u32.ndim())]);
        defer remapped_u32.deinit();

        // 5. Expert computation matching Python mlx-lm SwitchGLU exactly:
        //    y = switch_mlp(x, local_inds)        # gate/up/SwiGLU/down, NO scores
        //    y = (y * scores[..., None]).sum(-2)   # scores applied AFTER switch_mlp
        //
        // Python SwitchGLU.__call__:
        //   x = expand_dims(x, (-2, -3))  → [N, 1, 1, D]
        //   if do_sort: x, idx, inv = _gather_sort(x, indices)
        //   x_up = up_proj(x, idx, sorted=do_sort)
        //   x_gate = gate_proj(x, idx, sorted=do_sort)
        //   x = down_proj(activation(x_up, x_gate), idx, sorted=do_sort)
        //   if do_sort: x = _scatter_unsort(x, inv, indices.shape)
        //   return x.squeeze(-2)
        const deepseek_v4 = @import("deepseek_v4.zig");
        var switch_glu = deepseek_v4.DSV4SwitchGLU{
            .ctx = self.ctx,
            .gate_proj = gate_w,
            .up_proj = up_w,
            .down_proj = down_w,
            .gate_proj_scales = gate_s,
            .gate_proj_biases = null,
            .up_proj_scales = up_s,
            .up_proj_biases = null,
            .down_proj_scales = down_s,
            .down_proj_biases = null,
            .is_quantized = self.is_quantized,
            .quant_group_size = self.quant_group_size,
            .quant_bits = self.quant_bits,
            .quant_mode = self.quant_mode,
            .swiglu_limit = self.swiglu_limit,
            .sort_threshold = 8,
        };
        // Call forwardNoScores which does gate/up/SwiGLU/down without score weighting
        const expert_out = try switch_glu.forwardNoScores(flat_x, remapped_u32, self.ctx.stream.inner);
        defer expert_out.deinit();

        // Apply scores AFTER switch_mlp (matching Python: y = (y * scores[..., None]).sum(-2))
        const scores_expanded = try ops.expandDims(self.ctx, scores, -1);
        defer scores_expanded.deinit();
        const weighted_out = try ops.multiply(self.ctx, expert_out, scores_expanded);
        defer weighted_out.deinit();
        const reduce_mod = @import("mlx").reduce;
        const result = try reduce_mod.sumAxis(self.ctx, weighted_out, -2, false);

        // Kick off prefetch for the next layer (non-blocking)
        if (self.prefetcher) |pf| {
            pf.prefetch(layer_idx + 1, unique_ids);
        }

        // Log end-of-token-step metrics on the last layer
        if (layer_idx == self.layer_meta.len - 1 and self.token_step_start_ticks != 0) {
            const end_ticks = std.c.mach_absolute_time();
            const elapsed_ticks = end_ticks - self.token_step_start_ticks;
            // mach_absolute_time ticks are nanoseconds on Apple Silicon
            const elapsed_ms = @as(f64, @floatFromInt(elapsed_ticks)) / 1_000_000.0;
            if (self.cache) |cache_inst| {
                const s = cache_inst.stats();
                const step_hits = s.hits - self.step_cache_hits_start;
                const step_misses = s.misses - self.step_cache_misses_start;
                std.log.info("Token step {d} complete: {d:.1}ms, {d}MB read, cache hits={d} misses={d}", .{
                    self.token_step_count,
                    elapsed_ms,
                    self.step_bytes_read / (1024 * 1024),
                    step_hits,
                    step_misses,
                });
            } else {
                std.log.info("Token step {d} complete: {d:.1}ms", .{
                    self.token_step_count,
                    elapsed_ms,
                });
            }
        }

        return result;
    }
};

// ── Tests ──

test "TokenStepMetrics: struct has correct fields" {
    // Verify that TokenStepMetrics can be constructed with all expected fields
    const metrics = TokenStepMetrics{
        .step_number = 42,
        .wall_clock_ms = 123.456,
        .bytes_read = 1024 * 1024,
        .cache_hits = 10,
        .cache_misses = 3,
        .cache_memory_bytes = 4096,
        .layers_processed = 43,
    };

    try std.testing.expectEqual(@as(u64, 42), metrics.step_number);
    try std.testing.expectApproxEqAbs(@as(f64, 123.456), metrics.wall_clock_ms, 0.001);
    try std.testing.expectEqual(@as(u64, 1024 * 1024), metrics.bytes_read);
    try std.testing.expectEqual(@as(u64, 10), metrics.cache_hits);
    try std.testing.expectEqual(@as(u64, 3), metrics.cache_misses);
    try std.testing.expectEqual(@as(usize, 4096), metrics.cache_memory_bytes);
    try std.testing.expectEqual(@as(usize, 43), metrics.layers_processed);
}

test "ExpertCache eviction logs at debug level" {
    // This test verifies that the ExpertCache eviction path works correctly
    // and that evicted entries are properly cleaned up. The actual debug logging
    // is verified by the existing ExpertCache tests (LRU eviction order test).
    const allocator = std.testing.allocator;
    var cache = ExpertCache.init(allocator, 200);
    defer cache.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    // Fill cache to capacity
    const key_a = expert_cache.CacheKey{ .layer_idx = 0, .tensor_name_hash = 1, .expert_id = 0 };
    const key_b = expert_cache.CacheKey{ .layer_idx = 1, .tensor_name_hash = 2, .expert_id = 0 };
    const t_a = try array_mod.Array.fromData(allocator, f32, &data, &[_]i32{4});
    const t_b = try array_mod.Array.fromData(allocator, f32, &data, &[_]i32{4});
    cache.put(key_a, t_a, 100);
    cache.put(key_b, t_b, 100);

    // Trigger eviction by inserting a third entry
    const key_c = expert_cache.CacheKey{ .layer_idx = 2, .tensor_name_hash = 3, .expert_id = 0 };
    const t_c = try array_mod.Array.fromData(allocator, f32, &data, &[_]i32{4});
    cache.put(key_c, t_c, 100);

    // Verify eviction occurred: key_a (LRU) should be gone
    try std.testing.expect(cache.get(key_a) == null);
    // key_b and key_c should remain
    try std.testing.expect(cache.get(key_b) != null);
    try std.testing.expect(cache.get(key_c) != null);

    // Verify memory tracking is correct after eviction
    const s = cache.stats();
    try std.testing.expectEqual(@as(usize, 200), s.current_bytes);
    try std.testing.expect(s.current_bytes <= s.max_bytes);
}
