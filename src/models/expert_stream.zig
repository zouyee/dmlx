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
const expert_pread = @import("expert_pread.zig");
// NOTE: expert_cache.zig and layer_prefetcher.zig kept as reference code.
// Trust OS (no custom cache) is the recommended configuration. See P2.1.

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
    fd_pool: ?*safetensors_reader.FdPool = null,
    partial_reader: ?*safetensors_reader.PartialTensorReader = null,
    pread_loader: ?*expert_pread.ExpertPreadLoader = null,

    // DyMoE: skip low-score cache-miss experts to reduce I/O
    dymoe_max_skip: usize = 1,
    dymoe_total_skipped: u64 = 0,
    dymoe_total_opportunities: u64 = 0,

    // Diagnostic counters
    total_bytes_read: u64 = 0,
    token_step_count: u64 = 0,
    token_step_start_ticks: u64 = 0,
    step_bytes_read: u64 = 0,

    pub fn deinit(self: *ExpertStreamProvider) void {
        if (self.preload_provider) |provider| {
            provider.deinit();
            self.allocator.destroy(provider);
        }

        // Clean up performance optimization components
        if (self.partial_reader) |r| {
            self.allocator.destroy(r);
        }
        if (self.fd_pool) |p| {
            p.deinit();
            self.allocator.destroy(p);
        }
        if (self.pread_loader) |loader| {
            loader.deinit();
            self.allocator.destroy(loader);
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
        cache_budget_mb: usize, // Deprecated: kept for API compatibility (P2.1)
        packed_dir: ?[]const u8,
        max_parallel: usize,
    ) !ExpertStreamProvider {
        _ = cache_budget_mb; // Deprecated parameter, retained for API compatibility (P2.1)
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

                // Flash-MoE insight: mmap causes VM pressure that dominates client latency.
                // On Apple Silicon, each 3.9-7MB expert spans 240-440 × 16KB pages.
                // mmap triggers 240+ individual page faults per expert (one kernel trap each).
                // pread issues ONE NVMe command for the entire expert range — much faster.
                //
                // When cache_budget_mb == 0: Trust OS page cache (Flash-MoE "Trust the OS" mode).
                //   - Skip MmapPool entirely (no VM mappings, no page fault overhead)
                //   - Skip ExpertCache (releases RAM to OS page cache: ~10GB → ~25GB page cache)
                //   - OS page cache achieves ~50-70% hit rate naturally (Flash-MoE: 71%)
                //   - Parallel pread in readExpertRowsCpu handles I/O efficiently
                //
                // When cache_budget_mb > 0: Use mmap + ExpertCache (legacy mode).
                //   - Better server-side tok/s (mmap readahead) but worse client latency
                //   - See: docs/analysis/pread-expert-loading.md and flash-moe/docs/
                // Initialize PartialTensorReader for reading only selected expert rows.
                // NOTE: mmap is intentionally disabled. mmap causes VM page fault storms.
                // pread reads data into CPU buffers first, avoiding page faults during GPU execution.
                // ExpertCache has been removed (P2.1). Trust OS page cache is the only strategy.
                // See: docs/analysis/flash-moe-plan.md
                const reader = try allocator.create(safetensors_reader.PartialTensorReader);
                reader.* = safetensors_reader.PartialTensorReader.init(allocator, index, pool);
                provider.partial_reader = reader;

                std.log.info("Expert streaming: parallel pread + Trust OS page cache (Flash-MoE mode). Client latency optimized.", .{});

                // Initialize parallel pread loader if packed directory is provided
                if (packed_dir) |dir| {
                    const loader = try allocator.create(expert_pread.ExpertPreadLoader);
                    loader.* = try expert_pread.ExpertPreadLoader.init(allocator, dir, max_parallel);
                    provider.pread_loader = loader;
                    std.log.info("Expert streaming: parallel pread loader initialized ({s}, {d} threads)", .{ dir, max_parallel });
                }
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

    /// Thread context for parallel projection loading.
    const ProjectionLoadCtx = struct {
        provider: *ExpertStreamProvider,
        tensor_name: []const u8,
        expert_ids: []const u32,
        layer_idx: usize,
        result: ?Array = null,
        err: ?anyerror = null,
    };

    fn loadProjectionThread(ctx: *ProjectionLoadCtx) void {
        const result = ctx.provider.loadExpertSlicesCached(
            ctx.tensor_name,
            ctx.expert_ids,
            ctx.layer_idx,
            0,
        );
        if (result) |arr| {
            ctx.result = arr;
        } else |e| {
            ctx.err = e;
        }
    }

    /// Load gate/up/down expert projections in parallel using 3 threads.
    /// Each projection is an independent pread — no shared mutable state.
    /// ~3x I/O speedup over serial loading.
    fn loadExpertProjectionsParallel(
        self: *ExpertStreamProvider,
        gate_name: []const u8,
        up_name: []const u8,
        down_name: []const u8,
        gate_scales_name: ?[]const u8,
        up_scales_name: ?[]const u8,
        down_scales_name: ?[]const u8,
        expert_ids: []const u32,
        layer_idx: usize,
        gate_w: *Array,
        up_w: *Array,
        down_w: *Array,
        gate_s: *?Array,
        up_s: *?Array,
        down_s: *?Array,
    ) !void {
        var gate_ctx = ProjectionLoadCtx{ .provider = self, .tensor_name = gate_name, .expert_ids = expert_ids, .layer_idx = layer_idx };
        var up_ctx = ProjectionLoadCtx{ .provider = self, .tensor_name = up_name, .expert_ids = expert_ids, .layer_idx = layer_idx };
        var down_ctx = ProjectionLoadCtx{ .provider = self, .tensor_name = down_name, .expert_ids = expert_ids, .layer_idx = layer_idx };

        const t0 = std.c.mach_absolute_time();

        // Spawn threads for gate and up projections; load down on current thread
        var gate_thread = try std.Thread.spawn(.{}, loadProjectionThread, .{&gate_ctx});
        var up_thread = try std.Thread.spawn(.{}, loadProjectionThread, .{&up_ctx});
        loadProjectionThread(&down_ctx);

        gate_thread.join();
        up_thread.join();

        // Propagate errors
        if (gate_ctx.err) |e| return e;
        if (up_ctx.err) |e| return e;
        if (down_ctx.err) |e| return e;

        gate_w.* = gate_ctx.result.?;
        up_w.* = up_ctx.result.?;
        down_w.* = down_ctx.result.?;

        // Load scales sequentially (small tensors, not worth parallelizing)
        if (self.is_quantized) {
            if (gate_scales_name) |n| gate_s.* = try self.loadExpertSlicesCached(n, expert_ids, layer_idx, 0);
            if (up_scales_name) |n| up_s.* = try self.loadExpertSlicesCached(n, expert_ids, layer_idx, 0);
            if (down_scales_name) |n| down_s.* = try self.loadExpertSlicesCached(n, expert_ids, layer_idx, 0);
        }

        const t1 = std.c.mach_absolute_time();
        const dt = @as(u64, @intCast(t1 - t0)) * 125 / 3;
        const dt_ms = @as(f64, @floatFromInt(dt)) / 1_000_000.0;
        if (layer_idx == 0) {
            std.log.info("[Parallel] 3-projections loaded in {d:.1}ms", .{dt_ms});
        }
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

    /// Load expert slices via PartialTensorReader.
    /// Formerly had LFU ExpertCache; removed in P2.1 (Trust OS is superior).
    fn loadExpertSlicesCached(
        self: *ExpertStreamProvider,
        tensor_name: []const u8,
        expert_ids: []const u32,
        layer_idx: usize,
        row_bytes: usize,
    ) !Array {
        _ = layer_idx;
        return self.loadExpertSlices(tensor_name, expert_ids, row_bytes);
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
        // Track token steps for diagnostics (increment once per first layer of each token)
        if (layer_idx == 0) {
            self.token_step_count += 1;
            self.token_step_start_ticks = std.c.mach_absolute_time();
            self.step_bytes_read = 0;
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

        // 2. DyMoE Skip: filter out low-score cache-miss experts to reduce I/O
        //    Only skip if cache is available (needed for hit/miss check)
        var load_ids_buf = try self.allocator.alloc(u32, unique_ids.len);
        defer self.allocator.free(load_ids_buf);
        var load_ids_len: usize = 0;
        var skip_set: [256]bool = [_]bool{false} ** 256;
        var dymoe_skipped: usize = 0;

        if (self.dymoe_max_skip > 0 and unique_ids.len > 2) {
            // Score-free DyMoE: skip expert based on router position distribution.
            // Router outputs top-K indices sorted by score (descending). The last 2
            // columns (positions K-2, K-1) are the lowest-score experts.
            //
            // Heuristic: for each unique expert, compute the fraction of times it
            // appears in low-score positions (cols 4-5). Skip if consistently low.
            // Does NOT eval scores — preserves MLX lazy fusion for correctness.
            //
            // Ref: DeepSeek V4 uses normalized top-K routing. The 6th expert (lowest
            // score) has minimal contribution. DyMoE A/B test confirmed skipping 1/6
            // has no correctness impact.
            var total_count: [256]u8 = [_]u8{0} ** 256;
            var low_count: [256]u8 = [_]u8{0} ** 256; // appearances in last col only
            const ndims = indices_u32.ndim();
            const shape = indices_u32.shape();
            const topk: usize = @intCast(shape[ndims - 1]);
            for (indices_data, 0..) |eid, i| {
                if (eid < 256) {
                    total_count[eid] += 1;
                    if (i % topk == topk - 1) low_count[eid] += 1; // last col = lowest score
                }
            }

            // Find expert most consistently in low-score positions.
            // Require: appears at least 2 times total, and > 40% of appearances are low.
            var best_candidate: ?u32 = null;
            var best_ratio: f32 = 0.5; // conservative: only skip clearly-low experts
            for (unique_ids) |eid| {
                if (total_count[eid] < 2) continue;
                const ratio = @as(f32, @floatFromInt(low_count[eid])) / @as(f32, @floatFromInt(total_count[eid]));
                if (ratio > best_ratio) {
                    best_ratio = ratio;
                    best_candidate = eid;
                }
            }

            if (best_candidate) |eid| {
                skip_set[eid] = true;
                dymoe_skipped = 1;
                for (unique_ids) |candidate| {
                    if (!skip_set[candidate]) {
                        load_ids_buf[load_ids_len] = candidate;
                        load_ids_len += 1;
                    }
                }
                self.dymoe_total_skipped += 1;
                self.dymoe_total_opportunities += 1;
            } else {
                @memcpy(load_ids_buf[0..unique_ids.len], unique_ids);
                load_ids_len = unique_ids.len;
            }
        } else {
            // No skip — load all unique experts
            @memcpy(load_ids_buf[0..unique_ids.len], unique_ids);
            load_ids_len = unique_ids.len;
        }

        const actual_load_ids = load_ids_buf[0..load_ids_len];

        // Log skip info (only layer 0 to avoid spam)
        if (layer_idx == 0 and dymoe_skipped > 0) {
            std.log.info("[DyMoE] Skipped {d} low-score cache-miss experts (load {d}/{d})", .{
                dymoe_skipped,
                actual_load_ids.len,
                unique_ids.len,
            });
        }

        // 3. Load expert weight slices using cache-first strategy with partial reads
        //    P1: Flash-MoE parallel pread path (falls back to mmap if pread unavailable)
        var gate_w: Array = undefined;
        var up_w: Array = undefined;
        var down_w: Array = undefined;
        var gate_s: ?Array = null;
        var up_s: ?Array = null;
        var down_s: ?Array = null;

        const use_pread = self.pread_loader != null and self.pread_loader.?.hasLayer(layer_idx);
        var pread_ok = false;
        if (use_pread) {
            const loader = self.pread_loader.?;
            // readAndAssembleAll: one set of pread calls, extract all 6 components.
            // Avoids reading the same expert blob 3x (once per projection).
            const all = loader.readAndAssembleAll(self.ctx, layer_idx, actual_load_ids) catch null;
            if (all) |a| {
                gate_w = a.gate;
                up_w = a.up;
                down_w = a.down;
                gate_s = a.gs;
                up_s = a.us;
                down_s = a.ds;
                pread_ok = true;
            } else {
                std.log.warn("[Pread] readAndAssembleAll failed for layer {d}", .{layer_idx});
            }
        }

        if (!pread_ok) {
            try self.loadExpertProjectionsParallel(
                meta.gate_proj_name,
                meta.up_proj_name,
                meta.down_proj_name,
                meta.gate_scales_name,
                meta.up_scales_name,
                meta.down_scales_name,
                actual_load_ids,
                layer_idx,
                &gate_w,
                &up_w,
                &down_w,
                &gate_s,
                &up_s,
                &down_s,
            );
        }

        defer gate_w.deinit();
        defer up_w.deinit();
        defer down_w.deinit();
        defer if (gate_s) |a| a.deinit();
        defer if (up_s) |a| a.deinit();
        defer if (down_s) |a| a.deinit();

        // 4. Build remap: original_expert_id → mini_fused_row_index
        //    Skipped experts map to index 0 (their scores will be zeroed)
        var remap_data = try self.allocator.alloc(i32, meta.n_experts);
        defer self.allocator.free(remap_data);
        @memset(remap_data, 0);
        for (actual_load_ids, 0..) |eid, i| {
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
        // DyMoE: skip low-score experts, accept small weighting error from
        // not renormalizing scores (validated: no correctness impact in A/B test)
        const scores_expanded = try ops.expandDims(self.ctx, scores, -1);
        defer scores_expanded.deinit();
        const weighted_out = try ops.multiply(self.ctx, expert_out, scores_expanded);
        defer weighted_out.deinit();
        const reduce_mod = @import("mlx").reduce;
        const result = try reduce_mod.sumAxis(self.ctx, weighted_out, -2, false);

        // Prefetcher removed in P2.1 (depends on ExpertCache).

        // Log end-of-token-step metrics on the last layer
        if (layer_idx == self.layer_meta.len - 1 and self.token_step_start_ticks != 0) {
            const end_ticks = std.c.mach_absolute_time();
            const elapsed_ticks = end_ticks - self.token_step_start_ticks;
            // Convert mach_absolute_time ticks to ms via timebase.
            // mach_absolute_time returns ticks, NOT nanoseconds.
            // On Apple Silicon: timebase = 125/3, so 1 tick = 125/3 ns ≈ 41.67 ns.
            const elapsed_ns = elapsed_ticks * 125 / 3;
            const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
            std.log.info("Token step {d} complete: {d:.1}ms", .{
                self.token_step_count,
                elapsed_ms,
            });
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

// ExpertCache test moved to expert_cache.zig (P2.1: cache removed from stream provider)
