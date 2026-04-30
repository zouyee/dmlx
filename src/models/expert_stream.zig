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
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const ops = @import("../ops.zig");
const safetensors_reader = @import("../io/safetensors_reader.zig");
const quantize_mod = @import("../quantize.zig");
const shape_mod = @import("../ops/shape.zig");
const expert_preload = @import("expert_preload.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const TensorIndex = safetensors_reader.TensorIndex;

/// Expert loading strategy.
pub const ExpertLoadStrategy = enum {
    preload,  // Option 1: Preload subset at init (matches Python vmlx)
    stream,   // Option 2: Stream on-demand from disk (experimental)
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

    pub fn deinit(self: *ExpertStreamProvider) void {
        if (self.preload_provider) |provider| {
            provider.deinit();
            self.allocator.destroy(provider);
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
                // Stream mode doesn't need initialization - loads on demand
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
        
        const info = self.index.entries.get(tensor_name) orelse return error.TensorNotFound;
        
        // Load the FULL tensor from disk (this is the key fix!)
        const full_tensor = try self.index.loadTensor(tensor_name);
        defer full_tensor.deinit();
        
        // If all experts are selected, return the full tensor
        const n_experts = @as(usize, @intCast(info.shape[0]));
        if (expert_ids.len >= n_experts) {
            return ops.copy(self.ctx, full_tensor);
        }
        
        // Slice to get only the selected experts: full_tensor[expert_ids, ...]
        // Convert expert_ids to an MLX array for indexing
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

    /// Streaming forward (Option 2): Load experts on-demand from disk.
    /// This is the experimental approach with lower memory but more complexity.
    fn streamingForward(
        self: *ExpertStreamProvider,
        layer_idx: usize,
        flat_x: Array,
        indices: Array,
        scores: Array,
    ) !Array {
        const meta = self.layer_meta[layer_idx];

        // 1. Collect unique expert IDs from indices
        const indices_data = try indices.dataSlice(u32);
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
        // Sort for sequential disk access (though we now load full tensor, sorting still helps cache)
        std.mem.sort(u32, unique_ids, {}, std.sort.asc(u32));

        // 2. Load only the needed expert weight slices from SSD
        // Note: row_bytes parameter removed - we now load full tensor then slice
        const gate_w = try self.loadExpertSlices(meta.gate_proj_name, unique_ids, 0);
        defer gate_w.deinit();
        const up_w = try self.loadExpertSlices(meta.up_proj_name, unique_ids, 0);
        defer up_w.deinit();
        const down_w = try self.loadExpertSlices(meta.down_proj_name, unique_ids, 0);
        defer down_w.deinit();

        // Load scales if quantized
        var gate_s: ?Array = null;
        var up_s: ?Array = null;
        var down_s: ?Array = null;
        defer if (gate_s) |a| a.deinit();
        defer if (up_s) |a| a.deinit();
        defer if (down_s) |a| a.deinit();
        if (self.is_quantized) {
            if (meta.gate_scales_name) |n| { gate_s = try self.loadExpertSlices(n, unique_ids, 0); }
            if (meta.up_scales_name) |n| { up_s = try self.loadExpertSlices(n, unique_ids, 0); }
            if (meta.down_scales_name) |n| { down_s = try self.loadExpertSlices(n, unique_ids, 0); }
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

        // Remap indices: [N, topk] original IDs → [N, topk] mini-fused row indices
        var remapped_raw = c.c.mlx_array_new();
        try c.check(c.c.mlx_take(&remapped_raw, remap_arr.inner, indices.inner, self.ctx.stream.inner));
        const remapped = Array.fromHandle(remapped_raw);
        defer remapped.deinit();
        const remapped_u32 = try ops.astype(self.ctx, remapped, .uint32);
        defer remapped_u32.deinit();

        // 4. Run gather_mm / gather_qmm with the mini fused weights
        const x_4d = try ops.expandDims(self.ctx, flat_x, -2);
        defer x_4d.deinit();
        const x_exp = try ops.expandDims(self.ctx, x_4d, -2);
        defer x_exp.deinit();

        const total_indices = indices.size();
        const route_shape = indices.shape();
        const topk = @as(usize, @intCast(route_shape[route_shape.len - 1]));

        // Flatten indices for gather dispatch
        const flat_indices = try ops.reshape(self.ctx, remapped_u32, &[_]i32{@intCast(total_indices)});
        defer flat_indices.deinit();
        const flat_scores = try ops.reshape(self.ctx, scores, &[_]i32{@intCast(total_indices)});
        defer flat_scores.deinit();

        // For quantized weights, DON'T transpose - gatherQmm expects [n, out, in] format
        // For float weights, transpose to [n, in, out]
        // This matches DSV4SwitchGLU.dispatchGatherMm behavior

        // Expand x for each token's topk experts
        const x_flat = try shape_mod.flatten(self.ctx, x_exp, 0, @as(i32, @intCast(x_exp.ndim())) - 3);
        defer x_flat.deinit();
        const topk_scalar = try ops.scalarI32(self.ctx, @intCast(topk));
        defer topk_scalar.deinit();

        // Token indices for gathering
        const idx_range = try ops.arange(self.ctx, 0, @floatFromInt(total_indices), 1, .int32);
        defer idx_range.deinit();
        const token_idx = try ops.divide(self.ctx, idx_range, topk_scalar);
        defer token_idx.deinit();
        const token_idx_i32 = try ops.astype(self.ctx, token_idx, .int32);
        defer token_idx_i32.deinit();
        const sx = try shape_mod.takeAxis(self.ctx, x_flat, token_idx_i32, 0);
        defer sx.deinit();

        // Dispatch matmul (float path — quantized path uses gatherQmm)
        if (self.is_quantized) {
            const qconfig = quantize_mod.QuantConfig{
                .group_size = self.quant_group_size,
                .bits = self.quant_bits,
                .mode = if (std.mem.eql(u8, self.quant_mode, "mxfp4")) .mxfp4 else .affine,
            };
            // gatherQmm: x, w (packed, NOT transposed), scales, biases, lhs_indices, rhs_indices, transpose, config, sorted
            // Use original weights (gate_w, up_w, down_w) without transpose
            const x_gate = try quantize_mod.gatherQmm(self.ctx, sx, gate_w, gate_s.?, null, null, flat_indices, true, qconfig, false);
            defer x_gate.deinit();
            const x_up = try quantize_mod.gatherQmm(self.ctx, sx, up_w, up_s.?, null, null, flat_indices, true, qconfig, false);
            defer x_up.deinit();

            // SwiGLU
            const silu_gate = try ops.multiply(self.ctx, x_gate, try ops.sigmoid(self.ctx, x_gate));
            defer silu_gate.deinit();
            const hidden = try ops.multiply(self.ctx, silu_gate, x_up);
            defer hidden.deinit();

            // Down projection
            var x_down = try quantize_mod.gatherQmm(self.ctx, hidden, down_w, down_s.?, null, null, flat_indices, true, qconfig, false);
            defer x_down.deinit();

            // Multiply by routing scores
            const scores_exp = try ops.expandDims(self.ctx, flat_scores, -1);
            defer scores_exp.deinit();
            const scores_exp2 = try ops.expandDims(self.ctx, scores_exp, -1);
            defer scores_exp2.deinit();
            const weighted = try ops.multiply(self.ctx, x_down, scores_exp2);
            defer weighted.deinit();

            // Reshape and sum over topk
            const n_tokens = total_indices / topk;
            const hidden_dim = @as(usize, @intCast(flat_x.shape()[1]));
            const reshaped = try ops.reshape(self.ctx, weighted, &[_]i32{ @intCast(n_tokens), @intCast(topk), 1, @intCast(hidden_dim) });
            defer reshaped.deinit();
            const squeezed = try shape_mod.squeezeAxes(self.ctx, reshaped, &[_]i32{2});
            defer squeezed.deinit();
            const reduce_mod = @import("../ops/reduce.zig");
            return reduce_mod.sumAxis(self.ctx, squeezed, 1, false);
        } else {
            // Float path with gatherMm - needs transposed weights
            const gate_t = try ops.transposeAxes(self.ctx, gate_w, &[_]i32{ 0, 2, 1 });
            defer gate_t.deinit();
            const up_t = try ops.transposeAxes(self.ctx, up_w, &[_]i32{ 0, 2, 1 });
            defer up_t.deinit();
            const down_t = try ops.transposeAxes(self.ctx, down_w, &[_]i32{ 0, 2, 1 });
            defer down_t.deinit();
            
            const x_gate = try ops.gatherMm(self.ctx, sx, gate_t, null, flat_indices, false);
            defer x_gate.deinit();
            const x_up = try ops.gatherMm(self.ctx, sx, up_t, null, flat_indices, false);
            defer x_up.deinit();

            const silu_gate = try ops.multiply(self.ctx, x_gate, try ops.sigmoid(self.ctx, x_gate));
            defer silu_gate.deinit();
            const hidden = try ops.multiply(self.ctx, silu_gate, x_up);
            defer hidden.deinit();

            const x_down = try ops.gatherMm(self.ctx, hidden, down_t, null, flat_indices, false);
            defer x_down.deinit();

            const scores_exp = try ops.expandDims(self.ctx, flat_scores, -1);
            defer scores_exp.deinit();
            const scores_exp2 = try ops.expandDims(self.ctx, scores_exp, -1);
            defer scores_exp2.deinit();
            const weighted = try ops.multiply(self.ctx, x_down, scores_exp2);
            defer weighted.deinit();

            const n_tokens = total_indices / topk;
            const hidden_dim = @as(usize, @intCast(flat_x.shape()[1]));
            const reshaped = try ops.reshape(self.ctx, weighted, &[_]i32{ @intCast(n_tokens), @intCast(topk), 1, @intCast(hidden_dim) });
            defer reshaped.deinit();
            const squeezed = try shape_mod.squeezeAxes(self.ctx, reshaped, &[_]i32{2});
            defer squeezed.deinit();
            const reduce_mod = @import("../ops/reduce.zig");
            return reduce_mod.sumAxis(self.ctx, squeezed, 1, false);
        }
    }
};
