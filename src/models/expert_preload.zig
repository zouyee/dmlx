/// Expert preloading strategy for smelt mode (Option 1).
///
/// This approach matches the Python vmlx implementation:
/// 1. At model load: Load full expert tensors, slice to get subset, keep in memory
/// 2. At inference: Use preloaded weights with remap and cache_bias routing
/// 3. Memory: Higher than streaming but stable and proven to work
///
/// Advantages:
/// - Proven approach (matches working Python implementation)
/// - Simpler than streaming (no disk I/O during inference)
/// - Predictable memory usage
///
/// Trade-offs:
/// - Higher memory usage than true streaming
/// - Still saves memory vs loading all experts (10-50% vs 100%)
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const safetensors_reader = @import("mlx").safetensors_reader;
const quantize_mod = @import("mlx").quantize;
const shape_mod = @import("mlx").shape;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const TensorIndex = safetensors_reader.TensorIndex;

/// Preloaded expert weights for a single layer.
pub const PreloadedLayerExperts = struct {
    // Fused expert weights: [n_loaded, intermediate, hidden] or [n_loaded, hidden, intermediate]
    gate_proj: Array,
    up_proj: Array,
    down_proj: Array,

    // Quantization scales/biases (optional)
    gate_scales: ?Array = null,
    up_scales: ?Array = null,
    down_scales: ?Array = null,
    gate_biases: ?Array = null,
    up_biases: ?Array = null,
    down_biases: ?Array = null,

    // Remap: global expert ID → local slot index
    // Shape: [n_total_experts], values in [0, n_loaded)
    remap: Array,

    // Cache bias for routing: loaded experts = 0.0, others = -1000.0
    // Shape: [n_total_experts]
    cache_bias: Array,

    n_loaded: usize,
    n_total: usize,

    pub fn deinit(self: *PreloadedLayerExperts) void {
        self.gate_proj.deinit();
        self.up_proj.deinit();
        self.down_proj.deinit();
        if (self.gate_scales) |s| s.deinit();
        if (self.up_scales) |s| s.deinit();
        if (self.down_scales) |s| s.deinit();
        if (self.gate_biases) |b| b.deinit();
        if (self.up_biases) |b| b.deinit();
        if (self.down_biases) |b| b.deinit();
        self.remap.deinit();
        self.cache_bias.deinit();
    }
};

/// Expert preloading provider - loads a subset of experts at initialization.
pub const ExpertPreloadProvider = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    index: *TensorIndex,

    // Per-layer preloaded experts
    layers: []PreloadedLayerExperts,

    // Quantization config
    is_quantized: bool,
    quant_group_size: i32,
    quant_bits: u8,
    quant_mode: []const u8,

    pub fn deinit(self: *ExpertPreloadProvider) void {
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
    }

    /// Initialize provider and preload expert subset for all layers.
    ///
    /// expert_ids: List of expert indices to load (e.g., [0,1,2,...,N-1] for first N experts)
    /// layer_meta: Metadata for each layer (tensor names, shapes, etc.)
    pub fn init(
        allocator: std.mem.Allocator,
        ctx: EagerContext,
        index: *TensorIndex,
        expert_ids: []const u32,
        layer_meta: []const LayerMeta,
        is_quantized: bool,
        quant_group_size: i32,
        quant_bits: u8,
        quant_mode: []const u8,
    ) !ExpertPreloadProvider {
        const n_layers = layer_meta.len;
        const layers = try allocator.alloc(PreloadedLayerExperts, n_layers);
        errdefer allocator.free(layers);

        std.log.info("Preloading {d} experts for {d} layers...", .{ expert_ids.len, n_layers });

        for (layer_meta, 0..) |meta, i| {
            std.log.info("  Layer {d}: loading experts {any}", .{ i, expert_ids });

            // Load and slice each projection
            const gate_proj = try loadAndSliceExpert(allocator, index, meta.gate_proj_name, expert_ids);
            errdefer gate_proj.deinit();

            const up_proj = try loadAndSliceExpert(allocator, index, meta.up_proj_name, expert_ids);
            errdefer up_proj.deinit();

            const down_proj = try loadAndSliceExpert(allocator, index, meta.down_proj_name, expert_ids);
            errdefer down_proj.deinit();

            // Load scales if quantized
            var gate_scales: ?Array = null;
            var up_scales: ?Array = null;
            var down_scales: ?Array = null;
            if (is_quantized) {
                if (meta.gate_scales_name) |name| {
                    gate_scales = try loadAndSliceExpert(allocator, index, name, expert_ids);
                }
                if (meta.up_scales_name) |name| {
                    up_scales = try loadAndSliceExpert(allocator, index, name, expert_ids);
                }
                if (meta.down_scales_name) |name| {
                    down_scales = try loadAndSliceExpert(allocator, index, name, expert_ids);
                }
            }

            // Build remap: global expert ID → local slot index
            const n_total = meta.n_experts;
            var remap_data = try allocator.alloc(i32, n_total);
            defer allocator.free(remap_data);
            @memset(remap_data, 0); // Default to 0 for unloaded experts
            for (expert_ids, 0..) |eid, slot| {
                remap_data[eid] = @intCast(slot);
            }
            const remap = try Array.fromData(allocator, i32, remap_data, &[_]i32{@intCast(n_total)});
            errdefer remap.deinit();

            // Build cache_bias: loaded = 0.0, unloaded = -1000.0
            var bias_data = try allocator.alloc(f32, n_total);
            defer allocator.free(bias_data);
            @memset(bias_data, -1000.0);
            for (expert_ids) |eid| {
                bias_data[eid] = 0.0;
            }
            const cache_bias = try Array.fromData(allocator, f32, bias_data, &[_]i32{@intCast(n_total)});
            errdefer cache_bias.deinit();

            layers[i] = PreloadedLayerExperts{
                .gate_proj = gate_proj,
                .up_proj = up_proj,
                .down_proj = down_proj,
                .gate_scales = gate_scales,
                .up_scales = up_scales,
                .down_scales = down_scales,
                .remap = remap,
                .cache_bias = cache_bias,
                .n_loaded = expert_ids.len,
                .n_total = n_total,
            };
        }

        std.log.info("Preloading complete: {d} experts × {d} layers", .{ expert_ids.len, n_layers });

        return ExpertPreloadProvider{
            .allocator = allocator,
            .ctx = ctx,
            .index = index,
            .layers = layers,
            .is_quantized = is_quantized,
            .quant_group_size = quant_group_size,
            .quant_bits = quant_bits,
            .quant_mode = quant_mode,
        };
    }

    /// Forward pass using preloaded experts.
    pub fn forward(
        self: *ExpertPreloadProvider,
        layer_idx: usize,
        flat_x: Array,
        indices: Array,
        scores: Array,
    ) !Array {
        const layer = &self.layers[layer_idx];

        // Remap indices: [N, topk] global IDs → [N, topk] local slot indices
        var remapped_raw = c.c.mlx_array_new();
        try c.check(c.c.mlx_take(&remapped_raw, layer.remap.inner, indices.inner, self.ctx.stream.inner));
        const remapped = Array.fromHandle(remapped_raw);
        defer remapped.deinit();
        const remapped_u32 = try ops.astype(self.ctx, remapped, .uint32);
        defer remapped_u32.deinit();

        // Prepare input for gather_mm
        const x_4d = try ops.expandDims(self.ctx, flat_x, -2);
        defer x_4d.deinit();
        const x_exp = try ops.expandDims(self.ctx, x_4d, -2);
        defer x_exp.deinit();

        const route_shape = indices.shape();
        const topk = @as(usize, @intCast(route_shape[route_shape.len - 1]));
        const total_indices = indices.size();

        // Flatten indices and scores
        const flat_indices = try ops.reshape(self.ctx, remapped_u32, &[_]i32{@intCast(total_indices)});
        defer flat_indices.deinit();
        const flat_scores = try ops.reshape(self.ctx, scores, &[_]i32{@intCast(total_indices)});
        defer flat_scores.deinit();

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

        // Dispatch matmul
        if (self.is_quantized) {
            return self.quantizedForward(sx, flat_indices, flat_scores, layer, topk, total_indices, flat_x);
        } else {
            return self.floatForward(sx, flat_indices, flat_scores, layer, topk, total_indices, flat_x);
        }
    }

    fn quantizedForward(
        self: *ExpertPreloadProvider,
        sx: Array,
        flat_indices: Array,
        flat_scores: Array,
        layer: *const PreloadedLayerExperts,
        topk: usize,
        total_indices: usize,
        flat_x: Array,
    ) !Array {
        const qconfig = quantize_mod.QuantConfig{
            .group_size = self.quant_group_size,
            .bits = self.quant_bits,
            .mode = if (std.mem.eql(u8, self.quant_mode, "mxfp4")) .mxfp4 else .affine,
        };

        // Gate and up projections
        const x_gate = try quantize_mod.gatherQmm(self.ctx, sx, layer.gate_proj, layer.gate_scales.?, layer.gate_biases, null, flat_indices, true, qconfig, false);
        defer x_gate.deinit();

        const x_up = try quantize_mod.gatherQmm(self.ctx, sx, layer.up_proj, layer.up_scales.?, layer.up_biases, null, flat_indices, true, qconfig, false);
        defer x_up.deinit();

        // SwiGLU activation
        const sigmoid_gate = try ops.sigmoid(self.ctx, x_gate);
        defer sigmoid_gate.deinit();
        const silu_gate = try ops.multiply(self.ctx, x_gate, sigmoid_gate);
        defer silu_gate.deinit();
        const hidden = try ops.multiply(self.ctx, silu_gate, x_up);
        defer hidden.deinit();

        // Down projection
        const x_down = try quantize_mod.gatherQmm(self.ctx, hidden, layer.down_proj, layer.down_scales.?, layer.down_biases, null, flat_indices, true, qconfig, false);
        defer x_down.deinit();

        // Apply routing scores
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
        const reduce_mod = @import("mlx").reduce;
        return reduce_mod.sumAxis(self.ctx, squeezed, 1, false);
    }

    fn floatForward(
        self: *ExpertPreloadProvider,
        sx: Array,
        flat_indices: Array,
        flat_scores: Array,
        layer: *const PreloadedLayerExperts,
        topk: usize,
        total_indices: usize,
        flat_x: Array,
    ) !Array {
        // Transpose weights for gatherMm
        const gate_t = try ops.transposeAxes(self.ctx, layer.gate_proj, &[_]i32{ 0, 2, 1 });
        defer gate_t.deinit();
        const up_t = try ops.transposeAxes(self.ctx, layer.up_proj, &[_]i32{ 0, 2, 1 });
        defer up_t.deinit();
        const down_t = try ops.transposeAxes(self.ctx, layer.down_proj, &[_]i32{ 0, 2, 1 });
        defer down_t.deinit();

        // Gate and up projections
        const x_gate = try ops.gatherMm(self.ctx, sx, gate_t, null, flat_indices, false);
        defer x_gate.deinit();
        const x_up = try ops.gatherMm(self.ctx, sx, up_t, null, flat_indices, false);
        defer x_up.deinit();

        // SwiGLU activation
        const sigmoid_gate = try ops.sigmoid(self.ctx, x_gate);
        defer sigmoid_gate.deinit();
        const silu_gate = try ops.multiply(self.ctx, x_gate, sigmoid_gate);
        defer silu_gate.deinit();
        const hidden = try ops.multiply(self.ctx, silu_gate, x_up);
        defer hidden.deinit();

        // Down projection
        const x_down = try ops.gatherMm(self.ctx, hidden, down_t, null, flat_indices, false);
        defer x_down.deinit();

        // Apply routing scores
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
        const reduce_mod = @import("mlx").reduce;
        return reduce_mod.sumAxis(self.ctx, squeezed, 1, false);
    }

    /// Get cache_bias for a specific layer (for router integration).
    pub fn getCacheBias(self: *ExpertPreloadProvider, layer_idx: usize) Array {
        return self.layers[layer_idx].cache_bias;
    }
};

/// Layer metadata for preloading.
pub const LayerMeta = struct {
    gate_proj_name: []const u8,
    up_proj_name: []const u8,
    down_proj_name: []const u8,
    gate_scales_name: ?[]const u8,
    up_scales_name: ?[]const u8,
    down_scales_name: ?[]const u8,
    n_experts: usize,
};

/// Load full tensor and slice to get expert subset.
/// This is the key operation that matches Python vmlx's _load_expert_subset.
///
/// IMPORTANT: We eval() the sliced result immediately to allow the full tensor
/// to be freed. Without eval(), MLX's lazy evaluation keeps the full tensor
/// alive as a dependency, causing OOM when loading multiple layers.
fn loadAndSliceExpert(
    allocator: std.mem.Allocator,
    index: *TensorIndex,
    tensor_name: []const u8,
    expert_ids: []const u32,
) !Array {
    // Load full tensor from disk
    const full_tensor = try index.loadTensor(tensor_name);
    defer full_tensor.deinit();

    const full_shape = full_tensor.shape();
    const n_experts = @as(usize, @intCast(full_shape[0]));

    // If all experts selected, return full tensor
    if (expert_ids.len >= n_experts) {
        const ctx = ops.EagerContext.init(allocator);
        const result = try ops.copy(ctx, full_tensor);
        try result.eval();
        return result;
    }

    // Slice to get subset: full_tensor[expert_ids, ...]
    const indices_arr = try Array.fromData(allocator, u32, expert_ids, &[_]i32{@intCast(expert_ids.len)});
    defer indices_arr.deinit();

    const ctx = ops.EagerContext.init(allocator);
    const indices_i32 = try ops.astype(ctx, indices_arr, .int32);
    defer indices_i32.deinit();

    const result = try shape_mod.takeAxis(ctx, full_tensor, indices_i32, 0);
    // Force evaluation so the full tensor can be freed immediately.
    // Without this, MLX keeps full_tensor alive as a lazy dependency,
    // causing peak memory to be N_layers × full_tensor_size → OOM.
    try result.eval();
    return result;
}
