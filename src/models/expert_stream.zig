/// Expert weight streaming for memory-constrained MoE inference.
///
/// Instead of loading all 256 experts into memory, this module loads only
/// the experts selected by the router on each forward pass, directly from
/// safetensors files on SSD via pread random access.
///
/// On a 48GB Mac running DeepSeek V4 Flash 4-bit (151GB on disk):
/// - Without streaming: OOM (needs ~138GB for expert weights alone)
/// - With streaming: ~10GB (attention + shared expert + 8 active experts per step)
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const ops = @import("../ops.zig");
const safetensors_reader = @import("../io/safetensors_reader.zig");
const quantize_mod = @import("../quantize.zig");
const shape_mod = @import("../ops/shape.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const TensorIndex = safetensors_reader.TensorIndex;

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

/// Streams expert weights from SSD on demand.
pub const ExpertStreamProvider = struct {
    allocator: std.mem.Allocator,
    index: *TensorIndex,
    layer_meta: []LayerExpertMeta,
    ctx: EagerContext,
    is_quantized: bool,
    quant_group_size: i32,
    quant_bits: u8,
    quant_mode: []const u8,
    swiglu_limit: f32,

    pub fn deinit(self: *ExpertStreamProvider) void {
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

    /// Load a subset of experts from a fused tensor on disk.
    /// Returns a mini fused tensor [n_selected, ...] containing only the requested expert rows.
    /// `expert_ids` is a sorted, deduplicated list of expert indices to load.
    fn loadExpertSlices(
        self: *ExpertStreamProvider,
        tensor_name: []const u8,
        expert_ids: []const u32,
        row_bytes: usize,
    ) !Array {
        const info = self.index.entries.get(tensor_name) orelse return error.TensorNotFound;

        const posix = @cImport(@cInclude("fcntl.h"));
        const unistd = @cImport(@cInclude("unistd.h"));

        const path_z = try self.allocator.dupeZ(u8, info.shard_path);
        defer self.allocator.free(path_z);
        const fd = posix.open(path_z.ptr, posix.O_RDONLY);
        if (fd < 0) return error.FileNotFound;
        defer _ = unistd.close(fd);

        const n_selected = expert_ids.len;
        const total_bytes = n_selected * row_bytes;
        const buf = try self.allocator.alloc(u8, total_bytes);
        defer self.allocator.free(buf); // Safe to free after MLX copies the data

        // Read each expert's row via pread
        for (expert_ids, 0..) |eid, i| {
            const offset: i64 = @intCast(info.data_offset_start + @as(u64, eid) * @as(u64, row_bytes));
            const dest = buf[i * row_bytes .. (i + 1) * row_bytes];
            const r = unistd.pread(fd, dest.ptr, row_bytes, offset);
            if (r < @as(isize, @intCast(row_bytes))) {
                std.log.err("Failed to read expert {d} from {s}: read {d} bytes, expected {d}", .{eid, tensor_name, r, row_bytes});
                return error.IncompleteRead;
            }
        }

        // Build shape: [n_selected, rest_of_dims...]
        var shape_i32 = try self.allocator.alloc(i32, info.shape.len);
        defer self.allocator.free(shape_i32);
        shape_i32[0] = @intCast(n_selected);
        for (info.shape[1..], 1..) |s, idx| {
            shape_i32[idx] = @intCast(s);
        }

        const mlx_dtype = safetensors_reader.dtypeFromString(info.dtype_str) orelse return error.UnsupportedDtype;
        
        // Create array from data - MLX will copy the data internally
        const arr_raw = c.c.mlx_array_new_data(buf.ptr, shape_i32.ptr, @intCast(shape_i32.len), mlx_dtype);
        const arr = Array.fromHandle(arr_raw);
        
        // Force evaluation to ensure data is copied before we free buf
        try arr.eval();
        
        return arr;
    }

    /// Load the selected experts for a given layer and run the SwiGLU forward pass.
    /// `flat_x`: [N, hidden_size] input tokens
    /// `indices`: [N, topk] expert IDs from router
    /// `scores`: [N, topk] routing weights
    /// Returns: [N, hidden_size] expert output
    pub fn streamForward(
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
        // Sort for sequential disk access
        std.mem.sort(u32, unique_ids, {}, std.sort.asc(u32));

        // 2. Load only the needed expert weight slices from SSD
        const gate_w = try self.loadExpertSlices(meta.gate_proj_name, unique_ids, meta.expert_row_bytes);
        defer gate_w.deinit();
        const up_w = try self.loadExpertSlices(meta.up_proj_name, unique_ids, meta.expert_row_bytes);
        defer up_w.deinit();
        const down_w = try self.loadExpertSlices(meta.down_proj_name, unique_ids, meta.expert_row_bytes);
        defer down_w.deinit();

        // Load scales if quantized
        var gate_s: ?Array = null;
        var up_s: ?Array = null;
        var down_s: ?Array = null;
        defer if (gate_s) |a| a.deinit();
        defer if (up_s) |a| a.deinit();
        defer if (down_s) |a| a.deinit();
        if (self.is_quantized) {
            if (meta.gate_scales_name) |n| { gate_s = try self.loadExpertSlices(n, unique_ids, meta.expert_scale_row_bytes); }
            if (meta.up_scales_name) |n| { up_s = try self.loadExpertSlices(n, unique_ids, meta.expert_scale_row_bytes); }
            if (meta.down_scales_name) |n| { down_s = try self.loadExpertSlices(n, unique_ids, meta.expert_scale_row_bytes); }
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

        // Transpose weights for matmul: [n, out, in] → [n, in, out]
        const gate_t = try ops.transposeAxes(self.ctx, gate_w, &[_]i32{ 0, 2, 1 });
        defer gate_t.deinit();
        const up_t = try ops.transposeAxes(self.ctx, up_w, &[_]i32{ 0, 2, 1 });
        defer up_t.deinit();
        const down_t = try ops.transposeAxes(self.ctx, down_w, &[_]i32{ 0, 2, 1 });
        defer down_t.deinit();

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
            // gatherQmm: x, w (packed), scales, biases, lhs_indices, rhs_indices, transpose, config, sorted
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
            // Float path with gatherMm
            const x_gate = try ops.gatherMm(self.ctx, sx, gate_t, null, flat_indices, false);
            defer x_gate.deinit();
            const x_up = try ops.gatherMm(self.ctx, sx, up_t, null, flat_indices, false);
            defer x_up.deinit();

            const silu_gate = try ops.multiply(self.ctx, x_gate, try ops.sigmoid(self.ctx, x_gate));
            defer silu_gate.deinit();
            const hidden = try ops.multiply(self.ctx, silu_gate, x_up);
            defer hidden.deinit();

            var x_down = try ops.gatherMm(self.ctx, hidden, down_t, null, flat_indices, false);
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
