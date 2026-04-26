/// Neural network layers for MLX-like operations
/// Provides Linear, BatchNorm, Dropout, LSTM, GRU layers
const std = @import("std");
const array_mod = @import("../array.zig");
const dtype_mod = @import("../dtype.zig");
const ops_mod = @import("../ops.zig");
const fast_mod = @import("fast.zig");
const shape_mod = @import("shape.zig");
const creation_mod = @import("creation.zig");

pub const Array = array_mod.Array;
pub const Dtype = dtype_mod.Dtype;
const ShapeElem = array_mod.ShapeElem;
const EagerContext = ops_mod.EagerContext;

// ============================================================
// Linear Layer
// ============================================================

pub const Linear = struct {
    ctx: EagerContext,
    weight: Array, // (output_dims, input_dims)
    bias: ?Array, // (output_dims,)
    input_dims: usize,
    output_dims: usize,

    pub fn init(ctx: EagerContext, input_dims: usize, output_dims: usize, bias: bool) !Linear {
        // Initialize weights with Kaiming-style initialization
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(input_dims)));

        const weight_shape = [_]ShapeElem{ @intCast(output_dims), @intCast(input_dims) };
        const weight = try ops_mod.random(ctx, &weight_shape, dtype_mod.float32, @truncate(@as(u64, @bitCast(@as(f32, @floatFromInt(std.mem.readInt(u32, &.{ 0x3dcccccd, 0x3dcccccd }, .little)))))));

        const weight_data = try weight.dataSliceMut(f32);
        for (0..weight_data.len) |i| {
            weight_data[i] = (std.crypto.random.float(f32) * 2 - 1) * scale;
        }

        var bias_arr: ?Array = null;
        if (bias) {
            const bias_shape = [_]ShapeElem{@intCast(output_dims)};
            bias_arr = try ops_mod.zeros(ctx, &bias_shape, dtype_mod.float32);
        }

        return Linear{
            .ctx = ctx,
            .weight = weight,
            .bias = bias_arr,
            .input_dims = input_dims,
            .output_dims = output_dims,
        };
    }

    pub fn forward(self: *Linear, input: Array) !Array {
        const weight_t = try ops_mod.transpose(self.ctx, self.weight);
        defer weight_t.deinit();
        const result = try ops_mod.matmul(self.ctx, input, weight_t);
        if (self.bias) |b| {
            return ops_mod.add(self.ctx, result, b);
        }
        return result;
    }
};

// ============================================================
// BatchNorm (1D over batch dimension)
// ============================================================

pub const BatchNorm = struct {
    ctx: EagerContext,
    num_features: usize,
    momentum: f32,
    gamma: Array, // scale
    beta: Array, // bias
    running_mean: Array,
    running_var: Array,
    training: bool,

    pub fn init(ctx: EagerContext, num_features: usize, momentum: f32) !BatchNorm {
        const shape = [_]ShapeElem{@intCast(num_features)};

        const gamma = try array_mod.ones(ctx.allocator, &shape, dtype_mod.float32);
        const beta = try ops_mod.zeros(ctx, &shape, dtype_mod.float32);
        const running_mean = try ops_mod.zeros(ctx, &shape, dtype_mod.float32);
        const running_var = try array_mod.ones(ctx.allocator, &shape, dtype_mod.float32);

        return BatchNorm{
            .ctx = ctx,
            .num_features = num_features,
            .momentum = momentum,
            .gamma = gamma,
            .beta = beta,
            .running_mean = running_mean,
            .running_var = running_var,
            .training = true,
        };
    }

    pub fn forward(self: *BatchNorm, input: Array) !Array {
        // input: (batch, num_features) or (batch, num_features, ...)
        // Normalize over num_features dimension
        const shape = input.shape();
        std.debug.assert(shape.len >= 2);

        const batch = @as(usize, @intCast(shape[0]));
        const num_features = @as(usize, @intCast(shape[1]));

        const out_shape = shape;
        const out = try array_mod.zeros(self.ctx.allocator, out_shape, input.dtype());
        const src = try input.dataSliceMut(f32);
        const dst = try out.dataSliceMut(f32);
        const gamma_data = try self.gamma.dataSliceMut(f32);
        const beta_data = try self.beta.dataSliceMut(f32);

        const eps: f32 = 1e-5;

        if (self.training) {
            // Compute batch mean
            var mean_buf = try self.ctx.allocator.alloc(f32, num_features);
            defer self.ctx.allocator.free(mean_buf);
            @memset(&mean_buf, 0);

            for (0..batch) |n| {
                for (0..num_features) |f| {
                    mean_buf[f] += src[n * num_features + f];
                }
            }
            for (0..num_features) |f| {
                mean_buf[f] /= @as(f32, @floatFromInt(batch));
            }

            // Compute batch variance
            var var_buf = try self.ctx.allocator.alloc(f32, num_features);
            defer self.ctx.allocator.free(var_buf);

            for (0..batch) |n| {
                for (0..num_features) |f| {
                    const diff = src[n * num_features + f] - mean_buf[f];
                    var_buf[f] += diff * diff;
                }
            }
            for (0..num_features) |f| {
                var_buf[f] /= @as(f32, @floatFromInt(batch));
            }

            // Normalize
            for (0..batch) |n| {
                for (0..num_features) |f| {
                    const idx = n * num_features + f;
                    const normalized = (src[idx] - mean_buf[f]) / @sqrt(var_buf[f] + eps);
                    dst[idx] = gamma_data[f] * normalized + beta_data[f];
                }
            }

            // Update running stats
            const running_mean_data = try self.running_mean.dataSliceMut(f32);
            const running_var_data = try self.running_var.dataSliceMut(f32);
            for (0..num_features) |f| {
                running_mean_data[f] = self.momentum * mean_buf[f] + (1 - self.momentum) * running_mean_data[f];
                running_var_data[f] = self.momentum * var_buf[f] + (1 - self.momentum) * running_var_data[f];
            }
        } else {
            // Use running statistics
            const mean_data = try self.running_mean.dataSliceMut(f32);
            const var_data = try self.running_var.dataSliceMut(f32);

            for (0..batch) |n| {
                for (0..num_features) |f| {
                    const idx = n * num_features + f;
                    const normalized = (src[idx] - mean_data[f]) / @sqrt(var_data[f] + eps);
                    dst[idx] = gamma_data[f] * normalized + beta_data[f];
                }
            }
        }

        return out;
    }

    pub fn setTraining(self: *BatchNorm, training: bool) void {
        self.training = training;
    }
};

// ============================================================
// Dropout
// ============================================================

pub const Dropout = struct {
    ctx: EagerContext,
    p: f32, // probability of dropping
    training: bool,

    pub fn init(ctx: EagerContext, p: f32) Dropout {
        return Dropout{
            .ctx = ctx,
            .p = p,
            .training = true,
        };
    }

    pub fn forward(self: *Dropout, input: Array) !Array {
        if (!self.training) {
            return ops_mod.copy(self.ctx, input);
        }

        const out = try array_mod.zeros(self.ctx.allocator, input.shape(), input.dtype());
        const src = try input.dataSliceMut(f32);
        const dst = try out.dataSliceMut(f32);
        const size = input.size();
        const scale: f32 = 1.0 / (1.0 - self.p);

        for (0..size) |i| {
            dst[i] = if (std.crypto.random.float(f32) < self.p) 0 else src[i] * scale;
        }

        return out;
    }

    pub fn setTraining(self: *Dropout, training: bool) void {
        self.training = training;
    }
};

// ============================================================
// LSTM (Long Short-Term Memory)
// ============================================================

pub const LSTM = struct {
    ctx: EagerContext,
    input_size: usize,
    hidden_size: usize,
    weight_ih: Array, // (4 * hidden_size, input_size)
    weight_hh: Array, // (4 * hidden_size, hidden_size)
    bias_ih: Array, // (4 * hidden_size,)
    bias_hh: Array, // (4 * hidden_size,)

    pub fn init(ctx: EagerContext, input_size: usize, hidden_size: usize) !LSTM {
        const scale_ih = @sqrt(2.0 / @as(f32, @floatFromInt(input_size)));
        const scale_hh = @sqrt(2.0 / @as(f32, @floatFromInt(hidden_size)));

        const w_ih_shape = [_]ShapeElem{ @intCast(4 * hidden_size), @intCast(input_size) };
        const w_hh_shape = [_]ShapeElem{ @intCast(4 * hidden_size), @intCast(hidden_size) };
        const b_shape = [_]ShapeElem{@intCast(4 * hidden_size)};

        const weight_ih = try ops_mod.random(ctx, &w_ih_shape, dtype_mod.float32, @truncate(@as(u64, @bitCast(@as(f32, 0.0)))));
        const weight_hh = try ops_mod.random(ctx, &w_hh_shape, dtype_mod.float32, @truncate(@as(u64, @bitCast(@as(f32, 0.0)))));
        const bias_ih = try ops_mod.zeros(ctx, &b_shape, dtype_mod.float32);
        const bias_hh = try ops_mod.zeros(ctx, &b_shape, dtype_mod.float32);

        // Initialize with random values
        const w_ih_data = try weight_ih.dataSliceMut(f32);
        const w_hh_data = try weight_hh.dataSliceMut(f32);
        for (0..w_ih_data.len) |i| {
            w_ih_data[i] = (std.crypto.random.float(f32) * 2 - 1) * scale_ih;
        }
        for (0..w_hh_data.len) |i| {
            w_hh_data[i] = (std.crypto.random.float(f32) * 2 - 1) * scale_hh;
        }

        return LSTM{
            .ctx = ctx,
            .input_size = input_size,
            .hidden_size = hidden_size,
            .weight_ih = weight_ih,
            .weight_hh = weight_hh,
            .bias_ih = bias_ih,
            .bias_hh = bias_hh,
        };
    }

    pub fn forward(self: *LSTM, input: Array, h0: ?Array, c0: ?Array) !LSTMOutput {
        // input: (batch, seq_len, input_size)
        // h0: (num_layers, batch, hidden_size) optional
        // c0: (num_layers, batch, hidden_size) optional
        // Returns: (batch, seq_len, hidden_size), (batch, hidden_size), (batch, hidden_size)

        const shape = input.shape();
        std.debug.assert(shape.len == 3);
        const batch: i32 = shape[0];
        const seq_len = @as(usize, @intCast(shape[1]));
        const hs: i32 = @intCast(self.hidden_size);

        // Initialize hidden/cell state: (batch, hidden_size)
        var prev_h = if (h0) |h|
            try ops_mod.reshape(self.ctx, h, &[_]ShapeElem{ batch, hs })
        else
            try creation_mod.zeros(self.ctx, &[_]ShapeElem{ batch, hs }, dtype_mod.float32);

        var prev_c = if (c0) |cc|
            try ops_mod.reshape(self.ctx, cc, &[_]ShapeElem{ batch, hs })
        else
            try creation_mod.zeros(self.ctx, &[_]ShapeElem{ batch, hs }, dtype_mod.float32);

        // Transpose weights for matmul: (input_size, 4*hidden) and (hidden, 4*hidden)
        const w_ih_t = try ops_mod.transpose(self.ctx, self.weight_ih);
        defer w_ih_t.deinit();
        const w_hh_t = try ops_mod.transpose(self.ctx, self.weight_hh);
        defer w_hh_t.deinit();

        // Collect per-timestep outputs
        var outputs = try self.ctx.allocator.alloc(Array, seq_len);
        defer self.ctx.allocator.free(outputs);

        for (0..seq_len) |t| {
            const t_i: i32 = @intCast(t);
            // Slice input at timestep t: (batch, input_size)
            const x_t = try shape_mod.slice(self.ctx, input, &[_]i32{ 0, t_i, 0 }, &[_]i32{ batch, t_i + 1, @intCast(self.input_size) }, &[_]i32{ 1, 1, 1 });
            defer x_t.deinit();
            const x_t_2d = try ops_mod.reshape(self.ctx, x_t, &[_]ShapeElem{ batch, @intCast(self.input_size) });
            defer x_t_2d.deinit();

            // gates = x_t @ W_ih^T + h @ W_hh^T + b_ih + b_hh  -> (batch, 4*hidden)
            const x_w = try ops_mod.matmul(self.ctx, x_t_2d, w_ih_t);
            defer x_w.deinit();
            const h_w = try ops_mod.matmul(self.ctx, prev_h, w_hh_t);
            defer h_w.deinit();
            const gates_1 = try ops_mod.add(self.ctx, x_w, h_w);
            defer gates_1.deinit();
            const gates_2 = try ops_mod.add(self.ctx, gates_1, self.bias_ih);
            defer gates_2.deinit();
            const gates = try ops_mod.add(self.ctx, gates_2, self.bias_hh);
            defer gates.deinit();

            // Split gates into i, f, o, g along axis 1
            const i_gate = try shape_mod.slice(self.ctx, gates, &[_]i32{ 0, 0 }, &[_]i32{ batch, hs }, &[_]i32{ 1, 1 });
            defer i_gate.deinit();
            const f_gate = try shape_mod.slice(self.ctx, gates, &[_]i32{ 0, hs }, &[_]i32{ batch, 2 * hs }, &[_]i32{ 1, 1 });
            defer f_gate.deinit();
            const o_gate = try shape_mod.slice(self.ctx, gates, &[_]i32{ 0, 2 * hs }, &[_]i32{ batch, 3 * hs }, &[_]i32{ 1, 1 });
            defer o_gate.deinit();
            const g_gate = try shape_mod.slice(self.ctx, gates, &[_]i32{ 0, 3 * hs }, &[_]i32{ batch, 4 * hs }, &[_]i32{ 1, 1 });
            defer g_gate.deinit();

            // Apply activations via GPU ops
            const i_act = try ops_mod.sigmoid(self.ctx, i_gate);
            defer i_act.deinit();
            const f_act = try ops_mod.sigmoid(self.ctx, f_gate);
            defer f_act.deinit();
            const o_act = try ops_mod.sigmoid(self.ctx, o_gate);
            defer o_act.deinit();
            const g_act = try ops_mod.tanh(self.ctx, g_gate);
            defer g_act.deinit();

            // c_t = f * c_{t-1} + i * g
            const fc = try ops_mod.multiply(self.ctx, f_act, prev_c);
            defer fc.deinit();
            const ig = try ops_mod.multiply(self.ctx, i_act, g_act);
            defer ig.deinit();
            const new_c = try ops_mod.add(self.ctx, fc, ig);

            // h_t = o * tanh(c_t)
            const tanh_c = try ops_mod.tanh(self.ctx, new_c);
            defer tanh_c.deinit();
            const new_h = try ops_mod.multiply(self.ctx, o_act, tanh_c);

            // Free previous states (unless they were the initial ones on first iter)
            if (t > 0 or h0 == null) prev_h.deinit();
            if (t > 0 or c0 == null) prev_c.deinit();
            prev_h = new_h;
            prev_c = new_c;

            // Store output: expand to (batch, 1, hidden) for later concat
            outputs[t] = try ops_mod.expandDims(self.ctx, new_h, 1);
        }

        // Concatenate all timestep outputs: (batch, seq_len, hidden_size)
        const output = try shape_mod.concatenateAxis(self.ctx, outputs, 1);
        for (outputs) |o| o.deinit();

        // Final hidden/cell: (batch, hidden_size)
        const final_h = try ops_mod.copy(self.ctx, prev_h);
        const final_c = try ops_mod.copy(self.ctx, prev_c);
        prev_h.deinit();
        prev_c.deinit();

        return LSTMOutput{
            .output = output,
            .hidden = final_h,
            .cell = final_c,
        };
    }
};

pub const LSTMOutput = struct {
    output: Array, // (batch, seq_len, hidden_size)
    hidden: Array, // (batch, hidden_size)
    cell: Array, // (batch, hidden_size)
};

// ============================================================
// GRU (Gated Recurrent Unit)
// ============================================================

pub const GRU = struct {
    ctx: EagerContext,
    input_size: usize,
    hidden_size: usize,
    weight_ih: Array, // (3 * hidden_size, input_size)
    weight_hh: Array, // (3 * hidden_size, hidden_size)
    bias_ih: Array, // (3 * hidden_size,)
    bias_hh: Array, // (3 * hidden_size,)

    pub fn init(ctx: EagerContext, input_size: usize, hidden_size: usize) !GRU {
        const scale_ih = @sqrt(2.0 / @as(f32, @floatFromInt(input_size)));
        const scale_hh = @sqrt(2.0 / @as(f32, @floatFromInt(hidden_size)));

        const w_ih_shape = [_]ShapeElem{ @intCast(3 * hidden_size), @intCast(input_size) };
        const w_hh_shape = [_]ShapeElem{ @intCast(3 * hidden_size), @intCast(hidden_size) };
        const b_shape = [_]ShapeElem{@intCast(3 * hidden_size)};

        const weight_ih = try ops_mod.random(ctx, &w_ih_shape, dtype_mod.float32, 0);
        const weight_hh = try ops_mod.random(ctx, &w_hh_shape, dtype_mod.float32, 0);
        const bias_ih = try ops_mod.zeros(ctx, &b_shape, dtype_mod.float32);
        const bias_hh = try ops_mod.zeros(ctx, &b_shape, dtype_mod.float32);

        const w_ih_data = try weight_ih.dataSliceMut(f32);
        const w_hh_data = try weight_hh.dataSliceMut(f32);
        for (0..w_ih_data.len) |i| {
            w_ih_data[i] = (std.crypto.random.float(f32) * 2 - 1) * scale_ih;
        }
        for (0..w_hh_data.len) |i| {
            w_hh_data[i] = (std.crypto.random.float(f32) * 2 - 1) * scale_hh;
        }

        return GRU{
            .ctx = ctx,
            .input_size = input_size,
            .hidden_size = hidden_size,
            .weight_ih = weight_ih,
            .weight_hh = weight_hh,
            .bias_ih = bias_ih,
            .bias_hh = bias_hh,
        };
    }

    pub fn forward(self: *GRU, input: Array, h0: ?Array) !GRUOutput {
        // input: (batch, seq_len, input_size)
        // h0: (batch, hidden_size) optional
        // Returns: (batch, seq_len, hidden_size), (batch, hidden_size)

        const shape = input.shape();
        std.debug.assert(shape.len == 3);
        const batch: i32 = shape[0];
        const seq_len = @as(usize, @intCast(shape[1]));
        const hs: i32 = @intCast(self.hidden_size);

        var prev_h = h0 orelse try creation_mod.zeros(self.ctx, &[_]ShapeElem{ batch, hs }, dtype_mod.float32);

        // Transpose weights for matmul
        const w_ih_t = try ops_mod.transpose(self.ctx, self.weight_ih);
        defer w_ih_t.deinit();
        const w_hh_t = try ops_mod.transpose(self.ctx, self.weight_hh);
        defer w_hh_t.deinit();

        var outputs = try self.ctx.allocator.alloc(Array, seq_len);
        defer self.ctx.allocator.free(outputs);

        for (0..seq_len) |t| {
            const t_i: i32 = @intCast(t);
            // Slice input at timestep t: (batch, input_size)
            const x_t = try shape_mod.slice(self.ctx, input, &[_]i32{ 0, t_i, 0 }, &[_]i32{ batch, t_i + 1, @intCast(self.input_size) }, &[_]i32{ 1, 1, 1 });
            defer x_t.deinit();
            const x_t_2d = try ops_mod.reshape(self.ctx, x_t, &[_]ShapeElem{ batch, @intCast(self.input_size) });
            defer x_t_2d.deinit();

            // gates = x_t @ W_ih^T + h @ W_hh^T + b_ih + b_hh  -> (batch, 3*hidden)
            const x_w = try ops_mod.matmul(self.ctx, x_t_2d, w_ih_t);
            defer x_w.deinit();
            const h_w = try ops_mod.matmul(self.ctx, prev_h, w_hh_t);
            defer h_w.deinit();
            const gates_1 = try ops_mod.add(self.ctx, x_w, h_w);
            defer gates_1.deinit();
            const gates_2 = try ops_mod.add(self.ctx, gates_1, self.bias_ih);
            defer gates_2.deinit();
            const gates = try ops_mod.add(self.ctx, gates_2, self.bias_hh);
            defer gates.deinit();

            // Split into z (update), r (reset), n (new) along axis 1
            const z_raw = try shape_mod.slice(self.ctx, gates, &[_]i32{ 0, 0 }, &[_]i32{ batch, hs }, &[_]i32{ 1, 1 });
            defer z_raw.deinit();
            const r_raw = try shape_mod.slice(self.ctx, gates, &[_]i32{ 0, hs }, &[_]i32{ batch, 2 * hs }, &[_]i32{ 1, 1 });
            defer r_raw.deinit();
            const n_raw = try shape_mod.slice(self.ctx, gates, &[_]i32{ 0, 2 * hs }, &[_]i32{ batch, 3 * hs }, &[_]i32{ 1, 1 });
            defer n_raw.deinit();

            // Apply activations via GPU ops
            const z_gate = try ops_mod.sigmoid(self.ctx, z_raw);
            defer z_gate.deinit();
            const r_gate = try ops_mod.sigmoid(self.ctx, r_raw);
            defer r_gate.deinit();
            // Note: reset gate is part of the combined gate computation above
            const n_gate = try ops_mod.tanh(self.ctx, n_raw);
            defer n_gate.deinit();

            // h_t = (1 - z) * h_{t-1} + z * n
            const one = try ops_mod.scalarF32(self.ctx, 1.0);
            defer one.deinit();
            const one_minus_z = try ops_mod.subtract(self.ctx, one, z_gate);
            defer one_minus_z.deinit();
            const keep = try ops_mod.multiply(self.ctx, one_minus_z, prev_h);
            defer keep.deinit();
            const update = try ops_mod.multiply(self.ctx, z_gate, n_gate);
            defer update.deinit();
            const new_h = try ops_mod.add(self.ctx, keep, update);

            if (t > 0 or h0 == null) prev_h.deinit();
            prev_h = new_h;

            outputs[t] = try ops_mod.expandDims(self.ctx, new_h, 1);
        }

        const output = try shape_mod.concatenateAxis(self.ctx, outputs, 1);
        for (outputs) |o| o.deinit();

        const final_h = try ops_mod.copy(self.ctx, prev_h);
        prev_h.deinit();

        return GRUOutput{
            .output = output,
            .hidden = final_h,
        };
    }
};

pub const GRUOutput = struct {
    output: Array, // (batch, seq_len, hidden_size)
    hidden: Array, // (batch, hidden_size)
};

// ============================================================
// GroupNorm (Group Normalization)
// ============================================================

pub const GroupNorm = struct {
    ctx: EagerContext,
    num_groups: usize,
    num_channels: usize,
    eps: f32,
    gamma: Array, // scale
    beta: Array, // bias

    pub fn init(ctx: EagerContext, num_groups: usize, num_channels: usize, eps: f32) !GroupNorm {
        // num_channels must be divisible by num_groups
        std.debug.assert(num_channels % num_groups == 0);

        const shape = [_]ShapeElem{@intCast(num_channels)};
        const gamma = try array_mod.ones(ctx.allocator, &shape, dtype_mod.float32);
        const beta = try ops_mod.zeros(ctx, &shape, dtype_mod.float32);

        return GroupNorm{
            .ctx = ctx,
            .num_groups = num_groups,
            .num_channels = num_channels,
            .eps = eps,
            .gamma = gamma,
            .beta = beta,
        };
    }

    pub fn forward(self: *GroupNorm, input: Array) !Array {
        // input: (batch, num_channels, ...)
        const shape = input.shape();
        std.debug.assert(shape.len >= 2);
        std.debug.assert(shape[1] == @as(ShapeElem, @intCast(self.num_channels)));

        const batch = @as(usize, @intCast(shape[0]));
        const channels = self.num_channels;
        const num_groups = self.num_groups;
        const channels_per_group = channels / num_groups;

        // Compute remaining dimensions
        var spatial_size: usize = 1;
        for (2..shape.len) |i| {
            spatial_size *= @as(usize, @intCast(shape[i]));
        }

        const out_shape = shape;
        const out = try array_mod.zeros(self.ctx.allocator, out_shape, input.dtype());
        const src = try input.dataSliceMut(f32);
        const dst = try out.dataSliceMut(f32);
        const gamma_data = try self.gamma.dataSliceMut(f32);
        const beta_data = try self.beta.dataSliceMut(f32);

        // Process each sample
        for (0..batch) |b| {
            // Process each group
            for (0..num_groups) |g| {
                // Compute mean for this group
                var sum: f32 = 0;
                const group_offset = b * channels * spatial_size + g * channels_per_group * spatial_size;

                for (0..channels_per_group) |c| {
                    const channel_offset = group_offset + c * spatial_size;
                    for (0..spatial_size) |s| {
                        sum += src[channel_offset + s];
                    }
                }

                const group_size = channels_per_group * spatial_size;
                const mean = sum / @as(f32, @floatFromInt(group_size));

                // Compute variance for this group
                var var_sum: f32 = 0;
                for (0..channels_per_group) |c| {
                    const channel_offset = group_offset + c * spatial_size;
                    for (0..spatial_size) |s| {
                        const diff = src[channel_offset + s] - mean;
                        var_sum += diff * diff;
                    }
                }
                const variance = var_sum / @as(f32, @floatFromInt(group_size));

                // Normalize and scale
                const inv_std = 1.0 / @sqrt(variance + self.eps);
                for (0..channels_per_group) |c| {
                    const channel_offset = group_offset + c * spatial_size;
                    for (0..spatial_size) |s| {
                        const idx = channel_offset + s;
                        const normalized = (src[idx] - mean) * inv_std;
                        dst[idx] = normalized * gamma_data[g * channels_per_group + c] + beta_data[g * channels_per_group + c];
                    }
                }
            }
        }

        return out;
    }
};

// ============================================================
// InstanceNorm
// ============================================================

pub const InstanceNorm = struct {
    ctx: EagerContext,
    num_features: usize,
    eps: f32,
    gamma: Array, // scale
    beta: Array, // bias

    pub fn init(ctx: EagerContext, num_features: usize, eps: f32) !InstanceNorm {
        const shape = [_]ShapeElem{@intCast(num_features)};
        const gamma = try array_mod.ones(ctx.allocator, &shape, dtype_mod.float32);
        const beta = try ops_mod.zeros(ctx, &shape, dtype_mod.float32);

        return InstanceNorm{
            .ctx = ctx,
            .num_features = num_features,
            .eps = eps,
            .gamma = gamma,
            .beta = beta,
        };
    }

    pub fn forward(self: *InstanceNorm, input: Array) !Array {
        // input: (batch, num_features, ...)
        // InstanceNorm normalizes each instance (batch element) independently
        const shape = input.shape();
        std.debug.assert(shape.len >= 2);
        std.debug.assert(shape[1] == @as(ShapeElem, @intCast(self.num_features)));

        const batch = @as(usize, @intCast(shape[0]));
        const channels = self.num_features;

        // Compute remaining dimensions (spatial)
        var spatial_size: usize = 1;
        for (2..shape.len) |i| {
            spatial_size *= @as(usize, @intCast(shape[i]));
        }

        const out_shape = shape;
        const out = try array_mod.zeros(self.ctx.allocator, out_shape, input.dtype());
        const src = try input.dataSliceMut(f32);
        const dst = try out.dataSliceMut(f32);
        const gamma_data = try self.gamma.dataSliceMut(f32);
        const beta_data = try self.beta.dataSliceMut(f32);

        // Process each instance (batch element)
        for (0..batch) |b| {
            // Compute mean over spatial dimensions for each channel
            for (0..channels) |c| {
                var sum: f32 = 0;
                const channel_offset = b * channels * spatial_size + c * spatial_size;

                for (0..spatial_size) |s| {
                    sum += src[channel_offset + s];
                }
                const mean = sum / @as(f32, @floatFromInt(spatial_size));

                // Compute variance
                var var_sum: f32 = 0;
                for (0..spatial_size) |s| {
                    const diff = src[channel_offset + s] - mean;
                    var_sum += diff * diff;
                }
                const variance = var_sum / @as(f32, @floatFromInt(spatial_size));

                // Normalize and scale
                const inv_std = 1.0 / @sqrt(variance + self.eps);
                for (0..spatial_size) |s| {
                    const idx = channel_offset + s;
                    const normalized = (src[idx] - mean) * inv_std;
                    dst[idx] = normalized * gamma_data[c] + beta_data[c];
                }
            }
        }

        return out;
    }
};

// ============================================================
// Sinusoidal Positional Encoding
// ============================================================

pub fn sinusoidalPositionalEncoding(ctx: EagerContext, dims: usize, max_len: usize) !Array {
    // Creates sinusoidal positional encoding: (max_len, dims)
    const shape = [_]ShapeElem{ @intCast(max_len), @intCast(dims) };
    const out = try array_mod.zeros(ctx.allocator, &shape, dtype_mod.float32);
    const out_data = try out.dataSliceMut(f32);

    const half_dims = dims / 2;

    for (0..max_len) |pos| {
        for (0..half_dims) |i| {
            const freq = @exp(@as(f32, @floatFromInt(i)) * (-@log(10000.0) / @as(f32, @floatFromInt(half_dims))));
            const angle = @as(f32, @floatFromInt(pos)) * freq;

            const idx = pos * dims + i;
            out_data[idx] = @sin(angle);
            out_data[idx + half_dims] = @cos(angle);
        }
    }

    return out;
}

// ============================================================
// Simple RNN
// ============================================================

pub const RNN = struct {
    ctx: EagerContext,
    input_size: usize,
    hidden_size: usize,
    weight_ih: Array, // (hidden_size, input_size)
    weight_hh: Array, // (hidden_size, hidden_size)
    bias_ih: Array, // (hidden_size,)
    bias_hh: Array, // (hidden_size,)

    pub fn init(ctx: EagerContext, input_size: usize, hidden_size: usize) !RNN {
        const scale_ih = @sqrt(2.0 / @as(f32, @floatFromInt(input_size)));
        const scale_hh = @sqrt(2.0 / @as(f32, @floatFromInt(hidden_size)));

        const w_ih_shape = [_]ShapeElem{ @intCast(hidden_size), @intCast(input_size) };
        const w_hh_shape = [_]ShapeElem{ @intCast(hidden_size), @intCast(hidden_size) };
        const b_shape = [_]ShapeElem{@intCast(hidden_size)};

        const weight_ih = try ops_mod.random(ctx, &w_ih_shape, dtype_mod.float32, 0);
        const weight_hh = try ops_mod.random(ctx, &w_hh_shape, dtype_mod.float32, 0);
        const bias_ih = try ops_mod.zeros(ctx, &b_shape, dtype_mod.float32);
        const bias_hh = try ops_mod.zeros(ctx, &b_shape, dtype_mod.float32);

        const w_ih_data = try weight_ih.dataSliceMut(f32);
        const w_hh_data = try weight_hh.dataSliceMut(f32);
        for (0..w_ih_data.len) |i| {
            w_ih_data[i] = (std.crypto.random.float(f32) * 2 - 1) * scale_ih;
        }
        for (0..w_hh_data.len) |i| {
            w_hh_data[i] = (std.crypto.random.float(f32) * 2 - 1) * scale_hh;
        }

        return RNN{
            .ctx = ctx,
            .input_size = input_size,
            .hidden_size = hidden_size,
            .weight_ih = weight_ih,
            .weight_hh = weight_hh,
            .bias_ih = bias_ih,
            .bias_hh = bias_hh,
        };
    }

    pub fn forward(self: *RNN, input: Array, h0: ?Array) !RNNOutput {
        // input: (batch, seq_len, input_size)
        // h0: (batch, hidden_size) optional
        const shape = input.shape();
        std.debug.assert(shape.len == 3);
        const batch: i32 = shape[0];
        const seq_len = @as(usize, @intCast(shape[1]));
        const hs: i32 = @intCast(self.hidden_size);

        var prev_h = h0 orelse try creation_mod.zeros(self.ctx, &[_]ShapeElem{ batch, hs }, dtype_mod.float32);

        // Transpose weights for matmul
        const w_ih_t = try ops_mod.transpose(self.ctx, self.weight_ih);
        defer w_ih_t.deinit();
        const w_hh_t = try ops_mod.transpose(self.ctx, self.weight_hh);
        defer w_hh_t.deinit();

        var outputs = try self.ctx.allocator.alloc(Array, seq_len);
        defer self.ctx.allocator.free(outputs);

        for (0..seq_len) |t| {
            const t_i: i32 = @intCast(t);
            // Slice input at timestep t: (batch, input_size)
            const x_t = try shape_mod.slice(self.ctx, input, &[_]i32{ 0, t_i, 0 }, &[_]i32{ batch, t_i + 1, @intCast(self.input_size) }, &[_]i32{ 1, 1, 1 });
            defer x_t.deinit();
            const x_t_2d = try ops_mod.reshape(self.ctx, x_t, &[_]ShapeElem{ batch, @intCast(self.input_size) });
            defer x_t_2d.deinit();

            // RNN: h_t = tanh(x @ W_ih^T + h_{t-1} @ W_hh^T + b_ih + b_hh)
            const x_w = try ops_mod.matmul(self.ctx, x_t_2d, w_ih_t);
            defer x_w.deinit();
            const h_w = try ops_mod.matmul(self.ctx, prev_h, w_hh_t);
            defer h_w.deinit();
            const pre_1 = try ops_mod.add(self.ctx, x_w, h_w);
            defer pre_1.deinit();
            const pre_2 = try ops_mod.add(self.ctx, pre_1, self.bias_ih);
            defer pre_2.deinit();
            const pre_act = try ops_mod.add(self.ctx, pre_2, self.bias_hh);
            defer pre_act.deinit();

            // Apply tanh via GPU op
            const new_h = try ops_mod.tanh(self.ctx, pre_act);

            if (t > 0 or h0 == null) prev_h.deinit();
            prev_h = new_h;

            outputs[t] = try ops_mod.expandDims(self.ctx, new_h, 1);
        }

        const output = try shape_mod.concatenateAxis(self.ctx, outputs, 1);
        for (outputs) |o| o.deinit();

        const final_h = try ops_mod.copy(self.ctx, prev_h);
        prev_h.deinit();

        return RNNOutput{
            .output = output,
            .hidden = final_h,
        };
    }
};

pub const RNNOutput = struct {
    output: Array, // (batch, seq_len, hidden_size)
    hidden: Array, // (batch, hidden_size)
};

// ============================================================
// MultiHeadAttention
// ============================================================

pub const MultiHeadAttention = struct {
    ctx: EagerContext,
    dims: usize,
    num_heads: usize,
    head_dim: usize,
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    out_proj: Linear,

    pub fn init(ctx: EagerContext, dims: usize, num_heads: usize) !MultiHeadAttention {
        std.debug.assert(dims % num_heads == 0);
        const head_dim = dims / num_heads;

        const query_proj = try Linear.init(ctx, dims, dims, false);
        const key_proj = try Linear.init(ctx, dims, dims, false);
        const value_proj = try Linear.init(ctx, dims, dims, false);
        const out_proj = try Linear.init(ctx, dims, dims, true);

        return MultiHeadAttention{
            .ctx = ctx,
            .dims = dims,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .query_proj = query_proj,
            .key_proj = key_proj,
            .value_proj = value_proj,
            .out_proj = out_proj,
        };
    }

    pub fn forward(self: *MultiHeadAttention, queries: Array, keys: Array, values: Array) !Array {
        // queries: (batch, q_len, dims)
        // keys: (batch, k_len, dims)
        // values: (batch, v_len, dims)
        const q_shape = queries.shape();
        const batch = @as(usize, @intCast(q_shape[0]));
        const q_len = @as(usize, @intCast(q_shape[1]));

        const k_shape = keys.shape();
        const k_len = @as(usize, @intCast(k_shape[1]));

        const v_shape = values.shape();
        const v_len = @as(usize, @intCast(v_shape[1]));

        // Project Q, K, V
        const q_proj = try self.query_proj.forward(queries);
        const k_proj = try self.key_proj.forward(keys);
        const v_proj = try self.value_proj.forward(values);

        // Reshape for multi-head attention: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        const q_reshaped = try self.reshapeForAttention(q_proj, batch, q_len);
        const k_reshaped = try self.reshapeForAttention(k_proj, batch, k_len);
        const v_reshaped = try self.reshapeForAttention(v_proj, batch, v_len);

        // Compute scaled dot-product attention
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const attn_out = try scaledDotProductAttention(self.ctx, q_reshaped, k_reshaped, v_reshaped, scale);

        // Reshape back: (batch, num_heads, q_len, head_dim) -> (batch, q_len, num_heads, head_dim) -> (batch, q_len, dims)
        const attn_reshaped = try self.reshapeFromAttention(attn_out, batch, q_len);

        // Final projection
        const out = try self.out_proj.forward(attn_reshaped);

        return out;
    }

    fn reshapeForAttention(self: *MultiHeadAttention, input: Array, batch: usize, seq_len: usize) !Array {
        // (batch, seq_len, dims) -> (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        const input_reshaped = try ops_mod.reshape(self.ctx, input, &[_]ShapeElem{ @intCast(batch), @intCast(seq_len), @intCast(self.num_heads), @intCast(self.head_dim) });
        return try ops_mod.transpose(self.ctx, input_reshaped);
    }

    fn reshapeFromAttention(self: *MultiHeadAttention, input: Array, batch: usize, seq_len: usize) !Array {
        // (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, dims)
        const input_reshaped = try ops_mod.transpose(self.ctx, input);
        return try ops_mod.reshape(self.ctx, input_reshaped, &[_]ShapeElem{ @intCast(batch), @intCast(seq_len), @intCast(self.dims) });
    }
};

// Helper function for scaled dot product attention
fn scaledDotProductAttention(ctx: EagerContext, query: Array, key: Array, value: Array, scale: f32) !Array {
    // query: (batch, num_heads, q_len, head_dim)
    // key: (batch, num_heads, k_len, head_dim)
    // value: (batch, num_heads, v_len, head_dim)
    const q_shape = query.shape();
    const k_shape = key.shape();
    const v_shape = value.shape();

    const batch = @as(usize, @intCast(q_shape[0]));
    const num_heads = @as(usize, @intCast(q_shape[1]));
    const q_len = @as(usize, @intCast(q_shape[2]));
    const head_dim = @as(usize, @intCast(q_shape[3]));
    const k_len = @as(usize, @intCast(k_shape[2]));
    const v_len = @as(usize, @intCast(v_shape[2]));

    const out_shape = [_]ShapeElem{ @intCast(batch), @intCast(num_heads), @intCast(q_len), @intCast(head_dim) };
    const out = try array_mod.zeros(ctx.allocator, &out_shape, query.dtype());

    const q_data = try query.dataSliceMut(f32);
    const k_data = try key.dataSliceMut(f32);
    const v_data = try value.dataSliceMut(f32);
    const out_data = try out.dataSliceMut(f32);

    for (0..batch) |b| {
        for (0..num_heads) |h| {
            for (0..q_len) |qi| {
                var scores = try ctx.allocator.alloc(f32, k_len);
                defer ctx.allocator.free(scores);

                for (0..k_len) |ki| {
                    var dot: f32 = 0;
                    for (0..head_dim) |d| {
                        const q_idx = b * num_heads * q_len * head_dim + h * q_len * head_dim + qi * head_dim + d;
                        const k_idx = b * num_heads * k_len * head_dim + h * k_len * head_dim + ki * head_dim + d;
                        dot += q_data[q_idx] * k_data[k_idx];
                    }
                    scores[ki] = dot * scale;
                }

                // Softmax
                var max_score: f32 = -std.math.inf(f32);
                for (0..k_len) |ki| max_score = @max(max_score, scores[ki]);

                var sum_exp: f32 = 0;
                for (0..k_len) |ki| {
                    scores[ki] = @exp(scores[ki] - max_score);
                    sum_exp += scores[ki];
                }
                for (0..k_len) |ki| scores[ki] /= sum_exp;

                // Apply attention to values
                for (0..head_dim) |d| {
                    var dot: f32 = 0;
                    for (0..k_len) |ki| {
                        const v_idx = b * num_heads * v_len * head_dim + h * v_len * head_dim + ki * head_dim + d;
                        dot += scores[ki] * v_data[v_idx];
                    }
                    const out_idx = b * num_heads * q_len * head_dim + h * q_len * head_dim + qi * head_dim + d;
                    out_data[out_idx] = dot;
                }
            }
        }
    }

    return out;
}

// ============================================================
// Transformer Encoder Layer
// ============================================================

pub const TransformerEncoderLayer = struct {
    self_attn: MultiHeadAttention,
    feed_forward: struct {
        linear1: Linear,
        linear2: Linear,
    },
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: Dropout,

    pub fn init(ctx: EagerContext, dims: usize, num_heads: usize, feed_forward_dim: usize, dropout_p: f32) !TransformerEncoderLayer {
        const self_attn = try MultiHeadAttention.init(ctx, dims, num_heads);
        const linear1 = try Linear.init(ctx, dims, feed_forward_dim, true);
        const linear2 = try Linear.init(ctx, feed_forward_dim, dims, true);
        const norm1 = try LayerNorm.init(ctx, dims);
        const norm2 = try LayerNorm.init(ctx, dims);
        const dropout = Dropout.init(ctx, dropout_p);

        return TransformerEncoderLayer{
            .self_attn = self_attn,
            .feed_forward = .{ .linear1 = linear1, .linear2 = linear2 },
            .norm1 = norm1,
            .norm2 = norm2,
            .dropout = dropout,
        };
    }

    pub fn forward(self: *TransformerEncoderLayer, src: Array, src_mask: ?Array) !Array {
        _ = src_mask;
        // Self attention with residual
        const attn_out = try self.self_attn.forward(src, src, src);
        const attn_dropped = try self.dropout.forward(attn_out);
        const norm1_out = try self.norm1.forward(src, attn_dropped);

        // Feed forward with residual
        const ff1_out = try self.feed_forward.linear1.forward(norm1_out);
        const ff1_relu = try ops_mod.relu(self.ctx, ff1_out);
        const ff2_out = try self.feed_forward.linear2.forward(ff1_relu);
        const ff2_dropped = try self.dropout.forward(ff2_out);
        const norm2_out = try self.norm2.forward(norm1_out, ff2_dropped);

        return norm2_out;
    }
};

// Simple LayerNorm for use in Transformer
const LayerNorm = struct {
    ctx: EagerContext,
    weight: Array,
    bias: Array,
    eps: f32,

    pub fn init(ctx: EagerContext, dims: usize) !LayerNorm {
        const shape = [_]ShapeElem{@intCast(dims)};
        const weight = try array_mod.ones(ctx.allocator, &shape, dtype_mod.float32);
        const bias = try ops_mod.zeros(ctx, &shape, dtype_mod.float32);
        return LayerNorm{ .ctx = ctx, .weight = weight, .bias = bias, .eps = 1e-5 };
    }

    pub fn forward(self: *LayerNorm, input: Array, residual: Array) !Array {
        const shape = input.shape();
        const dims = @as(usize, @intCast(shape[shape.len - 1]));
        const size = input.size();
        const batch = size / dims;

        const out = try array_mod.zeros(self.ctx.allocator, shape, input.dtype());
        const src = try input.dataSliceMut(f32);
        const res_data = try residual.dataSliceMut(f32);
        const dst = try out.dataSliceMut(f32);
        const w_data = try self.weight.dataSliceMut(f32);
        const b_data = try self.bias.dataSliceMut(f32);

        for (0..batch) |b| {
            // Compute mean
            var mean: f32 = 0;
            for (0..dims) |d| {
                mean += src[b * dims + d];
            }
            mean /= @as(f32, @floatFromInt(dims));

            // Compute variance
            var var_sum: f32 = 0;
            for (0..dims) |d| {
                const diff = src[b * dims + d] - mean;
                var_sum += diff * diff;
            }
            const variance = var_sum / @as(f32, @floatFromInt(dims));

            // Normalize
            const inv_std = 1.0 / @sqrt(variance + self.eps);
            for (0..dims) |d| {
                const idx = b * dims + d;
                const normalized = (src[idx] - mean) * inv_std;
                dst[idx] = normalized * w_data[d] + b_data[d] + res_data[idx];
            }
        }

        return out;
    }
};

// ============================================================
// Transformer Decoder Layer
// ============================================================

pub const TransformerDecoderLayer = struct {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    feed_forward: struct {
        linear1: Linear,
        linear2: Linear,
    },
    norm1: DecoderLayerNorm,
    norm2: DecoderLayerNorm,
    norm3: DecoderLayerNorm,
    dropout: Dropout,

    pub fn init(ctx: EagerContext, dims: usize, num_heads: usize, feed_forward_dim: usize, dropout_p: f32) !TransformerDecoderLayer {
        const self_attn = try MultiHeadAttention.init(ctx, dims, num_heads);
        const cross_attn = try MultiHeadAttention.init(ctx, dims, num_heads);
        const linear1 = try Linear.init(ctx, dims, feed_forward_dim, true);
        const linear2 = try Linear.init(ctx, feed_forward_dim, dims, true);
        const norm1 = try DecoderLayerNorm.init(ctx, dims);
        const norm2 = try DecoderLayerNorm.init(ctx, dims);
        const norm3 = try DecoderLayerNorm.init(ctx, dims);
        const dropout = Dropout.init(ctx, dropout_p);

        return TransformerDecoderLayer{
            .self_attn = self_attn,
            .cross_attn = cross_attn,
            .feed_forward = .{ .linear1 = linear1, .linear2 = linear2 },
            .norm1 = norm1,
            .norm2 = norm2,
            .norm3 = norm3,
            .dropout = dropout,
        };
    }

    pub fn forward(self: *TransformerDecoderLayer, tgt: Array, memory: Array, tgt_mask: ?Array, memory_mask: ?Array) !Array {
        _ = tgt_mask;
        _ = memory_mask;
        // Self attention with residual
        const self_attn_out = try self.self_attn.forward(tgt, tgt, tgt);
        const self_attn_dropped = try self.dropout.forward(self_attn_out);
        const norm1_out = try self.norm1.forward(tgt, self_attn_dropped);

        // Cross attention with residual
        const cross_attn_out = try self.cross_attn.forward(norm1_out, memory, memory);
        const cross_attn_dropped = try self.dropout.forward(cross_attn_out);
        const norm2_out = try self.norm2.forward(norm1_out, cross_attn_dropped);

        // Feed forward with residual
        const ff1_out = try self.feed_forward.linear1.forward(norm2_out);
        const ff1_relu = try ops_mod.relu(self.ctx, ff1_out);
        const ff2_out = try self.feed_forward.linear2.forward(ff1_relu);
        const ff2_dropped = try self.dropout.forward(ff2_out);
        const norm3_out = try self.norm3.forward(norm2_out, ff2_dropped);

        return norm3_out;
    }
};

const DecoderLayerNorm = struct {
    ctx: EagerContext,
    weight: Array,
    bias: Array,
    eps: f32,

    pub fn init(ctx: EagerContext, dims: usize) !DecoderLayerNorm {
        const shape = [_]ShapeElem{@intCast(dims)};
        const weight = try array_mod.ones(ctx.allocator, &shape, dtype_mod.float32);
        const bias = try ops_mod.zeros(ctx, &shape, dtype_mod.float32);
        return DecoderLayerNorm{ .ctx = ctx, .weight = weight, .bias = bias, .eps = 1e-5 };
    }

    pub fn forward(self: *DecoderLayerNorm, input: Array, residual: Array) !Array {
        const shape = input.shape();
        const dims = @as(usize, @intCast(shape[shape.len - 1]));
        const size = input.size();
        const batch = size / dims;

        const out = try array_mod.zeros(self.ctx.allocator, shape, input.dtype());
        const src = try input.dataSliceMut(f32);
        const res_data = try residual.dataSliceMut(f32);
        const dst = try out.dataSliceMut(f32);
        const w_data = try self.weight.dataSliceMut(f32);
        const b_data = try self.bias.dataSliceMut(f32);

        for (0..batch) |b| {
            var mean: f32 = 0;
            for (0..dims) |d| {
                mean += src[b * dims + d];
            }
            mean /= @as(f32, @floatFromInt(dims));

            var var_sum: f32 = 0;
            for (0..dims) |d| {
                const diff = src[b * dims + d] - mean;
                var_sum += diff * diff;
            }
            const variance = var_sum / @as(f32, @floatFromInt(dims));

            const inv_std = 1.0 / @sqrt(variance + self.eps);
            for (0..dims) |d| {
                const idx = b * dims + d;
                const normalized = (src[idx] - mean) * inv_std;
                dst[idx] = normalized * w_data[d] + b_data[d] + res_data[idx];
            }
        }

        return out;
    }
};

// ============================================================
// RMSNorm (Root Mean Square Layer Normalization)
// Used in LLaMA, Mistral, Qwen, and modern LLMs
// Unlike LayerNorm, RMSNorm does not subtract the mean
// ============================================================

pub const RMSNorm = struct {
    ctx: EagerContext,
    dims: usize,
    eps: f32,
    weight: Array, // (dims,) — learnable scale

    pub fn init(ctx: EagerContext, dims: usize, eps: f32) !RMSNorm {
        const shape = [_]ShapeElem{@intCast(dims)};
        // Initialize weight to ones (standard for RMSNorm)
        const weight = try array_mod.ones(ctx.allocator, &shape, dtype_mod.float32);
        return RMSNorm{
            .ctx = ctx,
            .dims = dims,
            .eps = eps,
            .weight = weight,
        };
    }

    pub fn forward(self: *RMSNorm, input: Array) !Array {
        // Delegate to mlx_fast_rms_norm for GPU acceleration
        return fast_mod.rmsNorm(self.ctx, input, self.weight, self.eps);
    }
};

// ============================================================
// Embedding Layer
// Maps token indices to dense vectors
// ============================================================

pub const Embedding = struct {
    ctx: EagerContext,
    num_embeddings: usize,
    embedding_dim: usize,
    weight: Array, // (num_embeddings, embedding_dim)

    pub fn init(ctx: EagerContext, num_embeddings: usize, embedding_dim: usize) !Embedding {
        const shape = [_]ShapeElem{ @intCast(num_embeddings), @intCast(embedding_dim) };
        const weight = try ops_mod.random(ctx, &shape, dtype_mod.float32, 0);

        // Xavier-like initialization
        const scale = @sqrt(1.0 / @as(f32, @floatFromInt(num_embeddings)));
        const w_data = try weight.dataSliceMut(f32);
        for (0..w_data.len) |i| {
            w_data[i] = (std.crypto.random.float(f32) * 2.0 - 1.0) * scale;
        }

        return Embedding{
            .ctx = ctx,
            .num_embeddings = num_embeddings,
            .embedding_dim = embedding_dim,
            .weight = weight,
        };
    }

    pub fn forward(self: *Embedding, indices: Array) !Array {
        // GPU-accelerated: use mlx_take_axis to gather embeddings along axis 0
        return shape_mod.takeAxis(self.ctx, self.weight, indices, 0);
    }
};

// ============================================================
// RoPE (Rotary Position Embedding)
// Used in LLaMA, Mistral, and modern LLMs
// Applies rotation to pairs of dimensions based on position
// ============================================================

pub const RoPE = struct {
    ctx: EagerContext,
    head_dim: usize,
    max_seq_len: usize,
    theta: f32, // base frequency (default: 10000.0)
    cos_cache: Array, // (max_seq_len, head_dim/2)
    sin_cache: Array, // (max_seq_len, head_dim/2)

    pub fn init(ctx: EagerContext, head_dim: usize, max_seq_len: usize, theta: f32) !RoPE {
        std.debug.assert(head_dim % 2 == 0);

        const half_dim = head_dim / 2;
        const cache_shape = [_]ShapeElem{ @intCast(max_seq_len), @intCast(half_dim) };

        const cos_cache = try array_mod.zeros(ctx.allocator, &cache_shape, dtype_mod.float32);
        const sin_cache = try array_mod.zeros(ctx.allocator, &cache_shape, dtype_mod.float32);

        const cos_data = @constCast(try cos_cache.dataPtr(f32))[0 .. max_seq_len * half_dim];
        const sin_data = @constCast(try sin_cache.dataPtr(f32))[0 .. max_seq_len * half_dim];

        // Precompute frequencies
        for (0..max_seq_len) |pos| {
            for (0..half_dim) |i| {
                const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;
                const idx = pos * half_dim + i;
                cos_data[idx] = @cos(angle);
                sin_data[idx] = @sin(angle);
            }
        }

        return RoPE{
            .ctx = ctx,
            .head_dim = head_dim,
            .max_seq_len = max_seq_len,
            .theta = theta,
            .cos_cache = cos_cache,
            .sin_cache = sin_cache,
        };
    }

    /// Apply RoPE to query/key tensor.
    /// Input shape: (batch, num_heads, seq_len, head_dim)
    /// Output shape: same, with rotation applied
    pub fn apply(self: *RoPE, input: Array) !Array {
        const shape = input.shape();
        std.debug.assert(shape.len == 4);
        std.debug.assert(shape[3] == @as(ShapeElem, @intCast(self.head_dim)));

        const batch = @as(usize, @intCast(shape[0]));
        const num_heads = @as(usize, @intCast(shape[1]));
        const seq_len = @as(usize, @intCast(shape[2]));
        const head_dim = self.head_dim;
        const half_dim = head_dim / 2;

        const out = try array_mod.zeros(self.ctx.allocator, shape, input.dtype());
        const src = try input.dataSliceMut(f32);
        const dst = try out.dataSliceMut(f32);
        const cos_data = try self.cos_cache.dataSliceMut(f32);
        const sin_data = try self.sin_cache.dataSliceMut(f32);

        for (0..batch) |b| {
            for (0..num_heads) |h| {
                for (0..seq_len) |s| {
                    for (0..half_dim) |i| {
                        const base_idx = ((b * num_heads + h) * seq_len + s) * head_dim;
                        const x1 = src[base_idx + i];
                        const x2 = src[base_idx + half_dim + i];

                        const cache_idx = s * half_dim + i;
                        const c = cos_data[cache_idx];
                        const s_ = sin_data[cache_idx];

                        dst[base_idx + i] = x1 * c - x2 * s_;
                        dst[base_idx + half_dim + i] = x1 * s_ + x2 * c;
                    }
                }
            }
        }

        return out;
    }
};
