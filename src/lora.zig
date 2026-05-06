/// LoRA (Low-Rank Adaptation) for efficient fine-tuning.
///
/// For a target weight matrix W, LoRA adds a low-rank update:
///   W' = W + (alpha / r) * (A @ B)
/// where:
///   - W is frozen (original pretrained weight)
///   - A: [out_features, r] initialized from Gaussian
///   - B: [r, in_features] initialized to zeros
///   - r = rank (typically 4-64)
///   - alpha = scaling factor (typically 2*r)
///
/// Only A and B are trained; W remains frozen.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

/// A single LoRA adapter for one weight matrix.
pub const LoRALayer = struct {
    ctx: EagerContext,
    rank: usize,
    alpha: f32,
    scale: f32,

    // Low-rank matrices
    a: Array, // [out_features, rank]
    b: Array, // [rank, in_features]

    pub fn init(
        ctx: EagerContext,
        out_features: usize,
        in_features: usize,
        rank: usize,
        alpha: f32,
        stream: c.c.mlx_stream,
    ) !LoRALayer {
        std.debug.assert(rank <= @min(out_features, in_features));

        const a_shape = &[_]i32{ @intCast(out_features), @intCast(rank) };
        const b_shape = &[_]i32{ @intCast(rank), @intCast(in_features) };

        // A: Gaussian init (Kaiming)
        var a_arr: c.c.mlx_array = undefined;
        try c.check(c.c.mlx_random_normal(&a_arr, a_shape.ptr, a_shape.len, c.c.MLX_FLOAT32, 0.0, 1.0, .{ .ctx = null }, stream));
        const a = Array.fromHandle(a_arr);

        // B: Zero init
        var b_arr: c.c.mlx_array = undefined;
        try c.check(c.c.mlx_zeros(&b_arr, b_shape.ptr, b_shape.len, c.c.MLX_FLOAT32, stream));
        const b = Array.fromHandle(b_arr);

        return .{
            .ctx = ctx,
            .rank = rank,
            .alpha = alpha,
            .scale = alpha / @as(f32, @floatFromInt(rank)),
            .a = a,
            .b = b,
        };
    }

    pub fn deinit(self: *LoRALayer) void {
        self.a.deinit();
        self.b.deinit();
    }

    /// Compute the LoRA update: (A @ B) * scale
    pub fn forward(self: *LoRALayer, x: Array) !Array {
        // x: [..., in_features]
        // lora = x @ B^T @ A^T * scale
        const bt = try ops.transpose(self.ctx, self.b);
        defer bt.deinit();
        const at = try ops.transpose(self.ctx, self.a);
        defer at.deinit();

        const x_bt = try ops.matmul(self.ctx, x, bt);
        defer x_bt.deinit();
        const out = try ops.matmul(self.ctx, x_bt, at);

        if (self.scale != 1.0) {
            const scaled = try ops.multiply(self.ctx, out, try ops.scalar(self.ctx, self.scale, .float32));
            out.deinit();
            return scaled;
        }
        return out;
    }

    /// Apply LoRA to a base linear output: base_out + lora(x)
    pub fn apply(self: *LoRALayer, base_out: Array, x: Array) !Array {
        const lora_out = try self.forward(x);
        defer lora_out.deinit();
        return ops.add(self.ctx, base_out, lora_out);
    }
};

/// Collection of LoRA adapters applied to specific layers.
pub const LoRAModel = struct {
    allocator: std.mem.Allocator,
    adapters: std.StringHashMap(LoRALayer),
    target_modules: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator) LoRAModel {
        return .{
            .allocator = allocator,
            .adapters = std.StringHashMap(LoRALayer).init(allocator),
            .target_modules = std.ArrayList([]const u8).empty,
        };
    }

    pub fn deinit(self: *LoRAModel) void {
        var it = self.adapters.valueIterator();
        while (it.next()) |adapter| {
            adapter.deinit();
        }
        self.adapters.deinit();
        for (self.target_modules.items) |m| {
            self.allocator.free(m);
        }
        self.target_modules.deinit(self.allocator);
    }

    /// Add a LoRA adapter for a target layer.
    pub fn addAdapter(
        self: *LoRAModel,
        name: []const u8,
        ctx: EagerContext,
        out_features: usize,
        in_features: usize,
        rank: usize,
        alpha: f32,
        stream: c.c.mlx_stream,
    ) !void {
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);

        var adapter = try LoRALayer.init(ctx, out_features, in_features, rank, alpha, stream);
        errdefer adapter.deinit();

        try self.adapters.put(key, adapter);
        try self.target_modules.append(self.allocator, key);
    }

    /// Get an adapter by name.
    pub fn getAdapter(self: *LoRAModel, name: []const u8) ?*LoRALayer {
        const ptr = self.adapters.getPtr(name);
        if (ptr) |p| return p;
        return null;
    }

    /// Merge LoRA weights into base weights (for inference speedup).
    /// W_merged = W + (A @ B) * scale
    pub fn mergeInto(self: *LoRAModel, base_weights: *std.StringHashMap(Array), stream: c.c.mlx_stream) !void {
        var it = self.adapters.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            const adapter = entry.value_ptr.*;

            const base = base_weights.getPtr(name) orelse continue;

            // Compute A @ B
            const a_bt = try ops.matmul(adapter.ctx, adapter.a, adapter.b);
            defer a_bt.deinit();

            var merged: c.c.mlx_array = undefined;
            if (adapter.scale != 1.0) {
                const scaled = try ops.multiply(adapter.ctx, a_bt, try ops.scalar(adapter.ctx, adapter.scale, .float32));
                defer scaled.deinit();
                try c.check(c.c.mlx_add(&merged, base.inner, scaled.inner, stream));
            } else {
                try c.check(c.c.mlx_add(&merged, base.inner, a_bt.inner, stream));
            }

            base.deinit();
            base.* = Array.fromHandle(merged);
        }
    }

    /// Return all trainable parameters (A and B matrices) as a flat list.
    pub fn collectParams(self: *LoRAModel, allocator: std.mem.Allocator) ![]Array {
        var params = std.ArrayList(Array).empty;
        errdefer params.deinit(allocator);

        var it = self.adapters.valueIterator();
        while (it.next()) |adapter| {
            try params.append(allocator, adapter.a);
            try params.append(allocator, adapter.b);
        }

        return params.toOwnedSlice(allocator);
    }

    /// Return pointers to all trainable parameters (for optimizer in-place updates).
    pub fn collectParamPtrs(self: *LoRAModel, allocator: std.mem.Allocator) ![]*Array {
        var ptrs = std.ArrayList(*Array).empty;
        errdefer ptrs.deinit(allocator);

        var it = self.adapters.valueIterator();
        while (it.next()) |adapter| {
            try ptrs.append(allocator, &adapter.a);
            try ptrs.append(allocator, &adapter.b);
        }

        return ptrs.toOwnedSlice(allocator);
    }

    /// Set LoRA parameters from a flat array (used in training).
    pub fn setParams(self: *LoRAModel, params: []const Array) void {
        var idx: usize = 0;
        var it = self.adapters.valueIterator();
        while (it.next()) |adapter| {
            adapter.a.deinit();
            var a_copy = c.c.mlx_array_new();
            _ = c.c.mlx_array_set(&a_copy, params[idx].inner);
            adapter.a = Array.fromHandle(a_copy);
            idx += 1;

            adapter.b.deinit();
            var b_copy = c.c.mlx_array_new();
            _ = c.c.mlx_array_set(&b_copy, params[idx].inner);
            adapter.b = Array.fromHandle(b_copy);
            idx += 1;
        }
    }
};
