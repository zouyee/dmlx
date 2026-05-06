/// Flow Matching Scheduler for rectified-flow diffusion (Flux-style).
const std = @import("std");
const c = @import("../c.zig");
const ops = @import("../ops.zig");
const array_mod = @import("../array.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

pub const FlowMatchingScheduler = struct {
    num_train_timesteps: usize,
    num_inference_steps: usize,
    timesteps: std.ArrayList(f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_train_timesteps: usize) !FlowMatchingScheduler {
        var timesteps = std.ArrayList(f32).empty;
        errdefer timesteps.deinit(allocator);
        return FlowMatchingScheduler{
            .num_train_timesteps = num_train_timesteps,
            .num_inference_steps = 0,
            .timesteps = timesteps,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FlowMatchingScheduler) void {
        self.timesteps.deinit(self.allocator);
    }

    /// Generate linearly-spaced timesteps from 1.0 down to 1/N (exclusive of 0).
    pub fn setTimesteps(self: *FlowMatchingScheduler, inference_steps: usize) !void {
        self.num_inference_steps = inference_steps;
        self.timesteps.clearRetainingCapacity();
        const n: f32 = @floatFromInt(inference_steps);
        var i: usize = 0;
        while (i < inference_steps) : (i += 1) {
            const t = 1.0 - @as(f32, @floatFromInt(i)) / n;
            try self.timesteps.append(self.allocator, t);
        }
    }

    /// Single Euler step for rectified flow:
    ///   x_{t_next} = x_t + (t_next - t) * v_theta(x_t, t)
    pub fn step(self: *const FlowMatchingScheduler, ctx: EagerContext, model_output: Array, timestep: f32, sample: Array) !Array {
        const idx = self.findTimestepIndex(timestep);
        if (idx + 1 >= self.timesteps.items.len) {
            return ops.copy(ctx, sample);
        }
        const t_next = self.timesteps.items[idx + 1];
        const dt = t_next - timestep;

        const dt_arr = try ops.scalarF32(ctx, dt);
        defer dt_arr.deinit();
        const scaled = try ops.multiply(ctx, model_output, dt_arr);
        defer scaled.deinit();
        return ops.add(ctx, sample, scaled);
    }

    /// Add noise to a clean sample at timestep t:
    ///   x_t = (1 - t) * sample + t * noise
    pub fn addNoise(self: *const FlowMatchingScheduler, ctx: EagerContext, sample: Array, noise: Array, timestep: f32) !Array {
        _ = self;
        const one_minus_t = try ops.scalarF32(ctx, 1.0 - timestep);
        defer one_minus_t.deinit();
        const t_arr = try ops.scalarF32(ctx, timestep);
        defer t_arr.deinit();

        const sample_scaled = try ops.multiply(ctx, sample, one_minus_t);
        defer sample_scaled.deinit();
        const noise_scaled = try ops.multiply(ctx, noise, t_arr);
        defer noise_scaled.deinit();
        return ops.add(ctx, sample_scaled, noise_scaled);
    }

    fn findTimestepIndex(self: *const FlowMatchingScheduler, timestep: f32) usize {
        for (self.timesteps.items, 0..) |t, i| {
            if (@abs(t - timestep) < 1e-5) return i;
        }
        return 0;
    }
};

// ============================================================
// Tests
// ============================================================

test "FlowMatchingScheduler timesteps" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    var sched = try FlowMatchingScheduler.init(allocator, 1000);
    defer sched.deinit();

    try sched.setTimesteps(4);
    try std.testing.expectEqual(@as(usize, 4), sched.timesteps.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sched.timesteps.items[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), sched.timesteps.items[3], 1e-5);
}

test "FlowMatchingScheduler addNoise" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    var ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    var sched = try FlowMatchingScheduler.init(allocator, 1000);
    defer sched.deinit();
    try sched.setTimesteps(4);

    const sample = try array_mod.ones(allocator, &[_]i32{ 1, 2, 2 }, .float32);
    defer sample.deinit();
    const noise = try array_mod.ones(allocator, &[_]i32{ 1, 2, 2 }, .float32);
    defer noise.deinit();

    const noised = try sched.addNoise(ctx, sample, noise, 0.5);
    defer noised.deinit();

    const shape = noised.shape();
    try std.testing.expectEqual(@as(usize, 3), shape.len);
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 2), shape[1]);
    try std.testing.expectEqual(@as(i32, 2), shape[2]);
}

test "FlowMatchingScheduler step" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    var ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    var sched = try FlowMatchingScheduler.init(allocator, 1000);
    defer sched.deinit();
    try sched.setTimesteps(4);

    const sample = try array_mod.ones(allocator, &[_]i32{ 1, 2, 2 }, .float32);
    defer sample.deinit();
    const model_output = try array_mod.zeros(allocator, &[_]i32{ 1, 2, 2 }, .float32);
    defer model_output.deinit();

    const next = try sched.step(ctx, model_output, 1.0, sample);
    defer next.deinit();

    const shape = next.shape();
    try std.testing.expectEqual(@as(usize, 3), shape.len);
}
