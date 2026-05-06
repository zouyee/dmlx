/// Simplified VAE decoder for diffusion models.
/// Operates on NCHW tensors externally; internally converts to NHWC for MLX conv.
const std = @import("std");
const c = @import("../c.zig");
const ops = @import("../ops.zig");
const conv_mod = @import("../ops/conv.zig");
const array_mod = @import("../array.zig");
const activations = @import("../ops/activations.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

pub const VAEConfig = struct {
    in_channels: usize = 3,
    out_channels: usize = 3,
    latent_channels: usize = 16,
    block_out_channels: []const usize = &.{ 128, 256, 512 },
};

/// Minimal decoder: conv_in -> mid -> upsample -> conv_out.
pub const VAEDecoder = struct {
    config: VAEConfig,
    allocator: std.mem.Allocator,

    // conv_in: [latent_channels] -> first block channel
    conv_in_weight: Array,
    conv_in_bias: ?Array,

    // mid block (simplified as one conv)
    mid_conv_weight: Array,
    mid_conv_bias: ?Array,

    // upsample transposed conv: upsample by 2x
    upsample_weight: Array,
    upsample_bias: ?Array,

    // conv_out: last block channel -> 3 RGB
    conv_out_weight: Array,
    conv_out_bias: ?Array,

    pub fn init(allocator: std.mem.Allocator, config: VAEConfig, _ctx: EagerContext) !VAEDecoder {
        _ = _ctx;
        const latent_c: i32 = @intCast(config.latent_channels);
        const first_ch: i32 = @intCast(config.block_out_channels[0]);
        const last_ch: i32 = @intCast(config.block_out_channels[config.block_out_channels.len - 1]);
        const out_ch: i32 = @intCast(config.out_channels);

        // conv_in: [C_out, 3, 3, C_in]
        const conv_in_w = try array_mod.zeros(allocator, &[_]i32{ first_ch, 3, 3, latent_c }, .float32);
        const conv_in_b = try array_mod.zeros(allocator, &[_]i32{first_ch}, .float32);

        // mid_conv
        const mid_w = try array_mod.zeros(allocator, &[_]i32{ first_ch, 3, 3, first_ch }, .float32);
        const mid_b = try array_mod.zeros(allocator, &[_]i32{first_ch}, .float32);

        // upsample: transposed conv 2x, [last_ch, 4, 4, last_ch] for 2x upsample
        const up_w = try array_mod.zeros(allocator, &[_]i32{ last_ch, 4, 4, last_ch }, .float32);
        const up_b = try array_mod.zeros(allocator, &[_]i32{last_ch}, .float32);

        // conv_out
        const conv_out_w = try array_mod.zeros(allocator, &[_]i32{ out_ch, 3, 3, last_ch }, .float32);
        const conv_out_b = try array_mod.zeros(allocator, &[_]i32{out_ch}, .float32);

        return VAEDecoder{
            .config = config,
            .allocator = allocator,
            .conv_in_weight = conv_in_w,
            .conv_in_bias = conv_in_b,
            .mid_conv_weight = mid_w,
            .mid_conv_bias = mid_b,
            .upsample_weight = up_w,
            .upsample_bias = up_b,
            .conv_out_weight = conv_out_w,
            .conv_out_bias = conv_out_b,
        };
    }

    pub fn deinit(self: *VAEDecoder) void {
        self.conv_in_weight.deinit();
        if (self.conv_in_bias) |b| b.deinit();
        self.mid_conv_weight.deinit();
        if (self.mid_conv_bias) |b| b.deinit();
        self.upsample_weight.deinit();
        if (self.upsample_bias) |b| b.deinit();
        self.conv_out_weight.deinit();
        if (self.conv_out_bias) |b| b.deinit();
    }

    /// Decode latent [B, C_latent, H, W] -> image [B, 3, H*8, W*8].
    pub fn decode(self: *VAEDecoder, ctx: EagerContext, latent: Array) !Array {
        // NCHW -> NHWC
        var x = try ops.transposeAxes(ctx, latent, &[_]i32{ 0, 2, 3, 1 });

        // conv_in
        x = try conv_mod.conv2d(ctx, x, self.conv_in_weight, 1, 1, 1, 1, 1, 1, 1);
        if (self.conv_in_bias) |b| {
            const b_reshaped = try ops.reshape(ctx, b, &[_]i32{ 1, 1, 1, -1 });
            defer b_reshaped.deinit();
            const x_new = try ops.add(ctx, x, b_reshaped);
            x.deinit();
            x = x_new;
        }
        x = try activations.silu(ctx, x);

        // mid conv
        x = try conv_mod.conv2d(ctx, x, self.mid_conv_weight, 1, 1, 1, 1, 1, 1, 1);
        if (self.mid_conv_bias) |b| {
            const b_reshaped = try ops.reshape(ctx, b, &[_]i32{ 1, 1, 1, -1 });
            defer b_reshaped.deinit();
            const x_new = try ops.add(ctx, x, b_reshaped);
            x.deinit();
            x = x_new;
        }
        x = try activations.silu(ctx, x);

        // upsample by 2x using transposed conv
        x = try conv_mod.convTranspose2d(ctx, x, self.upsample_weight, 2, 2, 1, 1, 1, 1, 0, 0, 1);
        if (self.upsample_bias) |b| {
            const b_reshaped = try ops.reshape(ctx, b, &[_]i32{ 1, 1, 1, -1 });
            defer b_reshaped.deinit();
            const x_new = try ops.add(ctx, x, b_reshaped);
            x.deinit();
            x = x_new;
        }
        x = try activations.silu(ctx, x);

        // conv_out
        x = try conv_mod.conv2d(ctx, x, self.conv_out_weight, 1, 1, 1, 1, 1, 1, 1);
        if (self.conv_out_bias) |b| {
            const b_reshaped = try ops.reshape(ctx, b, &[_]i32{ 1, 1, 1, -1 });
            defer b_reshaped.deinit();
            const x_new = try ops.add(ctx, x, b_reshaped);
            x.deinit();
            x = x_new;
        }

        // NHWC -> NCHW
        const x_nchw = try ops.transposeAxes(ctx, x, &[_]i32{ 0, 3, 1, 2 });
        x.deinit();
        return x_nchw;
    }
};

// ============================================================
// Tests
// ============================================================

test "VAEDecoder init and deinit" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    var ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    const config = VAEConfig{
        .in_channels = 3,
        .out_channels = 3,
        .latent_channels = 16,
        .block_out_channels = &.{ 128, 256, 512 },
    };

    var decoder = try VAEDecoder.init(allocator, config, ctx);
    defer decoder.deinit();
}

test "VAEDecoder decode shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    var ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    const config = VAEConfig{
        .in_channels = 3,
        .out_channels = 3,
        .latent_channels = 16,
        .block_out_channels = &.{ 128, 256, 512 },
    };

    var decoder = try VAEDecoder.init(allocator, config, ctx);
    defer decoder.deinit();

    // latent [B=1, C=16, H=4, W=4]
    const latent = try array_mod.zeros(allocator, &[_]i32{ 1, 16, 4, 4 }, .float32);
    defer latent.deinit();

    const image = try decoder.decode(ctx, latent);
    defer image.deinit();

    const shape = image.shape();
    try std.testing.expectEqual(@as(usize, 4), shape.len);
    try std.testing.expectEqual(@as(i32, 1), shape[0]); // B
    try std.testing.expectEqual(@as(i32, 3), shape[1]); // RGB
    // H and W doubled once by upsample (simplified decoder uses one 2x upsample)
    try std.testing.expectEqual(@as(i32, 8), shape[2]); // H*2
    try std.testing.expectEqual(@as(i32, 8), shape[3]); // W*2
}

test "VAEConfig defaults" {
    const config = VAEConfig{};
    try std.testing.expectEqual(@as(usize, 3), config.in_channels);
    try std.testing.expectEqual(@as(usize, 3), config.out_channels);
    try std.testing.expectEqual(@as(usize, 16), config.latent_channels);
}
