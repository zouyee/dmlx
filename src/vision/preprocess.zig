/// Image preprocessing for Vision Transformer models.
/// Normalizes images using ImageNet statistics and patchifies them.
const std = @import("std");
const c = @import("mlx").c;
const ops = @import("mlx").ops;
const array_mod = @import("mlx").array;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

pub const imagenet_mean = [3]f32{ 0.485, 0.456, 0.406 };
pub const imagenet_std = [3]f32{ 0.229, 0.224, 0.225 };

/// Normalize an image using ImageNet mean and std.
/// Input shape: [H, W, 3]. Output shape: [H, W, 3].
pub fn normalizeImageNet(ctx: EagerContext, image: Array) !Array {
    const scale_255 = try ops.scalarF32(ctx, 255.0);
    defer scale_255.deinit();
    const scaled = try ops.divide(ctx, image, scale_255);
    defer scaled.deinit();

    const mean_arr = try Array.fromData(ctx.allocator, f32, &imagenet_mean, &[_]i32{3});
    defer mean_arr.deinit();
    const centered = try ops.subtract(ctx, scaled, mean_arr);
    defer centered.deinit();

    const std_arr = try Array.fromData(ctx.allocator, f32, &imagenet_std, &[_]i32{3});
    defer std_arr.deinit();
    return ops.divide(ctx, centered, std_arr);
}

/// Reshape an image [H, W, C] into patches [num_patches, patch_dim].
pub fn patchify(ctx: EagerContext, image: Array, patch_size: usize) !Array {
    const shape = image.shape();
    std.debug.assert(shape.len == 3);
    const h = @as(usize, @intCast(shape[0]));
    const w = @as(usize, @intCast(shape[1]));
    const channels = @as(usize, @intCast(shape[2]));
    std.debug.assert(h % patch_size == 0);
    std.debug.assert(w % patch_size == 0);

    const num_patches_h = h / patch_size;
    const num_patches_w = w / patch_size;
    const patch_dim = patch_size * patch_size * channels;

    const reshaped = try ops.reshape(ctx, image, &[_]i32{
        @intCast(num_patches_h),
        @intCast(patch_size),
        @intCast(num_patches_w),
        @intCast(patch_size),
        @intCast(channels),
    });
    defer reshaped.deinit();

    const transposed = try ops.transposeAxes(ctx, reshaped, &[_]i32{ 0, 2, 1, 3, 4 });
    defer transposed.deinit();

    return ops.reshape(ctx, transposed, &[_]i32{
        @intCast(num_patches_h * num_patches_w),
        @intCast(patch_dim),
    });
}

/// Full preprocessing pipeline: create array from data, normalize, and patchify.
/// `image_data` must contain `image_size * image_size * 3` f32 values in HWC order.
pub fn preprocessForViT(ctx: EagerContext, image_data: []const f32, image_size: usize, patch_size: usize) !Array {
    const image = try Array.fromData(ctx.allocator, f32, image_data, &[_]i32{ @intCast(image_size), @intCast(image_size), 3 });
    defer image.deinit();
    const normalized = try normalizeImageNet(ctx, image);
    defer normalized.deinit();
    return patchify(ctx, normalized, patch_size);
}

// ============================================================
// Tests
// ============================================================

test "normalizeImageNet scales and shifts correctly" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    // Create a 1x1x3 image with value 255.0
    const data = &[_]f32{ 255.0, 255.0, 255.0 };
    const image = try Array.fromData(allocator, f32, data, &[_]i32{ 1, 1, 3 });
    defer image.deinit();

    const out = try normalizeImageNet(ctx, image);
    defer out.deinit();

    try std.testing.expectEqual(@as(i32, 1), out.shape()[0]);
    try std.testing.expectEqual(@as(i32, 1), out.shape()[1]);
    try std.testing.expectEqual(@as(i32, 3), out.shape()[2]);

    const out_data = try out.dataSlice(f32);
    // After divide by 255: 1.0, then subtract mean, divide by std
    const expected = [_]f32{
        (1.0 - imagenet_mean[0]) / imagenet_std[0],
        (1.0 - imagenet_mean[1]) / imagenet_std[1],
        (1.0 - imagenet_mean[2]) / imagenet_std[2],
    };
    for (expected, out_data) |e, a| {
        try std.testing.expectApproxEqAbs(e, a, 1e-5);
    }
}

test "patchify reshapes image into patches" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    // 4x4 image with 3 channels = 48 values
    var data: [48]f32 = undefined;
    for (&data, 0..) |*v, i| v.* = @floatFromInt(i);
    const image = try Array.fromData(allocator, f32, &data, &[_]i32{ 4, 4, 3 });
    defer image.deinit();

    const patches = try patchify(ctx, image, 2);
    defer patches.deinit();

    // 4 patches of dimension 2*2*3 = 12
    try std.testing.expectEqual(@as(i32, 4), patches.shape()[0]);
    try std.testing.expectEqual(@as(i32, 12), patches.shape()[1]);
}

test "preprocessForViT end-to-end" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    var data: [48]f32 = undefined;
    for (&data, 0..) |*v, i| v.* = @floatFromInt(i + 1);
    const out = try preprocessForViT(ctx, &data, 4, 2);
    defer out.deinit();

    try std.testing.expectEqual(@as(i32, 4), out.shape()[0]);
    try std.testing.expectEqual(@as(i32, 12), out.shape()[1]);
}
