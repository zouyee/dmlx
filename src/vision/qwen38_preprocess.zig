//! Qwen3.8 (Qwen3-VL) image preprocessing: stb_image decode -> smart_resize
//! (exact port of mlx_vlm qwen3_vl `_smart_resize_image`, itself an HF port)
//! -> bilinear resize -> normalize (x/255 - 0.5)/0.5 -> temporal duplication
//! -> patch flatten in merge-block NDHWC order matching qwen38_vit.patchEmbed.
//!
//! Pixel row order is (grid_t, bh, bw, ir, ic) — merge-block order, NOT plain
//! row-major — and each row is (channel, temporal, ky, kx) flattened, matching
//! the reference processor's reshape/transpose (0,1,4,7,5,8,3,2,6,9).
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

extern fn stbi_load_from_memory(buffer: [*c]const u8, len: c_int, x: *c_int, y: *c_int, channels_in_file: *c_int, desired_channels: c_int) [*c]u8;
extern fn stbi_image_free(data: ?*anyopaque) void;

// preprocessor_config.json (Qwen3.8-27B-4bit): size.shortest_edge/longest_edge,
// patch_size, temporal_patch_size, merge_size, image_mean/std = 0.5.
// Coupled with cfg.vision (qwen38_loader.Qwen38VisionConfig): the adapter divides
// grid_thw by cfg.vision.merge, so a checkpoint with different patch/temporal/merge
// must update BOTH places.
pub const PATCH: usize = 16;
pub const TEMPORAL_PATCH: usize = 2;
pub const MERGE: usize = 2;
pub const MIN_PIXELS: i64 = 65536;
pub const MAX_PIXELS: i64 = 16777216;

pub const PreprocessOut = struct {
    pixels: Array, // [grid_h*grid_w, 3*2*16*16] bf16 (t duplicated)
    grid_thw: [3]i32, // [1, h/16, w/16]
    n_image_pad: usize, // (h/32)*(w/32)

    pub fn deinit(self: *PreprocessOut) void {
        self.pixels.deinit();
    }
};

/// Python round(): round half to even (NOT Zig's @round, which is half-away).
fn pyRound(x: f64) i64 {
    const f = @floor(x);
    const diff = x - f;
    const fi: i64 = @intFromFloat(f);
    if (diff > 0.5) return fi + 1;
    if (diff < 0.5) return fi;
    return if (@mod(fi, 2) == 0) fi else fi + 1;
}

/// Exact port of `_smart_resize_image` (processing_qwen3_vl.py:231-254):
/// round h/w to multiples of `factor`, then clamp total pixels into
/// [min_pixels, max_pixels] preserving aspect via sqrt scaling.
pub fn smartResize(height: i64, width: i64, factor: i64, min_pixels: i64, max_pixels: i64) ![2]i64 {
    const hf: f64 = @floatFromInt(height);
    const wf: f64 = @floatFromInt(width);
    if (@max(hf, wf) / @min(hf, wf) > 200.0) return error.AspectRatioTooExtreme;
    const ff: f64 = @floatFromInt(factor);
    var h_bar = pyRound(hf / ff) * factor;
    var w_bar = pyRound(wf / ff) * factor;
    if (h_bar * w_bar > max_pixels) {
        const beta = @sqrt(hf * wf / @as(f64, @floatFromInt(max_pixels)));
        h_bar = @max(factor, @as(i64, @intFromFloat(@floor(hf / beta / ff))) * factor);
        w_bar = @max(factor, @as(i64, @intFromFloat(@floor(wf / beta / ff))) * factor);
    } else if (h_bar * w_bar < min_pixels) {
        const beta = @sqrt(@as(f64, @floatFromInt(min_pixels)) / (hf * wf));
        h_bar = @as(i64, @intFromFloat(@ceil(hf * beta / ff))) * factor;
        w_bar = @as(i64, @intFromFloat(@ceil(wf * beta / ff))) * factor;
    }
    return .{ h_bar, w_bar };
}

/// Bilinear resize (half-pixel mapping, like PIL/torch align_corners=false)
/// of an interleaved RGB u8 image into a normalized f32 HWC buffer:
/// out[dy][dx][c] = (bilinear(rgb)/255 - 0.5) / 0.5.
fn resizeBilinearNormalized(rgb: []const u8, h: usize, w: usize, out: []f32, rh: usize, rw: usize) void {
    const scale_h = @as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(rh));
    const scale_w = @as(f64, @floatFromInt(w)) / @as(f64, @floatFromInt(rw));
    for (0..rh) |dy| {
        var sy = (@as(f64, @floatFromInt(dy)) + 0.5) * scale_h - 0.5;
        sy = @max(0.0, @min(sy, @as(f64, @floatFromInt(h - 1))));
        const y0: usize = @intFromFloat(@floor(sy));
        const y1: usize = @min(y0 + 1, h - 1);
        const fy: f32 = @floatCast(sy - @as(f64, @floatFromInt(y0)));
        for (0..rw) |dx| {
            var sx = (@as(f64, @floatFromInt(dx)) + 0.5) * scale_w - 0.5;
            sx = @max(0.0, @min(sx, @as(f64, @floatFromInt(w - 1))));
            const x0: usize = @intFromFloat(@floor(sx));
            const x1: usize = @min(x0 + 1, w - 1);
            const fx: f32 = @floatCast(sx - @as(f64, @floatFromInt(x0)));
            for (0..3) |ch| {
                const p00: f32 = @floatFromInt(rgb[(y0 * w + x0) * 3 + ch]);
                const p01: f32 = @floatFromInt(rgb[(y0 * w + x1) * 3 + ch]);
                const p10: f32 = @floatFromInt(rgb[(y1 * w + x0) * 3 + ch]);
                const p11: f32 = @floatFromInt(rgb[(y1 * w + x1) * 3 + ch]);
                const top = p00 + (p01 - p00) * fx;
                const bot = p10 + (p11 - p10) * fx;
                const v = top + (bot - top) * fy;
                out[(dy * rw + dx) * 3 + ch] = (v / 255.0 - 0.5) / 0.5;
            }
        }
    }
}

/// Patch flatten in merge-block order (reference _process_one transpose
/// (0,1,4,7,5,8,3,2,6,9)): rows iterate (bh, bw, ir, ic); each row is
/// (channel, temporal, ky, kx) with the single frame duplicated along t.
fn patchifyNorm(img: []const f32, rh: usize, rw: usize, out: []f32) void {
    const gh = rh / PATCH;
    const gw = rw / PATCH;
    const row_dim = 3 * TEMPORAL_PATCH * PATCH * PATCH;
    var row: usize = 0;
    for (0..gh / MERGE) |bh| {
        for (0..gw / MERGE) |bw| {
            for (0..MERGE) |ir| {
                for (0..MERGE) |ic| {
                    const py = (bh * MERGE + ir) * PATCH;
                    const px = (bw * MERGE + ic) * PATCH;
                    const dst = out[row * row_dim ..][0..row_dim];
                    for (0..3) |ch| {
                        for (0..TEMPORAL_PATCH) |t| {
                            for (0..PATCH) |ky| {
                                for (0..PATCH) |kx| {
                                    dst[((ch * TEMPORAL_PATCH + t) * PATCH + ky) * PATCH + kx] =
                                        img[((py + ky) * rw + (px + kx)) * 3 + ch];
                                }
                            }
                        }
                    }
                    row += 1;
                }
            }
        }
    }
}

fn preprocessRgb(allocator: std.mem.Allocator, ctx: EagerContext, rgb: []const u8, h: usize, w: usize) !PreprocessOut {
    const size = try smartResize(@intCast(h), @intCast(w), @intCast(PATCH * MERGE), MIN_PIXELS, MAX_PIXELS);
    const rh: usize = @intCast(size[0]);
    const rw: usize = @intCast(size[1]);

    const norm = try allocator.alloc(f32, rh * rw * 3);
    defer allocator.free(norm);
    resizeBilinearNormalized(rgb, h, w, norm, rh, rw);

    const gh = rh / PATCH;
    const gw = rw / PATCH;
    const n_patches = gh * gw; // grid_t = 1
    const row_dim = 3 * TEMPORAL_PATCH * PATCH * PATCH;
    const flat = try allocator.alloc(f32, n_patches * row_dim);
    defer allocator.free(flat);
    patchifyNorm(norm, rh, rw, flat);

    const pixels_f32 = try Array.fromData(allocator, f32, flat, &[_]i32{ @intCast(n_patches), @intCast(row_dim) });
    defer pixels_f32.deinit();
    const pixels = try ops.astype(ctx, pixels_f32, .bfloat16);
    return .{
        .pixels = pixels,
        .grid_thw = .{ 1, @intCast(gh), @intCast(gw) },
        .n_image_pad = (gh / MERGE) * (gw / MERGE),
    };
}

pub fn preprocessImageBytes(allocator: std.mem.Allocator, ctx: EagerContext, bytes: []const u8) !PreprocessOut {
    var x: c_int = 0;
    var y: c_int = 0;
    var ch: c_int = 0;
    const len: c_int = std.math.cast(c_int, bytes.len) orelse return error.ImageTooLarge;
    const data = stbi_load_from_memory(bytes.ptr, len, &x, &y, &ch, 3);
    if (data == null) return error.ImageDecodeFailed;
    defer stbi_image_free(data);
    const w: usize = @intCast(x);
    const h: usize = @intCast(y);
    return preprocessRgb(allocator, ctx, data[0 .. w * h * 3], h, w);
}

pub fn preprocessImageFile(allocator: std.mem.Allocator, ctx: EagerContext, path: []const u8) !PreprocessOut {
    const bytes = try readFileBytes(allocator, path);
    defer allocator.free(bytes);
    return preprocessImageBytes(allocator, ctx, bytes);
}

// ============================================================
// Tests
// ============================================================

test "smartResize matches reference cases" {
    // 64x96: below min_pixels -> upscale to 224x320 (verified against
    // mlx_vlm _smart_resize_image with min=65536, max=16777216).
    try std.testing.expectEqual([2]i64{ 224, 320 }, try smartResize(64, 96, 32, MIN_PIXELS, MAX_PIXELS));
    // Already-aligned mid-size image stays put.
    try std.testing.expectEqual([2]i64{ 512, 768 }, try smartResize(512, 768, 32, MIN_PIXELS, MAX_PIXELS));
    // Python round() half-to-even: 48/32 = 1.5 -> 2; 80/32 = 2.5 -> 2.
    try std.testing.expectEqual([2]i64{ 64, 64 }, try smartResize(48, 80, 32, 3136, MAX_PIXELS));
    // Above max_pixels -> downscale preserving aspect.
    const r = try smartResize(8192, 8192, 32, MIN_PIXELS, MAX_PIXELS);
    try std.testing.expect(r[0] * r[1] <= MAX_PIXELS);
    try std.testing.expect(@mod(r[0], 32) == 0 and @mod(r[1], 32) == 0);
    try std.testing.expectError(error.AspectRatioTooExtreme, smartResize(32, 6500 * 32, 32, MIN_PIXELS, MAX_PIXELS));
}

test "preprocessImageFile: 64x96 PNG fixture end-to-end" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    var out = try preprocessImageFile(allocator, ctx, "tests/fixtures/qwen38_test_64x96.png");
    defer out.deinit();

    try std.testing.expectEqual([3]i32{ 1, 14, 20 }, out.grid_thw);
    try std.testing.expectEqual(@as(usize, 70), out.n_image_pad);
    try std.testing.expectEqual(@as(i32, 280), out.pixels.shape()[0]);
    try std.testing.expectEqual(@as(i32, 3 * 2 * 16 * 16), out.pixels.shape()[1]);

    const as_f32 = try ops.astype(ctx, out.pixels, .float32);
    defer as_f32.deinit();
    try as_f32.eval();
    const data = try as_f32.dataSlice(f32);
    var sum: f64 = 0;
    for (data) |v| sum += v;
    const mean = sum / @as(f64, @floatFromInt(data.len));
    // Expected: PIL bilinear resize + normalize on the same fixture, computed
    // from tests/fixtures/qwen38_test_64x96.png via the reference pipeline
    // (bicubic reference gives -0.0036874; bilinear gives -0.0034153).
    try std.testing.expectApproxEqAbs(@as(f64, -0.0034153075), mean, 1e-3);
}

test "preprocessImageBytes: same result as file path" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    const bytes = try readFileBytes(allocator, "tests/fixtures/qwen38_test_64x96.png");
    defer allocator.free(bytes);

    var out = try preprocessImageBytes(allocator, ctx, bytes);
    defer out.deinit();
    try std.testing.expectEqual([3]i32{ 1, 14, 20 }, out.grid_thw);
    try std.testing.expectEqual(@as(usize, 70), out.n_image_pad);
}

test "patchifyNorm: merge-block row order and (c,t,ky,kx) element order" {
    const allocator = std.testing.allocator;
    const h = 64;
    const w = 64;
    const img = try allocator.alloc(f32, h * w * 3);
    defer allocator.free(img);
    for (0..h) |y| {
        for (0..w) |x| {
            for (0..3) |ch| {
                img[(y * w + x) * 3 + ch] = @floatFromInt(y * 16384 + x * 4 + ch);
            }
        }
    }
    const row_dim = 3 * 2 * 16 * 16;
    const out = try allocator.alloc(f32, 16 * row_dim);
    defer allocator.free(out);
    patchifyNorm(img, h, w, out);

    // Row 1 = (bh0, bw0, ir0, ic1): patch top-left (0, 16).
    try std.testing.expectEqual(img[(0 * w + 16) * 3 + 0], out[1 * row_dim]);
    // Temporal duplication: t=1 identical to t=0.
    try std.testing.expectEqual(out[1 * row_dim], out[1 * row_dim + 16 * 16]);
    // Channel-major: channel 1 block starts after 2*16*16.
    try std.testing.expectEqual(img[(0 * w + 16) * 3 + 1], out[1 * row_dim + 2 * 16 * 16]);
    // Row 2 = (ir1, ic0): top-left (16, 0); channel 2 offset is 4*16*16.
    try std.testing.expectEqual(img[(16 * w + 0) * 3 + 2], out[2 * row_dim + 4 * 16 * 16]);
    // Row 4 = (bh0, bw1): top-left (0, 32).
    try std.testing.expectEqual(img[(0 * w + 32) * 3 + 0], out[4 * row_dim]);
    // Row 8 = (bh1, bw0): top-left (32, 0).
    try std.testing.expectEqual(img[(32 * w + 0) * 3 + 0], out[8 * row_dim]);
    // Within-patch (ky=3, kx=5) of row 4 = source pixel (3, 37).
    try std.testing.expectEqual(img[(3 * w + 37) * 3 + 0], out[4 * row_dim + 3 * 16 + 5]);
}

const MAX_IMAGE_FILE_BYTES: usize = 64 * 1024 * 1024;

test "resizeBilinearNormalized: 2x2 -> 4x4 known pattern" {
    // All channels equal; source pixels [[0,255],[255,0]].
    const rgb = [2 * 2 * 3]u8{
        0,   0,   0,   255, 255, 255,
        255, 255, 255, 0,   0,   0,
    };
    var out: [4 * 4 * 3]f32 = undefined;
    resizeBilinearNormalized(&rgb, 2, 2, &out, 4, 4);
    // Half-pixel bilinear: interior taps at 0.25/0.75 give 63.75/95.625/
    // 159.375/191.25 -> normalized -0.5/-0.25/0.25/0.5; edges clamp.
    const expected = [4][4]f32{
        .{ -1.0, -0.5, 0.5, 1.0 },
        .{ -0.5, -0.25, 0.25, 0.5 },
        .{ 0.5, 0.25, -0.25, -0.5 },
        .{ 1.0, 0.5, -0.5, -1.0 },
    };
    for (0..4) |y| {
        for (0..4) |x| {
            for (0..3) |ch| {
                try std.testing.expectApproxEqAbs(expected[y][x], out[(y * 4 + x) * 3 + ch], 1e-6);
            }
        }
    }
}

test "readFileBytes: ENOENT maps to FileNotFound" {
    try std.testing.expectError(error.FileNotFound, readFileBytes(std.testing.allocator, "tests/fixtures/definitely_missing_9f3a.png"));
}

fn readFileBytes(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);
    const fd = std.c.open(path_z.ptr, .{});
    if (fd < 0) return switch (@as(std.c.E, @enumFromInt(std.c._errno().*))) {
        .NOENT => error.FileNotFound,
        .ACCES => error.AccessDenied,
        else => error.OpenFailed,
    };
    defer _ = std.c.close(fd);
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    var chunk: [1 << 16]u8 = undefined;
    while (true) {
        const n = try std.posix.read(fd, &chunk);
        if (n == 0) break;
        if (buf.items.len + n > MAX_IMAGE_FILE_BYTES) return error.FileTooLarge;
        try buf.appendSlice(allocator, chunk[0..n]);
    }
    return buf.toOwnedSlice(allocator);
}
