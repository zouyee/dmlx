const std = @import("std");

pub const MropeResult = struct {
    position_ids: [3][]i32, // t/h/w channels, len = seq_len
    rope_delta: i32, // max(pos)+1 - seq_len

    pub fn deinit(self: *MropeResult, allocator: std.mem.Allocator) void {
        for (self.position_ids) |ch| allocator.free(ch);
    }
};

pub const ComputeError = error{ ImageBlockMismatch, InvalidGridDims, OutOfMemory };

/// ids: full prompt token ids; image_blocks: one per image, in order of
/// appearance: grid_t=1, grid_h/=merge, grid_w/=merge already divided.
pub fn computePositionIds(
    allocator: std.mem.Allocator,
    ids: []const u32,
    image_blocks: []const [3]i32, // (t, h_merge, w_merge) per image
    image_token: u32, // 248056
    vision_start: u32, // 248053
) ComputeError!MropeResult {
    // Mirrors LanguageModel.get_rope_index (image-only subset): an image is
    // detected by a vision_start token immediately followed by image_token.
    var n_images: usize = 0;
    if (ids.len > 0) {
        for (ids[0 .. ids.len - 1], ids[1..]) |tok, next| {
            if (tok == vision_start and next == image_token) n_images += 1;
        }
    }
    if (n_images > image_blocks.len) return error.ImageBlockMismatch;

    var channels: [3][]i32 = undefined;
    var n_alloc: usize = 0;
    errdefer for (channels[0..n_alloc]) |ch| allocator.free(ch);
    for (&channels) |*ch| {
        ch.* = try allocator.alloc(i32, ids.len);
        n_alloc += 1;
    }

    var max_pos: i32 = -1;
    var st: usize = 0;
    var block_idx: usize = 0;
    while (block_idx < n_images) {
        // Find the next vision_start + image_token pair; the text run
        // [st, ed) includes the vision_start token and everything before the
        // first pad. Same detection rule as the counting pass above, so the
        // two passes cannot disagree on what constitutes an image block.
        var ed: usize = ids.len;
        if (ids.len > 0) {
            var i = st;
            while (i + 1 < ids.len) : (i += 1) {
                if (ids[i] == vision_start and ids[i + 1] == image_token) {
                    ed = i + 1;
                    break;
                }
            }
        }
        if (ed == ids.len) return error.ImageBlockMismatch;
        const block = image_blocks[block_idx];
        if (block[0] <= 0 or block[1] <= 0 or block[2] <= 0)
            return error.InvalidGridDims;
        const gt: usize = @intCast(block[0]);
        const gh: usize = @intCast(block[1]);
        const gw: usize = @intCast(block[2]);
        const n = gt * gh * gw;
        if (ed + n > ids.len) return error.ImageBlockMismatch;

        const st_idx: i32 = max_pos + 1; // 0 for the first segment
        const text_len: i32 = @intCast(ed - st);
        for (st..ed, 0..) |tok_i, i| {
            const p = st_idx + @as(i32, @intCast(i));
            for (channels) |ch| ch[tok_i] = p;
        }

        const base = st_idx + text_len;
        for (0..gt) |ti| {
            for (0..gh) |hi| {
                for (0..gw) |wi| {
                    const k = ed + ti * gh * gw + hi * gw + wi;
                    channels[0][k] = base + @as(i32, @intCast(ti));
                    channels[1][k] = base + @as(i32, @intCast(hi));
                    channels[2][k] = base + @as(i32, @intCast(wi));
                }
            }
        }

        const block_max = base + @as(i32, @intCast(@max(gt, @max(gh, gw)))) - 1;
        max_pos = @max(if (text_len > 0) base - 1 else -1, block_max);
        st = ed + n;
        block_idx += 1;
    }

    // Trailing text run.
    if (st < ids.len) {
        const st_idx: i32 = max_pos + 1;
        for (st..ids.len, 0..) |tok_i, i| {
            const p = st_idx + @as(i32, @intCast(i));
            for (channels) |ch| ch[tok_i] = p;
        }
        max_pos = st_idx + @as(i32, @intCast(ids.len - st)) - 1;
    }

    return .{
        .position_ids = channels,
        .rope_delta = max_pos + 1 - @as(i32, @intCast(ids.len)),
    };
}

/// Interleaved frequency selector: slot j takes channel (j mod 3) within its
/// section run. Mirrors `_interleaved_position_selector` in rope_utils.py.
pub fn interleavedPositionSelector(mrope_section: [3]u32, comptime freq_dim: usize) [freq_dim]u2 {
    var selector: [freq_dim]u2 = @splat(0);
    inline for (1..3) |dim| {
        const limit = @min(@as(usize, mrope_section[dim]) * 3, freq_dim);
        var idx: usize = dim;
        while (idx < limit) : (idx += 3) selector[idx] = dim;
    }
    return selector;
}

/// Runtime-freq_dim variant of interleavedPositionSelector (rotary dims come
/// from the parsed config, not a comptime constant). Writes `out.len` slots.
pub fn interleavedPositionSelectorRuntime(mrope_section: [3]u32, out: []u2) void {
    @memset(out, 0);
    for (1..3) |dim| {
        const limit = @min(@as(usize, mrope_section[dim]) * 3, out.len);
        var idx: usize = dim;
        while (idx < limit) : (idx += 3) out[idx] = @intCast(dim);
    }
}

const testing = std.testing;

// Oracle values verified against mlx_vlm LanguageModel.get_rope_index
// (qwen3_5/language.py) with spatial_merge_size=2 and grid (1,4,4)
// -> merge-divided block (1,2,2), 4 image pads.

test "computePositionIds: single image mid-sequence" {
    const vs: u32 = 248053;
    const img: u32 = 248056;
    const ve: u32 = 248054;
    const ids = [_]u32{ 10, 11, vs, img, img, img, img, ve, 20, 21 };
    const blocks = [_][3]i32{.{ 1, 2, 2 }};
    var res = try computePositionIds(testing.allocator, &ids, &blocks, img, vs);
    defer res.deinit(testing.allocator);

    const expected_t = [_]i32{ 0, 1, 2, 3, 3, 3, 3, 5, 6, 7 };
    const expected_h = [_]i32{ 0, 1, 2, 3, 3, 4, 4, 5, 6, 7 };
    const expected_w = [_]i32{ 0, 1, 2, 3, 4, 3, 4, 5, 6, 7 };
    try testing.expectEqualSlices(i32, &expected_t, res.position_ids[0]);
    try testing.expectEqualSlices(i32, &expected_h, res.position_ids[1]);
    try testing.expectEqualSlices(i32, &expected_w, res.position_ids[2]);
    try testing.expectEqual(@as(i32, -2), res.rope_delta);
}

test "computePositionIds: two images" {
    const vs: u32 = 248053;
    const img: u32 = 248056;
    const ve: u32 = 248054;
    const ids = [_]u32{ vs, img, img, img, img, ve, 5, vs, img, img, img, img, ve, 7, 8 };
    const blocks = [_][3]i32{ .{ 1, 2, 2 }, .{ 1, 2, 2 } };
    var res = try computePositionIds(testing.allocator, &ids, &blocks, img, vs);
    defer res.deinit(testing.allocator);

    const expected_t = [_]i32{ 0, 1, 1, 1, 1, 3, 4, 5, 6, 6, 6, 6, 8, 9, 10 };
    const expected_h = [_]i32{ 0, 1, 1, 2, 2, 3, 4, 5, 6, 6, 7, 7, 8, 9, 10 };
    const expected_w = [_]i32{ 0, 1, 2, 1, 2, 3, 4, 5, 6, 7, 6, 7, 8, 9, 10 };
    try testing.expectEqualSlices(i32, &expected_t, res.position_ids[0]);
    try testing.expectEqualSlices(i32, &expected_h, res.position_ids[1]);
    try testing.expectEqualSlices(i32, &expected_w, res.position_ids[2]);
    try testing.expectEqual(@as(i32, -4), res.rope_delta);
}

test "computePositionIds: image at start" {
    const vs: u32 = 248053;
    const img: u32 = 248056;
    const ve: u32 = 248054;
    const ids = [_]u32{ vs, img, img, img, img, ve, 3, 4, 5 };
    const blocks = [_][3]i32{.{ 1, 2, 2 }};
    var res = try computePositionIds(testing.allocator, &ids, &blocks, img, vs);
    defer res.deinit(testing.allocator);

    const expected_t = [_]i32{ 0, 1, 1, 1, 1, 3, 4, 5, 6 };
    const expected_h = [_]i32{ 0, 1, 1, 2, 2, 3, 4, 5, 6 };
    const expected_w = [_]i32{ 0, 1, 2, 1, 2, 3, 4, 5, 6 };
    try testing.expectEqualSlices(i32, &expected_t, res.position_ids[0]);
    try testing.expectEqualSlices(i32, &expected_h, res.position_ids[1]);
    try testing.expectEqualSlices(i32, &expected_w, res.position_ids[2]);
    try testing.expectEqual(@as(i32, -2), res.rope_delta);
}

test "computePositionIds: text-only" {
    const ids = [_]u32{ 1, 2, 3, 4 };
    var res = try computePositionIds(testing.allocator, &ids, &.{}, 248056, 248053);
    defer res.deinit(testing.allocator);
    const expected = [_]i32{ 0, 1, 2, 3 };
    for (res.position_ids) |ch| try testing.expectEqualSlices(i32, &expected, ch);
    try testing.expectEqual(@as(i32, 0), res.rope_delta);
}

test "interleavedPositionSelector: [11,11,10] over 32 slots" {
    const sel = interleavedPositionSelector(.{ 11, 11, 10 }, 32);
    const expected = [32]u2{ 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1 };
    try testing.expectEqual(expected, sel);
}

test "interleavedPositionSelectorRuntime agrees with the comptime variant" {
    const section = [3]u32{ 11, 11, 10 };
    inline for (.{ 1, 4, 32, 48 }) |freq_dim| {
        var buf: [freq_dim]u2 = undefined;
        interleavedPositionSelectorRuntime(section, &buf);
        try testing.expectEqual(interleavedPositionSelector(section, freq_dim), buf);
    }
    // Non-default sections, incl. truncation when section*3 exceeds freq_dim.
    var buf4: [4]u2 = undefined;
    interleavedPositionSelectorRuntime(.{ 8, 8, 16 }, &buf4);
    try testing.expectEqual([4]u2{ 0, 1, 2, 0 }, buf4);
    var buf8: [8]u2 = undefined;
    interleavedPositionSelectorRuntime(.{ 2, 1, 1 }, &buf8);
    try testing.expectEqual([8]u2{ 0, 1, 2, 0, 0, 0, 0, 0 }, buf8);
}

test "computePositionIds: failure paths" {
    const vs: u32 = 248053;
    const img: u32 = 248056;
    const ve: u32 = 248054;
    const ids = [_]u32{ 10, vs, img, img, img, img, ve, 20 };

    // Too few blocks for the detected images.
    try testing.expectError(
        error.ImageBlockMismatch,
        computePositionIds(testing.allocator, &ids, &.{}, img, vs),
    );

    // Wrong pad count: block claims 8 pads but only 4 follow.
    const bad_blocks = [_][3]i32{.{ 1, 2, 4 }};
    try testing.expectError(
        error.ImageBlockMismatch,
        computePositionIds(testing.allocator, &ids, &bad_blocks, img, vs),
    );

    // Negative/zero grid dims are rejected instead of trapping in @intCast.
    const neg_blocks = [_][3]i32{.{ 1, -2, 2 }};
    try testing.expectError(
        error.InvalidGridDims,
        computePositionIds(testing.allocator, &ids, &neg_blocks, img, vs),
    );
    const zero_blocks = [_][3]i32{.{ 1, 0, 2 }};
    try testing.expectError(
        error.InvalidGridDims,
        computePositionIds(testing.allocator, &ids, &zero_blocks, img, vs),
    );

    // A bare image_token without preceding vision_start is text, not an image.
    var res = try computePositionIds(testing.allocator, &ids, &.{}, vs, img);
    defer res.deinit(testing.allocator);
    const expected = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    for (res.position_ids) |ch| try testing.expectEqualSlices(i32, &expected, ch);
    try testing.expectEqual(@as(i32, 0), res.rope_delta);
}
