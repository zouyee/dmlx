/// Shared vision prompt assembly (CLI `dmlx chat --image` and the server's
/// OpenAI content-parts path): run each image through the model's VisionOps
/// (preprocess + ViT tower), produce the concatenated tower features/blocks
/// plus the literal <|vision_start|><|image_pad|>*n<|vision_end|> prompt
/// blocks, and (for the CLI) assemble + tokenize the chat-template prompt.
const std = @import("std");
const generation = @import("generation.zig");
const tokenizer_mod = @import("tokenizer.zig");
const shape_mod = @import("mlx").shape;
const ops = @import("mlx").ops;

const Array = @import("mlx").array.Array;

/// Tower output for a whole request plus the literal prompt block text.
pub const ProcessedImages = struct {
    data: generation.VisionData, // owned
    /// "<|vision_start|><|image_pad|>*n<|vision_end|>" per image, in order.
    blocks_text: []u8, // owned
    n_pads: usize,

    pub fn deinit(self: *ProcessedImages, allocator: std.mem.Allocator) void {
        self.data.deinit(allocator);
        allocator.free(self.blocks_text);
    }
};

/// Preprocess + tower-forward every image (in prompt order) and build the
/// concatenated features/blocks plus the literal vision block text.
pub fn processImages(
    allocator: std.mem.Allocator,
    ctx: ops.EagerContext,
    vops: *const generation.VisionOps,
    vops_ctx: *anyopaque,
    images: []const generation.VisionImage,
) !ProcessedImages {
    if (images.len == 0) return error.NoImages;

    const outs = try allocator.alloc(generation.VisionImageOut, images.len);
    var n_done: usize = 0;
    defer {
        for (outs[0..n_done]) |*o| o.features.deinit();
        allocator.free(outs);
    }
    for (images, 0..) |img, i| {
        outs[i] = try vops.processImage(vops_ctx, img);
        n_done += 1;
    }

    // Concatenate features in prompt order; collect blocks.
    var features: Array = undefined;
    if (outs.len == 1) {
        features = outs[0].features;
        n_done = 0; // ownership moved
    } else {
        const arrs = try allocator.alloc(Array, outs.len);
        defer allocator.free(arrs);
        for (outs, 0..) |*o, i| arrs[i] = o.features;
        features = try shape_mod.concatenateAxis(ctx, arrs, 0);
        // Individual arrays stay owned by outs and are freed by the defer.
    }
    errdefer features.deinit();

    const blocks = try allocator.alloc([3]i32, outs.len);
    errdefer allocator.free(blocks);
    for (outs, 0..) |*o, i| blocks[i] = o.block;

    var blocks_text = std.ArrayList(u8).empty;
    errdefer blocks_text.deinit(allocator);
    var n_pads: usize = 0;
    for (outs) |*o| {
        try blocks_text.appendSlice(allocator, "<|vision_start|>");
        for (0..o.n_pads) |_| try blocks_text.appendSlice(allocator, "<|image_pad|>");
        try blocks_text.appendSlice(allocator, "<|vision_end|>");
        n_pads += o.n_pads;
    }

    return .{
        .data = .{ .features = features, .blocks = blocks },
        .blocks_text = try blocks_text.toOwnedSlice(allocator),
        .n_pads = n_pads,
    };
}

/// Sanity check used by both entry points: the encoded prompt must carry
/// exactly the pads the tower produced (guards against a tokenizer that
/// splits the special tokens).
pub fn validatePadCount(tokens: []const u32, image_token_id: u32, want_pads: usize) !void {
    var n_pads: usize = 0;
    for (tokens) |t| n_pads += @intFromBool(t == image_token_id);
    if (n_pads != want_pads) {
        // warn under `zig build test` (the runner fails tests on err logs).
        if (@import("builtin").is_test)
            std.log.warn("vision prompt: {d} image pads encoded, tower produced {d}", .{ n_pads, want_pads })
        else
            std.log.err("vision prompt: {d} image pads encoded, tower produced {d}", .{ n_pads, want_pads });
        return error.ImageBlockMismatch;
    }
}

pub const VisionPromptOut = struct {
    tokens: []u32, // owned
    data: generation.VisionData, // owned

    pub fn deinit(self: *VisionPromptOut, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
        self.data.deinit(allocator);
    }
};

/// Full CLI flow: processImages + assemble a single-user-message prompt
/// (vision blocks first, then the text) + chat template + tokenize.
pub fn build(
    allocator: std.mem.Allocator,
    ctx: ops.EagerContext,
    chat_template: *tokenizer_mod.ChatTemplate,
    tok: tokenizer_mod.TokenizerStrategy,
    vops: *const generation.VisionOps,
    vops_ctx: *anyopaque,
    system_prompt: ?[]const u8,
    user_text: []const u8,
    images: []const generation.VisionImage,
    image_token_id: u32,
) !VisionPromptOut {
    var proc = try processImages(allocator, ctx, vops, vops_ctx, images);
    defer allocator.free(proc.blocks_text);
    errdefer proc.data.deinit(allocator);

    var content = std.ArrayList(u8).empty;
    defer content.deinit(allocator);
    try content.appendSlice(allocator, proc.blocks_text);
    try content.append(allocator, '\n');
    try content.appendSlice(allocator, user_text);

    var messages = std.ArrayList(tokenizer_mod.ChatMessage).empty;
    defer messages.deinit(allocator);
    if (system_prompt) |sp| {
        if (sp.len > 0) try messages.append(allocator, .{ .role = "system", .content = sp });
    }
    try messages.append(allocator, .{ .role = "user", .content = content.items });

    const prompt_text = try chat_template.apply(messages.items, true);
    defer allocator.free(prompt_text);

    const tokens = try tok.encode(prompt_text, false, allocator);
    errdefer allocator.free(tokens);

    try validatePadCount(tokens, image_token_id, proc.n_pads);

    return .{ .tokens = tokens, .data = proc.data };
}

test "validatePadCount: matches and mismatch" {
    const image_token: u32 = 248056;
    const ids = [_]u32{ 248045, 846, 198, 248053, 248056, 248056, 248054, 198, 72240 };
    try validatePadCount(&ids, image_token, 2);
    try std.testing.expectError(error.ImageBlockMismatch, validatePadCount(&ids, image_token, 1));
    try std.testing.expectError(error.ImageBlockMismatch, validatePadCount(&ids, image_token, 3));
}
