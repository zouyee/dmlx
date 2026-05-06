/// Vision Transformer (ViT) model implementation.
const std = @import("std");
const c = @import("../c.zig");
const ops = @import("../ops.zig");
const nn = @import("../ops/nn.zig");
const array_mod = @import("../array.zig");
const shape_mod = @import("../ops/shape.zig");
const fast_mod = @import("../ops/fast.zig");
const activations = @import("../ops/activations.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const ShapeElem = array_mod.ShapeElem;

// ============================================================
// Configuration
// ============================================================

pub const ViTConfig = struct {
    image_size: usize,
    patch_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    intermediate_size: usize,
};

// ============================================================
// LayerNorm helper
// ============================================================

const LayerNorm = struct {
    ctx: EagerContext,
    weight: Array,
    bias: Array,
    eps: f32,

    pub fn init(ctx: EagerContext, dims: usize) !LayerNorm {
        const shape = [_]ShapeElem{@intCast(dims)};
        const weight = try array_mod.ones(ctx.allocator, &shape, .float32);
        const bias = try array_mod.zeros(ctx.allocator, &shape, .float32);
        return .{ .ctx = ctx, .weight = weight, .bias = bias, .eps = 1e-5 };
    }

    pub fn forward(self: *LayerNorm, x: Array) !Array {
        return fast_mod.layerNorm(self.ctx, x, self.weight, self.bias, self.eps);
    }

    pub fn deinit(self: *LayerNorm) void {
        self.weight.deinit();
        self.bias.deinit();
    }
};

// ============================================================
// Patch Embedding
// ============================================================

pub const ViTPatchEmbed = struct {
    linear: nn.Linear,

    pub fn init(ctx: EagerContext, patch_dim: usize, hidden_size: usize) !ViTPatchEmbed {
        return .{ .linear = try nn.Linear.init(ctx, patch_dim, hidden_size, true) };
    }

    pub fn forward(self: *ViTPatchEmbed, x: Array) !Array {
        return self.linear.forward(x);
    }

    pub fn deinit(self: *ViTPatchEmbed) void {
        self.linear.weight.deinit();
        if (self.linear.bias) |b| b.deinit();
    }
};

// ============================================================
// MLP
// ============================================================

pub const ViTMLP = struct {
    ctx: EagerContext,
    fc1: nn.Linear,
    fc2: nn.Linear,

    pub fn init(ctx: EagerContext, hidden_size: usize, intermediate_size: usize) !ViTMLP {
        const fc1 = try nn.Linear.init(ctx, hidden_size, intermediate_size, true);
        const fc2 = try nn.Linear.init(ctx, intermediate_size, hidden_size, true);
        return .{ .ctx = ctx, .fc1 = fc1, .fc2 = fc2 };
    }

    pub fn forward(self: *ViTMLP, x: Array) !Array {
        const h = try self.fc1.forward(x);
        defer h.deinit();
        const act = try activations.gelu(self.ctx, h);
        defer act.deinit();
        return self.fc2.forward(act);
    }

    pub fn deinit(self: *ViTMLP) void {
        self.fc1.weight.deinit();
        if (self.fc1.bias) |b| b.deinit();
        self.fc2.weight.deinit();
        if (self.fc2.bias) |b| b.deinit();
    }
};

// ============================================================
// Attention
// ============================================================

pub const ViTAttention = struct {
    ctx: EagerContext,
    attn: nn.MultiHeadAttention,

    pub fn init(ctx: EagerContext, hidden_size: usize, num_heads: usize) !ViTAttention {
        return .{ .ctx = ctx, .attn = try nn.MultiHeadAttention.init(ctx, hidden_size, num_heads) };
    }

    pub fn forward(self: *ViTAttention, x: Array) !Array {
        return self.attn.forward(x, x, x);
    }

    pub fn deinit(self: *ViTAttention) void {
        self.attn.query_proj.weight.deinit();
        if (self.attn.query_proj.bias) |b| b.deinit();
        self.attn.key_proj.weight.deinit();
        if (self.attn.key_proj.bias) |b| b.deinit();
        self.attn.value_proj.weight.deinit();
        if (self.attn.value_proj.bias) |b| b.deinit();
        self.attn.out_proj.weight.deinit();
        if (self.attn.out_proj.bias) |b| b.deinit();
    }
};

// ============================================================
// Transformer Block
// ============================================================

pub const ViTBlock = struct {
    ctx: EagerContext,
    norm1: LayerNorm,
    attn: ViTAttention,
    norm2: LayerNorm,
    mlp: ViTMLP,

    pub fn init(ctx: EagerContext, hidden_size: usize, num_heads: usize, intermediate_size: usize) !ViTBlock {
        const norm1 = try LayerNorm.init(ctx, hidden_size);
        const attn = try ViTAttention.init(ctx, hidden_size, num_heads);
        const norm2 = try LayerNorm.init(ctx, hidden_size);
        const mlp = try ViTMLP.init(ctx, hidden_size, intermediate_size);
        return .{ .ctx = ctx, .norm1 = norm1, .attn = attn, .norm2 = norm2, .mlp = mlp };
    }

    pub fn forward(self: *ViTBlock, x: Array) !Array {
        const normed1 = try self.norm1.forward(x);
        defer normed1.deinit();
        const attn_out = try self.attn.forward(normed1);
        defer attn_out.deinit();
        const h1 = try ops.add(self.ctx, x, attn_out);
        defer h1.deinit();

        const normed2 = try self.norm2.forward(h1);
        defer normed2.deinit();
        const mlp_out = try self.mlp.forward(normed2);
        defer mlp_out.deinit();
        return ops.add(self.ctx, h1, mlp_out);
    }

    pub fn deinit(self: *ViTBlock) void {
        self.norm1.deinit();
        self.attn.deinit();
        self.norm2.deinit();
        self.mlp.deinit();
    }
};

// ============================================================
// Full ViT Model
// ============================================================

pub const ViTModel = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    config: ViTConfig,
    patch_embed: ViTPatchEmbed,
    cls_token: Array,
    pos_embed: Array,
    blocks: []ViTBlock,
    final_norm: LayerNorm,

    pub fn init(allocator: std.mem.Allocator, ctx: EagerContext, config: ViTConfig) !ViTModel {
        const patch_dim = config.patch_size * config.patch_size * 3;
        const num_patches = (config.image_size / config.patch_size) * (config.image_size / config.patch_size);

        var patch_embed = try ViTPatchEmbed.init(ctx, patch_dim, config.hidden_size);
        errdefer patch_embed.deinit();

        const cls_shape = [_]ShapeElem{ 1, @intCast(config.hidden_size) };
        const cls_token = try array_mod.zeros(ctx.allocator, &cls_shape, .float32);
        errdefer cls_token.deinit();

        const pos_shape = [_]ShapeElem{ @intCast(num_patches + 1), @intCast(config.hidden_size) };
        const pos_embed = try array_mod.zeros(ctx.allocator, &pos_shape, .float32);
        errdefer pos_embed.deinit();

        var blocks = try allocator.alloc(ViTBlock, config.num_layers);
        errdefer allocator.free(blocks);
        for (0..config.num_layers) |i| {
            blocks[i] = try ViTBlock.init(ctx, config.hidden_size, config.num_heads, config.intermediate_size);
            errdefer for (0..i + 1) |j| blocks[j].deinit();
        }

        const final_norm = try LayerNorm.init(ctx, config.hidden_size);
        errdefer final_norm.deinit();

        return .{
            .allocator = allocator,
            .ctx = ctx,
            .config = config,
            .patch_embed = patch_embed,
            .cls_token = cls_token,
            .pos_embed = pos_embed,
            .blocks = blocks,
            .final_norm = final_norm,
        };
    }

    pub fn deinit(self: *ViTModel) void {
        self.patch_embed.deinit();
        self.cls_token.deinit();
        self.pos_embed.deinit();
        for (self.blocks) |*block| block.deinit();
        self.allocator.free(self.blocks);
        self.final_norm.deinit();
    }

    pub fn forward(self: *ViTModel, ctx_: EagerContext, pixel_values: Array) !Array {
        // pixel_values: [batch, num_patches, patch_dim]
        var hidden = try self.patch_embed.forward(pixel_values);

        const shape = hidden.shape();
        const batch = shape[0];
        const hidden_size: i32 = @intCast(self.config.hidden_size);

        // Broadcast CLS token to [batch, 1, hidden_size]
        const cls_bc = try ops.broadcastTo(ctx_, self.cls_token, &[_]i32{ batch, 1, hidden_size });
        defer cls_bc.deinit();

        // Concatenate CLS + patches along seq axis (1)
        var to_concat = [_]Array{ cls_bc, hidden };
        var hidden_cls = try shape_mod.concatenateAxis(ctx_, &to_concat, 1);
        defer hidden_cls.deinit();

        // Add position embeddings
        const num_patches_plus_1: i32 = @intCast((self.config.image_size / self.config.patch_size) * (self.config.image_size / self.config.patch_size) + 1);
        const pos_bc = try ops.broadcastTo(ctx_, self.pos_embed, &[_]i32{ batch, num_patches_plus_1, hidden_size });
        defer pos_bc.deinit();
        hidden = try ops.add(ctx_, hidden_cls, pos_bc);

        // Pass through transformer blocks
        for (self.blocks) |*block| {
            const block_out = try block.forward(hidden);
            hidden.deinit();
            hidden = block_out;
        }

        // Final layer norm
        const normed = try self.final_norm.forward(hidden);
        hidden.deinit();
        return normed;
    }
};

// ============================================================
// Tests
// ============================================================

test "ViTPatchEmbed forward shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    var embed = try ViTPatchEmbed.init(ctx, 12, 32);
    defer embed.deinit();

    const input = try array_mod.zeros(allocator, &[_]i32{ 2, 4, 12 }, .float32);
    defer input.deinit();

    const out = try embed.forward(input);
    defer out.deinit();

    try std.testing.expectEqual(@as(i32, 2), out.shape()[0]);
    try std.testing.expectEqual(@as(i32, 4), out.shape()[1]);
    try std.testing.expectEqual(@as(i32, 32), out.shape()[2]);
}

test "ViTAttention forward shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    var attn = try ViTAttention.init(ctx, 32, 4);
    defer attn.deinit();

    const input = try array_mod.zeros(allocator, &[_]i32{ 1, 4, 32 }, .float32);
    defer input.deinit();

    const out = try attn.forward(input);
    defer out.deinit();

    try std.testing.expectEqual(@as(i32, 1), out.shape()[0]);
    try std.testing.expectEqual(@as(i32, 4), out.shape()[1]);
    try std.testing.expectEqual(@as(i32, 32), out.shape()[2]);
}

test "ViTModel forward shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    const config = ViTConfig{
        .image_size = 8,
        .patch_size = 2,
        .hidden_size = 16,
        .num_layers = 1,
        .num_heads = 2,
        .intermediate_size = 32,
    };

    var model = try ViTModel.init(allocator, ctx, config);
    defer model.deinit();

    // 8x8 image -> 16 patches, patch_dim = 2*2*3 = 12
    const pixel_values = try array_mod.zeros(allocator, &[_]i32{ 1, 16, 12 }, .float32);
    defer pixel_values.deinit();

    const out = try model.forward(ctx, pixel_values);
    defer out.deinit();

    try std.testing.expectEqual(@as(i32, 1), out.shape()[0]);
    try std.testing.expectEqual(@as(i32, 17), out.shape()[1]); // 16 patches + 1 CLS
    try std.testing.expectEqual(@as(i32, 16), out.shape()[2]);
}
