/// LLaVA multi-modal model.
///
/// Wraps a vision tower (ViT), multi-modal projector, and language model.
/// For now, the vision tower weights are stored raw (encoder forward pass
/// is not yet implemented); text-only inference delegates to the language model.
const std = @import("std");
const c = @import("mlx").c;
const ops = @import("mlx").ops;
const nn = @import("mlx").nn;
const array_mod = @import("mlx").array;
const llama = @import("../models/llama.zig");
const kvcache = @import("../kvcache.zig");
const generation = @import("../generation.zig");
const vit = @import("vit.zig");
const activations = @import("mlx").activations;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const KVCacheStrategy = kvcache.KVCacheStrategy;

// ============================================================
// Configuration
// ============================================================

pub const LlavaConfig = struct {
    vit_config: vit.ViTConfig,
    projector_hidden_size: usize,
    language_hidden_size: usize,
};

// ============================================================
// Multimodal Projector
// ============================================================

pub const MultimodalProjector = struct {
    ctx: EagerContext,
    linear_1: nn.Linear,
    linear_2: nn.Linear,

    pub fn init(ctx: EagerContext, hidden_size: usize, projector_hidden_size: usize, language_hidden_size: usize) !MultimodalProjector {
        const l1 = try nn.Linear.init(ctx, hidden_size, projector_hidden_size, true);
        const l2 = try nn.Linear.init(ctx, projector_hidden_size, language_hidden_size, true);
        return .{ .ctx = ctx, .linear_1 = l1, .linear_2 = l2 };
    }

    pub fn forward(self: *MultimodalProjector, x: Array) !Array {
        const h = try self.linear_1.forward(x);
        defer h.deinit();
        const act = try activations.gelu(self.ctx, h);
        defer act.deinit();
        return self.linear_2.forward(act);
    }

    pub fn deinit(self: *MultimodalProjector) void {
        self.linear_1.weight.deinit();
        if (self.linear_1.bias) |b| b.deinit();
        self.linear_2.weight.deinit();
        if (self.linear_2.bias) |b| b.deinit();
    }
};

// ============================================================
// Legacy alias for backward compatibility with loaders
// ============================================================

pub const LlavaProjector = MultimodalProjector;

// ============================================================
// LlavaModel
// ============================================================

pub const LlavaModel = struct {
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    vit: vit.ViTModel,
    projector: MultimodalProjector,
    language_model: llama.LlamaModel,

    pub fn deinit(self: *LlavaModel) void {
        self.vit.deinit();
        self.projector.deinit();
        self.language_model.deinit();
    }

    /// Text-only inference delegates to the underlying language model.
    pub fn forward(self: *LlavaModel, input: Array, mask: ?Array, caches: ?[]KVCacheStrategy) !Array {
        return self.language_model.forward(input, mask, caches);
    }

    pub fn forwardWithHidden(self: *LlavaModel, input: Array, mask: ?Array, caches: ?[]KVCacheStrategy) !generation.ForwardWithHiddenResult {
        return self.language_model.forwardWithHidden(input, mask, caches);
    }

    /// Encode an image through ViT + projector, returning image tokens.
    /// `pixel_values` shape: [batch, num_patches, patch_dim]
    pub fn encodeImage(self: *LlavaModel, pixel_values: Array) !Array {
        const vit_out = try self.vit.forward(self.ctx, pixel_values);
        defer vit_out.deinit();
        return self.projector.forward(vit_out);
    }
};

// ============================================================
// Tests
// ============================================================

test "LlavaModel init and deinit" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    const vit_config = vit.ViTConfig{
        .image_size = 8,
        .patch_size = 2,
        .hidden_size = 32,
        .num_layers = 1,
        .num_heads = 2,
        .intermediate_size = 64,
    };
    const v = try vit.ViTModel.init(allocator, ctx, vit_config);

    const proj = try MultimodalProjector.init(ctx, 32, 64, 128);

    // Minimal dummy language model config
    const config = llama.LlamaConfig{
        .vocab_size = 100,
        .hidden_size = 128,
        .num_hidden_layers = 1,
        .num_attention_heads = 4,
        .num_key_value_heads = 4,
        .intermediate_size = 256,
        .rms_norm_eps = 1e-5,
    };

    var weights = std.StringHashMap(Array).init(allocator);
    defer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    try weights.put(try allocator.dupe(u8, "embed_tokens.weight"), try array_mod.zeros(allocator, &[_]i32{ 100, 128 }, .float32));
    try weights.put(try allocator.dupe(u8, "norm.weight"), try array_mod.ones(allocator, &[_]i32{128}, .float32));
    try weights.put(try allocator.dupe(u8, "lm_head.weight"), try array_mod.zeros(allocator, &[_]i32{ 100, 128 }, .float32));
    try weights.put(try allocator.dupe(u8, "layers.0.input_layernorm"), try array_mod.ones(allocator, &[_]i32{128}, .float32));
    try weights.put(try allocator.dupe(u8, "layers.0.post_attention_layernorm"), try array_mod.ones(allocator, &[_]i32{128}, .float32));
    try weights.put(try allocator.dupe(u8, "layers.0.attention.wq"), try array_mod.zeros(allocator, &[_]i32{ 128, 128 }, .float32));
    try weights.put(try allocator.dupe(u8, "layers.0.attention.wk"), try array_mod.zeros(allocator, &[_]i32{ 128, 128 }, .float32));
    try weights.put(try allocator.dupe(u8, "layers.0.attention.wv"), try array_mod.zeros(allocator, &[_]i32{ 128, 128 }, .float32));
    try weights.put(try allocator.dupe(u8, "layers.0.attention.wo"), try array_mod.zeros(allocator, &[_]i32{ 128, 128 }, .float32));
    try weights.put(try allocator.dupe(u8, "layers.0.mlp.gate_proj"), try array_mod.zeros(allocator, &[_]i32{ 256, 128 }, .float32));
    try weights.put(try allocator.dupe(u8, "layers.0.mlp.up_proj"), try array_mod.zeros(allocator, &[_]i32{ 256, 128 }, .float32));
    try weights.put(try allocator.dupe(u8, "layers.0.mlp.down_proj"), try array_mod.zeros(allocator, &[_]i32{ 128, 256 }, .float32));

    const lm_stream = c.c.mlx_default_cpu_stream_new();
    defer _ = c.c.mlx_stream_free(lm_stream);
    const lm_model = try @import("../models/llama_loader.zig").buildModel(allocator, &config, &weights, ctx, lm_stream, null);

    var model = LlavaModel{
        .allocator = allocator,
        .ctx = ctx,
        .vit = v,
        .projector = proj,
        .language_model = lm_model,
    };

    model.deinit();
}
