/// Flux-style Diffusion Transformer (DiT) scaffolding.
const std = @import("std");
const c = @import("../c.zig");
const ops = @import("../ops.zig");
const nn = @import("../ops/nn.zig");
const array_mod = @import("../array.zig");
const activations = @import("../ops/activations.zig");
const random_mod = @import("../ops/random.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

pub const FluxConfig = struct {
    in_channels: usize = 16,
    out_channels: usize = 16,
    num_layers: usize = 4,
    num_heads: usize = 4,
    hidden_size: usize = 256,
    mlp_ratio: f32 = 4.0,
    patch_size: usize = 1,
};

// ============================================================
// Flux DiT Block with adaLN-zero conditioning
// ============================================================

pub const FluxDiTBlock = struct {
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    mlp_hidden: usize,

    // Self-attention weights
    wq: Array,
    wk: Array,
    wv: Array,
    wo: Array,

    // MLP weights (SwiGLU)
    w_gate: Array,
    w_up: Array,
    w_down: Array,

    // AdaLN projections
    adaLN_attn: nn.Linear,
    adaLN_mlp: nn.Linear,

    pub fn init(_allocator: std.mem.Allocator, config: FluxConfig, ctx: EagerContext) !FluxDiTBlock {
        _ = _allocator;
        const hidden: i32 = @intCast(config.hidden_size);
        const _heads: i32 = @intCast(config.num_heads);
        const head_dim = @divExact(config.hidden_size, config.num_heads);
        const mlp_hidden: i32 = @intCast(@as(usize, @intFromFloat(@as(f32, @floatFromInt(config.hidden_size)) * config.mlp_ratio)));
        _ = _heads;

        var rng = std.Random.DefaultPrng.init(42);
        const key_arr = try random_mod.key(rng.random().int(u64));
        defer key_arr.deinit();

        const wq = try random_mod.normal(ctx, &[_]i32{ hidden, hidden }, .float32, 0.0, 0.02, key_arr);
        const wk = try random_mod.normal(ctx, &[_]i32{ hidden, hidden }, .float32, 0.0, 0.02, key_arr);
        const wv = try random_mod.normal(ctx, &[_]i32{ hidden, hidden }, .float32, 0.0, 0.02, key_arr);
        const wo = try random_mod.normal(ctx, &[_]i32{ hidden, hidden }, .float32, 0.0, 0.02, key_arr);

        const w_gate = try random_mod.normal(ctx, &[_]i32{ mlp_hidden, hidden }, .float32, 0.0, 0.02, key_arr);
        const w_up = try random_mod.normal(ctx, &[_]i32{ mlp_hidden, hidden }, .float32, 0.0, 0.02, key_arr);
        const w_down = try random_mod.normal(ctx, &[_]i32{ hidden, mlp_hidden }, .float32, 0.0, 0.02, key_arr);

        const adaLN_attn = try nn.Linear.init(ctx, @intCast(hidden), @intCast(3 * hidden), true);
        const adaLN_mlp = try nn.Linear.init(ctx, @intCast(hidden), @intCast(3 * hidden), true);

        return FluxDiTBlock{
            .hidden_size = config.hidden_size,
            .num_heads = config.num_heads,
            .head_dim = head_dim,
            .mlp_hidden = @intCast(mlp_hidden),
            .wq = wq,
            .wk = wk,
            .wv = wv,
            .wo = wo,
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
            .adaLN_attn = adaLN_attn,
            .adaLN_mlp = adaLN_mlp,
        };
    }

    pub fn deinit(self: *FluxDiTBlock) void {
        self.wq.deinit();
        self.wk.deinit();
        self.wv.deinit();
        self.wo.deinit();
        self.w_gate.deinit();
        self.w_up.deinit();
        self.w_down.deinit();
        self.adaLN_attn.weight.deinit();
        if (self.adaLN_attn.bias) |b| b.deinit();
        self.adaLN_mlp.weight.deinit();
        if (self.adaLN_mlp.bias) |b| b.deinit();
    }

    /// Forward pass.
    /// x: [B, N, D], t_emb: [B, D] -> [B, N, D]
    pub fn forward(self: *FluxDiTBlock, ctx: EagerContext, x: Array, t_emb: Array) !Array {
        // --- Attention branch with adaLN ---
        const cond_attn = try self.adaLN_attn.forward(t_emb);
        defer cond_attn.deinit();
        const cond_attn_3d = try ops.reshape(ctx, cond_attn, &[_]i32{ -1, 1, @intCast(3 * self.hidden_size) });
        defer cond_attn_3d.deinit();

        const scale_attn = try ops.slice(ctx, cond_attn_3d, &[_]i32{ 0, 0, 0 }, &[_]i32{ -1, 1, @intCast(self.hidden_size) }, &[_]i32{});
        defer scale_attn.deinit();
        const shift_attn = try ops.slice(ctx, cond_attn_3d, &[_]i32{ 0, 0, @intCast(self.hidden_size) }, &[_]i32{ -1, 1, @intCast(2 * self.hidden_size) }, &[_]i32{});
        defer shift_attn.deinit();
        const gate_attn = try ops.slice(ctx, cond_attn_3d, &[_]i32{ 0, 0, @intCast(2 * self.hidden_size) }, &[_]i32{ -1, 1, @intCast(3 * self.hidden_size) }, &[_]i32{});
        defer gate_attn.deinit();

        const x_normed = try ops.multiply(ctx, x, try ops.add(ctx, try ops.scalarF32(ctx, 1.0), scale_attn));
        defer x_normed.deinit();
        const x_shifted = try ops.add(ctx, x_normed, shift_attn);
        defer x_shifted.deinit();

        const attn_out = try self.selfAttention(ctx, x_shifted);
        defer attn_out.deinit();
        const gated_attn = try ops.multiply(ctx, attn_out, gate_attn);
        defer gated_attn.deinit();
        const x_after_attn = try ops.add(ctx, x, gated_attn);

        // --- MLP branch with adaLN ---
        const cond_mlp = try self.adaLN_mlp.forward(t_emb);
        defer cond_mlp.deinit();
        const cond_mlp_3d = try ops.reshape(ctx, cond_mlp, &[_]i32{ -1, 1, @intCast(3 * self.hidden_size) });
        defer cond_mlp_3d.deinit();

        const scale_mlp = try ops.slice(ctx, cond_mlp_3d, &[_]i32{ 0, 0, 0 }, &[_]i32{ -1, 1, @intCast(self.hidden_size) }, &[_]i32{});
        defer scale_mlp.deinit();
        const shift_mlp = try ops.slice(ctx, cond_mlp_3d, &[_]i32{ 0, 0, @intCast(self.hidden_size) }, &[_]i32{ -1, 1, @intCast(2 * self.hidden_size) }, &[_]i32{});
        defer shift_mlp.deinit();
        const gate_mlp = try ops.slice(ctx, cond_mlp_3d, &[_]i32{ 0, 0, @intCast(2 * self.hidden_size) }, &[_]i32{ -1, 1, @intCast(3 * self.hidden_size) }, &[_]i32{});
        defer gate_mlp.deinit();

        const x_normed2 = try ops.multiply(ctx, x_after_attn, try ops.add(ctx, try ops.scalarF32(ctx, 1.0), scale_mlp));
        defer x_normed2.deinit();
        const x_shifted2 = try ops.add(ctx, x_normed2, shift_mlp);
        defer x_shifted2.deinit();

        const mlp_out = try self.mlpForward(ctx, x_shifted2);
        defer mlp_out.deinit();
        const gated_mlp = try ops.multiply(ctx, mlp_out, gate_mlp);
        defer gated_mlp.deinit();
        return ops.add(ctx, x_after_attn, gated_mlp);
    }

    fn selfAttention(self: *FluxDiTBlock, ctx: EagerContext, x: Array) !Array {
        const shape = x.shape();
        const batch = shape[0];
        const seq_len = shape[1];
        const hidden: i32 = @intCast(self.hidden_size);
        const heads: i32 = @intCast(self.num_heads);
        const head_dim: i32 = @intCast(self.head_dim);

        const wq_t = try ops.transpose(ctx, self.wq);
        defer wq_t.deinit();
        const wk_t = try ops.transpose(ctx, self.wk);
        defer wk_t.deinit();
        const wv_t = try ops.transpose(ctx, self.wv);
        defer wv_t.deinit();

        var q = try ops.matmul(ctx, x, wq_t);
        var k = try ops.matmul(ctx, x, wk_t);
        var v = try ops.matmul(ctx, x, wv_t);
        defer q.deinit();
        defer k.deinit();
        defer v.deinit();

        q = try ops.reshape(ctx, q, &[_]i32{ batch, seq_len, heads, head_dim });
        k = try ops.reshape(ctx, k, &[_]i32{ batch, seq_len, heads, head_dim });
        v = try ops.reshape(ctx, v, &[_]i32{ batch, seq_len, heads, head_dim });

        const q_t = try ops.transposeAxes(ctx, q, &[_]i32{ 0, 2, 1, 3 });
        const k_t = try ops.transposeAxes(ctx, k, &[_]i32{ 0, 2, 1, 3 });
        const v_t = try ops.transposeAxes(ctx, v, &[_]i32{ 0, 2, 1, 3 });
        defer q_t.deinit();
        defer k_t.deinit();
        defer v_t.deinit();

        const scale = try ops.scalarF32(ctx, 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))));
        defer scale.deinit();

        const qk = try ops.matmul(ctx, q_t, try ops.transposeAxes(ctx, k_t, &[_]i32{ 0, 1, 3, 2 }));
        defer qk.deinit();
        const qk_scaled = try ops.multiply(ctx, qk, scale);
        defer qk_scaled.deinit();
        const attn_weights = try ops.softmax(ctx, qk_scaled, &[_]i32{3});
        defer attn_weights.deinit();
        const attn_out = try ops.matmul(ctx, attn_weights, v_t);
        defer attn_out.deinit();

        const attn_out_t = try ops.transposeAxes(ctx, attn_out, &[_]i32{ 0, 2, 1, 3 });
        defer attn_out_t.deinit();
        const attn_out_rs = try ops.reshape(ctx, attn_out_t, &[_]i32{ batch, seq_len, hidden });
        defer attn_out_rs.deinit();

        const wo_t = try ops.transpose(ctx, self.wo);
        defer wo_t.deinit();
        return ops.matmul(ctx, attn_out_rs, wo_t);
    }

    fn mlpForward(self: *FluxDiTBlock, ctx: EagerContext, x: Array) !Array {
        const w_gate_t = try ops.transpose(ctx, self.w_gate);
        defer w_gate_t.deinit();
        const w_up_t = try ops.transpose(ctx, self.w_up);
        defer w_up_t.deinit();
        const w_down_t = try ops.transpose(ctx, self.w_down);
        defer w_down_t.deinit();

        const gate = try ops.matmul(ctx, x, w_gate_t);
        defer gate.deinit();
        const up = try ops.matmul(ctx, x, w_up_t);
        defer up.deinit();
        const gate_act = try activations.silu(ctx, gate);
        defer gate_act.deinit();
        const hidden = try ops.multiply(ctx, gate_act, up);
        defer hidden.deinit();
        return ops.matmul(ctx, hidden, w_down_t);
    }
};

// ============================================================
// Flux Model
// ============================================================

pub const FluxModel = struct {
    config: FluxConfig,
    allocator: std.mem.Allocator,
    patch_embed: nn.Linear,
    blocks: []FluxDiTBlock,
    final_linear: nn.Linear,

    pub fn init(allocator: std.mem.Allocator, config: FluxConfig, ctx: EagerContext) !FluxModel {
        const patch_dim: i32 = @intCast(config.in_channels * config.patch_size * config.patch_size);
        const hidden: i32 = @intCast(config.hidden_size);

        var patch_embed = try nn.Linear.init(ctx, @intCast(patch_dim), @intCast(hidden), true);
        errdefer {
            patch_embed.weight.deinit();
            if (patch_embed.bias) |b| b.deinit();
        }

        const blocks = try allocator.alloc(FluxDiTBlock, config.num_layers);
        errdefer allocator.free(blocks);

        for (0..config.num_layers) |i| {
            blocks[i] = try FluxDiTBlock.init(allocator, config, ctx);
        }

        var final_linear = try nn.Linear.init(ctx, @intCast(hidden), @intCast(patch_dim), true);
        errdefer {
            final_linear.weight.deinit();
            if (final_linear.bias) |b| b.deinit();
        }

        return FluxModel{
            .config = config,
            .allocator = allocator,
            .patch_embed = patch_embed,
            .blocks = blocks,
            .final_linear = final_linear,
        };
    }

    pub fn deinit(self: *FluxModel) void {
        self.patch_embed.weight.deinit();
        if (self.patch_embed.bias) |b| b.deinit();
        for (self.blocks) |*block| {
            block.deinit();
        }
        self.allocator.free(self.blocks);
        self.final_linear.weight.deinit();
        if (self.final_linear.bias) |b| b.deinit();
    }

    /// Forward pass.
    /// x_patches: [B, N, patch_dim], t_emb: [B, hidden_size] -> [B, N, patch_dim]
    pub fn forward(self: *FluxModel, ctx: EagerContext, x_patches: Array, t_emb: Array) !Array {
        var h = try self.patch_embed.forward(x_patches);
        for (self.blocks) |*block| {
            const h_new = try block.forward(ctx, h, t_emb);
            h.deinit();
            h = h_new;
        }
        const out = try self.final_linear.forward(h);
        h.deinit();
        return out;
    }
};

// ============================================================
// Flux Pipeline
// ============================================================

pub const FluxPipeline = struct {
    allocator: std.mem.Allocator,
    dit: FluxModel,
    // Text encoder is a stub; pipeline accepts pre-computed prompt_embeds.
    // vae_decoder is optional for end-to-end generation.

    pub fn init(allocator: std.mem.Allocator, config: FluxConfig, ctx: EagerContext) !FluxPipeline {
        var dit = try FluxModel.init(allocator, config, ctx);
        errdefer dit.deinit();
        return FluxPipeline{
            .allocator = allocator,
            .dit = dit,
        };
    }

    pub fn deinit(self: *FluxPipeline) void {
        self.dit.deinit();
    }

    /// Generate images from prompt embeddings.
    /// prompt_embeds: [B, T, hidden_size] (ignored in this stub — t_emb is derived from timestep)
    /// height, width: spatial resolution of the output image (must be divisible by 8)
    /// num_steps: number of denoising steps
    pub fn generate(self: *FluxPipeline, ctx: EagerContext, prompt_embeds: Array, height: usize, width: usize, num_steps: usize) !Array {
        _ = prompt_embeds;
        const batch = 1;
        const latent_h = height / 8;
        const latent_w = width / 8;
        const patch_size = self.dit.config.patch_size;
        const num_patches = (latent_h / patch_size) * (latent_w / patch_size);
        const patch_dim: i32 = @intCast(self.dit.config.in_channels * patch_size * patch_size);

        // Random latent noise
        var rng = std.Random.DefaultPrng.init(42);
        const key_arr = try random_mod.key(rng.random().int(u64));
        defer key_arr.deinit();
        const latent_shape = &[_]i32{ @intCast(batch), @intCast(self.dit.config.in_channels), @intCast(latent_h), @intCast(latent_w) };
        var latent = try random_mod.normal(ctx, latent_shape, .float32, 0.0, 1.0, key_arr);

        // Simple timestep embedding (just scalar broadcast for scaffolding)
        var scheduler = try @import("scheduler.zig").FlowMatchingScheduler.init(self.allocator, 1000);
        defer scheduler.deinit();
        try scheduler.setTimesteps(num_steps);

        for (scheduler.timesteps.items) |t| {
            // Patchify latent [B, C, H, W] -> [B, N, patch_dim]
            var patches = try ops.reshape(ctx, latent, &[_]i32{ @intCast(batch), @intCast(self.dit.config.in_channels), @intCast(latent_h / patch_size), @intCast(patch_size), @intCast(latent_w / patch_size), @intCast(patch_size) });
            patches = try ops.transposeAxes(ctx, patches, &[_]i32{ 0, 2, 4, 3, 5, 1 });
            patches = try ops.reshape(ctx, patches, &[_]i32{ @intCast(batch), @intCast(num_patches), patch_dim });
            defer patches.deinit();

            // Timestep embedding: simple scalar broadcast [B, hidden_size]
            const t_emb_scalar = try ops.scalarF32(ctx, t);
            defer t_emb_scalar.deinit();
            const t_emb = try ops.broadcastTo(ctx, t_emb_scalar, &[_]i32{ @intCast(batch), @intCast(self.dit.config.hidden_size) });
            defer t_emb.deinit();

            const model_output = try self.dit.forward(ctx, patches, t_emb);
            defer model_output.deinit();

            // Unpatchify [B, N, patch_dim] -> [B, C, H, W]
            var out_rs = try ops.reshape(ctx, model_output, &[_]i32{ @intCast(batch), @intCast(latent_h / patch_size), @intCast(latent_w / patch_size), @intCast(patch_size), @intCast(patch_size), @intCast(self.dit.config.in_channels) });
            out_rs = try ops.transposeAxes(ctx, out_rs, &[_]i32{ 0, 5, 1, 3, 2, 4 });
            const out_latent = try ops.reshape(ctx, out_rs, &[_]i32{ @intCast(batch), @intCast(self.dit.config.in_channels), @intCast(latent_h), @intCast(latent_w) });
            defer out_latent.deinit();

            const new_latent = try scheduler.step(ctx, out_latent, t, latent);
            latent.deinit();
            latent = new_latent;
        }

        return latent;
    }
};

// ============================================================
// Tests
// ============================================================

test "FluxDiTBlock forward shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    var ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    const config = FluxConfig{
        .in_channels = 16,
        .out_channels = 16,
        .num_layers = 2,
        .num_heads = 4,
        .hidden_size = 64,
        .mlp_ratio = 2.0,
    };

    var block = try FluxDiTBlock.init(allocator, config, ctx);
    defer block.deinit();

    const x = try array_mod.zeros(allocator, &[_]i32{ 1, 4, 64 }, .float32);
    defer x.deinit();
    const t_emb = try array_mod.zeros(allocator, &[_]i32{ 1, 64 }, .float32);
    defer t_emb.deinit();

    const out = try block.forward(ctx, x, t_emb);
    defer out.deinit();

    const shape = out.shape();
    try std.testing.expectEqual(@as(usize, 3), shape.len);
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 4), shape[1]);
    try std.testing.expectEqual(@as(i32, 64), shape[2]);
}

test "FluxModel forward shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    var ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    const config = FluxConfig{
        .in_channels = 16,
        .out_channels = 16,
        .num_layers = 2,
        .num_heads = 4,
        .hidden_size = 64,
        .mlp_ratio = 2.0,
        .patch_size = 1,
    };

    var model = try FluxModel.init(allocator, config, ctx);
    defer model.deinit();

    const x = try array_mod.zeros(allocator, &[_]i32{ 1, 16, 16 }, .float32);
    defer x.deinit();
    const t_emb = try array_mod.zeros(allocator, &[_]i32{ 1, 64 }, .float32);
    defer t_emb.deinit();

    const out = try model.forward(ctx, x, t_emb);
    defer out.deinit();

    const shape = out.shape();
    try std.testing.expectEqual(@as(usize, 3), shape.len);
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 16), shape[1]);
    try std.testing.expectEqual(@as(i32, 16), shape[2]);
}

test "FluxPipeline generate shape" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    var ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    const config = FluxConfig{
        .in_channels = 4,
        .out_channels = 4,
        .num_layers = 1,
        .num_heads = 2,
        .hidden_size = 32,
        .mlp_ratio = 2.0,
        .patch_size = 1,
    };

    var pipeline = try FluxPipeline.init(allocator, config, ctx);
    defer pipeline.deinit();

    const prompt_embeds = try array_mod.zeros(allocator, &[_]i32{ 1, 10, 32 }, .float32);
    defer prompt_embeds.deinit();

    const result = try pipeline.generate(ctx, prompt_embeds, 16, 16, 2);
    defer result.deinit();

    const shape = result.shape();
    try std.testing.expectEqual(@as(usize, 4), shape.len);
    try std.testing.expectEqual(@as(i32, 1), shape[0]);
    try std.testing.expectEqual(@as(i32, 4), shape[1]);
    try std.testing.expectEqual(@as(i32, 2), shape[2]); // 16/8
    try std.testing.expectEqual(@as(i32, 2), shape[3]); // 16/8
}
