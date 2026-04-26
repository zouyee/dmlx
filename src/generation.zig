/// Three-layer generation API for LLM text generation.
///
/// Provides three abstraction levels:
///   1. `generateStep` — single forward pass → one sampled token
///   2. `streamGenerate` — loop calling generateStep, invokes callback per token
///   3. `generate` — loop calling generateStep, returns full token sequence
///
/// All layers build on `ModelVTable` for model-agnostic inference and
/// `KVCacheStrategy` for pluggable KV cache backends.
const std = @import("std");
const c = @import("c.zig");
const ops = @import("ops.zig");
const array_mod = @import("array.zig");
const sampling_mod = @import("sampling.zig");
const kvcache = @import("kvcache.zig");
const shape_mod = @import("ops/shape.zig");
const array_arena_mod = @import("array_arena.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const KVCacheStrategy = kvcache.KVCacheStrategy;
const SamplerConfig = sampling_mod.SamplerConfig;
const ScopedArrayArena = array_arena_mod.ScopedArrayArena;

// ============================================================
// Model VTable — runtime-polymorphic model interface
// ============================================================

pub const ModelConfig = struct {
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    hidden_size: usize,
    /// Per-layer compression ratios for heterogeneous KV cache sizing (DeepSeek V4).
    /// Empty for models without compressed attention (e.g. LLaMA).
    /// Values: 0 or 1 = sliding-window-only, ~4 = CSA, ~128 = HCA.
    compress_ratios: []const usize = &.{},
};

pub const ModelVTable = struct {
    forward: *const fn (
        ctx: *anyopaque,
        input: Array,
        mask: ?Array,
        caches: ?[]KVCacheStrategy,
    ) anyerror!Array,
    deinit: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) void,
    config: ModelConfig,
    ptr: *anyopaque,
};

// ============================================================
// GenerateConfig
// ============================================================

pub const GenerateConfig = struct {
    max_tokens: usize = 256,
    temperature: f32 = 0.8,
    top_k: usize = 50,
    top_p: f32 = 1.0,
    stop_tokens: []const u32 = &.{},
    seed: u64 = 0,
};

// ============================================================
// Layer 1: generateStep — single forward pass → sampled token
// ============================================================

/// Execute one forward pass through the model and sample a single token.
///
/// `tokens` should be shaped [1, seq_len] (batch=1). The function runs
/// the model forward pass, extracts the last-position logits, and samples
/// a token using the provided sampler configuration.
///
/// Returns the sampled token ID.
pub fn generateStep(
    model: ModelVTable,
    tokens: Array,
    caches: []KVCacheStrategy,
    sampler: *SamplerConfig,
    ctx: EagerContext,
) !u32 {
    var arena = ScopedArrayArena.init(ctx.allocator);
    defer arena.deinit();

    // Forward pass: [1, seq_len] → [1, seq_len, vocab_size]
    const logits = try arena.track(try model.forward(model.ptr, tokens, null, caches));

    // Extract last token logits: [1, seq_len, vocab] → [1, 1, vocab]
    const logits_shape = logits.shape();
    const seq_len = logits_shape[1];
    const vocab_size = logits_shape[2];

    const last_logits = try arena.track(try ops.slice(
        ctx,
        logits,
        &[_]i32{ 0, seq_len - 1, 0 },
        &[_]i32{ 1, seq_len, vocab_size },
        &[_]i32{ 1, 1, 1 },
    ));

    // Squeeze batch and seq dims: [1, 1, vocab] → [vocab]
    const squeezed = try arena.track(try shape_mod.squeezeAxes(ctx, last_logits, &[_]i32{ 0, 1 }));

    // Cast to float32 for CPU sampling
    const logits_f32 = try arena.track(try ops.astype(ctx, squeezed, .float32));

    // Sample
    return sampler.sample(logits_f32, ctx.allocator);
}

// ============================================================
// Layer 2: streamGenerate — per-token callback streaming
// ============================================================

/// Generate tokens one at a time, invoking `callback` after each token.
///
/// The callback receives the newly generated token and a flag indicating
/// whether generation is complete. Generation stops when:
///   - `config.max_tokens` tokens have been produced
///   - A stop token from `config.stop_tokens` is generated
///
/// Implemented in terms of `generateStep`.
pub fn streamGenerate(
    model: ModelVTable,
    prompt_tokens: []const u32,
    config: GenerateConfig,
    caches: []KVCacheStrategy,
    ctx: EagerContext,
    callback: *const fn (token: u32, is_done: bool) void,
) !void {
    var sampler = SamplerConfig{
        .temperature = config.temperature,
        .top_k = config.top_k,
        .top_p = config.top_p,
        .prng = std.Random.DefaultPrng.init(config.seed),
    };

    // Prefill: run the full prompt through the model
    if (prompt_tokens.len > 0) {
        const prompt_arr = try Array.fromData(
            ctx.allocator,
            u32,
            prompt_tokens,
            &[_]i32{ 1, @intCast(prompt_tokens.len) },
        );
        defer prompt_arr.deinit();

        const first_token = try generateStep(model, prompt_arr, caches, &sampler, ctx);

        if (isStopToken(first_token, config.stop_tokens)) {
            callback(first_token, true);
            return;
        }
        if (config.max_tokens <= 1) {
            callback(first_token, true);
            return;
        }
        callback(first_token, false);

        // Decode loop: feed one token at a time
        var prev_token = first_token;
        var generated: usize = 1;

        while (generated < config.max_tokens) {
            const input_arr = try Array.fromData(
                ctx.allocator,
                u32,
                &[_]u32{prev_token},
                &[_]i32{ 1, 1 },
            );
            defer input_arr.deinit();

            const next_token = try generateStep(model, input_arr, caches, &sampler, ctx);
            generated += 1;

            const is_stop = isStopToken(next_token, config.stop_tokens);
            const is_done = is_stop or generated >= config.max_tokens;

            callback(next_token, is_done);

            if (is_done) return;

            prev_token = next_token;
        }
    }
}

// ============================================================
// Layer 3: generate — collect full token sequence
// ============================================================

/// Generate a complete token sequence and return it.
///
/// Implemented in terms of `generateStep`. Returns only the newly
/// generated tokens (does not include the prompt).
///
/// Caller owns the returned slice and must free it with `ctx.allocator`.
pub fn generate(
    model: ModelVTable,
    prompt_tokens: []const u32,
    config: GenerateConfig,
    caches: []KVCacheStrategy,
    ctx: EagerContext,
) ![]u32 {
    var sampler = SamplerConfig{
        .temperature = config.temperature,
        .top_k = config.top_k,
        .top_p = config.top_p,
        .prng = std.Random.DefaultPrng.init(config.seed),
    };

    var result = std.ArrayList(u32).empty;
    errdefer result.deinit(ctx.allocator);

    // Prefill: run the full prompt through the model
    if (prompt_tokens.len > 0) {
        const prompt_arr = try Array.fromData(
            ctx.allocator,
            u32,
            prompt_tokens,
            &[_]i32{ 1, @intCast(prompt_tokens.len) },
        );
        defer prompt_arr.deinit();

        const first_token = try generateStep(model, prompt_arr, caches, &sampler, ctx);
        try result.append(ctx.allocator, first_token);

        if (isStopToken(first_token, config.stop_tokens) or config.max_tokens <= 1) {
            return result.toOwnedSlice(ctx.allocator);
        }

        // Decode loop: feed one token at a time
        var prev_token = first_token;

        while (result.items.len < config.max_tokens) {
            const input_arr = try Array.fromData(
                ctx.allocator,
                u32,
                &[_]u32{prev_token},
                &[_]i32{ 1, 1 },
            );
            defer input_arr.deinit();

            const next_token = try generateStep(model, input_arr, caches, &sampler, ctx);
            try result.append(ctx.allocator, next_token);

            if (isStopToken(next_token, config.stop_tokens)) break;

            prev_token = next_token;
        }
    }

    return result.toOwnedSlice(ctx.allocator);
}

// ============================================================
// Helpers
// ============================================================

fn isStopToken(token: u32, stop_tokens: []const u32) bool {
    for (stop_tokens) |st| {
        if (token == st) return true;
    }
    return false;
}
