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
const c = @import("mlx").c;
const ops = @import("mlx").ops;
const array_mod = @import("mlx").array;
const sampling_mod = @import("sampling.zig");
const kvcache = @import("kvcache.zig");
const shape_mod = @import("mlx").shape;
const array_arena_mod = @import("mlx").array_arena;

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

/// Result of forwardWithHidden — logits and last-layer hidden states.
pub const ForwardWithHiddenResult = struct {
    logits: Array,
    hidden: Array,
};

pub const ModelVTable = struct {
    forward: *const fn (
        ctx: *anyopaque,
        input: Array,
        mask: ?Array,
        caches: ?[]KVCacheStrategy,
    ) anyerror!Array,
    /// Optional: forward pass that also returns the last-layer hidden states.
    /// Required for EAGLE speculative decoding.
    forwardWithHidden: ?*const fn (
        ctx: *anyopaque,
        input: Array,
        mask: ?Array,
        caches: ?[]KVCacheStrategy,
    ) anyerror!ForwardWithHiddenResult = null,
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
    /// Repetition penalty (1.0 = no penalty, >1.0 reduces repeated tokens).
    repetition_penalty: f32 = 1.0,
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
/// Returns the sampled token and its log-probability.
pub fn generateStep(
    model: ModelVTable,
    tokens: Array,
    caches: []KVCacheStrategy,
    sampler: *SamplerConfig,
    ctx: EagerContext,
) !sampling_mod.SampleResult {
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
        .repetition_penalty = config.repetition_penalty,
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

        var generated_tokens = std.ArrayList(u32).empty;
        defer generated_tokens.deinit(ctx.allocator);

        const first_result = try generateStep(model, prompt_arr, caches, &sampler, ctx);

        if (isStopToken(first_result.token, config.stop_tokens)) {
            callback(first_result.token, true);
            return;
        }
        if (config.max_tokens <= 1) {
            callback(first_result.token, true);
            return;
        }
        callback(first_result.token, false);
        try generated_tokens.append(ctx.allocator, first_result.token);
        sampler.context_tokens = generated_tokens.items;

        // Decode loop: feed one token at a time
        var prev_token = first_result.token;
        var generated: usize = 1;

        while (generated < config.max_tokens) {
            const input_arr = try Array.fromData(
                ctx.allocator,
                u32,
                &[_]u32{prev_token},
                &[_]i32{ 1, 1 },
            );
            defer input_arr.deinit();

            const next_result = try generateStep(model, input_arr, caches, &sampler, ctx);
            generated += 1;

            const is_stop = isStopToken(next_result.token, config.stop_tokens);
            const is_done = is_stop or generated >= config.max_tokens;

            callback(next_result.token, is_done);

            if (is_done) return;

            try generated_tokens.append(ctx.allocator, next_result.token);
            sampler.context_tokens = generated_tokens.items;
            prev_token = next_result.token;
        }
    }
}

/// Context-aware streaming generation. Same as `streamGenerate` but passes
/// a user-provided context pointer to the callback, enabling per-request
/// state without mutable statics.
pub fn streamGenerateCtx(
    model: ModelVTable,
    prompt_tokens: []const u32,
    config: GenerateConfig,
    caches: []KVCacheStrategy,
    ctx: EagerContext,
    ctx_ptr: *anyopaque,
    callback: *const fn (ctx_ptr: *anyopaque, token: u32, is_done: bool) void,
) !void {
    var sampler = SamplerConfig{
        .temperature = config.temperature,
        .top_k = config.top_k,
        .top_p = config.top_p,
        .prng = std.Random.DefaultPrng.init(config.seed),
        .repetition_penalty = config.repetition_penalty,
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

        var generated_tokens = std.ArrayList(u32).empty;
        defer generated_tokens.deinit(ctx.allocator);

        const first_result = try generateStep(model, prompt_arr, caches, &sampler, ctx);

        if (isStopToken(first_result.token, config.stop_tokens)) {
            callback(ctx_ptr, first_result.token, true);
            return;
        }
        if (config.max_tokens <= 1) {
            callback(ctx_ptr, first_result.token, true);
            return;
        }
        callback(ctx_ptr, first_result.token, false);
        try generated_tokens.append(ctx.allocator, first_result.token);
        sampler.context_tokens = generated_tokens.items;

        // Decode loop: feed one token at a time
        var prev_token = first_result.token;
        var generated: usize = 1;

        while (generated < config.max_tokens) {
            const input_arr = try Array.fromData(
                ctx.allocator,
                u32,
                &[_]u32{prev_token},
                &[_]i32{ 1, 1 },
            );
            defer input_arr.deinit();

            const next_result = try generateStep(model, input_arr, caches, &sampler, ctx);
            generated += 1;

            const is_stop = isStopToken(next_result.token, config.stop_tokens);
            const is_done = is_stop or generated >= config.max_tokens;

            callback(ctx_ptr, next_result.token, is_done);

            if (is_done) return;

            try generated_tokens.append(ctx.allocator, next_result.token);
            sampler.context_tokens = generated_tokens.items;
            prev_token = next_result.token;
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
        .repetition_penalty = config.repetition_penalty,
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

        var generated_tokens = std.ArrayList(u32).empty;
        defer generated_tokens.deinit(ctx.allocator);

        const first_result = try generateStep(model, prompt_arr, caches, &sampler, ctx);
        try result.append(ctx.allocator, first_result.token);
        try generated_tokens.append(ctx.allocator, first_result.token);
        sampler.context_tokens = generated_tokens.items;

        if (isStopToken(first_result.token, config.stop_tokens) or config.max_tokens <= 1) {
            return result.toOwnedSlice(ctx.allocator);
        }

        // Decode loop: feed one token at a time
        var prev_token = first_result.token;

        while (result.items.len < config.max_tokens) {
            const input_arr = try Array.fromData(
                ctx.allocator,
                u32,
                &[_]u32{prev_token},
                &[_]i32{ 1, 1 },
            );
            defer input_arr.deinit();

            const next_result = try generateStep(model, input_arr, caches, &sampler, ctx);
            try result.append(ctx.allocator, next_result.token);
            try generated_tokens.append(ctx.allocator, next_result.token);
            sampler.context_tokens = generated_tokens.items;

            if (isStopToken(next_result.token, config.stop_tokens)) break;

            prev_token = next_result.token;
        }
    }

    return result.toOwnedSlice(ctx.allocator);
}

/// Speculative decoding generation loop using a PLD drafter.
/// Falls back to single-token generation when no draft is proposed.
/// KV cache is rolled back on partial draft rejection to maintain correctness.
pub fn streamGenerateSpeculative(
    model: ModelVTable,
    prompt_tokens: []const u32,
    config: GenerateConfig,
    caches: []KVCacheStrategy,
    ctx: EagerContext,
    drafter: *const @import("speculative.zig").PldDrafter,
    callback: *const fn (token: u32, is_done: bool) void,
) !void {
    var sampler = SamplerConfig{
        .temperature = config.temperature,
        .top_k = config.top_k,
        .top_p = config.top_p,
        .prng = std.Random.DefaultPrng.init(config.seed),
        .repetition_penalty = config.repetition_penalty,
    };

    var context = std.ArrayList(u32).empty;
    defer context.deinit(ctx.allocator);
    try context.appendSlice(ctx.allocator, prompt_tokens);

    // Prefill: run the full prompt through the model
    if (prompt_tokens.len > 0) {
        const prompt_arr = try Array.fromData(
            ctx.allocator,
            u32,
            prompt_tokens,
            &[_]i32{ 1, @intCast(prompt_tokens.len) },
        );
        defer prompt_arr.deinit();

        const first_result = try generateStep(model, prompt_arr, caches, &sampler, ctx);

        if (isStopToken(first_result.token, config.stop_tokens)) {
            callback(first_result.token, true);
            return;
        }
        if (config.max_tokens <= 1) {
            callback(first_result.token, true);
            return;
        }
        callback(first_result.token, false);
        try context.append(ctx.allocator, first_result.token);
        sampler.context_tokens = context.items;

        // Decode loop with speculative decoding
        var generated: usize = 1;
        while (generated < config.max_tokens) {
            // Save KV cache lengths for potential rollback
            const cache_lens = try ctx.allocator.alloc(usize, caches.len);
            defer ctx.allocator.free(cache_lens);
            for (caches, 0..) |cache, i| {
                cache_lens[i] = cache.currentLen();
            }

            const draft = drafter.propose(context.items, drafter.n, ctx.allocator);
            if (draft) |d| {
                defer ctx.allocator.free(d);
                const vr = try @import("speculative.zig").verifyDraft(
                    model,
                    context.items,
                    d,
                    caches,
                    ctx,
                    config.seed + generated,
                );
                defer ctx.allocator.free(vr.tokens);

                // Roll back rejected draft token KV entries
                if (vr.accepted < d.len) {
                    for (caches, 0..) |cache, i| {
                        cache.rollback(cache_lens[i] + vr.accepted);
                    }
                }

                // Emit accepted draft tokens
                for (0..vr.accepted) |j| {
                    generated += 1;
                    const is_stop = isStopToken(vr.tokens[j], config.stop_tokens);
                    const is_done = is_stop or generated >= config.max_tokens;
                    callback(vr.tokens[j], is_done);
                    try context.append(ctx.allocator, vr.tokens[j]);
                    sampler.context_tokens = context.items;
                    if (is_done) return;
                }

                // The last token (bonus or resampled) needs a single-token forward
                // to correctly update KV cache, because verifyDraft's forward
                // either didn't cover it (bonus) or covered a different token (rejected).
                const last_token = vr.tokens[vr.accepted];
                const input_arr = try Array.fromData(
                    ctx.allocator,
                    u32,
                    &[_]u32{last_token},
                    &[_]i32{ 1, 1 },
                );
                defer input_arr.deinit();
                const logits = try model.forward(model.ptr, input_arr, null, caches);
                defer logits.deinit();

                generated += 1;
                const is_stop = isStopToken(last_token, config.stop_tokens);
                const is_done = is_stop or generated >= config.max_tokens;
                callback(last_token, is_done);
                try context.append(ctx.allocator, last_token);
                sampler.context_tokens = context.items;
                if (is_done) return;
            } else {
                // Fallback: single-token autoregressive generation
                const input_arr = try Array.fromData(
                    ctx.allocator,
                    u32,
                    &[_]u32{context.items[context.items.len - 1]},
                    &[_]i32{ 1, 1 },
                );
                defer input_arr.deinit();

                const next_result = try generateStep(model, input_arr, caches, &sampler, ctx);
                generated += 1;
                sampler.context_tokens = context.items;
                const is_stop = isStopToken(next_result.token, config.stop_tokens);
                const is_done = is_stop or generated >= config.max_tokens;
                callback(next_result.token, is_done);
                try context.append(ctx.allocator, next_result.token);
                if (is_done) return;
            }
        }
    }
}

// ============================================================
// Helpers
// ============================================================

/// EAGLE speculative decoding generation loop.
/// Uses the base model's hidden states to drive an EAGLE draft head.
/// Falls back to single-token generation when EAGLE drafter is not ready.
pub fn streamGenerateEagle(
    model: ModelVTable,
    prompt_tokens: []const u32,
    config: GenerateConfig,
    caches: []KVCacheStrategy,
    ctx: EagerContext,
    eagle_drafter: *const @import("speculative.zig").EagleDrafter,
    callback: *const fn (token: u32, is_done: bool) void,
) !void {
    var sampler = SamplerConfig{
        .temperature = config.temperature,
        .top_k = config.top_k,
        .top_p = config.top_p,
        .prng = std.Random.DefaultPrng.init(config.seed),
        .repetition_penalty = config.repetition_penalty,
    };

    var context = std.ArrayList(u32).empty;
    defer context.deinit(ctx.allocator);
    try context.appendSlice(ctx.allocator, prompt_tokens);

    // Prefill: run the full prompt through the model
    if (prompt_tokens.len > 0) {
        const prompt_arr = try Array.fromData(
            ctx.allocator,
            u32,
            prompt_tokens,
            &[_]i32{ 1, @intCast(prompt_tokens.len) },
        );
        defer prompt_arr.deinit();

        const first_result = try generateStep(model, prompt_arr, caches, &sampler, ctx);

        if (isStopToken(first_result.token, config.stop_tokens)) {
            callback(first_result.token, true);
            return;
        }
        if (config.max_tokens <= 1) {
            callback(first_result.token, true);
            return;
        }
        callback(first_result.token, false);
        try context.append(ctx.allocator, first_result.token);
        sampler.context_tokens = context.items;

        // Decode loop with EAGLE speculative decoding
        var generated: usize = 1;
        while (generated < config.max_tokens) {
            // Single-token forward to get hidden state for EAGLE draft
            const last_token = context.items[context.items.len - 1];
            const input_arr = try Array.fromData(
                ctx.allocator,
                u32,
                &[_]u32{last_token},
                &[_]i32{ 1, 1 },
            );
            defer input_arr.deinit();

            // Need forwardWithHidden to get hidden states
            if (model.forwardWithHidden) |fwh| {
                const fw_result = try fwh(model.ptr, input_arr, null, caches);
                defer fw_result.logits.deinit();
                defer fw_result.hidden.deinit();

                // Save KV cache lengths for potential rollback
                const cache_lens = try ctx.allocator.alloc(usize, caches.len);
                defer ctx.allocator.free(cache_lens);
                for (caches, 0..) |cache, i| {
                    cache_lens[i] = cache.currentLen();
                }

                // Get draft tokens from EAGLE draft head
                const draft = try eagle_drafter.propose(
                    fw_result.hidden,
                    eagle_drafter.num_draft_tokens,
                    ctx,
                    ctx.allocator,
                );

                if (draft) |d| {
                    defer ctx.allocator.free(d);

                    // Rollback KV cache to before the single-token forward
                    // (we'll re-verify with the full draft sequence)
                    for (caches, 0..) |cache, i| {
                        cache.rollback(cache_lens[i]);
                    }

                    const vr = try @import("speculative.zig").verifyDraft(
                        model,
                        context.items,
                        d,
                        caches,
                        ctx,
                        config.seed + generated,
                    );
                    defer ctx.allocator.free(vr.tokens);

                    // Roll back rejected draft token KV entries
                    if (vr.accepted < d.len) {
                        for (caches, 0..) |cache, i| {
                            cache.rollback(cache_lens[i] + vr.accepted);
                        }
                    }

                    // Emit accepted draft tokens
                    for (0..vr.accepted) |j| {
                        generated += 1;
                        const is_stop = isStopToken(vr.tokens[j], config.stop_tokens);
                        const is_done = is_stop or generated >= config.max_tokens;
                        callback(vr.tokens[j], is_done);
                        try context.append(ctx.allocator, vr.tokens[j]);
                        sampler.context_tokens = context.items;
                        if (is_done) return;
                    }

                    // The last token (bonus or resampled)
                    const last_draft_token = vr.tokens[vr.accepted];
                    const bonus_input = try Array.fromData(
                        ctx.allocator,
                        u32,
                        &[_]u32{last_draft_token},
                        &[_]i32{ 1, 1 },
                    );
                    defer bonus_input.deinit();
                    const bonus_logits = try model.forward(model.ptr, bonus_input, null, caches);
                    defer bonus_logits.deinit();

                    generated += 1;
                    const is_stop = isStopToken(last_draft_token, config.stop_tokens);
                    const is_done = is_stop or generated >= config.max_tokens;
                    callback(last_draft_token, is_done);
                    try context.append(ctx.allocator, last_draft_token);
                    sampler.context_tokens = context.items;
                    if (is_done) return;
                } else {
                    // EAGLE returned no draft, fall back to standard generation
                    // (the single-token forward already updated KV cache)
                    generated += 1;
                    // Extract token from the logits we already computed
                    const last_logits = try ops.slice(
                        ctx,
                        fw_result.logits,
                        &[_]i32{ 0, 0, 0 },
                        &[_]i32{ 1, 1, @intCast(model.config.vocab_size) },
                        &[_]i32{ 1, 1, 1 },
                    );
                    defer last_logits.deinit();
                    const squeezed = try shape_mod.squeezeAxes(ctx, last_logits, &[_]i32{ 0, 1 });
                    defer squeezed.deinit();
                    const logits_f32 = try ops.astype(ctx, squeezed, .float32);
                    defer logits_f32.deinit();
                    const result = sampler.sample(logits_f32, ctx.allocator);

                    const is_stop = isStopToken(result.token, config.stop_tokens);
                    const is_done = is_stop or generated >= config.max_tokens;
                    callback(result.token, is_done);
                    try context.append(ctx.allocator, result.token);
                    sampler.context_tokens = context.items;
                    if (is_done) return;
                }
            } else {
                // Model doesn't support forwardWithHidden, fall back to standard generation
                const next_result = try generateStep(model, input_arr, caches, &sampler, ctx);
                generated += 1;
                const is_stop = isStopToken(next_result.token, config.stop_tokens);
                const is_done = is_stop or generated >= config.max_tokens;
                callback(next_result.token, is_done);
                try context.append(ctx.allocator, next_result.token);
                sampler.context_tokens = context.items;
                if (is_done) return;
            }
        }
    }
}

fn isStopToken(token: u32, stop_tokens: []const u32) bool {
    for (stop_tokens) |st| {
        if (token == st) return true;
    }
    return false;
}
