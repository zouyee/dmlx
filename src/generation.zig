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
    /// Optional MTP (multi-token prediction) speculative-decoding capability.
    /// Null for models without a drafter head. See MtpOps.
    mtp: ?*const MtpOps = null,
    /// Optional vision capability (Qwen3.8). Null for text-only models.
    vision: ?*const VisionOps = null,
    /// Optional: text embedding from the last-layer hidden state
    /// (/v1/embeddings). Contract: `ids` are raw-text tokens (no chat
    /// template); the op runs a FRESH prefill (resets any recurrent/cache
    /// state), takes the LAST position's POST-final-norm hidden, L2-normalizes
    /// it, and returns it as an owned [hidden] f32 slice (allocator-owned).
    /// Null for models without hidden-state access — the server maps that to
    /// a 501 embedding_not_supported error.
    embed: ?*const fn (ctx: *anyopaque, allocator: std.mem.Allocator, ids: []const u32) anyerror![]f32 = null,
    deinit: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) void,
    config: ModelConfig,
    ptr: *anyopaque,
};

// ============================================================
// MTP speculative decoding vtable (Qwen3.8; see qwen38_mtp.zig)
// ============================================================

/// One drafter/prefill step output: token logits [vocab] f32 (owned) +
/// pre-final-norm hidden [1, hidden] bf16 (owned) for chaining.
pub const MtpStepOut = struct {
    logits: Array,
    hidden: Array,
};

/// Greedy GPU-chained drafter step output: the argmax token id kept on GPU
/// ([1] i32, unevaled) + chained hidden. Lets the engine run the whole draft
/// chain without a CPU sync per step; one sync reads all N tokens at the end.
pub const MtpGreedyDraftOut = struct {
    tok_gpu: Array,
    hidden: Array,
    /// DMLX_MTP_AVTRACE only: [2] f32 (l_max, top1-top2 margin) from the
    /// drafter head's exact rescore, kept on GPU for the batch readback.
    /// Null on every non-traced step.
    feat: ?Array = null,
};

// ============================================================
// Vision input plumbing (Qwen3.8; tower+preprocess live behind the vtable)
// ============================================================

/// One image supplied by a caller: a filesystem path (CLI) or encoded image
/// bytes (server data-URL decode).
pub const VisionImage = union(enum) {
    path: []const u8,
    bytes: []const u8,
};

/// Tower output for ONE image: merge-divided features plus the prompt
/// bookkeeping the caller needs to place the image-pad tokens.
pub const VisionImageOut = struct {
    features: Array, // [n_pads, hidden] bf16 (owned by the caller)
    block: [3]i32, // (grid_t, grid_h/merge, grid_w/merge)
    n_pads: usize,
};

/// Full per-request vision payload: per-image features concatenated in
/// prompt order plus one merge-divided block per image. Owned by the
/// container (CLI scope / RequestState); deinit releases both.
pub const VisionData = struct {
    features: Array, // [total_pads, hidden] bf16
    blocks: [][3]i32,

    pub fn deinit(self: *VisionData, allocator: std.mem.Allocator) void {
        self.features.deinit();
        allocator.free(self.blocks);
    }
};

/// Vision capability: image preprocessing + tower forward are one op (the
/// adapter lazily loads the tower weights), the prefill splices features in
/// at the image-pad positions and runs MRoPE. Null for text-only models.
pub const VisionOps = struct {
    /// Token id of the image-pad placeholder (Qwen3.8: 248056 <|image_pad|>).
    /// Required: set from the model config by the implementing adapter.
    image_token_id: u32,
    /// Preprocess one image and run the ViT tower → features + block + pad
    /// count (owned by the caller).
    processImage: *const fn (ctx: *anyopaque, image: VisionImage) anyerror!VisionImageOut,
    /// Vision prefill: ids carry the image-pad tokens; features/blocks are
    /// the concatenated tower outputs in pad order. Resets backbone state
    /// and installs the MRoPE rope_delta for subsequent decode steps.
    /// Returns last-position logits [1, 1, vocab] (owned).
    prefill: *const fn (ctx: *anyopaque, ids: []const u32, vision: VisionData) anyerror!Array,
};

/// Opaque per-round verify state (model-specific; e.g. Qwen38VerifyOut box).
pub const MtpVerifyHandle = *anyopaque;

/// MTP capability: all model state (backbone caches, drafter module + its KV
/// cache) lives behind `ctx` (the adapter). The engine drives the rounds and
/// does all sampling/acceptance.
pub const MtpOps = struct {
    /// Lazily load the drafter and set the round size. Idempotent; returns
    /// false (with a log) when no drafter checkpoint is available.
    enable: *const fn (ctx: *anyopaque, num_drafts: usize) anyerror!bool,
    /// Prefill the prompt (resets backbone + drafter state) → last-position
    /// logits + hidden.
    prefill: *const fn (ctx: *anyopaque, ids: []const u32) anyerror!MtpStepOut,
    /// Optional vision variant of prefill: backbone MRoPE prefill + drafter
    /// priming, with the drafter's rope offset shifted by the backbone's
    /// rope_delta. Null → MTP cannot serve vision prompts.
    prefillVision: ?*const fn (ctx: *anyopaque, ids: []const u32, vision: VisionData) anyerror!MtpStepOut = null,
    /// One drafter step: (hidden, token) → draft logits + chained hidden.
    draft: *const fn (ctx: *anyopaque, hidden: Array, token: u32) anyerror!MtpStepOut,
    /// Greedy-only GPU-chained drafter step: (hidden, tok_gpu [1] i32) →
    /// argmax token kept on GPU + chained hidden. Null → per-step CPU
    /// sampling fallback.
    draftGreedy: ?*const fn (ctx: *anyopaque, hidden: Array, tok_gpu: Array) anyerror!MtpGreedyDraftOut = null,
    /// Backbone verify over ids (continuation, no reset) → opaque handle.
    verify: *const fn (ctx: *anyopaque, ids: []const u32) anyerror!MtpVerifyHandle,
    /// Logits column i of the verify handle → [vocab] f32 (owned).
    verifyLogits: *const fn (ctx: *anyopaque, handle: MtpVerifyHandle, i: usize) anyerror!Array,
    /// Greedy-only fast path: argmax over ALL verify positions in one GPU op
    /// → i32 [L] (owned, unevaled). One sync reads every row, replacing L
    /// per-row sampler round-trips (each pulls the full vocab to CPU).
    /// Null → the engine falls back to per-row sampling.
    verifyArgmax: ?*const fn (ctx: *anyopaque, handle: MtpVerifyHandle) anyerror!Array = null,
    /// Pre-final-norm hidden row i of the verify handle → [1, hidden] (owned).
    verifyHidden: *const fn (ctx: *anyopaque, handle: MtpVerifyHandle, i: usize) anyerror!Array,
    /// Commit `accepted` ∈ 1..L verify positions (backbone + drafter caches).
    /// `tokens` is the full verify input [t_next, d1..dN] — the drafter cache
    /// is truncated to the round start (+ t_next) and accepted drafts are
    /// REPLAYED with the TARGET's verify hidden states (mlx-vlm
    /// accept_verified_tokens semantics), not kept from the draft chain.
    commit: *const fn (ctx: *anyopaque, handle: MtpVerifyHandle, accepted: usize, tokens: []const u32) void,
    /// Discard the whole verify pass (backbone + drafter caches restored).
    rollback: *const fn (ctx: *anyopaque, handle: MtpVerifyHandle) void,
    /// Free the verify handle (call after commit/rollback).
    freeVerify: *const fn (ctx: *anyopaque, handle: MtpVerifyHandle) void,
    num_drafts: usize,
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
    /// Optional cancellation flag (server: set by the HTTP fiber on client
    /// disconnect). Polled once per decode step; when set, generation
    /// truncates early and returns the tokens produced so far.
    cancel_flag: ?*const std.atomic.Value(bool) = null,
};

/// Whether cancellation has been requested for this generation run.
pub fn cancelRequested(config: GenerateConfig) bool {
    const flag = config.cancel_flag orelse return false;
    return flag.load(.acquire);
}

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
/// Batched generation step. Performs a single forward pass for multiple requests.
///
/// Args:
///   model: The model to run forward on.
///   tokens: Input token IDs, shape [batch, seq_len].
///   caches: KV caches (one per layer), shared across the batch.
///   samplers: One sampler per request in the batch.
///   ctx: Eager context for array operations.
///
/// Returns:
///   A slice of SampleResult, one per request. Caller owns the slice and must free it.
pub fn generateBatchStep(
    model: ModelVTable,
    tokens: Array,
    caches: []KVCacheStrategy,
    samplers: []*SamplerConfig,
    ctx: EagerContext,
) ![]sampling_mod.SampleResult {
    var arena = ScopedArrayArena.init(ctx.allocator);
    defer arena.deinit();

    // Forward pass: [batch, seq_len] → [batch, seq_len, vocab_size]
    const logits = try arena.track(try model.forward(model.ptr, tokens, null, caches));

    const logits_shape = logits.shape();
    const batch = @as(usize, @intCast(logits_shape[0]));
    const seq_len = @as(usize, @intCast(logits_shape[1]));
    const vocab_size = @as(usize, @intCast(logits_shape[2]));

    var results = try ctx.allocator.alloc(sampling_mod.SampleResult, batch);

    for (0..batch) |b| {
        // Extract last token logits for this batch: [1, 1, vocab]
        const last_logits = try arena.track(try ops.slice(
            ctx,
            logits,
            &[_]i32{ @intCast(b), @intCast(seq_len - 1), 0 },
            &[_]i32{ @intCast(b + 1), @intCast(seq_len), @intCast(vocab_size) },
            &[_]i32{ 1, 1, 1 },
        ));

        // Squeeze batch and seq dims: [1, 1, vocab] → [vocab]
        const squeezed = try arena.track(try shape_mod.squeezeAxes(ctx, last_logits, &[_]i32{ 0, 1 }));

        // Cast to float32 for CPU sampling
        const logits_f32 = try arena.track(try ops.astype(ctx, squeezed, .float32));

        // Sample
        results[b] = try samplers[b].sample(logits_f32, ctx.allocator);
    }

    return results;
}

pub fn generateStep(
    model: ModelVTable,
    tokens: Array,
    caches: []KVCacheStrategy,
    sampler: *SamplerConfig,
    ctx: EagerContext,
) !sampling_mod.SampleResult {
    var samplers = [_]*SamplerConfig{sampler};
    const results = try generateBatchStep(model, tokens, caches, &samplers, ctx);
    defer ctx.allocator.free(results);
    return results[0];
}

/// Sample one token from prefill logits [1, L, vocab] or [1, 1, vocab]
/// (last position). Mirrors the tail of generateBatchStep.
fn samplePrefillLogits(logits: Array, sampler: *SamplerConfig, ctx: EagerContext) !sampling_mod.SampleResult {
    var arena = ScopedArrayArena.init(ctx.allocator);
    defer arena.deinit();
    const logits_shape = logits.shape();
    const seq_len: usize = @intCast(logits_shape[1]);
    const vocab_size: usize = @intCast(logits_shape[2]);
    const last_logits = try arena.track(try ops.slice(
        ctx,
        logits,
        &[_]i32{ 0, @intCast(seq_len - 1), 0 },
        &[_]i32{ 1, @intCast(seq_len), @intCast(vocab_size) },
        &[_]i32{ 1, 1, 1 },
    ));
    const squeezed = try arena.track(try shape_mod.squeezeAxes(ctx, last_logits, &[_]i32{ 0, 1 }));
    const logits_f32 = try arena.track(try ops.astype(ctx, squeezed, .float32));
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
            if (cancelRequested(config)) return;
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
    return streamGenerateCtxImpl(model, prompt_tokens, config, caches, ctx, null, ctx_ptr, callback);
}

/// Vision variant of streamGenerateCtx: the prefill runs through the
/// model's VisionOps (MRoPE + feature splice); decode steps are unchanged.
pub fn streamGenerateCtxVision(
    model: ModelVTable,
    prompt_tokens: []const u32,
    config: GenerateConfig,
    caches: []KVCacheStrategy,
    ctx: EagerContext,
    vision: VisionData,
    ctx_ptr: *anyopaque,
    callback: *const fn (ctx_ptr: *anyopaque, token: u32, is_done: bool) void,
) !void {
    return streamGenerateCtxImpl(model, prompt_tokens, config, caches, ctx, vision, ctx_ptr, callback);
}

fn streamGenerateCtxImpl(
    model: ModelVTable,
    prompt_tokens: []const u32,
    config: GenerateConfig,
    caches: []KVCacheStrategy,
    ctx: EagerContext,
    vision: ?VisionData,
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
        var generated_tokens = std.ArrayList(u32).empty;
        defer generated_tokens.deinit(ctx.allocator);

        const first_result = if (vision) |vis| blk: {
            const vops = model.vision orelse return error.VisionNotSupported;
            const prefill_logits = try vops.prefill(model.ptr, prompt_tokens, vis);
            defer prefill_logits.deinit();
            break :blk try samplePrefillLogits(prefill_logits, &sampler, ctx);
        } else blk: {
            const prompt_arr = try Array.fromData(
                ctx.allocator,
                u32,
                prompt_tokens,
                &[_]i32{ 1, @intCast(prompt_tokens.len) },
            );
            defer prompt_arr.deinit();
            break :blk try generateStep(model, prompt_arr, caches, &sampler, ctx);
        };

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
            if (cancelRequested(config)) return;
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
    return generateImpl(model, prompt_tokens, config, caches, ctx, null);
}

/// Vision variant of generate: prefill via the model's VisionOps, then the
/// plain single-token decode loop.
pub fn generateWithVision(
    model: ModelVTable,
    prompt_tokens: []const u32,
    config: GenerateConfig,
    caches: []KVCacheStrategy,
    ctx: EagerContext,
    vision: VisionData,
) ![]u32 {
    return generateImpl(model, prompt_tokens, config, caches, ctx, vision);
}

fn generateImpl(
    model: ModelVTable,
    prompt_tokens: []const u32,
    config: GenerateConfig,
    caches: []KVCacheStrategy,
    ctx: EagerContext,
    vision: ?VisionData,
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
        var generated_tokens = std.ArrayList(u32).empty;
        defer generated_tokens.deinit(ctx.allocator);

        const first_result = if (vision) |vis| blk: {
            const vops = model.vision orelse return error.VisionNotSupported;
            const prefill_logits = try vops.prefill(model.ptr, prompt_tokens, vis);
            defer prefill_logits.deinit();
            break :blk try samplePrefillLogits(prefill_logits, &sampler, ctx);
        } else blk: {
            const prompt_arr = try Array.fromData(
                ctx.allocator,
                u32,
                prompt_tokens,
                &[_]i32{ 1, @intCast(prompt_tokens.len) },
            );
            defer prompt_arr.deinit();
            break :blk try generateStep(model, prompt_arr, caches, &sampler, ctx);
        };
        try result.append(ctx.allocator, first_result.token);
        try generated_tokens.append(ctx.allocator, first_result.token);
        sampler.context_tokens = generated_tokens.items;

        if (isStopToken(first_result.token, config.stop_tokens) or config.max_tokens <= 1) {
            return result.toOwnedSlice(ctx.allocator);
        }

        // Decode loop: feed one token at a time
        var prev_token = first_result.token;

        while (result.items.len < config.max_tokens) {
            if (cancelRequested(config)) break;
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
            if (cancelRequested(config)) return;
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
            if (cancelRequested(config)) return;
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

pub fn isStopToken(token: u32, stop_tokens: []const u32) bool {
    for (stop_tokens) |st| {
        if (token == st) return true;
    }
    return false;
}

// ============================================================
// Tests
// ============================================================

const test_vocab = 8;

fn testForward(_: *anyopaque, input: Array, _: ?Array, _: ?[]KVCacheStrategy) anyerror!Array {
    var data = [_]f32{0} ** test_vocab;
    data[3] = 1.0; // greedy always picks token 3
    const in_shape = input.shape();
    return Array.fromData(std.testing.allocator, f32, &data, &[_]i32{ 1, @intCast(in_shape[1]), test_vocab });
}

fn testDeinit(_: *anyopaque, _: std.mem.Allocator) void {}

fn testModel() ModelVTable {
    return .{
        .forward = &testForward,
        .deinit = &testDeinit,
        .config = .{
            .num_layers = 0,
            .num_kv_heads = 0,
            .head_dim = 0,
            .vocab_size = test_vocab,
            .hidden_size = 0,
        },
        .ptr = @ptrCast(@constCast(&test_model_sentinel)),
    };
}

var test_model_sentinel: u8 = 0;

test "cancelRequested: null flag is false, set flag is true" {
    var config = GenerateConfig{};
    try std.testing.expect(!cancelRequested(config));
    var flag = std.atomic.Value(bool).init(true);
    config.cancel_flag = &flag;
    try std.testing.expect(cancelRequested(config));
    flag.store(false, .release);
    try std.testing.expect(!cancelRequested(config));
}

test "streamGenerateCtx: pre-set cancel flag truncates after the prefill token" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    var caches = [_]KVCacheStrategy{};

    var flag = std.atomic.Value(bool).init(true);
    const config = GenerateConfig{
        .max_tokens = 16,
        .temperature = 0,
        .cancel_flag = &flag,
    };

    const CbCtx = struct {
        calls: usize = 0,
        last_is_done: bool = false,
        fn callback(ptr: *anyopaque, _: u32, is_done: bool) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.calls += 1;
            self.last_is_done = is_done;
        }
    };
    var cb_ctx = CbCtx{};

    try streamGenerateCtx(
        testModel(),
        &[_]u32{1},
        config,
        &caches,
        ctx,
        @ptrCast(&cb_ctx),
        &CbCtx.callback,
    );

    // Prefill token is delivered (non-final), then the decode loop sees the
    // flag and returns without generating further tokens.
    try std.testing.expectEqual(@as(usize, 1), cb_ctx.calls);
    try std.testing.expect(!cb_ctx.last_is_done);
}

test "generate: cancel flag set mid-run truncates the token sequence" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    var caches = [_]KVCacheStrategy{};

    var flag = std.atomic.Value(bool).init(false);
    const config = GenerateConfig{
        .max_tokens = 16,
        .temperature = 0,
        .cancel_flag = &flag,
    };

    const model = testModel();
    // No cancel: full run to max_tokens.
    const full = try generate(model, &[_]u32{1}, config, &caches, ctx);
    defer allocator.free(full);
    try std.testing.expectEqual(@as(usize, 16), full.len);

    // Cancel requested before the call: only the prefill token survives.
    flag.store(true, .release);
    const truncated = try generate(model, &[_]u32{1}, config, &caches, ctx);
    defer allocator.free(truncated);
    try std.testing.expectEqual(@as(usize, 1), truncated.len);
}
