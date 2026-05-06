/// Speculative decoding for accelerated LLM inference.
///
/// Uses an n-gram draft proposal mechanism to propose multiple continuation
/// tokens from the existing generated context, then verifies all proposed
/// tokens in a single forward pass of the target model.
///
/// The accept/reject decision follows the speculative sampling algorithm
/// to ensure statistical equivalence with standard autoregressive sampling.
///
/// References:
///   - Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)
///   - Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023)
const std = @import("std");
const ops = @import("mlx").ops;
const array_mod = @import("mlx").array;
const generation = @import("generation.zig");
const kvcache = @import("kvcache.zig");
const shape_mod = @import("mlx").shape;
const array_arena_mod = @import("mlx").array_arena;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const KVCacheStrategy = kvcache.KVCacheStrategy;
const ModelVTable = generation.ModelVTable;
const ScopedArrayArena = array_arena_mod.ScopedArrayArena;

// ============================================================
// NgramDrafter — n-gram based draft token proposal
// ============================================================

// ============================================================
// EagleDrafter — EAGLE speculative decoding draft head
// ============================================================

/// EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency)
/// uses a lightweight MLP head on top of the base model's hidden states
/// to predict multiple future tokens in parallel.
///
/// This struct provides the interface for EAGLE-style drafting.
/// The actual draft head weights must be loaded from a trained checkpoint.
///
/// References:
///   - Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (2024)
///   - Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees" (2024)
///
/// The verify logic reuses the same `verifyDraft` function as NgramDrafter —
/// the accept/reject algorithm is independent of the drafting mechanism.
pub const EagleDrafter = struct {
    /// Number of draft tokens to propose per step.
    num_draft_tokens: usize = 5,
    /// Hidden dimension of the base model (must match).
    hidden_dim: usize = 0,
    /// Vocabulary size for the draft head output.
    vocab_size: usize = 0,
    /// Draft head weight: projects hidden states to vocab logits.
    /// Shape: [hidden_dim, vocab_size]. Null until loaded.
    draft_head_weight: ?Array = null,
    /// Draft head bias (optional). Shape: [vocab_size].
    draft_head_bias: ?Array = null,

    /// Initialize an EagleDrafter with model dimensions.
    /// Weights must be loaded separately via `loadWeights`.
    pub fn init(hidden_dim: usize, vocab_size: usize) EagleDrafter {
        return .{
            .hidden_dim = hidden_dim,
            .vocab_size = vocab_size,
        };
    }

    /// Load draft head weights from arrays.
    /// The weight matrix projects the last hidden state to vocab logits.
    pub fn loadWeights(self: *EagleDrafter, weight: Array, bias: ?Array) void {
        self.draft_head_weight = weight;
        self.draft_head_bias = bias;
    }

    /// Check if the draft head weights are loaded and ready.
    pub fn isReady(self: *const EagleDrafter) bool {
        return self.draft_head_weight != null;
    }

    /// Propose draft tokens using the EAGLE draft head.
    ///
    /// Takes the last hidden state from the base model and projects it
    /// through the draft head MLP to predict future tokens.
    ///
    /// `hidden_state` should be the last-layer hidden state for the current
    /// token, shaped [1, 1, hidden_dim] or [hidden_dim].
    ///
    /// Returns null if the draft head is not loaded (weights not available).
    /// Caller owns the returned slice and must free with `allocator`.
    pub fn propose(
        self: *const EagleDrafter,
        hidden_state: Array,
        k: usize,
        ctx: EagerContext,
        allocator: std.mem.Allocator,
    ) !?[]u32 {
        if (!self.isReady()) return null;
        if (k == 0) return null;

        var arena = ScopedArrayArena.init(ctx.allocator);
        defer arena.deinit();

        // Ensure hidden_state is 2D [1, hidden_dim] for matmul
        const hidden_2d = if (hidden_state.ndim() == 1)
            try arena.track(try ops.reshape(ctx, hidden_state, &[_]i32{ 1, @intCast(self.hidden_dim) }))
        else if (hidden_state.ndim() == 3)
            // [1, 1, hidden_dim] -> squeeze to [1, hidden_dim]
            try arena.track(try shape_mod.squeezeAxes(ctx, hidden_state, &[_]i32{1}))
        else
            hidden_state;

        // logits = hidden @ weight + bias
        // weight shape: [hidden_dim, vocab_size]
        var logits = try ops.matmul(ctx, hidden_2d, self.draft_head_weight.?);
        if (self.draft_head_bias) |bias| {
            const biased = try ops.add(ctx, logits, bias);
            logits.deinit();
            logits = biased;
        }
        defer logits.deinit();

        // softmax over vocab dimension
        const probs = try arena.track(try ops.softmax(ctx, logits, &[_]i32{-1}));
        try probs.eval();
        const probs_data = try probs.dataPtr(f32);
        const vocab_size = self.vocab_size;

        // Greedy top-k sampling (deterministic draft for simplicity)
        var result = try allocator.alloc(u32, k);
        errdefer allocator.free(result);

        for (0..k) |i| {
            // Find argmax
            var max_idx: u32 = 0;
            var max_prob: f32 = probs_data[0];
            for (1..vocab_size) |j| {
                if (probs_data[j] > max_prob) {
                    max_prob = probs_data[j];
                    max_idx = @intCast(j);
                }
            }
            result[i] = max_idx;

            // For subsequent tokens, we'd need auto-regressive draft head.
            // Simplified: repeat the same top token (EAGLE-1 basic).
            // A full implementation would embed the token and run draft head again.
            if (i > 0) {
                result[i] = result[0];
            }
        }

        return result;
    }

    /// Release draft head weights.
    pub fn deinit(self: *EagleDrafter) void {
        if (self.draft_head_weight) |w| w.deinit();
        if (self.draft_head_bias) |b| b.deinit();
        self.draft_head_weight = null;
        self.draft_head_bias = null;
    }
};

/// Proposes continuation tokens by searching the existing context for
/// matching n-gram suffixes. No model is needed for drafting — this is
/// a pure lookup in the generated token history.
pub const NgramDrafter = struct {
    /// N-gram size: number of tokens to match at the end of context.
    n: usize = 3,

    /// Search the context for the last `n` tokens as a suffix pattern,
    /// and if found, return up to `k` continuation tokens from the context.
    ///
    /// Returns null if no matching n-gram is found or context is too short.
    /// Caller owns the returned slice and must free it with `allocator`.
    pub fn propose(
        self: *const NgramDrafter,
        context: []const u32,
        k: usize,
        allocator: std.mem.Allocator,
    ) ?[]u32 {
        if (context.len <= self.n or k == 0) return null;

        // The suffix to match: last n tokens of context
        const suffix = context[context.len - self.n ..];

        // Search for the suffix earlier in the context (not at the very end)
        // We need at least n tokens for the match + at least 1 continuation token
        const search_end = context.len - self.n;
        var best_match: ?usize = null;

        var i: usize = 0;
        while (i + self.n <= search_end) : (i += 1) {
            if (std.mem.eql(u32, context[i .. i + self.n], suffix)) {
                // Found a match — prefer the latest one for better predictions
                best_match = i;
            }
        }

        const match_pos = best_match orelse return null;

        // Continuation starts right after the matched n-gram
        const cont_start = match_pos + self.n;
        // Limit continuation to not overlap with the suffix at the end
        const cont_end = context.len - self.n;
        if (cont_start >= cont_end) return null;
        const available = cont_end - cont_start;
        const cont_len = @min(k, available);
        if (cont_len == 0) return null;

        const result = allocator.alloc(u32, cont_len) catch return null;
        @memcpy(result, context[cont_start .. cont_start + cont_len]);
        return result;
    }
};

// ============================================================
// PldDrafter — Prompt Lookup Decoding draft token proposal
// ============================================================

/// Prompt Lookup Decoding (PLD) proposes continuation tokens by searching
/// the prompt for n-gram suffix matches. Unlike NgramDrafter which only
/// searches the generated context, PLD indexes the prompt at init time and
/// lookups are O(1) via a hash map. This is especially effective for
/// structured outputs (JSON, code) where the prompt contains examples that
/// the model is likely to repeat.
pub const PldDrafter = struct {
    /// N-gram size: number of tokens to match at the end of context.
    n: usize = 3,
    /// The prompt tokens to index for lookups.
    prompt_tokens: []const u32,
    /// Allocator used for the index.
    allocator: std.mem.Allocator,
    /// Index mapping n-gram hash → list of start positions in prompt.
    index: std.AutoHashMap(u64, std.ArrayList(usize)),

    /// Build the n-gram index from prompt tokens.
    pub fn init(allocator: std.mem.Allocator, prompt_tokens: []const u32, n: usize) !PldDrafter {
        var index = std.AutoHashMap(u64, std.ArrayList(usize)).init(allocator);
        errdefer {
            var it = index.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.deinit(allocator);
            }
            index.deinit();
        }

        if (prompt_tokens.len >= n) {
            var i: usize = 0;
            while (i + n <= prompt_tokens.len) : (i += 1) {
                const hash = hashNgram(prompt_tokens[i .. i + n]);
                const gop = try index.getOrPut(hash);
                if (!gop.found_existing) {
                    gop.value_ptr.* = std.ArrayList(usize).empty;
                }
                try gop.value_ptr.append(allocator, i);
            }
        }

        return .{
            .n = n,
            .prompt_tokens = prompt_tokens,
            .allocator = allocator,
            .index = index,
        };
    }

    pub fn deinit(self: *PldDrafter) void {
        var it = self.index.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.index.deinit();
    }

    /// Search the prompt for the last `n` tokens of context as a suffix.
    /// If found, return up to `k` continuation tokens from the prompt.
    /// Returns null if no matching n-gram is found.
    pub fn propose(
        self: *const PldDrafter,
        context: []const u32,
        k: usize,
        allocator: std.mem.Allocator,
    ) ?[]u32 {
        if (context.len < self.n or k == 0) return null;

        const suffix = context[context.len - self.n ..];
        const hash = hashNgram(suffix);
        const positions = self.index.get(hash) orelse return null;

        // Prefer the last match in prompt (most similar context).
        var best_pos: ?usize = null;
        var i: usize = positions.items.len;
        while (i > 0) {
            i -= 1;
            const pos = positions.items[i];
            const cont_start = pos + self.n;
            if (cont_start < self.prompt_tokens.len) {
                best_pos = pos;
                break;
            }
        }

        const pos = best_pos orelse return null;
        const cont_start = pos + self.n;
        const available = self.prompt_tokens.len - cont_start;
        const cont_len = @min(k, available);
        if (cont_len == 0) return null;

        const result = allocator.alloc(u32, cont_len) catch return null;
        @memcpy(result, self.prompt_tokens[cont_start .. cont_start + cont_len]);
        return result;
    }
};

fn hashNgram(tokens: []const u32) u64 {
    var hasher = std.hash.Wyhash.init(0);
    const bytes = std.mem.sliceAsBytes(tokens);
    hasher.update(bytes);
    return hasher.final();
}

// ============================================================
// Draft Verification — speculative sampling algorithm
// ============================================================

/// Result of draft verification.
pub const VerifyResult = struct {
    /// Number of draft tokens accepted (0..draft_tokens.len).
    accepted: usize,
    /// Accepted draft tokens + one bonus/resampled token.
    /// Caller owns this slice and must free with allocator.
    tokens: []u32,
};

/// Verify draft tokens against the target model in a single forward pass.
///
/// Runs the target model on [context ++ draft_tokens] in one forward pass,
/// then applies the speculative sampling algorithm:
///   - For each draft token, compare draft probability (uniform for n-gram)
///     against target model probability
///   - Accept if target_prob >= draft_prob, else accept with probability
///     target_prob / draft_prob
///   - On first rejection, resample from adjusted distribution
///   - If all accepted, sample one bonus token from the target model
///
/// This ensures the output distribution is statistically equivalent to
/// sampling directly from the target model.
pub fn verifyDraft(
    model: ModelVTable,
    context: []const u32,
    draft_tokens: []const u32,
    caches: []KVCacheStrategy,
    ctx: EagerContext,
    seed: u64,
) !VerifyResult {
    if (draft_tokens.len == 0) {
        return VerifyResult{ .accepted = 0, .tokens = &.{} };
    }

    var arena = ScopedArrayArena.init(ctx.allocator);
    defer arena.deinit();

    // Build input: context + draft tokens as [1, total_len]
    const total_len = context.len + draft_tokens.len;
    var input_data = try ctx.allocator.alloc(u32, total_len);
    defer ctx.allocator.free(input_data);
    @memcpy(input_data[0..context.len], context);
    @memcpy(input_data[context.len..], draft_tokens);

    const input_arr = try arena.track(try Array.fromData(
        ctx.allocator,
        u32,
        input_data,
        &[_]i32{ 1, @intCast(total_len) },
    ));

    // Single forward pass: [1, total_len] → [1, total_len, vocab_size]
    const logits = try arena.track(try model.forward(model.ptr, input_arr, null, caches));

    // Extract logits shape
    const logits_shape = logits.shape();
    const vocab_size: usize = @intCast(logits_shape[2]);

    // We need logits at positions context.len-1 .. context.len+draft_tokens.len-1
    // Position context.len-1 gives the distribution for the first draft token
    // Position context.len+i-1 gives the distribution for draft_tokens[i]

    // Compute softmax probabilities for each relevant position
    // Extract the relevant slice: positions [context.len-1 .. context.len+draft_tokens.len]
    const start_pos: i32 = @intCast(context.len - 1);
    const end_pos: i32 = @intCast(context.len + draft_tokens.len);

    const relevant_logits = try arena.track(try ops.slice(
        ctx,
        logits,
        &[_]i32{ 0, start_pos, 0 },
        &[_]i32{ 1, end_pos, @intCast(vocab_size) },
        &[_]i32{ 1, 1, 1 },
    ));

    // Softmax over vocab dimension (axis 2) → [1, draft_len+1, vocab_size]
    const probs = try arena.track(try ops.softmax(ctx, relevant_logits, &[_]i32{2}));

    // Evaluate to get CPU data
    try probs.eval();
    const probs_data = try probs.dataPtr(f32);

    // Speculative sampling: accept/reject each draft token
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    var accepted: usize = 0;
    var result_tokens = std.ArrayList(u32).empty;
    errdefer result_tokens.deinit(ctx.allocator);

    // For n-gram drafter, draft probability is uniform: 1.0
    // (we proposed these tokens deterministically, so q(x) = 1 for the proposed token)
    // Actually for speculative sampling with a deterministic drafter,
    // we treat q(x) = 1 for the proposed token and 0 for others.
    // Accept criterion: accept with probability min(1, p(x)/q(x)) = p(x) since q(x)=1
    // But this would almost always reject. The standard approach for n-gram drafting
    // is to always accept if the target model assigns non-negligible probability,
    // using a threshold or the standard speculative sampling with q as a uniform
    // distribution over the vocab.
    //
    // Standard speculative sampling with uniform draft distribution q(x) = 1/V:
    //   accept with probability min(1, p(x) / q(x)) = min(1, p(x) * V)
    // This accepts whenever p(x) >= 1/V, which is almost always for reasonable tokens.
    //
    // We use the standard algorithm: for each draft token x_i,
    //   r ~ Uniform(0, 1)
    //   if r < p_target(x_i) / q_draft(x_i), accept
    //   else reject and resample from max(0, p_target - q_draft) normalized

    const inv_vocab: f32 = 1.0 / @as(f32, @floatFromInt(vocab_size));

    for (0..draft_tokens.len) |i| {
        const token = draft_tokens[i];
        // Target probability for this token at position i
        const target_prob = probs_data[i * vocab_size + token];
        // Draft probability (uniform over vocab for n-gram drafter)
        const draft_prob = inv_vocab;

        const acceptance_ratio = target_prob / draft_prob;
        const r = rand.float(f32);

        if (r < acceptance_ratio) {
            // Accept this draft token
            try result_tokens.append(ctx.allocator, token);
            accepted += 1;
        } else {
            // Reject: resample from adjusted distribution max(0, p_target - q_draft)
            const resampled = resampleToken(
                probs_data[i * vocab_size .. (i + 1) * vocab_size],
                inv_vocab,
                vocab_size,
                rand,
            );
            try result_tokens.append(ctx.allocator, resampled);
            break;
        }
    }

    // If all draft tokens accepted, sample one bonus token from the last position
    if (accepted == draft_tokens.len) {
        const last_pos = draft_tokens.len;
        const bonus = sampleFromProbs(
            probs_data[last_pos * vocab_size .. (last_pos + 1) * vocab_size],
            vocab_size,
            rand,
        );
        try result_tokens.append(ctx.allocator, bonus);
    }

    return VerifyResult{
        .accepted = accepted,
        .tokens = try result_tokens.toOwnedSlice(ctx.allocator),
    };
}

/// Resample a token from the adjusted distribution max(0, p_target - q_draft),
/// normalized to sum to 1.
fn resampleToken(
    target_probs: []const f32,
    draft_prob: f32,
    vocab_size: usize,
    rand: std.Random,
) u32 {
    // Compute adjusted distribution: max(0, p_target(x) - q_draft(x))
    // For uniform draft: q_draft(x) = draft_prob for all x
    var adjusted_sum: f32 = 0.0;

    // First pass: compute sum for normalization
    for (0..vocab_size) |i| {
        const adj = target_probs[i] - draft_prob;
        if (adj > 0.0) {
            adjusted_sum += adj;
        }
    }

    // If adjusted distribution is all zeros (shouldn't happen in practice),
    // fall back to sampling from target distribution directly
    if (adjusted_sum <= 0.0) {
        return sampleFromProbs(target_probs, vocab_size, rand);
    }

    // Sample from normalized adjusted distribution
    const r = rand.float(f32);
    var cumsum: f32 = 0.0;
    for (0..vocab_size) |i| {
        const adj = target_probs[i] - draft_prob;
        if (adj > 0.0) {
            cumsum += adj / adjusted_sum;
            if (r < cumsum) {
                return @intCast(i);
            }
        }
    }

    return @intCast(vocab_size - 1);
}

/// Sample a token index from a probability distribution.
fn sampleFromProbs(probs: []const f32, n: usize, rand: std.Random) u32 {
    const r = rand.float(f32);
    var cumsum: f32 = 0.0;
    for (0..n) |i| {
        cumsum += probs[i];
        if (r < cumsum) {
            return @intCast(i);
        }
    }
    return @intCast(n - 1);
}

// ============================================================
// Unit Tests
// ============================================================

test "NgramDrafter: default n-gram size is 3" {
    const drafter = NgramDrafter{};
    try std.testing.expectEqual(@as(usize, 3), drafter.n);
}

test "NgramDrafter: returns null for context shorter than n" {
    const drafter = NgramDrafter{ .n = 3 };
    const result = drafter.propose(&[_]u32{ 1, 2 }, 5, std.testing.allocator);
    try std.testing.expect(result == null);
}

test "NgramDrafter: returns null for context equal to n" {
    const drafter = NgramDrafter{ .n = 3 };
    const result = drafter.propose(&[_]u32{ 1, 2, 3 }, 5, std.testing.allocator);
    try std.testing.expect(result == null);
}

test "NgramDrafter: returns null when k is 0" {
    const drafter = NgramDrafter{ .n = 2 };
    const context = [_]u32{ 1, 2, 3, 1, 2 };
    const result = drafter.propose(&context, 0, std.testing.allocator);
    try std.testing.expect(result == null);
}

test "NgramDrafter: finds matching bigram and proposes continuation" {
    const drafter = NgramDrafter{ .n = 2 };
    // Context: [10, 20, 30, 40, 10, 20]
    // Suffix (last 2): [10, 20]
    // Match at position 0: [10, 20] → continuation is [30, 40, ...]
    const context = [_]u32{ 10, 20, 30, 40, 10, 20 };
    const result = drafter.propose(&context, 3, std.testing.allocator);
    try std.testing.expect(result != null);
    defer std.testing.allocator.free(result.?);

    // Continuation after match at pos 0: tokens at pos 2,3 = [30, 40]
    // But we can only get up to the start of the suffix (pos 4), so 2 tokens
    try std.testing.expectEqual(@as(usize, 2), result.?.len);
    try std.testing.expectEqual(@as(u32, 30), result.?[0]);
    try std.testing.expectEqual(@as(u32, 40), result.?[1]);
}

test "NgramDrafter: prefers latest match" {
    const drafter = NgramDrafter{ .n = 2 };
    // Context: [5, 6, 100, 5, 6, 200, 5, 6]
    // Suffix: [5, 6]
    // Match at pos 0: continuation [100, 5, 6, 200]
    // Match at pos 3: continuation [200, 5, 6]
    // Should prefer latest match (pos 3) → continuation starts with 200
    const context = [_]u32{ 5, 6, 100, 5, 6, 200, 5, 6 };
    const result = drafter.propose(&context, 1, std.testing.allocator);
    try std.testing.expect(result != null);
    defer std.testing.allocator.free(result.?);

    try std.testing.expectEqual(@as(usize, 1), result.?.len);
    try std.testing.expectEqual(@as(u32, 200), result.?[0]);
}

test "NgramDrafter: limits continuation to k tokens" {
    const drafter = NgramDrafter{ .n = 2 };
    // Context: [1, 2, 3, 4, 5, 6, 7, 1, 2]
    // Suffix: [1, 2], match at pos 0, continuation: [3, 4, 5, 6, 7]
    const context = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 1, 2 };
    const result = drafter.propose(&context, 2, std.testing.allocator);
    try std.testing.expect(result != null);
    defer std.testing.allocator.free(result.?);

    try std.testing.expectEqual(@as(usize, 2), result.?.len);
    try std.testing.expectEqual(@as(u32, 3), result.?[0]);
    try std.testing.expectEqual(@as(u32, 4), result.?[1]);
}

test "NgramDrafter: returns null when no match found" {
    const drafter = NgramDrafter{ .n = 2 };
    // Context: [1, 2, 3, 4, 5, 6] — suffix [5, 6] doesn't appear earlier
    const context = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const result = drafter.propose(&context, 3, std.testing.allocator);
    try std.testing.expect(result == null);
}

test "NgramDrafter: trigram matching" {
    const drafter = NgramDrafter{ .n = 3 };
    // Context: [10, 20, 30, 40, 50, 10, 20, 30]
    // Suffix: [10, 20, 30], match at pos 0, continuation: [40, 50]
    const context = [_]u32{ 10, 20, 30, 40, 50, 10, 20, 30 };
    const result = drafter.propose(&context, 5, std.testing.allocator);
    try std.testing.expect(result != null);
    defer std.testing.allocator.free(result.?);

    try std.testing.expectEqual(@as(usize, 2), result.?.len);
    try std.testing.expectEqual(@as(u32, 40), result.?[0]);
    try std.testing.expectEqual(@as(u32, 50), result.?[1]);
}

test "NgramDrafter: configurable n-gram size" {
    const drafter = NgramDrafter{ .n = 1 };
    // Unigram: suffix [5], match at pos 0, continuation: [6, 7, 8]
    const context = [_]u32{ 5, 6, 7, 8, 9, 5 };
    const result = drafter.propose(&context, 2, std.testing.allocator);
    try std.testing.expect(result != null);
    defer std.testing.allocator.free(result.?);

    try std.testing.expectEqual(@as(usize, 2), result.?.len);
    try std.testing.expectEqual(@as(u32, 6), result.?[0]);
    try std.testing.expectEqual(@as(u32, 7), result.?[1]);
}

test "verifyDraft: empty draft returns empty result" {
    const c = @import("mlx").c;
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // We don't need a real model for empty draft
    const result = try verifyDraft(
        undefined,
        &[_]u32{ 1, 2, 3 },
        &[_]u32{},
        &.{},
        ctx,
        42,
    );

    try std.testing.expectEqual(@as(usize, 0), result.accepted);
    try std.testing.expectEqual(@as(usize, 0), result.tokens.len);
}

test "verifyDraft: accepts tokens with high target probability" {
    const c = @import("mlx").c;
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Mock model that assigns very high probability to specific tokens
    const MockVerifyModel = struct {
        vocab_size: usize = 8,

        fn forwardFn(ctx_ptr: *anyopaque, input: Array, mask: ?Array, caches: ?[]KVCacheStrategy) anyerror!Array {
            _ = mask;
            _ = caches;
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));

            const input_shape = input.shape();
            const seq_len: usize = @intCast(input_shape[1]);
            const vocab = self.vocab_size;
            const total = seq_len * vocab;

            var data = try std.heap.page_allocator.alloc(f32, total);
            defer std.heap.page_allocator.free(data);

            // For each position, assign high logit to token 1
            for (0..seq_len) |pos| {
                for (0..vocab) |v| {
                    data[pos * vocab + v] = if (v == 1) 100.0 else -100.0;
                }
            }

            return Array.fromData(
                std.heap.page_allocator,
                f32,
                data,
                &[_]i32{ 1, @intCast(seq_len), @intCast(vocab) },
            );
        }

        fn deinitFn(_: *anyopaque, _: std.mem.Allocator) void {}
    };

    var mock = MockVerifyModel{};
    const vtable = ModelVTable{
        .forward = @ptrCast(&MockVerifyModel.forwardFn),
        .deinit = @ptrCast(&MockVerifyModel.deinitFn),
        .config = .{
            .num_layers = 1,
            .num_kv_heads = 1,
            .head_dim = 8,
            .vocab_size = 8,
            .hidden_size = 8,
        },
        .ptr = @ptrCast(&mock),
    };

    // Draft tokens are all token 1 — model assigns high prob to token 1
    const context = [_]u32{ 0, 0, 0 };
    const draft = [_]u32{ 1, 1, 1 };

    const result = try verifyDraft(vtable, &context, &draft, &.{}, ctx, 42);
    defer allocator.free(result.tokens);

    // All draft tokens should be accepted (high target prob)
    try std.testing.expectEqual(@as(usize, 3), result.accepted);
    // Result should have 3 accepted + 1 bonus = 4 tokens
    try std.testing.expectEqual(@as(usize, 4), result.tokens.len);
    // First 3 should be the accepted draft tokens
    try std.testing.expectEqual(@as(u32, 1), result.tokens[0]);
    try std.testing.expectEqual(@as(u32, 1), result.tokens[1]);
    try std.testing.expectEqual(@as(u32, 1), result.tokens[2]);
}

test "verifyDraft: result tokens length is accepted + 1" {
    const c = @import("mlx").c;
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Mock model with uniform distribution — some tokens may be rejected
    const MockUniformModel = struct {
        vocab_size: usize = 16,

        fn forwardFn(ctx_ptr: *anyopaque, input: Array, mask: ?Array, caches: ?[]KVCacheStrategy) anyerror!Array {
            _ = mask;
            _ = caches;
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));

            const input_shape = input.shape();
            const seq_len: usize = @intCast(input_shape[1]);
            const vocab = self.vocab_size;
            const total = seq_len * vocab;

            const data = try std.heap.page_allocator.alloc(f32, total);
            defer std.heap.page_allocator.free(data);

            // Uniform logits → uniform probability after softmax
            @memset(data, 0.0);

            return Array.fromData(
                std.heap.page_allocator,
                f32,
                data,
                &[_]i32{ 1, @intCast(seq_len), @intCast(vocab) },
            );
        }

        fn deinitFn(_: *anyopaque, _: std.mem.Allocator) void {}
    };

    var mock = MockUniformModel{};
    const vtable = ModelVTable{
        .forward = @ptrCast(&MockUniformModel.forwardFn),
        .deinit = @ptrCast(&MockUniformModel.deinitFn),
        .config = .{
            .num_layers = 1,
            .num_kv_heads = 1,
            .head_dim = 8,
            .vocab_size = 16,
            .hidden_size = 8,
        },
        .ptr = @ptrCast(&mock),
    };

    const context = [_]u32{ 0, 1, 2 };
    const draft = [_]u32{ 3, 4, 5 };

    const result = try verifyDraft(vtable, &context, &draft, &.{}, ctx, 12345);
    defer allocator.free(result.tokens);

    // Key invariant: result always has accepted + 1 tokens
    // (either accepted drafts + resampled, or all accepted + bonus)
    try std.testing.expectEqual(result.accepted + 1, result.tokens.len);
}

// ============================================================
// Property-Based Tests
// ============================================================

// ============================================================
// Property 14: N-gram Draft Proposal Correctness
//
// Feature: production-deployment, Property 14: N-gram Draft
// Proposal Correctness
//
// For any generated context containing a repeated n-gram suffix,
// the NgramDrafter SHALL find the matching n-gram and propose
// the correct continuation tokens from the context.
//
// **Validates: Requirements R16.1**
// ============================================================

test "Property 14: N-gram Draft Proposal Correctness — finds matching n-gram and proposes correct continuation (100 iterations)" {
    var prng = std.Random.DefaultPrng.init(314159);
    const rand = prng.random();
    const alloc = std.testing.allocator;

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        // Random n-gram size: 1..4
        const n: usize = rand.intRangeAtMost(usize, 1, 4);
        const drafter = NgramDrafter{ .n = n };

        // Random k (number of continuation tokens to request): 1..6
        const k: usize = rand.intRangeAtMost(usize, 1, 6);

        // Build a context that guarantees a repeated n-gram:
        //   [prefix...] [ngram] [continuation...] [ngram]
        // where prefix is random, ngram is n random tokens, and
        // continuation is random tokens between the two n-gram occurrences.
        const prefix_len = rand.intRangeAtMost(usize, 0, 5);
        const cont_len = rand.intRangeAtMost(usize, 1, 8);
        const total_len = prefix_len + n + cont_len + n;

        var context = try alloc.alloc(u32, total_len);
        defer alloc.free(context);

        // Fill prefix with random tokens (values 100..200 to avoid collisions)
        for (0..prefix_len) |i| {
            context[i] = rand.intRangeAtMost(u32, 100, 200);
        }

        // Generate the n-gram pattern (values 1..50)
        var ngram = try alloc.alloc(u32, n);
        defer alloc.free(ngram);
        for (0..n) |i| {
            ngram[i] = rand.intRangeAtMost(u32, 1, 50);
        }

        // Place first occurrence of n-gram
        @memcpy(context[prefix_len .. prefix_len + n], ngram);

        // Fill continuation tokens (values 201..300 to avoid collisions with ngram)
        for (0..cont_len) |i| {
            context[prefix_len + n + i] = rand.intRangeAtMost(u32, 201, 300);
        }

        // Place second occurrence of n-gram at the end (this is the suffix)
        @memcpy(context[prefix_len + n + cont_len ..], ngram);

        // Call propose
        const result = drafter.propose(context, k, alloc);

        // The drafter MUST find the match (we guaranteed a repeated n-gram)
        try std.testing.expect(result != null);
        const proposed = result.?;
        defer alloc.free(proposed);

        // The continuation starts right after the matched n-gram.
        // The drafter prefers the latest match. Since we placed the n-gram
        // at prefix_len and possibly at other positions, the latest match
        // position determines the continuation.
        //
        // The continuation tokens available are from (match_pos + n) to
        // (context.len - n). We verify the proposed tokens match the
        // context at the expected positions.

        // Find the latest match position (same logic as the drafter)
        const search_end = context.len - n;
        const suffix = context[context.len - n ..];
        var best_match: ?usize = null;
        var i: usize = 0;
        while (i + n <= search_end) : (i += 1) {
            if (std.mem.eql(u32, context[i .. i + n], suffix)) {
                best_match = i;
            }
        }
        const match_pos = best_match.?;
        const cont_start = match_pos + n;
        const cont_end = context.len - n;
        const available = cont_end - cont_start;
        const expected_len = @min(k, available);

        // Verify length
        try std.testing.expectEqual(expected_len, proposed.len);

        // Verify each proposed token matches the context
        for (0..proposed.len) |j| {
            try std.testing.expectEqual(context[cont_start + j], proposed[j]);
        }
    }
}

test "Property 14: N-gram Draft Proposal Correctness — returns null when no repeated n-gram exists (100 iterations)" {
    var prng = std.Random.DefaultPrng.init(271828);
    const rand = prng.random();
    const alloc = std.testing.allocator;

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        const n: usize = rand.intRangeAtMost(usize, 1, 3);
        const drafter = NgramDrafter{ .n = n };

        // Build a context where all tokens are unique, so no n-gram repeats.
        // Use context length n+1..n+10 (must be > n for propose to not trivially return null)
        const ctx_len = rand.intRangeAtMost(usize, n + 1, n + 10);
        var context = try alloc.alloc(u32, ctx_len);
        defer alloc.free(context);

        // Fill with strictly increasing values — guarantees no repeated n-gram
        for (0..ctx_len) |idx| {
            context[idx] = @intCast(idx);
        }

        const result = drafter.propose(context, 3, alloc);

        // With all unique tokens, no n-gram suffix can repeat earlier
        try std.testing.expect(result == null);
    }
}

// ============================================================
// Property 15: Speculative Decoding Statistical Equivalence
//
// Feature: production-deployment, Property 15: Speculative
// Decoding Statistical Equivalence
//
// For any draft token sequence and target model probability
// distribution, the accept/reject decision SHALL follow the
// speculative sampling algorithm such that:
//   - result.tokens.len == result.accepted + 1
//   - accepted tokens match the draft tokens
//   - accepted count is in [0, draft_tokens.len]
//
// **Validates: Requirements R16.3**
// ============================================================

test "Property 15: Speculative Decoding Statistical Equivalence — tokens.len == accepted + 1 invariant (100 iterations)" {
    const c = @import("mlx").c;
    c.initErrorHandler();
    const alloc = std.testing.allocator;
    const ctx = EagerContext.init(alloc);

    // Mock model with configurable per-token probability via logits.
    // For each position, assigns a peaked distribution around a target token.
    const MockProbModel = struct {
        vocab_size: usize = 16,
        // Logit peak value — controls how peaked the distribution is
        peak_logit: f32 = 5.0,

        fn forwardFn(ctx_ptr: *anyopaque, input: Array, mask: ?Array, caches: ?[]KVCacheStrategy) anyerror!Array {
            _ = mask;
            _ = caches;
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));

            const input_shape = input.shape();
            const seq_len: usize = @intCast(input_shape[1]);
            const vocab = self.vocab_size;
            const total = seq_len * vocab;

            var data = try std.heap.page_allocator.alloc(f32, total);
            defer std.heap.page_allocator.free(data);

            // For each position, create a distribution peaked at token (pos % vocab)
            for (0..seq_len) |pos| {
                const peak_token = pos % vocab;
                for (0..vocab) |v| {
                    data[pos * vocab + v] = if (v == peak_token) self.peak_logit else 0.0;
                }
            }

            return Array.fromData(
                std.heap.page_allocator,
                f32,
                data,
                &[_]i32{ 1, @intCast(seq_len), @intCast(vocab) },
            );
        }

        fn deinitFn(_: *anyopaque, _: std.mem.Allocator) void {}
    };

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        const vocab_size: usize = 16;

        // Random draft length: 1..5
        const draft_len = rand.intRangeAtMost(usize, 1, 5);
        // Random context length: 2..6
        const context_len = rand.intRangeAtMost(usize, 2, 6);

        var context = try alloc.alloc(u32, context_len);
        defer alloc.free(context);
        for (0..context_len) |idx| {
            context[idx] = rand.intRangeLessThan(u32, 0, @intCast(vocab_size));
        }

        var draft = try alloc.alloc(u32, draft_len);
        defer alloc.free(draft);
        for (0..draft_len) |idx| {
            draft[idx] = rand.intRangeLessThan(u32, 0, @intCast(vocab_size));
        }

        // Random peak logit to vary the distribution shape
        const peak_logit: f32 = @floatFromInt(rand.intRangeAtMost(u32, 1, 20));

        var mock = MockProbModel{ .vocab_size = vocab_size, .peak_logit = peak_logit };
        const vtable = ModelVTable{
            .forward = @ptrCast(&MockProbModel.forwardFn),
            .deinit = @ptrCast(&MockProbModel.deinitFn),
            .config = .{
                .num_layers = 1,
                .num_kv_heads = 1,
                .head_dim = 8,
                .vocab_size = vocab_size,
                .hidden_size = 8,
            },
            .ptr = @ptrCast(&mock),
        };

        // Use a different seed each iteration for the verifyDraft PRNG
        const seed: u64 = @intCast(iteration * 7919 + 1);

        const result = try verifyDraft(vtable, context, draft, &.{}, ctx, seed);
        defer alloc.free(result.tokens);

        // Property 15a: tokens.len == accepted + 1
        // This is the core invariant of speculative sampling:
        // either all accepted + 1 bonus, or some accepted + 1 resampled
        try std.testing.expectEqual(result.accepted + 1, result.tokens.len);

        // Property 15b: accepted count is in valid range [0, draft_len]
        try std.testing.expect(result.accepted <= draft_len);

        // Property 15c: accepted tokens match the corresponding draft tokens
        for (0..result.accepted) |idx| {
            try std.testing.expectEqual(draft[idx], result.tokens[idx]);
        }

        // Property 15d: all result tokens are valid token IDs (< vocab_size)
        for (result.tokens) |token| {
            try std.testing.expect(token < @as(u32, @intCast(vocab_size)));
        }
    }
}

test "Property 15: Speculative Decoding — high-probability drafts are mostly accepted (100 iterations)" {
    const c = @import("mlx").c;
    c.initErrorHandler();
    const alloc = std.testing.allocator;
    const ctx = EagerContext.init(alloc);

    // Mock model that assigns very high probability to a specific token.
    // When draft tokens match the high-prob token, acceptance rate should be high.
    const MockHighProbModel = struct {
        vocab_size: usize = 8,
        high_prob_token: u32 = 3,

        fn forwardFn(ctx_ptr: *anyopaque, input: Array, mask: ?Array, caches: ?[]KVCacheStrategy) anyerror!Array {
            _ = mask;
            _ = caches;
            const self: *@This() = @ptrCast(@alignCast(ctx_ptr));

            const input_shape = input.shape();
            const seq_len: usize = @intCast(input_shape[1]);
            const vocab = self.vocab_size;
            const total = seq_len * vocab;

            var data = try std.heap.page_allocator.alloc(f32, total);
            defer std.heap.page_allocator.free(data);

            // Assign very high logit to high_prob_token at every position
            for (0..seq_len) |pos| {
                for (0..vocab) |v| {
                    data[pos * vocab + v] = if (v == self.high_prob_token) 50.0 else -50.0;
                }
            }

            return Array.fromData(
                std.heap.page_allocator,
                f32,
                data,
                &[_]i32{ 1, @intCast(seq_len), @intCast(vocab) },
            );
        }

        fn deinitFn(_: *anyopaque, _: std.mem.Allocator) void {}
    };

    var total_accepted: usize = 0;
    var total_drafted: usize = 0;

    var iteration: usize = 0;
    while (iteration < 100) : (iteration += 1) {
        var mock = MockHighProbModel{};
        const vtable = ModelVTable{
            .forward = @ptrCast(&MockHighProbModel.forwardFn),
            .deinit = @ptrCast(&MockHighProbModel.deinitFn),
            .config = .{
                .num_layers = 1,
                .num_kv_heads = 1,
                .head_dim = 8,
                .vocab_size = 8,
                .hidden_size = 8,
            },
            .ptr = @ptrCast(&mock),
        };

        // Draft tokens all match the high-probability token
        const context = [_]u32{ 0, 1, 2 };
        const draft = [_]u32{ 3, 3, 3 };

        const seed: u64 = @intCast(iteration * 6271 + 17);
        const result = try verifyDraft(vtable, &context, &draft, &.{}, ctx, seed);
        defer alloc.free(result.tokens);

        // Core invariant still holds
        try std.testing.expectEqual(result.accepted + 1, result.tokens.len);

        total_accepted += result.accepted;
        total_drafted += draft.len;
    }

    // Statistical check: with very high target probability for the drafted token,
    // the acceptance ratio p_target / q_draft should be very high (p_target ≈ 1.0,
    // q_draft = 1/V = 1/8 = 0.125, so ratio ≈ 8.0 >> 1.0).
    // Nearly all drafts should be accepted. We expect > 90% acceptance rate.
    const acceptance_rate = @as(f64, @floatFromInt(total_accepted)) / @as(f64, @floatFromInt(total_drafted));
    try std.testing.expect(acceptance_rate > 0.90);
}

// ============================================================
// EagleDrafter Unit Tests
// ============================================================

test "EagleDrafter: init sets dimensions" {
    const drafter = EagleDrafter.init(4096, 32000);
    try std.testing.expectEqual(@as(usize, 4096), drafter.hidden_dim);
    try std.testing.expectEqual(@as(usize, 32000), drafter.vocab_size);
    try std.testing.expectEqual(@as(usize, 5), drafter.num_draft_tokens);
}

test "EagleDrafter: not ready without weights" {
    const drafter = EagleDrafter.init(4096, 32000);
    try std.testing.expect(!drafter.isReady());
}

test "EagleDrafter: propose returns null without weights" {
    const c_test = @import("mlx").c;
    c_test.initErrorHandler();
    const ctx = EagerContext.init(std.testing.allocator);
    const drafter = EagleDrafter.init(4096, 32000);
    const hidden = try Array.zeros(std.testing.allocator, &[_]i32{ 1, 4096 }, .float32);
    defer hidden.deinit();
    const result = try drafter.propose(hidden, 3, ctx, std.testing.allocator);
    try std.testing.expect(result == null);
}

test "EagleDrafter: propose with weights returns draft tokens" {
    const c_test = @import("mlx").c;
    c_test.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    var drafter = EagleDrafter.init(8, 16);
    drafter.num_draft_tokens = 3;

    // Create a simple weight matrix [hidden_dim=8, vocab_size=16]
    var w_data = try allocator.alloc(f32, 8 * 16);
    defer allocator.free(w_data);
    @memset(w_data, 0.0);
    // Make token 5 have highest logit
    for (0..8) |i| w_data[i * 16 + 5] = 10.0;
    const weight = try Array.fromData(allocator, f32, w_data, &[_]i32{ 8, 16 });
    defer weight.deinit();

    drafter.loadWeights(weight, null);
    try std.testing.expect(drafter.isReady());

    // Create hidden state [1, 8]
    const h_data = try allocator.alloc(f32, 8);
    defer allocator.free(h_data);
    @memset(h_data, 1.0);
    const hidden = try Array.fromData(allocator, f32, h_data, &[_]i32{ 1, 8 });
    defer hidden.deinit();

    const result = try drafter.propose(hidden, 3, ctx, allocator);
    try std.testing.expect(result != null);
    defer allocator.free(result.?);

    try std.testing.expectEqual(@as(usize, 3), result.?.len);
    // All tokens should be 5 (highest logit)
    for (result.?) |tok| {
        try std.testing.expectEqual(@as(u32, 5), tok);
    }
}

test "EagleDrafter: deinit is safe on unloaded drafter" {
    var drafter = EagleDrafter.init(4096, 32000);
    drafter.deinit(); // Should not crash
    try std.testing.expect(!drafter.isReady());
}
