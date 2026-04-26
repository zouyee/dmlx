/// Sampling strategies for LLM text generation.
///
/// All samplers operate on CPU over logits arrays copied from GPU.
/// This is efficient because:
///   - Each decode step generates only one token per sequence
///   - Logits vector (vocab_size) is small to copy CPU↔GPU
///   - Sampling logic is branching-heavy (poor fit for GPU SIMT)
const std = @import("std");
const c = @import("c.zig");
const array_mod = @import("array.zig");

const Array = array_mod.Array;

/// Sampler interface — all samplers conform to this signature.
pub const Sampler = *const fn (
    logits: Array, // [vocab_size] float32 logits
    allocator: std.mem.Allocator,
    temperature: f32,
) error{ MlxError, OutOfMemory }!u32;

/// Greedy sampling: always pick the highest logit.
pub fn greedy(logits: Array, allocator: std.mem.Allocator, temperature: f32) !u32 {
    _ = allocator;
    _ = temperature;

    try logits.eval();
    const data = try logits.dataPtr(f32);
    const vocab_size = logits.size();

    var max_idx: usize = 0;
    var max_val: f32 = data[0];
    for (1..vocab_size) |i| {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return @intCast(max_idx);
}

/// Temperature scaling + sample.
pub fn temperatureSample(logits: Array, allocator: std.mem.Allocator, temperature: f32, rand: std.Random) !u32 {
    if (temperature <= 0.0) {
        return greedy(logits, allocator, temperature);
    }

    try logits.eval();
    const data = try logits.dataPtr(f32);
    const vocab_size = logits.size();

    // Apply temperature and compute softmax
    var probs = try allocator.alloc(f32, vocab_size);
    defer allocator.free(probs);

    var max_logit: f32 = data[0];
    for (1..vocab_size) |i| {
        if (data[i] > max_logit) max_logit = data[i];
    }

    var sum_exp: f32 = 0;
    for (0..vocab_size) |i| {
        probs[i] = @exp((data[i] - max_logit) / temperature);
        sum_exp += probs[i];
    }
    for (0..vocab_size) |i| {
        probs[i] /= sum_exp;
    }

    return sampleFromProbs(probs, vocab_size, rand);
}

/// Top-K sampling: keep only k highest-probability tokens, then sample.
pub fn topKSampler(logits: Array, allocator: std.mem.Allocator, temperature: f32, k: usize, rand: std.Random) !u32 {
    try logits.eval();
    const data = try logits.dataPtr(f32);
    const vocab_size = logits.size();
    const k_actual = @min(k, vocab_size);

    // Apply temperature
    var scored = try allocator.alloc(ScoredToken, vocab_size);
    defer allocator.free(scored);

    for (0..vocab_size) |i| {
        scored[i] = .{
            .token = @intCast(i),
            .score = if (temperature <= 0.0) data[i] else data[i] / temperature,
        };
    }

    // Partial sort to find top k (simple selection for small vocab)
    // For vocab_size ~ 100k, full sort is acceptable on CPU
    std.sort.insertion(ScoredToken, scored[0..vocab_size], {}, scoredGreater);

    // Softmax over top-k
    var top_probs = try allocator.alloc(f32, k_actual);
    defer allocator.free(top_probs);

    const max_score: f32 = scored[0].score;
    var sum_exp: f32 = 0;
    for (0..k_actual) |i| {
        top_probs[i] = @exp(scored[i].score - max_score);
        sum_exp += top_probs[i];
    }
    for (0..k_actual) |i| {
        top_probs[i] /= sum_exp;
    }

    const idx = sampleFromProbs(top_probs, k_actual, rand);
    return scored[idx].token;
}

/// Top-P (nucleus) sampling: keep tokens with cumulative probability <= p.
pub fn topPSampler(logits: Array, allocator: std.mem.Allocator, temperature: f32, p: f32, rand: std.Random) !u32 {
    if (p >= 1.0) {
        return temperatureSample(logits, allocator, temperature, rand);
    }

    try logits.eval();
    const data = try logits.dataPtr(f32);
    const vocab_size = logits.size();

    // Apply temperature and score
    var scored = try allocator.alloc(ScoredToken, vocab_size);
    defer allocator.free(scored);

    for (0..vocab_size) |i| {
        scored[i] = .{
            .token = @intCast(i),
            .score = if (temperature <= 0.0) data[i] else data[i] / temperature,
        };
    }

    // Sort descending by score
    std.sort.insertion(ScoredToken, scored[0..vocab_size], {}, scoredGreater);

    // Compute softmax probabilities
    var probs = try allocator.alloc(f32, vocab_size);
    defer allocator.free(probs);

    const max_score: f32 = scored[0].score;
    var sum_exp: f32 = 0;
    for (0..vocab_size) |i| {
        probs[i] = @exp(scored[i].score - max_score);
        sum_exp += probs[i];
    }
    for (0..vocab_size) |i| {
        probs[i] /= sum_exp;
    }

    // Find nucleus cutoff
    var cumsum: f32 = 0;
    var cutoff: usize = vocab_size;
    for (0..vocab_size) |i| {
        cumsum += probs[i];
        if (cumsum > p) {
            cutoff = i + 1;
            break;
        }
    }

    // Renormalize nucleus
    var nucleus_sum: f32 = 0;
    for (0..cutoff) |i| {
        nucleus_sum += probs[i];
    }
    for (0..cutoff) |i| {
        probs[i] /= nucleus_sum;
    }

    const idx = sampleFromProbs(probs[0..cutoff], cutoff, rand);
    return scored[idx].token;
}

/// Combined Top-K + Top-P sampler (standard practice).
pub fn topKTopPSampler(
    logits: Array,
    allocator: std.mem.Allocator,
    temperature: f32,
    k: usize,
    p: f32,
    rand: std.Random,
) !u32 {
    if (temperature <= 0.0) {
        return greedy(logits, allocator, temperature);
    }

    try logits.eval();
    const data = try logits.dataPtr(f32);
    const vocab_size = logits.size();

    var scored = try allocator.alloc(ScoredToken, vocab_size);
    defer allocator.free(scored);

    for (0..vocab_size) |i| {
        scored[i] = .{
            .token = @intCast(i),
            .score = data[i] / temperature,
        };
    }

    std.sort.insertion(ScoredToken, scored[0..vocab_size], {}, scoredGreater);

    // Apply top-k first
    const k_actual = @min(k, vocab_size);

    // Compute probs over top-k
    var probs = try allocator.alloc(f32, k_actual);
    defer allocator.free(probs);

    const max_score: f32 = scored[0].score;
    var sum_exp: f32 = 0;
    for (0..k_actual) |i| {
        probs[i] = @exp(scored[i].score - max_score);
        sum_exp += probs[i];
    }
    for (0..k_actual) |i| {
        probs[i] /= sum_exp;
    }

    // Then apply top-p on the sorted top-k
    var cumsum: f32 = 0;
    var cutoff: usize = k_actual;
    for (0..k_actual) |i| {
        cumsum += probs[i];
        if (cumsum > p) {
            cutoff = i + 1;
            break;
        }
    }

    // Renormalize
    var nucleus_sum: f32 = 0;
    for (0..cutoff) |i| {
        nucleus_sum += probs[i];
    }
    for (0..cutoff) |i| {
        probs[i] /= nucleus_sum;
    }

    const idx = sampleFromProbs(probs[0..cutoff], cutoff, rand);
    return scored[idx].token;
}

// ------------------------------------------------------------------
// Internal helpers
// ------------------------------------------------------------------

const ScoredToken = struct {
    token: u32,
    score: f32,
};

fn scoredGreater(_: void, a: ScoredToken, b: ScoredToken) bool {
    return a.score > b.score;
}

/// Sample an index from a probability distribution using uniform random.
fn sampleFromProbs(probs: []const f32, n: usize, rand: std.Random) u32 {
    const r = rand.float(f32);
    var cumsum: f32 = 0;
    for (0..n) |i| {
        cumsum += probs[i];
        if (r < cumsum) {
            return @intCast(i);
        }
    }
    return @intCast(n - 1);
}

/// Configurable sampler that bundles temperature, top_k, top_p, repetition_penalty.
pub const SamplerConfig = struct {
    temperature: f32 = 1.0,
    top_k: usize = 50,
    top_p: f32 = 1.0,
    /// Repetition penalty (1.0 = no penalty, >1.0 reduces repeated tokens).
    repetition_penalty: f32 = 1.0,
    /// Token history for repetition penalty (caller manages this).
    context_tokens: ?[]const u32 = null,
    prng: std.Random.DefaultPrng,

    pub fn init(seed: u64) SamplerConfig {
        return .{
            .temperature = 1.0,
            .top_k = 50,
            .top_p = 1.0,
            .repetition_penalty = 1.0,
            .context_tokens = null,
            .prng = std.Random.DefaultPrng.init(seed),
        };
    }

    pub fn sample(self: *SamplerConfig, logits: Array, allocator: std.mem.Allocator) !u32 {
        // Apply repetition penalty if enabled
        var penalized_logits = logits;
        var owned_penalty_array = false;
        defer if (owned_penalty_array) penalized_logits.deinit();

        if (self.repetition_penalty > 1.0) {
            if (self.context_tokens) |ctx_tokens| {
                if (ctx_tokens.len > 0) {
                    penalized_logits = try applyRepetitionPenalty(logits, ctx_tokens, self.repetition_penalty, allocator);
                    owned_penalty_array = true;
                }
            }
        }

        if (self.temperature <= 0.0) {
            return greedy(penalized_logits, allocator, self.temperature);
        }
        if (self.top_p >= 1.0 and self.top_k >= penalized_logits.size()) {
            return temperatureSample(penalized_logits, allocator, self.temperature, self.prng.random());
        }
        return topKTopPSampler(penalized_logits, allocator, self.temperature, self.top_k, self.top_p, self.prng.random());
    }
};

/// Apply repetition penalty to logits for tokens that appear in context.
/// For each token in context_tokens, if its logit is positive, divide by penalty;
/// if negative, multiply by penalty. This reduces the probability of repeated tokens.
fn applyRepetitionPenalty(logits: Array, context_tokens: []const u32, penalty: f32, allocator: std.mem.Allocator) !Array {
    try logits.eval();
    const data = try logits.dataPtr(f32);
    const vocab_size = logits.size();

    var penalized = try allocator.alloc(f32, vocab_size);
    defer allocator.free(penalized);

    // Copy original logits
    @memcpy(penalized, data[0..vocab_size]);

    // Apply penalty to tokens that appeared in context
    for (context_tokens) |tok| {
        if (tok < vocab_size) {
            if (penalized[tok] > 0) {
                penalized[tok] /= penalty;
            } else {
                penalized[tok] *= penalty;
            }
        }
    }

    // Create new Array from penalized logits
    return Array.fromData(allocator, f32, penalized, &[_]i32{@intCast(vocab_size)});
}
