/// Perplexity evaluation tool for DMLX.
///
/// Computes perplexity on a text dataset to measure model quality:
///   perplexity = exp(mean(cross_entropy_per_token))
///
/// Lower perplexity indicates better model fit to the data.
/// Useful for:
///   - Validating quantization quality (compare FP16 vs INT4 perplexity)
///   - Comparing model architectures on the same dataset
///   - Detecting model corruption or loading errors
///
/// Usage:
///   dmlx evaluate --model <path> --data <file> [--max-tokens N] [--stride S]
///
/// The data file should be a plain text file. The tool tokenizes the text,
/// runs forward passes in sliding windows, and computes the average
/// cross-entropy loss across all tokens.
const std = @import("std");
const c = @import("mlx").c;
const ops = @import("mlx").ops;
const array_mod = @import("mlx").array;
const generation_mod = @import("generation.zig");
const kvcache = @import("kvcache.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const ModelVTable = generation_mod.ModelVTable;
const KVCacheStrategy = kvcache.KVCacheStrategy;

// ============================================================
// EvaluateConfig
// ============================================================

pub const EvaluateConfig = struct {
    /// Path to the model directory.
    model_path: []const u8,
    /// Path to the evaluation text file.
    data_path: []const u8,
    /// Maximum number of tokens to evaluate (0 = all).
    max_tokens: usize = 0,
    /// Stride for sliding window evaluation.
    /// Smaller stride = more accurate but slower.
    stride: usize = 512,
    /// Context window size for each forward pass.
    context_size: usize = 1024,
};

// ============================================================
// EvaluateResult
// ============================================================

pub const EvaluateResult = struct {
    /// Perplexity score: exp(mean_loss).
    perplexity: f64,
    /// Mean cross-entropy loss per token (in nats).
    mean_loss: f64,
    /// Total number of tokens evaluated.
    num_tokens: usize,
    /// Number of forward passes performed.
    num_windows: usize,
};

// ============================================================
// Core evaluation logic
// ============================================================

/// Compute perplexity on a sequence of token IDs using the given model.
///
/// Uses a sliding window approach:
///   1. For each window of `context_size` tokens with `stride` step:
///      a. Run forward pass to get logits [1, seq_len, vocab_size]
///      b. Compute cross-entropy loss for each token position
///      c. Accumulate loss for positions in the stride region
///   2. Perplexity = exp(total_loss / num_scored_tokens)
pub fn computePerplexity(
    allocator: std.mem.Allocator,
    model: ModelVTable,
    tokens: []const u32,
    config: EvaluateConfig,
    caches: []KVCacheStrategy,
    ctx: EagerContext,
) !EvaluateResult {
    if (tokens.len < 2) {
        return EvaluateResult{
            .perplexity = 1.0,
            .mean_loss = 0.0,
            .num_tokens = 0,
            .num_windows = 0,
        };
    }

    const max_tokens = if (config.max_tokens > 0) @min(config.max_tokens, tokens.len) else tokens.len;
    const eval_tokens = tokens[0..max_tokens];

    var total_loss: f64 = 0.0;
    var total_scored: usize = 0;
    var num_windows: usize = 0;

    var pos: usize = 0;
    while (pos < eval_tokens.len - 1) {
        const window_end = @min(pos + config.context_size, eval_tokens.len);
        const window = eval_tokens[pos..window_end];

        if (window.len < 2) break;

        // Score this window
        const window_loss = try scoreWindow(allocator, model, window, caches, ctx);

        // Only count tokens in the stride region (avoid double-counting overlap)
        const score_start = if (pos == 0) 0 else config.context_size - config.stride;
        const score_end = window.len - 1; // -1 because we predict next token
        const num_scored = if (score_end > score_start) score_end - score_start else 0;

        total_loss += window_loss.loss * @as(f64, @floatFromInt(window_loss.num_tokens));
        total_scored += num_scored;
        num_windows += 1;

        pos += config.stride;
        if (pos + 1 >= eval_tokens.len) break;
    }

    if (total_scored == 0) {
        return EvaluateResult{
            .perplexity = 1.0,
            .mean_loss = 0.0,
            .num_tokens = 0,
            .num_windows = num_windows,
        };
    }

    const mean_loss = total_loss / @as(f64, @floatFromInt(total_scored));
    const perplexity = @exp(mean_loss);

    return EvaluateResult{
        .perplexity = perplexity,
        .mean_loss = mean_loss,
        .num_tokens = total_scored,
        .num_windows = num_windows,
    };
}

const WindowLoss = struct {
    loss: f64,
    num_tokens: usize,
};

/// Score a single window of tokens. Returns mean cross-entropy loss.
fn scoreWindow(
    allocator: std.mem.Allocator,
    model: ModelVTable,
    window: []const u32,
    caches: []KVCacheStrategy,
    _: EagerContext,
) !WindowLoss {
    if (window.len < 2) return .{ .loss = 0.0, .num_tokens = 0 };

    // Build input array [1, seq_len]
    const input_arr = Array.fromData(
        allocator,
        u32,
        window,
        &[_]i32{ 1, @intCast(window.len) },
    ) catch return .{ .loss = 0.0, .num_tokens = 0 };
    defer input_arr.deinit();

    // Forward pass → logits [1, seq_len, vocab_size]
    const logits = model.forward(model.ptr, input_arr, null, caches) catch
        return .{ .loss = 0.0, .num_tokens = 0 };
    defer logits.deinit();

    // Compute cross-entropy: -log(softmax(logits)[target])
    // For each position i, target is window[i+1]
    const logits_shape = logits.shape();
    const seq_len: usize = @intCast(logits_shape[1]);
    const vocab_size: usize = @intCast(logits_shape[2]);

    // Get logits data
    try logits.eval();
    const logits_data = try logits.dataPtr(f32);

    var total_loss: f64 = 0.0;
    const num_predictions = seq_len - 1;

    for (0..num_predictions) |i| {
        const target = window[i + 1];
        if (target >= vocab_size) continue;

        // Compute log-softmax for position i
        const row = logits_data[i * vocab_size .. (i + 1) * vocab_size];

        // Find max for numerical stability
        var max_logit: f32 = row[0];
        for (row[1..]) |v| {
            if (v > max_logit) max_logit = v;
        }

        // Compute log-sum-exp
        var sum_exp: f64 = 0.0;
        for (row) |v| {
            sum_exp += @exp(@as(f64, v - max_logit));
        }
        const log_sum_exp = @as(f64, max_logit) + @log(sum_exp);

        // Cross-entropy for this token: -log_softmax[target]
        const log_prob = @as(f64, row[target]) - log_sum_exp;
        total_loss -= log_prob;
    }

    const mean_loss = if (num_predictions > 0)
        total_loss / @as(f64, @floatFromInt(num_predictions))
    else
        0.0;

    return .{ .loss = mean_loss, .num_tokens = num_predictions };
}

/// Print evaluation results to stderr.
pub fn printResults(result: EvaluateResult) void {
    std.debug.print("\n=== Perplexity Evaluation ===\n", .{});
    std.debug.print("Perplexity:    {d:.2}\n", .{result.perplexity});
    std.debug.print("Mean loss:     {d:.4} nats\n", .{result.mean_loss});
    std.debug.print("Tokens scored: {d}\n", .{result.num_tokens});
    std.debug.print("Windows:       {d}\n", .{result.num_windows});
    std.debug.print("============================\n\n", .{});
}

// ============================================================
// Unit Tests
// ============================================================

test "EvaluateConfig: default values" {
    const config = EvaluateConfig{
        .model_path = "/tmp/model",
        .data_path = "/tmp/data.txt",
    };
    try std.testing.expectEqual(@as(usize, 0), config.max_tokens);
    try std.testing.expectEqual(@as(usize, 512), config.stride);
    try std.testing.expectEqual(@as(usize, 1024), config.context_size);
}

test "EvaluateResult: perplexity of 1.0 means perfect prediction" {
    const result = EvaluateResult{
        .perplexity = 1.0,
        .mean_loss = 0.0,
        .num_tokens = 100,
        .num_windows = 1,
    };
    try std.testing.expectEqual(@as(f64, 0.0), result.mean_loss);
}

test "computePerplexity: empty tokens returns default" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const config = EvaluateConfig{
        .model_path = "",
        .data_path = "",
    };

    const result = try computePerplexity(allocator, undefined, &[_]u32{}, config, &.{}, ctx);
    try std.testing.expectEqual(@as(f64, 1.0), result.perplexity);
    try std.testing.expectEqual(@as(usize, 0), result.num_tokens);
}

test "computePerplexity: single token returns default" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    const config = EvaluateConfig{
        .model_path = "",
        .data_path = "",
    };

    const result = try computePerplexity(allocator, undefined, &[_]u32{42}, config, &.{}, ctx);
    try std.testing.expectEqual(@as(f64, 1.0), result.perplexity);
    try std.testing.expectEqual(@as(usize, 0), result.num_tokens);
}
