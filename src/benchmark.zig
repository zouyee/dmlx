/// Benchmark tool for measuring LLM inference performance.
///
/// Measures key metrics:
///   - TTFT (time to first token)
///   - ITL (inter-token latency, mean)
///   - Throughput (tokens per second)
///   - Peak memory usage (MB)
///
/// Requirements: R25.1, R25.2, R25.3
const std = @import("std");
const memory = @import("memory.zig");

// ============================================================
// BenchmarkConfig
// ============================================================

pub const BenchmarkConfig = struct {
    model_path: []const u8,
    input_tokens: usize = 32,
    output_tokens: usize = 128,
    warmup_runs: usize = 1,
    num_runs: usize = 3,
};

// ============================================================
// BenchmarkResult
// ============================================================

pub const BenchmarkResult = struct {
    /// Time to first token in milliseconds.
    ttft_ms: f64,
    /// Mean inter-token latency in milliseconds.
    itl_ms: f64,
    /// Throughput in tokens per second.
    throughput_tps: f64,
    /// Peak process memory usage in megabytes.
    peak_memory_mb: f64,
};

// ============================================================
// RunMetrics — per-run timing data
// ============================================================

pub const RunMetrics = struct {
    ttft_ns: u64,
    total_gen_ns: u64,
    tokens_generated: usize,
};

// ============================================================
// Benchmark runner
// ============================================================

/// Run a benchmark using the provided generation function.
///
/// `gen_fn` is a callback that performs one full generation run:
///   - Takes synthetic input tokens (slice of u32) and the desired output token count.
///   - Returns a `RunMetrics` with timing data for that run.
///
/// This design decouples the benchmark harness from model loading,
/// allowing the CLI layer to set up the model/caches and pass in a
/// closure that calls the generation API.
pub fn runBenchmark(
    config: BenchmarkConfig,
    gen_fn: *const fn (input_tokens: []const u32, output_tokens: usize) anyerror!RunMetrics,
) !BenchmarkResult {
    // Build synthetic input: all 1s (token ID 1).
    var input_buf: [4096]u32 = undefined;
    const input_len = @min(config.input_tokens, input_buf.len);
    for (input_buf[0..input_len]) |*t| t.* = 1;
    const input_tokens = input_buf[0..input_len];

    // Warmup runs (discard results).
    for (0..config.warmup_runs) |_| {
        _ = try gen_fn(input_tokens, config.output_tokens);
    }

    // Timed runs.
    var total_ttft_ns: u64 = 0;
    var total_gen_ns: u64 = 0;
    var total_tokens: usize = 0;
    var peak_mem: usize = 0;

    for (0..config.num_runs) |_| {
        const metrics = try gen_fn(input_tokens, config.output_tokens);

        total_ttft_ns += metrics.ttft_ns;
        total_gen_ns += metrics.total_gen_ns;
        total_tokens += metrics.tokens_generated;

        // Sample peak memory after each run.
        const mem_now = memory.getProcessMemoryBytes();
        if (mem_now > peak_mem) peak_mem = mem_now;
    }

    const num_runs_f: f64 = @floatFromInt(config.num_runs);
    const total_tokens_f: f64 = @floatFromInt(total_tokens);
    const total_gen_s: f64 = @as(f64, @floatFromInt(total_gen_ns)) / 1_000_000_000.0;

    // Mean TTFT across runs.
    const mean_ttft_ms: f64 = @as(f64, @floatFromInt(total_ttft_ns)) / (num_runs_f * 1_000_000.0);

    // Mean ITL: total generation time / total tokens generated (excluding first token).
    // ITL covers the decode phase only (after first token).
    const decode_tokens = if (total_tokens > config.num_runs)
        total_tokens - config.num_runs // subtract one first-token per run
    else
        total_tokens;
    const mean_itl_ms: f64 = if (decode_tokens > 0)
        (@as(f64, @floatFromInt(total_gen_ns - total_ttft_ns)) / @as(f64, @floatFromInt(decode_tokens))) / 1_000_000.0
    else
        0.0;

    // Throughput: total tokens / total generation time.
    const throughput: f64 = if (total_gen_s > 0.0) total_tokens_f / total_gen_s else 0.0;

    const peak_mb: f64 = @as(f64, @floatFromInt(peak_mem)) / (1024.0 * 1024.0);

    return BenchmarkResult{
        .ttft_ms = mean_ttft_ms,
        .itl_ms = mean_itl_ms,
        .throughput_tps = throughput,
        .peak_memory_mb = peak_mb,
    };
}

/// Format and print benchmark results to stderr.
pub fn printResults(result: BenchmarkResult) void {
    std.debug.print(
        \\
        \\=== Benchmark Results ===
        \\  TTFT (time to first token): {d:.2} ms
        \\  ITL  (inter-token latency): {d:.2} ms
        \\  Throughput:                  {d:.2} tokens/s
        \\  Peak memory:                 {d:.1} MB
        \\
    , .{
        result.ttft_ms,
        result.itl_ms,
        result.throughput_tps,
        result.peak_memory_mb,
    });
}

// ============================================================
// Tests
// ============================================================

test "BenchmarkConfig default values" {
    const cfg = BenchmarkConfig{ .model_path = "/tmp/model" };
    try std.testing.expectEqual(@as(usize, 32), cfg.input_tokens);
    try std.testing.expectEqual(@as(usize, 128), cfg.output_tokens);
    try std.testing.expectEqual(@as(usize, 1), cfg.warmup_runs);
    try std.testing.expectEqual(@as(usize, 3), cfg.num_runs);
    try std.testing.expectEqualStrings("/tmp/model", cfg.model_path);
}

test "BenchmarkConfig custom values" {
    const cfg = BenchmarkConfig{
        .model_path = "/models/llama",
        .input_tokens = 64,
        .output_tokens = 256,
        .warmup_runs = 2,
        .num_runs = 5,
    };
    try std.testing.expectEqual(@as(usize, 64), cfg.input_tokens);
    try std.testing.expectEqual(@as(usize, 256), cfg.output_tokens);
    try std.testing.expectEqual(@as(usize, 2), cfg.warmup_runs);
    try std.testing.expectEqual(@as(usize, 5), cfg.num_runs);
}

test "BenchmarkResult stores metrics correctly" {
    const result = BenchmarkResult{
        .ttft_ms = 42.5,
        .itl_ms = 12.3,
        .throughput_tps = 81.3,
        .peak_memory_mb = 1024.0,
    };
    try std.testing.expectEqual(@as(f64, 42.5), result.ttft_ms);
    try std.testing.expectEqual(@as(f64, 12.3), result.itl_ms);
    try std.testing.expectEqual(@as(f64, 81.3), result.throughput_tps);
    try std.testing.expectEqual(@as(f64, 1024.0), result.peak_memory_mb);
}

test "runBenchmark with mock generation function" {
    const Mock = struct {
        var call_count: usize = 0;

        fn generate(input_tokens: []const u32, output_tokens: usize) anyerror!RunMetrics {
            // Verify synthetic input is all 1s.
            for (input_tokens) |t| {
                if (t != 1) return error.TestUnexpectedResult;
            }
            call_count += 1;
            _ = output_tokens;
            return RunMetrics{
                .ttft_ns = 10_000_000, // 10 ms
                .total_gen_ns = 100_000_000, // 100 ms
                .tokens_generated = 10,
            };
        }
    };

    Mock.call_count = 0;

    const config = BenchmarkConfig{
        .model_path = "/tmp/test",
        .input_tokens = 8,
        .output_tokens = 10,
        .warmup_runs = 1,
        .num_runs = 3,
    };

    const result = try runBenchmark(config, &Mock.generate);

    // warmup (1) + timed (3) = 4 total calls
    try std.testing.expectEqual(@as(usize, 4), Mock.call_count);

    // Mean TTFT: 10ms across 3 runs = 10ms
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), result.ttft_ms, 0.01);

    // Throughput: 30 tokens / 0.3s = 100 tps
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), result.throughput_tps, 0.1);

    // Peak memory should be > 0 (we're a running process)
    try std.testing.expect(result.peak_memory_mb > 0.0);
}

test "runBenchmark with zero warmup runs" {
    const Mock = struct {
        var call_count: usize = 0;

        fn generate(_: []const u32, _: usize) anyerror!RunMetrics {
            call_count += 1;
            return RunMetrics{
                .ttft_ns = 5_000_000,
                .total_gen_ns = 50_000_000,
                .tokens_generated = 5,
            };
        }
    };

    Mock.call_count = 0;

    const config = BenchmarkConfig{
        .model_path = "/tmp/test",
        .input_tokens = 4,
        .output_tokens = 5,
        .warmup_runs = 0,
        .num_runs = 2,
    };

    const result = try runBenchmark(config, &Mock.generate);

    // No warmup, just 2 timed runs
    try std.testing.expectEqual(@as(usize, 2), Mock.call_count);

    // Mean TTFT: 5ms
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result.ttft_ms, 0.01);

    // Throughput: 10 tokens / 0.1s = 100 tps
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), result.throughput_tps, 0.1);
}
