/// Process Memory Limiter — enforces memory budgets for the inference engine.
///
/// Provides configurable memory limits (absolute or percentage-based) and
/// triggers LRU eviction of ModelPool entries and TieredKVCache cold-tier
/// offloading when the process exceeds its budget. Rejects new requests
/// with a descriptive error when memory cannot be reduced below the limit.
///
/// Requirements: R23.1, R23.2, R23.3
const std = @import("std");
const model_pool_mod = @import("model_pool.zig");
const tiered_mod = @import("kvcache/tiered.zig");

const ModelPool = model_pool_mod.ModelPool;
const TieredKVCache = tiered_mod.TieredKVCache;

// ------------------------------------------------------------------
// macOS system calls via cImport
// ------------------------------------------------------------------

const darwin = @cImport({
    @cInclude("sys/types.h");
    @cInclude("sys/sysctl.h");
    @cInclude("mach/mach.h");
});

// ------------------------------------------------------------------
// MemoryConfig
// ------------------------------------------------------------------

pub const MemoryConfig = struct {
    /// Absolute memory limit in bytes. Takes precedence over max_percent if set.
    max_bytes: ?usize = null,
    /// Memory limit as a percentage of total system RAM (0.0–1.0).
    max_percent: ?f32 = null,
    /// Safety margin reserved for the OS and other processes.
    safety_margin_bytes: usize = 512 * 1024 * 1024, // 512 MB

    /// Resolve the effective memory limit in bytes.
    /// Returns null if neither max_bytes nor max_percent is configured.
    pub fn effectiveLimit(self: MemoryConfig) ?usize {
        if (self.max_bytes) |abs| return abs;
        if (self.max_percent) |pct| {
            const total = getSystemMemoryBytes();
            if (total == 0) return null;
            const limit_f: f64 = @as(f64, @floatFromInt(total)) * @as(f64, pct);
            return @intFromFloat(@max(limit_f, 0.0));
        }
        return null;
    }
};

// ------------------------------------------------------------------
// System memory queries (macOS / Apple Silicon)
// ------------------------------------------------------------------

/// Return total physical RAM in bytes via sysctl hw.memsize.
/// Returns 0 on failure.
pub fn getSystemMemoryBytes() usize {
    var mem_size: u64 = 0;
    var size: usize = @sizeOf(u64);
    var mib = [_]c_int{ darwin.CTL_HW, darwin.HW_MEMSIZE };
    const rc = darwin.sysctl(&mib, mib.len, @ptrCast(&mem_size), &size, null, 0);
    if (rc != 0) return 0;
    return @intCast(mem_size);
}

/// Return current process resident memory (RSS) in bytes via mach_task_basic_info.
/// Returns 0 on failure.
pub fn getProcessMemoryBytes() usize {
    var info: darwin.mach_task_basic_info_data_t = undefined;
    var count: darwin.mach_msg_type_number_t = @intCast(@sizeOf(darwin.mach_task_basic_info_data_t) / @sizeOf(darwin.natural_t));
    const kr = darwin.task_info(
        darwin.mach_task_self(),
        darwin.MACH_TASK_BASIC_INFO,
        @ptrCast(&info),
        &count,
    );
    if (kr != darwin.KERN_SUCCESS) return 0;
    return @intCast(info.resident_size);
}

// ------------------------------------------------------------------
// Memory enforcement
// ------------------------------------------------------------------

pub const MemoryError = error{
    MemoryLimitExceeded,
};

/// Maximum number of eviction rounds before giving up.
const max_eviction_rounds: usize = 64;

/// Enforce the configured memory limit by evicting ModelPool entries (LRU)
/// and offloading TieredKVCache blocks to SSD until process memory drops
/// below the limit.
///
/// Returns `error.MemoryLimitExceeded` if memory cannot be reduced below
/// the configured limit after exhausting eviction options.
pub fn enforceMemoryLimit(
    pool: ?*ModelPool,
    tiered_cache: ?*TieredKVCache,
    config: MemoryConfig,
) MemoryError!void {
    const limit = config.effectiveLimit() orelse return; // no limit configured

    var rounds: usize = 0;
    while (rounds < max_eviction_rounds) : (rounds += 1) {
        const current = getProcessMemoryBytes();
        if (current <= limit) return; // within budget

        // Strategy 1: offload TieredKVCache hot blocks to SSD (cheaper).
        if (tiered_cache) |tc| {
            if (tc.hotUsedCount() > 0) {
                if (tc.findLRUPage()) |lru_idx| {
                    tc.evictToSSD(lru_idx) catch {};
                    continue; // re-check after eviction
                }
            }
        }

        // Strategy 2: evict LRU model from the pool (more expensive).
        if (pool) |p| {
            if (p.count() > 0) {
                p.evictLRU() catch {};
                continue; // re-check after eviction
            }
        }

        // Nothing left to evict — give up.
        break;
    }

    // Final check after all eviction attempts.
    const final_usage = getProcessMemoryBytes();
    if (final_usage > limit) {
        std.log.warn(
            "memory limiter: process using {d} bytes, limit is {d} bytes — rejecting request",
            .{ final_usage, limit },
        );
        return MemoryError.MemoryLimitExceeded;
    }
}

// ------------------------------------------------------------------
// Auto max_kv_size configuration (R24.1, R24.2, R24.3, R24.4)
// ------------------------------------------------------------------

/// Represents the max_kv_size setting: either auto-computed or an explicit value.
pub const MaxKvSize = union(enum) {
    /// Automatically compute based on available device memory.
    auto,
    /// User-specified explicit token count.
    explicit: usize,
};

/// Parse a `--max-kv-size` CLI argument string.
/// Accepts "auto" or a positive integer.
pub fn parseMaxKvSize(value: []const u8) !MaxKvSize {
    if (std.mem.eql(u8, value, "auto")) return .auto;
    const n = std.fmt.parseInt(usize, value, 10) catch return error.InvalidMaxKvSize;
    if (n == 0) return error.InvalidMaxKvSize;
    return .{ .explicit = n };
}

/// Compute the maximum KV cache token capacity based on available device memory.
///
/// Formula: (total_RAM - model_bytes - safety_margin) / (2 * num_kv_heads * head_dim * (kv_bits / 8) * num_layers)
///
/// Uses the system's actual total RAM via `getSystemMemoryBytes()`.
/// Returns 0 if the system query fails or available memory is insufficient.
///
/// Requirements: R24.1, R24.2
pub fn autoMaxKvSize(
    model_bytes: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_bits: u8,
) usize {
    const total_ram = getSystemMemoryBytes();
    return autoMaxKvSizeWithTotalRam(total_ram, model_bytes, num_layers, num_kv_heads, head_dim, kv_bits);
}

/// Testable variant that accepts total_ram as a parameter instead of querying the OS.
pub fn autoMaxKvSizeWithTotalRam(
    total_ram: usize,
    model_bytes: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_bits: u8,
) usize {
    const safety_margin = (MemoryConfig{}).safety_margin_bytes; // 512 MB default

    // Guard against underflow: if model + safety exceeds total RAM, return 0.
    const overhead = model_bytes +| safety_margin; // saturating add
    if (total_ram <= overhead) return 0;
    const available = total_ram - overhead;

    // For sub-byte quantization (4-bit), we compute bits_per_token and divide
    // at the end to avoid integer division truncating to 0.
    //
    // bits_per_token = 2 * num_kv_heads * head_dim * kv_bits * num_layers
    // bytes_per_token = bits_per_token / 8
    // max_tokens = available / bytes_per_token = (available * 8) / bits_per_token
    const bits_per_token = 2 *| (num_kv_heads *| (head_dim *| (@as(usize, kv_bits) *| num_layers)));
    if (bits_per_token == 0) return 0;

    // Use (available * 8) / bits_per_token to preserve precision for sub-byte kv_bits.
    // Saturating multiply to avoid overflow on very large RAM values.
    const available_bits = available *| 8;
    return available_bits / bits_per_token;
}

/// Resolve a `MaxKvSize` to a concrete token count.
/// For `.auto`, computes based on device memory and model parameters.
/// For `.explicit`, returns the user-specified value directly (R24.3).
pub fn resolveMaxKvSize(
    setting: MaxKvSize,
    model_bytes: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_bits: u8,
) usize {
    return switch (setting) {
        .auto => autoMaxKvSize(model_bytes, num_layers, num_kv_heads, head_dim, kv_bits),
        .explicit => |n| n,
    };
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

test "MemoryConfig: default safety margin is 512MB" {
    const cfg = MemoryConfig{};
    try std.testing.expectEqual(@as(usize, 512 * 1024 * 1024), cfg.safety_margin_bytes);
}

test "MemoryConfig: effectiveLimit returns null when unconfigured" {
    const cfg = MemoryConfig{};
    try std.testing.expect(cfg.effectiveLimit() == null);
}

test "MemoryConfig: effectiveLimit returns max_bytes when set" {
    const cfg = MemoryConfig{ .max_bytes = 4 * 1024 * 1024 * 1024 };
    try std.testing.expectEqual(@as(usize, 4 * 1024 * 1024 * 1024), cfg.effectiveLimit().?);
}

test "MemoryConfig: effectiveLimit computes percentage of system RAM" {
    const cfg = MemoryConfig{ .max_percent = 0.5 };
    const limit = cfg.effectiveLimit();
    // On any real system, 50% of RAM should be > 0.
    if (limit) |l| {
        try std.testing.expect(l > 0);
    }
}

test "MemoryConfig: max_bytes takes precedence over max_percent" {
    const cfg = MemoryConfig{
        .max_bytes = 1024,
        .max_percent = 0.9,
    };
    try std.testing.expectEqual(@as(usize, 1024), cfg.effectiveLimit().?);
}

test "getSystemMemoryBytes returns non-zero on macOS" {
    const mem = getSystemMemoryBytes();
    // Any Apple Silicon Mac has at least 8 GB.
    try std.testing.expect(mem >= 8 * 1024 * 1024 * 1024);
}

test "getProcessMemoryBytes returns non-zero for running process" {
    const mem = getProcessMemoryBytes();
    // A running test process should use at least some memory.
    try std.testing.expect(mem > 0);
}

test "enforceMemoryLimit: no-op when no limit configured" {
    // Should return immediately without error.
    try enforceMemoryLimit(null, null, MemoryConfig{});
}

test "enforceMemoryLimit: no-op when limit is very high" {
    // Set limit to something absurdly high — should pass without eviction.
    const cfg = MemoryConfig{ .max_bytes = std.math.maxInt(usize) };
    try enforceMemoryLimit(null, null, cfg);
}

test "enforceMemoryLimit: returns error when limit is impossibly low and nothing to evict" {
    // Set limit to 1 byte — process will always exceed this, and with no
    // pool or cache to evict from, it should return MemoryLimitExceeded.
    const cfg = MemoryConfig{ .max_bytes = 1 };
    const result = enforceMemoryLimit(null, null, cfg);
    try std.testing.expectError(MemoryError.MemoryLimitExceeded, result);
}

test "enforceMemoryLimit: attempts pool eviction when over limit" {
    const allocator = std.testing.allocator;

    // Create a small model pool with one model.
    var pool = ModelPool.init(allocator, 10000);
    defer pool.deinit();

    const stubLoader = struct {
        fn load(name: []const u8, path: []const u8) anyerror!model_pool_mod.LoadedModel {
            _ = path;
            return model_pool_mod.LoadedModel{
                .vtable = null,
                .name = name,
                .path = "",
                .memory_bytes = name.len * 100,
                .last_used = 0,
            };
        }
    }.load;

    _ = try pool.getOrLoad("test_model", "/m/test", stubLoader);
    try std.testing.expectEqual(@as(usize, 1), pool.count());

    // Set an impossibly low limit — enforceMemoryLimit will try to evict
    // the model from the pool, but process RSS won't actually drop enough.
    const cfg = MemoryConfig{ .max_bytes = 1 };
    const result = enforceMemoryLimit(&pool, null, cfg);

    // The model should have been evicted (pool tried to help).
    try std.testing.expectEqual(@as(usize, 0), pool.count());

    // But process memory is still above 1 byte, so we get the error.
    try std.testing.expectError(MemoryError.MemoryLimitExceeded, result);
}

// ------------------------------------------------------------------
// autoMaxKvSize tests (R24.1, R24.2, R24.3, R24.4)
// ------------------------------------------------------------------

test "autoMaxKvSizeWithTotalRam: basic formula correctness" {
    // 32 GB total RAM, 4 GB model, 512 MB safety margin
    // 32 layers, 8 kv_heads, 128 head_dim, 16-bit
    // available = 32GB - 4GB - 512MB = 27.5 GB = 29_528_211_456
    // bits_per_token = 2 * 8 * 128 * 16 * 32 = 1_048_576
    // max_tokens = (29_528_211_456 * 8) / 1_048_576 = 225_280
    const total_ram: usize = 32 * 1024 * 1024 * 1024; // 32 GB
    const model_bytes: usize = 4 * 1024 * 1024 * 1024; // 4 GB
    const result = autoMaxKvSizeWithTotalRam(total_ram, model_bytes, 32, 8, 128, 16);
    try std.testing.expectEqual(@as(usize, 225_280), result);
}

test "autoMaxKvSizeWithTotalRam: 8-bit KV halves bytes_per_token" {
    const total_ram: usize = 32 * 1024 * 1024 * 1024;
    const model_bytes: usize = 4 * 1024 * 1024 * 1024;
    const result_16 = autoMaxKvSizeWithTotalRam(total_ram, model_bytes, 32, 8, 128, 16);
    const result_8 = autoMaxKvSizeWithTotalRam(total_ram, model_bytes, 32, 8, 128, 8);
    // 8-bit should yield ~2x the tokens of 16-bit (same available memory, half bytes_per_token)
    try std.testing.expectEqual(result_16 * 2, result_8);
}

test "autoMaxKvSizeWithTotalRam: returns 0 when model exceeds RAM" {
    const total_ram: usize = 4 * 1024 * 1024 * 1024;
    const model_bytes: usize = 8 * 1024 * 1024 * 1024; // model larger than RAM
    const result = autoMaxKvSizeWithTotalRam(total_ram, model_bytes, 32, 8, 128, 16);
    try std.testing.expectEqual(@as(usize, 0), result);
}

test "autoMaxKvSizeWithTotalRam: 4-bit KV yields 4x tokens vs 16-bit" {
    const total_ram: usize = 32 * 1024 * 1024 * 1024;
    const model_bytes: usize = 4 * 1024 * 1024 * 1024;
    const result_16 = autoMaxKvSizeWithTotalRam(total_ram, model_bytes, 32, 8, 128, 16);
    const result_4 = autoMaxKvSizeWithTotalRam(total_ram, model_bytes, 32, 8, 128, 4);
    // 4-bit should yield 4x the tokens of 16-bit (same available memory, 1/4 bits_per_token)
    try std.testing.expectEqual(result_16 * 4, result_4);
}

test "autoMaxKvSizeWithTotalRam: returns 0 when num_layers is 0" {
    const total_ram: usize = 32 * 1024 * 1024 * 1024;
    const result = autoMaxKvSizeWithTotalRam(total_ram, 0, 0, 8, 128, 16);
    try std.testing.expectEqual(@as(usize, 0), result);
}

test "autoMaxKvSize: returns non-zero for reasonable config on real hardware" {
    // Use actual system RAM — any Apple Silicon Mac should yield a positive result
    // for a small model config.
    const result = autoMaxKvSize(
        1 * 1024 * 1024 * 1024, // 1 GB model
        32, // layers
        8, // kv_heads
        128, // head_dim
        16, // kv_bits
    );
    try std.testing.expect(result > 0);
}

test "parseMaxKvSize: parses 'auto'" {
    const result = try parseMaxKvSize("auto");
    try std.testing.expect(result == .auto);
}

test "parseMaxKvSize: parses integer" {
    const result = try parseMaxKvSize("4096");
    try std.testing.expectEqual(@as(usize, 4096), result.explicit);
}

test "parseMaxKvSize: rejects zero" {
    const result = parseMaxKvSize("0");
    try std.testing.expectError(error.InvalidMaxKvSize, result);
}

test "parseMaxKvSize: rejects non-numeric" {
    const result = parseMaxKvSize("foo");
    try std.testing.expectError(error.InvalidMaxKvSize, result);
}

test "resolveMaxKvSize: explicit returns value directly" {
    const result = resolveMaxKvSize(.{ .explicit = 2048 }, 0, 32, 8, 128, 16);
    try std.testing.expectEqual(@as(usize, 2048), result);
}

test "resolveMaxKvSize: auto delegates to autoMaxKvSize" {
    const result = resolveMaxKvSize(.auto, 1 * 1024 * 1024 * 1024, 32, 8, 128, 16);
    // Should match autoMaxKvSize with same params
    const expected = autoMaxKvSize(1 * 1024 * 1024 * 1024, 32, 8, 128, 16);
    try std.testing.expectEqual(expected, result);
}
