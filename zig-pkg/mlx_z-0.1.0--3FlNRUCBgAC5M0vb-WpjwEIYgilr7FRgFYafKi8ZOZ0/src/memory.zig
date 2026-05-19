/// Memory management wrappers for mlx-c (memory.h).
/// Provides GPU cache control, memory limit configuration,
/// and peak/active memory querying.
const c = @import("c.zig");

/// Free all cached memory (arrays, compiled functions, etc.).
pub fn clearCache() !void {
    return c.check(c.c.mlx_clear_cache());
}

/// Get the current active memory in bytes.
pub fn getActiveMemory() !usize {
    var res: usize = 0;
    try c.check(c.c.mlx_get_active_memory(&res));
    return res;
}

/// Get the current cache memory in bytes.
pub fn getCacheMemory() !usize {
    var res: usize = 0;
    try c.check(c.c.mlx_get_cache_memory(&res));
    return res;
}

/// Get the current memory limit in bytes.
/// Returns 0 if no limit is set.
pub fn getMemoryLimit() !usize {
    var res: usize = 0;
    try c.check(c.c.mlx_get_memory_limit(&res));
    return res;
}

/// Get the peak active memory in bytes since last reset.
pub fn getPeakMemory() !usize {
    var res: usize = 0;
    try c.check(c.c.mlx_get_peak_memory(&res));
    return res;
}

/// Reset the peak memory counter.
pub fn resetPeakMemory() !void {
    return c.check(c.c.mlx_reset_peak_memory());
}

/// Set the cache memory limit in bytes.
/// Note: the mlx-c API writes the old limit to `res`.
pub fn setCacheLimit(limit: usize) !usize {
    var old: usize = 0;
    try c.check(c.c.mlx_set_cache_limit(&old, limit));
    return old;
}

/// Set the maximum memory limit in bytes.
/// Note: the mlx-c API writes the old limit to `res`.
pub fn setMemoryLimit(limit: usize) !usize {
    var old: usize = 0;
    try c.check(c.c.mlx_set_memory_limit(&old, limit));
    return old;
}

/// Set the wired memory limit in bytes.
/// Note: the mlx-c API writes the old limit to `res`.
pub fn setWiredLimit(limit: usize) !usize {
    var old: usize = 0;
    try c.check(c.c.mlx_set_wired_limit(&old, limit));
    return old;
}
