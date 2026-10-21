/// Metal GPU diagnostic wrappers for mlx-c (metal.h).
const c = @import("c.zig");

/// Check if the Metal GPU backend is available.
pub fn isAvailable() !bool {
    var res: bool = false;
    try c.check(c.c.mlx_metal_is_available(&res));
    return res;
}

/// Start Metal GPU frame capture. The capture is saved to `path` when stopped.
/// Useful for profiling and debugging GPU operations.
pub fn startCapture(path: []const u8) !void {
    return c.check(c.c.mlx_metal_start_capture(path.ptr));
}

/// Stop the Metal GPU frame capture.
pub fn stopCapture() !void {
    return c.check(c.c.mlx_metal_stop_capture());
}
