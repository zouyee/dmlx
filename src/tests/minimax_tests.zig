/// MiniMax model tests.
///
/// Inline tests in `models/minimax.zig` and `models/minimax_loader.zig`
/// are compiled and run via these imports.
const minimax = @import("../models/minimax.zig");
const minimax_loader = @import("../models/minimax_loader.zig");

// Force compilation of model modules (inline tests run automatically).
test {
    _ = minimax;
    _ = minimax_loader;
}
