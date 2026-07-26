//! Synthetic smoke tests for native_loader config parsing.
//! No real model required — writes a minimal config.json to a temp dir and
//! verifies defaults + overrides, so the native loader path has CI coverage
//! (the real 141GB model is unavailable on CI runners).

const std = @import("std");
const config_mod = @import("../native_loader/config.zig");

fn writeSyntheticConfig(contents: []const u8) !void {
    _ = std.c.mkdir("/tmp/dmlx_cfg_test", 0o755);
    const f = std.c.fopen("/tmp/dmlx_cfg_test/config.json", "w") orelse return error.WriteFailed;
    defer _ = std.c.fclose(f);
    if (contents.len > 0) {
        try std.testing.expectEqual(contents.len, std.c.fwrite(contents.ptr, 1, contents.len, f));
    }
}

test "native config: defaults when fields absent" {
    try writeSyntheticConfig("{}");
    const cfg = try config_mod.parse(std.testing.allocator, "/tmp/dmlx_cfg_test");

    try std.testing.expectEqual(@as(u32, 43), cfg.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 256), cfg.n_routed_experts);
    try std.testing.expectEqual(@as(u32, 6), cfg.num_experts_per_tok);
    try std.testing.expectEqual(@as(u32, 1), cfg.eos_token_id); // DSV4 fallback default
}

test "native config: overrides incl. eos_token_id" {
    try writeSyntheticConfig(
        \\{
        \\  "num_hidden_layers": 61,
        \\  "vocab_size": 163840,
        \\  "eos_token_id": 2,
        \\  "rms_norm_eps": 1e-5
        \\}
    );
    const cfg = try config_mod.parse(std.testing.allocator, "/tmp/dmlx_cfg_test");

    try std.testing.expectEqual(@as(u32, 61), cfg.num_hidden_layers);
    try std.testing.expectEqual(@as(u32, 163840), cfg.vocab_size);
    try std.testing.expectEqual(@as(u32, 2), cfg.eos_token_id); // read from config.json, not hardcoded
    try std.testing.expectApproxEqAbs(@as(f32, 1e-5), cfg.rms_norm_eps, 1e-12);
}

test "native config: missing config.json errors" {
    _ = std.c.mkdir("/tmp/dmlx_cfg_test_empty", 0o755);
    _ = std.c.unlink("/tmp/dmlx_cfg_test_empty/config.json");
    try std.testing.expectError(error.FileNotFound, config_mod.parse(std.testing.allocator, "/tmp/dmlx_cfg_test_empty"));
}
