const std = @import("std");

pub fn isBooleanFlag(flag: []const u8, include_raw: bool) bool {
    return std.mem.eql(u8, flag, "--smelt") or
        std.mem.eql(u8, flag, "--mlock-backbone") or
        std.mem.eql(u8, flag, "--metal-moe") or
        std.mem.eql(u8, flag, "--metal-full") or
        std.mem.eql(u8, flag, "--native") or
        std.mem.eql(u8, flag, "--mlx") or
        std.mem.eql(u8, flag, "--distributed") or
        (include_raw and std.mem.eql(u8, flag, "--raw"));
}

pub fn applyEngineSelector(flag: []const u8, native: *bool) void {
    if (std.mem.eql(u8, flag, "--native")) native.* = true;
    if (std.mem.eql(u8, flag, "--mlx")) native.* = false;
}

fn scan(args: []const []const u8, include_raw: bool, default_native: bool) struct { native: bool, port_seen: bool, prompt_seen: bool } {
    var native = default_native;
    var port_seen = false;
    var prompt_seen = false;
    var i: usize = 0;
    while (i < args.len) {
        const flag = args[i];
        const is_bool = isBooleanFlag(flag, include_raw);
        if (!is_bool and i + 1 >= args.len) break;
        applyEngineSelector(flag, &native);
        if (std.mem.eql(u8, flag, "--port")) port_seen = true;
        if (std.mem.eql(u8, flag, "--prompt")) prompt_seen = true;
        i += if (is_bool) 1 else 2;
    }
    return .{ .native = native, .port_seen = port_seen, .prompt_seen = prompt_seen };
}

test "serve mlx selector is valueless and terminal-safe" {
    const followed = scan(&.{ "--model", "model", "--mlx", "--port", "9001" }, false, true);
    try std.testing.expect(!followed.native);
    try std.testing.expect(followed.port_seen);
    const terminal = scan(&.{ "--model", "model", "--mlx" }, false, true);
    try std.testing.expect(!terminal.native);
}

test "chat mlx selector preserves following prompt" {
    const result = scan(&.{ "--model", "model", "--mlx", "--prompt", "hello", "--max-tokens", "30" }, true, false);
    try std.testing.expect(!result.native);
    try std.testing.expect(result.prompt_seen);
}

test "engine selectors use last-selector-wins semantics" {
    try std.testing.expect(scan(&.{ "--mlx", "--native" }, false, true).native);
    try std.testing.expect(!scan(&.{ "--native", "--mlx" }, false, true).native);
}
