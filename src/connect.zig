/// `dmlx connect` — wire a local `dmlx serve` instance into agent harnesses
/// (Claude Code, Codex, aider, Cline).
///
/// `dmlx connect` lists supported harnesses and detects their CLIs.
/// `dmlx connect <name>` prints the exact configuration for one harness.
/// `dmlx connect <name> --apply` writes it (claude-code and codex only),
/// backing up any file it modifies.
const std = @import("std");

const Harness = enum {
    claude_code,
    codex,
    aider,
    cline,

    fn name(self: Harness) []const u8 {
        return switch (self) {
            .claude_code => "claude-code",
            .codex => "codex",
            .aider => "aider",
            .cline => "cline",
        };
    }

    fn cli(self: Harness) []const u8 {
        return switch (self) {
            .claude_code => "claude",
            .codex => "codex",
            .aider => "aider",
            .cline => "code",
        };
    }

    fn fromName(s: []const u8) ?Harness {
        inline for (@typeInfo(Harness).@"enum".fields) |f| {
            const h: Harness = @enumFromInt(f.value);
            if (std.mem.eql(u8, s, h.name())) return h;
        }
        return null;
    }
};

const ModelNote =
    \\
    \\Recommended server side (agent workloads want clean output, not
    \\reasoning traces):
    \\  DMLX_CHAT_EMPTY_THINK=1 dmlx serve --model <model> --port {d}
    \\
;

pub fn run(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !u8 {
    var apply = false;
    var port: u16 = 8080;
    var target: ?Harness = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--apply")) {
            apply = true;
        } else if (std.mem.eql(u8, a, "--port")) {
            i += 1;
            if (i >= args.len) return usage();
            port = std.fmt.parseInt(u16, args[i], 10) catch return usage();
        } else if (Harness.fromName(a)) |h| {
            target = h;
        } else {
            return usage();
        }
    }

    if (target) |h| {
        return try connectOne(allocator, io, h, port, apply);
    }

    std.debug.print("\ndmlx connect — use a local dmlx server from an agent harness\n\n", .{});
    std.debug.print("  {s:<13} {s:<10} {s}\n", .{ "harness", "cli", "detected" });
    inline for (@typeInfo(Harness).@"enum".fields) |f| {
        const h: Harness = @enumFromInt(f.value);
        const found = try which(allocator, io, h.cli());
        std.debug.print("  {s:<13} {s:<10} {s}\n", .{ h.name(), h.cli(), if (found) "yes" else "no" });
    }
    std.debug.print(
        \\
        \\Run 'dmlx connect <harness>' for the config snippet, add --apply to
        \\write it where supported (claude-code, codex).
        \\
    , .{});
    return 0;
}

fn usage() u8 {
    std.debug.print("usage: dmlx connect [claude-code|codex|aider|cline] [--apply] [--port <n>]\n", .{});
    return 1;
}

fn which(allocator: std.mem.Allocator, io: std.Io, exe: []const u8) !bool {
    const result = std.process.run(allocator, io, .{
        .argv = &.{ "which", exe },
        .stdout_limit = .limited(1 << 12),
    }) catch return false;
    allocator.free(result.stdout);
    allocator.free(result.stderr);
    return switch (result.term) {
        .exited => |code| code == 0,
        else => false,
    };
}

fn homeDir(buf: []u8) ![]const u8 {
    const home_c = std.c.getenv("HOME") orelse return error.NoHome;
    const home = std.mem.span(home_c);
    if (home.len >= buf.len) return error.NoHome;
    @memcpy(buf[0..home.len], home);
    return buf[0..home.len];
}

fn writeFileAll(io: std.Io, path: []const u8, data: []const u8) !void {
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, data);
}

fn connectOne(allocator: std.mem.Allocator, io: std.Io, h: Harness, port: u16, apply: bool) !u8 {
    var home_buf: [4096]u8 = undefined;
    const home = try homeDir(&home_buf);

    switch (h) {
        .claude_code => {
            std.debug.print(
                \\
                \\== claude-code ==
                \\Point Claude Code at the local server (Anthropic Messages API):
                \\  export ANTHROPIC_BASE_URL=http://127.0.0.1:{d}
                \\  export ANTHROPIC_AUTH_TOKEN=dummy
                \\  export ANTHROPIC_MODEL=qwen3.8-27b
                \\  export ANTHROPIC_SMALL_FAST_MODEL=qwen3.8-27b
                \\
            , .{port});
            if (apply) {
                const path = try std.fmt.allocPrint(allocator, "{s}/.claude/settings.json", .{home});
                defer allocator.free(path);
                try applyClaudeCode(allocator, io, path, port);
            } else {
                std.debug.print("  (or run with --apply to write these into ~/.claude/settings.json)\n", .{});
            }
        },
        .codex => {
            std.debug.print(
                \\
                \\== codex ==
                \\Codex config (~/.codex/config.toml) — dmlx speaks wire_api "chat":
                \\  [model_providers.dmlx]
                \\  name = "dmlx (local)"
                \\  base_url = "http://127.0.0.1:{d}/v1"
                \\  wire_api = "chat"
                \\
                \\Then run: codex --profile dmlx  (or set model_provider/model)
                \\Note: /v1/responses is not implemented yet, so wire_api must
                \\stay "chat".
                \\
            , .{port});
            if (apply) {
                const path = try std.fmt.allocPrint(allocator, "{s}/.codex/config.toml", .{home});
                defer allocator.free(path);
                try applyCodex(allocator, io, path, port);
            } else {
                std.debug.print("  (or run with --apply to append the provider to ~/.codex/config.toml)\n", .{});
            }
        },
        .aider => {
            std.debug.print(
                \\
                \\== aider ==
                \\  export OPENAI_API_BASE=http://127.0.0.1:{d}/v1
                \\  export OPENAI_API_KEY=dummy
                \\  aider --model openai/qwen3.8-27b
                \\
            , .{port});
        },
        .cline => {
            std.debug.print(
                \\
                \\== cline ==
                \\In the Cline settings UI choose "OpenAI Compatible" and set:
                \\  Base URL:  http://127.0.0.1:{d}/v1
                \\  API Key:   dummy
                \\  Model ID:  qwen3.8-27b
                \\  [x] Supports images: off   [x] Supports prompt cache: off
                \\
            , .{port});
        },
    }
    std.debug.print(ModelNote, .{port});
    return 0;
}

/// Merge ANTHROPIC_* env entries into ~/.claude/settings.json, backing up
/// the original file first.
fn applyClaudeCode(allocator: std.mem.Allocator, io: std.Io, path: []const u8, port: u16) !void {
    var root: std.json.Value = undefined;
    // Mutations of the parsed tree must use the parser's own arena: the
    // tree's internal buffers belong to it, and freeing them through any
    // other allocator corrupts the heap.
    var json_alloc = allocator;
    if (std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 20))) |text| {
        // NOTE: `root` borrows slices from `text`; the process exits right
        // after writing, so the buffer intentionally stays allocated.
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
        root = parsed.value;
        json_alloc = parsed.arena.allocator();
        const backup = try std.fmt.allocPrint(allocator, "{s}.bak-dmlx", .{path});
        defer allocator.free(backup);
        try writeFileAll(io, backup, text);
        std.debug.print("  backup: {s}\n", .{backup});
    } else |_| {
        if (std.fs.path.dirname(path)) |dir| try std.Io.Dir.cwd().createDirPath(io, dir);
        root = .{ .object = std.json.ObjectMap.empty };
    }
    if (root != .object) return error.InvalidSettings;

    var gop = try root.object.getOrPut(json_alloc, "env");
    if (!gop.found_existing or gop.value_ptr.* != .object) {
        gop.value_ptr.* = .{ .object = std.json.ObjectMap.empty };
    }
    const env = &gop.value_ptr.object;
    const base_url = try std.fmt.allocPrint(json_alloc, "http://127.0.0.1:{d}", .{port});
    try env.put(json_alloc, "ANTHROPIC_BASE_URL", .{ .string = base_url });
    try env.put(json_alloc, "ANTHROPIC_AUTH_TOKEN", .{ .string = "dummy" });
    try env.put(json_alloc, "ANTHROPIC_MODEL", .{ .string = "qwen3.8-27b" });
    try env.put(json_alloc, "ANTHROPIC_SMALL_FAST_MODEL", .{ .string = "qwen3.8-27b" });

    const out = try std.json.Stringify.valueAlloc(allocator, root, .{ .whitespace = .indent_2 });
    defer allocator.free(out);
    try writeFileAll(io, path, out);
    std.debug.print("  wrote: {s}\n", .{path});
}

/// Append the dmlx provider block to ~/.codex/config.toml if absent.
fn applyCodex(allocator: std.mem.Allocator, io: std.Io, path: []const u8, port: u16) !void {
    const block = try std.fmt.allocPrint(allocator,
        \\
        \\[model_providers.dmlx]
        \\name = "dmlx (local)"
        \\base_url = "http://127.0.0.1:{d}/v1"
        \\wire_api = "chat"
        \\
    , .{port});
    defer allocator.free(block);

    var existing: []const u8 = "";
    if (std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 20))) |text| {
        existing = text;
        if (std.mem.indexOf(u8, text, "[model_providers.dmlx]") != null) {
            allocator.free(text);
            std.debug.print("  already present: {s} (no changes)\n", .{path});
            return;
        }
        const backup = try std.fmt.allocPrint(allocator, "{s}.bak-dmlx", .{path});
        defer allocator.free(backup);
        try writeFileAll(io, backup, text);
        std.debug.print("  backup: {s}\n", .{backup});
    } else |_| {
        if (std.fs.path.dirname(path)) |dir| try std.Io.Dir.cwd().createDirPath(io, dir);
    }
    defer if (existing.len > 0) allocator.free(existing);

    const merged = try std.fmt.allocPrint(allocator, "{s}\n{s}", .{ existing, block });
    defer allocator.free(merged);
    try writeFileAll(io, path, merged);
    std.debug.print("  wrote: {s}\n", .{path});
    std.debug.print("  note: set 'model_provider = \"dmlx\"' and 'model = \"qwen3.8-27b\"' (or a [profiles.dmlx] section) to use it by default\n", .{});
}
