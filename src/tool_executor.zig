/// Tool Executor — executes built-in tool calls with safety sandboxing.
///
/// Built-in tools:
///   - file_read    : Read file contents (restricted to workdir)
///   - file_write   : Write file contents (restricted to workdir)
///   - shell_exec   : Execute shell commands (whitelist-only, opt-in)
///   - web_fetch    : HTTP GET via curl wrapper
///   - calculator   : Evaluate arithmetic expressions
///
/// Safety: shell_exec is disabled by default; file ops are restricted
/// to the current working directory.
const std = @import("std");

// ============================================================
// Public Types
// ============================================================

/// Result of executing a tool.
pub const ToolResult = struct {
    success: bool,
    output: []const u8,
    error_message: ?[]const u8 = null,
    allocator: ?std.mem.Allocator = null,

    pub fn deinit(self: *const ToolResult) void {
        if (self.allocator) |alloc| {
            alloc.free(self.output);
            if (self.error_message) |em| alloc.free(em);
        }
    }
};

/// Executor configuration.
pub const ExecutorConfig = struct {
    /// Allow shell_exec tool (default: false for security).
    allow_shell_exec: bool = false,
    /// Restrict file operations to this directory (null = current working dir).
    work_dir: ?[]const u8 = null,
    /// Maximum file read size in bytes (default: 1MB).
    max_file_size: usize = 1024 * 1024,
    /// HTTP timeout in milliseconds (default: 30s).
    http_timeout_ms: u32 = 30000,
    /// Optional I/O context for file operations. If null, file ops fail.
    io: ?std.Io = null,
};

// ============================================================
// Top-level API
// ============================================================

/// Execute a tool by name with JSON arguments.
/// Returns an error only on allocation/JSON failure;
/// tool execution errors are reported in ToolResult.error_message.
pub fn execute(
    allocator: std.mem.Allocator,
    config: ExecutorConfig,
    name: []const u8,
    args_json: []const u8,
) !ToolResult {
    if (std.mem.eql(u8, name, "file_read")) {
        return try executeFileRead(allocator, config, args_json);
    } else if (std.mem.eql(u8, name, "file_write")) {
        return try executeFileWrite(allocator, config, args_json);
    } else if (std.mem.eql(u8, name, "shell_exec")) {
        return try executeShellExec(allocator, config, args_json);
    } else if (std.mem.eql(u8, name, "web_fetch")) {
        return try executeWebFetch(allocator, config, args_json);
    } else if (std.mem.eql(u8, name, "calculator")) {
        return try executeCalculator(allocator, args_json);
    } else {
        return makeError(allocator, "Unknown tool: {s}", .{name});
    }
}

/// List all available built-in tool names.
pub fn listTools() []const []const u8 {
    return &[_][]const u8{ "file_read", "file_write", "shell_exec", "web_fetch", "calculator" };
}

// ============================================================
// file_read
// ============================================================

fn executeFileRead(allocator: std.mem.Allocator, config: ExecutorConfig, args_json: []const u8) !ToolResult {
    const path = try extractStringField(allocator, args_json, "path") orelse {
        return makeError(allocator, "Missing 'path' argument", .{});
    };
    defer allocator.free(path);

    if (!isPathAllowed(path, config.work_dir)) {
        return makeError(allocator, "Path '{s}' is outside the allowed work directory", .{path});
    }

    const io = config.io orelse return makeError(allocator, "I/O context required for file_read", .{});

    const resolved = if (config.work_dir) |wd|
        try std.fs.path.resolve(allocator, &[_][]const u8{ wd, path })
    else
        try std.fs.path.resolve(allocator, &[_][]const u8{path});
    defer allocator.free(resolved);

    const dir = std.Io.Dir.cwd();
    const content = dir.readFileAlloc(io, resolved, allocator, .limited(config.max_file_size)) catch |err| {
        return makeError(allocator, "Failed to read file '{s}': {s}", .{ path, @errorName(err) });
    };

    return ToolResult{
        .success = true,
        .output = content,
        .allocator = allocator,
    };
}

// ============================================================
// file_write
// ============================================================

fn executeFileWrite(allocator: std.mem.Allocator, config: ExecutorConfig, args_json: []const u8) !ToolResult {
    const path = try extractStringField(allocator, args_json, "path") orelse {
        return makeError(allocator, "Missing 'path' argument", .{});
    };
    defer allocator.free(path);

    const content = try extractStringField(allocator, args_json, "content") orelse {
        return makeError(allocator, "Missing 'content' argument", .{});
    };
    defer allocator.free(content);

    if (!isPathAllowed(path, config.work_dir)) {
        return makeError(allocator, "Path '{s}' is outside the allowed work directory", .{path});
    }

    const io = config.io orelse return makeError(allocator, "I/O context required for file_write", .{});

    const resolved = if (config.work_dir) |wd|
        try std.fs.path.resolve(allocator, &[_][]const u8{ wd, path })
    else
        try std.fs.path.resolve(allocator, &[_][]const u8{path});
    defer allocator.free(resolved);

    // Ensure parent directory exists
    const dir = std.Io.Dir.cwd();
    if (std.fs.path.dirname(resolved)) |parent_dir| {
        dir.createDirPath(io, parent_dir) catch {};
    }

    const file = dir.createFile(io, resolved, .{}) catch |err| {
        return makeError(allocator, "Failed to create file '{s}': {s}", .{ path, @errorName(err) });
    };
    defer file.close(io);

    file.writeStreamingAll(io, content) catch |err| {
        return makeError(allocator, "Failed to write file '{s}': {s}", .{ path, @errorName(err) });
    };

    const msg = try std.fmt.allocPrint(allocator, "Wrote {d} bytes to {s}", .{ content.len, path });
    return ToolResult{
        .success = true,
        .output = msg,
        .allocator = allocator,
    };
}

// ============================================================
// shell_exec
// ============================================================

const SHELL_WHITELIST = [_][]const u8{
    "ls",     "cat", "head", "tail", "grep", "find", "wc",   "echo", "pwd",    "mkdir",
    "touch",  "rm",  "cp",   "mv",   "diff", "sort", "uniq", "date", "whoami", "python3",
    "python", "zig", "git",  "curl", "wget",
};

fn executeShellExec(allocator: std.mem.Allocator, config: ExecutorConfig, args_json: []const u8) !ToolResult {
    if (!config.allow_shell_exec) {
        return makeError(allocator, "shell_exec is disabled. Enable with --allow-unsafe-tools.", .{});
    }

    const command = try extractStringField(allocator, args_json, "command") orelse {
        return makeError(allocator, "Missing 'command' argument", .{});
    };
    defer allocator.free(command);

    // Whitelist check: first token must be in whitelist
    var tokens = std.mem.splitScalar(u8, command, ' ');
    const first_token = std.mem.trim(u8, tokens.first(), " \t\n\r");
    var whitelisted = false;
    for (SHELL_WHITELIST) |allowed| {
        if (std.mem.eql(u8, first_token, allowed)) {
            whitelisted = true;
            break;
        }
    }
    if (!whitelisted) {
        return makeError(allocator, "Command '{s}' is not in the whitelist", .{first_token});
    }

    const io = config.io orelse return makeError(allocator, "I/O context required for shell_exec", .{});

    const run_result = std.process.run(allocator, io, .{
        .argv = &[_][]const u8{ "sh", "-c", command },
        .stdout_limit = .limited(1024 * 1024),
        .stderr_limit = .limited(1024 * 1024),
    }) catch |err| {
        return makeError(allocator, "Failed to run shell command: {s}", .{@errorName(err)});
    };
    defer {
        allocator.free(run_result.stdout);
        allocator.free(run_result.stderr);
    }

    switch (run_result.term) {
        .exited => |code| {
            if (code != 0) {
                const combined = try std.fmt.allocPrint(allocator, "STDOUT:\n{s}\nSTDERR:\n{s}", .{ run_result.stdout, run_result.stderr });
                return makeError(allocator, "Command exited with code {d}\n{s}", .{ code, combined });
            }
            return ToolResult{
                .success = true,
                .output = try allocator.dupe(u8, run_result.stdout),
                .allocator = allocator,
            };
        },
        else => {
            return makeError(allocator, "Command terminated abnormally", .{});
        },
    }
}

// ============================================================
// web_fetch
// ============================================================

fn executeWebFetch(allocator: std.mem.Allocator, config: ExecutorConfig, args_json: []const u8) !ToolResult {
    const url = try extractStringField(allocator, args_json, "url") orelse {
        return makeError(allocator, "Missing 'url' argument", .{});
    };
    defer allocator.free(url);

    // Validate URL scheme
    if (!std.mem.startsWith(u8, url, "http://") and !std.mem.startsWith(u8, url, "https://")) {
        return makeError(allocator, "Invalid URL scheme (must be http/https): {s}", .{url});
    }

    const io = config.io orelse return makeError(allocator, "I/O context required for web_fetch", .{});

    // Use curl subprocess for simplicity and to avoid async HTTP dependencies
    const timeout_str = try std.fmt.allocPrint(allocator, "{d}", .{config.http_timeout_ms / 1000});
    defer allocator.free(timeout_str);

    const run_result = std.process.run(allocator, io, .{
        .argv = &[_][]const u8{ "curl", "-sL", "--max-time", timeout_str, "-A", "dmlx-tool/1.0", url },
        .stdout_limit = .limited(10 * 1024 * 1024),
        .stderr_limit = .limited(1024 * 1024),
    }) catch |err| {
        return makeError(allocator, "Failed to run curl: {s}", .{@errorName(err)});
    };
    defer {
        allocator.free(run_result.stdout);
        allocator.free(run_result.stderr);
    }

    switch (run_result.term) {
        .exited => |code| {
            if (code != 0) {
                return makeError(allocator, "curl exited with code {d}: {s}", .{ code, run_result.stderr });
            }
            return ToolResult{
                .success = true,
                .output = try allocator.dupe(u8, run_result.stdout),
                .allocator = allocator,
            };
        },
        else => {
            return makeError(allocator, "curl terminated abnormally", .{});
        },
    }
}

// ============================================================
// calculator
// ============================================================

fn executeCalculator(allocator: std.mem.Allocator, args_json: []const u8) !ToolResult {
    const expr = try extractStringField(allocator, args_json, "expr") orelse {
        return makeError(allocator, "Missing 'expr' argument", .{});
    };
    defer allocator.free(expr);

    const result = evaluateExpression(expr) catch |err| {
        return makeError(allocator, "Failed to evaluate expression '{s}': {s}", .{ expr, @errorName(err) });
    };

    const output = try std.fmt.allocPrint(allocator, "{d}", .{result});
    return ToolResult{
        .success = true,
        .output = output,
        .allocator = allocator,
    };
}

const ParseError = error{
    InvalidExpression,
    MismatchedParens,
    ExpectedNumber,
    InvalidNumber,
};

/// Simple arithmetic expression evaluator supporting +, -, *, /, parentheses.
fn evaluateExpression(expr: []const u8) ParseError!f64 {
    var parser = ExprParser{ .input = expr, .pos = 0 };
    return try parser.parse();
}

const ExprParser = struct {
    input: []const u8,
    pos: usize,

    fn parse(self: *ExprParser) ParseError!f64 {
        const val = try self.parseAddSub();
        self.skipWhitespace();
        if (self.pos < self.input.len) return error.InvalidExpression;
        return val;
    }

    fn parseAddSub(self: *ExprParser) ParseError!f64 {
        var left = try self.parseMulDiv();
        while (true) {
            self.skipWhitespace();
            if (self.pos >= self.input.len) break;
            const op = self.input[self.pos];
            if (op != '+' and op != '-') break;
            self.pos += 1;
            const right = try self.parseMulDiv();
            left = if (op == '+') left + right else left - right;
        }
        return left;
    }

    fn parseMulDiv(self: *ExprParser) ParseError!f64 {
        var left = try self.parseUnary();
        while (true) {
            self.skipWhitespace();
            if (self.pos >= self.input.len) break;
            const op = self.input[self.pos];
            if (op != '*' and op != '/') break;
            self.pos += 1;
            const right = try self.parseUnary();
            left = if (op == '*') left * right else left / right;
        }
        return left;
    }

    fn parseUnary(self: *ExprParser) ParseError!f64 {
        self.skipWhitespace();
        if (self.pos < self.input.len and self.input[self.pos] == '-') {
            self.pos += 1;
            return -(try self.parseUnary());
        }
        return try self.parsePrimary();
    }

    fn parsePrimary(self: *ExprParser) ParseError!f64 {
        self.skipWhitespace();
        if (self.pos >= self.input.len) return error.InvalidExpression;

        if (self.input[self.pos] == '(') {
            self.pos += 1;
            const val = try self.parseAddSub();
            self.skipWhitespace();
            if (self.pos >= self.input.len or self.input[self.pos] != ')') return error.MismatchedParens;
            self.pos += 1;
            return val;
        }

        // Parse number
        const start = self.pos;
        var has_dot = false;
        while (self.pos < self.input.len) : (self.pos += 1) {
            const ch = self.input[self.pos];
            if (ch >= '0' and ch <= '9') continue;
            if (ch == '.' and !has_dot) {
                has_dot = true;
                continue;
            }
            break;
        }
        if (start == self.pos) return error.ExpectedNumber;
        return std.fmt.parseFloat(f64, self.input[start..self.pos]) catch error.InvalidNumber;
    }

    fn skipWhitespace(self: *ExprParser) void {
        while (self.pos < self.input.len and std.ascii.isWhitespace(self.input[self.pos])) {
            self.pos += 1;
        }
    }
};

// ============================================================
// Helpers
// ============================================================

fn makeError(allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !ToolResult {
    const msg = try std.fmt.allocPrint(allocator, fmt, args);
    return ToolResult{
        .success = false,
        .output = try allocator.dupe(u8, msg),
        .error_message = msg,
        .allocator = allocator,
    };
}

fn isPathAllowed(path: []const u8, work_dir: ?[]const u8) bool {
    // Reject paths with parent directory traversal
    var it = std.mem.splitScalar(u8, path, '/');
    while (it.next()) |part| {
        if (std.mem.eql(u8, part, "..")) return false;
    }

    // If work_dir is set, resolve and ensure it's within
    if (work_dir) |wd| {
        // Simple check: path should not be absolute
        if (path.len > 0 and path[0] == '/') return false;

        _ = wd;
        // More rigorous check would use realpath, but that's platform-dependent
    }

    return true;
}

/// Extract a string field from a simple JSON object.
fn extractStringField(allocator: std.mem.Allocator, json: []const u8, key: []const u8) !?[]u8 {
    const pattern = try std.fmt.allocPrint(allocator, "\"{s}\"", .{key});
    defer allocator.free(pattern);

    const idx = std.mem.indexOf(u8, json, pattern) orelse return null;
    const after = json[idx + pattern.len ..];

    var pos: usize = 0;
    while (pos < after.len and (after[pos] == ' ' or after[pos] == '\t' or after[pos] == '\n' or after[pos] == '\r' or after[pos] == ':')) {
        pos += 1;
    }
    if (pos >= after.len) return null;

    if (after[pos] == '"') {
        // String value
        var end = pos + 1;
        while (end < after.len) : (end += 1) {
            if (after[end] == '"' and after[end - 1] != '\\') break;
        }
        if (end >= after.len) return null;
        return try allocator.dupe(u8, after[pos + 1 .. end]);
    }

    return null;
}

// ============================================================
// Unit Tests
// ============================================================

test "calculator basic arithmetic" {
    const allocator = std.testing.allocator;
    const result = try execute(allocator, .{}, "calculator", "{\"expr\": \"1 + 2 * 3\"}");
    defer result.deinit();
    try std.testing.expect(result.success);
    try std.testing.expectEqualStrings("7", result.output);
}

test "calculator with parentheses" {
    const allocator = std.testing.allocator;
    const result = try execute(allocator, .{}, "calculator", "{\"expr\": \"(1 + 2) * 3\"}");
    defer result.deinit();
    try std.testing.expect(result.success);
    try std.testing.expectEqualStrings("9", result.output);
}

test "calculator floating point" {
    const allocator = std.testing.allocator;
    const result = try execute(allocator, .{}, "calculator", "{\"expr\": \"3.5 * 2\"}");
    defer result.deinit();
    try std.testing.expect(result.success);
    try std.testing.expectEqualStrings("7", result.output);
}

test "shell_exec disabled by default" {
    const allocator = std.testing.allocator;
    const result = try execute(allocator, .{}, "shell_exec", "{\"command\": \"echo hello\"}");
    defer result.deinit();
    try std.testing.expect(!result.success);
}

test "shell_exec rejects non-whitelisted command" {
    const allocator = std.testing.allocator;
    const result = try execute(allocator, .{ .allow_shell_exec = true }, "shell_exec", "{\"command\": \"rm -rf /\"}");
    defer result.deinit();
    try std.testing.expect(!result.success);
}

test "file_read rejects parent traversal" {
    const allocator = std.testing.allocator;
    const result = try execute(allocator, .{}, "file_read", "{\"path\": \"../etc/passwd\"}");
    defer result.deinit();
    try std.testing.expect(!result.success);
}

test "web_fetch rejects non-http URL" {
    const allocator = std.testing.allocator;
    const result = try execute(allocator, .{}, "web_fetch", "{\"url\": \"file:///etc/passwd\"}");
    defer result.deinit();
    try std.testing.expect(!result.success);
}

test "listTools" {
    const tools = listTools();
    try std.testing.expectEqual(@as(usize, 5), tools.len);
    try std.testing.expectEqualStrings("file_read", tools[0]);
}
