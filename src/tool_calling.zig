/// Tool Calling Parser — converts model-generated text into structured tool calls.
///
/// Supports multiple model families with different output conventions:
///   - Llama:  <|python_tag|>{"name": "...", "arguments": {...}}
///   - Qwen:   <tool_call>\n{...}\n</tool_call>
///   - DeepSeek: <｜tool｜>... (or similar custom tags)
///
/// The parser is model-family-agnostic at the top level — it tries each format
/// in order and returns the first successful parse.
const std = @import("std");

// ============================================================
// Public Types
// ============================================================

/// A single parsed tool call.
pub const ToolCall = struct {
    id: []const u8,
    type: []const u8,
    function: Function,

    pub const Function = struct {
        name: []const u8,
        arguments: []const u8, // JSON object string
    };
};

/// Parser result containing zero or more tool calls.
pub const ParseResult = struct {
    calls: []const ToolCall,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *const ParseResult) void {
        for (self.calls) |*call| {
            self.allocator.free(call.id);
            self.allocator.free(call.type);
            self.allocator.free(call.function.name);
            self.allocator.free(call.function.arguments);
        }
        self.allocator.free(self.calls);
    }
};

/// Model family hints which parsing strategy to prefer.
pub const ModelFamily = enum {
    llama,
    qwen,
    deepseek,
    unknown,
};

// ============================================================
// Top-level API
// ============================================================

/// Parse generated text for tool calls.
/// Tries all known formats and returns the first match.
/// Returns null if no tool calls are detected.
pub fn parse(allocator: std.mem.Allocator, text: []const u8, family: ModelFamily) !?ParseResult {
    switch (family) {
        .llama => {
            if (try parseLlamaFormat(allocator, text)) |result| return result;
            if (try parseQwenFormat(allocator, text)) |result| return result;
            if (try parseGenericJsonFormat(allocator, text)) |result| return result;
        },
        .qwen => {
            if (try parseQwenFormat(allocator, text)) |result| return result;
            if (try parseLlamaFormat(allocator, text)) |result| return result;
            if (try parseGenericJsonFormat(allocator, text)) |result| return result;
        },
        .deepseek => {
            if (try parseDeepSeekFormat(allocator, text)) |result| return result;
            if (try parseGenericJsonFormat(allocator, text)) |result| return result;
        },
        .unknown => {
            if (try parseLlamaFormat(allocator, text)) |result| return result;
            if (try parseQwenFormat(allocator, text)) |result| return result;
            if (try parseDeepSeekFormat(allocator, text)) |result| return result;
            if (try parseGenericJsonFormat(allocator, text)) |result| return result;
        },
    }
    return null;
}

/// Detect model family from model name string.
pub fn detectFamily(model_name: []const u8) ModelFamily {
    // Check case-insensitively without allocating
    const Check = struct {
        fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
            var h_i: usize = 0;
            while (h_i + needle.len <= haystack.len) : (h_i += 1) {
                var match = true;
                for (needle, 0..) |n_ch, n_i| {
                    if (std.ascii.toLower(haystack[h_i + n_i]) != std.ascii.toLower(n_ch)) {
                        match = false;
                        break;
                    }
                }
                if (match) return true;
            }
            return false;
        }
    };
    if (Check.containsIgnoreCase(model_name, "llama") or
        Check.containsIgnoreCase(model_name, "mistral"))
        return .llama;
    if (Check.containsIgnoreCase(model_name, "qwen"))
        return .qwen;
    if (Check.containsIgnoreCase(model_name, "deepseek"))
        return .deepseek;
    return .unknown;
}

// ============================================================
// Llama Format: <|python_tag|>{"name": "...", "arguments": {...}}
// ============================================================

fn parseLlamaFormat(allocator: std.mem.Allocator, text: []const u8) !?ParseResult {
    const tag = "<|python_tag|>";
    const tag_idx = std.mem.indexOf(u8, text, tag) orelse return null;
    const json_start = tag_idx + tag.len;
    const json_text = std.mem.trimStart(u8, text[json_start..], " \t\n\r");

    return try parseSingleToolCallJson(allocator, json_text);
}

// ============================================================
// Qwen Format: <tool_call>\n{...}\n</tool_call>
// ============================================================

fn parseQwenFormat(allocator: std.mem.Allocator, text: []const u8) !?ParseResult {
    const open_tag = "<tool_call>";
    const close_tag = "</tool_call>";

    var calls = std.ArrayList(ToolCall).empty;
    errdefer {
        for (calls.items) |*call| {
            allocator.free(call.id);
            allocator.free(call.type);
            allocator.free(call.function.name);
            allocator.free(call.function.arguments);
        }
        calls.deinit(allocator);
    }

    var search_start: usize = 0;
    var call_idx: usize = 1;
    while (search_start < text.len) {
        const open_idx = std.mem.indexOfPos(u8, text, search_start, open_tag) orelse break;
        const content_start = open_idx + open_tag.len;
        const close_idx = std.mem.indexOfPos(u8, text, content_start, close_tag) orelse break;

        const content = std.mem.trim(u8, text[content_start..close_idx], " \t\n\r");
        if (content.len > 0) {
            if (try parseToolCallObject(allocator, content, call_idx)) |call| {
                try calls.append(allocator, call);
                call_idx += 1;
            }
        }
        search_start = close_idx + close_tag.len;
    }

    if (calls.items.len == 0) {
        calls.deinit(allocator);
        return null;
    }

    return ParseResult{
        .calls = try calls.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

// ============================================================
// DeepSeek Format: <｜tool｜>{...}  (or similar)
// ============================================================

fn parseDeepSeekFormat(allocator: std.mem.Allocator, text: []const u8) !?ParseResult {
    // DeepSeek uses various tool markers; try common ones
    const markers = [_][]const u8{
        "<｜tool｜>",
        "<|tool|>",
        "### Tool:",
    };

    for (markers) |marker| {
        const marker_idx = std.mem.indexOf(u8, text, marker) orelse continue;
        const after = std.mem.trimStart(u8, text[marker_idx + marker.len..], " \t\n\r");
        if (try parseSingleToolCallJson(allocator, after)) |result| return result;
    }

    return null;
}

// ============================================================
// Generic JSON Format: {"name": "...", "arguments": {...}}
// ============================================================

fn parseGenericJsonFormat(allocator: std.mem.Allocator, text: []const u8) !?ParseResult {
    // Look for JSON object with "name" and "arguments" keys
    const name_key = "\"name\"";
    if (std.mem.indexOf(u8, text, name_key) == null) return null;
    const args_key = "\"arguments\"";
    if (std.mem.indexOf(u8, text, args_key) == null) return null;

    // Try to extract the outermost JSON object
    const brace_start = std.mem.indexOf(u8, text, "{") orelse return null;
    var depth: i32 = 0;
    var brace_end: ?usize = null;
    for (text[brace_start..], 0..) |ch, i| {
        if (ch == '{') depth += 1;
        if (ch == '}') {
            depth -= 1;
            if (depth == 0) {
                brace_end = brace_start + i + 1;
                break;
            }
        }
    }
    const end = brace_end orelse return null;
    return try parseSingleToolCallJson(allocator, text[brace_start..end]);
}

// ============================================================
// JSON Object Parsing Helpers
// ============================================================

fn parseSingleToolCallJson(allocator: std.mem.Allocator, json_text: []const u8) !?ParseResult {
    var calls = std.ArrayList(ToolCall).empty;
    errdefer {
        for (calls.items) |*call| {
            allocator.free(call.id);
            allocator.free(call.type);
            allocator.free(call.function.name);
            allocator.free(call.function.arguments);
        }
        calls.deinit(allocator);
    }

    if (try parseToolCallObject(allocator, json_text, 1)) |call| {
        try calls.append(allocator, call);
    }

    if (calls.items.len == 0) {
        calls.deinit(allocator);
        return null;
    }

    return ParseResult{
        .calls = try calls.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

fn parseToolCallObject(allocator: std.mem.Allocator, json_text: []const u8, idx: usize) !?ToolCall {
    // Extract "name" value
    const fn_name = try extractJsonStringValue(allocator, json_text, "name") orelse return null;
    errdefer allocator.free(fn_name);

    // Extract "arguments" value (should be a JSON object string)
    const args = try extractJsonStringValue(allocator, json_text, "arguments") orelse
        try extractJsonObjectValue(allocator, json_text, "arguments") orelse
        try allocator.dupe(u8, "{}");
    errdefer allocator.free(args);

    const id = try std.fmt.allocPrint(allocator, "call_{d}", .{idx});
    errdefer allocator.free(id);

    const type_str = try allocator.dupe(u8, "function");
    errdefer allocator.free(type_str);

    return ToolCall{
        .id = id,
        .type = type_str,
        .function = .{
            .name = fn_name,
            .arguments = args,
        },
    };
}

/// Extract a string value for a given key from JSON text.
/// Very simple parser — assumes `"key": "value"` format.
fn extractJsonStringValue(allocator: std.mem.Allocator, text: []const u8, key: []const u8) !?[]u8 {
    const key_pattern = try std.fmt.allocPrint(allocator, "\"{s}\"", .{key});
    defer allocator.free(key_pattern);

    const key_idx = std.mem.indexOf(u8, text, key_pattern) orelse return null;
    const after_key = text[key_idx + key_pattern.len..];

    // Skip whitespace and colon
    var val_start: usize = 0;
    while (val_start < after_key.len and (after_key[val_start] == ' ' or after_key[val_start] == '\t' or after_key[val_start] == '\n' or after_key[val_start] == '\r' or after_key[val_start] == ':')) {
        val_start += 1;
    }
    if (val_start >= after_key.len or after_key[val_start] != '"') return null;

    // Find closing quote (skip escaped quotes)
    var pos = val_start + 1;
    while (pos < after_key.len) : (pos += 1) {
        if (after_key[pos] == '"' and after_key[pos - 1] != '\\') break;
    }
    if (pos >= after_key.len) return null;

    return try allocator.dupe(u8, after_key[val_start + 1 .. pos]);
}

/// Extract a JSON object value for a given key.
fn extractJsonObjectValue(allocator: std.mem.Allocator, text: []const u8, key: []const u8) !?[]u8 {
    const key_pattern = try std.fmt.allocPrint(allocator, "\"{s}\"", .{key});
    defer allocator.free(key_pattern);

    const key_idx = std.mem.indexOf(u8, text, key_pattern) orelse return null;
    const after_key = text[key_idx + key_pattern.len..];

    // Skip whitespace and colon
    var val_start: usize = 0;
    while (val_start < after_key.len and (after_key[val_start] == ' ' or after_key[val_start] == '\t' or after_key[val_start] == '\n' or after_key[val_start] == '\r' or after_key[val_start] == ':')) {
        val_start += 1;
    }
    if (val_start >= after_key.len or after_key[val_start] != '{') return null;

    // Find matching brace
    var depth: i32 = 0;
    var pos = val_start;
    while (pos < after_key.len) : (pos += 1) {
        if (after_key[pos] == '{') depth += 1;
        if (after_key[pos] == '}') {
            depth -= 1;
            if (depth == 0) break;
        }
    }
    if (depth != 0) return null;

    return try allocator.dupe(u8, after_key[val_start .. pos + 1]);
}

// ============================================================
// Prompt Building — inject tool definitions into system message
// ============================================================

/// Build a tool-calling system prompt from tool definitions.
/// Caller owns returned string.
pub fn buildToolSystemPrompt(
    allocator: std.mem.Allocator,
    tools: []const ToolDefinition,
    model_family: ModelFamily,
) ![]u8 {
    switch (model_family) {
        .llama => return try buildLlamaToolPrompt(allocator, tools),
        .qwen => return try buildQwenToolPrompt(allocator, tools),
        .deepseek => return try buildDeepSeekToolPrompt(allocator, tools),
        .unknown => return try buildGenericToolPrompt(allocator, tools),
    }
}

pub const ToolDefinition = struct {
    name: []const u8,
    description: ?[]const u8 = null,
    parameters: ?[]const u8 = null, // JSON schema string
};

fn buildLlamaToolPrompt(allocator: std.mem.Allocator, tools: []const ToolDefinition) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    try buf.appendSlice(allocator,
        \\You are a helpful assistant with tool calling capabilities.
        \\When you need to use a tool, output a JSON object in this format:
        \\  {"name": "<tool_name>", "arguments": {<args>}}
        \\Available tools:
        \\
    );

    for (tools) |tool| {
        try buf.appendSlice(allocator, "- ");
        try buf.appendSlice(allocator, tool.name);
        if (tool.description) |desc| {
            try buf.appendSlice(allocator, ": ");
            try buf.appendSlice(allocator, desc);
        }
        try buf.appendSlice(allocator, "\n");
        if (tool.parameters) |params| {
            try buf.appendSlice(allocator, "  Parameters: ");
            try buf.appendSlice(allocator, params);
            try buf.appendSlice(allocator, "\n");
        }
    }

    return try buf.toOwnedSlice(allocator);
}

fn buildQwenToolPrompt(allocator: std.mem.Allocator, tools: []const ToolDefinition) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    try buf.appendSlice(allocator,
        \\You are a helpful assistant with tool calling capabilities.
        \\When you need to use a tool, wrap the tool call in <tool_call> tags:
        \\  <tool_call>
        \\  {"name": "<tool_name>", "arguments": {<args>}}
        \\  </tool_call>
        \\Available tools:
        \\
    );

    for (tools) |tool| {
        try buf.appendSlice(allocator, "- ");
        try buf.appendSlice(allocator, tool.name);
        if (tool.description) |desc| {
            try buf.appendSlice(allocator, ": ");
            try buf.appendSlice(allocator, desc);
        }
        try buf.appendSlice(allocator, "\n");
        if (tool.parameters) |params| {
            try buf.appendSlice(allocator, "  Parameters: ");
            try buf.appendSlice(allocator, params);
            try buf.appendSlice(allocator, "\n");
        }
    }

    return try buf.toOwnedSlice(allocator);
}

fn buildDeepSeekToolPrompt(allocator: std.mem.Allocator, tools: []const ToolDefinition) ![]u8 {
    // DeepSeek uses a similar format to Llama
    return try buildLlamaToolPrompt(allocator, tools);
}

fn buildGenericToolPrompt(allocator: std.mem.Allocator, tools: []const ToolDefinition) ![]u8 {
    return try buildLlamaToolPrompt(allocator, tools);
}

// ============================================================
// Unit Tests
// ============================================================

test "parse Llama format" {
    const allocator = std.testing.allocator;
    const text = "Some text <|python_tag|>{\"name\": \"file_read\", \"arguments\": {\"path\": \"/tmp/test.txt\"}}";

    const result = try parse(allocator, text, .llama);
    try std.testing.expect(result != null);
    defer if (result) |*r| r.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.?.calls.len);
    try std.testing.expectEqualStrings("file_read", result.?.calls[0].function.name);
}

test "parse Qwen format" {
    const allocator = std.testing.allocator;
    const text =
        \\Let me check.
        \\<tool_call>
        \\{"name": "calculator", "arguments": {"expr": "1+1"}}
        \\</tool_call>
    ;

    const result = try parse(allocator, text, .qwen);
    try std.testing.expect(result != null);
    defer if (result) |*r| r.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.?.calls.len);
    try std.testing.expectEqualStrings("calculator", result.?.calls[0].function.name);
}

test "parse generic JSON format" {
    const allocator = std.testing.allocator;
    const text = "{\"name\": \"shell_exec\", \"arguments\": {\"command\": \"ls\"}}";

    const result = try parse(allocator, text, .unknown);
    try std.testing.expect(result != null);
    defer if (result) |*r| r.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.?.calls.len);
    try std.testing.expectEqualStrings("shell_exec", result.?.calls[0].function.name);
}

test "buildToolSystemPrompt" {
    const allocator = std.testing.allocator;
    const tools = &[_]ToolDefinition{
        .{ .name = "file_read", .description = "Read a file", .parameters = "{\"path\": {\"type\": \"string\"}}" },
    };

    const prompt = try buildToolSystemPrompt(allocator, tools, .llama);
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "file_read") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "Read a file") != null);
}

test "detectFamily" {
    try std.testing.expectEqual(ModelFamily.llama, detectFamily("Llama-3-8B"));
    try std.testing.expectEqual(ModelFamily.qwen, detectFamily("Qwen2.5-7B"));
    try std.testing.expectEqual(ModelFamily.deepseek, detectFamily("DeepSeek-V3"));
    try std.testing.expectEqual(ModelFamily.unknown, detectFamily("Unknown"));
}
