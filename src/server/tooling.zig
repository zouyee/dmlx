/// Tool call detection utilities.
const std = @import("std");

fn detectToolCalls(allocator: std.mem.Allocator, text: []const u8) ?[]u8 {
    // Look for a JSON object pattern with "name" field
    const name_key = "\"name\"";
    const args_key = "\"arguments\"";

    const name_idx = std.mem.indexOf(u8, text, name_key) orelse return null;
    _ = std.mem.indexOf(u8, text, args_key) orelse return null;

    // Find the enclosing braces
    const brace_start = std.mem.lastIndexOf(u8, text[0..name_idx], "{") orelse return null;
    // Find matching closing brace (simple scan)
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
    const tool_json = text[brace_start..end];

    // Extract the function name from the detected JSON
    const name_start = (std.mem.indexOf(u8, tool_json, name_key) orelse return null) + name_key.len;
    const rest = std.mem.trimStart(u8, tool_json[name_start..], " \t\n:");
    if (rest.len == 0 or rest[0] != '"') return null;
    const fn_name_end = std.mem.indexOf(u8, rest[1..], "\"") orelse return null;
    const fn_name = rest[1 .. 1 + fn_name_end];

    // Extract arguments substring
    const args_start_idx = std.mem.indexOf(u8, tool_json, args_key) orelse return null;
    const args_rest = tool_json[args_start_idx + args_key.len ..];
    const args_colon = std.mem.indexOf(u8, args_rest, "{") orelse return null;
    var args_depth: i32 = 0;
    var args_end_idx: ?usize = null;
    for (args_rest[args_colon..], 0..) |ch, i| {
        if (ch == '{') args_depth += 1;
        if (ch == '}') {
            args_depth -= 1;
            if (args_depth == 0) {
                args_end_idx = args_colon + i + 1;
                break;
            }
        }
    }
    const arguments_str = if (args_end_idx) |ae| args_rest[args_colon..ae] else "{}";

    // Build tool_calls array JSON
    return std.fmt.allocPrint(allocator,
        \\[{{"id":"call_1","type":"function","function":{{"name":"{s}","arguments":"{s}"}}}}]
    , .{ fn_name, arguments_str }) catch return null;
}

// ------------------------------------------------------------------
// Anthropic Messages API types
// ------------------------------------------------------------------
