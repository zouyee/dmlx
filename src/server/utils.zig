/// Shared server utilities.
const std = @import("std");

/// Escape a string for JSON embedding.
pub fn jsonEscapeInto(list: *std.ArrayList(u8), allocator: std.mem.Allocator, text: []const u8) !void {
    for (text) |ch| {
        switch (ch) {
            '"' => try list.appendSlice(allocator, "\\\""),
            '\\' => try list.appendSlice(allocator, "\\\\"),
            0x08 => try list.appendSlice(allocator, "\\b"),
            0x0C => try list.appendSlice(allocator, "\\f"),
            '\n' => try list.appendSlice(allocator, "\\n"),
            '\r' => try list.appendSlice(allocator, "\\r"),
            '\t' => try list.appendSlice(allocator, "\\t"),
            else => {
                if (ch < 0x20) {
                    var buf: [6]u8 = undefined;
                    const s = try std.fmt.bufPrint(&buf, "\\u{x:0>4}", .{ch});
                    try list.appendSlice(allocator, s);
                } else {
                    try list.append(allocator, ch);
                }
            },
        }
    }
}

/// Detect the architecture name from config.json's "architectures" field.
pub fn detectArchitecture(config_json: []const u8) []const u8 {
    const key = "\"architectures\"";
    if (std.mem.indexOf(u8, config_json, key)) |idx| {
        const rest = config_json[idx + key.len ..];
        // Find first quoted string inside the array
        if (std.mem.indexOf(u8, rest, "\"")) |q1| {
            const after_q1 = rest[q1 + 1 ..];
            if (std.mem.indexOf(u8, after_q1, "\"")) |q2| {
                return after_q1[0..q2];
            }
        }
    }
    // Fallback: detect via model_type
    const mt_key = "\"model_type\":";
    if (std.mem.indexOf(u8, config_json, mt_key)) |idx| {
        const mt_start = idx + mt_key.len;
        const rest2 = std.mem.trimStart(u8, config_json[mt_start..], " \n\t");
        if (rest2.len > 0 and rest2[0] == '"') {
            if (std.mem.indexOf(u8, rest2[1..], "\"")) |end| {
                const model_type = rest2[1 .. 1 + end];
                if (std.mem.eql(u8, model_type, "deepseek_v4")) return "DeepseekV4ForCausalLM";
            }
        }
    }
    return "LlamaForCausalLM";
}

/// Keep-alive threshold in milliseconds (5 seconds).
pub const keep_alive_threshold_ms: i64 = 5000;
