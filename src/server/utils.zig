/// Shared server utilities.
const std = @import("std");

/// Escape a string for JSON embedding.
///
/// Generated token text may contain partial UTF-8 sequences (byte-level BPE
/// tokens split at a max_tokens boundary). JSON must be valid Unicode, so
/// invalid bytes are replaced with U+FFFD instead of being copied verbatim.
pub fn jsonEscapeInto(list: *std.ArrayList(u8), allocator: std.mem.Allocator, text: []const u8) !void {
    var i: usize = 0;
    while (i < text.len) {
        const ch = text[i];
        i += 1;
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
                } else if (ch < 0x80) {
                    try list.append(allocator, ch);
                } else {
                    const seq_len = std.unicode.utf8ByteSequenceLength(ch) catch {
                        try list.appendSlice(allocator, "\xEF\xBF\xBD");
                        continue;
                    };
                    if (i - 1 + seq_len <= text.len) {
                        const seq = text[i - 1 ..][0..seq_len];
                        if (std.unicode.utf8Decode(seq)) |_| {
                            try list.appendSlice(allocator, seq);
                            i += seq_len - 1;
                        } else |_| {
                            try list.appendSlice(allocator, "\xEF\xBF\xBD");
                        }
                    } else {
                        try list.appendSlice(allocator, "\xEF\xBF\xBD");
                    }
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
