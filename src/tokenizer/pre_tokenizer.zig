/// Pre-tokenizers for HuggingFace TokenizerFast format.
///
/// Supports:
///   - Sequence: compose multiple pre-tokenizers in order
///   - Split: regex-based splitting (specialized for DeepSeek/Llama-3 patterns)
///   - ByteLevel: byte-to-Unicode mapping (GPT-2 style)
///
/// No external regex library is used; instead, each Split pattern has a
/// dedicated matcher that understands the exact Unicode categories needed.
const std = @import("std");

// ============================================================================
// Byte-level encoding (GPT-2 / DeepSeek)
// ============================================================================

/// Build the 256-byte → Unicode codepoint table used by ByteLevel BPE.
/// Mirrors the Python `bytes_to_unicode()` in transformers.
fn initBytesToUnicode() [256]u21 {
    var result: [256]u21 = undefined;

    // Bytes that keep their original value
    var bs: [256]u8 = undefined;
    var bs_len: usize = 0;

    for (0x21..0x7F) |b| {
        bs[bs_len] = @intCast(b);
        bs_len += 1;
    }
    for (0xA1..0xAD) |b| {
        bs[bs_len] = @intCast(b);
        bs_len += 1;
    }
    for (0xAE..0x100) |b| {
        bs[bs_len] = @intCast(b);
        bs_len += 1;
    }

    for (bs[0..bs_len]) |b| {
        result[b] = b;
    }

    var n: u21 = 0;
    for (0..256) |b| {
        var found = false;
        for (bs[0..bs_len]) |bb| {
            if (bb == b) {
                found = true;
                break;
            }
        }
        if (!found) {
            result[b] = 0x100 + n;
            n += 1;
        }
    }

    return result;
}

var bytes_to_unicode_table: [256]u21 = undefined;
var bytes_to_unicode_initialized = false;

pub fn getBytesToUnicode() [256]u21 {
    if (!bytes_to_unicode_initialized) {
        bytes_to_unicode_table = initBytesToUnicode();
        bytes_to_unicode_initialized = true;
    }
    return bytes_to_unicode_table;
}

fn encodeUtf8Cp(cp: u21, buffer: []u8) !u3 {
    if (cp <= 0x7F) {
        if (buffer.len < 1) return error.BufferTooSmall;
        buffer[0] = @intCast(cp);
        return 1;
    } else if (cp <= 0x7FF) {
        if (buffer.len < 2) return error.BufferTooSmall;
        buffer[0] = @intCast(0b11000000 | (cp >> 6));
        buffer[1] = @intCast(0b10000000 | (cp & 0b00111111));
        return 2;
    } else if (cp <= 0xFFFF) {
        if (buffer.len < 3) return error.BufferTooSmall;
        buffer[0] = @intCast(0b11100000 | (cp >> 12));
        buffer[1] = @intCast(0b10000000 | ((cp >> 6) & 0b00111111));
        buffer[2] = @intCast(0b10000000 | (cp & 0b00111111));
        return 3;
    } else if (cp <= 0x10FFFF) {
        if (buffer.len < 4) return error.BufferTooSmall;
        buffer[0] = @intCast(0b11110000 | (cp >> 18));
        buffer[1] = @intCast(0b10000000 | ((cp >> 12) & 0b00111111));
        buffer[2] = @intCast(0b10000000 | ((cp >> 6) & 0b00111111));
        buffer[3] = @intCast(0b10000000 | (cp & 0b00111111));
        return 4;
    }
    return error.CodepointTooLarge;
}

// ============================================================================
// Unicode helpers (simplified but covering common ranges)
// ============================================================================

fn utf8ByteSequenceLength(first_byte: u8) u3 {
    if (first_byte < 0b10000000) return 1;
    if (first_byte < 0b11100000) return 2;
    if (first_byte < 0b11110000) return 3;
    return 4;
}

fn decodeCp(text: []const u8) !struct { cp: u21, len: u3 } {
    if (text.len == 0) return error.InvalidUtf8;
    const len = utf8ByteSequenceLength(text[0]);
    if (text.len < len) return error.InvalidUtf8;
    const cp = try std.unicode.utf8Decode(text[0..len]);
    return .{ .cp = cp, .len = len };
}

fn isNewlineByte(b: u8) bool {
    return b == '\n' or b == '\r';
}

fn isNewlineCp(cp: u21) bool {
    return cp == '\n' or cp == '\r' or cp == 0x85 or cp == 0x0B or cp == 0x0C;
}

fn isWhitespaceCp(cp: u21) bool {
    return cp == ' ' or cp == '\t' or cp == '\n' or cp == '\r' or
        cp == 0x0B or cp == 0x0C or cp == 0xA0 or
        cp == 0x1680 or cp == 0x2000 or cp == 0x2001 or cp == 0x2002 or
        cp == 0x2003 or cp == 0x2004 or cp == 0x2005 or cp == 0x2006 or
        cp == 0x2007 or cp == 0x2008 or cp == 0x2009 or cp == 0x200A or
        cp == 0x2028 or cp == 0x2029 or cp == 0x202F or cp == 0x205F or
        cp == 0x3000;
}

fn isAsciiLetter(b: u8) bool {
    return (b >= 'A' and b <= 'Z') or (b >= 'a' and b <= 'z');
}

fn isAsciiDigit(b: u8) bool {
    return b >= '0' and b <= '9';
}

fn isPunctuationCp(cp: u21) bool {
    if (cp >= 0x21 and cp <= 0x2F) return true;
    if (cp >= 0x3A and cp <= 0x40) return true;
    if (cp >= 0x5B and cp <= 0x60) return true;
    if (cp >= 0x7B and cp <= 0x7E) return true;
    if (cp >= 0x2000 and cp <= 0x206F) return true;
    if (cp >= 0x2E00 and cp <= 0x2E7F) return true;
    if (cp >= 0x3000 and cp <= 0x303F) return true;
    if (cp >= 0xFE10 and cp <= 0xFE1F) return true;
    if (cp >= 0xFE30 and cp <= 0xFE4F) return true;
    if (cp >= 0xFE50 and cp <= 0xFE6F) return true;
    if (cp >= 0xFF00 and cp <= 0xFFEF) return true;
    return false;
}

fn isSymbolCp(cp: u21) bool {
    if (cp >= 0x20A0 and cp <= 0x20CF) return true;
    if (cp >= 0x2100 and cp <= 0x214F) return true;
    if (cp >= 0x2190 and cp <= 0x21FF) return true;
    if (cp >= 0x2200 and cp <= 0x22FF) return true;
    if (cp >= 0x2300 and cp <= 0x23FF) return true;
    if (cp >= 0x2400 and cp <= 0x243F) return true;
    if (cp >= 0x2440 and cp <= 0x245F) return true;
    if (cp >= 0x2460 and cp <= 0x24FF) return true;
    if (cp >= 0x2500 and cp <= 0x257F) return true;
    if (cp >= 0x2580 and cp <= 0x259F) return true;
    if (cp >= 0x25A0 and cp <= 0x25FF) return true;
    if (cp >= 0x2600 and cp <= 0x26FF) return true;
    if (cp >= 0x2700 and cp <= 0x27BF) return true;
    if (cp >= 0x27C0 and cp <= 0x27EF) return true;
    if (cp >= 0x27F0 and cp <= 0x27FF) return true;
    if (cp >= 0x2800 and cp <= 0x28FF) return true;
    if (cp >= 0x2900 and cp <= 0x297F) return true;
    if (cp >= 0x2980 and cp <= 0x29FF) return true;
    if (cp >= 0x2A00 and cp <= 0x2AFF) return true;
    if (cp >= 0x2B00 and cp <= 0x2BFF) return true;
    return false;
}

fn isLetterCp(cp: u21) bool {
    if (cp >= 'A' and cp <= 'Z') return true;
    if (cp >= 'a' and cp <= 'z') return true;
    if (cp >= 0xC0 and cp <= 0xD6) return true;
    if (cp >= 0xD8 and cp <= 0xF6) return true;
    if (cp >= 0xF8 and cp <= 0xFF) return true;
    if (cp >= 0x100 and cp <= 0x17F) return true;
    if (cp >= 0x180 and cp <= 0x24F) return true;
    if (cp >= 0x400 and cp <= 0x4FF) return true;
    if (cp >= 0x500 and cp <= 0x52F) return true;
    if (cp >= 0x1E00 and cp <= 0x1EFF) return true;
    if (cp >= 0x370 and cp <= 0x3FF) return true;
    if (cp >= 0x1F00 and cp <= 0x1FFF) return true;
    if (cp >= 0x600 and cp <= 0x6FF) return true;
    if (cp >= 0x750 and cp <= 0x77F) return true;
    if (cp >= 0x4E00 and cp <= 0x9FFF) return true;
    if (cp >= 0x3400 and cp <= 0x4DBF) return true;
    if (cp >= 0x3040 and cp <= 0x309F) return true;
    if (cp >= 0x30A0 and cp <= 0x30FF) return true;
    if (cp >= 0xAC00 and cp <= 0xD7AF) return true;
    if (cp >= 0x1100 and cp <= 0x11FF) return true;
    if (cp >= 0x3130 and cp <= 0x318F) return true;
    return false;
}

fn isMarkCp(cp: u21) bool {
    if (cp >= 0x0300 and cp <= 0x036F) return true;
    if (cp >= 0x1AB0 and cp <= 0x1AFF) return true;
    if (cp >= 0x1DC0 and cp <= 0x1DFF) return true;
    if (cp >= 0x20D0 and cp <= 0x20FF) return true;
    if (cp >= 0xFE20 and cp <= 0xFE2F) return true;
    return false;
}

fn isNumberCp(cp: u21) bool {
    if (cp >= '0' and cp <= '9') return true;
    if (cp >= 0x660 and cp <= 0x669) return true;
    if (cp >= 0x6F0 and cp <= 0x6F9) return true;
    if (cp >= 0x966 and cp <= 0x96F) return true;
    if (cp >= 0x9E6 and cp <= 0x9EF) return true;
    if (cp >= 0xA66 and cp <= 0xA6F) return true;
    if (cp >= 0xAE6 and cp <= 0xAEF) return true;
    if (cp >= 0xB66 and cp <= 0xB6F) return true;
    if (cp >= 0xBE6 and cp <= 0xBEF) return true;
    if (cp >= 0xC66 and cp <= 0xC6F) return true;
    if (cp >= 0xCE6 and cp <= 0xCEF) return true;
    if (cp >= 0xD66 and cp <= 0xD6F) return true;
    if (cp >= 0xE50 and cp <= 0xE59) return true;
    if (cp >= 0xED0 and cp <= 0xED9) return true;
    if (cp >= 0xF20 and cp <= 0xF29) return true;
    if (cp >= 0x1040 and cp <= 0x1049) return true;
    if (cp >= 0x17E0 and cp <= 0x17E9) return true;
    if (cp >= 0x1810 and cp <= 0x1819) return true;
    if (cp >= 0xFF10 and cp <= 0xFF19) return true;
    return false;
}

fn isCjkKanaCp(cp: u21) bool {
    if (cp >= 0x4E00 and cp <= 0x9FFF) return true;
    if (cp >= 0x3400 and cp <= 0x4DBF) return true;
    if (cp >= 0x3040 and cp <= 0x309F) return true;
    if (cp >= 0x30A0 and cp <= 0x30FF) return true;
    if (cp >= 0x31F0 and cp <= 0x31FF) return true;
    if (cp >= 0xF900 and cp <= 0xFAFF) return true;
    if (cp >= 0x2E80 and cp <= 0x2EFF) return true;
    if (cp >= 0x2F00 and cp <= 0x2FDF) return true;
    if (cp >= 0x3005 and cp <= 0x3007) return true;
    return false;
}

// ============================================================================
// Pattern matching for Split pre-tokenizer
// ============================================================================

pub const Pattern = enum {
    digits_1_3,
    cjk_kana,
    deepseek_word,
};

const MatchResult = struct { start: usize, end: usize };

fn matchPattern(pat: Pattern, text: []const u8) MatchResult {
    if (text.len == 0) return .{ .start = 0, .end = 0 };

    switch (pat) {
        .digits_1_3 => {
            var i: usize = 0;
            while (i < text.len) {
                const res = decodeCp(text[i..]) catch { i += 1; continue; };
                if (isNumberCp(res.cp)) {
                    var end = i + res.len;
                    var count: usize = 1;
                    while (count < 3 and end < text.len) {
                        const next = decodeCp(text[end..]) catch break;
                        if (!isNumberCp(next.cp)) break;
                        end += next.len;
                        count += 1;
                    }
                    return .{ .start = i, .end = end };
                }
                i += res.len;
            }
        },
        .cjk_kana => {
            var i: usize = 0;
            while (i < text.len) {
                const res = decodeCp(text[i..]) catch { i += 1; continue; };
                if (isCjkKanaCp(res.cp)) {
                    var end = i + res.len;
                    while (end < text.len) {
                        const next = decodeCp(text[end..]) catch break;
                        if (!isCjkKanaCp(next.cp)) break;
                        end += next.len;
                    }
                    return .{ .start = i, .end = end };
                }
                i += res.len;
            }
        },
        .deepseek_word => {
            return matchDeepseekWord(text);
        },
    }

    return .{ .start = 0, .end = 0 };
}

/// DeepSeek pattern 3 (adapted from cl100k_base style).
/// Searches for the left-most match using left-priority, greedy branches.
fn matchDeepseekWord(text: []const u8) MatchResult {
    if (text.len == 0) return .{ .start = 0, .end = 0 };

    var pos: usize = 0;
    while (pos < text.len) {
        const res0 = decodeCp(text[pos..]) catch {
            pos += 1;
            continue;
        };

        // Branch 1: punctuation followed by ASCII letters
        if (isPunctuationCp(res0.cp)) {
            var j: usize = pos + res0.len;
            while (j < text.len) {
                if (!isAsciiLetter(text[j])) break;
                j += 1;
            }
            if (j > pos + res0.len) return .{ .start = pos, .end = j };
        }

        // Branch 2: optional non-letter/punct/symbol prefix, then letters/marks
        {
            var j: usize = pos;
            if (j < text.len) {
                const res = decodeCp(text[j..]) catch {
                    pos += res0.len;
                    continue;
                };
                if (!isNewlineCp(res.cp) and !isLetterCp(res.cp) and !isPunctuationCp(res.cp) and !isSymbolCp(res.cp)) {
                    j += res.len;
                }
            }
            const start_letters = j;
            while (j < text.len) {
                const res = decodeCp(text[j..]) catch break;
                if (!isLetterCp(res.cp) and !isMarkCp(res.cp)) break;
                j += res.len;
            }
            if (j > start_letters) return .{ .start = pos, .end = j };
        }

        // Branch 3: optional space, then punctuation/symbols, then optional newlines
        {
            var j: usize = pos;
            if (j < text.len and text[j] == ' ') j += 1;
            const start_punct = j;
            while (j < text.len) {
                const res = decodeCp(text[j..]) catch break;
                if (!isPunctuationCp(res.cp) and !isSymbolCp(res.cp)) break;
                j += res.len;
            }
            if (j > start_punct) {
                while (j < text.len) {
                    const res = decodeCp(text[j..]) catch break;
                    if (!isNewlineCp(res.cp)) break;
                    j += res.len;
                }
                return .{ .start = pos, .end = j };
            }
        }

        // Branch 4: optional whitespace, then newlines
        {
            var j: usize = pos;
            while (j < text.len) {
                const res = decodeCp(text[j..]) catch break;
                if (!isWhitespaceCp(res.cp)) break;
                j += res.len;
            }
            const start_newlines = j;
            while (j < text.len) {
                const res = decodeCp(text[j..]) catch break;
                if (!isNewlineCp(res.cp)) break;
                j += res.len;
            }
            if (j > start_newlines) return .{ .start = pos, .end = j };
        }

        // Branch 5: whitespace not followed by non-whitespace
        {
            var j: usize = pos;
            while (j < text.len) {
                const res = decodeCp(text[j..]) catch break;
                if (!isWhitespaceCp(res.cp)) break;
                j += res.len;
            }
            if (j > pos) {
                if (j >= text.len) {
                    return .{ .start = pos, .end = j };
                }
                const res = decodeCp(text[j..]) catch return .{ .start = pos, .end = j };
                if (isWhitespaceCp(res.cp)) {
                    return .{ .start = pos, .end = j };
                }
            }
        }

        // Branch 6: any whitespace
        {
            var j: usize = pos;
            while (j < text.len) {
                const res = decodeCp(text[j..]) catch break;
                if (!isWhitespaceCp(res.cp)) break;
                j += res.len;
            }
            if (j > pos) return .{ .start = pos, .end = j };
        }

        pos += res0.len;
    }

    return .{ .start = 0, .end = 0 };
}

// ============================================================================
// PreTokenizer types
// ============================================================================

pub const PreTokenizer = union(enum) {
    sequence: Sequence,
    split: Split,
    byte_level: ByteLevel,

    pub fn preTokenize(self: PreTokenizer, allocator: std.mem.Allocator, text: []const u8) error{OutOfMemory}!std.ArrayList([]const u8) {
        switch (self) {
            .sequence => |s| return s.preTokenize(allocator, text),
            .split => |sp| return sp.preTokenize(allocator, text),
            .byte_level => |bl| return bl.preTokenize(allocator, text),
        }
    }

    pub fn deinit(self: *PreTokenizer, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .sequence => |*s| s.deinit(allocator),
            .split => |*sp| sp.deinit(allocator),
            .byte_level => {},
        }
    }
};

pub const Sequence = struct {
    pretokenizers: []PreTokenizer,

    pub fn preTokenize(self: Sequence, allocator: std.mem.Allocator, text: []const u8) error{OutOfMemory}!std.ArrayList([]const u8) {
        var tokens = std.ArrayList([]const u8).empty;
        errdefer {
            for (tokens.items) |t| allocator.free(t);
            tokens.deinit(allocator);
        }
        try tokens.append(allocator, try allocator.dupe(u8, text));

        for (self.pretokenizers) |pt| {
            var new_tokens = std.ArrayList([]const u8).empty;
            errdefer {
                for (new_tokens.items) |t| allocator.free(t);
                new_tokens.deinit(allocator);
            }

            for (tokens.items) |token| {
                var sub = try pt.preTokenize(allocator, token);
                defer sub.deinit(allocator);
                for (sub.items) |s| {
                    try new_tokens.append(allocator, s);
                }
            }

            for (tokens.items) |t| allocator.free(t);
            tokens.deinit(allocator);
            tokens = new_tokens;
        }

        return tokens;
    }

    pub fn deinit(self: *Sequence, allocator: std.mem.Allocator) void {
        for (self.pretokenizers) |*pt| {
            pt.deinit(allocator);
        }
        allocator.free(self.pretokenizers);
    }
};

pub const Split = struct {
    pattern: Pattern,
    behavior: Behavior,
    invert: bool,

    pub const Behavior = enum {
        isolated,
        removed,
        merged_with_previous,
        merged_with_next,
        contiguous,
    };

    pub fn preTokenize(self: Split, allocator: std.mem.Allocator, text: []const u8) !std.ArrayList([]const u8) {
        var result = std.ArrayList([]const u8).empty;
        errdefer {
            for (result.items) |t| allocator.free(t);
            result.deinit(allocator);
        }

        var pos: usize = 0;
        while (pos < text.len) {
            const m = matchPattern(self.pattern, text[pos..]);
            if (m.end == 0) {
                // No more matches, add remaining text
                const remaining = try allocator.dupe(u8, text[pos..]);
                try result.append(allocator, remaining);
                break;
            }

            if (self.invert) {
                // Inverted: non-matching parts are the delimiters
                // Matching parts are kept as-is (with behavior applied)
                // For simplicity, we treat invert=true the same as keeping
                // matches and non-matches alternating.
                // This matches the common case where invert=false.
                // TODO: implement proper invert logic if needed.
                if (m.start > 0) {
                    const prefix = try allocator.dupe(u8, text[pos..pos + m.start]);
                    try result.append(allocator, prefix);
                }
                const matched = try allocator.dupe(u8, text[pos + m.start..pos + m.end]);
                try result.append(allocator, matched);
            } else {
                // Normal: matching parts are isolated
                if (m.start > 0) {
                    const prefix = try allocator.dupe(u8, text[pos..pos + m.start]);
                    try result.append(allocator, prefix);
                }
                const matched = try allocator.dupe(u8, text[pos + m.start..pos + m.end]);
                try result.append(allocator, matched);
            }

            pos += m.end;
        }

        return result;
    }

    pub fn deinit(self: *Split, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }
};

pub const ByteLevel = struct {
    add_prefix_space: bool,

    pub fn preTokenize(self: ByteLevel, allocator: std.mem.Allocator, text: []const u8) !std.ArrayList([]const u8) {
        var result = std.ArrayList([]const u8).empty;
        errdefer {
            for (result.items) |t| allocator.free(t);
            result.deinit(allocator);
        }

        // Compute output size
        var out_chars: usize = text.len;
        if (self.add_prefix_space and (text.len == 0 or text[0] != ' ')) {
            out_chars += 1;
        }

        // Each byte maps to a Unicode codepoint (max 4 UTF-8 bytes)
        var out = try allocator.alloc(u8, out_chars * 4);
        defer allocator.free(out);
        var out_pos: usize = 0;

        if (self.add_prefix_space and (text.len == 0 or text[0] != ' ')) {
            const len = encodeUtf8Cp(getBytesToUnicode()[' '], out[out_pos..]) catch unreachable;
            out_pos += len;
        }

        for (text) |byte| {
            const len = encodeUtf8Cp(getBytesToUnicode()[byte], out[out_pos..]) catch unreachable;
            out_pos += len;
        }

        const owned = try allocator.dupe(u8, out[0..out_pos]);
        try result.append(allocator, owned);
        return result;
    }
};

// ============================================================================
// Decoder types
// ============================================================================

pub const Decoder = union(enum) {
    byte_level: ByteLevelDecoder,
    replace: ReplaceDecoder,
    fuse: FuseDecoder,
    strip: StripDecoder,
    byte_fallback: ByteFallbackDecoder,
    sequence: SequenceDecoder,

    pub fn decode(self: Decoder, allocator: std.mem.Allocator, text: []const u8) error{OutOfMemory}![]const u8 {
        switch (self) {
            .byte_level => |bl| return bl.decode(allocator, text),
            .replace => |r| return r.decode(allocator, text),
            .fuse => |f| return f.decode(allocator, text),
            .strip => |s| return s.decode(allocator, text),
            .byte_fallback => |bf| return bf.decode(allocator, text),
            .sequence => |seq| return seq.decode(allocator, text),
        }
    }

    pub fn deinit(self: *Decoder, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .byte_level => {},
            .replace => |*r| r.deinit(allocator),
            .fuse => {},
            .strip => |*s| s.deinit(allocator),
            .byte_fallback => {},
            .sequence => |*seq| seq.deinit(allocator),
        }
    }
};

pub const ByteFallbackDecoder = struct {
    pub fn decode(_: ByteFallbackDecoder, allocator: std.mem.Allocator, text: []const u8) error{OutOfMemory}![]const u8 {
        var result = std.ArrayList(u8).empty;
        errdefer result.deinit(allocator);

        var i: usize = 0;
        while (i < text.len) {
            if (i + 6 <= text.len and
                text[i] == '<' and
                text[i + 1] == '0' and
                text[i + 2] == 'x' and
                text[i + 5] == '>')
            {
                const hex_str = text[i + 3 .. i + 5];
                if (std.fmt.parseInt(u8, hex_str, 16)) |byte| {
                    try result.append(allocator, byte);
                    i += 6;
                    continue;
                } else |_| {}
            }
            try result.append(allocator, text[i]);
            i += 1;
        }

        return result.toOwnedSlice(allocator);
    }
};

pub const SequenceDecoder = struct {
    decoders: []Decoder,

    pub fn decode(self: SequenceDecoder, allocator: std.mem.Allocator, text: []const u8) error{OutOfMemory}![]const u8 {
        var current: []const u8 = try allocator.dupe(u8, text);
        errdefer allocator.free(current);

        for (self.decoders) |dec| {
            const next = try dec.decode(allocator, current);
            allocator.free(current);
            current = next;
        }

        return current;
    }

    pub fn deinit(self: *SequenceDecoder, allocator: std.mem.Allocator) void {
        for (self.decoders) |*d| {
            d.deinit(allocator);
        }
        allocator.free(self.decoders);
    }
};

pub const ByteLevelDecoder = struct {
    add_prefix_space: bool,

    /// Reverse the bytes_to_unicode mapping.
    pub fn decode(self: ByteLevelDecoder, allocator: std.mem.Allocator, text: []const u8) error{OutOfMemory}![]const u8 {
        // Build reverse mapping: codepoint -> byte
        var reverse: [0x180]u8 = undefined; // 0x100 to 0x17F covers all mapped values
        for (0..256) |b| {
            const cp = getBytesToUnicode()[b];
            if (cp >= 0x100 and cp < 0x180) {
                reverse[cp - 0x100] = @intCast(b);
            }
        }

        var out = try allocator.alloc(u8, text.len * 4); // worst case: every byte is 4-byte UTF-8
        errdefer allocator.free(out);
        var out_pos: usize = 0;

        var i: usize = 0;
        var first = true;
        while (i < text.len) {
            const cp_res = decodeCp(text[i..]) catch {
                // Invalid UTF-8: skip byte
                i += 1;
                continue;
            };
            const cp = cp_res.cp;
            const len = cp_res.len;

            // Map back to byte
            var byte: u8 = undefined;
            if (cp < 0x100) {
                byte = @intCast(cp);
            } else if (cp >= 0x100 and cp < 0x180) {
                byte = reverse[cp - 0x100];
            } else {
                // Fallback: write raw UTF-8 bytes
                @memcpy(out[out_pos..out_pos + len], text[i..i + len]);
                out_pos += len;
                i += len;
                first = false;
                continue;
            }

            // Handle add_prefix_space: strip leading space
            if (self.add_prefix_space and first and byte == ' ') {
                i += len;
                first = false;
                continue;
            }

            out[out_pos] = byte;
            out_pos += 1;
            i += len;
            first = false;
        }

        return try allocator.realloc(out, out_pos);
    }
};

pub const ReplaceDecoder = struct {
    pattern: []const u8,
    content: []const u8,

    pub fn decode(self: ReplaceDecoder, allocator: std.mem.Allocator, text: []const u8) ![]const u8 {
        // Simple string replacement
        if (self.pattern.len == 0) return try allocator.dupe(u8, text);

        var result = std.ArrayList(u8).empty;
        errdefer result.deinit(allocator);

        var i: usize = 0;
        while (i < text.len) {
            if (std.mem.startsWith(u8, text[i..], self.pattern)) {
                try result.appendSlice(allocator, self.content);
                i += self.pattern.len;
            } else {
                try result.append(allocator, text[i]);
                i += 1;
            }
        }

        return result.toOwnedSlice(allocator);
    }

    pub fn deinit(self: *ReplaceDecoder, allocator: std.mem.Allocator) void {
        allocator.free(self.pattern);
        allocator.free(self.content);
    }
};

pub const FuseDecoder = struct {
    pub fn decode(_: FuseDecoder, allocator: std.mem.Allocator, text: []const u8) ![]const u8 {
        return try allocator.dupe(u8, text);
    }
};

pub const StripDecoder = struct {
    content: []const u8,
    start: usize,
    stop: usize,

    pub fn decode(self: StripDecoder, allocator: std.mem.Allocator, text: []const u8) error{OutOfMemory}![]const u8 {
        const begin = @min(self.start, text.len);
        const end = if (self.stop > text.len) 0 else text.len - self.stop;
        if (end <= begin) return try allocator.dupe(u8, "");
        return try allocator.dupe(u8, text[begin..end]);
    }

    pub fn deinit(self: *StripDecoder, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
    }
};

// ============================================================================
// Normalizer types
// ============================================================================

pub const Normalizer = union(enum) {
    none,
    prepend: PrependNormalizer,
    replace: ReplaceNormalizer,
    sequence: SequenceNormalizer,

    pub fn normalize(self: Normalizer, allocator: std.mem.Allocator, text: []const u8) error{OutOfMemory}![]const u8 {
        switch (self) {
            .none => return try allocator.dupe(u8, text),
            .prepend => |p| return p.normalize(allocator, text),
            .replace => |r| return r.normalize(allocator, text),
            .sequence => |s| return s.normalize(allocator, text),
        }
    }

    pub fn deinit(self: *Normalizer, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .none => {},
            .prepend => |*p| p.deinit(allocator),
            .replace => |*r| r.deinit(allocator),
            .sequence => |*s| s.deinit(allocator),
        }
    }
};

pub const PrependNormalizer = struct {
    prepend: []const u8,

    pub fn normalize(self: PrependNormalizer, allocator: std.mem.Allocator, text: []const u8) ![]const u8 {
        const result = try allocator.alloc(u8, self.prepend.len + text.len);
        @memcpy(result[0..self.prepend.len], self.prepend);
        @memcpy(result[self.prepend.len..], text);
        return result;
    }

    pub fn deinit(self: *PrependNormalizer, allocator: std.mem.Allocator) void {
        allocator.free(self.prepend);
    }
};

pub const ReplaceNormalizer = struct {
    pattern: []const u8,
    content: []const u8,

    pub fn normalize(self: ReplaceNormalizer, allocator: std.mem.Allocator, text: []const u8) ![]const u8 {
        if (self.pattern.len == 0) return try allocator.dupe(u8, text);

        var result = std.ArrayList(u8).empty;
        errdefer result.deinit(allocator);

        var i: usize = 0;
        while (i < text.len) {
            if (std.mem.startsWith(u8, text[i..], self.pattern)) {
                try result.appendSlice(allocator, self.content);
                i += self.pattern.len;
            } else {
                try result.append(allocator, text[i]);
                i += 1;
            }
        }

        return result.toOwnedSlice(allocator);
    }

    pub fn deinit(self: *ReplaceNormalizer, allocator: std.mem.Allocator) void {
        allocator.free(self.pattern);
        allocator.free(self.content);
    }
};

pub const SequenceNormalizer = struct {
    normalizers: []Normalizer,

    pub fn normalize(self: SequenceNormalizer, allocator: std.mem.Allocator, text: []const u8) error{OutOfMemory}![]const u8 {
        var current: []const u8 = try allocator.dupe(u8, text);
        errdefer allocator.free(current);

        for (self.normalizers) |norm| {
            const next = try norm.normalize(allocator, current);
            allocator.free(current);
            current = next;
        }

        return current;
    }

    pub fn deinit(self: *SequenceNormalizer, allocator: std.mem.Allocator) void {
        for (self.normalizers) |*n| {
            n.deinit(allocator);
        }
        allocator.free(self.normalizers);
    }
};

// ============================================================================
// AddedToken
// ============================================================================

pub const AddedToken = struct {
    id: u32,
    content: []const u8,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,

    pub fn deinit(self: *AddedToken, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
    }
};
