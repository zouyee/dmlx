/// Incremental streaming detokenizer.
///
/// Fixes the "streaming SSE text loses spaces" bug: decoding each generated
/// token individually (`tokenizer.decode(&[_]u32{token})`) applies the
/// decoder pipeline (leading-space strip, byte-level mapping, byte fallback)
/// to a single token in isolation, so a leading-space token like "▁world"
/// decodes to "world" and concatenated deltas lose every word boundary.
///
/// Strategy: cumulative decode + diff. Each new token is appended to the
/// request's token list and the WHOLE sequence is decoded with the same
/// full-sequence decode the non-streaming path uses, so the leading-space
/// strip applies exactly once (first token), byte-fallback pairs spanning
/// token boundaries reassemble, and concatenated deltas are byte-identical
/// to `decode(all_tokens)` by construction.
///
/// The one hazard is multibyte UTF-8 split across tokens (byte-level BPE):
/// the cumulative decode may end with a partial codepoint. addToken holds
/// back an incomplete trailing sequence and emits it once the missing bytes
/// arrive; flush() emits the remainder verbatim at end-of-stream.
///
/// Cost is O(total_text) per token (quadratic over a request's generation);
/// per-step decode of a few KB of text is negligible against per-token
/// generation latency.
const std = @import("std");
const interface = @import("interface.zig");

pub const StreamingDetokenizer = struct {
    allocator: std.mem.Allocator,
    tokenizer: interface.TokenizerStrategy,
    /// Tokens fed so far (the decode input prefix).
    tokens: std.ArrayList(u32),
    /// Bytes of the cumulative decode already emitted to the caller.
    emitted_len: usize,
    /// True after flush() ran; flush is idempotent.
    flushed: bool,

    pub fn init(allocator: std.mem.Allocator, tokenizer: interface.TokenizerStrategy) StreamingDetokenizer {
        return .{
            .allocator = allocator,
            .tokenizer = tokenizer,
            .tokens = std.ArrayList(u32).empty,
            .emitted_len = 0,
            .flushed = false,
        };
    }

    pub fn deinit(self: *StreamingDetokenizer) void {
        self.tokens.deinit(self.allocator);
    }

    /// Feed one token. Returns the newly decodable text increment (owned;
    /// caller frees), or null when the token's bytes are still incomplete
    /// (held back: partial UTF-8 codepoint at the end of the cumulative
    /// decode).
    pub fn addToken(self: *StreamingDetokenizer, token: u32) !?[]u8 {
        self.flushed = false;
        try self.tokens.append(self.allocator, token);
        const full = try self.tokenizer.decode(self.tokens.items, self.allocator);
        defer freeConst(self.allocator, full);

        const safe_len = validUtf8PrefixLen(full);
        if (safe_len <= self.emitted_len) return null;
        const delta = try self.allocator.dupe(u8, full[self.emitted_len..safe_len]);
        self.emitted_len = safe_len;
        return delta;
    }

    /// Emit everything still held back, including a trailing partial or
    /// invalid UTF-8 sequence (matching what the non-streaming full decode
    /// would return). Call once when the final token has been fed.
    pub fn flush(self: *StreamingDetokenizer) !?[]u8 {
        if (self.flushed) return null;
        self.flushed = true;
        if (self.tokens.items.len == 0) return null;
        const full = try self.tokenizer.decode(self.tokens.items, self.allocator);
        defer freeConst(self.allocator, full);

        if (full.len <= self.emitted_len) return null;
        const delta = try self.allocator.dupe(u8, full[self.emitted_len..]);
        self.emitted_len = full.len;
        return delta;
    }
};

/// Length of the longest prefix of `bytes` that is well-formed UTF-8.
fn validUtf8PrefixLen(bytes: []const u8) usize {
    var i: usize = 0;
    while (i < bytes.len) {
        const n = std.unicode.utf8ByteSequenceLength(bytes[i]) catch return i;
        if (i + n > bytes.len) return i;
        _ = std.unicode.utf8Decode(bytes[i .. i + n]) catch return i;
        i += n;
    }
    return i;
}

fn freeConst(allocator: std.mem.Allocator, s: []const u8) void {
    allocator.free(@constCast(s));
}
