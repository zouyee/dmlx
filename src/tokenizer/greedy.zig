/// Greedy longest-match tokenizer strategy.
///
/// Reads `tokenizer.json` and provides encode/decode.
/// This is a simplified encoder that does not apply BPE merges.
/// For accurate encoding, use a dedicated BPE/SentencePiece backend.
const std = @import("std");
const interface = @import("interface.zig");

pub const GreedyTokenizer = struct {
    allocator: std.mem.Allocator,

    // Vocabulary
    vocab: std.StringHashMap(u32), // token string -> id
    ids_to_tokens: std.AutoHashMap(u32, []const u8),

    // Special tokens
    bos_token: ?[]const u8,
    eos_token: ?[]const u8,
    pad_token: ?[]const u8,
    unk_token: ?[]const u8,
    bos_id: ?u32,
    eos_id: ?u32,
    pad_id: ?u32,
    unk_id: ?u32,

    pub fn init(allocator: std.mem.Allocator) GreedyTokenizer {
        return .{
            .allocator = allocator,
            .vocab = std.StringHashMap(u32).init(allocator),
            .ids_to_tokens = std.AutoHashMap(u32, []const u8).init(allocator),
            .bos_token = null,
            .eos_token = null,
            .pad_token = null,
            .unk_token = null,
            .bos_id = null,
            .eos_id = null,
            .pad_id = null,
            .unk_id = null,
        };
    }

    pub fn deinit(self: *GreedyTokenizer) void {
        var it = self.vocab.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.vocab.deinit();

        var it2 = self.ids_to_tokens.iterator();
        while (it2.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.ids_to_tokens.deinit();

        if (self.bos_token) |t| self.allocator.free(t);
        if (self.eos_token) |t| self.allocator.free(t);
        if (self.pad_token) |t| self.allocator.free(t);
        if (self.unk_token) |t| self.allocator.free(t);
    }

    /// Load tokenizer from a `tokenizer.json` file.
    pub fn loadFromFile(self: *GreedyTokenizer, io: std.Io, path: []const u8) !void {
        const content = try std.Io.Dir.cwd().readFileAlloc(io, path, self.allocator, .limited(50 * 1024 * 1024));
        defer self.allocator.free(content);
        return self.loadFromJson(content);
    }

    /// Load tokenizer from JSON string.
    pub fn loadFromJson(self: *GreedyTokenizer, json_text: []const u8) !void {
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, json_text, .{});
        defer parsed.deinit();

        const root = parsed.value.object;

        // === Load vocab ===
        const model = root.get("model") orelse return error.MissingModel;
        const vocab = model.object.get("vocab") orelse return error.MissingVocab;

        var vocab_it = vocab.object.iterator();
        while (vocab_it.next()) |entry| {
            const id: u32 = switch (entry.value_ptr.*) {
                .integer => |n| @intCast(n),
                .float => |f| @intFromFloat(f),
                else => continue,
            };

            // vocab: avoid leaking memory when key already exists
            if (self.vocab.contains(entry.key_ptr.*)) {
                self.vocab.getPtr(entry.key_ptr.*).?.* = id;
            } else {
                const owned_key = try self.allocator.dupe(u8, entry.key_ptr.*);
                errdefer self.allocator.free(owned_key);
                try self.vocab.put(owned_key, id);
            }

            // ids_to_tokens: free old token string when replacing
            if (self.ids_to_tokens.contains(id)) {
                const old = self.ids_to_tokens.getPtr(id).?.*;
                self.allocator.free(old);
                const owned_token = try self.allocator.dupe(u8, entry.key_ptr.*);
                errdefer self.allocator.free(owned_token);
                self.ids_to_tokens.getPtr(id).?.* = owned_token;
            } else {
                const owned_token = try self.allocator.dupe(u8, entry.key_ptr.*);
                errdefer self.allocator.free(owned_token);
                try self.ids_to_tokens.put(id, owned_token);
            }
        }

        // === Load added_tokens ===
        const added = root.get("added_tokens");
        if (added) |at| {
            for (at.array.items) |item| {
                const obj = item.object;
                const content_str = obj.get("content") orelse continue;
                const token_str = content_str.string;
                const id_val = obj.get("id") orelse continue;
                const id: u32 = switch (id_val) {
                    .integer => |n| @intCast(n),
                    .float => |f| @intFromFloat(f),
                    else => continue,
                };

                // vocab
                if (self.vocab.contains(token_str)) {
                    self.vocab.getPtr(token_str).?.* = id;
                } else {
                    const owned_key = try self.allocator.dupe(u8, token_str);
                    errdefer self.allocator.free(owned_key);
                    try self.vocab.put(owned_key, id);
                }

                // ids_to_tokens
                if (self.ids_to_tokens.contains(id)) {
                    const old = self.ids_to_tokens.getPtr(id).?.*;
                    self.allocator.free(old);
                    const owned_token = try self.allocator.dupe(u8, token_str);
                    errdefer self.allocator.free(owned_token);
                    self.ids_to_tokens.getPtr(id).?.* = owned_token;
                } else {
                    const owned_token = try self.allocator.dupe(u8, token_str);
                    errdefer self.allocator.free(owned_token);
                    try self.ids_to_tokens.put(id, owned_token);
                }
            }
        }

        // === Load special tokens from post_processor ===
        const post_processor = root.get("post_processor");
        if (post_processor) |pp| {
            const special_tokens = pp.object.get("special_tokens");
            if (special_tokens) |st| {
                var st_it = st.object.iterator();
                while (st_it.next()) |entry| {
                    const name = entry.key_ptr.*;
                    const info = entry.value_ptr.*.object;

                    const id_val = info.get("id") orelse continue;
                    const id: u32 = switch (id_val) {
                        .integer => |n| @intCast(n),
                        .float => |f| @intFromFloat(f),
                        else => continue,
                    };

                    if (std.mem.eql(u8, name, "bos_token")) {
                        self.bos_id = id;
                        if (self.ids_to_tokens.get(id)) |tok| {
                            self.bos_token = try self.allocator.dupe(u8, tok);
                        }
                    } else if (std.mem.eql(u8, name, "eos_token")) {
                        self.eos_id = id;
                        if (self.ids_to_tokens.get(id)) |tok| {
                            self.eos_token = try self.allocator.dupe(u8, tok);
                        }
                    } else if (std.mem.eql(u8, name, "pad_token")) {
                        self.pad_id = id;
                        if (self.ids_to_tokens.get(id)) |tok| {
                            self.pad_token = try self.allocator.dupe(u8, tok);
                        }
                    } else if (std.mem.eql(u8, name, "unk_token")) {
                        self.unk_id = id;
                        if (self.ids_to_tokens.get(id)) |tok| {
                            self.unk_token = try self.allocator.dupe(u8, tok);
                        }
                    }
                }
            }
        }
    }

    /// Decode a sequence of token IDs back to text.
    pub fn decode(self: *GreedyTokenizer, ids: []const u32) ![]const u8 {
        var result = std.ArrayList(u8).empty;
        errdefer result.deinit(self.allocator);

        for (ids) |id| {
            if (self.ids_to_tokens.get(id)) |token| {
                const decoded = try decodeToken(self.allocator, token);
                defer if (decoded.ptr != token.ptr) self.allocator.free(decoded);
                try result.appendSlice(self.allocator, decoded);
            }
        }

        return result.toOwnedSlice(self.allocator);
    }

    /// Encode text to token IDs using greedy longest-match.
    pub fn encode(self: *GreedyTokenizer, text: []const u8, add_special_tokens: bool) ![]u32 {
        var result = std.ArrayList(u32).empty;
        errdefer result.deinit(self.allocator);

        if (add_special_tokens) {
            if (self.bos_id) |id| {
                try result.append(self.allocator, id);
            }
        }

        var i: usize = 0;
        while (i < text.len) {
            var longest_len: usize = 0;
            var longest_id: u32 = self.unk_id orelse 0;

            var len: usize = @min(text.len - i, 64);
            while (len > 0) : (len -= 1) {
                const sub = text[i .. i + len];
                if (self.vocab.get(sub)) |id| {
                    longest_len = len;
                    longest_id = id;
                    break;
                }
            }

            if (longest_len == 0) {
                if (self.unk_id) |id| {
                    try result.append(self.allocator, id);
                }
                i += 1;
            } else {
                try result.append(self.allocator, longest_id);
                i += longest_len;
            }
        }

        if (add_special_tokens) {
            if (self.eos_id) |id| {
                try result.append(self.allocator, id);
            }
        }

        return result.toOwnedSlice(self.allocator);
    }

    pub fn vocabSize(self: GreedyTokenizer) usize {
        return self.vocab.count();
    }

    // ------------------------------------------------------------------
    // Strategy interface implementation
    // ------------------------------------------------------------------

    pub fn asStrategy(self: *GreedyTokenizer) interface.TokenizerStrategy {
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    const vtable: interface.VTable = .{
        .encode = encodeImpl,
        .decode = decodeImpl,
        .vocabSize = vocabSizeImpl,
        .deinit = deinitImpl,
    };

    fn encodeImpl(ptr: *anyopaque, text: []const u8, add_special_tokens: bool, allocator: std.mem.Allocator) ![]u32 {
        _ = allocator;
        const self: *GreedyTokenizer = @ptrCast(@alignCast(ptr));
        return self.encode(text, add_special_tokens);
    }

    fn decodeImpl(ptr: *anyopaque, ids: []const u32, allocator: std.mem.Allocator) ![]const u8 {
        _ = allocator;
        const self: *GreedyTokenizer = @ptrCast(@alignCast(ptr));
        return self.decode(ids);
    }

    fn vocabSizeImpl(ptr: *anyopaque) usize {
        const self: *GreedyTokenizer = @ptrCast(@alignCast(ptr));
        return self.vocabSize();
    }

    fn deinitImpl(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        _ = allocator;
        const self: *GreedyTokenizer = @ptrCast(@alignCast(ptr));
        self.deinit();
    }
};

/// Decode a single token string, handling byte-fallback encoding.
fn decodeToken(allocator: std.mem.Allocator, token: []const u8) error{OutOfMemory}![]const u8 {
    if (token.len == 6 and std.mem.startsWith(u8, token, "<0x") and std.mem.endsWith(u8, token, ">")) {
        const hex_str = token[3..5];
        const byte = std.fmt.parseInt(u8, hex_str, 16) catch return allocator.dupe(u8, token);
        const result = try allocator.alloc(u8, 1);
        result[0] = byte;
        return result;
    }

    if (std.mem.startsWith(u8, token, "Ġ")) {
        const rest = token[2..]; // "Ġ" is 2 bytes in UTF-8
        const result = try allocator.alloc(u8, 1 + rest.len);
        result[0] = ' ';
        @memcpy(result[1..], rest);
        return result;
    }

    if (std.mem.eql(u8, token, "Ċ")) return allocator.dupe(u8, "\n");
    if (std.mem.eql(u8, token, "ĉ")) return allocator.dupe(u8, "\t");

    return allocator.dupe(u8, token);
}
