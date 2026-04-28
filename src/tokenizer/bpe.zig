/// Byte-Pair Encoding (BPE) tokenizer strategy.
///
/// Supports HF tokenizer.json format with:
///   - BPE model (vocab + merges)
///   - Simplified normalizer (Prepend + Replace)
///   - Simplified decoder (Replace + ByteFallback + Fuse + Strip)
///
/// This is not a full tokenizers.rs implementation; it targets Llama-family
/// tokenizers (TinyLlama, Llama-2, Mistral, etc.).
const std = @import("std");
const interface = @import("interface.zig");
const pt = @import("pre_tokenizer.zig");

pub const BpeTokenizer = struct {
    allocator: std.mem.Allocator,

    vocab: std.StringHashMap(u32),
    ids_to_tokens: std.AutoHashMap(u32, []const u8),
    merges: std.StringHashMap(usize), // "first second" -> rank

    unk_id: ?u32,
    bos_id: ?u32,
    eos_id: ?u32,

    // Configurable pipeline (loaded from tokenizer.json)
    normalizer: pt.Normalizer,
    pre_tokenizer: ?pt.PreTokenizer,
    decoder: ?pt.Decoder,

    pub fn init(allocator: std.mem.Allocator) BpeTokenizer {
        return .{
            .allocator = allocator,
            .vocab = std.StringHashMap(u32).init(allocator),
            .ids_to_tokens = std.AutoHashMap(u32, []const u8).init(allocator),
            .merges = std.StringHashMap(usize).init(allocator),
            .unk_id = null,
            .bos_id = null,
            .eos_id = null,
            .normalizer = .none,
            .pre_tokenizer = null,
            .decoder = null,
        };
    }

    pub fn deinit(self: *BpeTokenizer) void {
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

        var it3 = self.merges.iterator();
        while (it3.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.merges.deinit();

        self.normalizer.deinit(self.allocator);
        if (self.pre_tokenizer) |*p| p.deinit(self.allocator);
        if (self.decoder) |*d| d.deinit(self.allocator);
    }

    pub fn loadFromFile(self: *BpeTokenizer, io: std.Io, path: []const u8) !void {
        const content = try std.Io.Dir.cwd().readFileAlloc(io, path, self.allocator, .limited(50 * 1024 * 1024));
        defer self.allocator.free(content);
        return self.loadFromJson(content);
    }

    pub fn loadFromJson(self: *BpeTokenizer, json_text: []const u8) !void {
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
            const key = try self.allocator.dupe(u8, entry.key_ptr.*);
            errdefer self.allocator.free(key);
            try self.vocab.put(key, id);

            const token_copy = try self.allocator.dupe(u8, entry.key_ptr.*);
            errdefer self.allocator.free(token_copy);
            if (self.ids_to_tokens.contains(id)) {
                const old = self.ids_to_tokens.getPtr(id).?.*;
                self.allocator.free(old);
                self.ids_to_tokens.getPtr(id).?.* = token_copy;
            } else {
                try self.ids_to_tokens.put(id, token_copy);
            }
        }

        // === Load merges ===
        const merges_arr = model.object.get("merges");
        if (merges_arr) |ma| {
            for (ma.array.items, 0..) |item, rank| {
                // Merges can be either:
                // - string format: "first second" (GPT-2 style)
                // - array format: ["first", "second"] (Qwen3 style)
                var first: []const u8 = undefined;
                var second: []const u8 = undefined;
                switch (item) {
                    .string => |merge_str| {
                        const space_idx = std.mem.indexOfScalar(u8, merge_str, ' ') orelse continue;
                        first = merge_str[0..space_idx];
                        second = merge_str[space_idx + 1 ..];
                    },
                    .array => |arr| {
                        if (arr.items.len < 2) continue;
                        first = switch (arr.items[0]) {
                            .string => |s| s,
                            else => continue,
                        };
                        second = switch (arr.items[1]) {
                            .string => |s| s,
                            else => continue,
                        };
                    },
                    else => continue,
                }
                const key_len = first.len + 1 + second.len;
                const owned = try self.allocator.alloc(u8, key_len);
                errdefer self.allocator.free(owned);
                @memcpy(owned[0..first.len], first);
                owned[first.len] = 0;
                @memcpy(owned[first.len + 1 ..], second);
                try self.merges.put(owned, rank);
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

                if (self.vocab.contains(token_str)) {
                    self.vocab.getPtr(token_str).?.* = id;
                } else {
                    const owned_key = try self.allocator.dupe(u8, token_str);
                    errdefer self.allocator.free(owned_key);
                    try self.vocab.put(owned_key, id);
                }

                const owned_token = try self.allocator.dupe(u8, token_str);
                errdefer self.allocator.free(owned_token);
                if (self.ids_to_tokens.contains(id)) {
                    const old = self.ids_to_tokens.getPtr(id).?.*;
                    self.allocator.free(old);
                    self.ids_to_tokens.getPtr(id).?.* = owned_token;
                } else {
                    try self.ids_to_tokens.put(id, owned_token);
                }
            }
        }

        // === Extract special token IDs ===
        self.unk_id = self.vocab.get("<unk>");
        self.bos_id = self.vocab.get("<s>");
        self.eos_id = self.vocab.get("</s>");

        // === Load normalizer ===
        if (root.get("normalizer")) |norm_val| {
            if (norm_val != .null) {
                self.normalizer = try parseNormalizer(self.allocator, norm_val);
            }
        }

        // === Load pre-tokenizer ===
        if (root.get("pre_tokenizer")) |pt_val| {
            if (pt_val != .null) {
                self.pre_tokenizer = parsePreTokenizer(self.allocator, pt_val) catch |err| blk: {
                    // If we can't parse the pre-tokenizer, fall back to legacy behavior
                    std.log.warn("Failed to parse pre_tokenizer ({}), using fallback\n", .{err});
                    break :blk null;
                };
            }
        }

        // === Load decoder ===
        if (root.get("decoder")) |dec_val| {
            if (dec_val != .null) {
                self.decoder = parseDecoder(self.allocator, dec_val) catch |err| blk: {
                    std.log.warn("Failed to parse decoder ({}), using fallback\n", .{err});
                    break :blk null;
                };
            }
        }
    }

    pub fn encode(self: *BpeTokenizer, text: []const u8, add_special_tokens: bool) ![]u32 {
        var result = std.ArrayList(u32).empty;
        errdefer result.deinit(self.allocator);

        if (add_special_tokens) {
            if (self.bos_id) |id| try result.append(self.allocator, id);
        }

        // Detect legacy mode (old tokenizer.json without config fields)
        const is_legacy = (self.normalizer == .none and self.pre_tokenizer == null and self.decoder == null);

        // Normalize
        const normalized = if (is_legacy)
            try self.legacyNormalize(text)
        else
            try self.normalizer.normalize(self.allocator, text);
        defer self.allocator.free(normalized);

        // Pre-tokenize
        var segments: std.ArrayList([]const u8) = undefined;
        if (self.pre_tokenizer) |pre_tok| {
            segments = try pre_tok.preTokenize(self.allocator, normalized);
        } else if (is_legacy) {
            segments = try self.legacyPreTokenize(normalized);
        } else {
            segments = std.ArrayList([]const u8).empty;
            try segments.append(self.allocator, try self.allocator.dupe(u8, normalized));
        }
        defer {
            for (segments.items) |s| self.allocator.free(s);
            segments.deinit(self.allocator);
        }

        // Encode each segment independently
        for (segments.items) |segment| {
            // Split into code points (initial tokens)
            var tokens = try splitCodepoints(self.allocator, segment);
            defer {
                for (tokens.items) |t| self.allocator.free(t);
                tokens.deinit(self.allocator);
            }

            // Apply BPE merges
            while (true) {
                var best_rank: usize = std.math.maxInt(usize);
                var best_idx: usize = std.math.maxInt(usize);

                for (0..tokens.items.len -| 1) |i| {
                    const a = tokens.items[i];
                    const b = tokens.items[i + 1];
                    var key_buf: [512]u8 = undefined;
                    if (a.len + 1 + b.len > key_buf.len) continue;
                    @memcpy(key_buf[0..a.len], a);
                    key_buf[a.len] = 0;
                    @memcpy(key_buf[a.len + 1 .. a.len + 1 + b.len], b);
                    if (self.merges.get(key_buf[0 .. a.len + 1 + b.len])) |rank| {
                        if (rank < best_rank) {
                            best_rank = rank;
                            best_idx = i;
                        }
                    }
                }

                if (best_idx == std.math.maxInt(usize)) break;

                // Merge tokens[best_idx] and tokens[best_idx+1]
                const merged = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ tokens.items[best_idx], tokens.items[best_idx + 1] });
                self.allocator.free(tokens.items[best_idx]);
                self.allocator.free(tokens.items[best_idx + 1]);
                tokens.items[best_idx] = merged;
                _ = tokens.orderedRemove(best_idx + 1);
            }

            // Map tokens to IDs
            for (tokens.items) |token| {
                const id = self.vocab.get(token) orelse self.unk_id orelse 0;
                try result.append(self.allocator, id);
            }
        }

        if (add_special_tokens) {
            if (self.eos_id) |id| try result.append(self.allocator, id);
        }

        return result.toOwnedSlice(self.allocator);
    }

    pub fn decode(self: *BpeTokenizer, ids: []const u32) ![]const u8 {
        var builder = std.ArrayList(u8).empty;
        errdefer builder.deinit(self.allocator);

        for (ids) |id| {
            if (self.ids_to_tokens.get(id)) |token| {
                try builder.appendSlice(self.allocator, token);
            }
        }

        const fused = try builder.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(fused);

        const is_legacy = (self.normalizer == .none and self.pre_tokenizer == null and self.decoder == null);

        if (self.decoder) |dec| {
            const decoded = try dec.decode(self.allocator, fused);
            self.allocator.free(fused);
            return decoded;
        } else if (is_legacy) {
            const decoded = try legacyDecode(self.allocator, fused);
            self.allocator.free(fused);
            return decoded;
        }

        // No decoder configured: passthrough
        return fused;
    }

    pub fn vocabSize(self: BpeTokenizer) usize {
        return self.vocab.count();
    }

    // ------------------------------------------------------------------
    // Strategy interface
    // ------------------------------------------------------------------

    pub fn asStrategy(self: *BpeTokenizer) interface.TokenizerStrategy {
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
        const self: *BpeTokenizer = @ptrCast(@alignCast(ptr));
        return self.encode(text, add_special_tokens);
    }

    fn decodeImpl(ptr: *anyopaque, ids: []const u32, allocator: std.mem.Allocator) ![]const u8 {
        _ = allocator;
        const self: *BpeTokenizer = @ptrCast(@alignCast(ptr));
        return self.decode(ids);
    }

    fn vocabSizeImpl(ptr: *anyopaque) usize {
        const self: *BpeTokenizer = @ptrCast(@alignCast(ptr));
        return self.vocabSize();
    }

    fn deinitImpl(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        _ = allocator;
        const self: *BpeTokenizer = @ptrCast(@alignCast(ptr));
        self.deinit();
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn legacyNormalize(self: *BpeTokenizer, text: []const u8) error{OutOfMemory}![]const u8 {
        // Simplified normalizer: Prepend("▁") + Replace(" ", "▁")
        var result = std.ArrayList(u8).empty;
        errdefer result.deinit(self.allocator);

        // Prepend "▁" if text doesn't start with it
        if (!std.mem.startsWith(u8, text, "▁")) {
            try result.appendSlice(self.allocator, "▁");
        }

        // Replace spaces with "▁"
        for (text) |byte| {
            if (byte == ' ') {
                try result.appendSlice(self.allocator, "▁");
            } else {
                try result.append(self.allocator, byte);
            }
        }

        return result.toOwnedSlice(self.allocator);
    }

    /// Pre-tokenize: split by word boundaries (▁).
    /// Each segment starts with "▁" (except possibly the first if text is empty).
    fn legacyPreTokenize(self: *BpeTokenizer, text: []const u8) error{OutOfMemory}!std.ArrayList([]const u8) {
        var result = std.ArrayList([]const u8).empty;
        errdefer {
            for (result.items) |s| self.allocator.free(s);
            result.deinit(self.allocator);
        }

        const underscore = "▁";
        var start: usize = 0;
        var i: usize = 0;

        while (i < text.len) {
            if (std.mem.startsWith(u8, text[i..], underscore)) {
                if (i > start) {
                    const segment = try self.allocator.dupe(u8, text[start..i]);
                    errdefer self.allocator.free(segment);
                    try result.append(self.allocator, segment);
                }
                start = i;
                i += underscore.len;
            } else {
                i += 1;
            }
        }

        if (start < text.len) {
            const segment = try self.allocator.dupe(u8, text[start..]);
            errdefer self.allocator.free(segment);
            try result.append(self.allocator, segment);
        }

        return result;
    }
};

fn splitCodepoints(allocator: std.mem.Allocator, text: []const u8) error{ OutOfMemory, InvalidUtf8 }!std.ArrayList([]const u8) {
    var result = std.ArrayList([]const u8).empty;
    errdefer {
        for (result.items) |item| allocator.free(item);
        result.deinit(allocator);
    }

    const view = try std.unicode.Utf8View.init(text);
    var iter = view.iterator();
    while (iter.nextCodepointSlice()) |slice| {
        const copy = try allocator.dupe(u8, slice);
        errdefer allocator.free(copy);
        try result.append(allocator, copy);
    }
    return result;
}

fn legacyDecode(allocator: std.mem.Allocator, text: []const u8) error{OutOfMemory}![]const u8 {
    // Step 1: Replace "▁" with " "
    var after_replace = std.ArrayList(u8).empty;
    errdefer after_replace.deinit(allocator);

    var i: usize = 0;
    const underscore = "▁";
    while (i < text.len) {
        if (std.mem.startsWith(u8, text[i..], underscore)) {
            try after_replace.append(allocator, ' ');
            i += underscore.len;
        } else {
            try after_replace.append(allocator, text[i]);
            i += 1;
        }
    }

    // Step 2: ByteFallback (<0xXX>)
    var after_bytes = std.ArrayList(u8).empty;
    errdefer after_bytes.deinit(allocator);

    var j: usize = 0;
    while (j < after_replace.items.len) {
        if (j + 6 <= after_replace.items.len and
            after_replace.items[j] == '<' and
            after_replace.items[j + 1] == '0' and
            after_replace.items[j + 2] == 'x' and
            after_replace.items[j + 5] == '>')
        {
            const hex_str = after_replace.items[j + 3 .. j + 5];
            if (std.fmt.parseInt(u8, hex_str, 16)) |byte| {
                try after_bytes.append(allocator, byte);
                j += 6;
                continue;
            } else |_| {}
        }
        try after_bytes.append(allocator, after_replace.items[j]);
        j += 1;
    }

    // Step 3: Strip leading space (start=1, stop=0)
    var start: usize = 0;
    if (after_bytes.items.len > 0 and after_bytes.items[0] == ' ') {
        start = 1;
    }

    const final_len = after_bytes.items.len - start;
    const final = try allocator.alloc(u8, final_len);
    @memcpy(final, after_bytes.items[start..]);

    after_replace.deinit(allocator);
    after_bytes.deinit(allocator);
    return final;
}

// === JSON config parsers for normalizer / pre-tokenizer / decoder ===

fn parseNormalizer(allocator: std.mem.Allocator, val: std.json.Value) !pt.Normalizer {
    const typ = val.object.get("type") orelse return .none;
    const type_str = typ.string;

    if (std.mem.eql(u8, type_str, "Sequence")) {
        const arr = val.object.get("normalizers") orelse return .none;
        const items = try allocator.alloc(pt.Normalizer, arr.array.items.len);
        errdefer allocator.free(items);
        for (arr.array.items, 0..) |item, i| {
            items[i] = try parseNormalizer(allocator, item);
        }
        return .{ .sequence = .{ .normalizers = items } };
    } else if (std.mem.eql(u8, type_str, "Prepend")) {
        const prep = val.object.get("prepend") orelse return .none;
        const owned = try allocator.dupe(u8, prep.string);
        return .{ .prepend = .{ .prepend = owned } };
    } else if (std.mem.eql(u8, type_str, "Replace")) {
        const pattern = val.object.get("pattern") orelse return .none;
        const content = val.object.get("content") orelse return .none;
        const pat_str = if (pattern.object.get("String")) |s| s.string else pattern.string;
        const pat_owned = try allocator.dupe(u8, pat_str);
        errdefer allocator.free(pat_owned);
        const cont_owned = try allocator.dupe(u8, content.string);
        return .{ .replace = .{ .pattern = pat_owned, .content = cont_owned } };
    }

    return .none;
}

fn parsePreTokenizer(allocator: std.mem.Allocator, val: std.json.Value) !pt.PreTokenizer {
    const typ = val.object.get("type") orelse return error.InvalidPreTokenizer;
    const type_str = typ.string;

    if (std.mem.eql(u8, type_str, "Sequence")) {
        const arr = val.object.get("pretokenizers") orelse return error.InvalidPreTokenizer;
        const items = try allocator.alloc(pt.PreTokenizer, arr.array.items.len);
        errdefer allocator.free(items);
        for (arr.array.items, 0..) |item, i| {
            items[i] = try parsePreTokenizer(allocator, item);
        }
        return .{ .sequence = .{ .pretokenizers = items } };
    } else if (std.mem.eql(u8, type_str, "Split")) {
        const pattern_val = val.object.get("pattern") orelse return error.InvalidPreTokenizer;
        const behavior_val = val.object.get("behavior") orelse return error.InvalidPreTokenizer;
        const invert_val = val.object.get("invert") orelse return error.InvalidPreTokenizer;

        const pattern = try parseSplitPattern(pattern_val);
        const behavior = parseSplitBehavior(behavior_val.string);
        const invert = invert_val.bool;

        return .{ .split = .{ .pattern = pattern, .behavior = behavior, .invert = invert } };
    } else if (std.mem.eql(u8, type_str, "ByteLevel")) {
        const add_prefix_space = if (val.object.get("add_prefix_space")) |v| v.bool else false;
        return .{ .byte_level = .{ .add_prefix_space = add_prefix_space } };
    }

    return error.UnsupportedPreTokenizer;
}

fn parseSplitPattern(val: std.json.Value) !pt.Pattern {
    if (val.object.get("Regex")) |regex_val| {
        const pattern_str = regex_val.string;
        if (std.mem.eql(u8, pattern_str, "\\p{N}{1,3}")) {
            return .digits_1_3;
        } else if (std.mem.eql(u8, pattern_str, "[一-龥぀-ゟ゠-ヿ]+")) {
            return .cjk_kana;
        } else if (std.mem.indexOf(u8, pattern_str, "[A-Za-z]+|") != null) {
            // DeepSeek / cl100k_base complex word pattern
            return .deepseek_word;
        }
    }
    return error.UnsupportedPattern;
}

fn parseSplitBehavior(str: []const u8) pt.Split.Behavior {
    if (std.mem.eql(u8, str, "Isolated")) return .isolated;
    if (std.mem.eql(u8, str, "Removed")) return .removed;
    if (std.mem.eql(u8, str, "MergedWithPrevious")) return .merged_with_previous;
    if (std.mem.eql(u8, str, "MergedWithNext")) return .merged_with_next;
    if (std.mem.eql(u8, str, "Contiguous")) return .contiguous;
    return .isolated;
}

fn parseDecoder(allocator: std.mem.Allocator, val: std.json.Value) !pt.Decoder {
    const typ = val.object.get("type") orelse return error.InvalidDecoder;
    const type_str = typ.string;

    if (std.mem.eql(u8, type_str, "ByteLevel")) {
        const add_prefix_space = if (val.object.get("add_prefix_space")) |v| v.bool else true;
        return .{ .byte_level = .{ .add_prefix_space = add_prefix_space } };
    } else if (std.mem.eql(u8, type_str, "Replace")) {
        const pattern = val.object.get("pattern") orelse return error.InvalidDecoder;
        const content = val.object.get("content") orelse return error.InvalidDecoder;
        const pat_str = if (pattern.object.get("String")) |s| s.string else pattern.string;
        const pat_owned = try allocator.dupe(u8, pat_str);
        errdefer allocator.free(pat_owned);
        const cont_owned = try allocator.dupe(u8, content.string);
        return .{ .replace = .{ .pattern = pat_owned, .content = cont_owned } };
    } else if (std.mem.eql(u8, type_str, "Fuse")) {
        return .{ .fuse = .{} };
    } else if (std.mem.eql(u8, type_str, "Strip")) {
        const content = val.object.get("content") orelse return error.InvalidDecoder;
        const start = if (val.object.get("start")) |v| @as(usize, @intCast(v.integer)) else 0;
        const stop = if (val.object.get("stop")) |v| @as(usize, @intCast(v.integer)) else 0;
        const cont_owned = try allocator.dupe(u8, content.string);
        return .{ .strip = .{ .content = cont_owned, .start = start, .stop = stop } };
    } else if (std.mem.eql(u8, type_str, "ByteFallback")) {
        return .{ .byte_fallback = .{} };
    } else if (std.mem.eql(u8, type_str, "Sequence")) {
        const arr = val.object.get("decoders") orelse return error.InvalidDecoder;
        const items = try allocator.alloc(pt.Decoder, arr.array.items.len);
        errdefer allocator.free(items);
        for (arr.array.items, 0..) |item, i| {
            items[i] = try parseDecoder(allocator, item);
        }
        return .{ .sequence = .{ .decoders = items } };
    }

    return error.UnsupportedDecoder;
}
