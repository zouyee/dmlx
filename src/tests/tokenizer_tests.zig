const std = @import("std");
const mlx = @import("../root.zig");

const GreedyTokenizer = mlx.tokenizer.GreedyTokenizer;
const BpeTokenizer = mlx.tokenizer.BpeTokenizer;
const TokenizerStrategy = mlx.tokenizer.TokenizerStrategy;

const test_json =
    \\{
    \\  "model": {
    \\    "vocab": {
    \\      "hello": 0,
    \\      "world": 1,
    \\      " ": 2,
    \\      "<s>": 3,
    \\      "</s>": 4,
    \\      "<unk>": 5
    \\    }
    \\  },
    \\  "post_processor": {
    \\    "special_tokens": {
    \\      "bos_token": {"id": 3},
    \\      "eos_token": {"id": 4},
    \\      "unk_token": {"id": 5}
    \\    }
    \\  }
    \\}
;

test "GreedyTokenizer loadFromJson populates vocab" {
    const allocator = std.testing.allocator;
    var tokenizer = GreedyTokenizer.init(allocator);
    defer tokenizer.deinit();

    try tokenizer.loadFromJson(test_json);

    try std.testing.expectEqual(@as(usize, 6), tokenizer.vocabSize());
    try std.testing.expectEqual(@as(u32, 0), tokenizer.vocab.get("hello").?);
    try std.testing.expectEqual(@as(u32, 1), tokenizer.vocab.get("world").?);
}

test "GreedyTokenizer encode without special tokens" {
    const allocator = std.testing.allocator;
    var tokenizer = GreedyTokenizer.init(allocator);
    defer tokenizer.deinit();

    try tokenizer.loadFromJson(test_json);

    const ids = try tokenizer.encode("hello world", false);
    defer allocator.free(ids);

    // Greedy longest match: "hello" (5 chars) -> 0, " " -> 2, "world" -> 1
    try std.testing.expectEqualSlices(u32, &[_]u32{ 0, 2, 1 }, ids);
}

test "GreedyTokenizer encode with special tokens" {
    const allocator = std.testing.allocator;
    var tokenizer = GreedyTokenizer.init(allocator);
    defer tokenizer.deinit();

    try tokenizer.loadFromJson(test_json);

    const ids = try tokenizer.encode("hello", true);
    defer allocator.free(ids);

    // BOS + hello + EOS
    try std.testing.expectEqualSlices(u32, &[_]u32{ 3, 0, 4 }, ids);
}

test "GreedyTokenizer decode" {
    const allocator = std.testing.allocator;
    var tokenizer = GreedyTokenizer.init(allocator);
    defer tokenizer.deinit();

    try tokenizer.loadFromJson(test_json);

    const text = try tokenizer.decode(&[_]u32{ 0, 2, 1 });
    defer allocator.free(text);

    try std.testing.expectEqualStrings("hello world", text);
}

test "GreedyTokenizer roundtrip" {
    const allocator = std.testing.allocator;
    var tokenizer = GreedyTokenizer.init(allocator);
    defer tokenizer.deinit();

    try tokenizer.loadFromJson(test_json);

    const original = "hello world";
    const ids = try tokenizer.encode(original, false);
    defer allocator.free(ids);

    const decoded = try tokenizer.decode(ids);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings(original, decoded);
}

test "TokenizerStrategy interface works" {
    const allocator = std.testing.allocator;
    var backend = GreedyTokenizer.init(allocator);
    defer backend.deinit();

    try backend.loadFromJson(test_json);

    var strategy = backend.asStrategy();
    try std.testing.expectEqual(@as(usize, 6), strategy.vocabSize());

    const ids = try strategy.encode("hello", false, allocator);
    defer allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{0}, ids);

    const text = try strategy.decode(&[_]u32{0, 2, 1}, allocator);
    defer allocator.free(text);
    try std.testing.expectEqualStrings("hello world", text);
}

test "GreedyTokenizer decodeToken byte fallback" {
    const allocator = std.testing.allocator;
    const decoded = try decodeTokenInternal(allocator, "<0x0A>");
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings("\n", decoded);
}

const bpe_test_json =
    \\{
    \\  "model": {
    \\    "type": "BPE",
    \\    "vocab": {
    \\      "▁": 0,
    \\      "h": 1,
    \\      "e": 2,
    \\      "l": 3,
    \\      "o": 4,
    \\      "▁h": 5,
    \\      "he": 6,
    \\      "ll": 7,
    \\      "hello": 8,
    \\      "<unk>": 9,
    \\      "<s>": 10,
    \\      "</s>": 11
    \\    },
    \\    "merges": [
    \\      "▁ h",
    \\      "h e",
    \\      "l l",
    \\      "he ll",
    \\      "hell o"
    \\    ]
    \\  }
    \\}
;

test "BpeTokenizer loadFromJson populates vocab and merges" {
    const allocator = std.testing.allocator;
    var tokenizer = BpeTokenizer.init(allocator);
    defer tokenizer.deinit();

    try tokenizer.loadFromJson(bpe_test_json);

    try std.testing.expectEqual(@as(usize, 12), tokenizer.vocabSize());
    try std.testing.expectEqual(@as(u32, 8), tokenizer.vocab.get("hello").?);
    try std.testing.expectEqual(@as(usize, 5), tokenizer.merges.count());
}

test "BpeTokenizer encode applies BPE merges" {
    const allocator = std.testing.allocator;
    var tokenizer = BpeTokenizer.init(allocator);
    defer tokenizer.deinit();

    try tokenizer.loadFromJson(bpe_test_json);

    // "hello" normalizes to "▁hello", then BPE splits into code points and merges
    // ▁ + h + e + l + l + o
    // First merge: "▁ h" (rank 0) -> "▁h"
    // Then: "l l" (rank 2) -> "ll"
    // No more merges available ("h e" is no longer adjacent after "▁ h" merge)
    const ids = try tokenizer.encode("hello", false);
    defer allocator.free(ids);

    try std.testing.expectEqualSlices(u32, &[_]u32{ 5, 2, 7, 4 }, ids);
}

test "BpeTokenizer decode and roundtrip" {
    const allocator = std.testing.allocator;
    var tokenizer = BpeTokenizer.init(allocator);
    defer tokenizer.deinit();

    try tokenizer.loadFromJson(bpe_test_json);

    const ids = try tokenizer.encode("hello", false);
    defer allocator.free(ids);

    const text = try tokenizer.decode(ids);
    defer allocator.free(text);

    try std.testing.expectEqualStrings("hello", text);
}

test "BpeTokenizer encode with special tokens" {
    const allocator = std.testing.allocator;
    var tokenizer = BpeTokenizer.init(allocator);
    defer tokenizer.deinit();

    try tokenizer.loadFromJson(bpe_test_json);

    const ids = try tokenizer.encode("hello", true);
    defer allocator.free(ids);

    // <s> = 10, then encoded tokens, then </s> = 11
    try std.testing.expectEqualSlices(u32, &[_]u32{ 10, 5, 2, 7, 4, 11 }, ids);
}

test "BpeTokenizer encode multi-word with pre-tokenizer" {
    const allocator = std.testing.allocator;
    var tokenizer = BpeTokenizer.init(allocator);
    defer tokenizer.deinit();

    try tokenizer.loadFromJson(bpe_test_json);

    // "hello hello" normalizes to "▁hello▁hello", pre-tokenized into
    // ["▁hello", "▁hello"]. Each segment encodes independently.
    const ids = try tokenizer.encode("hello hello", false);
    defer allocator.free(ids);

    // Each "hello" encodes to [5, 2, 7, 4]
    try std.testing.expectEqualSlices(u32, &[_]u32{ 5, 2, 7, 4, 5, 2, 7, 4 }, ids);
}

test "BpeTokenizer decode with byte fallback" {
    const allocator = std.testing.allocator;
    var tokenizer = BpeTokenizer.init(allocator);
    defer tokenizer.deinit();

    // Construct a tiny tokenizer with a byte-fallback token
    const json =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "<0x0A>": 0,
        \\      "<unk>": 1,
        \\      "<s>": 2,
        \\      "</s>": 3
        \\    },
        \\    "merges": []
        \\  }
        \\}
    ;
    try tokenizer.loadFromJson(json);

    const text = try tokenizer.decode(&[_]u32{0});
    defer allocator.free(text);

    try std.testing.expectEqualStrings("\n", text);
}

test "BpeTokenizer asStrategy interface" {
    const allocator = std.testing.allocator;
    var backend = BpeTokenizer.init(allocator);
    defer backend.deinit();

    try backend.loadFromJson(bpe_test_json);

    var strategy = backend.asStrategy();
    try std.testing.expectEqual(@as(usize, 12), strategy.vocabSize());

    const ids = try strategy.encode("hello", false, allocator);
    defer allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 5, 2, 7, 4 }, ids);

    const text = try strategy.decode(ids, allocator);
    defer allocator.free(text);
    try std.testing.expectEqualStrings("hello", text);
}

test "BpeTokenizer TinyLlama integration" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const path = "/tmp/tiny_tokenizer.json";

    var tokenizer = BpeTokenizer.init(allocator);
    defer tokenizer.deinit();

    tokenizer.loadFromFile(io, path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    // Reference values from Python tokenizers library (Tokenizer.from_file)
    // "Hello world" -> [15043, 3186]
    {
        const ids = try tokenizer.encode("Hello world", false);
        defer allocator.free(ids);
        try std.testing.expectEqualSlices(u32, &[_]u32{ 15043, 3186 }, ids);

        const text = try tokenizer.decode(ids);
        defer allocator.free(text);
        try std.testing.expectEqualStrings("Hello world", text);
    }

    // "hello" -> [22172]
    {
        const ids = try tokenizer.encode("hello", false);
        defer allocator.free(ids);
        try std.testing.expectEqualSlices(u32, &[_]u32{22172}, ids);

        const text = try tokenizer.decode(ids);
        defer allocator.free(text);
        try std.testing.expectEqualStrings("hello", text);
    }

    // "This is a test." -> [910, 338, 263, 1243, 29889]
    {
        const ids = try tokenizer.encode("This is a test.", false);
        defer allocator.free(ids);
        try std.testing.expectEqualSlices(u32, &[_]u32{ 910, 338, 263, 1243, 29889 }, ids);

        const text = try tokenizer.decode(ids);
        defer allocator.free(text);
        try std.testing.expectEqualStrings("This is a test.", text);
    }

    // "你好世界" -> [29871, 30919, 31076, 30793, 30967]
    {
        const ids = try tokenizer.encode("你好世界", false);
        defer allocator.free(ids);
        try std.testing.expectEqualSlices(u32, &[_]u32{ 29871, 30919, 31076, 30793, 30967 }, ids);

        const text = try tokenizer.decode(ids);
        defer allocator.free(text);
        try std.testing.expectEqualStrings("你好世界", text);
    }
}

// Re-export decodeToken for testing (it's private in greedy.zig)
const greedy = @import("../tokenizer/greedy.zig");
fn decodeTokenInternal(allocator: std.mem.Allocator, token: []const u8) ![]const u8 {
    // This is a hack: we can't access private functions, but we can test decode behavior
    // through the public decode() by constructing a tokenizer with that token.
    var tokenizer = GreedyTokenizer.init(allocator);
    defer tokenizer.deinit();

    const json = try std.fmt.allocPrint(allocator,
        \\{{"model":{{"vocab":{{"{s}":99}}}},"post_processor":{{}}}}
    , .{token});
    defer allocator.free(json);

    try tokenizer.loadFromJson(json);
    return tokenizer.decode(&[_]u32{99});
}


// ============================================================================
// Pre-tokenizer unit tests
// ============================================================================

const pt = @import("../tokenizer/pre_tokenizer.zig");

test "bytesToUnicode maps ASCII printable to itself" {
    // 'A' (0x41) should map to itself
    const table = pt.getBytesToUnicode();
    try std.testing.expectEqual(@as(u21, 'A'), table[0x41]);
    try std.testing.expectEqual(@as(u21, 'z'), table[0x7A]);
    // Space (0x20) is NOT in the self-mapping list, so it maps to 0x100+
    try std.testing.expect(table[0x20] >= 0x100);
}

test "ByteLevel pre-tokenizer maps bytes to visible chars" {
    const allocator = std.testing.allocator;
    const bl = pt.ByteLevel{ .add_prefix_space = false };
    var result = try bl.preTokenize(allocator, "hi");
    defer {
        for (result.items) |s| allocator.free(s);
        result.deinit(allocator);
    }
    try std.testing.expectEqual(@as(usize, 1), result.items.len);
    // "h" and "i" are ASCII printable, so they map to themselves
    try std.testing.expectEqualStrings("hi", result.items[0]);
}

test "ByteLevel pre-tokenizer with add_prefix_space" {
    const allocator = std.testing.allocator;
    const bl = pt.ByteLevel{ .add_prefix_space = true };
    var result = try bl.preTokenize(allocator, "hi");
    defer {
        for (result.items) |s| allocator.free(s);
        result.deinit(allocator);
    }
    // Space (0x20) maps to a visible char; since "hi" doesn't start with space,
    // a prefix space is added.
    try std.testing.expect(result.items[0].len > 2);
}

test "Split digits_1_3 pattern" {
    const allocator = std.testing.allocator;
    const split = pt.Split{ .pattern = .digits_1_3, .behavior = .isolated, .invert = false };
    var result = try split.preTokenize(allocator, "abc123def45");
    defer {
        for (result.items) |s| allocator.free(s);
        result.deinit(allocator);
    }
    try std.testing.expectEqual(@as(usize, 4), result.items.len);
    try std.testing.expectEqualStrings("abc", result.items[0]);
    try std.testing.expectEqualStrings("123", result.items[1]);
    try std.testing.expectEqualStrings("def", result.items[2]);
    try std.testing.expectEqualStrings("45", result.items[3]);
}

test "Split cjk_kana pattern" {
    const allocator = std.testing.allocator;
    const split = pt.Split{ .pattern = .cjk_kana, .behavior = .isolated, .invert = false };
    var result = try split.preTokenize(allocator, "a你好b");
    defer {
        for (result.items) |s| allocator.free(s);
        result.deinit(allocator);
    }
    try std.testing.expectEqual(@as(usize, 3), result.items.len);
    try std.testing.expectEqualStrings("a", result.items[0]);
    try std.testing.expectEqualStrings("你好", result.items[1]);
    try std.testing.expectEqualStrings("b", result.items[2]);
}

test "Split deepseek_word pattern on ASCII" {
    const allocator = std.testing.allocator;
    const split = pt.Split{ .pattern = .deepseek_word, .behavior = .isolated, .invert = false };
    var result = try split.preTokenize(allocator, "hello world");
    defer {
        for (result.items) |s| allocator.free(s);
        result.deinit(allocator);
    }
    // DeepSeek pattern left-snaps space to next word: ["hello", " world"]
    try std.testing.expectEqual(@as(usize, 2), result.items.len);
    try std.testing.expectEqualStrings("hello", result.items[0]);
    try std.testing.expectEqualStrings(" world", result.items[1]);
}

test "Sequence pre-tokenizer composes splits" {
    const allocator = std.testing.allocator;
    const items = try allocator.alloc(pt.PreTokenizer, 2);
    items[0] = .{ .split = .{ .pattern = .digits_1_3, .behavior = .isolated, .invert = false } };
    items[1] = .{ .split = .{ .pattern = .cjk_kana, .behavior = .isolated, .invert = false } };
    var seq = pt.Sequence{ .pretokenizers = items };
    defer seq.deinit(allocator);

    var result = try seq.preTokenize(allocator, "a123你好b");
    defer {
        for (result.items) |s| allocator.free(s);
        result.deinit(allocator);
    }
    // First split digits: ["a", "123", "你好b"]
    // Then split CJK: ["a", "123", "", "你好", "b"]
    try std.testing.expect(result.items.len >= 4);
}

test "ByteFallbackDecoder" {
    const allocator = std.testing.allocator;
    const bf = pt.ByteFallbackDecoder{};
    const decoded = try bf.decode(allocator, "a<0x0A>b");
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings("a\nb", decoded);
}

test "StripDecoder removes prefix/suffix" {
    const allocator = std.testing.allocator;
    const strip = pt.StripDecoder{ .content = " ", .start = 1, .stop = 0 };
    const decoded = try strip.decode(allocator, " Hello world");
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings("Hello world", decoded);
}

test "DeepSeek-style tokenizer JSON load and encode/decode" {
    const allocator = std.testing.allocator;
    var tokenizer = BpeTokenizer.init(allocator);
    defer tokenizer.deinit();

    // Minimal DeepSeek-like tokenizer.json
    const json =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "<unk>": 0,
        \\      "<｜begin▁of▁sentence｜>": 1,
        \\      "<｜end▁of▁sentence｜>": 2,
        \\      "h": 3,
        \\      "e": 4,
        \\      "l": 5,
        \\      "o": 6,
        \\      "he": 7,
        \\      "ll": 8,
        \\      "hello": 9
        \\    },
        \\    "merges": [
        \\      "h e",
        \\      "l l"
        \\    ]
        \\  },
        \\  "pre_tokenizer": {
        \\    "type": "Sequence",
        \\    "pretokenizers": [
        \\      {"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated", "invert": false},
        \\      {"type": "Split", "pattern": {"Regex": "[一-龥぀-ゟ゠-ヿ]+"}, "behavior": "Isolated", "invert": false},
        \\      {"type": "Split", "pattern": {"Regex": "[!\\\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
        \\      {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
        \\    ]
        \\  },
        \\  "decoder": {
        \\    "type": "ByteLevel",
        \\    "add_prefix_space": true,
        \\    "trim_offsets": true,
        \\    "use_regex": true
        \\  }
        \\}
    ;

    try tokenizer.loadFromJson(json);

    // Verify pre_tokenizer was loaded
    try std.testing.expect(tokenizer.pre_tokenizer != null);
    try std.testing.expect(tokenizer.decoder != null);

    // "hello" should encode through the pipeline
    const ids = try tokenizer.encode("hello", false);
    defer allocator.free(ids);
    try std.testing.expect(ids.len > 0);

    // Round-trip decode
    const text = try tokenizer.decode(ids);
    defer allocator.free(text);
    try std.testing.expectEqualStrings("hello", text);
}
