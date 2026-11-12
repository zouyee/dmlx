/// POST /v1/tokenize (and POST /tokenize alias).
///
/// Raw-text tokenization endpoint for agent frameworks doing context
/// budgeting. The request `model` field is accepted but ignored — this is a
/// single-model server and the startup tokenizer is always used. Encoding is
/// raw BPE with NO chat template: counts match what the model sees for plain
/// text, not for a templated chat prompt.
///
/// `add_special` (default false) only takes effect when the loaded tokenizer
/// defines BOS/EOS ids (e.g. Llama-style tokenizers). Qwen BPE tokenizers
/// carry no BOS/EOS concept, so for them the flag is accepted and ignored.
///
/// Response: {"tokens": [id, ...], "count": N}
/// Errors:   OpenAI-style {"error":{...,"type":"invalid_request_error",...}}
const std = @import("std");
const root = @import("../root.zig");

const TokenizerStrategy = root.tokenizer.TokenizerStrategy;

/// Cheap DoS guard: reject text payloads above 1MB.
pub const max_text_bytes: usize = 1024 * 1024;

const TokenizeRequest = struct {
    model: ?[]const u8 = null, // ignored: single-model server
    text: ?[]const u8 = null,
    add_special: ?bool = null,
};

pub const TokenizeResult = struct {
    status: u16,
    body: []u8,
};

fn invalidRequestJson(allocator: std.mem.Allocator, code: []const u8, message: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\{{"error":{{"message":"{s}","type":"invalid_request_error","code":"{s}"}}}}
    , .{ message, code });
}

/// Build the success body: {"tokens":[...],"count":N}.
pub fn tokenizeJson(
    allocator: std.mem.Allocator,
    strategy: TokenizerStrategy,
    text: []const u8,
    add_special: bool,
) ![]u8 {
    const ids = try strategy.encode(text, add_special, allocator);
    defer allocator.free(ids);

    var out = std.ArrayList(u8).empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"tokens\":[");
    for (ids, 0..) |id, i| {
        if (i != 0) try out.append(allocator, ',');
        const num = try std.fmt.allocPrint(allocator, "{d}", .{id});
        defer allocator.free(num);
        try out.appendSlice(allocator, num);
    }
    const tail = try std.fmt.allocPrint(allocator, "],\"count\":{d}}}", .{ids.len});
    defer allocator.free(tail);
    try out.appendSlice(allocator, tail);
    return out.toOwnedSlice(allocator);
}

/// Parse + validate + tokenize a request body. Caller frees result.body.
/// All client errors map to a 400 invalid_request_error body.
pub fn handleTokenizeRequest(
    allocator: std.mem.Allocator,
    strategy: TokenizerStrategy,
    body: []const u8,
) !TokenizeResult {
    const parsed = std.json.parseFromSlice(TokenizeRequest, allocator, body, .{
        .ignore_unknown_fields = true,
    }) catch {
        return .{
            .status = 400,
            .body = try invalidRequestJson(allocator, "invalid_json", "request body must be a JSON object"),
        };
    };
    defer parsed.deinit();

    const text = parsed.value.text orelse {
        return .{
            .status = 400,
            .body = try invalidRequestJson(allocator, "missing_text", "text is required and must be a string"),
        };
    };

    if (text.len > max_text_bytes) {
        return .{
            .status = 400,
            .body = try invalidRequestJson(allocator, "text_too_large", "text exceeds the 1MB limit"),
        };
    }

    return .{
        .status = 200,
        .body = try tokenizeJson(allocator, strategy, text, parsed.value.add_special orelse false),
    };
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

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
    \\  },
    \\  "post_processor": {
    \\    "special_tokens": {
    \\      "bos_token": {"id": 10},
    \\      "eos_token": {"id": 11}
    \\    }
    \\  }
    \\}
;

fn testStrategy(allocator: std.mem.Allocator, backend: *root.tokenizer.BpeTokenizer) !TokenizerStrategy {
    backend.* = root.tokenizer.BpeTokenizer.init(allocator);
    try backend.loadFromJson(bpe_test_json);
    return backend.asStrategy();
}

test "handleTokenizeRequest encodes raw text with no specials by default" {
    const allocator = std.testing.allocator;
    var backend: root.tokenizer.BpeTokenizer = undefined;
    defer backend.deinit();
    const strategy = try testStrategy(allocator, &backend);

    const result = try handleTokenizeRequest(allocator, strategy, "{\"model\":\"whatever\",\"text\":\"hello\"}");
    defer allocator.free(result.body);

    try std.testing.expectEqual(@as(u16, 200), result.status);
    // Same ids as the BpeTokenizer unit test: "hello" -> [5, 2, 7, 4].
    try std.testing.expectEqualStrings("{\"tokens\":[5,2,7,4],\"count\":4}", result.body);
}

test "handleTokenizeRequest add_special adds BOS/EOS when the tokenizer defines them" {
    const allocator = std.testing.allocator;
    var backend: root.tokenizer.BpeTokenizer = undefined;
    defer backend.deinit();
    const strategy = try testStrategy(allocator, &backend);

    const result = try handleTokenizeRequest(allocator, strategy, "{\"text\":\"hello\",\"add_special\":true}");
    defer allocator.free(result.body);

    try std.testing.expectEqual(@as(u16, 200), result.status);
    try std.testing.expectEqualStrings("{\"tokens\":[10,5,2,7,4,11],\"count\":6}", result.body);
}

test "handleTokenizeRequest rejects missing or non-string text" {
    const allocator = std.testing.allocator;
    var backend: root.tokenizer.BpeTokenizer = undefined;
    defer backend.deinit();
    const strategy = try testStrategy(allocator, &backend);

    for ([_][]const u8{
        "{\"model\":\"x\"}",
        "{\"text\":123}",
        "not json",
    }) |body| {
        const result = try handleTokenizeRequest(allocator, strategy, body);
        defer allocator.free(result.body);
        try std.testing.expectEqual(@as(u16, 400), result.status);
        try std.testing.expect(std.mem.indexOf(u8, result.body, "invalid_request_error") != null);
    }
}

test "handleTokenizeRequest rejects text above the 1MB cap" {
    const allocator = std.testing.allocator;
    var backend: root.tokenizer.BpeTokenizer = undefined;
    defer backend.deinit();
    const strategy = try testStrategy(allocator, &backend);

    var body = std.ArrayList(u8).empty;
    defer body.deinit(allocator);
    try body.appendSlice(allocator, "{\"text\":\"");
    try body.appendNTimes(allocator, 'a', max_text_bytes + 1);
    try body.appendSlice(allocator, "\"}");

    const result = try handleTokenizeRequest(allocator, strategy, body.items);
    defer allocator.free(result.body);
    try std.testing.expectEqual(@as(u16, 400), result.status);
    try std.testing.expect(std.mem.indexOf(u8, result.body, "text_too_large") != null);
}

test "handleTokenizeRequest matches the real Qwen tokenizer (golden parity)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const tok_path = "/Users/zouyee/models/Qwen3.8-27B-4bit/tokenizer.json";

    std.Io.Dir.cwd().access(io, tok_path, .{}) catch return error.SkipZigTest;

    var backend = root.tokenizer.BpeTokenizer.init(allocator);
    defer backend.deinit();
    try backend.loadFromFile(io, tok_path);
    const strategy = backend.asStrategy();

    // Golden ids from tests/golden/qwen38_tokenizer_parity.json (HF tokenizers).
    const result = try handleTokenizeRequest(allocator, strategy, "{\"text\":\"Hello world\"}");
    defer allocator.free(result.body);

    try std.testing.expectEqual(@as(u16, 200), result.status);
    try std.testing.expectEqualStrings("{\"tokens\":[9419,1814],\"count\":2}", result.body);

    // Qwen BPE has no BOS/EOS: add_special must be accepted and ignored.
    const with_special = try handleTokenizeRequest(allocator, strategy, "{\"text\":\"Hello world\",\"add_special\":true}");
    defer allocator.free(with_special.body);
    try std.testing.expectEqualStrings(result.body, with_special.body);
}
// probe comment
