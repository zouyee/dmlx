/// POST /v1/detokenize (+ /detokenize alias) and
/// POST /v1/messages/count_tokens.
///
/// detokenize: {"model","tokens":[id,...]} → {"text": "..."}. Uses the
/// tokenizer's FULL-SEQUENCE decode (BpeTokenizer.decode: concat token
/// strings, then run the decoder once), which preserves leading spaces —
/// the known streamed per-token decode bug (HANDOFF: "Streaming SSE text
/// loses spaces") does not apply here because that path decodes one token
/// at a time; the round-trip test below pins this.
///
/// count_tokens (Anthropic shape): {"model","messages":[...]} →
/// {"input_tokens": N}. Applies the REAL chat template and tokenizer via
/// openai.countPromptTokens, so N equals usage.prompt_tokens for the same
/// messages sent to /v1/chat/completions. Image parts contribute their
/// preprocess pad count (no tower forward) — see countPromptTokens.
const std = @import("std");
const root = @import("../root.zig");
const state_mod = @import("state.zig");
const openai_mod = @import("openai.zig");
const utils_mod = @import("utils.zig");

const TokenizerStrategy = root.tokenizer.TokenizerStrategy;
const ServerState = state_mod.ServerState;

/// DoS guard on the tokens array length.
pub const max_tokens_len: usize = 100_000;

pub const DetokenizeResult = struct {
    status: u16,
    body: []u8,
};

fn invalidRequestJson(allocator: std.mem.Allocator, code: []const u8, message: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\{{"error":{{"message":"{s}","type":"invalid_request_error","code":"{s}"}}}}
    , .{ message, code });
}

/// Parse {"tokens":[...]}, validating every element. Returns owned []u32.
pub fn parseTokenIds(allocator: std.mem.Allocator, body: []const u8) !union(enum) {
    ok: []u32,
    err: []u8,
} {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        return .{ .err = try invalidRequestJson(allocator, "invalid_json", "request body must be a JSON object") };
    };
    defer parsed.deinit();
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return .{ .err = try invalidRequestJson(allocator, "invalid_json", "request body must be a JSON object") },
    };
    const arr = switch (obj.get("tokens") orelse
        return .{ .err = try invalidRequestJson(allocator, "missing_tokens", "'tokens' is required and must be an array of non-negative integers") }) {
        .array => |a| a,
        else => return .{ .err = try invalidRequestJson(allocator, "invalid_tokens", "'tokens' must be an array of non-negative integers") },
    };
    if (arr.items.len > max_tokens_len) {
        return .{ .err = try invalidRequestJson(allocator, "too_many_tokens", "tokens array exceeds the 100k limit") };
    }
    const ids = try allocator.alloc(u32, arr.items.len);
    errdefer allocator.free(ids);
    for (arr.items, 0..) |item, i| {
        switch (item) {
            .integer => |n| {
                if (n < 0 or n > std.math.maxInt(u32)) {
                    allocator.free(ids);
                    return .{ .err = try invalidRequestJson(allocator, "invalid_token_id", "token ids must be non-negative integers") };
                }
                ids[i] = @intCast(n);
            },
            else => {
                allocator.free(ids);
                return .{ .err = try invalidRequestJson(allocator, "invalid_token_id", "token ids must be non-negative integers") };
            },
        }
    }
    return .{ .ok = ids };
}

/// POST /v1/detokenize handler. Caller frees result.body.
pub fn handleDetokenizeRequest(
    allocator: std.mem.Allocator,
    strategy: TokenizerStrategy,
    body: []const u8,
) !DetokenizeResult {
    const outcome = try parseTokenIds(allocator, body);
    const ids = switch (outcome) {
        .ok => |ids| ids,
        .err => |b| return .{ .status = 400, .body = b },
    };
    defer allocator.free(ids);

    const text = try strategy.decode(ids, allocator);
    defer allocator.free(@constCast(text));

    var escaped = std.ArrayList(u8).empty;
    defer escaped.deinit(allocator);
    try utils_mod.jsonEscapeInto(&escaped, allocator, text);
    return .{
        .status = 200,
        .body = try std.fmt.allocPrint(allocator, "{{\"text\":\"{s}\"}}", .{escaped.items}),
    };
}

/// POST /v1/messages/count_tokens handler. Anthropic request shape:
/// {"model","messages":[{"role","content"}], "system"?: string}. Content
/// may be a string or an array of text/image_url parts (same as chat).
pub fn handleCountTokensRequest(
    allocator: std.mem.Allocator,
    state: *ServerState,
    body: []const u8,
) !DetokenizeResult {
    const CountRequest = struct {
        model: ?[]const u8 = null, // ignored: single-model server
        messages: []openai_mod.ChatCompletionRequest.Message,
        system: ?[]const u8 = null,
    };
    const parsed = std.json.parseFromSlice(CountRequest, allocator, body, .{
        .ignore_unknown_fields = true,
    }) catch {
        return .{
            .status = 400,
            .body = try invalidRequestJson(allocator, "invalid_json", "request body must be a JSON object with a 'messages' array"),
        };
    };
    defer parsed.deinit();
    const req = parsed.value;

    if (req.messages.len == 0) {
        return .{
            .status = 400,
            .body = try invalidRequestJson(allocator, "missing_messages", "'messages' must be a non-empty array"),
        };
    }

    const chat_req = openai_mod.ChatCompletionRequest{
        .model = req.model orelse "default",
        .messages = req.messages,
    };
    const n = openai_mod.countPromptTokens(allocator, state, chat_req, req.system) catch |err| {
        const msg: []const u8 = switch (err) {
            error.VisionNotSupported => "vision input is not supported by the loaded model",
            error.ImageTooLarge => "decoded image exceeds the 20MB limit",
            error.BadImageUrl => "unsupported image_url (data:...;base64 URLs and absolute file paths only)",
            error.InvalidContent => "invalid message content (string or text/image_url parts only)",
            else => return err,
        };
        return .{
            .status = 400,
            .body = try invalidRequestJson(allocator, @errorName(err), msg),
        };
    };

    return .{
        .status = 200,
        .body = try std.fmt.allocPrint(allocator, "{{\"input_tokens\":{d}}}", .{n}),
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
    \\  }
    \\}
;

test "handleDetokenizeRequest round-trips the toy tokenizer" {
    const allocator = std.testing.allocator;
    var backend = root.tokenizer.BpeTokenizer.init(allocator);
    defer backend.deinit();
    try backend.loadFromJson(bpe_test_json);
    const strategy = backend.asStrategy();

    // "hello" encodes to [5, 2, 7, 4] (tokenize.zig golden); detokenize
    // must hand back exactly "hello".
    const result = try handleDetokenizeRequest(allocator, strategy, "{\"model\":\"m\",\"tokens\":[5,2,7,4]}");
    defer allocator.free(result.body);
    try std.testing.expectEqual(@as(u16, 200), result.status);
    try std.testing.expectEqualStrings("{\"text\":\"hello\"}", result.body);

    // Full-sequence decode preserves the leading ▁ space mapping.
    const spaced = try handleDetokenizeRequest(allocator, strategy, "{\"tokens\":[0,5,2,7,4]}");
    defer allocator.free(spaced.body);
    try std.testing.expectEqualStrings("{\"text\":\" hello\"}", spaced.body);
}

test "handleDetokenizeRequest validation" {
    const allocator = std.testing.allocator;
    var backend = root.tokenizer.BpeTokenizer.init(allocator);
    defer backend.deinit();
    try backend.loadFromJson(bpe_test_json);
    const strategy = backend.asStrategy();

    for ([_][]const u8{
        "{\"model\":\"m\"}",
        "{\"tokens\":\"nope\"}",
        "{\"tokens\":[-1]}",
        "{\"tokens\":[1.5]}",
        "{\"tokens\":[4294967296]}",
        "not json",
    }) |body| {
        const result = try handleDetokenizeRequest(allocator, strategy, body);
        defer allocator.free(result.body);
        try std.testing.expectEqual(@as(u16, 400), result.status);
        try std.testing.expect(std.mem.indexOf(u8, result.body, "invalid_request_error") != null);
    }

    // Empty array is valid and decodes to the empty string.
    const empty = try handleDetokenizeRequest(allocator, strategy, "{\"tokens\":[]}");
    defer allocator.free(empty.body);
    try std.testing.expectEqual(@as(u16, 200), empty.status);
    try std.testing.expectEqualStrings("{\"text\":\"\"}", empty.body);
}

test "handleDetokenizeRequest rejects arrays above the 100k cap" {
    const allocator = std.testing.allocator;
    var backend = root.tokenizer.BpeTokenizer.init(allocator);
    defer backend.deinit();
    try backend.loadFromJson(bpe_test_json);
    const strategy = backend.asStrategy();

    var body = std.ArrayList(u8).empty;
    defer body.deinit(allocator);
    try body.appendSlice(allocator, "{\"tokens\":[");
    for (0..max_tokens_len + 1) |i| {
        if (i > 0) try body.append(allocator, ',');
        try body.append(allocator, '1');
    }
    try body.appendSlice(allocator, "]}");
    const result = try handleDetokenizeRequest(allocator, strategy, body.items);
    defer allocator.free(result.body);
    try std.testing.expectEqual(@as(u16, 400), result.status);
    try std.testing.expect(std.mem.indexOf(u8, result.body, "too_many_tokens") != null);
}

test "detokenize round-trip with the real Qwen tokenizer (golden parity)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const tok_path = "/Users/zouyee/models/Qwen3.8-27B-4bit/tokenizer.json";

    std.Io.Dir.cwd().access(io, tok_path, .{}) catch return error.SkipZigTest;

    var backend = root.tokenizer.BpeTokenizer.init(allocator);
    defer backend.deinit();
    try backend.loadFromFile(io, tok_path);
    const strategy = backend.asStrategy();

    // "Hello world" -> [9419,1814] (tests/golden/qwen38_tokenizer_parity.json);
    // full-sequence decode must round-trip exactly, including via encode.
    const result = try handleDetokenizeRequest(allocator, strategy, "{\"tokens\":[9419,1814]}");
    defer allocator.free(result.body);
    try std.testing.expectEqual(@as(u16, 200), result.status);
    try std.testing.expectEqualStrings("{\"text\":\"Hello world\"}", result.body);
}
