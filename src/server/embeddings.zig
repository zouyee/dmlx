/// POST /v1/embeddings — OpenAI-compatible text embeddings.
///
/// Extraction/pooling contract: the model adapter's `embed` op tokenizes
/// raw text (NO chat template, mirroring /v1/tokenize), runs a fresh
/// prefill that resets all recurrent/cache state, and pools the LAST
/// position's POST-final-norm last-layer hidden state, L2-normalized.
/// Last-token pooling is the standard choice for decoder-only LMs. Models
/// without hidden-state access (DeepSeek V4 native path, adapters with no
/// `embed` op) get a 501 embedding_not_supported error.
///
/// Request:  {"model": "...", "input": "..." | ["...", ...],
///            "encoding_format": "float" (default) | "base64"}
/// Response: {"object":"list","data":[{"object":"embedding","index":0,
///            "embedding":[...] or "<base64 f32 LE>"}],
///            "model":"<id>","usage":{"prompt_tokens":N,"total_tokens":N}}
/// Errors:   OpenAI-style {"error":{...}} bodies.
const std = @import("std");
const root = @import("../root.zig");
const utils_mod = @import("utils.zig");

const TokenizerStrategy = root.tokenizer.TokenizerStrategy;

/// Cheap DoS guards: at most 32 inputs per request, 1MB of text total.
pub const max_batch_items: usize = 32;
pub const max_text_bytes: usize = 1024 * 1024;

const EmbeddingRequest = struct {
    model: ?[]const u8 = null, // ignored: single-model server
    input: ?std.json.Value = null,
    encoding_format: ?[]const u8 = null,
};

pub const EmbedResult = struct {
    status: u16,
    body: []u8,
};

fn errorJson(allocator: std.mem.Allocator, err_type: []const u8, code: []const u8, message: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\{{"error":{{"message":"{s}","type":"{s}","code":"{s}"}}}}
    , .{ message, err_type, code });
}

fn invalidRequest(allocator: std.mem.Allocator, code: []const u8, message: []const u8) !EmbedResult {
    return .{
        .status = 400,
        .body = try errorJson(allocator, "invalid_request_error", code, message),
    };
}

/// Runs one embedding forward. In production this routes through the
/// engine-loop thread (GPU forwards are thread-affine — see
/// engine/embed_queue.zig); tests inject a direct runner.
pub const EmbedRunner = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, ids: []const u32) anyerror![]f32,
};

/// Production runner: push the task onto the engine's embed queue and block
/// the HTTP fiber until the engine loop completes it. `ctx` is *ServerState.
pub fn engineEmbedRunner(ctx_ptr: *anyopaque, allocator: std.mem.Allocator, ids: []const u32) anyerror![]f32 {
    const state_mod = @import("state.zig");
    const engine = @import("../engine/root.zig");
    const state: *state_mod.ServerState = @ptrCast(@alignCast(ctx_ptr));
    var task = engine.EmbedTask{ .ids = ids, .allocator = allocator };
    state.embed_queue.push(&task);
    while (!task.done.load(.acquire)) {
        state.io.sleep(.fromMilliseconds(1), .awake) catch {};
    }
    if (task.err) |err| return err;
    return task.result.?;
}

/// Parse + validate + run a embeddings request. `runner` is null when the
/// loaded model has no hidden-state access (→ 501). Caller frees result.body.
pub fn handleEmbeddingsRequest(
    allocator: std.mem.Allocator,
    strategy: TokenizerStrategy,
    runner: ?EmbedRunner,
    model_id: []const u8,
    body: []const u8,
) !EmbedResult {
    const parsed = std.json.parseFromSlice(EmbeddingRequest, allocator, body, .{
        .ignore_unknown_fields = true,
    }) catch {
        return invalidRequest(allocator, "invalid_json", "request body must be a JSON object");
    };
    defer parsed.deinit();

    const encoding_format = parsed.value.encoding_format orelse "float";
    if (!std.mem.eql(u8, encoding_format, "float") and !std.mem.eql(u8, encoding_format, "base64")) {
        return invalidRequest(allocator, "unsupported_encoding_format", "encoding_format must be float or base64");
    }

    const input = parsed.value.input orelse {
        return invalidRequest(allocator, "missing_input", "input is required (string or array of strings)");
    };

    // Collect the input strings (single string or batch array).
    var texts = std.ArrayList([]const u8).empty;
    defer texts.deinit(allocator);
    var total_bytes: usize = 0;
    switch (input) {
        .string => |s| try texts.append(allocator, s),
        .array => |arr| {
            if (arr.items.len == 0) {
                return invalidRequest(allocator, "empty_input", "input array must not be empty");
            }
            if (arr.items.len > max_batch_items) {
                return invalidRequest(allocator, "batch_too_large", "input array exceeds the 32-item limit");
            }
            for (arr.items) |item| {
                switch (item) {
                    .string => |s| try texts.append(allocator, s),
                    else => return invalidRequest(allocator, "invalid_input", "input array items must be strings"),
                }
            }
        },
        else => return invalidRequest(allocator, "invalid_input", "input must be a string or an array of strings"),
    }
    for (texts.items) |s| total_bytes += s.len;
    if (total_bytes > max_text_bytes) {
        return invalidRequest(allocator, "text_too_large", "input text exceeds the 1MB limit");
    }

    const embed = runner orelse {
        return .{
            .status = 501,
            .body = try errorJson(allocator, "server_error", "embedding_not_supported", "the loaded model does not expose hidden states for embeddings"),
        };
    };

    // Tokenize (raw BPE, no chat template) and embed each input.
    var embeddings = std.ArrayList([]f32).empty;
    defer {
        for (embeddings.items) |e| allocator.free(e);
        embeddings.deinit(allocator);
    }
    var prompt_tokens: usize = 0;
    for (texts.items) |text| {
        const ids = try strategy.encode(text, false, allocator);
        defer allocator.free(ids);
        if (ids.len == 0) {
            return invalidRequest(allocator, "empty_input", "input must produce at least one token");
        }
        prompt_tokens += ids.len;
        const vec = embed.run(embed.ctx, allocator, ids) catch {
            return .{
                .status = 500,
                .body = try errorJson(allocator, "server_error", "embedding_failed", "embedding forward pass failed"),
            };
        };
        try embeddings.append(allocator, vec);
    }

    // Response body. ~60KB of JSON per 5120-dim float item — the caller
    // must write it via writeJsonResponseAlloc (the 8KB stack-buffer variant
    // would overflow).
    var out = std.ArrayList(u8).empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"object\":\"list\",\"data\":[");
    for (embeddings.items, 0..) |vec, i| {
        if (i != 0) try out.append(allocator, ',');
        const head = try std.fmt.allocPrint(allocator, "{{\"object\":\"embedding\",\"index\":{d},\"embedding\":", .{i});
        defer allocator.free(head);
        try out.appendSlice(allocator, head);
        if (std.mem.eql(u8, encoding_format, "base64")) {
            // Little-endian f32 bytes, base64-encoded (host is LE arm64;
            // OpenAI's base64 format is LE f32 too).
            const bytes = std.mem.sliceAsBytes(vec);
            const enc = std.base64.standard.Encoder;
            const b64 = try allocator.alloc(u8, enc.calcSize(bytes.len));
            defer allocator.free(b64);
            _ = enc.encode(b64, bytes);
            try out.append(allocator, '"');
            try out.appendSlice(allocator, b64);
            try out.append(allocator, '"');
        } else {
            try out.append(allocator, '[');
            for (vec, 0..) |v, j| {
                if (j != 0) try out.append(allocator, ',');
                const num = try std.fmt.allocPrint(allocator, "{d}", .{v});
                defer allocator.free(num);
                try out.appendSlice(allocator, num);
            }
            try out.append(allocator, ']');
        }
        try out.append(allocator, '}');
    }
    var escaped_model = std.ArrayList(u8).empty;
    defer escaped_model.deinit(allocator);
    try utils_mod.jsonEscapeInto(&escaped_model, allocator, model_id);
    const tail = try std.fmt.allocPrint(
        allocator,
        "],\"model\":\"{s}\",\"usage\":{{\"prompt_tokens\":{d},\"total_tokens\":{d}}}}}",
        .{ escaped_model.items, prompt_tokens, prompt_tokens },
    );
    defer allocator.free(tail);
    try out.appendSlice(allocator, tail);
    return .{ .status = 200, .body = try out.toOwnedSlice(allocator) };
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
    \\      "<unk>": 9
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

fn testStrategy(allocator: std.mem.Allocator, backend: *root.tokenizer.BpeTokenizer) !TokenizerStrategy {
    backend.* = root.tokenizer.BpeTokenizer.init(allocator);
    try backend.loadFromJson(bpe_test_json);
    return backend.asStrategy();
}

fn mockEmbed(ctx: *anyopaque, allocator: std.mem.Allocator, ids: []const u32) anyerror![]f32 {
    _ = ctx;
    // Fixed vector scaled by token count so batch items differ.
    const scale: f32 = @floatFromInt(ids.len);
    const out = try allocator.dupe(f32, &[_]f32{ 0.1 * scale, -0.2, 0.3 });
    return out;
}

var mock_ctx: u8 = 0;

fn mockRunner() EmbedRunner {
    return .{ .ctx = &mock_ctx, .run = &mockEmbed };
}

test "embeddings: single string input, float format, response shape" {
    const allocator = std.testing.allocator;
    var backend: root.tokenizer.BpeTokenizer = undefined;
    defer backend.deinit();
    const strategy = try testStrategy(allocator, &backend);
    const runner = mockRunner();

    const result = try handleEmbeddingsRequest(allocator, strategy, runner, "test-model", "{\"model\":\"m\",\"input\":\"hello\"}");
    defer allocator.free(result.body);

    try std.testing.expectEqual(@as(u16, 200), result.status);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result.body, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expectEqualStrings("list", obj.get("object").?.string);
    const data = obj.get("data").?.array;
    try std.testing.expectEqual(@as(usize, 1), data.items.len);
    const e0 = data.items[0].object;
    try std.testing.expectEqualStrings("embedding", e0.get("object").?.string);
    try std.testing.expectEqual(@as(i64, 0), e0.get("index").?.integer);
    const emb = e0.get("embedding").?.array;
    try std.testing.expectEqual(@as(usize, 3), emb.items.len);
    // "hello" -> [5, 2, 7, 4] (4 tokens) in the test BPE: vec scaled by 4.
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), emb.items[0].float, 1e-6);
    try std.testing.expectEqualStrings("test-model", obj.get("model").?.string);
    const usage = obj.get("usage").?.object;
    try std.testing.expectEqual(@as(i64, 4), usage.get("prompt_tokens").?.integer);
    try std.testing.expectEqual(@as(i64, 4), usage.get("total_tokens").?.integer);
}

test "embeddings: batch input, indices and usage sum" {
    const allocator = std.testing.allocator;
    var backend: root.tokenizer.BpeTokenizer = undefined;
    defer backend.deinit();
    const strategy = try testStrategy(allocator, &backend);
    const runner = mockRunner();

    // "hello" -> 4 tokens, "he" -> 2 tokens ([1, 2]; the "h e" merge only
    // applies once both symbols survive earlier merges).
    const result = try handleEmbeddingsRequest(allocator, strategy, runner, "m", "{\"input\":[\"hello\",\"he\"]}");
    defer allocator.free(result.body);

    try std.testing.expectEqual(@as(u16, 200), result.status);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result.body, .{});
    defer parsed.deinit();
    const data = parsed.value.object.get("data").?.array;
    try std.testing.expectEqual(@as(usize, 2), data.items.len);
    try std.testing.expectEqual(@as(i64, 0), data.items[0].object.get("index").?.integer);
    try std.testing.expectEqual(@as(i64, 1), data.items[1].object.get("index").?.integer);
    const usage = parsed.value.object.get("usage").?.object;
    try std.testing.expectEqual(@as(i64, 6), usage.get("prompt_tokens").?.integer);
}

test "embeddings: base64 encoding_format returns LE f32 bytes" {
    const allocator = std.testing.allocator;
    var backend: root.tokenizer.BpeTokenizer = undefined;
    defer backend.deinit();
    const strategy = try testStrategy(allocator, &backend);
    const runner = mockRunner();

    const result = try handleEmbeddingsRequest(allocator, strategy, runner, "m", "{\"input\":\"he\",\"encoding_format\":\"base64\"}");
    defer allocator.free(result.body);

    try std.testing.expectEqual(@as(u16, 200), result.status);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result.body, .{});
    defer parsed.deinit();
    const b64 = parsed.value.object.get("data").?.array.items[0].object.get("embedding").?.string;
    const dec = std.base64.standard.Decoder;
    const raw = try allocator.alloc(u8, try dec.calcSizeForSlice(b64));
    defer allocator.free(raw);
    try dec.decode(raw, b64);
    try std.testing.expectEqual(@as(usize, 12), raw.len); // 3 f32
    const floats: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, raw));
    // "he" tokenizes to 2 ids in the test BPE → mock scale is 2.
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), floats[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, -0.2), floats[1], 1e-6);
}

test "embeddings: validation errors" {
    const allocator = std.testing.allocator;
    var backend: root.tokenizer.BpeTokenizer = undefined;
    defer backend.deinit();
    const strategy = try testStrategy(allocator, &backend);
    const runner = mockRunner();

    // missing input
    {
        const r = try handleEmbeddingsRequest(allocator, strategy, runner, "m", "{\"model\":\"m\"}");
        defer allocator.free(r.body);
        try std.testing.expectEqual(@as(u16, 400), r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, "missing_input") != null);
    }
    // non-string input
    {
        const r = try handleEmbeddingsRequest(allocator, strategy, runner, "m", "{\"input\":123}");
        defer allocator.free(r.body);
        try std.testing.expectEqual(@as(u16, 400), r.status);
    }
    // empty batch
    {
        const r = try handleEmbeddingsRequest(allocator, strategy, runner, "m", "{\"input\":[]}");
        defer allocator.free(r.body);
        try std.testing.expectEqual(@as(u16, 400), r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, "empty_input") != null);
    }
    // batch over 32
    {
        var body = std.ArrayList(u8).empty;
        defer body.deinit(allocator);
        try body.appendSlice(allocator, "{\"input\":[");
        for (0..max_batch_items + 1) |i| {
            if (i != 0) try body.append(allocator, ',');
            try body.appendSlice(allocator, "\"x\"");
        }
        try body.appendSlice(allocator, "]}");
        const r = try handleEmbeddingsRequest(allocator, strategy, runner, "m", body.items);
        defer allocator.free(r.body);
        try std.testing.expectEqual(@as(u16, 400), r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, "batch_too_large") != null);
    }
    // over 1MB of text
    {
        var body = std.ArrayList(u8).empty;
        defer body.deinit(allocator);
        try body.appendSlice(allocator, "{\"input\":\"");
        try body.appendNTimes(allocator, 'a', max_text_bytes + 1);
        try body.appendSlice(allocator, "\"}");
        const r = try handleEmbeddingsRequest(allocator, strategy, runner, "m", body.items);
        defer allocator.free(r.body);
        try std.testing.expectEqual(@as(u16, 400), r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, "text_too_large") != null);
    }
    // unsupported encoding_format
    {
        const r = try handleEmbeddingsRequest(allocator, strategy, runner, "m", "{\"input\":\"he\",\"encoding_format\":\"int8\"}");
        defer allocator.free(r.body);
        try std.testing.expectEqual(@as(u16, 400), r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, "unsupported_encoding_format") != null);
    }
    // empty string tokenizes to zero ids
    {
        const r = try handleEmbeddingsRequest(allocator, strategy, runner, "m", "{\"input\":\"\"}");
        defer allocator.free(r.body);
        try std.testing.expectEqual(@as(u16, 400), r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, "empty_input") != null);
    }
    // invalid JSON
    {
        const r = try handleEmbeddingsRequest(allocator, strategy, runner, "m", "not json");
        defer allocator.free(r.body);
        try std.testing.expectEqual(@as(u16, 400), r.status);
    }
}

test "embeddings: model without embed op gets 501" {
    const allocator = std.testing.allocator;
    var backend: root.tokenizer.BpeTokenizer = undefined;
    defer backend.deinit();
    const strategy = try testStrategy(allocator, &backend);
    const r = try handleEmbeddingsRequest(allocator, strategy, null, "m", "{\"input\":\"hello\"}");
    defer allocator.free(r.body);
    try std.testing.expectEqual(@as(u16, 501), r.status);
    try std.testing.expect(std.mem.indexOf(u8, r.body, "embedding_not_supported") != null);
}
