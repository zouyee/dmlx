/// POST /v1/completions — legacy OpenAI text completions.
///
/// Raw-text completion: the prompt is tokenized AS IS (no chat template, no
/// role wrapping) and fed straight into the same engine queue the chat path
/// uses. Vision-token rejection still applies — the request carries
/// `vision = null`, so a raw prompt that tokenizes into the vision id range
/// fails in the model's text prefill entry (qwen38.rejectVisionTokens), same
/// as any non-vision path.
///
/// `prompt` accepts a string or an array of strings. Batch prompts are
/// processed SERIALLY (the engine serves one generation at a time through
/// the request queue anyway); each array item produces one choice with its
/// own index, and usage sums over the whole batch.
const std = @import("std");
const engine = @import("../engine/root.zig");
const config_mod = @import("config.zig");
const state_mod = @import("state.zig");
const utils_mod = @import("utils.zig");
const http_mod = @import("http.zig");
const sse_mod = @import("sse.zig");
const metrics_mod = @import("metrics.zig");
const openai_mod = @import("openai.zig");
const responses_mod = @import("responses.zig");

const ServerConfig = config_mod.ServerConfig;
const ServerState = state_mod.ServerState;
const jsonEscapeInto = utils_mod.jsonEscapeInto;

pub const ParseOutcome = union(enum) {
    ok: ParsedRequest,
    /// Complete JSON error body to return with HTTP 400.
    err: []u8,

    pub fn deinit(self: ParseOutcome, allocator: std.mem.Allocator) void {
        switch (self) {
            .ok => |o| {
                var mut = o;
                mut.deinit(allocator);
            },
            .err => |b| allocator.free(b),
        }
    }
};

pub const ParsedRequest = struct {
    model: []u8, // owned
    /// One entry per prompt (batch). List and entries owned.
    prompts: [][]u8,
    max_tokens: ?u32,
    temperature: ?f32,
    top_p: ?f32,
    top_k: ?u32,
    seed: ?u64,
    stop: ?[][]u8, // owned list + entries
    stream: bool,

    pub fn deinit(self: *ParsedRequest, allocator: std.mem.Allocator) void {
        allocator.free(self.model);
        for (self.prompts) |p| allocator.free(p);
        allocator.free(self.prompts);
        if (self.stop) |stops| {
            for (stops) |s| allocator.free(s);
            allocator.free(stops);
        }
    }
};

fn invalidRequest(allocator: std.mem.Allocator, code: []const u8, message: []const u8) !ParseOutcome {
    return .{ .err = try openai_mod.invalidRequestJson(allocator, code, message) };
}

/// Parse + validate a /v1/completions body. Messages-style fields
/// ("messages") are rejected loudly; unknown fields are ignored (consistent
/// with /v1/tokenize). All client errors come back as `.err` (400 body).
pub fn parseRequest(allocator: std.mem.Allocator, body: []const u8) !ParseOutcome {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        return invalidRequest(allocator, "invalid_json", "request body is not valid JSON");
    };
    defer parsed.deinit();
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return invalidRequest(allocator, "invalid_json", "request body must be a JSON object"),
    };

    if (obj.get("messages") != null) {
        return invalidRequest(allocator, "unsupported_field", "'messages' is not supported by /v1/completions (use /v1/chat/completions); 'prompt' is raw text here");
    }

    const model = switch (obj.get("model") orelse
        return invalidRequest(allocator, "missing_model", "'model' is required")) {
        .string => |s| s,
        else => return invalidRequest(allocator, "invalid_model", "'model' must be a string"),
    };

    const prompt_val = obj.get("prompt") orelse
        return invalidRequest(allocator, "missing_prompt", "'prompt' is required (string or array of strings)");
    var prompts = std.ArrayList([]u8).empty;
    defer {
        for (prompts.items) |p| allocator.free(p);
        prompts.deinit(allocator);
    }
    switch (prompt_val) {
        .string => |s| try prompts.append(allocator, try allocator.dupe(u8, s)),
        .array => |items| {
            if (items.items.len == 0)
                return invalidRequest(allocator, "invalid_prompt", "'prompt' array must not be empty");
            for (items.items) |item| {
                switch (item) {
                    .string => |s| try prompts.append(allocator, try allocator.dupe(u8, s)),
                    else => return invalidRequest(allocator, "invalid_prompt", "'prompt' array items must be strings"),
                }
            }
        },
        else => return invalidRequest(allocator, "invalid_prompt", "'prompt' must be a string or an array of strings"),
    }

    var max_tokens: ?u32 = null;
    if (obj.get("max_tokens")) |v| {
        switch (v) {
            .integer => |n| {
                if (n <= 0 or n > std.math.maxInt(u32))
                    return invalidRequest(allocator, "invalid_max_tokens", "'max_tokens' must be a positive integer");
                max_tokens = @intCast(n);
            },
            .null => {},
            else => return invalidRequest(allocator, "invalid_max_tokens", "'max_tokens' must be a positive integer"),
        }
    }

    var temperature: ?f32 = null;
    if (obj.get("temperature")) |v| {
        switch (v) {
            .float => |f| temperature = @floatCast(f),
            .integer => |n| temperature = @floatFromInt(n),
            .null => {},
            else => return invalidRequest(allocator, "invalid_temperature", "'temperature' must be a number"),
        }
    }

    var top_p: ?f32 = null;
    if (obj.get("top_p")) |v| {
        switch (v) {
            .float => |f| top_p = @floatCast(f),
            .integer => |n| top_p = @floatFromInt(n),
            .null => {},
            else => return invalidRequest(allocator, "invalid_top_p", "'top_p' must be a number"),
        }
        if (top_p) |p| {
            if (!(p > 0.0 and p <= 1.0))
                return invalidRequest(allocator, "InvalidTopP", "top_p must be in the range (0, 1]");
        }
    }

    var top_k: ?u32 = null;
    if (obj.get("top_k")) |v| {
        switch (v) {
            .integer => |n| {
                if (n < 1)
                    return invalidRequest(allocator, "InvalidTopK", "top_k must be >= 1");
                top_k = @intCast(n);
            },
            .null => {},
            else => return invalidRequest(allocator, "InvalidTopK", "top_k must be >= 1"),
        }
    }

    var seed: ?u64 = null;
    if (obj.get("seed")) |v| {
        switch (v) {
            .integer => |n| {
                if (n < 0) return invalidRequest(allocator, "invalid_seed", "'seed' must be a non-negative integer");
                seed = @intCast(n);
            },
            .null => {},
            else => return invalidRequest(allocator, "invalid_seed", "'seed' must be a non-negative integer"),
        }
    }

    var stream = false;
    if (obj.get("stream")) |v| {
        switch (v) {
            .bool => |b| stream = b,
            .null => {},
            else => return invalidRequest(allocator, "invalid_stream", "'stream' must be a boolean"),
        }
    }

    var stop: ?[][]u8 = null;
    errdefer if (stop) |stops| {
        for (stops) |s| allocator.free(s);
        allocator.free(stops);
    };
    if (obj.get("stop")) |v| {
        var stops = std.ArrayList([]u8).empty;
        defer {
            for (stops.items) |s| allocator.free(s);
            stops.deinit(allocator);
        }
        switch (v) {
            .string => |s| try stops.append(allocator, try allocator.dupe(u8, s)),
            .array => |items| {
                for (items.items) |item| {
                    switch (item) {
                        .string => |s| try stops.append(allocator, try allocator.dupe(u8, s)),
                        else => return invalidRequest(allocator, "invalid_stop", "'stop' array items must be strings"),
                    }
                }
            },
            .null => {},
            else => return invalidRequest(allocator, "invalid_stop", "'stop' must be a string or an array of strings"),
        }
        if (stops.items.len > 0) {
            const owned = try allocator.alloc([]u8, stops.items.len);
            var n_owned: usize = 0;
            errdefer {
                for (owned[0..n_owned]) |s| allocator.free(s);
                allocator.free(owned);
            }
            for (stops.items, 0..) |s, i| {
                owned[i] = try allocator.dupe(u8, s);
                n_owned += 1;
            }
            stop = owned;
        }
    }

    const model_owned = try allocator.dupe(u8, model);
    errdefer allocator.free(model_owned);
    const prompts_owned = try allocator.alloc([]u8, prompts.items.len);
    var n_prompts_owned: usize = 0;
    errdefer {
        for (prompts_owned[0..n_prompts_owned]) |p| allocator.free(p);
        allocator.free(prompts_owned);
    }
    for (prompts.items, 0..) |p, i| {
        prompts_owned[i] = try allocator.dupe(u8, p);
        n_prompts_owned += 1;
    }

    return .{ .ok = .{
        .model = model_owned,
        .prompts = prompts_owned,
        .max_tokens = max_tokens,
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .seed = seed,
        .stop = stop,
        .stream = stream,
    } };
}

/// One generation round for one raw prompt: encode (no chat template),
/// submit to the engine queue, drain the completion signal. Mirrors the
/// wait loop in openai.generateChatCompletion minus template/tools.
const GenResult = struct {
    text: []u8, // owned
    finish_reason: []const u8, // "stop" | "length"
    completion_tokens: usize,
    prompt_tokens: usize,
    /// Owned engine error message when generation failed.
    err: ?[]u8,

    fn deinit(self: *GenResult, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
        if (self.err) |e| allocator.free(e);
    }
};

/// `on_token` (streaming path) is called with each non-empty token text as
/// it arrives; null = accumulate only.
fn runPrompt(
    allocator: std.mem.Allocator,
    io: std.Io,
    state: *ServerState,
    server_config: ServerConfig,
    req: ParsedRequest,
    prompt_text: []const u8,
    streaming: bool,
    on_token: ?*const fn (ctx: *anyopaque, token_text: []const u8) anyerror!void,
    on_token_ctx: *anyopaque,
) !GenResult {
    // Raw tokenization — no chat template, no special tokens.
    const prompt_tokens = try state.tokenizer_strategy.encode(prompt_text, false, allocator);
    defer allocator.free(prompt_tokens);

    const max_tokens = if (req.max_tokens) |mt| @min(mt, @as(u32, @intCast(server_config.max_tokens))) else server_config.max_tokens;
    const temperature = req.temperature orelse server_config.temperature;
    const seed = req.seed orelse @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toMilliseconds()));

    const request_config = engine.RequestConfig{
        .prompt_tokens = prompt_tokens,
        .max_tokens = max_tokens,
        .temperature = temperature,
        .top_k = if (req.top_k) |k| @as(usize, k) else server_config.top_k,
        .top_p = req.top_p orelse server_config.top_p,
        .seed = seed,
        .stop_strings = req.stop,
        .stop_tokens = state.stop_tokens,
        .streaming = streaming,
        .model_name = req.model,
        // vision = null: raw text prompts never splice tower features, and
        // the text prefill rejects vision-range token ids outright.
    };

    const request_id = state.next_request_id.fetchAdd(1, .monotonic);
    const req_state = try engine.RequestState.init(allocator, request_id, request_config);
    // Pushed to the engine queue unconditionally below (no fallible call in
    // between): wait for the engine's post-generation bookkeeping before
    // freeing, not just for the final token event.
    defer {
        req_state.waitEngineDone(io);
        req_state.deinit();
    }

    _ = state.active_requests.fetchAdd(1, .monotonic);
    defer _ = state.active_requests.fetchSub(1, .monotonic);

    var node = engine.QueueNode.init(req_state);
    state.request_queue.push(&node);

    var all_text = std.ArrayList(u8).empty;
    defer all_text.deinit(allocator);
    var finish_reason: []const u8 = "stop";
    while (true) {
        const event = req_state.completion.waitForToken(io) catch break;
        if (event) |e| {
            defer if (e.token_text.len > 0) allocator.free(e.token_text);
            if (e.token_text.len > 0) {
                try all_text.appendSlice(allocator, e.token_text);
                if (on_token) |cb| try cb(on_token_ctx, e.token_text);
            }
            if (e.is_final) {
                if (e.finish_reason) |fr| {
                    finish_reason = switch (fr) {
                        .stop => "stop",
                        .length => "length",
                        .error_ => "stop",
                    };
                }
                break;
            }
        } else break;
    }

    // Engine may still be finishing after the final event; wait so the
    // token_count/error reads below and the deferred deinit are race-free.
    req_state.waitEngineDone(io);

    metrics_mod.global.addTokenCounts(@intCast(prompt_tokens.len), @intCast(req_state.token_count));

    if (req_state.completion.hasError(io)) {
        const err_msg = req_state.completion.getError(io) orelse "Generation failed";
        return .{
            .text = try allocator.dupe(u8, ""),
            .finish_reason = finish_reason,
            .completion_tokens = req_state.token_count,
            .prompt_tokens = prompt_tokens.len,
            .err = try allocator.dupe(u8, err_msg),
        };
    }

    return .{
        .text = try all_text.toOwnedSlice(allocator),
        .finish_reason = finish_reason,
        .completion_tokens = req_state.token_count,
        .prompt_tokens = prompt_tokens.len,
        .err = null,
    };
}

fn appendChoiceJson(out: *std.ArrayList(u8), allocator: std.mem.Allocator, index: usize, text: []const u8, finish_reason: []const u8) !void {
    var escaped = std.ArrayList(u8).empty;
    defer escaped.deinit(allocator);
    try jsonEscapeInto(&escaped, allocator, text);
    try out.print(allocator,
        \\{{"text":"{s}","index":{d},"finish_reason":"{s}","logprobs":null}}
    , .{ escaped.items, index, finish_reason });
}

fn serverErrorBody(allocator: std.mem.Allocator, message: []const u8) ![]u8 {
    var escaped = std.ArrayList(u8).empty;
    defer escaped.deinit(allocator);
    try jsonEscapeInto(&escaped, allocator, message);
    return std.fmt.allocPrint(allocator,
        \\{{"error":{{"message":"{s}","type":"server_error","code":"generation_failed"}}}}
    , .{escaped.items});
}

pub fn handleCompletions(
    allocator: std.mem.Allocator,
    io: std.Io,
    state: *ServerState,
    connection: std.Io.net.Stream,
    server_config: ServerConfig,
    body: []const u8,
    status_out: *u16,
) !void {
    const outcome = try parseRequest(allocator, body);
    defer outcome.deinit(allocator);
    const req = switch (outcome) {
        .err => |err_body| {
            status_out.* = 400;
            try http_mod.writeJsonResponse(connection, io, 400, err_body);
            return;
        },
        .ok => |o| o,
    };

    const completion_id = try responses_mod.randomHexId(allocator, io, "cmpl-");
    defer allocator.free(completion_id);
    const created = @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toSeconds()));

    if (req.stream) {
        try handleCompletionsStream(allocator, io, state, connection, server_config, req, completion_id, created, status_out);
        return;
    }

    // Batch prompts run serially (see file header); usage sums over the batch.
    var choices = std.ArrayList(u8).empty;
    defer choices.deinit(allocator);
    try choices.append(allocator, '[');
    var total_prompt: usize = 0;
    var total_completion: usize = 0;
    for (req.prompts, 0..) |prompt_text, i| {
        var result = try runPrompt(allocator, io, state, server_config, req, prompt_text, false, null, undefined);
        defer result.deinit(allocator);
        if (result.err) |err_msg| {
            const err_body = try serverErrorBody(allocator, err_msg);
            defer allocator.free(err_body);
            status_out.* = 500;
            try http_mod.writeJsonResponse(connection, io, 500, err_body);
            return;
        }
        total_prompt += result.prompt_tokens;
        total_completion += result.completion_tokens;
        if (i > 0) try choices.append(allocator, ',');
        try appendChoiceJson(&choices, allocator, i, result.text, result.finish_reason);
    }
    try choices.append(allocator, ']');

    var escaped_model = std.ArrayList(u8).empty;
    defer escaped_model.deinit(allocator);
    try jsonEscapeInto(&escaped_model, allocator, req.model);

    const response = try std.fmt.allocPrint(allocator,
        \\{{"id":"{s}","object":"text_completion","created":{d},"model":"{s}","choices":{s},"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
    , .{ completion_id, created, escaped_model.items, choices.items, total_prompt, total_completion, total_prompt + total_completion });
    defer allocator.free(response);
    status_out.* = 200;
    try http_mod.writeJsonResponseAlloc(connection, 200, response);
}

// ------------------------------------------------------------------
// Streaming: one SSE chunk per token with the text_completion shape,
// then [DONE]. Batch prompts stream sequentially, index per prompt.
// ------------------------------------------------------------------

const StreamCtx = struct {
    sse: *sse_mod.SSEWriter,
    allocator: std.mem.Allocator,
    completion_id: []const u8,
    created: u64,
    model_escaped: []const u8,
    index: usize,
    failed: bool = false,
};

fn streamTokenCb(ctx_ptr: *anyopaque, token_text: []const u8) anyerror!void {
    const ctx: *StreamCtx = @ptrCast(@alignCast(ctx_ptr));
    const chunk = formatStreamChunk(ctx.allocator, ctx.completion_id, ctx.created, ctx.model_escaped, ctx.index, token_text, null) catch {
        ctx.failed = true;
        return error.StreamChunkFailed;
    };
    defer ctx.allocator.free(chunk);
    ctx.sse.sendEvent(chunk) catch {
        ctx.failed = true;
        return error.StreamSendFailed;
    };
}

/// Format one streaming chunk. Null token_text + finish_reason = final chunk.
fn formatStreamChunk(
    allocator: std.mem.Allocator,
    completion_id: []const u8,
    created: u64,
    model_escaped: []const u8,
    index: usize,
    token_text: ?[]const u8,
    finish_reason: ?[]const u8,
) ![]u8 {
    var escaped = std.ArrayList(u8).empty;
    defer escaped.deinit(allocator);
    if (token_text) |t| try jsonEscapeInto(&escaped, allocator, t);
    const finish_part = if (finish_reason) |fr|
        try std.fmt.allocPrint(allocator, "\"{s}\"", .{fr})
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(finish_part);
    return std.fmt.allocPrint(allocator,
        \\{{"id":"{s}","object":"text_completion","created":{d},"model":"{s}","choices":[{{"text":"{s}","index":{d},"finish_reason":{s},"logprobs":null}}]}}
    , .{ completion_id, created, model_escaped, escaped.items, index, finish_part });
}

fn handleCompletionsStream(
    allocator: std.mem.Allocator,
    io: std.Io,
    state: *ServerState,
    connection: std.Io.net.Stream,
    server_config: ServerConfig,
    req: ParsedRequest,
    completion_id: []const u8,
    created: u64,
    status_out: *u16,
) !void {
    try sse_mod.writeSSEHeaders(connection, io);
    status_out.* = 200; // SSE headers committed; failures are in-stream events
    var sse = sse_mod.SSEWriter{ .connection = connection, .io = io };

    var escaped_model = std.ArrayList(u8).empty;
    defer escaped_model.deinit(allocator);
    try jsonEscapeInto(&escaped_model, allocator, req.model);

    for (req.prompts, 0..) |prompt_text, i| {
        var ctx = StreamCtx{
            .sse = &sse,
            .allocator = allocator,
            .completion_id = completion_id,
            .created = created,
            .model_escaped = escaped_model.items,
            .index = i,
        };
        var result = runPrompt(allocator, io, state, server_config, req, prompt_text, true, &streamTokenCb, &ctx) catch |err| {
            if (ctx.failed) return err; // connection is gone; give up
            return err;
        };
        defer result.deinit(allocator);

        if (result.err) |err_msg| {
            var escaped_err = std.ArrayList(u8).empty;
            defer escaped_err.deinit(allocator);
            try jsonEscapeInto(&escaped_err, allocator, err_msg);
            const err_chunk = try std.fmt.allocPrint(allocator,
                \\{{"error":{{"message":"{s}","type":"server_error","code":"generation_failed"}}}}
            , .{escaped_err.items});
            defer allocator.free(err_chunk);
            try sse.sendEvent(err_chunk);
            try sse.sendEvent("[DONE]");
            return;
        }

        const final_chunk = try formatStreamChunk(allocator, completion_id, created, escaped_model.items, i, "", result.finish_reason);
        defer allocator.free(final_chunk);
        try sse.sendEvent(final_chunk);
    }
    try sse.sendEvent("[DONE]");
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

fn expectParseOk(allocator: std.mem.Allocator, body: []const u8) !ParsedRequest {
    const outcome = try parseRequest(allocator, body);
    switch (outcome) {
        .ok => |o| return o,
        .err => |b| {
            defer allocator.free(b);
            std.debug.print("unexpected parse error: {s}\n", .{b});
            return error.TestUnexpectedResult;
        },
    }
}

fn expectParseErr(allocator: std.mem.Allocator, body: []const u8, needle: []const u8) !void {
    const outcome = try parseRequest(allocator, body);
    defer outcome.deinit(allocator);
    const err_body = switch (outcome) {
        .err => |b| b,
        .ok => return error.TestUnexpectedResult,
    };
    try std.testing.expect(std.mem.indexOf(u8, err_body, needle) != null);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, err_body, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(
        "invalid_request_error",
        parsed.value.object.get("error").?.object.get("type").?.string,
    );
}

test "parseRequest: single string prompt with defaults" {
    const allocator = std.testing.allocator;
    var req = try expectParseOk(allocator, "{\"model\":\"m\",\"prompt\":\"hello\"}");
    defer req.deinit(allocator);
    try std.testing.expectEqualStrings("m", req.model);
    try std.testing.expectEqual(@as(usize, 1), req.prompts.len);
    try std.testing.expectEqualStrings("hello", req.prompts[0]);
    try std.testing.expect(req.max_tokens == null);
    try std.testing.expect(!req.stream);
    try std.testing.expect(req.stop == null);
}

test "parseRequest: batch prompt array" {
    const allocator = std.testing.allocator;
    var req = try expectParseOk(allocator,
        \\{"model":"m","prompt":["a","b","c"],"max_tokens":8,"temperature":0,"top_p":1,"top_k":1,"seed":42,"stop":["\n","END"],"stream":true}
    );
    defer req.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 3), req.prompts.len);
    try std.testing.expectEqualStrings("c", req.prompts[2]);
    try std.testing.expectEqual(@as(u32, 8), req.max_tokens.?);
    try std.testing.expectEqual(@as(u64, 42), req.seed.?);
    try std.testing.expectEqual(@as(usize, 2), req.stop.?.len);
    try std.testing.expect(req.stream);
}

test "parseRequest: stop as bare string" {
    const allocator = std.testing.allocator;
    var req = try expectParseOk(allocator, "{\"model\":\"m\",\"prompt\":\"x\",\"stop\":\"\\n\"}");
    defer req.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), req.stop.?.len);
    try std.testing.expectEqualStrings("\n", req.stop.?[0]);
}

test "parseRequest rejects messages-style fields and bad values" {
    const allocator = std.testing.allocator;
    try expectParseErr(allocator,
        \\{"model":"m","messages":[{"role":"user","content":"hi"}]}
    , "messages");
    try expectParseErr(allocator, "{\"prompt\":\"x\"}", "model");
    try expectParseErr(allocator, "{\"model\":\"m\"}", "prompt");
    try expectParseErr(allocator, "{\"model\":\"m\",\"prompt\":[]}", "prompt");
    try expectParseErr(allocator, "{\"model\":\"m\",\"prompt\":[1]}", "strings");
    try expectParseErr(allocator, "{\"model\":\"m\",\"prompt\":\"x\",\"top_p\":0}", "top_p");
    try expectParseErr(allocator, "{\"model\":\"m\",\"prompt\":\"x\",\"top_k\":0}", "top_k");
    try expectParseErr(allocator, "{\"model\":\"m\",\"prompt\":\"x\",\"max_tokens\":-1}", "max_tokens");
    try expectParseErr(allocator, "{\"model\":\"m\",\"prompt\":\"x\",\"stream\":\"yes\"}", "stream");
    try expectParseErr(allocator, "not json", "valid JSON");
}

test "formatStreamChunk shape" {
    const allocator = std.testing.allocator;
    const chunk = try formatStreamChunk(allocator, "cmpl-ab12", 1794458459, "Qwen3", 2, "he\"llo\n", null);
    defer allocator.free(chunk);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, chunk, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expectEqualStrings("cmpl-ab12", obj.get("id").?.string);
    try std.testing.expectEqualStrings("text_completion", obj.get("object").?.string);
    const choice = obj.get("choices").?.array.items[0].object;
    try std.testing.expectEqualStrings("he\"llo\n", choice.get("text").?.string);
    try std.testing.expectEqual(@as(i64, 2), choice.get("index").?.integer);
    try std.testing.expect(std.meta.activeTag(choice.get("finish_reason").?) == .null);
    try std.testing.expect(std.meta.activeTag(choice.get("logprobs").?) == .null);

    const final = try formatStreamChunk(allocator, "cmpl-ab12", 1794458459, "Qwen3", 0, "", "length");
    defer allocator.free(final);
    const fp = try std.json.parseFromSlice(std.json.Value, allocator, final, .{});
    defer fp.deinit();
    try std.testing.expectEqualStrings(
        "length",
        fp.value.object.get("choices").?.array.items[0].object.get("finish_reason").?.string,
    );
}
