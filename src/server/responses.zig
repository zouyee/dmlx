/// OpenAI Responses API (`POST /v1/responses`) — minimal compatibility layer.
///
/// This is NOT a full Responses API implementation. It is a translation
/// layer onto the existing chat-completions machinery: the Responses request
/// is rewritten into an internal chat-completion request, executed by the
/// same engine flow, and the result is wrapped in a Responses object.
///
/// Supported request surface (everything else is a clean 400):
///   model (ignored — single-model server), input (string or message-item
///   array; content string or input_text/input_image parts), instructions
///   (prepended system message), max_output_tokens, temperature, top_p,
///   stream, store (default true: retain the completed/failed response
///   object in the bounded in-memory store for GET/DELETE /v1/responses/{id};
///   store:false skips retention), metadata (ignored).
///
/// Streaming emits the minimal SSE sequence the OpenAI SDK needs:
///   response.created → response.output_text.delta* → response.completed.
const std = @import("std");
const engine = @import("../engine/root.zig");
const config_mod = @import("config.zig");
const state_mod = @import("state.zig");
const utils_mod = @import("utils.zig");
const http_mod = @import("http.zig");
const sse_mod = @import("sse.zig");
const metrics_mod = @import("metrics.zig");
const openai_mod = @import("openai.zig");

const ServerConfig = config_mod.ServerConfig;
const ServerState = state_mod.ServerState;
const jsonEscapeInto = utils_mod.jsonEscapeInto;

// ------------------------------------------------------------------
// Request translation: Responses request → chat completion body
// ------------------------------------------------------------------

pub const TranslateOutcome = union(enum) {
    /// Translated request, ready to feed into the chat-completions flow.
    ok: struct {
        /// Chat-completions request body (JSON).
        body: []u8,
        stream: bool,
        /// OpenAI default is store=true; store:false = don't retain the
        /// response object in the in-memory store (response_store.zig).
        store: bool,
    },
    /// Complete JSON error body to return with HTTP 400.
    err: []u8,

    pub fn deinit(self: TranslateOutcome, allocator: std.mem.Allocator) void {
        switch (self) {
            .ok => |o| allocator.free(o.body),
            .err => |b| allocator.free(b),
        }
    }
};

/// Fields that are explicitly rejected with a tailored message when present.
const RejectedField = struct { name: []const u8, reason: []const u8 };
const rejected_fields = [_]RejectedField{
    .{ .name = "tools", .reason = "tool use is not supported by /v1/responses on this server" },
    .{ .name = "tool_choice", .reason = "tool use is not supported by /v1/responses on this server" },
    .{ .name = "previous_response_id", .reason = "chained responses are not supported (this server is stateless); send the full conversation in 'input'" },
    .{ .name = "background", .reason = "background mode is not supported" },
    .{ .name = "reasoning", .reason = "reasoning configuration is not supported" },
    .{ .name = "text", .reason = "structured text.format output is not supported (plain text only)" },
    .{ .name = "include", .reason = "the 'include' parameter is not supported" },
    .{ .name = "truncation", .reason = "the 'truncation' parameter is not supported" },
    .{ .name = "parallel_tool_calls", .reason = "tool use is not supported by /v1/responses on this server" },
    .{ .name = "max_tool_calls", .reason = "tool use is not supported by /v1/responses on this server" },
    .{ .name = "prompt", .reason = "prompt templates are not supported" },
    .{ .name = "stream_options", .reason = "stream_options is not supported" },
    .{ .name = "service_tier", .reason = "service_tier is not supported" },
    .{ .name = "modalities", .reason = "only text output is supported" },
    .{ .name = "audio", .reason = "audio output is not supported" },
    .{ .name = "top_logprobs", .reason = "logprobs are not supported" },
};

/// Fields that are accepted but have no effect on this server. `store` is
/// NOT in this list — it gates retention in the in-memory response store.
const ignored_fields = [_][]const u8{"metadata"};

fn invalidRequest(allocator: std.mem.Allocator, message: []const u8) !TranslateOutcome {
    return .{ .err = try openai_mod.invalidRequestJson(allocator, "unsupported_parameter", message) };
}

fn invalidRequestFmt(allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !TranslateOutcome {
    const message = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(message);
    return invalidRequest(allocator, message);
}

/// Translate a Responses API request body into an equivalent
/// chat-completions request body. `default_model` is used when the request
/// omits `model`. All validation failures come back as `.err` (400 body);
/// only OOM/parse failures propagate as errors.
pub fn translateRequest(
    allocator: std.mem.Allocator,
    body: []const u8,
    default_model: []const u8,
) !TranslateOutcome {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        return invalidRequest(allocator, "request body is not valid JSON");
    };
    defer parsed.deinit();
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return invalidRequest(allocator, "request body must be a JSON object"),
    };

    // Field allowlist / rejection pass — anything the translation cannot
    // honor fails loudly here rather than being silently dropped.
    var it = obj.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        if (std.mem.eql(u8, name, "model") or
            std.mem.eql(u8, name, "input") or
            std.mem.eql(u8, name, "instructions") or
            std.mem.eql(u8, name, "max_output_tokens") or
            std.mem.eql(u8, name, "temperature") or
            std.mem.eql(u8, name, "top_p") or
            std.mem.eql(u8, name, "store") or
            std.mem.eql(u8, name, "stream")) continue;
        var is_ignored = false;
        for (ignored_fields) |f| {
            if (std.mem.eql(u8, name, f)) {
                is_ignored = true;
                break;
            }
        }
        if (!is_ignored) {
            for (rejected_fields) |rf| {
                if (std.mem.eql(u8, name, rf.name)) {
                    return invalidRequestFmt(allocator, "unsupported field '{s}': {s}", .{ name, rf.reason });
                }
            }
            return invalidRequestFmt(allocator, "unsupported field '{s}': this server implements a minimal Responses API subset", .{name});
        }
    }

    var model: []const u8 = default_model;
    if (obj.get("model")) |m| {
        switch (m) {
            .string => |s| model = s,
            .null => {},
            else => return invalidRequest(allocator, "'model' must be a string"),
        }
    }

    const input = obj.get("input") orelse
        return invalidRequest(allocator, "missing required field 'input' (string or array of message items)");

    var instructions: ?[]const u8 = null;
    if (obj.get("instructions")) |v| {
        switch (v) {
            .string => |s| instructions = s,
            .null => {},
            else => return invalidRequest(allocator, "'instructions' must be a string"),
        }
    }

    var max_output_tokens: ?u32 = null;
    if (obj.get("max_output_tokens")) |v| {
        switch (v) {
            .integer => |n| {
                if (n <= 0) return invalidRequest(allocator, "'max_output_tokens' must be a positive integer");
                if (n > std.math.maxInt(u32))
                    return invalidRequestFmt(allocator, "'max_output_tokens' must be at most {d}", .{std.math.maxInt(u32)});
                max_output_tokens = @intCast(n);
            },
            .null => {},
            else => return invalidRequest(allocator, "'max_output_tokens' must be a positive integer"),
        }
    }

    var temperature: ?f64 = null;
    if (obj.get("temperature")) |v| {
        switch (v) {
            .float => |f| temperature = f,
            .integer => |n| temperature = @floatFromInt(n),
            .null => {},
            else => return invalidRequest(allocator, "'temperature' must be a number"),
        }
    }

    var top_p: ?f64 = null;
    if (obj.get("top_p")) |v| {
        switch (v) {
            .float => |f| top_p = f,
            .integer => |n| top_p = @floatFromInt(n),
            .null => {},
            else => return invalidRequest(allocator, "'top_p' must be a number"),
        }
        if (top_p) |pp| {
            // Matches openai_mod.validateSampling so the translated request
            // passes the chat path's validation assert.
            if (!(pp > 0.0 and pp <= 1.0))
                return invalidRequest(allocator, "top_p must be in the range (0, 1]");
        }
    }

    var stream = false;
    if (obj.get("stream")) |v| {
        switch (v) {
            .bool => |b| stream = b,
            .null => {},
            else => return invalidRequest(allocator, "'stream' must be a boolean"),
        }
    }

    var store = true;
    if (obj.get("store")) |v| {
        switch (v) {
            .bool => |b| store = b,
            .null => {},
            else => return invalidRequest(allocator, "'store' must be a boolean"),
        }
    }

    // Build the messages array JSON.
    var messages = std.ArrayList(u8).empty;
    defer messages.deinit(allocator);
    try messages.append(allocator, '[');
    var first = true;
    if (instructions) |instr| {
        try messages.appendSlice(allocator, "{\"role\":\"system\",\"content\":\"");
        try jsonEscapeInto(&messages, allocator, instr);
        try messages.appendSlice(allocator, "\"}");
        first = false;
    }
    switch (input) {
        .string => |s| {
            if (!first) try messages.append(allocator, ',');
            try messages.appendSlice(allocator, "{\"role\":\"user\",\"content\":\"");
            try jsonEscapeInto(&messages, allocator, s);
            try messages.appendSlice(allocator, "\"}");
        },
        .array => |items| {
            for (items.items) |item| {
                const r = try translateInputItem(allocator, &messages, item, &first);
                if (r) |outcome| return outcome;
            }
        },
        else => return invalidRequest(allocator, "'input' must be a string or an array of message items"),
    }
    try messages.append(allocator, ']');

    // Assemble the chat-completions body.
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"model\":\"");
    try jsonEscapeInto(&out, allocator, model);
    try out.appendSlice(allocator, "\",\"messages\":");
    try out.appendSlice(allocator, messages.items);
    if (max_output_tokens) |mt| try out.print(allocator, ",\"max_tokens\":{d}", .{mt});
    if (temperature) |t| try out.print(allocator, ",\"temperature\":{d}", .{t});
    if (top_p) |p| try out.print(allocator, ",\"top_p\":{d}", .{p});
    if (stream) try out.appendSlice(allocator, ",\"stream\":true");
    try out.append(allocator, '}');
    return .{ .ok = .{ .body = try out.toOwnedSlice(allocator), .stream = stream, .store = store } };
}

/// Append one Responses input item to the chat `messages` JSON array.
/// Returns an error outcome to propagate, or null on success.
fn translateInputItem(
    allocator: std.mem.Allocator,
    messages: *std.ArrayList(u8),
    item: std.json.Value,
    first: *bool,
) !?TranslateOutcome {
    const obj = switch (item) {
        .object => |o| o,
        else => return try invalidRequest(allocator, "each 'input' item must be a message object with 'role' and 'content'"),
    };
    if (obj.get("type")) |t| {
        const ts = switch (t) {
            .string => |s| s,
            else => return try invalidRequest(allocator, "input item 'type' must be a string"),
        };
        if (!std.mem.eql(u8, ts, "message")) {
            return try invalidRequestFmt(allocator, "unsupported input item type '{s}' (only 'message' items are supported)", .{ts});
        }
    }
    const role = switch (obj.get("role") orelse
        return try invalidRequest(allocator, "input message item is missing 'role'")) {
        .string => |s| s,
        else => return try invalidRequest(allocator, "input message 'role' must be a string"),
    };
    if (!std.mem.eql(u8, role, "user") and !std.mem.eql(u8, role, "assistant") and
        !std.mem.eql(u8, role, "system") and !std.mem.eql(u8, role, "developer"))
    {
        return try invalidRequestFmt(allocator, "unsupported role '{s}' in input message", .{role});
    }
    const content = obj.get("content") orelse
        return try invalidRequest(allocator, "input message item is missing 'content'");

    if (!first.*) try messages.append(allocator, ',');
    first.* = false;
    try messages.appendSlice(allocator, "{\"role\":\"");
    try jsonEscapeInto(messages, allocator, role);
    try messages.appendSlice(allocator, "\",\"content\":");
    switch (content) {
        .string => |s| {
            try messages.append(allocator, '"');
            try jsonEscapeInto(messages, allocator, s);
            try messages.append(allocator, '"');
        },
        .array => |parts| {
            try messages.append(allocator, '[');
            for (parts.items, 0..) |part, i| {
                if (i > 0) try messages.append(allocator, ',');
                if (try translateContentPart(allocator, messages, part)) |outcome| return outcome;
            }
            try messages.append(allocator, ']');
        },
        else => return try invalidRequest(allocator, "input message 'content' must be a string or an array of content parts"),
    }
    try messages.append(allocator, '}');
    return null;
}

/// Append one content part (input_text / output_text / input_image) as a
/// chat-completions content part (text / image_url).
fn translateContentPart(
    allocator: std.mem.Allocator,
    messages: *std.ArrayList(u8),
    part: std.json.Value,
) !?TranslateOutcome {
    const obj = switch (part) {
        .object => |o| o,
        else => return try invalidRequest(allocator, "content parts must be objects"),
    };
    const ty = switch (obj.get("type") orelse
        return try invalidRequest(allocator, "content part is missing 'type'")) {
        .string => |s| s,
        else => return try invalidRequest(allocator, "content part 'type' must be a string"),
    };
    if (std.mem.eql(u8, ty, "input_text") or std.mem.eql(u8, ty, "output_text")) {
        const text = switch (obj.get("text") orelse
            return try invalidRequestFmt(allocator, "'{s}' part is missing 'text'", .{ty})) {
            .string => |s| s,
            else => return try invalidRequestFmt(allocator, "'{s}' part 'text' must be a string", .{ty}),
        };
        try messages.appendSlice(allocator, "{\"type\":\"text\",\"text\":\"");
        try jsonEscapeInto(messages, allocator, text);
        try messages.appendSlice(allocator, "\"}");
        return null;
    }
    if (std.mem.eql(u8, ty, "input_image")) {
        const url = switch (obj.get("image_url") orelse
            return try invalidRequest(allocator, "'input_image' part is missing 'image_url'")) {
            .string => |s| s,
            else => return try invalidRequest(allocator, "'input_image' part 'image_url' must be a string (data:...;base64 URL or absolute file path)"),
        };
        try messages.appendSlice(allocator, "{\"type\":\"image_url\",\"image_url\":{\"url\":\"");
        try jsonEscapeInto(messages, allocator, url);
        try messages.appendSlice(allocator, "\"}}");
        return null;
    }
    return try invalidRequestFmt(allocator, "unsupported content part type '{s}' (supported: input_text, input_image)", .{ty});
}

// ------------------------------------------------------------------
// Response translation: chat completion result → Responses object
// ------------------------------------------------------------------

pub const ResponseStatus = struct {
    status: []const u8, // "completed" or "incomplete"
    incomplete_reason: ?[]const u8, // "max_output_tokens" when incomplete
};

/// Map a chat finish_reason onto the Responses status fields.
pub fn statusForFinishReason(finish_reason: []const u8) ResponseStatus {
    if (std.mem.eql(u8, finish_reason, "length")) {
        return .{ .status = "incomplete", .incomplete_reason = "max_output_tokens" };
    }
    return .{ .status = "completed", .incomplete_reason = null };
}

/// Format a Responses API object from the extracted pieces.
pub fn formatResponseJson(
    allocator: std.mem.Allocator,
    resp_id: []const u8,
    msg_id: []const u8,
    created_at: u64,
    model: []const u8,
    text: []const u8,
    finish_reason: []const u8,
    input_tokens: usize,
    output_tokens: usize,
) ![]u8 {
    const st = statusForFinishReason(finish_reason);
    var escaped_text = std.ArrayList(u8).empty;
    defer escaped_text.deinit(allocator);
    try jsonEscapeInto(&escaped_text, allocator, text);
    var escaped_model = std.ArrayList(u8).empty;
    defer escaped_model.deinit(allocator);
    try jsonEscapeInto(&escaped_model, allocator, model);

    const incomplete = if (st.incomplete_reason) |r|
        try std.fmt.allocPrint(allocator, "{{\"reason\":\"{s}\"}}", .{r})
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(incomplete);

    return std.fmt.allocPrint(allocator,
        \\{{"id":"{s}","object":"response","created_at":{d},"status":"{s}","model":"{s}","output":[{{"type":"message","id":"{s}","status":"completed","role":"assistant","content":[{{"type":"output_text","text":"{s}"}}]}}],"usage":{{"input_tokens":{d},"output_tokens":{d},"total_tokens":{d}}},"error":null,"incomplete_details":{s}}}
    , .{
        resp_id,
        created_at,
        st.status,
        escaped_model.items,
        msg_id,
        escaped_text.items,
        input_tokens,
        output_tokens,
        input_tokens + output_tokens,
        incomplete,
    });
}

/// Wrap a successful chat-completion response body in a Responses object.
/// Caller owns the returned slice.
pub fn buildResponseObject(
    allocator: std.mem.Allocator,
    chat_body: []const u8,
    resp_id: []const u8,
    msg_id: []const u8,
) ![]u8 {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, chat_body, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;

    const created: u64 = @intCast(obj.get("created").?.integer);
    const model = obj.get("model").?.string;
    const choice = obj.get("choices").?.array.items[0].object;
    const finish = switch (choice.get("finish_reason") orelse .null) {
        .string => |s| s,
        else => "stop",
    };
    const message = choice.get("message").?.object;
    const text: []const u8 = switch (message.get("content") orelse .null) {
        .string => |s| s,
        else => "",
    };
    const usage = obj.get("usage").?.object;
    const input_tokens: usize = @intCast(usage.get("prompt_tokens").?.integer);
    const output_tokens: usize = @intCast(usage.get("completion_tokens").?.integer);

    return formatResponseJson(allocator, resp_id, msg_id, created, model, text, finish, input_tokens, output_tokens);
}

// ------------------------------------------------------------------
// HTTP handler
// ------------------------------------------------------------------

pub fn randomHexId(allocator: std.mem.Allocator, io: std.Io, prefix: []const u8) ![]u8 {
    var bytes: [12]u8 = undefined;
    io.random(&bytes);
    const hex = "0123456789abcdef";
    var out = try allocator.alloc(u8, prefix.len + bytes.len * 2);
    @memcpy(out[0..prefix.len], prefix);
    for (bytes, 0..) |b, i| {
        out[prefix.len + i * 2] = hex[b >> 4];
        out[prefix.len + i * 2 + 1] = hex[b & 0xf];
    }
    return out;
}

pub fn handleResponses(
    allocator: std.mem.Allocator,
    io: std.Io,
    state: *ServerState,
    connection: std.Io.net.Stream,
    server_config: ServerConfig,
    body: []const u8,
    status_out: *u16,
) !void {
    const default_model = std.fs.path.basename(server_config.model_path);
    const outcome = try translateRequest(allocator, body, default_model);
    defer outcome.deinit(allocator);
    const translated = switch (outcome) {
        .err => |err_body| {
            status_out.* = 400;
            try http_mod.writeJsonResponse(connection, io, 400, err_body);
            return;
        },
        .ok => |o| o,
    };

    const resp_id = try randomHexId(allocator, io, "resp_");
    defer allocator.free(resp_id);
    const msg_id = try randomHexId(allocator, io, "msg_");
    defer allocator.free(msg_id);

    if (translated.stream) {
        const parsed = try std.json.parseFromSlice(openai_mod.ChatCompletionRequest, allocator, translated.body, .{});
        defer parsed.deinit();
        try handleResponsesStream(allocator, io, state, connection, server_config, parsed.value, resp_id, msg_id, translated.store, status_out);
        return;
    }

    const chat_response = try openai_mod.generateChatCompletion(allocator, io, state, translated.body, server_config);
    defer allocator.free(chat_response);

    // generateChatCompletion reports validation/generation failures as an
    // error body; pass those through, mapping the error type onto the HTTP
    // status (server_error → 500, everything else → 400).
    if (std.mem.indexOf(u8, chat_response, "\"error\"") != null) {
        const status = errorBodyStatus(allocator, chat_response);
        status_out.* = status;
        try http_mod.writeJsonResponse(connection, io, status, chat_response);
        return;
    }

    const response_json = try buildResponseObject(allocator, chat_response, resp_id, msg_id);
    defer allocator.free(response_json);
    if (translated.store) try state.response_store.put(resp_id, response_json);
    status_out.* = 200;
    try http_mod.writeJsonResponse(connection, io, 200, response_json);
}

/// HTTP status for a chat-path error body: 500 for server_error, else 400.
/// Bodies are machine-generated by openai.zig, so on any parse surprise we
/// conservatively fall back to 400.
fn errorBodyStatus(allocator: std.mem.Allocator, body: []const u8) u16 {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return 400;
    defer parsed.deinit();
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return 400,
    };
    const err = switch (obj.get("error") orelse return 400) {
        .object => |o| o,
        else => return 400,
    };
    const ty = switch (err.get("type") orelse return 400) {
        .string => |s| s,
        else => return 400,
    };
    return if (std.mem.eql(u8, ty, "server_error")) 500 else 400;
}

// ------------------------------------------------------------------
// Streaming: response.created → output_text.delta* → output_text.done
//            → response.completed
//
// NOTE: unlike the chat streaming path (streaming.zig), this handler does
// not emit SSE keep-alive comments during prefill; clients see silence
// until response.created. Trivial to add later if a client needs it.
// ------------------------------------------------------------------

fn sendResponsesEvent(sse: *sse_mod.SSEWriter, allocator: std.mem.Allocator, event: []const u8, payload: []const u8) !void {
    const data = try std.fmt.allocPrint(allocator, "event: {s}\ndata: {s}\n\n", .{ event, payload });
    defer allocator.free(data);
    var buf: [4096]u8 = undefined;
    var w = sse.connection.writer(sse.io, &buf);
    try w.interface.writeAll(data);
    try w.interface.flush();
}

fn handleResponsesStream(
    allocator: std.mem.Allocator,
    io: std.Io,
    state: *ServerState,
    connection: std.Io.net.Stream,
    server_config: ServerConfig,
    req: openai_mod.ChatCompletionRequest,
    resp_id: []const u8,
    msg_id: []const u8,
    store: bool,
    status_out: *u16,
) !void {
    const created = @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toSeconds()));

    // Build the prompt BEFORE committing SSE headers so preparation
    // failures still surface as a clean HTTP 400.
    var prepared = openai_mod.preparePrompt(allocator, state, req, null) catch |err| {
        const msg: []const u8 = switch (err) {
            error.VisionNotSupported => "vision input is not supported by the loaded model",
            error.ImageTooLarge => "decoded image exceeds the 20MB limit",
            error.BadImageUrl => "unsupported image_url (data:...;base64 URLs and absolute file paths only)",
            error.InvalidContent => "invalid message content",
            else => "prompt preparation failed",
        };
        const err_body = try openai_mod.invalidRequestJson(allocator, @errorName(err), msg);
        defer allocator.free(err_body);
        status_out.* = 400;
        try http_mod.writeJsonResponse(connection, io, 400, err_body);
        return;
    };
    defer prepared.deinit(allocator);
    const prompt_tokens = prepared.tokens;

    const max_tokens = if (req.max_tokens) |mt| @min(mt, @as(u32, @intCast(server_config.max_tokens))) else server_config.max_tokens;
    const temperature = req.temperature orelse server_config.temperature;
    const seed = req.seed orelse @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toMilliseconds()));

    const request_config = engine.RequestConfig{
        .prompt_tokens = prompt_tokens,
        .max_tokens = max_tokens,
        .temperature = temperature,
        .top_k = server_config.top_k,
        .top_p = req.top_p orelse server_config.top_p,
        .seed = seed,
        .stop_strings = req.stop,
        .stop_tokens = state.stop_tokens,
        .streaming = true,
        .model_name = req.model,
        .vision = prepared.vision,
    };

    const request_id = state.next_request_id.fetchAdd(1, .monotonic);
    const req_state = try engine.RequestState.init(allocator, request_id, request_config);
    defer req_state.deinit();
    prepared.vision = null;

    try sse_mod.writeSSEHeaders(connection, io);
    status_out.* = 200; // SSE headers committed as 200; failures are in-stream events
    var sse = sse_mod.SSEWriter{ .connection = connection, .io = io };

    var escaped_model = std.ArrayList(u8).empty;
    defer escaped_model.deinit(allocator);
    try jsonEscapeInto(&escaped_model, allocator, req.model);

    const created_payload = try std.fmt.allocPrint(allocator,
        \\{{"type":"response.created","response":{{"id":"{s}","object":"response","created_at":{d},"status":"in_progress","model":"{s}","output":[],"error":null,"incomplete_details":null}}}}
    , .{ resp_id, created, escaped_model.items });
    defer allocator.free(created_payload);
    try sendResponsesEvent(&sse, allocator, "response.created", created_payload);

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
                var escaped_delta = std.ArrayList(u8).empty;
                defer escaped_delta.deinit(allocator);
                try jsonEscapeInto(&escaped_delta, allocator, e.token_text);
                const delta_payload = try std.fmt.allocPrint(allocator,
                    \\{{"type":"response.output_text.delta","response_id":"{s}","item_id":"{s}","output_index":0,"content_index":0,"delta":"{s}"}}
                , .{ resp_id, msg_id, escaped_delta.items });
                defer allocator.free(delta_payload);
                try sendResponsesEvent(&sse, allocator, "response.output_text.delta", delta_payload);
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

    metrics_mod.global.addTokenCounts(@intCast(prompt_tokens.len), @intCast(req_state.token_count));

    if (req_state.completion.hasError(io)) {
        const err_msg = req_state.completion.getError(io) orelse "Generation failed";
        var escaped_err = std.ArrayList(u8).empty;
        defer escaped_err.deinit(allocator);
        try jsonEscapeInto(&escaped_err, allocator, err_msg);
        const failed_response = try std.fmt.allocPrint(allocator,
            \\{{"id":"{s}","object":"response","created_at":{d},"status":"failed","model":"{s}","output":[],"error":{{"message":"{s}","type":"server_error"}},"incomplete_details":null}}
        , .{ resp_id, created, escaped_model.items, escaped_err.items });
        defer allocator.free(failed_response);
        // Failed responses are retained too (store=false opts out) — clients
        // polling GET /v1/responses/{id} after a background-less stream can
        // see the terminal state.
        if (store) try state.response_store.put(resp_id, failed_response);
        const failed_payload = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.failed","response":{s}}}
        , .{failed_response});
        defer allocator.free(failed_payload);
        try sendResponsesEvent(&sse, allocator, "response.failed", failed_payload);
        return;
    }

    var escaped_full = std.ArrayList(u8).empty;
    defer escaped_full.deinit(allocator);
    try jsonEscapeInto(&escaped_full, allocator, all_text.items);
    const done_payload = try std.fmt.allocPrint(allocator,
        \\{{"type":"response.output_text.done","response_id":"{s}","item_id":"{s}","output_index":0,"content_index":0,"text":"{s}"}}
    , .{ resp_id, msg_id, escaped_full.items });
    defer allocator.free(done_payload);
    try sendResponsesEvent(&sse, allocator, "response.output_text.done", done_payload);

    const final_response = try formatResponseJson(
        allocator,
        resp_id,
        msg_id,
        created,
        req.model,
        all_text.items,
        finish_reason,
        prompt_tokens.len,
        req_state.token_count,
    );
    defer allocator.free(final_response);
    if (store) try state.response_store.put(resp_id, final_response);
    const completed_payload = try std.fmt.allocPrint(allocator,
        \\{{"type":"response.completed","response":{s}}}
    , .{final_response});
    defer allocator.free(completed_payload);
    try sendResponsesEvent(&sse, allocator, "response.completed", completed_payload);
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

fn expectTranslateErr(allocator: std.mem.Allocator, body: []const u8, needle: []const u8) !void {
    const outcome = try translateRequest(allocator, body, "m");
    defer outcome.deinit(allocator);
    const err_body = switch (outcome) {
        .err => |b| b,
        .ok => return error.TestUnexpectedResult,
    };
    try std.testing.expect(std.mem.indexOf(u8, err_body, needle) != null);
    // Error bodies must be valid JSON with the invalid_request_error shape.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, err_body, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(
        "invalid_request_error",
        parsed.value.object.get("error").?.object.get("type").?.string,
    );
}

fn expectTranslateOk(allocator: std.mem.Allocator, body: []const u8) !std.json.Parsed(std.json.Value) {
    const outcome = try translateRequest(allocator, body, "default-model");
    defer outcome.deinit(allocator);
    const chat_body = switch (outcome) {
        .ok => |o| o.body,
        .err => |b| {
            std.debug.print("unexpected translate error: {s}\n", .{b});
            return error.TestUnexpectedResult;
        },
    };
    return std.json.parseFromSlice(std.json.Value, allocator, chat_body, .{});
}

test "translateRequest: string input" {
    const allocator = std.testing.allocator;
    var parsed = try expectTranslateOk(allocator,
        \\{"model":"gpt-x","input":"Hello there","max_output_tokens":64,"temperature":0.5,"top_p":0.9}
    );
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expectEqualStrings("gpt-x", obj.get("model").?.string);
    const messages = obj.get("messages").?.array;
    try std.testing.expectEqual(@as(usize, 1), messages.items.len);
    const msg = messages.items[0].object;
    try std.testing.expectEqualStrings("user", msg.get("role").?.string);
    try std.testing.expectEqualStrings("Hello there", msg.get("content").?.string);
    try std.testing.expectEqual(@as(i64, 64), obj.get("max_tokens").?.integer);
    try std.testing.expectEqual(@as(f64, 0.5), obj.get("temperature").?.float);
    try std.testing.expectEqual(@as(f64, 0.9), obj.get("top_p").?.float);
    try std.testing.expect(obj.get("stream") == null);
}

test "translateRequest: default model, instructions prepended as system" {
    const allocator = std.testing.allocator;
    var parsed = try expectTranslateOk(allocator,
        \\{"input":"Hi","instructions":"You are terse.","store":false,"metadata":{"k":"v"}}
    );
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expectEqualStrings("default-model", obj.get("model").?.string);
    const messages = obj.get("messages").?.array;
    try std.testing.expectEqual(@as(usize, 2), messages.items.len);
    try std.testing.expectEqualStrings("system", messages.items[0].object.get("role").?.string);
    try std.testing.expectEqualStrings("You are terse.", messages.items[0].object.get("content").?.string);
    try std.testing.expectEqualStrings("user", messages.items[1].object.get("role").?.string);
}

test "translateRequest: array input with input_text parts" {
    const allocator = std.testing.allocator;
    var parsed = try expectTranslateOk(allocator,
        \\{"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"part one"},{"type":"input_text","text":"part two"}]}]}
    );
    defer parsed.deinit();
    const messages = parsed.value.object.get("messages").?.array;
    const parts = messages.items[0].object.get("content").?.array;
    try std.testing.expectEqual(@as(usize, 2), parts.items.len);
    try std.testing.expectEqualStrings("text", parts.items[0].object.get("type").?.string);
    try std.testing.expectEqualStrings("part two", parts.items[1].object.get("text").?.string);
}

test "translateRequest: input_image maps to image_url part" {
    const allocator = std.testing.allocator;
    var parsed = try expectTranslateOk(allocator,
        \\{"input":[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,aGVsbG8="},{"type":"input_text","text":"what is this?"}]}],"stream":true}
    );
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expect(obj.get("stream").?.bool);
    const parts = obj.get("messages").?.array.items[0].object.get("content").?.array;
    const img = parts.items[0].object;
    try std.testing.expectEqualStrings("image_url", img.get("type").?.string);
    try std.testing.expectEqualStrings(
        "data:image/png;base64,aGVsbG8=",
        img.get("image_url").?.object.get("url").?.string,
    );
}

test "translateRequest parses the store flag (default true)" {
    const allocator = std.testing.allocator;
    {
        const outcome = try translateRequest(allocator, "{\"input\":\"hi\"}", "m");
        defer outcome.deinit(allocator);
        try std.testing.expect(switch (outcome) {
            .ok => |o| o.store,
            .err => return error.TestUnexpectedResult,
        });
    }
    {
        const outcome = try translateRequest(allocator, "{\"input\":\"hi\",\"store\":false}", "m");
        defer outcome.deinit(allocator);
        try std.testing.expect(!(switch (outcome) {
            .ok => |o| o.store,
            .err => return error.TestUnexpectedResult,
        }));
    }
    try expectTranslateErr(allocator, "{\"input\":\"hi\",\"store\":\"no\"}", "store");
}

test "translateRequest rejects unsupported fields loudly" {
    const allocator = std.testing.allocator;
    try expectTranslateErr(allocator,
        \\{"input":"hi","tools":[{"type":"function"}]}
    , "tools");
    try expectTranslateErr(allocator,
        \\{"input":"hi","previous_response_id":"resp_abc"}
    , "stateless");
    try expectTranslateErr(allocator,
        \\{"input":"hi","background":true}
    , "background");
    try expectTranslateErr(allocator,
        \\{"input":"hi","reasoning":{"effort":"low"}}
    , "reasoning");
    try expectTranslateErr(allocator,
        \\{"input":"hi","text":{"format":{"type":"json_schema"}}}
    , "text");
    try expectTranslateErr(allocator,
        \\{"input":"hi","include":["reasoning.encrypted_content"]}
    , "include");
    try expectTranslateErr(allocator,
        \\{"input":"hi","truncation":"auto"}
    , "truncation");
    try expectTranslateErr(allocator,
        \\{"input":"hi","some_future_field":1}
    , "some_future_field");
}

test "translateRequest validates input shapes and top_p" {
    const allocator = std.testing.allocator;
    try expectTranslateErr(allocator,
        \\{"model":"m"}
    , "input");
    try expectTranslateErr(allocator,
        \\{"input":42}
    , "input");
    try expectTranslateErr(allocator,
        \\{"input":[{"role":"user"}]}
    , "content");
    try expectTranslateErr(allocator,
        \\{"input":[{"role":"user","content":[{"type":"input_file","file_data":"..."}]}]}
    , "input_file");
    try expectTranslateErr(allocator,
        \\{"input":[{"type":"function_call","name":"f"}]}
    , "function_call");
    try expectTranslateErr(allocator,
        \\{"input":"hi","top_p":0}
    , "top_p");
    try expectTranslateErr(allocator,
        \\{"input":"hi","top_p":1.5}
    , "top_p");
    try expectTranslateErr(allocator, "not json", "valid JSON");
}

test "translateRequest rejects max_output_tokens overflow" {
    const allocator = std.testing.allocator;
    // 5e9 exceeds maxInt(u32); must be a clean 400, never an @intCast panic.
    try expectTranslateErr(allocator,
        \\{"input":"hi","max_output_tokens":5000000000}
    , "max_output_tokens");
    try expectTranslateErr(allocator,
        \\{"input":"hi","max_output_tokens":4294967296}
    , "max_output_tokens");
    // maxInt(u32) itself is accepted and threaded through.
    var parsed = try expectTranslateOk(allocator,
        \\{"input":"hi","max_output_tokens":4294967295}
    );
    defer parsed.deinit();
    try std.testing.expectEqual(@as(i64, 4294967295), parsed.value.object.get("max_tokens").?.integer);
}

test "errorBodyStatus maps server_error to 500, others to 400" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(@as(u16, 500), errorBodyStatus(allocator,
        \\{"error":{"message":"boom","type":"server_error","code":"generation_failed"}}
    ));
    try std.testing.expectEqual(@as(u16, 400), errorBodyStatus(allocator,
        \\{"error":{"message":"bad","type":"invalid_request_error","code":"InvalidContent"}}
    ));
    try std.testing.expectEqual(@as(u16, 400), errorBodyStatus(allocator, "not json"));
    try std.testing.expectEqual(@as(u16, 400), errorBodyStatus(allocator,
        \\{"error":"string form"}
    ));
}

test "statusForFinishReason maps length to incomplete/max_output_tokens" {
    const st = statusForFinishReason("length");
    try std.testing.expectEqualStrings("incomplete", st.status);
    try std.testing.expectEqualStrings("max_output_tokens", st.incomplete_reason.?);
    const ok = statusForFinishReason("stop");
    try std.testing.expectEqualStrings("completed", ok.status);
    try std.testing.expect(ok.incomplete_reason == null);
}

test "buildResponseObject produces the Responses shape" {
    const allocator = std.testing.allocator;
    const chat =
        \\{"id":"chatcmpl-1","object":"chat.completion","created":1794458459,"model":"Qwen3","choices":[{"index":0,"message":{"role":"assistant","content":"hi \"there\"\n"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}
    ;
    const body = try buildResponseObject(allocator, chat, "resp_ab12", "msg_cd34");
    defer allocator.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expectEqualStrings("resp_ab12", obj.get("id").?.string);
    try std.testing.expectEqualStrings("response", obj.get("object").?.string);
    try std.testing.expectEqual(@as(i64, 1794458459), obj.get("created_at").?.integer);
    try std.testing.expectEqualStrings("completed", obj.get("status").?.string);
    try std.testing.expectEqualStrings("Qwen3", obj.get("model").?.string);
    const output = obj.get("output").?.array;
    try std.testing.expectEqual(@as(usize, 1), output.items.len);
    const item = output.items[0].object;
    try std.testing.expectEqualStrings("message", item.get("type").?.string);
    try std.testing.expectEqualStrings("msg_cd34", item.get("id").?.string);
    try std.testing.expectEqualStrings("assistant", item.get("role").?.string);
    const content = item.get("content").?.array;
    try std.testing.expectEqualStrings("output_text", content.items[0].object.get("type").?.string);
    try std.testing.expectEqualStrings("hi \"there\"\n", content.items[0].object.get("text").?.string);
    const usage = obj.get("usage").?.object;
    try std.testing.expectEqual(@as(i64, 11), usage.get("input_tokens").?.integer);
    try std.testing.expectEqual(@as(i64, 7), usage.get("output_tokens").?.integer);
    try std.testing.expectEqual(@as(i64, 18), usage.get("total_tokens").?.integer);
    try std.testing.expect(std.meta.activeTag(obj.get("error").?) == .null);
    try std.testing.expect(std.meta.activeTag(obj.get("incomplete_details").?) == .null);
}

test "buildResponseObject maps length finish to incomplete" {
    const allocator = std.testing.allocator;
    const chat =
        \\{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"partial"},"finish_reason":"length"}],"usage":{"prompt_tokens":3,"completion_tokens":9,"total_tokens":12}}
    ;
    const body = try buildResponseObject(allocator, chat, "resp_x", "msg_y");
    defer allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expectEqualStrings("incomplete", obj.get("status").?.string);
    try std.testing.expectEqualStrings(
        "max_output_tokens",
        obj.get("incomplete_details").?.object.get("reason").?.string,
    );
}
