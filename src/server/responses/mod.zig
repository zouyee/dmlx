/// OpenAI Responses API (`POST /v1/responses`) — compatibility layer.
///
/// This is NOT a full Responses API implementation. It is a translation
/// layer onto the existing chat-completions machinery:
///   - translate.zig rewrites the Responses request into an internal
///     chat-completion request (messages, tools, sampling params);
///   - the chat engine flow executes it unchanged;
///   - format.zig wraps the result in a Responses object and builds the
///     SSE event payloads.
///
/// Streaming emits: response.created → response.output_text.delta* →
/// response.output_item.done → response.completed. When the request carries
/// tools, text deltas are buffered and tool-call markup is parsed at end of
/// stream, so function_call items — never raw markup — reach the client.
const std = @import("std");
const engine = @import("../../engine/root.zig");
const tool_calling_mod = @import("../../tool_calling.zig");
const config_mod = @import("../config.zig");
const state_mod = @import("../state.zig");
const utils_mod = @import("../utils.zig");
const http_mod = @import("../http.zig");
const sse_mod = @import("../sse.zig");
const streaming_mod = @import("../streaming.zig");
const metrics_mod = @import("../metrics.zig");
const openai_mod = @import("../openai.zig");

pub const translate = @import("translate.zig");
pub const format = @import("format.zig");

pub const TranslateOutcome = translate.TranslateOutcome;
pub const translateRequest = translate.translateRequest;
pub const ResponseStatus = format.ResponseStatus;
pub const statusForFinishReason = format.statusForFinishReason;
pub const formatResponseJson = format.formatResponseJson;
pub const buildResponseObject = format.buildResponseObject;
pub const buildFailedResponseObject = format.buildFailedResponseObject;

const ServerConfig = config_mod.ServerConfig;
const ServerState = state_mod.ServerState;
const jsonEscapeInto = utils_mod.jsonEscapeInto;

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
    const outcome = try translate.translateRequest(allocator, body, default_model);
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
        try handleResponsesStream(allocator, io, state, connection, server_config, parsed.value, resp_id, msg_id, translated.store, translated.has_tools, status_out);
        return;
    }

    const chat_response = try openai_mod.generateChatCompletion(allocator, io, state, translated.body, server_config);
    defer allocator.free(chat_response);

    // generateChatCompletion reports validation/generation failures as an
    // error body; pass those through, mapping the error type onto the HTTP
    // status (server_error → 500, everything else → 400). Failed responses
    // are retained in the store like completed ones (store=false opts out),
    // matching the streaming path's response.failed retention.
    if (std.mem.indexOf(u8, chat_response, "\"error\"") != null) {
        const status = errorBodyStatus(allocator, chat_response);
        status_out.* = status;
        if (translated.store) {
            const created = @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toSeconds()));
            const failed = try format.buildFailedResponseObject(allocator, chat_response, resp_id, created, translated.model);
            defer allocator.free(failed);
            try state.response_store.put(resp_id, failed);
        }
        try http_mod.writeJsonResponse(connection, io, status, chat_response);
        return;
    }

    const response_json = try format.buildResponseObject(allocator, chat_response, resp_id, msg_id);
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
// Streaming: response.created → output_text.delta* → output_item.done
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

/// Accumulate one token's text and emit the output_text.delta SSE event.
fn deliverResponsesDelta(
    sse: *sse_mod.SSEWriter,
    allocator: std.mem.Allocator,
    all_text: *std.ArrayList(u8),
    token_text: []const u8,
    resp_id: []const u8,
    msg_id: []const u8,
) !void {
    try all_text.appendSlice(allocator, token_text);
    const delta_payload = try format.textDeltaPayload(allocator, resp_id, msg_id, token_text);
    defer allocator.free(delta_payload);
    try sendResponsesEvent(sse, allocator, "response.output_text.delta", delta_payload);
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
    has_tools: bool,
    status_out: *u16,
) !void {
    const created = @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toSeconds()));

    // Tool definitions inject a leading system prompt (same as the chat
    // non-streaming path) so the model knows the native call format.
    var tool_system: ?[]u8 = null;
    defer if (tool_system) |ts| allocator.free(ts);
    tool_system = try openai_mod.buildToolSystem(allocator, req);

    // Build the prompt BEFORE committing SSE headers so preparation
    // failures still surface as a clean HTTP 400.
    var prepared = openai_mod.preparePrompt(allocator, state, req, tool_system) catch |err| {
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
    // The wait is conditional: fallible SSE writes between here and the push
    // can return early with the request never submitted, and an unsubmitted
    // request never gets engine_done set.
    var req_submitted = false;
    defer {
        if (req_submitted) req_state.waitEngineDone(io);
        req_state.deinit();
    }
    prepared.vision = null;

    try sse_mod.writeSSEHeaders(connection, io);
    status_out.* = 200; // SSE headers committed as 200; failures are in-stream events
    var sse = sse_mod.SSEWriter{ .connection = connection, .io = io };

    var escaped_model = std.ArrayList(u8).empty;
    defer escaped_model.deinit(allocator);
    try jsonEscapeInto(&escaped_model, allocator, req.model);

    const created_payload = try format.createdPayload(allocator, resp_id, created, escaped_model.items);
    defer allocator.free(created_payload);
    try sendResponsesEvent(&sse, allocator, "response.created", created_payload);

    _ = state.active_requests.fetchAdd(1, .monotonic);
    defer _ = state.active_requests.fetchSub(1, .monotonic);

    var node = engine.QueueNode.init(req_state);
    state.request_queue.push(&node);
    req_submitted = true;

    var all_text = std.ArrayList(u8).empty;
    defer all_text.deinit(allocator);
    var finish_reason: []const u8 = "stop";

    while (true) {
        // Any failure inside this loop (write error, cancellation, OOM)
        // while the engine may still be mid-generation must go through
        // abortStream: cancel + drain the completion signal before this
        // frame unwinds, or req_state.deinit() frees the signal out from
        // under the engine's deliverToken calls.
        const event = req_state.completion.waitForToken(io) catch {
            streaming_mod.abortStream(io, req_state, allocator);
            return error.Canceled;
        };
        if (event) |e| {
            defer if (e.token_text.len > 0) allocator.free(e.token_text);
            if (e.token_text.len > 0) {
                if (has_tools) {
                    // Buffer: tool-call markup must never leak into text
                    // deltas. The parse-vs-stream decision is made once the
                    // full output is available, after the loop.
                    try all_text.appendSlice(allocator, e.token_text);
                } else {
                    deliverResponsesDelta(&sse, allocator, &all_text, e.token_text, resp_id, msg_id) catch |err| {
                        streaming_mod.abortStream(io, req_state, allocator);
                        return err;
                    };
                }
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
        const failed_payload = try format.failedPayload(allocator, failed_response);
        defer allocator.free(failed_payload);
        try sendResponsesEvent(&sse, allocator, "response.failed", failed_payload);
        return;
    }

    // Tool turns: parse the buffered output; on any detected call emit
    // function_call items, otherwise fall through to the plain-text path
    // (the deltas were withheld, so emit the full text in one delta first).
    if (has_tools) {
        const family = tool_calling_mod.detectFamily(req.model);
        if (try tool_calling_mod.parse(allocator, all_text.items, family)) |parse_result| {
            defer parse_result.deinit();
            if (parse_result.calls.len > 0) {
                var output = std.ArrayList(u8).empty;
                defer output.deinit(allocator);
                try output.append(allocator, '[');
                for (parse_result.calls, 0..) |call, i| {
                    if (i > 0) try output.append(allocator, ',');
                    // Item id mirrors the chat-side call id so the client's
                    // function_call_output call_id reference lines up.
                    try format.appendFunctionCallItem(&output, allocator, call.id, call.id, call.function.name, call.function.arguments);
                }
                try output.append(allocator, ']');

                // Codex ignores function_call_arguments.delta; a single
                // output_item.done per call carries the complete arguments.
                var item_buf = std.ArrayList(u8).empty;
                defer item_buf.deinit(allocator);
                for (parse_result.calls, 0..) |call, i| {
                    item_buf.clearRetainingCapacity();
                    try format.appendFunctionCallItem(&item_buf, allocator, call.id, call.id, call.function.name, call.function.arguments);
                    const payload = try format.itemDonePayload(allocator, resp_id, item_buf.items, i);
                    defer allocator.free(payload);
                    try sendResponsesEvent(&sse, allocator, "response.output_item.done", payload);
                }

                const final_response = try format.formatResponseJson(
                    allocator,
                    resp_id,
                    created,
                    req.model,
                    output.items,
                    "stop",
                    prompt_tokens.len,
                    req_state.token_count,
                );
                defer allocator.free(final_response);
                if (store) try state.response_store.put(resp_id, final_response);
                const completed_payload = try format.completedPayload(allocator, final_response);
                defer allocator.free(completed_payload);
                try sendResponsesEvent(&sse, allocator, "response.completed", completed_payload);
                return;
            }
        }
        // No tool call detected: deliver the withheld text, wrapped in a
        // message item so clients that track items by id (Codex errors on
        // "OutputTextDelta without active item") stay consistent.
        if (all_text.items.len > 0) {
            var item_buf = std.ArrayList(u8).empty;
            defer item_buf.deinit(allocator);
            try format.appendMessageItem(&item_buf, allocator, msg_id, "");
            const added_payload = try format.itemAddedPayload(allocator, resp_id, item_buf.items, 0);
            defer allocator.free(added_payload);
            try sendResponsesEvent(&sse, allocator, "response.output_item.added", added_payload);

            const delta_payload = try format.textDeltaPayload(allocator, resp_id, msg_id, all_text.items);
            defer allocator.free(delta_payload);
            try sendResponsesEvent(&sse, allocator, "response.output_text.delta", delta_payload);

            item_buf.clearRetainingCapacity();
            try format.appendMessageItem(&item_buf, allocator, msg_id, all_text.items);
            const done_payload = try format.itemDonePayload(allocator, resp_id, item_buf.items, 0);
            defer allocator.free(done_payload);
            try sendResponsesEvent(&sse, allocator, "response.output_item.done", done_payload);
        }
    }

    const done_payload = try format.textDonePayload(allocator, resp_id, msg_id, all_text.items);
    defer allocator.free(done_payload);
    try sendResponsesEvent(&sse, allocator, "response.output_text.done", done_payload);

    const output_json = try format.buildTextOutputJson(allocator, msg_id, all_text.items);
    defer allocator.free(output_json);
    const final_response = try format.formatResponseJson(
        allocator,
        resp_id,
        created,
        req.model,
        output_json,
        finish_reason,
        prompt_tokens.len,
        req_state.token_count,
    );
    defer allocator.free(final_response);
    if (store) try state.response_store.put(resp_id, final_response);
    const completed_payload = try format.completedPayload(allocator, final_response);
    defer allocator.free(completed_payload);
    try sendResponsesEvent(&sse, allocator, "response.completed", completed_payload);
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
