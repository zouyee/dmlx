/// Responses API response formatting: chat-completion results are wrapped
/// in Responses objects (`output` item arrays, usage, status), and SSE
/// event payloads are built for the streaming path.
///
/// Output items:
///   - message items carry assistant text (output_text content part);
///   - function_call items carry tool calls parsed from model output
///     (`arguments` stays a JSON string, per the Responses convention).
const std = @import("std");
const utils_mod = @import("../utils.zig");

const jsonEscapeInto = utils_mod.jsonEscapeInto;

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

/// Append an assistant message output item (output_text part) to a JSON
/// array under construction.
pub fn appendMessageItem(
    out: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    msg_id: []const u8,
    text: []const u8,
) !void {
    try out.appendSlice(allocator, "{\"type\":\"message\",\"id\":\"");
    try jsonEscapeInto(out, allocator, msg_id);
    try out.appendSlice(allocator, "\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"");
    try jsonEscapeInto(out, allocator, text);
    try out.appendSlice(allocator, "\"}]}");
}

/// Append a function_call output item. `arguments` is a JSON *string* and
/// is escaped, never re-serialized.
pub fn appendFunctionCallItem(
    out: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    item_id: []const u8,
    call_id: []const u8,
    name: []const u8,
    arguments: []const u8,
) !void {
    try out.appendSlice(allocator, "{\"type\":\"function_call\",\"id\":\"");
    try jsonEscapeInto(out, allocator, item_id);
    try out.appendSlice(allocator, "\",\"status\":\"completed\",\"call_id\":\"");
    try jsonEscapeInto(out, allocator, call_id);
    try out.appendSlice(allocator, "\",\"name\":\"");
    try jsonEscapeInto(out, allocator, name);
    try out.appendSlice(allocator, "\",\"arguments\":\"");
    try jsonEscapeInto(out, allocator, arguments);
    try out.appendSlice(allocator, "\"}");
}

/// Format a Responses API object from the extracted pieces. `output_json`
/// is a pre-built JSON array of output items (see append*Item above).
pub fn formatResponseJson(
    allocator: std.mem.Allocator,
    resp_id: []const u8,
    created_at: u64,
    model: []const u8,
    output_json: []const u8,
    finish_reason: []const u8,
    input_tokens: usize,
    output_tokens: usize,
) ![]u8 {
    const st = statusForFinishReason(finish_reason);
    var escaped_model = std.ArrayList(u8).empty;
    defer escaped_model.deinit(allocator);
    try jsonEscapeInto(&escaped_model, allocator, model);

    const incomplete = if (st.incomplete_reason) |r|
        try std.fmt.allocPrint(allocator, "{{\"reason\":\"{s}\"}}", .{r})
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(incomplete);

    return std.fmt.allocPrint(allocator,
        \\{{"id":"{s}","object":"response","created_at":{d},"status":"{s}","model":"{s}","output":{s},"usage":{{"input_tokens":{d},"output_tokens":{d},"total_tokens":{d}}},"error":null,"incomplete_details":{s}}}
    , .{
        resp_id,
        created_at,
        st.status,
        escaped_model.items,
        output_json,
        input_tokens,
        output_tokens,
        input_tokens + output_tokens,
        incomplete,
    });
}

/// Build the output-items JSON array for a plain text turn (single message
/// item). Caller owns the returned slice.
pub fn buildTextOutputJson(allocator: std.mem.Allocator, msg_id: []const u8, text: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.append(allocator, '[');
    try appendMessageItem(&out, allocator, msg_id, text);
    try out.append(allocator, ']');
    return out.toOwnedSlice(allocator);
}

/// Wrap a successful chat-completion response body in a Responses object.
/// Tool calls in the chat body become function_call output items; text
/// content (when present) becomes a leading message item.
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
    const text: ?[]const u8 = switch (message.get("content") orelse .null) {
        .string => |s| s,
        else => null,
    };
    const usage = obj.get("usage").?.object;
    const input_tokens: usize = @intCast(usage.get("prompt_tokens").?.integer);
    const output_tokens: usize = @intCast(usage.get("completion_tokens").?.integer);

    var output = std.ArrayList(u8).empty;
    defer output.deinit(allocator);
    try output.append(allocator, '[');
    var n_items: usize = 0;
    if (text) |t| {
        if (t.len > 0) {
            try appendMessageItem(&output, allocator, msg_id, t);
            n_items += 1;
        }
    }
    if (message.get("tool_calls")) |tcs| {
        if (tcs == .array) {
            for (tcs.array.items) |tc| {
                if (tc != .object) continue;
                const f = switch (tc.object.get("function") orelse continue) {
                    .object => |o| o,
                    else => continue,
                };
                const call_id = switch (tc.object.get("id") orelse continue) {
                    .string => |s| s,
                    else => continue,
                };
                const name = switch (f.get("name") orelse continue) {
                    .string => |s| s,
                    else => continue,
                };
                const arguments: []const u8 = switch (f.get("arguments") orelse .null) {
                    .string => |s| s,
                    else => "{}",
                };
                if (n_items > 0) try output.append(allocator, ',');
                // The item id mirrors the call id; both must round-trip so
                // the client's function_call_output references line up.
                try appendFunctionCallItem(&output, allocator, call_id, call_id, name, arguments);
                n_items += 1;
            }
        }
    }
    try output.append(allocator, ']');

    return formatResponseJson(allocator, resp_id, created, model, output.items, finish, input_tokens, output_tokens);
}

/// Wrap a FAILED chat-path error body in a failed Responses object (status
/// "failed", empty output, error carried over). Falls back to a generic
/// server_error when the body doesn't parse. Caller owns the result.
pub fn buildFailedResponseObject(
    allocator: std.mem.Allocator,
    chat_error_body: []const u8,
    resp_id: []const u8,
    created_at: u64,
    model: []const u8,
) ![]u8 {
    var message: []const u8 = "generation failed";
    var err_type: []const u8 = "server_error";
    var message_owned: ?[]u8 = null;
    defer if (message_owned) |m| allocator.free(m);
    var type_owned: ?[]u8 = null;
    defer if (type_owned) |t| allocator.free(t);
    if (std.json.parseFromSlice(std.json.Value, allocator, chat_error_body, .{}) catch null) |parsed| {
        defer parsed.deinit();
        if (parsed.value == .object) {
            if (parsed.value.object.get("error")) |e| {
                if (e == .object) {
                    // Dupe: the parsed tree dies at this block's end.
                    if (e.object.get("message")) |m| {
                        if (m == .string) {
                            message_owned = try allocator.dupe(u8, m.string);
                            message = message_owned.?;
                        }
                    }
                    if (e.object.get("type")) |t| {
                        if (t == .string) {
                            type_owned = try allocator.dupe(u8, t.string);
                            err_type = type_owned.?;
                        }
                    }
                }
            }
        }
    }
    var escaped_message = std.ArrayList(u8).empty;
    defer escaped_message.deinit(allocator);
    try jsonEscapeInto(&escaped_message, allocator, message);
    var escaped_model = std.ArrayList(u8).empty;
    defer escaped_model.deinit(allocator);
    try jsonEscapeInto(&escaped_model, allocator, model);

    var escaped_type = std.ArrayList(u8).empty;
    defer escaped_type.deinit(allocator);
    try jsonEscapeInto(&escaped_type, allocator, err_type);

    return std.fmt.allocPrint(allocator,
        \\{{"id":"{s}","object":"response","created_at":{d},"status":"failed","model":"{s}","output":[],"error":{{"message":"{s}","type":"{s}"}},"incomplete_details":null}}
    , .{ resp_id, created_at, escaped_model.items, escaped_message.items, escaped_type.items });
}

// ------------------------------------------------------------------
// SSE event payloads
// ------------------------------------------------------------------

pub fn createdPayload(allocator: std.mem.Allocator, resp_id: []const u8, created_at: u64, escaped_model: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\{{"type":"response.created","response":{{"id":"{s}","object":"response","created_at":{d},"status":"in_progress","model":"{s}","output":[],"error":null,"incomplete_details":null}}}}
    , .{ resp_id, created_at, escaped_model });
}

pub fn textDeltaPayload(allocator: std.mem.Allocator, resp_id: []const u8, msg_id: []const u8, delta: []const u8) ![]u8 {
    var escaped_delta = std.ArrayList(u8).empty;
    defer escaped_delta.deinit(allocator);
    try jsonEscapeInto(&escaped_delta, allocator, delta);
    return std.fmt.allocPrint(allocator,
        \\{{"type":"response.output_text.delta","response_id":"{s}","item_id":"{s}","output_index":0,"content_index":0,"delta":"{s}"}}
    , .{ resp_id, msg_id, escaped_delta.items });
}

pub fn textDonePayload(allocator: std.mem.Allocator, resp_id: []const u8, msg_id: []const u8, text: []const u8) ![]u8 {
    var escaped_full = std.ArrayList(u8).empty;
    defer escaped_full.deinit(allocator);
    try jsonEscapeInto(&escaped_full, allocator, text);
    return std.fmt.allocPrint(allocator,
        \\{{"type":"response.output_text.done","response_id":"{s}","item_id":"{s}","output_index":0,"content_index":0,"text":"{s}"}}
    , .{ resp_id, msg_id, escaped_full.items });
}

/// `response.output_item.done` carrying a complete output item JSON object
/// (built by appendMessageItem / appendFunctionCallItem).
pub fn itemDonePayload(allocator: std.mem.Allocator, resp_id: []const u8, item_json: []const u8, output_index: usize) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\{{"type":"response.output_item.done","response_id":"{s}","output_index":{d},"item":{s}}}
    , .{ resp_id, output_index, item_json });
}

/// `response.output_item.added` announcing an output item whose payload
/// follows in later events (the done event carries the complete item).
pub fn itemAddedPayload(allocator: std.mem.Allocator, resp_id: []const u8, item_json: []const u8, output_index: usize) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\{{"type":"response.output_item.added","response_id":"{s}","output_index":{d},"item":{s}}}
    , .{ resp_id, output_index, item_json });
}

pub fn completedPayload(allocator: std.mem.Allocator, response_json: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\{{"type":"response.completed","response":{s}}}
    , .{response_json});
}

pub fn failedPayload(allocator: std.mem.Allocator, response_json: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\{{"type":"response.failed","response":{s}}}
    , .{response_json});
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

test "buildFailedResponseObject carries the chat error into a failed response" {
    const allocator = std.testing.allocator;
    const body = try buildFailedResponseObject(allocator,
        \\{"error":{"message":"vision input \"bad\" is not supported","type":"invalid_request_error","code":"VisionNotSupported"}}
    , "resp_fail", 1794458459, "Qwen3");
    defer allocator.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expectEqualStrings("resp_fail", obj.get("id").?.string);
    try std.testing.expectEqualStrings("response", obj.get("object").?.string);
    try std.testing.expectEqualStrings("failed", obj.get("status").?.string);
    try std.testing.expectEqualStrings("Qwen3", obj.get("model").?.string);
    try std.testing.expectEqual(@as(usize, 0), obj.get("output").?.array.items.len);
    const err = obj.get("error").?.object;
    try std.testing.expectEqualStrings("vision input \"bad\" is not supported", err.get("message").?.string);
    try std.testing.expectEqualStrings("invalid_request_error", err.get("type").?.string);
    try std.testing.expect(std.meta.activeTag(obj.get("incomplete_details").?) == .null);
}

test "buildFailedResponseObject falls back on unparseable bodies" {
    const allocator = std.testing.allocator;
    const body = try buildFailedResponseObject(allocator, "not json", "resp_x", 1, "m");
    defer allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const err = parsed.value.object.get("error").?.object;
    try std.testing.expectEqualStrings("generation failed", err.get("message").?.string);
    try std.testing.expectEqualStrings("server_error", err.get("type").?.string);
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

test "buildResponseObject maps tool_calls to function_call items" {
    const allocator = std.testing.allocator;
    const chat =
        \\{"id":"chatcmpl-1","object":"chat.completion","created":5,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"shell","arguments":"{\"cmd\":\"ls -l\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}
    ;
    const body = try buildResponseObject(allocator, chat, "resp_t", "msg_t");
    defer allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const output = parsed.value.object.get("output").?.array;
    try std.testing.expectEqual(@as(usize, 1), output.items.len);
    const item = output.items[0].object;
    try std.testing.expectEqualStrings("function_call", item.get("type").?.string);
    try std.testing.expectEqualStrings("call_1", item.get("call_id").?.string);
    try std.testing.expectEqualStrings("shell", item.get("name").?.string);
    try std.testing.expectEqualStrings("{\"cmd\":\"ls -l\"}", item.get("arguments").?.string);
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

test "SSE payload builders escape content" {
    const allocator = std.testing.allocator;
    const delta = try textDeltaPayload(allocator, "resp_1", "msg_1", "a\"b\nc");
    defer allocator.free(delta);
    try std.testing.expect(std.mem.indexOf(u8, delta, "a\\\"b\\nc") != null);

    const done = try itemDonePayload(allocator, "resp_1", "{\"type\":\"function_call\"}", 2);
    defer allocator.free(done);
    try std.testing.expect(std.mem.indexOf(u8, done, "\"output_index\":2") != null);
}
