/// Anthropic Messages API compatibility.
const std = @import("std");
const root = @import("../root.zig");
const c = @import("mlx").c;
const generation_mod = @import("../generation.zig");
const kvcache = @import("../kvcache.zig");
const engine = @import("../engine/root.zig");
const config_mod = @import("config.zig");
const utils_mod = @import("utils.zig");
const state_mod = @import("state.zig");
const jsonEscapeInto = utils_mod.jsonEscapeInto;
const http_mod = @import("http.zig");
const writeJsonResponse = http_mod.writeJsonResponse;
const metrics_mod = @import("metrics.zig");
const sse_mod = @import("sse.zig");
const tool_calling_mod = @import("../tool_calling.zig");

const ServerConfig = config_mod.ServerConfig;
const ServerState = state_mod.ServerState;

const AnthropicRequest = struct {
    model: []const u8,
    messages: []AnthropicMessage,
    max_tokens: u32 = 1024,
    temperature: ?f32 = null,
    /// Per-request nucleus sampling override (must be in (0, 1]).
    top_p: ?f32 = null,
    /// Per-request top-k override (must be >= 1).
    top_k: ?u32 = null,
    stream: ?bool = false,
    system: ?[]const u8 = null,
    /// Anthropic tool_choice: {"type":"auto"|"any"|"none"|"tool"}. Mapped to
    /// the OpenAI policy: auto→auto, none→none, any→required, tool→named.
    /// "any"/"tool" would force a call; rejected.
    tool_choice: ?AnthropicToolChoice = null,

    pub const AnthropicMessage = struct {
        role: []const u8,
        content: []const u8,
    };

    pub const AnthropicToolChoice = struct {
        type: []const u8,
        name: ?[]const u8 = null,
    };
};

/// Anthropic tool_choice support policy: "auto" enables tools, "none" strips
/// them; "any"/"tool" would force a call and are rejected with the message
/// returned here.
pub fn unsupportedToolChoiceMsg(tc: AnthropicRequest.AnthropicToolChoice) ?[]const u8 {
    if (std.mem.eql(u8, tc.type, "auto") or std.mem.eql(u8, tc.type, "none")) return null;
    return "tool_choice 'any'/'tool' (forced calls) are not supported by this server (use 'auto' or 'none')";
}

// ------------------------------------------------------------------
// Response blocks
// ------------------------------------------------------------------

const ToolUseBlock = struct {
    id: []const u8,
    name: []const u8,
    /// Serialized JSON object string for the `input` field.
    input_json: []const u8,
};

const ResponseBlock = union(enum) {
    text: []const u8,
    tool_use: ToolUseBlock,
};

fn writeEscaped(w: anytype, allocator: std.mem.Allocator, text: []const u8) !void {
    var escaped = std.ArrayList(u8).empty;
    defer escaped.deinit(allocator);
    try jsonEscapeInto(&escaped, allocator, text);
    try w.appendSlice(allocator, escaped.items);
}

/// Serialize one content block into `out` (`{"type":"text",...}` or
/// `{"type":"tool_use",...}`).
fn appendContentBlockJson(out: *std.ArrayList(u8), allocator: std.mem.Allocator, block: ResponseBlock) !void {
    switch (block) {
        .text => |t| {
            try out.appendSlice(allocator, "{\"type\":\"text\",\"text\":\"");
            try writeEscaped(out, allocator, t);
            try out.appendSlice(allocator, "\"}");
        },
        .tool_use => |tu| {
            try out.appendSlice(allocator, "{\"type\":\"tool_use\",\"id\":\"");
            try writeEscaped(out, allocator, tu.id);
            try out.appendSlice(allocator, "\",\"name\":\"");
            try writeEscaped(out, allocator, tu.name);
            try out.appendSlice(allocator, "\",\"input\":");
            try out.appendSlice(allocator, if (tu.input_json.len > 0) tu.input_json else "{}");
            try out.appendSlice(allocator, "}");
        },
    }
}

/// Build the non-streaming Messages API response body.
fn formatAnthropicResponse(
    allocator: std.mem.Allocator,
    model: []const u8,
    blocks: []const ResponseBlock,
    input_tokens: usize,
    output_tokens: usize,
    stop_reason: []const u8,
) ![]u8 {
    var content = std.ArrayList(u8).empty;
    defer content.deinit(allocator);
    try content.appendSlice(allocator, "[");
    for (blocks, 0..) |b, i| {
        if (i > 0) try content.appendSlice(allocator, ",");
        try appendContentBlockJson(&content, allocator, b);
    }
    try content.appendSlice(allocator, "]");

    var escaped_model = std.ArrayList(u8).empty;
    defer escaped_model.deinit(allocator);
    try jsonEscapeInto(&escaped_model, allocator, model);

    return try std.fmt.allocPrint(allocator,
        \\{{"id":"msg_1","type":"message","role":"assistant","content":{s},"model":"{s}","stop_reason":"{s}","usage":{{"input_tokens":{d},"output_tokens":{d}}}}}
    , .{ content.items, escaped_model.items, stop_reason, input_tokens, output_tokens });
}

// ------------------------------------------------------------------
// Request translation (flexible content blocks, tools, system arrays)
// ------------------------------------------------------------------

const TranslateError = error{ InvalidRequest, OutOfMemory };

/// Render a historical assistant tool call in the same textual format the
/// injected tool system prompt asks the model to produce, so the prompt and
/// the conversation history stay consistent per model family. For Qwen this
/// is the native chat-template form:
///   <tool_call>\n<function=name>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>
fn appendHistoricalCall(out: *std.ArrayList(u8), allocator: std.mem.Allocator, family: tool_calling_mod.ModelFamily, name: []const u8, input: ?std.json.Value) !void {
    if (family == .qwen) {
        try out.appendSlice(allocator, "<tool_call>\n<function=");
        try out.appendSlice(allocator, name);
        try out.appendSlice(allocator, ">\n");
        if (input) |inp| {
            if (inp == .object) {
                var it = inp.object.iterator();
                while (it.next()) |entry| {
                    try out.appendSlice(allocator, "<parameter=");
                    try out.appendSlice(allocator, entry.key_ptr.*);
                    try out.appendSlice(allocator, ">\n");
                    switch (entry.value_ptr.*) {
                        .string => |s| try out.appendSlice(allocator, s),
                        else => |v| {
                            const j = try std.json.Stringify.valueAlloc(allocator, v, .{});
                            defer allocator.free(j);
                            try out.appendSlice(allocator, j);
                        },
                    }
                    try out.appendSlice(allocator, "\n</parameter>\n");
                }
            }
        }
        try out.appendSlice(allocator, "</function>\n</tool_call>");
        return;
    }
    try out.appendSlice(allocator, "<|python_tag|>");
    try out.appendSlice(allocator, "{\"name\": \"");
    try out.appendSlice(allocator, name);
    try out.appendSlice(allocator, "\", \"arguments\": ");
    if (input) |inp| {
        const j = try std.json.Stringify.valueAlloc(allocator, inp, .{});
        defer allocator.free(j);
        try out.appendSlice(allocator, if (j.len > 0) j else "{}");
    } else {
        try out.appendSlice(allocator, "{}");
    }
    try out.appendSlice(allocator, "}");
}

/// Flatten an Anthropic `content` value (string or block array) into plain
/// text for the chat template. tool_use/tool_result blocks are rendered as
/// text in the family's call format / "Tool 'name' result:" convention.
fn flattenContent(out: *std.ArrayList(u8), allocator: std.mem.Allocator, content: std.json.Value, family: tool_calling_mod.ModelFamily) TranslateError!void {
    switch (content) {
        .string => |s| try out.appendSlice(allocator, s),
        .array => |blocks| {
            for (blocks.items) |b| {
                const obj = switch (b) {
                    .object => |o| o,
                    else => continue,
                };
                const btype = if (obj.get("type")) |t| switch (t) {
                    .string => |s| s,
                    else => continue,
                } else continue;
                if (std.mem.eql(u8, btype, "text")) {
                    if (obj.get("text")) |t| {
                        switch (t) {
                            .string => |s| try out.appendSlice(allocator, s),
                            else => {},
                        }
                    }
                } else if (std.mem.eql(u8, btype, "tool_use")) {
                    const name = if (obj.get("name")) |n| (if (n == .string) n.string else "?") else "?";
                    try appendHistoricalCall(out, allocator, family, name, obj.get("input"));
                    try out.appendSlice(allocator, "\n");
                } else if (std.mem.eql(u8, btype, "tool_result")) {
                    // Qwen chat template wraps tool output in <tool_response>.
                    if (family == .qwen) {
                        try out.appendSlice(allocator, "<tool_response>\n");
                    } else {
                        try out.appendSlice(allocator, "Tool result: ");
                    }
                    if (obj.get("content")) |rc| {
                        switch (rc) {
                            .string => |s| try out.appendSlice(allocator, s),
                            .array => |parts| {
                                for (parts.items) |p| {
                                    if (p != .object) continue;
                                    const pt = p.object.get("type") orelse continue;
                                    if (pt != .string or !std.mem.eql(u8, pt.string, "text")) continue;
                                    const txt = p.object.get("text") orelse continue;
                                    if (txt == .string) try out.appendSlice(allocator, txt.string);
                                }
                            },
                            else => {},
                        }
                    }
                    if (family == .qwen) {
                        try out.appendSlice(allocator, "\n</tool_response>");
                    }
                    try out.appendSlice(allocator, "\n");
                }
            }
        },
        else => {},
    }
}

// ------------------------------------------------------------------
// SSE (simulated streaming from the completed generation)
// ------------------------------------------------------------------

fn sendSSE(sse: *sse_mod.SSEWriter, allocator: std.mem.Allocator, event: []const u8, data: []const u8) !void {
    _ = allocator;
    try sse.sendNamedEvent(event, data);
}

fn streamBlocks(
    allocator: std.mem.Allocator,
    sse: *sse_mod.SSEWriter,
    model: []const u8,
    blocks: []const ResponseBlock,
    stop_reason: []const u8,
    input_tokens: usize,
    output_tokens: usize,
) !void {
    var model_esc = std.ArrayList(u8).empty;
    defer model_esc.deinit(allocator);
    try jsonEscapeInto(&model_esc, allocator, model);

    const msg_start = try std.fmt.allocPrint(allocator,
        \\{{"type":"message_start","message":{{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"{s}","stop_reason":null,"usage":{{"input_tokens":{d},"output_tokens":0}}}}}}
    , .{ model_esc.items, input_tokens });
    defer allocator.free(msg_start);
    try sendSSE(sse, allocator, "message_start", msg_start);

    for (blocks, 0..) |b, i| {
        switch (b) {
            .text => |t| {
                const start = try std.fmt.allocPrint(allocator,
                    \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                , .{i});
                defer allocator.free(start);
                try sendSSE(sse, allocator, "content_block_start", start);

                var esc = std.ArrayList(u8).empty;
                defer esc.deinit(allocator);
                try jsonEscapeInto(&esc, allocator, t);
                const delta = try std.fmt.allocPrint(allocator,
                    \\{{"type":"content_block_delta","index":{d},"delta":{{"type":"text_delta","text":"{s}"}}}}
                , .{ i, esc.items });
                defer allocator.free(delta);
                try sendSSE(sse, allocator, "content_block_delta", delta);
            },
            .tool_use => |tu| {
                var id_esc = std.ArrayList(u8).empty;
                defer id_esc.deinit(allocator);
                try jsonEscapeInto(&id_esc, allocator, tu.id);
                var name_esc = std.ArrayList(u8).empty;
                defer name_esc.deinit(allocator);
                try jsonEscapeInto(&name_esc, allocator, tu.name);
                const start = try std.fmt.allocPrint(allocator,
                    \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"tool_use","id":"{s}","name":"{s}","input":{{}}}}}}
                , .{ i, id_esc.items, name_esc.items });
                defer allocator.free(start);
                try sendSSE(sse, allocator, "content_block_start", start);

                var args_esc = std.ArrayList(u8).empty;
                defer args_esc.deinit(allocator);
                try jsonEscapeInto(&args_esc, allocator, tu.input_json);
                const delta = try std.fmt.allocPrint(allocator,
                    \\{{"type":"content_block_delta","index":{d},"delta":{{"type":"input_json_delta","partial_json":"{s}"}}}}
                , .{ i, args_esc.items });
                defer allocator.free(delta);
                try sendSSE(sse, allocator, "content_block_delta", delta);
            },
        }
        const stop = try std.fmt.allocPrint(allocator,
            \\{{"type":"content_block_stop","index":{d}}}
        , .{i});
        defer allocator.free(stop);
        try sendSSE(sse, allocator, "content_block_stop", stop);
    }

    const msg_delta = try std.fmt.allocPrint(allocator,
        \\{{"type":"message_delta","delta":{{"stop_reason":"{s}"}},"usage":{{"output_tokens":{d}}}}}
    , .{ stop_reason, output_tokens });
    defer allocator.free(msg_delta);
    try sendSSE(sse, allocator, "message_delta", msg_delta);
    try sendSSE(sse, allocator, "message_stop", "{\"type\":\"message_stop\"}");
}

// ------------------------------------------------------------------
// Request handler
// ------------------------------------------------------------------

pub fn handleAnthropicMessages(
    allocator: std.mem.Allocator,
    io: std.Io,
    state: *ServerState,
    connection: std.Io.net.Stream,
    server_config: ServerConfig,
    body: []const u8,
    status_out: *u16,
) !void {
    // Request-scoped scratch space for composed message text and the tool
    // system prompt; everything built here dies when the handler returns.
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const arena = arena_inst.allocator();

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        status_out.* = 400;
        try writeJsonResponse(connection, io, 400, "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"invalid_json\"}}");
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        status_out.* = 400;
        try writeJsonResponse(connection, io, 400, "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"invalid_json\"}}");
        return;
    }
    const root_obj = parsed.value.object;

    const reject = struct {
        fn send(conn: std.Io.net.Stream, io2: std.Io, status: *u16, a: std.mem.Allocator, msg: []const u8) !void {
            const err_body = try std.fmt.allocPrint(a,
                \\{{"type":"error","error":{{"type":"invalid_request_error","message":"{s}"}}}}
            , .{msg});
            defer a.free(err_body);
            status.* = 400;
            try writeJsonResponse(conn, io2, 400, err_body);
        }
    }.send;

    const model: []const u8 = if (root_obj.get("model")) |m| switch (m) {
        .string => |s| s,
        else => "default",
    } else "default";

    var max_tokens: u32 = 1024;
    if (root_obj.get("max_tokens")) |mt| {
        switch (mt) {
            .integer => |v| max_tokens = @intCast(@max(1, v)),
            else => {},
        }
    }
    const temperature: ?f32 = if (root_obj.get("temperature")) |t| switch (t) {
        .float => |f| @floatCast(f),
        .integer => |v| @floatFromInt(v),
        else => null,
    } else null;
    const top_p: ?f32 = if (root_obj.get("top_p")) |t| switch (t) {
        .float => |f| @floatCast(f),
        .integer => |v| @floatFromInt(v),
        else => null,
    } else null;
    const top_k: ?u32 = if (root_obj.get("top_k")) |t| switch (t) {
        .integer => |v| @intCast(v),
        else => null,
    } else null;
    const want_stream = if (root_obj.get("stream")) |s| switch (s) {
        .bool => |b| b,
        else => false,
    } else false;

    // Validate sampling overrides and tool_choice support before doing work.
    {
        var msg: ?[]const u8 = null;
        if (top_p) |p| {
            if (!(p > 0.0 and p <= 1.0)) msg = "top_p must be in the range (0, 1]";
        }
        if (top_k) |k| {
            if (k == 0) msg = "top_k must be >= 1";
        }
        if (root_obj.get("tool_choice")) |tcv| {
            if (tcv == .object) {
                if (tcv.object.get("type")) |tv| {
                    if (tv == .string) {
                        msg = unsupportedToolChoiceMsg(.{ .type = tv.string }) orelse msg;
                    }
                }
            }
        }
        if (msg) |m| {
            try reject(connection, io, status_out, allocator, m);
            return;
        }
    }
    const tools_none = blk: {
        if (root_obj.get("tool_choice")) |tcv| {
            if (tcv == .object) {
                if (tcv.object.get("type")) |tv| {
                    if (tv == .string and std.mem.eql(u8, tv.string, "none")) break :blk true;
                }
            }
        }
        break :blk false;
    };

    const family = tool_calling_mod.detectFamily(model);

    // Map Anthropic messages to internal format. Content may be a string or
    // an array of text/tool_use/tool_result blocks.
    var messages = std.ArrayList(root.tokenizer.ChatMessage).empty;
    defer messages.deinit(allocator);

    if (root_obj.get("system")) |sys| {
        var sys_buf = std.ArrayList(u8).empty;
        try flattenContent(&sys_buf, arena, sys, family);
        if (sys_buf.items.len > 0) {
            try messages.append(allocator, .{ .role = "system", .content = sys_buf.items });
        }
    }

    if (root_obj.get("messages")) |msgs| {
        if (msgs == .array) {
            for (msgs.array.items) |mv| {
                if (mv != .object) continue;
                const role = if (mv.object.get("role")) |r| switch (r) {
                    .string => |s| s,
                    else => "user",
                } else "user";
                var buf = std.ArrayList(u8).empty;
                if (mv.object.get("content")) |cv| {
                    try flattenContent(&buf, arena, cv, family);
                }
                try messages.append(allocator, .{ .role = role, .content = buf.items });
            }
        }
    }

    // Tools: Anthropic shape {name, description, input_schema}. Injected as a
    // system prompt, same mechanism as the OpenAI endpoint.
    if (!tools_none) {
        if (root_obj.get("tools")) |tv| {
            if (tv == .array and tv.array.items.len > 0) {
                var tool_defs = std.ArrayList(tool_calling_mod.ToolDefinition).empty;
                for (tv.array.items) |t| {
                    if (t != .object) continue;
                    const name_v = t.object.get("name") orelse continue;
                    if (name_v != .string) continue;
                    const desc: ?[]const u8 = if (t.object.get("description")) |d| switch (d) {
                        .string => |s| s,
                        else => null,
                    } else null;
                    const schema: ?[]const u8 = if (t.object.get("input_schema")) |sc|
                        try std.json.Stringify.valueAlloc(arena, sc, .{})
                    else
                        null;
                    try tool_defs.append(arena, .{
                        .name = try arena.dupe(u8, name_v.string),
                        .description = if (desc) |d| try arena.dupe(u8, d) else null,
                        .parameters = schema,
                    });
                }
                if (tool_defs.items.len > 0) {
                    const sys_prompt = try tool_calling_mod.buildToolSystemPrompt(arena, tool_defs.items, family);
                    try messages.insert(allocator, 0, .{ .role = "system", .content = sys_prompt });
                }
            }
        }
    }

    // Apply chat template
    const prompt_text = try state.chat_template.apply(messages.items, true);
    defer allocator.free(prompt_text);

    // Tokenize
    const prompt_tokens = try state.tokenizer_strategy.encode(prompt_text, false, allocator);
    defer allocator.free(prompt_tokens);

    // Generate
    const capped_max = @min(max_tokens, @as(u32, @intCast(server_config.max_tokens)));
    const seed = @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toMilliseconds()));

    // Submit to engine queue.
    const request_config = engine.RequestConfig{
        .prompt_tokens = prompt_tokens,
        .max_tokens = capped_max,
        .temperature = temperature orelse server_config.temperature,
        .top_k = if (top_k) |k| @as(usize, k) else server_config.top_k,
        .top_p = top_p orelse server_config.top_p,
        .seed = seed,
        .stop_tokens = state.stop_tokens,
        .streaming = false,
        .model_name = model,
        .api_format = .anthropic,
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

    // Wait for completion
    var all_text = std.ArrayList(u8).empty;
    defer all_text.deinit(allocator);

    // Anthropic stop_reason mapping: engine .stop → "end_turn", .length → "max_tokens".
    var stop_reason: []const u8 = "end_turn";
    while (true) {
        const event = req_state.completion.waitForToken(io) catch break;
        if (event) |e| {
            defer if (e.token_text.len > 0) allocator.free(e.token_text);
            if (e.token_text.len > 0) {
                try all_text.appendSlice(allocator, e.token_text);
            }
            if (e.is_final) {
                if (e.finish_reason) |fr| {
                    stop_reason = switch (fr) {
                        .stop => "end_turn",
                        .length => "max_tokens",
                        .error_ => "end_turn",
                    };
                }
                break;
            }
        } else {
            break;
        }
    }

    // Engine may still be finishing after the final event; wait so the
    // token_count reads below and the deferred deinit are race-free.
    req_state.waitEngineDone(io);

    // Tool-call detection on the completed output.
    var blocks = std.ArrayList(ResponseBlock).empty;
    defer blocks.deinit(allocator);
    if (try tool_calling_mod.parse(allocator, all_text.items, family)) |pr| {
        defer pr.deinit();
        if (pr.calls.len > 0) {
            for (pr.calls) |call| {
                // pr owns these strings and is deinit'd at scope end — copy
                // into the request arena so the blocks outlive the parse result.
                try blocks.append(allocator, .{ .tool_use = .{
                    .id = try arena.dupe(u8, call.id),
                    .name = try arena.dupe(u8, call.function.name),
                    .input_json = try arena.dupe(u8, call.function.arguments),
                } });
            }
            stop_reason = "tool_use";
        }
    }
    if (blocks.items.len == 0) {
        try blocks.append(allocator, .{ .text = all_text.items });
    }

    metrics_mod.global.addTokenCounts(@intCast(prompt_tokens.len), @intCast(req_state.token_count));

    if (want_stream) {
        status_out.* = 200;
        try sse_mod.writeSSEHeaders(connection, io);
        var sse = sse_mod.SSEWriter{ .connection = connection, .io = io };
        try streamBlocks(allocator, &sse, model, blocks.items, stop_reason, prompt_tokens.len, req_state.token_count);
        return;
    }

    const response = try formatAnthropicResponse(
        allocator,
        model,
        blocks.items,
        prompt_tokens.len,
        req_state.token_count,
        stop_reason,
    );
    defer allocator.free(response);

    status_out.* = 200;
    try writeJsonResponse(connection, io, 200, response);
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

test "unsupportedToolChoiceMsg accepts auto/none, rejects any/tool" {
    try std.testing.expect(unsupportedToolChoiceMsg(.{ .type = "auto" }) == null);
    try std.testing.expect(unsupportedToolChoiceMsg(.{ .type = "none" }) == null);
    try std.testing.expect(unsupportedToolChoiceMsg(.{ .type = "any" }) != null);
    try std.testing.expect(unsupportedToolChoiceMsg(.{ .type = "tool", .name = "f" }) != null);
    try std.testing.expect(unsupportedToolChoiceMsg(.{ .type = "bogus" }) != null);
}

test "AnthropicRequest parses top_p/top_k/tool_choice" {
    const allocator = std.testing.allocator;
    const body =
        \\{"model":"m","max_tokens":8,"top_p":0.5,"top_k":5,"tool_choice":{"type":"tool","name":"f"},"messages":[{"role":"user","content":"hi"}]}
    ;
    const parsed = try std.json.parseFromSlice(AnthropicRequest, allocator, body, .{});
    defer parsed.deinit();
    const req = parsed.value;
    try std.testing.expectEqual(@as(f32, 0.5), req.top_p.?);
    try std.testing.expectEqual(@as(u32, 5), req.top_k.?);
    const tc = req.tool_choice.?;
    try std.testing.expectEqualStrings("tool", tc.type);
    try std.testing.expectEqualStrings("f", tc.name.?);
    try std.testing.expect(unsupportedToolChoiceMsg(tc) != null);
}

test "formatAnthropicResponse emits tool_use blocks with object input" {
    const allocator = std.testing.allocator;
    const blocks = [_]ResponseBlock{
        .{ .text = "Let me check." },
        .{ .tool_use = .{ .id = "toolu_1", .name = "get_weather", .input_json = "{\"city\":\"Paris\"}" } },
    };
    const json = try formatAnthropicResponse(allocator, "m", &blocks, 10, 5, "tool_use");
    defer allocator.free(json);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expectEqualStrings("tool_use", obj.get("stop_reason").?.string);
    const content = obj.get("content").?.array;
    try std.testing.expectEqual(@as(usize, 2), content.items.len);
    try std.testing.expectEqualStrings("text", content.items[0].object.get("type").?.string);
    const tu = content.items[1].object;
    try std.testing.expectEqualStrings("tool_use", tu.get("type").?.string);
    try std.testing.expectEqualStrings("get_weather", tu.get("name").?.string);
    // input must be an embedded JSON object, not an escaped string
    try std.testing.expect(tu.get("input").? == .object);
    try std.testing.expectEqualStrings("Paris", tu.get("input").?.object.get("city").?.string);
}

test "flattenContent renders block arrays and tool history" {
    const allocator = std.testing.allocator;
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const body =
        \\[{"type":"text","text":"weather?"},{"type":"tool_use","id":"t1","name":"get_weather","input":{"city":"Paris"}},{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"sunny"}]}]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    var out = std.ArrayList(u8).empty;
    defer out.deinit(arena_inst.allocator());
    try flattenContent(&out, arena_inst.allocator(), parsed.value, .qwen);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "weather?") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "<tool_response>\nsunny\n</tool_response>") != null);
}
