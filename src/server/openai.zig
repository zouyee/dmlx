/// OpenAI-compatible chat completion handler.
const std = @import("std");
const root = @import("../root.zig");
const c = @import("mlx").c;
const generation_mod = @import("../generation.zig");
const kvcache = @import("../kvcache.zig");
const tool_calling_mod = @import("../tool_calling.zig");
const tool_executor_mod = @import("../tool_executor.zig");
const engine = @import("../engine/root.zig");
const config_mod = @import("config.zig");
const utils_mod = @import("utils.zig");
const state_mod = @import("state.zig");
const vision_prompt = @import("../vision_prompt.zig");
const jsonEscapeInto = utils_mod.jsonEscapeInto;

const ServerConfig = config_mod.ServerConfig;
const ServerState = state_mod.ServerState;

pub const ChatCompletionRequest = struct {
    model: []const u8,
    messages: []Message,
    stream: ?bool = false,
    temperature: ?f32 = null,
    max_tokens: ?u32 = null,
    /// Optional seed for deterministic sampling.
    seed: ?u64 = null,
    /// Optional stop strings — generation stops when any of these are produced.
    stop: ?[]const []const u8 = null,
    /// Optional response format constraint for guided decoding.
    /// When `type` is "json_schema", the `schema` field contains a JSON schema string.
    /// When `type` is "regex", the `schema` field contains a regex pattern.
    /// Currently parsed but not enforced — see TODO comments in generation functions.
    response_format: ?ResponseFormat = null,
    /// Optional tool definitions for function calling.
    /// When provided, the model output is scanned for tool call patterns.
    tools: ?[]const ToolDefinition = null,

    pub const Message = struct {
        role: []const u8,
        /// String content, or an array of content parts
        /// ({"type":"text","text":...} / {"type":"image_url","image_url":{"url":...}}).
        content: std.json.Value,
    };

    pub const ResponseFormat = struct {
        type: []const u8, // "json_schema" or "regex"
        schema: ?[]const u8 = null, // JSON schema string or regex pattern
    };

    pub const ToolDefinition = struct {
        type: []const u8 = "function",
        function: ?ToolFunction = null,

        pub const ToolFunction = struct {
            name: []const u8,
            description: ?[]const u8 = null,
            parameters: ?std.json.Value = null,
        };
    };
};

/// Decoded image payload cap (20 MB) — plan risk register.
pub const max_decoded_image_bytes: usize = 20 * 1024 * 1024;

/// Extract the text of one message and collect any image inputs. Content
/// may be a plain string or an array of text/image_url parts.
fn extractMessageContent(
    allocator: std.mem.Allocator,
    content: std.json.Value,
    images: *std.ArrayList(generation_mod.VisionImage),
    decoded: *std.ArrayList([]u8),
) ![]const u8 {
    switch (content) {
        .string => |s| return allocator.dupe(u8, s),
        .array => |parts| {
            var text = std.ArrayList(u8).empty;
            errdefer text.deinit(allocator);
            for (parts.items) |part| {
                const obj = switch (part) {
                    .object => |o| o,
                    else => return error.InvalidContent,
                };
                const ty = switch (obj.get("type") orelse return error.InvalidContent) {
                    .string => |s| s,
                    else => return error.InvalidContent,
                };
                if (std.mem.eql(u8, ty, "text")) {
                    switch (obj.get("text") orelse return error.InvalidContent) {
                        .string => |s| try text.appendSlice(allocator, s),
                        else => return error.InvalidContent,
                    }
                } else if (std.mem.eql(u8, ty, "image_url")) {
                    const url = blk: {
                        const iu = obj.get("image_url") orelse return error.InvalidContent;
                        switch (iu) {
                            .string => |s| break :blk s,
                            .object => |io| break :blk switch (io.get("url") orelse return error.InvalidContent) {
                                .string => |s| s,
                                else => return error.InvalidContent,
                            },
                            else => return error.InvalidContent,
                        }
                    };
                    try loadImageRef(allocator, url, images, decoded);
                } else {
                    return error.InvalidContent;
                }
            }
            return text.toOwnedSlice(allocator);
        },
        else => return error.InvalidContent,
    }
}

/// Resolve one image_url: data URLs are base64-decoded inline (20 MB
/// decoded cap; note the 16 MB HTTP body cap in http.zig makes this
/// effectively ~12 MB of decoded image per request, since the base64 text
/// inflates the payload by ~4/3); absolute file paths are passed through
/// for local dev. Remote URLs are not fetched in v1.
fn loadImageRef(
    allocator: std.mem.Allocator,
    url: []const u8,
    images: *std.ArrayList(generation_mod.VisionImage),
    decoded: *std.ArrayList([]u8),
) !void {
    if (std.mem.startsWith(u8, url, "data:")) {
        const comma = std.mem.indexOfScalar(u8, url, ',') orelse return error.BadImageUrl;
        const meta = url[5..comma];
        if (!std.mem.endsWith(u8, meta, ";base64")) return error.BadImageUrl;
        const b64 = std.mem.trim(u8, url[comma + 1 ..], " \t\r\n");
        const size = std.base64.standard.Decoder.calcSizeForSlice(b64) catch |e| {
            std.log.warn("image_url base64 size failed ({s}), b64 len={d} tail={s}", .{ @errorName(e), b64.len, b64[@max(b64.len, 8) - 8 ..] });
            return error.BadImageUrl;
        };
        if (size > max_decoded_image_bytes) return error.ImageTooLarge;
        const buf = try allocator.alloc(u8, size);
        errdefer allocator.free(buf);
        std.base64.standard.Decoder.decode(buf, b64) catch return error.BadImageUrl;
        try decoded.append(allocator, buf);
        try images.append(allocator, .{ .bytes = buf });
        return;
    }
    if (std.mem.startsWith(u8, url, "/")) {
        try images.append(allocator, .{ .path = url });
        return;
    }
    return error.BadImageUrl;
}

/// Prompt ready for the engine: token ids + (for image requests) the
/// tower features/blocks the vision prefill consumes.
pub const PreparedPrompt = struct {
    tokens: []u32, // owned
    vision: ?generation_mod.VisionData, // owned when set

    pub fn deinit(self: *PreparedPrompt, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
        if (self.vision) |*v| v.deinit(allocator);
    }
};

/// Build the chat-template prompt from OpenAI messages, handling both
/// string content and array content parts. Image parts run through the
/// model's VisionOps (preprocess + ViT tower) inline.
pub fn preparePrompt(
    allocator: std.mem.Allocator,
    state: *ServerState,
    req: ChatCompletionRequest,
    extra_system: ?[]const u8,
) !PreparedPrompt {
    var images = std.ArrayList(generation_mod.VisionImage).empty;
    defer images.deinit(allocator);
    var decoded = std.ArrayList([]u8).empty;
    defer {
        for (decoded.items) |b| allocator.free(b);
        decoded.deinit(allocator);
    }

    const texts = try allocator.alloc([]const u8, req.messages.len);
    var n_texts: usize = 0;
    defer {
        for (texts[0..n_texts]) |t| allocator.free(t);
        allocator.free(texts);
    }
    var last_user_idx: ?usize = null;
    for (req.messages, 0..) |msg, i| {
        if (std.mem.eql(u8, msg.role, "user")) last_user_idx = i;
        texts[i] = try extractMessageContent(allocator, msg.content, &images, &decoded);
        n_texts += 1;
    }

    if (images.items.len == 0) {
        const tokens = try tokenizeMessages(allocator, state, req.messages, texts, extra_system, null);
        return .{ .tokens = tokens, .vision = null };
    }

    const vops = state.vtable.vision orelse return error.VisionNotSupported;
    const user_idx = last_user_idx orelse return error.InvalidContent;
    var proc = try vision_prompt.processImages(allocator, state.ctx, vops, state.vtable.ptr, images.items);
    defer allocator.free(proc.blocks_text);
    errdefer proc.data.deinit(allocator);

    const tokens = try tokenizeMessages(allocator, state, req.messages, texts, extra_system, .{
        .user_idx = user_idx,
        .blocks_text = proc.blocks_text,
    });
    errdefer allocator.free(tokens);
    try vision_prompt.validatePadCount(tokens, vops.image_token_id, proc.n_pads);
    return .{ .tokens = tokens, .vision = proc.data };
}

const VisionInsert = struct {
    user_idx: usize,
    blocks_text: []const u8,
};

/// Apply the chat template to the extracted message texts (optionally with
/// the vision blocks prepended to one user message) and tokenize.
fn tokenizeMessages(
    allocator: std.mem.Allocator,
    state: *ServerState,
    req_messages: []const ChatCompletionRequest.Message,
    texts: []const []const u8,
    extra_system: ?[]const u8,
    vision: ?VisionInsert,
) ![]u32 {
    var messages = std.ArrayList(root.tokenizer.ChatMessage).empty;
    defer messages.deinit(allocator);
    var merged: ?[]u8 = null;
    defer if (merged) |m| allocator.free(m);
    if (extra_system) |sys| {
        try messages.append(allocator, .{ .role = "system", .content = sys });
    }
    for (texts, 0..) |t, i| {
        var content = t;
        if (vision) |v| {
            if (i == v.user_idx) {
                merged = try std.fmt.allocPrint(allocator, "{s}\n{s}", .{ v.blocks_text, t });
                content = merged.?;
            }
        }
        try messages.append(allocator, .{ .role = req_messages[i].role, .content = content });
    }
    const prompt_text = try state.chat_template.apply(messages.items, true);
    defer allocator.free(prompt_text);
    return state.tokenizer_strategy.encode(prompt_text, false, allocator);
}

pub fn generateChatCompletion(allocator: std.mem.Allocator, io: std.Io, state: *ServerState, request_json: []const u8, server_config: ServerConfig) ![]u8 {
    const t_entry = std.c.mach_absolute_time();
    // 1. Parse request
    const parsed = try std.json.parseFromSlice(ChatCompletionRequest, allocator, request_json, .{});
    defer parsed.deinit();
    const req = parsed.value;

    // 2. Streaming is handled by handleStreamingCompletion; reject here as safety net
    if (req.stream orelse false) {
        return try allocator.dupe(u8, "{\"error\":\"streaming_not_supported\"}");
    }

    // 3-5. Build the prompt: extract text/images from the messages, run any
    // images through the vision tower inline, apply the chat template and
    // tokenize. Tool definitions inject a leading system prompt.
    var tool_system: ?[]u8 = null;
    defer if (tool_system) |ts| allocator.free(ts);
    if (req.tools) |tools| {
        const family = tool_calling_mod.detectFamily(req.model);
        var tool_defs = std.ArrayList(tool_calling_mod.ToolDefinition).empty;
        defer {
            for (tool_defs.items) |*td| {
                allocator.free(td.name);
                if (td.description) |d| allocator.free(d);
                if (td.parameters) |p| allocator.free(p);
            }
            tool_defs.deinit(allocator);
        }
        for (tools) |tool| {
            const fn_info = tool.function orelse continue;
            const params_json = if (fn_info.parameters) |params| blk: {
                const out = try std.json.Stringify.valueAlloc(allocator, params, .{});
                break :blk out;
            } else null;
            try tool_defs.append(allocator, .{
                .name = try allocator.dupe(u8, fn_info.name),
                .description = if (fn_info.description) |d| try allocator.dupe(u8, d) else null,
                .parameters = params_json,
            });
        }
        if (tool_defs.items.len > 0) {
            tool_system = try tool_calling_mod.buildToolSystemPrompt(allocator, tool_defs.items, family);
        }
    }

    var prepared = preparePrompt(allocator, state, req, tool_system) catch |err| {
        const msg: []const u8 = switch (err) {
            error.VisionNotSupported => "vision input is not supported by the loaded model",
            error.ImageTooLarge => "decoded image exceeds the 20MB limit",
            error.BadImageUrl => "unsupported image_url (data:...;base64 URLs and absolute file paths only)",
            error.InvalidContent => "invalid message content (string or text/image_url parts only)",
            else => return err,
        };
        return try std.fmt.allocPrint(allocator,
            \\{{"error":{{"message":"{s}","type":"invalid_request_error","code":"{s}"}}}}
        , .{ msg, @errorName(err) });
    };
    defer prepared.deinit(allocator);
    const prompt_tokens = prepared.tokens;

    // 6. Submit request to engine queue and wait for completion.
    const max_tokens = if (req.max_tokens) |mt| @min(mt, @as(u32, @intCast(server_config.max_tokens))) else server_config.max_tokens;
    const temperature = req.temperature orelse server_config.temperature;
    const seed = req.seed orelse @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toMilliseconds()));

    const request_config = engine.RequestConfig{
        .prompt_tokens = prompt_tokens,
        .max_tokens = max_tokens,
        .temperature = temperature,
        .top_k = server_config.top_k,
        .top_p = server_config.top_p,
        .seed = seed,
        .stop_strings = req.stop,
        .stop_tokens = state.stop_tokens,
        .streaming = false,
        .model_name = req.model,
        .guided_json_schema = if (req.response_format) |rf| blk: {
            if (std.mem.eql(u8, rf.type, "json_schema")) break :blk rf.schema;
            break :blk null;
        } else null,
        .guided_regex = if (req.response_format) |rf| blk: {
            if (std.mem.eql(u8, rf.type, "regex")) break :blk rf.schema;
            break :blk null;
        } else null,
        .vision = prepared.vision,
    };

    const request_id = state.next_request_id.fetchAdd(1, .monotonic);
    // On init failure `prepared`'s defer frees the vision payload.
    const req_state = try engine.RequestState.init(allocator, request_id, request_config);
    defer req_state.deinit();
    // Vision ownership has moved into the RequestState.
    prepared.vision = null;

    _ = state.active_requests.fetchAdd(1, .monotonic);
    defer _ = state.active_requests.fetchSub(1, .monotonic);

    const t_queued = std.c.mach_absolute_time();
    req_state.queued_time_ns = @intCast(t_queued);
    const prep_ms = @as(f64, @floatFromInt(t_queued - t_entry)) * 125.0 / 3_000_000.0;
    var node = engine.QueueNode.init(req_state);
    state.request_queue.push(&node);
    std.log.info("[HTTP] Request {d} submitted to queue (prepare={d:.1}ms)", .{ request_id, prep_ms });

    // Wait for the engine to deliver the complete response.
    var all_text = std.ArrayList(u8).empty;
    defer all_text.deinit(allocator);

    var finish_reason: []const u8 = "stop";
    const t_wait_start = std.c.mach_absolute_time();
    var wait_call_count: usize = 0;
    while (true) {
        wait_call_count += 1;
        std.log.info("[OpenAI] Waiting for token...", .{});
        const event = req_state.completion.waitForToken(io) catch break;
        std.log.info("[OpenAI] waitForToken returned", .{});
        if (event) |e| {
            defer if (e.token_text.len > 0) allocator.free(e.token_text);
            if (e.token_text.len > 0) {
                try all_text.appendSlice(allocator, e.token_text);
            }
            std.log.info("[OpenAI] Got token, is_final={}", .{e.is_final});
            if (e.is_final) {
                if (e.finish_reason) |fr| {
                    finish_reason = switch (fr) {
                        .stop => "stop",
                        .length => "length",
                        .error_ => "stop", // error detail is surfaced via hasError below
                    };
                }
                break;
            }
        } else {
            std.log.info("[OpenAI] waitForToken returned null", .{});
            break;
        }
    }

    const t_wait_end = std.c.mach_absolute_time();
    const wait_ms = @as(f64, @floatFromInt(t_wait_end - t_wait_start)) * 125.0 / 3_000_000.0;
    std.log.info("[OpenAI] Token loop done, all_text.len={d}, wait={d:.1}ms calls={d}", .{ all_text.items.len, wait_ms, wait_call_count });

    // Check if the engine signaled an error.
    if (req_state.completion.hasError(io)) {
        const err_msg = req_state.completion.getError(io) orelse "Generation failed";
        return try std.fmt.allocPrint(allocator,
            \\{{"error":{{"message":"{s}","type":"server_error","code":"generation_failed"}}}}
        , .{err_msg});
    }

    const final_text = all_text.items;
    const new_tokens_len = req_state.token_count;

    // 8. Tool calling loop (single round)
    if (req.tools) |tools| {
        _ = tools;
        const family = tool_calling_mod.detectFamily(req.model);
        if (try tool_calling_mod.parse(allocator, final_text, family)) |parse_result| {
            defer parse_result.deinit();

            if (parse_result.calls.len > 0) {
                // Execute tools and build tool results
                var tool_results = std.ArrayList(u8).empty;
                defer tool_results.deinit(allocator);

                const exec_config = tool_executor_mod.ExecutorConfig{
                    .allow_shell_exec = server_config.allow_unsafe_tools,
                    .max_file_size = 1024 * 1024,
                    .io = state.io,
                };

                for (parse_result.calls) |call| {
                    const result = try tool_executor_mod.execute(allocator, exec_config, call.function.name, call.function.arguments);
                    defer result.deinit();

                    const result_text = if (result.success) result.output else result.error_message orelse "error";
                    try tool_results.appendSlice(allocator, "Tool '");
                    try tool_results.appendSlice(allocator, call.function.name);
                    try tool_results.appendSlice(allocator, "' result: ");
                    try tool_results.appendSlice(allocator, result_text);
                    try tool_results.appendSlice(allocator, "\n");
                }

                // Build OpenAI-compatible response with tool_calls
                const created = @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toSeconds()));
                var tool_calls_json = std.ArrayList(u8).empty;
                defer tool_calls_json.deinit(allocator);
                try tool_calls_json.appendSlice(allocator, "[");
                for (parse_result.calls, 0..) |call, i| {
                    if (i > 0) try tool_calls_json.appendSlice(allocator, ",");
                    try tool_calls_json.print(allocator,
                        \\{{"id":"{s}","type":"function","function":{{"name":"{s}","arguments":"{s}"}}}}
                    , .{ call.id, call.function.name, call.function.arguments });
                }
                try tool_calls_json.appendSlice(allocator, "]");

                return try std.fmt.allocPrint(allocator,
                    \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":null,"tool_calls":{s}}},"finish_reason":"tool_calls"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
                , .{
                    created,
                    created,
                    req.model,
                    tool_calls_json.items,
                    prompt_tokens.len,
                    new_tokens_len,
                    prompt_tokens.len + new_tokens_len,
                });
            }
        }
    }

    // 9. Build standard OpenAI-compatible response
    const created = @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toSeconds()));

    // JSON-escape the output text
    var escaped = std.ArrayList(u8).empty;
    defer escaped.deinit(allocator);
    try jsonEscapeInto(&escaped, allocator, final_text);

    return try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
    , .{
        created,
        created,
        req.model,
        escaped.items,
        finish_reason,
        prompt_tokens.len,
        new_tokens_len,
        prompt_tokens.len + new_tokens_len,
    });
}

// ------------------------------------------------------------------
// SSE streaming completion handler (using Generation API streamGenerate)
// ------------------------------------------------------------------

/// Keep-alive threshold in milliseconds (5 seconds).
const keep_alive_threshold_ms: i64 = 5000;
