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
        content: []const u8,
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

pub fn generateChatCompletion(allocator: std.mem.Allocator, io: std.Io, state: *ServerState, request_json: []const u8, server_config: ServerConfig) ![]u8 {
    // 1. Parse request
    const parsed = try std.json.parseFromSlice(ChatCompletionRequest, allocator, request_json, .{});
    defer parsed.deinit();
    const req = parsed.value;

    // 2. Streaming is handled by handleStreamingCompletion; reject here as safety net
    if (req.stream orelse false) {
        return try allocator.dupe(u8, "{\"error\":\"streaming_not_supported\"}");
    }

    // 3. Build messages for chat template
    var messages = std.ArrayList(root.tokenizer.ChatMessage).empty;
    defer messages.deinit(allocator);

    // 3a. If tools are provided, inject tool system prompt as the first message
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
            const tool_prompt = try tool_calling_mod.buildToolSystemPrompt(allocator, tool_defs.items, family);
            defer allocator.free(tool_prompt);
            try messages.append(allocator, .{
                .role = "system",
                .content = tool_prompt,
            });
        }
    }

    for (req.messages) |msg| {
        try messages.append(allocator, .{
            .role = msg.role,
            .content = msg.content,
        });
    }

    // 4. Apply chat template
    const t_template_start = std.c.mach_absolute_time();
    const prompt_text = try state.chat_template.apply(messages.items, true);
    defer allocator.free(prompt_text);
    const t_template_end = std.c.mach_absolute_time();

    // 5. Tokenize (template already includes special tokens)
    const prompt_tokens = try state.tokenizer_strategy.encode(prompt_text, false, allocator);
    defer allocator.free(prompt_tokens);
    const t_tokenize_end = std.c.mach_absolute_time();

    std.log.info("[HTTP] Template={d}ms Tokenize={d}ms prompt_len={d}", .{
        (t_template_end - t_template_start) / 1_000_000,
        (t_tokenize_end - t_template_end) / 1_000_000,
        prompt_tokens.len,
    });

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
    };

    const request_id = state.next_request_id.fetchAdd(1, .monotonic);
    const req_state = try engine.RequestState.init(allocator, request_id, request_config);
    defer req_state.deinit();

    _ = state.active_requests.fetchAdd(1, .monotonic);
    defer _ = state.active_requests.fetchSub(1, .monotonic);

    const t_push_start = std.c.mach_absolute_time();
    var node = engine.QueueNode.init(req_state);
    state.request_queue.push(&node);
    std.log.info("[HTTP] Request {d} submitted to queue", .{request_id});

    // Wait for the engine to deliver the complete response.
    var all_text = std.ArrayList(u8).empty;
    defer all_text.deinit(allocator);

    const t_wait_start = std.c.mach_absolute_time();
    while (true) {
        const event = req_state.completion.waitForToken(io) catch break;
        if (event) |e| {
            defer if (e.token_text.len > 0) allocator.free(e.token_text);
            if (e.token_text.len > 0) {
                try all_text.appendSlice(allocator, e.token_text);
            }
            if (e.is_final) break;
        } else {
            break;
        }
    }
    const t_wait_end = std.c.mach_absolute_time();
    std.log.info("[HTTP] Timing: push→wait_start={d}ms, wait_duration={d}ms", .{
        (t_wait_start - t_push_start) / 1_000_000,
        (t_wait_end - t_wait_start) / 1_000_000,
    });

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
        \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
    , .{
        created,
        created,
        req.model,
        escaped.items,
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
