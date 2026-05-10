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

const ServerConfig = config_mod.ServerConfig;
const ServerState = state_mod.ServerState;

const AnthropicRequest = struct {
    model: []const u8,
    messages: []AnthropicMessage,
    max_tokens: u32 = 1024,
    temperature: ?f32 = null,
    stream: ?bool = false,
    system: ?[]const u8 = null,

    pub const AnthropicMessage = struct {
        role: []const u8,
        content: []const u8,
    };
};

/// Format an Anthropic Messages API response.
fn formatAnthropicResponse(
    allocator: std.mem.Allocator,
    model: []const u8,
    text: []const u8,
    input_tokens: usize,
    output_tokens: usize,
) ![]u8 {
    var escaped = std.ArrayList(u8).empty;
    defer escaped.deinit(allocator);
    try jsonEscapeInto(&escaped, allocator, text);

    return try std.fmt.allocPrint(allocator,
        \\{{"id":"msg_1","type":"message","role":"assistant","content":[{{"type":"text","text":"{s}"}}],"model":"{s}","stop_reason":"end_turn","usage":{{"input_tokens":{d},"output_tokens":{d}}}}}
    , .{ escaped.items, model, input_tokens, output_tokens });
}

// ------------------------------------------------------------------
// OpenAI request parsing & response generation (using Generation API)
// ------------------------------------------------------------------

pub fn handleAnthropicMessages(
    allocator: std.mem.Allocator,
    io: std.Io,
    state: *ServerState,
    connection: std.Io.net.Stream,
    server_config: ServerConfig,
    body: []const u8,
) !void {
    const parsed = std.json.parseFromSlice(AnthropicRequest, allocator, body, .{}) catch {
        try writeJsonResponse(connection, io, 400, "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"invalid_json\"}}");
        return;
    };
    defer parsed.deinit();
    const req = parsed.value;

    // Map Anthropic messages to internal format
    var messages = std.ArrayList(root.tokenizer.ChatMessage).empty;
    defer messages.deinit(allocator);

    // Add system message if provided
    if (req.system) |sys| {
        try messages.append(allocator, .{ .role = "system", .content = sys });
    }

    for (req.messages) |msg| {
        try messages.append(allocator, .{ .role = msg.role, .content = msg.content });
    }

    // Apply chat template
    const prompt_text = try state.chat_template.apply(messages.items, true);
    defer allocator.free(prompt_text);

    // Tokenize
    const prompt_tokens = try state.tokenizer_strategy.encode(prompt_text, false, allocator);
    defer allocator.free(prompt_tokens);

    // Generate
    const max_tokens = @min(req.max_tokens, @as(u32, @intCast(server_config.max_tokens)));
    const temperature = req.temperature orelse server_config.temperature;
    const seed = @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toMilliseconds()));

    // Submit to engine queue.
    const request_config = engine.RequestConfig{
        .prompt_tokens = prompt_tokens,
        .max_tokens = max_tokens,
        .temperature = temperature,
        .top_k = server_config.top_k,
        .top_p = server_config.top_p,
        .seed = seed,
        .streaming = false,
        .model_name = req.model,
        .api_format = .anthropic,
    };

    const request_id = state.next_request_id.fetchAdd(1, .monotonic);
    const req_state = try engine.RequestState.init(allocator, request_id, request_config);
    defer req_state.deinit();

    _ = state.active_requests.fetchAdd(1, .monotonic);
    defer _ = state.active_requests.fetchSub(1, .monotonic);

    var node = engine.QueueNode.init(req_state);
    state.request_queue.push(&node);

    // Wait for completion
    var all_text = std.ArrayList(u8).empty;
    defer all_text.deinit(allocator);

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

    // Format Anthropic response
    const response = try formatAnthropicResponse(
        allocator,
        req.model,
        all_text.items,
        prompt_tokens.len,
        req_state.token_count,
    );
    defer allocator.free(response);

    try writeJsonResponse(connection, io, 200, response);
}
