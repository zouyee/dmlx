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
    /// Since this endpoint does not implement tool execution at all, "auto"
    /// and "none" are accepted (both behave as none); "any"/"tool" would
    /// require forced-call machinery we don't have and are rejected.
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

/// Anthropic tool_choice support policy, mirroring the OpenAI one in
/// openai.zig: "auto"/"none" are accepted (this endpoint executes no tools,
/// so both behave as none); "any"/"tool" would force a call and are
/// rejected with the message returned here.
pub fn unsupportedToolChoiceMsg(tc: AnthropicRequest.AnthropicToolChoice) ?[]const u8 {
    if (std.mem.eql(u8, tc.type, "auto") or std.mem.eql(u8, tc.type, "none")) return null;
    return "tool_choice 'any'/'tool' (forced calls) are not supported by this server (use 'auto' or 'none')";
}

/// Format an Anthropic Messages API response.
fn formatAnthropicResponse(
    allocator: std.mem.Allocator,
    model: []const u8,
    text: []const u8,
    input_tokens: usize,
    output_tokens: usize,
    stop_reason: []const u8,
) ![]u8 {
    var escaped = std.ArrayList(u8).empty;
    defer escaped.deinit(allocator);
    try jsonEscapeInto(&escaped, allocator, text);

    return try std.fmt.allocPrint(allocator,
        \\{{"id":"msg_1","type":"message","role":"assistant","content":[{{"type":"text","text":"{s}"}}],"model":"{s}","stop_reason":"{s}","usage":{{"input_tokens":{d},"output_tokens":{d}}}}}
    , .{ escaped.items, model, stop_reason, input_tokens, output_tokens });
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
    status_out: *u16,
) !void {
    const parsed = std.json.parseFromSlice(AnthropicRequest, allocator, body, .{}) catch {
        status_out.* = 400;
        try writeJsonResponse(connection, io, 400, "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"invalid_json\"}}");
        return;
    };
    defer parsed.deinit();
    const req = parsed.value;

    // Validate sampling overrides and tool_choice support before doing work.
    {
        var reject: ?[]const u8 = null;
        if (req.top_p) |p| {
            if (!(p > 0.0 and p <= 1.0)) reject = "top_p must be in the range (0, 1]";
        }
        if (req.top_k) |k| {
            if (k == 0) reject = "top_k must be >= 1";
        }
        if (req.tool_choice) |tc| {
            reject = unsupportedToolChoiceMsg(tc) orelse reject;
        }
        if (reject) |msg| {
            const err_body = try std.fmt.allocPrint(allocator,
                \\{{"type":"error","error":{{"type":"invalid_request_error","message":"{s}"}}}}
            , .{msg});
            defer allocator.free(err_body);
            status_out.* = 400;
            try writeJsonResponse(connection, io, 400, err_body);
            return;
        }
    }

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
        .top_k = if (req.top_k) |k| @as(usize, k) else server_config.top_k,
        .top_p = req.top_p orelse server_config.top_p,
        .seed = seed,
        .stop_tokens = state.stop_tokens,
        .streaming = false,
        .model_name = req.model,
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

    // Format Anthropic response
    const response = try formatAnthropicResponse(
        allocator,
        req.model,
        all_text.items,
        prompt_tokens.len,
        req_state.token_count,
        stop_reason,
    );
    defer allocator.free(response);

    metrics_mod.global.addTokenCounts(@intCast(prompt_tokens.len), @intCast(req_state.token_count));
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
