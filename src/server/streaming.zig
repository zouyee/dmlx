/// Streaming completion handler.
const std = @import("std");
const root = @import("../root.zig");
const c = @import("mlx").c;
const generation_mod = @import("../generation.zig");
const kvcache = @import("../kvcache.zig");
const engine = @import("../engine/root.zig");
const config_mod = @import("config.zig");
const utils_mod = @import("utils.zig");
const state_mod = @import("state.zig");
const openai_mod = @import("openai.zig");
const sse_mod = @import("sse.zig");
const ChatCompletionRequest = openai_mod.ChatCompletionRequest;
const SSEWriter = sse_mod.SSEWriter;
const writeSSEHeaders = sse_mod.writeSSEHeaders;
const writeSSEEvent = sse_mod.writeSSEEvent;
const formatSSEChunk = sse_mod.formatSSEChunk;

const ServerConfig = config_mod.ServerConfig;
const ServerState = state_mod.ServerState;

pub fn handleStreamingCompletion(
    allocator: std.mem.Allocator,
    io: std.Io,
    state: *ServerState,
    connection: std.Io.net.Stream,
    server_config: ServerConfig,
    req: ChatCompletionRequest,
) !void {
    // 1. Send SSE response headers
    try writeSSEHeaders(connection, io);

    var sse = SSEWriter{ .connection = connection, .io = io };

    const created = @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toSeconds()));
    const completion_id = created; // Use timestamp as unique ID

    // 2. Send initial chunk with role. NOTE: this goes out BEFORE prompt
    // validation (step 3-5) — SSE headers/role are already committed once
    // the stream opens, so a later validation failure is reported as an
    // error event mid-stream (followed by [DONE]) rather than an HTTP error
    // status. That ordering is by design for SSE.
    const role_chunk = try formatSSEChunk(allocator, completion_id, created, req.model, null, null);
    defer allocator.free(role_chunk);
    try sse.sendEvent(role_chunk);

    // 3-5. Build the prompt (text + optional image content parts, chat
    // template, tokenize). Shared with the non-streaming handler.
    var prepared = openai_mod.preparePrompt(allocator, state, req, null) catch |err| {
        const msg: []const u8 = switch (err) {
            error.VisionNotSupported => "vision input is not supported by the loaded model",
            error.ImageTooLarge => "decoded image exceeds the 20MB limit",
            error.BadImageUrl => "unsupported image_url",
            error.InvalidContent => "invalid message content",
            else => "prompt preparation failed",
        };
        const chunk = try std.fmt.allocPrint(allocator,
            \\{{"error":{{"message":"{s}","type":"invalid_request_error"}}}}
        , .{msg});
        defer allocator.free(chunk);
        try sse.sendEvent(chunk);
        try sse.sendEvent("[DONE]");
        return;
    };
    defer prepared.deinit(allocator);
    const prompt_tokens = prepared.tokens;

    // 6. Setup generation config and submit to engine queue.
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
        .streaming = true,
        .model_name = req.model,
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

    var node = engine.QueueNode.init(req_state);
    state.request_queue.push(&node);

    // 7. Stream tokens from CompletionSignal to SSE.
    var token_count: usize = 0;
    _ = io.async(keepAliveLoop, .{ io, &sse, &token_count });

    while (true) {
        const event = req_state.completion.waitForToken(io) catch break;
        if (event) |e| {
            defer if (e.token_text.len > 0) allocator.free(e.token_text);
            token_count += 1;

            const finish: ?[]const u8 = if (e.is_final)
                if (e.finish_reason) |fr|
                    switch (fr) {
                        .stop => "stop",
                        .length => "length",
                        .error_ => "stop",
                    }
                else
                    "stop"
            else
                null;
            const chunk = try formatSSEChunk(allocator, completion_id, created, req.model, e.token_text, finish);
            defer allocator.free(chunk);
            try sse.sendEvent(chunk);

            if (e.is_final) {
                try sse.sendEvent("[DONE]");
                break;
            }
        } else {
            break;
        }
    }
}

/// SSE keep-alive fiber: sends a comment every 5s until the first token
/// is generated (indicating prefill is complete).
fn keepAliveLoop(io_arg: std.Io, sse_arg: *SSEWriter, token_count: *usize) void {
    while (token_count.* == 0) {
        io_arg.sleep(.fromMilliseconds(5000), .awake) catch break;
        if (token_count.* > 0) break;
        sse_arg.sendKeepAlive() catch break;
    }
}

// ------------------------------------------------------------------
// Anthropic Messages API handler
// ------------------------------------------------------------------
