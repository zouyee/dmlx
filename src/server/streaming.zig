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
const metrics_mod = @import("metrics.zig");
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
        .top_k = if (req.top_k) |k| @as(usize, k) else server_config.top_k,
        .top_p = req.top_p orelse server_config.top_p,
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
    // The request below is always pushed to the engine queue (no fallible
    // call in between), so waiting for the engine here cannot hang: the
    // final token event precedes the engine's post-generation bookkeeping.
    defer {
        req_state.waitEngineDone(io);
        req_state.deinit();
    }
    // Vision ownership has moved into the RequestState.
    prepared.vision = null;

    _ = state.active_requests.fetchAdd(1, .monotonic);
    defer _ = state.active_requests.fetchSub(1, .monotonic);

    var node = engine.QueueNode.init(req_state);
    state.request_queue.push(&node);

    // 7. Stream tokens from CompletionSignal to SSE.
    // The keep-alive fiber borrows `sse` and `stop_keepalive` from this
    // stack frame, so it MUST be cancelled before this function returns —
    // the defer guarantees that on every exit path (including a write error
    // from an aborted client connection). Without it the fiber can wake up
    // after the frame is gone and write through dangling pointers.
    var stop_keepalive = std.atomic.Value(bool).init(false);
    var ka_future = io.async(keepAliveLoop, .{ io, &sse, &stop_keepalive });
    var ka_running = true;
    defer {
        if (ka_running) {
            stop_keepalive.store(true, .release);
            ka_future.cancel(io);
        }
    }

    while (true) {
        const event = req_state.completion.waitForToken(io) catch {
            // Fiber cancelled mid-stream: the engine may still be
            // delivering into req_state.completion, so cancel + drain
            // before this frame unwinds (same contract as the write-error
            // path below).
            abortStream(io, req_state, allocator);
            break;
        };
        if (event) |e| {
            defer if (e.token_text.len > 0) allocator.free(e.token_text);
            if (ka_running) {
                // First token: prefill is done, keep-alives are no longer
                // needed. Join the fiber before any further writes so it
                // can never write concurrently with a token chunk.
                stop_keepalive.store(true, .release);
                ka_future.cancel(io);
                ka_running = false;
            }

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
            const chunk = formatSSEChunk(allocator, completion_id, created, req.model, e.token_text, finish) catch |err| {
                // OOM with the engine possibly mid-generation: same
                // cancel + drain contract before unwinding req_state.
                abortStream(io, req_state, allocator);
                return err;
            };
            defer allocator.free(chunk);
            sse.sendEvent(chunk) catch {
                // Client aborted mid-stream: the engine thread keeps
                // calling deliverToken into req_state.completion until the
                // request finishes, so this frame must not unwind (the
                // req_state defer would free the signal out from under the
                // engine). Cancel the request and drain the signal instead.
                abortStream(io, req_state, allocator);
                return;
            };

            if (e.is_final) {
                sse.sendEvent("[DONE]") catch {
                    abortStream(io, req_state, allocator);
                    return;
                };
                break;
            }
        } else {
            break;
        }
    }

    // Engine may still be finishing its post-generation bookkeeping
    // (token_count write, completion log) after the final event; wait so the
    // metrics read below and the deferred deinit are race-free.
    req_state.waitEngineDone(io);
    metrics_mod.global.addTokenCounts(@intCast(prompt_tokens.len), @intCast(req_state.token_count));
}

/// Client disconnected (or the handler is unwinding) mid-stream: request
/// cancellation (so a still-queued request never starts) and drain the
/// completion signal until the engine reports the request done. Only after
/// that is it safe for the caller's req_state.deinit() to run. Shared by
/// the chat (streaming.zig) and Responses (responses.zig) streaming loops.
pub fn abortStream(io: std.Io, req_state: *engine.RequestState, allocator: std.mem.Allocator) void {
    req_state.requestCancel();
    while (!req_state.engine_done.load(.acquire)) {
        while (req_state.completion.tryGetToken(io)) |e| {
            if (e.token_text.len > 0) allocator.free(e.token_text);
        }
        io.sleep(.fromMilliseconds(1), .awake) catch {};
    }
    // Final drain of anything queued between the last poll and engine_done.
    while (req_state.completion.tryGetToken(io)) |e| {
        if (e.token_text.len > 0) allocator.free(e.token_text);
    }
}

/// SSE keep-alive fiber: sends a comment every 5s until `stop` is set
/// (first token generated, or the handler is tearing down). The owner
/// cancels the fiber before its stack frame exits; `sleep` is the
/// cancellation point, and write failures (e.g. the client aborted the
/// connection) terminate the loop.
fn keepAliveLoop(io_arg: std.Io, sse_arg: *SSEWriter, stop: *std.atomic.Value(bool)) void {
    while (!stop.load(.acquire)) {
        io_arg.sleep(.fromMilliseconds(5000), .awake) catch break;
        if (stop.load(.acquire)) break;
        sse_arg.sendKeepAlive() catch break;
    }
}

// ------------------------------------------------------------------
// Anthropic Messages API handler
// ------------------------------------------------------------------
