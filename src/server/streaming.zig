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
const tool_calling_mod = @import("../tool_calling.zig");
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
    req_param: ChatCompletionRequest,
    raw_body: []const u8,
) !void {
    var req = req_param;
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
    // template, tokenize). Shared with the non-streaming handler. Tool
    // definitions inject a leading system prompt, same as non-streaming.
    // Mirrors the non-streaming handler: applies tool_choice "none"
    // stripping; validation already happened at HTTP dispatch.
    openai_mod.resolveToolChoice(&req) catch unreachable;
    var tool_system: ?[]u8 = null;
    defer if (tool_system) |ts| allocator.free(ts);
    tool_system = try openai_mod.buildToolSystem(allocator, req);
    var prepared = openai_mod.preparePrompt(allocator, state, req, tool_system) catch |err| {
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
    @import("http.zig").maybeDumpPromptBody(io, raw_body, prompt_tokens.len);

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

    // Tool-enabled requests: normal text streams live; output is buffered
    // only while it can still turn into a tool call (the family's open
    // marker, possibly split across tokens). Previously the whole
    // generation was buffered whenever tools were present, so agent
    // harnesses (always tool-enabled) looked frozen until the reply
    // finished. Families without a reliable open marker keep the old
    // full-buffer behavior.
    const tool_family = tool_calling_mod.detectFamily(req.model);
    const tool_marker: ?[]const u8 = if (tool_system != null) switch (tool_family) {
        .qwen => "<tool_call>",
        .llama => "<|python_tag|>",
        .deepseek => "<|tool|>",
        .unknown => null,
    } else null;
    const buffer_for_tools = tool_system != null;
    const ToolMode = enum { deciding, streaming, buffering };
    var tool_mode: ToolMode = .deciding;
    var buffered = std.ArrayList(u8).empty;
    defer buffered.deinit(allocator);
    var holdback = std.ArrayList(u8).empty;
    defer holdback.deinit(allocator);

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
            if (buffer_for_tools) {
                const finish_str: ?[]const u8 = if (e.is_final)
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

                if (tool_marker) |marker| {
                    if (tool_mode == .streaming) {
                        try holdback.appendSlice(allocator, e.token_text);
                    } else {
                        try buffered.appendSlice(allocator, e.token_text);
                    }
                    while (true) {
                        switch (tool_mode) {
                            .deciding => {
                                const trimmed = std.mem.trimStart(u8, buffered.items, " \t\n\r");
                                const n = @min(trimmed.len, marker.len);
                                if (std.mem.eql(u8, trimmed[0..n], marker[0..n])) {
                                    if (trimmed.len >= marker.len) tool_mode = .buffering;
                                    break; // still consistent with a tool call
                                }
                                tool_mode = .streaming; // plain text: go live
                            },
                            .streaming => {
                                const keep = marker.len - 1;
                                if (std.mem.indexOf(u8, holdback.items, marker)) |p| {
                                    // A tool call begins mid-reply: flush the
                                    // prose before it, buffer from the marker.
                                    if (p > 0) {
                                        if (ka_running) {
                                            stop_keepalive.store(true, .release);
                                            ka_future.cancel(io);
                                            ka_running = false;
                                        }
                                        sendContentChunk(&sse, allocator, completion_id, created, req.model, holdback.items[0..p], null) catch {
                                            abortStream(io, req_state, allocator);
                                            return;
                                        };
                                    }
                                    try buffered.appendSlice(allocator, holdback.items[p..]);
                                    holdback.clearRetainingCapacity();
                                    tool_mode = .buffering;
                                } else if (holdback.items.len > keep) {
                                    if (ka_running) {
                                        stop_keepalive.store(true, .release);
                                        ka_future.cancel(io);
                                        ka_running = false;
                                    }
                                    const flush_len = holdback.items.len - keep;
                                    sendContentChunk(&sse, allocator, completion_id, created, req.model, holdback.items[0..flush_len], null) catch {
                                        abortStream(io, req_state, allocator);
                                        return;
                                    };
                                    std.mem.copyForwards(u8, holdback.items[0..keep], holdback.items[flush_len..]);
                                    holdback.shrinkRetainingCapacity(keep);
                                }
                                break;
                            },
                            .buffering => break,
                        }
                    }
                    if (!e.is_final) continue;
                    if (ka_running) {
                        stop_keepalive.store(true, .release);
                        ka_future.cancel(io);
                        ka_running = false;
                    }
                    if (tool_mode == .streaming) {
                        if (holdback.items.len > 0) {
                            sendContentChunk(&sse, allocator, completion_id, created, req.model, holdback.items, null) catch {
                                abortStream(io, req_state, allocator);
                                return;
                            };
                        }
                        sendContentChunk(&sse, allocator, completion_id, created, req.model, "", finish_str orelse "stop") catch {
                            abortStream(io, req_state, allocator);
                            return;
                        };
                        sse.sendEvent("[DONE]") catch {
                            abortStream(io, req_state, allocator);
                            return;
                        };
                        break;
                    }
                    // .deciding (degenerate empty output) or .buffering:
                    // the whole reply is in buffered — parse as before.
                } else {
                    try buffered.appendSlice(allocator, e.token_text);
                    if (!e.is_final) continue;
                    if (ka_running) {
                        stop_keepalive.store(true, .release);
                        ka_future.cancel(io);
                        ka_running = false;
                    }
                }
                const finish: []const u8 = finish_str orelse "stop";
                var sent = false;
                if (tool_calling_mod.parse(allocator, buffered.items, tool_family) catch null) |pr| {
                    defer pr.deinit();
                    if (pr.calls.len > 0) {
                        const calls_json = try openai_mod.formatToolCallsDeltaJson(allocator, pr.calls);
                        defer allocator.free(calls_json);
                        const chunk = try std.fmt.allocPrint(allocator,
                            \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"tool_calls":{s}}},"finish_reason":"tool_calls"}}]}}
                        , .{ completion_id, created, req.model, calls_json });
                        defer allocator.free(chunk);
                        try sse.sendEvent(chunk);
                        sent = true;
                    }
                }
                if (!sent) {
                    sendContentChunk(&sse, allocator, completion_id, created, req.model, buffered.items, finish) catch {
                        abortStream(io, req_state, allocator);
                        return;
                    };
                }
                sse.sendEvent("[DONE]") catch {
                    abortStream(io, req_state, allocator);
                    return;
                };
                break;
            }
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
/// Format + send one SSE content chunk (frees its own buffer).
fn sendContentChunk(
    sse: *sse_mod.SSEWriter,
    allocator: std.mem.Allocator,
    completion_id: u64,
    created: u64,
    model: []const u8,
    text: []const u8,
    finish: ?[]const u8,
) !void {
    const chunk = try formatSSEChunk(allocator, completion_id, created, model, text, finish);
    defer allocator.free(chunk);
    try sse.sendEvent(chunk);
}

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
