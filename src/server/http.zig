/// HTTP request handling.
const std = @import("std");
const root = @import("../root.zig");
const c = @import("mlx").c;
const memory_mod = @import("../memory.zig");
const engine = @import("../engine/root.zig");
const config_mod = @import("config.zig");
const utils_mod = @import("utils.zig");
const state_mod = @import("state.zig");
const openai_mod = @import("openai.zig");
const streaming_mod = @import("streaming.zig");
const anthropic_mod = @import("anthropic.zig");
const ChatCompletionRequest = openai_mod.ChatCompletionRequest;
const handleStreamingCompletion = streaming_mod.handleStreamingCompletion;
const generateChatCompletion = openai_mod.generateChatCompletion;
const handleAnthropicMessages = anthropic_mod.handleAnthropicMessages;

const ServerConfig = config_mod.ServerConfig;
const ServerState = state_mod.ServerState;

fn handleRequest(
    allocator: std.mem.Allocator,
    io: std.Io,
    state: *ServerState,
    connection: std.Io.net.Stream,
    server_config: ServerConfig,
) !void {
    var buf: [65536]u8 = undefined;
    var total_read: usize = 0;

    // Read the complete HTTP request (headers + body).
    // The socket from listener.accept is non-blocking; when no data is
    // available we sleep briefly to yield the async fiber.
    var expected_total: ?usize = null;
    var would_block_count: usize = 0;
    while (total_read < buf.len) {
        const bytes_read = std.posix.read(connection.socket.handle, buf[total_read..]) catch |err| {
            if (err == error.WouldBlock) {
                // If we already know how many bytes to expect, check if we're done.
                if (expected_total) |et| {
                    if (total_read >= et) break;
                }
                would_block_count += 1;
                if (would_block_count > 100) break;
                io.sleep(.fromMilliseconds(1), .awake) catch break;
                continue;
            }
            return err;
        };
        if (bytes_read == 0) break;
        total_read += bytes_read;
        would_block_count = 0;

        // Check if we have the full headers
        if (std.mem.find(u8, buf[0..total_read], "\r\n\r\n")) |header_end| {
            const cl_prefix = "Content-Length: ";
            if (std.mem.find(u8, buf[0..header_end], cl_prefix)) |cl_start| {
                const cl_value_start = cl_start + cl_prefix.len;
                // Search up to header_end + 2 to include the \r\n after Content-Length
                const search_end = @min(buf.len, header_end + 2);
                const cl_end = std.mem.find(u8, buf[cl_value_start..search_end], "\r\n") orelse continue;
                const cl_str = buf[cl_value_start .. cl_value_start + cl_end];
                const content_length = std.fmt.parseInt(usize, cl_str, 10) catch continue;
                const body_start = header_end + 4;
                expected_total = body_start + content_length;
                if (total_read >= body_start + content_length) break;
            } else {
                break; // No body expected
            }
        }
    }

    if (total_read == 0) return;
    const request = buf[0..total_read];
    std.log.info("[HTTP] Request received: {d} bytes", .{total_read});

    if (std.mem.startsWith(u8, request, "POST /v1/chat/completions")) {
        // Enforce memory limit before processing the request.
        memory_mod.enforceMemoryLimit(
            if (state.model_pool) |*pool| pool else null,
            null,
            server_config.memory_config,
        ) catch {
            try writeJsonResponse(connection, io, 503, "{\"error\":{\"message\":\"Memory limit exceeded — server is under memory pressure. Please retry later.\",\"type\":\"server_error\",\"code\":\"memory_limit_exceeded\"}}");
            return;
        };

        // --- Multi-model routing integration point (ModelPool) ---
        // The current server loads a single model at startup and routes all
        // requests to it. To enable multi-model serving:
        //
        //   1. Parse the `model` field from the request body (already available
        //      in ChatCompletionRequest.model after JSON parsing below).
        //   2. Look up or load the requested model via ModelPool:
        //
        //        const requested_model = parsed.value.model;
        //        const loaded = try state.model_pool.?.getOrLoad(
        //            requested_model,
        //            model_path_for(requested_model),
        //            modelLoaderFn,
        //        );
        //        // Use loaded.vtable instead of state.vtable for this request.
        //
        //   3. ModelPool.getOrLoad handles LRU eviction automatically when the
        //      pool's memory budget is exceeded (R21.3, R21.4).
        //   4. The default single-model path below remains the fallback when
        //      the requested model name matches the startup model.
        //
        // This requires a model path resolver (mapping model names to filesystem
        // paths) and a loader function compatible with ModelPool.LoaderFn.
        // For now, all requests use the single model loaded at startup.

        // --- Scheduler integration point ---
        // When the server supports async I/O / multi-threaded request handling,
        // replace the serial generation below with a Scheduler-driven engine loop:
        //   1. Create a scheduler_mod.Request from the parsed chat completion request
        //   2. Call state.scheduler.?.addRequest(&req) to enqueue it
        //   3. In the engine loop: call scheduler.schedule() to pick requests
        //   4. Use batch_builder_mod to build a batched input tensor from scheduled requests
        //   5. Run the batched forward pass via state.vtable.forward(...)
        //   6. Call scheduler.postprocess(outputs) to append tokens and check stop conditions
        //   7. Stream results back to each request's connection via SSE
        // For now, requests are handled serially (single-threaded HTTP server).

        if (std.mem.find(u8, request, "\r\n\r\n")) |header_end| {
            const body = request[header_end + 4 ..];

            // Parse request to check for streaming
            const parsed = std.json.parseFromSlice(ChatCompletionRequest, allocator, body, .{}) catch {
                try writeJsonResponse(connection, io, 400, "{\"error\":\"invalid_json\"}");
                return;
            };
            defer parsed.deinit();

            if (parsed.value.stream orelse false) {
                try handleStreamingCompletion(allocator, io, state, connection, server_config, parsed.value);
            } else {
                const response = try generateChatCompletion(allocator, io, state, body, server_config);
                defer allocator.free(response);
                try writeJsonResponse(connection, io, 200, response);
            }
        } else {
            try writeJsonResponse(connection, io, 400, "{\"error\":\"bad_request\"}");
        }
    } else if (std.mem.startsWith(u8, request, "GET /health")) {
        try writeJsonResponse(connection, io, 200, "{\"status\":\"ok\"}");
    } else if (std.mem.startsWith(u8, request, "POST /v1/messages")) {
        // Anthropic Messages API compatibility endpoint
        if (std.mem.find(u8, request, "\r\n\r\n")) |header_end| {
            const body = request[header_end + 4 ..];
            try handleAnthropicMessages(allocator, io, state, connection, server_config, body);
        } else {
            try writeJsonResponse(connection, io, 400, "{\"error\":\"bad_request\"}");
        }
    } else {
        try writeJsonResponse(connection, io, 404, "{\"error\":\"not_found\"}");
    }
}

// ------------------------------------------------------------------
// HTTP response helpers
// ------------------------------------------------------------------

pub fn writeJsonResponse(connection: std.Io.net.Stream, io: std.Io, status: u16, body: []const u8) !void {
    var status_buf: [32]u8 = undefined;
    const status_text = switch (status) {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        503 => "Service Unavailable",
        else => "Error",
    };
    const status_line = try std.fmt.bufPrint(&status_buf, "HTTP/1.1 {d} {s}\r\n", .{ status, status_text });

    var cl_buf: [64]u8 = undefined;
    const cl_line = try std.fmt.bufPrint(&cl_buf, "Content-Length: {d}\r\n", .{body.len});

    try streamWriteAll(connection, io, status_line);
    try streamWriteAll(connection, io, "Content-Type: application/json\r\n");
    try streamWriteAll(connection, io, cl_line);
    try streamWriteAll(connection, io, "Connection: close\r\n");
    try streamWriteAll(connection, io, "\r\n");
    try streamWriteAll(connection, io, body);
}

pub fn streamWriteAll(stream: std.Io.net.Stream, io: std.Io, data: []const u8) !void {
    var buf: [4096]u8 = undefined;
    var writer = stream.writer(io, &buf);
    try writer.interface.writeAll(data);
    try writer.interface.flush();
}

/// Concurrent connection handler — runs in its own async fiber.
/// Owns the connection lifetime (closes on exit).
pub fn handleConnection(
    allocator: std.mem.Allocator,
    io: std.Io,
    state: *ServerState,
    connection: std.Io.net.Stream,
    config: ServerConfig,
) void {
    defer connection.close(io);
    handleRequest(allocator, io, state, connection, config) catch |err| {
        std.log.err("Request error: {}", .{err});
    };
}
