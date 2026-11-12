/// HTTP request handling.
const std = @import("std");
const root = @import("../root.zig");
const c = @import("mlx").c;
const memory_mod = @import("../memory.zig");
const engine = @import("../engine/root.zig");
const dynamic_buffer_mod = @import("../engine/dynamic_buffer.zig");
const config_mod = @import("config.zig");
const utils_mod = @import("utils.zig");
const state_mod = @import("state.zig");
const openai_mod = @import("openai.zig");
const streaming_mod = @import("streaming.zig");
const anthropic_mod = @import("anthropic.zig");
const responses_mod = @import("responses.zig");
const tokenize_mod = @import("tokenize.zig");
const embeddings_mod = @import("embeddings.zig");
const metrics_mod = @import("metrics.zig");
const ChatCompletionRequest = openai_mod.ChatCompletionRequest;
const handleStreamingCompletion = streaming_mod.handleStreamingCompletion;
const generateChatCompletion = openai_mod.generateChatCompletion;
const handleAnthropicMessages = anthropic_mod.handleAnthropicMessages;
const DynamicBuffer = dynamic_buffer_mod.DynamicBuffer;

const ServerConfig = config_mod.ServerConfig;
const ServerState = state_mod.ServerState;

fn handleRequest(
    allocator: std.mem.Allocator,
    io: std.Io,
    state: *ServerState,
    connection: std.Io.net.Stream,
    server_config: ServerConfig,
) !void {
    const t_handler_enter = std.c.mach_absolute_time();
    defer {
        const t_handler_exit = std.c.mach_absolute_time();
        const handler_ms = @as(f64, @floatFromInt(t_handler_exit - t_handler_enter)) * 125.0 / 3_000_000.0;
        std.log.info("[HTTP] Handler lifetime: {d:.1}ms", .{handler_ms});
    }
    // Use a growable DynamicBuffer instead of a fixed 64KB stack buffer.
    // Max size 16MB — requests larger than this are rejected with PayloadTooLarge.
    var dyn_buf = DynamicBuffer.init(16 * 1024 * 1024);
    defer dyn_buf.deinit(allocator);

    var read_buf: [4096]u8 = undefined;

    // Read the complete HTTP request (headers + body).
    //
    // Design note: We use std.posix.read on an O_NONBLOCK socket instead of
    // std.Io.net.Stream.reader. The Zig 0.16.0 Threaded backend's netReadPosix
    // treats EAGAIN as a bug (errnoBug), which makes std.Io.Reader unreliable
    // on non-blocking sockets. On blocking sockets, readSliceShort attempts to
    // fill the entire buffer, causing the worker thread to block indefinitely
    // when the client has sent all data but keeps the connection open.
    //
    // Reference: omlx (Python/FastAPI) and SwiftLM (Swift/Hummingbird) both
    // delegate HTTP request parsing to mature frameworks (Starlette/Hummingbird)
    // rather than handling raw socket reads themselves. In Zig 0.16.0, where
    // a production-grade HTTP framework is not yet available, std.posix.read
    // + O_NONBLOCK + explicit timeout is the most robust approach.
    var expected_total: ?usize = null;
    const t_http_enter = std.c.mach_absolute_time();
    const read_start = std.Io.Timestamp.now(io, .awake);
    const read_timeout_ns: i96 = 5_000_000_000; // 5 seconds total read timeout
    while (true) {
        const bytes_read = std.posix.read(connection.socket.handle, &read_buf) catch |err| {
            if (err == error.WouldBlock) {
                // If we already know how many bytes to expect, check if we're done.
                if (expected_total) |et| {
                    if (dyn_buf.len() >= et) break;
                }
                // Check total read timeout.
                const elapsed = std.Io.Timestamp.durationTo(read_start, std.Io.Timestamp.now(io, .awake)).toNanoseconds();
                if (elapsed >= read_timeout_ns) {
                    std.log.warn("[HTTP] Read timeout after {d}ms, received {d} bytes", .{
                        @divTrunc(elapsed, 1_000_000),
                        dyn_buf.len(),
                    });
                    break;
                }
                // Yield fiber briefly. io.sleep is safe here because we are
                // running inside an io.async fiber (not the main thread).
                io.sleep(.fromMilliseconds(1), .awake) catch break;
                continue;
            }
            return err;
        };
        if (bytes_read == 0) break;
        try dyn_buf.append(allocator, read_buf[0..bytes_read]);

        // Check if we have the full headers
        if (std.mem.find(u8, dyn_buf.items(), "\r\n\r\n")) |header_end| {
            const cl_prefix = "Content-Length: ";
            if (std.mem.find(u8, dyn_buf.items()[0..header_end], cl_prefix)) |cl_start| {
                const cl_value_start = cl_start + cl_prefix.len;
                const cl_end = std.mem.find(u8, dyn_buf.items()[cl_value_start..], "\r\n") orelse continue;
                const cl_str = dyn_buf.items()[cl_value_start .. cl_value_start + cl_end];
                const content_length = std.fmt.parseInt(usize, cl_str, 10) catch continue;
                const body_start = header_end + 4;
                expected_total = body_start + content_length;
                if (dyn_buf.len() >= body_start + content_length) break;
            } else {
                break; // No body expected
            }
        }
    }

    if (dyn_buf.len() == 0) return;
    const t_read_done = std.c.mach_absolute_time();
    const read_ms = @as(f64, @floatFromInt(t_read_done - t_http_enter)) * 125.0 / 3_000_000.0;
    const request = dyn_buf.items();
    std.log.info("[HTTP] Request received: {d} bytes (read={d:.1}ms)", .{ dyn_buf.len(), read_ms });

    // Prometheus metrics: record route/status/duration when the handler
    // returns. `status_code` defaults to 500 — a handler that throws before
    // writing a response is counted as a server error. Sub-handlers that
    // write responses internally (streaming, anthropic, responses) are
    // recorded as 200 on successful return.
    const route_idx = metrics_mod.Metrics.routeIndex(request);
    var status_code: u16 = 500;
    defer {
        const t_end = std.c.mach_absolute_time();
        const dur_ns = @as(u64, @intCast(t_end - t_http_enter)) * 125 / 3;
        metrics_mod.global.recordRequest(route_idx, status_code, dur_ns);
    }

    const libc = @cImport(@cInclude("unistd.h"));
    const msg = "[HTTP] Request received\n";
    _ = libc.write(2, msg, msg.len);

    if (std.mem.startsWith(u8, request, "POST /v1/chat/completions")) {
        // Enforce memory limit before processing the request.
        memory_mod.enforceMemoryLimit(
            if (state.model_pool) |*pool| pool else null,
            null,
            server_config.memory_config,
        ) catch {
            status_code = 503;
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
                status_code = 400;
                try writeJsonResponse(connection, io, 400, "{\"error\":\"invalid_json\"}");
                return;
            };
            defer parsed.deinit();

            // Reject invalid sampling overrides / unsupported tool_choice
            // forms up front so both the streaming and non-streaming paths
            // get a clean 400 instead of a mid-stream error event.
            if (try openai_mod.validateChatRequest(allocator, parsed.value)) |err_body| {
                defer allocator.free(err_body);
                status_code = 400;
                try writeJsonResponse(connection, io, 400, err_body);
                return;
            }

            if (parsed.value.stream orelse false) {
                status_code = 200;
                try handleStreamingCompletion(allocator, io, state, connection, server_config, parsed.value);
            } else {
                const t0 = std.c.mach_absolute_time();
                const response = try generateChatCompletion(allocator, io, state, body, server_config);
                defer allocator.free(response);
                const t1 = std.c.mach_absolute_time();
                status_code = 200;
                try writeJsonResponse(connection, io, 200, response);
                const t2 = std.c.mach_absolute_time();
                const gen_ms = @as(f64, @floatFromInt(t1 - t0)) * 125.0 / 3_000_000.0;
                const write_ms = @as(f64, @floatFromInt(t2 - t1)) * 125.0 / 3_000_000.0;
                std.log.info("[TIMING] generateChatCompletion={d:.1}ms writeResponse={d:.1}ms", .{ gen_ms, write_ms });
            }
        } else {
            status_code = 400;
            try writeJsonResponse(connection, io, 400, "{\"error\":\"bad_request\"}");
        }
    } else if (std.mem.startsWith(u8, request, "GET /v1/models")) {
        // OpenAI-compatible model listing. Single-model server: the id is
        // the basename of the configured model directory (what clients
        // should echo back in the `model` field), `created` is the server
        // start time. `state.model_name` holds the config.json architecture
        // name, which is not a useful client-facing id.
        const models_json = try formatModelsJson(
            allocator,
            std.fs.path.basename(server_config.model_path),
            state.start_time_unix,
        );
        defer allocator.free(models_json);
        status_code = 200;
        try writeJsonResponse(connection, io, 200, models_json);
    } else if (std.mem.startsWith(u8, request, "GET /health")) {
        const active = state.active_requests.load(.acquire);
        var escaped_model = std.ArrayList(u8).empty;
        defer escaped_model.deinit(allocator);
        try utils_mod.jsonEscapeInto(&escaped_model, allocator, state.model_name);
        const health_json = try std.fmt.allocPrint(
            allocator,
            "{{\"status\":\"ok\",\"model\":\"{s}\",\"active_requests\":{d}}}",
            .{ escaped_model.items, active },
        );
        defer allocator.free(health_json);
        status_code = 200;
        try writeJsonResponse(connection, io, 200, health_json);
    } else if (std.mem.startsWith(u8, request, "GET /metrics") or
        std.mem.startsWith(u8, request, "GET /v1/metrics"))
    {
        // Prometheus text exposition format (v0.0.4). Counters live in the
        // global metrics registry; gauges are computed at render time.
        const body = try metrics_mod.global.render(
            allocator,
            state.active_requests.load(.acquire),
            state.start_time_unix,
            std.Io.Timestamp.now(io, .real).toSeconds(),
        );
        defer allocator.free(body);
        status_code = 200;
        try writeResponseAlloc(connection, 200, "text/plain; version=0.0.4", body);
    } else if (std.mem.startsWith(u8, request, "POST /shutdown")) {
        engine.requestShutdown();
        status_code = 200;
        try writeJsonResponse(connection, io, 200, "{\"status\":\"shutting_down\"}");
    } else if (std.mem.startsWith(u8, request, "POST /v1/responses")) {
        // OpenAI Responses API — minimal compatibility layer (translation
        // onto the chat-completions machinery; see responses.zig).
        memory_mod.enforceMemoryLimit(
            if (state.model_pool) |*pool| pool else null,
            null,
            server_config.memory_config,
        ) catch {
            status_code = 503;
            try writeJsonResponse(connection, io, 503, "{\"error\":{\"message\":\"Memory limit exceeded — server is under memory pressure. Please retry later.\",\"type\":\"server_error\",\"code\":\"memory_limit_exceeded\"}}");
            return;
        };
        if (std.mem.find(u8, request, "\r\n\r\n")) |header_end| {
            const body = request[header_end + 4 ..];
            status_code = 200;
            try responses_mod.handleResponses(allocator, io, state, connection, server_config, body);
        } else {
            status_code = 400;
            try writeJsonResponse(connection, io, 400, "{\"error\":\"bad_request\"}");
        }
    } else if (std.mem.startsWith(u8, request, "POST /v1/tokenize") or
        std.mem.startsWith(u8, request, "POST /tokenize"))
    {
        // Raw-text tokenization for context budgeting (no chat template).
        if (std.mem.find(u8, request, "\r\n\r\n")) |header_end| {
            const body = request[header_end + 4 ..];
            const result = try tokenize_mod.handleTokenizeRequest(allocator, state.tokenizer_strategy, body);
            defer allocator.free(result.body);
            status_code = result.status;
            try writeJsonResponseAlloc(connection, result.status, result.body);
        } else {
            status_code = 400;
            try writeJsonResponse(connection, io, 400, "{\"error\":\"bad_request\"}");
        }
    } else if (std.mem.startsWith(u8, request, "POST /v1/embeddings")) {
        // OpenAI-compatible embeddings from the loaded model's hidden states.
        // Pooling/extraction: last-token POST-final-norm hidden, L2-normalized
        // (see embeddings.zig). The forward runs behind the model adapter's
        // embed op (null → 501 embedding_not_supported), serialized against
        // chat traffic by the adapter mutex.
        memory_mod.enforceMemoryLimit(
            if (state.model_pool) |*pool| pool else null,
            null,
            server_config.memory_config,
        ) catch {
            status_code = 503;
            try writeJsonResponse(connection, io, 503, "{\"error\":{\"message\":\"Memory limit exceeded — server is under memory pressure. Please retry later.\",\"type\":\"server_error\",\"code\":\"memory_limit_exceeded\"}}");
            return;
        };
        if (std.mem.find(u8, request, "\r\n\r\n")) |header_end| {
            const body = request[header_end + 4 ..];
            // The forward runs on the engine-loop thread via the embed task
            // queue (GPU work is thread-affine); null runner → 501.
            const runner: ?embeddings_mod.EmbedRunner = if (state.vtable.embed != null)
                .{ .ctx = state, .run = &embeddings_mod.engineEmbedRunner }
            else
                null;
            const result = try embeddings_mod.handleEmbeddingsRequest(
                allocator,
                state.tokenizer_strategy,
                runner,
                std.fs.path.basename(server_config.model_path),
                body,
            );
            defer allocator.free(result.body);
            status_code = result.status;
            // 5120 floats ≈ 60KB JSON per item — heap-allocated response.
            try writeJsonResponseAlloc(connection, result.status, result.body);
        } else {
            status_code = 400;
            try writeJsonResponse(connection, io, 400, "{\"error\":\"bad_request\"}");
        }
    } else if (std.mem.startsWith(u8, request, "POST /v1/messages")) {
        // Anthropic Messages API compatibility endpoint
        if (std.mem.find(u8, request, "\r\n\r\n")) |header_end| {
            const body = request[header_end + 4 ..];
            status_code = 200;
            try handleAnthropicMessages(allocator, io, state, connection, server_config, body);
        } else {
            status_code = 400;
            try writeJsonResponse(connection, io, 400, "{\"error\":\"bad_request\"}");
        }
    } else {
        status_code = 404;
        try writeJsonResponse(connection, io, 404, "{\"error\":\"not_found\"}");
    }
}

// ------------------------------------------------------------------
// HTTP response helpers
// ------------------------------------------------------------------

/// Build the GET /v1/models response body. The model id is JSON-escaped.
pub fn formatModelsJson(allocator: std.mem.Allocator, model_id: []const u8, created_unix: i64) ![]u8 {
    var escaped_id = std.ArrayList(u8).empty;
    defer escaped_id.deinit(allocator);
    try utils_mod.jsonEscapeInto(&escaped_id, allocator, model_id);
    return std.fmt.allocPrint(
        allocator,
        "{{\"object\":\"list\",\"data\":[{{\"id\":\"{s}\",\"object\":\"model\",\"created\":{d},\"owned_by\":\"dmlx\"}}]}}",
        .{ escaped_id.items, created_unix },
    );
}

test "formatModelsJson shape" {
    const allocator = std.testing.allocator;
    const body = try formatModelsJson(allocator, "Qwen3.8-27B-4bit", 1794458459);
    defer allocator.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const root_obj = parsed.value.object;
    try std.testing.expectEqualStrings("list", root_obj.get("object").?.string);
    const data = root_obj.get("data").?.array;
    try std.testing.expectEqual(@as(usize, 1), data.items.len);
    const entry = data.items[0].object;
    try std.testing.expectEqualStrings("Qwen3.8-27B-4bit", entry.get("id").?.string);
    try std.testing.expectEqualStrings("model", entry.get("object").?.string);
    try std.testing.expectEqual(@as(i64, 1794458459), entry.get("created").?.integer);
    try std.testing.expectEqualStrings("dmlx", entry.get("owned_by").?.string);
}

test "formatModelsJson escapes quotes and backslashes in the id" {
    const allocator = std.testing.allocator;
    const body = try formatModelsJson(allocator, "evil\"id\\name", 0);
    defer allocator.free(body);

    // Must still be valid JSON and round-trip to the raw id.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const data = parsed.value.object.get("data").?.array;
    try std.testing.expectEqualStrings("evil\"id\\name", data.items[0].object.get("id").?.string);
}

pub fn writeJsonResponse(connection: std.Io.net.Stream, io: std.Io, status: u16, body: []const u8) !void {
    _ = io;
    const status_text = switch (status) {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        else => "Error",
    };

    // Build entire HTTP response in a single buffer, then write once.
    var resp_buf: [8192]u8 = undefined;
    const resp = try std.fmt.bufPrint(&resp_buf, "HTTP/1.1 {d} {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ status, status_text, body.len, body });
    try writeAllFd(connection.socket.handle, resp);
    // Flush by sync'ing the fd (no-op on socket but won't hurt)
    _ = unistd_c.fsync(connection.socket.handle);
}

/// Heap-allocated variant of writeJsonResponse for bodies that can exceed
/// the 8KB stack buffer (e.g. /v1/tokenize output for large inputs).
pub fn writeJsonResponseAlloc(connection: std.Io.net.Stream, status: u16, body: []const u8) !void {
    try writeResponseAlloc(connection, status, "application/json", body);
}

/// Heap-allocated response writer with an explicit Content-Type.
pub fn writeResponseAlloc(connection: std.Io.net.Stream, status: u16, content_type: []const u8, body: []const u8) !void {
    const status_text = switch (status) {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        else => "Error",
    };
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const resp = try std.fmt.allocPrint(arena.allocator(), "HTTP/1.1 {d} {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ status, status_text, content_type, body.len, body });
    try writeAllFd(connection.socket.handle, resp);
    _ = unistd_c.fsync(connection.socket.handle);
}

const unistd_c = @cImport(@cInclude("unistd.h"));

fn writeAllFd(fd: c_int, data: []const u8) !void {
    var remaining = data;
    while (remaining.len > 0) {
        const n = unistd_c.write(fd, remaining.ptr, remaining.len);
        if (n < 0) return error.WriteFailed;
        remaining = remaining[@intCast(n)..];
    }
}

pub fn streamWriteAll(stream: std.Io.net.Stream, io: std.Io, data: []const u8) !void {
    // Fallback: use the original buffered writer.
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
    const t_conn_enter = std.c.mach_absolute_time();
    defer {
        const t_conn_exit = std.c.mach_absolute_time();
        const conn_ms = @as(f64, @floatFromInt(t_conn_exit - t_conn_enter)) * 125.0 / 3_000_000.0;
        std.log.info("[HTTP] Connection lifetime: {d:.1}ms", .{conn_ms});
    }
    defer connection.close(io);
    handleRequest(allocator, io, state, connection, config) catch |err| {
        std.log.err("Request error: {}", .{err});
    };
}
