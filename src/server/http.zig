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
    // Use a growable DynamicBuffer instead of a fixed 64KB stack buffer.
    // Max size 16MB — requests larger than this are rejected with PayloadTooLarge.
    var dyn_buf = DynamicBuffer.init(16 * 1024 * 1024);
    defer dyn_buf.deinit(allocator);

    var read_buf: [4096]u8 = undefined;

    // Read the complete HTTP request (headers + body).
    //
    // With thread-based handlers, the socket is in BLOCKING mode.
    // Set SO_RCVTIMEO to prevent indefinite blocking on slow/dead clients.
    // For fiber-based handlers (O_NONBLOCK), WouldBlock + io.sleep polling is used.
    {
        const sock_c = @cImport({
            @cInclude("sys/socket.h");
            @cInclude("sys/time.h");
        });
        const tv = sock_c.struct_timeval{ .tv_sec = 5, .tv_usec = 0 };
        _ = sock_c.setsockopt(connection.socket.handle, sock_c.SOL_SOCKET, sock_c.SO_RCVTIMEO, &tv, @sizeOf(@TypeOf(tv)));
    }
    var expected_total: ?usize = null;
    const read_start = std.Io.Timestamp.now(io, .awake);
    while (true) {
        const bytes_read = std.posix.read(connection.socket.handle, &read_buf) catch |err| {
            if (err == error.WouldBlock) {
                // Blocking socket with SO_RCVTIMEO: WouldBlock means timeout.
                // Non-blocking socket: WouldBlock means no data yet.
                if (expected_total) |et| {
                    if (dyn_buf.len() >= et) break;
                }
                // For blocking sockets, WouldBlock = timeout expired, stop reading.
                // For non-blocking sockets (legacy fiber path), yield briefly.
                const ts = std.c.timespec{ .sec = 0, .nsec = 1_000_000 }; // 1ms
                _ = std.c.nanosleep(&ts, null);
                // Check if we've been reading too long
                const elapsed = std.Io.Timestamp.durationTo(read_start, std.Io.Timestamp.now(io, .awake)).toNanoseconds();
                if (elapsed >= 5_000_000_000) {
                    break;
                }
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
    const request = dyn_buf.items();
    const read_end = std.Io.Timestamp.now(io, .awake);
    const read_elapsed_ns = std.Io.Timestamp.durationTo(read_start, read_end).toNanoseconds();
    std.log.info("[HTTP] Request received: {d} bytes (read took {d}ms)", .{ dyn_buf.len(), @divTrunc(read_elapsed_ns, 1_000_000) });

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
                const t_gen_start = std.c.mach_absolute_time();
                const response = try generateChatCompletion(allocator, io, state, body, server_config);
                defer allocator.free(response);
                const t_gen_end = std.c.mach_absolute_time();
                try writeJsonResponse(connection, io, 200, response);
                const t_write_end = std.c.mach_absolute_time();
                std.log.info("[HTTP] Response: gen={d}ms write={d}ms total={d}ms", .{
                    (t_gen_end - t_gen_start) / 1_000_000,
                    (t_write_end - t_gen_end) / 1_000_000,
                    (t_write_end - t_gen_start) / 1_000_000,
                });
            }
        } else {
            try writeJsonResponse(connection, io, 400, "{\"error\":\"bad_request\"}");
        }
    } else if (std.mem.startsWith(u8, request, "GET /health")) {
        const active = state.active_requests.load(.acquire);
        const health_json = try std.fmt.allocPrint(
            allocator,
            "{{\"status\":\"ok\",\"model\":\"{s}\",\"active_requests\":{d}}}",
            .{ state.model_name, active },
        );
        defer allocator.free(health_json);
        try writeJsonResponse(connection, io, 200, health_json);
    } else if (std.mem.startsWith(u8, request, "POST /shutdown")) {
        engine.requestShutdown();
        try writeJsonResponse(connection, io, 200, "{\"status\":\"shutting_down\"}");
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
    _ = io;
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

    // Use direct posix write (blocking) to avoid Zig IO scheduler delays.
    // The socket is O_NONBLOCK for async reads, but we temporarily switch to
    // blocking mode for writes to ensure data is flushed immediately.
    // See docs/analysis/socket-write-latency.md for the full investigation.
    const fc = @cImport({
        @cInclude("fcntl.h");
        @cInclude("sys/socket.h");
        @cInclude("netinet/tcp.h");
    });
    const fd = connection.socket.handle;
    const flags = fc.fcntl(fd, fc.F_GETFL, @as(c_int, 0));
    _ = fc.fcntl(fd, fc.F_SETFL, @as(c_int, flags & ~@as(c_int, fc.O_NONBLOCK))); // blocking
    // Disable Nagle's algorithm to flush data immediately
    const one: c_int = 1;
    _ = fc.setsockopt(fd, fc.IPPROTO_TCP, fc.TCP_NODELAY, &one, @sizeOf(c_int));
    defer _ = fc.fcntl(fd, fc.F_SETFL, @as(c_int, flags)); // restore

    posixWriteAll(fd, status_line);
    posixWriteAll(fd, "Content-Type: application/json\r\n");
    posixWriteAll(fd, cl_line);
    posixWriteAll(fd, "Connection: close\r\n");
    posixWriteAll(fd, "\r\n");
    posixWriteAll(fd, body);
}

/// Thread-based request handler using raw file descriptor.
/// Reads HTTP request via blocking libc read(), processes it,
/// and writes response via blocking libc write().
/// Does NOT depend on Zig IO scheduler at any point.
fn handleRequestRaw(
    allocator: std.mem.Allocator,
    state: *ServerState,
    fd: c_int,
    server_config: ServerConfig,
) !void {
    const libc = @cImport({
        @cInclude("unistd.h");
        @cInclude("sys/socket.h");
        @cInclude("netinet/tcp.h");
        @cInclude("sys/time.h");
    });

    // Set TCP_NODELAY to disable Nagle's algorithm (flush immediately)
    const one: c_int = 1;
    _ = libc.setsockopt(fd, libc.IPPROTO_TCP, libc.TCP_NODELAY, &one, @sizeOf(c_int));
    // Set read timeout (5s)
    const tv = libc.struct_timeval{ .tv_sec = 5, .tv_usec = 0 };
    _ = libc.setsockopt(fd, libc.SOL_SOCKET, libc.SO_RCVTIMEO, &tv, @sizeOf(@TypeOf(tv)));

    // Read HTTP request using blocking read
    var dyn_buf = DynamicBuffer.init(16 * 1024 * 1024);
    defer dyn_buf.deinit(allocator);

    var read_buf: [4096]u8 = undefined;
    var expected_total: ?usize = null;

    while (true) {
        const n = libc.read(fd, &read_buf, read_buf.len);
        if (n <= 0) break;
        const bytes_read: usize = @intCast(n);
        try dyn_buf.append(allocator, read_buf[0..bytes_read]);

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
                break;
            }
        }
    }

    if (dyn_buf.len() == 0) return;
    const request = dyn_buf.items();
    std.log.info("[HTTP] Request received: {d} bytes (raw thread)", .{dyn_buf.len()});

    // Route request
    if (std.mem.startsWith(u8, request, "POST /v1/chat/completions")) {
        if (std.mem.find(u8, request, "\r\n\r\n")) |header_end| {
            const body = request[header_end + 4 ..];

            const parsed = std.json.parseFromSlice(ChatCompletionRequest, allocator, body, .{}) catch {
                posixWriteAll(fd, "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: 20\r\nConnection: close\r\n\r\n{\"error\":\"bad_json\"}");
                return;
            };
            defer parsed.deinit();

            if (parsed.value.stream orelse false) {
                // For streaming, we need the Zig IO stream wrapper — fall back
                // TODO: implement raw streaming SSE writer
                posixWriteAll(fd, "HTTP/1.1 501 Not Implemented\r\nContent-Length: 42\r\nConnection: close\r\n\r\n{\"error\":\"streaming not supported in raw mode\"}");
                return;
            }

            const t_gen_start = std.c.mach_absolute_time();
            const response = try generateChatCompletion(allocator, state.io, state, body, server_config);
            defer allocator.free(response);
            const t_gen_end = std.c.mach_absolute_time();

            var cl_buf: [64]u8 = undefined;
            const cl_line = std.fmt.bufPrint(&cl_buf, "{d}", .{response.len}) catch "0";
            posixWriteAll(fd, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ");
            posixWriteAll(fd, cl_line);
            posixWriteAll(fd, "\r\nConnection: close\r\n\r\n");
            posixWriteAll(fd, response);
            const t_write_end = std.c.mach_absolute_time();
            std.log.info("[RAW] gen={d}ms write={d}ms", .{
                (t_gen_end - t_gen_start) / 1_000_000,
                (t_write_end - t_gen_end) / 1_000_000,
            });
        }
    } else if (std.mem.startsWith(u8, request, "GET /health")) {
        const active = state.active_requests.load(.acquire);
        const health_json = std.fmt.allocPrint(
            allocator,
            "{{\"status\":\"ok\",\"model\":\"{s}\",\"active_requests\":{d}}}",
            .{ state.model_name, active },
        ) catch return;
        defer allocator.free(health_json);

        var cl_buf: [64]u8 = undefined;
        const cl_line = std.fmt.bufPrint(&cl_buf, "{d}", .{health_json.len}) catch "0";
        posixWriteAll(fd, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ");
        posixWriteAll(fd, cl_line);
        posixWriteAll(fd, "\r\nConnection: close\r\n\r\n");
        posixWriteAll(fd, health_json);
    } else if (std.mem.startsWith(u8, request, "POST /shutdown")) {
        engine.requestShutdown();
        posixWriteAll(fd, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 24\r\nConnection: close\r\n\r\n{\"status\":\"shutting_down\"}");
    } else {
        posixWriteAll(fd, "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: 21\r\nConnection: close\r\n\r\n{\"error\":\"not_found\"}");
    }
}

fn posixWriteAll(fd: std.posix.fd_t, data: []const u8) void {
    const libc = @cImport(@cInclude("unistd.h"));
    var written: usize = 0;
    while (written < data.len) {
        const remaining = data[written..];
        const n = libc.write(fd, remaining.ptr, remaining.len);
        if (n <= 0) return;
        written += @intCast(n);
    }
}

pub fn streamWriteAll(stream: std.Io.net.Stream, io: std.Io, data: []const u8) !void {
    var buf: [4096]u8 = undefined;
    var writer = stream.writer(io, &buf);
    try writer.interface.writeAll(data);
    try writer.interface.flush();
}

/// Thread-based connection handler using raw file descriptor.
/// Does NOT use Zig IO at all — pure POSIX syscalls for read/write.
/// This avoids the ~20s Zig IO fiber scheduling delay entirely.
pub fn handleConnectionThreadRaw(
    allocator: std.mem.Allocator,
    state: *ServerState,
    fd: c_int,
    config: ServerConfig,
    accept_time: u64,
) void {
    const thread_start = std.c.mach_absolute_time();
    std.log.info("[HTTP] Thread started: accept→thread={d}ms", .{(thread_start - accept_time) / 1_000_000});
    const conn_start = std.c.mach_absolute_time();
    defer {
        const before_close = std.c.mach_absolute_time();
        const libc = @cImport(@cInclude("unistd.h"));
        _ = libc.close(fd);
        const after_close = std.c.mach_absolute_time();
        std.log.info("[HTTP] Connection lifetime: {d}ms (close took {d}ms)", .{
            (after_close - conn_start) / 1_000_000,
            (after_close - before_close) / 1_000_000,
        });
    }
    handleRequestRaw(allocator, state, fd, config) catch |err| {
        std.log.err("Request error: {}", .{err});
    };
}

/// Thread-based connection handler — runs in its own OS thread.
/// Does NOT use Zig IO (no fiber scheduling dependency).
pub fn handleConnectionThread(
    allocator: std.mem.Allocator,
    state: *ServerState,
    connection: std.Io.net.Stream,
    config: ServerConfig,
) void {
    const conn_start = std.c.mach_absolute_time();
    defer {
        const conn_end = std.c.mach_absolute_time();
        std.log.info("[HTTP] Connection lifetime: {d}ms", .{(conn_end - conn_start) / 1_000_000});
        const libc = @cImport(@cInclude("unistd.h"));
        _ = libc.close(connection.socket.handle);
    }
    handleRequest(allocator, state.io, state, connection, config) catch |err| {
        std.log.err("Request error: {}", .{err});
    };
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
    const conn_start = std.c.mach_absolute_time();
    defer {
        const conn_end = std.c.mach_absolute_time();
        std.log.info("[HTTP] Connection lifetime: {d}ms", .{(conn_end - conn_start) / 1_000_000});
        const libc = @cImport(@cInclude("unistd.h"));
        _ = libc.close(connection.socket.handle);
    }
    handleRequest(allocator, io, state, connection, config) catch |err| {
        std.log.err("Request error: {}", .{err});
    };
}
