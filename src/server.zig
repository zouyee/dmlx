/// DMLX HTTP server — module entry point.
///
/// Re-exports all server sub-modules. Use `server.start()` to launch.
const std = @import("std");
const root = @import("root.zig");
const c = @import("mlx").c;
const ops = @import("mlx").ops;
const memory_mod = @import("memory.zig");
const scheduler_mod = @import("scheduler.zig");
const kvcache = @import("kvcache.zig");
const engine = @import("engine/root.zig");

const EagerContext = ops.EagerContext;

pub const config = @import("server/config.zig");
pub const state = @import("server/state.zig");
pub const http = @import("server/http.zig");
pub const sse = @import("server/sse.zig");
pub const openai = @import("server/openai.zig");
pub const streaming = @import("server/streaming.zig");
pub const anthropic = @import("server/anthropic.zig");
pub const utils = @import("server/utils.zig");
pub const tooling = @import("server/tooling.zig");

pub const ServerConfig = config.ServerConfig;
pub const KvStrategy = config.KvStrategy;
pub const KvTier = config.KvTier;
pub const KvQuant = config.KvQuant;
pub const ServerState = state.ServerState;

// ------------------------------------------------------------------
// Server entry point
// ------------------------------------------------------------------

pub fn start(allocator: std.mem.Allocator, io: std.Io, server_config: ServerConfig) !void {
    var server_state = try state.loadModel(allocator, io, server_config);
    // NOTE: server is long-running; deinit happens on process exit.
    // defer server_state.deinit();

    // Initialize the Scheduler now that state has a stable address.
    const max_prefill_tokens: usize = 512; // default chunked prefill limit
    server_state.scheduler = scheduler_mod.Scheduler.init(allocator, &(server_state.block_manager.?), max_prefill_tokens);

    // Initialize Server V2 fields.
    server_state.request_queue = engine.RequestQueue.init();
    server_state.engine_running = std.atomic.Value(bool).init(true);
    server_state.active_requests = std.atomic.Value(u32).init(0);
    server_state.next_request_id = std.atomic.Value(u64).init(1);

    // Initialize and start the EngineLoop as an async fiber.
    const mc = server_state.vtable.config;
    _ = memory_mod.autoMaxKvSize;
    const clamped_max_seq = 8192;

    const engine_config = engine.EngineConfig{
        .allocator = allocator,
        .io = io,
        .request_queue = &server_state.request_queue,
        .model = server_state.vtable,
        .ctx = server_state.ctx,
        .stream = server_state.stream,
        .tokenizer = server_state.tokenizer_strategy,
        .config_content = if (server_state.dsv4_model != null) blk: {
            const config_path = try std.fs.path.join(allocator, &[_][]const u8{ server_config.model_path, "config.json" });
            defer allocator.free(config_path);
            const content = try std.Io.Dir.cwd().readFileAlloc(io, config_path, allocator, .limited(1024 * 1024));
            break :blk content;
        } else null,
        .layer_config = if (server_state.dsv4_model == null) kvcache.LayerConfig{
            .batch_size = 1,
            .num_heads = mc.num_kv_heads,
            .num_kv_heads = mc.num_kv_heads,
            .head_dim = mc.head_dim,
            .max_seq_len = clamped_max_seq,
            .dtype = .float32,
        } else null,
        .num_layers = mc.num_layers,
        .dsv4_model = server_state.dsv4_model,
        .speculative_ngram = server_state.speculative_ngram,
    };

    var engine_loop = try engine.EngineLoop.init(engine_config);
    server_state.engine_loop = engine_loop;

    // Install signal handlers for graceful shutdown.
    installSignalHandlers();
    std.log.info("Signal handlers installed (SIGTERM, SIGINT)", .{});

    // Start the accept loop in a dedicated OS thread.
    // Previously used io.async (fiber), but Zig 0.16 Threaded IO backend has
    // ~20s fiber scheduling delays when the main thread is busy with GPU ops.
    // Using a real OS thread ensures accept() is responsive immediately.
    // See docs/analysis/socket-write-latency.md for the full investigation.
    const accept_thread = std.Thread.spawn(.{}, acceptLoopThread, .{
        allocator, &server_state, server_config,
    }) catch {
        std.log.err("Failed to spawn accept loop thread", .{});
        return error.ThreadSpawnFailed;
    };
    defer accept_thread.join();

    // Run the engine loop on the main thread (current fiber).
    engineLoopRun(&engine_loop);

    // Engine stopped — perform graceful shutdown.
    std.log.info("Engine stopped, initiating graceful shutdown...", .{});

    // Drain pending requests from the queue.
    const pending = server_state.request_queue.drainAll(allocator) catch blk: {
        const empty: []const *engine.RequestState = &[_]*engine.RequestState{};
        break :blk empty;
    };
    defer allocator.free(pending);
    for (pending) |req| {
        req.completion.deliverError(io, "Server shutting down");
    }

    // Wait for in-flight requests to complete (up to 30 seconds).
    waitForActiveRequests(io, &server_state);

    // Final cleanup.
    std.log.info("Shutting down, cleaning up resources...", .{});
    server_state.deinit();
    std.log.info("Shutdown complete.", .{});
}

fn acceptLoopThread(allocator: std.mem.Allocator, server_state: *ServerState, server_config: ServerConfig) void {
    const posix_c = @cImport({
        @cInclude("sys/socket.h");
        @cInclude("netinet/in.h");
        @cInclude("unistd.h");
        @cInclude("arpa/inet.h");
    });

    // Create listening socket directly via POSIX
    const listen_fd = posix_c.socket(posix_c.AF_INET, posix_c.SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std.log.err("Failed to create socket", .{});
        return;
    }
    defer _ = posix_c.close(listen_fd);

    // SO_REUSEADDR
    const one: c_int = 1;
    _ = posix_c.setsockopt(listen_fd, posix_c.SOL_SOCKET, posix_c.SO_REUSEADDR, &one, @sizeOf(c_int));

    // Bind
    var addr: posix_c.struct_sockaddr_in = std.mem.zeroes(posix_c.struct_sockaddr_in);
    addr.sin_family = posix_c.AF_INET;
    addr.sin_port = std.mem.nativeToBig(u16, server_config.port);
    addr.sin_addr.s_addr = posix_c.INADDR_ANY;

    if (posix_c.bind(listen_fd, @ptrCast(&addr), @sizeOf(@TypeOf(addr))) < 0) {
        std.log.err("Failed to bind to port {d}", .{server_config.port});
        return;
    }

    if (posix_c.listen(listen_fd, 128) < 0) {
        std.log.err("Failed to listen", .{});
        return;
    }

    std.log.info("DMLX server listening on http://0.0.0.0:{d}", .{server_config.port});

    while (!engine.isShutdownRequested()) {
        var client_addr: posix_c.struct_sockaddr_in = std.mem.zeroes(posix_c.struct_sockaddr_in);
        var addr_len: posix_c.socklen_t = @sizeOf(@TypeOf(client_addr));
        const client_fd = posix_c.accept(listen_fd, @ptrCast(&client_addr), &addr_len);
        if (client_fd < 0) {
            if (engine.isShutdownRequested()) break;
            continue;
        }
        const accept_time = std.c.mach_absolute_time();
        std.log.info("[Accept] Connection accepted fd={d}", .{client_fd});

        // Wrap the raw fd for handleConnectionThread (uses raw fd directly)
        const thread = std.Thread.spawn(.{}, http.handleConnectionThreadRaw, .{
            allocator, server_state, client_fd, server_config, accept_time,
        }) catch {
            _ = posix_c.close(client_fd);
            continue;
        };
        thread.detach();
    }
}

fn acceptLoop(allocator: std.mem.Allocator, io: std.Io, server_state: *ServerState, server_config: ServerConfig) void {
    const address = std.Io.net.IpAddress.parseIp4("0.0.0.0", server_config.port) catch |err| {
        std.log.err("Failed to parse address: {}", .{err});
        return;
    };
    var listener = address.listen(io, .{ .reuse_address = true }) catch |err| {
        std.log.err("Failed to listen: {}", .{err});
        return;
    };
    defer listener.deinit(io);

    std.log.info("DMLX server listening on http://0.0.0.0:{d}", .{server_config.port});

    while (!engine.isShutdownRequested()) {
        const connection = listener.accept(io) catch |err| {
            if (engine.isShutdownRequested()) break;
            std.log.err("Failed to accept connection: {}", .{err});
            continue;
        };
        // Spawn a dedicated OS thread for each connection.
        // Previously used io.async (fiber-based), but Zig 0.16 Threaded IO backend
        // has ~20s fiber scheduling delays when the main thread is busy with GPU ops.
        // OS threads are scheduled by the kernel independently, avoiding this issue.
        // See docs/analysis/socket-write-latency.md for the full investigation.
        const thread = std.Thread.spawn(.{}, http.handleConnectionThread, .{
            allocator, server_state, connection, server_config,
        }) catch |err| {
            std.log.err("Failed to spawn connection thread: {}", .{err});
            const libc = @cImport(@cInclude("unistd.h"));
            _ = libc.close(connection.socket.handle);
            continue;
        };
        thread.detach();
    }

    std.log.info("Accept loop stopped, no longer accepting new connections.", .{});
}

fn engineLoopRun(loop: *engine.EngineLoop) void {
    loop.run();
}

fn installSignalHandlers() void {
    const handler = struct {
        fn handle(signo: c_int) callconv(.c) void {
            _ = signo;
            // NOTE: Only async-signal-safe operations here. No logging, no allocations.
            engine.requestShutdown();
        }
    };

    const csig = @cImport(@cInclude("signal.h"));
    var act: csig.struct_sigaction = .{};
    act.__sigaction_u.__sa_handler = @ptrCast(&handler.handle);
    _ = csig.sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    _ = csig.sigaction(csig.SIGTERM, &act, null);
    _ = csig.sigaction(csig.SIGINT, &act, null);
}

fn waitForActiveRequests(io: std.Io, server_state: *ServerState) void {
    const max_wait_ns: i96 = 30_000_000_000; // 30 seconds
    const start_time = std.Io.Timestamp.now(io, .awake);

    while (server_state.active_requests.load(.acquire) > 0) {
        const now = std.Io.Timestamp.now(io, .awake);
        const elapsed_ns = now.durationTo(start_time).toNanoseconds();
        if (elapsed_ns >= max_wait_ns) {
            const remaining = server_state.active_requests.load(.acquire);
            std.log.warn("Graceful shutdown timeout: {d} request(s) still in-flight", .{remaining});
            break;
        }
        engine.threadSleepMs(100);
    }

    if (server_state.active_requests.load(.acquire) == 0) {
        std.log.info("All in-flight requests completed.", .{});
    }
}
