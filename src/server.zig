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
const native_engine_mod = @import("native_engine.zig");

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

    // Install signal handlers for graceful shutdown.
    installSignalHandlers();
    std.log.info("Signal handlers installed (SIGTERM, SIGINT)", .{});

    // Start the accept loop in an async fiber.
    _ = io.async(acceptLoop, .{ allocator, io, &server_state, server_config });

    if (server_config.native) {
        // ------------------------------------------------------------------
        // Native (MLX-free) engine loop
        // ------------------------------------------------------------------
        std.log.info("Native mode: starting native engine loop", .{});
        nativeEngineLoop(&server_state);
    } else {
        // ------------------------------------------------------------------
        // MLX engine loop
        // ------------------------------------------------------------------
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
            .prefix_cache_max_entries = server_config.prefix_cache_entries,
        };

        var engine_loop = try engine.EngineLoop.init(engine_config);
        server_state.engine_loop = engine_loop;

        // Metal MoE: init and enable if flag set
        if (server_config.metal_moe) {
            const metal = @import("models/metal_moe.zig");
            if (metal.init()) {
                metal.setEnabled(true);
                std.log.info("Metal MoE: enabled", .{});
            } else {
                std.log.info("Metal MoE: init failed, using MLX fallback", .{});
            }
        }

        // Warmup: generate tokens to force backbone page-in before accepting connections.
        engine_loop.warmupBackbone();

        // Run the engine loop on the main thread (current fiber).
        engineLoopRun(&engine_loop);
    }

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
        const t_accept_start = std.c.mach_absolute_time();
        const connection = listener.accept(io) catch |err| {
            if (engine.isShutdownRequested()) break;
            std.log.err("Failed to accept connection: {}", .{err});
            continue;
        };
        const t_accept_done = std.c.mach_absolute_time();
        const accept_ms = @as(f64, @floatFromInt(t_accept_done - t_accept_start)) * 125.0 / 3_000_000.0;
        std.log.info("[ACCEPT] Connection accepted (accept took {d:.1}ms)", .{accept_ms});
        // Set socket to non-blocking mode for async fiber I/O.
        const fc = @cImport(@cInclude("fcntl.h"));
        const flags = fc.fcntl(connection.socket.handle, fc.F_GETFL, @as(c_int, 0));
        _ = fc.fcntl(connection.socket.handle, fc.F_SETFL, @as(c_int, flags | fc.O_NONBLOCK));
        const t_async_start = std.c.mach_absolute_time();
        _ = io.async(http.handleConnection, .{ allocator, io, server_state, connection, server_config });
        const t_async_done = std.c.mach_absolute_time();
        const async_ms = @as(f64, @floatFromInt(t_async_done - t_async_start)) * 125.0 / 3_000_000.0;
        if (async_ms > 10.0) {
            std.log.warn("[ACCEPT] io.async blocked for {d:.1}ms (thread pool full?)", .{async_ms});
        }
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

// ------------------------------------------------------------------
// Native (MLX-free) engine loop
// ------------------------------------------------------------------

fn nativeEngineLoop(server_state: *ServerState) void {
    const ne = server_state.native_engine.?;
    std.log.info("[NativeEngine] Native engine loop started", .{});
    defer std.log.info("[NativeEngine] Native engine loop exiting", .{});

    while (server_state.running and !engine.isShutdownRequested()) {
        const new_requests = server_state.request_queue.drainAll(server_state.allocator) catch {
            engine.threadSleepMs(1);
            continue;
        };
        defer server_state.allocator.free(new_requests);

        for (new_requests) |req| {
            if (req.isCancelled()) {
                req.completion.deliverError(server_state.io, "Request cancelled");
                continue;
            }
            processNativeRequest(server_state, ne, req);
        }

        if (new_requests.len == 0) {
            engine.threadSleepMs(1);
        }
    }
}

fn processNativeRequest(server_state: *ServerState, ne: *native_engine_mod.NativeEngine, req: *engine.RequestState) void {
    const engine_start = std.c.mach_absolute_time();
    req.start_time_ns = @intCast(engine_start);
    const queue_wait_ms = @as(f64, @floatFromInt(engine_start - req.queued_time_ns)) / 1_000_000.0;
    std.log.info("[NativeEngine] Request {d} dequeued, queue_wait={d:.1}ms", .{ req.id, queue_wait_ms });

    var sampler = root.sampling.SamplerConfig{
        .temperature = req.temperature,
        .top_k = req.top_k,
        .top_p = req.top_p,
        .prng = std.Random.DefaultPrng.init(req.seed),
    };

    if (req.streaming) {
        const StreamCtx = struct {
            req: *engine.RequestState,
            io: std.Io,
            allocator: std.mem.Allocator,
            tokenizer: root.tokenizer.TokenizerStrategy,
        };
        var stream_ctx = StreamCtx{
            .req = req,
            .io = server_state.io,
            .allocator = server_state.allocator,
            .tokenizer = server_state.tokenizer_strategy,
        };

        const cb = struct {
            fn callback(ctx_ptr: *anyopaque, token: u32, is_final: bool) void {
                const ctx: *StreamCtx = @ptrCast(@alignCast(ctx_ptr));
                const token_text = ctx.tokenizer.decode(&[_]u32{token}, ctx.allocator) catch "";
                defer if (token_text.len > 0) ctx.allocator.free(token_text);
                ctx.req.completion.deliverToken(ctx.io, token, token_text, is_final, if (is_final) .stop else null);
            }
        }.callback;

        const tokens = ne.generateWithCallback(
            req.prompt_tokens,
            req.max_tokens,
            &sampler,
            null,
            null,
            &stream_ctx,
            cb,
            null,
        ) catch |err| {
            std.log.err("[NativeEngine] generate failed: {}", .{err});
            req.completion.deliverError(server_state.io, "Generation failed");
            return;
        };
        defer server_state.allocator.free(tokens);
        req.token_count = @intCast(tokens.len);
    } else {
        const tokens = ne.generate(
            req.prompt_tokens,
            req.max_tokens,
            &sampler,
            null,
            null,
            null,
        ) catch |err| {
            std.log.err("[NativeEngine] generate failed: {}", .{err});
            req.completion.deliverError(server_state.io, "Generation failed");
            return;
        };
        defer server_state.allocator.free(tokens);

        var final_text = server_state.tokenizer_strategy.decode(tokens, server_state.allocator) catch {
            req.completion.deliverError(server_state.io, "Failed to decode tokens");
            return;
        };
        defer server_state.allocator.free(final_text);

        // Check stop strings
        if (req.stop_strings) |stop_strings| {
            for (stop_strings) |stop_str| {
                if (std.mem.indexOf(u8, final_text, stop_str)) |idx| {
                    final_text = final_text[0..idx];
                    break;
                }
            }
        }

        req.token_count = @intCast(tokens.len);
        req.completion.deliverToken(server_state.io, 0, final_text, true, .stop);
    }
}
