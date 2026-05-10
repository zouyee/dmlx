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
    };

    var engine_loop = engine.EngineLoop.init(engine_config);
    server_state.engine_loop = engine_loop;

    // Start the accept loop in an async fiber (may run on a worker thread).
    // The engine loop runs on the main thread so it shares the same MLX
    // stream that was used during model loading. MLX 0.31.2+ makes streams
    // thread-local; using the main thread avoids the worker-thread stream
    // mismatch bug.
    _ = io.async(acceptLoop, .{ allocator, io, &server_state, server_config });

    // Run the engine loop on the main thread (current fiber).
    engineLoopRun(&engine_loop);
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

    while (true) {
        const connection = listener.accept(io) catch |err| {
            std.log.err("Failed to accept connection: {}", .{err});
            continue;
        };
        // Set socket to non-blocking mode for async fiber I/O.
        const fc = @cImport(@cInclude("fcntl.h"));
        const flags = fc.fcntl(connection.socket.handle, fc.F_GETFL, @as(c_int, 0));
        _ = fc.fcntl(connection.socket.handle, fc.F_SETFL, @as(c_int, flags | fc.O_NONBLOCK));
        _ = io.async(http.handleConnection, .{ allocator, io, server_state, connection, server_config });
    }
}

fn engineLoopRun(loop: *engine.EngineLoop) void {
    loop.run();
}
