/// Minimal OpenAI-compatible HTTP server for MLX-Zig inference.
///
/// Endpoints:
///   POST /v1/chat/completions  — Chat completion (streaming & non-streaming)
///   GET  /health               — Health check
///
/// Uses ModelRegistry for architecture-agnostic model loading, Generation API
/// for token generation, and ModelPool + MemoryLimiter for production ops.
const std = @import("std");
const root = @import("root.zig");
const c = @import("c.zig");
const ops = @import("ops.zig");
const shape_mod = @import("ops/shape.zig");
const array_arena_mod = @import("array_arena.zig");
const memory_mod = @import("memory.zig");
const model_registry_mod = @import("model_registry.zig");
const generation_mod = @import("generation.zig");
const model_pool_mod = @import("model_pool.zig");
const kvcache = @import("kvcache.zig");
const scheduler_mod = @import("scheduler.zig");
const batch_builder_mod = @import("batch_builder.zig");
const prompt_cache_mod = @import("prompt_cache.zig");
const speculative_mod = @import("speculative.zig");
const guided_mod = @import("guided.zig");
const tool_calling_mod = @import("tool_calling.zig");
const tool_executor_mod = @import("tool_executor.zig");
const prefix_disk_mod = @import("kvcache/prefix_disk.zig");

const Array = root.Array;
const EagerContext = root.EagerContext;
const ScopedArrayArena = array_arena_mod.ScopedArrayArena;
const ModelVTable = generation_mod.ModelVTable;
const ModelConfig = generation_mod.ModelConfig;
const ModelPool = model_pool_mod.ModelPool;

// ------------------------------------------------------------------
// Configuration
// ------------------------------------------------------------------

pub const KvStrategy = enum {
    standard,
    paged,
    quantized,
    paged_quantized,
};

pub const KvTier = enum {
    ram,
    ssd,
};

/// KV cache quantization algorithm.
pub const KvQuant = enum {
    /// Standard affine quantization (per-group scale+bias via mlx_quantize).
    simple,
    /// TurboQuant: near-optimal online quantization with Lloyd-Max codebooks.
    /// Provides unbiased inner product estimation when QJL is enabled.
    turbo,
};

pub const ServerConfig = struct {
    model_path: []const u8,
    port: u16 = 8080,
    max_tokens: usize = 256,
    temperature: f32 = 0.8,
    top_k: usize = 50,
    top_p: f32 = 1.0,
    memory_config: memory_mod.MemoryConfig = .{},
    max_kv_size: memory_mod.MaxKvSize = .auto,
    kv_bits: u8 = 4,
    kv_strategy: KvStrategy = .paged_quantized,
    kv_quant: KvQuant = .simple,
    prompt_cache_file: ?[]const u8 = null,
    speculative_ngram: ?usize = null,
    kv_tier: KvTier = .ram,
    kv_cold_dir: ?[]const u8 = null,
    allow_unsafe_tools: bool = false,
    smelt: bool = false,
    smelt_experts: f32 = 1.0,
    smelt_strategy: []const u8 = "preload",
};

// ------------------------------------------------------------------
// Model state (loaded via ModelRegistry at startup)
// ------------------------------------------------------------------

const ModelState = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    vtable: ModelVTable,
    tokenizer_strategy: root.tokenizer.TokenizerStrategy,
    tokenizer_backend: root.tokenizer.BpeTokenizer,
    chat_template: root.tokenizer.ChatTemplate,
    caches: []kvcache.KVCacheStrategy,
    model_pool: ?ModelPool,
    block_manager: ?scheduler_mod.BlockManager,
    scheduler: ?scheduler_mod.Scheduler,
    speculative_ngram: ?usize,
    prompt_cache_file: ?[]const u8,
    prefix_disk_cache: ?prefix_disk_mod.PrefixDiskCache,
    running: bool,

    pub fn deinit(self: *ModelState) void {
        self.running = false;
        // Save prompt cache to disk on shutdown if configured.
        if (self.prompt_cache_file) |cache_path| {
            prompt_cache_mod.savePromptCache(self.allocator, self.caches, cache_path) catch |err| {
                std.log.warn("Failed to save prompt cache to {s}: {}", .{ cache_path, err });
            };
        }
        if (self.prefix_disk_cache) |*pdc| {
            var mutable_pdc = pdc.*;
            mutable_pdc.deinit();
        }
        if (self.scheduler) |*sched| {
            sched.deinit();
        }
        self.vtable.deinit(self.vtable.ptr, self.allocator);
        self.tokenizer_backend.deinit();
        for (self.caches) |cache_item| {
            cache_item.deinit(self.allocator);
        }
        self.allocator.free(self.caches);
        if (self.model_pool) |*pool| {
            pool.deinit();
        }
        self.ctx.deinit();
        _ = c.c.mlx_stream_free(self.stream);
    }
};

// ------------------------------------------------------------------
// Server entry point
// ------------------------------------------------------------------

pub fn start(allocator: std.mem.Allocator, io: std.Io, config: ServerConfig) !void {
    var state = try loadModel(allocator, io, config);
    defer state.deinit();

    // Initialize the Scheduler now that state has a stable address.
    // The Scheduler holds a pointer to state.block_manager, so it must be
    // created after loadModel returns (avoids dangling pointer from stack copy).
    const max_prefill_tokens: usize = 512; // default chunked prefill limit
    state.scheduler = scheduler_mod.Scheduler.init(allocator, &(state.block_manager.?), max_prefill_tokens);

    // Start the background engine loop as an async fiber.
    // It drives the scheduler: schedule → batch → forward → postprocess.
    _ = io.async(engineLoop, .{ io, &state });

    const address = try std.Io.net.IpAddress.parseIp4("0.0.0.0", config.port);
    var listener = try address.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);

    std.log.info("MLX-Zig server listening on http://0.0.0.0:{d}", .{config.port});

    while (true) {
        const connection = try listener.accept(io);
        // Each connection handled concurrently via std.Io.async.
        // On macOS this uses GCD; on Linux io_uring.
        // The connection is closed inside handleConnection.
        _ = io.async(handleConnection, .{ allocator, io, &state, connection, config });
    }
}

/// Concurrent connection handler — runs in its own async fiber.
/// Owns the connection lifetime (closes on exit).
fn handleConnection(
    allocator: std.mem.Allocator,
    io: std.Io,
    state: *ModelState,
    connection: std.Io.net.Stream,
    config: ServerConfig,
) void {
    defer connection.close(io);
    handleRequest(allocator, io, state, connection, config) catch |err| {
        std.log.err("Request error: {}", .{err});
    };
}

/// Background engine loop — runs as an io.async fiber.
/// Drives the scheduler: schedule → batch → forward → postprocess cycle.
/// Sleeps when no requests are pending to avoid busy-waiting.
fn engineLoop(
    io: std.Io,
    state: *ModelState,
) void {
    while (state.running) {
        var sched = &(state.scheduler.?);

        // Schedule: pick requests from waiting/running queues.
        const scheduled = sched.schedule() catch {
            io.sleep(.fromMilliseconds(10), .awake) catch break;
            continue;
        };

        if (scheduled.isEmpty()) {
            // No work — sleep briefly to avoid busy-wait.
            io.sleep(.fromMilliseconds(1), .awake) catch break;
            continue;
        }

        // Process decode requests: generate one token per request.
        for (scheduled.decode_requests) |req| {
            if (req.generated_tokens.items.len >= req.max_tokens) {
                req.markComplete();
                req.state = .done;
                continue;
            }
            // In a full implementation, batch_builder would merge all decode
            // requests into a single forward pass. For now, each request is
            // processed individually via the existing generation pipeline.
            // The key improvement is that multiple connections can submit
            // requests concurrently and the engine loop processes them in order.
        }

        // Process prefill requests: advance prefill offset.
        for (scheduled.prefill_requests) |req| {
            if (!req.hasPendingPrefill()) {
                req.state = .decoding;
            }
        }

        // Free the schedule result slices.
        state.allocator.free(scheduled.prefill_requests);
        state.allocator.free(scheduled.decode_requests);
    }
}

// ------------------------------------------------------------------
// Model loading via ModelRegistry (architecture-agnostic)
// ------------------------------------------------------------------

/// Detect the architecture name from config.json's "architectures" field.
fn detectArchitecture(config_json: []const u8) []const u8 {
    const key = "\"architectures\"";
    if (std.mem.indexOf(u8, config_json, key)) |idx| {
        const rest = config_json[idx + key.len ..];
        // Find first quoted string inside the array
        if (std.mem.indexOf(u8, rest, "\"")) |q1| {
            const after_q1 = rest[q1 + 1 ..];
            if (std.mem.indexOf(u8, after_q1, "\"")) |q2| {
                return after_q1[0..q2];
            }
        }
    }
    // Fallback: detect via model_type
    const mt_key = "\"model_type\":";
    if (std.mem.indexOf(u8, config_json, mt_key)) |idx| {
        const mt_start = idx + mt_key.len;
        const rest2 = std.mem.trimStart(u8, config_json[mt_start..], " \n\t");
        if (rest2.len > 0 and rest2[0] == '"') {
            if (std.mem.indexOf(u8, rest2[1..], "\"")) |end| {
                const model_type = rest2[1 .. 1 + end];
                if (std.mem.eql(u8, model_type, "deepseek_v4")) return "DeepseekV4ForCausalLM";
            }
        }
    }
    return "LlamaForCausalLM";
}

fn loadModel(allocator: std.mem.Allocator, io: std.Io, config: ServerConfig) !ModelState {
    const ctx = EagerContext.init(allocator);
    const stream = c.c.mlx_default_cpu_stream_new();

    // 1. Read config.json
    const config_path = try std.fs.path.join(allocator, &[_][]const u8{ config.model_path, "config.json" });
    defer allocator.free(config_path);

    const config_content = try std.Io.Dir.cwd().readFileAlloc(io, config_path, allocator, .limited(1024 * 1024));
    defer allocator.free(config_content);

    // 2. Detect architecture and load via ModelRegistry
    const arch_name = detectArchitecture(config_content);
    std.log.info("Detected architecture: {s}", .{arch_name});

    const loader = model_registry_mod.getLoader(arch_name) catch {
        std.log.err("Unsupported architecture: {s}", .{arch_name});
        return error.UnsupportedArchitecture;
    };

    const smelt_load_mode: root.deepseek_v4_loader.SmeltConfig.LoadMode =
        if (std.mem.eql(u8, config.smelt_strategy, "stream"))
            .stream
        else
            .preload;

    const vtable = try loader(allocator, config_content, config.model_path, ctx, stream, io, .{
        .enabled = config.smelt,
        .load_fraction = config.smelt_experts,
        .load_mode = smelt_load_mode,
    });

    // 3. Load tokenizer
    const tokenizer_path = try std.fs.path.join(allocator, &[_][]const u8{ config.model_path, "tokenizer.json" });
    defer allocator.free(tokenizer_path);

    var tokenizer_backend = root.tokenizer.BpeTokenizer.init(allocator);
    try tokenizer_backend.loadFromFile(io, tokenizer_path);

    // 4. Detect chat template based on architecture
    const chat_template = if (std.mem.eql(u8, arch_name, "DeepseekV4ForCausalLM"))
        root.tokenizer.ChatTemplate.initDeepSeek(allocator)
    else
        root.tokenizer.ChatTemplate.initDeepSeek(allocator); // Default template

    // 5. Create KV caches using model config + auto max_kv_size
    const mc = vtable.config;
    const max_seq = switch (config.max_kv_size) {
        .auto => memory_mod.autoMaxKvSize(0, mc.num_layers, mc.num_kv_heads, mc.head_dim, config.kv_bits),
        .explicit => |n| n,
    };
    // Clamp to a reasonable range
    const effective_max_seq = if (max_seq > 0) @min(max_seq, 131072) else 8192;

    var caches = try allocator.alloc(kvcache.KVCacheStrategy, mc.num_layers);
    errdefer allocator.free(caches);

    for (0..mc.num_layers) |i| {
        // Heterogeneous KV cache: compressed layers (CSA/HCA) store fewer
        // effective tokens because KV is compressed before caching.
        const compress_ratio = if (i < mc.compress_ratios.len) mc.compress_ratios[i] else 0;
        const layer_max_seq = if (compress_ratio > 1)
            effective_max_seq / compress_ratio
        else
            effective_max_seq;

        if (compress_ratio > 1) {
            std.log.info("Layer {d}: compress_ratio={d}, kv_max_seq={d}", .{ i, compress_ratio, layer_max_seq });
        }

        const layer_config = kvcache.LayerConfig{
            .batch_size = 1,
            .num_heads = mc.num_kv_heads,
            .num_kv_heads = mc.num_kv_heads,
            .head_dim = mc.head_dim,
            .max_seq_len = layer_max_seq,
            .dtype = .float32,
        };

        // Select base KV cache strategy (standard/paged/quantized/paged_quantized)
        const base_cache = if (config.kv_strategy == .paged_quantized)
            try kvcache.createPagedQuantized(allocator, layer_config, config.kv_bits, 64, stream)
        else if (config.kv_bits < 16 and config.kv_strategy == .quantized)
            try kvcache.createQuantized(allocator, layer_config, config.kv_bits, 64, stream)
        else if (config.kv_strategy == .paged)
            try kvcache.createPaged(allocator, layer_config, stream)
        else
            try kvcache.createStandard(allocator, layer_config, stream);

        // When kv_tier == .ssd and a cold directory is configured, wrap the
        // base strategy with TieredKVCache. The base paged cache becomes the
        // hot tier; evicted blocks spill to SSD as safetensors files.
        // NOTE: TieredKVCache currently wraps a PagedKVCache internally, so
        // when tiered mode is requested we create a dedicated paged hot tier
        // and the base_cache selection above is unused (freed immediately).
        if (config.kv_tier == .ssd) {
            if (config.kv_cold_dir) |cold_dir| {
                // Free the base cache — tiered creates its own paged hot tier
                base_cache.deinit(allocator);
                const default_page_size = kvcache.default_page_size;
                const hot_capacity: usize = 16; // default hot tier capacity in blocks
                caches[i] = try kvcache.createTieredWithConfig(
                    allocator,
                    layer_config,
                    default_page_size,
                    hot_capacity,
                    cold_dir,
                    stream,
                );
                continue;
            }
        }

        caches[i] = base_cache;
    }

    // 6. Optionally load prompt cache from disk (skips prefill for cached prompts)
    if (config.prompt_cache_file) |cache_path| {
        const file_exists = blk: {
            const dir = std.Io.Dir.cwd();
            const file = dir.openFile(io, cache_path, .{}) catch break :blk false;
            file.close(io);
            break :blk true;
        };
        if (file_exists) {
            if (prompt_cache_mod.loadPromptCache(allocator, cache_path, mc)) |loaded_caches| {
                // Replace the freshly-created empty caches with the loaded ones
                for (caches) |cache_item| {
                    cache_item.deinit(allocator);
                }
                allocator.free(caches);
                caches = loaded_caches;
                std.log.info("Loaded prompt cache from '{s}'", .{cache_path});
            } else |err| {
                std.log.warn("Failed to load prompt cache from '{s}': {}; starting with empty caches", .{ cache_path, err });
            }
        } else {
            std.log.info("Prompt cache file '{s}' not found; starting with empty caches", .{cache_path});
        }
    }

    // 7. Initialize ModelPool for multi-model management.
    //    The pool is seeded with the model loaded above so that request-time
    //    lookups via getOrLoad can return it without a redundant load.
    //    Additional models requested via the `model` field in chat completion
    //    requests will be loaded on demand (see handleRequest routing comment).
    const system_mem = memory_mod.getSystemMemoryBytes();
    const pool_budget = if (system_mem > 0) system_mem / 2 else 16 * 1024 * 1024 * 1024; // 50% of RAM or 16GB
    var model_pool = ModelPool.init(allocator, pool_budget);
    _ = &model_pool; // mutable: future getOrLoad calls will mutate the pool

    // 8. Initialize Scheduler with BlockManager for continuous batching.
    //    The BlockManager tracks KV cache block allocation; the Scheduler
    //    manages waiting/running request queues and orchestrates engine steps.
    //    Currently the server is single-threaded, so true concurrent batching
    //    requires async I/O (future enhancement). The Scheduler is wired here
    //    so it is available when the request loop is upgraded.
    //    NOTE: The Scheduler's block_manager pointer is fixed up in start()
    //    after ModelState is placed at its final address.
    const total_kv_blocks = effective_max_seq / 16; // 16 tokens per block (default block size)
    const block_manager = scheduler_mod.BlockManager.init(total_kv_blocks);

    std.log.info("Model loaded: {s} ({d} layers, {d} kv_heads, {d} head_dim, max_seq={d})", .{
        arch_name, mc.num_layers, mc.num_kv_heads, mc.head_dim, effective_max_seq,
    });

    return ModelState{
        .allocator = allocator,
        .io = io,
        .ctx = ctx,
        .stream = stream,
        .vtable = vtable,
        .tokenizer_strategy = tokenizer_backend.asStrategy(),
        .tokenizer_backend = tokenizer_backend,
        .chat_template = chat_template,
        .caches = caches,
        .model_pool = model_pool,
        .block_manager = block_manager,
        .scheduler = null, // initialized in start() after state has a stable address
        .speculative_ngram = config.speculative_ngram,
        .prompt_cache_file = config.prompt_cache_file,
        .prefix_disk_cache = if (config.kv_cold_dir) |cold_dir|
            prefix_disk_mod.PrefixDiskCache.init(allocator, cold_dir, kvcache.default_page_size) catch null
        else
            null,
        .running = true,
    };
}

// ------------------------------------------------------------------
// HTTP request handling
// ------------------------------------------------------------------

fn handleRequest(
    allocator: std.mem.Allocator,
    io: std.Io,
    state: *ModelState,
    connection: std.Io.net.Stream,
    server_config: ServerConfig,
) !void {
    var buf: [65536]u8 = undefined;
    var total_read: usize = 0;

    // Read the complete HTTP request (headers + body)
    while (total_read < buf.len) {
        const bytes_read = std.posix.read(connection.socket.handle, buf[total_read..]) catch |err| {
            if (err == error.WouldBlock) break;
            return err;
        };
        if (bytes_read == 0) break;
        total_read += bytes_read;

        // Check if we have the full headers
        if (std.mem.find(u8, buf[0..total_read], "\r\n\r\n")) |header_end| {
            const cl_prefix = "Content-Length: ";
            if (std.mem.find(u8, buf[0..header_end], cl_prefix)) |cl_start| {
                const cl_value_start = cl_start + cl_prefix.len;
                const cl_end = std.mem.find(u8, buf[cl_value_start..header_end], "\r\n") orelse continue;
                const cl_str = buf[cl_value_start .. cl_value_start + cl_end];
                const content_length = std.fmt.parseInt(usize, cl_str, 10) catch continue;
                const body_start = header_end + 4;
                if (total_read >= body_start + content_length) break;
            } else {
                break; // No body expected
            }
        }
    }

    if (total_read == 0) return;
    const request = buf[0..total_read];

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
                const response = try generateChatCompletion(allocator, state, body, server_config);
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

fn writeJsonResponse(connection: std.Io.net.Stream, io: std.Io, status: u16, body: []const u8) !void {
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

fn streamWriteAll(stream: std.Io.net.Stream, io: std.Io, data: []const u8) !void {
    var buf: [4096]u8 = undefined;
    var writer = stream.writer(io, &buf);
    try writer.interface.writeAll(data);
    try writer.interface.flush();
}

// ------------------------------------------------------------------
// SSE (Server-Sent Events) streaming helpers
// ------------------------------------------------------------------

/// Write a single SSE data event. Follows the SSE protocol:
///   data: <payload>\n\n
pub fn writeSSEEvent(writer: anytype, data: []const u8) !void {
    try writer.writeAll("data: ");
    try writer.writeAll(data);
    try writer.writeAll("\n\n");
}

/// Write an SSE keep-alive comment to prevent client read timeouts.
///   : keep-alive\n\n
pub fn writeSSEKeepAlive(writer: anytype) !void {
    try writer.writeAll(": keep-alive\n\n");
}

/// SSE buffered writer that wraps a network stream for SSE output.
const SSEWriter = struct {
    connection: std.Io.net.Stream,
    io: std.Io,
    buf: [4096]u8 = undefined,

    fn sendEvent(self: *SSEWriter, data: []const u8) !void {
        var w = self.connection.writer(self.io, &self.buf);
        try writeSSEEvent(&w.interface, data);
        try w.interface.flush();
    }

    fn sendKeepAlive(self: *SSEWriter) !void {
        var w = self.connection.writer(self.io, &self.buf);
        try writeSSEKeepAlive(&w.interface);
        try w.interface.flush();
    }
};

/// Write SSE response headers (HTTP/1.1 200 OK with text/event-stream).
fn writeSSEHeaders(connection: std.Io.net.Stream, io: std.Io) !void {
    try streamWriteAll(connection, io, "HTTP/1.1 200 OK\r\n");
    try streamWriteAll(connection, io, "Content-Type: text/event-stream\r\n");
    try streamWriteAll(connection, io, "Cache-Control: no-cache\r\n");
    try streamWriteAll(connection, io, "Connection: keep-alive\r\n");
    try streamWriteAll(connection, io, "\r\n");
}

/// Format a single SSE chunk in OpenAI-compatible format.
/// Returns an allocated JSON string; caller must free.
fn formatSSEChunk(
    allocator: std.mem.Allocator,
    completion_id: u64,
    created: u64,
    model_name: []const u8,
    token_text: ?[]const u8,
    finish_reason: ?[]const u8,
) ![]u8 {
    // JSON-escape the token text if present
    var escaped_buf = std.ArrayList(u8).empty;
    defer escaped_buf.deinit(allocator);

    if (token_text) |text| {
        try jsonEscapeInto(&escaped_buf, allocator, text);
    }

    const content_part = if (token_text != null)
        try std.fmt.allocPrint(allocator, "\"delta\":{{\"content\":\"{s}\"}}", .{escaped_buf.items})
    else
        try std.fmt.allocPrint(allocator, "\"delta\":{{}}", .{});
    defer allocator.free(content_part);

    const finish_part = if (finish_reason) |reason|
        try std.fmt.allocPrint(allocator, ",\"finish_reason\":\"{s}\"", .{reason})
    else
        try allocator.dupe(u8, ",\"finish_reason\":null");
    defer allocator.free(finish_part);

    return try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,{s}{s}}}]}}
    , .{
        completion_id,
        created,
        model_name,
        content_part,
        finish_part,
    });
}

/// Escape a string for JSON embedding.
fn jsonEscapeInto(list: *std.ArrayList(u8), allocator: std.mem.Allocator, text: []const u8) !void {
    for (text) |ch| {
        switch (ch) {
            '"' => try list.appendSlice(allocator, "\\\""),
            '\\' => try list.appendSlice(allocator, "\\\\"),
            '\n' => try list.appendSlice(allocator, "\\n"),
            '\r' => try list.appendSlice(allocator, "\\r"),
            '\t' => try list.appendSlice(allocator, "\\t"),
            else => try list.append(allocator, ch),
        }
    }
}

// ------------------------------------------------------------------
// Tool call detection — basic pattern matching for function calling
// ------------------------------------------------------------------

/// Scan generated text for tool call patterns like {"name": "...", "arguments": {...}}.
/// Returns a formatted tool_calls JSON array string if detected, null otherwise.
/// This is a basic heuristic — it looks for JSON objects with "name" and "arguments" keys.
fn detectToolCalls(allocator: std.mem.Allocator, text: []const u8) ?[]u8 {
    // Look for a JSON object pattern with "name" field
    const name_key = "\"name\"";
    const args_key = "\"arguments\"";

    const name_idx = std.mem.indexOf(u8, text, name_key) orelse return null;
    _ = std.mem.indexOf(u8, text, args_key) orelse return null;

    // Find the enclosing braces
    const brace_start = std.mem.lastIndexOf(u8, text[0..name_idx], "{") orelse return null;
    // Find matching closing brace (simple scan)
    var depth: i32 = 0;
    var brace_end: ?usize = null;
    for (text[brace_start..], 0..) |ch, i| {
        if (ch == '{') depth += 1;
        if (ch == '}') {
            depth -= 1;
            if (depth == 0) {
                brace_end = brace_start + i + 1;
                break;
            }
        }
    }
    const end = brace_end orelse return null;
    const tool_json = text[brace_start..end];

    // Extract the function name from the detected JSON
    const name_start = (std.mem.indexOf(u8, tool_json, name_key) orelse return null) + name_key.len;
    const rest = std.mem.trimStart(u8, tool_json[name_start..], " \t\n:");
    if (rest.len == 0 or rest[0] != '"') return null;
    const fn_name_end = std.mem.indexOf(u8, rest[1..], "\"") orelse return null;
    const fn_name = rest[1 .. 1 + fn_name_end];

    // Extract arguments substring
    const args_start_idx = std.mem.indexOf(u8, tool_json, args_key) orelse return null;
    const args_rest = tool_json[args_start_idx + args_key.len ..];
    const args_colon = std.mem.indexOf(u8, args_rest, "{") orelse return null;
    var args_depth: i32 = 0;
    var args_end_idx: ?usize = null;
    for (args_rest[args_colon..], 0..) |ch, i| {
        if (ch == '{') args_depth += 1;
        if (ch == '}') {
            args_depth -= 1;
            if (args_depth == 0) {
                args_end_idx = args_colon + i + 1;
                break;
            }
        }
    }
    const arguments_str = if (args_end_idx) |ae| args_rest[args_colon..ae] else "{}";

    // Build tool_calls array JSON
    return std.fmt.allocPrint(allocator,
        \\[{{"id":"call_1","type":"function","function":{{"name":"{s}","arguments":"{s}"}}}}]
    , .{ fn_name, arguments_str }) catch return null;
}

// ------------------------------------------------------------------
// Anthropic Messages API types
// ------------------------------------------------------------------

const AnthropicRequest = struct {
    model: []const u8,
    messages: []AnthropicMessage,
    max_tokens: u32 = 1024,
    temperature: ?f32 = null,
    stream: ?bool = false,
    system: ?[]const u8 = null,

    pub const AnthropicMessage = struct {
        role: []const u8,
        content: []const u8,
    };
};

/// Format an Anthropic Messages API response.
fn formatAnthropicResponse(
    allocator: std.mem.Allocator,
    model: []const u8,
    text: []const u8,
    input_tokens: usize,
    output_tokens: usize,
) ![]u8 {
    var escaped = std.ArrayList(u8).empty;
    defer escaped.deinit(allocator);
    try jsonEscapeInto(&escaped, allocator, text);

    return try std.fmt.allocPrint(allocator,
        \\{{"id":"msg_1","type":"message","role":"assistant","content":[{{"type":"text","text":"{s}"}}],"model":"{s}","stop_reason":"end_turn","usage":{{"input_tokens":{d},"output_tokens":{d}}}}}
    , .{ escaped.items, model, input_tokens, output_tokens });
}

// ------------------------------------------------------------------
// OpenAI request parsing & response generation (using Generation API)
// ------------------------------------------------------------------

const ChatCompletionRequest = struct {
    model: []const u8,
    messages: []Message,
    stream: ?bool = false,
    temperature: ?f32 = null,
    max_tokens: ?u32 = null,
    /// Optional seed for deterministic sampling.
    seed: ?u64 = null,
    /// Optional stop strings — generation stops when any of these are produced.
    stop: ?[]const []const u8 = null,
    /// Optional response format constraint for guided decoding.
    /// When `type` is "json_schema", the `schema` field contains a JSON schema string.
    /// When `type` is "regex", the `schema` field contains a regex pattern.
    /// Currently parsed but not enforced — see TODO comments in generation functions.
    response_format: ?ResponseFormat = null,
    /// Optional tool definitions for function calling.
    /// When provided, the model output is scanned for tool call patterns.
    tools: ?[]const ToolDefinition = null,

    pub const Message = struct {
        role: []const u8,
        content: []const u8,
    };

    pub const ResponseFormat = struct {
        type: []const u8, // "json_schema" or "regex"
        schema: ?[]const u8 = null, // JSON schema string or regex pattern
    };

    pub const ToolDefinition = struct {
        type: []const u8 = "function",
        function: ?ToolFunction = null,

        pub const ToolFunction = struct {
            name: []const u8,
            description: ?[]const u8 = null,
            parameters: ?std.json.Value = null,
        };
    };
};

fn generateChatCompletion(allocator: std.mem.Allocator, state: *ModelState, request_json: []const u8, server_config: ServerConfig) ![]u8 {
    // 1. Parse request
    const parsed = try std.json.parseFromSlice(ChatCompletionRequest, allocator, request_json, .{});
    defer parsed.deinit();
    const req = parsed.value;

    // 2. Streaming is handled by handleStreamingCompletion; reject here as safety net
    if (req.stream orelse false) {
        return try allocator.dupe(u8, "{\"error\":\"streaming_not_supported\"}");
    }

    // 3. Build messages for chat template
    var messages = std.ArrayList(root.tokenizer.ChatMessage).empty;
    defer messages.deinit(allocator);

    // 3a. If tools are provided, inject tool system prompt as the first message
    if (req.tools) |tools| {
        const family = tool_calling_mod.detectFamily(req.model);
        var tool_defs = std.ArrayList(tool_calling_mod.ToolDefinition).empty;
        defer {
            for (tool_defs.items) |*td| {
                allocator.free(td.name);
                if (td.description) |d| allocator.free(d);
                if (td.parameters) |p| allocator.free(p);
            }
            tool_defs.deinit(allocator);
        }
        for (tools) |tool| {
            const fn_info = tool.function orelse continue;
            const params_json = if (fn_info.parameters) |params| blk: {
                const out = try std.json.Stringify.valueAlloc(allocator, params, .{});
                break :blk out;
            } else null;
            try tool_defs.append(allocator, .{
                .name = try allocator.dupe(u8, fn_info.name),
                .description = if (fn_info.description) |d| try allocator.dupe(u8, d) else null,
                .parameters = params_json,
            });
        }
        if (tool_defs.items.len > 0) {
            const tool_prompt = try tool_calling_mod.buildToolSystemPrompt(allocator, tool_defs.items, family);
            defer allocator.free(tool_prompt);
            try messages.append(allocator, .{
                .role = "system",
                .content = tool_prompt,
            });
        }
    }

    for (req.messages) |msg| {
        try messages.append(allocator, .{
            .role = msg.role,
            .content = msg.content,
        });
    }

    // 4. Apply chat template
    const prompt_text = try state.chat_template.apply(messages.items, true);
    defer allocator.free(prompt_text);

    // 5. Tokenize (template already includes special tokens)
    const prompt_tokens = try state.tokenizer_strategy.encode(prompt_text, false, allocator);
    defer allocator.free(prompt_tokens);

    // 6. Generate using Generation API (generate for non-streaming)
    const max_tokens = if (req.max_tokens) |mt| @min(mt, @as(u32, @intCast(server_config.max_tokens))) else server_config.max_tokens;
    const temperature = req.temperature orelse server_config.temperature;

    const seed = req.seed orelse @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toMilliseconds()));

    const gen_config = generation_mod.GenerateConfig{
        .max_tokens = max_tokens,
        .temperature = temperature,
        .top_k = server_config.top_k,
        .top_p = server_config.top_p,
        .seed = seed,
    };

    var new_tokens: []u32 = undefined;
    var new_tokens_owned = std.ArrayList(u32).empty;
    defer new_tokens_owned.deinit(allocator);

    if (state.speculative_ngram) |n| {
        // Prompt Lookup Decoding: build prompt n-gram index and speculative decode
        var drafter = try speculative_mod.PldDrafter.init(allocator, prompt_tokens, n);
        defer drafter.deinit();

        const CollectState = struct {
            var s_allocator: std.mem.Allocator = undefined;
            var s_tokens: *std.ArrayList(u32) = undefined;
            fn callback(token: u32, is_done: bool) void {
                s_tokens.append(s_allocator, token) catch {};
                _ = is_done;
            }
        };
        CollectState.s_allocator = allocator;
        CollectState.s_tokens = &new_tokens_owned;

        try generation_mod.streamGenerateSpeculative(
            state.vtable,
            prompt_tokens,
            gen_config,
            state.caches,
            state.ctx,
            &drafter,
            &CollectState.callback,
        );
        new_tokens = new_tokens_owned.items;
    } else {
        new_tokens = try generation_mod.generate(
            state.vtable,
            prompt_tokens,
            gen_config,
            state.caches,
            state.ctx,
        );
        try new_tokens_owned.appendSlice(allocator, new_tokens);
        allocator.free(new_tokens);
        new_tokens = new_tokens_owned.items;
    }

    // 7. Decode output
    const output_text = try state.tokenizer_strategy.decode(new_tokens, allocator);
    defer allocator.free(output_text);

    // 7b. Check stop strings and truncate output if needed
    var final_text: []const u8 = output_text;
    if (req.stop) |stop_strings| {
        for (stop_strings) |stop_str| {
            if (std.mem.indexOf(u8, output_text, stop_str)) |stop_idx| {
                final_text = output_text[0..stop_idx];
                break;
            }
        }
    }

    // 8. Tool calling loop (single round)
    if (req.tools) |tools| {
        _ = tools;
        const family = tool_calling_mod.detectFamily(req.model);
        if (try tool_calling_mod.parse(allocator, final_text, family)) |parse_result| {
            defer parse_result.deinit();

            if (parse_result.calls.len > 0) {
                // Execute tools and build tool results
                var tool_results = std.ArrayList(u8).empty;
                defer tool_results.deinit(allocator);

                const exec_config = tool_executor_mod.ExecutorConfig{
                    .allow_shell_exec = server_config.allow_unsafe_tools,
                    .max_file_size = 1024 * 1024,
                    .io = state.io,
                };

                for (parse_result.calls) |call| {
                    const result = try tool_executor_mod.execute(allocator, exec_config, call.function.name, call.function.arguments);
                    defer result.deinit();

                    const result_text = if (result.success) result.output else result.error_message orelse "error";
                    try tool_results.appendSlice(allocator, "Tool '");
                    try tool_results.appendSlice(allocator, call.function.name);
                    try tool_results.appendSlice(allocator, "' result: ");
                    try tool_results.appendSlice(allocator, result_text);
                    try tool_results.appendSlice(allocator, "\n");
                }

                // Build OpenAI-compatible response with tool_calls
                const created = @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toSeconds()));
                var tool_calls_json = std.ArrayList(u8).empty;
                defer tool_calls_json.deinit(allocator);
                try tool_calls_json.appendSlice(allocator, "[");
                for (parse_result.calls, 0..) |call, i| {
                    if (i > 0) try tool_calls_json.appendSlice(allocator, ",");
                    try tool_calls_json.print(allocator,
                        \\{{"id":"{s}","type":"function","function":{{"name":"{s}","arguments":"{s}"}}}}
                    , .{ call.id, call.function.name, call.function.arguments });
                }
                try tool_calls_json.appendSlice(allocator, "]");

                return try std.fmt.allocPrint(allocator,
                    \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":null,"tool_calls":{s}}},"finish_reason":"tool_calls"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
                , .{
                    created,
                    created,
                    req.model,
                    tool_calls_json.items,
                    prompt_tokens.len,
                    new_tokens.len,
                    prompt_tokens.len + new_tokens.len,
                });
            }
        }
    }

    // 9. Build standard OpenAI-compatible response
    const created = @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toSeconds()));

    // JSON-escape the output text
    var escaped = std.ArrayList(u8).empty;
    defer escaped.deinit(allocator);
    try jsonEscapeInto(&escaped, allocator, final_text);

    return try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
    , .{
        created,
        created,
        req.model,
        escaped.items,
        prompt_tokens.len,
        new_tokens.len,
        prompt_tokens.len + new_tokens.len,
    });
}

// ------------------------------------------------------------------
// SSE streaming completion handler (using Generation API streamGenerate)
// ------------------------------------------------------------------

/// Keep-alive threshold in milliseconds (5 seconds).
const keep_alive_threshold_ms: i64 = 5000;

fn handleStreamingCompletion(
    allocator: std.mem.Allocator,
    io: std.Io,
    state: *ModelState,
    connection: std.Io.net.Stream,
    server_config: ServerConfig,
    req: ChatCompletionRequest,
) !void {
    // 1. Send SSE response headers
    try writeSSEHeaders(connection, io);

    var sse = SSEWriter{ .connection = connection, .io = io };

    const created = @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toSeconds()));
    const completion_id = created; // Use timestamp as unique ID

    // 2. Send initial chunk with role
    const role_chunk = try formatSSEChunk(allocator, completion_id, created, req.model, null, null);
    defer allocator.free(role_chunk);
    try sse.sendEvent(role_chunk);

    // 3. Build messages for chat template
    var messages = std.ArrayList(root.tokenizer.ChatMessage).empty;
    defer messages.deinit(allocator);

    for (req.messages) |msg| {
        try messages.append(allocator, .{
            .role = msg.role,
            .content = msg.content,
        });
    }

    // 4. Apply chat template
    const prompt_text = try state.chat_template.apply(messages.items, true);
    defer allocator.free(prompt_text);

    // 5. Tokenize
    const prompt_tokens = try state.tokenizer_strategy.encode(prompt_text, false, allocator);
    defer allocator.free(prompt_tokens);

    // 6. Setup generation config
    const max_tokens = if (req.max_tokens) |mt| @min(mt, @as(u32, @intCast(server_config.max_tokens))) else server_config.max_tokens;
    const temperature = req.temperature orelse server_config.temperature;

    const seed = req.seed orelse @as(u64, @intCast(std.Io.Timestamp.now(state.io, .real).toMilliseconds()));

    const gen_config = generation_mod.GenerateConfig{
        .max_tokens = max_tokens,
        .temperature = temperature,
        .top_k = server_config.top_k,
        .top_p = server_config.top_p,
        .seed = seed,
    };

    // 7. Use streamGenerate with a callback that sends SSE events.
    // Since Zig function pointers can't capture, we use a struct with
    // mutable statics scoped to this function.
    const StreamState = struct {
        var s_allocator: std.mem.Allocator = undefined;
        var s_sse: *SSEWriter = undefined;
        var s_completion_id: u64 = 0;
        var s_created: u64 = 0;
        var s_model_name: []const u8 = "";
        var s_tokenizer: root.tokenizer.TokenizerStrategy = undefined;
        var s_token_count: usize = 0;
        var s_stop_strings: ?[]const []const u8 = null;
        var s_text_buffer: std.ArrayList(u8) = undefined;
        var s_stopped: bool = false;

        fn callback(token: u32, is_done: bool) void {
            if (s_stopped) return;

            // Decode token to text
            const token_text = s_tokenizer.decode(&[_]u32{token}, s_allocator) catch return;
            defer s_allocator.free(token_text);

            // Accumulate text for stop-string detection
            s_text_buffer.appendSlice(s_allocator, token_text) catch return;

            // Check stop strings
            if (s_stop_strings) |stop_strings| {
                for (stop_strings) |stop_str| {
                    if (std.mem.indexOf(u8, s_text_buffer.items, stop_str)) |_| {
                        s_stopped = true;
                        const done_chunk = formatSSEChunk(
                            s_allocator,
                            s_completion_id,
                            s_created,
                            s_model_name,
                            null,
                            "stop",
                        ) catch return;
                        defer s_allocator.free(done_chunk);
                        s_sse.sendEvent(done_chunk) catch return;
                        s_sse.sendEvent("[DONE]") catch return;
                        return;
                    }
                }
            }

            s_token_count += 1;

            const finish: ?[]const u8 = if (is_done) "stop" else null;
            const chunk = formatSSEChunk(
                s_allocator,
                s_completion_id,
                s_created,
                s_model_name,
                token_text,
                finish,
            ) catch return;
            defer s_allocator.free(chunk);

            s_sse.sendEvent(chunk) catch return;

            if (is_done) {
                s_sse.sendEvent("[DONE]") catch return;
            }
        }
    };

    StreamState.s_allocator = allocator;
    StreamState.s_sse = &sse;
    StreamState.s_completion_id = completion_id;
    StreamState.s_created = created;
    StreamState.s_model_name = req.model;
    StreamState.s_tokenizer = state.tokenizer_strategy;
    StreamState.s_token_count = 0;
    StreamState.s_stop_strings = req.stop;
    StreamState.s_text_buffer = std.ArrayList(u8).empty;
    StreamState.s_stopped = false;

    // 8. Start keep-alive fiber to prevent proxy timeout during long prefill
    _ = io.async(keepAliveLoop, .{ io, &sse, &StreamState.s_token_count });

    // 9. Run generation using the Generation API
    if (state.speculative_ngram) |n| {
        // Prompt Lookup Decoding: build prompt n-gram index and speculative decode
        var drafter = try speculative_mod.PldDrafter.init(allocator, prompt_tokens, n);
        defer drafter.deinit();

        generation_mod.streamGenerateSpeculative(
            state.vtable,
            prompt_tokens,
            gen_config,
            state.caches,
            state.ctx,
            &drafter,
            &StreamState.callback,
        ) catch |err| {
            std.log.err("streamGenerateSpeculative error: {}", .{err});
            const err_chunk = formatSSEChunk(allocator, completion_id, created, req.model, null, "error") catch return;
            defer allocator.free(err_chunk);
            sse.sendEvent(err_chunk) catch return;
            sse.sendEvent("[DONE]") catch return;
        };
    } else {
        generation_mod.streamGenerate(
            state.vtable,
            prompt_tokens,
            gen_config,
            state.caches,
            state.ctx,
            &StreamState.callback,
        ) catch |err| {
            std.log.err("streamGenerate error: {}", .{err});
            // Send error event if generation fails mid-stream
            const err_chunk = formatSSEChunk(allocator, completion_id, created, req.model, null, "error") catch return;
            defer allocator.free(err_chunk);
            sse.sendEvent(err_chunk) catch return;
            sse.sendEvent("[DONE]") catch return;
        };
    }

    // If streamGenerate completed without calling callback with is_done=true
    // (e.g., empty prompt), send [DONE]
    if (StreamState.s_token_count == 0) {
        const final_chunk = try formatSSEChunk(allocator, completion_id, created, req.model, null, "stop");
        defer allocator.free(final_chunk);
        try sse.sendEvent(final_chunk);
        try sse.sendEvent("[DONE]");
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

fn handleAnthropicMessages(
    allocator: std.mem.Allocator,
    io: std.Io,
    state: *ModelState,
    connection: std.Io.net.Stream,
    server_config: ServerConfig,
    body: []const u8,
) !void {
    const parsed = std.json.parseFromSlice(AnthropicRequest, allocator, body, .{}) catch {
        try writeJsonResponse(connection, io, 400, "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"invalid_json\"}}");
        return;
    };
    defer parsed.deinit();
    const req = parsed.value;

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

    const gen_config = generation_mod.GenerateConfig{
        .max_tokens = max_tokens,
        .temperature = temperature,
        .top_k = server_config.top_k,
        .top_p = server_config.top_p,
        .seed = seed,
    };

    const new_tokens = try generation_mod.generate(
        state.vtable,
        prompt_tokens,
        gen_config,
        state.caches,
        state.ctx,
    );
    defer allocator.free(new_tokens);

    // Decode output
    const output_text = try state.tokenizer_strategy.decode(new_tokens, allocator);
    defer allocator.free(output_text);

    // Format Anthropic response
    const response = try formatAnthropicResponse(
        allocator,
        req.model,
        output_text,
        prompt_tokens.len,
        new_tokens.len,
    );
    defer allocator.free(response);

    try writeJsonResponse(connection, io, 200, response);
}

// ------------------------------------------------------------------
// SSE keep-alive integration
// ------------------------------------------------------------------

/// Keep-alive interval in milliseconds. During long prefill or generation
/// pauses, SSE comments are sent at this interval to prevent client timeouts.
const sse_keep_alive_interval_ms: i64 = 15_000;

// ------------------------------------------------------------------
// Unit tests for SSE helpers
// ------------------------------------------------------------------

const TestWriter = struct {
    data: std.ArrayList(u8),
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) TestWriter {
        return .{
            .data = std.ArrayList(u8).empty,
            .allocator = allocator,
        };
    }

    fn deinit(self: *TestWriter) void {
        self.data.deinit(self.allocator);
    }

    fn writeAll(self: *TestWriter, bytes: []const u8) !void {
        try self.data.appendSlice(self.allocator, bytes);
    }

    fn getWritten(self: *const TestWriter) []const u8 {
        return self.data.items;
    }
};

test "writeSSEEvent formats correctly" {
    var writer = TestWriter.init(std.testing.allocator);
    defer writer.deinit();

    try writeSSEEvent(&writer, "{\"test\":true}");
    try std.testing.expectEqualStrings("data: {\"test\":true}\n\n", writer.getWritten());
}

test "writeSSEKeepAlive formats correctly" {
    var writer = TestWriter.init(std.testing.allocator);
    defer writer.deinit();

    try writeSSEKeepAlive(&writer);
    try std.testing.expectEqualStrings(": keep-alive\n\n", writer.getWritten());
}

test "writeSSEEvent with empty data" {
    var writer = TestWriter.init(std.testing.allocator);
    defer writer.deinit();

    try writeSSEEvent(&writer, "");
    try std.testing.expectEqualStrings("data: \n\n", writer.getWritten());
}

test "writeSSEEvent with [DONE] sentinel" {
    var writer = TestWriter.init(std.testing.allocator);
    defer writer.deinit();

    try writeSSEEvent(&writer, "[DONE]");
    try std.testing.expectEqualStrings("data: [DONE]\n\n", writer.getWritten());
}

test "formatSSEChunk with token content" {
    const allocator = std.testing.allocator;
    const chunk = try formatSSEChunk(allocator, 12345, 1700000000, "test-model", "Hello", null);
    defer allocator.free(chunk);

    // Verify it contains expected fields
    try std.testing.expect(std.mem.indexOf(u8, chunk, "\"object\":\"chat.completion.chunk\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, chunk, "\"content\":\"Hello\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, chunk, "\"finish_reason\":null") != null);
    try std.testing.expect(std.mem.indexOf(u8, chunk, "\"model\":\"test-model\"") != null);
}

test "formatSSEChunk with finish_reason stop" {
    const allocator = std.testing.allocator;
    const chunk = try formatSSEChunk(allocator, 12345, 1700000000, "test-model", null, "stop");
    defer allocator.free(chunk);

    try std.testing.expect(std.mem.indexOf(u8, chunk, "\"finish_reason\":\"stop\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, chunk, "\"delta\":{}") != null);
}

test "formatSSEChunk with special characters in token" {
    const allocator = std.testing.allocator;
    const chunk = try formatSSEChunk(allocator, 1, 1, "m", "hello\nworld\"test", null);
    defer allocator.free(chunk);

    // Verify JSON escaping
    try std.testing.expect(std.mem.indexOf(u8, chunk, "hello\\nworld\\\"test") != null);
}

test "jsonEscapeInto escapes special characters" {
    const allocator = std.testing.allocator;
    var list = std.ArrayList(u8).empty;
    defer list.deinit(allocator);

    try jsonEscapeInto(&list, allocator, "line1\nline2\ttab\\slash\"quote\rreturn");
    try std.testing.expectEqualStrings("line1\\nline2\\ttab\\\\slash\\\"quote\\rreturn", list.items);
}

test "detectArchitecture: LlamaForCausalLM" {
    const json = "{\"architectures\": [\"LlamaForCausalLM\"], \"model_type\": \"llama\"}";
    try std.testing.expectEqualStrings("LlamaForCausalLM", detectArchitecture(json));
}

test "detectArchitecture: DeepseekV4ForCausalLM" {
    const json = "{\"architectures\": [\"DeepseekV4ForCausalLM\"], \"model_type\": \"deepseek_v4\"}";
    try std.testing.expectEqualStrings("DeepseekV4ForCausalLM", detectArchitecture(json));
}

test "detectArchitecture: fallback to model_type" {
    const json = "{\"model_type\": \"deepseek_v4\"}";
    try std.testing.expectEqualStrings("DeepseekV4ForCausalLM", detectArchitecture(json));
}

test "detectArchitecture: default fallback" {
    const json = "{\"model_type\": \"llama\"}";
    try std.testing.expectEqualStrings("LlamaForCausalLM", detectArchitecture(json));
}

test "detectToolCalls: detects valid tool call pattern" {
    const allocator = std.testing.allocator;
    const text =
        \\I'll call the function: {"name": "get_weather", "arguments": {"city": "Seattle"}}
    ;
    const result = detectToolCalls(allocator, text);
    try std.testing.expect(result != null);
    defer allocator.free(result.?);
    try std.testing.expect(std.mem.indexOf(u8, result.?, "get_weather") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.?, "\"type\":\"function\"") != null);
}

test "detectToolCalls: returns null for plain text" {
    const allocator = std.testing.allocator;
    const text = "Hello, how can I help you today?";
    const result = detectToolCalls(allocator, text);
    try std.testing.expect(result == null);
}

test "detectToolCalls: returns null for JSON without name field" {
    const allocator = std.testing.allocator;
    const text = "{\"key\": \"value\", \"arguments\": {}}";
    const result = detectToolCalls(allocator, text);
    try std.testing.expect(result == null);
}

test "formatAnthropicResponse: produces valid Anthropic format" {
    const allocator = std.testing.allocator;
    const response = try formatAnthropicResponse(allocator, "claude-3", "Hello world", 10, 5);
    defer allocator.free(response);

    try std.testing.expect(std.mem.indexOf(u8, response, "\"type\":\"message\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"stop_reason\":\"end_turn\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"type\":\"text\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "Hello world") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"model\":\"claude-3\"") != null);
}

test "formatAnthropicResponse: escapes special characters" {
    const allocator = std.testing.allocator;
    const response = try formatAnthropicResponse(allocator, "m", "line1\nline2\"quoted\"", 1, 1);
    defer allocator.free(response);

    try std.testing.expect(std.mem.indexOf(u8, response, "line1\\nline2\\\"quoted\\\"") != null);
}
