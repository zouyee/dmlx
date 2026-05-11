/// Engine Loop — Main inference fiber for the HTTP server.
///
/// Processes requests from the RequestQueue serially (one at a time),
/// using the model's VTable for model-agnostic inference or the DSV4
/// direct native path when a `DSV4Model` is configured.
/// Each request gets its own fresh KV caches to avoid cross-request
/// cache contamination.
///
/// For streaming requests: tokens are delivered via CompletionSignal
/// after each generate step, allowing the HTTP fiber to stream them
/// to the client in real time.
///
/// For non-streaming requests: all tokens are generated first, then
/// delivered as a single completion event.
const std = @import("std");
const c = @import("mlx").c;
const ops = @import("mlx").ops;
const array_mod = @import("mlx").array;
const generation_mod = @import("../generation.zig");
const kvcache = @import("../kvcache.zig");
const request_state = @import("request_state.zig");
const request_queue = @import("request_queue.zig");
const completion_signal = @import("completion_signal.zig");
const root = @import("../root.zig");
const dsv4_mod = @import("../models/deepseek_v4.zig");
const dsv4_loader = @import("../models/deepseek_v4_loader.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const ModelVTable = generation_mod.ModelVTable;
const GenerateConfig = generation_mod.GenerateConfig;
const KVCacheStrategy = kvcache.KVCacheStrategy;
const LayerConfig = kvcache.LayerConfig;
const RequestState = request_state.RequestState;
const RequestQueue = request_queue.RequestQueue;
const CompletionSignal = completion_signal.CompletionSignal;
const TokenEvent = completion_signal.TokenEvent;
const TokenFinishReason = completion_signal.TokenEvent.FinishReason;

/// Platform-native sleep that does NOT enter the std.Io event loop.
/// Safe to call from the main thread (where io.sleep would deadlock with
/// a pending listener.accept in another fiber).
pub fn threadSleepMs(ms: u64) void {
    const ts = std.c.timespec{
        .sec = @intCast(ms / 1000),
        .nsec = @intCast((ms % 1000) * 1_000_000),
    };
    _ = std.c.nanosleep(&ts, null);
}

/// Global shutdown flag. Set by signal handlers to request graceful shutdown.
pub var g_shutdown_requested = std.atomic.Value(bool).init(false);

/// Request engine shutdown (thread-safe, callable from signal handlers).
pub fn requestShutdown() void {
    std.log.info("[Engine] Shutdown requested", .{});
    g_shutdown_requested.store(true, .release);
}

/// Check if shutdown has been requested.
pub fn isShutdownRequested() bool {
    return g_shutdown_requested.load(.acquire);
}

/// Context passed to the streamGenerateCtx callback.
const StreamCallbackCtx = struct {
    signal: *CompletionSignal,
    io: std.Io,
    tokenizer: root.tokenizer.TokenizerStrategy,
    allocator: std.mem.Allocator,
    stop_strings: ?[]const []const u8,
    text_buffer: std.ArrayList(u8),
    token_count: usize,
    max_tokens: usize,
    stopped: bool,

    fn init(allocator: std.mem.Allocator, io: std.Io, signal: *CompletionSignal, tokenizer: root.tokenizer.TokenizerStrategy, stop_strings: ?[]const []const u8, max_tokens: usize) StreamCallbackCtx {
        return .{
            .signal = signal,
            .io = io,
            .tokenizer = tokenizer,
            .allocator = allocator,
            .stop_strings = stop_strings,
            .text_buffer = std.ArrayList(u8).empty,
            .token_count = 0,
            .max_tokens = max_tokens,
            .stopped = false,
        };
    }

    fn deinit(self: *StreamCallbackCtx) void {
        self.text_buffer.deinit(self.allocator);
    }

    fn callback(ctx_ptr: *anyopaque, token: u32, is_done: bool) void {
        const self: *StreamCallbackCtx = @ptrCast(@alignCast(ctx_ptr));
        if (self.stopped) return;

        // Decode token to text
        const token_text = self.tokenizer.decode(&[_]u32{token}, self.allocator) catch return;
        defer self.allocator.free(token_text);

        // Accumulate text for stop-string detection
        self.text_buffer.appendSlice(self.allocator, token_text) catch return;
        self.token_count += 1;

        // Check stop strings
        var finish_reason: ?TokenFinishReason = null;
        if (is_done) {
            finish_reason = .length;
        }
        if (self.stop_strings) |stop_strings| {
            for (stop_strings) |stop_str| {
                if (std.mem.indexOf(u8, self.text_buffer.items, stop_str)) |_| {
                    self.stopped = true;
                    finish_reason = .stop;
                    break;
                }
            }
        }

        const is_final = is_done or self.stopped or self.token_count >= self.max_tokens;
        if (is_final and finish_reason == null) {
            finish_reason = .length;
        }

        self.signal.deliverToken(self.io, token, token_text, is_final, finish_reason);
    }
};

/// EngineLoop configuration.
pub const EngineConfig = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    request_queue: *RequestQueue,
    model: ModelVTable,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    tokenizer: root.tokenizer.TokenizerStrategy,
    /// For DSV4 models: config JSON content (used to create caches).
    config_content: ?[]const u8,
    /// For non-DSV4 models: layer config used to create standard caches.
    layer_config: ?LayerConfig,
    num_layers: usize,
    /// Direct DSV4 model pointer. When set, the engine uses the native
    /// DSV4 generation path instead of the VTable.
    dsv4_model: ?*dsv4_mod.DSV4Model = null,
};

pub const EngineLoop = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    request_queue: *RequestQueue,
    model: ModelVTable,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    tokenizer: root.tokenizer.TokenizerStrategy,
    config_content: ?[]const u8,
    layer_config: ?LayerConfig,
    num_layers: usize,
    dsv4_model: ?*dsv4_mod.DSV4Model,
    running: std.atomic.Value(bool),

    pub fn init(config: EngineConfig) EngineLoop {
        return .{
            .allocator = config.allocator,
            .io = config.io,
            .request_queue = config.request_queue,
            .model = config.model,
            .ctx = config.ctx,
            .stream = config.stream,
            .tokenizer = config.tokenizer,
            .config_content = config.config_content,
            .layer_config = config.layer_config,
            .num_layers = config.num_layers,
            .dsv4_model = config.dsv4_model,
            .running = std.atomic.Value(bool).init(true),
        };
    }

    pub fn stop(self: *EngineLoop) void {
        self.running.store(false, .release);
    }

    /// Main loop. Runs in an io.async fiber (may be on a worker thread).
    pub fn run(self: *EngineLoop) void {
        std.log.info("[Engine] Engine loop started", .{});
        defer std.log.info("[Engine] Engine loop exiting", .{});
        while (self.running.load(.acquire) and !g_shutdown_requested.load(.acquire)) {
            const requests = self.request_queue.drainAll(self.allocator) catch {
                threadSleepMs(1);
                continue;
            };
            defer self.allocator.free(requests);

            if (requests.len == 0) {
                threadSleepMs(1);
                continue;
            }

            std.log.info("[Engine] Drained {d} requests", .{requests.len});

            // Process each request serially (one at a time).
            // Future: process multiple requests in a batch (Task 2).
            for (requests) |req| {
                if (req.isCancelled()) {
                    req.completion.deliverError(self.io, "Request cancelled");
                    continue;
                }
                std.log.info("[Engine] Processing request {d}", .{req.id});
                self.processRequest(req);
                logRequestCompletion(self.io, req);
            }
        }
    }

    fn processRequest(self: *EngineLoop, req: *RequestState) void {
        // Record start time for request latency tracking.
        req.start_time_ns = std.Io.Timestamp.now(self.io, .awake).nanoseconds;

        // Create fresh KV caches for this request.
        const caches = self.createCaches() catch |err| {
            std.log.err("EngineLoop: failed to create caches: {}", .{err});
            req.completion.deliverError(self.io, "Failed to create KV caches");
            return;
        };
        defer {
            for (caches) |cache| cache.deinit(self.allocator);
            self.allocator.free(caches);
        }

        const gen_config = GenerateConfig{
            .max_tokens = req.max_tokens,
            .temperature = req.temperature,
            .top_k = req.top_k,
            .top_p = req.top_p,
            .seed = req.seed,
            .stop_tokens = req.stop_tokens,
        };

        if (self.dsv4_model) |model| {
            if (req.streaming) {
                self.processDSV4StreamingRequest(req, caches, gen_config, model);
            } else {
                self.processDSV4NonStreamingRequest(req, caches, gen_config, model);
            }
        } else {
            if (req.streaming) {
                self.processStreamingRequest(req, caches, gen_config);
            } else {
                self.processNonStreamingRequest(req, caches, gen_config);
            }
        }
    }

    fn processStreamingRequest(
        self: *EngineLoop,
        req: *RequestState,
        caches: []KVCacheStrategy,
        gen_config: GenerateConfig,
    ) void {
        var stream_ctx = StreamCallbackCtx.init(
            self.allocator,
            self.io,
            &req.completion,
            self.tokenizer,
            req.stop_strings,
            req.max_tokens,
        );
        defer stream_ctx.deinit();

        generation_mod.streamGenerateCtx(
            self.model,
            req.prompt_tokens,
            gen_config,
            caches,
            self.ctx,
            @ptrCast(&stream_ctx),
            &StreamCallbackCtx.callback,
        ) catch |err| {
            std.log.err("EngineLoop: streamGenerateCtx failed: {}", .{err});
            req.completion.deliverError(self.io, "Generation failed");
            return;
        };

        // Record generated token count for logging.
        req.token_count = stream_ctx.token_count;

        // Ensure a final done event is sent if not already.
        if (!req.completion.isDone(self.io)) {
            req.completion.deliverDone(self.io, .stop);
        }
    }

    fn processNonStreamingRequest(
        self: *EngineLoop,
        req: *RequestState,
        caches: []KVCacheStrategy,
        gen_config: GenerateConfig,
    ) void {
        const tokens = generation_mod.generate(
            self.model,
            req.prompt_tokens,
            gen_config,
            caches,
            self.ctx,
        ) catch |err| {
            std.log.err("EngineLoop: generate failed: {}", .{err});
            req.completion.deliverError(self.io, "Generation failed");
            return;
        };
        defer self.allocator.free(tokens);

        // Decode all tokens to text.
        const text = self.tokenizer.decode(tokens, self.allocator) catch {
            req.completion.deliverError(self.io, "Failed to decode tokens");
            return;
        };
        defer self.allocator.free(text);

        // Check stop strings and truncate if needed.
        var final_text = text;
        if (req.stop_strings) |stop_strings| {
            for (stop_strings) |stop_str| {
                if (std.mem.indexOf(u8, text, stop_str)) |idx| {
                    final_text = text[0..idx];
                    break;
                }
            }
        }

        // Record token count and deliver the full text as a single token event.
        req.token_count = @intCast(tokens.len);
        req.completion.deliverToken(self.io, 0, final_text, true, .stop);
    }

    fn processDSV4StreamingRequest(
        self: *EngineLoop,
        req: *RequestState,
        caches: []KVCacheStrategy,
        gen_config: GenerateConfig,
        model: *dsv4_mod.DSV4Model,
    ) void {
        var sampler = root.sampling.SamplerConfig{
            .temperature = gen_config.temperature,
            .top_k = gen_config.top_k,
            .top_p = gen_config.top_p,
            .prng = std.Random.DefaultPrng.init(gen_config.seed),
            .repetition_penalty = gen_config.repetition_penalty,
        };

        const tokens = model.generate(req.prompt_tokens, req.max_tokens, &sampler, caches, self.stream) catch |err| {
            std.log.err("EngineLoop: DSV4 generate failed: {}", .{err});
            req.completion.deliverError(self.io, "Generation failed");
            return;
        };
        defer self.allocator.free(tokens);

        // Decode and deliver tokens one by one to simulate streaming.
        for (tokens, 0..) |token, i| {
            const token_text = self.tokenizer.decode(&[_]u32{token}, self.allocator) catch continue;
            defer self.allocator.free(token_text);

            const is_final = i == tokens.len - 1;
            req.completion.deliverToken(self.io, token, token_text, is_final, if (is_final) .stop else null);
        }

        // Ensure a final done event is sent if not already.
        if (!req.completion.isDone(self.io)) {
            req.completion.deliverDone(self.io, .stop);
        }
    }

    fn processDSV4NonStreamingRequest(
        self: *EngineLoop,
        req: *RequestState,
        caches: []KVCacheStrategy,
        gen_config: GenerateConfig,
        model: *dsv4_mod.DSV4Model,
    ) void {
        var sampler = root.sampling.SamplerConfig{
            .temperature = gen_config.temperature,
            .top_k = gen_config.top_k,
            .top_p = gen_config.top_p,
            .prng = std.Random.DefaultPrng.init(gen_config.seed),
            .repetition_penalty = gen_config.repetition_penalty,
        };

        const tokens = model.generate(req.prompt_tokens, req.max_tokens, &sampler, caches, self.stream) catch |err| {
            std.log.err("EngineLoop: DSV4 generate failed: {}", .{err});
            req.completion.deliverError(self.io, "Generation failed");
            return;
        };
        defer self.allocator.free(tokens);

        const text = self.tokenizer.decode(tokens, self.allocator) catch {
            req.completion.deliverError(self.io, "Failed to decode tokens");
            return;
        };
        defer self.allocator.free(text);

        // Check stop strings
        var final_text = text;
        if (req.stop_strings) |stop_strings| {
            for (stop_strings) |stop_str| {
                if (std.mem.indexOf(u8, text, stop_str)) |idx| {
                    final_text = text[0..idx];
                    break;
                }
            }
        }

        req.token_count = @intCast(tokens.len);
        req.completion.deliverToken(self.io, 0, final_text, true, .stop);
    }

    fn logRequestCompletion(io: std.Io, req: *RequestState) void {
        const end_time_ns = std.Io.Timestamp.now(io, .awake).nanoseconds;
        const duration_ns = end_time_ns - req.start_time_ns;
        const duration_ms = @as(f64, @floatFromInt(duration_ns)) / 1_000_000.0;
        const prompt_len = req.prompt_tokens.len;
        const gen_len = req.token_count;
        const tokens_per_sec = if (duration_ns > 0 and gen_len > 0)
            @as(f64, @floatFromInt(gen_len)) * 1_000_000_000.0 / @as(f64, @floatFromInt(duration_ns))
        else
            0.0;
        std.log.info(
            "[RequestLog] id={d} model={s} streaming={} prompt_tokens={d} generated_tokens={d} duration_ms={d:.2} tokens_per_sec={d:.2}",
            .{ req.id, req.model_name, req.streaming, prompt_len, gen_len, duration_ms, tokens_per_sec },
        );
    }

    fn createCaches(self: *EngineLoop) ![]KVCacheStrategy {
        if (self.config_content) |content| {
            if (self.dsv4_model != null) {
                // DSV4 direct path: use makeV4Caches for heterogeneous caches.
                const dsv4_config = try dsv4_loader.parseDSV4Config(self.allocator, content);
                defer dsv4_config.deinitClone(self.allocator);
                return try dsv4_loader.makeV4Caches(self.allocator, &dsv4_config, self.stream);
            }

            // DSV4 VTable path (fallback): parse config and create standard caches manually.
            const dsv4_config = try dsv4_loader.parseDSV4Config(self.allocator, content);
            defer dsv4_config.deinitClone(self.allocator);

            const caches = try self.allocator.alloc(KVCacheStrategy, dsv4_config.num_hidden_layers);
            errdefer self.allocator.free(caches);

            const effective_max_seq = @min(dsv4_config.max_position_embeddings, 131072);

            for (0..dsv4_config.num_hidden_layers) |i| {
                const compress_ratio = if (i < dsv4_config.compress_ratios.len) dsv4_config.compress_ratios[i] else 0;
                const layer_max_seq = if (compress_ratio > 1)
                    effective_max_seq / compress_ratio
                else
                    effective_max_seq;

                const layer_config = kvcache.LayerConfig{
                    .batch_size = 1,
                    .num_heads = dsv4_config.num_key_value_heads,
                    .num_kv_heads = dsv4_config.num_key_value_heads,
                    .head_dim = dsv4_config.head_dim,
                    .max_seq_len = layer_max_seq,
                    .dtype = .float32,
                };
                caches[i] = try kvcache.createStandard(self.allocator, layer_config, self.stream);
            }
            return caches;
        }

        // Non-DSV4 path: create standard caches per layer.
        const layer_config = self.layer_config orelse return error.NoLayerConfig;
        const caches = try self.allocator.alloc(KVCacheStrategy, self.num_layers);
        errdefer self.allocator.free(caches);

        for (0..self.num_layers) |i| {
            caches[i] = try kvcache.createStandard(self.allocator, layer_config, self.stream);
        }
        return caches;
    }
};
