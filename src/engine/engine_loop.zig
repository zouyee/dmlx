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
const array_arena_mod = @import("mlx").array_arena;
const shape_mod = @import("mlx").shape;
const generation_mod = @import("../generation.zig");
const kvcache = @import("../kvcache.zig");
const request_state = @import("request_state.zig");
const request_queue = @import("request_queue.zig");
const completion_signal = @import("completion_signal.zig");
const root = @import("../root.zig");
const dsv4_mod = @import("../models/deepseek_v4.zig");
const dsv4_loader = @import("../models/deepseek_v4_loader.zig");

const Array = array_mod.Array;
const ScopedArrayArena = array_arena_mod.ScopedArrayArena;
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
const sampling_mod = @import("../sampling.zig");
const SamplerConfig = sampling_mod.SamplerConfig;

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
    /// Maximum number of requests to batch together in the decode phase.
    max_batch_size: usize = 8,
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
    /// Batch KV caches for batched decode (one BatchKVCache per layer).
    /// Lazily initialized on first prefill via BatchKVCache.merge().
    /// Null until the first batch of requests is prefilled.
    batch_caches: ?[]KVCacheStrategy,
    max_batch_size: usize,
    /// Sequence index counter for assigning slots in batch caches.
    seq_index_counter: usize,
    /// Active requests currently in the batch (prefilling or decoding).
    active_requests: std.ArrayList(*RequestState),

    pub fn init(config: EngineConfig) !EngineLoop {
        // Note: batch_caches is lazily initialized via BatchKVCache.merge()
        // on first prefill. This avoids upfront allocation and matches the
        // mlx-lm architecture where each request gets independent caches
        // and BatchKVCache merges them at forward time.
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
            .batch_caches = null,
            .max_batch_size = config.max_batch_size,
            .seq_index_counter = 0,
            .active_requests = std.ArrayList(*RequestState).empty,
        };
    }

    pub fn deinit(self: *EngineLoop) void {
        if (self.batch_caches) |caches| {
            for (caches) |cache| cache.deinit(self.allocator);
            self.allocator.free(caches);
        }
        self.active_requests.deinit(self.allocator);
    }

    pub fn stop(self: *EngineLoop) void {
        self.running.store(false, .release);
    }

    /// Main loop. Runs in an io.async fiber (may be on a worker thread).
    pub fn run(self: *EngineLoop) void {
        std.log.info("[Engine] Engine loop started", .{});
        defer std.log.info("[Engine] Engine loop exiting", .{});
        while (self.running.load(.acquire) and !g_shutdown_requested.load(.acquire)) {
            // Drain new requests from the queue.
            const new_requests = self.request_queue.drainAll(self.allocator) catch {
                threadSleepMs(1);
                continue;
            };
            defer self.allocator.free(new_requests);

            // Separate DSV4 and non-DSV4 requests.
            var dsv4_requests: usize = 0;
            var non_dsv4: usize = 0;
            for (new_requests) |req| {
                if (req.isCancelled()) {
                    req.completion.deliverError(self.io, "Request cancelled");
                    continue;
                }
                if (self.dsv4_model != null) {
                    new_requests[dsv4_requests] = req;
                    dsv4_requests += 1;
                } else {
                    new_requests[non_dsv4] = req;
                    non_dsv4 += 1;
                }
            }

            // Process DSV4 requests serially (batching not yet supported for DSV4).
            for (new_requests[0..dsv4_requests]) |req| {
                self.processRequest(req);
            }

            // Decode active non-DSV4 requests first (generation batch).
            if (self.active_requests.items.len > 0) {
                self.step();
            }

            // Prefill new non-DSV4 requests and extend their private caches
            // into the shared caches for batched decode.
            if (non_dsv4 > 0) {
                std.log.info("[Engine] Prefilling {d} new requests", .{non_dsv4});
                self.prefillBatch(new_requests[0..non_dsv4]);
            }

            // If no work was done, sleep briefly.
            if (self.active_requests.items.len == 0 and non_dsv4 == 0 and dsv4_requests == 0) {
                threadSleepMs(1);
            }
        }

        // Drain remaining active requests before shutdown.
        while (self.active_requests.items.len > 0 and !g_shutdown_requested.load(.acquire)) {
            self.step();
        }
    }

    fn processRequest(self: *EngineLoop, req: *RequestState) void {
        // Record start time for request latency tracking.
        req.start_time_ns = std.Io.Timestamp.now(self.io, .awake).nanoseconds;

        // Create per-request caches to avoid cross-request cache contamination.
        // Shared caches are reserved for future batched decode; until the batch
        // prefill → merge → batch decode pipeline is fully implemented, each
        // request gets its own private caches.
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
        logRequestCompletion(self.io, req);
    }

    /// Batch prefill: each request gets its own private caches for prefill,
    /// then the caches are extended into shared_caches for batched decode.
    fn prefillBatch(self: *EngineLoop, requests: []*RequestState) void {
        if (requests.len == 0) return;

        // Allocate per-request caches.
        var req_caches = self.allocator.alloc([]KVCacheStrategy, requests.len) catch {
            std.log.err("[Engine] failed to allocate request caches", .{});
            for (requests) |req| req.completion.deliverError(self.io, "Prefill allocation failed");
            return;
        };
        @memset(req_caches, &[_]KVCacheStrategy{});
        defer {
            for (req_caches) |caches| {
                if (caches.len > 0) {
                    for (caches) |cache| cache.deinit(self.allocator);
                    self.allocator.free(caches);
                }
            }
            self.allocator.free(req_caches);
        }

        var success_mask = self.allocator.alloc(bool, requests.len) catch {
            std.log.err("[Engine] failed to allocate success mask", .{});
            return;
        };
        defer self.allocator.free(success_mask);
        @memset(success_mask, false);

        for (requests, 0..) |req, i| {
            req_caches[i] = self.createCaches() catch {
                std.log.err("[Engine] failed to create caches for request {d}", .{req.id});
                req.completion.deliverError(self.io, "Failed to create KV caches");
                continue;
            };
            success_mask[i] = true;

            // Init sampler.
            req.sampler = SamplerConfig.init(req.seed);
            req.sampler.temperature = req.temperature;
            req.sampler.top_k = req.top_k;
            req.sampler.top_p = req.top_p;

            // Prefill using private caches.
            const prompt_arr = Array.fromData(
                self.allocator,
                u32,
                req.prompt_tokens,
                &[_]i32{ 1, @intCast(req.prompt_tokens.len) },
            ) catch {
                req.completion.deliverError(self.io, "Failed to create prompt array");
                req.phase = .done;
                success_mask[i] = false;
                continue;
            };
            defer prompt_arr.deinit();

            const result = generation_mod.generateStep(self.model, prompt_arr, req_caches[i], &req.sampler, self.ctx) catch {
                req.completion.deliverError(self.io, "Generation failed during prefill");
                req.phase = .done;
                success_mask[i] = false;
                continue;
            };

            // Check stop tokens.
            if (generation_mod.isStopToken(result.token, req.stop_tokens) or req.max_tokens <= 1) {
                req.token_count = 1;
                const finish_reason: TokenFinishReason = if (generation_mod.isStopToken(result.token, req.stop_tokens)) .stop else .length;
                req.completion.deliverToken(self.io, result.token, "", true, finish_reason);
                req.phase = .done;
                success_mask[i] = false;
                continue;
            }

            // Store first token.
            req.generated_tokens.append(self.allocator, result.token) catch {
                req.completion.deliverError(self.io, "Failed to store generated token");
                req.phase = .done;
                success_mask[i] = false;
                continue;
            };
            req.token_count = 1;
            req.sampler.context_tokens = req.generated_tokens.items;

            // For streaming, deliver the first token immediately.
            if (req.streaming) {
                const token_text = self.tokenizer.decode(&[_]u32{result.token}, self.allocator) catch "";
                defer if (token_text.len > 0) self.allocator.free(token_text);
                req.completion.deliverToken(self.io, result.token, token_text, false, null);
            }
        }

        // Count successful prefills.
        var success_count: usize = 0;
        for (success_mask) |ok| {
            if (ok) success_count += 1;
        }

        // Merge per-request caches into batch_caches using BatchKVCache architecture.
        // Reference: mlx-lm BatchGenerator pattern:
        //   1. Each request gets independent KV caches
        //   2. Prefill each request's cache with its prompt
        //   3. Use BatchKVCache.merge() to combine caches into a batch
        //   4. Subsequent decode steps use the merged batch cache
        if (success_count > 0) {
            if (self.batch_caches) |batch_caches| {
                // Batch caches already exist: extend with new request caches.
                for (batch_caches, 0..) |batch_cache, layer_idx| {
                    if (!batch_cache.supportsExtend()) continue;

                    var sources = self.allocator.alloc(KVCacheStrategy, success_count) catch {
                        std.log.err("[Engine] failed to allocate extend sources", .{});
                        continue;
                    };
                    defer self.allocator.free(sources);

                    var idx: usize = 0;
                    for (requests, 0..) |_, i| {
                        if (success_mask[i]) {
                            sources[idx] = req_caches[i][layer_idx];
                            idx += 1;
                        }
                    }

                    batch_cache.extend(sources, self.allocator) catch |err| {
                        std.log.err("[Engine] failed to extend batch cache layer {d}: {}", .{ layer_idx, err });
                    };
                }
            } else {
                // No batch caches yet: create new BatchKVCache via merge().
                const batch_caches = self.allocator.alloc(KVCacheStrategy, self.num_layers) catch {
                    std.log.err("[Engine] failed to allocate batch_caches", .{});
                    return;
                };
                errdefer self.allocator.free(batch_caches);

                // For each layer, collect successful caches and merge.
                var any_layer_failed = false;
                for (0..self.num_layers) |layer_idx| {
                    // Collect caches for this layer.
                    var layer_caches = self.allocator.alloc(KVCacheStrategy, success_count) catch {
                        std.log.err("[Engine] failed to allocate layer_caches for merge", .{});
                        any_layer_failed = true;
                        break;
                    };
                    defer self.allocator.free(layer_caches);

                    var idx: usize = 0;
                    for (requests, 0..) |_, i| {
                        if (success_mask[i]) {
                            layer_caches[idx] = req_caches[i][layer_idx];
                            idx += 1;
                        }
                    }

                    // Merge layer caches into a new BatchKVCache.
                    const batch_kv = self.allocator.create(kvcache.batch.BatchKVCache) catch {
                        std.log.err("[Engine] failed to allocate BatchKVCache for layer {d}", .{layer_idx});
                        any_layer_failed = true;
                        break;
                    };
                    errdefer self.allocator.destroy(batch_kv);

                    batch_kv.* = kvcache.batch.BatchKVCache.merge(
                        self.allocator,
                        layer_caches,
                        self.stream,
                    ) catch |err| {
                        std.log.err("[Engine] BatchKVCache.merge failed for layer {d}: {}", .{ layer_idx, err });
                        self.allocator.destroy(batch_kv);
                        any_layer_failed = true;
                        break;
                    };

                    batch_caches[layer_idx] = batch_kv.asStrategy();
                }

                if (any_layer_failed) {
                    // Clean up any successfully created batch caches.
                    for (batch_caches[0..self.num_layers]) |cache| {
                        cache.deinit(self.allocator);
                    }
                    self.allocator.free(batch_caches);
                } else {
                    self.batch_caches = batch_caches;
                }
            }
        }

        // Add successfully-prefilled requests to active batch.
        for (requests) |req| {
            if (req.phase == .done) continue;
            req.phase = .decoding;
            req.seq_index = self.active_requests.items.len;
            self.active_requests.append(self.allocator, req) catch {
                req.completion.deliverError(self.io, "Failed to add request to active batch");
                req.phase = .done;
            };
        }
    }

    fn step(self: *EngineLoop) void {
        const batch = self.active_requests.items.len;
        if (batch == 0) return;

        // Build [batch, 1] input.
        var tokens = self.allocator.alloc(u32, batch) catch {
            std.log.err("[Engine] failed to allocate batch tokens", .{});
            return;
        };
        defer self.allocator.free(tokens);

        var samplers = self.allocator.alloc(*SamplerConfig, batch) catch {
            std.log.err("[Engine] failed to allocate samplers", .{});
            return;
        };
        defer self.allocator.free(samplers);

        for (self.active_requests.items, 0..) |req, i| {
            tokens[i] = if (req.generated_tokens.items.len > 0)
                req.generated_tokens.items[req.generated_tokens.items.len - 1]
            else
                req.prompt_tokens[req.prompt_tokens.len - 1];
            samplers[i] = &req.sampler;
        }

        const input_arr = Array.fromData(self.allocator, u32, tokens, &[_]i32{ @intCast(batch), 1 }) catch {
            std.log.err("[Engine] failed to create batch input array", .{});
            // Cannot proceed without input array - fail all requests
            for (self.active_requests.items, 0..) |_, i| {
                self.handleBatchError(i, "Failed to create batch input array");
            }
            return;
        };
        defer input_arr.deinit();

        // Use batch_caches for batched decode. Must be initialized by prefillBatch().
        const caches_to_use = self.batch_caches orelse {
            std.log.err("[Engine] batch_caches not initialized", .{});
            // Cannot proceed without caches - fail all requests
            for (self.active_requests.items, 0..) |_, i| {
                self.handleBatchError(i, "Batch caches not initialized");
            }
            return;
        };
        const results = generation_mod.generateBatchStep(self.model, input_arr, caches_to_use, samplers, self.ctx) catch {
            std.log.err("[Engine] batch generation failed", .{});
            // Batch generation failed - fail all requests in the batch
            for (self.active_requests.items, 0..) |_, i| {
                self.handleBatchError(i, "Batch generation failed");
            }
            return;
        };
        defer self.allocator.free(results);

        // Process results and track completed requests.
        var keep = self.allocator.alloc(bool, batch) catch {
            std.log.err("[Engine] failed to allocate keep array", .{});
            return;
        };
        defer self.allocator.free(keep);
        @memset(keep, true);

        var completed: usize = 0;
        for (self.active_requests.items, 0..) |req, i| {
            const result = results[i];
            req.token_count += 1;

            // Decode token.
            const token_text = self.tokenizer.decode(&[_]u32{result.token}, self.allocator) catch "";
            defer if (token_text.len > 0) self.allocator.free(token_text);

            // Check stop tokens.
            const is_stop = generation_mod.isStopToken(result.token, req.stop_tokens);
            const is_done = is_stop or req.token_count >= req.max_tokens;

            // Store token.
            req.generated_tokens.append(self.allocator, result.token) catch {
                req.completion.deliverError(self.io, "Failed to store token");
                keep[i] = false;
                completed += 1;
                continue;
            };

            // Update sampler context.
            req.sampler.context_tokens = req.generated_tokens.items;

            // Deliver token.
            if (req.streaming) {
                const finish_reason: ?TokenFinishReason = if (is_done) (if (is_stop) .stop else .length) else null;
                req.completion.deliverToken(self.io, result.token, token_text, is_done, finish_reason);
            }

            if (is_done) {
                if (!req.streaming) {
                    // Deliver full text for non-streaming.
                    const full_text = self.tokenizer.decode(req.generated_tokens.items, self.allocator) catch "";
                    defer self.allocator.free(full_text);
                    req.completion.deliverToken(self.io, 0, full_text, true, if (is_stop) .stop else .length);
                }
                keep[i] = false;
                completed += 1;
            }
        }

        // Filter completed requests.
        if (completed > 0) {
            // Build list of batch indices to keep (not seq_index).
            var keep_indices = self.allocator.alloc(usize, batch - completed) catch {
                std.log.err("[Engine] failed to allocate keep_indices", .{});
                return;
            };
            defer self.allocator.free(keep_indices);

            var idx: usize = 0;
            for (0..batch) |i| {
                if (keep[i]) {
                    keep_indices[idx] = i;
                    idx += 1;
                }
            }

            // Filter caches.
            if (self.batch_caches) |batch_caches| {
                for (batch_caches) |cache| {
                    if (cache.supportsFilter()) {
                        cache.filter(keep_indices, self.allocator) catch {
                            std.log.err("[Engine] failed to filter cache", .{});
                        };
                    }
                }
            }

            // Update active_requests and seq_index.
            var new_active = std.ArrayList(*RequestState).empty;
            for (self.active_requests.items, 0..) |req, i| {
                if (keep[i]) {
                    new_active.append(self.allocator, req) catch {};
                } else {
                    req.phase = .done;
                    logRequestCompletion(self.io, req);
                }
            }
            self.active_requests.deinit(self.allocator);
            self.active_requests = new_active;

            // Update seq_index for remaining requests to match their new positions.
            for (new_active.items, 0..) |req, new_idx| {
                req.seq_index = new_idx;
            }
        }
    }

    /// Handle a failed request during batch processing.
    /// Removes the request from the batch and delivers an error to its client.
    /// Other requests in the batch continue processing.
    fn handleBatchError(self: *EngineLoop, failed_req_index: usize, error_message: []const u8) void {
        if (failed_req_index >= self.active_requests.items.len) {
            std.log.err("[Engine] handleBatchError: invalid index {d}", .{failed_req_index});
            return;
        }

        const req = self.active_requests.items[failed_req_index];
        std.log.err("[Engine] Request {d} failed: {s}", .{ req.id, error_message });

        // Mark the request as done.
        req.phase = .done;

        // Deliver error to the client.
        req.completion.deliverError(self.io, error_message);

        // Log the failed request completion.
        logRequestCompletion(self.io, req);

        // If this is the only request, clear the batch.
        if (self.active_requests.items.len == 1) {
            self.active_requests.clearAndFree(self.allocator);
            return;
        }

        // Build list of indices to keep (all except the failed one).
        const keep_count = self.active_requests.items.len - 1;
        var keep_indices = self.allocator.alloc(usize, keep_count) catch {
            std.log.err("[Engine] failed to allocate keep_indices for error handling", .{});
            // Fallback: just remove from active_requests without cache filtering.
            _ = self.active_requests.orderedRemove(failed_req_index);
            return;
        };
        defer self.allocator.free(keep_indices);

        var idx: usize = 0;
        for (0..self.active_requests.items.len) |i| {
            if (i != failed_req_index) {
                keep_indices[idx] = i;
                idx += 1;
            }
        }

        // Filter caches to remove the failed request's slot.
        if (self.batch_caches) |batch_caches| {
            for (batch_caches) |cache| {
                if (cache.supportsFilter()) {
                    cache.filter(keep_indices, self.allocator) catch {
                        std.log.err("[Engine] failed to filter cache after error", .{});
                    };
                }
            }
        }

        // Update active_requests: remove the failed request.
        var new_active = std.ArrayList(*RequestState).empty;
        for (self.active_requests.items, 0..) |r, i| {
            if (i != failed_req_index) {
                new_active.append(self.allocator, r) catch {};
            }
        }
        self.active_requests.deinit(self.allocator);
        self.active_requests = new_active;

        // Update seq_index for remaining requests.
        for (new_active.items, 0..) |r, new_idx| {
            r.seq_index = new_idx;
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

        // Context for streaming callback
        const StreamCtx = struct {
            engine: *EngineLoop,
            req: *RequestState,
            token_count: usize,
        };
        var stream_ctx = StreamCtx{
            .engine = self,
            .req = req,
            .token_count = 0,
        };

        // Callback function that delivers tokens immediately
        const streamCallback = struct {
            fn callback(ctx_ptr: *anyopaque, token: u32, is_final: bool) void {
                const ctx: *StreamCtx = @ptrCast(@alignCast(ctx_ptr));
                ctx.token_count += 1;

                // Decode token to text
                const token_text = ctx.engine.tokenizer.decode(&[_]u32{token}, ctx.engine.allocator) catch "";
                defer if (token_text.len > 0) ctx.engine.allocator.free(token_text);

                // Deliver token immediately
                ctx.req.completion.deliverToken(
                    ctx.engine.io,
                    token,
                    token_text,
                    is_final,
                    if (is_final) .stop else null,
                );
            }
        }.callback;

        // Generate with streaming callback - tokens are delivered immediately
        const tokens = model.generateWithCallback(
            req.prompt_tokens,
            req.max_tokens,
            &sampler,
            caches,
            self.stream,
            &stream_ctx,
            streamCallback,
        ) catch |err| {
            std.log.err("EngineLoop: DSV4 generate failed: {}", .{err});
            req.completion.deliverError(self.io, "Generation failed");
            return;
        };
        defer self.allocator.free(tokens);

        // Record generated token count for logging
        req.token_count = @intCast(tokens.len);

        // Ensure a final done event is sent if not already
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

    /// Create per-request private KV caches (used for individual prefill).
    /// Uses StandardKVCache for non-DSV4 models (BatchKVCache is reserved
    /// for shared batched decode and cannot be used for empty-offset prefill).
    fn createCaches(self: *EngineLoop) ![]KVCacheStrategy {
        const allocator = self.allocator;
        const stream = self.stream;

        if (self.config_content) |content| {
            if (self.dsv4_model != null) {
                // DSV4 direct path
                const dsv4_config = try dsv4_loader.parseDSV4Config(allocator, content);
                defer dsv4_config.deinitClone(allocator);
                return try dsv4_loader.makeV4Caches(allocator, &dsv4_config, stream);
            }

            // DSV4 VTable path (fallback)
            const dsv4_config = try dsv4_loader.parseDSV4Config(allocator, content);
            defer dsv4_config.deinitClone(allocator);

            const caches = try allocator.alloc(KVCacheStrategy, dsv4_config.num_hidden_layers);
            errdefer allocator.free(caches);

            const effective_max_seq = @min(dsv4_config.max_position_embeddings, 131072);
            for (0..dsv4_config.num_hidden_layers) |i| {
                const compress_ratio = if (i < dsv4_config.compress_ratios.len) dsv4_config.compress_ratios[i] else 0;
                const layer_max_seq = if (compress_ratio > 1)
                    effective_max_seq / compress_ratio
                else
                    effective_max_seq;

                const lc = kvcache.LayerConfig{
                    .batch_size = 1,
                    .num_heads = dsv4_config.num_key_value_heads,
                    .num_kv_heads = dsv4_config.num_key_value_heads,
                    .head_dim = dsv4_config.head_dim,
                    .max_seq_len = layer_max_seq,
                    .dtype = .float32,
                };
                caches[i] = try kvcache.createStandard(allocator, lc, stream);
            }
            return caches;
        }

        // Non-DSV4 path: create standard caches per layer.
        const lc = self.layer_config orelse return error.NoLayerConfig;
        const caches = try allocator.alloc(KVCacheStrategy, self.num_layers);
        errdefer allocator.free(caches);

        for (0..self.num_layers) |i| {
            caches[i] = try kvcache.createStandard(allocator, lc, stream);
        }
        return caches;
    }
};

/// Create KV caches (shared across all requests in the batch).
fn createCachesShared(
    allocator: std.mem.Allocator,
    config_content: ?[]const u8,
    layer_config: ?LayerConfig,
    num_layers: usize,
    dsv4_model: ?*dsv4_mod.DSV4Model,
    stream: c.c.mlx_stream,
    batch_size: usize,
) ![]KVCacheStrategy {
    if (config_content) |content| {
        if (dsv4_model != null) {
            // DSV4 direct path: use makeV4Caches for heterogeneous caches.
            const dsv4_config = try dsv4_loader.parseDSV4Config(allocator, content);
            defer dsv4_config.deinitClone(allocator);
            return try dsv4_loader.makeV4Caches(allocator, &dsv4_config, stream);
        }

        // DSV4 VTable path (fallback): parse config and create standard caches manually.
        const dsv4_config = try dsv4_loader.parseDSV4Config(allocator, content);
        defer dsv4_config.deinitClone(allocator);

        const caches = try allocator.alloc(KVCacheStrategy, dsv4_config.num_hidden_layers);
        errdefer allocator.free(caches);

        const effective_max_seq = @min(dsv4_config.max_position_embeddings, 131072);

        for (0..dsv4_config.num_hidden_layers) |i| {
            const compress_ratio = if (i < dsv4_config.compress_ratios.len) dsv4_config.compress_ratios[i] else 0;
            const layer_max_seq = if (compress_ratio > 1)
                effective_max_seq / compress_ratio
            else
                effective_max_seq;

            const lc = kvcache.LayerConfig{
                .batch_size = batch_size,
                .num_heads = dsv4_config.num_key_value_heads,
                .num_kv_heads = dsv4_config.num_key_value_heads,
                .head_dim = dsv4_config.head_dim,
                .max_seq_len = layer_max_seq,
                .dtype = .float32,
            };
            caches[i] = try kvcache.createStandard(allocator, lc, stream);
        }
        return caches;
    }

    // Non-DSV4 path: create standard caches per layer.
    const lc = layer_config orelse return error.NoLayerConfig;
    const caches = try allocator.alloc(KVCacheStrategy, num_layers);
    errdefer allocator.free(caches);

    for (0..num_layers) |i| {
        caches[i] = try kvcache.createStandard(allocator, lc, stream);
    }
    return caches;
}
