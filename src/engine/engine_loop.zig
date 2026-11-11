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
const speculative_mod = @import("../speculative.zig");
const guided_mod = @import("../guided.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const content_class = @import("content_class.zig");

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
const qwen38_gdn_prof = @import("../models/qwen38_gdn.zig");
const verify_qmm_mod = @import("../models/verify_qmm.zig");
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
    stop_tokens: []const u32,
    text_buffer: std.ArrayList(u8),
    token_count: usize,
    max_tokens: usize,
    stopped: bool,

    fn init(allocator: std.mem.Allocator, io: std.Io, signal: *CompletionSignal, tokenizer: root.tokenizer.TokenizerStrategy, stop_strings: ?[]const []const u8, stop_tokens: []const u32, max_tokens: usize) StreamCallbackCtx {
        return .{
            .signal = signal,
            .io = io,
            .tokenizer = tokenizer,
            .allocator = allocator,
            .stop_strings = stop_strings,
            .stop_tokens = stop_tokens,
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

        // Resolve the real finish verdict: stop token → .stop, otherwise
        // is_done means max_tokens truncation → .length.
        var finish_reason: ?TokenFinishReason = null;
        if (is_done) {
            finish_reason = if (generation_mod.isStopToken(token, self.stop_tokens)) .stop else .length;
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

/// Derive the wire finish reason from a finished token sequence: a trailing
/// stop token → .stop; hitting the max_tokens cap → .length.
fn finishReasonForTokens(tokens: []const u32, gen_config: GenerateConfig) TokenFinishReason {
    if (tokens.len > 0 and generation_mod.isStopToken(tokens[tokens.len - 1], gen_config.stop_tokens)) return .stop;
    if (tokens.len >= gen_config.max_tokens) return .length;
    return .stop;
}

/// Three-way finish verdict for streaming callbacks that see only the final
/// token: configured stop token → .stop; max_tokens cap reached → .length;
/// otherwise the model's own internal EOS ended generation → .stop. (DSV4
/// stops on its internal EOS token, which is not in `stop_tokens`.)
fn streamFinalReason(token: u32, stop_tokens: []const u32, token_count: usize, max_tokens: usize) TokenFinishReason {
    if (generation_mod.isStopToken(token, stop_tokens)) return .stop;
    if (token_count >= max_tokens) return .length;
    return .stop;
}

/// softmax(logits / temperature) over the full vocab, on CPU.
fn softmaxProbs(logits: []const f32, temperature: f32, allocator: std.mem.Allocator) ![]f32 {
    const probs = try allocator.alloc(f32, logits.len);
    errdefer allocator.free(probs);
    const t = if (temperature > 1e-6) temperature else 1e-6;
    var max_l: f32 = -std.math.inf(f32);
    for (logits) |l| max_l = @max(max_l, l / t);
    var sum: f64 = 0;
    for (logits, 0..) |l, i| {
        const e = @exp((l / t) - max_l);
        probs[i] = e;
        sum += e;
    }
    const inv: f32 = @floatCast(1.0 / sum);
    for (probs) |*p| p.* *= inv;
    return probs;
}

/// Speculative-sampling acceptance probability for a draft token:
/// min(1, p_target / q_draft) with both distributions temperature-shaped.
fn monotonicNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

fn mtpAcceptProb(target_logits: Array, draft_logits: Array, token: u32, temperature: f32, allocator: std.mem.Allocator) !f32 {
    const p_data = try target_logits.dataSlice(f32);
    const q_data = try draft_logits.dataSlice(f32);
    const p = try softmaxProbs(p_data, temperature, allocator);
    defer allocator.free(p);
    const q = try softmaxProbs(q_data, temperature, allocator);
    defer allocator.free(q);
    const qd = q[token];
    if (qd <= 0) return 1.0;
    return @min(1.0, p[token] / qd);
}

/// Rejection resample from the adjusted distribution (p_target - q_draft)+,
/// normalized — the standard speculative-sampling correction step.
fn mtpResampleAdjusted(target_logits: Array, draft_logits: Array, temperature: f32, prng: *std.Random.DefaultPrng, allocator: std.mem.Allocator) !sampling_mod.SampleResult {
    const p_data = try target_logits.dataSlice(f32);
    const q_data = try draft_logits.dataSlice(f32);
    const p = try softmaxProbs(p_data, temperature, allocator);
    defer allocator.free(p);
    const q = try softmaxProbs(q_data, temperature, allocator);
    defer allocator.free(q);
    const adj = try allocator.alloc(f32, p.len);
    defer allocator.free(adj);
    var sum: f64 = 0;
    for (p, q, 0..) |pp, qq, i| {
        adj[i] = @max(0, pp - qq);
        sum += adj[i];
    }
    if (sum <= 0) {
        // Degenerate: fall back to the plain target distribution argmax.
        var best: usize = 0;
        for (p[1..], 1..) |v, i| {
            if (v > p[best]) best = i;
        }
        return .{ .token = @intCast(best), .logprob = 0 };
    }
    const r = prng.random().float(f32) * @as(f32, @floatCast(sum));
    var acc: f32 = 0;
    for (adj, 0..) |a, i| {
        acc += a;
        if (r < acc) return .{ .token = @intCast(i), .logprob = 0 };
    }
    return .{ .token = @intCast(adj.len - 1), .logprob = 0 };
}

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
    /// When set, requests are processed serially through the VTable with
    /// per-request caches passed through as-is (the model keeps its own
    /// internal state — e.g. GLM-5.3's KDA/MLA caches — and the batched
    /// KV-cache merge path is bypassed).
    vtable_serial: bool = false,
    /// Maximum number of requests to batch together in the decode phase.
    max_batch_size: usize = 8,
    /// Speculative decoding: n-gram size for draft proposal (null = disabled).
    speculative_ngram: ?usize = null,
    /// MTP speculative decoding: number of draft tokens per round (null =
    /// disabled). Requires a model with `ModelVTable.mtp` (Qwen3.8).
    speculative_mtp: ?usize = null,
    /// Prefix cache: max number of prompts to cache KV state for (0 = disabled).
    prefix_cache_max_entries: usize = 16,
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
    vtable_serial: bool,
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
    /// Speculative decoding: n-gram size for draft proposal (null = disabled).
    speculative_ngram: ?usize,
    /// MTP speculative decoding: UPPER BOUND on draft tokens per round
    /// (null = disabled); the per-round depth is chosen by `mtp_depth`.
    speculative_mtp: ?usize,
    /// Online MTP draft-depth controller. Lives on the engine rather than
    /// per request so its (content-independent) cost estimates survive
    /// across requests; see `MtpDepthController.resetForRequest`.
    mtp_depth: ?MtpDepthController,
    /// Prefix cache for skipping prefill on repeated prompts.
    prefix_cache: ?prefix_cache_mod.PrefixCacheManager,

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
            .vtable_serial = config.vtable_serial,
            .running = std.atomic.Value(bool).init(true),
            .batch_caches = null,
            .max_batch_size = config.max_batch_size,
            .seq_index_counter = 0,
            .active_requests = std.ArrayList(*RequestState).empty,
            .speculative_ngram = config.speculative_ngram,
            .speculative_mtp = config.speculative_mtp,
            .mtp_depth = null,
            .prefix_cache = if (config.prefix_cache_max_entries > 0)
                prefix_cache_mod.PrefixCacheManager.init(config.allocator, config.prefix_cache_max_entries, config.stream)
            else
                null,
        };
    }

    pub fn deinit(self: *EngineLoop) void {
        if (self.prefix_cache) |*pc| pc.deinit();
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
                if (self.dsv4_model != null or self.vtable_serial) {
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
        const engine_start = std.c.mach_absolute_time();
        req.start_time_ns = @intCast(engine_start);
        // Queue wait: time from HTTP push to engine pick-up
        const queue_wait_ms = @as(f64, @floatFromInt(engine_start - req.queued_time_ns)) / 1_000_000.0;
        std.log.info("[Engine] Request {d} dequeued, queue_wait={d:.1}ms", .{ req.id, queue_wait_ms });

        var caches: []KVCacheStrategy = undefined;
        var owns_caches = true;
        var prefix_hit = false;

        // Engine-owned copy of the prompt for prefix-cache use. The HTTP
        // handler frees req.prompt_tokens as soon as generation signals
        // completion, which can precede the post-generation pc.store below
        // (observed as "@memcpy arguments alias" in PrefixCacheManager.store
        // when the allocator recycled the freed block into the dupe target).
        var pc_prompt_owned = false;
        const pc_prompt: []const u32 = if (self.prefix_cache != null and !self.vtable_serial)
            self.allocator.dupe(u32, req.prompt_tokens) catch |err| blk: {
                std.log.warn("[Engine] prompt dupe for prefix cache failed: {}", .{err});
                break :blk req.prompt_tokens;
            }
        else
            req.prompt_tokens;
        pc_prompt_owned = pc_prompt.ptr != req.prompt_tokens.ptr;
        defer if (pc_prompt_owned) self.allocator.free(pc_prompt);

        // Try prefix cache first (not for vtable_serial models — their state
        // is internal and cannot be restored from a KV snapshot).
        if (!self.vtable_serial) {
            if (self.prefix_cache) |*pc| {
                if (pc.lookup(pc_prompt)) |maybe_cloned| {
                    if (maybe_cloned) |cloned| {
                        caches = cloned;
                        prefix_hit = true;
                        std.log.info("[Engine] Prefix cache hit for request {d}, skipping prefill", .{req.id});
                    }
                } else |err| {
                    std.log.warn("[Engine] Prefix cache lookup failed: {}", .{err});
                }
            }
        }

        if (!prefix_hit) {
            // Create per-request caches to avoid cross-request cache contamination.
            caches = self.createCaches() catch |err| {
                std.log.err("EngineLoop: failed to create caches: {}", .{err});
                req.completion.deliverError(self.io, "Failed to create KV caches");
                return;
            };
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
                self.processDSV4StreamingRequest(req, caches, gen_config, model, prefix_hit);
            } else {
                self.processDSV4NonStreamingRequest(req, caches, gen_config, model, prefix_hit);
            }
        } else {
            // Check for guided decoding constraints (non-streaming only).
            // Guided decoding applies logit masks at each step, which is
            // incompatible with the streaming callback architecture.
            if (!req.streaming and (req.guided_json_schema != null or req.guided_regex != null)) {
                self.processGuidedRequest(req, caches, gen_config);
            } else if (req.streaming) {
                self.processStreamingRequest(req, caches, gen_config);
            } else {
                self.processNonStreamingRequest(req, caches, gen_config);
            }
        }

        // After generation: store caches if this was a miss. vtable_serial
        // models are excluded like they are at lookup above: their state
        // cannot be restored from a KV snapshot, and there pc_prompt is NOT
        // engine-owned (req.prompt_tokens, freed by the HTTP handler when
        // generation signals completion) — storing it was a use-after-free
        // that panicked as "@memcpy arguments alias" in store's dupe.
        if (!prefix_hit and !self.vtable_serial and self.prefix_cache != null) {
            const pc = &self.prefix_cache.?;
            if (!pc.isCached(pc_prompt)) {
                if (pc.store(pc_prompt, caches)) {
                    owns_caches = false; // Caches now owned by prefix cache.
                } else |err| {
                    std.log.warn("[Engine] Failed to store caches in prefix cache: {}", .{err});
                }
            }
        }

        // Deinit caches if we still own them.
        if (owns_caches) {
            for (caches) |cache| cache.deinit(self.allocator);
            self.allocator.free(caches);
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

            const pf_t0: i128 = if (std.c.getenv("DMLX_PREFILL_TRACE") != null) monotonicNs() else 0;
            const result = generation_mod.generateStep(self.model, prompt_arr, req_caches[i], &req.sampler, self.ctx) catch {
                req.completion.deliverError(self.io, "Generation failed during prefill");
                req.phase = .done;
                success_mask[i] = false;
                continue;
            };
            if (pf_t0 != 0) {
                std.log.info("[PREFILL_TRACE] plain req={d} prompt={d} prefill_ms={d:.1}", .{ req.id, req.prompt_tokens.len, @as(f64, @floatFromInt(monotonicNs() - pf_t0)) / 1e6 });
            }

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
            req.stop_tokens,
            req.max_tokens,
        );
        defer stream_ctx.deinit();

        if (req.vision) |vis| {
            generation_mod.streamGenerateCtxVision(
                self.model,
                req.prompt_tokens,
                gen_config,
                caches,
                self.ctx,
                vis,
                @ptrCast(&stream_ctx),
                &StreamCallbackCtx.callback,
            ) catch |err| {
                std.log.err("EngineLoop: streamGenerateCtxVision failed: {}", .{err});
                req.completion.deliverError(self.io, "Generation failed");
                return;
            };
        } else generation_mod.streamGenerateCtx(
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
        // MTP speculative decoding takes precedence when enabled and the
        // model supports it (Qwen3.8 drafter); n-gram is the fallback when
        // --speculative-mtp was requested but no drafter is available.
        if (self.speculative_mtp) |n_drafts| {
            if (self.tryMtpRequest(req, caches, gen_config, n_drafts)) return;
        }
        // If speculative decoding is enabled, use the speculative path.
        if (req.vision == null) {
            if (self.speculative_ngram) |ngram_n| {
                self.processSpeculativeRequest(req, caches, gen_config, ngram_n);
                return;
            }
        }

        const pf_t0: i128 = if (std.c.getenv("DMLX_PREFILL_TRACE") != null) monotonicNs() else 0;
        const tokens = if (req.vision) |vis|
            generation_mod.generateWithVision(
                self.model,
                req.prompt_tokens,
                gen_config,
                caches,
                self.ctx,
                vis,
            ) catch |err| {
                std.log.err("EngineLoop: generateWithVision failed: {}", .{err});
                req.completion.deliverError(self.io, "Generation failed");
                return;
            }
        else
            generation_mod.generate(
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
        if (pf_t0 != 0) {
            std.log.info("[PREFILL_TRACE] plain req={d} prompt={d} generate_ms={d:.1}", .{ req.id, req.prompt_tokens.len, @as(f64, @floatFromInt(monotonicNs() - pf_t0)) / 1e6 });
        }

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
        req.completion.deliverToken(self.io, 0, final_text, true, finishReasonForTokens(tokens, gen_config));
    }

    /// Speculative decoding path: uses n-gram draft proposal + target model verification.
    /// Generates multiple tokens per forward pass when draft proposals are accepted.
    ///
    /// Architecture (following mlx-lm / omlx pattern):
    ///   1. Generate tokens normally until enough context for n-gram matching
    ///   2. Use NgramDrafter to propose continuation tokens from context
    ///   3. Verify proposals in a single forward pass via speculative sampling
    ///   4. Accept verified tokens, resample on rejection
    ///   5. Repeat until max_tokens or stop condition
    fn processSpeculativeRequest(
        self: *EngineLoop,
        req: *RequestState,
        caches: []KVCacheStrategy,
        gen_config: GenerateConfig,
        ngram_n: usize,
    ) void {
        const drafter = speculative_mod.NgramDrafter{ .n = ngram_n };
        const max_draft_tokens: usize = 5; // Number of tokens to propose per step

        // Build context: prompt + generated tokens
        var context = std.ArrayList(u32).empty;
        defer context.deinit(self.allocator);
        context.appendSlice(self.allocator, req.prompt_tokens) catch {
            req.completion.deliverError(self.io, "Speculative: context allocation failed");
            return;
        };

        // Initial prefill: run the prompt through the model to populate caches
        const first_result = generation_mod.generateStep(
            self.model,
            array_mod.Array.fromData(
                self.allocator,
                u32,
                req.prompt_tokens,
                &[_]i32{ 1, @intCast(req.prompt_tokens.len) },
            ) catch {
                req.completion.deliverError(self.io, "Speculative: failed to create prompt array");
                return;
            },
            caches,
            &req.sampler,
            self.ctx,
        ) catch |err| {
            std.log.err("EngineLoop: speculative prefill failed: {}", .{err});
            req.completion.deliverError(self.io, "Speculative: prefill failed");
            return;
        };

        context.append(self.allocator, first_result.token) catch {
            req.completion.deliverError(self.io, "Speculative: token append failed");
            return;
        };
        var total_generated: usize = 1;

        // Check if first token is a stop token
        if (generation_mod.isStopToken(first_result.token, gen_config.stop_tokens)) {
            const text = self.tokenizer.decode(context.items[req.prompt_tokens.len..], self.allocator) catch {
                req.completion.deliverError(self.io, "Failed to decode tokens");
                return;
            };
            defer self.allocator.free(text);
            req.token_count = total_generated;
            req.completion.deliverToken(self.io, 0, text, true, .stop);
            return;
        }

        // Main speculative decoding loop
        while (total_generated < gen_config.max_tokens) {
            // Try to propose draft tokens from context
            const draft_tokens = drafter.propose(context.items, max_draft_tokens, self.allocator);

            if (draft_tokens) |draft| {
                defer self.allocator.free(draft);

                // Verify draft tokens against the target model
                const verify_result = speculative_mod.verifyDraft(
                    self.model,
                    context.items,
                    draft,
                    caches,
                    self.ctx,
                    gen_config.seed +% total_generated,
                ) catch {
                    // Verification failed — fall back to normal generation for this step
                    const step_result = generation_mod.generateStep(
                        self.model,
                        array_mod.Array.fromData(
                            self.allocator,
                            u32,
                            context.items[context.items.len - 1 ..],
                            &[_]i32{ 1, 1 },
                        ) catch break,
                        caches,
                        &req.sampler,
                        self.ctx,
                    ) catch break;

                    context.append(self.allocator, step_result.token) catch break;
                    total_generated += 1;

                    if (generation_mod.isStopToken(step_result.token, gen_config.stop_tokens)) break;
                    continue;
                };
                defer self.allocator.free(verify_result.tokens);

                // Append accepted + bonus tokens to context
                var stopped = false;
                for (verify_result.tokens) |token| {
                    if (generation_mod.isStopToken(token, gen_config.stop_tokens)) {
                        stopped = true;
                        break;
                    }
                    context.append(self.allocator, token) catch break;
                    total_generated += 1;
                    if (total_generated >= gen_config.max_tokens) break;
                }

                if (stopped or total_generated >= gen_config.max_tokens) break;
            } else {
                // No draft available — fall back to normal single-token generation
                const input_token = context.items[context.items.len - 1];
                const step_result = generation_mod.generateStep(
                    self.model,
                    array_mod.Array.fromData(
                        self.allocator,
                        u32,
                        &[_]u32{input_token},
                        &[_]i32{ 1, 1 },
                    ) catch break,
                    caches,
                    &req.sampler,
                    self.ctx,
                ) catch break;

                context.append(self.allocator, step_result.token) catch break;
                total_generated += 1;

                if (generation_mod.isStopToken(step_result.token, gen_config.stop_tokens)) break;
            }
        }

        // Decode generated tokens to text
        const generated_slice = context.items[req.prompt_tokens.len..];
        const text = self.tokenizer.decode(generated_slice, self.allocator) catch {
            req.completion.deliverError(self.io, "Failed to decode tokens");
            return;
        };
        defer self.allocator.free(text);

        // Check stop strings and truncate if needed
        var final_text = text;
        if (req.stop_strings) |stop_strings| {
            for (stop_strings) |stop_str| {
                if (std.mem.indexOf(u8, text, stop_str)) |idx| {
                    final_text = text[0..idx];
                    break;
                }
            }
        }

        req.token_count = total_generated;
        req.completion.deliverToken(self.io, 0, final_text, true, finishReasonForTokens(generated_slice, gen_config));
    }

    /// Try the MTP path; returns false (after one warning) when the model has
    /// no MTP capability or the drafter fails to load — caller falls back.
    fn tryMtpRequest(
        self: *EngineLoop,
        req: *RequestState,
        caches: []KVCacheStrategy,
        gen_config: GenerateConfig,
        n_drafts: usize,
    ) bool {
        const ops_mtp = self.model.mtp orelse {
            std.log.warn("EngineLoop: --speculative-mtp set but model has no MTP capability; using normal decoding", .{});
            return false;
        };
        const ok = ops_mtp.enable(self.model.ptr, n_drafts) catch |err| {
            std.log.warn("EngineLoop: MTP drafter load failed ({}); using normal decoding", .{err});
            return false;
        };
        if (!ok) return false;
        self.processMtpRequest(req, caches, gen_config, ops_mtp, n_drafts);
        return true;
    }

    /// Online draft-depth controller for MTP speculative decoding.
    ///
    /// `--speculative-mtp N` is treated as the UPPER BOUND on draft depth,
    /// not a fixed depth, because the optimal depth is content-dependent.
    /// Per round the drafter cost is linear in depth (measured 4.5 ms per
    /// draft position) but the verify cost is superlinear: MLX dispatches
    /// `M < get_qmv_batch_limit` to the batched qmv kernel, whose cost climbs
    /// sharply once M >= 3 (measured qmm per 64-layer forward at M=1..5:
    /// 59.4 / 62.0 / 68.5 / 84.6 / 105.8 ms; the split-K GEMM path only
    /// engages at M >= 14 and is far worse in absolute terms, so there is no
    /// cheaper kernel to switch to). Acceptance meanwhile falls with depth,
    /// and how fast it falls depends entirely on the content: on a code
    /// prompt depth 2 wins, on a highly predictable prompt depth 3 wins.
    ///
    /// So instead of baking a constant, estimate ms-per-emitted-token for
    /// every depth and run the best one. Set `DMLX_MTP_FIXED_N=1` to pin the
    /// depth at the flag value (A/B).
    ///
    /// Estimator note: scoring each depth by an EMA of its own realized
    /// ms/token does NOT work — it was tried and lost to a fixed depth.
    /// Round cost is near-deterministic (measured 83/96/118/146 ms at depth
    /// 1/2/3/4, +-1 ms, and a depth switch costs nothing measurable), but
    /// tokens-per-round is a high-variance integer in 1..N+1. Estimating the
    /// ratio per depth from the handful of rounds that depth gets means a
    /// couple of unlucky zero-acceptance rounds poison a deep arm, greedy
    /// exploitation then starves it of samples, and the controller collapses
    /// onto depth 1.
    ///
    /// So estimate the two factors separately, which also lets every round
    /// inform every depth:
    ///   cost(n)   — running mean of round wall time at depth n (tiny variance)
    ///   p_i       — conditional probability that draft i is accepted given
    ///               the first i-1 were. A round at depth n observes p_1..p_j
    ///               for every j it reached, so acceptance evidence is POOLED
    ///               across depths instead of being per-arm.
    /// Expected tokens at depth n is then analytic for every n, including
    /// depths not run recently:
    ///   tokens(n) = 1 + sum_{i=1..n} prod_{j<=i} p_j
    /// and we run argmin_n cost(n)/tokens(n). A depth probe at max_n every
    /// `deep_probe_period` rounds refreshes p_i for the deep tail (which the
    /// chosen depth stops observing) at a cost of one round in 32.
    const MtpDepthController = struct {
        const max_depth = 16;
        /// Rounds between forced probes. A probe both refreshes that depth's
        /// cost estimate and (for deep probes) the pooled p_i tail.
        const probe_period = 16;
        /// The deep arm gets its OWN cadence, because the descending rotation
        /// below cannot reach it inside a real request. The rotation returns
        /// `max_n - (rounds / probe_period) % max_n`, so any one depth comes
        /// up every `max_n * probe_period` rounds = 112 at max_n = 7, while a
        /// 128-token request runs ~30 rounds. Depth 7 was therefore probed
        /// exactly never after exploration ended.
        ///
        /// Measured consequence (scripts/mtp_math_mode_probe.py,
        /// artifacts/mtp_math_modes.json, 8 draws/arm on the math prompt):
        /// requests split into two depth fixed points,
        ///   slow ~26 t/s  N=2:28r  N=7:4r   N=7 reads 32.0 ms/tok, E[t]=4.1
        ///   fast ~30 t/s  N=2:13r  N=7:13r  N=7 reads 28.4 ms/tok, E[t]=4.7
        /// and the `N=5:1r N=6:1r` tail in the slow trace is the rotation
        /// hitting slots 1 and 2 at rounds 16 and 32 — never slot 0.
        /// Depth 7 is genuinely ~15% better per token here (27.1-28.4 vs
        /// 31.3-32.8); the controller just could not get back to it. Whether
        /// a request lands fast or slow then depends on whether exploration
        /// happened to straddle math's low-acceptance opening phase, which is
        /// a coin flip, and it swamped every cost-model tweak we A/B'd.
        /// DEFAULT 0 (off): probing the deep arm more measured WORSE, and the
        /// reason reframes the whole problem. Set to 12 to re-enable.
        ///
        /// A/B at 8 draws per arm (artifacts/mtp_math_modes_deepprobe.json):
        /// a 12-round deep cadence plus the score latch below did fix
        /// reachability — all 8 draws got 7-13 rounds at depth 7, versus
        /// 3 of 8 stuck at `N=7:4r` without it — and still LOST, 27.4 vs
        /// 28.9 tok/s median. The traces show the deep arm's own estimate
        /// degrading as it was sampled more:
        ///   with cadence     N=7 reads 28.8-35.3 ms/tok, E[t] 3.71-4.53
        ///   parked (no cad.) N=7 reads 26.8-29.9 ms/tok, E[t] 4.46-4.93
        ///
        /// Acceptance evidence is decayed EVERY round (0.9), and a shallow
        /// round observes nothing past p_2, so while the incumbent sits at
        /// N=2 the `reached[3..7]` counters decay toward the prior. Returning
        /// to depth 7 for one or two rounds out of twelve therefore measures
        /// it against a tail that the intervening shallow rounds just erased,
        /// which underestimates E[t] and sends the argmin straight back to
        /// N=2. Interleaved sampling is self-defeating here; PARKING is what
        /// keeps the deep tail fresh, and it is self-sustaining once reached.
        ///
        /// So the slow fixed point is not "could not reach depth 7", it is
        /// "left depth 7 and could never get a fair re-measurement". The
        /// lever is therefore how the controller first parks, i.e. the
        /// exploration early exit, not the probe cadence.
        const deep_probe_period_default = 0;
        /// Cadence for the ladder probe in `pick` (rungs just above the
        /// incumbent). 6 rather than the rotation's 16: a ~30-round request
        /// has to get several samples of the rung for the argmin to act on
        /// it, and the rung is a cheaper round than a max_n probe.
        const ladder_probe_period_default = 6;
        /// Decayed-`reached` level below which a rung counts as unobserved.
        const ladder_stale_floor: f32 = 1.0;
        /// Decayed-`reached` at or above which the rung's score is trusted.
        /// Under decay 0.9 a rung probed every 6 rounds settles near 1.7, so
        /// 2.0 means "sampled at least as often as the probe cadence".
        const ladder_confident_floor: f32 = 2.0;
        /// How much worse a trusted rung may score before probing stops.
        ///
        /// Both this and `exploreGiveUpRatio` were set by eye and both MISS on
        /// code by a hair, which is expensive: the depth traces
        /// (artifacts/code_modes_20261103.json) show every draw parking at
        /// N=3 with N=3 the best depth in all ten, while 15 of ~45 rounds go
        /// to N=5 and N=7 —
        ///   draw[1]  N=3 31.7 ms/tok   N=5 33.5 (1.057x)   N=7 36.0 (1.136x)
        ///   draw[7]  N=3 46.7 ms/tok   N=5 54.5 (1.167x)   N=7 59.7 (1.278x)
        /// 1.057 is under the 1.10 give-up here and 1.136 is under the 1.15
        /// explore give-up, so neither hatch opens and code pays a ~33% round
        /// tax on depths it has already proven worse. Tightening both has to
        /// be measured against math, where deep genuinely wins and quitting
        /// early would cost the ladder's whole gain.
        const ladder_give_up_default: f64 = 1.10;
        fn ladderGiveUpRatio() f64 {
            const v = std.c.getenv("DMLX_MTP_LADDER_GIVE_UP") orelse
                return ladder_give_up_default;
            return std.fmt.parseFloat(f64, std.mem.span(v)) catch
                ladder_give_up_default;
        }
        const explore_give_up_default: f64 = 1.15;
        fn exploreGiveUpRatio() f64 {
            const v = std.c.getenv("DMLX_MTP_EXPLORE_GIVE_UP") orelse
                return explore_give_up_default;
            return std.fmt.parseFloat(f64, std.mem.span(v)) catch
                explore_give_up_default;
        }
        /// Tight ratios, applied only where the trend says acceptance is FLAT.
        /// These are the values the sweep found good for code and bad for
        /// math; gating them on the trend is what lets code have them without
        /// math paying for them.
        const explore_give_up_flat_default: f64 = 1.08;
        const ladder_give_up_flat_default: f64 = 1.04;
        fn exploreGiveUpFlat() f64 {
            const v = std.c.getenv("DMLX_MTP_EXPLORE_GIVE_UP_FLAT") orelse
                return explore_give_up_flat_default;
            return std.fmt.parseFloat(f64, std.mem.span(v)) catch
                explore_give_up_flat_default;
        }
        fn ladderGiveUpFlat() f64 {
            const v = std.c.getenv("DMLX_MTP_LADDER_GIVE_UP_FLAT") orelse
                return ladder_give_up_flat_default;
            return std.fmt.parseFloat(f64, std.mem.span(v)) catch
                ladder_give_up_flat_default;
        }
        /// OFF by default: the trend cannot be observed in time to use it.
        ///
        /// The idea was to let code have the tight give-up ratios while math
        /// keeps the loose ones, selecting by whether acceptance is rising.
        /// It takes math's whole loss and only part of code's gain, i.e. it is
        /// worse than either pure setting (8 draws/arm, base = loose):
        ///              code                    math
        ///   loose      26.2                    32.7
        ///   trend      26.7  +2.0% 40/64       30.4  -6.8% 20/64
        ///   tight      27.4  +4.9% 36/64       30.5  -6.8%  7/64
        ///
        /// The explore give-up fires at rounds 3-9, while math's acceptance
        /// rise begins around round 15. At the moment of the decision math
        /// looks flat too, so the gate hands it the tight ratio and throws
        /// away the ladder probe's gain. This is not a threshold to tune: the
        /// evidence that would justify persevering does not exist yet when the
        /// choice has to be made.
        ///
        /// Which bounds what this controller can do. Distinguishing
        /// "persevere" from "quit" needs information from BEYOND the current
        /// request's first ten rounds — prior knowledge about this kind of
        /// content. Within-request adaptation cannot supply it, so the
        /// remaining code/count gaps are not reachable from here; a
        /// cross-request memory keyed on content class would be, and omlx's
        /// persistent per-model settings are worth reading with that in mind.
        /// Set DMLX_MTP_TREND_GATE=1 to re-enable.
        fn trendGatingOff() bool {
            return std.c.getenv("DMLX_MTP_TREND_GATE") == null;
        }
        /// Slower evidence window for the trend comparison. 0.98 against the
        /// fast window's 0.9 is a ~50-round memory against ~10, so the pair
        /// spans math's documented ~15-round phase change.
        const trend_slow_decay: f32 = 0.98;
        /// Minimum rise in conditional acceptance to count as a trend rather
        /// than window noise.
        const trend_epsilon: f64 = 0.02;
        /// OFF by default: gating the probe on rung staleness measured away
        /// the entire math win (artifacts/ladder_stalegate_20261103.json,
        /// 3 interleaved passes, nonced):
        ///   gated 31.0   ungated 34.1   no probe at all 31.5
        /// `reached[rung]` is left above the floor by the exploration rounds
        /// and decays too slowly to drop under it inside a request, so the
        /// gate suppressed nearly every probe and the arm collapsed onto the
        /// no-probe behaviour. The same run also showed the other three
        /// prompts unaffected by the ungated probe (code 33.7 vs 33.3,
        /// prose 25.6 vs 25.8, count 28.3 vs 28.2), so the 2-4% dip seen in
        /// the previous run was noise and there is nothing here to protect
        /// against. Kept as an opt-in because the floor is the only part
        /// that was wrong, not the idea.
        fn ladderStaleOnly() bool {
            return std.c.getenv("DMLX_MTP_LADDER_STALE_ONLY") != null;
        }
        /// How many rungs above the incumbent to probe. 2 was the first value
        /// tried and the one the fixed-depth references pointed at (math wants
        /// 4 while the controller parks at 2); it has not been swept.
        const ladder_rung_default: usize = 2;
        /// Challenger must beat the incumbent by this factor to take over.
        /// Tunable because it was tuned BEFORE the ladder probe existed: the
        /// 10% margin was protecting against noise in evidence that nothing
        /// was refreshing, and now the rung above the incumbent is sampled
        /// every 6 rounds, so the same margin may just be blocking a real
        /// climb. The fixed-depth references say depth 4 is ~17% better than
        /// the controller's pick on math, but an 8%-better rung would be
        /// vetoed outright.
        const hysteresis_default: f64 = 0.90;
        fn hysteresis() f64 {
            const v = std.c.getenv("DMLX_MTP_HYSTERESIS") orelse
                return hysteresis_default;
            return std.fmt.parseFloat(f64, std.mem.span(v)) catch
                hysteresis_default;
        }
        fn ladderRung() usize {
            const v = std.c.getenv("DMLX_MTP_LADDER_RUNG") orelse
                return ladder_rung_default;
            return std.fmt.parseInt(usize, std.mem.span(v), 10) catch
                ladder_rung_default;
        }
        fn ladderProbePeriod() u64 {
            if (std.c.getenv("DMLX_MTP_NO_LADDER_PROBE") != null) return 0;
            const v = std.c.getenv("DMLX_MTP_LADDER_PROBE_PERIOD") orelse
                return ladder_probe_period_default;
            return std.fmt.parseInt(u64, std.mem.span(v), 10) catch
                ladder_probe_period_default;
        }
        fn deepProbePeriod() u64 {
            if (std.c.getenv("DMLX_MTP_NO_DEEP_PROBE") != null) return 0;
            const v = std.c.getenv("DMLX_MTP_DEEP_PROBE_PERIOD") orelse
                return deep_probe_period_default;
            return std.fmt.parseInt(u64, std.mem.span(v), 10) catch
                deep_probe_period_default;
        }
        /// Rounds spent at `max_n` at the start of every request before the
        /// argmin is trusted. DEFAULT 0 — measured net-negative, see below.
        ///
        /// The hypothesis was that the controller was stuck shallow for lack
        /// of deep evidence (on the code prompt it ran mean depth 2.02 with
        /// p_3/p_4 at one sample each, while omlx ran depth 4 and measured
        /// p_4 = 0.94 there). Forcing 3 deep rounds per request did raise
        /// tokens/round 2.49 -> 2.93 and depth 2.02 -> 3.00, but ms/round
        /// rose 85.8 -> 111.4 and the realized rate got WORSE: 41.3 ->
        /// 43.5 ms/emitted-token.
        ///
        /// So the controller's shallow choice was CORRECT: our cost(n) curve
        /// is steeper than omlx's, so our optimal depth genuinely is lower.
        /// The gap is cost-per-round at depth, not the depth policy
        /// (plan doc §27). Kept as a tunable because the right depth moves
        /// as soon as cost(n) is flattened — which is exactly what the
        /// coarse draft head in §27.3 would do.
        /// DEFAULT 3 as of the §38 routing fixes. It was 0 because forcing
        /// deep rounds measured net-negative under the OLD cost curve
        /// (§27.3). Those fixes flattened cost(n) — 85.2 / 98.7 / 113.4 ms at
        /// N=2/3/4, versus 88.2 / 108.8 / 120.3 before — and that exposed the
        /// opposite failure: with no forced exploration the controller
        /// COLLAPSED to mean depth 1.13 and 47.3 ms/token, because a
        /// depth-1 round never increments `reached[1]`, so p_2 stays at its
        /// Laplace prior, E[tokens(2)] stays pessimistic, and the argmin
        /// keeps choosing depth 1 — self-reinforcing. Three forced deep
        /// rounds per request break the loop: mean depth 2.93 and
        /// 38.3 ms/token, a 19% improvement end to end.
        /// DEFAULT 9 as of 2026-11-02 (was 3). The trunk-postnorm flip raised
        /// deep-tail acceptance (code p4 0.267 -> 0.677), so deep parking
        /// pays whenever the content supports it — but only once the tail
        /// p_i estimates are sampled enough for the argmin to commit.
        /// Sweep at 128-token requests, mean-of-4 tok/s (code/prose/math):
        ///   0: 35.6/23.4/31.2   3: 35.1/23.3/27.4   6: 41.7/22.4/33.0
        ///   9: 43.8/24.1/33.2   12: 43.2/22.6/31.2
        const explore_rounds_default = 9;
        fn exploreRounds() u32 {
            const v = std.c.getenv("DMLX_MTP_EXPLORE_ROUNDS") orelse return explore_rounds_default;
            return std.fmt.parseInt(u32, std.mem.span(v), 10) catch explore_rounds_default;
        }
        /// Cost is near-deterministic, so an EMA is safe and is preferable to
        /// a running mean: verify cost grows with context length, and a plain
        /// mean would leave unchosen depths frozen at their cheap, early
        /// calibration value — biasing the controller toward whatever it is
        /// NOT currently running.
        const cost_alpha = 0.25;
        /// Rounds kept per depth for the median cost estimate.
        ///
        /// omlx stores `deque(maxlen=8)` per depth and scores on
        /// `median(costs)` (omlx/patches/mlx_lm_mtp/batch_policy.py `score`),
        /// where we had only an EMA at alpha 0.25. The difference is
        /// robustness, and it matches the symptom exactly: one slow round
        /// moves an alpha-0.25 EMA by 25%, which is more than enough to flip
        /// the argmin off the incumbent, after which a stale estimate takes
        /// over for the rest of the request — two fixed points, i.e. the
        /// bimodal readings measured on code (per-request samples
        /// [33.2, 37.9, 33.7, 38.7, 33.0, 38.1] against omlx's tight
        /// 34.6-36.2). A median of 8 cannot be moved by a single outlier, so
        /// omlx lands in the same place every request.
        ///
        /// Round cost genuinely has outliers here — page faults, thermal
        /// steps, and the occasional long verify — and they carry no
        /// information about which depth is better, which is the case for a
        /// median rather than a mean over the same window.
        const cost_window = 8;
        fn costMedian() bool {
            return std.c.getenv("DMLX_MTP_COST_EMA") == null;
        }

        max_n: usize,
        fixed: bool,
        /// Pooled conditional acceptance: `reached[i]` rounds got as far as
        /// draft i+1, `accepted[i]` of them accepted it. Both are DECAYED
        /// every round (evidence window, not lifetime counts): acceptance is
        /// content-PHASE-dependent within a single request — on math the
        /// first ~15 rounds after prefill average k=2.6/7 while rounds 16+
        /// run k=7/7 (measured 2026-11-03). Lifetime counts poison the deep
        /// tail with the early phase and never recover, locking the request
        /// shallow; decayed evidence tracks the current phase in ~4 rounds.
        reached: [max_depth]f32 = @splat(0),
        accepted: [max_depth]f32 = @splat(0),
        /// A second, slower-decaying copy of the same evidence, used ONLY to
        /// ask whether acceptance is rising. See `trendRising`.
        reached_slow: [max_depth]f32 = @splat(0),
        accepted_slow: [max_depth]f32 = @splat(0),
        cost_ms: [max_depth]f64 = @splat(0),
        cost_n: [max_depth]u32 = @splat(0),
        /// Last `cost_window` round times per depth, for a MEDIAN cost.
        cost_ring: [max_depth][cost_window]f64 = @splat(@splat(0)),
        used: [max_depth]u32 = @splat(0),
        rounds: u64 = 0,
        /// Last argmin-chosen depth. Switching requires beating the incumbent
        /// by >10% (hysteresis): with decayed evidence the per-depth scores
        /// carry window noise, and on flat-acceptance content (prose) the
        /// argmin otherwise thrashes into wasted deep rounds.
        last_pick: usize = 0,
        /// Forced deep rounds after a near-full-acceptance probe: a strong
        /// deep round means the content phase changed in favor of depth, so
        /// re-explore immediately instead of waiting for the next 1-in-16
        /// probe slot.
        force_explore: u32 = 0,
        class_mem: [content_class.n_classes]ClassMemory = @splat(.{}),
        active_class: content_class.ContentClass = .general,
        /// Per-round acceptance ratios (k/n) of THIS request, for the
        /// first-half/second-half trend written back at `finishRequest`.
        round_acc: [64]f32 = @splat(0),
        round_acc_len: usize = 0,

        /// Cross-request content memory. Acceptance behaviour is a property
        /// of the content CLASS, not the single request: code stays flat and
        /// shallow-friendly, math starts bad and turns deep-friendly ~15
        /// rounds in. The per-request evidence cannot know which regime it
        /// is in until the explore decision is already behind it (see the
        /// trend-gate postmortem at `trendGatingOff`), so a prompt
        /// classifier (content_class.zig) keys a per-class memory that
        /// (a) seeds p_i at request start, replacing the content-blind
        /// Laplace prior, and (b) supplies the rising/flat verdict the
        /// give-up ratios need at round 3, not round 15. A wrong class only
        /// costs a weak prior: per-request evidence decays the seed away at
        /// 0.9/round and `.general` requests neither read nor write memory.
        /// Opt-in via DMLX_MTP_CLASS_MEMORY=1 until the A/B lands.
        const ClassMemory = struct {
            /// Requests folded in so far.
            samples: u32 = 0,
            /// EMA of the conditional acceptance p_i (1-based position i).
            p: [max_depth]f32 = @splat(0.5),
            /// EMA of (second-half acceptance) - (first-half acceptance)
            /// within a request: positive means the class turns
            /// deep-friendly as it goes (math), ~0 means flat (code/prose).
            rising: f32 = 0,
        };
        /// Seed strength in rounds-equivalent: weak enough to be overridden
        /// by ~10 rounds of live evidence, strong enough to inform the
        /// explore early-exit at rounds 3-9.
        const class_seed_weight: f32 = 3.0;
        /// Minimum folded requests before the class verdict drives ratios.
        const class_confident_default: u32 = 2;
        fn classConfidentSamples() u32 {
            const v = std.c.getenv("DMLX_MTP_CLASS_CONFIDENT") orelse
                return class_confident_default;
            return std.fmt.parseInt(u32, std.mem.span(v), 10) catch
                class_confident_default;
        }
        /// |rising| above which a class counts as rising (deep-friendly late).
        const class_rising_epsilon: f32 = 0.05;
        /// EMA weight of the newest request in the memory update.
        const class_mem_alpha: f32 = 0.5;
        /// Requests shorter than this carry no reliable class signal.
        const class_min_rounds: u64 = 6;

        fn classMemoryOn() bool {
            return std.c.getenv("DMLX_MTP_CLASS_MEMORY") != null;
        }

        /// The class memory's verdict on whether acceptance RISES over a
        /// request, or null when there is no confident opinion (feature off,
        /// general class, or too few samples). Callers fall back to the
        /// in-request trend (or the loose default) on null.
        fn classRising(self: *const MtpDepthController) ?bool {
            if (!classMemoryOn()) return null;
            const mem = &self.class_mem[@intFromEnum(self.active_class)];
            if (mem.samples < classConfidentSamples()) return null;
            return mem.rising > class_rising_epsilon;
        }
        /// Per-round evidence decay (see reached/accepted). 0.9 swept best
        /// of {0.75, 0.85, 0.9} on the 4-domain thinking A/B (2026-11-03):
        /// slower decay keeps math/count phase recovery while damping the
        /// prose/code thrash.
        const evidence_decay_default: f32 = 0.9;
        /// Decay only the positions a round observed (omlx's behaviour).
        fn decayReachedOnly() bool {
            return std.c.getenv("DMLX_MTP_DECAY_REACHED_ONLY") != null;
        }
        fn evidenceDecay() f32 {
            const s = std.c.getenv("DMLX_MTP_EVIDENCE_DECAY") orelse return evidence_decay_default;
            return std.fmt.parseFloat(f32, std.mem.span(s)) catch evidence_decay_default;
        }
        /// Revert switch for the round-1 cost exclusion below. Exists so the
        /// change is A/B-able: a cost-model change that cannot be toggled
        /// cannot be measured against its own baseline in an alternating run.
        fn countRound1Cost() bool {
            return std.c.getenv("DMLX_MTP_COUNT_ROUND1_COST") != null;
        }

        fn init(max_n: usize) MtpDepthController {
            return .{
                .max_n = @max(1, @min(max_n, max_depth)),
                .fixed = std.c.getenv("DMLX_MTP_FIXED_N") != null,
            };
        }

        /// Laplace-smoothed conditional acceptance for draft `i` (1-based),
        /// so an unobserved depth is optimistic-but-bounded rather than 0/0.
        fn condAccept(self: *const MtpDepthController, i: usize) f64 {
            const a: f64 = @floatCast(self.accepted[i - 1]);
            const r: f64 = @floatCast(self.reached[i - 1]);
            return (a + 1.0) / (r + 2.0);
        }

        /// Expected tokens emitted by a round at depth `n`: the first token
        /// is always emitted, then each draft prefix survives with the
        /// product of the conditional acceptances up to it.
        fn expectedTokens(self: *const MtpDepthController, n: usize) f64 {
            var total: f64 = 1.0;
            var prefix: f64 = 1.0;
            for (1..n + 1) |i| {
                prefix *= self.condAccept(i);
                total += prefix;
            }
            return total;
        }

        fn meanCostMs(self: *const MtpDepthController, n: usize) ?f64 {
            if (self.cost_n[n - 1] == 0) return null;
            if (!costMedian()) return self.cost_ms[n - 1];
            const have = @min(self.cost_n[n - 1], cost_window);
            var buf: [cost_window]f64 = undefined;
            for (0..have) |i| buf[i] = self.cost_ring[n - 1][i];
            const s = buf[0..have];
            std.mem.sort(f64, s, {}, std.sort.asc(f64));
            return if (s.len % 2 == 1)
                s[s.len / 2]
            else
                (s[s.len / 2 - 1] + s[s.len / 2]) / 2.0;
        }

        /// Estimated ms per emitted token at depth `n`, or null if we have
        /// never timed a round at that depth.
        fn score(self: *const MtpDepthController, n: usize) ?f64 {
            const cost = self.meanCostMs(n) orelse return null;
            return cost / self.expectedTokens(n);
        }

        /// Clear the per-request acceptance evidence but KEEP the cost
        /// estimates: `cost(n)` is a property of the hardware and the model,
        /// not of the content, so carrying it across requests skips
        /// calibration entirely on every request after the first. That
        /// matters because short requests only run ~12 rounds, and spending
        /// 4 of them calibrating was eating a third of the request.
        /// Acceptance IS content-dependent, so `p_i` starts fresh (it
        /// re-converges within a few rounds — every round updates p_1..p_k+1).
        fn resetForRequest(self: *MtpDepthController, class: content_class.ContentClass) void {
            self.reached = @splat(0);
            self.accepted = @splat(0);
            self.reached_slow = @splat(0);
            self.accepted_slow = @splat(0);
            self.used = @splat(0);
            self.rounds = 0;
            self.force_explore = 0;
            self.last_pick = 0;
            self.round_acc_len = 0;
            self.active_class = if (classMemoryOn()) class else .general;
            // Seed p_i from the class memory: the Laplace prior treats every
            // prompt as identical, so the explore early-exit judges a code
            // prompt's deep arm on three unlucky rounds and the math prompt's
            // deep tail starts from 0.5 optimism. The class prior is what
            // previous requests of this kind actually measured; the 0.9/round
            // decay washes it out within ~10 rounds if it is wrong.
            if (self.active_class != .general) {
                const mem = &self.class_mem[@intFromEnum(self.active_class)];
                if (mem.samples > 0) {
                    for (0..max_depth) |i| {
                        self.reached[i] = class_seed_weight;
                        self.accepted[i] = class_seed_weight * mem.p[i];
                        self.reached_slow[i] = class_seed_weight;
                        self.accepted_slow[i] = class_seed_weight * mem.p[i];
                    }
                }
                // Confident class: seat the incumbent at the class-optimal
                // depth directly. Without this a flat class skips explore and
                // then has to climb from depth 1 through the 10% hysteresis —
                // count measured the failure: argmin wanted 7 (26.8 vs 29.9
                // ms/tok) but the hysteresis threshold landed at 26.91 and
                // the request parked at 3 for 24 of 30 rounds (-7.7%). The
                // seeded score uses carried cost(n) + class p_i, i.e. exactly
                // the information the explore rounds would have re-measured.
                if (mem.samples >= classConfidentSamples()) {
                    var best: usize = 0;
                    var best_score: f64 = std.math.floatMax(f64);
                    for (1..self.max_n + 1) |n| {
                        const s = self.score(n) orelse continue;
                        if (s < best_score) {
                            best_score = s;
                            best = n;
                        }
                    }
                    self.last_pick = best;
                }
            }
        }

        /// Fold this request's evidence into its class memory. Called once
        /// at the end of a completed MTP request (not on error paths).
        fn finishRequest(self: *MtpDepthController) void {
            if (!classMemoryOn()) return;
            if (self.active_class == .general) return;
            if (self.rounds < class_min_rounds) return;
            const mem = &self.class_mem[@intFromEnum(self.active_class)];
            const alpha: f32 = if (mem.samples == 0) 1.0 else class_mem_alpha;
            // p_i: only positions with real evidence this request (a decayed
            // `reached` below 1 was barely observed and is prior-dominated).
            for (0..self.max_n) |i| {
                const r = self.reached[i];
                if (r < 2.0) continue;
                const p_req = self.accepted[i] / r;
                mem.p[i] = alpha * p_req + (1.0 - alpha) * mem.p[i];
            }
            // Trend: mean acceptance ratio of the second half of the rounds
            // minus the first half. Positive = the content turned
            // deep-friendly as it went (math); ~0 = flat (code/prose).
            if (self.round_acc_len >= 4) {
                const half = self.round_acc_len / 2;
                var first: f32 = 0;
                var second: f32 = 0;
                for (self.round_acc[0..half]) |v| first += v;
                for (self.round_acc[half..self.round_acc_len]) |v| second += v;
                first /= @floatFromInt(half);
                second /= @floatFromInt(self.round_acc_len - half);
                mem.rising = alpha * (second - first) + (1.0 - alpha) * mem.rising;
            }
            mem.samples += 1;
        }

        /// Is acceptance at position `i` RISING rather than merely low?
        ///
        /// This is the discriminator the give-up ratios could not express.
        /// Sweeping those ratios showed code and math wanting opposite values
        /// (artifacts/giveup_code_20261103.json, giveup_math_20261103.json):
        /// tightening to 1.08/1.04 moved code +4.4% at 40/64 pairwise, inside
        /// chance, while costing math -4.7% at 3/64 and -8.2% at 1/64, which
        /// is not. A threshold on the LEVEL of a score cannot separate them,
        /// because at the moment of the decision the two look the same —
        ///   math: acceptance rises after ~15 rounds, so early deep evidence
        ///         is pessimistic and persevering is what makes the ladder
        ///         probe pay
        ///   code: acceptance is stable, so early evidence is already right
        ///         and persevering just burns rounds at a worse depth
        /// What differs is the DERIVATIVE, and it needs no new measurement:
        /// comparing the fast evidence window against a slower one over the
        /// same observations reads the trend directly.
        fn trendRising(self: *const MtpDepthController, i: usize) bool {
            const r_fast: f64 = @floatCast(self.reached[i - 1]);
            const r_slow: f64 = @floatCast(self.reached_slow[i - 1]);
            // Both windows need real evidence or the comparison is prior-vs-prior.
            if (r_fast < 1.0 or r_slow < 2.0) return true;
            const fast = (@as(f64, @floatCast(self.accepted[i - 1])) + 1.0) / (r_fast + 2.0);
            const slow = (@as(f64, @floatCast(self.accepted_slow[i - 1])) + 1.0) / (r_slow + 2.0);
            return fast > slow + trend_epsilon;
        }

        /// Probe the rung unless we CONFIDENTLY know it is worse.
        ///
        /// The probe is not free — it spends one round in six at a deeper,
        /// dearer depth — so it has to be an investment, not a reflex. On
        /// math it pays for itself many times over, but on prose it is a pure
        /// tax, and the fixed-depth references show how large:
        /// (artifacts/shallow_depth_20261103.json, same session, ~1% harness
        /// precision from the null A/B in artifacts/null_ab_20261103.json)
        ///   prose   fix2 28.8   fix3 27.4   adaptive 25.9   omlx 28.6
        ///   math    fix3 37.5                adaptive 37.8   omlx 34.4
        /// Adaptive on prose is below BOTH neighbouring fixed depths, which
        /// means the loss is not a mis-picked depth — it is the cost of
        /// looking. Fixed depth 2 would beat omlx on prose outright.
        ///
        /// The distinction that makes this safe is between "bad because
        /// unobserved" and "bad because observed and genuinely poor".
        /// `reached[i]` separates them: a decayed, unsampled rung has a low
        /// `reached` and a score dragged down by the Laplace prior, which is
        /// exactly the math case the probe exists to rescue, so uncertainty
        /// must keep probing. A rung with real recent evidence and a clearly
        /// worse score is the prose case, and there is nothing to learn by
        /// paying for it again. Gating on the score ALONE would re-create the
        /// math trap, because decayed evidence also scores badly.
        fn ladderWorthProbing(self: *const MtpDepthController, rung: usize) bool {
            if (std.c.getenv("DMLX_MTP_LADDER_UNGATED") != null) return true;
            if (self.reached[rung - 1] < ladder_confident_floor) return true;
            const rung_score = self.score(rung) orelse return true;
            const inc_score = self.score(self.last_pick) orelse return true;
            // A confident class verdict beats the in-request trend: it is
            // available at round 3 instead of round 15 (see classRising).
            const rising = self.classRising() orelse
                (trendGatingOff() or self.trendRising(rung));
            const ratio = if (rising) ladderGiveUpRatio() else ladderGiveUpFlat();
            return rung_score <= inc_score * ratio;
        }

        /// Depth for the next round, in 1..max_n.
        fn pick(self: *MtpDepthController) usize {
            if (self.fixed or self.max_n == 1) return self.max_n;
            // Acceptance exploration, BEFORE the cost calibration below.
            //
            // One round at depth n yields evidence for p_1..p_{k+1} in a
            // single shot, while a shallow round says NOTHING about the deep
            // tail. So the cheap way to learn the whole p_i curve is to run
            // deep a few times; running shallow and waiting is not an option.
            //
            // Tying exploration to `cost_n == 0` (below) is not enough:
            // `resetForRequest` deliberately keeps cost estimates but clears
            // p_i, so from the second request onward the calibration loop
            // never fires and the only deep evidence is the 1-in-32 probe —
            // one sample, which Laplace smoothing keeps near the prior. The
            // controller then locks shallow forever.
            //
            // Measured on the code prompt before this (plan doc §27): mean
            // depth 2.02 with p_3 and p_4 at a single sample each, while
            // omlx — which starts at max depth and sweeps down — ran depth 4
            // and measured p_3 = 0.83, p_4 = 0.94 on the same prompt. Our
            // ms/round was BETTER (85.8 vs 92.3); the entire gap was
            // tokens/round (2.49 vs 3.20).
            // Explore at max_n, but with an EARLY EXIT: once the deep arm
            // has >= 3 samples, its score is real (pooled p_i populated by
            // the first deep rounds), and if it is clearly losing to the
            // best calibrated alternative the remaining explore rounds are
            // pure cost. Measured 2026-11-02: exploring all 9 rounds at
            // depth 7 taxed math ~8% (p4 collapse there) vs ceiling 4,
            // while exploring shallow (min(max_n,4)) cost code -18%
            // (deep-tail p_i stayed at the Laplace prior, so the argmin
            // never committed to depth 7 where code wins). Adaptive:
            // deep content keeps exploring, math-class content exits early.
            // A near-full deep probe means the content phase turned
            // deep-friendly: re-explore now rather than waiting out the
            // 1-in-16 probe cycle (math: the great phase starts ~15 rounds
            // in; the explore phase at rounds 1-9 sampled the bad one).
            if (self.force_explore > 0) {
                self.force_explore -= 1;
                return self.max_n;
            }
            // Confident-flat classes skip exploration entirely: the class
            // memory already holds this content's p_i curve (seeded at
            // reset), and the explore tax is the whole reason the memory
            // exists — on code it is 9 forced deep rounds plus the probes
            // they trigger, ~15 of ~45 rounds at depths the class has
            // already proven worse. Only a FLAT verdict skips: a rising
            // class (math) still needs the deep sampling, and an unknown
            // class explores as before. Mid-request phase changes are still
            // caught by the near-full-acceptance re-explore latch.
            const class_skip_explore = if (self.classRising()) |rising| !rising else false;
            if (!class_skip_explore and self.used[self.max_n - 1] < exploreRounds()) {
                const deep_used = self.used[self.max_n - 1];
                var keep_exploring = true;
                // The early exit is what drops a request into the slow fixed
                // point: the draws that ended at `N=7:4r` / `N=7:6r` are
                // exactly the ones where it fired, and they ran 26 t/s against
                // 29-30 for the draws that explored through and parked deep.
                // It judges the deep arm on 3 rounds taken during math's
                // documented low-acceptance opening (~15 rounds at k=2.6/7),
                // then never revisits — and per the deep-cadence result above,
                // revisiting cannot work, so this one decision is final.
                // Opt-out so the threshold stays A/B-able; it was tuned on
                // 2026-11-02, before evidence decay existed.
                if (deep_used >= 3 and std.c.getenv("DMLX_MTP_NO_EXPLORE_EARLY_EXIT") == null) {
                    if (self.score(self.max_n)) |deep_score| {
                        var alt: f64 = std.math.floatMax(f64);
                        var found = false;
                        for (1..self.max_n) |n| {
                            if (self.score(n)) |s| {
                                found = true;
                                if (s < alt) alt = s;
                            }
                        }
                        const rising = self.classRising() orelse
                            (trendGatingOff() or self.trendRising(self.max_n));
                        const ratio = if (rising) exploreGiveUpRatio() else exploreGiveUpFlat();
                        if (found and deep_score > alt * ratio) keep_exploring = false;
                    }
                }
                if (keep_exploring) return self.max_n;
            }
            // Calibrate any depth never yet timed, deepest first so the
            // pooled p_i table is populated from the very first round. Warm
            // cost estimates (carried across requests) skip this entirely.
            var uncal = self.max_n;
            while (uncal >= 1) : (uncal -= 1) {
                if (self.cost_n[uncal - 1] == 0) return uncal;
            }
            // Periodic probe, cycling from the deepest downward: the chosen
            // depth stops generating evidence for p_i beyond it, and every
            // unchosen depth's cost estimate would otherwise go stale.
            // Deep probe on its own short cadence (see deep_probe_period).
            // Checked BEFORE the rotation so it is never crowded out.
            const deep_period = deepProbePeriod();
            if (deep_period != 0 and self.rounds % deep_period == 0 and
                self.last_pick != self.max_n)
            {
                return self.max_n;
            }
            // Ladder probe: refresh the rungs immediately above the incumbent.
            //
            // The argmin cannot climb because `reached[i]` for i above the
            // incumbent decays every round (evidence decay 0.9) while a
            // shallow round observes nothing past p_2. Parked at N=2, depth
            // 4's E[tokens] is computed from a p_3/p_4 tail that has decayed
            // back toward the Laplace prior, so depth 4 looks worse than it
            // is and never gets chosen — self-reinforcing, one rung up from
            // the old depth-1 collapse.
            //
            // Why +2 and not max_n: measured (artifacts/
            // fixed_depth_20261103.json, 3 interleaved passes, nonced) the
            // best FIXED depth on math is 4, not 7 —
            //   math   adaptive 29.8   fix7 32.3   fix4 34.8
            // so the controller is leaving 17% on the table there, and the
            // target it needs to reach is two rungs up, not five. Probing
            // max_n instead measured WORSE (see deep_probe_period_default):
            // it is a dearer round and it refreshes p_5..p_7, which are not
            // what gates the step from 2 to 4.
            //
            // Adaptive is still the right default overall — it wins code
            // (33.8 vs 29.1), prose (23.8 vs 21.4) and count (28.8 vs 27.6)
            // against fix4 — so the goal is to keep that and recover math,
            // not to pin the depth.
            // Only fire when the rung's evidence is actually STALE. The probe
            // exists to refresh decayed evidence, so paying for it where the
            // rung is already being observed every round is pure overhead.
            // Content that parks deep (code, count) reaches the rung as a
            // matter of course; content that parks shallow (math) does not.
            //
            // Under decay 0.9 a rung sampled every round settles near
            // 1/(1-0.9) = 10, and an unsampled one falls below 1.0 in about
            // 22 rounds, so 1.0 reads as "effectively unobserved". Measured
            // cost of firing unconditionally (artifacts/
            // ladder_probe_20261103.json): math +8.9% but code -2.2%,
            // prose -2.3%, count -3.7%.
            const ladder_period = ladderProbePeriod();
            if (ladder_period != 0 and self.last_pick != 0 and
                self.rounds % ladder_period == 0)
            {
                const rung = @min(self.last_pick + ladderRung(), self.max_n);
                if (rung != self.last_pick and
                    (!ladderStaleOnly() or self.reached[rung - 1] < ladder_stale_floor) and
                    self.ladderWorthProbing(rung))
                {
                    return rung;
                }
            }
            if (self.rounds % probe_period == 0) {
                // Skip slot 0: that is `max_n`, which now has its own cadence,
                // and spending the rotation slot on it would leave the middle
                // depths staler than before.
                const slot = 1 + (self.rounds / probe_period) % (self.max_n - 1);
                return self.max_n - slot;
            }
            var best: usize = 1;
            var best_score: f64 = std.math.floatMax(f64);
            for (1..self.max_n + 1) |n| {
                const s = self.score(n) orelse continue;
                if (s < best_score) {
                    best_score = s;
                    best = n;
                }
            }
            // Hysteresis: keep the incumbent unless the challenger wins by
            // >10%. A phase change blows past 10% (math: 30 -> 20 ms/tok);
            // what this filters out is noise-level thrash.
            if (self.last_pick != 0 and best != self.last_pick) {
                if (self.score(self.last_pick)) |inc| {
                    if (best_score > inc * hysteresis()) best = self.last_pick;
                }
            }
            self.last_pick = best;
            return best;
        }

        /// Fold one completed round into the estimator. `k` is the number of
        /// accepted drafts, so drafts 1..k were accepted and (when k < n)
        /// draft k+1 was reached and rejected.
        fn record(self: *MtpDepthController, n: usize, round_ns: u64, k: usize) void {
            self.rounds +|= 1;
            if (n == 0 or n > max_depth) return;
            self.used[n - 1] +|= 1;
            // Per-round acceptance ratio, for the class-memory trend written
            // back at finishRequest. Ring is capped; a request longer than
            // 64 rounds simply measures its trend on the first 64.
            if (self.round_acc_len < self.round_acc.len) {
                self.round_acc[self.round_acc_len] =
                    @as(f32, @floatFromInt(k)) / @as(f32, @floatFromInt(n));
                self.round_acc_len += 1;
            }
            // The FIRST round of a request carries one-time per-request work
            // (drafter repriming over the prompt is lazy-evaluated into the
            // first draft phase: measured draft=31.9 ms on request 1 vs
            // ~9-13 ms steady). Explore always runs round 1 at max_n, so
            // cost(max_n) alone would absorb every request's priming —
            // systematically biasing the argmin against depth. Skip the cost
            // sample; still count the round for acceptance evidence.
            if (self.rounds > 1 or countRound1Cost()) {
                const ms = @as(f64, @floatFromInt(round_ns)) / 1e6;
                self.cost_ms[n - 1] = if (self.cost_n[n - 1] == 0)
                    ms
                else
                    cost_alpha * ms + (1.0 - cost_alpha) * self.cost_ms[n - 1];
                // Ring slot, oldest first: index by the sample count so the
                // window holds the last `cost_window` rounds at this depth.
                self.cost_ring[n - 1][self.cost_n[n - 1] % cost_window] = ms;
                self.cost_n[n - 1] +|= 1;
            }
            // Decay stale acceptance evidence toward the Laplace prior, then
            // fold this round in. ~4-round effective window at 0.75.
            const decay = evidenceDecay();
            const reached_upto = @min(k + 1, n);
            // Decay ONLY the positions this round actually observed.
            //
            // Decaying all of them (the original behaviour, still available
            // via DMLX_MTP_DECAY_ALL=1) is self-defeating: a shallow round
            // observes nothing past p_2, so parking at N=2 pulls p_3..p_7
            // back to the Laplace prior at 0.9 per round and the deep tail is
            // gone in ~20 rounds. The argmin then cannot see that depth is
            // worth taking, which is the depth trap the ladder probe (4f18d93)
            // was built to work around by force-feeding fresh samples.
            //
            // omlx does not have the trap because it does not have the decay.
            // `BatchPolicy.observe_mtp` (omlx/patches/mlx_lm_mtp/
            // batch_policy.py) runs `for position in range(depth)` and breaks
            // the moment a position was not reached, so an unobserved p_i
            // simply keeps its last value — a per-position EMA, not a global
            // pull toward a prior.
            //
            // Decay still earns its keep on the positions that ARE observed,
            // which is why this is a restriction rather than a removal:
            // acceptance is phase-dependent inside one request (on math the
            // first ~15 rounds average k=2.6/7 while rounds 16+ run k=7/7), so
            // lifetime counts would pin the estimate to the opening phase.
            // Decaying the observed prefix tracks the phase; leaving the
            // unobserved tail alone preserves what we last knew about it.
            // DEFAULT is still the global decay: restricting it measured a
            // small LOSS next to the ladder probe, which already keeps the
            // tail fresh, so the two are substitutes and the probe is the
            // better of the pair. Same session, 3 interleaved passes
            // (artifacts/decay_tail_20261103.json): global 33.6 vs
            // reached-only 33.3, and global was ahead on all four prompts.
            // A second session (artifacts/tail_vs_probe_20261103.json) put
            // global+no-probe at 33.2 against reached-only+probe at 33.1 and
            // reached-only+no-probe at 31.6 — so whichever single mechanism
            // keeps the tail alive is what matters, and having both is not
            // better than having the probe. Set DMLX_MTP_DECAY_REACHED_ONLY=1
            // for omlx's behaviour.
            const decay_upto: usize = if (decayReachedOnly()) reached_upto else max_depth;
            for (0..decay_upto) |i| {
                self.reached[i] *= decay;
                self.accepted[i] *= decay;
            }
            for (0..max_depth) |i| {
                self.reached_slow[i] *= trend_slow_decay;
                self.accepted_slow[i] *= trend_slow_decay;
            }
            for (1..reached_upto + 1) |i| self.reached_slow[i - 1] += 1;
            for (1..k + 1) |i| self.accepted_slow[i - 1] += 1;
            for (1..reached_upto + 1) |i| self.reached[i - 1] += 1;
            for (1..k + 1) |i| self.accepted[i - 1] += 1;
            // Near-full acceptance at a deep round (probe or chosen): the
            // content phase favors depth right now — force a few deep rounds
            // so the argmin sees fresh tail evidence immediately.
            // Opt-out: DMLX_MTP_NO_REEXPLORE=1 (A/B showed the re-entry
            // overshooting on code-class mid-acceptance content).
            //
            // Floor is max_n - 2, not max_n - 1 (2026-11-06, math mode
            // lottery): the ladder probe runs at incumbent+2, so from an
            // incumbent of 3 it samples depth 5 — and the old max_n-1 floor
            // never let a 5/5 probe fire the latch. The traces are unambiguous
            // (math thinking, 6 draws): slow draws park at N=3 and their
            // rung probes hit 5/5 THREE separate times (e.g. draw 2 rounds
            // 16/18/20) with no effect; fast draws differ only in having
            // parked at 7 early. A full-acceptance rung probe is exactly the
            // phase-change signal the latch was built for — it just could not
            // reach it. Restore the old floor with DMLX_MTP_REEXPLORE_FLOOR=6.
            const reexplore_floor: usize = blk: {
                const s = std.c.getenv("DMLX_MTP_REEXPLORE_FLOOR") orelse break :blk self.max_n -| 2;
                break :blk std.fmt.parseInt(usize, std.mem.span(s), 10) catch self.max_n -| 2;
            };
            if (std.c.getenv("DMLX_MTP_NO_REEXPLORE") == null and
                k + 1 >= n and n >= reexplore_floor and self.force_explore == 0)
            {
                self.force_explore = 3;
            }
            // Score latch: a deep probe that MEASURED better than the
            // incumbent earns two more rounds to firm the estimate up.
            //
            // The near-full-acceptance latch above is effectively dead at
            // ceiling 7: it needs k + 1 >= n, i.e. k >= 6, while the deep arm
            // runs E[t] ~= 4.7 (about 3.7 accepted of 7) even in its GOOD
            // phase. So the one amplifier the deep probe had almost never
            // fired, and a lone probe round cannot win on its own — Laplace
            // smoothing keeps a single sample near the prior and the 10%
            // hysteresis then protects the incumbent.
            //
            // Latching on the score instead tests exactly the quantity the
            // argmin uses, so it cannot fire on content where depth really
            // loses, and the hysteresis still gates the actual switch; all
            // this buys is enough samples for that decision to be made on
            // evidence rather than on one diluted round.
            // Part of the deep-cadence package, so it is off with it: on its
            // own it would be inert anyway (with the cadence disabled the
            // rotation skips slot 0, so `n == max_n` only happens during
            // exploration, when `last_pick` is still 0).
            if (deepProbePeriod() != 0 and
                std.c.getenv("DMLX_MTP_NO_SCORE_LATCH") == null and
                n == self.max_n and self.force_explore == 0 and
                self.last_pick != 0 and self.last_pick != self.max_n)
            {
                if (self.score(self.max_n)) |deep| {
                    if (self.score(self.last_pick)) |inc| {
                        if (deep < inc) self.force_explore = 2;
                    }
                }
            }
        }
    };

    /// MTP speculative decoding (Qwen3.8 drafter head). Per round:
    ///   1. Draft chain: N drafter steps from (hidden, t_next), N chosen
    ///      online by MtpDepthController (flag value = upper bound).
    ///   2. One backbone verify over [t_next, d1..dN] (continuation).
    ///   3. Acceptance: greedy longest-prefix at temperature==0, speculative
    ///      sampling (p_target/q_draft with resample from (p-q)+) otherwise.
    ///   4. Commit accepted positions / roll back the rest; emit.
    /// Bitwise-lossless at temperature 0: emitted == plain greedy sequence.
    fn processMtpRequest(
        self: *EngineLoop,
        req: *RequestState,
        caches: []KVCacheStrategy,
        gen_config: GenerateConfig,
        mtp: *const generation_mod.MtpOps,
        n_drafts: usize,
    ) void {
        const allocator = self.allocator;
        // The serial path never initializes req.sampler from the request
        // (only prefillBatch does); SamplerConfig defaults to temperature 1.0.
        // Initialize from gen_config so temperature=0 actually means greedy.
        req.sampler = SamplerConfig.init(gen_config.seed);
        req.sampler.temperature = gen_config.temperature;
        req.sampler.top_k = gen_config.top_k;
        req.sampler.top_p = gen_config.top_p;
        const greedy = req.sampler.temperature <= 0.0;

        var context = std.ArrayList(u32).empty;
        defer context.deinit(allocator);
        context.appendSlice(allocator, req.prompt_tokens) catch {
            req.completion.deliverError(self.io, "MTP: context allocation failed");
            return;
        };

        // Prefill (resets backbone + drafter) → first token.
        const pf_t0: i128 = if (std.c.getenv("DMLX_PREFILL_TRACE") != null) monotonicNs() else 0;
        const pf = if (req.vision) |vis| blk: {
            const pv = mtp.prefillVision orelse {
                req.completion.deliverError(self.io, "MTP: vision prompts not supported by this drafter");
                return;
            };
            break :blk pv(self.model.ptr, req.prompt_tokens, vis) catch |err| {
                std.log.err("EngineLoop: MTP vision prefill failed: {}", .{err});
                req.completion.deliverError(self.io, "MTP: prefill failed");
                return;
            };
        } else mtp.prefill(self.model.ptr, req.prompt_tokens) catch |err| {
            std.log.err("EngineLoop: MTP prefill failed: {}", .{err});
            req.completion.deliverError(self.io, "MTP: prefill failed");
            return;
        };
        defer pf.logits.deinit();
        if (pf_t0 != 0) {
            std.log.info("[PREFILL_TRACE] mtp req={d} prompt={d} prefill_ms={d:.1}", .{ req.id, req.prompt_tokens.len, @as(f64, @floatFromInt(monotonicNs() - pf_t0)) / 1e6 });
        }
        var hidden = pf.hidden; // owned; reassigned per round

        var t_next = req.sampler.sample(pf.logits, allocator) catch {
            pf.hidden.deinit();
            req.completion.deliverError(self.io, "MTP: sampling failed");
            return;
        };
        context.append(allocator, t_next.token) catch return;
        var total_generated: usize = 1;
        var drafts_proposed: usize = 0;
        var drafts_accepted: usize = 0;

        // Phase profiling (printed in the final log line).
        const prof_now = struct {
            fn f() i128 {
                var ts: std.c.timespec = undefined;
                _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
                return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
            }
        }.f;
        var prof_last: i128 = prof_now();
        var prof_draft_ns: u64 = 0;
        var prof_verify_ns: u64 = 0;
        var prof_accept_ns: u64 = 0;
        var prof_commit_ns: u64 = 0;
        var prof_rounds: u64 = 0;
        // DMLX_MEM_PROBE_ROUNDS=1: per-phase MLX active memory (leak hunt).
        var phase_mem: [4]usize = .{ 0, 0, 0, 0 };
        const mem_probe_on = std.c.getenv("DMLX_MEM_PROBE_ROUNDS") != null;
        const prof_lap = struct {
            fn f(now_fn: *const fn () i128, t: *i128, acc: *u64) void {
                const now = now_fn();
                const delta = now - t.*;
                acc.* +|= if (delta > 0) @intCast(delta) else 0;
                t.* = now;
            }
        }.f;

        var finish: TokenFinishReason = .length;
        var stopped = generation_mod.isStopToken(t_next.token, gen_config.stop_tokens);
        if (stopped) finish = .stop;

        // Drafter samples with the same shaping config but an independent prng.
        var draft_sampler = req.sampler;
        draft_sampler.prng = std.Random.DefaultPrng.init(gen_config.seed +% 0x5d1f);

        // Depth controller persists across requests (cost estimates are
        // hardware properties); only the acceptance evidence is per-request.
        // With DMLX_MTP_CLASS_MEMORY the prompt is classified once so the
        // controller can seed p_i and the give-up verdict from the
        // cross-request memory of this content class.
        const prompt_class: content_class.ContentClass = blk: {
            if (std.c.getenv("DMLX_MTP_CLASS_MEMORY") == null) break :blk .general;
            const prompt_text = self.tokenizer.decode(req.prompt_tokens, allocator) catch
                break :blk .general;
            defer allocator.free(prompt_text);
            break :blk content_class.classify(prompt_text);
        };
        if (self.mtp_depth == null or self.mtp_depth.?.max_n != @min(n_drafts, MtpDepthController.max_depth)) {
            self.mtp_depth = MtpDepthController.init(n_drafts);
        }
        self.mtp_depth.?.resetForRequest(prompt_class);
        const depth_ctl = &self.mtp_depth.?;

        // `DMLX_MTP_TOPK=1`: measure how often the target's own token sits in
        // the drafter's top-k at each depth. top1 reproduces p_i; the top1 ->
        // top2/top4 gap is the headroom a multi-branch draft would capture.
        // Needs the drafter's logits, which only the CPU draft path keeps, so
        // it disables the GPU-chained path — this is a DIAGNOSTIC mode and
        // costs throughput.
        const topk_probe = std.c.getenv("DMLX_MTP_TOPK") != null;
        const topk_depths: usize = 8;
        var topk_total = [_]u32{0} ** topk_depths;
        var topk_hit1 = [_]u32{0} ** topk_depths;
        var topk_hit2 = [_]u32{0} ** topk_depths;
        var topk_hit4 = [_]u32{0} ** topk_depths;
        var topk_hit8 = [_]u32{0} ** topk_depths;

        while (!stopped and total_generated < gen_config.max_tokens) {
            // Depth for this round (flag value is the upper bound). Never
            // draft past max_tokens: a deeper chain than the remaining budget
            // is pure waste, since emission is capped anyway.
            const remaining = gen_config.max_tokens - total_generated;
            const n_round = @max(1, @min(depth_ctl.pick(), remaining));
            const round_t0 = prof_now();

            // 1. Draft chain.
            var drafts = std.ArrayList(u32).empty;
            defer drafts.deinit(allocator);
            // DMLX_MTP_AVTRACE: per-step (l_max, margin) features, aligned
            // with `drafts` when the coarse head produced them.
            var draft_feats = std.ArrayList([2]f32).empty;
            defer draft_feats.deinit(allocator);
            var draft_logits = std.ArrayList(Array).empty;
            defer {
                for (draft_logits.items) |l| l.deinit();
                draft_logits.deinit(allocator);
            }
            var h_cur = hidden;
            var tok = t_next.token;
            var draft_failed = false;
            const gpu_chain = greedy and mtp.draftGreedy != null and !topk_probe;
            if (gpu_chain) blk: {
                // Greedy GPU-chained chain: token ids stay on GPU (argmax of
                // each step's logits feeds the next step); ONE sync reads all
                // N draft tokens at the end. Draft logits are dropped —
                // greedy acceptance never needs them.
                var gpu_toks = std.ArrayList(Array).empty;
                defer {
                    for (gpu_toks.items) |t2| t2.deinit();
                    gpu_toks.deinit(allocator);
                }
                var gpu_feats = std.ArrayList(Array).empty;
                defer {
                    for (gpu_feats.items) |f2| f2.deinit();
                    gpu_feats.deinit(allocator);
                }
                const tok0 = array_mod.Array.fromData(allocator, i32, &[_]i32{@intCast(tok)}, &[_]i32{1}) catch {
                    draft_failed = true;
                    break :blk;
                };
                defer tok0.deinit();
                var cur = tok0; // borrowed (tok0, then the last gpu_toks item)
                var steps: usize = 0;
                for (0..n_round) |_| {
                    const g = mtp.draftGreedy.?(self.model.ptr, h_cur, cur) catch {
                        draft_failed = true;
                        break :blk;
                    };
                    if (steps > 0) h_cur.deinit(); // previous chained hidden
                    steps += 1;
                    h_cur = g.hidden;
                    if (g.feat) |f| {
                        gpu_feats.append(allocator, f) catch {
                            f.deinit();
                        };
                    }
                    gpu_toks.append(allocator, g.tok_gpu) catch {
                        g.tok_gpu.deinit();
                        draft_failed = true;
                        break :blk;
                    };
                    cur = g.tok_gpu;
                }
                if (gpu_toks.items.len > 0) {
                    // Single sync for the whole chain.
                    const all = shape_mod.concatenateAxis(self.ctx, gpu_toks.items, 0) catch {
                        draft_failed = true;
                        break :blk;
                    };
                    defer all.deinit();
                    all.eval() catch {
                        draft_failed = true;
                        break :blk;
                    };
                    const ids = all.dataSlice(i32) catch {
                        draft_failed = true;
                        break :blk;
                    };
                    for (ids) |id| drafts.append(allocator, @intCast(id)) catch {
                        draft_failed = true;
                        break :blk;
                    };
                    if (gpu_feats.items.len == gpu_toks.items.len) {
                        if (shape_mod.concatenateAxis(self.ctx, gpu_feats.items, 0)) |fall| {
                            defer fall.deinit();
                            if (fall.eval()) |_| {
                                if (fall.dataSlice(f32)) |fv| {
                                    var fi: usize = 0;
                                    while (fi + 1 < fv.len) : (fi += 2) {
                                        draft_feats.append(allocator, .{ fv[fi], fv[fi + 1] }) catch break;
                                    }
                                } else |_| {}
                            } else |_| {}
                        } else |_| {}
                    }
                }
            } else for (0..n_round) |_| {
                const d_out = mtp.draft(self.model.ptr, h_cur, tok) catch {
                    draft_failed = true;
                    break;
                };
                if (drafts.items.len > 0) h_cur.deinit(); // previous chained hidden
                h_cur = d_out.hidden;
                const d = draft_sampler.sample(d_out.logits, allocator) catch {
                    d_out.logits.deinit();
                    draft_failed = true;
                    break;
                };
                draft_logits.append(allocator, d_out.logits) catch {
                    d_out.logits.deinit();
                    draft_failed = true;
                    break;
                };
                drafts.append(allocator, d.token) catch {
                    draft_failed = true;
                    break;
                };
                tok = d.token;
            }
            // h_cur (the last chained drafter hidden) is kept until after the
            // round so all exits can free it (see the defer below).
            if (draft_failed or drafts.items.len == 0) {
                h_cur.deinit();
                // Draft path failed (OOM-class): finish the request with plain
                // decode steps. Backbone state was untouched by the aborted
                // round (drafts only touch the drafter cache, which the next
                // prefill resets), so plain continuation is exact.
                std.log.warn("EngineLoop: MTP draft failed; finishing with plain decode", .{});
                hidden.deinit();
                while (total_generated < gen_config.max_tokens and !stopped) {
                    const arr = array_mod.Array.fromData(allocator, u32, &[_]u32{t_next.token}, &[_]i32{ 1, 1 }) catch break;
                    const st = generation_mod.generateStep(self.model, arr, caches, &req.sampler, self.ctx) catch break;
                    t_next = st;
                    context.append(allocator, t_next.token) catch break;
                    total_generated += 1;
                    if (generation_mod.isStopToken(t_next.token, gen_config.stop_tokens)) {
                        stopped = true;
                        finish = .stop;
                    }
                }
                break;
            }
            drafts_proposed += drafts.items.len;
            defer h_cur.deinit(); // all exits below (incl. error breaks)
            prof_lap(prof_now, &prof_last, &prof_draft_ns);
            prof_rounds +|= 1;
            if (mem_probe_on and prof_rounds <= 6) _ = c.c.mlx_get_active_memory(&phase_mem[0]);

            // 2. Verify [t_next, d1..dN] on the backbone (continuation).
            var vids = std.ArrayList(u32).empty;
            defer vids.deinit(allocator);
            vids.append(allocator, t_next.token) catch break;
            vids.appendSlice(allocator, drafts.items) catch break;
            const vh = mtp.verify(self.model.ptr, vids.items) catch |err| {
                std.log.err("EngineLoop: MTP verify failed: {}", .{err});
                break;
            };
            // The verify box pins (N+1) x ~150 MB x 48 GDN layers of retained
            // per-step states — it must be freed on EVERY exit from this
            // round (normal, stop, max_tokens, and all error paths).
            defer mtp.freeVerify(self.model.ptr, vh);
            prof_lap(prof_now, &prof_last, &prof_verify_ns);
            if (mem_probe_on and prof_rounds <= 6) _ = c.c.mlx_get_active_memory(&phase_mem[1]);

            // 3. Acceptance.
            // Greedy fast path: one batched argmax over all L verify rows
            // (single GPU op + single sync) instead of L per-row sampler
            // round-trips. NB: GPU argmax tie-breaks may differ from the CPU
            // first-max scan on exact f32 ties (measure-zero); both are valid
            // greedy tokens.
            var greedy_tokens: []i32 = &.{}; // borrowed from greedy_am_data
            var greedy_am: ?Array = null;
            var greedy_am_data: []i32 = &.{};
            defer if (greedy_am_data.len > 0) allocator.free(greedy_am_data);
            defer if (greedy_am) |am| am.deinit();
            if (greedy) {
                if (mtp.verifyArgmax) |va| {
                    if (va(self.model.ptr, vh)) |am| {
                        greedy_am = am;
                        if (am.dataSlice(i32)) |d| {
                            greedy_am_data = allocator.dupe(i32, d) catch &.{};
                            greedy_tokens = greedy_am_data;
                        } else |_| {}
                    } else |_| {}
                }
            }
            // Top-k probe: for every drafted position, rank the target's own
            // greedy token inside the drafter's logits. Rank is computed as
            // "how many logits strictly exceed it", which needs no sort.
            if (topk_probe and greedy and greedy_tokens.len > 0) {
                for (draft_logits.items, 0..) |dl, i| {
                    if (i >= topk_depths or i >= greedy_tokens.len) break;
                    const target_tok: usize = @intCast(greedy_tokens[i]);
                    const f32_logits = ops.astype(self.ctx, dl, .float32) catch continue;
                    defer f32_logits.deinit();
                    f32_logits.eval() catch continue;
                    const vals = f32_logits.dataSlice(f32) catch continue;
                    if (target_tok >= vals.len) continue;
                    const tv = vals[target_tok];
                    var rank: u32 = 0;
                    for (vals) |v| {
                        if (v > tv) {
                            rank += 1;
                            if (rank >= 8) break;
                        }
                    }
                    topk_total[i] += 1;
                    if (rank < 1) topk_hit1[i] += 1;
                    if (rank < 2) topk_hit2[i] += 1;
                    if (rank < 4) topk_hit4[i] += 1;
                    if (rank < 8) topk_hit8[i] += 1;
                }
            }

            var k: usize = 0; // accepted drafts
            var rejected = false;
            var stop_in_drafts = false;
            for (0..drafts.items.len) |i| {
                if (greedy and greedy_tokens.len > i) {
                    // Batched-argmax path: no per-row logits pull.
                    if (@as(u32, @intCast(greedy_tokens[i])) != drafts.items[i]) {
                        rejected = true;
                        break;
                    }
                    k += 1;
                    if (generation_mod.isStopToken(drafts.items[i], gen_config.stop_tokens)) {
                        stop_in_drafts = true;
                        break;
                    }
                    continue;
                }
                const v_i = mtp.verifyLogits(self.model.ptr, vh, i) catch {
                    rejected = true;
                    break;
                };
                defer v_i.deinit();
                if (greedy) {
                    const target = req.sampler.sample(v_i, allocator) catch {
                        rejected = true;
                        break;
                    };
                    if (target.token != drafts.items[i]) {
                        rejected = true;
                        break;
                    }
                    k += 1;
                    if (generation_mod.isStopToken(drafts.items[i], gen_config.stop_tokens)) {
                        stop_in_drafts = true;
                        break;
                    }
                } else {
                    // Speculative sampling: accept w.p. min(1, p/q); the
                    // rejection resample happens after the loop.
                    const accept = mtpAcceptProb(v_i, draft_logits.items[i], drafts.items[i], req.sampler.temperature, allocator) catch {
                        rejected = true;
                        break;
                    };
                    const r = req.sampler.prng.random().float(f32);
                    if (r >= accept) {
                        rejected = true;
                        break;
                    }
                    k += 1;
                    if (generation_mod.isStopToken(drafts.items[i], gen_config.stop_tokens)) {
                        stop_in_drafts = true;
                        break;
                    }
                }
            }
            // Acceptance labels for the AVTRACE features: positions before
            // the first rejection are 1, the rejecting position is 0, and
            // anything past it is unlabeled (same rule as vLLM AV fitting).
            if (std.c.getenv("DMLX_MTP_AVTRACE")) |trace_path_z| {
                if (draft_feats.items.len == drafts.items.len and drafts.items.len > 0) {
                    const labeled: usize = if (rejected) @min(k + 1, drafts.items.len) else k;
                    if (labeled > 0) {
                        var buf: [160]u8 = undefined;
                        const path = std.mem.span(trace_path_z);
                        if (std.Io.Dir.cwd().createFile(self.io, path, .{ .truncate = false })) |f| {
                            defer f.close(self.io);
                            const size = if (f.stat(self.io)) |st| st.size else |_| 0;
                            var off: u64 = size;
                            for (0..labeled) |i| {
                                const y: u8 = if (i < k) 1 else 0;
                                const line = std.fmt.bufPrint(&buf, "{{\"pos\":{d},\"lmax\":{d:.4},\"margin\":{d:.4},\"y\":{d}}}\n", .{ i, draft_feats.items[i][0], draft_feats.items[i][1], y }) catch continue;
                                f.writePositionalAll(self.io, line, off) catch break;
                                off += line.len;
                            }
                        } else |_| {}
                    }
                }
            }
            if (stop_in_drafts) {
                // An accepted draft is a stop token: emit through it and end.
                mtp.commit(self.model.ptr, vh, k + 1, vids.items);
                for (drafts.items[0..k]) |d| {
                    if (total_generated >= gen_config.max_tokens) break;
                    context.append(allocator, d) catch break;
                    total_generated += 1;
                    drafts_accepted += 1;
                }
                stopped = true;
                finish = .stop;
                break;
            }

            // 4. Next token: from V_k (rejection point) or bonus V_N.
            const next_pos = k; // logits position predicting the next token
            const next_sample = if (greedy and greedy_tokens.len > next_pos)
                // Batched argmax already holds this row's greedy token.
                sampling_mod.SampleResult{ .token = @intCast(greedy_tokens[next_pos]), .logprob = 0.0 }
            else blk: {
                const v_next = mtp.verifyLogits(self.model.ptr, vh, next_pos) catch break;
                defer v_next.deinit();
                break :blk if (greedy or !rejected)
                    // Greedy: exact target argmax (rejection) / sampler (bonus).
                    // temp>0 with all drafts accepted: plain bonus sample.
                    req.sampler.sample(v_next, allocator) catch break
                else
                    // temp>0 rejection: resample from the adjusted distribution
                    // (p_target - q_draft)+, preserving the target distribution.
                    mtpResampleAdjusted(v_next, draft_logits.items[k], req.sampler.temperature, &req.sampler.prng, allocator) catch break;
            };

            var hidden_next_opt: ?Array = mtp.verifyHidden(self.model.ptr, vh, next_pos) catch break;
            defer if (hidden_next_opt) |hn| hn.deinit();
            prof_lap(prof_now, &prof_last, &prof_accept_ns);
            if (mem_probe_on and prof_rounds <= 6) _ = c.c.mlx_get_active_memory(&phase_mem[2]);

            // Commit k+1 backbone positions (t_next + accepted drafts); the
            // adapter replays the accepted drafts through the drafter with
            // the target's verify hidden (see MtpOps.commit).
            if (std.c.getenv("DMLX_MTP_DEBUG") != null) {
                std.debug.print("MTP round: n={d} ms={d:.1} t={d} drafts={any} k={d} t'={d}\n", .{
                    n_round,
                    @as(f64, @floatFromInt(prof_now() - round_t0)) / 1e6,
                    t_next.token,
                    drafts.items,
                    k,
                    next_sample.token,
                });
            }
            mtp.commit(self.model.ptr, vh, k + 1, vids.items);
            prof_lap(prof_now, &prof_last, &prof_commit_ns);
            if (comptime true) {
                // DMLX_MEM_PROBE_ROUNDS=1: per-phase MLX active memory for
                // the first 6 rounds (leak hunt 2026-10-30).
                if (mem_probe_on and prof_rounds <= 6) {
                    _ = c.c.mlx_get_active_memory(&phase_mem[3]);
                    std.debug.print("MEMROUND r={d} after-draft={d:.1} after-verify={d:.1} after-accept={d:.1} after-commit={d:.1} MB\n", .{
                        prof_rounds,
                        @as(f64, @floatFromInt(phase_mem[0])) / 1048576.0,
                        @as(f64, @floatFromInt(phase_mem[1])) / 1048576.0,
                        @as(f64, @floatFromInt(phase_mem[2])) / 1048576.0,
                        @as(f64, @floatFromInt(phase_mem[3])) / 1048576.0,
                    });
                }
            }
            // Emit accepted drafts + the next sampled token, capped at
            // max_tokens (plain decode emits exactly max_tokens).
            for (drafts.items[0..k]) |d| {
                if (total_generated >= gen_config.max_tokens) break;
                context.append(allocator, d) catch break;
                total_generated += 1;
                drafts_accepted += 1;
            }
            if (total_generated >= gen_config.max_tokens) break;
            t_next = next_sample;
            context.append(allocator, t_next.token) catch break;
            total_generated += 1;
            hidden.deinit();
            hidden = hidden_next_opt.?;
            hidden_next_opt = null;
            if (generation_mod.isStopToken(t_next.token, gen_config.stop_tokens)) {
                stopped = true;
                finish = .stop;
            }
            // Feed this round's realized cost-per-token back to the depth
            // controller. Only full rounds are scored; the `break` paths
            // above (draft failure, stop token, max_tokens) are not
            // representative of steady-state cost.
            {
                const round_ns = prof_now() - round_t0;
                depth_ctl.record(n_round, if (round_ns > 0) @intCast(round_ns) else 0, k);
                if (std.c.getenv("DMLX_MTP_ROUND_LOG") != null) {
                    std.debug.print("MTP-ROUND r={d} n={d} k={d} ms={d:.1}\n", .{ depth_ctl.rounds, n_round, k, @as(f64, @floatFromInt(round_ns)) / 1e6 });
                }
            }
        }
        hidden.deinit();

        if (drafts_proposed > 0) {
            const ms = struct {
                fn f(ns: u64) f64 {
                    return @as(f64, @floatFromInt(ns)) / 1e6;
                }
            }.f;
            const rounds = if (prof_rounds > 0) prof_rounds else 1;
            // Depth controller trace: rounds spent at each depth and the
            // converged ms/token estimate that drove the choice.
            var depth_buf: [512]u8 = undefined;
            var depth_len: usize = 0;
            for (1..depth_ctl.max_n + 1) |n| {
                if (depth_ctl.used[n - 1] == 0) continue;
                const chunk = std.fmt.bufPrint(
                    depth_buf[depth_len..],
                    " N={d}:{d}r/{d:.1}ms/tok(E[t]={d:.2})",
                    .{ n, depth_ctl.used[n - 1], depth_ctl.score(n) orelse 0.0, depth_ctl.expectedTokens(n) },
                ) catch break;
                depth_len += chunk.len;
            }
            std.log.info("[Engine] MTP depth{s}", .{depth_buf[0..depth_len]});
            // Per-depth CONDITIONAL acceptance p_i = P(draft i accepted |
            // drafts 1..i-1 accepted), with the raw counts behind it. This is
            // the quantity that decides everything about depth: expected
            // tokens is 1 + sum_i prod_{j<=i} p_j. The aggregate
            // "acceptance %" below mixes depths and is NOT comparable across
            // engines (it depends on how early-break rounds are counted).
            var p_buf: [256]u8 = undefined;
            var p_len: usize = 0;
            for (1..depth_ctl.max_n + 1) |i| {
                const reached = depth_ctl.reached[i - 1];
                if (reached < 0.05) continue;
                const acc = depth_ctl.accepted[i - 1];
                const chunk = std.fmt.bufPrint(p_buf[p_len..], " p{d}={d:.3}(~{d:.1}/{d:.1})", .{
                    i,
                    if (reached > 0) @as(f64, @floatCast(acc / reached)) else 0.0,
                    @as(f64, @floatCast(acc)),
                    @as(f64, @floatCast(reached)),
                }) catch break;
                p_len += chunk.len;
            }
            std.log.info("[Engine] MTP cond-acceptance{s}", .{p_buf[0..p_len]});
            qwen38_gdn_prof.gdnVerifyProfReport();
            qwen38_gdn_prof.gdnPrefillProfReport(); // front sub-splits during verify (DMLX_GDN_PREFILL_PROF)
            verify_qmm_mod.tallyReport();
            if (topk_total[0] > 0) {
                // Drafter top-k hit rate per depth: how often the target's
                // own token was in the drafter's top-k. top1 == p_i; the
                // gap between top1 and top2/top4 is exactly the payoff a
                // multi-branch (tree) draft could capture, which is now
                // affordable because sg8's cost is flat in row count.
                var k_buf: [320]u8 = undefined;
                var k_len: usize = 0;
                for (0..topk_depths) |i| {
                    if (topk_total[i] == 0) continue;
                    const tot: f64 = @floatFromInt(topk_total[i]);
                    const chunk = std.fmt.bufPrint(k_buf[k_len..], " d{d}:top1={d:.3} top2={d:.3} top4={d:.3} top8={d:.3} (n={d})", .{
                        i + 1,
                        @as(f64, @floatFromInt(topk_hit1[i])) / tot,
                        @as(f64, @floatFromInt(topk_hit2[i])) / tot,
                        @as(f64, @floatFromInt(topk_hit4[i])) / tot,
                        @as(f64, @floatFromInt(topk_hit8[i])) / tot,
                        topk_total[i],
                    }) catch break;
                    k_len += chunk.len;
                }
                std.log.info("[Engine] MTP drafter top-k{s}", .{k_buf[0..k_len]});
            }
            std.log.info("[Engine] MTP request {d}: acceptance {d:.1}% ({d}/{d} drafts); ms/round draft={d:.1} verify={d:.1} accept={d:.1} commit={d:.1}", .{
                req.id,
                100.0 * @as(f64, @floatFromInt(drafts_accepted)) / @as(f64, @floatFromInt(drafts_proposed)),
                drafts_accepted,
                drafts_proposed,
                ms(prof_draft_ns / rounds),
                ms(prof_verify_ns / rounds),
                ms(prof_accept_ns / rounds),
                ms(prof_commit_ns / rounds),
            });
        }

        depth_ctl.finishRequest();
        if (MtpDepthController.classMemoryOn()) {
            std.log.info("[Engine] MTP class={s} samples={d}", .{
                @tagName(depth_ctl.active_class),
                depth_ctl.class_mem[@intFromEnum(depth_ctl.active_class)].samples,
            });
        }

        const generated_slice = context.items[req.prompt_tokens.len..];
        const text = self.tokenizer.decode(generated_slice, allocator) catch {
            req.completion.deliverError(self.io, "Failed to decode tokens");
            return;
        };
        defer allocator.free(text);
        var final_text = text;
        if (req.stop_strings) |stop_strings| {
            for (stop_strings) |stop_str| {
                if (std.mem.indexOf(u8, text, stop_str)) |idx| {
                    final_text = text[0..idx];
                    break;
                }
            }
        }
        req.token_count = total_generated;
        req.completion.deliverToken(self.io, 0, final_text, true, if (finish == .stop) .stop else finishReasonForTokens(generated_slice, gen_config));
    }

    /// Guided decoding path: uses FSM-based logit masking to constrain output.
    /// Supports JSON schema and regex pattern constraints.
    ///
    /// Interface design: creates a GuidedDecoder from the request's constraint,
    /// then uses generateGuided() for the actual generation loop.
    fn processGuidedRequest(
        self: *EngineLoop,
        req: *RequestState,
        caches: []KVCacheStrategy,
        gen_config: GenerateConfig,
    ) void {
        // Build the FSM from the constraint
        var fsm: guided_mod.FiniteStateMachine = undefined;
        if (req.guided_json_schema) |schema| {
            fsm = guided_mod.FiniteStateMachine.fromJsonSchema(self.allocator, schema) catch {
                req.completion.deliverError(self.io, "Failed to build guided decoding FSM from JSON schema");
                return;
            };
        } else if (req.guided_regex) |pattern| {
            fsm = guided_mod.FiniteStateMachine.fromRegex(self.allocator, pattern) catch {
                req.completion.deliverError(self.io, "Failed to build guided decoding FSM from regex");
                return;
            };
        } else {
            // No constraint — shouldn't reach here, but fall back to normal generation
            self.processNonStreamingRequest(req, caches, gen_config);
            return;
        }

        var decoder = guided_mod.GuidedDecoder.init(fsm);
        defer decoder.deinit();

        var sampler = SamplerConfig{
            .temperature = gen_config.temperature,
            .top_k = gen_config.top_k,
            .top_p = gen_config.top_p,
            .prng = std.Random.DefaultPrng.init(gen_config.seed),
            .repetition_penalty = gen_config.repetition_penalty,
        };

        const tokens = generateGuided(
            self.model,
            req.prompt_tokens,
            gen_config.max_tokens,
            &decoder,
            &sampler,
            caches,
            self.ctx,
            self.allocator,
        ) catch |err| {
            std.log.err("EngineLoop: guided generation failed: {}", .{err});
            req.completion.deliverError(self.io, "Guided generation failed");
            return;
        };
        defer self.allocator.free(tokens);

        // Decode tokens to text
        const text = self.tokenizer.decode(tokens, self.allocator) catch {
            req.completion.deliverError(self.io, "Failed to decode guided tokens");
            return;
        };
        defer self.allocator.free(text);

        req.token_count = @intCast(tokens.len);
        req.completion.deliverToken(self.io, 0, text, true, finishReasonForTokens(tokens, gen_config));
    }

    fn processDSV4StreamingRequest(
        self: *EngineLoop,
        req: *RequestState,
        caches: []KVCacheStrategy,
        gen_config: GenerateConfig,
        model: *dsv4_mod.DSV4Model,
        skip_prefill: bool,
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

                // Deliver token immediately via CompletionSignal.
                // CompletionSignal uses atomic spinlock + Darwin ulock for cross-thread
                // notification. The wakeWaiter() inside deliverToken wakes up the
                // HTTP worker thread immediately — no io.sleep needed here.
                //
                // Architecture: Engine runs on main thread, HTTP handlers on worker threads.
                // io.sleep(0) only yields to fibers on the SAME thread, which is useless
                // here. The cross-thread ulock wake is the correct synchronization
                // mechanism (matches omlx's asyncio.Event pattern).
                ctx.req.completion.deliverToken(
                    ctx.engine.io,
                    token,
                    token_text,
                    is_final,
                    if (is_final) streamFinalReason(token, ctx.req.stop_tokens, ctx.token_count, ctx.req.max_tokens) else null,
                );
            }
        }.callback;

        // Generate with streaming callback - tokens are delivered immediately
        const start_pos = if (skip_prefill) req.prompt_tokens.len else null;
        const tokens = model.generateWithCallback(
            req.prompt_tokens,
            req.max_tokens,
            &sampler,
            caches,
            self.stream,
            &stream_ctx,
            streamCallback,
            start_pos,
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
        skip_prefill: bool,
    ) void {
        var sampler = root.sampling.SamplerConfig{
            .temperature = gen_config.temperature,
            .top_k = gen_config.top_k,
            .top_p = gen_config.top_p,
            .prng = std.Random.DefaultPrng.init(gen_config.seed),
            .repetition_penalty = gen_config.repetition_penalty,
        };

        const start_pos = if (skip_prefill) req.prompt_tokens.len else null;
        const tokens = model.generate(req.prompt_tokens, req.max_tokens, &sampler, caches, self.stream, start_pos) catch |err| {
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
        req.completion.deliverToken(self.io, 0, final_text, true, finishReasonForTokens(tokens, gen_config));
    }

    fn logRequestCompletion(io: std.Io, req: *RequestState) void {
        _ = io;
        const end_ticks = std.c.mach_absolute_time();
        const duration_ticks = end_ticks - req.start_time_ns; // start_time_ns is actually ticks
        // Convert ticks to ns: timebase = 125/3 on Apple Silicon
        const duration_ns = @as(u64, @intCast(duration_ticks)) * 125 / 3;
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
        // Leak probe: active MLX bytes should return to baseline after each
        // request; monotonic growth == leaked arrays (see metal::malloc
        // resource-limit crashes on long MTP sessions).
        if (std.c.getenv("DMLX_MEM_PROBE") != null) {
            var active: usize = 0;
            var peak: usize = 0;
            _ = c.c.mlx_get_active_memory(&active);
            _ = c.c.mlx_get_peak_memory(&peak);
            std.log.info("[MemProbe] id={d} active={d:.2} GB peak={d:.2} GB", .{
                req.id,
                @as(f64, @floatFromInt(active)) / 1073741824.0,
                @as(f64, @floatFromInt(peak)) / 1073741824.0,
            });
        }
    }

    /// Warmup: generate a few tokens to force backbone page-in before real requests.
    /// Without this, the first real request pays a ~9s backbone cold-page-in penalty.
    /// This replaces the old expert-cache warmup removed in P2.1.
    pub fn warmupBackbone(self: *EngineLoop) void {
        if (self.dsv4_model == null) return;

        std.log.info("[Engine] Warming up backbone (pre-generating tokens to page-in weights)...", .{});

        const model = self.dsv4_model.?;
        var sampler = sampling_mod.SamplerConfig{
            .temperature = 0,
            .top_k = 1,
            .prng = std.Random.DefaultPrng.init(0),
        };

        const warmup_prompts = [_][]const u32{
            &[_]u32{1},
            &[_]u32{ 1, 100, 500 },
        };

        for (warmup_prompts, 0..) |prompt, i| {
            const caches = self.createCaches() catch {
                std.log.warn("[Engine] Warmup {d}: createCaches failed", .{i});
                continue;
            };
            defer {
                for (caches) |cache| cache.deinit(self.allocator);
                self.allocator.free(caches);
            }

            const tokens = model.generate(prompt, 1, &sampler, caches, self.stream, null) catch {
                std.log.warn("[Engine] Warmup {d}: generate failed", .{i});
                continue;
            };
            self.allocator.free(tokens);
        }

        std.log.info("[Engine] Backbone warmup complete ({d} prompts)", .{warmup_prompts.len});
        // Signal DyMoE to compute skip masks from collected expert stats
        if (self.dsv4_model) |m| m.finishExpertWarmup();
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

// ============================================================
// Guided Decoding Integration
// ============================================================

/// Generate tokens with guided decoding (JSON schema or regex constraint).
/// Uses the GuidedDecoder FSM to mask logits at each step, ensuring output
/// conforms to the specified grammar.
///
/// Interface design: accepts a GuidedDecoder that can be created from either
/// a JSON schema or regex pattern. The caller is responsible for creating
/// and destroying the decoder.
///
/// Extensibility: new constraint types can be added by implementing new
/// FSM builders in guided.zig without modifying this function.
pub fn generateGuided(
    model: ModelVTable,
    prompt_tokens: []const u32,
    max_tokens: usize,
    decoder: *guided_mod.GuidedDecoder,
    sampler: *SamplerConfig,
    caches: []KVCacheStrategy,
    ctx: EagerContext,
    allocator: std.mem.Allocator,
) ![]u32 {
    var result = std.ArrayList(u32).empty;
    errdefer result.deinit(allocator);

    // Prefill: run prompt through model
    const prompt_arr = try array_mod.Array.fromData(
        allocator,
        u32,
        prompt_tokens,
        &[_]i32{ 1, @intCast(prompt_tokens.len) },
    );
    defer prompt_arr.deinit();

    const prefill_logits = try model.forward(model.ptr, prompt_arr, null, caches);
    defer prefill_logits.deinit();

    // Extract last position logits: shape [1, seq_len, vocab] → [1, 1, vocab]
    const logits_shape = prefill_logits.shape();
    const seq_len: usize = @intCast(logits_shape[1]);
    const last_logits = try ops.slice(
        ctx,
        prefill_logits,
        &[_]i32{ 0, @intCast(seq_len - 1), 0 },
        &[_]i32{ 1, @intCast(seq_len), @intCast(logits_shape[2]) },
        &[_]i32{ 1, 1, 1 },
    );
    defer last_logits.deinit();

    // Apply guided decoding mask
    const masked_logits = try decoder.maskLogits(last_logits, ctx);
    defer masked_logits.deinit();

    // Sample first token using SamplerConfig
    const first_sample = try sampler.sample(masked_logits, allocator);
    try result.append(allocator, first_sample.token);
    decoder.advance(first_sample.token);

    // Autoregressive generation with guided masking
    var step: usize = 1;
    while (step < max_tokens and !decoder.isComplete()) : (step += 1) {
        const input_token = result.items[result.items.len - 1];
        const input_arr = try array_mod.Array.fromData(
            allocator,
            u32,
            &[_]u32{input_token},
            &[_]i32{ 1, 1 },
        );
        defer input_arr.deinit();

        const step_logits = try model.forward(model.ptr, input_arr, null, caches);
        defer step_logits.deinit();

        // Apply guided decoding mask
        const step_masked = try decoder.maskLogits(step_logits, ctx);
        defer step_masked.deinit();

        // Sample
        const step_sample = try sampler.sample(step_masked, allocator);
        try result.append(allocator, step_sample.token);
        decoder.advance(step_sample.token);
    }

    return try result.toOwnedSlice(allocator);
}

test "streamFinalReason: stop token vs max_tokens vs internal EOS" {
    const stop_tokens = [_]u32{ 100, 200 };
    // Configured stop token → .stop regardless of count.
    try std.testing.expectEqual(TokenFinishReason.stop, streamFinalReason(100, &stop_tokens, 3, 64));
    try std.testing.expectEqual(TokenFinishReason.stop, streamFinalReason(200, &stop_tokens, 64, 64));
    // Cap reached without a stop token → .length.
    try std.testing.expectEqual(TokenFinishReason.length, streamFinalReason(42, &stop_tokens, 64, 64));
    try std.testing.expectEqual(TokenFinishReason.length, streamFinalReason(42, &stop_tokens, 65, 64));
    // Internal EOS (not in stop_tokens, cap not reached) → .stop.
    try std.testing.expectEqual(TokenFinishReason.stop, streamFinalReason(1, &stop_tokens, 12, 64));
    // Empty stop list: cap → .length, otherwise → .stop.
    try std.testing.expectEqual(TokenFinishReason.length, streamFinalReason(1, &.{}, 8, 8));
    try std.testing.expectEqual(TokenFinishReason.stop, streamFinalReason(1, &.{}, 7, 8));
}
