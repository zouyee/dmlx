/// MLX-free inference engine for DeepSeek-V4-Flash.
/// Uses native_loader + engine.c/Metal directly. No MLX runtime dependency.
///
/// Design notes:
/// - Expert weights are loaded on-demand via packed binary pread (engine.c).
///   This matches the existing flash-moe streaming pattern and avoids OOM.
/// - Backbone weights (embed, lm_head, attention, norms) are loaded via
///   native_loader and held in CPU memory. engine.c copies them to GPU
///   buffers on each forward. Future optimization: create persistent MTLBuffers
///   to eliminate per-forward copies.
const std = @import("std");
const native_loader = @import("native_loader/loader.zig");
const metal = @import("metal_infer/engine.zig");
const sampling_mod = @import("sampling.zig");

const DIM = 4096;
const MHC_MULT = 4;
const N_LAYERS = 43;
// EOS token ID: 1 is the `</s>` token for DeepSeek-V4-Flash tokenizer.
// This matches the BpeTokenizer.eos_id loaded from tokenizer.json.
const EOS_TOKEN: u32 = 1;

pub const NativeEngine = struct {
    allocator: std.mem.Allocator,
    engine: *metal.Engine,
    weight_store: native_loader.NativeWeightStore,
    config: native_loader.DSV4NativeConfig,
    logits_buffer: []f32,
    eos_token: u32 = EOS_TOKEN,

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8, packed_dir: []const u8) !NativeEngine {
        // 1. Load config
        var loader = try native_loader.NativeEngineLoader.init(allocator, model_path);
        defer loader.deinit();
        const cfg = try loader.loadConfig();

        // 2. Load weights
        var store = try loader.loadWeights(cfg, 0);
        errdefer store.deinit();

        // 3. Init metal engine
        const engine = try metal.init(packed_dir);
        errdefer metal.deinit(engine);

        // 4. Set weights
        metal.setWeights(engine, store.weights);

        // 4b. Set compressor/indexer weights per layer (T2C.1)
        for (0..cfg.num_hidden_layers) |i| {
            const ratio = store.weights.compress_ratio[i];
            if (ratio > 0) {
                if (store.weights.comp_wkv[i]) |wkv| {
                    const wgate = store.weights.comp_wgate[i].?;
                    metal.setLayerCompressor(
                        engine,
                        i,
                        ratio,
                        metal.toCQuantWeight(wkv),
                        metal.toCQuantWeight(wgate),
                        if (store.weights.comp_ape[i].len > 0) store.weights.comp_ape[i].ptr else null,
                        if (store.weights.comp_norm[i].len > 0) store.weights.comp_norm[i].ptr else null,
                    );
                }
                if (ratio == 4) {
                    if (store.weights.idx_wq_b[i]) |wqb| {
                        metal.setLayerIndexer(
                            engine,
                            i,
                            metal.toCQuantWeight(wqb),
                            metal.toCQuantWeight(store.weights.idx_weights_proj[i].?),
                            metal.toCQuantWeight(store.weights.idx_comp_wkv[i].?),
                            metal.toCQuantWeight(store.weights.idx_comp_wgate[i].?),
                            if (store.weights.idx_comp_ape[i].len > 0) store.weights.idx_comp_ape[i].ptr else null,
                            if (store.weights.idx_comp_norm[i].len > 0) store.weights.idx_comp_norm[i].ptr else null,
                        );
                    }
                }
            }
        }

        // 5. Allocate logits buffer
        const logits = try allocator.alloc(f32, cfg.vocab_size);
        errdefer allocator.free(logits);

        // 6. SMELT startup preload — equivalent to MLX's ExpertPreloadProvider.
        //    Preloads top-N experts per layer into RAM at engine init.
        //    With routing bias (-1e9 for uncached experts), 100% of routing
        //    hits the cache → zero SSD I/O during decode → reads from RAM @100 GB/s.
        //
        //    Memory: 51 experts/layer × 43 layers × 13.4MB = 29.4 GB  (fits in 48GB)
        //    Startup cost: ~36s (29.4 GB / 0.82 GB/s SSD) — one-time per server start
        //    Decode speed: 3.46 GB/token / 100 GB/s + ~260ms GPU = ~295ms/tok ≈ 3.4 tok/s
        //
        //    NATIVE_SMELT_N=0 to disable (full SSD reads, ~4200ms/tok)
        const smelt_n_str = std.c.getenv("NATIVE_SMELT_N");
        const smelt_n: i32 = if (smelt_n_str) |s| std.fmt.parseInt(i32, std.mem.span(s), 10) catch 51 else 51;
        if (smelt_n > 0) {
            std.log.info("native_engine: SMELT preloading {d} experts/layer ({d:.1} GB) — please wait...", .{
                smelt_n,
                @as(f64, @floatFromInt(smelt_n)) * 43.0 * 13.4 / 1024.0,
            });
            // warmup_tokens=0: no routing stats → preloads experts 0..N-1 (uniform, same as MLX default)
            // Synchronous: blocks until all 29GB is loaded into RAM.
            // Routing bias activates immediately after preload completes.
            metal.smeltInit(engine, 0, smelt_n, 1e9);
            const n_loaded = metal.smeltFinishWarmup(engine);
            if (n_loaded > 0) {
                std.log.info("native_engine: SMELT ready — {d} experts/layer in RAM, routing bias active", .{n_loaded});
                // Gather mode: single-dispatch all K experts from the contiguous SMELT pool.
                // NOTE: Currently disabled by default — gather kernel with 13MB expert-stride causes
                // scattered cache-unfriendly memory access that is SLOWER than 6 separate contiguous reads.
                // Enable via NATIVE_GATHER=1 only for experimentation.
                const gather_env = std.c.getenv("NATIVE_GATHER");
                if (gather_env != null) {
                    const gather_ok = metal.initGatherMode(engine);
                    if (gather_ok > 0) {
                        std.log.info("native_engine: gather mode active (EXPERIMENTAL — may be slower)", .{});
                    } else {
                        std.log.warn("native_engine: gather mode init failed", .{});
                    }
                }
            } else {
                std.log.warn("native_engine: SMELT preload failed — falling back to SSD reads", .{});
            }
        } else {
            std.log.warn("native_engine: SMELT disabled (NATIVE_SMELT_N=0) — SSD reads each token (~4200ms/tok)", .{});
        }

        std.log.info("native_engine: initialized (layers={d}, vocab={d})", .{ cfg.num_hidden_layers, cfg.vocab_size });

        return .{
            .allocator = allocator,
            .engine = engine,
            .weight_store = store,
            .config = cfg,
            .logits_buffer = logits,
        };
    }

    pub fn deinit(self: *NativeEngine) void {
        self.allocator.free(self.logits_buffer);
        metal.deinit(self.engine);
        self.weight_store.deinit();
    }

    pub fn resetKv(self: *NativeEngine) void {
        metal.resetKv(self.engine);
    }

    // ------------------------------------------------------------------
    // Generate API (compatible with DSV4Model.generate)
    //
    // NOTE: `caches` and `stream` parameters are accepted for API compatibility
    // with the existing engine loop. In native mode, the Metal engine manages
    // its own KV cache (f16) and does not use MLX streams. Expert weights are
    // loaded on-demand via packed binary pread (no SMELT preloading needed).
    // ------------------------------------------------------------------

    pub fn generate(
        self: *NativeEngine,
        prompt_tokens: []const u32,
        max_new_tokens: usize,
        sampler_config: *sampling_mod.SamplerConfig,
        caches: anytype,
        stream: anytype,
        start_pos_override: ?usize,
    ) ![]u32 {
        _ = caches;
        _ = stream;

        std.log.info("native_engine: prompt_tokens={any}", .{prompt_tokens});
        const allocator = self.allocator;
        if (max_new_tokens == 0) {
            return try allocator.alloc(u32, 0);
        }

        // Skip BOS token (token 0) if present — MLX strips it before generation
        const has_bos = prompt_tokens.len > 0 and prompt_tokens[0] == EOS_TOKEN;
        const prompt_offset: usize = if (has_bos) @as(usize, 1) else @as(usize, 0);
        const effective_prompt_len = prompt_tokens.len - prompt_offset;
        var tokens = try allocator.alloc(u32, effective_prompt_len + max_new_tokens);
        defer allocator.free(tokens);
        @memcpy(tokens[0..effective_prompt_len], prompt_tokens[prompt_offset..]);

        var current_len = effective_prompt_len;
        var start_pos: usize = start_pos_override orelse 0;

        // Reset KV cache at the start of each new sequence
        self.resetKv();

        // Prefill: token-by-token forward (same path as decode for numerical consistency).
        // Using batch forward caused ~0.73% per-layer error vs MLX batch GPU path;
        // single-token forward is self-consistent and avoids this issue.
        if (start_pos_override == null and prompt_tokens.len > 0) {
            for (prompt_tokens, 0..) |tok, t| {
                var hidden: [MHC_MULT * DIM]f32 = undefined;
                metal.setTokenId(self.engine, @intCast(tok));
                metal.embed(self.engine, @intCast(tok), &hidden);
                try metal.forward(self.engine, hidden[0..], @intCast(start_pos + t));

                // On the last prompt token, compute logits for the first decode token
                if (t == prompt_tokens.len - 1) {
                    var compressed: [DIM]f32 = undefined;
                    metal.hyperHeadCompress(
                        self.weight_store.weights.hc_head_fn.ptr,
                        self.weight_store.weights.hc_head_base.ptr,
                        self.weight_store.weights.hc_head_scale.ptr,
                        &hidden,
                        &compressed,
                    );
                    try metal.getLogits(self.engine, &compressed, self.logits_buffer.ptr);

                    const next_token = sampleFromLogits(self.logits_buffer, self.config.vocab_size, sampler_config);
                    start_pos += prompt_tokens.len;
                    tokens[current_len] = next_token;
                    current_len += 1;

                    if (next_token == self.eos_token) {
                        std.log.info("native_engine: EOS after prefill", .{});
                    }
                }
            }
        }

        // Decode loop — signal SMELT that prefill is done (enables decode token counting for warmup)
        metal.smeltSetDecodePhase(self.engine);
        const decode_count = if (start_pos_override != null) max_new_tokens else max_new_tokens - 1;
        for (0..decode_count) |_| {
            if (current_len >= tokens.len) break;

            var hidden: [MHC_MULT * DIM]f32 = undefined;
            metal.setTokenId(self.engine, @intCast(tokens[current_len - 1]));
            metal.embed(self.engine, @intCast(tokens[current_len - 1]), &hidden);
            try metal.forward(self.engine, hidden[0..], @intCast(start_pos));

            var compressed: [DIM]f32 = undefined;
            metal.hyperHeadCompress(
                self.weight_store.weights.hc_head_fn.ptr,
                self.weight_store.weights.hc_head_base.ptr,
                self.weight_store.weights.hc_head_scale.ptr,
                &hidden,
                &compressed,
            );
            try metal.getLogits(self.engine, &compressed, self.logits_buffer.ptr);

            const next_token = sampleFromLogits(self.logits_buffer, self.config.vocab_size, sampler_config);
            tokens[current_len] = next_token;
            current_len += 1;
            start_pos += 1;

            if (next_token == self.eos_token) {
                std.log.info("native_engine: EOS token generated, stopping at pos={d}", .{start_pos});
                break;
            }
        }

        // Return only generated tokens (skip prompt)
        const result = try allocator.alloc(u32, current_len - prompt_tokens.len);
        @memcpy(result, tokens[prompt_tokens.len..current_len]);
        return result;
    }

    /// Streaming generate with callback (compatible with DSV4Model.generateWithCallback)
    pub fn generateWithCallback(
        self: *NativeEngine,
        prompt_tokens: []const u32,
        max_new_tokens: usize,
        sampler_config: *sampling_mod.SamplerConfig,
        caches: anytype,
        stream: anytype,
        callback_ctx: ?*anyopaque,
        callback: ?StreamCallback,
        start_pos_override: ?usize,
    ) ![]u32 {
        _ = caches;
        _ = stream;

        const allocator = self.allocator;
        if (max_new_tokens == 0) {
            return try allocator.alloc(u32, 0);
        }

        // Skip BOS token (token 0) if present — MLX strips it before generation
        const has_bos = prompt_tokens.len > 0 and prompt_tokens[0] == EOS_TOKEN;
        const prompt_offset: usize = if (has_bos) @as(usize, 1) else @as(usize, 0);
        const effective_prompt_len = prompt_tokens.len - prompt_offset;
        var tokens = try allocator.alloc(u32, effective_prompt_len + max_new_tokens);
        defer allocator.free(tokens);
        @memcpy(tokens[0..effective_prompt_len], prompt_tokens[prompt_offset..]);

        var current_len = effective_prompt_len;
        var start_pos: usize = start_pos_override orelse 0;

        // Reset KV cache
        self.resetKv();

        // Prefill: batch forward (proper transformer order: all tokens per layer).
        if (start_pos_override == null and prompt_tokens.len > 0) {
            // Token-by-token prefill (same path as decode for numerical consistency)
            for (prompt_tokens, 0..) |tok, t| {
                var hidden: [MHC_MULT * DIM]f32 = undefined;
                metal.setTokenId(self.engine, @intCast(tok));
                metal.embed(self.engine, @intCast(tok), &hidden);
                try metal.forward(self.engine, hidden[0..], @intCast(start_pos + t));

                if (t == prompt_tokens.len - 1) {
                    var compressed: [DIM]f32 = undefined;
                    metal.hyperHeadCompress(
                        self.weight_store.weights.hc_head_fn.ptr,
                        self.weight_store.weights.hc_head_base.ptr,
                        self.weight_store.weights.hc_head_scale.ptr,
                        &hidden,
                        &compressed,
                    );
                    try metal.getLogits(self.engine, &compressed, self.logits_buffer.ptr);

                    const next_token = sampleFromLogits(self.logits_buffer, self.config.vocab_size, sampler_config);
                    start_pos += prompt_tokens.len;
                    tokens[current_len] = next_token;
                    current_len += 1;

                    if (callback) |cb| {
                        const is_final = max_new_tokens == 1 or next_token == self.eos_token;
                        cb(callback_ctx.?, next_token, is_final);
                    }

                    if (next_token == self.eos_token) {
                        std.log.info("native_engine: EOS after prefill", .{});
                    }
                }
            }
        }

        // Decode loop — signal SMELT that prefill is done (enables decode token counting for warmup)
        metal.smeltSetDecodePhase(self.engine);
        const decode_count = if (start_pos_override != null) max_new_tokens else max_new_tokens - 1;
        for (0..decode_count) |i| {
            if (current_len >= tokens.len) break;

            var hidden: [MHC_MULT * DIM]f32 = undefined;
            metal.setTokenId(self.engine, @intCast(tokens[current_len - 1]));
            metal.embed(self.engine, @intCast(tokens[current_len - 1]), &hidden);
            try metal.forward(self.engine, hidden[0..], @intCast(start_pos));

            var compressed: [DIM]f32 = undefined;
            metal.hyperHeadCompress(
                self.weight_store.weights.hc_head_fn.ptr,
                self.weight_store.weights.hc_head_base.ptr,
                self.weight_store.weights.hc_head_scale.ptr,
                &hidden,
                &compressed,
            );
            try metal.getLogits(self.engine, &compressed, self.logits_buffer.ptr);

            const next_token = sampleFromLogits(self.logits_buffer, self.config.vocab_size, sampler_config);
            tokens[current_len] = next_token;
            current_len += 1;
            start_pos += 1;

            if (callback) |cb| {
                const is_final = i + 1 >= decode_count or next_token == self.eos_token;
                cb(callback_ctx.?, next_token, is_final);
            }

            if (next_token == self.eos_token) {
                std.log.info("native_engine: EOS token generated, stopping", .{});
                break;
            }
        }

        const result = try allocator.alloc(u32, current_len - prompt_tokens.len);
        @memcpy(result, tokens[prompt_tokens.len..current_len]);
        return result;
    }

    pub const StreamCallback = *const fn (ctx: *anyopaque, token: u32, is_final: bool) void;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// Simple greedy argmax sampling (temperature=0).
    /// When temperature > 0 is needed, this falls back to greedy for now.
    fn sampleFromLogits(logits: []f32, vocab_size: u32, sampler: *sampling_mod.SamplerConfig) u32 {
        _ = sampler;
        var max_idx: usize = 0;
        var max_val: f32 = logits[0];
        for (1..vocab_size) |i| {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        return @intCast(max_idx);
    }
};
