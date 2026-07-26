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
const dspark_mod = @import("dspark.zig");

const DIM = 4096;
const MHC_MULT = 4;
const N_LAYERS = 43;
// BOS token ID: 0 is the `<｜begin▁of▁sentence｜>` token for DeepSeek-V4-Flash.
// EOS token ID: read from the model's config.json (eos_token_id) at init;
// this constant is only the fallback default (DSV4: 1 = `<｜end▁of▁sentence｜>`).
const BOS_TOKEN: u32 = 0;
const EOS_TOKEN: u32 = 1;

pub const NativeEngine = struct {
    allocator: std.mem.Allocator,
    engine: *metal.Engine,
    weight_store: native_loader.NativeWeightStore,
    config: native_loader.DSV4NativeConfig,
    logits_buffer: []f32,
    eos_token: u32 = EOS_TOKEN,
    smelt_stats_path: []u8, // path to routing stats file, owned by NativeEngine
    dspark: ?dspark_mod.DSparkState = null, // DSpark Markov Head (legacy, optional)
    dspark_engine: ?*metal.DSparkEngine = null, // Full DSpark engine (new, optional)

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8, packed_dir: []const u8) !NativeEngine {
        return initWithDSpark(allocator, model_path, packed_dir, null);
    }

    pub fn initWithDSpark(allocator: std.mem.Allocator, model_path: []const u8, packed_dir: []const u8, dspark_dir: ?[]const u8) !NativeEngine {
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

        // Build stats file path: {packed_dir}/.smelt_routing_stats.bin
        const stats_path_buf = try std.fmt.allocPrint(allocator, "{s}/.smelt_routing_stats.bin\x00", .{packed_dir});
        errdefer allocator.free(stats_path_buf);

        if (smelt_n > 0) {
            std.log.info("native_engine: SMELT preloading {d} experts/layer ({d:.1} GB) — please wait...", .{
                smelt_n,
                @as(f64, @floatFromInt(smelt_n)) * 43.0 * 13.4 / 1024.0,
            });
            // Load routing stats from previous runs so smelt_finish_warmup
            // selects the ACTUAL hot experts rather than experts 0..N-1 by ID.
            // MLX-aligned approach: two phases are automatic, no manual pre-warmup needed.
            // - Phase 1 (no stats, default N=20): penalty=0, unbiased routing → discovers hot experts
            // - Phase 2 (stats loaded): penalty=1e3 steers routing to hot experts → I/O ≈ 0
            metal.smeltInit(engine, 0, smelt_n, 0.0);
            // Load stats AFTER smeltInit (smeltInit zeros routing_counts, so load must come after)
            const stats_loaded = metal.smeltLoadStats(engine, stats_path_buf[0 .. stats_path_buf.len - 1 :0].ptr);
            if (stats_loaded != 0) {
                // Stats available: hot experts loaded into SMELT (stats-based selection).
                // Keep penalty=0 so routing remains natural and correct for all prompts.
                // The performance benefit comes from SMELT caching the actual hot experts
                // (not the default 0..N-1), improving cache hit rate without routing bias.
                std.log.info("native_engine: Phase 2 — hot experts loaded from stats (natural routing, N={d})", .{smelt_n});
            } else {
                std.log.info("native_engine: Phase 1 — collecting routing stats (penalty=0, N={d})", .{smelt_n});
            }
            const n_loaded = metal.smeltFinishWarmup(engine);
            // Register stats path for periodic auto-save (protects against OOM kill)
            metal.smeltSetStatsPath(engine, stats_path_buf[0 .. stats_path_buf.len - 1 :0].ptr);
            if (n_loaded > 0) {
                std.log.info("native_engine: SMELT ready — {d} experts/layer in RAM (routing-stats based)", .{n_loaded});
                const gather_env = std.c.getenv("NATIVE_GATHER");
                if (gather_env != null) {
                    const gather_ok = metal.initGatherMode(engine);
                    if (gather_ok > 0) {
                        std.log.info("native_engine: gather mode active (EXPERIMENTAL)", .{});
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

        // Load DSpark Markov Head if directory provided
        var dspark_state: ?dspark_mod.DSparkState = null;
        if (dspark_dir) |dir| {
            dspark_state = dspark_mod.DSparkState.init(allocator, dir, cfg.vocab_size) catch |err| blk: {
                std.log.warn("native_engine: DSpark load failed ({any}), falling back to standard decode", .{err});
                break :blk null;
            };
        }

        // Initialize full DSpark engine if weights directory is available
        // (uses dspark_weights/ subdir for non-expert weights, packed_mtp_experts/ for INT8 experts)
        var dspark_full_engine: ?*metal.DSparkEngine = null;
        if (dspark_dir) |dir| {
            const weight_dir_z = try allocator.dupeZ(u8, dir);
            defer allocator.free(weight_dir_z);
            // Construct packed_mtp_experts path: sibling of dspark_weights dir
            const mtp_expert_path = try std.fmt.allocPrint(allocator, "{s}/../packed_mtp_experts\x00", .{dir});
            defer allocator.free(mtp_expert_path);
            dspark_full_engine = metal.dsparkInit(
                weight_dir_z,
                mtp_expert_path[0 .. mtp_expert_path.len - 1 :0],
                engine,
            );
            if (dspark_full_engine) |de| {
                metal.setDSparkEngine(engine, de);
                std.log.info("native_engine: DSpark full engine initialized (block_size=5)", .{});
            } else {
                std.log.warn("native_engine: DSpark full engine init failed, using Markov-only fallback", .{});
            }
        }

        return .{
            .allocator = allocator,
            .engine = engine,
            .weight_store = store,
            .config = cfg,
            .logits_buffer = logits,
            .eos_token = cfg.eos_token_id,
            .smelt_stats_path = stats_path_buf,
            .dspark = dspark_state,
            .dspark_engine = dspark_full_engine,
        };
    }

    pub fn deinit(self: *NativeEngine) void {
        // Save routing stats so next startup loads the correct hot experts
        metal.smeltSaveStats(self.engine, self.smelt_stats_path[0 .. self.smelt_stats_path.len - 1 :0].ptr);
        self.allocator.free(self.smelt_stats_path);
        self.allocator.free(self.logits_buffer);
        if (self.dspark) |*ds| ds.deinit();
        if (self.dspark_engine) |de| {
            metal.setDSparkEngine(self.engine, null);
            metal.dsparkDeinit(de);
        }
        metal.deinit(self.engine);
        self.weight_store.deinit();
    }

    pub fn resetKv(self: *NativeEngine) void {
        metal.resetKv(self.engine);
        if (self.dspark_engine) |de| metal.dsparkReset(de);
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

        // Skip BOS (token 0) and EOS (token 1) if present — MLX strips both before generation.
        const has_bos = prompt_tokens.len > 0 and prompt_tokens[0] == BOS_TOKEN;
        const has_eos = prompt_tokens.len > 0 and prompt_tokens[prompt_tokens.len - 1] == self.eos_token;
        const prompt_offset: usize = if (has_bos) 1 else 0;
        const eos_offset: usize = if (has_eos) 1 else 0;
        const effective_prompt_len = if (prompt_tokens.len > prompt_offset + eos_offset)
            prompt_tokens.len - prompt_offset - eos_offset
        else
            0;
        var tokens = try allocator.alloc(u32, effective_prompt_len + max_new_tokens);
        defer allocator.free(tokens);
        if (effective_prompt_len > 0) {
            @memcpy(tokens[0..effective_prompt_len], prompt_tokens[prompt_offset .. prompt_tokens.len - eos_offset]);
        }

        var current_len = effective_prompt_len;
        var start_pos: usize = start_pos_override orelse 0;

        // Reset KV cache at the start of each new sequence
        self.resetKv();

        // Prefill: token-by-token forward over effective prompt tokens only.
        if (start_pos_override == null and effective_prompt_len > 0) {
            // Use batched prefill for longer prompts (n>=8): amortizes GPU sync overhead.
            // Fall back to serial for short prompts (n<8): fewer GPU syncs, lower overhead.
            // Threshold: batched requires n*43 GPU syncs vs serial's 43 — only worth it for n>=8.
            if (effective_prompt_len >= 8) {
                const n = effective_prompt_len;
                const hidden_batch = try allocator.alloc(f32, n * MHC_MULT * DIM);
                defer allocator.free(hidden_batch);
                const token_ids_batch = try allocator.alloc(i32, n);
                defer allocator.free(token_ids_batch);

                for (tokens[0..n], 0..) |tok, t| {
                    token_ids_batch[t] = @intCast(tok);
                    metal.setTokenId(self.engine, @intCast(tok));
                    metal.embed(self.engine, @intCast(tok), hidden_batch[t * MHC_MULT * DIM ..][0 .. MHC_MULT * DIM]);
                }

                try metal.forwardBatch(self.engine, hidden_batch, n, @intCast(start_pos), token_ids_batch);

                var hidden_last: [MHC_MULT * DIM]f32 = undefined;
                @memcpy(&hidden_last, hidden_batch[(n - 1) * MHC_MULT * DIM ..][0 .. MHC_MULT * DIM]);

                var compressed: [DIM]f32 = undefined;
                metal.hyperHeadCompress(
                    self.weight_store.weights.hc_head_fn.ptr,
                    self.weight_store.weights.hc_head_base.ptr,
                    self.weight_store.weights.hc_head_scale.ptr,
                    &hidden_last,
                    &compressed,
                );
                try metal.getLogits(self.engine, &compressed, self.logits_buffer.ptr);

                const next_token = sampleFromLogits(self.logits_buffer, self.config.vocab_size, sampler_config);
                start_pos += effective_prompt_len;
                tokens[current_len] = next_token;
                current_len += 1;
                std.log.info("native_engine: effective_prompt_len={d} first_token={d} [batched]", .{ effective_prompt_len, next_token });

                if (next_token == self.eos_token) {
                    std.log.info("native_engine: EOS after prefill", .{});
                }
            } else {
                // Serial prefill: token-by-token (optimal for short prompts n<8)
                for (tokens[0..effective_prompt_len], 0..) |tok, t| {
                    var hidden: [MHC_MULT * DIM]f32 = undefined;
                    metal.setTokenId(self.engine, @intCast(tok));
                    metal.embed(self.engine, @intCast(tok), &hidden);
                    try metal.forward(self.engine, hidden[0..], @intCast(start_pos + t));

                    if (t == effective_prompt_len - 1) {
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
                        start_pos += effective_prompt_len;
                        tokens[current_len] = next_token;
                        current_len += 1;
                        std.log.info("native_engine: effective_prompt_len={d} first_token={d}", .{ effective_prompt_len, next_token });

                        if (next_token == self.eos_token) {
                            std.log.info("native_engine: EOS after prefill", .{});
                        }
                    }
                }
            }
        }

        // Decode loop — signal SMELT that prefill is done (enables decode token counting for warmup)
        metal.smeltSetDecodePhase(self.engine);
        const decode_count = if (start_pos_override != null) max_new_tokens else max_new_tokens - 1;
        var _t_forward_total: u64 = 0;
        var _t_logits_total: u64 = 0;
        var _t_decode_n: u64 = 0;
        const _do_decode_time = (std.c.getenv("NATIVE_DECODE_TIME") != null);

        // DSpark speculative decoding buffers (stack-allocated for block_size <= 8)
        const DSPARK_MAX_BLOCK = 8;
        var dspark_draft_tokens: [DSPARK_MAX_BLOCK]u32 = undefined;
        var dspark_draft_logits_buf: ?[]f32 = null;
        defer if (dspark_draft_logits_buf) |buf| self.allocator.free(buf);
        var dspark_total_drafted: usize = 0;
        var dspark_total_accepted: usize = 0;
        var dspark_total_steps: usize = 0;
        // NATIVE_DSPARK_EVAL=1: load the DSpark engine but do NOT speculate —
        // measure backbone position-0 top-1 accuracy per decode step instead
        // (DSpark restart milestone: top-1 >= 40% on real decode).
        const dspark_eval = std.c.getenv("NATIVE_DSPARK_EVAL") != null;
        var eval_hits: usize = 0;
        var eval_total: usize = 0;
        var eval_rank_sum: usize = 0;
        if (self.dspark_engine != null or self.dspark != null) {
            dspark_draft_logits_buf = try self.allocator.alloc(f32, DSPARK_MAX_BLOCK * @as(usize, self.config.vocab_size));
        }

        var remaining = decode_count;
        while (remaining > 0) {
            if (current_len >= tokens.len) break;

            // --- Standard single-token decode step (produces anchor logits) ---
            var hidden: [MHC_MULT * DIM]f32 = undefined;
            metal.setTokenId(self.engine, @intCast(tokens[current_len - 1]));
            metal.embed(self.engine, @intCast(tokens[current_len - 1]), &hidden);
            const _t0 = if (_do_decode_time) std.c.mach_absolute_time() else 0;
            try metal.forward(self.engine, hidden[0..], @intCast(start_pos));
            const _t1 = if (_do_decode_time) std.c.mach_absolute_time() else 0;

            var compressed: [DIM]f32 = undefined;
            metal.hyperHeadCompress(
                self.weight_store.weights.hc_head_fn.ptr,
                self.weight_store.weights.hc_head_base.ptr,
                self.weight_store.weights.hc_head_scale.ptr,
                &hidden,
                &compressed,
            );
            try metal.getLogits(self.engine, &compressed, self.logits_buffer.ptr);
            const _t2 = if (_do_decode_time) std.c.mach_absolute_time() else 0;
            if (_do_decode_time) {
                _t_forward_total += _t1 - _t0;
                _t_logits_total += _t2 - _t1;
                _t_decode_n += 1;
            }

            const next_token = sampleFromLogits(self.logits_buffer, self.config.vocab_size, sampler_config);
            tokens[current_len] = next_token;
            current_len += 1;
            start_pos += 1;
            remaining -= 1;

            if (next_token == self.eos_token) {
                std.log.info("native_engine: EOS token generated, stopping at pos={d}", .{start_pos});
                break;
            }

            // --- DSpark backbone position-0 top-1 evaluation ---
            // Backbone prediction from (anchor token fed this step, this step's
            // accumulated main_hidden) vs the token the target actually sampled.
            if (dspark_eval) {
                if (self.dspark_engine) |de| {
                    if (dspark_draft_logits_buf) |draft_buf| {
                        const vocab: usize = @intCast(self.config.vocab_size);
                        const anchor: u32 = tokens[current_len - 2];
                        const n = metal.dsparkForward(de, null, @intCast(anchor), @intCast(start_pos - 1), draft_buf[0..vocab], null);
                        if (n > 0) {
                            var pred: u32 = 0;
                            var best: f32 = -std.math.inf(f32);
                            var actual_rank: usize = 1;
                            const actual_logit = draft_buf[next_token];
                            for (draft_buf[0..vocab], 0..) |l, i| {
                                if (l > best) {
                                    best = l;
                                    pred = @intCast(i);
                                }
                                if (l > actual_logit) actual_rank += 1;
                            }
                            eval_total += 1;
                            if (pred == next_token) eval_hits += 1;
                            eval_rank_sum += actual_rank;
                            if (eval_total % 10 == 0) {
                                std.log.info("[dspark-eval] top1={d}/{d} ({d:.1}%) avg_rank={d:.0} last pred={d} actual={d}(rank {d})", .{
                                    eval_hits,
                                    eval_total,
                                    @as(f64, @floatFromInt(eval_hits)) * 100.0 / @as(f64, @floatFromInt(eval_total)),
                                    @as(f64, @floatFromInt(eval_rank_sum)) / @as(f64, @floatFromInt(eval_total)),
                                    pred,
                                    next_token,
                                    actual_rank,
                                });
                            }
                        }
                    }
                }
            }

            // --- DSpark speculative decoding (propose + verify) ---
            // Priority: use full DSpark engine if available, else fall back to Markov-only
            // (skipped entirely in NATIVE_DSPARK_EVAL mode — no speculation there)
            if (!dspark_eval) {
                if (self.dspark_engine) |de| {
                    if (remaining == 0 or current_len >= tokens.len) continue;

                    const max_draft: usize = @min(5, remaining, tokens.len - current_len);

                    // dspark_forward writes [block_size × vocab] logits
                    if (dspark_draft_logits_buf == null) continue;

                    const vocab: usize = @intCast(self.config.vocab_size);
                    const draft_buf = dspark_draft_logits_buf.?;

                    // Run simplified DSpark forward (embed → norm → lm_head per position)
                    const n_draft_raw = metal.dsparkForward(
                        de,
                        null,
                        @intCast(next_token),
                        @intCast(start_pos),
                        draft_buf[0 .. max_draft * vocab],
                        null,
                    );
                    if (n_draft_raw <= 0) continue;

                    // Use C-side Markov Head which processes ALL positions' logits independently
                    const n_proposed_raw = metal.dsparkMarkovSample(
                        de,
                        draft_buf[0 .. max_draft * vocab],
                        @intCast(next_token),
                        draft_buf[0 .. max_draft * vocab], // corrected in-place
                        dspark_draft_tokens[0..max_draft],
                    );
                    const n_proposed: usize = if (n_proposed_raw > 0) @intCast(n_proposed_raw) else 0;
                    if (n_proposed == 0) continue;

                    // Verify draft tokens against target model.
                    // Feed [anchor(next_token), d0..d_{v-1}] at positions start_pos..start_pos+v.
                    // Output at position start_pos+k predicts the token at start_pos+k+1:
                    //   out[0] (after anchor) ↔ d0, out[k] (after d_{k-1}) ↔ d_k.
                    // (Previously the anchor was never forwarded — drafts were fed starting at
                    //  start_pos, shifting the conditioning by one token → 0% acceptance and
                    //  correction tokens sampled from a corrupted context.)
                    const verify_len: usize = @intCast(n_proposed);
                    const batch_len: usize = verify_len + 1; // anchor + drafts
                    var verify_hidden = try self.allocator.alloc(f32, batch_len * MHC_MULT * DIM);
                    defer self.allocator.free(verify_hidden);
                    var verify_token_ids = try self.allocator.alloc(i32, batch_len);
                    defer self.allocator.free(verify_token_ids);

                    // Slot 0: anchor (next_token) — emitted by the standard step but not yet forwarded
                    verify_token_ids[0] = @intCast(next_token);
                    metal.setTokenId(self.engine, @intCast(next_token));
                    metal.embed(self.engine, @intCast(next_token), verify_hidden[0 .. MHC_MULT * DIM]);
                    for (0..verify_len) |t| {
                        const tok = dspark_draft_tokens[t];
                        verify_token_ids[t + 1] = @intCast(tok);
                        metal.setTokenId(self.engine, @intCast(tok));
                        metal.embed(self.engine, @intCast(tok), verify_hidden[(t + 1) * MHC_MULT * DIM ..][0 .. MHC_MULT * DIM]);
                    }
                    // Disable accumulation during verification (don't corrupt DSpark's main_hidden)
                    metal.setDSparkAccumulate(self.engine, false);
                    try metal.forwardBatch(self.engine, verify_hidden, @intCast(batch_len), @intCast(start_pos), verify_token_ids);

                    // Re-enable accumulation after verification
                    metal.setDSparkAccumulate(self.engine, true);

                    var accepted: usize = 0;
                    for (0..verify_len) |k| {
                        var verify_compressed: [DIM]f32 = undefined;
                        metal.hyperHeadCompress(
                            self.weight_store.weights.hc_head_fn.ptr,
                            self.weight_store.weights.hc_head_base.ptr,
                            self.weight_store.weights.hc_head_scale.ptr,
                            verify_hidden[k * MHC_MULT * DIM ..][0 .. MHC_MULT * DIM],
                            &verify_compressed,
                        );
                        try metal.getLogits(self.engine, &verify_compressed, self.logits_buffer.ptr);
                        const target_token = sampleFromLogits(self.logits_buffer, self.config.vocab_size, sampler_config);

                        if (target_token == dspark_draft_tokens[k]) {
                            tokens[current_len] = dspark_draft_tokens[k];
                            current_len += 1;
                            accepted += 1;
                            remaining -|= 1;
                            if (dspark_draft_tokens[k] == self.eos_token) break;
                        } else {
                            // Reject: correction token, conditioned on the CORRECT context
                            // (anchor + accepted drafts). Its KV is not written yet — it becomes
                            // the next loop iteration's anchor via tokens[current_len-1].
                            tokens[current_len] = target_token;
                            current_len += 1;
                            remaining -|= 1;
                            break;
                        }
                    }
                    // All drafts accepted → bonus token from out[verify_len] (after last draft)
                    if (accepted == verify_len and remaining > 0 and current_len < tokens.len) {
                        var bonus_compressed: [DIM]f32 = undefined;
                        metal.hyperHeadCompress(
                            self.weight_store.weights.hc_head_fn.ptr,
                            self.weight_store.weights.hc_head_base.ptr,
                            self.weight_store.weights.hc_head_scale.ptr,
                            verify_hidden[verify_len * MHC_MULT * DIM ..][0 .. MHC_MULT * DIM],
                            &bonus_compressed,
                        );
                        try metal.getLogits(self.engine, &bonus_compressed, self.logits_buffer.ptr);
                        const bonus_token = sampleFromLogits(self.logits_buffer, self.config.vocab_size, sampler_config);
                        tokens[current_len] = bonus_token;
                        current_len += 1;
                        remaining -|= 1;
                    }
                    // KV state: positions start_pos..start_pos+accepted hold [anchor, d0..d_{a-1}]
                    // — all correct. Discard speculative KV beyond that.
                    start_pos += accepted + 1;
                    dspark_total_drafted += verify_len;
                    dspark_total_accepted += accepted;
                    dspark_total_steps += 1;
                    metal.rollbackKv(self.engine, @intCast(start_pos));
                    if (current_len > 0 and tokens[current_len - 1] == self.eos_token) break;
                } else if (self.dspark) |*ds| {
                    if (remaining == 0 or current_len >= tokens.len) continue;

                    // Propose draft tokens using Markov Head
                    const max_draft = @min(@as(usize, ds.block_size), remaining, tokens.len - current_len);
                    const n_draft = ds.propose(
                        self.logits_buffer[0..@as(usize, self.config.vocab_size)],
                        next_token,
                        dspark_draft_tokens[0..max_draft],
                        dspark_draft_logits_buf.?,
                    );

                    if (n_draft == 0) continue;

                    // Build verification batch: embed all draft tokens and forwardBatch
                    const verify_len = @as(usize, n_draft);
                    var verify_hidden = try self.allocator.alloc(f32, verify_len * MHC_MULT * DIM);
                    defer self.allocator.free(verify_hidden);
                    var verify_token_ids = try self.allocator.alloc(i32, verify_len);
                    defer self.allocator.free(verify_token_ids);

                    for (0..verify_len) |t| {
                        const tok = dspark_draft_tokens[t];
                        verify_token_ids[t] = @intCast(tok);
                        metal.setTokenId(self.engine, @intCast(tok));
                        metal.embed(self.engine, @intCast(tok), verify_hidden[t * MHC_MULT * DIM ..][0 .. MHC_MULT * DIM]);
                    }

                    // forwardBatch processes all draft tokens in one pass,
                    // writing KV cache entries at positions [start_pos .. start_pos + verify_len - 1]
                    metal.setDSparkAccumulate(self.engine, false);
                    try metal.forwardBatch(self.engine, verify_hidden, @intCast(verify_len), @intCast(start_pos), verify_token_ids);
                    metal.setDSparkAccumulate(self.engine, true);

                    // Verify each draft position: get target logits, check if draft matches
                    var accepted: usize = 0;
                    for (0..verify_len) |k| {
                        var verify_compressed: [DIM]f32 = undefined;
                        metal.hyperHeadCompress(
                            self.weight_store.weights.hc_head_fn.ptr,
                            self.weight_store.weights.hc_head_base.ptr,
                            self.weight_store.weights.hc_head_scale.ptr,
                            verify_hidden[k * MHC_MULT * DIM ..][0 .. MHC_MULT * DIM],
                            &verify_compressed,
                        );
                        try metal.getLogits(self.engine, &verify_compressed, self.logits_buffer.ptr);

                        // Greedy verification: accept if target model agrees with draft
                        const target_token = sampleFromLogits(self.logits_buffer, self.config.vocab_size, sampler_config);

                        if (target_token == dspark_draft_tokens[k]) {
                            // Accepted — the draft token matches target
                            tokens[current_len] = dspark_draft_tokens[k];
                            current_len += 1;
                            accepted += 1;
                            remaining -|= 1;

                            if (dspark_draft_tokens[k] == self.eos_token) {
                                std.log.info("native_engine: EOS in draft (accepted pos {d})", .{k});
                                break;
                            }
                        } else {
                            // Rejected — use target's token as the bonus token
                            tokens[current_len] = target_token;
                            current_len += 1;
                            accepted += 1; // bonus token counts as 1 accepted position
                            remaining -|= 1;
                            break;
                        }
                    }

                    // If all draft tokens accepted, we need the bonus token from the last position
                    // (already handled: the last getLogits/sample above produced it in the reject case,
                    //  and if all accepted we continue to next iteration which does a fresh forward)

                    // Advance start_pos by accepted count and rollback KV
                    start_pos += accepted;
                    // Rollback KV cache: positions start_pos..start_pos+(verify_len-accepted) are invalid
                    if (accepted < verify_len) {
                        metal.rollbackKv(self.engine, @intCast(start_pos));
                    }

                    // Check if we hit EOS in the accepted tokens
                    if (current_len > 0 and tokens[current_len - 1] == self.eos_token) {
                        std.log.info("native_engine: EOS token generated (spec), stopping at pos={d}", .{start_pos});
                        break;
                    }
                }
            } // end if (!dspark_eval)
        }

        if (_do_decode_time and _t_decode_n > 0) {
            const fwd_ms = @as(u64, @intFromFloat(@as(f64, @floatFromInt(_t_forward_total)) * 125.0 / 3_000_000.0 / @as(f64, @floatFromInt(_t_decode_n))));
            const log_ms = @as(u64, @intFromFloat(@as(f64, @floatFromInt(_t_logits_total)) * 125.0 / 3_000_000.0 / @as(f64, @floatFromInt(_t_decode_n))));
            std.log.info("[DECODE_TIME] n={d} forward={d}ms logits={d}ms per_tok={d}ms", .{
                _t_decode_n, fwd_ms, log_ms, fwd_ms + log_ms,
            });
        }

        if (eval_total > 0) {
            std.log.info("[dspark-eval] FINAL backbone top-1 = {d}/{d} = {d:.1}% (restart milestone: >=40%)", .{
                eval_hits,
                eval_total,
                @as(f64, @floatFromInt(eval_hits)) * 100.0 / @as(f64, @floatFromInt(eval_total)),
            });
        }

        if (dspark_total_steps > 0) {
            std.log.info("[dspark-stats] steps={d} drafted={d} accepted={d} rate={d:.1}%", .{
                dspark_total_steps,
                dspark_total_drafted,
                dspark_total_accepted,
                if (dspark_total_drafted > 0)
                    @as(f64, @floatFromInt(dspark_total_accepted)) * 100.0 / @as(f64, @floatFromInt(dspark_total_drafted))
                else
                    0.0,
            });
        }

        // Return only generated tokens (skip effective prompt), and drop trailing EOS.
        var result_len = current_len - effective_prompt_len;
        if (result_len > 0 and tokens[current_len - 1] == self.eos_token) {
            result_len -= 1;
        }
        const result = try allocator.alloc(u32, result_len);
        @memcpy(result, tokens[effective_prompt_len .. effective_prompt_len + result_len]);
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

        // Skip BOS (token 0) and EOS (token 1) if present — MLX strips both before generation.
        const has_bos = prompt_tokens.len > 0 and prompt_tokens[0] == BOS_TOKEN;
        const has_eos = prompt_tokens.len > 0 and prompt_tokens[prompt_tokens.len - 1] == self.eos_token;
        const prompt_offset: usize = if (has_bos) 1 else 0;
        const eos_offset: usize = if (has_eos) 1 else 0;
        const effective_prompt_len = if (prompt_tokens.len > prompt_offset + eos_offset)
            prompt_tokens.len - prompt_offset - eos_offset
        else
            0;
        var tokens = try allocator.alloc(u32, effective_prompt_len + max_new_tokens);
        defer allocator.free(tokens);
        if (effective_prompt_len > 0) {
            @memcpy(tokens[0..effective_prompt_len], prompt_tokens[prompt_offset .. prompt_tokens.len - eos_offset]);
        }

        var current_len = effective_prompt_len;
        var start_pos: usize = start_pos_override orelse 0;

        // Reset KV cache
        self.resetKv();

        // Prefill: token-by-token forward over effective prompt tokens only.
        if (start_pos_override == null and effective_prompt_len > 0) {
            for (tokens[0..effective_prompt_len], 0..) |tok, t| {
                var hidden: [MHC_MULT * DIM]f32 = undefined;
                metal.setTokenId(self.engine, @intCast(tok));
                metal.embed(self.engine, @intCast(tok), &hidden);
                try metal.forward(self.engine, hidden[0..], @intCast(start_pos + t));

                if (t == effective_prompt_len - 1) {
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
                    start_pos += effective_prompt_len;
                    tokens[current_len] = next_token;
                    current_len += 1;
                    std.log.info("native_engine: effective_prompt_len={d} first_token={d}", .{ effective_prompt_len, next_token });

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

        // Return only generated tokens (skip effective prompt), and drop trailing EOS.
        var result_len = current_len - effective_prompt_len;
        if (result_len > 0 and tokens[current_len - 1] == self.eos_token) {
            result_len -= 1;
        }
        const result = try allocator.alloc(u32, result_len);
        @memcpy(result, tokens[effective_prompt_len .. effective_prompt_len + result_len]);
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
        std.log.info("sampleFromLogits: max_idx={d} max_val={d:.4} logit[0]={d:.4} logit[304]={d:.4}", .{ max_idx, max_val, logits[0], logits[304] });
        return @intCast(max_idx);
    }
};
