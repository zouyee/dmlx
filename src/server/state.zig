/// Server state and model loading.
const std = @import("std");
const root = @import("../root.zig");
const c = @import("mlx").c;
const ops = @import("mlx").ops;
const memory_mod = @import("../memory.zig");
const model_registry_mod = @import("../model_registry.zig");
const generation_mod = @import("../generation.zig");
const model_pool_mod = @import("../model_pool.zig");
const kvcache = @import("../kvcache.zig");
const scheduler_mod = @import("../scheduler.zig");
const prompt_cache_mod = @import("../prompt_cache.zig");
const prefix_disk_mod = @import("../kvcache/prefix_disk.zig");
const dsv4_mod = @import("../models/deepseek_v4.zig");
const dsv4_loader = @import("../models/deepseek_v4_loader.zig");
const engine = @import("../engine/root.zig");
const config_mod = @import("config.zig");
const utils_mod = @import("utils.zig");
const detectArchitecture = utils_mod.detectArchitecture;

const EagerContext = ops.EagerContext;
const ModelVTable = generation_mod.ModelVTable;
const ModelPool = model_pool_mod.ModelPool;
const ServerConfig = config_mod.ServerConfig;

// ------------------------------------------------------------------
// Model state (loaded via ModelRegistry at startup)
// ------------------------------------------------------------------

pub const ServerState = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    vtable: ModelVTable,
    tokenizer_strategy: root.tokenizer.TokenizerStrategy,
    tokenizer_backend: *root.tokenizer.BpeTokenizer,
    chat_template: root.tokenizer.ChatTemplate,
    model_name: []const u8,
    caches: []kvcache.KVCacheStrategy,
    model_pool: ?ModelPool,
    block_manager: ?scheduler_mod.BlockManager,
    scheduler: ?scheduler_mod.Scheduler,
    speculative_ngram: ?usize,
    prompt_cache_file: ?[]const u8,
    prefix_disk_cache: ?prefix_disk_mod.PrefixDiskCache,
    running: bool,
    dsv4_model: ?*dsv4_mod.DSV4Model = null,

    // Server V2: engine loop and request queue
    request_queue: engine.RequestQueue,
    engine_loop: ?engine.EngineLoop,
    engine_running: std.atomic.Value(bool),
    active_requests: std.atomic.Value(u32),
    next_request_id: std.atomic.Value(u64),

    pub fn deinit(self: *ServerState) void {
        self.running = false;
        self.engine_running.store(false, .release);
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
        self.allocator.destroy(self.tokenizer_backend);
        self.allocator.free(self.model_name);
        for (self.caches) |cache_item| {
            cache_item.deinit(self.allocator);
        }
        self.allocator.free(self.caches);
        if (self.model_pool) |*pool| {
            pool.deinit();
        }
        const cpu_stream = c.c.mlx_default_cpu_stream_new();
        _ = c.c.mlx_set_default_stream(cpu_stream);
        _ = c.c.mlx_stream_free(cpu_stream);
        self.ctx.deinit();
        // stream freed by ctx.deinit (initWithStream shares the handle).
    }
};

pub fn loadModel(allocator: std.mem.Allocator, io: std.Io, config: ServerConfig) !ServerState {
    const stream = c.c.mlx_default_gpu_stream_new();
    _ = c.c.mlx_set_default_stream(stream);
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });

    // 1. Read config.json
    const config_path = try std.fs.path.join(allocator, &[_][]const u8{ config.model_path, "config.json" });
    defer allocator.free(config_path);

    const config_content = try std.Io.Dir.cwd().readFileAlloc(io, config_path, allocator, .limited(1024 * 1024));
    defer allocator.free(config_content);

    // 2. Detect architecture and load via ModelRegistry
    const arch_name = detectArchitecture(config_content);
    std.log.info("Detected architecture: {s}", .{arch_name});
    const model_name = try allocator.dupe(u8, arch_name);

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

    // Extract DSV4 model for direct generate() — matches CLI path.
    // Extract dsv4_model for direct native generate path.
    var dsv4_model: ?*dsv4_mod.DSV4Model = null;
    if (std.mem.eql(u8, arch_name, "DeepseekV4ForCausalLM")) {
        const adapter: *model_registry_mod.DeepseekV4VTableAdapter = @ptrCast(@alignCast(vtable.ptr));
        dsv4_model = adapter.model;
    }

    // 3. Load tokenizer
    // Load tokenizer.
    const tokenizer_path = try std.fs.path.join(allocator, &[_][]const u8{ config.model_path, "tokenizer.json" });
    defer allocator.free(tokenizer_path);

    const tokenizer_backend = try allocator.create(root.tokenizer.BpeTokenizer);
    errdefer allocator.destroy(tokenizer_backend);
    tokenizer_backend.* = root.tokenizer.BpeTokenizer.init(allocator);
    try tokenizer_backend.loadFromFile(io, tokenizer_path);

    // 4. Detect chat template based on architecture
    // Create chat template.
    const chat_template = if (std.mem.eql(u8, arch_name, "DeepseekV4ForCausalLM"))
        root.tokenizer.ChatTemplate.initDeepSeek(allocator)
    else
        root.tokenizer.ChatTemplate.initDeepSeek(allocator); // Default template

    // 5. Create KV caches using model config + auto max_kv_size
    // Create KV caches.
    const mc = vtable.config;
    _ = memory_mod.autoMaxKvSize;
    const effective_max_seq = 8192;

    var caches: []kvcache.KVCacheStrategy = undefined;

    // DeepSeek V4: use specialized heterogeneous caches (DeepseekV4Cache / RotatingWithWindow)
    // that match the CLI path. StandardKVCache buffers are too large and cause GPU stalls.
    if (std.mem.eql(u8, arch_name, "DeepseekV4ForCausalLM")) {
        // Parse DSV4 config and create heterogeneous caches.
        var dsv4_config = try dsv4_loader.parseDSV4Config(allocator, config_content);
        defer {
            var cfg = dsv4_config;
            cfg.deinitClone(allocator);
        }
        // Create V4 caches via makeV4Caches.
        caches = try dsv4_loader.makeV4Caches(allocator, &dsv4_config, stream);
    } else {
        caches = try allocator.alloc(kvcache.KVCacheStrategy, mc.num_layers);
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

    var server_state = ServerState{
        .allocator = allocator,
        .io = io,
        .ctx = ctx,
        .stream = stream,
        .vtable = vtable,
        .tokenizer_strategy = undefined,
        .tokenizer_backend = tokenizer_backend,
        .model_name = model_name,
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
        .dsv4_model = dsv4_model,
        .running = true,
        .request_queue = undefined, // initialized in start()
        .engine_loop = null,
        .engine_running = std.atomic.Value(bool).init(false),
        .active_requests = std.atomic.Value(u32).init(0),
        .next_request_id = std.atomic.Value(u64).init(0),
    };
    server_state.tokenizer_strategy = server_state.tokenizer_backend.asStrategy();
    return server_state;
}

// ------------------------------------------------------------------
// HTTP request handling
// ------------------------------------------------------------------
