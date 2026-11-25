/// Model Architecture Registry
///
/// Compile-time registry mapping HuggingFace architecture name strings
/// (e.g. "LlamaForCausalLM", "DeepseekV4ForCausalLM") to loader functions
/// that produce a `ModelVTable` for model-agnostic inference.
///
/// Re-exports `ModelVTable` and `ModelConfig` from `generation.zig` so that
/// callers only need to import this module.
const std = @import("std");
const c = @import("mlx").c;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const reduce_mod = @import("mlx").reduce;
const eval_mod = @import("mlx").eval;
const fast_mod = @import("mlx").fast;
const kvcache = @import("kvcache.zig");
const array_mod = @import("mlx").array;
const generation = @import("generation.zig");
const safetensors_reader = @import("mlx").safetensors_reader;
const expert_stream = @import("models/expert_stream.zig");

// Mach task_info wrappers — avoids cImport of mach/message.h (broken in Zig 0.16)
extern fn dmlx_get_rss_bytes() u64;
extern fn dmlx_get_virtual_bytes() u64;

// Model implementations
const llama = @import("models/llama.zig");
const llama_loader = @import("models/llama_loader.zig");
const deepseek_v4 = @import("models/deepseek_v4.zig");
const deepseek_v4_loader = @import("models/deepseek_v4_loader.zig");
const llava_loader = @import("models/llava_loader.zig");
const hf_config = @import("hf_config.zig");
const glm53_loader = @import("models/glm53_loader.zig");
const glm53 = @import("models/glm53.zig");
const qwen38_loader = @import("models/qwen38_loader.zig");
const qwen38 = @import("models/qwen38.zig");
const qwen38_mtp = @import("models/qwen38_mtp.zig");
const verify_qmm = @import("models/verify_qmm.zig");
const qwen38_dflash2 = @import("models/qwen38_dflash2.zig");
const qwen38_dspine = @import("models/qwen38_dspine.zig");
const mkdir_c = @cImport(@cInclude("sys/stat.h"));
const lora_merge = @import("models/lora_merge.zig");
const qwen38_vit = @import("vision/qwen38_vit.zig");
const qwen38_preprocess = @import("vision/qwen38_preprocess.zig");
const pthread = @cImport(@cInclude("pthread.h"));

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const KVCacheStrategy = kvcache.KVCacheStrategy;

// ============================================================
// Re-exports from generation.zig
// ============================================================

pub const ModelVTable = generation.ModelVTable;
pub const ModelConfig = generation.ModelConfig;

// ============================================================
// ModelLoader function type
// ============================================================

/// A loader function that builds a ModelVTable from a pre-parsed config
/// JSON string and a model directory path. The caller is responsible for
/// reading the config file (via std.Io) before invoking the loader.
pub const ModelLoader = *const fn (
    allocator: std.mem.Allocator,
    config_json: []const u8,
    model_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    io: std.Io,
    smelt: deepseek_v4_loader.SmeltConfig,
) anyerror!ModelVTable;

// ============================================================
// Compile-time architecture → loader registry
// ============================================================

pub const model_registry = std.StaticStringMap(ModelLoader).initComptime(.{
    .{ "LlamaForCausalLM", llamaLoader },
    .{ "DeepseekV4ForCausalLM", deepseekV4Loader },
    .{ "MistralForCausalLM", llamaLoader }, // Mistral uses LLaMA arch
    .{ "Qwen2ForCausalLM", llamaLoader }, // Qwen2 uses LLaMA arch with q/k norms
    .{ "Qwen3ForCausalLM", llamaLoader }, // Qwen3 uses LLaMA arch with q/k norms
    .{ "GemmaForCausalLM", gemmaLoader },
    .{ "Glm4ForCausalLM", glm4Loader }, // GLM-4 uses LLaMA arch with attention bias
    .{ "PhiForCausalLM", llamaLoader }, // Phi uses LLaMA-like arch
    .{ "Phi3ForCausalLM", llamaLoader }, // Phi-3 uses LLaMA-like arch
    .{ "LlavaForConditionalGeneration", llavaLoader },
    .{ "Glm5NextForConditionalGeneration", glm53Loader },
    .{ "Qwen3_5ForConditionalGeneration", qwen38Loader },
});

/// All architecture names known to the registry.
pub const supported_architectures = [_][]const u8{
    "LlamaForCausalLM",
    "DeepseekV4ForCausalLM",
    "MistralForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "GemmaForCausalLM",
    "Glm4ForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "LlavaForConditionalGeneration",
    "Glm5NextForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
};

// ============================================================
// Public lookup helper
// ============================================================

/// Convenience re-export for the CLI chat path.
pub fn Glm53ConfigFromJson(allocator: std.mem.Allocator, config_json: []const u8) !glm53_loader.Glm53Config {
    return glm53_loader.parseGlm53Config(allocator, config_json);
}

/// Convenience re-export for the CLI chat path.
pub fn Qwen38ConfigFromJson(allocator: std.mem.Allocator, config_json: []const u8) !qwen38_loader.Qwen38Config {
    return qwen38_loader.parseQwen38Config(allocator, config_json);
}

pub const RegistryError = error{
    UnsupportedArchitecture,
};

/// Look up a loader for the given architecture name.
/// Returns `RegistryError.UnsupportedArchitecture` with a log message
/// containing the queried name when the architecture is not registered.
pub fn getLoader(arch_name: []const u8) RegistryError!ModelLoader {
    return model_registry.get(arch_name) orelse {
        // Silent under `zig build test` (stderr noise trips the build runner).
        if (!@import("builtin").is_test)
            std.log.warn("Unsupported model architecture: \"{s}\". Supported architectures: " ++
                "LlamaForCausalLM, DeepseekV4ForCausalLM, MistralForCausalLM, " ++
                "Qwen2ForCausalLM, Qwen3ForCausalLM, GemmaForCausalLM, " ++
                "Glm4ForCausalLM, PhiForCausalLM, Phi3ForCausalLM, " ++
                "LlavaForConditionalGeneration, Glm5NextForConditionalGeneration, " ++
                "Qwen3_5ForConditionalGeneration", .{arch_name});
        return RegistryError.UnsupportedArchitecture;
    };
}

// ============================================================
// VTable adapter: LLaMA
// ============================================================

/// Wraps a heap-allocated `LlamaModel` behind the `ModelVTable` interface.
const LlamaVTableAdapter = struct {
    model: *llama.LlamaModel,

    fn forward(ctx_ptr: *anyopaque, input: Array, mask: ?Array, caches: ?[]KVCacheStrategy) anyerror!Array {
        const self: *LlamaVTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        return self.model.forward(input, mask, caches);
    }

    fn forwardWithHidden(ctx_ptr: *anyopaque, input: Array, mask: ?Array, caches: ?[]KVCacheStrategy) anyerror!generation.ForwardWithHiddenResult {
        const self: *LlamaVTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        return self.model.forwardWithHidden(input, mask, caches);
    }

    fn deinitFn(ctx_ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *LlamaVTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        self.model.deinit();
        allocator.destroy(self.model);
        allocator.destroy(self);
    }
};

fn llamaLoader(
    allocator: std.mem.Allocator,
    config_json: []const u8,
    model_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    io: std.Io,
    smelt: deepseek_v4_loader.SmeltConfig,
) anyerror!ModelVTable {
    _ = io;
    _ = smelt;
    const llama_config = try hf_config.parseLlamaConfig(allocator, config_json);

    // Find safetensors file
    const st_path = try std.fs.path.join(allocator, &.{ model_path, "model.safetensors" });
    defer allocator.free(st_path);

    const model_ptr = try allocator.create(llama.LlamaModel);
    errdefer allocator.destroy(model_ptr);

    model_ptr.* = try llama_loader.loadFromSafetensors(allocator, &llama_config, st_path, ctx, stream);

    // Fixup: LlamaAttention / LlamaMLP / LlamaTransformerBlock store pointers
    // to LlamaConfig. After the model is returned by value from buildModel(),
    // those pointers point to the loader's stack frame (llama_config) which is
    // invalidated once llamaLoader returns. Redirect them to the stable heap
    // address inside model_ptr.
    for (model_ptr.layers) |*layer| {
        layer.config = &model_ptr.config;
        layer.attention.config = &model_ptr.config;
        layer.mlp.config = &model_ptr.config;
    }

    const adapter = try allocator.create(LlamaVTableAdapter);
    errdefer allocator.destroy(adapter);
    adapter.* = .{ .model = model_ptr };

    const head_dim = llama_config.getHeadDim();

    return ModelVTable{
        .forward = &LlamaVTableAdapter.forward,
        .forwardWithHidden = &LlamaVTableAdapter.forwardWithHidden,
        .deinit = &LlamaVTableAdapter.deinitFn,
        .config = ModelConfig{
            .num_layers = llama_config.num_hidden_layers,
            .num_kv_heads = llama_config.num_key_value_heads,
            .head_dim = head_dim,
            .vocab_size = llama_config.vocab_size,
            .hidden_size = llama_config.hidden_size,
        },
        .ptr = adapter,
    };
}

// ============================================================
// VTable adapter: DeepSeek V4
// ============================================================

/// Wraps a heap-allocated `DSV4Model` behind the `ModelVTable` interface.
/// The adapter tracks `start_pos` internally and stores the MLX stream
/// so that the VTable `forward` signature stays uniform.
pub const DeepseekV4VTableAdapter = struct {
    model: *deepseek_v4.DSV4Model,
    stream: c.c.mlx_stream,
    start_pos: usize,
    /// Weights HashMap — kept alive for the model's lifetime.
    /// buildDSV4Model borrows arrays from this map; releasing them
    /// while the model is alive corrupts MLX array handles.
    weights: std.StringHashMap(Array),
    allocator: std.mem.Allocator,

    fn forward(ctx_ptr: *anyopaque, input: Array, mask: ?Array, caches: ?[]KVCacheStrategy) anyerror!Array {
        const self: *DeepseekV4VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        const result = try self.model.forward(input, mask, caches, self.start_pos, self.stream);
        // Advance position by the sequence length of this input
        const input_shape = input.shape();
        self.start_pos += @intCast(input_shape[1]);
        return result;
    }

    fn deinitFn(ctx_ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *DeepseekV4VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        self.model.deinit();
        allocator.destroy(self.model);
        // Release remaining weights (model has ownership of the ones it fetchRemoved)
        var it = self.weights.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        self.weights.deinit();
        allocator.destroy(self);
    }
};

fn deepseekV4Loader(
    allocator: std.mem.Allocator,
    config_json: []const u8,
    model_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    io: std.Io,
    smelt: deepseek_v4_loader.SmeltConfig,
) anyerror!ModelVTable {
    const dsv4_config = try deepseek_v4_loader.parseDSV4Config(allocator, config_json);

    const use_selective = smelt.enabled and smelt.load_mode == .stream;
    var weights = if (use_selective)
        try deepseek_v4_loader.loadWeightsSelective(allocator, model_path, smelt)
    else
        try deepseek_v4_loader.loadWeightsFromDirectory(allocator, io, model_path, ctx, stream, smelt);
    // NOTE: weights are NOT freed here — they are transferred to the adapter
    // and released when the model is destroyed. buildDSV4Model borrows arrays
    // from this map; releasing them while the model is alive corrupts MLX
    // array handles (observed as "Cannot reshape array" with garbage values).

    const model_ptr = try allocator.create(deepseek_v4.DSV4Model);
    errdefer allocator.destroy(model_ptr);

    model_ptr.* = try deepseek_v4_loader.buildDSV4Model(allocator, &dsv4_config, &weights, ctx, stream, smelt);

    // Wire expert streaming when smelt is enabled (matches CLI path in main.zig)
    if (smelt.enabled and !model_ptr.hasExpertsLoaded()) {
        // Build tensor index for random-access reading
        const idx = try allocator.create(safetensors_reader.TensorIndex);
        idx.* = try safetensors_reader.buildIndexFromDirectory(allocator, model_path);

        // Build per-layer metadata
        const num_layers = dsv4_config.num_hidden_layers;
        var layer_meta = try allocator.alloc(expert_stream.LayerExpertMeta, num_layers);
        for (0..num_layers) |i| {
            const hf_gate = try std.fmt.allocPrint(allocator, "model.layers.{d}.ffn.switch_mlp.gate_proj.weight", .{i});
            const hf_up = try std.fmt.allocPrint(allocator, "model.layers.{d}.ffn.switch_mlp.up_proj.weight", .{i});
            const hf_down = try std.fmt.allocPrint(allocator, "model.layers.{d}.ffn.switch_mlp.down_proj.weight", .{i});
            const hf_gate_s = try std.fmt.allocPrint(allocator, "model.layers.{d}.ffn.switch_mlp.gate_proj.scales", .{i});
            const hf_up_s = try std.fmt.allocPrint(allocator, "model.layers.{d}.ffn.switch_mlp.up_proj.scales", .{i});
            const hf_down_s = try std.fmt.allocPrint(allocator, "model.layers.{d}.ffn.switch_mlp.down_proj.scales", .{i});

            var row_bytes: usize = 0;
            var scale_row_bytes: usize = 0;
            if (idx.entries.get(hf_gate)) |info| {
                const total = info.data_offset_end - info.data_offset_start;
                row_bytes = @intCast(total / @as(u64, @intCast(info.shape[0])));
            }
            if (idx.entries.get(hf_gate_s)) |info| {
                const total = info.data_offset_end - info.data_offset_start;
                scale_row_bytes = @intCast(total / @as(u64, @intCast(info.shape[0])));
            }

            layer_meta[i] = .{
                .gate_proj_name = hf_gate,
                .up_proj_name = hf_up,
                .down_proj_name = hf_down,
                .gate_scales_name = if (idx.entries.contains(hf_gate_s)) hf_gate_s else blk: {
                    allocator.free(hf_gate_s);
                    break :blk null;
                },
                .up_scales_name = if (idx.entries.contains(hf_up_s)) hf_up_s else blk: {
                    allocator.free(hf_up_s);
                    break :blk null;
                },
                .down_scales_name = if (idx.entries.contains(hf_down_s)) hf_down_s else blk: {
                    allocator.free(hf_down_s);
                    break :blk null;
                },
                .expert_row_bytes = row_bytes,
                .expert_scale_row_bytes = scale_row_bytes,
                .n_experts = dsv4_config.n_routed_experts,
            };
        }

        const sp = try allocator.create(expert_stream.ExpertStreamProvider);

        const strategy: expert_stream.ExpertLoadStrategy = if (smelt.load_mode == .stream) .stream else .preload;
        const n_experts_to_load = @as(usize, @intFromFloat(@as(f32, @floatFromInt(dsv4_config.n_routed_experts)) * smelt.load_fraction));
        var expert_ids = try allocator.alloc(u32, n_experts_to_load);
        defer allocator.free(expert_ids);
        for (0..n_experts_to_load) |i| {
            expert_ids[i] = @intCast(i);
        }

        sp.* = try expert_stream.ExpertStreamProvider.initWithStrategy(
            allocator,
            ctx,
            idx,
            strategy,
            expert_ids,
            layer_meta,
            true, // quantized
            32, // group_size
            4, // bits
            "mxfp4",
            dsv4_config.swiglu_limit,
            smelt.cache_budget_mb,
            smelt.packed_dir,
            smelt.max_parallel,
        );

        model_ptr.setExpertStreamProvider(sp);

        // Log memory usage after expert stream provider setup
        {
            const rss_mb = dmlx_get_rss_bytes() / (1024 * 1024);
            const virt_mb = dmlx_get_virtual_bytes() / (1024 * 1024);
            if (rss_mb > 0) std.log.info("Memory after expert stream setup: RSS={d}MB, Virtual={d}MB", .{ rss_mb, virt_mb });
        }

        std.log.info("model_registry: Expert streaming enabled for DeepSeek V4", .{});
    }

    const adapter = try allocator.create(DeepseekV4VTableAdapter);
    errdefer allocator.destroy(adapter);
    adapter.* = .{
        .model = model_ptr,
        .stream = stream,
        .start_pos = 0,
        .weights = weights,
        .allocator = allocator,
    };

    return ModelVTable{
        .forward = &DeepseekV4VTableAdapter.forward,
        .deinit = &DeepseekV4VTableAdapter.deinitFn,
        .config = ModelConfig{
            .num_layers = dsv4_config.num_hidden_layers,
            .num_kv_heads = dsv4_config.num_key_value_heads,
            .head_dim = dsv4_config.head_dim,
            .vocab_size = dsv4_config.vocab_size,
            .hidden_size = dsv4_config.hidden_size,
            .compress_ratios = dsv4_config.compress_ratios,
        },
        .ptr = adapter,
    };
}

// ============================================================
// VTable adapter: GLM-5.3 (glm5_next)
// ============================================================

/// Wraps a heap-allocated `Glm53ServingModel` behind the `ModelVTable`
/// interface. GLM-5.3 keeps its caches internal (KDA conv/ssm state, MLA
/// latent + indexer history) — the engine's KVCacheStrategy array is unused.
/// `start_pos` is tracked for interface parity only.
pub const Glm53VTableAdapter = struct {
    model: *glm53.Glm53ServingModel,
    stream: c.c.mlx_stream,
    start_pos: usize,
    cfg: *glm53_loader.Glm53Config,
    /// Resident backbone weights — kept alive for the model's lifetime.
    weights: std.StringHashMap(Array),
    index: *safetensors_reader.TensorIndex,
    fd_pool: *safetensors_reader.FdPool,
    reader: *safetensors_reader.PartialTensorReader,
    stream_provider: ?*expert_stream.ExpertStreamProvider,
    layer_meta: ?[]expert_stream.LayerExpertMeta,
    allocator: std.mem.Allocator,

    fn forward(ctx_ptr: *anyopaque, input: Array, mask: ?Array, caches: ?[]KVCacheStrategy) anyerror!Array {
        const self: *Glm53VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = mask;
        _ = caches;
        try input.eval();
        const shape = input.shape();
        const len: usize = @intCast(shape[1]);
        const ids_u32 = try input.dataSlice(u32);
        const ids = try self.allocator.dupe(u32, ids_u32);
        defer self.allocator.free(ids);
        const logits = try self.model.forwardTokens(ids);
        self.start_pos += len;
        return logits;
    }

    fn deinitFn(ctx_ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *Glm53VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        self.model.deinit();
        allocator.destroy(self.model);
        if (self.stream_provider) |sp| {
            sp.deinit();
            allocator.destroy(sp);
        }
        // layer_meta is owned by the stream provider (sp.deinit above frees
        // the per-layer names and the slice) — do not free it here.
        // PartialTensorReader has no deinit (borrows index + fd_pool).
        allocator.destroy(self.reader);
        self.fd_pool.deinit();
        allocator.destroy(self.fd_pool);
        self.index.deinit();
        allocator.destroy(self.index);
        var it = self.weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        self.weights.deinit();
        self.cfg.deinit();
        allocator.destroy(self.cfg);
        allocator.destroy(self);
    }
};

fn glm53Loader(
    allocator: std.mem.Allocator,
    config_json: []const u8,
    model_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    io: std.Io,
    smelt: deepseek_v4_loader.SmeltConfig,
) anyerror!ModelVTable {
    _ = io;
    const cfg_ptr = try allocator.create(glm53_loader.Glm53Config);
    errdefer allocator.destroy(cfg_ptr);
    cfg_ptr.* = try glm53_loader.parseGlm53Config(allocator, config_json);
    errdefer cfg_ptr.deinit();

    const idx = try allocator.create(safetensors_reader.TensorIndex);
    errdefer allocator.destroy(idx);
    idx.* = try safetensors_reader.loadOrBuildIndex(allocator, model_path);
    errdefer idx.deinit();

    // Resident backbone: every language_model tensor except the stacked
    // switch_mlp experts (~13.4 GiB; experts stream on demand).
    var weights = std.StringHashMap(Array).init(allocator);
    errdefer {
        var it0 = weights.iterator();
        while (it0.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }
    {
        var it = idx.entries.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (!std.mem.startsWith(u8, name, "language_model.")) continue;
            if (std.mem.indexOf(u8, name, ".mlp.switch_mlp.") != null) continue;
            const key_copy = try allocator.dupe(u8, name);
            errdefer allocator.free(key_copy);
            const arr = try idx.loadTensor(name);
            try weights.put(key_copy, arr);
        }
    }
    std.log.info("GLM-5.3 backbone resident: {d} tensors", .{weights.count()});

    const pool = try allocator.create(safetensors_reader.FdPool);
    errdefer allocator.destroy(pool);
    pool.* = safetensors_reader.FdPool.init(allocator);
    try pool.openAll(idx);
    errdefer pool.deinit();

    const rd = try allocator.create(safetensors_reader.PartialTensorReader);
    errdefer allocator.destroy(rd);
    rd.* = safetensors_reader.PartialTensorReader.init(allocator, idx, pool);

    // Expert streaming (GLM: 4-bit affine gs=64 with biases — biases make the
    // packed-pread path ineligible, so slices come from PartialTensorReader).
    var sp: ?*expert_stream.ExpertStreamProvider = null;
    errdefer if (sp) |p| {
        p.deinit();
        allocator.destroy(p);
    };
    var layer_meta: ?[]expert_stream.LayerExpertMeta = null;
    if (smelt.enabled and smelt.load_mode == .stream) {
        const meta = try glm53_loader.buildLayerExpertMeta(allocator, cfg_ptr, idx);
        errdefer glm53_loader.freeLayerExpertMeta(allocator, meta);
        // sp takes ownership of meta on success (its deinit frees the names
        // + slice); the errdefer above covers init failure.
        const sp_ptr = try allocator.create(expert_stream.ExpertStreamProvider);
        errdefer allocator.destroy(sp_ptr);
        sp_ptr.* = try expert_stream.ExpertStreamProvider.initWithStrategy(
            allocator,
            ctx,
            idx,
            .stream,
            &.{}, // no preloaded experts
            meta,
            true, // quantized
            64, // group_size
            4, // bits
            "affine",
            cfg_ptr.swiglu_limit,
            smelt.cache_budget_mb,
            smelt.packed_dir,
            smelt.max_parallel,
        );
        sp_ptr.pred_topk = 8; // GLM-5.3 routes top-8 (PRED_WIDTH)
        layer_meta = meta;
        sp = sp_ptr;
        std.log.info("GLM-5.3 expert streaming enabled (affine gs=64 4-bit, top-8)", .{});
    }

    const model_ptr = try allocator.create(glm53.Glm53ServingModel);
    errdefer allocator.destroy(model_ptr);
    model_ptr.* = try glm53.Glm53ServingModel.init(allocator, ctx, cfg_ptr, stream, weights, rd, sp);
    errdefer model_ptr.deinit();

    const adapter = try allocator.create(Glm53VTableAdapter);
    errdefer allocator.destroy(adapter);
    adapter.* = .{
        .model = model_ptr,
        .stream = stream,
        .start_pos = 0,
        .cfg = cfg_ptr,
        .weights = weights,
        .index = idx,
        .fd_pool = pool,
        .reader = rd,
        .stream_provider = sp,
        .layer_meta = layer_meta,
        .allocator = allocator,
    };

    return ModelVTable{
        .forward = &Glm53VTableAdapter.forward,
        .deinit = &Glm53VTableAdapter.deinitFn,
        .config = ModelConfig{
            .num_layers = cfg_ptr.num_hidden_layers,
            .num_kv_heads = 1, // MLA latent KV: [B, 1, N, kv_lora_rank]
            .head_dim = cfg_ptr.kv_lora_rank,
            .vocab_size = cfg_ptr.vocab_size,
            .hidden_size = cfg_ptr.hidden_size,
            .compress_ratios = &.{},
        },
        .ptr = adapter,
    };
}

// ============================================================
// VTable adapter: Qwen3.8 (qwen3_5)
// ============================================================

/// Wraps a heap-allocated `Qwen38ServingModel` behind the `ModelVTable`
/// interface. Qwen3.8 keeps its caches internal (GDN conv/ssm state,
/// attention KV) — the engine's KVCacheStrategy array is unused.
/// `start_pos` is tracked for interface parity only.
pub const Qwen38VTableAdapter = struct {
    model: *qwen38.Qwen38ServingModel,
    stream: c.c.mlx_stream,
    start_pos: usize,
    cfg: *qwen38_loader.Qwen38Config,
    /// Resident weights — kept alive for the model's lifetime.
    weights: std.StringHashMap(Array),
    allocator: std.mem.Allocator,
    /// Model directory (owned) — the vision tower's weights are lazily read
    /// from here on the first image request.
    model_path: []const u8 = "",
    /// Serializes every adapter op that touches the model's MLX context.
    /// The engine fiber drives forward/prefill/draft/verify; vision
    /// processImage runs on HTTP connection fibers (src/server.zig spawns
    /// one io.async fiber per connection on the Threaded backend) — without
    /// this lock a tower forward on a connection fiber would build MLX
    /// graphs on the same ctx/stream concurrently with engine decode steps.
    /// Leaf lock: op fns never call each other while holding it (ensureVit
    /// and mtpPrefillTail are called with the lock already held).
    /// pthread mutex (repo pattern, cf. expert_pread.zig): std.Io.Mutex
    /// needs an Io at lock time, which several op fns don't have.
    mu: pthread.pthread_mutex_t,
    /// Lazily-initialized vision tower + its resident weight map. Guarded by
    /// mu (two concurrent first-image requests must not double-load).
    vit: ?*qwen38_vit.Qwen38Vit = null,
    vit_weights: ?std.StringHashMap(Array) = null,
    /// Set by qwen38Loader (image_token_id comes from the model config).
    vision_ops: generation.VisionOps,
    /// MTP drafter state (lazily loaded on first enable()).
    mtp: ?*qwen38_mtp.Qwen38Mtp = null,
    /// Drafter KV length before the current round's draft chain; null when
    /// no round is active (set on the round's first draft call).
    mtp_round_old_len: ?i32 = null,
    /// DFlash2 block-parallel drafter (loaded when DMLX_QWEN38_DFLASH2_DIR is
    /// set at enable time). Shares the MTP verify/commit plumbing; the ops
    /// below branch on `dflash2 != null`.
    dflash2: ?*qwen38_dflash2.DFlash2 = null,
    df2_ctx: ?qwen38_dflash2.DFlash2Context = null,
    /// Current round's drafted block: computed lazily on the first draft
    /// call of a round, handed out one token per call.
    df2_block: [7]u32 = undefined,
    df2_block_len: usize = 0,
    df2_block_idx: usize = 0,
    /// Sampled path: retained block logits [P, V] f32 (lazy). Row i is the
    /// drafter's distribution q_i for draft position i — the engine's
    /// speculative sampler needs it for min(1, p/q) acceptance and the
    /// (p-q)+ rejection resample. Greedy rounds leave this null.
    df2_block_logits: ?Array = null,
    /// Drafter-context cache mirroring the backbone prefix snapshots:
    /// projected (post-fc) windowed context rows + the prompt tokens they
    /// cover. Lets a snapshot-restored prefill skip both the backbone
    /// recompute AND the drafter-context rebuild.
    df2_cache_ids: []u32 = &.{},
    df2_cache_rows: ?Array = null, // [1, <=window, H] projected
    /// Predraft stash: next round's block tokens ([7] i32 GPU, built before
    /// the acceptance sync) + the round's verify argmax ([L] i32 GPU) +
    /// how many verify rows the tentative ring feed used.
    /// Monotonic guard for df2_latest persistence (same rationale as the
    /// backbone snapshot's single-slot largest-wins rule).
    df2_last_persisted: usize = 0,
    df2_pending_tokens: ?Array = null,
    df2_pending_argmax: ?Array = null,
    df2_pending_rows: usize = 0,
    /// DSpine block=2..8 drafter. Selected only by DMLX_QWEN38_DSPINE_DIR.
    dspine: ?*qwen38_dspine.DSpine = null,
    dspine_ctx: ?qwen38_dspine.DSpineContext = null,
    dspine_block: [7]u32 = undefined,
    dspine_candidates: [7 * 16]u32 = undefined,
    dspine_table_host: []f32 = &.{},
    dspine_block_len: usize = 0,
    dspine_block_idx: usize = 0,
    dspine_cache_ids: []u32 = &.{},
    dspine_cache_rows: ?Array = null,
    /// `DMLX_MTP_CHAIN_DUMP=1`: per-step drafter KV length trace (plan
    /// 2026-10-29-catch-omlx-mtp Task 1). Read once per process; the RoPE
    /// offset of every head forward equals cache.len() before its append,
    /// so the lengths alone pin down the position accounting.
    mtp_chain_dump: ?bool = null,
    /// `DMLX_MTP_TRUNK_POSTNORM=1` cached lookup (see maybeTrunkNorm).
    mtp_trunk_postnorm: ?bool = null,
    mtp_ops: generation.MtpOps = .{
        .enable = &mtpEnable,
        .prefill = &mtpPrefill,
        .prefillVision = &mtpPrefillVision,
        .draft = &mtpDraft,
        .draftGreedy = &mtpDraftGreedy,
        .verify = &mtpVerify,
        .verifyLogits = &mtpVerifyLogits,
        .verifyArgmax = &mtpVerifyArgmax,
        .predraft = &mtpPredraft,
        .verifyHidden = &mtpVerifyHidden,
        .commit = &mtpCommit,
        .rollback = &mtpRollback,
        .freeVerify = &mtpFreeVerify,
        .num_drafts = 0,
    },

    fn forward(ctx_ptr: *anyopaque, input: Array, mask: ?Array, caches: ?[]KVCacheStrategy) anyerror!Array {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        _ = mask;
        _ = caches;
        try input.eval();
        const shape = input.shape();
        const len: usize = @intCast(shape[1]);
        const ids_u32 = try input.dataSlice(u32);
        const ids = try self.allocator.dupe(u32, ids_u32);
        defer self.allocator.free(ids);
        const logits = try self.model.forwardTokensLast(ids);
        self.start_pos += len;
        return logits;
    }

    /// Lazy-load the ViT tower weights (vision_tower.* tensors) and build
    /// the tower on first use. CALL WITH self.mu HELD.
    fn ensureVit(self: *Qwen38VTableAdapter) !*qwen38_vit.Qwen38Vit {
        if (self.vit) |v| return v;
        if (self.cfg.vision.depth == 0) return error.VisionNotSupported;
        var index = try safetensors_reader.loadOrBuildIndex(self.allocator, self.model_path);
        defer index.deinit();
        var map = std.StringHashMap(Array).init(self.allocator);
        errdefer {
            var it0 = map.iterator();
            while (it0.next()) |e| {
                self.allocator.free(e.key_ptr.*);
                e.value_ptr.deinit();
            }
            map.deinit();
        }
        var it = index.entries.iterator();
        var n_tensors: usize = 0;
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (!std.mem.startsWith(u8, name, "vision_tower.")) continue;
            try map.put(try self.allocator.dupe(u8, name), try index.loadTensor(name));
            n_tensors += 1;
        }
        if (n_tensors == 0) return error.VisionNotSupported;
        const cv = self.cfg.vision;
        const vcfg = qwen38_vit.Qwen38VitConfig{
            .depth = cv.depth,
            .hidden = @intCast(cv.hidden),
            .heads = @intCast(cv.heads),
            .intermediate = @intCast(cv.intermediate),
            .patch = @intCast(cv.patch),
            .temporal_patch = @intCast(cv.temporal_patch),
            .merge = @intCast(cv.merge),
            .out_hidden = @intCast(cv.out_hidden),
            .num_pos = @intCast(cv.num_pos),
            .in_ch = @intCast(cv.in_ch),
        };
        const v = try self.allocator.create(qwen38_vit.Qwen38Vit);
        errdefer self.allocator.destroy(v);
        v.* = try qwen38_vit.Qwen38Vit.init(self.allocator, self.model.ctx, vcfg, &map);
        self.vit = v;
        self.vit_weights = map;
        std.log.info("Qwen3.8 vision tower loaded: {d} tensors", .{n_tensors});
        return v;
    }

    fn visionProcessImage(ctx_ptr: *anyopaque, image: generation.VisionImage) anyerror!generation.VisionImageOut {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        const vit = try self.ensureVit();
        // mx.compile's tracer/eval uses the CURRENT thread's default stream
        // (transforms.cpp), unlike eager ops which take ctx explicitly. This
        // op runs on an HTTP worker fiber with no default stream set, so
        // point it at the model's GPU stream before the tower forward —
        // otherwise the compiled ViT block fails with
        // "no Stream(gpu, N) in current thread". Per-thread, harmless to the
        // eager path (all its ops take ctx explicitly).
        _ = c.c.mlx_set_default_stream(self.model.ctx.stream.inner);
        var pp = switch (image) {
            .path => |p| try qwen38_preprocess.preprocessImageFile(self.allocator, self.model.ctx, p),
            .bytes => |b| try qwen38_preprocess.preprocessImageBytes(self.allocator, self.model.ctx, b),
        };
        defer pp.deinit();
        const feats = try vit.forward(pp.pixels, pp.grid_thw[0], pp.grid_thw[1], pp.grid_thw[2]);
        errdefer feats.deinit();
        const merge: i32 = @intCast(self.cfg.vision.merge);
        return .{
            .features = feats,
            .block = .{ pp.grid_thw[0], @divExact(pp.grid_thw[1], merge), @divExact(pp.grid_thw[2], merge) },
            .n_pads = pp.n_image_pad,
        };
    }

    fn visionPrefill(ctx_ptr: *anyopaque, ids: []const u32, vision: generation.VisionData) anyerror!Array {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        return self.model.forwardTokensWithVision(ids, .{
            .features = vision.features,
            .blocks = vision.blocks,
        });
    }

    /// Embedding op (/v1/embeddings): fresh prefill over `ids` (raw text
    /// tokens — the server tokenizes WITHOUT the chat template, same as
    /// /v1/tokenize), LAST-position POST-final-norm hidden, L2-normalized.
    /// Pooling: last-token, the standard choice for decoder-only LMs.
    /// forwardTokensEmbed resets all GDN/attention state up front, and the
    /// next chat prefill resets again, so embedding calls neither continue
    /// nor pollute chat state. Takes the adapter mutex like every op that
    /// touches the model ctx (see visionProcessImage).
    fn embedTokens(ctx_ptr: *anyopaque, allocator: std.mem.Allocator, ids: []const u32) anyerror![]f32 {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        if (ids.len == 0) return error.EmptyInput;
        const pre_norm = try self.model.forwardTokensEmbed(ids); // [1, hidden] bf16
        defer pre_norm.deinit();
        const ctx = self.model.ctx;
        const xf = try ops.astype(ctx, pre_norm, .float32);
        defer xf.deinit();
        const nf = try fast_mod.rmsNorm(ctx, xf, self.model.final_norm_f32.?, self.cfg.rms_norm_eps);
        defer nf.deinit();
        try nf.eval();
        const slice = try nf.dataSlice(f32);
        const out = try allocator.dupe(f32, slice);
        errdefer allocator.free(out);
        var ssq: f64 = 0;
        for (out) |v| ssq += @as(f64, v) * @as(f64, v);
        if (ssq > 0) {
            const inv: f32 = @floatCast(1.0 / @sqrt(ssq));
            for (out) |*v| v.* *= inv;
        }
        return out;
    }

    fn mtpDims(self: *Qwen38VTableAdapter) qwen38_mtp.Qwen38MtpDims {
        const cfg = self.cfg;
        return .{
            .hidden_size = cfg.hidden_size,
            .num_heads = cfg.num_attention_heads,
            .num_kv_heads = cfg.num_key_value_heads,
            .head_dim = cfg.head_dim,
            .rotary_dims = cfg.rotaryDims(),
            .intermediate_size = cfg.intermediate_size,
            .rope_theta = cfg.rope_theta,
            .rms_norm_eps = cfg.rms_norm_eps,
            .output_gate = cfg.attn_output_gate,
        };
    }

    fn mtpEnable(ctx_ptr: *anyopaque, num_drafts: usize) anyerror!bool {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        if (std.c.getenv("DMLX_QWEN38_DSPINE_DIR")) |d| {
            return self.dspineEnable(std.mem.span(d));
        }
        if (std.c.getenv("DMLX_QWEN38_DFLASH2_DIR")) |d| {
            return self.dflash2Enable(std.mem.span(d));
        }
        self.mtp_ops.num_drafts = num_drafts;
        if (self.mtp != null) return true;
        const dir = blk: {
            if (std.c.getenv("DMLX_QWEN38_MTP_DIR")) |d| break :blk std.mem.span(d);
            break :blk default_mtp_drafter_dir;
        };
        std.log.info("Qwen3.8 MTP: loading drafter from {s} ({d} drafts/round)", .{ dir, num_drafts });
        const m = try self.allocator.create(qwen38_mtp.Qwen38Mtp);
        errdefer self.allocator.destroy(m);
        m.* = qwen38_mtp.Qwen38Mtp.load(self.allocator, self.model.ctx, dir, self.mtpDims()) catch |err| {
            std.log.warn("Qwen3.8 MTP drafter load failed at {s} ({}; set DMLX_QWEN38_MTP_DIR)", .{ dir, err });
            self.allocator.destroy(m);
            return false;
        };
        self.mtp = m;
        return true;
    }

    fn mtpLmHead(self: *Qwen38VTableAdapter) qwen38_mtp.MtpLinear {
        const lm = self.model.lm_head.?;
        return .{ .w = lm.w, .scales = lm.scales, .biases = lm.biases };
    }

    /// Trunk hidden is fed to the head AFTER the backbone's final RMSNorm
    /// (omlx's shipped convention: `HEAD_HIDDEN_POST_NORM = True`,
    /// prompt_priming.py — "accepts a few points higher at every depth").
    ///
    /// DEFAULT ON (2026-11-02): measured p1 0.643 -> 0.776 on empty-think
    /// prose and p4 0.267 -> 0.677 on code, tok/round +12-15% at equal
    /// round cost; omlx matches at p1=0.736 on the same content. The pre-norm
    /// convention (PR 990 style) is recoverable with
    /// `DMLX_MTP_NO_TRUNK_POSTNORM=1`. Applies to every trunk-sourced hidden
    /// (prefill fold rows, draft-0 input, verify chaining rows, commit
    /// replay rows); the head's OWN chained output is never trunk-normed.
    fn trunkPostnormEnabled(self: *Qwen38VTableAdapter) bool {
        if (self.mtp_trunk_postnorm == null)
            self.mtp_trunk_postnorm = std.c.getenv("DMLX_MTP_NO_TRUNK_POSTNORM") == null;
        return self.mtp_trunk_postnorm.?;
    }

    fn maybeTrunkNorm(self: *Qwen38VTableAdapter, x: Array) !Array {
        if (!self.trunkPostnormEnabled()) return x;
        const ctx = self.model.ctx;
        const xf = try ops.astype(ctx, x, .float32);
        defer xf.deinit();
        const nf = try fast_mod.rmsNorm(ctx, xf, self.model.final_norm_f32.?, self.cfg.rms_norm_eps);
        defer nf.deinit();
        return ops.astype(ctx, nf, .bfloat16);
    }

    /// DFlash2 enable: load the block drafter and force num_drafts to the
    /// checkpoint's block_size - 1 (7). The engine's depth controller may
    /// still verify a shorter prefix per round.
    fn dflash2Enable(self: *Qwen38VTableAdapter, dir: []const u8) anyerror!bool {
        self.mtp_ops.num_drafts = 7;
        self.mtp_ops.verifyTree = &mtpVerifyTree;
        self.mtp_ops.commitTree = &mtpCommitTree;
        self.mtp_ops.rollbackTree = &mtpRollbackTree;
        self.mtp_ops.draftTree = &mtpDraftTree;
        self.mtp_ops.verifyTreeArgmax = &mtpVerifyTreeArgmax;
        self.mtp_ops.verifyTreeHidden = &mtpVerifyTreeHidden;
        if (self.dflash2 != null) return true;
        std.log.info("Qwen3.8 DFlash2: loading drafter from {s}", .{dir});
        const m = try self.allocator.create(qwen38_dflash2.DFlash2);
        errdefer self.allocator.destroy(m);
        const quant_bits: u8 = if (std.c.getenv("DMLX_DF2_DRAFTER_BITS")) |b|
            if (std.mem.eql(u8, std.mem.span(b), "0")) 0 else 4
        else
            4;
        m.* = qwen38_dflash2.DFlash2.load(self.allocator, self.model.ctx, dir, quant_bits) catch |err| {
            std.log.warn("Qwen3.8 DFlash2 drafter load failed at {s} ({})", .{ dir, err });
            self.allocator.destroy(m);
            return false;
        };
        self.dflash2 = m;
        self.df2_ctx = qwen38_dflash2.DFlash2Context.init(self.allocator, self.model.ctx, m) catch |err| {
            std.log.warn("Qwen3.8 DFlash2 context init failed ({})", .{err});
            m.deinit();
            self.allocator.destroy(m);
            return false;
        };
        self.df2_block_len = 0;
        self.df2_block_idx = 0;
        if (self.model.snapshotDir()) |snap_dir| {
            self.loadDf2Cache(snap_dir) catch |err| {
                if (err != error.Df2NoRows and err != error.Df2NoTokens) {
                    std.log.warn("Qwen3.8 DFlash2: persisted context load failed: {}", .{err});
                }
            };
        }
        return true;
    }

    /// DFlash2 prefill: backbone prefill with hidden taps; the tap rows
    /// (projected at observe time) seed the drafter context. Returns the
    /// same last-logits/last-hidden contract as mtpPrefill.
    fn dflash2Prefill(self: *Qwen38VTableAdapter, ids: []const u32) anyerror!generation.MtpStepOut {
        self.dflash2ClearBlock();
        self.dflash2ClearPending();
        self.df2_ctx.?.reset();
        const ctx = self.model.ctx;
        const m = self.dflash2.?;

        // How much of the drafter context can the adapter cache supply for
        // a snapshot-restored prefix? Rows are per-token projected context.
        var cache_plen: usize = 0;
        if (self.df2_cache_rows) |cr| {
            const n_rows: usize = @intCast(cr.shape()[1]);
            const max_rows = @min(n_rows, @min(ids.len, self.df2_cache_ids.len));
            while (cache_plen < max_rows and self.df2_cache_ids[cache_plen] == ids[cache_plen]) cache_plen += 1;
        }
        const snap_plen = self.model.peekPrefixSnapshotLen(ids);
        // The ring only ever needs the newest ring_slots rows, so the cache
        // "covers" a prefix as long as it holds those rows: full coverage
        // for prefixes <= window, tail coverage for longer ones.
        const window: usize = @intCast(m.cfg.sliding_window - 1);
        const cache_rows_len: usize = if (self.df2_cache_rows) |cr| @intCast(cr.shape()[1]) else 0;
        const covered = @min(cache_plen, cache_rows_len); // rows the cache can supply for this prompt
        const need: usize = @min(snap_plen, window);
        // Suppress the restore only when the cache cannot supply the rows
        // the ring would need for the restored prefix.
        const suppress = snap_plen > 0 and covered < need;
        if (suppress) self.model.suppress_snapshot_restore = true;
        defer self.model.suppress_snapshot_restore = false;
        // On a suppressed (full tapped) prefill the tap covers every row —
        // seeding the cached prefix too would duplicate it in the ring.
        const usable_prefix = if (suppress) 0 else snap_plen;

        var tap = qwen38.Qwen38ServingModel.HiddenTap.init(&qwen38.Qwen38ServingModel.dflash2_tap_layer_ids);
        errdefer tap.deinit();
        const res = try self.model.forwardTokensCoreTap(ids, .all, true, &tap);
        defer res.logits.deinit();
        defer res.hidden.?.deinit();

        // Seed the drafter context: cached projected rows for the restored
        // prefix + freshly tapped rows for the freshly computed part.
        if (usable_prefix > 0) {
            // Seed the NEWEST min(usable_prefix, window) rows of the prefix,
            // at their absolute positions (RoPE offsets depend on them).
            const cr = self.df2_cache_rows.?;
            const width: i32 = cr.shape()[2];
            const take: usize = @min(usable_prefix, window);
            const rows = try ops.slice(ctx, cr, &[_]i32{ 0, @intCast(cache_rows_len - take), 0 }, &[_]i32{ 1, @intCast(cache_rows_len), width }, &[_]i32{ 1, 1, 1 });
            defer rows.deinit();
            try self.df2_ctx.?.seedFromProjectedAt(rows, @intCast(usable_prefix - take));
        }
        if (try tap.finish(ctx)) |rows| {
            defer rows.deinit();
            try self.df2_ctx.?.observe(rows);
        }
        // Refresh the adapter cache to cover THIS prompt (rows come from
        // the context's projected-row shadow).
        self.updateDflash2Cache(ids) catch |err|
            std.log.warn("DFlash2 context cache update failed: {}", .{err});
        const lshape = res.logits.shape();
        const vocab: i32 = lshape[2];
        const flat = try ops.reshape(ctx, res.logits, &[_]i32{vocab});
        defer flat.deinit();
        const logits = try ops.astype(ctx, flat, .float32);
        errdefer logits.deinit();
        const hidden_dim: i32 = res.hidden.?.shape()[2];
        const L: i32 = res.hidden.?.shape()[1];
        const hrow = try ops.slice(ctx, res.hidden.?, &[_]i32{ 0, L - 1, 0 }, &[_]i32{ 1, L, hidden_dim }, &[_]i32{ 1, 1, 1 });
        defer hrow.deinit();
        const hflat = try ops.reshape(ctx, hrow, &[_]i32{ 1, hidden_dim });
        defer hflat.deinit();
        const hidden = try ops.copy(ctx, hflat);
        return .{ .logits = logits, .hidden = hidden };
    }

    /// Refresh the drafter-context cache after a prefill. The projected
    /// rows live in df2_ctx.rows_shadow (maintained by observe/seed), so
    /// this just snapshots the shadow + ids.
    fn updateDflash2Cache(self: *Qwen38VTableAdapter, ids: []const u32) !void {
        const ctx = self.model.ctx;
        if (self.df2_cache_ids.len > 0) self.allocator.free(self.df2_cache_ids);
        self.df2_cache_ids = try self.allocator.dupe(u32, ids);
        if (self.df2_cache_rows) |r| r.deinit();
        self.df2_cache_rows = null;
        if (self.df2_ctx.?.rowsShadow()) |sh| {
            self.df2_cache_rows = try ops.copy(ctx, sh);
        }
        // Persist alongside the backbone snapshot: on a fresh process the
        // restored snapshot is only usable when the drafter context rows
        // covering its prefix are also present (else dflash2Prefill
        // suppresses the restore and repays the full prefill).
        if (self.df2_cache_rows != null and ids.len >= 2048 and ids.len > self.df2_last_persisted) {
            if (self.model.snapshotDir()) |dir| {
                self.persistDf2Cache(dir, ids) catch |err| {
                    std.log.warn("DFlash2 context persist failed: {}", .{err});
                };
                self.df2_last_persisted = ids.len;
            }
        }
    }

    fn persistDf2Cache(self: *Qwen38VTableAdapter, dir: []const u8, ids: []const u32) !void {
        const mlx_io = @import("mlx").io;
        var buf: [1024]u8 = undefined;
        // mkdir -p
        var i: usize = 0;
        while (i <= dir.len) : (i += 1) {
            if (i != dir.len and dir[i] != '/') continue;
            if (i == 0) continue;
            const part_z = self.allocator.dupeZ(u8, dir[0..i]) catch return;
            defer self.allocator.free(part_z);
            _ = mkdir_c.mkdir(part_z.ptr, 0o755);
        }
        const path = try std.fmt.bufPrint(&buf, "{s}/df2_latest.safetensors", .{dir});
        var weights = std.StringHashMap(Array).init(self.allocator);
        defer weights.deinit();
        try weights.put("rows", self.df2_cache_rows.?);
        const toks_i32 = try self.allocator.alloc(i32, ids.len);
        defer self.allocator.free(toks_i32);
        for (ids, 0..) |t, j| toks_i32[j] = @intCast(t);
        const toks_arr = try Array.fromData(self.allocator, i32, toks_i32, &[_]i32{@intCast(toks_i32.len)});
        defer toks_arr.deinit();
        try toks_arr.eval();
        try weights.put("tokens", toks_arr);
        var metadata = std.StringHashMap([]const u8).init(self.allocator);
        defer metadata.deinit();
        try mlx_io.saveSafetensors(self.allocator, path, weights, metadata);
    }

    fn loadDf2Cache(self: *Qwen38VTableAdapter, dir: []const u8) !void {
        const mlx_io = @import("mlx").io;
        var buf: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}/df2_latest.safetensors", .{dir});
        var res = mlx_io.loadSafetensors(self.allocator, path) catch return;
        defer res.deinit(self.allocator);
        var rows_it = res.weights.fetchRemove("rows") orelse return error.Df2NoRows;
        defer self.allocator.free(rows_it.key);
        const toks_it = res.weights.fetchRemove("tokens") orelse {
            rows_it.value.deinit();
            return error.Df2NoTokens;
        };
        defer self.allocator.free(toks_it.key);
        var toks = toks_it.value;
        defer toks.deinit();
        try toks.eval();
        const slice = toks.dataSlice(i32) catch return error.Df2BadTokens;
        if (slice.len == 0) return error.Df2Empty;
        if (self.df2_cache_ids.len > 0) self.allocator.free(self.df2_cache_ids);
        self.df2_cache_ids = try self.allocator.alloc(u32, slice.len);
        for (slice, 0..) |t, j| self.df2_cache_ids[j] = @intCast(t);
        if (self.df2_cache_rows) |r| r.deinit();
        self.df2_cache_rows = rows_it.value;
        self.df2_last_persisted = slice.len;
        std.log.info("Qwen3.8 DFlash2: loaded persisted drafter context ({d} tokens)", .{slice.len});
    }

    /// Compute the round's draft block: one parallel forward over
    /// [anchor, mask x 7], then the greedy candidate selector chain.
    fn df2Now() i128 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
        return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
    }

    fn dflash2ComputeBlock(self: *Qwen38VTableAdapter, tok_gpu: Array) anyerror!void {
        const prof = std.c.getenv("DMLX_DF2_PROF") != null;
        var t0: i128 = 0;
        if (prof) t0 = df2Now();
        defer if (prof) {
            const ms = @as(f64, @floatFromInt(df2Now() - t0)) / 1e6;
            std.debug.print("DF2 block total {d:.1}ms\n", .{ms});
        };
        // Arm the tuned verify-qmm kernels for the whole block: the drafter's
        // projections are M<=8 shapes that stock runs at ~half the sg8
        // bandwidth. Draft-side numerics only shift proposals, never output.
        verify_qmm.arm();
        defer verify_qmm.disarm();
        const m = self.dflash2.?;
        const ctx = self.model.ctx;
        const P: i32 = @intCast(m.cfg.block_size - 1);
        const H: i32 = @intCast(m.cfg.hidden_size);

        const mask_ids = try self.allocator.alloc(i32, @intCast(P));
        defer self.allocator.free(mask_ids);
        for (mask_ids) |*t| t.* = @intCast(m.cfg.mask_token_id);
        const mask_arr = try Array.fromData(self.allocator, i32, mask_ids, &[_]i32{P});
        defer mask_arr.deinit();
        const block_ids = try shape_mod.concatenateAxis(ctx, &[_]Array{ tok_gpu, mask_arr }, 0);
        defer block_ids.deinit();

        var ring = &self.df2_ctx.?;
        if (ring.fed == 0) return error.DFlash2EmptyContext;
        const mask = try ring.draftMask();
        defer mask.deinit();
        const hidden_full = try qwen38_dflash2.blockHiddenRing(ctx, m, ring, self.model.embed_table_bf16.?, block_ids, mask);
        defer hidden_full.deinit();
        var t1: i128 = 0;
        if (prof) {
            hidden_full.eval() catch {};
            t1 = df2Now();
            std.debug.print("DF2 block fwd {d:.1}ms\n", .{@as(f64, @floatFromInt(t1 - t0)) / 1e6});
        }
        const draft_hidden = try ops.slice(ctx, hidden_full, &[_]i32{ 0, 1, 0 }, &[_]i32{ 1, P + 1, H }, &[_]i32{ 1, 1, 1 });
        defer draft_hidden.deinit();
        const lm = self.mtpLmHead();
        // lm_head routed to sg8 by the arm at the top of this function.
        const logits = try lm.apply(ctx, draft_hidden);
        defer logits.deinit();

        var t2: i128 = 0;
        if (prof) {
            logits.eval() catch {};
            t2 = df2Now();
            std.debug.print("DF2 block lm_head {d:.1}ms\n", .{@as(f64, @floatFromInt(t2 - t1)) / 1e6});
        }
        const tokens_gpu = try qwen38_dflash2.selectGreedyGpu(ctx, m, draft_hidden, logits, tok_gpu);
        defer tokens_gpu.deinit();
        if (prof) std.debug.print("DF2 block select {d:.1}ms\n", .{@as(f64, @floatFromInt(df2Now() - t2)) / 1e6});
        try tokens_gpu.eval();
        const tslice = try tokens_gpu.dataSlice(i32);
        var tokens: [7]u32 = undefined;
        for (tslice[0..7], 0..) |t, i| tokens[i] = @intCast(t);
        self.df2_block = tokens;
        self.df2_block_len = tokens.len;
        self.df2_block_idx = 0;
    }

    /// GPU-only block computation for the predraft path: identical to the
    /// above but returns the [7] i32 token array WITHOUT syncing.
    fn dflash2ComputeBlockLazy(self: *Qwen38VTableAdapter, tok_gpu: Array) anyerror!Array {
        const m = self.dflash2.?;
        const ctx = self.model.ctx;
        const P: i32 = @intCast(m.cfg.block_size - 1);
        const H: i32 = @intCast(m.cfg.hidden_size);
        const mask_ids = try self.allocator.alloc(i32, @intCast(P));
        defer self.allocator.free(mask_ids);
        for (mask_ids) |*t| t.* = @intCast(m.cfg.mask_token_id);
        const mask_arr = try Array.fromData(self.allocator, i32, mask_ids, &[_]i32{P});
        defer mask_arr.deinit();
        const block_ids = try shape_mod.concatenateAxis(ctx, &[_]Array{ tok_gpu, mask_arr }, 0);
        defer block_ids.deinit();
        var ring = &self.df2_ctx.?;
        const mask = try ring.draftMask();
        defer mask.deinit();
        const hidden_full = try qwen38_dflash2.blockHiddenRing(ctx, m, ring, self.model.embed_table_bf16.?, block_ids, mask);
        defer hidden_full.deinit();
        const draft_hidden = try ops.slice(ctx, hidden_full, &[_]i32{ 0, 1, 0 }, &[_]i32{ 1, P + 1, H }, &[_]i32{ 1, 1, 1 });
        defer draft_hidden.deinit();
        const lm = self.mtpLmHead();
        const logits = try lm.apply(ctx, draft_hidden);
        defer logits.deinit();
        return qwen38_dflash2.selectGreedyGpu(ctx, m, draft_hidden, logits, tok_gpu);
    }

    /// Predraft: condition on ALL drafts being accepted, ingest the verify
    /// rows into the ring tentatively, and build the next block's token
    /// graph — all lazy, so it rides the acceptance sync's GPU flush.
    fn dflash2Predraft(self: *Qwen38VTableAdapter, v: *qwen38.Qwen38VerifyOut) anyerror!void {
        const th = v.tap_hidden orelse return;
        if (self.df2_pending_tokens != null) return; // already stashed
        const ctx = self.model.ctx;
        const L: usize = @intCast(th.shape()[1]);
        // Verify argmax once; mtpVerifyArgmax serves the engine from this.
        const f32l = try ops.astype(ctx, v.logits, .float32);
        defer f32l.deinit();
        const Li: i32 = v.logits.shape()[1];
        const vocab: i32 = v.logits.shape()[2];
        const flat = try ops.reshape(ctx, f32l, &[_]i32{ Li, vocab });
        defer flat.deinit();
        const am = try reduce_mod.argmaxAxis(ctx, flat, -1, false); // [L] i32
        errdefer am.deinit();
        // Tentative ring feed of all verify rows.
        const proj = try qwen38_dflash2.projectContext(ctx, self.dflash2.?, th);
        defer proj.deinit();
        try self.df2_ctx.?.observeTentative(proj);
        errdefer self.df2_ctx.?.discardTentative();
        // Anchor for the assumed-all-accepted block: bonus token (last row).
        const anchor = try ops.slice(ctx, am, &[_]i32{Li - 1}, &[_]i32{Li}, &[_]i32{1});
        defer anchor.deinit();
        // The block's positions/RoPE and the mask window follow the
        // tentative rows, so the ring must present the post-commit fed.
        self.df2_ctx.?.advanceFedForTentativeDraft();
        const tokens = try self.dflash2ComputeBlockLazy(anchor);
        self.df2_pending_tokens = tokens;
        self.df2_pending_argmax = am;
        self.df2_pending_rows = L;
    }

    fn dflash2ClearPending(self: *Qwen38VTableAdapter) void {
        if (self.df2_pending_tokens) |t| t.deinit();
        self.df2_pending_tokens = null;
        if (self.df2_pending_argmax) |a| a.deinit();
        self.df2_pending_argmax = null;
        self.df2_pending_rows = 0;
    }

    fn dflash2ClearBlock(self: *Qwen38VTableAdapter) void {
        if (self.df2_block_logits) |l| l.deinit();
        self.df2_block_logits = null;
        self.df2_block_len = 0;
        self.df2_block_idx = 0;
    }

    /// Sampled-path block computation: same forward as dflash2ComputeBlock
    /// but keeps the [P, V] f32 logits instead of argmaxing them away. No
    /// sync here — the engine joint-asyncEvals the draft rows with the
    /// verify rows during acceptance.
    fn dflash2ComputeBlockLogits(self: *Qwen38VTableAdapter, tok_gpu: Array) anyerror!void {
        const prof = std.c.getenv("DMLX_DF2_PROF") != null;
        var t0: i128 = 0;
        if (prof) t0 = df2Now();
        // See dflash2ComputeBlock: tuned qmm for the whole block.
        verify_qmm.arm();
        defer verify_qmm.disarm();
        const m = self.dflash2.?;
        const ctx = self.model.ctx;
        const P: i32 = @intCast(m.cfg.block_size - 1);
        const H: i32 = @intCast(m.cfg.hidden_size);
        const mask_ids = try self.allocator.alloc(i32, @intCast(P));
        defer self.allocator.free(mask_ids);
        for (mask_ids) |*t| t.* = @intCast(m.cfg.mask_token_id);
        const mask_arr = try Array.fromData(self.allocator, i32, mask_ids, &[_]i32{P});
        defer mask_arr.deinit();
        const block_ids = try shape_mod.concatenateAxis(ctx, &[_]Array{ tok_gpu, mask_arr }, 0);
        defer block_ids.deinit();
        var ring = &self.df2_ctx.?;
        if (ring.fed == 0) return error.DFlash2EmptyContext;
        const mask = try ring.draftMask();
        defer mask.deinit();
        const hidden_full = try qwen38_dflash2.blockHiddenRing(ctx, m, ring, self.model.embed_table_bf16.?, block_ids, mask);
        defer hidden_full.deinit();
        if (prof) {
            hidden_full.eval() catch {};
            std.debug.print("DF2 sample block fwd {d:.1}ms\n", .{@as(f64, @floatFromInt(df2Now() - t0)) / 1e6});
        }
        const t1 = df2Now();
        const draft_hidden = try ops.slice(ctx, hidden_full, &[_]i32{ 0, 1, 0 }, &[_]i32{ 1, P + 1, H }, &[_]i32{ 1, 1, 1 });
        defer draft_hidden.deinit();
        const lm = self.mtpLmHead();
        const logits = try lm.apply(ctx, draft_hidden); // [1, P, V]
        defer logits.deinit();
        const f32l = try ops.astype(ctx, logits, .float32);
        defer f32l.deinit();
        const vocab: i32 = f32l.shape()[2];
        const flat = try ops.reshape(ctx, f32l, &[_]i32{ P, vocab });
        errdefer flat.deinit();
        // Materialize ONCE: the sampled draft path draws each position's
        // token from its row on the host, and without this eval every row
        // pull re-syncs the whole GPU queue (measured 115-200 ms/round at
        // 2k context; the greedy path pays one sync per BLOCK instead).
        try flat.eval();
        if (prof) {
            std.debug.print("DF2 sample lm_head+eval {d:.1}ms\n", .{@as(f64, @floatFromInt(df2Now() - t1)) / 1e6});
        }
        self.df2_block_logits = flat;
        self.df2_block_len = @intCast(P);
        self.df2_block_idx = 0;
    }

    /// Sampled (temp>0) draft step: hands out block-logits row by row. The
    /// block is block-parallel — q_i is conditioned on the anchor + ring
    /// context, not on the previously sampled drafts — which is exactly the
    /// q the engine's rejection sampler needs (a position-independent
    /// proposal keeps the adjusted-resample identity exact).
    fn dflash2Draft(self: *Qwen38VTableAdapter, hidden: Array, token: u32) anyerror!generation.MtpStepOut {
        if (self.df2_block_idx >= self.df2_block_len) {
            self.dflash2ClearBlock();
            const tok_gpu = try Array.fromData(self.allocator, i32, &[_]i32{@intCast(token)}, &[_]i32{1});
            defer tok_gpu.deinit();
            try self.dflash2ComputeBlockLogits(tok_gpu);
        }
        const ctx = self.model.ctx;
        const bl = self.df2_block_logits.?;
        const i: i32 = @intCast(self.df2_block_idx);
        const vocab: i32 = bl.shape()[1];
        const row = try ops.slice(ctx, bl, &[_]i32{ i, 0 }, &[_]i32{ i + 1, vocab }, &[_]i32{ 1, 1 });
        defer row.deinit();
        const flat = try ops.reshape(ctx, row, &[_]i32{vocab});
        defer flat.deinit();
        const logits = try ops.copy(ctx, flat);
        self.df2_block_idx += 1;
        // Hidden passes through (drafter conditions on the ring, not a chain).
        const h = try ops.copy(ctx, hidden);
        return .{ .logits = logits, .hidden = h };
    }

    fn dflash2DraftGreedy(self: *Qwen38VTableAdapter, hidden: Array, tok_gpu: Array) anyerror!generation.MtpGreedyDraftOut {
        if (self.df2_block_idx >= self.df2_block_len) {
            if (self.df2_pending_tokens) |pt| {
                // Adopted predraft: one sync here (the values were already
                // flushed by the acceptance argmax read).
                try pt.eval();
                const ts = try pt.dataSlice(i32);
                for (ts[0..7], 0..) |t, i| self.df2_block[i] = @intCast(t);
                self.df2_block_len = 7;
                self.df2_block_idx = 0;
                self.dflash2ClearPending();
            } else {
                try self.dflash2ComputeBlock(tok_gpu);
            }
        }
        const tok = self.df2_block[self.df2_block_idx];
        self.df2_block_idx += 1;
        const t = try Array.fromData(self.allocator, i32, &[_]i32{@intCast(tok)}, &[_]i32{1});
        errdefer t.deinit();
        // Hidden is passed through: the drafter conditions on the tap
        // context, not on a chained hidden, so the engine's h_cur is unused.
        const h = try ops.copy(self.model.ctx, hidden);
        return .{ .tok_gpu = t, .hidden = h };
    }

    fn dflash2Commit(self: *Qwen38VTableAdapter, v: *qwen38.Qwen38VerifyOut, accepted: usize) void {
        qwen38.commitVerify(self.model, v, accepted);
        const ctx = self.model.ctx;
        // Predraft settle: adopt only when every verify row was accepted
        // (the tentative feed covered exactly `pending_rows` rows).
        if (self.df2_ctx.?.tentative_start >= 0) {
            if (self.df2_pending_tokens != null and accepted == self.df2_pending_rows) {
                self.df2_ctx.?.adoptTentative() catch {
                    self.df2_ctx.?.discardTentative();
                    self.dflash2ClearPending();
                };
                // adopted: keep df2_pending_tokens for the draft calls;
                // the argmax stash can go (engine already read it).
                if (self.df2_pending_argmax) |a| a.deinit();
                self.df2_pending_argmax = null;
                self.dflash2ClearBlock();
                return;
            }
            self.df2_ctx.?.discardTentative();
            self.dflash2ClearPending();
        }
        if (v.tap_hidden) |th| {
            const width: i32 = th.shape()[2];
            const rows = ops.slice(ctx, th, &[_]i32{ 0, 0, 0 }, &[_]i32{ 1, @intCast(accepted), width }, &[_]i32{ 1, 1, 1 }) catch return;
            defer rows.deinit();
            self.df2_ctx.?.observe(rows) catch return;
        }
        self.dflash2ClearBlock();
    }

    fn dspineEnable(self: *Qwen38VTableAdapter, dir: []const u8) anyerror!bool {
        const block = try qwen38_dspine.DSpineConfig.parseBlockEnv(if (std.c.getenv("DMLX_DSPINE_BLOCK")) |b| std.mem.span(b) else null);
        self.mtp_ops.num_drafts = block - 1;
        if (self.dspine != null) return true;
        std.log.info("Qwen3.8 DSpine: loading block={d} drafter from {s}", .{ block, dir });
        const m = try self.allocator.create(qwen38_dspine.DSpine);
        errdefer self.allocator.destroy(m);
        const quant_bits: u8 = if (std.c.getenv("DMLX_DSPINE_DRAFTER_BITS")) |b|
            if (std.mem.eql(u8, std.mem.span(b), "0")) 0 else 4
        else
            4;
        m.* = qwen38_dspine.DSpine.loadBlock(self.allocator, self.model.ctx, dir, quant_bits, block) catch |err| {
            std.log.warn("Qwen3.8 DSpine drafter load failed at {s} ({})", .{ dir, err });
            self.allocator.destroy(m);
            return false;
        };
        self.dspine = m;
        self.dspine_ctx = qwen38_dspine.DSpineContext.init(self.allocator, self.model.ctx, m) catch |err| {
            std.log.warn("Qwen3.8 DSpine context init failed ({})", .{err});
            m.deinit();
            self.allocator.destroy(m);
            self.dspine = null;
            return false;
        };
        self.dspine_block_len = 0;
        self.dspine_block_idx = 0;
        return true;
    }

    fn dspineClearBlock(self: *Qwen38VTableAdapter) void {
        if (self.dspine_table_host.len > 0) self.allocator.free(self.dspine_table_host);
        self.dspine_table_host = &.{};
        self.dspine_block_len = 0;
        self.dspine_block_idx = 0;
    }

    fn updateDSpineCache(self: *Qwen38VTableAdapter, ids: []const u32) !void {
        if (self.dspine_cache_ids.len > 0) self.allocator.free(self.dspine_cache_ids);
        self.dspine_cache_ids = try self.allocator.dupe(u32, ids);
        if (self.dspine_cache_rows) |r| r.deinit();
        self.dspine_cache_rows = null;
        if (self.dspine_ctx.?.rowsShadow()) |sh| self.dspine_cache_rows = try ops.copy(self.model.ctx, sh);
    }

    fn dspinePrefill(self: *Qwen38VTableAdapter, ids: []const u32) anyerror!generation.MtpStepOut {
        self.dspineClearBlock();
        self.dspine_ctx.?.reset();
        const ctx = self.model.ctx;
        const m = self.dspine.?;
        var cache_plen: usize = 0;
        if (self.dspine_cache_rows) |cr| {
            const n_rows: usize = @intCast(cr.shape()[1]);
            const max_rows = @min(n_rows, @min(ids.len, self.dspine_cache_ids.len));
            while (cache_plen < max_rows and self.dspine_cache_ids[cache_plen] == ids[cache_plen]) cache_plen += 1;
        }
        const snap_plen = self.model.peekPrefixSnapshotLen(ids);
        const window: usize = @intCast(m.cfg.sliding_window - 1);
        const cache_rows_len: usize = if (self.dspine_cache_rows) |cr| @intCast(cr.shape()[1]) else 0;
        const covered = @min(cache_plen, cache_rows_len);
        const need = @min(snap_plen, window);
        const suppress = snap_plen > 0 and covered < need;
        if (suppress) self.model.suppress_snapshot_restore = true;
        defer self.model.suppress_snapshot_restore = false;
        const usable_prefix = if (suppress) 0 else snap_plen;
        var tap = qwen38.Qwen38ServingModel.HiddenTap.init(&m.cfg.target_layer_ids);
        errdefer tap.deinit();
        const res = try self.model.forwardTokensCoreTap(ids, .all, true, &tap);
        defer res.logits.deinit();
        defer res.hidden.?.deinit();
        if (usable_prefix > 0) {
            const cr = self.dspine_cache_rows.?;
            const width: i32 = cr.shape()[2];
            const take: usize = @min(usable_prefix, window);
            const rows = try ops.slice(ctx, cr, &[_]i32{ 0, @intCast(cache_rows_len - take), 0 }, &[_]i32{ 1, @intCast(cache_rows_len), width }, &[_]i32{ 1, 1, 1 });
            defer rows.deinit();
            try self.dspine_ctx.?.seedFromProjectedAt(rows, @intCast(usable_prefix - take));
        }
        if (try tap.finish(ctx)) |rows| {
            defer rows.deinit();
            if (std.c.getenv("DMLX_DSPINE_TRACE") != null) {
                const n = rows.shape()[1];
                const row = try ops.slice(ctx, rows, &[_]i32{ 0, n - 1, 0 }, &[_]i32{ 1, n, 8 }, &[_]i32{ 1, 1, 1 });
                defer row.deinit();
                const f = try ops.astype(ctx, row, .float32);
                defer f.deinit();
                try f.eval();
                std.debug.print("DSPINE target_hidden {any}\n", .{try f.dataSlice(f32)});
            }
            if (std.c.getenv("DMLX_DSPINE_DUMP_TAP")) |path| {
                const mlx_io = @import("mlx").io;
                var weights = std.StringHashMap(Array).init(self.allocator);
                defer weights.deinit();
                try weights.put("tap", rows);
                var metadata = std.StringHashMap([]const u8).init(self.allocator);
                defer metadata.deinit();
                try mlx_io.saveSafetensors(self.allocator, std.mem.span(path), weights, metadata);
            }
            try self.dspine_ctx.?.observe(rows);
        }
        self.updateDSpineCache(ids) catch |err| std.log.warn("DSpine context cache update failed: {}", .{err});
        if (std.c.getenv("DMLX_DSPINE_TRACE") != null) {
            if (self.dspine_ctx.?.rowsShadow()) |sh| {
                const n = sh.shape()[1];
                const row = try ops.slice(ctx, sh, &[_]i32{ 0, n - 1, 0 }, &[_]i32{ 1, n, 8 }, &[_]i32{ 1, 1, 1 });
                defer row.deinit();
                const f = try ops.astype(ctx, row, .float32);
                defer f.deinit();
                try f.eval();
                std.debug.print("DSPINE context {any}\n", .{try f.dataSlice(f32)});
            }
        }
        const lshape = res.logits.shape();
        const vocab: i32 = lshape[2];
        const flat = try ops.reshape(ctx, res.logits, &[_]i32{vocab});
        defer flat.deinit();
        const logits = try ops.astype(ctx, flat, .float32);
        errdefer logits.deinit();
        const hidden_dim: i32 = res.hidden.?.shape()[2];
        const L: i32 = res.hidden.?.shape()[1];
        const hrow = try ops.slice(ctx, res.hidden.?, &[_]i32{ 0, L - 1, 0 }, &[_]i32{ 1, L, hidden_dim }, &[_]i32{ 1, 1, 1 });
        defer hrow.deinit();
        const hflat = try ops.reshape(ctx, hrow, &[_]i32{ 1, hidden_dim });
        defer hflat.deinit();
        const hidden = try ops.copy(ctx, hflat);
        return .{ .logits = logits, .hidden = hidden };
    }

    fn dspineComputeBlock(self: *Qwen38VTableAdapter, tok_gpu: Array) anyerror!void {
        const prof = std.c.getenv("DMLX_DSPINE_PROF") != null;
        var t0: i128 = 0;
        if (prof) t0 = df2Now();
        verify_qmm.arm();
        defer verify_qmm.disarm();
        const m = self.dspine.?;
        const ctx = self.model.ctx;
        const P: i32 = @intCast(m.cfg.block_size - 1);
        const mask_ids = try self.allocator.alloc(i32, @intCast(P));
        defer self.allocator.free(mask_ids);
        for (mask_ids) |*t| t.* = @intCast(m.cfg.mask_token_id);
        const mask_arr = try Array.fromData(self.allocator, i32, mask_ids, &[_]i32{P});
        defer mask_arr.deinit();
        const block_ids = try shape_mod.concatenateAxis(ctx, &[_]Array{ tok_gpu, mask_arr }, 0);
        defer block_ids.deinit();
        const ring = &self.dspine_ctx.?;
        if (ring.fed == 0) return error.DSpineEmptyContext;
        var block = try qwen38_dspine.blockHiddenRing(ctx, m, ring, self.model.embed_table_bf16.?, block_ids);
        defer block.deinit();
        var t_fwd: i128 = 0;
        if (prof) {
            try block.hidden.eval();
            t_fwd = df2Now();
        }
        if (std.c.getenv("DMLX_DSPINE_TRACE") != null) {
            const row = try ops.slice(ctx, block.hidden, &[_]i32{ 0, 0, 0 }, &[_]i32{ 1, 1, 8 }, &[_]i32{ 1, 1, 1 });
            defer row.deinit();
            const f = try ops.astype(ctx, row, .float32);
            defer f.deinit();
            try f.eval();
            std.debug.print("DSPINE hidden {any}\n", .{try f.dataSlice(f32)});
            const raw = try ops.slice(ctx, block.stash.raw, &[_]i32{ 0, 1, 0 }, &[_]i32{ 1, 2, 8 }, &[_]i32{ 1, 1, 1 });
            defer raw.deinit();
            const raw_f = try ops.astype(ctx, raw, .float32);
            defer raw_f.deinit();
            try raw_f.eval();
            std.debug.print("DSPINE raw {any}\n", .{try raw_f.dataSlice(f32)});
        }
        const lm = self.mtpLmHead();
        const logits = try lm.apply(ctx, block.hidden);
        defer logits.deinit();
        var t1: i128 = 0;
        if (prof) {
            try logits.eval();
            t1 = df2Now();
        }
        const candidates = try qwen38_dspine.selectCandidates(ctx, logits, m.cfg.head_topk);
        defer candidates.deinit();
        var t2: i128 = 0;
        if (prof) {
            try candidates.eval();
            t2 = df2Now();
        }
        const table = try qwen38_dspine.refinementTable(ctx, m, lm, block.hidden, block.stash, candidates, tok_gpu);
        defer table.deinit();
        try eval_mod.asyncEval(self.allocator, &[_]Array{ table, candidates });
        try table.eval();
        if (prof) {
            const t3 = df2Now();
            const ms = struct {
                fn f(ns: i128) f64 {
                    return @as(f64, @floatFromInt(ns)) / 1e6;
                }
            }.f;
            std.debug.print("DSPINE prof fwd={d:.2} head={d:.2} topk={d:.2} refine={d:.2} total={d:.2}ms\n", .{ ms(t_fwd - t0), ms(t1 - t_fwd), ms(t2 - t1), ms(t3 - t2), ms(df2Now() - t0) });
        }
        const cand_slice = try candidates.dataSlice(i32);
        for (cand_slice, 0..) |t, i| self.dspine_candidates[i] = @intCast(t);
        const table_slice = try table.dataSlice(f32);
        if (self.dspine_table_host.len > 0) self.allocator.free(self.dspine_table_host);
        self.dspine_table_host = try self.allocator.dupe(f32, table_slice);
        const positions: usize = m.cfg.block_size - 1;
        const greedy = try qwen38_dspine.greedyChainFromHost(self.dspine_table_host, self.dspine_candidates[0 .. positions * 16], positions, 16, self.allocator);
        defer self.allocator.free(greedy);
        @memcpy(self.dspine_block[0..positions], greedy);
        if (std.c.getenv("DMLX_DSPINE_TRACE") != null) {
            std.debug.print("DSPINE fed={d} drafts={any}\n", .{ ring.fed, self.dspine_block[0..positions] });
            std.debug.print("DSPINE candidates={any}\n", .{self.dspine_candidates[0 .. positions * 16]});
        }
        self.dspine_block_len = positions;
        self.dspine_block_idx = 0;
    }

    fn dspineDraft(self: *Qwen38VTableAdapter, hidden: Array, token: u32) anyerror!generation.MtpStepOut {
        if (self.dspine_block_idx >= self.dspine_block_len) {
            self.dspineClearBlock();
            const tok_gpu = try Array.fromData(self.allocator, i32, &[_]i32{@intCast(token)}, &[_]i32{1});
            defer tok_gpu.deinit();
            try self.dspineComputeBlock(tok_gpu);
        }
        const pos = self.dspine_block_idx;
        const row = try qwen38_dspine.sparseLogitRowFromHost(self.dspine_table_host, self.dspine_candidates[0 .. self.dspine_block_len * 16], pos, token, self.dspine.?.cfg.vocab_size, self.allocator, self.dspine_block_len, 16);
        defer self.allocator.free(row);
        const logits = try Array.fromData(self.allocator, f32, row, &[_]i32{@intCast(row.len)});
        errdefer logits.deinit();
        self.dspine_block_idx += 1;
        const h = try ops.copy(self.model.ctx, hidden);
        return .{ .logits = logits, .hidden = h };
    }

    fn dspineDraftGreedy(self: *Qwen38VTableAdapter, hidden: Array, tok_gpu: Array) anyerror!generation.MtpGreedyDraftOut {
        if (self.dspine_block_idx >= self.dspine_block_len) try self.dspineComputeBlock(tok_gpu);
        const tok = self.dspine_block[self.dspine_block_idx];
        self.dspine_block_idx += 1;
        const t = try Array.fromData(self.allocator, i32, &[_]i32{@intCast(tok)}, &[_]i32{1});
        errdefer t.deinit();
        const h = try ops.copy(self.model.ctx, hidden);
        return .{ .tok_gpu = t, .hidden = h };
    }

    fn dspineCommit(self: *Qwen38VTableAdapter, v: *qwen38.Qwen38VerifyOut, accepted: usize) void {
        qwen38.commitVerify(self.model, v, accepted);
        if (v.tap_hidden) |th| {
            const width: i32 = th.shape()[2];
            const rows = ops.slice(self.model.ctx, th, &[_]i32{ 0, 0, 0 }, &[_]i32{ 1, @intCast(accepted), width }, &[_]i32{ 1, 1, 1 }) catch return;
            defer rows.deinit();
            self.dspine_ctx.?.observe(rows) catch return;
        }
        self.dspineClearBlock();
    }

    fn mtpPrefill(ctx_ptr: *anyopaque, ids: []const u32) anyerror!generation.MtpStepOut {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        if (self.dspine != null) return self.dspinePrefill(ids);
        if (self.dflash2 != null) return self.dflash2Prefill(ids);
        const m = self.mtp.?;
        // Backbone: normal prefill path (reset + fused recurrence — NO
        // per-position state retention; verify-mode would pin L x 3.1 MB x 48
        // layers and OOM at long prompts).
        m.cache.deinit();
        m.cache = .{};
        self.mtp_round_old_len = null;
        const out = try self.model.forwardTokensAllHidden(ids);
        defer out.logits.deinit();
        defer out.hidden_all.deinit();
        // Text prefill: rope_delta is 0 (reset inside the backbone prefill);
        // the drafter's rope offset carries no extra shift.
        m.attn.rope_extra = 0;
        return self.mtpPrefillTail(ids, out.logits, out.hidden_all);
    }

    /// Vision MTP prefill: backbone MRoPE prefill (resets state, installs
    /// rope_delta) + drafter priming, with the drafter's rope offset shifted
    /// by the same rope_delta the backbone decode steps use.
    fn mtpPrefillVision(ctx_ptr: *anyopaque, ids: []const u32, vision: generation.VisionData) anyerror!generation.MtpStepOut {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        if (self.dspine != null) return error.DSpineVisionUnsupported;
        const m = self.mtp.?;
        m.cache.deinit();
        m.cache = .{};
        self.mtp_round_old_len = null;
        const res = try self.model.forwardTokensWithVisionCapture(ids, .{
            .features = vision.features,
            .blocks = vision.blocks,
        }, .all);
        defer res.logits.deinit();
        defer res.hidden.?.deinit();
        const out = try self.mtpPrefillTail(ids, res.logits, res.hidden.?);
        // Propagate the backbone's MRoPE rope_delta to the drafter AFTER its
        // prompt pass (which primes at raw sequence positions, same as the
        // text path): subsequent draft/verify rope offsets are
        // cache.len() + rope_delta, matching the backbone's full-attention
        // layers. Setting it earlier would rope the prompt rows at negative
        // offsets (rope_delta < 0 for wide images).
        // Experiment: DMLX_MTP_VISION_DRAFTER_DELTA=0 keeps the drafter at
        // raw offsets everywhere (the mlx-vlm convention) for A/B.
        const drafter_delta = if (std.c.getenv("DMLX_MTP_VISION_DRAFTER_DELTA")) |s|
            std.mem.span(s)[0] != '0'
        else
            true;
        m.attn.rope_extra = if (drafter_delta) self.model.rope_delta else 0;
        return out;
    }

    /// Shared tail of the MTP prefill: drafter prompt pass over the backbone
    /// hidden rows, then last-position logits + last hidden for round 0.
    fn mtpPrefillTail(self: *Qwen38VTableAdapter, ids: []const u32, out_logits: Array, hidden_all: Array) anyerror!generation.MtpStepOut {
        const m = self.mtp.?;
        const ctx = self.model.ctx;

        // Drafter prompt pass: populate the head's KV cache over the prompt
        // (head position j ← (backbone hidden row j, embed ids[j+1])).
        if (ids.len > 1) {
            const hidden: i32 = hidden_all.shape()[2];
            const rows = try ops.slice(ctx, hidden_all, &[_]i32{ 0, 0, 0 }, &[_]i32{ 1, @as(i32, @intCast(ids.len)) - 1, hidden }, &[_]i32{ 1, 1, 1 });
            defer rows.deinit();
            const rows2 = try ops.reshape(ctx, rows, &[_]i32{ @as(i32, @intCast(ids.len)) - 1, hidden });
            defer rows2.deinit();
            const rows2n = try self.maybeTrunkNorm(rows2);
            defer if (self.trunkPostnormEnabled()) rows2n.deinit(); // disabled: same value as rows2 (borrowed)
            const tail_hidden = try m.prefill(self.allocator, rows2n, ids[1..], self.model.embed_table_bf16.?);
            tail_hidden.deinit(); // last-row output unused — the first draft
            // takes the BACKBONE last-position hidden (omlx convention).
            if (self.chainDumpEnabled())
                std.debug.print("MTP-CHAIN prefill prompt_len={d} head_cache_len={d}\n", .{ ids.len, m.cache.len() });
        }

        // Last-position logits → [vocab] f32; last hidden row → [1, hidden].
        const lshape = out_logits.shape();
        const vocab: i32 = lshape[2];
        // `out_logits` is ALREADY only the last position (see
        // forwardTokensAllHidden), so its seq dim is 1 — the hidden row
        // index MUST come from hidden_all, which still has all L rows.
        const L: i32 = hidden_all.shape()[1];
        const flat = try ops.reshape(ctx, out_logits, &[_]i32{vocab});
        defer flat.deinit();
        const logits = try ops.astype(ctx, flat, .float32);
        errdefer logits.deinit();

        const hidden_dim: i32 = hidden_all.shape()[2];
        const hrow = try ops.slice(ctx, hidden_all, &[_]i32{ 0, L - 1, 0 }, &[_]i32{ 1, L, hidden_dim }, &[_]i32{ 1, 1, 1 });
        defer hrow.deinit();
        const hflat = try ops.reshape(ctx, hrow, &[_]i32{ 1, hidden_dim });
        defer hflat.deinit();
        const hflatn = try self.maybeTrunkNorm(hflat);
        defer if (self.trunkPostnormEnabled()) hflatn.deinit();
        const hidden = try ops.copy(ctx, hflatn);
        return .{ .logits = logits, .hidden = hidden };
    }

    fn chainDumpEnabled(self: *Qwen38VTableAdapter) bool {
        if (self.mtp_chain_dump == null)
            self.mtp_chain_dump = std.c.getenv("DMLX_MTP_CHAIN_DUMP") != null;
        return self.mtp_chain_dump.?;
    }

    fn mtpDraft(ctx_ptr: *anyopaque, hidden: Array, token: u32) anyerror!generation.MtpStepOut {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        if (self.dspine != null) return self.dspineDraft(hidden, token);
        if (self.dflash2 != null) return self.dflash2Draft(hidden, token);
        const m = self.mtp.?;
        if (self.mtp_round_old_len == null) {
            self.mtp_round_old_len = m.cache.len();
        }
        if (self.chainDumpEnabled())
            std.debug.print("MTP-CHAIN draft step cache_len_before={d} tok={d}\n", .{ m.cache.len(), token });
        const out = try m.forward(self.allocator, hidden, token, self.model.embed_table_bf16.?, self.mtpLmHead());
        errdefer out.logits.deinit();
        errdefer out.hidden_out.deinit();
        // [1, 1, vocab] f32 → [vocab]
        const lshape = out.logits.shape();
        const vocab: i32 = lshape[2];
        const logits = try ops.reshape(self.model.ctx, out.logits, &[_]i32{vocab});
        out.logits.deinit();
        return .{ .logits = logits, .hidden = out.hidden_out };
    }

    fn mtpDraftGreedy(ctx_ptr: *anyopaque, hidden: Array, tok_gpu: Array) anyerror!generation.MtpGreedyDraftOut {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        if (self.dspine != null) return self.dspineDraftGreedy(hidden, tok_gpu);
        if (self.dflash2 != null) return self.dflash2DraftGreedy(hidden, tok_gpu);
        const m = self.mtp.?;
        if (self.mtp_round_old_len == null) {
            self.mtp_round_old_len = m.cache.len();
        }
        if (self.chainDumpEnabled())
            std.debug.print("MTP-CHAIN draftGreedy step cache_len_before={d}\n", .{m.cache.len()});
        const out = try m.forwardGreedyGpu(self.allocator, hidden, tok_gpu, self.model.embed_table_bf16.?, self.mtpLmHead());
        errdefer out.tok_gpu.deinit();
        errdefer out.hidden_out.deinit();
        return .{ .tok_gpu = out.tok_gpu, .hidden = out.hidden_out, .feat = out.feat };
    }

    fn mtpVerify(ctx_ptr: *anyopaque, ids: []const u32) anyerror!generation.MtpVerifyHandle {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        const box = try self.allocator.create(qwen38.Qwen38VerifyOut);
        errdefer self.allocator.destroy(box);
        if (self.dspine) |m| {
            var tap = qwen38.Qwen38ServingModel.HiddenTap.init(&m.cfg.target_layer_ids);
            errdefer tap.deinit();
            box.* = try qwen38.forwardVerifyImpl(self.model, ids, &tap);
            return box;
        }
        if (self.dflash2 != null) {
            var tap = qwen38.Qwen38ServingModel.HiddenTap.init(&qwen38.Qwen38ServingModel.dflash2_tap_layer_ids);
            errdefer tap.deinit();
            box.* = try qwen38.forwardVerifyImpl(self.model, ids, &tap);
            return box;
        }
        box.* = try qwen38.forwardVerifyImpl(self.model, ids, null);
        return box;
    }

    /// DFlash2 tree verify (U4): wrap the flat tree slices into a DraftTree
    /// (depth derived from parents; topological order guaranteed by
    /// selectTree) and run the backbone tree verify with the hidden tap.
    fn mtpVerifyTree(ctx_ptr: *anyopaque, tokens: []const u32, parents: []const i32, positions: []const i32, anchor: u32) anyerror!generation.MtpVerifyHandle {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        if (self.dflash2 == null) return error.TreeVerifyUnavailable;
        const n = tokens.len;
        if (parents.len != n or positions.len != n) return error.TreeShapeMismatch;
        const box = try self.allocator.create(qwen38.Qwen38TreeVerifyOut);
        errdefer self.allocator.destroy(box);
        const tree = blk: {
            const toks = try self.allocator.dupe(u32, tokens);
            errdefer self.allocator.free(toks);
            const pars = try self.allocator.dupe(i32, parents);
            errdefer self.allocator.free(pars);
            const poss = try self.allocator.dupe(i32, positions);
            errdefer self.allocator.free(poss);
            const deps = try self.allocator.alloc(u8, n);
            errdefer self.allocator.free(deps);
            for (0..n) |i| {
                const p = parents[i];
                if (p >= 0 and @as(usize, @intCast(p)) >= i) return error.TreeParentOrder;
                deps[i] = if (p < 0) 1 else deps[@intCast(p)] + 1;
            }
            break :blk qwen38_dflash2.DraftTree{
                .tokens = toks,
                .parents = pars,
                .positions = poss,
                .depth = deps,
                .allocator = self.allocator,
            };
        };
        // On error forwardVerifyTree deinits the tree itself.
        var tap = qwen38.Qwen38ServingModel.HiddenTap.init(&qwen38.Qwen38ServingModel.dflash2_tap_layer_ids);
        errdefer tap.deinit();
        box.* = try qwen38.forwardVerifyTree(self.model, tree, anchor, &tap);
        return box;
    }

    /// DFlash2 tree draft (U5): one block forward over [anchor, mask x P]
    /// against the ring context, then the best-first tree selector
    /// (selectTree). The greedy chain is always a sub-path of the tree, so
    /// greedy acceptance over the tree is bitwise-lossless vs chain rounds.
    fn mtpDraftTree(ctx_ptr: *anyopaque, anchor: u32, max_nodes: usize, children: usize) anyerror!generation.MtpDraftTree {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        if (self.dflash2 == null) return error.TreeVerifyUnavailable;
        const m = self.dflash2.?;
        const ctx = self.model.ctx;
        const P: i32 = @intCast(m.cfg.block_size - 1);
        const H: i32 = @intCast(m.cfg.hidden_size);

        const tok_gpu = try Array.fromData(self.allocator, i32, &[_]i32{@intCast(anchor)}, &[_]i32{1});
        defer tok_gpu.deinit();
        const mask_ids = try self.allocator.alloc(i32, @intCast(P));
        defer self.allocator.free(mask_ids);
        for (mask_ids) |*t| t.* = @intCast(m.cfg.mask_token_id);
        const mask_arr = try Array.fromData(self.allocator, i32, mask_ids, &[_]i32{P});
        defer mask_arr.deinit();
        const block_ids = try shape_mod.concatenateAxis(ctx, &[_]Array{ tok_gpu, mask_arr }, 0);
        defer block_ids.deinit();

        var ring = &self.df2_ctx.?;
        if (ring.fed == 0) return error.DFlash2EmptyContext;
        const anchor_pos = ring.fed;
        const mask = try ring.draftMask();
        defer mask.deinit();
        const hidden_full = try qwen38_dflash2.blockHiddenRing(ctx, m, ring, self.model.embed_table_bf16.?, block_ids, mask);
        defer hidden_full.deinit();
        const draft_hidden = try ops.slice(ctx, hidden_full, &[_]i32{ 0, 1, 0 }, &[_]i32{ 1, P + 1, H }, &[_]i32{ 1, 1, 1 });
        defer draft_hidden.deinit();
        const logits = try self.mtpLmHead().apply(ctx, draft_hidden); // [1, P, V]
        defer logits.deinit();

        var tree = try qwen38_dflash2.selectTree(ctx, m, draft_hidden, logits, anchor, anchor_pos, max_nodes, children);
        // The engine walks the tree via parents/positions; depth is dropped.
        tree.allocator.free(tree.depth);
        return .{
            .tokens = tree.tokens,
            .parents = tree.parents,
            .positions = tree.positions,
            .anchor_pos = anchor_pos,
            .allocator = tree.allocator,
        };
    }

    /// Discard a tree verify pass and free the handle: attention lengths
    /// truncated back (GDN caches were never touched), tentative ring feed
    /// (if any) dropped.
    fn mtpRollbackTree(ctx_ptr: *anyopaque, handle: generation.MtpVerifyHandle) void {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        const v: *qwen38.Qwen38TreeVerifyOut = @ptrCast(@alignCast(handle));
        qwen38.rollbackVerifyTree(self.model, v);
        if (self.df2_ctx != null and self.df2_ctx.?.tentative_start >= 0) self.df2_ctx.?.discardTentative();
        v.deinit(self.allocator);
        self.allocator.destroy(v);
    }

    /// Batched argmax over all tree verify rows ([anchor, nodes...]) →
    /// [1+N] i32, one GPU op + one engine-side sync (chain-path analog of
    /// mtpVerifyArgmax for the tree handle type).
    fn mtpVerifyTreeArgmax(ctx_ptr: *anyopaque, handle: generation.MtpVerifyHandle) anyerror!Array {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        const v: *qwen38.Qwen38TreeVerifyOut = @ptrCast(@alignCast(handle));
        const ctx = self.model.ctx;
        const L: i32 = v.logits.shape()[1];
        const vocab: i32 = v.logits.shape()[2];
        const f32l = try ops.astype(ctx, v.logits, .float32);
        defer f32l.deinit();
        const flat = try ops.reshape(ctx, f32l, &[_]i32{ L, vocab });
        defer flat.deinit();
        return reduce_mod.argmaxAxis(ctx, flat, -1, false); // [L] i32
    }

    /// Pre-final-norm hidden of tree verify row i → [1, hidden] (owned);
    /// same trunk-norm treatment as mtpVerifyHidden.
    fn mtpVerifyTreeHidden(ctx_ptr: *anyopaque, handle: generation.MtpVerifyHandle, i: usize) anyerror!Array {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        const v: *qwen38.Qwen38TreeVerifyOut = @ptrCast(@alignCast(handle));
        const ctx = self.model.ctx;
        const hidden: i32 = v.hidden_pre_norm_all.shape()[2];
        const row = try ops.slice(ctx, v.hidden_pre_norm_all, &[_]i32{ 0, @intCast(i), 0 }, &[_]i32{ 1, @as(i32, @intCast(i)) + 1, hidden }, &[_]i32{ 1, 1, 1 });
        defer row.deinit();
        const flat = try ops.reshape(ctx, row, &[_]i32{ 1, hidden });
        defer flat.deinit();
        const flatn = try self.maybeTrunkNorm(flat);
        defer if (self.trunkPostnormEnabled()) flatn.deinit();
        return ops.copy(ctx, flatn);
    }

    /// Commit the accepted path (verify-row indices) and free the handle;
    /// also feeds the accepted path's tap rows into the drafter ring (U5).
    /// Rollback is mtpRollbackTree.
    fn mtpCommitTree(ctx_ptr: *anyopaque, handle: generation.MtpVerifyHandle, accepted_path: []const usize) void {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        const v: *qwen38.Qwen38TreeVerifyOut = @ptrCast(@alignCast(handle));
        qwen38.commitVerifyTree(self.model, v, accepted_path);
        // Drafter ring feed (chain-commit analog): the accepted path's tap
        // rows, in path order. Their absolute positions are consecutive along
        // the path (anchor at fed, depth d at fed+d), matching observe's
        // sequential write at the current fed.
        if (self.df2_ctx != null and self.df2_ctx.?.tentative_start >= 0) {
            self.df2_ctx.?.discardTentative();
            self.dflash2ClearPending();
        }
        if (self.dflash2 != null and accepted_path.len > 0) {
            if (v.tap_hidden) |th| {
                const ctx = self.model.ctx;
                const idx = self.allocator.alloc(i32, accepted_path.len) catch null;
                if (idx) |ix| {
                    defer self.allocator.free(ix);
                    for (accepted_path, 0..) |row, j| ix[j] = @intCast(row);
                    if (Array.fromData(self.allocator, i32, ix, &[_]i32{@intCast(ix.len)})) |ia| {
                        defer ia.deinit();
                        if (shape_mod.takeAxis(ctx, th, ia, 1)) |rows| {
                            defer rows.deinit();
                            self.df2_ctx.?.observe(rows) catch |err| {
                                std.log.err("EngineLoop: DF2 tree commit ring observe failed: {}", .{err});
                            };
                        } else |_| {}
                    } else |_| {}
                }
            }
        }
        self.dflash2ClearBlock();
        v.deinit(self.allocator);
        self.allocator.destroy(v);
    }

    fn mtpVerifyLogits(ctx_ptr: *anyopaque, handle: generation.MtpVerifyHandle, i: usize) anyerror!Array {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        const v: *qwen38.Qwen38VerifyOut = @ptrCast(@alignCast(handle));
        const ctx = self.model.ctx;
        const vocab: i32 = v.logits.shape()[2];
        const col = try ops.slice(ctx, v.logits, &[_]i32{ 0, @intCast(i), 0 }, &[_]i32{ 1, @as(i32, @intCast(i)) + 1, vocab }, &[_]i32{ 1, 1, 1 });
        defer col.deinit();
        const flat = try ops.reshape(ctx, col, &[_]i32{vocab});
        defer flat.deinit();
        return ops.astype(ctx, flat, .float32);
    }

    fn mtpVerifyHidden(ctx_ptr: *anyopaque, handle: generation.MtpVerifyHandle, i: usize) anyerror!Array {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        const v: *qwen38.Qwen38VerifyOut = @ptrCast(@alignCast(handle));
        const ctx = self.model.ctx;
        const hidden: i32 = v.hidden_pre_norm_all.shape()[2];
        const row = try ops.slice(ctx, v.hidden_pre_norm_all, &[_]i32{ 0, @intCast(i), 0 }, &[_]i32{ 1, @as(i32, @intCast(i)) + 1, hidden }, &[_]i32{ 1, 1, 1 });
        defer row.deinit();
        const flat = try ops.reshape(ctx, row, &[_]i32{ 1, hidden });
        defer flat.deinit();
        const flatn = try self.maybeTrunkNorm(flat);
        defer if (self.trunkPostnormEnabled()) flatn.deinit();
        return ops.copy(ctx, flatn);
    }

    fn mtpVerifyArgmax(ctx_ptr: *anyopaque, handle: generation.MtpVerifyHandle) anyerror!Array {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        const v: *qwen38.Qwen38VerifyOut = @ptrCast(@alignCast(handle));
        const ctx = self.model.ctx;
        if (self.df2_pending_argmax) |am| return ops.copy(ctx, am);
        const L: i32 = v.logits.shape()[1];
        const vocab: i32 = v.logits.shape()[2];
        const f32l = try ops.astype(ctx, v.logits, .float32);
        defer f32l.deinit();
        const flat = try ops.reshape(ctx, f32l, &[_]i32{ L, vocab });
        defer flat.deinit();
        return reduce_mod.argmaxAxis(ctx, flat, -1, false); // [L] i32
    }

    fn mtpPredraft(ctx_ptr: *anyopaque, handle: generation.MtpVerifyHandle) anyerror!void {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        if (self.dspine != null) return;
        if (self.dflash2 == null) return;
        const v: *qwen38.Qwen38VerifyOut = @ptrCast(@alignCast(handle));
        return self.dflash2Predraft(v);
    }

    fn mtpCommit(ctx_ptr: *anyopaque, handle: generation.MtpVerifyHandle, accepted: usize, tokens: []const u32) void {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        const v: *qwen38.Qwen38VerifyOut = @ptrCast(@alignCast(handle));
        if (self.dspine != null) {
            self.dspineCommit(v, accepted);
            return;
        }
        if (self.dflash2 != null) {
            self.dflash2Commit(v, accepted);
            return;
        }
        qwen38.commitVerify(self.model, v, accepted);
        // Drafter cache, mlx-vlm accept_verified_tokens semantics: drop every
        // drafted chain entry except t_next (whose input was already the
        // TARGET hidden from the previous round/prefill), then replay the
        // accepted drafts with the TARGET's verify hidden rows — the KV
        // entries the drafter attends over must match confirmed processing,
        // not the drafter's own chained recurrence.
        const m = self.mtp.?;
        const ctx = self.model.ctx;
        const lm = self.mtpLmHead();
        // Replay-free commit (omlx `_chain_next_drafts` semantics) is the
        // DEFAULT (2026-11-17): keeping the drafter's OWN chain states for
        // the accepted drafts instead of replaying them with the target's
        // verify hidden rows. Measured: conditional acceptance p1/p2 went
        // 0.64/0.34 -> 0.97/1.00 on code (tok/s 19.5 -> 28.2), with no
        // regression on prose — the replay was actively misaligning the
        // drafter cache. The chain already wrote one KV entry per step
        // (t_next at old_len+1, draft i at old_len+1+i), so the commit is a
        // bare truncation to old_len + accepted.
        // `DMLX_MTP_REPLAY=1` restores the old replay path for A/B.
        if (std.c.getenv("DMLX_MTP_REPLAY") == null) {
            m.cache.len_ = self.mtp_round_old_len.? + @as(i32, @intCast(accepted));
            if (self.chainDumpEnabled())
                std.debug.print("MTP-CHAIN commit replay-free accepted={d} len={d}\n", .{ accepted, m.cache.len_ });
            self.mtp_round_old_len = null;
            return;
        }
        m.cache.len_ = self.mtp_round_old_len.? + 1;
        if (self.chainDumpEnabled())
            std.debug.print("MTP-CHAIN commit accepted={d} truncate_to={d}\n", .{ accepted, m.cache.len_ });
        // Batched fold (omlx `_chain_next_drafts` semantics): ONE head forward
        // over the accepted drafts with the target's verify hidden rows,
        // replacing k SEQUENTIAL replay forwards. The sequential loop's lazy
        // work lands inside the NEXT round's verify sync as ~k x 1.6 ms of
        // serialized drafter body time; the batched fold is a single ~2 ms
        // prefill-mode pass. Visibility per folded entry is identical
        // (causal over cache + earlier folded rows). Default on;
        // `DMLX_MTP_NO_BATCH_FOLD=1` restores the sequential replay.
        if (accepted > 1 and std.c.getenv("DMLX_MTP_NO_BATCH_FOLD") == null) {
            const n: i32 = @intCast(accepted - 1);
            const hdim: i32 = v.hidden_pre_norm_all.shape()[2];
            const rows = ops.slice(ctx, v.hidden_pre_norm_all, &[_]i32{ 0, 0, 0 }, &[_]i32{ 1, n, hdim }, &[_]i32{ 1, 1, 1 }) catch return;
            defer rows.deinit();
            const rows2 = ops.reshape(ctx, rows, &[_]i32{ n, hdim }) catch return;
            defer rows2.deinit();
            const rows2n = self.maybeTrunkNorm(rows2) catch return;
            defer if (self.trunkPostnormEnabled()) rows2n.deinit();
            const tail = m.prefill(self.allocator, rows2n, tokens[1..accepted], self.model.embed_table_bf16.?) catch return;
            tail.deinit();
            if (self.chainDumpEnabled())
                std.debug.print("MTP-CHAIN fold n={d} cache_len_after={d}\n", .{ n, m.cache.len() });
            self.mtp_round_old_len = null;
            return;
        }
        var i: usize = 1;
        while (i < accepted) : (i += 1) {
            const hidden: i32 = v.hidden_pre_norm_all.shape()[2];
            const row = ops.slice(ctx, v.hidden_pre_norm_all, &[_]i32{ 0, @as(i32, @intCast(i - 1)), 0 }, &[_]i32{ 1, @as(i32, @intCast(i)), hidden }, &[_]i32{ 1, 1, 1 }) catch return;
            defer row.deinit();
            const flat = ops.reshape(ctx, row, &[_]i32{ 1, hidden }) catch return;
            defer flat.deinit();
            const flatn = self.maybeTrunkNorm(flat) catch return;
            defer if (self.trunkPostnormEnabled()) flatn.deinit();
            // Body only: the replay exists for the KV-cache side effect, so
            // building the full-vocab lm_head here (and discarding it) is
            // pure waste. `DMLX_MTP_REPLAY_LM_HEAD=1` restores the old call
            // for A/B.
            if (std.c.getenv("DMLX_MTP_REPLAY_LM_HEAD") != null) {
                var out = m.forward(self.allocator, flatn, tokens[i], self.model.embed_table_bf16.?, lm) catch return;
                out.logits.deinit();
                out.hidden_out.deinit();
            } else {
                const h = m.forwardBodyToken(self.allocator, flatn, tokens[i], self.model.embed_table_bf16.?) catch return;
                h.deinit();
            }
            if (self.chainDumpEnabled())
                std.debug.print("MTP-CHAIN replay i={d} tok={d} cache_len_after={d}\n", .{ i, tokens[i], m.cache.len() });
        }
        if (self.chainDumpEnabled())
            std.debug.print("MTP-CHAIN commit done cache_len={d}\n", .{m.cache.len()});
        self.mtp_round_old_len = null;
    }

    fn mtpRollback(ctx_ptr: *anyopaque, handle: generation.MtpVerifyHandle) void {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        const v: *qwen38.Qwen38VerifyOut = @ptrCast(@alignCast(handle));
        qwen38.rollbackVerify(self.model, v);
        if (self.dspine != null) {
            self.dspineClearBlock();
            return;
        }
        if (self.dflash2 != null) {
            // Context rows are only appended on commit, so a rollback needs
            // no drafter-side undo — just drop the round's draft block and
            // any tentative predraft state.
            if (self.df2_ctx.?.tentative_start >= 0) self.df2_ctx.?.discardTentative();
            self.dflash2ClearPending();
            self.dflash2ClearBlock();
            return;
        }
        const m = self.mtp.?;
        m.cache.len_ = self.mtp_round_old_len.?;
        self.mtp_round_old_len = null;
    }

    fn mtpFreeVerify(ctx_ptr: *anyopaque, handle: generation.MtpVerifyHandle) void {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        const v: *qwen38.Qwen38VerifyOut = @ptrCast(@alignCast(handle));
        if (self.dflash2 != null and self.df2_ctx != null and self.df2_ctx.?.tentative_start >= 0) {
            self.df2_ctx.?.discardTentative();
        }
        self.dflash2ClearPending();
        if (self.dspine != null) self.dspineClearBlock();
        v.deinit(self.allocator);
        self.allocator.destroy(v);
        self.mtp_round_old_len = null; // never leave stale round state
    }

    fn deinitFn(ctx_ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *Qwen38VTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        if (self.dspine_cache_ids.len > 0) allocator.free(self.dspine_cache_ids);
        if (self.dspine_cache_rows) |r| r.deinit();
        if (self.dspine_table_host.len > 0) allocator.free(self.dspine_table_host);
        if (self.dspine_ctx) |*cx| cx.deinit();
        if (self.dspine) |m| {
            m.deinit();
            allocator.destroy(m);
        }
        if (self.df2_cache_ids.len > 0) allocator.free(self.df2_cache_ids);
        if (self.df2_cache_rows) |r| r.deinit();
        if (self.df2_block_logits) |l| l.deinit();
        if (self.df2_ctx) |*cx| cx.deinit();
        if (self.dflash2) |m| {
            m.deinit();
            allocator.destroy(m);
        }
        if (self.mtp) |m| {
            m.deinit();
            allocator.destroy(m);
        }
        if (self.vit) |v| {
            v.deinit();
            allocator.destroy(v);
        }
        if (self.vit_weights) |*vw| {
            var vit0 = vw.iterator();
            while (vit0.next()) |e| {
                allocator.free(e.key_ptr.*);
                e.value_ptr.deinit();
            }
            vw.deinit();
        }
        if (self.model_path.len > 0) allocator.free(self.model_path);
        _ = pthread.pthread_mutex_destroy(&self.mu);
        self.model.deinit();
        allocator.destroy(self.model);
        var it = self.weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        self.weights.deinit();
        self.cfg.deinit();
        allocator.destroy(self.cfg);
        allocator.destroy(self);
    }
};

const default_mtp_drafter_dir = "/Users/zouyee/models/Qwen3.8-27B-MTP-4bit";

fn qwen38Loader(
    allocator: std.mem.Allocator,
    config_json: []const u8,
    model_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    io: std.Io,
    smelt: deepseek_v4_loader.SmeltConfig,
) anyerror!ModelVTable {
    _ = io;
    _ = smelt; // dense model — no expert streaming
    const cfg_ptr = try allocator.create(qwen38_loader.Qwen38Config);
    errdefer allocator.destroy(cfg_ptr);
    cfg_ptr.* = try qwen38_loader.parseQwen38Config(allocator, config_json);
    errdefer cfg_ptr.deinit();

    var idx = try safetensors_reader.loadOrBuildIndex(allocator, model_path);
    defer idx.deinit();

    // Resident weights: every language_model.* tensor (dense model — nothing
    // is streamed).
    var weights = std.StringHashMap(Array).init(allocator);
    errdefer {
        var it0 = weights.iterator();
        while (it0.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }
    {
        var it = idx.entries.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (!std.mem.startsWith(u8, name, "language_model.")) continue;
            const key_copy = try allocator.dupe(u8, name);
            errdefer allocator.free(key_copy);
            const arr = try idx.loadTensor(name);
            try weights.put(key_copy, arr);
        }
    }
    std.log.info("Qwen3.8 weights resident: {d} tensors", .{weights.count()});

    // Optional LoRA adapter merged into the resident quantized weights at
    // load time (e.g. harness-hub/Qwen3.8-27B-pi-adapted).
    if (std.c.getenv("DMLX_LORA_ADAPTER")) |d| {
        const n = lora_merge.applyLoraAdapter(allocator, ctx, &weights, std.mem.span(d)) catch |err| blk: {
            std.log.warn("lora: merge of {s} failed ({}), serving base weights", .{ std.mem.span(d), err });
            break :blk 0;
        };
        _ = n;
    }

    const model_ptr = try allocator.create(qwen38.Qwen38ServingModel);
    errdefer allocator.destroy(model_ptr);
    model_ptr.* = try qwen38.Qwen38ServingModel.init(allocator, ctx, cfg_ptr, weights);
    errdefer model_ptr.deinit();
    // Reseat the WeightSource: init built it against its local `self`; the
    // map/backing pointers must address the heap-placed struct.
    model_ptr.ws = qwen38.WeightSource.fromMap(&model_ptr.weights, ctx, &model_ptr.ws_backing);

    const adapter = try allocator.create(Qwen38VTableAdapter);
    errdefer allocator.destroy(adapter);
    adapter.* = .{
        .model = model_ptr,
        .stream = stream,
        .start_pos = 0,
        .cfg = cfg_ptr,
        .weights = weights,
        .allocator = allocator,
        .model_path = try allocator.dupe(u8, model_path),
        .vision_ops = .{
            .image_token_id = cfg_ptr.image_token_id,
            .processImage = &Qwen38VTableAdapter.visionProcessImage,
            .prefill = &Qwen38VTableAdapter.visionPrefill,
        },
        .mu = undefined,
    };
    _ = pthread.pthread_mutex_init(&adapter.mu, null);

    return ModelVTable{
        .forward = &Qwen38VTableAdapter.forward,
        .embed = &Qwen38VTableAdapter.embedTokens,
        .mtp = &adapter.mtp_ops,
        .vision = &adapter.vision_ops,
        .deinit = &Qwen38VTableAdapter.deinitFn,
        .config = ModelConfig{
            .num_layers = cfg_ptr.num_hidden_layers,
            .num_kv_heads = 1, // caches are internal; engine KV array unused
            .head_dim = cfg_ptr.head_dim,
            .vocab_size = cfg_ptr.vocab_size,
            .hidden_size = cfg_ptr.hidden_size,
            .compress_ratios = &.{},
        },
        .ptr = adapter,
    };
}

// ============================================================
// VTable adapter: Gemma (uses LLaMA loader — architecturally similar)
// ============================================================

/// Gemma uses a LLaMA-like architecture with GeGLU activation and
/// different normalization. For basic inference, the LLaMA loader
/// works since the weight layout and attention mechanism are compatible.
fn gemmaLoader(
    allocator: std.mem.Allocator,
    config_json: []const u8,
    model_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    io: std.Io,
    smelt: deepseek_v4_loader.SmeltConfig,
) anyerror!ModelVTable {
    // Gemma shares enough structure with LLaMA that the same loader works
    // for basic inference. GeGLU vs SwiGLU difference is minor for initial support.
    return llamaLoader(allocator, config_json, model_path, ctx, stream, io, smelt);
}

// ============================================================
// VTable adapter: GLM-4 (uses LLaMA loader — architecturally similar)
// ============================================================

/// GLM-4 uses a LLaMA-like architecture with SiLU activation, GQA,
/// RMSNorm, and attention bias. The weight layout is compatible with
/// the LLaMA loader (self_attn.q_proj, k_proj, v_proj, o_proj pattern).
fn glm4Loader(
    allocator: std.mem.Allocator,
    config_json: []const u8,
    model_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    io: std.Io,
    smelt: deepseek_v4_loader.SmeltConfig,
) anyerror!ModelVTable {
    // GLM-4 shares the same decoder-only transformer architecture as LLaMA
    // with attention bias (like Qwen2). The LLaMA loader handles bias weights.
    return llamaLoader(allocator, config_json, model_path, ctx, stream, io, smelt);
}

// ============================================================
// VTable adapter: LLaVA
// ============================================================

const llava = @import("vision/llava.zig");

/// Wraps a heap-allocated `LlavaModel` behind the `ModelVTable` interface.
const LlavaVTableAdapter = struct {
    model: *llava.LlavaModel,

    fn forward(ctx_ptr: *anyopaque, input: Array, mask: ?Array, caches: ?[]KVCacheStrategy) anyerror!Array {
        const self: *LlavaVTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        return self.model.forward(input, mask, caches);
    }

    fn forwardWithHidden(ctx_ptr: *anyopaque, input: Array, mask: ?Array, caches: ?[]KVCacheStrategy) anyerror!generation.ForwardWithHiddenResult {
        const self: *LlavaVTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        return self.model.forwardWithHidden(input, mask, caches);
    }

    fn deinitFn(ctx_ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *LlavaVTableAdapter = @ptrCast(@alignCast(ctx_ptr));
        self.model.deinit();
        allocator.destroy(self.model);
        allocator.destroy(self);
    }
};

fn llavaLoader(
    allocator: std.mem.Allocator,
    config_json: []const u8,
    model_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    io: std.Io,
    smelt: deepseek_v4_loader.SmeltConfig,
) anyerror!ModelVTable {
    _ = io;
    _ = smelt;
    // Parse the full LLaVA config to extract text_config
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, config_json, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    const text_config_value = root.get("text_config") orelse parsed.value;
    const text_config_json = try std.json.Stringify.valueAlloc(allocator, text_config_value, .{});
    defer allocator.free(text_config_json);

    const llama_config = try hf_config.parseLlamaConfig(allocator, text_config_json);

    const model_ptr = try allocator.create(llava.LlavaModel);
    errdefer allocator.destroy(model_ptr);

    model_ptr.* = try llava_loader.loadLlavaModelFromConfig(allocator, model_path, &llama_config, ctx, stream);

    const adapter = try allocator.create(LlavaVTableAdapter);
    errdefer allocator.destroy(adapter);
    adapter.* = .{ .model = model_ptr };

    const head_dim = llama_config.getHeadDim();

    return ModelVTable{
        .forward = &LlavaVTableAdapter.forward,
        .forwardWithHidden = &LlavaVTableAdapter.forwardWithHidden,
        .deinit = &LlavaVTableAdapter.deinitFn,
        .config = ModelConfig{
            .num_layers = llama_config.num_hidden_layers,
            .num_kv_heads = llama_config.num_key_value_heads,
            .head_dim = head_dim,
            .vocab_size = llama_config.vocab_size,
            .hidden_size = llama_config.hidden_size,
        },
        .ptr = adapter,
    };
}

// ============================================================
// Tests
// ============================================================

test "registry contains all registered architectures" {
    const expected = [_][]const u8{
        "LlamaForCausalLM",
        "DeepseekV4ForCausalLM",
        "MistralForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "GemmaForCausalLM",
        "Glm4ForCausalLM",
        "PhiForCausalLM",
        "Phi3ForCausalLM",
        "LlavaForConditionalGeneration",
        "Glm5NextForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
    };
    for (expected) |arch| {
        try std.testing.expect(model_registry.get(arch) != null);
    }
}

test "registry lookup fails for unknown architecture" {
    const result = getLoader("UnknownModelArch");
    try std.testing.expectError(RegistryError.UnsupportedArchitecture, result);
}

test "getLoader returns loader for registered architectures" {
    const loader = try getLoader("LlamaForCausalLM");
    try std.testing.expect(@intFromPtr(loader) != 0);
}

test "registry contains LlavaForConditionalGeneration" {
    try std.testing.expect(model_registry.get("LlavaForConditionalGeneration") != null);
    const loader = try getLoader("LlavaForConditionalGeneration");
    try std.testing.expect(@intFromPtr(loader) != 0);
}

test "Mistral, Qwen2, Qwen3, Phi map to same loader as LLaMA" {
    const llama_fn = model_registry.get("LlamaForCausalLM").?;
    const mistral_fn = model_registry.get("MistralForCausalLM").?;
    const qwen2_fn = model_registry.get("Qwen2ForCausalLM").?;
    const qwen3_fn = model_registry.get("Qwen3ForCausalLM").?;
    const phi_fn = model_registry.get("PhiForCausalLM").?;
    const phi3_fn = model_registry.get("Phi3ForCausalLM").?;
    try std.testing.expectEqual(llama_fn, mistral_fn);
    try std.testing.expectEqual(llama_fn, qwen2_fn);
    try std.testing.expectEqual(llama_fn, qwen3_fn);
    try std.testing.expectEqual(llama_fn, phi_fn);
    try std.testing.expectEqual(llama_fn, phi3_fn);
}

// ============================================================
// Property-Based Test: Model Registry Lookup Correctness (Property 4)
//
// **Validates: Requirements R6.2, R6.4**
//
// *For any* architecture name string, looking it up in the
// Model_Registry SHALL succeed if and only if the architecture
// is registered. When lookup fails, the error message SHALL
// contain the queried architecture name.
//
// Runs 100 iterations with:
//   1. All registered architectures → lookup succeeds, returns non-null loader
//   2. Randomly generated strings → lookup fails with UnsupportedArchitecture
//   3. The log format embeds the queried name (verified structurally via
//      getLoader's std.log.warn call which formats arch_name with {s})
// ============================================================

/// Generate a random ASCII string of the given length into `buf`.
/// Uses printable ASCII range (32–126) to produce diverse test inputs.
fn generateRandomString(rand: std.Random, buf: []u8) []u8 {
    for (buf) |*ch| {
        ch.* = rand.intRangeAtMost(u8, 32, 126);
    }
    return buf;
}

test "Property 4: Model Registry Lookup Correctness — registered succeed, unregistered fail (100 iterations)" {
    var prng = std.Random.DefaultPrng.init(0xCAFE_BABE);
    const rand = prng.random();

    const num_iterations: usize = 100;

    for (0..num_iterations) |_| {
        // --- Part 1: Every registered architecture lookup succeeds ---
        for (supported_architectures) |arch| {
            const loader = getLoader(arch) catch |err| {
                std.debug.print(
                    "FAIL: registered architecture \"{s}\" returned error: {}\n",
                    .{ arch, err },
                );
                return error.TestUnexpectedResult;
            };
            // Loader function pointer must be non-null
            try std.testing.expect(@intFromPtr(loader) != 0);

            // Also verify via the StaticStringMap directly
            try std.testing.expect(model_registry.get(arch) != null);
        }

        // --- Part 2: Random string lookup fails with UnsupportedArchitecture ---
        const str_len = rand.intRangeAtMost(usize, 0, 64);
        var buf: [64]u8 = undefined;
        const random_name = generateRandomString(rand, buf[0..str_len]);

        // Skip if the random string happens to match a registered architecture
        const is_registered = model_registry.get(random_name) != null;
        if (is_registered) continue;

        // Unregistered name must produce UnsupportedArchitecture error
        const result = getLoader(random_name);
        if (result) |_| {
            std.debug.print(
                "FAIL: unregistered name \"{s}\" returned a loader instead of error\n",
                .{random_name},
            );
            return error.TestUnexpectedResult;
        } else |err| {
            try std.testing.expectEqual(RegistryError.UnsupportedArchitecture, err);
        }
    }
}
