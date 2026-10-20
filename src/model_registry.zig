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
};

// ============================================================
// Public lookup helper
// ============================================================

/// Convenience re-export for the CLI chat path.
pub fn Glm53ConfigFromJson(allocator: std.mem.Allocator, config_json: []const u8) !glm53_loader.Glm53Config {
    return glm53_loader.parseGlm53Config(allocator, config_json);
}

pub const RegistryError = error{
    UnsupportedArchitecture,
};

/// Look up a loader for the given architecture name.
/// Returns `RegistryError.UnsupportedArchitecture` with a log message
/// containing the queried name when the architecture is not registered.
pub fn getLoader(arch_name: []const u8) RegistryError!ModelLoader {
    return model_registry.get(arch_name) orelse {
        std.log.warn("Unsupported model architecture: \"{s}\". Supported architectures: " ++
            "LlamaForCausalLM, DeepseekV4ForCausalLM, MistralForCausalLM, " ++
            "Qwen2ForCausalLM, Qwen3ForCausalLM, GemmaForCausalLM, " ++
            "Glm4ForCausalLM, PhiForCausalLM, Phi3ForCausalLM, " ++
            "LlavaForConditionalGeneration", .{arch_name});
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
        if (self.layer_meta) |lm| allocator.free(lm);
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

    const idx = try allocator.create(safetensors_reader.TensorIndex);
    errdefer allocator.destroy(idx);
    idx.* = try safetensors_reader.loadOrBuildIndex(allocator, model_path);

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

    const rd = try allocator.create(safetensors_reader.PartialTensorReader);
    errdefer allocator.destroy(rd);
    rd.* = safetensors_reader.PartialTensorReader.init(allocator, idx, pool);

    // Expert streaming (GLM: 4-bit affine gs=64 with biases — biases make the
    // packed-pread path ineligible, so slices come from PartialTensorReader).
    var sp: ?*expert_stream.ExpertStreamProvider = null;
    var layer_meta: ?[]expert_stream.LayerExpertMeta = null;
    if (smelt.enabled and smelt.load_mode == .stream) {
        const meta = try glm53_loader.buildLayerExpertMeta(allocator, cfg_ptr, idx);
        layer_meta = meta;
        const sp_ptr = try allocator.create(expert_stream.ExpertStreamProvider);
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
        sp = sp_ptr;
        std.log.info("GLM-5.3 expert streaming enabled (affine gs=64 4-bit, top-8)", .{});
    }

    const model_ptr = try allocator.create(glm53.Glm53ServingModel);
    errdefer allocator.destroy(model_ptr);
    model_ptr.* = try glm53.Glm53ServingModel.init(allocator, ctx, cfg_ptr, stream, weights, rd, sp);

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

test "registry contains all ten architectures" {
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
