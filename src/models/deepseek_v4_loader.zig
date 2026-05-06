/// DeepSeek-V4-Flash weight loader.
///
/// Supports:
/// - Single-file model.safetensors
/// - Sharded model-00001-of-NNNNN.safetensors with index.json
/// - Direct DeepSeek weight naming (no HF mapping needed)
/// - mlx-community HF naming (model.* prefix, gate_proj/up_proj/down_proj)
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const ops = @import("../ops.zig");
const nn = @import("../ops/nn.zig");
const creation = @import("../ops/creation.zig");
const io = @import("../io/mlx_io.zig");
const safetensors_reader = @import("../io/safetensors_reader.zig");
const deepseek_v4 = @import("deepseek_v4.zig");
const quantize_mod = @import("../quantize.zig");
const shape_mod = @import("../ops/shape.zig");
const cmp_mod = @import("../ops/comparison.zig");
const math_mod = @import("../ops/math.zig");
const kvcache = @import("../kvcache.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const DSV4Config = deepseek_v4.DSV4Config;
const DSV4Model = deepseek_v4.DSV4Model;

/// Consume a weight key and its associated quantized metadata (.scales, .biases) from the HashMap.
/// This prevents "Unused weight" warnings for quantized models where each weight has 3 keys.
fn consumeWeightKey(allocator: std.mem.Allocator, weights: *std.StringHashMap(Array), base_name: []const u8) void {
    // Try removing base_name itself (for keys like "attn.attn_sink", "attn.compressor.ape")
    if (weights.fetchRemove(base_name)) |kv| allocator.free(kv.key);

    // Try removing base_name.weight
    const suffixes = [_][]const u8{ ".weight", ".scales", ".biases" };
    for (suffixes) |suffix| {
        // Build key: base_name + suffix
        const key = std.fmt.allocPrint(allocator, "{s}{s}", .{ base_name, suffix }) catch continue;
        defer allocator.free(key);
        if (weights.fetchRemove(key)) |kv| allocator.free(kv.key);
    }
}

/// Check if a HF weight name is an expert weight (switch_mlp or ffn.experts).
fn isExpertWeight(name: []const u8) bool {
    return std.mem.indexOf(u8, name, "switch_mlp") != null or
        (std.mem.indexOf(u8, name, "ffn.experts.") != null and
            std.mem.indexOf(u8, name, "shared_experts") == null);
}

/// Entry with name and info for sequential I/O grouping.
const EntryWithName = struct {
    name: []const u8,
    info: safetensors_reader.TensorInfo,
};

/// Slice a fused expert tensor [n_experts, ...] to keep only experts marked true in the mask.
/// Returns a new tensor [n_loaded, ...] containing only the selected expert rows.
fn sliceFusedExperts(allocator: std.mem.Allocator, tensor: Array, mask: []const bool, ctx: EagerContext) !Array {
    // Count loaded experts and build index array
    var count: usize = 0;
    for (mask) |m| {
        if (m) count += 1;
    }
    if (count == 0 or count == mask.len) return tensor; // No slicing needed

    // Build indices array of loaded expert IDs
    var indices = try allocator.alloc(i32, count);
    defer allocator.free(indices);
    var idx: usize = 0;
    for (mask, 0..) |m, i| {
        if (m) {
            indices[idx] = @intCast(i);
            idx += 1;
        }
    }

    // Use mlx_take_axis to gather selected expert rows along axis 0
    const indices_arr = try Array.fromData(allocator, i32, indices, &[_]i32{@intCast(count)});
    defer indices_arr.deinit();
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_take_axis(&res, tensor.inner, indices_arr.inner, 0, ctx.stream.inner));
    return Array.fromHandle(res);
}

/// Parse expert index from HF weight name.
fn parseExpertIndexFromHF(name: []const u8) ?usize {
    if (std.mem.indexOf(u8, name, "switch_mlp") != null) return null;
    const prefix = "ffn.experts.";
    const idx = std.mem.indexOf(u8, name, prefix) orelse return null;
    const start = idx + prefix.len;
    var end = start;
    while (end < name.len and name[end] >= '0' and name[end] <= '9') end += 1;
    if (end == start) return null;
    return std.fmt.parseInt(usize, name[start..end], 10) catch null;
}

/// Dequantize a weight if it has associated .scales in the HashMap.
/// Returns the dequantized weight (or original if not quantized).
fn dequantIfNeeded(
    allocator: std.mem.Allocator,
    weights: *std.StringHashMap(Array),
    base_name: []const u8,
    weight: Array,
    config: *const DSV4Config,
    ctx: EagerContext,
) !Array {
    const scales_key = try std.fmt.allocPrint(allocator, "{s}.scales", .{base_name});
    defer allocator.free(scales_key);
    const scales = weights.get(scales_key) orelse {
        std.log.warn("dequantIfNeeded: scales not found for {s} (key: {s})", .{ base_name, scales_key });
        return weight;
    };

    const biases_key = try std.fmt.allocPrint(allocator, "{s}.biases", .{base_name});
    defer allocator.free(biases_key);
    const biases = weights.get(biases_key);

    const null_array: c.c.mlx_array = .{ .ctx = null };
    const biases_inner = if (biases) |b| b.inner else null_array;
    const is_mxfp4 = biases == null;
    const gs = if (is_mxfp4) 32 else config.quantize_default_group_size;
    const bits_val: i32 = @intCast(if (config.quantize_default_bits > 0) config.quantize_default_bits else 4);
    const opt_group: c.c.mlx_optional_int = .{ .value = gs, .has_value = true };
    const opt_bits: c.c.mlx_optional_int = .{ .value = bits_val, .has_value = true };
    const no_dtype: c.c.mlx_optional_dtype = .{ .value = c.c.MLX_BFLOAT16, .has_value = true };

    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_dequantize(&res, weight.inner, scales.inner, biases_inner, opt_group, opt_bits, if (is_mxfp4) "mxfp4" else "affine", null_array, no_dtype, ctx.stream.inner));

    if (weights.fetchRemove(scales_key)) |kv| allocator.free(kv.key);
    if (weights.fetchRemove(biases_key)) |kv| allocator.free(kv.key);

    return Array.fromHandle(res);
}

/// Map HF/mlx-lm weight names to our internal V4 naming convention.
fn mapV4WeightName(allocator: std.mem.Allocator, hf_name: []const u8) !?[]const u8 {
    // If it doesn't start with "model.", it's already internal format
    if (!std.mem.startsWith(u8, hf_name, "model.")) {
        // Handle lm_head → head mapping (lm_head is at top level, no model. prefix)
        if (std.mem.startsWith(u8, hf_name, "lm_head.")) {
            // lm_head.weight → head.weight, lm_head.scales → head.weight.scales, etc.
            const suffix = hf_name["lm_head.".len..];
            if (std.mem.eql(u8, suffix, "weight")) {
                return try allocator.dupe(u8, "head.weight");
            } else if (std.mem.eql(u8, suffix, "scales")) {
                return try allocator.dupe(u8, "head.weight.scales");
            } else if (std.mem.eql(u8, suffix, "biases")) {
                return try allocator.dupe(u8, "head.weight.biases");
            }
        }
        return null; // Already internal format
    }

    // Strip "model." prefix
    const stripped = hf_name["model.".len..];

    // embed_tokens → embed
    if (std.mem.startsWith(u8, stripped, "embed_tokens.")) {
        const suffix = stripped["embed_tokens.".len..];
        if (std.mem.eql(u8, suffix, "weight")) {
            return try allocator.dupe(u8, "embed.weight");
        } else if (std.mem.eql(u8, suffix, "scales")) {
            return try allocator.dupe(u8, "embed.weight.scales");
        } else if (std.mem.eql(u8, suffix, "biases")) {
            return try allocator.dupe(u8, "embed.weight.biases");
        }
        return try allocator.dupe(u8, stripped);
    }

    // norm.weight → norm.weight (just strip model. prefix)
    if (std.mem.eql(u8, stripped, "norm.weight")) {
        return try allocator.dupe(u8, "norm.weight");
    }

    // hc_head.* → hc_head.* (just strip model. prefix)
    if (std.mem.startsWith(u8, stripped, "hc_head.")) {
        return try allocator.dupe(u8, stripped);
    }

    // layers.N.* — need to map component names
    if (std.mem.startsWith(u8, stripped, "layers.")) {
        return try mapV4LayerWeight(allocator, stripped);
    }

    // Default: just strip model. prefix
    return try allocator.dupe(u8, stripped);
}

/// Map layer-level weight names.
/// Input: "layers.N.component.subcomponent..."
/// Handles MLP name mapping (gate_proj→w1, up_proj→w3, down_proj→w2)
/// and e_score_correction_bias → gate.bias
fn mapV4LayerWeight(allocator: std.mem.Allocator, name: []const u8) ![]const u8 {
    // Replace gate_proj→w1, up_proj→w3, down_proj→w2 in shared_experts and switch_mlp
    var result = try allocator.dupe(u8, name);
    errdefer allocator.free(result);

    // Check for MLP name patterns and replace
    const replacements = [_]struct { from: []const u8, to: []const u8 }{
        .{ .from = "shared_experts.gate_proj.", .to = "shared_experts.w1." },
        .{ .from = "shared_experts.up_proj.", .to = "shared_experts.w3." },
        .{ .from = "shared_experts.down_proj.", .to = "shared_experts.w2." },
        .{ .from = "shared_experts.gate_proj", .to = "shared_experts.w1" },
        .{ .from = "shared_experts.up_proj", .to = "shared_experts.w3" },
        .{ .from = "shared_experts.down_proj", .to = "shared_experts.w2" },
        .{ .from = "switch_mlp.gate_proj.", .to = "switch_mlp.w1." },
        .{ .from = "switch_mlp.up_proj.", .to = "switch_mlp.w3." },
        .{ .from = "switch_mlp.down_proj.", .to = "switch_mlp.w2." },
        .{ .from = "switch_mlp.gate_proj", .to = "switch_mlp.w1" },
        .{ .from = "switch_mlp.up_proj", .to = "switch_mlp.w3" },
        .{ .from = "switch_mlp.down_proj", .to = "switch_mlp.w2" },
        .{ .from = "ffn.gate.e_score_correction_bias", .to = "ffn.gate.bias" },
        .{ .from = "attn_hc.fn", .to = "hc_attn_fn" },
        .{ .from = "attn_hc.base", .to = "hc_attn_base" },
        .{ .from = "attn_hc.scale", .to = "hc_attn_scale" },
        .{ .from = "ffn_hc.fn", .to = "hc_ffn_fn" },
        .{ .from = "ffn_hc.base", .to = "hc_ffn_base" },
        .{ .from = "ffn_hc.scale", .to = "hc_ffn_scale" },
    };

    for (replacements) |r| {
        if (std.mem.indexOf(u8, result, r.from)) |idx| {
            const new_len = result.len - r.from.len + r.to.len;
            const new_result = try allocator.alloc(u8, new_len);
            @memcpy(new_result[0..idx], result[0..idx]);
            @memcpy(new_result[idx .. idx + r.to.len], r.to);
            @memcpy(new_result[idx + r.to.len ..], result[idx + r.from.len ..]);
            allocator.free(result);
            return new_result;
        }
    }

    return result;
}

/// Error set for loading.
pub const LoadError = error{
    MissingIndexJson,
    MissingShardFile,
    MissingWeight,
    InvalidConfig,
    UnsupportedArchitecture,
};

/// Smelt Mode configuration for partial expert loading.
/// When enabled, only a subset of experts are loaded into memory.
/// The router is biased to avoid selecting unloaded experts.
pub const SmeltConfig = struct {
    pub const Strategy = enum { uniform, first };
    pub const LoadMode = enum { preload, stream };

    /// Enable Smelt mode
    enabled: bool = false,
    /// Fraction of experts to load per layer (0.0-1.0).
    /// e.g., 0.5 means load 50% of routed experts.
    load_fraction: f32 = 1.0,
    /// Strategy for selecting which experts to load.
    /// "uniform": evenly spaced across expert indices.
    /// "first": load the first N experts.
    strategy: Strategy = .uniform,
    /// Load mode: preload (load subset at init) or stream (load on-demand from disk).
    /// In stream mode, smelt_mask is NOT used - router can select any expert.
    load_mode: LoadMode = .preload,

    /// Build a per-expert residency mask for the given number of experts.
    /// Caller owns the returned slice and must free with allocator.
    pub fn buildMask(self: SmeltConfig, allocator: std.mem.Allocator, n_experts: usize) ![]bool {
        var mask = try allocator.alloc(bool, n_experts);
        @memset(mask, false);
        if (!self.enabled or self.load_fraction >= 1.0) {
            @memset(mask, true);
            return mask;
        }
        const n_load = @max(1, @as(usize, @intFromFloat(@round(self.load_fraction * @as(f32, @floatFromInt(n_experts))))));
        switch (self.strategy) {
            .first => {
                for (0..n_load) |i| mask[i] = true;
            },
            .uniform => {
                const step = @as(f32, @floatFromInt(n_experts - 1)) / @as(f32, @floatFromInt(n_load - 1));
                for (0..n_load) |i| {
                    const idx = @min(n_experts - 1, @as(usize, @intFromFloat(@round(step * @as(f32, @floatFromInt(i))))));
                    mask[idx] = true;
                }
            },
        }
        return mask;
    }
};

/// Parse DeepSeek-V4 config.json into DSV4Config.
pub fn parseDSV4Config(allocator: std.mem.Allocator, json_text: []const u8) !DSV4Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    const root = parsed.value;
    const obj = root.object;

    // Architecture check
    const archs = obj.get("architectures") orelse return LoadError.InvalidConfig;
    const arch = archs.array.items[0].string;
    if (!std.mem.startsWith(u8, arch, "DeepseekV4")) {
        return LoadError.UnsupportedArchitecture;
    }

    // Helper to extract integer
    const getInt = struct {
        fn call(o: std.json.ObjectMap, key: []const u8) ?i64 {
            const val = o.get(key) orelse return null;
            switch (val) {
                .integer => |n| return n,
                .float => |f| return @intFromFloat(f),
                else => return null,
            }
        }
    }.call;

    // Helper to extract float
    const getFloat = struct {
        fn call(o: std.json.ObjectMap, key: []const u8) ?f64 {
            const val = o.get(key) orelse return null;
            switch (val) {
                .float => |f| return f,
                .integer => |n| return @floatFromInt(n),
                else => return null,
            }
        }
    }.call;

    // Parse compress_ratios array
    var compress_ratios = std.ArrayList(usize).empty;
    errdefer compress_ratios.deinit(allocator);
    if (obj.get("compress_ratios")) |ratios_val| {
        for (ratios_val.array.items) |item| {
            const ratio: usize = switch (item) {
                .integer => |n| @intCast(n),
                .float => |f| @intFromFloat(f),
                else => 0,
            };
            try compress_ratios.append(allocator, ratio);
        }
    }

    // Auto-fill compress_ratios if missing or empty, matching mlx-lm default:
    // [0, 128, 4, 128, 4, ...] (first and last layer = 0, alternating 128/4 in between)
    const n_layers: usize = @intCast(getInt(obj, "num_hidden_layers") orelse 43);
    if (compress_ratios.items.len == 0) {
        try compress_ratios.append(allocator, 0);
        const middle_layers = if (n_layers >= 2) n_layers - 2 else 0;
        var i: usize = 0;
        while (i < middle_layers) : (i += 1) {
            // i=0 -> 128, i=1 -> 4, i=2 -> 128, ...
            const ratio: usize = if (i % 2 == 0) 128 else 4;
            try compress_ratios.append(allocator, ratio);
        }
        if (n_layers >= 2) {
            try compress_ratios.append(allocator, 0);
        }
    }

    // Validate: truncate/pad to exactly num_hidden_layers
    if (compress_ratios.items.len > n_layers) {
        compress_ratios.shrinkRetainingCapacity(n_layers);
    }
    while (compress_ratios.items.len < n_layers) {
        try compress_ratios.append(allocator, 0);
    }

    // Parse rope_scaling
    var rope_scaling: ?DSV4Config.YarnRoPEConfig = null;
    if (obj.get("rope_scaling")) |rs| {
        const rs_obj = rs.object;
        rope_scaling = DSV4Config.YarnRoPEConfig{
            .factor = @floatCast(getFloat(rs_obj, "factor") orelse 16.0),
            .original_max_position_embeddings = @intCast(getInt(rs_obj, "original_max_position_embeddings") orelse 65536),
            .beta_fast = @floatCast(getFloat(rs_obj, "beta_fast") orelse 32.0),
            .beta_slow = @floatCast(getFloat(rs_obj, "beta_slow") orelse 1.0),
        };
    }

    // Parse scoring_func
    const scoring_func: DSV4Config.ScoringFunc = blk: {
        if (obj.get("scoring_func")) |sf| {
            const sf_str = sf.string;
            if (std.mem.eql(u8, sf_str, "softmax")) break :blk .softmax;
            if (std.mem.eql(u8, sf_str, "sigmoid")) break :blk .sigmoid;
            if (std.mem.eql(u8, sf_str, "sqrtsoftplus")) break :blk .sqrtsoftplus;
        }
        break :blk .sqrtsoftplus;
    };

    return DSV4Config{
        .vocab_size = @intCast(getInt(obj, "vocab_size") orelse 129280),
        .hidden_size = @intCast(getInt(obj, "hidden_size") orelse 4096),
        .num_hidden_layers = @intCast(getInt(obj, "num_hidden_layers") orelse 43),
        .num_attention_heads = @intCast(getInt(obj, "num_attention_heads") orelse 64),
        .head_dim = @intCast(getInt(obj, "head_dim") orelse 512),
        .num_key_value_heads = @intCast(getInt(obj, "num_key_value_heads") orelse 1),
        .q_lora_rank = @intCast(getInt(obj, "q_lora_rank") orelse 1024),
        .o_lora_rank = @intCast(getInt(obj, "o_lora_rank") orelse 1024),
        .o_groups = @intCast(getInt(obj, "o_groups") orelse 8),
        .qk_rope_head_dim = @intCast(getInt(obj, "qk_rope_head_dim") orelse 64),
        .max_position_embeddings = @intCast(getInt(obj, "max_position_embeddings") orelse 1048576),
        .n_routed_experts = @intCast(getInt(obj, "n_routed_experts") orelse 256),
        .n_shared_experts = @intCast(getInt(obj, "n_shared_experts") orelse 1),
        .num_experts_per_tok = @intCast(getInt(obj, "num_experts_per_tok") orelse 6),
        .moe_intermediate_size = @intCast(getInt(obj, "moe_intermediate_size") orelse 2048),
        .routed_scaling_factor = @floatCast(getFloat(obj, "routed_scaling_factor") orelse 1.5),
        .scoring_func = scoring_func,
        .norm_topk_prob = blk: {
            if (obj.get("norm_topk_prob")) |v| {
                switch (v) {
                    .bool => |b| break :blk b,
                    else => break :blk true,
                }
            }
            break :blk true;
        },
        .sliding_window = @intCast(getInt(obj, "sliding_window") orelse 128),
        .num_hash_layers = @intCast(getInt(obj, "num_hash_layers") orelse 3),
        .index_n_heads = @intCast(getInt(obj, "index_n_heads") orelse 64),
        .index_head_dim = @intCast(getInt(obj, "index_head_dim") orelse 128),
        .index_topk = @intCast(getInt(obj, "index_topk") orelse 512),
        .hc_mult = @intCast(getInt(obj, "hc_mult") orelse 4),
        .hc_sinkhorn_iters = @intCast(getInt(obj, "hc_sinkhorn_iters") orelse 20),
        .hc_eps = @floatCast(getFloat(obj, "hc_eps") orelse 1e-6),
        .rms_norm_eps = @floatCast(getFloat(obj, "rms_norm_eps") orelse 1e-6),
        .rope_theta = @floatCast(getFloat(obj, "rope_theta") orelse 10000.0),
        .compress_rope_theta = @floatCast(getFloat(obj, "compress_rope_theta") orelse 160000.0),
        .rope_scaling = rope_scaling,
        .compress_ratios = try compress_ratios.toOwnedSlice(allocator),
        .swiglu_limit = @floatCast(getFloat(obj, "swiglu_limit") orelse 10.0),
        // Parse global quantization defaults from "quantization" or "quantization_config"
        .quantize_default_bits = blk: {
            const qc = obj.get("quantization") orelse obj.get("quantization_config") orelse break :blk 0;
            if (qc != .object) break :blk 0;
            break :blk @intCast(getInt(qc.object, "bits") orelse 0);
        },
        .quantize_default_group_size = blk: {
            const qc = obj.get("quantization") orelse obj.get("quantization_config") orelse break :blk 64;
            if (qc != .object) break :blk 64;
            break :blk @intCast(getInt(qc.object, "group_size") orelse 64);
        },
        .quantize_default_mode = blk: {
            const qc = obj.get("quantization") orelse obj.get("quantization_config") orelse break :blk "affine";
            if (qc != .object) break :blk "affine";
            if (qc.object.get("mode")) |m| {
                if (m == .string) break :blk m.string;
            }
            break :blk "affine";
        },
    };
}

/// Load config from a directory path.
pub fn loadDSV4Config(allocator: std.mem.Allocator, io_ctx: std.Io, dir_path: []const u8) !DSV4Config {
    const config_path = try std.fs.path.join(allocator, &.{ dir_path, "config.json" });
    defer allocator.free(config_path);

    const content = try std.Io.Dir.cwd().readFileAlloc(io_ctx, config_path, allocator, .limited(1024 * 1024));
    defer allocator.free(content);

    return parseDSV4Config(allocator, content);
}

/// Load strategy for weight loading.
pub const LoadStrategy = enum { eager, selective };

/// Load weights using selective strategy — returns a lazy provider that loads tensors on-demand.
/// Only reads safetensors headers (~few KB per shard). Actual tensor data is loaded
/// from disk only when buildDSV4Model accesses each weight.
pub fn loadWeightsSelectiveLazy(
    allocator: std.mem.Allocator,
    dir_path: []const u8,
) !struct { index: *safetensors_reader.TensorIndex, provider: *safetensors_reader.LazyWeightProvider } {
    const index = try allocator.create(safetensors_reader.TensorIndex);
    index.* = try safetensors_reader.loadOrBuildIndex(allocator, dir_path);
    std.log.info("Indexed {d} tensors across shards", .{index.entries.count()});

    const provider = try allocator.create(safetensors_reader.LazyWeightProvider);
    provider.* = safetensors_reader.LazyWeightProvider.init(allocator, index, &mapV4WeightName);
    try provider.buildReverseMap();
    std.log.info("Built reverse map with {d} entries", .{provider.reverse_map.count()});

    return .{ .index = index, .provider = provider };
}

/// Load weights using selective strategy — loads all needed tensors into HashMap.
/// For machines with enough memory to hold all weights (~37GB for V4 Flash 4-bit).
pub fn loadWeightsSelective(
    allocator: std.mem.Allocator,
    dir_path: []const u8,
    smelt: SmeltConfig,
) !std.StringHashMap(Array) {
    const now = struct {
        fn ns() i128 {
            var ts: std.c.timespec = std.mem.zeroes(std.c.timespec);
            _ = std.c.clock_gettime(@enumFromInt(6), &ts);
            return @as(i128, ts.sec) * 1_000_000_000 + @as(i128, ts.nsec);
        }
    }.ns;
    const t0_total = now();

    // Build tensor index from shard headers (only reads first few KB per file)
    const t0_index = now();
    var index = try safetensors_reader.loadOrBuildIndex(allocator, dir_path);
    const t1_index = now();
    defer index.deinit();

    std.log.info("Indexed {d} tensors across shards", .{index.entries.count()});

    // Open all shard file descriptors once (avoids 2000+ open/close cycles)
    const t0_fd = now();
    var fd_pool = safetensors_reader.FdPool.init(allocator);
    defer fd_pool.deinit();
    try fd_pool.openAll(&index);
    const t1_fd = now();

    // mmap all shards for zero-syscall reads
    const t0_mmap = now();
    var mmap_pool = safetensors_reader.MmapPool.init(allocator);
    defer mmap_pool.deinit();
    mmap_pool.mmapAll(&index) catch |err| {
        std.log.warn("Failed to mmap shards: {}, falling back to pread", .{err});
    };
    const t1_mmap = now();

    // Build smelt mask
    var smelt_mask: ?[]bool = null;
    defer if (smelt_mask) |m| allocator.free(m);
    const skip_all_experts = smelt.enabled and smelt.load_mode == .stream;
    if (smelt.enabled and smelt.load_fraction < 1.0 and !skip_all_experts) {
        smelt_mask = try smelt.buildMask(allocator, 256);
    }

    // Count how many tensors to load (for progress reporting)
    var to_load: usize = 0;
    var to_skip: usize = 0;
    {
        var count_it = index.entries.iterator();
        while (count_it.next()) |entry| {
            const hf_name = entry.key_ptr.*;
            if (skip_all_experts and isExpertWeight(hf_name)) {
                to_skip += 1;
                continue;
            }
            if (std.mem.startsWith(u8, hf_name, "__metadata__") or std.mem.startsWith(u8, hf_name, "mtp.")) {
                to_skip += 1;
                continue;
            }
            to_load += 1;
        }
    }
    std.log.info("Loading {d} backbone weights ({d} expert weights skipped)", .{ to_load, to_skip });

    // Load each tensor using pre-opened file descriptors
    const t0_load = now();
    var weights = std.StringHashMap(Array).init(allocator);
    errdefer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    const posix_c = @cImport(@cInclude("unistd.h"));

    var loaded_count: usize = 0;
    var skipped_count: usize = 0;

    // Collect filtered entries into per-shard lists for sequential I/O
    var shard_groups = std.StringHashMap(std.ArrayList(EntryWithName)).init(allocator);
    defer {
        var it = shard_groups.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit(allocator);
        }
        shard_groups.deinit();
    }
    {
        var collect_it = index.entries.iterator();
        while (collect_it.next()) |entry| {
            const hf_name = entry.key_ptr.*;

            // Stream mode: skip ALL expert weights (loaded on-demand via ExpertStreamProvider)
            if (skip_all_experts and isExpertWeight(hf_name)) continue;

            // Skip expert weights for unloaded experts
            if (smelt_mask != null and isExpertWeight(hf_name)) {
                const eid = parseExpertIndexFromHF(hf_name);
                if (eid) |e| {
                    if (e < smelt_mask.?.len and !smelt_mask.?[e]) continue;
                }
            }

            // Skip __metadata__ and mtp.* weights
            if (std.mem.startsWith(u8, hf_name, "__metadata__") or std.mem.startsWith(u8, hf_name, "mtp.")) continue;

            const info = entry.value_ptr.*;
            const shard_name = info.shard_path;
            const gop = try shard_groups.getOrPut(shard_name);
            if (!gop.found_existing) {
                gop.value_ptr.* = std.ArrayList(EntryWithName).empty;
            }
            try gop.value_ptr.*.append(allocator, .{ .name = hf_name, .info = info });
        }
    }

    // Sort each shard's entries by data_offset_start for sequential I/O
    {
        var sort_it = shard_groups.iterator();
        while (sort_it.next()) |entry| {
            const items = entry.value_ptr.*.items;
            if (items.len > 1) {
                std.sort.insertion(EntryWithName, items, {}, struct {
                    fn lessThan(_: void, a: EntryWithName, b: EntryWithName) bool {
                        return a.info.data_offset_start < b.info.data_offset_start;
                    }
                }.lessThan);
            }
        }
    }

    // Load tensors in sequential order per shard
    {
        var shard_it = shard_groups.iterator();
        while (shard_it.next()) |shard_entry| {
            const entries_list = shard_entry.value_ptr.*;
            for (entries_list.items) |item| {
                const hf_name = item.name;
                const info = item.info;

                const data_len: usize = @intCast(info.data_offset_end - info.data_offset_start);

                // Map dtype and build shape first (needed for both paths)
                const mlx_dtype = safetensors_reader.dtypeFromString(info.dtype_str) orelse {
                    skipped_count += 1;
                    continue;
                };
                var shape_i32 = allocator.alloc(i32, info.shape.len) catch {
                    skipped_count += 1;
                    continue;
                };
                defer allocator.free(shape_i32);
                for (info.shape, 0..) |s, si| {
                    shape_i32[si] = @intCast(s);
                }

                // Try mmap path first (zero-copy pointer into mapped region)
                const tensor = if (mmap_pool.getSlice(info.shard_path, info.data_offset_start, data_len)) |slice| blk: {
                    const arr = c.c.mlx_array_new_data(slice.ptr, shape_i32.ptr, @intCast(shape_i32.len), mlx_dtype);
                    break :blk Array.fromHandle(arr);
                } else |_| blk: {
                    // Fallback to pread via FdPool
                    const fd = fd_pool.getFd(info.shard_path) catch {
                        skipped_count += 1;
                        continue;
                    };
                    const buf = allocator.alloc(u8, data_len) catch {
                        skipped_count += 1;
                        continue;
                    };
                    defer allocator.free(buf);
                    const read_len = posix_c.pread(fd, buf.ptr, data_len, @intCast(info.data_offset_start));
                    if (read_len < @as(isize, @intCast(data_len))) {
                        skipped_count += 1;
                        continue;
                    }
                    const arr = c.c.mlx_array_new_data(buf.ptr, shape_i32.ptr, @intCast(shape_i32.len), mlx_dtype);
                    break :blk Array.fromHandle(arr);
                };

                // Apply name mapping
                const mapped = try mapV4WeightName(allocator, hf_name);
                const key = mapped orelse try allocator.dupe(u8, hf_name);
                try weights.put(key, tensor);

                loaded_count += 1;
                if (loaded_count % 200 == 0) {
                    std.log.info("Loaded {d}/{d} tensors...", .{ loaded_count, to_load });
                }
            }
        }
    }

    const t1_load = now();
    const t1_total = now();
    std.log.info("TTFT breakdown: index={d}ms fd={d}ms mmap={d}ms load={d}ms total={d}ms", .{
        @divTrunc(t1_index - t0_index, 1_000_000),
        @divTrunc(t1_fd - t0_fd, 1_000_000),
        @divTrunc(t1_mmap - t0_mmap, 1_000_000),
        @divTrunc(t1_load - t0_load, 1_000_000),
        @divTrunc(t1_total - t0_total, 1_000_000),
    });

    std.log.info("Loaded {d} weights selectively", .{weights.count()});
    return weights;
}

/// Load weights from a model directory.
/// Automatically detects sharded vs single-file models.
/// `smelt` controls partial expert loading to reduce memory usage.
pub fn loadWeightsFromDirectory(
    allocator: std.mem.Allocator,
    io_ctx: std.Io,
    dir_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    smelt: SmeltConfig,
) !std.StringHashMap(Array) {
    // Try sharded loading first
    const index_path = try std.fs.path.join(allocator, &.{ dir_path, "model.safetensors.index.json" });
    defer allocator.free(index_path);

    const has_index = blk: {
        const dir = std.Io.Dir.cwd();
        const file = dir.openFile(io_ctx, index_path, .{}) catch break :blk false;
        file.close(io_ctx);
        break :blk true;
    };

    if (has_index) {
        var weights = try loadShardedWeights(allocator, io_ctx, dir_path, index_path, ctx, stream, smelt);
        try splitFusedExperts(allocator, &weights, ctx, smelt);
        std.log.info("Weights loaded: {d} entries (lazy, pre-buildDSV4Model)", .{weights.count()});
        return weights;
    }

    // Fall back to single file
    const model_path = try std.fs.path.join(allocator, &.{ dir_path, "model.safetensors" });
    defer allocator.free(model_path);

    var st = try io.loadSafetensors(allocator, model_path);
    defer {
        var w_it = st.weights.iterator();
        while (w_it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        st.weights.deinit();
        var m_it = st.metadata.iterator();
        while (m_it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        st.metadata.deinit();
    }

    // Transfer ownership of weights into a new HashMap
    var weights = std.StringHashMap(Array).init(allocator);
    errdefer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    var it = st.weights.iterator();
    while (it.next()) |entry| {
        // Apply HF → internal name mapping
        const mapped = try mapV4WeightName(allocator, entry.key_ptr.*);
        const key = mapped orelse try allocator.dupe(u8, entry.key_ptr.*);
        const weight = entry.value_ptr.*;
        // Keep weights in original dtype — no bfloat16→float32 conversion.
        try weights.put(key, weight);
    }

    try splitFusedExperts(allocator, &weights, ctx, smelt);
    return weights;
}

/// Parse expert index from a weight key such as "layers.0.ffn.experts.5.w1.weight".
/// Returns null if the key does not contain an expert index.
fn parseExpertIndex(key: []const u8) ?usize {
    const prefix = "ffn.experts.";
    const idx = std.mem.indexOf(u8, key, prefix) orelse return null;
    const start = idx + prefix.len;
    var end = start;
    while (end < key.len and std.ascii.isDigit(key[end])) end += 1;
    if (end == start) return null;
    return std.fmt.parseInt(usize, key[start..end], 10) catch null;
}

/// Convert scale array to float32. If scale is uint8, interpret as FP8 E8M0
/// exponent: result = exp((scale - 127) * ln(2)).
fn scaleToFloat(ctx: EagerContext, scale: Array) !Array {
    if (scale.dtype() == .uint8) {
        const scale_f32 = try ops.astype(ctx, scale, .float32);
        defer scale_f32.deinit();
        const offset = try ops.scalarF32(ctx, 127.0);
        defer offset.deinit();
        const shifted = try ops.subtract(ctx, scale_f32, offset);
        defer shifted.deinit();
        const ln2 = try ops.scalarF32(ctx, @log(2.0));
        defer ln2.deinit();
        const exponent = try ops.multiply(ctx, shifted, ln2);
        defer exponent.deinit();
        return ops.exp(ctx, exponent);
    }
    return ops.astype(ctx, scale, .float32);
}

/// Dequantize FP4-packed expert weights using lookup table (matching mlx-lm sanitize).
/// Each uint8 byte packs two FP4 values: low nibble and high nibble.
/// Table: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
fn dequantFp4(ctx: EagerContext, weight: Array, scale: Array, block_size: i32) !Array {
    const table_data = [16]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 };
    const table = try Array.fromData(ctx.allocator, f32, &table_data, &[_]i32{16});
    defer table.deinit();

    // Cast to uint8 for bitwise ops
    const packed_bytes = try ops.astype(ctx, weight, .uint8);
    defer packed_bytes.deinit();

    // Low nibble: packed & 0x0F
    const mask_0f = try Array.fromData(ctx.allocator, u8, &[_]u8{0x0F}, &[_]i32{1});
    defer mask_0f.deinit();
    const low = try cmp_mod.bitwiseAnd(ctx, packed_bytes, mask_0f);
    defer low.deinit();

    // High nibble: (packed >> 4) & 0x0F — use integer division by 16 as right shift
    const sixteen = try Array.fromData(ctx.allocator, u8, &[_]u8{16}, &[_]i32{1});
    defer sixteen.deinit();
    const shifted = try ops.divide(ctx, packed_bytes, sixteen);
    defer shifted.deinit();
    const high = try cmp_mod.bitwiseAnd(ctx, shifted, mask_0f);
    defer high.deinit();

    // Lookup: table[low], table[high]
    const low_i32 = try ops.astype(ctx, low, .int32);
    defer low_i32.deinit();
    const high_i32 = try ops.astype(ctx, high, .int32);
    defer high_i32.deinit();
    const val_low = try shape_mod.take(ctx, table, low_i32);
    defer val_low.deinit();
    const val_high = try shape_mod.take(ctx, table, high_i32);
    defer val_high.deinit();

    // Stack [low, high] along last axis and reshape to [rows, cols*2]
    // val_low, val_high: [rows, cols] → expand to [rows, cols, 1] each, concat on axis 2
    const low_exp = try ops.expandDims(ctx, val_low, -1);
    defer low_exp.deinit();
    const high_exp = try ops.expandDims(ctx, val_high, -1);
    defer high_exp.deinit();
    const stacked = try shape_mod.concatenateAxis(ctx, &[_]Array{ low_exp, high_exp }, -1);
    defer stacked.deinit();

    // Reshape to [rows, cols*2]
    const w_shape = weight.shape();
    const rows = w_shape[0];
    const cols = w_shape[1];
    const unpacked = try ops.reshape(ctx, stacked, &[_]i32{ rows, cols * 2 });
    defer unpacked.deinit();

    // Scale: repeat scale_to_float(scale) by block_size along last axis
    const scale_f32 = try scaleToFloat(ctx, scale);
    defer scale_f32.deinit();
    const scale_rep = try shape_mod.repeatAxis(ctx, scale_f32, block_size, -1);
    defer scale_rep.deinit();

    // Result: (unpacked * scale).astype(bfloat16)
    const scaled = try ops.multiply(ctx, unpacked, scale_rep);
    defer scaled.deinit();
    return ops.astype(ctx, scaled, .bfloat16);
}

/// Dequantize FP8-packed weights with block-wise scaling (matching mlx-lm sanitize).
/// weight is uint8 FP8 E4M3, scale has shape [m/block_size, n/block_size].
fn dequantFp8(ctx: EagerContext, weight: Array, scale: Array, block_size: i32) !Array {
    // Convert FP8 to bfloat16
    const w_bf16 = try quantize_mod.fromFp8(ctx, weight, .bfloat16);
    defer w_bf16.deinit();

    const scale_f32 = try scaleToFloat(ctx, scale);
    defer scale_f32.deinit();

    const w_shape = weight.shape();
    const m: i32 = w_shape[0];
    const n: i32 = w_shape[1];
    const bs = block_size;

    // Compute padding needed for block alignment
    const pad_m = @mod(-m, bs);
    const pad_n = @mod(-n, bs);

    // Pad weight if needed
    var w_padded: Array = undefined;
    var needs_pad_free = false;
    if (pad_m > 0 or pad_n > 0) {
        // Create zero-padded array by concatenating zeros
        const pm: i32 = m + pad_m;
        const pn: i32 = n + pad_n;
        var padded_raw = c.c.mlx_array_new();
        try c.check(c.c.mlx_zeros(&padded_raw, (&[_]i32{ pm, pn }).ptr, 2, c.c.MLX_BFLOAT16, ctx.stream.inner));
        const padded = Array.fromHandle(padded_raw);
        defer padded.deinit();
        // Copy original into top-left corner via slice_update
        w_padded = try shape_mod.sliceUpdate(ctx, padded, w_bf16, &[_]i32{ 0, 0 }, &[_]i32{ m, n }, &[_]i32{});
        needs_pad_free = true;
    } else {
        w_padded = w_bf16;
    }
    defer if (needs_pad_free) w_padded.deinit();

    const pm: i32 = m + pad_m;
    const pn: i32 = n + pad_n;

    // Reshape to [m/bs, bs, n/bs, bs]
    const reshaped = try ops.reshape(ctx, w_padded, &[_]i32{ @divExact(pm, bs), bs, @divExact(pn, bs), bs });
    defer reshaped.deinit();

    // Multiply by scale[:, None, :, None]
    const scale_exp = try ops.expandDims(ctx, scale_f32, 1);
    defer scale_exp.deinit();
    const scale_exp2 = try ops.expandDims(ctx, scale_exp, 3);
    defer scale_exp2.deinit();

    // Cast reshaped to float32 for multiplication
    const reshaped_f32 = try ops.astype(ctx, reshaped, .float32);
    defer reshaped_f32.deinit();
    const scaled = try ops.multiply(ctx, reshaped_f32, scale_exp2);
    defer scaled.deinit();

    // Reshape back to [pm, pn]
    const flat = try ops.reshape(ctx, scaled, &[_]i32{ pm, pn });
    defer flat.deinit();

    // Truncate padding: [:m, :n]
    if (pad_m > 0 or pad_n > 0) {
        const truncated = try ops.slice(ctx, flat, &[_]i32{ 0, 0 }, &[_]i32{ m, n }, &[_]i32{});
        defer truncated.deinit();
        return ops.astype(ctx, truncated, .bfloat16);
    }
    return ops.astype(ctx, flat, .bfloat16);
}

/// Detect if a weight key + data matches the FP4 expert pattern.
/// Returns true if: key contains ".ffn.experts." (not shared), dtype is int8/uint8,
/// and scale.shape[-1] * 16 == weight.shape[-1] (FP4 packing ratio).
fn isFp4ExpertWeight(key: []const u8, weight: Array, scale: Array) bool {
    if (std.mem.indexOf(u8, key, ".ffn.experts.") == null) return false;
    if (std.mem.indexOf(u8, key, ".shared_experts.") != null) return false;
    const dt = weight.dtype();
    if (dt != .int8 and dt != .uint8) return false;
    const w_shape = weight.shape();
    const s_shape = scale.shape();
    if (w_shape.len < 2 or s_shape.len < 1) return false;
    const w_cols = w_shape[w_shape.len - 1];
    const s_cols = s_shape[s_shape.len - 1];
    return s_cols * 16 == w_cols;
}

/// Remove all weights (including .weight, .scales, .biases) for experts
/// that are not resident according to `mask`.
fn removeUnloadedExperts(allocator: std.mem.Allocator, weights: *std.StringHashMap(Array), mask: []const bool) !void {
    var keys_to_remove = std.ArrayList([]u8).empty;
    defer {
        for (keys_to_remove.items) |k| allocator.free(k);
        keys_to_remove.deinit(allocator);
    }

    var iter = weights.iterator();
    while (iter.next()) |entry| {
        if (parseExpertIndex(entry.key_ptr.*)) |eid| {
            if (eid < mask.len and !mask[eid]) {
                try keys_to_remove.append(allocator, try allocator.dupe(u8, entry.key_ptr.*));
            }
        }
    }

    for (keys_to_remove.items) |key| {
        if (weights.fetchRemove(key)) |kv| {
            allocator.free(kv.key);
            kv.value.deinit();
        }
        allocator.free(key);
    }
}

/// Split fused switch_mlp expert weights [n_experts, out, in] into individual
/// expert weights [out, in] as layers.N.ffn.experts.E.w{1,2,3}.weight.
/// Also handles quantized .scales suffixes.
/// Respects `smelt` to skip splitting / dequantizing unloaded experts.
fn splitFusedExperts(allocator: std.mem.Allocator, weights: *std.StringHashMap(Array), ctx: EagerContext, smelt: SmeltConfig) !void {
    _ = ctx; // No longer used — dequantization moved to buildDSV4Model
    // --- Pre-compute Smelt mask if needed ---
    // Infer total number of experts from fused tensors or existing split keys.
    var n_experts: usize = 0;
    var iter = weights.iterator();
    while (iter.next()) |entry| {
        if (std.mem.indexOf(u8, entry.key_ptr.*, "ffn.switch_mlp.") != null) {
            const arr = entry.value_ptr.*;
            const shape = arr.shape();
            if (shape.len == 3 and @as(usize, @intCast(shape[0])) > n_experts) {
                n_experts = @intCast(shape[0]);
            }
        } else if (parseExpertIndex(entry.key_ptr.*)) |eid| {
            if (eid + 1 > n_experts) n_experts = eid + 1;
        }
    }

    var smelt_mask: ?[]bool = null;
    defer if (smelt_mask) |m| allocator.free(m);
    if (smelt.enabled and smelt.load_fraction < 1.0 and n_experts > 0) {
        smelt_mask = try smelt.buildMask(allocator, n_experts);
    }

    // === Step 1: Keep fused expert tensors as-is ===
    // Previously this step split fused switch_mlp weights into individual experts.
    // Now we keep them fused for gather_mm dispatch. The layer construction loop
    // in loadDSV4Model handles both fused and individual expert formats.
    // We only need to handle the dequantization step below.

    // === Step 2: Skip dequantization ===
    // All quantized weights stay packed. Dequantization happens lazily:
    // - Attention weights: quantizedMatmul in forward pass
    // - Expert weights: gatherQmm in forward pass
    // - Embedding/lm_head: dequantized in buildDSV4Model (needed for mlx_take)
    // - wo_a: dequantized in buildDSV4Model (needed for reshape)
    // - Shared expert: dequantized lazily in buildDSV4Model

    // === Step 3: Remove any remaining weights for unloaded experts ===
    // This handles both (a) already-split weights that were not fused, and
    // (b) any leftover metadata keys for skipped experts.
    if (smelt_mask) |mask| {
        try removeUnloadedExperts(allocator, weights, mask);
    }
}

/// Load sharded safetensors using index.json.
fn loadShardedWeights(
    allocator: std.mem.Allocator,
    io_ctx: std.Io,
    dir_path: []const u8,
    index_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    smelt: SmeltConfig,
) !std.StringHashMap(Array) {
    _ = stream;
    _ = ctx;
    // Read index.json
    const index_content = try std.Io.Dir.cwd().readFileAlloc(io_ctx, index_path, allocator, .limited(50 * 1024 * 1024));
    defer allocator.free(index_content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, index_content, .{});
    defer parsed.deinit();

    const weight_map = parsed.value.object.get("weight_map") orelse return LoadError.MissingIndexJson;
    const wm_obj = weight_map.object;

    // Build smelt mask to determine which expert weights to skip
    var smelt_mask: ?[]bool = null;
    defer if (smelt_mask) |m| allocator.free(m);
    // In stream mode, skip ALL expert weights regardless of load_fraction
    // (they'll be loaded on-demand via TensorIndex + ExpertStreamProvider).
    // In preload mode, only skip unselected experts when load_fraction < 1.0.
    const skip_all_experts = smelt.enabled and smelt.load_mode == .stream;
    if (smelt.enabled and smelt.load_fraction < 1.0 and !skip_all_experts) {
        smelt_mask = try smelt.buildMask(allocator, 256); // V4 has 256 experts
    }

    // Determine which shards to load: skip shards that ONLY contain unneeded expert weights
    var shard_set = std.StringHashMap(void).init(allocator);
    defer shard_set.deinit();

    var wm_it = wm_obj.iterator();
    while (wm_it.next()) |entry| {
        const weight_name = entry.key_ptr.*;
        const shard_file = entry.value_ptr.*.string;

        // Check if this weight should be skipped (unloaded expert or stream mode)
        if (skip_all_experts and isExpertWeight(weight_name)) {
            // Stream mode: skip ALL expert weights (loaded on-demand from SSD)
            continue;
        }
        if (smelt_mask != null and isExpertWeight(weight_name)) {
            const eid = parseExpertIndexFromHF(weight_name);
            if (eid) |e| {
                if (e < smelt_mask.?.len and !smelt_mask.?[e]) continue; // Skip this weight
            }
        }

        // This shard contains a needed weight — mark it for loading
        if (!shard_set.contains(shard_file)) {
            const key = try allocator.dupe(u8, shard_file);
            try shard_set.put(key, {});
        }
    }

    std.log.info("Loading {d} of {d} shards (smelt filtering)", .{ shard_set.count(), wm_obj.count() });

    // Result weights map
    var weights = std.StringHashMap(Array).init(allocator);
    errdefer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    // Load each shard, extract needed weights, free shard immediately
    var shard_it = shard_set.keyIterator();
    while (shard_it.next()) |shard_file_ptr| {
        const shard_file = shard_file_ptr.*;
        const shard_path = try std.fs.path.join(allocator, &.{ dir_path, shard_file });
        defer allocator.free(shard_path);

        std.log.info("Loading shard: {s}", .{shard_file});

        var st = try io.loadSafetensors(allocator, shard_path);
        // Free metadata immediately — we don't need it
        {
            var m_it = st.metadata.iterator();
            while (m_it.next()) |entry| {
                allocator.free(entry.key_ptr.*);
                allocator.free(entry.value_ptr.*);
            }
            st.metadata.deinit();
        }

        // Extract needed weights, free everything else
        var w_it = st.weights.iterator();
        while (w_it.next()) |entry| {
            const hf_name = entry.key_ptr.*;
            var keep = true;

            // Skip expert weights in smelt mode (will be streamed from SSD on demand)
            if (smelt.enabled and isExpertWeight(hf_name)) {
                if (skip_all_experts) {
                    // Stream mode: skip ALL expert weights
                    keep = false;
                } else {
                    const eid = parseExpertIndexFromHF(hf_name);
                    if (eid) |e| {
                        // Individual expert format: skip based on mask
                        if (smelt_mask) |mask| {
                            if (e < mask.len and !mask[e]) {
                                keep = false;
                            }
                        }
                    }
                    // Fused switch_mlp weights: always skip in smelt mode (streamed from SSD)
                    if (eid == null and std.mem.indexOf(u8, hf_name, "switch_mlp") != null) {
                        keep = false;
                    }
                }
            }

            if (keep) {
                const mapped = try mapV4WeightName(allocator, hf_name);
                const key = mapped orelse try allocator.dupe(u8, hf_name);
                try weights.put(key, entry.value_ptr.*);
            } else {
                // Free unneeded array to reclaim memory immediately
                entry.value_ptr.*.deinit();
            }
            // Free the original key string from safetensors
            allocator.free(entry.key_ptr.*);
        }
        st.weights.deinit();
        // Shard data is now freed — only extracted weights remain in `weights` HashMap
        // Clear MLX cache to release any internal buffers from this shard
        _ = c.c.mlx_clear_cache();
    }

    // Free shard_set keys
    var ss_it = shard_set.keyIterator();
    while (ss_it.next()) |k| allocator.free(k.*);

    return weights;
}

/// Build DSV4Model from loaded weights.
/// `smelt` controls partial expert loading (Smelt mode).
pub fn buildDSV4Model(
    allocator: std.mem.Allocator,
    config: *const DSV4Config,
    weights: *std.StringHashMap(Array),
    ctx: EagerContext,
    stream: c.c.mlx_stream,
    smelt: SmeltConfig,
) !DSV4Model {
    _ = stream;
    const num_layers = config.num_hidden_layers;
    const hidden_size = config.hidden_size;
    const vocab_size = config.vocab_size;
    const head_dim = config.head_dim;
    const rope_dim = config.qk_rope_head_dim;
    const n_routed_experts = config.n_routed_experts;
    _ = config.moe_intermediate_size;

    // === Embedding ===
    // Embedding weights must be dequantized because mlx_take (used in Embedding.forward)
    // doesn't support quantized packed arrays.
    var embed_weight = weights.get("embed.weight") orelse {
        std.log.err("Missing weight: embed.weight", .{});
        var dbg_it = weights.iterator();
        while (dbg_it.next()) |entry| {
            if (std.mem.startsWith(u8, entry.key_ptr.*, "embed")) {
                std.log.info("  Available: {s}", .{entry.key_ptr.*});
            }
        }
        return LoadError.MissingWeight;
    };
    // Dequantize if quantized (has .scales)
    const embed_scales = weights.get("embed.weight.scales");
    if (embed_scales != null) {
        const embed_biases = weights.get("embed.weight.biases");
        const null_array: c.c.mlx_array = .{ .ctx = null };
        const biases_inner = if (embed_biases) |b| b.inner else null_array;
        const embed_is_mxfp4 = embed_biases == null;
        const opt_group: c.c.mlx_optional_int = .{ .value = if (embed_is_mxfp4) 32 else config.quantize_default_group_size, .has_value = true };
        const opt_bits: c.c.mlx_optional_int = .{ .value = @as(i32, @intCast(if (config.quantize_default_bits > 0) config.quantize_default_bits else 4)), .has_value = true };
        const no_dtype: c.c.mlx_optional_dtype = .{ .value = c.c.MLX_BFLOAT16, .has_value = true };
        var deq_res = c.c.mlx_array_new();
        try c.check(c.c.mlx_dequantize(&deq_res, embed_weight.inner, embed_scales.?.inner, biases_inner, opt_group, opt_bits, if (embed_is_mxfp4) "mxfp4" else "affine", null_array, no_dtype, ctx.stream.inner));
        embed_weight = Array.fromHandle(deq_res);
    }
    const embed = nn.Embedding{
        .ctx = ctx,
        .num_embeddings = vocab_size,
        .embedding_dim = hidden_size,
        .weight = embed_weight,
    };
    if (weights.fetchRemove("embed.weight")) |kv| allocator.free(kv.key);
    consumeWeightKey(allocator, weights, "embed.weight");

    // === Output norms ===
    const norm_weight = weights.get("norm.weight") orelse return LoadError.MissingWeight;
    var norm = try nn.RMSNorm.init(ctx, hidden_size, config.rms_norm_eps);
    norm.weight.deinit();
    norm.weight = norm_weight;
    if (weights.fetchRemove("norm.weight")) |kv| allocator.free(kv.key);

    // === HyperHead (hc_head) weights ===
    // Keys may be "hc_head.fn" (from HF format) or "hc_head_fn" (legacy internal)
    var hc_head: ?deepseek_v4.HyperHead = null;
    const hc_fn = weights.get("hc_head.fn") orelse weights.get("hc_head_fn");
    const hc_base = weights.get("hc_head.base") orelse weights.get("hc_head_base");
    const hc_scale = weights.get("hc_head.scale") orelse weights.get("hc_head_scale");
    if (hc_fn != null and hc_base != null and hc_scale != null) {
        // Remove all possible key variants
        inline for (.{ "hc_head.fn", "hc_head_fn" }) |k| {
            if (weights.fetchRemove(k)) |kv| allocator.free(kv.key);
        }
        inline for (.{ "hc_head.base", "hc_head_base" }) |k| {
            if (weights.fetchRemove(k)) |kv| allocator.free(kv.key);
        }
        inline for (.{ "hc_head.scale", "hc_head_scale" }) |k| {
            if (weights.fetchRemove(k)) |kv| allocator.free(kv.key);
        }
        hc_head = deepseek_v4.HyperHead{
            .ctx = ctx,
            .fn_weight = hc_fn.?,
            .base = hc_base.?,
            .scale = hc_scale.?,
            .hc_mult = config.hc_mult,
            .norm_eps = config.rms_norm_eps,
        };
    }

    // === LM Head ===
    var lm_head_weight = weights.get("head.weight") orelse weights.get("lm_head.weight") orelse return LoadError.MissingWeight;
    lm_head_weight = try dequantIfNeeded(allocator, weights, "head.weight", lm_head_weight, config, ctx);
    const lm_head = lm_head_weight;
    if (weights.fetchRemove("head.weight")) |kv| allocator.free(kv.key);
    if (weights.fetchRemove("lm_head.weight")) |kv| allocator.free(kv.key);
    consumeWeightKey(allocator, weights, "head.weight");
    consumeWeightKey(allocator, weights, "lm_head.weight");

    // === Layers ===
    const layers = try allocator.alloc(deepseek_v4.DSV4TransformerBlock, num_layers);
    errdefer allocator.free(layers);

    for (0..num_layers) |i| {
        if (i % 10 == 0) std.log.info("Building layer {d}/{d}...", .{ i, num_layers });
        // Periodically clear MLX cache to reduce memory pressure during model construction
        if (i % 5 == 0) _ = c.c.mlx_clear_cache();
        const idx_fmt = try std.fmt.allocPrint(allocator, "layers.{d}.", .{i});
        defer allocator.free(idx_fmt);

        // --- Attention weights (keep quantized — use quantizedMatmul in forward) ---
        const wq_a_name = try std.fmt.allocPrint(allocator, "{s}attn.wq_a.weight", .{idx_fmt});
        defer allocator.free(wq_a_name);
        const wq_a = weights.get(wq_a_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(wq_a_name)) |kv| allocator.free(kv.key);
        const wq_a_s_name = try std.fmt.allocPrint(allocator, "{s}attn.wq_a.scales", .{idx_fmt});
        defer allocator.free(wq_a_s_name);
        const wq_a_scales: ?Array = blk_s: {
            const v = weights.get(wq_a_s_name);
            if (v != null) {
                if (weights.fetchRemove(wq_a_s_name)) |kv2| allocator.free(kv2.key);
            }
            break :blk_s v;
        };
        const wq_a_b_name = try std.fmt.allocPrint(allocator, "{s}attn.wq_a.biases", .{idx_fmt});
        defer allocator.free(wq_a_b_name);
        const wq_a_biases: ?Array = blk_b: {
            const v = weights.get(wq_a_b_name);
            if (v != null) {
                if (weights.fetchRemove(wq_a_b_name)) |kv2| allocator.free(kv2.key);
            }
            break :blk_b v;
        };

        const wq_b_name = try std.fmt.allocPrint(allocator, "{s}attn.wq_b.weight", .{idx_fmt});
        defer allocator.free(wq_b_name);
        const wq_b = weights.get(wq_b_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(wq_b_name)) |kv| allocator.free(kv.key);
        const wq_b_s_name = try std.fmt.allocPrint(allocator, "{s}attn.wq_b.scales", .{idx_fmt});
        defer allocator.free(wq_b_s_name);
        const wq_b_scales: ?Array = blk_s: {
            const v = weights.get(wq_b_s_name);
            if (v != null) {
                if (weights.fetchRemove(wq_b_s_name)) |kv2| allocator.free(kv2.key);
            }
            break :blk_s v;
        };
        const wq_b_b_name = try std.fmt.allocPrint(allocator, "{s}attn.wq_b.biases", .{idx_fmt});
        defer allocator.free(wq_b_b_name);
        const wq_b_biases: ?Array = blk_b: {
            const v = weights.get(wq_b_b_name);
            if (v != null) {
                if (weights.fetchRemove(wq_b_b_name)) |kv2| allocator.free(kv2.key);
            }
            break :blk_b v;
        };

        const wkv_name = try std.fmt.allocPrint(allocator, "{s}attn.wkv.weight", .{idx_fmt});
        defer allocator.free(wkv_name);
        const wkv = weights.get(wkv_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(wkv_name)) |kv| allocator.free(kv.key);
        const wkv_s_name = try std.fmt.allocPrint(allocator, "{s}attn.wkv.scales", .{idx_fmt});
        defer allocator.free(wkv_s_name);
        const wkv_scales: ?Array = blk_s: {
            const v = weights.get(wkv_s_name);
            if (v != null) {
                if (weights.fetchRemove(wkv_s_name)) |kv2| allocator.free(kv2.key);
            }
            break :blk_s v;
        };
        const wkv_b_name = try std.fmt.allocPrint(allocator, "{s}attn.wkv.biases", .{idx_fmt});
        defer allocator.free(wkv_b_name);
        const wkv_biases: ?Array = blk_b: {
            const v = weights.get(wkv_b_name);
            if (v != null) {
                if (weights.fetchRemove(wkv_b_name)) |kv2| allocator.free(kv2.key);
            }
            break :blk_b v;
        };

        const wo_a_name = try std.fmt.allocPrint(allocator, "{s}attn.wo_a.weight", .{idx_fmt});
        defer allocator.free(wo_a_name);
        const wo_a_raw = weights.get(wo_a_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(wo_a_name)) |kv| allocator.free(kv.key);
        // wo_a needs dequantize for reshape (grouped LoRA) — lazy node
        const wo_a_base = try std.fmt.allocPrint(allocator, "{s}attn.wo_a", .{idx_fmt});
        defer allocator.free(wo_a_base);
        const wo_a_deq = try dequantIfNeeded(allocator, weights, wo_a_base, wo_a_raw, config, ctx);

        // If wo_a is 2D but o_groups > 1, reshape to 3D for grouped LoRA path
        var wo_a = wo_a_deq;
        const attn_o_groups = config.o_groups;
        const attn_o_lora_rank = config.o_lora_rank;
        const attn_num_heads = config.num_attention_heads;
        const attn_head_dim = config.head_dim;
        if (wo_a.ndim() == 2 and attn_o_groups > 1 and attn_num_heads % attn_o_groups == 0) {
            const group_feat = (attn_num_heads * attn_head_dim) / attn_o_groups;
            wo_a = try ops.reshape(ctx, wo_a_deq, &[_]i32{ @intCast(attn_o_groups), @intCast(attn_o_lora_rank), @intCast(group_feat) });
        }

        const wo_b_name = try std.fmt.allocPrint(allocator, "{s}attn.wo_b.weight", .{idx_fmt});
        defer allocator.free(wo_b_name);
        const wo_b = weights.get(wo_b_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(wo_b_name)) |kv| allocator.free(kv.key);
        const wo_b_s_name = try std.fmt.allocPrint(allocator, "{s}attn.wo_b.scales", .{idx_fmt});
        defer allocator.free(wo_b_s_name);
        const wo_b_scales: ?Array = blk_s: {
            const v = weights.get(wo_b_s_name);
            if (v != null) {
                if (weights.fetchRemove(wo_b_s_name)) |kv2| allocator.free(kv2.key);
            }
            break :blk_s v;
        };
        const wo_b_b_name = try std.fmt.allocPrint(allocator, "{s}attn.wo_b.biases", .{idx_fmt});
        defer allocator.free(wo_b_b_name);
        const wo_b_biases: ?Array = blk_b: {
            const v = weights.get(wo_b_b_name);
            if (v != null) {
                if (weights.fetchRemove(wo_b_b_name)) |kv2| allocator.free(kv2.key);
            }
            break :blk_b v;
        };

        // Attention norms
        const q_norm_name = try std.fmt.allocPrint(allocator, "{s}attn.q_norm.weight", .{idx_fmt});
        defer allocator.free(q_norm_name);
        const q_norm_w = weights.get(q_norm_name) orelse return LoadError.MissingWeight;
        var q_norm = try nn.RMSNorm.init(ctx, config.q_lora_rank, config.rms_norm_eps);
        q_norm.weight.deinit();
        q_norm.weight = q_norm_w;
        if (weights.fetchRemove(q_norm_name)) |kv| allocator.free(kv.key);

        const kv_norm_name = try std.fmt.allocPrint(allocator, "{s}attn.kv_norm.weight", .{idx_fmt});
        defer allocator.free(kv_norm_name);
        const kv_norm_w = weights.get(kv_norm_name) orelse return LoadError.MissingWeight;
        var kv_norm = try nn.RMSNorm.init(ctx, head_dim, config.rms_norm_eps);
        kv_norm.weight.deinit();
        kv_norm.weight = kv_norm_w;
        if (weights.fetchRemove(kv_norm_name)) |kv| allocator.free(kv.key);

        // Attn sink: learnable logits for attention softmax denominator
        var sink_logits: ?Array = null;
        const attn_sink_name = try std.fmt.allocPrint(allocator, "{s}attn.attn_sink", .{idx_fmt});
        defer allocator.free(attn_sink_name);
        if (weights.get(attn_sink_name)) |sl| {
            sink_logits = sl;
            if (weights.fetchRemove(attn_sink_name)) |kv| allocator.free(kv.key);
        }

        // RoPE
        const compress_ratio = if (i < config.compress_ratios.len) config.compress_ratios[i] else 0;
        const rope_config = config.rope_scaling orelse DSV4Config.YarnRoPEConfig{};
        const rope = try deepseek_v4.DSV4YarnRoPE.init(ctx, rope_dim, config.max_position_embeddings, config.rope_theta, rope_config);

        // Compressor module (for CSA/HCA layers with compress_ratio > 0)
        var compressor: ?deepseek_v4.Compressor = null;
        var compress_gate_weight: ?Array = null;
        var compress_pos_bias: ?Array = null;
        if (compress_ratio > 0) {
            const comp_wkv_name = try std.fmt.allocPrint(allocator, "{s}attn.compressor.wkv.weight", .{idx_fmt});
            defer allocator.free(comp_wkv_name);
            const comp_wgate_name = try std.fmt.allocPrint(allocator, "{s}attn.compressor.wgate.weight", .{idx_fmt});
            defer allocator.free(comp_wgate_name);
            const comp_ape_name = try std.fmt.allocPrint(allocator, "{s}attn.compressor.ape", .{idx_fmt});
            defer allocator.free(comp_ape_name);
            const comp_norm_name = try std.fmt.allocPrint(allocator, "{s}attn.compressor.norm.weight", .{idx_fmt});
            defer allocator.free(comp_norm_name);

            // Try loading new-style Compressor weights
            const comp_wkv = weights.get(comp_wkv_name);
            const comp_wgate = weights.get(comp_wgate_name);
            const comp_ape = weights.get(comp_ape_name);
            const comp_norm = weights.get(comp_norm_name);

            if (comp_wkv != null and comp_wgate != null and comp_ape != null and comp_norm != null) {
                if (weights.fetchRemove(comp_wkv_name)) |kv| allocator.free(kv.key);
                if (weights.fetchRemove(comp_wgate_name)) |kv| allocator.free(kv.key);
                if (weights.fetchRemove(comp_ape_name)) |kv| allocator.free(kv.key);
                if (weights.fetchRemove(comp_norm_name)) |kv| allocator.free(kv.key);
                // Also consume quantized metadata (.scales/.biases)
                const comp_base = try std.fmt.allocPrint(allocator, "{s}attn.compressor.wkv", .{idx_fmt});
                defer allocator.free(comp_base);
                consumeWeightKey(allocator, weights, comp_base);
                const comp_gate_base = try std.fmt.allocPrint(allocator, "{s}attn.compressor.wgate", .{idx_fmt});
                defer allocator.free(comp_gate_base);
                consumeWeightKey(allocator, weights, comp_gate_base);

                const comp_overlap = compress_ratio == 4;
                const comp_out_dim = if (comp_overlap) head_dim * 2 else head_dim;
                _ = comp_out_dim;

                compressor = deepseek_v4.Compressor{
                    .ctx = ctx,
                    .wkv = comp_wkv.?,
                    .wgate = comp_wgate.?,
                    .ape = comp_ape.?,
                    .norm_weight = comp_norm.?,
                    .compress_ratio = compress_ratio,
                    .head_dim = head_dim,
                    .rope_head_dim = rope_dim,
                    .overlap = comp_overlap,
                    .out_dim = if (comp_overlap) head_dim * 2 else head_dim,
                    .norm_eps = config.rms_norm_eps,
                };
            }
            // Also try loading old-style compress_gate_weight (backward compat)
            if (compressor == null) {
                const cgw_name = try std.fmt.allocPrint(allocator, "{s}attn.compress_gate_weight", .{idx_fmt});
                defer allocator.free(cgw_name);
                if (weights.get(cgw_name)) |cgw| {
                    compress_gate_weight = cgw;
                    if (weights.fetchRemove(cgw_name)) |kv| allocator.free(kv.key);
                }
                const cpb_name = try std.fmt.allocPrint(allocator, "{s}attn.compress_pos_bias", .{idx_fmt});
                defer allocator.free(cpb_name);
                if (weights.get(cpb_name)) |cpb| {
                    compress_pos_bias = cpb;
                    if (weights.fetchRemove(cpb_name)) |kv| allocator.free(kv.key);
                }
            }
        }

        // Indexer module (for CSA layers with compress_ratio == 4)
        var indexer_new: ?deepseek_v4.Indexer = null;
        var indexer_legacy: ?deepseek_v4.LightningIndexer = null;
        if (compress_ratio == 4) {
            // Try new-style Indexer weights first
            const idx_wqb_name = try std.fmt.allocPrint(allocator, "{s}attn.indexer.wq_b.weight", .{idx_fmt});
            defer allocator.free(idx_wqb_name);
            const idx_wp_name = try std.fmt.allocPrint(allocator, "{s}attn.indexer.weights_proj.weight", .{idx_fmt});
            defer allocator.free(idx_wp_name);

            const idx_wqb = weights.get(idx_wqb_name);
            const idx_wp = weights.get(idx_wp_name);

            if (idx_wqb != null and idx_wp != null and compressor != null) {
                if (weights.fetchRemove(idx_wqb_name)) |kv| allocator.free(kv.key);
                if (weights.fetchRemove(idx_wp_name)) |kv| allocator.free(kv.key);
                // Consume quantized metadata for indexer weights
                const idx_wqb_base = try std.fmt.allocPrint(allocator, "{s}attn.indexer.wq_b", .{idx_fmt});
                defer allocator.free(idx_wqb_base);
                consumeWeightKey(allocator, weights, idx_wqb_base);
                const idx_wp_base = try std.fmt.allocPrint(allocator, "{s}attn.indexer.weights_proj", .{idx_fmt});
                defer allocator.free(idx_wp_base);
                consumeWeightKey(allocator, weights, idx_wp_base);

                // Load nested Indexer.compressor weights
                const idx_comp_wkv_name = try std.fmt.allocPrint(allocator, "{s}attn.indexer.compressor.wkv.weight", .{idx_fmt});
                defer allocator.free(idx_comp_wkv_name);
                const idx_comp_wgate_name = try std.fmt.allocPrint(allocator, "{s}attn.indexer.compressor.wgate.weight", .{idx_fmt});
                defer allocator.free(idx_comp_wgate_name);
                const idx_comp_ape_name = try std.fmt.allocPrint(allocator, "{s}attn.indexer.compressor.ape", .{idx_fmt});
                defer allocator.free(idx_comp_ape_name);
                const idx_comp_norm_name = try std.fmt.allocPrint(allocator, "{s}attn.indexer.compressor.norm.weight", .{idx_fmt});
                defer allocator.free(idx_comp_norm_name);

                const ic_wkv = weights.get(idx_comp_wkv_name) orelse return LoadError.MissingWeight;
                const ic_wgate = weights.get(idx_comp_wgate_name) orelse return LoadError.MissingWeight;
                const ic_ape = weights.get(idx_comp_ape_name) orelse return LoadError.MissingWeight;
                const ic_norm = weights.get(idx_comp_norm_name) orelse return LoadError.MissingWeight;
                if (weights.fetchRemove(idx_comp_wkv_name)) |kv| allocator.free(kv.key);
                if (weights.fetchRemove(idx_comp_wgate_name)) |kv| allocator.free(kv.key);
                if (weights.fetchRemove(idx_comp_ape_name)) |kv| allocator.free(kv.key);
                if (weights.fetchRemove(idx_comp_norm_name)) |kv| allocator.free(kv.key);
                // Consume quantized metadata for nested indexer.compressor
                const ic_wkv_base = try std.fmt.allocPrint(allocator, "{s}attn.indexer.compressor.wkv", .{idx_fmt});
                defer allocator.free(ic_wkv_base);
                consumeWeightKey(allocator, weights, ic_wkv_base);
                const ic_wgate_base = try std.fmt.allocPrint(allocator, "{s}attn.indexer.compressor.wgate", .{idx_fmt});
                defer allocator.free(ic_wgate_base);
                consumeWeightKey(allocator, weights, ic_wgate_base);

                const idx_head_dim = config.index_head_dim;
                const idx_overlap = compress_ratio == 4;

                indexer_new = deepseek_v4.Indexer{
                    .ctx = ctx,
                    .n_heads = config.index_n_heads,
                    .head_dim = idx_head_dim,
                    .index_topk = config.index_topk,
                    .wq_b = idx_wqb.?,
                    .weights_proj = idx_wp.?,
                    .compressor = deepseek_v4.Compressor{
                        .ctx = ctx,
                        .wkv = ic_wkv,
                        .wgate = ic_wgate,
                        .ape = ic_ape,
                        .norm_weight = ic_norm,
                        .compress_ratio = compress_ratio,
                        .head_dim = idx_head_dim,
                        .rope_head_dim = rope_dim,
                        .overlap = idx_overlap,
                        .out_dim = if (idx_overlap) idx_head_dim * 2 else idx_head_dim,
                        .norm_eps = config.rms_norm_eps,
                    },
                    .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(idx_head_dim))),
                };
            } else {
                // Fallback: try legacy LightningIndexer weights
                const wq_idx_name = try std.fmt.allocPrint(allocator, "{s}attn.indexer.wq.weight", .{idx_fmt});
                defer allocator.free(wq_idx_name);
                const wk_idx_name = try std.fmt.allocPrint(allocator, "{s}attn.indexer.wk.weight", .{idx_fmt});
                defer allocator.free(wk_idx_name);

                const wq_idx = weights.get(wq_idx_name);
                const wk_idx = weights.get(wk_idx_name);

                if (wq_idx != null and wk_idx != null) {
                    if (weights.fetchRemove(wq_idx_name)) |kv| allocator.free(kv.key);
                    if (weights.fetchRemove(wk_idx_name)) |kv| allocator.free(kv.key);
                    indexer_legacy = deepseek_v4.LightningIndexer{
                        .ctx = ctx,
                        .wq_index = wq_idx.?,
                        .wk_index = wk_idx.?,
                        .index_n_heads = config.index_n_heads,
                        .index_head_dim = config.index_head_dim,
                        .index_topk = config.index_topk,
                    };
                }
            }
        }

        // --- Attention struct ---
        const qgs = config.quantize_default_group_size;
        const qbits: u8 = if (config.quantize_default_bits > 0) config.quantize_default_bits else 4;
        const attention = deepseek_v4.DSV4Attention{
            .ctx = ctx,
            .config = config,
            .layer_idx = i,
            .wq_a = wq_a,
            .wq_a_scales = wq_a_scales,
            .wq_a_biases = wq_a_biases,
            .wq_b = wq_b,
            .wq_b_scales = wq_b_scales,
            .wq_b_biases = wq_b_biases,
            .q_norm = q_norm,
            .wkv = wkv,
            .wkv_scales = wkv_scales,
            .wkv_biases = wkv_biases,
            .kv_norm = kv_norm,
            .kv_b = null,
            .wo_a = wo_a,
            .wo_a_scales = null,
            .wo_a_biases = null,
            .wo_b = wo_b,
            .wo_b_scales = wo_b_scales,
            .wo_b_biases = wo_b_biases,
            .attn_quant_group_size = if (wq_a_biases == null) 32 else qgs,
            .attn_quant_bits = qbits,
            .attn_quant_mode = if (wq_a_biases == null) .mxfp4 else .affine,
            .rope = rope,
            .compress_ratio = compress_ratio,
            .compress_gate_weight = compress_gate_weight,
            .compress_pos_bias = compress_pos_bias,
            .indexer = indexer_legacy,
            .sink_logits = sink_logits,
        };

        // Consume quantized metadata for attention weights
        {
            const attn_bases = [_][]const u8{ "attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b" };
            for (attn_bases) |base| {
                const full = try std.fmt.allocPrint(allocator, "{s}{s}", .{ idx_fmt, base });
                defer allocator.free(full);
                consumeWeightKey(allocator, weights, full);
            }
        }

        // --- MoE weights ---
        // Gate
        const gate_weight_name = try std.fmt.allocPrint(allocator, "{s}ffn.gate.weight", .{idx_fmt});
        defer allocator.free(gate_weight_name);
        const gate_weight = weights.get(gate_weight_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(gate_weight_name)) |kv| allocator.free(kv.key);

        var gate_bias: ?Array = null;
        const gate_bias_name = try std.fmt.allocPrint(allocator, "{s}ffn.gate.bias", .{idx_fmt});
        defer allocator.free(gate_bias_name);
        if (weights.get(gate_bias_name)) |gb| {
            gate_bias = gb;
            if (weights.fetchRemove(gate_bias_name)) |kv| allocator.free(kv.key);
        }

        var tid2eid: ?Array = null;
        const tid2eid_name = try std.fmt.allocPrint(allocator, "{s}ffn.gate.tid2eid", .{idx_fmt});
        defer allocator.free(tid2eid_name);
        if (weights.get(tid2eid_name)) |te| {
            tid2eid = te;
            if (weights.fetchRemove(tid2eid_name)) |kv| allocator.free(kv.key);
        }

        const is_hash = i < config.num_hash_layers;

        // Build smelt_mask - auto-detect available experts if not explicitly using smelt
        var smelt_mask: []bool = undefined;
        var smelt_mask_owned: bool = false;

        if (smelt.enabled and smelt.load_mode == .preload) {
            // Preload mode: create smelt_mask to restrict router to loaded experts
            smelt_mask = try smelt.buildMask(allocator, n_routed_experts);
            smelt_mask_owned = true;
            std.log.info("Layer {d}: Preload mode - router restricted to {d}/{d} experts", .{
                i,
                @as(usize, @intFromFloat(@as(f32, @floatFromInt(n_routed_experts)) * smelt.load_fraction)),
                n_routed_experts,
            });
        } else if (smelt.enabled and smelt.load_mode == .stream) {
            // Stream mode: NO smelt_mask - router can select any expert (0-255)
            // Experts will be loaded on-demand from disk
            std.log.info("Layer {d}: Stream mode - router can select any expert (on-demand loading)", .{i});
            smelt_mask = undefined; // Don't allocate - we won't use it
            smelt_mask_owned = false;
        } else {
            // Auto-detect available experts
            smelt_mask = try allocator.alloc(bool, n_routed_experts);
            smelt_mask_owned = true;
            @memset(smelt_mask, false);

            var n_available: usize = 0;
            for (0..n_routed_experts) |e| {
                const ew1_name = try std.fmt.allocPrint(allocator, "{s}ffn.experts.{d}.w1.weight", .{ idx_fmt, e });
                defer allocator.free(ew1_name);
                if (weights.get(ew1_name) != null) {
                    smelt_mask[e] = true;
                    n_available += 1;
                }
            }

            if (n_available < n_routed_experts and n_available > 0) {
                std.log.warn("⚠️  Layer {d}: Partial expert model detected: {d}/{d} experts available", .{ i, n_available, n_routed_experts });
                std.log.warn("Auto-enabling smelt mode for this layer.", .{});
            }
        }

        // Don't defer free - gate takes ownership of smelt_mask
        // defer if (smelt_mask_owned) allocator.free(smelt_mask);

        const gate = deepseek_v4.DSV4Gate{
            .ctx = ctx,
            .weight = gate_weight,
            .bias = gate_bias,
            .tid2eid = tid2eid,
            .topk = config.num_experts_per_tok,
            .n_routed_experts = n_routed_experts,
            .route_scale = config.routed_scaling_factor,
            .scoring_func = config.scoring_func,
            .is_hash = is_hash,
            .smelt_mask = if (smelt_mask_owned and smelt.load_mode == .preload) smelt_mask else null,
            .allocator = if (smelt_mask_owned and smelt.load_mode == .preload) allocator else null,
        };

        // Shared expert — keep quantized (lazy, no memory allocation)
        const shared_w1_name = try std.fmt.allocPrint(allocator, "{s}ffn.shared_experts.w1.weight", .{idx_fmt});
        defer allocator.free(shared_w1_name);
        const shared_w1 = weights.get(shared_w1_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(shared_w1_name)) |kv| allocator.free(kv.key);

        const shared_w3_name = try std.fmt.allocPrint(allocator, "{s}ffn.shared_experts.w3.weight", .{idx_fmt});
        defer allocator.free(shared_w3_name);
        const shared_w3 = weights.get(shared_w3_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(shared_w3_name)) |kv| allocator.free(kv.key);

        const shared_w2_name = try std.fmt.allocPrint(allocator, "{s}ffn.shared_experts.w2.weight", .{idx_fmt});
        defer allocator.free(shared_w2_name);
        const shared_w2 = weights.get(shared_w2_name) orelse return LoadError.MissingWeight;
        if (weights.fetchRemove(shared_w2_name)) |kv| allocator.free(kv.key);

        // Load shared expert quantized scales/biases
        const se_w1_s_name = try std.fmt.allocPrint(allocator, "{s}ffn.shared_experts.w1.scales", .{idx_fmt});
        defer allocator.free(se_w1_s_name);
        const se_w1_scales: ?Array = blk_s: {
            const v = weights.get(se_w1_s_name);
            if (v != null) {
                if (weights.fetchRemove(se_w1_s_name)) |kv2| allocator.free(kv2.key);
            }
            break :blk_s v;
        };
        const se_w1_b_name = try std.fmt.allocPrint(allocator, "{s}ffn.shared_experts.w1.biases", .{idx_fmt});
        defer allocator.free(se_w1_b_name);
        const se_w1_biases: ?Array = blk_b: {
            const v = weights.get(se_w1_b_name);
            if (v != null) {
                if (weights.fetchRemove(se_w1_b_name)) |kv2| allocator.free(kv2.key);
            }
            break :blk_b v;
        };

        const se_w3_s_name = try std.fmt.allocPrint(allocator, "{s}ffn.shared_experts.w3.scales", .{idx_fmt});
        defer allocator.free(se_w3_s_name);
        const se_w3_scales: ?Array = blk_s: {
            const v = weights.get(se_w3_s_name);
            if (v != null) {
                if (weights.fetchRemove(se_w3_s_name)) |kv2| allocator.free(kv2.key);
            }
            break :blk_s v;
        };
        const se_w3_b_name = try std.fmt.allocPrint(allocator, "{s}ffn.shared_experts.w3.biases", .{idx_fmt});
        defer allocator.free(se_w3_b_name);
        const se_w3_biases: ?Array = blk_b: {
            const v = weights.get(se_w3_b_name);
            if (v != null) {
                if (weights.fetchRemove(se_w3_b_name)) |kv2| allocator.free(kv2.key);
            }
            break :blk_b v;
        };

        const se_w2_s_name = try std.fmt.allocPrint(allocator, "{s}ffn.shared_experts.w2.scales", .{idx_fmt});
        defer allocator.free(se_w2_s_name);
        const se_w2_scales: ?Array = blk_s: {
            const v = weights.get(se_w2_s_name);
            if (v != null) {
                if (weights.fetchRemove(se_w2_s_name)) |kv2| allocator.free(kv2.key);
            }
            break :blk_s v;
        };
        const se_w2_b_name = try std.fmt.allocPrint(allocator, "{s}ffn.shared_experts.w2.biases", .{idx_fmt});
        defer allocator.free(se_w2_b_name);
        const se_w2_biases: ?Array = blk_b: {
            const v = weights.get(se_w2_b_name);
            if (v != null) {
                if (weights.fetchRemove(se_w2_b_name)) |kv2| allocator.free(kv2.key);
            }
            break :blk_b v;
        };

        const shared_expert = deepseek_v4.DSV4Expert{
            .ctx = ctx,
            .w1 = shared_w1,
            .w2 = shared_w2,
            .w3 = shared_w3,
            .w1_scales = se_w1_scales,
            .w1_biases = se_w1_biases,
            .w2_scales = se_w2_scales,
            .w2_biases = se_w2_biases,
            .w3_scales = se_w3_scales,
            .w3_biases = se_w3_biases,
            .quant_group_size = if (se_w1_biases == null) 32 else config.quantize_default_group_size,
            .quant_bits = if (config.quantize_default_bits > 0) config.quantize_default_bits else 4,
            .quant_mode = if (se_w1_biases == null) .mxfp4 else .affine,
            .swiglu_limit = 0,
        };
        // Consume any remaining shared expert quantized metadata
        {
            const se_bases = [_][]const u8{ "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3" };
            for (se_bases) |base| {
                const full = try std.fmt.allocPrint(allocator, "{s}{s}", .{ idx_fmt, base });
                defer allocator.free(full);
                consumeWeightKey(allocator, weights, full);
            }
        }

        // Check if expert weights are available (smelt may have skipped them)
        const has_fused_experts = blk_check: {
            const check_name = try std.fmt.allocPrint(allocator, "{s}ffn.switch_mlp.w1.weight", .{idx_fmt});
            defer allocator.free(check_name);
            const check_name2 = try std.fmt.allocPrint(allocator, "{s}ffn.experts.0.w1.weight", .{idx_fmt});
            defer allocator.free(check_name2);
            break :blk_check weights.get(check_name) != null or weights.get(check_name2) != null;
        };

        const moe = deepseek_v4.DSV4MoE{
            .ctx = ctx,
            .gate = gate,
            .switch_mlp = blk: {
                // Try to load fused switch_mlp weights first (mlx-lm format)
                const gate_proj_name = try std.fmt.allocPrint(allocator, "{s}ffn.switch_mlp.w1.weight", .{idx_fmt});
                defer allocator.free(gate_proj_name);
                const up_proj_name = try std.fmt.allocPrint(allocator, "{s}ffn.switch_mlp.w3.weight", .{idx_fmt});
                defer allocator.free(up_proj_name);
                const down_proj_name = try std.fmt.allocPrint(allocator, "{s}ffn.switch_mlp.w2.weight", .{idx_fmt});
                defer allocator.free(down_proj_name);

                var fused_gate = weights.get(gate_proj_name);
                var fused_up = weights.get(up_proj_name);
                var fused_down = weights.get(down_proj_name);

                // If smelt skipped all expert weights, create a dummy SwitchGLU
                if (fused_gate == null and fused_up == null and fused_down == null) {
                    // Check if individual experts exist
                    const ew1_name = try std.fmt.allocPrint(allocator, "{s}ffn.experts.0.w1.weight", .{idx_fmt});
                    defer allocator.free(ew1_name);
                    if (weights.get(ew1_name) == null) {
                        // No expert weights at all — shared-expert-only mode
                        // Create a minimal dummy SwitchGLU (won't be called in forward)
                        var dummy_raw = c.c.mlx_array_new();
                        const dummy_shape = [_]i32{ 1, 1, 1 };
                        try c.check(c.c.mlx_zeros(&dummy_raw, &dummy_shape, 3, c.c.MLX_FLOAT32, ctx.stream.inner));
                        const dummy = Array.fromHandle(dummy_raw);
                        break :blk deepseek_v4.DSV4SwitchGLU{
                            .ctx = ctx,
                            .gate_proj = dummy,
                            .up_proj = dummy,
                            .down_proj = dummy,
                            .gate_proj_scales = null,
                            .gate_proj_biases = null,
                            .up_proj_scales = null,
                            .up_proj_biases = null,
                            .down_proj_scales = null,
                            .down_proj_biases = null,
                            .is_quantized = false,
                            .quant_group_size = 32,
                            .quant_bits = 4,
                            .quant_mode = "mxfp4",
                            .swiglu_limit = config.swiglu_limit,
                            .sort_threshold = 8,
                        };
                    }
                }

                if (fused_gate != null and fused_up != null and fused_down != null) {
                    // Fused format available — use directly
                    if (weights.fetchRemove(gate_proj_name)) |kv| allocator.free(kv.key);
                    if (weights.fetchRemove(up_proj_name)) |kv| allocator.free(kv.key);
                    if (weights.fetchRemove(down_proj_name)) |kv| allocator.free(kv.key);
                } else {
                    // Individual experts — stack into fused format
                    // This handles checkpoints with experts.{e}.w1.weight format

                    // First pass: detect which experts are actually available
                    var available_experts = std.ArrayList(usize).empty;
                    defer available_experts.deinit(allocator);

                    for (0..n_routed_experts) |e| {
                        const ew1_name = try std.fmt.allocPrint(allocator, "{s}ffn.experts.{d}.w1.weight", .{ idx_fmt, e });
                        defer allocator.free(ew1_name);
                        if (weights.get(ew1_name) != null) {
                            try available_experts.append(allocator, e);
                        }
                    }

                    const n_available = available_experts.items.len;

                    // If no experts or only partial experts, auto-enable smelt mode
                    if (n_available == 0) {
                        std.log.warn("⚠️  No expert weights found in model files", .{});
                        std.log.warn("This model may be shared-expert-only or incorrectly formatted.", .{});
                        // Continue with dummy SwitchGLU (already handled above)
                        const dummy = try creation.zeros(ctx, &[_]i32{ 1, 1, 1 }, .float32);
                        break :blk deepseek_v4.DSV4SwitchGLU{
                            .ctx = ctx,
                            .gate_proj = dummy,
                            .up_proj = dummy,
                            .down_proj = dummy,
                            .gate_proj_scales = null,
                            .gate_proj_biases = null,
                            .up_proj_scales = null,
                            .up_proj_biases = null,
                            .down_proj_scales = null,
                            .down_proj_biases = null,
                            .is_quantized = false,
                            .quant_group_size = 32,
                            .quant_bits = 4,
                            .quant_mode = "mxfp4",
                            .swiglu_limit = config.swiglu_limit,
                            .sort_threshold = 8,
                        };
                    }

                    if (n_available < n_routed_experts) {
                        std.log.warn("⚠️  Partial expert model detected: {d}/{d} experts available", .{ n_available, n_routed_experts });
                        std.log.warn("Auto-enabling smelt mode for partial expert loading.", .{});
                        std.log.warn("💡 Tip: Use --smelt flag explicitly to suppress this warning.", .{});
                    }

                    var gate_list = try allocator.alloc(Array, n_available);
                    defer allocator.free(gate_list);
                    var up_list = try allocator.alloc(Array, n_available);
                    defer allocator.free(up_list);
                    var down_list = try allocator.alloc(Array, n_available);
                    defer allocator.free(down_list);

                    for (available_experts.items, 0..) |e, idx| {
                        const ew1_name = try std.fmt.allocPrint(allocator, "{s}ffn.experts.{d}.w1.weight", .{ idx_fmt, e });
                        defer allocator.free(ew1_name);
                        const ew3_name = try std.fmt.allocPrint(allocator, "{s}ffn.experts.{d}.w3.weight", .{ idx_fmt, e });
                        defer allocator.free(ew3_name);
                        const ew2_name = try std.fmt.allocPrint(allocator, "{s}ffn.experts.{d}.w2.weight", .{ idx_fmt, e });
                        defer allocator.free(ew2_name);

                        gate_list[idx] = weights.get(ew1_name) orelse {
                            std.log.err("❌ Inconsistent expert weights: {s} missing but was detected earlier", .{ew1_name});
                            return LoadError.MissingWeight;
                        };
                        up_list[idx] = weights.get(ew3_name) orelse {
                            std.log.err("❌ Inconsistent expert weights: {s} missing", .{ew3_name});
                            return LoadError.MissingWeight;
                        };
                        down_list[idx] = weights.get(ew2_name) orelse {
                            std.log.err("❌ Inconsistent expert weights: {s} missing", .{ew2_name});
                            return LoadError.MissingWeight;
                        };

                        if (weights.fetchRemove(ew1_name)) |kv| allocator.free(kv.key);
                        if (weights.fetchRemove(ew3_name)) |kv| allocator.free(kv.key);
                        if (weights.fetchRemove(ew2_name)) |kv| allocator.free(kv.key);
                    }

                    // Stack: [n_available] × [out, in] → [n_available, out, in]
                    // Use expandDims + concatenate as stack equivalent
                    var gate_expanded = try allocator.alloc(Array, n_available);
                    defer allocator.free(gate_expanded);
                    var up_expanded = try allocator.alloc(Array, n_available);
                    defer allocator.free(up_expanded);
                    var down_expanded = try allocator.alloc(Array, n_available);
                    defer allocator.free(down_expanded);

                    for (0..n_available) |idx| {
                        gate_expanded[idx] = try ops.expandDims(ctx, gate_list[idx], 0);
                        up_expanded[idx] = try ops.expandDims(ctx, up_list[idx], 0);
                        down_expanded[idx] = try ops.expandDims(ctx, down_list[idx], 0);
                    }
                    defer for (0..n_available) |idx| {
                        gate_expanded[idx].deinit();
                        up_expanded[idx].deinit();
                        down_expanded[idx].deinit();
                    };

                    fused_gate = try shape_mod.concatenateAxis(ctx, gate_expanded, 0);
                    fused_up = try shape_mod.concatenateAxis(ctx, up_expanded, 0);
                    fused_down = try shape_mod.concatenateAxis(ctx, down_expanded, 0);
                }

                break :blk blk2: {
                    // Check if quantized scales exist for fused weights
                    const gate_scales_name = try std.fmt.allocPrint(allocator, "{s}ffn.switch_mlp.w1.scales", .{idx_fmt});
                    defer allocator.free(gate_scales_name);
                    const up_scales_name = try std.fmt.allocPrint(allocator, "{s}ffn.switch_mlp.w3.scales", .{idx_fmt});
                    defer allocator.free(up_scales_name);
                    const down_scales_name = try std.fmt.allocPrint(allocator, "{s}ffn.switch_mlp.w2.scales", .{idx_fmt});
                    defer allocator.free(down_scales_name);

                    const gate_scales = weights.get(gate_scales_name);
                    const up_scales = weights.get(up_scales_name);
                    const down_scales = weights.get(down_scales_name);

                    const is_quant = gate_scales != null and up_scales != null and down_scales != null;

                    // Load biases if present
                    var gate_biases: ?Array = null;
                    var up_biases: ?Array = null;
                    var down_biases: ?Array = null;
                    if (is_quant) {
                        if (weights.fetchRemove(gate_scales_name)) |kv| allocator.free(kv.key);
                        if (weights.fetchRemove(up_scales_name)) |kv| allocator.free(kv.key);
                        if (weights.fetchRemove(down_scales_name)) |kv| allocator.free(kv.key);

                        const gate_biases_name = try std.fmt.allocPrint(allocator, "{s}ffn.switch_mlp.w1.biases", .{idx_fmt});
                        defer allocator.free(gate_biases_name);
                        const up_biases_name = try std.fmt.allocPrint(allocator, "{s}ffn.switch_mlp.w3.biases", .{idx_fmt});
                        defer allocator.free(up_biases_name);
                        const down_biases_name = try std.fmt.allocPrint(allocator, "{s}ffn.switch_mlp.w2.biases", .{idx_fmt});
                        defer allocator.free(down_biases_name);

                        gate_biases = weights.get(gate_biases_name);
                        up_biases = weights.get(up_biases_name);
                        down_biases = weights.get(down_biases_name);
                        if (gate_biases != null) {
                            if (weights.fetchRemove(gate_biases_name)) |kv| allocator.free(kv.key);
                        }
                        if (up_biases != null) {
                            if (weights.fetchRemove(up_biases_name)) |kv| allocator.free(kv.key);
                        }
                        if (down_biases != null) {
                            if (weights.fetchRemove(down_biases_name)) |kv| allocator.free(kv.key);
                        }
                    }

                    // Expert weights always use mxfp4 (group_size=32, no biases)
                    // This is specified in the per-weight quantization_config in config.json
                    const qmode: []const u8 = if (gate_biases == null) "mxfp4" else config.quantize_default_mode;
                    const expert_qbits: u8 = if (config.quantize_default_bits > 0) config.quantize_default_bits else 4;
                    const qgroup: i32 = if (gate_biases == null) 32 else config.quantize_default_group_size;

                    break :blk2 deepseek_v4.DSV4SwitchGLU{
                        .ctx = ctx,
                        .gate_proj = fused_gate.?,
                        .up_proj = fused_up.?,
                        .down_proj = fused_down.?,
                        .gate_proj_scales = gate_scales,
                        .gate_proj_biases = gate_biases,
                        .up_proj_scales = up_scales,
                        .up_proj_biases = up_biases,
                        .down_proj_scales = down_scales,
                        .down_proj_biases = down_biases,
                        .is_quantized = is_quant,
                        .quant_group_size = qgroup,
                        .quant_bits = expert_qbits,
                        .quant_mode = qmode,
                        .swiglu_limit = config.swiglu_limit,
                        .sort_threshold = 8,
                    };
                };
            },
            .shared_expert = shared_expert,
            .n_routed_experts = n_routed_experts,
            .n_activated_experts = config.num_experts_per_tok,
            .experts_loaded = has_fused_experts,
            .expert_remap = if (smelt.enabled and smelt.load_fraction < 1.0 and has_fused_experts) blk_remap: {
                // Build remap: original_expert_id → sliced_row_index
                // For unloaded experts, map to 0 (will be masked by gate scores anyway)
                const mask = try smelt.buildMask(allocator, n_routed_experts);
                defer allocator.free(mask);
                var remap_data = try allocator.alloc(i32, n_routed_experts);
                defer allocator.free(remap_data);
                var sliced_idx: i32 = 0;
                for (mask, 0..) |m, ei| {
                    if (m) {
                        remap_data[ei] = sliced_idx;
                        sliced_idx += 1;
                    } else {
                        remap_data[ei] = 0; // Map unloaded experts to row 0
                    }
                }
                break :blk_remap try Array.fromData(allocator, i32, remap_data, &[_]i32{@intCast(n_routed_experts)});
            } else null,
        };

        // --- Layer norms ---
        const attn_norm_name = try std.fmt.allocPrint(allocator, "{s}attn_norm.weight", .{idx_fmt});
        defer allocator.free(attn_norm_name);
        const attn_norm_w = weights.get(attn_norm_name) orelse return LoadError.MissingWeight;
        var attn_norm = try nn.RMSNorm.init(ctx, hidden_size, config.rms_norm_eps);
        attn_norm.weight.deinit();
        attn_norm.weight = attn_norm_w;
        if (weights.fetchRemove(attn_norm_name)) |kv| allocator.free(kv.key);

        const ffn_norm_name = try std.fmt.allocPrint(allocator, "{s}ffn_norm.weight", .{idx_fmt});
        defer allocator.free(ffn_norm_name);
        const ffn_norm_w = weights.get(ffn_norm_name) orelse return LoadError.MissingWeight;
        var ffn_norm = try nn.RMSNorm.init(ctx, hidden_size, config.rms_norm_eps);
        ffn_norm.weight.deinit();
        ffn_norm.weight = ffn_norm_w;
        if (weights.fetchRemove(ffn_norm_name)) |kv| allocator.free(kv.key);

        // --- mHC parameters (optional, loaded but may not be used in simplified version) ---
        const hc_attn_fn_name = try std.fmt.allocPrint(allocator, "{s}hc_attn_fn", .{idx_fmt});
        defer allocator.free(hc_attn_fn_name);
        const hc_attn_fn = weights.get(hc_attn_fn_name);
        if (hc_attn_fn == null and i == 0) std.log.warn("mHC: hc_attn_fn NOT found for layer 0 (key: {s})", .{hc_attn_fn_name});
        if (hc_attn_fn != null and i == 0) std.log.info("mHC: hc_attn_fn found for layer 0, ndim={d}", .{hc_attn_fn.?.ndim()});
        if (weights.fetchRemove(hc_attn_fn_name)) |kv| allocator.free(kv.key);

        const hc_attn_base_name = try std.fmt.allocPrint(allocator, "{s}hc_attn_base", .{idx_fmt});
        defer allocator.free(hc_attn_base_name);
        const hc_attn_base = weights.get(hc_attn_base_name);
        if (weights.fetchRemove(hc_attn_base_name)) |kv| allocator.free(kv.key);

        const hc_attn_scale_name = try std.fmt.allocPrint(allocator, "{s}hc_attn_scale", .{idx_fmt});
        defer allocator.free(hc_attn_scale_name);
        const hc_attn_scale = weights.get(hc_attn_scale_name);
        if (weights.fetchRemove(hc_attn_scale_name)) |kv| allocator.free(kv.key);

        const hc_ffn_fn_name = try std.fmt.allocPrint(allocator, "{s}hc_ffn_fn", .{idx_fmt});
        defer allocator.free(hc_ffn_fn_name);
        const hc_ffn_fn = weights.get(hc_ffn_fn_name);
        if (weights.fetchRemove(hc_ffn_fn_name)) |kv| allocator.free(kv.key);

        const hc_ffn_base_name = try std.fmt.allocPrint(allocator, "{s}hc_ffn_base", .{idx_fmt});
        defer allocator.free(hc_ffn_base_name);
        const hc_ffn_base = weights.get(hc_ffn_base_name);
        if (weights.fetchRemove(hc_ffn_base_name)) |kv| allocator.free(kv.key);

        const hc_ffn_scale_name = try std.fmt.allocPrint(allocator, "{s}hc_ffn_scale", .{idx_fmt});
        defer allocator.free(hc_ffn_scale_name);
        const hc_ffn_scale = weights.get(hc_ffn_scale_name);
        if (weights.fetchRemove(hc_ffn_scale_name)) |kv| allocator.free(kv.key);

        // Use mHC if weights present, otherwise use dummy
        const dummy_arr = try array_mod.zeros(allocator, &[_]i32{1}, .float32);
        const hc_attn = deepseek_v4.DSV4HyperConn{
            .hc_fn = hc_attn_fn orelse dummy_arr,
            .hc_base = hc_attn_base orelse try array_mod.zeros(allocator, &[_]i32{1}, .float32),
            .hc_scale = hc_attn_scale orelse try array_mod.zeros(allocator, &[_]i32{1}, .float32),
            .hc_mult = config.hc_mult,
            .hc_sinkhorn_iters = config.hc_sinkhorn_iters,
            .hc_eps = config.hc_eps,
        };
        const hc_ffn = deepseek_v4.DSV4HyperConn{
            .hc_fn = hc_ffn_fn orelse try array_mod.zeros(allocator, &[_]i32{1}, .float32),
            .hc_base = hc_ffn_base orelse try array_mod.zeros(allocator, &[_]i32{1}, .float32),
            .hc_scale = hc_ffn_scale orelse try array_mod.zeros(allocator, &[_]i32{1}, .float32),
            .hc_mult = config.hc_mult,
            .hc_sinkhorn_iters = config.hc_sinkhorn_iters,
            .hc_eps = config.hc_eps,
        };

        layers[i] = deepseek_v4.DSV4TransformerBlock{
            .ctx = ctx,
            .config = config,
            .layer_idx = i,
            .attn_norm = attn_norm,
            .ffn_norm = ffn_norm,
            .attn = attention,
            .ffn = moe,
            .hc_attn = hc_attn,
            .hc_ffn = hc_ffn,
        };
    }

    // Consume top-level quantized metadata that buildDSV4Model doesn't directly use
    consumeWeightKey(allocator, weights, "embed.weight");
    consumeWeightKey(allocator, weights, "head.weight");
    consumeWeightKey(allocator, weights, "lm_head.weight");

    // Check for unmapped weights (skip quantization metadata suffixes)
    var remaining = weights.iterator();
    while (remaining.next()) |entry| {
        const k = entry.key_ptr.*;
        if (std.mem.endsWith(u8, k, ".scale")) continue;
        if (std.mem.endsWith(u8, k, ".scales")) continue;
        if (std.mem.endsWith(u8, k, ".biases")) continue;
        std.log.warn("Unused weight: {s}", .{k});
    }

    return deepseek_v4.DSV4Model{
        .allocator = allocator,
        .ctx = ctx,
        .config = config.*,
        .embed_tokens = embed,
        .layers = layers,
        .norm = norm,
        .hc_head = hc_head,
        .lm_head = lm_head,
    };
}

/// Create per-layer KV caches matching mlx-lm's Model.make_cache().
/// Layers with compress_ratio > 0 get DeepseekV4Cache (with compressor/indexer state).
/// Layers with compress_ratio == 0 get standard RotatingKVCache.
pub fn makeV4Caches(
    allocator: std.mem.Allocator,
    config: *const DSV4Config,
    stream: c.c.mlx_stream,
) ![]kvcache.KVCacheStrategy {
    const num_layers = config.num_hidden_layers;
    const caches = try allocator.alloc(kvcache.KVCacheStrategy, num_layers);
    errdefer allocator.free(caches);

    for (0..num_layers) |i| {
        const compress_ratio = if (i < config.compress_ratios.len) config.compress_ratios[i] else 0;
        const layer_config = kvcache.LayerConfig{
            .batch_size = 1,
            .num_heads = config.num_attention_heads,
            .num_kv_heads = 1,
            .head_dim = config.head_dim,
            .max_seq_len = @min(config.max_position_embeddings, 8192),
            .dtype = .float32,
        };

        if (compress_ratio > 0) {
            caches[i] = try kvcache.createDeepseekV4Cache(allocator, layer_config, config.sliding_window, stream);
        } else {
            caches[i] = try kvcache.createRotatingWithWindow(allocator, layer_config, config.sliding_window, stream);
        }
    }

    return caches;
}
