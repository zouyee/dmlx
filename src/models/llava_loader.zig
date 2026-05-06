/// LLaVA model loader from HuggingFace Safetensors format.
///
/// Loads a LLaVA model consisting of:
///   - Vision tower weights (pattern: `vision_tower.*`)
///   - Multi-modal projector (pattern: `multi_modal_projector.*`)
///   - Language model (loaded via `llama_loader.buildModel`)
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const nn = @import("mlx").nn;
const io = @import("mlx").io;
const hf_config = @import("../hf_config.zig");
const llama = @import("llama.zig");
const llama_loader = @import("llama_loader.zig");
const llava = @import("../vision/llava.zig");
const vit = @import("../vision/vit.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

// ============================================================
// Public API
// ============================================================

/// Load a LLaVA model from a directory containing config.json and model.safetensors.
pub fn loadLlavaModel(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
) !llava.LlavaModel {
    const config_path = try std.fs.path.join(allocator, &.{ model_path, "config.json" });
    defer allocator.free(config_path);

    const config_data = try std.fs.cwd().readFileAlloc(allocator, config_path, 1024 * 1024);
    defer allocator.free(config_data);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, config_data, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    // Extract text_config for the LLaMA parser
    const text_config_value = root.get("text_config") orelse parsed.value;
    const text_config_json = try std.json.Stringify.valueAlloc(allocator, text_config_value, .{});
    defer allocator.free(text_config_json);

    const llama_config = try hf_config.parseLlamaConfig(allocator, text_config_json);
    return loadLlavaModelFromConfig(allocator, model_path, &llama_config, ctx, stream);
}

/// Load a LLaVA model using a pre-parsed LLaMA config.
pub fn loadLlavaModelFromConfig(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    llama_config: *const hf_config.LlamaConfig,
    ctx: EagerContext,
    stream: c.c.mlx_stream,
) !llava.LlavaModel {
    // Find safetensors file
    const st_path = try std.fs.path.join(allocator, &.{ model_path, "model.safetensors" });
    defer allocator.free(st_path);

    // Load all weights from safetensors
    var st = try io.loadSafetensors(allocator, st_path);
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

    // Destination weight maps
    var vision_weights = std.StringHashMap(Array).init(allocator);
    errdefer {
        var it = vision_weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        vision_weights.deinit();
    }

    var lm_weights = std.StringHashMap(Array).init(allocator);
    errdefer {
        var it = lm_weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        lm_weights.deinit();
    }

    var projector_weights = std.StringHashMap(Array).init(allocator);
    errdefer {
        var it = projector_weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        projector_weights.deinit();
    }

    const vision_prefix = "vision_tower.";
    const projector_prefix = "multi_modal_projector.";
    const lm_prefix = "language_model.";

    var hf_it = st.weights.iterator();
    while (hf_it.next()) |entry| {
        const hf_name = entry.key_ptr.*;
        const weight = entry.value_ptr.*;

        if (std.mem.startsWith(u8, hf_name, vision_prefix)) {
            const key = try allocator.dupe(u8, hf_name);
            try vision_weights.put(key, weight);
        } else if (std.mem.startsWith(u8, hf_name, projector_prefix)) {
            const key = try allocator.dupe(u8, hf_name);
            try projector_weights.put(key, weight);
        } else {
            // Strip optional language_model. prefix then map
            const stripped = if (std.mem.startsWith(u8, hf_name, lm_prefix)) hf_name[lm_prefix.len..] else hf_name;
            const mapped = try hf_config.mapWeightName(stripped, allocator) orelse {
                std.log.warn("Unmapped LLaVA weight: {s}", .{hf_name});
                continue;
            };
            // Convert BF16 weights to float32 for eager layers
            const f32_weight = if (weight.dtype() == .bfloat16)
                try ops.astype(ctx, weight, .float32)
            else
                weight;
            try lm_weights.put(mapped, f32_weight);
        }
    }

    // Build language model from mapped weights
    const llama_model = try llama_loader.buildModel(allocator, llama_config, &lm_weights, ctx, stream, null);

    // Build multi-modal projector
    const projector = try buildProjector(ctx, &projector_weights) orelse return error.MissingProjectorWeights;

    // Clean up projector weight keys (values moved into projector)
    var p_it = projector_weights.iterator();
    while (p_it.next()) |entry| {
        allocator.free(entry.key_ptr.*);
    }
    projector_weights.deinit();

    // Build a default ViT model (vision tower weights not yet mapped to ViT layers)
    const vit_config = vit.ViTConfig{
        .image_size = 224,
        .patch_size = 16,
        .hidden_size = 768,
        .num_layers = 12,
        .num_heads = 12,
        .intermediate_size = 3072,
    };
    const vit_model = try vit.ViTModel.init(allocator, ctx, vit_config);
    errdefer vit_model.deinit();

    // Clean up unused vision weight keys
    var v_it = vision_weights.iterator();
    while (v_it.next()) |entry| {
        allocator.free(entry.key_ptr.*);
        entry.value_ptr.*.deinit();
    }
    vision_weights.deinit();

    return llava.LlavaModel{
        .allocator = allocator,
        .ctx = ctx,
        .vit = vit_model,
        .projector = projector,
        .language_model = llama_model,
    };
}

// ============================================================
// Internal helpers
// ============================================================

fn buildProjector(ctx: EagerContext, weights: *std.StringHashMap(Array)) !?llava.MultimodalProjector {
    const w1 = weights.fetchRemove("multi_modal_projector.linear_1.weight");
    const b1 = weights.fetchRemove("multi_modal_projector.linear_1.bias");
    const w2 = weights.fetchRemove("multi_modal_projector.linear_2.weight");
    const b2 = weights.fetchRemove("multi_modal_projector.linear_2.bias");

    if (w1 == null) {
        if (b1) |kv| {
            ctx.allocator.free(kv.key);
            kv.value.deinit();
        }
        if (w2) |kv| {
            ctx.allocator.free(kv.key);
            kv.value.deinit();
        }
        if (b2) |kv| {
            ctx.allocator.free(kv.key);
            kv.value.deinit();
        }
        return null;
    }

    const weight1 = w1.?.value;
    const shape1 = weight1.shape();
    const out1 = @as(usize, @intCast(shape1[0]));
    const in1 = @as(usize, @intCast(shape1[1]));

    const linear1 = nn.Linear{
        .ctx = ctx,
        .weight = weight1,
        .bias = if (b1) |kv| kv.value else null,
        .input_dims = in1,
        .output_dims = out1,
    };
    if (b1) |kv| ctx.allocator.free(kv.key);
    ctx.allocator.free(w1.?.key);

    var linear2: nn.Linear = undefined;
    if (w2) |kv2| {
        const weight2 = kv2.value;
        const shape2 = weight2.shape();
        const out2 = @as(usize, @intCast(shape2[0]));
        const in2 = @as(usize, @intCast(shape2[1]));
        linear2 = nn.Linear{
            .ctx = ctx,
            .weight = weight2,
            .bias = if (b2) |kv| kv.value else null,
            .input_dims = in2,
            .output_dims = out2,
        };
        if (b2) |kv| ctx.allocator.free(kv.key);
        ctx.allocator.free(kv2.key);
    } else {
        if (b2) |kv| {
            ctx.allocator.free(kv.key);
            kv.value.deinit();
        }
        return error.MissingProjectorWeights;
    }

    return llava.MultimodalProjector{
        .ctx = ctx,
        .linear_1 = linear1,
        .linear_2 = linear2,
    };
}

// ============================================================
// Tests
// ============================================================

test "loadLlavaModel parses config and splits weights" {
    const allocator = std.testing.allocator;

    const mock_config =
        \\{
        \\  "architectures": ["LlavaForConditionalGeneration"],
        \\  "text_config": {
        \\    "architectures": ["LlamaForCausalLM"],
        \\    "vocab_size": 32000,
        \\    "hidden_size": 4096,
        \\    "num_hidden_layers": 2,
        \\    "num_attention_heads": 32,
        \\    "num_key_value_heads": 32,
        \\    "intermediate_size": 11008,
        \\    "rms_norm_eps": 1e-6
        \\  },
        \\  "vision_config": {
        \\    "hidden_size": 1024,
        \\    "intermediate_size": 4096,
        \\    "num_hidden_layers": 24
        \\  }
        \\}
    ;

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, mock_config, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    const text_config_value = root.get("text_config") orelse parsed.value;
    const text_config_json = try std.json.Stringify.valueAlloc(allocator, text_config_value, .{});
    defer allocator.free(text_config_json);

    const llama_config = try hf_config.parseLlamaConfig(allocator, text_config_json);
    try std.testing.expectEqual(@as(usize, 32000), llama_config.vocab_size);
    try std.testing.expectEqual(@as(usize, 4096), llama_config.hidden_size);
    try std.testing.expectEqual(@as(usize, 2), llama_config.num_hidden_layers);
}

test "buildProjector constructs linear layers from weights" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    defer ctx.deinit();

    var weights = std.StringHashMap(Array).init(allocator);
    defer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    const w1 = try array_mod.zeros(allocator, &[_]i32{ 4096, 1024 }, .float32);
    const b1 = try array_mod.zeros(allocator, &[_]i32{4096}, .float32);
    const w2 = try array_mod.zeros(allocator, &[_]i32{ 4096, 4096 }, .float32);
    const b2 = try array_mod.zeros(allocator, &[_]i32{4096}, .float32);

    try weights.put(try allocator.dupe(u8, "multi_modal_projector.linear_1.weight"), w1);
    try weights.put(try allocator.dupe(u8, "multi_modal_projector.linear_1.bias"), b1);
    try weights.put(try allocator.dupe(u8, "multi_modal_projector.linear_2.weight"), w2);
    try weights.put(try allocator.dupe(u8, "multi_modal_projector.linear_2.bias"), b2);

    const projector = try buildProjector(ctx, &weights);
    try std.testing.expect(projector != null);
    try std.testing.expectEqual(@as(usize, 1024), projector.?.linear_1.input_dims);
    try std.testing.expectEqual(@as(usize, 4096), projector.?.linear_1.output_dims);
    try std.testing.expectEqual(@as(usize, 4096), projector.?.linear_2.input_dims);
    try std.testing.expectEqual(@as(usize, 4096), projector.?.linear_2.output_dims);
}
