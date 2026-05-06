/// HuggingFace Config parser for model architectures.
///
/// Reads `config.json` from a HF model directory and maps to
/// our internal LlamaConfig / model configurations.
const std = @import("std");
const models = @import("models/llama.zig");

pub const LlamaConfig = models.LlamaConfig;

/// Parse a HF config.json into LlamaConfig.
/// Supports LLaMA, LLaMA-2, Mistral, Qwen, Gemma architectures.
pub fn parseLlamaConfig(allocator: std.mem.Allocator, json_text: []const u8) !LlamaConfig {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    const root = parsed.value;
    const obj = root.object;

    // Architecture detection
    const archs = obj.get("architectures") orelse return error.MissingArchitecture;
    const arch = archs.array.items[0].string;

    const is_qwen = std.mem.startsWith(u8, arch, "Qwen2") or std.mem.startsWith(u8, arch, "Qwen3");
    const is_mistral = std.mem.startsWith(u8, arch, "Mistral");
    const is_llama = std.mem.startsWith(u8, arch, "Llama");
    const is_gemma = std.mem.startsWith(u8, arch, "Gemma");
    const is_glm4 = std.mem.startsWith(u8, arch, "Glm4");
    const is_phi = std.mem.startsWith(u8, arch, "Phi");

    if (!is_llama and !is_mistral and !is_qwen and !is_gemma and !is_glm4 and !is_phi) {
        return error.UnsupportedArchitecture;
    }

    // Extract common fields
    const vocab_size: usize = @intCast(getInt(obj, "vocab_size") orelse 32000);
    const hidden_size: usize = @intCast(getInt(obj, "hidden_size") orelse 4096);
    const num_hidden_layers: usize = @intCast(getInt(obj, "num_hidden_layers") orelse 32);
    const num_attention_heads: usize = @intCast(getInt(obj, "num_attention_heads") orelse 32);
    const intermediate_size: usize = @intCast(getInt(obj, "intermediate_size") orelse 11008);
    const rms_norm_eps: f32 = @floatCast(getFloat(obj, "rms_norm_eps") orelse 1e-6);

    // GQA support
    const num_key_value_heads: usize = blk: {
        if (getInt(obj, "num_key_value_heads")) |n| {
            break :blk @intCast(n);
        }
        break :blk num_attention_heads; // Default: MHA
    };

    // RoPE theta
    const rope_theta: f32 = blk: {
        if (getFloat(obj, "rope_theta")) |t| {
            break :blk @floatCast(t);
        }
        // Mistral uses 10000.0 by default
        // LLaMA-2 uses 10000.0
        // CodeLlama uses 1000000.0
        break :blk 10000.0;
    };

    const max_position_embeddings: usize = @intCast(getInt(obj, "max_position_embeddings") orelse 4096);

    // Quantization config (mlx-lm format: "quantization": {"bits": 4, "group_size": 64})
    var quantize_bits: u8 = 0;
    var quantize_group_size: i32 = 64;
    if (obj.get("quantization")) |quant_val| {
        if (quant_val == .object) {
            const quant_obj = quant_val.object;
            if (getInt(quant_obj, "bits")) |b| {
                quantize_bits = @intCast(b);
            }
            if (getInt(quant_obj, "group_size")) |g| {
                quantize_group_size = @intCast(g);
            }
        }
    }

    // Qwen2 specific adjustments
    const final_vocab_size = if (is_qwen and obj.contains("vocab_size")) blk: {
        // Qwen2 sometimes uses a larger vocab with special tokens
        break :blk vocab_size;
    } else vocab_size;

    // EOS token ID (can be integer or array of integers)
    const eos_token_id: ?u32 = blk: {
        const eos_val = obj.get("eos_token_id") orelse break :blk null;
        switch (eos_val) {
            .integer => |n| break :blk @intCast(n),
            .array => |arr| {
                if (arr.items.len > 0) {
                    switch (arr.items[0]) {
                        .integer => |n| break :blk @intCast(n),
                        else => break :blk null,
                    }
                }
                break :blk null;
            },
            else => break :blk null,
        }
    };

    // Explicit head_dim (Qwen3 uses head_dim=128 with hidden_size=1024, num_heads=16)
    const explicit_head_dim: ?usize = if (getInt(obj, "head_dim")) |hd| @intCast(hd) else null;

    // RoPE scaling (Phi-4, LLaMA-3.1)
    var rope_scaling_type: ?[]const u8 = null;
    var rope_scaling_factor: f32 = 1.0;
    var rope_scaling_original_max_pos: ?usize = null;
    if (obj.get("rope_scaling")) |rs| {
        if (rs == .object) {
            const rs_obj = rs.object;
            if (rs_obj.get("type")) |t| {
                if (t == .string) rope_scaling_type = try allocator.dupe(u8, t.string);
            }
            if (getFloat(rs_obj, "factor")) |f| rope_scaling_factor = @floatCast(f);
            if (getInt(rs_obj, "original_max_position_embeddings")) |omp| rope_scaling_original_max_pos = @intCast(omp);
        }
    }

    return LlamaConfig{
        .vocab_size = final_vocab_size,
        .hidden_size = hidden_size,
        .num_hidden_layers = num_hidden_layers,
        .num_attention_heads = num_attention_heads,
        .num_key_value_heads = num_key_value_heads,
        .intermediate_size = intermediate_size,
        .rms_norm_eps = rms_norm_eps,
        .rope_theta = rope_theta,
        .max_position_embeddings = max_position_embeddings,
        .head_dim = explicit_head_dim,
        .quantize_bits = quantize_bits,
        .quantize_group_size = quantize_group_size,
        .eos_token_id = eos_token_id,
        .rope_scaling_type = rope_scaling_type,
        .rope_scaling_factor = rope_scaling_factor,
        .rope_scaling_original_max_position_embeddings = rope_scaling_original_max_pos,
    };
}

/// Load config from a file path.
pub fn loadLlamaConfig(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !LlamaConfig {
    const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024));
    defer allocator.free(content);

    return parseLlamaConfig(allocator, content);
}

/// Helper: get integer from JSON object.
fn getInt(obj: std.json.ObjectMap, key: []const u8) ?i64 {
    const val = obj.get(key) orelse return null;
    switch (val) {
        .integer => |n| return n,
        .float => |f| return @intFromFloat(f),
        else => return null,
    }
}

/// Helper: get float from JSON object.
fn getFloat(obj: std.json.ObjectMap, key: []const u8) ?f64 {
    const val = obj.get(key) orelse return null;
    switch (val) {
        .float => |f| return f,
        .integer => |n| return @floatFromInt(n),
        else => return null,
    }
}

/// Weight name mapping from HF Safetensors to our internal naming.
/// HF:   "model.layers.0.self_attn.q_proj.weight"
/// Ours: "layers.0.attention.wq"
pub const WeightMapping = struct {
    hf_name: []const u8,
    internal_name: []const u8,
};

/// Standard LLaMA weight mapping.
pub const llama_weight_map = &[_]WeightMapping{
    .{ .hf_name = "model.embed_tokens.weight", .internal_name = "embed_tokens.weight" },
    .{ .hf_name = "model.norm.weight", .internal_name = "norm.weight" },
    .{ .hf_name = "lm_head.weight", .internal_name = "lm_head.weight" },
    // Layer-specific patterns (processed with index substitution)
};

/// Map a HF weight name to our internal name.
/// Pattern: "model.layers.{i}.self_attn.q_proj.weight" -> "layers.{i}.attention.wq"
pub fn mapWeightName(hf_name: []const u8, allocator: std.mem.Allocator) !?[]const u8 {
    // embed_tokens (including quantized scales/biases)
    if (std.mem.eql(u8, hf_name, "model.embed_tokens.weight")) {
        return try allocator.dupe(u8, "embed_tokens.weight");
    }
    if (std.mem.eql(u8, hf_name, "model.embed_tokens.scales")) {
        return try allocator.dupe(u8, "embed_tokens.weight.scales");
    }
    if (std.mem.eql(u8, hf_name, "model.embed_tokens.biases")) {
        return try allocator.dupe(u8, "embed_tokens.weight.biases");
    }
    // norm
    if (std.mem.eql(u8, hf_name, "model.norm.weight")) {
        return try allocator.dupe(u8, "norm.weight");
    }
    // lm_head (including quantized scales/biases)
    if (std.mem.eql(u8, hf_name, "lm_head.weight")) {
        return try allocator.dupe(u8, "lm_head.weight");
    }
    if (std.mem.eql(u8, hf_name, "lm_head.scales")) {
        return try allocator.dupe(u8, "lm_head.weight.scales");
    }
    if (std.mem.eql(u8, hf_name, "lm_head.biases")) {
        return try allocator.dupe(u8, "lm_head.weight.biases");
    }

    // Layer pattern: model.layers.{i}.{component}
    const prefix = "model.layers.";
    if (std.mem.startsWith(u8, hf_name, prefix)) {
        // Find the layer index
        const after_prefix = hf_name[prefix.len..];
        const dot_idx = std.mem.indexOf(u8, after_prefix, ".") orelse return null;
        const layer_idx = after_prefix[0..dot_idx];
        const component = after_prefix[dot_idx + 1 ..];

        // Map component
        const mapped = mapComponent(component) orelse return null;
        return try std.fmt.allocPrint(allocator, "layers.{s}.{s}", .{ layer_idx, mapped });
    }

    return null;
}

fn mapComponent(component: []const u8) ?[]const u8 {
    const map = std.StaticStringMap([]const u8).initComptime(.{
        .{ "self_attn.q_proj.weight", "attention.wq" },
        .{ "self_attn.k_proj.weight", "attention.wk" },
        .{ "self_attn.v_proj.weight", "attention.wv" },
        .{ "self_attn.o_proj.weight", "attention.wo" },
        // Attention bias terms (Qwen2 uses attention bias)
        .{ "self_attn.q_proj.bias", "attention.wq_bias" },
        .{ "self_attn.k_proj.bias", "attention.wk_bias" },
        .{ "self_attn.v_proj.bias", "attention.wv_bias" },
        .{ "self_attn.o_proj.bias", "attention.wo_bias" },
        .{ "self_attn.q_norm.weight", "attention.q_norm" },
        .{ "self_attn.k_norm.weight", "attention.k_norm" },
        .{ "mlp.gate_proj.weight", "mlp.gate_proj" },
        .{ "mlp.up_proj.weight", "mlp.up_proj" },
        .{ "mlp.down_proj.weight", "mlp.down_proj" },
        .{ "input_layernorm.weight", "input_layernorm" },
        .{ "post_attention_layernorm.weight", "post_attention_layernorm" },
        // Quantized weight suffixes (mlx-lm format)
        .{ "self_attn.q_proj.scales", "attention.wq.scales" },
        .{ "self_attn.q_proj.biases", "attention.wq.biases" },
        .{ "self_attn.k_proj.scales", "attention.wk.scales" },
        .{ "self_attn.k_proj.biases", "attention.wk.biases" },
        .{ "self_attn.v_proj.scales", "attention.wv.scales" },
        .{ "self_attn.v_proj.biases", "attention.wv.biases" },
        .{ "self_attn.o_proj.scales", "attention.wo.scales" },
        .{ "self_attn.o_proj.biases", "attention.wo.biases" },
        .{ "mlp.gate_proj.scales", "mlp.gate_proj.scales" },
        .{ "mlp.gate_proj.biases", "mlp.gate_proj.biases" },
        .{ "mlp.up_proj.scales", "mlp.up_proj.scales" },
        .{ "mlp.up_proj.biases", "mlp.up_proj.biases" },
        .{ "mlp.down_proj.scales", "mlp.down_proj.scales" },
        .{ "mlp.down_proj.biases", "mlp.down_proj.biases" },
    });
    return map.get(component);
}
