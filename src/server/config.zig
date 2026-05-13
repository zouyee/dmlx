/// Server configuration types.
const std = @import("std");
const memory_mod = @import("../memory.zig");

pub const KvStrategy = enum {
    standard,
    paged,
    quantized,
    paged_quantized,
};

pub const KvTier = enum {
    ram,
    ssd,
};

/// KV cache quantization algorithm.
pub const KvQuant = enum {
    /// Standard affine quantization (per-group scale+bias via mlx_quantize).
    simple,
    /// TurboQuant: near-optimal online quantization with Lloyd-Max codebooks.
    /// Provides unbiased inner product estimation when QJL is enabled.
    turbo,
};

pub const ServerConfig = struct {
    model_path: []const u8,
    port: u16 = 8080,
    max_tokens: usize = 256,
    temperature: f32 = 0.8,
    top_k: usize = 50,
    top_p: f32 = 1.0,
    memory_config: memory_mod.MemoryConfig = .{},
    max_kv_size: memory_mod.MaxKvSize = .auto,
    kv_bits: u8 = 4,
    kv_strategy: KvStrategy = .paged_quantized,
    kv_quant: KvQuant = .simple,
    prompt_cache_file: ?[]const u8 = null,
    speculative_ngram: ?usize = null,
    kv_tier: KvTier = .ram,
    kv_cold_dir: ?[]const u8 = null,
    allow_unsafe_tools: bool = false,
    smelt: bool = false,
    smelt_experts: f32 = 1.0,
    smelt_strategy: []const u8 = "preload",
    smelt_cache_mb: usize = 2048,
};
