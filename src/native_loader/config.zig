/// MLX-free config.json parser for DeepSeek-V4-Flash.
/// Only std.json — no MLX dependency.
const std = @import("std");

pub const DSV4NativeConfig = struct {
    num_hidden_layers: u32 = 43,
    vocab_size: u32 = 129280,
    hidden_size: u32 = 4096,
    intermediate_size: u32 = 2048,
    n_routed_experts: u32 = 256,
    num_experts_per_tok: u32 = 6,
    n_shared_experts: u32 = 1,
    use_mhc: bool = true,
    hc_mult: u32 = 4,
    // MLA
    num_attention_heads: u32 = 64,
    head_dim: u32 = 512,
    qk_rope_head_dim: u32 = 64,
    q_lora_rank: u32 = 1024,
    kv_lora_rank: u32 = 512,
    // mHC head compression
    hc_compress_mult: ?u32 = null,
    // Quantization group sizes
    expert_group_size: u32 = 32,
    attn_group_size: u32 = 64,
    // Routing
    n_hash_layers: u32 = 3,
    route_scale: f32 = 1.5,
    // Norms
    rms_norm_eps: f32 = 1e-6,
    hc_eps: f32 = 1e-6,
    // Compressor/Indexer (T2C.1 新增)
    compress_ratios: [43]u32 = [_]u32{0} ** 43, // per-layer compress ratio
    sliding_window: u32 = 128,
    index_topk: u32 = 512,
    index_n_heads: u32 = 64,
    index_head_dim: u32 = 128,
};

pub fn parse(allocator: std.mem.Allocator, model_dir: []const u8) !DSV4NativeConfig {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);

    const path_z = try allocator.dupeZ(u8, config_path);
    defer allocator.free(path_z);

    const fd = std.c.open(path_z.ptr, .{});
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);

    var stat: std.c.Stat = undefined;
    if (std.c.fstat(fd, &stat) != 0) return error.StatFailed;
    const file_size: usize = @intCast(stat.size);
    if (file_size > 4 * 1024 * 1024) return error.ConfigTooLarge;

    const content = try allocator.alloc(u8, file_size);
    errdefer allocator.free(content);
    var total: usize = 0;
    while (total < file_size) {
        const n = std.c.pread(fd, content.ptr + total, file_size - total, @intCast(total));
        if (n <= 0) return error.ReadFailed;
        total += @intCast(n);
    }
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();

    const obj = parsed.value.object;
    var cfg = DSV4NativeConfig{};

    if (getInt(obj, "num_hidden_layers")) |v| cfg.num_hidden_layers = @intCast(v);
    if (getInt(obj, "vocab_size")) |v| cfg.vocab_size = @intCast(v);
    if (getInt(obj, "hidden_size")) |v| cfg.hidden_size = @intCast(v);
    if (getInt(obj, "intermediate_size")) |v| cfg.intermediate_size = @intCast(v);
    if (getInt(obj, "n_routed_experts")) |v| cfg.n_routed_experts = @intCast(v);
    if (getInt(obj, "num_experts_per_tok")) |v| cfg.num_experts_per_tok = @intCast(v);
    if (getInt(obj, "n_shared_experts")) |v| cfg.n_shared_experts = @intCast(v);
    if (getInt(obj, "num_attention_heads")) |v| cfg.num_attention_heads = @intCast(v);
    if (getInt(obj, "head_dim")) |v| cfg.head_dim = @intCast(v);
    if (getInt(obj, "qk_rope_head_dim")) |v| cfg.qk_rope_head_dim = @intCast(v);
    if (getInt(obj, "q_lora_rank")) |v| cfg.q_lora_rank = @intCast(v);
    if (getInt(obj, "kv_lora_rank")) |v| cfg.kv_lora_rank = @intCast(v);

    // use_mhc
    if (obj.get("use_mhc")) |v| {
        cfg.use_mhc = v == .bool and v.bool;
    }
    if (getInt(obj, "hc_mult")) |v| cfg.hc_mult = @intCast(v);

    // Norm eps
    if (getFloat(obj, "rms_norm_eps")) |v| cfg.rms_norm_eps = v;
    if (getFloat(obj, "hc_eps")) |v| cfg.hc_eps = v;

    // route_scale
    if (getFloat(obj, "route_scale")) |v| cfg.route_scale = v;

    // Compressor/Indexer fields (T2C.1)
    if (getInt(obj, "sliding_window")) |v| cfg.sliding_window = @intCast(v);
    if (getInt(obj, "index_topk")) |v| cfg.index_topk = @intCast(v);
    if (getInt(obj, "index_n_heads")) |v| cfg.index_n_heads = @intCast(v);
    if (getInt(obj, "index_head_dim")) |v| cfg.index_head_dim = @intCast(v);

    // compress_ratios: JSON array → [43]u32
    if (obj.get("compress_ratios")) |ratios_val| {
        if (ratios_val == .array) {
            const items = ratios_val.array.items;
            const n = @min(items.len, 43);
            for (0..n) |idx| {
                cfg.compress_ratios[idx] = switch (items[idx]) {
                    .integer => |v| @intCast(v),
                    else => 0,
                };
            }
        }
    }

    return cfg;
}

fn getInt(obj: std.json.ObjectMap, key: []const u8) ?i64 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .integer => |i| i,
        .float => |f| @intFromFloat(f),
        else => null,
    };
}

fn getFloat(obj: std.json.ObjectMap, key: []const u8) ?f32 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}
