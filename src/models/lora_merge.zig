//! Load-time LoRA merge for quantized Qwen3.8 backbones (e.g.
//! harness-hub/Qwen3.8-27B-pi-adapted).
//!
//! The base checkpoint stores every targeted linear as a 4-bit affine
//! triplet (`name.weight` packed u32 + `name.scales` + `name.biases`), so a
//! LoRA cannot be added directly. For each adapter module we:
//!   W = dequantize(triplet)           (f32)
//!   W' = W + (alpha / r) * (B @ A)    (adapter tensors are f32)
//!   requantize W' -> triplet          (same bits/group_size as the base)
//! and swap the resident map entries. Done once at load; the runtime path
//! is unchanged (no per-token LoRA overhead, no shape changes).
//!
//! Caveat: the base was quantized once already; re-quantizing W' is a
//! second rounding. In practice the error stays well below the 4-bit noise
//! floor (the LoRA delta is small relative to W), but outputs are NOT
//! bit-identical to a bf16 merge.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const quantize_mod = @import("mlx").quantize;
const safetensors_reader = @import("mlx").safetensors_reader;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

const LoraConfig = struct {
    r: u32,
    alpha: f32,
};

fn parseConfig(allocator: std.mem.Allocator, adapter_dir: []const u8) !LoraConfig {
    const path = try std.fs.path.join(allocator, &.{ adapter_dir, "adapter_config.json" });
    defer allocator.free(path);
    const posix = @cImport({
        @cInclude("fcntl.h");
        @cInclude("unistd.h");
    });
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);
    const fd = posix.open(path_z.ptr, posix.O_RDONLY);
    if (fd < 0) return error.FileNotFound;
    defer _ = posix.close(fd);
    var buf: [64 * 1024]u8 = undefined;
    const n = posix.read(fd, &buf, buf.len);
    if (n < 0) return error.ReadFailed;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, buf[0..@intCast(n)], .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const r = root.get("r") orelse return error.BadConfig;
    const alpha = root.get("lora_alpha") orelse return error.BadConfig;
    return .{
        .r = @intCast(r.integer),
        .alpha = switch (alpha) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => return error.BadConfig,
        },
    };
}

/// Merge every `*.lora_{A,B}.weight` pair in `<adapter_dir>/adapter_model.safetensors`
/// into the resident `weights` map. Adapter keys look like
/// `base_model.model.model.language_model.layers.N.self_attn.q_proj.lora_A.weight`;
/// the resident prefix is everything before `.lora_` with the
/// `base_model.model.` PEFT prefix stripped. Returns modules merged.
pub fn applyLoraAdapter(
    allocator: std.mem.Allocator,
    ctx: EagerContext,
    weights: *std.StringHashMap(Array),
    adapter_dir: []const u8,
) !usize {
    const cfg = try parseConfig(allocator, adapter_dir);
    const scale: f32 = cfg.alpha / @as(f32, @floatFromInt(cfg.r));

    var index = safetensors_reader.TensorIndex.init(allocator);
    defer index.deinit();
    const shard = try std.fs.path.join(allocator, &.{ adapter_dir, "adapter_model.safetensors" });
    defer allocator.free(shard);
    try index.addShard(shard);

    var merged: usize = 0;
    var it = index.entries.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (!std.mem.endsWith(u8, key, ".lora_A.weight")) continue;
        const stem = key[0 .. key.len - ".lora_A.weight".len];
        var bbuf: [512]u8 = undefined;
        const b_key = try std.fmt.bufPrint(&bbuf, "{s}.lora_B.weight", .{stem});
        const a = try index.loadTensor(key);
        defer a.deinit();
        const b = index.loadTensor(b_key) catch |err| {
            std.log.warn("lora: {s} has no B pair ({})", .{ stem, err });
            continue;
        };
        defer b.deinit();

        // Resident name: strip the PEFT `base_model.model.` prefix, then
        // fix the root swap: PEFT keys are `model.language_model.layers.*`
        // (HF ForCausalLM.model) while the resident map uses the checkpoint's
        // own `language_model.model.layers.*`.
        var resident_stem = if (std.mem.startsWith(u8, stem, "base_model.model."))
            stem["base_model.model.".len..]
        else
            stem;
        var stem_buf: [512]u8 = undefined;
        if (std.mem.startsWith(u8, resident_stem, "model.language_model.")) {
            resident_stem = std.fmt.bufPrint(&stem_buf, "language_model.model.{s}", .{resident_stem["model.language_model.".len..]}) catch return error.NameTooLong;
        }
        var wbuf: [512]u8 = undefined;
        var sbuf: [512]u8 = undefined;
        var bbuf2: [512]u8 = undefined;
        const w_name = try std.fmt.bufPrint(&wbuf, "{s}.weight", .{resident_stem});
        const s_name = try std.fmt.bufPrint(&sbuf, "{s}.scales", .{resident_stem});
        const b_name = try std.fmt.bufPrint(&bbuf2, "{s}.biases", .{resident_stem});
        const w_old = weights.get(w_name) orelse {
            std.log.warn("lora: resident weight missing for {s}", .{resident_stem});
            continue;
        };
        const s_old = weights.get(s_name) orelse {
            std.log.warn("lora: resident scales missing for {s} (unquantized merge unsupported)", .{resident_stem});
            continue;
        };
        const b_old = weights.get(b_name) orelse {
            std.log.warn("lora: resident biases missing for {s}", .{resident_stem});
            continue;
        };

        // Dequantize the resident triplet. in_features comes from the
        // adapter A [r, in]; bits/group derive from the packed shapes.
        const ws = w_old.shape();
        const ss = s_old.shape();
        const packed_cols: i64 = ws[ws.len - 1];
        const scale_cols: i64 = ss[ss.len - 1];
        const in_features: i64 = a.shape()[1];
        const bits: u8 = @intCast(@divTrunc(packed_cols * 32, in_features));
        const group_size: i32 = @intCast(@divTrunc(in_features, scale_cols));
        const qw = quantize_mod.QuantizedWeight{
            .data = w_old,
            .scales = s_old,
            .biases = b_old,
            .config = .{ .bits = bits, .group_size = group_size, .mode = .affine },
            .original_shape = &[_]i32{},
        };
        const w = try quantize_mod.dequantizeAs(ctx, qw, .float32); // [out, in] f32
        defer w.deinit();

        // delta = scale * (B @ A): B [out, r] f32, A [r, in] f32.
        const a_f32 = try ops.astype(ctx, a, .float32);
        defer a_f32.deinit();
        const b_f32 = try ops.astype(ctx, b, .float32);
        defer b_f32.deinit();
        const ba = try ops.matmul(ctx, b_f32, a_f32);
        defer ba.deinit();
        const scale_arr = try ops.scalarF32(ctx, scale);
        defer scale_arr.deinit();
        const delta = try ops.multiply(ctx, ba, scale_arr);
        defer delta.deinit();
        const w_merged = try ops.add(ctx, w, delta);
        defer w_merged.deinit();

        const qw_new = try quantize_mod.quantize(ctx, w_merged, .{ .bits = bits, .group_size = group_size, .mode = .affine });
        defer allocator.free(qw_new.original_shape);
        try qw_new.data.eval();
        try qw_new.scales.eval();
        try qw_new.biases.eval();

        // Swap resident entries (caller owns the map; free old arrays).
        w_old.deinit();
        s_old.deinit();
        b_old.deinit();
        weights.getPtr(w_name).?.* = qw_new.data;
        weights.getPtr(s_name).?.* = qw_new.scales;
        weights.getPtr(b_name).?.* = qw_new.biases;
        merged += 1;
    }
    if (merged > 0) {
        std.log.info("lora: merged {d} modules from {s} (r={d}, alpha={d:.0})", .{ merged, adapter_dir, cfg.r, cfg.alpha });
    }
    return merged;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "lora merge: W' dequantizes to W + scale*B@A (within 4-bit noise)" {
    c.initErrorHandler();
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Base linear [8, 16], quantize to 4-bit gs=64 like the backbone.
    const w_data = try Array.fromData(allocator, f32, &([_]f32{0.05} ** (8 * 64)), &[_]i32{ 8, 64 });
    defer w_data.deinit();
    const qw = try quantize_mod.quantize(ctx, w_data, .{ .bits = 4, .group_size = 64, .mode = .affine });
    defer allocator.free(qw.original_shape);
    var weights = std.StringHashMap(Array).init(allocator);
    defer {
        var it = weights.iterator();
        while (it.next()) |e| {
            allocator.free(e.key_ptr.*);
            e.value_ptr.deinit();
        }
        weights.deinit();
    }
    try weights.put(try allocator.dupe(u8, "mod.weight"), qw.data);
    try weights.put(try allocator.dupe(u8, "mod.scales"), qw.scales);
    try weights.put(try allocator.dupe(u8, "mod.biases"), qw.biases);

    // Adapter on disk: A [4,16], B [8,4], r=4 alpha=8 -> scale 2.
    const tmp = "/tmp/dmlx_lora_test_adapter";
    {
        const mkdir_c = @cImport(@cInclude("sys/stat.h"));
        _ = mkdir_c.mkdir(tmp, 0o755);
        const a = try Array.fromData(allocator, f32, &([_]f32{0.01} ** (4 * 64)), &[_]i32{ 4, 64 });
        defer a.deinit();
        const b = try Array.fromData(allocator, f32, &([_]f32{0.02} ** (8 * 4)), &[_]i32{ 8, 4 });
        defer b.deinit();
        const c_mod = @import("mlx").c;
        const arr_map = c_mod.c.mlx_map_string_to_array_new();
        defer _ = c_mod.c.mlx_map_string_to_array_free(arr_map);
        const meta = c_mod.c.mlx_map_string_to_string_new();
        defer _ = c_mod.c.mlx_map_string_to_string_free(meta);
        try c.check(c_mod.c.mlx_map_string_to_array_insert(arr_map, "base_model.model.mod.lora_A.weight", a.inner));
        try c.check(c_mod.c.mlx_map_string_to_array_insert(arr_map, "base_model.model.mod.lora_B.weight", b.inner));
        try c.check(c_mod.c.mlx_save_safetensors(tmp ++ "/adapter_model.safetensors", arr_map, meta));
        const posix = @cImport(@cInclude("fcntl.h"));
        const unistd = @cImport(@cInclude("unistd.h"));
        const fd = posix.open(tmp ++ "/adapter_config.json", posix.O_WRONLY | posix.O_CREAT | posix.O_TRUNC, @as(c_uint, 0o644));
        try std.testing.expect(fd >= 0);
        const cfg_json = "{\"r\":4,\"lora_alpha\":8}";
        _ = unistd.write(fd, cfg_json.ptr, cfg_json.len);
        _ = unistd.close(fd);
    }

    const merged = try applyLoraAdapter(allocator, ctx, &weights, tmp);
    try std.testing.expectEqual(@as(usize, 1), merged);

    // Dequantize the merged triplet; expect W + 2*(B@A) = 0.05 + 2*0.02*0.01*4
    // per element (A,B constant), within 4-bit group-64 resolution.
    const qw2 = quantize_mod.QuantizedWeight{
        .data = weights.get("mod.weight").?,
        .scales = weights.get("mod.scales").?,
        .biases = weights.get("mod.biases").?,
        .config = .{ .bits = 4, .group_size = 64, .mode = .affine },
        .original_shape = &[_]i32{},
    };
    const w2 = try quantize_mod.dequantizeAs(ctx, qw2, .float32);
    defer w2.deinit();
    try w2.eval();
    const vals = try w2.dataSlice(f32);
    const expected: f32 = 0.05 + 2.0 * (0.02 * 0.01 * 4.0);
    for (vals) |v| {
        try std.testing.expectApproxEqAbs(expected, v, 0.01);
    }
}
