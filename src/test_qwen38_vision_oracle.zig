/// Qwen3.8 VISION oracle parity harness (Zig side).
///
/// Reads artifacts/qwen38_vision_oracle/*.npy + meta.json produced by
/// scripts/qwen38_vision_oracle_dump.py (mlx-vlm on the real checkpoint) and
/// replays the full vision path against the same weights:
///
///   1. preprocess tests/fixtures/vision/photo.png -> pixels/grid_thw
///      (diagnostic print; PNG so stb/PIL decode identically — JPEG IDCT
///      differences dominate the metric otherwise)
///   2. Qwen38Vit tower forward -> tower features, gate cos >= 0.999
///   3. computePositionIds -> MRoPE position_ids, gate EXACT
///   4. buildMropeCosSin -> per-slot cos/sin tables, gate max_abs < 1e-4
///   5. embed gather + mergeVisionFeatures -> merged embeddings, gate: text
///      positions exactly equal, image rows cos >= 0.99 (they echo the tower
///      output already covered by gate 2)
///   6. Qwen38ServingModel.forwardTokensWithVision -> last-position logits,
///      gate cos >= 0.995 (see noise-floor note below)
///
/// Noise floor (measured, reproducible via
/// scripts/compare_qwen38_vision_oracle.py artifacts/qwen38_vision_oracle{,_naive}):
/// the reference tower bf16-vs-f32 differs from ITSELF at cos 0.99934
/// (per-row min 0.9965), and python fused-vs-naive GDN differs at
/// logits_last cos 0.99940 / layer-63 hidden cos 0.98748 — with the
/// reference bf16 stack, an independent reimplementation
/// cannot exceed ~0.9995 on tower features and ~0.997 on last-position
/// logits for this 207-token multimodal prefill. The tower/merged/position/
/// cos-sin gates hold at their exact/0.999 targets; the logits gate is set
/// to 0.995 with the oracle-features diagnostic printed for context.
///
/// Standalone executable (not a zig test): the 27B model load perturbs
/// later GPU unit tests when they share a process. Exits 0 with a SKIP line
/// when the artifact dir or the local checkpoint is absent.
///
/// Usage: zig build test-qwen38-vision-oracle
/// Env:   DMLX_VISION_ORACLE_DIR  override the artifact directory
///        DMLX_VISION_BISECT=1    per-layer hidden bisection (depth-truncated
///                                reruns against lm_hidden_layer*.npy)
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;
const safetensors_reader = @import("mlx").safetensors_reader;
const qwen38_loader = @import("models/qwen38_loader.zig");
const qwen38_mrope = @import("models/qwen38_mrope.zig");
const qwen38 = @import("models/qwen38.zig");
const vit_mod = @import("vision/qwen38_vit.zig");
const preprocess = @import("vision/qwen38_preprocess.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

const ARTIFACTS_DIR = "artifacts/qwen38_vision_oracle";
const MODEL_DIR = "/Users/zouyee/models/Qwen3.8-27B-4bit";
const FIXTURE = "tests/fixtures/vision/photo.png";

const NpyArray = struct {
    shape: []i64,
    data: []align(16) u8, // raw payload, little-endian

    fn deinit(self: *NpyArray, allocator: std.mem.Allocator) void {
        allocator.free(self.shape);
        allocator.free(self.data);
    }

    fn f32s(self: *const NpyArray) []const f32 {
        return @alignCast(std.mem.bytesAsSlice(f32, self.data));
    }

    fn i64s(self: *const NpyArray) []const i64 {
        return @alignCast(std.mem.bytesAsSlice(i64, self.data));
    }
};

/// Minimal .npy v1/v2 reader: '<f4' and '<i8', C-order only.
fn readNpy(allocator: std.mem.Allocator, io: std.Io, path: []const u8, expect_descr: []const u8) !NpyArray {
    const raw = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 31)) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(raw);
    if (raw.len < 10 or !std.mem.eql(u8, raw[0..6], "\x93NUMPY")) return error.BadNpy;
    const major = raw[6];
    var hlen: usize = 0;
    var off: usize = 0;
    if (major == 1) {
        hlen = std.mem.readInt(u16, raw[8..10], .little);
        off = 10;
    } else {
        if (raw.len < 12) return error.BadNpy;
        hlen = std.mem.readInt(u32, raw[8..12], .little);
        off = 12;
    }
    if (off + hlen > raw.len) return error.BadNpy;
    const header = raw[off .. off + hlen];
    const dstart = (std.mem.indexOf(u8, header, "'descr': '") orelse return error.BadNpy) + "'descr': '".len;
    if (dstart + 3 > header.len) return error.BadNpy;
    if (!std.mem.eql(u8, header[dstart .. dstart + 3], expect_descr)) {
        std.log.err("{s}: descr {s} != {s}", .{ path, header[dstart .. dstart + 3], expect_descr });
        return error.BadNpy;
    }
    if (std.mem.indexOf(u8, header, "'fortran_order': True") != null) return error.BadNpy;
    const sstart = (std.mem.indexOf(u8, header, "'shape': (") orelse return error.BadNpy) + "'shape': (".len;
    const send = std.mem.indexOfScalarPos(u8, header, sstart, ')').?;
    var shape = std.ArrayList(i64).empty;
    errdefer shape.deinit(allocator);
    var it = std.mem.splitScalar(u8, header[sstart..send], ',');
    while (it.next()) |part| {
        const t = std.mem.trim(u8, part, " ");
        if (t.len == 0) continue;
        try shape.append(allocator, try std.fmt.parseInt(i64, t, 10));
    }
    const payload = raw[off + hlen ..];
    const data = try allocator.alignedAlloc(u8, .@"16", payload.len);
    errdefer allocator.free(data);
    @memcpy(data, payload);
    return .{ .shape = try shape.toOwnedSlice(allocator), .data = data };
}

fn cosSim(a: []const f32, b: []const f32) f64 {
    std.debug.assert(a.len == b.len);
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * y;
        na += @as(f64, x) * x;
        nb += @as(f64, y) * y;
    }
    return dot / (@sqrt(na) * @sqrt(nb));
}

fn maxAbsErr(a: []const f32, b: []const f32) f32 {
    var m: f32 = 0;
    for (a, b) |x, y| m = @max(m, @abs(x - y));
    return m;
}

fn arrayF32(ctx: EagerContext, arr: Array) !Array {
    if (arr.dtype() == .float32) return ops.copy(ctx, arr);
    return ops.astype(ctx, arr, .float32);
}

fn gate(ok: bool, name: []const u8) !void {
    if (!ok) {
        std.debug.print("FAIL {s}\n", .{name});
        return error.GateFailed;
    }
    std.debug.print("PASS {s}\n", .{name});
}

fn gateEq(comptime T: type, want: T, got: T, name: []const u8) !void {
    if (want != got) {
        std.debug.print("FAIL {s}: want {any} got {any}\n", .{ name, want, got });
        return error.GateFailed;
    }
}

const TraceCtx = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    ectx: EagerContext,
    artifacts_dir: []const u8,
};

/// Compare a tower stage against artifacts trace_<name>.npy (bisection aid;
/// silently skips stages whose trace file was not dumped).
fn traceHook(opaque_ctx: *anyopaque, name: []const u8, tensor: Array) anyerror!void {
    const tctx: *TraceCtx = @ptrCast(@alignCast(opaque_ctx));
    var pbuf: [256]u8 = undefined;
    const path = std.fmt.bufPrint(&pbuf, "{s}/trace_{s}.npy", .{ tctx.artifacts_dir, name }) catch unreachable;
    var np = readNpy(tctx.allocator, tctx.io, path, "<f4") catch |err| switch (err) {
        error.SkipZigTest => return, // trace file not dumped — skip stage
        else => return err,
    };
    defer np.deinit(tctx.allocator);
    const f = try arrayF32(tctx.ectx, tensor);
    defer f.deinit();
    try f.eval();
    const got = try f.dataSlice(f32);
    const want = np.f32s();
    if (got.len != want.len) {
        std.debug.print("trace {s}: SIZE MISMATCH got {d} want {d}\n", .{ name, got.len, want.len });
        return;
    }
    std.debug.print("trace {s}: cos={d:.8} max_abs={d:.6}\n", .{ name, cosSim(got, want), maxAbsErr(got, want) });
}

const Meta = struct {
    input_ids: []u32,
    n_image_pads: usize,
    grid_thw: [3]i64,
    rope_delta: i64,
    merge_size: i64,
    image_token_id: u32,
    image: []const u8,
};

fn loadMeta(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !std.json.Parsed(Meta) {
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 24)) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    // NOTE: `text` intentionally outlives the parse — string fields of Meta
    // (e.g. image) slice into it. Callers use an arena, so this is reclaimed
    // in bulk; do not free it here.
    return std.json.parseFromSlice(Meta, allocator, text, .{ .ignore_unknown_fields = true });
}

fn loadPrefixedWeights(
    allocator: std.mem.Allocator,
    index: *safetensors_reader.TensorIndex,
    prefix: []const u8,
) !std.StringHashMap(Array) {
    var map = std.StringHashMap(Array).init(allocator);
    errdefer {
        var it = map.iterator();
        while (it.next()) |e| {
            allocator.free(e.key_ptr.*);
            e.value_ptr.deinit();
        }
        map.deinit();
    }
    var it = index.entries.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (!std.mem.startsWith(u8, name, prefix)) continue;
        try map.put(try allocator.dupe(u8, name), try index.loadTensor(name));
    }
    return map;
}

pub fn main(init: std.process.Init) !void {
    c.initErrorHandler();
    run(init) catch |err| switch (err) {
        error.SkipZigTest => std.debug.print("SKIP: vision oracle artifacts or model absent\n", .{}),
        else => return err,
    };
}

fn run(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;

    // Skip when the oracle artifacts or the local checkpoint are absent.
    const artifacts_dir: []const u8 = if (std.c.getenv("DMLX_VISION_ORACLE_DIR")) |s|
        std.mem.span(s)
    else
        ARTIFACTS_DIR;
    std.Io.Dir.cwd().access(io, artifacts_dir, .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, MODEL_DIR, .{}) catch return error.SkipZigTest;

    const aa = allocator;

    const meta = try loadMeta(aa, io, try std.fmt.allocPrint(aa, "{s}/meta.json", .{artifacts_dir}));
    defer meta.deinit();
    // The dump must come from the same fixture this harness preprocesses.
    if (!std.mem.eql(u8, std.fs.path.basename(meta.value.image), std.fs.path.basename(FIXTURE))) {
        std.log.err("meta image '{s}' != fixture '{s}'", .{ meta.value.image, FIXTURE });
        return error.BadNpy;
    }
    const ids = meta.value.input_ids;
    const L = ids.len;

    var pixels_np = try readNpy(aa, io, try std.fmt.allocPrint(aa, "{s}/pixels.npy", .{artifacts_dir}), "<f4");
    var tower_np = try readNpy(aa, io, try std.fmt.allocPrint(aa, "{s}/tower_features.npy", .{artifacts_dir}), "<f4");
    var merged_np = try readNpy(aa, io, try std.fmt.allocPrint(aa, "{s}/merged_embeddings.npy", .{artifacts_dir}), "<f4");
    var pos_np = try readNpy(aa, io, try std.fmt.allocPrint(aa, "{s}/position_ids.npy", .{artifacts_dir}), "<i8");
    var cos_np = try readNpy(aa, io, try std.fmt.allocPrint(aa, "{s}/cos.npy", .{artifacts_dir}), "<f4");
    var sin_np = try readNpy(aa, io, try std.fmt.allocPrint(aa, "{s}/sin.npy", .{artifacts_dir}), "<f4");
    var logits_np = try readNpy(aa, io, try std.fmt.allocPrint(aa, "{s}/logits_last.npy", .{artifacts_dir}), "<f4");

    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });
    defer ctx.deinit();

    // ---- Config + weight index ----
    const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, MODEL_DIR ++ "/config.json", aa, .limited(1 << 20));
    var cfg = try qwen38_loader.parseQwen38Config(aa, cfg_text);
    defer cfg.deinit();
    var index = try safetensors_reader.loadOrBuildIndex(aa, MODEL_DIR);
    defer index.deinit();

    // ---- Gate 0 (diagnostic): preprocess pixels vs oracle ----
    var pp = try preprocess.preprocessImageFile(allocator, ctx, FIXTURE);
    defer pp.deinit();
    try gateEq(i64, 3, pos_np.shape[0], "position_ids channels");
    try gateEq(i64, @intCast(L), pos_np.shape[1], "position_ids length");
    for (0..3) |i|
        try gateEq(i64, meta.value.grid_thw[i], @as(i64, pp.grid_thw[i]), "grid_thw");
    try gateEq(usize, meta.value.n_image_pads, pp.n_image_pad, "n_image_pads");
    {
        const pf = try arrayF32(ctx, pp.pixels);
        defer pf.deinit();
        try pf.eval();
        const got = try pf.dataSlice(f32);
        const want = pixels_np.f32s();
        try gateEq(usize, want.len, got.len, "pixels length");
        const cos = cosSim(got, want);
        std.debug.print("pixels: cos={d:.6} max_abs={d:.4} (diagnostic)\n", .{ cos, maxAbsErr(got, want) });
    }

    // ---- Gate 1: tower features cos >= 0.999 ----
    var vision_map = try loadPrefixedWeights(aa, &index, "vision_tower.");
    defer {
        var it = vision_map.iterator();
        while (it.next()) |e| e.value_ptr.deinit();
        vision_map.deinit();
    }
    const vcfg = vit_mod.Qwen38VitConfig{
        .depth = cfg.vision.depth,
        .hidden = @intCast(cfg.vision.hidden),
        .heads = @intCast(cfg.vision.heads),
        .intermediate = @intCast(cfg.vision.intermediate),
        .patch = @intCast(cfg.vision.patch),
        .temporal_patch = @intCast(cfg.vision.temporal_patch),
        .merge = @intCast(cfg.vision.merge),
        .out_hidden = @intCast(cfg.vision.out_hidden),
        .num_pos = @intCast(cfg.vision.num_pos),
        .in_ch = @intCast(cfg.vision.in_ch),
    };
    var vit = try vit_mod.Qwen38Vit.init(allocator, ctx, vcfg, &vision_map);
    defer vit.deinit();
    {
        // Bisection diagnostic: feed the ORACLE pixels through the Zig tower.
        // cos ~1 isolates the remaining gap to preprocessing; < 0.999
        // implicates the tower itself.
        const oracle_pix_f32 = try Array.fromData(allocator, f32, @constCast(pixels_np.f32s()), &.{
            @intCast(pixels_np.shape[0]), @intCast(pixels_np.shape[1]),
        });
        defer oracle_pix_f32.deinit();
        const oracle_pix = try ops.astype(ctx, oracle_pix_f32, .bfloat16);
        defer oracle_pix.deinit();
        var tctx = TraceCtx{ .allocator = aa, .io = io, .ectx = ctx, .artifacts_dir = artifacts_dir };
        const of = try vit.forwardTraced(oracle_pix, pp.grid_thw[0], pp.grid_thw[1], pp.grid_thw[2], .{
            .ctx = &tctx,
            .func = traceHook,
        });
        defer of.deinit();
        const off32 = try arrayF32(ctx, of);
        defer off32.deinit();
        try off32.eval();
        std.debug.print("tower_from_oracle_pixels: cos={d:.8}\n", .{cosSim(try off32.dataSlice(f32), tower_np.f32s())});
    }
    const feats = try vit.forward(pp.pixels, pp.grid_thw[0], pp.grid_thw[1], pp.grid_thw[2]);
    defer feats.deinit();
    {
        const ff = try arrayF32(ctx, feats);
        defer ff.deinit();
        try ff.eval();
        const got = try ff.dataSlice(f32);
        const want = tower_np.f32s();
        try gateEq(usize, want.len, got.len, "tower_features length");
        const cos = cosSim(got, want);
        std.debug.print("tower_features: cos={d:.8} max_abs={d:.6}\n", .{ cos, maxAbsErr(got, want) });
        try gate(cos >= 0.999, "tower_features cos >= 0.999");
    }

    // ---- Gate 2: MRoPE position_ids exact ----
    const ms: i32 = @intCast(meta.value.merge_size);
    const blocks = [_][3]i32{.{ pp.grid_thw[0], @divExact(pp.grid_thw[1], ms), @divExact(pp.grid_thw[2], ms) }};
    var mres = try qwen38_mrope.computePositionIds(
        allocator,
        ids,
        &blocks,
        meta.value.image_token_id,
        cfg.vision_start_token_id,
    );
    defer mres.deinit(allocator);
    {
        const want = pos_np.i64s();
        var exact = true;
        for (0..3) |ch| {
            for (mres.position_ids[ch], 0..) |p, i| {
                if (want[ch * L + i] != @as(i64, p)) {
                    std.debug.print("position_ids mismatch ch={d} pos={d}: want {d} got {d}\n", .{ ch, i, want[ch * L + i], p });
                    exact = false;
                }
            }
        }
        try gate(exact, "position_ids exact");
        try gateEq(i64, meta.value.rope_delta, @as(i64, mres.rope_delta), "rope_delta");
        std.debug.print("position_ids: exact, rope_delta={d}\n", .{mres.rope_delta});
    }

    // ---- Gate 3: per-slot cos/sin tables (max_abs < 1e-4) ----
    var mr = try qwen38.buildMropeCosSin(allocator, &cfg, &mres, L);
    defer mr.deinit();
    for ([_]struct { arr: Array, np: *const NpyArray, name: []const u8 }{
        .{ .arr = mr.cos, .np = &cos_np, .name = "cos" },
        .{ .arr = mr.sin, .np = &sin_np, .name = "sin" },
    }) |case| {
        try case.arr.eval();
        const got = try case.arr.dataSlice(f32);
        const want = case.np.f32s();
        try gateEq(usize, want.len, got.len, "cos/sin length");
        const mae = maxAbsErr(got, want);
        std.debug.print("{s}: max_abs={d:.6} cos={d:.8}\n", .{ case.name, mae, cosSim(got, want) });
        try gate(mae < 1e-4, case.name);
    }

    // ---- Full language model for merged-embed + logits gates ----
    const weights = try loadPrefixedWeights(aa, &index, "language_model.");
    var model = try qwen38.Qwen38ServingModel.init(aa, ctx, &cfg, weights);
    model.ws = qwen38.WeightSource.fromMap(&model.weights, ctx, &model.ws_backing);
    defer {
        for (model.caches) |*lc| lc.deinit();
        aa.free(model.caches);
        for (model.layers) |*lp| lp.deinitShallow();
        aa.free(model.layers);
        if (model.embed_table_bf16) |*t| t.deinit();
        if (model.final_norm_f32) |*t| t.deinit();
        if (model.gdn_step_compiled) |*cl| cl.deinit();
        if (model.mlp_swiglu_compiled) |*cl| cl.deinit();
        if (model.mlp_swiglu_fused_compiled) |*cl| cl.deinit();
        if (model.attn_precache_compiled) |*cl| cl.deinit();
        var it = model.weights.iterator();
        while (it.next()) |e| e.value_ptr.deinit();
        model.weights.deinit();
    }

    // ---- Gate 4: merged embeddings — text rows exact, image rows cos >= 0.99 ----
    {
        const ids_i32 = try allocator.alloc(i32, L);
        for (ids, 0..) |t, i| ids_i32[i] = @intCast(t);
        const id_arr = try Array.fromData(allocator, i32, ids_i32, &[_]i32{@intCast(L)});
        defer id_arr.deinit();
        const rows = try shape_mod.takeAxis(ctx, model.embed_table_bf16.?, id_arr, 0);
        defer rows.deinit();
        const merged = try qwen38.mergeVisionFeatures(ctx, rows, ids, feats, cfg.image_token_id);
        defer merged.deinit();
        const mf = try arrayF32(ctx, merged);
        defer mf.deinit();
        try mf.eval();
        const got = try mf.dataSlice(f32);
        const want = merged_np.f32s();
        const hidden: usize = @intCast(cfg.hidden_size);
        try gateEq(usize, L * hidden, want.len, "merged oracle length");
        try gateEq(usize, want.len, got.len, "merged zig length");
        var img_cos_min: f64 = 1.0;
        var text_exact = true;
        for (0..L) |i| {
            const g_row = got[i * hidden .. (i + 1) * hidden];
            const w_row = want[i * hidden .. (i + 1) * hidden];
            if (ids[i] == meta.value.image_token_id) {
                img_cos_min = @min(img_cos_min, cosSim(g_row, w_row));
            } else {
                for (w_row, g_row) |w, g| {
                    if (w != g) {
                        std.debug.print("merged text row {d} mismatch\n", .{i});
                        text_exact = false;
                        break;
                    }
                }
            }
        }
        std.debug.print("merged_embeddings: text rows exact={} image-row min cos={d:.8}\n", .{ text_exact, img_cos_min });
        try gate(text_exact, "merged_embeddings text rows exact");
        // Image rows just echo the tower output (gated above) — sanity bound.
        try gate(img_cos_min >= 0.99, "merged_embeddings image rows cos >= 0.99");
    }

    // ---- Gate 5: last-position logits cos >= 0.995 (noise floor, see header) ----
    var logits_cos: f64 = 0;
    {
        // Bisection diagnostic: run the LM on the ORACLE tower features.
        // cos ~1 puts the remaining gap entirely on the tower; below target
        // implicates the Zig LM vision path (mrope apply / prefill).
        const oracle_feats_f32 = try Array.fromData(allocator, f32, @constCast(tower_np.f32s()), &.{
            @intCast(tower_np.shape[0]), @intCast(tower_np.shape[1]),
        });
        defer oracle_feats_f32.deinit();
        const oracle_feats = try ops.astype(ctx, oracle_feats_f32, .bfloat16);
        defer oracle_feats.deinit();
        const res2 = try model.forwardTokensWithVisionCapture(ids, .{ .features = oracle_feats, .blocks = &blocks }, .all);
        defer res2.logits.deinit();
        defer res2.hidden.?.deinit();
        const lf2 = try arrayF32(ctx, res2.logits);
        defer lf2.deinit();
        try lf2.eval();
        const got2_all = try lf2.dataSlice(f32);
        const want2 = logits_np.f32s();
        std.debug.print("logits_from_oracle_features: cos={d:.8}\n", .{cosSim(got2_all[got2_all.len - want2.len ..], want2)});

        // Pre-final-norm hidden after layer 63 vs the oracle capture.
        var h63_np = try readNpy(aa, io, try std.fmt.allocPrint(aa, "{s}/lm_hidden_layer63.npy", .{artifacts_dir}), "<f4");
        const hf = try arrayF32(ctx, res2.hidden.?);
        defer hf.deinit();
        try hf.eval();
        const hg = try hf.dataSlice(f32);
        const hw = h63_np.f32s();
        try gateEq(usize, hw.len, hg.len, "hidden length");
        const hidden_dim: usize = @intCast(cfg.hidden_size);
        var min_txt: f64 = 1.0;
        var min_img: f64 = 1.0;
        for (0..L) |i| {
            const cs = cosSim(hg[i * hidden_dim .. (i + 1) * hidden_dim], hw[i * hidden_dim .. (i + 1) * hidden_dim]);
            if (ids[i] == meta.value.image_token_id) min_img = @min(min_img, cs) else min_txt = @min(min_txt, cs);
        }
        std.debug.print("lm_hidden_layer63: overall cos={d:.8} max_abs={d:.4} min_row_cos text={d:.8} image={d:.8}\n", .{
            cosSim(hg, hw), maxAbsErr(hg, hw), min_txt, min_img,
        });
    }
    {
        const logits = try model.forwardTokensWithVision(ids, .{ .features = feats, .blocks = &blocks });
        defer logits.deinit();
        const lf = try arrayF32(ctx, logits);
        defer lf.deinit();
        try lf.eval();
        const got_all = try lf.dataSlice(f32);
        const want = logits_np.f32s();
        const vocab = want.len;
        // logits may be [1, 1, vocab] (last position only) or [1, L, vocab].
        const got = got_all[got_all.len - vocab ..];
        const cos = cosSim(got, want);
        std.debug.print("logits_last: cos={d:.8} max_abs={d:.6}\n", .{ cos, maxAbsErr(got, want) });
        logits_cos = cos;
    }

    // DMLX_VISION_BISECT=1: rerun the prefill depth-truncated (fresh serving
    // models) and diff against the oracle's per-layer hidden captures,
    // splitting text vs image rows.
    if (std.c.getenv("DMLX_VISION_BISECT") != null) {
        const oracle_feats_f32 = try Array.fromData(allocator, f32, @constCast(tower_np.f32s()), &.{
            @intCast(tower_np.shape[0]), @intCast(tower_np.shape[1]),
        });
        defer oracle_feats_f32.deinit();
        const oracle_feats = try ops.astype(ctx, oracle_feats_f32, .bfloat16);
        defer oracle_feats.deinit();
        for ([_]struct { depth: u32, name: []const u8 }{
            .{ .depth = 1, .name = "lm_hidden_layer0" },
            .{ .depth = 4, .name = "lm_hidden_layer3" },
            .{ .depth = 9, .name = "lm_hidden_layer8" },
            .{ .depth = 17, .name = "lm_hidden_layer16" },
            .{ .depth = 25, .name = "lm_hidden_layer24" },
            .{ .depth = 33, .name = "lm_hidden_layer32" },
            .{ .depth = 41, .name = "lm_hidden_layer40" },
            .{ .depth = 49, .name = "lm_hidden_layer48" },
            .{ .depth = 57, .name = "lm_hidden_layer56" },
        }) |d| {
            var cfg_d = cfg; // shallow copy; slices shared with cfg (not deinited)
            cfg_d.num_hidden_layers = d.depth;
            const w2 = try loadPrefixedWeights(aa, &index, "language_model.");
            var m2 = try qwen38.Qwen38ServingModel.init(aa, ctx, &cfg_d, w2);
            m2.ws = qwen38.WeightSource.fromMap(&m2.weights, ctx, &m2.ws_backing);
            defer {
                for (m2.caches) |*lc| lc.deinit();
                aa.free(m2.caches);
                for (m2.layers) |*lp| lp.deinitShallow();
                aa.free(m2.layers);
                if (m2.embed_table_bf16) |*t| t.deinit();
                if (m2.final_norm_f32) |*t| t.deinit();
                if (m2.gdn_step_compiled) |*cl| cl.deinit();
                if (m2.mlp_swiglu_compiled) |*cl| cl.deinit();
                if (m2.mlp_swiglu_fused_compiled) |*cl| cl.deinit();
                if (m2.attn_precache_compiled) |*cl| cl.deinit();
                var it = m2.weights.iterator();
                while (it.next()) |e| e.value_ptr.deinit();
                m2.weights.deinit();
            }
            const res = try m2.forwardTokensWithVisionCapture(ids, .{ .features = oracle_feats, .blocks = &blocks }, .all);
            defer res.logits.deinit();
            defer res.hidden.?.deinit();
            const hnp_path = try std.fmt.allocPrint(aa, "{s}/{s}.npy", .{ artifacts_dir, d.name });
            var hnp = try readNpy(aa, io, hnp_path, "<f4");
            const hf = try arrayF32(ctx, res.hidden.?);
            defer hf.deinit();
            try hf.eval();
            const hg = try hf.dataSlice(f32);
            const hw = hnp.f32s();
            try gateEq(usize, hw.len, hg.len, "hidden length");
            const hd: usize = @intCast(cfg.hidden_size);
            var min_txt: f64 = 1.0;
            var min_img: f64 = 1.0;
            var worst_i: usize = 0;
            var worst_c: f64 = 1.0;
            for (0..L) |i| {
                const cs = cosSim(hg[i * hd .. (i + 1) * hd], hw[i * hd .. (i + 1) * hd]);
                if (cs < worst_c) {
                    worst_c = cs;
                    worst_i = i;
                }
                if (ids[i] == meta.value.image_token_id) min_img = @min(min_img, cs) else min_txt = @min(min_txt, cs);
            }
            std.debug.print("{s}: overall cos={d:.8} max_abs={d:.4} min_row text={d:.8} image={d:.8} worst@{d}\n", .{
                d.name, cosSim(hg, hw), maxAbsErr(hg, hw), min_txt, min_img, worst_i,
            });
        }
    }

    try gate(logits_cos >= 0.995, "logits_last cos >= 0.995");
}
