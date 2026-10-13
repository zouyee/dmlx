/// Qwen3.8 oracle alignment harness (Zig side).
///
/// Loads /Users/zouyee/models/Qwen3.8-27B-4bit, runs the fixed prompt through
/// Qwen38Model (embed → 64 layers of GDN/full-attention + SwiGLU MLP with
/// plain residuals → norm → lm_head) with per-layer weight loading from the
/// safetensors index, and dumps each segment to OUT_DIR/{seg}.bin for
/// comparison against the Python oracle (artifacts/qwen38_oracle/*.npy) via
/// scripts/compare_qwen38_oracle.py.
///
/// bin format: [ndim: i32][dims: i32 x ndim][f32 row-major payload].
///
/// Usage:
///   zig build run-qwen38-oracle -- [--model-dir DIR] [--out-dir DIR]
///     [--seg SEG] [--tokens "id,id,..."]
/// SEG ∈ embed | layer0_gdn_out | layer3_attn_out | hidden_final | logits |
///       all (default all = full 64-layer run).
/// --tokens defaults to the ids baked in from scripts/qwen38_oracle_dump.py
/// meta.json (prompt "The capital of France is").
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const safetensors_reader = @import("mlx").safetensors_reader;
const qwen38_loader = @import("models/qwen38_loader.zig");
const qwen38 = @import("models/qwen38.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

/// Baked-in token ids for "The capital of France is" from
/// artifacts/qwen38_oracle/meta.json (kept in sync with the python dump).
const default_token_ids = [_]i32{ 760, 6511, 314, 9338, 369 };

fn dumpBin(allocator: std.mem.Allocator, out_dir: []const u8, name: []const u8, arr: Array) !void {
    std.debug.assert(arr.dtype() == .float32);
    try arr.eval();
    const sh = arr.shape();
    std.debug.assert(sh.len <= 8);
    const data = try arr.dataSlice(f32);

    var dir_buf: [1024]u8 = undefined;
    const dir_z = try std.fmt.bufPrintZ(&dir_buf, "{s}", .{out_dir});
    {
        var i: usize = 0;
        while (i < dir_z.len) : (i += 1) {
            if (dir_z[i] == '/' and i > 0) {
                dir_z[i] = 0;
                _ = std.c.mkdir(dir_z[0..i :0].ptr, 0o755);
                dir_z[i] = '/';
            }
        }
        _ = std.c.mkdir(dir_z.ptr, 0o755);
    }

    const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}.bin", .{ out_dir, name }, 0);
    const fp = std.c.fopen(path.ptr, "wb") orelse return error.OpenFailed;
    defer _ = std.c.fclose(fp);

    var header: [9]i32 = undefined;
    const ndim: i32 = @intCast(sh.len);
    header[0] = ndim;
    for (sh, 0..) |d, i| header[1 + i] = d;
    const header_bytes = std.mem.sliceAsBytes(header[0 .. 1 + sh.len]);
    if (std.c.fwrite(header_bytes.ptr, 1, header_bytes.len, fp) != header_bytes.len) return error.WriteFailed;
    const payload = std.mem.sliceAsBytes(data);
    if (payload.len > 0 and std.c.fwrite(payload.ptr, 1, payload.len, fp) != payload.len) return error.WriteFailed;
    std.log.info("dumped {s}.bin ndim={d} n={d}", .{ name, ndim, data.len });
}

fn loadConfig(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !qwen38_loader.Qwen38Config {
    const path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024));
    defer allocator.free(text);
    return qwen38_loader.parseQwen38Config(allocator, text);
}

const DumpCtx = struct {
    allocator: std.mem.Allocator,
    out_dir: []const u8,
    ectx: EagerContext,
};

fn dumpHook(opaque_ctx: *anyopaque, name: []const u8, tensor: Array) anyerror!void {
    const dctx: *DumpCtx = @ptrCast(@alignCast(opaque_ctx));
    const f32_arr = try ops.astype(dctx.ectx, tensor, .float32);
    defer f32_arr.deinit();
    try dumpBin(dctx.allocator, dctx.out_dir, name, f32_arr);
}

pub fn main(init: std.process.Init) !void {
    c.initErrorHandler();
    const allocator = init.arena.allocator();
    const io = init.io;

    var model_dir: []const u8 = "/Users/zouyee/models/Qwen3.8-27B-4bit";
    var out_dir: []const u8 = "artifacts/qwen38_oracle_zig";
    var seg: []const u8 = "all";
    var tokens_arg: ?[]const u8 = null;

    const argv = try init.minimal.args.toSlice(allocator);
    var ai: usize = 1;
    while (ai < argv.len) : (ai += 1) {
        const arg: []const u8 = argv[ai];
        if (std.mem.eql(u8, arg, "--model-dir")) {
            ai += 1;
            model_dir = if (ai < argv.len) argv[ai] else return error.MissingArgValue;
        } else if (std.mem.eql(u8, arg, "--out-dir")) {
            ai += 1;
            out_dir = if (ai < argv.len) argv[ai] else return error.MissingArgValue;
        } else if (std.mem.eql(u8, arg, "--seg")) {
            ai += 1;
            seg = if (ai < argv.len) argv[ai] else return error.MissingArgValue;
        } else if (std.mem.eql(u8, arg, "--tokens")) {
            ai += 1;
            tokens_arg = if (ai < argv.len) argv[ai] else return error.MissingArgValue;
        } else return error.UnknownArg;
    }

    var tokens_buf = std.ArrayList(i32).empty;
    if (tokens_arg) |ta| {
        var it = std.mem.splitScalar(u8, ta, ',');
        while (it.next()) |part| {
            if (part.len == 0) continue;
            try tokens_buf.append(allocator, try std.fmt.parseInt(i32, part, 10));
        }
    } else {
        try tokens_buf.appendSlice(allocator, &default_token_ids);
    }
    const token_ids = tokens_buf.items;
    std.log.info("tokens ({d}): {any}", .{ token_ids.len, token_ids });

    // Early-stop for cheap segment reruns: segments through layer N only need
    // layers 0..N.
    const max_layer: ?u32 = if (std.mem.eql(u8, seg, "embed"))
        0
    else if (std.mem.eql(u8, seg, "layer0_gdn_out"))
        0
    else if (std.mem.eql(u8, seg, "layer3_attn_out"))
        3
    else
        null;

    var cfg = try loadConfig(allocator, io, model_dir);
    defer cfg.deinit();

    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });
    defer ctx.deinit();

    var index = try safetensors_reader.loadOrBuildIndex(allocator, model_dir);
    defer index.deinit();

    var dctx = DumpCtx{ .allocator = allocator, .out_dir = out_dir, .ectx = ctx };

    const model = qwen38.Qwen38Model{
        .allocator = allocator,
        .ctx = ctx,
        .cfg = &cfg,
        .ws = qwen38.WeightSource.fromIndex(&index),
    };

    const logits = try model.forward(token_ids, .{
        .max_layer = max_layer,
        .hook = .{ .ctx = &dctx, .func = dumpHook },
    });
    defer logits.deinit();

    if (std.mem.eql(u8, seg, "all") or std.mem.eql(u8, seg, "logits") or std.mem.eql(u8, seg, "hidden_final")) {
        const logits_f32 = try ops.astype(ctx, logits, .float32);
        defer logits_f32.deinit();
        try dumpBin(allocator, out_dir, "logits", logits_f32);
    }
}
