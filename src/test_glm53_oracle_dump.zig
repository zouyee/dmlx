/// GLM-5.3 oracle alignment harness (Zig side).
///
/// Loads the prep'd checkpoint (~/models/GLM-5.3-Flash-MLX-mixed-4_8bit-dmlx),
/// runs the fixed 7-token fixture through Glm53Model (embed → 45 layers of
/// KDA/MLA attention + dense/MoE MLP with mHC apply_branch semantics →
/// norm → lm_head) with per-layer weight eviction and on-demand MoE expert
/// pread, and dumps each segment to artifacts/glm53_oracle/{seg}.bin for
/// comparison against the Python oracle fixtures
/// (tests/fixtures/glm53_oracle/*.npy) via scripts/compare_glm53_oracle.py.
///
/// bin format: [ndim: i32][dims: i32 x ndim][f32 row-major payload].
///
/// Usage:
///   zig build run-glm53-oracle-dump -- [--model-dir DIR] [--out-dir DIR]
///     [--seg SEG] [--full] [--layers 0,3,4]
/// SEG ∈ embed | layer0_kda_out | layer3_mla_out | layer3_indexer_topk |
///       hidden_final | logits | all (default all = full 45-layer run).
/// --full is an alias for --seg all. --layers restricts which layers run
/// their branches (debug smoke; residual passes through unlisted layers).
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const safetensors_reader = @import("mlx").safetensors_reader;
const glm53_loader = @import("models/glm53_loader.zig");
const glm53 = @import("models/glm53.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

const token_ids = [_]i32{ 17, 42, 314, 2718, 7, 99, 1234 };

fn dumpBin(allocator: std.mem.Allocator, out_dir: []const u8, name: []const u8, arr: Array) !void {
    std.debug.assert(arr.dtype() == .float32);
    try arr.eval();
    const sh = arr.shape();
    std.debug.assert(sh.len <= 8);
    const data = try arr.dataSlice(f32);

    var dir_buf: [1024]u8 = undefined;
    const dir_z = try std.fmt.bufPrintZ(&dir_buf, "{s}", .{out_dir});
    // mkdir -p: create each path component, ignoring EEXIST.
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

fn loadConfig(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) !glm53_loader.Glm53Config {
    const path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024));
    defer allocator.free(text);
    return glm53_loader.parseGlm53Config(allocator, text);
}

/// Hook context: dumps every segment the model reports as f32 .bin.
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

    var model_dir: []const u8 = "";
    var out_dir: []const u8 = "artifacts/glm53_oracle";
    var seg: []const u8 = "all";
    var layers_arg: ?[]const u8 = null;

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
        } else if (std.mem.eql(u8, arg, "--full")) {
            seg = "all";
        } else if (std.mem.eql(u8, arg, "--layers")) {
            ai += 1;
            layers_arg = if (ai < argv.len) argv[ai] else return error.MissingArgValue;
        } else return error.UnknownArg;
    }
    if (model_dir.len == 0) {
        const home_z = std.c.getenv("HOME") orelse return error.NoHome;
        const home = std.mem.span(home_z);
        model_dir = try std.fs.path.join(allocator, &.{ home, "models/GLM-5.3-Flash-MLX-mixed-4_8bit-dmlx" });
    }

    var layer_filter: ?[]const u32 = null;
    if (layers_arg) |la| {
        var list = std.ArrayList(u32).empty;
        var it = std.mem.splitScalar(u8, la, ',');
        while (it.next()) |part| {
            if (part.len == 0) continue;
            try list.append(allocator, try std.fmt.parseInt(u32, part, 10));
        }
        layer_filter = list.items;
    }

    // Early-stop for cheap segment reruns: segments through layer N only need
    // layers 0..N. embed-only runs no layers (empty filter skips every layer).
    const no_layers = [_]u32{};
    if (std.mem.eql(u8, seg, "embed")) layer_filter = &no_layers;
    const stop_after: ?u32 = if (std.mem.eql(u8, seg, "layer0_kda_out"))
        0
    else if (std.mem.eql(u8, seg, "layer3_mla_out") or std.mem.eql(u8, seg, "layer3_indexer_topk"))
        3
    else
        null;

    var cfg = try loadConfig(allocator, io, model_dir);
    defer cfg.deinit();

    const stream = c.c.mlx_default_gpu_stream_new();
    const ctx = EagerContext.initWithStream(allocator, .{ .inner = stream });
    // LIFO: declared before every Array/index defer below, so the stream is
    // released only after all arrays and the weight index are freed.
    defer ctx.deinit();

    var index = try safetensors_reader.loadOrBuildIndex(allocator, model_dir);
    defer index.deinit();

    // Expert row reads via PartialTensorReader's pread fallback (FdPool).
    // mmap zero-copy is deliberately NOT used here: it mis-evaluates managed
    // payload arrays in GPU ops on this path (NaN in MoE down proj, verified
    // 2026-09-17) and production expert streaming disables mmap for the same
    // class of reason ("VM page fault storms", see expert_stream.zig).
    var fd_pool = safetensors_reader.FdPool.init(allocator);
    defer fd_pool.deinit();
    try fd_pool.openAll(&index);
    var reader = safetensors_reader.PartialTensorReader.init(allocator, &index, &fd_pool);

    var dctx = DumpCtx{ .allocator = allocator, .out_dir = out_dir, .ectx = ctx };

    const model = glm53.Glm53Model{
        .allocator = allocator,
        .ctx = ctx,
        .cfg = &cfg,
        .index = &index,
        .reader = &reader,
        .stream = ctx.stream.inner,
    };

    const logits = try model.forward(&token_ids, .{
        .layer_filter = layer_filter,
        .stop_after = stop_after,
        .hook = .{ .ctx = &dctx, .func = dumpHook },
    });
    defer logits.deinit();

    if (std.mem.eql(u8, seg, "all") or std.mem.eql(u8, seg, "logits") or std.mem.eql(u8, seg, "hidden_final")) {
        const logits_f32 = try ops.astype(ctx, logits, .float32);
        defer logits_f32.deinit();
        try dumpBin(allocator, out_dir, "logits", logits_f32);
    }
}
