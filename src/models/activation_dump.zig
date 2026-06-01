//! Per-layer activation dumping for metal-vs-MLX numerical alignment.
//!
//! Gated by the `DSV4_DUMP_DIR` environment variable (off by default — zero
//! impact on normal runs). When set, the model forward pass writes each
//! layer's hidden state to `${DSV4_DUMP_DIR}/layer_<NN>.npy` as a float32
//! NumPy array, plus `${DSV4_DUMP_DIR}/final_norm.npy` and `logits.npy`.
//!
//! Use `scripts/compare_metal_mlx.py` to diff two dump dirs (one MLX, one
//! metal) layer-by-layer. See `docs/analysis/dsv4-first-class-support-plan.md`
//! Phase 1.
//!
//! Implementation note: uses C stdio (`std.c.fopen`/`fwrite`) rather than the
//! `std.Io` API, because `DSV4Model.forward` has no `std.Io` context in scope
//! and dumping must be a zero-friction side channel.

const std = @import("std");
const ops = @import("mlx").ops;
const array_mod = @import("mlx").array;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;

var cached_dir: ?[:0]const u8 = null;
var checked: bool = false;

/// Returns the dump directory (NUL-terminated) if `DSV4_DUMP_DIR` is set, else
/// null. Cached on first call. Creates the directory if needed.
pub fn dumpDir() ?[:0]const u8 {
    if (!checked) {
        checked = true;
        if (std.c.getenv("DSV4_DUMP_DIR")) |raw| {
            const dir = std.mem.span(raw);
            if (dir.len > 0) {
                // mkdir (ignore EEXIST). 0o755 perms.
                _ = std.c.mkdir(raw, 0o755);
                cached_dir = dir;
                std.log.info("[dump] activation dumping enabled -> {s}", .{dir});
            }
        }
    }
    return cached_dir;
}

pub fn enabled() bool {
    return dumpDir() != null;
}

/// Dump an array to `${DSV4_DUMP_DIR}/<name>.npy` as float32.
/// No-op if dumping is disabled. Errors are logged, not propagated, so dumping
/// never breaks a forward pass.
pub fn dump(ctx: EagerContext, name: []const u8, arr: Array) void {
    const dir = dumpDir() orelse return;
    dumpImpl(ctx, dir, name, arr) catch |err| {
        std.log.warn("[dump] failed to dump {s}: {any}", .{ name, err });
    };
}

/// Convenience: dump a per-layer tensor as `layer_<NN>.npy`.
pub fn dumpLayer(ctx: EagerContext, layer_idx: usize, arr: Array) void {
    if (!enabled()) return;
    var buf: [64]u8 = undefined;
    const name = std.fmt.bufPrint(&buf, "layer_{d:0>2}", .{layer_idx}) catch return;
    dump(ctx, name, arr);
}

fn dumpImpl(ctx: EagerContext, dir: [:0]const u8, name: []const u8, arr: Array) !void {
    // Convert to float32 and materialize.
    const f32_arr = try ops.astype(ctx, arr, .float32);
    defer f32_arr.deinit();
    try f32_arr.eval();
    const data = try f32_arr.dataPtr(f32);
    const n = f32_arr.size();
    const shape = f32_arr.shape();

    var path_buf: [1024]u8 = undefined;
    const path = try std.fmt.bufPrintZ(&path_buf, "{s}/{s}.npy", .{ dir, name });

    const fp = std.c.fopen(path.ptr, "wb") orelse return error.OpenFailed;
    defer _ = std.c.fclose(fp);

    try writeNpyHeader(fp, shape);
    const bytes = std.mem.sliceAsBytes(data[0..n]);
    if (bytes.len > 0) {
        const written = std.c.fwrite(bytes.ptr, 1, bytes.len, fp);
        if (written != bytes.len) return error.WriteFailed;
    }
}

/// Write a NumPy .npy v1.0 header for a float32 ('<f4') array.
fn writeNpyHeader(fp: *std.c.FILE, shape: []const i32) !void {
    var dict_buf: [256]u8 = undefined;
    var pos: usize = 0;
    pos += (try std.fmt.bufPrint(dict_buf[pos..], "{s}", .{"{'descr': '<f4', 'fortran_order': False, 'shape': ("})).len;
    for (shape, 0..) |dim, i| {
        if (i != 0) {
            pos += (try std.fmt.bufPrint(dict_buf[pos..], ", ", .{})).len;
        }
        pos += (try std.fmt.bufPrint(dict_buf[pos..], "{d}", .{dim})).len;
    }
    // NumPy expects a trailing comma for 1-D shapes; harmless for N-D too.
    if (shape.len == 1) {
        pos += (try std.fmt.bufPrint(dict_buf[pos..], ",", .{})).len;
    }
    pos += (try std.fmt.bufPrint(dict_buf[pos..], "), }}", .{})).len;
    const dict = dict_buf[0..pos];

    // Total header (magic(6) + ver(2) + len(2) + dict + padding + \n) must be
    // a multiple of 64.
    const prefix_len: usize = 6 + 2 + 2;
    const unpadded = prefix_len + dict.len + 1; // +1 for trailing newline
    const padded = std.mem.alignForward(usize, unpadded, 64);
    const pad = padded - unpadded;
    const header_len: u16 = @intCast(dict.len + pad + 1);

    var prefix: [10]u8 = .{ 0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00, 0, 0 };
    std.mem.writeInt(u16, prefix[8..10], header_len, .little);
    try writeAll(fp, &prefix);
    try writeAll(fp, dict);
    var i: usize = 0;
    while (i < pad) : (i += 1) try writeAll(fp, " ");
    try writeAll(fp, "\n");
}

fn writeAll(fp: *std.c.FILE, bytes: []const u8) !void {
    if (bytes.len == 0) return;
    const written = std.c.fwrite(bytes.ptr, 1, bytes.len, fp);
    if (written != bytes.len) return error.WriteFailed;
}
