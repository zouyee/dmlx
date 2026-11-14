/// `dmlx pull` — download a registered model from Hugging Face with resumable
/// transfers and SHA256 verification of LFS files.
///
/// Downloads shell out to the system curl (always present on macOS) so TLS,
/// proxy, and redirect handling stay with the platform. File listings come
/// from the HF tree API, which carries per-file LFS SHA256 used for
/// verification after each download.
const std = @import("std");
const catalog = @import("catalog.zig");

const HF_API = "https://huggingface.co/api/models";
const HF_RESOLVE = "https://huggingface.co";

const TreeEntry = struct {
    path: []const u8,
    type: []const u8 = "file",
    size: u64 = 0,
    /// LFS metadata; `oid` is the git-lfs SHA256 of the object.
    lfs: ?struct { oid: ?[]const u8 = null } = null,
};

const PullArgs = struct {
    alias: []const u8,
    force: bool = false,
};

pub fn run(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    var parsed: ?PullArgs = null;
    for (args) |arg| {
        if (std.mem.eql(u8, arg, "--force")) {
            if (parsed) |*p| p.force = true;
        } else if (std.mem.startsWith(u8, arg, "--source")) {
            std.log.err("--source: only the default Hugging Face source is supported right now", .{});
            return error.UnsupportedSource;
        } else if (std.mem.startsWith(u8, arg, "--")) {
            std.log.err("unknown pull option: {s}", .{arg});
            return error.UnknownOption;
        } else {
            if (parsed != null) return error.TooManyArguments;
            parsed = .{ .alias = arg };
        }
    }
    const pa = parsed orelse {
        std.log.err("usage: dmlx pull <model-alias> [--force]", .{});
        printAvailable();
        return error.MissingAlias;
    };

    const entry = catalog.find(pa.alias) orelse {
        std.log.err("unknown model: {s}", .{pa.alias});
        printAvailable();
        return error.UnknownModel;
    };

    const dest_dir = try catalog.installPath(allocator, entry);
    defer allocator.free(dest_dir);

    if (!pa.force and catalog.isCompleteInstall(io, dest_dir, entry)) {
        std.debug.print("{s} already installed at {s} (use --force to re-download)\n", .{
            entry.alias, dest_dir,
        });
        return;
    }

    // File listing with LFS digests.
    const tree_url = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}/tree/main?recursive=true",
        .{ HF_API, entry.hf_repo },
    );
    defer allocator.free(tree_url);
    std.debug.print("fetching file list: {s}\n", .{tree_url});
    const tree_json = try curlGet(allocator, io, tree_url, 64 << 20);
    defer allocator.free(tree_json);

    const parsed_tree = std.json.parseFromSlice(
        []TreeEntry,
        allocator,
        tree_json,
        .{ .ignore_unknown_fields = true, .allocate = .alloc_always },
    ) catch {
        std.log.err("failed to parse Hugging Face response for {s}", .{entry.hf_repo});
        return error.BadResponse;
    };
    defer parsed_tree.deinit();

    var total_bytes: u64 = 0;
    var file_count: usize = 0;
    for (parsed_tree.value) |fe| {
        if (!wantFile(fe)) continue;
        total_bytes += fe.size;
        file_count += 1;
    }
    std.debug.print("pulling {s} → {s}\n  {d} files, {d:.1} GB\n", .{
        entry.hf_repo,
        dest_dir,
        file_count,
        @as(f64, @floatFromInt(total_bytes)) / (1 << 30),
    });
    try std.Io.Dir.cwd().createDirPath(io, dest_dir);

    var done: usize = 0;
    for (parsed_tree.value) |fe| {
        if (!wantFile(fe)) continue;
        done += 1;
        try downloadFile(allocator, io, entry, dest_dir, fe, done, file_count, pa.force);
    }

    if (!catalog.isCompleteInstall(io, dest_dir, entry)) {
        std.log.err("download finished but required files are missing in {s}", .{dest_dir});
        return error.IncompleteInstall;
    }
    std.debug.print("installed {s} at {s}\n", .{ entry.alias, dest_dir });
    if (entry.needs_packed_experts) {
        std.debug.print(
            "note: this model also needs packed experts before serving:\n" ++
                "  python3 scripts/repack_experts.py {s}\n",
            .{dest_dir},
        );
    }
    if (entry.companion) |comp| {
        const comp_entry = catalog.find(comp).?;
        const comp_path = try catalog.installedPath(allocator, io, comp_entry);
        if (comp_path) |p| {
            allocator.free(p);
        } else {
            std.debug.print("note: companion model not installed: dmlx pull {s}\n", .{comp});
        }
    }
}

/// Skip directories and repo metadata that serve no purpose at runtime.
fn wantFile(fe: TreeEntry) bool {
    if (!std.mem.eql(u8, fe.type, "file")) return false;
    const base = std.fs.path.basename(fe.path);
    if (std.mem.eql(u8, base, ".gitattributes")) return false;
    if (std.mem.startsWith(u8, fe.path, ".cache/")) return false;
    return true;
}

fn curlGet(allocator: std.mem.Allocator, io: std.Io, url: []const u8, limit: usize) ![]u8 {
    const result = try std.process.run(allocator, io, .{
        .argv = &.{ "curl", "-fsSL", "--retry", "3", url },
        .stdout_limit = .limited(limit),
    });
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) {
            std.log.err("curl GET failed ({d}): {s}", .{ code, result.stderr });
            return error.DownloadFailed;
        },
        else => return error.DownloadFailed,
    }
    return result.stdout;
}

fn downloadFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    entry: *const catalog.Entry,
    dest_dir: []const u8,
    fe: TreeEntry,
    index: usize,
    total: usize,
    force: bool,
) !void {
    const dest_path = try std.fs.path.join(allocator, &.{ dest_dir, fe.path });
    defer allocator.free(dest_path);
    if (std.fs.path.dirname(fe.path)) |parent| {
        const parent_path = try std.fs.path.join(allocator, &.{ dest_dir, parent });
        defer allocator.free(parent_path);
        try std.Io.Dir.cwd().createDirPath(io, parent_path);
    }

    // Resume / skip: an existing file with a matching LFS digest is kept.
    if (!force) {
        if (sha256FileMatches(io, dest_path, fe)) {
            std.debug.print("[{d}/{d}] skip (verified) {s}\n", .{ index, total, fe.path });
            return;
        }
        if (fe.lfs == null and fe.size > 0) {
            const stat = std.Io.Dir.cwd().statFile(io, dest_path, .{}) catch null;
            if (stat) |s| {
                if (s.size == fe.size) {
                    std.debug.print("[{d}/{d}] skip (size match) {s}\n", .{ index, total, fe.path });
                    return;
                }
            }
        }
    }

    const url = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}/resolve/main/{s}",
        .{ HF_RESOLVE, entry.hf_repo, fe.path },
    );
    defer allocator.free(url);

    const part_path = try std.fmt.allocPrint(allocator, "{s}.part", .{dest_path});
    defer allocator.free(part_path);

    std.debug.print(
        "[{d}/{d}] {s} ({d:.2} GB)\n",
        .{ index, total, fe.path, @as(f64, @floatFromInt(fe.size)) / (1 << 30) },
    );
    // curl -C - resumes a partial .part file; stderr stays attached so the
    // progress meter reaches the user's terminal.
    var child = try std.process.spawn(io, .{
        .argv = &.{ "curl", "-fL", "--retry", "3", "-C", "-", "-o", part_path, url },
    });
    const term = try child.wait(io);
    switch (term) {
        .exited => |code| if (code != 0) {
            std.log.err("download failed ({d}): {s}", .{ code, fe.path });
            return error.DownloadFailed;
        },
        else => return error.DownloadFailed,
    }

    if (fe.lfs != null) {
        std.debug.print("  verifying sha256…\n", .{});
        if (!sha256FileMatches(io, part_path, fe)) {
            std.log.err("sha256 mismatch: {s} (kept as {s} for resume)", .{ fe.path, part_path });
            return error.ChecksumMismatch;
        }
    }
    try std.Io.Dir.renameAbsolute(part_path, dest_path, io);
}

fn sha256FileMatches(io: std.Io, path: []const u8, fe: TreeEntry) bool {
    const lfs = fe.lfs orelse return false;
    const expect_hex = lfs.oid orelse return false;
    const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
    defer f.close(io);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [1 << 20]u8 = undefined;
    while (true) {
        const n = f.readStreaming(io, &.{buf[0..]}) catch |e| switch (e) {
            error.EndOfStream => break,
            else => return false,
        };
        if (n == 0) break;
        hasher.update(buf[0..n]);
    }
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    const got_hex = std.fmt.bytesToHex(digest, .lower);
    return std.ascii.eqlIgnoreCase(&got_hex, expect_hex);
}

fn printAvailable() void {
    std.debug.print("\navailable models:\n", .{});
    for (&catalog.entries) |*e| {
        std.debug.print("  {s:<24} {s} (~{d:.0} GB, needs {d} GB RAM)\n", .{
            e.alias, e.blurb, e.approx_size_gb, e.min_ram_gb,
        });
    }
}
