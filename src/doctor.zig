/// `dmlx doctor` — hardware and runtime health check.
///
/// Reports PASS/WARN/FAIL per check and exits nonzero on any FAIL so the
/// output can be pasted straight into a bug report.
const std = @import("std");
const builtin = @import("builtin");
const catalog = @import("catalog.zig");
// Not re-exported by the mlx-zig bindings; declared against libmlxc directly.
extern "c" fn mlx_metal_is_available(res: *bool) c_int;
extern "c" fn _NSGetExecutablePath(buf: [*]u8, bufsize: *u32) c_int;

const Status = enum { pass, warn, fail, info };

var any_fail: bool = false;
var any_warn: bool = false;

fn report(status: Status, comptime fmt: []const u8, args: anytype) void {
    const tag = switch (status) {
        .pass => "PASS",
        .warn => "WARN",
        .fail => "FAIL",
        .info => "INFO",
    };
    switch (status) {
        .fail => any_fail = true,
        .warn => any_warn = true,
        else => {},
    }
    std.debug.print("  {s}  ", .{tag});
    std.debug.print(fmt, args);
    std.debug.print("\n", .{});
}

pub fn run(allocator: std.mem.Allocator, io: std.Io) !u8 {
    any_fail = false;
    any_warn = false;
    std.debug.print("\n== platform ==\n", .{});

    if (builtin.os.tag != .macos) {
        report(.fail, "dmlx requires macOS on Apple Silicon", .{});
        return 1;
    }
    if (builtin.cpu.arch != .aarch64) {
        report(.fail, "dmlx requires Apple Silicon (arm64); this is {s}", .{@tagName(builtin.cpu.arch)});
        return 1;
    }
    report(.pass, "architecture: arm64", .{});

    if (try capture(allocator, io, &.{ "sw_vers", "-productVersion" })) |v| {
        defer allocator.free(v);
        report(.pass, "macOS {s}", .{v});
    }

    var total_ram_gb: u64 = 0;
    if (try capture(allocator, io, &.{ "sysctl", "-n", "machdep.cpu.brand_string" })) |chip| {
        defer allocator.free(chip);
        report(.pass, "chip: {s}", .{chip});
    } else {
        report(.warn, "could not read chip name (sysctl)", .{});
    }
    if (try capture(allocator, io, &.{ "sysctl", "-n", "hw.memsize" })) |mem| {
        defer allocator.free(mem);
        const bytes = std.fmt.parseInt(u64, mem, 10) catch 0;
        total_ram_gb = bytes / (1 << 30);
        if (total_ram_gb >= 48) {
            report(.pass, "unified memory: {d} GB (all flagship models supported)", .{total_ram_gb});
        } else if (total_ram_gb >= 32) {
            report(.pass, "unified memory: {d} GB (Qwen3.8 ok; DeepSeek V4 needs 48 GB)", .{total_ram_gb});
        } else if (total_ram_gb > 0) {
            report(.warn, "unified memory: {d} GB — below the 32 GB recommended for flagship models", .{total_ram_gb});
        }
    }

    if (try freeMemoryGB(allocator, io)) |free_gb| {
        if (free_gb < 8) {
            report(.warn, "free memory now: {d:.1} GB — loading a model may exhaust memory", .{free_gb});
        } else {
            report(.pass, "free memory now: {d:.1} GB", .{free_gb});
        }
    }

    std.debug.print("\n== gpu ==\n", .{});
    var metal_ok: bool = false;
    if (mlx_metal_is_available(&metal_ok) == 0 and metal_ok) {
        report(.pass, "Metal device available (mlx)", .{});
    } else {
        report(.fail, "Metal is not available — MLX GPU path cannot run", .{});
    }

    std.debug.print("\n== runtime ==\n", .{});
    var exe_buf: [4096]u8 = undefined;
    var exe_size: u32 = exe_buf.len;
    if (_NSGetExecutablePath(&exe_buf, &exe_size) == 0) {
        const p = std.mem.sliceTo(exe_buf[0..], 0);
        const exe_dir = std.fs.path.dirname(p) orelse ".";
        const bundled = try std.fs.path.join(allocator, &.{ exe_dir, "..", "lib", "libmlxc.dylib" });
        defer allocator.free(bundled);
        if (std.Io.Dir.accessAbsolute(io, bundled, .{})) {
            report(.pass, "bundled mlx runtime: {s}", .{bundled});
            const metallib = try std.fs.path.join(allocator, &.{ exe_dir, "..", "lib", "mlx.metallib" });
            defer allocator.free(metallib);
            if (std.Io.Dir.accessAbsolute(io, metallib, .{})) {
                report(.pass, "bundled mlx.metallib present", .{});
            } else |_| {
                report(.fail, "mlx.metallib missing next to bundled libmlx — Metal kernels will fail", .{});
            }
        } else |_| {
            report(.info, "development layout (system mlx-c via pkg-config)", .{});
        }
    }

    std.debug.print("\n== models ==\n", .{});
    var installed_count: usize = 0;
    for (&catalog.entries) |*e| {
        const path = (try catalog.installedPath(allocator, io, e)) orelse {
            report(.info, "{s}: not installed (dmlx pull {s})", .{ e.alias, e.alias });
            continue;
        };
        defer allocator.free(path);
        installed_count += 1;
        if (!catalog.isCompleteInstall(io, path, e)) {
            report(.fail, "{s}: incomplete install at {s} — re-run dmlx pull {s}", .{ e.alias, path, e.alias });
            continue;
        }
        if (e.needs_packed_experts) {
            const packed_dir = try std.fs.path.join(allocator, &.{ path, "packed_experts" });
            defer allocator.free(packed_dir);
            if (std.Io.Dir.accessAbsolute(io, packed_dir, .{})) {
                report(.pass, "{s}: installed, packed experts ready", .{e.alias});
            } else |_| {
                report(.warn, "{s}: installed, but packed_experts missing — run: python3 scripts/repack_experts.py {s}", .{ e.alias, path });
            }
        } else {
            report(.pass, "{s}: installed at {s}", .{ e.alias, path });
        }
        if (e.companion) |comp_alias| {
            const comp = catalog.find(comp_alias).?;
            if (try catalog.installedPath(allocator, io, comp)) |cp| {
                allocator.free(cp);
            } else {
                report(.warn, "  companion {s} not installed — run: dmlx pull {s}", .{ comp_alias, comp_alias });
            }
        }
    }
    if (installed_count == 0) {
        report(.info, "no models installed yet — start with: dmlx pull qwen3.8-27b-4bit", .{});
    }

    std.debug.print("\n== recommendation ==\n", .{});
    if (total_ram_gb >= 48) {
        report(.info, "this machine can run every flagship path (Qwen3.8 + MTP, DeepSeek V4 native)", .{});
    } else if (total_ram_gb >= 32) {
        report(.info, "recommended: dmlx pull qwen3.8-27b-4bit && dmlx pull qwen3.8-27b-mtp-4bit", .{});
    }

    std.debug.print("\n", .{});
    if (any_fail) {
        std.debug.print("doctor: FAIL (see above)\n", .{});
        return 1;
    } else if (any_warn) {
        std.debug.print("doctor: PASS with warnings\n", .{});
    } else {
        std.debug.print("doctor: PASS\n", .{});
    }
    return 0;
}

/// Run a command and return trimmed stdout, or null on failure.
fn capture(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !?[]u8 {
    const result = std.process.run(allocator, io, .{
        .argv = argv,
        .stdout_limit = .limited(1 << 16),
    }) catch return null;
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) {
            allocator.free(result.stdout);
            return null;
        },
        else => {
            allocator.free(result.stdout);
            return null;
        },
    }
    const trimmed = std.mem.trim(u8, result.stdout, " \n\r\t");
    if (trimmed.len == result.stdout.len) return result.stdout;
    const owned = try allocator.dupe(u8, trimmed);
    allocator.free(result.stdout);
    return owned;
}

/// Free memory estimate from vm_stat (free + speculative pages).
fn freeMemoryGB(allocator: std.mem.Allocator, io: std.Io) !?f64 {
    const out = (try capture(allocator, io, &.{"vm_stat"})) orelse return null;
    defer allocator.free(out);
    var page_size: u64 = 16384;
    var pages: u64 = 0;
    var it = std.mem.splitScalar(u8, out, '\n');
    while (it.next()) |line| {
        if (std.mem.indexOf(u8, line, "page size of ") != null) {
            var num_buf: [16]u8 = undefined;
            var n: usize = 0;
            for (line) |ch| {
                if (ch >= '0' and ch <= '9' and n < num_buf.len) {
                    num_buf[n] = ch;
                    n += 1;
                }
            }
            if (n > 0) page_size = std.fmt.parseInt(u64, num_buf[0..n], 10) catch page_size;
            continue;
        }
        const free = std.mem.startsWith(u8, line, "Pages free:");
        const spec = std.mem.startsWith(u8, line, "Pages speculative:");
        if (!free and !spec) continue;
        var num_buf: [24]u8 = undefined;
        var n: usize = 0;
        for (line) |ch| {
            if (ch >= '0' and ch <= '9' and n < num_buf.len) {
                num_buf[n] = ch;
                n += 1;
            }
        }
        if (n > 0) pages += std.fmt.parseInt(u64, num_buf[0..n], 10) catch 0;
    }
    return @as(f64, @floatFromInt(pages * page_size)) / (1 << 30);
}
