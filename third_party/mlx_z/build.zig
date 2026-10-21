const std = @import("std");

fn configureMlxModule(b: *std.Build, module: *std.Build.Module, is_macos: bool, mlx_prefix: []const u8) void {
    module.linkSystemLibrary("mlxc", .{});

    const include_path = b.pathJoin(&.{ mlx_prefix, "include" });
    const lib_path = b.pathJoin(&.{ mlx_prefix, "lib" });
    module.addIncludePath(.{ .cwd_relative = include_path });
    module.addLibraryPath(.{ .cwd_relative = lib_path });

    if (is_macos) {
        module.linkFramework("Accelerate", .{});
        module.linkFramework("Metal", .{});
        module.linkFramework("Foundation", .{});
    }
}

/// Try to discover mlx-c prefix via pkg-config.
/// Returns the prefix path (e.g. "/opt/homebrew") or null if pkg-config fails.
fn pkgConfigMlxPrefix(b: *std.Build) ?[]const u8 {
    var code: u8 = undefined;
    const stdout = b.runAllowFail(
        &.{ "pkg-config", "--variable=prefix", "mlxc" },
        &code,
        .inherit,
    ) catch return null;

    const trimmed = std.mem.trim(u8, stdout, &std.ascii.whitespace);
    if (trimmed.len == 0) {
        b.allocator.free(stdout);
        return null;
    }
    return trimmed;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const is_macos = target.result.os.tag == .macos;

    // Resolve mlx-c prefix: -Dmlx_prefix > MLX_C_PREFIX env > pkg-config > /opt/homebrew fallback
    const mlx_prefix = blk: {
        if (b.option([]const u8, "mlx_prefix", "Path to mlx-c installation prefix")) |p| break :blk p;
        if (b.graph.environ_map.get("MLX_C_PREFIX")) |p| break :blk p;
        if (pkgConfigMlxPrefix(b)) |p| break :blk p;
        std.log.warn("mlx-c not found via -Dmlx_prefix, MLX_C_PREFIX, or pkg-config; falling back to /opt/homebrew", .{});
        break :blk "/opt/homebrew";
    };

    // Named module exposed to dependents (Zig 0.16 API)
    const mlx_z_module = b.addModule("mlx", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    configureMlxModule(b, mlx_z_module, is_macos, mlx_prefix);

    // --- Library (installed for standalone builds) ---
    const lib = b.addLibrary(.{
        .name = "mlx_z",
        .root_module = mlx_z_module,
        .linkage = .static,
    });
    b.installArtifact(lib);

    // --- Tests ---
    const lib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    configureMlxModule(b, lib_tests.root_module, is_macos, mlx_prefix);
    const run_lib_tests = b.addRunArtifact(lib_tests);
    if (b.args) |args| {
        run_lib_tests.addArgs(args);
    }
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_lib_tests.step);
}
