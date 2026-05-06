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
    // Run: pkg-config --variable=prefix mlxc
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
        // 1. Explicit build option takes priority
        if (b.option([]const u8, "mlx_prefix", "Path to mlx-c installation prefix")) |p| break :blk p;

        // 2. Environment variable
        if (b.graph.environ_map.get("MLX_C_PREFIX")) |p| break :blk p;

        // 3. pkg-config discovery
        if (pkgConfigMlxPrefix(b)) |p| break :blk p;

        // 4. Fallback to /opt/homebrew with a warning
        std.log.warn("mlx-c not found via -Dmlx_prefix, MLX_C_PREFIX, or pkg-config; falling back to /opt/homebrew", .{});
        break :blk "/opt/homebrew";
    };

    const zig_regex = b.dependency("zig_regex", .{
        .target = target,
        .optimize = optimize,
    });

    // --- Library ---
    const lib = b.addLibrary(.{
        .name = "dmlx",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .linkage = .static,
    });
    lib.root_module.addImport("regex", zig_regex.module("regex"));
    configureMlxModule(b, lib.root_module, is_macos, mlx_prefix);
    b.installArtifact(lib);

    // --- Tests ---
    const lib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    lib_tests.root_module.addImport("regex", zig_regex.module("regex"));
    configureMlxModule(b, lib_tests.root_module, is_macos, mlx_prefix);
    const run_lib_tests = b.addRunArtifact(lib_tests);
    if (b.args) |args| {
        run_lib_tests.addArgs(args);
    }
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_lib_tests.step);

    // --- Example ---
    const example = b.addExecutable(.{
        .name = "example",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/example.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    example.root_module.addImport("regex", zig_regex.module("regex"));
    configureMlxModule(b, example.root_module, is_macos, mlx_prefix);
    b.installArtifact(example);

    // --- CLI ---
    const cli = b.addExecutable(.{
        .name = "dmlx",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    cli.root_module.addImport("regex", zig_regex.module("regex"));
    configureMlxModule(b, cli.root_module, is_macos, mlx_prefix);
    b.installArtifact(cli);
}
