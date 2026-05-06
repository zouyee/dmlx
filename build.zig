const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zig_regex = b.dependency("zig_regex", .{
        .target = target,
        .optimize = optimize,
    });

    // mlx-zig handles mlx-c discovery and linking
    const mlx_z_dep = b.dependency("mlx_z", .{
        .target = target,
        .optimize = optimize,
    });
    const mlx_z_module = mlx_z_dep.module("mlx");

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
    lib.root_module.addImport("mlx", mlx_z_module);
    lib.root_module.addImport("regex", zig_regex.module("regex"));
    b.installArtifact(lib);

    // --- Tests ---
    const lib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    lib_tests.root_module.addImport("mlx", mlx_z_module);
    lib_tests.root_module.addImport("regex", zig_regex.module("regex"));
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
    example.root_module.addImport("mlx", mlx_z_module);
    example.root_module.addImport("regex", zig_regex.module("regex"));
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
    cli.root_module.addImport("mlx", mlx_z_module);
    cli.root_module.addImport("regex", zig_regex.module("regex"));
    b.installArtifact(cli);
}
