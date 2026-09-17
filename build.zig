const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    // Production defaults to compiled-in/runtime-disabled telemetry. The explicit
    // define prevents command-line and non-Zig builds from silently disagreeing.
    const gpu_timestamp_probe = b.option(
        bool,
        "gpu_timestamp_probe",
        "Compile the production GPU timestamp probe (default: true)",
    ) orelse true;
    const production_gpu_timestamp_probe_flag = if (gpu_timestamp_probe)
        "-DDMLX_GPU_TIMESTAMP_PROBE=1"
    else
        "-DDMLX_GPU_TIMESTAMP_PROBE=0";
    const decode_timeline_probe = b.option(
        bool,
        "decode_timeline_probe",
        "Compile the bounded decode timeline probe (default: false)",
    ) orelse false;
    const decode_timeline_probe_flag = if (decode_timeline_probe)
        "-DDMLX_DECODE_TIMELINE_PROBE=1"
    else
        "-DDMLX_DECODE_TIMELINE_PROBE=0";

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

    // --- native_loader load bench (run standalone) ---
    const load_bench = b.addExecutable(.{
        .name = "load-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/native_loader/load_bench.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_load_bench = b.addRunArtifact(load_bench);
    const run_load_bench_step = b.step("run-load-bench", "Run native loader weight benchmark");
    run_load_bench_step.dependOn(&run_load_bench.step);

    const native_loader_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/native_loader/test_loader.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_native_loader_test = b.addRunArtifact(native_loader_test);
    const native_loader_test_step = b.step("test-native-loader", "Test native safetensors loader");
    native_loader_test_step.dependOn(&run_native_loader_test.step);

    // --- Scalar/batch numerical oracle self-test (no model or Metal) ---
    const batch_oracle_test = b.addExecutable(.{
        .name = "test-batch-oracle",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    batch_oracle_test.root_module.addCSourceFile(.{
        .file = b.path("src/metal_infer/batch_oracle.c"),
        .flags = &.{ "-std=c11", "-DDMLX_BATCH_ORACLE_SELF_TEST" },
        .language = .c,
    });
    const run_batch_oracle_test = b.addRunArtifact(batch_oracle_test);

    const batch_capture_test = b.addExecutable(.{
        .name = "test-batch-capture",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    batch_capture_test.root_module.addCSourceFile(.{
        .file = b.path("src/metal_infer/batch_oracle.c"),
        .flags = &.{"-std=c11"},
        .language = .c,
    });
    batch_capture_test.root_module.addCSourceFile(.{
        .file = b.path("src/metal_infer/test_batch_oracle.c"),
        .flags = &.{ "-std=c11", "-DDMLX_BATCH_CAPTURE_SELF_TEST" },
        .language = .c,
    });
    batch_capture_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/engine.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .objective_c });
    batch_capture_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/routing_trace.c"), .flags = &.{"-std=c11"}, .language = .c });
    batch_capture_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mla_attention.m"), .flags = &.{"-fobjc-arc"}, .language = .objective_c });
    batch_capture_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mhc.c"), .flags = &.{}, .language = .c });
    batch_capture_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mach_compat.c"), .flags = &.{}, .language = .c });
    batch_capture_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_engine.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    batch_capture_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_attention.c"), .flags = &.{}, .language = .c });
    batch_capture_test.root_module.linkFramework("Metal", .{});
    batch_capture_test.root_module.linkFramework("Foundation", .{});
    batch_capture_test.root_module.linkFramework("Accelerate", .{});
    const run_batch_capture_test = b.addRunArtifact(batch_capture_test);

    // --- Privacy-safe routing trace writer self-test (no model or Metal) ---
    const routing_trace_test = b.addExecutable(.{
        .name = "test-routing-trace",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    routing_trace_test.root_module.addCSourceFile(.{
        .file = b.path("src/metal_infer/routing_trace.c"),
        .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror" },
        .language = .c,
    });
    routing_trace_test.root_module.addCSourceFile(.{
        .file = b.path("src/metal_infer/test_routing_trace.c"),
        .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror" },
        .language = .c,
    });
    const run_routing_trace_test = b.addRunArtifact(routing_trace_test);
    const routing_trace_test_step = b.step(
        "test-routing-trace",
        "Test bounded privacy-safe routing trace schema, ordering, and drops",
    );
    routing_trace_test_step.dependOn(&run_routing_trace_test.step);

    // --- Wave-2 bounded workspace and KV transaction self-test ---
    const mla_transaction_test = b.addExecutable(.{
        .name = "test-mla-transaction",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    mla_transaction_test.root_module.addCSourceFile(.{
        .file = b.path("src/metal_infer/test_mla_transaction.c"),
        .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror" },
        .language = .c,
    });
    mla_transaction_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/engine.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .objective_c });
    mla_transaction_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/routing_trace.c"), .flags = &.{"-std=c11"}, .language = .c });
    mla_transaction_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mla_attention.m"), .flags = &.{"-fobjc-arc"}, .language = .objective_c });
    mla_transaction_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mhc.c"), .flags = &.{}, .language = .c });
    mla_transaction_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mach_compat.c"), .flags = &.{}, .language = .c });
    mla_transaction_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_engine.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    mla_transaction_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_attention.c"), .flags = &.{}, .language = .c });
    mla_transaction_test.root_module.linkFramework("Metal", .{});
    mla_transaction_test.root_module.linkFramework("Foundation", .{});
    mla_transaction_test.root_module.linkFramework("Accelerate", .{});
    const run_mla_transaction_test = b.addRunArtifact(mla_transaction_test);
    const mla_transaction_test_step = b.step(
        "test-mla",
        "Test explicit batch config, bounded workspace, and KV transaction metadata",
    );
    mla_transaction_test_step.dependOn(&run_mla_transaction_test.step);

    // --- Q2 Layer I/O / GPU dependency partition tests (Task 2.1) ---
    const q2_partition_test = b.addExecutable(.{
        .name = "test-q2-layer-partition",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    q2_partition_test.root_module.addCSourceFile(.{
        .file = b.path("src/metal_infer/test_q2_layer_partition.c"),
        .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror", "-DDMLX_EXACT_Q4_IO_TESTING", "-DDMLX_EXACT_Q4_PRODUCER_TESTING" },
        .language = .c,
    });
    q2_partition_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/engine.c"), .flags = &.{ "-Wno-deprecated-declarations", "-DDMLX_EXACT_Q4_IO_TESTING", "-DDMLX_EXACT_Q4_PRODUCER_TESTING" }, .language = .objective_c });
    q2_partition_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/routing_trace.c"), .flags = &.{"-std=c11"}, .language = .c });
    q2_partition_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mla_attention.m"), .flags = &.{"-fobjc-arc"}, .language = .objective_c });
    q2_partition_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mhc.c"), .flags = &.{}, .language = .c });
    q2_partition_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mach_compat.c"), .flags = &.{}, .language = .c });
    q2_partition_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_engine.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    q2_partition_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_attention.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    q2_partition_test.root_module.linkFramework("Metal", .{});
    q2_partition_test.root_module.linkFramework("Foundation", .{});
    q2_partition_test.root_module.linkFramework("Accelerate", .{});
    const run_q2_partition_test = b.addRunArtifact(q2_partition_test);
    const q2_partition_test_step = b.step(
        "test-q2-layer-partition",
        "Test Q2 layer I/O/GPU dependency partition model and ordering invariants",
    );
    q2_partition_test_step.dependOn(&run_q2_partition_test.step);

    // --- Topology-preserving attention stage timestamp self-test ---
    const attention_stage_timestamp_test = b.addExecutable(.{
        .name = "test-attention-stage-timestamp",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    attention_stage_timestamp_test.root_module.addCSourceFile(.{
        .file = b.path("src/metal_infer/test_attention_stage_timestamp.c"),
        .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror", "-DDMLX_EXACT_Q4_PRODUCER_TESTING" },
        .language = .c,
    });
    attention_stage_timestamp_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/engine.c"), .flags = &.{ "-Wno-deprecated-declarations", "-DDMLX_EXACT_Q4_PRODUCER_TESTING" }, .language = .objective_c });
    attention_stage_timestamp_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/routing_trace.c"), .flags = &.{"-std=c11"}, .language = .c });
    attention_stage_timestamp_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mla_attention.m"), .flags = &.{"-fobjc-arc"}, .language = .objective_c });
    attention_stage_timestamp_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mhc.c"), .flags = &.{}, .language = .c });
    attention_stage_timestamp_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mach_compat.c"), .flags = &.{}, .language = .c });
    attention_stage_timestamp_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_engine.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    attention_stage_timestamp_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_attention.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    attention_stage_timestamp_test.root_module.linkFramework("Metal", .{});
    attention_stage_timestamp_test.root_module.linkFramework("Foundation", .{});
    attention_stage_timestamp_test.root_module.linkFramework("Accelerate", .{});
    const run_attention_stage_timestamp_test = b.addRunArtifact(attention_stage_timestamp_test);
    const attention_stage_timestamp_test_step = b.step(
        "test-attention-stage-timestamp",
        "Test attention stage timestamp mapping, validation, reset, and overflow",
    );
    attention_stage_timestamp_test_step.dependOn(&run_attention_stage_timestamp_test.step);

    // --- Exact-Q4 bounded producer storage/publication self-test ---
    // The disabled-overhead gate is a paired timing measurement; it is only
    // meaningful at production optimization. Pin ReleaseFast so a default
    // (Debug) `zig build test` does not spuriously fail the 1% gate.
    const exact_q4_producer_test = b.addExecutable(.{
        .name = "test-exact-q4-producer",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        }),
    });
    exact_q4_producer_test.root_module.addCSourceFile(.{
        .file = b.path("src/metal_infer/test_exact_q4_producer.c"),
        .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror", "-DDMLX_EXACT_Q4_PRODUCER_TESTING", "-DDMLX_GPU_TIMESTAMP_PROBE=1" },
        .language = .c,
    });
    exact_q4_producer_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/engine.c"), .flags = &.{ "-Wno-deprecated-declarations", "-DDMLX_EXACT_Q4_PRODUCER_TESTING", "-DDMLX_GPU_TIMESTAMP_PROBE=1" }, .language = .objective_c });
    exact_q4_producer_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/routing_trace.c"), .flags = &.{"-std=c11"}, .language = .c });
    exact_q4_producer_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mla_attention.m"), .flags = &.{"-fobjc-arc"}, .language = .objective_c });
    exact_q4_producer_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mhc.c"), .flags = &.{}, .language = .c });
    exact_q4_producer_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mach_compat.c"), .flags = &.{}, .language = .c });
    exact_q4_producer_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_engine.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    exact_q4_producer_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_attention.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    exact_q4_producer_test.root_module.linkFramework("Metal", .{});
    exact_q4_producer_test.root_module.linkFramework("Foundation", .{});
    exact_q4_producer_test.root_module.linkFramework("Accelerate", .{});
    const run_exact_q4_producer_test = b.addRunArtifact(exact_q4_producer_test);
    const exact_q4_producer_test_step = b.step(
        "test-exact-q4-producer",
        "Test bounded exact-Q4 producer lifecycle and publication",
    );
    exact_q4_producer_test_step.dependOn(&run_exact_q4_producer_test.step);

    // --- GPU timestamp probe compiled-out ABI/lifecycle self-test ---
    const gpu_timestamp_probe_disabled_test = b.addExecutable(.{
        .name = "test-gpu-timestamp-probe-disabled",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    gpu_timestamp_probe_disabled_test.root_module.addCSourceFile(.{
        .file = b.path("src/metal_infer/test_gpu_timestamp_probe_disabled.c"),
        .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror", "-DDMLX_GPU_TIMESTAMP_PROBE=0" },
        .language = .c,
    });
    gpu_timestamp_probe_disabled_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/engine.c"), .flags = &.{ "-Wno-deprecated-declarations", "-DDMLX_GPU_TIMESTAMP_PROBE=0" }, .language = .objective_c });
    gpu_timestamp_probe_disabled_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/routing_trace.c"), .flags = &.{"-std=c11"}, .language = .c });
    gpu_timestamp_probe_disabled_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mla_attention.m"), .flags = &.{"-fobjc-arc"}, .language = .objective_c });
    gpu_timestamp_probe_disabled_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mhc.c"), .flags = &.{}, .language = .c });
    gpu_timestamp_probe_disabled_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mach_compat.c"), .flags = &.{}, .language = .c });
    gpu_timestamp_probe_disabled_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_engine.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    gpu_timestamp_probe_disabled_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_attention.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    gpu_timestamp_probe_disabled_test.root_module.linkFramework("Metal", .{});
    gpu_timestamp_probe_disabled_test.root_module.linkFramework("Foundation", .{});
    gpu_timestamp_probe_disabled_test.root_module.linkFramework("Accelerate", .{});
    const run_gpu_timestamp_probe_disabled_test = b.addRunArtifact(gpu_timestamp_probe_disabled_test);
    const gpu_timestamp_probe_disabled_test_step = b.step(
        "test-gpu-timestamp-probe-disabled",
        "Test compiled-out GPU timestamp public API stubs",
    );
    gpu_timestamp_probe_disabled_test_step.dependOn(&run_gpu_timestamp_probe_disabled_test.step);

    // --- Exact-Q4 scalar expert-source / per-read I/O self-test ---
    const exact_q4_io_test = b.addExecutable(.{
        .name = "test-exact-q4-io",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    exact_q4_io_test.root_module.addCSourceFile(.{
        .file = b.path("src/metal_infer/test_exact_q4_io.c"),
        .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror", "-DDMLX_EXACT_Q4_IO_TESTING", "-DDMLX_EXACT_Q4_PRODUCER_TESTING" },
        .language = .c,
    });
    exact_q4_io_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/engine.c"), .flags = &.{ "-Wno-deprecated-declarations", "-DDMLX_EXACT_Q4_IO_TESTING", "-DDMLX_EXACT_Q4_PRODUCER_TESTING" }, .language = .objective_c });
    exact_q4_io_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/routing_trace.c"), .flags = &.{"-std=c11"}, .language = .c });
    exact_q4_io_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mla_attention.m"), .flags = &.{"-fobjc-arc"}, .language = .objective_c });
    exact_q4_io_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mhc.c"), .flags = &.{}, .language = .c });
    exact_q4_io_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mach_compat.c"), .flags = &.{}, .language = .c });
    exact_q4_io_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_engine.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    exact_q4_io_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_attention.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    exact_q4_io_test.root_module.linkFramework("Metal", .{});
    exact_q4_io_test.root_module.linkFramework("Foundation", .{});
    exact_q4_io_test.root_module.linkFramework("Accelerate", .{});
    const run_exact_q4_io_test = b.addRunArtifact(exact_q4_io_test);
    const exact_q4_io_test_step = b.step(
        "test-exact-q4-io",
        "Test scalar exact-Q4 expert-source and per-read I/O lifecycle",
    );
    exact_q4_io_test_step.dependOn(&run_exact_q4_io_test.step);

    // --- Exact-Q4 transactional scalar stage and MLA sidecar self-test ---
    const exact_q4_stages_test = b.addExecutable(.{
        .name = "test-exact-q4-stages",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    exact_q4_stages_test.root_module.addCSourceFile(.{
        .file = b.path("src/metal_infer/test_exact_q4_stages.c"),
        .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror", "-DDMLX_EXACT_Q4_PRODUCER_TESTING" },
        .language = .c,
    });
    exact_q4_stages_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/engine.c"), .flags = &.{ "-Wno-deprecated-declarations", "-DDMLX_EXACT_Q4_PRODUCER_TESTING" }, .language = .objective_c });
    exact_q4_stages_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/routing_trace.c"), .flags = &.{"-std=c11"}, .language = .c });
    exact_q4_stages_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mla_attention.m"), .flags = &.{"-fobjc-arc"}, .language = .objective_c });
    exact_q4_stages_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mhc.c"), .flags = &.{}, .language = .c });
    exact_q4_stages_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mach_compat.c"), .flags = &.{}, .language = .c });
    exact_q4_stages_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_engine.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    exact_q4_stages_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_attention.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    exact_q4_stages_test.root_module.linkFramework("Metal", .{});
    exact_q4_stages_test.root_module.linkFramework("Foundation", .{});
    exact_q4_stages_test.root_module.linkFramework("Accelerate", .{});
    const run_exact_q4_stages_test = b.addRunArtifact(exact_q4_stages_test);
    const exact_q4_stages_test_step = b.step(
        "test-exact-q4-stages",
        "Test transactional scalar stages and MLA path sidecar",
    );
    exact_q4_stages_test_step.dependOn(&run_exact_q4_stages_test.step);

    // Model-backed in-process oracle: production loader + scalar and current
    // batch APIs, with dynamically owned checkpoint captures.
    const batch_model_test = b.addExecutable(.{
        .name = "test-batch-oracle-model",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/test_batch_oracle_model.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    batch_model_test.root_module.addImport("mlx", mlx_z_module);
    batch_model_test.root_module.addImport("regex", zig_regex.module("regex"));
    batch_model_test.root_module.addCSourceFile(.{
        .file = b.path("src/metal_infer/batch_oracle.c"),
        .flags = &.{"-std=c11"},
        .language = .c,
    });
    batch_model_test.root_module.addCSourceFile(.{
        .file = b.path("src/metal_infer/test_batch_oracle.c"),
        .flags = &.{"-std=c11"},
        .language = .c,
    });
    batch_model_test.root_module.addCSourceFile(.{ .file = b.path("src/models/moe_metal_wrapper.c"), .flags = &.{"-fobjc-arc"}, .language = .objective_c });
    batch_model_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/engine.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .objective_c });
    batch_model_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/routing_trace.c"), .flags = &.{"-std=c11"}, .language = .c });
    batch_model_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mla_attention.m"), .flags = &.{"-fobjc-arc"}, .language = .objective_c });
    batch_model_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mhc.c"), .flags = &.{}, .language = .c });
    batch_model_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mach_compat.c"), .flags = &.{}, .language = .c });
    batch_model_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_engine.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    batch_model_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_attention.c"), .flags = &.{}, .language = .c });
    batch_model_test.root_module.linkFramework("Metal", .{});
    batch_model_test.root_module.linkFramework("Foundation", .{});
    batch_model_test.root_module.linkFramework("Accelerate", .{});
    const run_batch_model_test = b.addRunArtifact(batch_model_test);

    // Reuse the same model-backed executable for Task 1.3. This separate run
    // step accepts `zig build ... -- --output <path>` without changing the
    // argument-free Task 1.2 model test.
    const run_batch_oracle_model = b.addRunArtifact(batch_model_test);
    if (b.args) |args| {
        run_batch_oracle_model.addArgs(args);
    }
    const run_batch_oracle_model_step = b.step(
        "run-batch-oracle-model",
        "Run final-logits and chunk-partition oracle against the real model",
    );
    run_batch_oracle_model_step.dependOn(&run_batch_oracle_model.step);

    const batch_oracle_test_step = b.step(
        "test-batch-oracle",
        "Test scalar/batch tensor comparison, capture ordering, and real model checkpoints",
    );
    batch_oracle_test_step.dependOn(&run_batch_oracle_test.step);
    batch_oracle_test_step.dependOn(&run_batch_capture_test.step);
    batch_oracle_test_step.dependOn(&run_batch_model_test.step);

    // --- GLM-5.3 oracle alignment harness (pure Zig + mlx-c, no Metal C) ---
    // Dumps embed / per-layer branch outputs / hidden_final / logits segments
    // of the 7-token fixture to artifacts/glm53_oracle/*.bin for
    // scripts/compare_glm53_oracle.py.
    const glm53_oracle_dump = b.addExecutable(.{
        .name = "glm53-oracle-dump",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/test_glm53_oracle_dump.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    glm53_oracle_dump.root_module.addImport("mlx", mlx_z_module);
    const run_glm53_oracle_dump = b.addRunArtifact(glm53_oracle_dump);
    if (b.args) |args| {
        run_glm53_oracle_dump.addArgs(args);
    }
    const run_glm53_oracle_dump_step = b.step(
        "run-glm53-oracle-dump",
        "Dump GLM-5.3 oracle segments (embed, per-layer kda/mla/moe outs, indexer topk, hidden_final, logits) to artifacts/glm53_oracle",
    );
    run_glm53_oracle_dump_step.dependOn(&run_glm53_oracle_dump.step);

    // --- Wave-5 bounded phase-aware chunk planner table tests (pure Zig) ---
    const native_chunk_planner_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/native_chunk_planner.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_native_chunk_planner_tests = b.addRunArtifact(native_chunk_planner_tests);
    const native_chunk_planner_test_step = b.step(
        "test-native-chunk-planner",
        "Test phase, capacity, memory, tail, and size-one chunk planning",
    );
    native_chunk_planner_test_step.dependOn(&run_native_chunk_planner_tests.step);

    // --- Bounded decode timeline producer focused tests (Task 1.1) ---
    const decode_timeline_test = b.addExecutable(.{
        .name = "test-decode-timeline-producer",
        .root_module = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true }),
    });
    decode_timeline_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/test_decode_timeline_producer.c"), .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror", "-DDMLX_DECODE_TIMELINE_PROBE=1" }, .language = .c });
    decode_timeline_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/engine.c"), .flags = &.{ "-Wno-deprecated-declarations", "-DDMLX_DECODE_TIMELINE_PROBE=1" }, .language = .objective_c });
    decode_timeline_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/routing_trace.c"), .flags = &.{"-std=c11"}, .language = .c });
    decode_timeline_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mla_attention.m"), .flags = &.{"-fobjc-arc"}, .language = .objective_c });
    decode_timeline_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mhc.c"), .flags = &.{}, .language = .c });
    decode_timeline_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mach_compat.c"), .flags = &.{}, .language = .c });
    decode_timeline_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_engine.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    decode_timeline_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_attention.c"), .flags = &.{}, .language = .c });
    decode_timeline_test.root_module.linkFramework("Metal", .{});
    decode_timeline_test.root_module.linkFramework("Foundation", .{});
    decode_timeline_test.root_module.linkFramework("Accelerate", .{});
    const run_decode_timeline_test = b.addRunArtifact(decode_timeline_test);
    const decode_timeline_test_step = b.step("test-decode-timeline", "Test bounded decode timeline producer and schema");
    decode_timeline_test_step.dependOn(&run_decode_timeline_test.step);

    const decode_timeline_disabled_test = b.addExecutable(.{
        .name = "test-decode-timeline-probe-disabled",
        .root_module = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true }),
    });
    decode_timeline_disabled_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/test_decode_timeline_probe_disabled.c"), .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror", "-DDMLX_DECODE_TIMELINE_PROBE=0" }, .language = .c });
    decode_timeline_disabled_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/engine.c"), .flags = &.{ "-Wno-deprecated-declarations", "-DDMLX_DECODE_TIMELINE_PROBE=0" }, .language = .objective_c });
    decode_timeline_disabled_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/routing_trace.c"), .flags = &.{"-std=c11"}, .language = .c });
    decode_timeline_disabled_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mla_attention.m"), .flags = &.{"-fobjc-arc"}, .language = .objective_c });
    decode_timeline_disabled_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mhc.c"), .flags = &.{}, .language = .c });
    decode_timeline_disabled_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mach_compat.c"), .flags = &.{}, .language = .c });
    decode_timeline_disabled_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_engine.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    decode_timeline_disabled_test.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_attention.c"), .flags = &.{}, .language = .c });
    decode_timeline_disabled_test.root_module.linkFramework("Metal", .{});
    decode_timeline_disabled_test.root_module.linkFramework("Foundation", .{});
    decode_timeline_disabled_test.root_module.linkFramework("Accelerate", .{});
    const run_decode_timeline_disabled_test = b.addRunArtifact(decode_timeline_disabled_test);
    const decode_timeline_disabled_test_step = b.step("test-decode-timeline-disabled", "Test compiled-out decode timeline stubs");
    decode_timeline_disabled_test_step.dependOn(&run_decode_timeline_disabled_test.step);

    // --- Library ---
    const lib = b.addLibrary(.{
        .name = "dmlx_lib",
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
    // Metal MoE C wrapper — needed for test compilation
    lib_tests.root_module.addCSourceFile(.{ .file = b.path("src/models/moe_metal_wrapper.c"), .flags = &.{"-fobjc-arc"}, .language = .objective_c });
    lib_tests.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/engine.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .objective_c });
    lib_tests.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/routing_trace.c"), .flags = &.{"-std=c11"}, .language = .c });
    lib_tests.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mla_attention.m"), .flags = &.{"-fobjc-arc"}, .language = .objective_c });
    lib_tests.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mhc.c"), .flags = &.{}, .language = .c });
    lib_tests.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/mach_compat.c"), .flags = &.{}, .language = .c });
    lib_tests.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_engine.c"), .flags = &.{"-Wno-deprecated-declarations"}, .language = .c });
    lib_tests.root_module.addCSourceFile(.{ .file = b.path("src/metal_infer/dspark_attention.c"), .flags = &.{}, .language = .c });
    lib_tests.root_module.linkFramework("Metal", .{});
    lib_tests.root_module.linkFramework("Foundation", .{});
    lib_tests.root_module.linkFramework("Accelerate", .{});
    const run_lib_tests = b.addRunArtifact(lib_tests);
    if (b.args) |args| {
        run_lib_tests.addArgs(args);
    }
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&run_native_chunk_planner_tests.step);
    test_step.dependOn(&run_exact_q4_producer_test.step);
    test_step.dependOn(&run_exact_q4_io_test.step);
    test_step.dependOn(&run_exact_q4_stages_test.step);
    test_step.dependOn(&run_decode_timeline_test.step);
    test_step.dependOn(&run_decode_timeline_disabled_test.step);

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

    const cli_args_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli_flags.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_cli_args_tests = b.addRunArtifact(cli_args_tests);
    const cli_args_test_step = b.step("test-cli-args", "Run focused CLI argument parser tests");
    cli_args_test_step.dependOn(&run_cli_args_tests.step);

    // Metal MoE + Metal inference engine
    inline for (.{ lib, example, cli }) |target_step| {
        target_step.root_module.addCSourceFile(.{
            .file = b.path("src/models/moe_metal_wrapper.c"),
            .flags = &.{"-fobjc-arc"},
            .language = .objective_c,
        });
        target_step.root_module.addCSourceFile(.{
            .file = b.path("src/metal_infer/engine.c"),
            .flags = &.{ "-Wno-deprecated-declarations", production_gpu_timestamp_probe_flag, decode_timeline_probe_flag },
            .language = .objective_c,
        });
        target_step.root_module.addCSourceFile(.{
            .file = b.path("src/metal_infer/routing_trace.c"),
            .flags = &.{"-std=c11"},
            .language = .c,
        });
        target_step.root_module.addCSourceFile(.{
            .file = b.path("src/metal_infer/mla_attention.m"),
            .flags = &.{"-fobjc-arc"},
            .language = .objective_c,
        });
        target_step.root_module.addCSourceFile(.{
            .file = b.path("src/metal_infer/mhc.c"),
            .flags = &.{},
            .language = .c,
        });
        target_step.root_module.addCSourceFile(.{
            .file = b.path("src/metal_infer/mach_compat.c"),
            .flags = &.{},
            .language = .c,
        });
        target_step.root_module.addCSourceFile(.{
            .file = b.path("src/metal_infer/dspark_engine.c"),
            .flags = &.{"-Wno-deprecated-declarations"},
            .language = .c,
        });
        target_step.root_module.addCSourceFile(.{
            .file = b.path("src/metal_infer/dspark_attention.c"),
            .flags = &.{},
            .language = .c,
        });
        target_step.root_module.linkFramework("Metal", .{});
        target_step.root_module.linkFramework("Foundation", .{});
        target_step.root_module.linkFramework("Accelerate", .{});
    }
}
