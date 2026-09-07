const std = @import("std");
const native_engine = @import("native_engine.zig");
const metal = @import("metal_infer/engine.zig");

const DIM = 4096;
const MHC_MULT = 4;
const HIDDEN_SIZE = DIM * MHC_MULT;
const MAX_ORACLE_CHUNK = 5;
const token_ids = [_]i32{ 17, 42, 314, 2718, 7, 99, 1234 };
const prefill_v2_token_ids = [_]i32{
    17, 42, 314, 2718, 7, 99, 1234, 55, 8080, 222,
};

const LogitsReport = extern struct {
    status: c_int,
    has_first_divergence: c_int,
    first_index: usize,
    first_absolute_error: f64,
    first_relative_error: f64,
    max_absolute_error: f64,
    max_relative_error: f64,
    tolerance_absolute: f64,
    tolerance_relative: f64,
    layer: c_int,
    token: c_int,
};

const PartitionCase = struct {
    name: []const u8,
    chunks: []const usize,
};

extern fn dmlx_batch_capture_create() ?*anyopaque;
extern fn dmlx_batch_capture_clear(context: ?*anyopaque) void;
extern fn dmlx_batch_capture_destroy(context: ?*anyopaque) void;
extern fn dmlx_batch_capture_record(
    context: ?*anyopaque,
    record: [*c]const metal.DebugCaptureRecord,
) void;
extern fn dmlx_batch_capture_compare(
    scalar_context: ?*anyopaque,
    batch_context: ?*anyopaque,
    start_pos: c_int,
    n_tokens: c_int,
) c_int;
extern fn dmlx_batch_logits_compare(
    reference: [*c]const f32,
    actual: [*c]const f32,
    count: usize,
    token: c_int,
    report: *LogitsReport,
) c_int;
extern fn dmlx_task44_host_wrapper_test(engine: *metal.Engine) c_int;

fn captureCallback(
    context: ?*anyopaque,
    record: [*c]const metal.DebugCaptureRecord,
) callconv(.c) void {
    dmlx_batch_capture_record(context, record);
}

fn computeFinalLogits(
    engine: *native_engine.NativeEngine,
    final_residual: []const f32,
    logits_out: []f32,
) !void {
    if (final_residual.len != HIDDEN_SIZE or logits_out.len != engine.config.vocab_size) {
        return error.InvalidLogitsShape;
    }
    var compressed: [DIM]f32 = undefined;
    metal.hyperHeadCompress(
        engine.weight_store.weights.hc_head_fn.ptr,
        engine.weight_store.weights.hc_head_base.ptr,
        engine.weight_store.weights.hc_head_scale.ptr,
        final_residual.ptr,
        &compressed,
    );
    try metal.getLogits(engine.engine, &compressed, logits_out.ptr);
}

fn captureScalar(
    engine: *native_engine.NativeEngine,
    captures: ?*anyopaque,
    ids: []const i32,
    logits_out: ?[]f32,
) !void {
    if (ids.len == 0) return error.EmptyOracleInput;
    engine.resetKv();
    dmlx_batch_capture_clear(captures);
    metal.setDebugCapture(engine.engine, captureCallback, captures);
    defer metal.setDebugCapture(engine.engine, null, null);

    var final_residual: [HIDDEN_SIZE]f32 = undefined;
    for (ids, 0..) |token_id, position| {
        metal.setTokenId(engine.engine, token_id);
        metal.embed(engine.engine, token_id, &final_residual);
        try metal.forward(engine.engine, final_residual[0..], @intCast(position));
    }
    if (logits_out) |logits| try computeFinalLogits(engine, &final_residual, logits);
}

fn validatePartition(chunks: []const usize, token_count: usize) !void {
    if (chunks.len == 0) return error.EmptyPartition;
    var total: usize = 0;
    for (chunks) |chunk| {
        if (chunk == 0 or chunk > MAX_ORACLE_CHUNK) return error.InvalidOracleChunk;
        total = std.math.add(usize, total, chunk) catch return error.InvalidOracleChunk;
    }
    if (total != token_count) return error.InvalidOraclePartition;
}

fn capturePartition(
    allocator: std.mem.Allocator,
    engine: *native_engine.NativeEngine,
    captures: ?*anyopaque,
    ids: []const i32,
    chunks: []const usize,
    logits_out: ?[]f32,
) !void {
    try validatePartition(chunks, ids.len);
    engine.resetKv();
    dmlx_batch_capture_clear(captures);
    metal.setDebugCapture(engine.engine, captureCallback, captures);
    defer metal.setDebugCapture(engine.engine, null, null);

    const hidden_batch = try allocator.alloc(f32, ids.len * HIDDEN_SIZE);
    defer allocator.free(hidden_batch);
    for (ids, 0..) |token_id, index| {
        metal.setTokenId(engine.engine, token_id);
        metal.embed(
            engine.engine,
            token_id,
            hidden_batch[index * HIDDEN_SIZE ..][0..HIDDEN_SIZE].ptr,
        );
    }

    var offset: usize = 0;
    for (chunks) |chunk| {
        const end = offset + chunk;
        try metal.forwardBatch(
            engine.engine,
            hidden_batch[offset * HIDDEN_SIZE .. end * HIDDEN_SIZE],
            chunk,
            @intCast(offset),
            ids[offset..end],
        );
        offset = end;
    }

    if (logits_out) |logits| {
        try computeFinalLogits(
            engine,
            hidden_batch[(ids.len - 1) * HIDDEN_SIZE ..][0..HIDDEN_SIZE],
            logits,
        );
    }
}

fn runLegacyCase(
    allocator: std.mem.Allocator,
    engine: *native_engine.NativeEngine,
    scalar: ?*anyopaque,
    batch: ?*anyopaque,
    ids: []const i32,
    chunks: []const usize,
) !void {
    try captureScalar(engine, scalar, ids, null);
    try capturePartition(allocator, engine, batch, ids, chunks, null);
    if (dmlx_batch_capture_compare(scalar, batch, 0, @intCast(ids.len)) != 0) {
        return error.BatchOracleDivergence;
    }
    std.debug.print("real batch oracle: PASS chunks=[", .{});
    for (chunks, 0..) |chunk, index| {
        if (index != 0) std.debug.print(",", .{});
        std.debug.print("{d}", .{chunk});
    }
    std.debug.print("] tokens={d}\n", .{ids.len});
}

fn argmax(logits: []const f32) usize {
    var best: usize = 0;
    for (logits[1..], 1..) |value, index| {
        if (value > logits[best]) best = index;
    }
    return best;
}

fn writeUsizeArray(writer: anytype, values: []const usize) !void {
    try writer.writeByte('[');
    for (values, 0..) |value, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.print("{d}", .{value});
    }
    try writer.writeByte(']');
}

fn writeI32Array(writer: anytype, values: []const i32) !void {
    try writer.writeByte('[');
    for (values, 0..) |value, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.print("{d}", .{value});
    }
    try writer.writeByte(']');
}

fn writeResult(
    io: std.Io,
    output_path: []const u8,
    logits: []const f32,
    cases: []const PartitionCase,
    reports: []const LogitsReport,
    argmaxes: []const usize,
) !void {
    const file = try std.Io.Dir.cwd().createFile(io, output_path, .{});
    defer file.close(io);
    var buffer: [64 * 1024]u8 = undefined;
    var file_writer = std.Io.File.Writer.init(file, io, &buffer);
    const writer = &file_writer.interface;

    try writer.writeAll("{\n  \"schema_version\": 1,\n");
    try writer.writeAll("  \"runner\": \"dmlx-native-batch-oracle\",\n");
    try writer.writeAll("  \"generation\": {\"method\": \"production embed/forward/hyper-head/get-logits before sampling\", \"max_generated_tokens\": 0},\n");
    try writer.writeAll("  \"input_ids\": ");
    try writeI32Array(writer, &token_ids);
    try writer.print(",\n  \"final_token_position\": {d},\n", .{token_ids.len - 1});
    try writer.print("  \"vocab_size\": {d},\n", .{logits.len});
    try writer.print(
        "  \"central_tolerance\": {{\"source\": \"batch_oracle.c:F32/final_logits\", \"absolute\": {d}, \"relative\": {d}}},\n",
        .{ reports[0].tolerance_absolute, reports[0].tolerance_relative },
    );
    try writer.print("  \"reference\": {{\"path\": \"scalar\", \"argmax\": {d}, \"final_logits\": [", .{argmax(logits)});
    for (logits, 0..) |value, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.print("{d}", .{value});
    }
    try writer.writeAll("]},\n  \"partitions\": [\n");
    for (cases, 0..) |case, index| {
        const report = reports[index];
        try writer.print("    {{\"name\": \"{s}\", \"chunks\": ", .{case.name});
        try writeUsizeArray(writer, case.chunks);
        try writer.print(
            ", \"argmax\": {d}, \"status\": \"ok\", \"final_token_position\": {d}, \"max_absolute_error\": {d}, \"max_relative_error\": {d}}}{s}\n",
            .{
                argmaxes[index],
                report.token,
                report.max_absolute_error,
                report.max_relative_error,
                if (index + 1 == cases.len) "" else ",",
            },
        );
    }
    try writer.writeAll("  ]\n}\n");
    try writer.flush();
}

fn runPartitionOracle(
    allocator: std.mem.Allocator,
    io: std.Io,
    engine: *native_engine.NativeEngine,
    scalar: ?*anyopaque,
    batch: ?*anyopaque,
    output_path: []const u8,
) !void {
    const chunks_ones = [_]usize{ 1, 1, 1, 1, 1, 1, 1 };
    const chunks_twos = [_]usize{ 2, 2, 2, 1 };
    const chunks_odd = [_]usize{ 3, 3, 1 };
    const chunks_max_tail = [_]usize{ 5, 2 };
    const cases = [_]PartitionCase{
        .{ .name = "scalar-sized", .chunks = &chunks_ones },
        .{ .name = "batch-2-tail", .chunks = &chunks_twos },
        .{ .name = "odd-3-tail", .chunks = &chunks_odd },
        .{ .name = "max-5-tail", .chunks = &chunks_max_tail },
    };

    const scalar_logits = try allocator.alloc(f32, engine.config.vocab_size);
    defer allocator.free(scalar_logits);
    const batch_logits = try allocator.alloc(f32, engine.config.vocab_size);
    defer allocator.free(batch_logits);

    try captureScalar(engine, scalar, &token_ids, scalar_logits);
    var scalar_report: LogitsReport = undefined;
    if (dmlx_batch_logits_compare(
        scalar_logits.ptr,
        scalar_logits.ptr,
        scalar_logits.len,
        @intCast(token_ids.len - 1),
        &scalar_report,
    ) != 0) return error.NonFiniteScalarLogits;

    var reports: [cases.len]LogitsReport = undefined;
    var argmaxes: [cases.len]usize = undefined;
    for (cases, 0..) |case, index| {
        try capturePartition(allocator, engine, batch, &token_ids, case.chunks, batch_logits);
        if (dmlx_batch_capture_compare(scalar, batch, 0, token_ids.len) != 0) {
            return error.BatchOracleDivergence;
        }
        if (dmlx_batch_logits_compare(
            scalar_logits.ptr,
            batch_logits.ptr,
            scalar_logits.len,
            @intCast(token_ids.len - 1),
            &reports[index],
        ) != 0) return error.FinalLogitsDivergence;
        argmaxes[index] = argmax(batch_logits);
        if (argmaxes[index] != argmax(scalar_logits)) return error.ArgmaxDivergence;

        std.debug.print("chunk partition oracle: PASS {s} chunks=[", .{case.name});
        for (case.chunks, 0..) |chunk, chunk_index| {
            if (chunk_index != 0) std.debug.print(",", .{});
            std.debug.print("{d}", .{chunk});
        }
        std.debug.print("] argmax={d} max_abs={d:.6} max_rel={d:.6}\n", .{
            argmaxes[index],
            reports[index].max_absolute_error,
            reports[index].max_relative_error,
        });
    }

    // Every report must expose the same centralized tolerance and terminal
    // location selected by C.
    for (reports) |report| {
        if (report.tolerance_absolute != scalar_report.tolerance_absolute or
            report.tolerance_relative != scalar_report.tolerance_relative)
        {
            return error.InconsistentCentralTolerance;
        }
        if (report.layer != scalar_report.layer or
            report.token != @as(c_int, @intCast(token_ids.len - 1)))
        {
            return error.InvalidFinalLogitsLocation;
        }
    }
    try writeResult(io, output_path, scalar_logits, &cases, &reports, &argmaxes);
    std.debug.print("chunk partition oracle: wrote schema-v1 result {s}\n", .{output_path});
}

const OracleArgs = struct {
    output_path: ?[]const u8 = null,
    production_v2: bool = false,
};

fn parseArguments(args: []const []const u8) !OracleArgs {
    var parsed: OracleArgs = .{};
    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--output")) {
            if (parsed.output_path != null) return error.DuplicateOutputArgument;
            index += 1;
            if (index >= args.len) return error.MissingOutputPath;
            parsed.output_path = args[index];
        } else if (std.mem.eql(u8, arg, "--production-v2")) {
            if (parsed.production_v2) return error.DuplicateProductionV2Argument;
            parsed.production_v2 = true;
        } else {
            std.log.err("unknown batch oracle argument: {s}", .{arg});
            return error.UnknownArgument;
        }
    }
    if (parsed.production_v2 and parsed.output_path != null) {
        return error.IncompatibleOracleArguments;
    }
    return parsed;
}

fn exactOneEnv(name: [*:0]const u8) bool {
    const value = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

fn productionV2EnabledFromEnv() bool {
    return exactOneEnv("NATIVE_BATCH_PREFILL") and
        exactOneEnv("NATIVE_BATCH_MLA") and
        exactOneEnv("NATIVE_EXPERT_BUCKETS") and
        exactOneEnv("NATIVE_BATCH_EXPERT_KERNELS");
}

fn captureProductionV2Partition(
    allocator: std.mem.Allocator,
    engine: *native_engine.NativeEngine,
    captures: ?*anyopaque,
    ids: []const i32,
    chunks: []const usize,
    logits_out: []f32,
) !void {
    if (ids.len == 0 or chunks.len == 0) return error.EmptyProductionV2Input;
    var total: usize = 0;
    for (chunks) |chunk| {
        if (chunk == 0 or chunk > 8) return error.InvalidProductionV2Chunk;
        total = std.math.add(usize, total, chunk) catch
            return error.InvalidProductionV2Partition;
    }
    if (total != ids.len) return error.InvalidProductionV2Partition;

    engine.resetKv();
    if (captures != null) {
        dmlx_batch_capture_clear(captures);
        metal.setDebugCapture(engine.engine, captureCallback, captures);
    }
    defer if (captures != null) metal.setDebugCapture(engine.engine, null, null);
    const hidden_batch = try allocator.alloc(f32, ids.len * HIDDEN_SIZE);
    defer allocator.free(hidden_batch);
    for (ids, 0..) |token_id, index| {
        metal.setTokenId(engine.engine, token_id);
        metal.embed(
            engine.engine,
            token_id,
            hidden_batch[index * HIDDEN_SIZE ..][0..HIDDEN_SIZE].ptr,
        );
    }

    const before = metal.batchPathCounters(engine.engine);
    var offset: usize = 0;
    for (chunks) |chunk| {
        const end = offset + chunk;
        try metal.forwardPlannedBatch(
            engine.engine,
            hidden_batch[offset * HIDDEN_SIZE .. end * HIDDEN_SIZE],
            chunk,
            @intCast(offset),
            ids[offset..end],
        );
        std.debug.print(
            "production-v2 oracle: executed start={d} B={d}\n",
            .{ offset, chunk },
        );
        offset = end;
    }
    const after = metal.batchPathCounters(engine.engine);
    const expected_batched_delta: u64 = 43 * chunks.len;
    if (after.scalar_layer_calls != before.scalar_layer_calls or
        after.batched_moe_layers != before.batched_moe_layers + expected_batched_delta)
    {
        std.log.err(
            "production-v2 counter proof failed: scalar {d}->{d} batched {d}->{d} expected_batched_delta={d}",
            .{
                before.scalar_layer_calls,
                after.scalar_layer_calls,
                before.batched_moe_layers,
                after.batched_moe_layers,
                expected_batched_delta,
            },
        );
        return error.ProductionV2PathCounterFailure;
    }

    try computeFinalLogits(
        engine,
        hidden_batch[(ids.len - 1) * HIDDEN_SIZE ..][0..HIDDEN_SIZE],
        logits_out,
    );
    std.debug.print(
        "production-v2 counter proof: PASS chunks={d} scalar_delta=0 batched_moe_delta={d}\n",
        .{ chunks.len, expected_batched_delta },
    );
}

fn assertProductionV2Logits(
    reference: []const f32,
    actual: []const f32,
    final_position: usize,
    label: []const u8,
) !void {
    var report: LogitsReport = undefined;
    if (dmlx_batch_logits_compare(
        reference.ptr,
        actual.ptr,
        reference.len,
        @intCast(final_position),
        &report,
    ) != 0) return error.ProductionV2FinalLogitsDivergence;
    if (argmax(reference) != argmax(actual)) return error.ProductionV2ArgmaxDivergence;
    std.debug.print(
        "production-v2 logits: PASS {s} argmax={d} max_abs={d:.6} max_rel={d:.6}\n",
        .{ label, argmax(actual), report.max_absolute_error, report.max_relative_error },
    );
}

fn runProductionV2Oracle(
    allocator: std.mem.Allocator,
    engine: *native_engine.NativeEngine,
    scalar: ?*anyopaque,
    batch: ?*anyopaque,
) !void {
    if (!productionV2EnabledFromEnv()) {
        return error.ProductionV2RequiresAllFourFlags;
    }
    const scalar_logits = try allocator.alloc(f32, engine.config.vocab_size);
    defer allocator.free(scalar_logits);
    const batch_logits = try allocator.alloc(f32, engine.config.vocab_size);
    defer allocator.free(batch_logits);

    const batch_sizes = [_]usize{ 1, 2, 3, 5, 8 };
    for (batch_sizes) |batch_size| {
        const ids = prefill_v2_token_ids[0..batch_size];
        try captureScalar(engine, scalar, ids, scalar_logits);
        const chunks = [_]usize{batch_size};
        try captureProductionV2Partition(
            allocator,
            engine,
            batch,
            ids,
            &chunks,
            batch_logits,
        );
        if (dmlx_batch_capture_compare(scalar, batch, 0, @intCast(batch_size)) != 0) {
            return error.ProductionV2CheckpointDivergence;
        }
        var label_buffer: [32]u8 = undefined;
        const label = try std.fmt.bufPrint(&label_buffer, "single-B={d}", .{batch_size});
        try assertProductionV2Logits(
            scalar_logits,
            batch_logits,
            batch_size - 1,
            label,
        );
    }

    const chunks_2_8 = [_]usize{ 2, 8 };
    const chunks_5_5 = [_]usize{ 5, 5 };
    const chunks_3_3_3_1 = [_]usize{ 3, 3, 3, 1 };
    const partitions = [_]PartitionCase{
        .{ .name = "partition-2-8", .chunks = &chunks_2_8 },
        .{ .name = "partition-5-5", .chunks = &chunks_5_5 },
        .{ .name = "partition-3-3-3-1", .chunks = &chunks_3_3_3_1 },
    };
    try captureScalar(
        engine,
        scalar,
        &prefill_v2_token_ids,
        scalar_logits,
    );
    for (partitions) |partition| {
        try captureProductionV2Partition(
            allocator,
            engine,
            null,
            &prefill_v2_token_ids,
            partition.chunks,
            batch_logits,
        );
        try assertProductionV2Logits(
            scalar_logits,
            batch_logits,
            prefill_v2_token_ids.len - 1,
            partition.name,
        );
    }
    std.debug.print(
        "production-v2 oracle: PASS B=1,2,3,5,8 partitions=[2,8]/[5,5]/[3,3,3,1] nonzero_starts=true\n",
        .{},
    );
}

fn runGpuBatchRoutingOracle(engine: *native_engine.NativeEngine) !void {
    const batch_sizes = [_]u32{ 1, 2, 3, 5 };
    var logits: [5 * metal.ROUTING_EXPERT_COUNT]f32 = undefined;
    var bias: [metal.ROUTING_EXPERT_COUNT]f32 = undefined;
    var cached: [metal.ROUTING_EXPERT_COUNT]u8 = undefined;
    for (0..metal.ROUTING_EXPERT_COUNT) |expert| {
        bias[expert] = @as(f32, @floatFromInt(@as(i32, @intCast(expert % 7)) - 3)) * 0.015625;
        cached[expert] = @intFromBool(expert % 3 != 0);
    }
    for (0..5) |token| {
        for (0..metal.ROUTING_EXPERT_COUNT) |expert| {
            logits[token * metal.ROUTING_EXPERT_COUNT + expert] =
                @as(f32, @floatFromInt((expert * 17 + token * 13) % 41)) * 0.125 - 2.0;
        }
    }

    for (batch_sizes) |tokens| {
        const bytes = try metal.expertBucketWorkspaceBytes(tokens);
        var gpu_arena: metal.NativeBatchWorkspace = .{};
        try metal.initBatchWorkspace(&gpu_arena, tokens, bytes, bytes);
        defer metal.deinitBatchWorkspace(&gpu_arena);
        var reference_arena: metal.NativeBatchWorkspace = .{};
        try metal.initBatchWorkspace(&reference_arena, tokens, bytes, bytes);
        defer metal.deinitBatchWorkspace(&reference_arena);
        var gpu_view: metal.NativeExpertBucketWorkspace = .{};
        var reference_view: metal.NativeExpertBucketWorkspace = .{};
        try metal.bindExpertBucketWorkspace(&gpu_arena, tokens, &gpu_view);
        try metal.bindExpertBucketWorkspace(&reference_arena, tokens, &reference_view);

        const score_count = @as(usize, tokens) * metal.ROUTING_EXPERT_COUNT;
        try metal.routeExpertBucketBatch(
            engine.engine,
            logits[0..score_count],
            &bias,
            &cached,
            0.25,
            &gpu_view,
        );
        try metal.routeExpertBucketBatchReference(
            logits[0..score_count],
            &bias,
            &cached,
            0.25,
            &reference_view,
        );
        if (gpu_view.pair_capacity != tokens * metal.ROUTING_ACTIVE_COUNT) {
            return error.InvalidBatchRoutePairCount;
        }
        for (0..gpu_view.pair_capacity) |index| {
            const actual = gpu_view.activation_pairs[index];
            const expected = reference_view.activation_pairs[index];
            if (actual.token_id != expected.token_id or
                actual.expert_id != expected.expert_id or
                actual.route_rank != expected.route_rank)
            {
                return error.BatchRouteIdDivergence;
            }
            if (!std.math.isFinite(actual.weight) or
                @abs(actual.weight - expected.weight) > 2e-6)
            {
                return error.BatchRouteWeightDivergence;
            }
        }
        std.debug.print("GPU batch routing oracle: PASS B={d} pairs={d}\n", .{
            tokens,
            gpu_view.pair_capacity,
        });
    }

    // Explicit all-equal row proves deterministic lower-ID tie-breaking on GPU.
    var tie_logits = [_]f32{0.0} ** metal.ROUTING_EXPERT_COUNT;
    const tie_bytes = try metal.expertBucketWorkspaceBytes(1);
    var tie_arena: metal.NativeBatchWorkspace = .{};
    try metal.initBatchWorkspace(&tie_arena, 1, tie_bytes, tie_bytes);
    defer metal.deinitBatchWorkspace(&tie_arena);
    var tie_view: metal.NativeExpertBucketWorkspace = .{};
    try metal.bindExpertBucketWorkspace(&tie_arena, 1, &tie_view);
    try metal.routeExpertBucketBatch(
        engine.engine,
        &tie_logits,
        null,
        null,
        0.0,
        &tie_view,
    );
    for (0..metal.ROUTING_ACTIVE_COUNT) |rank| {
        const pair = tie_view.activation_pairs[rank];
        if (pair.expert_id != rank or pair.route_rank != rank or
            @abs(pair.weight - 0.25) > 1e-6)
        {
            return error.BatchRouteTieBreakDivergence;
        }
    }
    std.debug.print("GPU batch routing oracle: PASS tie-breaking\n", .{});
}

fn task45EnabledFromEnv() bool {
    const buckets_enabled = if (std.c.getenv("NATIVE_EXPERT_BUCKETS")) |value|
        std.mem.eql(u8, std.mem.span(value), "1")
    else
        false;
    const kernels_enabled = if (std.c.getenv("NATIVE_BATCH_EXPERT_KERNELS")) |value|
        std.mem.eql(u8, std.mem.span(value), "1")
    else
        false;
    return buckets_enabled and kernels_enabled;
}

fn assertTask45PathCounters(engine: *native_engine.NativeEngine) !void {
    const counters = metal.batchPathCounters(engine.engine);
    if (task45EnabledFromEnv()) {
        if (counters.scalar_layer_calls != 0 or
            counters.batched_moe_layers == 0)
        {
            std.log.err(
                "Task 4.5 enabled-path proof failed: scalar_layer_calls={d} batched_moe_layers={d}",
                .{ counters.scalar_layer_calls, counters.batched_moe_layers },
            );
            return error.Task45PathCounterFailure;
        }
        std.debug.print(
            "Task 4.5 enabled path: PASS scalar_layer_calls=0 batched_moe_layers={d}\n",
            .{counters.batched_moe_layers},
        );
    } else {
        if (counters.scalar_layer_calls == 0 or counters.batched_moe_layers != 0) {
            std.log.err(
                "Task 4.5 disabled-path proof failed: scalar_layer_calls={d} batched_moe_layers={d}",
                .{ counters.scalar_layer_calls, counters.batched_moe_layers },
            );
            return error.Task45DisabledPathCounterFailure;
        }
        std.debug.print(
            "Task 4.5 disabled path: PASS scalar_layer_calls={d} batched_moe_layers=0\n",
            .{counters.scalar_layer_calls},
        );
    }
}

fn assertTask45OversizedFallback(
    allocator: std.mem.Allocator,
    engine: *native_engine.NativeEngine,
) !void {
    if (!task45EnabledFromEnv()) return;
    const TestApi = struct {
        extern fn dmlx_task45_test_override_max_prefill_chunk(
            target: *metal.Engine,
            capacity: u32,
        ) u32;
    };
    const previous_capacity = TestApi.dmlx_task45_test_override_max_prefill_chunk(
        engine.engine,
        16,
    );
    if (previous_capacity != 8) return error.Task45UnexpectedConfiguredCapacity;
    defer _ = TestApi.dmlx_task45_test_override_max_prefill_chunk(
        engine.engine,
        previous_capacity,
    );

    const oversized_ids = [_]i32{ 17, 42, 314, 2718, 7, 99, 1234, 55, 8080 };
    const hidden_batch = try allocator.alloc(f32, oversized_ids.len * HIDDEN_SIZE);
    defer allocator.free(hidden_batch);

    engine.resetKv();
    for (oversized_ids, 0..) |token_id, index| {
        metal.setTokenId(engine.engine, token_id);
        metal.embed(
            engine.engine,
            token_id,
            hidden_batch[index * HIDDEN_SIZE ..][0..HIDDEN_SIZE].ptr,
        );
    }

    const before = metal.batchPathCounters(engine.engine);
    try metal.forwardBatch(
        engine.engine,
        hidden_batch,
        oversized_ids.len,
        0,
        &oversized_ids,
    );
    const after = metal.batchPathCounters(engine.engine);
    const expected_scalar_delta: u64 = oversized_ids.len * 43;
    if (after.scalar_layer_calls != before.scalar_layer_calls + expected_scalar_delta or
        after.batched_moe_layers != before.batched_moe_layers)
    {
        std.log.err(
            "Task 4.5 >8 fallback proof failed: scalar {d}->{d} batched {d}->{d}",
            .{
                before.scalar_layer_calls,
                after.scalar_layer_calls,
                before.batched_moe_layers,
                after.batched_moe_layers,
            },
        );
        return error.Task45OversizedFallbackFailure;
    }
    std.debug.print(
        "Task 4.5 >8 hard fallback: PASS chunk=9 configured_capacity=16 scalar_layer_delta={d} batched_moe_delta=0\n",
        .{expected_scalar_delta},
    );
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const parsed_args = try parseArguments(args);
    if (std.c.getenv("NATIVE_SMELT_N")) |value| {
        if (!std.mem.eql(u8, std.mem.span(value), "20")) {
            std.log.err("real batch oracle requires NATIVE_SMELT_N=20", .{});
            return error.UnsafeSmeltConfiguration;
        }
    }

    const home_ptr = std.c.getenv("HOME") orelse return error.HomeNotSet;
    const home = std.mem.span(home_ptr);
    const model_path = try std.fs.path.join(
        allocator,
        &.{ home, "models", "DeepSeek-V4-Flash-4bit" },
    );
    defer allocator.free(model_path);
    const packed_dir = try std.fs.path.join(
        allocator,
        &.{ model_path, "packed_experts" },
    );
    defer allocator.free(packed_dir);

    std.debug.print(
        "real batch oracle: model={s} packed={s} NATIVE_SMELT_N=20 max_generated_tokens=0\n",
        .{ model_path, packed_dir },
    );
    var engine = try native_engine.NativeEngine.init(allocator, model_path, packed_dir);
    defer engine.deinit();
    try runGpuBatchRoutingOracle(&engine);

    const scalar = dmlx_batch_capture_create() orelse return error.OutOfMemory;
    defer dmlx_batch_capture_destroy(scalar);
    const batch = dmlx_batch_capture_create() orelse return error.OutOfMemory;
    defer dmlx_batch_capture_destroy(batch);

    if (parsed_args.production_v2) {
        try runProductionV2Oracle(allocator, &engine, scalar, batch);
        return;
    }

    if (parsed_args.output_path) |path| {
        try runPartitionOracle(allocator, init.io, &engine, scalar, batch, path);
        if (dmlx_task44_host_wrapper_test(engine.engine) != 0) {
            return error.Task44HostWrapperFailure;
        }
        try assertTask45PathCounters(&engine);
        try assertTask45OversizedFallback(allocator, &engine);
        return;
    }

    const chunk_1 = [_]usize{1};
    const chunk_2 = [_]usize{2};
    const chunk_3 = [_]usize{3};
    const chunk_5 = [_]usize{5};
    const chunk_tail = [_]usize{ 5, 2 };
    try runLegacyCase(allocator, &engine, scalar, batch, token_ids[0..1], &chunk_1);
    try runLegacyCase(allocator, &engine, scalar, batch, token_ids[0..2], &chunk_2);
    try runLegacyCase(allocator, &engine, scalar, batch, token_ids[0..3], &chunk_3);
    try runLegacyCase(allocator, &engine, scalar, batch, token_ids[0..5], &chunk_5);
    try runLegacyCase(allocator, &engine, scalar, batch, token_ids[0..7], &chunk_tail);
    if (dmlx_task44_host_wrapper_test(engine.engine) != 0) {
        return error.Task44HostWrapperFailure;
    }
    try assertTask45PathCounters(&engine);
    try assertTask45OversizedFallback(allocator, &engine);
    std.debug.print("real batch oracle: PASS (B=1,2,3,5,tail=5+2)\n", .{});
}
