/// MLX-free inference engine for DeepSeek-V4-Flash.
/// Uses native_loader + engine.c/Metal directly. No MLX runtime dependency.
///
/// Design notes:
/// - Expert weights are loaded on-demand via packed binary pread (engine.c).
///   This matches the existing flash-moe streaming pattern and avoids OOM.
/// - Backbone weights (embed, lm_head, attention, norms) are loaded via
///   native_loader and held in CPU memory. engine.c copies them to GPU
///   buffers on each forward. Future optimization: create persistent MTLBuffers
///   to eliminate per-forward copies.
const std = @import("std");
const native_loader = @import("native_loader/loader.zig");
const chunk_planner = @import("native_chunk_planner.zig");
const metal = @import("metal_infer/engine.zig");
const sampling_mod = @import("sampling.zig");
const dspark_mod = @import("dspark.zig");

const DIM = 4096;
const MHC_MULT = 4;
const N_LAYERS = 43;
// BOS token ID: 0 is the `<｜begin▁of▁sentence｜>` token for DeepSeek-V4-Flash.
// EOS token ID: read from the model's config.json (eos_token_id) at init;
// this constant is only the fallback default (DSV4: 1 = `<｜end▁of▁sentence｜>`).
const BOS_TOKEN: u32 = 0;
const EOS_TOKEN: u32 = 1;
const MAX_TELEMETRY_TOKENS = 256;
const DSPARK_MAX_BLOCK: usize = 8;
const DEFAULT_ROUTING_TRACE_CAPACITY: usize = 4096;
const MAX_ROUTING_TRACE_CAPACITY: usize = 262_144;
const EXACT_Q4_STAGES_PER_TOKEN: usize = N_LAYERS * 9 + 1;
const EXACT_Q4_READS_PER_TOKEN: usize = N_LAYERS * 6;
const EXACT_Q4_DEFAULT_MAX_PROMPT_TOKENS: usize = 256;
const EXACT_Q4_MAX_GENERATED_TOKENS: usize = 100;
const DECODE_TIMELINE_EVENTS_PER_TOKEN: usize = N_LAYERS * 7 + 3;
const DECODE_TIMELINE_MAX_TOKENS: usize = 512;

const DecodeTimelineCapture = struct {
    output_fd: c_int,
    events: []metal.DecodeTimelineEvent,
    fn deinit(self: *DecodeTimelineCapture, allocator: std.mem.Allocator) void {
        _ = std.c.close(self.output_fd);
        allocator.free(self.events);
    }
};

const ExactQ4TokenInterval = struct {
    token_ordinal: u32,
    begin_ns: u64,
    end_ns: u64,
    steady: bool,
};

fn exactQ4Write(fd: c_int, bytes: []const u8) !void {
    var offset: usize = 0;
    while (offset < bytes.len) {
        const count = std.c.write(fd, bytes[offset..].ptr, bytes.len - offset);
        if (count <= 0) return error.ExactQ4OutputWriteFailed;
        offset += @intCast(count);
    }
}

fn exactQ4Print(fd: c_int, comptime format: []const u8, args: anytype) !void {
    var buffer: [1024]u8 = undefined;
    const rendered = std.fmt.bufPrint(&buffer, format, args) catch
        return error.ExactQ4OutputRecordTooLarge;
    try exactQ4Write(fd, rendered);
}

const ExactQ4Capture = struct {
    output_fd: c_int,
    stages: []metal.ExactQ4StageEvent,
    reads: []metal.ExactQ4ExpertReadEvent,
    mla_paths: []metal.MlaPathEvent,
    tokens: []ExactQ4TokenInterval,

    fn deinit(self: *ExactQ4Capture, allocator: std.mem.Allocator) void {
        _ = std.c.close(self.output_fd);
        allocator.free(self.tokens);
        allocator.free(self.mla_paths);
        allocator.free(self.reads);
        allocator.free(self.stages);
    }
};

comptime {
    if (chunk_planner.hard_batch_token_limit != metal.BATCH_HARD_TOKEN_LIMIT or
        chunk_planner.active_experts_per_token != metal.BATCH_ACTIVE_EXPERTS or
        chunk_planner.routed_expert_count != metal.BATCH_ROUTED_EXPERTS or
        chunk_planner.exact_q4_expert_bytes != metal.BATCH_EXACT_Q4_EXPERT_BYTES or
        chunk_planner.workspace_fixed_bytes != metal.BATCH_WORKSPACE_FIXED_BYTES or
        chunk_planner.workspace_per_token_bytes != metal.BATCH_WORKSPACE_PER_TOKEN_BYTES)
    {
        @compileError("native chunk planner constants drifted from the C engine contract");
    }
}

fn fnv1aUpdate(hash: *u64, bytes: []const u8) void {
    for (bytes) |byte| {
        hash.* ^= byte;
        hash.* *%= 1_099_511_628_211;
    }
}

fn fingerprintFileContent(
    hash: *u64,
    allocator: std.mem.Allocator,
    directory: []const u8,
    name: []const u8,
    required: bool,
) !bool {
    const path = try std.fs.path.join(allocator, &.{ directory, name });
    defer allocator.free(path);
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);
    const fd = std.c.open(path_z.ptr, .{});
    if (fd < 0) {
        if (!required) return false;
        return error.FileNotFound;
    }
    defer _ = std.c.close(fd);

    fnv1aUpdate(hash, name);
    var buffer: [64 * 1024]u8 = undefined;
    while (true) {
        const count = std.c.read(fd, &buffer, buffer.len);
        if (count < 0) return error.ReadFailed;
        if (count == 0) break;
        fnv1aUpdate(hash, buffer[0..@intCast(count)]);
    }
    return true;
}

fn fingerprintFileMetadata(
    hash: *u64,
    allocator: std.mem.Allocator,
    directory: []const u8,
    name: []const u8,
) !void {
    const path = try std.fs.path.join(allocator, &.{ directory, name });
    defer allocator.free(path);
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);
    const fd = std.c.open(path_z.ptr, .{});
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);
    var stat: std.c.Stat = undefined;
    if (std.c.fstat(fd, &stat) != 0) return error.StatFailed;
    fnv1aUpdate(hash, name);
    fnv1aUpdate(hash, std.mem.asBytes(&stat.size));
}

const PACKED_MANIFEST_VERSION: i64 = 1;
const PACKED_N_EXPERTS: i64 = 256;
const PACKED_EXPERT_BYTES: i64 = @intCast(chunk_planner.exact_q4_expert_bytes);
const PACKED_LAYER_BYTES: i64 = PACKED_EXPERT_BYTES * PACKED_N_EXPERTS;
const PACKED_COMPONENT_KEYS = [_][]const u8{
    "ffn.switch_mlp.gate_proj.weight",
    "ffn.switch_mlp.gate_proj.scales",
    "ffn.switch_mlp.up_proj.weight",
    "ffn.switch_mlp.up_proj.scales",
    "ffn.switch_mlp.down_proj.weight",
    "ffn.switch_mlp.down_proj.scales",
};
const PACKED_COMPONENT_BYTES = [_]i64{
    4_194_304,
    262_144,
    4_194_304,
    262_144,
    4_194_304,
    262_144,
};

fn jsonIntegerEquals(value: ?std.json.Value, expected: i64) bool {
    const present = value orelse return false;
    return switch (present) {
        .integer => |actual| actual == expected,
        else => false,
    };
}

fn jsonStringEquals(value: ?std.json.Value, expected: []const u8) bool {
    const present = value orelse return false;
    return switch (present) {
        .string => |actual| std.mem.eql(u8, actual, expected),
        else => false,
    };
}

fn readPackedManifest(
    allocator: std.mem.Allocator,
    packed_dir: []const u8,
) ![]u8 {
    const path = try std.fs.path.join(allocator, &.{ packed_dir, "manifest.json" });
    defer allocator.free(path);
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);
    const fd = std.c.open(path_z.ptr, .{});
    if (fd < 0) return error.InvalidExpertPackedDir;
    defer _ = std.c.close(fd);

    var stat: std.c.Stat = undefined;
    if (std.c.fstat(fd, &stat) != 0 or stat.size <= 0 or
        stat.size > 1024 * 1024)
    {
        return error.InvalidExpertPackedDir;
    }
    const contents = try allocator.alloc(u8, @intCast(stat.size));
    errdefer allocator.free(contents);
    var offset: usize = 0;
    while (offset < contents.len) {
        const count = std.c.read(fd, contents.ptr + offset, contents.len - offset);
        if (count <= 0) return error.InvalidExpertPackedDir;
        offset += @intCast(count);
    }
    return contents;
}

fn validatePackedManifest(
    allocator: std.mem.Allocator,
    contents: []const u8,
) !void {
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        contents,
        .{},
    ) catch return error.InvalidExpertPackedDir;
    defer parsed.deinit();
    const root = switch (parsed.value) {
        .object => |object| object,
        else => return error.InvalidExpertPackedDir,
    };
    if (!jsonIntegerEquals(root.get("version"), PACKED_MANIFEST_VERSION) or
        !jsonIntegerEquals(root.get("n_experts"), PACKED_N_EXPERTS))
    {
        return error.InvalidExpertPackedDir;
    }
    const layers_value = root.get("layers") orelse
        return error.InvalidExpertPackedDir;
    const layers = switch (layers_value) {
        .object => |object| object,
        else => return error.InvalidExpertPackedDir,
    };
    if (layers.count() != N_LAYERS) return error.InvalidExpertPackedDir;

    for (0..N_LAYERS) |layer| {
        var layer_key_buffer: [8]u8 = undefined;
        const layer_key = try std.fmt.bufPrint(&layer_key_buffer, "{d}", .{layer});
        const layer_value = layers.get(layer_key) orelse
            return error.InvalidExpertPackedDir;
        const metadata = switch (layer_value) {
            .object => |object| object,
            else => return error.InvalidExpertPackedDir,
        };
        var name_buffer: [32]u8 = undefined;
        const expected_name = try std.fmt.bufPrint(
            &name_buffer,
            "layer_{d:0>2}.bin",
            .{layer},
        );
        if (!jsonIntegerEquals(metadata.get("expert_size"), PACKED_EXPERT_BYTES) or
            !jsonIntegerEquals(metadata.get("n_experts"), PACKED_N_EXPERTS) or
            !jsonStringEquals(metadata.get("file"), expected_name))
        {
            return error.InvalidExpertPackedDir;
        }
        const components_value = metadata.get("component_sizes") orelse
            return error.InvalidExpertPackedDir;
        const components = switch (components_value) {
            .object => |object| object,
            else => return error.InvalidExpertPackedDir,
        };
        if (components.count() != PACKED_COMPONENT_KEYS.len) {
            return error.InvalidExpertPackedDir;
        }
        var component_total: i64 = 0;
        for (PACKED_COMPONENT_KEYS, PACKED_COMPONENT_BYTES) |key, expected| {
            if (!jsonIntegerEquals(components.get(key), expected)) {
                return error.InvalidExpertPackedDir;
            }
            component_total += expected;
        }
        if (component_total != PACKED_EXPERT_BYTES) {
            return error.InvalidExpertPackedDir;
        }
    }
}

// Fail before tokenizer/backbone/model execution unless the authoritative
// manifest and all 43 fixed-layout exact-Q4 layer extents match the production
// DSV4 packed contract. Raw layer files have no magic header, so the manifest
// supplies the layout/version identity while exact extents prevent deferred
// short reads without scanning roughly 147 GiB of payload bytes.
pub fn validatePackedExpertDir(
    allocator: std.mem.Allocator,
    packed_dir: []const u8,
) !void {
    if (packed_dir.len == 0) return error.MissingExpertPackedDir;
    const manifest = try readPackedManifest(allocator, packed_dir);
    defer allocator.free(manifest);
    try validatePackedManifest(allocator, manifest);

    for (0..N_LAYERS) |layer| {
        var name_buffer: [32]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buffer, "layer_{d:0>2}.bin", .{layer});
        const path = try std.fs.path.join(allocator, &.{ packed_dir, name });
        defer allocator.free(path);
        const path_z = try allocator.dupeZ(u8, path);
        defer allocator.free(path_z);
        const fd = std.c.open(path_z.ptr, .{});
        if (fd < 0) return error.InvalidExpertPackedDir;
        defer _ = std.c.close(fd);
        var stat: std.c.Stat = undefined;
        if (std.c.fstat(fd, &stat) != 0 or stat.size != PACKED_LAYER_BYTES) {
            return error.InvalidExpertPackedDir;
        }
    }
}

// Bounded artifact identity: hashes small config/index/manifest content and
// metadata for the exact 43 packed files opened by engine.c. It deliberately
// does not scan ~148GB of weights and never incorporates prompt/token data.
fn routingTraceModelFingerprint(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    packed_dir: []const u8,
) !u64 {
    var hash: u64 = 14_695_981_039_346_656_037;
    fnv1aUpdate(&hash, "dmlx-artifact-metadata-v1");
    _ = try fingerprintFileContent(&hash, allocator, model_path, "config.json", true);
    const has_index = try fingerprintFileContent(
        &hash,
        allocator,
        model_path,
        "model.safetensors.index.json",
        false,
    );
    if (!has_index) {
        try fingerprintFileMetadata(&hash, allocator, model_path, "model.safetensors");
    }
    _ = try fingerprintFileContent(&hash, allocator, packed_dir, "manifest.json", false);

    for (0..N_LAYERS) |layer| {
        var name_buffer: [32]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buffer, "layer_{d:0>2}.bin", .{layer});
        try fingerprintFileMetadata(&hash, allocator, packed_dir, name);
    }
    return hash;
}

const RequestTelemetry = struct {
    enabled: bool,
    request_start_ns: u64,
    decode_start_ns: u64 = 0,
    token_timestamps_ns: [MAX_TELEMETRY_TOKENS]u64 = [_]u64{0} ** MAX_TELEMETRY_TOKENS,
    recorded_tokens: usize = 0,
    dspark_candidate_count: u64 = 0,
    dspark_accepted_count: u64 = 0,
    dspark_verification_batches: u64 = 0,
    dspark_verification_ns: u64 = 0,

    fn init(enabled: bool) RequestTelemetry {
        return .{
            .enabled = enabled,
            .request_start_ns = if (enabled) metal.monotonicNs() else 0,
        };
    }

    fn markDecodeStart(self: *RequestTelemetry) void {
        if (self.enabled) self.decode_start_ns = metal.monotonicNs();
    }

    fn recordToken(self: *RequestTelemetry) void {
        if (!self.enabled) return;
        if (self.recorded_tokens < self.token_timestamps_ns.len) {
            self.token_timestamps_ns[self.recorded_tokens] = metal.monotonicNs();
            self.recorded_tokens += 1;
        }
    }

    fn recordDSparkVerification(
        self: *RequestTelemetry,
        candidates: usize,
        accepted: usize,
        elapsed_ns: u64,
    ) void {
        if (!self.enabled) return;
        self.dspark_candidate_count +|= candidates;
        self.dspark_accepted_count +|= accepted;
        self.dspark_verification_batches +|= 1;
        self.dspark_verification_ns +|= elapsed_ns;
    }

    fn emit(
        self: *RequestTelemetry,
        engine: *metal.Engine,
        prompt_tokens: usize,
        generated_tokens: usize,
    ) void {
        if (!self.enabled) return;
        const request_end_ns = metal.monotonicNs();
        const available = @min(self.recorded_tokens, generated_tokens);
        const first_token_ns = if (available > 0) self.token_timestamps_ns[0] else 0;
        metal.telemetryEmitJson(
            engine,
            self.request_start_ns,
            first_token_ns,
            self.decode_start_ns,
            request_end_ns,
            @intCast(prompt_tokens),
            @intCast(generated_tokens),
            self.token_timestamps_ns[0..available],
            self.dspark_candidate_count,
            self.dspark_accepted_count,
            self.dspark_verification_batches,
            self.dspark_verification_ns,
        );
    }
};

pub const DSparkVerifyRequestState = struct {
    position: usize,
    token_cursor: usize,
    emitted_tokens: usize,
    token_buffer: []u32,
    current_hidden: []f32,
};

fn f32SlicesOverlap(a: []const f32, b: []const f32) bool {
    if (a.len == 0 or b.len == 0) return false;
    const a_bytes = std.math.mul(usize, a.len, @sizeOf(f32)) catch return true;
    const b_bytes = std.math.mul(usize, b.len, @sizeOf(f32)) catch return true;
    const a_start = @intFromPtr(a.ptr);
    const b_start = @intFromPtr(b.ptr);
    const a_end = std.math.add(usize, a_start, a_bytes) catch return true;
    const b_end = std.math.add(usize, b_start, b_bytes) catch return true;
    return a_start < b_end and b_start < a_end;
}

/// Task 5.3 request-owned half of DSpark verification transactions. Target KV
/// and recurrent state are protected by NativeKVTransaction; this value stages
/// position, token cursor, current hidden and sampler RNG until a prefix is
/// accepted. No token becomes externally visible before commit, and abort is
/// forbidden once the external emission floor advances.
pub const DSparkVerifyRequestTransaction = struct {
    const max_staged_tokens = 8;

    active: bool = false,
    start_position: usize = 0,
    original_token_cursor: usize = 0,
    external_emit_floor: usize = 0,
    hidden_width: usize = 0,
    staged_tokens: []const u32 = &.{},
    staged_hidden_rows: []const f32 = &.{},
    committed_base_hidden: []const f32 = &.{},
    original_hidden: []f32 = &.{},
    original_prng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(0),
    row_prng: [max_staged_tokens]std.Random.DefaultPrng = undefined,
    recorded_rows: usize = 0,

    pub fn begin(
        state: *const DSparkVerifyRequestState,
        sampler: *const sampling_mod.SamplerConfig,
        staged_tokens: []const u32,
        staged_hidden_rows: []const f32,
        original_hidden_storage: []f32,
    ) !DSparkVerifyRequestTransaction {
        if (staged_tokens.len == 0 or
            staged_tokens.len > max_staged_tokens or
            state.current_hidden.len == 0 or
            original_hidden_storage.len != state.current_hidden.len or
            staged_hidden_rows.len != staged_tokens.len * state.current_hidden.len or
            state.token_cursor > state.token_buffer.len or
            state.emitted_tokens > state.token_cursor or
            f32SlicesOverlap(original_hidden_storage, state.current_hidden) or
            f32SlicesOverlap(original_hidden_storage, staged_hidden_rows) or
            f32SlicesOverlap(state.current_hidden, staged_hidden_rows))
        {
            return error.InvalidVerifyRequestState;
        }
        // The checkpoint remains caller-provided to avoid a hidden allocation,
        // but is required to be disjoint from both live and staged storage.
        // This makes @memcpy valid and prevents speculative writes from
        // destroying the only rollback image.
        @memcpy(original_hidden_storage, state.current_hidden);
        return .{
            .active = true,
            .start_position = state.position,
            .original_token_cursor = state.token_cursor,
            .external_emit_floor = state.emitted_tokens,
            .hidden_width = state.current_hidden.len,
            .staged_tokens = staged_tokens,
            .staged_hidden_rows = staged_hidden_rows,
            .original_hidden = original_hidden_storage,
            .original_prng = sampler.prng,
        };
    }

    pub fn setCommittedBaseHidden(
        self: *DSparkVerifyRequestTransaction,
        hidden: []const f32,
    ) !void {
        if (!self.active or hidden.len != self.hidden_width or
            f32SlicesOverlap(hidden, self.original_hidden) or
            f32SlicesOverlap(hidden, self.staged_hidden_rows))
        {
            return error.InvalidVerifyBaseHidden;
        }
        self.committed_base_hidden = hidden;
    }

    pub fn recordSampledRow(
        self: *DSparkVerifyRequestTransaction,
        row: usize,
        sampler: *const sampling_mod.SamplerConfig,
    ) !void {
        if (!self.active or row != self.recorded_rows or
            row >= self.staged_tokens.len)
        {
            return error.InvalidVerifyRowOrder;
        }
        self.row_prng[row] = sampler.prng;
        self.recorded_rows += 1;
    }

    pub fn commitPrefix(
        self: *DSparkVerifyRequestTransaction,
        accepted: usize,
        state: *DSparkVerifyRequestState,
        sampler: *sampling_mod.SamplerConfig,
    ) !void {
        if (!self.active or accepted > self.staged_tokens.len or
            accepted > self.recorded_rows or
            state.position != self.start_position or
            state.token_cursor != self.original_token_cursor or
            state.emitted_tokens != self.external_emit_floor or
            state.current_hidden.len != self.hidden_width or
            accepted > state.token_buffer.len - state.token_cursor)
        {
            return error.InvalidVerifyCommit;
        }
        if (accepted > 0) {
            @memcpy(
                state.token_buffer[state.token_cursor..][0..accepted],
                self.staged_tokens[0..accepted],
            );
            const hidden_start = (accepted - 1) * self.hidden_width;
            @memcpy(
                state.current_hidden,
                self.staged_hidden_rows[hidden_start..][0..self.hidden_width],
            );
            sampler.prng = self.row_prng[accepted - 1];
        } else {
            const base_hidden = if (self.committed_base_hidden.len == self.hidden_width)
                self.committed_base_hidden
            else
                self.original_hidden;
            @memcpy(state.current_hidden, base_hidden);
            sampler.prng = self.original_prng;
        }
        state.position = self.start_position + accepted;
        state.token_cursor = self.original_token_cursor + accepted;
        self.active = false;
    }

    pub fn abort(
        self: *DSparkVerifyRequestTransaction,
        state: *DSparkVerifyRequestState,
        sampler: *sampling_mod.SamplerConfig,
    ) !void {
        if (!self.active) return error.InvalidVerifyAbort;
        if (state.emitted_tokens != self.external_emit_floor) {
            return error.ExternalEmissionAdvanced;
        }
        if (state.current_hidden.len != self.hidden_width or
            self.original_token_cursor > state.token_buffer.len)
        {
            return error.InvalidVerifyAbort;
        }
        state.position = self.start_position;
        state.token_cursor = self.original_token_cursor;
        @memcpy(state.current_hidden, self.original_hidden);
        sampler.prng = self.original_prng;
        self.active = false;
    }
};

/// Shared scope guard for both legacy DSpark verification entries. The generic
/// context keeps the restoration behavior directly unit-testable with an
/// injected forward failure, without constructing the full 48GB model engine.
fn runDSparkVerificationScope(context: anytype) !void {
    context.setAccumulate(false);
    context.setPhase(.verify);
    defer {
        context.setAccumulate(true);
        context.setPhase(.decode);
    }
    try context.forwardBatch();
}

const DSparkVerificationForwardContext = struct {
    owner: *NativeEngine,
    hidden: []f32,
    n_tokens: usize,
    start_position: u32,
    token_ids: []const i32,

    fn setAccumulate(self: *@This(), enabled: bool) void {
        metal.setDSparkAccumulate(self.owner.engine, enabled);
    }

    fn setPhase(self: *@This(), phase: metal.TelemetryPhase) void {
        self.owner.setObservabilityPhase(phase);
    }

    fn forwardBatch(self: *@This()) !void {
        try metal.forwardBatch(
            self.owner.engine,
            self.hidden,
            self.n_tokens,
            self.start_position,
            self.token_ids,
        );
    }
};

const DSparkPostCommitSampleKind = enum { correction, bonus };

const DSparkFinalizeOutcome = struct {
    generated: usize,
    hit_eos: bool,
};

/// Shared production commit coordinator. Model execution decides the accepted
/// prefix; this routine enforces owner order (DSpark -> target -> request), then
/// samples correction/bonus only from the committed request PRNG checkpoint.
/// The generic context gives focused tests the exact production state machine
/// without constructing the 48GB model.
fn finalizeDSparkVerification(
    context: anytype,
    verify_len: usize,
    accepted: usize,
    rejection_row: ?usize,
    accepted_eos: bool,
) !DSparkFinalizeOutcome {
    if (verify_len == 0 or accepted > verify_len or
        (rejection_row != null and rejection_row.? != accepted) or
        (accepted_eos and accepted == 0))
    {
        return error.InvalidVerifyDecision;
    }

    try context.commitDSpark(accepted);
    try context.commitTarget(accepted + 1);
    try context.commitRequest(accepted);

    var generated = accepted;
    var hit_eos = accepted_eos;
    if (!hit_eos) {
        if (rejection_row) |row| {
            if (!context.canAppend()) return error.InvalidVerifyOutputCapacity;
            const correction = try context.samplePostCommit(.correction, row);
            try context.appendToken(correction);
            generated += 1;
            hit_eos = context.isEos(correction);
        } else if (accepted == verify_len and context.canAppend()) {
            const bonus = try context.samplePostCommit(.bonus, verify_len);
            try context.appendToken(bonus);
            generated += 1;
            hit_eos = context.isEos(bonus);
        }
    }
    return .{ .generated = generated, .hit_eos = hit_eos };
}

pub const NativeEngine = struct {
    allocator: std.mem.Allocator,
    engine: *metal.Engine,
    weight_store: native_loader.NativeWeightStore,
    config: native_loader.DSV4NativeConfig,
    logits_buffer: []f32,
    eos_token: u32 = EOS_TOKEN,
    smelt_stats_path: []u8, // path to routing stats file, owned by NativeEngine
    telemetry_enabled: bool = false,
    routing_trace_enabled: bool = false,
    batch_config: metal.NativeBatchConfig,
    planner: chunk_planner.Planner,
    dspark: ?dspark_mod.DSparkState = null, // DSpark Markov Head (legacy, optional)
    dspark_engine: ?*metal.DSparkEngine = null, // Full DSpark engine (new, optional)
    exact_q4_next_request_ordinal: u32 = 0,
    exact_q4_capture: ?ExactQ4Capture = null,
    decode_timeline_capture: ?DecodeTimelineCapture = null,

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8, packed_dir: []const u8) !NativeEngine {
        return initWithDSpark(allocator, model_path, packed_dir, null);
    }

    pub fn initWithDSpark(allocator: std.mem.Allocator, model_path: []const u8, packed_dir: []const u8, dspark_dir: ?[]const u8) !NativeEngine {
        // Validate the complete packed-expert contract before config/weight
        // loading so direct native callers fail before model execution.
        try validatePackedExpertDir(allocator, packed_dir);

        // 1. Load config
        var loader = try native_loader.NativeEngineLoader.init(allocator, model_path);
        defer loader.deinit();
        const cfg = try loader.loadConfig();

        // 2. Load weights
        var store = try loader.loadWeights(cfg, 0);
        errdefer store.deinit();

        // 3. Init metal engine
        const engine = try metal.init(packed_dir);
        errdefer metal.deinit(engine);

        // 4. Set weights
        metal.setWeights(engine, store.weights);

        // Task 1.4 request/layer telemetry is explicitly opt-in. The C engine
        // keeps fixed-size counters and pays only a branch when disabled.
        const telemetry_env = std.c.getenv("NATIVE_TELEMETRY");
        const telemetry_enabled = if (telemetry_env) |value|
            std.mem.eql(u8, std.mem.span(value), "1")
        else
            false;
        metal.telemetryEnable(engine, telemetry_enabled);
        const attention_stage_env = std.c.getenv("NATIVE_ATTENTION_STAGE_TIME");
        const attention_stage_enabled = telemetry_enabled and if (attention_stage_env) |value|
            std.mem.eql(u8, std.mem.span(value), "1")
        else
            false;
        _ = metal.attentionStageConfigure(engine, attention_stage_enabled);
        std.log.info("native_engine: telemetry {s} (NATIVE_TELEMETRY), attention stage timestamps {s}", .{
            if (telemetry_enabled) "enabled" else "disabled",
            if (attention_stage_enabled) "requested" else "disabled",
        });

        // Batch controls are immutable exact-1 startup opt-ins. Validation and
        // installation use the same config object, so later environment changes
        // cannot drift the C engine away from the logged NativeEngine state.
        const batch_prefill_enabled = if (std.c.getenv("NATIVE_BATCH_PREFILL")) |value|
            std.mem.eql(u8, std.mem.span(value), "1")
        else
            false;
        const batch_mla_enabled = if (std.c.getenv("NATIVE_BATCH_MLA")) |value|
            std.mem.eql(u8, std.mem.span(value), "1")
        else
            false;
        const expert_buckets_enabled = if (std.c.getenv("NATIVE_EXPERT_BUCKETS")) |value|
            std.mem.eql(u8, std.mem.span(value), "1")
        else
            false;
        const batch_expert_kernels_enabled = if (std.c.getenv("NATIVE_BATCH_EXPERT_KERNELS")) |value|
            std.mem.eql(u8, std.mem.span(value), "1")
        else
            false;
        const dspark_batch_verify_enabled = if (std.c.getenv("NATIVE_DSPARK_BATCH_VERIFY")) |value|
            std.mem.eql(u8, std.mem.span(value), "1")
        else
            false;
        const experimental_cache_enabled = if (std.c.getenv("NATIVE_EXPERIMENTAL_CACHE")) |value|
            std.mem.eql(u8, std.mem.span(value), "1")
        else
            false;
        const exact_q4_enabled = if (std.c.getenv("NATIVE_EXACT_Q4_CAPTURE")) |value|
            std.mem.eql(u8, std.mem.span(value), "1")
        else
            false;
        const decode_timeline_enabled = if (std.c.getenv("DSV4_DECODE_TIMELINE_ENABLE")) |value| blk: {
            const text = std.mem.span(value);
            if (std.mem.eql(u8, text, "0")) break :blk false;
            if (std.mem.eql(u8, text, "1")) break :blk true;
            return error.InvalidDecodeTimelineEnable;
        } else false;
        if (decode_timeline_enabled and (batch_prefill_enabled or batch_mla_enabled or
            expert_buckets_enabled or batch_expert_kernels_enabled or
            dspark_batch_verify_enabled or experimental_cache_enabled or
            exact_q4_enabled or dspark_dir != null or telemetry_enabled or attention_stage_enabled or
            (if (std.c.getenv("NATIVE_PREFETCH")) |value|
                std.mem.eql(u8, std.mem.span(value), "1")
            else
                false) or
            std.c.getenv("NATIVE_TIME_LAYERS") != null or
            std.c.getenv("NATIVE_PHASE_TIME") != null or
            std.c.getenv("MF_STAGE_TIME") != null or
            std.c.getenv("NATIVE_DECODE_TIME") != null))
        {
            return error.DecodeTimelineIncompatibleMode;
        }
        if (experimental_cache_enabled and (batch_prefill_enabled or batch_mla_enabled or
            expert_buckets_enabled or batch_expert_kernels_enabled or dspark_batch_verify_enabled or
            dspark_dir != null))
        {
            return error.ExperimentalCacheIncompatibleWithBatchPath;
        }
        try metal.setExperimentalCache(engine, experimental_cache_enabled);
        std.log.info(
            "native_engine: cache policy {s} (NATIVE_EXPERIMENTAL_CACHE exact-1, default-off)",
            .{if (experimental_cache_enabled) "lru_observed_hot_seeded" else "fixed_hot_observed"},
        );
        const batch_config = metal.NativeBatchConfig{
            .enable_batch_prefill = @intFromBool(batch_prefill_enabled),
            .enable_batch_mla = @intFromBool(batch_mla_enabled),
            .enable_expert_buckets = @intFromBool(expert_buckets_enabled),
            .enable_batched_expert_kernels = @intFromBool(batch_expert_kernels_enabled),
            .enable_dspark_batch_verify = @intFromBool(dspark_batch_verify_enabled),
            .max_prefill_chunk = 8,
            .max_verify_chunk = 5,
            .workspace_budget_bytes = 256 * 1024 * 1024,
        };
        try metal.validateBatchConfig(&batch_config);
        try metal.setBatchConfig(engine, &batch_config);
        std.log.info(
            "native_engine: batch flags prefill={d} mla={d} buckets={d} expert_kernels={d} dspark_verify={d} max_prefill={d} max_verify={d} workspace_mb={d}",
            .{
                batch_config.enable_batch_prefill,
                batch_config.enable_batch_mla,
                batch_config.enable_expert_buckets,
                batch_config.enable_batched_expert_kernels,
                batch_config.enable_dspark_batch_verify,
                batch_config.max_prefill_chunk,
                batch_config.max_verify_chunk,
                batch_config.workspace_budget_bytes / (1024 * 1024),
            },
        );

        // 4b. Set compressor/indexer weights per layer (T2C.1)
        for (0..cfg.num_hidden_layers) |i| {
            const ratio = store.weights.compress_ratio[i];
            if (ratio > 0) {
                if (store.weights.comp_wkv[i]) |wkv| {
                    const wgate = store.weights.comp_wgate[i].?;
                    metal.setLayerCompressor(
                        engine,
                        i,
                        ratio,
                        metal.toCQuantWeight(wkv),
                        metal.toCQuantWeight(wgate),
                        if (store.weights.comp_ape[i].len > 0) store.weights.comp_ape[i].ptr else null,
                        if (store.weights.comp_norm[i].len > 0) store.weights.comp_norm[i].ptr else null,
                    );
                }
                if (ratio == 4) {
                    if (store.weights.idx_wq_b[i]) |wqb| {
                        metal.setLayerIndexer(
                            engine,
                            i,
                            metal.toCQuantWeight(wqb),
                            metal.toCQuantWeight(store.weights.idx_weights_proj[i].?),
                            metal.toCQuantWeight(store.weights.idx_comp_wkv[i].?),
                            metal.toCQuantWeight(store.weights.idx_comp_wgate[i].?),
                            if (store.weights.idx_comp_ape[i].len > 0) store.weights.idx_comp_ape[i].ptr else null,
                            if (store.weights.idx_comp_norm[i].len > 0) store.weights.idx_comp_norm[i].ptr else null,
                        );
                    }
                }
            }
        }

        // 5. Allocate logits buffer
        const logits = try allocator.alloc(f32, cfg.vocab_size);
        errdefer allocator.free(logits);

        // 6. SMELT startup preload — bounded fixed-hot cache for the 48GB target.
        //    The safe default is N=20 (~11GB across 43 layers). Larger values
        //    are explicit experiments and must not become a runtime default.
        //    NATIVE_SMELT_N=0 disables the fixed-hot cache.
        const smelt_n_str = std.c.getenv("NATIVE_SMELT_N");
        const smelt_n: i32 = if (smelt_n_str) |s| std.fmt.parseInt(i32, std.mem.span(s), 10) catch 20 else 20;
        if (experimental_cache_enabled and smelt_n < 6)
            return error.ExperimentalCacheRequiresBoundedSmeltPool;

        const runtime_capacity = metal.batchRuntimeCapacity(&batch_config);
        const planner_memory = chunk_planner.MemoryBudget.forSmeltN(
            @intCast(@max(smelt_n, 0)),
        );
        const planner = chunk_planner.Planner.init(
            .{
                .batch_prefill = batch_config.enable_batch_prefill == 1,
                .batch_mla = batch_config.enable_batch_mla == 1,
                .expert_buckets = batch_config.enable_expert_buckets == 1,
                .batched_expert_kernels = batch_config.enable_batched_expert_kernels == 1,
                .dspark_batch_verify = batch_config.enable_dspark_batch_verify == 1,
            },
            .{
                .max_prefill_chunk = batch_config.max_prefill_chunk,
                .max_verify_chunk = batch_config.max_verify_chunk,
                .workspace_capacity_tokens = runtime_capacity.workspace_capacity_tokens,
                .workspace_budget_bytes = runtime_capacity.workspace_budget_bytes,
                .workspace_allocated_bytes = runtime_capacity.workspace_allocated_bytes,
                .mla_workspace_budget_bytes = runtime_capacity.workspace_budget_bytes,
                .batch_pool_slots = runtime_capacity.batch_pool_slots,
                .hard_token_limit = runtime_capacity.supported_capacity_tokens,
                .expert_bytes = runtime_capacity.expert_slot_bytes,
                .memory = planner_memory,
            },
        );
        std.log.info(
            "native_engine: chunk planner smelt_n={d} physical_gib={d} incremental_budget_mib={d} hard_batch={d} pool_slots={d}",
            .{
                planner_memory.smelt_n,
                planner_memory.physical_bytes / (1024 * 1024 * 1024),
                planner_memory.incrementalCapacity() / (1024 * 1024),
                runtime_capacity.supported_capacity_tokens,
                runtime_capacity.batch_pool_slots,
            },
        );

        // Task 1.5 routing trace is independently opt-in. The C writer owns a
        // single fixed-capacity buffer and accepts numeric routing metadata only.
        const routing_trace_env = std.c.getenv("NATIVE_ROUTING_TRACE");
        const routing_trace_enabled = routing_trace_env != null and
            std.mem.span(routing_trace_env.?).len > 0;
        if (routing_trace_enabled) {
            const capacity_env = std.c.getenv("NATIVE_ROUTING_TRACE_CAPACITY");
            const capacity = if (capacity_env) |value|
                std.fmt.parseInt(usize, std.mem.span(value), 10) catch
                    return error.InvalidRoutingTraceCapacity
            else
                DEFAULT_ROUTING_TRACE_CAPACITY;
            if (capacity == 0 or capacity > MAX_ROUTING_TRACE_CAPACITY)
                return error.InvalidRoutingTraceCapacity;
            const fingerprint = try routingTraceModelFingerprint(allocator, model_path, packed_dir);
            try metal.routingTraceConfigure(
                engine,
                routing_trace_env.?,
                capacity,
                fingerprint,
                @intCast(@max(smelt_n, 0)),
            );
            std.log.info(
                "native_engine: routing trace enabled (NATIVE_ROUTING_TRACE, capacity={d}, fingerprint={d})",
                .{ capacity, fingerprint },
            );
        } else {
            std.log.info("native_engine: routing trace disabled (NATIVE_ROUTING_TRACE)", .{});
        }

        // Build stats file path: {packed_dir}/.smelt_routing_stats.bin
        const stats_path_buf = try std.fmt.allocPrint(allocator, "{s}/.smelt_routing_stats.bin\x00", .{packed_dir});
        errdefer allocator.free(stats_path_buf);

        if (smelt_n > 0) {
            std.log.info("native_engine: SMELT preloading {d} experts/layer ({d:.1} GB) — please wait...", .{
                smelt_n,
                @as(f64, @floatFromInt(smelt_n)) * 43.0 * 13.4 / 1024.0,
            });
            // Load routing stats from previous runs so smelt_finish_warmup
            // selects the ACTUAL hot experts rather than experts 0..N-1 by ID.
            // MLX-aligned approach: two phases are automatic, no manual pre-warmup needed.
            // - Phase 1 (no stats, default N=20): penalty=0, unbiased routing → discovers hot experts
            // - Phase 2 (stats loaded): penalty=1e3 steers routing to hot experts → I/O ≈ 0
            metal.smeltInit(engine, 0, smelt_n, 0.0);
            // Load stats AFTER smeltInit (smeltInit zeros routing_counts, so load must come after)
            const stats_loaded = metal.smeltLoadStats(engine, stats_path_buf[0 .. stats_path_buf.len - 1 :0].ptr);
            if (stats_loaded != 0) {
                // Stats available: hot experts loaded into SMELT (stats-based selection).
                // Keep penalty=0 so routing remains natural and correct for all prompts.
                // The performance benefit comes from SMELT caching the actual hot experts
                // (not the default 0..N-1), improving cache hit rate without routing bias.
                std.log.info("native_engine: Phase 2 — hot experts loaded from stats (natural routing, N={d})", .{smelt_n});
            } else {
                std.log.info("native_engine: Phase 1 — collecting routing stats (penalty=0, N={d})", .{smelt_n});
            }
            const n_loaded = metal.smeltFinishWarmup(engine);
            // Register stats path for periodic auto-save (protects against OOM kill)
            metal.smeltSetStatsPath(engine, stats_path_buf[0 .. stats_path_buf.len - 1 :0].ptr);
            if (n_loaded > 0) {
                std.log.info("native_engine: SMELT ready — {d} experts/layer in RAM (routing-stats based)", .{n_loaded});
                const gather_env = std.c.getenv("NATIVE_GATHER");
                if (gather_env != null) {
                    const gather_ok = metal.initGatherMode(engine);
                    if (gather_ok > 0) {
                        std.log.info("native_engine: gather mode active (EXPERIMENTAL)", .{});
                    } else {
                        std.log.warn("native_engine: gather mode init failed", .{});
                    }
                }
            } else {
                std.log.warn("native_engine: SMELT preload failed — falling back to SSD reads", .{});
            }
        } else {
            std.log.warn("native_engine: SMELT disabled (NATIVE_SMELT_N=0) — SSD reads each token (~4200ms/tok)", .{});
        }

        std.log.info("native_engine: initialized (layers={d}, vocab={d})", .{ cfg.num_hidden_layers, cfg.vocab_size });

        // Load DSpark Markov Head if directory provided
        var dspark_state: ?dspark_mod.DSparkState = null;
        if (dspark_dir) |dir| {
            dspark_state = dspark_mod.DSparkState.init(allocator, dir, cfg.vocab_size) catch |err| blk: {
                std.log.warn("native_engine: DSpark load failed ({any}), falling back to standard decode", .{err});
                break :blk null;
            };
        }

        // Initialize full DSpark engine if weights directory is available
        // (uses dspark_weights/ subdir for non-expert weights, packed_mtp_experts/ for INT8 experts)
        var dspark_full_engine: ?*metal.DSparkEngine = null;
        if (dspark_dir) |dir| {
            const weight_dir_z = try allocator.dupeZ(u8, dir);
            defer allocator.free(weight_dir_z);
            // Construct packed_mtp_experts path: sibling of dspark_weights dir
            const mtp_expert_path = try std.fmt.allocPrint(allocator, "{s}/../packed_mtp_experts\x00", .{dir});
            defer allocator.free(mtp_expert_path);
            dspark_full_engine = metal.dsparkInit(
                weight_dir_z,
                mtp_expert_path[0 .. mtp_expert_path.len - 1 :0],
                engine,
            );
            if (dspark_full_engine) |de| {
                metal.setDSparkEngine(engine, de);
                std.log.info("native_engine: DSpark full engine initialized (block_size=5)", .{});
            } else {
                std.log.warn("native_engine: DSpark full engine init failed, using Markov-only fallback", .{});
            }
        }

        var exact_q4_capture: ?ExactQ4Capture = null;
        if (exact_q4_enabled) {
            const output_env = std.c.getenv("NATIVE_EXACT_Q4_RAW_JSONL") orelse
                return error.ExactQ4OutputPathRequired;
            const output_path = std.mem.span(output_env);
            if (output_path.len == 0) return error.ExactQ4OutputPathRequired;
            const prompt_capacity = if (std.c.getenv("NATIVE_EXACT_Q4_MAX_PROMPT_TOKENS")) |value|
                std.fmt.parseInt(usize, std.mem.span(value), 10) catch
                    return error.InvalidExactQ4PromptCapacity
            else
                EXACT_Q4_DEFAULT_MAX_PROMPT_TOKENS;
            if (prompt_capacity == 0) return error.InvalidExactQ4PromptCapacity;
            const token_capacity = std.math.add(usize, prompt_capacity, EXACT_Q4_MAX_GENERATED_TOKENS) catch
                return error.InvalidExactQ4PromptCapacity;
            const stage_capacity = std.math.mul(usize, token_capacity, EXACT_Q4_STAGES_PER_TOKEN) catch
                return error.InvalidExactQ4PromptCapacity;
            const read_capacity = std.math.mul(usize, token_capacity, EXACT_Q4_READS_PER_TOKEN) catch
                return error.InvalidExactQ4PromptCapacity;
            if (stage_capacity > metal.EXACT_Q4_MAX_STAGE_CAPACITY or
                read_capacity > metal.EXACT_Q4_MAX_READ_CAPACITY)
                return error.InvalidExactQ4PromptCapacity;

            const stages = try allocator.alloc(metal.ExactQ4StageEvent, stage_capacity);
            errdefer allocator.free(stages);
            const reads = try allocator.alloc(metal.ExactQ4ExpertReadEvent, read_capacity);
            errdefer allocator.free(reads);
            const mla_paths = try allocator.alloc(metal.MlaPathEvent, stage_capacity);
            errdefer allocator.free(mla_paths);
            const intervals = try allocator.alloc(ExactQ4TokenInterval, token_capacity);
            errdefer allocator.free(intervals);
            const output_path_z = try allocator.dupeZ(u8, output_path);
            defer allocator.free(output_path_z);
            const output_fd = std.c.open(output_path_z.ptr, .{
                .ACCMODE = .WRONLY,
                .CREAT = true,
                .APPEND = true,
            }, @as(std.c.mode_t, 0o600));
            if (output_fd < 0) return error.ExactQ4OutputOpenFailed;
            errdefer _ = std.c.close(output_fd);
            try metal.exactQ4Configure(engine, @intCast(stage_capacity), @intCast(read_capacity));
            errdefer metal.exactQ4Deconfigure(engine) catch {};
            try metal.exactQ4Enable(engine, true);
            exact_q4_capture = .{
                .output_fd = output_fd,
                .stages = stages,
                .reads = reads,
                .mla_paths = mla_paths,
                .tokens = intervals,
            };
            std.log.info("native_engine: exact-Q4 capture enabled path={s} tokens={d} stages={d} reads={d}", .{
                output_path, token_capacity, stage_capacity, read_capacity,
            });
        }

        var decode_timeline_capture: ?DecodeTimelineCapture = null;
        if (decode_timeline_enabled) {
            const output_env = std.c.getenv("DSV4_DECODE_TIMELINE_RAW_JSONL") orelse
                return error.DecodeTimelineOutputPathRequired;
            const output_path = std.mem.span(output_env);
            if (output_path.len == 0) return error.DecodeTimelineOutputPathRequired;
            const capacity = DECODE_TIMELINE_MAX_TOKENS * DECODE_TIMELINE_EVENTS_PER_TOKEN;
            const events = try allocator.alloc(metal.DecodeTimelineEvent, capacity);
            errdefer allocator.free(events);
            const output_path_z = try allocator.dupeZ(u8, output_path);
            defer allocator.free(output_path_z);
            const output_fd = std.c.open(output_path_z.ptr, .{
                .ACCMODE = .WRONLY,
                .CREAT = true,
                .APPEND = true,
            }, @as(std.c.mode_t, 0o600));
            if (output_fd < 0) return error.DecodeTimelineOutputOpenFailed;
            errdefer _ = std.c.close(output_fd);
            try metal.decodeTimelineConfigure(engine, @intCast(capacity));
            errdefer metal.decodeTimelineDeconfigure(engine) catch {};
            try metal.decodeTimelineEnable(engine, true);
            decode_timeline_capture = .{ .output_fd = output_fd, .events = events };
        }

        return .{
            .allocator = allocator,
            .engine = engine,
            .weight_store = store,
            .config = cfg,
            .logits_buffer = logits,
            .eos_token = cfg.eos_token_id,
            .smelt_stats_path = stats_path_buf,
            .telemetry_enabled = telemetry_enabled,
            .routing_trace_enabled = routing_trace_enabled,
            .batch_config = batch_config,
            .planner = planner,
            .dspark = dspark_state,
            .dspark_engine = dspark_full_engine,
            .exact_q4_capture = exact_q4_capture,
            .decode_timeline_capture = decode_timeline_capture,
        };
    }

    pub fn deinit(self: *NativeEngine) void {
        // Save routing stats so next startup loads the correct hot experts
        metal.smeltSaveStats(self.engine, self.smelt_stats_path[0 .. self.smelt_stats_path.len - 1 :0].ptr);
        self.allocator.free(self.smelt_stats_path);
        self.allocator.free(self.logits_buffer);
        if (self.dspark) |*ds| ds.deinit();
        if (self.dspark_engine) |de| {
            metal.setDSparkEngine(self.engine, null);
            metal.dsparkDeinit(de);
        }
        if (self.decode_timeline_capture) |*capture| {
            metal.decodeTimelineEnable(self.engine, false) catch {};
            metal.decodeTimelineDeconfigure(self.engine) catch {};
            capture.deinit(self.allocator);
        }
        if (self.exact_q4_capture) |*capture| {
            metal.exactQ4Enable(self.engine, false) catch |err|
                std.log.err("native_engine: exact-Q4 disable at deinit failed: {any}", .{err});
            metal.exactQ4Deconfigure(self.engine) catch |err|
                std.log.err("native_engine: exact-Q4 deconfigure at deinit failed: {any}", .{err});
            capture.deinit(self.allocator);
        }
        metal.deinit(self.engine);
        self.weight_store.deinit();
    }

    pub fn resetKv(self: *NativeEngine) void {
        metal.resetKv(self.engine);
        if (self.dspark_engine) |de| metal.dsparkReset(de);
    }

    fn setObservabilityPhase(self: *NativeEngine, phase: metal.TelemetryPhase) void {
        if (self.telemetry_enabled) metal.telemetrySetPhase(self.engine, phase);
        if (self.routing_trace_enabled) metal.routingTraceSetPhase(self.engine, phase);
    }

    fn beginRoutingTrace(self: *NativeEngine) !void {
        if (self.routing_trace_enabled) try metal.routingTraceRequestBegin(self.engine);
    }

    fn endRoutingTrace(self: *NativeEngine) void {
        if (self.routing_trace_enabled and !metal.routingTraceRequestEnd(self.engine)) {
            std.log.err("native_engine: routing trace request flush failed", .{});
        }
    }

    const ExactQ4Request = struct {
        enabled: bool,
        request_ordinal: u32 = 0,
        request_begin_ns: u64 = 0,
        next_token_ordinal: u32 = 0,
        token_count: usize = 0,
        token_open: bool = false,

        fn begin(owner: *NativeEngine) !ExactQ4Request {
            const enabled = metal.exactQ4RecordingEnabled(owner.engine);
            if (!enabled) return .{ .enabled = false };
            const request_begin_ns = metal.monotonicNs();
            const request_ordinal = owner.exact_q4_next_request_ordinal;
            const next_request_ordinal = std.math.add(
                u32,
                request_ordinal,
                1,
            ) catch return error.ExactQ4RequestOrdinalExhausted;
            try metal.exactQ4BeginRequest(owner.engine, request_ordinal);
            owner.exact_q4_next_request_ordinal = next_request_ordinal;
            return .{
                .enabled = true,
                .request_ordinal = request_ordinal,
                .request_begin_ns = request_begin_ns,
            };
        }

        fn beginToken(self: *ExactQ4Request, owner: *NativeEngine, steady: bool) !void {
            if (!self.enabled) return;
            const capture = &(owner.exact_q4_capture orelse return error.InvalidExactQ4CaptureState);
            const token_begin_ns = metal.monotonicNs();
            if (self.token_open) {
                capture.tokens[self.token_count - 1].end_ns = token_begin_ns;
                self.token_open = false;
            }
            if (self.token_count >= capture.tokens.len) return error.ExactQ4CapacityExceeded;
            const token_ordinal = self.next_token_ordinal;
            const next_token_ordinal = std.math.add(
                u32,
                token_ordinal,
                1,
            ) catch return error.ExactQ4TokenOrdinalExhausted;
            try metal.exactQ4BeginToken(owner.engine, token_ordinal);
            capture.tokens[self.token_count] = .{
                .token_ordinal = token_ordinal,
                .begin_ns = token_begin_ns,
                .end_ns = 0,
                .steady = steady,
            };
            self.token_count += 1;
            self.token_open = true;
            self.next_token_ordinal = next_token_ordinal;
        }

        fn appendLogitsSample(
            self: *const ExactQ4Request,
            owner: *NativeEngine,
            begin_ns: u64,
        ) !void {
            if (!self.enabled) return;
            const observed_end = metal.monotonicNs();
            const end_ns = if (observed_end > begin_ns)
                observed_end
            else
                std.math.add(u64, begin_ns, 1) catch return error.ExactQ4ClockExhausted;
            _ = try metal.exactQ4AppendStage(owner.engine, &.{
                .request_ordinal = 0,
                .token_ordinal = 0,
                .event_ordinal = 0,
                .layer = std.math.maxInt(u16),
                .stage = @intFromEnum(metal.ExactQ4Stage.logits_sample),
                .flags = metal.EXACT_Q4_EVENT_CPU_ACTIVE,
                .calls = 1,
                .begin_ns = begin_ns,
                .end_ns = end_ns,
                .bytes = 0,
            });
        }

        fn writeRawSnapshot(
            self: *const ExactQ4Request,
            owner: *NativeEngine,
            request_end_ns: u64,
            snapshot: metal.ExactQ4LatencySnapshot,
            mla: metal.ExactQ4MlaPathSnapshot,
        ) !void {
            var capture = &(owner.exact_q4_capture orelse return error.InvalidExactQ4CaptureState);
            const fd = capture.output_fd;
            const record_begin = std.c.lseek(fd, 0, 2);
            if (record_begin < 0) return error.ExactQ4OutputWriteFailed;
            errdefer {
                _ = std.c.ftruncate(fd, record_begin);
                _ = std.c.fsync(fd);
            }
            try exactQ4Print(fd, "{{\"producer\":{{\"abi_version\":{d},\"valid\":{d},\"enabled\":{d},\"finalized\":{d},\"request_ordinal\":{d},\"token_ordinal\":{d},\"stage_capacity\":{d},\"read_capacity\":{d},\"stage_count\":{d},\"read_count\":{d},\"dropped_stages\":{d},\"dropped_reads\":{d},\"incomplete\":{d},\"active_read_publishers\":{d}}},\"requests\":[{{\"request_ordinal\":{d},\"begin_ns\":{d},\"end_ns\":{d},\"tokens\":[", .{
                snapshot.abi_version,     snapshot.valid,                  snapshot.enabled,        snapshot.finalized,
                snapshot.request_ordinal, snapshot.token_ordinal,          snapshot.stage_capacity, snapshot.read_capacity,
                snapshot.stage_count,     snapshot.read_count,             snapshot.dropped_stages, snapshot.dropped_reads,
                snapshot.incomplete,      snapshot.active_read_publishers, self.request_ordinal,    self.request_begin_ns,
                request_end_ns,
            });
            for (capture.tokens[0..self.token_count], 0..) |token, index| {
                if (index != 0) try exactQ4Write(fd, ",");
                try exactQ4Print(fd, "{{\"token_ordinal\":{d},\"begin_ns\":{d},\"end_ns\":{d},\"steady\":{s}}}", .{
                    token.token_ordinal,                   token.begin_ns, token.end_ns,
                    if (token.steady) "true" else "false",
                });
            }
            try exactQ4Write(fd, "]}],\"stages\":[");
            for (capture.stages[0..snapshot.stage_count], 0..) |event, index| {
                if (index != 0) try exactQ4Write(fd, ",");
                try exactQ4Print(fd, "{{\"request_ordinal\":{d},\"token_ordinal\":{d},\"event_ordinal\":{d},\"layer\":{d},\"stage\":{d},\"flags\":{d},\"calls\":{d},\"begin_ns\":{d},\"end_ns\":{d},\"bytes\":{d}}}", .{
                    event.request_ordinal, event.token_ordinal, event.event_ordinal, event.layer,
                    event.stage,           event.flags,         event.calls,         event.begin_ns,
                    event.end_ns,          event.bytes,
                });
            }
            try exactQ4Write(fd, "],\"reads\":[");
            for (capture.reads[0..snapshot.read_count], 0..) |event, index| {
                if (index != 0) try exactQ4Write(fd, ",");
                try exactQ4Print(fd, "{{\"request_ordinal\":{d},\"token_ordinal\":{d},\"event_ordinal\":{d},\"layer\":{d},\"expert\":{d},\"route_rank\":{d},\"source\":{d},\"reserved\":{d},\"error_code\":{d},\"flags\":{d},\"submit_ns\":{d},\"worker_start_ns\":{d},\"complete_ns\":{d},\"requested_bytes\":{d},\"read_bytes\":{d}}}", .{
                    event.request_ordinal, event.token_ordinal,   event.event_ordinal, event.layer,
                    event.expert,          event.route_rank,      event.source,        event.reserved,
                    event.error_code,      event.flags,           event.submit_ns,     event.worker_start_ns,
                    event.complete_ns,     event.requested_bytes, event.read_bytes,
                });
            }
            try exactQ4Print(fd, "],\"mla_snapshot\":{{\"abi_version\":{d},\"valid\":{d},\"enabled\":{d},\"finalized\":{d},\"request_ordinal\":{d},\"token_ordinal\":{d},\"capacity\":{d},\"count\":{d},\"dropped\":{d},\"reserved\":{d}}},\"mla_paths\":[", .{
                mla.abi_version,   mla.valid,    mla.enabled, mla.finalized, mla.request_ordinal,
                mla.token_ordinal, mla.capacity, mla.count,   mla.dropped,   mla.reserved,
            });
            for (capture.mla_paths[0..mla.count], 0..) |event, index| {
                if (index != 0) try exactQ4Write(fd, ",");
                try exactQ4Print(fd, "{{\"request_ordinal\":{d},\"token_ordinal\":{d},\"event_ordinal\":{d},\"layer\":{d},\"path\":{d},\"flags\":{d}}}", .{
                    event.request_ordinal, event.token_ordinal, event.event_ordinal,
                    event.layer,           event.path,          event.flags,
                });
            }
            try exactQ4Write(fd, "]}\n");
            if (std.c.fsync(fd) != 0) return error.ExactQ4OutputSyncFailed;
        }

        fn finalizeCapture(self: *ExactQ4Request, owner: *NativeEngine) !void {
            if (!self.enabled) return;
            const capture = &(owner.exact_q4_capture orelse return error.InvalidExactQ4CaptureState);
            const request_end_ns = metal.monotonicNs();
            if (self.token_open) {
                capture.tokens[self.token_count - 1].end_ns = request_end_ns;
                self.token_open = false;
            }
            try metal.exactQ4Finalize(owner.engine);
            errdefer metal.exactQ4Reset(owner.engine) catch |err|
                std.log.err("native_engine: exact-Q4 reset after capture failure failed: {any}", .{err});
            const snapshot = try metal.exactQ4Snapshot(owner.engine, capture.stages, capture.reads);
            const mla = try metal.exactQ4SnapshotMlaPaths(owner.engine, capture.mla_paths);
            if (snapshot.valid != 1 or snapshot.finalized != 1 or snapshot.enabled != 1 or
                snapshot.dropped_stages != 0 or snapshot.dropped_reads != 0 or
                snapshot.incomplete != 0 or snapshot.active_read_publishers != 0 or
                mla.valid != 1 or mla.finalized != 1 or mla.enabled != 1 or
                mla.dropped != 0 or mla.reserved != 0 or
                mla.count != @as(u32, @intCast(self.token_count * N_LAYERS)))
                return error.InvalidExactQ4Snapshot;
            try self.writeRawSnapshot(owner, request_end_ns, snapshot, mla);
            try metal.exactQ4Reset(owner.engine);
        }

        fn finalize(self: *ExactQ4Request, owner: *NativeEngine) void {
            self.finalizeCapture(owner) catch |err| {
                std.log.err("native_engine: exact-Q4 capture failed closed; token result preserved: {any}", .{err});
                metal.exactQ4Reset(owner.engine) catch {};
                metal.exactQ4Enable(owner.engine, false) catch |disable_err|
                    std.log.err("native_engine: exact-Q4 disable after capture failure failed: {any}", .{disable_err});
            };
        }

        fn abortBeforeReset(self: *const ExactQ4Request, owner: *NativeEngine) void {
            if (!self.enabled) {
                owner.resetKv();
                return;
            }
            metal.exactQ4DrainAndAbortRequest(owner.engine) catch |err| {
                std.log.err(
                    "native_engine: exact-Q4 abort failed before KV reset: {any}",
                    .{err},
                );
                return;
            };
            owner.resetKv();
        }
    };

    const DecodeTimelineRequest = struct {
        enabled: bool,
        next_token: u32 = 0,

        fn begin(owner: *NativeEngine) !DecodeTimelineRequest {
            const enabled = metal.decodeTimelineRecordingEnabled(owner.engine);
            if (!enabled) return .{ .enabled = false };
            try metal.decodeTimelineBeginRequest(owner.engine);
            return .{ .enabled = true };
        }
        fn beginToken(self: *DecodeTimelineRequest, owner: *NativeEngine) !void {
            if (!self.enabled) return;
            try metal.decodeTimelineBeginToken(owner.engine, self.next_token);
            self.next_token = std.math.add(u32, self.next_token, 1) catch
                return error.DecodeTimelineTokenOrdinalExhausted;
        }
        fn appendSample(self: *const DecodeTimelineRequest, owner: *NativeEngine, begin_ns: u64) !void {
            if (!self.enabled) return;
            const end_ns = metal.decodeTimelineNowNs();
            try metal.decodeTimelineAppend(owner.engine, &.{
                .stage = @intFromEnum(metal.DecodeTimelineStage.token_sample),
                .begin_ns = begin_ns,
                .encode_done_ns = begin_ns,
                .pre_commit_ns = begin_ns,
                .post_commit_ns = begin_ns,
                .post_wait_ns = end_ns,
                .end_ns = end_ns,
            });
        }
        fn appendTokenForward(self: *const DecodeTimelineRequest, owner: *NativeEngine, begin_ns: u64) !void {
            if (!self.enabled) return;
            const end_ns = metal.decodeTimelineNowNs();
            try metal.decodeTimelineAppend(owner.engine, &.{
                .stage = @intFromEnum(metal.DecodeTimelineStage.token_forward),
                .begin_ns = begin_ns,
                .encode_done_ns = begin_ns,
                .pre_commit_ns = begin_ns,
                .post_commit_ns = begin_ns,
                .post_wait_ns = end_ns,
                .end_ns = end_ns,
            });
        }
        fn finalize(self: *DecodeTimelineRequest, owner: *NativeEngine) !void {
            if (!self.enabled) return;
            if (self.next_token == 0) return self.resetEmpty(owner);
            self.finalizeCapture(owner) catch |err| {
                std.log.err("native_engine: decode timeline failed closed: {any}", .{err});
                metal.decodeTimelineReset(owner.engine) catch {};
                metal.decodeTimelineEnable(owner.engine, false) catch {};
                return err;
            };
        }
        fn resetEmpty(self: *DecodeTimelineRequest, owner: *NativeEngine) !void {
            if (!self.enabled) return;
            try metal.decodeTimelineReset(owner.engine);
        }
        fn finalizeCapture(self: *DecodeTimelineRequest, owner: *NativeEngine) !void {
            _ = self;
            var capture = &(owner.decode_timeline_capture orelse
                return error.InvalidDecodeTimelineCaptureState);
            try metal.decodeTimelineFinalize(owner.engine);
            errdefer metal.decodeTimelineReset(owner.engine) catch {};
            const snapshot = try metal.decodeTimelineSnapshot(owner.engine, capture.events);
            if (snapshot.valid != 1 or snapshot.finalized != 1 or
                snapshot.dropped != 0 or snapshot.incomplete != 0 or
                snapshot.active_publishers != 0)
                return error.InvalidDecodeTimelineSnapshot;
            const fd = capture.output_fd;
            const record_begin = std.c.lseek(fd, 0, 2);
            if (record_begin < 0) return error.DecodeTimelineOutputWriteFailed;
            errdefer {
                const truncate_failed = std.c.ftruncate(fd, record_begin) != 0;
                const sync_failed = std.c.fsync(fd) != 0;
                if (truncate_failed or sync_failed) {
                    _ = std.c.close(fd);
                    capture.output_fd = -1;
                }
            }
            try exactQ4Print(fd, "{{\"abi_version\":{d},\"valid\":{d},\"count\":{d},\"events\":[", .{
                snapshot.abi_version, snapshot.valid, snapshot.count,
            });
            for (capture.events[0..snapshot.count], 0..) |event, index| {
                if (index != 0) try exactQ4Write(fd, ",");
                try exactQ4Print(fd, "{{\"token\":{d},\"layer\":{d},\"stage\":{d},\"status\":{d},\"begin_ns\":{d},\"encode_done_ns\":{d},\"pre_commit_ns\":{d},\"post_commit_ns\":{d},\"post_wait_ns\":{d},\"end_ns\":{d}}}", .{
                    event.token,        event.layer,          event.stage,         event.status,
                    event.begin_ns,     event.encode_done_ns, event.pre_commit_ns, event.post_commit_ns,
                    event.post_wait_ns, event.end_ns,
                });
            }
            try exactQ4Write(fd, "]}\n");
            if (std.c.fsync(fd) != 0) return error.DecodeTimelineOutputSyncFailed;
            try metal.decodeTimelineReset(owner.engine);
        }
    };

    const PrefillResult = struct {
        last_hidden: [MHC_MULT * DIM]f32,
        chunk_count: usize,
        batched_chunk_count: usize,
    };

    /// Execute the prompt through the phase-aware planner. All host allocation
    /// and any shrink happen before the selected chunk enters C. Once a batch
    /// forward starts, an error is request-fatal and resetKv discards all
    /// partially committed/recurrent state; no mutated-state retry is allowed.
    fn prefillPrompt(
        self: *NativeEngine,
        exact_q4: *ExactQ4Request,
        prompt_tokens: []const u32,
        base_position: usize,
    ) !PrefillResult {
        if (prompt_tokens.len == 0) return error.EmptyPrefill;

        var result: PrefillResult = .{
            .last_hidden = undefined,
            .chunk_count = 0,
            .batched_chunk_count = 0,
        };
        var cursor: usize = 0;
        while (cursor < prompt_tokens.len) {
            const remaining: u32 = @intCast(prompt_tokens.len - cursor);
            const request = chunk_planner.Request{
                .phase = .prefill,
                .remaining_tokens = remaining,
            };
            var plan = self.planner.plan(request);
            var hidden_batch: ?[]f32 = null;
            var token_ids_batch: ?[]i32 = null;

            while (plan.uses_batch) {
                const chunk_tokens: usize = @intCast(plan.batch_tokens);
                hidden_batch = self.allocator.alloc(f32, chunk_tokens * MHC_MULT * DIM) catch null;
                if (hidden_batch != null) {
                    token_ids_batch = self.allocator.alloc(i32, chunk_tokens) catch null;
                    if (token_ids_batch != null) break;
                    self.allocator.free(hidden_batch.?);
                    hidden_batch = null;
                }
                plan = self.planner.shrinkBeforeExecution(request, plan);
            }
            defer if (hidden_batch) |buffer| self.allocator.free(buffer);
            defer if (token_ids_batch) |buffer| self.allocator.free(buffer);

            const absolute_position = base_position + cursor;
            std.log.info(
                "native_engine: prefill chunk index={d} start={d} size={d} remaining={d} batch={any} reason={s} flags=mla:{d},buckets:{d},experts:{d} workspace={d} mla_workspace_budget={d} pool={d} host={d}",
                .{
                    result.chunk_count,
                    absolute_position,
                    plan.batch_tokens,
                    remaining,
                    plan.uses_batch,
                    @tagName(plan.reason),
                    self.batch_config.enable_batch_mla,
                    self.batch_config.enable_expert_buckets,
                    self.batch_config.enable_batched_expert_kernels,
                    plan.estimated_workspace_bytes,
                    plan.reserved_mla_workspace_bytes,
                    plan.reserved_pool_bytes,
                    plan.estimated_host_bytes,
                },
            );

            if (plan.uses_batch) {
                if (exact_q4.enabled) return error.ExactQ4BatchPathUnsupported;
                const chunk_tokens: usize = @intCast(plan.batch_tokens);
                const hidden = hidden_batch orelse return error.PrefillAllocationFailed;
                const token_ids = token_ids_batch orelse return error.PrefillAllocationFailed;
                for (0..chunk_tokens) |token| {
                    const id = prompt_tokens[cursor + token];
                    token_ids[token] = @intCast(id);
                    metal.setTokenId(self.engine, @intCast(id));
                    metal.embed(
                        self.engine,
                        @intCast(id),
                        hidden[token * MHC_MULT * DIM ..][0 .. MHC_MULT * DIM],
                    );
                }
                try metal.forwardPlannedBatch(
                    self.engine,
                    hidden,
                    chunk_tokens,
                    @intCast(absolute_position),
                    token_ids,
                );
                @memcpy(
                    &result.last_hidden,
                    hidden[(chunk_tokens - 1) * MHC_MULT * DIM ..][0 .. MHC_MULT * DIM],
                );
                cursor += chunk_tokens;
                result.batched_chunk_count += 1;
            } else {
                const id = prompt_tokens[cursor];
                metal.setTokenId(self.engine, @intCast(id));
                metal.embed(self.engine, @intCast(id), &result.last_hidden);
                try exact_q4.beginToken(self, false);
                try metal.forward(
                    self.engine,
                    result.last_hidden[0..],
                    @intCast(absolute_position),
                );
                cursor += 1;
            }
            result.chunk_count += 1;
        }
        std.log.info(
            "native_engine: prefill complete tokens={d} chunks={d} batched_chunks={d}",
            .{ prompt_tokens.len, result.chunk_count, result.batched_chunk_count },
        );
        return result;
    }

    const DSparkVerifyOutcome = struct {
        used_batch: bool = false,
        candidates: usize = 0,
        accepted: usize = 0,
        generated: usize = 0,
        hit_eos: bool = false,
    };

    /// Transactionally verify one DSpark proposal chunk. The planner's token
    /// count includes the already-emitted, not-yet-forwarded anchor, so the
    /// immutable max_verify_chunk=5 admits at most four draft candidates.
    /// Candidate tokens remain private until DSpark, target KV/recurrent state,
    /// and request/sampler state have all committed the same accepted prefix.
    fn verifyDSparkCandidates(
        self: *NativeEngine,
        anchor_token: u32,
        tokens: []u32,
        current_len: *usize,
        start_pos: *usize,
        remaining: *usize,
        externally_emitted_tokens: usize,
        current_hidden: []f32,
        sampler: *sampling_mod.SamplerConfig,
        telemetry: *RequestTelemetry,
        draft_tokens: *[DSPARK_MAX_BLOCK]u32,
        draft_logits: []f32,
    ) !DSparkVerifyOutcome {
        const dspark_engine = self.dspark_engine orelse return .{};
        if (remaining.* == 0 or current_len.* >= tokens.len) return .{};

        const planner_remaining: u32 = @intCast(@min(
            remaining.* + 1,
            DSPARK_MAX_BLOCK + 1,
        ));
        const request = chunk_planner.Request{
            .phase = .verify,
            .remaining_tokens = planner_remaining,
        };
        var plan = self.planner.plan(request);
        var verify_hidden: ?[]f32 = null;
        var verify_token_ids: ?[]i32 = null;
        while (plan.uses_batch) {
            const planned_tokens: usize = @intCast(plan.batch_tokens);
            verify_hidden = self.allocator.alloc(f32, planned_tokens * MHC_MULT * DIM) catch null;
            if (verify_hidden != null) {
                verify_token_ids = self.allocator.alloc(i32, planned_tokens) catch null;
                if (verify_token_ids != null) break;
                self.allocator.free(verify_hidden.?);
                verify_hidden = null;
            }
            plan = self.planner.shrinkBeforeExecution(request, plan);
        }
        defer if (verify_hidden) |buffer| self.allocator.free(buffer);
        defer if (verify_token_ids) |buffer| self.allocator.free(buffer);
        if (!plan.uses_batch) return .{};

        const max_drafts: usize = @min(
            @as(usize, plan.batch_tokens) - 1,
            remaining.*,
            tokens.len - current_len.*,
        );
        if (max_drafts == 0) return .{};
        const vocab: usize = @intCast(self.config.vocab_size);
        if (draft_logits.len < DSPARK_MAX_BLOCK * vocab) {
            return error.InvalidDSparkDraftBuffer;
        }

        // Proposal may mutate DSpark-local speculative state. From this point,
        // every error is request-fatal and resets both target and DSpark state;
        // no fallback retries against a partially advanced engine.
        const n_draft_raw = metal.dsparkForward(
            dspark_engine,
            null,
            @intCast(anchor_token),
            @intCast(start_pos.* - 1),
            draft_logits[0 .. DSPARK_MAX_BLOCK * vocab],
            null,
        );
        if (n_draft_raw <= 0) return .{};
        const n_proposed_raw = metal.dsparkMarkovSample(
            dspark_engine,
            draft_logits[0 .. DSPARK_MAX_BLOCK * vocab],
            @intCast(anchor_token),
            draft_logits[0 .. DSPARK_MAX_BLOCK * vocab],
            draft_tokens[0..],
        );
        const verify_len: usize = if (n_proposed_raw > 0)
            @min(@as(usize, @intCast(n_proposed_raw)), max_drafts)
        else
            0;
        if (verify_len == 0) return .{};

        const batch_len = verify_len + 1;
        const hidden_batch = verify_hidden.?[0 .. batch_len * MHC_MULT * DIM];
        const token_ids = verify_token_ids.?[0..batch_len];
        token_ids[0] = @intCast(anchor_token);
        metal.setTokenId(self.engine, @intCast(anchor_token));
        metal.embed(self.engine, @intCast(anchor_token), hidden_batch[0 .. MHC_MULT * DIM]);
        for (0..verify_len) |index| {
            const token = draft_tokens[index];
            token_ids[index + 1] = @intCast(token);
            metal.setTokenId(self.engine, @intCast(token));
            metal.embed(
                self.engine,
                @intCast(token),
                hidden_batch[(index + 1) * MHC_MULT * DIM ..][0 .. MHC_MULT * DIM],
            );
        }

        var original_hidden: [MHC_MULT * DIM]f32 = undefined;
        var request_state = DSparkVerifyRequestState{
            // The anchor is already part of the public token sequence. A target
            // commit always advances it, while this transaction stages drafts.
            .position = start_pos.* + 1,
            .token_cursor = current_len.*,
            .emitted_tokens = externally_emitted_tokens,
            .token_buffer = tokens,
            .current_hidden = current_hidden,
        };
        var request_transaction = try DSparkVerifyRequestTransaction.begin(
            &request_state,
            sampler,
            draft_tokens[0..verify_len],
            hidden_batch[MHC_MULT * DIM .. batch_len * MHC_MULT * DIM],
            &original_hidden,
        );
        var native_transaction = metal.NativeKVTransaction{};
        metal.initKvTransaction(&native_transaction);
        errdefer {
            if (request_transaction.active) {
                request_transaction.abort(&request_state, sampler) catch {};
            }
            if (native_transaction.active == 1) {
                metal.abortVerifyBatch(self.engine, &native_transaction) catch {};
            }
            self.resetKv();
        }

        std.log.info(
            "native_engine: verify chunk start={d} batch={d} drafts={d} reason={s} workspace={d} host={d}",
            .{
                start_pos.*,
                batch_len,
                verify_len,
                @tagName(plan.reason),
                plan.estimated_workspace_bytes,
                plan.estimated_host_bytes,
            },
        );
        const verification_start_ns = metal.monotonicNs();
        metal.setDSparkAccumulate(self.engine, false);
        self.setObservabilityPhase(.verify);
        const forward_result = metal.forwardVerifyBatch(
            self.engine,
            hidden_batch,
            batch_len,
            @intCast(start_pos.*),
            token_ids,
            &native_transaction,
        );
        metal.setDSparkAccumulate(self.engine, true);
        self.setObservabilityPhase(.decode);
        try forward_result;
        try request_transaction.setCommittedBaseHidden(hidden_batch[0 .. MHC_MULT * DIM]);

        var accepted: usize = 0;
        var rejection_row: ?usize = null;
        var accepted_eos = false;
        for (0..verify_len) |index| {
            var compressed: [DIM]f32 = undefined;
            metal.hyperHeadCompress(
                self.weight_store.weights.hc_head_fn.ptr,
                self.weight_store.weights.hc_head_base.ptr,
                self.weight_store.weights.hc_head_scale.ptr,
                hidden_batch[index * MHC_MULT * DIM ..][0 .. MHC_MULT * DIM],
                &compressed,
            );
            try metal.getLogits(self.engine, &compressed, self.logits_buffer.ptr);
            const target = sampleFromLogits(self.logits_buffer, self.config.vocab_size, sampler);
            try request_transaction.recordSampledRow(index, sampler);
            if (target != draft_tokens[index]) {
                rejection_row = index;
                break;
            }
            accepted += 1;
            if (target == self.eos_token) {
                accepted_eos = true;
                break;
            }
        }

        const ProductionFinalizeContext = struct {
            owner: *NativeEngine,
            native_transaction: *metal.NativeKVTransaction,
            request_transaction: *DSparkVerifyRequestTransaction,
            request_state: *DSparkVerifyRequestState,
            sampler: *sampling_mod.SamplerConfig,
            tokens: []u32,
            current_len: *usize,
            start_pos: *usize,
            remaining: *usize,
            hidden_batch: []f32,
            telemetry: *RequestTelemetry,

            fn commitDSpark(context: *@This(), count: usize) !void {
                try metal.dsparkCommit(
                    context.owner.engine,
                    @intCast(count),
                    @intCast(context.start_pos.*),
                );
            }

            fn commitTarget(context: *@This(), rows: usize) !void {
                try metal.commitVerifyBatch(
                    context.owner.engine,
                    context.native_transaction,
                    rows,
                );
            }

            fn commitRequest(context: *@This(), count: usize) !void {
                try context.request_transaction.commitPrefix(
                    count,
                    context.request_state,
                    context.sampler,
                );
                context.current_len.* = context.request_state.token_cursor;
                context.start_pos.* = context.request_state.position;
                context.remaining.* -= count;
                for (0..count) |_| context.telemetry.recordToken();
            }

            fn canAppend(context: *const @This()) bool {
                return context.remaining.* > 0 and
                    context.current_len.* < context.tokens.len;
            }

            fn samplePostCommit(
                context: *@This(),
                kind: DSparkPostCommitSampleKind,
                row: usize,
            ) !u32 {
                _ = kind;
                var compressed: [DIM]f32 = undefined;
                metal.hyperHeadCompress(
                    context.owner.weight_store.weights.hc_head_fn.ptr,
                    context.owner.weight_store.weights.hc_head_base.ptr,
                    context.owner.weight_store.weights.hc_head_scale.ptr,
                    context.hidden_batch[row * MHC_MULT * DIM ..][0 .. MHC_MULT * DIM],
                    &compressed,
                );
                try metal.getLogits(
                    context.owner.engine,
                    &compressed,
                    context.owner.logits_buffer.ptr,
                );
                return NativeEngine.sampleFromLogits(
                    context.owner.logits_buffer,
                    context.owner.config.vocab_size,
                    context.sampler,
                );
            }

            fn appendToken(context: *@This(), token: u32) !void {
                if (!context.canAppend()) return error.InvalidVerifyOutputCapacity;
                context.tokens[context.current_len.*] = token;
                context.current_len.* += 1;
                context.remaining.* -= 1;
                context.telemetry.recordToken();
            }

            fn isEos(context: *const @This(), token: u32) bool {
                return token == context.owner.eos_token;
            }
        };

        // All three owners commit before correction/bonus sampling. Any later
        // failure trips the request-fatal errdefer above and resets both engines.
        var finalize_context = ProductionFinalizeContext{
            .owner = self,
            .native_transaction = &native_transaction,
            .request_transaction = &request_transaction,
            .request_state = &request_state,
            .sampler = sampler,
            .tokens = tokens,
            .current_len = current_len,
            .start_pos = start_pos,
            .remaining = remaining,
            .hidden_batch = hidden_batch,
            .telemetry = telemetry,
        };
        const finalized = try finalizeDSparkVerification(
            &finalize_context,
            verify_len,
            accepted,
            rejection_row,
            accepted_eos,
        );
        const generated = finalized.generated;
        const hit_eos = finalized.hit_eos;

        const verification_end_ns = metal.monotonicNs();
        telemetry.recordDSparkVerification(
            verify_len,
            accepted,
            verification_end_ns - verification_start_ns,
        );
        return .{
            .used_batch = true,
            .candidates = verify_len,
            .accepted = accepted,
            .generated = generated,
            .hit_eos = hit_eos,
        };
    }

    // ------------------------------------------------------------------
    // Generate API (compatible with DSV4Model.generate)
    //
    // NOTE: `caches` and `stream` parameters are accepted for API compatibility
    // with the existing engine loop. In native mode, the Metal engine manages
    // its own KV cache (f16) and does not use MLX streams. Expert weights are
    // loaded on-demand via packed binary pread (no SMELT preloading needed).
    // ------------------------------------------------------------------

    pub fn generate(
        self: *NativeEngine,
        prompt_tokens: []const u32,
        max_new_tokens: usize,
        sampler_config: *sampling_mod.SamplerConfig,
        caches: anytype,
        stream: anytype,
        start_pos_override: ?usize,
        cancel_flag: ?*const std.atomic.Value(bool),
    ) ![]u32 {
        _ = caches;
        _ = stream;

        std.log.info("native_engine: prompt_tokens={any}", .{prompt_tokens});
        const allocator = self.allocator;
        var exact_q4 = try ExactQ4Request.begin(self);
        errdefer exact_q4.abortBeforeReset(self);
        var decode_timeline = try DecodeTimelineRequest.begin(self);
        errdefer if (decode_timeline.enabled) metal.decodeTimelineReset(self.engine) catch {};
        if (max_new_tokens == 0) {
            const result = try allocator.alloc(u32, 0);
            try decode_timeline.resetEmpty(self);
            exact_q4.finalize(self);
            return result;
        }
        var request_telemetry = RequestTelemetry.init(self.telemetry_enabled);
        try self.beginRoutingTrace();
        defer self.endRoutingTrace();
        if (self.telemetry_enabled) metal.telemetryReset(self.engine);
        self.setObservabilityPhase(.prefill);

        // Skip BOS (token 0) and EOS (token 1) if present — MLX strips both before generation.
        const has_bos = prompt_tokens.len > 0 and prompt_tokens[0] == BOS_TOKEN;
        const has_eos = prompt_tokens.len > 0 and prompt_tokens[prompt_tokens.len - 1] == self.eos_token;
        const prompt_offset: usize = if (has_bos) 1 else 0;
        const eos_offset: usize = if (has_eos) 1 else 0;
        const effective_prompt_len = if (prompt_tokens.len > prompt_offset + eos_offset)
            prompt_tokens.len - prompt_offset - eos_offset
        else
            0;
        var tokens = try allocator.alloc(u32, effective_prompt_len + max_new_tokens);
        defer allocator.free(tokens);
        if (effective_prompt_len > 0) {
            @memcpy(tokens[0..effective_prompt_len], prompt_tokens[prompt_offset .. prompt_tokens.len - eos_offset]);
        }

        var current_len = effective_prompt_len;
        var start_pos: usize = start_pos_override orelse 0;

        // Reset KV cache at the start of each new sequence
        self.resetKv();

        // Prefill uses the same planned chunk loop as the HTTP callback path.
        // The first sampled token is emitted only after every chunk commits.
        if (start_pos_override == null and effective_prompt_len > 0) {
            const prefill = try self.prefillPrompt(
                &exact_q4,
                tokens[0..effective_prompt_len],
                start_pos,
            );
            const logits_sample_begin_ns = if (exact_q4.enabled) metal.monotonicNs() else 0;
            var compressed: [DIM]f32 = undefined;
            metal.hyperHeadCompress(
                self.weight_store.weights.hc_head_fn.ptr,
                self.weight_store.weights.hc_head_base.ptr,
                self.weight_store.weights.hc_head_scale.ptr,
                &prefill.last_hidden,
                &compressed,
            );
            try metal.getLogits(self.engine, &compressed, self.logits_buffer.ptr);

            const next_token = sampleFromLogits(self.logits_buffer, self.config.vocab_size, sampler_config);
            try exact_q4.appendLogitsSample(self, logits_sample_begin_ns);
            start_pos += effective_prompt_len;
            tokens[current_len] = next_token;
            current_len += 1;
            request_telemetry.recordToken();
            std.log.info(
                "native_engine: effective_prompt_len={d} first_token={d} chunks={d} batched_chunks={d}",
                .{
                    effective_prompt_len,
                    next_token,
                    prefill.chunk_count,
                    prefill.batched_chunk_count,
                },
            );
            if (next_token == self.eos_token) {
                std.log.info("native_engine: EOS after prefill", .{});
            }
        }

        // Decode loop — signal SMELT that prefill is done (enables decode token counting for warmup)
        metal.smeltSetDecodePhase(self.engine);
        self.setObservabilityPhase(.decode);
        request_telemetry.markDecodeStart();
        const decode_count = if (start_pos_override != null) max_new_tokens else max_new_tokens - 1;
        var _t_forward_total: u64 = 0;
        var _t_logits_total: u64 = 0;
        var _t_decode_n: u64 = 0;
        const _do_decode_time = (std.c.getenv("NATIVE_DECODE_TIME") != null);

        // DSpark speculative decoding buffers (stack-allocated for block_size <= 8)
        var dspark_draft_tokens: [DSPARK_MAX_BLOCK]u32 = undefined;
        var dspark_draft_logits_buf: ?[]f32 = null;
        defer if (dspark_draft_logits_buf) |buf| self.allocator.free(buf);
        var dspark_total_drafted: usize = 0;
        var dspark_total_accepted: usize = 0;
        var dspark_total_steps: usize = 0;
        // NATIVE_DSPARK_EVAL=1: load the DSpark engine but do NOT speculate —
        // measure backbone position-0 top-1 accuracy per decode step instead
        // (DSpark restart milestone: top-1 >= 40% on real decode).
        const dspark_eval = std.c.getenv("NATIVE_DSPARK_EVAL") != null;
        var eval_hits: usize = 0;
        var eval_total: usize = 0;
        var eval_rank_sum: usize = 0;
        var eval_markov_hits: usize = 0;
        var eval_corrected_buf: ?[]f32 = null;
        if (self.dspark_engine != null or self.dspark != null) {
            dspark_draft_logits_buf = try self.allocator.alloc(f32, DSPARK_MAX_BLOCK * @as(usize, self.config.vocab_size));
            if (dspark_eval) eval_corrected_buf = try self.allocator.alloc(f32, DSPARK_MAX_BLOCK * @as(usize, self.config.vocab_size));
        }

        var remaining = decode_count;
        while (remaining > 0) {
            if (cancel_flag) |f| {
                if (f.load(.acquire)) break;
            }
            if (current_len >= tokens.len) break;

            // --- Standard single-token decode step (produces anchor logits) ---
            var hidden: [MHC_MULT * DIM]f32 = undefined;
            metal.setTokenId(self.engine, @intCast(tokens[current_len - 1]));
            metal.embed(self.engine, @intCast(tokens[current_len - 1]), &hidden);
            try exact_q4.beginToken(self, true);
            try decode_timeline.beginToken(self);
            const token_forward_begin_ns = if (decode_timeline.enabled)
                metal.decodeTimelineNowNs()
            else
                0;
            const _t0 = if (_do_decode_time) std.c.mach_absolute_time() else 0;
            try metal.forward(self.engine, hidden[0..], @intCast(start_pos));
            const _t1 = if (_do_decode_time) std.c.mach_absolute_time() else 0;

            const logits_sample_begin_ns = if (exact_q4.enabled) metal.monotonicNs() else 0;
            var compressed: [DIM]f32 = undefined;
            metal.hyperHeadCompress(
                self.weight_store.weights.hc_head_fn.ptr,
                self.weight_store.weights.hc_head_base.ptr,
                self.weight_store.weights.hc_head_scale.ptr,
                &hidden,
                &compressed,
            );
            const next_token = try self.sampleGreedyToken(&compressed, sampler_config);
            const _t2 = if (_do_decode_time) std.c.mach_absolute_time() else 0;
            if (_do_decode_time) {
                _t_forward_total += _t1 - _t0;
                _t_logits_total += _t2 - _t1;
                _t_decode_n += 1;
            }

            const sample_begin_ns = if (decode_timeline.enabled)
                metal.decodeTimelineNowNs()
            else
                0;
            try decode_timeline.appendSample(self, sample_begin_ns);
            try decode_timeline.appendTokenForward(self, token_forward_begin_ns);
            try exact_q4.appendLogitsSample(self, logits_sample_begin_ns);
            tokens[current_len] = next_token;
            current_len += 1;
            request_telemetry.recordToken();
            start_pos += 1;
            remaining -= 1;

            if (next_token == self.eos_token) {
                std.log.info("native_engine: EOS token generated, stopping at pos={d}", .{start_pos});
                break;
            }

            // --- DSpark backbone position-0 top-1 evaluation ---
            // Backbone prediction from (anchor token fed this step, this step's
            // accumulated main_hidden) vs the token the target actually sampled.
            if (dspark_eval) {
                if (self.dspark_engine) |de| {
                    if (dspark_draft_logits_buf) |draft_buf| {
                        const vocab: usize = @intCast(self.config.vocab_size);
                        const anchor: u32 = tokens[current_len - 2];
                        const n = metal.dsparkForward(de, null, @intCast(anchor), @intCast(start_pos - 1), draft_buf[0..vocab], null);
                        if (n > 0) {
                            var pred: u32 = 0;
                            var best: f32 = -std.math.inf(f32);
                            var actual_rank: usize = 1;
                            const actual_logit = draft_buf[next_token];
                            for (draft_buf[0..vocab], 0..) |l, i| {
                                if (l > best) {
                                    best = l;
                                    pred = @intCast(i);
                                }
                                if (l > actual_logit) actual_rank += 1;
                            }
                            eval_total += 1;
                            if (pred == next_token) eval_hits += 1;
                            eval_rank_sum += actual_rank;
                            // Markov-corrected top-1 (reference forward_head adds the
                            // Markov bias before sampling — system-level accuracy)
                            var m_pred: u32 = pred;
                            if (eval_corrected_buf) |cbuf| {
                                const blk = DSPARK_MAX_BLOCK * vocab;
                                @memcpy(cbuf[0..blk], draft_buf[0..blk]);
                                var toks: [8]u32 = undefined;
                                const nm = metal.dsparkMarkovSample(de, cbuf[0..vocab], @intCast(anchor), cbuf[0..vocab], toks[0..8]);
                                if (nm > 0) m_pred = toks[0];
                            }
                            if (m_pred == next_token) eval_markov_hits += 1;
                            if (eval_total % 10 == 0) {
                                std.log.info("[dspark-eval] top1={d}/{d} ({d:.1}%) markov={d}/{d} ({d:.1}%) avg_rank={d:.0} last pred={d} actual={d}(rank {d})", .{
                                    eval_hits,
                                    eval_total,
                                    @as(f64, @floatFromInt(eval_hits)) * 100.0 / @as(f64, @floatFromInt(eval_total)),
                                    eval_markov_hits,
                                    eval_total,
                                    @as(f64, @floatFromInt(eval_markov_hits)) * 100.0 / @as(f64, @floatFromInt(eval_total)),
                                    @as(f64, @floatFromInt(eval_rank_sum)) / @as(f64, @floatFromInt(eval_total)),
                                    pred,
                                    next_token,
                                    actual_rank,
                                });
                            }
                        }
                    }
                }
            }

            // --- DSpark speculative decoding (propose + transactional verify) ---
            // The full DSpark engine is the only production speculative source.
            // Markov-only remains loaded for evaluation but cannot bypass the
            // C KV/recurrent transaction or the planner's anchor-inclusive cap.
            if (!dspark_eval and self.batch_config.enable_dspark_batch_verify == 1) {
                if (dspark_draft_logits_buf) |draft_buf| {
                    if (exact_q4.enabled)
                        return error.ExactQ4BatchPathUnsupported;
                    const outcome = try self.verifyDSparkCandidates(
                        next_token,
                        tokens,
                        &current_len,
                        &start_pos,
                        &remaining,
                        0,
                        hidden[0..],
                        sampler_config,
                        &request_telemetry,
                        &dspark_draft_tokens,
                        draft_buf,
                    );
                    if (outcome.used_batch) {
                        dspark_total_drafted += outcome.candidates;
                        dspark_total_accepted += outcome.accepted;
                        dspark_total_steps += 1;
                    }
                    if (outcome.hit_eos) break;
                }
            } // end opt-in DSpark candidate verification
        }

        if (_do_decode_time and _t_decode_n > 0) {
            const fwd_ms = @as(u64, @intFromFloat(@as(f64, @floatFromInt(_t_forward_total)) * 125.0 / 3_000_000.0 / @as(f64, @floatFromInt(_t_decode_n))));
            const log_ms = @as(u64, @intFromFloat(@as(f64, @floatFromInt(_t_logits_total)) * 125.0 / 3_000_000.0 / @as(f64, @floatFromInt(_t_decode_n))));
            std.log.info("[DECODE_TIME] n={d} forward={d}ms logits={d}ms per_tok={d}ms", .{
                _t_decode_n, fwd_ms, log_ms, fwd_ms + log_ms,
            });
        }

        if (eval_total > 0) {
            std.log.info("[dspark-eval] FINAL backbone top-1 = {d}/{d} = {d:.1}% | markov top-1 = {d}/{d} = {d:.1}% (restart milestone: >=40%)", .{
                eval_hits,
                eval_total,
                @as(f64, @floatFromInt(eval_hits)) * 100.0 / @as(f64, @floatFromInt(eval_total)),
                eval_markov_hits,
                eval_total,
                @as(f64, @floatFromInt(eval_markov_hits)) * 100.0 / @as(f64, @floatFromInt(eval_total)),
            });
        }

        if (dspark_total_steps > 0) {
            std.log.info("[dspark-stats] steps={d} drafted={d} accepted={d} rate={d:.1}%", .{
                dspark_total_steps,
                dspark_total_drafted,
                dspark_total_accepted,
                if (dspark_total_drafted > 0)
                    @as(f64, @floatFromInt(dspark_total_accepted)) * 100.0 / @as(f64, @floatFromInt(dspark_total_drafted))
                else
                    0.0,
            });
        }

        // Return only generated tokens (skip effective prompt), and drop trailing EOS.
        var result_len = current_len - effective_prompt_len;
        if (result_len > 0 and tokens[current_len - 1] == self.eos_token) {
            result_len -= 1;
        }
        const result = try allocator.alloc(u32, result_len);
        @memcpy(result, tokens[effective_prompt_len .. effective_prompt_len + result_len]);
        try decode_timeline.finalize(self);
        exact_q4.finalize(self);
        request_telemetry.emit(self.engine, effective_prompt_len, result_len);
        return result;
    }

    /// Streaming generate with callback (compatible with DSV4Model.generateWithCallback)
    pub fn generateWithCallback(
        self: *NativeEngine,
        prompt_tokens: []const u32,
        max_new_tokens: usize,
        sampler_config: *sampling_mod.SamplerConfig,
        caches: anytype,
        stream: anytype,
        callback_ctx: ?*anyopaque,
        callback: ?StreamCallback,
        start_pos_override: ?usize,
        cancel_flag: ?*const std.atomic.Value(bool),
    ) ![]u32 {
        _ = caches;
        _ = stream;

        const allocator = self.allocator;
        var exact_q4 = try ExactQ4Request.begin(self);
        errdefer exact_q4.abortBeforeReset(self);
        var decode_timeline = try DecodeTimelineRequest.begin(self);
        errdefer if (decode_timeline.enabled) metal.decodeTimelineReset(self.engine) catch {};
        if (max_new_tokens == 0) {
            const result = try allocator.alloc(u32, 0);
            try decode_timeline.resetEmpty(self);
            exact_q4.finalize(self);
            return result;
        }
        var request_telemetry = RequestTelemetry.init(self.telemetry_enabled);
        try self.beginRoutingTrace();
        defer self.endRoutingTrace();
        if (self.telemetry_enabled) metal.telemetryReset(self.engine);
        self.setObservabilityPhase(.prefill);

        // Skip BOS (token 0) and EOS (token 1) if present — MLX strips both before generation.
        const has_bos = prompt_tokens.len > 0 and prompt_tokens[0] == BOS_TOKEN;
        const has_eos = prompt_tokens.len > 0 and prompt_tokens[prompt_tokens.len - 1] == self.eos_token;
        const prompt_offset: usize = if (has_bos) 1 else 0;
        const eos_offset: usize = if (has_eos) 1 else 0;
        const effective_prompt_len = if (prompt_tokens.len > prompt_offset + eos_offset)
            prompt_tokens.len - prompt_offset - eos_offset
        else
            0;
        var tokens = try allocator.alloc(u32, effective_prompt_len + max_new_tokens);
        defer allocator.free(tokens);
        if (effective_prompt_len > 0) {
            @memcpy(tokens[0..effective_prompt_len], prompt_tokens[prompt_offset .. prompt_tokens.len - eos_offset]);
        }

        var current_len = effective_prompt_len;
        var start_pos: usize = start_pos_override orelse 0;

        // Reset KV cache
        self.resetKv();

        // HTTP streaming shares the exact planned prefill implementation. No
        // callback is published until all prompt chunks commit successfully.
        if (start_pos_override == null and effective_prompt_len > 0) {
            const prefill = try self.prefillPrompt(
                &exact_q4,
                tokens[0..effective_prompt_len],
                start_pos,
            );
            const logits_sample_begin_ns = if (exact_q4.enabled) metal.monotonicNs() else 0;
            var compressed: [DIM]f32 = undefined;
            metal.hyperHeadCompress(
                self.weight_store.weights.hc_head_fn.ptr,
                self.weight_store.weights.hc_head_base.ptr,
                self.weight_store.weights.hc_head_scale.ptr,
                &prefill.last_hidden,
                &compressed,
            );
            try metal.getLogits(self.engine, &compressed, self.logits_buffer.ptr);

            const next_token = sampleFromLogits(self.logits_buffer, self.config.vocab_size, sampler_config);
            try exact_q4.appendLogitsSample(self, logits_sample_begin_ns);
            start_pos += effective_prompt_len;
            tokens[current_len] = next_token;
            current_len += 1;
            request_telemetry.recordToken();
            std.log.info(
                "native_engine: effective_prompt_len={d} first_token={d} chunks={d} batched_chunks={d} [http]",
                .{
                    effective_prompt_len,
                    next_token,
                    prefill.chunk_count,
                    prefill.batched_chunk_count,
                },
            );

            if (callback) |cb| {
                const is_final = max_new_tokens == 1 or
                    next_token == self.eos_token;
                cb(callback_ctx.?, next_token, is_final);
            }
            if (next_token == self.eos_token) {
                std.log.info("native_engine: EOS after prefill", .{});
            }
        }

        // Decode loop — signal SMELT that prefill is done (enables decode token counting for warmup)
        metal.smeltSetDecodePhase(self.engine);
        self.setObservabilityPhase(.decode);
        request_telemetry.markDecodeStart();
        const decode_count = if (start_pos_override != null) max_new_tokens else max_new_tokens - 1;
        var remaining = if (current_len > effective_prompt_len and
            tokens[current_len - 1] == self.eos_token)
            @as(usize, 0)
        else
            decode_count;
        var dspark_draft_tokens: [DSPARK_MAX_BLOCK]u32 = undefined;
        var dspark_draft_logits_buf: ?[]f32 = null;
        defer if (dspark_draft_logits_buf) |buffer| self.allocator.free(buffer);
        if (self.dspark_engine != null and
            self.batch_config.enable_dspark_batch_verify == 1)
        {
            dspark_draft_logits_buf = try self.allocator.alloc(
                f32,
                DSPARK_MAX_BLOCK * @as(usize, self.config.vocab_size),
            );
        }

        while (remaining > 0) {
            if (cancel_flag) |f| {
                if (f.load(.acquire)) break;
            }
            if (current_len >= tokens.len) break;

            var hidden: [MHC_MULT * DIM]f32 = undefined;
            metal.setTokenId(self.engine, @intCast(tokens[current_len - 1]));
            metal.embed(self.engine, @intCast(tokens[current_len - 1]), &hidden);
            try exact_q4.beginToken(self, true);
            try decode_timeline.beginToken(self);
            const token_forward_begin_ns = if (decode_timeline.enabled)
                metal.decodeTimelineNowNs()
            else
                0;
            try metal.forward(self.engine, hidden[0..], @intCast(start_pos));

            const logits_sample_begin_ns = if (exact_q4.enabled) metal.monotonicNs() else 0;
            var compressed: [DIM]f32 = undefined;
            metal.hyperHeadCompress(
                self.weight_store.weights.hc_head_fn.ptr,
                self.weight_store.weights.hc_head_base.ptr,
                self.weight_store.weights.hc_head_scale.ptr,
                &hidden,
                &compressed,
            );
            const next_token = try self.sampleGreedyToken(&compressed, sampler_config);

            const sample_begin_ns = if (decode_timeline.enabled)
                metal.decodeTimelineNowNs()
            else
                0;
            try decode_timeline.appendSample(self, sample_begin_ns);
            try decode_timeline.appendTokenForward(self, token_forward_begin_ns);
            try exact_q4.appendLogitsSample(self, logits_sample_begin_ns);
            tokens[current_len] = next_token;
            current_len += 1;
            request_telemetry.recordToken();
            start_pos += 1;
            remaining -= 1;

            if (callback) |cb| {
                const is_final = remaining == 0 or next_token == self.eos_token;
                cb(callback_ctx.?, next_token, is_final);
            }
            if (next_token == self.eos_token) {
                std.log.info("native_engine: EOS token generated, stopping", .{});
                break;
            }

            if (self.batch_config.enable_dspark_batch_verify == 1) {
                if (dspark_draft_logits_buf) |draft_buffer| {
                    if (exact_q4.enabled)
                        return error.ExactQ4BatchPathUnsupported;
                    const unpublished_start = current_len;
                    const emitted_floor = if (callback != null)
                        current_len - effective_prompt_len
                    else
                        0;
                    const outcome = try self.verifyDSparkCandidates(
                        next_token,
                        tokens,
                        &current_len,
                        &start_pos,
                        &remaining,
                        emitted_floor,
                        hidden[0..],
                        sampler_config,
                        &request_telemetry,
                        &dspark_draft_tokens,
                        draft_buffer,
                    );
                    // The helper returns only after all three transaction owners
                    // commit, so no rejected candidate can cross the callback
                    // emission floor.
                    if (callback) |cb| {
                        publishCommittedDSparkTokens(
                            tokens,
                            unpublished_start,
                            current_len,
                            remaining,
                            self.eos_token,
                            cb,
                            callback_ctx.?,
                        );
                    }
                    if (outcome.hit_eos) break;
                }
            }
        }

        // Return only generated tokens (skip effective prompt), and drop trailing EOS.
        var result_len = current_len - effective_prompt_len;
        if (result_len > 0 and tokens[current_len - 1] == self.eos_token) {
            result_len -= 1;
        }
        const result = try allocator.alloc(u32, result_len);
        @memcpy(result, tokens[effective_prompt_len .. effective_prompt_len + result_len]);
        try decode_timeline.finalize(self);
        exact_q4.finalize(self);
        request_telemetry.emit(self.engine, effective_prompt_len, result_len);
        return result;
    }

    pub const StreamCallback = *const fn (ctx: *anyopaque, token: u32, is_final: bool) void;

    /// Publish only the token range made durable by the shared verifier. Keeping
    /// this loop in one helper makes the post-commit callback boundary directly
    /// testable and prevents either generation API from exposing draft rows.
    fn publishCommittedDSparkTokens(
        tokens: []const u32,
        start: usize,
        end: usize,
        remaining: usize,
        eos_token: u32,
        callback: StreamCallback,
        callback_ctx: *anyopaque,
    ) void {
        std.debug.assert(start <= end and end <= tokens.len);
        for (tokens[start..end], 0..) |token, index| {
            const is_final = (index + 1 == end - start) and
                (remaining == 0 or token == eos_token);
            callback(callback_ctx, token, is_final);
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// Simple greedy argmax sampling (temperature=0).
    /// When temperature > 0 is needed, this falls back to greedy for now.
    fn sampleFromLogits(logits: []f32, vocab_size: u32, sampler: *sampling_mod.SamplerConfig) u32 {
        _ = sampler;
        var max_idx: usize = 0;
        var max_val: f32 = logits[0];
        for (1..vocab_size) |i| {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        std.log.info("sampleFromLogits: max_idx={d} max_val={d:.4} logit[0]={d:.4} logit[304]={d:.4}", .{ max_idx, max_val, logits[0], logits[304] });
        return @intCast(max_idx);
    }

    /// Greedy next-token selection for the decode loop. Prefers the GPU
    /// two-stage argmax (8-byte readback, no full logits readback); falls
    /// back to getLogits + CPU scan when the GPU path is unavailable.
    fn sampleGreedyToken(self: *NativeEngine, compressed: *const [DIM]f32, sampler: *sampling_mod.SamplerConfig) !u32 {
        var idx: u32 = 0;
        var val: f32 = 0;
        if (metal.getArgmax(self.engine, compressed, &idx, &val)) {
            std.log.info("sampleGreedyToken(gpu): max_idx={d} max_val={d:.4}", .{ idx, val });
            return idx;
        } else |_| {}
        try metal.getLogits(self.engine, compressed, self.logits_buffer.ptr);
        return sampleFromLogits(self.logits_buffer, self.config.vocab_size, sampler);
    }
};

test "DSpark request transaction commits A=0,1,middle,B prefixes" {
    const staged_tokens = [_]u32{ 101, 102, 103, 104 };
    const staged_hidden = [_]f32{
        10, 11, 12, 13,
        20, 21, 22, 23,
        30, 31, 32, 33,
        40, 41, 42, 43,
    };
    for ([_]usize{ 0, 1, 2, 4 }) |accepted| {
        var token_buffer = [_]u32{0} ** 8;
        var current_hidden = [_]f32{ 1, 2, 3, 4 };
        var original_hidden: [4]f32 = undefined;
        var state = DSparkVerifyRequestState{
            .position = 7,
            .token_cursor = 2,
            .emitted_tokens = 2,
            .token_buffer = &token_buffer,
            .current_hidden = &current_hidden,
        };
        var sampler = sampling_mod.SamplerConfig.init(0x5a17);
        var transaction = try DSparkVerifyRequestTransaction.begin(
            &state,
            &sampler,
            &staged_tokens,
            &staged_hidden,
            &original_hidden,
        );
        for (0..staged_tokens.len) |row| {
            _ = sampler.prng.random().int(u64);
            try transaction.recordSampledRow(row, &sampler);
        }
        try transaction.commitPrefix(accepted, &state, &sampler);
        try std.testing.expect(!transaction.active);
        try std.testing.expectEqual(@as(usize, 7 + accepted), state.position);
        try std.testing.expectEqual(@as(usize, 2 + accepted), state.token_cursor);
        try std.testing.expectEqualSlices(
            u32,
            staged_tokens[0..accepted],
            token_buffer[2 .. 2 + accepted],
        );
        if (accepted == 0) {
            try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4 }, &current_hidden);
        } else {
            const start = (accepted - 1) * 4;
            try std.testing.expectEqualSlices(f32, staged_hidden[start..][0..4], &current_hidden);
        }

        var expected_sampler = sampling_mod.SamplerConfig.init(0x5a17);
        for (0..accepted) |_| _ = expected_sampler.prng.random().int(u64);
        try std.testing.expectEqual(
            expected_sampler.prng.random().int(u64),
            sampler.prng.random().int(u64),
        );
    }
}

test "DSpark request transaction abort restores state and honors emission floor" {
    const staged_tokens = [_]u32{ 201, 202, 203 };
    const staged_hidden = [_]f32{
        10, 11, 12, 13,
        20, 21, 22, 23,
        30, 31, 32, 33,
    };
    var token_buffer = [_]u32{0} ** 8;
    var current_hidden = [_]f32{ 1, 2, 3, 4 };
    var original_hidden: [4]f32 = undefined;
    var state = DSparkVerifyRequestState{
        .position = 9,
        .token_cursor = 3,
        .emitted_tokens = 3,
        .token_buffer = &token_buffer,
        .current_hidden = &current_hidden,
    };
    var sampler = sampling_mod.SamplerConfig.init(77);
    var transaction = try DSparkVerifyRequestTransaction.begin(
        &state,
        &sampler,
        &staged_tokens,
        &staged_hidden,
        &original_hidden,
    );
    _ = sampler.prng.random().int(u64);
    try transaction.recordSampledRow(0, &sampler);
    state.position = 12;
    state.token_cursor = 5;
    @memset(state.current_hidden, 99);
    try transaction.abort(&state, &sampler);
    try std.testing.expectEqual(@as(usize, 9), state.position);
    try std.testing.expectEqual(@as(usize, 3), state.token_cursor);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4 }, &current_hidden);

    var floor_storage: [4]f32 = undefined;
    var floor_transaction = try DSparkVerifyRequestTransaction.begin(
        &state,
        &sampler,
        &staged_tokens,
        &staged_hidden,
        &floor_storage,
    );
    state.emitted_tokens += 1;
    try std.testing.expectError(
        error.ExternalEmissionAdvanced,
        floor_transaction.abort(&state, &sampler),
    );
    try std.testing.expect(floor_transaction.active);
    state.emitted_tokens -= 1;
    try floor_transaction.abort(&state, &sampler);
}

test "DSpark request transaction rejects hidden aliases and failed commit is atomic" {
    const one_token = [_]u32{301};
    var token_buffer = [_]u32{7} ** 8;
    var current_hidden = [_]f32{ 1, 2, 3, 4 };
    var separate_snapshot: [4]f32 = undefined;
    var separate_staged = [_]f32{ 10, 11, 12, 13 };
    var state = DSparkVerifyRequestState{
        .position = 11,
        .token_cursor = 2,
        .emitted_tokens = 2,
        .token_buffer = &token_buffer,
        .current_hidden = &current_hidden,
    };
    var sampler = sampling_mod.SamplerConfig.init(91);

    try std.testing.expectError(
        error.InvalidVerifyRequestState,
        DSparkVerifyRequestTransaction.begin(
            &state,
            &sampler,
            &one_token,
            &separate_staged,
            &current_hidden,
        ),
    );
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4 }, &current_hidden);

    try std.testing.expectError(
        error.InvalidVerifyRequestState,
        DSparkVerifyRequestTransaction.begin(
            &state,
            &sampler,
            &one_token,
            &current_hidden,
            &separate_snapshot,
        ),
    );
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4 }, &current_hidden);

    var transaction = try DSparkVerifyRequestTransaction.begin(
        &state,
        &sampler,
        &one_token,
        &separate_staged,
        &separate_snapshot,
    );
    _ = sampler.prng.random().int(u64);
    try transaction.recordSampledRow(0, &sampler);
    state.position += 1;
    @memset(state.current_hidden, 88);
    var expected_sampler = sampler;
    try std.testing.expectError(
        error.InvalidVerifyCommit,
        transaction.commitPrefix(1, &state, &sampler),
    );
    try std.testing.expect(transaction.active);
    try std.testing.expectEqualSlices(u32, &([_]u32{7} ** 8), &token_buffer);
    try std.testing.expectEqualSlices(f32, &([_]f32{88} ** 4), &current_hidden);
    try std.testing.expectEqual(
        expected_sampler.prng.random().int(u64),
        sampler.prng.random().int(u64),
    );

    state.position = 11;
    try transaction.abort(&state, &sampler);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4 }, &current_hidden);
}

test "DSpark verification scope restores accumulation and phase after forward failure" {
    const MockVerificationContext = struct {
        accumulate: bool = true,
        phase: metal.TelemetryPhase = .decode,
        forward_calls: usize = 0,

        fn setAccumulate(self: *@This(), enabled: bool) void {
            self.accumulate = enabled;
        }

        fn setPhase(self: *@This(), phase: metal.TelemetryPhase) void {
            self.phase = phase;
        }

        fn forwardBatch(self: *@This()) !void {
            self.forward_calls += 1;
            try std.testing.expect(!self.accumulate);
            try std.testing.expectEqual(metal.TelemetryPhase.verify, self.phase);
            return error.InjectedForwardFailure;
        }
    };

    var context = MockVerificationContext{};
    try std.testing.expectError(
        error.InjectedForwardFailure,
        runDSparkVerificationScope(&context),
    );
    try std.testing.expect(context.accumulate);
    try std.testing.expectEqual(metal.TelemetryPhase.decode, context.phase);
    try std.testing.expectEqual(@as(usize, 1), context.forward_calls);
}

test "DSpark request transaction commits anchor hidden for A=0 but abort restores prior hidden" {
    const staged_tokens = [_]u32{ 401, 402 };
    const staged_hidden = [_]f32{ 20, 21, 30, 31 };
    const anchor_hidden = [_]f32{ 10, 11 };
    var token_buffer = [_]u32{0} ** 6;
    var current_hidden = [_]f32{ 1, 2 };
    var original_hidden: [2]f32 = undefined;
    var state = DSparkVerifyRequestState{
        .position = 5,
        .token_cursor = 2,
        .emitted_tokens = 2,
        .token_buffer = &token_buffer,
        .current_hidden = &current_hidden,
    };
    var sampler = sampling_mod.SamplerConfig.init(123);
    var transaction = try DSparkVerifyRequestTransaction.begin(
        &state,
        &sampler,
        &staged_tokens,
        &staged_hidden,
        &original_hidden,
    );
    try transaction.setCommittedBaseHidden(&anchor_hidden);
    try transaction.commitPrefix(0, &state, &sampler);
    try std.testing.expectEqualSlices(f32, &anchor_hidden, &current_hidden);
    try std.testing.expectEqual(@as(usize, 5), state.position);
    try std.testing.expectEqual(@as(usize, 2), state.token_cursor);

    current_hidden = .{ 3, 4 };
    var abort_snapshot: [2]f32 = undefined;
    var abort_transaction = try DSparkVerifyRequestTransaction.begin(
        &state,
        &sampler,
        &staged_tokens,
        &staged_hidden,
        &abort_snapshot,
    );
    try abort_transaction.setCommittedBaseHidden(&anchor_hidden);
    @memset(&current_hidden, 99);
    try abort_transaction.abort(&state, &sampler);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 3, 4 }, &current_hidden);
}

test "production DSpark finalizer covers rejection, full acceptance, bonus and post-commit publication" {
    const MockFinalizeContext = struct {
        staged_tokens: [3]u32 = .{ 501, 502, 503 },
        token_buffer: [8]u32 = [_]u32{0} ** 8,
        cursor: usize = 2,
        remaining: usize = 6,
        eos_token: u32 = 999,
        dspark_committed: ?usize = null,
        target_committed: ?usize = null,
        request_committed: ?usize = null,
        owners_committed: bool = false,
        appended: bool = false,
        published_too_early: bool = false,
        sample_kind: ?DSparkPostCommitSampleKind = null,
        sampled_random: ?u64 = null,
        appended_token: ?u32 = null,
        correction_token: u32 = 701,
        bonus_token: u32 = 702,
        sampler: std.Random.DefaultPrng,
        original_prng: std.Random.DefaultPrng,
        row_prng: [3]std.Random.DefaultPrng,
        events: [16]u8 = [_]u8{0} ** 16,
        event_count: usize = 0,
        published_tokens: [4]u32 = [_]u32{0} ** 4,
        published_count: usize = 0,

        fn push(self: *@This(), event: u8) void {
            std.debug.assert(self.event_count < self.events.len);
            self.events[self.event_count] = event;
            self.event_count += 1;
        }

        fn commitDSpark(self: *@This(), count: usize) !void {
            if (self.event_count != 0) return error.InvalidMockCommitOrder;
            self.dspark_committed = count;
            self.push('D');
        }

        fn commitTarget(self: *@This(), rows: usize) !void {
            if (self.event_count != 1 or self.events[0] != 'D')
                return error.InvalidMockCommitOrder;
            self.target_committed = rows;
            self.push('T');
        }

        fn commitRequest(self: *@This(), count: usize) !void {
            if (self.event_count != 2 or self.events[1] != 'T' or
                count > self.staged_tokens.len or count > self.remaining or
                count > self.token_buffer.len - self.cursor)
            {
                return error.InvalidMockCommitOrder;
            }
            @memcpy(
                self.token_buffer[self.cursor..][0..count],
                self.staged_tokens[0..count],
            );
            self.cursor += count;
            self.remaining -= count;
            self.sampler = if (count == 0)
                self.original_prng
            else
                self.row_prng[count - 1];
            self.request_committed = count;
            self.owners_committed = true;
            self.push('R');
        }

        fn canAppend(self: *const @This()) bool {
            return self.remaining > 0 and self.cursor < self.token_buffer.len;
        }

        fn samplePostCommit(
            self: *@This(),
            kind: DSparkPostCommitSampleKind,
            row: usize,
        ) !u32 {
            if (!self.owners_committed or row > self.staged_tokens.len)
                return error.SampleBeforeCommit;
            self.sample_kind = kind;
            self.sampled_random = self.sampler.random().int(u64);
            self.push(if (kind == .correction) 'C' else 'B');
            return if (kind == .correction) self.correction_token else self.bonus_token;
        }

        fn appendToken(self: *@This(), token: u32) !void {
            if (!self.owners_committed or !self.canAppend())
                return error.AppendBeforeCommit;
            self.token_buffer[self.cursor] = token;
            self.cursor += 1;
            self.remaining -= 1;
            self.appended = true;
            self.appended_token = token;
            self.push('A');
        }

        fn isEos(self: *const @This(), token: u32) bool {
            return token == self.eos_token;
        }

        fn callback(raw: *anyopaque, token: u32, is_final: bool) void {
            _ = is_final;
            const self: *@This() = @ptrCast(@alignCast(raw));
            if (!self.owners_committed or !self.appended) {
                self.published_too_early = true;
            }
            std.debug.assert(self.published_count < self.published_tokens.len);
            self.published_tokens[self.published_count] = token;
            self.published_count += 1;
            self.push('P');
        }
    };

    const Scenario = struct {
        fn run(
            accepted: usize,
            rejection_row: ?usize,
            expected_kind: DSparkPostCommitSampleKind,
            expected_appended_token: u32,
        ) !void {
            var context = MockFinalizeContext{
                .sampler = std.Random.DefaultPrng.init(600),
                .original_prng = std.Random.DefaultPrng.init(600),
                .row_prng = .{
                    std.Random.DefaultPrng.init(700),
                    std.Random.DefaultPrng.init(701),
                    std.Random.DefaultPrng.init(702),
                },
            };
            const publish_start = context.cursor;
            const outcome = try finalizeDSparkVerification(
                &context,
                context.staged_tokens.len,
                accepted,
                rejection_row,
                false,
            );

            try std.testing.expectEqual(accepted, context.dspark_committed.?);
            try std.testing.expectEqual(accepted + 1, context.target_committed.?);
            try std.testing.expectEqual(accepted, context.request_committed.?);
            try std.testing.expectEqual(expected_kind, context.sample_kind.?);
            try std.testing.expectEqual(expected_appended_token, context.appended_token.?);
            try std.testing.expectEqual(accepted + 1, outcome.generated);
            try std.testing.expect(!outcome.hit_eos);
            try std.testing.expectEqualSlices(u8, "DTR", context.events[0..3]);
            try std.testing.expectEqual(
                if (expected_kind == .correction) @as(u8, 'C') else @as(u8, 'B'),
                context.events[3],
            );
            try std.testing.expectEqual(@as(u8, 'A'), context.events[4]);

            var expected_prng = std.Random.DefaultPrng.init(700 + accepted - 1);
            try std.testing.expectEqual(
                expected_prng.random().int(u64),
                context.sampled_random.?,
            );

            NativeEngine.publishCommittedDSparkTokens(
                &context.token_buffer,
                publish_start,
                context.cursor,
                context.remaining,
                context.eos_token,
                MockFinalizeContext.callback,
                &context,
            );
            try std.testing.expect(!context.published_too_early);
            try std.testing.expectEqual(outcome.generated, context.published_count);
            try std.testing.expectEqualSlices(
                u32,
                context.token_buffer[publish_start..context.cursor],
                context.published_tokens[0..context.published_count],
            );
            for (context.events[5..context.event_count]) |event| {
                try std.testing.expectEqual(@as(u8, 'P'), event);
            }
        }
    };

    // A=1 rejects row 1 and samples a correction from row-0's committed PRNG.
    try Scenario.run(1, 1, .correction, 701);
    // A=B commits all three drafts, then samples and publishes the bonus token.
    try Scenario.run(3, null, .bonus, 702);
}
