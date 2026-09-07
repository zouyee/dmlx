const std = @import("std");

/// Task 5.1 planner constants mirror the validated native engine contract.
/// The Wave-4 grouped expert kernels have a non-configurable B=8 boundary.
pub const hard_batch_token_limit: u32 = 8;
pub const active_experts_per_token: u32 = 6;
pub const routed_expert_count: u32 = 256;
pub const hash_routing_layers: u32 = 3;
pub const transformer_layers: u32 = 43;
pub const exact_q4_expert_bytes: u64 = 13_369_344;
pub const workspace_fixed_bytes: u64 = 8 * 1024 * 1024;
pub const workspace_per_token_bytes: u64 = 512 * 1024;
pub const hidden_host_bytes_per_token: u64 = 4 * 4096 * @sizeOf(f32);
pub const metadata_host_bytes_per_token: u64 = @sizeOf(i32) + @sizeOf(u32);

const gib: u64 = 1024 * 1024 * 1024;

pub const Phase = enum {
    prefill,
    verify,
    decode,
};

pub const Reason = enum {
    zero_remaining,
    decode_is_single_token,
    required_component_disabled,
    bounded_batch,
    size_one_fallback,
};

/// These values must come from one already-validated immutable startup config.
pub const ValidatedFeatures = struct {
    batch_prefill: bool,
    batch_mla: bool,
    expert_buckets: bool,
    batched_expert_kernels: bool,
    dspark_batch_verify: bool,

    pub fn enabledFor(self: ValidatedFeatures, phase: Phase) bool {
        const common = self.batch_mla and self.expert_buckets and
            self.batched_expert_kernels;
        return switch (phase) {
            .prefill => self.batch_prefill and common,
            .verify => self.dspark_batch_verify and common,
            .decode => false,
        };
    }
};

/// Explicit 48 GiB unified-memory accounting. The planner never treats the
/// whole machine as an allocation budget: OS/page-cache and failure headroom
/// remain reserved. Hot exact-Q4 experts are accounted from their real bytes.
pub const MemoryBudget = struct {
    physical_bytes: u64 = 48 * gib,
    non_expert_model_bytes: u64 = 5 * gib,
    attention_mhc_bytes: u64 = 2 * gib,
    metal_gpu_bytes: u64 = 3 * gib,
    os_page_cache_bytes: u64 = 5 * gib,
    safety_headroom_bytes: u64 = 4 * gib,
    hot_expert_bytes: u64,
    smelt_n: u32,

    pub fn forSmeltN(smelt_n: u32) MemoryBudget {
        // SMELT preloads all 256 experts in the three hash-routing layers and
        // N experts in each of the remaining score-routing layers. N=0 fully
        // disables SMELT and therefore has no resident hot-expert allocation.
        const resident_experts: u64 = if (smelt_n == 0)
            0
        else
            @as(u64, hash_routing_layers) * routed_expert_count +
                @as(u64, transformer_layers - hash_routing_layers) *
                    @min(smelt_n, routed_expert_count);
        return .{
            .smelt_n = smelt_n,
            .hot_expert_bytes = resident_experts * exact_q4_expert_bytes,
        };
    }

    pub fn n20() MemoryBudget {
        return forSmeltN(20);
    }

    pub fn fixedBytes(self: MemoryBudget) ?u64 {
        var total: u64 = 0;
        inline for (.{
            self.non_expert_model_bytes,
            self.attention_mhc_bytes,
            self.metal_gpu_bytes,
            self.os_page_cache_bytes,
            self.safety_headroom_bytes,
            self.hot_expert_bytes,
        }) |value| {
            total = checkedAdd(total, value) orelse return null;
        }
        return total;
    }

    pub fn incrementalCapacity(self: MemoryBudget) u64 {
        const fixed = self.fixedBytes() orelse return 0;
        return if (fixed < self.physical_bytes)
            self.physical_bytes - fixed
        else
            0;
    }
};

/// Runtime capacities are a snapshot derived from the installed immutable C
/// config. pool_slots and workspace_allocated_bytes describe allocations that
/// already exist; shrinking a chunk cannot pretend those bytes disappear.
pub const Limits = struct {
    max_prefill_chunk: u32,
    max_verify_chunk: u32,
    workspace_capacity_tokens: u32,
    workspace_budget_bytes: u64,
    workspace_allocated_bytes: u64,
    /// Production prefill-v2 allocates an independent exact-size MLA arena.
    /// Admission charges its full validated budget because the exact SDPA
    /// requirement also depends on absolute prefix length at execution time.
    mla_workspace_budget_bytes: u64,
    batch_pool_slots: u32,
    hard_token_limit: u32 = hard_batch_token_limit,
    expert_bytes: u64 = exact_q4_expert_bytes,
    memory: MemoryBudget = MemoryBudget.n20(),
};

pub const Request = struct {
    phase: Phase,
    remaining_tokens: u32,
    /// High-water observed from completed chunks. It is a lower bound for all
    /// later estimates, never a license to exceed the configured allocation.
    observed_workspace_high_water_bytes: u64 = 0,
};

pub const Plan = struct {
    batch_tokens: u32,
    uses_batch: bool,
    reason: Reason,
    conservative_unique_experts: u32,
    rolling_unique_experts: u32,
    estimated_workspace_bytes: u64,
    reserved_mla_workspace_bytes: u64,
    reserved_pool_bytes: u64,
    estimated_host_bytes: u64,
    estimated_incremental_bytes: u64,
};

pub const Planner = struct {
    features: ValidatedFeatures,
    limits: Limits,
    rolling_unique_q8: u64 = 0,
    rolling_samples: u32 = 0,

    pub fn init(features: ValidatedFeatures, limits: Limits) Planner {
        return .{ .features = features, .limits = limits };
    }

    /// Record a completed layer/chunk observation. The EWMA is advisory for
    /// I/O telemetry only; capacity and memory checks remain worst-case.
    pub fn observeUniqueExperts(self: *Planner, unique_experts: u32) void {
        const bounded = @min(unique_experts, routed_expert_count);
        const sample_q8 = @as(u64, bounded) * 256;
        if (self.rolling_samples == 0) {
            self.rolling_unique_q8 = sample_q8;
        } else {
            self.rolling_unique_q8 =
                (self.rolling_unique_q8 * 7 + sample_q8 + 7) / 8;
        }
        self.rolling_samples +|= 1;
    }

    pub fn plan(self: *const Planner, request: Request) Plan {
        if (request.remaining_tokens == 0) {
            return self.scalarPlan(request, 0, .zero_remaining);
        }
        if (request.phase == .decode) {
            return self.scalarPlan(request, 1, .decode_is_single_token);
        }
        if (!self.features.enabledFor(request.phase)) {
            return self.scalarPlan(request, 1, .required_component_disabled);
        }

        const phase_limit = switch (request.phase) {
            .prefill => self.limits.max_prefill_chunk,
            .verify => self.limits.max_verify_chunk,
            .decode => 1,
        };
        var candidate = @min(
            request.remaining_tokens,
            phase_limit,
            self.limits.workspace_capacity_tokens,
            self.limits.hard_token_limit,
        );

        // B=1 is deliberately the scalar fallback. Search downward so every
        // budget/capacity rejection happens before host allocation or KV state.
        while (candidate > 1) : (candidate -= 1) {
            if (self.batchPlan(request, candidate)) |result| return result;
        }
        return self.scalarPlan(request, 1, .size_one_fallback);
    }

    /// Re-plan below a failed, not-yet-executed host allocation. Callers must
    /// never use this after forwardBatch has begun mutating model state.
    pub fn shrinkBeforeExecution(
        self: *const Planner,
        request: Request,
        previous: Plan,
    ) Plan {
        if (!previous.uses_batch or previous.batch_tokens <= 2) {
            return self.scalarPlan(request, @min(request.remaining_tokens, 1), .size_one_fallback);
        }
        var smaller = request;
        smaller.remaining_tokens = @min(request.remaining_tokens, previous.batch_tokens - 1);
        return self.plan(smaller);
    }

    fn batchPlan(self: *const Planner, request: Request, tokens: u32) ?Plan {
        const conservative_unique = @min(
            routed_expert_count,
            tokens * active_experts_per_token,
        );
        // Pool capacity is checked at the worst-case route cardinality. Rolling
        // observations can never make an unsafe chunk appear feasible.
        if (conservative_unique > self.limits.batch_pool_slots) return null;

        const formula_workspace = workspaceEstimate(tokens) orelse return null;
        const estimated_workspace = @max(
            formula_workspace,
            request.observed_workspace_high_water_bytes,
        );
        if (estimated_workspace > self.limits.workspace_budget_bytes or
            estimated_workspace > self.limits.workspace_allocated_bytes)
        {
            return null;
        }

        const pool_bytes = checkedMul(
            self.limits.batch_pool_slots,
            self.limits.expert_bytes,
        ) orelse return null;
        const host_per_token = checkedAdd(
            hidden_host_bytes_per_token,
            metadata_host_bytes_per_token,
        ) orelse return null;
        const host_bytes = checkedMul(tokens, host_per_token) orelse return null;
        var incremental = checkedAdd(
            self.limits.workspace_allocated_bytes,
            self.limits.mla_workspace_budget_bytes,
        ) orelse return null;
        incremental = checkedAdd(incremental, pool_bytes) orelse return null;
        incremental = checkedAdd(incremental, host_bytes) orelse return null;
        if (incremental > self.limits.memory.incrementalCapacity()) return null;

        return .{
            .batch_tokens = tokens,
            .uses_batch = true,
            .reason = .bounded_batch,
            .conservative_unique_experts = conservative_unique,
            .rolling_unique_experts = self.rollingEstimate(tokens),
            .estimated_workspace_bytes = estimated_workspace,
            .reserved_mla_workspace_bytes = self.limits.mla_workspace_budget_bytes,
            .reserved_pool_bytes = pool_bytes,
            .estimated_host_bytes = host_bytes,
            .estimated_incremental_bytes = incremental,
        };
    }

    fn scalarPlan(
        self: *const Planner,
        request: Request,
        tokens: u32,
        reason: Reason,
    ) Plan {
        return .{
            .batch_tokens = tokens,
            .uses_batch = false,
            .reason = reason,
            .conservative_unique_experts = if (tokens == 0) 0 else active_experts_per_token,
            .rolling_unique_experts = if (tokens == 0) 0 else self.rollingEstimate(1),
            .estimated_workspace_bytes = request.observed_workspace_high_water_bytes,
            .reserved_mla_workspace_bytes = 0,
            .reserved_pool_bytes = 0,
            .estimated_host_bytes = 0,
            .estimated_incremental_bytes = 0,
        };
    }

    fn rollingEstimate(self: *const Planner, tokens: u32) u32 {
        const worst = @min(routed_expert_count, tokens * active_experts_per_token);
        if (self.rolling_samples == 0) return worst;
        const rounded = @as(u32, @intCast(@min(
            @as(u64, routed_expert_count),
            (self.rolling_unique_q8 + 255) / 256,
        )));
        // Two experts/token of safety margin is advisory; worst-case remains
        // separately reported and drives all hard pool/memory checks.
        return @min(worst, @max(active_experts_per_token, rounded +| tokens * 2));
    }
};

pub fn workspaceEstimate(tokens: u32) ?u64 {
    const variable = checkedMul(tokens, workspace_per_token_bytes) orelse return null;
    return checkedAdd(workspace_fixed_bytes, variable);
}

fn checkedAdd(a: u64, b: u64) ?u64 {
    if (b > std.math.maxInt(u64) - a) return null;
    return a + b;
}

fn checkedMul(a: anytype, b: u64) ?u64 {
    const lhs: u64 = @intCast(a);
    if (lhs != 0 and b > std.math.maxInt(u64) / lhs) return null;
    return lhs * b;
}

fn allFeatures() ValidatedFeatures {
    return .{
        .batch_prefill = true,
        .batch_mla = true,
        .expert_buckets = true,
        .batched_expert_kernels = true,
        .dspark_batch_verify = true,
    };
}

fn normalLimits() Limits {
    return .{
        .max_prefill_chunk = 8,
        .max_verify_chunk = 5,
        .workspace_capacity_tokens = 8,
        .workspace_budget_bytes = 256 * 1024 * 1024,
        .workspace_allocated_bytes = workspaceEstimate(8).?,
        .mla_workspace_budget_bytes = 256 * 1024 * 1024,
        .batch_pool_slots = 48,
    };
}

test "phase-aware planner bounds prefill verify decode and tails" {
    var planner = Planner.init(allFeatures(), normalLimits());
    const cases = [_]struct {
        phase: Phase,
        remaining: u32,
        expected: u32,
        batch: bool,
    }{
        .{ .phase = .prefill, .remaining = 19, .expected = 8, .batch = true },
        .{ .phase = .prefill, .remaining = 3, .expected = 3, .batch = true },
        .{ .phase = .verify, .remaining = 8, .expected = 5, .batch = true },
        .{ .phase = .verify, .remaining = 2, .expected = 2, .batch = true },
        .{ .phase = .decode, .remaining = 9, .expected = 1, .batch = false },
        .{ .phase = .prefill, .remaining = 0, .expected = 0, .batch = false },
    };
    for (cases) |case| {
        const result = planner.plan(.{
            .phase = case.phase,
            .remaining_tokens = case.remaining,
        });
        try std.testing.expectEqual(case.expected, result.batch_tokens);
        try std.testing.expectEqual(case.batch, result.uses_batch);
    }
}

test "disabled validated components force size one scalar" {
    const base = allFeatures();
    const cases = [_]ValidatedFeatures{
        .{ .batch_prefill = false, .batch_mla = base.batch_mla, .expert_buckets = base.expert_buckets, .batched_expert_kernels = base.batched_expert_kernels, .dspark_batch_verify = base.dspark_batch_verify },
        .{ .batch_prefill = base.batch_prefill, .batch_mla = false, .expert_buckets = base.expert_buckets, .batched_expert_kernels = base.batched_expert_kernels, .dspark_batch_verify = base.dspark_batch_verify },
        .{ .batch_prefill = base.batch_prefill, .batch_mla = base.batch_mla, .expert_buckets = false, .batched_expert_kernels = base.batched_expert_kernels, .dspark_batch_verify = base.dspark_batch_verify },
        .{ .batch_prefill = base.batch_prefill, .batch_mla = base.batch_mla, .expert_buckets = base.expert_buckets, .batched_expert_kernels = false, .dspark_batch_verify = base.dspark_batch_verify },
    };
    for (cases) |features| {
        var planner = Planner.init(features, normalLimits());
        const result = planner.plan(.{ .phase = .prefill, .remaining_tokens = 8 });
        try std.testing.expectEqual(@as(u32, 1), result.batch_tokens);
        try std.testing.expect(!result.uses_batch);
        try std.testing.expectEqual(Reason.required_component_disabled, result.reason);
    }

    var verify_features = base;
    verify_features.dspark_batch_verify = false;
    var verify_planner = Planner.init(verify_features, normalLimits());
    const verify = verify_planner.plan(.{ .phase = .verify, .remaining_tokens = 5 });
    try std.testing.expectEqual(@as(u32, 1), verify.batch_tokens);
    try std.testing.expect(!verify.uses_batch);
}

test "workspace and pool budgets shrink before execution" {
    var workspace_limits = normalLimits();
    workspace_limits.workspace_allocated_bytes = workspaceEstimate(3).?;
    var workspace_planner = Planner.init(allFeatures(), workspace_limits);
    const workspace_plan = workspace_planner.plan(.{ .phase = .prefill, .remaining_tokens = 8 });
    try std.testing.expectEqual(@as(u32, 3), workspace_plan.batch_tokens);

    var pool_limits = normalLimits();
    pool_limits.batch_pool_slots = 18;
    var pool_planner = Planner.init(allFeatures(), pool_limits);
    const pool_plan = pool_planner.plan(.{ .phase = .prefill, .remaining_tokens = 8 });
    try std.testing.expectEqual(@as(u32, 3), pool_plan.batch_tokens);

    const shrunk = pool_planner.shrinkBeforeExecution(
        .{ .phase = .prefill, .remaining_tokens = 8 },
        pool_plan,
    );
    try std.testing.expectEqual(@as(u32, 2), shrunk.batch_tokens);
    try std.testing.expect(shrunk.uses_batch);
}

test "size one is fail-safe when no batch estimate fits" {
    var limits = normalLimits();
    limits.workspace_allocated_bytes = workspace_fixed_bytes;
    var planner = Planner.init(allFeatures(), limits);
    const result = planner.plan(.{ .phase = .prefill, .remaining_tokens = 8 });
    try std.testing.expectEqual(@as(u32, 1), result.batch_tokens);
    try std.testing.expect(!result.uses_batch);
    try std.testing.expectEqual(Reason.size_one_fallback, result.reason);
}

test "workspace estimate never understates observed high water" {
    var planner = Planner.init(allFeatures(), normalLimits());
    const observed = workspaceEstimate(6).?;
    const result = planner.plan(.{
        .phase = .prefill,
        .remaining_tokens = 3,
        .observed_workspace_high_water_bytes = observed,
    });
    try std.testing.expect(result.uses_batch);
    try std.testing.expect(result.estimated_workspace_bytes >= observed);

    const impossible = planner.plan(.{
        .phase = .prefill,
        .remaining_tokens = 8,
        .observed_workspace_high_water_bytes = normalLimits().workspace_allocated_bytes + 1,
    });
    try std.testing.expect(!impossible.uses_batch);
    try std.testing.expect(impossible.estimated_workspace_bytes >=
        normalLimits().workspace_allocated_bytes + 1);
}

test "N equals 20 memory budget is explicit and bounded below 48 GiB" {
    const budget = MemoryBudget.n20();
    try std.testing.expectEqual(@as(u32, 20), budget.smelt_n);
    try std.testing.expectEqual(
        (@as(u64, hash_routing_layers) * routed_expert_count +
            @as(u64, transformer_layers - hash_routing_layers) * 20) *
            exact_q4_expert_bytes,
        budget.hot_expert_bytes,
    );
    const fixed = budget.fixedBytes().?;
    try std.testing.expect(fixed < budget.physical_bytes);
    try std.testing.expect(budget.incrementalCapacity() > 1024 * 1024 * 1024);

    var planner = Planner.init(allFeatures(), normalLimits());
    const plan = planner.plan(.{ .phase = .prefill, .remaining_tokens = 8 });
    try std.testing.expect(plan.uses_batch);
    try std.testing.expect(plan.estimated_incremental_bytes <= budget.incrementalCapacity());
}

test "rolling unique estimate is advisory and cannot bypass pool bound" {
    var planner = Planner.init(allFeatures(), normalLimits());
    planner.observeUniqueExperts(10);
    planner.observeUniqueExperts(12);
    const plan = planner.plan(.{ .phase = .prefill, .remaining_tokens = 8 });
    try std.testing.expect(plan.uses_batch);
    try std.testing.expect(plan.rolling_unique_experts <= plan.conservative_unique_experts);
    try std.testing.expectEqual(@as(u32, 48), plan.conservative_unique_experts);

    planner.limits.batch_pool_slots = 47;
    const bounded = planner.plan(.{ .phase = .prefill, .remaining_tokens = 8 });
    try std.testing.expectEqual(@as(u32, 7), bounded.batch_tokens);
    try std.testing.expectEqual(@as(u32, 42), bounded.conservative_unique_experts);
}

test "verify plan reserves one anchor and caps drafts at four" {
    var planner = Planner.init(allFeatures(), normalLimits());
    const plan = planner.plan(.{ .phase = .verify, .remaining_tokens = 9 });
    try std.testing.expect(plan.uses_batch);
    try std.testing.expectEqual(@as(u32, 5), plan.batch_tokens);
    try std.testing.expectEqual(@as(u32, 4), plan.batch_tokens - 1);
}
