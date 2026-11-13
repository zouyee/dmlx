/// Prometheus text exposition format (version 0.0.4) metrics for the server.
///
/// All counters are atomics — HTTP requests are handled on multiple io.async
/// fibers and token counts are recorded on the engine-loop thread, so every
/// mutation is a `fetchAdd` with .monotonic ordering. Gauges
/// (`dmlx_active_requests`, `dmlx_uptime_seconds`) are computed at render
/// time from ServerState.
const std = @import("std");

/// Canonical route labels for `dmlx_requests_total{endpoint=...}`. The
/// dispatch layer in http.zig matches on the same prefixes.
pub const route_names = [_][]const u8{
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/models",
    "/health",
    "/api/status",
    "/shutdown",
    "/v1/responses",
    "/v1/tokenize",
    "/tokenize",
    "/v1/detokenize",
    "/detokenize",
    "/v1/embeddings",
    "/v1/messages/count_tokens",
    "/v1/messages",
    "/metrics",
    "/v1/metrics",
    "other",
};

/// Status codes tracked per route; any other status lands in the last slot
/// and is rendered with its actual code.
pub const tracked_statuses = [_]u16{ 200, 400, 404, 500, 501, 503 };
const other_status_slot = tracked_statuses.len;

const RouteCounters = struct {
    status_counts: [tracked_statuses.len + 1]std.atomic.Value(u64),
    other_status_code: std.atomic.Value(u32),
    duration_sum_ns: std.atomic.Value(u64),
    duration_count: std.atomic.Value(u64),

    fn init() RouteCounters {
        var counts: [tracked_statuses.len + 1]std.atomic.Value(u64) = undefined;
        for (&counts) |*count| count.* = std.atomic.Value(u64).init(0);
        return .{
            .status_counts = counts,
            .other_status_code = std.atomic.Value(u32).init(0),
            .duration_sum_ns = std.atomic.Value(u64).init(0),
            .duration_count = std.atomic.Value(u64).init(0),
        };
    }
};

pub const Metrics = struct {
    routes: [route_names.len]RouteCounters,
    generation_tokens_total: std.atomic.Value(u64),
    prompt_tokens_total: std.atomic.Value(u64),
    mtp_drafts_total: std.atomic.Value(u64),
    mtp_accepted_drafts_total: std.atomic.Value(u64),

    pub fn init() Metrics {
        var routes: [route_names.len]RouteCounters = undefined;
        for (&routes) |*route| route.* = RouteCounters.init();
        return .{
            .routes = routes,
            .generation_tokens_total = std.atomic.Value(u64).init(0),
            .prompt_tokens_total = std.atomic.Value(u64).init(0),
            .mtp_drafts_total = std.atomic.Value(u64).init(0),
            .mtp_accepted_drafts_total = std.atomic.Value(u64).init(0),
        };
    }

    /// Map a raw HTTP request buffer to its canonical route index. Matches
    /// the same method+path prefixes as the dispatch chain in http.zig.
    pub fn routeIndex(request: []const u8) usize {
        const sp = std.mem.indexOfScalar(u8, request, ' ') orelse return route_names.len - 1;
        const path = request[sp + 1 ..];
        inline for (route_names, 0..) |name, i| {
            if (comptime !std.mem.eql(u8, name, "other")) {
                if (std.mem.startsWith(u8, path, name)) return i;
            }
        }
        return route_names.len - 1; // "other"
    }

    pub fn recordRequest(self: *Metrics, route_idx: usize, status: u16, duration_ns: u64) void {
        const route = &self.routes[route_idx];
        const slot = inline for (tracked_statuses, 0..) |code, i| {
            if (status == code) break i;
        } else other_status_slot;
        if (slot == other_status_slot) {
            // First writer wins for the rendered code; counts still accumulate.
            _ = route.other_status_code.cmpxchgStrong(0, status, .monotonic, .monotonic);
        }
        _ = route.status_counts[slot].fetchAdd(1, .monotonic);
        _ = route.duration_sum_ns.fetchAdd(duration_ns, .monotonic);
        _ = route.duration_count.fetchAdd(1, .monotonic);
    }

    pub fn addTokenCounts(self: *Metrics, prompt_tokens: u64, generated_tokens: u64) void {
        _ = self.prompt_tokens_total.fetchAdd(prompt_tokens, .monotonic);
        _ = self.generation_tokens_total.fetchAdd(generated_tokens, .monotonic);
    }

    /// Total HTTP requests handled across all routes (for /api/status).
    pub fn totalRequests(self: *Metrics) u64 {
        var total: u64 = 0;
        for (&self.routes) |*route| {
            total += route.duration_count.load(.monotonic);
        }
        return total;
    }

    pub fn addMtpDrafts(self: *Metrics, proposed: u64, accepted: u64) void {
        _ = self.mtp_drafts_total.fetchAdd(proposed, .monotonic);
        _ = self.mtp_accepted_drafts_total.fetchAdd(accepted, .monotonic);
    }

    /// Render the Prometheus text exposition body. Caller owns the result.
    /// `now_unix` is passed in so tests can control uptime deterministically.
    pub fn render(
        self: *Metrics,
        allocator: std.mem.Allocator,
        active_requests: u32,
        start_time_unix: i64,
        now_unix: i64,
    ) ![]u8 {
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(allocator);

        try out.appendSlice(allocator, "# HELP dmlx_requests_total HTTP requests handled, by endpoint and status code.\n");
        try out.appendSlice(allocator, "# TYPE dmlx_requests_total counter\n");
        for (&self.routes, 0..) |*route, i| {
            inline for (tracked_statuses, 0..) |code, slot| {
                const count = route.status_counts[slot].load(.monotonic);
                if (count > 0) {
                    try out.print(allocator, "dmlx_requests_total{{endpoint=\"{s}\",status=\"{d}\"}} {d}\n", .{ route_names[i], code, count });
                }
            }
            const other_count = route.status_counts[other_status_slot].load(.monotonic);
            if (other_count > 0) {
                const code = route.other_status_code.load(.monotonic);
                try out.print(allocator, "dmlx_requests_total{{endpoint=\"{s}\",status=\"{d}\"}} {d}\n", .{ route_names[i], code, other_count });
            }
        }

        try out.appendSlice(allocator, "# HELP dmlx_request_duration_seconds Total time spent handling HTTP requests, by endpoint.\n");
        try out.appendSlice(allocator, "# TYPE dmlx_request_duration_seconds summary\n");
        for (&self.routes, 0..) |*route, i| {
            const count = route.duration_count.load(.monotonic);
            if (count > 0) {
                const sum_s = @as(f64, @floatFromInt(route.duration_sum_ns.load(.monotonic))) / 1e9;
                try out.print(allocator, "dmlx_request_duration_seconds_sum{{endpoint=\"{s}\"}} {d:.6}\n", .{ route_names[i], sum_s });
                try out.print(allocator, "dmlx_request_duration_seconds_count{{endpoint=\"{s}\"}} {d}\n", .{ route_names[i], count });
            }
        }

        try out.appendSlice(allocator, "# HELP dmlx_generation_tokens_total Completion tokens emitted.\n");
        try out.appendSlice(allocator, "# TYPE dmlx_generation_tokens_total counter\n");
        try out.print(allocator, "dmlx_generation_tokens_total {d}\n", .{self.generation_tokens_total.load(.monotonic)});
        try out.appendSlice(allocator, "# HELP dmlx_prompt_tokens_total Prompt tokens processed.\n");
        try out.appendSlice(allocator, "# TYPE dmlx_prompt_tokens_total counter\n");
        try out.print(allocator, "dmlx_prompt_tokens_total {d}\n", .{self.prompt_tokens_total.load(.monotonic)});

        try out.appendSlice(allocator, "# HELP dmlx_mtp_drafts_total MTP draft tokens proposed.\n");
        try out.appendSlice(allocator, "# TYPE dmlx_mtp_drafts_total counter\n");
        try out.print(allocator, "dmlx_mtp_drafts_total {d}\n", .{self.mtp_drafts_total.load(.monotonic)});
        try out.appendSlice(allocator, "# HELP dmlx_mtp_accepted_drafts_total MTP draft tokens accepted by the verifier.\n");
        try out.appendSlice(allocator, "# TYPE dmlx_mtp_accepted_drafts_total counter\n");
        try out.print(allocator, "dmlx_mtp_accepted_drafts_total {d}\n", .{self.mtp_accepted_drafts_total.load(.monotonic)});

        try out.appendSlice(allocator, "# HELP dmlx_active_requests Requests currently being processed.\n");
        try out.appendSlice(allocator, "# TYPE dmlx_active_requests gauge\n");
        try out.print(allocator, "dmlx_active_requests {d}\n", .{active_requests});

        const uptime: i64 = if (start_time_unix > 0 and now_unix > start_time_unix) now_unix - start_time_unix else 0;
        try out.appendSlice(allocator, "# HELP dmlx_uptime_seconds Seconds since the server started.\n");
        try out.appendSlice(allocator, "# TYPE dmlx_uptime_seconds gauge\n");
        try out.print(allocator, "dmlx_uptime_seconds {d}\n", .{uptime});

        return out.toOwnedSlice(allocator);
    }
};

/// Global server-wide instance; safe to touch from any thread.
pub var global = Metrics.init();

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

test "recordRequest increments per-route and per-status counters" {
    var m = Metrics.init();
    const idx = Metrics.routeIndex("POST /v1/chat/completions HTTP/1.1\r\n");
    try std.testing.expectEqualStrings("/v1/chat/completions", route_names[idx]);
    m.recordRequest(idx, 200, 1_500_000_000);
    m.recordRequest(idx, 200, 500_000_000);
    m.recordRequest(idx, 400, 100_000);
    try std.testing.expectEqual(@as(u64, 2), m.routes[idx].status_counts[0].load(.monotonic));
    try std.testing.expectEqual(@as(u64, 1), m.routes[idx].status_counts[1].load(.monotonic));
    try std.testing.expectEqual(@as(u64, 3), m.routes[idx].duration_count.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 2_000_100_000), m.routes[idx].duration_sum_ns.load(.monotonic));
}

test "routeIndex classifies known and unknown routes" {
    try std.testing.expectEqualStrings("/v1/models", route_names[Metrics.routeIndex("GET /v1/models HTTP/1.1")]);
    try std.testing.expectEqualStrings("/health", route_names[Metrics.routeIndex("GET /health HTTP/1.1")]);
    try std.testing.expectEqualStrings("/metrics", route_names[Metrics.routeIndex("GET /metrics HTTP/1.1")]);
    try std.testing.expectEqualStrings("/v1/metrics", route_names[Metrics.routeIndex("GET /v1/metrics HTTP/1.1")]);
    try std.testing.expectEqualStrings("/tokenize", route_names[Metrics.routeIndex("POST /tokenize HTTP/1.1")]);
    try std.testing.expectEqualStrings("/v1/tokenize", route_names[Metrics.routeIndex("POST /v1/tokenize HTTP/1.1")]);
    try std.testing.expectEqualStrings("other", route_names[Metrics.routeIndex("GET /nope HTTP/1.1")]);
}

test "render produces valid Prometheus text exposition" {
    const allocator = std.testing.allocator;
    var m = Metrics.init();
    m.recordRequest(Metrics.routeIndex("GET /health HTTP/1.1"), 200, 2_000_000_000);
    m.recordRequest(Metrics.routeIndex("POST /v1/chat/completions HTTP/1.1"), 400, 1_000_000);
    m.addTokenCounts(17, 42);
    m.addMtpDrafts(10, 7);

    const body = try m.render(allocator, 3, 1000, 1061);
    defer allocator.free(body);

    // Every line ends with \n (no trailing partial line).
    try std.testing.expect(body.len > 0);
    try std.testing.expect(body[body.len - 1] == '\n');

    try std.testing.expect(std.mem.indexOf(u8, body, "# TYPE dmlx_requests_total counter\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "# TYPE dmlx_request_duration_seconds summary\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "# TYPE dmlx_generation_tokens_total counter\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "# TYPE dmlx_prompt_tokens_total counter\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "# TYPE dmlx_active_requests gauge\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "# TYPE dmlx_uptime_seconds gauge\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "# TYPE dmlx_mtp_drafts_total counter\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "# TYPE dmlx_mtp_accepted_drafts_total counter\n") != null);

    try std.testing.expect(std.mem.indexOf(u8, body, "dmlx_requests_total{endpoint=\"/health\",status=\"200\"} 1\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "dmlx_requests_total{endpoint=\"/v1/chat/completions\",status=\"400\"} 1\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "dmlx_request_duration_seconds_sum{endpoint=\"/health\"} 2.000000\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "dmlx_request_duration_seconds_count{endpoint=\"/health\"} 1\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "dmlx_generation_tokens_total 42\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "dmlx_prompt_tokens_total 17\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "dmlx_mtp_drafts_total 10\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "dmlx_mtp_accepted_drafts_total 7\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "dmlx_active_requests 3\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "dmlx_uptime_seconds 61\n") != null);
}

test "render reports zero uptime when start time is unset" {
    const allocator = std.testing.allocator;
    var m = Metrics.init();
    const body = try m.render(allocator, 0, 0, 5000);
    defer allocator.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "dmlx_uptime_seconds 0\n") != null);
}
