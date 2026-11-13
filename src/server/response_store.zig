/// In-memory store for the Responses API: id → the full response object
/// JSON, retained after POST /v1/responses completes so clients can
/// GET /v1/responses/{id} or DELETE /v1/responses/{id} later.
///
/// Bounded: at most `max_entries` objects are retained; inserting past the
/// cap evicts the OLDEST entry (FIFO). Thread-safe via a pthread mutex (the
/// repo pattern, cf. model_registry.zig): the store is written from HTTP
/// connection fibers when a response completes.
const std = @import("std");
const pthread = @cImport(@cInclude("pthread.h"));

pub const max_entries: usize = 256;

pub const ResponseStore = struct {
    allocator: std.mem.Allocator,
    mu: pthread.pthread_mutex_t,
    map: std.StringHashMap([]u8),
    /// Insertion order of ids (oldest first) for FIFO eviction. Entries
    /// reference the same owned id copies as `map` keys.
    order: std.ArrayList([]u8),

    pub fn init(allocator: std.mem.Allocator) ResponseStore {
        var mu: pthread.pthread_mutex_t = undefined;
        _ = pthread.pthread_mutex_init(&mu, null);
        return .{
            .allocator = allocator,
            .mu = mu,
            .map = std.StringHashMap([]u8).init(allocator),
            .order = std.ArrayList([]u8).empty,
        };
    }

    pub fn deinit(self: *ResponseStore) void {
        var it = self.map.iterator();
        while (it.next()) |kv| {
            self.allocator.free(kv.key_ptr.*);
            self.allocator.free(kv.value_ptr.*);
        }
        self.map.deinit();
        self.order.deinit(self.allocator);
        _ = pthread.pthread_mutex_destroy(&self.mu);
    }

    /// Insert or replace the object for `id`, evicting oldest entries past
    /// the cap. Both slices are copied.
    pub fn put(self: *ResponseStore, id: []const u8, body: []const u8) !void {
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);

        const id_copy = try self.allocator.dupe(u8, id);
        errdefer self.allocator.free(id_copy);
        const body_copy = try self.allocator.dupe(u8, body);
        errdefer self.allocator.free(body_copy);

        const gop = try self.map.getOrPut(id_copy);
        if (gop.found_existing) {
            // Replace value; id already sits in `order`.
            self.allocator.free(gop.value_ptr.*);
            self.allocator.free(id_copy);
            gop.value_ptr.* = body_copy;
        } else {
            gop.key_ptr.* = id_copy;
            gop.value_ptr.* = body_copy;
            try self.order.append(self.allocator, id_copy);
        }

        while (self.order.items.len > max_entries) {
            const oldest = self.order.orderedRemove(0);
            if (self.map.fetchRemove(oldest)) |kv| {
                self.allocator.free(kv.key);
                self.allocator.free(kv.value);
            }
        }
    }

    /// Fetch a COPY of the stored object (caller frees), or null.
    pub fn get(self: *ResponseStore, allocator: std.mem.Allocator, id: []const u8) !?[]u8 {
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        const body = self.map.get(id) orelse return null;
        return try allocator.dupe(u8, body);
    }

    /// Remove `id`; returns true when it existed.
    pub fn remove(self: *ResponseStore, id: []const u8) bool {
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        const kv = self.map.fetchRemove(id) orelse return false;
        for (self.order.items, 0..) |oid, i| {
            if (oid.ptr == kv.key.ptr) {
                _ = self.order.orderedRemove(i);
                break;
            }
        }
        self.allocator.free(kv.key);
        self.allocator.free(kv.value);
        return true;
    }

    pub fn len(self: *ResponseStore) usize {
        _ = pthread.pthread_mutex_lock(&self.mu);
        defer _ = pthread.pthread_mutex_unlock(&self.mu);
        return self.map.count();
    }
};

/// Extract the {id} segment from a routed request line: the path between
/// "GET /v1/responses/" (or DELETE) and the end of the request target.
/// Returns null when the id is empty or carries a query string we don't
/// understand (query strings are simply stripped).
pub fn extractResponseId(request: []const u8, comptime method_prefix: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, request, method_prefix)) return null;
    const rest = request[method_prefix.len..];
    const end = std.mem.indexOfAny(u8, rest, " ?\r\n") orelse rest.len;
    const id = rest[0..end];
    if (id.len == 0) return null;
    return id;
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

test "put/get/remove round trip" {
    const allocator = std.testing.allocator;
    var store = ResponseStore.init(allocator);
    defer store.deinit();

    try store.put("resp_a", "{\"id\":\"resp_a\"}");
    try std.testing.expectEqual(@as(usize, 1), store.len());

    const got = (try store.get(allocator, "resp_a")).?;
    defer allocator.free(got);
    try std.testing.expectEqualStrings("{\"id\":\"resp_a\"}", got);

    try std.testing.expect((try store.get(allocator, "resp_b")) == null);
    try std.testing.expect(store.remove("resp_a"));
    try std.testing.expect(!store.remove("resp_a"));
    try std.testing.expectEqual(@as(usize, 0), store.len());
}

test "put replaces an existing id without growing" {
    const allocator = std.testing.allocator;
    var store = ResponseStore.init(allocator);
    defer store.deinit();

    try store.put("resp_a", "v1");
    try store.put("resp_a", "v2");
    try std.testing.expectEqual(@as(usize, 1), store.len());
    const got = (try store.get(allocator, "resp_a")).?;
    defer allocator.free(got);
    try std.testing.expectEqualStrings("v2", got);
}

test "store evicts the oldest entry past the cap" {
    const allocator = std.testing.allocator;
    var store = ResponseStore.init(allocator);
    defer store.deinit();

    for (0..max_entries + 10) |i| {
        const id = try std.fmt.allocPrint(allocator, "resp_{d}", .{i});
        defer allocator.free(id);
        try store.put(id, "x");
    }
    try std.testing.expectEqual(@as(usize, max_entries), store.len());
    // The first 10 were evicted, the rest survive.
    try std.testing.expect((try store.get(allocator, "resp_0")) == null);
    try std.testing.expect((try store.get(allocator, "resp_9")) == null);
    {
        const got = (try store.get(allocator, "resp_10")).?;
        defer allocator.free(got);
        try std.testing.expectEqualStrings("x", got);
    }
    {
        const id = try std.fmt.allocPrint(allocator, "resp_{d}", .{max_entries + 9});
        defer allocator.free(id);
        const got = (try store.get(allocator, id)).?;
        defer allocator.free(got);
        try std.testing.expectEqualStrings("x", got);
    }
}

test "extractResponseId pulls the id from the request line" {
    try std.testing.expectEqualStrings(
        "resp_abc123",
        extractResponseId("GET /v1/responses/resp_abc123 HTTP/1.1\r\n", "GET /v1/responses/").?,
    );
    try std.testing.expectEqualStrings(
        "resp_x",
        extractResponseId("DELETE /v1/responses/resp_x?foo=bar HTTP/1.1", "DELETE /v1/responses/").?,
    );
    try std.testing.expect(extractResponseId("GET /v1/responses/ HTTP/1.1", "GET /v1/responses/") == null);
    try std.testing.expect(extractResponseId("POST /v1/responses HTTP/1.1", "GET /v1/responses/") == null);
}
