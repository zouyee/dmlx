/// EmbedQueue — MPSC atomic queue for /v1/embeddings tasks.
///
/// The MLX Metal backend keeps command encoders in thread-local storage
/// (metal::get_command_encoder throws "no Stream(gpu, N) in current thread"
/// for any stream whose encoder was registered on another thread), so model
/// forwards only work on the engine-loop thread. HTTP handler fibers push
/// an EmbedTask here and block on `done`; the EngineLoop drains the queue
/// each iteration and runs the model's embed op on its own thread.
///
/// Same intrusive-list pattern as RequestQueue (wait-free push via CAS,
/// drain via atomic swap + reversal for FIFO).
const std = @import("std");

/// One pending embedding task: raw token ids in, L2-normalized embedding
/// out. Owned by the pushing fiber; must stay valid until `done` is set.
pub const EmbedTask = struct {
    /// Raw-text token ids (borrowed from the pusher; alive for the wait).
    ids: []const u32,
    /// Allocator the engine uses for `result` (the pusher frees it).
    allocator: std.mem.Allocator,
    result: ?[]f32 = null,
    err: ?anyerror = null,
    done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    next: std.atomic.Value(?*EmbedTask) = std.atomic.Value(?*EmbedTask).init(null),
};

pub const EmbedQueue = struct {
    head: std.atomic.Value(?*EmbedTask),

    pub fn init() EmbedQueue {
        return .{ .head = std.atomic.Value(?*EmbedTask).init(null) };
    }

    /// Push a task (called from HTTP fiber, wait-free). The task must remain
    /// valid until the engine marks it done.
    pub fn push(self: *EmbedQueue, task: *EmbedTask) void {
        var current_head = self.head.load(.monotonic);
        while (true) {
            task.next.store(current_head, .monotonic);
            if (self.head.cmpxchgWeak(current_head, task, .release, .monotonic)) |updated| {
                current_head = updated;
            } else {
                return;
            }
        }
    }

    /// Drain all pending tasks (called from the engine loop). Single
    /// consumer, so no ABA. Caller owns the returned slice.
    pub fn drainAll(self: *EmbedQueue, allocator: std.mem.Allocator) ![]*EmbedTask {
        const list_head = self.head.swap(null, .acquire);
        if (list_head == null) {
            return try allocator.alloc(*EmbedTask, 0);
        }
        var count: usize = 0;
        var node: ?*EmbedTask = list_head;
        while (node != null) {
            count += 1;
            node = node.?.next.load(.monotonic);
        }
        var result = try allocator.alloc(*EmbedTask, count);
        var idx: usize = count;
        node = list_head;
        while (node) |n| {
            idx -= 1;
            result[idx] = n;
            node = n.next.load(.monotonic);
        }
        return result;
    }
};

test "EmbedQueue push/drain FIFO" {
    const allocator = std.testing.allocator;
    var q = EmbedQueue.init();
    var t1 = EmbedTask{ .ids = &[_]u32{1}, .allocator = allocator };
    var t2 = EmbedTask{ .ids = &[_]u32{2}, .allocator = allocator };
    q.push(&t1);
    q.push(&t2);
    const tasks = try q.drainAll(allocator);
    defer allocator.free(tasks);
    try std.testing.expectEqual(@as(usize, 2), tasks.len);
    try std.testing.expectEqual(@as(u32, 1), tasks[0].ids[0]);
    try std.testing.expectEqual(@as(u32, 2), tasks[1].ids[0]);
    // Drained: next drain is empty.
    const empty = try q.drainAll(allocator);
    defer allocator.free(empty);
    try std.testing.expectEqual(@as(usize, 0), empty.len);
}
