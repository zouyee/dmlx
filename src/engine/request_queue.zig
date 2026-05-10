/// RequestQueue — Multi-Producer Single-Consumer atomic queue.
///
/// HTTP handler fibers push requests (wait-free via atomic CAS on head).
/// The Engine fiber drains all pending requests in one atomic swap.
///
/// Implementation: intrusive singly-linked list with atomic head pointer.
/// Push is wait-free (single CAS). Drain is a single atomic exchange
/// followed by list reversal to restore FIFO order.
///
/// No heap allocation in the push path — Node is stack-allocated in the
/// HTTP handler fiber and lives until the Engine drains it.
const std = @import("std");
const RequestState = @import("request_state.zig").RequestState;

/// Intrusive node for the MPSC queue.
/// Stack-allocated in the HTTP handler fiber.
pub const Node = struct {
    request: *RequestState,
    next: std.atomic.Value(?*Node),

    pub fn init(request: *RequestState) Node {
        return .{
            .request = request,
            .next = std.atomic.Value(?*Node).init(null),
        };
    }
};

/// Multi-producer single-consumer atomic queue.
pub const RequestQueue = struct {
    head: std.atomic.Value(?*Node),

    pub fn init() RequestQueue {
        return .{
            .head = std.atomic.Value(?*Node).init(null),
        };
    }

    /// Push a request node (called from HTTP fiber, wait-free).
    ///
    /// The node must remain valid until the Engine fiber drains it.
    /// Typically the node is stack-allocated in the HTTP handler and
    /// the handler blocks on CompletionSignal after pushing.
    pub fn push(self: *RequestQueue, node: *Node) void {
        var current_head = self.head.load(.monotonic);
        while (true) {
            node.next.store(current_head, .monotonic);
            // Try to CAS head from current_head to node
            if (self.head.cmpxchgWeak(
                current_head,
                node,
                .release,
                .monotonic,
            )) |updated_head| {
                // CAS failed, retry with the updated head
                current_head = updated_head;
            } else {
                // CAS succeeded
                return;
            }
        }
    }

    /// Drain all pending requests (called from Engine fiber).
    ///
    /// Atomically swaps head to null, then reverses the list to get FIFO order.
    /// Returns a slice of *RequestState pointers. Caller owns the slice.
    ///
    /// This is the only function that removes nodes from the queue.
    /// Since there's a single consumer (Engine fiber), no ABA problem.
    pub fn drainAll(self: *RequestQueue, allocator: std.mem.Allocator) ![]*RequestState {
        // Atomically take the entire list
        const list_head = self.head.swap(null, .acquire);

        if (list_head == null) {
            return try allocator.alloc(*RequestState, 0);
        }

        // Count nodes and collect into array (reversed order)
        var count: usize = 0;
        var node: ?*Node = list_head;
        while (node != null) {
            count += 1;
            node = node.?.next.load(.monotonic);
        }

        // Allocate result slice
        var result = try allocator.alloc(*RequestState, count);

        // Fill in reverse order (list is LIFO, we want FIFO)
        var idx: usize = count;
        node = list_head;
        while (node) |n| {
            idx -= 1;
            result[idx] = n.request;
            node = n.next.load(.monotonic);
        }

        return result;
    }

    /// Check if the queue has any pending requests (non-blocking).
    pub fn isEmpty(self: *const RequestQueue) bool {
        return self.head.load(.acquire) == null;
    }
};
