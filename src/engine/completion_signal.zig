/// CompletionSignal — Engine→HTTP fiber synchronization.
///
/// The Engine fiber delivers tokens to the HTTP handler fiber through this
/// signal. The HTTP fiber polls on `waitForToken()` until a token is
/// available or the request is done.
///
/// Design: spin-wait polling with std.Io.Mutex (no Condition).
/// Condition.wait blocks the OS thread in some std.Io backends, causing
/// deadlock when the Engine fiber shares the same thread. Polling with
/// io.sleep yields the fiber cooperatively, allowing the Engine to run.
const std = @import("std");

/// A single token delivery event from Engine to HTTP fiber.
pub const TokenEvent = struct {
    /// The generated token ID.
    token_id: u32,
    /// Decoded text for this token (owned by the signal's arena).
    token_text: []const u8,
    /// Whether this is the final token (generation complete).
    is_final: bool,
    /// Finish reason (set only when is_final = true).
    finish_reason: ?FinishReason,

    pub const FinishReason = enum {
        stop,
        length,
        error_,
    };
};

/// Per-request signal for Engine→HTTP token delivery.
///
/// Thread-safe: Engine fiber calls deliver*, HTTP fiber calls wait/drain.
/// Uses std.Io.Mutex for short critical sections only (no blocking waits).
pub const CompletionSignal = struct {
    allocator: std.mem.Allocator,
    mutex: std.Io.Mutex,
    /// Tokens pending delivery (engine appends, HTTP drains).
    pending_tokens: std.ArrayList(TokenEvent),
    /// Owned text buffers for token_text in pending events.
    text_buffers: std.ArrayList([]const u8),
    /// Set to true when request is fully done.
    done: std.atomic.Value(bool),
    /// Error message if the request failed.
    error_msg: ?[]const u8,

    pub fn init(allocator: std.mem.Allocator) CompletionSignal {
        return .{
            .allocator = allocator,
            .mutex = std.Io.Mutex.init,
            .pending_tokens = std.ArrayList(TokenEvent).empty,
            .text_buffers = std.ArrayList([]const u8).empty,
            .done = std.atomic.Value(bool).init(false),
            .error_msg = null,
        };
    }

    pub fn deinit(self: *CompletionSignal) void {
        for (self.text_buffers.items) |buf| {
            self.allocator.free(buf);
        }
        self.text_buffers.deinit(self.allocator);
        self.pending_tokens.deinit(self.allocator);
        if (self.error_msg) |msg| {
            self.allocator.free(msg);
        }
    }

    /// Called by Engine fiber: deliver a token.
    pub fn deliverToken(self: *CompletionSignal, io: std.Io, token_id: u32, text: []const u8, is_final: bool, finish_reason: ?TokenEvent.FinishReason) void {
        self.mutex.lock(io) catch return;
        defer self.mutex.unlock(io);

        const text_copy = self.allocator.dupe(u8, text) catch {
            self.pending_tokens.append(self.allocator, .{
                .token_id = token_id,
                .token_text = "",
                .is_final = is_final,
                .finish_reason = finish_reason,
            }) catch return;
            if (is_final) self.done.store(true, .release);
            return;
        };
        self.text_buffers.append(self.allocator, text_copy) catch {
            self.allocator.free(text_copy);
            return;
        };

        self.pending_tokens.append(self.allocator, .{
            .token_id = token_id,
            .token_text = text_copy,
            .is_final = is_final,
            .finish_reason = finish_reason,
        }) catch return;

        if (is_final) self.done.store(true, .release);
    }

    /// Called by Engine fiber: signal error.
    pub fn deliverError(self: *CompletionSignal, io: std.Io, msg: []const u8) void {
        self.mutex.lock(io) catch return;
        defer self.mutex.unlock(io);

        self.error_msg = self.allocator.dupe(u8, msg) catch null;
        self.done.store(true, .release);
    }

    /// Called by Engine fiber: signal completion with no more tokens.
    pub fn deliverDone(self: *CompletionSignal, io: std.Io, finish_reason: TokenEvent.FinishReason) void {
        self.mutex.lock(io) catch return;
        defer self.mutex.unlock(io);

        self.pending_tokens.append(self.allocator, .{
            .token_id = 0,
            .token_text = "",
            .is_final = true,
            .finish_reason = finish_reason,
        }) catch {};
        self.done.store(true, .release);
    }

    fn removeTextBuffer(self: *CompletionSignal, text_ptr: [*]const u8) void {
        for (self.text_buffers.items, 0..) |buf, i| {
            if (buf.ptr == text_ptr) {
                _ = self.text_buffers.orderedRemove(i);
                return;
            }
        }
    }

    /// Called by HTTP fiber: poll until a token is available or done.
    /// Yields the fiber via io.sleep between polls to avoid busy-wait.
    /// Returns null when done with no more tokens pending.
    /// Caller owns the returned TokenEvent.token_text (must free with self.allocator).
    pub fn waitForToken(self: *CompletionSignal, io: std.Io) std.Io.Cancelable!?TokenEvent {
        while (true) {
            self.mutex.lock(io) catch return error.Canceled;
            const has_token = self.pending_tokens.items.len > 0;
            const is_done = self.done.load(.acquire);

            if (has_token) {
                const event = self.pending_tokens.orderedRemove(0);
                self.removeTextBuffer(event.token_text.ptr);
                self.mutex.unlock(io);
                return event;
            }

            if (is_done) {
                self.mutex.unlock(io);
                return null;
            }

            self.mutex.unlock(io);
            // Use shorter sleep to reduce latency between token delivery and HTTP response.
            // The HTTP fiber polls the completion signal while the Engine fiber generates tokens.
            // With 1ms sleep, there could be up to 1ms delay per token, adding significant
            // overhead when generating many tokens.
            io.sleep(.fromMicroseconds(100), .awake) catch return error.Canceled;
        }
    }

    /// Called by HTTP fiber: non-blocking check if a token is available.
    /// Returns null if no token is pending (does not block).
    /// Caller owns the returned TokenEvent.token_text (must free with self.allocator).
    pub fn tryGetToken(self: *CompletionSignal, io: std.Io) ?TokenEvent {
        self.mutex.lock(io) catch return null;
        defer self.mutex.unlock(io);

        if (self.pending_tokens.items.len > 0) {
            const event = self.pending_tokens.orderedRemove(0);
            self.removeTextBuffer(event.token_text.ptr);
            return event;
        }
        return null;
    }

    /// Called by HTTP fiber: wait with a timeout (for keep-alive).
    /// Returns null on timeout (no token available within duration).
    /// Caller owns the returned TokenEvent.token_text (must free with self.allocator).
    pub fn waitForTokenTimeout(self: *CompletionSignal, io: std.Io, timeout_ns: u64) std.Io.Cancelable!?TokenEvent {
        const start = std.Io.Timestamp.now(io, .monotonic);
        while (true) {
            self.mutex.lock(io) catch return error.Canceled;
            const has_token = self.pending_tokens.items.len > 0;
            const is_done = self.done.load(.acquire);

            if (has_token) {
                const event = self.pending_tokens.orderedRemove(0);
                self.removeTextBuffer(event.token_text.ptr);
                self.mutex.unlock(io);
                return event;
            }

            if (is_done) {
                self.mutex.unlock(io);
                return null;
            }

            self.mutex.unlock(io);

            const now = std.Io.Timestamp.now(io, .monotonic);
            const elapsed_ns = now.since(start).toNanoseconds();
            if (elapsed_ns >= timeout_ns) return null;

            io.sleep(.fromMilliseconds(1), .awake) catch return error.Canceled;
        }
    }

    /// Check if the signal indicates an error.
    pub fn hasError(self: *CompletionSignal, io: std.Io) bool {
        self.mutex.lock(io) catch return false;
        defer self.mutex.unlock(io);
        return self.error_msg != null;
    }

    /// Get the error message (if any). Caller does NOT own the returned slice.
    pub fn getError(self: *CompletionSignal, io: std.Io) ?[]const u8 {
        self.mutex.lock(io) catch return null;
        defer self.mutex.unlock(io);
        return self.error_msg;
    }

    /// Check if done (all tokens delivered or error).
    pub fn isDone(self: *CompletionSignal, _io: std.Io) bool {
        _ = _io;
        return self.done.load(.acquire);
    }
};
