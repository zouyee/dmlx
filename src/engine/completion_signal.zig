/// CompletionSignal — Engine→HTTP cross-thread synchronization.
///
/// The Engine thread delivers tokens to the HTTP handler thread through this
/// signal. The HTTP thread waits on `waitForToken()` until a token is
/// available or the request is done.
///
/// Design: Atomic spinlock + Darwin ulock for cross-thread notification.
///
/// Architecture (matching omlx's asyncio.Event pattern):
/// - Engine runs on the main thread (for MLX Metal GPU operations)
/// - HTTP handlers run on worker threads (via io.async)
/// - std.Io.Mutex/Condition are fiber-aware but require io parameter
/// - We use atomic spinlock + Darwin __ulock_wake/__ulock_wait for true
///   cross-thread wake without needing an io instance on the producer side
///
/// The ulock wake unblocks the waiting HTTP thread immediately when
/// a token is delivered, providing low-latency streaming without busy-wait.
/// This is the Zig equivalent of Python's asyncio.Event.set() pattern used in omlx.
const std = @import("std");

/// A single token delivery event from Engine to HTTP thread.
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
/// Thread-safe: Engine thread calls deliver*, HTTP thread calls wait/drain.
/// Uses atomic spinlock for the critical section and Darwin ulock for
/// cross-thread wake notification.
pub const CompletionSignal = struct {
    allocator: std.mem.Allocator,
    /// Spinlock for protecting pending_tokens and text_buffers.
    /// 0 = unlocked, 1 = locked.
    spinlock: std.atomic.Value(u32),
    /// Wake counter for cross-thread notification.
    /// Incremented on each delivery; HTTP thread uses __ulock_wait on this.
    wake_counter: std.atomic.Value(u32),
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
            .spinlock = std.atomic.Value(u32).init(0),
            .wake_counter = std.atomic.Value(u32).init(0),
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

    /// Acquire spinlock (busy-wait, very short critical sections only).
    inline fn acquire(self: *CompletionSignal) void {
        while (true) {
            if (self.spinlock.cmpxchgWeak(0, 1, .acquire, .monotonic) == null) return;
            std.atomic.spinLoopHint();
        }
    }

    /// Release spinlock.
    inline fn release(self: *CompletionSignal) void {
        self.spinlock.store(0, .release);
    }

    /// Wake the waiting HTTP thread via Darwin ulock.
    /// Increments wake_counter and issues a platform wake.
    fn wakeWaiter(self: *CompletionSignal) void {
        _ = self.wake_counter.fetchAdd(1, .release);
        // Darwin __ulock_wake: wake one thread waiting on this address.
        // UL_COMPARE_AND_WAIT = 0x00000001
        _ = std.c.__ulock_wake(
            @bitCast(@as(u32, 0x00000001)),
            @ptrCast(&self.wake_counter),
            0,
        );
    }

    /// Block until wake_counter changes from `expected`.
    /// Uses Darwin __ulock_wait for efficient cross-thread blocking.
    fn waitForWake(self: *CompletionSignal, expected: u32, timeout_us: u32) void {
        // Darwin __ulock_wait: block if *addr == expected.
        // UL_COMPARE_AND_WAIT = 0x00000001
        _ = std.c.__ulock_wait(
            @bitCast(@as(u32, 0x00000001)),
            @ptrCast(&self.wake_counter),
            @as(u64, expected),
            timeout_us,
        );
    }

    /// Called by Engine thread: deliver a token.
    /// Signals the waiting HTTP thread immediately via ulock wake.
    pub fn deliverToken(self: *CompletionSignal, io: std.Io, token_id: u32, text: []const u8, is_final: bool, finish_reason: ?TokenEvent.FinishReason) void {
        _ = io;
        const t0 = std.c.mach_absolute_time();
        self.acquire();

        const text_copy = self.allocator.dupe(u8, text) catch {
            self.pending_tokens.append(self.allocator, .{
                .token_id = token_id,
                .token_text = "",
                .is_final = is_final,
                .finish_reason = finish_reason,
            }) catch {
                self.release();
                return;
            };
            if (is_final) self.done.store(true, .release);
            self.release();
            self.wakeWaiter();
            return;
        };
        self.text_buffers.append(self.allocator, text_copy) catch {
            self.allocator.free(text_copy);
            self.release();
            return;
        };

        self.pending_tokens.append(self.allocator, .{
            .token_id = token_id,
            .token_text = text_copy,
            .is_final = is_final,
            .finish_reason = finish_reason,
        }) catch {
            self.release();
            return;
        };

        if (is_final) self.done.store(true, .release);
        self.release();
        self.wakeWaiter();
        const t1 = std.c.mach_absolute_time();
        std.log.info("[Signal] deliverToken id={d} final={} took {d}us", .{ token_id, is_final, (t1 - t0) / 1000 });
    }

    /// Called by Engine thread: signal error.
    pub fn deliverError(self: *CompletionSignal, io: std.Io, msg: []const u8) void {
        _ = io;
        self.acquire();

        self.error_msg = self.allocator.dupe(u8, msg) catch null;
        self.done.store(true, .release);
        self.release();
        self.wakeWaiter();
    }

    /// Called by Engine thread: signal completion with no more tokens.
    pub fn deliverDone(self: *CompletionSignal, io: std.Io, finish_reason: TokenEvent.FinishReason) void {
        _ = io;
        self.acquire();

        self.pending_tokens.append(self.allocator, .{
            .token_id = 0,
            .token_text = "",
            .is_final = true,
            .finish_reason = finish_reason,
        }) catch {};
        self.done.store(true, .release);
        self.release();
        self.wakeWaiter();
    }

    fn removeTextBuffer(self: *CompletionSignal, text_ptr: [*]const u8) void {
        for (self.text_buffers.items, 0..) |buf, i| {
            if (buf.ptr == text_ptr) {
                _ = self.text_buffers.orderedRemove(i);
                return;
            }
        }
    }

    /// Called by HTTP thread: wait until a token is available or done.
    /// Uses Darwin ulock for efficient cross-thread notification.
    /// Returns null when done with no more tokens pending.
    /// Caller owns the returned TokenEvent.token_text (must free with self.allocator).
    pub fn waitForToken(self: *CompletionSignal, io: std.Io) std.Io.Cancelable!?TokenEvent {
        _ = io;
        const wait_start = std.c.mach_absolute_time();
        while (true) {
            // Try to get a token under the spinlock.
            self.acquire();
            if (self.pending_tokens.items.len > 0) {
                const event = self.pending_tokens.orderedRemove(0);
                self.removeTextBuffer(event.token_text.ptr);
                self.release();
                const wait_end = std.c.mach_absolute_time();
                std.log.info("[Signal] waitForToken got id={d} waited {d}ms", .{ event.token_id, (wait_end - wait_start) / 1_000_000 });
                return event;
            }

            if (self.done.load(.acquire)) {
                self.release();
                return null;
            }
            self.release();

            // Poll with short sleep instead of __ulock_wait.
            // __ulock_wait may not correctly wake threads in Zig's io.async
            // worker thread pool. Use nanosleep(100μs) as a reliable fallback.
            const ts = std.c.timespec{ .sec = 0, .nsec = 100_000 }; // 100μs
            _ = std.c.nanosleep(&ts, null);
        }
    }

    /// Called by HTTP thread: non-blocking check if a token is available.
    /// Returns null if no token is pending (does not block).
    /// Caller owns the returned TokenEvent.token_text (must free with self.allocator).
    pub fn tryGetToken(self: *CompletionSignal, io: std.Io) ?TokenEvent {
        _ = io;
        self.acquire();
        defer self.release();

        if (self.pending_tokens.items.len > 0) {
            const event = self.pending_tokens.orderedRemove(0);
            self.removeTextBuffer(event.token_text.ptr);
            return event;
        }
        return null;
    }

    /// Called by HTTP thread: wait with a timeout (for keep-alive).
    /// Returns null on timeout (no token available within duration).
    /// Caller owns the returned TokenEvent.token_text (must free with self.allocator).
    pub fn waitForTokenTimeout(self: *CompletionSignal, io: std.Io, timeout_ns: u64) std.Io.Cancelable!?TokenEvent {
        _ = io;
        const start = std.time.nanoTimestamp();

        while (true) {
            const counter = self.wake_counter.load(.acquire);

            self.acquire();
            if (self.pending_tokens.items.len > 0) {
                const event = self.pending_tokens.orderedRemove(0);
                self.removeTextBuffer(event.token_text.ptr);
                self.release();
                return event;
            }

            if (self.done.load(.acquire)) {
                self.release();
                return null;
            }
            self.release();

            // Check timeout
            const now = std.time.nanoTimestamp();
            const elapsed_ns: u64 = @intCast(now - start);
            if (elapsed_ns >= timeout_ns) return null;

            // Wait with timeout (microseconds for __ulock_wait)
            const remaining_us: u32 = @intCast(@min(
                (timeout_ns - elapsed_ns) / 1000,
                std.math.maxInt(u32),
            ));
            self.waitForWake(counter, remaining_us);
        }
    }

    /// Check if the signal indicates an error.
    pub fn hasError(self: *CompletionSignal, io: std.Io) bool {
        _ = io;
        self.acquire();
        defer self.release();
        return self.error_msg != null;
    }

    /// Get the error message (if any). Caller does NOT own the returned slice.
    pub fn getError(self: *CompletionSignal, io: std.Io) ?[]const u8 {
        _ = io;
        self.acquire();
        defer self.release();
        return self.error_msg;
    }

    /// Check if done (all tokens delivered or error).
    pub fn isDone(self: *CompletionSignal, _io: std.Io) bool {
        _ = _io;
        return self.done.load(.acquire);
    }
};
