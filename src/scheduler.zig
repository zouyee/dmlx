/// Request Scheduler for continuous batching.
///
/// Manages waiting and running request queues, prioritizes decode-phase
/// requests over prefill-phase requests, and coordinates KV cache block
/// allocation/deallocation via a BlockManager.
///
/// Design: the scheduler operates in a three-phase Engine_Step loop:
///   1. schedule()    — select requests, allocate blocks, build batch
///   2. forward pass  — (external) run model on batched input
///   3. postprocess() — append tokens, check stop conditions, free blocks
const std = @import("std");
const paged = @import("kvcache/paged.zig");

// Re-export the real BlockManager from paged.zig.
pub const RealBlockManager = paged.BlockManager;

// ---------------------------------------------------------------------------
// BlockManager — thin wrapper around RealBlockManager for scheduler use.
// Provides a simplified API that the scheduler and its tests expect.
// ---------------------------------------------------------------------------

pub const BlockManager = struct {
    total_blocks: usize,
    used_blocks: usize,
    /// Optional reference to the real block manager for full functionality.
    real: ?*RealBlockManager,

    pub fn init(total_blocks: usize) BlockManager {
        return .{
            .total_blocks = total_blocks,
            .used_blocks = 0,
            .real = null,
        };
    }

    /// Create a BlockManager backed by a real RealBlockManager.
    pub fn initWithReal(real_bm: *RealBlockManager) BlockManager {
        return .{
            .total_blocks = real_bm.total_blocks,
            .used_blocks = real_bm.usedCount(),
            .real = real_bm,
        };
    }

    /// Check whether `num_blocks` can be allocated.
    pub fn canAllocate(self: *const BlockManager, num_blocks: usize) bool {
        if (self.real) |real| {
            return real.canAllocate(num_blocks);
        }
        return self.freeCount() >= num_blocks;
    }

    /// Allocate `num_blocks` for a request. Stores block ids in the request's block_ids list.
    pub fn allocateBlocks(self: *BlockManager, allocator: std.mem.Allocator, req: *Request, num_blocks: usize) !void {
        if (self.real) |real| {
            const block_ids = try real.allocateBlocks(req.id, num_blocks);
            defer allocator.free(block_ids);
            for (block_ids) |bid| {
                try req.block_ids.append(allocator, bid);
            }
            self.used_blocks = real.usedCount();
            return;
        }
        if (!self.canAllocate(num_blocks)) return error.InsufficientBlocks;
        for (0..num_blocks) |_| {
            const block_id = self.used_blocks;
            self.used_blocks += 1;
            try req.block_ids.append(allocator, block_id);
        }
    }

    /// Free all blocks owned by a request.
    pub fn freeBlocks(self: *BlockManager, req: *Request) void {
        if (self.real) |real| {
            real.freeBlocks(req.id);
            self.used_blocks = real.usedCount();
            req.block_ids.clearRetainingCapacity();
            return;
        }
        const freed = req.block_ids.items.len;
        if (freed <= self.used_blocks) {
            self.used_blocks -= freed;
        } else {
            self.used_blocks = 0;
        }
        req.block_ids.clearRetainingCapacity();
    }

    pub fn freeCount(self: *const BlockManager) usize {
        if (self.real) |real| {
            return real.freeCount();
        }
        return self.total_blocks - self.used_blocks;
    }
};

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

pub const RequestState = enum {
    waiting,
    prefilling,
    decoding,
    done,
};

pub const Request = struct {
    id: u64,
    prompt_tokens: []const u32,
    generated_tokens: std.ArrayList(u32),
    state: RequestState,
    block_ids: std.ArrayList(usize),
    max_tokens: usize,
    stop_tokens: []const u32,
    prefill_offset: usize,

    /// Completion flag — set to true by engine loop when generation finishes.
    done: bool,
    /// Final generated token sequence (set when done=true).
    result_tokens: ?[]const u32,

    pub fn init(
        _: std.mem.Allocator,
        id: u64,
        prompt_tokens: []const u32,
        max_tokens: usize,
        stop_tokens: []const u32,
    ) Request {
        return .{
            .id = id,
            .prompt_tokens = prompt_tokens,
            .generated_tokens = std.ArrayList(u32).empty,
            .state = .waiting,
            .block_ids = std.ArrayList(usize).empty,
            .max_tokens = max_tokens,
            .stop_tokens = stop_tokens,
            .prefill_offset = 0,
            .done = false,
            .result_tokens = null,
        };
    }

    pub fn deinit(self: *Request, allocator: std.mem.Allocator) void {
        self.generated_tokens.deinit(allocator);
        self.block_ids.deinit(allocator);
    }

    /// Mark this request as complete with the generated tokens.
    pub fn markComplete(self: *Request) void {
        self.result_tokens = self.generated_tokens.items;
        self.done = true;
    }

    /// Poll for completion. Returns generated tokens when done.
    /// Uses io.sleep for non-busy waiting between polls.
    pub fn waitForCompletion(self: *const Request, io: std.Io) []const u32 {
        while (!self.done) {
            io.sleep(.fromMilliseconds(1), .awake) catch break;
        }
        return self.result_tokens orelse &[_]u32{};
    }

    /// Total sequence length so far (prompt consumed + generated).
    pub fn seqLen(self: *const Request) usize {
        return self.prefill_offset + self.generated_tokens.items.len;
    }

    /// Check if a token is a stop token for this request.
    pub fn isStopToken(self: *const Request, token: u32) bool {
        for (self.stop_tokens) |st| {
            if (token == st) return true;
        }
        return false;
    }

    /// Returns true if the request still has prompt tokens remaining to prefill.
    pub fn hasPendingPrefill(self: *const Request) bool {
        return self.prefill_offset < self.prompt_tokens.len;
    }

    /// Compute the number of tokens in the current prefill chunk, capped by max_prefill_tokens.
    pub fn currentPrefillChunkLen(self: *const Request, max_prefill_tokens: usize) usize {
        const remaining = self.prompt_tokens.len - self.prefill_offset;
        return @min(remaining, max_prefill_tokens);
    }
};

// ---------------------------------------------------------------------------
// Token output from a forward pass (one per request in the batch)
// ---------------------------------------------------------------------------

pub const TokenOutput = struct {
    request_id: u64,
    token: u32,
};

// ---------------------------------------------------------------------------
// Schedule result
// ---------------------------------------------------------------------------

pub const ScheduleResult = struct {
    prefill_requests: []*Request,
    decode_requests: []*Request,

    /// Number of blocks needed for new prefill requests (informational).
    blocks_needed: usize,

    pub fn totalRequests(self: *const ScheduleResult) usize {
        return self.prefill_requests.len + self.decode_requests.len;
    }

    pub fn isEmpty(self: *const ScheduleResult) bool {
        return self.totalRequests() == 0;
    }
};

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

pub const Scheduler = struct {
    allocator: std.mem.Allocator,
    waiting: std.ArrayList(*Request),
    running: std.ArrayList(*Request),
    block_manager: *BlockManager,
    max_prefill_tokens: usize,
    /// Default number of blocks to allocate per new request.
    blocks_per_request: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        block_manager: *BlockManager,
        max_prefill_tokens: usize,
    ) Scheduler {
        return .{
            .allocator = allocator,
            .waiting = std.ArrayList(*Request).empty,
            .running = std.ArrayList(*Request).empty,
            .block_manager = block_manager,
            .max_prefill_tokens = max_prefill_tokens,
            .blocks_per_request = 1,
        };
    }

    pub fn deinit(self: *Scheduler) void {
        self.waiting.deinit(self.allocator);
        self.running.deinit(self.allocator);
    }

    /// Add a new request to the waiting queue.
    pub fn addRequest(self: *Scheduler, req: *Request) !void {
        req.state = .waiting;
        try self.waiting.append(self.allocator, req);
    }

    /// Schedule one Engine_Step.
    ///
    /// Priority: all running (decode-phase) requests are scheduled first,
    /// then waiting (prefill-phase) requests are promoted if blocks are available.
    ///
    /// Chunked prefill: when a request's prompt exceeds `max_prefill_tokens`,
    /// only `max_prefill_tokens` tokens are processed per Engine_Step. The
    /// request stays in `prefilling` state across multiple steps until the
    /// entire prompt is consumed. Decode steps for other active requests
    /// continue during chunked prefill.
    pub fn schedule(self: *Scheduler) !ScheduleResult {
        var decode_list = std.ArrayList(*Request).empty;
        errdefer decode_list.deinit(self.allocator);
        var prefill_list = std.ArrayList(*Request).empty;
        errdefer prefill_list.deinit(self.allocator);

        // 1. Categorize running requests: decoding requests go to decode_list,
        //    requests still mid-prefill (chunked) go to prefill_list.
        for (self.running.items) |req| {
            if (req.state == .prefilling and req.hasPendingPrefill()) {
                // Still has prompt chunks remaining — schedule as prefill.
                try prefill_list.append(self.allocator, req);
            } else {
                try decode_list.append(self.allocator, req);
            }
        }

        // 2. Promote waiting requests if blocks are available.
        var still_waiting = std.ArrayList(*Request).empty;
        errdefer still_waiting.deinit(self.allocator);

        var blocks_needed: usize = 0;

        for (self.waiting.items) |req| {
            if (self.block_manager.canAllocate(self.blocks_per_request)) {
                // Allocate blocks and move to prefill.
                try self.block_manager.allocateBlocks(self.allocator, req, self.blocks_per_request);
                req.state = .prefilling;
                // prefill_offset starts at 0; the first chunk will be
                // [0..min(prompt_len, max_prefill_tokens)].
                try prefill_list.append(self.allocator, req);
                try self.running.append(self.allocator, req);
            } else {
                // Not enough blocks — keep waiting.
                blocks_needed += self.blocks_per_request;
                try still_waiting.append(self.allocator, req);
            }
        }

        // Replace waiting queue with requests that couldn't be scheduled.
        self.waiting.deinit(self.allocator);
        self.waiting = still_waiting;

        return .{
            .decode_requests = try decode_list.toOwnedSlice(self.allocator),
            .prefill_requests = try prefill_list.toOwnedSlice(self.allocator),
            .blocks_needed = blocks_needed,
        };
    }

    /// Post-process after a forward pass: append generated tokens, check
    /// stop conditions, and free blocks for completed requests.
    ///
    /// For chunked prefill: when a prefilling request still has remaining
    /// prompt tokens, advance `prefill_offset` by the chunk size processed
    /// in this step. The request stays in `prefilling` state until the
    /// entire prompt is consumed. Only after the final chunk does the
    /// request transition to `decoding` and begin generating tokens.
    ///
    /// Requests in `decoding` state receive a generated token from the
    /// forward pass output and are checked for stop conditions.
    pub fn postprocess(self: *Scheduler, outputs: []const TokenOutput) !void {
        // Build a map from request_id → token for quick lookup.
        for (outputs) |output| {
            for (self.running.items) |req| {
                if (req.id == output.request_id) {
                    if (req.state == .prefilling) {
                        // Advance prefill_offset by the chunk that was just processed.
                        const chunk_len = req.currentPrefillChunkLen(self.max_prefill_tokens);
                        req.prefill_offset += chunk_len;

                        if (!req.hasPendingPrefill()) {
                            // All prompt tokens consumed — transition to decoding.
                            // The token from this output is the first generated token.
                            req.state = .decoding;
                            try req.generated_tokens.append(self.allocator, output.token);

                            // Check stop conditions on the first generated token.
                            const is_stop = req.isStopToken(output.token);
                            const at_max = req.generated_tokens.items.len >= req.max_tokens;
                            if (is_stop or at_max) {
                                req.state = .done;
                            }
                        }
                        // If still has pending prefill, stay in prefilling state.
                        // No token is appended — the forward pass for a prefill chunk
                        // doesn't produce a generation token until the final chunk.
                    } else {
                        // Decoding state: append the generated token.
                        try req.generated_tokens.append(self.allocator, output.token);

                        // Check stop conditions.
                        const is_stop = req.isStopToken(output.token);
                        const at_max = req.generated_tokens.items.len >= req.max_tokens;
                        if (is_stop or at_max) {
                            req.state = .done;
                        }
                    }
                    break;
                }
            }
        }

        // Remove completed requests from running queue and free their blocks.
        var i: usize = 0;
        while (i < self.running.items.len) {
            if (self.running.items[i].state == .done) {
                const req = self.running.items[i];
                self.block_manager.freeBlocks(req);
                _ = self.running.orderedRemove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Convenience: number of active (running + waiting) requests.
    pub fn activeCount(self: *const Scheduler) usize {
        return self.running.items.len + self.waiting.items.len;
    }
};
