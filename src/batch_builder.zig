/// Continuous Batching — BatchBuilder module.
///
/// Concatenates token sequences from multiple active requests into a single
/// batched input tensor, builds per-request position indices and attention
/// masks ensuring each request attends only to its own tokens.
///
/// Used by the scheduler's Engine_Step loop:
///   schedule() → BatchBuilder.build() → forward pass → postprocess()
///
/// Requirements: R13.1, R13.2, R13.3
const std = @import("std");
const array_mod = @import("array.zig");
const ops = @import("ops.zig");
const shape_ops = @import("ops/shape.zig");
const creation = @import("ops/creation.zig");
const scheduler_mod = @import("scheduler.zig");

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const Request = scheduler_mod.Request;
const ScheduleResult = scheduler_mod.ScheduleResult;

/// Result of building a batched input from multiple requests.
pub const BatchResult = struct {
    /// Concatenated token ids: shape [total_tokens]
    batched_tokens: Array,
    /// Per-token position indices: shape [total_tokens]
    position_ids: Array,
    /// Attention mask: shape [total_tokens, total_tokens]
    /// Each element is 0.0 (attend) or -inf (block).
    attention_mask: Array,
    /// Total number of tokens in the batch.
    total_tokens: usize,
    /// Number of requests in the batch.
    num_requests: usize,

    pub fn deinit(self: *BatchResult) void {
        self.batched_tokens.deinit();
        self.position_ids.deinit();
        self.attention_mask.deinit();
    }
};

/// Collect the token sequence for a single request.
/// For prefill requests: the prompt tokens (from prefill_offset onward).
/// For decode requests: the last generated token (or prompt tail if no generated tokens yet).
fn getRequestTokens(allocator: std.mem.Allocator, req: *const Request) ![]const u32 {
    switch (req.state) {
        .prefilling, .waiting => {
            // Prefill: return prompt tokens from current offset onward.
            const start = req.prefill_offset;
            const end = req.prompt_tokens.len;
            if (start >= end) {
                // Edge case: prefill already consumed all prompt tokens,
                // return last token as a single-token decode.
                if (end > 0) {
                    const result = try allocator.alloc(u32, 1);
                    result[0] = req.prompt_tokens[end - 1];
                    return result;
                }
                return &[_]u32{};
            }
            const len = end - start;
            const result = try allocator.alloc(u32, len);
            @memcpy(result, req.prompt_tokens[start..end]);
            return result;
        },
        .decoding => {
            // Decode: return the last generated token.
            const result = try allocator.alloc(u32, 1);
            if (req.generated_tokens.items.len > 0) {
                result[0] = req.generated_tokens.items[req.generated_tokens.items.len - 1];
            } else if (req.prompt_tokens.len > 0) {
                result[0] = req.prompt_tokens[req.prompt_tokens.len - 1];
            } else {
                allocator.free(result);
                return &[_]u32{};
            }
            return result;
        },
        .done => return &[_]u32{},
    }
}

/// Build a BatchResult from a ScheduleResult.
///
/// Concatenates all request token sequences into a single flat tensor,
/// builds position_ids (per-request sequence positions), and builds a
/// causal attention mask that isolates each request's tokens.
pub fn build(
    allocator: std.mem.Allocator,
    schedule_result: *const ScheduleResult,
    ctx: EagerContext,
) !BatchResult {
    const total_requests = schedule_result.totalRequests();
    if (total_requests == 0) {
        return emptyBatch(ctx);
    }

    // Collect all requests (prefill first, then decode).
    const all_requests = try allocator.alloc(*Request, total_requests);
    defer allocator.free(all_requests);
    var idx: usize = 0;
    for (schedule_result.prefill_requests) |req| {
        all_requests[idx] = req;
        idx += 1;
    }
    for (schedule_result.decode_requests) |req| {
        all_requests[idx] = req;
        idx += 1;
    }

    // Gather per-request token sequences and compute total length.
    const req_tokens = try allocator.alloc([]const u32, total_requests);
    defer {
        for (req_tokens) |tokens| {
            if (tokens.len > 0) allocator.free(tokens);
        }
        allocator.free(req_tokens);
    }

    var total_tokens: usize = 0;
    for (all_requests, 0..) |req, i| {
        req_tokens[i] = try getRequestTokens(allocator, req);
        total_tokens += req_tokens[i].len;
    }

    if (total_tokens == 0) {
        return emptyBatch(ctx);
    }

    // Build flat token buffer, position buffer, and mask.
    const all_token_ids = try allocator.alloc(u32, total_tokens);
    defer allocator.free(all_token_ids);
    const all_positions = try allocator.alloc(i32, total_tokens);
    defer allocator.free(all_positions);

    // Attention mask: total_tokens x total_tokens
    // 0.0 = attend, large negative = block
    const mask_size = total_tokens * total_tokens;
    const mask_data = try allocator.alloc(f32, mask_size);
    defer allocator.free(mask_data);

    // Initialize mask to all-blocked (-inf).
    const neg_inf: f32 = -std.math.inf(f32);
    @memset(mask_data, neg_inf);

    var offset: usize = 0;
    for (all_requests, 0..) |req, i| {
        const tokens = req_tokens[i];
        const seq_len = tokens.len;

        // Copy token ids.
        @memcpy(all_token_ids[offset .. offset + seq_len], tokens);

        // Build position ids for this request.
        // For prefill: positions start at prefill_offset.
        // For decode: position is seqLen - 1 (the current step position).
        const base_pos: usize = switch (req.state) {
            .prefilling, .waiting => req.prefill_offset,
            .decoding => if (req.seqLen() > 0) req.seqLen() - seq_len else 0,
            .done => 0,
        };
        for (0..seq_len) |j| {
            all_positions[offset + j] = @intCast(base_pos + j);
        }

        // Set attention mask: this request's tokens can attend to each other
        // (causal within the request's span in the batch).
        for (0..seq_len) |row| {
            for (0..seq_len) |col| {
                if (col <= row) {
                    // Causal: token at position row can attend to positions 0..row
                    mask_data[(offset + row) * total_tokens + (offset + col)] = 0.0;
                }
            }
        }

        offset += seq_len;
    }

    // Create MLX arrays from the buffers.
    const total_i32: i32 = @intCast(total_tokens);
    const batched_tokens = try Array.fromData(
        allocator,
        u32,
        all_token_ids,
        &[_]i32{total_i32},
    );
    errdefer batched_tokens.deinit();

    const position_ids = try Array.fromData(
        allocator,
        i32,
        all_positions,
        &[_]i32{total_i32},
    );
    errdefer position_ids.deinit();

    const attention_mask = try Array.fromData(
        allocator,
        f32,
        mask_data,
        &[_]i32{ total_i32, total_i32 },
    );

    return .{
        .batched_tokens = batched_tokens,
        .position_ids = position_ids,
        .attention_mask = attention_mask,
        .total_tokens = total_tokens,
        .num_requests = total_requests,
    };
}

/// Create an empty batch result (no requests to process).
fn emptyBatch(ctx: EagerContext) !BatchResult {
    _ = ctx;
    const empty_u32 = try Array.fromData(
        std.heap.page_allocator,
        u32,
        &[_]u32{},
        &[_]i32{0},
    );
    errdefer empty_u32.deinit();

    const empty_i32 = try Array.fromData(
        std.heap.page_allocator,
        i32,
        &[_]i32{},
        &[_]i32{0},
    );
    errdefer empty_i32.deinit();

    const empty_mask = try Array.fromData(
        std.heap.page_allocator,
        f32,
        &[_]f32{},
        &[_]i32{ 0, 0 },
    );

    return .{
        .batched_tokens = empty_u32,
        .position_ids = empty_i32,
        .attention_mask = empty_mask,
        .total_tokens = 0,
        .num_requests = 0,
    };
}

/// Check if a token position in the mask is allowed to attend.
/// Returns true if mask value is 0.0 (attend), false if -inf (blocked).
pub fn isAttending(mask: []const f32, total_tokens: usize, row: usize, col: usize) bool {
    const val = mask[row * total_tokens + col];
    return val == 0.0;
}
