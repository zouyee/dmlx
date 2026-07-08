/// DSpark Markov Head — lightweight speculative decoding for DeepSeek-V4-Flash.
///
/// The Markov Head provides a transition bias that conditions each draft token
/// on the previously sampled token, mitigating the "multi-modal collision" problem
/// of independent parallel prediction.
///
/// Architecture (from DSpark paper):
///   B(x_{k-1}, ·) = W1[x_{k-1}] · W2^T  ∈ R^{vocab_size}
///   corrected_logits[k] = base_logits[k] + B(x_{k-1}, ·)
///
/// Weight format (produced by scripts/extract_markov_weights.py):
///   markov_w1.bin  [vocab_size × markov_rank] f32, row-major  (Embedding lookup)
///   markov_w2.bin  [vocab_size × markov_rank] f32, row-major  (Linear projection)
///
/// Memory layout:
///   W1[token_id] = markov_w1[token_id * rank .. (token_id+1) * rank]
///   bias[v] = sum(W1[prev_token][r] * W2[v][r] for r in 0..rank)
///         = dot(W1[prev_token], W2[v])
const std = @import("std");

pub const MARKOV_RANK: usize = 256;
pub const DSPARK_BLOCK_SIZE: usize = 5;

pub const DSparkState = struct {
    allocator: std.mem.Allocator,
    /// W1: [vocab_size, markov_rank] f32
    w1: []const f32,
    /// W2: [vocab_size, markov_rank] f32
    w2: []const f32,
    /// vocab_size (from config)
    vocab_size: u32,
    /// markov_rank (from config, default 256)
    markov_rank: u32,
    /// draft block size (from config, default 5)
    block_size: u32,

    /// Load DSpark Markov Head weights from a directory containing markov_w1.bin and markov_w2.bin.
    pub fn init(allocator: std.mem.Allocator, dspark_dir: []const u8, vocab_size: u32) !DSparkState {
        const w1_path = try std.fs.path.join(allocator, &.{ dspark_dir, "markov_w1.bin" });
        defer allocator.free(w1_path);
        const w2_path = try std.fs.path.join(allocator, &.{ dspark_dir, "markov_w2.bin" });
        defer allocator.free(w2_path);

        // Read config if available to get rank/block_size
        var markov_rank: u32 = MARKOV_RANK;
        var block_size: u32 = DSPARK_BLOCK_SIZE;
        const config_path = try std.fs.path.join(allocator, &.{ dspark_dir, "dspark_config.json" });
        defer allocator.free(config_path);
        const config_path_z = try allocator.dupeZ(u8, config_path);
        defer allocator.free(config_path_z);
        const cfg_fd = std.c.open(config_path_z.ptr, .{});
        if (cfg_fd >= 0) {
            defer _ = std.c.close(cfg_fd);
            var buf: [4096]u8 = undefined;
            const n = std.c.read(cfg_fd, &buf, buf.len);
            if (n > 0) {
                const json_str = buf[0..@intCast(n)];
                if (std.mem.indexOf(u8, json_str, "\"markov_rank\"")) |idx| {
                    const after = json_str[idx + 14 ..];
                    if (parseU32FromJson(after)) |val| markov_rank = val;
                }
                if (std.mem.indexOf(u8, json_str, "\"block_size\"")) |idx| {
                    const after = json_str[idx + 13 ..];
                    if (parseU32FromJson(after)) |val| block_size = val;
                }
            }
        } else {
            std.log.info("dspark: no dspark_config.json found, using defaults (rank={d}, block_size={d})", .{ markov_rank, block_size });
        }

        const n_floats = @as(usize, vocab_size) * @as(usize, markov_rank);
        const expected_size = n_floats * @sizeOf(f32);

        // Load w1
        const w1_data = try readBinaryFile(allocator, w1_path, expected_size);
        errdefer allocator.free(w1_data);

        // Load w2
        const w2_data = try readBinaryFile(allocator, w2_path, expected_size);
        errdefer allocator.free(w2_data);

        const w1_f32: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, w1_data));
        const w2_f32: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, w2_data));

        std.log.info("dspark: loaded Markov Head (vocab={d}, rank={d}, block_size={d}, {d:.1}MB)", .{
            vocab_size,
            markov_rank,
            block_size,
            @as(f64, @floatFromInt(expected_size * 2)) / (1024.0 * 1024.0),
        });

        return .{
            .allocator = allocator,
            .w1 = w1_f32,
            .w2 = w2_f32,
            .vocab_size = vocab_size,
            .markov_rank = markov_rank,
            .block_size = block_size,
        };
    }

    pub fn deinit(self: *DSparkState) void {
        const rank = @as(usize, self.markov_rank);
        const vocab = @as(usize, self.vocab_size);
        const byte_len = vocab * rank * @sizeOf(f32);
        // Free the backing byte slices
        const w1_bytes: [*]const u8 = @ptrCast(self.w1.ptr);
        const w2_bytes: [*]const u8 = @ptrCast(self.w2.ptr);
        self.allocator.free(w1_bytes[0..byte_len]);
        self.allocator.free(w2_bytes[0..byte_len]);
    }

    /// Get the embedding vector for a given token from W1.
    /// Returns a slice of length markov_rank.
    pub fn getW1Embedding(self: *const DSparkState, token_id: u32) []const f32 {
        const rank = @as(usize, self.markov_rank);
        const offset = @as(usize, token_id) * rank;
        return self.w1[offset .. offset + rank];
    }

    /// Compute Markov transition bias for the given previous token and add to logits.
    /// bias[v] = dot(W1[prev_token], W2[v]) for all v in [0, vocab_size)
    ///
    /// This is the core operation: a single matmul of shape [1, rank] × [rank, vocab_size]
    /// which produces [1, vocab_size] bias that gets added to base_logits.
    pub fn addMarkovBias(self: *const DSparkState, prev_token: u32, logits: []f32) void {
        const rank = @as(usize, self.markov_rank);
        const vocab = @as(usize, self.vocab_size);
        const w1_row = self.getW1Embedding(prev_token);

        // logits[v] += dot(w1_row, w2[v*rank..(v+1)*rank])
        // This is a [rank] · [vocab_size, rank]^T → [vocab_size] operation
        var v: usize = 0;
        while (v < vocab) : (v += 1) {
            const w2_row = self.w2[v * rank .. v * rank + rank];
            var acc: f32 = 0.0;
            var r: usize = 0;
            // Unroll by 8 for better vectorization
            const rank_8 = rank & ~@as(usize, 7);
            while (r < rank_8) : (r += 8) {
                acc += w1_row[r + 0] * w2_row[r + 0] +
                    w1_row[r + 1] * w2_row[r + 1] +
                    w1_row[r + 2] * w2_row[r + 2] +
                    w1_row[r + 3] * w2_row[r + 3] +
                    w1_row[r + 4] * w2_row[r + 4] +
                    w1_row[r + 5] * w2_row[r + 5] +
                    w1_row[r + 6] * w2_row[r + 6] +
                    w1_row[r + 7] * w2_row[r + 7];
            }
            while (r < rank) : (r += 1) {
                acc += w1_row[r] * w2_row[r];
            }
            logits[v] += acc;
        }
    }

    /// Propose draft tokens using Markov Head correction on base logits.
    /// Given an anchor token and its logits, iteratively:
    ///   1. Add Markov bias conditioned on prev token
    ///   2. Sample (greedy argmax for temperature=0)
    ///   3. Use sampled token as prev for next position
    ///
    /// base_logits: the logits from the target model's last decode step (the anchor token's output).
    ///              This is used as the base logits for ALL draft positions (simplified DSpark).
    ///              In full DSpark, each position would have its own parallel backbone logits.
    /// anchor_token: the token that was just generated (serves as x_{k-1} for position 1)
    /// draft_out: output buffer for draft token IDs (at least block_size elements)
    /// draft_logits_out: output buffer for corrected logits used for drafting [block_size × vocab_size]
    ///                   (needed for speculative sampling verification)
    ///
    /// Returns number of draft tokens proposed (up to block_size).
    pub fn propose(
        self: *const DSparkState,
        base_logits: []const f32,
        anchor_token: u32,
        draft_out: []u32,
        draft_logits_buf: []f32,
    ) u32 {
        const vocab = @as(usize, self.vocab_size);
        const bs = @min(@as(usize, self.block_size), draft_out.len);

        var prev_token = anchor_token;
        var k: usize = 0;
        while (k < bs) : (k += 1) {
            // Copy base logits to working buffer for this position
            const offset = k * vocab;
            @memcpy(draft_logits_buf[offset .. offset + vocab], base_logits[0..vocab]);

            // Add Markov transition bias
            self.addMarkovBias(prev_token, draft_logits_buf[offset .. offset + vocab]);

            // Greedy argmax (temperature=0 drafting for best acceptance rate)
            var max_idx: u32 = 0;
            var max_val: f32 = draft_logits_buf[offset];
            var v: usize = 1;
            while (v < vocab) : (v += 1) {
                if (draft_logits_buf[offset + v] > max_val) {
                    max_val = draft_logits_buf[offset + v];
                    max_idx = @intCast(v);
                }
            }

            draft_out[k] = max_idx;
            prev_token = max_idx;
        }
        return @intCast(bs);
    }
};

// ============================================================
// Helpers
// ============================================================

fn readBinaryFile(allocator: std.mem.Allocator, path: []const u8, expected_size: usize) ![]u8 {
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    const fd = std.c.open(path_z.ptr, .{});
    if (fd < 0) {
        std.log.err("dspark: failed to open {s}", .{path});
        return error.FileNotFound;
    }
    defer _ = std.c.close(fd);

    // Verify file size
    var stat: std.c.Stat = undefined;
    if (std.c.fstat(fd, &stat) != 0) return error.StatFailed;
    const file_size = @as(usize, @intCast(stat.size));
    if (file_size != expected_size) {
        std.log.err("dspark: {s} size mismatch: got {d} bytes, expected {d}", .{ path, file_size, expected_size });
        return error.SizeMismatch;
    }

    // Allocate buffer and read
    const buf = try allocator.alloc(u8, expected_size);
    errdefer allocator.free(buf);

    var total_read: usize = 0;
    while (total_read < expected_size) {
        const n = std.c.read(fd, buf.ptr + total_read, expected_size - total_read);
        if (n <= 0) {
            std.log.err("dspark: read error at offset {d}/{d}", .{ total_read, expected_size });
            return error.ReadFailed;
        }
        total_read += @intCast(n);
    }

    return buf;
}

/// Parse a u32 from JSON text like ": 256," or ": 5}"
fn parseU32FromJson(text: []const u8) ?u32 {
    // Skip to first digit
    var i: usize = 0;
    while (i < text.len and (text[i] == ' ' or text[i] == ':' or text[i] == '\t')) : (i += 1) {}
    if (i >= text.len) return null;

    // Collect digits
    var result: u32 = 0;
    var found = false;
    while (i < text.len and text[i] >= '0' and text[i] <= '9') : (i += 1) {
        result = result * 10 + @as(u32, text[i] - '0');
        found = true;
    }
    return if (found) result else null;
}
