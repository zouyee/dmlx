/// PagedKVCache — Block/page-based KV cache for continuous batching.
///
/// Strategy: allocate KV storage in fixed-size pages (blocks).
/// Each sequence gets a page table mapping logical positions to physical pages.
/// Pages can be reused across sequences after filter/reset.
///
/// Current implementation uses pages for memory management and continuous batching,
/// while maintaining contiguous cached arrays for attention compatibility.
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const ops = @import("../ops.zig");
const shape_mod = @import("../ops/shape.zig");
const iface = @import("interface.zig");
const quantized_mod = @import("quantized.zig");

const Array = array_mod.Array;
const KVSlice = iface.KVSlice;
const KVCacheStrategy = iface.KVCacheStrategy;
const LayerConfig = iface.LayerConfig;
const QuantizedTuple = quantized_mod.QuantizedKVCache.QuantizedTuple;

/// Default page size tuned for Apple Silicon Metal GPU.
pub const default_page_size = 32;

// ---------------------------------------------------------------------------
// Block — enhanced KV cache block with ref_count, hash, access tracking
// ---------------------------------------------------------------------------

pub const Block = struct {
    keys: ?Array,
    values: ?Array,
    used: bool,
    ref_count: usize,
    hash: ?u64,
    last_access: u64,
    tokens_used: usize,
};

// ---------------------------------------------------------------------------
// BlockManager — manages a pool of blocks with alloc/free/CoW/prefix caching
// ---------------------------------------------------------------------------

pub const BlockManager = struct {
    allocator: std.mem.Allocator,
    free_blocks: std.ArrayList(usize),
    block_pool: std.ArrayList(Block),
    req_to_blocks: std.AutoHashMap(u64, std.ArrayList(usize)),
    block_hashes: std.AutoHashMap(u64, usize),
    total_blocks: usize,
    access_counter: u64,

    pub fn init(allocator: std.mem.Allocator, total_blocks: usize) !BlockManager {
        var free_blocks = std.ArrayList(usize).empty;
        var block_pool = std.ArrayList(Block).empty;

        // Pre-allocate all blocks into the pool and free list.
        try block_pool.ensureTotalCapacity(allocator, total_blocks);
        try free_blocks.ensureTotalCapacity(allocator, total_blocks);

        for (0..total_blocks) |i| {
            block_pool.appendAssumeCapacity(.{
                .keys = null,
                .values = null,
                .used = false,
                .ref_count = 0,
                .hash = null,
                .last_access = 0,
                .tokens_used = 0,
            });
            // Push in reverse so that pop gives lowest index first.
            free_blocks.appendAssumeCapacity(total_blocks - 1 - i);
        }

        return .{
            .allocator = allocator,
            .free_blocks = free_blocks,
            .block_pool = block_pool,
            .req_to_blocks = std.AutoHashMap(u64, std.ArrayList(usize)).init(allocator),
            .block_hashes = std.AutoHashMap(u64, usize).init(allocator),
            .total_blocks = total_blocks,
            .access_counter = 0,
        };
    }

    pub fn deinit(self: *BlockManager) void {
        self.free_blocks.deinit(self.allocator);
        for (self.block_pool.items) |*block| {
            if (block.keys) |k| k.deinit();
            if (block.values) |v| v.deinit();
        }
        self.block_pool.deinit(self.allocator);
        var it = self.req_to_blocks.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.req_to_blocks.deinit();
        self.block_hashes.deinit();
    }

    /// Check if enough free blocks exist for an allocation.
    pub fn canAllocate(self: *const BlockManager, num_blocks: usize) bool {
        return self.free_blocks.items.len >= num_blocks;
    }

    /// Allocate `num_blocks` from the free pool for `req_id`.
    /// Returns a slice of allocated block indices (owned by the caller).
    pub fn allocateBlocks(self: *BlockManager, req_id: u64, num_blocks: usize) ![]usize {
        if (!self.canAllocate(num_blocks)) return error.InsufficientBlocks;

        self.access_counter += 1;

        const result = try self.allocator.alloc(usize, num_blocks);
        errdefer self.allocator.free(result);

        for (0..num_blocks) |i| {
            const block_id = self.free_blocks.pop().?;
            const block = &self.block_pool.items[block_id];
            block.used = true;
            block.ref_count = 1;
            block.last_access = self.access_counter;
            block.tokens_used = 0;
            block.hash = null;
            result[i] = block_id;
        }

        // Track per-request ownership.
        const gop = try self.req_to_blocks.getOrPut(req_id);
        if (!gop.found_existing) {
            gop.value_ptr.* = std.ArrayList(usize).empty;
        }
        for (result) |block_id| {
            try gop.value_ptr.append(self.allocator, block_id);
        }

        return result;
    }

    /// Free all blocks owned by `req_id`, returning them to the free pool.
    /// Blocks with ref_count > 1 (shared) have their ref_count decremented
    /// instead of being freed.
    pub fn freeBlocks(self: *BlockManager, req_id: u64) void {
        const entry = self.req_to_blocks.fetchRemove(req_id);
        if (entry) |kv| {
            var block_list = kv.value;
            for (block_list.items) |block_id| {
                if (block_id < self.block_pool.items.len) {
                    const block = &self.block_pool.items[block_id];
                    if (block.ref_count > 1) {
                        block.ref_count -= 1;
                    } else {
                        block.ref_count = 0;
                        block.used = false;
                        block.hash = null;
                        block.tokens_used = 0;
                        self.free_blocks.append(self.allocator, block_id) catch {};
                    }
                }
            }
            block_list.deinit(self.allocator);
        }
    }

    /// Copy-on-Write: if the block at `block_id` is shared (ref_count > 1),
    /// allocate a new block, copy the data, decrement the original's ref_count,
    /// and return the new block's id. If not shared, returns the same block_id.
    pub fn copyOnWrite(self: *BlockManager, block_id: usize) !usize {
        if (block_id >= self.block_pool.items.len) return error.InvalidBlockId;

        const block = &self.block_pool.items[block_id];
        if (block.ref_count <= 1) return block_id;

        // Need a free block for the copy.
        if (self.free_blocks.items.len == 0) return error.InsufficientBlocks;

        const new_id = self.free_blocks.pop().?;
        const new_block = &self.block_pool.items[new_id];

        // Copy metadata.
        new_block.used = true;
        new_block.ref_count = 1;
        new_block.hash = block.hash;
        new_block.last_access = self.access_counter;
        new_block.tokens_used = block.tokens_used;

        // Copy array data (if present) using mlx_array_set.
        if (block.keys) |k| {
            var k_copy = c.c.mlx_array_new();
            try c.check(c.c.mlx_array_set(&k_copy, k.inner));
            new_block.keys = Array.fromHandle(k_copy);
        } else {
            new_block.keys = null;
        }
        if (block.values) |v| {
            var v_copy = c.c.mlx_array_new();
            try c.check(c.c.mlx_array_set(&v_copy, v.inner));
            new_block.values = Array.fromHandle(v_copy);
        } else {
            new_block.values = null;
        }

        // Decrement original's ref_count.
        block.ref_count -= 1;

        return new_id;
    }

    /// Number of free blocks available.
    pub fn freeCount(self: *const BlockManager) usize {
        return self.free_blocks.items.len;
    }

    /// Number of used blocks.
    pub fn usedCount(self: *const BlockManager) usize {
        return self.total_blocks - self.free_blocks.items.len;
    }

    // -----------------------------------------------------------------------
    // Prefix caching
    // -----------------------------------------------------------------------

    /// Compute a deterministic hash for a block given the previous block's hash
    /// and the token IDs in the current block.
    pub fn hashBlock(prev_hash: u64, token_ids: []const u32) u64 {
        var hasher = std.hash.Wyhash.init(prev_hash);
        // Hash the raw bytes of the token_ids slice.
        const bytes = std.mem.sliceAsBytes(token_ids);
        hasher.update(bytes);
        return hasher.final();
    }

    /// Search for cached blocks matching a token prefix.
    /// Returns a list of block IDs that can be reused (caller owns the slice).
    pub fn findCachedPrefix(self: *BlockManager, token_ids: []const u32, block_size: usize) ![]usize {
        if (block_size == 0) return try self.allocator.alloc(usize, 0);

        var result = std.ArrayList(usize).empty;
        errdefer result.deinit(self.allocator);

        var prev_hash: u64 = 0;
        var offset: usize = 0;

        while (offset + block_size <= token_ids.len) {
            const block_tokens = token_ids[offset .. offset + block_size];
            const hash = hashBlock(prev_hash, block_tokens);

            if (self.block_hashes.get(hash)) |cached_block_id| {
                const block = &self.block_pool.items[cached_block_id];
                if (block.used and block.tokens_used == block_size) {
                    // Reuse this block — increment ref_count.
                    block.ref_count += 1;
                    self.access_counter += 1;
                    block.last_access = self.access_counter;
                    try result.append(self.allocator, cached_block_id);
                    prev_hash = hash;
                    offset += block_size;
                    continue;
                }
            }
            // No cached block found for this position — stop searching.
            break;
        }

        return try result.toOwnedSlice(self.allocator);
    }

    /// Register a block's hash in the block_hashes map for prefix caching.
    pub fn registerBlockHash(self: *BlockManager, block_id: usize, hash: u64) !void {
        if (block_id >= self.block_pool.items.len) return error.InvalidBlockId;
        self.block_pool.items[block_id].hash = hash;
        try self.block_hashes.put(hash, block_id);
    }
};

// ---------------------------------------------------------------------------
// Legacy Page type (used internally by PagedKVCache)
// ---------------------------------------------------------------------------

/// A physical page of KV storage.
/// Shape: [1, num_kv_heads, page_size, head_dim]
/// When quantized (kv_bits < 16), keys/values hold quantized packed data via QuantizedTuple.
const Page = struct {
    keys: Array,
    values: Array,
    /// Quantized storage for keys (used when kv_bits < 16).
    quantized_keys: ?QuantizedTuple,
    /// Quantized storage for values (used when kv_bits < 16).
    quantized_values: ?QuantizedTuple,
    used: bool,
    ref_count: usize,
};

/// Per-sequence page table entry.
const PageTableEntry = struct {
    physical: usize,
};

/// Per-sequence cache state.
const SequenceState = struct {
    pages: std.ArrayList(PageTableEntry),
    cached_len: usize,
};

/// PagedKVCache manages a pool of physical pages and per-sequence page tables.
pub const PagedKVCache = struct {
    allocator: std.mem.Allocator,

    pages: std.ArrayList(Page),
    sequences: std.ArrayList(SequenceState),

    page_size: usize,
    batch_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_pages: usize,
    dtype: @import("../dtype.zig").Dtype,

    // Quantization parameters (kv_bits=16 means no quantization).
    kv_bits: u8,
    group_size: i32,
    el_per_int: i32,

    // Contiguous cached arrays per batch entry for attention compatibility.
    cached_keys: std.ArrayList(?Array),
    cached_values: std.ArrayList(?Array),

    // Prefix caching: maps page hash → physical page index for reuse.
    page_hashes: std.AutoHashMap(u64, usize),
    // Per-sequence running hash for prefix chain computation.
    seq_prev_hashes: std.ArrayList(u64),

    pub const vtable: iface.VTable = .{
        .updateAndFetch = updateAndFetchImpl,
        .currentLen = currentLenImpl,
        .reset = resetImpl,
        .filter = filterImpl,
        .deinit = deinitImpl,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        config: LayerConfig,
        page_size: usize,
        kv_bits: u8,
        group_size: i32,
        stream: c.c.mlx_stream,
    ) !PagedKVCache {
        _ = stream;

        // Validate kv_bits.
        switch (kv_bits) {
            4, 8, 16 => {},
            else => return error.InvalidQuantBits,
        }

        const el_per_int: i32 = if (kv_bits < 16) @divTrunc(8 * @as(i32, @sizeOf(u32)), @as(i32, kv_bits)) else 1;
        const max_pages = (config.max_seq_len + page_size - 1) / page_size * config.batch_size;

        var pages = std.ArrayList(Page).empty;
        errdefer pages.deinit(allocator);

        var sequences = std.ArrayList(SequenceState).empty;
        errdefer {
            for (sequences.items) |*seq| seq.pages.deinit(allocator);
            sequences.deinit(allocator);
        }

        var cached_keys = std.ArrayList(?Array).empty;
        errdefer cached_keys.deinit(allocator);
        var cached_values = std.ArrayList(?Array).empty;
        errdefer cached_values.deinit(allocator);

        try sequences.ensureTotalCapacity(allocator, config.batch_size);
        try cached_keys.ensureTotalCapacity(allocator, config.batch_size);
        try cached_values.ensureTotalCapacity(allocator, config.batch_size);

        for (0..config.batch_size) |_| {
            sequences.appendAssumeCapacity(.{
                .pages = std.ArrayList(PageTableEntry).empty,
                .cached_len = 0,
            });
            cached_keys.appendAssumeCapacity(null);
            cached_values.appendAssumeCapacity(null);
        }

        return .{
            .allocator = allocator,
            .pages = pages,
            .sequences = sequences,
            .page_size = page_size,
            .batch_size = config.batch_size,
            .num_kv_heads = config.num_kv_heads,
            .head_dim = config.head_dim,
            .max_pages = max_pages,
            .dtype = config.dtype,
            .kv_bits = kv_bits,
            .group_size = group_size,
            .el_per_int = el_per_int,
            .cached_keys = cached_keys,
            .cached_values = cached_values,
            .page_hashes = std.AutoHashMap(u64, usize).init(allocator),
            .seq_prev_hashes = blk: {
                var h = std.ArrayList(u64).empty;
                try h.ensureTotalCapacity(allocator, config.batch_size);
                for (0..config.batch_size) |_| h.appendAssumeCapacity(0);
                break :blk h;
            },
        };
    }

    pub fn asStrategy(self: *PagedKVCache) KVCacheStrategy {
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn allocPage(self: *PagedKVCache, stream: c.c.mlx_stream) !usize {
        for (self.pages.items, 0..) |*page, i| {
            if (!page.used) {
                page.used = true;
                page.ref_count = 1;
                return i;
            }
        }

        if (self.pages.items.len >= self.max_pages) {
            return error.PagePoolExhausted;
        }

        if (self.kv_bits < 16) {
            // Quantized page: allocate packed_data + scales + biases for keys and values.
            const base_shape = &[_]i32{
                1,
                @intCast(self.num_kv_heads),
                @intCast(self.page_size),
            };
            const head_dim_i32: i32 = @intCast(self.head_dim);

            const q_keys = try initQuantizedPageBuffer(base_shape, head_dim_i32, self.el_per_int, self.group_size, stream);
            const q_values = try initQuantizedPageBuffer(base_shape, head_dim_i32, self.el_per_int, self.group_size, stream);

            // Also allocate dummy float arrays for the keys/values fields (zero-size placeholder).
            const shape = &[_]i32{ 1, @intCast(self.num_kv_heads), @intCast(self.page_size), @intCast(self.head_dim) };
            var keys_arr = c.c.mlx_array_new();
            var values_arr = c.c.mlx_array_new();
            try c.check(c.c.mlx_zeros(&keys_arr, shape.ptr, shape.len, @intCast(@intFromEnum(self.dtype)), stream));
            try c.check(c.c.mlx_zeros(&values_arr, shape.ptr, shape.len, @intCast(@intFromEnum(self.dtype)), stream));

            try self.pages.append(self.allocator, .{
                .keys = Array.fromHandle(keys_arr),
                .values = Array.fromHandle(values_arr),
                .quantized_keys = q_keys,
                .quantized_values = q_values,
                .used = true,
                .ref_count = 1,
            });
        } else {
            // Standard float page.
            const shape = &[_]i32{
                1,
                @intCast(self.num_kv_heads),
                @intCast(self.page_size),
                @intCast(self.head_dim),
            };

            var keys_arr = c.c.mlx_array_new();
            var values_arr = c.c.mlx_array_new();

            try c.check(c.c.mlx_zeros(&keys_arr, shape.ptr, shape.len, @intCast(@intFromEnum(self.dtype)), stream));
            try c.check(c.c.mlx_zeros(&values_arr, shape.ptr, shape.len, @intCast(@intFromEnum(self.dtype)), stream));

            try self.pages.append(self.allocator, .{
                .keys = Array.fromHandle(keys_arr),
                .values = Array.fromHandle(values_arr),
                .quantized_keys = null,
                .quantized_values = null,
                .used = true,
                .ref_count = 1,
            });
        }

        return self.pages.items.len - 1;
    }

    fn seqLen(self: *PagedKVCache, batch_idx: usize) usize {
        if (batch_idx >= self.sequences.items.len) return 0;
        return self.sequences.items[batch_idx].cached_len;
    }

    fn freeCached(self: *PagedKVCache, batch_idx: usize) void {
        if (self.cached_keys.items[batch_idx]) |arr| {
            arr.deinit();
            self.cached_keys.items[batch_idx] = null;
        }
        if (self.cached_values.items[batch_idx]) |arr| {
            arr.deinit();
            self.cached_values.items[batch_idx] = null;
        }
    }

    // ------------------------------------------------------------------
    // VTable implementations
    // ------------------------------------------------------------------

    fn updateAndFetchImpl(
        ctx: *anyopaque,
        keys: Array,
        values: Array,
        stream: c.c.mlx_stream,
    ) anyerror!KVSlice {
        const self: *PagedKVCache = @ptrCast(@alignCast(ctx));
        const shape = keys.shape();
        const batch = @as(usize, @intCast(shape[0]));
        const seq_len = @as(usize, @intCast(shape[2]));

        var out_keys = try self.allocator.alloc(Array, batch);
        defer self.allocator.free(out_keys);
        var out_values = try self.allocator.alloc(Array, batch);
        defer self.allocator.free(out_values);

        for (0..batch) |b| {
            const seq = &self.sequences.items[b];

            // Ensure we have enough pages for new tokens.
            const current_len = seq.cached_len;
            const new_total = current_len + seq_len;
            const pages_needed = (new_total + self.page_size - 1) / self.page_size;
            while (seq.pages.items.len < pages_needed) {
                const phys_idx = try self.allocPage(stream);
                try seq.pages.append(self.allocator, .{ .physical = phys_idx });
            }

            // Write new tokens into pages via slice_update.
            var written: usize = 0;
            while (written < seq_len) {
                const global_pos = current_len + written;
                const page_idx = global_pos / self.page_size;
                const page_offset = global_pos % self.page_size;
                const pt = seq.pages.items[page_idx];
                const page = &self.pages.items[pt.physical];

                const write_len = @min(seq_len - written, self.page_size - page_offset);

                // Slice new keys for this batch and write range.
                const k_start = &[_]i32{ @intCast(b), 0, @intCast(written), 0 };
                const k_stop = &[_]i32{ @intCast(b + 1), std.math.maxInt(i32), @intCast(written + write_len), std.math.maxInt(i32) };
                const k_strides = &[_]i32{ 1, 1, 1, 1 };
                var new_k_slice= c.c.mlx_array_new();
                try c.check(c.c.mlx_slice(
                    &new_k_slice,
                    keys.inner,
                    k_start.ptr,
                    k_start.len,
                    k_stop.ptr,
                    k_stop.len,
                    k_strides.ptr,
                    k_strides.len,
                    stream,
                ));
                defer _ = c.c.mlx_array_free(new_k_slice);

                // Slice new values for this batch and write range.
                const v_start = &[_]i32{ @intCast(b), 0, @intCast(written), 0 };
                const v_stop = &[_]i32{ @intCast(b + 1), std.math.maxInt(i32), @intCast(written + write_len), std.math.maxInt(i32) };
                const v_strides = &[_]i32{ 1, 1, 1, 1 };
                var new_v_slice= c.c.mlx_array_new();
                try c.check(c.c.mlx_slice(
                    &new_v_slice,
                    values.inner,
                    v_start.ptr,
                    v_start.len,
                    v_stop.ptr,
                    v_stop.len,
                    v_strides.ptr,
                    v_strides.len,
                    stream,
                ));
                defer _ = c.c.mlx_array_free(new_v_slice);

                if (self.kv_bits < 16) {
                    // Quantized path: quantize the slice, then write quantized data into page.
                    const q_k = try quantizeArray(Array.fromHandle(new_k_slice), self.group_size, @intCast(self.kv_bits), stream);
                    defer {
                        q_k.packed_data.deinit();
                        q_k.scales.deinit();
                        q_k.biases.deinit();
                    }
                    const q_v = try quantizeArray(Array.fromHandle(new_v_slice), self.group_size, @intCast(self.kv_bits), stream);
                    defer {
                        q_v.packed_data.deinit();
                        q_v.scales.deinit();
                        q_v.biases.deinit();
                    }

                    // Write quantized keys into page's quantized storage.
                    try writeQuantizedSlice(&page.quantized_keys.?, q_k, page_offset, page_offset + write_len, stream);
                    try writeQuantizedSlice(&page.quantized_values.?, q_v, page_offset, page_offset + write_len, stream);
                } else {
                    // Standard float path: write raw data into page.
                    const upd_k_start = &[_]i32{ 0, 0, @intCast(page_offset), 0 };
                    const upd_k_stop = &[_]i32{ 1, std.math.maxInt(i32), @intCast(page_offset + write_len), std.math.maxInt(i32) };
                    const upd_k_strides = &[_]i32{ 1, 1, 1, 1 };
                    var updated_keys= c.c.mlx_array_new();
                    try c.check(c.c.mlx_slice_update(
                        &updated_keys,
                        page.keys.inner,
                        new_k_slice,
                        upd_k_start.ptr,
                        upd_k_start.len,
                        upd_k_stop.ptr,
                        upd_k_stop.len,
                        upd_k_strides.ptr,
                        upd_k_strides.len,
                        stream,
                    ));
                    page.keys.deinit();
                    page.keys = Array.fromHandle(updated_keys);

                    const upd_v_start = &[_]i32{ 0, 0, @intCast(page_offset), 0 };
                    const upd_v_stop = &[_]i32{ 1, std.math.maxInt(i32), @intCast(page_offset + write_len), std.math.maxInt(i32) };
                    const upd_v_strides = &[_]i32{ 1, 1, 1, 1 };
                    var updated_values= c.c.mlx_array_new();
                    try c.check(c.c.mlx_slice_update(
                        &updated_values,
                        page.values.inner,
                        new_v_slice,
                        upd_v_start.ptr,
                        upd_v_start.len,
                        upd_v_stop.ptr,
                        upd_v_stop.len,
                        upd_v_strides.ptr,
                        upd_v_strides.len,
                        stream,
                    ));
                    page.values.deinit();
                    page.values = Array.fromHandle(updated_values);
                }

                written += write_len;

                // Register page hash when a page is fully written (prefix caching).
                // A full page can be identified by its content hash for future reuse.
                if (page_offset + write_len == self.page_size) {
                    const prev_hash = self.seq_prev_hashes.items[b];
                    // Use page index as a simple content identifier for hashing.
                    // In a full implementation, token IDs would be hashed here.
                    var hasher = std.hash.Wyhash.init(prev_hash);
                    hasher.update(std.mem.asBytes(&pt.physical));
                    hasher.update(std.mem.asBytes(&page_idx));
                    const page_hash = hasher.final();
                    self.page_hashes.put(page_hash, pt.physical) catch {};
                    self.seq_prev_hashes.items[b] = page_hash;
                }
            }

            seq.cached_len = new_total;

            // Gather pages into contiguous array for this batch entry.
            if (self.kv_bits < 16) {
                // Quantized path: dequantize each page, then gather.
                if (seq.pages.items.len == 1) {
                    const pt = seq.pages.items[0];
                    const page = &self.pages.items[pt.physical];
                    const dk = try dequantizeTuple(page.quantized_keys.?, new_total, self.group_size, @intCast(self.kv_bits), stream);
                    const dv = try dequantizeTuple(page.quantized_values.?, new_total, self.group_size, @intCast(self.kv_bits), stream);
                    out_keys[b] = dk;
                    out_values[b] = dv;
                } else {
                    var page_keys_arr = try self.allocator.alloc(Array, seq.pages.items.len);
                    defer self.allocator.free(page_keys_arr);
                    var page_values_arr = try self.allocator.alloc(Array, seq.pages.items.len);
                    defer self.allocator.free(page_values_arr);

                    for (seq.pages.items, 0..) |pt, pi| {
                        const page = &self.pages.items[pt.physical];
                        const is_last = (pi == seq.pages.items.len - 1);
                        const slice_len = if (is_last) (new_total - pi * self.page_size) else self.page_size;

                        page_keys_arr[pi] = try dequantizeTuple(page.quantized_keys.?, slice_len, self.group_size, @intCast(self.kv_bits), stream);
                        page_values_arr[pi] = try dequantizeTuple(page.quantized_values.?, slice_len, self.group_size, @intCast(self.kv_bits), stream);
                    }

                    // Concatenate along axis 2 (seq_len).
                    const k_vec2 = c.c.mlx_vector_array_new();
                    defer _ = c.c.mlx_vector_array_free(k_vec2);
                    for (page_keys_arr) |arr| {
                        try c.check(c.c.mlx_vector_array_append_data(k_vec2, &arr.inner, 1));
                    }
                    var k_concat2 = c.c.mlx_array_new();
                    try c.check(c.c.mlx_concatenate_axis(&k_concat2, k_vec2, 2, stream));
                    out_keys[b] = Array.fromHandle(k_concat2);

                    const v_vec2 = c.c.mlx_vector_array_new();
                    defer _ = c.c.mlx_vector_array_free(v_vec2);
                    for (page_values_arr) |arr| {
                        try c.check(c.c.mlx_vector_array_append_data(v_vec2, &arr.inner, 1));
                    }
                    var v_concat2 = c.c.mlx_array_new();
                    try c.check(c.c.mlx_concatenate_axis(&v_concat2, v_vec2, 2, stream));
                    out_values[b] = Array.fromHandle(v_concat2);

                    for (page_keys_arr) |arr| arr.deinit();
                    for (page_values_arr) |arr| arr.deinit();
                }
            } else if (seq.pages.items.len == 1) {
                const pt = seq.pages.items[0];
                const page = &self.pages.items[pt.physical];
                // Slice to actual length.
                const g_start = &[_]i32{ 0, 0, 0, 0 };
                const g_strides = &[_]i32{ 1, 1, 1, 1 };
                const g_k_stop = &[_]i32{ 1, std.math.maxInt(i32), @intCast(new_total), std.math.maxInt(i32) };
                var k_contig= c.c.mlx_array_new();
                try c.check(c.c.mlx_slice(
                    &k_contig,
                    page.keys.inner,
                    g_start.ptr,
                    g_start.len,
                    g_k_stop.ptr,
                    g_k_stop.len,
                    g_strides.ptr,
                    g_strides.len,
                    stream,
                ));
                var v_contig= c.c.mlx_array_new();
                try c.check(c.c.mlx_slice(
                    &v_contig,
                    page.values.inner,
                    g_start.ptr,
                    g_start.len,
                    g_k_stop.ptr,
                    g_k_stop.len,
                    g_strides.ptr,
                    g_strides.len,
                    stream,
                ));
                out_keys[b] = Array.fromHandle(k_contig);
                out_values[b] = Array.fromHandle(v_contig);
            } else {
                var page_keys = try self.allocator.alloc(Array, seq.pages.items.len);
                defer self.allocator.free(page_keys);
                var page_values = try self.allocator.alloc(Array, seq.pages.items.len);
                defer self.allocator.free(page_values);

                for (seq.pages.items, 0..) |pt, pi| {
                    const page = &self.pages.items[pt.physical];
                    const is_last = (pi == seq.pages.items.len - 1);
                    const slice_len = if (is_last) (new_total - pi * self.page_size) else self.page_size;

                    const p_start = &[_]i32{ 0, 0, 0, 0 };
                    const p_stop = &[_]i32{ 1, std.math.maxInt(i32), @intCast(slice_len), std.math.maxInt(i32) };
                    const p_strides = &[_]i32{ 1, 1, 1, 1 };
                    var k_slice= c.c.mlx_array_new();
                    try c.check(c.c.mlx_slice(
                        &k_slice,
                        page.keys.inner,
                        p_start.ptr,
                        p_start.len,
                        p_stop.ptr,
                        p_stop.len,
                        p_strides.ptr,
                        p_strides.len,
                        stream,
                    ));
                    page_keys[pi] = Array.fromHandle(k_slice);

                    var v_slice= c.c.mlx_array_new();
                    try c.check(c.c.mlx_slice(
                        &v_slice,
                        page.values.inner,
                        p_start.ptr,
                        p_start.len,
                        p_stop.ptr,
                        p_stop.len,
                        p_strides.ptr,
                        p_strides.len,
                        stream,
                    ));
                    page_values[pi] = Array.fromHandle(v_slice);
                }

                // Concatenate along axis 2 (seq_len).
                const k_vec = c.c.mlx_vector_array_new();
                defer _ = c.c.mlx_vector_array_free(k_vec);
                for (page_keys) |arr| {
                    try c.check(c.c.mlx_vector_array_append_data(k_vec, &arr.inner, 1));
                }
                var k_concat= c.c.mlx_array_new();
                try c.check(c.c.mlx_concatenate_axis(&k_concat, k_vec, 2, stream));
                out_keys[b] = Array.fromHandle(k_concat);

                const v_vec = c.c.mlx_vector_array_new();
                defer _ = c.c.mlx_vector_array_free(v_vec);
                for (page_values) |arr| {
                    try c.check(c.c.mlx_vector_array_append_data(v_vec, &arr.inner, 1));
                }
                var v_concat= c.c.mlx_array_new();
                try c.check(c.c.mlx_concatenate_axis(&v_concat, v_vec, 2, stream));
                out_values[b] = Array.fromHandle(v_concat);

                for (page_keys) |arr| arr.deinit();
                for (page_values) |arr| arr.deinit();
            }

            // Update cached arrays — store copies so caller can independently free the returned arrays.
            self.freeCached(b);
            var ck = c.c.mlx_array_new();
            c.check(c.c.mlx_array_set(&ck, out_keys[b].inner)) catch {};
            self.cached_keys.items[b] = Array.fromHandle(ck);
            var cv = c.c.mlx_array_new();
            c.check(c.c.mlx_array_set(&cv, out_values[b].inner)) catch {};
            self.cached_values.items[b] = Array.fromHandle(cv);
        }

        // Concatenate all batch entries along axis 0.
        if (batch == 1) {
            return .{ .keys = out_keys[0], .values = out_values[0] };
        }

        const k_vec = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(k_vec);
        for (out_keys) |arr| {
            try c.check(c.c.mlx_vector_array_append_data(k_vec, &arr.inner, 1));
        }
        var k_result= c.c.mlx_array_new();
        try c.check(c.c.mlx_concatenate_axis(&k_result, k_vec, 0, stream));

        const v_vec = c.c.mlx_vector_array_new();
        defer _ = c.c.mlx_vector_array_free(v_vec);
        for (out_values) |arr| {
            try c.check(c.c.mlx_vector_array_append_data(v_vec, &arr.inner, 1));
        }
        var v_result= c.c.mlx_array_new();
        try c.check(c.c.mlx_concatenate_axis(&v_result, v_vec, 0, stream));

        // Clean up per-batch arrays since we now own result.
        for (out_keys) |arr| arr.deinit();
        for (out_values) |arr| arr.deinit();

        return .{ .keys = Array.fromHandle(k_result), .values = Array.fromHandle(v_result) };
    }

    fn currentLenImpl(ctx: *anyopaque) usize {
        const self: *PagedKVCache = @ptrCast(@alignCast(ctx));
        var max_len: usize = 0;
        for (self.sequences.items) |seq| {
            if (seq.cached_len > max_len) max_len = seq.cached_len;
        }
        return max_len;
    }

    fn resetImpl(ctx: *anyopaque) void {
        const self: *PagedKVCache = @ptrCast(@alignCast(ctx));
        for (self.pages.items) |*page| {
            page.used = false;
            page.ref_count = 0;
        }
        for (self.sequences.items, 0..) |*seq, i| {
            seq.pages.clearRetainingCapacity();
            seq.cached_len = 0;
            self.freeCached(i);
        }
    }

    fn filterImpl(
        ctx: *anyopaque,
        indices: []const usize,
        allocator: std.mem.Allocator,
    ) !void {
        const self: *PagedKVCache = @ptrCast(@alignCast(ctx));
        _ = allocator;

        var new_sequences = std.ArrayList(SequenceState).empty;
        errdefer {
            for (new_sequences.items) |*seq| seq.pages.deinit(self.allocator);
            new_sequences.deinit(self.allocator);
        }

        var new_cached_keys = std.ArrayList(?Array).empty;
        errdefer new_cached_keys.deinit(self.allocator);
        var new_cached_values = std.ArrayList(?Array).empty;
        errdefer new_cached_values.deinit(self.allocator);

        // Mark all pages as potentially unused.
        for (self.pages.items) |*page| {
            page.ref_count = 0;
        }

        for (indices) |idx| {
            if (idx >= self.sequences.items.len) return error.InvalidBatchIndex;
            const seq = self.sequences.items[idx];
            try new_sequences.append(self.allocator, seq);
            try new_cached_keys.append(self.allocator, self.cached_keys.items[idx]);
            try new_cached_values.append(self.allocator, self.cached_values.items[idx]);

            // Increment ref counts for pages still in use.
            for (seq.pages.items) |pt| {
                self.pages.items[pt.physical].ref_count += 1;
            }
        }

        // Free sequences and cached arrays that are no longer referenced.
        for (self.sequences.items, 0..) |*seq, i| {
            var keep = false;
            for (indices) |idx| {
                if (idx == i) {
                    keep = true;
                    break;
                }
            }
            if (!keep) {
                seq.pages.deinit(self.allocator);
                self.freeCached(i);
            }
        }

        self.sequences.deinit(self.allocator);
        self.sequences = new_sequences;
        self.cached_keys.deinit(self.allocator);
        self.cached_keys = new_cached_keys;
        self.cached_values.deinit(self.allocator);
        self.cached_values = new_cached_values;
        self.batch_size = indices.len;

        // Mark unused pages.
        for (self.pages.items) |*page| {
            if (page.ref_count == 0) {
                page.used = false;
            }
        }
    }

    fn deinitImpl(ctx: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *PagedKVCache = @ptrCast(@alignCast(ctx));
        for (self.pages.items) |*page| {
            page.keys.deinit();
            page.values.deinit();
            if (page.quantized_keys) |qk| {
                qk.packed_data.deinit();
                qk.scales.deinit();
                qk.biases.deinit();
            }
            if (page.quantized_values) |qv| {
                qv.packed_data.deinit();
                qv.scales.deinit();
                qv.biases.deinit();
            }
        }
        self.pages.deinit(self.allocator);
        for (self.sequences.items, 0..) |*seq, i| {
            seq.pages.deinit(self.allocator);
            self.freeCached(i);
        }
        self.sequences.deinit(self.allocator);
        self.cached_keys.deinit(self.allocator);
        self.cached_values.deinit(self.allocator);
        self.page_hashes.deinit();
        self.seq_prev_hashes.deinit(self.allocator);
        allocator.destroy(self);
    }
};

pub fn createPaged(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    stream: c.c.mlx_stream,
) !KVCacheStrategy {
    const cache = try allocator.create(PagedKVCache);
    errdefer allocator.destroy(cache);
    cache.* = try PagedKVCache.init(allocator, config, default_page_size, 16, 64, stream);
    return cache.asStrategy();
}

pub fn createPagedWithSize(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    page_size: usize,
    stream: c.c.mlx_stream,
) !KVCacheStrategy {
    const cache = try allocator.create(PagedKVCache);
    errdefer allocator.destroy(cache);
    cache.* = try PagedKVCache.init(allocator, config, page_size, 16, 64, stream);
    return cache.asStrategy();
}

/// Create a paged KV cache with quantized storage.
/// kv_bits: 4, 8, or 16 (16 = no quantization, same as createPaged).
pub fn createPagedQuantized(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    kv_bits: u8,
    group_size: i32,
    stream: c.c.mlx_stream,
) !KVCacheStrategy {
    const cache = try allocator.create(PagedKVCache);
    errdefer allocator.destroy(cache);
    cache.* = try PagedKVCache.init(allocator, config, default_page_size, kv_bits, group_size, stream);
    return cache.asStrategy();
}

/// Convenience factory for 4-bit paged+quantized KV cache (group_size=64).
pub fn createPagedQuantized4Bit(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    stream: c.c.mlx_stream,
) !KVCacheStrategy {
    return createPagedQuantized(allocator, config, 4, 64, stream);
}

// ---------------------------------------------------------------------------
// Quantization helpers (reuse patterns from quantized.zig)
// ---------------------------------------------------------------------------

fn initQuantizedPageBuffer(
    shape: []const i32,
    dim: i32,
    el_per_int: i32,
    group_size: i32,
    stream: c.c.mlx_stream,
) !QuantizedTuple {
    const packed_shape = &[_]i32{ shape[0], shape[1], shape[2], @divTrunc(dim, el_per_int) };
    const scale_shape = &[_]i32{ shape[0], shape[1], shape[2], @divTrunc(dim, group_size) };

    var packed_arr = c.c.mlx_array_new();
    var scales = c.c.mlx_array_new();
    var biases = c.c.mlx_array_new();

    try c.check(c.c.mlx_zeros(&packed_arr, packed_shape.ptr, 4, c.c.MLX_UINT32, stream));
    try c.check(c.c.mlx_zeros(&scales, scale_shape.ptr, 4, c.c.MLX_FLOAT16, stream));
    try c.check(c.c.mlx_zeros(&biases, scale_shape.ptr, 4, c.c.MLX_FLOAT16, stream));

    return .{
        .packed_data = Array.fromHandle(packed_arr),
        .scales = Array.fromHandle(scales),
        .biases = Array.fromHandle(biases),
    };
}

fn quantizeArray(
    arr: Array,
    group_size: i32,
    bits: i32,
    stream: c.c.mlx_stream,
) !QuantizedTuple {
    var vec = c.c.mlx_vector_array_new();
    defer _ = c.c.mlx_vector_array_free(vec);

    const opt_group: c.c.mlx_optional_int = .{ .value = group_size, .has_value = true };
    const opt_bits: c.c.mlx_optional_int = .{ .value = bits, .has_value = true };

    try c.check(c.c.mlx_quantize(
        &vec,
        arr.inner,
        opt_group,
        opt_bits,
        "affine",
        .{ .ctx = null },
        stream,
    ));

    var packed_arr = c.c.mlx_array_new();
    var scales_arr = c.c.mlx_array_new();
    var biases_arr = c.c.mlx_array_new();

    try c.check(c.c.mlx_vector_array_get(&packed_arr, vec, 0));
    try c.check(c.c.mlx_vector_array_get(&scales_arr, vec, 1));
    try c.check(c.c.mlx_vector_array_get(&biases_arr, vec, 2));

    return .{
        .packed_data = Array.fromHandle(packed_arr),
        .scales = Array.fromHandle(scales_arr),
        .biases = Array.fromHandle(biases_arr),
    };
}

fn writeQuantizedSlice(
    dst: *QuantizedTuple,
    src: QuantizedTuple,
    start: usize,
    end: usize,
    stream: c.c.mlx_stream,
) !void {
    const s = &[_]i32{ 0, 0, @intCast(start), 0 };
    const e = &[_]i32{ std.math.maxInt(i32), std.math.maxInt(i32), @intCast(end), std.math.maxInt(i32) };
    const st = &[_]i32{ 1, 1, 1, 1 };

    var new_packed = c.c.mlx_array_new();
    var new_scales = c.c.mlx_array_new();
    var new_biases = c.c.mlx_array_new();

    try c.check(c.c.mlx_slice_update(&new_packed, dst.packed_data.inner, src.packed_data.inner, s, 4, e, 4, st, 4, stream));
    try c.check(c.c.mlx_slice_update(&new_scales, dst.scales.inner, src.scales.inner, s, 4, e, 4, st, 4, stream));
    try c.check(c.c.mlx_slice_update(&new_biases, dst.biases.inner, src.biases.inner, s, 4, e, 4, st, 4, stream));

    dst.packed_data.deinit();
    dst.scales.deinit();
    dst.biases.deinit();
    dst.packed_data = Array.fromHandle(new_packed);
    dst.scales = Array.fromHandle(new_scales);
    dst.biases = Array.fromHandle(new_biases);
}

fn dequantizeTuple(tuple: QuantizedTuple, seq_len: usize, group_size: i32, bits: i32, stream: c.c.mlx_stream) !Array {
    const start = &[_]i32{ 0, 0, 0, 0 };
    const stop = &[_]i32{ std.math.maxInt(i32), std.math.maxInt(i32), @intCast(seq_len), std.math.maxInt(i32) };
    const strides = &[_]i32{ 1, 1, 1, 1 };

    var packed_sliced = c.c.mlx_array_new();
    var scales_sliced = c.c.mlx_array_new();
    var biases_sliced = c.c.mlx_array_new();

    try c.check(c.c.mlx_slice(&packed_sliced, tuple.packed_data.inner, start, 4, stop, 4, strides, 4, stream));
    try c.check(c.c.mlx_slice(&scales_sliced, tuple.scales.inner, start, 4, stop, 4, strides, 4, stream));
    try c.check(c.c.mlx_slice(&biases_sliced, tuple.biases.inner, start, 4, stop, 4, strides, 4, stream));

    const opt_group: c.c.mlx_optional_int = .{ .value = group_size, .has_value = true };
    const opt_bits: c.c.mlx_optional_int = .{ .value = bits, .has_value = true };
    const no_dtype: c.c.mlx_optional_dtype = .{ .value = c.c.MLX_FLOAT16, .has_value = false };

    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_dequantize(
        &res,
        packed_sliced,
        scales_sliced,
        biases_sliced,
        opt_group,
        opt_bits,
        "affine",
        .{ .ctx = null },
        no_dtype,
        stream,
    ));

    _ = c.c.mlx_array_free(packed_sliced);
    _ = c.c.mlx_array_free(scales_sliced);
    _ = c.c.mlx_array_free(biases_sliced);

    return Array.fromHandle(res);
}
