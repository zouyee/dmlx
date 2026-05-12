/// RadixKVCache — Prefix-tree KV cache for multi-turn conversation acceleration.
///
/// Strategy: maintain a radix tree (prefix tree) where each node stores a token sequence
/// and its corresponding KV offset in an underlying StandardKVCache.
///
/// On a new prompt:
///   1. Walk the radix tree to find the longest matching prefix.
///   2. If match length > 0: reuse cached KV (skip prefill for matched prefix).
///   3. Append unmatched suffix to tree and cache new KV.
///
/// This is inspired by SGLang's RadixAttention, adapted for Zig's zero-overhead
/// data structures and Apple Silicon unified memory.
///
/// Pros: massive speedup for multi-turn chat (2nd+ turn may skip 90%+ of prefill).
/// Cons: tree management overhead, memory grows with unique prompt history.
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const iface = @import("interface.zig");

const Array = array_mod.Array;
const KVSlice = iface.KVSlice;
const KVCacheStrategy = iface.KVCacheStrategy;
const LayerConfig = iface.LayerConfig;

/// A node in the radix tree.
const Node = struct {
    /// Token sequence represented by this node.
    tokens: []const u32,

    /// Offset into the underlying KV cache buffer.
    kv_offset: usize,

    /// Length of the token sequence.
    len: usize,

    /// Child nodes keyed by the first token of the child's sequence.
    children: std.HashMap(u32, *Node, std.hash_map.defaultContext(u32), std.hash_map.default_max_load_percentage),

    /// Parent node (for eviction).
    parent: ?*Node,

    /// Reference count (how many active sessions share this node).
    ref_count: usize,

    pub fn init(allocator: std.mem.Allocator, tokens: []const u32, kv_offset: usize, parent: ?*Node) !*Node {
        const node = try allocator.create(Node);
        node.tokens = try allocator.dupe(u32, tokens);
        node.kv_offset = kv_offset;
        node.len = tokens.len;
        node.children = std.HashMap(u32, *Node, std.hash_map.defaultContext(u32), std.hash_map.default_max_load_percentage).init(allocator);
        node.parent = parent;
        node.ref_count = 1;
        return node;
    }

    pub fn deinit(self: *Node, allocator: std.mem.Allocator) void {
        var it = self.children.valueIterator();
        while (it.next()) |child| {
            child.*.deinit(allocator);
        }
        self.children.deinit();
        allocator.free(self.tokens);
        allocator.destroy(self);
    }
};

/// Match result from radix tree lookup.
pub const MatchResult = struct {
    /// Number of tokens matched (may span multiple nodes).
    match_len: usize,

    /// Node where the match ends.
    node: *Node,

    /// Whether the match is exact (all tokens in prompt matched).
    exact: bool,

    /// Number of tokens matched within the final node (0 if match ended before this node).
    node_match_len: usize,
};

/// RadixKVCache combines a radix tree with an underlying flat KV buffer.
pub const RadixKVCache = struct {
    allocator: std.mem.Allocator,

    // Underlying flat cache (similar to StandardKVCache but append-only).
    keys: Array,
    values: Array,
    offset: usize,

    // Radix tree root.
    root: *Node,

    // Shape parameters.
    batch_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,

    pub const vtable: iface.VTable = .{
        .updateAndFetch = updateAndFetchImpl,
        .currentLen = currentLenImpl,
        .reset = resetImpl,
        .filter = null, // Radix cache doesn't support arbitrary batch filtering.
        .extend = null,
        .deinit = deinitImpl,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        config: LayerConfig,
        stream: c.c.mlx_stream,
    ) !RadixKVCache {
        const shape = &[_]i32{
            @intCast(config.batch_size),
            @intCast(config.num_kv_heads),
            @intCast(config.max_seq_len),
            @intCast(config.head_dim),
        };

        var keys_arr = c.c.mlx_array_new();
        var values_arr = c.c.mlx_array_new();

        try c.check(c.c.mlx_zeros(&keys_arr, shape.ptr, shape.len, @intCast(@intFromEnum(config.dtype)), stream));
        try c.check(c.c.mlx_zeros(&values_arr, shape.ptr, shape.len, @intCast(@intFromEnum(config.dtype)), stream));

        const root = try Node.init(allocator, &.{}, 0, null);

        return .{
            .allocator = allocator,
            .keys = Array.fromHandle(keys_arr),
            .values = Array.fromHandle(values_arr),
            .offset = 0,
            .root = root,
            .batch_size = config.batch_size,
            .num_kv_heads = config.num_kv_heads,
            .head_dim = config.head_dim,
            .max_seq_len = config.max_seq_len,
        };
    }

    pub fn asStrategy(self: *RadixKVCache) KVCacheStrategy {
        return .{ .ptr = self, .vtable = &vtable };
    }

    /// Match a token sequence against the radix tree.
    /// Returns the match length and the node where matching stopped.
    pub fn match(self: *RadixKVCache, tokens: []const u32) MatchResult {
        var node = self.root;
        var matched: usize = 0;

        while (matched < tokens.len) {
            const next_token = tokens[matched];
            const child = node.children.get(next_token) orelse break;

            // Compare token sequence with child's tokens.
            const max_cmp = @min(child.len, tokens.len - matched);
            var i: usize = 0;
            while (i < max_cmp) : (i += 1) {
                if (child.tokens[i] != tokens[matched + i]) break;
            }

            matched += i;

            if (i < child.len) {
                // Partial match within child node.
                return .{ .match_len = matched, .node = child, .exact = false, .node_match_len = i };
            }

            // Full child match, descend.
            node = child;
            if (matched == tokens.len) {
                return .{ .match_len = matched, .node = child, .exact = true, .node_match_len = child.len };
            }
        }

        return .{ .match_len = matched, .node = node, .exact = matched == tokens.len, .node_match_len = if (node == self.root) 0 else node.len };
    }

    /// Insert a new token sequence into the tree, starting from a match result.
    /// The underlying KV cache must already contain the KV for these tokens.
    pub fn insert(self: *RadixKVCache, tokens: []const u32, match_result: MatchResult) !void {
        if (match_result.exact) {
            // Exact match — just increment ref count.
            match_result.node.ref_count += 1;
            return;
        }

        var node = match_result.node;
        const remaining = tokens[match_result.match_len..];

        if (remaining.len == 0) return;

        // If we stopped mid-node, split the node.
        if (match_result.node_match_len > 0 and match_result.node_match_len < node.len) {
            const split_at = match_result.node_match_len;

            // Create suffix node with the remaining tokens of the original node.
            const suffix_tokens = node.tokens[split_at..];
            const suffix = try Node.init(self.allocator, suffix_tokens, node.kv_offset + split_at, node);
            // Move original children to suffix node.
            var it = node.children.iterator();
            while (it.next()) |entry| {
                try suffix.children.put(entry.key_ptr.*, entry.value_ptr.*);
                entry.value_ptr.*.parent = suffix;
            }
            node.children.clearRetainingCapacity();

            // Truncate original node to prefix.
            const new_prefix = try self.allocator.dupe(u32, node.tokens[0..split_at]);
            self.allocator.free(node.tokens);
            node.tokens = new_prefix;
            node.len = split_at;

            // Add suffix as child of prefix.
            try node.children.put(suffix_tokens[0], suffix);

            // Add the new token sequence as a sibling of suffix.
            const new_child = try Node.init(self.allocator, remaining, self.offset, node);
            try node.children.put(remaining[0], new_child);
        } else {
            // Create new child from current node.
            const child = try Node.init(self.allocator, remaining, self.offset, node);
            try node.children.put(remaining[0], child);
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
        // RadixKVCache.updateAndFetch works like StandardKVCache for the KV part.
        // The radix tree matching happens at the model level before calling this.
        const self: *RadixKVCache = @ptrCast(@alignCast(ctx));
        const seq_len = @as(usize, @intCast(keys.shape()[2]));
        const new_offset = self.offset + seq_len;

        if (new_offset > self.max_seq_len) {
            return error.CacheOverflow;
        }

        try sliceUpdateKV(&self.keys, keys, self.offset, new_offset, stream);
        try sliceUpdateKV(&self.values, values, self.offset, new_offset, stream);
        self.offset = new_offset;

        const k = try sliceFetch(self.keys, self.offset, stream);
        const v = try sliceFetch(self.values, self.offset, stream);
        return .{ .keys = k, .values = v };
    }

    fn currentLenImpl(ctx: *anyopaque) usize {
        const self: *RadixKVCache = @ptrCast(@alignCast(ctx));
        return self.offset;
    }

    fn resetImpl(ctx: *anyopaque) void {
        const self: *RadixKVCache = @ptrCast(@alignCast(ctx));
        self.offset = 0;
        self.root.deinit(self.allocator);
        self.root = Node.init(self.allocator, &.{}, 0, null) catch unreachable;
    }

    fn deinitImpl(ctx: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *RadixKVCache = @ptrCast(@alignCast(ctx));
        self.keys.deinit();
        self.values.deinit();
        self.root.deinit(self.allocator);
        allocator.destroy(self);
    }

    // ------------------------------------------------------------------
    // Helpers (duplicated from standard.zig — could be shared)
    // ------------------------------------------------------------------

    fn sliceUpdateKV(
        buffer: *Array,
        new_kv: Array,
        offset: usize,
        end_offset: usize,
        stream: c.c.mlx_stream,
    ) !void {
        const start = &[_]i32{ 0, 0, @intCast(offset), 0 };
        const stop = &[_]i32{ std.math.maxInt(i32), std.math.maxInt(i32), @intCast(end_offset), std.math.maxInt(i32) };
        const strides = &[_]i32{ 1, 1, 1, 1 };

        var res = c.c.mlx_array_new();
        try c.check(c.c.mlx_slice_update(
            &res,
            buffer.inner,
            new_kv.inner,
            start.ptr,
            start.len,
            stop.ptr,
            stop.len,
            strides.ptr,
            strides.len,
            stream,
        ));
        buffer.deinit();
        buffer.* = Array.fromHandle(res);
    }

    fn sliceFetch(buffer: Array, offset: usize, stream: c.c.mlx_stream) !Array {
        const start = &[_]i32{ 0, 0, 0, 0 };
        const stop = &[_]i32{ std.math.maxInt(i32), std.math.maxInt(i32), @intCast(offset), std.math.maxInt(i32) };
        const strides = &[_]i32{ 1, 1, 1, 1 };

        var res = c.c.mlx_array_new();
        try c.check(c.c.mlx_slice(
            &res,
            buffer.inner,
            start.ptr,
            start.len,
            stop.ptr,
            stop.len,
            strides.ptr,
            strides.len,
            stream,
        ));
        return Array.fromHandle(res);
    }
};

/// Factory for RadixKVCache.
pub fn createRadix(
    allocator: std.mem.Allocator,
    config: LayerConfig,
    stream: c.c.mlx_stream,
) !KVCacheStrategy {
    const cache = try allocator.create(RadixKVCache);
    errdefer allocator.destroy(cache);
    cache.* = try RadixKVCache.init(allocator, config, stream);
    return cache.asStrategy();
}
