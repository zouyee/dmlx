/// Tokenizer Strategy Interface — Runtime-polymorphic text encoding/decoding.
///
/// Design goals:
///   1. Switch tokenizer backend (greedy, BPE, SentencePiece) without recompiling.
///   2. Unified lifecycle: caller creates strategy, uses it, then deinits.
///   3. Zero overhead: VTable is a comptime constant per strategy.
const std = @import("std");

pub const VTable = struct {
    /// Encode text to token IDs.
    encode: *const fn (
        ptr: *anyopaque,
        text: []const u8,
        add_special_tokens: bool,
        allocator: std.mem.Allocator,
    ) anyerror![]u32,

    /// Decode token IDs back to text.
    decode: *const fn (
        ptr: *anyopaque,
        ids: []const u32,
        allocator: std.mem.Allocator,
    ) anyerror![]const u8,

    /// Return vocabulary size.
    vocabSize: *const fn (ptr: *anyopaque) usize,

    /// Release all resources held by this strategy.
    deinit: *const fn (ptr: *anyopaque, allocator: std.mem.Allocator) void,
};

/// Runtime-polymorphic handle to a tokenizer strategy.
pub const TokenizerStrategy = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub fn encode(
        self: TokenizerStrategy,
        text: []const u8,
        add_special_tokens: bool,
        allocator: std.mem.Allocator,
    ) ![]u32 {
        return self.vtable.encode(self.ptr, text, add_special_tokens, allocator);
    }

    pub fn decode(
        self: TokenizerStrategy,
        ids: []const u32,
        allocator: std.mem.Allocator,
    ) ![]const u8 {
        return self.vtable.decode(self.ptr, ids, allocator);
    }

    pub fn vocabSize(self: TokenizerStrategy) usize {
        return self.vtable.vocabSize(self.ptr);
    }

    pub fn deinit(self: TokenizerStrategy, allocator: std.mem.Allocator) void {
        return self.vtable.deinit(self.ptr, allocator);
    }
};
