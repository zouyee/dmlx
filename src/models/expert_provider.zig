/// Expert provider interface for MoE layer inference.
///
/// Decouples the MoE layer (DSV4MoE) from concrete expert loading strategies,
/// enabling pluggable implementations: preload, stream, hybrid, etc.
///
/// Design: vtable pattern with opaque pointer, matching Zig std library conventions.
const std = @import("std");
const array_mod = @import("mlx").array;

const Array = array_mod.Array;

pub const ExpertProvider = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Forward pass for a single MoE layer.
        /// `flat_x`: [N, D] flattened input tokens.
        /// `indices`: [N, topk] selected expert IDs.
        /// `scores`: [N, topk] routing scores.
        forward: *const fn (ptr: *anyopaque, layer_idx: usize, flat_x: Array, indices: Array, scores: Array) anyerror!Array,

        /// Optional cache bias for router (preload strategy only).
        /// Returns null if not supported.
        getCacheBias: ?*const fn (ptr: *anyopaque, layer_idx: usize) ?Array,

        /// Clean up provider-owned resources.
        deinit: *const fn (ptr: *anyopaque) void,
    };

    /// Execute forward pass for the given layer.
    pub fn forward(self: ExpertProvider, layer_idx: usize, flat_x: Array, indices: Array, scores: Array) !Array {
        return self.vtable.forward(self.ptr, layer_idx, flat_x, indices, scores);
    }

    /// Get cache bias for router, if available.
    pub fn getCacheBias(self: ExpertProvider, layer_idx: usize) ?Array {
        if (self.vtable.getCacheBias) |f| {
            return f(self.ptr, layer_idx);
        }
        return null;
    }

    /// Release provider resources.
    pub fn deinit(self: ExpertProvider) void {
        self.vtable.deinit(self.ptr);
    }
};
