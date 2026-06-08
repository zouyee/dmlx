/// NativeEngineLoader — top-level entry point for MLX-free weight loading.
pub const std = @import("std");
pub const safetensors = @import("safetensors.zig");
pub const config_mod = @import("config.zig");
pub const weights_mod = @import("weights.zig");

pub const TensorIndex = safetensors.TensorIndex;
pub const DSV4NativeConfig = config_mod.DSV4NativeConfig;
pub const NativeWeightStore = weights_mod.NativeWeightStore;
pub const EngineWeights = weights_mod.EngineWeights;

pub const NativeEngineLoader = struct {
    allocator: std.mem.Allocator,
    model_path: []const u8,

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !NativeEngineLoader {
        return .{
            .allocator = allocator,
            .model_path = try allocator.dupe(u8, model_path),
        };
    }

    pub fn deinit(self: *NativeEngineLoader) void {
        self.allocator.free(self.model_path);
    }

    pub fn loadConfig(self: *const NativeEngineLoader) !DSV4NativeConfig {
        return config_mod.parse(self.allocator, self.model_path);
    }

    /// Load all weights. Returns a NativeWeightStore that owns all memory.
    /// Caller is responsible for calling store.deinit() when done.
    /// Logs progress to stderr; times out after timeout_ms (0 = no timeout).
    pub fn loadWeights(
        self: *const NativeEngineLoader,
        cfg: DSV4NativeConfig,
        timeout_ms: u64,
    ) !NativeWeightStore {
        var ts0: std.c.timespec = undefined;
        _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts0);
        const t0_ms: u64 = @intCast(ts0.sec * 1000 + @divTrunc(ts0.nsec, 1_000_000));
        std.log.info("native_loader: building safetensors index from {s}", .{self.model_path});
        var idx = try safetensors.buildIndex(self.allocator, self.model_path);
        defer idx.deinit();
        std.log.info("native_loader: index ready, {d} tensors, starting weight load...", .{idx.entries.count()});

        const progress = struct {
            fn cb(layer: usize, total: usize) void {
                if (layer % 5 == 0 or layer == total - 1) {
                    std.log.info("native_loader: layer {d}/{d}", .{ layer + 1, total });
                }
            }
        }.cb;

        const store = try weights_mod.loadAll(self.allocator, &idx, cfg, &progress, timeout_ms);
        var ts1: std.c.timespec = undefined;
        _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts1);
        const t1_ms: u64 = @intCast(ts1.sec * 1000 + @divTrunc(ts1.nsec, 1_000_000));
        std.log.info("native_loader: loaded in {d}ms", .{t1_ms - t0_ms});
        return store;
    }
};
