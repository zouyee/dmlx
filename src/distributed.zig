/// Multi-Mac distributed inference via mlx-c distributed API.
///
/// Provides Zig wrappers for the MLX distributed communication primitives,
/// enabling tensor parallelism across multiple Apple Silicon Macs connected
/// via network (typically Thunderbolt or Ethernet).
///
/// ## Architecture
///
/// MLX distributed uses MPI-style collective operations:
///   - `allSum` / `allGather` / `allMax` / `allMin` — collective reductions
///   - `send` / `recv` — point-to-point communication
///   - `sumScatter` — reduce-scatter for gradient aggregation
///
/// ## Usage
///
/// ```zig
/// // Initialize distributed runtime
/// var group = try DistributedGroup.init(true);
/// defer group.deinit();
///
/// if (group.isInitialized()) {
///     const rank = try group.rank();
///     const size = try group.size();
///     std.log.info("Rank {d}/{d}", .{rank, size});
///
///     // All-reduce a tensor across all ranks
///     const reduced = try allSum(my_tensor, group, stream);
/// }
/// ```
///
/// ## Requirements
///
/// - mlx-c with distributed support compiled in (requires MPI backend)
/// - Multiple Macs on the same network
/// - Launch via `mpirun -np <N> mlx-zig serve --model <path>`
///
/// ## Tensor Parallelism Strategy
///
/// For LLM inference, the recommended approach is:
///   1. Shard attention heads across ranks (each rank handles num_heads/N heads)
///   2. All-reduce after attention output projection
///   3. Shard MLP intermediate dimension across ranks
///   4. All-reduce after MLP down projection
///   5. Each rank holds a full copy of embeddings and layer norms
///
/// This minimizes communication while distributing the compute-heavy operations.
const std = @import("std");
const c = @import("c.zig");
const array_mod = @import("array.zig");
const dtype_mod = @import("dtype.zig");

const Array = array_mod.Array;

/// A distributed communication group.
/// Wraps `mlx_distributed_group` from mlx-c.
pub const DistributedGroup = struct {
    inner: c.c.mlx_distributed_group,
    initialized: bool,

    /// Initialize the distributed runtime and create a group.
    /// If `strict` is true, returns an error if initialization fails.
    /// If `strict` is false, creates an uninitialized group (single-rank mode).
    pub fn init(strict: bool) !DistributedGroup {
        const group = c.c.mlx_distributed_init(strict, null);
        const is_initialized = group.ctx != null;
        if (!is_initialized and strict) {
            return error.DistributedInitFailed;
        }
        return .{
            .inner = group,
            .initialized = is_initialized,
        };
    }

    /// Create a non-initialized group (for single-rank operation).
    pub fn initEmpty() DistributedGroup {
        return .{
            .inner = .{ .ctx = null },
            .initialized = false,
        };
    }

    pub fn deinit(self: *DistributedGroup) void {
        _ = self;
        // mlx_distributed_group has no explicit free in this mlx-c version
    }

    /// Check if the distributed runtime was successfully initialized.
    pub fn isInitialized(self: *const DistributedGroup) bool {
        return self.initialized;
    }

    /// Get the rank (process ID) of this node in the group.
    pub fn rank(self: *const DistributedGroup) !i32 {
        var r: i32 = 0;
        try c.check(c.c.mlx_distributed_group_rank(&r, self.inner));
        return r;
    }

    /// Get the total number of ranks in the group.
    pub fn size(self: *const DistributedGroup) !i32 {
        var s: i32 = 0;
        try c.check(c.c.mlx_distributed_group_size(&s, self.inner));
        return s;
    }
};

// ------------------------------------------------------------------
// Collective operations
// ------------------------------------------------------------------

/// All-reduce sum: compute element-wise sum across all ranks.
/// Each rank receives the same result.
pub fn allSum(x: Array, group: DistributedGroup, stream: c.c.mlx_stream) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_distributed_all_sum(&res, x.inner, group.inner, stream));
    return Array.fromHandle(res);
}

/// All-gather: concatenate tensors from all ranks.
/// Each rank receives the full concatenated result.
pub fn allGather(x: Array, group: DistributedGroup, stream: c.c.mlx_stream) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_distributed_all_gather(&res, x.inner, group.inner, stream));
    return Array.fromHandle(res);
}

/// All-reduce max: compute element-wise maximum across all ranks.
pub fn allMax(x: Array, group: DistributedGroup, stream: c.c.mlx_stream) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_distributed_all_max(&res, x.inner, group.inner, stream));
    return Array.fromHandle(res);
}

/// All-reduce min: compute element-wise minimum across all ranks.
pub fn allMin(x: Array, group: DistributedGroup, stream: c.c.mlx_stream) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_distributed_all_min(&res, x.inner, group.inner, stream));
    return Array.fromHandle(res);
}

/// Sum-scatter: reduce-scatter operation.
/// Each rank receives a different shard of the reduced result.
pub fn sumScatter(x: Array, group: DistributedGroup, stream: c.c.mlx_stream) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_distributed_sum_scatter(&res, x.inner, group.inner, stream));
    return Array.fromHandle(res);
}

// ------------------------------------------------------------------
// Point-to-point operations
// ------------------------------------------------------------------

/// Send a tensor to a specific destination rank.
pub fn send(x: Array, dst: i32, group: DistributedGroup, stream: c.c.mlx_stream) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_distributed_send(&res, x.inner, dst, group.inner, stream));
    return Array.fromHandle(res);
}

/// Receive a tensor from a specific source rank.
/// The received tensor will have the same shape and dtype as `like`.
pub fn recvLike(like: Array, src: i32, group: DistributedGroup, stream: c.c.mlx_stream) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_distributed_recv_like(&res, like.inner, src, group.inner, stream));
    return Array.fromHandle(res);
}

/// Receive a tensor with explicit shape and dtype from a source rank.
pub fn recv(
    shape: []const i32,
    dtype: c.c.mlx_dtype,
    src: i32,
    group: DistributedGroup,
    stream: c.c.mlx_stream,
) !Array {
    var res = c.c.mlx_array_new();
    try c.check(c.c.mlx_distributed_recv(
        &res,
        shape.ptr,
        @intCast(shape.len),
        dtype,
        src,
        group.inner,
        stream,
    ));
    return Array.fromHandle(res);
}

// ------------------------------------------------------------------
// Synchronization
// ------------------------------------------------------------------

/// Barrier: synchronize all ranks in the group.
/// Uses a scalar all-reduce as a synchronization primitive.
pub fn barrier(group: DistributedGroup, stream: c.c.mlx_stream) !void {
    const scalar_data: f32 = 1.0;
    const scalar_arr = c.c.mlx_array_new_data(&scalar_data, &[_]i32{1}, 1, c.c.MLX_FLOAT32);
    defer _ = c.c.mlx_array_free(scalar_arr);
    var res = c.c.mlx_array_new();
    defer _ = c.c.mlx_array_free(res);
    try c.check(c.c.mlx_distributed_all_sum(&res, scalar_arr, group.inner, stream));
}

// ------------------------------------------------------------------
// Unit Tests
// ------------------------------------------------------------------

test "DistributedGroup: initEmpty creates non-initialized group" {
    var group = DistributedGroup.initEmpty();
    defer group.deinit();
    try std.testing.expect(!group.isInitialized());
}

test "DistributedGroup: barrier with empty group is safe" {
    var group = DistributedGroup.initEmpty();
    defer group.deinit();
    const stream = c.c.mlx_default_cpu_stream_new();
    defer _ = c.c.mlx_stream_free(stream);
    // Barrier on empty group should be a no-op (scalar all-reduce with null group)
    try barrier(group, stream);
}
