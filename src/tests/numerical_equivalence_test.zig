/// Property 8: Numerical equivalence with unoptimized path
///
/// **Validates: Requirements 5.3**
///
/// For any input tensor and expert selection indices, the optimized partial-read path
/// (PartialTensorReader.readExpertRows) SHALL produce output that is bitwise identical
/// to the unoptimized path (TensorIndex.loadTensor + takeAxis along axis 0).
///
/// This test creates temporary safetensors files with known fused expert tensors,
/// then compares both loading paths across 100 iterations with random expert selections.
const std = @import("std");
const c = @import("../c.zig");
const array_mod = @import("../array.zig");
const ops = @import("../ops.zig");
const safetensors_reader = @import("../io/safetensors_reader.zig");
const shape_mod = @import("../ops/shape.zig");

const Array = array_mod.Array;
const TensorIndex = safetensors_reader.TensorIndex;
const FdPool = safetensors_reader.FdPool;
const PartialTensorReader = safetensors_reader.PartialTensorReader;
const EagerContext = ops.EagerContext;

const posix_c = @cImport({
    @cInclude("fcntl.h");
    @cInclude("unistd.h");
});

/// Write a minimal safetensors file containing a single fused expert tensor.
///
/// Format: [8-byte LE u64 header_len] [JSON header] [tensor data...]
/// The tensor has shape [n_experts, dim1, dim2] with F32 dtype.
/// Each element is filled with a deterministic random pattern.
fn writeMockSafetensors(
    allocator: std.mem.Allocator,
    path: []const u8,
    tensor_name: []const u8,
    n_experts: u32,
    dim1: u32,
    dim2: u32,
    seed: u64,
) !void {
    const row_elements: usize = @as(usize, dim1) * @as(usize, dim2);
    const total_elements: usize = @as(usize, n_experts) * row_elements;
    const elem_bytes: usize = 4; // f32
    const total_data_bytes: usize = total_elements * elem_bytes;

    // Generate deterministic tensor data as f32 values
    const data = try allocator.alloc(f32, total_elements);
    defer allocator.free(data);

    var rng = std.Random.DefaultPrng.init(seed);
    const random = rng.random();
    for (0..total_elements) |i| {
        // Generate random f32 values in a reasonable range
        data[i] = random.float(f32) * 2.0 - 1.0;
    }

    // Build JSON header string
    const header_json = try std.fmt.allocPrint(
        allocator,
        "{{\"{s}\": {{\"dtype\": \"F32\", \"shape\": [{d}, {d}, {d}], \"data_offsets\": [0, {d}]}}}}",
        .{ tensor_name, n_experts, dim1, dim2, total_data_bytes },
    );
    defer allocator.free(header_json);

    const header_len: u64 = @intCast(header_json.len);

    // Write the file using C-level I/O (matching codebase pattern)
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    const fd = posix_c.open(path_z.ptr, posix_c.O_WRONLY | posix_c.O_CREAT | posix_c.O_TRUNC, @as(c_uint, 0o644));
    if (fd < 0) return error.FileCreateFailed;
    defer _ = posix_c.close(fd);

    // Write 8-byte LE header length
    var header_len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &header_len_buf, header_len, .little);
    _ = posix_c.write(fd, &header_len_buf, 8);

    // Write JSON header
    _ = posix_c.write(fd, header_json.ptr, header_json.len);

    // Write tensor data (raw f32 bytes)
    const data_bytes: [*]const u8 = @ptrCast(data.ptr);
    _ = posix_c.write(fd, data_bytes, total_data_bytes);
}

/// Remove a temporary file, ignoring errors.
fn removeTempFile(path: []const u8, allocator: std.mem.Allocator) void {
    const path_z = allocator.dupeZ(u8, path) catch return;
    defer allocator.free(path_z);
    _ = posix_c.unlink(path_z.ptr);
}

test "Feature: stream-mode-performance, Property 8: Numerical equivalence with unoptimized path" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);

    // Use a fixed base seed for reproducibility, varied per iteration
    var iteration_rng = std.Random.DefaultPrng.init(0xDEAD_BEEF_CAFE_1234);
    const iter_random = iteration_rng.random();

    const tmp_path = "/tmp/mlx_zig_numerical_equiv_test.safetensors";

    // Run 100 iterations with random configurations
    for (0..100) |iteration| {
        // Generate random tensor dimensions for this iteration
        // n_experts: 4..16, dim1: 2..8, dim2: 2..8
        const n_experts: u32 = iter_random.intRangeAtMost(u32, 4, 16);
        const dim1: u32 = iter_random.intRangeAtMost(u32, 2, 8);
        const dim2: u32 = iter_random.intRangeAtMost(u32, 2, 8);
        const data_seed: u64 = iter_random.int(u64);

        const tensor_name = "fused_experts.weight";

        // 1. Write mock safetensors file
        try writeMockSafetensors(allocator, tmp_path, tensor_name, n_experts, dim1, dim2, data_seed);
        defer removeTempFile(tmp_path, allocator);

        // 2. Build TensorIndex from the file
        var index = TensorIndex.init(allocator);
        defer index.deinit();
        try index.addShard(tmp_path);

        // 3. Generate random expert selection (1..n_experts unique expert IDs)
        const n_selected: u32 = iter_random.intRangeAtMost(u32, 1, n_experts);
        var selected_set = std.AutoHashMap(u32, void).init(allocator);
        defer selected_set.deinit();

        while (selected_set.count() < n_selected) {
            const eid = iter_random.intRangeLessThan(u32, 0, n_experts);
            try selected_set.put(eid, {});
        }

        var expert_ids = try allocator.alloc(u32, selected_set.count());
        defer allocator.free(expert_ids);
        {
            var it = selected_set.keyIterator();
            var i: usize = 0;
            while (it.next()) |k| {
                expert_ids[i] = k.*;
                i += 1;
            }
        }
        std.mem.sort(u32, expert_ids, {}, std.sort.asc(u32));

        // === OLD PATH: loadTensor + takeAxis ===
        const full_tensor = try index.loadTensor(tensor_name);
        defer full_tensor.deinit();
        try full_tensor.eval();

        // Create indices array for takeAxis
        const indices_arr = try Array.fromData(allocator, u32, expert_ids, &[_]i32{@intCast(expert_ids.len)});
        defer indices_arr.deinit();
        const indices_i32 = try ops.astype(ctx, indices_arr, .int32);
        defer indices_i32.deinit();

        const old_result = try shape_mod.takeAxis(ctx, full_tensor, indices_i32, 0);
        defer old_result.deinit();
        try old_result.eval();

        // === NEW PATH: PartialTensorReader.readExpertRows ===
        var fd_pool = FdPool.init(allocator);
        defer fd_pool.deinit();
        try fd_pool.openAll(&index);

        var reader = PartialTensorReader.init(allocator, &index, &fd_pool);
        const new_result = try reader.readExpertRows(tensor_name, expert_ids);
        defer new_result.deinit();
        try new_result.eval();

        // === COMPARE: bitwise identical ===
        // Verify shapes match
        const old_shape = old_result.shape();
        const new_shape = new_result.shape();
        try std.testing.expectEqual(old_shape.len, new_shape.len);
        for (old_shape, new_shape) |os, ns| {
            try std.testing.expectEqual(os, ns);
        }

        // Verify sizes match
        try std.testing.expectEqual(old_result.size(), new_result.size());

        // Verify data is bitwise identical using f32 data access
        const old_data = try old_result.dataSlice(f32);
        const new_data = try new_result.dataSlice(f32);
        try std.testing.expectEqual(old_data.len, new_data.len);

        for (old_data, new_data, 0..) |old_val, new_val, elem_idx| {
            // Bitwise comparison via reinterpret as u32
            const old_bits: u32 = @bitCast(old_val);
            const new_bits: u32 = @bitCast(new_val);
            if (old_bits != new_bits) {
                std.debug.print(
                    "\nBitwise mismatch at iteration {d}, element {d}: old={e} (0x{x:0>8}) new={e} (0x{x:0>8})\n" ++
                        "  n_experts={d}, dim1={d}, dim2={d}, n_selected={d}\n" ++
                        "  expert_ids={any}\n",
                    .{
                        iteration,    elem_idx,
                        old_val,      old_bits,
                        new_val,      new_bits,
                        n_experts,    dim1,
                        dim2,         expert_ids.len,
                        expert_ids,
                    },
                );
                try std.testing.expect(false);
            }
        }
    }
}
