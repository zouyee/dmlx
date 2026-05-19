const std = @import("std");
const mlx = @import("root.zig");
const c = mlx.c;

// ============================================================
// Foundation
// ============================================================

test "init mlx error handler" {
    c.initErrorHandler();
}

// ============================================================
// Core Tests (from dmlx core_tests.zig)
// ============================================================

test "array creation and properties" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const data = [_]f32{ 1, 2, 3, 4 };
    const arr = try mlx.Array.fromData(alloc, f32, &data, &[_]i32{ 2, 2 });
    defer arr.deinit();

    try std.testing.expectEqual(arr.ndim(), 2);
    try std.testing.expectEqual(arr.size(), 4);
    try std.testing.expectEqual(arr.shape()[0], 2);
    try std.testing.expectEqual(arr.shape()[1], 2);
    try std.testing.expectEqual(arr.dtype(), .float32);
}

test "matmul" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const b_data = [_]f32{ 5, 6, 7, 8 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 2 });
    defer a.deinit();
    const b = try mlx.Array.fromData(alloc, f32, &b_data, &[_]i32{ 2, 2 });
    defer b.deinit();

    const ctx = mlx.EagerContext.init(alloc);
    const c_mat = try mlx.ops.matmul(ctx, a, b);
    defer c_mat.deinit();

    const result = try c_mat.dataSlice(f32);
    try std.testing.expectEqual(result[0], 19.0);
    try std.testing.expectEqual(result[1], 22.0);
    try std.testing.expectEqual(result[2], 43.0);
    try std.testing.expectEqual(result[3], 50.0);
}

test "element-wise ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 2 });
    defer a.deinit();
    const b = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 2 });
    defer b.deinit();

    const ctx = mlx.EagerContext.init(alloc);
    const sum_arr = try mlx.ops.add(ctx, a, b);
    defer sum_arr.deinit();
    const sum_data = try sum_arr.dataSlice(f32);
    try std.testing.expectEqual(sum_data[0], 2.0);
    try std.testing.expectEqual(sum_data[1], 4.0);
}

test "softmax" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const logits = try mlx.Array.fromSlice(alloc, f32, &[_]f32{ 2.0, 1.0, 0.1 });
    defer logits.deinit();

    const ctx = mlx.EagerContext.init(alloc);
    const probs = try mlx.ops.softmax(ctx, logits, &[_]i32{});
    defer probs.deinit();

    const p = try probs.dataSlice(f32);
    try std.testing.expectApproxEqAbs(p[0], 0.659, 0.01);
    try std.testing.expectApproxEqAbs(p[1], 0.242, 0.01);
    try std.testing.expectApproxEqAbs(p[2], 0.098, 0.01);
}

test "zeros and ones" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const z = try mlx.Array.zeros(alloc, &[_]i32{ 2, 3 }, .float32);
    defer z.deinit();
    try std.testing.expectEqual(z.size(), 6);
    const z_data = try z.dataSlice(f32);
    for (z_data) |v| try std.testing.expectEqual(v, 0.0);

    const o = try mlx.Array.ones(alloc, &[_]i32{ 3, 3 }, .float32);
    defer o.deinit();
    const o_data = try o.dataSlice(f32);
    for (o_data) |v| try std.testing.expectEqual(v, 1.0);
}

// ============================================================
// Memory Management
// ============================================================

test "memory: get cached and active memory" {
    c.initErrorHandler();
    const active = try mlx.memory.getActiveMemory();
    const cached = try mlx.memory.getCacheMemory();
    _ = active;
    _ = cached;
}

test "memory: get and set memory limit" {
    c.initErrorHandler();
    const prev = try mlx.memory.setMemoryLimit(8 * 1024 * 1024 * 1024);
    _ = prev;
}

test "memory: get and set cache limit" {
    c.initErrorHandler();
    const prev = try mlx.memory.setCacheLimit(512 * 1024 * 1024);
    _ = prev;
}

test "memory: peak memory and reset" {
    c.initErrorHandler();
    const peak = try mlx.memory.getPeakMemory();
    _ = peak;
    try mlx.memory.resetPeakMemory();
}

test "memory: clear cache" {
    c.initErrorHandler();
    try mlx.memory.clearCache();
}

// ============================================================
// Math Ops (from dmlx math_tests.zig)
// ============================================================

test "math ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 2 });
    defer a.deinit();
    const b = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 2 });
    defer b.deinit();

    const ctx = mlx.EagerContext.init(alloc);

    const s = try mlx.math.sign(ctx, a);
    defer s.deinit();
    try std.testing.expectEqual(s.ndim(), 2);

    const f = try mlx.math.floor(ctx, a);
    defer f.deinit();
    try std.testing.expectEqual(f.ndim(), 2);

    const r = try mlx.math.round(ctx, a, 0);
    defer r.deinit();
    try std.testing.expectEqual(r.ndim(), 2);

    const mx = try mlx.math.maximum(ctx, a, b);
    defer mx.deinit();
    try std.testing.expectEqual(mx.ndim(), 2);

    const p = try mlx.math.power(ctx, a, b);
    defer p.deinit();
    try std.testing.expectEqual(p.ndim(), 2);
}

test "math: stop gradient" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a = try mlx.Array.fromSlice(alloc, f32, &[_]f32{ 1, 2, 3 });
    defer a.deinit();

    const ctx = mlx.EagerContext.init(alloc);
    const s = try mlx.math.stopGradient(ctx, a);
    defer s.deinit();

    try std.testing.expectEqual(s.ndim(), 1);
    try std.testing.expectEqual(s.size(), 3);
}

// ============================================================
// Shape Ops (from dmlx shape_tests.zig)
// ============================================================

test "shape ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 3 });
    defer a.deinit();

    const ctx = mlx.EagerContext.init(alloc);

    const r = try mlx.shape.reshape(ctx, a, &[_]i32{ 3, 2 });
    defer r.deinit();
    try std.testing.expectEqual(r.shape()[0], 3);
    try std.testing.expectEqual(r.shape()[1], 2);

    const t = try mlx.shape.transpose(ctx, a);
    defer t.deinit();
    try std.testing.expectEqual(t.shape()[0], 3);
    try std.testing.expectEqual(t.shape()[1], 2);

    const s = try mlx.shape.squeeze(ctx, a);
    defer s.deinit();
    try std.testing.expectEqual(s.ndim(), 2);

    const idx = try mlx.Array.fromData(alloc, i32, &[_]i32{0}, &[_]i32{1});
    defer idx.deinit();
    const tk = try mlx.shape.take(ctx, a, idx);
    defer tk.deinit();
}

test "shape: stack" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const ctx = mlx.EagerContext.init(alloc);

    const a = try mlx.Array.fromSlice(alloc, f32, &[_]f32{ 1, 2, 3 });
    defer a.deinit();
    const b = try mlx.Array.fromSlice(alloc, f32, &[_]f32{ 4, 5, 6 });
    defer b.deinit();

    const stacked = try mlx.shape.stack(ctx, &[_]mlx.Array{ a, b });
    defer stacked.deinit();

    try std.testing.expectEqual(stacked.ndim(), 2);
    try std.testing.expectEqual(stacked.shape()[0], 2);
    try std.testing.expectEqual(stacked.shape()[1], 3);
}

// ============================================================
// Comparison Ops (from dmlx comparison_tests.zig)
// ============================================================

test "comparison ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 2 });
    defer a.deinit();
    const b = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 2 });
    defer b.deinit();

    const ctx = mlx.EagerContext.init(alloc);

    const eq = try mlx.comparison.equal(ctx, a, b);
    defer eq.deinit();
    try std.testing.expectEqual(eq.ndim(), 2);

    const gt = try mlx.comparison.greater(ctx, a, b);
    defer gt.deinit();
    try std.testing.expectEqual(gt.ndim(), 2);

    const all_r = try mlx.comparison.all(ctx, a, false);
    defer all_r.deinit();
    try std.testing.expectEqual(all_r.ndim(), 0);

    const any_r = try mlx.comparison.any(ctx, a, false);
    defer any_r.deinit();
    try std.testing.expectEqual(any_r.ndim(), 0);

    const allclose_r = try mlx.comparison.allClose(ctx, a, b, 1e-5, 1e-8, false);
    defer allclose_r.deinit();
    try std.testing.expectEqual(allclose_r.ndim(), 0);
}

// ============================================================
// Creation Ops (from dmlx creation_tests.zig)
// ============================================================

test "creation ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const ctx = mlx.EagerContext.init(alloc);

    const z = try mlx.creation.zeros(ctx, &[_]i32{ 2, 3 }, .float32);
    defer z.deinit();
    try std.testing.expectEqual(z.size(), 6);

    const o = try mlx.creation.ones(ctx, &[_]i32{ 3, 3 }, .float32);
    defer o.deinit();
    try std.testing.expectEqual(o.size(), 9);

    const e = try mlx.creation.eye(ctx, 4, 4, .float32);
    defer e.deinit();
    try std.testing.expectEqual(e.shape()[0], 4);
    try std.testing.expectEqual(e.shape()[1], 4);

    const ar = try mlx.creation.arange(ctx, 0, 10, 1, .float32);
    defer ar.deinit();
    try std.testing.expectEqual(ar.size(), 10);

    const ls = try mlx.creation.linspace(ctx, 0, 1, 5, .float32);
    defer ls.deinit();
    try std.testing.expectEqual(ls.size(), 5);
}

// ============================================================
// FFT Ops (from dmlx fft_tests.zig)
// ============================================================

test "fft ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{4});
    defer a.deinit();

    const ctx = mlx.EagerContext.init(alloc);

    const f = try mlx.fft.fft(ctx, a, 4, -1);
    defer f.deinit();
    try std.testing.expectEqual(f.ndim(), 1);

    const rf = try mlx.fft.rfft(ctx, a, 4, -1);
    defer rf.deinit();
    try std.testing.expectEqual(rf.ndim(), 1);
}

test "fft: fftfreq" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const ctx = mlx.EagerContext.init(alloc);

    const freqs = try mlx.fft.fftfreq(ctx, 8, 1.0);
    defer freqs.deinit();
    try std.testing.expectEqual(freqs.size(), 8);
}

test "fft: rfftfreq" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const ctx = mlx.EagerContext.init(alloc);

    const freqs = try mlx.fft.rfftfreq(ctx, 8, 1.0);
    defer freqs.deinit();
    try std.testing.expectEqual(freqs.size(), 5);
}

// ============================================================
// Linalg Ops (from dmlx linalg_tests.zig)
// ============================================================

test "linalg ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 4, 7, 2, 6 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 2 });
    defer a.deinit();

    const ctx = mlx.EagerContext.init(alloc);

    const inv_a = try mlx.linalg.inv(ctx, a);
    defer inv_a.deinit();
    try std.testing.expectEqual(inv_a.ndim(), 2);

    const qr = try mlx.linalg.qr(ctx, a);
    defer qr.q.deinit();
    defer qr.r.deinit();
    try std.testing.expectEqual(qr.q.ndim(), 2);
    try std.testing.expectEqual(qr.r.ndim(), 2);
}

test "linalg: tensordot" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a = try mlx.Array.fromSlice(alloc, f32, &[_]f32{ 1, 2, 3, 4 });
    defer a.deinit();
    const a2d = try mlx.shape.reshape(mlx.EagerContext.init(alloc), a, &[_]i32{ 2, 2 });
    defer a2d.deinit();
    const b2d = try mlx.shape.reshape(mlx.EagerContext.init(alloc), a, &[_]i32{ 2, 2 });
    defer b2d.deinit();

    const ctx = mlx.EagerContext.init(alloc);
    const result = try mlx.linalg.tensordotAxis(ctx, a2d, b2d, 1);
    defer result.deinit();

    try std.testing.expectEqual(result.ndim(), 2);
}

// ============================================================
// Reduce Ops (from dmlx reduce_tests.zig)
// ============================================================

test "reduce ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{ 2, 3 });
    defer a.deinit();

    const ctx = mlx.EagerContext.init(alloc);

    const s = try mlx.reduce.sumAxis(ctx, a, 0, false);
    defer s.deinit();
    try std.testing.expectEqual(s.ndim(), 1);

    const m = try mlx.reduce.meanAxis(ctx, a, 1, false);
    defer m.deinit();
    try std.testing.expectEqual(m.ndim(), 1);

    const am = try mlx.reduce.argmaxAxis(ctx, a, 0, false);
    defer am.deinit();

    const cs = try mlx.reduce.cumsum(ctx, a, 0, false, false);
    defer cs.deinit();
    try std.testing.expectEqual(cs.ndim(), 2);

    const tk = try mlx.reduce.topk(ctx, a, 2);
    defer tk.deinit();
}

// ============================================================
// Random Ops (from dmlx random_tests.zig)
// ============================================================

test "random ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const ctx = mlx.EagerContext.init(alloc);

    const k = try mlx.random.key(42);
    defer k.deinit();

    const n = try mlx.random.normal(ctx, &[_]i32{ 2, 2 }, .float32, 0.0, 1.0, k);
    defer n.deinit();
    try std.testing.expectEqual(n.ndim(), 2);

    const u = try mlx.random.uniform(ctx, k, k, &[_]i32{ 2, 2 }, .float32, k);
    defer u.deinit();
    try std.testing.expectEqual(u.ndim(), 2);
}

// ============================================================
// Sort Ops (from dmlx sort_tests.zig)
// ============================================================

test "sort ops" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const a_data = [_]f32{ 3, 1, 2 };
    const a = try mlx.Array.fromData(alloc, f32, &a_data, &[_]i32{3});
    defer a.deinit();

    const ctx = mlx.EagerContext.init(alloc);

    const s = try mlx.sort.sort(ctx, a);
    defer s.deinit();
    try std.testing.expectEqual(s.ndim(), 1);

    const as = try mlx.sort.argsort(ctx, a);
    defer as.deinit();
    try std.testing.expectEqual(as.ndim(), 1);

    const pt = try mlx.sort.partition(ctx, a, 1);
    defer pt.deinit();
    try std.testing.expectEqual(pt.ndim(), 1);
}

// ============================================================
// Batch Ops (einsum)
// ============================================================

test "ops: einsum" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const ctx = mlx.EagerContext.init(alloc);

    const a = try mlx.Array.fromSlice(alloc, f32, &[_]f32{ 1, 2, 3, 4 });
    defer a.deinit();
    const a2d = try mlx.shape.reshape(ctx, a, &[_]i32{ 2, 2 });
    defer a2d.deinit();

    const result = try mlx.batch.einsum(ctx, "ij->ji", &[_]mlx.Array{a2d});
    defer result.deinit();

    try std.testing.expectEqual(result.ndim(), 2);
}

// ============================================================
// I/O: Safetensors (from dmlx safetensors_tests.zig)
// ============================================================

test "save and load safetensors roundtrip" {
    const allocator = std.testing.allocator;
    const full_path = "/tmp/mlx_z_test_roundtrip.safetensors";

    var weights = std.StringHashMap(mlx.Array).init(allocator);
    defer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const arr = c.c.mlx_array_new_data(&data, &[_]c_int{ 2, 2 }, 2, c.c.MLX_FLOAT32);
    const owned_key = try allocator.dupe(u8, "weight");
    try weights.put(owned_key, mlx.Array.fromHandle(arr));

    var metadata = std.StringHashMap([]const u8).init(allocator);
    defer {
        var it = metadata.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        metadata.deinit();
    }
    const meta_key = try allocator.dupe(u8, "format");
    const meta_val = try allocator.dupe(u8, "test");
    try metadata.put(meta_key, meta_val);

    try mlx.io.saveSafetensors(allocator, full_path, weights, metadata);

    var loaded = try mlx.io.loadSafetensors(allocator, full_path);
    defer loaded.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), loaded.weights.count());
    const loaded_arr = loaded.weights.get("weight").?;
    const shape = loaded_arr.shape();
    try std.testing.expectEqual(@as(i32, 2), shape[0]);
    try std.testing.expectEqual(@as(i32, 2), shape[1]);

    const loaded_data = try loaded_arr.dataSlice(f32);
    try std.testing.expectEqual(@as(f32, 1.0), loaded_data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), loaded_data[1]);
    try std.testing.expectEqual(@as(f32, 3.0), loaded_data[2]);
    try std.testing.expectEqual(@as(f32, 4.0), loaded_data[3]);

    try std.testing.expectEqual(@as(usize, 1), loaded.metadata.count());
    try std.testing.expectEqualStrings("test", loaded.metadata.get("format").?);
}

test "loadSafetensors empty metadata" {
    const allocator = std.testing.allocator;
    const full_path = "/tmp/mlx_z_test_empty.safetensors";

    var weights = std.StringHashMap(mlx.Array).init(allocator);
    defer {
        var it = weights.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        weights.deinit();
    }

    const data = [_]f32{42.0};
    const arr = c.c.mlx_array_new_data(&data, &[_]c_int{1}, 1, c.c.MLX_FLOAT32);
    const owned_key = try allocator.dupe(u8, "scalar");
    try weights.put(owned_key, mlx.Array.fromHandle(arr));

    var metadata = std.StringHashMap([]const u8).init(allocator);
    defer {
        var it = metadata.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        metadata.deinit();
    }

    try mlx.io.saveSafetensors(allocator, full_path, weights, metadata);

    var loaded = try mlx.io.loadSafetensors(allocator, full_path);
    defer loaded.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), loaded.weights.count());
    try std.testing.expectEqual(@as(usize, 0), loaded.metadata.count());
}

// ============================================================
// Array Arena (from dmlx arena_tests.zig)
// ============================================================

const ScopedArrayArena = mlx.array_arena.ScopedArrayArena;

fn simulatedForwardPass(alloc: std.mem.Allocator, ctx: mlx.EagerContext, input: mlx.Array) !mlx.Array {
    var arena = ScopedArrayArena.init(alloc);
    defer arena.deinit();

    const zeros = try arena.track(try mlx.creation.zeros(ctx, &[_]i32{4}, .float32));
    const ones_arr = try arena.track(try mlx.creation.ones(ctx, &[_]i32{4}, .float32));
    const sum1 = try arena.track(try mlx.ops.add(ctx, input, zeros));
    const sum2 = try arena.track(try mlx.ops.add(ctx, sum1, ones_arr));
    const prod = try arena.track(try mlx.ops.multiply(ctx, sum2, ones_arr));

    return mlx.ops.add(ctx, prod, ones_arr);
}

test "arena cleanup — final output valid after arena deinit" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const ctx = mlx.EagerContext.init(alloc);

    var prng = std.Random.DefaultPrng.init(12345);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 20) : (iteration += 1) {
        var input_data: [4]f32 = undefined;
        for (&input_data) |*v| {
            v.* = rand.float(f32) * 20.0 - 10.0;
        }

        const input = try mlx.Array.fromData(alloc, f32, &input_data, &[_]i32{4});
        defer input.deinit();

        const result = try simulatedForwardPass(alloc, ctx, input);
        defer result.deinit();

        try result.eval();
        const data = try result.dataSlice(f32);
        try std.testing.expectEqual(@as(usize, 4), data.len);
        for (data, 0..) |val, i| {
            const expected = input_data[i] + 2.0;
            try std.testing.expectApproxEqAbs(expected, val, 1e-5);
        }
    }
}

test "arena cleanup — multiple tracked arrays freed, untracked survives" {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const ctx = mlx.EagerContext.init(alloc);

    var prng = std.Random.DefaultPrng.init(99999);
    const rand = prng.random();

    var iteration: usize = 0;
    while (iteration < 20) : (iteration += 1) {
        const dim: i32 = @intCast(rand.intRangeAtMost(u32, 1, 8));

        var arena = ScopedArrayArena.init(alloc);

        const z = try arena.track(try mlx.creation.zeros(ctx, &[_]i32{dim}, .float32));
        const o = try arena.track(try mlx.creation.ones(ctx, &[_]i32{dim}, .float32));
        const sum_arr = try arena.track(try mlx.ops.add(ctx, z, o));
        const doubled = try arena.track(try mlx.ops.add(ctx, sum_arr, o));

        const final_result = try mlx.ops.add(ctx, doubled, o);
        arena.deinit();

        defer final_result.deinit();
        try final_result.eval();
        const data = try final_result.dataSlice(f32);

        for (data) |val| {
            try std.testing.expectApproxEqAbs(@as(f32, 3.0), val, 1e-5);
        }
    }
}

// ============================================================
// Device Info
// ============================================================

test "device: count" {
    c.initErrorHandler();
    const cpu_count = try mlx.device.deviceCount(.cpu);
    try std.testing.expect(cpu_count > 0);
}

test "device: info" {
    c.initErrorHandler();
    const dev = mlx.device.Device.cpu();
    defer dev.deinit();
    const info = try mlx.device.DeviceInfo.new(dev);
    defer info.deinit();

    const has_name = try info.hasKey("device_name");
    if (has_name) {
        const name = try info.getString("device_name");
        _ = name;
    }
}

// ============================================================
// Metal
// ============================================================

test "metal: is available" {
    c.initErrorHandler();
    const avail = try mlx.metal.isAvailable();
    _ = avail;
}
