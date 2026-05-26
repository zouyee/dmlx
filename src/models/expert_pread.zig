/// Parallel pread loader for packed expert binary files.
///
/// Reads from per-layer binary files (created by scripts/repack_experts.py)
/// where each expert is a contiguous blob: gate+scales+up+scales+down+scales.
///
/// Layout per expert in layer_XX.bin:
///   [w1.weight] [w1.scales] [w3.weight] [w3.scales] [w2.weight] [w2.scales]
///   All experts stored sequentially: expert_offset = expert_id * expert_size
const std = @import("std");
const c = @import("mlx").c;
const array_mod = @import("mlx").array;
const ops = @import("mlx").ops;
const shape_mod = @import("mlx").shape;

const Array = array_mod.Array;
const EagerContext = ops.EagerContext;
const unistd = @cImport(@cInclude("unistd.h"));
const pthread = @cImport(@cInclude("pthread.h"));

const MAX_PARALLEL: usize = 18;
const MAX_LAYERS: usize = 64;

/// Per-expert I/O task for the persistent thread pool.
const IOTask = struct {
    fd: c_int,
    buf: [*]u8,
    size: usize,
    offset: i64,
    result: isize,
};

/// Persistent I/O thread pool (flash-moe pattern).
/// Eliminates pthread_create/join overhead per expert per layer.
const IOPool = struct {
    mutex: pthread.pthread_mutex_t,
    work_ready: pthread.pthread_cond_t,
    work_done: pthread.pthread_cond_t,
    tasks: [MAX_PARALLEL]IOTask,
    num_tasks: usize,
    tasks_done: usize,
    generation: usize,
    shutdown: bool,
    threads: [MAX_PARALLEL]std.Thread,
};

pub const ExpertPreadLoader = struct {
    allocator: std.mem.Allocator,
    layer_fds: [MAX_LAYERS]c_int,
    n_layers: usize,
    expert_size: usize,
    n_experts: usize,
    buffers: [MAX_PARALLEL][]u8,
    max_parallel: usize,
    pool: IOPool,
    component_offsets: [6]usize,
    component_sizes: [6]usize,
    layer_idx_to_packed: [MAX_LAYERS]?usize,

    pub fn init(allocator: std.mem.Allocator, packed_dir: []const u8, max_parallel: usize) !ExpertPreadLoader {
        var self = ExpertPreadLoader{
            .allocator = allocator,
            .layer_fds = [_]c_int{-1} ** MAX_LAYERS,
            .n_layers = 0,
            .expert_size = 0,
            .n_experts = 256,
            .buffers = undefined,
            .max_parallel = @min(max_parallel, MAX_PARALLEL),
            .pool = undefined,
            .component_offsets = [_]usize{0} ** 6,
            .component_sizes = [_]usize{0} ** 6,
            .layer_idx_to_packed = [_]?usize{null} ** MAX_LAYERS,
        };

        // Load manifest
        var manifest_path_buf: [4096]u8 = undefined;
        const manifest_path = try std.fmt.bufPrint(&manifest_path_buf, "{s}/manifest.json", .{packed_dir});
        const c_stdio = @cImport(@cInclude("stdio.h"));
        const fp = c_stdio.fopen(manifest_path.ptr, "r");
        if (fp == null) return error.FileNotFound;
        defer _ = c_stdio.fclose(fp);

        _ = c_stdio.fseek(fp, 0, c_stdio.SEEK_END);
        const manifest_len = c_stdio.ftell(fp);
        _ = c_stdio.fseek(fp, 0, c_stdio.SEEK_SET);
        const manifest_data = try allocator.alloc(u8, @intCast(manifest_len));
        defer allocator.free(manifest_data);
        _ = c_stdio.fread(manifest_data.ptr, 1, @intCast(manifest_len), fp);

        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, manifest_data, .{});
        defer parsed.deinit();
        const root = parsed.value.object;
        self.n_experts = @intCast(root.get("n_experts").?.integer);

        const layers_obj = root.get("layers").?.object;
        var layers_it = layers_obj.iterator();
        while (layers_it.next()) |entry| {
            const layer_idx = std.fmt.parseInt(usize, entry.key_ptr.*, 10) catch continue;
            const layer_meta = entry.value_ptr.*.object;
            const expert_size: usize = @intCast(layer_meta.get("expert_size").?.integer);
            if (self.expert_size == 0) self.expert_size = expert_size;

            if (layer_meta.get("component_sizes")) |cs_val| {
                const cs = cs_val.object;
                const comp_keys = [_][]const u8{
                    "ffn.switch_mlp.gate_proj.weight", "ffn.switch_mlp.gate_proj.scales",
                    "ffn.switch_mlp.up_proj.weight",   "ffn.switch_mlp.up_proj.scales",
                    "ffn.switch_mlp.down_proj.weight", "ffn.switch_mlp.down_proj.scales",
                };
                var off: usize = 0;
                for (comp_keys, 0..) |key, ci| {
                    if (cs.get(key)) |size_val| {
                        const sz: usize = @intCast(size_val.integer);
                        self.component_offsets[ci] = off;
                        self.component_sizes[ci] = sz;
                        off += sz;
                    }
                }
            }

            var path_buf: [4096]u8 = undefined;
            const file_name = if (layer_meta.get("file")) |f| f.string else blk: {
                break :blk try std.fmt.bufPrint(&path_buf, "layer_{d:0>2}.bin", .{layer_idx});
            };
            var full_buf: [4096]u8 = undefined;
            const full_path = try std.fmt.bufPrint(&full_buf, "{s}/{s}", .{ packed_dir, file_name });
            const c_fcntl = @cImport(@cInclude("fcntl.h"));
            const fd = c_fcntl.open(full_path.ptr, c_fcntl.O_RDONLY);
            if (fd < 0) continue;
            _ = std.c.fcntl(fd, std.c.F.RDAHEAD, @as(c_int, 1));

            self.layer_fds[self.n_layers] = fd;
            self.layer_idx_to_packed[layer_idx] = self.n_layers;
            self.n_layers += 1;
        }

        // flash-moe: 2MB-aligned buffers for optimal DMA (3.6x faster).
        // posix_memalign ensures SSD DMA controller transfers at full speed.
        const ALIGN_2MB = 2 * 1024 * 1024;
        const aligned_size = (self.expert_size + ALIGN_2MB - 1) & ~@as(usize, ALIGN_2MB - 1);
        for (0..self.max_parallel) |i| {
            var ptr: ?*anyopaque = null;
            if (std.c.posix_memalign(&ptr, ALIGN_2MB, aligned_size) != 0) {
                return error.OutOfMemory;
            }
            self.buffers[i] = @as([*]u8, @ptrCast(ptr.?))[0..self.expert_size];
        }

        std.log.info("[ExpertPreadLoader] {d} layers, {d} experts/layer, {d}MB/expert", .{ self.n_layers, self.n_experts, self.expert_size / (1024 * 1024) });
        return self;
    }

    /// Start persistent I/O thread pool. Must be called AFTER the loader
    /// is stored at a stable heap address (self pointer must outlive threads).
    pub fn startPool(self: *ExpertPreadLoader) !void {
        _ = pthread.pthread_mutex_init(&self.pool.mutex, null);
        _ = pthread.pthread_cond_init(&self.pool.work_ready, null);
        _ = pthread.pthread_cond_init(&self.pool.work_done, null);
        self.pool.num_tasks = 0;
        self.pool.tasks_done = 0;
        self.pool.generation = 0;
        self.pool.shutdown = false;
        for (0..self.max_parallel) |i| {
            self.pool.threads[i] = try std.Thread.spawn(.{}, ioPoolWorker, .{ self, i });
        }
        std.log.info("[ExpertPreadLoader] I/O pool started: {d} threads", .{self.max_parallel});
    }

    pub fn deinit(self: *ExpertPreadLoader) void {
        _ = pthread.pthread_mutex_lock(&self.pool.mutex);
        self.pool.shutdown = true;
        _ = pthread.pthread_cond_broadcast(&self.pool.work_ready);
        _ = pthread.pthread_mutex_unlock(&self.pool.mutex);
        for (0..self.max_parallel) |i| {
            self.pool.threads[i].join();
        }

        for (0..self.n_layers) |i| {
            if (self.layer_fds[i] != -1) _ = std.c.close(self.layer_fds[i]);
        }
        for (0..self.max_parallel) |i| {
            if (self.buffers[i].len > 0) std.c.free(@ptrCast(self.buffers[i].ptr));
        }
    }

    pub fn hasLayer(self: *const ExpertPreadLoader, layer_idx: usize) bool {
        return layer_idx < MAX_LAYERS and self.layer_idx_to_packed[layer_idx] != null;
    }

    /// Read experts via persistent I/O thread pool (flash-moe pattern).
    /// Eliminates pthread_create/join per expert per layer (200+/token).
    fn readExperts(self: *ExpertPreadLoader, layer_idx: usize, expert_ids: []const u32) !usize {
        const packed_idx = self.layer_idx_to_packed[layer_idx] orelse return error.LayerNotFound;
        const fd = self.layer_fds[packed_idx];
        const n = @min(expert_ids.len, self.max_parallel);

        // std.Thread.spawn — stable, no race. Pool kept for future optimization.
        var threads: [MAX_PARALLEL]?std.Thread = .{null} ** MAX_PARALLEL;
        var results: [MAX_PARALLEL]isize = .{0} ** MAX_PARALLEL;
        for (0..n) |i| {
            const offset: i64 = @intCast(@as(u64, expert_ids[i]) * self.expert_size);
            threads[i] = std.Thread.spawn(.{}, preadWorker, .{ fd, self.buffers[i].ptr, self.expert_size, offset, &results[i] }) catch {
                results[i] = unistd.pread(fd, self.buffers[i].ptr, self.buffers[i].len, offset);
                continue;
            };
        }
        for (0..n) |i| if (threads[i]) |t| t.join();
        var success: usize = 0;
        for (0..n) |i| {
            if (results[i] == @as(isize, @intCast(self.expert_size))) success += 1;
        }
        return success;
    }

    /// Read experts once, return all 6 weight components.
    /// Avoids redundant reads when loading gate+up+down projections.
    pub fn readAndAssembleAll(
        self: *ExpertPreadLoader,
        ctx: EagerContext,
        layer_idx: usize,
        expert_ids: []const u32,
    ) !struct { gate: Array, up: Array, down: Array, gs: ?Array, us: ?Array, ds: ?Array } {
        if (expert_ids.len == 0) return error.EmptyBatch;
        if (expert_ids.len > self.max_parallel) {
            std.log.warn("[Pread] TooManyExperts layer {d}: {d} > {d}", .{ layer_idx, expert_ids.len, self.max_parallel });
            return error.TooManyExperts;
        }

        const nread = self.readExperts(layer_idx, expert_ids) catch |e| {
            std.log.warn("[Pread] readExperts error layer {d}: {}", .{ layer_idx, e });
            return e;
        };
        if (nread != expert_ids.len) {
            std.log.warn("[Pread] readExperts partial layer {d}: {d}/{d}", .{ layer_idx, nread, expert_ids.len });
        }
        const n = expert_ids.len;
        const gate = try self.assembleComponent(ctx, 0, n);
        const up = try self.assembleComponent(ctx, 2, n);
        const down = try self.assembleComponent(ctx, 4, n);
        const gs: ?Array = if (self.component_sizes[1] > 0) try self.assembleComponent(ctx, 1, n) else null;
        const us: ?Array = if (self.component_sizes[3] > 0) try self.assembleComponent(ctx, 3, n) else null;
        const ds: ?Array = if (self.component_sizes[5] > 0) try self.assembleComponent(ctx, 5, n) else null;
        return .{ .gate = gate, .up = up, .down = down, .gs = gs, .us = us, .ds = ds };
    }

    fn assembleComponent(self: *ExpertPreadLoader, ctx: EagerContext, component_idx: usize, count: usize) !Array {
        const comp_off = self.component_offsets[component_idx];
        const comp_sz = self.component_sizes[component_idx];
        if (count == 0 or comp_sz == 0) return error.EmptyBatch;

        const sd = getComponentShapeDtype(component_idx, @intCast(count));
        const shape = sd.shape[0..sd.shape_len];

        if (count == 1) {
            const data = self.buffers[0][comp_off..][0..comp_sz];
            return Array.fromHandle(c.c.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), sd.dtype));
        }

        var arrs = try self.allocator.alloc(Array, count);
        defer self.allocator.free(arrs);
        for (0..count) |i| {
            const data = self.buffers[i][comp_off..][0..comp_sz];
            const ssd = getComponentShapeDtype(component_idx, 1);
            const sshape = ssd.shape[0..ssd.shape_len];
            arrs[i] = Array.fromHandle(c.c.mlx_array_new_data(data.ptr, sshape.ptr, @intCast(sshape.len), ssd.dtype));
        }
        defer for (arrs) |a| a.deinit();
        return try shape_mod.concatenateAxis(ctx, arrs, 0);
    }
};

fn getComponentShapeDtype(component_idx: usize, n_experts: i32) struct { shape: [3]i32, shape_len: usize, dtype: c.c.mlx_dtype } {
    const shapes = comptime [6]struct { s0: i32, s1: i32, d: c.c.mlx_dtype }{
        .{ .s0 = 2048, .s1 = 512, .d = c.c.MLX_UINT32 },
        .{ .s0 = 2048, .s1 = 128, .d = c.c.MLX_UINT8 },
        .{ .s0 = 2048, .s1 = 512, .d = c.c.MLX_UINT32 },
        .{ .s0 = 2048, .s1 = 128, .d = c.c.MLX_UINT8 },
        .{ .s0 = 4096, .s1 = 256, .d = c.c.MLX_UINT32 },
        .{ .s0 = 4096, .s1 = 64, .d = c.c.MLX_UINT8 },
    };
    const s = shapes[component_idx];
    return .{ .shape = .{ n_experts, s.s0, s.s1 }, .shape_len = 3, .dtype = s.d };
}

fn ioPoolWorker(loader: *ExpertPreadLoader, tid: usize) void {
    var gen: usize = 0;
    _ = pthread.pthread_mutex_lock(&loader.pool.mutex);
    while (!loader.pool.shutdown) {
        while (loader.pool.generation == gen and !loader.pool.shutdown) {
            _ = pthread.pthread_cond_wait(&loader.pool.work_ready, &loader.pool.mutex);
        }
        if (loader.pool.shutdown) break;
        gen = loader.pool.generation;
        const n = loader.pool.num_tasks;
        _ = pthread.pthread_mutex_unlock(&loader.pool.mutex);

        // Strided work distribution (flash-moe pattern): thread `tid` processes
        // tasks at indices tid, tid+N, tid+2N, ...
        var i: usize = tid;
        while (i < n) : (i += loader.max_parallel) {
            const t = &loader.pool.tasks[i];
            const buf = t.buf[0..t.size];
            var total: usize = 0;
            var off = t.offset;
            while (total < t.size) {
                const nr = unistd.pread(t.fd, buf[total..].ptr, buf[total..].len, off);
                if (nr < 0 or nr == 0) break;
                total += @intCast(nr);
                off += @intCast(nr);
            }
            t.result = if (total == t.size) @intCast(total) else -1;
        }

        _ = pthread.pthread_mutex_lock(&loader.pool.mutex);
        // Count completed tasks (not threads). Race-safe: only count
        // if generation matches (slow worker from prev gen is ignored).
        if (loader.pool.generation == gen) {
            // Count each task we completed
            var ti: usize = tid;
            while (ti < n) : (ti += loader.max_parallel) {
                loader.pool.tasks_done += 1;
            }
            _ = pthread.pthread_cond_signal(&loader.pool.work_done);
        }
    }
    _ = pthread.pthread_mutex_unlock(&loader.pool.mutex);
}

fn preadWorker(fd: c_int, buf: [*]u8, size: usize, offset: i64, result: *isize) void {
    const slice = buf[0..size];
    var total: usize = 0;
    var off = offset;
    while (total < size) {
        const n = unistd.pread(fd, slice[total..].ptr, slice[total..].len, off);
        if (n < 0) {
            result.* = -1;
            return;
        }
        if (n == 0) break;
        total += @intCast(n);
        off += @intCast(n);
    }
    result.* = if (total == size) @intCast(total) else -1;
}
