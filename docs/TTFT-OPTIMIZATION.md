# TTFT 优化方案 v3

## 现状回顾

DeepSeek-V4-Flash-4bit, 33 shards (~150GB), MacBook Pro 48GB RAM

```
阶段                                              耗时      占比
──────────────────────────────────────────────────────────────
A. 读取 model.safetensors.index.json (80KB)       ~0.5s
B. JSON 解析 + 去重 33 个分片                      ~0.5s
C. 33 个分片串行: open→read(header)→parse→close    ~66s
                                          索引小计  ~67s     49%
──────────────────────────────────────────────────────────────
D. FdPool 打开 33 分片                             ~1s
E. MmapPool mmap 33 分片                          ~2s
F. 遍历 2481 条目计数                              ~0.5s
G. 加载 2223 个 backbone 权重 (mmap→mlx_array)     ~65s
                                          加载小计  ~68.5s   51%
──────────────────────────────────────────────────────────────
                                          总计      ~137s
```

原方案的问题：
1. P1（并行 header 读取）在 P0 落地后无增量收益，定位不准
2. 权重加载阶段（65s）完全没有优化方案
3. 缺少对 `loadTensor` 中重复 open/close 的分析（每次调用都 open→pread→close）
4. 组合效果表中 P0+P1 = P0，说明 P1 价值被高估

---

## 优化方案（重新排序）

### Phase 1: 索引阶段优化（67s → 1s）

#### P0: 二进制索引缓存 ⭐⭐⭐

**原理**: `buildIndexFromDirectory` 每次启动都串行解析 33 个分片的 JSON header，
但 header 内容在模型文件不变时完全相同。序列化 `TensorIndex` 为紧凑二进制格式，
后续启动直接 mmap 读取。

**实现**:

```zig
// 缓存文件格式: model.mlxidx
// ┌──────────────────────────────────────────────┐
// │ Header (固定 40 bytes)                        │
// │   magic: "MLXI" (4 bytes)                    │
// │   version: u32                               │
// │   index_json_size: u64  (失效检测用)           │
// │   index_json_mtime: i64 (失效检测用)           │
// │   entry_count: u32                           │
// │   string_table_offset: u64                   │
// ├──────────────────────────────────────────────┤
// │ Entry Offset Table (entry_count × u64)       │
// │   entry_offsets[0]: u64  → Entry[0] 的偏移    │
// │   entry_offsets[1]: u64  → Entry[1] 的偏移    │
// │   ...                                        │
// ├──────────────────────────────────────────────┤
// │ Entry[0] (变长):                              │
// │   name_off(u32), name_len(u16),              │
// │   dtype(u8), ndim(u8),                       │
// │   shape[ndim] (i64),  ← ndim 不固定，故变长   │
// │   offset_start(u64), offset_end(u64),        │
// │   shard_path_off(u32), shard_path_len(u16)   │
// │ Entry[1]: ...                                │
// ├──────────────────────────────────────────────┤
// │ String Table: 所有 name 和 path 的拼接         │
// └──────────────────────────────────────────────┘
// 注: Entry 因 shape[ndim] 而变长，通过 Entry Offset Table
// 支持 O(1) 随机访问，mmap 后无需顺序扫描。

pub fn loadOrBuildIndex(allocator: Allocator, dir_path: []const u8) !TensorIndex {
    const cache_path = join(dir_path, "model.mlxidx");
    if (isCacheValid(cache_path, dir_path)) {
        return deserializeIndex(allocator, cache_path);  // mmap + 解析 ~1s
    }
    var index = try buildIndexFromDirectory(allocator, dir_path);
    serializeIndex(&index, cache_path);  // 写缓存供下次使用
    return index;
}
```

**失效检测**: 检查 `model.safetensors.index.json` 的 mtime + file_size 双重校验。
mtime 在 NFS/APFS 上可能有精度问题，加上 file_size 作为辅助可以覆盖绝大多数场景。

**收益**: 索引 67s → 1s。所有模式受益。

**实现复杂度**: 低。纯 I/O 序列化，不涉及模型逻辑。

---

#### P0.5: 首次运行并行 header 解析（仅无缓存时生效）

**原理**: 首次运行（无 `.mlxidx` 缓存）时，33 个分片的 header 解析仍需 67s。
用线程池并行化可以将首次运行从 67s 降到 ~10s。

**实现**:

```zig
// 每个分片独立解析到自己的 TensorIndex，最后合并（避免共享 HashMap 的 race condition）
pub fn buildIndexFromDirectoryParallel(allocator: Allocator, dir_path: []const u8) !TensorIndex {
    // ... 收集 shard_paths 同前 ...

    // 每个分片独立解析到自己的 TensorIndex，最后合并
    var per_shard_indices = try allocator.alloc(TensorIndex, shard_paths.len);
    
    var pool = try std.Thread.Pool.init(.{ .allocator = allocator });
    defer pool.deinit();
    
    for (shard_paths, 0..) |path, i| {
        pool.spawn(parseShardWorker, .{ &per_shard_indices[i], path });
    }
    pool.waitAll();
    
    // 合并所有 per-shard index 到一个 TensorIndex
    return mergeIndices(allocator, per_shard_indices);
}
```

**注意**: 这不是独立方案，而是 P0 的补充 — 只在首次运行时触发。
有了 P0 缓存后，后续运行不会走这条路径。

**线程安全**: 每个分片解析到独立的 `TensorIndex`，最后单线程 `mergeIndices` 合并，
不存在共享状态竞争。`TensorIndex.addShard` 本身不需要改造为线程安全。

**收益**: 首次运行 67s → ~10s（8 线程并发，受 NVMe 带宽限制）。

---

#### P1: Stream 模式懒索引（方案已弃用 → 降级为 P2/P3 的加载阶段优化）

**弃用原因**（2026-05-03 验证）:

原方案假设 shard 4-33 是纯 expert 分片可以跳过，但实际验证发现：
- **33/33 个分片全部包含 backbone 权重**（每分片 15~82 个 backbone tensor）
- **0 个纯 expert 分片**
- Shard 命名 `model-00004` 只是序号，与内容无关

```python
# 验证结果: 每个分片的 backbone tensor 数量
model-00001: 47    model-00004: 34    ...    model-00033: 15
model-00002: 48    全部 33 个分片都有 backbone，一个都不能跳过
model-00003: 82
```

因此，通过跳过 shard 来减少索引时间不可行。Stream 模式的真正收益来自**加载阶段**:
- **P2 零拷贝**: 省去 mmap→mlx_array 的 memcpy（已确认 `mlx_array_new_data_managed_payload` API 可用）
- **P3 顺序加载**: 按分片批量加载，充分发挥 OS readahead
- Expert 权重的跳过已在代码中实现（`isExpertWeight` 过滤，仅加载 1750 backbone tensor）

索引阶段的优化依赖 **P0 二进制缓存**（所有模式通用，67s → 1s）。

---

### Phase 2: 权重加载阶段优化（65s → 40s）

这是原方案完全忽略的部分。当前 `loadWeightsSelective` 的加载循环：

```
对 2223 个 backbone 权重逐个:
  mmap_pool.getSlice() → mlx_array_new_data() (内部 memcpy)
```

瓶颈分析：
- `mlx_array_new_data` 会 **拷贝** 数据到 MLX 管理的内存，即使数据已经 mmap 了
- 2223 个 tensor 的总大小约 7-10GB（backbone 部分），memcpy 本身约 3-5s
- 剩余时间花在：HashMap 操作、name mapping、shape 转换、MLX 内部分配

#### P2: mmap 零拷贝加载 ⭐⭐

**原理**: 当前 mmap 路径通过 `mlx_array_new_data` 拷贝数据（内部 `allocator::malloc` + `std::copy`）。
mlx-c **已经提供**零拷贝 API，可以直接使用。

**API 确认** (mlx-c/mlx/c/array.h):

mlx-c 提供两个零拷贝构造函数：

```c
// 变体 1: dtor 回调接收 data 指针
mlx_array mlx_array_new_data_managed(
    void* data, const int* shape, int dim, mlx_dtype dtype,
    void (*dtor)(void*));

// 变体 2: dtor 回调接收独立的 payload 指针（更灵活）
mlx_array mlx_array_new_data_managed_payload(
    void* data, const int* shape, int dim, mlx_dtype dtype,
    void* payload, void (*dtor)(void*));
```

**内部机制** (mlx/mlx/array.cpp):

```cpp
array::array(void* data, Shape shape, Dtype dtype,
             const std::function<void(void*)>& deleter) {
  auto buffer = allocator::make_buffer(data, nbytes());
  if (buffer.ptr() == nullptr) {
    // 回退路径: malloc + memcpy（与 mlx_array_new_data 相同）
  } else {
    // 零拷贝路径: 直接包装外部指针为 Metal buffer
  }
}
```

Metal 后端的 `make_buffer` 调用 `MTL::Device::newBuffer(ptr, size, StorageModeShared, nullptr)`，
将外部指针直接注册为 Metal 可访问的 GPU buffer，**不拷贝数据**。

**零拷贝成功条件**:
- 指针必须页对齐 ⚠️（见下方分析）
- size 应为页大小倍数（大 tensor 通常满足 ✅，小 tensor 可能不满足 ⚠️）
- 内存在 array 生命周期内有效（MmapPool 在程序退出时才 munmap ✅）

如果 `newBuffer` 返回 nullptr（指针非页对齐、size 非页对齐等），MLX 内部自动回退到拷贝路径，无需额外处理。

**指针对齐问题**:

`mmap` 返回的基址是页对齐的，但 `mmap_pool.getSlice(shard_path, offset, length)` 返回的是
`base_ptr + data_offset_start`。单个 tensor 的 `data_offset_start` 取决于它在 shard 文件中的
排列位置，**不一定页对齐**（页大小 = 16KB on Apple Silicon）。

这意味着大部分 tensor 的 slice 指针实际上不是页对齐的，`MTL::Device::newBuffer(ptr, ...)` 会
返回 nullptr，回退到拷贝路径。零拷贝的实际命中率取决于 safetensors 文件中 tensor 的排列方式。

**实测优先**: 由于命中率不确定，P2 的实际收益可能从"省 3-5s"到"几乎为零"。
建议 Sprint 1 落地后先用验证方法（对比指针地址）统计命中率，再决定是否值得保留。
即使命中率低，改动本身很小（3 处函数调用替换），不会引入风险。

**实现**:

```zig
// 当前代码 (safetensors_reader.zig, deepseek_v4_loader.zig):
// 总是拷贝 — mlx_array_new_data 内部 malloc + std::copy
const arr = c.c.mlx_array_new_data(
    slice.ptr, shape_i32.ptr, @intCast(shape_i32.len), mlx_dtype,
);

// 零拷贝版本:
// 使用 mlx_array_new_data_managed_payload，传入 noop deleter
// 因为 MmapPool 统一管理 mmap 生命周期，array 释放时不需要做任何事
const arr = c.c.mlx_array_new_data_managed_payload(
    @constCast(@ptrCast(slice.ptr)),   // data: mmap 区域指针
    shape_i32.ptr,
    @intCast(shape_i32.len),
    mlx_dtype,
    null,                               // payload: 不需要
    noopDeleter,                        // dtor: 空操作
);

fn noopDeleter(_: ?*anyopaque) callconv(.c) void {}
```

**需要修改的位置**:
1. `src/io/safetensors_reader.zig` — `TensorIndex.loadTensor()` (行 163)
2. `src/models/deepseek_v4_loader.zig` — `loadWeightsSelective()` mmap 路径 (行 651)
3. `src/models/deepseek_v4_loader.zig` — pread fallback 路径 (行 670)
4. `src/io/safetensors_reader.zig` — `PartialTensorReader` 的 mmap 路径 (行 741)

其中位置 3（pread fallback）仍需拷贝，因为 buffer 是临时分配的。
位置 1/2/4 的 mmap 路径可以改为零拷贝。

**收益**: 省去 ~7GB memcpy（约 3-5s）。但由于指针对齐问题（见上），实际零拷贝命中率不确定。

最好情况（大部分 tensor 恰好页对齐）: 加载 65s → 55-60s。
最差情况（几乎全部回退到拷贝）: 加载 65s → 65s（无收益，但也无损害）。

需要实测确认。改动量极小（3 处调用替换），即使收益为零也不会引入风险。

注意: 65s 加载时间中 memcpy 只占一部分，其余耗时来自：
- `MTL::Device::newBuffer` **注册 Metal resource**（即使零拷贝也有此开销，约每个 tensor 1-5ms）
- residency set 更新 + Metal 内部加锁
- HashMap 插入 2223 次 + name mapping + shape 数组分配
- OS 页面调度（mmap 首次访问触发 page fault）

因此零拷贝的实际收益取决于指针对齐命中率。预估加载时间从 65s → 55-65s（不确定）。
要确定性地降低加载时间，需要配合 P3（顺序 I/O）减少 page fault 开销。

**Metal 开销验证**: 建议落地后实测对比 `mlx_array_new_data` vs `_managed_payload` 的实际计时，
不要依赖预估。Metal `newBuffer(ptr)` 和 `malloc+memcpy+newBuffer(null)` 的耗时可能并非直观差距。

**实现复杂度**: 低。API 已存在，只需替换函数调用 + 添加 noop deleter。
不需要修改 mlx-c。

**生命周期安全性分析**:
- `MmapPool` 在 `ExpertStreamProvider.deinit()` 或程序退出时才 munmap
- 所有 backbone 权重 array 在模型推理期间持续存活
- 推理结束 → array deinit（noop deleter 被调用）→ MmapPool deinit（munmap）
- 顺序正确，不存在 use-after-munmap 风险

---

#### P3: 按分片批量加载（减少 HashMap 开销）

**原理**: 当前加载循环遍历 `index.entries`（HashMap 迭代），对每个 entry 做：
1. 检查是否跳过（expert/metadata 过滤）
2. dtype 转换 + shape 构建
3. mmap getSlice
4. mlx_array_new_data
5. name mapping
6. 插入 weights HashMap

HashMap 迭代是随机顺序，导致 mmap 访问模式也是随机的，不利于 OS 预读。

**实现**:

```zig
// 按 shard_path 分组，每组内按 offset 排序，顺序读取
pub fn loadWeightsSelectiveBatched(allocator: Allocator, ...) !StringHashMap(Array) {
    // 1. 按 shard 分组
    var by_shard = StringHashMap(ArrayList(TensorInfo)).init(allocator);
    var idx_it = index.entries.iterator();
    while (idx_it.next()) |entry| {
        // ... 过滤逻辑同前 ...
        const list = try by_shard.getOrPut(entry.value_ptr.shard_path);
        if (!list.found_existing) list.value_ptr.* = ArrayList(TensorInfo).init(allocator);
        try list.value_ptr.append(entry.value_ptr.*);
    }
    
    // 2. 每个 shard 内按 offset 排序
    for (by_shard.values()) |*list| {
        std.sort.sort(TensorInfo, list.items, {}, offsetLessThan);
    }
    
    // 3. 顺序读取 — OS readahead 生效
    // 配合 madvise(MADV_SEQUENTIAL) 已经设置，顺序访问会触发预读
    for (by_shard) |shard_path, list| {
        for (list.items) |info| {
            // ... 加载逻辑同前 ...
        }
    }
}
```

**收益**: 顺序 I/O 比随机 I/O 快 2-5x（NVMe 上差距较小，但仍有收益）。
预估加载时间从 65s → 40-50s。

---

#### P4: 合并 fd 复用（buildIndex → FdPool）

**原理**: `buildIndexFromDirectory` 中 `addShard` 对每个分片 open→read→close，
然后 `FdPool.openAll` 又对同样的 33 个分片 open。重复了 33 次 open + 33 次 close。

**实现**:

```zig
// 修改 addShard 返回 fd 而不是 close
pub fn addShardKeepFd(self: *TensorIndex, shard_path: []const u8) !c_int {
    // ... 解析 header 同前 ...
    // 不 close fd，返回给调用者
    return fd;
}

// buildIndexFromDirectory 收集所有 fd，传给 FdPool
pub fn buildIndexFromDirectory(allocator: Allocator, dir_path: []const u8) !struct { 
    index: TensorIndex, 
    fds: StringHashMap(c_int) 
} {
    // ... 
    for (shards) |shard| {
        const fd = try index.addShardKeepFd(shard_path);
        try fds.put(shard_path, fd);
    }
    return .{ .index = index, .fds = fds };
}
```

**收益**: 省去 33 次 close + 33 次 open。在 NVMe + APFS 上 open/close 每次约 10-50μs，
33 × 2 ≈ 0.6-3.3ms，性能收益可忽略。此方案的真正价值是**代码简洁性** — 避免对同一组
文件重复打开，减少资源泄漏风险。

---

### Phase 3: 端到端流水线优化

#### P5: 索引与加载流水线化 ⭐

**原理**: 当前流程是严格串行的：先索引全部 33 分片，再加载全部权重。
但实际上，一个分片索引完成后就可以立即开始加载该分片的权重，不需要等其他分片。

**实现**:

```zig
// 生产者-消费者模式
// 生产者线程: 逐个解析 shard header → 发送到 channel
// 消费者线程: 从 channel 接收 → mmap → 加载该 shard 的权重

const ShardReady = struct {
    shard_path: []const u8,
    entries: []TensorEntry,
};

pub fn loadWeightsPipelined(allocator: Allocator, dir_path: []const u8) !StringHashMap(Array) {
    var channel = Channel(ShardReady).init();
    
    // 生产者: 解析 shard headers
    const producer = try std.Thread.spawn(.{}, struct {
        fn run(ch: *Channel(ShardReady), shards: [][]const u8) void {
            for (shards) |shard| {
                const entries = parseShard(shard);
                ch.send(.{ .shard_path = shard, .entries = entries });
            }
            ch.close();
        }
    }.run, .{ &channel, shard_paths });
    
    // 消费者: 加载权重
    while (channel.recv()) |ready| {
        mmapAndLoadShard(ready.shard_path, ready.entries, &weights);
    }
    producer.join();
}
```

**收益**: 索引和加载重叠执行。理论上总时间 ≈ max(索引, 加载) 而非 索引 + 加载。
但由于 NVMe 带宽是共享的，实际收益约 20-30%。
预估: 137s → 90-100s（无 P0 缓存时）。

**注意**: 与 P0 缓存组合后，索引阶段只需 1s，流水线的收益就很小了。
此方案主要价值在首次运行场景。

**弃用建议**: P5 需要 Channel + 双线程协调，复杂度高但仅首次运行有 20-30% 收益。
P0 缓存落地后首次运行也只需 P0.5 并行解析，无需流水线。
建议标记为"长期可选"，Sprint 优先做 P0+P2+P3。

---

## 优化范围说明

本文的"零拷贝"不限于 mmap→array，而是贯穿整个推理流水线的数据搬运消除：

```
当前: Storage → OS page cache → mmap buffer → memcpy → MLX array → GPU
P0:   二进制索引直接 mmap → 省去 JSON 解析 + 33 次 open/read
P2:   mmap buffer → Metal newBuffer(ptr) → 省去 memcpy
P3:   顺序 I/O → OS readahead → 省去随机 page fault
```

收益是多维度的：**TTFT 延时**（用户感知）、**CPU 利用率**（page fault + memcpy 省去）、**代码简洁性**（P4 fd 复用）。

---

## 组合效果预估

### 非 Stream 模式（加载全部 backbone 权重，2223 个 tensor）

| 方案组合 | 索引 | 加载 | 总计 | 提升 | 实现难度 |
|---------|------|------|------|------|---------|
| 当前 | 67s | 68.5s | **137s** | — | — |
| P0 二进制缓存 | 1s | 68.5s | **69.5s** | 49%↓ | 低 |
| P0 + P4 fd复用 | 1s | 68s | **69s** | 50%↓ | 低 |
| P0 + P2 零拷贝 | 1s | 55-65s | **56-66s** | 52-55%↓ | 低-中 |
| P0 + P3 顺序加载 | 1s | 45s | **46s** | 66%↓ | 中 |
| P0 + P2 + P3 | 1s | 40-45s | **41-46s** | 66-70%↓ | 中 |

### Stream 模式（仅加载 backbone，expert 按需流式，~1750 个 tensor）

Stream 模式下 `loadWeightsSelective` 通过 `isExpertWeight` 跳过 expert 权重，
实际只加载 ~1750 个 backbone tensor（vs 非 stream 的 2223 个），加载基线更低。

| 方案组合 | 索引 | 加载 | 总计 | 提升 | 实现难度 |
|---------|------|------|------|------|---------|
| 当前 | 67s | 55s | **122s** | — | — |
| P0 二进制缓存 | 1s | 55s | **56s** | 54%↓ | 低 |
| P0 + P2 零拷贝 | 1s | 43-55s | **44-56s** | 54-60%↓ | 低-中 |
| P0 + P3 顺序加载 | 1s | 35s | **36s** | 70%↓ | 中 |
| P0 + P2 + P3 | 1s | 30-35s | **31-36s** | 70-75%↓ | 中 |

注: Stream 模式加载基线 ~55s（vs 非 stream 68.5s），因为跳过了 ~470 个 expert 权重。
原 P1 懒索引因 33/33 分片全部含 backbone 而弃用，索引阶段优化统一依赖 P0 缓存。

---

## 实施路线图

### Sprint 1（1-2 天）— 低风险高收益

1. **P0 二进制索引缓存**
   - 实现 `serializeIndex` / `deserializeIndex`
   - 实现 `loadOrBuildIndex` 入口函数
   - mtime + file_size 双重失效检测
   - 预期收益: 所有模式 TTFT -49%

2. **P4 合并 fd 复用**
   - 修改 `addShard` 保留 fd
   - `buildIndexFromDirectory` 返回 fd map
   - `FdPool` 接受外部 fd map
   - 预期收益: 代码简洁性（性能收益 <1ms，可忽略）

3. **P2 mmap 零拷贝** ← 提前到 Sprint 1（API 已确认可用）
   - 将 `mlx_array_new_data` 替换为 `mlx_array_new_data_managed_payload` + noop deleter
   - 仅修改 mmap 路径（3 处），pread fallback 保持不变
   - 确认 MmapPool 生命周期覆盖所有 array
   - **落地后先实测零拷贝命中率**（指针对齐问题，见 P2 详述）
   - 预期收益: 加载 -0~10s（取决于命中率，改动量极小无风险）

### Sprint 2（1-2 天）— 加载阶段深度优化

4. **P3 按分片顺序加载**
   - 重构 `loadWeightsSelective` 的迭代顺序
   - 按 shard 分组 + offset 排序
   - 预期收益: 加载 -20~30%

### Sprint 3（可选）— 首次运行优化

5. **P0.5 并行 header 解析**
   - 仅在无缓存时触发
   - 线程池并行解析 shard header
   - 预期收益: 首次运行 67s → ~10s

6. **P5 索引-加载流水线**
   - 生产者-消费者模式
   - 仅在无缓存的首次运行时有意义
   - 预期收益: 首次运行 -20~30%

---

## 与原方案的差异总结

| 变更 | 原方案 | 新方案 | 原因 |
|------|--------|--------|------|
| P1 Stream 懒索引 | 独立方案，排第二 | **弃用** | 2026-05-03 验证: 33/33 分片全含 backbone，0 个纯 expert 分片 |
| 并行 header 读取 | P1，独立方案 | 降级为 P0.5，仅首次运行 | P0 缓存后无增量收益 |
| 权重加载优化 | 无 | 新增 P2 零拷贝 + P3 顺序加载 | 65s 加载是最大瓶颈 |
| P2 零拷贝 | 需调研 API | API 已确认可用，提前到 Sprint 1 | `mlx_array_new_data_managed_payload` 已存在于 mlx-c |
| P2 收益预估 | 65s → 30-40s | 65s → 55-65s（不确定） | memcpy 仅占 3-5s，且指针对齐问题可能导致大部分回退到拷贝 |
| P4 fd 复用 | 收益 3-5s | 收益 <1ms | NVMe+APFS 上 open/close 约 10-50μs/次，价值在代码简洁性 |
| 失效检测 | 仅 mtime | mtime + file_size | NFS/APFS 兼容性 |
| 非 stream 目标 | 71s (48%↓) | 41-46s (66-70%↓) | 补充了加载阶段优化 |
| Stream 目标 | 2s (98%↓) | 31-36s (70-75%↓) | P1 弃用后 stream 也需加载全部 backbone |

---

## 验证方法

每个优化落地后需要实测验证，避免依赖预估数字。

**打点方案**: 在关键阶段前后插入 `std.time.nanoTimestamp()` 计时：

```zig
const t0 = std.time.nanoTimestamp();
var index = try loadOrBuildIndex(allocator, dir_path);
const t1 = std.time.nanoTimestamp();
try fd_pool.openAll(&index);
const t2 = std.time.nanoTimestamp();
try mmap_pool.mmapAll(&index);
const t3 = std.time.nanoTimestamp();
var weights = try loadWeightsSelective(allocator, ...);
const t4 = std.time.nanoTimestamp();

std.log.info("TTFT breakdown: index={d}ms fd={d}ms mmap={d}ms load={d}ms total={d}ms", .{
    @divFloor(t1 - t0, 1_000_000),
    @divFloor(t2 - t1, 1_000_000),
    @divFloor(t3 - t2, 1_000_000),
    @divFloor(t4 - t3, 1_000_000),
    @divFloor(t4 - t0, 1_000_000),
});
```

**P2 零拷贝验证**: 替换后对比 `mlx_array_data_*` 返回的指针与 mmap slice 指针：
- 相等 → 零拷贝路径
- 不等 → 回退到拷贝（统计回退比例，确认是否符合预期）

**P3 顺序加载验证**: 对比优化前后的 page fault 数量：
```bash
# macOS: 用 /usr/bin/time -l 查看 page faults
/usr/bin/time -l ./zig-out/bin/mlx-zig --model ... 2>&1 | grep "page faults"
```

**回归测试**: 每次优化后运行 `bash scripts/best_test.sh`，确保 7/7 prompt 全部通过。

---

## 附录: mlx-c 零拷贝 API 调研结论

**结论**: mlx-c 已原生支持零拷贝构造，不需要修改 mlx-c 代码。

**调用链路**:
```
mlx_array_new_data_managed_payload (mlx-c/mlx/c/array.cpp)
  → mlx::core::array(void* data, Shape, Dtype, deleter)  (mlx/mlx/array.cpp)
    → allocator::make_buffer(data, nbytes)                (mlx/mlx/allocator.h)
      → MetalAllocator::make_buffer(ptr, size)            (mlx/mlx/backend/metal/allocator.cpp)
        → MTL::Device::newBuffer(ptr, size, StorageModeShared, nullptr)
          成功 → 零拷贝: 外部指针直接注册为 Metal GPU buffer
          失败 → 回退: malloc + memcpy（与 mlx_array_new_data 相同）
```

**Metal newBuffer 零拷贝条件**:
- 指针页对齐（mmap 基址满足，但 `base + tensor_offset` 通常不满足 ⚠️）
- size 为页大小倍数（大 tensor 通常满足，小 tensor 自动回退到拷贝）
- `ResourceStorageModeShared`（Apple Silicon 统一内存架构，CPU/GPU 共享地址空间）

**关键风险**: safetensors 文件中 tensor 按顺序紧密排列，单个 tensor 的 `data_offset_start`
取决于前面所有 tensor 的累计大小，不保证页对齐（Apple Silicon 页大小 = 16KB）。
因此 `mmap_pool.getSlice()` 返回的指针大概率不是页对齐的，`newBuffer` 会返回 nullptr 回退到拷贝。
实际命中率需要实测确认。

**验证方法**: 替换后对比 `mlx_array_data_*` 返回的指针是否等于 mmap slice 指针。
如果相等，说明走了零拷贝路径；如果不等，说明回退到了拷贝。
