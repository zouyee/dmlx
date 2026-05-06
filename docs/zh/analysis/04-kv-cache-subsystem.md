# 第四章 KV Cache 子系统

## 4.1 策略接口设计（`kvcache/interface.zig`）

VTable 运行时多态设计：

```zig
pub const VTable = struct {
    updateAndFetch: *const fn (ctx: *anyopaque, keys: Array, values: Array, stream: mlx_stream) anyerror!KVSlice,
    currentLen: *const fn (ctx: *anyopaque) usize,
    reset: *const fn (ctx: *anyopaque) void,
    filter: ?*const fn (ctx: *anyopaque, indices: []const usize, allocator: Allocator) anyerror!void,
    rollback: ?*const fn (ctx: *anyopaque, to_len: usize) void,
    deinit: *const fn (ctx: *anyopaque, allocator: Allocator) void,
};
```

设计亮点：运行时策略切换 + comptime 内部特化 + 注意力层完全解耦。

## 4.2 六级策略对比

| 策略 | 特点 | 适用场景 | 代码位置 |
|------|------|---------|---------|
| Standard | 简单连续缓冲区 | 单请求、短序列 | `kvcache/standard.zig` |
| Rotating | 环形缓冲区，固定窗口 | 超长序列（避免 OOM） | `kvcache/rotating.zig` |
| Quantized | 4/8/16 bit KV 压缩 | 内存受限 | `kvcache/quantized.zig` |
| Paged | 32-token 页 + 页表 + CoW | 连续批处理（**默认**） | `kvcache/paged.zig` |
| PagedQuantized | Paged + Quantized 组合 | 极致内存优化 | `kvcache/paged.zig` |
| Tiered | RAM hot + SSD cold + LRU | 超长上下文 + 多模型 | `kvcache/tiered.zig` |

## 4.3 PagedKVCache（`kvcache/paged.zig`，1,152 行）

### 核心设计

- **页大小**：默认 32 tokens（针对 Apple Silicon Metal GPU 内存对齐调优）
- **BlockManager**：管理 free pool、per-request 块映射、CoW 机制
- **前缀哈希**：`hashBlock` 使用 Wyhash 计算滚动哈希
- **Copy-on-Write**：共享块 `ref_count > 1` 时分配新块并拷贝数据

### updateAndFetch 算法流程

1. **分配页**：`new_total = cached_len + seq_len`，按需分配新页
2. **写入 KV**：通过 `mlx_slice_update` 将 keys/values 写入对应页
3. **注册哈希**：页写满时计算哈希，注册到 `page_hashes` 映射
4. **Gather 输出**：将分散的页拼接为连续的 `[batch, heads, seq, dim]` 数组
5. **量化路径**： quantized 页需先 dequantize 再 concatenate

## 4.4 TieredKVCache（`kvcache/tiered.zig`）

- 包装 `PagedKVCache` 作为 hot tier
- 超出 `hot_capacity` 时 LRU 页写入 SSD：`{cold_dir}/block_{id}.safetensors`
- `restoreFromSSD`：从 safetensors 文件恢复块到 hot tier

## 4.5 Prompt Cache（`prompt_cache.zig`，563 行）

支持 save/load KV cache 状态到 safetensors 文件。

### 🔴 高危漏洞

```zig
// prompt_cache.zig:74
const std_cache: *StandardKVCache = @ptrCast(@alignCast(cache.ptr));
```

**问题**：`savePromptCache` 接收 `[]KVCacheStrategy`（运行时多态），但直接将 `cache.ptr` 强制转换为 `*StandardKVCache`。

**后果**：
- `PagedKVCache` 的 `ptr` 指向 `PagedKVCache` 结构体，字段布局完全不同
- 访问 `.offset` 时读取的是 `pages` 指针的一部分——数值无意义
- 访问 `.keys`/`.values` 时可能读取到 `PageTableEntry` 数组的指针，导致 segfault

**触发条件**：使用默认配置 `--kv-strategy paged_quantized` + `--prompt-cache-file` 时必现。

**修复建议**：
1. 在 `KVCacheStrategy.VTable` 中增加 `saveState`/`loadState` 方法
2. 或添加运行时类型检查：`std.debug.assert(cache.vtable == &StandardKVCache.vtable)`
