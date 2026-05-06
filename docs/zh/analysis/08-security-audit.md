# 第八章 安全边界与代码审查

## 8.1 `@constCast` 全库统计

全库共 **10 处** `@constCast` 调用：

| 位置 | 行号 | 用途 | 风险 |
|------|------|------|------|
| `array.zig` | 150 | `dataSliceMut`：将 const 指针转为可变 | **高** |
| `tree.zig` | 302 | `treeMapInPlace`：递归遍历字段指针 | 中 |
| `tree.zig` | 317 | `treeToArrayPtrs`：收集 Array 指针 | 低 |
| `guided.zig` | 85 | `FiniteStateMachine.deinit`：`[]State` → `*State` | 低 |
| `safetensors_reader.zig` | 494 | `mmap` 区域指针转换 | 中 |
| `safetensors_reader.zig` | 520 | `munmap` 解除 const 限制 | 低 |
| `minimax.zig` | 59-60 | RoPE sin/cos cache 初始化 | **高** |
| `deepseek_v4.zig` | 198-199 | YARN RoPE sin/cos cache 初始化 | **高** |
| `deepseek_v4.zig` | 399 | Attention mask 初始化 | **高** |

### `dataSliceMut` 泛滥

`array.zig:148`：
```zig
pub fn dataSliceMut(self: Array, comptime T: type) ![]T {
    const ptr = try self.dataPtr(T);
    return @constCast(ptr)[0..self.size()];
}
```

调用统计：
- `ops/nn.zig`：**34 处**
- `models/minimax.zig`：**4 处**
- 合计：**38 处**

**项目声称已修复**（`ROADMAP.md`）："安全：`@constCast` 绕过 CoW → 全部改用 mlx-c 算子链 ✅"

**实际状态**：修复未完全完成。`nn.zig` 中 BatchNorm、LSTM、GRU、RNN、MultiHeadAttention、RoPE、Embedding 仍通过 `dataSliceMut` 使用纯 CPU 标量循环。

## 8.2 `prompt_cache.zig` 类型安全漏洞（P0）

```zig
// prompt_cache.zig:74
const std_cache: *StandardKVCache = @ptrCast(@alignCast(cache.ptr));
```

**问题**：接收 `[]KVCacheStrategy`（运行时多态），但强制转换为 `*StandardKVCache`。

**PagedKVCache 与 StandardKVCache 布局差异**：

```zig
// StandardKVCache
pub const StandardKVCache = struct {
    keys: Array,
    values: Array,
    offset: usize,
};

// PagedKVCache
pub const PagedKVCache = struct {
    pages: []Page,
    sequences: []SequenceState,
    page_size: usize,
    page_hashes: std.HashMap(...),
    // ...
};
```

当 `cache.ptr` 实际指向 `PagedKVCache` 时：
- `std_cache.offset` 读取的是 `PagedKVCache.pages` 指针的低 64 位——无意义
- `std_cache.keys` 读取的是 `PagedKVCache.sequences` 指针——后续 `sliceCache` 操作 segfault

**触发条件**：默认配置 `--kv-strategy paged_quantized` + `--prompt-cache-file` 时必现。

## 8.3 `distributed.zig` 资源泄漏

```zig
pub fn deinit(self: *DistributedGroup) void {
    _ = self;
    // mlx_distributed_group has no explicit free in this mlx-c version
}
```

频繁创建/销毁会产生资源泄漏，长期运行服务中风险累积。

## 8.4 `model_pool.zig` VTable 可选类型

```zig
pub const LoadedModel = struct {
    vtable: ?ModelVTable,  // 可选！
    // ...
};
```

`getOrLoad` 加载后 `vtable` 设为 null，`deinit` 中仅当 `vtable != null` 时释放资源——如果始终为 null，模型资源泄漏。
