# 第二章 核心基础设施深度分析

## 2.1 `c.zig`：错误处理的演进

### 已修复：v0.3.0 → 当前

v0.3.0 审计发现错误处理为 P0 问题：`check(rc)` 仅返回 `error.MlxError`，无上下文。

当前已修复：

```zig
pub fn check(rc: c_int) !void {
    if (rc != 0) {
        const msg = mlx_get_last_error();
        std.log.err("MLX error: {s}", .{std.mem.span(msg)});
        return error.MlxError;
    }
}
```

### 仍未修复

错误联合类型仍单一（`error.MlxError`），未根据 mlx-c 错误码细分为：
- `MlxOOM`（内存不足）
- `MlxInvalidArg`（非法参数）
- `MlxDeviceError`（设备错误）

## 2.2 `array.zig`：API 设计矛盾

### `allocator` 参数误导（P2）

`fromData`、`zeros`、`ones` 接受 `allocator` 参数但完全忽略：

```zig
pub fn fromData(allocator: std.mem.Allocator, comptime T: type, ...) !Array {
    _ = allocator;  // 完全没用
    // ...
}
```

因为 mlx-c 内部管理内存，这误导用户以为 Zig allocator 在管理 Array 内存。

### `strides()` 64 位假设（P2）

```zig
pub fn strides(self: Array) []i64 {
    // 假设 size_t == i64（64位平台）
}
```

在 32 位平台上会截断。

## 2.3 `ops.zig`：API 冗余

`ops.zig` 与 `ops/shape.zig`、`ops/math.zig` 存在功能重复：

| 操作 | ops.zig 位置 | ops/ 子模块位置 |
|------|-------------|----------------|
| reshape | `ops.zig` | `ops/shape.zig` |
| transpose | `ops.zig` | `ops/shape.zig` |
| softmax | `ops.zig` | `ops/math.zig` |
| relu | `ops.zig` | `ops/activations.zig` |

建议：`ops.zig` 只保留 EagerContext 和最核心的二元/一元算子，其余委托给子模块。

## 2.4 `EagerContext`：Stream 生命周期缺陷

```zig
pub fn init(allocator: std.mem.Allocator) EagerContext {
    return .{
        .allocator = allocator,
        .stream = .{ .inner = c.c.mlx_default_cpu_stream_new() },
    };
}
```

**问题**：每次调用 `init` 都会创建新的 mlx_stream，但 `EagerContext` 仍无 `deinit` 方法释放它。

**影响**：频繁创建 `EagerContext` 会产生 stream 资源泄漏。

**建议**：添加 `deinit` 方法，或改为获取全局默认 stream 的引用（非创建新实例）。

## 2.5 构建系统（`build.zig`）

### 四级 mlx-c 探测（已修复 P1 问题）

优先级：`-Dmlx_prefix` > `MLX_C_PREFIX` env > `pkg-config --variable=prefix mlxc` > `/opt/homebrew` fallback

```zig
const mlx_prefix = blk: {
    if (b.option([]const u8, "mlx_prefix", ...)) |p| break :blk p;
    if (b.graph.environ_map.get("MLX_C_PREFIX")) |p| break :blk p;
    if (pkgConfigMlxPrefix(b)) |p| break :blk p;
    break :blk "/opt/homebrew";
};
```

### 构建产物

- `libmlx-zig.a`：静态库
- `mlx-zig`：CLI 工具
- `example`：示例程序
- `test`：测试 runner

### 平台框架链接

构建系统检测目标平台，在 macOS 下链接 Accelerate、Metal、Foundation 框架以启用 GPU 加速。
