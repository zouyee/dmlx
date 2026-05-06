# Chapter 2: Core Infrastructure Deep Analysis

## 2.1 `c.zig`: Error Handling Evolution

### Fixed: v0.3.0 → Current

The v0.3.0 audit identified error handling as a P0 issue: `check(rc)` only returned `error.MlxError` with no context.

Now fixed:

```zig
pub fn check(rc: c_int) !void {
    if (rc != 0) {
        const msg = mlx_get_last_error();
        std.log.err("MLX error: {s}", .{std.mem.span(msg)});
        return error.MlxError;
    }
}
```

### Still Unresolved

The error union type remains singular (`error.MlxError`), not subdivided by mlx-c error codes into:
- `MlxOOM` (out of memory)
- `MlxInvalidArg` (invalid argument)
- `MlxDeviceError` (device error)

## 2.2 `array.zig`: API Design Contradictions

### Misleading `allocator` Parameter (P2)

`fromData`, `zeros`, `ones` accept an `allocator` parameter but completely ignore it:

```zig
pub fn fromData(allocator: std.mem.Allocator, comptime T: type, ...) !Array {
    _ = allocator;  // completely unused
    // ...
}
```

Since mlx-c manages memory internally, this misleads users into thinking the Zig allocator manages Array memory.

### `strides()` 64-bit Assumption (P2)

```zig
pub fn strides(self: Array) []i64 {
    // assumes size_t == i64 (64-bit platforms)
}
```

On 32-bit platforms this would cause truncation.

## 2.3 `ops.zig`: API Redundancy

`ops.zig` has functional duplication with `ops/shape.zig`, `ops/math.zig`:

| Operation | ops.zig Location | ops/ Submodule Location |
|------|-------------|----------------|
| reshape | `ops.zig` | `ops/shape.zig` |
| transpose | `ops.zig` | `ops/shape.zig` |
| softmax | `ops.zig` | `ops/math.zig` |
| relu | `ops.zig` | `ops/activations.zig` |

Recommendation: `ops.zig` should only retain `EagerContext` and the most core binary/unary operators, delegating the rest to submodules.

## 2.4 `EagerContext`: Stream Lifecycle Defect

```zig
pub fn init(allocator: std.mem.Allocator) EagerContext {
    return .{
        .allocator = allocator,
        .stream = .{ .inner = c.c.mlx_default_cpu_stream_new() },
    };
}
```

**Problem**: Each call to `init` creates a new mlx_stream, but `EagerContext` still lacks a `deinit` method to release it.

**Impact**: Frequent creation of `EagerContext` causes stream resource leaks.

**Recommendation**: Add a `deinit` method, or switch to obtaining a reference to the global default stream (rather than creating a new instance).

## 2.5 Build System (`build.zig`)

### Four-Level mlx-c Detection (Fixed P1 Issue)

Priority: `-Dmlx_prefix` > `MLX_C_PREFIX` env > `pkg-config --variable=prefix mlxc` > `/opt/homebrew` fallback

```zig
const mlx_prefix = blk: {
    if (b.option([]const u8, "mlx_prefix", ...)) |p| break :blk p;
    if (b.graph.environ_map.get("MLX_C_PREFIX")) |p| break :blk p;
    if (pkgConfigMlxPrefix(b)) |p| break :blk p;
    break :blk "/opt/homebrew";
};
```

### Build Artifacts

- `libdmlx.a`: static library
- `dmlx`: CLI tool
- `example`: example program
- `test`: test runner

### Platform Framework Linking

The build system detects the target platform and links Accelerate, Metal, and Foundation frameworks on macOS to enable GPU acceleration.
