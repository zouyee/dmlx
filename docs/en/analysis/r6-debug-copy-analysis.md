# R6 深度分析：Expert Stream DEBUG 拷贝

> **日期**: 2026-05-08  
> **关联 Spec**: `.kiro/specs/perf-recovery-plan/` (R6)  
> **文件**: `src/models/expert_stream.zig:283-289`  
> **结论**: 可安全移除，预期提升 2-3 tok/s

---

## 数据流全链路追踪

```
loadExpertSlicesCached()
  └─ loadExpertSlices()
       └─ PartialTensorReader.readExpertRows()    ← 创建张量
            └─ mlx_array_new_data(buf.ptr, ...)   ← 内部拷贝
       └─ ops.copy() + eval()                     ← DEBUG 强制拷贝 + GPU 同步
  └─ 返回给 streamingForward()
       └─ DSV4SwitchGLU.forwardNoScores()
            └─ dispatchGatherMm()
                 └─ gatherQmm() / gatherMm()     ← 消费张量
```

---

## 关键发现 1：`readExpertRows` 已保证连续性

`mlx-zig/src/io/safetensors_reader.zig:1040-1100` 中的 `readExpertRows`（复数形式）：

```zig
// 分配单个连续 buffer
const total_bytes = expert_ids.len * row_bytes;
const buf = try self.allocator.alloc(u8, total_bytes);

// 逐行 memcpy 到连续 buffer
for (expert_ids, 0..) |eid, i| {
    const range = self.computeExpertByteRange(&info, eid);
    const slice = try pool.getSlice(info.shard_path, range.offset, range.length);
    @memcpy(dest, slice);
}

// mlx_array_new_data 再次拷贝到 MLX 内部管理的内存
const arr = c.c.mlx_array_new_data(buf.ptr, shape_i32.ptr, ...);
```

**C 头文件明确声明**（`mlx-c/mlx/c/array.h:120`）：
> `@param data A buffer which will be **copied**.`

所以 `readExpertRows` 返回的张量：
- ✅ 数据在 MLX 内部管理的内存中（已拷贝）
- ✅ shape 为 `[n_selected, D1, D2, ...]`
- ✅ stride 为标准连续布局（从 shape 推导的 row-major strides）

---

## 关键发现 2：`readExpertRow`（单数）才有零拷贝风险

注意区分两个函数：

| 函数 | API | 内存语义 | stride 保证 |
|------|-----|----------|-------------|
| `readExpertRow` (单数) | `mlx_array_new_data_managed_payload` | **零拷贝** — 直接指向 mmap 区域 | ⚠️ 取决于 mmap 对齐 |
| `readExpertRows` (复数) | `mlx_array_new_data` | **拷贝** — MLX 内部分配 | ✅ 标准连续 |

当前 `loadExpertSlices` 使用的是 `readExpertRows`（复数），所以**不存在零拷贝的 stride 风险**。

---

## 关键发现 3：DEBUG 注释的历史原因

注释写道：
> "The partial-read path (mmap/pread + mlx_array_new_data) **may** create tensors with non-standard strides"

这个注释是**不准确的**。`mlx_array_new_data` 从连续 buffer 创建张量，结果必然是标准连续 stride。这个 DEBUG 块可能是在开发早期、`readExpertRows` 实现尚未稳定时添加的防御性代码。

---

## 关键发现 4：`gatherQmm` 的 stride 要求

`gatherQmm` 调用 MLX C API `mlx_gather_qmm`，它期望权重张量 `w` 为 `[n_experts, out, in]` 格式。关键路径：

```zig
// expert_stream.zig: streamingForward()
var switch_glu = DSV4SwitchGLU{
    .gate_proj = gate_w,   // ← 来自 loadExpertSlices
    .up_proj = up_w,
    .down_proj = down_w,
    ...
};

// deepseek_v4.zig: forwardNoScores()
const gate_proj_t = try ops.transposeAxes(self.ctx, self.gate_proj, &[_]i32{0, 2, 1});
// ...
return quantize_mod.gatherQmm(self.ctx, x, weight, scales, biases, null, indices_arr, true, ...);
```

`gatherQmm` 接收的 `weight` 参数是 `self.gate_proj`（未转置的原始权重），`transpose_w=true` 告诉 MLX 内部处理转置。MLX 的 `gather_qmm` 内核对输入张量的 stride 要求是**标准连续**（因为量化格式的 bit-packing 依赖连续内存布局）。

---

## 关键发现 5：MLX `mlx_array_new_data` 拷贝语义确认

从 MLX C++ 源码 (`mlx/mlx/array.h:68-76`)：

```cpp
/* Build an array from a raw pointer. The constructor will attempt to use the
 * input data without a copy. The deleter will be called when the array no
 * longer needs the underlying memory */
explicit array(void* data, Shape shape, Dtype dtype,
               const std::function<void(void*)>& deleter);
```

但 `mlx_array_new_data`（无 managed）走的是 `mlx_array_set_data` 路径，它将 `const void*` 转为类型指针后调用迭代器构造函数 — 这个路径**拷贝数据**。

对比：
- `mlx_array_new_data` → 拷贝语义（`const void*`，无 deleter）
- `mlx_array_new_data_managed_payload` → 零拷贝语义（`void*`，有 deleter）

---

## 性能影响量化

```
每次 loadExpertSlices 调用:
  ops.copy()  → GPU 内存分配 + 数据拷贝 (带宽消耗)
  eval()      → CPU 等待 GPU 完成 (pipeline stall)

每个 token 的 streamingForward:
  6 次 loadExpertSlicesCached (gate_w, up_w, down_w × weight + scales)
  × cache miss 率 ~60%
  = ~3.6 次 loadExpertSlices 调用/token
  × 每次 ~6ms (copy + eval overhead)
  ≈ 21.6ms/token 额外开销
```

在 85ms/token 的目标下，这是 **25% 的额外延迟**。

---

## 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| `gatherQmm` stride 不兼容 | **极低** | 推理错误 | `readExpertRows` 已保证连续 |
| mxfp4 packing 格式问题 | **极低** | 量化解码错误 | `mlx_array_new_data` 已拷贝到标准布局 |
| MLX lazy eval 语义变化 | **低** | 内存峰值增加 | 移除 eval 后张量延迟物化，可能增加 peak memory |

---

## 结论与建议

**可以安全移除整个 DEBUG 块**，理由：

1. `readExpertRows` 使用 `mlx_array_new_data`（拷贝语义），返回的张量已经是标准连续布局
2. `gatherQmm` 的 stride 要求被 `mlx_array_new_data` 的拷贝语义天然满足
3. 注释中的担忧（"may create tensors with non-standard strides"）对 `readExpertRows` 不成立

---

## 推荐变更

```zig
// 之前 (DEBUG):
if (self.partial_reader) |reader| {
    const arr = try reader.readExpertRows(tensor_name, expert_ids);
    // DEBUG: Force copy to ensure contiguous strides.
    const copied = try ops.copy(self.ctx, arr);
    arr.deinit();
    try copied.eval();
    return copied;
}

// 之后:
if (self.partial_reader) |reader| {
    return try reader.readExpertRows(tensor_name, expert_ids);
}
```

---

## 备选方案（如果正确性测试失败）

```zig
// 保留 copy 但移除 eval（延迟拷贝，不阻塞 GPU pipeline）
if (self.partial_reader) |reader| {
    const arr = try reader.readExpertRows(tensor_name, expert_ids);
    const copied = try ops.copy(self.ctx, arr);
    arr.deinit();
    return copied;  // 不 eval，让 MLX lazy 调度
}
```

---

## 验证步骤

1. 移除 DEBUG 块
2. `zig build` 编译通过
3. `scripts/best_test.sh` 7/7 PASS
4. 测量 tok/s — 预期提升 2-3 tok/s（从 ~10 到 ~12）
