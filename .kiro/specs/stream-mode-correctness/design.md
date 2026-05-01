# Design: Stream Mode Expert Loading Correctness

## Problem Analysis

### Symptom
Stream mode generates incorrect output despite:
- ✅ No crashes or segfaults
- ✅ No NaN/Inf in logits
- ✅ Correct expert weight shapes loaded
- ✅ Correct remap array construction
- ❌ **Output is semantically wrong** (Korean text for "What is 2+2?")

### Evidence from Debug Logs

**Test case (second token, Layer 0):**
```
Router selected: [87, 0, 31, 0, 22, 0]
Unique experts (sorted): [0, 22, 31, 87]
Remap construction:
  remap[0] = 0
  remap[22] = 1
  remap[31] = 2
  remap[87] = 3

Expected remapped: [3, 0, 2, 0, 1, 0]
Actual remapped:   [3, 2, 1, 0, 0, 0]

Manual verification:
  indices[0]=87 → remap[87]=3, got 3 ✓
  indices[1]=0  → remap[0]=0,  got 2 ✗
  indices[2]=31 → remap[31]=2, got 1 ✗
  indices[3]=0  → remap[0]=0,  got 0 ✓
  indices[4]=22 → remap[22]=1, got 0 ✗
  indices[5]=0  → remap[0]=0,  got 0 ✓
```

### Root Cause Hypotheses

#### H1: dataSlice() Memory Layout Mismatch ⭐ **MOST LIKELY**
**Theory**: `indices` is a 2D array `[1, 6]`. When we call `dataSlice(u32)`, it reads memory in a specific order (row-major). But `mlx_take(remap, indices)` might process the array in a different order or preserve a different memory layout.

**Evidence**:
- Alternating pattern in verification: indices[0,3,5] correct, indices[1,2,4] wrong
- This suggests every other element is being read from the wrong position
- MLX arrays can have non-contiguous strides

**Test**: Print the raw memory layout of both `indices` and `remapped` arrays, check strides.

#### H2: mlx_take() Axis Confusion
**Theory**: `mlx_take(remap, indices)` without an axis parameter might be treating the 2D `indices` array differently than expected.

**Evidence**:
- `mlx_take` signature: `mlx_take(output, array, indices, stream)`
- No axis parameter means it operates on flattened array
- But `indices` is 2D `[1, 6]`

**Test**: Compare with `mlx_take_axis(remap, indices, 0)` or flatten indices first.

#### H3: Remap Array Indexing Issue
**Theory**: The remap array construction is correct, but when MLX indexes it with a 2D indices array, something goes wrong.

**Evidence**:
- Remap values are correct when checked individually
- But the take operation produces wrong results

**Test**: Create a minimal test case with small arrays to isolate the issue.

#### H4: Indices Array Corruption
**Theory**: The `indices` array passed to `streamingForward` is already corrupted or has unexpected values.

**Evidence**:
- `dataSlice` shows [87, 0, 31, 0, 22, 0]
- But maybe the actual array content is different?

**Test**: Use MLX's item() API to read individual elements instead of dataSlice.

## Diagnostic Plan

### Phase 1: Minimal Reproduction (Priority: CRITICAL)

Create a standalone test that isolates the remap operation:

```zig
test "mlx_take remap behavior" {
    const allocator = std.testing.allocator;
    const ctx = EagerContext.init(allocator);
    
    // Simulate the exact scenario from logs
    const remap_data = [_]i32{ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                               -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                               -1, -1, 1, -1, -1, -1, -1, -1, -1, -1,
                               -1, 2, -1, -1, -1, -1, -1, -1, -1, -1,
                               // ... fill to 256 ...
                               -1, -1, -1, -1, -1, -1, -1, 3 }; // remap[87]=3
    
    const remap = try Array.fromData(allocator, i32, &remap_data, &[_]i32{256});
    defer remap.deinit();
    
    // Indices as 2D array [1, 6]
    const indices_data = [_]u32{ 87, 0, 31, 0, 22, 0 };
    const indices = try Array.fromData(allocator, u32, &indices_data, &[_]i32{1, 6});
    defer indices.deinit();
    
    // Perform take
    var remapped_raw = c.c.mlx_array_new();
    try c.check(c.c.mlx_take(&remapped_raw, remap.inner, indices.inner, ctx.stream.inner));
    const remapped = Array.fromHandle(remapped_raw);
    defer remapped.deinit();
    try remapped.eval();
    
    // Check results
    const result = try remapped.dataSlice(i32);
    try std.testing.expectEqual(@as(i32, 3), result[0]); // remap[87]
    try std.testing.expectEqual(@as(i32, 0), result[1]); // remap[0]
    try std.testing.expectEqual(@as(i32, 2), result[2]); // remap[31]
    try std.testing.expectEqual(@as(i32, 0), result[3]); // remap[0]
    try std.testing.expectEqual(@as(i32, 1), result[4]); // remap[22]
    try std.testing.expectEqual(@as(i32, 0), result[5]); // remap[0]
}
```

**Expected outcome**: This test will either pass (proving the issue is elsewhere) or fail (confirming the mlx_take behavior issue).

### Phase 2: Array Layout Investigation (Priority: HIGH)

Add diagnostic logging to understand array memory layout:

```zig
// In streamingForward, after building remap and indices:
std.log.info("indices: shape={any}, strides={any}, size={d}", 
    .{ indices.shape(), indices.strides(), indices.size() });
std.log.info("remap: shape={any}, strides={any}, size={d}", 
    .{ remap_arr.shape(), remap_arr.strides(), remap_arr.size() });

// After mlx_take:
std.log.info("remapped: shape={any}, strides={any}, size={d}", 
    .{ remapped.shape(), remapped.strides(), remapped.size() });

// Read using item() instead of dataSlice:
for (0..6) |i| {
    const idx_val = try indices.item(u32, &[_]usize{0, i});
    const remap_val = try remapped.item(i32, &[_]usize{0, i});
    std.log.info("  [0,{d}]: indices={d}, remapped={d}, expected={d}", 
        .{ i, idx_val, remap_val, remap_data[idx_val] });
}
```

### Phase 3: Flatten-First Approach (Priority: MEDIUM)

Test if flattening indices before remapping fixes the issue:

```zig
// Instead of:
// const remapped = mlx_take(remap_arr, indices)

// Try:
const indices_flat = try ops.reshape(ctx, indices, &[_]i32{@intCast(indices.size())});
defer indices_flat.deinit();
const remapped_flat = mlx_take(remap_arr, indices_flat);
defer remapped_flat.deinit();
const remapped = try ops.reshape(ctx, remapped_flat, indices.shape());
```

### Phase 4: Compare with Preload Mode (Priority: LOW)

Verify that preload mode uses the exact same remap logic and it works:

```zig
// In expert_preload.zig, add same diagnostic logging
// Compare the array shapes, strides, and results
```

### Phase 5: Python MLX Reference Test (Priority: LOW)

Create a Python script to test the exact same operation:

```python
import mlx.core as mx

remap = mx.zeros(256, dtype=mx.int32)
remap[0] = 0
remap[22] = 1
remap[31] = 2
remap[87] = 3

indices = mx.array([[87, 0, 31, 0, 22, 0]], dtype=mx.uint32)

remapped = remap[indices]
print(f"remapped: {remapped}")
# Expected: [[3, 0, 2, 0, 1, 0]]
```

## Proposed Solutions

### Solution A: Use Flattened Indices (RECOMMENDED)

**Rationale**: Avoid any potential 2D array layout issues by working with 1D arrays throughout.

**Implementation**:
```zig
// 1. Flatten indices immediately after receiving them
const indices_flat = try ops.reshape(self.ctx, indices, &[_]i32{@intCast(indices.size())});
defer indices_flat.deinit();

// 2. Build remap as before (1D array)
const remap_arr = try Array.fromData(...);
defer remap_arr.deinit();

// 3. Remap using flattened indices
var remapped_raw = c.c.mlx_array_new();
try c.check(c.c.mlx_take(&remapped_raw, remap_arr.inner, indices_flat.inner, self.ctx.stream.inner));
const remapped_flat = Array.fromHandle(remapped_raw);
defer remapped_flat.deinit();

// 4. Reshape back to original shape if needed (but we flatten again later anyway)
const remapped_u32 = try ops.astype(self.ctx, remapped_flat, .uint32);
defer remapped_u32.deinit();

// Continue with existing code...
```

**Pros**:
- Avoids any 2D array layout ambiguity
- Simpler to reason about
- Matches the pattern used later in the code (we flatten for gatherQmm anyway)

**Cons**:
- Slightly more verbose
- Doesn't fix the root cause if it's a deeper MLX issue

### Solution B: Use mlx_take_axis

**Rationale**: Explicitly specify axis=0 to ensure consistent behavior.

**Implementation**:
```zig
// Instead of mlx_take, use mlx_take_axis with axis=-1 (last axis)
try c.check(c.c.mlx_take_axis(&remapped_raw, remap_arr.inner, indices.inner, -1, self.ctx.stream.inner));
```

**Pros**:
- More explicit about which axis to index
- Might handle 2D arrays better

**Cons**:
- Unclear if this is the right axis
- Might not solve the underlying issue

### Solution C: Use item() for Verification

**Rationale**: If dataSlice() has layout issues, use MLX's item() API for element access.

**Implementation**:
```zig
// For debugging only - too slow for production
for (0..indices.size()) |i| {
    const idx = try indices.item(u32, &[_]usize{i / topk, i % topk});
    const expected = remap_data[idx];
    const actual = try remapped.item(i32, &[_]usize{i / topk, i % topk});
    if (expected != actual) {
        std.log.err("Mismatch at [{d},{d}]: indices={d}, expected={d}, got={d}", 
            .{ i / topk, i % topk, idx, expected, actual });
    }
}
```

**Pros**:
- Bypasses any dataSlice layout issues
- Provides definitive verification

**Cons**:
- Too slow for production use
- Only useful for debugging

## Implementation Strategy

### Step 1: Run Minimal Reproduction Test
- Implement the test from Phase 1
- If it fails, we've confirmed the mlx_take issue
- If it passes, the issue is elsewhere (indices corruption, etc.)

### Step 2: Add Diagnostic Logging
- Implement Phase 2 logging
- Run with `--max-tokens 1` to minimize output
- Analyze strides and memory layout

### Step 3: Implement Solution A (Flatten-First)
- Modify `streamingForward` to flatten indices before remapping
- Test with multiple prompts
- Verify output correctness

### Step 4: Verify with Preload Mode Comparison
- If available, run same prompt with preload mode (small expert count)
- Compare token-by-token output
- Ensure cosine similarity > 0.9999

### Step 5: Clean Up and Document
- Remove debug logging
- Add inline comments explaining the flatten-first approach
- Update STREAM_MODE_STATUS.md with findings

## Correctness Properties

### P1: Remap Correctness
**Property**: For all expert IDs `e` in `unique_ids`, `remap[e]` equals the index of `e` in `unique_ids`.

**Test**:
```zig
for (unique_ids, 0..) |eid, i| {
    try std.testing.expectEqual(@as(i32, @intCast(i)), remap_data[eid]);
}
```

### P2: Take Operation Correctness
**Property**: For all positions `i` in `indices`, `remapped[i]` equals `remap[indices[i]]`.

**Test**:
```zig
for (0..indices.size()) |i| {
    const idx = indices_data[i];
    const expected = remap_data[idx];
    const actual = remapped_data[i];
    try std.testing.expectEqual(expected, actual);
}
```

### P3: Expert Weight Selection Correctness
**Property**: When gatherQmm is called with `remapped[i]`, it should use the expert weights corresponding to the original `indices[i]`.

**Test**: Compare stream mode output with preload mode output (token-level).

### P4: Output Semantic Correctness
**Property**: For a given prompt, the generated output should be semantically relevant.

**Test**: Manual verification of 10 test prompts (math, factual, creative).

## Testing Plan

### Unit Tests

```zig
// tests/expert_remap_test.zig
test "remap construction" { ... }
test "mlx_take with 1D indices" { ... }
test "mlx_take with 2D indices" { ... }
test "flatten-remap-reshape round-trip" { ... }
```

### Integration Tests

```zig
// tests/stream_mode_test.zig
test "stream mode single token generation" { ... }
test "stream mode vs preload mode equivalence" { ... }
test "stream mode semantic correctness" { ... }
```

### Manual Test Cases

1. "What is 2+2?" → Should contain "4"
2. "The capital of France is" → Should contain "Paris"
3. "Write a haiku about" → Should be poetic
4. "Translate to Spanish: Hello" → Should contain "Hola"
5. "Explain quantum computing" → Should be technical

## Success Criteria

1. ✅ Minimal reproduction test passes
2. ✅ All unit tests pass
3. ✅ Stream mode generates correct output for 10/10 test prompts
4. ✅ Stream mode output matches preload mode (when testable)
5. ✅ No performance regression (generation time within 10% of baseline)

## References

- Python vmlx: `vmlx/vmlx_engine/utils/smelt_loader.py` (TurboRouteWrapper.forward)
- Preload mode: `mlx-zig/src/models/expert_preload.zig`
- MLX C API: `mlx_take`, `mlx_take_axis` documentation
- Context transfer: Previous debugging session logs
