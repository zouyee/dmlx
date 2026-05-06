"""
Python MLX reference test for remap/indices behavior.

This test verifies the expected behavior of MLX's array indexing (take)
with the exact remap/indices scenario from the stream-mode-correctness spec.

See: .kiro/specs/stream-mode-correctness/design.md

Usage: python tests/test_mlx_take.py
Requires: pip install mlx
"""
import mlx.core as mx

remap = mx.zeros(256, dtype=mx.int32)
remap[0] = 0
remap[22] = 1
remap[31] = 2
remap[87] = 3

# Test with 2D indices [1, 6]
indices_2d = mx.array([[87, 0, 31, 0, 22, 0]], dtype=mx.uint32)
remapped_2d = remap[indices_2d]
print(f"2D indices: {indices_2d}")
print(f"2D remapped: {remapped_2d}")
print(f"Expected: [[3, 0, 2, 0, 1, 0]]")
assert remapped_2d.tolist() == [[3, 0, 2, 0, 1, 0]], f"FAIL: got {remapped_2d.tolist()}"

# Test with 1D indices [6]
indices_1d = mx.array([87, 0, 31, 0, 22, 0], dtype=mx.uint32)
remapped_1d = remap[indices_1d]
print(f"1D indices: {indices_1d}")
print(f"1D remapped: {remapped_1d}")
print(f"Expected: [3, 0, 2, 0, 1, 0]")
assert remapped_1d.tolist() == [3, 0, 2, 0, 1, 0], f"FAIL: got {remapped_1d.tolist()}"

print("\nAll tests passed!")
