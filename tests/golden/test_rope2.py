#!/usr/bin/env python3
"""Further test MLX fast.rope freqs parameter."""
import mlx.core as mx

def test_freqs_1d():
    """Test 1D freqs."""
    batch, heads, seq_len, dims = 1, 2, 4, 8
    x = mx.random.normal((batch, heads, seq_len, dims))
    
    # Standard rope without freqs
    out1 = mx.fast.rope(x, dims=dims, traditional=False, base=10000.0, scale=1.0, offset=0)
    
    # With 1D freqs
    freqs = mx.array([0.1, 0.2, 0.3, 0.4])
    out2 = mx.fast.rope(x, dims=dims, traditional=False, base=None, scale=1.0, offset=0, freqs=freqs)
    
    print(f"Without freqs: {out1[0, 0, 0, :4].tolist()}")
    print(f"With freqs:    {out2[0, 0, 0, :4].tolist()}")
    print(f"Same? {mx.allclose(out1, out2).item()}")

def test_freqs_vs_base():
    """Test if freqs overrides base."""
    batch, heads, seq_len, dims = 1, 1, 2, 4
    x = mx.ones((batch, heads, seq_len, dims))
    
    # With base=10000
    out1 = mx.fast.rope(x, dims=dims, traditional=False, base=10000.0, scale=1.0, offset=0)
    
    # With custom freqs - try to match base=10000 behavior
    # theta_i = base^(-2i/dims) = 10000^(-2i/4) = 10000^(-i/2)
    freqs = mx.array([1.0 / 10000.0**0.5, 1.0 / 10000.0**1.0])  # [100^(-1), 10000^(-1)]
    out2 = mx.fast.rope(x, dims=dims, traditional=False, base=None, scale=1.0, offset=0, freqs=freqs)
    
    print(f"base=10000: {out1[0, 0, 0, :].tolist()}")
    print(f"custom freqs: {out2[0, 0, 0, :].tolist()}")

def test_scale_effect():
    """Test if scale actually changes anything."""
    batch, heads, seq_len, dims = 1, 1, 8, 8
    x = mx.ones((batch, heads, seq_len, dims))
    
    out1 = mx.fast.rope(x, dims=dims, traditional=False, base=10000.0, scale=1.0, offset=0)
    out2 = mx.fast.rope(x, dims=dims, traditional=False, base=10000.0, scale=4.0, offset=0)
    
    # Check if any difference
    diff = mx.abs(out1 - out2).max()
    print(f"Max diff between scale=1 and scale=4: {diff.item()}")
    print(f"Scale 1, pos 0: {out1[0, 0, 0, :].tolist()}")
    print(f"Scale 4, pos 0: {out2[0, 0, 0, :].tolist()}")
    print(f"Scale 1, pos 7: {out1[0, 0, 7, :].tolist()}")
    print(f"Scale 4, pos 7: {out2[0, 0, 7, :].tolist()}")

if __name__ == "__main__":
    print("=== Test 1D freqs ===")
    test_freqs_1d()
    print("\n=== Test freqs vs base ===")
    test_freqs_vs_base()
    print("\n=== Test scale effect ===")
    test_scale_effect()
