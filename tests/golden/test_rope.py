#!/usr/bin/env python3
"""Test MLX fast.rope and understand its API."""
import mlx.core as mx

def test_fast_rope_basic():
    """Test basic fast.rope behavior."""
    batch, heads, seq_len, dims = 1, 2, 4, 8
    x = mx.random.normal((batch, heads, seq_len, dims))
    
    # Standard rope
    out = mx.fast.rope(x, dims=dims, traditional=False, base=10000.0, scale=1.0, offset=0)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    print(f"Output sample: {out[0, 0, 0, :4].tolist()}")
    
    # With offset
    out_offset = mx.fast.rope(x, dims=dims, traditional=False, base=10000.0, scale=1.0, offset=10)
    print(f"With offset 10: {out_offset[0, 0, 0, :4].tolist()}")
    
    return out

def test_fast_rope_freqs():
    """Test fast.rope with custom freqs."""
    batch, heads, seq_len, dims = 1, 2, 4, 8
    x = mx.random.normal((batch, heads, seq_len, dims))
    
    max_seq_len = 128
    
    # Try [max_seq_len, dims//2]
    freqs = mx.random.normal((max_seq_len, dims // 2))
    try:
        out = mx.fast.rope(x, dims=dims, traditional=False, base=None, scale=1.0, offset=0, freqs=freqs)
        print(f"freqs shape {freqs.shape} works: {out.shape}")
    except Exception as e:
        print(f"freqs shape {freqs.shape} failed: {e}")
    
    # Try [max_seq_len, dims]
    freqs2 = mx.random.normal((max_seq_len, dims))
    try:
        out2 = mx.fast.rope(x, dims=dims, traditional=False, base=None, scale=1.0, offset=0, freqs=freqs2)
        print(f"freqs shape {freqs2.shape} works: {out2.shape}")
    except Exception as e:
        print(f"freqs shape {freqs2.shape} failed: {e}")
    
    # Try [max_seq_len, dims//2, 2]
    freqs3 = mx.random.normal((max_seq_len, dims // 2, 2))
    try:
        out3 = mx.fast.rope(x, dims=dims, traditional=False, base=None, scale=1.0, offset=0, freqs=freqs3)
        print(f"freqs shape {freqs3.shape} works: {out3.shape}")
    except Exception as e:
        print(f"freqs shape {freqs3.shape} failed: {e}")

def test_yarn_rope():
    """Compare fast.rope with YARN scaling vs manual implementation."""
    batch, heads, seq_len, dims = 1, 2, 4, 8
    x = mx.random.normal((batch, heads, seq_len, dims))
    
    out1 = mx.fast.rope(x, dims=dims, traditional=False, base=10000.0, scale=1.0, offset=0)
    out2 = mx.fast.rope(x, dims=dims, traditional=False, base=10000.0, scale=2.0, offset=0)
    
    print(f"Scale 1.0: {out1[0, 0, 0, :4].tolist()}")
    print(f"Scale 2.0: {out2[0, 0, 0, :4].tolist()}")

if __name__ == "__main__":
    print("=== Test basic fast.rope ===")
    test_fast_rope_basic()
    print("\n=== Test fast.rope with freqs ===")
    test_fast_rope_freqs()
    print("\n=== Test YARN-like scaling ===")
    test_yarn_rope()
