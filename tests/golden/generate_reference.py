#!/usr/bin/env python3
"""Generate reference outputs for golden tests."""
import mlx.core as mx
import json
import struct

def save_array(arr, path):
    """Save MLX array as binary file with shape/dtype metadata."""
    arr = mx.array(arr)  # Ensure it's an mlx array
    np_arr = arr.tolist() if arr.size < 1000 else None  # Small arrays as JSON
    
    # Save raw bytes
    flat = arr.reshape(-1)
    if arr.dtype == mx.float32:
        fmt = 'f'
        dtype_str = 'float32'
    elif arr.dtype == mx.float16:
        fmt = 'e'
        dtype_str = 'float16'
    elif arr.dtype == mx.bfloat16:
        # bfloat16 not directly supported in struct, convert to float32
        flat = flat.astype(mx.float32)
        fmt = 'f'
        dtype_str = 'bfloat16'
    elif arr.dtype == mx.int32:
        fmt = 'i'
        dtype_str = 'int32'
    else:
        raise ValueError(f"Unsupported dtype: {arr.dtype}")
    
    data_bytes = struct.pack(f'<{flat.size}{fmt}', *flat.tolist())
    
    with open(path, 'wb') as f:
        f.write(data_bytes)
    
    # Save metadata
    meta = {
        'shape': list(arr.shape),
        'dtype': dtype_str,
    }
    with open(path + '.json', 'w') as f:
        json.dump(meta, f)

def generate_rope_reference():
    """Generate reference RoPE output."""
    mx.random.seed(42)
    batch, heads, seq_len, dims = 1, 2, 4, 8
    x = mx.random.normal((batch, heads, seq_len, dims))
    
    out = mx.fast.rope(x, dims=dims, traditional=False, base=10000.0, scale=1.0, offset=0)
    
    save_array(x, 'tests/golden/rope_input.bin')
    save_array(out, 'tests/golden/rope_output.bin')
    print("Generated rope reference data")

def generate_rmsnorm_reference():
    """Generate reference RMSNorm output."""
    mx.random.seed(42)
    dims = 8
    x = mx.random.normal((1, 2, 4, dims))
    weight = mx.ones((dims,))
    
    # Use mlx.nn.RMSNorm
    import mlx.nn as nn
    rms = nn.RMSNorm(dims)
    out = rms(x)
    
    save_array(x, 'tests/golden/rmsnorm_input.bin')
    save_array(out, 'tests/golden/rmsnorm_output.bin')
    print("Generated rmsnorm reference data")

if __name__ == "__main__":
    generate_rope_reference()
    generate_rmsnorm_reference()
