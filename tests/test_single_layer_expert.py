"""
Single-layer expert forward comparison test.

Loads layer 0's expert weights from safetensors, constructs a SwitchGLU,
runs a forward pass with known input, and prints the output for comparison
with mlx-zig's stream mode.

Usage: PYTHONPATH=../mlx-lm python3 tests/test_single_layer_expert.py
"""
import sys
sys.path.insert(0, '../mlx-lm')

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import json
import struct
from pathlib import Path

MODEL_DIR = Path.home() / "models" / "DeepSeek-V4-Flash-4bit"

def load_tensor_from_safetensors(tensor_name: str) -> mx.array:
    """Load a single tensor by name from the model's safetensors files."""
    index_path = MODEL_DIR / "model.safetensors.index.json"
    with open(index_path) as f:
        idx = json.load(f)
    
    shard_file = idx["weight_map"].get(tensor_name)
    if shard_file is None:
        raise KeyError(f"Tensor {tensor_name} not found in index")
    
    shard_path = MODEL_DIR / shard_file
    # Use mx.load which handles safetensors natively
    weights = mx.load(str(shard_path))
    return weights[tensor_name]

def main():
    print("Loading layer 0 expert weights...")
    
    # Load fused expert weights for layer 0
    gate_w = load_tensor_from_safetensors("model.layers.0.ffn.switch_mlp.gate_proj.weight")
    gate_s = load_tensor_from_safetensors("model.layers.0.ffn.switch_mlp.gate_proj.scales")
    up_w = load_tensor_from_safetensors("model.layers.0.ffn.switch_mlp.up_proj.weight")
    up_s = load_tensor_from_safetensors("model.layers.0.ffn.switch_mlp.up_proj.scales")
    down_w = load_tensor_from_safetensors("model.layers.0.ffn.switch_mlp.down_proj.weight")
    down_s = load_tensor_from_safetensors("model.layers.0.ffn.switch_mlp.down_proj.scales")
    
    print(f"gate_w: shape={gate_w.shape}, dtype={gate_w.dtype}")
    print(f"gate_s: shape={gate_s.shape}, dtype={gate_s.dtype}")
    print(f"up_w: shape={up_w.shape}, dtype={up_w.dtype}")
    print(f"down_w: shape={down_w.shape}, dtype={down_w.dtype}")
    
    # Read quantization config
    with open(MODEL_DIR / "config.json") as f:
        config = json.load(f)
    qc = config.get("quantization_config", {})
    expert_qc = qc.get("model.layers.0.ffn.switch_mlp.gate_proj", {})
    group_size = expert_qc.get("group_size", 32)
    bits = expert_qc.get("bits", 4)
    mode = expert_qc.get("mode", "mxfp4")
    print(f"Quantization: group_size={group_size}, bits={bits}, mode={mode}")
    
    # Create a simple test input: random [1, 4096] tensor
    mx.random.seed(42)
    x = mx.random.normal((1, 4096)) * 0.1  # Small values like real hidden states
    
    # Create test indices: select experts [0, 1, 2, 3, 4, 5] for 1 token with topk=6
    indices = mx.array([[0, 1, 2, 3, 4, 5]], dtype=mx.uint32)
    
    print(f"\nInput x: shape={x.shape}, mean={x.mean().item():.4f}")
    print(f"Indices: {indices.tolist()}")
    
    # Method 1: Use mlx-lm's SwitchGLU (the reference implementation)
    from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear
    
    # Build a SwitchGLU with the loaded weights
    n_experts = gate_w.shape[0]
    hidden_size = 4096  # DeepSeek V4 hidden_size
    intermediate_size = gate_w.shape[1]  # moe_intermediate_size
    
    print(f"\nn_experts={n_experts}, hidden_size={hidden_size}, intermediate_size={intermediate_size}")
    
    # Create QuantizedSwitchLinear instances directly
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU
    
    gate_proj = QuantizedSwitchLinear(hidden_size, intermediate_size, n_experts, bias=False, group_size=group_size, bits=bits, mode=mode)
    gate_proj.weight = gate_w
    gate_proj.scales = gate_s
    gate_proj.biases = None
    
    up_proj = QuantizedSwitchLinear(hidden_size, intermediate_size, n_experts, bias=False, group_size=group_size, bits=bits, mode=mode)
    up_proj.weight = up_w
    up_proj.scales = up_s
    up_proj.biases = None
    
    down_proj = QuantizedSwitchLinear(intermediate_size, hidden_size, n_experts, bias=False, group_size=group_size, bits=bits, mode=mode)
    down_proj.weight = down_w
    down_proj.scales = down_s
    down_proj.biases = None
    
    # Build SwitchGLU with quantized projections
    switch_glu = SwitchGLU(hidden_size, intermediate_size, n_experts, bias=False)
    switch_glu.gate_proj = gate_proj
    switch_glu.up_proj = up_proj
    switch_glu.down_proj = down_proj
    
    # Check if LimitedSwiGLU is needed
    swiglu_limit = config.get("swiglu_limit", 0)
    print(f"swiglu_limit={swiglu_limit}")
    if swiglu_limit > 0:
        from mlx_lm.models.deepseek_v4 import LimitedSwiGLU
        switch_glu.activation = LimitedSwiGLU(swiglu_limit)
    
    # Run forward
    print("\nRunning SwitchGLU forward...")
    y = switch_glu(x, indices)
    mx.eval(y)
    
    print(f"SwitchGLU output: shape={y.shape}")
    print(f"  mean={y.mean().item():.6f}")
    print(f"  std={y.std().item():.6f}")
    print(f"  max={y.max().item():.6f}")
    print(f"  min={y.min().item():.6f}")
    print(f"  first 5 values: {y[0, 0, :5].tolist()}")
    print(f"  last 5 values: {y[0, 0, -5:].tolist()}")
    
    # Method 2: Manual gather_qmm (matching what Zig does)
    print("\nRunning manual gather_qmm...")
    x_exp = mx.expand_dims(mx.expand_dims(x, -2), -3)  # [1, 1, 1, 4096]
    
    x_gate = mx.gather_qmm(
        x_exp, gate_w, gate_s, None,
        rhs_indices=indices,
        transpose=True,
        group_size=group_size, bits=bits, mode=mode,
        sorted_indices=False,
    )
    x_up = mx.gather_qmm(
        x_exp, up_w, up_s, None,
        rhs_indices=indices,
        transpose=True,
        group_size=group_size, bits=bits, mode=mode,
        sorted_indices=False,
    )
    
    # SwiGLU with limit
    if swiglu_limit > 0:
        gate_clipped = mx.minimum(x_gate, swiglu_limit)
        gate_clipped = mx.maximum(gate_clipped, -swiglu_limit)
        up_clipped = mx.minimum(x_up, swiglu_limit)
        up_clipped = mx.maximum(up_clipped, -swiglu_limit)
        activated = nn.silu(gate_clipped) * up_clipped
    else:
        activated = nn.silu(x_gate) * x_up
    
    x_down = mx.gather_qmm(
        activated, down_w, down_s, None,
        rhs_indices=indices,
        transpose=True,
        group_size=group_size, bits=bits, mode=mode,
        sorted_indices=False,
    )
    
    y_manual = x_down.squeeze(-2)  # [1, 6, 4096]
    mx.eval(y_manual)
    
    print(f"Manual output: shape={y_manual.shape}")
    print(f"  mean={y_manual.mean().item():.6f}")
    print(f"  std={y_manual.std().item():.6f}")
    print(f"  max={y_manual.max().item():.6f}")
    print(f"  min={y_manual.min().item():.6f}")
    print(f"  first 5 values: {y_manual[0, 0, :5].tolist()}")
    
    # Compare
    diff = mx.abs(y - y_manual).max().item()
    print(f"\nMax absolute difference: {diff:.8f}")
    if diff < 1e-4:
        print("✅ SwitchGLU and manual gather_qmm match!")
    else:
        print("❌ MISMATCH between SwitchGLU and manual gather_qmm")

if __name__ == "__main__":
    main()
